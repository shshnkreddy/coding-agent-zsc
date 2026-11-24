"""Greedy, convention-free clustering policy for the 2-player 16-card Yokai task.

Design notes (see experiment ledger A01):
  - Memory is trivial (a scalar); the environment already exposes cumulative
    personal color knowledge to an active seat, so no private state is needed.
  - Observation targets prioritize cards unknown to both seats (using the
    public partner-seen indicator) over cards only the partner has already
    inspected, spreading the tight observation budget across the deck.
  - Colors are clustered toward fixed 2x2 zones that quarter the publicly
    known reset block (x,y in 3..6), derived from the public color IDs
    (0..3) and the deterministic reset geometry, not a private per-episode
    convention, so the scheme is meant to be legible to independently
    generated partners reasoning from the same spec. Any cell within a
    color's zone counts as equally good (distance 0), which avoids two
    seats contending over one exact cell while keeping the zones close to
    the reset block to fit the tight 14-move budget.
  - A known card already connected (through same-color-only adjacency) to
    every other same-color card this seat knows is "settled" and is not
    targeted again, avoiding wasted back-and-forth moves between seats.
  - Among destinations offering equal zone improvement, a secondary bonus
    favors cells adjacent to more already-known same-color companions
    (packing into one mutually adjacent cluster instead of leaving cards
    diagonal within a partially filled zone). When no known card needs a
    move, a mandatory move of an unknown-color card mildly prefers leaving
    the reset block so it stops occupying a cell some color's zone may need.
  - Hints are always revealed before any are placed (delays locking, keeps
    observation opportunities open); placement prefers guaranteed-correct
    matches (smallest matching set first) and otherwise gambles the largest
    remaining set on an unknown card.
  - The game is only ended early once this seat has personally verified all
    16 colors and confirmed every color forms one connected group.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 4,
    'deck_size': 16,
    'grid_size': 10,
    'num_hints': 7,
    'num_nodes': 23,
    'node_features': 15,
    'num_actions': 2325,
    'policy_structure': 'shared_local',
}

NUM_CARDS = 16
NUM_COLORS = 4
NUM_HINTS = 7
GRID = 10
NEG_INF = -1e9

# A 2x2 target zone per color, splitting the deterministic reset block
# (x,y in 3..6) into quadrants, derived purely from public color IDs and the
# publicly known reset geometry, not from a private per-episode convention.
# Any cell inside a color's zone counts as equally good (distance 0), which
# avoids two seats contending over a single exact anchor cell while still
# giving both seats the same convergence target.
ZONE_X_LO = (3.0, 3.0, 5.0, 5.0)
ZONE_X_HI = (4.0, 4.0, 6.0, 6.0)
ZONE_Y_LO = (3.0, 5.0, 3.0, 5.0)
ZONE_Y_HI = (4.0, 6.0, 4.0, 6.0)


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _connectivity_closure(mask_adj):
    # mask_adj: (16,16) bool, includes self-loops. Boolean transitive closure
    # via repeated squaring; 4 squarings cover paths up to length 16.
    r = mask_adj
    for _ in range(4):
        step = (r.astype(jnp.int32) @ r.astype(jnp.int32)) > 0
        r = r | step
    return r


def agent_act(obs, reset, memory, rng):
    node_features = obs.node_features
    adjacency = obs.adjacency_matrix
    positions = obs.positions
    legal_actions = obs.legal_actions

    card_features = node_features[:NUM_CARDS]
    hint_features = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS]

    color_onehot = card_features[:, 0:4]
    known = jnp.any(color_onehot != 0, axis=-1)
    color_id = jnp.argmax(color_onehot, axis=-1).astype(jnp.int32)
    locked = card_features[:, 10] > 0
    partner_seen = card_features[:, 11] > 0

    pos_x = positions[:, 0].astype(jnp.int32)
    pos_y = positions[:, 1].astype(jnp.int32)

    substep = card_features[0, 13].astype(jnp.int32)
    active = card_features[0, 12] > 0

    # ---- solved check (only trustworthy when all 16 colors are personally known) ----
    all_known = jnp.all(known)
    card_adj = adjacency[:NUM_CARDS, :NUM_CARDS] > 0
    same_color = color_id[:, None] == color_id[None, :]
    masked_adj = card_adj & same_color
    reach = _connectivity_closure(masked_adj)
    connected_ok = jnp.all(reach | (~same_color), axis=-1)
    all_connected = jnp.all(connected_ok)
    solved = all_known & all_connected

    # Prioritize cards unknown to both seats (public partner-seen indicator)
    # over cards the partner has already inspected, to spread limited
    # observation budget across the whole deck rather than duplicating the
    # partner's personal knowledge.
    obs_priority = jnp.where(known, 0.0, jnp.where(partner_seen, 1.0, 2.0))

    # ---- substep 0: observe an unknown card, or end if solved ----
    def substep0():
        obs_score = obs_priority
        obs_legal = legal_actions[1:1 + NUM_CARDS]
        obs_masked = jnp.where(obs_legal, obs_score, NEG_INF)
        end_legal = legal_actions[2324]
        end_score = jnp.where(solved & end_legal, 1000.0, NEG_INF)
        combined = jnp.concatenate([obs_masked, end_score[None]])
        best_idx = jnp.argmax(combined)
        return jnp.where(
            best_idx == NUM_CARDS,
            jnp.int32(2324),
            (best_idx + 1).astype(jnp.int32),
        )

    # ---- substep 1: observe a different unknown card ----
    def substep1():
        obs_score = obs_priority
        obs_legal = legal_actions[1:1 + NUM_CARDS]
        obs_masked = jnp.where(obs_legal, obs_score, NEG_INF)
        best_idx = jnp.argmax(obs_masked)
        return (best_idx + 1).astype(jnp.int32)

    # ---- substep 2: move a known, not-yet-settled card toward its color zone ----
    def substep2():
        x_lo = jnp.asarray(ZONE_X_LO, dtype=jnp.float32)[color_id]
        x_hi = jnp.asarray(ZONE_X_HI, dtype=jnp.float32)[color_id]
        y_lo = jnp.asarray(ZONE_Y_LO, dtype=jnp.float32)[color_id]
        y_hi = jnp.asarray(ZONE_Y_HI, dtype=jnp.float32)[color_id]
        pos_xf = pos_x.astype(jnp.float32)
        pos_yf = pos_y.astype(jnp.float32)

        def zone_dist(x, y, xlo, xhi, ylo, yhi):
            dx = jnp.maximum(0.0, jnp.maximum(xlo - x, x - xhi))
            dy = jnp.maximum(0.0, jnp.maximum(ylo - y, y - yhi))
            return dx + dy

        known_pair = known[:, None] & known[None, :]
        masked_adj_known = card_adj & same_color & known_pair
        reach_known = _connectivity_closure(masked_adj_known)
        not_self = ~jnp.eye(NUM_CARDS, dtype=bool)
        candidates = known_pair & same_color & not_self
        has_candidates = jnp.any(candidates, axis=1)
        has_unconnected = jnp.any(candidates & (~reach_known), axis=1)

        current_dist = zone_dist(pos_xf, pos_yf, x_lo, x_hi, y_lo, y_hi)
        # A known card is settled once it is connected (through same-color-
        # only adjacency) to every other same-color card this seat knows.
        # A lone known card (no known companion yet) is never marked settled
        # here, since being inside the zone alone does not guarantee the
        # strict orthogonal adjacency the scoring rule requires once other
        # companions are discovered.
        settled = has_candidates & (~has_unconnected)
        tier = jnp.where(known & (~settled), 0.0, -2000.0)
        xs = jnp.arange(GRID, dtype=jnp.float32)
        ys = jnp.arange(GRID, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(xs, ys, indexing='ij')  # (10,10)
        new_dist = zone_dist(
            gx[None, :, :], gy[None, :, :],
            x_lo[:, None, None], x_hi[:, None, None],
            y_lo[:, None, None], y_hi[:, None, None],
        )
        improvement = current_dist[:, None, None] - new_dist

        # Secondary tie-break: among destinations offering the same zone
        # improvement (e.g., any still-empty cell inside the zone counts
        # equally), prefer one orthogonally adjacent to an already-known
        # same-color companion. This favors packing same-color cards into a
        # single mutually adjacent cluster instead of leaving them diagonal
        # (technically both "in zone" but not yet connected) inside a
        # partially filled zone.
        dist_to_cards = (jnp.abs(gx.reshape(-1)[None, :] - pos_xf[:, None]) +
                          jnp.abs(gy.reshape(-1)[None, :] - pos_yf[:, None]))
        adjacent_to_card = dist_to_cards == 1.0  # (16 cards, 100 cells)
        companions_i = candidates.astype(jnp.float32)  # (16,16)
        adjacency_bonus_flat = companions_i @ adjacent_to_card.astype(jnp.float32)
        adjacency_bonus = adjacency_bonus_flat.reshape(NUM_CARDS, GRID, GRID)

        # When forced to move a card of unknown color (no known card needs a
        # move this turn), mildly prefer evicting it out of the reset block
        # if it currently sits inside it, so it stops occupying a cell some
        # color's zone may need, without spending a move that could instead
        # go to a known card.
        in_block_now = (pos_xf >= 3.0) & (pos_xf <= 6.0) & (pos_yf >= 3.0) & (pos_yf <= 6.0)
        dest_in_block = ((gx >= 3.0) & (gx <= 6.0) & (gy >= 3.0) & (gy <= 6.0))[None, :, :]
        evict_bonus = jnp.where(in_block_now[:, None, None] & (~dest_in_block), 0.1, 0.0)

        value = jnp.where(
            known[:, None, None],
            tier[:, None, None] + improvement + 0.3 * adjacency_bonus,
            -0.5 + evict_bonus,
        )
        flat_value = value.reshape(-1)
        move_legal = legal_actions[17:17 + NUM_CARDS * GRID * GRID]
        masked_value = jnp.where(move_legal, flat_value, NEG_INF)
        any_legal_move = jnp.any(move_legal)
        best_idx = jnp.argmax(masked_value)
        move_action = (17 + best_idx).astype(jnp.int32)
        fallback = jnp.argmax(legal_actions).astype(jnp.int32)
        return jnp.where(any_legal_move, move_action, fallback)

    # ---- substep 3: reveal an unrevealed hint, else place the best-matching one ----
    def substep3():
        hint_colorset = hint_features[:, 4:8] != 0
        hint_revealed = jnp.any(hint_colorset, axis=-1)
        hint_unrevealed = ~hint_revealed
        popcount = jnp.sum(hint_colorset, axis=-1).astype(jnp.float32)

        reveal_legal = legal_actions[1617:1617 + NUM_HINTS]
        reveal_value = jnp.where(
            hint_unrevealed & reveal_legal,
            (NUM_HINTS - 1 - jnp.arange(NUM_HINTS)).astype(jnp.float32),
            NEG_INF,
        )
        reveal_possible = jnp.any(hint_unrevealed & reveal_legal)
        reveal_action = (1617 + jnp.argmax(reveal_value)).astype(jnp.int32)

        match = hint_colorset[:, color_id]  # (7, 16) bool
        known_b = known[None, :]
        score = jnp.where(
            known_b & match,
            2.0 + (NUM_COLORS - popcount)[:, None] * 0.1,
            jnp.where(
                known_b & (~match),
                -0.5,
                1.0 + popcount[:, None] * 0.1,
            ),
        )
        action_idx = (1624 + jnp.arange(NUM_HINTS)[:, None] * 100 +
                      pos_x[None, :] * 10 + pos_y[None, :]).astype(jnp.int32)
        place_legal = jnp.take(legal_actions, action_idx.reshape(-1)).reshape(NUM_HINTS, NUM_CARDS)
        place_masked = jnp.where(place_legal & (~locked[None, :]), score, NEG_INF)
        flat_place = place_masked.reshape(-1)
        any_place = jnp.any(place_legal & (~locked[None, :]))
        best_place_idx = jnp.argmax(flat_place)
        place_action = action_idx.reshape(-1)[best_place_idx]
        fallback = jnp.argmax(legal_actions).astype(jnp.int32)

        chosen_place = jnp.where(any_place, place_action, fallback)
        return jnp.where(reveal_possible, reveal_action, chosen_place)

    computed_action = lax.switch(substep, [substep0, substep1, substep2, substep3])

    safe_action = jnp.where(
        legal_actions[computed_action],
        computed_action,
        jnp.argmax(legal_actions).astype(jnp.int32),
    )
    final_action = jnp.where(active, safe_action, jnp.int32(0))

    zero_mem = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero_mem, memory)
    debug_trace = {
        'known_count': jnp.sum(known).astype(jnp.int32),
        'substep': substep,
        'active': active.astype(jnp.int32),
        'action': final_action,
        'known_mask': known,
    }
    return final_action, next_memory, debug_trace
