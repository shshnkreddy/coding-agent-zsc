"""Knowledge-driven connectivity policy for two-player 9-card Yokai.

Strategy (uses only locally observable public/private state, no private
convention with the partner):
  - Track own observed card colors privately in memory across own turns
    (colors are only visible in the observation during the observing
    player's own active turn and disappear at turn end).
  - Infer a card's color by elimination when the remaining unknown cards
    must all share the one color whose count is not yet exhausted.
  - Each own turn: observe unknown cards (substeps 0/1), then make the
    legal move that most improves same-known-color adjacency and pushes
    fully-known colors toward full connectivity (substep 2).
  - At substep 3, always reveal an unrevealed hint while any remain
    (unrevealed=5 > revealed=2 > correct-placed=1 > incorrect-placed=-1
    in the terminal score 5U+2R+C-I, so revealing is preferred to
    placing whenever a choice exists). Only place once forced (no
    unrevealed hints remain), choosing the (hint, cell) pair most
    likely correct given known/inferred colors.
  - End the game immediately (substep 0) once own knowledge proves every
    color is fully connected, to avoid unnecessary further hint reveals
    or placements once the terminal condition is already met.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'connectivity_knowledge_agent',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 3,
    'deck_size': 9,
    'grid_size': 9,
    'num_hints': 4,
    'num_nodes': 13,
    'node_features': 13,
    'num_actions': 1068,
    'policy_structure': 'shared_local',
}

NEG = -1e9
NUM_CARDS = 9
NUM_COLORS = 3
NUM_HINTS = 4


def init_agent():
    return {
        "known_color": jnp.full((NUM_CARDS,), -1, dtype=jnp.int32),
        "am_player0": jnp.asarray(0, dtype=jnp.int32),
    }


def _orthogonal_adjacent(positions):
    dx = jnp.abs(positions[:, None, 0] - positions[None, :, 0])
    dy = jnp.abs(positions[:, None, 1] - positions[None, :, 1])
    dist = dx + dy
    eye = jnp.eye(NUM_CARDS, dtype=bool)
    return (dist == 1) & (~eye)


def _transitive_closure(adj_bool):
    reach = adj_bool | jnp.eye(NUM_CARDS, dtype=bool)
    reach_i = reach.astype(jnp.int32)
    for _ in range(4):
        reach_i = jnp.minimum(reach_i + reach_i @ reach_i, 1)
    return reach_i > 0


def _effective_color(known_color):
    counts = jnp.stack([jnp.sum(known_color == c) for c in range(NUM_COLORS)])
    remaining = NUM_COLORS - counts
    full_mask = remaining == 0
    num_full = jnp.sum(full_mask)
    open_color = jnp.argmax(remaining).astype(jnp.int32)
    can_infer = num_full == (NUM_COLORS - 1)
    unknown_mask = known_color == -1
    return jnp.where(unknown_mask & can_infer, open_color, known_color)


def _connectivity_score(effective_color, positions):
    adj = _orthogonal_adjacent(positions)
    known_mask = effective_color != -1
    same_color = effective_color[:, None] == effective_color[None, :]
    color_adj = adj & same_color & known_mask[:, None] & known_mask[None, :]
    direct_pairs = jnp.sum(color_adj).astype(jnp.float32) * 0.5
    # Smooth potential so a move that merely shortens the gap to a same
    # known-color partner (without yet reaching full adjacency, e.g. when
    # the ideal target cell is blocked) still scores strictly better than
    # a neutral move, giving the single-step greedy search a gradient
    # across multiple turns instead of only rewarding exact adjacency.
    same_known_pair = same_color & known_mask[:, None] & known_mask[None, :]
    eye = jnp.eye(NUM_CARDS, dtype=bool)
    same_known_pair = same_known_pair & (~eye)
    manhattan = (
        jnp.abs(positions[:, None, 0] - positions[None, :, 0])
        + jnp.abs(positions[:, None, 1] - positions[None, :, 1])
    ).astype(jnp.float32)
    proximity = jnp.where(same_known_pair, 1.0 / (1.0 + manhattan), 0.0)
    proximity_potential = jnp.sum(proximity) * 0.5 * 0.2
    reach = _transitive_closure(color_adj)
    bonus = jnp.float32(0.0)
    all_connected = jnp.bool_(True)
    connected_per_color = []
    for c in range(NUM_COLORS):
        members = effective_color == c
        count_c = jnp.sum(members)
        fully_known = count_c == 3
        valid = (~(members[:, None] & members[None, :])) | reach
        connected_c = fully_known & jnp.all(valid)
        bonus = bonus + connected_c.astype(jnp.float32) * 100.0
        all_connected = all_connected & connected_c
        connected_per_color.append(connected_c)
    connected_per_color = jnp.stack(connected_per_color)
    return direct_pairs + bonus + proximity_potential, all_connected, connected_per_color


def _same_color_prob(effective_color):
    """Closed-form (9,9) pairwise P(card i and card j share a color), using
    only this seat's own knowledge. Both known: deterministic 0/1. One
    known: P = remaining count of that color / number of still-unknown
    cards. Both unknown: exchangeable two-draw-without-replacement
    probability over the remaining per-color counts (same value for every
    unknown pair, since unknown cards are i.i.d. draws from this seat's
    perspective)."""
    known_mask = effective_color != -1
    counts = jnp.stack([jnp.sum(effective_color == c) for c in range(NUM_COLORS)])
    remaining = (NUM_COLORS - counts).astype(jnp.float32)
    num_unknown = jnp.sum(remaining)

    known_same = (
        (effective_color[:, None] == effective_color[None, :])
        & known_mask[:, None] & known_mask[None, :]
    ).astype(jnp.float32)

    safe_color = jnp.where(known_mask, effective_color, 0)
    p_vec = remaining[safe_color] / jnp.maximum(num_unknown, 1.0)  # (9,)

    both_unknown_p = jnp.sum(remaining * (remaining - 1.0)) / jnp.maximum(
        num_unknown * (num_unknown - 1.0), 1.0
    )

    mixed = jnp.where(
        known_mask[:, None] & (~known_mask[None, :]),
        jnp.broadcast_to(p_vec[:, None], (NUM_CARDS, NUM_CARDS)),
        0.0,
    ) + jnp.where(
        (~known_mask[:, None]) & known_mask[None, :],
        jnp.broadcast_to(p_vec[None, :], (NUM_CARDS, NUM_CARDS)),
        0.0,
    )

    both_known_mask = known_mask[:, None] & known_mask[None, :]
    either_known_mask = known_mask[:, None] | known_mask[None, :]
    same_prob = jnp.where(
        both_known_mask,
        known_same,
        jnp.where(either_known_mask, mixed, both_unknown_p),
    )
    # Only informative for pairs not already fully known (those are handled
    # exactly by the deterministic direct_pairs/bonus terms elsewhere).
    return jnp.where(both_known_mask, 0.0, same_prob)


def agent_act(obs, reset, memory, rng):
    del rng
    known_color = memory["known_color"]
    known_color = jnp.where(reset, jnp.full((NUM_CARDS,), -1, dtype=jnp.int32), known_color)

    active_now = (obs.node_features[0, 10] > 0).astype(jnp.int32)
    am_player0 = jnp.where(reset, active_now, memory["am_player0"])
    # Seats run identical code; searching in different orders by seat
    # avoids both seats redundantly learning the same cards at the same
    # time and spreads early knowledge across the whole card range
    # (not just the two endpoints), so more cards are known early by at
    # least one seat instead of only becoming known on a seat's last turn.
    p0_pref = jnp.array([8, 3, 7, 2, 6, 1, 5, 0, 4], dtype=jnp.float32)  # evens first
    p1_pref = jnp.array([4, 8, 3, 7, 2, 6, 1, 5, 0], dtype=jnp.float32)  # odds first
    observe_tie_break = jnp.where(am_player0 > 0, p0_pref, p1_pref) * 0.01

    card_color_onehot = obs.node_features[0:NUM_CARDS, 0:3].astype(jnp.int32)
    revealed_now = jnp.sum(card_color_onehot, axis=1) > 0
    decoded = jnp.argmax(card_color_onehot, axis=1).astype(jnp.int32)
    known_color = jnp.where(revealed_now, decoded, known_color)

    effective_color = _effective_color(known_color)
    positions = obs.positions.astype(jnp.int32)

    _, all_connected, connected_per_color = _connectivity_score(effective_color, positions)

    substep = obs.node_features[0, 11].astype(jnp.int32)
    legal = obs.legal_actions

    locked = obs.node_features[0:NUM_CARDS, 8] > 0
    unknown = known_color == -1

    def substep0_scores(_):
        noop = jnp.zeros((1,), dtype=jnp.float32)
        observe_scores = (
            jnp.where(unknown, 10.0, 0.0) + observe_tie_break - jnp.where(locked, 1e6, 0.0)
        ).astype(jnp.float32)
        move = jnp.full((729,), NEG)
        reveal = jnp.full((NUM_HINTS,), NEG)
        place = jnp.full((NUM_HINTS * 81,), NEG)
        end = jnp.where(all_connected, jnp.float32(1e6), NEG).reshape((1,))
        return jnp.concatenate([noop, observe_scores, move, reveal, place, end])

    def substep1_scores(_):
        noop = jnp.zeros((1,), dtype=jnp.float32)
        observe_scores = (
            jnp.where(unknown, 10.0, 0.0)
            + observe_tie_break
            - jnp.where(locked, 1e6, 0.0)
            - jnp.where(revealed_now, 1e6, 0.0)
        ).astype(jnp.float32)
        move = jnp.full((729,), NEG)
        reveal = jnp.full((NUM_HINTS,), NEG)
        place = jnp.full((NUM_HINTS * 81,), NEG)
        end = jnp.full((1,), NEG)
        return jnp.concatenate([noop, observe_scores, move, reveal, place, end])

    def substep2_scores(_):
        noop = jnp.zeros((1,), dtype=jnp.float32)
        observe_scores = jnp.full((NUM_CARDS,), NEG)

        idx = jnp.arange(729)
        i_idx = idx // 81
        xy = idx % 81
        x_idx = xy // 9
        y_idx = xy % 9

        same_color_prob = _same_color_prob(effective_color)

        def cand_score(i, x, y):
            new_positions = positions.at[i].set(jnp.array([x, y], dtype=jnp.int32))
            score, _, _ = _connectivity_score(effective_color, new_positions)
            # Small tie-break: keep cards clustered (more orthogonal
            # neighbors) so a currently-unknown card that later turns out
            # to match a nearby color is already well placed. Weighted
            # far below any real connectivity gain (max +100/+1 per pair).
            new_adj = _orthogonal_adjacent(new_positions)
            density = jnp.sum(new_adj[i]).astype(jnp.float32)
            # Expected same-color adjacency for pairs not already fully
            # known (e.g. a still-unknown card next to a known-color card
            # of likely-matching color), weighted well below the
            # deterministic direct_pairs/bonus terms so confirmed
            # knowledge always dominates this probabilistic nudge.
            prob_bonus = jnp.sum(new_adj.astype(jnp.float32) * same_color_prob) * 0.5 * 0.12
            return score + density * 0.05 + prob_bonus

        move_scores = jax.vmap(cand_score)(i_idx, x_idx, y_idx).astype(jnp.float32)
        reveal = jnp.full((NUM_HINTS,), NEG)
        place = jnp.full((NUM_HINTS * 81,), NEG)
        end = jnp.full((1,), NEG)
        return jnp.concatenate([noop, observe_scores, move_scores, reveal, place, end])

    def substep3_scores(_):
        noop = jnp.zeros((1,), dtype=jnp.float32)
        observe_scores = jnp.full((NUM_CARDS,), NEG)
        move = jnp.full((729,), NEG)

        hint_revealed = jnp.sum(obs.node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 3:6], axis=1) > 0
        hint_placed = obs.node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 8] > 0
        hint_set = obs.node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 3:6] > 0

        reveal_scores = jnp.where(~hint_revealed, 1000.0, NEG).astype(jnp.float32)

        grid_x, grid_y = jnp.meshgrid(jnp.arange(9), jnp.arange(9), indexing="ij")
        pos_x = positions[:, 0]
        pos_y = positions[:, 1]
        match = (pos_x[:, None, None] == grid_x[None]) & (pos_y[:, None, None] == grid_y[None])
        occupied = jnp.any(match, axis=0)
        color_plus1 = jnp.sum(jnp.where(match, (effective_color[:, None, None] + 1), 0), axis=0)
        color_grid = color_plus1 - 1

        hint_set_f = hint_set.astype(jnp.float32)
        color_onehot_grid = jax.nn.one_hot(jnp.clip(color_grid, 0, 2), 3)
        match_score_grid = jnp.einsum("hc,xyc->hxy", hint_set_f, color_onehot_grid)
        known_here = (color_grid != -1) & occupied
        base_place = jnp.where(known_here[None, :, :], jnp.where(match_score_grid > 0, 500.0, -500.0), 0.0)
        # Locking a card whose color group is not yet fully connected can
        # strand a card that is still needed to move; prefer to lock cards
        # whose group is already connected when a same-correctness
        # alternative exists (weighted well below the correct/incorrect
        # swing so hint correctness always dominates this tie-break).
        group_connected_grid = jnp.einsum(
            "c,xyc->xy", connected_per_color.astype(jnp.float32), color_onehot_grid
        )
        lock_risk_penalty = jnp.where(known_here & (group_connected_grid < 0.5), 50.0, 0.0)
        base_place = base_place - lock_risk_penalty[None, :, :]
        eligible_h = hint_revealed & (~hint_placed)
        place_scores_grid = jnp.where(eligible_h[:, None, None], base_place, NEG)
        place_scores = place_scores_grid.reshape(NUM_HINTS * 81).astype(jnp.float32)

        end = jnp.full((1,), NEG)
        return jnp.concatenate([noop, observe_scores, move, reveal_scores, place_scores, end])

    branches = [substep0_scores, substep1_scores, substep2_scores, substep3_scores]
    final_scores = jax.lax.switch(substep, branches, 0)

    masked_scores = jnp.where(legal, final_scores, NEG)
    action = jnp.argmax(masked_scores).astype(jnp.int32)

    next_memory = {"known_color": known_color, "am_player0": am_player0}
    trace = {
        "all_connected": all_connected.astype(jnp.int32).reshape(()),
        "known_color": known_color,
    }
    return action, next_memory, trace
