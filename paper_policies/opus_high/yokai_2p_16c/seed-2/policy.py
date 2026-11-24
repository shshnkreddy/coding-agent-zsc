"""Two-player Yokai policy: convention-free greedy colour clustering.

No private signalling scheme is used.  Every decision comes from publicly
visible board state plus this seat's own colour observations, so a partner
produced by an independent search needs to share no protocol with this code.
The only things assumed about the partner are that it also tries to make each
colour connected and that it prefers correct hint placements.

Design, in the order the four substeps run:

* Substep 0/1 (observe).  Targets are cards this seat has not seen, preferring
  ones the partner has not inspected either.  A seat gets at most 14 of the 16
  colours in a fourteen-turn game, so the pair only learns every colour if the
  two seats cover different cards; self-play confirms that joint coverage
  matters far more than agreeing on the same cards (76% vs 19% solved).
  Counting inference fills in the rest whenever only one colour is still short.

* Substep 0 (stop).  The seat ends the game as soon as it can certify that all
  four colours are connected.  The certificate is exact: colours known to this
  seat must be connected, and the remaining unknown cards are enumerated over
  every completion consistent with four cards per colour.  It never depends on
  the partner, so it stays sound in cross-play.

* Substep 2 (move).  Candidate destinations are the 4-neighbourhoods of the
  current cards, which provably covers every legal move because the board must
  stay connected.  A cheap shortlist is refined by an exact potential built
  from three partner-agnostic terms: relocations still owed per colour (the
  smaller of stranded-card count and MST bridging cost), compactness of each
  colour, and the expected relocations owed by cards this seat has not seen.
  Colour groups therefore assemble wherever their cards already are, which is
  public information, rather than at any privately agreed location.

* Substep 3 (hints).  Placing a hint locks its card forever, so all seven hints
  are revealed before any placement; the horizon is fourteen turns either way,
  so this costs nothing and keeps cards movable. Placements then prefer cards
  whose colour is known to lie in the hint set and that are already settled
  inside a connected group.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'greedy_cluster',
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

N_CARDS = 16
N_COLORS = 4
N_HINTS = 7
PER_COLOR = 4
GRID = 10
TOP_K = 20
N_SLOTS = 3                 # unknown cards enumerated in the end-game test
BIG = 1.0e9
FAR = 1.0e4

# Hint h has |set| = 1,1,2,2,2,3,3 by the documented reset rule.
HINT_SIZE_TUPLE = (1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0)
DELTA_TUPLE = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Potential weights.
W_GAP = 2.0
W_PAIR = 3.5
W_CENTER = 0.02
W_UNKNOWN = 2.5             # weight on expected work owed by unseen cards
W_MOVED = 0.6               # discourage shuffling an already relocated card
W_MINE = 3.0                # prefer cards only this seat can place deliberately
END_VALUE = 6.0             # expected value of playing on to the step cap


def init_agent():
    return {'colors': jnp.full((N_CARDS,), -1, dtype=jnp.int32)}


def _manhattan(pos):
    """Pairwise Manhattan distances, pos [..., 16, 2] -> [..., 16, 16]."""
    return jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]).sum(-1)


def _color_masks(colors, known):
    """[..., 16, 4] boolean membership of each card in each colour class."""
    return (colors[..., :, None] == jnp.arange(N_COLORS)) & known[..., :, None]


def _mst_gaps(dist, masks):
    """Missing grid steps per colour, shape [..., 4].  Zero iff connected.

    ``dist`` is [..., 16, 16] and ``masks`` is [..., 16, 4]; leading dimensions
    of the two arguments broadcast against each other.
    """
    dist_f = dist.astype(jnp.float32)
    idx = jnp.arange(N_CARDS)
    gaps = []
    for c in range(N_COLORS):
        mask = masks[..., c]
        count = mask.sum(-1)
        seed = jnp.argmax(mask, -1)
        in_tree = (idx == seed[..., None]) & mask
        length = jnp.zeros((), dtype=jnp.float32)
        for _ in range(PER_COLOR - 1):
            reach = jnp.where(in_tree[..., :, None], dist_f, FAR).min(-2)
            reach = jnp.where(mask & (~in_tree), reach, FAR)
            step = reach.min(-1)
            pick = jnp.argmin(reach, -1)
            valid = step < FAR / 2.0
            length = length + jnp.where(valid, step, 0.0)
            in_tree = in_tree | ((idx == pick[..., None]) & valid[..., None])
        gaps.append(length - jnp.maximum(count - 1, 0).astype(jnp.float32))
    return jnp.stack(gaps, axis=-1)


def _stranded(dist, masks):
    """Cards of each colour that are outside its largest connected group.

    A relocation can put a card anywhere, so the number of cards separated from
    the biggest group of their colour is a direct estimate of the remaining
    move cost; ``_mst_gaps`` is the alternative estimate that counts bridging
    cards, and the smaller of the two is the better upper bound.
    """
    idx = jnp.arange(N_CARDS)
    same = ((masks[..., :, None, :] & masks[..., None, :, :]).any(-1)
            & (dist == 1))
    adj = same | jnp.eye(N_CARDS, dtype=bool)
    lab = jnp.broadcast_to(idx, adj.shape[:-1])
    for _ in range(PER_COLOR - 1):
        lab = jnp.minimum(lab, jnp.where(adj, lab[..., None, :], N_CARDS).min(-1))
    group = (lab[..., :, None] == lab[..., None, :]).sum(-1)          # [..., 16]
    counts = masks.sum(-2)                                            # [..., 4]
    largest = jnp.max(jnp.where(masks, group[..., None], 0), axis=-2)  # [..., 4]
    return (counts - largest).astype(jnp.float32)


def _work(dist, masks):
    """Estimated relocations still needed per colour, [..., 4].  Zero iff connected."""
    return jnp.minimum(_stranded(dist, masks), _mst_gaps(dist, masks))


def _nearest_color_distance(dist, masks):
    """dist_to[..., i, c] = distance from card i to the closest known card of c."""
    offset = jnp.where(masks, 0.0, FAR)
    return jnp.min(dist[..., :, :, None].astype(jnp.float32) + offset[..., None, :, :],
                   axis=-2)


def _potential(pos, colors, known, known_count, n_unknown):
    """Lower is better.  Batched over any leading candidate axes of ``pos``."""
    dist = _manhattan(pos)
    masks = _color_masks(colors, known)
    gap = _work(dist, masks).sum(-1)

    same = ((colors[:, None] == colors[None, :]) & known[:, None] & known[None, :]
            & (~jnp.eye(N_CARDS, dtype=bool)))
    pair = 0.5 * (jnp.minimum(dist.astype(jnp.float32), 4.0) * same).sum((-1, -2))
    center = jnp.abs(pos.astype(jnp.float32) - 4.5).sum((-1, -2))

    # Expected relocations still owed by cards whose colour this seat has not
    # seen: an unknown card needs no move only if it already touches the group
    # of whichever colour it turns out to be.
    dist_c = _nearest_color_distance(dist, masks)
    share = (PER_COLOR - known_count).astype(jnp.float32) / jnp.maximum(
        n_unknown, 1).astype(jnp.float32)
    away = jnp.clip((dist_c - 1.0) * 0.5, 0.0, 1.0)
    unk = (away * share[None, :] * (~known)[:, None]).sum((-1, -2))

    return W_GAP * (gap + W_UNKNOWN * unk) + W_PAIR * pair + W_CENTER * center


def _deduce(colors):
    """Counting inference: if only one colour still misses cards, fill it in."""
    known = colors >= 0
    known_count = _color_masks(colors, known).sum(-2)
    missing = known_count < PER_COLOR
    forced = jnp.argmax(missing)
    colors = jnp.where((~known) & (missing.sum() == 1), forced, colors)
    known = colors >= 0
    return colors, known, _color_masks(colors, known).sum(-2)


def _combos():
    """All colour assignments for the enumerated unknown slots."""
    grid = jnp.stack(jnp.meshgrid(*([jnp.arange(N_COLORS)] * N_SLOTS), indexing='ij'),
                     axis=-1)
    return grid.reshape(-1, N_SLOTS)


def _solved_probability(adjacent, known, known_count):
    """Probability that every colour is connected, given the seat's knowledge.

    Unknown colours are uniform over the assignments consistent with the four
    cards per colour.  An unknown card counts as connected when it touches a
    card already known to have the colour it is assigned, which is sufficient
    (and only misses chains of several unknown cards of one colour).
    """
    unknown = ~known
    n_unknown = unknown.sum()
    order = jnp.argsort(jnp.where(unknown, 0, N_CARDS) + jnp.arange(N_CARDS))
    slots = order[:N_SLOTS]
    slot_valid = unknown[slots]                               # [3]

    combos = _combos()
    missing = (PER_COLOR - known_count)                       # [4]
    counts = ((combos[:, :, None] == jnp.arange(N_COLORS)) & slot_valid[None, :, None]
              ).sum(1)
    consistent = jnp.all(counts == missing[None, :], axis=-1)

    fits = adjacent[slots]
    ok = jnp.all(fits[jnp.arange(N_SLOTS)[None, :], combos] | (~slot_valid[None, :]),
                 axis=-1)
    total = jnp.maximum(consistent.sum(), 1)
    prob = (consistent & ok).sum().astype(jnp.float32) / total.astype(jnp.float32)
    return jnp.where(n_unknown <= N_SLOTS, prob, 0.0)


def agent_act(obs, reset, memory, rng):
    nf = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    pos = obs.positions.astype(jnp.int32)
    idx = jnp.arange(N_CARDS)

    active = nf[0, 12] > 0
    substep = nf[0, 13]

    # ---- own colour knowledge (perfect memory, restored while active) ----
    own = nf[:N_CARDS, 0:4]
    seen_now = own.sum(-1) > 0
    color_now = jnp.argmax(own, axis=-1)
    prev = jnp.where(reset, jnp.full((N_CARDS,), -1, dtype=jnp.int32), memory['colors'])
    colors, known, known_count = _deduce(jnp.where(seen_now, color_now, prev))
    n_unknown = N_CARDS - known.sum()
    next_memory = {'colors': colors}

    partner_seen = nf[:N_CARDS, 11] > 0
    masks = _color_masks(colors, known)

    hint_sets = nf[N_CARDS:N_CARDS + N_HINTS, 4:8] > 0
    hint_placed = nf[N_CARDS:N_CARDS + N_HINTS, 10] > 0
    hint_pos = nf[N_CARDS:N_CARDS + N_HINTS, 8:10]
    hint_revealed = hint_sets.any(-1)

    dist_now = _manhattan(pos)
    gaps_now = _work(dist_now, masks)
    connected = jnp.all(gaps_now <= 0.0)
    dist_c_now = _nearest_color_distance(dist_now, masks)

    # ---------------- value of stopping right now ----------------
    card_in_set = hint_sets[:, jnp.clip(colors, 0, N_COLORS - 1)]          # [7, 16]
    share = (PER_COLOR - known_count).astype(jnp.float32) / jnp.maximum(
        n_unknown, 1).astype(jnp.float32)
    p_unknown = (hint_sets.astype(jnp.float32) * share[None, :]).sum(-1)   # [7]
    p_correct = jnp.where(known[None, :], card_in_set.astype(jnp.float32),
                          p_unknown[:, None])                              # [7, 16]

    at_hint = jnp.all(pos[None, :, :] == hint_pos[:, None, :], axis=-1)    # [7, 16]
    placed_p = (at_hint * p_correct).sum(-1)
    unrevealed = (~hint_revealed).sum().astype(jnp.float32)
    waiting = (hint_revealed & (~hint_placed)).sum().astype(jnp.float32)
    score_now = (5.0 * unrevealed + 2.0 * waiting
                 + (hint_placed * (2.0 * placed_p - 1.0)).sum())

    p_solved = jnp.where(connected,
                         _solved_probability(dist_c_now <= 1.0, known, known_count), 0.0)
    end_now = (p_solved > 0.999) | (p_solved * score_now >= END_VALUE)

    # ---------------- substep 0 / 1: which card to observe ----------------
    obs_ids = 1 + idx
    obs_legal = legal[obs_ids]
    obs_score = (jnp.where(known, 0.0, 6.0)
                 + jnp.where(partner_seen, 0.0, 2.0)
                 + 0.01 * (N_CARDS - idx).astype(jnp.float32))
    observe_action = obs_ids[jnp.argmax(jnp.where(obs_legal, obs_score, -BIG))]
    action0 = jnp.where(end_now & legal[2324], 2324, observe_action)

    # ---------------- substep 2: pick a move ----------------
    # Board connectivity forces every legal destination to touch some card, so
    # the 4-neighbourhood of the current cards covers all legal moves.
    deltas = jnp.asarray(DELTA_TUPLE, dtype=jnp.int32)
    cells = jnp.clip(pos[:, None, :] + deltas[None, :, :], 0, GRID - 1).reshape(-1, 2)
    n_cells = cells.shape[0]
    move_ids = 17 + idx[:, None] * 100 + cells[None, :, 0] * 10 + cells[None, :, 1]
    move_legal = legal[move_ids]                                  # [16, 64]

    same_pair = ((colors[:, None] == colors[None, :]) & known[:, None] & known[None, :]
                 & (~jnp.eye(N_CARDS, dtype=bool)))
    cell_dist = jnp.abs(cells[:, None, :] - pos[None, :, :]).sum(-1).astype(jnp.float32)
    masked = cell_dist[None, :, :] + jnp.where(same_pair, 0.0, FAR)[:, None, :]
    near_new = masked.min(-1)                                     # [16, 64]
    sum_new = jnp.where(masked < FAR / 2.0, jnp.minimum(masked, 4.0), 0.0).sum(-1)
    cur = dist_now.astype(jnp.float32) + jnp.where(same_pair, 0.0, FAR)
    near_old = cur.min(-1)
    sum_old = jnp.where(cur < FAR / 2.0, jnp.minimum(cur, 4.0), 0.0).sum(-1)

    # Relocating a card whose colour is unknown is a lottery, so the shortlist
    # only ranks cards this seat can place deliberately.
    cost_old = jnp.where(known,
                         W_GAP * jnp.clip(near_old - 1.0, 0.0, 1.0) + W_PAIR * sum_old,
                         0.0)
    cost_new = jnp.where(known[:, None],
                         W_GAP * jnp.clip(near_new - 1.0, 0.0, 1.0) + W_PAIR * sum_new,
                         0.0)

    home = jnp.stack([3 + idx // PER_COLOR, 3 + idx % PER_COLOR], axis=-1)
    # The partner cannot deliberately place a card it has never inspected, so
    # spend this seat's relocations on the cards only it knows.
    displaced = (jnp.any(pos != home, axis=-1).astype(jnp.float32)
                 - (W_MINE / W_MOVED) * (known & (~partner_seen)).astype(jnp.float32))

    gain = (cost_old[:, None] - cost_new
            - W_CENTER * jnp.abs(cells.astype(jnp.float32) - 4.5).sum(-1)[None, :]
            - W_MOVED * displaced[:, None])
    proxy = jnp.where(move_legal, gain, -BIG)

    top_vals, top_idx = jax.lax.top_k(proxy.reshape(-1), TOP_K)
    cand_card = top_idx // n_cells
    cand_cell = cells[top_idx % n_cells]                          # [K, 2]
    cand_pos = jnp.where(idx[None, :, None] == cand_card[:, None, None],
                         cand_cell[:, None, :], pos[None, :, :])
    cand_cost = _potential(cand_pos, colors, known, known_count, n_unknown)
    cand_cost = cand_cost + W_MOVED * displaced[cand_card]
    cand_cost = jnp.where(top_vals > -BIG / 2.0, cand_cost, BIG)
    cand_cost = cand_cost + 1.0e-4 * jnp.arange(TOP_K, dtype=jnp.float32)
    best = jnp.argmin(cand_cost)
    move_action = (17 + cand_card[best] * 100
                   + cand_cell[best, 0] * 10 + cand_cell[best, 1])

    # ---------------- substep 3: hint economy ----------------
    reveal_ids = 1617 + jnp.arange(N_HINTS)
    reveal_legal = legal[reveal_ids]
    hint_size = jnp.asarray(HINT_SIZE_TUPLE, dtype=jnp.float32)
    reveal_action = reveal_ids[jnp.argmax(jnp.where(reveal_legal, hint_size, -BIG))]
    has_reveal = jnp.any(reveal_legal)

    place_ids = (1624 + jnp.arange(N_HINTS)[:, None] * 100
                 + pos[None, :, 0] * 10 + pos[None, :, 1])        # [7, 16]
    place_legal = legal[place_ids]

    same_neighbors = ((dist_now == 1) & same_pair).sum(-1)
    card_gap = gaps_now[jnp.clip(colors, 0, N_COLORS - 1)]
    attached = known & (same_neighbors > 0) & (card_gap <= 0.0)
    complete = known & (known_count[jnp.clip(colors, 0, N_COLORS - 1)] == PER_COLOR)

    place_value = (24.0 * p_correct - 12.0
                   + 10.0 * attached[None, :].astype(jnp.float32)
                   + 6.0 * complete[None, :].astype(jnp.float32)
                   - 8.0 * (~known)[None, :].astype(jnp.float32))
    place_value = jnp.where(place_legal, place_value, -BIG)
    flat_place = place_value.reshape(-1)
    best_place = jnp.argmax(flat_place)
    place_action = place_ids.reshape(-1)[best_place]

    # Placement locks the card for the rest of the episode, so all seven hints
    # are revealed first and cards are only locked once no reveal is left; the
    # forced fourteen-turn horizon makes this free in score terms.
    action3 = jnp.where(has_reveal, reveal_action, place_action)

    # ---------------- assemble ----------------
    action = jnp.where(substep == 0, action0,
                       jnp.where(substep == 1, observe_action,
                                 jnp.where(substep == 2, move_action, action3)))
    action = jnp.where(active, action, 0).astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))

    trace = {
        'gap': gaps_now.sum().astype(jnp.float32),
        'n_known': known.sum().astype(jnp.int32),
        'p_solved': p_solved.astype(jnp.float32),
        'score_now': score_now.astype(jnp.float32),
        'end_now': end_now,
        'substep': substep.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
