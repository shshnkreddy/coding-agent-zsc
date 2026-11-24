"""Locally-greedy Yokai coordination policy.

Stateless per-substep scoring: build one score for every one of the 2325
actions, mask by obs.legal_actions, and argmax. Perfect memory means the
environment itself re-supplies this seat's previously observed colors
whenever it is active, so no private memory is needed beyond a fixed-shape
placeholder.

Heuristics (all derived only from this seat's own observation):
- Observe: prefer cards unseen by both seats, then cards unseen by this
  seat but seen by the partner (partner-seen flag), to grow joint coverage.
- Move: among legal destinations, prefer cells that increase the moved
  card's same-known-color adjacency, plus a smaller pull toward a
  fixed per-color anchor cell computed from public game constants alone
  (see _ANCHOR_X/_ANCHOR_Y below). Because every independently generated
  seat can derive the same anchor from public information with no
  coordination, cards of a color discovered by different players at
  different times still converge on the same region. Legality (and thus
  whole-board connectivity) is enforced by the environment's mask, not
  recomputed here.
- Hint substep: always reveal an unrevealed hint before placing (safe,
  informative, and critically delays the irreversible lock that placement
  imposes on a card until it has had time to reach its anchor -- placing
  early risks permanently stranding a not-yet-repositioned card). Once all
  hints are revealed, place on a card with a known matching color when
  possible (guaranteed correct), preferring among those the card closest
  to its own anchor (least likely to still need to move); otherwise prefer
  larger hint color sets on unknown-color cards (higher expected value)
  and avoid known non-matching cards.
- Never choose the explicit end-game action: a single seat can observe at
  most 14 of 16 cards over the episode, so it can never verify full board
  connectivity on its own, and ending on an unconnected board scores 0
  (identical to the worst case of just continuing).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'greedy_connectivity',
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

_GRID = 10
_NUM_CARDS = 16
_NUM_HINTS = 7
_NUM_COLORS = 4
_BIG = 10.0
_REVEAL_SCORE = 1000.0
_ALPHA = 0.08
_BETA = 0.7
_ETA = 0.5
# Fixed per-color anchor cells, derived only from public game constants
# (the reset rule places card i at (floor(i/4)+3, i%4+3), i.e. the initial
# 4x4 block spans x,y in [3,6]) and the public color ID (0..3). Every seat
# computes the identical anchor independently with no coordination: split
# the initial block into four 2x2 sub-quadrants and send color c toward
# its sub-quadrant's center.
_ANCHOR_X = (3.5, 5.5, 3.5, 5.5)
_ANCHOR_Y = (3.5, 3.5, 5.5, 5.5)


def init_agent():
    return jnp.zeros((), dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng

    card_features = obs.node_features[:_NUM_CARDS]
    hint_features = obs.node_features[_NUM_CARDS:_NUM_CARDS + _NUM_HINTS]

    colors_onehot = card_features[:, 0:4].astype(jnp.float32)
    known = jnp.sum(colors_onehot, axis=-1) > 0
    known_f = known.astype(jnp.float32)
    partner_seen = card_features[:, 11].astype(jnp.float32)

    x = obs.positions[:, 0].astype(jnp.int32)
    y = obs.positions[:, 1].astype(jnp.int32)

    hint_sets = hint_features[:, 4:8].astype(jnp.float32)
    hint_setsize = jnp.sum(hint_sets, axis=-1)

    # --- observe scores (16,) ---
    observe_scores = jnp.where(
        known, -1000.0, jnp.where(partner_seen > 0, 1.0, 2.0)
    )

    # --- move scores (16, 10, 10) -> flattened (1600,) ---
    x_onehot = jax.nn.one_hot(x, _GRID, dtype=jnp.float32)
    y_onehot = jax.nn.one_hot(y, _GRID, dtype=jnp.float32)

    occupancy = jnp.einsum('nc,nx,ny->cxy', colors_onehot, x_onehot, y_onehot)
    contrib = jnp.einsum('nc,nx,ny->ncxy', colors_onehot, x_onehot, y_onehot)
    adjusted = occupancy[None, :, :, :] - contrib  # (16,4,10,10)

    padded = jnp.pad(adjusted, ((0, 0), (0, 0), (1, 1), (1, 1)))
    neighbor_sum = (
        padded[:, :, 0:_GRID, 1:_GRID + 1]
        + padded[:, :, 2:_GRID + 2, 1:_GRID + 1]
        + padded[:, :, 1:_GRID + 1, 0:_GRID]
        + padded[:, :, 1:_GRID + 1, 2:_GRID + 2]
    )  # (16,4,10,10)

    dest_same = jnp.einsum('nc,ncxy->nxy', colors_onehot, neighbor_sum)  # (16,10,10)
    total_known_neighbors = jnp.sum(neighbor_sum, axis=1)  # (16,10,10)
    dest_diff = known_f[:, None, None] * (total_known_neighbors - dest_same)

    cur_same = dest_same[jnp.arange(_NUM_CARDS), x, y]  # (16,)

    anchor_x_table = jnp.asarray(_ANCHOR_X, dtype=jnp.float32)
    anchor_y_table = jnp.asarray(_ANCHOR_Y, dtype=jnp.float32)
    anchor_x = colors_onehot @ anchor_x_table  # (16,)
    anchor_y = colors_onehot @ anchor_y_table  # (16,)
    xs = jnp.arange(_GRID, dtype=jnp.float32)
    dist_x = jnp.abs(xs[None, :] - anchor_x[:, None])  # (16,10)
    dist_y = jnp.abs(xs[None, :] - anchor_y[:, None])  # (16,10)
    dist_grid = dist_x[:, :, None] + dist_y[:, None, :]  # (16,10,10)
    cur_dist = jnp.abs(x.astype(jnp.float32) - anchor_x) + jnp.abs(y.astype(jnp.float32) - anchor_y)  # (16,)
    dist_improvement = known_f[:, None, None] * (cur_dist[:, None, None] - dist_grid)

    improvement = (
        dest_same - cur_same[:, None, None] - _ALPHA * dest_diff + _BETA * dist_improvement
    )
    move_scores = improvement.reshape(_NUM_CARDS * _GRID * _GRID)

    # --- hint reveal scores (7,) ---
    reveal_scores = jnp.full((_NUM_HINTS,), _REVEAL_SCORE)

    # --- hint place scores (7, 10, 10) -> flattened (700,) ---
    occupant_onehot = jnp.einsum('nc,nx,ny->xyc', colors_onehot, x_onehot, y_onehot)  # (10,10,4)
    occupant_known = jnp.einsum('n,nx,ny->xy', known_f, x_onehot, y_onehot)  # (10,10)

    match = jnp.einsum('xyc,hc->hxy', occupant_onehot, hint_sets)  # (7,10,10) in {0,1}
    # Among multiple guaranteed-correct (h, card) choices, prefer locking a
    # card that is already close to its own color's anchor: placing a hint
    # is irreversible (the card can never move or be re-observed again), so
    # spending a hint on a card that still needs to travel would strand it
    # away from its cluster.
    occupant_dist = jnp.einsum(
        'n,nx,ny->xy', known_f * cur_dist, x_onehot, y_onehot
    )  # (10,10)
    known_term = occupant_known[None, :, :] * (
        (2.0 * match - 1.0) * _BIG - _ETA * occupant_dist[None, :, :]
    )
    unknown_term = (1.0 - occupant_known[None, :, :]) * (
        (hint_setsize / 4.0)[:, None, None] * 2.0 - 1.0
    )
    place_scores = (known_term + unknown_term).reshape(_NUM_HINTS * _GRID * _GRID)

    # --- certain early-end check ---
    # Each color has exactly 4 cards. If this seat has personally observed
    # all 4 members of at least 3 colors, the 4th color's membership is
    # forced by elimination: full board color knowledge is then *certain*,
    # not a guess. Positions are always public, so connectivity of every
    # color can then be verified exactly, and ending immediately banks any
    # still-unrevealed/unplaced hints at their higher U/R value instead of
    # letting the mandatory 14-turn hint cycle force everything down to C/I.
    color_counts = jnp.sum(colors_onehot, axis=0)  # (4,)
    complete_colors = color_counts >= 4.0
    num_complete = jnp.sum(complete_colors.astype(jnp.float32))
    missing_color_onehot = (~complete_colors).astype(jnp.float32)  # one-hot iff exactly 1 missing

    deduced_onehot = jnp.where(
        known[:, None], colors_onehot, missing_color_onehot[None, :]
    )  # (16,4), valid only when num_complete >= 3

    color_grid = jnp.einsum('nc,nx,ny->cxy', deduced_onehot, x_onehot, y_onehot)  # (4,10,10)

    def _connected(occ):
        flat = occ.reshape(_GRID * _GRID)
        seed_idx = jnp.argmax(flat)
        reach = jax.nn.one_hot(seed_idx, _GRID * _GRID, dtype=jnp.float32).reshape(_GRID, _GRID)
        reach = reach * occ
        for _ in range(4):
            padded = jnp.pad(reach, ((1, 1), (1, 1)))
            expanded = (
                padded[0:_GRID, 1:_GRID + 1]
                + padded[2:_GRID + 2, 1:_GRID + 1]
                + padded[1:_GRID + 1, 0:_GRID]
                + padded[1:_GRID + 1, 2:_GRID + 2]
                + reach
            ) > 0
            reach = expanded.astype(jnp.float32) * occ
        return jnp.sum(reach) > 3.5

    all_connected = jnp.all(jnp.stack([_connected(color_grid[c]) for c in range(_NUM_COLORS)]))
    can_end = (num_complete >= 3.0) & all_connected

    noop_score = jnp.zeros((1,), dtype=jnp.float32)
    end_score = jnp.where(can_end, jnp.full((1,), 1e6), jnp.full((1,), -1e9)).astype(jnp.float32)

    full_score = jnp.concatenate([
        noop_score,
        observe_scores.astype(jnp.float32),
        move_scores.astype(jnp.float32),
        reveal_scores.astype(jnp.float32),
        place_scores.astype(jnp.float32),
        end_score,
    ])

    final_score = jnp.where(obs.legal_actions, full_score, -jnp.inf)
    action = jnp.argmax(final_score).astype(jnp.int32)

    zero = jnp.zeros((), dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)
    return action, next_memory, {}
