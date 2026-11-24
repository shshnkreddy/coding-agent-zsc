"""Grounded greedy-connectivity heuristic for two-player 9-card Yokai.

Design summary (see policy_ledger A01 for full rationale):

- Stateless per call: perfect-memory observations already replay every color
  this seat has personally observed, so no private memory is needed to track
  known colors across turns. Memory is kept as an unused scalar placeholder.
- A single length-1068 score vector is computed every call for *all* actions
  (observe, move, reveal, place, end, no-op), then masked by
  ``obs.legal_actions`` and argmax'd. Because each substep's legal actions
  only ever come from one or two of these categories, giving each category a
  distinct score magnitude band lets one unconditional vector double as a
  per-substep policy without any Python-level branching on traced values.
- Move scoring rewards relocating a card whose color this seat *knows*
  (directly observed, or logically deduced once two colors are fully
  accounted for) next to other known-same-color cards, and mildly penalizes
  disturbing cards this seat cannot reason about, preferring to disturb
  peripheral (few-neighbor) unknown cards over central ones.
- Hint scoring always prefers revealing over placing while any unrevealed
  hint remains legal, since revealed-unplaced (2 pts) beats even a correct
  placement (1 pt); once a placement is forced, only known-color-matching
  targets are preferred, with unknown targets preferred over known-wrong
  ones.
- The end-game action is only favored when this seat has logically certain
  knowledge of the full board (via direct observation plus elimination, since
  each color has exactly 3 members) and can verify all three colors are
  connected under that knowledge; otherwise ending is scored far below every
  other legal option so it is only ever taken when it is the sole legal
  action.

This relies only on locally observable/derivable facts (own past
observations, public positions/locks/hint reveals/adjacency), not on any
private symbolic convention, so it is meant to remain sensible against an
independently generated partner.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_greedy_connectivity',
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

_NUM_CARDS = 9
_NUM_COLORS = 3
_NUM_HINTS = 4
_GRID = 9
_NUM_CELLS = _GRID * _GRID  # 81
_NUM_ACTIONS = 1068


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    node_features = obs.node_features.astype(jnp.float32)
    adjacency = obs.adjacency_matrix.astype(jnp.float32)
    positions = obs.positions.astype(jnp.float32)  # (9, 2): x, y

    card_rows = node_features[:_NUM_CARDS]  # (9, 13)
    hint_rows = node_features[_NUM_CARDS:_NUM_CARDS + _NUM_HINTS]  # (4, 13)

    own_color_onehot = card_rows[:, 0:3]  # (9, 3), zero if unknown
    hint_colorset = hint_rows[:, 3:6]  # (4, 3), zero until revealed

    # --- card-card adjacency (public, from current positions) ---
    eye9 = jnp.eye(_NUM_CARDS, dtype=jnp.float32)
    adj_cards = adjacency[:_NUM_CARDS, :_NUM_CARDS]
    adj_cards_nodiag = adj_cards * (1.0 - eye9)
    current_neighbors = jnp.sum(adj_cards_nodiag, axis=1)  # (9,)

    # --- logical certainty via direct observation + elimination ---
    is_observed_self = jnp.sum(own_color_onehot, axis=1) > 0.0  # (9,)
    count_per_color = jnp.sum(own_color_onehot, axis=0)  # (3,)
    exhausted = count_per_color >= 3.0  # (3,)
    num_exhausted = jnp.sum(exhausted.astype(jnp.int32))
    fully_certain = num_exhausted >= 2

    not_exhausted = jnp.logical_not(exhausted).astype(jnp.float32)  # (3,)
    # One-hot for the sole remaining color exactly when num_exhausted == 2;
    # unused (gated by fully_certain) otherwise.
    deduced_onehot = not_exhausted

    known_full = jnp.where(
        is_observed_self[:, None],
        own_color_onehot,
        jnp.where(fully_certain, deduced_onehot[None, :], jnp.zeros((1, _NUM_COLORS))),
    )  # (9, 3)
    is_known_vec = (jnp.sum(known_full, axis=1) > 0.0).astype(jnp.float32)  # (9,)

    # --- per-color connectivity check under current knowledge ---
    m = known_full.T  # (3, 9)
    pair_edges = m @ adj_cards_nodiag @ m.T  # (3, 3)
    edges_per_color = jnp.diagonal(pair_edges) / 2.0  # (3,)
    connected_per_color = edges_per_color >= 2.0  # (3,)
    end_desirable = jnp.logical_and(fully_certain, jnp.all(connected_per_color))

    # A color shows a clear red flag when at least two known members exist
    # with zero adjacency among them; used below to steer both observation
    # priority (resolve the ambiguity) and placement risk (A03).
    count_known_full_per_color = jnp.sum(known_full, axis=0)  # (3,)
    obviously_disorganized = jnp.logical_and(
        count_known_full_per_color >= 2.0, edges_per_color < 1.0
    ).astype(jnp.float32)  # (3,)

    # --- observe scoring: indices [1, 10) for cards 0..8 ---
    # Re-observing an already-known (to this seat) unlocked card is legal but
    # wastes an observation slot, so unknown cards must dominate the score.
    observe_pref = adj_cards_nodiag @ is_known_vec  # (9,)
    observe_score = (1.0 - is_known_vec) * (100.0 + observe_pref) + is_known_vec * (-1000.0)

    # --- move scoring: indices [10, 739), 9 cards x 81 cells ---
    same_color_known = known_full @ known_full.T  # (9, 9): 1 iff both known & same color
    same_color_known_nodiag = same_color_known * (1.0 - eye9)
    old_adj_vec = jnp.sum(same_color_known_nodiag * adj_cards_nodiag, axis=1)  # (9,)

    cell_x = jnp.repeat(jnp.arange(_GRID, dtype=jnp.float32), _GRID)  # (81,) x outer
    cell_y = jnp.tile(jnp.arange(_GRID, dtype=jnp.float32), _GRID)  # (81,) y inner

    dx = cell_x[:, None] - positions[None, :, 0]  # (81, 9)
    dy = cell_y[:, None] - positions[None, :, 1]  # (81, 9)
    manhattan = jnp.abs(dx) + jnp.abs(dy)
    adjacent_cell_k = (manhattan == 1.0).astype(jnp.float32)  # (81, 9)

    new_adj_matrix = same_color_known_nodiag @ adjacent_cell_k.T  # (9, 81)
    gain = new_adj_matrix - old_adj_vec[:, None]  # (9, 81)

    # Secondary signal: even when no legal move creates immediate adjacency,
    # prefer moves that shrink the distance to the nearest known-same-color
    # partner, so multi-turn convergence is guided rather than aimless.
    big = 1000.0
    dist_current = jnp.abs(positions[:, None, 0] - positions[None, :, 0]) + \
        jnp.abs(positions[:, None, 1] - positions[None, :, 1])  # (9, 9)
    masked_dist_current = jnp.where(same_color_known_nodiag > 0.0, dist_current, big)
    old_min_dist = jnp.min(masked_dist_current, axis=1)  # (9,)
    has_partner = jnp.any(same_color_known_nodiag > 0.0, axis=1)  # (9,)

    dist_broadcast = jnp.where(
        same_color_known_nodiag[:, None, :] > 0.0, manhattan[None, :, :], big
    )  # (9, 81, 9)
    new_min_dist = jnp.min(dist_broadcast, axis=2)  # (9, 81)
    dist_gain = jnp.where(has_partner[:, None], old_min_dist[:, None] - new_min_dist, 0.0)

    # A known card with no known same-color partner yet cannot be usefully
    # clustered (gain/dist_gain are both 0 for it), so it must fall back to
    # the same low-disruption filler treatment as a truly unknown card
    # instead of scoring a dominant neutral 0 for every destination.
    actionable = is_known_vec * has_partner.astype(jnp.float32)  # (9,)
    lone_known = is_known_vec * (1.0 - has_partner.astype(jnp.float32))  # (9,)
    filler_score = -2.0 - current_neighbors[:, None]
    move_score = actionable[:, None] * (gain * 10.0 + dist_gain * 1.0) + \
        lone_known[:, None] * (-1.0 - 0.5 * current_neighbors[:, None]) + \
        (1.0 - actionable[:, None] - lone_known[:, None]) * filler_score
    move_score_flat = move_score.reshape(-1)  # (729,), i outer, cell inner

    # --- hint scoring ---
    reveal_score = jnp.full((_NUM_HINTS,), 1000.0, dtype=jnp.float32)

    occupied = (
        (cell_x[:, None] == positions[None, :, 0]) &
        (cell_y[:, None] == positions[None, :, 1])
    ).astype(jnp.float32)  # (81, 9)
    cell_known_color = occupied @ known_full  # (81, 3)
    known_sum = jnp.sum(cell_known_color, axis=1)  # (81,)

    dot_hc = hint_colorset @ cell_known_color.T  # (4, 81)
    correct = (dot_hc > 0.0).astype(jnp.float32)  # (4, 81)
    incorrect = jnp.broadcast_to(known_sum[None, :], (_NUM_HINTS, _NUM_CELLS)) * \
        (dot_hc <= 0.0).astype(jnp.float32)

    # Locking (placing) a card whose color shows a clear red flag -- at least
    # two known same-color members with zero adjacency among them -- risks
    # permanently freezing a broken cluster, so mildly discourage it. (A
    # stricter version requiring full 3-member proof of non-connectivity
    # regressed self-play badly since that proof is rarely available before
    # full certainty; this only fires on an already-visible problem.)
    cell_disorganized = cell_known_color @ obviously_disorganized  # (81,)

    place_score = -500.0 + correct * 100.0 - incorrect * 100.0 - \
        jnp.broadcast_to(cell_disorganized[None, :], (_NUM_HINTS, _NUM_CELLS)) * 80.0
    place_score_flat = place_score.reshape(-1)  # (324,), h outer, cell inner

    end_score = jnp.where(end_desirable, 1e6, -1e6)

    scores = jnp.zeros((_NUM_ACTIONS,), dtype=jnp.float32)
    scores = scores.at[0].set(0.0)
    scores = scores.at[1:10].set(observe_score)
    scores = scores.at[10:739].set(move_score_flat)
    scores = scores.at[739:743].set(reveal_score)
    scores = scores.at[743:1067].set(place_score_flat)
    scores = scores.at[1067].set(end_score)

    noise = jax.random.uniform(rng, shape=(_NUM_ACTIONS,), minval=-0.01, maxval=0.01)
    masked_scores = jnp.where(obs.legal_actions, scores + noise, -jnp.inf)
    action = jnp.argmax(masked_scores).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)

    trace = {
        "fully_certain": fully_certain.astype(jnp.int32),
        "all_connected": jnp.all(connected_per_color).astype(jnp.int32),
        "num_observed_self": jnp.sum(is_observed_self.astype(jnp.int32)),
        "num_exhausted": num_exhausted,
        "num_locked": jnp.sum(card_rows[:, 8]).astype(jnp.int32),
    }
    return action, next_memory, trace
