"""Greedy connectivity-seeking policy for two-player 9-card Yokai.

Design summary (see context/environment/README.md for mechanics):
- Positions are always public; only card colors are hidden until observed.
- Each seat keeps a private belief of colors it has personally revealed
  (memory persists across its own turns; the environment clears the
  transient own-color observation features at each turn boundary).
- Every action type (observe / move / reveal-hint / place-hint / end-game)
  gets a heuristic score; the environment's legal_actions mask is applied
  before argmax, so legality is always respected regardless of substep.
- Ending the game (action 1067) as soon as all colors are known and fully
  connected is preferred over taking further forced hint actions, since the
  scoring formula 5*U + 2*R + C - I strictly favors leaving hints unrevealed.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'greedy_connectivity_v1',
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

_NEG = -1e6


def init_agent():
    return jnp.full((9,), -1, dtype=jnp.int8)


def _transitive_closure(adjacency):
    reach = adjacency | jnp.eye(9, dtype=bool)

    def body(r, _):
        prod = jnp.matmul(r.astype(jnp.int32), r.astype(jnp.int32)) > 0
        return r | prod, None

    reach, _ = jax.lax.scan(body, reach, xs=None, length=4)
    return reach


def _manhattan(pos):
    dx = jnp.abs(pos[:, 0:1] - pos[:, 0:1].T)
    dy = jnp.abs(pos[:, 1:2] - pos[:, 1:2].T)
    return dx + dy


def _same_known_color(colors):
    known = colors != -1
    return (colors[:, None] == colors[None, :]) & known[:, None] & known[None, :]


def _color_connected_sum(colors, pos):
    dist = _manhattan(pos)
    same = _same_known_color(colors)
    adjacency = (dist == 1) & same
    reach = _transitive_closure(adjacency)
    total = jnp.float32(0.0)
    for color in range(3):
        mask = colors == color
        count = jnp.sum(mask)
        pair_ok = jnp.all(jnp.where(mask[:, None] & mask[None, :], reach, True))
        total = total + jnp.where(count == 3, pair_ok.astype(jnp.float32), 0.0)
    return total


def _shaping_sum(colors, pos):
    dist = _manhattan(pos)
    same = _same_known_color(colors)
    upper = jnp.triu(jnp.ones((9, 9), dtype=bool), 1)
    pair_mask = same & upper
    total_dist = jnp.sum(jnp.where(pair_mask, dist, 0)).astype(jnp.float32)
    adjacent_pairs = jnp.sum(jnp.where(pair_mask & (dist == 1), 1, 0)).astype(jnp.float32)
    return jnp.float32(-0.1) * total_dist + jnp.float32(5.0) * adjacent_pairs


def _per_card_safe_to_lock(colors, pos):
    """True for cards whose color group is fully known and already connected."""
    dist = _manhattan(pos)
    same = _same_known_color(colors)
    adjacency = (dist == 1) & same
    reach = _transitive_closure(adjacency)
    safe = jnp.zeros(9, dtype=bool)
    for color in range(3):
        mask = colors == color
        count = jnp.sum(mask)
        pair_ok = jnp.all(jnp.where(mask[:, None] & mask[None, :], reach, True))
        good = (count == 3) & pair_ok
        safe = jnp.where(mask & good, True, safe)
    return safe


def _infer_full_board(observed_color):
    """Deduce colors of unseen cards once two colors are fully accounted for.

    Each color has exactly 3 cards. If a seat has directly observed all 3
    members of two colors, every remaining unseen card must be the third
    color with certainty, even without observing it directly.
    """
    counts = jnp.stack([jnp.sum(observed_color == c) for c in range(3)])
    full = counts == 3
    num_full = jnp.sum(full)
    missing_color = jnp.argmin(jnp.where(full, 999, counts)).astype(jnp.int8)
    should_infer = num_full == 2
    return jnp.where((observed_color == -1) & should_infer, missing_color, observed_color)


def agent_act(obs, reset, memory, rng):
    node_features = obs.node_features
    positions = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions

    color_onehot = node_features[0:9, 0:3]
    reveal_mask = jnp.any(color_onehot != 0, axis=-1)
    revealed_color = jnp.argmax(color_onehot, axis=-1).astype(jnp.int8)
    observed_color = jnp.where(reveal_mask, revealed_color, memory)
    observed_color = jnp.where(reset, jnp.full((9,), -1, dtype=jnp.int8), observed_color)
    known_color = _infer_full_board(observed_color)

    hint_colorset = node_features[9:13, 3:6] != 0  # (4, 3) bool

    current_conn = _color_connected_sum(known_color, positions)
    all_known = jnp.all(known_color != -1)
    fully_connected = all_known & (current_conn == jnp.float32(3.0))

    def move_score(i, x, y):
        new_pos = positions.at[i].set(jnp.array([x, y], dtype=jnp.int32))
        old_xy = positions[i]
        travel = jnp.abs(old_xy[0] - x) + jnp.abs(old_xy[1] - y)
        return jnp.float32(1000.0) * _color_connected_sum(known_color, new_pos) + \
            _shaping_sum(known_color, new_pos) - jnp.float32(0.03) * travel.astype(jnp.float32)

    idx = jnp.arange(9)
    score_y = jax.vmap(move_score, in_axes=(None, None, 0))
    score_xy = jax.vmap(score_y, in_axes=(None, 0, None))
    score_ixy = jax.vmap(score_xy, in_axes=(0, None, None))
    move_scores = score_ixy(idx, idx, idx).reshape(729)

    observe_scores = jnp.where(known_color == -1, jnp.float32(10.0), jnp.float32(1.0))

    reveal_scores = jnp.full((4,), jnp.float32(100.0))

    # A placement locks its card forever, so a known card whose color group
    # is not yet fully connected should not be locked in even if its color
    # matches the hint: it may still need to move to finish the puzzle.
    safe_to_lock = _per_card_safe_to_lock(known_color, positions)  # (9,) bool
    is_known = known_color != -1
    kc_safe = jnp.clip(known_color, 0, 2)
    contains = hint_colorset[:, kc_safe]  # (4, 9) bool
    safe_b = jnp.broadcast_to(safe_to_lock[None, :], contains.shape)
    known_b = jnp.broadcast_to(is_known[None, :], contains.shape)
    known_score = jnp.where(
        contains,
        jnp.where(safe_b, jnp.float32(1000.0), jnp.float32(200.0)),
        jnp.where(safe_b, jnp.float32(-200.0), jnp.float32(-1000.0)),
    )
    place_score_hi = jnp.where(known_b, known_score, jnp.float32(50.0))
    idx_xy = positions[:, 0] * 9 + positions[:, 1]
    place_full = jnp.full((4, 81), _NEG)
    place_full = place_full.at[:, idx_xy].set(place_score_hi)
    place_scores = place_full.reshape(324)

    end_score = jnp.where(fully_connected, jnp.float32(1e7), _NEG)

    score = jnp.concatenate([
        jnp.array([_NEG]),
        observe_scores,
        move_scores,
        reveal_scores,
        place_scores,
        jnp.array([end_score]),
    ])

    jitter = jax.random.uniform(rng, shape=score.shape, minval=0.0, maxval=1e-3,
                                 dtype=jnp.float32)
    score = score + jitter

    masked = jnp.where(legal, score, -jnp.inf)
    action = jnp.argmax(masked).astype(jnp.int32)

    return action, observed_color, {}
