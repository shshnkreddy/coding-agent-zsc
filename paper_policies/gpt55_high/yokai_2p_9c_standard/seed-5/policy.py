"""Convention-readable Yokai policy for the 9-card cooperative task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


UNKNOWN = -1


def _initial_memory():
    return {
        "known": jnp.full((9,), UNKNOWN, dtype=jnp.int32),
        "belief": jnp.full((9,), UNKNOWN, dtype=jnp.int32),
        "prev_pos": jnp.zeros((9, 2), dtype=jnp.int32),
        "role": jnp.asarray(UNKNOWN, dtype=jnp.int32),
    }


def init_agent():
    return _initial_memory()


def _reset_memory(memory, reset):
    init = _initial_memory()
    return {
        "known": jnp.where(reset, init["known"], memory["known"]),
        "belief": jnp.where(reset, init["belief"], memory["belief"]),
        "prev_pos": jnp.where(reset, init["prev_pos"], memory["prev_pos"]),
        "role": jnp.where(reset, init["role"], memory["role"]),
    }


def _update_memory(obs, memory, reset):
    memory = _reset_memory(memory, reset)
    features = obs.node_features.astype(jnp.int32)
    role = jnp.where(reset, features[0, 10], memory["role"]).astype(jnp.int32)
    colors_onehot = features[:9, 0:3]
    observed = jnp.sum(colors_onehot, axis=1) > 0
    observed_color = jnp.argmax(colors_onehot, axis=1).astype(jnp.int32)
    known = jnp.where(observed, observed_color, memory["known"])

    positions = obs.positions.astype(jnp.int32)
    moved = jnp.any(positions != memory["prev_pos"], axis=1) & (~reset)
    row_color = positions[:, 1] - 3
    row_signal = (row_color >= 0) & (row_color < 3)

    belief0 = jnp.where(observed, observed_color, memory["belief"])
    inferable = (belief0 < 0) & moved & row_signal
    belief = jnp.where(inferable, row_color, belief0)
    belief = jnp.where(known >= 0, known, belief)
    belief = _complete_single_unknown(belief)

    next_memory = {"known": known, "belief": belief, "prev_pos": positions, "role": role}
    return next_memory


def _complete_single_unknown(colors):
    unknown = colors < 0
    unknown_count = jnp.sum(unknown.astype(jnp.int32))
    counts = jnp.stack([
        jnp.sum((colors == 0).astype(jnp.int32)),
        jnp.sum((colors == 1).astype(jnp.int32)),
        jnp.sum((colors == 2).astype(jnp.int32)),
    ])
    missing_color = jnp.argmax(3 - counts).astype(jnp.int32)
    fill = unknown & (unknown_count == 1)
    return jnp.where(fill, missing_color, colors)


def _first_true(mask, offset):
    return (jnp.argmax(mask).astype(jnp.int32) + jnp.asarray(offset, dtype=jnp.int32))


def _last_true(mask, offset):
    index = jnp.asarray(mask.shape[0] - 1, dtype=jnp.int32) - jnp.argmax(mask[::-1]).astype(jnp.int32)
    return index + jnp.asarray(offset, dtype=jnp.int32)


def _clustered(positions, colors):
    known_all = jnp.all(colors >= 0)
    dx = jnp.abs(positions[:, None, 0] - positions[None, :, 0])
    dy = jnp.abs(positions[:, None, 1] - positions[None, :, 1])
    adjacent = (dx + dy) == 1
    eye = jnp.eye(9, dtype=bool)

    connected = []
    for color in range(3):
        mask = colors == color
        same_adj = (adjacent | eye) & mask[:, None] & mask[None, :]
        reach = jax.nn.one_hot(jnp.argmax(mask.astype(jnp.int32)), 9, dtype=bool) & mask
        for _ in range(8):
            reach = mask & (reach | jnp.any(same_adj & reach[None, :], axis=1))
        connected.append(jnp.sum(reach.astype(jnp.int32)) == 3)
    return known_all & connected[0] & connected[1] & connected[2]


def _clustered_batch(positions, colors):
    known_all = jnp.all(colors >= 0)
    dx = jnp.abs(positions[:, :, None, 0] - positions[:, None, :, 0])
    dy = jnp.abs(positions[:, :, None, 1] - positions[:, None, :, 1])
    adjacent = (dx + dy) == 1
    eye = jnp.eye(9, dtype=bool)[None, :, :]

    connected = []
    for color in range(3):
        mask = colors == color
        same_adj = (adjacent | eye) & mask[None, :, None] & mask[None, None, :]
        reach = (jax.nn.one_hot(jnp.argmax(mask.astype(jnp.int32)), 9, dtype=bool) & mask)[None, :]
        for _ in range(8):
            reach = mask[None, :] & (reach | jnp.any(same_adj & reach[:, None, :], axis=2))
        connected.append(jnp.sum(reach.astype(jnp.int32), axis=1) == 3)
    return known_all & connected[0] & connected[1] & connected[2]


def _role_ordered(mask, role, offset):
    low = _first_true(mask, offset)
    high = _last_true(mask, offset)
    return jnp.where(role == 1, low, high).astype(jnp.int32)


def _observe_action(obs, known, role):
    features = obs.node_features.astype(jnp.int32)
    unlocked = features[:9, 8] == 0
    legal_obs = obs.legal_actions[1:10]
    preferred = legal_obs & unlocked & (known < 0)
    any_preferred = jnp.any(preferred)
    choice = jnp.where(any_preferred, _role_ordered(preferred, role, 1), _role_ordered(legal_obs, role, 1))
    return choice.astype(jnp.int32)


def _board_scores(positions, colors):
    card_ids = jnp.arange(9, dtype=jnp.int32)
    move_ids = jnp.arange(729, dtype=jnp.int32)
    move_card = move_ids // 81
    move_cell = move_ids % 81
    move_x = move_cell // 9
    move_y = move_cell % 9

    pos = positions[None, :, :]
    dest = jnp.stack([move_x, move_y], axis=1)
    moved_pos = jnp.where(move_card[:, None, None] == card_ids[None, :, None], dest[:, None, :], pos)

    known = colors >= 0
    target_y = colors + 3
    y_dist = jnp.abs(moved_pos[:, :, 1] - target_y[None, :])
    x_low = jnp.maximum(0, 3 - moved_pos[:, :, 0])
    x_high = jnp.maximum(0, moved_pos[:, :, 0] - 5)
    x_dist = x_low + x_high
    dist = (3 * y_dist + x_dist) * known[None, :]
    distance_score = -jnp.sum(dist, axis=1)

    dx = jnp.abs(moved_pos[:, :, None, 0] - moved_pos[:, None, :, 0])
    dy = jnp.abs(moved_pos[:, :, None, 1] - moved_pos[:, None, :, 1])
    adjacent = (dx + dy) == 1
    same_color = (colors[:, None] == colors[None, :]) & known[:, None] & known[None, :]
    upper = card_ids[:, None] < card_ids[None, :]
    edge_score = 14 * jnp.sum((adjacent & same_color[None, :, :] & upper[None, :, :]).astype(jnp.int32), axis=(1, 2))

    in_row = (moved_pos[:, :, 1] == target_y[None, :]) & known[None, :]
    row_score = 6 * jnp.sum(in_row.astype(jnp.int32), axis=1)
    moved_known = colors[move_card] >= 0
    known_move_bonus = jnp.where(moved_known, 2, -4)
    return distance_score + edge_score + row_score + known_move_bonus


def _move_action(obs, belief):
    legal_moves = obs.legal_actions[10:739]
    positions = obs.positions.astype(jnp.int32)
    scores = _board_scores(positions, belief)
    card_ids = jnp.arange(9, dtype=jnp.int32)
    move_ids = jnp.arange(729, dtype=jnp.int32)
    move_card = move_ids // 81
    move_cell = move_ids % 81
    move_x = move_cell // 9
    move_y = move_cell % 9
    dest = jnp.stack([move_x, move_y], axis=1)
    candidate_positions = jnp.where(move_card[:, None, None] == card_ids[None, :, None], dest[:, None, :], positions[None, :, :])
    cluster_after = _clustered_batch(candidate_positions, belief)
    cluster_now = _clustered(positions, belief)
    cluster_bonus = jnp.where(cluster_after, jnp.asarray(5000, dtype=jnp.int32), jnp.where(cluster_now, jnp.asarray(-5000, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)))
    scores = scores + cluster_bonus
    masked = jnp.where(legal_moves, scores, jnp.asarray(-1000000, dtype=jnp.int32))
    return (jnp.argmax(masked).astype(jnp.int32) + 10)


def _hint_action(obs, known):
    legal_reveal = obs.legal_actions[739:743]
    reveal = _first_true(legal_reveal, 739)

    features = obs.node_features.astype(jnp.int32)
    hint_colors = features[9:13, 3:6] > 0
    positions = obs.positions.astype(jnp.int32)
    place_ids = jnp.arange(324, dtype=jnp.int32)
    place_hint = place_ids // 81
    place_cell = place_ids % 81
    place_x = place_cell // 9
    place_y = place_cell % 9
    card_at_place = jnp.argmax(
        ((positions[:, 0][None, :] == place_x[:, None]) & (positions[:, 1][None, :] == place_y[:, None])).astype(jnp.int32),
        axis=1,
    )
    card_known = known[card_at_place] >= 0
    hint_has_color = hint_colors[place_hint, jnp.maximum(known[card_at_place], 0)]
    target_row = positions[card_at_place, 1] == (known[card_at_place] + 3)
    place_score = 100 * (card_known & hint_has_color).astype(jnp.int32) + 5 * target_row.astype(jnp.int32)
    legal_place = obs.legal_actions[743:1067]
    masked_place = jnp.where(legal_place, place_score, jnp.asarray(-1000000, dtype=jnp.int32))
    place = jnp.argmax(masked_place).astype(jnp.int32) + 743

    return jnp.where(jnp.any(legal_reveal), reveal, place).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    next_memory = _update_memory(obs, memory, reset)
    known = next_memory["known"]
    belief = next_memory["belief"]
    role = next_memory["role"]
    substep = obs.node_features[0, 11].astype(jnp.int32)
    positions = obs.positions.astype(jnp.int32)

    can_end = _clustered(positions, belief) & obs.legal_actions[1067]

    def step0(_):
        return jnp.where(can_end, jnp.asarray(1067, dtype=jnp.int32), _observe_action(obs, known, role))

    def step1(_):
        return _observe_action(obs, known, role)

    def step2(_):
        return _move_action(obs, belief)

    def step3(_):
        return _hint_action(obs, known)

    proposed = jax.lax.switch(substep, (step0, step1, step2, step3), operand=None)
    fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    action = jnp.where(obs.legal_actions[proposed], proposed, fallback).astype(jnp.int32)

    trace = {
        "known_count": jnp.sum((known >= 0).astype(jnp.int32)),
        "belief_count": jnp.sum((belief >= 0).astype(jnp.int32)),
        "clustered": _clustered(positions, belief),
        "role": role,
        "substep": substep,
    }
    return action, next_memory, trace
