"""Canonical public sorting policy for the 16-card Yokai task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'canonical_region_sorter', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NUM_CARDS = 16
TARGET_CELLS = (
    (3, 3), (3, 4), (4, 3), (4, 4),
    (3, 5), (3, 6), (4, 5), (4, 6),
    (5, 3), (5, 4), (6, 3), (6, 4),
    (5, 5), (5, 6), (6, 5), (6, 6),
)
TARGET_COLORS = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
PARKING_CELLS = (
    (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 2), (4, 2), (5, 2), (6, 2),
    (7, 3), (7, 4), (7, 5), (7, 6),
    (3, 7), (4, 7), (5, 7), (6, 7),
)


def _target_cells():
    return jnp.asarray(TARGET_CELLS, dtype=jnp.int32)


def _target_colors():
    return jnp.asarray(TARGET_COLORS, dtype=jnp.int32)


def _move_dests():
    return jnp.concatenate(
        [_target_cells(), jnp.asarray(PARKING_CELLS, dtype=jnp.int32)],
        axis=0,
    )


def init_agent():
    return {
        "colors": jnp.full((NUM_CARDS,), -1, dtype=jnp.int32),
        "prev_pos": jnp.zeros((NUM_CARDS, 2), dtype=jnp.int32),
    }


def _target_region_color(pos):
    x = pos[..., 0]
    y = pos[..., 1]
    left = (x >= 3) & (x <= 4)
    right = (x >= 5) & (x <= 6)
    low = (y >= 3) & (y <= 4)
    high = (y >= 5) & (y <= 6)
    c0 = left & low
    c1 = left & high
    c2 = right & low
    c3 = right & high
    color = jnp.full(x.shape, -1, dtype=jnp.int32)
    color = jnp.where(c0, 0, color)
    color = jnp.where(c1, 1, color)
    color = jnp.where(c2, 2, color)
    color = jnp.where(c3, 3, color)
    return color


def _first_legal(legal_actions):
    return jnp.argmax(legal_actions).astype(jnp.int32)


def _select_scored_action(action_ids, scores, legal_actions, fallback):
    legal = legal_actions[action_ids]
    valid_scores = jnp.where(legal, scores, -1_000_000)
    best = jnp.argmax(valid_scores)
    has_valid = jnp.any(legal & (scores > -999_999))
    return jnp.where(has_valid, action_ids[best], fallback).astype(jnp.int32)


def _refresh_memory(obs, reset, memory):
    pos = obs.positions.astype(jnp.int32)
    reset_colors = jnp.full((NUM_CARDS,), -1, dtype=jnp.int32)
    colors = jnp.where(reset, reset_colors, memory["colors"])
    prev_pos = jnp.where(reset, pos, memory["prev_pos"])

    color_bits = obs.node_features[:NUM_CARDS, 0:4].astype(jnp.int32)
    saw_color = jnp.sum(color_bits, axis=1) > 0
    observed_color = jnp.argmax(color_bits, axis=1).astype(jnp.int32)
    colors = jnp.where(saw_color, observed_color, colors)

    changed = jnp.any(pos != prev_pos, axis=1) & (~reset)
    dest_region = _target_region_color(pos)
    inferred = changed & (dest_region >= 0) & (colors < 0)
    colors = jnp.where(inferred, dest_region, colors)
    colors = _complete_by_counts(colors)

    next_memory = {"colors": colors, "prev_pos": pos}
    return colors, next_memory


def _complete_by_counts(colors):
    counts = jnp.asarray(
        [
            jnp.sum(colors == 0),
            jnp.sum(colors == 1),
            jnp.sum(colors == 2),
            jnp.sum(colors == 3),
        ],
        dtype=jnp.int32,
    )
    missing = 4 - counts
    unknown = colors < 0
    unknown_count = jnp.sum(unknown)
    positive_missing = missing > 0
    only_one_missing_color = jnp.sum(positive_missing) == 1
    inferred_color = jnp.argmax(missing).astype(jnp.int32)
    can_infer = (unknown_count > 0) & only_one_missing_color
    return jnp.where(unknown & can_infer, inferred_color, colors)


def _all_colors_connected(obs, colors):
    adj = obs.adjacency_matrix[:NUM_CARDS, :NUM_CARDS].astype(jnp.bool_)
    all_known = jnp.all(colors >= 0)
    connected = jnp.asarray(True)
    for color in range(4):
        mask = colors == color
        seed = jnp.argmax(mask).astype(jnp.int32)
        reach = jax.nn.one_hot(seed, NUM_CARDS, dtype=jnp.bool_) & mask
        for _ in range(NUM_CARDS):
            reach = reach | (jnp.any(adj & reach[None, :], axis=1) & mask)
        connected = connected & (jnp.sum(reach) == jnp.sum(mask)) & (jnp.sum(mask) == 4)
    return all_known & connected


def _observe_action(obs, colors):
    card_ids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    other_seen = obs.node_features[:NUM_CARDS, 11].astype(jnp.int32) > 0
    unknown = colors < 0
    scores = jnp.where(unknown & (~other_seen), 1000, 0)
    scores = jnp.where(unknown & other_seen, 800, scores)
    scores = jnp.where(~unknown, 10, scores)
    action_ids = 1 + card_ids
    return _select_scored_action(action_ids, scores, obs.legal_actions, _first_legal(obs.legal_actions))


def _move_action(obs, colors):
    card_ids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    move_dests = _move_dests()
    dest_is_target = jnp.arange(32, dtype=jnp.int32) < 16
    dest_target_colors = jnp.concatenate(
        [_target_colors(), jnp.full((16,), -1, dtype=jnp.int32)],
        axis=0,
    )
    pos = obs.positions.astype(jnp.int32)
    locks = obs.node_features[:NUM_CARDS, 10].astype(jnp.int32) > 0
    current_region = _target_region_color(pos)
    known = colors >= 0
    correct = known & (current_region == colors)

    occupied = jnp.any(
        jnp.all(move_dests[:, None, :] == pos[None, :, :], axis=2),
        axis=1,
    )
    dest_empty = ~occupied

    card_idx = card_ids[:, None]
    dest_xy = move_dests[None, :, :]
    action_ids = (
        17
        + card_idx * 100
        + dest_xy[:, :, 0] * 10
        + dest_xy[:, :, 1]
    ).astype(jnp.int32)

    target_dest = dest_is_target[None, :]
    parking_dest = ~target_dest
    dest_color = dest_target_colors[None, :]
    empty = dest_empty[None, :]

    movable = (~locks)[:, None] & known[:, None]
    match_target = target_dest & empty & (colors[:, None] == dest_color) & (~correct)[:, None]
    from_parking = current_region[:, None] < 0
    target_score = jnp.where(from_parking, 2100, 1800)

    wrong_in_target = (current_region >= 0) & (current_region != colors)
    parking_useful = parking_dest & empty & wrong_in_target[:, None]
    parking_neutral = parking_dest & empty & correct[:, None]

    scores = jnp.full((NUM_CARDS, 32), -1_000_000, dtype=jnp.int32)
    scores = jnp.where(movable & match_target, target_score, scores)
    scores = jnp.where(movable & parking_useful, 1000, scores)
    scores = jnp.where(movable & parking_neutral, 200, scores)

    flat_actions = action_ids.reshape((-1,))
    flat_scores = scores.reshape((-1,))
    return _select_scored_action(flat_actions, flat_scores, obs.legal_actions, _first_legal(obs.legal_actions))


def _hint_action(obs, colors):
    hint_ids = jnp.arange(7, dtype=jnp.int32)
    legal = obs.legal_actions
    reveal_actions = 1617 + hint_ids
    reveal_legal = legal[reveal_actions]
    first_reveal = reveal_actions[jnp.argmax(reveal_legal)].astype(jnp.int32)
    has_reveal = jnp.any(reveal_legal)

    pos = obs.positions.astype(jnp.int32)
    locks = obs.node_features[:NUM_CARDS, 10].astype(jnp.int32) > 0
    current_region = _target_region_color(pos)
    known = colors >= 0
    correct = known & (current_region == colors)
    hint_colors = obs.node_features[16:23, 4:8].astype(jnp.int32) > 0

    hint_idx = hint_ids[:, None]
    xy = pos[None, :, :]
    place_actions = (
        1624
        + hint_idx * 100
        + xy[:, :, 0] * 10
        + xy[:, :, 1]
    ).astype(jnp.int32)
    color_index = jnp.maximum(colors, 0)
    hint_matches = hint_colors[:, color_index] & (colors[None, :] >= 0)
    card_ok = (~locks) & known
    scores = jnp.full((7, NUM_CARDS), -1_000_000, dtype=jnp.int32)
    scores = jnp.maximum(
        scores,
        jnp.where(hint_matches & correct[None, :] & card_ok[None, :], 1200, -1_000_000),
    )
    scores = jnp.maximum(
        scores,
        jnp.where(correct[None, :] & card_ok[None, :], 900, -1_000_000),
    )
    scores = jnp.maximum(
        scores,
        jnp.where(hint_matches & card_ok[None, :], 100, -1_000_000),
    )
    place = _select_scored_action(
        place_actions.reshape((-1,)),
        scores.reshape((-1,)),
        legal,
        _first_legal(legal),
    )
    return jnp.where(has_reveal, first_reveal, place).astype(jnp.int32)


def _end_or_observe_action(obs, colors):
    pos = obs.positions.astype(jnp.int32)
    region = _target_region_color(pos)
    solved = jnp.all((colors >= 0) & (region == colors))
    connected = _all_colors_connected(obs, colors)
    can_end = obs.legal_actions[2324]
    observe = _observe_action(obs, colors)
    return jnp.where((solved | connected) & can_end, 2324, observe).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    colors, next_memory = _refresh_memory(obs, reset, memory)
    substep = obs.node_features[0, 13].astype(jnp.int32)
    active = obs.node_features[0, 12].astype(jnp.int32) > 0

    action = jax.lax.switch(
        substep,
        (
            lambda _: _end_or_observe_action(obs, colors),
            lambda _: _observe_action(obs, colors),
            lambda _: _move_action(obs, colors),
            lambda _: _hint_action(obs, colors),
        ),
        operand=None,
    )
    action = jnp.where(active, action, 0).astype(jnp.int32)
    trace = {
        "known_count": jnp.sum(colors >= 0).astype(jnp.int32),
        "chosen_action": action,
    }
    return action, next_memory, trace
