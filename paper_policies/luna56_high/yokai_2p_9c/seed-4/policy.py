"""A color-aware, convention-light Yokai policy."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'color_cluster_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.concatenate((-
        jnp.ones((9,), dtype=jnp.int32),
        jnp.asarray([-1], dtype=jnp.int32),
    ))


def _remember(obs, memory, reset):
    old = jnp.where(reset, init_agent(), memory)
    old_colors = old[:9]
    colors = obs.node_features[:9, :3]
    visible = jnp.any(colors != 0, axis=1)
    seen = jnp.argmax(colors, axis=1).astype(jnp.int32)
    updated = jnp.where(visible, seen, old_colors)
    counts = jnp.asarray([jnp.sum(updated == color) for color in range(3)])
    remaining = 3 - counts
    only_one_color_left = jnp.sum(remaining > 0) == 1
    inferred = jnp.argmax(remaining).astype(jnp.int32)
    updated = jnp.where((updated < 0) & only_one_color_left, inferred, updated)
    return jnp.concatenate((updated, old[9:10]))


def _is_solved(positions, colors):
    known = colors >= 0
    grid = positions[:, None, :] - positions[None, :, :]
    adjacent = jnp.sum(jnp.abs(grid), axis=-1) == 1
    eye = jnp.eye(9, dtype=bool)
    result = jnp.asarray(True)
    for color in range(3):
        pair = known[:, None] & known[None, :] & (colors[:, None] == color) & (colors[None, :] == color)
        reach = (adjacent | eye) & pair

        def close(_, value):
            return jnp.logical_or(value, jnp.any(value[:, :, None] & value[None, :, :], axis=1))

        reach = jax.lax.fori_loop(0, 8, close, reach)
        result = result & jnp.all(jnp.where(pair, reach, True))
    return jnp.all(known) & result


def _observe_action(obs, colors):
    ids = jnp.arange(9, dtype=jnp.int32)
    unlocked = obs.node_features[:9, 8] == 0
    partner_seen = obs.node_features[:9, 9] != 0
    unknown = colors < 0
    score = 1000 * (unknown & unlocked).astype(jnp.int32)
    score = score + 100 * ((~partner_seen) & unlocked).astype(jnp.int32)
    score = score - ids
    actions = ids + 1
    values = jnp.where(obs.legal_actions[1:10], score, -1000000)
    return actions[jnp.argmax(values)]


def _move_action(obs, colors, avoid_card):
    ids = jnp.arange(729, dtype=jnp.int32)
    card = ids // 81
    cell = ids % 81
    dest = jnp.stack((cell // 9, cell % 9), axis=1)
    positions = obs.positions
    delta = dest[:, None, :] - positions[None, :, :]
    neighbors = jnp.sum(jnp.abs(delta), axis=-1) == 1
    known_card = colors[card] >= 0
    same = known_card[:, None] & (colors[card, None] == colors[None, :]) & neighbors
    same_count = jnp.sum(same, axis=1).astype(jnp.int32)
    all_count = jnp.sum(neighbors, axis=1).astype(jnp.int32)
    target = known_card & (dest[:, 0] == (3 + colors[card])) & (dest[:, 1] >= 3) & (dest[:, 1] <= 5)
    current = positions[card]
    settled = known_card & (current[:, 0] == (3 + colors[card])) & (current[:, 1] >= 3) & (current[:, 1] <= 5)
    score = 1000 * known_card.astype(jnp.int32) + 100 * same_count + all_count
    score = score + 500 * target.astype(jnp.int32) - 500 * settled.astype(jnp.int32)
    score = score - 5000 * (card == avoid_card).astype(jnp.int32)
    actions = ids + 10
    values = jnp.where(obs.legal_actions[10:739], score, -1000000)
    return actions[jnp.argmax(values)]


def _hint_action(obs, colors):
    reveal_legal = obs.legal_actions[739:743]
    reveal_score = jnp.where(reveal_legal, 1000 - jnp.arange(4, dtype=jnp.int32), -1000000)
    reveal_action = 739 + jnp.argmax(reveal_score)

    ids = jnp.arange(324, dtype=jnp.int32)
    hint = ids // 81
    cell = ids % 81
    dest = jnp.stack((cell // 9, cell % 9), axis=1)
    at_dest = jnp.all(obs.positions[:, None, :] == dest[None, :, :], axis=-1)
    card = jnp.argmax(at_dest, axis=0)
    card_color = colors[card]
    hint_colors = obs.node_features[9 + hint, 3:6]
    correct = (card_color >= 0) & (hint_colors[jnp.arange(324), card_color] != 0)
    place_score = 1000 * correct.astype(jnp.int32) - hint
    place_actions = ids + 743
    place_values = jnp.where(obs.legal_actions[743:1067], place_score, -1000000)
    place_action = place_actions[jnp.argmax(place_values)]
    return jnp.where(jnp.any(reveal_legal), reveal_action, place_action)


def agent_act(obs, reset, memory, rng):
    remembered = _remember(obs, memory, reset)
    colors = remembered[:9]
    last_moved = remembered[9]
    active = obs.node_features[0, 10] != 0
    substep = obs.node_features[0, 11]
    observe = _observe_action(obs, colors)
    move = _move_action(obs, colors, last_moved)
    hint = _hint_action(obs, colors)
    solved = _is_solved(obs.positions, colors)
    end = jnp.asarray(1067, dtype=jnp.int32)
    action = jnp.where(substep == 0, jnp.where(solved, end, observe), observe)
    action = jnp.where(substep == 2, move, action)
    action = jnp.where(substep == 3, hint, action)
    move_card = (move - 10) // 81
    next_last = jnp.where(active & (substep == 2), move_card, remembered[9])
    next_memory = jnp.concatenate((colors, next_last[None]))
    next_memory = jnp.where(reset, init_agent(), next_memory)
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32))
    return action.astype(jnp.int32), next_memory, {}
