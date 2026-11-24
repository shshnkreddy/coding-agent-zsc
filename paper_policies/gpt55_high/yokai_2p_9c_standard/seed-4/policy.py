"""Convention-light Yokai policy for cooperative color clustering."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'zone_cluster_reveal_safe',
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

NUM_CARDS = 9
NUM_HINTS = 4
NUM_ACTIONS = 1068
UNKNOWN = -1


def init_agent():
    return {
        'known_colors': jnp.full((NUM_CARDS,), UNKNOWN, dtype=jnp.int32),
    }


def _reset_memory(memory, reset):
    base = init_agent()
    return {
        'known_colors': jnp.where(reset, base['known_colors'], memory['known_colors']),
    }


def _update_known(obs, known):
    color_bits = obs.node_features[:NUM_CARDS, 0:3].astype(jnp.int32)
    visible = jnp.sum(color_bits, axis=1) > 0
    colors = jnp.argmax(color_bits, axis=1).astype(jnp.int32)
    known = jnp.where(visible, colors, known)

    counts = jnp.asarray([jnp.sum((known == c).astype(jnp.int32)) for c in range(3)])
    remaining = 3 - counts
    only_color_left = jnp.sum((remaining > 0).astype(jnp.int32)) == 1
    missing_color = jnp.argmax(remaining).astype(jnp.int32)
    inferred = jnp.where(known < 0, missing_color, known)
    return jnp.where(only_color_left, inferred, known)


def _first_legal(mask):
    idx = jnp.arange(mask.shape[0], dtype=jnp.int32)
    scores = jnp.where(mask, 100000 - idx, -1000000)
    return jnp.argmax(scores).astype(jnp.int32)


def _choose_observe(obs, known):
    legal = obs.legal_actions[1:10]
    ids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    unknown = known < 0
    scores = jnp.where(legal, unknown.astype(jnp.int32) * 100 - ids, -1000000)
    return (1 + jnp.argmax(scores)).astype(jnp.int32)


def _color_connected(obs, known, color):
    members = known == color
    count = jnp.sum(members.astype(jnp.int32))
    adj = obs.adjacency_matrix[:NUM_CARDS, :NUM_CARDS].astype(jnp.int32)
    pair_mask = members[:, None] & members[None, :]
    off_diag = 1 - jnp.eye(NUM_CARDS, dtype=jnp.int32)
    directed_edges = jnp.sum(adj * pair_mask.astype(jnp.int32) * off_diag)
    return (count == 3) & (directed_edges >= 4)


def _all_known_connected(obs, known):
    all_known = jnp.all(known >= 0)
    c0 = _color_connected(obs, known, 0)
    c1 = _color_connected(obs, known, 1)
    c2 = _color_connected(obs, known, 2)
    return all_known & c0 & c1 & c2


def _zone_distance(pos, color):
    # Three vertical color lanes inside the original 3x3 area.
    target_x = color + 3
    y = pos[1]
    dy = jnp.maximum(jnp.asarray(0, dtype=jnp.int32), jnp.abs(y - 4) - 1)
    return jnp.abs(pos[0] - target_x) + dy


def _board_cost(positions, known):
    costs = jnp.asarray([
        jnp.where(known[i] >= 0, _zone_distance(positions[i], known[i]), 0)
        for i in range(NUM_CARDS)
    ], dtype=jnp.int32)
    return jnp.sum(costs)


def _position_same_color_edges(positions, known):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=2) == 1
    same_known = (known[:, None] >= 0) & (known[:, None] == known[None, :])
    off_diag = 1 - jnp.eye(NUM_CARDS, dtype=jnp.int32)
    return jnp.sum(adjacent.astype(jnp.int32) * same_known.astype(jnp.int32) * off_diag)


def _position_complete_groups(positions, known):
    c0 = _position_color_complete(positions, known, 0)
    c1 = _position_color_complete(positions, known, 1)
    c2 = _position_color_complete(positions, known, 2)
    return c0.astype(jnp.int32) + c1.astype(jnp.int32) + c2.astype(jnp.int32)


def _position_color_complete(positions, known, color):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=2) == 1
    off_diag = 1 - jnp.eye(NUM_CARDS, dtype=jnp.int32)
    members = known == color
    count = jnp.sum(members.astype(jnp.int32))
    pair_mask = members[:, None] & members[None, :]
    directed_edges = jnp.sum(adjacent.astype(jnp.int32) * pair_mask.astype(jnp.int32) * off_diag)
    return (count == 3) & (directed_edges >= 4)


def _move_score(obs, known, action):
    rel = action - 10
    card = rel // 81
    cell = rel - card * 81
    x = cell // 9
    y = cell - x * 9
    positions = obs.positions.astype(jnp.int32)
    dest = jnp.asarray([x, y], dtype=jnp.int32)
    moved_positions = positions.at[card].set(dest)
    cost = _board_cost(moved_positions, known)
    same_edges = _position_same_color_edges(moved_positions, known)
    complete_groups = _position_complete_groups(moved_positions, known)

    own_known = known[card] >= 0
    card_gain = jnp.where(
        own_known,
        _zone_distance(positions[card], known[card]) - _zone_distance(dest, known[card]),
        0,
    )
    central = -(jnp.abs(x - 4) + jnp.abs(y - 4))
    return 1000 * complete_groups + 100 * same_edges - 10 * cost + 10 * card_gain + central


def _choose_move(obs, known):
    actions = jnp.arange(10, 739, dtype=jnp.int32)
    legal = obs.legal_actions[10:739]
    scores = jax.vmap(lambda a: _move_score(obs, known, a))(actions)
    scores = jnp.where(legal, scores, -100000000)
    return (10 + jnp.argmax(scores)).astype(jnp.int32)


def _choose_hint(obs, known):
    reveal_legal = obs.legal_actions[739:743]
    reveal_action = (739 + _first_legal(reveal_legal)).astype(jnp.int32)

    actions = jnp.arange(743, 1067, dtype=jnp.int32)
    legal = obs.legal_actions[743:1067]
    rel = actions - 743
    hint = rel // 81
    cell = rel - hint * 81
    x = cell // 9
    y = cell - x * 9
    positions = obs.positions.astype(jnp.int32)
    at_cell = (positions[:, 0][None, :] == x[:, None]) & (positions[:, 1][None, :] == y[:, None])
    card = jnp.argmax(at_cell.astype(jnp.int32), axis=1)
    card_known = known[card] >= 0
    card_color = jnp.maximum(known[card], 0)

    hint_sets = obs.node_features[9:13, 3:6].astype(jnp.int32)
    compatible = hint_sets[hint, card_color] > 0
    dist = jax.vmap(lambda p, c: _zone_distance(p, c))(positions[card], card_color)
    delta = jnp.abs(positions[card][:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=2) == 1
    same_color_neighbor = (known[None, :] == card_color[:, None]) & (known[None, :] >= 0)
    not_self = jnp.arange(NUM_CARDS, dtype=jnp.int32)[None, :] != card[:, None]
    same_degree = jnp.sum((adjacent & same_color_neighbor & not_self).astype(jnp.int32), axis=1)
    complete_flags = jnp.asarray([
        _position_color_complete(positions, known, c) for c in range(3)
    ], dtype=jnp.bool_)
    in_complete_group = complete_flags[card_color] & card_known
    scores = (
        compatible.astype(jnp.int32) * 10000
        + in_complete_group.astype(jnp.int32) * 1000
        + same_degree * 100
        + card_known.astype(jnp.int32) * 50
        - dist * 5
        - card
    )
    place_action = (743 + jnp.argmax(jnp.where(legal, scores, -100000000))).astype(jnp.int32)
    safe_place = jnp.any(legal & compatible & card_known)
    preserve_value = _all_known_connected(obs, known) & safe_place
    return jnp.where(jnp.any(reveal_legal) & ~preserve_value, reveal_action, place_action)


def _active_action(obs, known):
    substep = obs.node_features[0, 11].astype(jnp.int32)

    def step0(_):
        end_now = _all_known_connected(obs, known) & obs.legal_actions[1067]
        return jnp.where(end_now, jnp.asarray(1067, dtype=jnp.int32), _choose_observe(obs, known))

    return jax.lax.switch(
        substep,
        (step0, lambda _: _choose_observe(obs, known), lambda _: _choose_move(obs, known), lambda _: _choose_hint(obs, known)),
        operand=None,
    )


def agent_act(obs, reset, memory, rng):
    memory = _reset_memory(memory, reset)
    known = _update_known(obs, memory['known_colors'])
    active = obs.node_features[0, 10] == 1
    proposed = _active_action(obs, known)
    action = jnp.where(active, proposed, jnp.asarray(0, dtype=jnp.int32))
    next_memory = {'known_colors': known}
    trace = {
        'known_count': jnp.sum((known >= 0).astype(jnp.int32)),
        'would_end': _all_known_connected(obs, known),
        'proposed_action': proposed,
    }
    return action.astype(jnp.int32), next_memory, trace
