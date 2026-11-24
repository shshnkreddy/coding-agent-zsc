"""Color-aware, fixed-shape policy for the two-player Yokai board."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'color_lane_greedy_v1',
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


def init_agent():
    return (
        jnp.zeros((9, 3), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _first_legal(mask):
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _connected_by_color(adj, colors):
    reach = adj[:9, :9].astype(jnp.int32)
    eye = jnp.eye(9, dtype=jnp.int32)
    reach = jnp.maximum(reach, eye)
    # Fixed-shape transitive closure for nine nodes.
    for _ in range(3):
        reach = jnp.matmul(reach, reach) > 0
        reach = reach.astype(jnp.int32)
    same = colors[:, None] == colors[None, :]
    present = jnp.sum(same.astype(jnp.int32), axis=1) > 0
    good = jnp.all(jnp.where(same, reach, True), axis=1)
    return jnp.all(good) & jnp.all(present)


def _permutation_table():
    """Build the fixed 8! lexicographic permutation table with JAX ops."""
    ranks = jnp.arange(40320, dtype=jnp.int32)
    values = jnp.arange(1, 9, dtype=jnp.int32)
    factorial = jnp.asarray([5040, 720, 120, 24, 6, 2, 1, 1], dtype=jnp.int32)
    used = jnp.zeros((40320, 8), dtype=jnp.bool_)
    result = jnp.zeros((40320, 8), dtype=jnp.int32)
    for column in range(8):
        available = ~used
        ordinal = jnp.cumsum(available.astype(jnp.int32), axis=1) - 1
        digit = (ranks // factorial[column]) % (8 - column)
        choose = available & (ordinal == digit[:, None])
        chosen = jnp.sum(jnp.where(choose, values[None, :], 0), axis=1)
        result = result.at[:, column].set(chosen)
        used = used | choose
    return result


def _move_choice(obs, known, global_turn):
    positions = obs.positions.astype(jnp.int32)
    labels = jnp.argmax(known, axis=1).astype(jnp.int32)
    confidence = jnp.sum(known, axis=1) > 0
    missing_count = jnp.sum((~confidence).astype(jnp.int32))
    remaining = 3 - jnp.sum(known, axis=0)
    missing_label = jnp.argmax(remaining).astype(jnp.int32)
    inferred = jnp.where(
        confidence[:, None],
        known,
        jnp.eye(3, dtype=jnp.int32)[missing_label],
    )
    labels = jnp.argmax(inferred, axis=1).astype(jnp.int32)
    confidence = confidence | (missing_count <= 1)

    # A 3x3 square in the initially empty upper-left area, with card 0 kept
    # at its lower-right root. The order is a fixed connected growth path.
    skeleton = jnp.asarray(
        [[2, 3], [3, 2], [1, 3], [2, 2], [3, 1], [1, 2], [2, 1], [1, 1]],
        dtype=jnp.int32,
    )
    legal = obs.legal_actions[10:739]
    final_positions = jnp.concatenate((jnp.asarray([[3, 3]], dtype=jnp.int32), skeleton), axis=0)
    delta = final_positions[:, None, :] - final_positions[None, :, :]
    reach = (jnp.sum(jnp.abs(delta), axis=-1) == 1).astype(jnp.int32)
    reach = jnp.maximum(reach, jnp.eye(9, dtype=jnp.int32))
    for _ in range(3):
        reach = (jnp.matmul(reach, reach) > 0).astype(jnp.int32)

    perm = _permutation_table()
    # Enumerate the 60 colorings of the 3x3 square whose three color groups
    # are connected. The best permutation is the one compatible with the
    # greatest number of such completions of the currently known colors.
    valid_colorings = jnp.asarray([
        [0, 0, 0, 1, 1, 2, 1, 2, 2],
        [0, 0, 0, 1, 2, 2, 1, 2, 1],
        [0, 0, 0, 2, 1, 1, 2, 1, 2],
        [0, 0, 0, 2, 2, 1, 2, 1, 1],
        [0, 0, 1, 0, 1, 1, 2, 2, 2],
        [0, 0, 1, 0, 1, 2, 1, 2, 2],
        [0, 0, 1, 0, 2, 1, 2, 1, 2],
        [0, 0, 1, 2, 0, 1, 2, 1, 2],
        [0, 0, 2, 0, 1, 2, 1, 2, 1],
        [0, 0, 2, 0, 2, 1, 2, 1, 1],
        [0, 0, 2, 0, 2, 2, 1, 1, 1],
        [0, 0, 2, 1, 0, 2, 1, 2, 1],
        [0, 1, 0, 1, 0, 2, 1, 2, 2],
        [0, 1, 0, 1, 1, 0, 2, 2, 2],
        [0, 1, 0, 1, 2, 0, 1, 2, 2],
        [0, 1, 0, 2, 1, 0, 2, 1, 2],
        [0, 2, 0, 1, 2, 0, 1, 2, 1],
        [0, 2, 0, 2, 0, 1, 2, 1, 1],
        [0, 2, 0, 2, 1, 0, 2, 1, 1],
        [0, 2, 0, 2, 2, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 2, 2, 2],
        [1, 0, 1, 0, 1, 2, 0, 2, 2],
        [1, 0, 1, 0, 2, 1, 0, 2, 2],
        [1, 0, 1, 2, 0, 1, 2, 0, 2],
        [1, 1, 0, 1, 0, 0, 2, 2, 2],
        [1, 1, 0, 1, 0, 2, 0, 2, 2],
        [1, 1, 0, 1, 2, 0, 2, 0, 2],
        [1, 1, 0, 2, 1, 0, 2, 0, 2],
        [1, 1, 1, 0, 0, 2, 0, 2, 2],
        [1, 1, 1, 0, 2, 2, 0, 2, 0],
        [1, 1, 1, 2, 0, 0, 2, 0, 2],
        [1, 1, 1, 2, 2, 0, 2, 0, 0],
        [1, 1, 2, 0, 1, 2, 0, 2, 0],
        [1, 1, 2, 1, 0, 2, 0, 2, 0],
        [1, 1, 2, 1, 2, 0, 2, 0, 0],
        [1, 1, 2, 1, 2, 2, 0, 0, 0],
        [1, 2, 1, 0, 2, 1, 0, 2, 0],
        [1, 2, 1, 2, 0, 1, 2, 0, 0],
        [1, 2, 1, 2, 1, 0, 2, 0, 0],
        [1, 2, 1, 2, 2, 1, 0, 0, 0],
        [2, 0, 2, 0, 0, 2, 1, 1, 1],
        [2, 0, 2, 0, 1, 2, 0, 1, 1],
        [2, 0, 2, 0, 2, 1, 0, 1, 1],
        [2, 0, 2, 1, 0, 2, 1, 0, 1],
        [2, 1, 2, 0, 1, 2, 0, 1, 0],
        [2, 1, 2, 1, 0, 2, 1, 0, 0],
        [2, 1, 2, 1, 1, 2, 0, 0, 0],
        [2, 1, 2, 1, 2, 0, 1, 0, 0],
        [2, 2, 0, 1, 2, 0, 1, 0, 1],
        [2, 2, 0, 2, 0, 0, 1, 1, 1],
        [2, 2, 0, 2, 0, 1, 0, 1, 1],
        [2, 2, 0, 2, 1, 0, 1, 0, 1],
        [2, 2, 1, 0, 2, 1, 0, 1, 0],
        [2, 2, 1, 2, 0, 1, 0, 1, 0],
        [2, 2, 1, 2, 1, 0, 1, 0, 0],
        [2, 2, 1, 2, 1, 1, 0, 0, 0],
        [2, 2, 2, 0, 0, 1, 0, 1, 1],
        [2, 2, 2, 0, 1, 1, 0, 1, 0],
        [2, 2, 2, 1, 0, 0, 1, 0, 1],
        [2, 2, 2, 1, 1, 0, 1, 0, 0],
    ], dtype=jnp.int32)
    cards_at_position = jnp.concatenate((
        jnp.zeros((perm.shape[0], 1), dtype=jnp.int32),
        perm,
    ), axis=1)
    card_ids = jnp.arange(9, dtype=jnp.int32)
    positions_of_card = jnp.argmax(
        (cards_at_position[:, :, None] == card_ids[None, None, :]).astype(jnp.int32),
        axis=1,
    )
    colors_at_card = jnp.take_along_axis(
        jnp.broadcast_to(valid_colorings[None, :, :], (perm.shape[0], 60, 9)),
        positions_of_card[:, None, :],
        axis=2,
    )
    compatible = jnp.all(
        jnp.where(confidence[None, None, :], colors_at_card == labels[None, None, :], True),
        axis=2,
    )
    score = jnp.sum(compatible.astype(jnp.int32), axis=1)

    # Publicly recover which card already occupies each filled skeleton slot.
    occupies = jnp.all(positions[:, None, :] == skeleton[None, :, :], axis=2)
    actual_prefix = jnp.argmax(occupies.astype(jnp.int32), axis=0)
    slot_index = jnp.arange(8, dtype=jnp.int32)
    prefix_ok = jnp.all(
        jnp.where(slot_index[None, :] < global_turn, perm == actual_prefix[None, :], True),
        axis=1,
    )

    g = jnp.minimum(global_turn, 7)
    target = skeleton[g]
    target_action = 10 + perm[:, g] * 81 + target[0] * 9 + target[1]
    current_legal = obs.legal_actions[target_action]
    valid = prefix_ok & current_legal & (global_turn < 7)
    score = jnp.where(valid, score, jnp.asarray(-100000, dtype=jnp.int32))
    chosen = jnp.argmax(score)
    planned_action = target_action[chosen]
    fallback = _first_legal(obs.legal_actions[10:739]) + 10
    return jnp.where(jnp.any(valid), planned_action, fallback).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    known0, seat0, own_turn0, global_turn0 = memory
    nf = obs.node_features.astype(jnp.int32)
    current = nf[:9, :3]
    has_current = jnp.sum(current, axis=1) > 0
    known = jnp.where(has_current[:, None], current, known0)

    active = nf[0, 10] > 0
    substep = nf[0, 11]
    seat = jnp.where(reset, 1 - nf[0, 10], seat0)
    own_turn = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), own_turn0)
    global_turn = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), global_turn0)

    # Both seats use the same public observation order and therefore do not
    # require a private role convention to learn the deck.
    first_id = jnp.minimum(8, own_turn * 2)
    second_id = jnp.where(own_turn < 4, own_turn * 2 + 1, 0)
    observe_first = 1 + first_id
    observe_second = 1 + second_id

    legal = obs.legal_actions
    fallback = _first_legal(legal)
    action0 = jnp.where(legal[observe_first], observe_first, fallback)
    action1 = jnp.where(legal[observe_second], observe_second, fallback)
    move_action = _move_choice(obs, known, global_turn)

    confidence = jnp.sum(known, axis=1) > 0
    missing_count = jnp.sum((~confidence).astype(jnp.int32))
    remaining = 3 - jnp.sum(known, axis=0)
    missing_label = jnp.argmax(remaining).astype(jnp.int32)
    inferred = jnp.where(
        confidence[:, None],
        known,
        jnp.eye(3, dtype=jnp.int32)[missing_label],
    )
    inferred_labels = jnp.argmax(inferred, axis=1)
    fully_known = missing_count <= 1
    solved = fully_known & _connected_by_color(obs.adjacency_matrix, inferred_labels)
    end_action = jnp.where(legal[1067] & solved, jnp.asarray(1067, dtype=jnp.int32), fallback)

    revealed = jnp.sum(nf[9:13, 3:6], axis=1) > 0
    reveal_ids = 739 + jnp.arange(4, dtype=jnp.int32)
    reveal_mask = legal[reveal_ids] & (~revealed)
    reveal_action = jnp.where(jnp.any(reveal_mask), _first_legal(reveal_mask) + 739, fallback)

    # Once all hints are public, place each on a correct card that is already
    # in the filled prefix. Such a card will never be selected for a later
    # move, so hint locking cannot invalidate the layout plan.
    filled_slots = jnp.arange(8, dtype=jnp.int32) <= jnp.minimum(global_turn, 7)
    in_filled_square = jnp.any(
        (obs.positions[:, None, :] == jnp.asarray(
            [[2, 3], [3, 2], [1, 3], [2, 2], [3, 1], [1, 2], [2, 1], [1, 1]],
            dtype=jnp.int32,
        )[None, :, :]).all(axis=2) & filled_slots[None, :],
        axis=1,
    ) | jnp.all(obs.positions == jnp.asarray([3, 3], dtype=jnp.int32), axis=1)
    hint_sets = nf[9:13, 3:6] > 0
    place_hints = jnp.repeat(jnp.arange(4, dtype=jnp.int32), 9)
    place_cards = jnp.tile(jnp.arange(9, dtype=jnp.int32), 4)
    place_x = obs.positions[place_cards, 0]
    place_y = obs.positions[place_cards, 1]
    place_ids = 743 + place_hints * 81 + place_x * 9 + place_y
    place_legal = legal[place_ids]
    place_correct = hint_sets[place_hints, inferred_labels[place_cards]]
    place_safe = in_filled_square[place_cards]
    place_unused = nf[9 + place_hints, 8] == 0
    place_score = (
        place_correct.astype(jnp.int32) * 100
        + place_safe.astype(jnp.int32) * 10
        - place_cards
    )
    place_valid = place_legal & place_safe & place_unused
    place_score = jnp.where(place_valid, place_score, jnp.asarray(-100000, dtype=jnp.int32))
    place_action = jnp.where(
        jnp.any(place_valid),
        place_ids[jnp.argmax(place_score)],
        fallback,
    )

    action = jnp.where(substep == 0, end_action, action0)
    action = jnp.where(substep == 1, action1, action)
    action = jnp.where(substep == 2, move_action, action)
    action = jnp.where(substep == 3, jnp.where(jnp.any(reveal_mask), reveal_action, place_action), action)
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32))

    next_own_turn = jnp.where((~reset) & active & (substep == 3), own_turn + 1, own_turn)
    next_global_turn = jnp.where((~reset) & (substep == 3), global_turn + 1, global_turn)
    next_memory = (known, seat, next_own_turn, next_global_turn)
    return action.astype(jnp.int32), next_memory, {}
