"""A memory-based, convention-independent Yokai policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'scan_cluster_end',
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
        -jnp.ones((9,), dtype=jnp.int32),
        jnp.zeros((9,), dtype=jnp.bool_),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _first_legal(mask, offset=0):
    """Select the lowest legal action in a fixed-size interval."""
    n = mask.shape[0]
    ids = jnp.arange(n, dtype=jnp.int32)
    score = jnp.where(mask, -ids, -1000000)
    return jnp.asarray(offset, dtype=jnp.int32) + jnp.argmax(score).astype(jnp.int32)


def _preferred_observation(mask, start):
    """Pick the first legal ID at/after start, wrapping only if needed."""
    ids = jnp.arange(9, dtype=jnp.int32)
    preferred = mask & (ids >= start)
    preferred_action = 1 + jnp.argmax(jnp.where(preferred, -ids, -1000000)).astype(jnp.int32)
    fallback_action = _first_legal(mask, 1)
    return jnp.where(jnp.any(preferred), preferred_action, fallback_action)


def _color_connected(colors, adjacency):
    """Check connectivity of all three color-induced card graphs."""
    same_color = colors[:, None] == colors[None, :]
    color_adjacency = adjacency.astype(jnp.int32) * same_color.astype(jnp.int32)
    reach = color_adjacency
    for _ in range(8):
        reach = jnp.where((reach @ color_adjacency) > 0, 1, reach)
    return jnp.all(jnp.where(same_color, reach > 0, True))


def _robust_color_connected(colors, known, adjacency):
    """Check connectivity for every valid completion of unknown cards."""
    assignments = jnp.arange(27, dtype=jnp.int32)
    digits = jnp.stack(
        (assignments // 9, (assignments // 3) % 3, assignments % 3), axis=1
    )
    unknown_rank = jnp.cumsum((~known).astype(jnp.int32)) - 1
    clipped_rank = jnp.clip(unknown_rank, 0, 2)
    filled_unknown = digits[:, clipped_rank]
    filled_colors = jnp.where(known[None, :], colors[None, :], filled_unknown)
    counts = jnp.sum(
        filled_colors[:, :, None] == jnp.arange(3, dtype=jnp.int32)[None, None, :],
        axis=1,
    )
    valid = jnp.all(counts == 3, axis=1)

    same_color = filled_colors[:, :, None] == filled_colors[:, None, :]
    color_adjacency = adjacency[None, :, :].astype(jnp.int32) * same_color.astype(jnp.int32)
    reach = color_adjacency
    for _ in range(8):
        reach = jnp.where((reach @ color_adjacency) > 0, 1, reach)
    connected = jnp.all(jnp.where(same_color, reach > 0, True), axis=(1, 2))
    return jnp.any(valid) & jnp.all(jnp.where(valid, connected, True))


def agent_act(obs, reset, memory, rng):
    old_colors, old_known, old_cursor = memory
    active = obs.node_features[0, 10] > 0
    colors0 = jnp.where(reset, -jnp.ones((9,), dtype=jnp.int32), old_colors)
    known0 = jnp.where(reset, jnp.zeros((9,), dtype=jnp.bool_), old_known)
    cursor = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), old_cursor)

    visible = jnp.sum(obs.node_features[:9, :3].astype(jnp.int32), axis=1) > 0
    visible_colors = jnp.argmax(obs.node_features[:9, :3], axis=1).astype(jnp.int32)
    colors = jnp.where(visible, visible_colors, colors0)
    known = known0 | visible

    # The reset distribution has exactly three cards of each color. Once
    # eight cards have been seen, the ninth color is forced by that invariant.
    counts = jnp.sum(colors[:, None] == jnp.arange(3, dtype=jnp.int32)[None, :], axis=0)
    missing = ~known
    inferred_color = jnp.argmin(counts).astype(jnp.int32)
    complete_colors = jnp.where(
        missing & (jnp.sum(known.astype(jnp.int32)) >= 8),
        inferred_color,
        colors,
    )
    complete = jnp.sum(known.astype(jnp.int32)) >= 8
    connected = complete & _color_connected(
        complete_colors, obs.adjacency_matrix[:9, :9]
    )
    robust_connected = _robust_color_connected(
        colors, known, obs.adjacency_matrix[:9, :9]
    )

    substep = obs.node_features[0, 11].astype(jnp.int32)

    # Scan two fresh stable IDs per active turn. The fourth active turn sees
    # IDs 6 and 7; ID 8 is then inferred from the fixed deck multiplicity.
    first_id = jnp.minimum(cursor, jnp.asarray(8, dtype=jnp.int32))
    second_id = jnp.minimum(cursor + 1, jnp.asarray(8, dtype=jnp.int32))
    observe0 = _preferred_observation(obs.legal_actions[1:10], first_id)
    observe1 = _preferred_observation(obs.legal_actions[1:10], second_id)

    # If a full color map is available, prefer moves that put a card into a
    # canonical color column. Otherwise make the shortest legal geometry
    # change, which keeps the public board compact while information accrues.
    ids = jnp.arange(9, dtype=jnp.int32)
    rank = jnp.sum(
        (ids[None, :] < ids[:, None])
        & (complete_colors[:, None] == complete_colors[None, :]),
        axis=1,
    )
    target_x = 3 + complete_colors
    target_y = 3 + rank
    move_ids = jnp.arange(729, dtype=jnp.int32)
    move_card = move_ids // 81
    move_cell = move_ids % 81
    move_x = move_cell // 9
    move_y = move_cell % 9
    current_x = obs.positions[move_card, 0]
    current_y = obs.positions[move_card, 1]
    distance = jnp.abs(move_x - current_x) + jnp.abs(move_y - current_y)
    reaches_target = (move_x == target_x[move_card]) & (move_y == target_y[move_card])
    legal_moves = obs.legal_actions[10:739]
    # Test every legal destination against the actual color-induced graph.
    # This lets the final informed move repair a board without requiring a
    # multi-turn private agreement about temporary target positions.
    source_mask = move_card[:, None] == ids[None, :]
    new_x = jnp.where(source_mask, move_x[:, None], obs.positions[None, :, 0])
    new_y = jnp.where(source_mask, move_y[:, None], obs.positions[None, :, 1])
    delta_x = new_x[:, :, None] - new_x[:, None, :]
    delta_y = new_y[:, :, None] - new_y[:, None, :]
    new_adjacency = (
        ((jnp.abs(delta_x) + jnp.abs(delta_y)) == 1)
        | jnp.eye(9, dtype=jnp.bool_)[None, :, :]
    ).astype(jnp.int32)
    color_adjacency = new_adjacency * (
        complete_colors[None, :, None] == complete_colors[None, None, :]
    ).astype(jnp.int32)
    new_reach = color_adjacency
    for _ in range(8):
        new_reach = jnp.where(
            (new_reach @ color_adjacency) > 0, 1, new_reach
        )
    candidate_connected = jnp.all(
        jnp.where(
            complete_colors[None, :, None] == complete_colors[None, None, :],
            new_reach > 0,
            True,
        ),
        axis=(1, 2),
    )
    adjacency_density = jnp.sum(new_adjacency, axis=(1, 2)).astype(jnp.int32)
    move_score = jnp.where(
        candidate_connected & complete,
        jnp.asarray(100000, dtype=jnp.int32),
        jnp.where(
            reaches_target & complete,
            jnp.asarray(10000, dtype=jnp.int32),
            jnp.where(~complete, adjacency_density * 1000 - distance, -distance),
        ),
    )
    move_score = jnp.where(legal_moves, move_score, -1000000)
    move_action = 10 + jnp.argmax(move_score).astype(jnp.int32)

    # Substep 3 must resolve a hint. Reveal the first still-hidden hint so the
    # policy never relies on a private hint convention.
    hint_revealed = jnp.sum(obs.node_features[9:13, 3:6].astype(jnp.int32), axis=1) > 0
    reveal_mask = obs.legal_actions[739:743] & ~hint_revealed
    reveal_action = _first_legal(reveal_mask, 739)
    any_reveal = jnp.any(reveal_mask)
    placement_mask = obs.legal_actions[743:1067]
    place_ids = jnp.arange(324, dtype=jnp.int32)
    place_hint = place_ids // 81
    place_cell = place_ids % 81
    place_x = place_cell // 9
    place_y = place_cell % 9
    at_card = (
        (obs.positions[:, 0][None, :] == place_x[:, None])
        & (obs.positions[:, 1][None, :] == place_y[:, None])
    )
    place_card = jnp.argmax(at_card.astype(jnp.int32), axis=1)
    place_known = known[place_card]
    place_color = colors[place_card]
    place_hint_set = obs.node_features[9 + place_hint, 3:6]
    place_correct = place_known & place_hint_set[jnp.arange(324), place_color]
    place_score = jnp.where(
        placement_mask & place_correct,
        jnp.asarray(100000, dtype=jnp.int32),
        jnp.where(placement_mask, 0, -1000000),
    )
    placement_action = 743 + jnp.argmax(place_score).astype(jnp.int32)
    fallback_action = _first_legal(obs.legal_actions, 0)

    active_action = jnp.where(
        substep == 0,
        jnp.where(
            (connected | robust_connected) & obs.legal_actions[1067],
            jnp.asarray(1067, dtype=jnp.int32),
            observe0,
        ),
        jnp.where(
            substep == 1,
            observe1,
            jnp.where(
                substep == 2,
                move_action,
                jnp.where(any_reveal, reveal_action, placement_action),
            ),
        ),
    )
    action = jnp.where(active, active_action, jnp.asarray(0, dtype=jnp.int32))
    next_cursor = jnp.where(active & (substep == 3), jnp.minimum(cursor + 2, 8), cursor)
    next_memory = (colors, known, next_cursor)
    return action.astype(jnp.int32), next_memory, {}
