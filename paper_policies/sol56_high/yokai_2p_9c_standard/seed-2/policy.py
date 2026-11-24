"""Locally informed, convention-light policy for the two-player Yokai game."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'memory_adjacency',
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
    return {
        'colors': jnp.full((9,), -1, dtype=jnp.int8),
        'partner_seen': jnp.zeros((9,), dtype=jnp.bool_),
        'role': jnp.asarray(-1, dtype=jnp.int8),
    }


def _completed_colors(colors, positions):
    known = colors >= 0
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    complete = jnp.asarray(True)
    for color in range(3):
        members = known & (colors == color)
        edges = jnp.sum(adjacent & members[:, None] & members[None, :] & upper)
        complete = complete & (jnp.sum(members) == 3) & (edges >= 2)
    return complete


def _posterior(colors, positions, allowed):
    """Exact posterior under observations and trustworthy placed-hint constraints."""
    codes = jnp.arange(3 ** 9, dtype=jnp.int32)[:, None]
    powers = (3 ** jnp.arange(9, dtype=jnp.int32))[None, :]
    assignments = ((codes // powers) % 3).astype(jnp.int8)
    known = colors >= 0
    base_consistent = jnp.all(
        (~known)[None, :] | (assignments == colors[None, :]), axis=1
    )
    for color in range(3):
        base_consistent = base_consistent & (jnp.sum(assignments == color, axis=1) == 3)
    allowed_batch = jnp.broadcast_to(allowed[None, :, :], (3 ** 9, 9, 3))
    assignment_allowed = jnp.take_along_axis(
        allowed_batch, assignments[:, :, None].astype(jnp.int32), axis=2
    )[:, :, 0]
    constraint_score = jnp.sum(assignment_allowed, axis=1)
    best_score = jnp.max(jnp.where(base_consistent, constraint_score, -1))
    consistent = base_consistent & (constraint_score == best_score)

    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    solved = jnp.ones((3 ** 9,), dtype=jnp.bool_)
    for color in range(3):
        members = assignments == color
        edges = jnp.sum(
            members[:, :, None] & members[:, None, :] & adjacent[None, :, :] & upper[None, :, :],
            axis=(1, 2),
        )
        solved = solved & (edges >= 2)
    valid_count = jnp.maximum(jnp.sum(consistent), 1)
    probability = jnp.sum(consistent & solved).astype(jnp.float32) / valid_count
    marginals = jnp.stack(
        [
            jnp.sum(consistent[:, None] & (assignments == color), axis=0) / valid_count
            for color in range(3)
        ],
        axis=1,
    )
    return probability, marginals


def _move_action(obs, colors, allowed):
    positions = obs.positions
    move_index = jnp.arange(729, dtype=jnp.int32)
    card = move_index // 81
    cell = move_index % 81
    destination = jnp.stack((cell // 9, cell % 9), axis=1)
    card_axis = jnp.arange(9)[None, :, None]
    candidates = jnp.where(
        card_axis == card[:, None, None],
        destination[:, None, :],
        positions[None, :, :],
    )

    distance = jnp.sum(
        jnp.abs(candidates[:, :, None, :] - candidates[:, None, :, :]), axis=-1
    )
    known = colors >= 0
    same = known[:, None] & known[None, :] & (colors[:, None] == colors[None, :])
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    pairs = same & upper
    same_edges = jnp.sum((distance == 1) & pairs[None, :, :], axis=(1, 2))
    pair_distance = jnp.sum(jnp.minimum(distance, 8) * pairs[None, :, :], axis=(1, 2))
    triples = [
        (a, b, c)
        for a in range(9)
        for b in range(a + 1, 9)
        for c in range(b + 1, 9)
    ]
    triplets = jnp.asarray(
        [[card in triple for card in range(9)] for triple in triples], dtype=jnp.bool_
    )
    partitions = jnp.asarray(
        [
            (
                i,
                j,
                triples.index(
                    tuple(
                        card
                        for card in range(9)
                        if card not in first and card not in second
                    )
                ),
            )
            for i, first in enumerate(triples)
            for j, second in enumerate(triples)
            if all(card not in first for card in second)
        ],
        dtype=jnp.int32,
    )
    candidate_adjacent = distance == 1
    triplet_pairs = triplets[:, :, None] & triplets[:, None, :] & upper[None, :, :]
    triplet_edges = jnp.sum(
        candidate_adjacent[:, None, :, :] & triplet_pairs[None, :, :, :], axis=(2, 3)
    )
    triplet_connected = triplet_edges >= 2
    base_valid_by_color = jnp.zeros((3, 84), dtype=jnp.bool_)
    for color in range(3):
        known_here = known & (colors == color)
        known_elsewhere = known & (colors != color)
        base_valid_triplet = (
            jnp.all((~known_here)[None, :] | triplets, axis=1)
            & jnp.all((~known_elsewhere)[None, :] | (~triplets), axis=1)
        )
        base_valid_by_color = base_valid_by_color.at[color].set(base_valid_triplet)

    base_assignment_valid = jnp.all(
        jnp.stack(
            [base_valid_by_color[c, partitions[:, c]] for c in range(3)], axis=1
        ),
        axis=1,
    )
    partition_colors = jnp.sum(
        jnp.stack(
            [color * triplets[partitions[:, color]] for color in range(3)], axis=0
        ),
        axis=0,
    ).astype(jnp.int32)
    allowed_batch = jnp.broadcast_to(allowed[None, :, :], (partitions.shape[0], 9, 3))
    partition_allowed = jnp.take_along_axis(
        allowed_batch, partition_colors[:, :, None], axis=2
    )[:, :, 0]
    constraint_score = jnp.sum(partition_allowed, axis=1)
    best_score = jnp.max(
        jnp.where(base_assignment_valid, constraint_score, -1)
    )
    assignment_valid = base_assignment_valid & (constraint_score == best_score)
    denominator = jnp.maximum(jnp.sum(assignment_valid), 1)
    expected_completed = jnp.zeros((729,), dtype=jnp.float32)
    triplet_index = jnp.arange(84, dtype=jnp.int32)
    for color in range(3):
        weights = jnp.sum(
            assignment_valid[:, None]
            & (partitions[:, color, None] == triplet_index[None, :]),
            axis=0,
        )
        expected_completed = expected_completed + jnp.sum(
            triplet_connected * weights[None, :], axis=1
        ) / denominator

    # Unknown cards may embody useful work or information from the partner, so
    # disturb them only when legality leaves no known-card alternative.
    unknown_move = colors[card] < 0
    score = (
        1000.0 * expected_completed
        + 100.0 * same_edges
        - 3.0 * pair_distance
        - 40.0 * unknown_move
    )
    legal = obs.legal_actions[10:739]
    score = jnp.where(legal, score, -100000)
    return (10 + jnp.argmax(score)).astype(jnp.int32)


def _hint_action(obs, colors, marginals):
    features = obs.node_features
    hint_colors = features[9:13, 3:6] > 0
    revealed = jnp.any(hint_colors, axis=1)
    placed = features[9:13, 8] > 0
    can_reveal = (~revealed) & (~placed)
    reveal_scores = jnp.where(can_reveal, -jnp.arange(4), -100)
    reveal_action = (739 + jnp.argmax(reveal_scores)).astype(jnp.int32)

    place_index = jnp.arange(324, dtype=jnp.int32)
    hint = place_index // 81
    cell = place_index % 81
    xy = jnp.stack((cell // 9, cell % 9), axis=1)
    at_cell = jnp.all(obs.positions[None, :, :] == xy[:, None, :], axis=-1)
    target = jnp.argmax(at_cell, axis=1)
    target_known = colors[target] >= 0
    target_color = jnp.clip(colors[target].astype(jnp.int32), 0, 2)
    contains = hint_colors[hint, target_color]
    probability_correct = jnp.sum(
        marginals[target] * hint_colors[hint].astype(jnp.float32), axis=1
    )

    delta = jnp.abs(obs.positions[:, None, :] - obs.positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    same = (
        (colors[:, None] >= 0)
        & (colors[None, :] >= 0)
        & (colors[:, None] == colors[None, :])
    )
    settled_degree = jnp.sum(adjacent & same, axis=1)
    locked = features[:9, 8] > 0
    same_color_already_locked = jnp.any(
        locked[None, :] & (colors[None, :] == colors[target, None]), axis=1
    ) & target_known
    score = (
        240.0 * probability_correct
        - 120.0
        + 12 * settled_degree[target]
        + 5 * same_color_already_locked
        - hint
    )
    legal = obs.legal_actions[743:1067]
    score = jnp.where(legal, score, -100000)
    place_action = (743 + jnp.argmax(score)).astype(jnp.int32)
    return jnp.where(jnp.any(can_reveal), reveal_action, place_action)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.node_features
    active = features[0, 10] > 0
    substep = features[0, 11].astype(jnp.int32)

    colors = jnp.where(reset, jnp.full((9,), -1, dtype=jnp.int8), memory['colors'])
    partner_seen = jnp.where(reset, jnp.zeros((9,), dtype=jnp.bool_), memory['partner_seen'])
    role = jnp.where(
        reset,
        jnp.where(active, jnp.asarray(0, jnp.int8), jnp.asarray(1, jnp.int8)),
        memory['role'],
    )

    visible = jnp.sum(features[:9, 0:3], axis=1) > 0
    visible_color = jnp.argmax(features[:9, 0:3], axis=1).astype(jnp.int8)
    colors = jnp.where(visible, visible_color, colors)

    # Treat every placed hint, including a singleton, as public evidence in
    # the maximum-consistency posterior rather than irreversibly copying it
    # into private color memory.
    hint_colors = features[9:13, 3:6] > 0
    placed_links = (features[9:13, 8] > 0)[:, None] & (
        obs.adjacency_matrix[9:13, :9] > 0
    )
    has_hint_constraint = jnp.any(placed_links, axis=0)
    hint_allowed = jnp.any(
        placed_links[:, :, None] & hint_colors[:, None, :], axis=0
    )
    allowed = jnp.where(has_hint_constraint[:, None], hint_allowed, True)

    # With eight identities known, the fixed 3/3/3 deck determines the ninth.
    known_count = jnp.sum(colors >= 0)
    color_counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    missing_color = jnp.argmax(3 - color_counts).astype(jnp.int8)
    colors = jnp.where((known_count == 8) & (colors < 0), missing_color, colors)

    solution_probability, marginals = _posterior(colors, obs.positions, allowed)
    forced = jnp.sum(marginals > 0, axis=1) == 1
    forced_color = jnp.argmax(marginals, axis=1).astype(jnp.int8)
    colors = jnp.where((colors < 0) & forced, forced_color, colors)
    known_count = jnp.sum(colors >= 0)

    public_partner_seen = features[:9, 9] > 0
    partner_seen = partner_seen | public_partner_seen

    legal_observe = obs.legal_actions[1:10]
    unknown_legal = legal_observe & (colors < 0)
    repeat_partner = unknown_legal & partner_seen
    fresh = unknown_legal & (~partner_seen)
    uncertainty = 1.0 - jnp.max(marginals, axis=1)
    fresh_score = 1000.0 * uncertainty - 0.01 * jnp.arange(9)
    fresh_choice = jnp.argmax(jnp.where(fresh, fresh_score, -100000.0))
    use_repeat = (substep == 0) & jnp.any(repeat_partner)
    observe_card = jnp.where(
        use_repeat,
        jnp.argmax(repeat_partner),
        jnp.where(jnp.any(fresh), fresh_choice, jnp.argmax(unknown_legal)),
    )
    observe_card = jnp.where(jnp.any(unknown_legal), observe_card, jnp.argmax(legal_observe))
    observe_action = (1 + observe_card).astype(jnp.int32)

    solved = _completed_colors(colors, obs.positions)
    placed_count = jnp.sum(features[9:13, 8] > 0)
    all_hints_revealed = jnp.all(jnp.any(features[9:13, 3:6] > 0, axis=1))
    final_trust = (placed_count >= 3) & all_hints_revealed
    should_end = jnp.where(
        final_trust, solution_probability >= 0.15, solution_probability >= 0.75
    )
    first_action = jnp.where(should_end & obs.legal_actions[1067], 1067, observe_action)
    move_action = _move_action(obs, colors, allowed)
    hint_action = _hint_action(obs, colors, marginals)
    active_action = jnp.choose(
        substep,
        jnp.stack((first_action, observe_action, move_action, hint_action)),
        mode='clip',
    ).astype(jnp.int32)
    action = jnp.where(active, active_action, jnp.asarray(0, jnp.int32))

    finish_own_turn = active & (substep == 3)
    partner_seen = jnp.where(finish_own_turn, jnp.zeros((9,), dtype=jnp.bool_), partner_seen)
    next_memory = {'colors': colors, 'partner_seen': partner_seen, 'role': role}
    trace = {
        'colors': colors,
        'known_count': known_count.astype(jnp.int32),
        'locks': features[:9, 8],
        'positions': obs.positions,
        'role': role.astype(jnp.int32),
        'solution_probability': solution_probability,
        'solved': solved,
    }
    return action, next_memory, trace
