"""Local, convention-light policy for the two-player 9-card Yokai game."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "bare_candidate",
    "adapter": "yokai",
    "adapter_version": 1,
    "num_agents": 2,
    "num_colors": 3,
    "deck_size": 9,
    "grid_size": 9,
    "num_hints": 4,
    "num_nodes": 13,
    "node_features": 13,
    "num_actions": 1068,
    "policy_structure": "shared_local",
}


def init_agent():
    return {
        "colors": jnp.full((9,), -1, dtype=jnp.int8),
        "partner_seen": jnp.zeros((9,), dtype=jnp.bool_),
        "positions": jnp.zeros((9, 2), dtype=jnp.int8),
    }


def _remember(obs, reset, memory):
    old_colors = jnp.where(reset, jnp.full((9,), -1, jnp.int8), memory["colors"])
    visible = jnp.sum(obs.node_features[:9, :3], axis=1) > 0
    seen_colors = jnp.argmax(obs.node_features[:9, :3], axis=1).astype(jnp.int8)
    colors = jnp.where(visible, seen_colors, old_colors)

    old_partner = jnp.where(reset, jnp.zeros((9,), jnp.bool_), memory["partner_seen"])
    partner_seen = old_partner | (obs.node_features[:9, 9] > 0)
    positions = obs.positions.astype(jnp.int8)
    return {
        "colors": colors,
        "partner_seen": partner_seen,
        "positions": positions,
    }


def _infer_last_color(colors):
    """Infer the sole unknown from the public three-of-each deck composition."""
    known = colors >= 0
    unknown_count = jnp.sum(~known)
    counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    remaining = jnp.argmax(3 - counts).astype(jnp.int8)
    return jnp.where((~known) & (unknown_count == 1), remaining, colors)


def _infer_hint_colors(obs, colors):
    """Combine literal placed-hint sets with remaining deck capacity."""
    hint_sets = obs.node_features[9:13, 3:6] > 0
    placed = obs.node_features[9:13, 8] > 0
    targets = obs.adjacency_matrix[9:13, :9] > 0
    attached = placed[:, None] & targets
    has_hint = jnp.any(attached, axis=0)
    forbidden = jnp.any(
        attached[:, :, None] & ~hint_sets[:, None, :],
        axis=0,
    )
    known_counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    possible = ~forbidden & (known_counts[None, :] < 3)
    determined = has_hint & (jnp.sum(possible, axis=1) == 1)
    inferred = jnp.argmax(possible, axis=1).astype(jnp.int8)
    return jnp.where((colors < 0) & determined, inferred, colors)


def _color_edges(positions, colors):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), k=1)
    known_same = (
        (colors[:, None] == colors[None, :])
        & (colors[:, None] >= 0)
        & upper
    )
    per_color = jnp.stack(
        [jnp.sum(adjacent & known_same & (colors[:, None] == c)) for c in range(3)]
    )
    return per_color, adjacent


def _is_solved(positions, colors):
    edges, _ = _color_edges(positions, colors)
    counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    return jnp.all(counts == 3) & jnp.all(edges >= 2)


def _completion_counts(obs, colors):
    """Count valid and solved board assignments for local color evidence."""
    positions = obs.positions
    unknown = colors < 0
    unknown_count = jnp.sum(unknown)
    unknown_ids = jnp.nonzero(unknown, size=5, fill_value=0)[0]
    codes = jnp.arange(243, dtype=jnp.int32)
    assignments = jnp.stack(
        [
            codes // 81,
            (codes // 27) % 3,
            (codes // 9) % 3,
            (codes // 3) % 3,
            codes % 3,
        ],
        axis=1,
    ).astype(jnp.int8)
    candidates = jnp.broadcast_to(colors[None, :], (243, 9))
    for slot in range(5):
        card_id = unknown_ids[slot]
        replacement = jnp.where(
            slot < unknown_count, assignments[:, slot], candidates[:, card_id]
        )
        candidates = candidates.at[:, card_id].set(replacement)

    counts = jnp.stack(
        [jnp.sum(candidates == c, axis=1) for c in range(3)], axis=1
    )
    hint_sets = obs.node_features[9:13, 3:6] > 0
    placed = obs.node_features[9:13, 8] > 0
    hint_targets = obs.adjacency_matrix[9:13, :9] > 0
    forbidden = jnp.any(
        placed[:, None, None]
        & hint_targets[:, :, None]
        & ~hint_sets[:, None, :],
        axis=0,
    )
    allowed = ~forbidden
    candidate_allowed = jnp.take_along_axis(
        allowed[None, :, :],
        jnp.clip(candidates, 0, 2)[:, :, None],
        axis=2,
    )[:, :, 0]
    valid = jnp.all(counts == 3, axis=1) & jnp.all(candidate_allowed, axis=1)
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), k=1)
    edges = jnp.stack(
        [
            jnp.sum(
                adjacent[None, :, :]
                & upper[None, :, :]
                & (candidates[:, :, None] == c)
                & (candidates[:, None, :] == c),
                axis=(1, 2),
            )
            for c in range(3)
        ],
        axis=1,
    )
    candidate_solved = jnp.all(edges >= 2, axis=1)
    solution_count = jnp.where(
        unknown_count <= 5,
        jnp.sum(valid & candidate_solved),
        0,
    ).astype(jnp.int32)
    valid_count = jnp.where(
        unknown_count <= 5,
        jnp.sum(valid),
        0,
    ).astype(jnp.int32)
    return solution_count, valid_count


def _observe_action(obs, colors, partner_seen):
    legal = obs.legal_actions[1:10]
    unknown = colors < 0
    ids = jnp.arange(9, dtype=jnp.int32)
    score = unknown.astype(jnp.int32) * 100
    score += ((~partner_seen) & unknown).astype(jnp.int32) * 10
    score += 8 - ids
    score = jnp.where(legal, score, -100000)
    return (1 + jnp.argmax(score)).astype(jnp.int32)


def _move_action(obs, colors):
    actions = jnp.arange(729, dtype=jnp.int32)
    card = actions // 81
    cell = actions % 81
    destination = jnp.stack([cell // 9, cell % 9], axis=1)

    old_positions = obs.positions.astype(jnp.int32)
    selector = jax.nn.one_hot(card, 9, dtype=jnp.int32)[:, :, None]
    positions = (
        old_positions[None, :, :] * (1 - selector)
        + destination[:, None, :] * selector
    )

    delta = jnp.abs(positions[:, :, None, :] - positions[:, None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), k=1)
    same = (
        (colors[:, None] == colors[None, :])
        & (colors[:, None] >= 0)
        & upper
    )
    different = (
        (colors[:, None] != colors[None, :])
        & (colors[:, None] >= 0)
        & (colors[None, :] >= 0)
        & upper
    )
    known_counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    deficits = (3 - known_counts).astype(jnp.float32)
    unknown_count = jnp.maximum(jnp.sum(colors < 0), 1).astype(jnp.float32)
    prior = deficits / unknown_count
    hint_sets = obs.node_features[9:13, 3:6] > 0
    placed_hints = obs.node_features[9:13, 8] > 0
    hint_targets = obs.adjacency_matrix[9:13, :9] > 0
    forbidden = jnp.any(
        placed_hints[:, None, None]
        & hint_targets[:, :, None]
        & ~hint_sets[:, None, :],
        axis=0,
    )
    allowed = ~forbidden
    restricted_prior = prior[None, :] * allowed.astype(jnp.float32)
    restricted_total = jnp.sum(restricted_prior, axis=1, keepdims=True)
    restricted_prior = jnp.where(
        restricted_total > 0,
        restricted_prior / jnp.maximum(restricted_total, 1e-6),
        prior[None, :],
    )
    known_one_hot = jax.nn.one_hot(jnp.clip(colors, 0, 2), 3)
    probabilities = jnp.where(
        (colors >= 0)[:, None], known_one_hot, restricted_prior
    )
    expected_same = jnp.sum(
        probabilities[:, None, :] * probabilities[None, :, :], axis=2
    ) * upper
    same_edges = jnp.sum(adjacent & same[None, :, :], axis=(1, 2))
    diff_edges = jnp.sum(adjacent & different[None, :, :], axis=(1, 2))
    pair_distance = jnp.sum(delta, axis=-1)
    same_distance = jnp.sum(pair_distance * same[None, :, :], axis=(1, 2))
    expected_edges = jnp.sum(adjacent * expected_same[None, :, :], axis=(1, 2))
    expected_distance = jnp.sum(
        pair_distance * expected_same[None, :, :], axis=(1, 2)
    )
    edges_by_color = jnp.stack(
        [
            jnp.sum(
                adjacent
                & same[None, :, :]
                & (colors[None, :, None] == c),
                axis=(1, 2),
            )
            for c in range(3)
        ],
        axis=1,
    )
    complete_bonus = jnp.sum(edges_by_color >= 2, axis=1)

    travel = jnp.sum(jnp.abs(destination - old_positions[card]), axis=1)
    moved_known = (colors[card] >= 0).astype(jnp.int32)
    score = (
        same_edges.astype(jnp.int32) * 140
        + complete_bonus.astype(jnp.int32) * 260
        - same_distance.astype(jnp.int32) * 4
        - diff_edges.astype(jnp.int32) * 2
        - travel.astype(jnp.int32)
        + moved_known * 12
        + expected_edges * 30.0
        - expected_distance * 1.0
    )
    legal = obs.legal_actions[10:739]
    score = jnp.where(legal, score, -1000000)
    return (10 + jnp.argmax(score)).astype(jnp.int32), jnp.max(score)


def _hint_action(obs, colors, partner_seen):
    features = obs.node_features
    hint_sets = features[9:13, 3:6] > 0
    hidden = jnp.sum(hint_sets, axis=1) == 0
    unplaced = features[9:13, 8] == 0
    reveal_legal = obs.legal_actions[739:743]

    reveal_score = jnp.array([4, 3, 2, 1], dtype=jnp.int32)
    reveal_score = jnp.where(reveal_legal & hidden & unplaced, reveal_score, -10000)
    reveal_action = (739 + jnp.argmax(reveal_score)).astype(jnp.int32)
    can_reveal = jnp.any(reveal_legal)

    actions = jnp.arange(324, dtype=jnp.int32)
    hint = actions // 81
    cell = actions % 81
    xy = jnp.stack([cell // 9, cell % 9], axis=1)
    match = jnp.all(obs.positions[None, :, :] == xy[:, None, :], axis=2)
    target = jnp.argmax(match, axis=1)
    target_color = colors[target]
    known = target_color >= 0
    safe_color = jnp.clip(target_color, 0, 2)
    compatible = hint_sets[hint, safe_color] & known

    _, adjacency = _color_edges(obs.positions, colors)
    same_neighbor_count = jnp.sum(
        adjacency
        & (colors[:, None] == colors[None, :])
        & (colors[:, None] >= 0),
        axis=1,
    )
    stable = same_neighbor_count[target]
    hint_size = jnp.sum(hint_sets[hint], axis=1)
    place_score = (
        compatible.astype(jnp.int32) * 1000
        + known.astype(jnp.int32) * 100
        + stable.astype(jnp.int32) * 40
        + partner_seen[target].astype(jnp.int32) * 20
        + (3 - hint_size).astype(jnp.int32) * 3
        + (8 - target).astype(jnp.int32)
    )
    place_legal = obs.legal_actions[743:1067]
    place_score = jnp.where(place_legal, place_score, -100000)
    place_action = (743 + jnp.argmax(place_score)).astype(jnp.int32)
    return jnp.where(can_reveal, reveal_action, place_action).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    next_memory = _remember(obs, reset, memory)
    colors = _infer_last_color(next_memory["colors"])
    planning_colors = _infer_last_color(_infer_hint_colors(obs, colors))
    active = obs.node_features[0, 10] > 0
    substep = obs.node_features[0, 11].astype(jnp.int32)
    solved = _is_solved(obs.positions, planning_colors)
    placed_count = jnp.sum(obs.node_features[9:13, 8] > 0)
    solution_count, valid_count = _completion_counts(obs, planning_colors)
    completion_evidence = solution_count > 0
    observation_overlap = jnp.sum(
        (colors >= 0) & next_memory["partner_seen"]
    )
    two_turn_evidence = (
        (placed_count == 2)
        & (observation_overlap <= 3)
        & (valid_count > 0)
        & (3 * solution_count >= valid_count)
        & ~((solution_count == 27) & (valid_count == 54))
    )
    three_turn_evidence = (
        (placed_count == 1)
        & (observation_overlap <= 1)
        & (valid_count > 0)
        & (10 * solution_count >= valid_count)
        & ~(
            (solution_count == 2)
            & ((valid_count == 16) | (valid_count == 20))
        )
    )
    safe_to_end = (
        solved
        | ((placed_count == 3) & completion_evidence)
        | two_turn_evidence
        | three_turn_evidence
    )

    observe_action = _observe_action(obs, colors, next_memory["partner_seen"])
    move_action, move_score = _move_action(obs, planning_colors)
    hint_action = _hint_action(obs, planning_colors, next_memory["partner_seen"])
    step_action = jax.lax.switch(
        substep,
        (
            lambda: jnp.where(safe_to_end & obs.legal_actions[1067], 1067, observe_action),
            lambda: observe_action,
            lambda: move_action,
            lambda: hint_action,
        ),
    ).astype(jnp.int32)
    action = jnp.where(active, step_action, 0).astype(jnp.int32)
    trace = {
        "colors": colors,
        "completion_evidence": completion_evidence,
        "completion_solution_count": solution_count,
        "completion_valid_count": valid_count,
        "move_score": move_score.astype(jnp.int32),
        "observation_overlap": observation_overlap.astype(jnp.int32),
        "solved": solved,
        "substep": substep,
    }
    return action, next_memory, trace
