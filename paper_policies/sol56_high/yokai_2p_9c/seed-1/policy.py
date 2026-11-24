"""Evidence-driven local policy for the two-player 9-card Yōkai game."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_clusterer', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        'positions': jnp.zeros((9, 2), dtype=jnp.int32),
        'partner_moved': jnp.asarray(-1, dtype=jnp.int32),
    }


def _public_hint_support(obs):
    hint_features = obs.node_features[9:13]
    hint_colors = hint_features[:, 3:6].astype(jnp.int32)
    placed = hint_features[:, 8] == 1
    colocated = jnp.all(
        obs.positions[:, None, :] == hint_features[None, :, 6:8].astype(jnp.int32), axis=2
    )
    public_sets = (colocated & placed[None, :]).astype(jnp.int32) @ hint_colors
    informed = obs.node_features[:9, 9:10] > 0
    return jnp.where(
        informed & (jnp.sum(public_sets, axis=1, keepdims=True) > 0),
        public_sets > 0,
        True,
    )


def _known_colors(obs):
    one_hot = obs.node_features[:9, :3].astype(jnp.int32)
    known = jnp.sum(one_hot, axis=1) > 0
    color = jnp.argmax(one_hot, axis=1).astype(jnp.int32)

    hint_features = obs.node_features[9:13]
    hint_colors = hint_features[:, 3:6].astype(jnp.int32)
    singleton = (jnp.sum(hint_colors, axis=1) == 1) & (hint_features[:, 8] == 1)
    colocated = jnp.all(
        obs.positions[:, None, :] == hint_features[None, :, 6:8].astype(jnp.int32), axis=2
    )
    public_exact = (colocated & singleton[None, :]).astype(jnp.int32) @ hint_colors
    public_exact = public_exact * (obs.node_features[:9, 9:10] > 0)
    has_public_exact = jnp.sum(public_exact, axis=1) > 0
    color = jnp.where((~known) & has_public_exact, jnp.argmax(public_exact, axis=1), color)
    known = known | has_public_exact

    counts = jnp.sum(one_hot, axis=0)
    counts = counts + jnp.sum(public_exact * (~(jnp.sum(one_hot, axis=1) > 0))[:, None], axis=0)
    missing_color = jnp.argmax(3 - counts).astype(jnp.int32)
    inferred = (~known) & (jnp.sum(known) == 8)
    return jnp.where(inferred, missing_color, color), known | inferred


def _observe_action(obs, color_known):
    legal = obs.legal_actions[1:10]
    partner_seen = obs.node_features[:9, 9].astype(jnp.int32)
    priority = (1 - partner_seen) * 100 + (9 - jnp.arange(9, dtype=jnp.int32))
    priority = jnp.where(color_known, -1000, priority)
    priority = jnp.where(legal, priority, -10000)
    return (1 + jnp.argmax(priority)).astype(jnp.int32)


def _move_action(obs, colors, known, posterior_same):
    k = jnp.arange(729, dtype=jnp.int32)
    card = k // 81
    cell = k % 81
    dest = jnp.stack((cell // 9, cell % 9), axis=1)
    positions = obs.positions.astype(jnp.int32)
    proposed = jnp.broadcast_to(positions, (729, 9, 2))
    row = jnp.arange(729, dtype=jnp.int32)
    proposed = proposed.at[row, card].set(dest)
    delta = jnp.abs(proposed[:, :, None, :] - proposed[:, None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), k=1)
    both_known = known[:, None] & known[None, :]
    same = (colors[:, None] == colors[None, :]) & both_known & upper
    different = (colors[:, None] != colors[None, :]) & both_known & upper
    same_edges = jnp.sum(adjacent & same[None, :, :], axis=(1, 2))
    diff_edges = jnp.sum(adjacent & different[None, :, :], axis=(1, 2))
    total_edges = jnp.sum(adjacent & upper[None, :, :], axis=(1, 2))
    travel = jnp.sum(jnp.abs(dest - positions[card]), axis=1)

    expected_same = jnp.sum(
        adjacent * posterior_same[None, :, :] * upper[None, :, :], axis=(1, 2)
    )
    expected_different = total_edges - expected_same
    score = (1000.0 * expected_same - 20.0 * expected_different
             - 150.0 * diff_edges + 5.0 * total_edges - travel)
    score = jnp.where(obs.legal_actions[10:739], score, -1000000)
    return (10 + jnp.argmax(score)).astype(jnp.int32)


def _hint_action(obs, colors, known):
    reveal_legal = obs.legal_actions[739:743]
    reveal = (739 + jnp.argmax(reveal_legal)).astype(jnp.int32)
    k = jnp.arange(324, dtype=jnp.int32)
    hint = k // 81
    cell = k % 81
    target_xy = jnp.stack((cell // 9, cell % 9), axis=1)
    positions = obs.positions.astype(jnp.int32)
    match = jnp.all(target_xy[:, None, :] == positions[None, :, :], axis=2)
    target_card = jnp.argmax(match, axis=1)
    target_exists = jnp.any(match, axis=1)
    hint_colors = obs.node_features[9:13, 3:6].astype(jnp.int32)
    compatible = hint_colors[hint, colors[target_card]] > 0
    compatible = compatible & known[target_card] & target_exists
    card_delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    card_adj = jnp.sum(card_delta, axis=-1) == 1
    same_neighbor = card_adj & known[:, None] & known[None, :] & (colors[:, None] == colors[None, :])
    stable_degree = jnp.sum(same_neighbor, axis=1)[target_card]
    color_edges = jnp.asarray([
        jnp.sum(same_neighbor & (colors[:, None] == c)) // 2 for c in range(3)
    ])
    completed_group = color_edges[colors[target_card]] >= 2
    partner_seen = obs.node_features[:9, 9].astype(jnp.int32)[target_card]
    hint_size = jnp.sum(hint_colors[hint], axis=1)
    communication = jnp.where(hint_size == 1, 1 - partner_seen, partner_seen)
    score = (compatible.astype(jnp.int32) * 10000 + completed_group * 6000
             + communication * 2000 + stable_degree * 50 - hint_size)
    place_legal = obs.legal_actions[743:1067]
    has_safe_placement = jnp.any(place_legal & compatible & completed_group)
    score = jnp.where(place_legal, score, -1000000)
    place = (743 + jnp.argmax(score)).astype(jnp.int32)
    choose_place = has_safe_placement | (~jnp.any(reveal_legal))
    return jnp.where(choose_place, place, reveal)


def _completion_counts(obs, colors, known, partner_moved):
    codes = jnp.arange(3 ** 9, dtype=jnp.int32)
    powers = 3 ** jnp.arange(9, dtype=jnp.int32)
    assignments = (codes[:, None] // powers[None, :]) % 3
    balanced = jnp.all(
        jnp.stack([jnp.sum(assignments == c, axis=1) for c in range(3)], axis=1) == 3,
        axis=1,
    )
    allowed = _public_hint_support(obs)
    assignment_allowed = jnp.sum(
        jnp.eye(3, dtype=jnp.int32)[assignments] * allowed[None, :, :], axis=2
    ) > 0
    card_consistent = jnp.where(known[None, :], assignments == colors[None, :], assignment_allowed)
    consistent = jnp.all(card_consistent, axis=1)
    feasible = balanced & consistent

    positions = obs.positions.astype(jnp.int32)
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    card_adjacent = jnp.sum(delta, axis=-1) == 1
    adjacent = card_adjacent & jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), k=1)
    connected = []
    for c in range(3):
        in_color = assignments == c
        edges = jnp.sum(
            adjacent[None, :, :] & in_color[:, :, None] & in_color[:, None, :], axis=(1, 2)
        )
        connected.append(edges >= 2)
    solved = feasible & jnp.all(jnp.stack(connected, axis=1), axis=1)
    feasible_count = jnp.sum(feasible).astype(jnp.int32)
    pair_same = assignments[:, :, None] == assignments[:, None, :]
    uniform_same = jnp.sum(pair_same * feasible[:, None, None], axis=0) / jnp.maximum(1, feasible_count)
    moved = jnp.clip(partner_moved, 0, 8)
    partner_seen = obs.node_features[:9, 9] > 0
    informed_neighbors = card_adjacent[moved, :] & partner_seen
    same_new_neighbors = jnp.sum(
        pair_same[:, moved, :] * informed_neighbors[None, :], axis=1
    )
    likelihood = (1 + 8 * same_new_neighbors).astype(jnp.int32)
    weighted_feasible = feasible.astype(jnp.int32) * likelihood
    weight_sum = jnp.sum(weighted_feasible)
    action_informed_same = jnp.sum(
        pair_same * weighted_feasible[:, None, None], axis=0
    ) / jnp.maximum(1, weight_sum)
    partner_saw_moved = partner_seen[moved]
    use_action_signal = (partner_moved >= 0) & partner_saw_moved
    posterior_same = jnp.where(
        use_action_signal, action_informed_same, uniform_same
    )
    solved_count = jnp.sum(solved).astype(jnp.int32)
    return feasible_count, solved_count, posterior_same


def agent_act(obs, reset, memory, rng):
    del rng
    active = obs.node_features[0, 10] == 1
    substep = obs.node_features[0, 11].astype(jnp.int32)
    previous_positions = jnp.where(reset, obs.positions, memory['positions'])
    position_changed = jnp.any(obs.positions != previous_positions, axis=1)
    saw_partner_move = (~reset) & (~active) & (substep == 3) & jnp.any(position_changed)
    remembered_move = jnp.where(reset, -1, memory['partner_moved'])
    partner_moved = jnp.where(saw_partner_move, jnp.argmax(position_changed), remembered_move)
    colors, known = _known_colors(obs)
    feasible_count, solved_count, posterior_same = _completion_counts(
        obs, colors, known, partner_moved
    )
    observe = _observe_action(obs, known)
    move = _move_action(obs, colors, known, posterior_same)
    hint = _hint_action(obs, colors, known)
    active_action = jnp.where(substep < 2, observe, jnp.where(substep == 2, move, hint))
    partner_seen = obs.node_features[:9, 9] > 0
    novel_partner_evidence = jnp.sum(partner_seen & (~known))
    certain = solved_count == feasible_count
    placed_hints = jnp.sum(obs.node_features[9:13, 8] > 0)
    likely = jnp.where(
        placed_hints <= 1,
        4 * solved_count >= feasible_count,
        3 * solved_count >= feasible_count,
    )
    certified = (feasible_count > 0) & jnp.where(
        novel_partner_evidence >= 2, likely, certain
    )
    active_action = jnp.where((substep == 0) & certified, 1067, active_action)
    action = jnp.where(active, active_action, 0).astype(jnp.int32)
    next_memory = {
        'positions': obs.positions.astype(jnp.int32),
        'partner_moved': partner_moved.astype(jnp.int32),
    }
    adjacency = obs.adjacency_matrix[:9, :9].astype(jnp.int32)
    known_same = known[:, None] & known[None, :] & (colors[:, None] == colors[None, :])
    trace = {
        'known_count': jnp.sum(known).astype(jnp.int32),
        'known_same_edges': (jnp.sum(adjacency * known_same) // 2).astype(jnp.int32),
        'novel_partner_evidence': novel_partner_evidence.astype(jnp.int32),
        'partner_moved': partner_moved.astype(jnp.int32),
        'feasible_assignments': feasible_count,
        'solved_assignments': solved_count,
    }
    return action, next_memory, trace
