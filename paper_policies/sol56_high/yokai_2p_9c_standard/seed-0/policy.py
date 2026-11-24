"""Memory-based, convention-light Yōkai policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        'colors': jnp.full((9,), -1, dtype=jnp.int8),
        'last_positions': jnp.full((9, 2), -1, dtype=jnp.int32),
        'moved': jnp.zeros((9,), dtype=jnp.bool_),
        'partner_seen': jnp.zeros((9,), dtype=jnp.bool_),
    }


def _remember_colors(obs, memory, reset):
    colors = jnp.where(reset, jnp.full_like(memory['colors'], -1), memory['colors'])
    shown = obs.node_features[:9, :3]
    visible = jnp.sum(shown, axis=1) > 0
    colors = jnp.where(visible, jnp.argmax(shown, axis=1).astype(jnp.int8), colors)
    unknown = colors < 0
    counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    missing_color = jnp.argmin(counts).astype(jnp.int8)
    infer = jnp.sum(unknown) == 1
    return jnp.where(infer & unknown, missing_color, colors)


def _infer_hint_colors(obs, colors):
    """Apply only color facts forced by deck counts and public correct hints."""
    n = jnp.arange(27, dtype=jnp.int32)
    digits = jnp.stack([(n // 9) % 3, (n // 3) % 3, n % 3], axis=1)
    unknown = colors < 0
    rank = jnp.cumsum(unknown.astype(jnp.int32)) - 1
    guesses = digits[:, jnp.clip(rank, 0, 2)]
    candidates = jnp.where(
        unknown[None, :], guesses, colors[None, :].astype(jnp.int32))
    counts_ok = jnp.all(jnp.stack(
        [jnp.sum(candidates == c, axis=1) for c in range(3)], axis=1) == 3,
        axis=1)
    hint_sets = obs.node_features[9:13, 3:6]
    hint_ok = jnp.ones((27,), dtype=jnp.bool_)
    for h in range(4):
        placed = obs.node_features[9 + h, 8] > 0
        target = jnp.argmax(obs.adjacency_matrix[9 + h, :9])
        allowed = hint_sets[h, candidates[:, target]] > 0
        hint_ok = hint_ok & (~placed | allowed)
    feasible = counts_ok & hint_ok & (jnp.sum(unknown) <= 3)
    first = jnp.argmax(feasible.astype(jnp.int32))
    inferred = candidates[first]
    forced = jnp.all(~feasible[:, None] | (candidates == inferred[None, :]), axis=0)
    return jnp.where(unknown & jnp.any(feasible) & forced,
                     inferred.astype(jnp.int8), colors)


def _posterior_data(obs, colors):
    """Same-color probabilities, scaled to 27, under public constraints."""
    n = jnp.arange(27, dtype=jnp.int32)
    digits = jnp.stack([(n // 9) % 3, (n // 3) % 3, n % 3], axis=1)
    unknown = colors < 0
    rank = jnp.cumsum(unknown.astype(jnp.int32)) - 1
    candidates = jnp.where(
        unknown[None, :], digits[:, jnp.clip(rank, 0, 2)],
        colors[None, :].astype(jnp.int32))
    counts_ok = jnp.all(jnp.stack(
        [jnp.sum(candidates == c, axis=1) for c in range(3)], axis=1) == 3,
        axis=1)
    hint_sets = obs.node_features[9:13, 3:6]
    hint_ok = jnp.ones((27,), dtype=jnp.bool_)
    for h in range(4):
        target = jnp.argmax(obs.adjacency_matrix[9 + h, :9])
        allowed = hint_sets[h, candidates[:, target]] > 0
        hint_ok = hint_ok & ((obs.node_features[9 + h, 8] == 0) | allowed)
    feasible = counts_ok & hint_ok & (jnp.sum(unknown) <= 3)
    same_samples = candidates[:, :, None] == candidates[:, None, :]
    numerator = jnp.sum(same_samples & feasible[:, None, None], axis=0)
    denominator = jnp.maximum(jnp.sum(feasible), 1)
    posterior = (numerator * 27 + denominator // 2) // denominator
    known = colors >= 0
    known_same = ((colors[:, None] == colors[None, :])
                  & known[:, None] & known[None, :])
    weights = jnp.where(
        jnp.any(feasible), posterior, known_same.astype(jnp.int32) * 27)
    return weights, candidates, feasible


def _infer_partner_move(obs, colors, last_positions, changed):
    """Read a late, freshly observed goal-directed attachment as color evidence."""
    new_delta = jnp.abs(
        obs.positions[:, None, :] - obs.positions[None, :, :])
    old_delta = jnp.abs(
        last_positions[:, None, :] - last_positions[None, :, :])
    new_adjacent = jnp.sum(new_delta, axis=-1) == 1
    old_adjacent = jnp.sum(old_delta, axis=-1) == 1
    known = colors >= 0
    new_counts = jnp.stack([
        jnp.sum(new_adjacent & known[None, :] & (colors[None, :] == c), axis=1)
        for c in range(3)], axis=1)
    old_counts = jnp.stack([
        jnp.sum(old_adjacent & known[None, :] & (colors[None, :] == c), axis=1)
        for c in range(3)], axis=1)
    inferred = jnp.argmax(new_counts, axis=1).astype(jnp.int8)
    total_known_neighbors = jnp.sum(new_counts, axis=1)
    unanimous = jnp.max(new_counts, axis=1) == total_known_neighbors
    strengthened = (
        jnp.max(new_counts, axis=1)
        > old_counts[jnp.arange(9), inferred.astype(jnp.int32)])
    enough_hints_revealed = jnp.all(
        jnp.sum(obs.node_features[9:13, 3:6], axis=1) > 0)
    placed_count = jnp.sum(obs.node_features[9:13, 8] > 0)
    eligible = ((obs.node_features[0, 10] == 0)
                & (obs.node_features[0, 11] == 3)
                & enough_hints_revealed & (placed_count <= 1)
                & jnp.all(last_positions >= 0))
    evidence = (eligible & changed & (obs.node_features[:9, 9] > 0)
                & (colors < 0) & (total_known_neighbors > 0)
                & unanimous & strengthened)
    return jnp.where(evidence, inferred, colors)


def _observe_action(obs, colors, partner_seen):
    legal = obs.legal_actions[1:10]
    unknown = legal & (colors < 0)
    shared_unknown = unknown & partner_seen
    shared_choice = jnp.argmin(jnp.where(shared_unknown, jnp.arange(9), 20))
    unknown_choice = jnp.argmin(jnp.where(unknown, jnp.arange(9), 20))
    fallback = jnp.argmin(jnp.where(legal, jnp.arange(9), 20))
    choice = jnp.where(jnp.any(shared_unknown), shared_choice,
                       jnp.where(jnp.any(unknown), unknown_choice, fallback))
    return (1 + choice).astype(jnp.int32)


def _edge_data(positions, colors):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    known = colors >= 0
    same = (colors[:, None] == colors[None, :]) & known[:, None] & known[None, :]
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    return adjacent, same, upper


def _solved(positions, colors):
    adjacent, same, upper = _edge_data(positions, colors)
    edge_counts = jnp.stack([
        jnp.sum(adjacent & same & upper & (colors[:, None] == c))
        for c in range(3)
    ])
    counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    return jnp.all(counts == 3) & jnp.all(edge_counts >= 2)


def _robust_solved(obs, colors):
    """True when every deck/hint-consistent completion is already solved."""
    n = jnp.arange(27, dtype=jnp.int32)
    digits = jnp.stack([(n // 9) % 3, (n // 3) % 3, n % 3], axis=1)
    unknown = colors < 0
    rank = jnp.cumsum(unknown.astype(jnp.int32)) - 1
    guesses = digits[:, jnp.clip(rank, 0, 2)]
    candidates = jnp.where(unknown[None, :], guesses, colors[None, :].astype(jnp.int32))
    counts_ok = jnp.all(jnp.stack([jnp.sum(candidates == c, axis=1) for c in range(3)], axis=1) == 3, axis=1)

    hint_sets = obs.node_features[9:13, 3:6]
    hint_ok = jnp.ones((27,), dtype=jnp.bool_)
    for h in range(4):
        placed = obs.node_features[9 + h, 8] > 0
        target = jnp.argmax(obs.adjacency_matrix[9 + h, :9])
        allowed = hint_sets[h, candidates[:, target]] > 0
        hint_ok = hint_ok & (~placed | allowed)
    feasible = counts_ok & hint_ok & (jnp.sum(unknown) <= 3)

    delta = jnp.abs(obs.positions[:, None, :] - obs.positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    candidate_solved = jnp.ones((27,), dtype=jnp.bool_)
    for c in range(3):
        mask = candidates == c
        edges = jnp.sum(adjacent[None, :, :] & mask[:, :, None] & mask[:, None, :] & upper[None, :, :], axis=(1, 2))
        candidate_solved = candidate_solved & (edges >= 2)
    return jnp.any(feasible) & jnp.all(~feasible | candidate_solved)


def _move_action(obs, colors, moved_before):
    actions = jnp.arange(1068, dtype=jnp.int32)
    raw = jnp.clip(actions - 10, 0, 728)
    card = raw // 81
    cell = raw % 81
    dest = jnp.stack([cell // 9, cell % 9], axis=1)
    card_axis = jnp.arange(9)[None, :, None]
    post = jnp.where(card_axis == card[:, None, None], dest[:, None, :], obs.positions[None, :, :])
    delta = jnp.abs(post[:, :, None, :] - post[:, None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    known = colors >= 0
    same = (colors[:, None] == colors[None, :]) & known[:, None] & known[None, :]
    different = (colors[:, None] != colors[None, :]) & known[:, None] & known[None, :]
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    same_edges = jnp.sum(adjacent & same[None, :, :] & upper[None, :, :], axis=(1, 2))
    posterior_same, color_samples, feasible_samples = _posterior_data(obs, colors)
    expected_edges = jnp.sum(
        adjacent * posterior_same[None, :, :] * upper[None, :, :], axis=(1, 2))
    sample_same = color_samples[:, :, None] == color_samples[:, None, :]
    sample_edges = jnp.sum(
        adjacent[:, None, :, :] & sample_same[None, :, :, :]
        & upper[None, None, :, :], axis=(2, 3))
    worst_edges = jnp.min(
        jnp.where(feasible_samples[None, :], sample_edges, 99), axis=1)
    diff_edges = jnp.sum(adjacent & different[None, :, :] & upper[None, :, :], axis=(1, 2))
    complete_bonus = jnp.zeros((1068,), dtype=jnp.int32)
    finishable = jnp.zeros((1068,), dtype=jnp.bool_)
    stranded_cost = jnp.zeros((1068,), dtype=jnp.int32)
    board_degree = jnp.sum(adjacent, axis=2)
    locked = obs.node_features[:9, 8] > 0
    for c in range(3):
        c_edges = jnp.sum(
            adjacent & same[None, :, :] & upper[None, :, :]
            & (colors[None, :, None] == c), axis=(1, 2))
        complete_bonus = complete_bonus + jnp.where(
            (jnp.sum(colors == c) == 3) & (c_edges >= 2), 1, 0)
        c_mask = colors == c
        c_degree = jnp.sum(adjacent & c_mask[None, :, None] & c_mask[None, None, :], axis=2)
        stranded_cost = stranded_cost + jnp.where(
            jnp.sum(colors == c) == 3,
            jnp.sum(jnp.where(c_mask[None, :] & (c_degree == 0), board_degree, 0), axis=1),
            0)
        isolated = jnp.argmin(jnp.where(c_mask[None, :], c_degree, 20), axis=1)
        remain = jnp.arange(9)[None, :] != isolated[:, None]
        start = jnp.argmax(remain, axis=1)
        reach = jnp.arange(9)[None, :] == start[:, None]
        for _ in range(8):
            reach = (reach | jnp.any(adjacent & reach[:, :, None], axis=1)) & remain
        removal_connected = jnp.all(reach | ~remain, axis=1)
        finishable = finishable | (
            (jnp.sum(colors == c) == 3) & (c_edges == 1)
            & ~locked[isolated] & removal_connected)
    moved_known = colors[jnp.clip(card, 0, 8)] >= 0
    placed_count = jnp.sum(obs.node_features[9:13, 8] > 0)
    setup_bonus = ((placed_count == 2) & (complete_bonus == 2) & finishable).astype(jnp.int32)

    posterior_turn = (placed_count <= 1) & (jnp.sum(colors < 0) <= 3)
    edge_utility = jnp.where(
        posterior_turn,
        worst_edges * 5000 + expected_edges * 4,
        same_edges * 100)
    utility = (complete_bonus * 1000 + edge_utility
               + setup_bonus * 5000 - stranded_cost * 50
               + moved_known.astype(jnp.int32) * 3)
    score = utility * 2000 - actions
    valid = obs.legal_actions & (actions >= 10) & (actions < 739)
    return jnp.argmax(jnp.where(valid, score, -10**9)).astype(jnp.int32)


def _hint_action(obs, colors, partner_seen):
    reveal_mask = obs.legal_actions[739:743]
    reveal = (739 + jnp.argmin(jnp.where(reveal_mask, jnp.arange(4), 10))).astype(jnp.int32)
    actions = jnp.arange(1068, dtype=jnp.int32)
    raw = jnp.clip(actions - 743, 0, 323)
    hint = raw // 81
    cell = raw % 81
    x = cell // 9
    y = cell % 9
    at_cell = ((obs.positions[:, 0][None, :] == x[:, None])
               & (obs.positions[:, 1][None, :] == y[:, None]))
    card = jnp.argmax(at_cell, axis=1)
    exists = jnp.any(at_cell, axis=1)
    card_color = colors[card]
    hint_sets = obs.node_features[9:13, 3:6]
    correct = (card_color >= 0) & (hint_sets[hint, jnp.clip(card_color, 0, 2)] > 0)
    adjacent, same, _ = _edge_data(obs.positions, colors)
    same_degree = jnp.sum(adjacent & same, axis=1)
    board_degree = jnp.sum(adjacent, axis=1)
    color_complete = jnp.zeros((9,), dtype=jnp.bool_)
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    for c in range(3):
        edges = jnp.sum(adjacent & same & upper & (colors[:, None] == c))
        color_complete = color_complete | ((colors == c) & (jnp.sum(colors == c) == 3) & (edges >= 2))
    _, color_samples, feasible_samples = _posterior_data(obs, colors)
    sample_complete_card = jnp.zeros((27, 9), dtype=jnp.bool_)
    for c in range(3):
        c_mask = color_samples == c
        c_edges = jnp.sum(
            adjacent[None, :, :] & c_mask[:, :, None] & c_mask[:, None, :]
            & upper[None, :, :], axis=(1, 2))
        sample_complete_card = sample_complete_card | (
            c_mask & (c_edges >= 2)[:, None])
    sample_count = jnp.maximum(jnp.sum(feasible_samples), 1)
    posterior_complete = (
        jnp.sum(sample_complete_card & feasible_samples[:, None], axis=0) * 27
        + sample_count // 2) // sample_count
    completion_value = jnp.where(
        color_complete, 3000, posterior_complete * 100)
    utility = (correct.astype(jnp.int32) * 10000
               + partner_seen[card].astype(jnp.int32) * 50
               + completion_value[card]
               + same_degree[card] * 500 + board_degree[card] * 300
               - card * 8 - hint)
    valid = obs.legal_actions & (actions >= 743) & (actions < 1067) & exists
    place = jnp.argmax(jnp.where(valid, utility, -10**9)).astype(jnp.int32)
    return jnp.where(jnp.any(reveal_mask), reveal, place)


def agent_act(obs, reset, memory, rng):
    del rng
    colors = _remember_colors(obs, memory, reset)
    had_positions = jnp.all(memory['last_positions'] >= 0, axis=1)
    changed = jnp.any(obs.positions != memory['last_positions'], axis=1) & had_positions
    colors = _infer_partner_move(
        obs, colors, memory['last_positions'], changed)
    colors = _infer_hint_colors(obs, colors)
    moved_before = jnp.where(reset, jnp.zeros_like(memory['moved']), memory['moved']) | changed
    partner_seen = jnp.where(
        reset, jnp.zeros_like(memory['partner_seen']), memory['partner_seen'])
    partner_seen = partner_seen | (obs.node_features[:9, 9] > 0)
    substep = obs.node_features[0, 11].astype(jnp.int32)
    active = obs.node_features[0, 10] > 0
    observe = _observe_action(obs, colors, partner_seen)
    move = _move_action(obs, colors, moved_before)
    hint = _hint_action(obs, colors, partner_seen)
    at_zero = jnp.where(_robust_solved(obs, colors), jnp.int32(1067), observe)
    active_action = jnp.select(
        [substep == 0, substep == 1, substep == 2, substep == 3],
        [at_zero, observe, move, hint], default=jnp.int32(0))
    action = jnp.where(active, active_action, jnp.int32(0))
    trace = {
        'colors': colors,
        'locked': obs.node_features[:9, 8],
        'positions': obs.positions,
        'known_count': jnp.sum(colors >= 0).astype(jnp.int32),
        'solved': _solved(obs.positions, colors),
    }
    next_memory = {
        'colors': colors,
        'last_positions': obs.positions,
        'moved': moved_before,
        'partner_seen': partner_seen,
    }
    return action.astype(jnp.int32), next_memory, trace
