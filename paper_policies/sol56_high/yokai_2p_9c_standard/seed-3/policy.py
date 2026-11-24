"""Locally informed, convention-light Yōkai policy."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_greedy_very_strong_partner_edges', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "colors": jnp.full((9,), -1, dtype=jnp.int8),
        "partner_seen": jnp.zeros((9,), dtype=jnp.bool_),
        "positions": jnp.zeros((9, 2), dtype=jnp.int32),
        "partner_edges": jnp.zeros((9, 9), dtype=jnp.bool_),
    }


def _remember_colors(obs, reset, memory):
    old = jnp.where(reset, jnp.full((9,), -1, dtype=jnp.int8), memory["colors"])
    visible = jnp.sum(obs.node_features[:9, :3], axis=1) > 0
    seen = jnp.argmax(obs.node_features[:9, :3], axis=1).astype(jnp.int8)
    colors = jnp.where(visible, seen, old)

    hint_sets = obs.node_features[9:13, 3:6]
    hint_positions = obs.node_features[9:13, 6:8].astype(jnp.int32)
    single_placed = (
        (obs.node_features[9:13, 8] > 0) & (jnp.sum(hint_sets, axis=1) == 1)
    )
    card_matches = jnp.all(
        obs.positions[:, None, :] == hint_positions[None, :, :], axis=2
    ) & single_placed[None, :]
    public_hint_color = jnp.max(
        jnp.where(
            card_matches,
            jnp.argmax(hint_sets, axis=1).astype(jnp.int8)[None, :],
            jnp.int8(-1),
        ),
        axis=1,
    )
    colors = jnp.where((colors < 0) & (public_hint_color >= 0), public_hint_color, colors)

    # Once two color multiplicities are exhausted, every still unknown card is
    # forced to be the third color.
    counts = jnp.sum(colors[:, None] == jnp.arange(3, dtype=jnp.int8)[None, :], axis=0)
    forced = jnp.sum(counts == 3) >= 2
    remaining_color = jnp.argmin(counts).astype(jnp.int8)
    return jnp.where((colors < 0) & forced, remaining_color, colors)


def _board_is_solved(positions, colors):
    ii, jj = jnp.triu_indices(9, 1)
    distances = jnp.sum(jnp.abs(positions[ii] - positions[jj]), axis=1)
    adjacent = distances == 1

    def connected(color):
        same = (colors[ii] == color) & (colors[jj] == color)
        # Three grid cells cannot make an orthogonal triangle.  Thus at least
        # two edges among the three same-color cards is exactly connectivity.
        return jnp.sum(adjacent & same) >= 2

    all_known = jnp.all(colors >= 0)
    return all_known & connected(0) & connected(1) & connected(2)


def _completion_candidates(obs, colors):
    """Color completions and validity through five locally unknown cards."""
    unknown_count = jnp.sum(colors < 0)
    unknown_ids = jnp.argsort(colors >= 0)[:5]
    code = jnp.arange(243, dtype=jnp.int32)
    assignments = jnp.stack(
        tuple(((code // (3 ** digit)) % 3).astype(jnp.int8) for digit in range(5)),
        axis=1,
    )
    candidates = jnp.broadcast_to(colors[None, :], (243, 9))
    active_slots = jnp.arange(5) < unknown_count
    updates = jnp.where(
        active_slots[None, :], assignments, colors[unknown_ids][None, :]
    )
    candidates = candidates.at[:, unknown_ids].set(updates)

    counts = jnp.sum(
        candidates[:, :, None] == jnp.arange(3, dtype=jnp.int8)[None, None, :],
        axis=1,
    )
    valid = (
        jnp.all(counts == 3, axis=1)
        & (unknown_count <= 5)
    )

    hint_positions = obs.node_features[9:13, 6:8].astype(jnp.int32)
    hint_placed = obs.node_features[9:13, 8] > 0
    targets = jnp.argmax(
        jnp.all(hint_positions[:, None, :] == obs.positions[None, :, :], axis=2),
        axis=1,
    )
    target_colors = jnp.take_along_axis(
        candidates, jnp.broadcast_to(targets[None, :], (243, 4)), axis=1
    )
    hint_sets = obs.node_features[9:13, 3:6]
    membership = jnp.take_along_axis(
        hint_sets[None, :, :], jnp.maximum(target_colors, 0)[:, :, None], axis=2
    )[:, :, 0] > 0
    valid = valid & jnp.all((~hint_placed[None, :]) | membership, axis=1)
    return candidates, valid, unknown_count


def _solve_probability(obs, colors):
    candidates, valid, unknown_count = _completion_candidates(obs, colors)

    ii, jj = jnp.triu_indices(9, 1)
    adjacent = jnp.sum(jnp.abs(obs.positions[ii] - obs.positions[jj]), axis=1) == 1

    def connected(color):
        same = (candidates[:, ii] == color) & (candidates[:, jj] == color)
        return jnp.sum(same & adjacent[None, :], axis=1) >= 2

    solved = connected(0) & connected(1) & connected(2)
    probability = jnp.sum(valid & solved) / jnp.maximum(jnp.sum(valid), 1)
    exact = _board_is_solved(obs.positions, colors).astype(jnp.float32)
    return jnp.where(unknown_count == 0, exact, probability.astype(jnp.float32))


def _observe_action(obs, colors, partner_seen, rng):
    legal = obs.legal_actions[1:10]
    noise = jax.random.uniform(rng, (9,), minval=0.0, maxval=1.0)
    value = (
        (colors < 0).astype(jnp.float32) * 100.0
        + (~partner_seen).astype(jnp.float32) * 10.0
        + noise
    )
    return (1 + jnp.argmax(jnp.where(legal, value, -1.0e9))).astype(jnp.int32)


def _move_action(obs, colors, partner_edges):
    k = jnp.arange(729, dtype=jnp.int32)
    moved = k // 81
    rem = k % 81
    destination = jnp.stack((rem // 9, rem % 9), axis=1)

    card_selector = jax.nn.one_hot(moved, 9, dtype=jnp.bool_)
    candidate_positions = jnp.where(
        card_selector[:, :, None], destination[:, None, :], obs.positions[None, :, :]
    )
    ii, jj = jnp.triu_indices(9, 1)
    distances = jnp.sum(
        jnp.abs(candidate_positions[:, ii] - candidate_positions[:, jj]), axis=2
    )
    adjacent = distances == 1
    ci, cj = colors[ii], colors[jj]
    known_pair = (ci >= 0) & (cj >= 0)
    counts = jnp.sum(
        colors[:, None] == jnp.arange(3, dtype=jnp.int8)[None, :], axis=0
    )
    remaining = (3 - counts).astype(jnp.float32)
    unknown_count = jnp.sum(colors < 0).astype(jnp.float32)
    denom = jnp.maximum(unknown_count, 1.0)
    unknown_pair_denom = jnp.maximum(unknown_count * (unknown_count - 1.0), 1.0)
    unknown_pair_equal = jnp.sum(remaining * (remaining - 1.0)) / unknown_pair_denom
    ci_safe = jnp.maximum(ci, 0)
    cj_safe = jnp.maximum(cj, 0)
    equal_probability = jnp.where(
        known_pair,
        (ci == cj).astype(jnp.float32),
        jnp.where(
            ci >= 0,
            remaining[ci_safe] / denom,
            jnp.where(cj >= 0, remaining[cj_safe] / denom, unknown_pair_equal),
        ),
    )
    completion_colors, completion_valid, completion_unknown = _completion_candidates(
        obs, colors
    )
    completion_count = jnp.sum(completion_valid)
    exact_equal_probability = jnp.sum(
        completion_valid[:, None]
        & (completion_colors[:, ii] == completion_colors[:, jj]),
        axis=0,
    ) / jnp.maximum(completion_count, 1)
    use_exact_marginal = (completion_unknown <= 5) & (completion_count > 0)
    equal_probability = jnp.where(
        use_exact_marginal, exact_equal_probability, equal_probability
    )

    # Directly score terminal feasibility late in the game.  A fixed loop over
    # the 27 unique completions for at most three unknown cards avoids a large
    # action-by-completion broadcast under batched evaluation.
    def accumulate_solved(n, totals):
        candidate = completion_colors[n]

        def connected(color):
            same = (candidate[ii] == color) & (candidate[jj] == color)
            return jnp.sum(adjacent & same[None, :], axis=1) >= 2

        solved = connected(0) & connected(1) & connected(2)
        return totals + (solved & completion_valid[n]).astype(jnp.float32)

    solved_counts = jax.lax.fori_loop(
        0, 27, accumulate_solved, jnp.zeros((729,), dtype=jnp.float32)
    )
    late_completion_count = jnp.sum(completion_valid[:27])
    candidate_solve_probability = jnp.where(
        completion_unknown <= 3,
        solved_counts / jnp.maximum(late_completion_count, 1),
        0.0,
    )

    same_edges = jnp.sum(adjacent * equal_probability[None, :], axis=1)
    same_distance = jnp.sum(distances * equal_probability[None, :], axis=1)
    different_edges = jnp.sum(
        adjacent * (1.0 - equal_probability[None, :]), axis=1
    )
    preserved_partner_edges = jnp.sum(
        adjacent & partner_edges[ii, jj][None, :], axis=1
    )
    total_distance = jnp.sum(distances, axis=1)
    moved_known = colors[moved] >= 0

    # The dominant term joins cards whose equality is direct local evidence.
    # Distance breaks plateaus; unlike a message code, this geometric intent is
    # legible to a partner following any ordinary clustering strategy.
    value = (
        100 * same_edges
        - 6 * same_distance
        - 20 * different_edges
        - total_distance
        + 4 * moved_known.astype(jnp.int32)
        + 96 * preserved_partner_edges
        + 300 * candidate_solve_probability
    )
    legal = obs.legal_actions[10:739]
    choice = jnp.argmax(jnp.where(legal, value, -1.0e9))
    return (10 + choice).astype(jnp.int32)


def _hint_action(obs, colors):
    reveal_legal = obs.legal_actions[739:743]
    reveal_action = (739 + jnp.argmax(reveal_legal)).astype(jnp.int32)
    can_reveal = jnp.any(reveal_legal)

    k = jnp.arange(324, dtype=jnp.int32)
    hint = k // 81
    rem = k % 81
    destination = jnp.stack((rem // 9, rem % 9), axis=1)
    target_matches = jnp.all(destination[:, None, :] == obs.positions[None, :, :], axis=2)
    target = jnp.argmax(target_matches, axis=1)
    target_color = colors[target]
    target_known = target_color >= 0
    safe_color = jnp.maximum(target_color, 0)
    hint_sets = obs.node_features[9:13, 3:6]
    compatible = target_known & (hint_sets[hint, safe_color] > 0)

    pos_distance = jnp.sum(
        jnp.abs(obs.positions[target, None, :] - obs.positions[None, :, :]), axis=2
    )
    same_neighbors = jnp.sum(
        (pos_distance == 1)
        & (colors[None, :] == target_color[:, None])
        & (target_color[:, None] >= 0),
        axis=1,
    )
    hint_size = jnp.sum(hint_sets[hint], axis=1)
    place_value = (
        1000 * compatible.astype(jnp.int32)
        + 30 * target_known.astype(jnp.int32)
        + 10 * same_neighbors
        - hint_size
    )
    place_legal = obs.legal_actions[743:1067]
    place_choice = jnp.argmax(jnp.where(place_legal, place_value, -1000000))
    place_action = (743 + place_choice).astype(jnp.int32)
    can_place = jnp.any(place_legal)
    revealed_count = jnp.sum(jnp.sum(hint_sets, axis=1) > 0)
    reveal_first_three = can_reveal & (revealed_count < 3)
    return jnp.where(reveal_first_three | (~can_place), reveal_action, place_action)


def agent_act(obs, reset, memory, rng):
    colors = _remember_colors(obs, reset, memory)
    substep = obs.node_features[0, 11].astype(jnp.int32)
    active = obs.node_features[0, 10] > 0
    solved = _board_is_solved(obs.positions, colors)
    solve_probability = _solve_probability(obs, colors)
    placed_count = jnp.sum(obs.node_features[9:13, 8] > 0)
    late_feasible = (placed_count >= 3) & (solve_probability > 0.0)
    penultimate_likely = (placed_count == 2) & (solve_probability > 0.19)
    turn_six_likely = (placed_count == 1) & (solve_probability > 0.31)
    confident_solved = solved | late_feasible | penultimate_likely | turn_six_likely

    old_partner_seen = jnp.where(
        reset, jnp.zeros((9,), dtype=jnp.bool_), memory["partner_seen"]
    )
    reported_partner_seen = obs.node_features[:9, 9] > 0
    partner_seen = old_partner_seen | ((~active) & reported_partner_seen)

    old_positions = jnp.where(reset, obs.positions, memory["positions"])
    changed_cards = jnp.any(obs.positions != old_positions, axis=1)
    partner_move_seen = (~active) & jnp.any(changed_cards) & (~reset)
    partner_card = jnp.argmax(changed_cards)
    distances_now = jnp.sum(
        jnp.abs(obs.positions[:, None, :] - obs.positions[None, :, :]), axis=2
    )
    incident = (
        (jnp.arange(9)[:, None] == partner_card)
        | (jnp.arange(9)[None, :] == partner_card)
    )
    partner_knows_pair = partner_seen[:, None] & partner_seen[None, :]
    new_partner_edges = (distances_now == 1) & incident & partner_knows_pair
    old_partner_edges = jnp.where(
        reset, jnp.zeros((9, 9), dtype=jnp.bool_), memory["partner_edges"]
    )
    partner_edges = jnp.where(partner_move_seen, new_partner_edges, old_partner_edges)

    observe_action = _observe_action(obs, colors, partner_seen, rng)
    move_action = _move_action(obs, colors, partner_edges)
    hint_action = _hint_action(obs, colors)
    action = jax.lax.switch(
        substep,
        (
            lambda: jnp.where(confident_solved, jnp.int32(1067), observe_action),
            lambda: observe_action,
            lambda: move_action,
            lambda: hint_action,
        ),
    )
    fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    action = jnp.where(active & obs.legal_actions[action], action, fallback)
    trace = {
        "known_colors": colors,
        "locally_solved": solved,
        "solve_probability": solve_probability,
        "substep": substep,
    }
    next_memory = {
        "colors": colors,
        "partner_seen": partner_seen,
        "positions": obs.positions,
        "partner_edges": partner_edges,
    }
    return action, next_memory, trace
