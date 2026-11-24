"""Locally grounded greedy clustering policy for two-player Yokai."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'exact_hint_filter_v20', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

PERMUTATIONS_4 = (
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1],
    [0, 3, 1, 2], [0, 3, 2, 1], [1, 0, 2, 3], [1, 0, 3, 2],
    [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0],
    [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 0, 2, 1],
    [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
)


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _known_color_connected_flags(colors, known, adjacency):
    """Locally certified completeness and connectivity for each color."""
    card_adj = adjacency[:16, :16].astype(jnp.bool_)

    def one_color(c):
        members = known & (colors == c)
        start = jnp.argmax(members.astype(jnp.int32))
        reached = jnp.arange(16) == start

        def expand(_, reach):
            return reach | jnp.any(card_adj & reach[:, None] & members[:, None], axis=0)

        reached = jax.lax.fori_loop(0, 16, expand, reached)
        return (jnp.sum(members) == 4) & jnp.all((~members) | reached)

    return jax.vmap(one_color)(jnp.arange(4))


def _guaranteed_solved_with_four_unknown(colors, known, constraint_oh, adjacency):
    """True only if every multiset-consistent completion is solved."""
    unknown_count = jnp.sum(~known)
    unknown_ids = jnp.nonzero(~known, size=4, fill_value=0)[0]
    codes = jnp.arange(256, dtype=jnp.int32)
    powers = jnp.asarray([1, 4, 16, 64], dtype=jnp.int32)
    assignments = (codes[:, None] // powers[None, :]) % 4
    completions = jnp.broadcast_to(colors[None, :], (256, 16))
    completions = completions.at[:, unknown_ids].set(assignments)
    counts = jnp.sum(jax.nn.one_hot(completions, 4, dtype=jnp.int32), axis=1)
    valid = jnp.all(counts == 4, axis=1)
    unknown_constraints = constraint_oh[unknown_ids]
    constrained = jnp.sum(unknown_constraints, axis=1) > 0
    allowed = jax.vmap(
        lambda assignment: unknown_constraints[jnp.arange(4), assignment]
    )(assignments) > 0
    valid = valid & jnp.all((~constrained[None, :]) | allowed, axis=1)
    members = completions[:, None, :] == jnp.arange(4)[None, :, None]
    card_adj = adjacency[:16, :16].astype(jnp.bool_)
    start = jnp.argmax(members.astype(jnp.int32), axis=2)
    reached = jnp.arange(16)[None, None, :] == start[:, :, None]

    def expand(_, reach):
        neighbors = jnp.any(card_adj[None, None, :, :] & reach[:, :, :, None], axis=2)
        return reach | (neighbors & members)

    reached = jax.lax.fori_loop(0, 16, expand, reached)
    completion_solved = jnp.all((~members) | reached, axis=(1, 2))
    valid_count = jnp.sum(valid).astype(jnp.int32)
    solved_count = jnp.sum(valid & completion_solved).astype(jnp.int32)
    guaranteed = ((unknown_count == 4) & (valid_count > 0)
                  & (solved_count == valid_count))
    return guaranteed, solved_count, valid_count


def _late_move_solution_count(colors, known, color_oh, constraint_oh,
                              positions, moved, dest_x, dest_y):
    """Number of the two possible late color completions solved per move."""
    remaining = 4 - jnp.sum(color_oh, axis=0)
    color_ids = jnp.arange(4, dtype=jnp.int32)
    first = jnp.argmax((remaining > 0).astype(jnp.int32))
    second = jnp.where(
        remaining[first] >= 2,
        first,
        jnp.argmax(((remaining > 0) & (color_ids != first)).astype(jnp.int32)),
    )
    unknown_ids = jnp.nonzero(~known, size=2, fill_value=0)[0]
    completions = jnp.broadcast_to(colors[None, :], (2, 16))
    assignments = jnp.asarray([[first, second], [second, first]], dtype=jnp.int32)
    completions = completions.at[:, unknown_ids].set(assignments)
    unknown_constraints = constraint_oh[unknown_ids]
    constrained = jnp.sum(unknown_constraints, axis=1) > 0
    allowed = jax.vmap(
        lambda assignment: unknown_constraints[jnp.arange(2), assignment]
    )(assignments) > 0
    consistent = jnp.all((~constrained[None, :]) | allowed, axis=1)

    candidate_positions = jnp.broadcast_to(positions[None, :, :], (1600, 16, 2))
    rows = jnp.arange(1600, dtype=jnp.int32)
    destinations = jnp.stack((dest_x, dest_y), axis=1)
    candidate_positions = candidate_positions.at[rows, moved, :].set(destinations)

    def completion_score(completion):
        def color_connectivity(c):
            card_ids = jnp.nonzero(completion == c, size=4, fill_value=0)[0]
            color_positions = candidate_positions[:, card_ids, :]
            adjacency = (
                jnp.abs(color_positions[:, :, None, 0] - color_positions[:, None, :, 0])
                + jnp.abs(color_positions[:, :, None, 1] - color_positions[:, None, :, 1]) == 1
            )
            reached = adjacency | jnp.eye(4, dtype=jnp.bool_)[None, :, :]

            def expand(_, reach):
                return reach | jnp.any(
                    reach[:, :, :, None] & adjacency[:, None, :, :], axis=2)

            reached = jax.lax.fori_loop(0, 4, expand, reached)
            return (jnp.all(reached, axis=(1, 2)),
                    jnp.sum(reached, axis=(1, 2)).astype(jnp.int32))

        connected, reachability = jax.vmap(color_connectivity)(color_ids)
        world_solved = jnp.all(connected, axis=0).astype(jnp.int32)
        return 1000 * world_solved + jnp.sum(reachability, axis=0)

    scores = jax.vmap(completion_score)(completions).astype(jnp.int32)
    # When both missing cards have the same color, the two rows are the same
    # completion and intentionally receive equal weight (a maximum of two).
    return jnp.sum(scores * consistent[:, None].astype(jnp.int32), axis=0)


def _four_unknown_connectivity_score(colors, known, color_oh, constraint_oh,
                                     positions, moved, dest_x, dest_y):
    """Sum of connected color classes over all remaining-slot permutations."""
    remaining = 4 - jnp.sum(color_oh, axis=0)
    cumulative = jnp.cumsum(remaining)
    slots = jax.vmap(
        lambda k: jnp.argmax((cumulative > k).astype(jnp.int32))
    )(jnp.arange(4, dtype=jnp.int32))
    assignments = slots[jnp.asarray(PERMUTATIONS_4, dtype=jnp.int32)]
    unknown_ids = jnp.nonzero(~known, size=4, fill_value=0)[0]
    completions = jnp.broadcast_to(colors[None, :], (24, 16))
    completions = completions.at[:, unknown_ids].set(assignments)
    unknown_constraints = constraint_oh[unknown_ids]
    constrained = jnp.sum(unknown_constraints, axis=1) > 0
    allowed = jax.vmap(
        lambda assignment: unknown_constraints[jnp.arange(4), assignment]
    )(assignments) > 0
    consistent = jnp.all((~constrained[None, :]) | allowed, axis=1)

    candidate_positions = jnp.broadcast_to(positions[None, :, :], (1600, 16, 2))
    rows = jnp.arange(1600, dtype=jnp.int32)
    destinations = jnp.stack((dest_x, dest_y), axis=1)
    candidate_positions = candidate_positions.at[rows, moved, :].set(destinations)
    color_ids = jnp.arange(4, dtype=jnp.int32)

    def add_completion(k, total):
        completion = completions[k]

        def color_connected(c):
            card_ids = jnp.nonzero(completion == c, size=4, fill_value=0)[0]
            color_positions = candidate_positions[:, card_ids, :]
            adjacency = (
                jnp.abs(color_positions[:, :, None, 0] - color_positions[:, None, :, 0])
                + jnp.abs(color_positions[:, :, None, 1] - color_positions[:, None, :, 1]) == 1
            )
            reached = adjacency | jnp.eye(4, dtype=jnp.bool_)[None, :, :]

            def expand(_, reach):
                return reach | jnp.any(
                    reach[:, :, :, None] & adjacency[:, None, :, :], axis=2)

            reached = jax.lax.fori_loop(0, 4, expand, reached)
            connected = jnp.all(reached, axis=(1, 2)).astype(jnp.int32)
            pair_reachability = jnp.sum(reached, axis=(1, 2)).astype(jnp.int32)
            return 1000 * connected + pair_reachability

        completion_score = jnp.sum(
            jax.vmap(color_connected)(color_ids), axis=0)
        return total + consistent[k].astype(jnp.int32) * completion_score

    return jax.lax.fori_loop(
        0, 24, add_completion, jnp.zeros((1600,), dtype=jnp.int32))


def _six_unknown_connectivity_score(colors, known, color_oh, constraint_oh,
                                    positions, moved, dest_x, dest_y, rng):
    """Monte Carlo same-color reachability for six unseen cards."""
    remaining = 4 - jnp.sum(color_oh, axis=0)
    cumulative = jnp.cumsum(remaining)
    slots = jax.vmap(
        lambda k: jnp.argmax((cumulative > k).astype(jnp.int32))
    )(jnp.arange(6, dtype=jnp.int32))
    keys = jax.random.split(rng, 24)
    permutation_ids = jax.vmap(
        lambda key: jax.random.permutation(key, jnp.arange(6, dtype=jnp.int32))
    )(keys)
    assignments = slots[permutation_ids]
    unknown_ids = jnp.nonzero(~known, size=6, fill_value=0)[0]
    completions = jnp.broadcast_to(colors[None, :], (24, 16))
    completions = completions.at[:, unknown_ids].set(assignments)
    unknown_constraints = constraint_oh[unknown_ids]
    constrained = jnp.sum(unknown_constraints, axis=1) > 0
    allowed = jax.vmap(
        lambda assignment: unknown_constraints[jnp.arange(6), assignment]
    )(assignments) > 0
    consistent = jnp.all((~constrained[None, :]) | allowed, axis=1)

    candidate_positions = jnp.broadcast_to(positions[None, :, :], (1600, 16, 2))
    rows = jnp.arange(1600, dtype=jnp.int32)
    destinations = jnp.stack((dest_x, dest_y), axis=1)
    candidate_positions = candidate_positions.at[rows, moved, :].set(destinations)
    color_ids = jnp.arange(4, dtype=jnp.int32)

    def add_completion(k, total):
        completion = completions[k]

        def color_reachability(c):
            card_ids = jnp.nonzero(completion == c, size=4, fill_value=0)[0]
            color_positions = candidate_positions[:, card_ids, :]
            adjacency = (
                jnp.abs(color_positions[:, :, None, 0] - color_positions[:, None, :, 0])
                + jnp.abs(color_positions[:, :, None, 1] - color_positions[:, None, :, 1]) == 1
            )
            reached = adjacency | jnp.eye(4, dtype=jnp.bool_)[None, :, :]

            def expand(_, reach):
                return reach | jnp.any(
                    reach[:, :, :, None] & adjacency[:, None, :, :], axis=2)

            reached = jax.lax.fori_loop(0, 4, expand, reached)
            return jnp.sum(reached, axis=(1, 2)).astype(jnp.int32)

        completion_score = jnp.sum(
            jax.vmap(color_reachability)(color_ids), axis=0)
        return total + consistent[k].astype(jnp.int32) * completion_score

    return jax.lax.fori_loop(
        0, 24, add_completion, jnp.zeros((1600,), dtype=jnp.int32))


def agent_act(obs, reset, memory, rng):
    features = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    active = features[0, 12] == 1
    substep = features[0, 13]
    personal_color_oh = features[:16, :4]
    personal_known = jnp.sum(personal_color_oh, axis=1) > 0
    personal_colors = jnp.argmax(personal_color_oh, axis=1)
    hint_sets_all = features[16:23, 4:8] > 0
    hint_placed = features[16:23, 10] > 0
    singleton = hint_placed & (jnp.sum(hint_sets_all, axis=1) == 1)
    hint_positions = features[16:23, 8:10]
    hint_on_card = jnp.all(
        hint_positions[:, None, :] == features[None, :16, 8:10], axis=2)
    placed_constraint_oh = jnp.max(
        (hint_placed[:, None, None] & hint_on_card[:, :, None]
         & hint_sets_all[:, None, :]).astype(jnp.int32), axis=0)
    inferred_color_oh = jnp.max(
        (singleton[:, None, None] & hint_on_card[:, :, None]
         & hint_sets_all[:, None, :]).astype(jnp.int32), axis=0)
    color_oh = jnp.where(personal_known[:, None], personal_color_oh,
                         inferred_color_oh)
    known = jnp.sum(color_oh, axis=1) > 0
    colors = jnp.argmax(color_oh, axis=1)
    partner_seen = features[:16, 11] > 0
    positions = obs.positions.astype(jnp.int32)
    card_adj = obs.adjacency_matrix[:16, :16].astype(jnp.bool_)
    ids = jnp.arange(16, dtype=jnp.int32)

    # Cover globally unseen cards on both observations while any remain.
    observe_legal = legal[1:17]
    base_observe = (~known).astype(jnp.int32) * 1000
    fresh_bonus = (~partner_seen).astype(jnp.int32) * 40
    degree = jnp.sum(card_adj, axis=1).astype(jnp.int32) - 1
    observe_score = base_observe + fresh_bonus
    observe_score = observe_score + (4 - degree) * 2 - ids
    observe_score = jnp.where(observe_legal, observe_score, -100000)
    observe_action = 1 + jnp.argmax(observe_score)

    # Evaluate all encoded relocations using only locally known colors.
    move_offsets = jnp.arange(1600, dtype=jnp.int32)
    moved = move_offsets // 100
    cell = move_offsets % 100
    dest_x = cell // 10
    dest_y = cell % 10
    other = jnp.arange(16, dtype=jnp.int32)[None, :]
    new_near = (jnp.abs(dest_x[:, None] - positions[None, :, 0])
                + jnp.abs(dest_y[:, None] - positions[None, :, 1]) == 1)
    new_near = new_near & (other != moved[:, None])
    old_near = card_adj[moved, :] & (other != moved[:, None])
    moved_known = known[moved]
    other_known = known[None, :]
    same = (colors[moved, None] == colors[None, :]) & moved_known[:, None] & other_known
    different = (colors[moved, None] != colors[None, :]) & moved_known[:, None] & other_known
    remaining = 4.0 - jnp.sum(color_oh.astype(jnp.float32), axis=0)
    unknown_prior = remaining / jnp.maximum(jnp.sum(remaining), 1.0)
    has_constraint = jnp.sum(placed_constraint_oh, axis=1) > 0
    allowed_colors = jnp.where(
        has_constraint[:, None], placed_constraint_oh,
        jnp.ones_like(placed_constraint_oh)).astype(jnp.float32)
    constrained_prior = allowed_colors * unknown_prior[None, :]
    constrained_prior = constrained_prior / jnp.maximum(
        jnp.sum(constrained_prior, axis=1, keepdims=True), 1.0)
    beliefs = jnp.where(
        known[:, None], color_oh.astype(jnp.float32), constrained_prior)
    match_probability = jnp.sum(beliefs[moved, None, :] * beliefs[None, :, :], axis=2)
    new_same = jnp.sum(new_near & same, axis=1).astype(jnp.int32)
    old_same = jnp.sum(old_near & same, axis=1).astype(jnp.int32)
    new_diff = jnp.sum(new_near & different, axis=1).astype(jnp.int32)
    new_degree = jnp.sum(new_near, axis=1).astype(jnp.int32)
    old_degree = jnp.sum(old_near, axis=1).astype(jnp.int32)
    expected_delta = jnp.sum(
        (new_near.astype(jnp.float32) - old_near.astype(jnp.float32))
        * match_probability, axis=1)
    preserved_same = jnp.sum(new_near & old_near & same, axis=1).astype(jnp.int32)
    abandoned_cluster = (old_same > 0) & (preserved_same == 0)

    # Label same-color components after removing each possible moved card.
    # This lets a candidate value the full component(s) it joins, rather than
    # treating two redundant neighbors as better than one large component.
    known_same_graph = (card_adj & known[:, None] & known[None, :]
                        & (colors[:, None] == colors[None, :]))
    removed = jnp.eye(16, dtype=jnp.bool_)
    graphs = (known_same_graph[None, :, :] & (~removed[:, :, None])
              & (~removed[:, None, :]))
    labels = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (16, 16))

    def propagate(_, current_labels):
        neighbor_labels = jnp.where(graphs, current_labels[:, None, :], 99)
        return jnp.minimum(current_labels, jnp.min(neighbor_labels, axis=2))

    labels = jax.lax.fori_loop(0, 16, propagate, labels)
    component_sizes = jnp.sum(labels[:, :, None] == labels[:, None, :], axis=2).astype(jnp.int32)
    rows = jnp.arange(1600, dtype=jnp.int32)

    def joined_size(neighbor_mask):
        def add_neighbor(j, state):
            seen_labels, total = state
            label = labels[moved, j]
            valid = neighbor_mask[:, j] & same[:, j]
            already = seen_labels[rows, label]
            add = valid & (~already)
            total = total + jnp.where(add, component_sizes[moved, j], 0)
            seen_labels = seen_labels.at[rows, label].set(already | valid)
            return seen_labels, total

        initial = (jnp.zeros((1600, 16), dtype=jnp.bool_),
                   jnp.zeros((1600,), dtype=jnp.int32))
        return jax.lax.fori_loop(0, 16, add_neighbor, initial)[1]

    component_delta = joined_size(new_near) - joined_size(old_near)
    move_score = 120 * (new_same - old_same) + 18 * new_same
    move_score = move_score + jnp.rint(30.0 * expected_delta).astype(jnp.int32)
    move_score = move_score + 24 * ((old_same == 0) & (new_same > 0)).astype(jnp.int32)
    move_score = move_score - 18 * abandoned_cluster.astype(jnp.int32)
    move_score = move_score + 35 * component_delta
    move_score = move_score - 4 * new_diff + 3 * new_degree - old_degree
    move_score = move_score - (~moved_known).astype(jnp.int32) * 20
    move_score = move_score - jnp.abs(dest_x - 4) - jnp.abs(dest_y - 4)
    late_solution_count = _late_move_solution_count(
        personal_colors, personal_known, personal_color_oh,
        placed_constraint_oh, positions, moved, dest_x, dest_y)
    move_score = move_score + jnp.where(jnp.sum(personal_known) == 14,
                                        10000 * late_solution_count, 0)
    four_unknown_score = jax.lax.cond(
        jnp.sum(personal_known) == 12,
        lambda _: _four_unknown_connectivity_score(
            personal_colors, personal_known, personal_color_oh,
            placed_constraint_oh, positions, moved, dest_x, dest_y),
        lambda _: jnp.zeros((1600,), dtype=jnp.int32),
        operand=None,
    )
    move_score = move_score + 1000 * four_unknown_score
    six_unknown_score = jax.lax.cond(
        jnp.sum(personal_known) == 10,
        lambda _: _six_unknown_connectivity_score(
            personal_colors, personal_known, personal_color_oh,
            placed_constraint_oh, positions, moved, dest_x, dest_y, rng),
        lambda _: jnp.zeros((1600,), dtype=jnp.int32),
        operand=None,
    )
    move_score = move_score + six_unknown_score // 4
    move_score = jnp.where(legal[17:1617], move_score, -100000)
    move_action = 17 + jnp.argmax(move_score)

    # Reveal every hint before placement, then lock a known compatible card.
    reveal_legal = legal[1617:1624]
    has_reveal = jnp.any(reveal_legal)
    reveal_action = 1617 + jnp.argmax(reveal_legal.astype(jnp.int32))

    place_offsets = jnp.arange(700, dtype=jnp.int32)
    hint = place_offsets // 100
    pcell = place_offsets % 100
    px = pcell // 10
    py = pcell % 10
    at_cell = ((positions[None, :, 0] == px[:, None])
               & (positions[None, :, 1] == py[:, None]))
    target = jnp.argmax(at_cell.astype(jnp.int32), axis=1)
    target_known = known[target]
    hint_sets = hint_sets_all
    compatible = hint_sets[hint, colors[target]]
    target_same_neighbors = jnp.sum(
        card_adj[target, :] & known[None, :]
        & (colors[target, None] == colors[None, :]), axis=1).astype(jnp.int32)
    hint_size = jnp.sum(hint_sets[hint], axis=1).astype(jnp.int32)
    place_score = jnp.where(target_known, jnp.where(compatible, 1000, -1000), 20 * hint_size)
    completed_colors = _known_color_connected_flags(
        colors, known, obs.adjacency_matrix)
    place_score = place_score + 400 * (target_known & completed_colors[colors[target]]).astype(jnp.int32)
    place_score = place_score + 30 * target_same_neighbors + degree[target] - target
    place_score = jnp.where(legal[1624:2324], place_score, -100000)
    place_action = 1624 + jnp.argmax(place_score)
    hint_action = jnp.where(has_reveal, reveal_action, place_action)

    personal_completed_colors = _known_color_connected_flags(
        personal_colors, personal_known, obs.adjacency_matrix)
    fully_known_solved = jnp.all(personal_known) & jnp.all(personal_completed_colors)
    guaranteed_partial, solved_completions, valid_completions = _guaranteed_solved_with_four_unknown(
        personal_colors, personal_known, placed_constraint_oh,
        obs.adjacency_matrix)
    solved = fully_known_solved | guaranteed_partial
    step0_action = jnp.where(solved & legal[2324], jnp.int32(2324), observe_action)
    active_action = jax.lax.switch(
        substep, (lambda: step0_action, lambda: observe_action,
                  lambda: move_action, lambda: hint_action))
    action = jnp.where(active, active_action, jnp.int32(0)).astype(jnp.int32)
    next_memory = jnp.where(reset, jnp.int32(0), memory)
    trace = {
        'known_count': jnp.sum(personal_known).astype(jnp.int32),
        'inferred_count': jnp.sum(known & (~personal_known)).astype(jnp.int32),
        'guaranteed_partial': guaranteed_partial,
        'solved_local': solved,
    }
    return action, next_memory, trace
