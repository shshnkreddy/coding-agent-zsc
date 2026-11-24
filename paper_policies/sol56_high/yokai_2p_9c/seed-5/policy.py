"""Convention-oriented local policy for the two-player 9-card Yokai game."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'canonical_row_clusterer',
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
    return {'colors': jnp.full((9,), -1, dtype=jnp.int8)}


def agent_act(obs, reset, memory, rng):
    del rng
    nf = obs.node_features
    legal = obs.legal_actions
    positions = obs.positions

    old_colors = jnp.where(
        reset, jnp.full((9,), -1, dtype=jnp.int8), memory['colors'])
    visible = jnp.sum(nf[:9, :3], axis=1) > 0
    visible_colors = jnp.argmax(nf[:9, :3], axis=1).astype(jnp.int8)
    colors = jnp.where(visible, visible_colors, old_colors)
    next_memory = {'colors': colors}

    # A placed singleton hint is an exact public label when competent partners
    # place compatible hints.  Use it for planning even if the lock prevented
    # this seat from observing that card directly.
    observed_known = colors >= 0
    hint0_matches = jnp.all(
        positions == nf[9, 6:8].astype(jnp.int32)[None, :], axis=1)
    singleton_labeled = hint0_matches & (nf[9, 8] == 1)
    singleton_color = jnp.argmax(nf[9, 3:6]).astype(jnp.int8)
    evidence_colors = jnp.where(
        (~observed_known) & singleton_labeled, singleton_color, colors)

    # With the fixed 3-3-3 deck, eight exact labels determine the ninth card.
    evidence_known = evidence_colors >= 0
    color_counts = jnp.stack(
        [jnp.sum(evidence_colors == c) for c in range(3)])
    missing_color = jnp.argmax(3 - color_counts).astype(jnp.int8)
    exactly_one_unknown = jnp.sum(~evidence_known) == 1
    plan_colors = jnp.where(
        (~evidence_known) & exactly_one_unknown, missing_color, evidence_colors)

    # At the start of a late turn each seat knows six cards.  Enumerate the
    # only 27 assignments to the remaining three, retain exact 3-3-3 decks,
    # and end only when *every* possible true layout is already successful.
    unknown_ids = jnp.nonzero(~observed_known, size=3, fill_value=0)[0]
    codes = jnp.arange(27, dtype=jnp.int8)
    completions = jnp.stack(
        (codes // 9, (codes // 3) % 3, codes % 3), axis=1)
    assignments = jnp.broadcast_to(colors, (27, 9))
    assignments = assignments.at[:, unknown_ids].set(completions)
    completion_counts = jnp.stack(
        [jnp.sum(assignments == c, axis=1) for c in range(3)], axis=1)
    hint_nodes = nf[9:13]
    hint_target_match = jnp.all(
        hint_nodes[:, None, 6:8].astype(jnp.int32)
        == positions[None, :, :], axis=2)
    hinted_cards = jnp.argmax(hint_target_match.astype(jnp.int32), axis=1)
    assignment_hint_colors = assignments[:, hinted_cards]
    hint_membership = jnp.sum(
        jnp.eye(3, dtype=jnp.int8)[assignment_hint_colors]
        * hint_nodes[None, :, 3:6], axis=2) > 0
    respects_placed_hints = jnp.all(
        (hint_nodes[None, :, 8] == 0) | hint_membership, axis=1)
    valid_completion = (
        jnp.all(completion_counts == 3, axis=1) & respects_placed_hints)
    valid_total = jnp.sum(valid_completion)
    completion_equal = assignments[:, :, None] == assignments[:, None, :]
    completion_same_probability = jnp.sum(
        completion_equal.astype(jnp.float32)
        * valid_completion[:, None, None].astype(jnp.float32), axis=0
    ) / jnp.maximum(valid_total.astype(jnp.float32), 1.0)
    completion_same_probability = completion_same_probability * (
        ~jnp.eye(9, dtype=jnp.bool_)).astype(jnp.float32)
    completion_votes = jnp.stack([
        jnp.sum(valid_completion[:, None] & (assignments == c), axis=0)
        for c in range(3)
    ], axis=1)
    consensus_color = jnp.argmax(completion_votes, axis=1).astype(jnp.int8)
    consensus_known = (
        (jnp.sum(observed_known) == 6) & (valid_total > 0)
        & (jnp.max(completion_votes, axis=1) == valid_total))
    plan_colors = jnp.where(
        (plan_colors < 0) & consensus_known, consensus_color, plan_colors)
    plan_known_after_hints = plan_colors >= 0
    plan_counts_after_hints = jnp.stack(
        [jnp.sum(plan_colors == c) for c in range(3)])
    last_plan_color = jnp.argmax(3 - plan_counts_after_hints).astype(jnp.int8)
    plan_colors = jnp.where(
        (plan_colors < 0) & (jnp.sum(~plan_known_after_hints) == 1),
        last_plan_color, plan_colors)
    card_adjacency = obs.adjacency_matrix[:9, :9].astype(jnp.bool_)
    card_adjacency = card_adjacency & (~jnp.eye(9, dtype=jnp.bool_))
    connected_by_color = []
    for c in range(3):
        members = assignments == c
        directed_edges = jnp.sum(
            members[:, :, None] & members[:, None, :]
            & card_adjacency[None, :, :], axis=(1, 2))
        connected_by_color.append(directed_edges >= 4)
    successful_completion = jnp.all(jnp.stack(connected_by_color, axis=1), axis=1)
    posterior_success = jnp.sum(
        valid_completion & successful_completion).astype(jnp.float32) / jnp.maximum(
            valid_total.astype(jnp.float32), 1.0)
    can_end_six = (
        (jnp.sum(observed_known) == 6)
        & jnp.any(valid_completion)
        & (~jnp.any(valid_completion & (~successful_completion))))
    canonical_labels = positions[:, 0] - 3
    canonical_in_range = jnp.all(
        (canonical_labels >= 0) & (canonical_labels < 3))
    canonical_counts = jnp.stack([
        jnp.sum(canonical_labels == c) for c in range(3)])
    canonical_connected = []
    for c in range(3):
        row_members = canonical_labels == c
        row_edges = jnp.sum(
            row_members[:, None] & row_members[None, :] & card_adjacency)
        canonical_connected.append(row_edges >= 4)
    canonical_observations_agree = jnp.all(
        (~observed_known) | (colors == canonical_labels))
    hinted_canonical_labels = jnp.clip(
        canonical_labels[hinted_cards], 0, 2).astype(jnp.int32)
    canonical_hint_membership = jnp.sum(
        jnp.eye(3, dtype=jnp.int8)[hinted_canonical_labels]
        * hint_nodes[:, 3:6], axis=1) > 0
    canonical_hints_agree = jnp.all(
        (hint_nodes[:, 8] == 0) | canonical_hint_membership)
    locked_canonical_rows = jnp.stack([
        jnp.any((canonical_labels == c) & (nf[:9, 8] == 1))
        for c in range(3)])
    placed_hint_count = jnp.sum(hint_nodes[:, 8] == 1)
    shown_hint_count = jnp.sum(jnp.any(hint_nodes[:, 3:6] > 0, axis=1))
    revealed_unplaced_count = shown_hint_count - placed_hint_count
    unrevealed_hint_count = 4 - shown_hint_count
    successful_score_now = (
        5 * unrevealed_hint_count + 2 * revealed_unplaced_count
        + placed_hint_count).astype(jnp.float32)
    canonical_public_evidence = (
        (placed_hint_count >= 3)
        | ((placed_hint_count == 2) & (jnp.sum(locked_canonical_rows) >= 2)))
    canonical_end = (
        (jnp.sum(observed_known) == 6) & canonical_in_range
        & canonical_public_evidence
        & jnp.all(canonical_counts == 3)
        & jnp.all(jnp.stack(canonical_connected))
        & canonical_observations_agree & canonical_hints_agree)
    posterior_value_end = (
        (jnp.sum(observed_known) == 6) & (valid_total > 0)
        & (posterior_success * successful_score_now > 3.9))
    can_end = can_end_six | posterior_value_end

    actions = jnp.arange(1068, dtype=jnp.int32)
    substep = nf[0, 11].astype(jnp.int32)
    active = nf[0, 10] == 1

    # Spread observations across cards not yet seen by either player.
    observe_card = jnp.clip(actions - 1, 0, 8)
    is_observe = (actions >= 1) & (actions < 10)
    obs_unknown = colors[observe_card] < 0
    partner_unseen = nf[observe_card, 9] == 0
    observe_score = (
        jnp.where(is_observe, 1000.0, -1.0e7)
        + 300.0 * obs_unknown.astype(jnp.float32)
        + 60.0 * partner_unseen.astype(jnp.float32)
        - observe_card.astype(jnp.float32)
    )

    # Score every possible post-move layout. Known same-color cards are pulled
    # together, and color IDs map to x rows 3, 4, 5 as a public convention.
    move_card = jnp.clip((actions - 10) // 81, 0, 8)
    move_offset = jnp.mod(jnp.maximum(actions - 10, 0), 81)
    move_dest = jnp.stack((move_offset // 9, move_offset % 9), axis=1)
    known = plan_colors >= 0
    same = (
        known[:, None]
        & known[None, :]
        & (plan_colors[:, None] == plan_colors[None, :])
        & (~jnp.eye(9, dtype=jnp.bool_))
    )
    card_dist = jnp.sum(
        jnp.abs(positions[:, None, :] - positions[None, :, :]), axis=-1)
    remaining_counts = jnp.maximum(3 - color_counts, 0).astype(jnp.float32)
    unknown_total = jnp.sum(remaining_counts)
    unknown_marginal = remaining_counts / jnp.maximum(unknown_total, 1.0)
    label_marginals = jnp.where(
        known[:, None],
        jnp.eye(3, dtype=jnp.float32)[jnp.clip(plan_colors, 0, 2)],
        unknown_marginal[None, :])
    count_same_probability = label_marginals @ label_marginals.T
    both_unknown = (~known[:, None]) & (~known[None, :])
    two_unknown_same = jnp.sum(
        remaining_counts * jnp.maximum(remaining_counts - 1.0, 0.0)
    ) / jnp.maximum(unknown_total * (unknown_total - 1.0), 1.0)
    count_same_probability = jnp.where(
        both_unknown, two_unknown_same, count_same_probability)
    count_same_probability = count_same_probability * (
        ~jnp.eye(9, dtype=jnp.bool_)).astype(jnp.float32)
    hard_same = same.astype(jnp.float32)
    same_f = jnp.where(
        (jnp.sum(observed_known) == 6) & (valid_total > 0),
        completion_same_probability,
        jnp.where(
            (jnp.sum(observed_known) == 4) | (jnp.sum(observed_known) == 7),
            count_same_probability,
            hard_same))
    base_adjacency = jnp.sum((card_dist == 1).astype(jnp.float32) * same_f)
    base_distance = jnp.sum(jnp.minimum(card_dist, 6).astype(jnp.float32) * same_f)
    new_distance = jnp.sum(
        jnp.abs(move_dest[:, None, :] - positions[None, :, :]), axis=-1)
    mover_same = same_f[move_card, :]
    old_mover_distance = card_dist[move_card, :]
    adjacency_delta = 2.0 * jnp.sum(
        ((new_distance == 1).astype(jnp.float32)
         - (old_mover_distance == 1).astype(jnp.float32)) * mover_same,
        axis=1)
    adjacency_value = base_adjacency + adjacency_delta
    mover_known = known[move_card]
    base_color_edges = jnp.stack([
        jnp.sum((card_dist == 1) * same * (plan_colors[:, None] == c))
        for c in range(3)
    ]).astype(jnp.float32)
    mover_color = jnp.clip(plan_colors[move_card], 0, 2).astype(jnp.int32)
    color_edge_delta = (
        jnp.eye(3, dtype=jnp.float32)[mover_color]
        * adjacency_delta[:, None] * mover_known[:, None])
    completed_colors = jnp.sum(
        (base_color_edges[None, :] + color_edge_delta) >= 4.0, axis=1)
    distance_cost = base_distance + 2.0 * jnp.sum(
        (jnp.minimum(new_distance, 6).astype(jnp.float32)
         - jnp.minimum(old_mover_distance, 6).astype(jnp.float32)) * mover_same,
        axis=1)
    target_rows = (3 + jnp.maximum(plan_colors, 0)).astype(jnp.int32)
    base_row_cost = jnp.sum(
        jnp.abs(positions[:, 0] - target_rows) * known.astype(jnp.int32))
    row_cost = (base_row_cost
        + (jnp.abs(move_dest[:, 0] - target_rows[move_card])
           - jnp.abs(positions[move_card, 0] - target_rows[move_card]))
          * known[move_card].astype(jnp.int32)).astype(jnp.float32)
    is_move = (actions >= 10) & (actions < 739)
    observed_count = jnp.sum(observed_known)
    row_weight = jnp.where(
        observed_count == 6, 9.0,
        jnp.where(observed_count == 4, 8.5, 7.0))
    move_score = (
        45.0 * adjacency_value + 60.0 * completed_colors
        - 1.0 * distance_cost - row_weight * row_cost
        + 25.0 * mover_known.astype(jnp.float32)
        - 0.0001 * actions.astype(jnp.float32))
    move_score = jnp.where(is_move, move_score, -1.0e7)

    # Reveal hints before locking. Once placements are forced, use a locally
    # known compatible card and prefer a stable, already clustered anchor.
    is_reveal = (actions >= 739) & (actions < 743)
    is_place = (actions >= 743) & (actions < 1067)
    hint_id = jnp.clip((actions - 743) // 81, 0, 3)
    place_offset = jnp.mod(jnp.maximum(actions - 743, 0), 81)
    place_xy = jnp.stack((place_offset // 9, place_offset % 9), axis=1)
    at_target = jnp.all(positions[None, :, :] == place_xy[:, None, :], axis=-1)
    target_card = jnp.argmax(at_target.astype(jnp.int32), axis=1)
    target_known = known[target_card]
    target_color = jnp.maximum(plan_colors[target_card], 0).astype(jnp.int32)
    hint_colors = nf[9 + hint_id, 3:6]
    compatible = hint_colors[jnp.arange(1068), target_color] > 0
    partner_already_knows = (nf[target_card, 9] > 0) | (hint_id == 0)
    same_neighbors = jnp.sum((card_dist == 1) & same, axis=1)
    anchor_quality = same_neighbors[target_card].astype(jnp.float32)
    row_quality = -jnp.abs(
        positions[target_card, 0] - (3 + target_color)).astype(jnp.float32)
    place_score = (
        500.0 * (target_known & compatible).astype(jnp.float32)
        - 800.0 * (target_known & (~compatible)).astype(jnp.float32)
        + 350.0 * (target_known & compatible & partner_already_knows).astype(jnp.float32)
        + 400.0 * (target_known & compatible & (anchor_quality > 0)).astype(jnp.float32)
        + 20.0 * anchor_quality + 3.0 * row_quality
        - 0.0001 * actions.astype(jnp.float32))
    hint_score = jnp.where(
        is_reveal, 2000.0 - actions.astype(jnp.float32), -1.0e7)
    hint_score = jnp.where(is_place, place_score, hint_score)

    scores = jnp.where(
        substep < 2, observe_score,
        jnp.where(substep == 2, move_score, hint_score))
    scores = jnp.where(legal, scores, -1.0e9)
    chosen = jnp.argmax(scores).astype(jnp.int32)
    chosen = jnp.where(
        (substep == 0) & can_end & legal[1067],
        jnp.asarray(1067, dtype=jnp.int32), chosen)
    action = jnp.where(active, chosen, jnp.asarray(0, dtype=jnp.int32))

    trace = {
        'known_count': jnp.sum(observed_known).astype(jnp.int32),
        'substep': substep,
        'can_end': can_end,
        'canonical_end': canonical_end,
        'plan_colors': plan_colors,
        'positions': positions,
        'locks': nf[:9, 8],
    }
    return action, next_memory, trace
