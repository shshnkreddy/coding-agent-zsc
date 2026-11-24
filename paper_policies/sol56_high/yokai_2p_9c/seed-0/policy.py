"""A convention-light, private-knowledge clustering policy for Yōkai."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'partner_deduction_cluster', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "colors": jnp.full((9,), -1, dtype=jnp.int32),
        "previous_positions": jnp.zeros((9, 2), dtype=jnp.int32),
        "partner_affinity": jnp.zeros((9, 9), dtype=jnp.int32),
        "last_partner_card": jnp.int32(-1),
        "partner_seen_history": jnp.zeros((9,), dtype=jnp.bool_),
        "pending_partner_seen": jnp.zeros((9,), dtype=jnp.bool_),
    }


def _effective_colors(colors):
    """Infer the final unseen card from the known 3/3/3 multiplicities."""
    known = colors >= 0
    counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    missing_color = jnp.argmin(counts).astype(jnp.int32)
    infer = (jnp.sum(known) == 8) | (jnp.sum(counts == 3) == 2)
    return jnp.where((~known) & infer, missing_color, colors)


def _apply_hint_deductions(obs, colors):
    """Use competent correct placements as public, logically constrained evidence."""
    result = colors
    for _ in range(4):
        for h in range(4):
            counts = jnp.stack([jnp.sum(result == c) for c in range(3)])
            hinted = obs.node_features[9 + h, 3:6] != 0
            available = hinted & (counts < 3)
            target = jnp.argmax(obs.adjacency_matrix[9 + h, :9]).astype(jnp.int32)
            placed = obs.node_features[9 + h, 8] != 0
            forced = jnp.sum(available) == 1
            inferred = jnp.argmax(available).astype(jnp.int32)
            use = placed & forced & (result[target] < 0)
            result = result.at[target].set(jnp.where(use, inferred, result[target]))

    unknown = result < 0
    num_unknown = jnp.sum(unknown)
    rank = jnp.cumsum(unknown.astype(jnp.int32)) - 1
    case = jnp.arange(27, dtype=jnp.int32)
    digits = jnp.stack([(case // 9) % 3, (case // 3) % 3, case % 3], axis=1)
    assigned = jnp.take(digits, jnp.clip(rank, 0, 2), axis=1)
    labels = jnp.where(unknown[None, :], assigned, result[None, :])
    counts = jnp.stack([jnp.sum(labels == c, axis=1) for c in range(3)], axis=1)
    allowed = _hint_allowed(obs)
    allowed_label = jnp.take_along_axis(
        allowed[None, :, :], labels[:, :, None], axis=2)[:, :, 0]
    valid = jnp.all(counts == 3, axis=1) & jnp.all(
        allowed_label | (~unknown)[None, :], axis=1)
    votes = jnp.stack([
        jnp.sum(valid[:, None] & (labels == c), axis=0) for c in range(3)
    ], axis=1)
    valid_count = jnp.sum(valid)
    forced_global = (jnp.max(votes, axis=1) == valid_count) & (valid_count > 0)
    inferred_global = jnp.argmax(votes, axis=1).astype(jnp.int32)
    use_global = (unknown & forced_global & (num_unknown >= 2)
                  & (num_unknown <= 3))
    result = jnp.where(use_global, inferred_global, result)
    return result


def _reachability(adj, colors):
    """Same-color transitive closure, with unknown cards excluded."""
    known = colors >= 0
    same = (colors[:, None] == colors[None, :]) & known[:, None] & known[None, :]
    reach = adj & same
    reach = reach | (jnp.eye(9, dtype=jnp.bool_) & known[:, None])
    for _ in range(4):
        reach = (jnp.einsum("...ik,...kj->...ij", reach.astype(jnp.int32),
                            reach.astype(jnp.int32)) > 0)
    return reach, same


def _groups_complete(positions, colors):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adj = jnp.sum(delta, axis=-1) == 1
    reach, same = _reachability(adj, colors)
    pairs = jnp.triu(same, 1)
    return jnp.all((~pairs) | reach) & jnp.all(colors >= 0)


def _hint_allowed(obs):
    """Per-card colors supported by any public placed hint."""
    allowed = jnp.ones((9, 3), dtype=jnp.bool_)
    hint_sets = obs.node_features[9:13, 3:6] != 0
    for h in range(4):
        target = jnp.argmax(obs.adjacency_matrix[9 + h, :9]).astype(jnp.int32)
        placed = obs.node_features[9 + h, 8] != 0
        restricted = jnp.where(placed, hint_sets[h], allowed[target])
        allowed = allowed.at[target].set(restricted)
    return allowed


def _robustly_complete(obs, colors):
    """Certify completion for every valid labeling of two/three unknowns."""
    exact = _groups_complete(obs.positions, _effective_colors(colors))
    unknown = colors < 0
    num_unknown = jnp.sum(unknown)
    # Each unknown card receives its rank's digit in a base-three assignment.
    # There are only 3^3 cases because this certificate is enabled late.
    rank = jnp.cumsum(unknown.astype(jnp.int32)) - 1
    case = jnp.arange(27, dtype=jnp.int32)
    digits = jnp.stack([(case // 9) % 3, (case // 3) % 3, case % 3], axis=1)
    assigned_unknown = jnp.take(digits, jnp.clip(rank, 0, 2), axis=1)
    labels = jnp.where(unknown[None, :], assigned_unknown, colors[None, :])
    counts = jnp.stack([jnp.sum(labels == c, axis=1) for c in range(3)], axis=1)
    allowed = _hint_allowed(obs)
    allowed_label = jnp.take_along_axis(
        allowed[None, :, :], labels[:, :, None], axis=2)[:, :, 0]
    valid = jnp.all(counts == 3, axis=1) & jnp.all(
        allowed_label | (~unknown)[None, :], axis=1)

    delta = jnp.abs(obs.positions[:, None, :] - obs.positions[None, :, :])
    adj = jnp.sum(delta, axis=-1) == 1
    same = labels[:, :, None] == labels[:, None, :]
    reach = adj[None, :, :] & same
    reach = reach | jnp.eye(9, dtype=jnp.bool_)[None, :, :]
    for _ in range(4):
        reach = (jnp.einsum("...ik,...kj->...ij", reach.astype(jnp.int32),
                            reach.astype(jnp.int32)) > 0)
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    case_complete = jnp.all((~(same & upper[None, :, :])) | reach,
                            axis=(1, 2))
    certificate = (jnp.any(valid) & jnp.all((~valid) | case_complete)
                   & (num_unknown >= 2) & (num_unknown <= 3))
    return exact | certificate


def _choose_observe(obs, colors, last_partner_card, pending_partner_seen):
    legal = obs.legal_actions[1:10]
    locked = obs.node_features[:9, 8] != 0
    unknown = colors < 0
    partner_seen = obs.node_features[:9, 9] != 0
    recent_move = unknown & (jnp.arange(9, dtype=jnp.int32) == last_partner_card)
    score = (unknown.astype(jnp.int32) * 100
             + (unknown & partner_seen).astype(jnp.int32) * 30
             + (unknown & pending_partner_seen).astype(jnp.int32) * 10
             + recent_move.astype(jnp.int32) * 50
             - locked.astype(jnp.int32) * 1000)
    score = score * 20 - jnp.arange(9, dtype=jnp.int32)
    return (1 + jnp.argmax(jnp.where(legal, score, -100000))).astype(jnp.int32)


def _choose_move(obs, colors, partner_affinity):
    old = obs.positions.astype(jnp.int32)
    a = jnp.arange(729, dtype=jnp.int32)
    card = a // 81
    rem = a % 81
    dest = jnp.stack([rem // 9, rem % 9], axis=1)
    candidate = jnp.broadcast_to(old, (729, 9, 2))
    candidate = candidate.at[jnp.arange(729), card].set(dest)
    delta = jnp.abs(candidate[:, :, None, :] - candidate[:, None, :, :])
    adj = jnp.sum(delta, axis=-1) == 1

    effective = _effective_colors(colors)
    reach, same = _reachability(adj, effective)
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    connected_pairs = jnp.sum(reach & same & upper, axis=(1, 2))
    same_edges = jnp.sum(adj & same & upper, axis=(1, 2))
    pair_distance = jnp.sum(jnp.sum(delta, axis=-1) * same * upper,
                            axis=(1, 2))
    old_delta = jnp.abs(old[:, None, :] - old[None, :, :])
    old_adj = jnp.sum(old_delta, axis=-1) == 1
    changed_edges = jnp.sum((adj != old_adj[None, :, :]) & upper, axis=(1, 2))
    distance = jnp.sum(jnp.abs(dest - old[card]), axis=1)
    moved_known = (effective[card] >= 0).astype(jnp.int32)
    both_known = (effective[:, None] >= 0) & (effective[None, :] >= 0)
    compatible_affinity = partner_affinity * ((~both_known) | same)
    affinity_edges = jnp.sum(adj * compatible_affinity[None, :, :] * upper,
                             axis=(1, 2))
    known_onehot = jnp.stack([effective == c for c in range(3)], axis=1)
    touches_color = (jnp.einsum("aij,jc->aic", adj.astype(jnp.int32),
                                known_onehot.astype(jnp.int32)) > 0)
    frontier_coverage = jnp.sum(touches_color & (effective < 0)[None, :, None],
                                axis=(1, 2))
    t = (touches_color & (effective < 0)[None, :, None]).astype(jnp.int32)

    # Public correct placements constrain the remaining hidden assignments.
    # With no informative placed hint this exactly reproduces the previous
    # six-case (2/2/2) and three-case (1/2/3) frontier objective.
    allowed = _hint_allowed(obs)
    unknown = effective < 0
    rank = jnp.cumsum(unknown.astype(jnp.int32)) - 1
    color_counts = jnp.sum(known_onehot, axis=0)
    remaining = 3 - color_counts
    cumulative = jnp.cumsum(remaining)
    base_assignment = jnp.argmax(
        cumulative[None, :] > jnp.arange(3, dtype=jnp.int32)[:, None], axis=1)
    permutations = jnp.asarray(((0, 1, 2), (0, 2, 1), (1, 0, 2),
                                (1, 2, 0), (2, 0, 1), (2, 1, 0)),
                               dtype=jnp.int32)
    assignments = base_assignment[permutations]
    assigned_unknown = jnp.take(assignments, jnp.clip(rank, 0, 2), axis=1)
    allowed_label = jnp.take_along_axis(
        allowed[None, :, :], assigned_unknown[:, :, None], axis=2)[:, :, 0]
    valid_case = jnp.all(allowed_label | (~unknown)[None, :], axis=1)
    label_onehot = jnp.stack([assigned_unknown == c for c in range(3)], axis=2)
    match_count = jnp.einsum("aic,kic->ak", t,
                             label_onehot.astype(jnp.int32))
    valid_count = jnp.maximum(jnp.sum(valid_case), 1)
    covered_count = jnp.sum((match_count > 0) & valid_case[None, :], axis=1)
    two_count = jnp.sum((match_count > 1) & valid_case[None, :], axis=1)
    six_known = (jnp.sum(~unknown) == 6) & jnp.any(valid_case)
    assignment_score = ((covered_count * 720 + two_count * 150) // valid_count
                        + frontier_coverage * 10) * six_known
    score = (connected_pairs * 1000 + same_edges * 40 - pair_distance * 5
             + moved_known * 4
             + affinity_edges * 120
             + assignment_score
             - changed_edges * 3 - distance)
    legal = obs.legal_actions[10:739]
    return (10 + jnp.argmax(jnp.where(legal, score, -10000000))).astype(jnp.int32)


def _choose_hint(obs, colors):
    legal_reveal = obs.legal_actions[739:743]
    any_reveal = jnp.any(legal_reveal)
    reveal_action = (739 + jnp.argmax(legal_reveal)).astype(jnp.int32)

    a = jnp.arange(324, dtype=jnp.int32)
    hint = a // 81
    rem = a % 81
    xy = jnp.stack([rem // 9, rem % 9], axis=1)
    at_card = jnp.all(xy[:, None, :] == obs.positions[None, :, :], axis=2)
    card = jnp.argmax(at_card, axis=1)
    exists = jnp.any(at_card, axis=1)
    hint_sets = obs.node_features[9:13, 3:6] != 0
    known = colors[card] >= 0
    safe_color = jnp.clip(colors[card], 0, 2)
    correct = known & hint_sets[hint, safe_color]
    partner_seen = obs.node_features[:9, 9] != 0
    information_safe = partner_seen[card]
    card_adj = obs.adjacency_matrix[:9, :9] != 0
    same_known = ((colors[:, None] == colors[None, :])
                  & (colors[:, None] >= 0) & (colors[None, :] >= 0))
    same_degree = jnp.sum(card_adj & same_known, axis=1) - 1
    public_degree = jnp.sum(card_adj, axis=1) - 1
    score = (correct.astype(jnp.int32) * 1000 + known.astype(jnp.int32) * 100
             + information_safe.astype(jnp.int32)
             + same_degree[card] * 10 + public_degree[card] - card)
    legal_place = obs.legal_actions[743:1067] & exists
    place_action = (743 + jnp.argmax(jnp.where(legal_place, score, -100000))).astype(jnp.int32)
    return jnp.where(any_reveal, reveal_action, place_action)


def agent_act(obs, reset, memory, rng):
    del rng
    base_colors = jnp.where(reset, jnp.full((9,), -1, dtype=jnp.int32),
                            memory["colors"])
    visible = obs.node_features[:9, :3]
    seen_now = jnp.sum(visible, axis=1) > 0
    seen_color = jnp.argmax(visible, axis=1).astype(jnp.int32)
    observed_colors = jnp.where(seen_now, seen_color, base_colors)
    colors = _apply_hint_deductions(obs, observed_colors)
    substep = obs.node_features[0, 11].astype(jnp.int32)
    active = obs.node_features[0, 10] != 0
    partner_seen_now = obs.node_features[:9, 9] != 0
    partner_seen_base = jnp.where(
        reset, jnp.zeros((9,), dtype=jnp.bool_), memory["partner_seen_history"])
    pending_base = jnp.where(
        reset, jnp.zeros((9,), dtype=jnp.bool_), memory["pending_partner_seen"])
    pending_partner_seen = pending_base | (partner_seen_now & (~partner_seen_base))
    clear_pending = active & (substep == 2)
    pending_next = jnp.where(clear_pending,
                             jnp.zeros((9,), dtype=jnp.bool_),
                             pending_partner_seen)

    positions = obs.positions.astype(jnp.int32)
    previous = jnp.where(reset, positions, memory["previous_positions"])
    old_delta = jnp.abs(previous[:, None, :] - previous[None, :, :])
    old_adj = jnp.sum(old_delta, axis=-1) == 1
    new_delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    new_adj = jnp.sum(new_delta, axis=-1) == 1
    changed = jnp.any(positions != previous, axis=1)
    moved_card = jnp.argmax(changed).astype(jnp.int32)
    partner_move = (~active) & (substep == 3) & jnp.any(changed) & (~reset)
    last_partner_base = jnp.where(reset, jnp.int32(-1),
                                  memory["last_partner_card"])
    last_partner_card = jnp.where(partner_move, moved_card, last_partner_base)
    created = new_adj[moved_card] & (~old_adj[moved_card])
    created = created & (jnp.arange(9, dtype=jnp.int32) != moved_card)
    partner_seen_cards = obs.node_features[:9, 9] != 0
    shared_created = created & partner_seen_cards
    neighbor_color_counts = jnp.stack([
        jnp.sum(shared_created & (colors == c)) for c in range(3)
    ])
    inferred_move_color = jnp.argmax(neighbor_color_counts).astype(jnp.int32)
    current_color_counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    move_deduction = (partner_move & (colors[moved_card] < 0)
                      & partner_seen_cards[moved_card]
                      & (jnp.max(neighbor_color_counts) >= 2)
                      & (jnp.sum(neighbor_color_counts > 0) == 1)
                      & (current_color_counts[inferred_move_color] < 3))
    colors = colors.at[moved_card].set(
        jnp.where(move_deduction, inferred_move_color, colors[moved_card]))
    effective = _effective_colors(colors)
    signal = created.astype(jnp.int32) * partner_move.astype(jnp.int32)
    added = jnp.zeros((9, 9), dtype=jnp.int32)
    added = added.at[moved_card, :].set(signal)
    added = jnp.maximum(added, added.T)
    affinity_base = jnp.where(reset, jnp.zeros((9, 9), dtype=jnp.int32),
                              memory["partner_affinity"])
    affinity = jnp.maximum(affinity_base, added)
    complete = _robustly_complete(obs, colors)

    observe_action = _choose_observe(obs, colors, last_partner_card,
                                     pending_partner_seen)
    move_action = _choose_move(obs, colors, affinity)
    hint_action = _choose_hint(obs, effective)
    active_action = jnp.where(substep == 0,
                              jnp.where(complete, jnp.int32(1067), observe_action),
                              jnp.where(substep == 1, observe_action,
                                        jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, active_action, jnp.int32(0)).astype(jnp.int32)
    trace = {
        "known_count": jnp.sum(colors >= 0).astype(jnp.int32),
        "public_deductions": (jnp.sum(colors >= 0)
                              - jnp.sum(observed_colors >= 0)).astype(jnp.int32),
        "inferred_complete": complete,
        "partner_links": (jnp.sum(affinity) // 2).astype(jnp.int32),
        "move_deduction": move_deduction,
    }
    next_memory = {
        "colors": colors,
        "previous_positions": positions,
        "partner_affinity": affinity,
        "last_partner_card": last_partner_card,
        "partner_seen_history": partner_seen_now,
        "pending_partner_seen": pending_next,
    }
    return action, next_memory, trace
