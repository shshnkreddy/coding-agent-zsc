"""Convention-legible cooperative policy for the 16-card Yokai game."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'early_stop_v26', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "known": jnp.full((16,), -1, dtype=jnp.int8),
        "inferred": jnp.full((16,), -1, dtype=jnp.int8),
        "strong_inferred": jnp.full((16,), -1, dtype=jnp.int8),
        "prev_pos": jnp.zeros((16, 2), dtype=jnp.int32),
        "calls": jnp.asarray(0, dtype=jnp.int32),
    }


def _masked_argmax(score, legal):
    return jnp.argmax(jnp.where(legal, score, -1.0e9)).astype(jnp.int32)


def _labels_connected(labels, adj):
    """Whether four cards of every fully specified label form one component."""
    same = labels[:, None] == labels[None, :]
    reach = adj & same & (labels[:, None] >= 0)
    for _ in range(4):
        reach = (reach.astype(jnp.int16) @ reach.astype(jnp.int16)) > 0
    counts = jnp.stack([jnp.sum(labels == c) for c in range(4)])
    connected = jnp.stack([
        jnp.sum(reach & ((labels == c)[:, None]) & ((labels == c)[None, :])) == 16
        for c in range(4)
    ])
    return jnp.all(counts == 4) & jnp.all(connected)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.node_features
    positions = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    active = features[0, 12] == 1
    substep = features[0, 13].astype(jnp.int32)

    base = init_agent()
    known0 = jnp.where(reset, base["known"], memory["known"])
    inferred0 = jnp.where(reset, base["inferred"], memory["inferred"])
    strong0 = jnp.where(reset, base["strong_inferred"], memory["strong_inferred"])
    prev0 = jnp.where(reset, positions, memory["prev_pos"])
    calls0 = jnp.where(reset, base["calls"], memory["calls"])

    visible = jnp.any(features[:16, :4] != 0, axis=1)
    visible_color = jnp.argmax(features[:16, :4], axis=1).astype(jnp.int8)
    known = jnp.where(visible, visible_color, known0)

    # Infer a partner-moved card from labelled neighbors at its destination.
    moved = jnp.any(positions != prev0, axis=1)
    dest_label = (positions[:, 0] - 3).astype(jnp.int8)
    canonical_dest = (dest_label >= 0) & (dest_label < 4)
    prior_labels = jnp.where(known >= 0, known, inferred0)
    card_adj = obs.adjacency_matrix[:16, :16] != 0
    votes = jnp.stack([
        jnp.sum(card_adj & (prior_labels[None, :] == c), axis=1) - (prior_labels == c).astype(jnp.int32)
        for c in range(4)
    ], axis=1)
    max_votes = jnp.max(votes, axis=1)
    old_dist = jnp.sum(jnp.abs(prev0[:, None, :] - positions[None, :, :]), axis=2)
    distinct_cards = jnp.arange(16)[:, None] != jnp.arange(16)[None, :]
    newly_adjacent = card_adj & (old_dist != 1) & distinct_cards
    neighbor_label = jnp.argmax(votes, axis=1).astype(jnp.int8)
    neighbor_confident = ((max_votes > 0)
                          & (jnp.sum(votes == max_votes[:, None], axis=1) == 1))
    new_winner_edge = jnp.any(
        newly_adjacent & (prior_labels[None, :] == neighbor_label[:, None]), axis=1
    )
    neighbor_confident = neighbor_confident & new_winner_edge
    strong_prior = jnp.where(known >= 0, known, strong0)
    strong_votes = jnp.stack([
        jnp.sum(card_adj & (strong_prior[None, :] == c), axis=1)
        - (strong_prior == c).astype(jnp.int32)
        for c in range(4)
    ], axis=1)
    strong_neighbor_label = jnp.argmax(strong_votes, axis=1).astype(jnp.int8)
    strong_max_votes = jnp.max(strong_votes, axis=1)
    strong_new_edge = jnp.any(
        newly_adjacent & (strong_prior[None, :] == strong_neighbor_label[:, None]), axis=1
    )
    strong_neighbor_confident = ((strong_max_votes > 0)
                                  & (jnp.sum(strong_votes == strong_max_votes[:, None], axis=1) == 1)
                                  & strong_new_edge)
    partner_seen = features[:16, 11] != 0
    partner_move = moved & (~active) & partner_seen
    row_signal = canonical_dest & (positions[:, 0] != prev0[:, 0]) & (max_votes == 0)
    inferred_move = jnp.where(neighbor_confident, neighbor_label, dest_label)
    infer_ok = partner_move & (neighbor_confident | row_signal)
    inferred = jnp.where(infer_ok, inferred_move, inferred0)
    inferred = jnp.where(known >= 0, known, inferred)
    strong_inferred = jnp.where(
        partner_move & strong_neighbor_confident, strong_neighbor_label, strong0
    )
    strong_inferred = jnp.where(known >= 0, known, strong_inferred)
    public_hint_sets = features[16:23, 4:8] != 0
    card_hint_edges = obs.adjacency_matrix[:16, 16:23] != 0
    attached_allowed = (card_hint_edges.astype(jnp.int16) @ public_hint_sets.astype(jnp.int16)) > 0
    pre_known_counts = jnp.stack([jnp.sum(known == c) for c in range(4)])
    hint_possible = (attached_allowed & partner_seen[:, None]
                     & (pre_known_counts[None, :] < 4))
    forced_hint = jnp.sum(hint_possible, axis=1) == 1
    hint_label = jnp.argmax(hint_possible, axis=1).astype(jnp.int8)
    inferred = jnp.where((known < 0) & forced_hint, hint_label, inferred)
    strong_inferred = jnp.where((known < 0) & forced_hint, hint_label, strong_inferred)
    raw_labels = jnp.where(known >= 0, known, inferred)
    strong_labels = jnp.where(known >= 0, known, strong_inferred)
    strong_counts = jnp.stack([jnp.sum(strong_labels == c) for c in range(4)])
    strong_residual = 4 - strong_counts
    strong_unknown = strong_labels < 0
    strong_forced = ((jnp.sum(strong_residual > 0) == 1)
                     & jnp.all(strong_residual >= 0)
                     & (jnp.sum(strong_unknown) == jnp.sum(strong_residual))
                     & (jnp.sum(known >= 0) >= 14))
    strong_forced_color = jnp.argmax(strong_residual).astype(jnp.int8)
    strong_stop_labels = jnp.where(
        strong_unknown & strong_forced, strong_forced_color, strong_labels
    )

    # Public beliefs must respect the exact multiplicity of four per color.
    known_counts = jnp.stack([jnp.sum(known == c) for c in range(4)])
    capacity = 4 - known_counts
    inferred_candidate = (known < 0) & (inferred >= 0)
    ids = jnp.arange(16)
    earlier = ids[None, :] < ids[:, None]
    same_inferred = inferred[None, :] == inferred[:, None]
    inferred_rank = jnp.sum(earlier & same_inferred & inferred_candidate[None, :], axis=1)
    inferred_clean = jnp.where(
        inferred_candidate & (inferred_rank < capacity[jnp.maximum(inferred, 0)]), inferred, -1
    ).astype(jnp.int8)
    labels0 = jnp.where(known >= 0, known, inferred_clean)
    labelled_counts = jnp.stack([jnp.sum(labels0 == c) for c in range(4)])
    residual = 4 - labelled_counts
    unknown = labels0 < 0
    forced = (jnp.sum(residual > 0) == 1) & (jnp.sum(unknown) == jnp.sum(residual))
    forced_color = jnp.argmax(residual).astype(jnp.int8)
    labels = jnp.where(unknown & forced, forced_color, labels0)

    # Public partner-seen flags partition observations between the two seats.
    observe_legal = legal[1:17]
    unknown_self = known < 0
    obs_priority = (unknown_self.astype(jnp.float32) * 100.0
                    + ((~partner_seen) & unknown_self).astype(jnp.float32) * 1000.0
                    + ((inferred_clean < 0) & unknown_self).astype(jnp.float32) * 500.0
                    - jnp.arange(16, dtype=jnp.float32) * 0.01)
    observe_action = 1 + _masked_argmax(obs_priority, observe_legal)

    # Exhaustively score legal relocations by canonical-row progress, direct
    # same-color adjacency, and compactness among personally known cards.
    aa = jnp.arange(1600, dtype=jnp.int32)
    mover = aa // 100
    rem = aa % 100
    dx = rem // 10
    dy = rem % 10
    mover_color = known[mover].astype(jnp.int32)
    mover_known = mover_color >= 0
    old = positions[mover]
    target_x = 3 + jnp.maximum(mover_color, 0)

    other_known = labels[None, :] >= 0
    same_color = labels[None, :] == mover_color[:, None]
    not_self = jnp.arange(16)[None, :] != mover[:, None]
    peers = other_known & same_color & not_self
    old_dist = jnp.abs(old[:, None, 0] - positions[None, :, 0]) + jnp.abs(old[:, None, 1] - positions[None, :, 1])
    new_dist = jnp.abs(dx[:, None] - positions[None, :, 0]) + jnp.abs(dy[:, None] - positions[None, :, 1])
    old_edges = jnp.sum(peers & (old_dist == 1), axis=1)
    new_edges = jnp.sum(peers & (new_dist == 1), axis=1)
    old_sum = jnp.sum(jnp.where(peers, old_dist, 0), axis=1)
    new_sum = jnp.sum(jnp.where(peers, new_dist, 0), axis=1)
    old_row_error = jnp.abs(old[:, 0] - target_x)
    new_row_error = jnp.abs(dx - target_x)
    row_gain = old_row_error - new_row_error
    edge_gain = new_edges - old_edges
    compact_gain = old_sum - new_sum
    # Reward attaching to a previously separate believed component.
    belief_same = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
    reach = card_adj & belief_same
    for _ in range(4):
        reach = (reach.astype(jnp.int16) @ reach.astype(jnp.int16)) > 0
    new_neighbors = peers & (new_dist == 1)
    merge_gain = jnp.sum(new_neighbors & (~reach[mover]), axis=1)
    # Evaluate the complete believed color graph after each candidate move.
    candidate_graph = jnp.broadcast_to((card_adj & belief_same)[None, :, :], (1600, 16, 16))
    self_edge = jnp.arange(16)[None, :] == mover[:, None]
    candidate_row = new_neighbors | self_edge
    candidate_index = jnp.arange(1600)
    candidate_graph = candidate_graph.at[candidate_index, mover, :].set(candidate_row)
    candidate_graph = candidate_graph.at[candidate_index, :, mover].set(candidate_row)
    candidate_reach = candidate_graph
    for _ in range(4):
        candidate_reach = (candidate_reach.astype(jnp.int16) @ candidate_reach.astype(jnp.int16)) > 0
    component_gain = (jnp.sum(candidate_reach, axis=(1, 2)) - jnp.sum(reach)).astype(jnp.float32)
    centrality = -jnp.abs(dy.astype(jnp.float32) - 4.5)
    move_score = (mover_known.astype(jnp.float32) * 100000.0
                  + component_gain * 10000.0
                  + merge_gain.astype(jnp.float32) * 300.0
                  + edge_gain.astype(jnp.float32) * 340.0
                  + compact_gain.astype(jnp.float32) * 10.0
                  + row_gain.astype(jnp.float32) * 60.0
                  - new_row_error.astype(jnp.float32) * 5.0
                  + centrality * 0.1
                  - mover.astype(jnp.float32) * 0.001)
    move_action = 17 + _masked_argmax(move_score, legal[17:1617])

    # Preserve mobility by revealing all hints before placing any.
    reveal_score = -jnp.arange(7, dtype=jnp.float32)
    reveal_action = 1617 + _masked_argmax(reveal_score, legal[1617:1624])

    pa = jnp.arange(700, dtype=jnp.int32)
    ph = pa // 100
    prem = pa % 100
    px = prem // 10
    py = prem % 10
    at_target = (positions[:, 0][None, :] == px[:, None]) & (positions[:, 1][None, :] == py[:, None])
    target_card = jnp.argmax(at_target, axis=1)
    target_known_color = known[target_card].astype(jnp.int32)
    target_label = labels[target_card].astype(jnp.int32)
    hint_sets = features[16:23, 4:8] != 0
    safe_known = (target_known_color >= 0) & hint_sets[ph, jnp.maximum(target_known_color, 0)]
    safe_inferred = (target_label >= 0) & hint_sets[ph, jnp.maximum(target_label, 0)]
    hint_size = jnp.sum(hint_sets[ph], axis=1)
    target_neighbors = jnp.sum(card_adj[target_card] & (labels[None, :] == target_label[:, None]), axis=1)
    target_component = jnp.sum(reach[target_card], axis=1)
    complete_component = (target_label >= 0) & (target_component == 4)
    clustered_known = (target_known_color >= 0) & (target_neighbors > 1)
    clustered_inferred = (target_known_color < 0) & (target_label >= 0) & (target_neighbors > 1)
    place_score = (complete_component.astype(jnp.float32) * 50000.0
                   + target_component.astype(jnp.float32) * 2000.0
                   + clustered_known.astype(jnp.float32) * 20000.0
                   + clustered_inferred.astype(jnp.float32) * 15000.0
                   + safe_known.astype(jnp.float32) * 100000.0
                   + safe_inferred.astype(jnp.float32) * 200.0
                   - hint_size.astype(jnp.float32) * 100.0
                   + ph.astype(jnp.float32) * 0.1
                   - target_card.astype(jnp.float32) * 0.001)
    place_action = 1624 + _masked_argmax(place_score, legal[1624:2324])
    any_reveal_legal = jnp.any(legal[1617:1624])
    any_complete_place = jnp.any(legal[1624:2324] & complete_component & safe_known)
    any_singleton_place = jnp.any(legal[1624:2324] & (hint_size == 1) & safe_known & (target_component >= 3))
    opportunistic_place = any_complete_place | any_singleton_place
    hint_action = jnp.where(any_reveal_legal & (~opportunistic_place), reveal_action, place_action)

    # End only when private/public evidence jointly supports a solved board.
    adj = obs.adjacency_matrix[:16, :16] != 0
    belief_solved = _labels_connected(labels, adj)
    raw_belief_solved = _labels_connected(raw_labels, adj)
    row_ok = (positions[:, 0] >= 3) & (positions[:, 0] <= 6)
    known_agrees = jnp.all((known < 0) | (positions[:, 0] == 3 + known.astype(jnp.int32)))
    row_counts = jnp.stack([jnp.sum(positions[:, 0] == 3 + c) for c in range(4)])
    canonical_solved = (jnp.sum(known >= 0) >= 8) & jnp.all(row_ok) & jnp.all(row_counts == 4) & known_agrees
    strong_belief_solved = _labels_connected(strong_stop_labels, adj)
    weak_label_count = jnp.sum((known < 0) & (strong_inferred < 0) & (inferred >= 0))
    hybrid_belief_solved = raw_belief_solved & (weak_label_count <= 1)
    should_end = (legal[2324] & (strong_belief_solved | hybrid_belief_solved)
                  & (jnp.sum(known >= 0) >= 10))

    proposed = jnp.where(substep == 0, observe_action,
               jnp.where(substep == 1, observe_action,
               jnp.where(substep == 2, move_action, hint_action)))
    proposed = jnp.where((substep == 0) & should_end, 2324, proposed)
    action = jnp.where(active, proposed, 0).astype(jnp.int32)

    next_memory = {
        "known": known.astype(jnp.int8),
        "inferred": inferred.astype(jnp.int8),
        "strong_inferred": strong_inferred.astype(jnp.int8),
        "prev_pos": positions,
        "calls": calls0 + 1,
    }
    trace = {
        "known_count": jnp.sum(known >= 0).astype(jnp.int32),
        "inferred_count": jnp.sum((known < 0) & (labels >= 0)).astype(jnp.int32),
        "belief_solved": belief_solved,
        "raw_belief_solved": raw_belief_solved,
        "strong_belief_solved": strong_belief_solved,
        "canonical_solved": canonical_solved,
    }
    return action, next_memory, trace
