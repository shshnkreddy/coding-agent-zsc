"""Convention-legible heuristic policy for the 16-card Yōkai game."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'canonical_row_cooperator',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 4,
    'deck_size': 16,
    'grid_size': 10,
    'num_hints': 7,
    'num_nodes': 23,
    'node_features': 15,
    'num_actions': 2325,
    'policy_structure': 'shared_local',
}


def init_agent():
    return {
        'previous_positions': jnp.zeros((16, 2), dtype=jnp.int32),
        'signalled_color': jnp.full((16,), -1, dtype=jnp.int8),
        'signal_checked': jnp.zeros((16,), dtype=jnp.bool_),
        'row_matches': jnp.asarray(0, dtype=jnp.int8),
        'row_mismatches': jnp.asarray(0, dtype=jnp.int8),
    }


def _beliefs(obs, signalled_color, row_trusted):
    own = obs.node_features[:16, :4].astype(jnp.float32)
    own_known = own.sum(axis=1) > 0
    hint_sets = obs.node_features[16:23, 4:8].astype(jnp.float32)
    card_hint_edges = obs.adjacency_matrix[:16, 16:23].astype(jnp.float32)
    public_set = card_hint_edges @ hint_sets
    public_size = public_set.sum(axis=1)
    public_belief = public_set / jnp.maximum(public_size[:, None], 1.0)

    safe_label = jnp.clip(signalled_color.astype(jnp.int32), 0, 3)
    signal = jax.nn.one_hot(safe_label, 4, dtype=jnp.float32)
    has_signal = signalled_color >= 0
    remaining = jnp.maximum(4.0 - own.sum(axis=0), 0.0)
    residual = remaining / jnp.maximum(remaining.sum(), 1.0)
    residual_exact = (remaining > 0).sum() == 1
    signal_valid = has_signal & (remaining[safe_label] > 0)
    signal_exact = signal_valid & row_trusted
    belief = jnp.broadcast_to(residual[None, :], (16, 4))
    belief = jnp.where((public_size > 0)[:, None], public_belief, belief)
    belief = jnp.where(signal_exact[:, None], signal, belief)
    belief = jnp.where((~own_known & residual_exact)[:, None],
                       residual[None, :], belief)
    belief = jnp.where(own_known[:, None], own, belief)
    count_exact = ~own_known & residual_exact
    confidence = own_known | (public_size == 1) | signal_exact | count_exact
    evidence_strength = jnp.where(
        own_known | (public_size == 1) | count_exact, 1.0,
        jnp.where(signal_exact, 0.25, 0.0))
    return belief, confidence, own_known, evidence_strength


def _groups_connected(obs, labels):
    adjacency = obs.adjacency_matrix[:16, :16].astype(jnp.bool_)
    okay = jnp.asarray(True)
    for color in range(4):
        members = labels == color
        seed = jax.nn.one_hot(jnp.argmax(members.astype(jnp.int32)), 16,
                              dtype=jnp.bool_)
        reached = seed & members
        for _ in range(16):
            reached = reached | ((adjacency @ reached.astype(jnp.int32)) > 0)
            reached = reached & members
        okay = okay & (members.sum() == 4) & jnp.all(~members | reached)
    return okay


def _choose_observation(obs, own_known):
    legal = obs.legal_actions[1:17]
    partner_seen = obs.node_features[:16, 11] > 0
    card_id = jnp.arange(16, dtype=jnp.int32)
    priority = ((~own_known & ~partner_seen).astype(jnp.int32) * 1000
                + (~own_known).astype(jnp.int32) * 100
                - card_id)
    priority = jnp.where(legal, priority, -100000)
    return (1 + jnp.argmax(priority)).astype(jnp.int32)


def _component_reach_without_each_card(obs, labels):
    """Same-label reachability after removing each possible moved card."""
    adjacency = obs.adjacency_matrix[:16, :16].astype(jnp.bool_)
    same_label = labels[:, None] == labels[None, :]
    cards = jnp.arange(16, dtype=jnp.int32)
    present = cards[None, :] != cards[:, None]
    graph = (adjacency & same_label)[None, :, :]
    graph = graph & present[:, :, None] & present[:, None, :]
    reached = jnp.eye(16, dtype=jnp.bool_)[None, :, :] & present[:, :, None]
    for _ in range(16):
        reached = reached | ((reached.astype(jnp.int32)
                              @ graph.astype(jnp.int32)) > 0)
    return reached


def _confident_component_sizes(obs, labels, confidence):
    adjacency = obs.adjacency_matrix[:16, :16].astype(jnp.bool_)
    graph = (adjacency & (labels[:, None] == labels[None, :])
             & confidence[:, None] & confidence[None, :])
    reached = jnp.eye(16, dtype=jnp.bool_) & confidence[:, None]
    for _ in range(16):
        reached = reached | ((reached.astype(jnp.int32)
                              @ graph.astype(jnp.int32)) > 0)
    return reached.sum(axis=1)


def _choose_move(obs, belief, confidence, evidence_strength):
    move_legal = obs.legal_actions[17:1617]
    index = jnp.arange(1600, dtype=jnp.int32)
    card = index // 100
    cell = index % 100
    x = cell // 10
    y = cell % 10
    positions = obs.positions.astype(jnp.int32)
    old_x = positions[card, 0]
    old_y = positions[card, 1]
    moved_belief = belief[card]
    colors = jnp.arange(4, dtype=jnp.float32)
    target_x = 3.0 + colors

    def location_value(px, py):
        row_cost = (moved_belief * jnp.abs(px[:, None].astype(jnp.float32)
                                          - target_x[None, :])).sum(axis=1)
        y_cost = jnp.maximum(3 - py, 0) + jnp.maximum(py - 6, 0)
        return -30.0 * row_cost - 6.0 * y_cost.astype(jnp.float32)

    candidate_value = location_value(x, y)
    source_value = location_value(old_x, old_y)

    dx = jnp.abs(x[:, None] - positions[None, :, 0])
    dy = jnp.abs(y[:, None] - positions[None, :, 1])
    adjacent = (dx + dy) == 1
    adjacent = adjacent & (jnp.arange(16)[None, :] != card[:, None])
    same_probability = moved_belief @ belief.T
    candidate_neighbors = (adjacent.astype(jnp.float32)
                           * same_probability).sum(axis=1)

    labels = jnp.argmax(belief, axis=1)
    component_reach = _component_reach_without_each_card(obs, labels)
    component_sizes = component_reach.sum(axis=2).astype(jnp.float32)
    candidate_component = jnp.max(
        adjacent.astype(jnp.float32) * same_probability
        * component_sizes[card], axis=1)
    candidate_union = jnp.max(
        adjacent[:, :, None].astype(jnp.float32)
        * same_probability[:, :, None]
        * component_reach[card].astype(jnp.float32), axis=1).sum(axis=1)

    old_dx = jnp.abs(old_x[:, None] - positions[None, :, 0])
    old_dy = jnp.abs(old_y[:, None] - positions[None, :, 1])
    old_adjacent = (old_dx + old_dy) == 1
    old_adjacent = old_adjacent & (jnp.arange(16)[None, :] != card[:, None])
    source_neighbors = (old_adjacent.astype(jnp.float32)
                        * same_probability).sum(axis=1)
    source_component = jnp.max(
        old_adjacent.astype(jnp.float32) * same_probability
        * component_sizes[card], axis=1)
    source_union = jnp.max(
        old_adjacent[:, :, None].astype(jnp.float32)
        * same_probability[:, :, None]
        * component_reach[card].astype(jnp.float32), axis=1).sum(axis=1)

    delta = candidate_value - source_value + 52.5 * (
        candidate_neighbors - source_neighbors) + 5.0 * (
        candidate_component - source_component)
    score = (100.0 * delta
             + 1000.0 * evidence_strength[card]
             - 0.05 * (jnp.abs(x - 4.5) + jnp.abs(y - 4.5))
             - index.astype(jnp.float32) * 1e-6)
    score = jnp.where(move_legal, score, -1e9)
    return (17 + jnp.argmax(score)).astype(jnp.int32)


def _choose_hint(obs, belief, confidence, evidence_strength):
    reveal_legal = obs.legal_actions[1617:1624]
    reveal_action = (1617 + jnp.argmax(reveal_legal)).astype(jnp.int32)

    index = jnp.arange(700, dtype=jnp.int32)
    hint = index // 100
    cell = index % 100
    x = cell // 10
    y = cell % 10
    place_legal = obs.legal_actions[1624:2324]
    positions = obs.positions.astype(jnp.int32)
    at_cell = ((positions[None, :, 0] == x[:, None])
               & (positions[None, :, 1] == y[:, None]))
    target_card = jnp.argmax(at_cell.astype(jnp.int32), axis=1)
    hint_sets = obs.node_features[16:23, 4:8].astype(jnp.float32)
    match_probability = (hint_sets[hint] * belief[target_card]).sum(axis=1)
    predicted_color = jnp.argmax(belief[target_card], axis=1)
    settled = positions[target_card, 0] == (3 + predicted_color)
    card_adjacency = obs.adjacency_matrix[:16, :16].astype(jnp.float32)
    same_probability = belief[target_card] @ belief.T
    neighbor_support = (card_adjacency[target_card]
                        * same_probability).sum(axis=1) - 1.0
    exact_evidence = evidence_strength >= 0.999
    component_sizes = _confident_component_sizes(
        obs, jnp.argmax(belief, axis=1), confidence)
    score = (1000.0 * match_probability
             + 20.0 * confidence[target_card].astype(jnp.float32)
             + 100.0 * evidence_strength[target_card]
             + 10.0 * neighbor_support
             - index.astype(jnp.float32) * 1e-4)
    score = jnp.where(place_legal, score, -1e9)
    place_action = (1624 + jnp.argmax(score)).astype(jnp.int32)
    early_eligible = (place_legal & exact_evidence[target_card]
                      & (component_sizes[target_card] >= 3)
                      & (match_probability > 0.999))
    early_score = jnp.where(early_eligible, score, -1e9)
    early_place_action = (1624 + jnp.argmax(early_score)).astype(jnp.int32)
    while_reveals_remain = jnp.where(jnp.any(early_eligible),
                                     early_place_action, reveal_action)
    return jnp.where(jnp.any(reveal_legal), while_reveals_remain, place_action)


def agent_act(obs, reset, memory, rng):
    del rng
    initial = init_agent()
    previous = jnp.where(reset, initial['previous_positions'],
                         memory['previous_positions'])
    signalled = jnp.where(reset, initial['signalled_color'],
                          memory['signalled_color'])
    signal_checked = jnp.where(reset, initial['signal_checked'],
                               memory['signal_checked'])
    row_matches = jnp.where(reset, initial['row_matches'],
                            memory['row_matches'])
    row_mismatches = jnp.where(reset, initial['row_mismatches'],
                               memory['row_mismatches'])
    substep = obs.node_features[0, 13].astype(jnp.int32)
    active = obs.node_features[0, 12] > 0
    moved = jnp.any(obs.positions != previous, axis=1)
    moved_card = jnp.argmax(moved.astype(jnp.int32))
    destination_row = obs.positions[moved_card, 0] - 3
    valid_row = (destination_row >= 0) & (destination_row < 4)
    mover_knew_card = obs.node_features[moved_card, 11] > 0
    valid_signal = ((~active) & (substep == 3) & (moved.sum() == 1)
                    & valid_row & mover_knew_card)
    inferred_update = jnp.where(jnp.arange(16) == moved_card,
                                destination_row.astype(jnp.int8), signalled)
    signalled = jnp.where(valid_signal, inferred_update, signalled)

    own = obs.node_features[:16, :4]
    own_known_now = own.sum(axis=1) > 0
    newly_checkable = own_known_now & (signalled >= 0) & (~signal_checked)
    observed_label = jnp.argmax(own, axis=1).astype(jnp.int8)
    row_matches = row_matches + jnp.sum(
        newly_checkable & (observed_label == signalled)).astype(jnp.int8)
    row_mismatches = row_mismatches + jnp.sum(
        newly_checkable & (observed_label != signalled)).astype(jnp.int8)
    signal_checked = signal_checked | newly_checkable
    row_trusted = row_matches >= row_mismatches

    belief, confidence, own_known, evidence_strength = _beliefs(
        obs, signalled, row_trusted)
    labels = jnp.argmax(belief, axis=1)
    can_end = jnp.all(confidence) & _groups_connected(obs, labels)

    observe_action = _choose_observation(obs, own_known)
    step0_action = jnp.where(can_end & obs.legal_actions[2324],
                             jnp.asarray(2324, dtype=jnp.int32),
                             observe_action)
    move_action = _choose_move(obs, belief, confidence, evidence_strength)
    hint_action = _choose_hint(obs, belief, confidence, evidence_strength)
    proposed = jax.lax.switch(substep, (
        lambda: step0_action,
        lambda: observe_action,
        lambda: move_action,
        lambda: hint_action,
    ))
    action = jnp.where(active, proposed, jnp.asarray(0, dtype=jnp.int32))
    next_memory = {
        'previous_positions': obs.positions.astype(jnp.int32),
        'signalled_color': signalled.astype(jnp.int8),
        'signal_checked': signal_checked,
        'row_matches': row_matches.astype(jnp.int8),
        'row_mismatches': row_mismatches.astype(jnp.int8),
    }
    trace = {
        'known_count': own_known.sum().astype(jnp.int32),
        'confident_count': confidence.sum().astype(jnp.int32),
        'can_end': can_end,
        'row_trusted': row_trusted,
    }
    return action.astype(jnp.int32), next_memory, trace
