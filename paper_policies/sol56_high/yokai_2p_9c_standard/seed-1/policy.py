"""Memory-based, convention-light policy for the two-player Yōkai task."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'shared_knowledge_clusterer', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

def init_agent():
    return {
        'colors': jnp.full((9,), -1, dtype=jnp.int8),
        'partner_seen': jnp.zeros((9,), dtype=jnp.bool_),
    }


def _updated_memory(obs, reset, memory):
    colors = jnp.where(reset, jnp.full((9,), -1, dtype=jnp.int8), memory['colors'])
    partner_seen = jnp.where(reset, jnp.zeros((9,), dtype=jnp.bool_), memory['partner_seen'])
    visible = jnp.any(obs.node_features[:9, :3] != 0, axis=1)
    visible_colors = jnp.argmax(obs.node_features[:9, :3], axis=1).astype(jnp.int8)
    colors = jnp.where(visible, visible_colors, colors)
    partner_seen = partner_seen | (obs.node_features[:9, 9] != 0)
    singleton = obs.node_features[9, 3:6] != 0
    singleton_placed = obs.node_features[9, 8] != 0
    singleton_target = jnp.argmax(obs.adjacency_matrix[9, :9])
    singleton_color = jnp.argmax(singleton).astype(jnp.int8)
    accept_singleton = (singleton_placed & partner_seen[singleton_target]
                        & (colors[singleton_target] < 0))
    colors = colors.at[singleton_target].set(
        jnp.where(accept_singleton, singleton_color, colors[singleton_target])
    )
    partner_seen = partner_seen.at[singleton_target].set(
        partner_seen[singleton_target] | singleton_placed
    )
    unknown = colors < 0
    counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    missing_color = jnp.argmin(counts).astype(jnp.int8)
    # More generally, once two complete triplets are known, every still-hidden
    # card must have the remaining color (occasionally useful before turn 8).
    inference_certain = (jnp.sum(unknown) == 1) | (jnp.sum(counts == 3) >= 2)
    colors = jnp.where(inference_certain & unknown, missing_color, colors)
    return {'colors': colors, 'partner_seen': partner_seen}


def _observe_action(obs, memory):
    legal = obs.legal_actions[1:10]
    unknown = memory['colors'] < 0
    unlocked = obs.node_features[:9, 8] == 0
    degree = jnp.sum(obs.adjacency_matrix[:9, :9] != 0, axis=1) - 1
    priority = (100 * (unknown & memory['partner_seen']).astype(jnp.int32)
                + 50 * unknown.astype(jnp.int32)
                + 10 * degree.astype(jnp.int32)
                + unlocked.astype(jnp.int32)
                - jnp.arange(9, dtype=jnp.int32))
    return (1 + jnp.argmax(jnp.where(legal, priority, -10000))).astype(jnp.int32)


def _move_action(obs, memory):
    k = jnp.arange(729, dtype=jnp.int32)
    moved_card = k // 81
    cell = k % 81
    destination = jnp.stack([cell // 9, cell % 9], axis=1)
    candidates = jnp.broadcast_to(obs.positions, (729, 9, 2))
    candidates = candidates.at[jnp.arange(729), moved_card].set(destination)
    delta = candidates[:, :, None, :] - candidates[:, None, :, :]
    manhattan = jnp.sum(jnp.abs(delta), axis=-1)
    colors = memory['colors'].astype(jnp.int32)
    known = colors >= 0
    same = (known[:, None] & known[None, :]
            & (colors[:, None] == colors[None, :])
            & (~jnp.eye(9, dtype=jnp.bool_)))
    shared = memory['partner_seen'] & known
    shared_pair = same & shared[:, None] & shared[None, :]
    # Conditional on observations, unknown identities are exchangeable.  Use
    # the exact remaining color counts to score the expected same-color cost.
    counts = jnp.stack([jnp.sum(colors == c) for c in range(3)]).astype(jnp.float32)
    remaining = jnp.maximum(3.0 - counts, 0.0)
    n_unknown = jnp.maximum(jnp.sum(~known).astype(jnp.float32), 1.0)
    safe_colors = jnp.maximum(colors, 0)
    known_unknown_p = remaining[safe_colors] / n_unknown
    both_unknown_p = (jnp.sum(remaining * (remaining - 1.0))
                      / jnp.maximum(n_unknown * (n_unknown - 1.0), 1.0))
    expected_same = jnp.where(
        known[:, None] & known[None, :], same.astype(jnp.float32),
        jnp.where(known[:, None] & (~known[None, :]), known_unknown_p[:, None],
                  jnp.where((~known[:, None]) & known[None, :], known_unknown_p[None, :],
                            both_unknown_p)),
    )
    expected_same = expected_same * (1.0 - jnp.eye(9, dtype=jnp.float32))
    pair_cost = jnp.sum(manhattan * expected_same[None, :, :], axis=(1, 2))
    shared_cost = jnp.sum(manhattan * shared_pair[None, :, :], axis=(1, 2))
    adjacency_reward = jnp.sum((manhattan == 1) * expected_same[None, :, :], axis=(1, 2))
    old = obs.positions[moved_card]
    move_distance = jnp.sum(jnp.abs(destination - old), axis=1)
    score = (-20.0 * pair_cost - 10.0 * shared_cost
             + 300.0 * adjacency_reward - move_distance)
    legal = obs.legal_actions[10:739]
    return (10 + jnp.argmax(jnp.where(legal, score, -10000000))).astype(jnp.int32)


def _hint_action(obs, memory):
    reveal_legal = obs.legal_actions[739:743]
    any_reveal = jnp.any(reveal_legal)
    reveal_action = (739 + jnp.argmax(reveal_legal)).astype(jnp.int32)
    k = jnp.arange(324, dtype=jnp.int32)
    hint = k // 81
    cell = k % 81
    x = cell // 9
    y = cell % 9
    pos = obs.positions
    at_cell = (pos[:, 0][None, :] == x[:, None]) & (pos[:, 1][None, :] == y[:, None])
    card = jnp.argmax(at_cell, axis=1)
    occupied = jnp.any(at_cell, axis=1)
    color = memory['colors'][card].astype(jnp.int32)
    known = color >= 0
    hint_colors = obs.node_features[9:13, 3:6] != 0
    matches = known & hint_colors[hint, jnp.maximum(color, 0)]
    adj = obs.adjacency_matrix[:9, :9] != 0
    same = ((memory['colors'][:, None] == memory['colors'][None, :])
            & (memory['colors'][:, None] >= 0))
    same_degree = jnp.sum(adj & same, axis=1) - 1
    hint_size = jnp.sum(hint_colors, axis=1)
    placed_count = jnp.sum(obs.node_features[9:13, 8] != 0)
    early_flexibility = (placed_count < 3) * hint_size[hint]
    informative_singleton = (matches & (hint_size[hint] == 1)
                             & (~memory['partner_seen'][card]))
    placement_score = (500 * informative_singleton + 1000 * matches + 300 * same_degree[card]
                       + 50 * early_flexibility + 20 * known - card)
    place_legal = obs.legal_actions[743:1067] & occupied
    place_action = (743 + jnp.argmax(jnp.where(place_legal, placement_score, -100000))).astype(jnp.int32)
    return jnp.where(any_reveal, reveal_action, place_action)


def _solved(obs, colors):
    known_all = jnp.all(colors >= 0)
    adj = obs.adjacency_matrix[:9, :9].astype(jnp.int32)
    adj = adj * (1 - jnp.eye(9, dtype=jnp.int32))
    edge_counts = jnp.stack([
        jnp.sum(adj * ((colors[:, None] == c) & (colors[None, :] == c))) // 2
        for c in range(3)
    ])
    return known_all & jnp.all(edge_counts >= 2)


def _posterior_value(obs, colors):
    """Exact immediate value using remembered colors and the deck multiset."""
    codes = jnp.arange(3 ** 5, dtype=jnp.int32)
    combos = ((codes[:, None] // (3 ** jnp.arange(5))) % 3).astype(jnp.int8)
    unknown = colors < 0
    unknown_rank = jnp.cumsum(unknown.astype(jnp.int32)) - 1
    proposed = combos[:, jnp.clip(unknown_rank, 0, 4)]
    assignments = jnp.where(unknown[None, :], proposed, colors[None, :])
    balanced = jnp.all(jnp.stack([
        jnp.sum(assignments == c, axis=1) == 3 for c in range(3)
    ], axis=1), axis=1)
    adj = obs.adjacency_matrix[:9, :9].astype(jnp.int32)
    adj = adj * (1 - jnp.eye(9, dtype=jnp.int32))
    connected = []
    for c in range(3):
        members = assignments == c
        edges = jnp.sum(
            adj[None, :, :] * members[:, :, None] * members[:, None, :], axis=(1, 2)
        ) // 2
        connected.append(edges >= 2)
    hint_colors = obs.node_features[9:13, 3:6] != 0
    placed = obs.node_features[9:13, 8] != 0
    targets = jnp.argmax(obs.adjacency_matrix[9:13, :9], axis=1)
    target_colors = assignments[:, targets]
    correct = jnp.take_along_axis(
        hint_colors[None, :, :], target_colors[:, :, None], axis=2
    )[:, :, 0]
    plausible = balanced & jnp.all((~placed)[None, :] | correct, axis=1)
    successful = plausible & jnp.all(jnp.stack(connected, axis=1), axis=1)
    probability = (jnp.sum(successful).astype(jnp.float32)
                   / jnp.maximum(jnp.sum(plausible), 1))
    revealed = jnp.any(hint_colors, axis=1)
    all_correct_score = 5 * jnp.sum(~revealed) + 2 * jnp.sum(revealed & (~placed)) + jnp.sum(placed)
    return probability * all_correct_score


def agent_act(obs, reset, memory, rng):
    del rng
    memory = _updated_memory(obs, reset, memory)
    active = obs.node_features[0, 10] != 0
    substep = obs.node_features[0, 11].astype(jnp.int32)
    observe_action = _observe_action(obs, memory)
    move_action = _move_action(obs, memory)
    hint_action = _hint_action(obs, memory)
    regular = jnp.where(substep < 2, observe_action,
                        jnp.where(substep == 2, move_action, hint_action))
    posterior_value = _posterior_value(obs, memory['colors'])
    unknown_count = jnp.sum(memory['colors'] < 0)
    posterior_stop = ((unknown_count > 0) & (unknown_count <= 5)
                      & (posterior_value > 3.9))
    end_now = ((substep == 0) & _solved(obs, memory['colors'])
               & obs.legal_actions[1067]) | ((substep == 0) & posterior_stop
                                             & obs.legal_actions[1067])
    action = jnp.where(active, jnp.where(end_now, 1067, regular), 0).astype(jnp.int32)
    trace = {
        'known_count': jnp.sum(memory['colors'] >= 0).astype(jnp.int32),
        'partner_seen_count': jnp.sum(memory['partner_seen']).astype(jnp.int32),
        'solved_belief': _solved(obs, memory['colors']),
        'posterior_value': posterior_value,
    }
    return action, memory, trace
