"""Local Bayesian card memory and convention-free color clustering."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_reward_stopping', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def assignments():
    rows = []
    for n in range(3**9):
        row = tuple((n // (3**i)) % 3 for i in range(9))
        if row.count(0) == 3 and row.count(1) == 3:
            rows.append(row)
    return jnp.asarray(rows, dtype=jnp.int32)


def init_agent():
    return {'known': jnp.full((9,), -1, jnp.int32), 'partner_seen': jnp.zeros((9,), jnp.bool_), 'prev_pos': jnp.zeros((9, 2), jnp.int32), 'evidence': jnp.zeros((9, 9), jnp.float32)}


def beliefs(obs, known, evidence):
    worlds = assignments()
    consistent = jnp.all((known[None, :] < 0) | (worlds == known[None, :]), axis=1)
    hint_links = obs.adjacency_matrix[9:, :9].astype(jnp.float32)
    hint_sets = obs.node_features[9:, 3:6].astype(jnp.float32)
    hint_placed = obs.node_features[9:, 8]
    allowed = hint_links.T @ hint_sets
    has_hint = jnp.sum(hint_links, axis=0) > 0
    world_hint_ok = jnp.take_along_axis(allowed[None, :, :], worlds[:, :, None], axis=2)[:, :, 0]
    likelihood = jnp.prod(jnp.where(has_hint[None, :], 0.02 + 0.98*world_hint_ok, 1.0), axis=1)
    log_evidence = jnp.einsum('wij,ij->w', (worlds[:, :, None] == worlds[:, None, :]).astype(jnp.float32), evidence) * 0.5
    weights = consistent.astype(jnp.float32) * likelihood * jnp.exp(log_evidence - jnp.max(log_evidence))
    weights = weights / jnp.maximum(jnp.sum(weights), 1e-20)
    onehot = jax.nn.one_hot(worlds, 3)
    probs = jnp.einsum('w,wic->ic', weights, onehot)
    same = jnp.einsum('w,wij->ij', weights, (worlds[:, :, None] == worlds[:, None, :]).astype(jnp.float32))
    adj = obs.adjacency_matrix[:9, :9].astype(jnp.float32) - jnp.eye(9)
    edge_counts = jnp.einsum('wij,ij->w', (worlds[:, :, None] == worlds[:, None, :]).astype(jnp.float32), adj) * 0.5
    solved = jnp.sum(weights * (edge_counts >= 5.99))
    return probs, same, solved


def move_scores(pos, same):
    cards = jnp.arange(729) // 81
    cells = jnp.arange(729) % 81
    dest = jnp.stack((cells // 9, cells % 9), axis=1)
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    newdist = jnp.abs(dest[:, None, :] - pos[None, :, :]).sum(-1)
    other = jnp.arange(9)[None, :] != cards[:, None]
    affinity = same[cards] * other
    delta_edges = jnp.sum(affinity * ((newdist == 1).astype(jnp.float32) - (dist[cards] == 1)), axis=1)
    delta_distance = jnp.sum(affinity * (newdist - dist[cards]), axis=1)
    return delta_edges - 0.015*delta_distance


def hint_action(obs, probs, same):
    legal = obs.legal_actions
    pos = obs.positions
    adj = obs.adjacency_matrix[:9, :9].astype(jnp.float32) - jnp.eye(9)
    settled = jnp.sum(same * adj, axis=1)
    # Favor compatible cards whose same-color neighbors already support them.
    hint_prob = obs.node_features[9:, 3:6].astype(jnp.float32) @ probs.T
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    locked = obs.node_features[:9, 8].astype(jnp.float32)
    lock_conflict = jnp.sum(same * locked[None, :] * (dist > 2), axis=1)
    placement_score = 3.0*hint_prob + 0.7*settled[None, :] - 3.0*lock_conflict[None, :]
    placement_actions = 743 + jnp.arange(4)[:, None]*81 + (pos[:, 0]*9 + pos[:, 1])[None, :]
    placement_score = jnp.where(legal[placement_actions], placement_score, -1e9)
    safe_early = (hint_prob > 0.995) & (settled[None, :] > 0.99) & (lock_conflict[None, :] < 0.01)
    # Singletons communicate a color, while pairs are reserved for a firm cluster.
    safe_early = safe_early & ((jnp.arange(4)[:, None] == 0) | (settled[None, :] > 1.9))
    can_reveal = jnp.any(legal[739:743])
    early_score = jnp.where(safe_early, placement_score + (jnp.arange(4)[:, None] == 0)*0.5, -1e9)
    use_early = can_reveal & (jnp.max(early_score) > -1e8)
    placement_score = jnp.where(use_early, early_score, placement_score)
    flat = jnp.argmax(placement_score)
    place = placement_actions.reshape(-1)[flat]
    reveal_scores = jnp.where(legal[739:743], jnp.asarray([4., 3., 2., 1.]), -1e9)
    reveal = 739 + jnp.argmax(reveal_scores)
    return jnp.where(can_reveal & ~use_early, reveal, place).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    memory = jax.tree.map(lambda x, z: jnp.where(reset, z, x), memory, init_agent())
    visible = jnp.sum(obs.node_features[:9, :3], axis=1) > 0
    known = jnp.where(visible, jnp.argmax(obs.node_features[:9, :3], axis=1), memory['known']).astype(jnp.int32)
    partner_seen = memory['partner_seen'] | (obs.node_features[:9, 9] > 0)
    old_adj = jnp.abs(memory['prev_pos'][:, None, :] - memory['prev_pos'][None, :, :]).sum(-1) == 1
    new_adj = jnp.abs(obs.positions[:, None, :] - obs.positions[None, :, :]).sum(-1) == 1
    delta = new_adj.astype(jnp.float32) - old_adj.astype(jnp.float32)
    partner_move = (obs.node_features[0, 10] == 0) & (obs.node_features[0, 11] == 3)
    seen_pair = partner_seen[:, None] & partner_seen[None, :]
    evidence = jnp.clip(memory['evidence'] + jnp.where(partner_move, 2.0 * delta * seen_pair, 0.0), -4.0, 4.0)
    probs, same, solved = beliefs(obs, known, evidence)
    entropy = -jnp.sum(probs * jnp.log(jnp.maximum(probs, 1e-12)), axis=1)
    adj = obs.adjacency_matrix[:9, :9].astype(jnp.float32) - jnp.eye(9)
    info_score = entropy + 0.05*(adj @ visible.astype(jnp.float32)) + 0.001*jax.random.uniform(rng, (9,))
    observe = 1 + jnp.argmax(jnp.where(obs.legal_actions[1:10], info_score, -1e9))
    move_vals = move_scores(obs.positions, same)
    move = 10 + jnp.argmax(jnp.where(obs.legal_actions[10:739], move_vals, -1e9))
    hint = hint_action(obs, probs, same)
    substep = obs.node_features[0, 11].astype(jnp.int32)
    action = jnp.where(substep < 2, observe, jnp.where(substep == 2, move, hint))
    hint_sets = obs.node_features[9:, 3:6].astype(jnp.float32)
    revealed = jnp.sum(hint_sets, axis=1) > 0
    placed = obs.node_features[9:, 8] > 0
    hint_card_probs = obs.adjacency_matrix[9:, :9].astype(jnp.float32) @ probs
    correct_probs = jnp.sum(hint_card_probs * hint_sets, axis=1)
    potential_reward = jnp.sum(jnp.where(placed, 2*correct_probs - 1, jnp.where(revealed, 2., 5.)))
    should_end = (solved > 0.95) | ((solved > 0.65) & (solved * potential_reward > 4.4))
    action = jnp.where(obs.legal_actions[1067] & should_end, 1067, action)
    action = jnp.where(obs.node_features[0, 10] > 0, action, 0).astype(jnp.int32)
    return action, {'known': known, 'partner_seen': partner_seen, 'prev_pos': obs.positions, 'evidence': evidence}, {'known': known, 'probs': probs, 'solved': solved, 'expected_edges': jnp.sum(same*adj)*0.5}
