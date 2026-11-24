"""Local color memory and count-constrained cooperative grouping."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'adaptive_partner_evidence', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def assignments():
    rows = []
    for n in range(19683):
        row = tuple((n // (3 ** i)) % 3 for i in range(9))
        if row.count(0) == 3 and row.count(1) == 3:
            rows.append(row)
    return jnp.asarray(rows, jnp.int32)


def init_agent():
    return {'colors': jnp.full((9,), -1, jnp.int32),
            'partner_seen': jnp.zeros((9,), jnp.bool_),
            'positions': jnp.zeros((9, 2), jnp.int32),
            'relations': jnp.zeros((9, 9), jnp.float32),
            'probabilities': jnp.full((9, 3), 1./3., jnp.float32),
            'trust': jnp.asarray(1., jnp.float32)}


def move_values(pos, same):
    cells = jnp.stack((jnp.arange(81) // 9, jnp.arange(81) % 9), axis=1)
    d = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
    nd = jnp.sum(jnp.abs(cells[:, None, :] - pos[None, :, :]), axis=-1)
    old = jnp.sum(same * ((d == 1) - .08 * jnp.minimum(d, 5)), axis=1)
    new = same @ ((nd == 1) - .08 * jnp.minimum(nd, 5)).T
    compact = jnp.sum(d, axis=1)[:, None] - (jnp.sum(nd, axis=1)[:, None] - nd).T
    return (new - old[:, None] + .002 * compact).reshape(-1)


def geometric_legal(pos, locked):
    d = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
    keep = jnp.arange(9)[:, None] != jnp.arange(9)[None, :]
    reach = (d[None, :, :] <= 1) & keep[:, :, None] & keep[:, None, :]
    def closure(k, r):
        return r | (r[:, :, k, None] & r[:, None, k, :])
    reach = jax.lax.fori_loop(0, 9, closure, reach)
    cells = jnp.stack((jnp.arange(81) // 9, jnp.arange(81) % 9), axis=1)
    nd = jnp.sum(jnp.abs(cells[:, None, :] - pos[None, :, :]), axis=-1)
    touched = jnp.einsum('ijk,xk->ixj', reach.astype(jnp.float32), (nd == 1).astype(jnp.float32)) > 0
    connected = jnp.all(touched | (~keep[:, None, :]), axis=-1)
    vacant = ~jnp.any(nd == 0, axis=-1)
    return (connected & vacant[None, :] & (~locked[:, None])).reshape(-1)


def observation_values(pos, locked, ass, weights, probs):
    onehot = jax.nn.one_hot(ass, 3)
    case_weights = (weights[:, None, None] * onehot).reshape((1680, 27))
    equivalence = (ass[:, :, None] == ass[:, None, :]).astype(jnp.float32).reshape((1680, 81))
    joints = jnp.matmul(case_weights.T, equivalence, precision=jax.lax.Precision.HIGHEST)
    conditional = joints.reshape((27, 9, 9)) / jnp.maximum(probs.reshape((27, 1, 1)), 1e-8)
    conditional *= (1 - jnp.eye(9))[None, :, :]
    values = jax.vmap(move_values, in_axes=(None, 0))(pos, conditional)
    allowed = geometric_legal(pos, locked)
    best = jnp.max(jnp.where(allowed[None, :], values, -1e9), axis=1).reshape((9, 3))
    return jnp.sum(probs * best, axis=1)


def future_value(pos, same, locked):
    scores = move_values(pos, same)
    occupancy = jnp.any((pos[:, 0, None] * 9 + pos[:, 1, None]) == jnp.arange(81)[None, :], axis=0)
    allowed = ((~locked[:, None]) & (~occupancy[None, :])).reshape(-1)
    values, actions = jax.lax.top_k(jnp.where(allowed, scores, -1e9), 24)
    ids = actions // 81
    cells = actions % 81
    boards = jnp.broadcast_to(pos, (24, 9, 2))
    boards = boards.at[jnp.arange(24), ids].set(jnp.stack((cells // 9, cells % 9), axis=1))
    d = jnp.sum(jnp.abs(boards[:, :, None, :] - boards[:, None, :, :]), axis=-1)
    adj = d == 1
    reach = jnp.broadcast_to(jnp.arange(9) == 0, (24, 9))
    def spread(_, r):
        return r | jnp.any(adj & r[:, :, None], axis=1)
    reach = jax.lax.fori_loop(0, 8, spread, reach)
    connected = jnp.all(reach, axis=1)
    return jnp.maximum(0., jnp.max(jnp.where(connected, values, -1e9)))


def planned_move(pos, same, locked, legal):
    values, actions = jax.lax.top_k(jnp.where(legal, move_values(pos, same), -1e9), 16)
    ids = actions // 81
    cells = actions % 81
    boards = jnp.broadcast_to(pos, (16, 9, 2))
    boards = boards.at[jnp.arange(16), ids].set(jnp.stack((cells // 9, cells % 9), axis=1))
    continuation = jax.vmap(future_value, in_axes=(0, None, None))(boards, same, locked)
    return 10 + actions[jnp.argmax(values + .8 * continuation)]


def agent_act(obs, reset, memory, rng):
    colors = jnp.where(reset, -1, memory['colors'])
    f = obs.node_features
    visible = jnp.sum(f[:9, :3], axis=1) > 0
    last_probs = jnp.where(reset, 1./3., memory['probabilities'])
    seen_color = jnp.argmax(f[:9, :3], axis=1)
    revealed_probability = jnp.take_along_axis(last_probs, seen_color[:, None], axis=1)[:, 0]
    surprising = jnp.any(visible & (colors < 0) & (revealed_probability < .06))
    trust = jnp.where(reset, 1., memory['trust'])
    trust = jnp.maximum(.25, trust * jnp.where(surprising, .5, 1.))
    colors = jnp.where(visible, jnp.argmax(f[:9, :3], axis=1), colors).astype(jnp.int32)
    pos = obs.positions
    partner_seen = jnp.where(reset, False, memory['partner_seen']) | (f[:9, 9] > 0)
    previous_pos = jnp.where(reset, pos, memory['positions'])
    previous_dist = jnp.sum(jnp.abs(previous_pos[:, None, :] - previous_pos[None, :, :]), axis=-1)
    current_dist = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
    partner_move = (f[0, 10] == 0) & (f[0, 11] == 3) & jnp.any(pos != previous_pos)
    seen_count = partner_seen[:, None].astype(jnp.float32) + partner_seen[None, :].astype(jnp.float32)
    confidence = trust * jnp.where(seen_count == 2, 2.5, jnp.where(seen_count == 1, .8, .2))
    delta = (current_dist == 1).astype(jnp.float32) - (previous_dist == 1).astype(jnp.float32)
    delta -= .06 * (current_dist - previous_dist)
    relations = jnp.where(reset, 0., memory['relations']) * jnp.where(surprising, .5, 1.)
    relations = jnp.clip(relations + jnp.where(partner_move, confidence * delta, 0.), -5., 5.)
    legal = obs.legal_actions
    locked = f[:9, 8] > 0
    hcolors = f[9:, 3:6].astype(jnp.float32)
    placed = f[9:, 8] > 0
    revealed = jnp.sum(hcolors, axis=1) > 0
    targets = obs.adjacency_matrix[9:, :9].astype(jnp.float32)
    sets = jnp.einsum('hi,hc->ic', targets, hcolors)
    hinted = jnp.sum(sets, axis=1) > 0
    # Hints are evidence of intent, not infallible observations.
    ass = assignments()
    consistent = jnp.all((colors[None, :] < 0) | (ass == colors[None, :]), axis=1)
    matching_hints = jnp.take_along_axis(sets[None, :, :], ass[:, :, None], axis=2)[:, :, 0] > .5
    all_hints_compatible = jnp.any(consistent & jnp.all(matching_hints | (~hinted[None, :]), axis=1))
    hint_floor = jnp.where(all_hints_compatible, .03, .25)
    hint_like = jnp.where(hinted[:, None], hint_floor + (1 - hint_floor) * sets, 1.)
    likelihood = jnp.prod(jnp.take_along_axis(hint_like[None, :, :], ass[:, :, None], axis=2)[:, :, 0], axis=1)
    energy = jnp.sum((ass[:, :, None] == ass[:, None, :]) * relations[None, :, :], axis=(1, 2)) / 2
    energy -= jnp.max(jnp.where(consistent, energy, -1e9))
    weights = consistent.astype(jnp.float32) * likelihood * jnp.exp(energy)
    weights = weights / jnp.maximum(jnp.sum(weights), 1e-20)
    probs = jnp.einsum('s,sic->ic', weights, jax.nn.one_hot(ass, 3))
    same = jnp.einsum('s,sij->ij', weights, (ass[:, :, None] == ass[:, None, :]).astype(jnp.float32))
    same = same * (1 - jnp.eye(9))
    dist = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
    adj = (dist == 1).astype(jnp.float32)
    edges = jnp.sum((ass[:, :, None] == ass[:, None, :]) * adj[None, :, :], axis=(1, 2)) / 2
    pwin = jnp.sum(weights * (edges == 6))
    unrevealed_count = jnp.sum(~revealed)
    pending_count = jnp.sum(revealed & (~placed))
    placed_count = jnp.sum(placed)
    hint_correct = jnp.sum(jnp.take_along_axis(sets[None, :, :], ass[:, :, None], axis=2)[:, :, 0], axis=1)
    scenario_reward = 5 * unrevealed_count + 2 * pending_count + 2 * hint_correct - placed_count
    stop_value = jnp.sum(weights * (edges == 6) * scenario_reward)
    next_cost = jnp.where(pending_count > 0, 1, 3)
    continue_ceiling = 5 * unrevealed_count + 2 * pending_count + placed_count - next_cost
    stop_dominates = stop_value > continue_ceiling + .02
    entropy = -jnp.sum(probs * jnp.log(jnp.maximum(probs, 1e-20)), axis=1)
    # Prefer unknown cards near already identified cards and near the center.
    influence = jnp.sum(adj * (colors[None, :] >= 0), axis=1)
    observe_score = observation_values(pos, locked, ass, weights, probs) + .15 * entropy + .0001 * jax.random.uniform(rng, (9,))
    observation_action = 1 + jnp.argmax(jnp.where(legal[1:10], observe_score, -1e9))
    observation_action = jnp.where(legal[1067] & ((pwin > .90) | stop_dominates), 1067, observation_action)

    move_action = planned_move(pos, same, locked, legal[10:739])

    match = jnp.einsum('ic,hc->hi', probs, hcolors)
    support = jnp.sum(same * adj, axis=1)
    lock_bad = jnp.sum(same * locked[None, :] * jnp.maximum(dist - 2, 0), axis=1)
    card_score = .35 * support - 2. * lock_bad + .03 * jnp.sum(adj, axis=1)
    available_moves = geometric_legal(pos, locked)
    gains = jnp.where(available_moves, move_values(pos, same), -1e9)
    without_card = jnp.max(jnp.where(jnp.arange(729)[None, :] // 81 != jnp.arange(9)[:, None], gains[None, :], -1e9), axis=1)
    block_cost = jnp.maximum(0., jnp.max(gains) - without_card)
    equality = (ass[:, :, None] == ass[:, None, :]).astype(jnp.float32)
    degrees = jnp.sum(equality * adj[None, :, :], axis=-1)
    group_edges = jnp.sum(equality * degrees[:, None, :], axis=-1) / 2
    completed = jnp.sum(weights[:, None] * (group_edges == 2), axis=0)
    anchored = jnp.sum(same * locked[None, :], axis=1)
    card_score -= 1.5 * block_cost + .4 * anchored * (1 - completed)
    information = (.15 * (3 - jnp.sum(hcolors, axis=1))[:, None]) * (~partner_seen[None, :])
    hint_card_score = 3 * match + card_score[None, :] + information
    # Placements address cells; gather occupancy and reject empty cells through legal mask.
    cell_ids = jnp.arange(81)
    at = (pos[:, 0, None] * 9 + pos[:, 1, None]) == cell_ids[None, :]
    place_scores = jnp.einsum('hi,ix->hx', hint_card_score, at.astype(jnp.float32)).reshape(-1)
    place_action = 743 + jnp.argmax(jnp.where(legal[743:1067], place_scores, -1e9))
    reveal_action = 739 + jnp.argmax(legal[739:743])
    hint_action = jnp.where(jnp.any(legal[739:743]), reveal_action, place_action)
    safe_card = (support > .85) & (lock_bad < .01) & (block_cost < .15) & ((anchored < .1) | (completed > .9)) & (~locked)
    safe_hint_card = safe_card[None, :] & (match > .99) & revealed[:, None] & (~placed[:, None])
    early_scores = jnp.where(safe_hint_card, hint_card_score, -1e9)
    best_early = jnp.argmax(early_scores)
    early_hint = best_early // 9
    early_card = best_early % 9
    early_action = 743 + early_hint * 81 + pos[early_card, 0] * 9 + pos[early_card, 1]
    hint_action = jnp.where((jnp.max(early_scores) > -1e8) & legal[early_action], early_action, hint_action)
    step = f[0, 11]
    action = jnp.where(step < 2, observation_action, jnp.where(step == 2, move_action, hint_action))
    action = jnp.where(f[0, 10] > 0, action, 0).astype(jnp.int32)
    return action, {'colors': colors, 'partner_seen': partner_seen, 'positions': pos, 'relations': relations, 'probabilities': probs, 'trust': trust}, {'colors': colors, 'trust': trust, 'hint_floor': hint_floor, 'pwin': pwin, 'stop_value': stop_value, 'continue_ceiling': continue_ceiling, 'probabilities': probs, 'positions': pos, 'step': step, 'active': f[0, 10]}
