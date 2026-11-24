"""Local evidence and geometric color grouping for cooperative Yokai."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return {"colors": jnp.zeros((16, 4), jnp.float32),
            "positions": jnp.zeros((16, 2), jnp.int32),
            "relations": jnp.zeros((16, 16), jnp.float32),
            "future": jnp.zeros((16,), jnp.bool_)}


def beliefs(obs, seen):
    hints = obs.node_features[16:, 4:8].astype(jnp.float32)
    attached = obs.adjacency_matrix[:16, 16:].astype(jnp.float32)
    support = attached @ hints
    allowed = jnp.where(jnp.sum(support, axis=1, keepdims=True) > 0, support, 1.)
    known = jnp.sum(seen, axis=1, keepdims=True) > 0
    domains = jnp.where(known, seen > 0, allowed > 0)
    def force(_, d):
        singleton = jnp.sum(d, axis=1, keepdims=True) == 1
        fixed_count = jnp.sum(d & singleton, axis=0, keepdims=True)
        possible_count = jnp.sum(d, axis=0, keepdims=True)
        d = d & ((fixed_count < 4) | singleton)
        forced = d & (possible_count == 4)
        d = jnp.where(jnp.any(forced, axis=1, keepdims=True), forced, d)
        return d
    domains = jax.lax.fori_loop(0, 16, force, domains)
    # Fall back to own observations if an independently supplied hint conflicts.
    domains = jnp.where(jnp.any(domains, axis=1, keepdims=True), domains, jnp.where(known, seen > 0, True))
    known = jnp.sum(domains, axis=1, keepdims=True) == 1
    seen = domains.astype(jnp.float32)
    p = seen / jnp.sum(seen, axis=1, keepdims=True)
    def balance(_, p):
        remain = jnp.maximum(4. - jnp.sum(jnp.where(known, seen, 0.), axis=0), 0.)
        u = jnp.where(known, 0., p)
        u = u * remain / jnp.maximum(jnp.sum(u, axis=0), 1e-6)
        u = u / jnp.maximum(jnp.sum(u, axis=1, keepdims=True), 1e-6)
        return jnp.where(known, seen, u)
    return jax.lax.fori_loop(0, 8, balance, p)


def removed_components(edge):
    ids = jnp.arange(16)
    present = ids[None, :] != ids[:, None]
    edges = edge[None] & present[:, :, None] & present[:, None, :]
    labels = jnp.broadcast_to(ids, (16, 16))
    def propagate(_, lab):
        return jnp.minimum(lab, jnp.min(jnp.where(edges, lab[:, None, :], 16), axis=-1))
    labels = jax.lax.fori_loop(0, 15, propagate, labels)
    reach = (labels[:, :, None] == labels[:, None, :]) & present[:, :, None] & present[:, None, :]
    reps = (labels == ids[None, :]) & present
    return reach, reps


def motion_beliefs(p, relations):
    fixed = jnp.max(p, axis=1, keepdims=True) > .999
    remaining = jnp.maximum(4. - jnp.sum(jnp.where(fixed, p, 0.), axis=0), 0.)
    def update(_, q):
        q = p * jnp.exp(3. * (relations @ q))
        q = q / jnp.maximum(jnp.sum(q, axis=1, keepdims=True), 1e-6)
        u = jnp.where(fixed, 0., q)
        u = u * remaining / jnp.maximum(jnp.sum(u, axis=0), 1e-6)
        u = u / jnp.maximum(jnp.sum(u, axis=1, keepdims=True), 1e-6)
        return jnp.where(fixed, p, u)
    return jax.lax.fori_loop(0, 8, update, p)


def geometry(obs, p, relations, positions=None):
    ids = jnp.arange(16)
    present = ids[None, :] != ids[:, None]
    pos = obs.positions if positions is None else positions
    dist = jnp.sum(jnp.abs(pos[:, None] - pos[None, :]), axis=-1)
    certain = jnp.max(p, axis=1) > .999
    color = jnp.argmax(p, axis=1)
    edge = (dist == 1) & (color[:, None] == color[None, :]) & certain[:, None] & certain[None, :]
    # Compute the color components after removing each possible moving card.
    reach, reps = removed_components(edge)
    cells = jnp.stack((jnp.arange(100) // 10, jnp.arange(100) % 10), axis=1)
    cd = jnp.sum(jnp.abs(cells[:, None, :] - pos[None, :, :]), axis=-1)
    near = (cd == 1).astype(jnp.float32)
    touched = jnp.einsum('dj,ijk->idk', near, reach.astype(jnp.float32)) > 0
    oldtouched = jnp.einsum('ij,ijk->ik', (dist == 1).astype(jnp.float32), reach.astype(jnp.float32)) > 0
    same = p @ p.T
    confidence = jnp.where(certain, 1., .12 + .6 * jnp.maximum((jnp.max(p, axis=1) - .25) / .75, 0.))
    weights = same * confidence[:, None] * confidence[None, :]
    weights = jnp.where(certain[:, None] & certain[None, :], weights, jnp.maximum(weights, relations))
    comp_gain = jnp.sum((touched.astype(jnp.float32) - oldtouched[:, None, :]) * reps[:, None, :] * weights[:, None, :], axis=-1)
    edge_gain = jnp.einsum('dj,ij->id', near, weights * present) - jnp.sum((dist == 1) * weights, axis=1)[:, None]
    distance_gain = jnp.sum((dist[:, None, :] - cd[None]) * weights[:, None, :] * present[:, None, :], axis=-1) / jnp.maximum(jnp.sum(weights * present, axis=-1)[:, None], 1.)
    degree = jnp.sum(near, axis=1)[None, :] - near.T
    scores = 12. * comp_gain + .65 * edge_gain + .7 * distance_gain + .025 * degree
    fullreach, fullreps = removed_components(dist == 1)
    touch_count = jnp.einsum('dj,ijk->idk', near, fullreach.astype(jnp.float32))
    fulltouch = touch_count > 0
    vacant = ~jnp.any(cd == 0, axis=1)
    legal = vacant[None, :] & jnp.all(fulltouch | ~fullreps[:, None, :], axis=-1) & (obs.node_features[:16, 10:11] == 0)
    unrestricted = jnp.max(jnp.where(vacant[None, :], scores, -1e6), axis=1)
    constrained = jnp.max(jnp.where(legal, scores, -1e6), axis=1)
    blocked = (jnp.sum(fullreps, axis=1) > 1) & certain & (obs.node_features[:16, 10] == 0) & (constrained < 6.) & (unrestricted > 6.)
    bridge_all = jnp.all(fulltouch | ~fullreps[:, None, :], axis=-1)
    comp_size = jnp.sum(fullreach, axis=1)
    loses_contact = (near[None, :, :] > 0) & (touch_count == 1) & (comp_size[:, None, :] > 1)
    bridge = bridge_all[:, :, None] & ~loses_contact & present[:, None, :]
    bridge_value = jnp.max(bridge * (blocked * jnp.minimum(unrestricted - constrained, 12.))[:, None, None], axis=0).T
    raw_scores = jnp.where(legal, scores, -1e6)
    scores = jnp.where(legal, scores + .35 * bridge_value, -1e6)
    # Expected value of learning the moving card's color before choosing its destination.
    target = p * confidence[:, None]
    cg = jnp.einsum('idk,kc->idc', (touched.astype(jnp.float32) - oldtouched[:, None, :]) * reps[:, None, :], target)
    eg = jnp.einsum('idk,kc->idc', (near[None] - (dist == 1)[:, None, :]) * present[:, None, :], target)
    dg = jnp.einsum('idk,kc->idc', (dist[:, None, :] - cd[None]) * present[:, None, :], target)
    dg /= jnp.maximum(jnp.einsum('ik,kc->ic', present, target)[:, None, :], 1.)
    conditional = 12. * cg + .65 * eg + .7 * dg + .025 * degree[:, :, None]
    conditional = jnp.where(legal[:, :, None], conditional, -1e6)
    best_now = jnp.maximum(jnp.max(scores), 0.)
    info_value = jnp.sum(p * jnp.maximum(jnp.max(conditional, axis=1) - best_now, 0.), axis=1)
    return scores, edge, same, certain, info_value, raw_scores


def prepare_move(obs, p, relations, scores, raw_scores):
    _, first = jax.lax.top_k(scores.reshape(-1), 8)
    reveals = jnp.sum(jnp.any(obs.node_features[16:, 4:8] > 0, axis=1))
    placements = jnp.sum(obs.node_features[16:, 10])
    elapsed_turns = reveals + placements
    def follow(action):
        card, cell = action // 100, action % 100
        pos = obs.positions.at[card].set(jnp.stack((cell // 10, cell % 10)))
        adj = jnp.sum(jnp.abs(pos[:, None] - pos[None, :]), axis=-1) == 1
        future = geometry(obs, p, relations * adj, pos)[5]
        future = jnp.where((obs.node_features[:16, 11:12] == 1) | (elapsed_turns < 12), future, -1e6)
        best = jnp.argmax(future)
        return jnp.max(future), best // 100
    values, movers = jax.vmap(follow)(first)
    total = raw_scores.reshape(-1)[first] + .9 * values
    choice = jnp.argmax(total)
    useful = (elapsed_turns < 13) & (jnp.max(scores) < 8.) & (jnp.max(total) > jnp.max(raw_scores) + 4.) & (values[choice] > 6.)
    return first[choice], movers[choice], useful


def agent_act(obs, reset, memory, rng):
    memory = jax.tree.map(lambda x: jnp.where(reset, jnp.zeros_like(x), x), memory)
    features = obs.node_features
    visible = features[:16, :4].astype(jnp.float32)
    seen = jnp.where(jnp.sum(visible, axis=1, keepdims=True) > 0, visible, memory['colors'])
    p = beliefs(obs, seen)
    moved = jnp.any(obs.positions != memory['positions'], axis=1) & ~reset
    adj = jnp.sum(jnp.abs(obs.positions[:, None] - obs.positions[None, :]), axis=-1) == 1
    oldadj = jnp.sum(jnp.abs(memory['positions'][:, None] - memory['positions'][None, :]), axis=-1) == 1
    partner_seen = features[:16, 11] == 1
    suggested = moved[:, None] & adj & ~oldadj & partner_seen[:, None] & partner_seen[None, :] & (features[0, 12] == 0)
    evidence = .9 * suggested / jnp.maximum(jnp.sum(suggested, axis=1, keepdims=True), 1)
    relations = jnp.maximum(memory['relations'] * adj, evidence + evidence.T)
    planning = motion_beliefs(p, relations)
    scores, _, _, _, info_value, raw_scores = geometry(obs, planning, relations)
    certain = jnp.max(p, axis=1) > .999
    same = p @ p.T
    edge = adj & (same > .999) & certain[:, None] & certain[None, :]
    uncertainty = 1. - jnp.max(p, axis=1)
    # Gather fresh evidence, particularly about recent partner relocations.
    observe_score = uncertainty * (10. + 2. * moved + .2 * (1 - features[:16, 11]))
    observe_score += .8 * info_value * (~certain)
    observe_score += .01 * jnp.sum(obs.adjacency_matrix[:16, :16], axis=1)
    observe_action = 1 + jnp.argmax(jnp.where(obs.legal_actions[1:17], observe_score, -1e6))
    move_action = 17 + jnp.argmax(jnp.where(obs.legal_actions[17:1617].reshape(16, 100), scores, -1e6))
    planned_action, planned_mover, useful_plan = prepare_move(obs, planning, relations, scores, raw_scores)
    use_plan = useful_plan & (jnp.sum(certain) >= 12) & (features[0, 12] == 1) & (features[0, 13] == 2)
    move_action = jnp.where(use_plan, 17 + planned_action, move_action)
    future = jnp.where((features[0, 12] == 1) & (features[0, 13] == 2), (jnp.arange(16) == planned_mover) & use_plan, memory['future'])
    hintset = features[16:, 4:8].astype(jnp.float32)
    match = hintset @ p.T
    reach = edge | jnp.eye(16, dtype=jnp.bool_)
    def closure(_, r):
        return (r.astype(jnp.int32) @ r.astype(jnp.int32)) > 0
    reach = jax.lax.fori_loop(0, 4, closure, reach)
    settled = jnp.sum(edge, axis=1).astype(jnp.float32)
    need_move = jnp.maximum(jnp.max(scores, axis=1), 0.)
    split_anchor = jnp.sum((same > .999) & ~reach & (features[:16, 10] == 1)[None, :], axis=1)
    hint_card_score = 100. * match + 3. * settled[None, :] - .5 * need_move[None, :] - 30. * split_anchor[None, :] - 30. * future[None, :]
    # Favor precise truthful public evidence when choosing among available hints.
    hint_card_score += (4. - jnp.sum(hintset, axis=1))[:, None] * 7.
    hint_card_score += (1. - features[:16, 11])[None, :] * (4. - jnp.sum(hintset, axis=1))[:, None]
    at = obs.positions[:, 0] * 10 + obs.positions[:, 1]
    place_scores = jnp.full((7, 100), -1e6).at[:, at].set(hint_card_score)
    place_scores = jnp.where(obs.legal_actions[1624:2324].reshape(7, 100), place_scores, -1e6)
    place_action = 1624 + jnp.argmax(place_scores)
    reveal = obs.legal_actions[1617:1624]
    hint_size = jnp.sum(hintset, axis=1)[:, None]
    precise = (hint_size == 1) & (settled[None, :] >= 1.)
    useful_pair = (hint_size == 2) & (settled[None, :] >= 2.) & (features[:16, 11][None, :] == 0) & (split_anchor[None, :] == 0)
    inform = (precise | useful_pair) & (match > .999) & (need_move[None, :] < 6.)
    inform_scores = jnp.full((7, 100), -1e6).at[:, at].set(jnp.where(inform, hint_card_score, -1e6))
    inform_scores = jnp.where(obs.legal_actions[1624:2324].reshape(7, 100), inform_scores, -1e6)
    can_inform = jnp.max(inform_scores) > -1e5
    hint_action = jnp.where(can_inform, 1624 + jnp.argmax(inform_scores), jnp.where(jnp.any(reveal), 1617 + jnp.argmax(reveal), place_action))
    # A connected graph of each known color certifies success when all are known.
    solved = jnp.all(certain) & jnp.all((same < .999) | reach)
    substep = features[0, 13]
    action = jnp.where(substep < 2, observe_action, jnp.where(substep == 2, move_action, hint_action))
    action = jnp.where((substep == 0) & solved, 2324, action)
    action = jnp.where(features[0, 12] == 1, action, 0).astype(jnp.int32)
    future = jnp.where((features[0, 12] == 1) & (features[0, 13] == 3), False, future)
    memory = {'colors': seen, 'positions': obs.positions, 'relations': relations, 'future': future}
    return action, memory, {'known': jnp.sum(certain), 'solved': solved, 'move_value': jnp.max(scores), 'belief': p, 'planning': planning, 'positions': obs.positions, 'locked': features[:16, 10]}
