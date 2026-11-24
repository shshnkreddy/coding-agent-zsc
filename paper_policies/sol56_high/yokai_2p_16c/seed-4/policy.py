"""Convention-light Yokai policy based on public canonical color rows."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'public_row_clusterer', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.full((16,), -1, dtype=jnp.int8)


def _remember(obs, reset, memory):
    known = jnp.where(reset, jnp.full_like(memory, -1), memory)
    colors = obs.node_features[:16, :4]
    visible = jnp.sum(colors, axis=1) > 0
    seen_color = jnp.argmax(colors, axis=1).astype(jnp.int8)
    return jnp.where(visible, seen_color, known)


def _observe_action(obs, known):
    ids = jnp.arange(16, dtype=jnp.int32)
    partner_seen = obs.node_features[:16, 11] > 0
    unlocked = obs.node_features[:16, 10] == 0
    legal = obs.legal_actions[1:17]
    unknown = known < 0
    degree = jnp.sum(obs.adjacency_matrix[:16, :16] > 0, axis=1) - 1
    priority = (unknown.astype(jnp.int32) * 1000
                + (unknown & ~partner_seen).astype(jnp.int32) * 100
                + unlocked.astype(jnp.int32) * 10
                + (8 - jnp.minimum(degree, 8)) * 20 - ids)
    return (jnp.argmax(jnp.where(legal, priority, -100000)) + 1).astype(jnp.int32)


def _effective_colors(obs, known):
    """Add only public, mechanically checkable color evidence."""
    hint_sets = obs.node_features[16:23, 4:8]
    singleton = (jnp.sum(hint_sets, axis=1) == 1) & (obs.node_features[16:23, 10] > 0)
    hint_color = jnp.argmax(hint_sets, axis=1).astype(jnp.int8)
    attached = obs.adjacency_matrix[:16, 16:23] > 0
    has_public_color = jnp.any(attached & singleton[None, :], axis=1)
    public_color = jnp.argmax((attached & singleton[None, :]).astype(jnp.int32), axis=1)
    public_color = hint_color[public_color]
    effective = jnp.where((known < 0) & has_public_color, public_color, known)

    placed = obs.node_features[16:23, 10] > 0
    attached_placed = attached & placed[None, :]
    has_hint = jnp.any(attached_placed, axis=1)

    # With exactly two unseen cards, the known four-of-each deck can force
    # their colors.  When two assignments remain, current public row evidence
    # breaks the symmetry rather than a seat-specific convention.
    counts = jnp.stack([jnp.sum(effective == c) for c in range(4)])
    deficit = 4 - counts
    unknown = effective < 0
    ids = jnp.arange(16, dtype=jnp.int32)
    u0 = jnp.argmax(unknown.astype(jnp.int32))
    u1 = jnp.argmax((unknown & (ids != u0)).astype(jnp.int32))
    m0 = jnp.argmax((deficit > 0).astype(jnp.int32))
    deficit2 = deficit.at[m0].add(-1)
    m1 = jnp.argmax((deficit2 > 0).astype(jnp.int32))
    x0 = obs.positions[u0, 0]
    x1 = obs.positions[u1, 0]
    allowed = (attached_placed.astype(jnp.int32) @ hint_sets.astype(jnp.int32)) > 0
    direct_conflict = ((has_hint[u0] & ~allowed[u0, m0]).astype(jnp.int32)
                       + (has_hint[u1] & ~allowed[u1, m1]).astype(jnp.int32))
    swapped_conflict = ((has_hint[u0] & ~allowed[u0, m1]).astype(jnp.int32)
                        + (has_hint[u1] & ~allowed[u1, m0]).astype(jnp.int32))
    direct = (20 * direct_conflict + jnp.abs(x0 - (3 + m0))
              + jnp.abs(x1 - (3 + m1)))
    swapped = (20 * swapped_conflict + jnp.abs(x0 - (3 + m1))
               + jnp.abs(x1 - (3 + m0)))
    c0 = jnp.where(direct <= swapped, m0, m1).astype(jnp.int8)
    c1 = jnp.where(direct <= swapped, m1, m0).astype(jnp.int8)
    valid = (jnp.sum(unknown) == 2) & jnp.all(deficit >= 0) & (jnp.sum(deficit) == 2)
    effective = jnp.where(valid & (ids == u0), c0, effective)
    effective = jnp.where(valid & (ids == u1), c1, effective)
    return effective


def _move_action(obs, known):
    known = _effective_colors(obs, known)
    k = jnp.arange(1600, dtype=jnp.int32)
    card = k // 100
    rem = k % 100
    dx = rem // 10
    dy = rem % 10
    pos = obs.positions
    sx = pos[card, 0]
    sy = pos[card, 1]
    color = known[card].astype(jnp.int32)
    is_known = color >= 0

    target_x = 3 + jnp.maximum(color, 0)
    old_row_error = jnp.abs(sx - target_x)
    new_row_error = jnp.abs(dx - target_x)
    row_gain = old_row_error - new_row_error
    old_y_error = jnp.maximum(3 - sy, 0) + jnp.maximum(sy - 6, 0)
    new_y_error = jnp.maximum(3 - dy, 0) + jnp.maximum(dy - 6, 0)

    other_color = known[None, :].astype(jnp.int32)
    other_known = other_color >= 0
    same = other_known & (other_color == color[:, None])
    different = other_known & (other_color != color[:, None])
    not_self = jnp.arange(16, dtype=jnp.int32)[None, :] != card[:, None]
    old_dist = jnp.abs(sx[:, None] - pos[None, :, 0]) + jnp.abs(sy[:, None] - pos[None, :, 1])
    new_dist = jnp.abs(dx[:, None] - pos[None, :, 0]) + jnp.abs(dy[:, None] - pos[None, :, 1])
    old_same_edges = jnp.sum(same & not_self & (old_dist == 1), axis=1)
    new_same_edges = jnp.sum(same & not_self & (new_dist == 1), axis=1)
    new_diff_edges = jnp.sum(different & not_self & (new_dist == 1), axis=1)
    same_distance_gain = jnp.sum(jnp.where(same & not_self, old_dist - new_dist, 0), axis=1)

    base_dist = (jnp.abs(pos[:, None, 0] - pos[None, :, 0])
                 + jnp.abs(pos[:, None, 1] - pos[None, :, 1]))
    candidate_adj = jnp.broadcast_to(base_dist == 1, (1600, 16, 16))
    ci = jnp.arange(1600, dtype=jnp.int32)
    candidate_adj = candidate_adj.at[ci, card, :].set(new_dist == 1)
    candidate_adj = candidate_adj.at[ci, :, card].set(new_dist == 1)
    candidate_adj = candidate_adj.at[ci, card, card].set(True)
    same_global = ((known[:, None] == known[None, :])
                   & (known[:, None] >= 0) & (known[None, :] >= 0))
    same_graph = candidate_adj & same_global[None, :, :]
    reach = jnp.broadcast_to(jnp.eye(16, dtype=jnp.bool_), (1600, 16, 16)) | same_graph
    for _ in range(3):
        reach = reach | ((reach.astype(jnp.int8) @ same_graph.astype(jnp.int8)) > 0)
    connected_pairs = jnp.sum(reach & same_global[None, :, :], axis=(1, 2))

    score = (80 * connected_pairs
             + 64 * row_gain
             + 8 * (old_y_error - new_y_error)
             + 5 * same_distance_gain
             - 3 * new_diff_edges
             - card)
    legal = obs.legal_actions[17:1617] & is_known
    return (17 + jnp.argmax(jnp.where(legal, score, -1000000))).astype(jnp.int32)


def _reveal_action(obs):
    ids = jnp.arange(7, dtype=jnp.int32)
    legal = obs.legal_actions[1617:1624]
    return (1617 + jnp.argmax(jnp.where(legal, 10 - ids, -100000))).astype(jnp.int32)


def _place_action(obs, known):
    k = jnp.arange(700, dtype=jnp.int32)
    hint = k // 100
    rem = k % 100
    x = rem // 10
    y = rem % 10
    pos = obs.positions
    at_cell = (pos[None, :, 0] == x[:, None]) & (pos[None, :, 1] == y[:, None])
    card = jnp.argmax(at_cell, axis=1)
    color = known[card].astype(jnp.int32)
    card_known = color >= 0
    hint_colors = obs.node_features[16:23, 4:8]
    compatible = hint_colors[hint, jnp.maximum(color, 0)] > 0
    correct_row = x == (3 + jnp.maximum(color, 0))
    partner_seen = obs.node_features[card, 11] > 0

    dist = jnp.abs(x[:, None] - pos[None, :, 0]) + jnp.abs(y[:, None] - pos[None, :, 1])
    same = (known[None, :] == color[:, None]) & (known[None, :] >= 0)
    same_neighbors = jnp.sum(same & (dist == 1), axis=1)
    same_global = ((known[:, None] == known[None, :])
                   & (known[:, None] >= 0) & (known[None, :] >= 0))
    graph = (obs.adjacency_matrix[:16, :16] > 0) & same_global
    reach = jnp.eye(16, dtype=jnp.bool_) | graph
    for _ in range(3):
        reach = reach | ((reach.astype(jnp.int8) @ graph.astype(jnp.int8)) > 0)
    component_size = jnp.sum(reach & same_global, axis=1)
    width = jnp.sum(hint_colors[hint], axis=1)
    score = (card_known.astype(jnp.int32) * 1000
             + compatible.astype(jnp.int32) * 4000
             + component_size[card] * 250
             + correct_row.astype(jnp.int32) * 20
             + partner_seen.astype(jnp.int32) * 40
             + same_neighbors * 10
             - width * 2 - hint - card)
    legal = obs.legal_actions[1624:2324]
    return (1624 + jnp.argmax(jnp.where(legal, score, -1000000))).astype(jnp.int32)


def _all_known_connected(obs, known):
    all_known = jnp.all(known >= 0)
    adj = obs.adjacency_matrix[:16, :16] > 0
    reach = jnp.eye(16, dtype=jnp.bool_)
    for _ in range(4):
        reach = reach | ((reach.astype(jnp.int32) @ adj.astype(jnp.int32)) > 0)
    same = known[:, None] == known[None, :]
    return all_known & jnp.all(jnp.where(same, reach, True))


def _certainly_connected(obs, known):
    """Prove success for every deck-consistent completion of <=4 unknowns."""
    unknown = known < 0
    unknown_count = jnp.sum(unknown)
    order = jnp.argsort(jnp.where(unknown, 0, 1), stable=True)
    slots = order[:4]
    code = jnp.arange(256, dtype=jnp.int32)
    digits = jnp.stack([
        code % 4,
        (code // 4) % 4,
        (code // 16) % 4,
        (code // 64) % 4,
    ], axis=1).astype(jnp.int8)
    completions = jnp.broadcast_to(known[None, :], (256, 16))
    for k in range(4):
        previous = completions[:, slots[k]]
        value = jnp.where(k < unknown_count, digits[:, k], previous)
        completions = completions.at[:, slots[k]].set(value)

    counts = jnp.stack([jnp.sum(completions == c, axis=1) for c in range(4)], axis=1)
    valid = (unknown_count <= 4) & jnp.all(counts == 4, axis=1)
    same = completions[:, :, None] == completions[:, None, :]
    graph = (obs.adjacency_matrix[None, :16, :16] > 0) & same
    reach = jnp.broadcast_to(jnp.eye(16, dtype=jnp.bool_), (256, 16, 16)) | graph
    for _ in range(3):
        reach = reach | ((reach.astype(jnp.int8) @ graph.astype(jnp.int8)) > 0)
    solved = jnp.all(jnp.where(same, reach, True), axis=(1, 2))
    return jnp.any(valid) & jnp.all(jnp.where(valid, solved, True))


def agent_act(obs, reset, memory, rng):
    del rng
    known = _remember(obs, reset, memory)
    active = obs.node_features[0, 12] > 0
    substep = obs.node_features[0, 13].astype(jnp.int32)

    observe = _observe_action(obs, known)
    move = _move_action(obs, known)
    reveal = _reveal_action(obs)
    place = _place_action(obs, known)
    can_reveal = jnp.any(obs.legal_actions[1617:1624])
    hint_action = jnp.where(can_reveal, reveal, place)
    proven_done = _all_known_connected(obs, known) | _certainly_connected(obs, known)
    end_now = (substep == 0) & obs.legal_actions[2324] & proven_done
    active_action = jnp.where(substep < 2, observe,
                    jnp.where(substep == 2, move, hint_action))
    active_action = jnp.where(end_now, jnp.asarray(2324, jnp.int32), active_action)
    action = jnp.where(active, active_action, jnp.asarray(0, jnp.int32))
    trace = {
        'known_colors': known,
        'known_count': jnp.sum(known >= 0).astype(jnp.int32),
        'positions': obs.positions,
        'substep': substep,
    }
    return action.astype(jnp.int32), known, trace
