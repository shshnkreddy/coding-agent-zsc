"""Evidence-based Yokai policy for the 9-card task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'near_hint_clusterer', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "colors": jnp.zeros((9, 3), dtype=jnp.int8),
        "seen": jnp.zeros((9,), dtype=jnp.bool_),
        "seat": jnp.asarray(0, dtype=jnp.int32),
    }


def _first_legal(mask, offset):
    return (offset + jnp.argmax(mask).astype(jnp.int32)).astype(jnp.int32)


def _center_cost(x, y):
    dx = jnp.maximum(jnp.asarray(0, dtype=jnp.int32), jnp.maximum(3 - x, x - 5))
    dy = jnp.maximum(jnp.asarray(0, dtype=jnp.int32), jnp.maximum(3 - y, y - 5))
    return dx + dy


def _estimate_colors(colors, seen):
    counts = jnp.sum(colors.astype(jnp.int32), axis=0)
    remaining = 3 - counts
    unknown = jnp.logical_not(seen)
    unknown_count = jnp.sum(unknown.astype(jnp.int32))
    single_unknown = unknown_count == 1
    inferred = (remaining > 0).astype(jnp.int8)
    est = jnp.where(seen[:, None], colors, jnp.where(single_unknown, inferred[None, :], jnp.zeros((9, 3), dtype=jnp.int8)))
    effective = jnp.logical_or(seen, jnp.logical_and(single_unknown, unknown))
    est_color = jnp.argmax(est, axis=1).astype(jnp.int32)
    return est, effective, est_color, unknown_count


def _target_data(est_color, effective):
    ids = jnp.arange(9, dtype=jnp.int32)
    same_color = est_color[:, None] == est_color[None, :]
    lower_id = ids[None, :] < ids[:, None]
    rank = jnp.sum(jnp.logical_and(jnp.logical_and(same_color, lower_id), effective[None, :]).astype(jnp.int32), axis=1)
    rank = jnp.minimum(rank, 2)
    target_x = 3 + est_color
    target_y = 3 + rank
    return target_x.astype(jnp.int32), target_y.astype(jnp.int32)


def _card_costs(positions, est_color, effective):
    target_x, target_y = _target_data(est_color, effective)
    x = positions[:, 0].astype(jnp.int32)
    y = positions[:, 1].astype(jnp.int32)
    exact_cost = jnp.abs(x - target_x) + jnp.abs(y - target_y)
    public_cost = _center_cost(x, y)
    return jnp.where(effective, 20 * exact_cost, public_cost).astype(jnp.int32)


def _move_action(obs, positions, est_color, effective):
    legal = obs.legal_actions[10:739]

    a = jnp.arange(729, dtype=jnp.int32)
    card = a // 81
    rem = a - card * 81
    x = rem // 9
    y = rem - x * 9

    ids = jnp.arange(9, dtype=jnp.int32)
    moved = ids[None, :] == card[:, None]
    after_x = jnp.where(moved, x[:, None], positions[:, 0][None, :].astype(jnp.int32))
    after_y = jnp.where(moved, y[:, None], positions[:, 1][None, :].astype(jnp.int32))

    dx = jnp.abs(after_x[:, :, None] - after_x[:, None, :])
    dy = jnp.abs(after_y[:, :, None] - after_y[:, None, :])
    upper = ids[:, None] < ids[None, :]
    same = est_color[:, None] == est_color[None, :]
    pair_mask = jnp.logical_and(jnp.logical_and(same, upper), jnp.logical_and(effective[:, None], effective[None, :]))
    pair_cost = jnp.sum(jnp.where(pair_mask[None, :, :], dx + dy, 0), axis=(1, 2))

    row_cost = jnp.sum(jnp.where(effective[None, :], jnp.abs(after_x - (3 + est_color)[None, :]), 0), axis=1)
    public_cost = jnp.sum(_center_cost(after_x, after_y), axis=1)
    after = 20 * pair_cost + 5 * row_cost + public_cost
    score = jnp.where(legal, -after * 1000 - a, -1000000000)
    return (10 + jnp.argmax(score).astype(jnp.int32)).astype(jnp.int32)


def _observe_action(obs, seen, seat):
    legal = obs.legal_actions[1:10]
    ids = jnp.arange(9, dtype=jnp.int32)
    unknown_legal = jnp.logical_and(legal, jnp.logical_not(seen))
    rank_p0 = 8 - ids
    rank_p1 = ids
    rank = jnp.where(seat == 0, rank_p0, rank_p1)
    preferred_score = jnp.where(unknown_legal, rank, -1000)
    fallback_score = jnp.where(legal, rank, -1000)
    has_unknown = jnp.any(unknown_legal)
    choice = jnp.where(has_unknown, jnp.argmax(preferred_score), jnp.argmax(fallback_score))
    return (1 + choice.astype(jnp.int32)).astype(jnp.int32)


def _connectivity_end(obs, est_color, effective, unknown_count):
    adj = obs.adjacency_matrix[:9, :9].astype(jnp.bool_)
    ids = jnp.arange(9, dtype=jnp.int32)

    connected_all = jnp.asarray(True)
    counts_ok = jnp.asarray(True)
    for color in range(3):
        mask = jnp.logical_and(effective, est_color == color)
        count = jnp.sum(mask.astype(jnp.int32))
        start = jnp.argmax(mask).astype(jnp.int32)
        reach = jnp.logical_and(ids == start, mask)
        for _ in range(8):
            reach = jnp.logical_and(jnp.any(jnp.logical_and(adj, reach[None, :]), axis=1), mask)
        connected_all = jnp.logical_and(connected_all, jnp.sum(reach.astype(jnp.int32)) == count)
        counts_ok = jnp.logical_and(counts_ok, count == 3)
    return jnp.logical_and(unknown_count <= 1, jnp.logical_and(counts_ok, connected_all))


def _public_canonical_end(obs, positions, est_color, effective, seen):
    known_count = jnp.sum(seen.astype(jnp.int32))
    no_reveals_left = jnp.logical_not(jnp.any(obs.legal_actions[739:743]))
    adj = obs.adjacency_matrix[:9, :9].astype(jnp.bool_)
    ids = jnp.arange(9, dtype=jnp.int32)

    connected_evidence = jnp.asarray(True)
    enough_per_color = jnp.asarray(True)
    for color in range(3):
        mask = jnp.logical_and(effective, est_color == color)
        count = jnp.sum(mask.astype(jnp.int32))
        start = jnp.argmax(mask).astype(jnp.int32)
        reach = jnp.logical_and(ids == start, mask)
        for _ in range(8):
            reach = jnp.logical_and(jnp.any(jnp.logical_and(adj, reach[None, :]), axis=1), mask)
        connected_evidence = jnp.logical_and(connected_evidence, jnp.sum(reach.astype(jnp.int32)) == count)
        enough_per_color = jnp.logical_and(enough_per_color, count >= 1)
    return jnp.logical_and(
        no_reveals_left,
        jnp.logical_and(known_count >= 6, jnp.logical_and(connected_evidence, enough_per_color)),
    )


def _place_or_reveal_action(obs, nf, positions, est, effective, est_color):
    reveal_legal = obs.legal_actions[739:743]
    reveal_action = _first_legal(reveal_legal, 739)

    legal = obs.legal_actions[743:1067]
    hset = nf[9:13, 3:6].astype(jnp.int32)
    target_x, target_y = _target_data(est_color, effective)

    a = jnp.arange(324, dtype=jnp.int32)
    hint = a // 81
    rem = a - hint * 81
    x = rem // 9
    y = rem - x * 9
    at_xy = jnp.logical_and(positions[:, 0][None, :] == x[:, None], positions[:, 1][None, :] == y[:, None])
    card = jnp.argmax(at_xy.astype(jnp.int32), axis=1)
    card_known = effective[card]
    correct = jnp.sum(hset[hint] * est[card].astype(jnp.int32), axis=1) > 0
    px = positions[:, 0].astype(jnp.int32)
    py = positions[:, 1].astype(jnp.int32)
    same_color = jnp.logical_and(effective[None, :], est_color[None, :] == est_color[card][:, None])
    distance = jnp.abs(px[None, :] - px[card][:, None]) + jnp.abs(py[None, :] - py[card][:, None])
    adjacent = distance == 1
    near = jnp.logical_and(distance > 0, distance <= 2)
    clustered = jnp.any(jnp.logical_and(same_color, adjacent), axis=1)
    near_cluster = jnp.any(jnp.logical_and(same_color, near), axis=1)
    final_slot = jnp.logical_and(positions[card, 0] == target_x[card], positions[card, 1] == target_y[card])
    score = (
        correct.astype(jnp.int32) * 1000
        + clustered.astype(jnp.int32) * 500
        + near_cluster.astype(jnp.int32) * 300
        + final_slot.astype(jnp.int32) * 50
        + card_known.astype(jnp.int32) * 10
        - a
    )
    legal_score = jnp.where(legal, score, -1000000000)
    best_score = jnp.max(legal_score)
    place_action = (743 + jnp.argmax(legal_score).astype(jnp.int32)).astype(jnp.int32)
    good_place = best_score >= 1300
    return jnp.where(jnp.logical_and(jnp.any(reveal_legal), jnp.logical_not(good_place)), reveal_action, place_action).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    nf = obs.node_features
    positions = obs.positions.astype(jnp.int32)
    active = nf[0, 10] == 1
    substep = nf[0, 11].astype(jnp.int32)

    old_colors = jnp.where(reset, jnp.zeros((9, 3), dtype=jnp.int8), memory["colors"])
    old_seen = jnp.where(reset, jnp.zeros((9,), dtype=jnp.bool_), memory["seen"])
    seat = jnp.where(reset, jnp.where(active, 0, 1), memory["seat"]).astype(jnp.int32)

    visible = nf[:9, :3].astype(jnp.int8)
    visible_now = jnp.sum(visible.astype(jnp.int32), axis=1) > 0
    colors = jnp.where(visible_now[:, None], visible, old_colors)
    seen = jnp.logical_or(old_seen, visible_now)

    est, effective, est_color, unknown_count = _estimate_colors(colors, seen)
    certain_end = _connectivity_end(obs, est_color, effective, unknown_count)
    public_end = _public_canonical_end(obs, positions, est_color, effective, seen)
    should_end = jnp.logical_or(certain_end, public_end)

    observe_action = _observe_action(obs, seen, seat)
    end_action = jnp.asarray(1067, dtype=jnp.int32)
    act0 = jnp.where(jnp.logical_and(obs.legal_actions[1067], should_end), end_action, observe_action)
    act1 = observe_action
    act2 = _move_action(obs, positions, est_color, effective)
    act3 = _place_or_reveal_action(obs, nf, positions, est, effective, est_color)
    action = jnp.where(substep == 0, act0, jnp.where(substep == 1, act1, jnp.where(substep == 2, act2, act3)))
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32)).astype(jnp.int32)

    next_memory = {"colors": colors, "seen": seen, "seat": seat}
    trace = {
        "known_count": jnp.sum(seen.astype(jnp.int32)),
        "unknown_count": unknown_count,
        "should_end": should_end,
        "seat": seat,
    }
    return action, next_memory, trace
