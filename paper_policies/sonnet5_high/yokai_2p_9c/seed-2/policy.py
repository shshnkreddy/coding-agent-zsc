"""Cooperative Yokai policy: reactive, memory-light, convention-free.

Design summary (see ledger for rationale):
- No private memory is needed because the observation already restores this
  seat's own past color observations whenever it is active (perfect memory).
- Color knowledge for unseen cards is completed by elimination whenever two
  of the three colors have all three members already identified.
- Ending the game blind loses in expectation (P(already fully clustered) at
  reset is only ~3.6%), so the policy actively verifies and repairs
  connectivity before ending.
- Reward math (5*U + 2*R + C - I) makes voluntarily placing a hint always at
  least as bad in expectation as leaving it revealed-but-unplaced, so hints
  are only ever revealed, never placed, unless placement is forced because no
  unrevealed hint remains.
- All choices are derived solely from the local observation (no signaling
  convention with a partner is assumed or required).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
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

NEG_INF = -1e9


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _decode(obs):
    card_feat = obs.node_features[0:9].astype(jnp.int32)
    hint_feat = obs.node_features[9:13].astype(jnp.int32)

    color_onehot = card_feat[:, 0:3]
    known_mask = jnp.sum(color_onehot, axis=-1) > 0
    obs_color = jnp.argmax(color_onehot, axis=-1)

    positions = obs.positions.astype(jnp.int32)
    locked = card_feat[:, 8] > 0
    active = card_feat[0, 10] > 0
    substep = card_feat[0, 11]

    hint_colorset = hint_feat[:, 3:6]
    hint_revealed = jnp.sum(hint_colorset, axis=-1) > 0
    hint_placed = hint_feat[:, 8] > 0

    known_color = jnp.where(known_mask, obs_color, -1)
    counts = jnp.stack([jnp.sum(known_color == c) for c in range(3)])
    full_mask3 = counts == 3
    num_full = jnp.sum(full_mask3)
    inferred_color = jnp.argmin(jnp.where(full_mask3, 3, jnp.arange(3)))
    best_known_color = jnp.where(
        known_mask, known_color, jnp.where(num_full == 2, inferred_color, -1)
    )

    return dict(
        positions=positions,
        locked=locked,
        active=active,
        substep=substep,
        hint_colorset=hint_colorset,
        hint_revealed=hint_revealed,
        hint_placed=hint_placed,
        best_known_color=best_known_color,
    )


def _adjacent(p, q):
    return (jnp.abs(p[..., 0] - q[..., 0]) + jnp.abs(p[..., 1] - q[..., 1])) == 1


def _connectivity_state(positions, color, locked):
    """End only once every color is certain (direct observation plus
    elimination) and connected, or once nothing further can be learned or
    fixed. Ending on partial certainty was tried and rejected (see ledger
    A01): the two still-unknown colors are jointly connected by chance only
    ~14% of the time even after one color is confirmed, which is far worse
    in expectation than waiting.
    """
    counts = jnp.stack([jnp.sum(color == c) for c in range(3)])

    pos_i = positions[:, None, :]
    pos_j = positions[None, :, :]
    adj = _adjacent(pos_i, pos_j)

    connected_flags = []
    bad_actionable_flags = []
    for c in range(3):
        member = color == c
        pair = member[:, None] & member[None, :]
        edge_count = jnp.sum(adj & pair) // 2
        connected_flags.append(edge_count >= 2)
        bad_actionable_flags.append(jnp.any(member & (~locked)))
    connected_flags = jnp.stack(connected_flags)
    bad_actionable_flags = jnp.stack(bad_actionable_flags)

    unresolved = color == -1
    can_progress = jnp.any(unresolved & (~locked))
    bad_group = (counts == 3) & (~connected_flags)
    bad_actionable = jnp.any(bad_group & bad_actionable_flags)

    should_end = ~(can_progress | bad_actionable)
    known_count = jnp.sum(color != -1)
    any_bad_group = jnp.any(bad_group)
    num_locked_unknown = jnp.sum(unresolved & locked)
    return should_end, known_count, any_bad_group, num_locked_unknown


def _observe_action(locked, color, legal_actions):
    idx = jnp.arange(9)
    unknown = color == -1
    score = (
        2.0 * (unknown & (~locked)).astype(jnp.float32)
        + 1.0 * (~locked & (~unknown)).astype(jnp.float32)
        - 0.001 * idx.astype(jnp.float32)
    )
    legal_slice = legal_actions[1:10]
    score = jnp.where(legal_slice, score, NEG_INF)
    best = jnp.argmax(score)
    return (1 + best).astype(jnp.int32)


def _move_action(positions, color):
    idx = jnp.arange(9)
    xs = jnp.arange(9)
    ys = jnp.arange(9)
    X, Y = jnp.meshgrid(xs, ys, indexing='ij')  # [9,9], X[x,y]=x, Y[x,y]=y

    counts = jnp.stack([jnp.sum(color == c) for c in range(3)])
    pos_i = positions[:, None, :]
    pos_j = positions[None, :, :]
    adj_all = _adjacent(pos_i, pos_j)
    connected_flags = []
    for c in range(3):
        member = color == c
        pair = member[:, None] & member[None, :]
        edge_count = jnp.sum(adj_all & pair) // 2
        connected_flags.append(edge_count >= 2)
    connected_flags = jnp.stack(connected_flags)
    urgent_color = (counts == 3) & (~connected_flags)  # [3] bool

    per_card_scores = []
    for i in range(9):
        ci = color[i]
        same_color = (color == ci) & (idx != i)
        # neighbor count of card i at its CURRENT position
        old_adj = _adjacent(positions[i], positions) & (idx != i)
        old_count = jnp.sum(old_adj & same_color)

        # neighbor count of card i at every CANDIDATE (x, y) cell
        pj_x = positions[:, 0]
        pj_y = positions[:, 1]
        dist = jnp.abs(X[:, :, None] - pj_x[None, None, :]) + jnp.abs(
            Y[:, :, None] - pj_y[None, None, :]
        )
        adj_new = dist == 1  # [9,9,9] (x,y,j)
        new_count = jnp.sum(
            adj_new & same_color[None, None, :], axis=-1
        )  # [9,9]

        delta = (new_count - old_count).astype(jnp.float32)
        ci_safe = jnp.where(ci == -1, 0, ci)
        urgent_bonus = jnp.where(ci != -1, jnp.where(urgent_color[ci_safe], 1.0, 0.0), 0.0)
        move_dist = (jnp.abs(X - positions[i, 0]) + jnp.abs(Y - positions[i, 1])).astype(
            jnp.float32
        )
        unknown_score = -1.0 - 0.01 * move_dist
        score_i = jnp.where(ci != -1, delta + urgent_bonus, unknown_score)
        per_card_scores.append(score_i)

    scores = jnp.stack(per_card_scores)  # [9,9,9] = (i,x,y)
    return scores.reshape(729)


def _hint_reveal_action(hint_revealed):
    idx = jnp.arange(4)
    score = jnp.where(~hint_revealed, -idx.astype(jnp.float32), NEG_INF)
    best = jnp.argmax(score)
    return (739 + best).astype(jnp.int32)


def _hint_place_action(color, locked, hint_colorset, hint_placed, positions):
    color_safe = jnp.where(color == -1, 0, color)
    color_onehot = jax.nn.one_hot(color_safe, 3)
    in_set = jnp.einsum(
        'hc,nc->hn', hint_colorset.astype(jnp.float32), color_onehot
    )
    known = (color != -1)[None, :]
    score_known = jnp.where(known, jnp.where(in_set > 0, 1.0, -1.0), 0.0)

    pos_i = positions[:, None, :]
    pos_j = positions[None, :, :]
    adj = _adjacent(pos_i, pos_j)
    same_color = color[:, None] == color[None, :]
    same_color_neighbors = jnp.sum(
        adj & same_color & (color[:, None] != -1) & (jnp.arange(9)[:, None] != jnp.arange(9)[None, :]),
        axis=-1,
    )
    stability_bonus = jnp.minimum(same_color_neighbors.astype(jnp.float32), 2.0) * 0.1

    score_known = score_known * 10.0 + stability_bonus[None, :]

    valid = (~hint_placed)[:, None] & (~locked)[None, :]
    score = jnp.where(valid, score_known, NEG_INF)

    h_idx = jnp.arange(4)
    action_ids = (
        743 + h_idx[:, None] * 81 + positions[None, :, 0] * 9 + positions[None, :, 1]
    )

    flat_score = score.reshape(36)
    flat_action = action_ids.reshape(36)
    best = jnp.argmax(flat_score)
    return flat_action[best].astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    d = _decode(obs)
    legal_actions = obs.legal_actions

    should_end, known_count, any_bad_group, num_locked_unknown = _connectivity_state(
        d['positions'], d['best_known_color'], d['locked']
    )
    observe_action = _observe_action(d['locked'], d['best_known_color'], legal_actions)

    action0 = jnp.where(should_end, jnp.asarray(1067, dtype=jnp.int32), observe_action)
    action1 = observe_action

    move_scores = _move_action(d['positions'], d['best_known_color'])
    move_scores = jnp.where(legal_actions[10:739], move_scores, NEG_INF)
    action2 = (10 + jnp.argmax(move_scores)).astype(jnp.int32)

    unrevealed_count = jnp.sum(~d['hint_revealed'])
    reveal_action = _hint_reveal_action(d['hint_revealed'])
    place_action = _hint_place_action(
        d['best_known_color'], d['locked'], d['hint_colorset'], d['hint_placed'], d['positions']
    )
    action3 = jnp.where(unrevealed_count > 0, reveal_action, place_action)

    chosen = jnp.select(
        [d['substep'] == 0, d['substep'] == 1, d['substep'] == 2, d['substep'] == 3],
        [action0, action1, action2, action3],
        default=jnp.argmax(legal_actions).astype(jnp.int32),
    ).astype(jnp.int32)

    fallback = jnp.argmax(legal_actions).astype(jnp.int32)
    safe_action = jnp.where(legal_actions[chosen], chosen, fallback)

    final_action = jnp.where(d['active'], safe_action, jnp.asarray(0, dtype=jnp.int32))

    zero_mem = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero_mem, memory)

    trace = {
        'substep': d['substep'].astype(jnp.int32),
        'active': d['active'].astype(jnp.bool_),
        'should_end': should_end.astype(jnp.bool_),
        'known_count': known_count.astype(jnp.int32),
        'any_bad_group': any_bad_group.astype(jnp.bool_),
        'num_locked_unknown': num_locked_unknown.astype(jnp.int32),
    }

    return final_action, next_memory, trace
