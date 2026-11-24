"""Local-memory, convention-light greedy policy for two-player Yōkai."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'legible_greedy_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "colors": jnp.full((9,), -1, dtype=jnp.int8),
        "partner_move_start": jnp.zeros((9, 2), dtype=jnp.int32),
        "partner_seen": jnp.zeros((9,), dtype=jnp.bool_),
    }


def _first_legal(mask):
    return jnp.argmax(mask).astype(jnp.int32)


def _move_action(obs, colors):
    """Choose the legal relocation with the best visible clustering score."""
    k = jnp.arange(729, dtype=jnp.int32)
    moved = k // 81
    rem = k % 81
    destinations = jnp.stack((rem // 9, rem % 9), axis=1)
    boards = jnp.broadcast_to(obs.positions[None, :, :], (729, 9, 2))
    boards = boards.at[k, moved, :].set(destinations)
    delta = jnp.abs(boards[:, :, None, :] - boards[:, None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1

    known = colors >= 0
    pair_known = known[:, None] & known[None, :]
    same = pair_known & (colors[:, None] == colors[None, :])
    different = pair_known & ~same
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    same_edges = jnp.sum(adjacent & same[None, :, :] & upper[None, :, :], axis=(1, 2))
    diff_edges = jnp.sum(adjacent & different[None, :, :] & upper[None, :, :], axis=(1, 2))
    total_edges = jnp.sum(adjacent & upper[None, :, :], axis=(1, 2))
    distances = jnp.sum(delta, axis=-1)
    near_same = jnp.sum((distances == 2) & same[None, :, :] &
                        upper[None, :, :], axis=(1, 2))
    diagonal_same = jnp.sum((delta[..., 0] == 1) & (delta[..., 1] == 1) &
                            same[None, :, :] & upper[None, :, :], axis=(1, 2))
    center_distance = jnp.abs(destinations[:, 0] - 4) + jnp.abs(destinations[:, 1] - 4)
    moving_known = known[moved].astype(jnp.int32)
    score = (same_edges * 100 - diff_edges * 3 + total_edges * 3
             + near_same + diagonal_same * 2
             + moving_known - center_distance)
    legal = obs.legal_actions[10:739]
    score = jnp.where(legal, score, jnp.asarray(-100000, dtype=score.dtype))
    return (10 + jnp.argmax(score)).astype(jnp.int32)


def _hint_action(obs, colors):
    features = obs.node_features
    hint_sets = features[9:13, 3:6] > 0
    xs = obs.positions[:, 0]
    ys = obs.positions[:, 1]
    reveal_mask = obs.legal_actions[739:743]
    reveal_action = (739 + jnp.argmax(reveal_mask)).astype(jnp.int32)

    place_mask = obs.legal_actions[743:1067].reshape(4, 9, 9)
    card_legal = place_mask[:, xs, ys]
    safe_color = jnp.maximum(colors, 0).astype(jnp.int32)
    compatible = hint_sets[:, safe_color].T & (colors[:, None] >= 0)
    compatible = compatible.T & card_legal
    flat_good = compatible.reshape(-1)
    flat_legal = card_legal.reshape(-1)
    card_adj = obs.adjacency_matrix[:9, :9] > 0
    same = (colors[:, None] == colors[None, :]) & (colors[:, None] >= 0)
    same_degree = jnp.sum(card_adj & same, axis=1) - 1
    board_degree = jnp.sum(card_adj, axis=1) - 1
    card_rank = (same_degree * 100 + board_degree * 10
                 - jnp.arange(9, dtype=jnp.int32))
    good_rank = jnp.where(compatible, 10000 + card_rank[None, :], -100000)
    legal_rank = jnp.where(card_legal, card_rank[None, :], -100000)
    choice = jnp.where(jnp.any(flat_good), jnp.argmax(good_rank.reshape(-1)),
                       jnp.argmax(legal_rank.reshape(-1)))
    hint_id = choice // 9
    card_id = choice % 9
    place_action = (743 + hint_id * 81 + xs[card_id] * 9 + ys[card_id]).astype(jnp.int32)

    can_reveal = jnp.any(reveal_mask)
    fallback = _first_legal(obs.legal_actions)
    return jnp.where(can_reveal, reveal_action,
                     jnp.where(jnp.any(flat_legal), place_action, fallback))


def _known_solved(positions, colors):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adj = jnp.sum(delta, axis=-1) == 1
    counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    edges = jnp.stack([
        jnp.sum(adj & ((colors == c)[:, None] & (colors == c)[None, :])) // 2
        for c in range(3)
    ])
    return jnp.all((counts == 3) & (edges >= 2))


def _three_unknown_completion_counts(positions, colors):
    """Count deck-consistent and solved completions of three unknown cards."""
    unknown = colors < 0
    unknown_ids = jnp.nonzero(unknown, size=3, fill_value=0)[0]
    code = jnp.arange(27, dtype=jnp.int8)
    assignments = jnp.stack((code % 3, (code // 3) % 3, code // 9), axis=1)
    completions = jnp.broadcast_to(colors[None, :], (27, 9))
    completions = completions.at[:, unknown_ids[0]].set(assignments[:, 0])
    completions = completions.at[:, unknown_ids[1]].set(assignments[:, 1])
    completions = completions.at[:, unknown_ids[2]].set(assignments[:, 2])

    counts = jnp.stack([jnp.sum(completions == c, axis=1) for c in range(3)], axis=1)
    valid = jnp.all(counts == 3, axis=1)
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adj = jnp.sum(delta, axis=-1) == 1
    edges = jnp.stack([
        jnp.sum(adj[None, :, :] &
                ((completions == c)[:, :, None] & (completions == c)[:, None, :]),
                axis=(1, 2)) // 2
        for c in range(3)
    ], axis=1)
    completion_solved = jnp.all(edges >= 2, axis=1)
    return jnp.sum(valid), jnp.sum(valid & completion_solved)


def agent_act(obs, reset, memory, rng):
    del rng
    colors0 = jnp.where(reset, jnp.full((9,), -1, dtype=jnp.int8), memory["colors"])
    seen0 = jnp.where(reset, jnp.zeros((9,), dtype=jnp.bool_), memory["partner_seen"])
    move_start0 = jnp.where(reset, jnp.zeros((9, 2), dtype=jnp.int32),
                            memory["partner_move_start"])
    visible = obs.node_features[:9, :3]
    has_color = jnp.sum(visible, axis=1) > 0
    observed_colors = jnp.argmax(visible, axis=1).astype(jnp.int8)
    colors = jnp.where(has_color, observed_colors, colors0)
    # Exact multiset deduction: if only one color remains below its fixed deck
    # multiplicity, every still-unknown card must have that color.  This
    # includes the usual eight-observation inference but can trigger earlier.
    color_counts = jnp.stack([jnp.sum(colors == c) for c in range(3)])
    remaining_color = jnp.argmin(color_counts).astype(jnp.int8)
    sole_remaining = jnp.sum(color_counts < 3) == 1
    colors = jnp.where(sole_remaining & (colors < 0), remaining_color, colors)
    partner_seen = seen0 | (obs.node_features[:9, 9] > 0)
    active = obs.node_features[0, 10] > 0
    substep = obs.node_features[0, 11].astype(jnp.int32)
    capture_partner_move = (~active) & (substep == 2)
    partner_move_start = jnp.where(capture_partner_move, obs.positions, move_start0)
    next_memory = {
        "colors": colors,
        "partner_move_start": partner_move_start,
        "partner_seen": partner_seen,
    }
    unlocked = obs.node_features[:9, 8] == 0
    unknown = colors < 0
    observe_choices = obs.legal_actions[1:10] & unlocked & unknown
    observe_action = jnp.where(jnp.any(observe_choices),
                               1 + jnp.argmax(observe_choices),
                               1 + jnp.argmax(obs.legal_actions[1:10])).astype(jnp.int32)

    solved = _known_solved(obs.positions, colors)
    valid_completions, solved_completions = _three_unknown_completion_counts(
        obs.positions, colors)
    _, solved_before = _three_unknown_completion_counts(partner_move_start, colors)
    placed_hints = jnp.sum(obs.node_features[9:13, 8] > 0)
    moved_cards = jnp.any(obs.positions != partner_move_start, axis=1)
    moved_unknown = jnp.any(moved_cards & (colors < 0))
    card_adj = obs.adjacency_matrix[:9, :9] > 0
    known_same = ((colors[:, None] == colors[None, :]) &
                  (colors[:, None] >= 0))
    known_same_degree = jnp.sum(card_adj & known_same, axis=1) - 1
    moved_known_pair = jnp.any(moved_cards & (colors >= 0) &
                               (known_same_degree > 0))
    before_delta = jnp.abs(partner_move_start[:, None, :] -
                           partner_move_start[None, :, :])
    before_degree = jnp.sum(jnp.sum(before_delta, axis=-1) == 1, axis=1)
    after_degree = jnp.sum(card_adj, axis=1) - 1
    moved_before_degree = jnp.max(jnp.where(moved_cards, before_degree, -1))
    moved_after_degree = jnp.max(jnp.where(moved_cards, after_degree, -1))
    topology_stable_pair = (moved_known_pair &
        ((moved_after_degree == 1) |
         (moved_after_degree == moved_before_degree)))
    three_completion_evidence = ((valid_completions == 3) &
        ((solved_completions == 2) |
         ((solved_completions == 1) &
          (((topology_stable_pair |
             (moved_unknown & (moved_after_degree == 1))) &
            (solved_before == 1)) |
           (moved_unknown & (solved_before != 1)) |
           (moved_known_pair & (solved_before != 1))))))
    six_completion_evidence = ((valid_completions == 6) &
        (((solved_before == 1) & (solved_completions == 2)) |
         ((solved_before == 0) & (solved_completions == 1) &
          moved_known_pair)))
    strong_board_evidence = three_completion_evidence | six_completion_evidence
    trust_partner = ((jnp.sum(colors < 0) == 3) &
                     strong_board_evidence & (placed_hints == 3) &
                     (jnp.sum(partner_seen) >= 8))
    step0_action = jnp.where((solved | trust_partner) & obs.legal_actions[1067],
                             jnp.asarray(1067, dtype=jnp.int32), observe_action)
    actions = jnp.stack((step0_action, observe_action,
                         _move_action(obs, colors), _hint_action(obs, colors)))
    chosen = actions[jnp.clip(substep, 0, 3)]
    action = jnp.where(active, chosen, jnp.asarray(0, dtype=jnp.int32))
    trace = {
        "colors": colors,
        "known": jnp.sum(colors >= 0).astype(jnp.int32),
        "positions": obs.positions,
        "solved": solved,
        "solved_before": solved_before,
        "trust_partner": trust_partner,
    }
    return action, next_memory, trace
