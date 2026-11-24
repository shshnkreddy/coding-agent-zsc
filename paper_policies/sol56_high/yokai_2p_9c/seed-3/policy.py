"""Locally evidence-based greedy policy for the two-player Yokai game."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_greedy', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

def init_agent():
    # [seat inferred at reset, completed active turns]
    return jnp.zeros((2,), dtype=jnp.int32)


def _color_knowledge(features):
    private_one_hot = features[:9, :3].astype(jnp.int32)
    hint_sets = features[9:13, 3:6].astype(jnp.int32)
    placed = features[9:13, 8] > 0
    same_position = jnp.all(
        features[:9, None, 6:8] == features[None, 9:13, 6:8], axis=2
    )
    attached = same_position & placed[None, :]
    attached_sets = (attached.astype(jnp.int32) @ hint_sets > 0).astype(jnp.int32)
    exact = private_one_hot
    has_private = jnp.sum(private_one_hot, axis=1) > 0
    has_hint = jnp.sum(attached_sets, axis=1) > 0
    possible = jnp.where(
        has_private[:, None],
        private_one_hot,
        jnp.where(has_hint[:, None], attached_sets,
                  jnp.ones((9, 3), dtype=jnp.int32)),
    )
    # Propagate the literal hint sets and the exact three-cards-per-color rule.
    for _ in range(9):
        unknown = jnp.sum(exact, axis=1) == 0
        counts_now = jnp.sum(exact, axis=0)
        remaining = 3 - counts_now
        possible = jnp.where(
            unknown[:, None],
            possible * (remaining > 0)[None, :],
            exact,
        )
        singleton = unknown & (jnp.sum(possible, axis=1) == 1)
        exact = jnp.maximum(exact, jnp.where(singleton[:, None], possible, 0))

        unknown = jnp.sum(exact, axis=1) == 0
        counts_now = jnp.sum(exact, axis=0)
        remaining = 3 - counts_now
        support = jnp.sum(possible * unknown[:, None], axis=0)
        forced = (possible > 0) & unknown[:, None] & \
                 (support == remaining)[None, :] & (remaining > 0)[None, :]
        uniquely_forced = jnp.sum(forced, axis=1) == 1
        exact = jnp.maximum(
            exact,
            jnp.where(uniquely_forced[:, None], forced.astype(jnp.int32), 0),
        )
        possible = jnp.where((jnp.sum(exact, axis=1) > 0)[:, None], exact,
                             possible)
    one_hot = exact
    known = jnp.sum(one_hot, axis=1) > 0
    color = jnp.argmax(one_hot, axis=1).astype(jnp.int32)
    counts = jnp.sum(one_hot, axis=0)
    can_infer = jnp.sum(known) == 8
    missing_color = jnp.argmin(counts).astype(jnp.int32)
    effective = jnp.where(known, color, jnp.where(can_infer, missing_color, -1))
    effective_known = known | (can_infer & (~known))
    return effective, effective_known


def _color_geometry(positions, colors, known):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    distance = jnp.sum(delta, axis=-1)
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), k=1)
    same = (colors[:, None] == colors[None, :]) & known[:, None] & known[None, :] & upper
    edges = jnp.sum(same & (distance == 1))
    same_distance = jnp.sum(jnp.where(same, jnp.minimum(distance, 12), 0))
    color_edges = jnp.stack([
        jnp.sum(same & (distance == 1) & (colors[:, None] == c)) for c in range(3)
    ])
    color_counts = jnp.stack([jnp.sum(known & (colors == c)) for c in range(3)])
    solved_colors = (color_counts == 3) & (color_edges >= 2)
    return edges, same_distance, solved_colors, distance


def _observe_action(obs, colors, known):
    del colors
    legal = obs.legal_actions[1:10]
    partner_seen = obs.node_features[:9, 9] > 0
    # Complementary coverage lets each seat contribute distinct color evidence.
    score = (~known).astype(jnp.int32) * 100 + (~partner_seen).astype(jnp.int32) * 20
    score = score - jnp.arange(9, dtype=jnp.int32)
    score = jnp.where(legal, score, -100000)
    return (1 + jnp.argmax(score)).astype(jnp.int32)


def _move_action(obs, colors, known):
    positions = obs.positions.astype(jnp.int32)
    move_index = jnp.arange(729, dtype=jnp.int32)
    moved_card = move_index // 81
    cell = move_index % 81
    destination = jnp.stack([cell // 9, cell % 9], axis=1)
    card_rows = jnp.arange(9, dtype=jnp.int32)[None, :, None]
    candidates = jnp.where(
        card_rows == moved_card[:, None, None],
        destination[:, None, :],
        positions[None, :, :],
    )

    delta = jnp.abs(candidates[:, :, None, :] - candidates[:, None, :, :])
    distance = jnp.sum(delta, axis=-1)
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), k=1)
    same = ((colors[:, None] == colors[None, :]) &
            known[:, None] & known[None, :] & upper)
    same_edges = jnp.sum(same[None, :, :] & (distance == 1), axis=(1, 2))
    locked = obs.node_features[:9, 8] > 0
    anchored_same = same & (locked[:, None] | locked[None, :])
    anchored_edges = jnp.sum(
        anchored_same[None, :, :] & (distance == 1), axis=(1, 2)
    )
    same_distance = jnp.sum(
        jnp.where(same[None, :, :], jnp.minimum(distance, 12), 0), axis=(1, 2)
    )

    solved_bonus = jnp.zeros((729,), dtype=jnp.int32)
    for c in range(3):
        mask = same & (colors[:, None] == c)
        edge_count = jnp.sum(mask[None, :, :] & (distance == 1), axis=(1, 2))
        count = jnp.sum(known & (colors == c))
        solved_bonus = solved_bonus + ((count == 3) & (edge_count >= 2)) * 500

    all_upper = upper[None, :, :]
    compactness = jnp.sum(jnp.where(all_upper, distance, 0), axis=(1, 2))
    old_pos = positions[moved_card]
    travel = jnp.sum(jnp.abs(destination - old_pos), axis=1)
    moving_known = known[moved_card].astype(jnp.int32)
    score = (solved_bonus + same_edges * 100 + anchored_edges * 125
             - same_distance * 4
             - compactness * 3
             - travel * 2 + moving_known * 6)
    legal = obs.legal_actions[10:739]
    score = jnp.where(legal, score, -100000000)
    return (10 + jnp.argmax(score)).astype(jnp.int32), jnp.max(score)


def _hint_action(obs, colors, known, distance):
    features = obs.node_features
    hint_sets = features[9:13, 3:6].astype(jnp.int32)
    partner_unseen = features[:9, 9] <= 0

    action_index = jnp.arange(328, dtype=jnp.int32)
    full_action = action_index + 739
    legal = obs.legal_actions[739:1067]
    is_reveal = action_index < 4

    place_index = jnp.maximum(action_index - 4, 0)
    hint = place_index // 81
    cell = place_index % 81
    xy = jnp.stack([cell // 9, cell % 9], axis=1)
    at_cell = jnp.all(obs.positions[None, :, :] == xy[:, None, :], axis=2)
    target = jnp.argmax(at_cell, axis=1)
    target_color = colors[target]
    target_known = known[target]
    safe_color = jnp.maximum(target_color, 0)
    matches = target_known & (hint_sets[hint, safe_color] > 0)

    same_neighbor = jnp.sum(
        (distance[target] == 1) & known[None, :] &
        (colors[None, :] == target_color[:, None]), axis=1
    )
    neighbor_bonus = jnp.where(same_neighbor >= 2, 250, same_neighbor * 125)
    public_degree = jnp.sum(distance[target] == 1, axis=1)
    set_size = jnp.sum(hint_sets[hint], axis=1)

    reveal_score = 750 - action_index
    place_score = (matches.astype(jnp.int32) * 600 +
                   target_known.astype(jnp.int32) * 100 +
                   neighbor_bonus +
                   - public_degree * 10 +
                   partner_unseen[target].astype(jnp.int32) * 150 +
                   (set_size == 1).astype(jnp.int32) * 100 - target)
    score = jnp.where(is_reveal, reveal_score, place_score)
    score = jnp.where(legal, score, -1000000)
    return full_action[jnp.argmax(score)].astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.node_features
    active = features[0, 10] > 0
    substep = features[0, 11].astype(jnp.int32)

    reset_memory = jnp.asarray([(~active).astype(jnp.int32), 0], dtype=jnp.int32)
    memory = jnp.where(reset, reset_memory, memory)

    colors, known = _color_knowledge(features)
    _, _, solved_colors, distance = _color_geometry(obs.positions, colors, known)
    fully_known = jnp.sum(known) == 9
    solved = fully_known & jnp.all(solved_colors)

    observe_action = _observe_action(obs, colors, known)
    move_action, move_score = _move_action(obs, colors, known)
    hint_action = _hint_action(obs, colors, known, distance)
    active_action = jnp.where(
        substep == 0,
        jnp.where(solved & obs.legal_actions[1067], 1067, observe_action),
        jnp.where(substep == 1, observe_action,
                  jnp.where(substep == 2, move_action, hint_action)),
    ).astype(jnp.int32)
    action = jnp.where(active, active_action, 0).astype(jnp.int32)

    completed = active & (substep == 3)
    next_memory = memory.at[1].add(completed.astype(jnp.int32))
    trace = {
        "known": jnp.sum(known).astype(jnp.int32),
        "solved": solved,
        "move_score": move_score.astype(jnp.int32),
        "turns": next_memory[1],
    }
    return action, next_memory, trace
