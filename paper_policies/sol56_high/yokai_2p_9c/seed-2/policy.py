"""Locally legible greedy policy for the two-player nine-card Yokai game."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'balanced_pair_greedy_v21', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "own_pair": jnp.full((2,), -1, dtype=jnp.int32),
        "partner_pair": jnp.full((2,), -1, dtype=jnp.int32),
        "seen": jnp.zeros((9,), dtype=jnp.bool_),
        "positions": jnp.zeros((9, 2), dtype=jnp.int32),
        "same": jnp.eye(9, dtype=jnp.bool_),
    }


def _exact_colors(obs, same_relation):
    """Propagate own colors, rational public hints, and the 3/3/3 inventory."""
    features = obs.node_features.astype(jnp.int32)
    own = features[:9, :3]
    own_known = jnp.sum(own, axis=1) == 1

    hint_colors = features[9:13, 3:6]
    linked = obs.adjacency_matrix[:9, 9:13].astype(jnp.int32)
    hinted_domain = (linked @ hint_colors).clip(0, 1).astype(jnp.bool_)
    has_hint = jnp.sum(hinted_domain, axis=1) > 0
    possible = jnp.where(has_hint[:, None], hinted_domain,
                         jnp.ones((9, 3), dtype=jnp.bool_))
    possible = jnp.where(own_known[:, None], own.astype(jnp.bool_), possible)

    # Close high-confidence same-card relations inferred from conspicuous
    # partner moves, then intersect all public/private domains in a component.
    relation = same_relation | jnp.eye(9, dtype=jnp.bool_)
    for _ in range(3):
        relation = relation | ((relation.astype(jnp.int32) @ relation.astype(jnp.int32)) > 0)
    excluded = relation.astype(jnp.int32) @ (~possible).astype(jnp.int32)
    intersection = excluded == 0
    usable_intersection = jnp.sum(intersection, axis=1) > 0
    possible = jnp.where(usable_intersection[:, None], intersection, possible)

    # Fixed-shape constraint propagation. A color already assigned three
    # times is removed elsewhere; if only three cards can still take a color,
    # all three are forced to it.
    for _ in range(9):
        exact = jnp.sum(possible, axis=1) == 1
        assigned = jnp.sum(possible & exact[:, None], axis=0)
        saturated = assigned >= 3
        possible = possible & ~(~exact[:, None] & saturated[None, :])
        available = jnp.sum(possible, axis=0)
        forced_colors = available == 3
        forced_here = possible & forced_colors[None, :]
        uniquely_forced = jnp.sum(forced_here, axis=1) == 1
        possible = jnp.where(uniquely_forced[:, None], forced_here, possible)

    exact = jnp.sum(possible, axis=1) == 1
    colors = jnp.where(exact[:, None], possible, False).astype(jnp.int32)
    return colors, exact, possible.astype(jnp.int32)


def _connected_coloring(obs, colors, exact):
    adj = obs.adjacency_matrix[:9, :9].astype(jnp.bool_)
    same = (colors @ colors.T) > 0
    graph = adj & same & exact[:, None] & exact[None, :]
    # A color has only three cards, so paths of length at most two suffice.
    reach = graph | ((graph.astype(jnp.int32) @ graph.astype(jnp.int32)) > 0)
    per_color = []
    for color in range(3):
        mask = colors[:, color].astype(jnp.bool_) & exact
        reached = jnp.sum(reach & mask[:, None] & mask[None, :])
        per_color.append((jnp.sum(mask) == 3) & (reached == 9))
    return jnp.all(jnp.stack(per_color))


def _observe_action(obs, colors, exact):
    del colors
    legal = obs.legal_actions[1:10]
    other_seen = obs.node_features[:9, 9].astype(jnp.bool_)
    ids = jnp.arange(9, dtype=jnp.int32)
    # Cover cards the partner has not covered before duplicating work.
    priority = ((~exact).astype(jnp.int32) * 100
                + ((~exact) & (~other_seen)).astype(jnp.int32) * 20
                - ids)
    priority = jnp.where(legal, priority, -100000)
    return (1 + jnp.argmax(priority)).astype(jnp.int32)


def _move_action(obs, colors, exact, possible, memory):
    same = ((colors @ colors.T) > 0) & exact[:, None] & exact[None, :]
    different = ((colors @ colors.T) == 0) & exact[:, None] & exact[None, :]
    overlap = (possible @ possible.T) > 0
    restricted = (jnp.sum(possible, axis=1) < 3) & (~exact)
    one_exact = exact[:, None] ^ exact[None, :]
    informative = one_exact & (restricted[:, None] | restricted[None, :])
    soft_supported = informative & overlap
    soft_conflict = informative & (~overlap)
    upper = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)
    old = obs.positions.astype(jnp.int32)

    # Scan candidates instead of materializing [729, 9, 9] arrays.  This is
    # important when the official evaluator batches 1,024 episodes.
    def consider(action, best):
        best_score, best_action = best
        raw = action - 10
        moved = raw // 81
        cell = raw % 81
        dest = jnp.stack((cell // 9, cell % 9))
        candidate_pos = old.at[moved].set(dest)
        delta = jnp.abs(candidate_pos[:, None, :] - candidate_pos[None, :, :])
        adj = jnp.sum(delta, axis=-1) == 1
        same_graph = (adj & same) | jnp.eye(9, dtype=jnp.bool_)
        reach2 = same_graph | ((same_graph.astype(jnp.int32) @ same_graph.astype(jnp.int32)) > 0)
        connected_pairs = jnp.sum(reach2 & same & upper)
        same_edges = jnp.sum(adj & same & upper)
        different_edges = jnp.sum(adj & different & upper)
        supported_edges = jnp.sum(adj & soft_supported & upper)
        conflict_edges = jnp.sum(adj & soft_conflict & upper)
        just_seen = (moved == memory[0]) | (moved == memory[1])
        semantic = (connected_pairs * 100 + same_edges * 12 - different_edges * 2
                    + supported_edges * 4 - conflict_edges * 4
                    + exact[moved].astype(jnp.int32) * 3
                    + just_seen.astype(jnp.int32) * 3)
        score = semantic * 2048 - action
        score = jnp.where(obs.legal_actions[action], score, -2**30)
        better = score > best_score
        return jnp.where(better, score, best_score), jnp.where(better, action, best_action)

    _, best_action = jax.lax.fori_loop(
        10, 739, consider,
        (jnp.asarray(-2**30, dtype=jnp.int32), jnp.asarray(10, dtype=jnp.int32)),
    )
    return best_action


def _hint_action(obs, colors, exact):
    # Revealing every hint first postpones irreversible card locks.
    reveal_legal = obs.legal_actions[739:743]
    reveal = (739 + jnp.argmax(reveal_legal)).astype(jnp.int32)

    actions = jnp.arange(1068, dtype=jnp.int32)
    raw = (actions - 743).clip(0, 323)
    hint = raw // 81
    cell = raw % 81
    target_xy = jnp.stack((cell // 9, cell % 9), axis=1)
    hits = jnp.all(target_xy[:, None, :] == obs.positions[None, :, :], axis=2)
    target = jnp.argmax(hits, axis=1)
    hint_colors = obs.node_features[9:13, 3:6].astype(jnp.int32)
    compatible = jnp.sum(colors[target] * hint_colors[hint], axis=1) > 0
    card_adj = obs.adjacency_matrix[:9, :9].astype(jnp.bool_)
    same = ((colors @ colors.T) > 0) & exact[:, None] & exact[None, :]
    settled_degree = jnp.sum(card_adj & same, axis=1)
    semantic = (compatible.astype(jnp.int32) * exact[target].astype(jnp.int32) * 100
                + settled_degree[target] * 5
                + exact[target].astype(jnp.int32) * 2)
    score = semantic * 2048 - actions
    legal_place = obs.legal_actions & (actions >= 743) & (actions < 1067)
    place = jnp.argmax(jnp.where(legal_place, score, -2**30)).astype(jnp.int32)
    chosen_target = target[place]
    other_seen = obs.node_features[:9, 9].astype(jnp.bool_)
    singleton_placement = jnp.sum(hint_colors[hint[place]]) == 1
    revealed_count = jnp.sum(jnp.sum(hint_colors, axis=1) > 0)
    placed_count = jnp.sum(obs.node_features[9:13, 8] > 0)
    enough_shared_history = (revealed_count + placed_count) >= 3
    safe_early_lock = (jnp.any(legal_place) & compatible[place] & exact[chosen_target]
                       & (settled_degree[chosen_target] >= 2)
                       & (other_seen[chosen_target] | singleton_placement
                          | enough_shared_history))
    return jnp.where(jnp.any(reveal_legal) & (~safe_early_lock), reveal, place)


def agent_act(obs, reset, memory, rng):
    del rng
    initial = init_agent()
    memory = jax.tree_util.tree_map(lambda fresh, old: jnp.where(reset, fresh, old),
                                    initial, memory)
    substep = obs.node_features[0, 11].astype(jnp.int32)
    active = obs.node_features[0, 10].astype(jnp.bool_)
    colors, exact, possible = _exact_colors(obs, memory["same"])

    def choose_active(_):
        def first_observe(_):
            observe = _observe_action(obs, colors, exact)
            clustered = _connected_coloring(obs, colors, exact)
            return jnp.where(clustered & obs.legal_actions[1067], 1067, observe)

        branches = (
            first_observe,
            lambda _: _observe_action(obs, colors, exact),
            lambda _: _move_action(obs, colors, exact, possible, memory["own_pair"]),
            lambda _: _hint_action(obs, colors, exact),
        )
        return jax.lax.switch(substep, branches, operand=None)

    action = jax.lax.cond(
        active, choose_active, lambda _: jnp.asarray(0, dtype=jnp.int32), operand=None
    ).astype(jnp.int32)

    observed_id = action - 1
    own_pair = jnp.where(
        active & (substep == 0),
        jnp.stack((observed_id, jnp.asarray(-1, dtype=jnp.int32))),
        memory["own_pair"],
    )
    own_pair = jnp.where(
        active & (substep == 1),
        jnp.stack((own_pair[0], observed_id)),
        own_pair,
    )

    # While inactive, feature 9 exposes which stable IDs the partner newly
    # inspected. At substep 3, positions reveal the intervening move.
    other_seen = obs.node_features[:9, 9].astype(jnp.bool_)
    newly_seen = other_seen & (~memory["seen"])
    new_id = jnp.argmax(newly_seen).astype(jnp.int32)
    has_new = jnp.sum(newly_seen) == 1
    partner_pair = jnp.where(
        (~active) & (substep == 0),
        jnp.full((2,), -1, dtype=jnp.int32),
        memory["partner_pair"],
    )
    partner_pair = jnp.where(
        (~active) & (substep == 1) & has_new,
        jnp.stack((new_id, jnp.asarray(-1, dtype=jnp.int32))),
        partner_pair,
    )
    partner_pair = jnp.where(
        (~active) & (substep == 2) & has_new,
        jnp.stack((partner_pair[0], new_id)),
        partner_pair,
    )

    changed = jnp.any(obs.positions != memory["positions"], axis=1)
    moved = jnp.argmax(changed).astype(jnp.int32)
    p0, p1 = memory["partner_pair"][0], memory["partner_pair"][1]
    moved_is_pair = (moved == p0) | (moved == p1)
    other = jnp.where(moved == p0, p1, p0)
    moved_safe = moved.clip(0, 8)
    other_safe = other.clip(0, 8)
    old_adjacent = jnp.sum(jnp.abs(memory["positions"][moved_safe]
                                   - memory["positions"][other_safe])) == 1
    new_adjacent = jnp.sum(jnp.abs(obs.positions[moved_safe]
                                   - obs.positions[other_safe])) == 1
    signal = ((~active) & (substep == 3) & (jnp.sum(changed) == 1)
              & moved_is_pair & (p0 >= 0) & (p1 >= 0)
              & (~old_adjacent) & new_adjacent)
    edge = (jax.nn.one_hot(moved_safe, 9, dtype=jnp.bool_)[:, None]
            & jax.nn.one_hot(other_safe, 9, dtype=jnp.bool_)[None, :])
    edge = edge | edge.T
    same = memory["same"] | (edge & signal)
    new_memory = {
        "own_pair": own_pair,
        "partner_pair": partner_pair,
        "seen": other_seen,
        "positions": obs.positions.astype(jnp.int32),
        "same": same,
    }
    return action, new_memory, {}
