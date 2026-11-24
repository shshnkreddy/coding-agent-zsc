"""Partner-robust heuristic policy for the 9-card Yokai task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'memory_cluster_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

PAIR_A_VALUES = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7)
PAIR_B_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5,
                 6, 7, 8, 4, 5, 6, 7, 8, 5, 6, 7, 8, 6, 7, 8, 7, 8, 8)


def _card_ids():
    return jnp.arange(9, dtype=jnp.int32)


def _move_tables():
    offsets = jnp.arange(729, dtype=jnp.int32)
    return offsets // 81, (offsets % 81) // 9, offsets % 9


def _pair_tables():
    return jnp.asarray(PAIR_A_VALUES, dtype=jnp.int32), jnp.asarray(PAIR_B_VALUES, dtype=jnp.int32)


def init_agent():
    return {
        "colors": jnp.full((9,), -1, dtype=jnp.int32),
        "other_seen": jnp.zeros((9,), dtype=jnp.bool_),
        "seat": jnp.asarray(-1, dtype=jnp.int32),
        "calls": jnp.asarray(0, dtype=jnp.int32),
    }


def _reset_memory(active):
    return {
        "colors": jnp.full((9,), -1, dtype=jnp.int32),
        "other_seen": jnp.zeros((9,), dtype=jnp.bool_),
        "seat": jnp.where(active, jnp.asarray(0, jnp.int32), jnp.asarray(1, jnp.int32)),
        "calls": jnp.asarray(0, dtype=jnp.int32),
    }


def _update_memory(obs, reset, memory):
    active = obs.node_features[0, 10] == 1
    base = jax.tree_util.tree_map(lambda new, old: jnp.where(reset, new, old),
                                  _reset_memory(active), memory)
    visible = obs.node_features[:9, 0:3].astype(jnp.int32)
    saw_color = jnp.sum(visible, axis=1) > 0
    seen_color = jnp.argmax(visible, axis=1).astype(jnp.int32)
    colors = jnp.where(saw_color, seen_color, base["colors"])
    other_seen = base["other_seen"] | (obs.node_features[:9, 9] == 1)
    return {
        "colors": colors,
        "other_seen": other_seen,
        "seat": base["seat"],
        "calls": base["calls"] + jnp.asarray(1, dtype=jnp.int32),
    }


def _effective_colors(colors):
    known = colors >= 0
    counts = jnp.asarray([jnp.sum((colors == c).astype(jnp.int32)) for c in range(3)])
    unknown_count = jnp.sum((~known).astype(jnp.int32))
    missing = jnp.argmax(3 - counts).astype(jnp.int32)
    inferred = (unknown_count == 1) & (~known)
    eff_colors = jnp.where(inferred, missing, colors)
    eff_known = known | inferred
    return eff_colors, eff_known


def _color_edge_counts(positions, eff_colors, eff_known):
    pair_a, pair_b = _pair_tables()
    pa = positions[..., pair_a, :]
    pb = positions[..., pair_b, :]
    adjacent = jnp.sum(jnp.abs(pa - pb), axis=-1) == 1
    same_known = (
        eff_known[pair_a]
        & eff_known[pair_b]
        & (eff_colors[pair_a] == eff_colors[pair_b])
    )
    same_edges = adjacent & same_known
    totals = []
    connected = []
    for color in range(3):
        members = eff_known & (eff_colors == color)
        member_count = jnp.sum(members.astype(jnp.int32))
        color_pair = members[pair_a] & members[pair_b]
        edge_count = jnp.sum((adjacent & color_pair).astype(jnp.int32), axis=-1)
        totals.append(edge_count)
        connected.append((member_count == 3) & (edge_count >= 2))
    return jnp.sum(same_edges.astype(jnp.int32), axis=-1), jnp.stack(totals), jnp.stack(connected)


def _board_solved(positions, eff_colors, eff_known):
    _, _, connected = _color_edge_counts(positions, eff_colors, eff_known)
    return jnp.all(connected)


def _partial_confident_solved(positions, colors):
    known = colors >= 0
    _, color_edges, _ = _color_edge_counts(positions, colors, known)
    checks = []
    for color in range(3):
        count = jnp.sum(((colors == color) & known).astype(jnp.int32))
        edges = color_edges[color]
        checks.append((count >= 2) & (edges >= (count - 1)))
    return jnp.all(jnp.stack(checks))


def _first_by_score(mask, score):
    safe_score = jnp.where(mask, score, jnp.full_like(score, -1000000))
    return jnp.argmax(safe_score).astype(jnp.int32)


def _observe_action(obs, mem):
    card_ids = _card_ids()
    legal = obs.legal_actions[1:10]
    unlocked = obs.node_features[:9, 8] == 0
    unknown = mem["colors"] < 0
    prefer_desc = mem["seat"] == 1
    rank = jnp.where(prefer_desc, card_ids, 8 - card_ids)
    unknown_mask = legal & unlocked & unknown
    any_unknown = jnp.any(unknown_mask)
    score = jnp.where(any_unknown, rank + 20 * unknown.astype(jnp.int32), rank)
    idx = _first_by_score(legal & unlocked, score)
    return (1 + idx).astype(jnp.int32)


def _move_action(obs, mem):
    positions = obs.positions.astype(jnp.int32)
    eff_colors, eff_known = _effective_colors(mem["colors"])
    legal = obs.legal_actions[10:739]
    card_ids = _card_ids()
    move_cards, move_x, move_y = _move_tables()

    dest = jnp.stack([move_x, move_y], axis=1)
    is_moved = card_ids[None, :, None] == move_cards[:, None, None]
    cand_pos = jnp.where(is_moved, dest[:, None, :], positions[None, :, :])

    same_edges, color_edges, connected = _color_edge_counts(cand_pos, eff_colors, eff_known)
    solved = jnp.all(connected, axis=0)
    moved_known = eff_known[move_cards]
    moved_color = jnp.maximum(eff_colors[move_cards], 0)

    target_y = 3 + moved_color
    row_fit = -jnp.abs(move_y - target_y)
    center_fit = -jnp.abs(move_x - 4)
    color_progress = jnp.sum(color_edges, axis=0)
    desired_y = 3 + jnp.maximum(eff_colors, 0)
    known_weight = eff_known.astype(jnp.int32)
    board_row_fit = -jnp.sum(
        jnp.abs(cand_pos[:, :, 1] - desired_y[None, :]) * known_weight[None, :],
        axis=1,
    )
    board_center_fit = -jnp.sum(
        jnp.abs(cand_pos[:, :, 0] - 4) * known_weight[None, :],
        axis=1,
    )

    score = (
        solved.astype(jnp.int32) * 10000
        + same_edges.astype(jnp.int32) * 600
        + color_progress.astype(jnp.int32) * 80
        + board_row_fit.astype(jnp.int32) * 75
        + board_center_fit.astype(jnp.int32) * 6
        + moved_known.astype(jnp.int32) * 40
        + row_fit.astype(jnp.int32) * 4
        + center_fit.astype(jnp.int32)
        - (~moved_known).astype(jnp.int32) * 120
        - move_cards
    )
    return (10 + _first_by_score(legal, score)).astype(jnp.int32)


def _hint_action(obs, mem):
    card_ids = _card_ids()
    hint_features = obs.node_features[9:13, :]
    revealed = jnp.sum(hint_features[:, 3:6].astype(jnp.int32), axis=1) > 0
    placed = hint_features[:, 8] == 1
    reveal_legal = obs.legal_actions[739:743]
    reveal_mask = reveal_legal & (~revealed)
    reveal_action = 739 + _first_by_score(reveal_mask, 3 - jnp.arange(4, dtype=jnp.int32))

    eff_colors, eff_known = _effective_colors(mem["colors"])
    card_x = obs.positions[:, 0]
    card_y = obs.positions[:, 1]
    hint_sets = hint_features[:, 3:6].astype(jnp.int32)

    best_action = jnp.asarray(743, dtype=jnp.int32)
    best_score = jnp.asarray(-1000000, dtype=jnp.int32)
    for h in range(4):
        hint_ok = revealed[h] & (~placed[h])
        color_ok = hint_sets[h, jnp.maximum(eff_colors, 0)] > 0
        card_ok = eff_known & color_ok & (obs.node_features[:9, 8] == 0)
        place_actions = 743 + h * 81 + card_x * 9 + card_y
        legal = obs.legal_actions[place_actions]
        shared = mem["other_seen"] & eff_known
        scores = (
            card_ok.astype(jnp.int32) * 1000
            + shared.astype(jnp.int32) * 350
            - card_ids
            - h * 20
        )
        idx = _first_by_score(legal & hint_ok, scores)
        action = place_actions[idx].astype(jnp.int32)
        score = jnp.max(jnp.where(legal & hint_ok, scores, jnp.full((9,), -1000000, dtype=jnp.int32)))
        better = score > best_score
        best_score = jnp.where(better, score, best_score)
        best_action = jnp.where(better, action, best_action)

    fallback = jnp.argmax(obs.legal_actions[743:1067]).astype(jnp.int32) + 743
    place_action = jnp.where(best_score > -1000000, best_action, fallback.astype(jnp.int32))
    return jnp.where(jnp.any(reveal_mask), reveal_action.astype(jnp.int32), place_action.astype(jnp.int32))


def _active_action(obs, mem):
    substep = obs.node_features[0, 11].astype(jnp.int32)
    eff_colors, eff_known = _effective_colors(mem["colors"])
    solved = _board_solved(obs.positions.astype(jnp.int32), eff_colors, eff_known)
    can_end = obs.legal_actions[1067] & solved

    def step0(_):
        return jnp.where(can_end, jnp.asarray(1067, dtype=jnp.int32), _observe_action(obs, mem))

    branches = (
        step0,
        lambda _: _observe_action(obs, mem),
        lambda _: _move_action(obs, mem),
        lambda _: _hint_action(obs, mem),
    )
    return jax.lax.switch(substep, branches, operand=None)


def agent_act(obs, reset, memory, rng):
    mem = _update_memory(obs, reset, memory)
    active = obs.node_features[0, 10] == 1
    action = jax.lax.cond(
        active,
        lambda _: _active_action(obs, mem),
        lambda _: jnp.asarray(0, dtype=jnp.int32),
        operand=None,
    )
    trace = {
        "known_count": jnp.sum((mem["colors"] >= 0).astype(jnp.int32)),
        "other_seen_count": jnp.sum(mem["other_seen"].astype(jnp.int32)),
        "seat": mem["seat"],
        "action": action,
    }
    return action.astype(jnp.int32), mem, trace
