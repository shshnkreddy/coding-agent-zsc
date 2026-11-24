"""Partner-readable heuristic for the 9-card Yokai task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'public_region_yokai', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

UNKNOWN = 3
TARGET_BASE_X = 2


def init_agent():
    return {
        "colors": jnp.full((9,), UNKNOWN, dtype=jnp.int32),
        "prev_pos": jnp.full((9, 2), -1, dtype=jnp.int32),
    }


def _pick_by_score(valid, score, fallback):
    valid = valid.astype(jnp.bool_)
    any_valid = jnp.any(valid)
    scored = jnp.where(valid, score, jnp.full_like(score, -1_000_000))
    return jnp.where(any_valid, jnp.argmax(scored).astype(jnp.int32), fallback)


def _min_legal_action(mask):
    idx = jnp.arange(mask.shape[0], dtype=jnp.int32)
    score = jnp.where(mask, 2_000 - idx, -1_000_000)
    return jnp.argmax(score).astype(jnp.int32)


def _own_known_colors(obs):
    card_feat = obs.node_features[:9]
    color_bits = card_feat[:, :3] > 0
    own_known = jnp.any(color_bits, axis=1)
    own_color = jnp.argmax(color_bits.astype(jnp.int32), axis=1).astype(jnp.int32)
    return own_known, own_color


def _target_good(pos, colors):
    x = pos[:, 0]
    y = pos[:, 1]
    return (colors < 3) & (x == colors + TARGET_BASE_X) & (y >= 3) & (y <= 5)


def _public_colors(pos, colors):
    x = pos[:, 0]
    y = pos[:, 1]
    in_signal_region = (x >= TARGET_BASE_X) & (x < TARGET_BASE_X + 3) & (y >= 3) & (y <= 5)
    positional_color = (x - TARGET_BASE_X).astype(jnp.int32)
    return jnp.where((colors == UNKNOWN) & in_signal_region, positional_color, colors)


def _update_memory(obs, reset, memory):
    reset_colors = jnp.full((9,), UNKNOWN, dtype=jnp.int32)
    colors = jnp.where(reset, reset_colors, memory["colors"])
    prev_pos = jnp.where(reset, obs.positions, memory["prev_pos"])

    own_known, own_color = _own_known_colors(obs)
    colors = jnp.where(own_known, own_color, colors)

    moved = jnp.any(obs.positions != prev_pos, axis=1)
    x = obs.positions[:, 0]
    signal_color = (x - TARGET_BASE_X).astype(jnp.int32)
    in_signal_region = (x >= TARGET_BASE_X) & (x < TARGET_BASE_X + 3)
    public_signal = moved & in_signal_region & (colors == UNKNOWN)
    colors = jnp.where(public_signal, signal_color, colors)
    return colors


def _clustered(obs, colors):
    colors = _public_colors(obs.positions, colors)
    adj = obs.adjacency_matrix[:9, :9] > 0
    all_known = jnp.all(colors < 3)

    def color_connected(c):
        mask = colors == c
        count_ok = jnp.sum(mask.astype(jnp.int32)) == 3
        root = jnp.argmax(mask.astype(jnp.int32))
        reach = mask & adj[root]
        for _ in range(8):
            expanded = (jnp.matmul(reach.astype(jnp.int32), adj.astype(jnp.int32)) > 0) & mask
            reach = reach | expanded
        return count_ok & jnp.all(jnp.where(mask, reach, True))

    ok0 = color_connected(jnp.asarray(0, dtype=jnp.int32))
    ok1 = color_connected(jnp.asarray(1, dtype=jnp.int32))
    ok2 = color_connected(jnp.asarray(2, dtype=jnp.int32))
    return all_known & ok0 & ok1 & ok2


def _observe_action(obs, colors):
    card_ids = jnp.arange(9, dtype=jnp.int32)
    obs_actions = 1 + card_ids
    own_known, _ = _own_known_colors(obs)
    legal = obs.legal_actions[obs_actions]
    partner_seen = obs.node_features[:9, 9] > 0
    unknown_public = colors == UNKNOWN
    unlocked = obs.node_features[:9, 8] == 0
    useful = legal & unlocked
    score = (
        unknown_public.astype(jnp.int32) * 20_000
        + (~own_known).astype(jnp.int32) * 8_000
        + (~partner_seen).astype(jnp.int32) * 500
        + (8 - card_ids)
    )
    picked_card = _pick_by_score(useful, score, jnp.asarray(0, dtype=jnp.int32))
    return (picked_card + 1).astype(jnp.int32)


def _move_action(obs, colors):
    card_ids = jnp.arange(9, dtype=jnp.int32)
    dest_ids = jnp.arange(81, dtype=jnp.int32)
    legal_moves = obs.legal_actions[10:739].reshape((9, 81))
    pos = obs.positions
    locked = obs.node_features[:9, 8] > 0
    good = _target_good(pos, colors)
    movable_known = (colors < 3) & (~locked) & (~good)

    target_y = jnp.asarray([3, 4, 5], dtype=jnp.int32)
    exact_x = colors[:, None] + TARGET_BASE_X
    exact_y = target_y[None, :]
    exact_actions = 10 + card_ids[:, None] * 81 + exact_x * 9 + exact_y
    exact_legal = jnp.take(obs.legal_actions, exact_actions, mode="clip")
    exact_valid = movable_known[:, None] & exact_legal
    wrong_target_blocker = (
        (pos[:, 0] >= TARGET_BASE_X)
        & (pos[:, 0] < TARGET_BASE_X + 3)
        & (pos[:, 1] >= 3)
        & (pos[:, 1] <= 5)
        & (pos[:, 0] != colors + TARGET_BASE_X)
    )
    exact_score = (
        wrong_target_blocker[:, None].astype(jnp.int32) * 10_000
        + (8 - card_ids[:, None]) * 100
        + (2 - jnp.abs(exact_y - 4)) * 10
    )
    exact_flat = _pick_by_score(
        exact_valid.reshape((-1,)),
        exact_score.reshape((-1,)),
        jnp.asarray(0, dtype=jnp.int32),
    )
    exact_action = jnp.take(exact_actions.reshape((-1,)), exact_flat, mode="clip")
    has_exact = jnp.any(exact_valid)

    dest_x = (dest_ids // 9).astype(jnp.int32)
    dest_y = (dest_ids % 9).astype(jnp.int32)
    dist = jnp.abs(dest_x[None, :] - (colors[:, None] + TARGET_BASE_X)) + jnp.abs(dest_y[None, :] - 4)
    own_column = dest_x[None, :] == (colors[:, None] + TARGET_BASE_X)
    center_band = (dest_y[None, :] >= 3) & (dest_y[None, :] <= 5)
    soft_valid = movable_known[:, None] & legal_moves
    soft_score = (
        own_column.astype(jnp.int32) * 5_000
        + center_band.astype(jnp.int32) * 1_000
        + wrong_target_blocker[:, None].astype(jnp.int32) * 20_000
        - dist.astype(jnp.int32) * 100
        + (8 - card_ids[:, None])
    )
    soft_flat = _pick_by_score(
        soft_valid.reshape((-1,)),
        soft_score.reshape((-1,)),
        jnp.asarray(0, dtype=jnp.int32),
    )
    soft_card = soft_flat // 81
    soft_dest = soft_flat % 81
    soft_action = (10 + soft_card * 81 + soft_dest).astype(jnp.int32)
    has_soft = jnp.any(soft_valid)

    pair_adj = (
        jnp.abs(dest_x[None, :, None] - pos[None, None, :, 0])
        + jnp.abs(dest_y[None, :, None] - pos[None, None, :, 1])
    ) == 1
    same_color = (
        (colors[:, None] == colors[None, :])
        & (colors[:, None] < 3)
        & (card_ids[:, None] != card_ids[None, :])
    )
    diff_color = (
        (colors[:, None] != colors[None, :])
        & (colors[:, None] < 3)
        & (colors[None, :] < 3)
    )
    same_adj = jnp.sum((pair_adj & same_color[:, None, :]).astype(jnp.int32), axis=2)
    diff_adj = jnp.sum((pair_adj & diff_color[:, None, :]).astype(jnp.int32), axis=2)
    dest_in_own_target = own_column & center_band
    cluster_valid = soft_valid & (same_adj > 0) & own_column
    cluster_score = (
        same_adj * 60_000
        + dest_in_own_target.astype(jnp.int32) * 8_000
        + own_column.astype(jnp.int32) * 2_000
        + wrong_target_blocker[:, None].astype(jnp.int32) * 20_000
        - diff_adj * 1_000
        - dist.astype(jnp.int32) * 20
        + (8 - card_ids[:, None])
    )
    cluster_flat = _pick_by_score(
        cluster_valid.reshape((-1,)),
        cluster_score.reshape((-1,)),
        jnp.asarray(0, dtype=jnp.int32),
    )
    cluster_card = cluster_flat // 81
    cluster_dest = cluster_flat % 81
    cluster_action = (10 + cluster_card * 81 + cluster_dest).astype(jnp.int32)
    has_cluster = jnp.any(cluster_valid)

    fallback = _min_legal_action(obs.legal_actions)
    slot_action = jnp.where(has_exact, exact_action, jnp.where(has_soft, soft_action, fallback))
    return jnp.where(has_cluster, cluster_action, slot_action).astype(jnp.int32)


def _hint_action(obs, colors):
    card_ids = jnp.arange(9, dtype=jnp.int32)
    hint_ids = jnp.arange(4, dtype=jnp.int32)
    reveal_actions = 739 + hint_ids
    reveal_legal = obs.legal_actions[reveal_actions]
    reveal_score = 100 - hint_ids
    reveal_idx = _pick_by_score(reveal_legal, reveal_score, jnp.asarray(0, dtype=jnp.int32))
    reveal_action = 739 + reveal_idx
    has_reveal = jnp.any(reveal_legal)

    hint_sets = obs.node_features[9:13, 3:6] > 0
    hint_unplaced = obs.node_features[9:13, 8] == 0
    card_unlocked = obs.node_features[:9, 8] == 0
    public_colors = _public_colors(obs.positions, colors)
    good = _target_good(obs.positions, public_colors)
    card_x = obs.positions[:, 0]
    card_y = obs.positions[:, 1]
    place_actions = 743 + hint_ids[:, None] * 81 + card_x[None, :] * 9 + card_y[None, :]
    place_legal = jnp.take(obs.legal_actions, place_actions, mode="clip")
    known = colors < 3
    card_color_match = jnp.take_along_axis(hint_sets, colors[None, :].clip(0, 2), axis=1)
    place_valid = place_legal & hint_unplaced[:, None] & card_unlocked[None, :]
    place_score = (
        good[None, :].astype(jnp.int32) * 50_000
        + card_color_match.astype(jnp.int32) * 20_000
        + known[None, :].astype(jnp.int32) * 500
        + (3 - hint_ids[:, None]) * 20
        + (8 - card_ids[None, :])
    )
    place_flat = _pick_by_score(
        place_valid.reshape((-1,)),
        place_score.reshape((-1,)),
        jnp.asarray(0, dtype=jnp.int32),
    )
    place_action = jnp.take(place_actions.reshape((-1,)), place_flat, mode="clip")
    fallback = _min_legal_action(obs.legal_actions)
    chosen_place = jnp.where(jnp.any(place_valid), place_action, fallback)
    return jnp.where(has_reveal, reveal_action, chosen_place).astype(jnp.int32)


def _substep0(obs, colors):
    end = jnp.asarray(1067, dtype=jnp.int32)
    can_end = obs.legal_actions[end]
    should_end = can_end & _clustered(obs, colors)
    return jnp.where(should_end, end, _observe_action(obs, colors)).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    colors = _update_memory(obs, reset, memory)
    substep = obs.node_features[0, 11].astype(jnp.int32)
    inactive = obs.node_features[0, 10] == 0

    action = jax.lax.switch(
        substep,
        (
            lambda: _substep0(obs, colors),
            lambda: _observe_action(obs, colors),
            lambda: _move_action(obs, colors),
            lambda: _hint_action(obs, colors),
        ),
    )
    action = jnp.where(inactive, jnp.asarray(0, dtype=jnp.int32), action).astype(jnp.int32)

    next_memory = {
        "colors": colors,
        "prev_pos": obs.positions.astype(jnp.int32),
    }
    trace = {
        "known_cards": jnp.sum((colors < 3).astype(jnp.int32)),
        "clustered": _clustered(obs, colors),
    }
    return action, next_memory, trace
