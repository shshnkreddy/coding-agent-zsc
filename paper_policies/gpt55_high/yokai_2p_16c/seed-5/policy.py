"""Convention-based Yokai policy for cross-play robustness."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'fixed_home_candidate',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 4,
    'deck_size': 16,
    'grid_size': 10,
    'num_hints': 7,
    'num_nodes': 23,
    'node_features': 15,
    'num_actions': 2325,
    'policy_structure': 'shared_local',
}


def init_agent():
    return {
        'known_color': jnp.full((16,), -1, dtype=jnp.int8),
    }


def _home_origin(color):
    color_i = color.astype(jnp.int32)
    return 3 + 2 * (color_i // 2), 3 + 2 * (color_i % 2)


def _in_color_home_xy(x, y, color):
    x0, y0 = _home_origin(color)
    return (x >= x0) & (x <= x0 + 1) & (y >= y0) & (y <= y0 + 1)


def _dist_to_home_xy(x, y, color):
    x0, y0 = _home_origin(color)
    dx = jnp.maximum(jnp.maximum(x0 - x, x - (x0 + 1)), 0)
    dy = jnp.maximum(jnp.maximum(y0 - y, y - (y0 + 1)), 0)
    return dx + dy


def _fallback_legal(legal_actions):
    return jnp.argmax(legal_actions).astype(jnp.int32)


def _observe_action(obs, known_color):
    legal = obs.legal_actions[1:17]
    ids = jnp.arange(16, dtype=jnp.int32)
    locked = obs.node_features[:16, 10] > 0
    partner_seen = obs.node_features[:16, 11] > 0
    unknown = known_color < 0
    fresh_for_team = unknown & (~partner_seen)
    score = (
        fresh_for_team.astype(jnp.int32) * 4000
        + (unknown & (~locked)).astype(jnp.int32) * 2000
        + ((~partner_seen) & (~locked)).astype(jnp.int32) * 100
        - ids
    )
    score = jnp.where(legal, score, -1000000)
    return (1 + jnp.argmax(score)).astype(jnp.int32)


def _end_condition(obs, known_color):
    pos = obs.positions.astype(jnp.int32)
    x = pos[:, 0]
    y = pos[:, 1]
    in_target_grid = (x >= 3) & (x <= 6) & (y >= 3) & (y <= 6)
    partner_seen = obs.node_features[:16, 11] > 0
    seen_by_someone = (known_color >= 0) | partner_seen

    colors = jnp.maximum(known_color.astype(jnp.int32), 0)
    known_in_home = _in_color_home_xy(x, y, colors)
    known_consistent = (known_color < 0) | known_in_home

    return (
        obs.legal_actions[2324]
        & jnp.all(in_target_grid)
        & jnp.all(seen_by_someone)
        & jnp.all(known_consistent)
    )


def _move_action(obs, known_color):
    move_legal = obs.legal_actions[17:1617]
    move_id = jnp.arange(1600, dtype=jnp.int32)
    card = move_id // 100
    rem = move_id - card * 100
    dst_x = rem // 10
    dst_y = rem - dst_x * 10

    card_color = known_color[card].astype(jnp.int32)
    known = card_color >= 0
    cur = obs.positions[card].astype(jnp.int32)
    cur_x = cur[:, 0]
    cur_y = cur[:, 1]

    in_own_home_now = _in_color_home_xy(cur_x, cur_y, jnp.maximum(card_color, 0))
    dst_in_own_home = _in_color_home_xy(dst_x, dst_y, jnp.maximum(card_color, 0))
    dist = _dist_to_home_xy(dst_x, dst_y, jnp.maximum(card_color, 0))

    in_any_home_now = (cur_x >= 3) & (cur_x <= 6) & (cur_y >= 3) & (cur_y <= 6)
    current_home_color = ((cur_x >= 5).astype(jnp.int32) * 2
                          + (cur_y >= 5).astype(jnp.int32))
    wrong_home_now = in_any_home_now & known & (current_home_color != card_color)

    score = (
        2000
        + dst_in_own_home.astype(jnp.int32) * 8000
        + wrong_home_now.astype(jnp.int32) * 1200
        - in_own_home_now.astype(jnp.int32) * 7000
        - dist.astype(jnp.int32) * 25
        - card.astype(jnp.int32)
    )
    score = jnp.where(move_legal & known, score, -1000000)
    best_idx = jnp.argmax(score).astype(jnp.int32)
    best_score = jnp.max(score)
    preferred = (17 + best_idx).astype(jnp.int32)
    fallback = _fallback_legal(obs.legal_actions)
    return jnp.where(best_score > -999000, preferred, fallback).astype(jnp.int32)


def _hint_action(obs, known_color):
    reveal_legal = obs.legal_actions[1617:1624]
    reveal = (1617 + jnp.argmax(reveal_legal)).astype(jnp.int32)
    can_reveal = jnp.any(reveal_legal)

    place_legal = obs.legal_actions[1624:2324]
    place_id = jnp.arange(700, dtype=jnp.int32)
    hint = place_id // 100
    rem = place_id - hint * 100
    x = rem // 10
    y = rem - x * 10

    pos = obs.positions.astype(jnp.int32)
    at_xy = (pos[:, 0] == x[:, None]) & (pos[:, 1] == y[:, None])
    has_card = jnp.any(at_xy, axis=1)
    card = jnp.argmax(at_xy.astype(jnp.int32), axis=1)
    color = known_color[card].astype(jnp.int32)
    known = has_card & (color >= 0)
    hint_colors = obs.node_features[16 + hint, 4:8] > 0
    hint_matches = hint_colors[place_id, jnp.maximum(color, 0)]
    in_home = _in_color_home_xy(x, y, jnp.maximum(color, 0))

    score = (
        (known & hint_matches & in_home).astype(jnp.int32) * 5000
        + (known & hint_matches).astype(jnp.int32) * 2000
        + in_home.astype(jnp.int32) * 50
        - hint.astype(jnp.int32)
    )
    score = jnp.where(place_legal, score, -1000000)
    place = (1624 + jnp.argmax(score)).astype(jnp.int32)
    fallback = _fallback_legal(obs.legal_actions)
    place = jnp.where(jnp.any(place_legal), place, fallback).astype(jnp.int32)
    return jnp.where(can_reveal, reveal, place).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    empty = init_agent()
    known = jnp.where(reset, empty['known_color'], memory['known_color'])

    color_bits = obs.node_features[:16, 0:4] > 0
    visible = jnp.any(color_bits, axis=1)
    visible_color = jnp.argmax(color_bits.astype(jnp.int32), axis=1).astype(jnp.int8)
    known = jnp.where(visible, visible_color, known).astype(jnp.int8)

    active = obs.node_features[0, 12] > 0
    substep = obs.node_features[0, 13].astype(jnp.int32)

    observe = _observe_action(obs, known)
    step0 = jnp.where(_end_condition(obs, known), jnp.asarray(2324, dtype=jnp.int32), observe)
    move = _move_action(obs, known)
    hint = _hint_action(obs, known)

    action = _fallback_legal(obs.legal_actions)
    action = jnp.where(substep == 0, step0, action)
    action = jnp.where(substep == 1, observe, action)
    action = jnp.where(substep == 2, move, action)
    action = jnp.where(substep == 3, hint, action)
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32)).astype(jnp.int32)

    next_memory = {'known_color': known}
    return action, next_memory, {}
