"""Deterministic Yokai policy for robust, convention-light color sorting."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'canonical_sorter_v1',
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

_BIG_NEG = -1_000_000
_TARGET_CELLS_RAW = (
    ((3, 3), (3, 4), (4, 3), (4, 4)),
    ((5, 3), (5, 4), (6, 3), (6, 4)),
    ((3, 5), (3, 6), (4, 5), (4, 6)),
    ((5, 5), (5, 6), (6, 5), (6, 6)),
)


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _canonical_color_xy(x, y):
    in_x0 = (x >= 3) & (x <= 4)
    in_x1 = (x >= 5) & (x <= 6)
    in_y0 = (y >= 3) & (y <= 4)
    in_y1 = (y >= 5) & (y <= 6)
    color = jnp.where(in_x1 & in_y0, 1, 0)
    color = jnp.where(in_x0 & in_y1, 2, color)
    color = jnp.where(in_x1 & in_y1, 3, color)
    return jnp.where((in_x0 | in_x1) & (in_y0 | in_y1), color, -1)


def _in_any_target_xy(x, y):
    return _canonical_color_xy(x, y) >= 0


def _dist_to_color_targets_xy(x, y, color):
    targets = jnp.asarray(_TARGET_CELLS_RAW, dtype=jnp.int32)[color]
    d = jnp.abs(targets[:, 0] - x) + jnp.abs(targets[:, 1] - y)
    return jnp.min(d)


def _is_own_target_xy(x, y, color):
    return _canonical_color_xy(x, y) == color


def _argmax_legal(scores, legal):
    masked = jnp.where(legal, scores, jnp.asarray(_BIG_NEG, dtype=jnp.int32))
    return jnp.argmax(masked).astype(jnp.int32)


def _observe_action(obs):
    features = obs.node_features
    legal = obs.legal_actions[1:17]
    known = jnp.sum(features[:16, 0:4], axis=1) > 0
    locked = features[:16, 10] > 0
    partner_seen = features[:16, 11] > 0
    unknown = (~known) & (~locked)
    scores = (
        unknown.astype(jnp.int32) * 1000
        + ((~partner_seen) & unknown).astype(jnp.int32) * 300
        - jnp.arange(16, dtype=jnp.int32)
    )
    idx = _argmax_legal(scores, legal)
    return (idx + 1).astype(jnp.int32)


def _canonical_layout_complete(obs):
    positions = obs.positions
    all_target_cells = jnp.asarray(_TARGET_CELLS_RAW, dtype=jnp.int32).reshape((16, 2))
    eq = (
        (positions[:, None, 0] == all_target_cells[None, :, 0])
        & (positions[:, None, 1] == all_target_cells[None, :, 1])
    )
    located = jnp.all(jnp.any(eq, axis=1))
    known = jnp.sum(obs.node_features[:16, 0:4], axis=1) > 0
    color = jnp.argmax(obs.node_features[:16, 0:4], axis=1).astype(jnp.int32)
    region_color = _canonical_color_xy(positions[:, 0], positions[:, 1])
    known_consistent = jnp.all((~known) | (region_color == color))
    partner_seen = obs.node_features[:16, 11] > 0
    collectively_seen = jnp.all(known | partner_seen)
    return located & collectively_seen & known_consistent


def _move_action(obs):
    features = obs.node_features
    positions = obs.positions
    legal = obs.legal_actions[17:1617]

    colors_oh = features[:16, 0:4]
    known = jnp.sum(colors_oh, axis=1) > 0
    color = jnp.argmax(colors_oh, axis=1).astype(jnp.int32)

    move_ids = jnp.arange(1600, dtype=jnp.int32)
    card = move_ids // 100
    x = (move_ids % 100) // 10
    y = move_ids % 10
    cur_x = positions[card, 0]
    cur_y = positions[card, 1]
    card_known = known[card]
    card_color = color[card]

    cur_dist = jax.vmap(_dist_to_color_targets_xy)(cur_x, cur_y, card_color)
    dst_dist = jax.vmap(_dist_to_color_targets_xy)(x, y, card_color)
    cur_own = jax.vmap(_is_own_target_xy)(cur_x, cur_y, card_color)
    dst_own = jax.vmap(_is_own_target_xy)(x, y, card_color)
    dst_any_target = _in_any_target_xy(x, y)
    cur_public_color = _canonical_color_xy(cur_x, cur_y)
    cur_in_target = cur_public_color >= 0
    known_wrong_cell = card_known & cur_in_target & (cur_public_color != card_color)
    unknown_in_target = (~card_known) & cur_in_target

    improve = cur_dist - dst_dist
    known_score = (
        card_known.astype(jnp.int32)
        * (improve * 35 + dst_own.astype(jnp.int32) * 250 - cur_own.astype(jnp.int32) * 180)
    )

    center_dist = jnp.abs(x - 4) + jnp.abs(y - 4)
    blocker_score = (
        ((known_wrong_cell | unknown_in_target) & (~dst_any_target)).astype(jnp.int32)
        * (120 - center_dist * 4)
    )
    filler_penalty = dst_any_target.astype(jnp.int32) * 30
    tie = -card * 2 - x - y
    scores = known_score + blocker_score - filler_penalty + tie
    idx = _argmax_legal(scores, legal)
    return (idx + 17).astype(jnp.int32)


def _hint_action(obs):
    features = obs.node_features
    reveal_legal = obs.legal_actions[1617:1624]
    revealed = jnp.sum(features[16:23, 4:8], axis=1) > 0
    reveal_scores = (~revealed).astype(jnp.int32) * 100 - jnp.arange(7, dtype=jnp.int32)
    reveal_idx = _argmax_legal(reveal_scores, reveal_legal)
    reveal_action = (1617 + reveal_idx).astype(jnp.int32)
    can_reveal = jnp.any(reveal_legal)

    positions = obs.positions
    colors_oh = features[:16, 0:4]
    known = jnp.sum(colors_oh, axis=1) > 0
    card_color = jnp.argmax(colors_oh, axis=1).astype(jnp.int32)

    cell_match = (
        (positions[:, None, 0] == (jnp.arange(100, dtype=jnp.int32) // 10)[None, :])
        & (positions[:, None, 1] == (jnp.arange(100, dtype=jnp.int32) % 10)[None, :])
    )
    cell_has_card = jnp.any(cell_match, axis=0)
    cell_card = jnp.argmax(cell_match.astype(jnp.int32), axis=0).astype(jnp.int32)

    place_ids = jnp.arange(700, dtype=jnp.int32)
    place_hint = place_ids // 100
    place_x = (place_ids % 100) // 10
    place_y = place_ids % 10
    place_cell = place_x * 10 + place_y
    place_card = cell_card[place_cell]
    place_known = known[place_card] & cell_has_card[place_cell]
    hint_sets = features[16:23, 4:8] > 0
    contains = hint_sets[place_hint, card_color[place_card]]
    own_target = jax.vmap(_is_own_target_xy)(
        positions[place_card, 0], positions[place_card, 1], card_color[place_card]
    )
    place_scores = (
        (place_known & contains).astype(jnp.int32) * 300
        + own_target.astype(jnp.int32) * 80
        - place_hint * 3
        - place_x
        - place_y
    )
    place_idx = _argmax_legal(place_scores, obs.legal_actions[1624:2324])
    place_action = (1624 + place_idx).astype(jnp.int32)
    return jnp.where(can_reveal, reveal_action, place_action).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    substep = obs.node_features[0, 13].astype(jnp.int32)
    fallback = _argmax_legal(jnp.arange(2325, dtype=jnp.int32), obs.legal_actions)
    solved = _canonical_layout_complete(obs)
    observe = _observe_action(obs)
    move = _move_action(obs)
    hint = _hint_action(obs)
    end_or_observe = jnp.where(solved & obs.legal_actions[2324], 2324, observe)
    candidates = jnp.asarray([end_or_observe, observe, move, hint], dtype=jnp.int32)
    action = candidates[jnp.clip(substep, 0, 3)]
    action = jnp.where(obs.legal_actions[action], action, fallback).astype(jnp.int32)
    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    return action, next_memory, {}
