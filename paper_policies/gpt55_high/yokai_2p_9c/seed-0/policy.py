"""Deterministic local Yokai policy with public, simple conventions."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
NUM_HINTS = 4

OBSERVE_BASE = 1
MOVE_BASE = 10
REVEAL_BASE = 739
PLACE_BASE = 743
END_GAME = 1067

PAIR_I_VALUES = (0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1,
                 2, 2, 2, 2, 2, 2,
                 3, 3, 3, 3, 3,
                 4, 4, 4, 4,
                 5, 5, 5,
                 6, 6,
                 7)
PAIR_J_VALUES = (1, 2, 3, 4, 5, 6, 7, 8,
                 2, 3, 4, 5, 6, 7, 8,
                 3, 4, 5, 6, 7, 8,
                 4, 5, 6, 7, 8,
                 5, 6, 7, 8,
                 6, 7, 8,
                 7, 8,
                 8)
TARGET_VALUES = (
    ((3, 2), (3, 3), (3, 4)),
    ((4, 2), (4, 3), (4, 4)),
    ((5, 2), (5, 3), (5, 4)),
)


def _card_ids():
    return jnp.arange(NUM_CARDS, dtype=jnp.int32)


def _move_ids():
    return jnp.arange(729, dtype=jnp.int32)


def _place_ids():
    return jnp.arange(324, dtype=jnp.int32)


def _pair_ids():
    return (
        jnp.asarray(PAIR_I_VALUES, dtype=jnp.int32),
        jnp.asarray(PAIR_J_VALUES, dtype=jnp.int32),
    )


def init_agent():
    return jnp.zeros((NUM_CARDS, NUM_COLORS), dtype=jnp.int8)


def _known_from_memory(memory, obs):
    seen_now = obs.node_features[:NUM_CARDS, :NUM_COLORS]
    return jnp.maximum(memory, seen_now).astype(jnp.int8)


def _color_ids_and_known(colors_onehot):
    counts = jnp.sum(colors_onehot.astype(jnp.int32), axis=0)
    known = jnp.sum(colors_onehot, axis=1) > 0
    unknown_count = NUM_CARDS - jnp.sum(known.astype(jnp.int32))
    missing_color = jnp.argmin(counts)
    raw_color = jnp.argmax(colors_onehot.astype(jnp.int32), axis=1).astype(jnp.int32)
    inferred = jnp.logical_or(known, unknown_count == 1)
    color_id = jnp.where(known, raw_color, missing_color)
    return color_id, inferred, known, counts


def _first_true(mask, offset):
    idx = jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)
    return (offset + idx).astype(jnp.int32)


def _observe_action(obs, colors_onehot):
    legal = obs.legal_actions[OBSERVE_BASE:OBSERVE_BASE + NUM_CARDS]
    unknown = jnp.sum(colors_onehot, axis=1) == 0
    partner_seen = obs.node_features[:NUM_CARDS, 9] > 0
    preferred = jnp.logical_and(jnp.logical_and(legal, unknown), jnp.logical_not(partner_seen))
    own_unknown = jnp.logical_and(legal, unknown)
    card_ids = _card_ids()
    # Stable-ID order is easy for partners to anticipate; fall back to any
    # legal observe action if all remaining cards are already known or locked.
    scores = jnp.where(
        preferred,
        100 - card_ids,
        jnp.where(own_unknown, 50 - card_ids, jnp.where(legal, 10 - card_ids, -1000)),
    )
    card = jnp.argmax(scores).astype(jnp.int32)
    return (OBSERVE_BASE + card).astype(jnp.int32)


def _positions_after_move(positions):
    move_ids = _move_ids()
    move_card = move_ids // 81
    dest_x = (move_ids % 81) // 9
    dest_y = move_ids % 9
    dest = jnp.stack([dest_x, dest_y], axis=1)
    return jnp.where(
        _card_ids()[None, :, None] == move_card[:, None, None],
        dest[:, None, :],
        positions[None, :, :],
    )


def _same_color_edge_counts(all_positions, color_id, color_known):
    pair_i, pair_j = _pair_ids()
    pi = all_positions[:, pair_i, :]
    pj = all_positions[:, pair_j, :]
    adjacent = jnp.sum(jnp.abs(pi - pj), axis=2) == 1
    same = color_id[pair_i] == color_id[pair_j]
    known_pair = jnp.logical_and(color_known[pair_i], color_known[pair_j])
    same_edges = jnp.logical_and(adjacent, jnp.logical_and(same, known_pair))

    counts = []
    for color in range(NUM_COLORS):
        in_color = jnp.logical_and(color_id == color, color_known)
        color_pair = jnp.logical_and(in_color[pair_i], in_color[pair_j])
        counts.append(jnp.sum(jnp.logical_and(adjacent, color_pair).astype(jnp.int32), axis=1))
    color_edges = jnp.stack(counts, axis=1)
    return jnp.sum(same_edges.astype(jnp.int32), axis=1), color_edges


def _target_distance(all_positions, color_id, color_known):
    targets = jnp.asarray(TARGET_VALUES, dtype=jnp.int32)
    target = targets[color_id]
    distances = jnp.sum(jnp.abs(all_positions[:, :, None, :] - target[None, :, :, :]), axis=3)
    best = jnp.min(distances, axis=2)
    return jnp.sum(jnp.where(color_known[None, :], best, 0), axis=1)


def _target_color_for_positions(positions):
    targets = jnp.asarray(TARGET_VALUES, dtype=jnp.int32)
    distances = jnp.sum(jnp.abs(positions[:, None, None, :] - targets[None, :, :, :]), axis=3)
    return jnp.argmin(jnp.min(distances, axis=2), axis=1).astype(jnp.int32)


def _move_action(obs, colors_onehot):
    positions = obs.positions.astype(jnp.int32)
    color_id, color_known, _, _ = _color_ids_and_known(colors_onehot)
    legal = obs.legal_actions[MOVE_BASE:MOVE_BASE + 729]
    all_positions = _positions_after_move(positions)

    same_edges, color_edges = _same_color_edge_counts(all_positions, color_id, color_known)
    color_card_counts = jnp.stack(
        [jnp.sum(jnp.logical_and(color_id == c, color_known).astype(jnp.int32)) for c in range(NUM_COLORS)]
    )
    enough_known = color_card_counts >= 2
    connected_groups = jnp.sum(
        jnp.logical_and(color_edges >= 1, enough_known[None, :]).astype(jnp.int32), axis=1
    )
    complete_groups = jnp.sum(
        jnp.logical_and(color_edges >= 2, color_card_counts[None, :] == 3).astype(jnp.int32), axis=1
    )
    target_penalty = _target_distance(all_positions, color_id, color_known)

    moved_card = _move_ids() // 81
    moved_known = color_known[moved_card]
    scores = (
        220 * complete_groups
        + 80 * connected_groups
        + 22 * same_edges
        - 3 * target_penalty
        + moved_known.astype(jnp.int32)
    )
    scores = jnp.where(legal, scores, -1_000_000)
    return (MOVE_BASE + jnp.argmax(scores).astype(jnp.int32)).astype(jnp.int32)


def _all_estimated_connected(positions, color_id, color_known):
    known_count = jnp.sum(color_known.astype(jnp.int32))
    pair_i, pair_j = _pair_ids()
    pi = positions[pair_i]
    pj = positions[pair_j]
    adjacent = jnp.sum(jnp.abs(pi - pj), axis=1) == 1
    checks = []
    for color in range(NUM_COLORS):
        in_color = color_id == color
        edge_count = jnp.sum(
            jnp.logical_and(adjacent, jnp.logical_and(in_color[pair_i], in_color[pair_j])).astype(jnp.int32)
        )
        checks.append(edge_count >= 2)
    return jnp.logical_and(known_count >= 8, jnp.all(jnp.stack(checks)))


def _target_layout_end_ready(positions, color_id, own_known):
    known_count = jnp.sum(own_known.astype(jnp.int32))
    target_color = _target_color_for_positions(positions)
    consistent = jnp.all(jnp.where(own_known, target_color == color_id, True))
    pair_i, pair_j = _pair_ids()
    pi = positions[pair_i]
    pj = positions[pair_j]
    adjacent = jnp.sum(jnp.abs(pi - pj), axis=1) == 1
    checks = []
    for color in range(NUM_COLORS):
        in_color = target_color == color
        edge_count = jnp.sum(
            jnp.logical_and(adjacent, jnp.logical_and(in_color[pair_i], in_color[pair_j])).astype(jnp.int32)
        )
        checks.append(edge_count >= 2)
    return jnp.logical_and(known_count >= 6, jnp.logical_and(consistent, jnp.all(jnp.stack(checks))))


def _substep0_action(obs, colors_onehot):
    color_id, inferred_known, own_known, _ = _color_ids_and_known(colors_onehot)
    positions = obs.positions.astype(jnp.int32)
    connected = _all_estimated_connected(positions, color_id, inferred_known)
    visible_ready = _target_layout_end_ready(positions, color_id, own_known)
    can_end = obs.legal_actions[END_GAME]
    should_end = jnp.logical_and(can_end, jnp.logical_or(connected, visible_ready))
    return jnp.where(should_end, END_GAME, _observe_action(obs, colors_onehot)).astype(jnp.int32)


def _hint_action(obs, colors_onehot):
    hint_features = obs.node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS]
    hint_colors = hint_features[:, 3:6].astype(jnp.int32)
    revealed = jnp.sum(hint_colors, axis=1) > 0
    placed = hint_features[:, 8] > 0

    legal_reveal = obs.legal_actions[REVEAL_BASE:REVEAL_BASE + NUM_HINTS]
    reveal_mask = jnp.logical_and(legal_reveal, jnp.logical_not(revealed))
    reveal_action = _first_true(reveal_mask, REVEAL_BASE)

    positions = obs.positions.astype(jnp.int32)
    color_id, _, known, _ = _color_ids_and_known(colors_onehot)
    place_ids = _place_ids()
    place_hint = place_ids // 81
    place_cell = place_ids % 81
    place_x = place_cell // 9
    place_y = place_cell % 9
    legal_place = obs.legal_actions[PLACE_BASE:PLACE_BASE + 324]

    same_cell = jnp.logical_and(positions[:, 0][None, :] == place_x[:, None],
                                positions[:, 1][None, :] == place_y[:, None])
    card_at_cell = jnp.argmax(same_cell.astype(jnp.int32), axis=1)
    has_card = jnp.any(same_cell, axis=1)
    card_color = color_id[card_at_cell]
    correct_known = jnp.logical_and(known[card_at_cell], hint_colors[place_hint, card_color] > 0)
    unplaced_hint = jnp.logical_not(placed[place_hint])

    # Prefer correct public hints on cards we know, and otherwise pick the
    # earliest legal placement to avoid illegal-action failures.
    place_scores = (
        1000 * correct_known.astype(jnp.int32)
        + 30 * has_card.astype(jnp.int32)
        + 5 * unplaced_hint.astype(jnp.int32)
        - place_ids
    )
    place_scores = jnp.where(legal_place, place_scores, -1_000_000)
    place_action = (PLACE_BASE + jnp.argmax(place_scores).astype(jnp.int32)).astype(jnp.int32)

    return jnp.where(jnp.any(reveal_mask), reveal_action, place_action).astype(jnp.int32)


def _active_action(obs, colors_onehot):
    substep = obs.node_features[0, 11].astype(jnp.int32)
    return jax.lax.switch(
        substep,
        (
            lambda _: _substep0_action(obs, colors_onehot),
            lambda _: _observe_action(obs, colors_onehot),
            lambda _: _move_action(obs, colors_onehot),
            lambda _: _hint_action(obs, colors_onehot),
        ),
        operand=None,
    )


def agent_act(obs, reset, memory, rng):
    del rng
    clean_memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    next_memory = _known_from_memory(clean_memory, obs)
    active = obs.node_features[0, 10] > 0
    action = jax.lax.cond(
        active,
        lambda _: _active_action(obs, next_memory),
        lambda _: jnp.asarray(0, dtype=jnp.int32),
        operand=None,
    )
    trace = {
        "known_cards": jnp.sum((jnp.sum(next_memory, axis=1) > 0).astype(jnp.int32)),
        "substep": obs.node_features[0, 11].astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
