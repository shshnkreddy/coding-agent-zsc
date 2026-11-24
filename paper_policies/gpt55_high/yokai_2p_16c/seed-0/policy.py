"""Deterministic public-convention policy for the 16-card Yokai task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'public_row_sorter', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NUM_CARDS = 16
UNKNOWN = -1


def init_agent():
    return jnp.full((NUM_CARDS,), UNKNOWN, dtype=jnp.int8)


def _first_legal(mask, offset):
    idx = jnp.argmax(mask).astype(jnp.int32)
    return idx + jnp.asarray(offset, dtype=jnp.int32)


def _known_colors_from_obs(obs, memory, reset):
    base = jnp.where(reset, jnp.full_like(memory, UNKNOWN), memory)
    visible = obs.node_features[:NUM_CARDS, :4].astype(jnp.int32)
    seen = jnp.sum(visible, axis=1) > 0
    colors = jnp.argmax(visible, axis=1).astype(jnp.int8)
    active = obs.node_features[0, 12] > 0
    return jnp.where(active & seen, colors, base)


def _safe_place_scores(obs, memory):
    place_ids = jnp.arange(700, dtype=jnp.int32)
    place_hint = place_ids // 100
    place_x = (place_ids % 100) // 10
    place_y = place_ids % 10
    colors = jnp.arange(4, dtype=jnp.int32)
    legal = obs.legal_actions[1624:2324]
    positions = obs.positions
    hints = obs.node_features[16:23, 4:8].astype(jnp.int32)

    same_x = positions[:, 0][None, :] == place_x[:, None]
    same_y = positions[:, 1][None, :] == place_y[:, None]
    at_cell = same_x & same_y
    card_at_cell = jnp.argmax(at_cell.astype(jnp.int32), axis=1)
    has_card = jnp.any(at_cell, axis=1)

    color = memory[card_at_cell].astype(jnp.int32)
    known = (color >= 0) & has_card
    hint_size = jnp.sum(hints[place_hint], axis=1)
    hint_has_color = jnp.sum(hints[place_hint] * (colors[None, :] == color[:, None]), axis=1) > 0
    known_correct = known & hint_has_color
    known_wrong = known & (~hint_has_color)

    target_y = color + 3
    in_lane = (place_y == target_y) & (place_x >= 2) & (place_x <= 7)
    in_home = in_lane & (place_x >= 3) & (place_x <= 6)
    public_lane = (place_y >= 3) & (place_y <= 6) & (place_x >= 2) & (place_x <= 7)
    score = (
        known_correct.astype(jnp.int32) * 7000
        - known_wrong.astype(jnp.int32) * 5000
        + hint_size.astype(jnp.int32) * 700
        + in_home.astype(jnp.int32) * 400
        + in_lane.astype(jnp.int32) * 100
        + public_lane.astype(jnp.int32) * 50
        + (699 - place_ids)
    )
    safe = legal & known_correct & in_lane
    return jnp.where(legal, score, -1000000), safe


def _choose_observe(obs, memory):
    card_ids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    legal = obs.legal_actions[1:17]
    partner_seen = obs.node_features[:NUM_CARDS, 11] > 0
    unknown = memory < 0
    score = (
        unknown.astype(jnp.int32) * 1000
        + (~partner_seen).astype(jnp.int32) * 50
        + (15 - card_ids)
    )
    masked = jnp.where(legal, score, -1000000)
    return jnp.argmax(masked).astype(jnp.int32) + 1


def _choose_move(obs, memory):
    move_ids = jnp.arange(1600, dtype=jnp.int32)
    move_card = move_ids // 100
    move_x = (move_ids % 100) // 10
    move_y = move_ids % 10
    legal = obs.legal_actions[17:1617]
    positions = obs.positions
    color = memory[move_card].astype(jnp.int32)
    known = color >= 0

    cur_x = positions[move_card, 0]
    cur_y = positions[move_card, 1]
    target_y = color + 3

    dest_lane = (move_y == target_y) & (move_x >= 2) & (move_x <= 7)
    dest_home = dest_lane & (move_x >= 3) & (move_x <= 6)
    cur_lane = (cur_y == target_y) & (cur_x >= 2) & (cur_x <= 7)
    cur_home = cur_lane & (cur_x >= 3) & (cur_x <= 6)
    cur_central = (cur_x >= 3) & (cur_x <= 6) & (cur_y >= 3) & (cur_y <= 6)
    cur_wrong_row = cur_central & (cur_y != target_y)

    center_penalty = jnp.abs(move_x * 2 - 9)
    row_penalty = jnp.abs(move_y - target_y)
    score = (
        known.astype(jnp.int32) * 10000
        + dest_home.astype(jnp.int32) * 5000
        + dest_lane.astype(jnp.int32) * 3500
        + cur_wrong_row.astype(jnp.int32) * 1200
        - cur_home.astype(jnp.int32) * 7000
        - row_penalty.astype(jnp.int32) * 40
        - center_penalty.astype(jnp.int32) * 4
        + (1599 - move_ids)
    )
    masked = jnp.where(legal & known, score, -1000000)
    best = jnp.argmax(masked).astype(jnp.int32)
    fallback = _first_legal(legal, 0)
    has_known_move = jnp.max(masked) > -1000000
    return jnp.where(has_known_move, best + 17, fallback + 17).astype(jnp.int32)


def _public_sorted_enough(obs, memory):
    colors = jnp.arange(4, dtype=jnp.int32)
    positions = obs.positions
    x = positions[:, 0]
    y = positions[:, 1]
    known = memory >= 0
    target_y = memory.astype(jnp.int32) + 3
    known_mismatch = known & (y != target_y)
    lane = (x >= 2) & (x <= 7) & (y >= 3) & (y <= 6)
    row_counts = jnp.sum(y[:, None] == (colors[None, :] + 3), axis=0)
    balanced_rows = jnp.all(row_counts == 4)
    partner_seen = obs.node_features[:NUM_CARDS, 11] > 0
    evidence_count = jnp.sum(known | partner_seen)
    return (
        (evidence_count >= 12)
        & jnp.all(lane)
        & balanced_rows
        & (~jnp.any(known_mismatch))
    )


def _choose_hint(obs, memory):
    reveal_legal = obs.legal_actions[1617:1624]
    place_scores, safe_place = _safe_place_scores(obs, memory)
    reveal_action = _first_legal(reveal_legal, 1617)
    place_action = jnp.argmax(place_scores).astype(jnp.int32) + 1624
    fallback_place = _first_legal(obs.legal_actions[1624:2324], 1624)
    any_reveal = jnp.any(reveal_legal)
    any_safe_place = jnp.any(safe_place)
    return jnp.where(
        any_safe_place,
        place_action,
        jnp.where(any_reveal, reveal_action, jnp.where(jnp.any(obs.legal_actions[1624:2324]), place_action, fallback_place)),
    ).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    next_memory = _known_colors_from_obs(obs, memory, reset)
    substep = obs.node_features[0, 13].astype(jnp.int32)

    can_end = obs.legal_actions[2324]
    sorted_enough = _public_sorted_enough(obs, next_memory)
    should_end = can_end & sorted_enough

    observe_action = _choose_observe(obs, next_memory)
    move_action = _choose_move(obs, next_memory)
    hint_action = _choose_hint(obs, next_memory)

    step_action = jax.lax.switch(
        substep,
        (
            lambda: jnp.where(should_end, jnp.asarray(2324, dtype=jnp.int32), observe_action),
            lambda: observe_action,
            lambda: move_action,
            lambda: hint_action,
        ),
    )
    action = jnp.where(obs.legal_actions[0], jnp.asarray(0, dtype=jnp.int32), step_action)
    trace = {
        "known_count": jnp.sum(next_memory >= 0).astype(jnp.int32),
        "substep": substep,
        "sorted_enough": sorted_enough,
    }
    return action.astype(jnp.int32), next_memory, trace
