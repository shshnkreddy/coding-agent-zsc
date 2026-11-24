"""Convention-robust Yokai policy for the local observation interface."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'canonical_clusterer', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
NUM_HINTS = 4
END_GAME = 1067

INITIAL_POSITION_ROWS = ((3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5))


def init_agent():
    return {
        "known": jnp.zeros((NUM_CARDS, NUM_COLORS), dtype=jnp.int8),
        "seat0": jnp.asarray(False),
    }


def _card_ids():
    return jnp.arange(NUM_CARDS, dtype=jnp.int32)


def _move_constants():
    offsets = jnp.arange(NUM_CARDS * 81, dtype=jnp.int32)
    rem = offsets % 81
    return offsets // 81, rem // 9, rem % 9, 10 + offsets


def _place_constants():
    offsets = jnp.arange(NUM_HINTS * 81, dtype=jnp.int32)
    rem = offsets % 81
    return offsets // 81, rem // 9, rem % 9, 743 + offsets


def _initial_positions():
    return jnp.asarray(INITIAL_POSITION_ROWS, dtype=jnp.int32)


def _one_hot3(index):
    clipped = jnp.clip(index, 0, NUM_COLORS - 1)
    return jnp.eye(NUM_COLORS, dtype=jnp.int8)[clipped]


def _masked_argmin(values, mask):
    large = jnp.asarray(1_000_000_000, dtype=jnp.int32)
    return jnp.argmin(jnp.where(mask, values, large)).astype(jnp.int32)


def _first_legal_action(legal_actions):
    return jnp.argmax(legal_actions).astype(jnp.int32)


def _update_memory(obs, reset, memory):
    active = obs.node_features[0, 10] == 1
    base_known = jnp.where(reset, jnp.zeros_like(memory["known"]), memory["known"])
    seat0 = jnp.where(reset, active, memory["seat0"])

    visible = obs.node_features[:NUM_CARDS, :NUM_COLORS].astype(jnp.int8)
    visible_mask = jnp.sum(visible, axis=1, keepdims=True) > 0

    positions = obs.positions.astype(jnp.int32)
    moved = jnp.any(positions != _initial_positions(), axis=1, keepdims=True)
    in_signal_column = (positions[:, 0:1] >= 3) & (positions[:, 0:1] <= 5)
    signal_color = positions[:, 0] - 3
    signal = _one_hot3(signal_color)
    signal = jnp.where(moved & in_signal_column, signal, jnp.zeros_like(signal))
    signal_mask = jnp.sum(signal, axis=1, keepdims=True) > 0

    empty = jnp.sum(base_known, axis=1, keepdims=True) == 0
    known = jnp.where(signal_mask & empty, signal, base_known)
    known = jnp.where(visible_mask, visible, known).astype(jnp.int8)
    return {"known": known, "seat0": seat0}


def _effective_colors(known):
    card_known = jnp.sum(known, axis=1) > 0
    counts = jnp.sum(known, axis=0).astype(jnp.int32)
    unknown_count = NUM_CARDS - jnp.sum(card_known.astype(jnp.int32))
    remaining = 3 - counts
    missing_color = jnp.argmax(remaining).astype(jnp.int32)
    inferred = _one_hot3(missing_color)
    can_infer_last = (unknown_count == 1) & jnp.all(remaining >= 0) & (jnp.sum(remaining) == 1)
    effective = jnp.where((~card_known[:, None]) & can_infer_last, inferred, known)
    effective = effective.astype(jnp.int8)
    eff_known = jnp.sum(effective, axis=1) > 0
    eff_counts = jnp.sum(effective, axis=0).astype(jnp.int32)
    complete = jnp.all(eff_known) & jnp.all(eff_counts == 3)
    return effective, eff_known, complete


def _targets(effective):
    colors = jnp.argmax(effective, axis=1).astype(jnp.int32)
    same = colors[:, None] == colors[None, :]
    card_ids = _card_ids()
    lower_id = card_ids[None, :] < card_ids[:, None]
    rank = jnp.sum((same & lower_id).astype(jnp.int32), axis=1)
    return jnp.stack([3 + colors, 3 + rank], axis=1).astype(jnp.int32)


def _colors_connected(positions, effective, complete):
    colors = jnp.argmax(effective, axis=1).astype(jnp.int32)
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=2) == 1

    ok = jnp.asarray(True)
    for color in range(NUM_COLORS):
        mask = colors == color
        color_edges = adjacent & mask[:, None] & mask[None, :]
        edge_count = jnp.sum(color_edges.astype(jnp.int32)) // 2
        ok = ok & (jnp.sum(mask.astype(jnp.int32)) == 3) & (edge_count >= 2)
    return complete & ok


def _choose_observe(obs, known, seat0):
    legal = obs.legal_actions[1:10]
    unknown = jnp.sum(known, axis=1) == 0
    card_ids = _card_ids()
    priority = jnp.where(seat0, card_ids, 8 - card_ids)
    unknown_choice = legal & unknown
    idx_unknown = _masked_argmin(priority, unknown_choice)
    idx_any = _masked_argmin(priority, legal)
    has_unknown = jnp.any(unknown_choice)
    return (1 + jnp.where(has_unknown, idx_unknown, idx_any)).astype(jnp.int32)


def _choose_move(obs, effective, eff_known):
    positions = obs.positions.astype(jnp.int32)
    targets = _targets(effective)
    before_card = jnp.sum(jnp.abs(positions - targets), axis=1).astype(jnp.int32)
    known_i = eff_known.astype(jnp.int32)
    total_before = jnp.sum(before_card * known_i)

    move_cards, move_x, move_y, move_actions = _move_constants()
    card = move_cards
    card_ids = _card_ids()
    destinations = jnp.stack([move_x, move_y], axis=1)
    moved_positions = jnp.where(
        card_ids[None, :, None] == card[:, None, None],
        destinations[:, None, :],
        positions[None, :, :],
    )

    colors = jnp.argmax(effective, axis=1).astype(jnp.int32)
    delta = jnp.abs(moved_positions[:, :, None, :] - moved_positions[:, None, :, :])
    adjacent = jnp.sum(delta, axis=3) == 1
    deficit = jnp.zeros((NUM_CARDS * 81,), dtype=jnp.int32)
    total_edges = jnp.zeros((NUM_CARDS * 81,), dtype=jnp.int32)
    for color in range(NUM_COLORS):
        mask = (colors == color) & eff_known
        count = jnp.sum(mask.astype(jnp.int32))
        desired = jnp.maximum(jnp.minimum(count - 1, 2), 0)
        edges = jnp.sum((adjacent & mask[None, :, None] & mask[None, None, :]).astype(jnp.int32), axis=(1, 2)) // 2
        deficit = deficit + jnp.maximum(desired - edges, 0)
        total_edges = total_edges + edges

    target_for_move = targets[card]
    new_dist = jnp.abs(move_x - target_for_move[:, 0]) + jnp.abs(move_y - target_for_move[:, 1])
    total_after = total_before - before_card[card] * known_i[card] + new_dist * known_i[card]
    moving_unknown_penalty = (1 - known_i[card]) * 5000
    score = deficit * 2000 - total_edges * 50 + total_after * 20 + new_dist + moving_unknown_penalty

    legal = obs.legal_actions[move_actions]
    idx = _masked_argmin(score.astype(jnp.int32), legal)
    return move_actions[idx].astype(jnp.int32)


def _choose_hint(obs, effective, eff_known, connected):
    reveal_legal = obs.legal_actions[739:743]
    reveal_idx = _masked_argmin(jnp.arange(NUM_HINTS, dtype=jnp.int32), reveal_legal)
    reveal_action = (739 + reveal_idx).astype(jnp.int32)

    positions = obs.positions.astype(jnp.int32)
    targets = _targets(effective)
    colors = jnp.argmax(effective, axis=1).astype(jnp.int32)
    at_target = jnp.all(positions == targets, axis=1)

    place_hints, place_x, place_y, place_actions = _place_constants()
    px = place_x[:, None]
    py = place_y[:, None]
    card_here = (positions[:, 0] == px) & (positions[:, 1] == py)
    card_at_place = jnp.argmax(card_here.astype(jnp.int32), axis=1)
    has_card = jnp.any(card_here, axis=1)
    hint_sets = obs.node_features[9:13, 3:6].astype(jnp.int32)
    hint_has_color = hint_sets[place_hints, colors[card_at_place]] > 0
    card_known = eff_known[card_at_place]
    card_ready = at_target[card_at_place]

    place_legal = obs.legal_actions[place_actions]
    good = has_card & card_known & hint_has_color
    good_ready = place_legal & good & card_ready & connected
    score = (
        (1 - good.astype(jnp.int32)) * 10000
        + (1 - card_ready.astype(jnp.int32)) * 100
        + place_hints * 10
        + card_at_place
    )
    place_idx = _masked_argmin(score.astype(jnp.int32), place_legal)
    place_action = place_actions[place_idx].astype(jnp.int32)
    good_ready_idx = _masked_argmin(score.astype(jnp.int32), good_ready)
    good_ready_action = place_actions[good_ready_idx].astype(jnp.int32)

    reveal_or_forced_place = jnp.where(jnp.any(reveal_legal), reveal_action, place_action)
    return jnp.where(jnp.any(good_ready), good_ready_action, reveal_or_forced_place).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    next_memory = _update_memory(obs, reset, memory)
    known = next_memory["known"]
    effective, eff_known, complete = _effective_colors(known)

    substep = obs.node_features[0, 11].astype(jnp.int32)
    positions = obs.positions.astype(jnp.int32)
    connected = _colors_connected(positions, effective, complete)

    def step0(_):
        observe = _choose_observe(obs, known, next_memory["seat0"])
        end = jnp.asarray(END_GAME, dtype=jnp.int32)
        return jnp.where(obs.legal_actions[END_GAME] & connected, end, observe)

    def step1(_):
        return _choose_observe(obs, known, next_memory["seat0"])

    def step2(_):
        return _choose_move(obs, effective, eff_known)

    def step3(_):
        return _choose_hint(obs, effective, eff_known, connected)

    proposed = jax.lax.switch(substep, (step0, step1, step2, step3), operand=None)
    fallback = _first_legal_action(obs.legal_actions)
    action = jnp.where(obs.legal_actions[proposed], proposed, fallback).astype(jnp.int32)
    trace = {
        "known_count": jnp.sum(jnp.sum(known, axis=1) > 0).astype(jnp.int32),
        "complete": complete,
        "connected": connected,
        "seat0": next_memory["seat0"],
    }
    return action, next_memory, trace
