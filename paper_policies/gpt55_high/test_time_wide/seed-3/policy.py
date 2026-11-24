"""Deterministic split-role policy for the wide Overcooked layout."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'visible_signal_service_loop', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

DIR_UP = 0
DIR_DOWN = 1
DIR_RIGHT = 2
DIR_LEFT = 3

FLOOR_POS_DATA = (
    (
        [1, 1], [1, 2], [1, 3], [1, 4],
        [2, 1], [2, 2], [2, 3], [2, 4],
        [4, 1], [4, 2], [4, 3], [4, 4],
        [5, 1], [5, 2], [5, 3], [5, 4],
    )
)

STATIC_DATA = (
    (
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    )
)


def _expected_crops():
    floor_pos = jnp.asarray(FLOOR_POS_DATA, dtype=jnp.int32)
    static = jnp.asarray(STATIC_DATA, dtype=jnp.int32)
    rows = floor_pos[:, 0]
    cols = floor_pos[:, 1]
    rr = rows[:, None, None] + jnp.arange(5, dtype=jnp.int32)[None, :, None] - 2
    cc = cols[:, None, None] + jnp.arange(5, dtype=jnp.int32)[None, None, :] - 2
    in_bounds = (rr >= 0) & (rr < 7) & (cc >= 0) & (cc < 6)
    rr_clip = jnp.clip(rr, 0, 6)
    cc_clip = jnp.clip(cc, 0, 5)
    crops = static[rr_clip, cc_clip, :]
    return jnp.where(in_bounds[..., None], crops, 0)


def _infer_pos(obs):
    floor_pos = jnp.asarray(FLOOR_POS_DATA, dtype=jnp.int32)
    expected_crops = _expected_crops()
    static_obs = obs[:, :, 22:32]
    diffs = jnp.sum(jnp.abs(expected_crops - static_obs[None, :, :, :]), axis=(1, 2, 3))
    idx = jnp.argmin(diffs)
    return floor_pos[idx, 0], floor_pos[idx, 1]


def _facing(obs):
    face = obs[2, 2, 1:5]
    obs_dir = jnp.argmax(face).astype(jnp.int32)
    return jnp.asarray([DIR_UP, DIR_DOWN, DIR_RIGHT, DIR_LEFT], dtype=jnp.int32)[obs_dir]


def _turn_action(direction):
    return jax.lax.switch(direction, (lambda: UP, lambda: DOWN, lambda: RIGHT, lambda: LEFT))


def _move_to(row, col, target_row, target_col, target_dir, facing, do_interact):
    row_diff = target_row - row
    col_diff = target_col - col
    move = jnp.where(row_diff > 0, DOWN, jnp.where(row_diff < 0, UP, jnp.where(col_diff > 0, RIGHT, jnp.where(col_diff < 0, LEFT, STAY))))
    at_target = (row == target_row) & (col == target_col)
    turn = _turn_action(target_dir)
    use = jnp.where(do_interact, INTERACT, STAY)
    return jnp.where(at_target, jnp.where(facing == target_dir, use, turn), move)


def _cell(obs, self_row, self_col, global_row, global_col, channel):
    li = global_row - self_row + 2
    lj = global_col - self_col + 2
    in_view = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    value = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), channel]
    return jnp.where(in_view, value, jnp.asarray(0, dtype=jnp.int32))


def _partner_signal(obs, self_row, self_col):
    partner = obs[:, :, 11]
    visible = jnp.sum(partner) > 0
    flat = jnp.argmax(jnp.reshape(partner, (25,))).astype(jnp.int32)
    local_row = flat // 5
    local_col = flat - local_row * 5
    partner_row = self_row + local_row - 2
    partner_col = self_col + local_col - 2
    partner_up = obs[local_row, local_col, 12] > 0
    signal_visible = visible & partner_up & (partner_row == 4) & ((partner_col == 1) | (partner_col == 3))
    signal_ingredient = jnp.where(partner_col <= 2, 0, 1).astype(jnp.int32)
    return signal_ingredient, signal_visible


def _top_policy(obs, memory):
    row, col = _infer_pos(obs)
    facing = _facing(obs)
    old_ing = memory[0]
    old_valid = memory[1]
    wait = memory[2]
    signal_ingredient, signal_seen = _partner_signal(obs, row, col)
    accept_signal = signal_seen & (wait <= 0)
    ingredient = jnp.where(accept_signal, signal_ingredient, old_ing)
    valid = jnp.where(accept_signal, jnp.asarray(1, dtype=jnp.int32), old_valid)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ing0 = obs[2, 2, 7] > 0
    held_ing1 = obs[2, 2, 8] > 0
    held_raw = held_ing0 | held_ing1

    pot_col = jnp.where(ingredient == 0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    pot_count = _cell(obs, row, col, 3, pot_col, 34 + ingredient)
    pot_other = _cell(obs, row, col, 3, pot_col, 35 - ingredient)
    pot_cooked = _cell(obs, row, col, 3, pot_col, 33) > 0
    pot_timer = _cell(obs, row, col, 3, pot_col, 44)
    pot_has_food = (pot_count + pot_other) > 0
    pot_ready = pot_cooked & pot_has_food
    pot_needs_raw = (pot_count < 3) & (pot_other == 0) & (~pot_cooked) & (pot_timer == 0)

    other_ingredient = 1 - ingredient
    other_pot_col = jnp.where(other_ingredient == 0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    other_count = _cell(obs, row, col, 3, other_pot_col, 34 + other_ingredient)
    other_wrong = _cell(obs, row, col, 3, other_pot_col, 35 - other_ingredient)
    other_cooked = _cell(obs, row, col, 3, other_pot_col, 33) > 0
    other_timer = _cell(obs, row, col, 3, other_pot_col, 44)
    other_needs_raw = (other_count < 3) & (other_wrong == 0) & (~other_cooked) & (other_timer == 0)
    prefill_other = (pot_timer > 10) & other_needs_raw

    observe_action = _move_to(row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(2, dtype=jnp.int32), DIR_DOWN, facing, False)
    deliver_action = _move_to(row, col, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(2, dtype=jnp.int32), DIR_UP, facing, True)
    plate_action = _move_to(row, col, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32), DIR_UP, facing, True)
    pot_action = _move_to(row, col, jnp.asarray(2, dtype=jnp.int32), pot_col, DIR_DOWN, facing, True)
    raw_ingredient = jnp.where(held_ing0, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    raw_pot_col = jnp.where(raw_ingredient == 0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    raw_pot_action = _move_to(row, col, jnp.asarray(2, dtype=jnp.int32), raw_pot_col, DIR_DOWN, facing, True)
    fetch0_action = _move_to(row, col, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32), DIR_LEFT, facing, True)
    fetch1_action = _move_to(row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(4, dtype=jnp.int32), DIR_RIGHT, facing, True)
    fetch_action = jnp.where(ingredient == 0, fetch0_action, fetch1_action)
    fetch_other_action = jnp.where(other_ingredient == 0, fetch0_action, fetch1_action)

    empty_action = jnp.where(
        valid == 0,
        observe_action,
        jnp.where(
            prefill_other,
            fetch_other_action,
            jnp.where(pot_ready | (pot_count >= 3) | (pot_timer > 0), plate_action, jnp.where(pot_needs_raw, fetch_action, observe_action)),
        ),
    )
    action = jnp.where(held_cooked, deliver_action, jnp.where(held_plate, pot_action, jnp.where(held_raw, raw_pot_action, empty_action)))

    is_delivery_interact = held_cooked & (row == 1) & (col == 2) & (facing == DIR_UP) & (action == INTERACT)
    next_wait = jnp.where(
        is_delivery_interact,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.where((valid == 0) & (wait > 0), wait - 1, jnp.asarray(0, dtype=jnp.int32)),
    )
    next_valid = jnp.where(is_delivery_interact, jnp.asarray(0, dtype=jnp.int32), valid)
    next_memory = jnp.asarray([ingredient, next_valid, next_wait], dtype=jnp.int32)
    trace = {
        'row': row,
        'col': col,
        'ingredient': ingredient,
        'valid': next_valid,
        'pot_count': pot_count,
        'pot_timer': pot_timer,
        'action': action,
    }
    return action, next_memory, trace


def _bottom_policy(obs, memory):
    row, col = _infer_pos(obs)
    facing = _facing(obs)
    recipe0 = _cell(obs, row, col, 6, 2, 40)
    recipe1 = _cell(obs, row, col, 6, 2, 41)
    old_ing = memory[0]
    ingredient = jnp.where(recipe0 > recipe1, jnp.asarray(0, dtype=jnp.int32), jnp.where(recipe1 > recipe0, jnp.asarray(1, dtype=jnp.int32), old_ing))
    signal_col = jnp.where(ingredient == 0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    action = _move_to(row, col, jnp.asarray(4, dtype=jnp.int32), signal_col, DIR_UP, facing, False)
    next_memory = jnp.asarray([ingredient], dtype=jnp.int32)
    trace = {'row': row, 'col': col, 'ingredient': ingredient, 'action': action}
    return action, next_memory, trace


def init_agent_0():
    return jnp.asarray([0, 0, 2], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    start = init_agent_0()
    memory = jnp.where(reset, start, memory)
    return _top_policy(obs, memory)


def init_agent_1():
    return jnp.asarray([0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    start = init_agent_1()
    memory = jnp.where(reset, start, memory)
    return _bottom_policy(obs, memory)
