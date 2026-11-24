"""JAX-compatible policy for the test_time_simple Overcooked layout."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'counter_signal_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

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

def _first_pos(mask):
    flat = jnp.argmax(mask.reshape(-1).astype(jnp.int32))
    return flat // 5, flat % 5


def _right_room_position(obs):
    gi, gj = _first_pos(obs[:, :, 21] > 0)
    return 4 - gi, 9 - gj


def _left_room_position(obs):
    ri, rj = _first_pos(obs[:, :, 23] > 0)
    return 4 - ri, 2 - rj


def _facing(obs):
    center = obs[2, 2]
    return jnp.argmax(center[1:5].astype(jnp.int32))


def _move_or_face(row, col, target_row, target_col, face_dir):
    row = jnp.asarray(row, dtype=jnp.int32)
    col = jnp.asarray(col, dtype=jnp.int32)
    target_row = jnp.asarray(target_row, dtype=jnp.int32)
    target_col = jnp.asarray(target_col, dtype=jnp.int32)
    face_dir = jnp.asarray(face_dir, dtype=jnp.int32)

    vertical = jnp.where(row < target_row, DOWN, UP)
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    move = jnp.where(row != target_row, vertical, horizontal)
    face_action = jnp.where(
        face_dir == DIR_RIGHT,
        RIGHT,
        jnp.where(face_dir == DIR_DOWN, DOWN, jnp.where(face_dir == DIR_LEFT, LEFT, UP)),
    )
    at_target = (row == target_row) & (col == target_col)
    return jnp.where(at_target, face_action, move)


def _task_action(row, col, facing, target_row, target_col, face_dir):
    face_action = _move_or_face(row, col, target_row, target_col, face_dir)
    at_target = (row == target_row) & (col == target_col)
    return jnp.where(at_target & (facing == face_dir), INTERACT, face_action)


def _right_task(row, col, facing, task, target_ing):
    use_bottom_plate = row > 2
    plate_row = jnp.where(use_bottom_plate, 3, 1)
    ing_row = jnp.where(target_ing == 0, 1, 3)

    a_ing = _task_action(row, col, facing, ing_row, 6, DIR_RIGHT)
    a_pot = _task_action(row, col, facing, 2, 5, DIR_LEFT)
    a_plate = _task_action(row, col, facing, plate_row, 5, DIR_LEFT)
    a_deliver = _task_action(row, col, facing, 2, 6, DIR_RIGHT)
    a_wait_pot = _move_or_face(row, col, 2, 5, DIR_LEFT)
    drop_row = jnp.where(row > 2, 3, 1)
    drop_face = jnp.where(row > 2, DIR_DOWN, DIR_UP)
    a_drop = _task_action(row, col, facing, drop_row, 5, drop_face)

    return jax.lax.switch(
        task,
        (
            lambda: a_ing,
            lambda: a_pot,
            lambda: a_plate,
            lambda: a_wait_pot,
            lambda: a_deliver,
            lambda: STAY,
            lambda: a_drop,
        ),
    )


def _counter_signals(obs, self_row, self_col):
    del self_col
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    global_rows = self_row + rows - 2
    wall_with_marker = (obs[:, :, 20] > 0) & (obs[:, :, 33] > 0)
    top = jnp.any(wall_with_marker & (global_rows == 1))
    bottom = jnp.any(wall_with_marker & (global_rows == 3))
    return top, bottom


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    row, col = _left_room_position(obs)
    facing = _facing(obs)
    recipe0 = jnp.max(obs[:, :, 36])
    recipe1 = jnp.max(obs[:, :, 37])
    target = jnp.where(recipe1 > recipe0, 1, 0).astype(jnp.int32)

    top_signal, bottom_signal = _counter_signals(obs, row, col)
    has_marker = obs[2, 2, 9] > 0
    correct_signal = jnp.where(target == 0, top_signal, bottom_signal)
    wrong_signal = jnp.where(target == 0, bottom_signal, top_signal)

    fetch_row = jnp.where(target == 0, 1, 3)
    fetch_face = jnp.where(target == 0, DIR_UP, DIR_DOWN)
    place_row = jnp.where(target == 0, 1, 3)
    wrong_row = jnp.where(target == 0, 3, 1)

    a_fetch = _task_action(row, col, facing, fetch_row, 2, fetch_face)
    a_place = _task_action(row, col, facing, place_row, 2, DIR_RIGHT)
    a_take_wrong = _task_action(row, col, facing, wrong_row, 2, DIR_RIGHT)

    empty_task = jnp.where(wrong_signal, 2, jnp.where(correct_signal, 3, 0))
    task = jnp.where(has_marker, 1, empty_task)
    action = jax.lax.switch(
        task,
        (
            lambda: a_fetch,
            lambda: a_place,
            lambda: a_take_wrong,
            lambda: STAY,
        ),
    )

    next_memory = jnp.asarray([target], dtype=jnp.int32)
    trace = {
        'target': target,
        'top_signal': top_signal,
        'bottom_signal': bottom_signal,
        'has_marker': has_marker,
    }
    return action, next_memory, trace


def init_agent_1():
    return jnp.zeros((6,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    row, col = _right_room_position(obs)
    facing = _facing(obs)

    top_signal, bottom_signal = _counter_signals(obs, row, col)
    seen_signal = top_signal | bottom_signal

    inv = obs[2, 2, 5:10]
    has_plate = inv[0] > 0
    has_cooked = inv[1] > 0
    has_ing0 = inv[2] > 0
    has_ing1 = inv[3] > 0
    has_raw = has_ing0 | has_ing1

    marker_seen = (memory[5] > 0) | seen_signal
    fallback_started = (memory[4] > 0) | ((memory[1] > 12) & (~marker_seen))
    just_delivered = (memory[2] > 0) & (~has_cooked) & (~has_plate) & (~has_raw)
    prior_target = memory[0]
    fallback_target = jnp.where(just_delivered & fallback_started & (~marker_seen), 1 - prior_target, prior_target)
    signal_target = jnp.where(bottom_signal & ~top_signal, 1, jnp.where(top_signal & ~bottom_signal, 0, fallback_target))
    target = signal_target.astype(jnp.int32)
    known_signal = marker_seen | fallback_started
    has_target_raw = jnp.where(target == 0, has_ing0, has_ing1)

    pot_mask = obs[:, :, 22] > 0
    pot_count0 = jnp.max(jnp.where(pot_mask, obs[:, :, 31], 0))
    pot_count1 = jnp.max(jnp.where(pot_mask, obs[:, :, 32], 0))
    pot_cooked = jnp.max(jnp.where(pot_mask, obs[:, :, 30], 0)) > 0
    pot_timer = jnp.max(jnp.where(pot_mask, obs[:, :, 39], 0))
    pot_total = pot_count0 + pot_count1
    target_count = jnp.where(target == 0, pot_count0, pot_count1)
    matching_pot = target_count == pot_total

    should_add = (pot_total < 3) & (~pot_cooked) & (pot_timer == 0)
    waiting = (pot_timer > 0) | ((pot_total >= 3) & (~pot_cooked))
    ready = pot_cooked
    cooldown = jnp.where(just_delivered, 8, jnp.maximum(memory[3] - 1, 0))
    should_wait_for_signal = ((~known_signal) | (cooldown > 0)) & (~has_plate) & (~has_raw) & (~has_cooked) & (pot_total == 0) & (~pot_cooked)

    task = jnp.asarray(5, dtype=jnp.int32)
    task = jnp.where(has_cooked, 4, task)
    task = jnp.where((~has_cooked) & ready & has_plate & matching_pot, 1, task)
    task = jnp.where((~has_cooked) & ready & (~has_plate) & matching_pot, 2, task)
    task = jnp.where((~has_cooked) & waiting & has_plate, 3, task)
    task = jnp.where((~has_cooked) & waiting & (~has_plate), 2, task)
    task = jnp.where((~has_cooked) & should_add & has_target_raw, 1, task)
    task = jnp.where((~has_cooked) & should_add & (~has_raw), 0, task)
    task = jnp.where((~has_cooked) & has_raw & (~has_target_raw), 6, task)
    task = jnp.where(should_wait_for_signal, 3, task)

    action = _right_task(row, col, facing, task, target)
    next_memory = jnp.asarray(
        [
            target,
            memory[1] + 1,
            has_cooked.astype(jnp.int32),
            cooldown,
            fallback_started.astype(jnp.int32),
            marker_seen.astype(jnp.int32),
        ],
        dtype=jnp.int32,
    )
    trace = {
        'target': target,
        'task': task,
        'top_signal': top_signal,
        'bottom_signal': bottom_signal,
        'pot_total': pot_total,
        'pot_timer': pot_timer,
        'pot_cooked': pot_cooked,
    }
    return action, next_memory, trace
