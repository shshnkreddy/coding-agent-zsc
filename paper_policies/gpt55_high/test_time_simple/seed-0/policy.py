"""Counter-signal finite-state policy for the test_time_simple layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'counter_signal_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

ACT_RIGHT = 0
ACT_DOWN = 1
ACT_LEFT = 2
ACT_UP = 3
ACT_STAY = 4
ACT_INTERACT = 5

DIR_UP = 0
DIR_DOWN = 1
DIR_RIGHT = 2
DIR_LEFT = 3

PHASE_WAIT_SIGNAL = 0
PHASE_WORK = 1
WAIT_STEPS = 7


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _action_for_dir(direction):
    action = jnp.where(direction == DIR_UP, ACT_UP, ACT_RIGHT)
    action = jnp.where(direction == DIR_DOWN, ACT_DOWN, action)
    action = jnp.where(direction == DIR_LEFT, ACT_LEFT, action)
    return action


def _move_to(self_row, self_col, target_row, target_col):
    action = jnp.where(self_row < target_row, ACT_DOWN, ACT_STAY)
    action = jnp.where(self_row > target_row, ACT_UP, action)
    action = jnp.where((self_row == target_row) & (self_col < target_col), ACT_RIGHT, action)
    action = jnp.where((self_row == target_row) & (self_col > target_col), ACT_LEFT, action)
    return action


def _interact_from(obs, self_row, self_col, target_row, target_col, direction):
    at_target = (self_row == target_row) & (self_col == target_col)
    face_action = jnp.where(_facing(obs) == direction, ACT_INTERACT, _action_for_dir(direction))
    return jnp.where(at_target, face_action, _move_to(self_row, self_col, target_row, target_col))


def _channel_any(obs, channel):
    return jnp.sum(obs[:, :, channel]) > 0


def _left_position(obs):
    idx = jnp.argmax(obs[:, :, 23]).astype(jnp.int32)
    local_row = idx // 5
    local_col = idx - local_row * 5
    return (jnp.asarray(4, dtype=jnp.int32) - local_row,
            jnp.asarray(2, dtype=jnp.int32) - local_col)


def _right_position(obs):
    idx = jnp.argmax(obs[:, :, 21]).astype(jnp.int32)
    local_row = idx // 5
    local_col = idx - local_row * 5
    return (jnp.asarray(4, dtype=jnp.int32) - local_row,
            jnp.asarray(9, dtype=jnp.int32) - local_col)


def _go_get_signal_item(obs, self_row, self_col):
    use_top = self_row <= jnp.asarray(2, dtype=jnp.int32)
    target_row = jnp.where(use_top, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    direction = jnp.where(use_top, DIR_UP, DIR_DOWN)
    return _interact_from(obs, self_row, self_col, target_row, jnp.asarray(2, dtype=jnp.int32), direction)


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    self_row, self_col = _left_position(obs)
    recipe_wants_0 = jnp.sum(obs[:, :, 36]) > 0
    marker_present = _channel_any(obs, 33)
    holding_signal_item = obs[2, 2, 9] > 0
    empty_handed = jnp.sum(obs[2, 2, 5:10]) == 0

    place_marker = _interact_from(
        obs, self_row, self_col,
        jnp.asarray(2, dtype=jnp.int32), jnp.asarray(2, dtype=jnp.int32), DIR_RIGHT,
    )
    clear_marker = place_marker
    ready_idle = jnp.where(
        (self_row == 2) & (self_col == 2),
        ACT_STAY,
        _move_to(self_row, self_col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(2, dtype=jnp.int32)),
    )
    fetch_item = _go_get_signal_item(obs, self_row, self_col)

    action_for_zero = jnp.where(marker_present, ready_idle,
                                jnp.where(holding_signal_item, place_marker, fetch_item))
    action_for_one = jnp.where(marker_present & empty_handed, clear_marker,
                               jnp.where(holding_signal_item, ready_idle, fetch_item))
    action = jnp.where(recipe_wants_0, action_for_zero, action_for_one)

    next_memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    trace = {
        'recipe0': recipe_wants_0.astype(jnp.int32),
        'marker': marker_present.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    return jnp.asarray([0, 1, 0, 0], dtype=jnp.int32)


def _pot_value(obs, channel):
    mask = obs[:, :, 22] > 0
    return jnp.sum(jnp.where(mask, obs[:, :, channel], 0))


def _get_plate(obs, self_row, self_col, target):
    plate_row = jnp.where(target == 0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    return _interact_from(obs, self_row, self_col, plate_row, jnp.asarray(5, dtype=jnp.int32), DIR_LEFT)


def _get_ingredient(obs, self_row, self_col, target):
    supply_row = jnp.where(target == 0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    return _interact_from(obs, self_row, self_col, supply_row, jnp.asarray(6, dtype=jnp.int32), DIR_RIGHT)


def _go_pot(obs, self_row, self_col):
    return _interact_from(
        obs, self_row, self_col,
        jnp.asarray(2, dtype=jnp.int32), jnp.asarray(5, dtype=jnp.int32), DIR_LEFT,
    )


def _go_delivery(obs, self_row, self_col):
    return _interact_from(
        obs, self_row, self_col,
        jnp.asarray(2, dtype=jnp.int32), jnp.asarray(6, dtype=jnp.int32), DIR_RIGHT,
    )


def _wait_at_reader(self_row, self_col):
    at_reader = (self_row == 2) & (self_col == 5)
    return jnp.where(at_reader, ACT_STAY,
                     _move_to(self_row, self_col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(5, dtype=jnp.int32)))


def agent_1_act(obs, reset, memory, rng):
    del rng
    base_memory = jnp.where(reset, init_agent_1(), memory)
    phase = base_memory[0]
    target = base_memory[1]
    wait_count = base_memory[2]
    mature_waiting = base_memory[3]

    self_row, self_col = _right_position(obs)
    marker_visible = _channel_any(obs, 33)
    can_read_signal = self_col == 5
    observed_target = jnp.where(marker_visible, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    target = jnp.where((phase == PHASE_WAIT_SIGNAL) & can_read_signal, observed_target, target)

    inv_plate = obs[2, 2, 5] > 0
    inv_cooked = obs[2, 2, 6] > 0
    inv0 = obs[2, 2, 7] > 0
    inv1 = obs[2, 2, 8] > 0
    empty_handed = jnp.sum(obs[2, 2, 5:10]) == 0
    holding_dish = inv_plate & inv_cooked
    holding_target_raw = jnp.where(target == 0, inv0, inv1)

    pot_i0 = _pot_value(obs, 31)
    pot_i1 = _pot_value(obs, 32)
    pot_i2 = _pot_value(obs, 33)
    pot_total = pot_i0 + pot_i1 + pot_i2
    pot_cooked = _pot_value(obs, 30) > 0
    pot_timer = _pot_value(obs, 39)
    pot_busy = (pot_timer > 0) | ((pot_total >= 3) & (~pot_cooked))

    add_raw = _go_pot(obs, self_row, self_col)
    get_ing = _get_ingredient(obs, self_row, self_col, target)
    get_plate = _get_plate(obs, self_row, self_col, target)
    pickup_dish = _go_pot(obs, self_row, self_col)
    deliver = _go_delivery(obs, self_row, self_col)
    wait_with_plate = jnp.where(
        (self_row == 2) & (self_col == 5),
        ACT_STAY,
        _move_to(self_row, self_col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(5, dtype=jnp.int32)),
    )

    work_action = jnp.where(holding_target_raw, add_raw, jnp.where(empty_handed, get_ing, ACT_STAY))
    cooking_action = jnp.where(inv_plate & (~inv_cooked), wait_with_plate,
                               jnp.where(empty_handed, get_plate, ACT_STAY))
    ready_action = jnp.where(inv_plate & (~inv_cooked), pickup_dish,
                             jnp.where(empty_handed, get_plate, ACT_STAY))
    work_action = jnp.where(pot_busy, cooking_action, work_action)
    work_action = jnp.where(pot_cooked, ready_action, work_action)
    work_action = jnp.where(holding_dish, deliver, work_action)

    wait_action = _wait_at_reader(self_row, self_col)
    action = jnp.where(phase == PHASE_WAIT_SIGNAL, wait_action, work_action)

    next_wait = jnp.where((phase == PHASE_WAIT_SIGNAL) & (self_row == 2) & (self_col == 5),
                          wait_count + 1, wait_count)
    absence_threshold = jnp.where(mature_waiting > 0, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(WAIT_STEPS, dtype=jnp.int32))
    positive_signal_ready = (phase == PHASE_WAIT_SIGNAL) & can_read_signal & marker_visible
    wait_done = positive_signal_ready | ((phase == PHASE_WAIT_SIGNAL) & (next_wait >= absence_threshold))
    facing_right = _facing(obs) == DIR_RIGHT
    delivered_now = (
        (phase == PHASE_WORK) & holding_dish
        & (self_row == 2) & (self_col == 6) & facing_right
    )
    next_phase = jnp.where(wait_done, PHASE_WORK, phase)
    next_phase = jnp.where(delivered_now, PHASE_WAIT_SIGNAL, next_phase)
    next_wait = jnp.where((next_phase == PHASE_WORK) | delivered_now, jnp.asarray(0, dtype=jnp.int32), next_wait)
    next_mature = jnp.where(wait_done | (phase == PHASE_WORK), jnp.asarray(1, dtype=jnp.int32), mature_waiting)
    next_memory = jnp.asarray([next_phase, target, next_wait, next_mature], dtype=jnp.int32)
    next_memory = jnp.where(reset, init_agent_1(), next_memory)

    trace = {
        'phase': phase,
        'target': target,
        'wait': wait_count,
        'mature': mature_waiting,
        'pot_total': pot_total.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
