"""XP-oriented policy for the test_time_wide Overcooked layout."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'signal_cook_candidate',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

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

UNKNOWN = 0
TARGET_0 = 1
TARGET_1 = 2

PHASE_SIGNAL = 0
PHASE_WORK = 1

FLOOR_POSITIONS = [
        [1, 1], [1, 2], [1, 3], [1, 4],
        [2, 1], [2, 2], [2, 3], [2, 4],
        [4, 1], [4, 2], [4, 3], [4, 4],
        [5, 1], [5, 2], [5, 3], [5, 4],
]

STATIC_MAP = [
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
]


def _top_memory0():
    return jnp.asarray([0, 0, 2], dtype=jnp.int32)


def _bottom_memory0():
    return jnp.asarray([0], dtype=jnp.int32)


def _static_crop_for_position(pos):
    static_map = jnp.asarray(STATIC_MAP, dtype=jnp.int32)
    rows = pos[0] + jnp.arange(5, dtype=jnp.int32) - 2
    cols = pos[1] + jnp.arange(5, dtype=jnp.int32) - 2
    rr = jnp.clip(rows, 0, 6)
    cc = jnp.clip(cols, 0, 5)
    crop = static_map[rr[:, None], cc[None, :], :]
    valid = ((rows[:, None] >= 0) & (rows[:, None] < 7) &
             (cols[None, :] >= 0) & (cols[None, :] < 6))
    return jnp.where(valid[:, :, None], crop, 0)

def _global_pos(obs):
    floor_positions = jnp.asarray(FLOOR_POSITIONS, dtype=jnp.int32)
    static_crops = jax.vmap(_static_crop_for_position)(floor_positions)
    seen = obs[:, :, 22:32]
    scores = jnp.sum(jnp.equal(static_crops, seen[None, :, :, :]), axis=(1, 2, 3))
    return floor_positions[jnp.argmax(scores)]


def _facing_dir(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _action_to_dir(action):
    return jnp.asarray(
        jnp.where(action == RIGHT, DIR_RIGHT,
                  jnp.where(action == DOWN, DIR_DOWN,
                            jnp.where(action == LEFT, DIR_LEFT, DIR_UP))),
        dtype=jnp.int32,
    )


def _move_towards(pos, target):
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    return jnp.asarray(
        jnp.where(dr > 0, DOWN,
                  jnp.where(dr < 0, UP,
                            jnp.where(dc > 0, RIGHT,
                                      jnp.where(dc < 0, LEFT, STAY)))),
        dtype=jnp.int32,
    )


def _move_towards_hfirst(pos, target):
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    return jnp.asarray(
        jnp.where(dc > 0, RIGHT,
                  jnp.where(dc < 0, LEFT,
                            jnp.where(dr > 0, DOWN,
                                      jnp.where(dr < 0, UP, STAY)))),
        dtype=jnp.int32,
    )


def _go_face_or_interact(obs, pos, target_cell, face_action):
    move = _move_towards(pos, target_cell)
    at_target = jnp.all(pos == target_cell)
    desired_dir = _action_to_dir(face_action)
    facing_ok = _facing_dir(obs) == desired_dir
    return jnp.asarray(
        jnp.where(at_target,
                  jnp.where(facing_ok, INTERACT, face_action),
                  move),
        dtype=jnp.int32,
    )


def _go_face_or_stay(obs, pos, target_cell, face_action):
    move = _move_towards(pos, target_cell)
    at_target = jnp.all(pos == target_cell)
    desired_dir = _action_to_dir(face_action)
    facing_ok = _facing_dir(obs) == desired_dir
    return jnp.asarray(
        jnp.where(at_target,
                  jnp.where(facing_ok, STAY, face_action),
                  move),
        dtype=jnp.int32,
    )


def _go_face_or_stay_hfirst(obs, pos, target_cell, face_action):
    move = _move_towards_hfirst(pos, target_cell)
    at_target = jnp.all(pos == target_cell)
    desired_dir = _action_to_dir(face_action)
    facing_ok = _facing_dir(obs) == desired_dir
    return jnp.asarray(
        jnp.where(at_target,
                  jnp.where(facing_ok, STAY, face_action),
                  move),
        dtype=jnp.int32,
    )


def _local_cell_for_global(pos, cell):
    return cell - pos + jnp.asarray([2, 2], dtype=jnp.int32)


def _cell_value(obs, pos, cell, channel):
    loc = _local_cell_for_global(pos, cell)
    visible = ((loc[0] >= 0) & (loc[0] < 5) & (loc[1] >= 0) & (loc[1] < 5))
    rr = jnp.clip(loc[0], 0, 4)
    cc = jnp.clip(loc[1], 0, 4)
    return jnp.where(visible, obs[rr, cc, channel], 0)


def _held_kind(obs):
    center = obs[2, 2, :]
    held_plate = center[5] > 0
    held_cooked = center[6] > 0
    held_0 = center[7] > 0
    held_1 = center[8] > 0
    return held_plate, held_cooked, held_0, held_1


def _partner_signal(obs, pos, old_target):
    flat = jnp.reshape(obs[:, :, 11], (25,))
    idx = jnp.argmax(flat)
    visible = jnp.max(flat) > 0
    local = jnp.asarray([idx // 5, idx % 5], dtype=jnp.int32)
    partner = pos + local - jnp.asarray([2, 2], dtype=jnp.int32)
    partner_up = obs[local[0], local[1], 12] > 0
    deliberate = visible & partner_up
    is_left_signal = deliberate & jnp.all(partner == jnp.asarray([4, 1], dtype=jnp.int32))
    is_right_signal = deliberate & jnp.all(partner == jnp.asarray([4, 3], dtype=jnp.int32))
    return jnp.asarray(
        jnp.where(is_left_signal, TARGET_0,
                  jnp.where(is_right_signal, TARGET_1, old_target)),
        dtype=jnp.int32,
    )


def _recipe_target_from_obs(obs, old_target):
    r0 = jnp.sum(obs[:, :, 40])
    r1 = jnp.sum(obs[:, :, 41])
    seen = (r0 + r1) > 0
    decoded = jnp.where(r1 > r0, TARGET_1, TARGET_0)
    return jnp.asarray(jnp.where(seen, decoded, old_target), dtype=jnp.int32)


def init_agent_0():
    return _top_memory0()


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _top_memory0(), memory)
    pos = _global_pos(obs)

    old_phase = memory[0]
    old_target = memory[1]
    cooldown = memory[2]
    raw_signal_target = _partner_signal(obs, pos, old_target)
    signal_target = jnp.where(cooldown > 0, old_target, raw_signal_target)
    locked = (old_phase == PHASE_WORK) & (old_target != UNKNOWN)
    target = jnp.where(locked, old_target, signal_target)
    phase = jnp.where(target == UNKNOWN, PHASE_SIGNAL,
                      jnp.where(old_phase == PHASE_SIGNAL, PHASE_WORK, old_phase))

    held_plate, held_cooked, held_0, held_1 = _held_kind(obs)
    holding_any = held_plate | held_cooked | held_0 | held_1

    is_target_1 = target == TARGET_1
    pot_cell = jnp.asarray([3, 1], dtype=jnp.int32)
    pot_stand = jnp.asarray([2, 1], dtype=jnp.int32)
    supply_stand = jnp.where(is_target_1,
                             jnp.asarray([2, 1], dtype=jnp.int32),
                             jnp.asarray([1, 1], dtype=jnp.int32))
    supply_face = LEFT

    pot_count_0 = _cell_value(obs, pos, pot_cell, 34)
    pot_count_1 = _cell_value(obs, pos, pot_cell, 35)
    pot_timer = _cell_value(obs, pos, pot_cell, 44)
    pot_cooked = _cell_value(obs, pos, pot_cell, 33) > 0
    pot_count = jnp.where(is_target_1, pot_count_1, pot_count_0)
    pot_full_or_busy = (pot_count >= 3) | (pot_timer > 0) | pot_cooked

    holding_target_raw = jnp.where(is_target_1, held_1, held_0)

    act_watch = _move_towards(pos, jnp.asarray([2, 2], dtype=jnp.int32))
    act_get_raw = _go_face_or_interact(obs, pos, supply_stand, supply_face)
    act_add_raw = _go_face_or_interact(obs, pos, pot_stand, DOWN)
    act_get_plate = _go_face_or_interact(obs, pos, jnp.asarray([1, 3], dtype=jnp.int32), UP)
    act_scoop = _go_face_or_interact(obs, pos, pot_stand, DOWN)
    act_deliver = _go_face_or_interact(obs, pos, jnp.asarray([1, 2], dtype=jnp.int32), UP)
    act_wait_pot = _go_face_or_stay(obs, pos, pot_stand, DOWN)

    work_action = jnp.where(
        held_cooked,
        act_deliver,
        jnp.where(
            held_plate,
            jnp.where(pot_cooked, act_scoop, act_wait_pot),
            jnp.where(
                holding_target_raw,
                act_add_raw,
                jnp.where(
                    holding_any,
                    STAY,
                    jnp.where(
                        pot_cooked,
                        act_get_plate,
                        jnp.where(pot_full_or_busy, act_get_plate, act_get_raw),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    action = jnp.where(phase == PHASE_SIGNAL, act_watch, work_action).astype(jnp.int32)
    delivering_now = (action == INTERACT) & held_cooked & jnp.all(pos == jnp.asarray([1, 2], dtype=jnp.int32)) & (_facing_dir(obs) == DIR_UP)
    next_phase = jnp.where(delivering_now, PHASE_SIGNAL, phase)
    next_target = jnp.where(delivering_now, UNKNOWN, target)
    next_cooldown = jnp.maximum(cooldown - 1, 0)
    next_memory = jnp.stack([next_phase, next_target, next_cooldown]).astype(jnp.int32)
    trace = {
        'phase': next_phase,
        'target': next_target,
        'cooldown': next_cooldown,
        'row': pos[0],
        'col': pos[1],
    }
    return action, next_memory, trace


def init_agent_1():
    return _bottom_memory0()


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _bottom_memory0(), memory)
    pos = _global_pos(obs)
    target = _recipe_target_from_obs(obs, memory[0])
    signal_cell = jnp.where(target == TARGET_1,
                            jnp.asarray([4, 3], dtype=jnp.int32),
                            jnp.asarray([4, 1], dtype=jnp.int32))
    action = _go_face_or_stay_hfirst(obs, pos, signal_cell, UP)
    next_memory = jnp.expand_dims(target, 0).astype(jnp.int32)
    trace = {
        'target': target,
        'row': pos[0],
        'col': pos[1],
    }
    return action, next_memory, trace
