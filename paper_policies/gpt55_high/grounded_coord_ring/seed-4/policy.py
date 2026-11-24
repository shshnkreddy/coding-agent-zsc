"""Hand-built policy for the grounded_coord_ring Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'solo_center_cook_v1',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
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

UNKNOWN_TARGET = 2
IDX_DELIVER = 4 * 9 + 5


def _face_action(direction):
    return jnp.asarray(
        jnp.where(
            direction == DIR_RIGHT,
            RIGHT,
            jnp.where(direction == DIR_DOWN, DOWN, jnp.where(direction == DIR_LEFT, LEFT, UP)),
        ),
        dtype=jnp.int32,
    )


def _facing(obs):
    return jnp.asarray(jnp.argmax(obs[2, 2, 1:5]), dtype=jnp.int32)


def _locate_a1(obs):
    pot_flat = jnp.argmax(jnp.reshape(obs[:, :, 22], (25,)))
    pot_local_row = pot_flat // 5
    pot_local_col = pot_flat - pot_local_row * 5
    row = 6 - pot_local_row
    col = 6 - pot_local_col
    return jnp.asarray(row, dtype=jnp.int32), jnp.asarray(col, dtype=jnp.int32)


def _valid_a1(row, col):
    inside_box = (row >= 3) & (row <= 5) & (col >= 3) & (col <= 5)
    not_pot = (row != 4) | (col != 4)
    return inside_box & not_pot


def _ring_index(row, col):
    idx = jnp.where((row == 3) & (col == 3), 0, 0)
    idx = jnp.where((row == 3) & (col == 4), 1, idx)
    idx = jnp.where((row == 3) & (col == 5), 2, idx)
    idx = jnp.where((row == 4) & (col == 5), 3, idx)
    idx = jnp.where((row == 5) & (col == 5), 4, idx)
    idx = jnp.where((row == 5) & (col == 4), 5, idx)
    idx = jnp.where((row == 5) & (col == 3), 6, idx)
    idx = jnp.where((row == 4) & (col == 3), 7, idx)
    return jnp.asarray(idx, dtype=jnp.int32)


def _go_to(row, col, target_row, target_col):
    next_rows = jnp.asarray([row, row + 1, row, row - 1], dtype=jnp.int32)
    next_cols = jnp.asarray([col + 1, col, col - 1, col], dtype=jnp.int32)
    next_idx = _ring_index(next_rows, next_cols)
    target_idx = _ring_index(target_row, target_col)
    diff = jnp.abs(next_idx - target_idx)
    dist = jnp.minimum(diff, 8 - diff)
    penalty = jnp.where(_valid_a1(next_rows, next_cols), 0, 1000)
    action = jnp.argmin(dist + penalty)
    at_target = (row == target_row) & (col == target_col)
    return jnp.asarray(jnp.where(at_target, STAY, action), dtype=jnp.int32)


def _interact_at(obs, row, col, target_row, target_col, desired_dir):
    at_target = (row == target_row) & (col == target_col)
    facing_ok = _facing(obs) == desired_dir
    orient = _face_action(jnp.asarray(desired_dir, dtype=jnp.int32))
    nav = _go_to(row, col, target_row, target_col)
    return jnp.where(at_target, jnp.where(facing_ok, INTERACT, orient), nav)


def _center_bool(obs, channel):
    return obs[2, 2, channel] > 0


def _pot_value(obs, channel):
    mask = obs[:, :, 22] > 0
    return jnp.sum(jnp.where(mask, obs[:, :, channel], 0))


def _choose(cond, then_action, else_action):
    return jnp.asarray(jnp.where(cond, then_action, else_action), dtype=jnp.int32)


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    next_memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    return jnp.asarray(STAY, dtype=jnp.int32), next_memory, {}


def init_agent_1():
    return jnp.asarray([UNKNOWN_TARGET, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    row, col = _locate_a1(obs)
    target0 = jnp.where(reset, UNKNOWN_TARGET, memory[0])
    step = jnp.where(reset, 0, memory[1])

    recipe0 = jnp.sum(obs[:, :, 36])
    recipe1 = jnp.sum(obs[:, :, 37])
    seen_recipe = (recipe0 > 0) | (recipe1 > 0)
    observed_target = jnp.where(recipe0 > 0, 0, jnp.where(recipe1 > 0, 1, target0))
    target = jnp.asarray(jnp.where(seen_recipe, observed_target, target0), dtype=jnp.int32)

    held_plate = _center_bool(obs, 5)
    held_cooked = _center_bool(obs, 6)
    held_raw0 = _center_bool(obs, 7)
    held_raw1 = _center_bool(obs, 8)
    held_raw_target = jnp.where(target == 0, held_raw0, held_raw1)
    hands_empty = ~(held_plate | held_cooked | held_raw0 | held_raw1 | _center_bool(obs, 9))

    pot_cooked = _pot_value(obs, 30) > 0
    pot0 = _pot_value(obs, 31)
    pot1 = _pot_value(obs, 32)
    pot2 = _pot_value(obs, 33)
    pot_total = pot0 + pot1 + pot2
    pot_timer = _pot_value(obs, 39)
    pot_needs_ingredients = (pot_total < 3) & (~pot_cooked) & (pot_timer == 0)
    pot_busy = (pot_timer > 0) | ((pot_total >= 3) & (~pot_cooked))
    target_unknown = target == UNKNOWN_TARGET

    act_activate = _interact_at(obs, row, col, 3, 4, DIR_UP)
    act_get_0 = _interact_at(obs, row, col, 3, 3, DIR_LEFT)
    act_get_1 = _interact_at(obs, row, col, 5, 3, DIR_LEFT)
    act_get_ingredient = jnp.where(target == 0, act_get_0, act_get_1)
    act_add = _interact_at(obs, row, col, 4, 3, DIR_RIGHT)
    act_get_plate = _interact_at(obs, row, col, 3, 5, DIR_RIGHT)
    act_wait_plate = _interact_at(obs, row, col, 4, 5, DIR_LEFT)
    act_plate = _interact_at(obs, row, col, 4, 5, DIR_LEFT)
    act_deliver = _interact_at(obs, row, col, 4, 5, DIR_RIGHT)

    action = jnp.asarray(STAY, dtype=jnp.int32)
    action = _choose(target_unknown & hands_empty & (step < 360), act_activate, action)
    action = _choose(pot_needs_ingredients & (~target_unknown) & hands_empty, act_get_ingredient, action)
    action = _choose(pot_needs_ingredients & (~target_unknown) & held_raw_target, act_add, action)
    action = _choose(pot_busy & hands_empty, act_get_plate, action)
    action = _choose(pot_busy & held_plate, act_wait_plate, action)
    action = _choose(pot_cooked & hands_empty, act_get_plate, action)
    action = _choose(pot_cooked & held_plate, act_plate, action)
    action = _choose(held_cooked, act_deliver, action)

    idx = row * 9 + col
    delivered_now = held_cooked & (idx == IDX_DELIVER) & (_facing(obs) == DIR_RIGHT) & (action == INTERACT)
    next_target = jnp.asarray(jnp.where(delivered_now, UNKNOWN_TARGET, target), dtype=jnp.int32)
    next_step = jnp.minimum(step + 1, 400)
    next_memory = jnp.asarray([next_target, next_step], dtype=jnp.int32)

    trace = {
        'pos_idx': idx,
        'target': target,
        'pot_total': pot_total,
        'pot_timer': pot_timer,
        'pot_cooked': pot_cooked,
    }
    return action, next_memory, trace
