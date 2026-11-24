"""Hand-built JAX policy for the grounded_coord_simple Overcooked task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'button_relay_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

DIR_RIGHT = 2
DIR_LEFT = 3


def _local_feature_pos(obs, channel):
    flat = jnp.argmax(obs[:, :, channel].reshape((-1,)).astype(jnp.int32))
    return flat // 5, flat % 5


def _infer_left_position(obs):
    lr, lc = _local_feature_pos(obs, 23)
    return 4 - lr, 2 - lc


def _infer_right_position(obs):
    lr, lc = _local_feature_pos(obs, 22)
    return 4 - lr, 6 - lc


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _recipe_from_view(obs):
    r0 = jnp.max(obs[:, :, 36])
    r1 = jnp.max(obs[:, :, 37])
    return jnp.where(r0 > r1, jnp.asarray(0, jnp.int32), jnp.where(r1 > 0, jnp.asarray(1, jnp.int32), jnp.asarray(-1, jnp.int32)))


def _move_toward(row, col, tr, tc):
    vertical = jnp.where(tr > row, DOWN, UP)
    horizontal = jnp.where(tc > col, RIGHT, LEFT)
    return jnp.where(col != tc, horizontal, jnp.where(row != tr, vertical, STAY))


def _at(row, col, target_row, target_col):
    return (row == target_row) & (col == target_col)


def _act_at_object(obs, row, col, target_row, target_col, required_facing, face_action):
    facing = _facing(obs)
    at_target = _at(row, col, target_row, target_col)
    return jnp.where(at_target, jnp.where(facing == required_facing, INTERACT, face_action), _move_toward(row, col, target_row, target_col))


def _held(obs):
    inv = obs[2, 2, 5:10]
    held_plate = inv[0] > 0
    held_cooked = inv[1] > 0
    held_i0 = inv[2] > 0
    held_i1 = inv[3] > 0
    return held_plate, held_cooked, held_i0, held_i1


def _pot_index(obs):
    pot_mask = obs[:, :, 22] > 0
    flat = jnp.argmax(pot_mask.reshape((-1,)).astype(jnp.int32))
    return flat // 5, flat % 5


def _pot_state(obs):
    pr, pc = _pot_index(obs)
    cooked = obs[pr, pc, 30] > 0
    count0 = obs[pr, pc, 31]
    count1 = obs[pr, pc, 32]
    timer = obs[pr, pc, 39]
    total = count0 + count1 + obs[pr, pc, 33]
    return cooked, count0, count1, total, timer


def init_agent_0():
    # [last_recipe, activation_cooldown]
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)

    row, col = _infer_left_position(obs)
    recipe = _recipe_from_view(obs)
    last_recipe = memory[0]
    cooldown = jnp.maximum(memory[1] - 1, 0)

    at_button_cell = _at(row, col, 2, 2)
    facing = _facing(obs)
    know_recipe = recipe >= 0
    changed = know_recipe & (recipe != last_recipe)
    periodic = know_recipe & (cooldown <= 0)
    should_signal = changed | periodic

    action = jnp.where(
        at_button_cell,
        jnp.where(should_signal, jnp.where(facing == DIR_RIGHT, INTERACT, RIGHT), STAY),
        _move_toward(row, col, 2, 2),
    )

    did_interact = at_button_cell & should_signal & (facing == DIR_RIGHT)
    next_last = jnp.where(know_recipe, recipe, last_recipe)
    next_cooldown = jnp.where(did_interact, jnp.asarray(10000, dtype=jnp.int32), cooldown)
    next_memory = jnp.stack([next_last, next_cooldown])
    trace = {
        'recipe': recipe,
        'cooldown': next_cooldown,
        'signal': did_interact,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # [known_recipe, post_delivery_watch]
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)

    row, col = _infer_right_position(obs)
    seen_recipe = _recipe_from_view(obs)
    known_recipe = jnp.where(seen_recipe >= 0, seen_recipe, memory[0])
    watch = jnp.maximum(memory[1] - 1, 0)

    held_plate, held_cooked, held_i0, held_i1 = _held(obs)
    holding_any = held_plate | held_cooked | held_i0 | held_i1
    cooked, pot0, pot1, pot_total, timer = _pot_state(obs)
    pot_ready = cooked
    pot_cooking = timer > 0
    recipe_known = known_recipe >= 0
    target_pot_count = jnp.where(known_recipe == 0, pot0, pot1)
    pot_has_wrong = ((known_recipe == 0) & (pot1 > 0)) | ((known_recipe == 1) & (pot0 > 0))
    can_add = recipe_known & (~pot_ready) & (~pot_cooking) & (pot_total < 3) & (~pot_has_wrong)

    observe_action = jnp.where(_at(row, col, 2, 5), STAY, _move_toward(row, col, 2, 5))
    deliver_action = _act_at_object(obs, row, col, 2, 6, DIR_RIGHT, RIGHT)
    pot_action = _act_at_object(obs, row, col, 2, 5, DIR_LEFT, LEFT)
    plate_upper = _act_at_object(obs, row, col, 1, 5, DIR_LEFT, LEFT)
    get_i0 = _act_at_object(obs, row, col, 1, 6, DIR_RIGHT, RIGHT)
    get_i1 = _act_at_object(obs, row, col, 3, 6, DIR_RIGHT, RIGHT)
    get_ing = jnp.where(known_recipe == 0, get_i0, get_i1)

    wait_for_signal = (watch > 0) & (seen_recipe < 0)
    held_target_ing = (~held_plate) & (~held_cooked) & (((known_recipe == 0) & held_i0) | ((known_recipe == 1) & held_i1))
    need_plate = pot_ready & (~held_plate) & (~held_cooked)

    action = observe_action
    action = jnp.where(recipe_known & (~holding_any) & can_add & (target_pot_count < 3), get_ing, action)
    action = jnp.where(~recipe_known, observe_action, action)
    action = jnp.where(wait_for_signal, observe_action, action)
    action = jnp.where(recipe_known & (~holding_any) & pot_cooking, plate_upper, action)
    action = jnp.where(recipe_known & (~holding_any) & need_plate, plate_upper, action)
    action = jnp.where(recipe_known & held_target_ing & can_add, pot_action, action)
    action = jnp.where(recipe_known & held_plate & pot_ready & (~held_cooked), pot_action, action)
    action = jnp.where(recipe_known & held_cooked, deliver_action, action)

    delivered_now = held_cooked & _at(row, col, 2, 6) & (_facing(obs) == DIR_RIGHT)
    next_watch = jnp.where(delivered_now, jnp.asarray(3, dtype=jnp.int32), watch)
    next_memory = jnp.stack([known_recipe, next_watch])
    trace = {
        'recipe': known_recipe,
        'seen_recipe': seen_recipe,
        'watch': next_watch,
        'pot_total': pot_total,
        'pot_ready': pot_ready,
        'pot_timer': timer,
        'row': row,
        'col': col,
        'held_plate': held_plate,
        'held_cooked': held_cooked,
        'held_i0': held_i0,
        'held_i1': held_i1,
    }
    return action.astype(jnp.int32), next_memory, trace
