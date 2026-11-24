"""Public-signal policy for the grounded_coord_simple Overcooked task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'public_button_cook',
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

FACE_UP = 0
FACE_DOWN = 1
FACE_RIGHT = 2
FACE_LEFT = 3


def _landmark_local(obs, static_channel):
    mask = obs[:, :, static_channel]
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    local_row = jnp.sum(mask * rows).astype(jnp.int32)
    local_col = jnp.sum(mask * cols).astype(jnp.int32)
    return local_row, local_col


def _infer_agent_0_pos(obs):
    local_row, local_col = _landmark_local(obs, 23)
    return 4 - local_row, 2 - local_col


def _infer_agent_1_pos(obs):
    local_row, local_col = _landmark_local(obs, 22)
    return 4 - local_row, 6 - local_col


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_action(face):
    return jnp.where(
        face == FACE_RIGHT,
        RIGHT,
        jnp.where(face == FACE_LEFT, LEFT, jnp.where(face == FACE_DOWN, DOWN, UP)),
    )


def _move_to(row, col, target_row, target_col):
    return jnp.where(
        row < target_row,
        DOWN,
        jnp.where(
            row > target_row,
            UP,
            jnp.where(col < target_col, RIGHT, jnp.where(col > target_col, LEFT, STAY)),
        ),
    )


def _go_interact(row, col, facing, target_row, target_col, target_face):
    at_target = (row == target_row) & (col == target_col)
    return jnp.where(
        at_target,
        jnp.where(facing == target_face, INTERACT, _face_action(target_face)),
        _move_to(row, col, target_row, target_col),
    )


def _display_recipe(obs, static_channel):
    mask = obs[:, :, static_channel]
    count0 = jnp.sum(mask * obs[:, :, 36])
    count1 = jnp.sum(mask * obs[:, :, 37])
    has_recipe = (count0 + count1) > 0
    recipe = jnp.where(count1 > count0, 1, 0).astype(jnp.int32)
    return has_recipe, recipe


def _pot_state(obs):
    mask = obs[:, :, 22]
    cooked = jnp.sum(mask * obs[:, :, 30])
    ing0 = jnp.sum(mask * obs[:, :, 31])
    ing1 = jnp.sum(mask * obs[:, :, 32])
    ing2 = jnp.sum(mask * obs[:, :, 33])
    timer = jnp.sum(mask * obs[:, :, 39])
    total = ing0 + ing1 + ing2
    return cooked, ing0, ing1, ing2, total, timer


def init_agent_0():
    return jnp.asarray([-1], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    signaled = jnp.where(reset, -jnp.ones_like(memory), memory)[0]
    row, col = _infer_agent_0_pos(obs)
    facing = _facing(obs)
    has_recipe, recipe = _display_recipe(obs, 23)

    needs_signal = has_recipe & (recipe != signaled)
    signal_action = _go_interact(row, col, facing, 2, 2, FACE_RIGHT)
    ready_wait = jnp.where((row == 2) & (col == 2), STAY, _move_to(row, col, 2, 2))
    action = jnp.where(needs_signal, signal_action, ready_wait).astype(jnp.int32)

    did_signal = needs_signal & (row == 2) & (col == 2) & (facing == FACE_RIGHT)
    next_signaled = jnp.where(did_signal, recipe, signaled)
    next_memory = jnp.asarray([next_signaled], dtype=jnp.int32)
    trace = {
        'recipe': recipe,
        'signaled': next_signaled,
        'needs_signal': needs_signal,
    }
    return action, next_memory, trace


def init_agent_1():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, jnp.asarray([-1, 0], dtype=jnp.int32), memory)
    row, col = _infer_agent_1_pos(obs)
    facing = _facing(obs)

    button_has, button_recipe = _display_recipe(obs, 24)
    recipe = jnp.where(button_has, button_recipe, base[0]).astype(jnp.int32)
    check_count = jnp.where(button_has, 0, base[1]).astype(jnp.int32)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held0 = obs[2, 2, 7]
    held1 = obs[2, 2, 8]
    held2 = obs[2, 2, 9]
    held_raw = (held0 + held1 + held2) > 0
    held_dish = held_plate & held_cooked
    empty_hands = ~(held_plate | held_cooked | held_raw)
    held_dish_matches_recipe = jnp.where(
        recipe == 0,
        (held0 == 3) & (held1 == 0) & (held2 == 0),
        (held1 == 3) & (held0 == 0) & (held2 == 0),
    )

    pot_cooked, pot0, pot1, pot2, pot_total, pot_timer = _pot_state(obs)
    pot_ready = (pot_cooked > 0) & (pot_timer == 0)
    pot_idle_open = (pot_cooked == 0) & (pot_timer == 0) & (pot_total < 3)
    pot_matches_recipe = jnp.where(
        recipe == 0,
        (pot0 == 3) & (pot1 == 0) & (pot2 == 0),
        (pot1 == 3) & (pot0 == 0) & (pot2 == 0),
    )
    partial_pot_matches_recipe = jnp.where(
        recipe == 0,
        (pot1 == 0) & (pot2 == 0),
        (pot0 == 0) & (pot2 == 0),
    )

    deliver_action = _go_interact(row, col, facing, 2, 6, FACE_RIGHT)
    check_action = jnp.where((row == 2) & (col == 5), STAY, _move_to(row, col, 2, 5))

    plate_row = jnp.where(row <= 2, 1, 3)
    plate_action = _go_interact(row, col, facing, plate_row, 5, FACE_LEFT)
    pot_action = _go_interact(row, col, facing, 2, 5, FACE_LEFT)

    ingr_row = jnp.where(recipe == 0, 1, 3)
    ingredient_action = _go_interact(row, col, facing, ingr_row, 6, FACE_RIGHT)

    unknown = recipe < 0
    at_check = (row == 2) & (col == 5)
    checking = (check_count > 0) & (~button_has)
    held_matches_recipe = ((recipe == 0) & (held0 > 0)) | ((recipe == 1) & (held1 > 0))
    should_plate_ready = pot_ready & pot_matches_recipe & held_plate & (~held_cooked)
    should_get_plate = pot_ready & pot_matches_recipe & empty_hands
    should_insert = held_raw & held_matches_recipe & pot_idle_open & partial_pot_matches_recipe
    should_wait_plate = held_plate & (~pot_ready)
    should_fetch_ingredient = empty_hands & pot_idle_open & (~unknown)
    should_prepare_plate = empty_hands & (~pot_idle_open)

    action = STAY
    action = jnp.where(should_prepare_plate, plate_action, action)
    action = jnp.where(should_fetch_ingredient, ingredient_action, action)
    action = jnp.where(should_wait_plate, pot_action, action)
    action = jnp.where(should_insert, pot_action, action)
    action = jnp.where(should_get_plate, plate_action, action)
    action = jnp.where(should_plate_ready, pot_action, action)
    action = jnp.where(checking, check_action, action)
    action = jnp.where(unknown, check_action, action)
    deliverable_dish = held_dish & held_dish_matches_recipe
    action = jnp.where(deliverable_dish, deliver_action, action).astype(jnp.int32)

    will_deliver = deliverable_dish & (row == 2) & (col == 6) & (facing == FACE_RIGHT)
    decremented_check = jnp.maximum(check_count - jnp.asarray(1, dtype=jnp.int32), 0)
    next_check = jnp.where(checking & at_check, decremented_check, check_count)
    next_check = jnp.where(will_deliver, 3, next_check)
    next_check = jnp.where(button_has, 0, next_check)
    next_memory = jnp.asarray([recipe, next_check], dtype=jnp.int32)
    trace = {
        'recipe': recipe,
        'check': next_check,
        'pot_total': pot_total,
        'pot_timer': pot_timer,
    }
    return action, next_memory, trace
