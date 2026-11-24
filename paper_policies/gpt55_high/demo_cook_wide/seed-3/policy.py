"""Cross-play oriented policy for the demo_cook_wide Overcooked task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "reader_loader_top_deliverer",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 6,
    "max_width": 11,
    "num_ingredients": 4,
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


def _first_pos(mask):
    flat = jnp.argmax(mask.reshape((25,)).astype(jnp.int32))
    return flat // 5, flat % 5, jnp.any(mask)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _held_plate(obs):
    return obs[2, 2, 5] > 0


def _held_cooked(obs):
    return obs[2, 2, 6] > 0


def _held_raw(obs):
    return jnp.sum(obs[2, 2, 7:11]) > 0


def _hands_empty(obs):
    return ~(_held_plate(obs) | _held_cooked(obs) | _held_raw(obs))


def _blocked_cell(obs, r, c):
    return jnp.sum(obs[r, c, 22:32]) > 0


def _blocked_action(obs, action):
    blocked_right = _blocked_cell(obs, 2, 3)
    blocked_down = _blocked_cell(obs, 3, 2)
    blocked_left = _blocked_cell(obs, 2, 1)
    blocked_up = _blocked_cell(obs, 1, 2)
    return (
        ((action == RIGHT) & blocked_right)
        | ((action == DOWN) & blocked_down)
        | ((action == LEFT) & blocked_left)
        | ((action == UP) & blocked_up)
    )


def _safe(action, fallback, obs):
    return jnp.where(_blocked_action(obs, action), fallback, action)


def _side_or_corridor(action, obs):
    opposite = jnp.where(action == RIGHT, LEFT, RIGHT)
    fallback = jnp.where(_blocked_action(obs, UP), opposite, UP)
    return jnp.where(_blocked_action(obs, action), fallback, action)


def _move_to_cell(obs, target_r, target_c, fallback):
    dr = target_r - 2
    dc = target_c - 2
    horizontal = jnp.where(dc > 0, RIGHT, LEFT)
    vertical = jnp.where(dr > 0, DOWN, UP)
    action = jnp.where(dc != 0, horizontal, jnp.where(dr != 0, vertical, STAY))
    return _safe(action, fallback, obs)


def _face_or_interact(obs, direction):
    face = _facing(obs)
    turn = jnp.where(
        direction == DIR_UP,
        UP,
        jnp.where(direction == DIR_DOWN, DOWN, jnp.where(direction == DIR_RIGHT, RIGHT, LEFT)),
    )
    return jnp.where(face == direction, INTERACT, turn)


def _pot_features(obs):
    mask = obs[:, :, 24] > 0
    row, col, visible = _first_pos(mask)
    item_total = jnp.sum((obs[:, :, 33:38] * mask[:, :, None]).astype(jnp.int32))
    cooked = jnp.sum((obs[:, :, 33] * mask).astype(jnp.int32)) > 0
    return row, col, visible, item_total > 0, cooked


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    pot_r, pot_c, pot_visible, pot_has_item, pot_cooked = _pot_features(obs)
    del pot_r
    pot_mask = obs[:, :, 24] > 0
    pot_count0 = jnp.sum((obs[:, :, 34] * pot_mask).astype(jnp.int32))
    pot_count1 = jnp.sum((obs[:, :, 35] * pot_mask).astype(jnp.int32))
    pot_total_raw = jnp.sum((obs[:, :, 34:38] * pot_mask[:, :, None]).astype(jnp.int32))
    pot_timer = jnp.sum((obs[:, :, 44] * pot_mask).astype(jnp.int32))
    raw_idle = pot_visible & (~pot_cooked) & (pot_timer == 0)
    helper_needed = raw_idle & (pot_total_raw == 1) & ((pot_count0 == 1) | (pot_count1 == 1))
    helper_target1 = pot_count1 > pot_count0

    at_center = pot_visible & (pot_c == 2)
    move_center = jnp.where(pot_c > 2, RIGHT, jnp.where(pot_c < 2, LEFT, STAY))
    move_center = jnp.where(pot_visible, move_center, STAY)

    ing0_adjacent = obs[2, 1, 28] > 0
    ing1_adjacent = obs[2, 3, 29] > 0
    take_ing0 = jnp.where(ing0_adjacent, _face_or_interact(obs, DIR_LEFT), move_center)
    take_ing0 = jnp.where(at_center, LEFT, take_ing0)
    take_ing1 = jnp.where(ing1_adjacent, _face_or_interact(obs, DIR_RIGHT), move_center)
    take_ing1 = jnp.where(at_center, RIGHT, take_ing1)
    take_ingredient = jnp.where(helper_target1, take_ing1, take_ing0)

    plate_adjacent = obs[1, 2, 27] > 0
    take_plate = jnp.where(plate_adjacent, _face_or_interact(obs, DIR_UP), LEFT)
    take_plate = jnp.where(at_center, LEFT, take_plate)

    can_add_raw = raw_idle & (pot_total_raw < 3)
    add_raw = jnp.where(at_center, _face_or_interact(obs, DIR_DOWN), move_center)
    drop_raw = jnp.where(obs[3, 2, 22] > 0, _face_or_interact(obs, DIR_DOWN), LEFT)
    drop_raw = jnp.where(at_center, LEFT, drop_raw)
    raw_action = jnp.where(can_add_raw, add_raw, drop_raw)

    plate_action = jnp.where(pot_cooked, _face_or_interact(obs, DIR_DOWN), STAY)
    plate_action = jnp.where(at_center, plate_action, move_center)

    deliver_action = jnp.where(at_center, _face_or_interact(obs, DIR_UP), move_center)
    idle_action = jnp.where(at_center, STAY, move_center)

    action = idle_action
    action = jnp.where(_hands_empty(obs) & helper_needed, take_ingredient, action)
    action = jnp.where(_hands_empty(obs) & pot_has_item & (~helper_needed), take_plate, action)
    action = jnp.where(_held_raw(obs), raw_action, action)
    action = jnp.where(_held_plate(obs), plate_action, action)
    action = jnp.where(_held_cooked(obs), deliver_action, action)

    zero = jnp.zeros((1,), dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)
    return action.astype(jnp.int32), next_memory, {}


def init_agent_1():
    return jnp.zeros((4,), dtype=jnp.int32)


def _home_to_recipe(obs):
    recipe_mask = obs[:, :, 25] > 0
    rr, rc, rvis = _first_pos(recipe_mask)
    desired_r = rr - 1
    desired_c = rc
    to_recipe = _move_to_cell(obs, desired_r, desired_c, STAY)

    bottom_rows = jnp.arange(5, dtype=jnp.int32)[:, None] >= 2
    side_supply = ((obs[:, :, 28] > 0) | (obs[:, :, 29] > 0)) & bottom_rows
    left_side = jnp.any(side_supply[:, :2])
    right_side = jnp.any(side_supply[:, 3:])
    sweep = jnp.where(left_side, RIGHT, jnp.where(right_side, LEFT, RIGHT))
    sweep = _side_or_corridor(sweep, obs)
    return jnp.where(rvis, to_recipe, sweep)


def _fetch_from_left_supply(obs, recipe):
    bottom_rows = jnp.arange(5, dtype=jnp.int32)[:, None] >= 2
    mask0 = (obs[:, :, 28] > 0) & bottom_rows
    mask1 = (obs[:, :, 29] > 0) & bottom_rows
    mask = jnp.where(recipe == 0, mask0, mask1)
    sr, sc, visible = _first_pos(mask)

    desired_r0 = sr
    desired_c0 = sc + 1
    desired_r1 = sr - 1
    desired_c1 = sc
    desired_r = jnp.where(recipe == 0, desired_r0, desired_r1)
    desired_c = jnp.where(recipe == 0, desired_c0, desired_c1)

    face_dir = jnp.where(recipe == 0, DIR_LEFT, DIR_DOWN)
    at_supply = (desired_r == 2) & (desired_c == 2)
    interact = _face_or_interact(obs, face_dir)
    approach = _move_to_cell(obs, desired_r, desired_c, UP)
    go_left = _safe(LEFT, UP, obs)
    return jnp.where(visible, jnp.where(at_supply, interact, approach), go_left)


def _return_to_pot(obs):
    pot_mask = obs[:, :, 24] > 0
    pr, pc, pvis = _first_pos(pot_mask)
    desired_r = pr + 1
    desired_c = pc
    at_pot = (desired_r == 2) & (desired_c == 2)
    interact = _face_or_interact(obs, DIR_UP)
    approach = _move_to_cell(obs, desired_r, desired_c, UP)
    go_right = _safe(RIGHT, UP, obs)
    return jnp.where(pvis, jnp.where(at_pot, interact, approach), go_right)


def agent_1_act(obs, reset, memory, rng):
    del rng
    zero = jnp.zeros((4,), dtype=jnp.int32)
    mem = jnp.where(reset, zero, memory)
    phase = mem[0]
    recipe = mem[1]
    count = mem[2]
    wait = mem[3]

    recipe0 = jnp.max(obs[:, :, 40])
    recipe1 = jnp.max(obs[:, :, 41])
    recipe_visible = (recipe0 + recipe1) > 0
    recipe_cell = obs[:, :, 25] > 0
    rr, rc, rvis = _first_pos(recipe_cell)
    at_recipe_station = rvis & (rr == 3) & (rc == 2)
    observed_recipe = jnp.where(recipe1 > recipe0, 1, 0).astype(jnp.int32)
    recipe = jnp.where(recipe_visible, observed_recipe, recipe)
    phase = jnp.where(recipe_visible & at_recipe_station & (phase == 0), 1, phase)

    pot_mask = obs[:, :, 24] > 0
    pot_visible = jnp.any(pot_mask)
    pot_raw_total = jnp.sum((obs[:, :, 34:38] * pot_mask[:, :, None]).astype(jnp.int32))
    pot_cooked = jnp.sum((obs[:, :, 33] * pot_mask).astype(jnp.int32)) > 0
    pot_timer = jnp.sum((obs[:, :, 44] * pot_mask).astype(jnp.int32))
    pot_busy = pot_visible & ((pot_raw_total >= 3) | (pot_timer > 0) | pot_cooked)
    partner_completed_fill = (phase == 1) & _hands_empty(obs) & pot_busy
    phase = jnp.where(partner_completed_fill, 2, phase)
    wait = jnp.where(partner_completed_fill, 22, wait)

    wait_action = _home_to_recipe(obs)
    fetch_action = _fetch_from_left_supply(obs, recipe)
    return_action = _return_to_pot(obs)
    seek_action = _home_to_recipe(obs)

    action = seek_action
    action = jnp.where((phase == 1) & _hands_empty(obs), fetch_action, action)
    action = jnp.where((phase == 1) & _held_raw(obs), return_action, action)
    action = jnp.where(phase == 2, wait_action, action)

    insert_attempt = (
        (phase == 1)
        & _held_raw(obs)
        & (action == INTERACT)
        & (_facing(obs) == DIR_UP)
    )
    new_count = count + insert_attempt.astype(jnp.int32)
    filled = new_count >= 3

    next_phase = phase
    next_count = new_count
    next_wait = wait
    next_phase = jnp.where(filled, 2, next_phase)
    next_wait = jnp.where(filled, 22, next_wait)

    dec_wait = jnp.maximum(wait - 1, 0)
    wait_done = (phase == 2) & (wait <= 1)
    next_wait = jnp.where(phase == 2, dec_wait, next_wait)
    next_phase = jnp.where(wait_done, 0, next_phase)
    next_count = jnp.where(wait_done, 0, next_count)

    next_memory = jnp.stack(
        [
            next_phase.astype(jnp.int32),
            recipe.astype(jnp.int32),
            next_count.astype(jnp.int32),
            next_wait.astype(jnp.int32),
        ]
    )
    return action.astype(jnp.int32), next_memory, {}
