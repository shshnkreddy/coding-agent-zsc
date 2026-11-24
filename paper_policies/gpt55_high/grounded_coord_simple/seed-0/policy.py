"""Deterministic Overcooked policy for the grounded_coord_simple layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'button_signal_cook_loop', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

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
UNKNOWN_RECIPE = -1


def _self_facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _first_local(mask):
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    present = mask.astype(jnp.int32)
    return jnp.sum(rows * present), jnp.sum(cols * present)


def _pose_from_anchor(obs, channel, anchor_r, anchor_c):
    loc_r, loc_c = _first_local(obs[:, :, channel] > 0)
    self_r = jnp.asarray(anchor_r + 2, dtype=jnp.int32) - loc_r
    self_c = jnp.asarray(anchor_c + 2, dtype=jnp.int32) - loc_c
    return self_r, self_c


def _go_to_pose(self_r, self_c, facing, target_r, target_c, target_facing):
    target_r = jnp.asarray(target_r, dtype=jnp.int32)
    target_c = jnp.asarray(target_c, dtype=jnp.int32)
    target_facing = jnp.asarray(target_facing, dtype=jnp.int32)
    at_target = (self_r == target_r) & (self_c == target_c)
    turn_action = jnp.where(
        target_facing == DIR_UP,
        jnp.asarray(UP, dtype=jnp.int32),
        jnp.where(
            target_facing == DIR_DOWN,
            jnp.asarray(DOWN, dtype=jnp.int32),
            jnp.where(target_facing == DIR_RIGHT, jnp.asarray(RIGHT, dtype=jnp.int32), jnp.asarray(LEFT, dtype=jnp.int32)),
        ),
    )
    row_action = jnp.where(self_r < target_r, jnp.asarray(DOWN, dtype=jnp.int32), jnp.asarray(UP, dtype=jnp.int32))
    col_action = jnp.where(self_c < target_c, jnp.asarray(RIGHT, dtype=jnp.int32), jnp.asarray(LEFT, dtype=jnp.int32))
    move_action = jnp.where(self_r != target_r, row_action, col_action)
    pose_action = jnp.where(facing == target_facing, INTERACT, turn_action)
    return jnp.where(at_target, pose_action, move_action)


def _pot_stats(obs):
    mask = (obs[:, :, 22] > 0).astype(jnp.int32)
    cooked = jnp.sum(obs[:, :, 30] * mask)
    ing0 = jnp.sum(obs[:, :, 31] * mask)
    ing1 = jnp.sum(obs[:, :, 32] * mask)
    ing2 = jnp.sum(obs[:, :, 33] * mask)
    timer = jnp.sum(obs[:, :, 39] * mask)
    total = ing0 + ing1 + ing2
    return cooked, ing0, ing1, ing2, timer, total


def _visible_recipe(obs):
    r0 = jnp.sum(obs[:, :, 36])
    r1 = jnp.sum(obs[:, :, 37])
    visible = (r0 + r1) > 0
    recipe = jnp.where(r1 > r0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    return visible, recipe


def init_agent_0():
    return jnp.zeros((2,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    base = init_agent_0()
    mem = jnp.where(reset, base, memory)
    cooldown = jnp.maximum(mem[0] - jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    elapsed = mem[1]

    self_r, self_c = _pose_from_anchor(obs, 23, 2, 0)
    facing = _self_facing(obs)
    cooked, ing0, ing1, ing2, timer, total = _pot_stats(obs)
    pot_empty = (total == 0) & (cooked == 0) & (timer == 0)

    nav_action = _go_to_pose(self_r, self_c, facing, 2, 2, DIR_RIGHT)
    at_signal_pose = (self_r == 2) & (self_c == 2) & (facing == DIR_RIGHT)
    should_press = at_signal_pose & pot_empty & (cooldown == 0) & (elapsed < 350)
    action = jnp.where(should_press, jnp.asarray(INTERACT, dtype=jnp.int32), jnp.where(at_signal_pose, jnp.asarray(STAY, dtype=jnp.int32), nav_action))
    next_cooldown = jnp.where(should_press, jnp.asarray(35, dtype=jnp.int32), cooldown)
    next_elapsed = elapsed + jnp.asarray(1, dtype=jnp.int32)
    return action, jnp.asarray([next_cooldown, next_elapsed], dtype=jnp.int32), {}


def init_agent_1():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    base = init_agent_1()
    mem = jnp.where(reset, base, memory)

    self_r, self_c = _pose_from_anchor(obs, 22, 2, 4)
    facing = _self_facing(obs)
    visible, seen_recipe = _visible_recipe(obs)
    recipe = jnp.where(visible, seen_recipe, mem[0])
    known_recipe = recipe >= 0

    cooked, ing0, ing1, ing2, timer, total = _pot_stats(obs)
    pot_ready = (cooked > 0) & (timer == 0) & (total > 0)
    pot_busy = (timer > 0) | (total >= 3)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = (obs[2, 2, 7] + obs[2, 2, 8] + obs[2, 2, 9]) > 0

    ingredient_target_r = jnp.where(recipe == 1, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    ingredient_action = _go_to_pose(self_r, self_c, facing, ingredient_target_r, 6, DIR_RIGHT)
    plate_target_r = jnp.where(recipe == 1, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    plate_action = _go_to_pose(self_r, self_c, facing, plate_target_r, 5, DIR_LEFT)
    pot_action = _go_to_pose(self_r, self_c, facing, 2, 5, DIR_LEFT)
    delivery_action = _go_to_pose(self_r, self_c, facing, 2, 6, DIR_RIGHT)
    wait_for_signal_action = jnp.where(
        (self_r == 2) & (self_c == 5) & (facing == DIR_LEFT),
        jnp.asarray(STAY, dtype=jnp.int32),
        _go_to_pose(self_r, self_c, facing, 2, 5, DIR_LEFT),
    )

    empty_handed_action = jnp.where(
        pot_ready | pot_busy,
        plate_action,
        ingredient_action,
    )
    known_action = jnp.where(
        held_cooked,
        delivery_action,
        jnp.where(
            held_plate,
            pot_action,
            jnp.where(held_raw, pot_action, empty_handed_action),
        ),
    )
    action = jnp.where(known_recipe, known_action, wait_for_signal_action)

    will_deliver = held_cooked & (self_r == 2) & (self_c == 6) & (facing == DIR_RIGHT) & (action == INTERACT)
    next_recipe = jnp.where(will_deliver, jnp.asarray(UNKNOWN_RECIPE, dtype=jnp.int32), recipe)
    next_mem = jnp.asarray([next_recipe, jnp.asarray(0, dtype=jnp.int32)], dtype=jnp.int32)
    return action, next_mem, {}
