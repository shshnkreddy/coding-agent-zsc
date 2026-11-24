"""Recipe-button convention policy for the grounded coordination layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'button_solo_chef_v1',
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

DIR_LEFT = 3
DIR_RIGHT = 2

CENTER = 2
UNKNOWN_RECIPE = 2
BUTTON_PERIOD = 80


def _loc(mask):
    idx = jnp.argmax(mask.reshape((-1,)))
    return idx // 5, idx % 5


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _move_to(row, col):
    row = row.astype(jnp.int32)
    col = col.astype(jnp.int32)
    vertical = jnp.where(row < CENTER, UP, DOWN)
    horizontal = jnp.where(col < CENTER, LEFT, RIGHT)
    return jnp.where(
        row != CENTER,
        vertical,
        jnp.where(col != CENTER, horizontal, STAY),
    ).astype(jnp.int32)


def _go_interact(obs, stand_row, stand_col, direction):
    at_stand = (stand_row == CENTER) & (stand_col == CENTER)
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[direction]
    return jnp.where(
        at_stand,
        jnp.where(_facing(obs) == direction, INTERACT, face_action),
        _move_to(stand_row, stand_col),
    ).astype(jnp.int32)


def _recipe_from_display(obs):
    recipe_0 = jnp.max(obs[:, :, 36])
    recipe_1 = jnp.max(obs[:, :, 37])
    seen = (recipe_0 + recipe_1) > 0
    target = jnp.where(recipe_1 > recipe_0, 1, 0).astype(jnp.int32)
    return seen, target


def _pot_info(obs):
    pot_r, pot_c = _loc(obs[:, :, 22] > 0)
    counts = obs[pot_r, pot_c, 31:34].astype(jnp.int32)
    cooked = obs[pot_r, pot_c, 30] > 0
    timer = obs[pot_r, pot_c, 39].astype(jnp.int32)
    return pot_r, pot_c, counts, cooked, timer


def _go_pot(obs):
    pot_r, pot_c, _, _, _ = _pot_info(obs)
    return _go_interact(obs, pot_r, pot_c + 1, DIR_LEFT)


def _wait_at_pot(obs):
    pot_r, pot_c, _, _, _ = _pot_info(obs)
    stand_r = pot_r
    stand_c = pot_c + 1
    at_stand = (stand_r == CENTER) & (stand_c == CENTER)
    return jnp.where(
        at_stand,
        jnp.where(_facing(obs) == DIR_LEFT, STAY, LEFT),
        _move_to(stand_r, stand_c),
    ).astype(jnp.int32)


def _go_plate(obs):
    plate_r, plate_c = _loc(obs[:, :, 25] > 0)
    return _go_interact(obs, plate_r, plate_c + 1, DIR_LEFT)


def _go_delivery(obs):
    goal_r, goal_c = _loc(obs[:, :, 21] > 0)
    return _go_interact(obs, goal_r, goal_c - 1, DIR_RIGHT)


def _go_ingredient(obs, target):
    supply = jnp.where(target == 0, obs[:, :, 26] > 0, obs[:, :, 27] > 0)
    ing_r, ing_c = _loc(supply)
    return _go_interact(obs, ing_r, ing_c - 1, DIR_RIGHT)


def init_agent_0():
    # [step, button cooldown, last observed recipe ingredient, previous pot occupied]
    return jnp.asarray([0, 0, 2, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero_mem = init_agent_0()
    mem = jnp.where(reset, zero_mem, memory)
    step = mem[0]
    cooldown = mem[1]
    last_recipe = mem[2]
    prev_pot_occupied = mem[3] > 0

    seen, recipe = _recipe_from_display(obs)
    recipe = jnp.where(seen, recipe, last_recipe).astype(jnp.int32)
    recipe_changed = seen & (recipe != last_recipe)

    pot_visible = jnp.max(obs[:, :, 22]) > 0
    _, _, pot_counts, pot_cooked, pot_timer = _pot_info(obs)
    pot_occupied = pot_visible & (
        (jnp.sum(pot_counts) > 0) | pot_cooked | (pot_timer > 0)
    )
    pot_emptied = prev_pot_occupied & (~pot_occupied)
    useful_window = step < 350
    display_stale = cooldown <= (BUTTON_PERIOD - 10)
    pulse_due = useful_window & (
        (cooldown <= 0) | pot_emptied | (recipe_changed & display_stale)
    )

    button_r, button_c = _loc(obs[:, :, 24] > 0)
    pulse_action = _go_interact(obs, button_r, button_c - 1, DIR_RIGHT)
    action = jnp.where(pulse_due, pulse_action, STAY).astype(jnp.int32)

    did_press = pulse_due & (pulse_action == INTERACT)
    next_cooldown = jnp.where(
        did_press,
        BUTTON_PERIOD,
        jnp.maximum(cooldown - 1, 0),
    ).astype(jnp.int32)
    next_memory = jnp.asarray(
        [
            step + 1,
            next_cooldown,
            jnp.where(seen, recipe, last_recipe),
            pot_occupied.astype(jnp.int32),
        ],
        dtype=jnp.int32,
    )
    return action, next_memory, {}


def init_agent_1():
    # [current target ingredient: 0, 1, or UNKNOWN_RECIPE]
    return jnp.asarray([2], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    zero_mem = init_agent_1()
    mem = jnp.where(reset, zero_mem, memory)
    stored_target = mem[0]

    seen, display_target = _recipe_from_display(obs)
    target = jnp.where(seen, display_target, stored_target).astype(jnp.int32)
    known_target = target != UNKNOWN_RECIPE

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:10]) > 0

    _, _, pot_counts, pot_cooked, pot_timer = _pot_info(obs)
    pot_total = jnp.sum(pot_counts)
    pot_full = pot_total >= 3
    cooking = pot_timer > 0

    action_deliver = _go_delivery(obs)
    action_plate_to_pot = _go_pot(obs)
    action_raw_to_pot = _go_pot(obs)
    action_unknown = _wait_at_pot(obs)
    action_get_plate = _go_plate(obs)
    action_get_ingredient = _go_ingredient(obs, jnp.minimum(target, 1))
    action_wait = _wait_at_pot(obs)

    empty_hand_action = jnp.where(
        ~known_target,
        action_unknown,
        jnp.where(
            pot_cooked,
            action_get_plate,
            jnp.where(
                (~pot_full) & (~cooking),
                action_get_ingredient,
                action_get_plate,
            ),
        ),
    )

    action = jnp.where(
        held_cooked,
        action_deliver,
        jnp.where(
            held_plate,
            jnp.where(pot_cooked, action_plate_to_pot, action_wait),
            jnp.where(
                held_raw,
                action_raw_to_pot,
                empty_hand_action,
            ),
        ),
    ).astype(jnp.int32)

    did_deliver = held_cooked & (action == INTERACT)
    next_target = jnp.where(did_deliver, UNKNOWN_RECIPE, target)
    next_memory = jnp.asarray([next_target], dtype=jnp.int32)
    return action, next_memory, {}
