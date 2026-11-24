"""Convention-light Overcooked policy for the grounded_coord_ring task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'inner_solo_button_cook',
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

FACE_UP = 0
FACE_RIGHT = 2
FACE_LEFT = 3

FIRST_ACTION = [
    [4, 0, 0, 1, 0, 1, 1, 0],
    [2, 4, 0, 2, 0, 2, 0, 0],
    [2, 2, 4, 2, 1, 1, 1, 1],
    [3, 3, 3, 4, 1, 1, 1, 1],
    [3, 3, 3, 1, 4, 1, 1, 1],
    [3, 3, 0, 3, 0, 4, 0, 0],
    [2, 0, 0, 2, 0, 2, 4, 0],
    [2, 3, 3, 2, 3, 2, 2, 4],
]


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del obs, rng
    next_memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    return jnp.asarray(STAY, dtype=jnp.int32), next_memory, {}


def init_agent_1():
    # [remembered target ingredient, target valid flag]
    return jnp.zeros((2,), dtype=jnp.int32)


def _locate_inner_node(obs):
    pot_flat = jnp.argmax(jnp.reshape(obs[:, :, 22], (25,)))
    pot_local_r = pot_flat // 5
    pot_local_c = pot_flat - pot_local_r * 5
    row = 6 - pot_local_r
    col = 6 - pot_local_c
    node = jnp.where(row == 3, col - 3, 0)
    node = jnp.where((row == 4) & (col == 3), 3, node)
    node = jnp.where((row == 4) & (col == 5), 4, node)
    node = jnp.where(row == 5, col + 2, node)
    return jnp.asarray(node, dtype=jnp.int32)


def _facing(obs):
    return jnp.asarray(jnp.argmax(obs[2, 2, 1:5]), dtype=jnp.int32)


def _nav_action(node, target_node):
    table = jnp.asarray(FIRST_ACTION, dtype=jnp.int32)
    return table[node, target_node]


def _interact_at(node, facing, stand_node, face_id, turn_action):
    at_stand = node == jnp.asarray(stand_node, dtype=jnp.int32)
    ready = at_stand & (facing == face_id)
    return jnp.where(ready, INTERACT, jnp.where(at_stand, turn_action, _nav_action(node, stand_node)))


def _face_or_wait_at(node, facing, stand_node, face_id, turn_action):
    at_stand = node == jnp.asarray(stand_node, dtype=jnp.int32)
    ready = at_stand & (facing == face_id)
    return jnp.where(ready, STAY, jnp.where(at_stand, turn_action, _nav_action(node, stand_node)))


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, jnp.zeros_like(memory), memory)

    node = _locate_inner_node(obs)
    facing = _facing(obs)

    recipe_0 = jnp.max(obs[:, :, 36])
    recipe_1 = jnp.max(obs[:, :, 37])
    recipe_visible = (recipe_0 + recipe_1) > 0
    observed_target = jnp.where(recipe_1 > recipe_0, 1, 0).astype(jnp.int32)
    target = jnp.where(recipe_visible, observed_target, memory[0])
    target_valid = (memory[1] > 0) | recipe_visible

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_i0 = obs[2, 2, 7]
    held_i1 = obs[2, 2, 8]
    held_i2 = obs[2, 2, 9]
    held_raw = (held_i0 + held_i1 + held_i2) > 0
    held_matches = ((target == 0) & (held_i0 == 3) & (held_i1 == 0) & (held_i2 == 0)) | (
        (target == 1) & (held_i0 == 0) & (held_i1 == 3) & (held_i2 == 0)
    )

    pot_mask = obs[:, :, 22]
    pot_cooked = jnp.sum(pot_mask * obs[:, :, 30]) > 0
    pot_i0 = jnp.sum(pot_mask * obs[:, :, 31])
    pot_i1 = jnp.sum(pot_mask * obs[:, :, 32])
    pot_i2 = jnp.sum(pot_mask * obs[:, :, 33])
    pot_total = pot_i0 + pot_i1 + pot_i2
    pot_timer = jnp.sum(pot_mask * obs[:, :, 39])
    pot_busy = (pot_timer > 0) | (pot_total >= 3)
    pot_matches = ((target == 0) & (pot_i0 == 3) & (pot_i1 == 0) & (pot_i2 == 0)) | (
        (target == 1) & (pot_i0 == 0) & (pot_i1 == 3) & (pot_i2 == 0)
    )
    pot_bad = ((target == 0) & ((pot_i1 + pot_i2) > 0)) | (
        (target == 1) & ((pot_i0 + pot_i2) > 0)
    )

    read_recipe = _interact_at(node, facing, 1, FACE_UP, UP)
    get_i0 = _interact_at(node, facing, 0, FACE_LEFT, LEFT)
    get_i1 = _interact_at(node, facing, 5, FACE_LEFT, LEFT)
    get_ingredient = jnp.where(target == 0, get_i0, get_i1)
    add_to_pot = _interact_at(node, facing, 3, FACE_RIGHT, RIGHT)
    get_plate = _interact_at(node, facing, 2, FACE_RIGHT, RIGHT)
    pickup_pot = _interact_at(node, facing, 4, FACE_LEFT, LEFT)
    wait_pot = _face_or_wait_at(node, facing, 4, FACE_LEFT, LEFT)
    deliver_right = _interact_at(node, facing, 4, FACE_RIGHT, RIGHT)

    can_deliver = held_cooked & held_matches
    can_pickup = pot_cooked & pot_matches & held_plate
    should_plate = (pot_cooked | pot_busy) & (~held_plate) & (~held_cooked) & (~pot_bad)
    should_wait = pot_busy & held_plate & (~pot_cooked)
    should_add = held_raw & (~pot_busy) & (~pot_bad)
    should_fetch = (~held_raw) & (~held_plate) & (~held_cooked) & (pot_total < 3) & (~pot_bad)

    action = jnp.where(
        ~target_valid,
        read_recipe,
        jnp.where(
            can_deliver,
            deliver_right,
            jnp.where(
                can_pickup,
                pickup_pot,
                jnp.where(
                    should_plate,
                    get_plate,
                    jnp.where(
                        should_wait,
                        wait_pot,
                        jnp.where(should_add, add_to_pot, jnp.where(should_fetch, get_ingredient, STAY)),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    delivered_now = can_deliver & (action == INTERACT) & (node == 4) & (facing == FACE_RIGHT)
    next_target = jnp.where(recipe_visible, observed_target, target)
    next_valid = jnp.where(delivered_now, 0, jnp.where(recipe_visible, 1, target_valid.astype(jnp.int32)))
    next_memory = jnp.stack([next_target.astype(jnp.int32), next_valid.astype(jnp.int32)])
    trace = {
        'node': node,
        'target': target,
        'valid': target_valid.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'action': action,
    }
    return action, next_memory, trace
