"""Deterministic Overcooked V2 policy for grounded_coord_ring."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'center_solo_cook_v1',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}

STAY_ID = 4
INTERACT_ID = 5
DIR_TO_ACTION_DATA = [3, 1, 0, 2]

# NEXT_TO_TARGET[target_center_index, current_center_index]
NEXT_TO_TARGET_DATA = [
    [4, 2, 2, 3, 3, 3, 2, 2],
    [0, 4, 2, 3, 3, 3, 2, 3],
    [0, 0, 4, 3, 3, 3, 0, 3],
    [1, 2, 2, 4, 3, 3, 2, 2],
    [0, 0, 1, 3, 4, 0, 0, 3],
    [1, 2, 2, 1, 1, 4, 2, 2],
    [1, 2, 0, 1, 1, 0, 4, 2],
    [0, 0, 1, 1, 1, 0, 0, 4],
]

TARGET_TOP_BUTTON = 1   # (3, 4), face up
TARGET_ING0 = 0         # (3, 3), face left
TARGET_ING1 = 5         # (5, 3), face left
TARGET_POT_ADD = 3      # (4, 3), face right
TARGET_PLATE = 2        # (3, 5), face right
TARGET_POT_PLATE = 4    # (4, 5), face left
TARGET_DELIVER = 4      # (4, 5), face right


def _initial_memory():
    return jnp.asarray([-1], dtype=jnp.int32)


def _infer_global_pos(obs):
    pot_flat = jnp.argmax(obs[:, :, 22].reshape((25,))).astype(jnp.int32)
    pot_local_r = pot_flat // 5
    pot_local_c = pot_flat - pot_local_r * 5
    return jnp.stack(
        [
            jnp.asarray(6, dtype=jnp.int32) - pot_local_r,
            jnp.asarray(6, dtype=jnp.int32) - pot_local_c,
        ]
    )


def _center_index(global_pos):
    row = global_pos[0]
    col = global_pos[1]
    top_idx = col - jnp.asarray(3, dtype=jnp.int32)
    mid_idx = jnp.where(col < 4, 3, 4)
    bot_idx = col + jnp.asarray(2, dtype=jnp.int32)
    return jnp.where(row == 3, top_idx, jnp.where(row == 4, mid_idx, bot_idx)).astype(jnp.int32)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _go_or_interact(pos_idx, face_dir, target_idx, desired_face):
    dir_to_action = jnp.asarray(DIR_TO_ACTION_DATA, dtype=jnp.int32)
    next_to_target = jnp.asarray(NEXT_TO_TARGET_DATA, dtype=jnp.int32)
    at_target = pos_idx == target_idx
    face_action = dir_to_action[desired_face]
    turn_or_use = jnp.where(face_dir == desired_face, INTERACT_ID, face_action)
    return jnp.where(at_target, turn_or_use, next_to_target[target_idx, pos_idx])


def _go_or_wait(pos_idx, face_dir, target_idx, desired_face):
    dir_to_action = jnp.asarray(DIR_TO_ACTION_DATA, dtype=jnp.int32)
    next_to_target = jnp.asarray(NEXT_TO_TARGET_DATA, dtype=jnp.int32)
    at_target = pos_idx == target_idx
    face_action = dir_to_action[desired_face]
    turn_or_wait = jnp.where(face_dir == desired_face, STAY_ID, face_action)
    return jnp.where(at_target, turn_or_wait, next_to_target[target_idx, pos_idx])


def _recipe_from_obs(obs, previous_recipe):
    recipe0 = jnp.sum(obs[:, :, 36])
    recipe1 = jnp.sum(obs[:, :, 37])
    return jnp.where(recipe0 > 0, 0, jnp.where(recipe1 > 0, 1, previous_recipe))


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del obs, rng
    zero = jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(STAY_ID, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    return _initial_memory()


def agent_1_act(obs, reset, memory, rng):
    del rng
    base_memory = _initial_memory()
    memory = jnp.where(reset, base_memory, memory)

    global_pos = _infer_global_pos(obs)
    pos_idx = _center_index(global_pos)
    face_dir = _facing(obs)

    recipe = _recipe_from_obs(obs, memory[0])
    recipe_known = recipe >= 0

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ingredients = jnp.sum(obs[2, 2, 7:10]) > 0

    pot_mask = obs[:, :, 22]
    pot_cooked = jnp.sum(obs[:, :, 30] * pot_mask) > 0
    pot_counts = jnp.asarray(
        [
            jnp.sum(obs[:, :, 31] * pot_mask),
            jnp.sum(obs[:, :, 32] * pot_mask),
            jnp.sum(obs[:, :, 33] * pot_mask),
        ],
        dtype=jnp.int32,
    )
    pot_total = jnp.sum(pot_counts)
    pot_timer = jnp.sum(obs[:, :, 39] * pot_mask)

    act_recipe = _go_or_interact(pos_idx, face_dir, TARGET_TOP_BUTTON, jnp.asarray(0, dtype=jnp.int32))
    act_fetch0 = _go_or_interact(pos_idx, face_dir, TARGET_ING0, jnp.asarray(3, dtype=jnp.int32))
    act_fetch1 = _go_or_interact(pos_idx, face_dir, TARGET_ING1, jnp.asarray(3, dtype=jnp.int32))
    act_fetch = jnp.where(recipe == 0, act_fetch0, act_fetch1)
    act_add = _go_or_interact(pos_idx, face_dir, TARGET_POT_ADD, jnp.asarray(2, dtype=jnp.int32))
    act_plate_pickup = _go_or_interact(pos_idx, face_dir, TARGET_PLATE, jnp.asarray(2, dtype=jnp.int32))
    act_wait_plate = _go_or_wait(pos_idx, face_dir, TARGET_POT_PLATE, jnp.asarray(3, dtype=jnp.int32))
    act_scoop = _go_or_interact(pos_idx, face_dir, TARGET_POT_PLATE, jnp.asarray(3, dtype=jnp.int32))
    act_deliver = _go_or_interact(pos_idx, face_dir, TARGET_DELIVER, jnp.asarray(2, dtype=jnp.int32))

    pot_busy = (pot_total >= 3) | (pot_timer > 0) | pot_cooked
    action_when_empty = jnp.where(
        pot_cooked | pot_busy,
        act_plate_pickup,
        act_fetch,
    )
    action_known = action_when_empty
    action_known = jnp.where(held_ingredients, act_add, action_known)
    action_known = jnp.where(held_plate & (~held_cooked) & pot_cooked, act_scoop, action_known)
    action_known = jnp.where(held_plate & (~held_cooked) & (~pot_cooked), act_wait_plate, action_known)
    action_known = jnp.where(held_cooked, act_deliver, action_known)

    action = jnp.where(recipe_known, action_known, act_recipe)

    delivering_now = (
        held_cooked
        & (pos_idx == TARGET_DELIVER)
        & (face_dir == jnp.asarray(2, dtype=jnp.int32))
        & (action == INTERACT_ID)
    )
    next_recipe = jnp.where(delivering_now, -1, recipe)
    next_memory = jnp.asarray([next_recipe], dtype=jnp.int32)

    trace = {
        'recipe': next_recipe,
        'pos_r': global_pos[0],
        'pos_c': global_pos[1],
        'pot_total': pot_total,
        'pot_timer': pot_timer,
    }
    return action.astype(jnp.int32), next_memory, trace
