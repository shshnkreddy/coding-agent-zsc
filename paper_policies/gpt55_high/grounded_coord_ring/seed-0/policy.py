"""Convention-robust Overcooked policy for the grounded_coord_ring layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'inner_solo_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}

ACT_RIGHT = 0
ACT_DOWN = 1
ACT_LEFT = 2
ACT_UP = 3
ACT_STAY = 4
ACT_INTERACT = 5

FACE_UP = 0
FACE_DOWN = 1
FACE_RIGHT = 2
FACE_LEFT = 3

UNKNOWN_TARGET = -1


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    action = ACT_STAY
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


def init_agent_1():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _move_action_for_face(face):
    return jnp.where(
        face == FACE_UP,
        ACT_UP,
        jnp.where(
            face == FACE_DOWN,
            ACT_DOWN,
            jnp.where(face == FACE_RIGHT, ACT_RIGHT, ACT_LEFT),
        ),
    )


def _self_global_from_pot(obs):
    rows = jnp.asarray(
        [[0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3],
         [4, 4, 4, 4, 4]],
        dtype=jnp.int32,
    )
    cols = jnp.asarray(
        [[0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4]],
        dtype=jnp.int32,
    )
    pot = obs[:, :, 22].astype(jnp.int32)
    pot_r = jnp.sum(rows * pot)
    pot_c = jnp.sum(cols * pot)
    # The pot is fixed at global (4, 4); local offset reveals self location.
    self_r = jnp.asarray(6, dtype=jnp.int32) - pot_r
    self_c = jnp.asarray(6, dtype=jnp.int32) - pot_c
    return self_r, self_c


def _go_to(obs, target_r, target_c, face, do_interact):
    self_r, self_c = _self_global_from_pot(obs)
    facing = _facing(obs)
    at_row = self_r == target_r
    at_col = self_c == target_c
    at_target = at_row & at_col
    face_action = _move_action_for_face(face)
    interact_action = jnp.where(do_interact, ACT_INTERACT, ACT_STAY)
    action_at_target = jnp.where(facing == face, interact_action, face_action)
    row_action = jnp.where(self_r < target_r, ACT_DOWN, ACT_UP)
    col_action = jnp.where(self_c < target_c, ACT_RIGHT, ACT_LEFT)

    crosses_pot_col = ((self_c < 4) & (target_c > 4)) | ((self_c > 4) & (target_c < 4))
    cross_row = jnp.where(target_r > 4, 5, 3)
    cross_row_action = jnp.where(self_r < cross_row, ACT_DOWN, ACT_UP)
    side_col = jnp.where(target_c > 4, 5, 3)
    leave_pot_col_action = jnp.where(self_c < side_col, ACT_RIGHT, ACT_LEFT)

    crossing_action = jnp.where(self_r == cross_row, col_action, cross_row_action)
    avoid_pot_col = (self_c == 4) & (self_r != target_r)
    move_action = jnp.where(at_row, col_action, row_action)
    move_action = jnp.where(avoid_pot_col, leave_pot_col_action, move_action)
    move_action = jnp.where(crosses_pot_col, crossing_action, move_action)
    return jnp.where(at_target, action_at_target, move_action)


def _visible_target(obs, old_target):
    recipe0 = jnp.sum(obs[:, :, 36])
    recipe1 = jnp.sum(obs[:, :, 37])
    seen0 = recipe0 >= 3
    seen1 = recipe1 >= 3
    return jnp.where(seen0, 0, jnp.where(seen1, 1, old_target)).astype(jnp.int32)


def _pot_value(obs, channel):
    pot = obs[:, :, 22].astype(obs.dtype)
    return jnp.sum(obs[:, :, channel] * pot)


def agent_1_act(obs, reset, memory, rng):
    old_memory = jnp.where(reset, init_agent_1(), memory)
    old_target = old_memory[0]
    old_side = old_memory[1]
    target = _visible_target(obs, old_target)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ing0 = obs[2, 2, 7] > 0
    held_ing1 = obs[2, 2, 8] > 0
    held_any = held_plate | held_cooked | held_ing0 | held_ing1 | (obs[2, 2, 9] > 0)

    pot_cooked = _pot_value(obs, 30) > 0
    pot_timer = _pot_value(obs, 39).astype(jnp.int32)
    pot_ing0 = _pot_value(obs, 31).astype(jnp.int32)
    pot_ing1 = _pot_value(obs, 32).astype(jnp.int32)
    pot_count = jnp.where(target == 0, pot_ing0, pot_ing1)
    target_unknown = target < 0
    cooking = pot_timer > 0
    pot_ready = pot_cooked
    pot_needs_ingredients = (~pot_ready) & (~cooking) & (pot_count < 3)
    holding_target_ingredient = jnp.where(target == 0, held_ing0, held_ing1)

    learn_top = _go_to(obs, 3, 4, FACE_UP, True)
    learn_bottom = _go_to(obs, 5, 4, FACE_DOWN, True)
    learn_recipe = jnp.where(old_side == 1, learn_bottom, learn_top)
    get_ing0 = _go_to(obs, 3, 3, FACE_LEFT, True)
    get_ing1 = _go_to(obs, 5, 3, FACE_LEFT, True)
    get_target_ing = jnp.where(target == 0, get_ing0, get_ing1)
    add_to_pot = _go_to(obs, 4, 3, FACE_RIGHT, True)
    get_plate = _go_to(obs, 3, 5, FACE_RIGHT, True)
    wait_or_pick_pot = _go_to(obs, 4, 5, FACE_LEFT, True)
    deliver = _go_to(obs, 4, 5, FACE_RIGHT, True)

    action = ACT_STAY
    action = jnp.where(pot_needs_ingredients & (~held_any), get_target_ing, action)
    action = jnp.where(pot_needs_ingredients & holding_target_ingredient, add_to_pot, action)
    action = jnp.where(cooking & (~held_any), get_plate, action)
    action = jnp.where(cooking & held_plate, wait_or_pick_pot, action)
    action = jnp.where(pot_ready & (~held_any), get_plate, action)
    action = jnp.where(pot_ready & held_plate, wait_or_pick_pot, action)
    action = jnp.where(held_cooked, deliver, action)
    action = jnp.where(target_unknown, learn_recipe, action)

    self_r, self_c = _self_global_from_pot(obs)
    delivery_interact = (
        held_cooked
        & (self_r == 4)
        & (self_c == 5)
        & (_facing(obs) == FACE_RIGHT)
        & (action == ACT_INTERACT)
    )
    next_target = jnp.where(delivery_interact, UNKNOWN_TARGET, target)
    next_side = jnp.where(delivery_interact, target, old_side)
    next_memory = jnp.asarray([next_target, next_side], dtype=jnp.int32)
    trace = {
        "target": next_target,
        "pot_timer": pot_timer,
        "side": next_side,
        "action": action,
    }
    return action, next_memory, trace
