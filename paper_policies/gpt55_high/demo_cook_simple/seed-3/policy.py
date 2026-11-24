"""Grounded role policy for the demo_cook_simple Overcooked layout."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_roles_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

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

LOCAL_R_DATA = [[0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4]]
LOCAL_C_DATA = [[0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4]]

# Static channels 20:29 for the known 5x11 board.
STATIC_DATA = [
    [[1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0]],
]

AGENT0_R_DATA = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
AGENT0_C_DATA = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
AGENT1_R_DATA = [1, 1, 2, 2, 3, 3]
AGENT1_C_DATA = [8, 9, 8, 9, 8, 9]


def _static_crop(row, col):
    local_r = jnp.asarray(LOCAL_R_DATA, dtype=jnp.int32)
    local_c = jnp.asarray(LOCAL_C_DATA, dtype=jnp.int32)
    static = jnp.asarray(STATIC_DATA, dtype=jnp.int32)
    gr = row + local_r - 2
    gc = col + local_c - 2
    in_bounds = (gr >= 0) & (gr < 5) & (gc >= 0) & (gc < 11)
    clipped_r = jnp.clip(gr, 0, 4)
    clipped_c = jnp.clip(gc, 0, 10)
    return jnp.where(in_bounds[..., None], static[clipped_r, clipped_c, :], 0)


def _infer_position(obs, cand_r, cand_c):
    obs_static = obs[:, :, 20:29]
    scores = []
    for idx in range(cand_r.shape[0]):
        diff = jnp.abs(obs_static - _static_crop(cand_r[idx], cand_c[idx]))
        scores.append(jnp.sum(diff))
    scores = jnp.stack(scores)
    best = jnp.argmin(scores)
    return cand_r[best], cand_c[best], scores[best]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _turn_or_interact(obs, target_dir):
    face = _facing(obs)
    turn = jax.lax.switch(
        target_dir,
        (lambda: UP, lambda: DOWN, lambda: RIGHT, lambda: LEFT),
    )
    return jnp.where(face == target_dir, INTERACT, turn)


def _go_to(row, col, target_r, target_c, final_dir, obs):
    move = jnp.where(
        col < target_c,
        RIGHT,
        jnp.where(col > target_c, LEFT, jnp.where(row < target_r, DOWN, UP)),
    )
    at_target = (row == target_r) & (col == target_c)
    return jnp.where(at_target, _turn_or_interact(obs, final_dir), move)


def _cell_value(obs, self_r, self_c, global_r, global_c, channel):
    local_r = jnp.asarray(LOCAL_R_DATA, dtype=jnp.int32)
    local_c = jnp.asarray(LOCAL_C_DATA, dtype=jnp.int32)
    gr = self_r + local_r - 2
    gc = self_c + local_c - 2
    mask = (gr == global_r) & (gc == global_c)
    return jnp.sum(jnp.where(mask, obs[:, :, channel], 0))


def _agent0_recipe_target(obs, old_target):
    recipe0 = jnp.sum(obs[:, :, 36])
    recipe1 = jnp.sum(obs[:, :, 37])
    target = jnp.where(recipe0 > 0, 0, jnp.where(recipe1 > 0, 1, old_target))
    known = (target == 0) | (target == 1)
    return target.astype(jnp.int32), known


def _agent0_policy(obs, memory):
    cand_r = jnp.asarray(AGENT0_R_DATA, dtype=jnp.int32)
    cand_c = jnp.asarray(AGENT0_C_DATA, dtype=jnp.int32)
    row, col, loc_err = _infer_position(obs, cand_r, cand_c)
    target, known = _agent0_recipe_target(obs, memory[0])
    holding0 = obs[2, 2, 7] > 0
    holding1 = obs[2, 2, 8] > 0
    holding_raw = holding0 | holding1

    pot_i0 = _cell_value(obs, row, col, 2, 7, 31)
    pot_i1 = _cell_value(obs, row, col, 2, 7, 32)
    pot_cooked = _cell_value(obs, row, col, 2, 7, 30) > 0
    pot_timer = _cell_value(obs, row, col, 2, 7, 39)
    pot_visible = _cell_value(obs, row, col, 2, 7, 22) > 0
    pot_total = pot_i0 + pot_i1
    target_count = jnp.where(target == 0, pot_i0, pot_i1)
    other_count = jnp.where(target == 0, pot_i1, pot_i0)
    pot_accepting = (~pot_visible) | ((pot_timer == 0) & (~pot_cooked) & (pot_total < 3))
    pot_needs_more = known & ((~pot_visible) | ((pot_timer == 0) & (~pot_cooked) & (other_count == 0) & (pot_total < 3) & (target_count < 3)))

    fetch0 = _go_to(row, col, 1, 1, DIR_LEFT, obs)
    fetch1 = _go_to(row, col, 3, 1, DIR_LEFT, obs)
    fetch = jnp.where(target == 0, fetch0, fetch1)
    pot_action = _go_to(row, col, 2, 6, DIR_RIGHT, obs)
    drop_raw = _go_to(row, col, 1, 6, DIR_RIGHT, obs)
    idle_at_pot = jnp.where((row == 2) & (col == 6), STAY, pot_action)

    empty_action = jnp.where(known & pot_needs_more, fetch, idle_at_pot)
    raw_action = jnp.where(pot_accepting, pot_action, drop_raw)
    action = jnp.where(holding_raw, raw_action, empty_action)
    action = jnp.where(loc_err == 0, action, STAY)
    return action.astype(jnp.int32), target.astype(jnp.int32)


def _agent1_policy(obs, memory):
    del memory
    cand_r = jnp.asarray(AGENT1_R_DATA, dtype=jnp.int32)
    cand_c = jnp.asarray(AGENT1_C_DATA, dtype=jnp.int32)
    row, col, loc_err = _infer_position(obs, cand_r, cand_c)
    holding_plate = obs[2, 2, 5] > 0
    holding_cooked = obs[2, 2, 6] > 0
    holding0 = obs[2, 2, 7] > 0
    holding1 = obs[2, 2, 8] > 0
    holding_raw = holding0 | holding1
    holding_dish = holding_plate & holding_cooked

    pot_i0 = _cell_value(obs, row, col, 2, 7, 31)
    pot_i1 = _cell_value(obs, row, col, 2, 7, 32)
    pot_cooked = _cell_value(obs, row, col, 2, 7, 30) > 0
    pot_timer = _cell_value(obs, row, col, 2, 7, 39)
    pot_visible = _cell_value(obs, row, col, 2, 7, 22) > 0
    pot_total = pot_i0 + pot_i1
    pot_accepting = (~pot_visible) | ((pot_timer == 0) & (~pot_cooked) & (pot_total < 3))
    pure_partial = pot_visible & (pot_timer == 0) & (~pot_cooked) & (pot_total == 1) & ((pot_i0 == 1) | (pot_i1 == 1))

    top_plate = _go_to(row, col, 1, 9, DIR_RIGHT, obs)
    bottom_plate = _go_to(row, col, 3, 9, DIR_RIGHT, obs)
    get_plate = jnp.where(row <= 2, top_plate, bottom_plate)
    pot_action = _go_to(row, col, 2, 8, DIR_LEFT, obs)
    fetch0 = _go_to(row, col, 1, 8, DIR_UP, obs)
    fetch1 = _go_to(row, col, 3, 8, DIR_DOWN, obs)
    help_fetch = jnp.where(pot_i0 == 1, fetch0, fetch1)
    drop_raw = _go_to(row, col, 1, 8, DIR_LEFT, obs)
    deliver = _go_to(row, col, 2, 9, DIR_RIGHT, obs)

    wait_empty_pot = jnp.where((row == 2) & (col == 8), STAY, pot_action)
    empty_pot = pot_visible & (pot_timer == 0) & (~pot_cooked) & (pot_total == 0)
    empty_action = jnp.where(pure_partial, help_fetch, jnp.where(empty_pot, wait_empty_pot, get_plate))
    raw_action = jnp.where(pot_accepting, pot_action, drop_raw)
    action = jnp.where(
        holding_dish,
        deliver,
        jnp.where(holding_raw, raw_action, jnp.where(holding_plate, pot_action, empty_action)),
    )
    action = jnp.where(loc_err == 0, action, STAY)
    return action.astype(jnp.int32)


def init_agent_0():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    base = init_agent_0()
    mem = jnp.where(reset, base, memory)
    action, target = _agent0_policy(obs, mem)
    next_memory = jnp.asarray([target, mem[1] + 1], dtype=jnp.int32)
    next_memory = jnp.where(reset, base.at[1].set(1), next_memory)
    trace = {"target": next_memory[0], "step": next_memory[1]}
    return action, next_memory, trace


def init_agent_1():
    return jnp.asarray([0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base = init_agent_1()
    mem = jnp.where(reset, base, memory)
    action = _agent1_policy(obs, mem)
    next_memory = jnp.asarray([mem[0] + 1], dtype=jnp.int32)
    next_memory = jnp.where(reset, base.at[0].set(1), next_memory)
    trace = {"step": next_memory[0]}
    return action, next_memory, trace
