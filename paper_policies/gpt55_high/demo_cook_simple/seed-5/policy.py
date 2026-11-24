"""Deterministic role-split policy for the demo_cook_simple layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'visible_recipe_role_split', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

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

UNKNOWN_RECIPE = 2
WAIT_LIMIT = 11

AGENT0_POSITIONS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
)

AGENT1_POSITIONS = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))


def _static_board():
    board = []
    walls = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10),
        (1, 7), (2, 0), (3, 7),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 7), (4, 9), (4, 10),
    }
    goals = {(2, 10)}
    pots = {(2, 7)}
    recipes = {(0, 5), (4, 5)}
    plates = {(1, 10), (3, 10)}
    ing0 = {(0, 8), (1, 0)}
    ing1 = {(3, 0), (4, 8)}
    ing2 = {(0, 6), (4, 6)}
    for r in range(5):
        row = []
        for c in range(11):
            row.append(
                [
                    int((r, c) in walls),
                    int((r, c) in goals),
                    int((r, c) in pots),
                    int((r, c) in recipes),
                    0,
                    int((r, c) in plates),
                    int((r, c) in ing0),
                    int((r, c) in ing1),
                    int((r, c) in ing2),
                ]
            )
        board.append(row)
    return board


def _crop_for(pos):
    static_board = _static_board()
    pr, pc = pos
    crop = []
    for dr in range(-2, 3):
        row = []
        for dc in range(-2, 3):
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < 5 and 0 <= cc < 11:
                row.append(static_board[rr][cc])
            else:
                row.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        crop.append(row)
    return crop


def _crops_for(positions):
    return [_crop_for(pos) for pos in positions]


def _localize(obs, positions_tuple):
    positions = jnp.asarray(positions_tuple, dtype=jnp.int32)
    crops = jnp.asarray(_crops_for(positions_tuple), dtype=jnp.int32)
    static_obs = obs[:, :, 20:29]
    diff = jnp.abs(crops - static_obs)
    scores = jnp.sum(diff, axis=(1, 2, 3))
    idx = jnp.argmin(scores)
    return positions[idx, 0], positions[idx, 1]


def _face(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _move_to(row, col, target_row, target_col):
    row = row.astype(jnp.int32)
    col = col.astype(jnp.int32)
    target_row = jnp.asarray(target_row, dtype=jnp.int32)
    target_col = jnp.asarray(target_col, dtype=jnp.int32)
    action = jnp.asarray(STAY, dtype=jnp.int32)
    action = jnp.where(row > target_row, UP, action)
    action = jnp.where(row < target_row, DOWN, action)
    action = jnp.where(col > target_col, LEFT, action)
    action = jnp.where(col < target_col, RIGHT, action)
    return action.astype(jnp.int32)


def _action_for_face(direction):
    action = jnp.asarray(RIGHT, dtype=jnp.int32)
    action = jnp.where(direction == FACE_DOWN, DOWN, action)
    action = jnp.where(direction == FACE_LEFT, LEFT, action)
    action = jnp.where(direction == FACE_UP, UP, action)
    return action.astype(jnp.int32)


def _go_or_interact(row, col, face, target_row, target_col, direction):
    at_target = (row == jnp.asarray(target_row, dtype=jnp.int32)) & (col == jnp.asarray(target_col, dtype=jnp.int32))
    move = _move_to(row, col, target_row, target_col)
    orient = _action_for_face(jnp.asarray(direction, dtype=jnp.int32))
    return jnp.where(at_target, jnp.where(face == direction, INTERACT, orient), move).astype(jnp.int32)


def init_agent_0():
    return jnp.asarray([UNKNOWN_RECIPE, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    row, col = _localize(obs, AGENT0_POSITIONS)
    face = _face(obs)

    init_memory = jnp.asarray([UNKNOWN_RECIPE, 0], dtype=jnp.int32)
    active_memory = jnp.where(reset, init_memory, memory).astype(jnp.int32)
    old_target = active_memory[0]
    old_wait = active_memory[1]
    recipe_0 = jnp.sum(obs[:, :, 36])
    recipe_1 = jnp.sum(obs[:, :, 37])
    sees_recipe = (recipe_0 + recipe_1) > 0
    seen_target = jnp.where(recipe_1 > recipe_0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    target = jnp.where(sees_recipe, seen_target, old_target).astype(jnp.int32)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = (obs[2, 2, 7] + obs[2, 2, 8] + obs[2, 2, 9]) > 0
    holding_raw_only = held_raw & (~held_plate) & (~held_cooked)
    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_plate = jnp.sum(obs[:, :, 15]) > 0
    partner_cooked = jnp.sum(obs[:, :, 16]) > 0
    partner_can_assist = partner_visible & (~partner_plate) & (~partner_cooked)

    pot_mask = obs[:, :, 22] > 0
    pot_visible = jnp.sum(pot_mask.astype(jnp.int32)) > 0
    pot_timer = jnp.sum(jnp.where(pot_mask, obs[:, :, 39], 0))
    pot_cooked = jnp.sum(jnp.where(pot_mask, obs[:, :, 30], 0))
    pot_total = jnp.sum(jnp.where(pot_mask, obs[:, :, 31] + obs[:, :, 32] + obs[:, :, 33], 0))
    pot_busy = pot_visible & ((pot_timer > 0) | (pot_cooked > 0) | (pot_total >= 3))
    pot_partial = pot_visible & (pot_timer == 0) & (pot_cooked == 0) & (pot_total > 0) & (pot_total < 3)
    at_pot = (row == 2) & (col == 6)
    wait_for_assist = pot_partial & (~holding_raw_only) & partner_can_assist & (old_wait < WAIT_LIMIT)

    to_recipe = _move_to(row, col, 2, 5)
    to_pot = _go_or_interact(row, col, face, 2, 6, FACE_RIGHT)
    fetch_0 = _go_or_interact(row, col, face, 1, 1, FACE_LEFT)
    fetch_1 = _go_or_interact(row, col, face, 3, 1, FACE_LEFT)
    fetch = jnp.where(target == 0, fetch_0, fetch_1)
    wait_at_pot = jnp.where((row == 2) & (col == 6), STAY, _move_to(row, col, 2, 6))
    partial_wait = jnp.where(at_pot, STAY, _move_to(row, col, 2, 6))

    action = fetch
    action = jnp.where(wait_for_assist, partial_wait, action)
    action = jnp.where(pot_busy, wait_at_pot, action)
    action = jnp.where(target == UNKNOWN_RECIPE, to_recipe, action)
    action = jnp.where(holding_raw_only, to_pot, action)
    next_wait = jnp.where(wait_for_assist & at_pot, old_wait + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    next_memory = jnp.asarray([target, next_wait], dtype=jnp.int32)

    trace = {
        "row": row,
        "col": col,
        "target": target,
        "wait": next_wait,
        "partner_can_assist": partner_can_assist.astype(jnp.int32),
        "pot_busy": pot_busy.astype(jnp.int32),
        "pot_total": pot_total.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    row, col = _localize(obs, AGENT1_POSITIONS)
    face = _face(obs)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = (obs[2, 2, 7] + obs[2, 2, 8] + obs[2, 2, 9]) > 0
    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_plate = jnp.sum(obs[:, :, 15]) > 0
    partner_cooked = jnp.sum(obs[:, :, 16]) > 0
    partner_raw = jnp.sum(obs[:, :, 17] + obs[:, :, 18] + obs[:, :, 19]) > 0
    partner_empty = partner_visible & (~partner_plate) & (~partner_cooked) & (~partner_raw)

    pot_mask = obs[:, :, 22] > 0
    pot_visible = jnp.sum(pot_mask.astype(jnp.int32)) > 0
    pot_timer = jnp.sum(jnp.where(pot_mask, obs[:, :, 39], 0))
    pot_cooked = jnp.sum(jnp.where(pot_mask, obs[:, :, 30], 0))
    pot_count_0 = jnp.sum(jnp.where(pot_mask, obs[:, :, 31], 0))
    pot_count_1 = jnp.sum(jnp.where(pot_mask, obs[:, :, 32], 0))
    pot_total = pot_count_0 + pot_count_1 + jnp.sum(jnp.where(pot_mask, obs[:, :, 33], 0))
    pot_busy = pot_visible & ((pot_timer > 0) | (pot_cooked > 0) | (pot_total >= 3))
    assist_one = pot_total == 1
    assist_two = (pot_total == 2) & partner_empty
    assist_partial = pot_visible & (pot_timer == 0) & (pot_cooked == 0) & (assist_one | assist_two)

    upper_plate = _go_or_interact(row, col, face, 1, 9, FACE_RIGHT)
    lower_plate = _go_or_interact(row, col, face, 3, 9, FACE_RIGHT)
    get_plate = jnp.where(row <= 2, upper_plate, lower_plate)
    plate_pot = _go_or_interact(row, col, face, 2, 8, FACE_LEFT)
    deliver = _go_or_interact(row, col, face, 2, 9, FACE_RIGHT)
    fetch_0 = _go_or_interact(row, col, face, 1, 8, FACE_UP)
    fetch_1 = _go_or_interact(row, col, face, 3, 8, FACE_DOWN)
    assist_fetch = jnp.where(pot_count_0 > pot_count_1, fetch_0, fetch_1)
    wait_empty = jnp.where((row == 2) & (col == 8), _action_for_face(jnp.asarray(FACE_LEFT, dtype=jnp.int32)), _move_to(row, col, 2, 8))

    action = wait_empty
    action = jnp.where(pot_busy, get_plate, action)
    action = jnp.where(assist_partial, assist_fetch, action)
    action = jnp.where(held_raw, plate_pot, action)
    action = jnp.where(held_plate, plate_pot, action)
    action = jnp.where(held_cooked, deliver, action)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    trace = {
        "row": row,
        "col": col,
        "held_plate": held_plate.astype(jnp.int32),
        "held_cooked": held_cooked.astype(jnp.int32),
        "held_raw": held_raw.astype(jnp.int32),
        "partner_empty": partner_empty.astype(jnp.int32),
        "pot_total": pot_total.astype(jnp.int32),
        "assist_one": assist_partial.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
