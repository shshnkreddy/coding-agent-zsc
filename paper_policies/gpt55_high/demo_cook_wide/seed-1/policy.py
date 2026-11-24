"""Deterministic Overcooked V2 policy for the demo_cook_wide layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'recipe_loader_finisher', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

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


def _static_map():
    grid = jnp.zeros((6, 11, 10), dtype=jnp.int32)
    walls = (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10),
        (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
        (3, 0), (3, 10), (4, 3), (4, 7),
        (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10),
    )
    for r, c in walls:
        grid = grid.at[r, c, 0].set(1)
    for r, c in ((0, 5),):
        grid = grid.at[r, c, 1].set(1)
    for r, c in ((2, 5),):
        grid = grid.at[r, c, 2].set(1)
    for r, c in ((4, 5),):
        grid = grid.at[r, c, 3].set(1)
    for r, c in ((0, 4), (0, 6)):
        grid = grid.at[r, c, 5].set(1)
    for r, c in ((1, 3), (4, 0), (4, 10)):
        grid = grid.at[r, c, 6].set(1)
    for r, c in ((1, 7), (5, 1), (5, 9)):
        grid = grid.at[r, c, 7].set(1)
    for r, c in ((4, 4), (4, 6)):
        grid = grid.at[r, c, 9].set(1)
    return grid


AGENT0_POSITIONS = ((1, 4), (1, 5), (1, 6))
AGENT1_POSITIONS = (
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 1), (4, 2), (4, 8), (4, 9),
)


def _localize(obs, positions):
    padded_static = jnp.pad(_static_map(), ((2, 2), (2, 2), (0, 0)))
    patches = jnp.stack([padded_static[r:r + 5, c:c + 5, :] for r, c in positions])
    position_array = jnp.asarray(positions, dtype=jnp.int32)
    visible_static = obs[:, :, 22:32].astype(jnp.int32)
    matches = jnp.sum((patches == visible_static[None, :, :, :]).astype(jnp.int32), axis=(1, 2, 3))
    return position_array[jnp.argmax(matches)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5].astype(jnp.int32)).astype(jnp.int32)


def _face_action(direction):
    return jnp.where(
        direction == DIR_UP,
        UP,
        jnp.where(direction == DIR_DOWN, DOWN, jnp.where(direction == DIR_RIGHT, RIGHT, LEFT)),
    )


def _top_move_to_col(pos, target_col):
    col = pos[1]
    return jnp.where(col < target_col, RIGHT, jnp.where(col > target_col, LEFT, STAY))


def _top_interact_at(pos, face, target_col, direction):
    at_target = pos[1] == target_col
    turn_or_use = jnp.where(face == direction, INTERACT, _face_action(direction))
    return jnp.where(at_target, turn_or_use, _top_move_to_col(pos, target_col))


def _bottom_move_to_center(pos):
    row = pos[0]
    col = pos[1]
    return jnp.where(
        row > 3,
        UP,
        jnp.where(col < 5, RIGHT, jnp.where(col > 5, LEFT, STAY)),
    )


def _bottom_move_to_left_supply(pos):
    row = pos[0]
    col = pos[1]
    return jnp.where(
        (row > 3) & (col > 2),
        UP,
        jnp.where(
            row == 3,
            jnp.where(col > 1, LEFT, DOWN),
            jnp.where(col > 1, LEFT, STAY),
        ),
    )


def _bottom_move_to_drop(pos):
    row = pos[0]
    col = pos[1]
    return jnp.where(
        row > 3,
        UP,
        jnp.where(col < 4, RIGHT, jnp.where(col > 4, LEFT, STAY)),
    )


def _bottom_interact_pot(pos, face):
    at_center = (pos[0] == 3) & (pos[1] == 5)
    turn_or_use = jnp.where(face == DIR_UP, INTERACT, UP)
    return jnp.where(at_center, turn_or_use, _bottom_move_to_center(pos))


def _bottom_interact_left_supply(pos, face, ingredient):
    at_supply_spot = (pos[0] == 4) & (pos[1] == 1)
    wanted_dir = jnp.where(ingredient == 0, DIR_LEFT, DIR_DOWN)
    turn_or_use = jnp.where(face == wanted_dir, INTERACT, _face_action(wanted_dir))
    return jnp.where(at_supply_spot, turn_or_use, _bottom_move_to_left_supply(pos))


def _bottom_drop_wrong_raw(pos, face):
    at_drop = (pos[0] == 3) & (pos[1] == 4)
    turn_or_use = jnp.where(face == DIR_UP, INTERACT, UP)
    return jnp.where(at_drop, turn_or_use, _bottom_move_to_drop(pos))


def _held(obs):
    center = obs[2, 2, :]
    plate = center[5] > 0
    cooked = center[6] > 0
    ing0 = center[7] > 0
    ing1 = center[8] > 0
    has_counts = ing0 | ing1 | (center[9] > 0) | (center[10] > 0)
    raw = has_counts & (~plate) & (~cooked)
    dish = plate & cooked
    empty = ~(plate | raw | cooked)
    raw_kind = jnp.where(ing1, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    return empty, plate & ~cooked, dish, raw, raw_kind


def _pot_state(obs):
    mask = (obs[:, :, 24] > 0).astype(jnp.int32)
    cooked = jnp.sum(obs[:, :, 33] * mask) > 0
    count0 = jnp.sum(obs[:, :, 34] * mask)
    count1 = jnp.sum(obs[:, :, 35] * mask)
    total = count0 + count1 + jnp.sum(obs[:, :, 36] * mask) + jnp.sum(obs[:, :, 37] * mask)
    timer = jnp.sum(obs[:, :, 44] * mask)
    return cooked, total, timer


def _recipe_target(obs, previous):
    r0 = jnp.sum(obs[:, :, 40])
    r1 = jnp.sum(obs[:, :, 41])
    visible = (r0 + r1) > 0
    seen_target = jnp.where(r1 > r0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    return jnp.where(visible, seen_target, previous)


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    pos = _localize(obs, AGENT0_POSITIONS)
    face = _facing(obs)
    empty, holding_plate, holding_dish, raw, _ = _held(obs)
    pot_cooked, pot_total, pot_timer = _pot_state(obs)
    pot_busy = (pot_total >= 3) | (pot_timer > 0) | pot_cooked

    plate_col = jnp.where(pos[1] > 5, jnp.asarray(6, dtype=jnp.int32), jnp.asarray(4, dtype=jnp.int32))
    get_plate = _top_interact_at(pos, face, plate_col, DIR_UP)
    take_pot = _top_interact_at(pos, face, jnp.asarray(5, dtype=jnp.int32), DIR_DOWN)
    deliver = _top_interact_at(pos, face, jnp.asarray(5, dtype=jnp.int32), DIR_UP)
    wait_center = jnp.where(pos[1] == 5, jnp.where(face == DIR_DOWN, STAY, DOWN), _top_move_to_col(pos, jnp.asarray(5, dtype=jnp.int32)))

    action = wait_center
    action = jnp.where(empty & pot_busy, get_plate, action)
    action = jnp.where(holding_plate, jnp.where(pot_cooked, take_pot, wait_center), action)
    action = jnp.where(holding_dish, deliver, action)
    action = jnp.where(raw, wait_center, action)
    trace = {
        'row': pos[0],
        'col': pos[1],
        'pot_total': pot_total.astype(jnp.int32),
        'pot_cooked': pot_cooked.astype(jnp.int32),
    }
    return action.astype(jnp.int32), memory, trace


def init_agent_1():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base_memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    pos = _localize(obs, AGENT1_POSITIONS)
    face = _facing(obs)
    target = _recipe_target(obs, base_memory[0])
    next_memory = base_memory.at[0].set(target)
    empty, _, _, raw, raw_kind = _held(obs)
    pot_cooked, pot_total, pot_timer = _pot_state(obs)
    pot_open = (pot_timer == 0) & (~pot_cooked) & (pot_total < 3)
    raw_matches = raw_kind == target

    fetch = _bottom_interact_left_supply(pos, face, target)
    insert = _bottom_interact_pot(pos, face)
    idle = jnp.where((pos[0] == 3) & (pos[1] == 5), STAY, _bottom_move_to_center(pos))
    drop = _bottom_drop_wrong_raw(pos, face)

    action = idle
    action = jnp.where(empty & pot_open, fetch, action)
    action = jnp.where(raw & raw_matches & pot_open, insert, action)
    action = jnp.where(raw & (~raw_matches), drop, action)
    trace = {
        'row': pos[0],
        'col': pos[1],
        'target': target,
        'pot_total': pot_total.astype(jnp.int32),
        'pot_cooked': pot_cooked.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
