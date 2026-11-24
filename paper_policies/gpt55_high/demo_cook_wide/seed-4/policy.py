"""Geometry-based Overcooked V2 policy for the demo_cook_wide layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'public_geometry_roles',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
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

CANDIDATE_POS_LIST = (
    (1, 4), (1, 5), (1, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 1), (4, 2), (4, 8), (4, 9),
)


def _make_static_map():
    grid = [[0] * 10 for _ in range(6 * 11)]

    def mark(row, col, channel):
        grid[row * 11 + col][channel] = 1

    for row, col in (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10),
        (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
        (3, 0), (3, 10), (4, 3), (4, 7),
        (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10),
    ):
        mark(row, col, 0)

    mark(0, 5, 1)
    mark(2, 5, 2)
    mark(4, 5, 3)
    mark(0, 4, 5)
    mark(0, 6, 5)
    mark(1, 3, 6)
    mark(4, 0, 6)
    mark(4, 10, 6)
    mark(1, 7, 7)
    mark(5, 1, 7)
    mark(5, 9, 7)
    mark(4, 4, 9)
    mark(4, 6, 9)
    return jnp.asarray(grid, dtype=jnp.int32).reshape((6, 11, 10))


def _crop_for(static_map, row, col):
    cells = []
    for dr in range(-2, 3):
        row_cells = []
        for dc in range(-2, 3):
            rr = row + dr
            cc = col + dc
            if 0 <= rr < 6 and 0 <= cc < 11:
                row_cells.append(static_map[rr, cc])
            else:
                row_cells.append(jnp.zeros((10,), dtype=jnp.int32))
        cells.append(jnp.stack(row_cells, axis=0))
    return jnp.stack(cells, axis=0)

def _position_from_obs(obs):
    static_map = _make_static_map()
    candidate_pos = jnp.asarray(CANDIDATE_POS_LIST, dtype=jnp.int32)
    static_crops = jnp.stack(
        [_crop_for(static_map, row, col) for row, col in CANDIDATE_POS_LIST],
        axis=0,
    )
    static_obs = obs[:, :, 22:32]
    diff = jnp.sum(jnp.abs(static_crops - static_obs[None, :, :, :]), axis=(1, 2, 3))
    idx = jnp.argmin(diff)
    return candidate_pos[idx]


def _facing(obs):
    dirs = obs[2, 2, 1:5]
    return jnp.asarray(jnp.argmax(dirs), dtype=jnp.int32)


def _face_or_interact(obs, desired_dir):
    face = _facing(obs)
    move = jnp.select(
        [desired_dir == DIR_UP, desired_dir == DIR_DOWN, desired_dir == DIR_RIGHT, desired_dir == DIR_LEFT],
        [UP, DOWN, RIGHT, LEFT],
        default=STAY,
    )
    return jnp.where(face == desired_dir, INTERACT, move)


def _move_top_to(pos, target_col):
    col = pos[1]
    return jnp.where(col < target_col, RIGHT, jnp.where(col > target_col, LEFT, STAY))


def _move_lower_to_pot(pos):
    row = pos[0]
    col = pos[1]
    horizontal = jnp.where(col < 5, RIGHT, jnp.where(col > 5, LEFT, STAY))
    return jnp.where(row == 4, UP, horizontal)


def _move_lower_to_left_station(pos):
    row = pos[0]
    col = pos[1]
    from_row3 = jnp.where(col > 1, LEFT, DOWN)
    from_row4 = jnp.where(col > 1, LEFT, STAY)
    return jnp.where(row == 3, from_row3, from_row4)


def _held_plate(obs):
    return obs[2, 2, 5] > 0


def _held_cooked(obs):
    return obs[2, 2, 6] > 0


def _held_raw(obs):
    return jnp.sum(obs[2, 2, 7:11]) > 0


def _held_ingredient(obs):
    return jnp.asarray(jnp.argmax(obs[2, 2, 7:11]), dtype=jnp.int32)


def _pot_features(obs):
    mask = obs[:, :, 24] > 0
    visible = jnp.any(mask)
    mask_i = mask.astype(jnp.int32)
    cooked = jnp.max(obs[:, :, 33] * mask_i)
    count0 = jnp.max(obs[:, :, 34] * mask_i)
    count1 = jnp.max(obs[:, :, 35] * mask_i)
    count2 = jnp.max(obs[:, :, 36] * mask_i)
    count3 = jnp.max(obs[:, :, 37] * mask_i)
    total = count0 + count1 + count2 + count3
    timer = jnp.max(obs[:, :, 44] * mask_i)
    return visible, cooked, total, timer, count0, count1


def _recipe_features(obs, memory_target, memory_seen):
    recipe0 = jnp.max(obs[:, :, 40])
    recipe1 = jnp.max(obs[:, :, 41])
    visible = (recipe0 + recipe1) > 0
    observed_target = jnp.where(recipe1 > recipe0, 1, 0).astype(jnp.int32)
    target = jnp.where(visible, observed_target, memory_target)
    seen = jnp.where(visible, jnp.asarray(1, dtype=jnp.int32), memory_seen)
    return target, seen, visible


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    base_memory = jnp.where(reset, init_agent_0(), memory)
    pos = _position_from_obs(obs)
    held_plate = _held_plate(obs)
    held_cooked = _held_cooked(obs)
    holding_raw = _held_raw(obs)
    held_ing = _held_ingredient(obs)
    _, pot_cooked, pot_total, pot_timer, pot_count0, pot_count1 = _pot_features(obs)

    at_center = (pos[0] == 1) & (pos[1] == 5)
    at_left_plate = (pos[0] == 1) & (pos[1] == 4)
    at_right_plate = (pos[0] == 1) & (pos[1] == 6)
    at_left_supply = at_left_plate
    at_right_supply = at_right_plate

    deliver_action = jnp.where(at_center, _face_or_interact(obs, DIR_UP), _move_top_to(pos, 5))
    plate_pot_action = jnp.where(at_center, _face_or_interact(obs, DIR_DOWN), _move_top_to(pos, 5))
    get_left_plate = jnp.where(at_left_plate, _face_or_interact(obs, DIR_UP), _move_top_to(pos, 4))
    get_right_plate = jnp.where(at_right_plate, _face_or_interact(obs, DIR_UP), _move_top_to(pos, 6))
    get_plate_action = jnp.where(pos[1] <= 5, get_left_plate, get_right_plate)
    pot_ingredient = jnp.where(pot_count1 > pot_count0, 1, 0).astype(jnp.int32)
    pot_raw_waiting = (pot_cooked == 0) & (pot_timer == 0) & (pot_total == 1)
    get_ing0_action = jnp.where(at_left_supply, _face_or_interact(obs, DIR_LEFT), _move_top_to(pos, 4))
    get_ing1_action = jnp.where(at_right_supply, _face_or_interact(obs, DIR_RIGHT), _move_top_to(pos, 6))
    get_ingredient_action = jnp.where(pot_ingredient == 0, get_ing0_action, get_ing1_action)
    cooking_needs_plate = (pot_timer > 0) | (pot_total >= 3)
    empty_action = jnp.where(
        pot_cooked > 0,
        get_plate_action,
        jnp.where(
            pot_raw_waiting,
            get_ingredient_action,
            jnp.where(cooking_needs_plate, get_plate_action, jnp.where(at_center, STAY, _move_top_to(pos, 5))),
        ),
    )
    raw_can_deposit = (pot_cooked == 0) & (pot_timer == 0) & (pot_total > 0) & (pot_total < 3) & (held_ing == pot_ingredient)
    discard_col = jnp.where(held_ing == 0, 4, 6)
    discard_dir = jnp.where(held_ing == 0, DIR_DOWN, DIR_DOWN)
    discard_action = jnp.where(pos[1] == discard_col, _face_or_interact(obs, discard_dir), _move_top_to(pos, discard_col))
    raw_action = jnp.where(raw_can_deposit, plate_pot_action, discard_action)

    action = jnp.where(
        held_cooked,
        deliver_action,
        jnp.where(held_plate, plate_pot_action, jnp.where(holding_raw, raw_action, empty_action)),
    )
    trace = {
        'row': pos[0],
        'col': pos[1],
        'pot_cooked': pot_cooked.astype(jnp.int32),
        'held_mode': (held_plate.astype(jnp.int32) + 2 * held_cooked.astype(jnp.int32) + 4 * holding_raw.astype(jnp.int32)),
    }
    return action.astype(jnp.int32), base_memory, trace


def init_agent_1():
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    old = jnp.where(reset, init_agent_1(), memory)
    pos = _position_from_obs(obs)
    target, seen, recipe_visible = _recipe_features(obs, old[0], old[1])
    pot_visible, pot_cooked, pot_total, pot_timer, _, _ = _pot_features(obs)
    loads = jnp.where(pot_visible, pot_total, old[2]).astype(jnp.int32)

    holding_raw = _held_raw(obs)
    held_ing = _held_ingredient(obs)
    at_pot = (pos[0] == 3) & (pos[1] == 5)
    at_station = (pos[0] == 4) & (pos[1] == 1)
    pot_busy = (pot_cooked > 0) | (pot_timer > 0) | (loads >= 3)

    seek_recipe_action = jnp.where(at_pot, STAY, _move_lower_to_pot(pos))
    deposit_action = jnp.where(at_pot, _face_or_interact(obs, DIR_UP), _move_lower_to_pot(pos))
    face_station = jnp.where(target == 0, DIR_LEFT, DIR_DOWN)
    pickup_action = jnp.where(at_station, _face_or_interact(obs, face_station), _move_lower_to_left_station(pos))
    wait_action = jnp.where(at_pot, STAY, _move_lower_to_pot(pos))

    can_deposit = holding_raw & (held_ing == target) & (~pot_busy)
    raw_action = jnp.where(can_deposit, deposit_action, wait_action)
    empty_action = jnp.where(seen == 0, seek_recipe_action, jnp.where(pot_busy, wait_action, pickup_action))
    action = jnp.where(holding_raw, raw_action, empty_action)

    did_deposit = holding_raw & can_deposit & at_pot & (_facing(obs) == DIR_UP)
    next_loads = jnp.where(did_deposit, jnp.minimum(loads + 1, 3), loads)
    next_loads = jnp.where((pot_visible & (pot_total == 0) & (pot_timer == 0) & (pot_cooked == 0)), 0, next_loads)
    next_memory = jnp.asarray([target, seen, next_loads], dtype=jnp.int32)

    trace = {
        'row': pos[0],
        'col': pos[1],
        'target': target,
        'seen': seen,
        'loads': next_loads,
        'pot_busy': pot_busy.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
