"""Convention-oriented Overcooked policy for demo_cook_wide."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "recipe_seed_and_plate",
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
UNKNOWN_ING = 2


def _static_map():
    cells = [[[0 for _ in range(10)] for _ in range(11)] for _ in range(6)]

    def set_cell(r, c, k):
        cells[r][c][k] = 1

    for r, c in (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10),
        (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8),
        (2, 9), (2, 10), (3, 0), (3, 10), (4, 3), (4, 7),
        (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        (5, 10),
    ):
        set_cell(r, c, 0)
    set_cell(0, 5, 1)
    set_cell(2, 5, 2)
    set_cell(4, 5, 3)
    set_cell(0, 4, 5)
    set_cell(0, 6, 5)
    for r, c in ((1, 3), (4, 0), (4, 10)):
        set_cell(r, c, 6)
    for r, c in ((1, 7), (5, 1), (5, 9)):
        set_cell(r, c, 7)
    for r, c in ((4, 4), (4, 6)):
        set_cell(r, c, 9)
    return cells


def _crop(cells, row, col):
    out = []
    for dr in range(-2, 3):
        row_cells = []
        rr = row + dr
        for dc in range(-2, 3):
            cc = col + dc
            if 0 <= rr < 6 and 0 <= cc < 11:
                row_cells.append(cells[rr][cc])
            else:
                row_cells.append([0 for _ in range(10)])
        out.append(row_cells)
    return out


def _zero_mem():
    return jnp.asarray([0, 2, 0, 0], dtype=jnp.int32)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_or_interact(obs, direction, move_action):
    return jnp.where(_facing(obs) == direction, INTERACT, move_action)


def _top_col(obs):
    at_left = obs[2, 1, 28] > 0
    at_right = obs[2, 3, 29] > 0
    return jnp.where(at_left, 4, jnp.where(at_right, 6, 5)).astype(jnp.int32)


def _top_move_to(col, target):
    return jnp.where(col < target, RIGHT, jnp.where(col > target, LEFT, STAY))


def _top_at_col_action(obs, col, target, direction, move_action):
    return jnp.where(col == target, _face_or_interact(obs, direction, move_action), _top_move_to(col, target))


def _top_add_pot(obs, col):
    return _top_at_col_action(obs, col, 5, DIR_DOWN, DOWN)


def _top_deliver(obs, col):
    return _top_at_col_action(obs, col, 5, DIR_UP, UP)


def _top_pick_plate(obs, col):
    right_plate = col == 6
    at_plate = (col == 4) | right_plate
    target = jnp.where(right_plate, 6, 4)
    action_at = _face_or_interact(obs, DIR_UP, UP)
    return jnp.where(at_plate, action_at, _top_move_to(col, target))


def _top_pick_ing(obs, col, desired):
    pick0 = _top_at_col_action(obs, col, 4, DIR_LEFT, LEFT)
    pick1 = _top_at_col_action(obs, col, 6, DIR_RIGHT, RIGHT)
    return jnp.where(desired == 0, pick0, pick1)


def _top_drop(obs, col):
    target = jnp.where(col > 5, 6, 4)
    return jnp.where(col == target, _face_or_interact(obs, DIR_DOWN, DOWN), _top_move_to(col, target))


def _top_wait_watch(col):
    return _top_move_to(col, 5)


def _pot_features(obs):
    pot = obs[:, :, 24]
    present = jnp.sum(pot) > 0
    cooked = jnp.sum(pot * obs[:, :, 33]) > 0
    c0 = jnp.sum(pot * obs[:, :, 34]).astype(jnp.int32)
    c1 = jnp.sum(pot * obs[:, :, 35]).astype(jnp.int32)
    c2 = jnp.sum(pot * obs[:, :, 36]).astype(jnp.int32)
    c3 = jnp.sum(pot * obs[:, :, 37]).astype(jnp.int32)
    timer = jnp.sum(pot * obs[:, :, 44]).astype(jnp.int32)
    total = c0 + c1 + c2 + c3
    pure0 = (c0 > 0) & (c1 == 0) & (c2 == 0) & (c3 == 0)
    pure1 = (c1 > 0) & (c0 == 0) & (c2 == 0) & (c3 == 0)
    desired = jnp.where(pure0, 0, jnp.where(pure1, 1, UNKNOWN_ING)).astype(jnp.int32)
    idle_space = present & (timer == 0) & (~cooked) & (total < 3)
    return present, cooked, timer, total, desired, idle_space


def init_agent_0():
    return _zero_mem()


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _zero_mem(), memory)
    col = _top_col(obs)
    _, pot_cooked, pot_timer, pot_total, pot_desired, pot_idle_space = _pot_features(obs)
    holding_plate = obs[2, 2, 5] > 0
    holding_cooked = obs[2, 2, 6] > 0
    holding0 = obs[2, 2, 7] > 0
    holding1 = obs[2, 2, 8] > 0
    holding_ing = holding0 | holding1
    held_desired = jnp.where(holding0, 0, jnp.where(holding1, 1, UNKNOWN_ING)).astype(jnp.int32)

    known_need = pot_idle_space & (pot_total > 0) & (pot_desired != UNKNOWN_ING)
    held_fits = known_need & (held_desired == pot_desired)

    empty_action = jnp.where(
        pot_cooked | (pot_timer > 0),
        _top_pick_plate(obs, col),
        jnp.where(known_need, _top_pick_ing(obs, col, pot_desired), _top_wait_watch(col)),
    )
    plate_action = jnp.where(pot_cooked, _top_add_pot(obs, col), _top_wait_watch(col))
    ingredient_action = jnp.where(held_fits, _top_add_pot(obs, col), _top_drop(obs, col))

    action = empty_action
    action = jnp.where(holding_plate, plate_action, action)
    action = jnp.where(holding_ing, ingredient_action, action)
    action = jnp.where(holding_cooked, _top_deliver(obs, col), action)

    next_memory = memory.at[0].add(1)
    return action.astype(jnp.int32), next_memory, {}


def init_agent_1():
    return _zero_mem()


def _locate_agent_1(obs):
    positions = jnp.asarray(
        [
            [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
            [3, 9], [4, 1], [4, 2], [4, 8], [4, 9],
        ],
        dtype=jnp.int32,
    )
    static = _static_map()
    crops = jnp.asarray([_crop(static, p[0], p[1]) for p in (
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        (3, 9), (4, 1), (4, 2), (4, 8), (4, 9),
    )], dtype=jnp.int32)
    diff = jnp.abs(crops - obs[:, :, 22:32])
    err = jnp.sum(diff, axis=(1, 2, 3))
    idx = jnp.argmin(err)
    return positions[idx, 0], positions[idx, 1]


def _bottom_to_pot(row, col):
    return jnp.where(
        row == 4,
        UP,
        jnp.where(col < 5, RIGHT, jnp.where(col > 5, LEFT, STAY)),
    )


def _bottom_to_left_station(row, col):
    return jnp.where(
        row == 4,
        jnp.where(col == 1, STAY, jnp.where(col == 2, LEFT, UP)),
        jnp.where(col == 1, DOWN, jnp.where(col > 1, LEFT, RIGHT)),
    )


def _bottom_to_right_station(row, col):
    return jnp.where(
        row == 4,
        jnp.where(col == 9, STAY, jnp.where(col == 8, RIGHT, UP)),
        jnp.where(col == 9, DOWN, jnp.where(col < 9, RIGHT, LEFT)),
    )


def _bottom_fetch(obs, row, col, desired):
    use_left = col <= 5
    at_left = (row == 4) & (col == 1)
    at_right = (row == 4) & (col == 9)
    face0 = jnp.where(use_left, DIR_LEFT, DIR_RIGHT)
    move0 = jnp.where(use_left, LEFT, RIGHT)
    face_needed = jnp.where(desired == 0, face0, DIR_DOWN)
    move_needed = jnp.where(desired == 0, move0, DOWN)
    pickup = _face_or_interact(obs, face_needed, move_needed)
    nav = jnp.where(use_left, _bottom_to_left_station(row, col), _bottom_to_right_station(row, col))
    return jnp.where((use_left & at_left) | ((~use_left) & at_right), pickup, nav)


def _bottom_add_pot(obs, row, col):
    at_pot = (row == 3) & (col == 5)
    return jnp.where(at_pot, _face_or_interact(obs, DIR_UP, UP), _bottom_to_pot(row, col))


def _bottom_drop(obs, row, col):
    at_drop = (row == 3) & (col == 4)
    nav = jnp.where(row == 4, UP, jnp.where(col < 4, RIGHT, jnp.where(col > 4, LEFT, STAY)))
    return jnp.where(at_drop, _face_or_interact(obs, DIR_UP, UP), nav)


def _recipe_from_obs(obs, old_desired):
    recipe = obs[:, :, 25]
    r0 = jnp.sum(recipe * obs[:, :, 40])
    r1 = jnp.sum(recipe * obs[:, :, 41])
    return jnp.where(r0 > 0, 0, jnp.where(r1 > 0, 1, old_desired)).astype(jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _zero_mem(), memory)
    row, col = _locate_agent_1(obs)
    desired = _recipe_from_obs(obs, memory[1])
    pot_present, pot_cooked, pot_timer, pot_total, pot_desired, pot_idle_space = _pot_features(obs)

    holding0 = obs[2, 2, 7] > 0
    holding1 = obs[2, 2, 8] > 0
    holding_ing = holding0 | holding1
    held = jnp.where(holding0, 0, jnp.where(holding1, 1, UNKNOWN_ING)).astype(jnp.int32)
    known_recipe = desired != UNKNOWN_ING
    pot_compatible = (pot_total == 0) | (pot_desired == desired)
    should_add = pot_present & pot_idle_space & pot_compatible & known_recipe
    should_wait = pot_present & ((pot_timer > 0) | pot_cooked | (~pot_compatible) | (~pot_idle_space))
    seeded_wait = pot_present & pot_idle_space & pot_compatible & known_recipe & (pot_total > 0)
    same_seed_state = seeded_wait & (pot_total == memory[2])
    stall_count = jnp.where(same_seed_state, jnp.minimum(memory[3] + 1, 1000), 0).astype(jnp.int32)
    fallback_fetch = seeded_wait & (stall_count > 8)
    should_fetch = (should_add & (pot_total == 0)) | (~pot_present) | fallback_fetch

    empty_action = jnp.where(
        known_recipe,
        jnp.where(
            should_fetch,
            _bottom_fetch(obs, row, col, desired),
            jnp.where(should_wait, _bottom_to_pot(row, col), _bottom_to_pot(row, col)),
        ),
        _bottom_to_pot(row, col),
    )
    holding_action = jnp.where(
        known_recipe & (held != desired),
        _bottom_drop(obs, row, col),
        jnp.where(should_add | (~pot_present), _bottom_add_pot(obs, row, col), _bottom_to_pot(row, col)),
    )
    action = jnp.where(holding_ing, holding_action, empty_action)
    next_memory = jnp.asarray([memory[0] + 1, desired, pot_total, stall_count], dtype=jnp.int32)
    return action.astype(jnp.int32), next_memory, {}
