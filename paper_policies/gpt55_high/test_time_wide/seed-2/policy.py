"""Convention-robust deterministic policy for the wide Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'legible_signal_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _static_map():
    grid = [[[0 for _ in range(10)] for _ in range(6)] for _ in range(7)]
    for r, c in (
        (0, 0), (0, 1), (0, 4), (0, 5), (3, 0), (3, 2), (3, 4),
        (3, 5), (5, 0), (5, 5), (6, 0), (6, 1), (6, 3), (6, 4), (6, 5)
    ):
        grid[r][c][0] = 1
    grid[0][2][1] = 1
    for r, c in ((3, 1), (3, 3)):
        grid[r][c][2] = 1
    grid[6][2][3] = 1
    grid[0][3][5] = 1
    for r, c in ((1, 0), (1, 5)):
        grid[r][c][6] = 1
    for r, c in ((2, 0), (2, 5)):
        grid[r][c][7] = 1
    for r, c in ((4, 0), (4, 5)):
        grid[r][c][9] = 1
    return grid


def _candidate_crops():
    static = _static_map()
    crops = []
    rows = []
    cols = []
    for r in range(7):
        for c in range(6):
            crop = []
            for i in range(5):
                line = []
                for j in range(5):
                    gr = r + i - 2
                    gc = c + j - 2
                    if 0 <= gr < 7 and 0 <= gc < 6:
                        line.append(static[gr][gc])
                    else:
                        line.append([0 for _ in range(10)])
                crop.append(line)
            crops.append(crop)
            rows.append(r)
            cols.append(c)
    return (
        jnp.asarray(crops, dtype=jnp.int32),
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(cols, dtype=jnp.int32),
    )


def _initial_top_memory():
    return jnp.zeros((4,), dtype=jnp.int32)


def _locate(obs):
    candidate_crops, candidate_rows, candidate_cols = _candidate_crops()
    static_obs = obs[:, :, 22:32]
    diffs = jnp.sum(jnp.abs(candidate_crops - static_obs[None, :, :, :]), axis=(1, 2, 3))
    idx = jnp.argmin(diffs)
    return candidate_rows[idx], candidate_cols[idx]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _held_block(obs):
    return obs[2, 2, 5:11]


def _move_to(row, col, target_row, target_col):
    row_action = jnp.where(row < target_row, DOWN, UP)
    col_action = jnp.where(col < target_col, RIGHT, LEFT)
    return jnp.where(row != target_row, row_action, jnp.where(col != target_col, col_action, STAY))


def _go_interact(obs, row, col, target_row, target_col, face_dir):
    at_target = (row == target_row) & (col == target_col)
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[face_dir]
    action_at_target = jnp.where(_facing(obs) == face_dir, INTERACT, face_action)
    return jnp.where(at_target, action_at_target, _move_to(row, col, target_row, target_col))


def _go_face_or_stay(obs, row, col, target_row, target_col, face_dir):
    at_target = (row == target_row) & (col == target_col)
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[face_dir]
    action_at_target = jnp.where(_facing(obs) == face_dir, STAY, face_action)
    return jnp.where(at_target, action_at_target, _move_to(row, col, target_row, target_col))


def _cell_channel(obs, self_row, self_col, global_row, global_col, channel):
    local_row = global_row - self_row + 2
    local_col = global_col - self_col + 2
    rows5 = jnp.arange(5, dtype=jnp.int32)[:, None]
    cols5 = jnp.arange(5, dtype=jnp.int32)[None, :]
    mask = (rows5 == local_row) & (cols5 == local_col)
    return jnp.sum(obs[:, :, channel] * mask.astype(obs.dtype))


def _pot_values(obs, self_row, self_col, target):
    pot_col = jnp.where(target == 0, 1, 3).astype(jnp.int32)
    cooked = _cell_channel(obs, self_row, self_col, 3, pot_col, 33)
    ing0 = _cell_channel(obs, self_row, self_col, 3, pot_col, 34)
    ing1 = _cell_channel(obs, self_row, self_col, 3, pot_col, 35)
    timer = _cell_channel(obs, self_row, self_col, 3, pot_col, 44)
    total = ing0 + ing1
    target_count = jnp.where(target == 0, ing0, ing1)
    return total, target_count, cooked, timer


def _partner_signal(obs, self_row, self_col):
    visible = jnp.sum(obs[:, :, 11]) > 0
    flat = jnp.argmax(jnp.reshape(obs[:, :, 11], (-1,))).astype(jnp.int32)
    local_row = flat // 5
    local_col = flat - local_row * 5
    partner_row = self_row + local_row - 2
    partner_col = self_col + local_col - 2
    at_signal_row = partner_row == 4
    at_left = partner_col == 1
    at_right = partner_col == 3
    signal_visible = visible & at_signal_row & (at_left | at_right)
    signal_target = jnp.where(at_right, 1, 0).astype(jnp.int32)
    return signal_visible, signal_target


def _ingredient_pickup_action(obs, row, col, target):
    supply_row = jnp.where(target == 0, 1, 2).astype(jnp.int32)
    supply_col = jnp.where(target == 0, 1, 4).astype(jnp.int32)
    face_dir = jnp.where(target == 0, 3, 2).astype(jnp.int32)
    return _go_interact(obs, row, col, supply_row, supply_col, face_dir)


def _top_action(obs, memory):
    row, col = _locate(obs)
    signal_visible, signal_target = _partner_signal(obs, row, col)
    route_alt = memory[3] == 1

    target = jnp.where(signal_visible, signal_target, memory[0])
    valid = jnp.where(signal_visible, 1, memory[1])
    wait_count = jnp.where(valid == 1, memory[2], memory[2] + 1)
    use_fallback = (valid == 0) & (wait_count > 30)
    target = jnp.where(use_fallback, 0, target).astype(jnp.int32)
    valid = jnp.where(use_fallback, 1, valid).astype(jnp.int32)

    held = _held_block(obs)
    has_plate = held[0] > 0
    has_cooked = held[1] > 0
    held_ing0 = held[2] > 0
    held_ing1 = held[3] > 0
    has_ingredient = held_ing0 | held_ing1
    held_target = jnp.where(held_ing1, 1, 0).astype(jnp.int32)

    stand_col = jnp.where(target == 0, 1, 3).astype(jnp.int32)
    held_stand_col = jnp.where(held_target == 0, 1, 3).astype(jnp.int32)
    total, target_count, cooked, timer = _pot_values(obs, row, col, target)
    alt_target = 1 - target
    alt_total, _, alt_cooked, alt_timer = _pot_values(obs, row, col, alt_target)
    pot_needs_ingredient = (total < 3) & (cooked == 0) & (timer == 0)
    pot_ready = cooked > 0
    alt_needs_ingredient = (alt_total < 3) & (alt_cooked == 0) & (alt_timer == 0)
    prep_alt = ((timer > 7) | route_alt) & alt_needs_ingredient

    observe_action = _go_face_or_stay(obs, row, col, 2, 2, 1)
    add_action = _go_interact(obs, row, col, 2, held_stand_col, 1)
    plate_action = _go_interact(obs, row, col, 1, 3, 0)
    scoop_action = _go_interact(obs, row, col, 2, stand_col, 1)
    deliver_action = _go_interact(obs, row, col, 1, 2, 0)
    fetch_action = _ingredient_pickup_action(obs, row, col, target)
    fetch_alt_action = _ingredient_pickup_action(obs, row, col, alt_target)

    empty_action = jnp.where(
        valid == 0,
        observe_action,
        jnp.where(
            pot_ready,
            plate_action,
            jnp.where(prep_alt, fetch_alt_action, jnp.where(pot_needs_ingredient, fetch_action, plate_action)),
        ),
    )
    action = jnp.where(
        has_cooked,
        deliver_action,
        jnp.where(
            has_plate,
            jnp.where(pot_ready, scoop_action, _go_face_or_stay(obs, row, col, 2, stand_col, 1)),
            jnp.where(has_ingredient, add_action, empty_action),
        ),
    ).astype(jnp.int32)

    just_delivery = has_cooked & (row == 1) & (col == 2) & (_facing(obs) == 0) & (action == INTERACT)
    next_valid = jnp.where(just_delivery, 0, valid).astype(jnp.int32)
    next_wait = jnp.where((signal_visible | just_delivery), 0, wait_count).astype(jnp.int32)
    any_held = has_ingredient | has_plate | has_cooked
    continue_route = prep_alt & (action == fetch_alt_action) & (~any_held) & (next_valid == 1)
    next_route = jnp.where(continue_route, 1, 0).astype(jnp.int32)
    next_memory = jnp.asarray([target, next_valid, next_wait, next_route], dtype=jnp.int32)
    trace = {
        'row': row,
        'col': col,
        'target': target,
        'valid': next_valid,
        'pot_total': total,
        'pot_cooked': cooked,
        'pot_timer': timer,
        'alt_total': alt_total,
    }
    return action, next_memory, trace


def init_agent_0():
    return _initial_top_memory()


def agent_0_act(obs, reset, memory, rng):
    del rng
    start = _initial_top_memory()
    memory = jnp.where(reset, start, memory)
    return _top_action(obs, memory)


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def _bottom_target(obs):
    recipe0 = jnp.sum(obs[:, :, 40])
    recipe1 = jnp.sum(obs[:, :, 41])
    return jnp.where(recipe1 > recipe0, 1, 0).astype(jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del reset, rng
    row, col = _locate(obs)
    target = _bottom_target(obs)
    target_col = jnp.where(target == 0, 1, 3).astype(jnp.int32)
    action = _go_face_or_stay(obs, row, col, 4, target_col, 0).astype(jnp.int32)
    trace = {'row': row, 'col': col, 'target': target}
    return action, target, trace
