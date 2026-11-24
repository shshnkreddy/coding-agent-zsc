"""Role-specialized policy for the demo_cook_simple Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'robust_role_split_v1',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

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


def _build_static_templates():
    maps = [[[0 for _ in range(9)] for _ in range(11)] for _ in range(5)]

    def mark(row, col, channel):
        maps[row][col][channel - 20] = 1

    for row, col in (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10),
        (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
        (4, 7), (4, 9), (4, 10),
    ):
        mark(row, col, 20)
    mark(2, 10, 21)
    mark(2, 7, 22)
    mark(0, 5, 23)
    mark(4, 5, 23)
    mark(1, 10, 25)
    mark(3, 10, 25)
    mark(0, 8, 26)
    mark(1, 0, 26)
    mark(3, 0, 27)
    mark(4, 8, 27)
    mark(0, 6, 28)
    mark(4, 6, 28)

    floor = {
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
        (1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9),
    }

    templates = []
    rows = []
    cols = []
    valid = []
    for row in range(5):
        for col in range(11):
            crop = []
            for local_r in range(5):
                line = []
                for local_c in range(5):
                    global_r = row + local_r - 2
                    global_c = col + local_c - 2
                    if 0 <= global_r < 5 and 0 <= global_c < 11:
                        line.append(maps[global_r][global_c])
                    else:
                        line.append([0 for _ in range(9)])
                crop.append(line)
            templates.append(crop)
            rows.append(row)
            cols.append(col)
            valid.append(1 if (row, col) in floor else 0)
    return (
        jnp.asarray(templates, dtype=jnp.int32),
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(cols, dtype=jnp.int32),
        jnp.asarray(valid, dtype=jnp.int32),
    )


def _infer_position(obs):
    static_templates, candidate_rows, candidate_cols, candidate_valid = _build_static_templates()
    static_obs = obs[:, :, 20:29]
    diff = jnp.abs(static_templates - static_obs)
    scores = -jnp.sum(diff, axis=(1, 2, 3))
    scores = jnp.where(candidate_valid > 0, scores, -100000)
    index = jnp.argmax(scores)
    return candidate_rows[index], candidate_cols[index]


def _face(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _held(obs):
    center = obs[2, 2]
    plate = center[5] > 0
    dish = center[6] > 0
    raw_counts = center[7:10]
    raw_total = jnp.sum(raw_counts)
    raw_id = jnp.argmax(raw_counts).astype(jnp.int32)
    return plate, dish, raw_total > 0, raw_id


def _recipe_target(obs, old_target):
    recipe_0 = jnp.sum(obs[:, :, 36])
    recipe_1 = jnp.sum(obs[:, :, 37])
    visible = (recipe_0 + recipe_1) > 0
    target = jnp.where(recipe_1 > recipe_0, 1, 0).astype(jnp.int32)
    return jnp.where(visible, target, old_target), visible


def _pot_state(obs):
    mask = obs[:, :, 22] > 0
    visible = jnp.any(mask)
    cooked = jnp.any((obs[:, :, 30] > 0) & mask)
    count_0 = jnp.sum(jnp.where(mask, obs[:, :, 31], 0))
    count_1 = jnp.sum(jnp.where(mask, obs[:, :, 32], 0))
    count_2 = jnp.sum(jnp.where(mask, obs[:, :, 33], 0))
    total = count_0 + count_1 + count_2
    timer = jnp.max(jnp.where(mask, obs[:, :, 39], 0))
    idle = visible & (timer == 0) & (~cooked)
    accepts_raw = idle & (total < 3)
    partial = accepts_raw & (total > 0)
    return visible, cooked, total, timer, accepts_raw, partial, count_0, count_1, count_2


def _move_toward(row, col, target_row, target_col):
    return jnp.where(
        col < target_col,
        RIGHT,
        jnp.where(
            col > target_col,
            LEFT,
            jnp.where(row < target_row, DOWN, jnp.where(row > target_row, UP, STAY)),
        ),
    )


def _go_interact(row, col, face, station_row, station_col, face_id, orient_action):
    at_station = (row == station_row) & (col == station_col)
    return jnp.where(
        at_station,
        jnp.where(face == face_id, INTERACT, orient_action),
        _move_toward(row, col, station_row, station_col),
    )


def _trace(row, col, action, mode):
    return {
        'row': row.astype(jnp.int32),
        'col': col.astype(jnp.int32),
        'action': action.astype(jnp.int32),
        'mode': mode.astype(jnp.int32),
    }


def init_agent_0():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    base_memory = init_agent_0()
    memory = jnp.where(reset, base_memory, memory)
    row, col = _infer_position(obs)
    face = _face(obs)
    plate, dish, has_raw, raw_id = _held(obs)
    del plate, dish, raw_id
    target, recipe_visible = _recipe_target(obs, memory[0])
    pot_visible, pot_cooked, pot_total, pot_timer, pot_accepts, _, _, _, _ = _pot_state(obs)

    no_target = target < 0
    pot_busy = pot_cooked | (pot_timer > 0) | (pot_total >= 3)
    should_drop = has_raw & (memory[1] > 0) & (pot_visible & (~pot_accepts)) & (target < -1)

    action_to_pot = _go_interact(row, col, face, 2, 6, FACE_RIGHT, RIGHT)
    action_drop = _go_interact(row, col, face, 1, 4, FACE_UP, UP)
    action_get_0 = _go_interact(row, col, face, 1, 1, FACE_LEFT, LEFT)
    action_get_1 = _go_interact(row, col, face, 3, 1, FACE_LEFT, LEFT)
    action_get = jnp.where(target == 1, action_get_1, action_get_0)
    action_home = _move_toward(row, col, 2, 6)

    action = jnp.where(
        should_drop,
        action_drop,
        jnp.where(
            has_raw,
            action_to_pot,
            jnp.where(
                no_target | (pot_visible & pot_busy),
                action_home,
                action_get,
            ),
        ),
    ).astype(jnp.int32)

    mode = jnp.where(
        should_drop,
        4,
        jnp.where(has_raw, 2, jnp.where(no_target | (pot_visible & pot_busy), 0, 1)),
    ).astype(jnp.int32)
    next_drop = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.asarray([target, next_drop], dtype=jnp.int32)
    return action, next_memory, _trace(row, col, action, mode + jnp.where(recipe_visible, 10, 0))


def init_agent_1():
    return jnp.asarray([0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base_memory = init_agent_1()
    memory = jnp.where(reset, base_memory, memory)
    row, col = _infer_position(obs)
    face = _face(obs)
    has_plate, has_dish, has_raw, raw_id = _held(obs)
    pot_visible, pot_cooked, pot_total, pot_timer, pot_accepts, pot_partial, count_0, count_1, _ = _pot_state(obs)

    partial_kind = jnp.where(count_1 > count_0, 1, 0).astype(jnp.int32)
    same_raw = has_raw & (raw_id == partial_kind)
    should_deposit_raw = same_raw & pot_accepts & pot_partial & (pot_total == 1)
    should_drop_raw = has_raw & (~should_deposit_raw)
    should_assist = (~has_raw) & (~has_plate) & (~has_dish) & pot_partial & (pot_total == 1) & (pot_timer == 0)
    need_plate = (~has_raw) & (~has_plate) & (~has_dish) & (~should_assist) & (
        pot_cooked | (pot_timer > 0) | (pot_total >= 3)
    )

    action_to_pot = _go_interact(row, col, face, 2, 8, FACE_LEFT, LEFT)
    action_drop = _go_interact(row, col, face, 1, 9, FACE_UP, UP)
    action_deliver = _go_interact(row, col, face, 2, 9, FACE_RIGHT, RIGHT)
    action_plate_top = _go_interact(row, col, face, 1, 9, FACE_RIGHT, RIGHT)
    action_plate_bottom = _go_interact(row, col, face, 3, 9, FACE_RIGHT, RIGHT)
    action_get_plate = jnp.where(row <= 2, action_plate_top, action_plate_bottom)
    action_get_0 = _go_interact(row, col, face, 1, 8, FACE_UP, UP)
    action_get_1 = _go_interact(row, col, face, 3, 8, FACE_DOWN, DOWN)
    action_get_raw = jnp.where(partial_kind == 1, action_get_1, action_get_0)
    wait_at_pot = jnp.where((row == 2) & (col == 8), STAY, _move_toward(row, col, 2, 8))
    plate_at_pot = jnp.where(pot_cooked, action_to_pot, wait_at_pot)

    action = jnp.where(
        has_dish,
        action_deliver,
        jnp.where(
            has_plate,
            plate_at_pot,
            jnp.where(
                should_deposit_raw,
                action_to_pot,
                jnp.where(
                    should_drop_raw,
                    action_drop,
                    jnp.where(
                        should_assist,
                        action_get_raw,
                        jnp.where(need_plate, action_get_plate, wait_at_pot),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    mode = jnp.where(
        has_dish,
        5,
        jnp.where(
            has_plate,
            4,
            jnp.where(
                should_deposit_raw,
                3,
                jnp.where(should_drop_raw, 6, jnp.where(should_assist, 2, jnp.where(need_plate, 1, 0))),
            ),
        ),
    ).astype(jnp.int32)
    return action, memory, _trace(row, col, action, mode + jnp.where(pot_visible, 10, 0))
