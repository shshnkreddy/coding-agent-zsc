"""Convention-light Overcooked V2 policy for grounded_coord_ring."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "inner_solo_cook_v1",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}

STAY = 4
INTERACT = 5
FACE_UP = 0
FACE_DOWN = 1
FACE_RIGHT = 2
FACE_LEFT = 3
BIG = 10000

LAYOUT = (
    "WWW2R2WWW",
    "W       W",
    "W WWLWW W",
    "2 0   B 2",
    "RAXAP X R",
    "2 1   B 2",
    "W WWLWW W",
    "W       W",
    "WWW2R2WWW",
)

STATIC_CHANNEL_BY_CHAR = {
    "W": 0,
    "X": 1,
    "P": 2,
    "R": 3,
    "L": 4,
    "B": 5,
    "0": 6,
    "1": 7,
    "2": 8,
}

INNER_WALKABLE = (
    (False, False, False, False, False, False, False, False, False),
    (False, False, False, False, False, False, False, False, False),
    (False, False, False, False, False, False, False, False, False),
    (False, False, False, True, True, True, False, False, False),
    (False, False, False, True, False, True, False, False, False),
    (False, False, False, True, True, True, False, False, False),
    (False, False, False, False, False, False, False, False, False),
    (False, False, False, False, False, False, False, False, False),
    (False, False, False, False, False, False, False, False, False),
)

INNER_DISTS = (
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, 1, 0, 1, BIG, BIG, BIG),
        (BIG, BIG, BIG, 2, BIG, 2, BIG, BIG, BIG),
        (BIG, BIG, BIG, 3, 4, 3, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, 0, 1, 2, BIG, BIG, BIG),
        (BIG, BIG, BIG, 1, BIG, 3, BIG, BIG, BIG),
        (BIG, BIG, BIG, 2, 3, 4, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, 2, 3, 4, BIG, BIG, BIG),
        (BIG, BIG, BIG, 1, BIG, 3, BIG, BIG, BIG),
        (BIG, BIG, BIG, 0, 1, 2, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, 1, 2, 3, BIG, BIG, BIG),
        (BIG, BIG, BIG, 0, BIG, 4, BIG, BIG, BIG),
        (BIG, BIG, BIG, 1, 2, 3, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, 2, 1, 0, BIG, BIG, BIG),
        (BIG, BIG, BIG, 3, BIG, 1, BIG, BIG, BIG),
        (BIG, BIG, BIG, 4, 3, 2, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, 3, 2, 1, BIG, BIG, BIG),
        (BIG, BIG, BIG, 4, BIG, 0, BIG, BIG, BIG),
        (BIG, BIG, BIG, 3, 2, 1, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, 3, 4, 3, BIG, BIG, BIG),
        (BIG, BIG, BIG, 2, BIG, 2, BIG, BIG, BIG),
        (BIG, BIG, BIG, 1, 0, 1, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, 4, 3, 2, BIG, BIG, BIG),
        (BIG, BIG, BIG, 3, BIG, 1, BIG, BIG, BIG),
        (BIG, BIG, BIG, 2, 1, 0, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
)

OUTER_DISTS = (
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, 3, 4, 5, 6, 7, 8, 9, BIG),
        (BIG, 2, BIG, BIG, BIG, BIG, BIG, 10, BIG),
        (BIG, 1, BIG, BIG, BIG, BIG, BIG, 11, BIG),
        (BIG, 0, BIG, BIG, BIG, BIG, BIG, 12, BIG),
        (BIG, 1, BIG, BIG, BIG, BIG, BIG, 11, BIG),
        (BIG, 2, BIG, BIG, BIG, BIG, BIG, 10, BIG),
        (BIG, 3, 4, 5, 6, 7, 8, 9, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, 2, 3, 4, 5, 6, 7, 8, BIG),
        (BIG, 1, BIG, BIG, BIG, BIG, BIG, 9, BIG),
        (BIG, 0, BIG, BIG, BIG, BIG, BIG, 10, BIG),
        (BIG, 1, BIG, BIG, BIG, BIG, BIG, 11, BIG),
        (BIG, 2, BIG, BIG, BIG, BIG, BIG, 12, BIG),
        (BIG, 3, BIG, BIG, BIG, BIG, BIG, 11, BIG),
        (BIG, 4, 5, 6, 7, 8, 9, 10, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, 4, 5, 6, 7, 8, 9, 10, BIG),
        (BIG, 3, BIG, BIG, BIG, BIG, BIG, 11, BIG),
        (BIG, 2, BIG, BIG, BIG, BIG, BIG, 12, BIG),
        (BIG, 1, BIG, BIG, BIG, BIG, BIG, 11, BIG),
        (BIG, 0, BIG, BIG, BIG, BIG, BIG, 10, BIG),
        (BIG, 1, BIG, BIG, BIG, BIG, BIG, 9, BIG),
        (BIG, 2, 3, 4, 5, 6, 7, 8, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, 2, 1, 0, 1, 2, 3, 4, BIG),
        (BIG, 3, BIG, BIG, BIG, BIG, BIG, 5, BIG),
        (BIG, 4, BIG, BIG, BIG, BIG, BIG, 6, BIG),
        (BIG, 5, BIG, BIG, BIG, BIG, BIG, 7, BIG),
        (BIG, 6, BIG, BIG, BIG, BIG, BIG, 8, BIG),
        (BIG, 7, BIG, BIG, BIG, BIG, BIG, 9, BIG),
        (BIG, 8, 9, 10, 11, 12, 11, 10, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
    (
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
        (BIG, 8, 9, 10, 11, 12, 11, 10, BIG),
        (BIG, 7, BIG, BIG, BIG, BIG, BIG, 9, BIG),
        (BIG, 6, BIG, BIG, BIG, BIG, BIG, 8, BIG),
        (BIG, 5, BIG, BIG, BIG, BIG, BIG, 7, BIG),
        (BIG, 4, BIG, BIG, BIG, BIG, BIG, 6, BIG),
        (BIG, 3, BIG, BIG, BIG, BIG, BIG, 5, BIG),
        (BIG, 2, 1, 0, 1, 2, 3, 4, BIG),
        (BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG),
    ),
)

T_READ = 0
T_ING0 = 1
T_ING1 = 2
T_POT_LEFT = 3
T_PLATE = 4
T_POT_RIGHT = 5
T_BOTTOM_LOAD = 6
T_PLATE_BOTTOM = 7

O_HOME = 0
O_ING0 = 1
O_ING1 = 2
O_TOP_COUNTER = 3
O_BOTTOM_COUNTER = 4


def _agent_0_initial():
    return jnp.asarray([-1], dtype=jnp.int32)


def _agent_1_initial():
    return jnp.asarray([0, -1, 0, 0], dtype=jnp.int32)


def _static_crop(center_r, center_c):
    crop = []
    for local_r in range(5):
        row = []
        global_r = center_r + local_r - 2
        for local_c in range(5):
            global_c = center_c + local_c - 2
            cell = [0] * 9
            if 0 <= global_r < 9 and 0 <= global_c < 9:
                char = LAYOUT[global_r][global_c]
                if char in STATIC_CHANNEL_BY_CHAR:
                    cell[STATIC_CHANNEL_BY_CHAR[char]] = 1
            row.append(cell)
        crop.append(row)
    return crop


def _all_static_crops():
    return jnp.asarray(
        [_static_crop(row, col) for row in range(9) for col in range(9)],
        dtype=jnp.int32,
    )


def _pos_mask():
    flags = []
    for row in range(9):
        for col in range(9):
            flags.append(LAYOUT[row][col] in (" ", "A"))
    return jnp.asarray(flags, dtype=bool)


def _infer_pos(obs):
    matches = obs[:, :, 20:29][None, :, :, :] == _all_static_crops()
    scores = jnp.sum(matches.astype(jnp.int32), axis=(1, 2, 3))
    scores = jnp.where(_pos_mask(), scores, -1)
    idx = jnp.argmax(scores).astype(jnp.int32)
    return idx // 9, idx % 9


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _feature_at(obs, pos_r, pos_c, global_r, global_c, channel):
    local_r = global_r - pos_r + 2
    local_c = global_c - pos_c + 2
    visible = (local_r >= 0) & (local_r < 5) & (local_c >= 0) & (local_c < 5)
    safe_r = jnp.clip(local_r, 0, 4)
    safe_c = jnp.clip(local_c, 0, 4)
    value = obs[safe_r, safe_c, channel]
    return jnp.where(visible, value, jnp.asarray(0, dtype=obs.dtype))


def _visible_at(pos_r, pos_c, global_r, global_c):
    local_r = global_r - pos_r + 2
    local_c = global_c - pos_c + 2
    return (local_r >= 0) & (local_r < 5) & (local_c >= 0) & (local_c < 5)


def _item_sum_at(obs, pos_r, pos_c, global_r, global_c):
    total = jnp.asarray(0, dtype=obs.dtype)
    total = total + _feature_at(obs, pos_r, pos_c, global_r, global_c, 29)
    total = total + _feature_at(obs, pos_r, pos_c, global_r, global_c, 30)
    total = total + _feature_at(obs, pos_r, pos_c, global_r, global_c, 31)
    total = total + _feature_at(obs, pos_r, pos_c, global_r, global_c, 32)
    total = total + _feature_at(obs, pos_r, pos_c, global_r, global_c, 33)
    return total


def _inner_nav_action(pos_r, pos_c, target_idx):
    dists = jnp.asarray(INNER_DISTS, dtype=jnp.int32)
    walkable = jnp.asarray(INNER_WALKABLE, dtype=bool)
    dist = dists[target_idx]
    here_dist = dist[pos_r, pos_c]
    next_r = jnp.clip(pos_r + jnp.asarray([0, 1, 0, -1], dtype=jnp.int32), 0, 8)
    next_c = jnp.clip(pos_c + jnp.asarray([1, 0, -1, 0], dtype=jnp.int32), 0, 8)
    next_dist = dist[next_r, next_c]
    next_dist = jnp.where(walkable[next_r, next_c], next_dist, BIG)
    best_dir = jnp.argmin(next_dist).astype(jnp.int32)
    action = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)[best_dir]
    return jnp.where(here_dist == 0, STAY, action)


def _outer_nav_action(pos_r, pos_c, target_idx):
    dists = jnp.asarray(OUTER_DISTS, dtype=jnp.int32)
    dist = dists[target_idx]
    here_dist = dist[pos_r, pos_c]
    next_r = jnp.clip(pos_r + jnp.asarray([0, 1, 0, -1], dtype=jnp.int32), 0, 8)
    next_c = jnp.clip(pos_c + jnp.asarray([1, 0, -1, 0], dtype=jnp.int32), 0, 8)
    next_dist = dist[next_r, next_c]
    best_dir = jnp.argmin(next_dist).astype(jnp.int32)
    action = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)[best_dir]
    return jnp.where(here_dist == 0, STAY, action)


def _at_target(pos_r, pos_c, target_idx):
    dists = jnp.asarray(INNER_DISTS, dtype=jnp.int32)
    return dists[target_idx, pos_r, pos_c] == 0


def _at_pose(pos_r, pos_c, facing, target_idx, face_dir):
    return _at_target(pos_r, pos_c, target_idx) & (facing == face_dir)


def _outer_at_target(pos_r, pos_c, target_idx):
    dists = jnp.asarray(OUTER_DISTS, dtype=jnp.int32)
    return dists[target_idx, pos_r, pos_c] == 0


def _face_action(face_dir):
    return jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[face_dir]


def _go_face(pos_r, pos_c, facing, target_idx, face_dir):
    at_target = _at_target(pos_r, pos_c, target_idx)
    face = _face_action(face_dir)
    nav = _inner_nav_action(pos_r, pos_c, target_idx)
    return jnp.where(at_target, jnp.where(facing == face_dir, STAY, face), nav)


def _go_interact(pos_r, pos_c, facing, target_idx, face_dir):
    at_target = _at_target(pos_r, pos_c, target_idx)
    at_pose = at_target & (facing == face_dir)
    face = _face_action(face_dir)
    nav = _inner_nav_action(pos_r, pos_c, target_idx)
    return jnp.where(at_target, jnp.where(at_pose, INTERACT, face), nav)


def _outer_go_interact(pos_r, pos_c, facing, target_idx, face_dir):
    at_target = _outer_at_target(pos_r, pos_c, target_idx)
    at_pose = at_target & (facing == face_dir)
    face = _face_action(face_dir)
    nav = _outer_nav_action(pos_r, pos_c, target_idx)
    return jnp.where(at_target, jnp.where(at_pose, INTERACT, face), nav)


def init_agent_0():
    return _agent_0_initial()


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _agent_0_initial(), memory)
    pos_r, pos_c = _infer_pos(obs)
    facing = _facing(obs)

    recipe_0 = jnp.sum(obs[:, :, 36])
    recipe_1 = jnp.sum(obs[:, :, 37])
    sees_recipe = (recipe_0 + recipe_1) > 0
    observed_target = jnp.where(recipe_0 >= recipe_1, 0, 1).astype(jnp.int32)
    target = jnp.where(sees_recipe, observed_target, memory[0])

    held_cooked = obs[2, 2, 6] > 0
    held_ingredient_0 = obs[2, 2, 7] > 0
    held_ingredient_1 = obs[2, 2, 8] > 0
    held_raw = held_ingredient_0 | held_ingredient_1
    held_any = held_cooked | held_raw | (obs[2, 2, 5] > 0)

    home_action = _outer_nav_action(pos_r, pos_c, O_HOME)
    deliver_action = _outer_go_interact(pos_r, pos_c, facing, O_HOME, FACE_RIGHT)
    pickup_idx = jnp.where(target == 0, O_ING0, O_ING1)
    pickup_action = _outer_go_interact(pos_r, pos_c, facing, pickup_idx, FACE_RIGHT)
    top_occupied = _item_sum_at(obs, pos_r, pos_c, 2, 3) > 0
    bottom_occupied = _item_sum_at(obs, pos_r, pos_c, 6, 3) > 0
    top_visible = _visible_at(pos_r, pos_c, 2, 3)
    bottom_visible = _visible_at(pos_r, pos_c, 6, 3)
    target_counter_occupied = jnp.where(target == 0, top_occupied, bottom_occupied)
    target_counter_visible = jnp.where(target == 0, top_visible, bottom_visible)
    target_counter_open = target_counter_visible & (~target_counter_occupied)
    deposit_idx = jnp.where(held_ingredient_0, O_TOP_COUNTER, O_BOTTOM_COUNTER)
    deposit_face = jnp.where(held_ingredient_0, 1, FACE_UP).astype(jnp.int32)
    deposit_action = _outer_go_interact(pos_r, pos_c, facing, deposit_idx, deposit_face)

    has_target = target >= 0
    empty_action = jnp.where(has_target & target_counter_open, pickup_action, home_action)
    action = jnp.where(
        held_cooked,
        deliver_action,
        jnp.where(held_raw, deposit_action, jnp.where(held_any, home_action, empty_action)),
    )

    trace = {
        "row": pos_r,
        "col": pos_c,
        "target": target,
        "counter_open": target_counter_open,
        "helper": jnp.asarray(1, dtype=jnp.int32),
    }
    return action.astype(jnp.int32), jnp.asarray([target], dtype=jnp.int32), trace


def init_agent_1():
    return _agent_1_initial()


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _agent_1_initial(), memory)
    phase = memory[0]
    target = memory[1]
    added = memory[2]
    step_count = memory[3]

    pos_r, pos_c = _infer_pos(obs)
    facing = _facing(obs)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:10]) > 0
    empty_handed = ~(held_plate | held_cooked | held_raw)

    recipe_0 = jnp.sum(obs[:, :, 36])
    recipe_1 = jnp.sum(obs[:, :, 37])
    sees_recipe = (recipe_0 + recipe_1) > 0
    observed_target = jnp.where(recipe_0 >= recipe_1, 0, 1).astype(jnp.int32)

    can_use_seen_recipe = ((phase == 0) | (phase == 1)) & sees_recipe
    phase_eff = jnp.where(can_use_seen_recipe, 2, phase)
    target_eff = jnp.where(can_use_seen_recipe, observed_target, target)
    added_eff = jnp.where(can_use_seen_recipe, jnp.asarray(0, dtype=jnp.int32), added)

    can_start_cycle = step_count < 360
    read_action = jnp.where(
        can_start_cycle,
        _go_interact(pos_r, pos_c, facing, T_READ, FACE_UP),
        STAY,
    )
    ing_target_idx = jnp.where(target_eff == 0, T_ING0, T_ING1)
    pickup_action = _go_interact(pos_r, pos_c, facing, ing_target_idx, FACE_LEFT)
    add_top_action = _go_interact(pos_r, pos_c, facing, T_READ, FACE_DOWN)
    add_bottom_action = _go_interact(pos_r, pos_c, facing, T_BOTTOM_LOAD, FACE_UP)
    add_action = jnp.where(target_eff == 0, add_top_action, add_bottom_action)
    load_action = jnp.where(empty_handed, pickup_action, jnp.where(held_raw, add_action, STAY))

    pot_cooked = _feature_at(obs, pos_r, pos_c, 4, 4, 30) > 0
    top_plate_action = _go_interact(pos_r, pos_c, facing, T_PLATE, FACE_RIGHT)
    bottom_plate_action = _go_interact(pos_r, pos_c, facing, T_PLATE_BOTTOM, FACE_RIGHT)
    get_plate_action = jnp.where(target_eff == 0, top_plate_action, bottom_plate_action)
    wait_pot_action = _go_face(pos_r, pos_c, facing, T_POT_RIGHT, FACE_LEFT)
    scoop_action = _go_interact(pos_r, pos_c, facing, T_POT_RIGHT, FACE_LEFT)
    plate_action = jnp.where(pot_cooked, scoop_action, wait_pot_action)
    deliver_action = _go_interact(pos_r, pos_c, facing, T_POT_RIGHT, FACE_RIGHT)
    cook_action = jnp.where(
        held_cooked,
        deliver_action,
        jnp.where(held_plate, plate_action, jnp.where(empty_handed, get_plate_action, STAY)),
    )

    active_deliver_phase = (phase_eff == 4) | ((phase_eff == 3) & held_cooked)
    action = jnp.where(
        phase_eff == 0,
        read_action,
        jnp.where(
            phase_eff == 1,
            STAY,
            jnp.where(
                phase_eff == 2,
                load_action,
                jnp.where(active_deliver_phase, deliver_action, cook_action),
            ),
        ),
    )

    pressed_button = (phase_eff == 0) & can_start_cycle & _at_pose(pos_r, pos_c, facing, T_READ, FACE_UP)
    added_top = (target_eff == 0) & _at_pose(pos_r, pos_c, facing, T_READ, FACE_DOWN)
    added_bottom = (target_eff == 1) & _at_pose(pos_r, pos_c, facing, T_BOTTOM_LOAD, FACE_UP)
    added_ingredient = (phase_eff == 2) & held_raw & (added_top | added_bottom)
    scooped_dish = (phase_eff == 3) & held_plate & pot_cooked & _at_pose(pos_r, pos_c, facing, T_POT_RIGHT, FACE_LEFT)
    delivered = active_deliver_phase & held_cooked & _at_pose(pos_r, pos_c, facing, T_POT_RIGHT, FACE_RIGHT)

    next_phase = phase_eff
    next_target = target_eff
    next_added = added_eff
    next_phase = jnp.where(pressed_button, jnp.asarray(1, dtype=jnp.int32), next_phase)
    next_added = jnp.where(added_ingredient, added_eff + 1, next_added)
    next_phase = jnp.where(added_ingredient & (added_eff >= 2), jnp.asarray(3, dtype=jnp.int32), next_phase)
    next_phase = jnp.where(scooped_dish, jnp.asarray(4, dtype=jnp.int32), next_phase)
    next_phase = jnp.where(delivered, jnp.asarray(0, dtype=jnp.int32), next_phase)
    next_target = jnp.where(delivered, jnp.asarray(-1, dtype=jnp.int32), next_target)
    next_added = jnp.where(delivered, jnp.asarray(0, dtype=jnp.int32), next_added)

    next_step_count = step_count + 1
    next_memory = jnp.asarray([next_phase, next_target, next_added, next_step_count], dtype=jnp.int32)
    trace = {
        "row": pos_r,
        "col": pos_c,
        "phase": next_phase,
        "target": next_target,
        "added": next_added,
        "step": next_step_count,
        "pot_cooked": pot_cooked,
    }
    return action.astype(jnp.int32), next_memory, trace
