"""Convention-robust fixed-layout policy for demo_cook_wide."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'role_split_recipe_reader', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

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

HEIGHT = 6
WIDTH = 11

STATIC_CHANNELS = (
    (0, 0, 22), (0, 1, 22), (0, 2, 22), (0, 3, 22), (0, 7, 22), (0, 8, 22), (0, 9, 22), (0, 10, 22),
    (1, 0, 22), (1, 1, 22), (1, 2, 22), (1, 8, 22), (1, 9, 22), (1, 10, 22),
    (2, 0, 22), (2, 1, 22), (2, 2, 22), (2, 3, 22), (2, 4, 22), (2, 6, 22), (2, 7, 22), (2, 8, 22), (2, 9, 22), (2, 10, 22),
    (3, 0, 22), (3, 10, 22), (4, 3, 22), (4, 7, 22),
    (5, 0, 22), (5, 2, 22), (5, 3, 22), (5, 4, 22), (5, 5, 22), (5, 6, 22), (5, 7, 22), (5, 8, 22), (5, 10, 22),
    (0, 5, 23),
    (2, 5, 24),
    (4, 5, 25),
    (0, 4, 27), (0, 6, 27),
    (1, 3, 28), (4, 0, 28), (4, 10, 28),
    (1, 7, 29), (5, 1, 29), (5, 9, 29),
    (4, 4, 31), (4, 6, 31),
)

TOP_CANDIDATES = ((1, 4), (1, 5), (1, 6))
BOTTOM_CANDIDATES = (
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 1), (4, 2), (4, 8), (4, 9),
)


def _static_map():
    cells = [[[0 for _ in range(10)] for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for r, c, ch in STATIC_CHANNELS:
        cells[r][c][ch - 22] = 1
    return cells


def _candidate_crops(candidates):
    static_map = _static_map()
    all_crops = []
    for gr, gc in candidates:
        crop = []
        for lr in range(5):
            row = []
            for lc in range(5):
                rr = gr + lr - 2
                cc = gc + lc - 2
                if 0 <= rr < HEIGHT and 0 <= cc < WIDTH:
                    row.append(static_map[rr][cc])
                else:
                    row.append([0 for _ in range(10)])
            crop.append(row)
        all_crops.append(crop)
    return all_crops


def _locate(obs, candidates):
    crops = jnp.asarray(_candidate_crops(candidates), dtype=jnp.int32)
    rows = jnp.asarray([r for r, _ in candidates], dtype=jnp.int32)
    cols = jnp.asarray([c for _, c in candidates], dtype=jnp.int32)
    static_obs = obs[:, :, 22:32]
    scores = jnp.sum(jnp.asarray(static_obs == crops, dtype=jnp.int32), axis=(1, 2, 3))
    idx = jnp.argmax(scores)
    return rows[idx], cols[idx]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5])


def _face_or_interact(obs, wanted_face):
    face = _facing(obs)
    face_action = jnp.asarray(3, dtype=jnp.int32)
    face_action = jnp.where(wanted_face == FACE_DOWN, DOWN, face_action)
    face_action = jnp.where(wanted_face == FACE_RIGHT, RIGHT, face_action)
    face_action = jnp.where(wanted_face == FACE_LEFT, LEFT, face_action)
    return jnp.where(face == wanted_face, INTERACT, face_action)


def _held_counts(obs):
    return obs[2, 2, 5], obs[2, 2, 6], obs[2, 2, 7], obs[2, 2, 8], obs[2, 2, 9], obs[2, 2, 10]


def _pot_values(obs):
    pot = obs[:, :, 24]
    item = obs[:, :, 32:38]
    counts = jnp.sum(item[:, :, 2:6] * pot[:, :, None], axis=(0, 1))
    total = jnp.sum(counts)
    cooked = jnp.max(obs[:, :, 33] * pot)
    timer = jnp.max(obs[:, :, 44] * pot)
    visible = jnp.max(pot)
    return visible, total, cooked, timer


def _pot_ingredient_counts(obs):
    pot = obs[:, :, 24]
    item = obs[:, :, 34:38]
    return jnp.sum(item * pot[:, :, None], axis=(0, 1))


def _top_move_to_col(col, target_col):
    return jnp.where(col < target_col, RIGHT, jnp.where(col > target_col, LEFT, STAY))


def _top_interact_at(obs, col, target_col, face):
    at_target = col == target_col
    move = _top_move_to_col(col, target_col)
    return jnp.where(at_target, _face_or_interact(obs, face), move)


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    _, col = _locate(obs, TOP_CANDIDATES)
    base_memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    assist_state = base_memory[0]
    held_plate, held_cooked, held_i0, held_i1, held_i2, held_i3 = _held_counts(obs)
    held_raw = held_i0 + held_i1 + held_i2 + held_i3
    empty_hands = (held_plate + held_cooked + held_raw) == 0
    has_empty_plate = (held_plate == 1) & (held_cooked == 0) & (held_raw == 0)
    has_dish = held_cooked == 1
    has_raw = held_raw > 0
    _, pot_total, pot_cooked, pot_timer = _pot_values(obs)
    pot_counts = _pot_ingredient_counts(obs)
    known_partial_pot = (pot_total > 0) & (pot_total < 3) & (pot_cooked == 0) & (pot_timer == 0)
    homogeneous_known = ((pot_counts[0] > 0) & (pot_counts[1] == 0)) | ((pot_counts[1] > 0) & (pot_counts[0] == 0))
    can_start_help = empty_hands & (assist_state == 0) & known_partial_pot & homogeneous_known
    continuing_help = empty_hands & (assist_state == 1) & known_partial_pot & homogeneous_known
    helping_now = can_start_help | continuing_help

    plate_col = jnp.where(col >= 6, jnp.asarray(6, dtype=jnp.int32), jnp.asarray(4, dtype=jnp.int32))
    get_plate = _top_interact_at(obs, col, plate_col, FACE_UP)
    deliver = _top_interact_at(obs, col, jnp.asarray(5, dtype=jnp.int32), FACE_UP)
    take_pot = _top_interact_at(obs, col, jnp.asarray(5, dtype=jnp.int32), FACE_DOWN)
    add_raw = _top_interact_at(obs, col, jnp.asarray(5, dtype=jnp.int32), FACE_DOWN)
    fetch_0 = _top_interact_at(obs, col, jnp.asarray(4, dtype=jnp.int32), FACE_LEFT)
    fetch_1 = _top_interact_at(obs, col, jnp.asarray(6, dtype=jnp.int32), FACE_RIGHT)
    fetch_matching = jnp.where(pot_counts[0] > 0, fetch_0, fetch_1)

    wait_at_pot = _top_move_to_col(col, jnp.asarray(5, dtype=jnp.int32))
    wait_at_pot = jnp.where((col == 5) & (_facing(obs) != FACE_DOWN), DOWN, wait_at_pot)
    wait_at_pot = jnp.where((col == 5) & (_facing(obs) == FACE_DOWN), STAY, wait_at_pot)

    with_plate = jnp.where(pot_cooked == 1, take_pot, wait_at_pot)
    pot_started = (pot_total > 0) | (pot_timer > 0) | (pot_cooked == 1)
    empty_action = jnp.where(pot_started, get_plate, wait_at_pot)
    empty_action = jnp.where(helping_now, fetch_matching, empty_action)
    action = jnp.where(empty_hands, empty_action, STAY)
    action = jnp.where(has_empty_plate, with_plate, action)
    action = jnp.where(has_raw, add_raw, action)
    action = jnp.where(has_dish, deliver, action)

    next_state = assist_state
    next_state = jnp.where(can_start_help | continuing_help, jnp.asarray(1, dtype=jnp.int32), next_state)
    next_state = jnp.where(has_raw, jnp.asarray(2, dtype=jnp.int32), next_state)
    next_state = jnp.where((assist_state == 1) & empty_hands & (~known_partial_pot), jnp.asarray(2, dtype=jnp.int32), next_state)
    next_state = jnp.where((pot_total == 0) & empty_hands, jnp.asarray(0, dtype=jnp.int32), next_state)
    next_memory = jnp.asarray([next_state], dtype=jnp.int32)
    trace = {
        'can_help': helping_now.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'pot_i0': pot_counts[0].astype(jnp.int32),
        'pot_i1': pot_counts[1].astype(jnp.int32),
        'helped': assist_state.astype(jnp.int32),
        'empty': empty_hands.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


def _bottom_nav_to(row, col, target_row, target_col):
    if target_row == 3:
        target_col = jnp.asarray(target_col, dtype=jnp.int32)
        return jnp.where(row == 4, UP, jnp.where(col < target_col, RIGHT, jnp.where(col > target_col, LEFT, STAY)))

    left_target = target_col <= 2
    target_col = jnp.asarray(target_col, dtype=jnp.int32)
    if left_target:
        in_right_pocket = (row == 4) & (col > 2)
        from_row3 = (row == 3) & (col != target_col)
        at_drop = (row == 3) & (col == target_col)
        from_pocket = (row == 4) & (col != target_col)
        move_row3 = jnp.where(col < target_col, RIGHT, LEFT)
        action = jnp.where(in_right_pocket, UP, STAY)
        action = jnp.where(from_row3, move_row3, action)
        action = jnp.where(at_drop, DOWN, action)
        action = jnp.where(from_pocket, jnp.where(col < target_col, RIGHT, LEFT), action)
        return action

    in_left_pocket = (row == 4) & (col < 8)
    from_row3 = (row == 3) & (col != target_col)
    at_drop = (row == 3) & (col == target_col)
    from_pocket = (row == 4) & (col != target_col)
    move_row3 = jnp.where(col < target_col, RIGHT, LEFT)
    action = jnp.where(in_left_pocket, UP, STAY)
    action = jnp.where(from_row3, move_row3, action)
    action = jnp.where(at_drop, DOWN, action)
    action = jnp.where(from_pocket, jnp.where(col < target_col, RIGHT, LEFT), action)
    return action


def _bottom_interact_at(obs, row, col, target_row, target_col, face):
    at_target = (row == target_row) & (col == target_col)
    move = _bottom_nav_to(row, col, target_row, target_col)
    return jnp.where(at_target, _face_or_interact(obs, face), move)


def init_agent_1():
    return jnp.asarray([-1], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    row, col = _locate(obs, BOTTOM_CANDIDATES)
    base_memory = jnp.where(reset, jnp.asarray([-1], dtype=jnp.int32), memory)

    recipe0_seen = jnp.sum(obs[:, :, 40])
    recipe1_seen = jnp.sum(obs[:, :, 41])
    recipe_seen = (recipe0_seen + recipe1_seen) > 0
    seen_recipe = jnp.where(recipe1_seen > recipe0_seen, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    recipe = jnp.where(recipe_seen, seen_recipe, base_memory[0])
    next_memory = jnp.asarray([recipe], dtype=jnp.int32)

    held_plate, held_cooked, held_i0, held_i1, held_i2, held_i3 = _held_counts(obs)
    held_raw = held_i0 + held_i1 + held_i2 + held_i3
    holding_any = (held_plate + held_cooked + held_raw) > 0
    holding_raw = held_raw > 0

    pot_visible, pot_total, pot_cooked, pot_timer = _pot_values(obs)
    pot_busy = (pot_visible == 1) & ((pot_total >= 3) | (pot_cooked == 1) | (pot_timer > 0))
    recipe_unknown = recipe < 0

    read_recipe = _bottom_nav_to(row, col, 3, 5)
    wait_center = jnp.where((row == 3) & (col == 5), STAY, read_recipe)
    add_to_pot = _bottom_interact_at(obs, row, col, 3, 5, FACE_UP)
    fetch_0 = _bottom_interact_at(obs, row, col, 4, 1, FACE_LEFT)
    fetch_1 = _bottom_interact_at(obs, row, col, 4, 9, FACE_DOWN)
    fetch_recipe = jnp.where(recipe == 0, fetch_0, fetch_1)

    action = fetch_recipe
    action = jnp.where(recipe_unknown, read_recipe, action)
    action = jnp.where(pot_busy & (~holding_any), wait_center, action)
    action = jnp.where(holding_raw, add_to_pot, action)

    return action.astype(jnp.int32), next_memory, {}
