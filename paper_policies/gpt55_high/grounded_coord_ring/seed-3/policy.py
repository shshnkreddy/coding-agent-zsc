"""Geometry-grounded Overcooked policy for grounded_coord_ring."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_coord_ring_roles',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}

INTERACT = 5
STAY = 4
FACE_UP = 0
FACE_DOWN = 1
FACE_RIGHT = 2
FACE_LEFT = 3
INF = 1000

OUTSIDE = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 1), (2, 7), (3, 1), (3, 7), (4, 1), (4, 7), (5, 1), (5, 7),
    (6, 1), (6, 7), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
)
INSIDE = ((3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5))

A0_READ = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 3, 4, 5, 6, 7, 8, 9, 1000),
    (1000, 2, 1000, 1000, 1000, 1000, 1000, 10, 1000),
    (1000, 1, 1000, 1000, 1000, 1000, 1000, 11, 1000),
    (1000, 0, 1000, 1000, 1000, 1000, 1000, 12, 1000),
    (1000, 1, 1000, 1000, 1000, 1000, 1000, 11, 1000),
    (1000, 2, 1000, 1000, 1000, 1000, 1000, 10, 1000),
    (1000, 3, 4, 5, 6, 7, 8, 9, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A0_ING0 = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 2, 3, 4, 5, 6, 7, 8, 1000),
    (1000, 1, 1000, 1000, 1000, 1000, 1000, 9, 1000),
    (1000, 0, 1000, 1000, 1000, 1000, 1000, 10, 1000),
    (1000, 1, 1000, 1000, 1000, 1000, 1000, 11, 1000),
    (1000, 2, 1000, 1000, 1000, 1000, 1000, 12, 1000),
    (1000, 3, 1000, 1000, 1000, 1000, 1000, 11, 1000),
    (1000, 4, 5, 6, 7, 8, 9, 10, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A0_ING1 = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 4, 5, 6, 7, 8, 9, 10, 1000),
    (1000, 3, 1000, 1000, 1000, 1000, 1000, 11, 1000),
    (1000, 2, 1000, 1000, 1000, 1000, 1000, 12, 1000),
    (1000, 1, 1000, 1000, 1000, 1000, 1000, 11, 1000),
    (1000, 0, 1000, 1000, 1000, 1000, 1000, 10, 1000),
    (1000, 1, 1000, 1000, 1000, 1000, 1000, 9, 1000),
    (1000, 2, 3, 4, 5, 6, 7, 8, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A0_DROP = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 2, 1, 0, 1, 2, 3, 4, 1000),
    (1000, 3, 1000, 1000, 1000, 1000, 1000, 5, 1000),
    (1000, 4, 1000, 1000, 1000, 1000, 1000, 6, 1000),
    (1000, 5, 1000, 1000, 1000, 1000, 1000, 7, 1000),
    (1000, 6, 1000, 1000, 1000, 1000, 1000, 8, 1000),
    (1000, 7, 1000, 1000, 1000, 1000, 1000, 9, 1000),
    (1000, 8, 9, 10, 11, 12, 11, 10, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A1_BUTTON = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1, 0, 1, 1000, 1000, 1000),
    (1000, 1000, 1000, 2, 1000, 2, 1000, 1000, 1000),
    (1000, 1000, 1000, 3, 4, 3, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A1_ING0 = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 0, 1, 2, 1000, 1000, 1000),
    (1000, 1000, 1000, 1, 1000, 3, 1000, 1000, 1000),
    (1000, 1000, 1000, 2, 3, 4, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A1_ING1 = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 2, 3, 4, 1000, 1000, 1000),
    (1000, 1000, 1000, 1, 1000, 3, 1000, 1000, 1000),
    (1000, 1000, 1000, 0, 1, 2, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A1_TOP_COUNTER = A1_ING0
A1_POT = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1, 2, 3, 1000, 1000, 1000),
    (1000, 1000, 1000, 0, 1000, 4, 1000, 1000, 1000),
    (1000, 1000, 1000, 1, 2, 3, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A1_PLATE = (
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 2, 1, 0, 1000, 1000, 1000),
    (1000, 1000, 1000, 3, 1000, 1, 1000, 1000, 1000),
    (1000, 1000, 1000, 4, 3, 2, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
)
A1_DELIVER = A1_POT


def init_agent_0():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def init_agent_1():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def _static_value(row, col, channel):
    value = 0
    if channel == 0:
        value = int((row, col) in (
            (0, 0), (0, 1), (0, 2), (0, 6), (0, 7), (0, 8), (1, 0), (1, 8),
            (2, 0), (2, 2), (2, 3), (2, 5), (2, 6), (2, 8), (6, 0), (6, 2),
            (6, 3), (6, 5), (6, 6), (6, 8), (7, 0), (7, 8), (8, 0), (8, 1),
            (8, 2), (8, 6), (8, 7), (8, 8)))
    elif channel == 1:
        value = int((row, col) in ((4, 2), (4, 6)))
    elif channel == 2:
        value = int((row, col) == (4, 4))
    elif channel == 3:
        value = int((row, col) in ((0, 4), (4, 0), (4, 8), (8, 4)))
    elif channel == 4:
        value = int((row, col) in ((2, 4), (6, 4)))
    elif channel == 5:
        value = int((row, col) in ((3, 6), (5, 6)))
    elif channel == 6:
        value = int((row, col) == (3, 2))
    elif channel == 7:
        value = int((row, col) == (5, 2))
    elif channel == 8:
        value = int((row, col) in ((0, 3), (0, 5), (3, 0), (3, 8), (5, 0), (5, 8), (8, 3), (8, 5)))
    return value


def _candidate_score(obs, row, col):
    score = jnp.asarray(0, dtype=jnp.int32)
    for li in range(5):
        for lj in range(5):
            gr = row + li - 2
            gc = col + lj - 2
            off_map = gr < 0 or gr >= 9 or gc < 0 or gc >= 9
            for ch in range(9):
                expected = 0 if off_map else _static_value(gr, gc, ch)
                score = score + (obs[li, lj, 20 + ch] == expected).astype(jnp.int32)
    return score


def _infer_pos(obs, candidates):
    scores = []
    for row, col in candidates:
        scores.append(_candidate_score(obs, row, col))
    idx = jnp.argmax(jnp.asarray(scores, dtype=jnp.int32)).astype(jnp.int32)
    coords = jnp.asarray(candidates, dtype=jnp.int32)
    return coords[idx, 0], coords[idx, 1]


def _recipe_from_obs(obs):
    r0 = jnp.sum(obs[:, :, 36])
    r1 = jnp.sum(obs[:, :, 37])
    return jnp.where(r0 > 0, 0, jnp.where(r1 > 0, 1, -1)).astype(jnp.int32)


def _held(obs):
    inv = obs[2, 2, 5:10]
    plate = inv[0] > 0
    cooked = inv[1] > 0
    raw0 = inv[2] > 0
    raw1 = inv[3] > 0
    raw2 = inv[4] > 0
    empty = ~(plate | cooked | raw0 | raw1 | raw2)
    raw = (raw0 | raw1 | raw2) & ~(plate | cooked)
    raw_kind = jnp.where(raw1, 1, jnp.where(raw0, 0, 2)).astype(jnp.int32)
    return empty, plate, cooked, raw, raw_kind


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _at_global(obs, row, col, gr, gc, ch):
    li = gr - row + 2
    lj = gc - col + 2
    visible = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    val = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), ch]
    return jnp.where(visible, val, 0)


def _nav(row, col, face, dist_tuple, target_r, target_c, target_face):
    dist = jnp.asarray(dist_tuple, dtype=jnp.int32)
    dir_to_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)
    at_target = (row == target_r) & (col == target_c)
    wanted_face_action = dir_to_action[target_face]
    face_or_interact = jnp.where(face == target_face, INTERACT, wanted_face_action)
    choices = jnp.asarray([
        dist[row, jnp.minimum(col + 1, 8)],
        dist[jnp.minimum(row + 1, 8), col],
        dist[row, jnp.maximum(col - 1, 0)],
        dist[jnp.maximum(row - 1, 0), col],
    ], dtype=jnp.int32)
    move = jnp.argmin(choices).astype(jnp.int32)
    move = jnp.where(jnp.min(choices) >= INF, STAY, move)
    return jnp.where(at_target, face_or_interact, move).astype(jnp.int32)


def _trace(row, col, recipe, mode):
    return {
        'row': row.astype(jnp.int32),
        'col': col.astype(jnp.int32),
        'recipe': recipe.astype(jnp.int32),
        'mode': mode.astype(jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    row, col = _infer_pos(obs, OUTSIDE)
    face = _facing(obs)
    empty, plate, cooked, raw, raw_kind = _held(obs)
    del plate, cooked, raw_kind

    seen_recipe = _recipe_from_obs(obs)
    old_recipe = jnp.where(reset, -1, memory[0])
    recipe = jnp.where(seen_recipe >= 0, seen_recipe, old_recipe).astype(jnp.int32)

    read_action = _nav(row, col, face, A0_READ, 4, 1, FACE_LEFT)
    fetch0_action = _nav(row, col, face, A0_ING0, 3, 1, FACE_RIGHT)
    fetch1_action = _nav(row, col, face, A0_ING1, 5, 1, FACE_RIGHT)
    fetch_action = jnp.where(recipe == 1, fetch1_action, fetch0_action)
    drop_action = _nav(row, col, face, A0_DROP, 1, 3, FACE_DOWN)

    mode = jnp.where(recipe < 0, 0, jnp.where(raw, 2, 1)).astype(jnp.int32)
    action = jnp.where(recipe < 0, read_action, jnp.where(raw, drop_action, fetch_action))
    action = jnp.where(empty | raw, action, STAY).astype(jnp.int32)

    next_recipe = jnp.where(reset, seen_recipe, recipe)
    next_memory = jnp.asarray([next_recipe, mode], dtype=jnp.int32)
    return action, next_memory, _trace(row, col, recipe, mode)


def agent_1_act(obs, reset, memory, rng):
    del rng
    row, col = _infer_pos(obs, INSIDE)
    face = _facing(obs)
    empty, plate, cooked_hold, raw, raw_kind = _held(obs)
    del raw_kind

    displayed_recipe = _recipe_from_obs(obs)
    top_raw0 = _at_global(obs, row, col, 2, 3, 31) > 0
    top_raw1 = _at_global(obs, row, col, 2, 3, 32) > 0
    remembered = jnp.where(reset, -1, memory[0])
    recipe = jnp.where(displayed_recipe >= 0, displayed_recipe, remembered)
    recipe = recipe.astype(jnp.int32)

    pot0 = _at_global(obs, row, col, 4, 4, 31)
    pot1 = _at_global(obs, row, col, 4, 4, 32)
    pot2 = _at_global(obs, row, col, 4, 4, 33)
    pot_total = pot0 + pot1 + pot2
    pot_cooked = _at_global(obs, row, col, 4, 4, 30) > 0
    pot_timer = _at_global(obs, row, col, 4, 4, 39)
    pot_busy = (pot_total >= 3) | (pot_timer > 0) | pot_cooked
    target_counter = ((recipe == 0) & top_raw0) | ((recipe == 1) & top_raw1)

    button_action = _nav(row, col, face, A1_BUTTON, 3, 4, FACE_UP)
    get0_action = _nav(row, col, face, A1_ING0, 3, 3, FACE_LEFT)
    get1_action = _nav(row, col, face, A1_ING1, 5, 3, FACE_LEFT)
    get_supply_action = jnp.where(recipe == 1, get1_action, get0_action)
    get_counter_action = _nav(row, col, face, A1_TOP_COUNTER, 3, 3, FACE_UP)
    get_raw_action = jnp.where(target_counter, get_counter_action, get_supply_action)
    pot_action = _nav(row, col, face, A1_POT, 4, 3, FACE_RIGHT)
    plate_action = _nav(row, col, face, A1_PLATE, 3, 5, FACE_RIGHT)
    deliver_action = _nav(row, col, face, A1_DELIVER, 4, 3, FACE_LEFT)

    needs_recipe = recipe < 0
    mode = jnp.asarray(1, dtype=jnp.int32)
    action = get_raw_action

    mode = jnp.where(needs_recipe, 0, mode)
    action = jnp.where(needs_recipe, button_action, action)
    mode = jnp.where(cooked_hold, 6, mode)
    action = jnp.where(cooked_hold, deliver_action, action)
    mode = jnp.where(plate & ~cooked_hold & pot_cooked, 5, mode)
    action = jnp.where(plate & ~cooked_hold & pot_cooked, pot_action, action)
    mode = jnp.where(plate & ~cooked_hold & ~pot_cooked, 4, mode)
    action = jnp.where(plate & ~cooked_hold & ~pot_cooked, pot_action, action)
    mode = jnp.where(raw, 2, mode)
    action = jnp.where(raw, pot_action, action)
    get_plate = empty & (recipe >= 0) & pot_busy
    mode = jnp.where(get_plate, 3, mode)
    action = jnp.where(get_plate, plate_action, action)
    wait_for_pot = empty & (recipe >= 0) & (pot_timer > 0) & ~pot_cooked
    mode = jnp.where(wait_for_pot, 4, mode)
    action = jnp.where(wait_for_pot, pot_action, action)

    after_delivery = cooked_hold & (action == INTERACT)
    next_recipe = jnp.where(after_delivery, -1, recipe).astype(jnp.int32)
    next_memory = jnp.asarray([next_recipe, mode], dtype=jnp.int32)
    return action.astype(jnp.int32), next_memory, _trace(row, col, recipe, mode)
