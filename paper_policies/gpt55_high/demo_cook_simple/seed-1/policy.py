"""Map-aware deterministic policy for the demo_cook_simple Overcooked task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'division_of_labor_v1',
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

H = 5
W = 11
PAD = 2


def _build_static_map():
    rows = []
    wall = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10),
        (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
        (4, 7), (4, 9), (4, 10),
    }
    goal = {(2, 10)}
    pot = {(2, 7)}
    recipe = {(0, 5), (4, 5)}
    plate = {(1, 10), (3, 10)}
    ing0 = {(0, 8), (1, 0)}
    ing1 = {(3, 0), (4, 8)}
    ing2 = {(0, 6), (4, 6)}
    for r in range(H):
        row = []
        for c in range(W):
            row.append([
                int((r, c) in wall),
                int((r, c) in goal),
                int((r, c) in pot),
                int((r, c) in recipe),
                0,
                int((r, c) in plate),
                int((r, c) in ing0),
                int((r, c) in ing1),
                int((r, c) in ing2),
            ])
        rows.append(row)
    return rows


def _static_crops():
    static = _build_static_map()
    padded = (
        [[[0] * 9 for _ in range(W + 2 * PAD)] for _ in range(PAD)]
        + [[[0] * 9 for _ in range(PAD)] + row + [[0] * 9 for _ in range(PAD)] for row in static]
        + [[[0] * 9 for _ in range(W + 2 * PAD)] for _ in range(PAD)]
    )
    crops = [
        [
            [
                padded[r + lr][c + lc][ch]
                for ch in range(9)
            ]
            for lr in range(5)
            for lc in range(5)
        ]
        for r in range(H)
        for c in range(W)
    ]
    return jnp.asarray(crops, dtype=jnp.int32).reshape((H * W, 5, 5, 9))


def _mem0():
    return jnp.zeros((6,), dtype=jnp.int32)


def _mem1():
    return jnp.zeros((4,), dtype=jnp.int32)


def init_agent_0():
    return _mem0()


def init_agent_1():
    return _mem1()


def _localize(obs):
    obs_static = obs[:, :, 20:29]
    diffs = jnp.abs(_static_crops() - obs_static[None, :, :, :])
    scores = -jnp.sum(diffs, axis=(1, 2, 3))
    idx = jnp.argmax(scores).astype(jnp.int32)
    return idx // W, idx % W


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _move_for_face(face):
    action = jnp.where(face == FACE_UP, UP, STAY)
    action = jnp.where(face == FACE_DOWN, DOWN, action)
    action = jnp.where(face == FACE_RIGHT, RIGHT, action)
    action = jnp.where(face == FACE_LEFT, LEFT, action)
    return action


def _nav_to(row, col, target_r, target_c):
    return jnp.where(
        col < target_c,
        RIGHT,
        jnp.where(
            col > target_c,
            LEFT,
            jnp.where(row < target_r, DOWN, jnp.where(row > target_r, UP, STAY)),
        ),
    )


def _interact_from(obs, row, col, stand_r, stand_c, face):
    at_stand = (row == stand_r) & (col == stand_c)
    facing = _facing(obs)
    turn = _move_for_face(face)
    return jnp.where(at_stand, jnp.where(facing == face, INTERACT, turn), _nav_to(row, col, stand_r, stand_c))


def _visible_recipe(obs):
    counts = jnp.sum(obs[:, :, 36:39], axis=(0, 1)).astype(jnp.int32)
    has_recipe = jnp.sum(counts) > 0
    recipe = jnp.where(counts[1] > counts[0], jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    return recipe, has_recipe


def _held_counts(obs):
    return obs[2, 2, 7:10].astype(jnp.int32)


def _held_ingredient(obs):
    counts = _held_counts(obs)
    has_ing = jnp.sum(counts) > 0
    ing = jnp.argmax(counts).astype(jnp.int32)
    return ing, has_ing


def _has_plate(obs):
    return obs[2, 2, 5] > 0


def _has_cooked_dish(obs):
    return (obs[2, 2, 5] > 0) & (obs[2, 2, 6] > 0)


def _pot_stats(obs):
    mask = obs[:, :, 22].astype(jnp.int32)
    item = obs[:, :, 29:34].astype(jnp.int32) * mask[:, :, None]
    counts = jnp.sum(item[:, :, 2:5], axis=(0, 1))
    total = jnp.sum(counts).astype(jnp.int32)
    cooked = jnp.sum(item[:, :, 1]) > 0
    timer = jnp.sum(obs[:, :, 39].astype(jnp.int32) * mask)
    visible = jnp.sum(mask) > 0
    return total, cooked, timer, visible


def _pot_ingredient(obs):
    mask = obs[:, :, 22].astype(jnp.int32)
    counts = jnp.sum(obs[:, :, 31:34].astype(jnp.int32) * mask[:, :, None], axis=(0, 1))
    return jnp.argmax(counts).astype(jnp.int32), jnp.sum(counts).astype(jnp.int32)


def _agent0_action(obs, memory):
    row, col = _localize(obs)
    seen_recipe, has_seen_recipe = _visible_recipe(obs)
    known = (memory[1] > 0) | has_seen_recipe
    recipe = jnp.where(has_seen_recipe, seen_recipe, memory[0])
    recipe = jnp.where(known, recipe, jnp.asarray(0, dtype=jnp.int32))

    held_ing, has_ing = _held_ingredient(obs)
    pot_total, pot_cooked, pot_timer, pot_visible = _pot_stats(obs)
    pot_busy = pot_visible & ((pot_total >= 3) | pot_cooked | (pot_timer > 0))
    just_emptied = (memory[3] > 0) & pot_visible & (pot_total == 0) & (~pot_busy)
    cooldown = jnp.maximum(memory[4] - jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    cooldown = jnp.where(just_emptied, jnp.asarray(3, dtype=jnp.int32), cooldown)
    cooling_off = cooldown > 0
    partial_wait = jnp.where(
        pot_visible & (pot_total > 0) & (pot_total < 3) & (~has_ing),
        memory[5] + jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    wait_for_partner_fill = (partial_wait > 0) & (partial_wait <= jnp.asarray(12, dtype=jnp.int32))
    wrong_ing = has_ing & (held_ing != recipe)

    inspect_recipe = _nav_to(row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(5, dtype=jnp.int32))
    drop_wrong = _interact_from(obs, row, col, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(4, dtype=jnp.int32), FACE_UP)
    feed_pot = _interact_from(obs, row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(6, dtype=jnp.int32), FACE_RIGHT)
    get_ing0 = _interact_from(obs, row, col, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32), FACE_LEFT)
    get_ing1 = _interact_from(obs, row, col, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32), FACE_LEFT)
    fetch = jnp.where(recipe == 0, get_ing0, get_ing1)

    action = fetch
    action = jnp.where(wait_for_partner_fill, STAY, action)
    action = jnp.where(pot_busy & (~has_ing), STAY, action)
    action = jnp.where(has_ing, feed_pot, action)
    action = jnp.where(wrong_ing, drop_wrong, action)
    action = jnp.where(cooling_off, STAY, action)
    action = jnp.where(~known, inspect_recipe, action)

    next_memory = jnp.stack([
        recipe,
        jnp.where(known, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
        pot_total,
        jnp.where(pot_busy, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
        cooldown,
        partial_wait,
    ])
    trace = {
        'row': row,
        'col': col,
        'recipe': recipe,
        'known': known,
        'pot_total': pot_total,
        'has_ing': has_ing,
        'cooldown': cooldown,
        'partial_wait': partial_wait,
    }
    return action, next_memory, trace


def _agent1_action(obs, memory):
    del memory
    row, col = _localize(obs)
    pot_total, pot_cooked, pot_timer, pot_visible = _pot_stats(obs)
    pot_ing, pot_count = _pot_ingredient(obs)
    held_ing, holding_raw = _held_ingredient(obs)
    pot_exists = pot_visible & (pot_total > 0)
    pot_ready = pot_visible & pot_cooked & (pot_timer == 0)
    pot_idle_partial = pot_visible & (~pot_cooked) & (pot_timer == 0) & (pot_total > 0) & (pot_total < 3)
    holding_plate = _has_plate(obs) & (~_has_cooked_dish(obs))
    holding_dish = _has_cooked_dish(obs)
    holding_raw = holding_raw & (~holding_plate) & (~holding_dish)

    top_plate = row <= 2
    get_plate_top = _interact_from(obs, row, col, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(9, dtype=jnp.int32), FACE_RIGHT)
    get_plate_bot = _interact_from(obs, row, col, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(9, dtype=jnp.int32), FACE_RIGHT)
    get_plate = jnp.where(top_plate, get_plate_top, get_plate_bot)
    plate_pot = _interact_from(obs, row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(8, dtype=jnp.int32), FACE_LEFT)
    deliver = _interact_from(obs, row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(9, dtype=jnp.int32), FACE_RIGHT)
    get_ing0 = _interact_from(obs, row, col, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(8, dtype=jnp.int32), FACE_UP)
    get_ing1 = _interact_from(obs, row, col, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(8, dtype=jnp.int32), FACE_DOWN)
    get_pot_ing = jnp.where(pot_ing == 0, get_ing0, get_ing1)
    drop_raw_top = _interact_from(obs, row, col, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(9, dtype=jnp.int32), FACE_UP)
    drop_raw_bot = _interact_from(obs, row, col, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(9, dtype=jnp.int32), FACE_DOWN)
    drop_raw = jnp.where(row <= 2, drop_raw_top, drop_raw_bot)

    wait_pot = _nav_to(row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(8, dtype=jnp.int32))
    raw_to_pot = jnp.where(
        pot_visible & (~pot_cooked) & (pot_timer == 0) & (pot_total < 3) & (held_ing == pot_ing),
        plate_pot,
        drop_raw,
    )

    action = wait_pot
    action = jnp.where(pot_idle_partial & (~holding_raw), get_pot_ing, action)
    action = jnp.where((pot_total >= 3) | (pot_timer > 0) | pot_ready, get_plate, action)
    action = jnp.where(holding_raw, raw_to_pot, action)
    action = jnp.where(holding_plate, jnp.where(pot_ready, plate_pot, wait_pot), action)
    action = jnp.where(holding_dish, deliver, action)
    action = jnp.where((~holding_plate) & (~holding_dish) & pot_ready, get_plate, action)
    action = jnp.where((~holding_plate) & (~holding_dish) & (~holding_raw) & (~pot_exists), wait_pot, action)

    next_memory = jnp.stack([
        pot_total,
        jnp.where(pot_ready, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
        pot_count,
        jnp.where(holding_dish, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
    ])
    trace = {
        'row': row,
        'col': col,
        'pot_total': pot_total,
        'pot_ready': pot_ready,
        'plate': holding_plate,
        'dish': holding_dish,
        'raw': holding_raw,
    }
    return action, next_memory, trace


def agent_0_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, _mem0(), memory)
    return _agent0_action(obs, base)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, _mem1(), memory)
    return _agent1_action(obs, base)
