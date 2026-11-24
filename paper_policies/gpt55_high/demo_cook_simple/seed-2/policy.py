"""Convention-light scripted policy for the demo_cook_simple OCV2 layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'role_script_xp',
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

DIR_RIGHT = 2
DIR_LEFT = 3
DIR_UP = 0
DIR_DOWN = 1
DIR_TO_ACTION = (3, 1, 0, 2)

UNKNOWN_TARGET = 2
WAIT_AFTER_THIRD_ADD = 0
WAIT_AFTER_EMPTY_POT = 1
WAIT_FOR_PARTNER_ADD = 12
PLATE_PREFETCH_IDLE = 24


def _make_templates(positions):
    features = (
        (0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0),
        (0, 7, 0), (0, 9, 0), (0, 10, 0), (1, 7, 0), (2, 0, 0),
        (3, 7, 0), (4, 0, 0), (4, 1, 0), (4, 2, 0), (4, 3, 0),
        (4, 4, 0), (4, 7, 0), (4, 9, 0), (4, 10, 0),
        (2, 10, 1), (2, 7, 2),
        (0, 5, 3), (4, 5, 3),
        (1, 10, 5), (3, 10, 5),
        (0, 8, 6), (1, 0, 6),
        (3, 0, 7), (4, 8, 7),
        (0, 6, 8), (4, 6, 8),
    )
    crops = []
    for row, col in positions:
        crop = [[[0 for _ in range(9)] for _ in range(5)] for _ in range(5)]
        for obj_row, obj_col, chan in features:
            local_row = obj_row - row + 2
            local_col = obj_col - col + 2
            if 0 <= local_row < 5 and 0 <= local_col < 5:
                crop[local_row][local_col][chan] = 1
        crops.append(crop)
    return jnp.asarray(crops, dtype=jnp.int32)


AGENT0_POSITIONS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
)
AGENT1_POSITIONS = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))


def _locate(obs, positions):
    templates = _make_templates(positions)
    positions_array = jnp.asarray(positions, dtype=jnp.int32)
    static_obs = obs[:, :, 20:29]
    diffs = jnp.sum(jnp.abs(templates - static_obs), axis=(1, 2, 3))
    idx = jnp.argmin(diffs)
    return positions_array[idx, 0], positions_array[idx, 1]


def _self_facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _go_to(row, col, target_row, target_col):
    return jnp.where(
        col < target_col,
        RIGHT,
        jnp.where(
            col > target_col,
            LEFT,
            jnp.where(row < target_row, DOWN, jnp.where(row > target_row, UP, STAY)),
        ),
    )


def _face_then_interact(obs, desired_dir):
    facing = _self_facing(obs)
    face_action = jnp.asarray(DIR_TO_ACTION, dtype=jnp.int32)[desired_dir]
    return jnp.where(facing == desired_dir, INTERACT, face_action)


def _go_face_interact(obs, row, col, target_row, target_col, desired_dir):
    move = _go_to(row, col, target_row, target_col)
    at_target = (row == target_row) & (col == target_col)
    return jnp.where(at_target, _face_then_interact(obs, desired_dir), move)


def _visible_target(obs, old_target):
    recipe0_visible = jnp.any(obs[:, :, 36] > 0)
    recipe1_visible = jnp.any(obs[:, :, 37] > 0)
    return jnp.where(
        recipe0_visible,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(recipe1_visible, jnp.asarray(1, dtype=jnp.int32), old_target),
    )


def _pot_nonempty_visible(obs):
    pot_mask = obs[:, :, 22] > 0
    pot_item_total = jnp.sum(
        (obs[:, :, 29] + obs[:, :, 30] + obs[:, :, 31] + obs[:, :, 32] + obs[:, :, 33])
        * pot_mask
    )
    pot_timer = jnp.sum(obs[:, :, 39] * pot_mask)
    return jnp.any(pot_mask) & ((pot_item_total > 0) | (pot_timer > 0))


def _pot_stats(obs):
    pot_mask = obs[:, :, 22] > 0
    ing0 = jnp.sum(obs[:, :, 31] * pot_mask)
    ing1 = jnp.sum(obs[:, :, 32] * pot_mask)
    ing2 = jnp.sum(obs[:, :, 33] * pot_mask)
    ingredients = jnp.asarray([ing0, ing1, ing2], dtype=jnp.int32)
    total = jnp.sum(ingredients)
    timer = jnp.sum(obs[:, :, 39] * pot_mask)
    cooked = jnp.any((obs[:, :, 30] > 0) & pot_mask)
    return jnp.any(pot_mask), total, timer, cooked, jnp.argmax(ingredients).astype(jnp.int32)


def _held_ingredient(obs):
    held_counts = obs[2, 2, 7:10]
    return jnp.argmax(held_counts).astype(jnp.int32), jnp.any(held_counts > 0)


def _held_plate(obs):
    return obs[2, 2, 5] > 0


def _held_cooked(obs):
    return obs[2, 2, 6] > 0


def init_agent_0():
    return jnp.asarray([2, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    base = init_agent_0()
    memory = jnp.where(reset, base, memory)

    row, col = _locate(obs, AGENT0_POSITIONS)
    target = _visible_target(obs, memory[0])
    inserted = memory[1]
    wait = memory[2]
    delivery_grace = memory[3]
    held_kind, has_ingredient = _held_ingredient(obs)

    pot_visible, pot_total, pot_timer, pot_cooked, pot_kind = _pot_stats(obs)
    pot_busy_visible = _pot_nonempty_visible(obs)
    pot_fullish = pot_visible & ((pot_total >= 3) | (pot_timer > 0) | pot_cooked)
    pot_partial_progress = (
        pot_visible
        & (pot_total > inserted)
        & (pot_total < 3)
        & (pot_timer == 0)
        & (~pot_cooked)
    )
    inserted = jnp.where(pot_partial_progress, pot_total, inserted)
    inserted = jnp.where(pot_fullish, jnp.asarray(3, dtype=jnp.int32), inserted)
    finished_seen = (inserted >= 3) & (wait <= 0) & (~pot_busy_visible)
    inserted = jnp.where(finished_seen, jnp.asarray(0, dtype=jnp.int32), inserted)
    delivery_grace = jnp.where(finished_seen, WAIT_AFTER_EMPTY_POT, delivery_grace)

    source_row = jnp.where(target == 0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    source_col = jnp.asarray(1, dtype=jnp.int32)
    target_known = target != UNKNOWN_TARGET
    waiting_for_helper = (inserted > 0) & (wait > 0) & pot_busy_visible
    busy = (inserted >= 3) | waiting_for_helper | (delivery_grace > 0)
    wrong_ingredient = has_ingredient & target_known & (held_kind != target)

    read_action = _go_to(row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(6, dtype=jnp.int32))
    pickup_action = _go_face_interact(obs, row, col, source_row, source_col, DIR_LEFT)
    pot_action = _go_face_interact(
        obs,
        row,
        col,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(6, dtype=jnp.int32),
        DIR_RIGHT,
    )
    drop_action = _go_face_interact(
        obs,
        row,
        col,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(4, dtype=jnp.int32),
        jnp.asarray(DIR_UP, dtype=jnp.int32),
    )

    action = jnp.where(
        (~target_known) | busy,
        read_action,
        jnp.where(
            wrong_ingredient,
            drop_action,
            jnp.where(has_ingredient, pot_action, pickup_action),
        ),
    )

    at_pot = (row == 2) & (col == 6)
    add_attempt = (
        (action == INTERACT)
        & at_pot
        & (_self_facing(obs) == DIR_RIGHT)
        & has_ingredient
        & (inserted < 3)
        & (~wrong_ingredient)
    )
    new_inserted = jnp.where(add_attempt, jnp.minimum(inserted + 1, 3), inserted)
    just_filled = add_attempt & (new_inserted >= 3)
    new_wait = jnp.where(wait > 0, wait - 1, wait)
    new_wait = jnp.where(
        add_attempt & (new_inserted < 3),
        jnp.asarray(WAIT_FOR_PARTNER_ADD, dtype=jnp.int32),
        new_wait,
    )
    new_wait = jnp.where(just_filled, WAIT_AFTER_THIRD_ADD, new_wait)
    new_grace = jnp.where((delivery_grace > 0) & (~finished_seen), delivery_grace - 1, delivery_grace)

    next_memory = jnp.asarray([target, new_inserted, new_wait, new_grace], dtype=jnp.int32)
    trace = {
        'target': target,
        'inserted': new_inserted,
        'wait': new_wait,
        'grace': new_grace,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    row, col = _locate(obs, AGENT1_POSITIONS)

    has_plate = _held_plate(obs)
    has_cooked = _held_cooked(obs)
    held_kind, has_ingredient = _held_ingredient(obs)
    pot_visible, pot_total, pot_timer, pot_cooked, pot_kind = _pot_stats(obs)
    pot_partial = pot_visible & (pot_total > 0) & (pot_total < 3) & (pot_timer == 0) & (~pot_cooked)
    pot_full_or_cooking = pot_visible & ((pot_total >= 3) | (pot_timer > 0) | pot_cooked)
    empty_pot_idle = (
        pot_visible
        & (pot_total == 0)
        & (pot_timer == 0)
        & (~pot_cooked)
        & (~has_plate)
        & (~has_cooked)
        & (~has_ingredient)
    )
    idle_count = jnp.where(empty_pot_idle, memory + 1, jnp.asarray(0, dtype=jnp.int32))
    prefetch_plate = idle_count >= PLATE_PREFETCH_IDLE

    get_plate = _go_face_interact(
        obs,
        row,
        col,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(9, dtype=jnp.int32),
        DIR_RIGHT,
    )
    plate_from_pot = _go_face_interact(
        obs,
        row,
        col,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(8, dtype=jnp.int32),
        DIR_LEFT,
    )
    deliver = _go_face_interact(
        obs,
        row,
        col,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(9, dtype=jnp.int32),
        DIR_RIGHT,
    )
    wait_at_pot = _go_to(row, col, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(8, dtype=jnp.int32))
    ingredient_source_row = jnp.where(
        pot_kind == 0,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
    )
    ingredient_source_dir = jnp.where(
        pot_kind == 0,
        jnp.asarray(DIR_UP, dtype=jnp.int32),
        jnp.asarray(DIR_DOWN, dtype=jnp.int32),
    )
    get_matching_ingredient = _go_face_interact(
        obs,
        row,
        col,
        ingredient_source_row,
        jnp.asarray(8, dtype=jnp.int32),
        ingredient_source_dir,
    )
    add_ingredient = _go_face_interact(
        obs,
        row,
        col,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(8, dtype=jnp.int32),
        DIR_LEFT,
    )

    pot_cooked_visible = jnp.any((obs[:, :, 22] > 0) & (obs[:, :, 30] > 0) & (obs[:, :, 39] == 0))
    plate_action = jnp.where(pot_cooked_visible, plate_from_pot, wait_at_pot)
    empty_hand_action = jnp.where(
        pot_partial,
        get_matching_ingredient,
        jnp.where(pot_full_or_cooking | prefetch_plate, get_plate, wait_at_pot),
    )
    ingredient_action = jnp.where(pot_partial, add_ingredient, wait_at_pot)
    action = jnp.where(
        has_cooked,
        deliver,
        jnp.where(
            has_plate,
            plate_action,
            jnp.where(has_ingredient, ingredient_action, empty_hand_action),
        ),
    )
    trace = {
        'has_plate': has_plate,
        'has_cooked': has_cooked,
        'pot_ready': pot_cooked_visible,
        'pot_total': pot_total,
        'idle': idle_count,
    }
    return action.astype(jnp.int32), idle_count, trace
