"""Convention-readable Overcooked policy for demo_cook_simple."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'recipe_cook_courier',
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


_CANDIDATE_POSITIONS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    (1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9),
)


def _infer_pos(obs):
    positions = jnp.asarray(_CANDIDATE_POSITIONS, dtype=jnp.int32)
    local_r = jnp.arange(5, dtype=jnp.int32)
    local_c = jnp.arange(5, dtype=jnp.int32)
    global_r = positions[:, 0, None, None] + local_r[None, :, None] - 2
    global_c = positions[:, 1, None, None] + local_c[None, None, :] - 2

    def at(coords):
        points = jnp.asarray(coords, dtype=jnp.int32)
        same_r = global_r[:, :, :, None] == points[None, None, None, :, 0]
        same_c = global_c[:, :, :, None] == points[None, None, None, :, 1]
        return jnp.any(same_r & same_c, axis=3).astype(jnp.int32)

    wall = at((
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10),
        (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
        (4, 7), (4, 9), (4, 10),
    ))
    goal = at(((2, 10),))
    pot = at(((2, 7),))
    recipe = at(((0, 5), (4, 5)))
    button = jnp.zeros_like(wall)
    plate = at(((1, 10), (3, 10)))
    ing0 = at(((0, 8), (1, 0)))
    ing1 = at(((3, 0), (4, 8)))
    ing2 = at(((0, 6), (4, 6)))
    expected = jnp.stack([wall, goal, pot, recipe, button, plate, ing0, ing1, ing2], axis=3)
    static = obs[:, :, 20:29]
    diff = jnp.sum(jnp.abs(expected - static), axis=(1, 2, 3))
    idx = jnp.argmin(diff).astype(jnp.int32)
    pos = positions[idx]
    return pos[0], pos[1], idx


def _facing(obs):
    dirs = obs[2, 2, 1:5]
    return jnp.argmax(dirs).astype(jnp.int32)


def _held_counts(obs):
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ings = obs[2, 2, 7:10]
    held_ing_total = jnp.sum(held_ings)
    held_ing = jnp.argmax(held_ings).astype(jnp.int32)
    empty = (~held_plate) & (~held_cooked) & (held_ing_total == 0)
    return held_plate, held_cooked, held_ings, held_ing_total, held_ing, empty


def _pot_info(obs):
    pot_mask = obs[:, :, 22] > 0
    visible = jnp.sum(pot_mask.astype(jnp.int32)) > 0
    counts = jnp.sum(obs[:, :, 31:34] * pot_mask[:, :, None].astype(jnp.int32), axis=(0, 1))
    cooked = jnp.sum((obs[:, :, 30] > 0).astype(jnp.int32) * pot_mask.astype(jnp.int32)) > 0
    timer = jnp.sum(obs[:, :, 39] * pot_mask.astype(jnp.int32))
    total = jnp.sum(counts)
    return visible, counts, total, cooked, timer


def _recipe_target(obs, old_target, old_known):
    rec0 = jnp.sum(obs[:, :, 36])
    rec1 = jnp.sum(obs[:, :, 37])
    visible = (rec0 + rec1) > 0
    seen_target = jnp.where(rec1 > rec0, 1, 0).astype(jnp.int32)
    target = jnp.where(visible, seen_target, old_target).astype(jnp.int32)
    known = jnp.where(visible, 1, old_known).astype(jnp.int32)
    return target, known


def _nav_to(row, col, tr, tc):
    horizontal = jnp.where(col < tc, RIGHT, LEFT)
    vertical = jnp.where(row < tr, DOWN, UP)
    return jnp.where(col != tc, horizontal, jnp.where(row != tr, vertical, STAY))


def _face_or_interact(row, col, facing, tr, tc, desired_facing, face_action):
    at_target = (row == tr) & (col == tc)
    ready = facing == desired_facing
    nav = _nav_to(row, col, tr, tc)
    return jnp.where(at_target, jnp.where(ready, INTERACT, face_action), nav)


def _left_stand_and_face_for_ing(ing):
    is_one = ing == 1
    return (
        jnp.where(is_one, 3, 1).astype(jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
        LEFT,
    )


def _agent0_action(obs, memory):
    row, col, pos_idx = _infer_pos(obs)
    facing = _facing(obs)
    old_target = memory[0]
    old_known = memory[1]
    old_count = memory[2]
    target, known = _recipe_target(obs, old_target, old_known)
    held_plate, held_cooked, held_ings, held_ing_total, held_ing, empty = _held_counts(obs)
    pot_visible, pot_counts, pot_total_seen, pot_cooked, pot_timer = _pot_info(obs)
    pot_total = jnp.where(pot_visible, pot_total_seen, old_count).astype(jnp.int32)
    has_wrong_pot_item = jnp.sum(pot_counts) != pot_counts[target]
    useful_partial = (pot_total < 3) & (~pot_cooked) & (pot_timer == 0) & (~has_wrong_pot_item)

    stand_r, stand_c, face_dir, face_action = _left_stand_and_face_for_ing(target)
    fetch = _face_or_interact(row, col, facing, stand_r, stand_c, face_dir, face_action)
    add_to_pot = _face_or_interact(row, col, facing, 2, 6, 2, RIGHT)
    read_pos = _nav_to(row, col, 2, 6)

    holding_target = (held_ing_total > 0) & (held_ings[target] > 0)
    should_add = holding_target & useful_partial
    should_fetch = empty & (known > 0) & useful_partial
    action = jnp.where(
        should_add,
        add_to_pot,
        jnp.where(should_fetch, fetch, read_pos),
    ).astype(jnp.int32)

    did_add = should_add & (action == INTERACT) & (row == 2) & (col == 6) & (facing == 2)
    next_count = jnp.where(pot_visible, pot_total_seen, old_count)
    next_count = jnp.where(did_add, jnp.minimum(next_count + 1, 3), next_count).astype(jnp.int32)
    next_memory = jnp.stack([target, known, next_count]).astype(jnp.int32)
    mode = jnp.where(should_add, 2, jnp.where(should_fetch, 1, 0)).astype(jnp.int32)
    trace = {
        'row': row,
        'col': col,
        'pos_idx': pos_idx,
        'target': target,
        'known': known,
        'pot_total': pot_total,
        'mode': mode,
    }
    return action, next_memory, trace


def init_agent_0():
    return jnp.asarray([0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero = init_agent_0()
    active_memory = jnp.where(reset, zero, memory)
    action, next_memory, trace = _agent0_action(obs, active_memory)
    next_memory = jnp.where(reset, zero, next_memory)
    return action, next_memory, trace


def _right_supply_for_ing(ing):
    is_one = ing == 1
    return (
        jnp.where(is_one, 3, 1).astype(jnp.int32),
        jnp.asarray(8, dtype=jnp.int32),
        jnp.where(is_one, 1, 0).astype(jnp.int32),
        jnp.where(is_one, DOWN, UP).astype(jnp.int32),
    )


def _agent1_action(obs, memory):
    row, col, pos_idx = _infer_pos(obs)
    facing = _facing(obs)
    held_plate, held_cooked, held_ings, held_ing_total, held_ing, empty = _held_counts(obs)
    pot_visible, pot_counts, pot_total, pot_cooked, pot_timer = _pot_info(obs)
    pot_one_kind = jnp.sum((pot_counts > 0).astype(jnp.int32)) == 1
    pot_kind = jnp.argmax(pot_counts).astype(jnp.int32)
    pot_partial = pot_visible & pot_one_kind & (pot_kind < 2) & (pot_total == 1) & (~pot_cooked) & (pot_timer == 0)
    partner_counts = jnp.sum(obs[:, :, 17:20], axis=(0, 1))
    partner_total = jnp.sum(partner_counts)
    partner_kind = jnp.argmax(partner_counts).astype(jnp.int32)
    partner_signal = (
        pot_visible
        & (pot_total == 0)
        & (~pot_cooked)
        & (pot_timer == 0)
        & (partner_total == 1)
        & (partner_kind < 2)
    )
    assist_kind = jnp.where(pot_partial, pot_kind, partner_kind).astype(jnp.int32)

    top_plate = row <= 2
    plate_r = jnp.where(top_plate, 1, 3).astype(jnp.int32)
    get_plate = _face_or_interact(row, col, facing, plate_r, 9, 2, RIGHT)
    pickup_pot = _face_or_interact(row, col, facing, 2, 8, 3, LEFT)
    deliver = _face_or_interact(row, col, facing, 2, 9, 2, RIGHT)
    wait_pot = _nav_to(row, col, 2, 8)

    supply_r, supply_c, supply_face, supply_face_action = _right_supply_for_ing(assist_kind)
    fetch_ing = _face_or_interact(row, col, facing, supply_r, supply_c, supply_face, supply_face_action)
    add_ing = _face_or_interact(row, col, facing, 2, 8, 3, LEFT)

    holding_ing = held_ing_total > 0
    can_add_held = holding_ing & (pot_total < 3) & (~pot_cooked) & (pot_timer == 0)
    should_fetch_ing = empty & (pot_partial | partner_signal)
    should_get_plate = empty & pot_visible & ((pot_total >= 2) | (pot_timer > 0) | pot_cooked)
    should_pickup = held_plate & pot_cooked

    action = jnp.where(
        held_cooked,
        deliver,
        jnp.where(
            should_pickup,
            pickup_pot,
            jnp.where(
                can_add_held,
                add_ing,
                jnp.where(
                    should_fetch_ing,
                    fetch_ing,
                    jnp.where(should_get_plate, get_plate, wait_pot),
                ),
            ),
        ),
    ).astype(jnp.int32)

    mode = jnp.where(
        held_cooked,
        4,
        jnp.where(should_pickup, 3, jnp.where(can_add_held, 2, jnp.where(should_fetch_ing, 1, jnp.where(should_get_plate, 5, 0)))),
    ).astype(jnp.int32)
    next_memory = memory
    trace = {
        'row': row,
        'col': col,
        'pos_idx': pos_idx,
        'pot_total': pot_total,
        'pot_kind': pot_kind,
        'assist_kind': assist_kind,
        'mode': mode,
    }
    return action, next_memory, trace


def init_agent_1():
    return jnp.asarray([0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    zero = init_agent_1()
    active_memory = jnp.where(reset, zero, memory)
    action, next_memory, trace = _agent1_action(obs, active_memory)
    next_memory = jnp.where(reset, zero, next_memory)
    return action, next_memory, trace
