"""Deterministic role policy for the demo_cook_wide Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'station_role_xp', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

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


def _make_patterns(positions):
    static = [[[0 for _ in range(10)] for _ in range(11)] for _ in range(6)]

    walls = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10),
        (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
        (3, 0), (3, 10), (4, 3), (4, 7),
        (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10),
    ]
    for r, c in walls:
        static[r][c][0] = 1
    for r, c in [(0, 5)]:
        static[r][c][1] = 1
    for r, c in [(2, 5)]:
        static[r][c][2] = 1
    for r, c in [(4, 5)]:
        static[r][c][3] = 1
    for r, c in [(0, 4), (0, 6)]:
        static[r][c][5] = 1
    for r, c in [(1, 3), (4, 0), (4, 10)]:
        static[r][c][6] = 1
    for r, c in [(1, 7), (5, 1), (5, 9)]:
        static[r][c][7] = 1
    for r, c in [(4, 4), (4, 6)]:
        static[r][c][9] = 1

    patterns = []
    for pr, pc in positions:
        crop = []
        for lr in range(5):
            row = []
            for lc in range(5):
                gr = pr + lr - 2
                gc = pc + lc - 2
                if 0 <= gr < 6 and 0 <= gc < 11:
                    row.append(static[gr][gc])
                else:
                    row.append([0 for _ in range(10)])
            crop.append(row)
        patterns.append(crop)
    return jnp.asarray(patterns, dtype=jnp.int32)


AGENT0_POSITIONS = ((1, 4), (1, 5), (1, 6))
AGENT1_POSITIONS = (
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 1), (4, 2), (4, 8), (4, 9),
)


def _locate(obs, patterns, positions, default_pos):
    static_crop = obs[:, :, 22:32]
    matches = jnp.all(patterns == static_crop[None, :, :, :], axis=(1, 2, 3))
    idx = jnp.argmax(matches.astype(jnp.int32))
    found = jnp.any(matches)
    return jnp.where(found, positions[idx], default_pos)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_or_interact(obs, desired_dir):
    facing = _facing(obs)
    turn_action = jnp.where(
        desired_dir == DIR_UP,
        UP,
        jnp.where(desired_dir == DIR_DOWN, DOWN, jnp.where(desired_dir == DIR_RIGHT, RIGHT, LEFT)),
    )
    return jnp.where(facing == desired_dir, INTERACT, turn_action)


def _held_plate(obs):
    return obs[2, 2, 5] > 0


def _held_cooked(obs):
    return obs[2, 2, 6] > 0


def _held_raw(obs):
    item_counts = jnp.sum(obs[2, 2, 7:11])
    return (item_counts > 0) & (obs[2, 2, 5] == 0) & (obs[2, 2, 6] == 0)


def _pot_status(obs):
    pot = obs[:, :, 24]
    cooked = jnp.max(pot * obs[:, :, 33]) > 0
    ing0 = jnp.max(pot * obs[:, :, 34])
    ing1 = jnp.max(pot * obs[:, :, 35])
    ing2 = jnp.max(pot * obs[:, :, 36])
    ing3 = jnp.max(pot * obs[:, :, 37])
    count = ing0 + ing1 + ing2 + ing3
    timer = jnp.max(pot * obs[:, :, 44])
    empty = (count == 0) & (timer == 0) & (~cooked)
    return count, ing0, ing1, cooked, timer, empty


def _recipe_update(obs, memory):
    recipe0 = jnp.max(obs[:, :, 40])
    recipe1 = jnp.max(obs[:, :, 41])
    seen = (recipe0 + recipe1) > 0
    target = jnp.where(recipe1 > recipe0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    next_target = jnp.where(seen, target, memory[0])
    next_known = jnp.where(seen, jnp.asarray(1, dtype=jnp.int32), memory[1])
    return next_target, next_known


def _top_move_to_col(col, target_col):
    return jnp.where(col < target_col, RIGHT, jnp.where(col > target_col, LEFT, STAY))


def init_agent_0():
    # helped-this-cycle flag, previous raw-add attempt
    return jnp.asarray([0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    base_memory = jnp.where(reset, init_agent_0(), memory)
    positions = jnp.asarray(AGENT0_POSITIONS, dtype=jnp.int32)
    patterns = _make_patterns(AGENT0_POSITIONS)
    pos = _locate(obs, patterns, positions, jnp.asarray([1, 5], dtype=jnp.int32))
    col = pos[1]

    cooked = _held_cooked(obs)
    plate = _held_plate(obs)
    raw = _held_raw(obs)
    pot_count, pot_ing0, pot_ing1, pot_cooked, pot_timer, pot_empty = _pot_status(obs)
    raw_drop_succeeded = (base_memory[1] > 0) & (~raw)
    helped = jnp.where(raw_drop_succeeded, jnp.asarray(1, dtype=jnp.int32), base_memory[0])
    helped = jnp.where(pot_empty, jnp.asarray(0, dtype=jnp.int32), helped)

    target_plate_col = jnp.where(col > 5, jnp.asarray(6, dtype=jnp.int32), jnp.asarray(4, dtype=jnp.int32))
    help_ingredient = jnp.where(pot_ing1 > pot_ing0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    help_col = jnp.where(help_ingredient == 0, jnp.asarray(4, dtype=jnp.int32), jnp.asarray(6, dtype=jnp.int32))
    help_dir = jnp.where(help_ingredient == 0, DIR_LEFT, DIR_RIGHT)
    should_help = (pot_count == 1) & (helped == 0)
    wait_empty = (pot_count == 0) & (pot_timer == 0) & (~pot_cooked)

    move_to_center = _top_move_to_col(col, jnp.asarray(5, dtype=jnp.int32))
    deliver = jnp.where(col == 5, _face_or_interact(obs, DIR_UP), move_to_center)
    plate_pot = jnp.where(col == 5, _face_or_interact(obs, DIR_DOWN), move_to_center)
    feed_pot = jnp.where(col == 5, _face_or_interact(obs, DIR_DOWN), move_to_center)
    fetch_help = jnp.where(
        col == help_col,
        _face_or_interact(obs, help_dir),
        _top_move_to_col(col, help_col),
    )
    get_plate = jnp.where(
        col == target_plate_col,
        _face_or_interact(obs, DIR_UP),
        _top_move_to_col(col, target_plate_col),
    )
    empty_action = jnp.where(should_help, fetch_help, jnp.where(wait_empty, move_to_center, get_plate))
    action = jnp.where(cooked, deliver, jnp.where(plate, plate_pot, jnp.where(raw, feed_pot, empty_action)))

    add_intent = (raw & (col == 5) & (_facing(obs) == DIR_DOWN) & (action == INTERACT)).astype(jnp.int32)
    next_memory = jnp.stack([helped, add_intent]).astype(jnp.int32)
    trace = {
        'row': pos[0],
        'col': pos[1],
        'held_plate': plate.astype(jnp.int32),
        'held_cooked': cooked.astype(jnp.int32),
        'held_raw': raw.astype(jnp.int32),
        'helped': helped,
        'pot_count': pot_count,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # target ingredient, recipe known, deposited count, previous add attempt, full-pot wait timer
    return jnp.asarray([0, 0, 0, 0, 0], dtype=jnp.int32)


def _bottom_move_to(pos, target_row, target_col):
    row = pos[0]
    col = pos[1]
    same_spur = ((target_col <= 2) & (col <= 2)) | ((target_col >= 8) & (col >= 8))
    at_target = (row == target_row) & (col == target_col)

    row4_action = jnp.where(
        at_target,
        STAY,
        jnp.where(
            same_spur & (col < target_col),
            RIGHT,
            jnp.where(same_spur & (col > target_col), LEFT, UP),
        ),
    )
    row3_action = jnp.where(
        col < target_col,
        RIGHT,
        jnp.where(col > target_col, LEFT, jnp.where(target_row > row, DOWN, STAY)),
    )
    return jnp.where(row == 4, row4_action, row3_action)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base_memory = jnp.where(reset, init_agent_1(), memory)
    positions = jnp.asarray(AGENT1_POSITIONS, dtype=jnp.int32)
    patterns = _make_patterns(AGENT1_POSITIONS)
    pos = _locate(obs, patterns, positions, jnp.asarray([3, 5], dtype=jnp.int32))
    row = pos[0]
    col = pos[1]

    target, known = _recipe_update(obs, base_memory)
    was_add_attempt = base_memory[3] > 0
    now_holding_raw = _held_raw(obs)
    add_succeeded = was_add_attempt & (~now_holding_raw)
    deposited = jnp.where(add_succeeded, jnp.minimum(base_memory[2] + 1, 3), base_memory[2])
    pot_count, unused_i0, unused_i1, unused_cooked, pot_timer, pot_empty = _pot_status(obs)
    del unused_i0, unused_i1, unused_cooked
    pot_is_filled_or_cooking = (pot_count >= 3) | (pot_timer > 0)
    deposited = jnp.where(pot_is_filled_or_cooking, jnp.asarray(3, dtype=jnp.int32), deposited)
    wait_timer = jnp.where(deposited >= 3, base_memory[4] + 1, jnp.asarray(0, dtype=jnp.int32))
    del pot_count
    partner_cooked = jnp.max(obs[:, :, 17]) > 0
    observed_delivery_done = (wait_timer > 20) & pot_empty & (~partner_cooked) & (pot_timer == 0)
    cycle_done = (deposited >= 3) & (observed_delivery_done | (wait_timer > 34))
    deposited = jnp.where(cycle_done, jnp.asarray(0, dtype=jnp.int32), deposited)
    wait_timer = jnp.where(cycle_done, jnp.asarray(0, dtype=jnp.int32), wait_timer)

    holding_raw = _held_raw(obs)
    full_wait = deposited >= 3
    need_recipe = known == 0
    use_left_supply = col <= 5
    pickup_col = jnp.where(use_left_supply, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(9, dtype=jnp.int32))
    pickup_dir = jnp.where(
        target == 1,
        DIR_DOWN,
        jnp.where(use_left_supply, DIR_LEFT, DIR_RIGHT),
    )

    move_recipe = _bottom_move_to(pos, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(5, dtype=jnp.int32))
    move_pot = _bottom_move_to(pos, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(5, dtype=jnp.int32))
    move_pickup = _bottom_move_to(pos, jnp.asarray(4, dtype=jnp.int32), pickup_col)

    at_pot = (row == 3) & (col == 5)
    at_pickup = (row == 4) & (col == pickup_col)
    feed_pot = jnp.where(at_pot, _face_or_interact(obs, DIR_UP), move_pot)
    pickup = jnp.where(at_pickup, _face_or_interact(obs, pickup_dir), move_pickup)
    waiting = jnp.where(at_pot, STAY, move_pot)
    action = jnp.where(
        full_wait,
        waiting,
        jnp.where(holding_raw, feed_pot, jnp.where(need_recipe, move_recipe, pickup)),
    )

    add_intent = (holding_raw & at_pot & (_facing(obs) == DIR_UP) & (action == INTERACT)).astype(jnp.int32)
    next_memory = jnp.stack([target, known, deposited, add_intent, wait_timer]).astype(jnp.int32)
    trace = {
        'row': pos[0],
        'col': pos[1],
        'target': target,
        'known': known,
        'deposited': deposited,
        'wait_timer': wait_timer,
        'held_raw': holding_raw.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
