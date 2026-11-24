"""Cooperation grounded in visible ingredients and pot contents."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_with_recovery_space', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

# Immutable geometry from the public specification. Runtime localization compares
# the static observation to each possible floor-cell crop.
_LAYOUT = ('WWWWBXBWWWW', 'WWW0 A 1WWW', 'WWWWWPWWWWW',
           'W    A    W', '0  W3R3W  0', 'W1WWWWWWW1W')
_STATIC = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'L': 4, 'B': 5,
           '0': 6, '1': 7, '2': 8, '3': 9}
def localize_lower(obs):
    lower = tuple((r, c) for r in range(3, 5) for c in range(11)
                  if _LAYOUT[r][c] in ' A')
    templates = tuple(tuple(tuple(tuple(
        int(0 <= r+i-2 < 6 and 0 <= c+j-2 < 11 and
            _STATIC.get(_LAYOUT[max(0, min(5, r+i-2))][max(0, min(10, c+j-2))], -1) == k)
        for k in range(10)) for j in range(5)) for i in range(5)) for r, c in lower)
    scores = jnp.sum(jnp.abs(jnp.asarray(templates, jnp.int32) - obs[None, :, :, 22:32]), axis=(1, 2, 3))
    return jnp.asarray(lower, jnp.int32)[jnp.argmin(scores)]


def face_or_interact(direction, desired):
    return jnp.where(direction == desired, 5, jnp.asarray([3, 1, 0, 2], jnp.int32)[desired])


def upper_to(x, direction, target_x, desired):
    return jnp.where(x < target_x, 0, jnp.where(x > target_x, 2, face_or_interact(direction, desired)))


def lower_to(row, x, direction, target_row, target_x, desired):
    # All long routes use the unbroken corridor at row 3.
    lateral = jnp.where(x < target_x, 0, 2)
    move = jnp.where(x != target_x, jnp.where(row == 4, 3, lateral),
                     jnp.where(row < target_row, 1, 3))
    return jnp.where((row == target_row) & (x == target_x), face_or_interact(direction, desired), move)


def decode(obs):
    inv = obs[2, 2, 5:11]
    direction = jnp.argmax(obs[2, 2, 1:5])
    pot = jnp.sum(obs[:, :, 32:38] * obs[:, :, 24:25], axis=(0, 1))
    timer = jnp.sum(obs[:, :, 44])
    other = jnp.sum(obs[:, :, 16:22], axis=(0, 1))
    return inv, direction, pot, timer, other


def init_agent_0():
    # time, recipe, fresh-hand permission, prior pot count, pot stall,
    # time since delivery, prior carried ingredient, stationary cue duration
    return jnp.asarray([0, -1, 1, 0, 0, 100, -1, 0, 0,
                        -1, 0, 0, 0, -1, -1, 0, -1, 0, 0, -1], jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_0(), memory)
    inv, direction, pot, timer, other = decode(obs)
    pj = jnp.argmax(jnp.sum(obs[:, :, 24], axis=0))
    x = 7 - pj
    n = jnp.sum(pot[2:])
    held_raw = (jnp.sum(inv[2:]) > 0) & (inv[0] == 0)
    visible_partner = jnp.sum(obs[:, :, 11]) > 0
    other_raw = (jnp.sum(other[2:4]) == 1) & (jnp.sum(other[4:]) == 0) & (other[0] == 0)
    other_type = jnp.argmax(other[2:])
    other_x = x + jnp.argmax(jnp.sum(obs[:, :, 11], axis=0)) - 2
    other_direction = jnp.argmax(jnp.sum(obs[:, :, 12:16], axis=(0, 1)))
    side_cue = jnp.where(other_x < 5, 0, jnp.where(other_x > 5, 1,
                         jnp.where((other_direction == 3) | (other_direction == 0), 0, 1)))
    side_cue = jnp.where(visible_partner, side_cue, -1)
    side_stable = jnp.where((side_cue >= 0) & (side_cue == memory[9]), memory[10]+1, 0)
    response = (memory[13] >= 0) & (side_cue >= 0) & (side_cue != memory[13]) & (side_stable >= 3) & (memory[5] >= 2) & (memory[5] <= 12)
    mapping = jnp.where(response, jnp.bitwise_xor(memory[13], memory[14]), memory[11])
    locked = (memory[12] > 0) | response
    cue_stable = jnp.where(other_raw & (other_type == memory[6]) & (other_x == 5), memory[7]+1, 0)
    stalled = jnp.where(n == memory[3], memory[4]+1, 0)
    carrier = (memory[8] > 0) | ((n == 2) & other_raw & (cue_stable >= 6) & (stalled >= 6))
    # A persistent literal ingredient can be a signal rather than a promised
    # deposit. Allow it again after a delivery guard instead of waiting forever.
    persistent_cue = other_raw & (cue_stable >= 4) & (memory[5] >= 8)
    fresh = (memory[2] > 0) | (visible_partner & (jnp.sum(other) == 0)) | persistent_cue
    ingredient = jnp.where((memory[1] < 0) & fresh & other_raw, jnp.argmax(other[2:]), memory[1])
    ingredient = jnp.where(n > 0, jnp.argmax(pot[2:]), ingredient)
    invalid_pot = (jnp.sum(pot[4:]) > 0) | ((pot[2] > 0) & (pot[3] > 0))
    ingredient = jnp.where((n > 0) & invalid_pot, jnp.where(other_raw, other_type, 0), ingredient)
    # Literal ingredient evidence supersedes a speculative fallback, including
    # when it arrives after a guessed pot has begun cooking.
    truth = jnp.where((memory[17] > 0) & other_raw & (memory[19] < 0), other_type, memory[19])
    ingredient = jnp.where(truth >= 0, truth, ingredient)
    grace = jnp.where((memory[15] > 0) & (side_stable >= 4), 6, 20)
    grace = jnp.where(locked & (side_stable >= 3), 3, grace)
    fallback = (ingredient < 0) & (n == 0) & (jnp.sum(inv) == 0) & (stalled >= grace) & (memory[5] >= grace) & ~other_raw
    alternate = jnp.where(memory[14] >= 0, 1-memory[14], 0)
    unconfirmed = jnp.where((side_cue == memory[13]) & (memory[14] >= 0), alternate, side_cue)
    guess = jnp.where(side_cue >= 0, jnp.where(locked, jnp.bitwise_xor(side_cue, mapping), unconfirmed), alternate)
    ingredient = jnp.where(fallback, guess, ingredient)
    cycle_cue = jnp.where(fallback, side_cue, memory[16])
    fallback_cycle = (memory[17] > 0) | fallback
    ingredient_x = jnp.where(ingredient == 0, 4, 6)
    ingredient_dir = jnp.where(ingredient == 0, 3, 2)
    fetch = upper_to(x, direction, ingredient_x, ingredient_dir)
    put = upper_to(x, direction, 5, 1)
    plate_x = jnp.where(x <= 5, 4, 6)
    plate = upper_to(x, direction, plate_x, 0)
    deliver = upper_to(x, direction, 5, 0)
    counters = obs[3, jnp.asarray([6, 8]) - x, 32:38]
    free = jnp.sum(counters, axis=1) == 0
    reusable = (counters[:, 0] == 0) & (counters[:, 1] == 0) & (jnp.sum(counters[:, 2:], axis=1) == 1)
    reusable = reusable & (counters[:, 2 + jnp.maximum(ingredient, 0)] > 0) & (ingredient >= 0)
    distance = jnp.abs(jnp.asarray([4, 6]) - x)
    counter_cost = distance + jnp.where(distance > 0, 1, (direction != 1).astype(jnp.int32)) + 1
    supply_distance = jnp.abs(x-ingredient_x)
    supply_cost = supply_distance + ((supply_distance == 0) & (direction != ingredient_dir)).astype(jnp.int32) + 1
    reuse_index = jnp.argmin(jnp.where(reusable, counter_cost, 100))
    reuse_x = jnp.asarray([4, 6])[reuse_index]
    arrival_steps = supply_cost + jnp.abs(ingredient_x-5) + 2
    unseen_partner_arrival = jnp.where(x == 5, 5, 4)
    overflow_risk = other_raw | (arrival_steps > unseen_partner_arrival)
    reuse_is_faster = jnp.any(reusable) & ((counter_cost[reuse_index] <= supply_cost) |
                       carrier | invalid_pot | fallback_cycle | ((n == 2) & ~jnp.any(free) & overflow_risk))
    fetch = jnp.where(reuse_is_faster, upper_to(x, direction, reuse_x, 1), fetch)
    drop_x = jnp.asarray([4, 6])[jnp.argmin(jnp.where(free, distance, 100))]
    drop = upper_to(x, direction, drop_x, 1)
    wait = jnp.where(x < 5, 0, jnp.where(x > 5, 2, 4))
    action = jnp.where(ingredient >= 0, fetch, wait)
    # The partner can finish the pot without causing a surplus pickup.
    action = jnp.where((n == 2) & other_raw & (stalled < 6), wait, action)
    action = jnp.where((n >= 3) | (timer > 0) | (pot[1] > 0), plate, action)
    raw_counters = (jnp.sum(counters[:, :2], axis=1) == 0) & (jnp.sum(counters[:, 2:], axis=1) == 1)
    stocked = jnp.any((counters[:, 2:4] > 0) & raw_counters[:, None], axis=0)
    prep_type = jnp.argmin(stocked.astype(jnp.int32))
    prep = upper_to(x, direction, jnp.where(prep_type == 0, 4, 6), jnp.where(prep_type == 0, 3, 2))
    prep_now = (timer >= 12) & jnp.any(free) & ~jnp.all(stocked) & ~carrier & ~invalid_pot & ~fallback_cycle
    action = jnp.where(prep_now, prep, action)
    wrong_raw = (truth >= 0) & (jnp.argmax(inv[2:]) != truth)
    action = jnp.where(held_raw, jnp.where((n < 3) & (timer == 0) & (pot[1] == 0) & ~wrong_raw, put, drop), action)
    plate_wait = jnp.where((x == 5) & (direction == 1), 4, put)
    action = jnp.where(inv[0] > 0, jnp.where(pot[1] > 0, put, plate_wait), action)
    valid_dish = (jnp.sum(inv[2:]) == 3) & ((inv[2] == 3) | (inv[3] == 3))
    bad_dish = (inv[1] > 0) & (~valid_dish | ((truth >= 0) & (inv[2+jnp.maximum(truth, 0)] != 3)))
    action = jnp.where(inv[1] > 0, jnp.where(bad_dish, jnp.where(jnp.any(free), drop, wait), deliver), action)
    delivered = (inv[1] > 0) & valid_dish & ~bad_dish & (x == 5) & (direction == 0) & (action == 5)
    discarded = bad_dish & jnp.any(free) & (x == drop_x) & (direction == 1) & (action == 5)
    pending_cue = jnp.where(response, -1, memory[13])
    pending_cue = jnp.where(delivered, jnp.where(fallback_cycle, cycle_cue, -1), pending_cue)
    last_guess = jnp.where(delivered, jnp.argmax(inv[2:]), memory[14])
    same_cue_count = jnp.where(delivered & fallback_cycle,
                       jnp.where((cycle_cue >= 0) & (cycle_cue == memory[13]), memory[18]+1, 0), memory[18])
    locked = locked & (same_cue_count < 4)
    same_cue_count = jnp.where(same_cue_count >= 4, 0, same_cue_count)
    cleared = delivered | discarded
    ingredient = jnp.where(cleared, -1, ingredient)
    fresh = fresh & ~cleared
    age = jnp.where(cleared, 0, memory[5]+1)
    new_memory = jnp.asarray([memory[0]+1, ingredient, fresh.astype(jnp.int32), n, stalled, age,
                              jnp.where(other_raw, other_type, -1), cue_stable, carrier.astype(jnp.int32),
                              side_cue, side_stable, mapping, locked.astype(jnp.int32), pending_cue, last_guess,
                              ((memory[15] > 0) | fallback).astype(jnp.int32), jnp.where(cleared, -1, cycle_cue),
                              (fallback_cycle & ~cleared).astype(jnp.int32), same_cue_count, jnp.where(cleared, -1, truth)], jnp.int32)
    return action.astype(jnp.int32), new_memory, {'pot_n': n, 'x': x, 'raw_type': ingredient, 'fresh': fresh, 'stall': stalled, 'carrier': carrier, 'fallback': fallback_cycle, 'cue': side_cue, 'mapping': mapping, 'locked': locked}


def init_agent_1():
    # time, recipe, mode, previous pot count, unchanged count steps,
    # four observed counter item codes, task ingredient, task counter
    return jnp.asarray([0, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0], jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_1(), memory)
    inv, direction, pot, timer, other = decode(obs)
    row, x = localize_lower(obs)
    recipe = jnp.sum(obs[:, :, 40:44], axis=(0, 1))
    target = jnp.where(jnp.sum(recipe) > 0, jnp.argmax(recipe), memory[1])
    pot_visible = jnp.sum(obs[:, :, 24]) > 0
    n = jnp.sum(pot[2:])
    stable = jnp.where(pot_visible & (n == memory[3]), memory[4] + 1, 0)
    held_raw = (jnp.sum(inv[2:]) > 0) & (inv[0] == 0)
    empty = jnp.sum(inv) == 0
    raw_type = jnp.argmax(inv[2:])
    partner_dish = other[1] > 0
    pantry_x = jnp.asarray([3, 7, 4, 6], jnp.int32)
    pantry_row = jnp.asarray([4, 4, 2, 2], jnp.int32)
    pantry_dir = jnp.asarray([1, 1, 0, 0], jnp.int32)
    pantry_visible = (jnp.abs(pantry_x - x) <= 2) & (jnp.abs(pantry_row - row) <= 2)
    pantry_items = obs[jnp.clip(pantry_row-row+2, 0, 4), jnp.clip(pantry_x-x+2, 0, 4), 32:38]
    pantry_raw = (jnp.sum(pantry_items[:, 2:4], axis=1) == 1) & (jnp.sum(pantry_items[:, :2], axis=1) == 0)
    pantry_codes = jnp.where(jnp.sum(pantry_items, axis=1) == 0, 0,
                            jnp.where(pantry_raw, jnp.argmax(pantry_items[:, 2:], axis=1)+1, 9))
    stocks = jnp.where(pantry_visible, pantry_codes, memory[5:9])
    available = stocks == target+1
    free = (stocks == 0) & (jnp.arange(4) < 2)
    distance = jnp.abs(pantry_x-x) + jnp.abs(pantry_x-5)
    matching_side = jnp.argmin(jnp.where(available, distance, 100))
    preferred_side = jnp.clip(target, 0, 1)
    free_side = jnp.where(free[preferred_side], preferred_side, jnp.argmin(jnp.where(free, distance, 100)))

    mode = memory[2]
    mode = jnp.where(((mode == 1) | (mode == 3)) & held_raw, 0, mode)
    mode = jnp.where((mode == 2) & held_raw, 4, mode)
    mode = jnp.where((mode == 4) & empty, 0, mode)
    idle = (mode == 0) & empty
    ready_to_help = (n == 0) | ((n < 3) & ((stable >= 8) | ((other[0] > 0) & (other[1] == 0))))
    start_load = idle & (target >= 0) & pot_visible & ready_to_help & ~partner_dish
    # Starting early in cooking guarantees stowing before the next delivery.
    start_stock = idle & (target >= 0) & (timer >= 16) & jnp.any(free) & ~jnp.any(available)
    mode = jnp.where(start_load, jnp.where(jnp.any(available), 3, 1), mode)
    mode = jnp.where(start_stock, 2, mode)
    task_type = jnp.where(start_load | start_stock, target, memory[9])
    task_side = jnp.where(start_load, jnp.where(jnp.any(available), matching_side, jnp.clip(target, 0, 1)), memory[10])
    task_side = jnp.where(start_stock, free_side, task_side)
    supply_x = jnp.where(task_side == 0, 1, 9)
    supply_dir = jnp.where(task_type == 0, jnp.where(task_side == 0, 3, 2), 1)
    fetch = lower_to(row, x, direction, 4, supply_x, supply_dir)
    pantry_act = lower_to(row, x, direction, 3, pantry_x[task_side], pantry_dir[task_side])
    home = lower_to(row, x, direction, 3, 5, 0)
    wait_home = jnp.where((row == 3) & (x == 5), 4, home)
    can_put = (n < 3) & (timer == 0) & (pot[1] == 0) & ~partner_dish
    put = jnp.where(pot_visible & ~can_put, wait_home, home)
    # Store an obsolete ingredient away from the pot, preserving it for reuse.
    drop_x = jnp.where(jnp.any(free), pantry_x[free_side], jnp.where(x <= 5, 2, 8))
    drop_dir = jnp.where(jnp.any(free), 1, 0)
    drop = lower_to(row, x, direction, 3, drop_x, drop_dir)
    action = wait_home
    action = jnp.where((mode == 1) | (mode == 2), fetch, action)
    action = jnp.where(mode == 3, pantry_act, action)
    action = jnp.where(held_raw, jnp.where(raw_type == target, put, drop), action)
    action = jnp.where(mode == 4, pantry_act, action)
    new_memory = jnp.asarray([memory[0]+1, target, mode, n, stable, stocks[0], stocks[1], stocks[2], stocks[3], task_type, task_side], jnp.int32)
    return action.astype(jnp.int32), new_memory, {'position': jnp.asarray([row,x]), 'recipe': target, 'pot_n': n, 'mode': mode, 'stocks': stocks}
