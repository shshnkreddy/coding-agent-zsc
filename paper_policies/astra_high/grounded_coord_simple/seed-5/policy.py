"""Grounded recipe communication and local, observation-driven cooking."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def locate(obs, channel, global_row, global_col):
    index = jnp.argmax(obs[:, :, channel].reshape(-1))
    return global_row + 2 - index // 5, global_col + 2 - index % 5


def approach(row, col, facing, target_row, target_col, direction):
    move = jnp.where(row < target_row, 1, jnp.where(row > target_row, 3,
           jnp.where(col < target_col, 0, jnp.where(col > target_col, 2, 4))))
    turn = jnp.asarray([3, 1, 0, 2], jnp.int32)[direction]
    return jnp.where((row == target_row) & (col == target_col),
                     jnp.where(facing == direction, 5, turn), move).astype(jnp.int32)


def init_agent_0():
    return jnp.asarray([0, 0, 100, 0, 0, -1, 0, 6, 0], jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_0(), memory)
    time, old_count, since_signal, empty_age, retries, old_recipe, last_fill, fill_gap, warned = memory
    row, col = locate(obs, 23, 2, 0)
    facing = jnp.argmax(obs[2, 2, 1:5])
    pot = jnp.sum(obs[:, :, 29:34] * obs[:, :, 22:23], axis=(0, 1))
    count = jnp.sum(pot[2:])
    cleared = (old_count > 0) & (count == 0) & (col == 2)
    inserted = (count > old_count) & (col == 2)
    fill_gap = jnp.where(inserted & (old_count > 0), jnp.clip(time - last_fill, 6, 24), fill_gap)
    last_fill = jnp.where(inserted, time, last_fill)
    warned = jnp.where(cleared, 0, warned)
    empty_age = jnp.where(cleared, 0, empty_age + 1)
    since_signal = jnp.where(cleared, 100, since_signal + 1)
    retries = jnp.where(cleared, 0, retries)
    recipe = jnp.argmax(jnp.sum(obs[:, :, 36:39] * obs[:, :, 23:24], axis=(0, 1)))
    changed = (old_recipe >= 0) & (recipe != old_recipe)
    display_active = jnp.sum(obs[:, :, 36:39] * obs[:, :, 24:25]) > 0
    retry_base = jnp.maximum(20, (3 * fill_gap) // 2)
    retry_interval = retry_base * jnp.left_shift(jnp.asarray(1, jnp.int32), jnp.minimum(jnp.maximum(retries - 1, 0), 3))
    signal_delay = jnp.maximum(3, (2 * fill_gap) // 3 - 5)
    next_cycle_steps = 20 + (11 * fill_gap) // 3
    cycle_signal = (count == 0) & (((empty_age >= signal_delay) & (since_signal >= retry_interval)) | changed)
    correction = (count > 0) & (jnp.argmax(pot[2:]) != recipe) & (warned == 0) & (since_signal >= 10)
    ready = (cycle_signal | correction) & ~display_active & (time + next_cycle_steps < 400)
    action = approach(row, col, facing, 2, 2, 2)
    at_button = (row == 2) & (col == 2) & (facing == 2)
    action = jnp.where(at_button & ~ready, 4, action).astype(jnp.int32)
    activated = at_button & ready
    since_signal = jnp.where(activated, 0, since_signal)
    warned = jnp.where(activated & correction, 1, warned)
    retries = retries + activated.astype(jnp.int32)
    memory = jnp.stack([time + 1, jnp.where(col == 2, count, old_count), since_signal, empty_age, retries, recipe, last_fill, fill_gap, warned]).astype(jnp.int32)
    return action, memory, {'pot_count': count, 'activated': activated, 'empty_age': empty_age, 'fill_gap': fill_gap}


def counter_options(obs, row, col, facing, recipe, source):
    # All four usable counters are directly observable from the middle row.
    counter_rows = jnp.asarray([0, 0, 4, 4], jnp.int32)
    counter_cols = jnp.asarray([5, 6, 5, 6], jnp.int32)
    local_rows, local_cols = counter_rows - row + 2, counter_cols - col + 2
    seen = (local_rows >= 0) & (local_rows < 5) & (local_cols >= 0) & (local_cols < 5)
    items = obs[jnp.clip(local_rows, 0, 4), jnp.clip(local_cols, 0, 4), 29:34]
    standing_rows = jnp.where(counter_rows == 0, 1, 3)
    directions = jnp.where(counter_rows == 0, 0, 1)
    distance = jnp.abs(row - standing_rows) + jnp.abs(col - counter_cols)
    free = seen & (jnp.sum(items, axis=1) == 0)
    soup = seen & (items[:, 1] > 0) & (items[:, 2 + recipe] == 3)
    raw = seen & (items[:, 0] == 0) & (items[:, 1] == 0) & (items[:, 2 + source] == 1)
    def select(mask):
        index = jnp.argmin(jnp.where(mask, distance, 100))
        action = approach(row, col, facing, standing_rows[index], counter_cols[index], directions[index])
        return jnp.any(mask), action
    return select(free), select(soup), select(raw)


def init_agent_1():
    # time, recipe, confidence (0 unknown / 1 displayed / 2 inferred),
    # previous goal interaction, wait count, ever informed, silent cycles
    return jnp.asarray([0, 0, 0, 0, 0, 0, 0], jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_1(), memory)
    time, recipe, valid, old_goal, waiting, informed, silent_cycles = memory
    row, col = locate(obs, 22, 2, 4)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    dish, plate = inv[1] > 0, inv[0] > 0
    raw = (jnp.sum(inv[2:]) > 0) & ~dish
    empty = jnp.sum(inv) == 0
    delivered = (old_goal > 0) & ~dish
    valid = jnp.where(delivered, 0, valid)
    waiting = jnp.where(delivered, 0, waiting)

    displayed = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    visible = jnp.sum(displayed) > 0
    recipe = jnp.where(visible, jnp.argmax(displayed), recipe)
    valid = jnp.where(visible, 1, valid)
    informed = jnp.maximum(informed, visible.astype(jnp.int32))
    silent_cycles = jnp.where(visible, 0, silent_cycles)
    waiting = jnp.where((valid == 0) & empty, waiting + 1, waiting)
    wait_limit = jnp.where(time < 40, 20, jnp.where((informed == 0) | (silent_cycles >= 3), 4, 10))
    fallback = (valid == 0) & (waiting >= wait_limit)
    silent_cycles = silent_cycles + fallback.astype(jnp.int32)
    recipe = jnp.where(fallback & ((informed == 0) | (silent_cycles >= 3)), 1 - recipe, recipe)
    valid = jnp.where(fallback, 2, valid)

    pot = jnp.sum(obs[:, :, 29:34] * obs[:, :, 22:23], axis=(0, 1))
    count = jnp.sum(pot[2:])
    cooked = pot[1] > 0
    # A partly filled pot cannot be undone: finish it homogeneously, then
    # store it if a later grounded display says the dish would be wrong.
    source = jnp.where(count > 0, jnp.argmax(pot[2:]), recipe)
    source_row = jnp.where(source == 0, 1, 3)
    get_raw = approach(row, col, facing, source_row, 6, 2)
    pot_action = approach(row, col, facing, 2, 5, 3)
    get_plate = approach(row, col, facing, jnp.where(row == 3, 3, 1), 5, 3)
    deliver = approach(row, col, facing, 2, 6, 2)
    wait_display = jnp.where(pot_action == 5, 4, pot_action)

    (free_counter, store_action), (stored_soup, soup_action), (stored_raw, raw_action) = counter_options(obs, row, col, facing, recipe, source)
    get_raw = jnp.where(stored_raw, raw_action, get_raw)
    acquire = jnp.where((count == 0) & stored_soup, soup_action, get_raw)
    mismatched = jnp.argmax(inv[2:]) != recipe
    park_dish = dish & mismatched & (valid == 1) & free_counter
    park_raw = raw & mismatched & (valid == 1) & (count == 0) & free_counter

    action = jnp.where(dish, jnp.where(park_dish, store_action, deliver),
             jnp.where(plate, jnp.where(cooked, pot_action, jnp.where(pot_action == 5, 4, pot_action)),
             jnp.where(raw, jnp.where(park_raw, store_action, pot_action),
             jnp.where(count >= 3, get_plate,
             jnp.where((valid > 0) | (count > 0), acquire, wait_display)))))
    goal_attempt = dish & (row == 2) & (col == 6) & (facing == 2) & (action == 5)
    memory = jnp.stack([time + 1, recipe, valid, goal_attempt.astype(jnp.int32), waiting, informed, silent_cycles]).astype(jnp.int32)
    trace = {'recipe': recipe, 'valid': valid, 'pot_count': count,
             'waiting': waiting, 'delivered': delivered, 'row': row, 'col': col,
             'silent_cycles': silent_cycles, 'park_dish': park_dish,
             'park_raw': park_raw, 'stored_soup': stored_soup}
    return action.astype(jnp.int32), memory, trace
