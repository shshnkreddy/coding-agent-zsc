"""Grounded recipe service and locally reactive cooking."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'reusable_dishes_v10', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def locate(obs, channel, row, col):
    index = jnp.argmax(obs[:, :, channel].reshape(-1))
    return jnp.asarray(row + 2, jnp.int32) - index // 5, jnp.asarray(col + 2, jnp.int32) - index % 5


def approach(row, col, facing, target_row, target_col, target_facing):
    """Reach a floor interaction site and face its object."""
    turn = jnp.asarray([3, 1, 0, 2], jnp.int32)[target_facing]
    action = jnp.where(facing == target_facing, 5, turn)
    action = jnp.where(col != target_col, jnp.where(col < target_col, 0, 2), action)
    action = jnp.where(row != target_row, jnp.where(row < target_row, 1, 3), action)
    return action.astype(jnp.int32)


def init_agent_0():
    # Pot count, signal age, stalled age, unanswered signals, time,
    # previous recipe, post-collection signal delay, insertion interval,
    # previous collection time, shortest observed full-cycle interval.
    return jnp.asarray([0, 100, 0, 0, 0, -1, 0, 6, -1, -1], jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_0(), memory)
    row, col = locate(obs, 24, 2, 3)
    facing = jnp.argmax(obs[2, 2, 1:5])
    visible = jnp.any(obs[:, :, 22] > 0)
    counts = jnp.sum(obs[:, :, 31:34] * obs[:, :, 22:23])
    display = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    recipe = jnp.argmax(display).astype(jnp.int32)
    progress = visible & (counts != memory[0])
    emptied = progress & (counts == 0)
    collection_gap = memory[4] - memory[8]
    cycle_time = jnp.where(emptied & (memory[8] >= 0), jnp.where(memory[9] > 0, jnp.minimum(memory[9], collection_gap), collection_gap), memory[9])
    collection_time = jnp.where(emptied, memory[4], memory[8])
    stalled_age = jnp.where(progress, 0, memory[2] + 1)
    insertion_interval = jnp.where(progress & (counts > memory[0]), jnp.clip(memory[2] + 1, 6, 90), memory[7])
    stall_threshold = jnp.maximum(12, 2 * insertion_interval)
    attempts = jnp.where(progress, 0, memory[3])
    delay = jnp.where(emptied, 2, jnp.maximum(memory[6] - 1, 0))
    retry_interval = 16 * jnp.left_shift(jnp.asarray(1, jnp.int32), jnp.clip(attempts - 1, 0, 3))
    retry_due = memory[1] >= retry_interval
    recipe_changed = (recipe != memory[5]) & (memory[5] >= 0) & (memory[1] >= 9)
    need_signal = ((counts == 0) & (delay == 0) & retry_due) | ((counts > 0) & (stalled_age >= stall_threshold) & retry_due) | recipe_changed
    remaining_cycle = jnp.where(cycle_time > 0, cycle_time - 1, 39)
    enough_time = jnp.where(counts == 0, memory[4] + remaining_cycle < 400, memory[4] < 364 + 6 * counts)
    ready_to_signal = visible & (counts < 3) & need_signal & enough_time
    at_button = (row == 2) & (col == 2) & (facing == 2)
    movement = approach(row, col, facing, 2, 2, 2)
    action = jnp.where(at_button, jnp.where(ready_to_signal, 5, 4), movement).astype(jnp.int32)
    activated = at_button & (action == 5)
    next_memory = jnp.stack([jnp.where(visible, counts, memory[0]), jnp.where(activated, 0, memory[1] + 1), stalled_age, attempts + activated.astype(jnp.int32), memory[4] + 1, recipe, delay, insertion_interval, collection_time, cycle_time])
    return action, next_memory, {'pot_count': counts, 'signal': activated, 'stalled_age': stalled_age, 'attempts': attempts, 'insertion_interval': insertion_interval, 'cycle_time': cycle_time, 'time': memory[4]}


def init_agent_1():
    # Known recipe, time, wait age, next blind guess, fallback mode, ever seen;
    # four remembered counter item codes (zero denotes an empty counter).
    return jnp.asarray([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], jnp.int32)


def route_cost(row, col, facing, target_row, target_col, target_facing):
    dr, dc = target_row - row, target_col - col
    end_facing = jnp.where(dc != 0, jnp.where(dc > 0, 2, 3), jnp.where(dr != 0, jnp.where(dr > 0, 1, 0), facing))
    return jnp.abs(dr) + jnp.abs(dc) + (end_facing != target_facing).astype(jnp.int32) + 1


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_1(), memory)
    row, col = locate(obs, 22, 2, 4)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    display = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    displayed = jnp.sum(display) > 0
    recipe = jnp.where(displayed, jnp.argmax(display), memory[0])
    known = recipe >= 0
    ever_seen = (memory[5] > 0) | displayed
    wait_age = jnp.where(known, 0, memory[2] + 1)
    timeout = jnp.where(ever_seen, 56, 96)
    fallback = (~known) & ((memory[4] > 0) | (wait_age >= timeout))
    pot = jnp.sum(obs[:, :, 29:34] * obs[:, :, 22:23], axis=(0, 1))
    timer = jnp.sum(obs[:, :, 39] * obs[:, :, 22])
    pot_count = jnp.sum(pot[2:])
    initial_guess = jnp.where(memory[1] == 0, (row == 3).astype(jnp.int32), memory[3])
    goal = jnp.where(pot_count > 0, jnp.argmax(pot[2:]), jnp.where(known, recipe, initial_guess))
    pot_action = approach(row, col, facing, 2, 5, 3)
    supply_row = jnp.where(goal == 1, 3, 1)
    supply_action = approach(row, col, facing, supply_row, 6, 2)
    supply_cost = route_cost(row, col, facing, supply_row, 6, 2)

    # Counter contents are visible physical ingredients, not communication codes.
    counter_rows = jnp.asarray([0, 0, 4, 4], jnp.int32)
    counter_cols = jnp.asarray([5, 6, 5, 6], jnp.int32)
    floor_rows = jnp.asarray([1, 1, 3, 3], jnp.int32)
    counter_dirs = jnp.asarray([0, 0, 1, 1], jnp.int32)
    lr, lc = counter_rows - row + 2, counter_cols - col + 2
    counter_visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    items = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), 29:34]
    item_codes = jnp.where(jnp.sum(items, axis=1) == 0, 0, jnp.where(items[:, 1] > 0, 4 + jnp.argmax(items[:, 2:], axis=1), jnp.where(items[:, 0] > 0, 7, 1 + jnp.argmax(items[:, 2:], axis=1))))
    stock = jnp.where(counter_visible, item_codes, memory[6:10]).astype(jnp.int32)
    costs = route_cost(row, col, facing, floor_rows, counter_cols, counter_dirs)
    matching_raw = counter_visible & (items[:, 0] == 0) & (items[:, 1] == 0) & (jnp.sum(items[:, 2:], axis=1) == 1) & (items[:, goal + 2] == 1)
    take_index = jnp.argmin(jnp.where(matching_raw, costs, 1000))
    take_counter = jnp.any(matching_raw) & (costs[take_index] <= supply_cost + 1)
    take_action = approach(row, col, facing, floor_rows[take_index], counter_cols[take_index], counter_dirs[take_index])
    source_action = jnp.where(take_counter, take_action, supply_action)
    free_counter = counter_visible & (jnp.sum(items, axis=1) == 0)
    drop_index = jnp.argmin(jnp.where(free_counter, costs, 1000))
    drop_action = approach(row, col, facing, floor_rows[drop_index], counter_cols[drop_index], counter_dirs[drop_index])

    # Preparing a raw ingredient costs its source route plus three storage
    # steps; four further steps acquire a plate and return facing the pot.
    stage_goal = jnp.where(stock[0] == 0, 0, 1)
    stage_row = jnp.where(stage_goal == 1, 3, 1)
    stage_cost = route_cost(row, col, facing, stage_row, 6, 2) + 7
    empty_hands = jnp.sum(inv) == 0
    can_stage = (timer > 0) & empty_hands & ((stock[0] == 0) | (stock[2] == 0)) & (timer >= stage_cost)
    stage_action = approach(row, col, facing, stage_row, 6, 2)
    store_index = jnp.where(jnp.argmax(inv[2:]) == 1, 2, 0)
    store_index = jnp.where(stock[store_index] == 0, store_index, drop_index)
    store_action = approach(row, col, facing, floor_rows[store_index], counter_cols[store_index], counter_dirs[store_index])
    plate_action = approach(row, col, facing, jnp.where(row == 3, 3, 1), 5, 3)
    delivery_action = approach(row, col, facing, 2, 6, 2)
    view_action = jnp.where(col == 5, 4, approach(row, col, facing, row, 5, 3))
    matching_dishes = (stock == recipe + 4) & known
    dish_index = jnp.argmin(jnp.where(matching_dishes, costs, 1000))
    dish_action = approach(row, col, facing, floor_rows[dish_index], counter_cols[dish_index], counter_dirs[dish_index])
    take_dish = known & empty_hands & (pot_count == 0) & jnp.any(matching_dishes)
    # Preserve the two nearest counters for raw stock when possible.
    free_storage = stock == 0
    save_costs = costs + jnp.where(counter_cols == 5, 4, 0)
    save_index = jnp.argmin(jnp.where(free_storage, save_costs, 1000))
    save_dish_action = approach(row, col, facing, floor_rows[save_index], counter_cols[save_index], counter_dirs[save_index])
    preparing_unknown = (~ever_seen) & (~known) & (~fallback) & (pot_count == 0) & empty_hands
    action = jnp.where(known | fallback | (pot_count > 0) | preparing_unknown, source_action, view_action)
    action = jnp.where((timer > 0) | (pot[1] > 0) | (pot_count >= 3), jnp.where(can_stage, stage_action, plate_action), action)
    action = jnp.where(take_dish, dish_action, action)
    holding_raw = (jnp.sum(inv[2:]) > 0) & (inv[0] == 0)
    mismatched_raw = holding_raw & known & (pot_count == 0) & (jnp.argmax(inv[2:]) != recipe)
    raw_action = jnp.where(mismatched_raw & jnp.any(free_counter), store_action, pot_action)
    # Speculative collection is harmless only while insertion awaits evidence.
    awaiting_initial_recipe = (~known) & (~fallback) & (pot_count == 0)
    raw_wait = jnp.where((row == 2) & (col == 5), 4, pot_action)
    raw_action = jnp.where(awaiting_initial_recipe, raw_wait, raw_action)
    raw_action = jnp.where((timer > 0) | (pot[1] > 0), store_action, raw_action)
    action = jnp.where(holding_raw, raw_action, action)
    action = jnp.where(inv[0] > 0, jnp.where((pot_action == 5) & (pot[1] == 0), 4, pot_action), action)
    matches_known = known & jnp.all(inv[2:] == 3 * (jnp.arange(3) == recipe))
    wrong_dish = (inv[1] > 0) & known & (~matches_known)
    cooked_action = jnp.where(wrong_dish & jnp.any(free_storage), save_dish_action, delivery_action)
    cooked_action = jnp.where(wrong_dish & (~jnp.any(free_storage)) & (memory[1] >= 360), 4, cooked_action)
    action = jnp.where(inv[1] > 0, cooked_action, action).astype(jnp.int32)
    delivering = (inv[1] > 0) & (action == 5) & (row == 2) & (col == 6) & (facing == 2)
    # A known wrong dish leaves the target unchanged, so its recipe stays valid.
    next_recipe = jnp.where(delivering & matches_known, -1, recipe)
    next_wait = jnp.where(delivering, 0, wait_age)
    next_guess = jnp.where(delivering, 1 - jnp.argmax(inv[2:]), initial_guess)
    next_memory = jnp.concatenate([jnp.stack([next_recipe, memory[1] + 1, next_wait, next_guess, fallback.astype(jnp.int32), ever_seen.astype(jnp.int32)]), stock]).astype(jnp.int32)
    return action, next_memory, {'recipe': recipe, 'goal': goal, 'fallback': fallback, 'pot_count': pot_count, 'pot_timer': timer, 'row': row, 'col': col, 'delivering': delivering, 'staging': can_stage, 'preparing_unknown': preparing_unknown, 'save_dish': wrong_dish, 'take_dish': take_dish, 'stock': stock, 'time': memory[1]}
