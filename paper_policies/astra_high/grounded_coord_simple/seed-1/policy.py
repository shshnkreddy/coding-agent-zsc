"""Grounded display coordination and independent kitchen execution."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_display_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def init_agent_0():
    # Display cooldown, step, attempts, last pot count, delivery delay, progress seen.
    return jnp.zeros((6,), dtype=jnp.int32)


def init_agent_1():
    # Recipe ID, recipe valid, elapsed steps, ever observed, wait age, unconfirmed deliveries, last display, changes observed, cache task, last delivery, cycle observed, late signals.
    return jnp.zeros((12,), dtype=jnp.int32)


def locate(obs, channel, global_row, global_col):
    index = jnp.argmax(obs[:, :, channel])
    return global_row + 2 - index // 5, global_col + 2 - index % 5


def service(row, col, facing, target_row, target_col, direction):
    """Move to a floor service cell, orient toward its object, interact."""
    turn = jnp.asarray([3, 1, 0, 2], jnp.int32)[direction]
    at = (row == target_row) & (col == target_col)
    move = jnp.where(row != target_row, jnp.where(row < target_row, 1, 3), jnp.where(col < target_col, 0, 2))
    return jnp.where(at, jnp.where(facing == direction, 5, turn), move).astype(jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_0(), memory)
    row, col = locate(obs, 24, 2, 3)
    facing = jnp.argmax(obs[2, 2, 1:5])
    count = jnp.sum(obs[:, :, 31:34] * obs[:, :, 22, None])
    empty = count == 0
    ready = (row == 2) & (col == 2) & (facing == 2)
    navigate = service(row, col, facing, 2, 2, 2)
    attempts = jnp.where(empty, memory[2], 0)
    cleared = empty & (memory[3] > 0)
    delay = jnp.where(cleared, 2, jnp.maximum(0, memory[4] - 1))
    progress = (memory[5] > 0) | (count > 0)
    activate = ready & empty & (delay == 0) & (memory[0] == 0) & (memory[1] < 362)
    action = jnp.where(ready, jnp.where(activate, 5, 4), navigate)
    interval = jnp.where(progress, 30 * (2 ** jnp.minimum(attempts, 2)), 10 * (2 ** jnp.minimum(attempts, 3)))
    cooldown = jnp.where(activate, interval - 1, jnp.where(empty, jnp.maximum(0, memory[0] - 1), 0))
    next_memory = jnp.array([cooldown, memory[1] + 1, attempts + activate.astype(jnp.int32), count, delay, progress.astype(jnp.int32)], jnp.int32)
    return action.astype(jnp.int32), next_memory, {'pot_count':count,'activate':activate,'position':jnp.array([row,col])}


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_1(), memory)
    row, col = locate(obs, 22, 2, 4)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inventory = obs[2, 2, 5:10]
    recipe = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    visible = jnp.sum(recipe) > 0
    seen = visible | (memory[3] > 0)
    late_signal = visible & (memory[10] == 0) & (memory[5] > 0) & (memory[2] - memory[9] > 8)
    late_signals = late_signal | (memory[11] > 0)
    changed = visible & (memory[3] > 0) & (jnp.argmax(recipe) != memory[6])
    change_signaler = changed | (memory[7] > 0)
    silent = (~seen & (memory[5] > 0)) | ((memory[5] >= 2) & ~change_signaler)
    wait_limit = jnp.where(silent, 3, jnp.where(~seen, 18, jnp.where(change_signaler & ~late_signals, 6, 12)))
    timeout = memory[4] >= wait_limit
    fallback = ~visible & (memory[1] == 0) & timeout
    # Without reliable signals, alternating beats repeating a permanently stale target:
    # a wrong delivery preserves the recipe, so the other ingredient is then correct.
    switch_guess = fallback & (((memory[5] >= 2) & ~change_signaler) | ~seen)
    ingredient = jnp.where(visible, jnp.argmax(recipe), jnp.where(switch_guess, 1 - memory[0], memory[0])).astype(jnp.int32)
    valid = visible | (memory[1] > 0) | timeout
    pot = jnp.sum(obs[:, :, 29:34] * obs[:, :, 22, None], axis=(0, 1))
    pot_count = jnp.sum(pot[2:])
    pot_timer = jnp.max(obs[:, :, 39])
    cooked = pot[1] > 0
    has_plate = inventory[0] > 0
    has_dish = inventory[1] > 0
    has_raw = (jnp.sum(inventory[2:]) > 0) & ~has_dish
    to_pot = service(row, col, facing, 2, 5, 3)
    needed = jnp.where((pot_count > 0) & (pot_count < 3), jnp.argmax(pot[2:]), ingredient)
    to_supply = service(row, col, facing, jnp.where(needed == 0, 1, 3), 6, 2)
    to_plate = service(row, col, facing, jnp.where(row <= 2, 1, 3), 5, 3)
    to_goal = service(row, col, facing, 2, 6, 2)
    # Use actual empty wall counters to put aside items contradicted by a display.
    counter_rows = jnp.array([0, 4, 0, 4], jnp.int32)
    counter_cols = jnp.array([5, 5, 6, 6], jnp.int32)
    local_rows, local_cols = counter_rows - row + 2, counter_cols - col + 2
    in_view = (local_rows >= 0) & (local_rows < 5) & (local_cols >= 0) & (local_cols < 5)
    counter_items = obs[jnp.clip(local_rows, 0, 4), jnp.clip(local_cols, 0, 4), 29:34]
    empty_counter = in_view & (jnp.sum(counter_items, axis=-1) == 0)
    service_rows = jnp.where(counter_rows == 0, 1, 3)
    distances = jnp.abs(service_rows - row) + jnp.abs(counter_cols - col)
    counter_index = jnp.argmin(jnp.where(empty_counter, distances, 100))
    to_counter = service(row, col, facing, service_rows[counter_index], counter_cols[counter_index], jnp.where(counter_rows[counter_index] == 0, 0, 1))
    wrong_raw = has_raw & (jnp.argmax(inventory[2:]) != needed) & valid
    wrong_dish = has_dish & (jnp.argmax(inventory[2:]) != ingredient) & valid
    discard = (wrong_raw | wrong_dish) & jnp.any(empty_counter)
    # Reuse actual visible inventory; a stored dish is safe only before a new pot starts.
    raw_match = in_view & (counter_items[:, 0] == 0) & (counter_items[:, 1] == 0) & (jnp.sum(counter_items[:, 2:], axis=1) == 1) & (jnp.argmax(counter_items[:, 2:], axis=1) == needed)
    dish_match = in_view & (counter_items[:, 1] > 0) & (jnp.sum(counter_items[:, 2:], axis=1) == 3) & (jnp.argmax(counter_items[:, 2:], axis=1) == ingredient) & (pot_count == 0)
    reusable = (raw_match & (pot_count < 3)) | dish_match
    reuse_index = jnp.argmin(jnp.where(reusable, distances - dish_match.astype(jnp.int32) * 20, 100))
    to_reuse = service(row, col, facing, service_rows[reuse_index], counter_cols[reuse_index], jnp.where(counter_rows[reuse_index] == 0, 0, 1))
    # The display is visible from every row of column five, so avoid an unnecessary detour.
    await_recipe = jnp.where(col == 5, 4, 2)
    action = jnp.where(valid, jnp.where(jnp.any(reusable), to_reuse, to_supply), await_recipe)
    action = jnp.where((pot_count >= 3) | cooked | (pot_timer > 0), to_plate, action)
    action = jnp.where(has_plate | has_raw, to_pot, action)
    action = jnp.where(has_dish, to_goal, action)
    action = jnp.where(discard, to_counter, action)
    # Stage ingredients while the pot cooks. The two column-five counters
    # take one fewer step to fetch from than either supply. Each staging trip
    # finishes before plating is due; the partner need not know about this cache.
    preferred_cache = jnp.where(ingredient == 0, 0, 1)
    cache_index = jnp.where(empty_counter[preferred_cache], preferred_cache, 1 - preferred_cache)
    start_cache = ~late_signals & (memory[8] == 0) & ~has_plate & ~has_raw & ~has_dish & (pot_timer >= 12) & jnp.any(empty_counter[:2])
    cache_task = jnp.where(start_cache, cache_index + 1, memory[8])
    cache_target = jnp.clip(cache_task - 1, 0, 1)
    cache_direction = jnp.where(cache_target == 0, 0, 1)
    cache_row = service_rows[cache_target]
    cache_drop = service(row, col, facing, cache_row, 5, cache_direction)
    observe_cache = ~late_signals & (cache_task == 0) & ~has_plate & ~has_raw & ~has_dish & (pot_timer >= 13) & (row != 2) & ~jnp.all(in_view[:2])
    action = jnp.where(observe_cache, jnp.where(row < 2, 1, 3), action)
    action = jnp.where(cache_task > 0, jnp.where(has_raw, cache_drop, to_supply), action)
    cache_finished = (cache_task > 0) & has_raw & (row == cache_row) & (col == 5) & (facing == cache_direction) & (action == 5)
    cache_task = jnp.where(cache_finished, 0, cache_task)
    delivering = ~discard & has_dish & (row == 2) & (col == 6) & (facing == 2) & (action == 5)
    valid = valid & ~delivering
    wait_age = jnp.where(delivering | valid, 0, memory[4] + 1)
    unconfirmed = jnp.where(visible, 0, memory[5]) + delivering.astype(jnp.int32)
    cycle_observed = (visible | (memory[10] > 0)) & ~delivering
    delivery_time = jnp.where(delivering, memory[2], memory[9])
    next_memory = jnp.array([ingredient, valid.astype(jnp.int32), memory[2] + 1, seen.astype(jnp.int32), wait_age, unconfirmed, jnp.where(visible, jnp.argmax(recipe), memory[6]), change_signaler.astype(jnp.int32), cache_task, delivery_time, cycle_observed.astype(jnp.int32), late_signals.astype(jnp.int32)], jnp.int32)
    return action.astype(jnp.int32), next_memory, {'recipe':ingredient,'valid':valid,'pot_count':pot_count,'pot_timer':pot_timer,'position':jnp.array([row,col]),'inventory':inventory,'delivery':delivering,'cache_task':cache_task,'late_signals':late_signals}
