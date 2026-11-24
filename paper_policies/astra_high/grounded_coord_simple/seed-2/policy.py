"""Grounded button communication and locally observed cooking."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'robust_grounded_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def init_agent_0():
    # Time, signal age, previous pot count, unanswered retries, progress age.
    return jnp.array([0, 100, 0, 0, 0], dtype=jnp.int32)


def init_agent_1():
    # Time, target, fallback, wait, confirmation, responsiveness, periodicity,
    # last delivery time, and last visible recipe time. All memory is private.
    return jnp.array([0, -1, 0, 0, 0, 0, 0, -1, -1000], dtype=jnp.int32)


def position(obs, channel, global_row, global_col):
    index = jnp.argmax(obs[:, :, channel])
    return global_row + 2 - index // 5, global_col + 2 - index % 5


def navigate(row, col, target_row, target_col, direction, facing, interact=True):
    """Navigate within a rectangular room, then face and interact."""
    turn = jnp.array([3, 1, 0, 2], dtype=jnp.int32)[direction]
    at_target = jnp.where(facing == direction, 5 if interact else 4, turn)
    return jnp.where(row < target_row, 1,
           jnp.where(row > target_row, 3,
           jnp.where(col < target_col, 0,
           jnp.where(col > target_col, 2, at_target)))).astype(jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_0(), memory)
    row, col = position(obs, 24, 2, 3)
    facing = jnp.argmax(obs[2, 2, 1:5])
    pot_count = jnp.sum(obs[:, :, 31:34] * obs[:, :, 22:23])
    age = memory[1] + 1
    progress = pot_count != memory[2]
    progress_age = jnp.where(progress, 0, memory[4] + 1)
    retries = jnp.where(progress, 0, memory[3])
    interval = jnp.array([9, 12, 24, 48, 72], dtype=jnp.int32)[jnp.minimum(retries, 4)]
    needs_display = (pot_count == 0) | ((pot_count < 3) & (progress_age >= 18))
    can_finish = memory[0] <= 358 + 6 * pot_count
    reveal = needs_display & (age >= interval) & can_finish
    action = navigate(row, col, 2, 2, 2, facing, False)
    action = jnp.where((row == 2) & (col == 2) & (facing == 2) & reveal, 5, action)
    age = jnp.where(action == 5, 0, age)
    retries = retries + (action == 5).astype(jnp.int32)
    next_memory = jnp.array([memory[0] + 1, age, pot_count, retries, progress_age], dtype=jnp.int32)
    return action, next_memory, {'pot_count': pot_count, 'signal_age': age, 'progress_age': progress_age}


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_1(), memory)
    row, col = position(obs, 22, 2, 4)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    recipe = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    visible_recipe = jnp.sum(recipe) > 0
    last_seen = jnp.where(visible_recipe, memory[0], memory[8])
    signal_age = memory[0] - last_seen
    target = jnp.where(visible_recipe, jnp.argmax(recipe), memory[1])
    waiting = jnp.where(target < 0, memory[3] + 1, 0)
    target = jnp.where((target < 0) & (waiting >= jnp.where((memory[6] > 0) & (signal_age <= 80), 45, 12)), memory[2], target)
    confidence = jnp.where(visible_recipe, 1, memory[4])
    pot = jnp.sum(obs[:, :, 29:34] * obs[:, :, 22:23], axis=(0, 1))
    pot_count = jnp.sum(pot[2:])
    pot_timer = jnp.sum(obs[:, :, 39] * obs[:, :, 22])
    responsive = jnp.maximum(memory[5], (visible_recipe & (memory[7] >= 0) & (memory[8] >= 0) & (memory[8] < memory[7] - 20)).astype(jnp.int32))
    periodic = jnp.maximum(memory[6], (visible_recipe & (pot_count == 3) & (pot_timer > 0) & (pot_timer <= 14)).astype(jnp.int32))
    desired = jnp.where(pot_count > 0, jnp.argmax(pot[2:]), target)
    ingredient_row = jnp.where(desired == 0, 1, 3)
    supply_action = navigate(row, col, ingredient_row, 6, 2, facing)
    plate_row = jnp.where(row <= 2, 1, 3)
    plate_action = navigate(row, col, plate_row, 5, 3, facing)
    pot_action = navigate(row, col, 2, 5, 3, facing)
    wait_action = navigate(row, col, 2, 5, 3, facing, False)
    delivery_action = navigate(row, col, 2, 6, 2, facing)
    empty_action = jnp.where(pot_count >= 3, plate_action,
                  jnp.where(target >= 0, supply_action, wait_action))
    action = jnp.where(inv[1] > 0, delivery_action,
             jnp.where(inv[0] > 0, jnp.where(pot[1] > 0, pot_action, wait_action),
             jnp.where(jnp.sum(inv[2:]) > 0, pot_action, empty_action)))
    # Counters are public storage, useful when a late display corrects a guess.
    counter_rows = jnp.array([0, 0, 4, 4], dtype=jnp.int32)
    counter_cols = jnp.array([5, 6, 5, 6], dtype=jnp.int32)
    adjacent_rows = jnp.array([1, 1, 3, 3], dtype=jnp.int32)
    directions = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    local_rows = counter_rows - row + 2
    local_cols = counter_cols - col + 2
    visible = (local_rows >= 0) & (local_rows < 5) & (local_cols >= 0) & (local_cols < 5)
    items = obs[jnp.clip(local_rows, 0, 4), jnp.clip(local_cols, 0, 4), 29:34]
    empty_counters = visible & (jnp.sum(items, axis=1) == 0)
    distances = jnp.abs(row - adjacent_rows) + jnp.abs(col - counter_cols)
    drop_index = jnp.argmin(jnp.where(empty_counters, distances, 100))
    drop_action = navigate(row, col, adjacent_rows[drop_index], counter_cols[drop_index], directions[drop_index], facing)
    raw_mismatch = (inv[0] == 0) & (jnp.sum(inv[2:]) > 0) & (desired >= 0) & (jnp.argmax(inv[2:]) != desired)
    dish_mismatch = (inv[1] > 0) & (((confidence > 0) & (jnp.argmax(inv[2:]) != target)) | (jnp.max(inv[2:]) != 3))
    must_store = raw_mismatch | dish_mismatch
    fallback_drop_action = jnp.where(jnp.all(visible), action, wait_action)
    action = jnp.where(must_store, jnp.where(jnp.any(empty_counters), drop_action, fallback_drop_action), action)
    matching_dish = visible & (items[:, 1] > 0) & (items[:, jnp.maximum(target, 0) + 2] == 3) & (target >= 0) & (pot_count == 0)
    matching_raw = visible & (items[:, 0] == 0) & (items[:, 1] == 0) & (items[:, jnp.maximum(desired, 0) + 2] == 1) & (desired >= 0) & (pot_count < 3)
    reusable = matching_dish | matching_raw
    reuse_index = jnp.argmin(jnp.where(reusable, distances - matching_dish.astype(jnp.int32) * 10, 100))
    reuse_action = navigate(row, col, adjacent_rows[reuse_index], counter_cols[reuse_index], directions[reuse_index], facing)
    action = jnp.where((jnp.sum(inv) == 0) & jnp.any(reusable), reuse_action, action)
    delivered = (inv[1] > 0) & (row == 2) & (col == 6) & (facing == 2) & (action == 5)
    fallback = jnp.where(delivered, jnp.where((confidence > 0) | (signal_age <= 100) | ((responsive > 0) & (signal_age <= 160)), jnp.argmax(inv[2:]), 1 - jnp.argmax(inv[2:])), memory[2])
    target = jnp.where(delivered, -1, target)
    confidence = jnp.where(delivered, 0, confidence)
    waiting = jnp.where(delivered, 0, waiting)
    last_delivery = jnp.where(delivered, memory[0], memory[7])
    next_memory = jnp.array([memory[0] + 1, target, fallback, waiting, confidence, responsive, periodic, last_delivery, last_seen], dtype=jnp.int32)
    return action.astype(jnp.int32), next_memory, {'target': target, 'pot_count': pot_count, 'position': jnp.array([row, col]), 'confidence': confidence, 'responsive': responsive, 'periodic': periodic, 'inventory': inv, 'delivered': delivered, 'storing': must_store, 'signal_age': signal_age}
