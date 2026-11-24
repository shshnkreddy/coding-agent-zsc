"""Reactive, geometry-grounded policy for grounded_coord_simple."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'reactive_single_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


_ROWS = (1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 2, 2)
_COLS = (1, 2, 1, 2, 1, 2, 5, 6, 5, 6, 5, 6, 3, 4)
_NEXT = (
    (4, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (3, 3, 4, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (3, 3, 2, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (3, 3, 3, 3, 4, 0, 3, 3, 3, 3, 3, 3, 3, 3),
    (3, 3, 3, 3, 2, 4, 3, 3, 3, 3, 3, 3, 3, 3),
    (1, 1, 1, 1, 1, 1, 4, 0, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2, 3, 3, 4, 0, 1, 1, 2, 2),
    (2, 2, 2, 2, 2, 2, 3, 3, 2, 4, 1, 1, 2, 2),
    (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0, 3, 3),
    (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 3, 3),
    (2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 4, 0),
    (2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 4),
)
_IDX_RECIPE = 3
_IDX_POT_STAND = 8
_IDX_TOP_PLATE_STAND = 6
_IDX_BOTTOM_PLATE_STAND = 10
_IDX_DELIVERY_STAND = 9
_IDX_ING0_STAND = 7
_IDX_ING1_STAND = 11


def _position(obs):
    rr = jnp.arange(5, dtype=jnp.int32)[:, None]
    cc = jnp.arange(5, dtype=jnp.int32)[None, :]
    recipe = obs[:, :, 23] > 0
    pot = obs[:, :, 22] > 0
    recipe_n = jnp.maximum(jnp.sum(recipe), 1)
    pot_n = jnp.maximum(jnp.sum(pot), 1)
    recipe_row = jnp.sum(rr * recipe) / recipe_n
    recipe_col = jnp.sum(cc * recipe) / recipe_n
    pot_row = jnp.sum(rr * pot) / pot_n
    pot_col = jnp.sum(cc * pot) / pot_n
    global_row = jnp.where(jnp.any(recipe), 4 - recipe_row, 4 - pot_row)
    global_col = jnp.where(jnp.any(recipe), 2 - recipe_col, 6 - pot_col)
    rows = jnp.asarray(_ROWS, dtype=jnp.int32)
    cols = jnp.asarray(_COLS, dtype=jnp.int32)
    return jnp.argmax((rows == global_row) & (cols == global_col)).astype(jnp.int32)


def _recipe(obs, old):
    seen0 = jnp.any(obs[:, :, 36] > 0)
    seen1 = jnp.any(obs[:, :, 37] > 0)
    return jnp.where(seen1, jnp.asarray(1, jnp.int32),
                     jnp.where(seen0, jnp.asarray(0, jnp.int32), old))


def _pot_info(obs):
    is_pot = obs[:, :, 22] > 0
    count = jnp.sum(obs[:, :, 31:34] * is_pot[:, :, None])
    cooked = jnp.any((obs[:, :, 30] > 0) & is_pot)
    timer = jnp.sum(obs[:, :, 39] * is_pot)
    visible = jnp.any(is_pot)
    return count.astype(jnp.int32), cooked, timer.astype(jnp.int32), visible


def _agent0(obs, reset, memory, rng):
    """Left-room coordinator: expose the current recipe through button L."""
    del rng
    next_table = jnp.asarray(_NEXT, dtype=jnp.int32)
    initial = jnp.asarray([-1, 0, 0], dtype=jnp.int32)
    memory = jnp.where(reset, initial, memory)
    pos = _position(obs)
    target = _recipe(obs, memory[0])
    mode = memory[1]
    tick = memory[2]
    center = obs[2, 2]
    facing_right = center[3] > 0
    changed = (target >= 0) & (memory[0] >= 0) & (target != memory[0])
    mode = jnp.where(changed, 0, mode)
    desired = jnp.where(mode == 0, _IDX_RECIPE, 0)
    action = next_table[pos, desired]
    at_button = pos == _IDX_RECIPE
    activate = (mode == 0) & at_button & facing_right & (target >= 0)
    action = jnp.where((mode == 0) & at_button & (~facing_right), 0, action)
    action = jnp.where(activate, 5, action)
    next_tick = jnp.where(activate, 0, (tick + 1) % 50)
    # Re-open the button periodically so a right-room partner can observe a
    # newly sampled recipe even when it equals the previous one.
    next_mode = jnp.where(activate, 1, mode)
    next_mode = jnp.where((next_tick == 49) & (mode == 1), 0, next_mode)
    next_memory = jnp.asarray([target, next_mode, next_tick], dtype=jnp.int32)
    trace = {'pos': pos, 'target': target, 'mode': mode, 'action': action}
    return action.astype(jnp.int32), next_memory, trace


def _agent1(obs, reset, memory, rng):
    """Right-room chef: read L, then perform a complete pure-ingredient cycle."""
    del rng
    next_table = jnp.asarray(_NEXT, dtype=jnp.int32)
    initial = jnp.asarray([-1, 5], dtype=jnp.int32)
    memory = jnp.where(reset, initial, memory)
    pos = _position(obs)
    target = _recipe(obs, memory[0])
    mode = memory[1]
    center = obs[2, 2]
    raw0 = center[7] > 0
    raw1 = center[8] > 0
    held_plate = center[5] > 0
    held_cooked = center[6] > 0
    held_raw = (raw0 | raw1) & (~held_cooked)
    empty = ~(held_raw | held_plate | held_cooked)
    facing_right = center[3] > 0
    facing_left = center[4] > 0
    pot_count, pot_cooked, pot_timer, pot_visible = _pot_info(obs)

    mode = jnp.where(held_raw, 1, mode)
    mode = jnp.where(held_plate & (~held_cooked), 3, mode)
    mode = jnp.where(held_cooked, 4, mode)
    mode = jnp.where(empty & (mode == 1) & pot_visible & (pot_count < 3), 0, mode)
    mode = jnp.where(empty & (mode == 4), 5, mode)
    mode = jnp.where(empty & pot_cooked, 3, mode)
    mode = jnp.where(empty & pot_visible & (pot_timer > 0), 2, mode)
    mode = jnp.where(empty & pot_visible & (pot_count >= 3) & (~pot_cooked), 2, mode)
    mode = jnp.where((mode == 5) & (target >= 0) &
                     (jnp.any(obs[:, :, 36] > 0) | jnp.any(obs[:, :, 37] > 0)), 0, mode)

    source_idx = jnp.where(target == 1, _IDX_ING1_STAND, _IDX_ING0_STAND)
    desired = jnp.where(mode == 5, _IDX_TOP_PLATE_STAND, source_idx)
    desired = jnp.where(mode == 1, _IDX_POT_STAND, desired)
    desired = jnp.where(mode == 2, _IDX_POT_STAND, desired)
    desired = jnp.where(mode == 3, _IDX_BOTTOM_PLATE_STAND, desired)
    desired = jnp.where((mode == 3) & held_plate, _IDX_POT_STAND, desired)
    desired = jnp.where(mode == 4, _IDX_DELIVERY_STAND, desired)
    action = next_table[pos, desired]
    at_source = pos == source_idx
    at_pot = pos == _IDX_POT_STAND
    at_plate = pos == _IDX_BOTTOM_PLATE_STAND
    at_delivery = pos == _IDX_DELIVERY_STAND
    do_interact = (mode == 0) & empty & at_source & facing_right
    do_interact = do_interact | ((mode == 1) & held_raw & at_pot & facing_left)
    do_interact = do_interact | ((mode == 3) & empty & at_plate & facing_left)
    do_interact = do_interact | ((mode == 3) & held_plate & at_pot & facing_left)
    do_interact = do_interact | ((mode == 4) & held_cooked & at_delivery & facing_right)
    action = jnp.where(((mode == 0) & empty & at_source & (~facing_right)), 0, action)
    action = jnp.where(((mode == 1) & held_raw & at_pot & (~facing_left)), 2, action)
    action = jnp.where(((mode == 3) & ((empty & at_plate) | (held_plate & at_pot)) &
                        (~facing_left)), 2, action)
    action = jnp.where(((mode == 4) & held_cooked & at_delivery & (~facing_right)), 0, action)
    action = jnp.where(do_interact, 5, action)
    next_mode = jnp.where((mode == 4) & held_cooked & at_delivery & facing_right, 5, mode)
    next_target = jnp.where((mode == 4) & held_cooked & at_delivery & facing_right, -1, target)
    next_memory = jnp.asarray([next_target, next_mode], dtype=jnp.int32)
    trace = {
        'pos': pos, 'target': target, 'mode': mode, 'action': action,
        'recipe_seen': jnp.any(obs[:, :, 36] > 0) | jnp.any(obs[:, :, 37] > 0),
        'pot_visible': pot_visible, 'pot_count': pot_count, 'held_raw': held_raw,
        'held_plate': held_plate, 'held_cooked': held_cooked,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_0():
    return jnp.asarray([-1, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    return _agent0(obs, reset, memory, rng)


def init_agent_1():
    return jnp.asarray([-1, 5], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    return _agent1(obs, reset, memory, rng)
