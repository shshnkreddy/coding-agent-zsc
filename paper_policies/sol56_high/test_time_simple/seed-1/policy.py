"""Spatially signalled, observation-driven policy for test_time_simple."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'spatial_signal_pipeline',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _location(channel):
    """Local row and column of the (unique) nonzero cell in a channel."""
    flat = jnp.argmax(channel.reshape((-1,)))
    return flat // 5, flat % 5


def _self_facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _go_to(row, col, target_row, target_col):
    """Deterministic row-first motion through either room's 3x2 floor."""
    vertical = jnp.where(row < target_row, DOWN, UP)
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    return jnp.where(row != target_row, vertical,
                     jnp.where(col != target_col, horizontal, STAY))


def _use_from(obs, row, col, stand_row, stand_col, object_facing):
    """Walk to an interaction square, turn against the object, then use it."""
    at_stand = (row == stand_row) & (col == stand_col)
    facing = _self_facing(obs)
    turn_action = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)[object_facing]
    ready_action = jnp.where(facing == object_facing, INTERACT, turn_action)
    return jnp.where(at_stand, ready_action,
                     _go_to(row, col, stand_row, stand_col))


def init_agent_0():
    return jnp.zeros((2,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    # The indicator at global (2, 0) is visible everywhere in the left room.
    rr, rc = _location(obs[:, :, 23])
    row = jnp.int32(4) - rr
    col = jnp.int32(2) - rc

    recipe0 = jnp.sum(obs[:, :, 36])
    recipe1 = jnp.sum(obs[:, :, 37])
    desired_row = jnp.where(recipe0 >= recipe1, jnp.int32(1), jnp.int32(3))
    other_row = jnp.int32(4) - desired_row

    desired_lr = desired_row - row + 2
    desired_lc = jnp.int32(3) - col + 2
    other_lr = other_row - row + 2
    other_lc = desired_lc
    desired_marked = obs[desired_lr, desired_lc, 33] > 0
    other_marked = obs[other_lr, other_lc, 33] > 0
    holding = jnp.sum(obs[2, 2, 5:10]) > 0

    # Recycle a stale marker on recipe flips; otherwise fetch ingredient 2.
    put_marker = _use_from(obs, row, col, desired_row, jnp.int32(2), jnp.int32(2))
    take_old = _use_from(obs, row, col, other_row, jnp.int32(2), jnp.int32(2))
    take_supply = _use_from(obs, row, col, desired_row, jnp.int32(2),
                            jnp.where(desired_row == 1, jnp.int32(0), jnp.int32(1)))
    acquire = jnp.where(other_marked, take_old, take_supply)
    action = jnp.where(holding, put_marker,
                       jnp.where(desired_marked, STAY, acquire))

    next_memory = jnp.asarray([desired_row, memory[1] + 1], dtype=jnp.int32)
    next_memory = jnp.where(reset, jnp.asarray([desired_row, 0], dtype=jnp.int32),
                            next_memory)
    trace = {
        'recipe_row': desired_row,
        'desired_marked': desired_marked,
        'other_marked': other_marked,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # target, previous dish flag, grace counter, refresh phase
    return jnp.asarray([-1, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    pot_r, pot_c = _location(obs[:, :, 22])
    row = jnp.int32(4) - pot_r
    col = jnp.int32(6) - pot_c

    old_target = jnp.where(reset, jnp.int32(-1), memory[0])
    # Ingredient 2 is not in either recipe, making it a natural signal token.
    # Read only the two communicating counters, excluding ingredient 2 in a pot.
    signal_col = jnp.maximum(jnp.int32(5) - col, 0)
    top_row = jnp.int32(3) - row
    bottom_row = jnp.int32(5) - row
    can_see_counters = col == 5
    top_signal = can_see_counters & (obs[top_row, signal_col, 33] > 0)
    bottom_signal = can_see_counters & (obs[bottom_row, signal_col, 33] > 0)
    one_signal = top_signal ^ bottom_signal
    two_signals = top_signal & bottom_signal & (old_target >= 0)
    signal_valid = one_signal | two_signals
    observed_target = jnp.where(one_signal,
                                jnp.where(top_signal, jnp.int32(0), jnp.int32(1)),
                                jnp.int32(1) - old_target)

    target = jnp.where(signal_valid, observed_target, old_target)

    plate = obs[2, 2, 5] > 0
    dish = obs[2, 2, 6] > 0
    raw0 = obs[2, 2, 7] > 0
    raw1 = obs[2, 2, 8] > 0
    raw = raw0 | raw1 | (obs[2, 2, 9] > 0)
    held_type = jnp.where(raw0, jnp.int32(0),
                          jnp.where(raw1, jnp.int32(1), jnp.int32(2)))
    raw_mismatch = raw & (held_type != target)
    empty_handed = ~(plate | dish | raw)

    pot_plate = obs[pot_r, pot_c, 29] > 0
    pot_cooked = obs[pot_r, pot_c, 30] > 0
    pot_count = jnp.sum(obs[pot_r, pot_c, 31:34])
    pot_timer = obs[pot_r, pot_c, 39]
    pot_ready = pot_cooked & (pot_timer == 0)
    pot_needs_food = (~pot_cooked) & (pot_timer == 0) & (pot_count < 3)

    prev_dish = jnp.where(reset, jnp.int32(0), memory[1])
    delivered = (prev_dish > 0) & (~dish)
    old_cooldown = jnp.where(reset, jnp.int32(0), memory[2])
    cooldown = jnp.where(delivered, jnp.int32(3), jnp.maximum(old_cooldown - 1, 0))
    old_phase = jnp.where(reset, jnp.int32(0), memory[3])
    base_phase = jnp.where(delivered, jnp.int32(1), old_phase)
    can_observe_marker = col == 5
    changed_signal = signal_valid & (observed_target != old_target)
    marker_removed = (base_phase == 1) & can_observe_marker & (~signal_valid)
    grace_phase = jnp.where(changed_signal | marker_removed | (cooldown == 0),
                            jnp.int32(0), jnp.int32(1))
    wait_phase = jnp.where(signal_valid, jnp.int32(0), jnp.int32(2))
    refresh_phase = jnp.where(base_phase == 1, grace_phase,
                              jnp.where(base_phase == 2, wait_phase, jnp.int32(0)))
    target = jnp.where(marker_removed, jnp.int32(1) - old_target, target)

    use_pot = _use_from(obs, row, col, jnp.int32(2), jnp.int32(5), jnp.int32(3))
    use_goal = _use_from(obs, row, col, jnp.int32(2), jnp.int32(6), jnp.int32(2))
    ingredient_row = jnp.where(target == 0, jnp.int32(1), jnp.int32(3))
    take_ingredient = _use_from(obs, row, col, ingredient_row, jnp.int32(6), jnp.int32(2))
    cache_global_row = jnp.where(target == 0, jnp.int32(0), jnp.int32(4))
    cache_lr = jnp.clip(cache_global_row - row + 2, 0, 4)
    cache_lc = jnp.clip(jnp.int32(6) - col + 2, 0, 4)
    cache_channel = jnp.int32(31) + jnp.clip(target, 0, 1)
    cache_visible = jnp.abs(cache_global_row - row) <= 2
    cache_present = cache_visible & (obs[cache_lr, cache_lc, cache_channel] > 0)
    cache_facing = jnp.where(target == 0, jnp.int32(0), jnp.int32(1))
    take_cache = _use_from(obs, row, col, ingredient_row, jnp.int32(6), cache_facing)
    take_food = jnp.where(cache_present, take_cache, take_ingredient)

    held_cache_row = jnp.where(held_type == 0, jnp.int32(1), jnp.int32(3))
    held_cache_facing = jnp.where(held_type == 0, jnp.int32(0), jnp.int32(1))
    put_cache = _use_from(obs, row, col, held_cache_row, jnp.int32(6),
                          held_cache_facing)
    plate_row = jnp.where(target == 1, jnp.int32(3), jnp.int32(1))
    take_plate = _use_from(obs, row, col, plate_row, jnp.int32(5), jnp.int32(3))
    observe_signal = _go_to(row, col, jnp.int32(2), jnp.int32(5))

    empty_action = jnp.where(pot_ready, take_plate,
                             jnp.where(pot_needs_food, take_food, take_plate))
    work_action = jnp.where(dish, use_goal,
                            jnp.where(plate, use_pot,
                                      jnp.where(raw, jnp.where(raw_mismatch, put_cache,
                                                               use_pot), empty_action)))
    paused = (target < 0) | ((refresh_phase > 0) & empty_handed)
    action = jnp.where(paused, observe_signal, work_action)

    next_memory = jnp.asarray([target, dish.astype(jnp.int32), cooldown,
                               refresh_phase], dtype=jnp.int32)
    next_memory = jnp.where(reset,
                            jnp.asarray([target, dish.astype(jnp.int32), cooldown, 0],
                                        dtype=jnp.int32),
                            next_memory)
    trace = {
        'target': target,
        'signal_visible': signal_valid,
        'cooldown': cooldown,
        'refresh_phase': refresh_phase,
        'pot_count': pot_count,
        'pot_timer': pot_timer,
        'raw_mismatch': raw_mismatch,
    }
    return action.astype(jnp.int32), next_memory, trace
