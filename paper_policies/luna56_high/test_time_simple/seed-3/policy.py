"""Coordinated policy for the isolated two-room kitchen.

Seat 0 can see the always-on recipe indicator but cannot reach the kitchen.
Seat 1 can reach every kitchen object but cannot see that indicator.  The
counter at (2, 3) is visible from both rooms, so seat 0 uses an ingredient-2
token there as a one-bit message: token present means recipe 000, and token
absent means recipe 111.  Both seats use fixed schedules, with no hidden
state or host interaction.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def _lookup(value, keys, values):
    result = jnp.asarray(4, dtype=jnp.int32)
    for key, action_value in zip(keys, values):
        result = jnp.where(
            value == key,
            jnp.asarray(action_value, dtype=jnp.int32),
            result,
        )
    return result


def init_agent_0():
    # mode, clock, remembered target, maintenance kind (0 none, 1 clear, 2 place)
    return jnp.asarray([0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    mode = jnp.where(reset, 0, memory[0]).astype(jnp.int32)
    clock = jnp.where(reset, 0, memory[1]).astype(jnp.int32)
    remembered_target = jnp.where(reset, 0, memory[2]).astype(jnp.int32)
    maintenance_kind = jnp.where(reset, 0, memory[3]).astype(jnp.int32)
    seen_target = jnp.where(obs[:, :, 37].sum() > 0, 1, 0).astype(jnp.int32)
    initial_read = (mode == 0) & (clock == 7)
    refresh_read = (mode == 1) & (clock == 0)
    read_now = initial_read | refresh_read
    target = jnp.where(read_now, seen_target, remembered_target)

    # Initial positioning and token placement for recipe 000.
    initial_keys = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    initial_actions = (3, 3, 3, 2, 2, 2, 1, 3, 0, 3, 5, 1, 0, 5)
    initial_action = _lookup(clock, initial_keys, initial_actions)
    initial_action = jnp.where((clock >= 7) & (target == 1), 4, initial_action)

    # At a refresh, remove an old token or start the route that places one.
    refresh_action = jnp.where(
        (remembered_target == 0) & (target == 1), 5,
        jnp.where((remembered_target == 1) & (target == 0), 3, 4),
    ).astype(jnp.int32)
    clear_action = _lookup(clock, (6, 5, 4, 3, 2, 1), (2, 2, 5, 4, 4, 4))
    place_action = _lookup(clock, (6, 5, 4, 3, 2, 1), (0, 3, 5, 1, 0, 5))
    maintenance_action = jnp.where(
        maintenance_kind == 1, clear_action,
        jnp.where(maintenance_kind == 2, place_action, 4),
    )
    mode1_action = jnp.where(clock == 0, refresh_action, 4).astype(jnp.int32)
    action = jnp.where(mode == 0, initial_action,
                       jnp.where(mode == 1, mode1_action, maintenance_action))

    initial_done = (mode == 0) & (clock == 14)
    refresh_done = (mode == 1) & (clock == 0)
    next_mode = mode
    next_clock = clock + 1
    next_kind = maintenance_kind
    next_mode = jnp.where(initial_done, 1, next_mode)
    next_clock = jnp.where(initial_done, 40, next_clock)
    next_mode = jnp.where(refresh_done, 2, next_mode)
    next_clock = jnp.where(refresh_done, 6, next_clock)
    next_kind = jnp.where(
        refresh_done,
        jnp.where(
            (remembered_target == 0) & (target == 1), 1,
            jnp.where((remembered_target == 1) & (target == 0), 2, 0),
        ),
        next_kind,
    )
    maintenance_done = (mode == 2) & (clock == 1)
    next_mode = jnp.where(maintenance_done, 1, next_mode)
    next_clock = jnp.where(maintenance_done, 41, next_clock)
    next_kind = jnp.where(maintenance_done, 0, next_kind)
    next_clock = jnp.where((mode == 1) & (clock > 0), clock - 1, next_clock)
    next_clock = jnp.where((mode == 2) & (clock > 1), clock - 1, next_clock)
    next_memory = jnp.asarray([next_mode, next_clock, target, next_kind], dtype=jnp.int32)
    return action.astype(jnp.int32), next_memory, {}


def init_agent_1():
    # mode, cycle clock, target (0 or 1)
    return jnp.asarray([0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    mode = jnp.where(reset, 0, memory[0]).astype(jnp.int32)
    clock = jnp.where(reset, 0, memory[1]).astype(jnp.int32)
    remembered_target = jnp.where(reset, 0, memory[2]).astype(jnp.int32)
    token_present = obs[2, 0, 33] > 0
    initial_start = (mode == 0) & (clock == 14)
    cycle_start = (mode == 1) & (clock == 0)
    start = initial_start | cycle_start
    signaled_target = jnp.where(token_present, 0, 1).astype(jnp.int32)
    target = jnp.where(start, signaled_target, remembered_target)

    initial_action = _lookup(clock, (0, 1, 2, 3, 4, 5, 6), (3, 3, 3, 2, 2, 2, 1))
    initial_action = jnp.where(clock >= 7, 4, initial_action)

    cycle_keys_0 = tuple(range(42))
    cycle_actions_0 = (
        3, 0, 5, 1, 2, 5,
        3, 0, 5, 1, 2, 5,
        3, 0, 5, 1, 2, 5,
        3, 2, 5, 1, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 0, 0, 5, 2,
    )
    cycle_keys_1 = tuple(range(42))
    cycle_actions_1 = (
        1, 0, 5, 3, 2, 5,
        1, 0, 5, 3, 2, 5,
        1, 0, 5, 3, 2, 5,
        3, 2, 5, 1, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 0, 0, 5, 2,
    )
    cycle_clock = jnp.where(initial_start, 0, clock)
    cycle_action_0 = _lookup(cycle_clock, cycle_keys_0, cycle_actions_0)
    cycle_action_1 = _lookup(cycle_clock, cycle_keys_1, cycle_actions_1)
    cycle_action = jnp.where(target == 1, cycle_action_1, cycle_action_0)
    action = jnp.where(mode == 0, initial_action,
                       jnp.where(mode == 1, cycle_action, 4))
    action = jnp.where(initial_start, cycle_action, action)

    next_mode = mode
    next_clock = clock + 1
    next_mode = jnp.where(initial_start, 1, next_mode)
    next_clock = jnp.where(initial_start, 1, next_clock)
    end_0 = (mode == 1) & (target == 0) & (clock == 41)
    end_1 = (mode == 1) & (target == 1) & (clock == 41)
    cycle_end = end_0 | end_1
    next_mode = jnp.where(cycle_end, 2, next_mode)
    next_clock = jnp.where(cycle_end, 6, next_clock)
    maintenance_done = (mode == 2) & (clock == 1)
    next_mode = jnp.where(maintenance_done, 1, next_mode)
    next_clock = jnp.where(maintenance_done, 0, next_clock)
    next_clock = jnp.where((mode == 2) & (clock > 1), clock - 1, next_clock)
    next_memory = jnp.asarray([next_mode, next_clock, target], dtype=jnp.int32)
    return action.astype(jnp.int32), next_memory, {}
