"""Convention-robust solo worker for the asymmetric test_time_simple board."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'alternating_solo_worker',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


# Codes correspond to observation channels 20:29; zero denotes floor.
_BOARD = (
    [1, 1, 9, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 6, 0, 0, 7],
    [4, 0, 0, 1, 3, 0, 0, 2],
    [1, 0, 0, 1, 6, 0, 0, 8],
    [1, 1, 9, 1, 1, 1, 1, 1],
)
_SPAWN_CANDIDATES = (
    [1, 5], [1, 6], [2, 5], [2, 6], [3, 5], [3, 6],
)


def _static_crop(position):
    board = jnp.asarray(_BOARD, dtype=jnp.int32)
    codes_to_check = jnp.arange(1, 10, dtype=jnp.int32)
    rows = position[0] + jnp.arange(5, dtype=jnp.int32) - 2
    cols = position[1] + jnp.arange(5, dtype=jnp.int32) - 2
    rr = jnp.clip(rows[:, None], 0, 4)
    cc = jnp.clip(cols[None, :], 0, 7)
    valid = ((rows[:, None] >= 0) & (rows[:, None] < 5) &
             (cols[None, :] >= 0) & (cols[None, :] < 8))
    codes = jnp.where(valid, board[rr, cc], 0)
    return (codes[..., None] == codes_to_check).astype(jnp.int32)


def _infer_position(obs):
    candidates = jnp.asarray(_SPAWN_CANDIDATES, dtype=jnp.int32)
    expected_static = jax.vmap(_static_crop)(candidates)
    observed = obs[:, :, 20:29]
    error = jnp.sum(jnp.abs(observed[None, ...] - expected_static), axis=(1, 2, 3))
    return candidates[jnp.argmin(error)]


def _facing_action(direction):
    return jnp.where(direction == 0, 3,
                     jnp.where(direction == 1, 1,
                               jnp.where(direction == 2, 0, 2)))


def _navigate(position, target, desired_facing, facing):
    at_target = jnp.all(position == target)
    vertical = jnp.where(position[0] < target[0], 1,
                         jnp.where(position[0] > target[0], 3, 4))
    horizontal = jnp.where(position[1] < target[1], 0,
                           jnp.where(position[1] > target[1], 2, 4))
    travel = jnp.where(vertical != 4, vertical, horizontal)
    turn_or_interact = jnp.where(
        facing == desired_facing, jnp.asarray(5, dtype=jnp.int32),
        _facing_action(desired_facing),
    )
    return jnp.where(at_target, turn_or_interact, travel).astype(jnp.int32)


def _advance_position(position, action):
    floor = (jnp.asarray(_BOARD, dtype=jnp.int32) == 0)
    dr = jnp.asarray([0, 1, 0, -1, 0, 0], dtype=jnp.int32)[action]
    dc = jnp.asarray([1, 0, -1, 0, 0, 0], dtype=jnp.int32)[action]
    candidate = position + jnp.asarray([dr, dc], dtype=jnp.int32)
    inside = ((candidate[0] >= 0) & (candidate[0] < 5) &
              (candidate[1] >= 0) & (candidate[1] < 8))
    safe = jnp.clip(candidate, jnp.asarray([0, 0]), jnp.asarray([4, 7]))
    can_move = (action < 4) & inside & floor[safe[0], safe[1]]
    return jnp.where(can_move, candidate, position)


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero = jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(4, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    # [row, col, delivery parity, raw ingredients in pot, stage]
    return jnp.asarray([2, 5, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    reset_position = _infer_position(obs)
    position = jnp.where(reset, reset_position, memory[:2])
    cycle = memory[2]
    pot_count = memory[3]
    stage = memory[4]
    target_ingredient = cycle % 2

    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_target = obs[2, 2, 7 + target_ingredient] > 0
    pot_ready = jnp.max(obs[:, :, 22] * obs[:, :, 30]) > 0

    effective_stage = jnp.where(held_cooked, 3,
                                jnp.where(held_plate, 2,
                                          jnp.where((stage == 0) & (pot_count >= 3),
                                                    1, stage)))
    supply_stand = jnp.where(target_ingredient == 0,
                             jnp.asarray([1, 6], dtype=jnp.int32),
                             jnp.asarray([3, 6], dtype=jnp.int32))

    gather_action = jnp.where(
        held_target,
        _navigate(position, jnp.asarray([2, 5], dtype=jnp.int32), 3, facing),
        _navigate(position, supply_stand, 2, facing),
    )
    cook_action = jnp.where(
        pot_ready,
        _navigate(position, jnp.asarray([1, 5], dtype=jnp.int32), 3, facing),
        jnp.where(
            jnp.all(position == jnp.asarray([1, 5], dtype=jnp.int32)),
            jnp.asarray(4, dtype=jnp.int32),
            _navigate(position, jnp.asarray([1, 5], dtype=jnp.int32), 3, facing),
        ),
    )
    plate_action = _navigate(position, jnp.asarray([2, 5], dtype=jnp.int32), 3, facing)
    deliver_action = _navigate(position, jnp.asarray([2, 6], dtype=jnp.int32), 2, facing)
    action = jnp.where(effective_stage == 0, gather_action,
                       jnp.where(effective_stage == 1, cook_action,
                                 jnp.where(effective_stage == 2, plate_action,
                                           deliver_action)))

    at_pot = jnp.all(position == jnp.asarray([2, 5], dtype=jnp.int32))
    add_ingredient = (effective_stage == 0) & (action == 5) & at_pot & held_target
    new_pot_count = jnp.minimum(3, pot_count + add_ingredient.astype(jnp.int32))
    at_plate = jnp.all(position == jnp.asarray([1, 5], dtype=jnp.int32))
    pick_plate = ((effective_stage == 1) & pot_ready & (action == 5) & at_plate)
    pick_food = ((effective_stage == 2) & (action == 5) & at_pot & held_plate)
    at_goal = jnp.all(position == jnp.asarray([2, 6], dtype=jnp.int32))
    deliver = (effective_stage == 3) & (action == 5) & at_goal & held_cooked

    next_stage = effective_stage
    next_stage = jnp.where(new_pot_count >= 3, 1, next_stage)
    next_stage = jnp.where(pick_plate, 2, next_stage)
    next_stage = jnp.where(pick_food, 3, next_stage)
    next_stage = jnp.where(deliver, 0, next_stage)
    next_pot_count = jnp.where(deliver, 0, new_pot_count)
    next_cycle = cycle + deliver.astype(jnp.int32)
    next_position = _advance_position(position, action)
    next_memory = jnp.asarray([
        next_position[0], next_position[1], next_cycle, next_pot_count, next_stage,
    ], dtype=jnp.int32)
    return action, next_memory, {}
