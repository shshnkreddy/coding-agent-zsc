"""Natural spatial signaling and reactive production for test_time_simple."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "counter_row_signal_v7",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 8,
    "num_ingredients": 3,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _location_from_landmark(obs, channel, landmark_row, landmark_col):
    """Recover global self position from a unique, always-visible landmark."""
    flat = jnp.argmax(obs[:, :, channel].reshape(-1))
    local_row = flat // 5
    local_col = flat % 5
    row = jnp.asarray(landmark_row + 2, jnp.int32) - local_row
    col = jnp.asarray(landmark_col + 2, jnp.int32) - local_col
    return row, col


def _move_to(row, col, target_row, target_col):
    vertical = jnp.where(row < target_row, DOWN, UP)
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    return jnp.where(row != target_row, vertical,
                     jnp.where(col != target_col, horizontal, STAY))


def _use_object(row, col, facing, target_row, target_col, face_dir, face_action):
    move = _move_to(row, col, target_row, target_col)
    at_target = (row == target_row) & (col == target_col)
    use = jnp.where(facing == face_dir, INTERACT, face_action)
    return jnp.where(at_target, use, move)


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    row, col = _location_from_landmark(obs, 23, 2, 0)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    recipe_0 = jnp.sum(obs[:, :, 36])
    recipe_1 = jnp.sum(obs[:, :, 37])
    target = jnp.where(recipe_1 > recipe_0, 1, 0).astype(jnp.int32)

    marker_col = 5 - col
    upper_row = 3 - row
    lower_row = 5 - row
    upper = obs[upper_row, marker_col, 33] > 0
    lower = obs[lower_row, marker_col, 33] > 0
    desired = jnp.where(target == 0, upper, lower)
    opposite = jnp.where(target == 0, lower, upper)
    held = jnp.sum(obs[2, 2, 7:10]) > 0
    desired_row = jnp.where(target == 0, 1, 3).astype(jnp.int32)
    opposite_row = jnp.where(target == 0, 3, 1).astype(jnp.int32)

    place = _use_object(row, col, facing, desired_row, 2,
                        jnp.asarray(2, jnp.int32), RIGHT)
    retrieve = _use_object(row, col, facing, opposite_row, 2,
                           jnp.asarray(2, jnp.int32), RIGHT)
    supply_face = jnp.where(target == 0, 0, 1).astype(jnp.int32)
    supply_action = jnp.where(target == 0, UP, DOWN)
    acquire = _use_object(row, col, facing, desired_row, 2,
                          supply_face, supply_action)
    # Remaining beside the active marker makes a recipe flip immediately
    # observable: the old marker is removed on the first post-delivery action.
    monitor = _move_to(row, col, desired_row, 2)
    empty_action = jnp.where(opposite, retrieve,
                             jnp.where(~desired, acquire, monitor))
    action = jnp.where(held, place, empty_action).astype(jnp.int32)
    next_memory = memory.at[0].add(jnp.asarray(1, jnp.int32))
    trace = {"target": target, "upper_marker": upper,
             "lower_marker": lower}
    return action, next_memory, trace


def init_agent_1():
    # phase, remembered target, post-delivery wait, diagnostic step
    return jnp.asarray([0, -1, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    phase, remembered, wait, step = memory
    row, col = _location_from_landmark(obs, 21, 2, 7)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    at_reader = (row == 2) & (col == 5)
    upper = obs[1, 0, 33] > 0
    central = obs[2, 0, 33] > 0
    lower = obs[3, 0, 33] > 0
    row_code = upper != lower
    refresh_gap = ((phase == 4) & (remembered >= 0) & (~upper) &
                   (~lower) & (~central))
    row_target = jnp.where(lower, 1, 0).astype(jnp.int32)
    initial_code = (phase == 0) & row_code
    changed_code = ((phase == 4) & row_code &
                    ((row_target != remembered) | (wait >= 2)))
    unambiguous = at_reader & (initial_code | changed_code | refresh_gap)
    # Natural row code takes priority.  A gap is meaningful only immediately
    # after a delivery with a previously observed row code: prompt removal of
    # that old marker indicates that it is moving to the opposite recipe row.
    observed_target = jnp.where(
        row_code, row_target, 1 - remembered).astype(jnp.int32)

    plate = obs[2, 2, 5] > 0
    cooked_held = obs[2, 2, 6] > 0
    raw_held = jnp.sum(obs[2, 2, 7:10]) > 0
    empty_hands = ~(plate | cooked_held | raw_held)

    pot_row = 4 - row
    pot_col = 6 - col
    pot_total = jnp.sum(obs[pot_row, pot_col, 31:34])
    pot_cooked = obs[pot_row, pot_col, 30] > 0
    pot_timer = obs[pot_row, pot_col, 39]
    pot_full = (pot_total >= 3) | pot_cooked | (pot_timer > 0)

    got_signal = unambiguous
    target = jnp.where(got_signal, observed_target, remembered)
    to_plate = (phase == 1) & pot_full & ~raw_held
    to_deliver = (phase == 2) & cooked_held
    delivered = (phase == 3) & empty_hands

    next_phase = phase
    next_phase = jnp.where(got_signal, 1, next_phase)
    next_phase = jnp.where(to_plate, 2, next_phase)
    next_phase = jnp.where(to_deliver, 3, next_phase)
    next_phase = jnp.where(delivered, 4, next_phase)
    next_phase = jnp.where((phase == 4) & unambiguous, 1, next_phase)
    reading = ((phase == 0) | (phase == 4)) & at_reader
    next_wait = jnp.where(delivered | got_signal, 0,
                          jnp.where(reading, wait + 1, wait))

    inspect_action = _move_to(row, col, 2, 5)
    supply_row = jnp.where(target == 0, 1, 3).astype(jnp.int32)
    fetch = _use_object(row, col, facing, supply_row, 6,
                        jnp.asarray(2, jnp.int32), RIGHT)
    insert = _use_object(row, col, facing, 2, 5,
                         jnp.asarray(3, jnp.int32), LEFT)
    ingredient_action = jnp.where(raw_held, insert,
                                  jnp.where(pot_full, STAY, fetch))

    plate_row = jnp.where(target == 0, 1, 3).astype(jnp.int32)
    get_plate = _use_object(row, col, facing, plate_row, 5,
                            jnp.asarray(3, jnp.int32), LEFT)
    scoop = _use_object(row, col, facing, 2, 5,
                        jnp.asarray(3, jnp.int32), LEFT)
    plate_action = jnp.where(plate | cooked_held, scoop, get_plate)

    deliver_action = _use_object(row, col, facing, 2, 6,
                                 jnp.asarray(2, jnp.int32), RIGHT)
    # Use the phase implied by this observation immediately.  This matters at
    # pot-full, cooked-pickup, delivery, and signal-refresh boundaries.
    action = jnp.where(next_phase == 0, inspect_action,
             jnp.where(next_phase == 1, ingredient_action,
             jnp.where(next_phase == 2, plate_action,
             jnp.where(next_phase == 3, deliver_action, inspect_action))))

    next_memory = jnp.stack([
        next_phase.astype(jnp.int32), target.astype(jnp.int32),
        next_wait.astype(jnp.int32), (step + 1).astype(jnp.int32),
    ])
    trace = {"phase": phase, "target": target, "pot_total": pot_total,
             "signal_valid": unambiguous}
    return action.astype(jnp.int32), next_memory, trace
