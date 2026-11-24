"""Grounded policy for the asymmetric ring kitchen.

The inner seat can reach every task-critical object, so it follows a fully
autonomous, observation-driven cooking loop. Global position is recovered
from the pot, which is visible from every cell in the inner room.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_inner_loop",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}


RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def init_agent_0():
    return jnp.zeros((2,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del obs, rng
    next_memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    return STAY, next_memory, {}


def init_agent_1():
    # phase: 0=query, 1=fill, 2=deliver, 3=direct finish,
    #        4=stage/wait, 5=finish with staged plate, 6=fallback finish
    return jnp.asarray([-1, 0, 0], dtype=jnp.int32)


def _navigate(row, col, target_row, target_col):
    """Exact shortest path on the eight-cell ring around the central pot."""
    index = jnp.where(
        row == 3,
        col - 3,
        jnp.where((row == 4) & (col == 5), 3, jnp.where(row == 5, 9 - col, 7)),
    )
    target_index = jnp.where(
        target_row == 3,
        target_col - 3,
        jnp.where(
            (target_row == 4) & (target_col == 5),
            3,
            jnp.where(target_row == 5, 9 - target_col, 7),
        ),
    )
    clockwise = jnp.where(index < 2, RIGHT, jnp.where(index < 4, DOWN, jnp.where(index < 6, LEFT, UP)))
    counterclockwise = jnp.where(
        index == 0,
        DOWN,
        jnp.where(index < 3, LEFT, jnp.where(index < 5, UP, jnp.where(index < 7, RIGHT, DOWN))),
    )
    clockwise_distance = (target_index - index) % 8
    step = jnp.where(clockwise_distance <= 4, clockwise, counterclockwise)
    return jnp.where(index == target_index, STAY, step)


def _go_and_use(row, col, target_row, target_col, face_action, is_facing):
    at_target = (row == target_row) & (col == target_col)
    use = jnp.where(is_facing, INTERACT, face_action)
    return jnp.where(at_target, use, _navigate(row, col, target_row, target_col))


def agent_1_act(obs, reset, memory, rng):
    del rng
    target = jnp.where(reset, jnp.int32(-1), memory[0])
    phase = jnp.where(reset, jnp.int32(0), memory[1])
    timestep = jnp.where(reset, jnp.int32(0), memory[2])

    pot_flat = jnp.argmax(obs[:, :, 22].reshape(-1))
    pot_local_row = pot_flat // 5
    pot_local_col = pot_flat % 5
    row = jnp.int32(6) - pot_local_row
    col = jnp.int32(6) - pot_local_col

    facing_up = obs[2, 2, 1] > 0
    facing_down = obs[2, 2, 2] > 0
    facing_right = obs[2, 2, 3] > 0
    facing_left = obs[2, 2, 4] > 0

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_0 = obs[2, 2, 7] > 0
    held_1 = obs[2, 2, 8] > 0
    hands_empty = ~(held_plate | held_cooked | held_0 | held_1 | (obs[2, 2, 9] > 0))

    # Observation-triggered phase transitions occur before action selection.
    # This lets the policy use a still-active button's newly resampled recipe
    # immediately after a delivery.
    finishing = (phase == 1) | (phase == 3) | (phase == 4) | (phase == 5) | (phase == 6)
    phase = jnp.where(finishing & held_cooked, jnp.int32(2), phase)
    delivered = (phase == 2) & (~held_cooked)
    phase = jnp.where(delivered, jnp.int32(0), phase)
    target = jnp.where(delivered, jnp.int32(-1), target)

    recipe_0 = jnp.sum(obs[:, :, 36])
    recipe_1 = jnp.sum(obs[:, :, 37])
    recipe_visible = (recipe_0 + recipe_1) > 0
    observed_target = jnp.where(recipe_1 > recipe_0, jnp.int32(1), jnp.int32(0))
    target = jnp.where((phase == 0) & recipe_visible, observed_target, target)
    phase = jnp.where((phase == 0) & recipe_visible, jnp.int32(1), phase)

    pot_r = pot_local_row
    pot_c = pot_local_col
    pot_count = obs[pot_r, pot_c, 31] + obs[pot_r, pot_c, 32] + obs[pot_r, pot_c, 33]
    pot_ready = obs[pot_r, pot_c, 30] > 0
    pot_timer = obs[pot_r, pot_c, 39]

    recipe_action = _go_and_use(row, col, 3, 4, UP, facing_up)

    use_bottom = target == 1
    supply_row = jnp.where(use_bottom, jnp.int32(5), jnp.int32(3))
    pot_side_row = supply_row
    plate_row = supply_row
    holding_target = jnp.where(use_bottom, held_1, held_0)

    pickup_action = _go_and_use(row, col, supply_row, 3, LEFT, facing_left)
    put_action = _go_and_use(
        row,
        col,
        pot_side_row,
        4,
        jnp.where(use_bottom, UP, DOWN),
        jnp.where(use_bottom, facing_up, facing_down),
    )
    plate_action = _go_and_use(row, col, plate_row, 5, RIGHT, facing_right)
    scoop_right_action = _go_and_use(row, col, 4, 5, LEFT, facing_left)

    filling_action = jnp.where(holding_target, put_action, pickup_action)
    outward_action = jnp.where(use_bottom, DOWN, UP)
    facing_outward = jnp.where(use_bottom, facing_down, facing_up)
    at_button = (row == pot_side_row) & (col == 4)

    counter_row = jnp.where(use_bottom, jnp.int32(6), jnp.int32(2))
    counter_local_row = jnp.clip(counter_row - row + 2, 0, 4)
    counter_local_col = jnp.clip(jnp.int32(7) - col, 0, 4)
    counter_plate = obs[counter_local_row, counter_local_col, 29] > 0
    counter_any = jnp.sum(obs[counter_local_row, counter_local_col, 29:34]) > 0

    button_use = jnp.where(facing_outward, INTERACT, outward_action)
    renewal_action = jnp.where(
        at_button,
        jnp.where(pot_timer <= 4, button_use, STAY),
        _navigate(row, col, pot_side_row, 4),
    )
    renewal_action = jnp.where(timestep > 392, STAY, renewal_action)
    activate_direct = (
        (phase == 1)
        & (pot_count >= 3)
        & hands_empty
        & counter_any
        & at_button
        & (pot_timer <= 4)
        & facing_outward
        & (timestep <= 392)
    )

    staging_action = jnp.where(
        held_plate,
        _go_and_use(row, col, plate_row, 5, outward_action, facing_outward),
        plate_action,
    )
    staged_renewal_action = jnp.where(
        at_button,
        jnp.where(
            pot_timer <= 6,
            button_use,
            jnp.where((pot_timer == 7) & (~facing_outward), outward_action, STAY),
        ),
        _navigate(row, col, pot_side_row, 4),
    )
    staged_renewal_action = jnp.where(timestep > 391, STAY, staged_renewal_action)
    stage_phase_action = jnp.where(counter_plate & hands_empty, staged_renewal_action, staging_action)
    staging_conflict = (phase == 4) & held_plate & counter_any
    stage_phase_action = jnp.where(
        staging_conflict,
        scoop_right_action,
        stage_phase_action,
    )
    activate_staged = (
        (phase == 4)
        & counter_plate
        & hands_empty
        & at_button
        & (pot_timer <= 6)
        & facing_outward
        & (timestep <= 391)
    )

    final_cycle = (phase == 1) & (pot_count >= 3) & (timestep > 350)
    direct_late_action = jnp.where(held_plate, scoop_right_action, plate_action)
    start_staging = (phase == 1) & (pot_count >= 3) & hands_empty & (~counter_any) & (~final_cycle)
    full_action = jnp.where(final_cycle, direct_late_action, jnp.where(start_staging, staging_action, renewal_action))
    cook_action = jnp.where(pot_count < 3, filling_action, full_action)

    finish_action = jnp.where(
        held_plate,
        scoop_right_action,
        plate_action,
    )
    retrieve_action = _go_and_use(row, col, plate_row, 5, outward_action, facing_outward)
    staged_finish_action = jnp.where(
        held_plate,
        scoop_right_action,
        jnp.where(counter_plate, retrieve_action, plate_action),
    )

    delivery_action = _go_and_use(row, col, 4, 5, RIGHT, facing_right)

    # The cutoff decides whether to *start* a post-delivery query. Once the
    # agent has left the goal, finish that query even as the clock advances.
    too_late_at_goal = (timestep > 356) & (row == 4) & (col == 5)
    recipe_action = jnp.where(too_late_at_goal, STAY, recipe_action)
    action = jnp.where(
        phase == 0,
        recipe_action,
        jnp.where(
            phase == 2,
            delivery_action,
            jnp.where(
                phase == 3,
                finish_action,
                jnp.where(phase == 4, stage_phase_action, jnp.where(phase == 5, staged_finish_action, jnp.where(phase == 6, finish_action, cook_action))),
            ),
        ),
    )
    next_phase = jnp.where(start_staging, jnp.int32(4), phase)
    next_phase = jnp.where(activate_direct, jnp.int32(3), next_phase)
    next_phase = jnp.where(activate_staged, jnp.int32(5), next_phase)
    next_phase = jnp.where(staging_conflict, jnp.int32(6), next_phase)
    next_phase = jnp.where(final_cycle, jnp.int32(6), next_phase)
    next_memory = jnp.stack([target, next_phase, timestep + 1]).astype(jnp.int32)
    trace = {
        "phase": phase,
        "target": target,
        "global_row": row,
        "global_col": col,
        "pot_count": pot_count,
        "pot_ready": pot_ready,
        "pot_timer": pot_timer,
        "counter_plate": counter_plate,
        "timestep": timestep,
    }
    return action.astype(jnp.int32), next_memory, trace
