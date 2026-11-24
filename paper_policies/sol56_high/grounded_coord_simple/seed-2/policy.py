"""Convention-robust policy for the split grounded_coord_simple kitchen.

The left player exposes the recipe through the map's recipe button. The right
player reacts to that public display and to the visible pot state; no private
action convention is needed.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "public_button_roles_v6",
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


def _position_from_object(obs, channel, object_row, object_col):
    """Recover global self position from one always-visible static object."""
    flat_index = jnp.argmax(obs[:, :, channel])
    local_row = flat_index // 5
    local_col = flat_index % 5
    row = jnp.asarray(object_row + 2, jnp.int32) - local_row
    col = jnp.asarray(object_col + 2, jnp.int32) - local_col
    return row, col


def _move_to(row, col, target_row, target_col):
    """Deterministic row-first navigation on either two-column room."""
    action = STAY
    action = jnp.where(col > target_col, LEFT, action)
    action = jnp.where(col < target_col, RIGHT, action)
    action = jnp.where(row > target_row, UP, action)
    action = jnp.where(row < target_row, DOWN, action)
    return action


def _at(row, col, target_row, target_col):
    return (row == target_row) & (col == target_col)


def init_agent_0():
    # [initial sent, pot armed, step, pending, awaiting progress, wait, retried]
    return jnp.zeros((7,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    row, col = _position_from_object(obs, 24, 2, 3)  # recipe button L
    home = _at(row, col, 2, 2)

    pot_amount = jnp.sum(obs[:, :, 31:34])
    pot_cooked = jnp.max(obs[:, :, 30])
    pot_nonempty = (pot_amount > 0) | (pot_cooked > 0)

    initial_sent = memory[0] > 0
    armed = (memory[1] > 0) | (home & pot_nonempty)
    productive_horizon = memory[2] < 358
    pending = memory[3] > 0
    awaiting_progress = (memory[4] > 0) & (~pot_nonempty)
    progress_wait = jnp.where(
        awaiting_progress, memory[5] + 1, jnp.asarray(0, jnp.int32)
    )
    retried = memory[6] > 0
    retry_due = (
        home
        & awaiting_progress
        & (progress_wait >= 12)
        & (~retried)
        & productive_horizon
    )
    pot_clear_event = armed & (~pot_nonempty) & productive_horizon
    needs_signal = home & ((~initial_sent) | pending | retry_due)

    action = _move_to(row, col, 2, 2)
    facing_right = obs[2, 2, 3] > 0
    signal_action = jnp.where(facing_right, INTERACT, RIGHT)
    action = jnp.where(needs_signal, signal_action, action)

    pressed = needs_signal & facing_right
    next_initial = initial_sent | pressed
    next_armed = jnp.where(pressed, False, armed)
    next_pending = jnp.where(pressed, False, pending | pot_clear_event)
    next_awaiting = jnp.where(
        pot_nonempty, False, jnp.where(pressed, True, awaiting_progress)
    )
    next_progress_wait = jnp.where(
        pot_nonempty | pressed, 0, progress_wait
    )
    next_retried = jnp.where(pressed, retry_due, retried)
    next_memory = jnp.asarray(
        [
            next_initial,
            next_armed,
            memory[2] + 1,
            next_pending,
            next_awaiting,
            next_progress_wait,
            next_retried,
        ],
        dtype=jnp.int32,
    )
    trace = {
        "global_position": jnp.asarray([row, col], dtype=jnp.int32),
        "pot_amount": pot_amount.astype(jnp.int32),
        "signal_needed": needs_signal,
        "step": memory[2],
        "pending": pending,
        "progress_wait": progress_wait,
        "retry_due": retry_due,
    }
    return action, next_memory, trace


def init_agent_1():
    # [latched target, had dish, last delivered target, no-display wait]
    return jnp.asarray([-1, 0, -1, 0], dtype=jnp.int32)


def _interact_at(row, col, target_row, target_col, facing, face_action):
    at_target = _at(row, col, target_row, target_col)
    return jnp.where(
        at_target,
        jnp.where(facing, INTERACT, face_action),
        _move_to(row, col, target_row, target_col),
    )


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    row, col = _position_from_object(obs, 22, 2, 4)  # central pot

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:10]) > 0
    held_dish = held_plate & held_cooked
    empty_plate = held_plate & (~held_cooked)

    # A held dish disappearing is locally observable evidence of a delivery.
    delivered = (memory[1] > 0) & (~held_dish)
    previous_target = memory[0]
    last_target = jnp.where(delivered, previous_target, memory[2])
    target = jnp.where(delivered, -1, previous_target)
    no_display_steps = jnp.where(delivered, 0, memory[3])

    displayed_counts = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    display_visible = jnp.sum(displayed_counts) > 0
    displayed_target = jnp.argmax(displayed_counts).astype(jnp.int32)
    waiting_without_display = (target < 0) & (last_target >= 0) & (~display_visible)
    no_display_steps = jnp.where(
        waiting_without_display, no_display_steps + 1, 0
    )
    fallback = waiting_without_display & (no_display_steps >= 12)
    target = jnp.where(fallback, last_target, target)
    target = jnp.where(display_visible, displayed_target, target)
    no_display_steps = jnp.where(target >= 0, 0, no_display_steps)

    pot_counts = jnp.sum(obs[:, :, 31:34], axis=(0, 1))
    pot_amount = jnp.sum(pot_counts)
    pot_cooked = jnp.max(obs[:, :, 30]) > 0

    face_left = obs[2, 2, 4] > 0
    face_right = obs[2, 2, 3] > 0

    # Empty hands: wait for the public display, fetch the requested ingredient,
    # or use cooking time to fetch a plate.
    wait_action = _move_to(row, col, 2, 5)
    need_ingredient = (target >= 0) & (pot_amount < 3)
    supply_row = jnp.where(target == 0, 1, 3)
    ingredient_action = _interact_at(
        row, col, supply_row, jnp.asarray(6), face_right, RIGHT
    )
    plate_action = _interact_at(row, col, 1, 5, face_left, LEFT)
    empty_action = jnp.where(
        target < 0,
        wait_action,
        jnp.where(need_ingredient, ingredient_action, plate_action),
    )

    # Held items have unambiguous object-grounded destinations.
    raw_action = _interact_at(row, col, 2, 5, face_left, LEFT)
    plate_to_pot = _interact_at(row, col, 2, 5, face_left, LEFT)
    waiting_at_pot = jnp.where(face_left, STAY, LEFT)
    plate_wait_action = jnp.where(
        _at(row, col, 2, 5), waiting_at_pot, _move_to(row, col, 2, 5)
    )
    plate_action_ready = jnp.where(
        pot_cooked, plate_to_pot, plate_wait_action
    )
    dish_action = _interact_at(row, col, 2, 6, face_right, RIGHT)

    action = empty_action
    action = jnp.where(held_raw, raw_action, action)
    action = jnp.where(empty_plate, plate_action_ready, action)
    action = jnp.where(held_dish, dish_action, action)

    next_memory = jnp.asarray(
        [target, held_dish, last_target, no_display_steps], dtype=jnp.int32
    )
    trace = {
        "global_position": jnp.asarray([row, col], dtype=jnp.int32),
        "target": target,
        "pot_amount": pot_amount.astype(jnp.int32),
        "pot_cooked": pot_cooked,
        "display_visible": display_visible,
        "no_display_steps": no_display_steps,
        "fallback": fallback,
    }
    return action, next_memory, trace
