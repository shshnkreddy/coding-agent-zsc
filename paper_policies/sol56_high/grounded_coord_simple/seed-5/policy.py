"""Grounded asymmetric controller for the two-room Overcooked layout.

The left player reads the free recipe display and briefly turns on the shared
button display. The right player reads that literal display and performs the
entire cook / plate / deliver cycle. No action encodes a private convention.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_signal_cook",
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


def _position_from_anchor(obs, channel, anchor_row, anchor_col):
    """Recover global self position from one always-visible static object."""
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    mask = obs[:, :, channel]
    local_row = jnp.sum(rows * mask)
    local_col = jnp.sum(cols * mask)
    return (jnp.int32(anchor_row + 2) - local_row,
            jnp.int32(anchor_col + 2) - local_col)


def _move_to(row, col, target_row, target_col):
    """Shortest unobstructed motion inside either 3x2 room."""
    vertical = jnp.where(row < target_row, DOWN, UP)
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    return jnp.where(row != target_row, vertical,
                     jnp.where(col != target_col, horizontal, STAY))


def _face_or_interact(obs, desired_action):
    # Facing channels: up=1, down=2, right=3, left=4.
    face_channel = jnp.where(
        desired_action == UP, 1,
        jnp.where(desired_action == DOWN, 2,
                  jnp.where(desired_action == RIGHT, 3, 4)))
    already_facing = obs[2, 2, face_channel] > 0
    return jnp.where(already_facing, INTERACT, desired_action)


def init_agent_0():
    # activated, saw_nonempty_pot, post-plating delay, age, step, last recipe
    return jnp.zeros((6,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    initial = init_agent_0()
    mem = jnp.where(reset, initial, memory)
    row, col = _position_from_anchor(obs, 23, 2, 0)  # always-on R

    activated = mem[0]
    saw_pot = mem[1]
    delay = mem[2]
    age = mem[3]
    episode_step = mem[4]
    last_signaled_recipe = mem[5]

    free_display = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    current_recipe = jnp.argmax(free_display).astype(jnp.int32)

    # Read the pot itself rather than other grid items used as marker tokens.
    pot_local_row = 4 - row
    pot_local_col = jnp.minimum(6 - col, 4)
    pot_amount = jnp.where(
        col == 2,
        jnp.sum(obs[pot_local_row, pot_local_col, 31:34]),
        0)
    pot_nonempty = pot_amount > 0
    newly_cleared = (saw_pot > 0) & (~pot_nonempty)
    saw_next = jnp.where(pot_nonempty, 1, saw_pot)
    saw_next = jnp.where(newly_cleared, 0, saw_next)
    # Once armed at one, keep the signal request pending through any travel or
    # facing action until a real interaction clears it below.
    counted_delay = jnp.where(delay > 1, delay - 1, delay)
    delay_next = jnp.where(newly_cleared, 2, counted_delay)

    at_home = (row == 2) & (col == 2)
    facing_right = obs[2, 2, 3] > 0
    initial_signal = activated == 0
    delivery_signal = (delay == 1) & (activated > 0)
    recipe_changed = ((activated > 0)
                      & (current_recipe != last_signaled_recipe))
    # A sparse retry lets a slow unfamiliar partner recover a missed display.
    retry_signal = (age >= 70) & (~pot_nonempty) & (delay == 0)
    useful_horizon = episode_step < 360
    should_signal = initial_signal | (useful_horizon & (
        delivery_signal | retry_signal | recipe_changed))

    travel = _move_to(row, col, jnp.int32(2), jnp.int32(2))
    signal_action = jnp.where(facing_right, INTERACT, RIGHT)
    signal_path = jnp.where(at_home, signal_action, travel)
    action = jnp.where(should_signal, signal_path,
                       jnp.where(at_home, STAY, travel))
    did_signal = at_home & should_signal & facing_right

    activated_next = jnp.where(did_signal, 1, activated)
    age_next = jnp.where(did_signal, 0, age + 1)
    last_recipe_next = jnp.where(
        did_signal, current_recipe, last_signaled_recipe)
    next_memory = jnp.stack([
        activated_next,
        saw_next,
        jnp.where(did_signal, 0, delay_next),
        age_next,
        episode_step + 1,
        last_recipe_next,
    ]).astype(jnp.int32)
    trace = {
        "phase": activated_next,
        "pot_seen": saw_next,
        "signal_age": age_next,
        "signal_now": did_signal.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # current, last, wait age, marker candidate/stability, blind-search flag
    return jnp.asarray([-1, -1, 0, -1, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    mem = jnp.where(reset, initial, memory)
    row, col = _position_from_anchor(obs, 22, 2, 4)  # pot

    displayed = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    display_present = jnp.sum(displayed) > 0
    display_recipe = jnp.argmax(displayed).astype(jnp.int32)
    waiting = mem[0] < 0
    wait_age = jnp.where(waiting & (~display_present), mem[2] + 1, 0)
    top_marker = (col == 5) & (obs[3 - row, 0, 33] > 0)
    bottom_marker = (col == 5) & (obs[5 - row, 0, 33] > 0)
    marker_unique = top_marker != bottom_marker
    marker_recipe = jnp.where(bottom_marker, 1, 0).astype(jnp.int32)
    same_marker = marker_unique & (marker_recipe == mem[3])
    marker_stability = jnp.where(
        marker_unique, jnp.where(same_marker, mem[4] + 1, 1), 0)
    marker_candidate = jnp.where(marker_unique, marker_recipe, -1)
    marker_present = marker_unique & (marker_stability >= 5)
    repeat_fallback = waiting & (~display_present) & (mem[1] >= 0) & (wait_age >= 10)
    initial_blind = (waiting & (~display_present) & (~marker_present)
                     & (mem[1] < 0) & (wait_age >= 15))
    continued_blind = (waiting & (~display_present) & (~marker_present)
                       & (mem[5] > 0) & (mem[1] >= 0) & (wait_age >= 6))
    blind_recipe = jnp.where(initial_blind, 0, 1 - mem[1])
    recipe = jnp.where(display_present, display_recipe,
                       jnp.where(marker_present, marker_recipe,
                                 jnp.where(initial_blind | continued_blind,
                                           blind_recipe,
                                           jnp.where(repeat_fallback,
                                                     mem[1], mem[0]))))

    inv = obs[2, 2, 5:10]
    holding_plate = inv[0] > 0
    holding_cooked = inv[1] > 0
    holding_raw = jnp.sum(inv[2:5]) > 0

    pot_local_row = 4 - row
    pot_local_col = 6 - col
    pot_counts = obs[pot_local_row, pot_local_col, 31:34]
    pot_amount = jnp.sum(pot_counts)
    pot_cooked = obs[pot_local_row, pot_local_col, 30] > 0
    pot_timer = obs[pot_local_row, pot_local_col, 39]
    pot_busy = (pot_timer > 0) | (pot_amount >= 3)

    # Actions for the fixed access cells.
    to_pot = _move_to(row, col, jnp.int32(2), jnp.int32(5))
    pot_action = jnp.where((row == 2) & (col == 5),
                           _face_or_interact(obs, LEFT), to_pot)

    to_goal = _move_to(row, col, jnp.int32(2), jnp.int32(6))
    goal_action = jnp.where((row == 2) & (col == 6),
                            _face_or_interact(obs, RIGHT), to_goal)

    # Use the plate pile on the same side as the requested ingredient.
    plate_row = jnp.where(recipe == 1, 3, 1).astype(jnp.int32)
    to_plate = _move_to(row, col, plate_row, jnp.int32(5))
    plate_action = jnp.where((row == plate_row) & (col == 5),
                             _face_or_interact(obs, LEFT), to_plate)

    ingredient_row = jnp.where(recipe == 1, 3, 1).astype(jnp.int32)
    to_ingredient = _move_to(row, col, ingredient_row, jnp.int32(6))
    ingredient_action = jnp.where(
        (row == ingredient_row) & (col == 6),
        _face_or_interact(obs, RIGHT), to_ingredient)

    to_display = _move_to(row, col, jnp.int32(2), jnp.int32(5))

    # Priority follows inventory, then visible pot state, then recipe state.
    action = jnp.where(
        holding_cooked, goal_action,
        jnp.where(
            holding_raw, pot_action,
            jnp.where(
                holding_plate, jnp.where(pot_cooked, pot_action, to_pot),
                jnp.where(
                    recipe < 0, to_display,
                    jnp.where(pot_cooked | pot_busy, plate_action,
                              ingredient_action)))))

    # The goal interaction consumes the dish. Forget the old request so a
    # stale private memory can never seed the next pot.
    delivering = (holding_cooked & (row == 2) & (col == 6)
                  & (obs[2, 2, 3] > 0))
    recipe_next = jnp.where(delivering, -1, recipe)
    last_next = jnp.where(delivering, recipe, mem[1])
    wait_next = jnp.where(delivering | (recipe >= 0), 0, wait_age)
    candidate_next = jnp.where(delivering, -1, marker_candidate)
    stability_next = jnp.where(delivering, 0, marker_stability)
    blind_next = jnp.where(
        display_present | marker_present, 0,
        jnp.where(initial_blind, 1, mem[5]))
    next_memory = jnp.stack([
        recipe_next, last_next, wait_next, candidate_next, stability_next,
        blind_next
    ]).astype(jnp.int32)
    trace = {
        "recipe": recipe_next,
        "pot_amount": pot_amount.astype(jnp.int32),
        "pot_timer": pot_timer.astype(jnp.int32),
        "display": display_present.astype(jnp.int32),
        "marker": marker_present.astype(jnp.int32),
        "blind": blind_next.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
