"""Map-grounded cooperative policy for the split Overcooked kitchen."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_canonical_adaptive_v7",
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


def _position_from_landmark(obs, channel, landmark_row, landmark_col):
    """Recover global position from a unique static landmark in the 5x5 crop."""
    idx = jnp.argmax(obs[:, :, channel].reshape((-1,)))
    local_row = idx // jnp.int32(5)
    local_col = idx % jnp.int32(5)
    row = jnp.int32(landmark_row + 2) - local_row
    col = jnp.int32(landmark_col + 2) - local_col
    return row, col


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _navigate(row, col, target_row, target_col):
    """Deterministically move to a floor target (vertical, then horizontal)."""
    vertical = jnp.where(row < target_row, DOWN, UP)
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    return jnp.where(row != target_row, vertical,
                     jnp.where(col != target_col, horizontal, STAY))


def _go_and_use(obs, row, col, target_row, target_col, desired_facing,
                face_action):
    """Reach an interaction square, turn toward its object, then interact."""
    at_target = (row == target_row) & (col == target_col)
    use_action = jnp.where(_facing(obs) == desired_facing, INTERACT,
                           face_action)
    return jnp.where(at_target, use_action,
                     _navigate(row, col, target_row, target_col))


def init_agent_0():
    return {
        "signaled": jnp.int32(0),
        "signal_age": jnp.int32(0),
        "prev_pot_total": jnp.int32(0),
        "steps": jnp.int32(0),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    z = jnp.int32(0)
    signaled = jnp.where(reset, z, memory["signaled"])
    signal_age = jnp.where(reset, z, memory["signal_age"])
    prev_pot = jnp.where(reset, z, memory["prev_pot_total"])
    steps = jnp.where(reset, z, memory["steps"])

    # The always-on indicator at (2,0) is visible everywhere in the left room.
    row, col = _position_from_landmark(obs, 23, 2, 0)

    # The pot at (2,4) is visible from the signaling square (2,2).
    pot_lr = jnp.clip(jnp.int32(4) - row, 0, 4)
    pot_lc = jnp.clip(jnp.int32(6) - col, 0, 4)
    pot_visible = (jnp.abs(row - jnp.int32(2)) <= 2) & (
        jnp.abs(col - jnp.int32(4)) <= 2)
    pot_total_seen = (obs[pot_lr, pot_lc, 31] +
                      obs[pot_lr, pot_lc, 32] +
                      obs[pot_lr, pot_lc, 33])
    pot_total = jnp.where(pot_visible, pot_total_seen, prev_pot)

    # Clearing the pot means a plated dish is about to be delivered. Pressing
    # now leaves the button active when the freshly sampled recipe appears.
    cleared = (prev_pot > 0) & (pot_total == 0)
    enough_horizon = steps < jnp.int32(358)
    retry = (signal_age >= jnp.int32(10)) & (pot_total == 0)
    need_signal = (signaled == 0) | (enough_horizon & (cleared | retry))
    signal_action = _go_and_use(
        obs, row, col, jnp.int32(2), jnp.int32(2), jnp.int32(2), RIGHT)
    action = jnp.where(need_signal, signal_action, STAY)
    pressed = need_signal & (row == 2) & (col == 2) & (
        _facing(obs) == jnp.int32(2))

    next_memory = {
        "signaled": jnp.where(pressed, jnp.int32(1), signaled),
        "signal_age": jnp.where(pressed, z, signal_age + jnp.int32(1)),
        "prev_pot_total": pot_total.astype(jnp.int32),
        "steps": steps + jnp.int32(1),
    }
    trace = {
        "row": row,
        "col": col,
        "pot_total": pot_total.astype(jnp.int32),
        "signal": need_signal.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    return {
        "recipe": jnp.int32(-1),
        "last_recipe": jnp.int32(-1),
        "unknown_age": jnp.int32(0),
        "steps": jnp.int32(0),
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    recipe = jnp.where(reset, jnp.int32(-1), memory["recipe"])
    last_recipe = jnp.where(reset, jnp.int32(-1), memory["last_recipe"])
    unknown_age = jnp.where(reset, jnp.int32(0), memory["unknown_age"])
    steps = jnp.where(reset, jnp.int32(0), memory["steps"])

    # The delivery station at (2,7) is visible everywhere in the right room.
    row, col = _position_from_landmark(obs, 21, 2, 7)

    # Read the button at (2,3) from column 5. Clipping keeps indexing static
    # and the explicit visibility guard rejects the clipped value at column 6.
    btn_lr = jnp.clip(jnp.int32(4) - row, 0, 4)
    btn_lc = jnp.clip(jnp.int32(5) - col, 0, 4)
    button_visible = (jnp.abs(row - 2) <= 2) & (jnp.abs(col - 3) <= 2)
    shown_0 = obs[btn_lr, btn_lc, 36]
    shown_1 = obs[btn_lr, btn_lc, 37]
    valid_shown = button_visible & ((shown_0 == 3) | (shown_1 == 3))
    shown_recipe = jnp.where(shown_0 == 3, jnp.int32(0), jnp.int32(1))
    # A persistent ingredient-2 marker on the upper/lower central counter is
    # another grounded signal. Immediately after delivery, accept only a
    # changed marker; an unchanged marker may simply be stale.
    marker_top_lr = jnp.clip(jnp.int32(3) - row, 0, 4)
    marker_bottom_lr = jnp.clip(jnp.int32(5) - row, 0, 4)
    marker_lc = jnp.clip(jnp.int32(5) - col, 0, 4)
    marker_visible = col == 5
    marker_top = marker_visible & (obs[marker_top_lr, marker_lc, 33] > 0)
    marker_bottom = marker_visible & (obs[marker_bottom_lr, marker_lc, 33] > 0)
    valid_marker = marker_top ^ marker_bottom
    marker_recipe = jnp.where(marker_top, jnp.int32(0), jnp.int32(1))
    accept_marker = (recipe < 0) & valid_marker & ((last_recipe < 0) |
                                                   (marker_recipe != last_recipe))
    communicated = valid_shown | accept_marker
    communicated_recipe = jnp.where(valid_shown, shown_recipe, marker_recipe)
    was_unknown = recipe < 0
    unknown_age = jnp.where(
        communicated,
        jnp.int32(0),
        jnp.where(was_unknown, unknown_age + jnp.int32(1), jnp.int32(0)),
    )
    recipe = jnp.where(communicated, communicated_recipe, recipe)
    last_recipe = jnp.where(communicated, communicated_recipe, last_recipe)
    silent_repeat = (recipe < 0) & (last_recipe >= 0) & (
        unknown_age >= jnp.int32(12))
    recipe = jnp.where(silent_repeat, last_recipe, recipe)

    # Pot is always visible in this six-cell room.
    pot_lr = jnp.clip(jnp.int32(4) - row, 0, 4)
    pot_lc = jnp.clip(jnp.int32(6) - col, 0, 4)
    pot_0 = obs[pot_lr, pot_lc, 31]
    pot_1 = obs[pot_lr, pot_lc, 32]
    pot_total = pot_0 + pot_1 + obs[pot_lr, pot_lc, 33]
    pot_cooked = obs[pot_lr, pot_lc, 30] > 0
    pot_timer = obs[pot_lr, pot_lc, 39]

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_0 = obs[2, 2, 7]
    held_1 = obs[2, 2, 8]
    held_raw = (held_0 + held_1 + obs[2, 2, 9]) > 0
    held_matches = ((recipe == 0) & (held_0 > 0)) | (
        (recipe == 1) & (held_1 > 0))

    wait_for_recipe = _navigate(row, col, jnp.int32(2), jnp.int32(5))

    ingredient_row = jnp.where(recipe == 0, jnp.int32(1), jnp.int32(3))
    fetch_ingredient = _go_and_use(
        obs, row, col, ingredient_row, jnp.int32(6), jnp.int32(2), RIGHT)
    add_to_pot = _go_and_use(
        obs, row, col, jnp.int32(2), jnp.int32(5), jnp.int32(3), LEFT)
    fetch_plate = _go_and_use(
        obs, row, col, jnp.int32(1), jnp.int32(5), jnp.int32(3), LEFT)
    deliver = _go_and_use(
        obs, row, col, jnp.int32(2), jnp.int32(6), jnp.int32(2), RIGHT)

    # If a delayed public signal corrects a silence-based guess while a raw
    # ingredient is held, keep it out of the pot. Prefer the outer counter in
    # column 6, falling back to column 5 if that counter is already occupied.
    discard_floor_row = jnp.where(held_0 > 0, jnp.int32(1), jnp.int32(3))
    discard_counter_row = jnp.where(held_0 > 0, jnp.int32(0), jnp.int32(4))
    counter6_lr = jnp.clip(discard_counter_row - row + jnp.int32(2), 0, 4)
    counter6_lc = jnp.clip(jnp.int32(8) - col, 0, 4)
    counter6_occupied = jnp.sum(obs[counter6_lr, counter6_lc, 29:34]) > 0
    discard_floor_col = jnp.where(counter6_occupied,
                                  jnp.int32(5), jnp.int32(6))
    discard_facing = jnp.where(held_0 > 0, jnp.int32(0), jnp.int32(1))
    discard_turn = jnp.where(held_0 > 0, UP, DOWN)
    discard_raw = _go_and_use(
        obs, row, col, discard_floor_row, discard_floor_col,
        discard_facing, discard_turn)

    empty_action = jnp.where(pot_total < 3, fetch_ingredient, fetch_plate)
    at_pot_square = (row == 2) & (col == 5)
    wait_facing_pot = jnp.where(
        at_pot_square,
        jnp.where(_facing(obs) == jnp.int32(3), STAY, LEFT),
        _navigate(row, col, jnp.int32(2), jnp.int32(5)),
    )
    plate_action = jnp.where(pot_cooked, add_to_pot, wait_facing_pot)
    known_action = jnp.where(
        held_cooked,
        deliver,
        jnp.where(held_raw, jnp.where(held_matches, add_to_pot, discard_raw),
                  jnp.where(held_plate, plate_action, empty_action)),
    )
    action = jnp.where(recipe < 0, wait_for_recipe, known_action)

    # Forget the old target after issuing the guaranteed goal interaction.
    delivering_now = held_cooked & (row == 2) & (col == 6) & (
        _facing(obs) == jnp.int32(2))
    next_recipe = jnp.where(delivering_now, jnp.int32(-1), recipe)
    next_memory = {
        "recipe": next_recipe.astype(jnp.int32),
        "last_recipe": jnp.where(delivering_now, recipe,
                                 last_recipe).astype(jnp.int32),
        "unknown_age": jnp.where(delivering_now, jnp.int32(0),
                                 unknown_age).astype(jnp.int32),
        "steps": steps + jnp.int32(1),
    }
    trace = {
        "row": row,
        "col": col,
        "recipe": recipe,
        "unknown_age": unknown_age,
        "pot_total": pot_total.astype(jnp.int32),
        "pot_timer": pot_timer.astype(jnp.int32),
        "inventory": (held_raw.astype(jnp.int32) +
                      2 * held_plate.astype(jnp.int32) +
                      4 * held_cooked.astype(jnp.int32)),
    }
    return action.astype(jnp.int32), next_memory, trace
