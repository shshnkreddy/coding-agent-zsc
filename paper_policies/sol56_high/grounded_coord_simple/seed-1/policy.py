"""Grounded cooperative policy for the two-room Overcooked layout."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_recipe_relay_v15",
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


def _location(mask):
    """Return the local displacement from self to a visible singleton object."""
    index = jnp.argmax(mask.reshape((-1,))).astype(jnp.int32)
    return index // 5 - 2, index % 5 - 2


def _navigate_to_object(obs, mask, stand_col_offset, facing_channel, face_action,
                        interact_ready):
    """Navigate in an open 3x2 room to a fixed interaction position."""
    obj_dr, obj_dc = _location(mask)
    dr = obj_dr
    dc = obj_dc + jnp.asarray(stand_col_offset, dtype=jnp.int32)
    vertical = jnp.where(dr < 0, UP, DOWN)
    horizontal = jnp.where(dc < 0, LEFT, RIGHT)
    move = jnp.where(dr != 0, vertical, horizontal)
    at_stand = (dr == 0) & (dc == 0)
    facing = obs[2, 2, facing_channel] > 0
    stopped = jnp.where(interact_ready, INTERACT, STAY)
    at_action = jnp.where(facing, stopped, face_action)
    return jnp.where(at_stand, at_action, move)


def _navigate_vertical_object(obs, mask):
    """Interact with a top/bottom counter object from the open room."""
    obj_dr, obj_dc = _location(mask)
    above = obj_dr < 0
    dr = obj_dr + jnp.where(above, 1, -1)
    dc = obj_dc
    vertical = jnp.where(dr < 0, UP, DOWN)
    horizontal = jnp.where(dc < 0, LEFT, RIGHT)
    move = jnp.where(dc != 0, horizontal, vertical)
    at_stand = (dr == 0) & (dc == 0)
    facing = jnp.where(above, obs[2, 2, 1] > 0, obs[2, 2, 2] > 0)
    face_action = jnp.where(above, UP, DOWN)
    at_action = jnp.where(facing, INTERACT, face_action)
    return jnp.where(at_stand, at_action, move)


def _recipe_from_display(obs):
    count_0 = jnp.sum(obs[:, :, 36])
    count_1 = jnp.sum(obs[:, :, 37])
    visible = (count_0 + count_1) > 0
    recipe = jnp.where(count_1 > count_0, 1, 0).astype(jnp.int32)
    return visible, recipe


def init_agent_0():
    # known recipe, signal pending, previous pot occupied, ack, time, stall
    return jnp.asarray([-1, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    initial = init_agent_0()
    memory = jnp.where(reset, initial, memory)
    known, pending_old, previous_pot, old_ack, timestep, old_stall = memory

    _, recipe = _recipe_from_display(obs)  # the free R display is always visible
    pot_occupied = (jnp.sum(obs[:, :, 29:34]) > 0).astype(jnp.int32)
    pot_cleared = (previous_pot > 0) & (pot_occupied == 0)
    ack = jnp.where(pot_occupied > 0, 0, jnp.maximum(old_ack - 1, 0))
    retry = (old_ack == 1) & (pot_occupied == 0)
    stall = jnp.where(
        pot_occupied > 0,
        0,
        jnp.where(pot_cleared, 11, jnp.maximum(old_stall - 1, 0)),
    )
    stalled = (old_stall == 1) & (pot_occupied == 0)
    changed = recipe != known
    useful_horizon = timestep < 360
    pending = (pending_old > 0) | ((changed | retry | stalled) & useful_horizon)

    button = obs[:, :, 24]
    action = _navigate_to_object(
        obs, button, stand_col_offset=-1, facing_channel=3,
        face_action=RIGHT, interact_ready=pending,
    )
    did_signal = pending & (action == INTERACT)
    next_known = jnp.where(did_signal, recipe, known)
    next_pending = jnp.where(did_signal, 0, pending.astype(jnp.int32))
    next_ack = jnp.where(did_signal, 10, ack)
    next_stall = jnp.where(did_signal, 0, stall)
    next_memory = jnp.asarray(
        [next_known, next_pending, pot_occupied, next_ack, timestep + 1,
         next_stall],
        dtype=jnp.int32,
    )
    trace = {
        "recipe": recipe,
        "pending": pending.astype(jnp.int32),
        "pot_occupied": pot_occupied,
        "ack": ack,
        "timestep": timestep,
        "stall": stall,
    }
    return action, next_memory, trace


def init_agent_1():
    # remembered recipe (-1 means unknown), awaiting post-delivery display
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    memory = jnp.where(reset, initial, memory)
    old_recipe, old_awaiting = memory

    display_visible, displayed_recipe = _recipe_from_display(obs)
    recipe = jnp.where(display_visible, displayed_recipe, old_recipe)
    awaiting = jnp.where(
        display_visible, 0, jnp.maximum(old_awaiting - 1, 0)
    ).astype(jnp.int32)

    plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_0, held_1, held_2 = obs[2, 2, 7:10]
    held_raw = (held_0 + held_1 + held_2) > 0
    raw_matches = ((recipe == 0) & (held_0 > 0)) | (
        (recipe == 1) & (held_1 > 0)
    )
    dish = plate & held_cooked
    dish_matches = (
        ((recipe == 0) & (held_0 == 3) & (held_1 == 0) & (held_2 == 0))
        | ((recipe == 1) & (held_0 == 0) & (held_1 == 3) & (held_2 == 0))
    )
    empty_plate = plate & (~held_cooked)

    pot_mask = obs[:, :, 22]
    pot_cooked = jnp.sum(obs[:, :, 30] * pot_mask) > 0
    pot_timer = jnp.sum(obs[:, :, 39] * pot_mask)
    pot_item_total = jnp.sum(obs[:, :, 29:34] * pot_mask[:, :, None])

    ingredient_0 = obs[:, :, 26]
    ingredient_1 = obs[:, :, 27]
    ingredient_mask = jnp.where(recipe == 1, ingredient_1, ingredient_0)
    plate_mask = obs[:, :, 25]
    goal_mask = obs[:, :, 21]

    item_total = jnp.sum(obs[:, :, 29:34], axis=2)
    current_column = jnp.arange(5) == 2
    empty_counter = obs[:, :, 20] * (item_total == 0) * current_column[None, :]
    empty_counter_visible = jnp.sum(empty_counter) > 0
    cached_0 = obs[:, :, 20] * (obs[:, :, 31] > 0)
    cached_1 = obs[:, :, 20] * (obs[:, :, 32] > 0)
    cached_ingredient = jnp.where(recipe == 1, cached_1, cached_0)
    cache_visible = jnp.sum(cached_ingredient) > 0
    grid_match_0 = ((obs[:, :, 31] == 3) & (obs[:, :, 32] == 0)
                    & (obs[:, :, 33] == 0))
    grid_match_1 = ((obs[:, :, 31] == 0) & (obs[:, :, 32] == 3)
                    & (obs[:, :, 33] == 0))
    grid_match = jnp.where(recipe == 1, grid_match_1, grid_match_0)
    cached_dish = (obs[:, :, 20] * (obs[:, :, 29] > 0)
                   * (obs[:, :, 30] > 0) * grid_match)
    cached_dish_visible = jnp.sum(cached_dish) > 0

    go_pot_with_raw = _navigate_to_object(
        obs, pot_mask, stand_col_offset=1, facing_channel=4,
        face_action=LEFT, interact_ready=jnp.asarray(True),
    )
    go_pot_with_plate = _navigate_to_object(
        obs, pot_mask, stand_col_offset=1, facing_channel=4,
        face_action=LEFT, interact_ready=pot_cooked,
    )
    wait_for_recipe = _navigate_to_object(
        obs, pot_mask, stand_col_offset=1, facing_channel=4,
        face_action=LEFT, interact_ready=jnp.asarray(False),
    )
    get_ingredient = _navigate_to_object(
        obs, ingredient_mask, stand_col_offset=-1, facing_channel=3,
        face_action=RIGHT, interact_ready=jnp.asarray(True),
    )
    get_plate = _navigate_to_object(
        obs, plate_mask, stand_col_offset=1, facing_channel=4,
        face_action=LEFT, interact_ready=jnp.asarray(True),
    )
    deliver = _navigate_to_object(
        obs, goal_mask, stand_col_offset=-1, facing_channel=3,
        face_action=RIGHT, interact_ready=jnp.asarray(True),
    )
    quarantine_raw = _navigate_vertical_object(obs, empty_counter)
    get_cached = _navigate_vertical_object(obs, cached_ingredient)
    quarantine_dish = _navigate_vertical_object(obs, empty_counter)
    get_cached_dish = _navigate_vertical_object(obs, cached_dish)

    # Inventory observations make insertion, scooping, and delivery reactive.
    fetch_raw = jnp.where(cache_visible, get_cached, get_ingredient)
    empty_action = jnp.where((pot_timer > 0) | pot_cooked, get_plate, fetch_raw)
    empty_action = jnp.where(
        cached_dish_visible & (pot_item_total == 0), get_cached_dish, empty_action
    )
    has_recipe = recipe >= 0
    empty_action = jnp.where(has_recipe & (awaiting == 0), empty_action, wait_for_recipe)
    stale_raw_action = jnp.where(
        empty_counter_visible, quarantine_raw, go_pot_with_raw
    )
    wrong_dish_action = jnp.where(
        empty_counter_visible, quarantine_dish, deliver
    )
    action = jnp.where(
        dish,
        jnp.where(dish_matches, deliver, wrong_dish_action),
        jnp.where(empty_plate, go_pot_with_plate,
                  jnp.where(held_raw,
                            jnp.where(raw_matches, go_pot_with_raw,
                                      stale_raw_action),
                            empty_action)),
    )

    submitted = dish & dish_matches & (action == INTERACT)
    next_awaiting = jnp.where(submitted, 4, awaiting).astype(jnp.int32)
    next_memory = jnp.asarray([recipe, next_awaiting], dtype=jnp.int32)
    trace = {
        "recipe": recipe,
        "awaiting": awaiting.astype(jnp.int32),
        "inventory_kind": jnp.where(dish, 3, jnp.where(empty_plate, 2,
                                      jnp.where(held_raw, 1, 0))).astype(jnp.int32),
        "pot_timer": pot_timer.astype(jnp.int32),
        "raw_matches": raw_matches.astype(jnp.int32),
        "dish_matches": dish_matches.astype(jnp.int32),
    }
    return action, next_memory, trace
