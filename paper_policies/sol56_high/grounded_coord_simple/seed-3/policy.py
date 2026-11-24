"""Grounded controller for the two-room Overcooked layout.

The left player publishes the freely visible recipe using the environment's
recipe button. The right player treats that display as the source of truth
and runs a reactive cooking loop. No action sequence is used as a private
message.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_button_cook_v6",
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


def _object_relative_position(obs, channel, global_row, global_col):
    """Infer self global position from one known, always-visible object."""
    flat = jnp.argmax(obs[:, :, channel])
    local_row = flat // 5
    local_col = flat % 5
    row = jnp.asarray(global_row + 2, jnp.int32) - local_row
    col = jnp.asarray(global_col + 2, jnp.int32) - local_col
    return row, col


def _go_interact(obs, row, col, target_row, target_col, face_action):
    """Move on the six-cell room floor, then face and interact with an object."""
    at_target = (row == target_row) & (col == target_col)
    face_channel = jnp.where(
        face_action == RIGHT,
        3,
        jnp.where(face_action == DOWN, 2, jnp.where(face_action == LEFT, 4, 1)),
    )
    facing = obs[2, 2, face_channel] > 0
    at_action = jnp.where(facing, INTERACT, face_action)
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    vertical = jnp.where(row < target_row, DOWN, UP)
    travel = jnp.where(row != target_row, vertical, horizontal)
    return jnp.where(at_target, at_action, travel)


def _global_item(obs, row, col, global_row, global_col):
    """Return a known cell's visible item block, safely handling crop edges."""
    local_row = jnp.asarray(global_row + 2, jnp.int32) - row
    local_col = jnp.asarray(global_col + 2, jnp.int32) - col
    visible = (local_row >= 0) & (local_row < 5) & (local_col >= 0) & (local_col < 5)
    safe_row = jnp.clip(local_row, 0, 4)
    safe_col = jnp.clip(local_col, 0, 4)
    return obs[safe_row, safe_col, 29:34] * visible.astype(jnp.int32)


def init_agent_0():
    # last recipe published, unacknowledged-display age, elapsed steps
    return jnp.asarray([-1, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, init_agent_0(), memory)

    recipe_mask = obs[:, :, 23:24]
    recipe_counts = jnp.sum(obs[:, :, 36:39] * recipe_mask, axis=(0, 1))
    recipe_id = jnp.argmax(recipe_counts).astype(jnp.int32)

    pot_mask = obs[:, :, 22]
    pot_cooked = jnp.sum(obs[:, :, 30] * pot_mask) > 0
    pot_ingredients = jnp.sum(obs[:, :, 31:34] * pot_mask[:, :, None], axis=(0, 1))
    pot_count = jnp.sum(pot_ingredients)
    matching_evidence = pot_ingredients[recipe_id] > 0
    new_recipe = recipe_id != base[0]
    waiting_for_evidence = (base[1] > 0) & ~matching_evidence
    need_refresh = waiting_for_evidence & (base[1] >= 8) & (pot_count < 3)
    need_publish = new_recipe | need_refresh

    # R at (2,0) is always visible throughout the left room.
    row, col = _object_relative_position(obs, 23, 2, 0)
    publish_action = _go_interact(obs, row, col, 2, 2, RIGHT)
    action = jnp.where(need_publish, publish_action, STAY)
    successful_press = need_publish & (row == 2) & (col == 2) & (obs[2, 2, 3] > 0)

    next_last = jnp.where(successful_press, recipe_id, base[0])
    next_age = jnp.where(waiting_for_evidence, base[1] + 1, 0)
    next_age = jnp.where(successful_press, 1, next_age)
    next_memory = jnp.asarray([next_last, next_age, base[2] + 1], dtype=jnp.int32)
    trace = {
        "recipe": recipe_id,
        "publish": need_publish.astype(jnp.int32),
        "pot_ready": pot_cooked.astype(jnp.int32),
        "awaiting": waiting_for_evidence.astype(jnp.int32),
    }
    return action, next_memory, trace


def init_agent_1():
    # stored recipe id, post-delivery return-to-button flag, elapsed steps
    return jnp.asarray([-1, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, init_agent_1(), memory)

    # The canonical button is the sole recipe source available in this room.
    button_mask = obs[:, :, 24:25]
    displayed = jnp.sum(obs[:, :, 36:39] * button_mask, axis=(0, 1))
    display_active = jnp.sum(displayed) > 0
    displayed_id = jnp.argmax(displayed).astype(jnp.int32)
    recipe_id = jnp.where(display_active, displayed_id, base[0])

    # P at (2,4) is visible from every reachable cell in the right room.
    row, col = _object_relative_position(obs, 22, 2, 4)
    center = obs[2, 2]
    held_plate = center[5] > 0
    held_cooked = center[6] > 0
    held_raw = jnp.sum(center[7:10]) > 0
    held_raw_id = jnp.argmax(center[7:10]).astype(jnp.int32)

    pot_mask = obs[:, :, 22]
    pot_count = jnp.sum(obs[:, :, 31:34] * pot_mask[:, :, None])
    pot_cooked = jnp.sum(obs[:, :, 30] * pot_mask) > 0
    pot_timer = jnp.sum(obs[:, :, 39] * pot_mask)
    pot_busy = (pot_count >= 3) | pot_cooked | (pot_timer > 0)

    to_pot = _go_interact(obs, row, col, 2, 5, LEFT)
    to_delivery = _go_interact(obs, row, col, 2, 6, RIGHT)
    to_ing0 = _go_interact(obs, row, col, 1, 6, RIGHT)
    to_ing1 = _go_interact(obs, row, col, 3, 6, RIGHT)
    to_plate_top = _go_interact(obs, row, col, 1, 5, LEFT)
    to_plate_bottom = _go_interact(obs, row, col, 3, 5, LEFT)
    to_plate = jnp.where(row <= 2, to_plate_top, to_plate_bottom)

    # Four ordinary wall counters provide safe, observable recovery storage.
    item_t5 = _global_item(obs, row, col, 0, 5)
    item_t6 = _global_item(obs, row, col, 0, 6)
    item_b5 = _global_item(obs, row, col, 4, 5)
    item_b6 = _global_item(obs, row, col, 4, 6)
    safe_recipe = jnp.clip(recipe_id, 0, 2)
    counter_key = jnp.where(
        item_t5[2 + safe_recipe] > 0,
        0,
        jnp.where(
            item_t6[2 + safe_recipe] > 0,
            1,
            jnp.where(item_b5[2 + safe_recipe] > 0, 2, jnp.where(item_b6[2 + safe_recipe] > 0, 3, -1)),
        ),
    )
    counter_row = jnp.where(counter_key < 2, 1, 3)
    counter_col = jnp.where((counter_key == 0) | (counter_key == 2), 5, 6)
    counter_face = jnp.where(counter_key < 2, UP, DOWN)
    from_counter = _go_interact(obs, row, col, counter_row, counter_col, counter_face)
    to_supply = jnp.where(recipe_id == 0, to_ing0, to_ing1)
    to_ingredient = jnp.where(counter_key >= 0, from_counter, to_supply)

    empty_t5 = jnp.sum(item_t5) == 0
    empty_t6 = jnp.sum(item_t6) == 0
    empty_b5 = jnp.sum(item_b5) == 0
    empty_b6 = jnp.sum(item_b6) == 0
    top_first = held_raw_id == 0
    top_slot = jnp.where(empty_t5, 0, jnp.where(empty_t6, 1, jnp.where(empty_b5, 2, jnp.where(empty_b6, 3, -1))))
    bottom_slot = jnp.where(empty_b5, 2, jnp.where(empty_b6, 3, jnp.where(empty_t5, 0, jnp.where(empty_t6, 1, -1))))
    store_key = jnp.where(top_first, top_slot, bottom_slot)
    store_row = jnp.where(store_key < 2, 1, 3)
    store_col = jnp.where((store_key == 0) | (store_key == 2), 5, 6)
    store_face = jnp.where(store_key < 2, UP, DOWN)
    store_mismatch = jnp.where(
        store_key >= 0,
        _go_interact(obs, row, col, store_row, store_col, store_face),
        STAY,
    )
    raw_mismatch = held_raw & (recipe_id >= 0) & (held_raw_id != recipe_id)

    # After delivery, first return from column 6 to column 5, where a changed
    # recipe display is visible. If there is no display, the stored recipe
    # remains valid and mismatch recovery handles any later update.
    post_delivery = base[1] > 0
    returning_to_button = post_delivery & (col > 5)
    release_post = post_delivery & (col == 5)

    empty_action = jnp.where(
        recipe_id < 0,
        jnp.where((row == 2) & (col == 5), STAY, _go_interact(obs, row, col, 2, 5, LEFT)),
        jnp.where(pot_busy, to_plate, to_ingredient),
    )
    useful_action = jnp.where(
        held_cooked,
        to_delivery,
        jnp.where(
            held_plate,
            to_pot,
            jnp.where(held_raw, jnp.where(raw_mismatch, store_mismatch, to_pot), empty_action),
        ),
    )
    action = jnp.where(returning_to_button, LEFT, useful_action)

    issued_delivery = held_cooked & (row == 2) & (col == 6) & (center[3] > 0) & (action == INTERACT)
    next_post = jnp.where(release_post, 0, base[1])
    next_post = jnp.where(issued_delivery, 1, next_post)
    next_memory = jnp.asarray([recipe_id, next_post, base[2] + 1], dtype=jnp.int32)
    trace = {
        "recipe": recipe_id,
        "display": display_active.astype(jnp.int32),
        "pot_count": pot_count.astype(jnp.int32),
        "pot_ready": pot_cooked.astype(jnp.int32),
        "post_delivery": post_delivery.astype(jnp.int32),
        "raw_mismatch": raw_mismatch.astype(jnp.int32),
    }
    return action, next_memory, trace
