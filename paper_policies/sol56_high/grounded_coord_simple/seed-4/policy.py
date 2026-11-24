"""Grounded cooperative policy for the split-kitchen Overcooked layout."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_button_cook",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 8,
    "num_ingredients": 3,
}


STAY = 4
INTERACT = 5


def _target_delta(mask, offset_row, offset_col):
    """Displacement from self to target+offset for a visible object."""
    flat_index = jnp.argmax(mask.reshape((-1,)))
    target_row = flat_index // 5
    target_col = flat_index % 5
    return (target_row - 2 + offset_row, target_col - 2 + offset_col)


def _move_to(mask, offset_row, offset_col):
    """Move vertically, then horizontally, toward target+offset."""
    dr, dc = _target_delta(mask, offset_row, offset_col)
    vertical = jnp.where(dr > 0, jnp.int32(1), jnp.int32(3))
    horizontal = jnp.where(dc > 0, jnp.int32(0), jnp.int32(2))
    action = jnp.where(dr != 0, vertical, jnp.where(dc != 0, horizontal, STAY))
    arrived = (dr == 0) & (dc == 0)
    return action.astype(jnp.int32), arrived


def _approach_and_interact(mask, offset_row, offset_col, face_action, face_channel, obs):
    """Navigate adjacent to an object, face it, then interact."""
    move, arrived = _move_to(mask, offset_row, offset_col)
    facing = obs[2, 2, face_channel] > 0
    at_object_action = jnp.where(facing, INTERACT, jnp.int32(face_action))
    return jnp.where(arrived, at_object_action, move).astype(jnp.int32), arrived & facing


def init_agent_0():
    # seen, cooldown, step, marker age, button mode, refresh, absolute age,
    # marker offered, marker-confirmed mode
    return jnp.zeros((9,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    seen_nonempty = memory[0] > 0
    cooldown = memory[1]
    step = memory[2]
    empty_age = memory[3]
    button_required = memory[4] > 0
    refresh_state = memory[5]
    absolute_age = memory[6]
    marker_offered = memory[7] > 0
    marker_confirmed = memory[8] > 0

    pot_mask = obs[:, :, 22]
    pot_0 = jnp.sum(obs[:, :, 31] * pot_mask)
    pot_1 = jnp.sum(obs[:, :, 32] * pot_mask)
    pot_2 = jnp.sum(obs[:, :, 33] * pot_mask)
    pot_ingredients = jnp.sum(obs[:, :, 31:34] * pot_mask[:, :, None])
    pot_cooked = jnp.sum(obs[:, :, 30] * pot_mask) > 0
    pot_timer = jnp.sum(obs[:, :, 39] * pot_mask)
    pot_nonempty = (pot_ingredients > 0) | pot_cooked | (pot_timer > 0)

    # Pot emptying is the grounded cue that plating just occurred and the
    # button should cover the imminent delivery/resample transition.
    newly_empty = seen_nonempty & (~pot_nonempty)
    seen_next = seen_nonempty | pot_nonempty
    seen_next = jnp.where(newly_empty, False, seen_next)
    cooldown = jnp.maximum(cooldown - 1, 0)
    cooldown = jnp.where(newly_empty, 0, cooldown)
    button_mask = obs[:, :, 24]
    button_action, can_interact = _approach_and_interact(
        button_mask, 0, -1, 0, 3, obs
    )

    # A persistent, cost-free backup signal uses ingredient 2 (which is never
    # requested) on the upper separator counter for recipe 0 and the lower
    # separator counter for recipe 1.  The right room can see but cannot reach
    # these counters, so the marker cannot contaminate the pot.
    recipe_1 = jnp.sum(obs[:, :, 37]) > 0
    button_index = jnp.argmax(button_mask.reshape((-1,)))
    button_row = button_index // 5
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    marker_items = obs[:, :, 33]
    top_marker = jnp.sum(marker_items * (rows < button_row)) > 0
    bottom_marker = jnp.sum(marker_items * (rows > button_row)) > 0
    desired_marker = jnp.where(recipe_1, bottom_marker, top_marker)
    wrong_marker = jnp.where(recipe_1, top_marker, bottom_marker)
    desired_row_offset = jnp.where(recipe_1, 1, -1)
    wrong_row_offset = -desired_row_offset
    held_marker = obs[2, 2, 9] > 0
    center_action, center_arrived = _move_to(button_mask, 0, -1)

    # Refresh the marker on every plating edge, even if the sampled recipe is
    # unchanged.  State 1 removes the old marker; state 2 replaces it using
    # the post-delivery recipe.  This gives the observer a freshness edge.
    refresh_state = jnp.where(newly_empty, 1, refresh_state)
    refresh_state = jnp.where(
        (refresh_state == 1) & held_marker, 3, refresh_state
    )
    refresh_state = jnp.where(
        (refresh_state == 3) & center_arrived, 2, refresh_state
    )
    refresh_state = jnp.where(
        (refresh_state == 2) & (~held_marker) & desired_marker,
        0,
        refresh_state,
    )
    marker_ready = desired_marker & (refresh_state == 0)
    empty_age = jnp.where(
        pot_nonempty | (~marker_ready),
        0,
        jnp.minimum(empty_age + 1, 1000),
    )
    marker_offered = marker_offered | (marker_ready & (~pot_nonempty))
    undecided = (~button_required) & (~marker_confirmed)
    pot_matches_recipe = jnp.where(
        recipe_1,
        (pot_1 > 0) & (pot_0 == 0) & (pot_2 == 0),
        (pot_0 > 0) & (pot_1 == 0) & (pot_2 == 0),
    )
    pot_contradiction = pot_nonempty & (~pot_matches_recipe)
    good_marker_start = (
        pot_nonempty & undecided & marker_offered & pot_matches_recipe
    )
    bad_start = pot_nonempty & undecided & (
        (~marker_offered) | (~pot_matches_recipe)
    )
    corrective_button = bad_start | (pot_contradiction & marker_confirmed)
    # Confirm only behavior that both follows the marker offer and matches the
    # true recipe. Any contradiction switches permanently to explicit mode.
    marker_confirmed = (marker_confirmed | good_marker_start) & (~pot_contradiction)
    button_required = button_required | bad_start | pot_contradiction
    undecided = (~button_required) & (~marker_confirmed)
    absolute_age = jnp.where(
        pot_nonempty, 0, jnp.minimum(absolute_age + 1, 1000)
    )
    delayed_button = undecided & (absolute_age >= 14)
    need_press = (
        (((~seen_next) & (button_required | delayed_button)) | corrective_button)
        & (cooldown == 0)
        & (step < 375)
    )

    desired_counter_action, _ = _approach_and_interact(
        button_mask, desired_row_offset, -1, 0, 3, obs
    )
    wrong_counter_action, _ = _approach_and_interact(
        button_mask, wrong_row_offset, -1, 0, 3, obs
    )
    actual_row_offset = jnp.where(top_marker, -1, 1)
    actual_counter_action, _ = _approach_and_interact(
        button_mask, actual_row_offset, -1, 0, 3, obs
    )
    supply_action_top, _ = _approach_and_interact(
        button_mask, -1, -1, 3, 1, obs
    )
    supply_action_bottom, _ = _approach_and_interact(
        button_mask, 1, -1, 1, 2, obs
    )
    supply_action = jnp.where(recipe_1, supply_action_bottom, supply_action_top)
    station_move, station_arrived = _move_to(
        button_mask, desired_row_offset, -1
    )
    facing_right = obs[2, 2, 3] > 0
    station_action = jnp.where(
        station_arrived,
        jnp.where(facing_right, STAY, jnp.int32(0)),
        station_move,
    )
    idle_action = jnp.where(button_required, center_action, station_action)
    normal_marker_action = jnp.where(
        held_marker,
        desired_counter_action,
        jnp.where(
            desired_marker,
            idle_action,
            jnp.where(wrong_marker, wrong_counter_action, supply_action),
        ),
    )
    refresh_remove_action = jnp.where(
        top_marker | bottom_marker, actual_counter_action, normal_marker_action
    )
    marker_action = jnp.where(
        refresh_state == 1,
        refresh_remove_action,
        jnp.where(refresh_state == 3, center_action, normal_marker_action),
    )

    # Finish placing a held marker because the button rejects interactions
    # while hands are occupied. Otherwise button communication has priority.
    action = jnp.where(
        need_press & (~held_marker), button_action, marker_action
    ).astype(jnp.int32)
    pressed = need_press & (~held_marker) & can_interact
    button_required_next = button_required | pressed
    cooldown_next = jnp.where(pressed, 10, cooldown)
    next_memory = jnp.stack(
        [
            seen_next.astype(jnp.int32),
            cooldown_next.astype(jnp.int32),
            step + 1,
            empty_age.astype(jnp.int32),
            button_required_next.astype(jnp.int32),
            refresh_state.astype(jnp.int32),
            absolute_age.astype(jnp.int32),
            marker_offered.astype(jnp.int32),
            marker_confirmed.astype(jnp.int32),
        ]
    )
    trace = {
        "pot_nonempty": pot_nonempty,
        "seen_nonempty": seen_next,
        "button_cooldown": cooldown_next,
        "need_press": need_press,
        "step": step,
        "marker_correct": desired_marker,
        "held_marker": held_marker,
        "empty_age": empty_age,
        "button_required": button_required_next,
        "refresh_state": refresh_state,
        "absolute_age": absolute_age,
        "marker_confirmed": marker_confirmed,
        "pot_contradiction": pot_contradiction,
    }
    return action, next_memory, trace


def init_agent_1():
    # recipe, wait, attempts, quarantine counters, trusted, saw marker absent
    return jnp.asarray([-1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    recipe = memory[0]
    wait = memory[1]
    attempts = memory[2]
    used_top_counter = memory[3] > 0
    used_bottom_counter = memory[4] > 0
    recipe_trusted = memory[5] > 0
    saw_marker_absent = memory[6] > 0

    shown_0 = jnp.sum(obs[:, :, 36]) > 0
    shown_1 = jnp.sum(obs[:, :, 37]) > 0
    display_visible = shown_0 | shown_1
    shown_recipe = jnp.where(shown_1, 1, 0).astype(jnp.int32)

    pot_mask = obs[:, :, 22]
    pot_index = jnp.argmax(pot_mask.reshape((-1,)))
    pot_row = pot_index // 5
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    marker_items = obs[:, :, 33]
    marker_0 = jnp.sum(marker_items * (rows < pot_row)) > 0
    marker_1 = jnp.sum(marker_items * (rows > pot_row)) > 0
    marker_visible = marker_0 | marker_1
    marker_recipe = jnp.where(marker_1, 1, 0).astype(jnp.int32)
    marker_area_visible = jnp.sum(obs[:, :, 24]) > 0

    wait = jnp.where(recipe_trusted, 0, wait + 1)
    # After a delivery, let the left partner update a possibly stale marker.
    # The dynamic button display remains immediate and takes precedence.
    saw_marker_absent = saw_marker_absent | (
        (~recipe_trusted) & marker_area_visible & (~marker_visible)
    )
    accept_marker = (
        (~recipe_trusted)
        & marker_visible
        & ((wait >= 5) | saw_marker_absent)
    )
    recipe = jnp.where(accept_marker, marker_recipe, recipe)
    recipe = jnp.where(display_visible, shown_recipe, recipe)
    recipe_trusted = recipe_trusted | accept_marker | display_visible
    wait = jnp.where(recipe_trusted, 0, wait)

    # Avoid total deadlock with a partner that never presses the button.
    # Alternating guesses ensures any wrong attempt is followed by a match.
    fallback_ready = (recipe < 0) & (wait >= 9)
    recipe = jnp.where(fallback_ready, attempts % 2, recipe)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ingredient = jnp.sum(obs[2, 2, 7:10]) > 0
    held_0 = obs[2, 2, 7] > 0
    held_1 = obs[2, 2, 8] > 0
    held_wrong = ((held_0 & (recipe == 1)) | (held_1 & (recipe == 0)))

    pot_total = jnp.sum(obs[:, :, 31:34] * pot_mask[:, :, None])
    pot_cooked = jnp.sum(obs[:, :, 30] * pot_mask) > 0
    pot_timer = jnp.sum(obs[:, :, 39] * pot_mask)
    pot_cooking = pot_timer > 0

    pot_action, _ = _approach_and_interact(pot_mask, 0, 1, 2, 4, obs)
    drop_top_action, drop_top_interact = _approach_and_interact(
        pot_mask, -1, 1, 3, 1, obs
    )
    drop_bottom_action, drop_bottom_interact = _approach_and_interact(
        pot_mask, 1, 1, 1, 2, obs
    )
    goal_action, goal_interact = _approach_and_interact(
        obs[:, :, 21], 0, -1, 0, 3, obs
    )
    plate_action, _ = _approach_and_interact(obs[:, :, 25], 0, 1, 2, 4, obs)
    ingredient_0_action, _ = _approach_and_interact(
        obs[:, :, 26], 0, -1, 0, 3, obs
    )
    ingredient_1_action, _ = _approach_and_interact(
        obs[:, :, 27], 0, -1, 0, 3, obs
    )
    ingredient_action = jnp.where(
        recipe == 1, ingredient_1_action, ingredient_0_action
    )
    wait_action, _ = _move_to(pot_mask, 0, 1)

    need_plate = pot_cooked | pot_cooking | (pot_total >= 3)
    empty_action = jnp.where(
        recipe < 0,
        wait_action,
        jnp.where(need_plate, plate_action, ingredient_action),
    )
    plate_at_pot_action = jnp.where(pot_cooked, pot_action, wait_action)
    choose_top_drop = ~used_top_counter
    can_quarantine = ~(used_top_counter & used_bottom_counter)
    quarantine_action = jnp.where(
        choose_top_drop, drop_top_action, drop_bottom_action
    )
    ingredient_carry_action = jnp.where(
        held_wrong & can_quarantine, quarantine_action, pot_action
    )
    action = jnp.where(
        held_cooked,
        goal_action,
        jnp.where(
            held_plate,
            plate_at_pot_action,
            jnp.where(held_ingredient, ingredient_carry_action, empty_action),
        ),
    ).astype(jnp.int32)

    delivered_attempt = held_cooked & goal_interact
    recipe_next = jnp.where(delivered_attempt, -1, recipe)
    wait_next = jnp.where(delivered_attempt, 0, wait)
    trusted_next = recipe_trusted & (~delivered_attempt)
    absent_next = (
        saw_marker_absent
        & (~accept_marker)
        & (~display_visible)
        & (~delivered_attempt)
    )
    attempts_next = attempts + delivered_attempt.astype(jnp.int32)
    quarantined_top = (
        held_wrong & can_quarantine & choose_top_drop & drop_top_interact
    )
    quarantined_bottom = (
        held_wrong & can_quarantine & (~choose_top_drop) & drop_bottom_interact
    )
    next_memory = jnp.stack(
        [
            recipe_next.astype(jnp.int32),
            wait_next.astype(jnp.int32),
            attempts_next,
            (used_top_counter | quarantined_top).astype(jnp.int32),
            (used_bottom_counter | quarantined_bottom).astype(jnp.int32),
            trusted_next.astype(jnp.int32),
            absent_next.astype(jnp.int32),
        ]
    )
    trace = {
        "recipe": recipe,
        "display_visible": display_visible,
        "marker_visible": marker_visible,
        "wait": wait,
        "attempts": attempts,
        "pot_total": pot_total,
        "pot_cooked": pot_cooked,
        "pot_timer": pot_timer,
        "delivered_attempt": delivered_attempt,
        "held_wrong": held_wrong,
        "recipe_trusted": recipe_trusted,
        "saw_marker_absent": saw_marker_absent,
    }
    return action, next_memory, trace
