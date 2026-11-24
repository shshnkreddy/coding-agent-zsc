"""Locally observable policy for the two-room test_time_wide layout.

The lower player can see the free recipe indicator but cannot obtain useful
ingredients. It exposes the recipe with a simple left/right position. The
upper player observes that position and completes one dish at a time.
"""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "visible_recipe_pipeline",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 7,
    "max_width": 6,
    "num_ingredients": 4,
}


WAIT_SIGNAL = 0
GET_INGREDIENT = 1
ADD_INGREDIENT = 2
GET_PLATE = 3
SCOOP = 4
DELIVER = 5


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _upper_position(obs):
    """Recover global position from the visible plate pile at (0, 3)."""
    idx = jnp.argmax(obs[:, :, 27])
    local_row = idx // 5
    local_col = idx % 5
    return (2 - local_row).astype(jnp.int32), (5 - local_col).astype(jnp.int32)


def _lower_position(obs):
    """Recover global position from the visible indicator at (6, 2)."""
    idx = jnp.argmax(obs[:, :, 25])
    local_row = idx // 5
    local_col = idx % 5
    return (8 - local_row).astype(jnp.int32), (4 - local_col).astype(jnp.int32)


def _move_to(row, col, target_row, target_col):
    action = jnp.asarray(4, dtype=jnp.int32)
    action = jnp.where(col > target_col, jnp.asarray(2, jnp.int32), action)
    action = jnp.where(col < target_col, jnp.asarray(0, jnp.int32), action)
    action = jnp.where(row > target_row, jnp.asarray(3, jnp.int32), action)
    action = jnp.where(row < target_row, jnp.asarray(1, jnp.int32), action)
    return action


def _use_object(obs, row, col, stand_row, stand_col, face_id, face_action):
    at_target = (row == stand_row) & (col == stand_col)
    nav = _move_to(row, col, stand_row, stand_col)
    ready = _facing(obs) == jnp.asarray(face_id, dtype=jnp.int32)
    use_or_turn = jnp.where(
        ready, jnp.asarray(5, jnp.int32), jnp.asarray(face_action, jnp.int32)
    )
    return jnp.where(at_target, use_or_turn, nav)


def _global_value(obs, row, col, global_row, global_col, channel):
    """Read a known global cell when it lies in the local crop, else zero."""
    local_row = global_row - row + 2
    local_col = global_col - col + 2
    visible = (
        (local_row >= 0)
        & (local_row < 5)
        & (local_col >= 0)
        & (local_col < 5)
    )
    local_row = jnp.clip(local_row, 0, 4)
    local_col = jnp.clip(local_col, 0, 4)
    return jnp.where(visible, obs[local_row, local_col, channel], 0)


def init_agent_0():
    # phase, ingredient, deposited, wait, clock, raw code, code col,
    # inverted mapping, mapping confirmed, consecutive unchanged probes
    return jnp.asarray([0, 0, 0, 0, 0, -1, -1, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    m = jnp.where(reset, init_agent_0(), memory)
    (
        phase,
        ingredient,
        deposited,
        signal_wait,
        clock,
        last_signal,
        last_signal_col,
        inverted,
        confirmed,
        unchanged_probes,
    ) = m
    row, col = _upper_position(obs)

    partner_mask = obs[:, :, 11]
    partner_visible = jnp.sum(partner_mask) > 0
    partner_idx = jnp.argmax(partner_mask)
    partner_local_col = partner_idx % 5
    partner_global_col = col + partner_local_col - 2
    decoded_signal = (partner_global_col >= 3).astype(jnp.int32)
    partner_faces_right = jnp.sum(obs[:, :, 14]) > 0
    partner_faces_left = jnp.sum(obs[:, :, 15]) > 0
    horizontal_facing = partner_faces_left | partner_faces_right
    facing_signal = partner_faces_right.astype(jnp.int32)

    # During production, opportunistically remember the signaler's stable
    # location.  Do not overwrite it during the post-delivery motion sample.
    stable_observation = (phase != WAIT_SIGNAL) & partner_visible
    last_signal_col = jnp.where(
        stable_observation, partner_global_col, last_signal_col
    )

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:11]) > 0

    wait_next = jnp.where(phase == WAIT_SIGNAL, signal_wait + 1, signal_wait)
    settle_steps = jnp.where(last_signal < 0, jnp.asarray(4), jnp.asarray(1))
    signal_ready = (phase == WAIT_SIGNAL) & (wait_next >= settle_steps)
    fallback = jax.random.bernoulli(rng).astype(jnp.int32)
    have_position_reference = last_signal_col >= 0
    moved_from_prior = partner_global_col != last_signal_col
    refreshed_recipe = jnp.where(moved_from_prior, 1 - last_signal, last_signal)
    spatial_or_motion = jnp.where(
        (last_signal < 0) | (~have_position_reference),
        decoded_signal,
        refreshed_recipe,
    )
    spatial_or_motion = jnp.where(
        horizontal_facing, facing_signal, spatial_or_motion
    )
    observed_code = jnp.where(partner_visible, spatial_or_motion, fallback)

    # A changed code after delivery proves that the preceding dish was right:
    # only a correct delivery resamples the recipe.  Until that evidence
    # arrives, try the opposite binary convention after three unchanged probes.
    is_refresh = last_signal >= 0
    code_changed = is_refresh & (observed_code != last_signal)
    newly_confirmed = (confirmed > 0) | (signal_ready & code_changed)
    counted_unchanged = jnp.where(
        signal_ready & is_refresh & (~code_changed) & (confirmed == 0),
        unchanged_probes + 1,
        unchanged_probes,
    )
    counted_unchanged = jnp.where(
        signal_ready & code_changed, jnp.asarray(0), counted_unchanged
    )
    probe_limit = jnp.where(inverted == 0, jnp.asarray(3), jnp.asarray(2))
    toggle_mapping = (
        signal_ready
        & is_refresh
        & (~code_changed)
        & (confirmed == 0)
        & (counted_unchanged >= probe_limit)
    )
    inverted = jnp.where(toggle_mapping, 1 - inverted, inverted)
    unchanged_probes = jnp.where(
        toggle_mapping, jnp.asarray(0), counted_unchanged
    )
    confirmed = newly_confirmed.astype(jnp.int32)
    selected_ingredient = jnp.bitwise_xor(observed_code, inverted)
    ingredient = jnp.where(signal_ready, selected_ingredient, ingredient)
    last_signal = jnp.where(signal_ready, observed_code, last_signal)
    last_signal_col = jnp.where(
        signal_ready & partner_visible, partner_global_col, last_signal_col
    )
    phase = jnp.where(signal_ready, GET_INGREDIENT, phase)

    got_ingredient = (phase == GET_INGREDIENT) & held_raw
    phase = jnp.where(got_ingredient, ADD_INGREDIENT, phase)

    finished_add = (phase == ADD_INGREDIENT) & (~held_raw)
    deposited_next = deposited + finished_add.astype(jnp.int32)
    full = finished_add & (deposited_next >= 3)
    phase = jnp.where(finished_add & (~full), GET_INGREDIENT, phase)
    phase = jnp.where(full, GET_PLATE, phase)
    deposited = jnp.where(finished_add, deposited_next, deposited)

    got_plate = (phase == GET_PLATE) & held_plate
    phase = jnp.where(got_plate, SCOOP, phase)
    got_dish = (phase == SCOOP) & held_cooked
    phase = jnp.where(got_dish, DELIVER, phase)
    delivered = (phase == DELIVER) & (~held_cooked)
    phase = jnp.where(delivered, WAIT_SIGNAL, phase)
    deposited = jnp.where(delivered, jnp.asarray(0, jnp.int32), deposited)
    wait_next = jnp.where(delivered, jnp.asarray(0, jnp.int32), wait_next)

    action = _move_to(row, col, jnp.asarray(2), jnp.asarray(2))
    supply_row = jnp.where(ingredient == 0, jnp.asarray(1), jnp.asarray(2))
    get_action = _use_object(obs, row, col, supply_row, jnp.asarray(1), 3, 2)
    action = jnp.where(phase == GET_INGREDIENT, get_action, action)

    add_action = _use_object(obs, row, col, jnp.asarray(2), jnp.asarray(1), 1, 1)
    action = jnp.where(phase == ADD_INGREDIENT, add_action, action)
    plate_action = _use_object(obs, row, col, jnp.asarray(1), jnp.asarray(3), 0, 3)
    action = jnp.where(phase == GET_PLATE, plate_action, action)
    scoop_action = _use_object(obs, row, col, jnp.asarray(2), jnp.asarray(1), 1, 1)
    action = jnp.where(phase == SCOOP, scoop_action, action)
    deliver_action = _use_object(obs, row, col, jnp.asarray(1), jnp.asarray(2), 0, 3)
    action = jnp.where(phase == DELIVER, deliver_action, action)

    next_memory = jnp.stack(
        [
            phase,
            ingredient,
            deposited,
            wait_next,
            clock + 1,
            last_signal,
            last_signal_col,
            inverted,
            confirmed,
            unchanged_probes,
        ]
    ).astype(jnp.int32)
    trace = {
        "phase": phase,
        "recipe": ingredient,
        "deposited": deposited,
        "row": row,
        "col": col,
        "partner_visible": partner_visible.astype(jnp.int32),
        "facing_signal": horizontal_facing.astype(jnp.int32),
        "signal_col": last_signal_col,
        "inverted": inverted,
        "confirmed": confirmed,
        "unchanged": unchanged_probes,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # recipe, clock, actively servicing a physical handoff
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    m = jnp.where(reset, init_agent_1(), memory)
    row, col = _lower_position(obs)
    recipe_0 = jnp.sum(obs[:, :, 40])
    recipe_1 = jnp.sum(obs[:, :, 41])
    recipe = (recipe_1 > recipe_0).astype(jnp.int32)
    engaged = m[2]

    target_col = jnp.where(recipe == 0, jnp.asarray(1), jnp.asarray(4))
    at_target = (row == 4) & (col == target_col)
    nav = _move_to(row, col, jnp.asarray(4), target_col)
    desired_facing = jnp.where(recipe == 0, jnp.asarray(3), jnp.asarray(2))
    desired_face_action = jnp.where(recipe == 0, jnp.asarray(2), jnp.asarray(0))
    face_or_wait = jnp.where(
        _facing(obs) == desired_facing,
        jnp.asarray(4, jnp.int32),
        desired_face_action.astype(jnp.int32),
    )
    action = jnp.where(at_target, face_or_wait, nav)

    # Convention-free fallback cooperation: if an upper partner places useful
    # items on either shared counter, physically process them in the lower room.
    # This entire branch is dormant with the self-play upper policy.
    held_plate_flag = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw_0 = obs[2, 2, 7] > 0
    held_raw_1 = obs[2, 2, 8] > 0
    held_raw = held_raw_0 | held_raw_1
    held_target = jnp.where(recipe == 0, held_raw_0, held_raw_1)
    empty_hands = ~(held_plate_flag | held_raw)
    empty_plate = held_plate_flag & (~held_cooked)

    def cell(global_col, channel):
        return _global_value(obs, row, col, 3, global_col, channel)

    c2_total = sum(cell(2, channel) for channel in range(32, 38))
    c4_total = sum(cell(4, channel) for channel in range(32, 38))
    c2_empty = c2_total == 0
    c4_empty = c4_total == 0
    target_channel = 34 + recipe
    c2_raw_target = (
        (cell(2, target_channel) > 0)
        & (cell(2, 32) == 0)
        & (cell(2, 33) == 0)
        & (cell(2, 34) + cell(2, 35) == cell(2, target_channel))
    )
    c4_raw_target = (
        (cell(4, target_channel) > 0)
        & (cell(4, 32) == 0)
        & (cell(4, 33) == 0)
        & (cell(4, 34) + cell(4, 35) == cell(4, target_channel))
    )
    raw_counter_available = c2_raw_target | c4_raw_target
    raw_counter_col = jnp.where(c2_raw_target, jnp.asarray(2), jnp.asarray(4))

    def pot(global_col, channel):
        return _global_value(obs, row, col, 3, global_col, channel)

    left_total = sum(pot(1, channel) for channel in range(34, 38))
    right_total = sum(pot(3, channel) for channel in range(34, 38))
    left_target_count = pot(1, target_channel)
    right_target_count = pot(3, target_channel)
    left_accepts = (
        (pot(1, 33) == 0)
        & (pot(1, 44) == 0)
        & (left_total < 3)
        & (left_total == left_target_count)
    )
    right_accepts = (
        (pot(3, 33) == 0)
        & (pot(3, 44) == 0)
        & (right_total < 3)
        & (right_total == right_target_count)
    )
    pot_available = left_accepts | right_accepts
    use_right_pot = right_accepts & ((~left_accepts) | (right_total > left_total))
    selected_pot_col = jnp.where(use_right_pot, jnp.asarray(3), jnp.asarray(1))
    left_ready = (pot(1, 33) > 0) & (left_target_count == 3)
    right_ready = (pot(3, 33) > 0) & (right_target_count == 3)
    ready_pot = left_ready | right_ready
    ready_pot_col = jnp.where(left_ready, jnp.asarray(1), jnp.asarray(3))

    c2_plate = (cell(2, 32) > 0) & (cell(2, 33) == 0)
    c4_plate = (cell(4, 32) > 0) & (cell(4, 33) == 0)
    plate_counter_available = c2_plate | c4_plate
    plate_counter_col = jnp.where(c2_plate, jnp.asarray(2), jnp.asarray(4))
    empty_counter_available = c2_empty | c4_empty
    empty_counter_col = jnp.where(c2_empty, jnp.asarray(2), jnp.asarray(4))

    pot_in_progress = (
        ((left_total > 0) & (left_total == left_target_count))
        | ((right_total > 0) & (right_total == right_target_count))
    )
    counter_has_cooked = (cell(2, 33) > 0) | (cell(4, 33) > 0)

    work_mode = jnp.asarray(0, jnp.int32)
    monitor_handoff = (engaged > 0) & empty_hands & pot_in_progress
    monitor_action = _move_to(row, col, jnp.asarray(4), jnp.asarray(2))
    action = jnp.where(monitor_handoff, monitor_action, action)
    work_mode = jnp.where(monitor_handoff, jnp.asarray(7), work_mode)

    fetch_raw = empty_hands & raw_counter_available
    fetch_raw_action = _use_object(
        obs, row, col, jnp.asarray(4), raw_counter_col, 0, 3
    )
    action = jnp.where(fetch_raw, fetch_raw_action, action)
    work_mode = jnp.where(fetch_raw, jnp.asarray(1), work_mode)
    engaged = jnp.where(fetch_raw, jnp.asarray(1), engaged)

    load_pot = held_raw & held_target & pot_available
    load_action = _use_object(
        obs, row, col, jnp.asarray(4), selected_pot_col, 0, 3
    )
    action = jnp.where(load_pot, load_action, action)
    work_mode = jnp.where(load_pot, jnp.asarray(2), work_mode)

    return_wrong_raw = held_raw & (~held_target) & empty_counter_available
    return_raw_action = _use_object(
        obs, row, col, jnp.asarray(4), empty_counter_col, 0, 3
    )
    action = jnp.where(return_wrong_raw, return_raw_action, action)
    work_mode = jnp.where(return_wrong_raw, jnp.asarray(3), work_mode)

    scoop_ready = empty_plate & ready_pot
    scoop_action = _use_object(
        obs, row, col, jnp.asarray(4), ready_pot_col, 0, 3
    )
    action = jnp.where(scoop_ready, scoop_action, action)
    work_mode = jnp.where(scoop_ready, jnp.asarray(4), work_mode)

    return_dish = held_cooked & empty_counter_available
    return_dish_action = _use_object(
        obs, row, col, jnp.asarray(4), empty_counter_col, 0, 3
    )
    action = jnp.where(return_dish, return_dish_action, action)
    work_mode = jnp.where(return_dish, jnp.asarray(5), work_mode)

    fetch_plate = empty_hands & ready_pot & plate_counter_available
    fetch_plate_action = _use_object(
        obs, row, col, jnp.asarray(4), plate_counter_col, 0, 3
    )
    action = jnp.where(fetch_plate, fetch_plate_action, action)
    work_mode = jnp.where(fetch_plate, jnp.asarray(6), work_mode)

    finished_handoff = (engaged > 0) & empty_hands & counter_has_cooked
    engaged = jnp.where(finished_handoff, jnp.asarray(0), engaged)
    next_memory = jnp.stack([recipe, m[1] + 1, engaged]).astype(jnp.int32)
    trace = {"recipe": recipe, "row": row, "col": col, "work_mode": work_mode}
    return action.astype(jnp.int32), next_memory, trace
