"""Convention-legible policy for the two-room Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'pot_side_signal',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}


def _locate(obs, upper):
    # The unique delivery cell (0,2) is visible everywhere in the upper room;
    # the unique recipe indicator (6,2) is visible everywhere in the lower.
    channel = 23 if upper else 25
    flat = jnp.argmax(obs[:, :, channel])
    local_row = flat // 5
    local_col = flat % 5
    object_row = 0 if upper else 6
    return jnp.asarray([object_row + 2 - local_row, 4 - local_col],
                       dtype=jnp.int32)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _navigate(obs, pos, target_row, target_col, target_facing, interact=True):
    """Move on a room floor, then face and optionally interact."""
    row, col = pos[0], pos[1]
    at_target = (row == target_row) & (col == target_col)
    vertical = jnp.where(row < target_row, 1, 3)
    horizontal = jnp.where(col < target_col, 0, 2)
    row_differs = row != target_row
    col_differs = col != target_col
    vertical_first = target_facing >= 2
    choose_vertical = row_differs & ((~col_differs) | vertical_first)
    move = jnp.where(choose_vertical, vertical, horizontal)
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[target_facing]
    finish = jnp.where(_facing(obs) == target_facing,
                       jnp.asarray(5 if interact else 4, dtype=jnp.int32),
                       face_action)
    return jnp.where(at_target, finish, move).astype(jnp.int32)


def _global_cell(obs, pos, row, col):
    local_row = row - pos[0] + 2
    local_col = col - pos[1] + 2
    visible = ((local_row >= 0) & (local_row < 5) &
               (local_col >= 0) & (local_col < 5))
    cell = obs[jnp.clip(local_row, 0, 4), jnp.clip(local_col, 0, 4)]
    return cell, visible


def init_agent_0():
    # elapsed steps, believed recipe (-1 unknown), previous cooked-in-hand,
    # selected pot column, candidate signal, consecutive signal observations,
    # and time spent waiting for evidence
    # Final fields are signal polarity (0 natural / 1 reversed), consecutive
    # unchanged-delivery evidence, last raw signal, and confirmed-polarity flag.
    return jnp.asarray([0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0],
                       dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_0()
    mem = jnp.where(reset, initial, memory)
    pos = _locate(obs, True)
    inv = obs[2, 2, 5:11]
    has_plate = inv[0] > 0
    has_cooked = inv[1] > 0
    has_raw = jnp.sum(inv[2:]) > 0
    delivered = (mem[2] > 0) & (~has_cooked)
    old_recipe = mem[1]
    unknown_age = jnp.where(delivered, 0,
                            jnp.where(old_recipe < 0, mem[6] + 1, 0))

    left, left_visible = _global_cell(obs, pos, 3, 1)
    right, right_visible = _global_cell(obs, pos, 3, 3)
    left_counter, left_counter_visible = _global_cell(obs, pos, 3, 2)
    right_counter, right_counter_visible = _global_cell(obs, pos, 3, 4)
    left_marker = ((left_visible & (left[37] > 0)) |
                   (left_counter_visible & (left_counter[37] > 0)))
    right_marker = ((right_visible & (right[37] > 0)) |
                    (right_counter_visible & (right_counter[37] > 0)))
    marker_valid = (left_marker != right_marker) & (unknown_age >= 6)
    marker_right = marker_valid & right_marker

    # Decode a legible signal: the lower player waits directly below the left
    # or right pot. Intermediate travel positions are deliberately ignored.
    partner_flat = jnp.argmax(obs[:, :, 11])
    partner_visible = jnp.max(obs[:, :, 11]) > 0
    partner_lr = partner_flat // 5
    partner_lc = partner_flat % 5
    partner_row = pos[0] + partner_lr - 2
    partner_col = pos[1] + partner_lc - 2
    signal_left = partner_visible & (partner_row == 4) & (partner_col == 1)
    signal_right = (partner_visible & (partner_row == 4) &
                    ((partner_col == 3) | (partner_col == 4)))
    position_valid = signal_left | signal_right
    signal_valid = position_valid | marker_valid
    decoded_right = jnp.where(position_valid, signal_right, marker_right)
    signaled_recipe = jnp.where(decoded_right, 1, 0).astype(jnp.int32)
    # Debounce the exact position rather than only its decoded meaning, so a
    # lower agent walking through the right half cannot create a false signal.
    marker_code = jnp.where(marker_right, 13, 11)
    signal_position = jnp.where(position_valid, partner_col,
                                marker_code).astype(jnp.int32)
    same_candidate = signal_valid & (mem[4] == signal_position)
    signal_count = jnp.where(signal_valid,
                             jnp.where(same_candidate, mem[5] + 1, 1), 0)
    candidate = jnp.where(signal_valid, signal_position, -1)
    hands_empty = ~(has_plate | has_cooked | has_raw)
    signal_accepted = (signal_valid & (signal_count >= 2) & hands_empty &
                       (old_recipe < 0))

    has_prior_signal = mem[9] >= 0
    raw_changed = (signal_accepted & has_prior_signal &
                   (signaled_recipe != mem[9]))
    raw_same = (signal_accepted & has_prior_signal &
                (signaled_recipe == mem[9]))
    polarity_confirmed = mem[10] > 0
    same_evidence = jnp.where(
        raw_changed, 0,
        jnp.where(raw_same & (~polarity_confirmed), mem[8] + 1, mem[8]))
    flip_polarity = (raw_same & (~polarity_confirmed) &
                     (same_evidence >= 6))
    polarity = jnp.where(flip_polarity, 1 - mem[7], mem[7])
    same_evidence = jnp.where(flip_polarity, 0, same_evidence)
    polarity_confirmed = polarity_confirmed | raw_changed
    mapped_recipe = jnp.bitwise_xor(signaled_recipe, polarity).astype(jnp.int32)
    last_raw_signal = jnp.where(signal_accepted, signaled_recipe, mem[9])
    recipe = jnp.where(signal_accepted, mapped_recipe, old_recipe)
    recipe = jnp.where(delivered, -1, recipe)
    recipe_changed = signal_accepted & (mapped_recipe != old_recipe)
    candidate = jnp.where(delivered, -1, candidate)
    signal_count = jnp.where(delivered, 0, signal_count)
    natural_pot = jnp.where(recipe == 0, 1, 3).astype(jnp.int32)
    previous_pot = jnp.where(recipe_changed | (mem[3] == 0), natural_pot, mem[3])

    # Schedule both pots from visible public state. A stale dish is never
    # plated: it remains available in case a later recipe switches back.
    target_index = jnp.clip(recipe, 0, 3)
    left_total = jnp.sum(left[34:38])
    right_total = jnp.sum(right[34:38])
    left_target = left[34:38][target_index]
    right_target = right[34:38][target_index]
    left_ready = (left_visible & (left[33] > 0) &
                  (left_total == 3) & (left_target == 3))
    right_ready = (right_visible & (right[33] > 0) &
                   (right_total == 3) & (right_target == 3))
    left_fillable = (left_visible & (left[33] == 0) & (left[44] == 0) &
                     (left_total < 3) & (left_target == left_total))
    right_fillable = (right_visible & (right[33] == 0) & (right[44] == 0) &
                      (right_total < 3) & (right_target == right_total))
    left_cooking = (left_visible & (left[44] > 0) &
                    (left_total == 3) & (left_target == 3))
    right_cooking = (right_visible & (right[44] > 0) &
                     (right_total == 3) & (right_target == 3))
    any_ready = left_ready | right_ready
    any_fillable = left_fillable | right_fillable
    any_cooking = left_cooking | right_cooking
    ready_col = jnp.where(left_ready, 1, 3)
    prefer_left_fill = left_fillable & (~right_fillable |
                        (left_total > right_total) |
                        ((left_total == right_total) & (natural_pot == 1)))
    fill_col = jnp.where(prefer_left_fill, 1, 3)
    prefer_left_cooking = left_cooking & (~right_cooking |
                          (left[44] <= right[44]))
    cooking_col = jnp.where(prefer_left_cooking, 1, 3)
    scheduled_pot = jnp.where(any_ready, ready_col,
                              jnp.where(any_fillable, fill_col,
                                        jnp.where(any_cooking, cooking_col,
                                                  natural_pot)))
    previous_fillable = jnp.where(previous_pot == 1,
                                  left_fillable, right_fillable)
    raw_pot = jnp.where(previous_fillable, previous_pot,
                        jnp.where(any_fillable, fill_col, previous_pot))
    previous_serviceable = jnp.where(
        previous_pot == 1, left_ready | left_cooking,
        right_ready | right_cooking)
    plate_fallback = jnp.where(any_ready, ready_col,
                               jnp.where(any_cooking, cooking_col,
                                         previous_pot))
    plate_pot = jnp.where(previous_serviceable, previous_pot,
                          plate_fallback)
    pot_col = jnp.where(has_raw, raw_pot,
                        jnp.where(has_plate, plate_pot,
                                  scheduled_pot)).astype(jnp.int32)

    # Modes: synchronize, carry ingredient, carry plate, deliver, inspect,
    # fetch ingredient, fetch plate.
    watch_action = _navigate(obs, pos, 2, 2, 1, interact=False)
    pot_action = _navigate(obs, pos, 2, pot_col, 1, interact=True)
    plate_action = _navigate(obs, pos, 1, 3, 0, interact=True)
    delivery_action = _navigate(obs, pos, 1, 2, 0, interact=True)
    supply_col = jnp.where(pot_col == 1, 1, 4)
    supply_facing = jnp.where(pot_col == 1, 3, 2)
    supply_row = recipe + 1
    supply_action = _navigate(obs, pos, supply_row, supply_col,
                              supply_facing, interact=True)

    both_visible = left_visible & right_visible
    inspect_action = _navigate(obs, pos, 2, 2, 1, interact=False)
    useful_pot = any_ready | any_fillable | any_cooking
    scheduled_action = jnp.where(any_ready, plate_action,
                                 jnp.where(any_fillable, supply_action,
                                           jnp.where(any_cooking, plate_action,
                                                     watch_action)))
    scheduled_mode = jnp.where(any_ready, 6,
                               jnp.where(any_fillable, 5,
                                         jnp.where(any_cooking, 6, 0)))
    at_supply = ((pos[0] == supply_row) & (pos[1] == supply_col) &
                 any_fillable)
    empty_action = jnp.where(at_supply, supply_action,
                             jnp.where(both_visible, scheduled_action,
                                       inspect_action))
    empty_mode = jnp.where(at_supply, 5,
                           jnp.where(both_visible, scheduled_mode, 4))
    known_action = jnp.where(has_cooked, delivery_action,
                             jnp.where(has_plate, pot_action,
                                       jnp.where(has_raw, pot_action, empty_action)))
    known_mode = jnp.where(has_cooked, 3,
                           jnp.where(has_plate, 2,
                                     jnp.where(has_raw, 1, empty_mode)))
    action = jnp.where(recipe < 0, watch_action, known_action).astype(jnp.int32)
    mode = jnp.where(recipe < 0, 0, known_mode).astype(jnp.int32)

    next_memory = jnp.asarray([mem[0] + 1, recipe,
                               has_cooked.astype(jnp.int32), pot_col,
                               candidate, signal_count,
                               jnp.where(recipe < 0, unknown_age, 0),
                               polarity, same_evidence, last_raw_signal,
                               polarity_confirmed.astype(jnp.int32)],
                              dtype=jnp.int32)
    trace = {
        'left_state': jnp.asarray([left[33], left_total, left_target,
                                   left[44], left_visible], dtype=jnp.int32),
        'mode': mode,
        'polarity_state': jnp.asarray([polarity, same_evidence,
                                       polarity_confirmed], dtype=jnp.int32),
        'pot_col': pot_col,
        'recipe': recipe,
        'schedule_flags': jnp.asarray([any_ready, any_fillable, any_cooking,
                                       left_fillable, right_fillable,
                                       left_cooking, right_cooking],
                                      dtype=jnp.int32),
        'right_state': jnp.asarray([right[33], right_total, right_target,
                                    right[44], right_visible], dtype=jnp.int32),
        'row_col': pos,
    }
    return action, next_memory, trace


def init_agent_1():
    # elapsed, current recipe, phase (position / relocate / create / persist),
    # consecutive observations at the position signal, known marker column
    return jnp.asarray([0, -1, 0, 0, -1], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, init_agent_1(), memory)
    pos = _locate(obs, False)
    displayed_counts = jnp.sum(obs[:, :, 40:44], axis=(0, 1))
    recipe = jnp.argmax(displayed_counts).astype(jnp.int32)
    changed = recipe != mem[1]
    phase = jnp.where(changed, 0, mem[2])
    prior_signal_count = jnp.where(changed, 0, mem[3])
    target_col = jnp.where(recipe == 0, 1, 3)
    signal_action = _navigate(obs, pos, 4, target_col, 0, interact=False)
    at_signal = ((pos[0] == 4) & (pos[1] == target_col) &
                 (_facing(obs) == 0))
    signal_count = jnp.where(phase == 0,
                             jnp.where(at_signal, prior_signal_count + 1, 0),
                             prior_signal_count)
    signal_complete = (phase == 0) & at_signal & (signal_count >= 2)

    # Add a redundant, harmless counter marker after the body-position cue is
    # established: ingredient 3 at (3,2) means recipe 0; at (3,4), recipe 1.
    desired_marker_col = jnp.where(recipe == 0, 2, 4)
    old_marker_col = jnp.where(recipe == 0, 4, 2)
    desired_cell, desired_visible = _global_cell(
        obs, pos, 3, desired_marker_col)
    old_cell, old_visible = _global_cell(obs, pos, 3, old_marker_col)
    desired_present = desired_visible & (desired_cell[37] > 0)
    old_present = old_visible & (old_cell[37] > 0)
    holding_marker = obs[2, 2, 10] > 0
    source_col = jnp.where(recipe == 0, 1, 4)
    source_facing = jnp.where(recipe == 0, 3, 2)
    fetch_action = _navigate(obs, pos, 4, source_col,
                             source_facing, interact=True)
    place_action = _navigate(obs, pos, 4, desired_marker_col,
                             0, interact=True)
    remove_action = _navigate(obs, pos, 4, old_marker_col,
                              0, interact=True)
    discard_action = _navigate(obs, pos, 5, 1, 3, interact=True)
    relocate_action = jnp.where(
        holding_marker,
        jnp.where(desired_present, discard_action, place_action),
        remove_action)
    create_action = jnp.where(holding_marker, place_action,
                              jnp.where(desired_present, signal_action,
                                        fetch_action))
    known_old_marker = ((mem[4] > 0) &
                        (mem[4] != desired_marker_col))
    after_signal_phase = jnp.where(known_old_marker, 1, 2)
    next_phase = jnp.where(signal_complete, after_signal_phase, phase)
    at_old_pose = (pos[0] == 4) & (pos[1] == old_marker_col)
    old_missing = ((phase == 1) & at_old_pose & old_visible &
                   (~old_present) & (~holding_marker))
    next_phase = jnp.where(old_missing, 2, next_phase)
    marker_complete = (((phase == 1) | (phase == 2)) &
                       desired_present & (~old_present) & (~holding_marker))
    next_phase = jnp.where(marker_complete, 3, next_phase)
    next_marker_col = jnp.where(marker_complete, desired_marker_col, mem[4])
    action = jnp.where(phase == 0, signal_action,
                       jnp.where(phase == 1, relocate_action,
                                 jnp.where(phase == 2, create_action,
                                           signal_action))).astype(jnp.int32)
    next_memory = jnp.asarray([mem[0] + 1, recipe, next_phase,
                               signal_count, next_marker_col], dtype=jnp.int32)
    trace = {
        'marker_flags': jnp.asarray([desired_present, old_present,
                                     holding_marker], dtype=jnp.int32),
        'phase': phase,
        'recipe': recipe,
        'row_col': pos,
        'target_col': target_col,
    }
    return action, next_memory, trace
