"""Observable, layout-aware cooperation policy for demo_cook_wide."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "observable_role_pipeline",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 6,
    "max_width": 11,
    "num_ingredients": 4,
}


RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


# Static layout channels 22:32, indexed relative to that slice. Predictions
# are constructed once using only fixed Python control flow.
_STATIC_OBJECTS = {
    0: ((0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9),
        (0, 10), (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7),
        (2, 8), (2, 9), (2, 10), (3, 0), (3, 10), (4, 3), (4, 7),
        (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
        (5, 8), (5, 10)),
    1: ((0, 5),),
    2: ((2, 5),),
    3: ((4, 5),),
    5: ((0, 4), (0, 6)),
    6: ((1, 3), (4, 0), (4, 10)),
    7: ((1, 7), (5, 1), (5, 9)),
    9: ((4, 4), (4, 6)),
}


def _static_grid():
    grid = jnp.zeros((6, 11, 10), dtype=jnp.int32)
    for channel, positions in _STATIC_OBJECTS.items():
        for row, col in positions:
            grid = grid.at[row, col, channel].set(1)
    return grid


def _predictions(candidates):
    grid = _static_grid()
    result = []
    for row, col in candidates:
        crop = jnp.zeros((5, 5, 10), dtype=jnp.int32)
        for i in range(5):
            for j in range(5):
                rr, cc = row + i - 2, col + j - 2
                if 0 <= rr < 6 and 0 <= cc < 11:
                    crop = crop.at[i, j].set(grid[rr, cc])
        result.append(crop)
    return jnp.stack(result)


_POS_0 = ((1, 4), (1, 5), (1, 6))
_POS_1 = (
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
    (3, 8), (3, 9), (4, 1), (4, 2), (4, 8), (4, 9))


def _position(obs, candidates):
    predictions = _predictions(candidates)
    actual = obs[:, :, 22:32]
    error = jnp.sum(jnp.abs(predictions - actual[None]), axis=(1, 2, 3))
    positions = jnp.asarray(candidates, dtype=jnp.int32)
    return positions[jnp.argmin(error)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_then_interact(facing, desired):
    face_action = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)
    return jnp.where(facing == desired, INTERACT, face_action[desired])


def _upper_go(col, target_col):
    return jnp.where(col < target_col, RIGHT,
                     jnp.where(col > target_col, LEFT, STAY))


def _upper_use(col, facing, stand_col, desired_face):
    travel = _upper_go(col, stand_col)
    use = _face_then_interact(facing, jnp.int32(desired_face))
    return jnp.where(col == stand_col, use, travel)


def _pot_values(obs):
    mask = obs[:, :, 24]
    ing0 = jnp.sum(obs[:, :, 34] * mask).astype(jnp.int32)
    ing1 = jnp.sum(obs[:, :, 35] * mask).astype(jnp.int32)
    cooked = jnp.sum(obs[:, :, 33] * mask).astype(jnp.int32)
    timer = jnp.sum(obs[:, :, 44] * mask).astype(jnp.int32)
    visible = jnp.sum(mask) > 0
    return ing0, ing1, cooked, timer, visible


def init_agent_0():
    # step, inferred ingredient (0 unknown, 1 ing0, 2 ing1), stall,
    # last pot total, previous held-dish flag
    return jnp.zeros((5,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    step, remembered, stall, last_total, previous_dish = memory
    pos = _position(obs, _POS_0)
    col = pos[1]
    facing = _facing(obs)

    held_plate = obs[2, 2, 5] > 0
    held_dish = obs[2, 2, 6] > 0
    held0 = obs[2, 2, 7] > 0
    held1 = obs[2, 2, 8] > 0
    held_raw = held0 | held1
    pot0, pot1, pot_cooked, pot_timer, _ = _pot_values(obs)
    pot_total = pot0 + pot1
    ready = pot_cooked > 0
    cooking = pot_timer > 0

    # The two wall counters immediately below upper columns 4 and 6 are both
    # visible from every upper floor cell.
    counter4 = obs[3, jnp.int32(6) - col, 32:38]
    counter6 = obs[3, jnp.int32(8) - col, 32:38]
    counter4_empty = jnp.sum(counter4) == 0
    counter6_empty = jnp.sum(counter6) == 0

    partner0 = jnp.sum(obs[:, :, 18]) > 0
    partner1 = jnp.sum(obs[:, :, 19]) > 0
    partner_signal = partner0 | partner1
    # From any upper-room column, global (3, 5) is on local row 4.
    center_local_col = jnp.int32(7) - col
    partner_at_center = obs[4, center_local_col, 11] > 0
    signaled_code = jnp.where(partner0, jnp.int32(1),
                              jnp.where(partner1, jnp.int32(2), remembered))
    inferred = jnp.where(pot0 > 0, jnp.int32(1),
                         jnp.where(pot1 > 0, jnp.int32(2), signaled_code))
    delivered = (previous_dish > 0) & (~held_dish)
    inferred = jnp.where(delivered, jnp.int32(0), inferred)

    same_partial = (pot_total == last_total) & (pot_total > 0) & (pot_total < 3)
    next_stall = jnp.where(same_partial, stall + 1, jnp.int32(0))

    deliver_action = _upper_use(col, facing, 5, 0)
    scoop_action = _upper_use(col, facing, 5, 1)
    plate_wait_action = jnp.where(
        col == 5, jnp.where(facing == 1, STAY, DOWN),
        _upper_go(col, jnp.int32(5)))
    add_action = _upper_use(col, facing, 5, 1)
    preferred_dump_col = jnp.where(held0, jnp.int32(4), jnp.int32(6))
    alternate_dump_col = jnp.where(held0, jnp.int32(6), jnp.int32(4))
    preferred_empty = jnp.where(held0, counter4_empty, counter6_empty)
    alternate_empty = jnp.where(held0, counter6_empty, counter4_empty)
    dump_col = jnp.where(preferred_empty, preferred_dump_col,
                         jnp.where(alternate_empty, alternate_dump_col,
                                   preferred_dump_col))
    dump_action = _upper_use(col, facing, dump_col, 1)
    raw_action = jnp.where((pot_total < 3) & (~cooking) & (~ready),
                           add_action, dump_action)
    fetch_plate_action = _upper_use(col, facing, 4, 0)
    plate_action = jnp.where(ready, scoop_action, plate_wait_action)

    can_help = (inferred > 0) & (~cooking) & (~ready) & (
        (pot_total == 1) |
        ((pot_total == 2) & (((next_stall >= 1) & partner_at_center &
                              (~partner_signal)) | (next_stall >= 18))) |
        ((pot_total == 0) & partner_signal))
    source_col = jnp.where(inferred == 1, jnp.int32(4), jnp.int32(6))
    source_face = jnp.where(inferred == 1, jnp.int32(3), jnp.int32(2))
    counter4_matches = jnp.where(inferred == 1, counter4[2] > 0,
                                 counter4[3] > 0)
    counter6_matches = jnp.where(inferred == 1, counter6[2] > 0,
                                 counter6[3] > 0)
    has_stored_match = counter4_matches | counter6_matches
    stored_col = jnp.where(counter4_matches, jnp.int32(4), jnp.int32(6))
    supply_action = _upper_use(col, facing, source_col, source_face)
    stored_action = _upper_use(col, facing, stored_col, 1)
    source_action = jnp.where(has_stored_match, stored_action, supply_action)
    empty_action = jnp.where(cooking | ready, fetch_plate_action,
                             jnp.where(can_help, source_action,
                                       _upper_go(col, jnp.int32(5))))

    action = jnp.where(held_dish, deliver_action,
                       jnp.where(held_raw, raw_action,
                                 jnp.where(held_plate, plate_action, empty_action)))
    next_memory = jnp.asarray(
        [step + 1, inferred, next_stall, pot_total, held_dish.astype(jnp.int32)],
        dtype=jnp.int32)
    trace = {
        "position": pos,
        "inferred_recipe": inferred,
        "pot_total": pot_total,
        "stall": next_stall,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # step, recipe, cooldown, last visible pot total, supply side,
    # previous partner-held-dish flag, two-item-pot wait counter,
    # staged-item bit mask, staging mode (0 normal, 1/2 staging, 3 retrieved),
    # and whether the current full-pot cycle is still awaiting delivery
    return jnp.zeros((10,), dtype=jnp.int32)


def _lower_to(row, col, target_row, target_col):
    on_spine = row == 3
    horizontal = jnp.where(col < target_col, RIGHT,
                           jnp.where(col > target_col, LEFT, STAY))
    to_spine = jnp.where(row > 3, UP, horizontal)
    to_alcove = jnp.where(col == target_col, DOWN, horizontal)
    other_alcove = (row > 3) & (jnp.abs(col - target_col) > 1)
    action_for_row4 = jnp.where(on_spine, to_alcove,
                                jnp.where(other_alcove, UP, horizontal))
    return jnp.where(target_row == 3, to_spine, action_for_row4)


def _lower_use(row, col, facing, stand_row, stand_col, desired_face):
    at_target = (row == stand_row) & (col == stand_col)
    travel = _lower_to(row, col, jnp.int32(stand_row), stand_col)
    use = _face_then_interact(facing, desired_face)
    return jnp.where(at_target, use, travel)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    (step, remembered, cooldown, last_total, side,
     previous_partner_dish, partial_wait, stage_mask, stage_mode,
     cycle_active) = memory
    pos = _position(obs, _POS_1)
    row, col = pos[0], pos[1]
    facing = _facing(obs)

    display0 = jnp.sum(obs[:, :, 40]) > 0
    display1 = jnp.sum(obs[:, :, 41]) > 0
    recipe = jnp.where(display0, jnp.int32(1),
                       jnp.where(display1, jnp.int32(2), remembered))
    held0 = obs[2, 2, 7] > 0
    held1 = obs[2, 2, 8] > 0
    held_raw = held0 | held1
    partner_raw = (jnp.sum(obs[:, :, 18]) > 0) | (jnp.sum(obs[:, :, 19]) > 0)
    pot0, pot1, pot_cooked, pot_timer, pot_visible = _pot_values(obs)
    pot_total = pot0 + pot1

    partner_dish = jnp.sum(obs[:, :, 17]) > 0
    emptied = pot_visible & (last_total >= 3) & (pot_total == 0)
    delivery_seen = (previous_partner_dish > 0) & (~partner_dish)
    decayed_cooldown = jnp.maximum(cooldown - 1, 0)
    next_cooldown = jnp.where(emptied, jnp.int32(8), decayed_cooldown)
    next_cooldown = jnp.where(partner_dish, jnp.maximum(next_cooldown, 3),
                              next_cooldown)
    next_cooldown = jnp.where(delivery_seen, jnp.int32(0), next_cooldown)
    newly_active = pot_visible & ((pot_timer > 0) | (pot_cooked > 0) |
                                  (pot_total >= 3))
    delivery_without_dish_view = ((cycle_active > 0) & pot_visible &
                                  (pot_total == 0) & (~partner_dish))
    next_cycle_active = jnp.where(newly_active, jnp.int32(1), cycle_active)
    next_cycle_active = jnp.where(delivery_seen | delivery_without_dish_view,
                                  jnp.int32(0), next_cycle_active)
    next_cooldown = jnp.where(delivery_without_dish_view,
                              jnp.int32(0), next_cooldown)

    existing_code = jnp.where(pot0 > 0, jnp.int32(1),
                              jnp.where(pot1 > 0, jnp.int32(2), recipe))
    wrong_partial = ((pot0 > 0) & (recipe == 2)) | ((pot1 > 0) & (recipe == 1))
    desired = jnp.where(wrong_partial, existing_code, recipe)
    chosen_side = jnp.where(side > 0, side,
                            jnp.where(col <= 5, jnp.int32(1), jnp.int32(2)))
    pickup_col = jnp.where(chosen_side == 1, jnp.int32(1), jnp.int32(9))
    pickup_face0 = jnp.where(chosen_side == 1, jnp.int32(3), jnp.int32(2))
    pickup_face = jnp.where(desired == 1, pickup_face0, jnp.int32(1))
    fetch_action = _lower_use(row, col, facing, 4, pickup_col, pickup_face)

    add_action = _lower_use(row, col, facing, 3, jnp.int32(5), jnp.int32(0))
    will_add = (held_raw & (row == 3) & (col == 5) & (facing == 0) &
                (pot_total < 3) & (pot_timer == 0) & (pot_cooked == 0))
    idle_partial = (pot_timer == 0) & (pot_cooked == 0) & (pot_total < 3)
    same_two = (pot_total == 2) & (last_total == 2)
    next_partial_wait = jnp.where(
        pot_total == 2,
        jnp.where(same_two, partial_wait + 1, jnp.int32(0)),
        jnp.int32(0))
    upper_idle_center = ((row == 3) & (col == 5) &
                         (obs[0, 2, 11] > 0) & (~partner_raw))
    passive_evidence = (next_partial_wait >= 3) & upper_idle_center
    await_assist = (((pot_total == 2) & (next_partial_wait < 8) &
                     (~passive_evidence)) | partner_raw)

    # During cooking, cache one unit of each possible next recipe on the two
    # isolated lower-room counters: ingredient 0 at (4,3), ingredient 1 at
    # (4,7).  The new target is consulted before either cache is consumed.
    missing0 = (jnp.bitwise_and(stage_mask, jnp.int32(1)) == 0)
    missing1 = (jnp.bitwise_and(stage_mask, jnp.int32(2)) == 0)
    staging_in_progress = (stage_mode == 1) | (stage_mode == 2)
    active_window = ((next_cycle_active > 0) | (pot_timer > 0) |
                     (pot_cooked > 0) | (pot_total >= 3) |
                     (next_cooldown > 0))
    fresh_stage_code = jnp.where(missing0, jnp.int32(1), jnp.int32(2))
    stage_code = jnp.where(staging_in_progress, stage_mode, fresh_stage_code)
    stage_pick_col = jnp.where(stage_code == 1, jnp.int32(1), jnp.int32(9))
    stage_pick_face = jnp.where(stage_code == 1, jnp.int32(3), jnp.int32(1))
    stage_fetch_action = _lower_use(
        row, col, facing, 4, stage_pick_col, stage_pick_face)
    held_code = jnp.where(held0, jnp.int32(1), jnp.int32(2))
    redundant_to_cache = (held_raw & (~staging_in_progress) & active_window &
                          ((held0 & missing0) | (held1 & missing1)))
    deposit_code = jnp.where(staging_in_progress, stage_mode, held_code)
    deposit_col = jnp.where(deposit_code == 1, jnp.int32(2), jnp.int32(8))
    deposit_face = jnp.where(deposit_code == 1, jnp.int32(2), jnp.int32(3))
    deposit_action = _lower_use(row, col, facing, 4, deposit_col, deposit_face)
    will_deposit = (held_raw & (staging_in_progress | redundant_to_cache) & (row == 4) &
                    (col == deposit_col) & (facing == deposit_face))

    stage_needed = missing0 | missing1

    staged0 = jnp.bitwise_and(stage_mask, jnp.int32(1)) > 0
    staged1 = jnp.bitwise_and(stage_mask, jnp.int32(2)) > 0
    has_desired_stage = ((desired == 1) & staged0) | ((desired == 2) & staged1)
    # Retrieve from the spine side of the same counters.  This avoids entering
    # and leaving an alcove: (3,3) faces down to (4,3), and (3,7) faces down
    # to (4,7).
    retrieve_col = jnp.where(desired == 1, jnp.int32(3), jnp.int32(7))
    retrieve_face = jnp.int32(1)
    retrieve_action = _lower_use(
        row, col, facing, 3, retrieve_col, retrieve_face)
    will_retrieve = ((~held_raw) & has_desired_stage & idle_partial &
                     (next_cooldown == 0) & (next_cycle_active == 0) & (row == 3) &
                     (col == retrieve_col) & (facing == retrieve_face))

    may_fetch = ((recipe > 0) & idle_partial & (next_cooldown == 0) &
                 (~await_assist) & (~has_desired_stage))
    center_action = _lower_to(row, col, jnp.int32(3), jnp.int32(5))
    normal_empty_action = jnp.where(
        has_desired_stage & idle_partial & (next_cooldown == 0),
        retrieve_action,
        jnp.where(may_fetch | (side > 0), fetch_action, center_action))
    staging_empty_action = jnp.where(stage_needed | staging_in_progress,
                                     stage_fetch_action, center_action)
    empty_action = jnp.where(active_window | staging_in_progress,
                             staging_empty_action, normal_empty_action)
    # If both agents arrive with the third ingredient, yield the pot to the
    # visible upper partner (agent-0 interactions execute first), then cache
    # our redundant item once the pot becomes full.
    yield_to_partner = (held_raw & (~staging_in_progress) & (pot_total == 2) &
                        partner_raw)
    held_action = jnp.where(
        staging_in_progress | redundant_to_cache, deposit_action,
        jnp.where(yield_to_partner, center_action, add_action))
    action = jnp.where(held_raw, held_action, empty_action)

    next_side = jnp.where((may_fetch | (side > 0)) & (~held_raw), chosen_side, side)
    next_side = jnp.where(will_add, jnp.int32(0), next_side)
    next_side = jnp.where(active_window | staging_in_progress | will_retrieve,
                          jnp.int32(0), next_side)
    next_stage_mask = jnp.where(
        will_deposit,
        jnp.bitwise_or(stage_mask, jnp.where(deposit_code == 1,
                                             jnp.int32(1), jnp.int32(2))),
        stage_mask)
    consumed_bit = jnp.where(desired == 1, jnp.int32(1), jnp.int32(2))
    next_stage_mask = jnp.where(
        will_retrieve,
        jnp.bitwise_and(next_stage_mask, jnp.bitwise_not(consumed_bit)),
        next_stage_mask)
    starting_stage = ((~held_raw) & (active_window | staging_in_progress) &
                      (stage_needed | staging_in_progress))
    next_stage_mode = jnp.where(starting_stage, stage_code, stage_mode)
    next_stage_mode = jnp.where(will_deposit, jnp.int32(0), next_stage_mode)
    next_stage_mode = jnp.where(will_retrieve, jnp.int32(3), next_stage_mode)
    next_stage_mode = jnp.where(will_add, jnp.int32(0), next_stage_mode)
    observed_total = jnp.where(pot_visible, pot_total, last_total)
    next_memory = jnp.asarray(
        [step + 1, recipe, next_cooldown, observed_total, next_side,
         partner_dish.astype(jnp.int32), next_partial_wait,
         next_stage_mask, next_stage_mode, next_cycle_active], dtype=jnp.int32)
    trace = {
        "position": pos,
        "recipe": recipe,
        "desired": desired,
        "pot_total": pot_total,
        "cooldown": next_cooldown,
        "side": next_side,
        "partial_wait": next_partial_wait,
        "stage_mask": next_stage_mask,
        "stage_mode": next_stage_mode,
        "cycle_active": next_cycle_active,
    }
    return action.astype(jnp.int32), next_memory, trace
