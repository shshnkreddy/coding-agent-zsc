"""Observation-driven policy for the two-room test_time_wide layout."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "landmark_roles_v1",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 7,
    "max_width": 6,
    "num_ingredients": 4,
}


RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

UPPER_POS = [[1, 1], [1, 2], [1, 3], [1, 4],
             [2, 1], [2, 2], [2, 3], [2, 4]]
LOWER_POS = [[4, 1], [4, 2], [4, 3], [4, 4],
             [5, 1], [5, 2], [5, 3], [5, 4]]

WALLS = [[0, 0], [0, 1], [0, 4], [0, 5], [3, 0], [3, 2], [3, 4],
         [3, 5], [5, 0], [5, 5], [6, 0], [6, 1], [6, 3], [6, 4],
         [6, 5]]
GOALS = [[0, 2]]
POTS = [[3, 1], [3, 3]]
RECIPE_INDICATORS = [[6, 2]]
PLATES = [[0, 3]]
ING0 = [[1, 0], [1, 5]]
ING1 = [[2, 0], [2, 5]]
ING2 = [[-9, -9]]
ING3 = [[4, 0], [4, 5]]


def _presence(rr, cc, coords):
    coords = jnp.asarray(coords, dtype=jnp.int32)
    return jnp.any(
        (rr[..., None] == coords[:, 0]) & (cc[..., None] == coords[:, 1]),
        axis=-1,
    ).astype(jnp.int32)


def _expected_static(candidates):
    axis = jnp.arange(5, dtype=jnp.int32) - jnp.int32(2)
    rr = candidates[:, 0, None, None] + axis[None, :, None]
    cc = candidates[:, 1, None, None] + axis[None, None, :]
    zeros = jnp.zeros_like(rr + cc)
    return jnp.stack(
        [_presence(rr, cc, WALLS),
         _presence(rr, cc, GOALS),
         _presence(rr, cc, POTS),
         _presence(rr, cc, RECIPE_INDICATORS),
         zeros,
         _presence(rr, cc, PLATES),
         _presence(rr, cc, ING0),
         _presence(rr, cc, ING1),
         _presence(rr, cc, ING2),
         _presence(rr, cc, ING3)],
        axis=-1,
    )


def _locate(obs, candidates):
    candidates = jnp.asarray(candidates, dtype=jnp.int32)
    signatures = _expected_static(candidates)
    seen = obs[:, :, 22:32]
    mismatch = jnp.sum(jnp.abs(signatures - seen[None, :, :, :]), axis=(1, 2, 3))
    return candidates[jnp.argmin(mismatch)]


def _navigate(pos, target):
    """Shortest path inside either unobstructed 2x4 room."""
    return jnp.where(
        pos[0] < target[0], DOWN,
        jnp.where(
            pos[0] > target[0], UP,
            jnp.where(pos[1] < target[1], RIGHT,
                      jnp.where(pos[1] > target[1], LEFT, STAY)),
        ),
    ).astype(jnp.int32)


def _navigate_horizontal_first(pos, target):
    return jnp.where(
        pos[1] < target[1], RIGHT,
        jnp.where(
            pos[1] > target[1], LEFT,
            jnp.where(pos[0] < target[0], DOWN,
                      jnp.where(pos[0] > target[0], UP, STAY)),
        ),
    ).astype(jnp.int32)


def _use_station(obs, pos, target, move_action, facing_channel):
    at_target = jnp.all(pos == target)
    correctly_facing = obs[2, 2, facing_channel] > 0
    station_action = jnp.where(correctly_facing, INTERACT, move_action)
    vertical_station = (move_action == UP) | (move_action == DOWN)
    travel_action = jnp.where(vertical_station,
                              _navigate_horizontal_first(pos, target),
                              _navigate(pos, target))
    return jnp.where(at_target, station_action, travel_action).astype(jnp.int32)


def _global_cell_block(obs, pos, global_row, global_col, start, stop):
    li = jnp.int32(global_row) - pos[0] + jnp.int32(2)
    lj = jnp.int32(global_col) - pos[1] + jnp.int32(2)
    visible = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    li = jnp.clip(li, 0, 4)
    lj = jnp.clip(lj, 0, 4)
    return jnp.where(visible, obs[li, lj, start:stop], jnp.zeros((stop - start,), jnp.int32))


def init_agent_0():
    # step, recipe, cue candidate/count, committed pot/flag, mapping mode,
    # last delivered signal symbol, natural/probe same-symbol counts, pending.
    return jnp.asarray([0, -1, -1, 0, 1, 0, 0, -1, 0, 0, 0],
                       dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    m = jnp.where(reset, init_agent_0(), memory)
    pos = _locate(obs, UPPER_POS)

    partner_flat = jnp.argmax(obs[:, :, 11].reshape(-1))
    pi = partner_flat // jnp.int32(5)
    pj = partner_flat % jnp.int32(5)
    partner_visible = jnp.sum(obs[:, :, 11]) > 0
    partner_row = pos[0] + pi - jnp.int32(2)
    partner_col = pos[1] + pj - jnp.int32(2)
    central = partner_visible & (partner_row == 4) & (partner_col == 2)
    central_left = central & (obs[pi, pj, 15] > 0)
    central_right = central & (obs[pi, pj, 14] > 0)
    faces_up = partner_visible & (obs[pi, pj, 12] > 0)
    counter_left = faces_up & (partner_row == 4) & (partner_col == 2)
    counter_right = faces_up & (partner_row == 4) & (partner_col == 4)
    pot_left = partner_visible & (partner_row == 4) & (partner_col == 1)
    pot_right = partner_visible & (partner_row == 4) & (partner_col == 3)
    far_right = partner_visible & (partner_row == 4) & (partner_col == 4)
    partner_holds_3 = partner_visible & (obs[pi, pj, 21] > 0)
    cue0 = (pot_left | central_left | counter_left) & (~partner_holds_3)
    cue1 = (pot_right | central_right | counter_right | far_right) & \
        (~partner_holds_3)
    position_cue = jnp.where(cue0, jnp.int32(0),
                             jnp.where(cue1, jnp.int32(1), jnp.int32(-1)))
    partner_holds_0 = partner_visible & (obs[pi, pj, 18] > 0)
    partner_holds_1 = partner_visible & (obs[pi, pj, 19] > 0)
    direct_cue = jnp.where(partner_holds_0, jnp.int32(0),
                           jnp.where(partner_holds_1, jnp.int32(1), jnp.int32(-1)))
    cue = jnp.where(direct_cue >= 0, direct_cue, position_cue)
    same_cue = (cue >= 0) & (cue == m[2])
    cue_count = jnp.where(cue < 0, jnp.int32(0),
                          jnp.where(same_cue, jnp.minimum(m[3] + 1, 9), 1))
    directional_arrival = (pot_left & (obs[pi, pj, 15] > 0)) | \
        (pot_right & (obs[pi, pj, 14] > 0))
    strong_pot_cue = (faces_up & (pot_left | pot_right)) | directional_arrival
    enough_evidence = (cue_count >= 3) | \
        (strong_pot_cue & (cue_count >= 2)) | (direct_cue >= 0)
    resolves_delivery = enough_evidence & (m[10] > 0) & (m[7] >= 0)
    cue_changed = cue != m[7]
    natural_same = jnp.where(
        (m[6] == 0) & resolves_delivery,
        jnp.where(cue_changed, jnp.int32(0), m[8] + 1), m[8])
    probe_same = jnp.where(
        (m[6] == 1) & resolves_delivery,
        jnp.where(cue_changed, jnp.int32(0), m[9] + 1), m[9])
    start_inverse_probe = (m[6] == 0) & resolves_delivery & \
        (~cue_changed) & (natural_same >= 3)
    confirm_natural = (m[6] == 0) & resolves_delivery & cue_changed
    confirm_inverse = (m[6] == 1) & resolves_delivery & cue_changed
    reject_inverse = (m[6] == 1) & resolves_delivery & \
        (~cue_changed) & (probe_same >= 1)
    mapping_mode = jnp.where(
        start_inverse_probe, jnp.int32(1),
        jnp.where(confirm_inverse, jnp.int32(2),
                  jnp.where(confirm_natural, jnp.int32(3),
                            jnp.where(reject_inverse, jnp.int32(0), m[6]))))
    natural_same = jnp.where(
        start_inverse_probe, jnp.int32(0),
        jnp.where(confirm_natural, jnp.int32(0),
                  jnp.where(reject_inverse, jnp.int32(2), natural_same)))
    probe_same = jnp.where(start_inverse_probe | confirm_inverse | reject_inverse,
                           jnp.int32(0), probe_same)
    inverse_mapping = (mapping_mode == 1) | (mapping_mode == 2)
    mapped_cue = jnp.where(inverse_mapping, jnp.int32(1) - cue, cue)
    accepted = jnp.where(enough_evidence, mapped_cue, m[1])
    last_symbol = jnp.where(enough_evidence, cue, m[7])
    pending_delivery = jnp.where(resolves_delivery, jnp.int32(0), m[10])

    inv = obs[2, 2, 5:11]
    holds_plate = inv[0] > 0
    holds_cooked = inv[1] > 0
    holds_raw = jnp.sum(inv[2:]) > 0
    empty = jnp.sum(inv) == 0

    recipe_index = jnp.clip(accepted, 0, 1)
    indexed_pot_col = jnp.where(accepted == 1, jnp.int32(3), jnp.int32(1))
    left_item = _global_cell_block(obs, pos, 3, 1, 32, 38)
    right_item = _global_cell_block(obs, pos, 3, 3, 32, 38)
    left_timer = _global_cell_block(obs, pos, 3, 1, 44, 45)[0]
    right_timer = _global_cell_block(obs, pos, 3, 3, 44, 45)[0]
    left_count = jnp.sum(left_item[2:])
    right_count = jnp.sum(right_item[2:])
    left_wrong = (left_count - left_item[2 + recipe_index]) > 0
    right_wrong = (right_count - right_item[2 + recipe_index]) > 0
    left_fillable = (~left_wrong) & (left_item[1] == 0) & \
        (left_timer == 0) & (left_count < 3)
    right_fillable = (~right_wrong) & (right_item[1] == 0) & \
        (right_timer == 0) & (right_count < 3)
    left_partial = left_fillable & (left_count > 0)
    right_partial = right_fillable & (right_count > 0)
    indexed_left_available = (indexed_pot_col == 1) & left_fillable
    selected_pot_col = jnp.where(
        left_partial, jnp.int32(1),
        jnp.where(right_partial, jnp.int32(3),
                  jnp.where(indexed_left_available, jnp.int32(1),
                            jnp.where(right_fillable, jnp.int32(3), jnp.int32(1)))),
    )
    fill_exists = left_fillable | right_fillable
    active_fill_col = jnp.where(m[5] > 0, m[4], selected_pot_col)
    effective_fill_exists = fill_exists | (m[5] > 0)
    active_item = jnp.where(active_fill_col == 1, left_item, right_item)
    active_timer = jnp.where(active_fill_col == 1, left_timer, right_timer)
    active_count = jnp.sum(active_item[2:])
    active_wrong = (active_count - active_item[2 + recipe_index]) > 0
    active_can_accept = (~active_wrong) & (active_item[1] == 0) & \
        (active_timer == 0) & (active_count < 3)
    pot_item = jnp.where(selected_pot_col == 1, left_item, right_item)
    pot_timer = jnp.where(selected_pot_col == 1, left_timer, right_timer)
    pot_cooked = pot_item[1] > 0
    pot_count = jnp.sum(pot_item[2:])

    left_ready = (left_item[1] > 0) & (left_count == 3) & \
        (left_item[2 + recipe_index] == 3)
    right_ready = (right_item[1] > 0) & (right_count == 3) & \
        (right_item[2 + recipe_index] == 3)
    left_cooking = (left_timer > 0) & (left_count == 3) & \
        (left_item[2 + recipe_index] == 3)
    right_cooking = (right_timer > 0) & (right_count == 3) & \
        (right_item[2 + recipe_index] == 3)
    any_ready = left_ready | right_ready
    any_cooking = left_cooking | right_cooking
    indexed_ready = jnp.where(indexed_pot_col == 1, left_ready, right_ready)
    indexed_cooking = jnp.where(indexed_pot_col == 1, left_cooking, right_cooking)
    ready_col = jnp.where(indexed_ready, indexed_pot_col,
                          jnp.where(left_ready, jnp.int32(1), jnp.int32(3)))
    cooking_col = jnp.where(indexed_cooking, indexed_pot_col,
                            jnp.where(left_cooking, jnp.int32(1), jnp.int32(3)))
    service_col = jnp.where(any_ready, ready_col, cooking_col)
    need_plate = any_ready | ((~fill_exists) & any_cooking)

    counter2 = _global_cell_block(obs, pos, 3, 2, 32, 38)
    counter4 = _global_cell_block(obs, pos, 3, 4, 32, 38)
    preferred_counter_col = jnp.where(accepted == 1, jnp.int32(4), jnp.int32(2))
    alternate_counter_col = jnp.where(preferred_counter_col == 2,
                                      jnp.int32(4), jnp.int32(2))
    preferred_counter = jnp.where(preferred_counter_col == 2, counter2, counter4)
    alternate_counter = jnp.where(preferred_counter_col == 2, counter4, counter2)

    preferred_raw = (preferred_counter[2 + recipe_index] == 1) & \
        (jnp.sum(preferred_counter[2:]) == 1) & (preferred_counter[0] == 0)
    alternate_raw = (alternate_counter[2 + recipe_index] == 1) & \
        (jnp.sum(alternate_counter[2:]) == 1) & (alternate_counter[0] == 0)
    raw_counter_exists = preferred_raw | alternate_raw
    raw_counter_col = jnp.where(preferred_raw, preferred_counter_col,
                                alternate_counter_col)

    preferred_plate = (preferred_counter[0] > 0) & (preferred_counter[1] == 0) & \
        (jnp.sum(preferred_counter[2:]) == 0)
    alternate_plate = (alternate_counter[0] > 0) & (alternate_counter[1] == 0) & \
        (jnp.sum(alternate_counter[2:]) == 0)
    plate_counter_exists = preferred_plate | alternate_plate
    plate_counter_col = jnp.where(preferred_plate, preferred_counter_col,
                                  alternate_counter_col)

    preferred_dish = (preferred_counter[0] > 0) & (preferred_counter[1] > 0) & \
        (preferred_counter[2 + recipe_index] == 3) & \
        (jnp.sum(preferred_counter[2:]) == 3)
    alternate_dish = (alternate_counter[0] > 0) & (alternate_counter[1] > 0) & \
        (alternate_counter[2 + recipe_index] == 3) & \
        (jnp.sum(alternate_counter[2:]) == 3)
    dish_counter_exists = preferred_dish | alternate_dish
    dish_counter_col = jnp.where(preferred_dish, preferred_counter_col,
                                 alternate_counter_col)

    waiting_target = jnp.asarray([2, 2], dtype=jnp.int32)
    fill_pot_target = jnp.asarray([2, active_fill_col], dtype=jnp.int32)
    service_pot_target = jnp.asarray([2, service_col], dtype=jnp.int32)
    plate_target = jnp.asarray([1, 3], dtype=jnp.int32)
    goal_target = jnp.asarray([1, 2], dtype=jnp.int32)
    ing0_left = jnp.asarray([1, 1], dtype=jnp.int32)
    ing0_right = jnp.asarray([1, 4], dtype=jnp.int32)
    ing1_left = jnp.asarray([2, 1], dtype=jnp.int32)
    ing1_right = jnp.asarray([2, 4], dtype=jnp.int32)

    unknown_action = _navigate(pos, waiting_target)
    ingredient0_action = jnp.where(
        active_fill_col == 1,
        _use_station(obs, pos, ing0_left, LEFT, 4),
        _use_station(obs, pos, ing0_right, RIGHT, 3),
    )
    ingredient1_action = jnp.where(
        active_fill_col == 1,
        _use_station(obs, pos, ing1_left, LEFT, 4),
        _use_station(obs, pos, ing1_right, RIGHT, 3),
    )
    ingredient_action = jnp.where(accepted == 0,
                                  ingredient0_action, ingredient1_action)
    ordinary_empty_action = jnp.where(
        need_plate,
        _use_station(obs, pos, plate_target, UP, 1),
        jnp.where(effective_fill_exists, ingredient_action, STAY),
    )
    raw_counter_target = jnp.asarray([2, raw_counter_col], dtype=jnp.int32)
    plate_counter_target = jnp.asarray([2, plate_counter_col], dtype=jnp.int32)
    dish_counter_target = jnp.asarray([2, dish_counter_col], dtype=jnp.int32)
    empty_action = jnp.where(
        dish_counter_exists,
        _use_station(obs, pos, dish_counter_target, DOWN, 2),
        jnp.where(
            raw_counter_exists & effective_fill_exists,
            _use_station(obs, pos, raw_counter_target, DOWN, 2),
            jnp.where(
                plate_counter_exists & need_plate,
                _use_station(obs, pos, plate_counter_target, DOWN, 2),
                ordinary_empty_action,
            ),
        ),
    )
    plate_action = jnp.where(
        any_ready | any_cooking,
        _use_station(obs, pos, service_pot_target, DOWN, 2),
        STAY,
    )
    raw_action = jnp.where(
        effective_fill_exists & active_can_accept,
        _use_station(obs, pos, fill_pot_target, DOWN, 2),
        STAY,
    )
    cooked_action = _use_station(obs, pos, goal_target, UP, 1)
    known_action = jnp.where(
        holds_cooked, cooked_action,
        jnp.where(holds_plate, plate_action,
                  jnp.where(holds_raw, raw_action,
                            jnp.where(empty, empty_action, STAY))),
    )
    action = jnp.where(accepted >= 0, known_action, unknown_action).astype(jnp.int32)

    attempted_delivery = holds_cooked & jnp.all(pos == goal_target) & (action == INTERACT)
    starting_fill = empty & (accepted >= 0) & (~dish_counter_exists) & \
        (~need_plate) & fill_exists & (m[5] == 0)
    deposit_attempt = holds_raw & jnp.all(pos == fill_pot_target) & \
        (action == INTERACT)
    abort_fill = holds_raw & (m[5] > 0) & (~active_can_accept)
    next_work_col = jnp.where(starting_fill, selected_pot_col, m[4])
    next_committed = jnp.where(deposit_attempt | abort_fill, jnp.int32(0),
                               jnp.where(starting_fill, jnp.int32(1), m[5]))
    next_accepted = jnp.where(attempted_delivery, jnp.int32(-1), accepted)
    next_candidate = jnp.where(attempted_delivery, jnp.int32(-1), cue)
    next_count = jnp.where(attempted_delivery, jnp.int32(0), cue_count)
    next_work_col = jnp.where(attempted_delivery, jnp.int32(1), next_work_col)
    next_committed = jnp.where(attempted_delivery, jnp.int32(0), next_committed)
    pending_delivery = jnp.where(attempted_delivery, jnp.int32(1),
                                 pending_delivery)
    next_memory = jnp.asarray([m[0] + 1, next_accepted, next_candidate,
                               next_count, next_work_col, next_committed,
                               mapping_mode, last_symbol, natural_same,
                               probe_same, pending_delivery], dtype=jnp.int32)
    trace = {
        "position": pos,
        "recipe": next_accepted,
        "cue": cue,
        "mapping": jnp.asarray([mapping_mode, natural_same, probe_same],
                               dtype=jnp.int32),
        "pot": jnp.asarray([pot_count, pot_timer, pot_cooked,
                            selected_pot_col, service_col], dtype=jnp.int32),
        "inventory": inv,
    }
    return action, next_memory, trace


def init_agent_1():
    # step, currently displayed recipe, active handoff counter column
    return jnp.asarray([0, -1, 2], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    m = jnp.where(reset, init_agent_1(), memory)
    pos = _locate(obs, LOWER_POS)
    count0 = jnp.sum(obs[:, :, 40])
    count1 = jnp.sum(obs[:, :, 41])
    observed_recipe = jnp.where(count0 > 0, jnp.int32(0),
                                jnp.where(count1 > 0, jnp.int32(1), m[1]))
    indexed_col = jnp.where(observed_recipe == 0, jnp.int32(1), jnp.int32(3))
    signal_target = jnp.asarray([4, indexed_col], dtype=jnp.int32)
    at_target = jnp.all(pos == signal_target)
    faces_up = obs[2, 2, 1] > 0
    signal_action = jnp.where(
        at_target,
        jnp.where(faces_up, STAY, UP),
        _navigate_horizontal_first(pos, signal_target),
    ).astype(jnp.int32)

    # Respond to concrete work placed on either shared counter. This supports
    # partners that use a handoff workflow, while the ordinary self-play path
    # remains the passive signal above.
    counter2 = _global_cell_block(obs, pos, 3, 2, 32, 38)
    counter4 = _global_cell_block(obs, pos, 3, 4, 32, 38)
    preferred_col = jnp.where(observed_recipe == 0, jnp.int32(2), jnp.int32(4))
    preferred_item = jnp.where(preferred_col == 2, counter2, counter4)
    alternate_item = jnp.where(preferred_col == 2, counter4, counter2)
    alternate_col = jnp.where(preferred_col == 2, jnp.int32(4), jnp.int32(2))
    recipe_slot = jnp.int32(2) + observed_recipe
    preferred_raw = (preferred_item[recipe_slot] > 0) & (jnp.sum(preferred_item[2:]) == 1)
    alternate_raw = (alternate_item[recipe_slot] > 0) & (jnp.sum(alternate_item[2:]) == 1)

    other_col = jnp.where(indexed_col == 1, jnp.int32(3), jnp.int32(1))
    indexed_pot = _global_cell_block(obs, pos, 3, indexed_col, 32, 38)
    other_pot = _global_cell_block(obs, pos, 3, other_col, 32, 38)
    indexed_wrong = (jnp.sum(indexed_pot[2:]) -
                     indexed_pot[2 + observed_recipe]) > 0
    other_wrong = (jnp.sum(other_pot[2:]) -
                   other_pot[2 + observed_recipe]) > 0
    use_other_pot = indexed_wrong & (~other_wrong)
    work_col = jnp.where(use_other_pot, other_col, indexed_col)
    pot_item = jnp.where(use_other_pot, other_pot, indexed_pot)
    pot_ready = pot_item[1] > 0
    pot_recipe_ready = pot_ready & (pot_item[2 + observed_recipe] == 3) & \
        (jnp.sum(pot_item[2:]) == 3)
    pot_accepts_raw = (jnp.sum(pot_item[2:]) < 3) & (~pot_ready)
    preferred_plate = (preferred_item[0] > 0) & (~(preferred_item[1] > 0))
    alternate_plate = (alternate_item[0] > 0) & (~(alternate_item[1] > 0))
    take_preferred = (pot_accepts_raw & preferred_raw) | \
        (pot_recipe_ready & preferred_plate)
    take_alternate = (~take_preferred) & \
        ((pot_accepts_raw & alternate_raw) | (pot_recipe_ready & alternate_plate))
    has_handoff = take_preferred | take_alternate
    handoff_col = jnp.where(take_alternate, alternate_col, preferred_col)
    handoff_target = jnp.asarray([4, handoff_col], dtype=jnp.int32)
    pickup_action = _use_station(obs, pos, handoff_target, UP, 1)

    inv = obs[2, 2, 5:11]
    holds_plate = inv[0] > 0
    holds_cooked = inv[1] > 0
    holds_target_raw = inv[2 + observed_recipe] > 0
    holds_any_raw = jnp.sum(inv[2:]) > 0
    pot_target = jnp.asarray([4, work_col], dtype=jnp.int32)
    pot_action = _use_station(obs, pos, pot_target, UP, 1)
    return_target = jnp.asarray([4, m[2]], dtype=jnp.int32)
    return_action = _use_station(obs, pos, return_target, UP, 1)
    raw_action = jnp.where(holds_target_raw & pot_accepts_raw,
                           pot_action, return_action)
    plate_action = jnp.where(pot_recipe_ready, pot_action, return_action)
    action = jnp.where(
        holds_cooked, return_action,
        jnp.where(holds_plate, plate_action,
                  jnp.where(holds_any_raw, raw_action,
                  jnp.where(has_handoff, pickup_action, signal_action)),
        ),
    ).astype(jnp.int32)
    beginning_pickup = (jnp.sum(inv) == 0) & has_handoff
    next_source = jnp.where(beginning_pickup, handoff_col, m[2])
    next_memory = jnp.asarray([m[0] + 1, observed_recipe, next_source], dtype=jnp.int32)
    trace = {"position": pos, "recipe": observed_recipe,
             "handoff": jnp.asarray([has_handoff, next_source], dtype=jnp.int32),
             "work_pot": work_col}
    return action, next_memory, trace
