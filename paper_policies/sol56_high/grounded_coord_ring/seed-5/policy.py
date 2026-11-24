"""Grounded reactive policy for the fixed coordination-ring layout."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_ring_controller",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}


# Positions are ordered clockwise around each seat's ring. This makes shortest
# path routing a small amount of fixed-shape integer arithmetic.
_INNER_POS = (
    (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (4, 3),
)
_INNER_CW = (0, 0, 1, 1, 2, 2, 3, 3)
_INNER_CCW = (1, 2, 2, 3, 3, 0, 0, 1)

_OUTER_POS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
    (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
    (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
)
_OUTER_CW = (
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
)
_OUTER_CCW = (
    1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
)

# Static tile codes corresponding to observation channels 20:29. Crops of
# this known map let the outer player localize without dead reckoning.
_TILES = (
    (1, 1, 1, 9, 4, 9, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 1, 1, 5, 1, 1, 0, 1),
    (9, 0, 7, 0, 0, 0, 6, 0, 9),
    (4, 0, 2, 0, 3, 0, 2, 0, 4),
    (9, 0, 8, 0, 0, 0, 6, 0, 9),
    (1, 0, 1, 1, 5, 1, 1, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 1, 1, 9, 4, 9, 1, 1, 1),
)
_TURN_ACTION = (3, 1, 0, 2)  # U,D,R,L


def _ring_route(index, target, facing, desired_facing, cw_actions, ccw_actions):
    """Go to a ring index, turn against the adjacent object, then interact."""
    cw = jnp.asarray(cw_actions, dtype=jnp.int32)
    ccw = jnp.asarray(ccw_actions, dtype=jnp.int32)
    turn_action = jnp.asarray(_TURN_ACTION, dtype=jnp.int32)
    n = jnp.asarray(len(cw_actions), dtype=jnp.int32)
    clockwise_distance = (target - index) % n
    move = jnp.where(
        clockwise_distance <= (n // 2), cw[index], ccw[index]
    )
    at_target = index == target
    pose_action = jnp.where(facing == desired_facing, 5, turn_action[desired_facing])
    return jnp.where(at_target, pose_action, move)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _displayed_recipe(obs, old_recipe):
    counts = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    seen = jnp.max(counts) > 0
    observed = jnp.argmax(counts).astype(jnp.int32)
    return jnp.where(seen, observed, old_recipe), seen


def _outer_index(obs):
    tiles = jnp.asarray(_TILES, dtype=jnp.int32)
    padded_tiles = jnp.pad(tiles, ((2, 2), (2, 2)))
    templates = jnp.stack(
        [padded_tiles[r:r + 5, c:c + 5] for r, c in _OUTER_POS]
    )
    weights = jnp.arange(1, 10, dtype=jnp.int32)
    observed_tiles = jnp.sum(obs[:, :, 20:29] * weights, axis=-1)
    mismatch = jnp.sum(templates != observed_tiles[None, :, :], axis=(1, 2))
    return jnp.argmin(mismatch).astype(jnp.int32)


def _item_at_global(obs, pos, row, col):
    lr = jnp.asarray(row, dtype=jnp.int32) - pos[0] + 2
    lc = jnp.asarray(col, dtype=jnp.int32) - pos[1] + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    item = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), 29:34]
    return item, visible


def init_agent_0():
    # elapsed step, mode, remembered recipe, alternating scan side.
    # Modes: 0 monitor, 1 raw stage, 2/3 scan upper/lower right counter,
    # 4/5 retrieve upper/lower left counter.
    return jnp.asarray([0, 0, -1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_0()
    mem = jnp.where(reset, initial, memory)
    step, mode, old_recipe, scan_side = mem

    index = _outer_index(obs)
    outer_pos = jnp.asarray(_OUTER_POS, dtype=jnp.int32)
    pos = outer_pos[index]
    facing = _facing(obs)
    recipe, recipe_seen = _displayed_recipe(obs, old_recipe)

    held_counts = obs[2, 2, 7:10]
    held_raw = jnp.sum(held_counts) > 0
    held_cooked = obs[2, 2, 6] > 0
    safe_recipe = jnp.clip(recipe, 0, 2)
    held_matches = ((recipe >= 0) & held_cooked
                    & (jnp.sum(held_counts) == 3)
                    & (held_counts[safe_recipe] == 3))

    # A cooked dish handed off by any partner is taken to the right goal.
    delivery_action = _ring_route(
        index, jnp.asarray(9, jnp.int32), facing, jnp.asarray(3, jnp.int32),
        _OUTER_CW, _OUTER_CCW,
    )

    # Grounded ingredient convention: ingredient 0 uses the upper shared
    # counter (2,3); ingredient 1 uses the lower shared counter (6,3).
    use_top = recipe == 0
    counter_row = jnp.where(use_top, 2, 6)
    counter_item, counter_visible = _item_at_global(obs, pos, counter_row, 3)
    counter_empty = jnp.sum(counter_item) == 0
    counter_cooked = counter_item[1] > 0

    upper_left, upper_left_visible = _item_at_global(obs, pos, 2, 3)
    lower_left, lower_left_visible = _item_at_global(obs, pos, 6, 3)
    upper_left_matches = (upper_left_visible & (upper_left[1] > 0)
                          & (jnp.sum(upper_left[2:5]) == 3)
                          & (upper_left[2 + safe_recipe] == 3))
    lower_left_matches = (lower_left_visible & (lower_left[1] > 0)
                          & (jnp.sum(lower_left[2:5]) == 3)
                          & (lower_left[2 + safe_recipe] == 3))

    monitor_index = jnp.asarray(21, dtype=jnp.int32)  # (4,1)
    pickup_index = jnp.where(use_top, 22, 20).astype(jnp.int32)
    deposit_index = jnp.where(use_top, 2, 16).astype(jnp.int32)
    deposit_facing = jnp.where(use_top, 1, 0).astype(jnp.int32)

    monitor_action = _ring_route(
        index, monitor_index, facing, jnp.asarray(3, jnp.int32),
        _OUTER_CW, _OUTER_CCW,
    )
    pickup_action = _ring_route(
        index, pickup_index, facing, jnp.asarray(2, jnp.int32),
        _OUTER_CW, _OUTER_CCW,
    )
    deposit_action = _ring_route(
        index, deposit_index, facing, deposit_facing,
        _OUTER_CW, _OUTER_CCW,
    )

    at_monitor = index == monitor_index
    may_start = at_monitor & (recipe >= 0) & counter_visible & counter_empty
    idle_choice = jnp.where(
        upper_left_matches, 4,
        jnp.where(lower_left_matches, 5,
                  jnp.where(may_start, 1, 2 + scan_side)),
    ).astype(jnp.int32)
    active_mode = jnp.where((mode == 0) & at_monitor, idle_choice, mode)

    # Resolve the counter pose and observed item for courier modes.
    courier_top = (active_mode == 2) | (active_mode == 4)
    courier_right = (active_mode == 2) | (active_mode == 3)
    courier_target = jnp.where(
        courier_top,
        jnp.where(courier_right, 4, 2),
        jnp.where(courier_right, 14, 16),
    ).astype(jnp.int32)
    courier_facing = jnp.where(courier_top, 1, 0).astype(jnp.int32)
    courier_row = jnp.where(courier_top, 2, 6)
    courier_col = jnp.where(courier_right, 5, 3)
    courier_item, courier_visible = _item_at_global(
        obs, pos, courier_row, courier_col
    )
    courier_matches = (courier_visible & (recipe >= 0)
                       & (courier_item[1] > 0)
                       & (jnp.sum(courier_item[2:5]) == 3)
                       & (courier_item[2 + safe_recipe] == 3))
    courier_route = _ring_route(
        index, courier_target, facing, courier_facing,
        _OUTER_CW, _OUTER_CCW,
    )
    at_courier_target = index == courier_target
    courier_action = jnp.where(
        at_courier_target & (~courier_matches), 4, courier_route
    )

    stage_action = jnp.where(held_raw, deposit_action, pickup_action)
    mode_action = jnp.where(active_mode == 1, stage_action, monitor_action)
    in_courier_mode = active_mode >= 2
    mode_action = jnp.where(in_courier_mode, courier_action, mode_action)

    # Re-check the public recipe before delivery. If it changed while a dish
    # was in transit, return the dish to its source counter instead of taking
    # a -20 wrong-delivery penalty.
    source_item_empty = jnp.sum(courier_item) == 0
    return_action = jnp.where(source_item_empty, courier_route, 4)
    cooked_action = jnp.where(held_matches, delivery_action, return_action)
    action = jnp.where(held_cooked, cooked_action, mode_action)

    delivered = held_matches & (delivery_action == 5)
    deposited = held_raw & (deposit_action == 5) & counter_empty
    returned = held_cooked & (~held_matches) & (return_action == 5)
    failed_scan = (in_courier_mode & (~held_cooked) & at_courier_target
                   & (~courier_matches))
    next_mode = jnp.where(deposited | delivered | returned | failed_scan,
                          0, active_mode)
    next_scan_side = jnp.where(failed_scan, 1 - scan_side, scan_side)
    next_memory = jnp.asarray(
        [step + 1, next_mode, recipe, next_scan_side], dtype=jnp.int32
    )
    trace = {
        "index": index,
        "mode": next_mode,
        "recipe": recipe,
        "recipe_seen": recipe_seen,
        "counter_empty": counter_empty,
        "counter_cooked": counter_cooked,
        "courier_matches": courier_matches,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # elapsed step, remembered recipe, correct deliveries, wrong clears,
    # whether a late-cook button refresh has armed the next recipe display.
    return jnp.asarray([0, -1, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_1()
    mem = jnp.where(reset, initial, memory)
    step, old_recipe, deliveries, wrong_clears, armed_i = mem
    armed = armed_i > 0

    facing = _facing(obs)
    pot_flat = jnp.argmax(obs[:, :, 22]).astype(jnp.int32)
    pot_lr, pot_lc = pot_flat // 5, pot_flat % 5
    global_pos = jnp.asarray([6 - pot_lr, 6 - pot_lc], dtype=jnp.int32)
    inner_pos = jnp.asarray(_INNER_POS, dtype=jnp.int32)
    distances = jnp.sum(jnp.abs(inner_pos - global_pos[None, :]), axis=1)
    index = jnp.argmin(distances).astype(jnp.int32)

    recipe, recipe_seen = _displayed_recipe(obs, old_recipe)
    pot_counts = obs[pot_lr, pot_lc, 31:34]
    pot_total = jnp.sum(pot_counts)
    pot_cooked = obs[pot_lr, pot_lc, 30] > 0
    pot_timer = obs[pot_lr, pot_lc, 39]
    pot_cooking = pot_timer > 0

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_counts = obs[2, 2, 7:10]
    held_raw = jnp.sum(held_counts) > 0
    empty_plate = held_plate & (~held_cooked)

    # All target poses are on the inner ring and face an adjacent fixed object.
    button_action = _ring_route(
        index, jnp.asarray(1, jnp.int32), facing, jnp.asarray(0, jnp.int32),
        _INNER_CW, _INNER_CCW,
    )
    dominant = jnp.argmax(pot_counts).astype(jnp.int32)
    safe_recipe_index = jnp.clip(recipe, 0, 2)
    desired_ingredient = jnp.where(
        (pot_total > 0) & (pot_counts[safe_recipe_index] == 0), dominant, recipe
    )
    ingredient_index = jnp.where(desired_ingredient == 0, 0, 6).astype(jnp.int32)
    stage_row = jnp.where(desired_ingredient == 0, 2, 6)
    stage_item, stage_visible = _item_at_global(obs, global_pos, stage_row, 3)
    stage_is_raw = ((stage_item[0] == 0) & (stage_item[1] == 0)
                    & (jnp.sum(stage_item[2:5]) == 1))
    staged_correct = (stage_visible & stage_is_raw
                      & (stage_item[2 + jnp.clip(desired_ingredient, 0, 2)] == 1))
    # The unlimited supply is adjacent at this same pose and already aligned
    # with the arrival direction, so it is always at least one step faster.
    ingredient_facing = jnp.asarray(3, dtype=jnp.int32)
    ingredient_action = _ring_route(
        index, ingredient_index, facing, ingredient_facing,
        _INNER_CW, _INNER_CCW,
    )
    held_kind = jnp.argmax(held_counts).astype(jnp.int32)
    insert_index = jnp.where(held_kind == 0, 1, 5).astype(jnp.int32)
    insert_facing = jnp.where(held_kind == 0, 1, 0).astype(jnp.int32)
    pot_insert_action = _ring_route(
        index, insert_index, facing, insert_facing,
        _INNER_CW, _INNER_CCW,
    )
    plate_action = _ring_route(
        index, jnp.asarray(2, jnp.int32), facing, jnp.asarray(2, jnp.int32),
        _INNER_CW, _INNER_CCW,
    )
    pot_pickup_action = _ring_route(
        index, jnp.asarray(3, jnp.int32), facing, jnp.asarray(3, jnp.int32),
        _INNER_CW, _INNER_CCW,
    )
    delivery_action = _ring_route(
        index, jnp.asarray(3, jnp.int32), facing, jnp.asarray(2, jnp.int32),
        _INNER_CW, _INNER_CCW,
    )

    cook_active = pot_cooking | pot_cooked
    public_armed = (cook_active & (~armed) & recipe_seen
                    & ((pot_timer <= 6) | pot_cooked))
    effective_armed = armed | public_armed
    refresh_needed = cook_active & (~effective_armed) & (step <= 354)
    at_button_pose = (index == 1) & (facing == 0)
    too_early = pot_cooking & (pot_timer > 6) & at_button_pose
    refresh_action = jnp.where(too_early, 4, button_action)
    cook_action = jnp.where(refresh_needed, refresh_action, plate_action)
    empty_action = jnp.where(cook_active, cook_action, ingredient_action)
    late_unknown_action = jnp.where(step < 360, button_action, 4)
    empty_action = jnp.where(recipe < 0, late_unknown_action, empty_action)

    action = empty_action
    action = jnp.where(held_raw, pot_insert_action, action)
    action = jnp.where(empty_plate, pot_pickup_action, action)
    action = jnp.where(held_cooked, delivery_action, action)

    delivering = held_cooked & (delivery_action == 5)
    held_total = jnp.sum(held_counts)
    safe_match_index = jnp.clip(recipe, 0, 2)
    dish_matches = ((recipe >= 0) & (held_total == 3)
                    & (held_counts[safe_match_index] == 3))
    correct_delivery = delivering & dish_matches
    wrong_delivery = delivering & (~dish_matches)
    next_recipe = jnp.where(correct_delivery, -1, recipe)
    armed_now = (refresh_needed & (refresh_action == 5)
                 & (~held_plate) & (~held_raw) & (~held_cooked))
    next_armed = jnp.where(delivering, False, effective_armed | armed_now)
    next_memory = jnp.asarray(
        [step + 1, next_recipe, deliveries + correct_delivery,
         wrong_clears + wrong_delivery, next_armed],
        dtype=jnp.int32,
    )
    trace = {
        "index": index,
        "recipe": recipe,
        "recipe_seen": recipe_seen,
        "pot_total": pot_total,
        "pot_timer": pot_timer,
        "pot_cooked": pot_cooked,
        "staged_correct": staged_correct,
        "armed": armed,
        "deliveries": deliveries,
        "wrong_clears": wrong_clears,
    }
    return action.astype(jnp.int32), next_memory, trace
