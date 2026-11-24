"""Reactive, convention-free chef/courier policy for demo_cook_wide."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "reactive_chef_courier",
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

FACE_UP = 0
FACE_DOWN = 1
FACE_RIGHT = 2
FACE_LEFT = 3


def _static_map():
    """Known map in the observation's compact static channel order."""
    static = jnp.zeros((6, 11, 10), dtype=jnp.int32)
    wall_r = jnp.asarray(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        dtype=jnp.int32,
    )
    wall_c = jnp.asarray(
        [0, 1, 2, 3, 7, 8, 9, 10, 0, 1, 2, 8, 9, 10, 0, 1, 2, 3, 4,
         6, 7, 8, 9, 10, 0, 10, 3, 7, 0, 2, 3, 4, 5, 6, 7, 8, 10],
        dtype=jnp.int32,
    )
    static = static.at[wall_r, wall_c, 0].set(1)
    static = static.at[0, 5, 1].set(1)
    static = static.at[2, 5, 2].set(1)
    static = static.at[4, 5, 3].set(1)
    static = static.at[jnp.asarray([0, 0]), jnp.asarray([4, 6]), 5].set(1)
    static = static.at[jnp.asarray([1, 4, 4]), jnp.asarray([3, 0, 10]), 6].set(1)
    static = static.at[jnp.asarray([1, 5, 5]), jnp.asarray([7, 1, 9]), 7].set(1)
    static = static.at[jnp.asarray([4, 4]), jnp.asarray([4, 6]), 9].set(1)
    return static


def _static_crop(padded_static, position):
    return jax.lax.dynamic_slice(
        padded_static, (position[0], position[1], jnp.int32(0)), (5, 5, 10)
    )


def _locate(obs, candidates):
    padded_static = jnp.pad(_static_map(), ((2, 2), (2, 2), (0, 0)))
    templates = jax.vmap(_static_crop, in_axes=(None, 0))(padded_static, candidates)
    error = jnp.sum(jnp.abs(templates - obs[:, :, 22:32]), axis=(1, 2, 3))
    return candidates[jnp.argmin(error)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_action(desired):
    return jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)[desired]


def _a0_station(col, facing, target_col, desired_facing):
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    at_target = col == target_col
    operate = jnp.where(facing == desired_facing, INTERACT, _face_action(desired_facing))
    return jnp.where(at_target, operate, horizontal)


def _a1_row3_station(row, col, facing, target_col, desired_facing):
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    at_target = (row == 3) & (col == target_col)
    travel = jnp.where(row == 4, UP, horizontal)
    operate = jnp.where(facing == desired_facing, INTERACT, _face_action(desired_facing))
    return jnp.where(at_target, operate, travel)


def _a1_supply(row, col, facing, use_right, ingredient_type):
    """Navigate to (4,1)/(4,9), face the selected supply, and interact."""
    target_col = jnp.where(use_right, jnp.int32(9), jnp.int32(1))
    same_side_pocket = jnp.where(use_right, col >= 8, col <= 2)
    on_endpoint = (row == 4) & (col == target_col)
    from_row4 = jnp.where(
        same_side_pocket, jnp.where(col < target_col, RIGHT, LEFT), UP
    )
    corridor = jnp.where(
        col == target_col, DOWN, jnp.where(col < target_col, RIGHT, LEFT)
    )
    travel = jnp.where(row == 4, from_row4, corridor)
    desired = jnp.where(
        ingredient_type == 0,
        jnp.where(use_right, FACE_RIGHT, FACE_LEFT),
        FACE_DOWN,
    )
    operate = jnp.where(facing == desired, INTERACT, _face_action(desired))
    return jnp.where(on_endpoint, operate, travel)


def _pot_view(obs):
    mask = obs[:, :, 24]
    counts = jnp.sum(obs[:, :, 34:38] * mask[:, :, None], axis=(0, 1))
    total = jnp.sum(counts)
    cooked = jnp.sum(obs[:, :, 33] * mask) > 0
    timer = jnp.sum(obs[:, :, 44] * mask)
    return counts, total, cooked, timer


def _counter_item(obs, row, col, counter_col):
    """Item counts on global counter (2,counter_col), zero when out of view."""
    local_row = jnp.clip(4 - row, 0, 4)
    local_col = jnp.clip(counter_col - col + 2, 0, 4)
    visible = (jnp.abs(2 - row) <= 2) & (jnp.abs(counter_col - col) <= 2)
    return obs[local_row, local_col, 34:38] * visible.astype(jnp.int32)


def _initial_memory():
    # target ingredient, elapsed steps, contributed to current pot, last pot total
    return jnp.asarray([-1, 0, 0, 0], dtype=jnp.int32)


def init_agent_0():
    return _initial_memory()


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _initial_memory(), memory)
    positions = jnp.asarray([[1, 4], [1, 5], [1, 6]], dtype=jnp.int32)
    position = _locate(obs, positions)
    col = position[1]
    facing = _facing(obs)

    inv = obs[2, 2, 5:11]
    has_plate = inv[0] > 0
    has_cooked = inv[1] > 0
    has_raw = jnp.sum(inv[2:]) > 0
    raw_type = jnp.argmax(inv[2:]).astype(jnp.int32)
    pot_counts, pot_total, pot_cooked, pot_timer = _pot_view(obs)
    pot_type = jnp.argmax(pot_counts).astype(jnp.int32)

    partner_inventory_empty = jnp.sum(obs[:, :, 16:22]) == 0
    partner_counts = jnp.sum(obs[:, :, 18:22], axis=(0, 1))
    partner_total = jnp.sum(partner_counts)
    partner_type = jnp.argmax(partner_counts).astype(jnp.int32)
    valid_partner_hint = (partner_total > 0) & (partner_type < 2)
    valid_pot_hint = (pot_total > 0) & (pot_type < 2)
    # memory[3] marks a one-step tentative counter approach. If the partner
    # merely passes that counter, discard the hint before considering fresh
    # observable evidence.
    target = jnp.where(memory[3] > 0, jnp.int32(-1), memory[0])
    target = jnp.where(valid_partner_hint, partner_type, target)

    left_cache = _counter_item(obs, position[0], col, jnp.int32(4))
    right_cache = _counter_item(obs, position[0], col, jnp.int32(6))
    below_left = obs[4, 6 - col, 11] > 0
    below_right = obs[4, 8 - col, 11] > 0
    left_type = jnp.argmax(left_cache).astype(jnp.int32)
    right_type = jnp.argmax(right_cache).astype(jnp.int32)
    left_intent = (
        below_left & partner_inventory_empty & (jnp.sum(left_cache) > 0) &
        (left_type < 2)
    )
    right_intent = (
        below_right & partner_inventory_empty & (jnp.sum(right_cache) > 0) &
        (right_type < 2)
    )
    counter_intent = left_intent | right_intent
    counter_type = jnp.where(left_intent, left_type, right_type)
    target = jnp.where(counter_intent, counter_type, target)
    target = jnp.where(valid_pot_hint, pot_type, target)

    deliver_action = _a0_station(col, facing, jnp.int32(5), FACE_UP)
    pot_action = _a0_station(col, facing, jnp.int32(5), FACE_DOWN)
    plate_col = jnp.where(col <= 5, jnp.int32(4), jnp.int32(6))
    get_plate_action = _a0_station(col, facing, plate_col, FACE_UP)
    supply_col = jnp.where(target == 0, jnp.int32(4), jnp.int32(6))
    supply_face = jnp.where(target == 0, FACE_LEFT, FACE_RIGHT)
    supply_action = _a0_station(col, facing, supply_col, supply_face)
    wait_at_pot = jnp.where(col < 5, RIGHT, jnp.where(col > 5, LEFT, STAY))

    raw_acceptable = (
        (target >= 0) & (target < 2) & (raw_type == target) &
        (pot_timer == 0) & (~pot_cooked) & (pot_total < 3) &
        ((pot_total == 0) | (pot_counts[target] > 0))
    )
    raw_cache_col = jnp.where(raw_type == 0, jnp.int32(4), jnp.int32(6))
    raw_cache_counts = _counter_item(obs, position[0], col, raw_cache_col)
    raw_cache_empty = jnp.sum(raw_cache_counts) == 0
    surplus_drop_action = _a0_station(
        col, facing, raw_cache_col, FACE_DOWN
    )
    pot_unavailable = pot_cooked | (pot_timer > 0) | (pot_total >= 3)
    overflow_recovery = pot_unavailable & raw_cache_empty
    raw_action = jnp.where(
        raw_acceptable,
        pot_action,
        jnp.where(overflow_recovery, surplus_drop_action, wait_at_pot),
    )
    plate_wait = jnp.where(
        col < 5,
        RIGHT,
        jnp.where(
            col > 5,
            LEFT,
            jnp.where(facing == FACE_DOWN, STAY, _face_action(FACE_DOWN)),
        ),
    )
    held_plate_action = jnp.where(pot_cooked, pot_action, plate_wait)

    should_get_plate = pot_cooked | (pot_timer > 0)
    # The bottom pot workstation is global (3,5), hence local row 4 and
    # local column 7-col from this top corridor.
    partner_waiting_at_pot = obs[4, 7 - col, 11] > 0
    first_contribution = (memory[2] == 0) & (
        (pot_total == 1) |
        ((pot_total == 0) & (valid_partner_hint | counter_intent))
    )
    requested_final_contribution = (
        (memory[2] > 0) & (pot_total == 2) & partner_waiting_at_pot &
        partner_inventory_empty
    )
    should_contribute = (
        (target >= 0) & (target < 2) &
        (first_contribution | requested_final_contribution)
    )
    empty_action = jnp.where(
        should_get_plate,
        get_plate_action,
        jnp.where(should_contribute, supply_action, wait_at_pot),
    )
    action = jnp.where(
        has_cooked,
        deliver_action,
        jnp.where(has_plate, held_plate_action, jnp.where(has_raw, raw_action, empty_action)),
    ).astype(jnp.int32)

    successful_add_attempt = has_raw & raw_acceptable & (action == INTERACT)
    contributed = jnp.where((pot_total == 0) & (~has_raw), 0, memory[2])
    contributed = jnp.where(successful_add_attempt, 1, contributed)
    delivering = has_cooked & (action == INTERACT)
    next_target = jnp.where(delivering, -1, target)
    partner_faces_counter = (
        (left_intent & (obs[4, 6 - col, 12] > 0)) |
        (right_intent & (obs[4, 8 - col, 12] > 0))
    )
    pending_counter_hint = counter_intent & (~partner_faces_counter)
    pending_counter_hint = jnp.where(delivering, False, pending_counter_hint)
    next_memory = jnp.asarray(
        [next_target, memory[1] + 1, contributed,
         pending_counter_hint.astype(jnp.int32)],
        dtype=jnp.int32,
    )
    trace = {
        "action": action,
        "col": col,
        "mode": (has_raw.astype(jnp.int32) + 2 * has_plate.astype(jnp.int32)
                 + 4 * has_cooked.astype(jnp.int32)),
        "pot_total": pot_total.astype(jnp.int32),
        "target": next_target.astype(jnp.int32),
        "overflow": (has_raw & overflow_recovery).astype(jnp.int32),
    }
    return action, next_memory, trace


def init_agent_1():
    return _initial_memory()


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _initial_memory(), memory)
    positions = jnp.asarray(
        [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
         [3, 8], [3, 9], [4, 1], [4, 2], [4, 8], [4, 9]],
        dtype=jnp.int32,
    )
    position = _locate(obs, positions)
    row, col = position[0], position[1]
    facing = _facing(obs)

    inv = obs[2, 2, 5:11]
    has_raw = jnp.sum(inv[2:]) > 0
    raw_type = jnp.argmax(inv[2:]).astype(jnp.int32)
    recipe_counts = jnp.sum(obs[:, :, 40:44], axis=(0, 1))
    sees_recipe = jnp.sum(recipe_counts) > 0
    seen_target = jnp.argmax(recipe_counts).astype(jnp.int32)
    target = jnp.where(sees_recipe, seen_target, memory[0])
    pot_counts, pot_total, pot_cooked, pot_timer = _pot_view(obs)
    pot_visible = jnp.sum(obs[:, :, 24]) > 0
    pot_type = jnp.argmax(pot_counts).astype(jnp.int32)
    contaminated = (
        (target >= 0) & (target < 2) & (pot_total > 0) &
        (pot_counts[target] != pot_total) & (pot_timer == 0) & (~pot_cooked)
    )
    # Complete an irreversible bad partial pot with an available ingredient so
    # that one wrong delivery clears it instead of deadlocking forever.
    # Ingredient 3 has adjacent supplies at (4,4)/(4,6), so it is an explicit
    # nearby waste filler that never masquerades as a configured recipe.
    work_target = jnp.where(contaminated, jnp.int32(3), target)

    pot_action = _a1_row3_station(row, col, facing, jnp.int32(5), FACE_UP)
    center_wait = jnp.where(
        (row != 3) | (col != 5),
        _a1_row3_station(row, col, facing, jnp.int32(5), FACE_UP),
        STAY,
    )
    use_right = col > 5
    ordinary_supply = _a1_supply(row, col, facing, use_right, work_target)
    cleanup_col = jnp.where(col > 5, jnp.int32(6), jnp.int32(4))
    cleanup_supply = _a1_row3_station(
        row, col, facing, cleanup_col, FACE_DOWN
    )
    supply_action = jnp.where(contaminated, cleanup_supply, ordinary_supply)

    default_cache_col = jnp.where(target == 0, jnp.int32(4), jnp.int32(6))
    left_counter = _counter_item(obs, row, col, jnp.int32(4))
    right_counter = _counter_item(obs, row, col, jnp.int32(6))
    left_has_target = (
        (target >= 0) & (target < 2) & (left_counter[target] > 0)
    )
    right_has_target = (
        (target >= 0) & (target < 2) & (right_counter[target] > 0)
    )
    cache_has_target = left_has_target | right_has_target
    cache_col = jnp.where(
        left_has_target,
        jnp.int32(4),
        jnp.where(right_has_target, jnp.int32(6), default_cache_col),
    )
    cache_action = _a1_row3_station(row, col, facing, cache_col, FACE_UP)

    partner_counts = jnp.sum(obs[:, :, 18:22], axis=(0, 1))
    partner_total = jnp.sum(partner_counts)
    partner_type = jnp.argmax(partner_counts).astype(jnp.int32)
    partner_carries_target = (
        (partner_total > 0) & (partner_type == target) & (target >= 0) &
        (target < 2)
    )
    partial_pot = (pot_total > 0) & (pot_total < 3) & (pot_timer == 0)
    same_partial = partial_pot & (pot_total == memory[3])
    effective_wait = jnp.where(same_partial, memory[2], jnp.int32(0))
    wait_for_partner = partial_pot & (
        partner_carries_target | (effective_wait < 2)
    )

    raw_matches = (
        (raw_type == work_target) & (work_target >= 0) & (work_target < 4)
    )
    pot_matches = contaminated | (pot_total == 0) | (pot_counts[target] > 0)
    partner_holds_cooked = jnp.sum(obs[:, :, 17]) > 0
    can_add = (
        raw_matches & pot_matches & (pot_total < 3) & (pot_timer == 0) &
        (~pot_cooked) & (~partner_holds_cooked)
    )
    # Ingredient-specific counters make the staged items self-describing and
    # keep an unexpected surplus out of the next pot.
    preferred_raw_cache = jnp.where(
        raw_type == 0, jnp.int32(4), jnp.int32(6)
    )
    preferred_occupied = jnp.where(
        preferred_raw_cache == 4,
        jnp.sum(left_counter) > 0,
        jnp.sum(right_counter) > 0,
    )
    other_occupied = jnp.where(
        preferred_raw_cache == 4,
        jnp.sum(right_counter) > 0,
        jnp.sum(left_counter) > 0,
    )
    other_cache = jnp.where(
        preferred_raw_cache == 4, jnp.int32(6), jnp.int32(4)
    )
    raw_cache_col = jnp.where(
        preferred_occupied & (~other_occupied), other_cache, preferred_raw_cache
    )
    drop_action = _a1_row3_station(row, col, facing, raw_cache_col, FACE_UP)
    raw_action = jnp.where(
        ~pot_visible, pot_action, jnp.where(can_add, pot_action, drop_action)
    )

    needs_ingredient = contaminated | (
        (target >= 0) & (target < 2) & (pot_timer == 0) & (~pot_cooked) &
        (pot_total < 3) & ((pot_total == 0) | (pot_counts[target] > 0))
    )
    waiting_for_delivery = (pot_total == 0) & partner_holds_cooked
    use_cached_first = (
        (pot_total == 0) & cache_has_target & (~waiting_for_delivery)
    )
    refill_cache = (
        (target >= 0) & (target < 2) & (pot_cooked | (pot_timer > 0)) &
        (~cache_has_target)
    )
    idle_action = center_wait
    empty_action = jnp.where(
        waiting_for_delivery,
        idle_action,
        jnp.where(
            use_cached_first,
            cache_action,
            jnp.where(
                refill_cache | (needs_ingredient & (~wait_for_partner)),
                supply_action,
                idle_action,
            ),
        ),
    )
    action = jnp.where(has_raw, raw_action, empty_action).astype(jnp.int32)

    next_wait = jnp.where(same_partial, memory[2] + 1, jnp.int32(0))
    next_memory = jnp.asarray(
        [target, memory[1] + 1, next_wait, pot_total], dtype=jnp.int32
    )
    trace = {
        "action": action,
        "col": col,
        "mode": has_raw.astype(jnp.int32),
        "pot_total": pot_total.astype(jnp.int32),
        "row": row,
        "target": target.astype(jnp.int32),
        "wait": effective_wait.astype(jnp.int32),
        "cache": cache_has_target.astype(jnp.int32),
        "contaminated": contaminated.astype(jnp.int32),
    }
    return action, next_memory, trace
