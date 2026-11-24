"""Landmark-driven policy for the separated test_time_simple kitchen."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'boundary_marker_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


def init_agent_0():
    # [last recipe, elapsed steps, protocol established]
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    """Maintain an intuitive upper/lower recipe marker on column 3."""
    memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    recipe_counts = jnp.sum(obs[:, :, 36:38], axis=(0, 1))
    recipe_type = jnp.argmax(recipe_counts)

    # The always-on R is at global (2, 0). Both signal counters and the
    # corresponding ingredient-2 supply can be addressed relative to it.
    recipe_row, recipe_col = _object_position(obs, 23)
    top_counter_row, bottom_counter_row = recipe_row - 1, recipe_row + 1
    counter_col = recipe_col + 3
    top_present = obs[top_counter_row, counter_col, 33] > 0
    bottom_present = obs[bottom_counter_row, counter_col, 33] > 0
    target_present = jnp.where(recipe_type == 0, top_present, bottom_present)
    wrong_present = jnp.where(recipe_type == 0, bottom_present, top_present)

    target_floor_row = recipe_row + (2 * recipe_type - 1)
    target_floor_col = recipe_col + 2
    wrong_floor_row = recipe_row + (1 - 2 * recipe_type)
    held_marker = obs[2, 2, 9] > 0

    get_supply = _navigate_to_square(
        obs, target_floor_row, target_floor_col,
        recipe_type, jnp.where(recipe_type == 0, 3, 1))
    put_target = _navigate_to_square(
        obs, target_floor_row, target_floor_col,
        jnp.asarray(2, jnp.int32), jnp.asarray(0, jnp.int32))
    get_wrong = _navigate_to_square(
        obs, wrong_floor_row, target_floor_col,
        jnp.asarray(2, jnp.int32), jnp.asarray(0, jnp.int32))

    action = jnp.where(
        held_marker, put_target,
        jnp.where(wrong_present, get_wrong,
                  jnp.where(target_present, 4, get_supply)))

    protocol_established = memory[2] | top_present | bottom_present
    next_memory = jnp.stack(
        (recipe_type, memory[1] + 1, protocol_established)).astype(jnp.int32)
    return action.astype(jnp.int32), next_memory, {
        'recipe_type': recipe_type.astype(jnp.int32),
        'marker_ready': target_present,
    }


def init_agent_1():
    # [dish type, elapsed, signal timeout, protocol seen,
    #  latched cache slot, cache ingredient type, cached item count]
    return jnp.asarray([0, 0, 8, 0, 0, 1, 0], dtype=jnp.int32)


def _object_position(obs, channel):
    flat = jnp.argmax(obs[:, :, channel].reshape(-1))
    return flat // 5, flat % 5


def _navigate_to_object(obs, channel, approach_from_right):
    """Move to the traversable neighbor of a blocking kitchen object."""
    row, col = _object_position(obs, channel)
    desired_row = row
    desired_col = col + jnp.where(approach_from_right, 1, -1)
    dr = desired_row - 2
    dc = desired_col - 2

    # Vertical-first paths are valid throughout the two-column right room.
    move = jnp.where(dr < 0, 3, jnp.where(dr > 0, 1,
                    jnp.where(dc < 0, 2, jnp.where(dc > 0, 0, 4))))

    # At the interaction square, first face the object, then interact.
    object_move = jnp.where(approach_from_right, 2, 0)
    required_facing = jnp.where(approach_from_right, 3, 2)  # LEFT / RIGHT
    facing = jnp.argmax(obs[2, 2, 1:5])
    at_square = (dr == 0) & (dc == 0)
    ready = at_square & (facing == required_facing)
    return jnp.where(ready, 5, jnp.where(at_square, object_move, move))


def _navigate_to_square(obs, desired_row, desired_col, required_facing, face_action):
    """Navigate to a local-indexed floor square, then interact in a direction."""
    dr = desired_row - 2
    dc = desired_col - 2
    move = jnp.where(dr < 0, 3, jnp.where(dr > 0, 1,
                    jnp.where(dc < 0, 2, jnp.where(dc > 0, 0, 4))))
    facing = jnp.argmax(obs[2, 2, 1:5])
    at_square = (dr == 0) & (dc == 0)
    return jnp.where(
        at_square & (facing == required_facing),
        5,
        jnp.where(at_square, face_action, move),
    )


def _move_to_square(desired_row, desired_col):
    """Move to a local-indexed floor square and stay once there."""
    dr = desired_row - 2
    dc = desired_col - 2
    return jnp.where(dr < 0, 3, jnp.where(dr > 0, 1,
                     jnp.where(dc < 0, 2, jnp.where(dc > 0, 0, 4))))


def _inner_counter_actions(obs, ingredient_type):
    """Actions/occupancy for the upper and lower column-5 counters."""
    pot_row, pot_col = _object_position(obs, 22)
    floor_col = pot_col + 1
    top_floor = pot_row - 1
    bottom_floor = pot_row + 1
    top_item_row = jnp.clip(pot_row - 2, 0, 4)
    bottom_item_row = jnp.clip(pot_row + 2, 0, 4)
    top_visible = pot_row >= 2
    bottom_visible = pot_row <= 2
    top_occupied = top_visible & (
        obs[top_item_row, floor_col, 31 + ingredient_type] > 0)
    bottom_occupied = bottom_visible & (
        obs[bottom_item_row, floor_col, 31 + ingredient_type] > 0)
    top_action = _navigate_to_square(
        obs, top_floor, floor_col,
        jnp.asarray(0, jnp.int32), jnp.asarray(3, jnp.int32))
    bottom_action = _navigate_to_square(
        obs, bottom_floor, floor_col,
        jnp.asarray(1, jnp.int32), jnp.asarray(1, jnp.int32))
    return top_action, bottom_action, top_occupied, bottom_occupied


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_1(), memory)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:10]) > 0

    pot_row, pot_col = _object_position(obs, 22)
    pot_total = jnp.sum(obs[pot_row, pot_col, 31:34])
    pot_cooked = obs[pot_row, pot_col, 30] > 0

    # Ingredient 2 is never in a recipe, so one on the inaccessible column-3
    # counters is unambiguous communication. Its row mirrors the two supplies.
    marker_present = jnp.sum(obs[:, :, 33]) > 0
    marker_flat = jnp.argmax(obs[:, :, 33].reshape(-1))
    marker_row = marker_flat // 5
    marker_type = (marker_row > pot_row).astype(jnp.int32)
    at_center = (pot_row == 2) & ((pot_col + 1) == 2)
    protocol_seen = memory[3] > 0
    signal_available = marker_present | protocol_seen
    signal_ready = at_center & signal_available
    # With two recipes, disappearance of an established persistent marker
    # means the signaler is carrying it to the opposite row. memory[0] is the
    # provisional opposite selected immediately after the last delivery.
    signaled_type = jnp.where(marker_present, marker_type, memory[0])
    safe_to_choose = (
        ((memory[2] == 0) | signal_ready) & (pot_total == 0) &
        (~held_plate) & (~held_cooked) & (~held_raw)
    )
    ingredient_type = jnp.where(
        safe_to_choose & signal_ready, signaled_type, memory[0])

    ingredient_channel = 26 + ingredient_type
    go_supply = _navigate_to_object(obs, ingredient_channel, False)
    go_pot = _navigate_to_object(obs, 22, True)
    go_delivery = _navigate_to_object(obs, 21, False)
    wait_at_center = _move_to_square(pot_row, pot_col + 1)

    next_type = 1 - ingredient_type
    pot_full = (pot_total >= 3) | pot_cooked
    cache_count = memory[6]
    cache_type = memory[5]
    active_cache_type = jnp.where(
        pot_full & (cache_count == 0), next_type, cache_type)
    cache_is_current = (cache_count > 0) & (cache_type == ingredient_type)
    staged_current = jnp.where(cache_is_current, cache_count, 0)
    staged_next = jnp.where(active_cache_type == next_type, cache_count, 0)

    cur_top_act, cur_bottom_act, cur_top, cur_bottom = _inner_counter_actions(
        obs, ingredient_type)
    cache_top_act, cache_bottom_act, _, _ = _inner_counter_actions(
        obs, active_cache_type)

    observed_retrieve_slot = jnp.where(cur_top, 0, 1)
    retrieve_slot = jnp.where(at_center, observed_retrieve_slot, memory[4])
    go_staged_current = jnp.where(
        retrieve_slot == 0, cur_top_act, cur_bottom_act)
    go_stage = jnp.where(
        memory[4] == 0, cache_top_act, cache_bottom_act)

    top_plate = _navigate_to_square(
        obs, pot_row - 1, pot_col + 1,
        jnp.asarray(3, jnp.int32), jnp.asarray(2, jnp.int32))
    bottom_plate = _navigate_to_square(
        obs, pot_row + 1, pot_col + 1,
        jnp.asarray(3, jnp.int32), jnp.asarray(2, jnp.int32))
    go_plate = jnp.where(memory[4] == 0, top_plate, bottom_plate)

    empty_hand_job = jnp.where(
        pot_full,
        jnp.where(cache_count < 2,
                  _navigate_to_object(obs, 26 + active_cache_type, False),
                  go_plate),
        jnp.where(cache_is_current, go_staged_current, go_supply),
    )

    # Inventory determines the immediate job; otherwise pot state does.
    normal_action = jnp.where(
        held_cooked,
        go_delivery,
        jnp.where(
            held_plate,
            go_pot,
            jnp.where(held_raw, jnp.where(pot_full, go_stage, go_pot), empty_hand_job),
        ),
    )
    waiting = (memory[2] > 0) & (~signal_ready)
    action = jnp.where(waiting, wait_at_center, normal_action)

    # Delivery always consumes a cooked dish. Alternate even after a wrong
    # delivery: the unchanged recipe then makes the opposite dish certain.
    delivering = held_cooked & (action == 5)
    next_dish_type = jnp.where(delivering, next_type, ingredient_type)
    next_cooldown = jnp.where(
        delivering, 8,
        jnp.where(signal_ready, 0, jnp.maximum(memory[2] - 1, 0)))
    next_protocol_seen = protocol_seen | (at_center & marker_present)

    empty_hands = (~held_plate) & (~held_cooked) & (~held_raw)
    planning_stage = pot_full & empty_hands & (cache_count < 2)
    planned_stage_slot = jnp.where(
        cache_count == 0, active_cache_type, 1 - active_cache_type)
    planning_retrieve = (~pot_full) & empty_hands & cache_is_current
    next_cache_slot = jnp.where(
        planning_stage, planned_stage_slot,
        jnp.where(planning_retrieve, retrieve_slot, memory[4]))

    staging = held_raw & pot_full & (action == 5) & (~waiting)
    retrieving = empty_hands & (~pot_full) & cache_is_current & (action == 5) & (~waiting)
    next_cache_count = jnp.clip(
        cache_count + staging.astype(jnp.int32) - retrieving.astype(jnp.int32),
        0, 2)
    next_cache_type = jnp.where(
        pot_full & (cache_count == 0), active_cache_type, cache_type)
    next_memory = jnp.stack(
        (next_dish_type, memory[1] + 1, next_cooldown,
         next_protocol_seen, next_cache_slot, next_cache_type,
         next_cache_count)).astype(jnp.int32)
    return action.astype(jnp.int32), next_memory, {
        'dish_type': ingredient_type,
        'pot_total': pot_total.astype(jnp.int32),
        'job_action': action.astype(jnp.int32),
        'staged_current': staged_current,
        'staged_next': staged_next,
        'marker_present': marker_present,
        'cooldown': memory[2],
        'protocol_seen': protocol_seen,
        'at_center': at_center,
        'signal_ready': signal_ready,
        'safe_to_choose': safe_to_choose,
        'signaled_type': signaled_type,
    }
