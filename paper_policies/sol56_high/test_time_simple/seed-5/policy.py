"""Grounded, reactive policy for the two-room Overcooked layout."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_counter_signal",
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


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _go_interact(row, col, target_row, target_col, face, obs):
    """Navigate on floor to a fixed interaction stance, then interact."""
    face_action = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)[face]
    at_row = row == target_row
    at_col = col == target_col
    move_row = jnp.where(row < target_row, DOWN, UP)
    move_col = jnp.where(col < target_col, RIGHT, LEFT)
    finish = jnp.where(_facing(obs) == face, INTERACT, face_action)
    return jnp.where(at_row, jnp.where(at_col, finish, move_col), move_row)


def _go_wait(row, col, target_row, target_col, face, obs):
    """Navigate to a fixed stance and face its object without touching it."""
    face_action = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)[face]
    at_row = row == target_row
    at_col = col == target_col
    move_row = jnp.where(row < target_row, DOWN, UP)
    move_col = jnp.where(col < target_col, RIGHT, LEFT)
    finish = jnp.where(_facing(obs) == face, STAY, face_action)
    return jnp.where(at_row, jnp.where(at_col, finish, move_col), move_row)


def _left_position(obs):
    # The always-on recipe indicator at global (2, 0) is visible everywhere
    # in the left room, and therefore provides an unambiguous local landmark.
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    indicator = obs[:, :, 23]
    local_row = jnp.sum(indicator * rows)
    local_col = jnp.sum(indicator * cols)
    return jnp.int32(4) - local_row, jnp.int32(2) - local_col


def _right_position(obs):
    # Likewise the delivery station at global (2, 7) is always visible from
    # the right room.
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    goal = obs[:, :, 21]
    local_row = jnp.sum(goal * rows)
    local_col = jnp.sum(goal * cols)
    return jnp.int32(4) - local_row, jnp.int32(9) - local_col


def _global_item(obs, row, col, global_row, global_col):
    local_row = global_row - row + jnp.int32(2)
    local_col = global_col - col + jnp.int32(2)
    return obs[local_row, local_col, 29:34]


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    row, col = _left_position(obs)

    # Recipe 0 is physically upper on the partner side and recipe 1 lower.
    # Mark the corresponding upper/lower shared counter with ingredient 2.
    target = jnp.where(jnp.sum(obs[:, :, 36]) > 0, jnp.int32(0), jnp.int32(1))
    top_present = jnp.sum(_global_item(obs, row, col, jnp.int32(1), jnp.int32(3))) > 0
    middle_present = jnp.sum(_global_item(obs, row, col, jnp.int32(2), jnp.int32(3))) > 0
    bottom_present = jnp.sum(_global_item(obs, row, col, jnp.int32(3), jnp.int32(3))) > 0
    # The spare marker is parked outside the partner's view at (1, 0).
    stash_present = jnp.sum(_global_item(obs, row, col, jnp.int32(1), jnp.int32(0))) > 0
    holding = jnp.sum(obs[2, 2, 5:10]) > 0

    top_action = _go_interact(row, col, jnp.int32(1), jnp.int32(2), jnp.int32(2), obs)
    middle_action = _go_interact(row, col, jnp.int32(2), jnp.int32(2), jnp.int32(2), obs)
    bottom_action = _go_interact(row, col, jnp.int32(3), jnp.int32(2), jnp.int32(2), obs)
    stash_action = _go_interact(row, col, jnp.int32(1), jnp.int32(1), jnp.int32(3), obs)
    top_supply = _go_interact(row, col, jnp.int32(1), jnp.int32(2), jnp.int32(0), obs)
    bottom_supply = _go_interact(row, col, jnp.int32(3), jnp.int32(2), jnp.int32(1), obs)
    reinforce_top = _go_wait(row, col, jnp.int32(1), jnp.int32(2), jnp.int32(2), obs)
    reinforce_bottom = _go_wait(row, col, jnp.int32(3), jnp.int32(2), jnp.int32(2), obs)

    # Recipe 0: top marker, middle empty, spare hidden. Recipe 1: bottom
    # marker plus a conventional middle-counter bit, with top empty.
    put_zero = jnp.where(~top_present, top_action, stash_action)
    source_zero = jnp.where(
        bottom_present,
        bottom_action,
        jnp.where(middle_present, middle_action, jnp.where(stash_present, stash_action, top_supply)),
    )
    tidy_zero = jnp.where(
        middle_present,
        middle_action,
        jnp.where(bottom_present, bottom_action, reinforce_top),
    )
    empty_zero = jnp.where(~top_present, source_zero, tidy_zero)

    put_one = jnp.where(~bottom_present, bottom_action, jnp.where(~middle_present, middle_action, stash_action))
    source_one = jnp.where(top_present, top_action, jnp.where(stash_present, stash_action, bottom_supply))
    fill_one = jnp.where(~bottom_present, source_one, jnp.where(~middle_present, source_one, reinforce_bottom))
    tidy_one = jnp.where(top_present, top_action, reinforce_bottom)
    empty_one = jnp.where((~bottom_present) | (~middle_present), fill_one, tidy_one)

    holding_action = jnp.where(target == 0, put_zero, put_one)
    empty_action = jnp.where(target == 0, empty_zero, empty_one)
    action = jnp.where(holding, holding_action, empty_action).astype(jnp.int32)
    trace = {
        "target": target,
        "top_present": top_present,
        "middle_present": middle_present,
        "bottom_present": bottom_present,
        "stash_present": stash_present,
        "row": row,
        "col": col,
    }
    return action, memory, trace


def init_agent_1():
    # [target, re-arm state, protocol (0 unknown, 1 spatial, 2 binary,
    # 3 visible-marker count),
    # number of signal-free initial inspections, fast-reader trust bit]
    return jnp.asarray([-1, -1, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_1()
    memory = jnp.where(reset, initial, memory)
    old_target = memory[0]
    wait = memory[1]
    old_mode = memory[2]
    old_ticks = memory[3]
    old_trust = memory[4]
    row, col = _right_position(obs)

    # Shared counters at column 3 are visible only from column 5. Avoid
    # indexing outside the crop while at column 6 by clipping, then gate the
    # result on visibility.
    top = _global_item(obs, row, col, jnp.int32(1), jnp.int32(3))
    bottom = _global_item(obs, row, col, jnp.int32(3), jnp.int32(3))
    can_see_signal = col == 5
    top_marked = can_see_signal & (jnp.sum(top) > 0)
    bottom_marked = can_see_signal & (jnp.sum(bottom) > 0)
    signal = jnp.where(
        top_marked & ~bottom_marked,
        jnp.int32(0),
        jnp.where(bottom_marked & ~top_marked, jnp.int32(1), jnp.int32(-1)),
    )
    middle = _global_item(obs, row, col, jnp.int32(2), jnp.int32(3))
    middle_marked = can_see_signal & (jnp.sum(middle) > 0)

    inspecting = (row == 2) & (col == 5)
    has_spatial = signal >= 0
    count_pattern = top_marked & (middle_marked | bottom_marked)
    visible_count = top_marked.astype(jnp.int32) + middle_marked.astype(jnp.int32) + bottom_marked.astype(jnp.int32)
    next_ticks_unknown = old_ticks + jnp.int32(1)
    binary_timeout = next_ticks_unknown >= jnp.int32(14)
    spatial_ready = (bottom_marked & ~top_marked) | (top_marked & ~bottom_marked & ~middle_marked & binary_timeout)

    unknown_target = jnp.where(
        count_pattern,
        jnp.int32(1),
        jnp.where(spatial_ready, signal,
        jnp.where(middle_marked, jnp.int32(1), jnp.where(binary_timeout, jnp.int32(0), old_target)),
        ),
    )
    unknown_mode = jnp.where(
        count_pattern,
        jnp.int32(3),
        jnp.where(spatial_ready, jnp.int32(1), jnp.where(middle_marked | (binary_timeout & ~top_marked), jnp.int32(2), old_mode)),
    )
    unknown_wait = jnp.where(unknown_target >= 0, jnp.int32(0), jnp.int32(-1))

    # Spatial re-arming resolves in one inspection. A binary-only convention
    # needs a settling window because a reader may maintain another public
    # signal first; sample the final middle-bit state when that window ends.
    spatial_target = jnp.where(has_spatial, signal, jnp.int32(1) - old_target)
    spatial_changed = (~has_spatial) | (signal != old_target)
    spatial_wait = jnp.where(
        spatial_changed | (old_trust > 0),
        jnp.int32(0),
        jnp.maximum(wait - jnp.int32(1), jnp.int32(0)),
    )
    binary_wait = jnp.maximum(wait - jnp.int32(1), jnp.int32(0))
    sampled_binary_target = jnp.where(middle_marked, jnp.int32(1), jnp.int32(0))
    binary_target = jnp.where(binary_wait == 0, sampled_binary_target, old_target)
    sampled_count_target = jnp.where(visible_count >= 2, jnp.int32(1), jnp.int32(0))
    count_target = jnp.where(binary_wait == 0, sampled_count_target, old_target)

    promote_count = (old_mode == 1) & (wait != 0) & count_pattern
    known_target = jnp.where(promote_count, jnp.int32(1), jnp.where(old_mode == 1, spatial_target, jnp.where(old_mode == 2, binary_target, count_target)))
    known_wait = jnp.where(promote_count, jnp.int32(0), jnp.where(old_mode == 1, spatial_wait, binary_wait))
    inspected_target = jnp.where(old_mode == 0, unknown_target, jnp.where(wait != 0, known_target, old_target))
    inspected_wait = jnp.where(old_mode == 0, unknown_wait, jnp.where(wait != 0, known_wait, wait))
    inspected_mode = jnp.where(old_mode == 0, unknown_mode, jnp.where(promote_count, jnp.int32(3), old_mode))
    inspected_ticks = jnp.where(old_mode == 0, next_ticks_unknown, old_ticks)
    learned_fast = (old_mode == 1) & (wait == 3) & spatial_changed & ~promote_count
    inspected_trust = jnp.where(learned_fast, jnp.int32(1), old_trust)
    next_wait_pre = jnp.where(inspecting, inspected_wait, wait)
    next_target_pre = jnp.where(inspecting, inspected_target, old_target)
    next_mode_pre = jnp.where(inspecting, inspected_mode, old_mode)
    next_ticks_pre = jnp.where(inspecting, inspected_ticks, old_ticks)
    next_trust_pre = jnp.where(inspecting, inspected_trust, old_trust)
    # Use the just-resolved state so useful work can begin on the same call.
    rearming = (next_target_pre < 0) | (next_wait_pre != 0)
    inspect_action = _go_wait(row, col, jnp.int32(2), jnp.int32(5), jnp.int32(3), obs)

    target = next_target_pre
    self_item = obs[2, 2, 5:10]
    held_plate = self_item[0] > 0
    held_cooked = self_item[1] > 0
    held_raw = jnp.sum(self_item[2:]) > 0

    pot = _global_item(obs, row, col, jnp.int32(2), jnp.int32(4))
    pot_cooked = pot[1] > 0
    pot_count = jnp.sum(pot[2:])

    to_pot = _go_interact(row, col, jnp.int32(2), jnp.int32(5), jnp.int32(3), obs)
    wait_at_pot = _go_wait(row, col, jnp.int32(2), jnp.int32(5), jnp.int32(3), obs)
    to_goal = _go_interact(row, col, jnp.int32(2), jnp.int32(6), jnp.int32(2), obs)
    ingredient_row = jnp.where(target == 0, jnp.int32(1), jnp.int32(3))
    take_ingredient = _go_interact(row, col, ingredient_row, jnp.int32(6), jnp.int32(2), obs)
    take_plate = _go_interact(row, col, ingredient_row, jnp.int32(5), jnp.int32(3), obs)

    empty_action = jnp.where(pot_cooked | (pot_count >= 3), take_plate, take_ingredient)
    plate_action = jnp.where(pot_cooked, to_pot, wait_at_pot)
    work_action = jnp.where(
        held_cooked,
        to_goal,
        jnp.where(held_raw, to_pot, jnp.where(held_plate, plate_action, empty_action)),
    )
    action = jnp.where(rearming, inspect_action, work_action).astype(jnp.int32)

    delivering = (~rearming) & held_cooked & (action == INTERACT)
    next_target = next_target_pre
    delivery_wait = jnp.where(next_mode_pre == 1, jnp.int32(3), jnp.int32(12))
    next_wait = jnp.where(delivering, delivery_wait, next_wait_pre)
    next_memory = jnp.stack([next_target, next_wait, next_mode_pre, next_ticks_pre, next_trust_pre]).astype(jnp.int32)
    trace = {
        "target": target,
        "signal": signal,
        "wait": wait,
        "mode": old_mode,
        "middle_marked": middle_marked,
        "inspecting": inspecting,
        "trust": old_trust,
        "pot_count": pot_count,
        "pot_cooked": pot_cooked,
        "row": row,
        "col": col,
    }
    return action, next_memory, trace
