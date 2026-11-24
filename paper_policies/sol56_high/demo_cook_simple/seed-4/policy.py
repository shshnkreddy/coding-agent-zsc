"""Reactive, convention-light policy for the demo_cook_simple layout."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'reactive_natural_roles', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5


def _static_map():
    """Padded static map used to identify a global position from a local crop."""
    grid = jnp.zeros((9, 15, 9), dtype=jnp.int32)
    walls = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9),
             (0, 10), (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2),
             (4, 3), (4, 4), (4, 7), (4, 9), (4, 10))
    for row, col in walls:
        grid = grid.at[row + 2, col + 2, 0].set(1)
    for row, col in ((2, 10),):
        grid = grid.at[row + 2, col + 2, 1].set(1)
    for row, col in ((2, 7),):
        grid = grid.at[row + 2, col + 2, 2].set(1)
    for row, col in ((0, 5), (4, 5)):
        grid = grid.at[row + 2, col + 2, 3].set(1)
    for row, col in ((1, 10), (3, 10)):
        grid = grid.at[row + 2, col + 2, 5].set(1)
    for row, col in ((0, 8), (1, 0)):
        grid = grid.at[row + 2, col + 2, 6].set(1)
    for row, col in ((3, 0), (4, 8)):
        grid = grid.at[row + 2, col + 2, 7].set(1)
    for row, col in ((0, 6), (4, 6)):
        grid = grid.at[row + 2, col + 2, 8].set(1)
    return grid


def _locate(obs, positions):
    grid = _static_map()
    def crop(pos):
        return jax.lax.dynamic_slice(grid, (pos[0], pos[1], 0), (5, 5, 9))
    expected = jax.vmap(crop)(positions)
    error = jnp.sum(jnp.abs(expected - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    return positions[jnp.argmin(error)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _navigate(row, col, target_row, target_col, desired_facing, face_action, facing):
    at_target = (row == target_row) & (col == target_col)
    final_action = jnp.where(facing == desired_facing, INTERACT, face_action)
    horizontal = jnp.where(col < target_col, RIGHT, LEFT)
    vertical = jnp.where(row < target_row, DOWN, UP)
    # Approach along the desired-facing axis last, often eliminating a
    # separate blocked movement solely to turn toward the object.
    wants_horizontal_facing = desired_facing >= 2
    action = jnp.where(wants_horizontal_facing,
        jnp.where(row != target_row, vertical, horizontal),
        jnp.where(col != target_col, horizontal, vertical))
    return jnp.where(at_target, final_action, action).astype(jnp.int32)


def _pot(obs):
    mask = obs[:, :, 22]
    counts = jnp.sum(obs[:, :, 31:34] * mask[:, :, None], axis=(0, 1))
    cooked = jnp.sum(obs[:, :, 30] * mask) > 0
    timer = jnp.sum(obs[:, :, 39] * mask)
    seen = jnp.sum(mask) > 0
    return counts, jnp.sum(counts), cooked, timer, seen


def _raw_type(obs, start_channel):
    counts = obs[2, 2, start_channel:start_channel + 3]
    return jnp.argmax(counts).astype(jnp.int32), jnp.sum(counts) > 0


def _partner_raw_type(obs):
    counts = jnp.sum(obs[:, :, 17:20], axis=(0, 1))
    return jnp.argmax(counts).astype(jnp.int32), jnp.sum(counts) > 0


def _ingredient_route(row, col, ingredient, facing, left_room):
    target_row = jnp.where(ingredient == 0, 1, 3)
    if left_room:
        return _navigate(row, col, target_row, 1, 3, LEFT, facing)
    desired = jnp.where(ingredient == 0, 0, 1)
    face_action = jnp.where(ingredient == 0, UP, DOWN)
    return _navigate(row, col, target_row, 8, desired, face_action, facing)


def init_agent_0():
    # target, committed fetch, prior pot total, unchanged wait, step, raw mode,
    # post-plating cooldown, next type to stock, stocked-this-cook flag.
    return jnp.asarray([-1, -1, -1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, init_agent_0(), memory)
    positions = jnp.asarray([(r, c) for r in range(1, 4) for c in range(1, 7)], dtype=jnp.int32)
    row, col = _locate(obs, positions)
    facing = _facing(obs)

    recipe0, recipe1 = jnp.sum(obs[:, :, 36]), jnp.sum(obs[:, :, 37])
    recipe_seen = (recipe0 + recipe1) > 0
    observed_target = jnp.where(recipe0 > recipe1, 0, 1).astype(jnp.int32)
    target = jnp.where(recipe_seen, observed_target, mem[0])
    pot_counts, pot_total, pot_cooked, pot_timer, pot_seen = _pot(obs)
    pot_idle = (pot_timer == 0) & (~pot_cooked)
    partial = (pot_total > 0) & (pot_total < 3) & pot_idle
    wait = jnp.where(partial & pot_seen & (pot_total == mem[2]), mem[3] + 1, 0).astype(jnp.int32)
    just_plated = pot_seen & (pot_total == 0) & (mem[2] >= 3)
    cooldown = jnp.where(just_plated, 2, jnp.maximum(mem[6] - 1, 0)).astype(jnp.int32)
    stocked_this_cook = jnp.where(pot_seen & (pot_total < 3), 0, mem[8]).astype(jnp.int32)
    previous_total = jnp.where(pot_seen, pot_total, mem[2]).astype(jnp.int32)

    held_type, holds_raw = _raw_type(obs, 7)
    partner_type, partner_holds_raw = _partner_raw_type(obs)
    partner_holds_plate = jnp.sum(obs[:, :, 15]) > 0
    partner_holds_dish = jnp.sum(obs[:, :, 16]) > 0
    holds_plate, holds_dish = obs[2, 2, 5] > 0, obs[2, 2, 6] > 0
    at_pot = (row == 2) & (col == 6)
    fetch_type, mode = mem[1], mem[5]
    base_action = _navigate(row, col, 2, 6, 2, RIGHT, facing)

    # Dedicated recipe-independent stashes on the pot-adjacent counters:
    # ingredient 0 at global (1,7), ingredient 1 at (3,7).
    top_stash = obs[1, 3, 29:34]
    bottom_stash = obs[3, 3, 29:34]
    top_free = jnp.sum(top_stash) == 0
    bottom_free = jnp.sum(bottom_stash) == 0
    top_has_zero = top_stash[2] > 0
    bottom_has_one = bottom_stash[3] > 0

    existing_type = jnp.where(pot_counts[0] > 0, 0, jnp.where(pot_counts[1] > 0, 1, 2))
    safe_index = jnp.maximum(target, 0)
    pot_has_wrong = (pot_total > 0) & (pot_counts[safe_index] != pot_total)
    desired_type = jnp.where(pot_has_wrong, existing_type, target).astype(jnp.int32)
    partner_completes = partner_holds_raw & (partner_type == desired_type) & ((pot_total + 1) >= 3)
    partner_inventory_busy = partner_holds_plate | partner_holds_dish
    ready_to_fetch = ((pot_total == 0) | partner_inventory_busy |
                      ((wait >= 8) & (~partner_completes)))
    can_fetch = at_pot & (target >= 0) & (pot_total < 3) & pot_idle & ready_to_fetch & (cooldown == 0)
    stash_available = jnp.where(desired_type == 0, top_has_zero, bottom_has_one)
    use_stash = can_fetch & stash_available

    both_missing = top_free & bottom_free
    stock_type = jnp.where(both_missing, mem[7], jnp.where(top_free, 0, 1)).astype(jnp.int32)
    stock_missing = top_free | bottom_free
    can_stock = at_pot & (pot_total >= 3) & stock_missing & (stocked_this_cook == 0)
    start_task = can_fetch | can_stock
    task_type = jnp.where(can_stock, stock_type, desired_type).astype(jnp.int32)
    target_stash_mode = jnp.where(desired_type == 0, 5, 6).astype(jnp.int32)
    stock_mode = jnp.where(stock_type == 0, 3, 4).astype(jnp.int32)
    task_mode = jnp.where(can_stock, stock_mode, jnp.where(use_stash, target_stash_mode, 0))
    chosen_fetch = jnp.where(start_task, task_type, fetch_type).astype(jnp.int32)
    working_mode = jnp.where(start_task, task_mode, mode).astype(jnp.int32)
    committed = (fetch_type >= 0) | start_task
    retrieve_top = _navigate(row, col, 1, 6, 2, RIGHT, facing)
    retrieve_bottom = _navigate(row, col, 3, 6, 2, RIGHT, facing)
    committed_action = jnp.where(working_mode == 5, retrieve_top,
        jnp.where(working_mode == 6, retrieve_bottom,
                  _ingredient_route(row, col, chosen_fetch, facing, True)))
    empty_action = jnp.where(committed, committed_action,
        jnp.where(at_pot, STAY, base_action))
    next_fetch = jnp.where(start_task, task_type, fetch_type)

    stale_for_new_recipe = (held_type != target) & ((pot_total == 0) | (pot_counts[safe_index] == pot_total))
    pot_unavailable_here = holds_raw & at_pot & ((pot_total >= 3) | (~pot_idle) | stale_for_new_recipe)
    # Stale overflow is parked on more distant exclusive counters, leaving the
    # two working stashes available for validated ingredients.
    top_occupied = jnp.sum(obs[0, 0, 29:34]) > 0
    selected_mode = jnp.where(top_occupied, 2, 1).astype(jnp.int32)
    active_mode = jnp.where(pot_unavailable_here & (mode < 3), selected_mode, mode)
    raw_to_pot = _navigate(row, col, 2, 6, 2, RIGHT, facing)
    raw_to_top = _navigate(row, col, 1, 4, 0, UP, facing)
    raw_to_bottom = _navigate(row, col, 3, 4, 1, DOWN, facing)
    raw_to_zero_stash = _navigate(row, col, 1, 6, 2, RIGHT, facing)
    raw_to_one_stash = _navigate(row, col, 3, 6, 2, RIGHT, facing)
    raw_action = jnp.where(active_mode == 1, raw_to_top,
        jnp.where(active_mode == 2, raw_to_bottom,
          jnp.where(active_mode == 3, raw_to_zero_stash,
            jnp.where(active_mode == 4, raw_to_one_stash, raw_to_pot))))
    depositing = holds_raw & (active_mode == 0) & at_pot & (facing == 2)
    parking_top = holds_raw & (active_mode == 1) & (row == 1) & (col == 4) & (facing == 0)
    parking_bottom = holds_raw & (active_mode == 2) & (row == 3) & (col == 4) & (facing == 1)
    stocking_zero = holds_raw & (active_mode == 3) & (row == 1) & (col == 6) & (facing == 2)
    stocking_one = holds_raw & (active_mode == 4) & (row == 3) & (col == 6) & (facing == 2)
    retrieving_zero = (~holds_raw) & (working_mode == 5) & (row == 1) & (col == 6) & (facing == 2)
    retrieving_one = (~holds_raw) & (working_mode == 6) & (row == 3) & (col == 6) & (facing == 2)
    transfer = (depositing | parking_top | parking_bottom | stocking_zero |
                stocking_one | retrieving_zero | retrieving_one)
    next_fetch = jnp.where(transfer, -1, next_fetch)
    next_mode = jnp.where(transfer, 0,
        jnp.where(pot_unavailable_here & (mode < 3), selected_mode, working_mode))
    next_stock_turn = jnp.where(stocking_zero, 1, jnp.where(stocking_one, 0, mem[7]))
    next_stocked = jnp.where(stocking_zero | stocking_one, 1, stocked_this_cook)

    action = jnp.where(holds_raw, raw_action, empty_action).astype(jnp.int32)
    next_memory = jnp.asarray([target, next_fetch, previous_total, wait, mem[4] + 1,
                               next_mode, cooldown, next_stock_turn, next_stocked], dtype=jnp.int32)
    trace = {'position': jnp.asarray([row, col], dtype=jnp.int32),
             'target': target, 'pot_total': pot_total.astype(jnp.int32),
             'fetch_type': next_fetch.astype(jnp.int32)}
    return action, next_memory, trace


def init_agent_1():
    # committed fetch, raw mode, step, previous visible partner row
    return jnp.asarray([-1, 0, 0, -1], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, init_agent_1(), memory)
    positions = jnp.asarray([(r, c) for r in range(1, 4) for c in range(8, 10)], dtype=jnp.int32)
    row, col = _locate(obs, positions)
    facing = _facing(obs)
    pot_counts, pot_total, pot_cooked, pot_timer, _ = _pot(obs)
    pot_idle = (pot_timer == 0) & (~pot_cooked)
    held_type, holds_raw = _raw_type(obs, 7)
    partner_type, partner_holds_raw = _partner_raw_type(obs)
    holds_plate, holds_dish = obs[2, 2, 5] > 0, obs[2, 2, 6] > 0
    at_pot = (row == 2) & (col == 8)

    existing_type = jnp.where(pot_counts[0] > 0, 0, jnp.where(pot_counts[1] > 0, 1, 2)).astype(jnp.int32)
    committed_type = mem[0]
    need_help = at_pot & (pot_total > 0) & (pot_total < 3) & pot_idle
    partner_will_finish = partner_holds_raw & ((pot_total + 1) >= 3)
    anticipatory_help = at_pot & (pot_total == 0) & pot_idle & partner_holds_raw
    partner_rows = jnp.sum(obs[:, :, 10], axis=1)
    partner_local_row = jnp.argmax(partner_rows).astype(jnp.int32)
    partner_seen = jnp.sum(partner_rows) > 0
    partner_faces_up = jnp.sum(obs[:, :, 11]) > 0
    partner_faces_down = jnp.sum(obs[:, :, 12]) > 0
    outward_facing = ((partner_local_row == 1) & partner_faces_up) | (
        (partner_local_row == 3) & partner_faces_down)
    inferred_intent = partner_seen & (mem[2] > 0) & outward_facing
    intent_help = at_pot & (pot_total == 0) & pot_idle & inferred_intent & (~partner_holds_raw)
    intent_type = jnp.where(partner_local_row == 1, 0, 1).astype(jnp.int32)
    start_fetch = (need_help & (~partner_will_finish)) | anticipatory_help | intent_help
    observed_type = jnp.where(pot_total > 0, existing_type,
                              jnp.where(partner_holds_raw, partner_type, intent_type))
    fetch_type = jnp.where(start_fetch, observed_type, committed_type).astype(jnp.int32)
    go_pot = _navigate(row, col, 2, 8, 3, LEFT, facing)
    fetch_action = _ingredient_route(row, col, fetch_type, facing, False)
    plate_row = jnp.where(row <= 2, 1, 3)
    get_plate = _navigate(row, col, plate_row, 9, 2, RIGHT, facing)
    pot_fullish = (pot_total >= 3) | (pot_timer > 0) | pot_cooked
    empty_action = jnp.where(committed_type >= 0, fetch_action,
        jnp.where(start_fetch, _ingredient_route(row, col, observed_type, facing, False),
                  jnp.where(pot_fullish, get_plate,
                            jnp.where(at_pot, STAY, go_pot))))
    plate_action = go_pot
    dish_action = _navigate(row, col, 2, 9, 2, RIGHT, facing)

    mode = mem[1]
    prefetch_confirmed = ((pot_total > 0) & (existing_type == held_type)) | (
        partner_holds_raw & (partner_type == held_type))
    prefetch_mismatch = (mode == 3) & (pot_total > 0) & (existing_type != held_type)
    pot_unavailable_here = holds_raw & at_pot & (((pot_total >= 3) | (~pot_idle)) | prefetch_mismatch)
    top_occupied = jnp.sum(obs[0, 3, 29:34]) > 0
    selected_mode = jnp.where(top_occupied, 2, 1).astype(jnp.int32)
    confirmed_mode = jnp.where((mode == 3) & prefetch_confirmed, 0, mode)
    active_mode = jnp.where(pot_unavailable_here, selected_mode, confirmed_mode)
    raw_to_top = _navigate(row, col, 1, 9, 0, UP, facing)
    raw_to_bottom = _navigate(row, col, 3, 9, 1, DOWN, facing)
    wait_or_face_pot = jnp.where(facing == 3, STAY, LEFT)
    raw_action = jnp.where(active_mode == 1, raw_to_top,
        jnp.where(active_mode == 2, raw_to_bottom,
                  jnp.where((active_mode == 3) & at_pot, wait_or_face_pot, go_pot)))
    depositing = holds_raw & (active_mode == 0) & at_pot & (facing == 3)
    parking_top = holds_raw & (active_mode == 1) & (row == 1) & (col == 9) & (facing == 0)
    parking_bottom = holds_raw & (active_mode == 2) & (row == 3) & (col == 9) & (facing == 1)
    transfer = depositing | parking_top | parking_bottom
    next_fetch = jnp.where(transfer, -1, fetch_type)
    base_mode = jnp.where(intent_help, 3, confirmed_mode)
    next_mode = jnp.where(transfer, 0, jnp.where(pot_unavailable_here, selected_mode, base_mode))

    action = jnp.where(holds_dish, dish_action,
        jnp.where(holds_plate, plate_action,
                  jnp.where(holds_raw, raw_action, empty_action))).astype(jnp.int32)
    next_partner_row = jnp.where(partner_seen, partner_local_row, -1)
    next_memory = jnp.asarray([next_fetch, next_mode, mem[2] + 1, next_partner_row], dtype=jnp.int32)
    trace = {'position': jnp.asarray([row, col], dtype=jnp.int32),
             'pot_total': pot_total.astype(jnp.int32),
             'fetch_type': next_fetch.astype(jnp.int32)}
    return action, next_memory, trace
