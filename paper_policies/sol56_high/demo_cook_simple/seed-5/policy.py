"""Convention-light policy for the two-room demo_cook_simple layout.

The west player reads the permanent recipe displays and supplies ingredients.
The east player helps with one ingredient after observing the pot, then handles
plates and delivery. Static landmarks in the local crop provide localization.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "landmark_pipeline_v1",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 11,
    "num_ingredients": 3,
}


# Indices are wall, goal, pot, recipe, button, plates, ingredient 0/1/2,
# and floor. Taking the first nine columns of a ten-way one-hot reconstructs
# observation channels 20:29, with floor and off-map cells both all-zero.
def _locate(obs, seat):
    codes = jnp.asarray(
        [
            [0, 0, 0, 0, 0, 3, 8, 0, 6, 0, 0],
            [6, 9, 9, 9, 9, 9, 9, 0, 9, 9, 5],
            [0, 9, 9, 9, 9, 9, 9, 2, 9, 9, 1],
            [7, 9, 9, 9, 9, 9, 9, 0, 9, 9, 5],
            [0, 0, 0, 0, 0, 3, 8, 0, 7, 0, 0],
        ],
        dtype=jnp.int32,
    )
    static = jnp.eye(10, dtype=jnp.int32)[codes][..., :9]
    padded = jnp.pad(static, ((2, 2), (2, 2), (0, 0)))
    if seat == 0:
        python_coords = [(r, c) for r in (1, 2, 3)
                         for c in (1, 2, 3, 4, 5, 6)]
    else:
        python_coords = [(r, c) for r in (1, 2, 3) for c in (8, 9)]
    coords = jnp.asarray(python_coords, dtype=jnp.int32)
    patches = jnp.stack(
        [padded[r : r + 5, c : c + 5, :] for r, c in python_coords]
    )
    seen = obs[:, :, 20:29]
    errors = jnp.sum(jnp.abs(patches - seen[None, ...]), axis=(1, 2, 3))
    index = jnp.argmin(errors)
    return coords[index, 0], coords[index, 1], errors[index]


def _navigate(row, col, target_row, target_col):
    """Shortest-path move in either rectangular room (horizontal first)."""
    horizontal = jnp.where(col < target_col, 0, 2)
    vertical = jnp.where(row < target_row, 1, 3)
    return jnp.where(col != target_col, horizontal, vertical).astype(jnp.int32)


def _navigate_vertical_first(row, col, target_row, target_col):
    """Shortest path whose final horizontal move preserves LEFT/RIGHT facing."""
    horizontal = jnp.where(col < target_col, 0, 2)
    vertical = jnp.where(row < target_row, 1, 3)
    return jnp.where(row != target_row, vertical, horizontal).astype(jnp.int32)


def _go_and_interact(obs, row, col, stand_row, stand_col,
                     face_action, face_channel):
    at_stand = (row == stand_row) & (col == stand_col)
    facing = obs[2, 2, face_channel] > 0
    terminal = jnp.where(facing, 5, face_action)
    return jnp.where(
        at_stand,
        terminal,
        _navigate(row, col, stand_row, stand_col),
    ).astype(jnp.int32)


def _go_and_interact_vertical(obs, row, col, stand_row, stand_col,
                              face_action, face_channel):
    at_stand = (row == stand_row) & (col == stand_col)
    facing = obs[2, 2, face_channel] > 0
    terminal = jnp.where(facing, 5, face_action)
    return jnp.where(
        at_stand,
        terminal,
        _navigate_vertical_first(row, col, stand_row, stand_col),
    ).astype(jnp.int32)


def _pot_features(obs):
    mask = obs[:, :, 22]
    total = jnp.sum(mask[:, :, None] * obs[:, :, 31:34])
    cooked = jnp.sum(mask * obs[:, :, 30]) > 0
    timer = jnp.sum(mask * obs[:, :, 39])
    return total.astype(jnp.int32), cooked, timer.astype(jnp.int32)


def init_agent_0():
    return {
        "target": jnp.asarray(-1, dtype=jnp.int32),
        "saw_full": jnp.asarray(0, dtype=jnp.int32),
        "empty_wait": jnp.asarray(0, dtype=jnp.int32),
        "partial_total": jnp.asarray(0, dtype=jnp.int32),
        "partial_wait": jnp.asarray(0, dtype=jnp.int32),
        "cache_0": jnp.asarray(0, dtype=jnp.int32),
        "cache_1": jnp.asarray(0, dtype=jnp.int32),
        "cache_0_occupied": jnp.asarray(0, dtype=jnp.int32),
        "cache_1_occupied": jnp.asarray(0, dtype=jnp.int32),
        "cook_remaining": jnp.asarray(0, dtype=jnp.int32),
        "steps": jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_0()
    old_target = jnp.where(reset, initial["target"], memory["target"])
    saw_full = jnp.where(reset, initial["saw_full"], memory["saw_full"])
    empty_wait = jnp.where(reset, initial["empty_wait"], memory["empty_wait"])
    old_partial_total = jnp.where(
        reset, initial["partial_total"], memory["partial_total"]
    )
    old_partial_wait = jnp.where(
        reset, initial["partial_wait"], memory["partial_wait"]
    )
    old_cache_0 = jnp.where(reset, initial["cache_0"], memory["cache_0"])
    old_cache_1 = jnp.where(reset, initial["cache_1"], memory["cache_1"])
    old_cache_0_occupied = jnp.where(
        reset, initial["cache_0_occupied"], memory["cache_0_occupied"]
    )
    old_cache_1_occupied = jnp.where(
        reset, initial["cache_1_occupied"], memory["cache_1_occupied"]
    )
    old_cook_remaining = jnp.where(
        reset, initial["cook_remaining"], memory["cook_remaining"]
    )
    steps = jnp.where(reset, initial["steps"], memory["steps"])

    row, col, loc_error = _locate(obs, 0)
    recipe_0 = jnp.sum(obs[:, :, 36])
    recipe_1 = jnp.sum(obs[:, :, 37])
    display_valid = (recipe_0 + recipe_1) > 0
    displayed_target = jnp.where(recipe_0 > 0, 0, 1).astype(jnp.int32)
    recipe_changed = display_valid & (old_target >= 0) & (displayed_target != old_target)
    target = jnp.where(display_valid, displayed_target, old_target)

    held_0 = obs[2, 2, 7]
    held_1 = obs[2, 2, 8]
    holding_raw = (held_0 + held_1) > 0
    # Divider-counter caches live at (1, 7) and (3, 7). Whenever visible,
    # actual public grid contents supersede private bookkeeping.
    cache_col_visible = col >= 5
    cache_0_cell = obs[jnp.clip(3 - row, 0, 4),
                       jnp.clip(9 - col, 0, 4), 29:34]
    cache_1_cell = obs[jnp.clip(5 - row, 0, 4),
                       jnp.clip(9 - col, 0, 4), 29:34]
    cache_0_seen = cache_0_cell[2] > 0
    cache_1_seen = cache_1_cell[3] > 0
    cache_0_occupied_seen = jnp.sum(cache_0_cell) > 0
    cache_1_occupied_seen = jnp.sum(cache_1_cell) > 0
    cache_0 = jnp.where(cache_col_visible, cache_0_seen, old_cache_0)
    cache_1 = jnp.where(cache_col_visible, cache_1_seen, old_cache_1)
    cache_0_occupied = jnp.where(
        cache_col_visible, cache_0_occupied_seen, old_cache_0_occupied
    )
    cache_1_occupied = jnp.where(
        cache_col_visible, cache_1_occupied_seen, old_cache_1_occupied
    )
    pot_total, pot_cooked, pot_timer = _pot_features(obs)
    cook_remaining = jnp.where(
        pot_timer > 0,
        pot_timer,
        jnp.maximum(old_cook_remaining - 1, 0),
    ).astype(jnp.int32)
    pot_visible = jnp.sum(obs[:, :, 22]) > 0
    full_now = pot_visible & ((pot_total >= 3) | pot_cooked | (pot_timer > 0))
    just_empty = (saw_full > 0) & pot_visible & (pot_total == 0) & (~pot_cooked)
    next_wait = jnp.where(just_empty, empty_wait + 1, 0)
    # The east station is two actions from the pot: move right, then interact.
    # Three empty-pot observations safely cover pickup and delivery even when
    # the newly sampled recipe repeats and therefore cannot signal completion.
    cycle_done = recipe_changed | (just_empty & (next_wait >= 3))
    next_saw_full = jnp.where(full_now, 1, jnp.where(cycle_done, 0, saw_full))
    next_wait = jnp.where(cycle_done, 0, next_wait)
    partial_now = pot_visible & (pot_total > 0) & (pot_total < 3) & (pot_timer == 0)
    partial_changed = pot_total != old_partial_total
    next_partial_wait = jnp.where(
        partial_now,
        jnp.where(partial_changed, 1, old_partial_wait + 1),
        0,
    )
    next_partial_total = jnp.where(partial_now, pot_total, 0)
    partner_plate_visible = jnp.sum(obs[:, :, 15]) > 0
    wait_for_helper = (
        partial_now & (pot_total == 1) & (next_partial_wait <= 5) & (~holding_raw)
        & (~partner_plate_visible)
    )

    # Before seeing a recipe, walk into range of either permanent display.
    seek_recipe = _navigate(row, col, 2, 5)
    supply_row = jnp.where(target == 0, 1, 3)
    get_ingredient = _go_and_interact_vertical(
        obs, row, col, supply_row, 1, 2, 4  # face LEFT
    )
    put_ingredient = _go_and_interact_vertical(
        obs, row, col, 2, 6, 0, 3  # face RIGHT
    )

    held_kind = jnp.where(held_0 > 0, 0, 1)
    cache_row = jnp.where(held_kind == 0, 1, 3)
    store_held = _go_and_interact_vertical(
        obs, row, col, cache_row, 6, 0, 3
    )
    held_cache_full = jnp.where(
        held_kind == 0, cache_0_occupied, cache_1_occupied
    )
    overflow_row = jnp.where(held_kind == 0, 1, 3)
    overflow_face_action = jnp.where(held_kind == 0, 3, 1)
    overflow_face_channel = jnp.where(held_kind == 0, 1, 2)
    store_overflow = _go_and_interact(
        obs, row, col, overflow_row, 4,
        overflow_face_action, overflow_face_channel,
    )
    store_or_overflow = jnp.where(
        held_cache_full, store_overflow, store_held
    )
    desired_cache_row = jnp.where(target == 0, 1, 3)
    take_desired_cache = _go_and_interact_vertical(
        obs, row, col, desired_cache_row, 6, 0, 3
    )
    get_cache_0 = _go_and_interact_vertical(obs, row, col, 1, 1, 2, 4)
    get_cache_1 = _go_and_interact_vertical(obs, row, col, 3, 1, 2, 4)

    waiting_cycle = (next_saw_full > 0) | full_now
    # With both caches ready, wait beside the cache matching the current
    # recipe. Repetition is equiprobable, so this shortens half of handoffs;
    # the pot and both permanent displays remain visible from either stand.
    cycle_camp_row = jnp.where(target == 0, 1, 3)
    at_cycle_camp = (row == cycle_camp_row) & (col == 6)
    facing_cache = obs[2, 2, 3] > 0
    cycle_action = jnp.where(
        at_cycle_camp,
        jnp.where(facing_cache, 4, 0),
        _navigate(row, col, cycle_camp_row, 6),
    )
    desired_cached = jnp.where(target == 0, cache_0, cache_1)
    ordinary = jnp.where(
        holding_raw,
        jnp.where(held_kind == target, put_ingredient, store_or_overflow),
        jnp.where(
            wait_for_helper,
            4,
            jnp.where(desired_cached, take_desired_cache, get_ingredient),
        ),
    )
    cache_0_fillable = cache_0_occupied == 0
    cache_1_fillable = cache_1_occupied == 0
    missing_cache = cache_0_fillable | cache_1_fillable
    cache_fetch_action = jnp.where(
        cache_0_fillable, get_cache_0, get_cache_1
    )
    safe_to_fetch_cache = (cook_remaining >= 14) | (col < 6)
    cache_fill_action = jnp.where(
        holding_raw,
        store_or_overflow,
        jnp.where(
            missing_cache & safe_to_fetch_cache,
            cache_fetch_action,
            cycle_action,
        ),
    )
    action = jnp.where(
        target < 0,
        seek_recipe,
        jnp.where(waiting_cycle, cache_fill_action, ordinary),
    ).astype(jnp.int32)

    next_memory = {
        "target": target.astype(jnp.int32),
        "saw_full": next_saw_full.astype(jnp.int32),
        "empty_wait": next_wait.astype(jnp.int32),
        "partial_total": next_partial_total.astype(jnp.int32),
        "partial_wait": next_partial_wait.astype(jnp.int32),
        "cache_0": cache_0.astype(jnp.int32),
        "cache_1": cache_1.astype(jnp.int32),
        "cache_0_occupied": cache_0_occupied.astype(jnp.int32),
        "cache_1_occupied": cache_1_occupied.astype(jnp.int32),
        "cook_remaining": cook_remaining,
        "steps": (steps + 1).astype(jnp.int32),
    }
    trace = {
        "row": row,
        "col": col,
        "loc_error": loc_error,
        "target": target,
        "pot_total": pot_total,
        "waiting": waiting_cycle.astype(jnp.int32),
        "cache_0": cache_0.astype(jnp.int32),
        "cache_1": cache_1.astype(jnp.int32),
        "cache_0_occupied": cache_0_occupied.astype(jnp.int32),
        "cache_1_occupied": cache_1_occupied.astype(jnp.int32),
        "held_kind": jnp.where(holding_raw, held_kind, -1).astype(jnp.int32),
        "cook_remaining": cook_remaining,
    }
    return action, next_memory, trace


def init_agent_1():
    return {
        "assist_kind": jnp.asarray(-1, dtype=jnp.int32),
        "partner_row": jnp.asarray(-1, dtype=jnp.int32),
        "partner_col": jnp.asarray(-1, dtype=jnp.int32),
        "steps": jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    steps = jnp.where(reset, 0, memory["steps"])
    old_assist_kind = jnp.where(reset, -1, memory["assist_kind"])
    old_partner_row = jnp.where(reset, -1, memory["partner_row"])
    old_partner_col = jnp.where(reset, -1, memory["partner_col"])
    row, col, loc_error = _locate(obs, 1)
    pot_total, pot_cooked, pot_timer = _pot_features(obs)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_0 = obs[2, 2, 7]
    held_1 = obs[2, 2, 8]
    holding_raw = (held_0 + held_1) > 0
    empty_handed = (~held_plate) & (~held_cooked) & (~holding_raw)

    pot_has_0 = jnp.sum(obs[:, :, 31] * obs[:, :, 22]) > 0
    partner_has_0 = jnp.sum(obs[:, :, 17] * obs[:, :, 10]) > 0
    partner_has_1 = jnp.sum(obs[:, :, 18] * obs[:, :, 10]) > 0
    partner_raw_visible = partner_has_0 | partner_has_1
    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_inventory = jnp.sum(obs[:, :, 15:20])
    partner_waiting_empty = partner_visible & (partner_inventory == 0)
    local_rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    local_cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    partner_row = jnp.sum(obs[:, :, 10] * (row + local_rows - 2))
    partner_col = jnp.sum(obs[:, :, 10] * (col + local_cols - 2))
    partner_arrived = (
        partner_visible
        & ((partner_row != old_partner_row) | (partner_col != old_partner_col))
    )
    upper_cache_has_0 = obs[jnp.clip(3 - row, 0, 4),
                            jnp.clip(9 - col, 0, 4), 31] > 0
    lower_cache_has_1 = obs[jnp.clip(5 - row, 0, 4),
                            jnp.clip(9 - col, 0, 4), 32] > 0
    partner_at_cache_0 = (
        partner_waiting_empty & partner_arrived
        & (partner_row == 1) & (partner_col == 6) & upper_cache_has_0
    )
    partner_at_cache_1 = (
        partner_waiting_empty & partner_arrived
        & (partner_row == 3) & (partner_col == 6) & lower_cache_has_1
    )
    partner_cache_signal = partner_at_cache_0 | partner_at_cache_1
    inferred_ingredient = jnp.where(
        pot_total > 0,
        jnp.where(pot_has_0, 0, 1),
        jnp.where(partner_has_0 | partner_at_cache_0, 0, 1),
    ).astype(jnp.int32)
    plate_row = jnp.where(row <= 2, 1, 3)
    plate_action = _go_and_interact_vertical(obs, row, col, plate_row, 9, 0, 3)
    pot_action = _go_and_interact_vertical(obs, row, col, 2, 8, 2, 4)
    delivery_action = _go_and_interact(obs, row, col, 2, 9, 0, 3)

    # If the pot filled while we fetched, park the redundant ingredient.
    raw_kind = jnp.where(held_0 > 0, 0, 1)
    stash_row = jnp.where(raw_kind == 0, 1, 3)
    stash_face_action = jnp.where(raw_kind == 0, 3, 1)
    stash_face_channel = jnp.where(raw_kind == 0, 1, 2)
    stash_action = _go_and_interact(
        obs, row, col, stash_row, 9, stash_face_action, stash_face_channel
    )
    raw_action = jnp.where(pot_total < 3, pot_action, stash_action)

    # A nonempty idle pot reveals the homogeneous recipe. Complete it, then
    # fetch a plate during the twenty-step cooking interval.
    should_help = (
        ((pot_total == 1) |
         ((pot_total == 2) & partner_waiting_empty) |
         ((pot_total == 0) & (partner_raw_visible | partner_cache_signal)))
        & (~pot_cooked) & (pot_timer == 0)
    )
    start_assist = empty_handed & should_help
    assist_kind = jnp.where(start_assist, inferred_ingredient, old_assist_kind)
    committed_assist = empty_handed & (assist_kind >= 0)
    chosen_ingredient = jnp.where(
        assist_kind >= 0, assist_kind, inferred_ingredient
    )
    ingredient_row = jnp.where(chosen_ingredient == 0, 1, 3)
    ingredient_face_action = jnp.where(chosen_ingredient == 0, 3, 1)
    ingredient_face_channel = jnp.where(chosen_ingredient == 0, 1, 2)
    get_ingredient = _go_and_interact(
        obs, row, col, ingredient_row, 8,
        ingredient_face_action, ingredient_face_channel,
    )
    empty_action = jnp.where(
        committed_assist,
        get_ingredient,
        jnp.where(
            pot_total >= 2,
            plate_action,
            jnp.where((row == 2) & (col == 8), 4,
                      _navigate(row, col, 2, 8)),
        ),
    )
    plate_wait = jnp.where(
        pot_cooked,
        pot_action,
        jnp.where(
            (row == 2) & (col == 8),
            jnp.where(obs[2, 2, 4] > 0, 4, 2),
            _navigate(row, col, 2, 8),
        ),
    )
    action = jnp.where(
        held_cooked,
        delivery_action,
        jnp.where(held_plate, plate_wait,
                  jnp.where(holding_raw, raw_action, empty_action)),
    ).astype(jnp.int32)

    next_assist_kind = jnp.where(empty_handed, assist_kind, -1)
    next_memory = {
        "assist_kind": next_assist_kind.astype(jnp.int32),
        "partner_row": jnp.where(
            partner_visible, partner_row, old_partner_row
        ).astype(jnp.int32),
        "partner_col": jnp.where(
            partner_visible, partner_col, old_partner_col
        ).astype(jnp.int32),
        "steps": (steps + 1).astype(jnp.int32),
    }
    trace = {
        "row": row,
        "col": col,
        "loc_error": loc_error,
        "pot_total": pot_total,
        "pot_timer": pot_timer,
        "help": (empty_handed & should_help).astype(jnp.int32),
        "assist_kind": assist_kind.astype(jnp.int32),
    }
    return action, next_memory, trace
