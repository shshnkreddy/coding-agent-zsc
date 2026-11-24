"""Convention-robust policy for the two-room test_time_wide layout.

The lower cook advertises the displayed recipe by waiting below one of the
two pots (left = ingredient 0, right = ingredient 1). The upper cook reads
that public signal and performs ingredient, plate and delivery work.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "dual_pot_adaptive_v1", "adapter": "overcooked_v2",
    "adapter_version": 1, "max_height": 7, "max_width": 6,
    "num_ingredients": 4,
}

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

def _mark(arr, channel, cells):
    for row, col in cells:
        arr = arr.at[row + 2, col + 2, channel].set(1)
    return arr

_floor_cells = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2),
    (2, 3), (2, 4), (4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2),
    (5, 3), (5, 4))

def _position(obs):
    # Constructing constants inside the traced function keeps module import
    # side-effect free, as required by the policy source validator.
    static = jnp.zeros((11, 10, 10), dtype=jnp.int32)
    static = _mark(static, 0, ((0, 0), (0, 1), (0, 4), (0, 5), (3, 0),
        (3, 2), (3, 4), (3, 5), (5, 0), (5, 5), (6, 0), (6, 1), (6, 3),
        (6, 4), (6, 5)))
    static = _mark(static, 1, ((0, 2),))
    static = _mark(static, 2, ((3, 1), (3, 3)))
    static = _mark(static, 3, ((6, 2),))
    static = _mark(static, 5, ((0, 3),))
    static = _mark(static, 6, ((1, 0), (1, 5)))
    static = _mark(static, 7, ((2, 0), (2, 5)))
    static = _mark(static, 9, ((4, 0), (4, 5)))
    positions = jnp.asarray(_floor_cells, dtype=jnp.int32)
    templates = jnp.stack([static[r:r + 5, c:c + 5, :] for r, c in _floor_cells])
    mismatch = jnp.sum(jnp.abs(templates - obs[:, :, 22:32]), axis=(1, 2, 3))
    return positions[jnp.argmin(mismatch)]

def _move_to(pos, target):
    vertical = jnp.where(pos[0] < target[0], DOWN, UP)
    horizontal = jnp.where(pos[1] < target[1], RIGHT, LEFT)
    return jnp.where(pos[0] != target[0], vertical,
        jnp.where(pos[1] != target[1], horizontal, STAY))

def _face_then_interact(obs, desired_direction, movement_action):
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    return jnp.where(facing == desired_direction, INTERACT, movement_action)

def _global_value(obs, pos, row, col, channel):
    lr = jnp.asarray(row, dtype=jnp.int32) - pos[0] + 2
    lc = jnp.asarray(col, dtype=jnp.int32) - pos[1] + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    value = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), channel]
    return jnp.where(visible, value, jnp.int32(0))

def init_agent_0():
    # locked target, previous cooked flag, steps, cue candidate, cue run,
    # no-cue age, delivery attempts, partner-seen, last partner row/col/dir,
    # and whether deliberate partner activity has been observed; then signal
    # mapping, confirmed flag, symbol used, pending outcome, same-cue streak,
    # last delivered type, and whether the current dish is an inverse probe
    return jnp.asarray((-1, 0, 0, -1, 0, 0, 0, 0, -1, -1, -1, 0,
        0, 0, -1, 0, 0, -1, 0), dtype=jnp.int32)

def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    pos = _position(obs)
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw0 = obs[2, 2, 7] > 0
    held_raw1 = obs[2, 2, 8] > 0
    held_raw = held_raw0 | held_raw1

    partner_visible = jnp.sum(obs[:, :, 11]) > 0
    partner_flat = jnp.argmax(obs[:, :, 11])
    partner_lr = (partner_flat // 5).astype(jnp.int32)
    partner_lc = (partner_flat % 5).astype(jnp.int32)
    partner_row = pos[0] + partner_lr - 2
    partner_col = pos[1] + partner_lc - 2
    partner_dir = jnp.argmax(obs[partner_lr, partner_lc, 12:16]).astype(jnp.int32)
    partner_up = obs[partner_lr, partner_lc, 12] > 0
    partner_right = obs[partner_lr, partner_lc, 14] > 0
    partner_left = obs[partner_lr, partner_lc, 15] > 0
    at_interface = partner_visible & (partner_row == 4)
    # Accept both common encodings after debouncing: ordinal left/right facing,
    # or simply occupying the left/right side of the pot interface.
    left_signal = at_interface & (partner_left | (partner_col <= 1))
    right_signal = at_interface & (partner_right | (partner_col >= 3))
    raw_cue = jnp.where(left_signal, jnp.int32(0),
        jnp.where(right_signal, jnp.int32(1), jnp.int32(-1)))
    partner_seen = memory[7] > 0
    pose_changed = partner_visible & partner_seen & (
        (partner_row != memory[8]) | (partner_col != memory[9]) |
        (partner_dir != memory[10]))
    partner_active = (memory[11] > 0) | pose_changed
    next_seen = partner_seen | partner_visible
    last_partner_row = jnp.where(partner_visible, partner_row, memory[8])
    last_partner_col = jnp.where(partner_visible, partner_col, memory[9])
    last_partner_dir = jnp.where(partner_visible, partner_dir, memory[10])
    same_cue = (raw_cue >= 0) & (raw_cue == memory[3])
    cue_run = jnp.where(raw_cue < 0, jnp.int32(0),
        jnp.where(same_cue, memory[4] + 1, jnp.int32(1)))
    cue_candidate = jnp.where(raw_cue >= 0, raw_cue, memory[3])
    stable_cue = partner_active & (raw_cue >= 0) & (cue_run >= 2)
    # A stable outward-facing pose at the corresponding blocked edge has an
    # intrinsic ordinal meaning, unlike an arbitrary pot-index convention.
    semantic_natural = stable_cue & (
        ((raw_cue == 0) & partner_left & (partner_col == 1)) |
        ((raw_cue == 1) & partner_right & (partner_col == 4)))
    base_mapping = jnp.where(semantic_natural, jnp.int32(0), memory[12])
    base_confirmed = (memory[13] > 0) | semantic_natural

    # Evaluate the previous delivery once a stable post-delivery cue arrives.
    # A changed cue proves the delivery was correct and identifies the mapping.
    evaluating = stable_cue & (memory[0] < 0) & (memory[15] > 0) & (memory[14] >= 0)
    cue_changed = evaluating & (raw_cue != memory[14])
    cue_same = evaluating & (raw_cue == memory[14])
    inferred_mapping = memory[17] ^ memory[14]
    ordinary_same = cue_same & (~base_confirmed) & (memory[18] == 0)
    streak_inc = memory[16] + 1
    start_probe = ordinary_same & (streak_inc >= 3)
    end_failed_probe = cue_same & (~base_confirmed) & (memory[18] > 0)
    mapping = jnp.where(cue_changed, inferred_mapping,
        jnp.where(start_probe | end_failed_probe, 1 - base_mapping, base_mapping))
    mapping = jnp.where(semantic_natural, jnp.int32(0), mapping)
    mapping_confirmed = base_confirmed | cue_changed
    same_streak = jnp.where(evaluating,
        jnp.where(ordinary_same & (~start_probe), streak_inc, jnp.int32(0)), memory[16])
    accepting = (memory[0] < 0) & stable_cue
    new_probe = accepting & start_probe
    mapped_cue = raw_cue ^ mapping
    # Once accepted, never redirect a partially executed dish because of noisy
    # partner motion. The target is unlocked only after delivery.
    accepted_recipe = jnp.where(accepting, mapped_cue, memory[0])
    just_delivered = (memory[1] > 0) & (~held_cooked)
    recipe = jnp.where(just_delivered, jnp.int32(-1), accepted_recipe)
    no_cue_age = jnp.where(just_delivered | stable_cue, jnp.int32(0), memory[5] + 1)
    delivery_attempts = memory[6] + just_delivered.astype(jnp.int32)
    fallback_delay = jnp.where((delivery_attempts > 0) & (~partner_active),
        jnp.int32(2), jnp.int32(24))
    fallback = (recipe < 0) & (no_cue_age >= fallback_delay)
    recipe = jnp.where(fallback, delivery_attempts % 2, recipe)
    locked_symbol = jnp.where(accepting, raw_cue, memory[14])
    locked_symbol = jnp.where(fallback, jnp.int32(-1), locked_symbol)
    pending_outcome = jnp.where(just_delivered, (memory[14] >= 0).astype(jnp.int32),
        jnp.where(accepting, jnp.int32(0), memory[15]))
    last_delivered = jnp.where(just_delivered, memory[0], memory[17])
    current_probe = jnp.where(just_delivered, memory[18],
        jnp.where(accepting, new_probe.astype(jnp.int32),
            jnp.where(fallback, jnp.int32(0), memory[18])))

    valid_recipe = recipe >= 0
    safe_recipe = jnp.clip(recipe, 0, 1)
    count0 = _global_value(obs, pos, 3, 1, 34)
    count1 = _global_value(obs, pos, 3, 3, 35)
    timer0 = _global_value(obs, pos, 3, 1, 44)
    timer1 = _global_value(obs, pos, 3, 3, 44)
    cooked0 = _global_value(obs, pos, 3, 1, 33) > 0
    cooked1 = _global_value(obs, pos, 3, 3, 33) > 0
    available0 = (count0 < 3) & (timer0 == 0) & (~cooked0)
    available1 = (count1 < 3) & (timer1 == 0) & (~cooked1)

    target_available = jnp.where(safe_recipe == 0, available0, available1)
    other_available = jnp.where(safe_recipe == 0, available1, available0)
    target_cooked = jnp.where(safe_recipe == 0, cooked0, cooked1)
    # Prefer finishing the requested pot. While it cooks, prepare the other
    # pure recipe so it may satisfy the next sampled target immediately.
    fill_type_with_target = jnp.where(target_available, safe_recipe, 1 - safe_recipe)
    fill_type_without_target = jnp.where(available0, jnp.int32(0), jnp.int32(1))
    fill_type = jnp.where(valid_recipe, fill_type_with_target, fill_type_without_target)

    raw_type = jnp.where(held_raw1, jnp.int32(1), jnp.int32(0))
    work_type = jnp.where(held_raw, raw_type, safe_recipe)
    work_pot_col = jnp.int32(1) + 2 * work_type
    pot_stand = jnp.asarray((2, work_pot_col), dtype=jnp.int32)
    at_pot = jnp.all(pos == pot_stand)
    pot_action = jnp.where(at_pot,
        _face_then_interact(obs, jnp.int32(1), DOWN), _move_to(pos, pot_stand))
    goal_stand = jnp.asarray((1, 2), dtype=jnp.int32)
    at_goal = jnp.all(pos == goal_stand)
    deliver_action = jnp.where(at_goal,
        _face_then_interact(obs, jnp.int32(0), UP), _move_to(pos, goal_stand))
    plate_action = jnp.where(at_pot & (~target_cooked), STAY, pot_action)

    plate_stand = jnp.asarray((1, 3), dtype=jnp.int32)
    at_plate = jnp.all(pos == plate_stand)
    get_plate = jnp.where(at_plate,
        _face_then_interact(obs, jnp.int32(0), UP), _move_to(pos, plate_stand))
    ingredient_stand = jnp.where(fill_type == 0,
        jnp.asarray((1, 1), dtype=jnp.int32), jnp.asarray((2, 4), dtype=jnp.int32))
    ingredient_direction = jnp.where(fill_type == 0, jnp.int32(3), jnp.int32(2))
    ingredient_move = jnp.where(fill_type == 0, LEFT, RIGHT)
    at_ingredient = jnp.all(pos == ingredient_stand)
    get_ingredient = jnp.where(at_ingredient,
        _face_then_interact(obs, ingredient_direction, ingredient_move),
        _move_to(pos, ingredient_stand))
    can_fill = jnp.where(valid_recipe, target_available | other_available,
                         available0 | available1)
    empty_with_target = jnp.where(target_cooked, get_plate,
        jnp.where(can_fill, get_ingredient, get_plate))
    # With no understood cue, productively fill both pure pots, then wait at a
    # central observation point until a cue arrives or the fallback activates.
    empty_without_target = jnp.where(can_fill, get_ingredient,
        _move_to(pos, jnp.asarray((2, 2), dtype=jnp.int32)))
    empty_action = jnp.where(valid_recipe, empty_with_target, empty_without_target)

    action = jnp.where(held_cooked, deliver_action,
        jnp.where(held_raw, pot_action,
            jnp.where(held_plate, plate_action, empty_action)))
    next_memory = jnp.asarray((recipe, held_cooked.astype(jnp.int32), memory[2] + 1,
        cue_candidate, cue_run, no_cue_age, delivery_attempts,
        next_seen.astype(jnp.int32), last_partner_row, last_partner_col,
        last_partner_dir, partner_active.astype(jnp.int32), mapping,
        mapping_confirmed.astype(jnp.int32), locked_symbol, pending_outcome,
        same_streak, last_delivered, current_probe), dtype=jnp.int32)
    trace = {"position": pos, "recipe": recipe, "pot0": count0, "pot1": count1,
        "no_cue_age": no_cue_age}
    return action.astype(jnp.int32), next_memory, trace

def init_agent_1():
    # displayed recipe, elapsed steps, steps since the display changed
    return jnp.asarray((-1, 0, 0), dtype=jnp.int32)

def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    pos = _position(obs)
    recipe0 = jnp.sum(obs[:, :, 40])
    recipe1 = jnp.sum(obs[:, :, 41])
    display_visible = (recipe0 + recipe1) > 0
    displayed = jnp.where(recipe0 > recipe1, jnp.int32(0), jnp.int32(1))
    recipe = jnp.where(display_visible, displayed, memory[0])
    safe_recipe = jnp.clip(recipe, 0, 1)
    changed = display_visible & (displayed != memory[0])
    recipe_age = jnp.where(changed, jnp.int32(0), memory[2] + 1)
    direction_phase = (recipe_age % 16) >= 12
    # In the position phase, wait immediately under the selected pot facing it.
    # In the direction phase, use a blocked outward-facing pose. Recipe 0 can
    # express both at (4,1); recipe 1 moves one cell right to (4,4).
    pot_col = jnp.int32(1) + 2 * safe_recipe
    edge_col = jnp.where(safe_recipe == 0, jnp.int32(1), jnp.int32(4))
    signal_col = jnp.where(direction_phase, edge_col, pot_col)
    signal_cell = jnp.asarray((4, signal_col), dtype=jnp.int32)
    at_signal = jnp.all(pos == signal_cell)
    horizontal = jnp.where(pos[1] < signal_col, RIGHT, LEFT)
    vertical = jnp.where(pos[0] < signal_cell[0], DOWN, UP)
    # Align the column before entering row 4, avoiding a one-frame pose below
    # the other pot that could be mistaken for a signal.
    approach = jnp.where(pos[1] != signal_col, horizontal, vertical)
    direction_action = jnp.where(safe_recipe == 0, LEFT, RIGHT)
    signal_action = jnp.where(direction_phase, direction_action, UP)
    action = jnp.where(recipe < 0, STAY,
        jnp.where(at_signal, signal_action, approach))
    next_memory = jnp.asarray((recipe, memory[1] + 1, recipe_age), dtype=jnp.int32)
    trace = {"position": pos, "recipe": recipe}
    return action.astype(jnp.int32), next_memory, trace
