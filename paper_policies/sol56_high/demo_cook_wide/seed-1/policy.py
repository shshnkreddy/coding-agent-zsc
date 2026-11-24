"""Reactive, observation-grounded policy for the demo_cook_wide layout."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'reactive_role_robust',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# Actions.
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _make_lower_templates():
    """Static 5x5 crops for every reachable lower-room floor cell."""
    grid = [[[0] * 10 for _ in range(11)] for _ in range(6)]
    walls = (
        (0,0),(0,1),(0,2),(0,3),(0,7),(0,8),(0,9),(0,10),
        (1,0),(1,1),(1,2),(1,8),(1,9),(1,10),
        (2,0),(2,1),(2,2),(2,3),(2,4),(2,6),(2,7),(2,8),(2,9),(2,10),
        (3,0),(3,10),(4,3),(4,7),
        (5,0),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,10),
    )
    for r, c in walls:
        grid[r][c][0] = 1
    for r, c, k in (
        (0,5,1), (2,5,2), (4,5,3), (0,4,5), (0,6,5),
        (1,3,6), (4,0,6), (4,10,6),
        (1,7,7), (5,1,7), (5,9,7),
        (4,4,9), (4,6,9),
    ):
        grid[r][c][k] = 1
    positions = [(3, c) for c in range(1, 10)] + [(4,1),(4,2),(4,8),(4,9)]
    crops = []
    for pr, pc in positions:
        crop = []
        for dr in range(-2, 3):
            row = []
            for dc in range(-2, 3):
                rr, cc = pr + dr, pc + dc
                row.append(grid[rr][cc] if 0 <= rr < 6 and 0 <= cc < 11 else [0] * 10)
            crop.append(row)
        crops.append(crop)
    return jnp.asarray(positions, dtype=jnp.int32), jnp.asarray(crops, dtype=jnp.int32)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_or_interact(facing, desired_facing, turn_action):
    return jnp.where(facing == desired_facing, INTERACT, turn_action)


def _upper_col(obs):
    left_at_one = obs[2, 1, 28] > 0
    left_at_zero = obs[2, 0, 28] > 0
    return jnp.where(left_at_one, 4, jnp.where(left_at_zero, 5, 6)).astype(jnp.int32)


def _upper_interact(col, facing, stand_col, desired_facing, turn_action):
    move = jnp.where(col < stand_col, RIGHT, LEFT)
    return jnp.where(col == stand_col,
                     _face_or_interact(facing, desired_facing, turn_action),
                     move).astype(jnp.int32)


def _lower_position(obs):
    lower_positions, lower_templates = _make_lower_templates()
    crop = obs[:, :, 22:32]
    mismatch = jnp.sum(jnp.abs(lower_templates - crop[None, :, :, :]), axis=(1, 2, 3))
    return lower_positions[jnp.argmin(mismatch)]


def _lower_go(row, col, facing, stand_row, stand_col, desired_facing, turn_action):
    """Route on the lower corridor/side alcoves to an interaction square."""
    at_target = (row == stand_row) & (col == stand_col)
    target_action = _face_or_interact(facing, desired_facing, turn_action)
    from_alcove = jnp.where(row == 4, UP, STAY)
    horizontal = jnp.where(col < stand_col, RIGHT, LEFT)
    on_corridor = jnp.where(col == stand_col,
                            jnp.where(stand_row == 4, DOWN, target_action),
                            horizontal)
    route = jnp.where(row == 4, from_alcove, on_corridor)
    return jnp.where(at_target, target_action, route).astype(jnp.int32)


def init_agent_0():
    # [elapsed decisions, alternating blind fallback target,
    #  empty-pot evidence wait, partner-was-raw, last partner type,
    #  current-cycle signal valid, current-cycle signal type]
    return jnp.asarray([0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, init_agent_0(), memory)
    step, fallback, empty_wait = base[0], base[1], base[2]
    partner_was_raw, last_partner_type = base[3], base[4]
    signal_valid, signal_type = base[5], base[6]
    col = _upper_col(obs)
    facing = _facing(obs)

    inv = obs[2, 2, 5:11]
    held_plate = inv[0] > 0
    held_cooked = inv[1] > 0
    held_raw_count = jnp.sum(inv[2:])
    held_raw = held_raw_count > 0
    held_type = jnp.argmax(inv[2:]).astype(jnp.int32)

    pot_mask = obs[:, :, 24:25]
    pot_counts = jnp.sum(obs[:, :, 34:38] * pot_mask, axis=(0, 1))
    pot_total = jnp.sum(pot_counts)
    pot_type = jnp.argmax(pot_counts).astype(jnp.int32)
    pot_cooked = jnp.sum(obs[:, :, 33] * obs[:, :, 24]) > 0
    pot_timer = jnp.sum(obs[:, :, 44])
    pot_busy = pot_cooked | (pot_timer > 0)

    partner_counts = jnp.sum(obs[:, :, 18:22], axis=(0, 1))
    partner_has_raw = ((jnp.sum(partner_counts) > 0)
                       & (jnp.sum(obs[:, :, 16:18]) == 0))
    partner_type = jnp.argmax(partner_counts).astype(jnp.int32)
    partner_visible = jnp.sum(obs[:, :, 11]) > 0
    fresh_signal = (partner_has_raw
                    & ((partner_was_raw == 0) | (partner_type != last_partner_type)))
    active_signal_valid = jnp.where(fresh_signal, 1, signal_valid)
    active_signal_type = jnp.where(fresh_signal, partner_type, signal_type)
    chosen = jnp.where(pot_total > 0, pot_type,
                       jnp.where(active_signal_valid > 0, active_signal_type, fallback))

    lc4 = 2 + (4 - col)
    lc6 = 2 + (6 - col)
    counter4_counts = obs[3, lc4, 34:38]
    counter6_counts = obs[3, lc6, 34:38]
    counter4_total = jnp.sum(obs[3, lc4, 32:38])
    counter6_total = jnp.sum(obs[3, lc6, 32:38])
    counter4_match = counter4_counts[chosen] > 0
    counter6_match = counter6_counts[chosen] > 0

    ingredient_action = jnp.where(
        counter4_match,
        _upper_interact(col, facing, jnp.int32(4), jnp.int32(1), DOWN),
        jnp.where(
            counter6_match,
            _upper_interact(col, facing, jnp.int32(6), jnp.int32(1), DOWN),
            jnp.where(chosen == 0,
                      _upper_interact(col, facing, jnp.int32(4), jnp.int32(3), LEFT),
                      _upper_interact(col, facing, jnp.int32(6), jnp.int32(2), RIGHT))))
    plate_stand = jnp.where(col <= 5, 4, 6).astype(jnp.int32)
    get_plate_action = _upper_interact(col, facing, plate_stand, jnp.int32(0), UP)
    wait_for_evidence = ((pot_total == 0) & (~pot_busy) & (active_signal_valid == 0)
                         & (empty_wait < 9))
    empty_action = jnp.where(pot_busy, get_plate_action,
                             jnp.where(wait_for_evidence, STAY, ingredient_action))

    empty_pot_agrees = ((pot_total == 0)
                        & ((active_signal_valid == 0) | (active_signal_type == held_type)))
    raw_matches = empty_pot_agrees | ((pot_total > 0) & (pot_type == held_type)
                                      & (~pot_busy) & (pot_total < 3))
    add_action = _upper_interact(col, facing, jnp.int32(5), jnp.int32(1), DOWN)
    discard4 = _upper_interact(col, facing, jnp.int32(4), jnp.int32(1), DOWN)
    discard6 = _upper_interact(col, facing, jnp.int32(6), jnp.int32(1), DOWN)
    discard_action = jnp.where(counter4_total == 0, discard4,
                               jnp.where(counter6_total == 0, discard6, STAY))
    raw_action = jnp.where(raw_matches, add_action, discard_action)

    plate_to_pot = _upper_interact(col, facing, jnp.int32(5), jnp.int32(1), DOWN)
    move_to_pot = jnp.where(col < 5, RIGHT, jnp.where(col > 5, LEFT, STAY))
    preposition_plate = jnp.where(
        col != 5,
        move_to_pot,
        jnp.where(pot_timer == 1, DOWN, STAY))
    held_plate_action = jnp.where(pot_cooked, plate_to_pot, preposition_plate)
    deliver_action = _upper_interact(col, facing, jnp.int32(5), jnp.int32(0), UP)

    action = jnp.where(held_cooked, deliver_action,
                       jnp.where(held_plate, held_plate_action,
                                 jnp.where(held_raw, raw_action, empty_action))).astype(jnp.int32)

    delivering = held_cooked & (col == 5) & (facing == 0)
    next_fallback = jnp.where(delivering, 1 - fallback, fallback)
    hands_empty = (~held_cooked) & (~held_plate) & (~held_raw)
    empty_no_evidence = (hands_empty & (pot_total == 0) & (~pot_busy)
                         & (active_signal_valid == 0))
    next_empty_wait = jnp.where(empty_no_evidence,
                                jnp.minimum(empty_wait + 1, 9), 0)
    next_partner_was_raw = jnp.where(
        partner_visible, partner_has_raw.astype(jnp.int32), partner_was_raw)
    next_last_partner_type = jnp.where(partner_has_raw, partner_type, last_partner_type)
    next_signal_valid = jnp.where(pot_total > 0, 0, active_signal_valid)
    next_memory = jnp.asarray([step + 1, next_fallback, next_empty_wait,
                               next_partner_was_raw, next_last_partner_type,
                               next_signal_valid, active_signal_type], dtype=jnp.int32)
    trace = {
        'chosen': chosen,
        'counter4': counter4_total.astype(jnp.int32),
        'counter6': counter6_total.astype(jnp.int32),
        'fallback': fallback,
        'held_raw': held_raw.astype(jnp.int32),
        'held_type': held_type,
        'pot_total': pot_total.astype(jnp.int32),
        'pos_col': col,
    }
    return action, next_memory, trace


def init_agent_1():
    # [elapsed decisions, last observed target, contributed-this-pot,
    #  discard side, previous pot count, stall age, upper-raw-seen, cycles]
    return jnp.asarray([0, -1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, init_agent_1(), memory)
    step, remembered_target = base[0], base[1]
    contributed, discard_side = base[2], base[3]
    previous_pot_total, stall_age = base[4], base[5]
    upper_raw_seen, completed_cycles = base[6], base[7]
    row, col = _lower_position(obs)
    facing = _facing(obs)

    recipe_counts = jnp.sum(obs[:, :, 40:44], axis=(0, 1))
    recipe_visible = jnp.sum(recipe_counts) > 0
    observed_target = jnp.argmax(recipe_counts).astype(jnp.int32)
    target = jnp.where(recipe_visible, observed_target, remembered_target).astype(jnp.int32)
    knows_target = target >= 0

    inv_counts = obs[2, 2, 7:11]
    held_raw = jnp.sum(inv_counts) > 0
    held_type = jnp.argmax(inv_counts).astype(jnp.int32)

    pot_mask = obs[:, :, 24:25]
    pot_counts = jnp.sum(obs[:, :, 34:38] * pot_mask, axis=(0, 1))
    pot_total = jnp.sum(pot_counts)
    pot_type = jnp.argmax(pot_counts).astype(jnp.int32)
    pot_cooked = jnp.sum(obs[:, :, 33] * obs[:, :, 24]) > 0
    pot_timer = jnp.sum(obs[:, :, 44])
    pot_visible = jnp.sum(obs[:, :, 24]) > 0
    pot_busy = pot_cooked | (pot_timer > 0)
    pot_accepts = (~pot_busy) & (pot_total < 3) & ((pot_total == 0) | (pot_type == target))
    # Once the pot has been cleared, the lower role may contribute to the new
    # cycle. Restricting it to one unit avoids carrying an old target through
    # the delivery transition.
    cycle_cleared = pot_visible & (pot_total == 0) & (~pot_busy)
    active_contributed = jnp.where(cycle_cleared, 0, contributed)
    pot_changed = pot_visible & (pot_total != previous_pot_total)
    active_stall = jnp.where(cycle_cleared | pot_changed | pot_busy, 0, stall_age)
    stall_limit = jnp.where((completed_cycles > 0) & (upper_raw_seen == 0), 0, 6)
    may_contribute = (active_contributed == 0) | (active_stall >= stall_limit)

    use_left = col <= 5
    stand_col = jnp.where(use_left, 1, 9).astype(jnp.int32)
    supply_facing = jnp.where(target == 0,
                              jnp.where(use_left, 3, 2), 1).astype(jnp.int32)
    supply_turn = jnp.where(target == 0,
                            jnp.where(use_left, LEFT, RIGHT), DOWN).astype(jnp.int32)
    fetch_action = _lower_go(row, col, facing, jnp.int32(4), stand_col,
                             supply_facing, supply_turn)
    center_action = _lower_go(row, col, facing, jnp.int32(3), jnp.int32(5),
                              jnp.int32(0), UP)

    empty_action = jnp.where(
        ~knows_target,
        center_action,
        jnp.where((~may_contribute) | (pot_visible & (~pot_accepts)),
                  center_action, fetch_action))
    held_matches_target = held_type == target
    held_can_add = held_matches_target & ((~pot_visible) | pot_accepts)
    # Eight distinct reachable wall counters. The first four are on the left,
    # where center-origin supply trips normally return; right-side counters
    # provide overflow capacity under repeated stale-item events.
    discard_cols = jnp.asarray([3, 3, 2, 1, 7, 7, 8, 9], dtype=jnp.int32)
    discard_facings = jnp.asarray([1, 0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32)
    discard_col = discard_cols[discard_side]
    discard_facing = discard_facings[discard_side]
    discard_turn = jnp.where(discard_facing == 1, DOWN, UP)
    discard_action = _lower_go(row, col, facing, jnp.int32(3), discard_col,
                               discard_facing, discard_turn)
    held_action = jnp.where(held_matches_target,
                            jnp.where(held_can_add, center_action, STAY),
                            discard_action)

    action = jnp.where(held_raw, held_action, empty_action).astype(jnp.int32)
    next_target = jnp.where(recipe_visible, observed_target, remembered_target)
    adding = held_raw & held_matches_target & pot_accepts & (row == 3) & (col == 5) & (facing == 0)
    discarding = (held_raw & (~held_matches_target) & (row == 3)
                  & (col == discard_col) & (facing == discard_facing))
    next_contributed = jnp.where(adding, 1, active_contributed)
    next_discard_side = jnp.where(discarding, (discard_side + 1) % 8, discard_side)
    next_previous_pot_total = jnp.where(pot_visible, pot_total, previous_pot_total)
    next_stall = jnp.where(
        ~pot_visible,
        active_stall,
        jnp.where((pot_total > 0) & (~pot_busy), active_stall + 1, 0))
    partner_has_raw = ((jnp.sum(obs[:, :, 18:22]) > 0)
                       & (jnp.sum(obs[:, :, 16:18]) == 0))
    next_upper_raw_seen = jnp.maximum(upper_raw_seen, partner_has_raw.astype(jnp.int32))
    cycle_completed = pot_visible & (previous_pot_total > 0) & (pot_total == 0)
    next_completed_cycles = completed_cycles + cycle_completed.astype(jnp.int32)
    next_memory = jnp.asarray([step + 1, next_target, next_contributed,
                               next_discard_side, next_previous_pot_total,
                               next_stall, next_upper_raw_seen,
                               next_completed_cycles], dtype=jnp.int32)
    trace = {
        'target': target,
        'pot_total': pot_total.astype(jnp.int32),
        'pos_row': row,
        'pos_col': col,
    }
    return action, next_memory, trace
