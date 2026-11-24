"""Convention-robust controller for the two natural roles in demo_cook_simple."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "adaptive_natural_roles",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 11,
    "num_ingredients": 3,
}


RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _static_map():
    """The public layout, encoded in observation static-channel order 20:29."""
    grid = jnp.zeros((5, 11, 9), dtype=jnp.int32)
    walls = (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9),
        (0, 10), (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2),
        (4, 3), (4, 4), (4, 7), (4, 9), (4, 10),
    )
    for r, c in walls:
        grid = grid.at[r, c, 0].set(1)
    grid = grid.at[2, 10, 1].set(1)
    grid = grid.at[2, 7, 2].set(1)
    for r, c in ((0, 5), (4, 5)):
        grid = grid.at[r, c, 3].set(1)
    for r, c in ((1, 10), (3, 10)):
        grid = grid.at[r, c, 5].set(1)
    for r, c in ((0, 8), (1, 0)):
        grid = grid.at[r, c, 6].set(1)
    for r, c in ((3, 0), (4, 8)):
        grid = grid.at[r, c, 7].set(1)
    for r, c in ((0, 6), (4, 6)):
        grid = grid.at[r, c, 8].set(1)
    return grid


_POS0_TUPLES = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
)
_POS1_TUPLES = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))


def _localize(obs, position_tuples):
    padded = jnp.pad(_static_map(), ((2, 2), (2, 2), (0, 0)))
    positions = jnp.asarray(position_tuples, dtype=jnp.int32)
    templates = jnp.stack([padded[r:r + 5, c:c + 5, :] for r, c in position_tuples])
    observed = obs[:, :, 20:29]
    error = jnp.sum(jnp.abs(templates - observed[None, :, :, :]), axis=(1, 2, 3))
    return positions[jnp.argmin(error)]


def _navigate(row, col, target_row, target_col, face_action, obs):
    """Walk Manhattan-style, then face an adjacent fixture and interact."""
    face_channel = jnp.where(
        face_action == UP,
        1,
        jnp.where(face_action == DOWN, 2, jnp.where(face_action == RIGHT, 3, 4)),
    )
    at_target = (row == target_row) & (col == target_col)
    facing = obs[2, 2, face_channel] > 0
    final_action = jnp.where(facing, INTERACT, face_action)
    move = jnp.where(
        row < target_row,
        DOWN,
        jnp.where(
            row > target_row,
            UP,
            jnp.where(col < target_col, RIGHT, jnp.where(col > target_col, LEFT, final_action)),
        ),
    )
    return jnp.where(at_target, final_action, move).astype(jnp.int32)


def _go_wait(row, col, target_row, target_col):
    return jnp.where(
        row < target_row,
        DOWN,
        jnp.where(
            row > target_row,
            UP,
            jnp.where(col < target_col, RIGHT, jnp.where(col > target_col, LEFT, STAY)),
        ),
    ).astype(jnp.int32)


def _pot(obs):
    mask = obs[:, :, 22]
    present = jnp.sum(mask) > 0
    counts = jnp.sum(obs[:, :, 31:34] * mask[:, :, None], axis=(0, 1))
    cooked = jnp.sum(obs[:, :, 30] * mask) > 0
    timer = jnp.sum(obs[:, :, 39] * mask)
    return present, counts, jnp.sum(counts), cooked, timer


def init_agent_0():
    # recipe, unchanged-pot ticks, last pot count, seen nonempty,
    # post-collection grace ticks, staging ingredient (-1/0/1), previously
    # holding raw, elapsed ticks, observed dish carrier, off-screen delivery,
    # delivery style (-1 unknown, 0 direct, 1 slow)
    return jnp.asarray(
        [-1, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1], dtype=jnp.int32
    )


def agent_0_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, init_agent_0(), memory)
    row, col = _localize(obs, _POS0_TUPLES)
    present, pot_counts, pot_count, pot_cooked, pot_timer = _pot(obs)
    del pot_counts

    displayed = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    recipe_seen = jnp.sum(displayed) > 0
    displayed_recipe = jnp.argmax(displayed[:2]).astype(jnp.int32)
    recipe_changed = recipe_seen & (mem[0] >= 0) & (displayed_recipe != mem[0])
    recipe = jnp.where(recipe_seen, displayed_recipe, mem[0])

    count_changed = present & (pot_count != mem[2])
    stable = jnp.where(present, jnp.where(count_changed, 0, mem[1] + 1), mem[1])
    last_count = jnp.where(present, pot_count, mem[2])
    emptied = present & (pot_count == 0) & (mem[3] > 0)
    seen_nonempty = jnp.where(
        present & (pot_count > 0), 1, jnp.where(emptied, 0, mem[3])
    )
    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_has_plate = jnp.sum(obs[:, :, 15]) > 0
    partner_has_dish = jnp.sum(obs[:, :, 16]) > 0
    service_evidence = (
        partner_visible
        & partner_has_plate
        & (pot_cooked | (pot_timer > 0) | (pot_count >= 3))
    )
    # Once the previously nonempty pot is observed empty, a visible right
    # partner with no plate cannot still be carrying the collected dish. This
    # also covers the case where staging made us miss the collection itself.
    returned_empty = partner_visible & (~partner_has_plate)
    remembered_delivery = (mem[9] > 0) | (recipe_changed & (mem[3] > 0))
    delivery_confirmed = ((mem[4] > 0) | emptied) & (
        recipe_changed | returned_empty | remembered_delivery
    )
    delivery_style = jnp.where(
        (mem[10] < 0) & delivery_confirmed & (mem[8] > 0),
        jnp.where(mem[4] >= 14, 0, 1),
        mem[10],
    )
    initial_grace = jnp.where(delivery_style == 0, 2, 16)
    grace = jnp.where(
        emptied,
        jnp.where(delivery_confirmed, 0, initial_grace),
        jnp.where(delivery_confirmed, 0, jnp.maximum(mem[4] - 1, 0)),
    )
    saw_dish = jnp.where(
        delivery_confirmed,
        0,
        jnp.where(
            service_evidence | partner_has_dish,
            1,
            jnp.where((grace == 0) & (pot_count == 0), 0, mem[8]),
        ),
    )
    delivery_flag = jnp.where(emptied, 0, remembered_delivery)

    inv = obs[2, 2, 5:10]
    held_raw = jnp.sum(inv[2:5]) > 0
    completed_transfer = (~held_raw) & (mem[6] > 0)
    mode = jnp.where(completed_transfer, -1, mem[5])
    pot_available = (~present) | (
        present & (pot_count < 3) & (~pot_cooked) & (pot_timer == 0)
    )

    add_action = _navigate(row, col, 2, 6, RIGHT, obs)
    park_action = _navigate(row, col, 1, 4, UP, obs)
    staged_row = jnp.where(mode == 0, 1, 3)
    stage_deposit_action = _navigate(row, col, staged_row, 6, RIGHT, obs)
    regular_raw_action = jnp.where(pot_available, add_action, park_action)
    raw_action = jnp.where(mode >= 0, stage_deposit_action, regular_raw_action)

    # Once the recipe is remembered, absence of the pot means we are simply on
    # the supply run already chosen while it was visible.
    know_state = recipe >= 0
    busy = pot_cooked | (pot_timer > 0) | (pot_count >= 3)
    # Column 6 sees the recipe indicators, pot, and the right pot-adjacent
    # cell, allowing delivery confirmation without a private timer.
    wait_action = _go_wait(row, col, 2, 6)
    supply_row = jnp.where(recipe == 0, 1, 3)
    distant_fetch_action = _navigate(row, col, supply_row, 1, LEFT, obs)

    # The two wall cells next to the pot are pass-through counters. During
    # cooking agent 0 keeps ingredient 0 on the upper one and ingredient 1 on
    # the lower one, making the next recipe's first ingredient a short pickup.
    upper_r = jnp.clip(1 - row + 2, 0, 4)
    lower_r = jnp.clip(3 - row + 2, 0, 4)
    counter_c = jnp.clip(7 - col + 2, 0, 4)
    upper_visible = (jnp.abs(1 - row) <= 2) & (jnp.abs(7 - col) <= 2)
    lower_visible = (jnp.abs(3 - row) <= 2) & (jnp.abs(7 - col) <= 2)
    upper_total = jnp.sum(obs[upper_r, counter_c, 29:34])
    lower_total = jnp.sum(obs[lower_r, counter_c, 29:34])
    upper_empty = upper_visible & (upper_total == 0)
    lower_empty = lower_visible & (lower_total == 0)
    staged_match = jnp.where(
        recipe == 0,
        upper_visible & (obs[upper_r, counter_c, 31] > 0),
        lower_visible & (obs[lower_r, counter_c, 32] > 0),
    )
    pickup_row = jnp.where(recipe == 0, 1, 3)
    staged_pickup_action = _navigate(row, col, pickup_row, 6, RIGHT, obs)
    fetch_action = jnp.where(staged_match, staged_pickup_action, distant_fetch_action)

    missing_stage = upper_empty | lower_empty
    stage_id = jnp.where(upper_empty, 0, 1)
    choose_stage = (mode < 0) & busy & missing_stage & know_state
    active_mode = jnp.where(choose_stage, stage_id, mode)
    stage_supply_row = jnp.where(active_mode == 0, 1, 3)
    stage_supply_action = _navigate(row, col, stage_supply_row, 1, LEFT, obs)
    should_fetch = ((pot_count == 0) & (grace == 0)) | (
        (pot_count > 0) & (pot_count < 3) & (stable >= 8)
    )
    empty_action = jnp.where(
        active_mode >= 0,
        stage_supply_action,
        jnp.where(
            ~know_state,
            wait_action,
            jnp.where(busy, wait_action, jnp.where(should_fetch, fetch_action, wait_action)),
        ),
    )
    action = jnp.where(held_raw, raw_action, empty_action).astype(jnp.int32)

    next_memory = jnp.asarray(
        [
            recipe,
            stable,
            last_count,
            seen_nonempty,
            grace,
            active_mode,
            held_raw,
            mem[7] + 1,
            saw_dish,
            delivery_flag,
            delivery_style,
        ],
        dtype=jnp.int32,
    )
    trace = {
        "action": action,
        "decision": jnp.asarray(
            [supply_row, should_fetch, fetch_action, know_state, active_mode],
            dtype=jnp.int32,
        ),
        "position": jnp.asarray([row, col], dtype=jnp.int32),
        "pot": jnp.asarray([pot_count, pot_timer, pot_cooked], dtype=jnp.int32),
        "recipe_stable_grace": jnp.asarray([recipe, stable, grace], dtype=jnp.int32),
    }
    return action, next_memory, trace


def init_agent_1():
    # elapsed, last pot count, one-tick convention probe, persistent partner
    # style (-1 unknown, 0 waits for help, 1 keeps loading), stable-count ticks
    return jnp.asarray([0, -1, 0, -1, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, init_agent_1(), memory)
    row, col = _localize(obs, _POS1_TUPLES)
    present, pot_counts, pot_count, pot_cooked, pot_timer = _pot(obs)
    del present

    inv = obs[2, 2, 5:10]
    held_plate = inv[0] > 0
    held_cooked = inv[1] > 0
    raw_counts = inv[2:5]
    held_raw = jnp.sum(raw_counts) > 0
    held_id = jnp.argmax(raw_counts).astype(jnp.int32)

    count_changed = pot_count != mem[1]
    count_stable = jnp.where(count_changed, 0, mem[4] + 1)
    first_insert = (
        (pot_count == 1) & (mem[1] == 0) & (pot_timer == 0) & (mem[3] < 0)
    )
    # One step after the first insertion, an independently loading left
    # partner is visible leaving along (row 1 or 3, column 6). A partner that
    # expects assistance instead moves to column 5 and waits out of view.
    partner_c = jnp.clip(6 - col + 2, 0, 4)
    upper_partner_r = jnp.clip(1 - row + 2, 0, 4)
    lower_partner_r = jnp.clip(3 - row + 2, 0, 4)
    departing_partner = (
        ((jnp.abs(6 - col) <= 2) & (jnp.abs(1 - row) <= 2)
         & (obs[upper_partner_r, partner_c, 10] > 0))
        | ((jnp.abs(6 - col) <= 2) & (jnp.abs(3 - row) <= 2)
           & (obs[lower_partner_r, partner_c, 10] > 0))
    )
    resolving_probe = (mem[2] > 0) & (pot_count == 1)
    style = jnp.where(resolving_probe, departing_partner.astype(jnp.int32), mem[3])
    aggressive = style > 0
    probe = jnp.where(first_insert, 1, jnp.where(resolving_probe, 0, mem[2]))

    deliver_action = _navigate(row, col, 2, 9, RIGHT, obs)
    pot_action = _navigate(row, col, 2, 8, LEFT, obs)
    plate_row = jnp.where(row <= 2, 1, 3)
    get_plate_action = _navigate(row, col, plate_row, 9, RIGHT, obs)
    plate_action = jnp.where(pot_cooked, pot_action, _go_wait(row, col, 2, 8))

    desired_id = jnp.argmax(pot_counts[:2]).astype(jnp.int32)
    desired_row = jnp.where(desired_id == 0, 1, 3)
    local_counter_r = jnp.clip(desired_row - row + 2, 0, 4)
    local_counter_c = jnp.clip(7 - col + 2, 0, 4)
    counter_match = obs[local_counter_r, local_counter_c, 31 + desired_id] > 0
    take_counter = _navigate(row, col, desired_row, 8, LEFT, obs)
    supply_face = jnp.where(desired_id == 0, UP, DOWN)
    take_supply = _navigate(row, col, desired_row, 8, supply_face, obs)
    get_ingredient_action = jnp.where(counter_match, take_counter, take_supply)

    partial = (pot_count > 0) & (pot_count < 3) & (~pot_cooked) & (pot_timer == 0)
    defer_to_returning_partner = aggressive & (pot_count >= 2) & (count_stable < 15)
    should_help = partial & (~first_insert) & (~defer_to_returning_partner)
    empty_action = jnp.where(
        should_help,
        get_ingredient_action,
        jnp.where(
            pot_cooked | (pot_timer > 0) | (pot_count >= 3),
            get_plate_action,
            _go_wait(row, col, 2, 8),
        ),
    )

    held_matches = pot_counts[held_id] > 0
    can_add = partial & held_matches
    park_row = jnp.where(held_id == 0, 1, 3)
    park_action = _navigate(row, col, park_row, 8, LEFT, obs)
    raw_action = jnp.where(can_add, pot_action, park_action)

    action = jnp.where(
        held_cooked,
        deliver_action,
        jnp.where(held_raw, raw_action, jnp.where(held_plate, plate_action, empty_action)),
    ).astype(jnp.int32)
    next_memory = jnp.asarray(
        [mem[0] + 1, pot_count, probe, style, count_stable], dtype=jnp.int32
    )
    trace = {
        "action": action,
        "position": jnp.asarray([row, col], dtype=jnp.int32),
        "pot": jnp.asarray([pot_count, pot_timer, pot_cooked], dtype=jnp.int32),
        "inventory": inv.astype(jnp.int32),
        "coordination": jnp.asarray(
            [probe, aggressive, count_stable, should_help], dtype=jnp.int32
        ),
    }
    return action, next_memory, trace
