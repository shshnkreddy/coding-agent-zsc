"""Convention-robust policy for the two-room demo_cook_wide layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'ingredient_relay', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

_walls = (
    (0,0),(0,1),(0,2),(0,3),(0,7),(0,8),(0,9),(0,10),
    (1,0),(1,1),(1,2),(1,8),(1,9),(1,10),
    (2,0),(2,1),(2,2),(2,3),(2,4),(2,6),(2,7),(2,8),(2,9),(2,10),
    (3,0),(3,10),(4,3),(4,7),
    (5,0),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,10),
)

_floor_positions = (
    (1,4),(1,5),(1,6),
    (3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),
    (4,1),(4,2),(4,8),(4,9),
)
def _position(obs):
    static = jnp.zeros((6, 11, 10), dtype=jnp.int32)
    for r, c in _walls:
        static = static.at[r, c, 0].set(1)
    static = static.at[0, 5, 1].set(1)
    static = static.at[2, 5, 2].set(1)
    static = static.at[4, 5, 3].set(1)
    for r, c in ((0, 4), (0, 6)):
        static = static.at[r, c, 5].set(1)
    for r, c in ((1, 3), (4, 0), (4, 10)):
        static = static.at[r, c, 6].set(1)
    for r, c in ((1, 7), (5, 1), (5, 9)):
        static = static.at[r, c, 7].set(1)
    for r, c in ((4, 4), (4, 6)):
        static = static.at[r, c, 9].set(1)
    padded = jnp.pad(static, ((2,2),(2,2),(0,0)))
    templates = jnp.stack([padded[r:r+5, c:c+5] for r, c in _floor_positions])
    positions = jnp.asarray(_floor_positions, dtype=jnp.int32)
    mismatch = jnp.sum(jnp.abs(templates - obs[:, :, 22:32]), axis=(1,2,3))
    return positions[jnp.argmin(mismatch)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _upper_move_to_column(col, target):
    return jnp.where(col < target, RIGHT, jnp.where(col > target, LEFT, STAY))


def _face_then_interact(facing, wanted_facing, face_action):
    return jnp.where(facing == wanted_facing, INTERACT, face_action)


def init_agent_0():
    # inferred ingredient, ingredients contributed this batch, grace counter
    return jnp.asarray([-1, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    remembered_kind, contributions, grace = memory
    pos = _position(obs)
    col = pos[1]
    facing = _facing(obs)
    inv = obs[2, 2, 5:11]
    has_plate = inv[0] > 0
    has_dish = inv[1] > 0
    held_counts = inv[2:6]
    has_raw = jnp.sum(held_counts) > 0
    held_kind = jnp.argmax(held_counts).astype(jnp.int32)

    pot_mask = obs[:, :, 24]
    pot_counts = jnp.sum(obs[:, :, 34:38] * pot_mask[:, :, None], axis=(0,1))
    pot_total = jnp.sum(pot_counts)
    pot_kind = jnp.argmax(pot_counts).astype(jnp.int32)
    pot_cooked = jnp.sum(obs[:, :, 33] * pot_mask) > 0
    pot_timer = jnp.sum(obs[:, :, 44] * pot_mask)
    partner_counts = jnp.sum(obs[:, :, 18:22], axis=(0,1))
    partner_has_raw = jnp.sum(partner_counts) > 0
    partner_kind = jnp.argmax(partner_counts).astype(jnp.int32)
    evidence_kind = jnp.where(pot_total > 0, pot_kind, partner_kind)
    has_evidence = (pot_total > 0) | partner_has_raw
    inferred = jnp.where(has_evidence, evidence_kind, remembered_kind)

    at_center = col == 5
    deliver_action = jnp.where(
        at_center, _face_then_interact(facing, jnp.int32(0), UP),
        _upper_move_to_column(col, jnp.int32(5)))
    plate_at_pot_action = jnp.where(
        at_center,
        jnp.where(pot_cooked, _face_then_interact(facing, jnp.int32(1), DOWN), STAY),
        _upper_move_to_column(col, jnp.int32(5)))
    raw_at_pot_action = jnp.where(
        at_center, _face_then_interact(facing, jnp.int32(1), DOWN),
        _upper_move_to_column(col, jnp.int32(5)))
    discard_col = jnp.where(held_kind == 0, jnp.int32(4), jnp.int32(6))
    discard_raw_action = jnp.where(
        col == discard_col, _face_then_interact(facing, jnp.int32(1), DOWN),
        _upper_move_to_column(col, discard_col))

    ingredient_col = jnp.where(inferred == 0, jnp.int32(4), jnp.int32(6))
    ingredient_face = jnp.where(inferred == 0, jnp.int32(3), jnp.int32(2))
    ingredient_face_action = jnp.where(inferred == 0, LEFT, RIGHT)
    fetch_ingredient = jnp.where(
        col == ingredient_col,
        _face_then_interact(facing, ingredient_face, ingredient_face_action),
        _upper_move_to_column(col, ingredient_col))
    plate_col = jnp.where(col <= 5, jnp.int32(4), jnp.int32(6))
    fetch_plate = jnp.where(
        col == plate_col, _face_then_interact(facing, jnp.int32(0), UP),
        _upper_move_to_column(col, plate_col))
    wait_center = _upper_move_to_column(col, jnp.int32(5))

    pot_active = (pot_total >= 3) | (pot_timer > 0) | pot_cooked
    batch_empty = (pot_total == 0) & (~has_raw) & (~has_plate) & (~has_dish)
    contributions = jnp.where(batch_empty, jnp.int32(0), contributions)
    waiting_partial = (pot_total > 0) & (pot_total < 3) & (contributions > 0)
    grace = jnp.where(waiting_partial & (~has_raw), grace + 1,
                      jnp.where(batch_empty, jnp.int32(0), grace))
    partner_mask = obs[:, :, 11]
    local_columns = jnp.arange(5, dtype=jnp.int32)[None, :]
    partner_global_col = jnp.sum(partner_mask * (col + local_columns - 2))
    partner_at_center = (jnp.sum(partner_mask) > 0) & (partner_global_col == 5)
    partner_empty = jnp.sum(obs[:, :, 16:22]) == 0
    signal_only_wait = partner_at_center & partner_empty
    courtesy_wait = waiting_partial & (grace < 20) & (~signal_only_wait)
    empty_action = jnp.where(
        pot_active, fetch_plate,
        jnp.where(courtesy_wait, wait_center,
                  jnp.where(has_evidence & (evidence_kind < 2), fetch_ingredient, wait_center)))
    action = jnp.where(
        has_dish, deliver_action,
        jnp.where(has_plate, plate_at_pot_action,
                  jnp.where(has_raw,
                            jnp.where(pot_active, discard_raw_action, raw_at_pot_action),
                            empty_action))).astype(jnp.int32)
    issuing_raw = has_raw & at_center & (facing == 1) & (~pot_active)
    contributions = jnp.where(issuing_raw, contributions + 1, contributions)
    next_memory = jnp.stack((inferred, contributions, grace)).astype(jnp.int32)
    return action, next_memory, {
        'position': pos,
        'inferred_ingredient': inferred,
        'contributions': contributions,
        'grace': grace,
        'pot_total': pot_total,
        'pot_timer': pot_timer,
    }


def init_agent_1():
    # target, phase, side, wait, saw cooking, saw dish, expected pot count
    return jnp.asarray([-1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def _lower_to_center(row, col):
    return jnp.where(row > 3, UP,
                     jnp.where(col < 5, RIGHT,
                               jnp.where(col > 5, LEFT, STAY)))


def _lower_to_station(row, col, station_col):
    on_row3 = row == 3
    row3_action = jnp.where(col < station_col, RIGHT,
                            jnp.where(col > station_col, LEFT, DOWN))
    row4_action = jnp.where(col < station_col, RIGHT,
                            jnp.where(col > station_col, LEFT, STAY))
    return jnp.where(on_row3, row3_action, row4_action)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    target, phase, side, wait_count, saw_cooking, saw_dish, expected_total = memory
    pos = _position(obs)
    row, col = pos[0], pos[1]
    facing = _facing(obs)
    inv_counts = obs[2, 2, 7:11]
    has_raw = jnp.sum(inv_counts) > 0
    pot_mask = obs[:, :, 24]
    pot_counts = jnp.sum(obs[:, :, 34:38] * pot_mask[:, :, None], axis=(0,1))
    pot_total = jnp.sum(pot_counts)
    pot_cooked = jnp.sum(obs[:, :, 33] * pot_mask) > 0
    pot_timer = jnp.sum(obs[:, :, 44] * pot_mask)
    pot_visible = jnp.sum(pot_mask) > 0
    recipe_counts = jnp.sum(obs[:, :, 40:44], axis=(0,1))
    recipe_visible = jnp.sum(recipe_counts) > 0
    shown_target = jnp.argmax(recipe_counts).astype(jnp.int32)
    target = jnp.where(recipe_visible, shown_target, target)
    partner_dish = jnp.sum(obs[:, :, 17]) > 0
    saw_dish = jnp.where(partner_dish, jnp.int32(1), saw_dish)
    cooking_now = pot_visible & ((pot_timer > 0) | pot_cooked | (pot_total >= 3))
    saw_cooking = jnp.where(cooking_now, jnp.int32(1), saw_cooking)

    phase = jnp.where((phase == 0) & (target >= 0), jnp.int32(1), phase)
    delivered = (phase == 3) & (saw_dish > 0) & (~partner_dish)
    phase = jnp.where(delivered, jnp.int32(1), phase)
    wait_count = jnp.where(delivered, jnp.int32(0), wait_count)
    saw_cooking = jnp.where(delivered, jnp.int32(0), saw_cooking)
    saw_dish = jnp.where(delivered, jnp.int32(0), saw_dish)
    expected_total = jnp.where(delivered, jnp.int32(0), expected_total)

    # Mirror the upper room's semantic arrangement: ingredient 0 is signaled
    # via the left supply and ingredient 1 via the right supply.
    new_side = jnp.where(target == 0, jnp.int32(0), jnp.int32(1))
    side = jnp.where((phase == 1) & (~has_raw) & (wait_count == 0), new_side, side)
    station_col = jnp.where(side == 0, jnp.int32(1), jnp.int32(9))
    at_station = (row == 4) & (col == station_col)
    wanted_face = jnp.where(target == 0,
                            jnp.where(side == 0, jnp.int32(3), jnp.int32(2)),
                            jnp.int32(1))
    face_action = jnp.where(target == 0, jnp.where(side == 0, LEFT, RIGHT), DOWN)
    fetch_action = jnp.where(
        at_station, _face_then_interact(facing, wanted_face, face_action),
        _lower_to_station(row, col, station_col))
    carry_action = jnp.where(
        (row == 3) & (col == 5), _face_then_interact(facing, jnp.int32(0), UP),
        _lower_to_center(row, col))

    issuing_load = (phase == 1) & has_raw & (row == 3) & (col == 5) & (facing == 0)
    expected_total = jnp.where(issuing_load, jnp.minimum(pot_total + 1, 3), expected_total)
    wait_count = jnp.where(issuing_load, jnp.int32(0), wait_count)
    phase2 = phase == 2
    wait_count = jnp.where(phase2, wait_count + 1, wait_count)
    # Wait at a side checkpoint that still sees the pot.  Our active upper
    # fills by count 7; only a passive/slow partner triggers the remote fetch.
    checkpoint_col = jnp.where(side == 0, jnp.int32(3), jnp.int32(7))
    at_checkpoint = (row == 3) & (col == checkpoint_col)
    no_partner_progress = pot_total <= expected_total
    partner_mask = obs[:, :, 11]
    local_columns = jnp.arange(5, dtype=jnp.int32)[None, :]
    partner_global_col = jnp.sum(partner_mask * (col + local_columns - 2))
    partner_at_center = (jnp.sum(partner_mask) > 0) & (partner_global_col == 5)
    partner_empty = jnp.sum(obs[:, :, 16:22]) == 0
    partner_waiting = partner_at_center & partner_empty
    retry_fetch = phase2 & (pot_total < 3) & (
        (at_checkpoint & (no_partner_progress | partner_waiting)) | (wait_count >= 7))
    action_phase = jnp.where(retry_fetch, jnp.int32(1), phase)
    action_phase = jnp.where(cooking_now, jnp.int32(3), action_phase)
    next_phase = jnp.where(issuing_load, jnp.int32(2), phase)
    next_phase = jnp.where(retry_fetch, jnp.int32(1), next_phase)
    next_phase = jnp.where(cooking_now, jnp.int32(3), next_phase)
    wait_count = jnp.where((next_phase == 1) & (~has_raw), jnp.int32(0), wait_count)

    discard_col = jnp.where(col <= 5, jnp.int32(3), jnp.int32(7))
    discard_action = jnp.where(
        (row == 3) & (col == discard_col),
        _face_then_interact(facing, jnp.int32(1), DOWN),
        jnp.where(row > 3, UP,
                  jnp.where(col < discard_col, RIGHT,
                            jnp.where(col > discard_col, LEFT, STAY))))
    idle_action = _lower_to_center(row, col)
    checkpoint_action = jnp.where(
        row > 3, UP,
        jnp.where(col < checkpoint_col, RIGHT,
                  jnp.where(col > checkpoint_col, LEFT, STAY)))
    phase1_action = jnp.where(has_raw, carry_action, fetch_action)
    phase3_action = jnp.where(has_raw, discard_action, idle_action)
    action = jnp.where(action_phase == 0, idle_action,
                       jnp.where(action_phase == 1, phase1_action,
                                 jnp.where(action_phase == 2, checkpoint_action,
                                           jnp.where(action_phase == 3, phase3_action, idle_action)))).astype(jnp.int32)
    next_memory = jnp.stack((target, next_phase, side, wait_count, saw_cooking,
                             saw_dish, expected_total)).astype(jnp.int32)
    return action, next_memory, {
        'position': pos,
        'target': target,
        'phase': next_phase,
        'pot_total': pot_total,
        'pot_timer': pot_timer,
    }
