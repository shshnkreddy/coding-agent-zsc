"""Geometry-grounded cooperative controller for grounded_coord_simple."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'reactive_communicator_producer',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


# Feature order is observation channels 20:29:
# wall, goal, pot, recipe indicator, button, plate pile, ingredients 0..2.
_W = (1, 0, 0, 0, 0, 0, 0, 0, 0)
_G = (0, 1, 0, 0, 0, 0, 0, 0, 0)
_P = (0, 0, 1, 0, 0, 0, 0, 0, 0)
_R = (0, 0, 0, 1, 0, 0, 0, 0, 0)
_L = (0, 0, 0, 0, 1, 0, 0, 0, 0)
_B = (0, 0, 0, 0, 0, 1, 0, 0, 0)
_I0 = (0, 0, 0, 0, 0, 0, 1, 0, 0)
_I1 = (0, 0, 0, 0, 0, 0, 0, 1, 0)
_I2 = (0, 0, 0, 0, 0, 0, 0, 0, 1)
_F = (0, 0, 0, 0, 0, 0, 0, 0, 0)

_STATIC_MAP = (
    [_W, _W, _I2, _W, _W, _W, _W, _W],
    [_W, _F, _F, _W, _B, _F, _F, _I0],
    [_R, _F, _F, _L, _P, _F, _F, _G],
    [_W, _F, _F, _W, _B, _F, _F, _I1],
    [_W, _W, _I2, _W, _W, _W, _W, _W],
)
_MOVE_FOR_FACE = (3, 1, 0, 2)  # up, down, right, left


def _position(obs):
    """Recover the global floor coordinate from the local static crop."""
    static_map = jnp.asarray(_STATIC_MAP, dtype=jnp.int32)
    padded_map = jnp.pad(static_map, ((2, 2), (2, 2), (0, 0)))
    local_maps = jnp.stack([
        padded_map[r:r + 5, c:c + 5]
        for r in range(5) for c in range(8)
    ]).reshape(5, 8, 5, 5, 9)
    actual = obs[:, :, 20:29]
    error = jnp.sum(jnp.abs(local_maps - actual[None, :, :, :]), axis=(2, 3, 4))
    flat = jnp.argmin(error.reshape(-1))
    return flat // 8, flat % 8


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _navigate(row, col, target_row, target_col):
    vertical = jnp.where(row > target_row, jnp.asarray(3, jnp.int32), jnp.asarray(1, jnp.int32))
    horizontal = jnp.where(col > target_col, jnp.asarray(2, jnp.int32), jnp.asarray(0, jnp.int32))
    return jnp.where(row != target_row, vertical,
                     jnp.where(col != target_col, horizontal, jnp.asarray(4, jnp.int32)))


def _face_or_interact(obs, desired_facing):
    facing = _facing(obs)
    move_for_face = jnp.asarray(_MOVE_FOR_FACE, dtype=jnp.int32)
    return jnp.where(facing == desired_facing, jnp.asarray(5, jnp.int32),
                     move_for_face[desired_facing])


def _pot_cell(obs, row, col):
    local_row = jnp.clip(2 + 2 - row, 0, 4)
    local_col = jnp.clip(2 + 4 - col, 0, 4)
    return obs[local_row, local_col]


def init_agent_0():
    # stage, has_seen_contents, signaled_this_cycle, reserved
    return jnp.zeros((4,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    initial = jnp.zeros((4,), dtype=jnp.int32)
    m = jnp.where(reset, initial, memory)
    stage, seen, signaled = m[0], m[1], m[2]
    row, col = _position(obs)
    at_button_stand = (row == 2) & (col == 2)
    initial_action = _navigate(row, col, 2, 2)
    button_action = _face_or_interact(obs, 2)
    initial_done = at_button_stand & (_facing(obs) == 2)
    pot = _pot_cell(obs, row, col)
    pot_has_contents = jnp.sum(pot[29:34]) > 0
    pot_ready = pot[30] > 0
    pot_empty = jnp.sum(pot[29:34]) == 0
    monitor_signal = (seen == 1) & (signaled == 0) & pot_ready & at_button_stand
    monitor_action = jnp.where(monitor_signal, button_action, jnp.asarray(4, jnp.int32))
    action = jnp.where(stage == 0,
                       jnp.where(at_button_stand, button_action, initial_action),
                       monitor_action)

    next_stage = jnp.where((stage == 0) & initial_done, 1, stage)
    next_seen = jnp.where((stage == 1) & pot_has_contents, 1, seen)
    next_signaled = jnp.where(monitor_signal, 1, signaled)
    next_seen = jnp.where((stage == 1) & (signaled == 1) & pot_empty, 0, next_seen)
    next_signaled = jnp.where((stage == 1) & (signaled == 1) & pot_empty, 0, next_signaled)
    next_memory = jnp.asarray((next_stage, next_seen, next_signaled, 0), dtype=jnp.int32)
    return action, next_memory, {}


def init_agent_1():
    # phase, recipe (0 or 1), ingredient count already added (0..2)
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    initial = jnp.zeros((3,), dtype=jnp.int32)
    m = jnp.where(reset, initial, memory)
    phase, recipe, count = m[0], m[1], m[2]
    row, col = _position(obs)
    facing = _facing(obs)
    at_staging = (row == 2) & (col == 5)
    holding_ingredient = jnp.sum(obs[2, 2, 7:10]) > 0
    displayed_0 = jnp.sum(obs[:, :, 36]) > 0
    displayed_1 = jnp.sum(obs[:, :, 37]) > 0
    displayed = displayed_0 | displayed_1
    displayed_recipe = jnp.where(displayed_1, jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32))
    recipe_now = jnp.where(displayed, displayed_recipe, recipe)

    supply_stand_row = jnp.where(recipe_now == 0, 1, 3)
    plate_stand_row = jnp.where(recipe_now == 0, 1, 3)
    a0 = _navigate(row, col, 2, 5)
    a1 = _navigate(row, col, supply_stand_row, 6)
    a1 = jnp.where((row == supply_stand_row) & (col == 6),
                   _face_or_interact(obs, 2), a1)
    a2 = _navigate(row, col, 2, 5)
    a2 = jnp.where((row == 2) & (col == 5), _face_or_interact(obs, 3), a2)
    a3 = _navigate(row, col, plate_stand_row, 5)
    a3 = jnp.where((row == plate_stand_row) & (col == 5),
                   _face_or_interact(obs, 3), a3)
    a4 = _navigate(row, col, 2, 5)
    pot = _pot_cell(obs, row, col)
    pot_ready = pot[30] > 0
    a5 = _face_or_interact(obs, 3)
    a6 = _navigate(row, col, 2, 6)
    a6 = jnp.where((row == 2) & (col == 6), _face_or_interact(obs, 2), a6)
    a7 = jnp.asarray(4, dtype=jnp.int32)
    actions = jnp.asarray((a0, a1, a2, a3, a4, a5, a6, a7), dtype=jnp.int32)
    action = actions[jnp.clip(phase, 0, 7)]

    next_phase = phase
    next_count = count
    next_phase = jnp.where((phase == 0) & at_staging & displayed, 1, next_phase)
    next_phase = jnp.where((phase == 1) & holding_ingredient, 2, next_phase)
    at_pot_stand = (row == 2) & (col == 5)
    adding = (phase == 2) & at_pot_stand & holding_ingredient & (facing == 3)
    next_count = jnp.where(adding, count + 1, next_count)
    next_phase = jnp.where(adding & (count >= 2), 3, next_phase)
    next_phase = jnp.where(adding & (count < 2), 1, next_phase)
    at_plate_stand = (row == plate_stand_row) & (col == 5)
    taking_plate = (phase == 3) & at_plate_stand & (facing == 3)
    next_phase = jnp.where(taking_plate, 4, next_phase)
    next_phase = jnp.where((phase == 4) & at_pot_stand & pot_ready, 5, next_phase)
    taking_dish = (phase == 5) & at_pot_stand & (facing == 3)
    next_phase = jnp.where(taking_dish, 6, next_phase)
    at_delivery_stand = (row == 2) & (col == 6)
    delivering = (phase == 6) & at_delivery_stand & (facing == 2)
    next_phase = jnp.where(delivering, 0, next_phase)
    next_count = jnp.where(delivering, 0, next_count)
    next_memory = jnp.asarray((next_phase, recipe_now, next_count), dtype=jnp.int32)
    return action, next_memory, {}
