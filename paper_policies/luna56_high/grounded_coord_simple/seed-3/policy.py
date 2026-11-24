"""Button-mediated, observable controller for the grounded layout."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'button_mediated_cycle',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


# Literal constants keep module scope free of executable computation.  Each
# static cell is [wall, goal, pot, recipe, button, plate].
_STATIC = (
    ((1,0,0,0,0,0), (1,0,0,0,0,0), (0,0,0,0,0,0), (1,0,0,0,0,0),
     (1,0,0,0,0,0), (1,0,0,0,0,0), (1,0,0,0,0,0), (1,0,0,0,0,0)),
    ((1,0,0,0,0,0), (0,0,0,0,0,0), (0,0,0,0,0,0), (1,0,0,0,0,0),
     (0,0,0,0,0,1), (0,0,0,0,0,0), (0,0,0,0,0,0), (0,0,0,0,0,0)),
    ((0,0,0,1,0,0), (0,0,0,0,0,0), (0,0,0,0,0,0), (0,0,0,0,1,0),
     (0,0,1,0,0,0), (0,0,0,0,0,0), (0,0,0,0,0,0), (0,1,0,0,0,0)),
    ((1,0,0,0,0,0), (0,0,0,0,0,0), (0,0,0,0,0,0), (1,0,0,0,0,0),
     (0,0,0,0,0,1), (0,0,0,0,0,0), (0,0,0,0,0,0), (0,0,0,0,0,0)),
    ((1,0,0,0,0,0), (1,0,0,0,0,0), (0,0,0,0,0,0), (1,0,0,0,0,0),
     (1,0,0,0,0,0), (1,0,0,0,0,0), (1,0,0,0,0,0), (1,0,0,0,0,0)),
)

_FLOOR_R = (1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3)
_FLOOR_C = (1, 2, 5, 6, 1, 2, 3, 5, 6, 1, 2, 5, 6)
# targets: button stand, ingredient-0 stand, ingredient-1 stand, pot stand,
# top plate stand, delivery stand
_TARGET_R = (2, 1, 3, 2, 1, 2)
_TARGET_C = (2, 6, 6, 5, 5, 6)
_ROUTES = (
    (0, 1, 4, 4, 0, 4, 2, 4, 4, 0, 3, 4, 4),
    (4, 4, 0, 4, 4, 4, 4, 0, 3, 4, 4, 0, 3),
    (4, 4, 0, 1, 4, 4, 4, 0, 1, 4, 4, 0, 4),
    (4, 4, 1, 2, 4, 4, 4, 4, 2, 4, 4, 3, 2),
    (4, 4, 4, 2, 4, 4, 4, 3, 2, 4, 4, 3, 2),
    (4, 4, 0, 1, 4, 4, 4, 0, 4, 4, 4, 0, 3),
)


def _route(target, floor_index):
    return jnp.asarray(_ROUTES, dtype=jnp.int32)[target, floor_index]


def _position(obs):
    board = jnp.asarray(_STATIC, dtype=jnp.int32)
    padded = jnp.pad(board, ((2, 2), (2, 2), (0, 0)))
    offsets = jnp.arange(5, dtype=jnp.int32)
    rows = jnp.asarray(_FLOOR_R, dtype=jnp.int32)
    cols = jnp.asarray(_FLOOR_C, dtype=jnp.int32)
    patches = padded[
        rows[:, None, None] + offsets[None, :, None],
        cols[:, None, None] + offsets[None, None, :],
    ]
    observed = obs[:, :, 20:26]
    agreement = jnp.sum(
        (observed[None, :, :, :] == patches).astype(jnp.int32),
        axis=(1, 2, 3),
    )
    index = jnp.argmax(agreement)
    return rows[index], cols[index], index


def _cell(obs, row, col, wanted_row, wanted_col, start, width):
    local_row = wanted_row - row + 2
    local_col = wanted_col - col + 2
    valid = ((local_row >= 0) & (local_row < 5) &
             (local_col >= 0) & (local_col < 5))
    safe_row = jnp.clip(local_row, 0, 4)
    safe_col = jnp.clip(local_col, 0, 4)
    value = obs[safe_row, safe_col, start:start + width]
    return jnp.where(valid, value, jnp.zeros((width,), dtype=obs.dtype))


def _at(row, col, target):
    rows = jnp.asarray(_TARGET_R, dtype=jnp.int32)
    cols = jnp.asarray(_TARGET_C, dtype=jnp.int32)
    return (row == rows[target]) & (col == cols[target])


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5])


def _face_or_interact(obs, desired_direction):
    facing = _facing(obs)
    movement = jnp.asarray((3, 1, 0, 2), dtype=jnp.int32)[desired_direction]
    return jnp.where(facing == desired_direction, jnp.asarray(5, dtype=jnp.int32),
                     movement)


def _pot(obs, row, col):
    return _cell(obs, row, col, 2, 4, 29, 5)


def _active_recipe(obs):
    zero = jnp.max(obs[:, :, 36])
    one = jnp.max(obs[:, :, 37])
    return (zero + one) > 0, jnp.where(one > 0, jnp.asarray(1, dtype=jnp.int32),
                                      jnp.asarray(0, dtype=jnp.int32))


def init_agent_0():
    # 0=activate, 1=wait for pot use, 2=wait for delivery
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    state = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    row, col, floor_index = _position(obs)
    at_button = _at(row, col, 0)
    activation_action = jnp.where(at_button, _face_or_interact(obs, 2),
                                  _route(0, floor_index))
    action = jnp.where(state == 0, activation_action,
                       jnp.asarray(4, dtype=jnp.int32))
    facing = _facing(obs)
    activated = (state == 0) & at_button & (facing == 2)
    pot_contents = _pot(obs, row, col)
    pot_used = (jnp.sum(pot_contents[2:5]) > 0) | (pot_contents[1] > 0)
    pot_empty = (jnp.sum(pot_contents[1:5]) == 0)
    next_state = state
    next_state = jnp.where(activated, 1, next_state)
    next_state = jnp.where((state == 1) & pot_used, 2, next_state)
    next_state = jnp.where((state == 2) & pot_empty, 0, next_state)
    trace = {'state': state, 'row': row, 'col': col, 'facing': _facing(obs)}
    return action, next_state, trace


def init_agent_1():
    # state, selected ingredient, number of raw units already deposited
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    initial = jnp.zeros((3,), dtype=jnp.int32)
    mem = jnp.where(reset, initial, memory)
    state, target, deposited = mem[0], mem[1], mem[2]
    row, col, floor_index = _position(obs)
    at_pot = _at(row, col, 3)
    pot_contents = _pot(obs, row, col)
    pot_total = jnp.sum(pot_contents[2:5])
    display_visible, display_target = _active_recipe(obs)
    supply_target = jnp.where(target == 0, jnp.asarray(1, dtype=jnp.int32),
                              jnp.asarray(2, dtype=jnp.int32))

    wait_action = _route(3, floor_index)
    get_action = jnp.where(
        _at(row, col, supply_target), _face_or_interact(obs, 2),
        _route(supply_target, floor_index))
    put_action = jnp.where(at_pot, _face_or_interact(obs, 3),
                           _route(3, floor_index))
    at_plate = _at(row, col, 4)
    plate_action = jnp.where(at_plate, _face_or_interact(obs, 3),
                             _route(4, floor_index))
    cooked = pot_contents[1] > 0
    holding_plate = obs[2, 2, 5] > 0
    collect_action = jnp.where(
        at_pot,
        jnp.where(cooked & holding_plate, _face_or_interact(obs, 3),
                  jnp.asarray(4, dtype=jnp.int32)),
        _route(3, floor_index))
    at_goal = _at(row, col, 5)
    deliver_action = jnp.where(at_goal, _face_or_interact(obs, 2),
                               _route(5, floor_index))
    action = jnp.where(state == 0, wait_action,
                       jnp.where(state == 1, get_action,
                                 jnp.where(state == 2, put_action,
                                           jnp.where(state == 3, plate_action,
                                                     jnp.where(state == 4,
                                                               collect_action,
                                                               deliver_action)))))

    facing = _facing(obs)
    start_cycle = ((state == 0) & at_pot & (pot_total == 0) &
                   display_visible)
    got_ingredient = ((state == 1) & _at(row, col, supply_target) &
                      (facing == 2))
    put_ingredient = (state == 2) & at_pot & (facing == 3)
    got_plate = (state == 3) & at_plate & (facing == 3)
    collected = ((state == 4) & at_pot & cooked & holding_plate &
                 (facing == 3))
    delivered = (state == 5) & at_goal & (facing == 2)

    next_state = state
    next_target = jnp.where(start_cycle, display_target, target)
    next_deposited = deposited
    next_state = jnp.where(start_cycle, 1, next_state)
    next_state = jnp.where(got_ingredient, 2, next_state)
    next_state = jnp.where(put_ingredient & (deposited < 2), 1, next_state)
    next_state = jnp.where(put_ingredient & (deposited == 2), 3, next_state)
    next_deposited = jnp.where(put_ingredient, deposited + 1, next_deposited)
    next_state = jnp.where(got_plate, 4, next_state)
    next_state = jnp.where(collected, 5, next_state)
    next_state = jnp.where(delivered, 0, next_state)
    next_deposited = jnp.where(delivered, 0, next_deposited)
    trace = {'state': state, 'row': row, 'col': col, 'facing': facing,
             'target': target, 'deposited': deposited,
             'pot_total': pot_total, 'display': display_visible.astype(jnp.int32)}
    return action, jnp.stack((next_state, next_target, next_deposited)), trace
