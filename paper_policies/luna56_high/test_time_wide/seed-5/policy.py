"""A geometry-driven cooperative policy for the two-room Overcooked task."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'boundary_signal_single_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}


# Channels 22:32, including the ingredient supply channels, are static.  A
# local crop can therefore be matched against the known board to recover the
# player's global floor coordinate without using hidden environment state.
_STATIC = [
    [[1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0]],
    [[0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,]],
    [[1,0,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,1]],
    [[1,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]],
    [[1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]],
]
_FLOORS = [(r, c) for r in (1, 2, 4, 5) for c in (1, 2, 3, 4)]


def _crop_for(row, col):
    crop = []
    for rr in range(row - 2, row + 3):
        line = []
        for cc in range(col - 2, col + 3):
            if 0 <= rr < 7 and 0 <= cc < 6:
                line.append(_STATIC[rr][cc])
            else:
                line.append([0] * 10)
        crop.append(line)
    return crop


_COORDS = _FLOORS


def _position(obs):
    coords = jnp.asarray(_COORDS, dtype=jnp.int32)
    patches = jnp.asarray([_crop_for(r, c) for r, c in _FLOORS], dtype=jnp.int32)
    visible_static = obs[:, :, 22:32]
    error = jnp.sum(jnp.abs(patches - visible_static[None, :, :, :]), axis=(1, 2, 3))
    return coords[jnp.argmin(error)]


def _at(pos, row, col):
    return (pos[0] == row) & (pos[1] == col)


def _route(pos, row, col):
    return jnp.where(
        pos[0] < row, 1,
        jnp.where(pos[0] > row, 3,
                  jnp.where(pos[1] < col, 0,
                            jnp.where(pos[1] > col, 2, 4))))


def _face(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _interact_at(obs, pos, row, col, facing, turn_action):
    there = _at(pos, row, col)
    return jnp.where(there, jnp.where(_face(obs) == facing, 5, turn_action), _route(pos, row, col))


def init_agent_0():
    # row, column, phase, number already placed, recipe
    return jnp.zeros((5,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(jnp.asarray(reset), jnp.zeros((5,), dtype=jnp.int32), memory)
    pos = _position(obs)
    phase = memory[2]
    placed = memory[3]
    recipe = memory[4]

    # The bottom policy signals at global (4,1), visible from top (2,1).
    at_communication = _at(pos, 2, 1)
    partner_at_boundary = obs[4, 2, 11] > 0
    partner_up = obs[4, 2, 12] > 0
    partner_left = obs[4, 2, 15] > 0
    signal = at_communication & partner_at_boundary & (partner_up | partner_left)
    signaled_recipe = jnp.where(partner_left, 1, 0)
    recipe = jnp.where(signal, signaled_recipe, recipe)

    phase_for_action = jnp.where((phase == 0) & signal, 1, phase)
    pot_ready = jnp.max(obs[:, :, 33]) > 0
    phase_for_action = jnp.where((phase_for_action == 3) & pot_ready, 4, phase_for_action)

    supply_row = 1 + recipe
    action_0 = _route(pos, 2, 1)
    action_1 = _interact_at(obs, pos, supply_row, 1, 3, 2)  # face left
    action_2 = _interact_at(obs, pos, 2, 1, 1, 1)           # face down
    action_3 = _route(pos, 2, 1)
    action_4 = _interact_at(obs, pos, 1, 3, 0, 3)           # face up
    action_5 = _interact_at(obs, pos, 2, 1, 1, 1)           # face down
    action_6 = _interact_at(obs, pos, 1, 2, 0, 3)           # face up
    action = action_0
    action = jnp.where(phase_for_action == 1, action_1, action)
    action = jnp.where(phase_for_action == 2, action_2, action)
    action = jnp.where(phase_for_action == 3, action_3, action)
    action = jnp.where(phase_for_action == 4, action_4, action)
    action = jnp.where(phase_for_action == 5, action_5, action)
    action = jnp.where(phase_for_action == 6, action_6, action)

    at_supply_facing = _at(pos, supply_row, 1) & (_face(obs) == 3)
    supply_done = (phase == 1) & at_supply_facing
    at_pot_facing = _at(pos, 2, 1) & (_face(obs) == 1)
    pot_done = (phase == 2) & at_pot_facing
    plate_done = (phase == 4) & _at(pos, 1, 3) & (_face(obs) == 0)
    pickup_done = (phase == 5) & at_pot_facing
    delivery_done = (phase == 6) & _at(pos, 1, 2) & (_face(obs) == 0)

    next_phase = phase
    next_phase = jnp.where((phase == 0) & signal, 1, next_phase)
    next_phase = jnp.where(supply_done, 2, next_phase)
    next_phase = jnp.where(pot_done, jnp.where(placed == 2, 3, 1), next_phase)
    next_phase = jnp.where((phase == 3) & pot_ready, 4, next_phase)
    next_phase = jnp.where(plate_done, 5, next_phase)
    next_phase = jnp.where(pickup_done, 6, next_phase)
    next_phase = jnp.where(delivery_done, 0, next_phase)
    next_placed = jnp.where(pot_done, placed + 1, placed)
    next_placed = jnp.where(delivery_done, 0, next_placed)
    next_memory = jnp.stack((pos[0], pos[1], next_phase, next_placed, recipe)).astype(jnp.int32)
    return action.astype(jnp.int32), next_memory, {}


def init_agent_1():
    # row, column, last observed recipe
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(jnp.asarray(reset), jnp.zeros((3,), dtype=jnp.int32), memory)
    pos = _position(obs)
    displayed_0 = jnp.sum(obs[:, :, 40])
    displayed_1 = jnp.sum(obs[:, :, 41])
    displayed = (displayed_0 + displayed_1) > 0
    observed_recipe = jnp.where(displayed_1 > displayed_0, 1, 0)
    recipe = jnp.where(displayed, observed_recipe, memory[2])

    # At (4,1), UP is the 000 signal and LEFT is the 111 signal.  Both are
    # blocked by static objects, so turning never displaces the signaler.
    desired_face = jnp.where(recipe == 0, 0, 3)
    desired_action = jnp.where(recipe == 0, 3, 2)
    action = jnp.where(
        _at(pos, 4, 1),
        jnp.where(_face(obs) == desired_face, 4, desired_action),
        _route(pos, 4, 1),
    )
    next_memory = jnp.stack((pos[0], pos[1], recipe)).astype(jnp.int32)
    return action.astype(jnp.int32), next_memory, {}
