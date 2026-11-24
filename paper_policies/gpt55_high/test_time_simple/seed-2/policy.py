"""Counter-marker Overcooked policy for the test_time_simple layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'counter_marker_candidate',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

A_RIGHT = 0
A_DOWN = 1
A_LEFT = 2
A_UP = 3
A_STAY = 4
A_INTERACT = 5

F_UP = 0
F_DOWN = 1
F_RIGHT = 2
F_LEFT = 3

PHASE_CHECK = 0
PHASE_WORK = 1
UNKNOWN = -1

LEFT_POSITIONS = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
RIGHT_POSITIONS = ((1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6))


def _static_map():
    grid = jnp.zeros((5, 8, 9), dtype=jnp.int32)
    walls = (
        (0, 0), (0, 1), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 0), (1, 3), (2, 3), (3, 0), (3, 3),
        (4, 0), (4, 1), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
    )
    for r, c in walls:
        grid = grid.at[r, c, 0].set(1)
    grid = grid.at[2, 7, 1].set(1)
    grid = grid.at[2, 4, 2].set(1)
    grid = grid.at[2, 0, 3].set(1)
    grid = grid.at[1, 4, 5].set(1)
    grid = grid.at[3, 4, 5].set(1)
    grid = grid.at[1, 7, 6].set(1)
    grid = grid.at[3, 7, 7].set(1)
    grid = grid.at[0, 2, 8].set(1)
    grid = grid.at[4, 2, 8].set(1)
    return grid


def _crops(position_data):
    padded_static = jnp.pad(_static_map(), ((2, 2), (2, 2), (0, 0)))
    crops = []
    for r, c in position_data:
        crops.append(padded_static[r:r + 5, c:c + 5, :])
    return jnp.stack(crops, axis=0)


def _init0():
    return jnp.asarray(0, dtype=jnp.int32)


def _init1():
    return jnp.asarray([0, -1], dtype=jnp.int32)


def _infer_pos(obs, position_data):
    positions = jnp.asarray(position_data, dtype=jnp.int32)
    crops = _crops(position_data)
    static_obs = obs[:, :, 20:29]
    scores = jnp.sum(static_obs[None, :, :, :] == crops, axis=(1, 2, 3))
    return positions[jnp.argmax(scores)]


def _face(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _at(pos, row, col):
    return (pos[0] == row) & (pos[1] == col)


def _cell_value(obs, pos, row, col, channel):
    lr = row - pos[0] + 2
    lc = col - pos[1] + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    lr = jnp.clip(lr, 0, 4)
    lc = jnp.clip(lc, 0, 4)
    return jnp.where(visible, obs[lr, lc, channel], jnp.asarray(0, dtype=obs.dtype))


def _held_any(obs):
    return jnp.sum(obs[2, 2, 5:10]) > 0


def _move_to(pos, row, col):
    action = jnp.where(pos[0] < row, A_DOWN, A_STAY)
    action = jnp.where(pos[0] > row, A_UP, action)
    action = jnp.where((pos[0] == row) & (pos[1] < col), A_RIGHT, action)
    action = jnp.where((pos[0] == row) & (pos[1] > col), A_LEFT, action)
    return action


def _face_action(face_id):
    action = jnp.where(face_id == F_UP, A_UP, A_RIGHT)
    action = jnp.where(face_id == F_DOWN, A_DOWN, action)
    action = jnp.where(face_id == F_LEFT, A_LEFT, action)
    return action


def _go_face(pos, face, row, col, face_id):
    move = _move_to(pos, row, col)
    turn = _face_action(face_id)
    return jnp.where(_at(pos, row, col), jnp.where(face == face_id, A_STAY, turn), move)


def _go_interact(pos, face, row, col, face_id):
    move = _move_to(pos, row, col)
    turn_or_use = jnp.where(face == face_id, A_INTERACT, _face_action(face_id))
    return jnp.where(_at(pos, row, col), turn_or_use, move)


def init_agent_0():
    return _init0()


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset, _init0(), memory)
    pos = _infer_pos(obs, LEFT_POSITIONS)
    face = _face(obs)

    recipe0 = jnp.sum(obs[:, :, 36]) > 0
    wants_top = recipe0

    top_marker = _cell_value(obs, pos, 1, 3, 33) > 0
    bottom_marker = _cell_value(obs, pos, 3, 3, 33) > 0
    correct_marker = jnp.where(wants_top, top_marker, bottom_marker)
    wrong_marker = jnp.where(wants_top, bottom_marker, top_marker)

    held2 = obs[2, 2, 9] > 0
    held_any = _held_any(obs)

    target_row = jnp.where(wants_top, 1, 3)
    wrong_row = jnp.where(wants_top, 3, 1)
    supply_face = jnp.where(wants_top, F_UP, F_DOWN)

    deposit_action = _go_interact(pos, face, target_row, 2, F_RIGHT)
    collect_wrong_action = _go_interact(pos, face, wrong_row, 2, F_RIGHT)
    fetch_action = _go_interact(pos, face, target_row, 2, supply_face)

    action = jnp.where(held2, deposit_action, A_STAY)
    action = jnp.where((~held_any) & wrong_marker, collect_wrong_action, action)
    action = jnp.where((~held_any) & (~wrong_marker) & (~correct_marker), fetch_action, action)
    action = jnp.where((~held_any) & correct_marker, A_STAY, action)
    return action.astype(jnp.int32), memory, {}


def init_agent_1():
    return _init1()


def _agent1_work_action(obs, pos, face, desired):
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ing0 = obs[2, 2, 7] > 0
    held_ing1 = obs[2, 2, 8] > 0
    wants0 = desired == 0
    held_target_raw = jnp.where(wants0, held_ing0, held_ing1)

    pot_cooked = _cell_value(obs, pos, 2, 4, 30) > 0
    pot_ing0 = _cell_value(obs, pos, 2, 4, 31)
    pot_ing1 = _cell_value(obs, pos, 2, 4, 32)
    pot_ing2 = _cell_value(obs, pos, 2, 4, 33)
    pot_timer = _cell_value(obs, pos, 2, 4, 39)
    pot_total = pot_ing0 + pot_ing1 + pot_ing2
    pot_busy = (pot_timer > 0) | pot_cooked | (pot_total >= 3)

    supply_row = jnp.where(wants0, 1, 3)
    plate_row = supply_row

    get_ingredient = _go_interact(pos, face, supply_row, 6, F_RIGHT)
    add_to_pot = _go_interact(pos, face, 2, 5, F_LEFT)
    get_plate = _go_interact(pos, face, plate_row, 5, F_LEFT)
    scoop = _go_interact(pos, face, 2, 5, F_LEFT)
    wait_pot = _go_face(pos, face, 2, 5, F_LEFT)
    deliver = _go_interact(pos, face, 2, 6, F_RIGHT)
    check = _move_to(pos, 2, 5)

    action = get_ingredient
    action = jnp.where(held_target_raw, add_to_pot, action)
    action = jnp.where((~held_target_raw) & pot_busy, get_plate, action)
    action = jnp.where(held_plate & (~pot_cooked), wait_pot, action)
    action = jnp.where(held_plate & pot_cooked, scoop, action)
    action = jnp.where(held_cooked, deliver, action)
    action = jnp.where(desired < 0, check, action)
    return action


def agent_1_act(obs, reset, memory, rng):
    base = _init1()
    memory = jnp.where(reset, base, memory)
    phase = memory[0]
    desired = memory[1]
    pos = _infer_pos(obs, RIGHT_POSITIONS)
    face = _face(obs)

    top_marker = _cell_value(obs, pos, 1, 3, 33) > 0
    bottom_marker = _cell_value(obs, pos, 3, 3, 33) > 0
    signal_visible = top_marker | bottom_marker
    signal_desired = jnp.where(top_marker, 0, 1)
    desired_now = jnp.where(signal_visible, signal_desired, desired)

    at_check = _at(pos, 2, 5)
    check_action = jnp.where(at_check, A_STAY, _move_to(pos, 2, 5))
    work_action = _agent1_work_action(obs, pos, face, desired_now)
    ready_to_work = at_check & signal_visible
    action = jnp.where(phase == PHASE_CHECK, jnp.where(ready_to_work, work_action, check_action), work_action)

    delivering_now = (obs[2, 2, 6] > 0) & (action == A_INTERACT) & _at(pos, 2, 6) & (face == F_RIGHT)
    next_phase = jnp.where((phase == PHASE_CHECK) & ready_to_work, PHASE_WORK, phase)
    next_phase = jnp.where(delivering_now, PHASE_CHECK, next_phase)
    next_desired = jnp.where((phase == PHASE_CHECK) & ready_to_work, desired_now, desired)
    next_desired = jnp.where(delivering_now, UNKNOWN, next_desired)
    next_memory = jnp.stack([next_phase, next_desired]).astype(jnp.int32)
    next_memory = jnp.where(reset, base, next_memory)
    return action.astype(jnp.int32), next_memory, {}
