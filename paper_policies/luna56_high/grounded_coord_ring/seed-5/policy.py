"""A geometry-grounded, single-cook controller for grounded_coord_ring."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'geometry_cook_v1',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}


_GRID = (
    'WWW2R2WWW',
    'W       W',
    'W WWLWW W',
    '2 0   B 2',
    'RAXAP X R',
    '2 1   B 2',
    'W WWLWW W',
    'W       W',
    'WWW2R2WWW',
)


def _static_cell(ch):
    return (ch == 'W', ch == 'X', ch == 'P', ch == 'R', ch == 'L',
            ch == 'B', ch == '0', ch == '1', ch == '2')


_INIT = (0, 0, 0, 0)


def _position(obs):
    crops = jnp.asarray(
        [[[_static_cell(_GRID[rr + i - 2][cc + j - 2])
           if 0 <= rr + i - 2 < 9 and 0 <= cc + j - 2 < 9
           else (0, 0, 0, 0, 0, 0, 0, 0, 0)
           for j in range(5)] for i in range(5)]
         for rr in range(9) for cc in range(9)], dtype=jnp.int32)
    delta = jnp.sum(jnp.abs(crops - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    flat = jnp.argmin(delta)
    return flat // 9, flat - (flat // 9) * 9


def _move_to(row, col, target_row, target_col, goal_mode=False):
    if goal_mode:
        horizontal = jnp.where(col < target_col, jnp.asarray(0, jnp.int32),
                               jnp.asarray(2, jnp.int32))
        vertical = jnp.where(row < target_row, jnp.asarray(1, jnp.int32),
                             jnp.asarray(3, jnp.int32))
        return jnp.where(col != target_col, horizontal, vertical)
    vertical = jnp.where(row < target_row, jnp.asarray(1, jnp.int32),
                         jnp.asarray(3, jnp.int32))
    horizontal = jnp.where(col < target_col, jnp.asarray(0, jnp.int32),
                           jnp.asarray(2, jnp.int32))
    return jnp.where(row != target_row, vertical, horizontal)


def _at(obs, row, col, channel):
    self_row, self_col = _position(obs)
    rr = jnp.clip(row - self_row + 2, 0, 4)
    cc = jnp.clip(col - self_col + 2, 0, 4)
    return obs[rr, cc, channel]


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero = jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(4, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    # [phase, selected ingredient, loaded count, reserved]
    return jnp.asarray(_INIT, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    m = jnp.where(reset, jnp.asarray(_INIT, dtype=jnp.int32), memory)
    phase, ingredient, loaded, reserved = m
    row, col = _position(obs)

    at_button_stand = (row == 3) & (col == 4)
    face_up = obs[2, 2, 1] == 1
    button_action = jnp.where(face_up, jnp.asarray(5, jnp.int32),
                              jnp.asarray(3, jnp.int32))
    button_route = _move_to(row, col, 3, 4)
    # From (5,4), going north would enter the pot at (4,4).  Step west into
    # the open column first, then ascend and approach the button from above.
    button_route = jnp.where((row > 3) & (col == 4),
                             jnp.asarray(2, jnp.int32), button_route)
    a0 = jnp.where(at_button_stand, button_action, button_route)
    is_phase0 = phase == 0
    activated = is_phase0 & at_button_stand & face_up

    target0 = jnp.sum(obs[:, :, 36])
    target1 = jnp.sum(obs[:, :, 37])
    selected = jnp.where(target1 > target0, jnp.asarray(1, jnp.int32),
                         jnp.asarray(0, jnp.int32))

    supply_row = jnp.where(ingredient == 1, jnp.asarray(5, jnp.int32),
                           jnp.asarray(3, jnp.int32))
    at_supply_stand = (row == supply_row) & (col == 3)
    supply_face = obs[2, 2, 4] == 1
    a_supply = jnp.where(supply_face, jnp.asarray(5, jnp.int32),
                         jnp.asarray(2, jnp.int32))
    # The lower supply is behind the pot: approach its column first, then
    # descend through (3,3), never attempting to walk through (4,4).
    lower_horizontal = jnp.where(col < 3, jnp.asarray(0, jnp.int32),
                                 jnp.asarray(2, jnp.int32))
    lower_vertical = jnp.where(row < supply_row, jnp.asarray(1, jnp.int32),
                               jnp.asarray(3, jnp.int32))
    lower_route = jnp.where(col != 3, lower_horizontal, lower_vertical)
    a2 = jnp.where(at_supply_stand, a_supply,
                   jnp.where(ingredient == 1, lower_route,
                             _move_to(row, col, supply_row, 3)))
    took_supply = (phase == 2) & at_supply_stand & supply_face

    at_pot_stand = (row == 3) & (col == 4)
    pot_face = obs[2, 2, 2] == 1
    a_pot = jnp.where(pot_face, jnp.asarray(5, jnp.int32),
                      jnp.asarray(1, jnp.int32))
    a3 = jnp.where(at_pot_stand, a_pot, _move_to(row, col, 3, 4))
    added = (phase == 3) & at_pot_stand & pot_face

    pot_cooked = _at(obs, 4, 4, 30) == 1
    a_wait = jnp.asarray(4, dtype=jnp.int32)

    at_plate_stand = (row == 3) & (col == 5)
    plate_face = obs[2, 2, 3] == 1
    a_plate = jnp.where(plate_face, jnp.asarray(5, jnp.int32),
                        jnp.asarray(0, jnp.int32))
    a6 = jnp.where(at_plate_stand, a_plate,
                   _move_to(row, col, 3, 5))
    took_plate = (phase == 6) & at_plate_stand & plate_face

    picked_dish = (phase == 7) & at_pot_stand & pot_face

    at_goal_stand = (row == 4) & (col == 5)
    goal_face = obs[2, 2, 3] == 1
    a_goal = jnp.where(goal_face, jnp.asarray(5, jnp.int32),
                       jnp.asarray(0, jnp.int32))
    a8 = jnp.where(at_goal_stand, a_goal,
                   _move_to(row, col, 4, 5, goal_mode=True))
    delivered = (phase == 8) & at_goal_stand & goal_face

    action = jnp.where(phase == 0, a0,
             jnp.where(phase == 1, jnp.asarray(4, jnp.int32),
             jnp.where(phase == 2, a2,
             jnp.where(phase == 3, a3,
             jnp.where(phase == 4, a2,
             jnp.where(phase == 5, a_wait,
             jnp.where(phase == 6, a6,
             jnp.where(phase == 7, a3, a8))))))))

    next_phase = jnp.where(activated, jnp.asarray(1, jnp.int32), phase)
    next_phase = jnp.where(phase == 1, jnp.asarray(2, jnp.int32), next_phase)
    next_phase = jnp.where(took_supply, jnp.asarray(3, jnp.int32), next_phase)
    next_phase = jnp.where(added & (loaded < 2), jnp.asarray(4, jnp.int32), next_phase)
    next_phase = jnp.where(added & (loaded >= 2), jnp.asarray(5, jnp.int32), next_phase)
    next_phase = jnp.where((phase == 4) & (loaded < 3), jnp.asarray(2, jnp.int32), next_phase)
    next_phase = jnp.where((phase == 4) & (loaded >= 3), jnp.asarray(5, jnp.int32), next_phase)
    next_phase = jnp.where((phase == 5) & pot_cooked, jnp.asarray(6, jnp.int32), next_phase)
    next_phase = jnp.where(took_plate, jnp.asarray(7, jnp.int32), next_phase)
    next_phase = jnp.where(picked_dish, jnp.asarray(8, jnp.int32), next_phase)
    next_phase = jnp.where(delivered, jnp.asarray(0, jnp.int32), next_phase)

    next_ingredient = jnp.where(phase == 1, selected, ingredient)
    next_loaded = jnp.where(activated, jnp.asarray(0, jnp.int32), loaded)
    next_loaded = jnp.where(added, loaded + 1, next_loaded)
    next_loaded = jnp.where(delivered, jnp.asarray(0, jnp.int32), next_loaded)
    next_memory = jnp.asarray([next_phase, next_ingredient, next_loaded, reserved], dtype=jnp.int32)
    trace = {
        'phase': jnp.asarray(phase, dtype=jnp.int32),
        'row': jnp.asarray(row, dtype=jnp.int32),
        'col': jnp.asarray(col, dtype=jnp.int32),
        'facing': jnp.asarray(jnp.argmax(obs[2, 2, 1:5]), dtype=jnp.int32),
        'ingredient': jnp.asarray(ingredient, dtype=jnp.int32),
        'loaded': jnp.asarray(loaded, dtype=jnp.int32),
    }
    return action, next_memory, trace
