"""A map-based, role-symmetric cooking controller.

The two rooms make a public geometric division of labour useful: seat 0 brings
the requested ingredient from the left room to the pot, while seat 1 plates
and delivers from the right room. All navigation state is private, but the
map, object locations, and recipe displays are public observations.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "geometry_cycle",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 11,
    "num_ingredients": 3,
}


_GRID = (
    "WWWWWR2W0WW",
    "0      W  B",
    "W     APA X",
    "1      W  B",
    "WWWWWR2W1WW",
)
_HEIGHT = 5
_WIDTH = 11
_FLOOR_CHARS = (" ", "A")
_CHANNEL_FOR_CHAR = {"W": 20, "X": 21, "P": 22, "R": 23,
                     "B": 25, "0": 26, "1": 27, "2": 28}


def _static_grid():
    values = []
    for row in _GRID:
        row_values = []
        for char in row:
            cell = [0] * 9
            channel = _CHANNEL_FOR_CHAR.get(char)
            if channel is not None:
                cell[channel - 20] = 1
            row_values.append(cell)
        values.append(row_values)
    return values


_TRAVERSABLE = (
    (False, False, False, False, False, False, False, False, False, False, False),
    (False, True, True, True, True, True, True, False, True, True, False),
    (False, True, True, True, True, True, True, False, True, True, False),
    (False, True, True, True, True, True, True, False, True, True, False),
    (False, False, False, False, False, False, False, False, False, False, False),
)


def _view_for(position, static):
    """Build the static-channel crop expected at a candidate global cell."""
    row, col = position
    view = []
    for dr in range(-2, 3):
        line = []
        for dc in range(-2, 3):
            rr, cc = row + dr, col + dc
            if 0 <= rr < _HEIGHT and 0 <= cc < _WIDTH:
                line.append(static[rr][cc])
            else:
                line.append([0] * 9)
        view.append(line)
    return view


_AGENT_0_CELLS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
)
_AGENT_1_CELLS = (
    (1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9),
)
_MOVE_DR = (0, 1, 0, -1, 0, 0)
_MOVE_DC = (1, 0, -1, 0, 0, 0)
_MOVE_FACE = (2, 1, 3, 0, 2, 2)


def _visible_recipe(obs):
    recipe_zero = jnp.max(obs[:, :, 36])
    recipe_one = jnp.max(obs[:, :, 37])
    visible = (recipe_zero + recipe_one) > 0
    ingredient = jnp.where(recipe_one > 0, jnp.asarray(1, dtype=jnp.int32),
                           jnp.asarray(0, dtype=jnp.int32))
    return visible, ingredient


def _locate(obs, cells):
    static = _static_grid()
    expected = jnp.asarray([_view_for(p, static) for p in cells], dtype=jnp.int32)
    matches = jnp.all(obs[None, :, :, 20:29] == expected, axis=(1, 2, 3))
    index = jnp.argmax(matches.astype(jnp.int32))
    coordinates = jnp.asarray(cells, dtype=jnp.int32)
    return coordinates[index]


def _advance_position(state, action):
    row, col, facing = state[0], state[1], state[2]
    move_dr = jnp.asarray(_MOVE_DR, dtype=jnp.int32)
    move_dc = jnp.asarray(_MOVE_DC, dtype=jnp.int32)
    move_face = jnp.asarray(_MOVE_FACE, dtype=jnp.int32)
    proposed_row = row + move_dr[action]
    proposed_col = col + move_dc[action]
    in_bounds = ((proposed_row >= 0) & (proposed_row < _HEIGHT)
                 & (proposed_col >= 0) & (proposed_col < _WIDTH))
    safe_row = jnp.clip(proposed_row, 0, _HEIGHT - 1)
    safe_col = jnp.clip(proposed_col, 0, _WIDTH - 1)
    traversable = jnp.asarray(_TRAVERSABLE, dtype=jnp.bool_)[safe_row, safe_col]
    moved = (action <= 3) & in_bounds & traversable
    next_row = jnp.where(moved, proposed_row, row)
    next_col = jnp.where(moved, proposed_col, col)
    next_facing = jnp.where(action <= 3, move_face[action], facing)
    return next_row, next_col, next_facing


def _route_action(row, col, facing, phase, target):
    source_goal = jnp.where(target == 1, jnp.asarray(2, dtype=jnp.int32),
                            jnp.asarray(1, dtype=jnp.int32))
    goal = jnp.where(phase == 0, jnp.asarray(0, dtype=jnp.int32),
                     jnp.where(phase == 1, source_goal,
                               jnp.asarray(3, dtype=jnp.int32)))
    goal_row = jnp.asarray((1, 1, 3, 2, 2, 1, 2), dtype=jnp.int32)[goal]
    goal_col = jnp.asarray((5, 1, 1, 6, 8, 9, 9), dtype=jnp.int32)[goal]
    at_goal = (row == goal_row) & (col == goal_col)
    vertical = jnp.where(row < goal_row, jnp.asarray(1, dtype=jnp.int32),
                         jnp.where(row > goal_row, jnp.asarray(3, dtype=jnp.int32),
                                   jnp.asarray(4, dtype=jnp.int32)))
    horizontal = jnp.where(col < goal_col, jnp.asarray(0, dtype=jnp.int32),
                           jnp.where(col > goal_col, jnp.asarray(2, dtype=jnp.int32),
                                     jnp.asarray(4, dtype=jnp.int32)))
    route = jnp.where(row != goal_row, vertical, horizontal)
    source_or_pot = (phase == 1) | (phase == 2)
    desired_face = jnp.where(phase == 1, jnp.asarray(3, dtype=jnp.int32),
                             jnp.asarray(2, dtype=jnp.int32))
    orient = jnp.where(facing == desired_face, jnp.asarray(5, dtype=jnp.int32),
                       jnp.where(desired_face == 3, jnp.asarray(2, dtype=jnp.int32),
                                 jnp.asarray(0, dtype=jnp.int32)))
    return jnp.where(at_goal,
                     jnp.where(source_or_pot, orient, jnp.asarray(4, dtype=jnp.int32)),
                     route)


def init_agent_0():
    return jnp.zeros((6,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    located = _locate(obs, _AGENT_0_CELLS)
    initial = jnp.asarray((located[0], located[1], 0, 0, 0, 0), dtype=jnp.int32)
    state = jnp.where(reset, initial, memory)
    visible, observed_target = _visible_recipe(obs)
    target = jnp.where(visible, observed_target, state[4])
    row, col, facing, phase, count = state[0], state[1], state[2], state[3], state[5]
    action = _route_action(row, col, facing, phase, target)
    source_row = jnp.where(target == 1, jnp.asarray(3, dtype=jnp.int32),
                          jnp.asarray(1, dtype=jnp.int32))
    goal_row = jnp.where(phase == 0, 1, jnp.where(phase == 1, source_row, 2))
    goal_col = jnp.where(phase == 0, 5, jnp.where(phase == 1, 1, 6))
    at_goal = (row == goal_row) & (col == goal_col)
    scan_done = (phase == 0) & at_goal
    took_ingredient = (phase == 1) & at_goal & (facing == 3) & (action == 5)
    added = (phase == 2) & at_goal & (facing == 2) & (action == 5)
    pot_empty = jnp.all(obs[2, 3, 29:34] == 0)
    finished_wait = (phase == 3) & at_goal & pot_empty
    next_phase = phase
    next_phase = jnp.where(scan_done, 1, next_phase)
    next_phase = jnp.where(took_ingredient, 2, next_phase)
    next_phase = jnp.where(added & (count < 2), 1, next_phase)
    next_phase = jnp.where(added & (count >= 2), 3, next_phase)
    next_phase = jnp.where(finished_wait, 1, next_phase)
    next_count = jnp.where(added, jnp.where(count >= 2, 0, count + 1), count)
    next_count = jnp.where(finished_wait, 0, next_count)
    moved_row, moved_col, moved_facing = _advance_position(state, action)
    next_memory = jnp.asarray((moved_row, moved_col, moved_facing, next_phase,
                               target, next_count), dtype=jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_memory, {}


def init_agent_1():
    return jnp.zeros((4,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    located = _locate(obs, _AGENT_1_CELLS)
    initial = jnp.asarray((located[0], located[1], 0, 0), dtype=jnp.int32)
    state = jnp.where(reset, initial, memory)
    row, col, facing, phase = state[0], state[1], state[2], state[3]
    goal = jnp.where(phase == 0, jnp.asarray(4, dtype=jnp.int32),
                     jnp.where(phase == 1, jnp.asarray(5, dtype=jnp.int32),
                               jnp.asarray(6, dtype=jnp.int32)))
    goal_row = jnp.asarray((2, 1, 2), dtype=jnp.int32)[phase]
    goal_col = jnp.asarray((8, 9, 9), dtype=jnp.int32)[phase]
    at_goal = (row == goal_row) & (col == goal_col)
    goal_row_for_route = jnp.asarray((2, 1, 2), dtype=jnp.int32)[phase]
    goal_col_for_route = jnp.asarray((8, 9, 9), dtype=jnp.int32)[phase]
    vertical = jnp.where(row < goal_row_for_route, jnp.asarray(1, dtype=jnp.int32),
                         jnp.where(row > goal_row_for_route, jnp.asarray(3, dtype=jnp.int32),
                                   jnp.asarray(4, dtype=jnp.int32)))
    horizontal = jnp.where(col < goal_col_for_route, jnp.asarray(0, dtype=jnp.int32),
                           jnp.where(col > goal_col_for_route, jnp.asarray(2, dtype=jnp.int32),
                                     jnp.asarray(4, dtype=jnp.int32)))
    route = jnp.where(row != goal_row_for_route, vertical, horizontal)
    desired_face = jnp.asarray((3, 2, 2), dtype=jnp.int32)[phase]
    orient = jnp.where(facing == desired_face, jnp.asarray(5, dtype=jnp.int32),
                       jnp.where(desired_face == 3, jnp.asarray(2, dtype=jnp.int32),
                                 jnp.asarray(0, dtype=jnp.int32)))
    action = jnp.where(at_goal, orient, route)
    pot_ready = jnp.asarray(obs[2, 1, 30] > 0)
    ready = (phase == 0) & at_goal & pot_ready
    took_plate = (phase == 1) & at_goal & (facing == 2) & (action == 5)
    delivered = (phase == 2) & at_goal & (facing == 2) & (action == 5)
    next_phase = jnp.where(ready, 1, phase)
    next_phase = jnp.where(took_plate, 2, next_phase)
    next_phase = jnp.where(delivered, 0, next_phase)
    moved_row, moved_col, moved_facing = _advance_position(state, action)
    next_memory = jnp.asarray((moved_row, moved_col, moved_facing, next_phase), dtype=jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_memory, {}
