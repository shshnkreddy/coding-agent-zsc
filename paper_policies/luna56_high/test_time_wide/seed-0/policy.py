"""Convention-light controller for the two-room Overcooked layout.

The upper room can access the useful ingredient supplies, plates, and delivery
station, but cannot see the recipe indicator.  The lower room can see the
indicator and uses the shared counter at (3, 2) as a one-bit marker: marker
present means ingredient 0, marker absent means ingredient 1.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'marker_mediated_chef',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}


def _first_cell(mask):
    """Return the first visible cell in a fixed 5x5 mask."""
    flat = mask.reshape(-1)
    index = jnp.argmax(flat.astype(jnp.int32))
    return index // 5, index % 5, jnp.any(mask)


def _central_counter(obs):
    """The wall between the two pots, identified by both pot neighbours."""
    wall = obs[:, :, 22] > 0
    pot = obs[:, :, 24] > 0
    left_pot = jnp.concatenate((jnp.zeros((5, 1), dtype=bool), pot[:, :-1]), axis=1)
    right_pot = jnp.concatenate((pot[:, 1:], jnp.zeros((5, 1), dtype=bool)), axis=1)
    return wall & left_pot & right_pot


def _facing_id(obs):
    # Observation IDs are UP, DOWN, RIGHT, LEFT; movement action IDs differ.
    return jnp.argmax(obs[2, 2, 1:5])


def _navigate(obs, mask, wanted_row, wanted_col, face_action):
    """Move until mask's cell has the requested local offset, then interact."""
    row, col, visible = _first_cell(mask)
    row_delta = row - wanted_row
    col_delta = col - wanted_col
    move = jnp.where(
        row_delta > 0,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.where(
            row_delta < 0,
            jnp.asarray(3, dtype=jnp.int32),
            jnp.where(
                col_delta > 0,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.where(col_delta < 0, jnp.asarray(2, dtype=jnp.int32), face_action),
            ),
        ),
    )
    aligned = (row_delta == 0) & (col_delta == 0)
    facing = _facing_id(obs)
    # Convert movement action IDs to the observation's facing IDs:
    # movement UP/DOWN/RIGHT/LEFT = 3/1/0/2, while facing = 0/1/2/3.
    wanted_facing = jnp.where(
        face_action == 3,
        0,
        jnp.where(face_action == 1, 1, jnp.where(face_action == 0, 2, 3)),
    )
    aligned_action = jnp.where(facing == wanted_facing, 5, face_action)
    return jnp.where(visible, jnp.where(aligned, aligned_action, move), 2)


def _navigate_left_supply(obs, mask):
    """Use the left supply consistently; steer away from a visible right one."""
    row, col, visible = _first_cell(mask)
    right_supply = visible & (col > 2)
    normal = _navigate(obs, mask, 2, 1, 2)
    return jnp.where(right_supply, jnp.asarray(2, dtype=jnp.int32), normal)


def init_agent_0():
    # phase, ingredients_added, target_ingredient, delay
    return jnp.zeros((4,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero_memory = jnp.zeros((4,), dtype=jnp.int32)
    memory = jnp.where(reset, zero_memory, memory)
    phase, added, target, delay = memory
    initial_memory = memory.at[3].set(jnp.where(reset, 10, delay))
    phase, added, target, delay = initial_memory

    counter_mask = _central_counter(obs)
    marker_row, marker_col, counter_visible = _first_cell(counter_mask)
    marker_here = jnp.where(
        counter_visible,
        obs[marker_row, marker_col, 37] > 0,
        jnp.asarray(False),
    )
    observed_target = jnp.where(marker_here, 0, 1).astype(jnp.int32)
    target_zero = target == 0
    supply_mask = jnp.where(target_zero, obs[:, :, 28] > 0, obs[:, :, 29] > 0)
    supply_row, supply_col, supply_visible = _first_cell(supply_mask)
    held_target = jnp.where(target_zero, obs[2, 2, 7] > 0, obs[2, 2, 8] > 0)
    pot_mask = obs[:, :, 24] > 0
    pot_row, pot_col, pot_visible = _first_cell(pot_mask)
    pot_ready = jnp.where(pot_visible, obs[pot_row, pot_col, 33] > 0, jnp.asarray(False))
    self_plate = obs[2, 2, 5] > 0
    self_cooked = obs[2, 2, 6] > 0

    read_action = _navigate(obs, counter_mask, 3, 2, 1)
    supply_action = _navigate_left_supply(obs, supply_mask)
    pot_action = _navigate(obs, pot_mask, 3, 2, 1)
    plate_action = _navigate(obs, obs[:, :, 27] > 0, 1, 2, 3)
    goal_action = _navigate(obs, obs[:, :, 23] > 0, 1, 2, 3)

    phase0_waiting = delay > 0
    action0 = jnp.asarray(4, dtype=jnp.int32)
    next_phase0 = jnp.where(phase0_waiting, 0, 1)
    next_delay0 = jnp.where(phase0_waiting, delay - 1, 0)

    # Reading the marker must not interact with the counter, since a present
    # marker is an ingredient-3 item that would otherwise be picked up.
    action1 = jnp.where(read_action == 5, jnp.asarray(4, dtype=jnp.int32), read_action)
    read_complete = read_action == 5
    next_phase1 = jnp.where(read_complete, 2, 1)
    next_target1 = jnp.where(read_complete, observed_target, target)

    action2 = jnp.where(held_target, pot_action, supply_action)
    at_pot = pot_action == 5
    added_after = jnp.where(held_target & at_pot, added + 1, added)
    next_phase2 = jnp.where(added_after >= 3, 3, 2)

    action3 = jnp.asarray(4, dtype=jnp.int32)
    next_phase3 = jnp.where(pot_ready, 4, 3)

    action4 = jnp.where(self_plate, jnp.asarray(4, dtype=jnp.int32), plate_action)
    next_phase4 = jnp.where(self_plate, 5, 4)

    action5 = jnp.where(self_cooked, jnp.asarray(4, dtype=jnp.int32), pot_action)
    next_phase5 = jnp.where(self_cooked, 6, 5)

    action6 = jnp.where(self_cooked, goal_action, jnp.asarray(4, dtype=jnp.int32))
    delivered = ~self_cooked
    next_phase6 = jnp.where(delivered, 0, 6)
    next_delay6 = jnp.where(delivered, 10, 0)

    action = jnp.where(
        phase == 0,
        action0,
        jnp.where(
            phase == 1,
            action1,
            jnp.where(
                phase == 2,
                action2,
                jnp.where(phase == 3, action3, jnp.where(phase == 4, action4, jnp.where(phase == 5, action5, action6))),
            ),
        ),
    ).astype(jnp.int32)
    next_phase = jnp.where(
        phase == 0,
        next_phase0,
        jnp.where(
            phase == 1,
            next_phase1,
            jnp.where(
                phase == 2,
                next_phase2,
                jnp.where(phase == 3, next_phase3, jnp.where(phase == 4, next_phase4, jnp.where(phase == 5, next_phase5, next_phase6))),
            ),
        ),
    )
    next_added = jnp.where(phase == 1, 0, jnp.where(phase == 2, added_after, added))
    next_target = jnp.where(phase == 1, next_target1, target)
    next_delay = jnp.where(phase == 0, next_delay0, jnp.where(phase == 6, next_delay6, delay))
    next_memory = jnp.stack((next_phase, next_added, next_target, next_delay)).astype(jnp.int32)
    trace = {
        'phase': phase.astype(jnp.int32),
        'added': added.astype(jnp.int32),
        'target': target.astype(jnp.int32),
        'delay': delay.astype(jnp.int32),
        'held_target': held_target.astype(jnp.int32),
        'pot_ready': pot_ready.astype(jnp.int32),
        'self_plate': self_plate.astype(jnp.int32),
        'self_cooked': self_cooked.astype(jnp.int32),
        'supply_row': supply_row.astype(jnp.int32),
        'supply_col': supply_col.astype(jnp.int32),
        'supply_visible': supply_visible.astype(jnp.int32),
        'facing': _facing_id(obs).astype(jnp.int32),
        'inventory': obs[2, 2, 5:11].astype(jnp.int32),
    }
    return action, next_memory, trace


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    zero = jnp.asarray(0, dtype=jnp.int32)
    memory = jnp.where(reset, zero, memory)
    counter_mask = _central_counter(obs)
    counter_row, counter_col, counter_visible = _first_cell(counter_mask)
    marker = jnp.where(counter_visible, obs[counter_row, counter_col, 37] > 0, False)
    recipe_one = jnp.max(obs[:, :, 41]) > 0
    need_marker = ~recipe_one
    held_marker = obs[2, 2, 10] > 0
    marker_action = _navigate(obs, counter_mask, 1, 2, 3)
    ingredient_action = _navigate_left_supply(obs, obs[:, :, 31] > 0)
    need_ingredient = need_marker & ~marker & ~held_marker
    need_place = need_marker & ~marker & held_marker
    need_clear = recipe_one & marker
    action = jnp.where(
        need_ingredient,
        ingredient_action,
        jnp.where(need_place | need_clear, marker_action, jnp.asarray(4, dtype=jnp.int32)),
    ).astype(jnp.int32)
    trace = {
        'recipe_one': recipe_one.astype(jnp.int32),
        'marker': marker.astype(jnp.int32),
        'held_marker': held_marker.astype(jnp.int32),
    }
    return action, memory, trace
