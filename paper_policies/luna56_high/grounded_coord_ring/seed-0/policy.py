"""Reactive room-specialist policy for grounded_coord_ring."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'room_specialist_button_memory',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}


# Static objects are encoded as 1..9 (wall, goal, pot, recipe, button, plate,
# ingredient 0, ingredient 1, ingredient 2); floor is 0.  This is a literal
# so importing the policy performs no array construction.
_STATIC_CODES = (
    (1, 1, 1, 9, 4, 9, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 1, 1, 5, 1, 1, 0, 1),
    (9, 7, 0, 0, 0, 0, 6, 0, 9),
    (4, 0, 2, 0, 3, 0, 2, 0, 4),
    (9, 8, 0, 0, 0, 0, 6, 0, 9),
    (1, 0, 1, 1, 5, 1, 1, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 1, 1, 9, 4, 9, 1, 1, 1),
)

_FLOOR = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 1, 1, 1, 1, 1, 1, 0),
    (0, 1, 0, 0, 0, 0, 0, 1, 0),
    (0, 1, 0, 1, 1, 1, 0, 1, 0),
    (0, 1, 0, 1, 0, 1, 0, 1, 0),
    (0, 1, 0, 1, 1, 1, 0, 1, 0),
    (0, 1, 0, 0, 0, 0, 0, 1, 0),
    (0, 1, 1, 1, 1, 1, 1, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
)


def _locate(obs, prior_index):
    """Match the observed static crop, breaking repeated-pattern ties by prior."""
    static = obs[:, :, 20:29]
    observed_codes = jnp.where(
        jnp.sum(static, axis=2) > 0,
        jnp.argmax(static, axis=2) + 1,
        jnp.asarray(0, dtype=jnp.int32),
    )
    coords = jnp.asarray([(r, c) for r in range(9) for c in range(9)], dtype=jnp.int32)
    patches = jnp.asarray([
        [[_STATIC_CODES[rr][cc] if 0 <= rr < 9 and 0 <= cc < 9 else 0
          for cc in range(c - 2, c + 3)] for rr in range(r - 2, r + 3)]
        for r in range(9) for c in range(9)
    ], dtype=jnp.int32)
    exact = jnp.sum(observed_codes[None, :, :] == patches, axis=(1, 2))
    prior = coords[prior_index]
    distance = jnp.sum(jnp.abs(coords - prior[None, :]), axis=1)
    index = jnp.argmax(exact * 1000 - distance)
    return index, coords[index]


def _navigate(pos, facing, target, desired_facing):
    """Use shortest-path navigation on the eight-cell central floor ring."""
    ring = jnp.asarray(((3, 3), (3, 4), (3, 5), (4, 5),
                        (5, 5), (5, 4), (5, 3), (4, 3)), dtype=jnp.int32)
    current_node = jnp.argmax(jnp.sum(ring == pos[None, :], axis=1))
    target_node = jnp.argmax(jnp.sum(ring == target[None, :], axis=1))
    clockwise = jnp.mod(target_node - current_node, 8)
    counterclockwise = jnp.mod(current_node - target_node, 8)
    use_clockwise = clockwise <= counterclockwise
    cw_actions = jnp.asarray((0, 0, 1, 1, 2, 2, 3, 3), dtype=jnp.int32)
    ccw_actions = jnp.asarray((1, 2, 2, 3, 3, 0, 0, 1), dtype=jnp.int32)
    move = jnp.where(use_clockwise, cw_actions[current_node], ccw_actions[current_node])
    turn = jnp.where(desired_facing == 0, 3,
           jnp.where(desired_facing == 1, 1,
           jnp.where(desired_facing == 2, 0, 2)))
    at_target = (current_node == target_node)
    correctly_facing = jnp.sum(facing == desired_facing) > 0
    return jnp.where(at_target,
                     jnp.where(correctly_facing, 5, turn),
                     move).astype(jnp.int32)


def _predict_position(pos, action):
    dr = jnp.where(action == 3, -1, jnp.where(action == 1, 1, 0))
    dc = jnp.where(action == 2, -1, jnp.where(action == 0, 1, 0))
    candidate = pos + jnp.asarray([dr, dc], dtype=jnp.int32)
    clipped = jnp.clip(candidate, 0, 8)
    floor = jnp.asarray(_FLOOR, dtype=jnp.int32)
    valid = (action < 4) & (floor[clipped[0], clipped[1]] > 0)
    return jnp.where(valid, candidate, pos)


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    # The outer-ring seat stays out of the central room and does not compete
    # with a partner's production actions.
    action = jnp.asarray(4, dtype=jnp.int32)
    return action, jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory), {}


def init_agent_1():
    # predicted map index, remembered target, and elapsed transition count
    return (jnp.asarray(39, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32))


def agent_1_act(obs, reset, memory, rng):
    prior = jnp.where(reset, jnp.asarray(39, dtype=jnp.int32), memory[0])
    step_count = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory[2])
    index, pos = _locate(obs, prior)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    shown_0 = jnp.sum(obs[:, :, 36]) > 0
    shown_1 = jnp.sum(obs[:, :, 37]) > 0
    old_target = jnp.where(reset, jnp.asarray(-1, dtype=jnp.int32), memory[1])
    target = jnp.where(shown_0, jnp.asarray(0, dtype=jnp.int32), old_target)
    target = jnp.where(shown_1, jnp.asarray(1, dtype=jnp.int32), target)

    self_cell = obs[2, 2]
    held_plate = self_cell[5] > 0
    held_cooked = self_cell[6] > 0
    held_0 = self_cell[7] > 0
    held_1 = self_cell[8] > 0
    held_raw = held_0 | held_1

    pot_mask = obs[:, :, 22]
    pot_count = jnp.sum(pot_mask * (obs[:, :, 31] + obs[:, :, 32] + obs[:, :, 33]))
    pot_cooked = jnp.sum(pot_mask * obs[:, :, 30]) > 0
    pot_timer = jnp.sum(pot_mask * obs[:, :, 39])
    pot_ready = pot_cooked & (pot_timer == 0)

    # Every target is a floor location adjacent to the relevant object.
    button_stand = jnp.asarray([3, 4], dtype=jnp.int32)
    ingredient_0_stand = jnp.asarray([3, 3], dtype=jnp.int32)
    ingredient_1_stand = jnp.asarray([5, 3], dtype=jnp.int32)
    pot_left = jnp.asarray([4, 3], dtype=jnp.int32)
    pot_right = jnp.asarray([4, 5], dtype=jnp.int32)
    plate_stand = jnp.asarray([3, 5], dtype=jnp.int32)
    delivery_stand = jnp.asarray([4, 5], dtype=jnp.int32)

    go_button = _navigate(pos, facing, button_stand, 0)
    go_ingredient = _navigate(
        pos, facing,
        jnp.where(target == 0, ingredient_0_stand, ingredient_1_stand), 3,
    )
    go_pot_left = _navigate(pos, facing, pot_left, 2)
    go_pot_right = _navigate(pos, facing, pot_right, 3)
    go_plate = _navigate(pos, facing, plate_stand, 2)
    go_delivery = _navigate(pos, facing, delivery_stand, 2)

    # Priority follows the physical item in hand, then the pot state.
    action_known = jnp.where(held_cooked, go_delivery,
                    jnp.where(held_plate, go_pot_right,
                    jnp.where(held_raw, go_pot_left,
                    jnp.where(pot_ready, go_plate,
                    jnp.where(pot_timer > 0, go_plate,
                    jnp.where(pot_count < 3, go_ingredient, go_pot_right))))))
    action = jnp.where(
        target < 0,
        jnp.where(step_count < 360, go_button, jnp.asarray(4, dtype=jnp.int32)),
        action_known,
    ).astype(jnp.int32)

    next_pos = _predict_position(pos, action)
    delivered = held_cooked & (action == 5) & (pos[0] == 4) & (pos[1] == 5)
    next_target = jnp.where(delivered, jnp.asarray(-1, dtype=jnp.int32), target)
    next_memory = (next_pos[0] * 9 + next_pos[1], next_target, step_count + 1)
    trace = {
        'pos': pos.astype(jnp.int32),
        'target': target.astype(jnp.int32),
        'action': action.astype(jnp.int32),
        'pot': pot_count.astype(jnp.int32),
        'timer': pot_timer.astype(jnp.int32),
        'held': self_cell[5:9].astype(jnp.int32),
    }
    return action, next_memory, trace
