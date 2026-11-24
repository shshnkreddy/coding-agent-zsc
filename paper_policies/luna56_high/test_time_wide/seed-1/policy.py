"""A convention-light solo cook for the upper room.

The lower room has no goal and no ingredient used by either possible recipe, so
the second seat deliberately stays out of the way. Agent 0 uses its local
static crop to recover its reset coordinate, then follows a fixed state
machine. The target recipe is not visible from the upper room in this layout;
successive dishes therefore alternate ingredient 0 and ingredient 1. A wrong
delivery leaves the target unchanged, so alternation changes the guess.
"""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'upper_room_solo_cycle',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}


# Static channels 22:32, in order: wall, goal, pot, recipe, button, plate,
# ingredient 0, ingredient 1, ingredient 2, ingredient 3.
_STATIC = [
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
]

_FLOOR = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
]
_DELTAS = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0], [0, 0]]


def _infer_position(obs):
    static = jnp.asarray(_STATIC, dtype=jnp.int32)
    cand_r = jnp.repeat(jnp.arange(7, dtype=jnp.int32), 6)
    cand_c = jnp.tile(jnp.arange(6, dtype=jnp.int32), 7)
    off = jnp.arange(-2, 3, dtype=jnp.int32)
    rows = cand_r[:, None, None] + off[None, :, None]
    cols = cand_c[:, None, None] + off[None, None, :]
    valid = (rows >= 0) & (rows < 7) & (cols >= 0) & (cols < 6)
    expected = static[jnp.clip(rows, 0, 6), jnp.clip(cols, 0, 5)]
    expected = jnp.where(valid[..., None], expected, 0)
    observed = obs[None, :, :, 22:32]
    error = jnp.sum(jnp.abs(expected - observed), axis=(1, 2, 3))
    best = jnp.argmin(error)
    return jnp.asarray([cand_r[best], cand_c[best]], dtype=jnp.int32)


def _move_position(pos, action):
    deltas = jnp.asarray(_DELTAS, dtype=jnp.int32)
    floor = jnp.asarray(_FLOOR, dtype=jnp.int32)
    candidate = pos + deltas[action]
    inside = (candidate[0] >= 0) & (candidate[0] < 7) & (candidate[1] >= 0) & (candidate[1] < 6)
    safe_r = jnp.clip(candidate[0], 0, 6)
    safe_c = jnp.clip(candidate[1], 0, 5)
    can_enter = inside & (floor[safe_r, safe_c] == 1)
    return jnp.where(can_enter, candidate, pos)


def _navigate(pos, target, facing_action):
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    at_target = (dr == 0) & (dc == 0)
    move_action = jnp.where(dr < 0, 3, jnp.where(dr > 0, 1, jnp.where(dc < 0, 2, 0)))
    action = jnp.where(at_target, facing_action, move_action).astype(jnp.int32)
    return action, _move_position(pos, action), at_target


def _with_phase(state, phase):
    return state.at[0].set(jnp.asarray(phase, dtype=jnp.int32))


def _nav_branch(state, target, facing_action, next_phase):
    action, new_pos, at_target = _navigate(state[1:3], target, facing_action)
    new_state = state.at[1:3].set(new_pos)
    new_state = jnp.where(at_target, _with_phase(new_state, next_phase), new_state)
    return action, new_state


def _agent_0_step(obs, state):
    desired = state[3]
    ingredient_target = jnp.asarray([[1, 1], [2, 1]], dtype=jnp.int32)[desired]
    pot_target = jnp.asarray([2, 1], dtype=jnp.int32)
    plate_target = jnp.asarray([1, 3], dtype=jnp.int32)
    delivery_target = jnp.asarray([1, 2], dtype=jnp.int32)

    def p0(_):
        return _nav_branch(state, ingredient_target, 2, 1)

    def p1(_):
        return jnp.asarray(5, dtype=jnp.int32), _with_phase(state, 2)

    def p2(_):
        return _nav_branch(state, pot_target, 1, 3)

    def p3(_):
        return jnp.asarray(5, dtype=jnp.int32), _with_phase(state, 4)

    def p4(_):
        return _nav_branch(state, ingredient_target, 2, 5)

    def p5(_):
        return jnp.asarray(5, dtype=jnp.int32), _with_phase(state, 6)

    def p6(_):
        return _nav_branch(state, pot_target, 1, 7)

    def p7(_):
        return jnp.asarray(5, dtype=jnp.int32), _with_phase(state, 8)

    def p8(_):
        return _nav_branch(state, ingredient_target, 2, 9)

    def p9(_):
        return jnp.asarray(5, dtype=jnp.int32), _with_phase(state, 10)

    def p10(_):
        return _nav_branch(state, pot_target, 1, 11)

    def p11(_):
        return jnp.asarray(5, dtype=jnp.int32), state.at[0].set(12).at[5].set(0)

    def p12(_):
        ready = jnp.any((obs[:, :, 24] == 1) & (obs[:, :, 33] == 1) & (obs[:, :, 44] == 0))
        def q0(_):
            action, pos, at_plate = _navigate(state[1:3], plate_target, 3)
            next_state = state.at[1:3].set(pos)
            next_state = jnp.where(at_plate, next_state.at[5].set(1), next_state)
            return jnp.where(at_plate, jnp.asarray(3, dtype=jnp.int32), action), next_state

        def q1(_):
            return jnp.asarray(5, dtype=jnp.int32), state.at[5].set(2)

        def q2(_):
            action, pos, at_pot = _navigate(state[1:3], pot_target, 1)
            next_state = state.at[1:3].set(pos)
            next_state = jnp.where(at_pot, next_state.at[5].set(3), next_state)
            return action, next_state

        def q3(_):
            return jnp.asarray(4, dtype=jnp.int32), state

        pre_action, pre_state = jax.lax.switch(state[5], (q0, q1, q2, q3), operand=None)
        ready_collect = ready & (state[5] == 3)
        collect_state = _with_phase(state, 17).at[5].set(0)
        return jnp.where(ready_collect, jnp.asarray(5, dtype=jnp.int32), pre_action), jnp.where(ready_collect, collect_state, pre_state)

    def p13(_):
        return _nav_branch(state, plate_target, 3, 14)

    def p14(_):
        return jnp.asarray(5, dtype=jnp.int32), _with_phase(state, 15)

    def p15(_):
        return _nav_branch(state, pot_target, 1, 16)

    def p16(_):
        return jnp.asarray(5, dtype=jnp.int32), _with_phase(state, 17)

    def p17(_):
        return _nav_branch(state, delivery_target, 3, 18)

    def p18(_):
        next_state = _with_phase(state, 0)
        next_state = next_state.at[3].set(1 - state[3])
        next_state = next_state.at[4].set(state[4] + 1)
        next_state = next_state.at[5].set(0)
        return jnp.asarray(5, dtype=jnp.int32), next_state

    branches = (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18)
    return jax.lax.switch(state[0], branches, operand=None)


def init_agent_0():
    return jnp.zeros((6,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    inferred = _infer_position(obs)
    reset_state = jnp.asarray([0, inferred[0], inferred[1], 0, 0, 0], dtype=jnp.int32)
    state = jnp.where(reset, reset_state, memory)
    action, next_state = _agent_0_step(obs, state)
    return action, next_state, {
        'phase': next_state[0],
        'row': next_state[1],
        'col': next_state[2],
        'desired': next_state[3],
        'subphase': next_state[5],
    }


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    zero = jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(4, dtype=jnp.int32), jnp.where(reset, zero, memory), {}
