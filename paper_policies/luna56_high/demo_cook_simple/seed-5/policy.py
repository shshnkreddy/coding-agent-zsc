"""Convention-light cooperative controller for the demo_cook_simple layout."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'topology_split_loader_server',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _blocked(obs, r, c):
    return jnp.any(obs[r, c, 20:29] > 0)


def _put(m, phase=None, target=None, count=None):
    if phase is not None:
        m = m.at[0].set(jnp.asarray(phase, dtype=jnp.int32))
    if target is not None:
        m = m.at[1].set(target)
    if count is not None:
        m = m.at[2].set(count)
    return m


def init_agent_0():
    return jnp.zeros((6,), dtype=jnp.int32)


def _a0_phase0(obs, m):
    right_block = _blocked(obs, 2, 3)
    up_block = _blocked(obs, 1, 2)
    corner = right_block & up_block
    recipe0 = jnp.any(obs[:, :, 36] > 0)
    target = jnp.where(recipe0, jnp.asarray(0, jnp.int32), jnp.asarray(1, jnp.int32))
    start_phase = jnp.where(target == 0, 1, 2)
    nm = jnp.where(corner, _put(m, phase=start_phase, target=target, count=jnp.asarray(0, jnp.int32)), m)
    action = jnp.where(corner, STAY, jnp.where(right_block, UP, RIGHT))
    return action, nm


def _a0_phase1(obs, m):
    at_source = obs[2, 1, 26] > 0
    return jnp.where(at_source, STAY, LEFT), _put(m, phase=jnp.where(at_source, 4, 1))


def _a0_phase2(obs, m):
    at_column = obs[2, 1, 26] > 0
    return jnp.where(at_column, STAY, LEFT), _put(m, phase=jnp.where(at_column, 3, 2))


def _a0_phase3(obs, m):
    at_row = _blocked(obs, 3, 2)
    return jnp.where(at_row, STAY, DOWN), _put(m, phase=jnp.where(at_row, 4, 3))


def _a0_phase4(obs, m):
    return LEFT, _put(m, phase=5)


def _a0_phase5(obs, m):
    return INTERACT, _put(m, phase=6)


def _a0_phase6(obs, m):
    at_pot_column = ((obs[1, 3, 22] > 0) |
                     (obs[2, 3, 22] > 0) |
                     (obs[3, 3, 22] > 0))
    return jnp.where(at_pot_column, STAY, RIGHT), _put(m, phase=jnp.where(at_pot_column, 7, 6))


def _a0_phase7(obs, m):
    at_pot_row = obs[2, 3, 22] > 0
    pot_below = obs[3, 3, 22] > 0
    # A blocked RIGHT both preserves the cell and establishes the facing
    # needed by the following INTERACT phase.
    action = jnp.where(at_pot_row, RIGHT, jnp.where(pot_below, DOWN, UP))
    return action, _put(m, phase=jnp.where(at_pot_row, 8, 7))


def _a0_phase8(obs, m):
    last = m[2] >= 2
    nm = _put(m, phase=jnp.where(last, 9, 10), count=m[2] + 1)
    return INTERACT, nm


def _a0_phase9(obs, m):
    empty = ~jnp.any(obs[2, 3, 29:34] > 0)
    return STAY, _put(m, phase=jnp.where(empty, 0, 9))


def _a0_phase10(obs, m):
    at_left_edge = _blocked(obs, 2, 1)
    return jnp.where(at_left_edge, STAY, LEFT), _put(m, phase=jnp.where(at_left_edge, 11, 10))


def _a0_phase11(obs, m):
    target0 = m[1] == 0
    up_done = _blocked(obs, 1, 2)
    down_done = _blocked(obs, 3, 2)
    done = jnp.where(target0, up_done, down_done)
    action = jnp.where(done, STAY, jnp.where(target0, UP, DOWN))
    return action, _put(m, phase=jnp.where(done, 4, 11))


def agent_0_act(obs, reset, memory, rng):
    del rng
    zero = jnp.zeros((6,), dtype=jnp.int32)
    m = jnp.where(reset, zero, memory)
    branches = (
        _a0_phase0, _a0_phase1, _a0_phase2, _a0_phase3,
        _a0_phase4, _a0_phase5, _a0_phase6, _a0_phase7,
        _a0_phase8, _a0_phase9, _a0_phase10, _a0_phase11,
    )
    action, nm = jax.lax.switch(m[0], branches, obs, m)
    pot = obs[:, :, 22] > 0
    pot_item = jnp.sum(pot * jnp.sum(obs[:, :, 29:34], axis=-1))
    pot_timer = jnp.max(obs[:, :, 39])
    trace = {'phase': m[0], 'target': m[1], 'count': m[2],
             'pot_item': pot_item, 'pot_timer': pot_timer}
    return jnp.asarray(action, dtype=jnp.int32), nm, trace


def init_agent_1():
    return jnp.zeros((4,), dtype=jnp.int32)


def _a1_phase0(obs, m):
    pot_left = obs[2, 1, 22] > 0
    left_block = _blocked(obs, 2, 1)
    ready_for_vertical = pot_left | left_block
    return jnp.where(ready_for_vertical, STAY, LEFT), _put(m, phase=jnp.where(ready_for_vertical, 1, 0))


def _a1_phase1(obs, m):
    pot_left = obs[2, 1, 22] > 0
    pot_below_left = obs[3, 1, 22] > 0
    action = jnp.where(pot_left, STAY, jnp.where(pot_below_left, DOWN, UP))
    return action, _put(m, phase=jnp.where(pot_left, 2, 1))


def _a1_phase2(obs, m):
    cooked = obs[2, 1, 30] > 0
    return STAY, _put(m, phase=jnp.where(cooked, 3, 2))


def _a1_phase3(obs, m):
    at_plate_column = jnp.any(obs[:3, 3, 25] > 0)
    return jnp.where(at_plate_column, STAY, RIGHT), _put(m, phase=jnp.where(at_plate_column, 4, 3))


def _a1_phase4(obs, m):
    at_plate_row = obs[2, 3, 25] > 0
    return jnp.where(at_plate_row, STAY, UP), _put(m, phase=jnp.where(at_plate_row, 5, 4))


def _a1_phase5(obs, m):
    return RIGHT, _put(m, phase=6)


def _a1_phase6(obs, m):
    return INTERACT, _put(m, phase=7)


def _a1_phase7(obs, m):
    at_left_edge = _blocked(obs, 2, 1)
    return jnp.where(at_left_edge, STAY, LEFT), _put(m, phase=jnp.where(at_left_edge, 8, 7))


def _a1_phase8(obs, m):
    at_pot_row = obs[2, 1, 22] > 0
    # Turn left into the blocked pot cell before the interaction step.
    return jnp.where(at_pot_row, LEFT, DOWN), _put(m, phase=jnp.where(at_pot_row, 9, 8))


def _a1_phase9(obs, m):
    return INTERACT, _put(m, phase=10)


def _a1_phase10(obs, m):
    at_goal_column = obs[2, 3, 21] > 0
    return jnp.where(at_goal_column, STAY, RIGHT), _put(m, phase=jnp.where(at_goal_column, 11, 10))


def _a1_phase11(obs, m):
    return INTERACT, _put(m, phase=12)


def _a1_phase12(obs, m):
    pot_left = obs[2, 1, 22] > 0
    return jnp.where(pot_left, STAY, LEFT), _put(m, phase=jnp.where(pot_left, 2, 12))


def agent_1_act(obs, reset, memory, rng):
    del rng
    zero = jnp.zeros((4,), dtype=jnp.int32)
    m = jnp.where(reset, zero, memory)
    branches = (
        _a1_phase0, _a1_phase1, _a1_phase2, _a1_phase3,
        _a1_phase4, _a1_phase5, _a1_phase6, _a1_phase7,
        _a1_phase8, _a1_phase9, _a1_phase10, _a1_phase11,
        _a1_phase12,
    )
    action, nm = jax.lax.switch(m[0], branches, obs, m)
    pot = obs[:, :, 22] > 0
    pot_item = jnp.sum(pot * jnp.sum(obs[:, :, 29:34], axis=-1))
    pot_timer = jnp.max(obs[:, :, 39])
    trace = {'phase': m[0], 'pot_item': pot_item, 'pot_timer': pot_timer}
    return jnp.asarray(action, dtype=jnp.int32), nm, trace
