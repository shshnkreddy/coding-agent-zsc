"""Reactive two-role controller for the demo cook layout.

The controller deliberately uses only the local crop.  Seat 0 is the
ingredient runner; seat 1 is the plate/delivery runner.  Navigation is based
on the nearest visible instance of a static object, which avoids relying on a
shared timing convention or on the randomized initial facing.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'reactive_roles_a01',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}


def _nearest_action(obs, channel, fallback, interact=True, left_only=False):
    """Move toward the nearest visible cell in a channel, or use fallback."""
    rows, cols = jnp.indices((5, 5))
    visible = obs[:, :, channel] > 0
    # Seat 0 must not chase the right-room supply that can appear at the
    # far-right edge of its unoccluded crop.  The left-room supplies are
    # always in columns 0..3 of that local crop when visible.
    if left_only:
        visible = visible & (cols < 4)
    distance = jnp.abs(rows - 2) + jnp.abs(cols - 2)
    score = jnp.where(visible, distance, 99)
    flat = jnp.argmin(score)
    row = flat // 5
    col = flat % 5
    found = jnp.any(visible)
    dr = row - 2
    dc = col - 2
    adjacent = (jnp.abs(dr) + jnp.abs(dc)) == 1
    vertical = jnp.where(dr < 0, 3, jnp.where(dr > 0, 1, 4))
    horizontal = jnp.where(dc < 0, 2, jnp.where(dc > 0, 0, 4))
    move = jnp.where(dr != 0, vertical, horizontal)
    toward = jnp.where(adjacent & interact, 5,
                       jnp.where(adjacent, 4, move))
    return jnp.asarray(jnp.where(found, toward, fallback), dtype=jnp.int32)


def _nearest_wait_action(obs, channel, fallback):
    """Move toward a target, but wait rather than interact when adjacent."""
    return _nearest_action(obs, channel, fallback, interact=False)


def init_agent_0():
    # 0/1: target ingredient, plus 2: recipe has been observed.
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    old_memory = jnp.where(reset, 0, memory)
    old_target = old_memory % 2
    was_known = old_memory >= 2

    recipe_zero = jnp.max(obs[:, :, 36]) > 0
    recipe_one = jnp.max(obs[:, :, 37]) > 0
    seen = recipe_zero | recipe_one
    target = jnp.where(recipe_one, 1, old_target)
    target = jnp.where(recipe_zero, 0, target)
    known = was_known | seen
    next_memory = 2 * known.astype(jnp.int32) + target.astype(jnp.int32)

    pot = obs[:, :, 22] > 0
    pot_raw = jnp.max(jnp.where(pot, obs[:, :, 31 + target], 0))
    pot_cooked = jnp.max(jnp.where(pot, obs[:, :, 30], 0))
    pot_timer = jnp.max(jnp.where(pot, obs[:, :, 39], 0))
    pot_busy = (pot_raw >= 3) | (pot_cooked > 0) | (pot_timer > 0)
    held = obs[2, 2, 7 + target] > 0

    to_supply = _nearest_action(obs, 26 + target, 2, interact=True,
                                left_only=True)
    to_pot = _nearest_action(obs, 22, 0, interact=True)
    normal_action = jnp.where(
        held,
        to_pot,
        jnp.where(pot_busy, 4, to_supply),
    )
    # First expose the recipe near the right edge of the left room.  Once it
    # is known, normal ingredient/pot behavior takes over.
    action = jnp.where(known, normal_action, 0)
    return action, next_memory, {}


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del reset, memory, rng
    self_plate = obs[2, 2, 5] > 0
    self_cooked = obs[2, 2, 6] > 0
    pot = obs[:, :, 22] > 0
    pot_raw0 = jnp.max(jnp.where(pot, obs[:, :, 31], 0))
    pot_raw1 = jnp.max(jnp.where(pot, obs[:, :, 32], 0))
    pot_total = pot_raw0 + pot_raw1
    pot_timer = jnp.max(jnp.where(pot, obs[:, :, 39], 0))
    pot_ready = jnp.max(jnp.where(pot, obs[:, :, 30], 0)) > 0
    plate_phase = pot_ready | (pot_timer > 0) | (pot_total >= 3)

    to_plate = _nearest_action(obs, 25, 0, interact=True)
    to_pot = _nearest_action(obs, 22, 2, interact=True)
    to_goal = _nearest_action(obs, 21, 0, interact=True)
    wait_at_pot = _nearest_wait_action(obs, 22, 2)
    action = jnp.where(
        self_cooked,
        to_goal,
        jnp.where(self_plate, to_pot,
                  jnp.where(plate_phase, to_plate, wait_at_pot)),
    )
    return action, jnp.asarray(0, dtype=jnp.int32), {}
