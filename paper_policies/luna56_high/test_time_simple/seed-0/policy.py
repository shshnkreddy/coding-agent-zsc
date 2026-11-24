"""A landmark-driven controller for the fixed test_time_simple layout."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'reactive_cycle_v1',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


def _move_action(direction):
    # Observation facing IDs are UP=0, DOWN=1, RIGHT=2, LEFT=3,
    # whereas movement action IDs are RIGHT=0, DOWN=1, LEFT=2, UP=3.
    return jnp.asarray((3, 1, 0, 2), dtype=jnp.int32)[direction]


def _navigate(obs, target_mask):
    """Move toward a visible object, then face and interact with it."""
    flat = target_mask.reshape((25,))
    index = jnp.argmax(flat)
    row = index // 5
    col = index - row * 5
    valid = jnp.max(target_mask) > 0
    dr = row - 2
    dc = col - 2
    adr = jnp.abs(dr)
    adc = jnp.abs(dc)

    vertical_dir = jnp.where(dr < 0, 0, 1)
    horizontal_dir = jnp.where(dc < 0, 3, 2)
    vertical = _move_action(vertical_dir)
    horizontal = _move_action(horizontal_dir)
    desired_dir = jnp.where(adr > 0, vertical_dir, horizontal_dir)
    desired_move = jnp.where(adr > 0, vertical, horizontal)

    # Prefer vertical movement on a diagonal.  This makes a visible corner
    # supply reachable without attempting to walk into the supply cell.
    move = jnp.where((adr > 1) | ((adr == 1) & (adc == 1)), vertical, horizontal)
    move = jnp.where((adr <= 1) & (adc <= 1) & (adr > 0), desired_move, move)
    adjacent = (adr + adc) == 1
    facing = jnp.argmax(obs[2, 2, 1:5])
    action = jnp.where(adjacent, jnp.where(facing == desired_dir, 5, desired_move), move)
    return jnp.where(valid, action, 4).astype(jnp.int32)


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero = jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(4, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    # phase: 0=get ingredient, 1=put ingredient in pot, 2=wait,
    # 3=get plate, 4=collect cooked pot, 5=deliver.
    # recipe: 0 or 1, selected independently at reset because the right room
    # has no view of the always-on recipe indicator.
    return jnp.asarray([0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    # The right room cannot see the recipe indicator.  A fixed convention is
    # less brittle for cross-play than tying an unobservable guess to reset
    # PRNG keys.
    sampled_recipe = jnp.asarray(0, dtype=jnp.int32)
    phase = jnp.where(reset, 0, memory[0]).astype(jnp.int32)
    recipe = jnp.where(reset, sampled_recipe, memory[1]).astype(jnp.int32)

    self_cell = obs[2, 2]
    held_ingredient = jnp.max(self_cell[7:10]) > 0
    held_plate = self_cell[5] > 0
    held_cooked = self_cell[6] > 0

    pot_mask = obs[:, :, 22]
    pot_index = jnp.argmax(pot_mask.reshape((25,)))
    pot_row = pot_index // 5
    pot_col = pot_index - pot_row * 5
    pot_counts = obs[:, :, 31:34][pot_row, pot_col]
    pot_total = jnp.sum(pot_counts)
    pot_cooked = obs[pot_row, pot_col, 30] > 0
    pot_timer = obs[pot_row, pot_col, 39]

    ingredient_mask = jnp.where(recipe == 0, obs[:, :, 26], obs[:, :, 27])
    actions = (
        _navigate(obs, ingredient_mask),
        _navigate(obs, pot_mask),
        jnp.asarray(4, dtype=jnp.int32),
        _navigate(obs, obs[:, :, 25]),
        _navigate(obs, pot_mask),
        _navigate(obs, obs[:, :, 21]),
    )
    action = actions[0]
    action = jnp.where(phase == 1, actions[1], action)
    action = jnp.where(phase == 2, actions[2], action)
    action = jnp.where(phase == 3, actions[3], action)
    action = jnp.where(phase == 4, actions[4], action)
    action = jnp.where(phase == 5, actions[5], action).astype(jnp.int32)

    next_phase = phase
    next_phase = jnp.where(phase == 0, jnp.where(held_ingredient, 1, 0), next_phase)
    next_phase = jnp.where(
        phase == 1,
        jnp.where(~held_ingredient, jnp.where(pot_total >= 3, 2, 0), 1),
        next_phase,
    )
    # The current phase-2 action is still STAY.  If one tick remains, the
    # following observation is ready, so begin the plate route immediately.
    next_phase = jnp.where(
        phase == 2,
        jnp.where(pot_cooked | (pot_timer <= 6), 3, 2),
        next_phase,
    )
    next_phase = jnp.where(phase == 3, jnp.where(held_plate, 4, 3), next_phase)
    next_phase = jnp.where(phase == 4, jnp.where(held_cooked, 5, 4), next_phase)
    next_phase = jnp.where(phase == 5, jnp.where(~held_cooked, 0, 5), next_phase)

    # Each of these phases only emits INTERACT when its visible target is
    # adjacent and correctly faced, so the interaction itself is sufficient
    # evidence to arm the next phase without an extra no-op call.
    did_interact = action == 5
    next_phase = jnp.where((phase == 0) & did_interact, 1, next_phase)
    next_phase = jnp.where(
        (phase == 1) & did_interact,
        jnp.where(pot_total >= 2, 2, 0),
        next_phase,
    )
    next_phase = jnp.where((phase == 3) & did_interact, 4, next_phase)
    next_phase = jnp.where((phase == 4) & did_interact, 5, next_phase)
    next_phase = jnp.where((phase == 5) & did_interact, 0, next_phase)

    delivered = (phase == 5) & (did_interact | (~held_cooked))
    next_recipe = jnp.where(delivered, 1 - recipe, recipe)
    next_memory = jnp.asarray([next_phase, next_recipe], dtype=jnp.int32)
    trace = {
        'phase': phase,
        'recipe': recipe,
        'pot_total': pot_total.astype(jnp.int32),
        'pot_cooked': pot_cooked,
        'pot_timer': pot_timer.astype(jnp.int32),
    }
    return action, next_memory, trace
