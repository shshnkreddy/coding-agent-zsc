"""A small, convention-independent worker for the upper room."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'alternating_upper_worker', 'adapter': 'overcooked_v2',
    'adapter_version': 1, 'max_height': 7, 'max_width': 6,
    'num_ingredients': 4,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _initial_memory():
    # [phase, phase_ticks, recipe_choice, ingredients_already_in_pot, started]
    return jnp.zeros((5,), dtype=jnp.int32)


def init_agent_0():
    return _initial_memory()


def _worker(obs, reset, memory):
    memory = jnp.where(reset, _initial_memory(), memory)
    phase, ticks, choice, ingredients, started = memory
    # During phase 15 the worker stands at (1, 3), with the right pot at
    # local view coordinate (4, 2).  This remains a fixed-shape read.
    pot_timer = obs[4, 2, 44]

    action = STAY
    action = jnp.where(phase == 0, RIGHT, action)
    action = jnp.where(phase == 1, jnp.where(choice == 0, UP, DOWN), action)
    action = jnp.where(phase == 2, RIGHT, action)
    action = jnp.where(phase == 3, INTERACT, action)
    action = jnp.where(phase == 4, jnp.where(choice == 0, DOWN, LEFT), action)
    action = jnp.where(phase == 5, jnp.where(choice == 0, LEFT, DOWN), action)
    action = jnp.where(phase == 6, DOWN, action)
    action = jnp.where(phase == 7, INTERACT, action)
    action = jnp.where(phase == 8, jnp.where(choice == 0, UP, RIGHT), action)
    action = jnp.where(phase == 9, RIGHT, action)
    action = jnp.where(phase == 10, INTERACT, action)
    action = jnp.where(phase == 12, UP, action)
    action = jnp.where(phase == 13, UP, action)
    action = jnp.where(phase == 14, INTERACT, action)
    action = jnp.where(phase == 15, jnp.where(pot_timer > 0, STAY, DOWN), action)
    action = jnp.where(phase == 16, INTERACT, action)
    action = jnp.where(phase == 17, UP, action)
    action = jnp.where(phase == 18, LEFT, action)
    action = jnp.where(phase == 19, UP, action)
    action = jnp.where(phase == 20, INTERACT, action)

    next_phase = phase
    next_ticks = ticks
    next_choice = choice
    next_ingredients = ingredients
    next_started = started

    initial_search = jnp.where(started == 0, 4, 1)
    after_search = jnp.where((started == 1) & (choice == 0), 2, 1)
    next_phase = jnp.where(phase == 0, jnp.where(ticks >= initial_search, after_search, 0), next_phase)
    next_ticks = jnp.where(phase == 0, jnp.where(ticks >= initial_search, 0, ticks + 1), next_ticks)
    align_done = jnp.where(started == 0, ticks >= 1, ticks >= 0)
    next_phase = jnp.where(phase == 1, jnp.where(align_done, 2, 1), next_phase)
    next_ticks = jnp.where(phase == 1, jnp.where(align_done, 0, ticks + 1), next_ticks)
    next_phase = jnp.where(phase == 2, 3, next_phase)
    next_phase = jnp.where(phase == 3, 4, next_phase)
    next_phase = jnp.where(phase == 4, 5, next_phase)
    next_phase = jnp.where(phase == 5, 6, next_phase)
    next_phase = jnp.where(phase == 6, 7, next_phase)

    third = (phase == 7) & (ingredients >= 2)
    next_phase = jnp.where(phase == 7, jnp.where(third, 12, 8), next_phase)
    next_ingredients = jnp.where(
        (phase == 7) & (ingredients < 2), ingredients + 1, next_ingredients
    )
    next_phase = jnp.where(phase == 8, 9, next_phase)
    next_phase = jnp.where(phase == 9, 10, next_phase)
    next_phase = jnp.where(phase == 10, 4, next_phase)
    next_phase = jnp.where(phase == 12, 13, next_phase)
    next_phase = jnp.where(phase == 13, 14, next_phase)
    next_phase = jnp.where(phase == 14, 15, next_phase)
    next_phase = jnp.where(phase == 15, jnp.where(pot_timer == 0, 16, 15), next_phase)
    next_phase = jnp.where(phase == 16, 17, next_phase)
    next_phase = jnp.where(phase == 17, 18, next_phase)
    next_phase = jnp.where(phase == 18, 19, next_phase)
    next_phase = jnp.where(phase == 19, 20, next_phase)
    next_phase = jnp.where(phase == 20, 0, next_phase)
    next_choice = jnp.where(phase == 20, 1 - choice, next_choice)
    next_ingredients = jnp.where(phase == 20, 0, next_ingredients)
    next_started = jnp.where(phase == 20, 1, next_started)

    next_memory = jnp.stack((next_phase, next_ticks, next_choice, next_ingredients, next_started))
    trace = {'phase': phase, 'choice': choice, 'pot_timer': pot_timer}
    return action, next_memory, trace


def agent_0_act(obs, reset, memory, rng):
    del rng
    return _worker(obs, reset, memory)


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del obs, rng
    zero = jnp.asarray(0, dtype=jnp.int32)
    return STAY, jnp.where(reset, zero, memory), {}
