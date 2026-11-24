"""Spatially grounded cooperative controller for demo_cook_simple."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'spatial_role_fsm',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}


def init_agent_0():
    # [phase, route_step, ingredients_loaded, target_ingredient, route_mode]
    # phase 0 read recipe, 1 fetch ingredient, 2 load pot, 3 await collection.
    return jnp.asarray([0, 0, 0, 2, 0], dtype=jnp.int32)


def _recipe_from_view(obs, old_recipe):
    displayed_0 = jnp.sum(obs[:, :, 36]) > 0
    displayed_1 = jnp.sum(obs[:, :, 37]) > 0
    displayed = displayed_0 | displayed_1
    recipe = jnp.where(displayed_1, jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32))
    return jnp.where(displayed, recipe, old_recipe)


def agent_0_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_0()
    m = jnp.where(reset, initial, memory)
    phase, step, loaded, old_recipe, mode = m
    recipe = _recipe_from_view(obs, old_recipe)
    recipe_is_one = recipe == 1

    # Initially go to the upper-left recipe indicator. After a pot is emptied,
    # move from the pot to that same indicator in two steps.
    initial_route_done = step >= 8
    cycle_route_done = step >= 1
    read_done = jnp.where(mode == 0, initial_route_done, cycle_route_done)
    read_action_initial = jnp.where(step < 6, 0, jnp.where(step < 9, 3, 4))
    read_action_cycle = jnp.where(step < 1, 3, jnp.where(step < 2, 2, 4))
    read_action = jnp.where(mode == 0, read_action_initial, read_action_cycle)

    # Reach column 1, then the row of the selected left-room supply.
    first_fetch = loaded == 0
    fetch_vertical_action_first = jnp.where(
        recipe_is_one,
        jnp.where(step < 8, 1, jnp.where(step < 9, 2, 4)),
        4,
    )
    fetch_vertical_action_later = jnp.where(
        recipe_is_one,
        jnp.where(step < 7, 1, jnp.where(step < 8, 2, 4)),
        jnp.where(step < 7, 3, jnp.where(step < 8, 2, 4)),
    )
    fetch_action = jnp.where(
        step < 6,
        2,
        jnp.where(first_fetch, fetch_vertical_action_first, fetch_vertical_action_later),
    )
    fetch_interact = jnp.where(
        first_fetch,
        jnp.where(recipe_is_one, step >= 9, step >= 6),
        step >= 8,
    )

    # Return to the pot at its left side and add the held ingredient.
    pot_vertical_action = jnp.where(recipe_is_one, 3, 1)
    pot_action = jnp.where(
        step < 6,
        0,
        jnp.where(step < 7, pot_vertical_action, jnp.where(step < 8, 0, 5)),
    )
    pot_interact = step >= 8
    last_ingredient = loaded >= 2

    pot_mask = obs[:, :, 22]
    pot_contents = jnp.sum(pot_mask * jnp.sum(obs[:, :, 31:34], axis=2))
    pot_empty = pot_contents == 0

    # Phase 0: read the recipe and move to the left supply.
    next_phase = jnp.where(phase == 0, jnp.where(read_done, 1, 0), phase)
    next_step = jnp.where(phase == 0, jnp.where(read_done, 0, step + 1), step)
    next_loaded = jnp.where(phase == 0, jnp.where(read_done, 0, loaded), loaded)
    next_mode = mode
    next_recipe = recipe
    action = read_action

    # Phase 1: fetch an ingredient and turn toward the pot.
    fetch_phase = jnp.where(fetch_interact, 2, 1)
    fetch_step = jnp.where(fetch_interact, 0, step + 1)
    next_phase = jnp.where(phase == 1, fetch_phase, next_phase)
    next_step = jnp.where(phase == 1, fetch_step, next_step)
    action = jnp.where(phase == 1, jnp.where(fetch_interact, 5, fetch_action), action)

    # Phase 2: load one ingredient; after the third, wait for collection.
    pot_phase = jnp.where(
        pot_interact, jnp.where(last_ingredient, 3, 1), 2
    )
    pot_loaded = jnp.where(pot_interact, jnp.where(last_ingredient, 3, loaded + 1), loaded)
    next_phase = jnp.where(phase == 2, pot_phase, next_phase)
    next_step = jnp.where(phase == 2, jnp.where(pot_interact, 0, step + 1), next_step)
    next_loaded = jnp.where(phase == 2, pot_loaded, next_loaded)
    action = jnp.where(phase == 2, pot_action, action)

    # Phase 3: the partner's collection empties the pot, then reread recipe.
    next_phase = jnp.where(phase == 3, jnp.where(pot_empty, 0, 3), next_phase)
    next_step = jnp.where(phase == 3, 0, next_step)
    next_loaded = jnp.where(phase == 3, jnp.where(pot_empty, 0, loaded), next_loaded)
    next_mode = jnp.where(phase == 3, jnp.where(pot_empty, 1, mode), next_mode)
    action = jnp.where(phase == 3, 4, action)

    next_memory = jnp.asarray(
        [next_phase, next_step, next_loaded, next_recipe, next_mode], dtype=jnp.int32
    )
    trace = {
        'phase': phase,
        'step': step,
        'loaded': loaded,
        'recipe': recipe,
        'pot_empty': pot_empty.astype(jnp.int32),
    }
    return jnp.asarray(action, dtype=jnp.int32), next_memory, trace


def init_agent_1():
    # phase 0 get plate, 1 wait at pot and collect, 2 deliver.
    return jnp.asarray([0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_1()
    m = jnp.where(reset, initial, memory)
    phase, step = m

    # Every right-room spawn can reach the upper plate pile this way.
    plate_action = jnp.where(step < 2, 3, jnp.where(step < 4, 0, 5))
    plate_interact = step >= 4

    # Move to the pot's right side and wait until its cooked flag is visible.
    pot_route_action = jnp.where(
        step < 1, 2, jnp.where(step < 2, 1, jnp.where(step < 3, 2, 4))
    )
    pot_cooked = jnp.sum(obs[:, :, 22] * obs[:, :, 30]) > 0
    pot_collect = (step >= 3) & pot_cooked

    # After collection, move to column 9 and interact with the goal at 10.
    deliver_action = jnp.where(step < 1, 0, 5)
    deliver = step >= 1

    next_phase = jnp.where(phase == 0, jnp.where(plate_interact, 1, 0), phase)
    next_step = jnp.where(phase == 0, jnp.where(plate_interact, 0, step + 1), step)
    action = plate_action

    next_phase = jnp.where(phase == 1, jnp.where(pot_collect, 2, 1), next_phase)
    next_step = jnp.where(phase == 1, jnp.where(pot_collect, 0, step + 1), next_step)
    action = jnp.where(phase == 1, jnp.where(pot_collect, 5, pot_route_action), action)

    next_phase = jnp.where(phase == 2, jnp.where(deliver, 0, 2), next_phase)
    next_step = jnp.where(phase == 2, jnp.where(deliver, 0, step + 1), next_step)
    action = jnp.where(phase == 2, deliver_action, action)

    next_memory = jnp.asarray([next_phase, next_step], dtype=jnp.int32)
    trace = {
        'phase': phase,
        'step': step,
        'pot_cooked': pot_cooked.astype(jnp.int32),
    }
    return jnp.asarray(action, dtype=jnp.int32), next_memory, trace
