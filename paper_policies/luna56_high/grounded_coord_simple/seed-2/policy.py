"""A grounded, convention-oriented controller for the small OCV2 layout.

The left room cannot reach the kitchen, so its useful contribution is to
refresh the recipe button. The right-room controller then performs the entire
physical cycle. All timing is local deterministic memory; object and
inventory checks make the action sequence tolerate randomized reset poses.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'button_signaled_right_worker',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


def _left_row_action(obs):
    """Move the left-room player toward row 2 using ingredient-2 markers."""
    top_marker = jnp.any(obs[1, :, 28] > 0)
    bottom_marker = jnp.any(obs[3, :, 28] > 0)
    at_c1 = obs[2, 1, 23] > 0
    return jnp.where(top_marker, jnp.asarray(1, jnp.int32),
                     jnp.where(bottom_marker, jnp.asarray(3, jnp.int32),
                               jnp.where(at_c1, jnp.asarray(0, jnp.int32),
                                         jnp.asarray(4, jnp.int32))))


def _left_at_center(obs):
    top_marker = jnp.any(obs[1, :, 28] > 0)
    bottom_marker = jnp.any(obs[3, :, 28] > 0)
    # The always-on R at c0 is one cell away at c1 and two at c2.
    at_c1 = obs[2, 1, 23] > 0
    return (~top_marker) & (~bottom_marker) & (~at_c1)


def init_agent_0():
    # [elapsed local steps]
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    state = jnp.where(reset, jnp.zeros((1,), dtype=jnp.int32), memory)
    t = state[0]
    cycle = t % 55
    centered = _left_at_center(obs)
    navigate = jnp.where(centered, jnp.asarray(4, jnp.int32), _left_row_action(obs))
    initial_face = jnp.where(centered, jnp.asarray(0, jnp.int32), navigate)
    signal = cycle == 4
    action = jnp.where(t < 3, navigate,
                       jnp.where(signal & centered, jnp.asarray(5, jnp.int32),
                                 jnp.where((cycle == 3) & centered, initial_face,
                                           jnp.asarray(4, jnp.int32))))
    next_memory = jnp.asarray([t + 1], dtype=jnp.int32)
    return action, next_memory, {}


def _right_row(obs):
    # Ingredient 0 is at row 1, ingredient 1 at row 3, and the goal at row 2.
    row1 = jnp.any(obs[2, :, 26] > 0)
    row3 = jnp.any(obs[2, :, 27] > 0)
    return jnp.where(row1, jnp.asarray(1, jnp.int32),
                     jnp.where(row3, jnp.asarray(3, jnp.int32),
                               jnp.asarray(2, jnp.int32)))


def _right_home_action(obs):
    row = _right_row(obs)
    row_action = jnp.where(row < 2, jnp.asarray(1, jnp.int32),
                           jnp.where(row > 2, jnp.asarray(3, jnp.int32),
                                     jnp.asarray(4, jnp.int32)))
    at_c7 = (obs[2, 2, 26] > 0) | (obs[2, 2, 27] > 0)
    pot_at_c6 = obs[2, 0, 22] > 0
    horizontal = jnp.where(at_c7 | pot_at_c6, jnp.asarray(2, jnp.int32),
                           jnp.asarray(4, jnp.int32))
    return jnp.where(row != 2, row_action, horizontal)


def init_agent_1():
    # [elapsed local steps, latest recipe: 0 or 1]
    return jnp.zeros((2,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    state = jnp.where(reset, jnp.zeros((2,), dtype=jnp.int32), memory)
    t = state[0]
    old_recipe = state[1]
    cycle = t % 55

    # The button is visible from the canonical home cell (row 2, col 5).
    displayed_1 = jnp.sum(obs[:, :, 37]) > 0
    displayed_any = (jnp.sum(obs[:, :, 36]) + jnp.sum(obs[:, :, 37])) > 0
    recipe = jnp.where(displayed_any, jnp.where(displayed_1, 1, 0), old_recipe)
    target_row = jnp.where(recipe == 0, 1, 3)

    # Nine actions per raw-ingredient loop. The inventory holds one unit, so
    # each unit is returned to the pot before the next one is collected:
    # vertical, right, right, pickup, left, left, vertical, face, load.
    unit_offset = cycle - 5
    in_collection = (unit_offset >= 0) & (unit_offset < 27)
    phase = unit_offset % 9
    collection_action = jnp.where(
        phase == 0,
        jnp.where(target_row == 1, 3, 1),
        jnp.where((phase == 1) | (phase == 2), 0,
                  jnp.where(phase == 3, 5,
                            jnp.where((phase == 4) | (phase == 5), 2,
                                      jnp.where(phase == 6,
                                                jnp.where(target_row == 1, 1, 3),
                                                jnp.where(phase == 7, 2, 5))))))

    plate_path = jnp.where(cycle == 32, 3,
                           jnp.where(cycle == 33, 2,
                                     jnp.where(cycle == 34, 5,
                                               jnp.where(cycle == 35, 1,
                                                         jnp.where(cycle == 36, 2, 4)))))
    holding_plate = obs[2, 2, 5] > 0
    pot_cooked = obs[2, 1, 30] > 0
    scoop_or_wait = jnp.where(holding_plate & pot_cooked,
                              jnp.asarray(5, jnp.int32),
                              jnp.asarray(4, jnp.int32))
    delivery_path = jnp.where(cycle == 52, 0,
                              jnp.where(cycle == 53, 0,
                                        jnp.where(cycle == 54, 5, 4)))

    action = jnp.asarray(4, dtype=jnp.int32)
    action = jnp.where(t < 6, _right_home_action(obs), action)
    action = jnp.where((cycle == 5) & displayed_any,
                       jnp.where(target_row == 1, 3, 1), action)
    action = jnp.where(in_collection, collection_action, action)
    # The three nine-action loops occupy cycles 5-31; then take a plate, wait
    # for cooking, and deliver.
    action = jnp.where((cycle >= 32) & (cycle <= 36), plate_path, action)
    action = jnp.where((cycle >= 37) & (cycle <= 51), scoop_or_wait, action)
    action = jnp.where((cycle >= 52) & (cycle <= 54), delivery_path, action)
    # The delivery interaction occurs at cycle 54; the next cycle is the
    # period boundary, where the player returns from c6 to canonical c5.
    action = jnp.where((cycle == 0) & (t >= 55), jnp.asarray(2, jnp.int32), action)

    next_memory = jnp.asarray([t + 1, recipe], dtype=jnp.int32)
    return action, next_memory, {}
