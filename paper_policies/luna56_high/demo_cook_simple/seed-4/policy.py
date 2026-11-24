"""Geometry-driven cooperative policy for the demo cook layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'geometry_cycle_v1',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}


def _recipe(obs):
    seen0 = jnp.any(obs[:, :, 36] > 0)
    seen1 = jnp.any(obs[:, :, 37] > 0)
    return jnp.where(seen0, jnp.asarray(0, jnp.int32),
                    jnp.where(seen1, jnp.asarray(1, jnp.int32),
                              jnp.asarray(2, jnp.int32)))


def _pot_empty_left(obs):
    return (obs[2, 3, 29] == 0) & (obs[2, 3, 30] == 0) & \
        (obs[2, 3, 31] == 0) & (obs[2, 3, 32] == 0) & \
        (obs[2, 3, 33] == 0)


def _agent0_step(obs, reset, memory):
    # phase, tick, ingredient (0/1/2 unknown), loaded_units
    initial = jnp.asarray([0, 0, 2, 0], dtype=jnp.int32)
    mem = jnp.where(reset, initial, memory)
    phase, tick, ingredient, loaded = mem
    visible_recipe = _recipe(obs)
    found = jnp.where(visible_recipe < 2, visible_recipe, ingredient)

    searching = phase == 0
    go_top = phase == 1
    go_bottom = phase == 2
    left_to_supply = phase == 3
    pickup = phase == 4
    align_pot = phase == 5
    right_to_pot = phase == 6
    deposit = phase == 7
    return_supply = phase == 8
    align_supply = phase == 9
    face_supply = phase == 10
    wait_pot = phase == 11

    # Defaults are only used for unreachable/unknown phase values.
    action = jnp.asarray(4, jnp.int32)
    next_mem = mem

    search_done = visible_recipe < 2
    search_mem = jnp.asarray([
        jnp.where(search_done, jnp.where(found == 0, 1, 2), 0),
        0, found, loaded
    ], jnp.int32)
    next_mem = jnp.where(searching, search_mem, next_mem)
    action = jnp.where(searching,
                       jnp.where(search_done,
                                 jnp.where(found == 0, 3, 1), 0), action)

    edge_up = jnp.any(obs[1, 2, 20:29] > 0)
    top_done = edge_up | (tick >= 2)
    top_phase = jnp.where(top_done, 3, 1).astype(jnp.int32)
    top_mem = jnp.asarray([top_phase, jnp.where(top_done, 0, tick + 1),
                           found, loaded], jnp.int32)
    next_mem = jnp.where(go_top, top_mem, next_mem)
    action = jnp.where(go_top, 3, action)

    edge_down = jnp.any(obs[3, 2, 20:29] > 0)
    bottom_done = edge_down | (tick >= 2)
    bottom_phase = jnp.where(bottom_done, 3, 2).astype(jnp.int32)
    bottom_mem = jnp.asarray([bottom_phase, jnp.where(bottom_done, 0, tick + 1),
                              found, loaded], jnp.int32)
    next_mem = jnp.where(go_bottom, bottom_mem, next_mem)
    action = jnp.where(go_bottom, 1, action)

    supply_visible = jnp.where(found == 0, obs[2, 1, 26] > 0,
                               obs[2, 1, 27] > 0)
    at_left_end = supply_visible | (tick >= 6)
    left_phase = jnp.where(at_left_end, 4, 3).astype(jnp.int32)
    left_mem = jnp.asarray([left_phase, jnp.where(at_left_end, 0, tick + 1),
                            found, loaded], jnp.int32)
    next_mem = jnp.where(left_to_supply, left_mem, next_mem)
    action = jnp.where(left_to_supply, 2, action)

    next_mem = jnp.where(pickup, jnp.asarray([5, 0, found, loaded], jnp.int32),
                         next_mem)
    action = jnp.where(pickup, 5, action)

    # Supplies are on row 1 (ingredient 0) or row 3 (ingredient 1), while
    # the pot is on row 2.  Align vertically before walking right to it.
    align_action = jnp.where(found == 0, 1, 3).astype(jnp.int32)
    supply_action = jnp.where(found == 0, 3, 1).astype(jnp.int32)
    next_mem = jnp.where(align_pot,
                         jnp.asarray([6, 0, found, loaded], jnp.int32),
                         next_mem)
    action = jnp.where(align_pot, align_action, action)

    right_phase = jnp.where(tick >= 5, 7, 6).astype(jnp.int32)
    right_mem = jnp.asarray([right_phase, jnp.where(tick >= 5, 0, tick + 1),
                             found, loaded], jnp.int32)
    next_mem = jnp.where(right_to_pot, right_mem, next_mem)
    action = jnp.where(right_to_pot, 0, action)

    after_deposit_phase = jnp.where(loaded + 1 >= 3, 11, 8).astype(jnp.int32)
    after_deposit = jnp.asarray([after_deposit_phase, 0, found, loaded + 1],
                                jnp.int32)
    next_mem = jnp.where(deposit, after_deposit, next_mem)
    action = jnp.where(deposit, 5, action)

    return_phase = jnp.where(tick >= 5, 9, 8).astype(jnp.int32)
    return_tick = jnp.where(tick >= 5, 0, tick + 1).astype(jnp.int32)
    next_mem = jnp.where(return_supply,
                         jnp.asarray([return_phase, return_tick, found, loaded], jnp.int32),
                         next_mem)
    action = jnp.where(return_supply, 2, action)

    next_mem = jnp.where(align_supply,
                         jnp.asarray([10, 0, found, loaded], jnp.int32), next_mem)
    action = jnp.where(align_supply, supply_action, action)

    next_mem = jnp.where(face_supply,
                         jnp.asarray([4, 0, found, loaded], jnp.int32), next_mem)
    action = jnp.where(face_supply, 2, action)

    wait_ready = _pot_empty_left(obs)
    wait_recipe = jnp.where(visible_recipe < 2, visible_recipe, found)
    wait_mem = jnp.asarray([
        jnp.where(wait_ready, jnp.where(wait_recipe == 0, 1, 2), 8),
        0, wait_recipe, jnp.where(wait_ready, 0, loaded)
    ], jnp.int32)
    next_mem = jnp.where(wait_pot, wait_mem, next_mem)
    action = jnp.where(wait_pot, 4, action)

    # Reset is also the first live decision; begin moving right to expose the
    # recipe when the randomized spawn is far from an indicator.
    action = jnp.where(reset, 0, action)
    next_mem = jnp.where(reset, initial, next_mem)
    return action, next_mem


def init_agent_0():
    return jnp.zeros((4,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    action, next_memory = _agent0_step(obs, reset, memory)
    return action, next_memory, {}


def _pot_ready_right(obs):
    return obs[2, 1, 30] > 0


def _agent1_step(obs, reset, memory):
    # phase, tick: move to plate, move to pot, wait, move to goal, deliver
    initial = jnp.asarray([0, 0], dtype=jnp.int32)
    mem = jnp.where(reset, initial, memory)
    phase, tick = mem
    up = phase == 0
    right_plate = phase == 1
    take_plate = phase == 2
    down = phase == 3
    left_pot = phase == 4
    wait = phase == 5
    right_goal = phase == 6
    deliver = phase == 7

    action = jnp.asarray(4, jnp.int32)
    next_mem = mem

    top_edge = jnp.any(obs[1, 2, 20:29] > 0)
    up_done = top_edge | (tick >= 2)
    up_mem = jnp.asarray([jnp.where(up_done, 1, 0),
                          jnp.where(up_done, 0, tick + 1)], jnp.int32)
    next_mem = jnp.where(up, up_mem, next_mem)
    action = jnp.where(up, 3, action)

    plate_right = obs[2, 3, 25] > 0
    plate_done = plate_right | (tick >= 2)
    plate_mem = jnp.asarray([jnp.where(plate_done, 2, 1),
                             jnp.where(plate_done, 0, tick + 1)], jnp.int32)
    next_mem = jnp.where(right_plate, plate_mem, next_mem)
    action = jnp.where(right_plate, 0, action)

    next_mem = jnp.where(take_plate, jnp.asarray([3, 0], jnp.int32), next_mem)
    action = jnp.where(take_plate, 5, action)
    next_mem = jnp.where(down, jnp.asarray([4, 0], jnp.int32), next_mem)
    action = jnp.where(down, 1, action)
    next_mem = jnp.where(left_pot, jnp.asarray([5, 0], jnp.int32), next_mem)
    action = jnp.where(left_pot, 2, action)

    ready = _pot_ready_right(obs)
    wait_mem = jnp.asarray([jnp.where(ready, 6, 5), 0], jnp.int32)
    next_mem = jnp.where(wait, wait_mem, next_mem)
    action = jnp.where(wait, jnp.where(ready, 5, 4), action)

    next_mem = jnp.where(right_goal, jnp.asarray([7, 0], jnp.int32), next_mem)
    action = jnp.where(right_goal, 0, action)
    next_mem = jnp.where(deliver, jnp.asarray([0, 0], jnp.int32), next_mem)
    action = jnp.where(deliver, 5, action)

    action = jnp.where(reset, 3, action)
    # The reset action itself is the first upward navigation step.
    next_mem = jnp.where(reset, jnp.asarray([0, 1], jnp.int32), next_mem)
    return action, next_mem


def init_agent_1():
    return jnp.zeros((2,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    action, next_memory = _agent1_step(obs, reset, memory)
    return action, next_memory, {}
