"""Observation-driven two-seat cook / plate / deliver policy."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "wide_recipe_roles_v1",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 6,
    "max_width": 11,
    "num_ingredients": 4,
}


def _seek_field(obs, field, fallback):
    """Move toward a visible field, then face its selected cell."""
    visible = jnp.any(field > 0)
    flat_index = jnp.argmax(field.reshape(-1))
    rr = flat_index // 5
    cc = flat_index - rr * 5
    dr = rr - 2
    dc = cc - 2
    distance = jnp.abs(dr) + jnp.abs(dc)
    # Prefer horizontal progress when diagonal.  That follows the wide
    # room's corridor around the side pockets instead of walking into their
    # blocking counters.
    move = jnp.where(dc < 0, 2, jnp.where(dc > 0, 0, jnp.where(dr < 0, 3, 1)))
    desired_facing = jnp.where(dc < 0, 3, jnp.where(dc > 0, 2, jnp.where(dr < 0, 0, 1)))
    facing = jnp.argmax(obs[2, 2, 1:5])
    face_ready = facing == desired_facing
    action = jnp.where(
        distance == 0,
        4,
        jnp.where(distance == 1, jnp.where(face_ready, 5, move), move),
    )
    return jnp.where(visible, action, fallback).astype(jnp.int32)


def _seek(obs, channel, fallback):
    """Move toward a visible static object, then face it and interact."""
    return _seek_field(obs, obs[:, :, channel], fallback)


def _pot_info(obs):
    field = obs[:, :, 24]
    visible = jnp.any(field > 0)
    idx = jnp.argmax(field.reshape(-1))
    rr = idx // 5
    cc = idx - rr * 5
    item = jnp.sum(obs[rr, cc, 32:38])
    raw_count = jnp.sum(obs[rr, cc, 34:38])
    cooked = obs[rr, cc, 33] > 0
    timer = obs[rr, cc, 44]
    return visible, item > 0, cooked, raw_count, timer


def _face_and_interact(obs, channel, ready):
    field = obs[:, :, channel]
    idx = jnp.argmax(field.reshape(-1))
    rr = idx // 5
    cc = idx - rr * 5
    dr = rr - 2
    dc = cc - 2
    distance = jnp.abs(dr) + jnp.abs(dc)
    move = jnp.where(dc < 0, 2, jnp.where(dc > 0, 0, jnp.where(dr < 0, 3, 1)))
    desired_facing = jnp.where(dc < 0, 3, jnp.where(dc > 0, 2, jnp.where(dr < 0, 0, 1)))
    facing = jnp.argmax(obs[2, 2, 1:5])
    face_ready = facing == desired_facing
    visible = jnp.any(field > 0)
    interact = (distance == 1) & face_ready & ready
    adjacent_wait = (distance == 1) & ~ready
    action = jnp.where(
        interact,
        5,
        jnp.where(distance == 0, 4, jnp.where(adjacent_wait, 4, move)),
    )
    return jnp.where(visible, action, 4).astype(jnp.int32)


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    empty_plate = obs[2, 2, 5] > 0
    cooked_dish = obs[2, 2, 6] > 0
    _, _, pot_cooked, _, _ = _pot_info(obs)
    plate_action = _seek(obs, 27, 4)
    pot_action = _face_and_interact(obs, 24, pot_cooked)
    goal_action = _seek(obs, 23, 4)
    action = jnp.where(cooked_dish, goal_action, jnp.where(empty_plate, pot_action, plate_action))
    next_memory = jnp.where(reset, jnp.asarray(0, jnp.int32), memory)
    return action, next_memory, {}


def init_agent_1():
    return jnp.asarray(-1, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    target = jnp.where(reset, -1, memory)
    recipe_cell = obs[:, :, 25] > 0
    recipe_zero = jnp.any(recipe_cell & (obs[:, :, 40] > 0))
    recipe_one = jnp.any(recipe_cell & (obs[:, :, 41] > 0))
    target = jnp.where(recipe_one, 1, jnp.where(recipe_zero, 0, target))

    held_raw = jnp.sum(obs[2, 2, 7:11]) > 0
    pot_visible, pot_occupied, pot_cooked, pot_raw_count, pot_timer = _pot_info(obs)
    del pot_visible

    wall_left = jnp.any(obs[2, :2, 22] > 0)
    wall_right = jnp.any(obs[2, 3:, 22] > 0)
    above_wall = obs[1, 2, 22] > 0
    center_fallback = jnp.where(
        above_wall,
        jnp.where(wall_left, 0, jnp.where(wall_right, 2, 0)),
        3,
    )
    learn_action = _seek(obs, 25, center_fallback)

    raw_fallback = jnp.where(
        above_wall,
        jnp.where(wall_left, 0, jnp.where(wall_right, 2, 0)),
        3,
    )
    raw_action = _face_and_interact(obs, 24, True)
    raw_action = jnp.where(jnp.any(obs[:, :, 24] > 0), raw_action, raw_fallback)

    lower_rows = jnp.asarray([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1]], dtype=obs.dtype)
    supply_field_0 = obs[:, :, 28] * lower_rows
    supply_field_1 = obs[:, :, 29] * lower_rows
    supply_visible_0 = jnp.any(supply_field_0 > 0)
    supply_visible_1 = jnp.any(supply_field_1 > 0)
    supply_action_0 = _seek_field(obs, supply_field_0, 4)
    supply_action_1 = _seek_field(obs, supply_field_1, 4)
    supply_action = jnp.where(target == 1, supply_action_1, supply_action_0)
    supply_visible = jnp.where(target == 1, supply_visible_1, supply_visible_0)
    supply_action = jnp.where(supply_visible, supply_action, 2)
    # At the mouth of either lower side pocket, ingredient 0 is diagonally
    # beyond a wall.  Horizontal-first seeking would repeatedly press into
    # that wall; step into the pocket first, then seek the adjacent supply.
    side_wall = (obs[2, 1, 22] > 0) | (obs[2, 3, 22] > 0)
    supply_action = jnp.where(
        (target == 0) & supply_visible & side_wall,
        1,
        supply_action,
    )

    # A full pot starts its timer on the same interaction that adds the third
    # unit, so timer/cooked state is the reliable transition barrier.  Raw
    # count alone is deliberately not used as a wait condition.
    pot_unavailable = pot_cooked | (pot_timer > 0)
    action = jnp.where(
        held_raw,
        raw_action,
        jnp.where(target < 0, learn_action, jnp.where(pot_unavailable, 4, supply_action)),
    )
    next_memory = target.astype(jnp.int32)
    trace = {
        "target": target.astype(jnp.int32),
        "held_raw": held_raw.astype(jnp.int32),
        "pot_occupied": pot_occupied.astype(jnp.int32),
        "pot_raw_count": pot_raw_count.astype(jnp.int32),
        "pot_timer": pot_timer.astype(jnp.int32),
        "pot_cooked": pot_cooked.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
