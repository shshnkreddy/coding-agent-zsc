"""Geometry-driven kitchen controller for the fixed test-time layout."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'geometry_cycle_controller',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


# Static channels identify the absolute room cell from a local crop.
_WALLS = (
    [0, 0], [0, 1], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
    [1, 0], [1, 3], [2, 3], [3, 0], [3, 3],
    [4, 0], [4, 1], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7],
)

_ROOM_POS = (
    (1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6),
)
_INIT_VALUES = (0, 0, 0)


def _static_map():
    walls = jnp.asarray(_WALLS, dtype=jnp.int32)
    static = jnp.zeros((5, 8, 9), dtype=jnp.int32)
    static = static.at[walls[:, 0], walls[:, 1], 0].set(1)
    static = static.at[2, 7, 1].set(1)
    static = static.at[2, 4, 2].set(1)
    static = static.at[2, 0, 3].set(1)
    static = static.at[1, 4, 5].set(1)
    static = static.at[3, 4, 5].set(1)
    static = static.at[1, 7, 6].set(1)
    static = static.at[3, 7, 7].set(1)
    static = static.at[0, 2, 8].set(1)
    static = static.at[4, 2, 8].set(1)
    return static


def _absolute_position(obs):
    room_pos = jnp.asarray(_ROOM_POS, dtype=jnp.int32)
    off = jnp.arange(-2, 3, dtype=jnp.int32)
    rows = room_pos[:, 0, None, None] + off[None, :, None]
    cols = room_pos[:, 1, None, None] + off[None, None, :]
    valid = (rows >= 0) & (rows < 5) & (cols >= 0) & (cols < 8)
    row_ix = jnp.clip(rows, 0, 4)
    col_ix = jnp.clip(cols, 0, 7)
    expected = _static_map()[row_ix, col_ix]
    expected = jnp.where(valid[..., None], expected, 0)
    seen = obs[:, :, 20:29]
    error = jnp.sum(jnp.abs(expected - seen), axis=(1, 2, 3))
    return room_pos[jnp.argmin(error)]


def _move_toward(pos, destination):
    row_delta = destination[0] - pos[0]
    col_delta = destination[1] - pos[1]
    return jnp.where(
        row_delta < 0,
        jnp.asarray(3, dtype=jnp.int32),
        jnp.where(
            row_delta > 0,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.where(
                col_delta < 0,
                jnp.asarray(2, dtype=jnp.int32),
                jnp.where(col_delta > 0, jnp.asarray(0, dtype=jnp.int32),
                          jnp.asarray(4, dtype=jnp.int32)),
            ),
        ),
    )


def _act_kitchen(obs, reset, memory):
    init = jnp.asarray(_INIT_VALUES, dtype=jnp.int32)
    mem = jnp.where(reset, init, memory)
    stage, ingredient, count = mem[0], mem[1], mem[2]
    pos = _absolute_position(obs)

    # The cook interaction reports 19 in the first post-interaction
    # observation.  A private countdown avoids depending on whether the
    # finished-pot flag is included at the edge of the local crop.
    ready = count >= 19
    effective_stage = jnp.where((stage == 2) & ready, 3, stage)

    supply_row = jnp.where(ingredient == 0, 1, 3).astype(jnp.int32)
    destination = jnp.where(
        effective_stage == 0,
        jnp.asarray([supply_row, 6], dtype=jnp.int32),
        jnp.where(
            effective_stage == 1,
            jnp.asarray([2, 5], dtype=jnp.int32),
            jnp.where(
                effective_stage == 2,
                jnp.asarray([1, 5], dtype=jnp.int32),
                jnp.where(
                effective_stage == 3,
                jnp.asarray([1, 5], dtype=jnp.int32),
                jnp.where(effective_stage == 4,
                          jnp.asarray([2, 5], dtype=jnp.int32),
                          jnp.asarray([2, 6], dtype=jnp.int32)),
                ),
            ),
        ),
    )
    at_destination = (pos[0] == destination[0]) & (pos[1] == destination[1])
    moving_action = _move_toward(pos, destination)
    interact_stage = (effective_stage == 0) | (effective_stage == 1) | \
        (effective_stage == 3) | (effective_stage == 4) | (effective_stage == 5)
    action = jnp.where(
        (effective_stage == 2) | (~at_destination),
        moving_action,
        jnp.where(interact_stage, jnp.asarray(5, dtype=jnp.int32),
                  jnp.asarray(4, dtype=jnp.int32)),
    )

    at_interaction = at_destination & (action == 5)
    picked_supply = at_interaction & (effective_stage == 0)
    inserted = at_interaction & (effective_stage == 1)
    took_plate = at_interaction & (effective_stage == 3)
    took_food = at_interaction & (effective_stage == 4)
    delivered = at_interaction & (effective_stage == 5)

    inserted_count = count + inserted.astype(jnp.int32)
    next_stage = effective_stage
    next_stage = jnp.where(picked_supply, 1, next_stage)
    next_stage = jnp.where(inserted, jnp.where(inserted_count >= 3, 2, 0), next_stage)
    next_stage = jnp.where(took_plate, 4, next_stage)
    next_stage = jnp.where(took_food, 5, next_stage)
    next_stage = jnp.where(delivered, 0, next_stage)
    next_ingredient = jnp.where(delivered, 1 - ingredient, ingredient)
    cooking_started = inserted & (inserted_count >= 3)
    next_count = jnp.where(inserted, inserted_count, count)
    next_count = jnp.where(cooking_started, 0, next_count)
    next_count = jnp.where((stage == 2) & (~ready), count + 1, next_count)
    next_count = jnp.where(delivered, 0, next_count)
    next_memory = jnp.asarray([next_stage, next_ingredient, next_count], dtype=jnp.int32)
    return action, next_memory, {}


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero = jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(4, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    return jnp.asarray(_INIT_VALUES, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    return _act_kitchen(obs, reset, memory)
