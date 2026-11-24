"""Reactive fixed-layout cooperative policy for demo_cook_wide."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'reactive_cooperative_cycle',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}


# Static channels are wall, goal, pot, recipe, button, plate pile, and the
# four ingredient supplies.  -1 denotes floor. Matching a complete local
# static patch gives a position estimate without using hidden state.
_CODES = (
    (0, 0, 0, 0, 5, 1, 5, 0, 0, 0, 0),
    (0, 0, 0, 6, -1, -1, -1, 7, 0, 0, 0),
    (0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
    (0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0),
    (6, -1, -1, 0, 9, 3, 9, 0, -1, -1, 6),
    (0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0),
)


def _position(obs):
    codes = jnp.asarray(_CODES, dtype=jnp.int32)
    safe_codes = jnp.maximum(codes, 0)
    static = jnp.eye(10, dtype=jnp.int32)[safe_codes]
    static = jnp.where((codes < 0)[..., None], 0, static)
    static_pad = jnp.pad(static, ((2, 2), (2, 2), (0, 0)))
    rows, cols = jnp.indices((6, 11), dtype=jnp.int32)
    off = jnp.arange(5, dtype=jnp.int32)
    patch_rows = rows.reshape(-1, 1, 1) + off.reshape(1, 5, 1)
    patch_cols = cols.reshape(-1, 1, 1) + off.reshape(1, 1, 5)
    static_patches = static_pad[patch_rows, patch_cols]
    actual = obs[:, :, 22:32]
    error = jnp.sum(jnp.abs(static_patches - actual[None, ...]), axis=(1, 2, 3))
    index = jnp.argmin(error)
    return rows.reshape(-1)[index], cols.reshape(-1)[index]


def _move_direct(row, col, target_row, target_col):
    return jnp.where(
        col < target_col,
        jnp.asarray(0, jnp.int32),
        jnp.where(
            col > target_col,
            jnp.asarray(2, jnp.int32),
            jnp.where(
                row < target_row,
                jnp.asarray(1, jnp.int32),
                jnp.where(row > target_row, jnp.asarray(3, jnp.int32), jnp.asarray(4, jnp.int32)),
            ),
        ),
    )


def _to_center(row, col):
    """Navigate lower-room spawn points to (3,5), avoiding the R cell."""
    return jnp.where(
        row > 3,
        jnp.asarray(3, jnp.int32),
        jnp.where(
            col < 5,
            jnp.asarray(0, jnp.int32),
            jnp.where(col > 5, jnp.asarray(2, jnp.int32), jnp.asarray(4, jnp.int32)),
        ),
    )


def _to_left_supply(row, col):
    """Reach the common left-side stand (4,1) from all lower spawns."""
    return jnp.where(
        row == 4,
        jnp.where(col > 1, jnp.asarray(2, jnp.int32), jnp.where(col < 1, jnp.asarray(0, jnp.int32), jnp.asarray(4, jnp.int32))),
        jnp.where(col > 1, jnp.asarray(2, jnp.int32), jnp.where(col < 1, jnp.asarray(0, jnp.int32), jnp.asarray(1, jnp.int32))),
    )


def _face_and_interact(obs, direction):
    facing = jnp.argmax(obs[2, 2, 1:5])
    turn_action = jnp.where(
        direction == 0,
        jnp.asarray(3, jnp.int32),
        jnp.where(direction == 1, jnp.asarray(1, jnp.int32), jnp.where(direction == 2, jnp.asarray(0, jnp.int32), jnp.asarray(2, jnp.int32))),
    )
    return jnp.where(facing == direction, jnp.asarray(5, jnp.int32), turn_action)


def init_agent_0():
    # phase: 0 acquire plate, 1 wait/plate, 2 deliver
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    row, col = _position(obs)
    phase = jnp.where(reset, jnp.asarray(0, jnp.int32), memory[0])
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0

    phase = jnp.where((phase == 0) & held_plate, 1, phase)
    phase = jnp.where((phase == 1) & held_cooked, 2, phase)
    phase = jnp.where((phase == 2) & (~held_cooked) & (~held_plate), 0, phase)

    at_plate_stand = (row == 1) & (col == 4)
    acquire = jnp.where(
        held_plate,
        jnp.asarray(4, jnp.int32),
        jnp.where(at_plate_stand, _face_and_interact(obs, 0), _move_direct(row, col, 1, 4)),
    )

    pot_cooked = obs[3, 2, 33] > 0
    at_pot_stand = (row == 1) & (col == 5)
    plate = jnp.where(
        held_cooked,
        jnp.asarray(4, jnp.int32),
        jnp.where(
            at_pot_stand & pot_cooked,
            _face_and_interact(obs, 1),
            jnp.where(at_pot_stand, jnp.asarray(4, jnp.int32), _move_direct(row, col, 1, 5)),
        ),
    )

    deliver = jnp.where(
        held_cooked,
        jnp.where((row == 1) & (col == 5), _face_and_interact(obs, 0), _move_direct(row, col, 1, 5)),
        jnp.asarray(4, jnp.int32),
    )
    action = jnp.where(phase == 0, acquire, jnp.where(phase == 1, plate, deliver))
    return action, phase[None], {}


def init_agent_1():
    # [phase, ingredients deposited, target ingredient, pot-ready-seen,
    #  partner-plated-seen].
    return jnp.zeros((5,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    row, col = _position(obs)
    phase = jnp.where(reset, jnp.asarray(0, jnp.int32), memory[0])
    count = jnp.where(reset, jnp.asarray(0, jnp.int32), memory[1])
    ingredient = jnp.where(reset, jnp.asarray(0, jnp.int32), memory[2])
    ready_seen = jnp.where(reset, jnp.asarray(0, jnp.int32), memory[3])
    plated_seen = jnp.where(reset, jnp.asarray(0, jnp.int32), memory[4])

    raw = jnp.sum(obs[2, 2, 7:11]) > 0
    at_center = (row == 3) & (col == 5)
    # The always-on R display can be read at any local cell in view.  This
    # saves only startup travel; after a delivery the agent is already at the
    # center stand.
    recipe0 = jnp.sum(obs[:, :, 40]) > 0
    recipe1 = jnp.sum(obs[:, :, 41]) > 0
    recipe_visible = recipe0 | recipe1
    partner_plated = jnp.sum(obs[:, :, 17]) > 0
    pot_cooked = obs[1, 2, 33] > 0
    pot_empty = jnp.sum(obs[1, 2, 32:38]) == 0

    read_now = (phase == 0) & recipe_visible
    ingredient = jnp.where(read_now, jnp.where(recipe1, 1, 0), ingredient)
    phase = jnp.where(read_now, 1, phase)

    phase = jnp.where((phase == 1) & raw, 2, phase)
    deposited = (phase == 2) & (~raw) & at_center
    next_count = count + deposited.astype(jnp.int32)
    phase = jnp.where(deposited & (next_count < 3), 1, phase)
    phase = jnp.where(deposited & (next_count >= 3), 3, phase)
    count = jnp.where(deposited, next_count, count)

    ready_seen = jnp.where((phase == 3) & pot_cooked, 1, ready_seen)
    plated_seen = jnp.where((phase == 3) & partner_plated, 1, plated_seen)
    cycle_done = (phase == 3) & (ready_seen > 0) & (plated_seen > 0) & pot_empty & (~partner_plated)
    phase = jnp.where(cycle_done, 0, phase)
    count = jnp.where(cycle_done, 0, count)
    ready_seen = jnp.where(cycle_done, 0, ready_seen)
    plated_seen = jnp.where(cycle_done, 0, plated_seen)

    to_center = _to_center(row, col)
    to_supply = _to_left_supply(row, col)
    read_action = jnp.where(recipe_visible, jnp.asarray(4, jnp.int32), to_center)
    collect_direction = jnp.where(ingredient == 1, jnp.asarray(1, jnp.int32), jnp.asarray(3, jnp.int32))
    collect = jnp.where(
        raw,
        jnp.asarray(4, jnp.int32),
        jnp.where((row == 4) & (col == 1), _face_and_interact(obs, collect_direction), to_supply),
    )
    deposit = jnp.where(
        raw,
        jnp.where(at_center, _face_and_interact(obs, 0), to_center),
        jnp.asarray(4, jnp.int32),
    )
    wait = jnp.asarray(4, jnp.int32)
    action = jnp.where(phase == 0, read_action, jnp.where(phase == 1, collect, jnp.where(phase == 2, deposit, wait)))
    next_memory = jnp.stack([phase, count, ingredient, ready_seen, plated_seen])
    return action, next_memory, {}
