"""A small map-grounded controller for the grounded_coord_ring layout."""

import jax.numpy as jnp
from jax import lax

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_inner_cook",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}

# 0=floor, then the nine static observation channels in order.
_CODES = (
    (1, 1, 1, 9, 4, 9, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 1, 1, 5, 1, 1, 0, 1),
    (9, 0, 7, 0, 0, 0, 6, 0, 9),
    (4, 0, 2, 0, 3, 0, 2, 0, 4),
    (9, 0, 8, 0, 0, 0, 6, 0, 9),
    (1, 0, 1, 1, 5, 1, 1, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 1, 1, 9, 4, 9, 1, 1, 1),
)


def _locate(obs, memory, reset):
    codes = jnp.asarray(_CODES, dtype=jnp.int32)
    static = (codes[..., None] == jnp.arange(1, 10, dtype=jnp.int32)).astype(jnp.int32)
    coords = jnp.stack(jnp.meshgrid(jnp.arange(9), jnp.arange(9), indexing="ij"), axis=-1).reshape(81, 2)
    dr = jnp.arange(5, dtype=jnp.int32) - 2
    patch_r = coords[:, 0, None, None] + dr[None, :, None]
    patch_c = coords[:, 1, None, None] + dr[None, None, :]
    in_r = (patch_r >= 0) & (patch_r < 9)
    in_c = (patch_c >= 0) & (patch_c < 9)
    static_patches = static[jnp.clip(patch_r, 0, 8), jnp.clip(patch_c, 0, 8)] * (in_r & in_c)[..., None]
    observed = obs[:, :, 20:29]
    mismatch = jnp.sum(jnp.abs(static_patches - observed[None, ...]), axis=(1, 2, 3))
    previous = memory[2:4]
    delta = jnp.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=jnp.int32)
    predicted = jnp.clip(previous + delta[jnp.clip(memory[4], 0, 4)], 0, 8)
    distance = jnp.sum(jnp.abs(coords - predicted[None, :]), axis=1)
    tie = jnp.where(reset, jnp.arange(81, dtype=jnp.int32), distance)
    index = jnp.argmin(mismatch * 1000 + tie)
    return coords[index, 0], coords[index, 1]


def _distance_field(target_r, target_c):
    codes = jnp.asarray(_CODES, dtype=jnp.int32)
    floor = codes == 0
    rows, cols = jnp.meshgrid(jnp.arange(9), jnp.arange(9), indexing="ij")
    distance = jnp.where((rows == target_r) & (cols == target_c), 0, 999)
    distance = jnp.where(floor, distance, 1000)

    def relax(_, current):
        up = jnp.concatenate((current[1:, :], jnp.full((1, 9), 1000, dtype=jnp.int32)), axis=0)
        down = jnp.concatenate((jnp.full((1, 9), 1000, dtype=jnp.int32), current[:-1, :]), axis=0)
        left = jnp.concatenate((current[:, 1:], jnp.full((9, 1), 1000, dtype=jnp.int32)), axis=1)
        right = jnp.concatenate((jnp.full((9, 1), 1000, dtype=jnp.int32), current[:, :-1]), axis=1)
        relaxed = jnp.minimum(current, 1 + jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right)))
        return jnp.where(floor, relaxed, 1000)

    return lax.fori_loop(0, 81, relax, distance)


def _move_or_stay(row, col, target_r, target_c):
    floor = jnp.asarray(_CODES, dtype=jnp.int32) == 0
    field = _distance_field(target_r, target_c)
    rr = jnp.clip(row, 0, 8)
    cc = jnp.clip(col, 0, 8)
    invalid = jnp.asarray(1000, dtype=jnp.int32)
    right = jnp.where((cc < 8) & floor[rr, cc + 1], field[rr, cc + 1], invalid)
    down = jnp.where((rr < 8) & floor[rr + 1, cc], field[rr + 1, cc], invalid)
    left = jnp.where((cc > 0) & floor[rr, cc - 1], field[rr, cc - 1], invalid)
    up = jnp.where((rr > 0) & floor[rr - 1, cc], field[rr - 1, cc], invalid)
    choices = jnp.stack((right, down, left, up, field[rr, cc]))
    movement = jnp.argmin(choices).astype(jnp.int32)
    return jnp.where((row == target_r) & (col == target_c), jnp.asarray(4, dtype=jnp.int32), movement)


def _at_action(row, col, target_r, target_c, interact):
    move = _move_or_stay(row, col, target_r, target_c)
    return jnp.where((row == target_r) & (col == target_c),
                     jnp.asarray(5 if interact else 4, dtype=jnp.int32), move)


def _interact_action(obs, row, col, target_r, target_c, desired_facing):
    """Reach a standing cell, turn toward its object, then interact."""
    move = _move_or_stay(row, col, target_r, target_c)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    turn_actions = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)  # up, down, right, left
    at = (row == target_r) & (col == target_c)
    oriented = jnp.where(facing == desired_facing, jnp.asarray(5, dtype=jnp.int32), turn_actions[desired_facing])
    return jnp.where(at, oriented, move)


def init_agent_0():
    return jnp.zeros((8,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero = jnp.zeros((8,), dtype=jnp.int32)
    return jnp.asarray(4, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    # [phase, target, estimated_row, estimated_col, previous_action, ...]
    return jnp.asarray([0, 0, 4, 4, 4, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    row, col = _locate(obs, memory, reset)
    phase = jnp.where(reset, 0, memory[0])

    display = obs[:, :, 36:39]
    seen_recipe = jnp.any(display > 0)
    shown_target = jnp.where(obs[:, :, 37].max() > 0, 1, 0).astype(jnp.int32)
    target = jnp.where(seen_recipe, shown_target, jnp.where(reset, 0, memory[1]))

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw0 = obs[2, 2, 7] > 0
    held_raw1 = obs[2, 2, 8] > 0
    held_raw = jnp.where(target == 0, held_raw0, held_raw1)

    pot_r = obs[:, :, 31:34]
    pot_cooked = obs[:, :, 30].max() > 0
    pot_timer = obs[:, :, 39].max()
    pot_raw = jnp.where(target == 0, pot_r[:, :, 0].max(), pot_r[:, :, 1].max())
    pot_cooking = pot_timer > 0
    pot_full_idle = (pot_raw >= 3) & ~pot_cooked & ~pot_cooking

    source_r = jnp.where(target == 0, 3, 5)
    insert_r = jnp.where(target == 0, 3, 5)
    button_r = jnp.asarray(3, dtype=jnp.int32)
    button_c = jnp.asarray(4, dtype=jnp.int32)
    plate_r = jnp.asarray(3, dtype=jnp.int32)
    plate_c = jnp.asarray(5, dtype=jnp.int32)
    deliver_r = jnp.asarray(4, dtype=jnp.int32)
    deliver_c = jnp.asarray(5, dtype=jnp.int32)

    sense_action = _interact_action(obs, row, col, button_r, button_c, 0)
    sense_done = (row == button_r) & (col == button_c)
    deliver_action = _interact_action(obs, row, col, deliver_r, deliver_c, 2)
    collect_action = _interact_action(obs, row, col, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(4, dtype=jnp.int32), 1)
    plate_action = _interact_action(obs, row, col, plate_r, plate_c, 2)
    insert_action = _interact_action(obs, row, col, insert_r, jnp.asarray(4, dtype=jnp.int32),
                                     jnp.where(target == 0, 1, 0))
    source_action = _interact_action(obs, row, col, source_r, jnp.asarray(3, dtype=jnp.int32), 3)
    wait_at_pot = _at_action(row, col, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(4, dtype=jnp.int32), False)

    working_action = jnp.where(held_cooked, deliver_action, source_action)
    working_action = jnp.where(~held_cooked & held_raw, insert_action, working_action)
    working_action = jnp.where(~held_cooked & ~held_raw & pot_cooked & held_plate, collect_action, working_action)
    working_action = jnp.where(~held_cooked & ~held_raw & pot_cooked & ~held_plate, plate_action, working_action)
    working_action = jnp.where(~held_cooked & ~held_raw & ~pot_cooked & pot_cooking,
                               jnp.where(held_plate, wait_at_pot, plate_action), working_action)
    working_action = jnp.where(~held_cooked & ~held_raw & ~pot_cooked & ~pot_cooking & pot_full_idle,
                               wait_at_pot, working_action)

    action = jnp.where(phase == 0, sense_action, working_action)
    next_phase = jnp.where((phase == 0) & sense_done & (sense_action == 5), 1, phase)
    delivered = (phase == 1) & held_cooked & (row == deliver_r) & (col == deliver_c) & (action == 5)
    next_phase = jnp.where(delivered, 0, next_phase)
    next_memory = jnp.stack((next_phase, target, row, col, action,
                             jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)))
    trace = {
        "row": row.astype(jnp.int32),
        "col": col.astype(jnp.int32),
        "phase": phase.astype(jnp.int32),
        "target": target.astype(jnp.int32),
        "pot_timer": pot_timer.astype(jnp.int32),
        "sense_action": sense_action.astype(jnp.int32),
        "working_action": working_action.astype(jnp.int32),
        "sense_distance": _distance_field(button_r, button_c)[row, col].astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
