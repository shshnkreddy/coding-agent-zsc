"""Reactive two-room Overcooked policy.

The left player publishes the recipe with an otherwise-unused ingredient-2
token: upper boundary counter means ingredient 0, lower means ingredient 1.
The right player reads that persistent, locally observable signal and handles
the complete cooking and delivery loop.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'counter_signal_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


STAY = 4
INTERACT = 5


def _navigate(row, col, facing, target_row, target_col, target_facing):
    """Walk to a floor square, turn toward its adjacent object, interact."""
    row = jnp.asarray(row, dtype=jnp.int32)
    col = jnp.asarray(col, dtype=jnp.int32)
    target_row = jnp.asarray(target_row, dtype=jnp.int32)
    target_col = jnp.asarray(target_col, dtype=jnp.int32)
    target_facing = jnp.asarray(target_facing, dtype=jnp.int32)
    move = jnp.where(
        row < target_row,
        jnp.asarray(1, jnp.int32),
        jnp.where(
            row > target_row,
            jnp.asarray(3, jnp.int32),
            jnp.where(
                col < target_col,
                jnp.asarray(0, jnp.int32),
                jnp.where(col > target_col,
                          jnp.asarray(2, jnp.int32), STAY),
            ),
        ),
    )
    at_target = (row == target_row) & (col == target_col)
    face_action = jnp.where(
        target_facing == 0, jnp.asarray(3, jnp.int32),
        jnp.where(
            target_facing == 1, jnp.asarray(1, jnp.int32),
            jnp.where(target_facing == 2,
                      jnp.asarray(0, jnp.int32),
                      jnp.asarray(2, jnp.int32)),
        ),
    )
    return jnp.where(
        at_target,
        jnp.where(facing == target_facing, INTERACT, face_action),
        move,
    )


def _pose_from_landmark(obs, channel, landmark_row, landmark_col):
    """Infer global pose from a static landmark that is always in view."""
    flat = jnp.argmax(obs[:, :, channel].reshape(-1))
    local_row = flat // 5
    local_col = flat % 5
    row = jnp.asarray(landmark_row + 2, jnp.int32) - local_row
    col = jnp.asarray(landmark_col + 2, jnp.int32) - local_col
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    return row, col, facing


def _counter_token_rows(obs, row, col):
    """Return ingredient-2 marker presence on boundary rows 1, 2, and 3."""
    local_rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    local_cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    global_rows = row + local_rows - 2
    global_cols = col + local_cols - 2
    token = obs[:, :, 33] > 0
    on_boundary = global_cols == 3
    t1 = jnp.any(token & on_boundary & (global_rows == 1))
    t2 = jnp.any(token & on_boundary & (global_rows == 2))
    t3 = jnp.any(token & on_boundary & (global_rows == 3))
    return t1, t2, t3


def init_agent_0():
    return jnp.asarray([0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_0()
    memory = jnp.where(reset, initial, memory)
    row, col, facing = _pose_from_landmark(obs, 23, 2, 0)
    recipe0 = jnp.sum(obs[:, :, 36])
    recipe1 = jnp.sum(obs[:, :, 37])
    recipe = jnp.where(recipe1 > recipe0,
                       jnp.asarray(1, jnp.int32),
                       jnp.asarray(0, jnp.int32))
    desired_row = jnp.asarray(1, jnp.int32) + 2 * recipe
    t1, t2, t3 = _counter_token_rows(obs, row, col)
    desired_present = jnp.where(recipe == 0, t1, t3)
    any_present = t1 | t2 | t3
    held2 = obs[2, 2, 9] > 0
    token_row = jnp.where(t2, jnp.asarray(2, jnp.int32),
                          jnp.where(t1, jnp.asarray(1, jnp.int32),
                                    jnp.asarray(3, jnp.int32)))
    pick_token = _navigate(row, col, facing, token_row, 2, 2)
    place_token = _navigate(row, col, facing, desired_row, 2, 2)
    missing = jnp.where(
        desired_present | any_present | held2,
        jnp.asarray(0, jnp.int32),
        jnp.minimum(memory[0] + 1, jnp.asarray(20, jnp.int32)),
    )
    get_source = _navigate(
        row, col, facing, desired_row, 2,
        jnp.where(recipe == 0,
                  jnp.asarray(0, jnp.int32),
                  jnp.asarray(1, jnp.int32)),
    )
    # Stay beside the active token.  On a recipe change the very first action
    # can therefore remove the now-stale signal, making invalidation observable
    # to the cook before it commits to the next batch.
    park_move = jnp.where(
        row < desired_row, jnp.asarray(1, jnp.int32),
        jnp.where(
            row > desired_row, jnp.asarray(3, jnp.int32),
            jnp.where(col < 2, jnp.asarray(0, jnp.int32),
                      jnp.where(col > 2, jnp.asarray(2, jnp.int32), STAY)),
        ),
    )
    action = jnp.where(
        held2,
        place_token,
        jnp.where(
            desired_present,
            park_move,
            jnp.where(any_present,
                      pick_token,
                      jnp.where(missing >= 1, get_source, STAY)),
        ),
    ).astype(jnp.int32)
    next_memory = jnp.asarray([missing, memory[1] + 1], dtype=jnp.int32)
    trace = {
        'recipe': recipe,
        'phase': jnp.where(desired_present, 0,
                           jnp.where(held2, 2,
                                     jnp.where(any_present, 1, 3))).astype(jnp.int32),
        'row': row,
        'col': col,
    }
    return action, next_memory, trace


def init_agent_1():
    return jnp.asarray([-1, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_1()
    memory = jnp.where(reset, initial, memory)
    row, col, facing = _pose_from_landmark(obs, 22, 2, 4)
    t1, _, t3 = _counter_token_rows(obs, row, col)
    signal_valid = t1 ^ t3
    signal_recipe = jnp.where(t3,
                              jnp.asarray(1, jnp.int32),
                              jnp.asarray(0, jnp.int32))
    cooled = memory[1] == 0
    can_latch = (memory[0] < 0) & cooled & signal_valid
    fallback_start = ((memory[0] < 0) & cooled & (~signal_valid)
                      & (memory[3] >= 24))
    syncing = (memory[1] > 0) & (memory[0] >= 0)
    old_marker_present = jnp.where(memory[0] == 0, t1, t3)
    # At column 5 every boundary marker is visible.  If the old persistent
    # marker has vanished after our delivery, the binary recipe necessarily
    # flipped; collection can begin before physical relocation completes.
    marker_invalidated = syncing & (col == 5) & (~old_marker_present)
    inferred_recipe = jnp.asarray(1, jnp.int32) - memory[0]
    recipe = jnp.where(
        marker_invalidated,
        inferred_recipe,
        jnp.where(can_latch, signal_recipe,
                  jnp.where(fallback_start,
                            jnp.asarray(0, jnp.int32), memory[0])),
    )
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held0 = obs[2, 2, 7] > 0
    held1 = obs[2, 2, 8] > 0
    held_raw = held0 | held1
    local_rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    local_cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    global_rows = row + local_rows - 2
    global_cols = col + local_cols - 2
    pot_mask = (global_rows == 2) & (global_cols == 4)
    pot0 = jnp.sum(jnp.where(pot_mask, obs[:, :, 31], 0))
    pot1 = jnp.sum(jnp.where(pot_mask, obs[:, :, 32], 0))
    pot_total = pot0 + pot1 + jnp.sum(jnp.where(pot_mask, obs[:, :, 33], 0))
    pot_cooked = jnp.any(pot_mask & (obs[:, :, 30] > 0))
    pot_timer = jnp.sum(jnp.where(pot_mask, obs[:, :, 39], 0))
    late_marker = (cooled & (~syncing) & signal_valid & (col == 5)
                   & (pot_total == 0) & (~held_raw) & (~held_plate)
                   & (~held_cooked))
    recipe = jnp.where(late_marker, signal_recipe, recipe)
    at_goal_pose = (row == 2) & (col == 6) & (facing == 2)
    delivering = held_cooked & at_goal_pose
    go_goal = _navigate(row, col, facing, 2, 6, 2)
    go_pot = _navigate(row, col, facing, 2, 5, 3)
    # Use the cook timer to get the plate carrier into pickup position.  Do
    # not repeatedly interact before the pot is ready.
    stage_at_pot = jnp.where(go_pot == INTERACT, STAY, go_pot)
    supply_row = jnp.asarray(1, jnp.int32) + 2 * jnp.maximum(recipe, 0)
    go_supply = _navigate(row, col, facing, supply_row, 6, 2)
    plate_row = jnp.where(row <= 2,
                          jnp.asarray(1, jnp.int32),
                          jnp.asarray(3, jnp.int32))
    go_plate = _navigate(row, col, facing, plate_row, 5, 3)
    waiting_for_recipe = recipe < 0
    cooldown = memory[1] > 0
    sync_wait = cooldown & (~marker_invalidated) & (memory[1] > 1)
    return_to_reader = jnp.where(
        row < 2, jnp.asarray(1, jnp.int32),
        jnp.where(row > 2, jnp.asarray(3, jnp.int32),
                  jnp.where(col > 5, jnp.asarray(2, jnp.int32), STAY)),
    )
    empty_action = jnp.where(
        pot_cooked,
        go_plate,
        jnp.where(pot_total < 3, go_supply, go_plate),
    )
    work_action = jnp.where(
        held_cooked,
        go_goal,
        jnp.where(
            held_plate,
            jnp.where(pot_cooked, go_pot, stage_at_pot),
            jnp.where(held_raw, go_pot, empty_action),
        ),
    )
    action = jnp.where(
        sync_wait,
        return_to_reader,
        jnp.where(waiting_for_recipe,
                  return_to_reader,
                  work_action),
    ).astype(jnp.int32)
    next_recipe = recipe
    next_cooldown = jnp.where(
        delivering,
        jnp.asarray(4, jnp.int32),
        jnp.where(marker_invalidated,
                  jnp.asarray(0, jnp.int32),
                  jnp.where(memory[1] > 0, memory[1] - 1,
                            jnp.asarray(0, jnp.int32))),
    )
    next_memory = jnp.asarray([
        next_recipe,
        next_cooldown,
        memory[2] + delivering.astype(jnp.int32),
        memory[3] + 1,
    ], dtype=jnp.int32)
    phase = jnp.where(
        sync_wait | waiting_for_recipe, 0,
        jnp.where(held_cooked, 5,
                  jnp.where(pot_cooked, 4,
                            jnp.where(pot_timer > 0, 3,
                                      jnp.where(held_raw, 2, 1)))),
    ).astype(jnp.int32)
    trace = {
        'recipe': recipe,
        'phase': phase,
        'pot_total': pot_total.astype(jnp.int32),
        'deliveries': memory[2],
        'row': row,
        'col': col,
    }
    return action, next_memory, trace
