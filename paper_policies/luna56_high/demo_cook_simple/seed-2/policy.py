"""A map-grounded, convention-robust policy for demo_cook_simple.

The two rooms provide a natural observable role convention. The left seat
handles ingredients and the right seat handles plates and delivery. Global
coordinates are reconstructed only from the static map crop in the local
observation; no hidden environment state is used.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'map_grounded_conveyor',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}


# Keep these as literals because the source validator rejects module-level
# executable array construction. They become JAX arrays inside _position.
_STATIC_MAP = [[[1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]]
_FLOOR = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (1, 8), (1, 9), (3, 8), (3, 9))


def _position(obs, fallback_r, fallback_c):
    """Recover global position by matching the visible static map crop."""
    static_map = jnp.asarray(_STATIC_MAP, dtype=jnp.int32)
    floor = jnp.asarray(_FLOOR, dtype=jnp.int32)
    padded_static = jnp.pad(static_map, ((2, 2), (2, 2), (0, 0)))
    static_patches = jnp.stack([
        padded_static[r:r + 5, c:c + 5] for r, c in _FLOOR
    ])
    visible = obs[:, :, 20:29]
    matches = jnp.all(static_patches == visible[None, :, :, :], axis=(1, 2, 3))
    found = jnp.any(matches)
    exact_index = jnp.argmax(matches)
    errors = jnp.sum(jnp.abs(static_patches - visible[None, :, :, :]), axis=(1, 2, 3))
    nearest_index = jnp.argmin(errors)
    index = jnp.where(found, exact_index, nearest_index)
    candidate = floor[index]
    fallback = jnp.asarray([fallback_r, fallback_c], dtype=jnp.int32)
    return jnp.where(found, candidate, fallback)


def _move_toward(position, target):
    """Move vertically first, then horizontally, on either open room."""
    row, col = position[0], position[1]
    target_row, target_col = target[0], target[1]
    return jnp.where(
        row < target_row,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.where(
            row > target_row,
            jnp.asarray(3, dtype=jnp.int32),
            jnp.where(
                col < target_col,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.where(col > target_col, jnp.asarray(2, dtype=jnp.int32),
                          jnp.asarray(4, dtype=jnp.int32)),
            ),
        ),
    )


def _right_position(obs):
    """Use the always-visible pot to locate any right-room cell."""
    flat_pot = jnp.argmax(obs[:, :, 22])
    pot_row = flat_pot // 5
    pot_col = flat_pot % 5
    return jnp.asarray([4 - pot_row, 9 - pot_col], dtype=jnp.int32)


def _at(position, target):
    return jnp.all(position == target)


def _face_or_interact(obs, position, target, facing):
    """At a service cell, turn by walking into the object, then interact."""
    at_target = _at(position, target)
    current_facing = jnp.argmax(obs[2, 2, 1:5])
    # Facing IDs are UP=0, DOWN=1, RIGHT=2, LEFT=3; movement IDs differ.
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[facing]
    action = jnp.where(current_facing == facing, 5, face_action)
    return jnp.where(at_target, action, _move_toward(position, target)), at_target & (current_facing == facing)


def init_agent_0():
    # phase, loaded-count, ingredient-kind, last-row, last-column, wait-stage
    return jnp.zeros((6,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    phase = jnp.where(reset, 0, memory[0])
    count = jnp.where(reset, 0, memory[1])
    kind = jnp.where(reset, 0, memory[2])
    wait_stage = jnp.where(reset, 0, memory[5])
    position = _position(obs, memory[3], memory[4])

    read_cell = jnp.asarray([1, 5], dtype=jnp.int32)
    source_cell = jnp.where(kind == 0, jnp.asarray([1, 1]), jnp.asarray([3, 1]))
    pot_left = jnp.asarray([2, 6], dtype=jnp.int32)

    at_read = _at(position, read_cell)
    at_source = _at(position, source_cell)
    at_pot = _at(position, pot_left)
    recipe_at_read = obs[1, 2, 36:39]
    recipe_at_pot = obs[0, 1, 36:39]
    sensed_kind_read = jnp.where(recipe_at_read[1] > recipe_at_read[0], 1, 0)
    sensed_kind_pot = jnp.where(recipe_at_pot[1] > recipe_at_pot[0], 1, 0)

    source_action, source_ready = _face_or_interact(obs, position, source_cell, 3)
    pot_action, pot_ready = _face_or_interact(obs, position, pot_left, 2)

    # At the pot, the partner briefly visibly holds the cooked dish after
    # scooping. Wait through that interval so the next batch uses the recipe
    # selected by the subsequent delivery, not the previous one.
    pot_item = obs[2, 3, 29:34]
    pot_empty = jnp.all(pot_item == 0)
    partner_cooked_visible = jnp.any(obs[:, :, 16] > 0)
    delivery_finished = at_pot & pot_empty & (~partner_cooked_visible) & (wait_stage >= 2)

    phase0_action = jnp.where(at_read, 4, _move_toward(position, read_cell))
    action = jnp.where(
        phase == 0,
        phase0_action,
        jnp.where(
            phase == 1,
            source_action,
            jnp.where(phase == 2, pot_action,
                      jnp.where(phase == 3,
                                jnp.where(at_pot, 4, _move_toward(position, pot_left)),
                                jnp.asarray(4, dtype=jnp.int32))),
        ),
    )

    next_phase = phase
    next_count = count
    next_kind = kind
    next_wait_stage = wait_stage
    next_phase = jnp.where((phase == 0) & at_read, 1, next_phase)
    next_kind = jnp.where((phase == 0) & at_read, sensed_kind_read, next_kind)
    next_phase = jnp.where((phase == 1) & source_ready, 2, next_phase)
    loaded_now = (phase == 2) & pot_ready
    next_count = jnp.where(loaded_now, count + 1, next_count)
    next_phase = jnp.where(loaded_now & (count < 2), 1, next_phase)
    next_phase = jnp.where(loaded_now & (count >= 2), 3, next_phase)
    next_phase = jnp.where((phase == 3) & delivery_finished, 1, next_phase)
    next_count = jnp.where((phase == 3) & delivery_finished, 0, next_count)
    next_kind = jnp.where((phase == 3) & delivery_finished, sensed_kind_pot, next_kind)
    next_wait_stage = jnp.where(phase != 3, 0, next_wait_stage)
    next_wait_stage = jnp.where((phase == 3) & partner_cooked_visible, 1, next_wait_stage)
    next_wait_stage = jnp.where(
        (phase == 3) & (~partner_cooked_visible) & (wait_stage == 1),
        2,
        next_wait_stage,
    )
    next_wait_stage = jnp.where((phase == 3) & delivery_finished, 0, next_wait_stage)

    next_memory = jnp.asarray([
        next_phase, next_count, next_kind, position[0], position[1], next_wait_stage,
    ], dtype=jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_memory, {
        'position': position,
        'phase': phase,
        'kind': kind,
        'count': count,
        'recipe_at_read': recipe_at_read,
        'recipe_at_pot': recipe_at_pot,
        'pot_item': pot_item,
        'delivery_finished': delivery_finished,
        'held': obs[2, 2, 5:10],
    }


def init_agent_1():
    # phase, last-row, last-column
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    phase = jnp.where(reset, 0, memory[0])
    position = _right_position(obs)

    plate_cell = jnp.asarray([1, 9], dtype=jnp.int32)
    pot_right = jnp.asarray([2, 8], dtype=jnp.int32)
    delivery_cell = jnp.asarray([2, 9], dtype=jnp.int32)

    plate_action, plate_ready = _face_or_interact(obs, position, plate_cell, 2)
    pot_action, pot_ready = _face_or_interact(obs, position, pot_right, 3)
    delivery_action, delivery_ready = _face_or_interact(obs, position, delivery_cell, 2)

    # At (2,8), the pot is local cell (2,1). A cooked flag distinguishes a
    # ready pot from both an empty pot and a pot still counting down.
    pot_cooked = obs[2, 1, 30] > 0
    scoop_now = (phase == 1) & pot_ready & pot_cooked

    action = jnp.where(
        phase == 0,
        plate_action,
        jnp.where(
            phase == 1,
            jnp.where(scoop_now, pot_action, jnp.where(_at(position, pot_right), 4, pot_action)),
            delivery_action,
        ),
    )

    next_phase = phase
    next_phase = jnp.where((phase == 0) & plate_ready, 1, next_phase)
    next_phase = jnp.where(scoop_now, 2, next_phase)
    next_phase = jnp.where((phase == 2) & delivery_ready, 0, next_phase)
    next_memory = jnp.asarray([next_phase, position[0], position[1]], dtype=jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_memory, {
        'position': position,
        'phase': phase,
        'pot_contents': obs[2, 1, 31:34],
        'held': obs[2, 2, 5:10],
    }
