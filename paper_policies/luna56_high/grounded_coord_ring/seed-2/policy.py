"""A local-observation cook loop for the grounded coordination ring."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'central_self_contained_loop',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}


def _navigate_or_interact(obs, mask, fallback):
    """Move toward a visible cardinally adjacent object, then interact."""
    flat = mask.reshape(-1)
    index = jnp.argmax(flat)
    row = index // 5
    col = index - row * 5
    dr = row - 2
    dc = col - 2
    present = jnp.any(mask)
    cardinally_adjacent = (jnp.abs(dr) + jnp.abs(dc)) == 1

    # Movement actions: RIGHT=0, DOWN=1, LEFT=2, UP=3.
    horizontal = jnp.where(dc > 0, 0, 2)
    vertical = jnp.where(dr > 0, 1, 3)
    blocked = jnp.any(obs[:, :, 20:29] > 0, axis=2)
    h_col = jnp.clip(2 + jnp.where(dc > 0, 1, -1), 0, 4)
    v_row = jnp.clip(2 + jnp.where(dr > 0, 1, -1), 0, 4)
    horizontal_clear = (~blocked[2, h_col]) & (jnp.abs(dc) > 0)
    vertical_clear = (~blocked[v_row, 2]) & (jnp.abs(dr) > 0)
    diagonal_move = jnp.where(
        (jnp.abs(dc) > 0) & horizontal_clear,
        horizontal,
        vertical,
    )
    move = jnp.where(
        cardinally_adjacent,
        jnp.where(jnp.abs(dc) > 0, horizontal, vertical),
        jnp.where(
            jnp.abs(dc) > 1,
            horizontal,
            jnp.where(jnp.abs(dr) > 1, vertical, diagonal_move),
        ),
    )
    wanted_facing = jnp.where(
        dr < 0, 0, jnp.where(dr > 0, 1, jnp.where(dc > 0, 2, 3))
    )
    facing = jnp.argmax(obs[2, 2, 1:5])
    facing_target = facing == wanted_facing
    interact = present & cardinally_adjacent & facing_target
    return jnp.where(interact, 5, jnp.where(present, move, fallback)).astype(jnp.int32)


def _init_memory():
    # phase, target ingredient (0/1), ingredients already inserted
    return jnp.zeros((3,), dtype=jnp.int32)


def init_agent_0():
    return _init_memory()


def init_agent_1():
    return _init_memory()


def _act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _init_memory(), memory)
    phase = memory[0]
    target = memory[1]
    inserted = memory[2]

    # The pot is visible throughout the compact central room.  The outer
    # ring seat deliberately stays still, so an independently authored
    # partner is free to use that room without a private convention.
    central = jnp.any(obs[:, :, 22] > 0)

    button_mask = obs[:, :, 24] > 0
    supply0_mask = obs[:, :, 26] > 0
    supply1_mask = obs[:, :, 27] > 0
    supply_mask = jnp.where(target == 0, supply0_mask, supply1_mask)
    pot_mask = obs[:, :, 22] > 0
    plate_mask = obs[:, :, 25] > 0
    goal_mask = obs[:, :, 21] > 0

    button_action = _navigate_or_interact(obs, button_mask, 3)
    supply_action = _navigate_or_interact(obs, supply_mask, 2)
    pot_action = _navigate_or_interact(obs, pot_mask, 2)
    plate_action = _navigate_or_interact(obs, plate_mask, 0)
    goal_action = _navigate_or_interact(obs, goal_mask, 2)

    recipe0 = jnp.sum(obs[:, :, 36])
    recipe1 = jnp.sum(obs[:, :, 37])
    recipe_visible = (recipe0 + recipe1) > 0
    sensed_target = jnp.where(recipe1 > recipe0, 1, 0).astype(jnp.int32)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:10]) > 0
    pot_ready = jnp.any((obs[:, :, 22] > 0) & (obs[:, :, 30] > 0))

    phase0 = jnp.where(recipe_visible, 1, 0)
    target0 = jnp.where(recipe_visible, sensed_target, target)

    phase1 = jnp.where(held_raw, 2, 1)
    phase2_interacted = held_raw & (pot_action == 5)
    next_inserted = jnp.where(phase2_interacted, inserted + 1, inserted)
    phase2 = jnp.where(
        phase2_interacted,
        jnp.where((inserted + 1) >= 3, 3, 1),
        2,
    )
    phase3 = jnp.where(pot_ready, 4, 3)
    phase4 = jnp.where(held_plate, 5, 4)
    phase5_interacted = held_plate & (pot_action == 5)
    phase5 = jnp.where(phase5_interacted, 6, 5)
    phase6_interacted = held_cooked & (goal_action == 5)
    phase6 = jnp.where(phase6_interacted, 0, 6)

    phase_next = jnp.where(
        phase == 0,
        phase0,
        jnp.where(
            phase == 1,
            phase1,
            jnp.where(
                phase == 2,
                phase2,
                jnp.where(
                    phase == 3,
                    phase3,
                    jnp.where(phase == 4, phase4, jnp.where(phase == 5, phase5, phase6)),
                ),
            ),
        ),
    )
    target_next = jnp.where((phase == 0) & recipe_visible, sensed_target, target)
    inserted_next = jnp.where((phase == 6) & phase6_interacted, 0, next_inserted)
    next_memory = jnp.stack((phase_next, target_next, inserted_next)).astype(jnp.int32)

    action_central = jnp.where(
        phase == 0,
        jnp.where(recipe_visible, supply_action, button_action),
        jnp.where(
            phase == 1,
            jnp.where(held_raw, pot_action, supply_action),
            jnp.where(
                phase == 2,
                pot_action,
                jnp.where(
                    phase == 3,
                    jnp.where(pot_ready, plate_action, 4),
                    jnp.where(
                        phase == 4,
                        jnp.where(held_plate, pot_action, plate_action),
                        jnp.where(phase == 5, pot_action, goal_action),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    action = jnp.where(central, action_central, 4).astype(jnp.int32)
    next_memory = jnp.where(central, next_memory, memory)
    trace = {
        'phase': phase.astype(jnp.int32),
        'central': central.astype(jnp.int32),
        'button_visible': jnp.any(button_mask).astype(jnp.int32),
        'recipe_visible': recipe_visible.astype(jnp.int32),
        'facing': jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32),
        'button_action': button_action.astype(jnp.int32),
        'button_row': (jnp.argmax(button_mask.reshape(-1)) // 5).astype(jnp.int32),
        'button_col': (jnp.argmax(button_mask.reshape(-1)) % 5).astype(jnp.int32),
    }
    return action, next_memory, trace


def agent_0_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)


def agent_1_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)
