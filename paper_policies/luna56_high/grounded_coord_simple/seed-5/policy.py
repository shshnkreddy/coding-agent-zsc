"""A small, symmetric reactive controller for the grounded OCV2 layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'symmetric_recipe_worker',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}


# Memory is [side, phase, target_ingredient, loaded_count, broadcaster_clock].
# side 0 is the left room (the broadcaster), side 1 is the right room (worker).
def _memory(side=0, phase=0, target=0, loaded=0, clock=0):
    return jnp.asarray([side, phase, target, loaded, clock], dtype=jnp.int32)


def init_agent_0():
    return _memory()


def init_agent_1():
    return _memory()


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _target_delta(obs, channel):
    flat = jnp.argmax(obs[:, :, channel].reshape(-1)).astype(jnp.int32)
    return flat // 5 - 2, flat % 5 - 2


def _toward(obs, channel):
    """Move toward a visible static object, then interact when adjacent."""
    present = jnp.any(obs[:, :, channel] != 0)
    dr, dc = _target_delta(obs, channel)
    vertical = dr != 0
    desired = jnp.where(vertical, jnp.where(dr < 0, 0, 1),
                        jnp.where(dc < 0, 3, 2))
    # desired direction is UP=0, DOWN=1, RIGHT=2, LEFT=3.
    move_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[desired]
    face = _facing(obs)
    adjacent = jnp.abs(dr) + jnp.abs(dc) == 1
    action = jnp.where(adjacent & (face == desired), 5, move_action)
    return jnp.where(present, action, jnp.asarray(4, dtype=jnp.int32))


def _left_act(obs, memory):
    button_action = _toward(obs, 24)
    clock = memory[4]
    activation_count = memory[3]
    due = ((clock == 0) | (clock >= 46)) & (activation_count < 8)
    action = jnp.where(due, button_action, jnp.asarray(4, dtype=jnp.int32))
    activated = due & (action == 5)
    next_clock = jnp.where(activated, 1,
                           jnp.where(clock == 0, 0, jnp.minimum(clock + 1, 46)))
    next_memory = memory.at[4].set(next_clock)
    next_memory = next_memory.at[3].set(jnp.where(activated, activation_count + 1,
                                                  activation_count))
    return action, next_memory


def _right_act(obs, memory):
    phase = memory[1]
    target = memory[2]
    loaded = memory[3]

    # Phase 0: center at right room row 2, column 5, then read the active
    # button display.  The goal's relative position identifies our row.
    goal_dr, goal_dc = _target_delta(obs, 21)
    center_move = jnp.where(goal_dr < 0, 3,
                            jnp.where(goal_dr > 0, 1,
                                      jnp.where(goal_dc < 2, 2, 4)))
    display = obs[:, :, 36:39]
    has_display = jnp.any(display != 0)
    at_center = (goal_dr == 0) & (goal_dc == 2)
    acquired = (phase == 0) & at_center & has_display
    observed_target = jnp.where(jnp.any(obs[:, :, 36] != 0), 0, 1).astype(jnp.int32)
    p0_action = jnp.where(at_center & has_display, 4, center_move)
    p0_memory = memory.at[1].set(jnp.where(acquired, 1, phase))
    p0_memory = p0_memory.at[2].set(jnp.where(acquired, observed_target, target))

    # Phase 1: fetch one unit of the selected ingredient.
    supply_channel = jnp.where(target == 0, 26, 27).astype(jnp.int32)
    p1_action = _toward(obs, supply_channel)
    picked = p1_action == 5
    p1_memory = memory.at[1].set(jnp.where(picked, 3, 1))

    # Phase 3: return to the pot and insert the held ingredient.
    p3_action = _toward(obs, 22)
    inserted = p3_action == 5
    full = inserted & (loaded == 2)
    p3_memory = memory.at[1].set(jnp.where(inserted, jnp.where(full, 4, 1), 3))
    p3_memory = p3_memory.at[3].set(jnp.where(inserted, loaded + 1, loaded))

    # Phase 4: cooked flag, rather than timer==0, marks readiness.
    pot_cooked = jnp.any((obs[:, :, 22] != 0) & (obs[:, :, 30] != 0))
    p4_action = jnp.where(pot_cooked, _toward(obs, 25), 4)
    p4_memory = memory.at[1].set(jnp.where(pot_cooked, 5, 4))

    # Phase 5: take the first (top) visible plate pile.
    p5_action = _toward(obs, 25)
    plated = p5_action == 5
    p5_memory = memory.at[1].set(jnp.where(plated, 6, 5))
    p5_memory = p5_memory.at[4].set(jnp.where(plated, 0, memory[4]))

    # Phase 6: an empty plate must return to the cooked pot to scoop the dish.
    # From the top plate the worker is at row 1, column 5; _toward handles the
    # short down/left/interact route to the pot.
    held_plate = obs[2, 2, 5] != 0
    held_cooked = obs[2, 2, 6] != 0
    scoop_action = _toward(obs, 22)
    scooped = held_plate & (~held_cooked) & (scoop_action == 5)
    p6_action = jnp.where(held_cooked, 4, scoop_action)
    p6_memory = memory.at[1].set(jnp.where(scooped | held_cooked, 7, 6))
    p6_memory = p6_memory.at[4].set(jnp.where(scooped | held_cooked, 0, memory[4]))

    # Phase 7: carry the plated dish to the goal and interact.
    p7_action = jnp.where(held_cooked, _toward(obs, 21), 4)
    attempted = held_cooked & (p7_action == 5)
    p7_memory = memory.at[1].set(jnp.where(attempted, 8, 7))
    p7_memory = p7_memory.at[4].set(jnp.where(attempted, 0, memory[4]))

    # Phase 8: verify the delivery.  A successful delivery clears the held
    # cooked flag; otherwise retry the interaction at the same location.
    p8_action = jnp.where(held_cooked, 5, 4)
    p8_memory = memory.at[1].set(jnp.where(held_cooked, 8, 0))
    p8_memory = p8_memory.at[2].set(jnp.where(held_cooked, target, 0))
    p8_memory = p8_memory.at[3].set(jnp.where(held_cooked, loaded, 0))
    p8_memory = p8_memory.at[4].set(jnp.where(held_cooked, 0, 0))

    action = jnp.where(phase == 0, p0_action,
             jnp.where(phase == 1, p1_action,
             jnp.where(phase == 3, p3_action,
             jnp.where(phase == 4, p4_action,
             jnp.where(phase == 5, p5_action,
             jnp.where(phase == 6, p6_action,
             jnp.where(phase == 7, p7_action, p8_action)))))))
    next_memory = jnp.where((phase == 0), p0_memory,
                    jnp.where((phase == 1), p1_memory,
                    jnp.where((phase == 3), p3_memory,
                    jnp.where((phase == 4), p4_memory,
                    jnp.where((phase == 5), p5_memory,
                    jnp.where((phase == 6), p6_memory,
                    jnp.where((phase == 7), p7_memory, p8_memory)))))))
    return action, next_memory


def _act(obs, reset, memory):
    # R is visible throughout the left spawn room and nowhere in the right
    # spawn room, providing a local deterministic role discriminator.
    is_left = jnp.any(obs[:, :, 23] != 0)
    reset_memory = _memory(jnp.where(is_left, 0, 1))
    base = jnp.where(reset, reset_memory, memory)
    left_action, left_memory = _left_act(obs, base)
    right_action, right_memory = _right_act(obs, base)
    action = jnp.where(base[0] == 0, left_action, right_action)
    next_memory = jnp.where((base[0] == 0), left_memory, right_memory)
    return action.astype(jnp.int32), next_memory


def agent_0_act(obs, reset, memory, rng):
    action, next_memory = _act(obs, reset, memory)
    gdr, gdc = _target_delta(obs, 21)
    return action, next_memory, {
        'side': next_memory[0], 'phase': next_memory[1],
        'held_plate': obs[2, 2, 5], 'held_cooked': obs[2, 2, 6],
        'goal_dr': gdr, 'goal_dc': gdc,
    }


def agent_1_act(obs, reset, memory, rng):
    action, next_memory = _act(obs, reset, memory)
    gdr, gdc = _target_delta(obs, 21)
    return action, next_memory, {
        'side': next_memory[0], 'phase': next_memory[1],
        'held_plate': obs[2, 2, 5], 'held_cooked': obs[2, 2, 6],
        'goal_dr': gdr, 'goal_dc': gdc,
    }
