"""A self-contained production policy for the right-hand room.

The right seat can reach every useful object.  It cannot see the recipe
indicator in the left room, so completed dishes alternate ingredient 0 and 1.
An incorrect delivery leaves the target unchanged; this makes the alternating
schedule self-correcting in expectation without requiring a shared signal.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'alternating_room_worker', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    action = jnp.asarray(4, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


def init_agent_1():
    # phase, ingredient_count, reserved, ingredient_parity
    return jnp.zeros((4,), dtype=jnp.int32)


def _target_navigation(obs, channel, target_on_left):
    """Move beside the unique static object in ``channel``.

    The right room is only three rows high and every useful object remains in
    the 5x5 observation, so the target's local coordinates are sufficient.
    For the two plate piles, argmax consistently selects the upper pile.
    """
    mask = obs[:, :, channel] > 0
    flat = jnp.argmax(mask.reshape((-1,)))
    target_row = flat // 5
    target_col = flat % 5
    on_row = target_row == 2
    if_left = target_on_left
    beside = on_row & jnp.where(if_left, target_col == 1, target_col == 3)
    vertical = jnp.where(target_row < 2, 3, jnp.where(target_row > 2, 1, 4))
    horizontal = jnp.where(
        if_left,
        jnp.where(target_col > 1, 0, jnp.where(target_col < 1, 2, 5)),
        jnp.where(target_col > 3, 0, jnp.where(target_col < 3, 2, 5)),
    )
    action = jnp.where(on_row, horizontal, vertical)
    return action, beside


def agent_1_act(obs, reset, memory, rng):
    # The two recipes are symmetric; this experiment starts on ingredient 1
    # to test whether reset sampling creates any useful observable bias.
    initial = jnp.asarray([0, 0, 0, 1], dtype=jnp.int32)
    state = jnp.where(reset, initial, memory)
    phase, ingredient_count, unused, parity = state

    # Phase 0: fetch the ingredient for this cycle.
    ingredient_mask = jnp.where(parity == 0, obs[:, :, 26], obs[:, :, 27]) > 0
    ingredient_flat = jnp.argmax(ingredient_mask.reshape((-1,)))
    ingredient_row = ingredient_flat // 5
    ingredient_col = ingredient_flat % 5
    ingredient_on_row = ingredient_row == 2
    ingredient_beside = ingredient_on_row & (ingredient_col == 3)
    ingredient_move = jnp.where(
        ingredient_row < 2,
        3,
        jnp.where(
            ingredient_row > 2,
            1,
            jnp.where(ingredient_col > 3, 0, jnp.where(ingredient_col < 3, 2, 5)),
        ),
    )
    facing_right = obs[2, 2, 3] > 0
    ingredient_done = ingredient_beside & facing_right
    ingredient_action = jnp.where(
        ingredient_beside,
        jnp.where(facing_right, 5, 0),
        ingredient_move,
    )

    # Phase 1: make three deposits into the pot.
    pot_move, pot_beside = _target_navigation(obs, 22, True)
    facing_left = obs[2, 2, 4] > 0
    pot_deposit = pot_beside & facing_left
    pot_action = jnp.where(pot_beside, jnp.where(facing_left, 5, 2), pot_move)

    # Phase 2: collect an empty plate from the upper pile.
    plate_move, plate_beside = _target_navigation(obs, 25, True)
    plate_pickup = plate_beside & facing_left
    plate_action = jnp.where(plate_beside, jnp.where(facing_left, 5, 2), plate_move)

    # Phase 3: wait at the pot until its cooked bit appears, then plate.
    pot_cooked = jnp.sum((obs[:, :, 22] > 0) * (obs[:, :, 30] > 0)) > 0
    pot_collect = pot_beside & pot_cooked & facing_left
    cook_action = jnp.where(
        pot_beside,
        jnp.where(pot_collect, 5, 2),
        pot_move,
    )

    # Phase 4: deliver at the goal and switch the blind recipe schedule.
    goal_move, goal_beside = _target_navigation(obs, 21, False)
    goal_delivery = goal_beside & facing_right
    goal_action = jnp.where(goal_beside, jnp.where(facing_right, 5, 0), goal_move)

    phase0_next = jnp.where(ingredient_done, 1, phase)
    # Picking the next raw unit must preserve how many units are already in
    # the pot; only delivery resets the cycle parity, not this counter.
    count_next = ingredient_count

    phase1_next = jnp.where(
        pot_deposit,
        jnp.where(ingredient_count == 2, 2, 0),
        phase,
    )
    count1_next = jnp.where(pot_deposit, ingredient_count + 1, ingredient_count)

    phase2_next = jnp.where(plate_pickup, 3, phase)
    phase3_next = jnp.where(pot_collect, 4, phase)
    phase4_next = jnp.where(goal_delivery, 0, phase)
    parity4_next = jnp.where(goal_delivery, 1 - parity, parity)

    next_phase = jnp.where(
        phase == 0,
        phase0_next,
        jnp.where(
            phase == 1,
            phase1_next,
            jnp.where(phase == 2, phase2_next, jnp.where(phase == 3, phase3_next, phase4_next)),
        ),
    )
    next_count = jnp.where(
        (phase == 4) & goal_delivery,
        0,
        jnp.where(phase == 0, count_next, jnp.where(phase == 1, count1_next, ingredient_count)),
    )
    next_parity = jnp.where((phase == 4) & goal_delivery, parity4_next, parity)
    next_state = jnp.stack((next_phase, next_count, unused, next_parity)).astype(jnp.int32)

    action = jnp.where(
        phase == 0,
        ingredient_action,
        jnp.where(
            phase == 1,
            pot_action,
            jnp.where(phase == 2, plate_action, jnp.where(phase == 3, cook_action, goal_action)),
        ),
    ).astype(jnp.int32)
    trace = {
        "phase": phase,
        "count": ingredient_count,
        "parity": parity,
        "pot_cooked": pot_cooked,
    }
    return action, next_state, trace
