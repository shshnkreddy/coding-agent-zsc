"""Cross-play-oriented controller for the split test_time_simple kitchen.

The left player displays the recipe with an otherwise unusable ingredient-2
token on one of the two divider counters (upper = ingredient 0, lower =
ingredient 1). The right player reads that persistent, externally observable
signal and handles the pot, plates, and deliveries reactively.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "divider_signal_reactive",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 8,
    "num_ingredients": 3,
}


RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _facing(obs):
    """Observation direction index: up=0, down=1, right=2, left=3."""
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _go(r, c, target_r, target_c, final_facing, obs, interact=True):
    """Navigate inside one of the obstacle-free 3x2 player rooms."""
    face = _facing(obs)
    face_action = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)[final_facing]
    arrived_action = jnp.where(
        face == final_facing,
        INTERACT if interact else STAY,
        face_action,
    )
    action = jnp.where(
        r < target_r,
        DOWN,
        jnp.where(
            r > target_r,
            UP,
            jnp.where(c < target_c, RIGHT, jnp.where(c > target_c, LEFT, arrived_action)),
        ),
    )
    return action.astype(jnp.int32)


def _global_cell(obs, self_r, self_c, global_r, global_c, channel):
    """Read a known global cell; callers gate cells that can leave view."""
    lr = jnp.clip(global_r - self_r + 2, 0, 4)
    lc = jnp.clip(global_c - self_c + 2, 0, 4)
    return obs[lr, lc, channel]


def init_agent_0():
    # [elapsed time, previous pot-nonempty flag, pulse hold counter]
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    elapsed, previous_pot_nonempty, pulse_count = memory

    # The always-on indicator at global (2, 0) uniquely localizes the player.
    indicator_flat = jnp.argmax(obs[:, :, 23]).astype(jnp.int32)
    indicator_lr = indicator_flat // 5
    indicator_lc = indicator_flat % 5
    r = (4 - indicator_lr).astype(jnp.int32)
    c = (2 - indicator_lc).astype(jnp.int32)

    recipe0 = obs[indicator_lr, indicator_lc, 36]
    recipe1 = obs[indicator_lr, indicator_lc, 37]
    target = jnp.where(recipe1 > recipe0, 1, 0).astype(jnp.int32)
    desired_r = jnp.where(target == 0, 1, 3).astype(jnp.int32)
    other_r = jnp.where(target == 0, 3, 1).astype(jnp.int32)

    top_signal = _global_cell(obs, r, c, 1, 3, 33) > 0
    bottom_signal = _global_cell(obs, r, c, 3, 3, 33) > 0
    desired_present = jnp.where(target == 0, top_signal, bottom_signal)
    other_present = jnp.where(target == 0, bottom_signal, top_signal)

    held2 = obs[2, 2, 9] > 0
    place = _go(r, c, desired_r, jnp.int32(2), jnp.int32(2), obs)
    retrieve = _go(r, c, other_r, jnp.int32(2), jnp.int32(2), obs)
    retrieve_current = _go(r, c, desired_r, jnp.int32(2), jnp.int32(2), obs)
    supply_face = jnp.where(target == 0, 0, 1).astype(jnp.int32)
    fetch = _go(r, c, desired_r, jnp.int32(2), supply_face, obs)
    # Park beside the current signal. This makes removal an immediate public
    # response to the pot being plated.
    park = jnp.where(
        r < desired_r,
        DOWN,
        jnp.where(r > desired_r, UP, jnp.where(c < 2, RIGHT, jnp.where(c > 2, LEFT, STAY))),
    ).astype(jnp.int32)

    empty_action = jnp.where(
        desired_present,
        park,
        jnp.where(other_present, retrieve, fetch),
    )
    ordinary_action = jnp.where(held2, place, empty_action)

    # From column 2 the pot is exactly at the edge of the local crop. Its
    # nonempty->empty transition means the dish was plated, shortly before
    # delivery. Pulse the current signal off long enough for the returning
    # partner to see a recipe-epoch boundary.
    pot0 = _global_cell(obs, r, c, 2, 4, 31)
    pot1 = _global_cell(obs, r, c, 2, 4, 32)
    pot2 = _global_cell(obs, r, c, 2, 4, 33)
    pot_nonempty = (c == 2) & ((pot0 + pot1 + pot2) > 0)
    pot_emptied = (previous_pot_nonempty > 0) & (~pot_nonempty) & (c == 2)
    hold_position = _go(r, c, desired_r, jnp.int32(2), jnp.int32(2), obs, interact=False)
    pulse_action = jnp.where(
        pulse_count > 1,
        hold_position,
        place,
    )
    action = jnp.where(
        pot_emptied & (~held2) & desired_present,
        retrieve_current,
        jnp.where(pulse_count > 0, pulse_action, ordinary_action),
    ).astype(jnp.int32)
    next_pulse = jnp.where(
        pot_emptied & (~held2) & desired_present,
        jnp.int32(4),
        jnp.maximum(pulse_count - jnp.int32(1), jnp.int32(0)),
    )
    next_memory = jnp.stack(
        (elapsed + jnp.int32(1), pot_nonempty.astype(jnp.int32), next_pulse)
    ).astype(jnp.int32)
    trace = {
        "target": target,
        "signal_top": top_signal.astype(jnp.int32),
        "signal_bottom": bottom_signal.astype(jnp.int32),
        "position": jnp.stack((r, c)),
        "pulse": pulse_count,
    }
    return action, next_memory, trace


def init_agent_1():
    # [elapsed, target, fallback wait, gap flag, prior top, prior bottom]
    return jnp.asarray([0, -1, 0, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    elapsed, old_target, wait_count, old_gap_seen, prior_top, prior_bottom = memory

    # The pot at global (2, 4) uniquely localizes the player in the right room.
    pot_flat = jnp.argmax(obs[:, :, 22]).astype(jnp.int32)
    pot_lr = pot_flat // 5
    pot_lc = pot_flat % 5
    r = (4 - pot_lr).astype(jnp.int32)
    c = (6 - pot_lc).astype(jnp.int32)

    top_signal = _global_cell(obs, r, c, 1, 3, 33) > 0
    bottom_signal = _global_cell(obs, r, c, 3, 3, 33) > 0
    at_reader = c == 5
    exclusive_signal = top_signal != bottom_signal
    rising_top = top_signal & (prior_top == 0)
    rising_bottom = bottom_signal & (prior_bottom == 0)
    unique_rising_edge = rising_top != rising_bottom
    valid_signal = exclusive_signal | unique_rising_edge
    observed_target = jnp.where(
        exclusive_signal,
        jnp.where(bottom_signal, 1, 0),
        jnp.where(rising_bottom, 1, 0),
    ).astype(jnp.int32)
    no_signal = (~top_signal) & (~bottom_signal)
    gap_seen = (old_gap_seen > 0) | ((wait_count > 0) & at_reader & no_signal)
    resumed_from_gap = gap_seen & at_reader & valid_signal
    target = jnp.where(
        at_reader & valid_signal,
        observed_target,
        jnp.where(at_reader & gap_seen, jnp.int32(-1), old_target),
    )
    waiting = (wait_count > 0) & (~resumed_from_gap)

    pot0 = obs[pot_lr, pot_lc, 31]
    pot1 = obs[pot_lr, pot_lc, 32]
    pot2 = obs[pot_lr, pot_lc, 33]
    pot_total = pot0 + pot1 + pot2
    pot_cooked = obs[pot_lr, pot_lc, 30] > 0
    pot_timer = obs[pot_lr, pot_lc, 39]

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held0 = obs[2, 2, 7] > 0
    held1 = obs[2, 2, 8] > 0
    held_raw = held0 | held1 | (obs[2, 2, 9] > 0)

    read_signal = _go(r, c, jnp.int32(2), jnp.int32(5), jnp.int32(3), obs, interact=False)
    use_pot = _go(r, c, jnp.int32(2), jnp.int32(5), jnp.int32(3), obs)
    get_plate = _go(r, c, jnp.int32(1), jnp.int32(5), jnp.int32(3), obs)
    deliver = _go(r, c, jnp.int32(2), jnp.int32(6), jnp.int32(2), obs)
    supply_r = jnp.where(target == 0, 1, 3).astype(jnp.int32)
    get_ingredient = _go(r, c, supply_r, jnp.int32(6), jnp.int32(2), obs)

    # A plate holder waits at the pot until pickup becomes possible.
    with_plate = use_pot
    empty_work = jnp.where(
        target < 0,
        read_signal,
        jnp.where(
            waiting,
            read_signal,
            jnp.where(
                pot_cooked,
                get_plate,
                jnp.where(pot_timer > 0, get_plate, jnp.where(pot_total < 3, get_ingredient, use_pot)),
            ),
        ),
    )
    action = jnp.where(
        held_cooked,
        deliver,
        jnp.where(held_plate, with_plate, jnp.where(held_raw, use_pot, empty_work)),
    ).astype(jnp.int32)

    delivered_now = held_cooked & (action == INTERACT)
    next_wait = jnp.where(
        delivered_now,
        jnp.int32(8),
        jnp.where(
            resumed_from_gap,
            jnp.int32(0),
            jnp.maximum(wait_count - jnp.int32(1), jnp.int32(0)),
        ),
    )
    next_gap = jnp.where(delivered_now | resumed_from_gap, jnp.int32(0), gap_seen.astype(jnp.int32))
    next_top = jnp.where(at_reader, top_signal.astype(jnp.int32), prior_top)
    next_bottom = jnp.where(at_reader, bottom_signal.astype(jnp.int32), prior_bottom)
    next_memory = jnp.stack(
        (elapsed + jnp.int32(1), target, next_wait, next_gap, next_top, next_bottom)
    ).astype(jnp.int32)
    trace = {
        "target": target,
        "cooldown": wait_count,
        "gap_seen": gap_seen.astype(jnp.int32),
        "valid_signal": valid_signal.astype(jnp.int32),
        "pot": jnp.asarray([pot0, pot1, pot2, pot_timer, pot_cooked], dtype=jnp.int32),
        "signals": jnp.stack((top_signal, bottom_signal)).astype(jnp.int32),
        "position": jnp.stack((r, c)),
    }
    return action, next_memory, trace
