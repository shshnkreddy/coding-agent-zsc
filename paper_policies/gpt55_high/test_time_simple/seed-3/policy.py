"""XP-oriented policy for the simple two-room Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'alternating_solo_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    zero = jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(STAY, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    # [next ingredient to cook, completed deliveries]
    return jnp.zeros((2,), dtype=jnp.int32)


def _agent_1_position(obs):
    at_top_right = obs[2, 3, 26] > 0
    at_bottom_right = obs[2, 3, 27] > 0
    at_mid_left = obs[2, 1, 22] > 0
    at_mid_right = obs[2, 3, 21] > 0
    plate_left = obs[2, 1, 25] > 0
    at_top_left = plate_left & (obs[2, 4, 26] > 0)
    at_bottom_left = plate_left & (obs[2, 4, 27] > 0)

    top = at_top_left | at_top_right
    middle = at_mid_left | at_mid_right
    bottom = at_bottom_left | at_bottom_right
    right_col = at_top_right | at_mid_right | at_bottom_right
    left_col = at_top_left | at_mid_left | at_bottom_left

    pos_code = (
        jnp.where(at_top_left, 0, 0)
        + jnp.where(at_top_right, 1, 0)
        + jnp.where(at_mid_left, 2, 0)
        + jnp.where(at_mid_right, 3, 0)
        + jnp.where(at_bottom_left, 4, 0)
        + jnp.where(at_bottom_right, 5, 0)
    )
    return top, middle, bottom, left_col, right_col, pos_code


def _nav_to(obs, target_row, target_right_col):
    top, middle, bottom, left_col, right_col, _ = _agent_1_position(obs)

    if target_row == 0:
        need_vertical = ~top
        vertical_action = UP
    elif target_row == 1:
        need_vertical = ~middle
        vertical_action = jnp.where(top, DOWN, UP)
    else:
        need_vertical = ~bottom
        vertical_action = DOWN

    need_horizontal = jnp.where(target_right_col, ~right_col, ~left_col)
    horizontal_action = jnp.where(target_right_col, RIGHT, LEFT)
    return jnp.where(
        need_vertical,
        vertical_action,
        jnp.where(need_horizontal, horizontal_action, STAY),
    )


def _face_or_interact(is_facing, face_action):
    return jnp.where(is_facing, INTERACT, face_action)


def _go_get_ingredient(obs, target):
    top, _, bottom, _, right_col, _ = _agent_1_position(obs)
    at_target0 = top & right_col
    at_target1 = bottom & right_col
    at_target = jnp.where(target == 0, at_target0, at_target1)
    nav0 = _nav_to(obs, 0, True)
    nav1 = _nav_to(obs, 2, True)
    nav = jnp.where(target == 0, nav0, nav1)
    use_supply = _face_or_interact(obs[2, 2, 3] > 0, RIGHT)
    return jnp.where(at_target, use_supply, nav)


def _go_pot(obs, interact_when_ready):
    _, middle, _, left_col, _, _ = _agent_1_position(obs)
    at_pot = middle & left_col
    use_pot = jnp.where(
        interact_when_ready,
        _face_or_interact(obs[2, 2, 4] > 0, LEFT),
        jnp.where(obs[2, 2, 4] > 0, STAY, LEFT),
    )
    return jnp.where(at_pot, use_pot, _nav_to(obs, 1, False))


def _go_plate(obs):
    top, _, _, left_col, _, _ = _agent_1_position(obs)
    at_plate = top & left_col
    use_plate = _face_or_interact(obs[2, 2, 4] > 0, LEFT)
    return jnp.where(at_plate, use_plate, _nav_to(obs, 0, False))


def _go_deliver(obs):
    _, middle, _, _, right_col, _ = _agent_1_position(obs)
    at_goal = middle & right_col
    use_goal = _face_or_interact(obs[2, 2, 3] > 0, RIGHT)
    return jnp.where(at_goal, use_goal, _nav_to(obs, 1, True))


def agent_1_act(obs, reset, memory, rng):
    _, _, bottom, _, _, _ = _agent_1_position(obs)
    reset_target = jnp.where(bottom, 1, 0)
    reset_mem = jnp.stack([reset_target, jnp.asarray(0, dtype=jnp.int32)])
    mem = jnp.where(reset, reset_mem, memory)
    target = mem[0]

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_i0 = obs[2, 2, 7] > 0
    held_i1 = obs[2, 2, 8] > 0
    held_i2 = obs[2, 2, 9] > 0
    held_raw = held_i0 | held_i1 | held_i2
    empty_handed = ~(held_plate | held_cooked | held_raw)

    pot_mask = obs[:, :, 22]
    pot_i0 = jnp.sum(obs[:, :, 31] * pot_mask)
    pot_i1 = jnp.sum(obs[:, :, 32] * pot_mask)
    pot_i2 = jnp.sum(obs[:, :, 33] * pot_mask)
    pot_count = pot_i0 + pot_i1 + pot_i2
    pot_cooked = jnp.sum(obs[:, :, 30] * pot_mask) > 0
    pot_timer = jnp.sum(obs[:, :, 39] * pot_mask)

    action_deliver = _go_deliver(obs)
    action_deposit = _go_pot(obs, True)
    action_collect = _go_pot(obs, pot_cooked)
    action_plate = _go_plate(obs)
    action_ingredient = _go_get_ingredient(obs, target)

    empty_plan = jnp.where(
        pot_cooked | (pot_timer > 0) | (pot_count >= 3),
        action_plate,
        action_ingredient,
    )
    action = jnp.where(
        held_cooked,
        action_deliver,
        jnp.where(
            held_raw,
            action_deposit,
            jnp.where(held_plate, action_collect, jnp.where(empty_handed, empty_plan, STAY)),
        ),
    )

    _, middle, _, _, right_col, pos_code = _agent_1_position(obs)
    delivery_attempt = held_cooked & middle & right_col & (obs[2, 2, 3] > 0) & (action == INTERACT)
    next_target = jnp.where(delivery_attempt, 1 - target, target)
    next_deliveries = jnp.where(delivery_attempt, mem[1] + 1, mem[1])
    next_memory = jnp.asarray([next_target, next_deliveries], dtype=jnp.int32)

    trace = {
        'target': target,
        'deliveries': mem[1],
        'pot_count': pot_count,
        'pot_timer': pot_timer,
        'pot_cooked': pot_cooked,
        'pos': pos_code,
        'action': action,
    }
    return action, next_memory, trace
