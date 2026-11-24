"""Convention-oriented policy for the grounded two-room Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'anchored_button_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

DIR_RIGHT = 2
DIR_LEFT = 3

UNKNOWN = 2


def _loc(mask):
    flat = jnp.argmax(mask.reshape(-1)).astype(jnp.int32)
    return flat // 5, flat % 5


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_action(direction):
    return jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[direction]


def _go_or_interact(obs, target_r, target_c, face_dir):
    facing = _facing(obs)
    at_target = (target_r == 2) & (target_c == 2)
    move = jnp.where(
        target_c > 2,
        RIGHT,
        jnp.where(
            target_c < 2,
            LEFT,
            jnp.where(target_r > 2, DOWN, jnp.where(target_r < 2, UP, STAY)),
        ),
    )
    turn = _face_action(face_dir)
    return jnp.where(at_target, jnp.where(facing == face_dir, INTERACT, turn), move)


def _recipe_from_display(obs):
    count0 = jnp.sum(obs[:, :, 36])
    count1 = jnp.sum(obs[:, :, 37])
    seen = (count0 + count1) > 0
    recipe = jnp.where(count1 > count0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    return recipe, seen


def _button_action(obs):
    br, bc = _loc(obs[:, :, 24] > 0)
    return _go_or_interact(obs, br, bc - 1, DIR_RIGHT)


def init_agent_0():
    # [last signaled recipe, post-interaction cooldown]
    return jnp.asarray([2, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    base = init_agent_0()
    mem = jnp.where(reset, base, memory)
    recipe, seen = _recipe_from_display(obs)
    cooldown = jnp.maximum(mem[1] - 1, 0)
    need_signal = seen & (recipe != mem[0]) & (cooldown == 0)
    signal_action = _button_action(obs)
    action = jnp.where(need_signal, signal_action, STAY)
    signaled_now = need_signal & (signal_action == INTERACT)
    next_last = jnp.where(signaled_now, recipe, mem[0])
    next_cooldown = jnp.where(signaled_now, jnp.asarray(3, dtype=jnp.int32), cooldown)
    next_mem = jnp.asarray([next_last, next_cooldown], dtype=jnp.int32)
    trace = {
        'recipe': recipe,
        'seen': seen,
        'need_signal': need_signal,
        'last': next_last,
    }
    return action, next_mem, trace


def init_agent_1():
    # [remembered recipe: 0, 1, or UNKNOWN; post-delivery confirmation timer]
    return jnp.asarray([2, 0], dtype=jnp.int32)


def _pot_status(obs):
    pot = obs[:, :, 22] > 0
    cooked = jnp.sum(jnp.where(pot, obs[:, :, 30], 0)) > 0
    c0 = jnp.sum(jnp.where(pot, obs[:, :, 31], 0))
    c1 = jnp.sum(jnp.where(pot, obs[:, :, 32], 0))
    c2 = jnp.sum(jnp.where(pot, obs[:, :, 33], 0))
    timer = jnp.sum(jnp.where(pot, obs[:, :, 39], 0))
    total = c0 + c1 + c2
    return cooked, total, timer


def _ingredient_action(obs, recipe):
    mask0 = obs[:, :, 26] > 0
    mask1 = obs[:, :, 27] > 0
    r0, c0 = _loc(mask0)
    r1, c1 = _loc(mask1)
    target_r = jnp.where(recipe == 0, r0, r1)
    target_c = jnp.where(recipe == 0, c0 - 1, c1 - 1)
    return _go_or_interact(obs, target_r, target_c, DIR_RIGHT)


def _pot_action(obs):
    pr, pc = _loc(obs[:, :, 22] > 0)
    return _go_or_interact(obs, pr, pc + 1, DIR_LEFT)


def _plate_action(obs, recipe):
    plates = obs[:, :, 25] > 0
    pr, pc = _loc(plates)
    r0, _ = _loc(obs[:, :, 26] > 0)
    r1, _ = _loc(obs[:, :, 27] > 0)
    supply_r = jnp.where(recipe == 0, r0, r1)
    row_match = plates & (jnp.arange(5, dtype=jnp.int32)[:, None] == supply_r)
    mr, mc = _loc(row_match)
    has_match = jnp.sum(row_match.astype(jnp.int32)) > 0
    target_r = jnp.where(has_match, mr, pr)
    target_c = jnp.where(has_match, mc + 1, pc + 1)
    return _go_or_interact(obs, target_r, target_c, DIR_LEFT)


def _delivery_action(obs):
    gr, gc = _loc(obs[:, :, 21] > 0)
    return _go_or_interact(obs, gr, gc - 1, DIR_RIGHT)


def _observe_button_action(obs):
    pr, pc = _loc(obs[:, :, 22] > 0)
    target_r = pr
    target_c = pc + 1
    return jnp.where((target_r == 2) & (target_c == 2), STAY, _go_or_interact(obs, target_r, target_c, DIR_LEFT))


def agent_1_act(obs, reset, memory, rng):
    base = init_agent_1()
    mem = jnp.where(reset, base, memory)
    seen_recipe, seen = _recipe_from_display(obs)
    recipe = jnp.where(seen, seen_recipe, mem[0])
    confirming = (mem[1] > 0) & (~seen)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = (obs[2, 2, 7] + obs[2, 2, 8] + obs[2, 2, 9]) > 0
    pot_cooked, pot_total, pot_timer = _pot_status(obs)
    known = (recipe != UNKNOWN) & (~confirming)

    wait_read = _observe_button_action(obs)
    deliver = _delivery_action(obs)
    plate = _plate_action(obs, recipe)
    pot = _pot_action(obs)
    ingredient = _ingredient_action(obs, recipe)

    action = jnp.where(
        ~known,
        wait_read,
        jnp.where(
            held_cooked,
            deliver,
            jnp.where(
                held_plate,
                jnp.where(pot_cooked, pot, jnp.where(pot_timer > 0, pot, STAY)),
                jnp.where(
                    held_raw,
                    jnp.where((pot_timer == 0) & (~pot_cooked) & (pot_total < 3), pot, STAY),
                    jnp.where(
                        pot_cooked,
                        plate,
                        jnp.where(pot_timer > 0, plate, jnp.where(pot_total < 3, ingredient, plate)),
                    ),
                ),
            ),
        ),
    )

    delivered_now = held_cooked & (action == INTERACT)
    dec_confirm = jnp.maximum(mem[1] - 1, 0)
    next_confirm = jnp.where(seen, 0, dec_confirm)
    next_confirm = jnp.where(delivered_now, jnp.asarray(5, dtype=jnp.int32), next_confirm)
    next_mem = jnp.asarray([recipe, next_confirm], dtype=jnp.int32)
    trace = {
        'recipe': recipe,
        'seen': seen,
        'confirm': mem[1],
        'pot_total': pot_total.astype(jnp.int32),
        'pot_cooked': pot_cooked,
        'held_cooked': held_cooked,
    }
    return action, next_mem, trace
