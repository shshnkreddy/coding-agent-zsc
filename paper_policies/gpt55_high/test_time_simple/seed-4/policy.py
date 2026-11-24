"""Geometry-based Overcooked policy for the test_time_simple layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'shared_counter_signal_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

DIR_UP = 0
DIR_DOWN = 1
DIR_RIGHT = 2
DIR_LEFT = 3


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _flat_coord(mask):
    idx = jnp.argmax(mask.reshape((-1,)))
    return (idx // _i32(5)).astype(jnp.int32), (idx % _i32(5)).astype(jnp.int32)


def _self_from_known_object(obs, channel, global_r, global_c):
    lr, lc = _flat_coord(obs[:, :, channel])
    return _i32(global_r + 2) - lr, _i32(global_c + 2) - lc


def _cell(obs, self_r, self_c, global_r, global_c, channel):
    lr = _i32(global_r + 2) - self_r
    lc = _i32(global_c + 2) - self_c
    in_view = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    lr = jnp.clip(lr, 0, 4)
    lc = jnp.clip(lc, 0, 4)
    return jnp.where(in_view, obs[lr, lc, channel], _i32(0))


def _item_any(obs, self_r, self_c, global_r, global_c):
    total = (
        _cell(obs, self_r, self_c, global_r, global_c, 29)
        + _cell(obs, self_r, self_c, global_r, global_c, 30)
        + _cell(obs, self_r, self_c, global_r, global_c, 31)
        + _cell(obs, self_r, self_c, global_r, global_c, 32)
        + _cell(obs, self_r, self_c, global_r, global_c, 33)
    )
    return total > 0


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _turn_action(direction):
    return jnp.where(
        direction == DIR_UP,
        UP,
        jnp.where(direction == DIR_DOWN, DOWN, jnp.where(direction == DIR_RIGHT, RIGHT, LEFT)),
    )


def _face_then_interact(obs, direction):
    return jnp.where(_facing(obs) == direction, INTERACT, _turn_action(direction))


def _move_to(self_r, self_c, target_r, target_c, arrived_action):
    return jnp.where(
        self_c < target_c,
        RIGHT,
        jnp.where(
            self_c > target_c,
            LEFT,
            jnp.where(
                self_r < target_r,
                DOWN,
                jnp.where(self_r > target_r, UP, arrived_action),
            ),
        ),
    )


def _move_to_row_first(self_r, self_c, target_r, target_c, arrived_action):
    return jnp.where(
        self_r < target_r,
        DOWN,
        jnp.where(
            self_r > target_r,
            UP,
            jnp.where(
                self_c < target_c,
                RIGHT,
                jnp.where(self_c > target_c, LEFT, arrived_action),
            ),
        ),
    )


def _go_interact(obs, self_r, self_c, target_r, target_c, direction):
    return _move_to(self_r, self_c, target_r, target_c, _face_then_interact(obs, direction))


def _go_interact_row_first(obs, self_r, self_c, target_r, target_c, direction):
    return _move_to_row_first(self_r, self_c, target_r, target_c, _face_then_interact(obs, direction))


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _i32(0), memory)

    self_r, self_c = _self_from_known_object(obs, 23, 2, 0)
    recipe0 = jnp.sum(obs[:, :, 36])
    recipe1 = jnp.sum(obs[:, :, 37])
    wants_one = recipe1 > recipe0

    desired_row = jnp.where(wants_one, _i32(3), _i32(1))
    other_row = jnp.where(wants_one, _i32(1), _i32(3))
    held_i2 = obs[2, 2, 9] > 0
    desired_has = _item_any(obs, self_r, self_c, desired_row, _i32(3))
    other_has = _item_any(obs, self_r, self_c, other_row, _i32(3))

    put_signal = _go_interact(obs, self_r, self_c, desired_row, _i32(2), DIR_RIGHT)
    remove_old = _go_interact(obs, self_r, self_c, other_row, _i32(2), DIR_RIGHT)

    supply_adj_row = desired_row
    supply_face = jnp.where(wants_one, DIR_DOWN, DIR_UP)
    fetch_signal = _go_interact(obs, self_r, self_c, supply_adj_row, _i32(2), supply_face)

    action = jnp.where(
        held_i2,
        put_signal,
        jnp.where(other_has, remove_old, jnp.where(desired_has, STAY, fetch_signal)),
    )
    trace = {
        'recipe1': wants_one.astype(jnp.int32),
        'desired_row': desired_row,
        'holding_i2': held_i2.astype(jnp.int32),
    }
    return action.astype(jnp.int32), memory, trace


def init_agent_1():
    # [remembered ingredient id, signal known, post-delivery cooldown, spare]
    return jnp.zeros((4,), dtype=jnp.int32)


def _right_plate_row(self_r):
    return jnp.where(self_r <= 2, _i32(1), _i32(3))


def _drop_wrong_raw(obs, self_r, self_c):
    return _go_interact(obs, self_r, self_c, _i32(1), _i32(5), DIR_UP)


def agent_1_act(obs, reset, memory, rng):
    del rng
    zero_mem = jnp.zeros((4,), dtype=jnp.int32)
    old = jnp.where(reset, zero_mem, memory)

    self_r, self_c = _self_from_known_object(obs, 21, 2, 7)

    top_signal = _item_any(obs, self_r, self_c, _i32(1), _i32(3))
    bottom_signal = _item_any(obs, self_r, self_c, _i32(3), _i32(3))
    exclusive_signal = top_signal ^ bottom_signal
    observed_target = jnp.where(bottom_signal, _i32(1), _i32(0))
    cooldown = old[2]
    resyncing = old[3]
    infer_flip = (resyncing == 1) & (cooldown == 0) & (~exclusive_signal)
    target = jnp.where(
        exclusive_signal,
        observed_target,
        jnp.where(infer_flip, _i32(1) - old[0], old[0]),
    )
    known = jnp.where(exclusive_signal | infer_flip, _i32(1), old[1])

    held_plate = obs[2, 2, 5] > 0
    held_dish = obs[2, 2, 6] > 0
    held_raw0 = obs[2, 2, 7] > 0
    held_raw1 = obs[2, 2, 8] > 0
    held_raw = held_raw0 | held_raw1
    held_target_raw = jnp.where(target == 0, held_raw0, held_raw1)

    pot_i0 = _cell(obs, self_r, self_c, _i32(2), _i32(4), 31)
    pot_i1 = _cell(obs, self_r, self_c, _i32(2), _i32(4), 32)
    pot_i2 = _cell(obs, self_r, self_c, _i32(2), _i32(4), 33)
    pot_total = pot_i0 + pot_i1 + pot_i2
    pot_cooked = _cell(obs, self_r, self_c, _i32(2), _i32(4), 30) > 0
    pot_timer = _cell(obs, self_r, self_c, _i32(2), _i32(4), 39)
    pot_busy = (pot_timer > 0) | (pot_total >= 3)

    go_pot = _go_interact(obs, self_r, self_c, _i32(2), _i32(5), DIR_LEFT)
    go_pot_from_raw = _go_interact_row_first(obs, self_r, self_c, _i32(2), _i32(5), DIR_LEFT)
    go_goal = _go_interact(obs, self_r, self_c, _i32(2), _i32(6), DIR_RIGHT)
    observe_signal = _move_to(self_r, self_c, _i32(2), _i32(5), STAY)
    wait_plate_at_pot = _move_to(
        self_r,
        self_c,
        _i32(2),
        _i32(5),
        jnp.where(_facing(obs) == DIR_LEFT, STAY, LEFT),
    )

    plate_row = _right_plate_row(self_r)
    get_plate = _go_interact(obs, self_r, self_c, plate_row, _i32(5), DIR_LEFT)

    supply_row = jnp.where(target == 0, _i32(1), _i32(3))
    get_ingredient = _go_interact_row_first(obs, self_r, self_c, supply_row, _i32(6), DIR_RIGHT)

    raw_action = jnp.where(
        held_target_raw & (pot_total < 3) & (pot_timer == 0) & (~pot_cooked),
        go_pot_from_raw,
        _drop_wrong_raw(obs, self_r, self_c),
    )
    empty_action = jnp.where(
        (known == 0) | (cooldown > 0),
        observe_signal,
        jnp.where(pot_cooked | pot_busy, get_plate, get_ingredient),
    )
    plate_action = jnp.where(pot_cooked, go_pot, wait_plate_at_pot)

    action = jnp.where(
        held_dish,
        go_goal,
        jnp.where(held_plate, plate_action, jnp.where(held_raw, raw_action, empty_action)),
    )

    at_goal = (self_r == 2) & (self_c == 6)
    delivery_attempt = held_dish & at_goal & (_facing(obs) == DIR_RIGHT) & (action == INTERACT)
    next_known = jnp.where(delivery_attempt, _i32(0), known)
    next_cooldown = jnp.where(delivery_attempt, _i32(1), jnp.maximum(cooldown - _i32(1), _i32(0)))
    next_resyncing = jnp.where(
        delivery_attempt,
        _i32(1),
        jnp.where(exclusive_signal | infer_flip, _i32(0), resyncing),
    )
    next_mem = jnp.stack([target, next_known, next_cooldown, next_resyncing]).astype(jnp.int32)

    trace = {
        'target': target,
        'known': known,
        'cooldown': cooldown,
        'pot_total': pot_total,
        'pot_cooked': pot_cooked.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_mem, trace
