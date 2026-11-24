"""Layout-specific cooperative policy for the supplied Overcooked V2 task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'visible_pot_side_recipe_policy',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

FACE_UP = 0
FACE_DOWN = 1
FACE_RIGHT = 2
FACE_LEFT = 3
UNKNOWN = -1

TOP_CELLS_LIT = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4))
BOT_CELLS_LIT = ((4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4))
STATIC_ENTRIES = (
    ((0, 0), 0), ((0, 1), 0), ((0, 4), 0), ((0, 5), 0),
    ((3, 0), 0), ((3, 2), 0), ((3, 4), 0), ((3, 5), 0),
    ((5, 0), 0), ((5, 5), 0), ((6, 0), 0), ((6, 1), 0),
    ((6, 3), 0), ((6, 4), 0), ((6, 5), 0),
    ((0, 2), 1), ((3, 1), 2), ((3, 3), 2), ((6, 2), 3),
    ((0, 3), 5), ((1, 0), 6), ((1, 5), 6), ((2, 0), 7),
    ((2, 5), 7), ((4, 0), 9), ((4, 5), 9),
)


def _static_value(row, col, chan):
    value = 0
    for (rr, cc), ch in STATIC_ENTRIES:
        if row == rr and col == cc and chan == ch:
            value = 1
    return value


def _make_crops(cells_lit):
    crops = []
    for row, col in cells_lit:
        rows = []
        for dr in range(-2, 3):
            cols = []
            for dc in range(-2, 3):
                rr = row + dr
                cc = col + dc
                if 0 <= rr < 7 and 0 <= cc < 6:
                    cols.append([_static_value(rr, cc, ch) for ch in range(10)])
                else:
                    cols.append([0 for _ in range(10)])
            rows.append(cols)
        crops.append(rows)
    return jnp.asarray(crops, dtype=jnp.int32)


def _locate(obs, cells_lit):
    cells = jnp.asarray(cells_lit, dtype=jnp.int32)
    crops = _make_crops(cells_lit)
    visible_static = obs[:, :, 22:32]
    diffs = jnp.sum(jnp.abs(crops - visible_static[None, :, :, :]), axis=(1, 2, 3))
    idx = jnp.argmin(diffs).astype(jnp.int32)
    return cells[idx]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _held(obs):
    inv = obs[2, 2, 5:11]
    has_plate = inv[0] > 0
    has_cooked = inv[1] > 0
    has_ing = jnp.sum(inv[2:]) > 0
    return has_plate, has_cooked, has_ing


def _cell_value(obs, self_pos, global_cell, chan):
    local = global_cell - self_pos + 2
    row = local[0]
    col = local[1]
    in_view = (row >= 0) & (row < 5) & (col >= 0) & (col < 5)
    safe_row = jnp.clip(row, 0, 4)
    safe_col = jnp.clip(col, 0, 4)
    return jnp.where(in_view, obs[safe_row, safe_col, chan], 0)


def _move_toward(pos, target):
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    horizontal = jnp.where(dc > 0, RIGHT, LEFT)
    vertical = jnp.where(dr > 0, DOWN, UP)
    return jnp.where(dc != 0, horizontal, vertical)


def _go_face_or_interact(pos, face, target, desired_face):
    at_target = jnp.all(pos == target)
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[desired_face]
    return jnp.where(
        at_target,
        jnp.where(face == desired_face, INTERACT, face_action),
        _move_toward(pos, target),
    )


def _go_face_or_stay(pos, face, target, desired_face):
    at_target = jnp.all(pos == target)
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[desired_face]
    return jnp.where(
        at_target,
        jnp.where(face == desired_face, STAY, face_action),
        _move_toward(pos, target),
    )


def init_agent_0():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    top_watch = jnp.asarray([2, 2], dtype=jnp.int32)
    top_deliver = jnp.asarray([1, 2], dtype=jnp.int32)
    top_plate = jnp.asarray([1, 3], dtype=jnp.int32)
    pot_stands = jnp.asarray([[2, 1], [2, 3]], dtype=jnp.int32)
    pot_cells = jnp.asarray([[3, 1], [3, 3]], dtype=jnp.int32)
    supply_stands = jnp.asarray([[1, 1], [2, 4]], dtype=jnp.int32)
    signal_stands = jnp.asarray([[4, 1], [4, 3]], dtype=jnp.int32)

    pos = _locate(obs, TOP_CELLS_LIT)
    face = _facing(obs)
    has_plate, has_cooked, has_ing = _held(obs)

    prev_side = jnp.where(reset, UNKNOWN, memory[0])
    cooldown = jnp.where(reset, 1, memory[1])
    partner_visible = jnp.sum(obs[:, :, 11]) > 0
    partner_idx = jnp.argmax(jnp.reshape(obs[:, :, 11], (-1,))).astype(jnp.int32)
    prow = partner_idx // 5
    pcol = partner_idx % 5
    partner_up = obs[prow, pcol, 12] > 0
    partner_global = pos + jnp.asarray([prow - 2, pcol - 2], dtype=jnp.int32)
    left_signal = partner_visible & partner_up & jnp.all(partner_global == signal_stands[0])
    right_signal = partner_visible & partner_up & jnp.all(partner_global == signal_stands[1])
    signal_seen = left_signal | right_signal
    signal_side = jnp.where(right_signal, 1, 0)
    can_accept_signal = (cooldown <= 0) & signal_seen
    observed_side = jnp.where(can_accept_signal, signal_side, prev_side)
    side = jnp.clip(observed_side, 0, 1)
    side_known = observed_side >= 0

    pot_cell = pot_cells[side]
    pot_stand = pot_stands[side]
    supply_stand = supply_stands[side]

    pot_cooked = _cell_value(obs, pos, pot_cell, 33) > 0
    pot_ings = (
        _cell_value(obs, pos, pot_cell, 34)
        + _cell_value(obs, pos, pot_cell, 35)
        + _cell_value(obs, pos, pot_cell, 36)
        + _cell_value(obs, pos, pot_cell, 37)
    )
    pot_timer = _cell_value(obs, pos, pot_cell, 44)
    pot_loading = (pot_ings < 3) & (~pot_cooked) & (pot_timer == 0)

    wait_action = _go_face_or_stay(pos, face, top_watch, FACE_DOWN)
    deliver_action = _go_face_or_interact(pos, face, top_deliver, FACE_UP)
    plate_action = _go_face_or_interact(pos, face, top_plate, FACE_UP)
    pot_action = _go_face_or_interact(pos, face, pot_stand, FACE_DOWN)
    supply_action = _go_face_or_interact(
        pos,
        face,
        supply_stand,
        jnp.where(side == 0, FACE_LEFT, FACE_RIGHT),
    )

    action = jnp.where(
        has_cooked,
        deliver_action,
        jnp.where(
            ~side_known,
            wait_action,
            jnp.where(
                has_plate | has_ing,
                pot_action,
                jnp.where(pot_loading, supply_action, plate_action),
            ),
        ),
    ).astype(jnp.int32)

    delivered_now = (
        has_cooked
        & jnp.all(pos == top_deliver)
        & (face == FACE_UP)
        & (action == INTERACT)
    )
    decayed_cooldown = jnp.maximum(cooldown - 1, 0)
    next_side = jnp.where(reset | delivered_now, UNKNOWN, observed_side)
    next_cooldown = jnp.where(delivered_now, 1, decayed_cooldown)
    return action, jnp.asarray([next_side, next_cooldown], dtype=jnp.int32), {
        'side': next_side,
        'pot_ings': pot_ings.astype(jnp.int32),
        'pot_timer': pot_timer.astype(jnp.int32),
    }


def init_agent_1():
    return jnp.asarray([0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del reset, memory, rng
    signal_stands = jnp.asarray([[4, 1], [4, 3]], dtype=jnp.int32)
    pos = _locate(obs, BOT_CELLS_LIT)
    face = _facing(obs)
    recipe0 = jnp.sum(obs[:, :, 40])
    recipe1 = jnp.sum(obs[:, :, 41])
    side = jnp.where(recipe1 > recipe0, 1, 0).astype(jnp.int32)
    target = signal_stands[side]
    action = _go_face_or_stay(pos, face, target, FACE_UP).astype(jnp.int32)
    return action, jnp.asarray([side], dtype=jnp.int32), {'side': side}
