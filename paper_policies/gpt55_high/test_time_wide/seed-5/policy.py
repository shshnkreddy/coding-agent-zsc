"""Hand-coded policy for the test_time_wide Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'positional_signal_cook',
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

TOP_POSITIONS = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4))
BOTTOM_POSITIONS = ((4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4))


def _features():
    grid = [[[0 for _ in range(10)] for _ in range(6)] for _ in range(7)]
    for r, c in [
        (0, 0), (0, 1), (0, 4), (0, 5), (3, 0), (3, 2), (3, 4), (3, 5),
        (5, 0), (5, 5), (6, 0), (6, 1), (6, 3), (6, 4), (6, 5),
    ]:
        grid[r][c][0] = 1
    grid[0][2][1] = 1
    grid[3][1][2] = 1
    grid[3][3][2] = 1
    grid[6][2][3] = 1
    grid[0][3][5] = 1
    grid[1][0][6] = 1
    grid[1][5][6] = 1
    grid[2][0][7] = 1
    grid[2][5][7] = 1
    grid[4][0][9] = 1
    grid[4][5][9] = 1
    return grid


def _make_crops(positions):
    base = _features()
    crops = []
    for pr, pc in positions:
        crop = []
        for lr in range(5):
            row = []
            for lc in range(5):
                gr = int(pr) + lr - 2
                gc = int(pc) + lc - 2
                if 0 <= gr < 7 and 0 <= gc < 6:
                    row.append(base[gr][gc])
                else:
                    row.append([0 for _ in range(10)])
            crop.append(row)
        crops.append(crop)
    return jnp.asarray(crops, dtype=jnp.int32)


LOOKOUT = (2, 2)
DELIVERY_STAND = (1, 2)
PLATE_STAND = (1, 3)
LEFT_POT_STAND_TOP = (2, 1)
RIGHT_POT_STAND_TOP = (2, 3)
LEFT_POT = (3, 1)
RIGHT_POT = (3, 3)
ING0_STAND = (1, 1)
ING1_STAND = (2, 4)
LEFT_SIGNAL = (4, 1)
RIGHT_SIGNAL = (4, 3)


def _pos(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _localize(obs, crops, positions):
    seen = obs[:, :, 22:32]
    mismatch = jnp.sum(jnp.abs(crops - seen[None, :, :, :]), axis=(1, 2, 3))
    return jnp.asarray(positions, dtype=jnp.int32)[jnp.argmin(mismatch)]


def _same_pos(a, b):
    return (a[0] == b[0]) & (a[1] == b[1])


def _move_towards(pos, target):
    action = STAY
    action = jnp.where(pos[1] < target[1], RIGHT, action)
    action = jnp.where(pos[1] > target[1], LEFT, action)
    same_col = pos[1] == target[1]
    action = jnp.where(same_col & (pos[0] < target[0]), DOWN, action)
    action = jnp.where(same_col & (pos[0] > target[0]), UP, action)
    return action


def _use_adjacent(pos, stand, face_action, face_ok):
    at_stand = _same_pos(pos, stand)
    return jnp.where(at_stand, jnp.where(face_ok, INTERACT, face_action), _move_towards(pos, stand))


def _cell_value(obs, pos, target, channel):
    lr = target[0] - pos[0] + 2
    lc = target[1] - pos[1] + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    lr_clip = jnp.clip(lr, 0, 4)
    lc_clip = jnp.clip(lc, 0, 4)
    return jnp.where(visible, obs[lr_clip, lc_clip, channel], jnp.asarray(0, dtype=obs.dtype))


def _top_memory():
    return jnp.zeros((3,), dtype=jnp.int32)


def init_agent_0():
    return _top_memory()


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _top_memory(), memory)
    phase = memory[0]
    recipe = memory[1]
    count = memory[2]

    pos = _localize(obs, _make_crops(TOP_POSITIONS), TOP_POSITIONS)
    face_up = obs[2, 2, 1] > 0
    face_down = obs[2, 2, 2] > 0
    face_left = obs[2, 2, 4] > 0
    face_right = obs[2, 2, 3] > 0
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ingredient_count = jnp.sum(obs[2, 2, 7:11])
    held_raw = (held_ingredient_count > 0) & (~held_plate) & (~held_cooked)
    held_dish = held_plate & held_cooked
    held_empty_plate = held_plate & (~held_cooked)

    partner_cells = obs[:, :, 11]
    has_partner = jnp.sum(partner_cells) > 0
    partner_flat = jnp.argmax(jnp.reshape(partner_cells, (25,)))
    partner_lr = partner_flat // 5
    partner_lc = partner_flat % 5
    partner_r = pos[0] + (partner_flat // 5) - 2
    partner_c = pos[1] + (partner_flat % 5) - 2
    partner_face_up = obs[partner_lr, partner_lc, 12] > 0
    partner_face_right = obs[partner_lr, partner_lc, 14] > 0
    partner_face_left = obs[partner_lr, partner_lc, 15] > 0
    intent_ready = count > 0
    intent_left = intent_ready & partner_face_left & (partner_c <= 2)
    intent_right = intent_ready & partner_face_right & (partner_c >= 2)
    settled_left = intent_ready & (partner_c <= 1) & partner_face_up
    settled_right = intent_ready & (partner_c >= 3) & partner_face_up
    signal0 = has_partner & (partner_r >= 4) & (settled_left | intent_left)
    signal1 = has_partner & (partner_r >= 4) & (settled_right | intent_right)
    has_signal = signal0 | signal1
    signaled_recipe = jnp.where(signal1, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    active_recipe = jnp.where((phase == 0) & has_signal, signaled_recipe, recipe)

    pot_stand = jnp.where(active_recipe == 0, _pos(LEFT_POT_STAND_TOP), _pos(RIGHT_POT_STAND_TOP))
    pot_cell = jnp.where(active_recipe == 0, _pos(LEFT_POT), _pos(RIGHT_POT))
    supply_stand = jnp.where(active_recipe == 0, _pos(ING0_STAND), _pos(ING1_STAND))
    supply_face_action = jnp.where(active_recipe == 0, LEFT, RIGHT)
    supply_face_ok = jnp.where(active_recipe == 0, face_left, face_right)

    get_ingredient = _use_adjacent(pos, supply_stand, supply_face_action, supply_face_ok)
    add_ingredient = _use_adjacent(pos, pot_stand, DOWN, face_down)
    phase1_action = jnp.where(held_raw, add_ingredient, get_ingredient)

    pot_cooked = _cell_value(obs, pos, pot_cell, 33) > 0
    get_plate = _use_adjacent(pos, _pos(PLATE_STAND), UP, face_up)
    pickup_plate = jnp.where(
        _same_pos(pos, pot_stand),
        jnp.where(pot_cooked, jnp.where(face_down, INTERACT, DOWN), STAY),
        _move_towards(pos, pot_stand),
    )
    deliver = _use_adjacent(pos, _pos(DELIVERY_STAND), UP, face_up)
    phase2_action = jnp.where(held_dish, deliver, jnp.where(held_empty_plate, pickup_plate, get_plate))

    wait_signal = jnp.where(_same_pos(pos, _pos(LOOKOUT)), STAY, _move_towards(pos, _pos(LOOKOUT)))
    action = jnp.where((phase == 1) | ((phase == 0) & has_signal), phase1_action, wait_signal)
    action = jnp.where(phase == 2, phase2_action, action)

    did_add = (phase == 1) & held_raw & _same_pos(pos, pot_stand) & face_down
    did_deliver = (phase == 2) & held_dish & _same_pos(pos, _pos(DELIVERY_STAND)) & face_up

    next_recipe = active_recipe
    next_count = jnp.where((phase == 0) & has_signal, jnp.asarray(0, dtype=jnp.int32), count)
    next_count = jnp.where(
        (phase == 0) & (~has_signal),
        jnp.minimum(count + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(9, dtype=jnp.int32)),
        next_count,
    )
    next_count = jnp.where(did_add, count + jnp.asarray(1, dtype=jnp.int32), next_count)
    next_count = jnp.where(did_deliver, jnp.asarray(0, dtype=jnp.int32), next_count)

    next_phase = phase
    next_phase = jnp.where((phase == 0) & has_signal, jnp.asarray(1, dtype=jnp.int32), next_phase)
    next_phase = jnp.where((phase == 1) & did_add & (count >= 2), jnp.asarray(2, dtype=jnp.int32), next_phase)
    next_phase = jnp.where((phase == 2) & did_deliver, jnp.asarray(0, dtype=jnp.int32), next_phase)

    next_memory = jnp.asarray([next_phase, next_recipe, next_count], dtype=jnp.int32)
    trace = {
        'phase': phase,
        'recipe': active_recipe,
        'count': count,
        'signal': has_signal.astype(jnp.int32),
    }
    return action, next_memory, trace


def _bottom_memory():
    return jnp.zeros((1,), dtype=jnp.int32)


def init_agent_1():
    return _bottom_memory()


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, _bottom_memory(), memory)
    pos = _localize(obs, _make_crops(BOTTOM_POSITIONS), BOTTOM_POSITIONS)
    recipe0_count = jnp.sum(obs[:, :, 40])
    recipe1_count = jnp.sum(obs[:, :, 41])
    recipe = jnp.where(recipe1_count > recipe0_count, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    target = jnp.where(recipe == 0, _pos(LEFT_SIGNAL), _pos(RIGHT_SIGNAL))
    face_up = obs[2, 2, 1] > 0
    at_target = _same_pos(pos, target)
    action = jnp.where(at_target, jnp.where(face_up, STAY, UP), _move_towards(pos, target))
    trace = {'recipe': recipe}
    return action, memory, trace
