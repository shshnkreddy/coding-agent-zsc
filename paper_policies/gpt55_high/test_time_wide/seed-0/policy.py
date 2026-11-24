"""Object-reactive Overcooked policy for the test_time_wide layout."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'visible_recipe_pot_service',
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

TOP_POSITIONS = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4))
BOTTOM_POSITIONS = ((4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4))


def _make_static():
    grid = jnp.zeros((7, 6, 10), dtype=jnp.int32)
    walls = jnp.asarray(
        [
            [0, 0], [0, 1], [0, 4], [0, 5], [3, 0], [3, 2], [3, 4], [3, 5],
            [5, 0], [5, 5], [6, 0], [6, 1], [6, 3], [6, 4], [6, 5],
        ],
        dtype=jnp.int32,
    )
    pots = jnp.asarray([[3, 1], [3, 3]], dtype=jnp.int32)
    grid = grid.at[walls[:, 0], walls[:, 1], 0].set(1)
    grid = grid.at[0, 2, 1].set(1)
    grid = grid.at[pots[:, 0], pots[:, 1], 2].set(1)
    grid = grid.at[6, 2, 3].set(1)
    grid = grid.at[0, 3, 5].set(1)
    grid = grid.at[jnp.asarray([1, 1]), jnp.asarray([0, 5]), 6].set(1)
    grid = grid.at[jnp.asarray([2, 2]), jnp.asarray([0, 5]), 7].set(1)
    grid = grid.at[jnp.asarray([4, 4]), jnp.asarray([0, 5]), 9].set(1)
    return grid

def _infer_pos(obs, candidates):
    crop = obs[:, :, 22:32]
    static_grid = _make_static()
    local_r = jnp.arange(5, dtype=jnp.int32)[:, None]
    local_c = jnp.arange(5, dtype=jnp.int32)[None, :]
    candidates = jnp.asarray(candidates, dtype=jnp.int32)

    def score(pos):
        gr = pos[0] + local_r - 2
        gc = pos[1] + local_c - 2
        inside = (gr >= 0) & (gr < 7) & (gc >= 0) & (gc < 6)
        grc = jnp.clip(gr, 0, 6)
        gcc = jnp.clip(gc, 0, 5)
        expected = static_grid[grc, gcc, :]
        expected = jnp.where(inside[:, :, None], expected, 0)
        return jnp.sum(jnp.abs(crop - expected))

    scores = jax.vmap(score)(candidates)
    return candidates[jnp.argmin(scores)]


def _self_facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_action(face):
    return jnp.asarray(jnp.array([3, 1, 0, 2], dtype=jnp.int32)[face], dtype=jnp.int32)


def _held_kind(obs):
    center = obs[2, 2, :]
    has_plate = center[5] > 0
    has_cooked = center[6] > 0
    has_i0 = center[7] > 0
    has_i1 = center[8] > 0
    return has_plate, has_cooked, has_i0, has_i1


def _cell(obs, self_pos, gr, gc, ch):
    lr = gr - self_pos[0] + 2
    lc = gc - self_pos[1] + 2
    inside = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    value = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), ch]
    return jnp.where(inside, value, jnp.asarray(0, dtype=obs.dtype))


def _move_toward(pos, target):
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    horizontal = jnp.where(dc > 0, RIGHT, LEFT)
    vertical = jnp.where(dr > 0, DOWN, UP)
    return jnp.where(dc != 0, horizontal, jnp.where(dr != 0, vertical, STAY))


def _go_face_interact(pos, facing, stand, face):
    at_target = (pos[0] == stand[0]) & (pos[1] == stand[1])
    turn = _face_action(face)
    return jnp.where(at_target, jnp.where(facing == face, INTERACT, turn), _move_toward(pos, stand))


def _go_face_wait(pos, facing, stand, face):
    at_target = (pos[0] == stand[0]) & (pos[1] == stand[1])
    turn = _face_action(face)
    return jnp.where(at_target, jnp.where(facing == face, STAY, turn), _move_toward(pos, stand))


def _partner_signal(obs, self_pos, recipe, age):
    visible = jnp.max(obs[:, :, 11]) > 0
    flat = jnp.argmax(obs[:, :, 11]).astype(jnp.int32)
    lr = flat // 5
    lc = flat - lr * 5
    pr = self_pos[0] + lr - 2
    pc = self_pos[1] + lc - 2
    partner_up = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), 12] > 0
    partner_right = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), 14] > 0
    partner_left = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), 15] > 0
    stable_signal = visible & partner_up
    directional_signal = visible & (age > 0) & (pr == 4)
    signal0 = stable_signal & (pr == 4) & (pc == 1)
    signal1 = stable_signal & (pr == 4) & (pc == 3)
    early0 = directional_signal & partner_left
    early1 = directional_signal & partner_right
    return jnp.where(signal0 | early0, 0, jnp.where(signal1 | early1, 1, recipe)).astype(jnp.int32)


def _recipe_from_indicator(obs, self_pos, old_recipe):
    r0 = _cell(obs, self_pos, 6, 2, 40)
    r1 = _cell(obs, self_pos, 6, 2, 41)
    seen = (r0 + r1) > 0
    recipe = jnp.where(r1 > r0, 1, 0).astype(jnp.int32)
    return jnp.where(seen, recipe, old_recipe).astype(jnp.int32)


def _pot_values(obs, pos, recipe):
    pot_col = jnp.where(recipe == 0, 1, 3)
    i0 = _cell(obs, pos, 3, pot_col, 34)
    i1 = _cell(obs, pos, 3, pot_col, 35)
    cooked = _cell(obs, pos, 3, pot_col, 33) > 0
    timer = _cell(obs, pos, 3, pot_col, 44)
    target_count = jnp.where(recipe == 0, i0, i1)
    other_count = jnp.where(recipe == 0, i1, i0)
    return pot_col, target_count, other_count, cooked, timer


def init_agent_0():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    base = init_agent_0()
    memory = jnp.where(reset, base, memory)
    pos = _infer_pos(obs, TOP_POSITIONS)
    facing = _self_facing(obs)
    recipe = _partner_signal(obs, pos, memory[0], memory[1])

    has_plate, has_cooked, has_i0, has_i1 = _held_kind(obs)
    known = recipe >= 0
    pot_col, target_count, other_count, pot_cooked, timer = _pot_values(obs, pos, jnp.maximum(recipe, 0))
    pot_ready = known & pot_cooked & (target_count == 3) & (other_count == 0)
    pot_needs = known & (~pot_cooked) & (timer == 0) & (target_count < 3) & (other_count == 0)
    pot_busy = known & ((timer > 0) | pot_ready)

    observe = jnp.asarray([2, 2], dtype=jnp.int32)
    deliver_stand = jnp.asarray([1, 2], dtype=jnp.int32)
    plate_stand = jnp.asarray([1, 3], dtype=jnp.int32)
    pot_stand = jnp.asarray([2, pot_col], dtype=jnp.int32)
    left_supply = jnp.asarray([jnp.where(recipe == 0, 1, 2), 1], dtype=jnp.int32)
    right_supply = jnp.asarray([jnp.where(recipe == 0, 1, 2), 4], dtype=jnp.int32)
    supply_stand = jnp.where(pot_col == 1, left_supply, right_supply)
    supply_face = jnp.where(pot_col == 1, FACE_LEFT, FACE_RIGHT)

    raw_matches = ((recipe == 0) & has_i0) | ((recipe == 1) & has_i1)
    raw_action = _go_face_interact(pos, facing, pot_stand, FACE_DOWN)
    supply_action = _go_face_interact(pos, facing, supply_stand, supply_face)
    plate_action = _go_face_interact(pos, facing, plate_stand, FACE_UP)
    wait_pot_action = _go_face_wait(pos, facing, pot_stand, FACE_DOWN)
    scoop_action = _go_face_interact(pos, facing, pot_stand, FACE_DOWN)
    observe_action = _go_face_wait(pos, facing, observe, FACE_DOWN)
    drop_action = _go_face_interact(pos, facing, observe, FACE_DOWN)
    deliver_action = _go_face_interact(pos, facing, deliver_stand, FACE_UP)

    action = observe_action
    empty = ~(has_plate | has_cooked | has_i0 | has_i1)
    raw_mismatch = known & (has_i0 | has_i1) & (~raw_matches)
    action = jnp.where(known & pot_needs & empty, supply_action, action)
    action = jnp.where(known & raw_matches, raw_action, action)
    action = jnp.where(known & pot_busy & empty, plate_action, action)
    action = jnp.where(known & has_plate, jnp.where(pot_ready, scoop_action, wait_pot_action), action)
    action = jnp.where(raw_mismatch, drop_action, action)
    action = jnp.where(has_cooked, deliver_action, action)

    delivering = has_cooked & (pos[0] == deliver_stand[0]) & (pos[1] == deliver_stand[1]) & (facing == FACE_UP)
    next_recipe = jnp.where(delivering, UNKNOWN, recipe)
    next_memory = jnp.asarray([next_recipe, memory[1] + 1], dtype=jnp.int32)
    next_memory = jnp.where(reset, base, next_memory)
    trace = {'recipe': next_recipe, 'row': pos[0], 'col': pos[1], 'action': action}
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    return jnp.asarray([-1, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    base = init_agent_1()
    memory = jnp.where(reset, base, memory)
    pos = _infer_pos(obs, BOTTOM_POSITIONS)
    facing = _self_facing(obs)
    recipe = _recipe_from_indicator(obs, pos, memory[0])
    signal_stand = jnp.asarray([4, jnp.where(recipe == 0, 1, 3)], dtype=jnp.int32)
    action = _go_face_wait(pos, facing, signal_stand, FACE_UP)
    next_memory = jnp.asarray([recipe, memory[1] + 1], dtype=jnp.int32)
    next_memory = jnp.where(reset, base, next_memory)
    trace = {'recipe': recipe, 'row': pos[0], 'col': pos[1], 'action': action}
    return action.astype(jnp.int32), next_memory, trace
