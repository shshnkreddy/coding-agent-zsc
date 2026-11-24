"""Evidence-driven controller for the grounded_coord_simple layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'button_recipe_cook_loop',
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

DIR_RIGHT = 2
DIR_LEFT = 3

CANDIDATES_DATA = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
    (1, 5),
    (1, 6),
    (2, 5),
    (2, 6),
    (3, 5),
    (3, 6),
)

WALL_POS_DATA = (
    (0, 0),
    (0, 1),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (1, 0),
    (1, 3),
    (3, 0),
    (3, 3),
    (4, 0),
    (4, 1),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
)


def _init0():
    return jnp.zeros((6,), dtype=jnp.int32)


def _init1():
    return jnp.asarray([0, -1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def _static_map():
    wall_pos = jnp.asarray(WALL_POS_DATA, dtype=jnp.int32)
    static_map = jnp.zeros((5, 8, 9), dtype=jnp.int32)
    static_map = static_map.at[wall_pos[:, 0], wall_pos[:, 1], 0].set(1)
    static_map = static_map.at[2, 7, 1].set(1)
    static_map = static_map.at[2, 4, 2].set(1)
    static_map = static_map.at[2, 0, 3].set(1)
    static_map = static_map.at[2, 3, 4].set(1)
    static_map = static_map.at[jnp.asarray([1, 3]), jnp.asarray([4, 4]), 5].set(1)
    static_map = static_map.at[1, 7, 6].set(1)
    static_map = static_map.at[3, 7, 7].set(1)
    static_map = static_map.at[jnp.asarray([0, 4]), jnp.asarray([2, 2]), 8].set(1)
    return static_map


def _localize(obs):
    candidates = jnp.asarray(CANDIDATES_DATA, dtype=jnp.int32)
    local_r = jnp.arange(5, dtype=jnp.int32).reshape(1, 5, 1)
    local_c = jnp.arange(5, dtype=jnp.int32).reshape(1, 1, 5)
    static_map = _static_map()
    rows = candidates[:, 0:1, None] + local_r - 2
    cols = candidates[:, 1:2, None] + local_c - 2
    in_bounds = (rows >= 0) & (rows < 5) & (cols >= 0) & (cols < 8)
    expected = static_map[jnp.clip(rows, 0, 4), jnp.clip(cols, 0, 7), :]
    expected = jnp.where(in_bounds[..., None], expected, 0)
    observed = obs[None, :, :, 20:29]
    same = jnp.equal(observed, expected).astype(jnp.int32)
    scores = jnp.sum(same, axis=(1, 2, 3))
    return candidates[jnp.argmax(scores)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _move_toward(pos, target):
    row = pos[0]
    col = pos[1]
    target_row = target[0]
    target_col = target[1]
    action = jnp.where(col < target_col, RIGHT, STAY)
    action = jnp.where(col > target_col, LEFT, action)
    action = jnp.where((col == target_col) & (row < target_row), DOWN, action)
    action = jnp.where((col == target_col) & (row > target_row), UP, action)
    return action


def _move_toward_for_face(pos, target, face_dir):
    row = pos[0]
    col = pos[1]
    target_row = target[0]
    target_col = target[1]
    vertical_first = ((face_dir == DIR_RIGHT) & (col < target_col)) | (
        (face_dir == DIR_LEFT) & (col > target_col)
    )
    vertical_action = jnp.where(row < target_row, DOWN, jnp.where(row > target_row, UP, STAY))
    normal_action = _move_toward(pos, target)
    return jnp.where((row != target_row) & vertical_first, vertical_action, normal_action)


def _at(pos, target):
    return (pos[0] == target[0]) & (pos[1] == target[1])


def _go_interact(pos, obs, target, face_dir):
    face = _facing(obs)
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[face_dir]
    at_target = _at(pos, target)
    return jnp.where(
        at_target,
        jnp.where(face == face_dir, INTERACT, face_action),
        _move_toward_for_face(pos, target, face_dir),
    )


def _held_counts(obs):
    return obs[2, 2, 7:10]


def _held_plate(obs):
    return obs[2, 2, 5] > 0


def _held_cooked(obs):
    return obs[2, 2, 6] > 0


def _recipe_seen(obs):
    count0 = jnp.sum(obs[:, :, 36])
    count1 = jnp.sum(obs[:, :, 37])
    seen = (count0 + count1) > 0
    recipe = jnp.where(count1 > count0, 1, 0).astype(jnp.int32)
    return seen, recipe


def _pot_occupied(obs):
    pot = obs[:, :, 22] > 0
    contents = obs[:, :, 30] + obs[:, :, 31] + obs[:, :, 32] + obs[:, :, 33] + obs[:, :, 39]
    return jnp.sum(jnp.where(pot, contents, 0)) > 0


def _pot_ready(obs):
    pot = obs[:, :, 22] > 0
    cooked = obs[:, :, 30] > 0
    return jnp.any(pot & cooked)


def init_agent_0():
    return _init0()


def agent_0_act(obs, reset, memory, rng):
    del rng
    base = _init0()
    memory = jnp.where(reset, base, memory)

    pos = _localize(obs)
    target = jnp.asarray([2, 2], dtype=jnp.int32)
    face = _facing(obs)
    at_button_cell = _at(pos, target)
    pot_has = _pot_occupied(obs)

    step = memory[0]
    cooldown = memory[1]
    previous_pot_has = memory[2] > 0
    initial_done = memory[3] > 0
    pickup_event = previous_pot_has & (~pot_has)
    fallback = initial_done & (step > 0) & (~pot_has)
    want_press = (~initial_done) | pickup_event | fallback
    allowed_press = want_press & (cooldown <= 0)

    press_action = jnp.where(face == DIR_RIGHT, INTERACT, RIGHT)
    action = jnp.where(
        at_button_cell,
        jnp.where(allowed_press, press_action, STAY),
        _move_toward(pos, target),
    ).astype(jnp.int32)

    did_press = at_button_cell & allowed_press & (face == DIR_RIGHT)
    next_memory = memory
    next_memory = next_memory.at[0].set(step + 1)
    next_memory = next_memory.at[1].set(jnp.where(did_press, 12, jnp.maximum(cooldown - 1, 0)))
    next_memory = next_memory.at[2].set(pot_has.astype(jnp.int32))
    next_memory = next_memory.at[3].set((initial_done | did_press).astype(jnp.int32))
    return action, next_memory, {
        'pos_r': pos[0],
        'pos_c': pos[1],
        'press': did_press,
        'pot_has': pot_has,
    }


def init_agent_1():
    return _init1()


def agent_1_act(obs, reset, memory, rng):
    del rng
    base = _init1()
    memory = jnp.where(reset, base, memory)

    pos = _localize(obs)
    phase = memory[0]
    recipe = memory[1]
    added = memory[2]
    step = memory[4]
    seen, visible_recipe = _recipe_seen(obs)
    recipe = jnp.where(seen, visible_recipe, recipe)
    have_recipe = recipe >= 0

    held = _held_counts(obs)
    desired_held = jnp.where(recipe == 0, held[0] > 0, held[1] > 0)
    plate = _held_plate(obs)
    cooked = _held_cooked(obs)
    ready = _pot_ready(obs)

    phase = jnp.where((phase == 0) & have_recipe, 1, phase)
    phase = jnp.where((phase == 1) & desired_held, 2, phase)
    phase = jnp.where((phase == 3) & ready, 4, phase)
    phase = jnp.where((phase == 4) & plate, 5, phase)
    phase = jnp.where((phase == 5) & cooked, 6, phase)

    recipe_target = jnp.asarray([2, 5], dtype=jnp.int32)
    ing0_target = jnp.asarray([1, 6], dtype=jnp.int32)
    ing1_target = jnp.asarray([3, 6], dtype=jnp.int32)
    pot_target = jnp.asarray([2, 5], dtype=jnp.int32)
    plate_target = jnp.asarray([1, 5], dtype=jnp.int32)
    goal_target = jnp.asarray([2, 6], dtype=jnp.int32)
    ing_target = jnp.where(recipe == 0, ing0_target, ing1_target)

    action_seek = jnp.where(have_recipe, STAY, _move_toward(pos, recipe_target))
    action_get_ing = _go_interact(pos, obs, ing_target, DIR_RIGHT)
    action_add_ing = _go_interact(pos, obs, pot_target, DIR_LEFT)
    action_wait = jnp.where(ready, STAY, _move_toward(pos, pot_target))
    action_get_plate = _go_interact(pos, obs, plate_target, DIR_LEFT)
    action_pickup = _go_interact(pos, obs, pot_target, DIR_LEFT)
    action_deliver = _go_interact(pos, obs, goal_target, DIR_RIGHT)

    action = action_seek
    action = jnp.where(phase == 1, action_get_ing, action)
    action = jnp.where(phase == 2, action_add_ing, action)
    action = jnp.where(phase == 3, action_wait, action)
    action = jnp.where(phase == 4, action_get_plate, action)
    action = jnp.where(phase == 5, action_pickup, action)
    action = jnp.where(phase == 6, action_deliver, action).astype(jnp.int32)

    face = _facing(obs)
    interacted = action == INTERACT
    did_get_ing = (phase == 1) & interacted & _at(pos, ing_target) & (face == DIR_RIGHT)
    did_add = (phase == 2) & interacted & _at(pos, pot_target) & (face == DIR_LEFT)
    did_get_plate = (phase == 4) & interacted & _at(pos, plate_target) & (face == DIR_LEFT)
    did_pickup = (phase == 5) & interacted & _at(pos, pot_target) & (face == DIR_LEFT) & ready
    did_deliver = (phase == 6) & interacted & _at(pos, goal_target) & (face == DIR_RIGHT)

    new_added = jnp.where(did_add, added + 1, added)
    next_phase = phase
    next_phase = jnp.where(did_get_ing, 2, next_phase)
    next_phase = jnp.where(did_add & (new_added < 3), 1, next_phase)
    next_phase = jnp.where(did_add & (new_added >= 3), 4, next_phase)
    next_phase = jnp.where((phase == 3) & ready, 4, next_phase)
    next_phase = jnp.where(did_get_plate, 5, next_phase)
    next_phase = jnp.where(did_pickup, 6, next_phase)
    next_phase = jnp.where(did_deliver, 0, next_phase)

    next_recipe = jnp.where(did_deliver, -1, recipe)
    next_added = jnp.where(did_deliver | did_pickup, 0, new_added)
    next_memory = memory
    next_memory = next_memory.at[0].set(next_phase.astype(jnp.int32))
    next_memory = next_memory.at[1].set(next_recipe.astype(jnp.int32))
    next_memory = next_memory.at[2].set(next_added.astype(jnp.int32))
    next_memory = next_memory.at[4].set(step + 1)
    return action, next_memory, {
        'pos_r': pos[0],
        'pos_c': pos[1],
        'phase': phase,
        'recipe': recipe,
        'added': added,
        'ready': ready,
    }
