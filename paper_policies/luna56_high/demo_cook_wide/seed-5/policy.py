"""A geometry-driven, seat-specialized cooperative controller."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'seat_specialized_cycle',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}


# Static object classes in channels 22:32. This localizes the lower-room seat
# at reset; later positions are tracked from the chosen movement action.
_STATIC_CODE = (
    [1, 1, 1, 1, 6, 2, 6, 1, 1, 1, 1],
    [1, 1, 1, 7, 0, 0, 0, 8, 1, 1, 1],
    [1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [7, 0, 0, 1, 10, 4, 10, 1, 0, 0, 7],
    [1, 8, 1, 1, 1, 1, 1, 1, 1, 8, 1],
)
_LOWER_CANDIDATES = (
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 1), (4, 2), (4, 8), (4, 9),
)
_STATIC_WEIGHTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def _static_code(obs):
    weights = jnp.asarray(_STATIC_WEIGHTS, dtype=jnp.int32)
    return jnp.sum(obs[:, :, 22:32] * weights, axis=-1)


def _localize_lower(obs):
    static_code = jnp.asarray(_STATIC_CODE, dtype=jnp.int32)
    padded_static = jnp.pad(static_code, ((2, 2), (2, 2)))
    candidates = jnp.asarray(_LOWER_CANDIDATES, dtype=jnp.int32)
    templates = jnp.stack([
        padded_static[r:r + 5, c:c + 5] for r, c in _LOWER_CANDIDATES
    ])
    code = _static_code(obs)
    error = jnp.sum(jnp.abs(templates - code[None, :, :]), axis=(1, 2))
    return candidates[jnp.argmin(error)]


def _lower_move_action(row, col, target_row, target_col):
    return jnp.where(
        row < target_row, 1,
        jnp.where(row > target_row, 3,
                  jnp.where(col < target_col, 0,
                            jnp.where(col > target_col, 2, 4))))


def _lower_move_to_supply(row, col):
    # Column 5's recipe marker and the other row-4 cells are blocked, so the
    # left supply is reached by traversing row 3 first, then descending at c1.
    return jnp.where(col > 1, 2,
                     jnp.where(row < 4, 1, 4))


def _lower_position_after(row, col, action):
    can_right = ((row == 3) & (col < 9)) | ((row == 4) & (col == 8))
    can_left = ((row == 3) & (col > 1)) | ((row == 4) & (col == 9))
    can_up = (row == 4) & ((col == 1) | (col == 2) | (col == 8) | (col == 9))
    can_down = (row == 3) & ((col == 1) | (col == 2) | (col == 8) | (col == 9))
    next_row = row + jnp.where((action == 1) & can_down, 1,
                               jnp.where((action == 3) & can_up, -1, 0))
    next_col = col + jnp.where((action == 0) & can_right, 1,
                               jnp.where((action == 2) & can_left, -1, 0))
    return next_row, next_col


def init_agent_0():
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    # The always-visible upper goal identifies the initial global column.
    goal_flat = jnp.argmax(obs[:, :, 23].reshape((-1,)))
    goal_row, goal_col = goal_flat // 5, goal_flat % 5
    initial = jnp.asarray([2 - goal_row, 7 - goal_col, 0], dtype=jnp.int32)
    state = jnp.where(reset, initial, memory)
    row, col, stage = state[0], state[1], state[2]

    plate_held = obs[2, 2, 5] > 0
    cooked_held = obs[2, 2, 6] > 0
    at_pot_stand = (row == 1) & (col == 5)
    pot_ready = at_pot_stand & (obs[3, 2, 33] > 0)
    facing = jnp.argmax(obs[2, 2, 1:5])

    effective = jnp.where((stage == 0) & plate_held, 1, stage)
    effective = jnp.where((stage == 1) & cooked_held, 2, effective)
    effective = jnp.where((stage == 2) & (~cooked_held), 0, effective)

    # Stage 0: obtain an empty plate from the left pile.
    plate_action = jnp.where(
        col < 4, 0,
        jnp.where(col > 4, 2,
                  jnp.where(facing == 0, 5, 3)))
    # Stage 1: wait beside the pot, then collect when it is cooked.
    pot_action = jnp.where(
        col < 5, 0,
        jnp.where(col > 5, 2,
                  jnp.where(facing != 1, 1,
                            jnp.where(pot_ready, 5, 4))))
    # Stage 2: the goal is directly above the same standing cell.
    delivery_action = jnp.where(facing == 0, 5, 3)
    action = jnp.where(effective == 0, plate_action,
                       jnp.where(effective == 1, pot_action, delivery_action))

    next_col = col + jnp.where((action == 0) & (col < 6), 1,
                               jnp.where((action == 2) & (col > 4), -1, 0))
    next_state = jnp.asarray([row, next_col, effective], dtype=jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_state, {}


def init_agent_1():
    return jnp.zeros((5,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    initial_position = _localize_lower(obs)
    initial = jnp.asarray([initial_position[0], initial_position[1], 0, 0, 0], dtype=jnp.int32)
    state = jnp.where(reset, initial, memory)
    row, col, stage = state[0], state[1], state[2]
    remembered_recipe, added = state[3], state[4]

    recipe_visible = jnp.max(obs[:, :, 38:44]) > 0
    recipe_is_one = jnp.max(obs[:, :, 41]) > 0
    recipe = jnp.where(recipe_visible, recipe_is_one.astype(jnp.int32), remembered_recipe)
    facing = jnp.argmax(obs[2, 2, 1:5])
    held_zero = obs[2, 2, 7]
    held_one = obs[2, 2, 8]
    held = jnp.where(recipe == 1, held_one, held_zero)
    pot_empty = jnp.sum(obs[1, 2, 32:38]) == 0
    added_after_insert = jnp.where((stage == 4) & (held == 0), added + 1, added)

    at_recipe = (row == 3) & (col == 5)
    at_supply = (row == 4) & (col == 1)
    at_pot = at_recipe
    effective = jnp.where((stage == 0) & at_recipe & recipe_visible, 1, stage)
    effective = jnp.where((stage == 1) & at_supply, 2, effective)
    effective = jnp.where((stage == 2) & (held > 0), 3, effective)
    effective = jnp.where((stage == 3) & at_pot, 4, effective)
    effective = jnp.where(
        (stage == 4) & (held == 0),
        jnp.where(added_after_insert >= 3, 5, 1), effective)
    effective = jnp.where((stage == 5) & pot_empty, 0, effective)

    move_to_recipe = _lower_move_action(row, col, 3, 5)
    move_to_supply = _lower_move_to_supply(row, col)
    move_to_pot = _lower_move_action(row, col, 3, 5)
    desired_face = jnp.where(recipe == 1, 1, 3)  # down for ingredient 1, left for 0
    collect_action = jnp.where(
        facing == desired_face, 5,
        jnp.where(desired_face == 1, 1, 2))
    add_action = jnp.where(facing == 0, 5, 3)
    action = jnp.where(effective == 0, move_to_recipe,
                       jnp.where(effective == 1, move_to_supply,
                                 jnp.where(effective == 2, collect_action,
                                           jnp.where(effective == 3, move_to_pot,
                                                     jnp.where(effective == 4, add_action,
                                                               jnp.asarray(4, dtype=jnp.int32))))))

    next_row, next_col = _lower_position_after(row, col, action)
    next_added = jnp.where((stage == 5) & pot_empty, 0, added_after_insert)
    next_state = jnp.asarray(
        [next_row, next_col, effective, recipe, next_added], dtype=jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_state, {}
