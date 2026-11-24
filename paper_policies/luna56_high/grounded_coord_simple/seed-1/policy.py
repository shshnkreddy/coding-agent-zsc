"""Grounded reactive policy for the small two-room Overcooked layout."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'recipe_signaled_autonomous_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Static channels 20:29 of the resolved map, used to localize the crop.
# Keep this as a literal because the policy loader disallows module-scope calls.
_STATIC_ROWS = (
    ((1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,1), (1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0)),
    ((1,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (0,0,0,0,0,1,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,1,0,0)),
    ((0,0,0,1,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,1,0,0,0,0), (0,0,1,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (0,1,0,0,0,0,0,0,0)),
    ((1,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (0,0,0,0,0,1,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,1,0)),
    ((1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0,1), (1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0)),
)


def _position(obs):
    static_map = jnp.asarray(_STATIC_ROWS, dtype=jnp.int32)
    padded = jnp.pad(static_map, ((2, 2), (2, 2), (0, 0)))
    candidate_r = jnp.repeat(jnp.arange(5), 8)
    candidate_c = jnp.tile(jnp.arange(8), 5)
    off = jnp.arange(5) - 2
    expected = padded[
        candidate_r[:, None, None] + off[None, :, None] + 2,
        candidate_c[:, None, None] + off[None, None, :] + 2,
    ]
    static = obs[:, :, 20:29]
    errors = jnp.sum((expected - static[None, :, :, :]) ** 2, axis=(1, 2, 3))
    ix = jnp.argmin(errors)
    return candidate_r[ix], candidate_c[ix]


def _go(row, col, target_row, target_col):
    return jnp.where(
        row < target_row, 1,
        jnp.where(row > target_row, 3,
                  jnp.where(col < target_col, 0,
                            jnp.where(col > target_col, 2, 4))))


def _recipe_from_visible(obs, fallback):
    has_zero = jnp.sum(obs[:, :, 36]) > 0
    has_one = jnp.sum(obs[:, :, 37]) > 0
    has_recipe = has_zero | has_one
    visible = jnp.where(has_one, jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray(0, dtype=jnp.int32))
    return jnp.where(has_recipe, visible, fallback), has_recipe


def init_agent_0():
    return jnp.asarray([-1, 1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    row, col = _position(obs)
    at_keeper = (row == 2) & (col == 2)
    reset_memory = jnp.asarray([-1, 1, 0], dtype=jnp.int32)
    old = jnp.where(reset, reset_memory, memory)
    old_recipe = old[0]
    old_pending = old[1] > 0
    old_clock = old[2]
    recipe, has_recipe = _recipe_from_visible(obs, old_recipe)
    changed = has_recipe & (recipe != old_recipe)
    periodic = old_clock >= 1000
    pending = old_pending | changed | periodic
    face_right = obs[2, 2, 3] > 0
    change_action = jnp.where(face_right, 5, 0)
    move_action = _go(row, col, 2, 2)
    activate = at_keeper & pending
    action = jnp.where(activate, change_action,
                       jnp.where(at_keeper, jnp.asarray(4, dtype=jnp.int32), move_action))
    cleared = activate & face_right
    next_pending = jnp.where(cleared, jnp.asarray(0, dtype=jnp.int32), pending.astype(jnp.int32))
    next_clock = jnp.where(cleared, jnp.asarray(0, dtype=jnp.int32), old_clock + 1)
    next_memory = jnp.asarray([recipe, next_pending, next_clock], dtype=jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_memory, {}


def init_agent_1():
    return jnp.asarray([-1, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    row, col = _position(obs)
    reset_memory = jnp.asarray([-1, 0, 0], dtype=jnp.int32)
    old = jnp.where(reset, reset_memory, memory)
    old_target = old[0]
    awaiting = old[1] > 0
    wait_count = old[2]
    target, has_visible_recipe = _recipe_from_visible(obs, old_target)
    known = target >= 0

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ingredient = (obs[2, 2, 7] > 0) | (obs[2, 2, 8] > 0) | (obs[2, 2, 9] > 0)

    pot = obs[:, :, 22]
    pot_count = jnp.sum(pot * (obs[:, :, 31] + obs[:, :, 32] + obs[:, :, 33]))
    pot_ready = jnp.sum(pot * obs[:, :, 30]) > 0
    pot_cooking = jnp.sum(pot * obs[:, :, 39]) > 0
    pot_full = pot_count >= 3

    at_ingredient = ((target == 1) & (row == 3) & (col == 6)) | ((target != 1) & (row == 1) & (col == 6))
    at_pot = (row == 2) & (col == 5)
    at_plate = (row == 1) & (col == 5)
    at_delivery = (row == 2) & (col == 6)
    at_keeper = (row == 2) & (col == 5)
    face_right = obs[2, 2, 3] > 0
    face_left = obs[2, 2, 4] > 0

    ingredient_row = jnp.where(target == 1, 3, 1)
    ingredient_action = jnp.where(at_ingredient,
                                   jnp.where(face_right, 5, 0),
                                   _go(row, col, ingredient_row, 6))
    pot_action = jnp.where(at_pot,
                           jnp.where(face_left, 5, 2),
                           _go(row, col, 2, 5))
    plate_action = jnp.where(at_plate,
                             jnp.where(face_left, 5, 2),
                             _go(row, col, 1, 5))
    delivery_action = jnp.where(at_delivery,
                                jnp.where(face_right, 5, 0),
                                _go(row, col, 2, 6))

    action_known = jnp.where(
        held_cooked, delivery_action,
        jnp.where(held_ingredient, pot_action,
                  jnp.where(held_plate,
                            jnp.where(pot_ready, pot_action, _go(row, col, 2, 5)),
                            jnp.where(pot_full | pot_cooking | pot_ready,
                                      plate_action, ingredient_action))))
    release_wait = awaiting & at_keeper & (has_visible_recipe | (wait_count >= 1))
    action = jnp.where(awaiting & ~release_wait, _go(row, col, 2, 5),
                       jnp.where(known, action_known, _go(row, col, 2, 5)))

    deliver_now = held_cooked & at_delivery & face_right
    next_awaiting = jnp.where(deliver_now, jnp.asarray(1, dtype=jnp.int32),
                              jnp.where(release_wait, jnp.asarray(0, dtype=jnp.int32),
                                        awaiting.astype(jnp.int32)))
    next_wait_count = jnp.where(deliver_now | release_wait, jnp.asarray(0, dtype=jnp.int32),
                                jnp.where(awaiting, wait_count + 1, jnp.asarray(0, dtype=jnp.int32)))
    next_memory = jnp.asarray([target, next_awaiting, next_wait_count], dtype=jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_memory, {}
