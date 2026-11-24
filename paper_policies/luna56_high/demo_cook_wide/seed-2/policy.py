"""A geometry-localized two-role controller for demo_cook_wide."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'geometry_role_controller',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# Static channels 22:32 for the known board.  A local static crop identifies
# the absolute floor tile, which is legal local evidence and handles random
# reset positions without a clock or host-side state.
_W = (1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
_STATIC = [
    [_W, _W, _W, _W, (0, 0, 0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0, 0, 0), _W, _W, _W, _W],
    [_W, _W, _W, (0, 0, 0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), _W, _W, _W],
    [_W, _W, _W, _W, _W, (0, 0, 0, 1, 0, 0, 0, 0, 0, 0), _W, _W, _W, _W, _W],
    [_W, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), _W],
    [(0, 0, 0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 1), _W, _W, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), _W, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
    [_W, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1), _W, _W, _W, _W, _W, _W, _W, (0, 0, 0, 0, 0, 0, 0, 0, 0, 1), _W],
]
_F = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
_W = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
_G = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
_P = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
_R = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
_B = (0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
_I0 = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
_I1 = (0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
_I3 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
_STATIC = [
    [_W, _W, _W, _W, _B, _G, _B, _W, _W, _W, _W],
    [_W, _W, _W, _I0, _F, _F, _F, _I1, _W, _W, _W],
    [_W, _W, _W, _W, _W, _P, _W, _W, _W, _W, _W],
    [_W, _F, _F, _F, _F, _F, _F, _F, _F, _F, _W],
    [_I0, _F, _F, _W, _I3, _R, _I3, _W, _F, _F, _I0],
    [_W, _I1, _W, _W, _W, _W, _W, _W, _W, _I1, _W],
]
_A0_CANDIDATES = ((1, 4), (1, 5), (1, 6))
_A1_CANDIDATES = ((3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 1), (4, 2), (4, 8), (4, 9))


def _locate(obs, candidates):
    static = jnp.asarray(_STATIC, dtype=jnp.int32)
    padded = jnp.pad(static, ((2, 2), (2, 2), (0, 0)))
    patches = jnp.stack(tuple(padded[r:r + 5, c:c + 5] for r, c in candidates))
    visible_static = obs[:, :, 22:32]
    matches = jnp.all(patches == visible_static[None, :, :, :], axis=(1, 2, 3))
    index = jnp.argmax(matches.astype(jnp.int32))
    return jnp.asarray(candidates, dtype=jnp.int32)[index]


def _facing(obs):
    # UP, DOWN, RIGHT, LEFT channels map to direction IDs 0, 1, 2, 3.
    index = jnp.argmax(obs[2, 2, 1:5].astype(jnp.int32))
    return jnp.asarray((0, 1, 2, 3), dtype=jnp.int32)[index]


def _move_to(pos, target):
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    return jnp.where(dr < 0, 3, jnp.where(dr > 0, 1, jnp.where(dc < 0, 2, jnp.where(dc > 0, 0, 4))))


def _face_or_interact(pos, stand, desired_facing, facing, permitted):
    at_stand = jnp.all(pos == stand)
    turn = jnp.asarray((3, 1, 0, 2), dtype=jnp.int32)[desired_facing]
    action = jnp.where(facing != desired_facing, turn, jnp.where(permitted, 5, 4))
    return jnp.where(at_stand, action, _move_to(pos, stand))


def _self_empty(obs):
    return jnp.sum(obs[2, 2, 5:11]) == 0


def _self_plate(obs):
    return obs[2, 2, 5] > 0


def _self_cooked(obs):
    return obs[2, 2, 6] > 0


def _pot_cooked(obs):
    return jnp.any((obs[:, :, 24] > 0) & (obs[:, :, 33] > 0))


def _pot_idle_with_space(obs):
    pot = obs[:, :, 24] > 0
    raw_count = jnp.sum(obs[:, :, 34:38] * pot[:, :, None])
    cooked = jnp.any(pot & (obs[:, :, 33] > 0))
    timer = jnp.sum(obs[:, :, 44])
    return (timer == 0) & (~cooked) & (raw_count < 3)


def _pot_empty(obs):
    pot = obs[:, :, 24] > 0
    item = jnp.sum(obs[:, :, 32:38] * pot[:, :, None])
    return item == 0


def init_agent_0():
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, init_agent_0(), memory)
    phase = jnp.clip(mem[0], 0, 2)
    pos = _locate(obs, _A0_CANDIDATES)
    facing = _facing(obs)
    plate = _self_plate(obs)
    cooked = _self_cooked(obs)

    def get_plate(m):
        action = _face_or_interact(pos, jnp.asarray((1, 4)), 0, facing, _self_empty(obs))
        picked = (action == 5) & _self_empty(obs)
        return action, m.at[0].set(jnp.where(picked, 1, 0))

    def wait_for_cooked(m):
        permitted = plate & _pot_cooked(obs)
        action = _face_or_interact(pos, jnp.asarray((1, 5)), 1, facing, permitted)
        scooped = (action == 5) & permitted
        return action, m.at[0].set(jnp.where(scooped, 2, 1))

    def deliver(m):
        permitted = cooked
        action = _face_or_interact(pos, jnp.asarray((1, 5)), 0, facing, permitted)
        delivered = (action == 5) & permitted
        return action, m.at[0].set(jnp.where(delivered, 0, jnp.where(cooked, 2, 1)))

    action, next_mem = jax.lax.switch(phase, (get_plate, wait_for_cooked, deliver), mem)
    return action.astype(jnp.int32), next_mem, {}


def init_agent_1():
    return jnp.asarray((0, 0, 0, 0), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, init_agent_1(), memory)
    phase = jnp.clip(mem[0], 0, 4)
    count = mem[1]
    recipe_seen = jnp.any(obs[:, :, 40] > 0) | jnp.any(obs[:, :, 41] > 0)
    sensed_recipe = jnp.where(jnp.sum(obs[:, :, 41]) > jnp.sum(obs[:, :, 40]), 1, 0)
    recipe = jnp.where(recipe_seen, sensed_recipe, mem[2])
    pos = _locate(obs, _A1_CANDIDATES)
    facing = _facing(obs)
    raw = jnp.sum(obs[2, 2, 7:11]) > 0
    empty = _self_empty(obs)

    def route_to_right(m):
        at_waypoint = jnp.all(pos == jnp.asarray((3, 9)))
        action = _move_to(pos, jnp.asarray((3, 9)))
        return action, m.at[0].set(jnp.where(at_waypoint, 1, 0)).at[2].set(recipe)

    def get_ingredient(m):
        desired = jnp.where(recipe == 0, 2, 1)
        action = _face_or_interact(pos, jnp.asarray((4, 9)), desired, facing, empty)
        picked = (action == 5) & empty
        return action, m.at[0].set(jnp.where(picked, 2, 1)).at[2].set(recipe)

    def route_to_pot(m):
        permitted = raw & _pot_idle_with_space(obs)
        action = _face_or_interact(pos, jnp.asarray((3, 5)), 0, facing, permitted)
        inserted = (action == 5) & permitted
        new_count = count + inserted.astype(jnp.int32)
        next_phase = jnp.where(inserted & (new_count >= 3), 4, jnp.where(inserted, 0, 2))
        return action, m.at[0].set(next_phase).at[1].set(jnp.where(inserted, new_count, count)).at[2].set(recipe)

    def same_pot_route(m):
        return route_to_pot(m)

    def wait_for_scoop(m):
        released = _pot_empty(obs) & (~_pot_cooked(obs))
        action = _face_or_interact(pos, jnp.asarray((3, 5)), 0, facing, False)
        return action, m.at[0].set(jnp.where(released, 0, 4)).at[1].set(jnp.where(released, 0, count)).at[2].set(recipe)

    action, next_mem = jax.lax.switch(phase, (route_to_right, get_ingredient, route_to_pot, same_pot_route, wait_for_scoop), mem)
    return action.astype(jnp.int32), next_mem, {}
