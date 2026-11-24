"""A small, observation-grounded inner-room cook."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'inner_room_cook_v1',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}


# 0=floor, 1=wall, 2=goal, 3=pot, 4=recipe, 5=button,
# 6=plate pile, 7/8/9=ingredient 0/1/2 supplies.
_CELL = (
    [1, 1, 1, 9, 4, 9, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 5, 1, 1, 0, 1],
    [9, 0, 7, 0, 0, 0, 6, 0, 9],
    [4, 0, 2, 0, 3, 0, 2, 0, 4],
    [9, 0, 8, 0, 0, 0, 6, 0, 9],
    [1, 0, 1, 1, 5, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 9, 4, 9, 1, 1, 1],
)


_NEXT = (
    (4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,4,2,4,4,4,4,4,4,3,4,3,4,4,4,4,4,4,3,0,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4),
    (4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,4,4,4,4,4,4,3,4,3,4,4,4,4,4,4,3,2,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4),
    (4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,2,1,4,4,4,4,4,4,1,4,1,4,4,4,4,4,4,4,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4),
    (4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,1,4,4,4,4,4,4,3,4,4,4,4,4,4,4,4,0,0,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4),
    (4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,4,4,4,4,4,4,4,3,4,3,4,4,4,4,4,4,3,0,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4),
)


def _locate(obs):
    cell = jnp.asarray(_CELL, dtype=jnp.int32)
    static = jnp.eye(10, dtype=jnp.int32)[cell][:, :, 1:]
    padded = jnp.pad(static, ((2, 2), (2, 2), (0, 0)))
    windows = jnp.stack([
        padded[r:r + 5, c:c + 5, :] for r in range(9) for c in range(9)
    ])
    matches = jnp.all(windows == obs[:, :, 20:29], axis=(1, 2, 3))
    return jnp.argmax(matches.astype(jnp.int32))


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5].astype(jnp.int32))


def _action_for(obs, target_r, target_c, stand_slot, position):
    row = position // 9
    col = position - row * 9
    table = jnp.asarray([
        _NEXT[0] + (4,), _NEXT[1] + (4,), _NEXT[2] + (4,),
        _NEXT[3] + (4,), _NEXT[4]
    ], dtype=jnp.int32)
    move = table[stand_slot, position]
    adjacent = (jnp.abs(row - target_r) + jnp.abs(col - target_c)) == 1
    dr = target_r - row
    dc = target_c - col
    desired = jnp.where(dr < 0, 0, jnp.where(dr > 0, 1,
                          jnp.where(dc > 0, 2, 3)))
    face_action = jnp.where(desired == 0, 3,
                     jnp.where(desired == 1, 1,
                     jnp.where(desired == 2, 0, 2)))
    correctly_facing = _facing(obs) == desired
    return jnp.where(adjacent,
                     jnp.where(correctly_facing, 5, face_action),
                     move).astype(jnp.int32), adjacent & correctly_facing


def init_agent_0():
    return jnp.zeros((5,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del obs, rng
    zero = jnp.zeros((5,), dtype=jnp.int32)
    return jnp.asarray(4, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    # phase, recipe (0/1), ingredients added, reserved counter, deliveries
    return jnp.zeros((5,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    initial = jnp.zeros((5,), dtype=jnp.int32)
    mem = jnp.where(reset, initial, memory)
    phase, recipe, added, reserved, deliveries = mem

    position = _locate(obs)
    visible_recipe = jnp.max(obs[:, :, 36:39], axis=(0, 1))
    recipe_seen = jnp.any(visible_recipe > 0)
    seen_recipe = jnp.where(visible_recipe[1] > 0, 1, 0).astype(jnp.int32)
    effective_recipe = jnp.where(recipe_seen, seen_recipe, recipe)

    target_r = jnp.where(phase == 0, 2,
                 jnp.where(phase == 1, jnp.where(effective_recipe == 0, 3, 5),
                 jnp.where((phase == 2) | (phase == 3) | (phase == 5), 4,
                 jnp.where(phase == 4, 3, 4))))
    target_c = jnp.where(phase == 0, 4,
                 jnp.where(phase == 1, 2,
                 jnp.where(phase == 4, 6,
                 jnp.where(phase == 6, 6, 4))))
    stand_slot = jnp.where(phase == 0, 0,
                   jnp.where(phase == 1,
                     jnp.where(effective_recipe == 0, 1, 2),
                   jnp.where(phase == 4, 4,
                   jnp.where(phase == 6, 3, 0))))
    action, interact = _action_for(obs, target_r, target_c, stand_slot, position)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:10]) > 0
    hands_empty = ~(held_plate | held_cooked | held_raw)

    row = position // 9
    col = position - row * 9
    pot_lr = 4 - row + 2
    pot_lc = 4 - col + 2
    pot_in_view = (pot_lr >= 0) & (pot_lr < 5) & (pot_lc >= 0) & (pot_lc < 5)
    safe_lr = jnp.clip(pot_lr, 0, 4)
    safe_lc = jnp.clip(pot_lc, 0, 4)
    pot_cooked = pot_in_view & (obs[safe_lr, safe_lc, 30] > 0)

    next_phase = phase
    next_added = added
    next_recipe = effective_recipe
    next_deliveries = deliveries
    next_phase = jnp.where((phase == 0) & interact, 1, next_phase)
    picked = (phase == 1) & interact & hands_empty
    next_phase = jnp.where(picked, 2, next_phase)
    inserted = (phase == 2) & interact & held_raw
    next_added = jnp.where(inserted, added + 1, next_added)
    next_phase = jnp.where(inserted & (added < 2), 1, next_phase)
    next_phase = jnp.where(inserted & (added >= 2), 3, next_phase)
    next_phase = jnp.where((phase == 3) & pot_cooked, 4, next_phase)
    plated = (phase == 4) & interact & hands_empty
    next_phase = jnp.where(plated, 5, next_phase)
    scooped = (phase == 5) & interact & held_plate
    next_phase = jnp.where(scooped, 6, next_phase)
    delivered = (phase == 6) & interact & held_cooked
    next_phase = jnp.where(delivered, 0, next_phase)
    next_added = jnp.where(delivered, 0, next_added)
    next_deliveries = jnp.where(delivered, deliveries + 1, next_deliveries)

    next_memory = jnp.stack([
        next_phase, next_recipe, next_added, reserved, next_deliveries
    ]).astype(jnp.int32)
    trace = {
        'phase': phase,
        'position': position,
        'recipe': effective_recipe,
        'added': added,
        'held_plate': held_plate,
        'held_cooked': held_cooked,
        'held_raw': held_raw,
        'pot_cooked': pot_cooked,
        'interact': interact,
    }
    return action, next_memory, trace
