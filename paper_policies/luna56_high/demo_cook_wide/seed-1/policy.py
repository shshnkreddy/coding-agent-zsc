"""Recipe-aware supplier and generic delivery controller."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'recipe_supplier_delivery',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}


# Static channels 22:32 identify every reachable cell in this fixed layout.
_WALLS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10),
    (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8),
    (2, 9), (2, 10), (3, 0), (3, 10), (4, 3), (4, 7),
    (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
    (5, 10),
)
_AGENT_0_CELLS = ((1, 4), (1, 5), (1, 6))
_AGENT_1_CELLS = (
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (3, 9), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (4, 8), (4, 9),
    (5, 1), (5, 9),
)


def _map_patches(cells):
    static = jnp.zeros((6, 11, 10), dtype=jnp.int32)
    walls = jnp.asarray(_WALLS)
    static = static.at[walls[:, 0], walls[:, 1], 0].set(1)
    static = static.at[0, 5, 1].set(1)
    static = static.at[2, 5, 2].set(1)
    static = static.at[4, 5, 3].set(1)
    static = static.at[jnp.asarray((0, 0)), jnp.asarray((4, 6)), 5].set(1)
    static = static.at[jnp.asarray((1, 4, 4)), jnp.asarray((3, 0, 10)), 6].set(1)
    static = static.at[jnp.asarray((1, 5, 5)), jnp.asarray((7, 1, 9)), 7].set(1)
    static = static.at[jnp.asarray((4, 4)), jnp.asarray((4, 6)), 9].set(1)
    padded = jnp.pad(static, ((4, 4), (4, 4), (0, 0)))
    patches = jnp.stack(tuple(padded[r + 2:r + 7, c + 2:c + 7] for r, c in cells))
    return patches, jnp.asarray(cells, dtype=jnp.int32)


def _position(obs, cells):
    patches, coords = _map_patches(cells)
    visible_static = obs[:, :, 22:32]
    error = jnp.sum(jnp.abs(patches - visible_static[None, :, :, :]), axis=(1, 2, 3))
    return coords[jnp.argmin(error)]


def _at(pos, row, col):
    return (pos[0] == row) & (pos[1] == col)


def _move_to(pos, row, col):
    return jnp.where(
        pos[0] < row, 1,
        jnp.where(pos[0] > row, 3,
                  jnp.where(pos[1] < col, 0,
                            jnp.where(pos[1] > col, 2, 4))))


def _route_agent_1(pos, row, col):
    at_target = _at(pos, row, col)
    at_corridor = _at(pos, 3, col)
    corridor_action = _move_to(pos, 3, col)
    target_action = _move_to(pos, row, col)
    return jnp.where(at_target, 4, jnp.where(at_corridor, target_action, corridor_action))


def _face_or_interact(obs, facing_id):
    movement_for_facing = jnp.asarray((3, 1, 0, 2), dtype=jnp.int32)
    return jnp.where(obs[2, 2, 1 + facing_id] > 0, 5, movement_for_facing[facing_id])


def _held(obs):
    return obs[2, 2, 5:11]


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    pos = _position(obs, _AGENT_0_CELLS)
    phase = memory[0]
    held = _held(obs)
    has_plate = held[0] > 0
    has_cooked = held[1] > 0
    empty_hands = jnp.sum(held) == 0
    at_plate = _at(pos, 1, 4)
    at_pot = _at(pos, 1, 5)
    pot_ready = obs[3, 2, 33] > 0

    go_plate = _move_to(pos, 1, 4)
    get_plate = jnp.where(at_plate, _face_or_interact(obs, 0), go_plate)
    wait_or_scoop = jnp.where(
        at_pot & pot_ready, _face_or_interact(obs, 1),
        jnp.where(at_pot, 4, _move_to(pos, 1, 5)))
    deliver = jnp.where(at_pot, _face_or_interact(obs, 0), _move_to(pos, 1, 5))
    action = jnp.where(phase == 0, get_plate,
                       jnp.where(phase == 1, wait_or_scoop, deliver))

    next_phase = phase
    next_phase = jnp.where((phase == 0) & has_plate, 1, next_phase)
    next_phase = jnp.where((phase == 1) & has_cooked, 2, next_phase)
    next_phase = jnp.where((phase == 2) & empty_hands, 0, next_phase)
    next_memory = next_phase[None]
    return action.astype(jnp.int32), jnp.where(reset, jnp.zeros_like(memory), next_memory), {}


def init_agent_1():
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    pos = _position(obs, _AGENT_1_CELLS)
    phase, loaded, recipe = memory[0], memory[1], memory[2]
    held = _held(obs)
    holding_raw = jnp.sum(held[2:]) > 0
    # phase 0=read, 1=supply, 2=load, 3=wait for pickup, 4=wait for delivery.
    # The loaded field is a short cooldown counter only in phase 4.
    # The indicator is an obstructing object; its display is visible from the
    # corridor cell immediately above it.
    at_recipe = _at(pos, 3, 5)
    at_supply = _at(pos, 4, 9)
    at_pot = _at(pos, 3, 5)

    supply_facing = jnp.where(recipe == 0, 2, 1)
    seek_recipe = _route_agent_1(pos, 3, 5)
    seek_supply = _route_agent_1(pos, 4, 9)
    seek_pot = _route_agent_1(pos, 3, 5)
    supply_action = jnp.where(
        holding_raw, 4,
        jnp.where(at_supply, _face_or_interact(obs, supply_facing), seek_supply))
    pot_action = jnp.where(
        holding_raw,
        jnp.where(at_pot, _face_or_interact(obs, 0), seek_pot),
        4)
    action = jnp.where(phase == 0, seek_recipe,
                       jnp.where(phase == 1, supply_action,
                                 jnp.where(phase == 2, pot_action, 4)))

    read_recipe = (phase == 0) & at_recipe
    recipe_seen = jnp.where(jnp.sum(obs[:, :, 40]) > 0, 0, 1).astype(jnp.int32)
    inserted = (phase == 2) & at_pot & (~holding_raw)
    pot_item_empty = jnp.sum(obs[1, 2, 32:38]) == 0
    pickup_finished = (phase == 3) & at_pot & pot_item_empty
    picked_up = (phase == 1) & at_supply & holding_raw

    next_recipe = jnp.where(read_recipe, recipe_seen, recipe)
    next_loaded = jnp.where(
        pickup_finished, 0,
        jnp.where(inserted, loaded + 1, loaded))
    next_phase = phase
    next_phase = jnp.where(read_recipe, 1, next_phase)
    next_phase = jnp.where(picked_up, 2, next_phase)
    next_phase = jnp.where(inserted, jnp.where(loaded >= 2, 3, 1), next_phase)
    next_phase = jnp.where(pickup_finished, 4, next_phase)
    next_phase = jnp.where((phase == 4) & (loaded <= 1), 0, next_phase)
    next_loaded = jnp.where(phase == 4, jnp.maximum(loaded - 1, 0), next_loaded)
    next_loaded = jnp.where(pickup_finished, 5, next_loaded)
    next_memory = jnp.stack((next_phase, next_loaded, next_recipe))
    return action.astype(jnp.int32), jnp.where(reset, jnp.zeros_like(memory), next_memory), {}
