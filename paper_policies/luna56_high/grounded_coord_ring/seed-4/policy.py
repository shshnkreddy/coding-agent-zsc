"""A layout-grounded, convention-light Overcooked policy."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_autonomous_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}

_STATIC_GRID = (((1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,1),(0,0,0,1,0,0,0,0,0),(0,0,0,0,0,0,0,0,1),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0)),((1,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0)),((1,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(0,0,0,0,1,0,0,0,0),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0)),((0,0,0,0,0,0,0,0,1),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,1,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,1,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,1)),((0,0,0,1,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,1,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,1,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,1,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,1,0,0,0,0,0)),((0,0,0,0,0,0,0,0,1),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,1,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,1,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,1)),((1,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(0,0,0,0,1,0,0,0,0),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0)),((1,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0)),((1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,1),(0,0,0,1,0,0,0,0,0),(0,0,0,0,0,0,0,0,1),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0),(1,0,0,0,0,0,0,0,0)))
_CENTRAL_CELLS = ((3,3),(3,4),(3,5),(4,3),(4,5),(5,3),(5,4),(5,5))
_OUTER_CELLS = ((1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,1),(2,7),(3,1),(3,7),(4,1),(4,7),(5,1),(5,7),(6,1),(6,7),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7))
_CENTRAL_NEXT = ((4,0,0,1,0,1,1,1),(2,4,0,2,0,2,0,0),(2,2,4,2,1,1,1,1),(3,3,3,4,3,1,1,1),(3,3,3,3,4,1,1,1),(3,3,3,3,0,4,0,0),(2,0,0,2,0,2,4,0),(3,3,3,2,3,2,2,4))
_OUTER_NEXT = ((4,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1),(2,4,0,0,0,0,0,2,0,2,0,2,0,2,0,2,0,2,2,2,2,2,0,0),(2,2,4,0,0,0,0,2,0,2,0,2,0,2,0,2,0,2,2,2,2,0,0,0),(2,2,2,4,0,0,0,2,0,2,0,2,0,2,0,2,0,2,2,2,0,0,0,0),(2,2,2,2,4,0,0,2,0,2,0,2,0,2,0,2,0,2,2,0,0,0,0,0),(2,2,2,2,2,4,0,2,0,2,0,2,0,2,0,2,0,2,0,0,0,0,0,0),(2,2,2,2,2,2,4,2,1,2,1,2,1,2,1,2,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,4,3,1,3,1,3,1,3,1,3,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,4,3,1,3,1,3,1,3,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,3,4,3,1,3,1,3,1,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,3,3,4,3,1,3,1,1,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,3,3,3,4,3,1,1,1,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,3,3,3,3,4,1,1,1,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,3,3,3,3,1,4,1,1,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,3,3,3,1,3,1,4,1,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,3,3,1,3,1,3,1,4,1,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,3,1,3,1,3,1,3,1,4,1,1,1,1,1,1,1),(3,3,3,3,3,3,3,3,0,3,0,3,0,3,0,3,0,4,0,0,0,0,0,0),(2,2,2,2,2,0,0,2,0,2,0,2,0,2,0,2,0,2,4,0,0,0,0,0),(2,2,2,2,0,0,0,2,0,2,0,2,0,2,0,2,0,2,2,4,0,0,0,0),(2,2,2,0,0,0,0,2,0,2,0,2,0,2,0,2,0,2,2,2,4,0,0,0),(2,2,0,0,0,0,0,2,0,2,0,2,0,2,0,2,0,2,2,2,2,4,0,0),(2,0,0,0,0,0,0,2,0,2,0,2,0,2,0,2,0,2,2,2,2,2,4,0),(3,3,3,3,3,3,3,2,3,2,3,2,3,2,3,2,3,2,2,2,2,2,2,4))


def _make_templates(centers):
    static_grid = jnp.asarray(_STATIC_GRID, dtype=jnp.int32)
    templates = []
    for row, col in centers:
        patch = []
        for dr in (-2, -1, 0, 1, 2):
            patch_row = []
            for dc in (-2, -1, 0, 1, 2):
                rr, cc = row + dr, col + dc
                if 0 <= rr < 9 and 0 <= cc < 9:
                    patch_row.append(static_grid[rr, cc])
                else:
                    patch_row.append(jnp.zeros((9,), dtype=jnp.int32))
            patch.append(jnp.stack(patch_row))
        templates.append(jnp.stack(patch))
    return jnp.stack(templates)


def _localize(obs, centers):
    templates = _make_templates(centers)
    static_obs = obs[:, :, 20:29]
    mismatch = jnp.sum(jnp.abs(templates - static_obs[None, :, :, :]), axis=(1, 2, 3))
    return jnp.argmin(mismatch).astype(jnp.int32)


def _face_or_interact(obs, face):
    move_action = jnp.asarray((3, 1, 0, 2), dtype=jnp.int32)[face]
    already_facing = obs[2, 2, 1 + face] > 0
    return jnp.where(already_facing, jnp.asarray(5, dtype=jnp.int32), move_action)


def _button_active(obs):
    button = obs[:, :, 24]
    recipe = obs[:, :, 36:39]
    return jnp.any((button[:, :, None] > 0) & (recipe > 0))


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del obs, rng
    zero = jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(4, dtype=jnp.int32), jnp.where(reset, zero, memory), {}


def init_agent_1():
    return jnp.zeros((2,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    phase = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory[0])
    recipe = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory[1])
    position = _localize(obs, _CENTRAL_CELLS)
    recipe_stand = jnp.asarray(1, dtype=jnp.int32)
    supply0_stand = jnp.asarray(0, dtype=jnp.int32)
    supply1_stand = jnp.asarray(5, dtype=jnp.int32)
    pot_stand = jnp.asarray(3, dtype=jnp.int32)
    plate_stand = jnp.asarray(2, dtype=jnp.int32)
    delivery_stand = jnp.asarray(4, dtype=jnp.int32)
    central_next = jnp.asarray(_CENTRAL_NEXT, dtype=jnp.int32)

    held = obs[2, 2, 7:10]
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    pot_mask = obs[:, :, 22]
    pot_counts = jnp.sum(obs[:, :, 31:34] * pot_mask[:, :, None], axis=(0, 1))
    pot_total = jnp.sum(pot_counts)
    pot_cooked = jnp.any((obs[:, :, 30] * pot_mask) > 0)

    displayed0 = obs[1, 2, 36] > 0
    displayed1 = obs[1, 2, 37] > 0
    displayed = displayed0 | displayed1
    learned_recipe = jnp.where(displayed1, jnp.asarray(1, dtype=jnp.int32),
                               jnp.asarray(0, dtype=jnp.int32))
    at_recipe = position == recipe_stand
    recipe_route = central_next[position, recipe_stand]
    recipe_action = _face_or_interact(obs, 0)
    action0 = jnp.where(at_recipe,
                        jnp.where(displayed, jnp.asarray(4, dtype=jnp.int32), recipe_action),
                        recipe_route)
    next_phase0 = jnp.where(at_recipe & displayed, jnp.asarray(1, dtype=jnp.int32), phase)
    next_recipe0 = jnp.where(at_recipe & displayed, learned_recipe, recipe)

    desired_held = jnp.where(recipe == 0, held[0], held[1])
    supply_stand = jnp.where(recipe == 0, supply0_stand, supply1_stand)
    at_pot = position == pot_stand
    at_supply = position == supply_stand
    pot_route = central_next[position, pot_stand]
    supply_route = central_next[position, supply_stand]
    pot_action = _face_or_interact(obs, 2)
    supply_action = _face_or_interact(obs, 3)
    holding_ingredient = desired_held > 0
    fill_action = jnp.where(holding_ingredient,
                            jnp.where(at_pot, pot_action, pot_route),
                            jnp.where(at_supply, supply_action, supply_route))
    fill_done = pot_total >= 3
    action1 = jnp.where(fill_done, jnp.asarray(4, dtype=jnp.int32), fill_action)
    next_phase1 = jnp.where(fill_done, jnp.asarray(2, dtype=jnp.int32), phase)

    action2 = jnp.asarray(4, dtype=jnp.int32)
    next_phase2 = jnp.where(pot_cooked, jnp.asarray(3, dtype=jnp.int32), phase)

    at_plate = position == plate_stand
    plate_route = central_next[position, plate_stand]
    plate_action = _face_or_interact(obs, 2)
    plate_to_pot_route = central_next[position, pot_stand]
    plate_to_pot_action = jnp.where(at_pot, pot_action, plate_to_pot_route)
    action3 = jnp.where(held_plate & ~held_cooked,
                        plate_to_pot_action,
                        jnp.where(at_plate, plate_action, plate_route))
    next_phase3 = jnp.where(held_cooked, jnp.asarray(4, dtype=jnp.int32), phase)

    at_delivery = position == delivery_stand
    delivery_route = central_next[position, delivery_stand]
    delivery_action = _face_or_interact(obs, 2)
    action4 = jnp.where(at_delivery, delivery_action, delivery_route)
    next_phase4 = jnp.where(~held_cooked, jnp.asarray(0, dtype=jnp.int32), phase)

    action = action0
    next_phase = next_phase0
    next_recipe = next_recipe0
    action = jnp.where(phase == 1, action1, action)
    next_phase = jnp.where(phase == 1, next_phase1, next_phase)
    action = jnp.where(phase == 2, action2, action)
    next_phase = jnp.where(phase == 2, next_phase2, next_phase)
    action = jnp.where(phase == 3, action3, action)
    next_phase = jnp.where(phase == 3, next_phase3, next_phase)
    action = jnp.where(phase == 4, action4, action)
    next_phase = jnp.where(phase == 4, next_phase4, next_phase)
    next_memory = jnp.stack((next_phase, next_recipe)).astype(jnp.int32)
    trace = {
        'phase': phase.astype(jnp.int32),
        'position': position.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'pot_cooked': pot_cooked.astype(jnp.int32),
        'held0': held[0].astype(jnp.int32),
        'held1': held[1].astype(jnp.int32),
        'held_plate': held_plate.astype(jnp.int32),
        'held_cooked': held_cooked.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
