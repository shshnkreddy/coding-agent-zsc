"""Convention-light policy for the grounded coordination ring.

The inner player can cook autonomously after consulting a recipe button. The
outer player reads the free indicator and keeps a matching ingredient on a
visible shared counter, useful to partners that implement handoffs but not
required by this policy's inner player.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_reactive_chef",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}


_BOARD = (
    "WWW2R2WWW",
    "W       W",
    "W WWLWW W",
    "2 0   B 2",
    "R X P X R",
    "2 1   B 2",
    "W WWLWW W",
    "W       W",
    "WWW2R2WWW",
)
_SYMBOLS = ("W", "X", "P", "R", "L", "B", "0", "1", "2")


def _static_template():
    values = [[[0] * 9 for _ in range(9)] for _ in range(9)]
    for row in range(9):
        for col in range(9):
            symbol = _BOARD[row][col]
            if symbol in _SYMBOLS:
                values[row][col][_SYMBOLS.index(symbol)] = 1
    return jnp.asarray(values, dtype=jnp.int32)


_OUTER_POSITIONS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 1), (2, 7), (3, 1), (3, 7), (4, 1), (4, 7),
    (5, 1), (5, 7), (6, 1), (6, 7),
    (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
)
_INNER_POSITIONS = (
    (3, 3), (3, 4), (3, 5), (4, 3),
    (4, 5), (5, 3), (5, 4), (5, 5),
)


def _route_table(positions):
    """Make a deterministic shortest-path first-action table."""
    position_set = set(positions)
    moves = ((0, 1, 0), (0, -1, 2), (1, 0, 1), (-1, 0, 3))
    table = [[[[4 for _ in range(9)] for _ in range(9)]
              for _ in range(9)] for _ in range(9)]
    for start in positions:
        for target in positions:
            if start == target:
                continue
            queue = [start]
            previous = {start: None}
            first_action = {}
            for current in queue:
                for dr, dc, action in moves:
                    nxt = (current[0] + dr, current[1] + dc)
                    if nxt in position_set and nxt not in previous:
                        previous[nxt] = current
                        first_action[nxt] = (action if current == start
                                             else first_action[current])
                        queue.append(nxt)
            table[start[0]][start[1]][target[0]][target[1]] = first_action[target]
    return jnp.asarray(table, dtype=jnp.int32)


def _localize(obs, positions):
    """Match the observed static crop against every legal global position."""
    static = _static_template()
    padded = jnp.pad(static, ((2, 2), (2, 2), (0, 0)))
    crops = jnp.stack([padded[row:row + 5, col:col + 5, :]
                       for row, col in positions])
    coords = jnp.asarray(positions, dtype=jnp.int32)
    scores = jnp.sum(jnp.abs(crops - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    return coords[jnp.argmin(scores)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _move_to(pos, target_row, target_col, routes):
    return routes[pos[0], pos[1], target_row, target_col]


def _use_object(pos, facing, floor_row, floor_col, desired_facing,
                turn_action, routes):
    at_target = (pos[0] == floor_row) & (pos[1] == floor_col)
    ready = at_target & (facing == desired_facing)
    travel = _move_to(pos, floor_row, floor_col, routes)
    return jnp.where(ready, jnp.int32(5),
                     jnp.where(at_target, jnp.int32(turn_action), travel))


def _wait_facing(pos, facing, floor_row, floor_col, desired_facing,
                 turn_action, routes):
    at_target = (pos[0] == floor_row) & (pos[1] == floor_col)
    oriented = at_target & (facing == desired_facing)
    travel = _move_to(pos, floor_row, floor_col, routes)
    return jnp.where(oriented, jnp.int32(4),
                     jnp.where(at_target, jnp.int32(turn_action), travel))


def _visible_recipe(obs):
    count0 = jnp.sum(obs[:, :, 36])
    count1 = jnp.sum(obs[:, :, 37])
    visible = (count0 + count1) > 0
    recipe = jnp.where(count1 > count0, jnp.int32(1), jnp.int32(0))
    return visible, recipe


def init_agent_0():
    # [phase, known_recipe, recipe_id]
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    phase, known, recipe = memory
    routes = _route_table(_OUTER_POSITIONS)
    pos = _localize(obs, _OUTER_POSITIONS)
    facing = _facing(obs)
    visible, observed_recipe = _visible_recipe(obs)
    recipe_changed = visible & ((known == 0) | (observed_recipe != recipe))
    recipe = jnp.where(visible, observed_recipe, recipe)
    known = jnp.where(visible, jnp.int32(1), known)
    # Inspect the newly relevant counter before fetching; it may already have
    # been stocked during an earlier occurrence of this recipe.
    phase = jnp.where(recipe_changed, jnp.int32(3), phase)

    held0 = obs[2, 2, 7] > 0
    held1 = obs[2, 2, 8] > 0
    held_cooked = obs[2, 2, 6] > 0
    holding = held0 | held1 | (obs[2, 2, 5] > 0) | held_cooked

    handoff_row = jnp.where(recipe == 0, jnp.int32(2), jnp.int32(6))
    local_r = handoff_row - pos[0] + 2
    local_c = jnp.int32(3) - pos[1] + 2
    in_view = ((local_r >= 0) & (local_r < 5) &
               (local_c >= 0) & (local_c < 5))
    safe_r = jnp.clip(local_r, 0, 4)
    safe_c = jnp.clip(local_c, 0, 4)
    counter_occupied = in_view & (jnp.sum(obs[safe_r, safe_c, 29:34]) > 0)
    counter_dish_is_plated = ((obs[safe_r, safe_c, 29] > 0) &
                              (obs[safe_r, safe_c, 30] > 0))
    counter_dish_matches = jnp.where(
        recipe == 0,
        (obs[safe_r, safe_c, 31] == 3) & (obs[safe_r, safe_c, 32] == 0),
        (obs[safe_r, safe_c, 32] == 3) & (obs[safe_r, safe_c, 31] == 0))
    matching_dish3 = in_view & counter_dish_is_plated & counter_dish_matches

    dish_row = handoff_row
    dish_local_r = dish_row - pos[0] + 2
    dish_local_c = jnp.int32(5) - pos[1] + 2
    dish_in_view = ((dish_local_r >= 0) & (dish_local_r < 5) &
                    (dish_local_c >= 0) & (dish_local_c < 5))
    dish_r = jnp.clip(dish_local_r, 0, 4)
    dish_c = jnp.clip(dish_local_c, 0, 4)
    dish_is_plated = ((obs[dish_r, dish_c, 29] > 0) &
                      (obs[dish_r, dish_c, 30] > 0))
    dish_matches = jnp.where(
        recipe == 0,
        (obs[dish_r, dish_c, 31] == 3) & (obs[dish_r, dish_c, 32] == 0),
        (obs[dish_r, dish_c, 32] == 3) & (obs[dish_r, dish_c, 31] == 0))
    matching_dish5 = dish_in_view & dish_is_plated & dish_matches
    matching_dish = matching_dish3 | matching_dish5

    phase = jnp.where((phase == 1) & holding, jnp.int32(2), phase)
    phase = jnp.where((phase == 2) & (~holding), jnp.int32(3), phase)
    phase = jnp.where((phase == 1) & counter_occupied & (~holding),
                      jnp.int32(3), phase)
    phase = jnp.where((phase == 3) & in_view & (~counter_occupied),
                      jnp.int32(1), phase)
    phase = jnp.where(matching_dish & (~holding), jnp.int32(4), phase)
    phase = jnp.where(held_cooked, jnp.int32(3), phase)

    read_action = _use_object(pos, facing, 4, 1, 3, 2, routes)
    read_action = jnp.where((pos[0] == 4) & (pos[1] == 1),
                            jnp.where(facing == 3, jnp.int32(4), jnp.int32(2)),
                            read_action)

    get0 = _use_object(pos, facing, 3, 1, 2, 0, routes)
    get1 = _use_object(pos, facing, 5, 1, 2, 0, routes)
    get_action = jnp.where(recipe == 0, get0, get1)
    put0 = _use_object(pos, facing, 1, 3, 1, 1, routes)
    put1 = _use_object(pos, facing, 7, 3, 0, 3, routes)
    put_action = jnp.where(recipe == 0, put0, put1)
    monitor0 = _wait_facing(pos, facing, 1, 4, 0, 3, routes)
    monitor1 = _wait_facing(pos, facing, 7, 4, 1, 1, routes)
    monitor_action = jnp.where(recipe == 0, monitor0, monitor1)
    pick_dish30 = _use_object(pos, facing, 1, 3, 1, 1, routes)
    pick_dish31 = _use_object(pos, facing, 7, 3, 0, 3, routes)
    pick_dish50 = _use_object(pos, facing, 1, 5, 1, 1, routes)
    pick_dish51 = _use_object(pos, facing, 7, 5, 0, 3, routes)
    pick_dish3 = jnp.where(recipe == 0, pick_dish30, pick_dish31)
    pick_dish5 = jnp.where(recipe == 0, pick_dish50, pick_dish51)
    pick_dish = jnp.where(matching_dish3, pick_dish3, pick_dish5)
    relay_delivery = _use_object(pos, facing, 4, 7, 3, 2, routes)
    clear_put_action = jnp.where(counter_occupied, monitor_action, put_action)
    phase_action = jnp.where(
        phase == 1, get_action,
        jnp.where(phase == 2, clear_put_action,
                  jnp.where(phase == 4, pick_dish, monitor_action)))
    action = jnp.where(known == 0, read_action,
                       jnp.where(held_cooked, relay_delivery, phase_action))
    next_memory = jnp.asarray([phase, known, recipe], dtype=jnp.int32)
    trace = {"phase": phase, "recipe": recipe, "pos": pos}
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # [phase, known_recipe, recipe_id, elapsed_step]
    return jnp.zeros((4,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    phase, known, recipe, step = memory
    routes = _route_table(_INNER_POSITIONS)
    pos = _localize(obs, _INNER_POSITIONS)
    facing = _facing(obs)
    visible, observed_recipe = _visible_recipe(obs)
    recipe = jnp.where(visible, observed_recipe, recipe)
    known = jnp.where(visible, jnp.int32(1), known)
    phase = jnp.where((phase == 0) & visible, jnp.int32(1), phase)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held0 = obs[2, 2, 7] > 0
    held1 = obs[2, 2, 8] > 0
    held_target = jnp.where(recipe == 0, held0, held1)

    pot_mask = obs[:, :, 22] > 0
    pot0 = jnp.sum(jnp.where(pot_mask, obs[:, :, 31], 0))
    pot1 = jnp.sum(jnp.where(pot_mask, obs[:, :, 32], 0))
    pot_count = pot0 + pot1
    pot_cooked = jnp.any(pot_mask & (obs[:, :, 30] > 0))
    pot_timer = jnp.sum(jnp.where(pot_mask, obs[:, :, 39], 0))

    phase = jnp.where((phase == 1) & ((pot_count >= 3) | (pot_timer > 0)),
                      jnp.int32(2), phase)
    phase = jnp.where(held_cooked, jnp.int32(3), phase)
    delivered = (phase == 3) & (~held_cooked)
    phase = jnp.where(delivered, jnp.int32(0), phase)
    known = jnp.where(delivered, jnp.int32(0), known)

    # A partner visibly holding one homogeneous raw ingredient is grounded
    # task evidence, unlike an arbitrary position/action code. Retain the
    # button as the fallback whenever that evidence is absent or ambiguous.
    partner_count0 = jnp.sum(obs[:, :, 17])
    partner_count1 = jnp.sum(obs[:, :, 18])
    partner0 = partner_count0 == 1
    partner1 = partner_count1 == 1
    partner_raw = ((jnp.sum(obs[:, :, 15]) == 0) &
                   (jnp.sum(obs[:, :, 16]) == 0) &
                   ((partner_count0 + partner_count1) == 1))
    partner_evidence = partner_raw & (partner0 ^ partner1)
    partner_recipe = jnp.where(partner1, jnp.int32(1), jnp.int32(0))
    use_partner = (phase == 0) & partner_evidence
    recipe = jnp.where(use_partner, partner_recipe, recipe)
    known = jnp.where(use_partner, jnp.int32(1), known)
    phase = jnp.where(use_partner, jnp.int32(1), phase)

    button_action = _use_object(pos, facing, 3, 4, 0, 3, routes)
    get0 = _use_object(pos, facing, 3, 3, 3, 2, routes)
    get1 = _use_object(pos, facing, 5, 3, 3, 2, routes)
    get_ingredient = jnp.where(recipe == 0, get0, get1)
    add0 = _use_object(pos, facing, 3, 4, 1, 1, routes)
    add1 = _use_object(pos, facing, 5, 4, 0, 3, routes)
    add_ingredient = jnp.where(recipe == 0, add0, add1)
    fill_action = jnp.where(held_target, add_ingredient, get_ingredient)

    plate0 = _use_object(pos, facing, 3, 5, 2, 0, routes)
    plate1 = _use_object(pos, facing, 5, 5, 2, 0, routes)
    get_plate = jnp.where(recipe == 0, plate0, plate1)
    take0 = _use_object(pos, facing, 3, 4, 1, 1, routes)
    take1 = _use_object(pos, facing, 5, 4, 0, 3, routes)
    take_dish = jnp.where(recipe == 0, take0, take1)
    wait0 = _wait_facing(pos, facing, 3, 4, 1, 1, routes)
    wait1 = _wait_facing(pos, facing, 5, 4, 0, 3, routes)
    wait_at_pot = jnp.where(recipe == 0, wait0, wait1)
    plate_action = jnp.where(held_plate,
                             jnp.where(pot_cooked, take_dish, wait_at_pot),
                             get_plate)
    deliver_action = _use_object(pos, facing, 4, 3, 3, 2, routes)

    action = jnp.where(phase == 0, button_action,
                       jnp.where(phase == 1, fill_action,
                                 jnp.where(phase == 2, plate_action,
                                           deliver_action)))
    action = jnp.where((phase == 0) & (step > 360), jnp.int32(4), action)
    next_memory = jnp.asarray([phase, known, recipe, step + 1], dtype=jnp.int32)
    trace = {"phase": phase, "recipe": recipe, "pos": pos,
             "pot_count": pot_count, "pot_timer": pot_timer}
    return action.astype(jnp.int32), next_memory, trace
