"""Observation-driven cooperative policy for the demo_cook_wide layout.

The lower player communicates the recipe by passing the actual requested raw
ingredient over a counter. The upper player treats observable ingredients as
evidence rather than relying on a private movement convention.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_counter_chefs', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _static_map():
    x = jnp.zeros((6, 11, 10), dtype=jnp.int32)
    walls = ((0,0),(0,1),(0,2),(0,3),(0,7),(0,8),(0,9),(0,10),
             (1,0),(1,1),(1,2),(1,8),(1,9),(1,10),
             (2,0),(2,1),(2,2),(2,3),(2,4),(2,6),(2,7),(2,8),(2,9),(2,10),
             (3,0),(3,10),(4,3),(4,7),
             (5,0),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,10))
    for r, c in walls:
        x = x.at[r, c, 0].set(1)
    x = x.at[0, 5, 1].set(1)
    x = x.at[2, 5, 2].set(1)
    x = x.at[4, 5, 3].set(1)
    x = x.at[0, 4, 5].set(1).at[0, 6, 5].set(1)
    for r, c in ((1,3),(4,0),(4,10)):
        x = x.at[r, c, 6].set(1)
    for r, c in ((1,7),(5,1),(5,9)):
        x = x.at[r, c, 7].set(1)
    for r, c in ((4,4),(4,6)):
        x = x.at[r, c, 9].set(1)
    return x


def _localize(obs, lower):
    top_positions = ((1,4),(1,5),(1,6))
    low_positions = ((3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),
                     (4,1),(4,2),(4,8),(4,9))
    py_positions = low_positions if lower else top_positions
    static = jnp.pad(_static_map(), ((2,2),(2,2),(0,0)))
    templates = jnp.stack([static[r:r+5, c:c+5, :] for r, c in py_positions])
    positions = jnp.asarray(py_positions, dtype=jnp.int32)
    seen = obs[:, :, 22:32]
    errors = jnp.sum(jnp.abs(templates - seen[None, :, :, :]), axis=(1,2,3))
    return positions[jnp.argmin(errors)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_or_interact(obs, direction):
    face_action = jnp.asarray((UP, DOWN, RIGHT, LEFT), dtype=jnp.int32)[direction]
    return jnp.where(_facing(obs) == direction, INTERACT, face_action)


def _nav_top(obs, pos, col, direction):
    action = _face_or_interact(obs, direction)
    action = jnp.where(pos[1] < col, RIGHT, action)
    action = jnp.where(pos[1] > col, LEFT, action)
    return action


def _nav_lower(obs, pos, dest_row, dest_col, direction, do_interact=True):
    final = _face_or_interact(obs, direction) if do_interact else STAY
    at_dest = (pos[0] == dest_row) & (pos[1] == dest_col)
    action = jnp.where(at_dest, final, STAY)
    action = jnp.where((pos[0] == 4) & ~at_dest, UP, action)
    on_corridor = pos[0] == 3
    action = jnp.where(on_corridor & (pos[1] < dest_col), RIGHT, action)
    action = jnp.where(on_corridor & (pos[1] > dest_col), LEFT, action)
    action = jnp.where(on_corridor & (pos[1] == dest_col) & (dest_row == 4), DOWN, action)
    return action


def _counter_values(obs, pos):
    i4 = 6 - pos[1]
    i6 = 8 - pos[1]
    c4_0 = obs[3, i4, 34]
    c4_1 = obs[3, i4, 35]
    c6_0 = obs[3, i6, 34]
    c6_1 = obs[3, i6, 35]
    return c4_0, c4_1, c6_0, c6_1


def init_agent_0():
    # recipe, wait, previous pot count, elapsed, partner style
    # style: 0 unknown, 1 continuous passer, 2 self-service expected
    # final field arms held-item evidence after an observed empty inventory.
    # Last field quarantines counter-only evidence after recipe resampling.
    return jnp.asarray((-1, 0, 0, 0, 0, 1, 0), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    m = jnp.where(reset, init_agent_0(), memory)
    pos = _localize(obs, False)

    inv_plate = obs[2, 2, 5] > 0
    inv_cooked = obs[2, 2, 6] > 0
    inv0 = obs[2, 2, 7] > 0
    inv1 = obs[2, 2, 8] > 0
    inv_raw = inv0 | inv1

    pot_mask = obs[:, :, 24]
    pot0 = jnp.sum(pot_mask * obs[:, :, 34])
    pot1 = jnp.sum(pot_mask * obs[:, :, 35])
    pot_count = pot0 + pot1
    pot_ready = jnp.sum(pot_mask * obs[:, :, 33]) > 0
    pot_timer = jnp.sum(pot_mask * obs[:, :, 44])

    c40, c41, c60, c61 = _counter_values(obs, pos)
    counter0 = (c40 + c60) > 0
    counter1 = (c41 + c61) > 0
    counter_raw = counter0 | counter1

    partner0 = jnp.sum(obs[:, :, 18]) > 0
    partner1 = jnp.sum(obs[:, :, 19]) > 0
    partner_visible = jnp.sum(obs[:, :, 11]) > 0
    partner_flat = jnp.argmax(obs[:, :, 11])
    partner_col = pos[1] + (partner_flat % 5) - 2
    # Counter placement is a completed handoff. Merely carrying an ingredient
    # can be stale work left over from the preceding recipe.
    recipe = m[0]
    quarantine = m[6]
    partner_raw = partner0 | partner1
    held_safe = jnp.where((recipe < 0) & partner_visible & ~partner_raw, 1, m[5])
    fresh0 = (recipe < 0) & (held_safe > 0) & partner0 & ~partner1
    fresh1 = (recipe < 0) & (held_safe > 0) & partner1 & ~partner0
    pot_evidence0 = (pot0 > 0) & (pot1 == 0)
    pot_evidence1 = (pot1 > 0) & (pot0 == 0)
    counter_evidence0 = counter0 & ~counter1
    counter_evidence1 = counter1 & ~counter0
    recipe = jnp.where((recipe < 0) & (quarantine == 0) & counter_evidence0, 0, recipe)
    recipe = jnp.where((recipe < 0) & (quarantine == 0) & counter_evidence1, 1, recipe)
    recipe = jnp.where(pot_evidence0, 0, recipe)
    recipe = jnp.where(pot_evidence1, 1, recipe)
    # A freshness-gated carry overrides conflicting items left on counters.
    recipe = jnp.where(fresh0, 0, recipe)
    recipe = jnp.where(fresh1, 1, recipe)
    held_safe = jnp.where(fresh0 | fresh1, 0, held_safe)
    quarantine = jnp.where(fresh0 | fresh1, 0, quarantine)

    c4_usable = ((recipe == 0) & (c40 > 0)) | ((recipe == 1) & (c41 > 0))
    c6_usable = ((recipe == 0) & (c60 > 0)) | ((recipe == 1) & (c61 > 0))
    usable_counter = c4_usable | c6_usable

    changed_pot = pot_count != m[2]
    waiting = (recipe >= 0) & (pot_count < 3) & ~usable_counter & ~inv_raw
    wait_time = jnp.where(waiting & ~changed_pot, m[1] + 1, 0)
    partner_style = jnp.where(usable_counter & (pot_count > 0), 1, m[4])
    partner_style = jnp.where((pot_count > 0) & (pot_count < 3) &
                              (partner0 | partner1), 1, partner_style)
    partner_style = jnp.where((m[4] == 0) & (pot_count > 0) & ~usable_counter &
                              partner_visible & (partner_col == 5) &
                              ~(partner0 | partner1), 2, partner_style)
    # Freshly carried evidence means a handoff may be imminent even when this
    # partner previously favored self-service; wait briefly for completion.
    partner_style = jnp.where(fresh0 | fresh1, 0, partner_style)

    get_plate = _nav_top(obs, pos, jnp.int32(4), jnp.int32(0))
    pick_c4 = _nav_top(obs, pos, jnp.int32(4), jnp.int32(1))
    pick_c6 = _nav_top(obs, pos, jnp.int32(6), jnp.int32(1))
    pick_counter = jnp.where(c4_usable, pick_c4, pick_c6)
    own_col = jnp.where(recipe == 0, jnp.int32(4), jnp.int32(6))
    own_dir = jnp.where(recipe == 0, jnp.int32(3), jnp.int32(2))
    fetch_own = _nav_top(obs, pos, own_col, own_dir)
    approach_own = jnp.where(pos[1] < own_col, RIGHT,
                             jnp.where(pos[1] > own_col, LEFT, STAY))
    fallback_limit = jnp.where(partner_style == 1, 30, 10)
    self_fetch = (partner_style == 2) | (wait_time >= fallback_limit)
    partner_style = jnp.where((wait_time >= fallback_limit) & (recipe >= 0), 2,
                              partner_style)
    empty_action = STAY
    empty_action = jnp.where((recipe >= 0) & (pot_count < 3), approach_own, empty_action)
    empty_action = jnp.where(self_fetch & (recipe >= 0) & (pot_count < 3), fetch_own, empty_action)
    empty_action = jnp.where(usable_counter & (pot_count < 3), pick_counter, empty_action)
    empty_action = jnp.where(pot_timer > 0, get_plate, empty_action)
    empty_action = jnp.where(pot_ready, get_plate, empty_action)

    raw_action = _nav_top(obs, pos, jnp.int32(5), jnp.int32(1))
    plate_action = jnp.where(pot_ready,
                             _nav_top(obs, pos, jnp.int32(5), jnp.int32(1)),
                             STAY)
    cooked_action = _nav_top(obs, pos, jnp.int32(5), jnp.int32(0))
    action = empty_action
    action = jnp.where(inv_raw, raw_action, action)
    action = jnp.where(inv_plate & ~inv_cooked, plate_action, action)
    action = jnp.where(inv_cooked, cooked_action, action)

    delivering = inv_cooked & (pos[1] == 5) & (_facing(obs) == 0) & (action == INTERACT)
    next_recipe = jnp.where(delivering, -1, recipe)
    next_wait = jnp.where(delivering, 0, wait_time)
    next_held_safe = jnp.where(delivering,
                               (partner_visible & ~partner_raw).astype(jnp.int32),
                               held_safe)
    next_quarantine = jnp.where(delivering, 1, quarantine)
    nxt = jnp.asarray((next_recipe, next_wait, pot_count, m[3] + 1,
                       partner_style, next_held_safe, next_quarantine), dtype=jnp.int32)
    trace = {'recipe': recipe, 'pot_count': pot_count, 'wait': wait_time,
             'counter_raw': counter_raw.astype(jnp.int32), 'position': pos,
             'usable_counter': usable_counter.astype(jnp.int32),
             'partner0': partner0.astype(jnp.int32), 'partner1': partner1.astype(jnp.int32),
             'delivering': delivering.astype(jnp.int32),
             'partner_style': partner_style,
             'held_safe': held_safe, 'fresh_held': (fresh0 | fresh1).astype(jnp.int32),
             'quarantine': quarantine}
    return action.astype(jnp.int32), nxt, trace


def init_agent_1():
    # recipe, mode, saw dish, pending, previous pot, assist wait,
    # unclaimed-counter wait, next pass-counter column
    return jnp.asarray((-1, 0, 0, 0, 0, 0, 0, 4), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    m = jnp.where(reset, init_agent_1(), memory)
    pos = _localize(obs, True)

    shown0 = jnp.sum(obs[:, :, 40]) > 0
    shown1 = jnp.sum(obs[:, :, 41]) > 0
    recipe = jnp.where(shown0, 0, jnp.where(shown1, 1, m[0]))
    inv0 = obs[2, 2, 7] > 0
    inv1 = obs[2, 2, 8] > 0
    inv_raw = inv0 | inv1

    pot_mask = obs[:, :, 24]
    pot_count = jnp.sum(pot_mask[:, :, None] * obs[:, :, 34:38])
    pot_ready = jnp.sum(pot_mask * obs[:, :, 33]) > 0
    pot_timer = jnp.sum(pot_mask * obs[:, :, 44])
    pot_full = (pot_count >= 3) | (pot_timer > 0) | pot_ready

    wall_raw = jnp.sum(obs[:, :, 22, None] * obs[:, :, 34:38]) > 0
    partner_visible = jnp.sum(obs[:, :, 11]) > 0
    partner_dish = jnp.sum(obs[:, :, 17]) > 0

    mode = jnp.where(pot_full, 1, m[1])
    saw_dish = jnp.where(partner_dish, 1, m[2])
    delivered = (mode == 1) & (saw_dish == 1) & partner_visible & ~partner_dish & ~pot_ready
    mode = jnp.where(delivered, 0, mode)
    saw_dish = jnp.where(delivered, 0, saw_dish)

    # Do not start another supply trip in the short gap between the chef
    # lifting a staged item and depositing it in the pot.
    pending = jnp.where(pot_count > m[4], 0, m[3])
    pending = jnp.where(delivered, 0, pending)
    partner_flat = jnp.argmax(obs[:, :, 11])
    partner_col = pos[1] + (partner_flat % 5) - 2
    partner_self_serving = partner_visible & (((recipe == 0) & (partner_col == 4)) |
                                               ((recipe == 1) & (partner_col == 6)))
    needs_progress = (mode == 0) & (pot_count > 0) & (pot_count < 3) & ~wall_raw & ~inv_raw & (pending == 0) & ~partner_self_serving
    assist_wait = jnp.where(needs_progress & (pot_count == m[4]), m[5] + 1, 0)
    assist_wait = jnp.where(delivered, 0, assist_wait)

    # Identify which symmetric pass holds an unclaimed item while near enough
    # to see it. Indices are clipped only to keep fixed-shape gathers safe.
    counter_row = jnp.clip(4 - pos[0], 0, 4)
    c4_col = jnp.clip(6 - pos[1], 0, 4)
    c6_col = jnp.clip(8 - pos[1], 0, 4)
    c4_visible = (jnp.abs(pos[1] - 4) <= 2) & (pos[0] <= 4)
    c6_visible = (jnp.abs(pos[1] - 6) <= 2) & (pos[0] <= 4)
    c4_has = c4_visible & (jnp.sum(obs[counter_row, c4_col, 34:38]) > 0)
    c6_has = c6_visible & (jnp.sum(obs[counter_row, c6_col, 34:38]) > 0)
    counter_wait = jnp.where((pending > 0) & wall_raw & ~inv_raw, m[6] + 1, 0)
    cycle_choice = reset | delivered | ((m[0] < 0) & (recipe >= 0))
    preferred_col = jnp.where(partner_visible & (partner_col == 6), 6, 4)
    target_col = jnp.where(cycle_choice, preferred_col, m[7])

    wait_action = _nav_lower(obs, pos, jnp.int32(3), jnp.int32(5), jnp.int32(0), False)
    supply_col = jnp.where(target_col == 6, jnp.int32(9), jnp.int32(1))
    left_dir = jnp.where(recipe == 0, jnp.int32(3), jnp.int32(1))
    right_dir = jnp.where(recipe == 0, jnp.int32(2), jnp.int32(1))
    supply_dir = jnp.where(target_col == 6, right_dir, left_dir)
    fetch_action = _nav_lower(obs, pos, jnp.int32(4), supply_col, supply_dir)
    stage_action = _nav_lower(obs, pos, jnp.int32(3), target_col, jnp.int32(0))
    occupied_col = jnp.where(c4_has, jnp.int32(4), jnp.int32(6))
    relocate_action = _nav_lower(obs, pos, jnp.int32(3), occupied_col, jnp.int32(0))

    provide_action = fetch_action
    provide_action = jnp.where(recipe < 0, wait_action, provide_action)
    provide_action = jnp.where(needs_progress & (assist_wait < 8), wait_action, provide_action)
    provide_action = jnp.where((wall_raw | (pending > 0)) & ~inv_raw, wait_action, provide_action)
    provide_action = jnp.where((counter_wait >= 12) & (c4_has | c6_has) & ~inv_raw,
                                relocate_action, provide_action)
    provide_action = jnp.where(inv_raw & ~wall_raw, stage_action, provide_action)
    provide_action = jnp.where(inv_raw & wall_raw, wait_action, provide_action)
    action = jnp.where(mode == 1, wait_action, provide_action)

    stage_issued = (mode == 0) & inv_raw & ~wall_raw & (pos[0] == 3) & (pos[1] == 4) & (_facing(obs) == 0) & (action == INTERACT)
    stage_issued = stage_issued | ((mode == 0) & inv_raw & ~wall_raw & (pos[0] == 3) & (pos[1] == 6) & (_facing(obs) == 0) & (action == INTERACT))
    pickup_issued = (mode == 0) & ~inv_raw & (counter_wait >= 12) & (pos[0] == 3) & (pos[1] == occupied_col) & (_facing(obs) == 0) & (action == INTERACT)
    pending = jnp.where(stage_issued, 1, pending)
    target_col = jnp.where(pickup_issued, jnp.where(occupied_col == 4, 6, 4), target_col)
    counter_wait = jnp.where(pickup_issued | stage_issued, 0, counter_wait)

    nxt = jnp.asarray((recipe, mode, saw_dish, pending, pot_count, assist_wait,
                       counter_wait, target_col), dtype=jnp.int32)
    trace = {'recipe': recipe, 'mode': mode, 'pot_count': pot_count,
             'saw_dish': saw_dish, 'position': pos, 'pending': pending,
             'assist_wait': assist_wait, 'counter_wait': counter_wait,
             'target_col': target_col}
    return action.astype(jnp.int32), nxt, trace
