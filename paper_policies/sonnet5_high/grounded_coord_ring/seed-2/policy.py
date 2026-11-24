"""Overcooked V2 grounded_coord_ring policy.

Board geometry is fixed and known in advance (see context/environment). The
map splits into two disconnected floor regions: an outer 24-cell ring
(agent_0) and an inner 8-cell ring around the pot (agent_1). Both rings are
simple cycles (each cell has exactly two floor neighbours), and every
interactable object is approached from a single adjacent floor cell.

Only agent_1's ring directly borders the pot, so agent_1 alone can complete
the whole recipe -> pot -> plate -> deliver loop and is written as a fully
self-sufficient reactive chef that never depends on a partner's behaviour.
agent_0 cannot reach the pot; it is written as a strictly-helpful assistant
that (a) announces the currently displayed recipe on a recipe button so a
partner confined to the inner ring can read it, and (b) stages one matching
raw ingredient per type on a pass-through counter that borders both rings
(capped, and re-enabled only once the counter is observed empty again, so
agent_0 can never get stuck holding an unplaceable item and go silent for
the rest of the episode). agent_1 opportunistically collects from either of
the two pass-through counters per ingredient type if one is visibly stocked,
else falls back to the ingredient supply. Neither of agent_0's actions can
lower the team's score if an independently generated partner ignores them.

Self-localisation uses no hidden state: the static-object crop returned in
every observation is matched (nearest neighbour) against the known crop
expected at each of the small number of reachable cells, which was verified
offline to be unique per cell for both rings. All map/table construction
below is pure Python executed inside functions (never at module scope), so
it becomes compile-time constant folding once the act functions are traced.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}

# ---------------------------------------------------------------------------
# Literal map data (grounded_coord_ring). No calls at module scope: only
# tuple/list/dict literals.
# ---------------------------------------------------------------------------

# Channel order within the 9-wide static block (obs[...,20:29]) per
# specification.json's static_channel_order plus the three ingredient-pile
# channels: wall, goal, pot, recipe, button, plate, ing0, ing1, ing2.
_WALL, _GOAL, _POT, _RECIPE, _BUTTON, _PLATE, _ING0, _ING1, _ING2 = 0, 1, 2, 3, 4, 5, 6, 7, 8

_WALLS = [(0, 0), (0, 1), (0, 2), (0, 6), (0, 7), (0, 8), (1, 0), (1, 8),
          (2, 0), (2, 2), (2, 3), (2, 5), (2, 6), (2, 8), (6, 0), (6, 2),
          (6, 3), (6, 5), (6, 6), (6, 8), (7, 0), (7, 8), (8, 0), (8, 1),
          (8, 2), (8, 6), (8, 7), (8, 8)]
_GOALS_XY = [(4, 2), (4, 6)]
_POTS_XY = [(4, 4)]
_RECIPES_XY = [(0, 4), (4, 0), (4, 8), (8, 4)]
_BUTTONS_XY = [(2, 4), (6, 4)]
_PLATES_XY = [(3, 6), (5, 6)]
_ING0S_XY = [(3, 2)]
_ING1S_XY = [(5, 2)]
_ING2S_XY = [(0, 3), (0, 5), (3, 0), (3, 8), (5, 0), (5, 8), (8, 3), (8, 5)]

# Outer ring (agent_0) and inner ring (agent_1); both simple cycles listed in
# a consistent (clockwise) walking order.
_CYCLE0 = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 7),
           (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (7, 6), (7, 5), (7, 4),
           (7, 3), (7, 2), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1)]
_CYCLE1 = [(3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (4, 3)]

# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing-channel order (obs[...,1:5] and obs[...,11:15]): UP, DOWN, RIGHT, LEFT.
_DIR_UP, _DIR_DOWN, _DIR_RIGHT, _DIR_LEFT = 0, 1, 2, 3
_FACING_TO_ACTION_L = [A_UP, A_DOWN, A_RIGHT, A_LEFT]

# Goal ids. There are two pass-through counters per ingredient type
# ((2,3)/(2,5) for ingredient 0, (6,3)/(6,5) for ingredient 1) so agent_0 can
# keep staging ingredients even if one counter is already occupied.
(_G_ING0, _G_ING1, _G_PLATE, _G_DELIVERY, _G_BUTTON,
 _G_HANDOFF0, _G_HANDOFF1, _G_POT, _G_HANDOFF0B, _G_HANDOFF1B) = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
_NUM_GOALS = 10

_HANDOFF0A_XY = (2, 3)
_HANDOFF0B_XY = (2, 5)
_HANDOFF1A_XY = (6, 3)
_HANDOFF1B_XY = (6, 5)

_GOAL_ENTRIES0 = [
    (_G_ING0, (3, 1), _DIR_RIGHT),
    (_G_ING1, (5, 1), _DIR_RIGHT),
    (_G_PLATE, (3, 7), _DIR_LEFT),
    (_G_DELIVERY, (4, 1), _DIR_RIGHT),
    (_G_BUTTON, (1, 4), _DIR_DOWN),
    (_G_HANDOFF0, (1, 3), _DIR_DOWN),
    (_G_HANDOFF1, (7, 3), _DIR_UP),
    (_G_POT, (1, 1), _DIR_UP),  # unused placeholder for agent_0
    (_G_HANDOFF0B, (1, 5), _DIR_DOWN),
    (_G_HANDOFF1B, (7, 5), _DIR_UP),
]

_GOAL_ENTRIES1 = [
    (_G_ING0, (3, 3), _DIR_LEFT),
    (_G_ING1, (5, 3), _DIR_LEFT),
    (_G_PLATE, (3, 5), _DIR_RIGHT),
    (_G_DELIVERY, (4, 3), _DIR_LEFT),
    (_G_BUTTON, (3, 4), _DIR_UP),
    (_G_HANDOFF0, (3, 3), _DIR_UP),
    (_G_HANDOFF1, (5, 3), _DIR_DOWN),
    (_G_POT, (4, 3), _DIR_RIGHT),
    (_G_HANDOFF0B, (3, 5), _DIR_UP),  # unused placeholder for agent_1
    (_G_HANDOFF1B, (5, 5), _DIR_DOWN),  # unused placeholder for agent_1
]

# Observation channel offsets.
CH_SELF_FACING = 1  # 1:5 one-hot UP,DOWN,RIGHT,LEFT
CH_SELF_PLATE = 5
CH_SELF_COOKED = 6
CH_SELF_ING0 = 7
CH_SELF_ING1 = 8
CH_STATIC = 20  # 20:29
CH_ITEM_ING0 = 31
CH_ITEM_ING1 = 32
CH_RECIPE_ING0 = 36
CH_RECIPE_ING1 = 37


# ---------------------------------------------------------------------------
# Pure-Python (no numpy) constant builders. These only ever run inside a
# function body (never at module scope), and only depend on the literals
# above, so they fold into compile-time constants once JAX traces the act
# functions.
# ---------------------------------------------------------------------------

def _build_static_grid():
    grid = [[[0 for _ in range(9)] for _ in range(9)] for _ in range(9)]

    def mark(cells, idx):
        for (r, c) in cells:
            grid[r][c][idx] = 1

    mark(_WALLS, _WALL)
    mark(_GOALS_XY, _GOAL)
    mark(_POTS_XY, _POT)
    mark(_RECIPES_XY, _RECIPE)
    mark(_BUTTONS_XY, _BUTTON)
    mark(_PLATES_XY, _PLATE)
    mark(_ING0S_XY, _ING0)
    mark(_ING1S_XY, _ING1)
    mark(_ING2S_XY, _ING2)
    return grid


def _crop_static(grid, r, c):
    out = [[[0 for _ in range(9)] for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            gr, gc = r + i - 2, c + j - 2
            if 0 <= gr < 9 and 0 <= gc < 9:
                out[i][j] = grid[gr][gc]
    return out


def _delta_action(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dc == 1:
        return A_RIGHT
    if dc == -1:
        return A_LEFT
    if dr == 1:
        return A_DOWN
    return A_UP


def _build_ring_tables(grid, cycle):
    n = len(cycle)
    coords = [[r, c] for (r, c) in cycle]
    crops = [_crop_static(grid, r, c) for (r, c) in cycle]
    fwd_action = [0] * n
    bwd_action = [0] * n
    for i in range(n):
        fwd_action[i] = _delta_action(cycle[i], cycle[(i + 1) % n])
        bwd_action[i] = _delta_action(cycle[i], cycle[(i - 1) % n])
    return coords, crops, fwd_action, bwd_action


def _build_goal_tables(cycle, entries):
    idx = [0] * _NUM_GOALS
    face = [0] * _NUM_GOALS
    for (gid, cell, dirn) in entries:
        idx[gid] = cycle.index(cell)
        face[gid] = dirn
    return idx, face


def _constants_for(cycle, goal_entries):
    grid = _build_static_grid()
    coords, crops, fwd, bwd = _build_ring_tables(grid, cycle)
    goal_idx, goal_face = _build_goal_tables(cycle, goal_entries)
    n = len(cycle)
    return {
        'coords': jnp.asarray(coords, dtype=jnp.int32),
        'crops': jnp.asarray(crops, dtype=jnp.int32),
        'fwd': jnp.asarray(fwd, dtype=jnp.int32),
        'bwd': jnp.asarray(bwd, dtype=jnp.int32),
        'goal_idx': jnp.asarray(goal_idx, dtype=jnp.int32),
        'goal_face': jnp.asarray(goal_face, dtype=jnp.int32),
        'n': n,
    }


def _facing_to_action_table():
    return jnp.asarray(_FACING_TO_ACTION_L, dtype=jnp.int32)


def _r_cells():
    return jnp.asarray(_RECIPES_XY, dtype=jnp.int32)


def _l_cells():
    return jnp.asarray(_BUTTONS_XY, dtype=jnp.int32)


def _pot_cell():
    return _POTS_XY[0]


# ---------------------------------------------------------------------------
# Runtime helpers (operate on traced observations).
# ---------------------------------------------------------------------------

def _localize(static_crop, expected_crops):
    diff = expected_crops - static_crop[None]
    sq = jnp.sum(diff * diff, axis=(1, 2, 3))
    return jnp.argmin(sq).astype(jnp.int32)


def _cell_visible_and_empty(obs, cur_rc, cell):
    dr = cell[0] - cur_rc[0] + 2
    dc = cell[1] - cur_rc[1] + 2
    visible = (dr >= 0) & (dr < 5) & (dc >= 0) & (dc < 5)
    total = jnp.asarray(0, jnp.int32)
    for ch in (29, 30, 31, 32, 33):
        total = total + _read_at(obs, dr, dc, ch)
    return visible, total == 0


def _read_at(obs, row, col, channel):
    valid = (row >= 0) & (row < 5) & (col >= 0) & (col < 5)
    rc = jnp.clip(row, 0, 4)
    cc = jnp.clip(col, 0, 4)
    val = obs[rc, cc, channel]
    return jnp.where(valid, val, 0)


def _scan_recipe(obs, cur_rc, cells):
    def body(carry, cell):
        found, ing = carry
        dr = cell[0] - cur_rc[0] + 2
        dc = cell[1] - cur_rc[1] + 2
        v0 = _read_at(obs, dr, dc, CH_RECIPE_ING0)
        v1 = _read_at(obs, dr, dc, CH_RECIPE_ING1)
        this_found = (v0 > 0) | (v1 > 0)
        this_ing = jnp.where(v0 > 0, 0, 1)
        found_next = found | this_found
        ing_next = jnp.where(found, ing, jnp.where(this_found, this_ing, ing))
        return (found_next, ing_next), None

    init = (jnp.asarray(False), jnp.asarray(0, jnp.int32))
    (found, ing), _ = jax.lax.scan(body, init, cells)
    return found, ing


def _nav_action(cur_index, goal_index, fwd_table, bwd_table, n):
    fwd_dist = jnp.mod(goal_index - cur_index, n)
    bwd_dist = jnp.mod(cur_index - goal_index, n)
    go_fwd = fwd_dist <= bwd_dist
    return jnp.where(go_fwd, fwd_table[cur_index], bwd_table[cur_index])


def _step_common(obs, cur_index, goal_id, ring, facing_to_action, interact_override):
    approach_index = ring['goal_idx'][goal_id]
    required_facing = ring['goal_face'][goal_id]
    self_facing = jnp.argmax(obs[2, 2, CH_SELF_FACING:CH_SELF_FACING + 4]).astype(jnp.int32)
    at_goal = cur_index == approach_index
    facing_ok = self_facing == required_facing
    move_action = _nav_action(cur_index, approach_index, ring['fwd'], ring['bwd'], ring['n'])
    turn_action = facing_to_action[required_facing]
    interact_action = jnp.where(interact_override, jnp.asarray(A_STAY, jnp.int32),
                                 jnp.asarray(A_INTERACT, jnp.int32))
    at_goal_action = jnp.where(facing_ok, interact_action, turn_action)
    action = jnp.where(at_goal, at_goal_action, move_action)
    return action, at_goal, facing_ok


# ---------------------------------------------------------------------------
# agent_0: outer-ring helper. Reads the always-on recipe indicators (free),
# announces recipe changes on a button so an inner-ring partner can see them,
# then (capped, self-healing) stages one matching raw ingredient per type on
# a pass-through counter.
# ---------------------------------------------------------------------------

def init_agent_0():
    return {
        'recipe': jnp.asarray(-1, dtype=jnp.int32),
        'announced': jnp.asarray(-1, dtype=jnp.int32),
        'staged0': jnp.asarray(0, dtype=jnp.int32),
        'staged1': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    ring = _constants_for(_CYCLE0, _GOAL_ENTRIES0)
    facing_to_action = _facing_to_action_table()
    r_cells = _r_cells()
    l_cells = _l_cells()

    zero_i = jnp.asarray(0, jnp.int32)
    prev_recipe = jnp.where(reset, jnp.asarray(-1, jnp.int32), memory['recipe'])
    prev_announced = jnp.where(reset, jnp.asarray(-1, jnp.int32), memory['announced'])
    prev_staged0 = jnp.where(reset, zero_i, memory['staged0'])
    prev_staged1 = jnp.where(reset, zero_i, memory['staged1'])

    static_crop = obs[:, :, CH_STATIC:CH_STATIC + 9]
    cur_index = _localize(static_crop, ring['crops'])
    cur_rc = ring['coords'][cur_index]

    found_r, ing_r = _scan_recipe(obs, cur_rc, r_cells)
    found_l, ing_l = _scan_recipe(obs, cur_rc, l_cells)
    found = found_r | found_l
    scanned_ing = jnp.where(found_r, ing_r, ing_l)
    new_recipe = jnp.where(found, scanned_ing, prev_recipe)

    self_ing0 = obs[2, 2, CH_SELF_ING0]
    self_ing1 = obs[2, 2, CH_SELF_ING1]
    holding_ing0 = self_ing0 > 0
    holding_ing1 = self_ing1 > 0
    holding_any = holding_ing0 | holding_ing1

    need_info = new_recipe == -1
    need_announce = (~need_info) & (new_recipe != prev_announced)
    button_phase = need_info | need_announce

    # A pass-through counter can only hold one item, and nothing guarantees a
    # partner ever collects it, so staging is capped at one ingredient per
    # type until the counter is confirmed empty again (self-healing): this
    # keeps agent_0 from ever getting permanently stuck holding an item it
    # cannot place, which would otherwise silence the (more broadly useful)
    # recipe announcements on the button for the rest of the episode.
    vis0, empty0 = _cell_visible_and_empty(obs, cur_rc, _HANDOFF0A_XY)
    vis1, empty1 = _cell_visible_and_empty(obs, cur_rc, _HANDOFF1A_XY)
    staged0 = jnp.where(vis0 & empty0, zero_i, prev_staged0)
    staged1 = jnp.where(vis1 & empty1, zero_i, prev_staged1)

    handoff_goal = jnp.where(holding_ing0, jnp.asarray(_G_HANDOFF0, jnp.int32),
                              jnp.asarray(_G_HANDOFF1, jnp.int32))
    ing0_available = staged0 == 0
    ing1_available = staged1 == 0
    can_fetch = jnp.where(new_recipe == 0, ing0_available, ing1_available)
    ing_goal = jnp.where(new_recipe == 0, jnp.asarray(_G_ING0, jnp.int32),
                          jnp.asarray(_G_ING1, jnp.int32))

    goal_id = jnp.where(
        holding_any, handoff_goal,
        jnp.where(button_phase, jnp.asarray(_G_BUTTON, jnp.int32),
                  jnp.where(can_fetch, ing_goal, jnp.asarray(_G_BUTTON, jnp.int32))))
    idle_at_button = (~button_phase) & (~holding_any) & (~can_fetch)

    action, at_goal, facing_ok = _step_common(obs, cur_index, goal_id, ring, facing_to_action,
                                               need_info | idle_at_button)

    pressed = at_goal & facing_ok & need_announce & (goal_id == _G_BUTTON) & (~idle_at_button)
    announced_next = jnp.where(pressed, new_recipe, prev_announced)

    just_dropped0 = holding_ing0 & at_goal & facing_ok & (goal_id == _G_HANDOFF0)
    just_dropped1 = holding_ing1 & at_goal & facing_ok & (goal_id == _G_HANDOFF1)
    staged0_next = jnp.where(just_dropped0, jnp.asarray(1, jnp.int32), staged0)
    staged1_next = jnp.where(just_dropped1, jnp.asarray(1, jnp.int32), staged1)

    next_memory = {
        'recipe': new_recipe.astype(jnp.int32),
        'announced': announced_next.astype(jnp.int32),
        'staged0': staged0_next.astype(jnp.int32),
        'staged1': staged1_next.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, {}


# ---------------------------------------------------------------------------
# agent_1: inner-ring chef. Fully self-sufficient: learns the recipe from a
# button (R is never in view from this ring), gathers matching ingredients,
# cooks, plates, and delivers, re-verifying the recipe after every delivery.
# ---------------------------------------------------------------------------

def init_agent_1():
    return {
        'recipe': jnp.asarray(-1, dtype=jnp.int32),
        'prev_dish': jnp.asarray(0, dtype=jnp.int32),
        'ing_src': jnp.asarray(-1, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    ring = _constants_for(_CYCLE1, _GOAL_ENTRIES1)
    facing_to_action = _facing_to_action_table()
    l_cells = _l_cells()
    pot_cell = _pot_cell()

    prev_recipe = jnp.where(reset, jnp.asarray(-1, jnp.int32), memory['recipe'])
    prev_dish = jnp.where(reset, jnp.asarray(0, jnp.int32), memory['prev_dish'])
    prev_ing_src = jnp.where(reset, jnp.asarray(-1, jnp.int32), memory['ing_src'])

    static_crop = obs[:, :, CH_STATIC:CH_STATIC + 9]
    cur_index = _localize(static_crop, ring['crops'])
    cur_rc = ring['coords'][cur_index]

    found_l, ing_l = _scan_recipe(obs, cur_rc, l_cells)
    scanned_recipe = jnp.where(found_l, ing_l, prev_recipe)

    self_plate = obs[2, 2, CH_SELF_PLATE]
    self_cooked = obs[2, 2, CH_SELF_COOKED]
    self_ing0 = obs[2, 2, CH_SELF_ING0]
    self_ing1 = obs[2, 2, CH_SELF_ING1]

    holding_dish = self_cooked > 0
    holding_plate_only = (self_plate > 0) & (self_cooked == 0)
    holding_ing = (self_ing0 > 0) | (self_ing1 > 0)
    empty_handed = (self_plate == 0) & (self_cooked == 0) & (self_ing0 == 0) & (self_ing1 == 0)

    delivered = (prev_dish > 0) & empty_handed
    recipe = jnp.where(delivered, jnp.asarray(-1, jnp.int32), scanned_recipe)

    pot_dr = pot_cell[0] - cur_rc[0] + 2
    pot_dc = pot_cell[1] - cur_rc[1] + 2
    pot_ing0 = _read_at(obs, pot_dr, pot_dc, CH_ITEM_ING0)
    pot_ing1 = _read_at(obs, pot_dr, pot_dc, CH_ITEM_ING1)
    pot_count = jnp.where(recipe == 0, pot_ing0, pot_ing1)

    need_info = recipe == -1
    need_ingredient = (~need_info) & (pot_count < 3)

    # Opportunistically collect a matching ingredient a helper may have
    # staged on a pass-through counter instead of walking to the supply;
    # falls back to the supply whenever nothing is visibly staged. The
    # choice is only made once per gathering phase (committed to memory):
    # re-deciding every step from instantaneous visibility would make the
    # agent oscillate as the counter enters/leaves the 5x5 crop while
    # walking.
    ing0_at_a = _read_at(obs, _HANDOFF0A_XY[0] - cur_rc[0] + 2, _HANDOFF0A_XY[1] - cur_rc[1] + 2,
                          CH_ITEM_ING0) > 0
    ing0_at_b = _read_at(obs, _HANDOFF0B_XY[0] - cur_rc[0] + 2, _HANDOFF0B_XY[1] - cur_rc[1] + 2,
                          CH_ITEM_ING0) > 0
    ing1_at_a = _read_at(obs, _HANDOFF1A_XY[0] - cur_rc[0] + 2, _HANDOFF1A_XY[1] - cur_rc[1] + 2,
                          CH_ITEM_ING1) > 0
    ing1_at_b = _read_at(obs, _HANDOFF1B_XY[0] - cur_rc[0] + 2, _HANDOFF1B_XY[1] - cur_rc[1] + 2,
                          CH_ITEM_ING1) > 0
    ing0_goal = jnp.where(ing0_at_a, jnp.asarray(_G_HANDOFF0, jnp.int32),
                           jnp.where(ing0_at_b, jnp.asarray(_G_HANDOFF0B, jnp.int32),
                                     jnp.asarray(_G_ING0, jnp.int32)))
    ing1_goal = jnp.where(ing1_at_a, jnp.asarray(_G_HANDOFF1, jnp.int32),
                           jnp.where(ing1_at_b, jnp.asarray(_G_HANDOFF1B, jnp.int32),
                                     jnp.asarray(_G_ING1, jnp.int32)))
    fresh_ing_goal = jnp.where(recipe == 0, ing0_goal, ing1_goal)
    is_ing_phase = (~holding_dish) & (~holding_plate_only) & (~holding_ing) & need_ingredient
    committed = prev_ing_src != -1
    chosen_ing_goal = jnp.where(committed, prev_ing_src, fresh_ing_goal)
    next_ing_src = jnp.where(is_ing_phase, chosen_ing_goal, jnp.asarray(-1, jnp.int32))

    goal_id = jnp.where(
        holding_dish, jnp.asarray(_G_DELIVERY, jnp.int32),
        jnp.where(
            holding_plate_only, jnp.asarray(_G_POT, jnp.int32),
            jnp.where(
                holding_ing, jnp.asarray(_G_POT, jnp.int32),
                jnp.where(
                    need_info, jnp.asarray(_G_BUTTON, jnp.int32),
                    jnp.where(need_ingredient, chosen_ing_goal, jnp.asarray(_G_PLATE, jnp.int32))))))

    action, _, _ = _step_common(obs, cur_index, goal_id, ring, facing_to_action, jnp.asarray(False))

    next_memory = {
        'recipe': recipe.astype(jnp.int32),
        'prev_dish': self_cooked.astype(jnp.int32),
        'ing_src': next_ing_src.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, {}
