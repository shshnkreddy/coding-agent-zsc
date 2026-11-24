"""Policy for grounded_coord_ring (Overcooked V2, XP objective).

Map facts (fixed, from context/environment):
  9x9 board. Pot at (4,4). Delivery (goal) at (4,2) and (4,6).
  Plate piles at (3,6) and (5,6). Ingredient supply 0 at (3,2), supply 1 at
  (5,2). Always-on recipe indicator (R) at (0,4),(4,0),(4,8),(8,4). Button
  recipe indicator (L) at (2,4) and (6,4). Shared handoff counters between
  the two rooms sit at (2,3),(2,5),(6,3),(6,5).

  agent_1 (inner room) occupies the 8-cell ring around the pot:
  (3,3),(3,4),(3,5),(4,3),(4,5),(5,3),(5,4),(5,5). From any of these cells
  every relevant object (pot, both supplies, both plates, both deliveries,
  both buttons) is reachable by facing the right neighbor, so agent_1 can
  complete an entire cook/plate/deliver cycle alone.

  agent_0 (outer ring) occupies the 24-cell ring one step further out and
  can never reach the pot directly, but can reach both ingredient supplies
  (from the far side), both plate piles, both deliveries, all recipe
  indicators/buttons, and the four shared handoff counters bordering the
  inner room.

Both act functions locate themselves each step by matching the observed
static-object layout (channels 20-28) against precomputed templates for
every reachable cell on the fixed map (no global position is otherwise
available). Once the current cell is known, movement is a simple walk
along the fixed ring topology, and any other landmark's exact offset can be
computed directly from known map coordinates. All constant tables are built
from plain-Python literals inside helper functions (never at module scope)
and converted to JAX arrays there.
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

# Action / facing constants (see context/environment/README.md).
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

_ASCII_ROWS = (
    "WWW2R2WWW",
    "W       W",
    "W WWLWW W",
    "2 0   B 2",
    "RAXAP X R",
    "2 1   B 2",
    "W WWLWW W",
    "W       W",
    "WWW2R2WWW",
)
_TYPE_OF_CHAR = {
    'W': 1, 'X': 2, 'P': 3, 'R': 4, 'L': 5, 'B': 6, '0': 7, '1': 8, '2': 9,
    'A': 0, ' ': 0,
}

_RING1_COORDS = ((3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (4, 3))
_RING0_COORDS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
    (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
)


def _build_grid():
    grid = []
    for row in _ASCII_ROWS:
        grid_row = []
        for ch in row:
            grid_row.append(_TYPE_OF_CHAR[ch])
        grid.append(grid_row)
    return grid


def _build_template(grid, r, c):
    tmpl = []
    for i in (-2, -1, 0, 1, 2):
        row = []
        for j in (-2, -1, 0, 1, 2):
            gr, gc = r + i, c + j
            if 0 <= gr < 9 and 0 <= gc < 9:
                row.append(grid[gr][gc])
            else:
                row.append(0)
        tmpl.append(row)
    return tmpl


def _action_for_delta(dr, dc):
    if dr == 1:
        return DOWN
    if dr == -1:
        return UP
    if dc == 1:
        return RIGHT
    return LEFT


def _build_fwd_bwd(coords):
    n = len(coords)
    fwd = []
    bwd = []
    for i in range(n):
        r0, c0 = coords[i]
        r1, c1 = coords[(i + 1) % n]
        fwd.append(_action_for_delta(r1 - r0, c1 - c0))
        rm1, cm1 = coords[(i - 1) % n]
        bwd.append(_action_for_delta(rm1 - r0, cm1 - c0))
    return fwd, bwd


def _ring_index(coords, target):
    for i, coord in enumerate(coords):
        if coord == target:
            return i
    raise ValueError("target not in ring")


def _build_ring1_tables():
    grid = _build_grid()
    templates = [_build_template(grid, r, c) for (r, c) in _RING1_COORDS]
    fwd, bwd = _build_fwd_bwd(_RING1_COORDS)
    idx = lambda coord: _ring_index(_RING1_COORDS, coord)
    return {
        'coords': jnp.asarray(_RING1_COORDS, dtype=jnp.int32),
        'templates': jnp.asarray(templates, dtype=jnp.int32),
        'fwd': jnp.asarray(fwd, dtype=jnp.int32),
        'bwd': jnp.asarray(bwd, dtype=jnp.int32),
        'supply_ring': jnp.asarray([idx((3, 3)), idx((5, 3))], dtype=jnp.int32),
        'supply_facing': jnp.asarray([LEFT, LEFT], dtype=jnp.int32),
        'plate_ring': jnp.asarray([idx((3, 5)), idx((5, 5))], dtype=jnp.int32),
        'plate_facing': jnp.asarray([RIGHT, RIGHT], dtype=jnp.int32),
        'pot_ring': jnp.asarray(
            [idx((3, 4)), idx((4, 5)), idx((5, 4)), idx((4, 3))], dtype=jnp.int32),
        'pot_facing': jnp.asarray([DOWN, LEFT, UP, RIGHT], dtype=jnp.int32),
        'delivery_ring': jnp.asarray([idx((4, 5)), idx((4, 3))], dtype=jnp.int32),
        'delivery_facing': jnp.asarray([RIGHT, LEFT], dtype=jnp.int32),
        'button_ring': jnp.asarray([idx((3, 4)), idx((5, 4))], dtype=jnp.int32),
        'button_facing': jnp.asarray([UP, DOWN], dtype=jnp.int32),
    }


def _build_ring0_tables():
    grid = _build_grid()
    templates = [_build_template(grid, r, c) for (r, c) in _RING0_COORDS]
    fwd, bwd = _build_fwd_bwd(_RING0_COORDS)
    idx = lambda coord: _ring_index(_RING0_COORDS, coord)
    counter_coords = ((2, 3), (2, 5), (6, 3), (6, 5))
    return {
        'coords': jnp.asarray(_RING0_COORDS, dtype=jnp.int32),
        'templates': jnp.asarray(templates, dtype=jnp.int32),
        'fwd': jnp.asarray(fwd, dtype=jnp.int32),
        'bwd': jnp.asarray(bwd, dtype=jnp.int32),
        'r_ring': jnp.asarray(
            [idx((1, 4)), idx((4, 1)), idx((4, 7)), idx((7, 4))], dtype=jnp.int32),
        'r_facing': jnp.asarray([UP, LEFT, RIGHT, DOWN], dtype=jnp.int32),
        'supply_ring': jnp.asarray([idx((3, 1)), idx((5, 1))], dtype=jnp.int32),
        'supply_facing': jnp.asarray([RIGHT, RIGHT], dtype=jnp.int32),
        'plate_ring': jnp.asarray([idx((3, 7)), idx((5, 7))], dtype=jnp.int32),
        'plate_facing': jnp.asarray([LEFT, LEFT], dtype=jnp.int32),
        'delivery_ring': jnp.asarray([idx((4, 1)), idx((4, 7))], dtype=jnp.int32),
        'delivery_facing': jnp.asarray([RIGHT, LEFT], dtype=jnp.int32),
        'counter_coords': jnp.asarray(counter_coords, dtype=jnp.int32),
        'counter_ring': jnp.asarray(
            [idx((1, 3)), idx((1, 5)), idx((7, 3)), idx((7, 5))], dtype=jnp.int32),
        'counter_facing': jnp.asarray([DOWN, DOWN, UP, UP], dtype=jnp.int32),
    }


def _pot_coord():
    return jnp.asarray([4, 4], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Shared per-step helpers.
# ---------------------------------------------------------------------------


def _static_type_grid(obs):
    weights = jnp.arange(1, 10, dtype=obs.dtype)
    return jnp.sum(obs[:, :, 20:29] * weights[None, None, :], axis=-1)


def _match_ring_index(type_grid, templates):
    diff = jnp.sum(jnp.abs(templates - type_grid[None, :, :]), axis=(1, 2))
    return jnp.argmin(diff).astype(jnp.int32)


def _current_facing_action(obs):
    up, down, right, left = obs[2, 2, 1], obs[2, 2, 2], obs[2, 2, 3], obs[2, 2, 4]
    return (up * UP + down * DOWN + right * RIGHT + left * LEFT).astype(jnp.int32)


def _nav_action(ring_idx, cur_facing_action, target_idx, target_facing, n, fwd_tbl, bwd_tbl):
    at_target = ring_idx == target_idx
    facing_ok = cur_facing_action == target_facing
    act_at_target = jnp.where(facing_ok, INTERACT, target_facing)
    fd = jnp.mod(target_idx - ring_idx, n)
    bd = jnp.mod(ring_idx - target_idx, n)
    move_fwd = fd <= bd
    act_move = jnp.where(move_fwd, fwd_tbl[ring_idx], bwd_tbl[ring_idx])
    return jnp.where(at_target, act_at_target, act_move).astype(jnp.int32)


def _nearest(ring_idx, cand_ring, cand_facing, n):
    fd = jnp.mod(cand_ring - ring_idx, n)
    bd = jnp.mod(ring_idx - cand_ring, n)
    dist = jnp.minimum(fd, bd)
    best = jnp.argmin(dist)
    return cand_ring[best], cand_facing[best]


def _scan_recipe(obs):
    ing0_total = jnp.sum(obs[:, :, 36])
    ing1_total = jnp.sum(obs[:, :, 37])
    other_total = (
        jnp.sum(obs[:, :, 34]) + jnp.sum(obs[:, :, 35]) + jnp.sum(obs[:, :, 38])
    )
    found = (ing0_total + ing1_total + other_total) > 0
    recipe_val = jnp.where(ing1_total > ing0_total, 1, 0).astype(jnp.int32)
    return found, recipe_val


def _read_cell(obs, r, c):
    r = jnp.clip(r, 0, 4).astype(jnp.int32)
    c = jnp.clip(c, 0, 4).astype(jnp.int32)
    return jax.lax.dynamic_slice(obs, (r, c, 0), (1, 1, obs.shape[-1]))[0, 0, :]


def _self_inventory(obs):
    return obs[2, 2, 5:10]


# ---------------------------------------------------------------------------
# agent_1: inner-room self-sufficient cook loop.
# ---------------------------------------------------------------------------


def init_agent_1():
    return {
        'have_recipe': jnp.asarray(0, dtype=jnp.int32),
        'recipe_known': jnp.asarray(0, dtype=jnp.int32),
        'delivered_count': jnp.asarray(0, dtype=jnp.int32),
        'activation_count': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    t = _build_ring1_tables()
    zero = jnp.asarray(0, dtype=jnp.int32)
    mem_have = jnp.where(reset, zero, memory['have_recipe'])
    mem_recipe = jnp.where(reset, zero, memory['recipe_known'])
    mem_delivered = jnp.where(reset, zero, memory['delivered_count'])
    mem_activation = jnp.where(reset, zero, memory['activation_count'])

    obs = obs.astype(jnp.int32)
    type_grid = _static_type_grid(obs)
    ring_idx = _match_ring_index(type_grid, t['templates'])
    self_coord = t['coords'][ring_idx]
    cur_facing = _current_facing_action(obs)

    inv = _self_inventory(obs)
    plate_flag, cooked_flag = inv[0], inv[1]
    ing0, ing1 = inv[2], inv[3]
    holding_empty = (plate_flag + cooked_flag + ing0 + ing1 + inv[4]) == 0
    holding_dish = cooked_flag > 0
    holding_plate_only = (plate_flag > 0) & (cooked_flag == 0)
    holding_ingredient = (~holding_dish) & (~holding_plate_only) & (~holding_empty)

    pot_coord = _pot_coord()
    pot_r = pot_coord[0] - self_coord[0] + 2
    pot_c = pot_coord[1] - self_coord[1] + 2
    pot_cell = _read_cell(obs, pot_r, pot_c)
    pot_ing_total = pot_cell[31] + pot_cell[32] + pot_cell[33]
    pot_ready = pot_cell[30] > 0
    pot_needs_more = pot_ing_total < 3
    del pot_ready

    found_display, recipe_val = _scan_recipe(obs)
    have_recipe = jnp.where(found_display, 1, mem_have)
    recipe_known = jnp.where(found_display, recipe_val, mem_recipe)

    b_deliver = holding_dish
    b_no_recipe = (~b_deliver) & (have_recipe == 0)
    b_have_plate = (~b_deliver) & (~b_no_recipe) & holding_plate_only
    b_have_ingredient = (~b_deliver) & (~b_no_recipe) & holding_ingredient
    b_need_plate = (~b_deliver) & (~b_no_recipe) & holding_empty & (~pot_needs_more)
    b_need_ingredient = (~b_deliver) & (~b_no_recipe) & holding_empty & pot_needs_more

    d_ring, d_facing = _nearest(ring_idx, t['delivery_ring'], t['delivery_facing'], 8)
    act_deliver = _nav_action(ring_idx, cur_facing, d_ring, d_facing, 8, t['fwd'], t['bwd'])

    bt_ring, bt_facing = _nearest(ring_idx, t['button_ring'], t['button_facing'], 8)
    act_button = _nav_action(ring_idx, cur_facing, bt_ring, bt_facing, 8, t['fwd'], t['bwd'])

    p_ring, p_facing = _nearest(ring_idx, t['pot_ring'], t['pot_facing'], 8)
    act_pot = _nav_action(ring_idx, cur_facing, p_ring, p_facing, 8, t['fwd'], t['bwd'])

    pl_ring, pl_facing = _nearest(ring_idx, t['plate_ring'], t['plate_facing'], 8)
    act_plate = _nav_action(ring_idx, cur_facing, pl_ring, pl_facing, 8, t['fwd'], t['bwd'])

    s_target = t['supply_ring'][recipe_known]
    s_facing = t['supply_facing'][recipe_known]
    act_supply = _nav_action(ring_idx, cur_facing, s_target, s_facing, 8, t['fwd'], t['bwd'])

    action = jnp.asarray(STAY, dtype=jnp.int32)
    action = jnp.where(b_need_ingredient, act_supply, action)
    action = jnp.where(b_need_plate, act_plate, action)
    action = jnp.where(b_have_ingredient, act_pot, action)
    action = jnp.where(b_have_plate, act_pot, action)
    action = jnp.where(b_no_recipe, act_button, action)
    action = jnp.where(b_deliver, act_deliver, action)
    action = action.astype(jnp.int32)

    just_delivered = b_deliver & (action == INTERACT)
    just_activated = b_no_recipe & (action == INTERACT)
    next_have_recipe = jnp.where(just_delivered, 0, have_recipe).astype(jnp.int32)
    next_recipe_known = recipe_known.astype(jnp.int32)
    next_delivered = (mem_delivered + just_delivered.astype(jnp.int32)).astype(jnp.int32)
    next_activation = (mem_activation + just_activated.astype(jnp.int32)).astype(jnp.int32)

    next_memory = {
        'have_recipe': next_have_recipe,
        'recipe_known': next_recipe_known,
        'delivered_count': next_delivered,
        'activation_count': next_activation,
    }
    trace = {
        'ring_idx': ring_idx.astype(jnp.int32),
        'have_recipe': have_recipe.astype(jnp.int32),
        'recipe_known': recipe_known.astype(jnp.int32),
        'pot_ing_total': pot_ing_total.astype(jnp.int32),
        'delivered_count': next_delivered,
        'activation_count': next_activation,
    }
    return action, next_memory, trace


# ---------------------------------------------------------------------------
# agent_0: outer-ring helper.
# ---------------------------------------------------------------------------


def init_agent_0():
    return {
        'have_recipe': jnp.asarray(0, dtype=jnp.int32),
        'recipe_known': jnp.asarray(0, dtype=jnp.int32),
        'counter_cursor': jnp.asarray(0, dtype=jnp.int32),
        'attempted_drop': jnp.asarray(0, dtype=jnp.int32),
        # Per shared counter: 0 empty, 1 ingredient, 2 plate, 3 plated dish.
        'counter_kind': jnp.zeros((4,), dtype=jnp.int32),
        # Per shared counter: sticky flag for a dish confirmed to mismatch
        # the known recipe, so it is never picked up (and never delivered)
        # again.
        'bad_dish': jnp.zeros((4,), dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    t = _build_ring0_tables()
    zero = jnp.asarray(0, dtype=jnp.int32)
    zero4 = jnp.zeros((4,), dtype=jnp.int32)
    mem_have = jnp.where(reset, zero, memory['have_recipe'])
    mem_recipe = jnp.where(reset, zero, memory['recipe_known'])
    mem_cursor = jnp.where(reset, zero, memory['counter_cursor'])
    mem_attempted = jnp.where(reset, zero, memory['attempted_drop'])
    mem_kind = jnp.where(reset, zero4, memory['counter_kind'])
    mem_bad_dish = jnp.where(reset, zero4, memory['bad_dish'])

    obs = obs.astype(jnp.int32)
    type_grid = _static_type_grid(obs)
    ring_idx = _match_ring_index(type_grid, t['templates'])
    self_coord = t['coords'][ring_idx]
    cur_facing = _current_facing_action(obs)

    inv = _self_inventory(obs)
    plate_flag, cooked_flag = inv[0], inv[1]
    ing0, ing1 = inv[2], inv[3]
    holding_empty = (plate_flag + cooked_flag + ing0 + ing1 + inv[4]) == 0
    holding_dish = cooked_flag > 0
    holding_other = (~holding_dish) & (~holding_empty)

    found_display, recipe_val = _scan_recipe(obs)
    have_recipe = jnp.where(found_display, 1, mem_have)
    recipe_known = jnp.where(found_display, recipe_val, mem_recipe)

    kind = mem_kind
    counter_coords = t['counter_coords']
    for k in range(4):
        off_r = counter_coords[k, 0] - self_coord[0] + 2
        off_c = counter_coords[k, 1] - self_coord[1] + 2
        visible = (off_r >= 0) & (off_r <= 4) & (off_c >= 0) & (off_c <= 4)
        cell = _read_cell(obs, off_r, off_c)
        cell_plate, cell_cooked = cell[29], cell[30]
        cell_ing_total = cell[31] + cell[32] + cell[33]
        observed_kind = jnp.where(
            cell_cooked > 0, 3, jnp.where(cell_plate > 0, 2, jnp.where(cell_ing_total > 0, 1, 0))
        ).astype(jnp.int32)
        kind = kind.at[k].set(jnp.where(visible, observed_kind, kind[k]))

    good_dish_mask = (kind == 3) & (mem_bad_dish == 0)
    any_dish = jnp.any(good_dish_mask)
    dish_k = jnp.argmax(good_dish_mask)
    count_ing = jnp.sum(kind == 1)
    count_plate = jnp.sum(kind == 2)
    any_empty = jnp.any(kind == 0)
    empty_k = jnp.argmax(kind == 0)
    del empty_k

    # A picked-up dish might not actually match the (freely observed)
    # recipe, e.g. if a partner cooked it before a recipe change. Only
    # deliver when it is verifiably correct; otherwise treat it like any
    # other held item to be quarantined on a counter instead of risking the
    # -20 wrong-delivery penalty.
    dish_matches = jnp.where(
        have_recipe == 0, True,
        jnp.where(recipe_known == 0, (ing0 == 3) & (ing1 == 0), (ing1 == 3) & (ing0 == 0)),
    )
    holding_bad_dish = holding_dish & (~dish_matches)
    holding_other = holding_other | holding_bad_dish

    b_deliver = holding_dish & (~holding_bad_dish)
    b_pickup_dish = (~holding_dish) & holding_empty & any_dish
    b_no_recipe = (~b_deliver) & (~b_pickup_dish) & (have_recipe == 0) & (~holding_other)
    b_carrying = (~b_deliver) & (~b_pickup_dish) & (~b_no_recipe) & holding_other
    b_fetch = (~b_deliver) & (~b_pickup_dish) & (~b_no_recipe) & (~b_carrying) & holding_empty

    dl_ring, dl_facing = _nearest(ring_idx, t['delivery_ring'], t['delivery_facing'], 24)
    act_deliver = _nav_action(ring_idx, cur_facing, dl_ring, dl_facing, 24, t['fwd'], t['bwd'])

    pd_ring = t['counter_ring'][dish_k]
    pd_facing = t['counter_facing'][dish_k]
    act_pickup = _nav_action(ring_idx, cur_facing, pd_ring, pd_facing, 24, t['fwd'], t['bwd'])

    r_ring, r_facing = _nearest(ring_idx, t['r_ring'], t['r_facing'], 24)
    act_read = _nav_action(ring_idx, cur_facing, r_ring, r_facing, 24, t['fwd'], t['bwd'])

    # Target a counter slot believed empty; if the last attempt at our
    # previously-chosen slot just failed (still holding something), exclude
    # it. If no slot is currently believed empty at all, fall back to a
    # plain round-robin patrol (advance one slot per step) so we keep moving
    # past every counter and refresh our (possibly stale) beliefs instead of
    # freezing forever on a single unreachable target.
    retry = (mem_attempted == 1) & holding_other
    idx4 = jnp.arange(4)
    kind_for_target = jnp.where(retry & (idx4 == mem_cursor), 1, kind)
    believed_empty = jnp.any(kind_for_target == 0)
    target_k = jnp.where(
        believed_empty, jnp.argmax(kind_for_target == 0), jnp.mod(mem_cursor + 1, 4)
    )
    c_ring = t['counter_ring'][target_k]
    c_facing = t['counter_facing'][target_k]
    act_carry = _nav_action(ring_idx, cur_facing, c_ring, c_facing, 24, t['fwd'], t['bwd'])

    # Prefer staging whichever of {ingredient, plate} is currently scarcer on
    # the shared counters, so a partner needing either can find it, without
    # ever piling up so much of one kind that the other can no longer fit.
    ing_target = t['supply_ring'][recipe_known]
    ing_facing = t['supply_facing'][recipe_known]
    pl_ring, pl_facing_arr = _nearest(ring_idx, t['plate_ring'], t['plate_facing'], 24)
    want_plate = (count_ing > 0) & (count_plate == 0)
    fetch_target = jnp.where(want_plate, pl_ring, ing_target)
    fetch_facing = jnp.where(want_plate, pl_facing_arr, ing_facing)
    act_fetch = _nav_action(ring_idx, cur_facing, fetch_target, fetch_facing, 24, t['fwd'], t['bwd'])

    action = jnp.asarray(STAY, dtype=jnp.int32)
    action = jnp.where(b_fetch, act_fetch, action)
    action = jnp.where(b_carrying, act_carry, action)
    action = jnp.where(b_no_recipe, act_read, action)
    action = jnp.where(b_pickup_dish, act_pickup, action)
    action = jnp.where(b_deliver, act_deliver, action)
    action = action.astype(jnp.int32)

    picked_up_now = b_pickup_dish & (action == INTERACT)
    kind = jnp.where(picked_up_now, kind.at[dish_k].set(0), kind)

    carry_interact_now = b_carrying & (action == INTERACT)
    next_attempted = jnp.where(b_carrying, carry_interact_now.astype(jnp.int32), 0)
    next_cursor = jnp.where(b_carrying, target_k, mem_cursor)

    mark_bad_now = carry_interact_now & holding_bad_dish
    bad_dish_next = jnp.where(mark_bad_now, mem_bad_dish.at[target_k].set(1), mem_bad_dish)

    next_memory = {
        'have_recipe': have_recipe.astype(jnp.int32),
        'recipe_known': recipe_known.astype(jnp.int32),
        'counter_cursor': next_cursor.astype(jnp.int32),
        'attempted_drop': next_attempted.astype(jnp.int32),
        'counter_kind': kind.astype(jnp.int32),
        'bad_dish': bad_dish_next.astype(jnp.int32),
    }
    trace = {
        'ring_idx': ring_idx.astype(jnp.int32),
        'have_recipe': have_recipe.astype(jnp.int32),
        'recipe_known': recipe_known.astype(jnp.int32),
        'kind': kind.astype(jnp.int32),
        'b_fetch': b_fetch.astype(jnp.int32),
        'b_carrying': b_carrying.astype(jnp.int32),
        'b_pickup_dish': b_pickup_dish.astype(jnp.int32),
        'b_deliver': b_deliver.astype(jnp.int32),
    }
    return action, next_memory, trace
