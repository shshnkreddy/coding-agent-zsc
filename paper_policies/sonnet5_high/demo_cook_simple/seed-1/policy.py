"""Geometry-grounded, role-specialized Overcooked-V2 (demo_cook_simple) policy.

Map facts (fixed, known layout) that this policy exploits, all derivable from
context/environment/README.md and specification.json:

  * The board is two rooms separated by a wall/pot column (col 7). Each seat
    is physically confined to its own room for the whole episode.
  * Only the left room (agent_0's room, cols 1-6) contains the always-on
    recipe indicator (cells (0,5) and (4,5)). Only agent_0 can ever see the
    target recipe.
  * Only the right room (agent_1's room, cols 8-9) contains the plate piles
    ((1,10),(3,10)) and the delivery station (2,10). Only agent_1 can ever
    plate a dish or deliver it.
  * Both rooms contain a supply of ingredient 0 and of ingredient 1 (the only
    two ingredients ever needed, since both configured recipes are pure
    3-of-a-kind of id 0 or id 1). Ingredient 2 is never needed by any recipe.
  * The pot sits at (2,7), directly adjacent to both rooms, so both seats can
    face and interact with it from their own side without crossing rooms.

Because of this, the sensible division of labor (agent_0 fetches ingredients
using its private view of the recipe; agent_1 fetches plates, waits at the
pot, and delivers) is forced by observability/reachability rather than an
arbitrary convention, so it should generalize to independently generated
partners occupying either seat.

Both seats self-localize every step by matching the always-static
observation channels (wall/goal/pot/recipe/button/plate/ingredient-pile,
channels 20-28) against a precomputed table of what that crop looks like from
every reachable floor cell on the known board. This avoids depending on any
unobserved global-position input.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'kitchen_role_split',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# --------------------------------------------------------------------------
# Static board knowledge (fixed layout; see context/environment/README.md).
# --------------------------------------------------------------------------

_BOARD_ROWS = 5
_BOARD_COLS = 11

# Static-channel order matches obs channels 20..28 inclusive:
# wall, goal, pot, recipe_indicator, button_indicator, plate_pile,
# ingredient0, ingredient1, ingredient2.
_WALLS = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10),
    (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (4, 7), (4, 9), (4, 10),
]
_GOAL = [(2, 10)]
_POT = [(2, 7)]
_RECIPE_INDICATOR = [(0, 5), (4, 5)]
_BUTTON_INDICATOR = []
_PLATE_PILE = [(1, 10), (3, 10)]
_ING0 = [(0, 8), (1, 0)]
_ING1 = [(3, 0), (4, 8)]
_ING2 = [(0, 6), (4, 6)]

_STATIC_LAYERS = [_WALLS, _GOAL, _POT, _RECIPE_INDICATOR, _BUTTON_INDICATOR,
                   _PLATE_PILE, _ING0, _ING1, _ING2]
_NUM_STATIC_LAYERS = 9

# All reachable floor cells across both rooms (see spawn_regions in
# specification.json); a player can only ever occupy one of these. Plain
# Python literals only (no calls), built without loops/comprehensions that
# would run at module scope beyond simple literal construction.
_ROOM0_CELLS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
)
_ROOM1_CELLS = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))
_ALL_CELLS = _ROOM0_CELLS + _ROOM1_CELLS
_NUM_CANDS = 24


def _build_constants():
    """Pure-Python (no calls at module scope) construction of lookup arrays.

    Invoked lazily from inside act functions so no function call happens at
    module import time.
    """
    grid = [[[0 for _ in range(_NUM_STATIC_LAYERS)] for _ in range(_BOARD_COLS)]
            for _ in range(_BOARD_ROWS)]
    for channel in range(_NUM_STATIC_LAYERS):
        for (r, c) in _STATIC_LAYERS[channel]:
            grid[r][c][channel] = 1
    static_grid = jnp.asarray(grid, dtype=jnp.int32)
    static_grid_padded = jnp.pad(static_grid, ((2, 2), (2, 2), (0, 0)))
    cand_rows = jnp.asarray([r for r, c in _ALL_CELLS], dtype=jnp.int32)
    cand_cols = jnp.asarray([c for r, c in _ALL_CELLS], dtype=jnp.int32)
    return static_grid_padded, cand_rows, cand_cols

# Channel indices.
_CH_FACE_UP, _CH_FACE_DOWN, _CH_FACE_RIGHT, _CH_FACE_LEFT = 1, 2, 3, 4
_CH_PLATE, _CH_COOKED, _CH_ING0, _CH_ING1 = 5, 6, 7, 8
_CH_ITEM_COOKED = 30
_CH_ITEM_ING0, _CH_ITEM_ING1 = 31, 32
_CH_RECIPE_ING0, _CH_RECIPE_ING1 = 36, 37
_CH_POT_TIMER = 39
_STATIC_START = 20

# Action ids.
_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5

# Room0 (agent_0) landmarks.
_HOME0 = (2, 6)          # sees recipe indicator and pot simultaneously.
_FETCH_STAND = {0: (1, 1), 1: (3, 1)}  # facing LEFT toward ingredient supply.

# Room1 (agent_1) landmarks.
_HOME1 = (2, 8)          # sees the pot (facing LEFT).
_PLATE_STAND = (1, 9)    # facing RIGHT toward plate pile (1, 10).
_DELIVERY_STAND = (2, 9)  # facing RIGHT toward delivery station (2, 10).


def _localize(obs):
    static_grid_padded, cand_rows, cand_cols = _build_constants()
    obs_static = obs[:, :, _STATIC_START:_STATIC_START + _NUM_STATIC_LAYERS].astype(jnp.int32)

    def score_for(idx):
        r = cand_rows[idx]
        c = cand_cols[idx]
        view = jax.lax.dynamic_slice(static_grid_padded, (r, c, 0), (5, 5, _NUM_STATIC_LAYERS))
        return -jnp.sum(jnp.abs(view - obs_static))

    scores = jax.vmap(score_for)(jnp.arange(_NUM_CANDS))
    best = jnp.argmax(scores)
    return cand_rows[best], cand_cols[best]


def _read_cell(obs, self_row, self_col, target_row, target_col, channel):
    li = target_row - self_row + 2
    lj = target_col - self_col + 2
    return jax.lax.dynamic_slice(obs, (li, lj, channel), (1, 1, 1))[0, 0, 0]


def _step_towards(self_row, self_col, target_row, target_col):
    dr = target_row - self_row
    dc = target_col - self_col
    return jnp.where(
        dr > 0, _DOWN,
        jnp.where(dr < 0, _UP,
                  jnp.where(dc > 0, _RIGHT,
                            jnp.where(dc < 0, _LEFT, _STAY))))


# --------------------------------------------------------------------------
# Agent 0: recipe-aware ingredient fetcher.
#
# Memory is (mode, cooldown_counter), both int32:
#   mode: 0 = idle/deciding at home, 1 = fetching ingredient 0,
#         2 = fetching ingredient 1.
#   cooldown_counter: counts down once the pot is observed empty/idle,
#     before a *new* batch (pot currently totally empty) is actually
#     started. This guards against a race: collecting a cooked dish
#     empties the pot immediately, but the recipe display only resamples
#     once the partner actually *delivers* that dish a few steps later.
#     Committing to a fetch target the instant the pot looks empty can
#     therefore lock in a stale (about-to-change) recipe id. Waiting a
#     short, fixed number of steps lets that delivery/resample settle
#     first. Batches already in progress (pot has partial contents) are
#     never subject to this cooldown -- they always continue with the id
#     already present, so ingredients are never mixed.
# --------------------------------------------------------------------------

_NEW_BATCH_COOLDOWN = 16

def init_agent_0():
    return (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))


def agent_0_act(obs, reset, memory, rng):
    del rng
    zero = jnp.asarray(0, dtype=jnp.int32)
    mode_in = jnp.where(reset, zero, memory[0])
    counter_in = jnp.where(reset, zero, memory[1])

    self_row, self_col = _localize(obs)
    at_home = (self_row == _HOME0[0]) & (self_col == _HOME0[1])
    facing_right = obs[2, 2, _CH_FACE_RIGHT] == 1
    facing_left = obs[2, 2, _CH_FACE_LEFT] == 1

    inv0 = obs[2, 2, _CH_ING0]
    inv1 = obs[2, 2, _CH_ING1]
    carrying = (inv0 > 0) | (inv1 > 0)

    pot_ing0 = _read_cell(obs, self_row, self_col, _POT[0][0], _POT[0][1], _CH_ITEM_ING0)
    pot_ing1 = _read_cell(obs, self_row, self_col, _POT[0][0], _POT[0][1], _CH_ITEM_ING1)
    pot_cooked = _read_cell(obs, self_row, self_col, _POT[0][0], _POT[0][1], _CH_ITEM_COOKED)
    pot_timer = _read_cell(obs, self_row, self_col, _POT[0][0], _POT[0][1], _CH_POT_TIMER)
    recipe0 = _read_cell(obs, self_row, self_col, _RECIPE_INDICATOR[0][0], _RECIPE_INDICATOR[0][1], _CH_RECIPE_ING0)

    pot_total = pot_ing0 + pot_ing1
    pot_idle = (pot_timer == 0) & (pot_cooked == 0)
    has_room = pot_total < 3
    pot_empty = pot_total == 0
    can_continue_partial = pot_idle & has_room & (~pot_empty)
    can_start_new = pot_idle & has_room & pot_empty
    # Never mix ingredient ids: continue whatever partial batch already sits
    # in the pot immediately; only consult the recipe display (after the
    # cooldown below) when the pot is completely empty.
    target_id_if_partial = jnp.where(pot_ing0 > 0, 0, 1)
    target_id_if_fresh = jnp.where(recipe0 > 0, 0, 1)

    ready_for_new = can_start_new & (counter_in <= 0)
    waiting_for_new = can_start_new & (counter_in > 0)
    depart = at_home & (can_continue_partial | ready_for_new)
    target_id = jnp.where(pot_empty, target_id_if_fresh, target_id_if_partial)

    # --- Branch: carrying an ingredient -> deliver it to the pot. ---
    action_carry = jnp.where(
        at_home,
        jnp.where(facing_right, _INTERACT, _RIGHT),
        _step_towards(self_row, self_col, _HOME0[0], _HOME0[1]),
    )

    # --- Branch: idle at home, deciding whether/what to fetch. ---
    fetch_row = jnp.where(target_id == 0, _FETCH_STAND[0][0], _FETCH_STAND[1][0])
    fetch_col = _FETCH_STAND[0][1]
    new_mode_idle = jnp.where(depart, jnp.where(target_id == 0, 1, 2), zero)
    action_idle = jnp.where(
        at_home,
        jnp.where(depart, _step_towards(self_row, self_col, fetch_row, fetch_col), _STAY),
        _step_towards(self_row, self_col, _HOME0[0], _HOME0[1]),
    )
    # Counter defaults to a full cooldown whenever a fresh batch is not
    # currently being awaited, so the instant the pot is first observed
    # empty/idle there is a genuine multi-step wait (counting down from
    # _NEW_BATCH_COOLDOWN to 0) before departing, rather than an immediate
    # (zero-wait) departure.
    full_cooldown = jnp.asarray(_NEW_BATCH_COOLDOWN, dtype=jnp.int32)
    new_counter_idle = jnp.where(waiting_for_new, counter_in - 1, full_cooldown)

    # --- Branch: mid-trip, fetching a specific ingredient id. ---
    trip_row = jnp.where(mode_in == 1, _FETCH_STAND[0][0], _FETCH_STAND[1][0])
    trip_col = _FETCH_STAND[0][1]
    at_trip_target = (self_row == trip_row) & (self_col == trip_col)
    action_fetch = jnp.where(
        at_trip_target,
        jnp.where(facing_left, _INTERACT, _LEFT),
        _step_towards(self_row, self_col, trip_row, trip_col),
    )

    action = jnp.where(
        carrying, action_carry,
        jnp.where(mode_in == 0, action_idle, action_fetch),
    )
    next_mode = jnp.where(
        carrying, zero,
        jnp.where(mode_in == 0, new_mode_idle, mode_in),
    )
    next_counter = jnp.where(
        carrying, zero,
        jnp.where(mode_in == 0, new_counter_idle, counter_in),
    )
    next_memory = (
        jnp.where(reset, zero, next_mode).astype(jnp.int32),
        jnp.where(reset, zero, next_counter).astype(jnp.int32),
    )

    return jnp.asarray(action, dtype=jnp.int32), next_memory, {}


# --------------------------------------------------------------------------
# Agent 1: pot-state-driven plater/deliverer. Fully reactive: the current
# inventory alone determines the sub-goal, so no persistent mode is needed.
# --------------------------------------------------------------------------

def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    self_row, self_col = _localize(obs)
    facing_right = obs[2, 2, _CH_FACE_RIGHT] == 1
    facing_left = obs[2, 2, _CH_FACE_LEFT] == 1

    plate_flag = obs[2, 2, _CH_PLATE]
    cooked_flag = obs[2, 2, _CH_COOKED]
    ing0 = obs[2, 2, _CH_ING0]
    ing1 = obs[2, 2, _CH_ING1]

    holding_dish = (plate_flag > 0) & (cooked_flag > 0)
    holding_plate_only = (plate_flag > 0) & (cooked_flag == 0)
    holding_nothing = (plate_flag == 0) & (cooked_flag == 0) & (ing0 == 0) & (ing1 == 0)

    pot_cooked = _read_cell(obs, self_row, self_col, _POT[0][0], _POT[0][1], _CH_ITEM_COOKED)
    pot_timer = _read_cell(obs, self_row, self_col, _POT[0][0], _POT[0][1], _CH_POT_TIMER)
    pot_ready = (pot_cooked > 0) & (pot_timer == 0)

    at_pot = (self_row == _HOME1[0]) & (self_col == _HOME1[1])
    at_delivery = (self_row == _DELIVERY_STAND[0]) & (self_col == _DELIVERY_STAND[1])
    at_plate = (self_row == _PLATE_STAND[0]) & (self_col == _PLATE_STAND[1])

    action_dish = jnp.where(
        at_delivery,
        jnp.where(facing_right, _INTERACT, _RIGHT),
        _step_towards(self_row, self_col, _DELIVERY_STAND[0], _DELIVERY_STAND[1]),
    )
    action_plate_only = jnp.where(
        at_pot,
        jnp.where(facing_left, jnp.where(pot_ready, _INTERACT, _STAY), _LEFT),
        _step_towards(self_row, self_col, _HOME1[0], _HOME1[1]),
    )
    action_nothing = jnp.where(
        at_plate,
        jnp.where(facing_right, _INTERACT, _RIGHT),
        _step_towards(self_row, self_col, _PLATE_STAND[0], _PLATE_STAND[1]),
    )

    action = jnp.where(
        holding_dish, action_dish,
        jnp.where(holding_plate_only, action_plate_only,
                  jnp.where(holding_nothing, action_nothing, _STAY)),
    )

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory).astype(jnp.int32)
    return jnp.asarray(action, dtype=jnp.int32), next_memory, {}
