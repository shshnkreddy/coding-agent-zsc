"""Cooperative Overcooked-v2 policy for demo_cook_wide.

Map analysis (fixed, known layout):
  Agent_0's reachable room is exactly 3 floor cells (1,4),(1,5),(1,6): a
  straight corridor. From it, both ingredient supplies (0 at (1,3), 1 at
  (1,7)), both plate piles (0,4)/(0,6), the delivery station (0,5) and the
  pot (2,5) are all directly interactable, but the recipe indicator (4,5)
  is never in view (view radius 2 never reaches row 4 from row 1).
  Agent_1's reachable room is the other 13 floor cells (row 3 cols 1-9 plus
  stub cells (4,1),(4,2),(4,8),(4,9)). The recipe indicator (4,5) is visible
  from (3,5), and the pot (2,5) is also directly interactable from (3,5)
  (it sits between the two rooms). Agent_1 never reaches a plate pile or the
  delivery station.

Division of labor forced by this geometry (not a private convention):
  - agent_1 is the only seat that can see the target recipe, so it is
    responsible for seeding the pot with a correctly-typed ingredient.
  - agent_0 can top off an already-started (non-empty, idle) pot using its
    own fast local ingredient access, and is the only seat that can plate
    and deliver. agent_0 never seeds an empty pot (that would require a
    blind 50/50 guess, which is useless in expectation and can actively
    corrupt a competent partner's correct seed).
  - Both seats refuse to add an ingredient whose type mismatches an
    already-non-empty pot, since a mixed dish can never be a correct
    delivery and there is no way to remove ingredients.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'room_aware_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# ---------------------------------------------------------------------------
# Action / direction constants (see context/environment/README.md).
# ---------------------------------------------------------------------------
ACT_RIGHT, ACT_DOWN, ACT_LEFT, ACT_UP, ACT_STAY, ACT_INTERACT = 0, 1, 2, 3, 4, 5
DIR_UP, DIR_DOWN, DIR_RIGHT, DIR_LEFT = 0, 1, 2, 3


def _dir_to_action(direction):
    # Movement action that makes the player face `direction` (no lookup
    # array needed, so no module-scope call/constant is required).
    return jnp.where(
        direction == DIR_UP, ACT_UP,
        jnp.where(direction == DIR_DOWN, ACT_DOWN,
                  jnp.where(direction == DIR_RIGHT, ACT_RIGHT, ACT_LEFT)))


# Observation channel indices.
CH_SELF_PLATE = 5
CH_SELF_COOKED = 6
CH_SELF_ING0 = 7
CH_SELF_ING1 = 8
CH_POT = 24                   # static "pot" marker
CH_ITEM_COOKED = 33
CH_ITEM_ING0 = 34
CH_ITEM_ING1 = 35
CH_RECIPE_ING0 = 40
CH_RECIPE_ING1 = 41
CH_POT_TIMER = 44
CH_STATIC_LO, CH_STATIC_HI = 22, 32  # static + ingredient-pile block (10 chans)


def _self_facing(obs):
    return jnp.argmax(obs[2, 2, 1:5])


# ---------------------------------------------------------------------------
# Static-map fingerprints used to localize agent_1 within its 13-cell room.
# These are pure compile-time constants derived from the known, fixed
# layout; no runtime host I/O is involved.
# ---------------------------------------------------------------------------
_WALLS = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10),
    (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
    (3, 0), (3, 10), (4, 3), (4, 7),
    (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10),
]
_GOAL = [(0, 5)]
_POT = [(2, 5)]
_RECIPE_IND = [(4, 5)]
_BUTTON_IND = []
_PLATE_PILE = [(0, 4), (0, 6)]
_ING0 = [(1, 3), (4, 0), (4, 10)]
_ING1 = [(1, 7), (5, 1), (5, 9)]
_ING2 = []
_ING3 = [(4, 4), (4, 6)]

_CHANNEL_COORDS = (
    _WALLS, _GOAL, _POT, _RECIPE_IND, _BUTTON_IND,
    _PLATE_PILE, _ING0, _ING1, _ING2, _ING3,
)

ROOM1_CELLS = (
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 1), (4, 2), (4, 8), (4, 9),
)
ROOM1_ROWS_LIST = (3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4)
ROOM1_COLS_LIST = (1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 8, 9)


def _static_cell(row, col):
    # Pure-Python (untraced) helper: builds one literal length-10 static
    # feature vector for a fixed, known board cell. No numpy, no host I/O.
    values = []
    for channel in range(10):
        coords = _CHANNEL_COORDS[channel]
        hit = 0
        for (cr, cc) in coords:
            if cr == row and cc == col:
                hit = 1
                break
        values.append(hit)
    return values


def _crop_fingerprint(row, col):
    rows = []
    for i in range(5):
        cols = []
        for j in range(5):
            gr, gc = row + i - 2, col + j - 2
            if 0 <= gr < 6 and 0 <= gc < 11:
                cols.append(_static_cell(gr, gc))
            else:
                cols.append([0] * 10)
        rows.append(cols)
    return rows


def _localize_room1(obs):
    # All fingerprint construction below is pure-Python over fixed, known
    # board coordinates (no dependence on traced values); it unrolls to
    # plain JAX constants when this function is traced.
    fingerprints = [_crop_fingerprint(r, c) for (r, c) in ROOM1_CELLS]
    room1_fp = jnp.asarray(fingerprints, dtype=jnp.int32)
    room1_rows = jnp.asarray(ROOM1_ROWS_LIST, dtype=jnp.int32)
    room1_cols = jnp.asarray(ROOM1_COLS_LIST, dtype=jnp.int32)

    obs_static = obs[:, :, CH_STATIC_LO:CH_STATIC_HI].astype(jnp.int32)
    diffs = jnp.sum(jnp.abs(room1_fp - obs_static[None, :, :, :]), axis=(1, 2, 3))
    idx = jnp.argmin(diffs)
    return room1_rows[idx], room1_cols[idx]


# ---------------------------------------------------------------------------
# Agent 0: corridor cook / plater / deliverer.
# ---------------------------------------------------------------------------
def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def _room0_nav_act(target_col, desired_dir, self_col, self_facing):
    at_target = self_col == target_col
    move = jnp.where(target_col > self_col, ACT_RIGHT, ACT_LEFT)
    face_act = jnp.where(self_facing == desired_dir, ACT_INTERACT, _dir_to_action(desired_dir))
    arrived_act = jnp.where(desired_dir >= 0, face_act, ACT_STAY)
    return jnp.where(at_target, arrived_act, move)


def agent_0_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)

    # The pot is always visible from anywhere in agent_0's 3-cell corridor;
    # its local column encodes self's true global column (pot is at col 5).
    pot_flat = jnp.argmax(obs[:, :, CH_POT].reshape(-1))
    pot_r, pot_c = pot_flat // 5, pot_flat % 5
    self_col = 7 - pot_c
    self_facing = _self_facing(obs)

    pot_cooked = obs[pot_r, pot_c, CH_ITEM_COOKED]
    pot_timer = obs[pot_r, pot_c, CH_POT_TIMER]
    pot_i0 = obs[pot_r, pot_c, CH_ITEM_ING0]
    pot_i1 = obs[pot_r, pot_c, CH_ITEM_ING1]
    pot_count = pot_i0 + pot_i1
    pot_idle = (pot_cooked == 0) & (pot_timer == 0)

    s_plate = obs[2, 2, CH_SELF_PLATE]
    s_cooked = obs[2, 2, CH_SELF_COOKED]
    s_i0 = obs[2, 2, CH_SELF_ING0]
    s_i1 = obs[2, 2, CH_SELF_ING1]

    holding_plated_dish = (s_plate == 1) & (s_cooked == 1)
    holding_empty_plate = (s_plate == 1) & (s_cooked == 0) & (s_i0 == 0) & (s_i1 == 0)
    holding_nothing = (s_plate == 0) & (s_cooked == 0) & (s_i0 == 0) & (s_i1 == 0)
    holding_ing0 = s_i0 == 1
    holding_ing1 = s_i1 == 1
    holding_ing = holding_ing0 | holding_ing1

    pot_ready = (pot_cooked == 1) & (pot_timer == 0)
    pot_needs_fill = pot_idle & (pot_count > 0) & (pot_count < 3)
    fetch0 = pot_i0 > 0

    held_type0_safe = pot_idle & (pot_count < 3) & ((pot_count == 0) | (pot_i0 > 0))
    held_type1_safe = pot_idle & (pot_count < 3) & ((pot_count == 0) | (pot_i1 > 0))
    safe_deposit = jnp.where(holding_ing0, held_type0_safe, held_type1_safe)

    # Earlier revisions trusted any ingredient sitting on the counters
    # flanking the pot ((2,4)/(2,6)) as a deliberate hand-off signal, seeding
    # an empty pot from it directly. Exploratory cross-play against a
    # synthetic partner that leaves a leftover ingredient near the pot
    # (e.g. an extra top-off drop that outlives its cooking cycle) showed
    # this is exploitable: a stale ingredient can survive across a recipe
    # resample and get blindly trusted as a fresh seed for the *new*
    # recipe, producing a wrong, cooked dish and a real -20 penalty
    # (observed mean return fell to +5 with episodes as low as -60, versus
    # 0 and never negative without this trust). Agent_0 cannot verify a
    # counter item's type against the current recipe (it never sees the
    # recipe), so unlike topping off an already-*pot*-resident ingredient
    # (whose type is fixed by an actual, timestamped deposit interaction
    # this episode), a counter item carries no freshness guarantee at all.
    # The safer default is to never seed from an unverified external
    # source; only ever build on what is already validly in the pot.

    cond_deliver = holding_plated_dish
    cond_collect = pot_ready & holding_empty_plate
    cond_deposit = holding_ing & safe_deposit
    cond_hold_wait = holding_ing & (~safe_deposit)
    cond_fetch_ing = holding_nothing & pot_needs_fill
    cond_fetch_plate = holding_nothing & pot_ready
    # While the pot is cooking (timer > 0) there is nothing to add or
    # collect yet, so use the ~20 idle steps productively: fetch an empty
    # plate now instead of waiting until the pot is actually ready.
    cond_prefetch_plate = holding_nothing & (pot_timer > 0)

    conds = [cond_deliver, cond_collect, cond_deposit, cond_hold_wait,
             cond_fetch_ing, cond_fetch_plate, cond_prefetch_plate]
    tc_choices = [5, 5, 5, 5, jnp.where(fetch0, 4, 6), 4, 4]
    dd_choices = [DIR_UP, DIR_DOWN, DIR_DOWN, -1,
                  jnp.where(fetch0, DIR_LEFT, DIR_RIGHT), DIR_UP, DIR_UP]

    target_col = jnp.select(conds, tc_choices, default=5)
    desired_dir = jnp.select(conds, dd_choices, default=-1)

    action = _room0_nav_act(target_col, desired_dir, self_col, self_facing)
    action = jnp.asarray(action, dtype=jnp.int32)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    trace = {
        'self_col': jnp.asarray(self_col, dtype=jnp.int32),
        'pot_count': jnp.asarray(pot_count, dtype=jnp.int32),
        'pot_i0': jnp.asarray(pot_i0, dtype=jnp.int32),
        'pot_i1': jnp.asarray(pot_i1, dtype=jnp.int32),
        's_i0': jnp.asarray(s_i0, dtype=jnp.int32),
        's_i1': jnp.asarray(s_i1, dtype=jnp.int32),
        'holding_plated_dish': jnp.asarray(holding_plated_dish, dtype=jnp.int32),
    }
    return action, next_memory, trace


# ---------------------------------------------------------------------------
# Agent 1: recipe scout / pot seeder.
# ---------------------------------------------------------------------------
STALL_THRESHOLD = 10


def init_agent_1():
    return {
        'believed_type': jnp.asarray(-1, dtype=jnp.int32),
        'fetch_type': jnp.asarray(-1, dtype=jnp.int32),
        'discard': jnp.asarray(0, dtype=jnp.int32),
        'stall_count': jnp.asarray(0, dtype=jnp.int32),
        'last_count': jnp.asarray(0, dtype=jnp.int32),
    }


def _room1_nav_action(self_row, self_col, target_row, target_col):
    same_row = self_row == target_row
    same_col = self_col == target_col
    arrived = same_row & same_col

    low_cluster_self = self_col <= 2
    high_cluster_self = self_col >= 8
    low_cluster_tgt = target_col <= 2
    high_cluster_tgt = target_col >= 8
    same_cluster = (low_cluster_self & low_cluster_tgt) | (high_cluster_self & high_cluster_tgt)

    need_up_first = (self_row == 4) & (~((target_row == 4) & same_cluster))
    move_horiz = jnp.where(target_col > self_col, ACT_RIGHT, ACT_LEFT)

    action = jnp.where(
        need_up_first, ACT_UP,
        jnp.where(self_row == target_row, move_horiz,
                  jnp.where(self_col != target_col, move_horiz, ACT_DOWN)))
    return action, arrived


def agent_1_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)

    self_row, self_col = _localize_room1(obs)
    self_facing = _self_facing(obs)
    at_center = (self_row == 3) & (self_col == 5)

    s_i0 = obs[2, 2, CH_SELF_ING0]
    s_i1 = obs[2, 2, CH_SELF_ING1]
    holding_ing = (s_i0 == 1) | (s_i1 == 1)
    holding_type0 = s_i0 == 1

    believed_type = memory['believed_type']
    fetch_type = memory['fetch_type']
    discard = memory['discard']
    stall_count = memory['stall_count']
    last_count = memory['last_count']

    # Fixed local offsets, valid only when at_center (self == (3,5)):
    # pot (2,5) -> local (1,2); recipe indicator (4,5) -> local (3,2).
    pot_cooked = obs[1, 2, CH_ITEM_COOKED]
    pot_timer = obs[1, 2, CH_POT_TIMER]
    pot_i0 = obs[1, 2, CH_ITEM_ING0]
    pot_i1 = obs[1, 2, CH_ITEM_ING1]
    pot_count = pot_i0 + pot_i1
    pot_idle = (pot_cooked == 0) & (pot_timer == 0)

    recipe0_shown = obs[3, 2, CH_RECIPE_ING0] > 0
    recipe1_shown = obs[3, 2, CH_RECIPE_ING1] > 0
    observed_type = jnp.where(recipe0_shown, 0, jnp.where(recipe1_shown, 1, believed_type))
    new_believed = jnp.where(at_center, observed_type, believed_type)

    # Agent_1's default job is only to seed an *empty* pot with the
    # currently-believed type. Topping off a partially filled pot is
    # normally agent_0's job (it is much closer to the supplies); agent_1
    # racing to "help" fill would mean it is frequently away on a redundant
    # fetch trip when a delivery happens and the recipe resamples, so it
    # could return still carrying a now-stale ingredient type. Restricting
    # the *default* case to pot_count == 0 keeps the gap between "read the
    # recipe" and "commit to fetch" essentially zero.
    #
    # Backstop: an independently generated partner in agent_0's seat is not
    # guaranteed to complete a partial pot on its own (only geometry forces
    # it to be the delivering seat, not that it must proactively top off).
    # If a partial, idle pot's ingredient count has not changed for many
    # consecutive steps while we sit here watching it, no one else is going
    # to finish it: fall back to fetching the rest ourselves. The long
    # threshold means this can essentially never fire against a competent,
    # fast-topping-off agent_0 (self-play finishes a top-off in well under
    # ten steps), so it only activates once genuinely stalled.
    waiting_partial = pot_idle & (pot_count > 0) & (pot_count < 3)
    count_unchanged = pot_count == last_count
    accumulating = at_center & waiting_partial & (~holding_ing) & (fetch_type < 0)
    new_stall = jnp.where(accumulating & count_unchanged, stall_count + 1,
                           jnp.where(at_center, jnp.asarray(0, dtype=jnp.int32), stall_count))
    new_last_count = jnp.where(at_center, pot_count, last_count)

    needs_seed = pot_idle & (pot_count == 0)
    needs_topoff_backstop = waiting_partial & (stall_count >= STALL_THRESHOLD)
    commit = (at_center & (needs_seed | needs_topoff_backstop)
              & (new_believed >= 0) & (fetch_type < 0) & (~holding_ing))
    new_fetch_type = jnp.where(commit, new_believed, fetch_type)

    # --- Deposit state: holding an ingredient -> go to (3,5), face up. ---
    nav_deposit, arr_deposit = _room1_nav_action(self_row, self_col, 3, 5)
    face_deposit = jnp.where(self_facing == DIR_UP, ACT_INTERACT, _dir_to_action(DIR_UP))
    ready_deposit = arr_deposit & (face_deposit == ACT_INTERACT)
    pot_available = pot_idle & (pot_count < 3)
    held_type0_safe = (pot_count == 0) | (pot_i0 > 0)
    held_type1_safe = (pot_count == 0) | (pot_i1 > 0)
    pot_contents_match = jnp.where(holding_type0, held_type0_safe, held_type1_safe)
    # Defense in depth: even if the pot's existing contents look compatible
    # (e.g. empty), refuse to deposit unless the *freshly re-read* recipe
    # (we are at (3,5), so the indicator is in view) still matches what we
    # are carrying. This protects against a delivery + recipe resample
    # happening while this ingredient was being fetched.
    recipe_match = jnp.where(holding_type0, new_believed == 0, new_believed == 1)
    safe_deposit = pot_available & pot_contents_match & recipe_match & (new_believed >= 0)
    # If the pot simply isn't available yet (full/cooking), the mismatch is
    # only temporary: wait. If it IS available but our held ingredient is
    # wrong for it (stale fetch or someone else seeded a different type),
    # waiting can never resolve that on its own: drop the ingredient on a
    # nearby counter instead of holding it (and blocking new fetches)
    # forever.
    should_discard_now = ready_deposit & pot_available & (~safe_deposit)
    deposit_act = jnp.where(
        ready_deposit,
        jnp.where(safe_deposit, ACT_INTERACT, ACT_STAY),
        face_deposit)
    action_deposit = jnp.where(arr_deposit, deposit_act, nav_deposit)

    # --- Discard state: holding a now-useless ingredient -> drop it on the
    # counter at (2,2), reached by standing at (3,2) facing up. Agent_0
    # cannot reach or interact with (2,2) at all (it is outside agent_0's
    # connected floor region), and never trusts an unverified counter item
    # as a pot seed, so a discarded, possibly-stale ingredient is inert.
    nav_discard, arr_discard = _room1_nav_action(self_row, self_col, 3, 2)
    face_discard = jnp.where(self_facing == DIR_UP, ACT_INTERACT, _dir_to_action(DIR_UP))
    action_discard = jnp.where(arr_discard, face_discard, nav_discard)

    do_discard = discard > 0
    action_hold = jnp.where(do_discard, action_discard, action_deposit)
    next_discard = jnp.where(~holding_ing, jnp.asarray(0, dtype=jnp.int32),
                              jnp.where(should_discard_now, jnp.asarray(1, dtype=jnp.int32), discard))

    # --- Fetch state: fetch_type committed -> go to (4,1), face L/DOWN. ---
    nav_fetch, arr_fetch = _room1_nav_action(self_row, self_col, 4, 1)
    fetch_dir = jnp.where(fetch_type == 0, DIR_LEFT, DIR_DOWN)
    face_fetch = jnp.where(self_facing == fetch_dir, ACT_INTERACT, _dir_to_action(fetch_dir))
    action_fetch = jnp.where(arr_fetch, face_fetch, nav_fetch)
    picked_up_now = arr_fetch & (self_facing == fetch_dir)

    # --- Decide state: idle -> go to (3,5), observe, maybe commit fetch. ---
    nav_decide, arr_decide = _room1_nav_action(self_row, self_col, 3, 5)
    action_decide = jnp.where(arr_decide, ACT_STAY, nav_decide)

    is_fetch_state = (~holding_ing) & (fetch_type >= 0)
    action = jnp.where(
        holding_ing, action_hold,
        jnp.where(is_fetch_state, action_fetch, action_decide))
    action = jnp.asarray(action, dtype=jnp.int32)

    out_fetch_type = jnp.where(
        holding_ing, jnp.asarray(-1, dtype=jnp.int32),
        jnp.where(is_fetch_state,
                  jnp.where(picked_up_now, jnp.asarray(-1, dtype=jnp.int32), fetch_type),
                  new_fetch_type))
    out_believed_type = new_believed

    # Once a fetch is committed (seed or backstop top-off), freeze the stall
    # counter's inputs so a long fetch round trip doesn't itself look like
    # more stalling once we return; new_stall/new_last_count already reset
    # to 0/current on the same step commit fires (fetch_type>=0 forces
    # accumulating False), so this simply carries that forward.
    next_memory = {
        'believed_type': jnp.where(reset, jnp.asarray(-1, dtype=jnp.int32), out_believed_type),
        'fetch_type': jnp.where(reset, jnp.asarray(-1, dtype=jnp.int32), out_fetch_type),
        'discard': jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), next_discard),
        'stall_count': jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), new_stall),
        'last_count': jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), new_last_count),
    }
    trace = {
        'self_row': jnp.asarray(self_row, dtype=jnp.int32),
        'self_col': jnp.asarray(self_col, dtype=jnp.int32),
        'believed_type': jnp.asarray(out_believed_type, dtype=jnp.int32),
        'recipe0_shown': jnp.asarray(recipe0_shown, dtype=jnp.int32),
        'recipe1_shown': jnp.asarray(recipe1_shown, dtype=jnp.int32),
        'pot_i0': jnp.asarray(pot_i0, dtype=jnp.int32),
        'pot_i1': jnp.asarray(pot_i1, dtype=jnp.int32),
        's_i0': jnp.asarray(s_i0, dtype=jnp.int32),
        's_i1': jnp.asarray(s_i1, dtype=jnp.int32),
        'at_center': jnp.asarray(at_center, dtype=jnp.int32),
        'stall_count': jnp.asarray(new_stall, dtype=jnp.int32),
    }
    return action, next_memory, trace
