"""Overcooked-v2 `demo_cook_wide` cooperative policy.

The layout hard-codes the division of labour, so no private convention is
needed (and none is used):

* Seat 0 is sealed inside the three-cell top room ``(1,4) (1,5) (1,6)``.  From
  there it reaches both ingredient piles ``(1,3)=ing0`` and ``(1,7)=ing1``,
  both plate piles, the pot ``(2,5)`` and the delivery station ``(0,5)``.  Its
  5x5 view never contains row 4, so it can *never* read the recipe indicator.
* Seat 1 roams the bottom room.  It reads the always-on recipe indicator
  ``(4,5)``, reaches ingredient piles of both types and the pot, but it can
  never touch a plate or the delivery station.  The two rooms are disjoint, so
  the seats can never collide.

The only thing that has to travel between the seats is *which ingredient the
recipe wants*, and the only convention-free carriers are physical: an
ingredient sitting in the pot, an ingredient carried by the partner while in
view, or an ingredient left on a shared counter.  Seat 0 reads those three
sources in that order of trust and never guesses while evidence is available;
seat 1 publishes them by carrying and inserting the right ingredient.  Both
halves are useful with an arbitrary partner: seat 0 is a complete solo cook
once it knows the target, and every seat-1 insertion both informs and advances
the dish.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'evidence_cook_informant',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# ---------------------------------------------------------------------------
# Public layout knowledge, precomputed as literals (no module-scope calls).
#
# Category codes of the six static object channels plus the four ingredient
# pile channels, i.e. 1=wall 2=goal 3=pot 4=recipe-indicator 5=button
# 6=plate-pile 7..10=ingredient pile 0..3, 0=floor or off-map.
# ---------------------------------------------------------------------------

VIEW = 2
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5
FACE_ACT_L = [A_UP, A_DOWN, A_RIGHT, A_LEFT]

# seat 0 reachable cells: (1,4) (1,5) (1,6)
ROW0_L = [1, 1, 1]
COL0_L = [4, 5, 6]
TPL0_L = [[[0, 0, 0, 0, 0], [1, 1, 6, 2, 6], [1, 7, 0, 0, 0], [1, 1, 1, 3, 1], [0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0], [1, 6, 2, 6, 1], [7, 0, 0, 0, 8], [1, 1, 3, 1, 1], [0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0], [6, 2, 6, 1, 1], [0, 0, 0, 8, 1], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]]]
NAV0_L = [[4, 0, 0], [2, 4, 0], [2, 2, 4]]

# seat 1 reachable cells, in index order
ROW1_L = [3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]
COL1_L = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 8, 9]
TPL1_L = [[[0, 1, 1, 1, 7], [0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 7, 0, 0, 1], [0, 1, 8, 1, 1]],
          [[1, 1, 1, 7, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [7, 0, 0, 1, 10], [1, 8, 1, 1, 1]],
          [[1, 1, 7, 0, 0], [1, 1, 1, 1, 3], [0, 0, 0, 0, 0], [0, 0, 1, 10, 4], [8, 1, 1, 1, 1]],
          [[1, 7, 0, 0, 0], [1, 1, 1, 3, 1], [0, 0, 0, 0, 0], [0, 1, 10, 4, 10], [1, 1, 1, 1, 1]],
          [[7, 0, 0, 0, 8], [1, 1, 3, 1, 1], [0, 0, 0, 0, 0], [1, 10, 4, 10, 1], [1, 1, 1, 1, 1]],
          [[0, 0, 0, 8, 1], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0], [10, 4, 10, 1, 0], [1, 1, 1, 1, 1]],
          [[0, 0, 8, 1, 1], [3, 1, 1, 1, 1], [0, 0, 0, 0, 0], [4, 10, 1, 0, 0], [1, 1, 1, 1, 8]],
          [[0, 8, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [10, 1, 0, 0, 7], [1, 1, 1, 8, 1]],
          [[8, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 0, 0, 7, 0], [1, 1, 8, 1, 0]],
          [[0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 7, 0, 0, 1], [0, 1, 8, 1, 1], [0, 0, 0, 0, 0]],
          [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [7, 0, 0, 1, 10], [1, 8, 1, 1, 1], [0, 0, 0, 0, 0]],
          [[1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [10, 1, 0, 0, 7], [1, 1, 1, 8, 1], [0, 0, 0, 0, 0]],
          [[1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 0, 0, 7, 0], [1, 1, 8, 1, 0], [0, 0, 0, 0, 0]]]
NAV1_L = [[4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
          [2, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
          [2, 2, 4, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
          [2, 2, 2, 4, 0, 0, 0, 0, 0, 2, 2, 0, 0],
          [2, 2, 2, 2, 4, 0, 0, 0, 0, 2, 2, 0, 0],
          [2, 2, 2, 2, 2, 4, 0, 0, 0, 2, 2, 0, 0],
          [2, 2, 2, 2, 2, 2, 4, 0, 0, 2, 2, 0, 0],
          [2, 2, 2, 2, 2, 2, 2, 4, 0, 2, 2, 1, 1],
          [2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 1, 1],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0, 3, 3],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 3, 3],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4]]
DIST1_L = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 8, 9],
           [1, 0, 1, 2, 3, 4, 5, 6, 7, 2, 1, 7, 8],
           [2, 1, 0, 1, 2, 3, 4, 5, 6, 3, 2, 6, 7],
           [3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 5, 6],
           [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 4, 5],
           [5, 4, 3, 2, 1, 0, 1, 2, 3, 6, 5, 3, 4],
           [6, 5, 4, 3, 2, 1, 0, 1, 2, 7, 6, 2, 3],
           [7, 6, 5, 4, 3, 2, 1, 0, 1, 8, 7, 1, 2],
           [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 2, 1],
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 9, 10],
           [2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 0, 8, 9],
           [8, 7, 6, 5, 4, 3, 2, 1, 2, 9, 8, 0, 1],
           [9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 1, 0]]

# seat 0 jobs: (stand cell index, facing, does it end in INTERACT)
J0_WAIT, J0_POT, J0_DELIVER, J0_ING0, J0_ING1 = 0, 1, 2, 3, 4
J0_PLATE_L, J0_PLATE_R, J0_CNT_L, J0_CNT_R = 5, 6, 7, 8
S0_STAND_L = [1, 1, 1, 0, 2, 0, 2, 0, 2]
S0_FACE_L = [F_DOWN, F_DOWN, F_UP, F_LEFT, F_RIGHT, F_UP, F_UP, F_DOWN, F_DOWN]
S0_INT_L = [0, 1, 1, 1, 1, 1, 1, 1, 1]

# seat 1 jobs
J1_WAIT, J1_POT, J1_ING0_L, J1_ING1_L, J1_ING0_R = 0, 1, 2, 3, 4
J1_ING1_R, J1_STG_L, J1_STG_R, J1_BIN_L, J1_BIN_R = 5, 6, 7, 8, 9
J1_HIDE = 10
S1_STAND_L = [4, 4, 9, 9, 12, 12, 2, 6, 10, 11, 9]
S1_FACE_L = [F_UP, F_UP, F_LEFT, F_DOWN, F_RIGHT, F_DOWN,
             F_DOWN, F_DOWN, F_DOWN, F_DOWN, F_LEFT]
S1_INT_L = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

SOLO_AFTER = 60     # steps without any evidence before seat 0 cooks on a hunch
COMMIT_WAIT = 20    # steps seat 0 holds the first unit waiting for confirmation
PATIENCE = 15       # steps seat 1 defers to an imminent recipe resample


def _i32(values):
    return jnp.asarray(values, dtype=jnp.int32)


def _categories(obs):
    """5x5 map-category grid implied by the static observation channels."""
    weights = jnp.arange(1, 11, dtype=jnp.int32)
    return jnp.sum(obs[:, :, 22:32] * weights[None, None, :], axis=2)


def _localize(obs, templates):
    grid = _categories(obs)
    diff = jnp.sum(jnp.abs(templates - grid[None, :, :]), axis=(1, 2))
    return jnp.argmin(diff).astype(jnp.int32)


def _cell(obs, row, col, gr, gc):
    """Full 45-vector of global cell (gr, gc); zeros when outside the crop."""
    lr = gr - row + VIEW
    lc = gc - col + VIEW
    inside = (lr >= 0) & (lr < 2 * VIEW + 1) & (lc >= 0) & (lc < 2 * VIEW + 1)
    value = obs[jnp.clip(lr, 0, 2 * VIEW), jnp.clip(lc, 0, 2 * VIEW)]
    return jnp.where(inside, value, jnp.zeros_like(value)), inside


def _facing(obs):
    return jnp.argmax(obs[VIEW, VIEW, 1:5]).astype(jnp.int32)


def _item_kind(cell):
    """(holds a raw ingredient, its type, holds an empty plate, is empty)."""
    counts = cell[34:38]
    plate = cell[32] > 0
    cooked = cell[33] > 0
    is_ing = (~plate) & (~cooked) & (jnp.sum(counts) > 0)
    kind = jnp.argmax(counts).astype(jnp.int32)
    return is_ing, kind, plate & (~cooked), jnp.sum(cell[32:38]) == 0


def _single_type(counts):
    """Ingredient id when exactly one recipe-legal type is present, else -1."""
    has0 = counts[0] > 0
    has1 = counts[1] > 0
    other = (counts[2] > 0) | (counts[3] > 0)
    one = jnp.int32(1)
    zero = jnp.int32(0)
    none = jnp.int32(-1)
    return jnp.where(has0 & (~has1) & (~other), zero,
                     jnp.where(has1 & (~has0) & (~other), one, none))


def _resolve(nav, cur, facing, stand, face, interact):
    """Walk to `stand`, turn to `face`, then interact (or wait)."""
    face_act = _i32(FACE_ACT_L)
    on_spot = jnp.where(facing == face,
                        jnp.where(interact, jnp.int32(A_INT), jnp.int32(A_STAY)),
                        face_act[face])
    return jnp.where(cur == stand, on_spot, nav[cur, stand]).astype(jnp.int32)


# ---------------------------------------------------------------------------
# seat 0 - cook, plate and deliver inside the sealed top room
# memory: [step, steps_without_evidence, solo_latch, last_delivered_type,
#          counter (2,4)/(2,6) holds my own stash,
#          counter (2,4)/(2,6) item predates my last delivery]
# ---------------------------------------------------------------------------


def init_agent_0():
    return jnp.asarray([0, 0, 0, -1, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    memory = jnp.where(reset, init_agent_0(), memory)
    step, no_ev, solo, last_del = memory[0], memory[1], memory[2], memory[3]
    mine_l, mine_r = memory[4], memory[5]
    stale_l, stale_r, hold_ctr = memory[6], memory[7], memory[8]

    cur = _localize(obs, _i32(TPL0_L))
    row = _i32(ROW0_L)[cur]
    col = _i32(COL0_L)[cur]
    facing = _facing(obs)

    inv = obs[VIEW, VIEW]
    inv_ing = inv[7:11]
    inv_total = jnp.sum(inv_ing)
    held_type = jnp.argmax(inv_ing).astype(jnp.int32)
    self_plate = inv[5] > 0
    self_cooked = inv[6] > 0
    holding_dish = self_plate & self_cooked
    holding_plate = self_plate & (~self_cooked)
    holding_ing = inv_total > 0

    pot, _ = _cell(obs, row, col, 2, 5)
    pot_ing = pot[34:38]
    pot_total = jnp.sum(pot_ing)
    pot_cooked = pot[33] > 0
    pot_cooking = pot[44] > 0
    pot_ready = pot_cooked & (~pot_cooking)
    pot_type = jnp.argmax(pot_ing).astype(jnp.int32)
    pot_homog = (pot_total > 0) & (pot_ing[pot_type] == pot_total) & (pot_type <= 1)
    pot_bad = (pot_total > 0) & (~pot_homog)
    pot_open = (~pot_cooking) & (~pot_cooked) & (pot_total < 3)

    cnt_l, _ = _cell(obs, row, col, 2, 4)
    cnt_r, _ = _cell(obs, row, col, 2, 6)
    l_ing, l_kind, l_plate, l_empty = _item_kind(cnt_l)
    r_ing, r_kind, r_plate, r_empty = _item_kind(cnt_r)

    partner_seen = jnp.sum(obs[:, :, 11]) > 0
    partner_ing = jnp.sum(obs[:, :, 18:22], axis=(0, 1))
    partner_raw = (jnp.sum(obs[:, :, 16]) == 0) & (jnp.sum(obs[:, :, 17]) == 0)

    # --- which ingredient does the current recipe most likely ask for? ------
    k_pot = jnp.where(pot_homog, pot_type, jnp.int32(-1))
    k_partner = jnp.where(partner_seen & partner_raw,
                          _single_type(partner_ing), jnp.int32(-1))
    # Only ingredients the *partner* left on a shared counter after my last
    # delivery carry information: my own parked leftovers say nothing, and an
    # item that predates the delivery was chosen for the previous recipe.
    fresh_l = l_ing & (mine_l == 0) & (stale_l == 0)
    fresh_r = r_ing & (mine_r == 0) & (stale_r == 0)
    counter_ing = (jnp.where(fresh_l, cnt_l[34:38], 0)
                   + jnp.where(fresh_r, cnt_r[34:38], 0))
    k_counter = _single_type(counter_ing)
    any_evidence = (k_pot >= 0) | (k_partner >= 0) | (k_counter >= 0)
    # Zero-information fallback.  A wrong delivery does not resample the
    # recipe, so always switching away from the last delivered ingredient is
    # correct with probability 2/3 in steady state, versus 1/2 for a fixed
    # guess and 0 for standing still.
    k_solo = jnp.where(solo > 0,
                       jnp.where(last_del == 0, jnp.int32(1), jnp.int32(0)),
                       jnp.int32(-1))
    k = jnp.where(k_pot >= 0, k_pot,
                  jnp.where(k_counter >= 0, k_counter,
                            jnp.where(k_partner >= 0, k_partner, k_solo)))

    # --- candidate jobs -----------------------------------------------------
    stash = jnp.where(l_empty, jnp.int32(J0_CNT_L),
                      jnp.where(r_empty, jnp.int32(J0_CNT_R), jnp.int32(J0_WAIT)))
    get_plate = jnp.where(l_plate, jnp.int32(J0_CNT_L),
                          jnp.where(r_plate, jnp.int32(J0_CNT_R),
                                    jnp.where(col >= 6, jnp.int32(J0_PLATE_R),
                                              jnp.int32(J0_PLATE_L))))
    pile_k = jnp.where(k == 1, jnp.int32(J0_ING1), jnp.int32(J0_ING0))
    take_k = jnp.where(l_ing & (l_kind == k), jnp.int32(J0_CNT_L),
                       jnp.where(r_ing & (r_kind == k), jnp.int32(J0_CNT_R), pile_k))
    # A spoiled pot can only be cleared by filling, cooking and binning it.
    fill_any = jnp.where(l_ing, jnp.int32(J0_CNT_L),
                         jnp.where(r_ing, jnp.int32(J0_CNT_R),
                                   jnp.where(col >= 6, jnp.int32(J0_ING1),
                                             jnp.int32(J0_ING0))))

    dish_ok = (holding_dish & (inv_total == 3)
               & (inv_ing[held_type] == 3) & (held_type <= 1))
    bin_dish = jnp.where(l_empty, jnp.int32(J0_CNT_L),
                         jnp.where(r_empty, jnp.int32(J0_CNT_R), jnp.int32(J0_DELIVER)))
    job_dish = jnp.where(dish_ok, jnp.int32(J0_DELIVER), bin_dish)

    job_plate = jnp.where(pot_ready, jnp.int32(J0_POT),
                          jnp.where(pot_open & ((k >= 0) | pot_bad), stash,
                                    jnp.int32(J0_WAIT)))

    # With both shared counters occupied there is nowhere to park a held item,
    # and a held ingredient cannot be exchanged for the plate the pot needs.
    # Adding it to the pot regardless spoils the batch but stays recoverable,
    # whereas holding on forever ends the episode.
    no_parking = (~l_empty) & (~r_empty)
    # An ingredient the partner is merely *carrying* may have been picked for
    # the previous recipe, so hold the first unit at the pot until something
    # committed confirms it: the partner's own insertion, a counter drop, or a
    # patience timeout when the partner never commits at all.
    confirmed = ((solo > 0) | (k_counter == held_type) | no_parking
                 | (hold_ctr >= COMMIT_WAIT))
    can_insert = pot_open & (pot_bad | (pot_homog & (pot_type == held_type))
                             | ((pot_total == 0) & confirmed))
    wait_confirm = pot_open & (pot_total == 0) & (k == held_type)
    job_hold_ing = jnp.where(can_insert, jnp.int32(J0_POT),
                             jnp.where(wait_confirm, jnp.int32(J0_WAIT), stash))

    # Only start carrying an ingredient when it can still be parked if the pot
    # closes in the two steps it takes to bring it over.
    from_counter = (l_ing & (l_kind == k)) | (r_ing & (r_kind == k))
    safe_fetch = (~no_parking) | (pot_total == 0) | from_counter
    job_free = jnp.where(pot_ready | pot_cooking, get_plate,
                         jnp.where(pot_bad, fill_any,
                                   jnp.where(pot_open & (k >= 0) & safe_fetch, take_k,
                                             jnp.int32(J0_WAIT))))

    job = jnp.where(holding_dish, job_dish,
                    jnp.where(holding_plate, job_plate,
                              jnp.where(holding_ing, job_hold_ing, job_free)))

    action = _resolve(_i32(NAV0_L), cur, facing,
                      _i32(S0_STAND_L)[job], _i32(S0_FACE_L)[job],
                      _i32(S0_INT_L)[job] > 0)

    delivered = (action == A_INT) & holding_dish & (cur == 1) & (facing == F_UP)
    new_last = jnp.where(delivered, held_type, last_del)
    new_no_ev = jnp.where(any_evidence, jnp.int32(0), no_ev + 1)
    new_solo = jnp.where((solo > 0) | (new_no_ev >= SOLO_AFTER),
                         jnp.int32(1), jnp.int32(0))
    hands_full = holding_dish | holding_plate | holding_ing
    used_l = (action == A_INT) & (cur == 0) & (facing == F_DOWN)
    used_r = (action == A_INT) & (cur == 2) & (facing == F_DOWN)
    new_mine_l = jnp.where(used_l, hands_full.astype(jnp.int32),
                           jnp.where(l_empty, jnp.int32(0), mine_l))
    new_mine_r = jnp.where(used_r, hands_full.astype(jnp.int32),
                           jnp.where(r_empty, jnp.int32(0), mine_r))
    new_stale_l = jnp.where(delivered, (~l_empty).astype(jnp.int32),
                            jnp.where(l_empty, jnp.int32(0), stale_l))
    new_stale_r = jnp.where(delivered, (~r_empty).astype(jnp.int32),
                            jnp.where(r_empty, jnp.int32(0), stale_r))
    new_hold = jnp.where(holding_ing & (pot_total == 0), hold_ctr + 1, jnp.int32(0))
    new_memory = jnp.stack([step + 1, new_no_ev, new_solo, new_last,
                            new_mine_l, new_mine_r,
                            new_stale_l, new_stale_r, new_hold]).astype(jnp.int32)

    trace = {
        'pos': cur,
        'job': job.astype(jnp.int32),
        'target': k.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'solo': new_solo,
    }
    return action.astype(jnp.int32), new_memory, trace


# ---------------------------------------------------------------------------
# seat 1 - read the recipe, carry the matching ingredient to the pot
# memory: [step, recipe, stage_left_type, stage_left_full, stage_right_type,
#          stage_right_full, bin_left_full, bin_right_full, pot_wants, wait]
# ---------------------------------------------------------------------------


def init_agent_1():
    return jnp.asarray([0, -1, -1, 0, -1, 0, 0, 0, 1, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    memory = jnp.where(reset, init_agent_1(), memory)
    step, mem_k = memory[0], memory[1]
    mem_sl, mem_lf, mem_sr, mem_rf = memory[2], memory[3], memory[4], memory[5]
    mem_bl, mem_br, mem_need, wait = memory[6], memory[7], memory[8], memory[9]

    dist = _i32(DIST1_L)
    stand = _i32(S1_STAND_L)

    cur = _localize(obs, _i32(TPL1_L))
    row = _i32(ROW1_L)[cur]
    col = _i32(COL1_L)[cur]
    facing = _facing(obs)

    inv = obs[VIEW, VIEW]
    inv_ing = inv[7:11]
    held_type = jnp.argmax(inv_ing).astype(jnp.int32)
    holding_ing = jnp.sum(inv_ing) > 0
    holding_other = (inv[5] > 0) | (inv[6] > 0)

    rec_cell, rec_seen = _cell(obs, row, col, 4, 5)
    rec_counts = rec_cell[40:44]
    k_seen = jnp.where(rec_seen & (jnp.sum(rec_counts) > 0),
                       jnp.argmax(rec_counts).astype(jnp.int32), jnp.int32(-1))
    k = jnp.where(k_seen >= 0, k_seen, mem_k)

    pot, pot_seen = _cell(obs, row, col, 2, 5)
    pot_ing = pot[34:38]
    pot_total = jnp.sum(pot_ing)
    pot_type = jnp.argmax(pot_ing).astype(jnp.int32)
    pot_homog = (pot_total > 0) & (pot_ing[pot_type] == pot_total)
    pot_accepts = ((pot[44] == 0) & (pot[33] == 0) & (pot_total < 3)
                   & ((pot_total == 0) | (pot_homog & (pot_type == k))))
    need = jnp.where(pot_seen, pot_accepts.astype(jnp.int32), mem_need)

    # Drop counters (4,3) (4,7) (5,2) (5,8) are all invisible from the top
    # room, so parking an ingredient there can never fake a recipe signal for
    # seat 0.  Occupancy is remembered while a counter is out of view, and the
    # type is tracked for the two staging counters next to the pot.
    stg_l, sl_seen = _cell(obs, row, col, 4, 3)
    stg_r, sr_seen = _cell(obs, row, col, 4, 7)
    bin_l, bl_seen = _cell(obs, row, col, 5, 2)
    bin_r, br_seen = _cell(obs, row, col, 5, 8)
    l_ing, l_kind, _, l_bare = _item_kind(stg_l)
    r_ing, r_kind, _, r_bare = _item_kind(stg_r)
    _, _, _, bl_bare = _item_kind(bin_l)
    _, _, _, br_bare = _item_kind(bin_r)
    sl = jnp.where(sl_seen, jnp.where(l_ing, l_kind, jnp.int32(-1)), mem_sl)
    sr = jnp.where(sr_seen, jnp.where(r_ing, r_kind, jnp.int32(-1)), mem_sr)
    lf = jnp.where(sl_seen, (~l_bare).astype(jnp.int32), mem_lf)
    rf = jnp.where(sr_seen, (~r_bare).astype(jnp.int32), mem_rf)
    bl = jnp.where(bl_seen, (~bl_bare).astype(jnp.int32), mem_bl)
    br = jnp.where(br_seen, (~br_bare).astype(jnp.int32), mem_br)

    # A partner carrying a cooked dish means the recipe is about to be
    # resampled; anything fetched or inserted now may be for the stale target.
    partner_seen = jnp.sum(obs[:, :, 11]) > 0
    partner_dish = partner_seen & (jnp.sum(obs[:, :, 17]) > 0)
    hold_off = partner_dish & (wait < PATIENCE)

    left_close = dist[cur, stand[J1_ING0_L]] <= dist[cur, stand[J1_ING0_R]]

    def _pick(free_a, job_a, free_b, job_b):
        nearer = jnp.where(dist[cur, stand[job_a]] <= dist[cur, stand[job_b]],
                           jnp.int32(job_a), jnp.int32(job_b))
        return jnp.where(free_a & free_b, nearer,
                         jnp.where(free_a, jnp.int32(job_a),
                                   jnp.where(free_b, jnp.int32(job_b), jnp.int32(-1))))

    # Never re-interact with a drop counter that is known to be occupied: that
    # is a no-op every step and would lock the seat out of the episode.
    stage_free = _pick(lf == 0, J1_STG_L, rf == 0, J1_STG_R)
    bin_free = _pick(bl == 0, J1_BIN_L, br == 0, J1_BIN_R)
    # With every drop counter full there is nothing useful to do with the held
    # item; idle at (4,1), which the top room cannot see, so that carrying a
    # now-meaningless ingredient does not look like a recipe signal.
    park = jnp.where(stage_free >= 0, stage_free,
                     jnp.where(bin_free >= 0, bin_free, jnp.int32(J1_HIDE)))

    insert = (k >= 0) & (held_type == k) & (need > 0) & (~hold_off)
    job_hold = jnp.where(holding_ing & insert, jnp.int32(J1_POT), park)

    depot_k = jnp.where(left_close,
                        jnp.where(k == 1, jnp.int32(J1_ING1_L), jnp.int32(J1_ING0_L)),
                        jnp.where(k == 1, jnp.int32(J1_ING1_R), jnp.int32(J1_ING0_R)))
    take_stage = jnp.where((k >= 0) & (sl == k), jnp.int32(J1_STG_L),
                           jnp.where((k >= 0) & (sr == k), jnp.int32(J1_STG_R),
                                     depot_k))
    # While the pot is busy, keep one unit of each legal ingredient staged
    # close to the pot so the next recipe can be served almost immediately.
    other = 1 - k
    room = (lf == 0) | (rf == 0)
    prefetch = jnp.where(((sl != k) & (sr != k)) & room, k,
                         jnp.where(((sl != other) & (sr != other)) & room, other,
                                   jnp.int32(-1)))
    depot_p = jnp.where(left_close,
                        jnp.where(prefetch == 1, jnp.int32(J1_ING1_L),
                                  jnp.int32(J1_ING0_L)),
                        jnp.where(prefetch == 1, jnp.int32(J1_ING1_R),
                                  jnp.int32(J1_ING0_R)))
    job_free = jnp.where(k < 0, jnp.int32(J1_WAIT),
                         jnp.where((need > 0) & (~hold_off), take_stage,
                                   jnp.where(prefetch >= 0, depot_p,
                                             jnp.int32(J1_WAIT))))

    job = jnp.where(holding_ing | holding_other, job_hold, job_free)
    action = _resolve(_i32(NAV1_L), cur, facing,
                      stand[job], _i32(S1_FACE_L)[job], _i32(S1_INT_L)[job] > 0)

    new_wait = jnp.where(partner_dish, wait + 1, jnp.int32(0))
    new_memory = jnp.stack([step + 1, k, sl, lf, sr, rf,
                            bl, br, need, new_wait]).astype(jnp.int32)

    trace = {
        'pos': cur,
        'job': job.astype(jnp.int32),
        'recipe': k.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'need': need.astype(jnp.int32),
    }
    return action.astype(jnp.int32), new_memory, trace
