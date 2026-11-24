"""Overcooked V2 test_time_simple policy.

Layout facts (see context/environment/README.md):
  * agent_0 lives in the left room (rows 1-3, cols 1-2). It always sees the
    always-on recipe indicator at (2,0) but can only reach ingredient-2
    supplies and the counters at column 3 / column 0.
  * agent_1 lives in the right room (rows 1-3, cols 5-6). It can reach the pot,
    plate piles, ingredient 0/1 supplies and the delivery station, but never
    sees the recipe indicator.  It sees the column-3 counters from column 5.
  * The agents never see each other.

Protocol (spatially aligned, convention-light):
  * agent_0 places one ingredient-2 unit on counter (1,3) for recipe [0,0,0]
    (same row as the ingredient-0 supply) or on (3,3) for recipe [1,1,1]
    (same row as the ingredient-1 supply), keeping the other counters clear.
  * agent_1 decodes: top-only -> 0, bottom-only -> 1, middle-only -> 1,
    several items -> 1, nothing -> unknown (default 0 after a long wait).
    It waits a few steps after each delivery before trusting the signal
    (recipe resamples on delivery), dumps mismatched dishes / ingredients on
    its own side counters instead of delivering wrong dishes, and reuses
    cached items when they match.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'counter_signal_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# actions
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# facing ids: UP=0 DOWN=1 RIGHT=2 LEFT=3 ; action -> facing
ACT2FACE_T = (2, 1, 3, 0)

# channels
CH_GOAL = 21
CH_REC_IND = 23
CH_ITEM = 29   # [plate, cooked, c0, c1, c2] = 29..33
CH_RECIPE = 34  # 34..38
CH_TIMER = 39

# agent_1 tuning
W_FRESH = 8        # steps after delivery before trusting an unchanged signal
W_LONG = 16        # steps with no signal before falling back to a guess
PERSIST = 5        # consecutive col-5 observations before a new item state counts as a change
NOTHING_LEARN = 10   # consecutive observations of 'nothing' before it counts as a persistent new state
NOTHING_PERSIST = 6  # observations of 'nothing' before a learned nothing-meaning is used
L_PRIOR = -3       # prior log2-odds that the default decode of a state is wrong (probe after 4 unchanged)
PROBE_FAIL_STEP = 1  # odds decrease after a probe delivery is followed by no change
L_BIG = 6          # clamp for learned certainty
# default decode of raw code (bit0 top, bit1 middle, bit2 bottom): -1 unknown
DEFAULT_DECODE_T = (-1, 0, 1, 1, 1, 1, 1, 1)

I32 = jnp.int32


def _i(x):
    return jnp.asarray(x, dtype=I32)


def cell(obs, pr, pc, gr, gc):
    """Return (40-vector at global cell, visible flag)."""
    li = gr - pr + 2
    lj = gc - pc + 2
    ok = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    v = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), :]
    return jnp.where(ok, v, 0), ok


def self_pos_from_channel(obs, ch, gr, gc):
    flat = jnp.argmax(obs[:, :, ch].reshape(-1))
    li = flat // 5
    lj = flat % 5
    return gr + 2 - li, gc + 2 - lj


def self_facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(I32)


def nav_action(pr, pc, gr, gc, dir_act):
    """One move toward (gr,gc). Rooms are full rectangles so any order is
    valid; when both axes are needed, move last along the axis of the desired
    final facing so that the arriving move already sets it."""
    dr = gr - pr
    dc = gc - pc
    vert = jnp.where(dr > 0, DOWN, UP)
    horiz = jnp.where(dc > 0, RIGHT, LEFT)
    both = (dr != 0) & (dc != 0)
    dir_is_vert = (dir_act == UP) | (dir_act == DOWN)
    single = jnp.where(dr != 0, vert, horiz)
    return jnp.where(both, jnp.where(dir_is_vert, horiz, vert), single).astype(I32)


def approach(pr, pc, face, gr, gc, dir_act, final=INTERACT):
    """Go to floor cell (gr,gc), face dir_act, then perform `final`."""
    at = (pr == gr) & (pc == gc)
    facing_ok = face == jnp.asarray(ACT2FACE_T, dtype=I32)[dir_act]
    return jnp.where(at, jnp.where(facing_ok, final, dir_act), nav_action(pr, pc, gr, gc, dir_act)).astype(I32)


def item_code(v5):
    """0 empty, 1 ingr0, 2 ingr1, 3 dish0, 4 dish1, 5 other."""
    plate, cooked, c0, c1, c2 = v5[0], v5[1], v5[2], v5[3], v5[4]
    empty = (plate == 0) & (cooked == 0) & (c0 == 0) & (c1 == 0) & (c2 == 0)
    raw0 = (plate == 0) & (cooked == 0) & (c0 == 1) & (c1 == 0) & (c2 == 0)
    raw1 = (plate == 0) & (cooked == 0) & (c1 == 1) & (c0 == 0) & (c2 == 0)
    dish0 = (cooked == 1) & (c0 == 3) & (c1 == 0) & (c2 == 0)
    dish1 = (cooked == 1) & (c1 == 3) & (c0 == 0) & (c2 == 0)
    code = jnp.where(empty, 0, jnp.where(raw0, 1, jnp.where(raw1, 2, jnp.where(dish0, 3, jnp.where(dish1, 4, 5)))))
    return code.astype(I32)


# --------------------------------------------------------------------------
# agent_0 : recipe signaller (left room)
# --------------------------------------------------------------------------

def init_agent_0():
    return {'t': _i(0)}


def agent_0_act(obs, reset, memory, rng):
    t = jnp.where(reset, 0, memory['t'] + 1).astype(I32)
    pr, pc = self_pos_from_channel(obs, CH_REC_IND, 2, 0)
    face = self_facing(obs)
    inv = obs[2, 2, 5:10]
    holding = jnp.any(inv > 0)

    rcell, _ = cell(obs, pr, pc, 2, 0)
    rec = rcell[CH_RECIPE:CH_RECIPE + 5]
    recipe = jnp.where(rec[3] > rec[2], 1, 0).astype(I32)
    tr = jnp.where(recipe == 0, 1, 3).astype(I32)

    def occupied(gr, gc):
        v, ok = cell(obs, pr, pc, gr, gc)
        return ok & jnp.any(v[CH_ITEM:CH_ITEM + 5] > 0)

    occ1 = occupied(1, 3)
    occ2 = occupied(2, 3)
    occ3 = occupied(3, 3)
    target_occ = jnp.where(recipe == 0, occ1, occ3)
    wrong1 = occ1 & (tr != 1)
    wrong2 = occ2
    wrong3 = occ3 & (tr != 3)
    any_wrong = wrong1 | wrong2 | wrong3

    # nearest wrong row
    d1 = jnp.where(wrong1, jnp.abs(pr - 1), 99)
    d2 = jnp.where(wrong2, jnp.abs(pr - 2), 99)
    d3 = jnp.where(wrong3, jnp.abs(pr - 3), 99)
    wrow = jnp.where((d1 <= d2) & (d1 <= d3), 1, jnp.where(d2 <= d3, 2, 3)).astype(I32)

    act_place = approach(pr, pc, face, tr, 2, RIGHT)
    act_pick = approach(pr, pc, face, wrow, 2, RIGHT)
    sup_row = jnp.where(tr == 1, 1, 3).astype(I32)
    sup_dir = jnp.where(tr == 1, UP, DOWN).astype(I32)
    act_fetch = approach(pr, pc, face, sup_row, 2, sup_dir)
    act_idle = approach(pr, pc, face, tr, 2, RIGHT, final=STAY)

    # dump a surplus held item on a side counter (fixed order)
    s10 = occupied(1, 0)
    s01 = occupied(0, 1)
    s30 = occupied(3, 0)
    s41 = occupied(4, 1)
    act_dump = jnp.where(~s10, approach(pr, pc, face, 1, 1, LEFT),
               jnp.where(~s01, approach(pr, pc, face, 1, 1, UP),
               jnp.where(~s30, approach(pr, pc, face, 3, 1, LEFT),
               jnp.where(~s41, approach(pr, pc, face, 3, 1, DOWN), STAY))))

    act_hold = jnp.where(~target_occ, act_place, act_dump)
    act_empty = jnp.where(any_wrong, act_pick, jnp.where(~target_occ, act_fetch, act_idle))
    action = jnp.where(holding, act_hold, act_empty).astype(I32)

    trace = {'recipe': recipe, 'pr': _i(pr), 'pc': _i(pc), 'holding': holding.astype(I32),
             'occ': jnp.stack([occ1, occ2, occ3]).astype(I32)}
    return action, {'t': t}, trace


# --------------------------------------------------------------------------
# agent_1 : cook (right room)
# --------------------------------------------------------------------------

# side counters usable by agent_1: (counter cell, approach cell, dir action)
SC_R_T = (0, 0, 4, 4)
SC_C_T = (5, 6, 5, 6)
AP_R_T = (1, 1, 3, 3)
AP_C_T = (5, 6, 5, 6)
AP_D_T = (UP, UP, DOWN, DOWN)


def init_agent_1():
    return {
        'raw': _i(0),           # last raw col-3 signal code seen from col 5 (bits top,mid,bottom)
        'stable': _i(0),        # consecutive col-5 observations with the same raw code
        'tsd': _i(0),           # steps since delivery / episode start
        'counters': jnp.zeros((4,), dtype=I32),  # remembered side counter items
        'raw_del': _i(-1),      # raw code at the time of the last delivery
        'changed': _i(0),       # 1 once a col-5 observation differed from raw_del
        'alt': _i(0),           # fallback guess when no signal; toggles each delivery
        'L': jnp.full((8,), L_PRIOR, dtype=I32),  # log-odds that default decode of state is wrong
        'nmap': _i(-1),         # learned recipe meaning of 'nothing' (-1 unknown)
        'd_del': _i(-1),        # dish type delivered last (-1 none)
        'learned': _i(1),       # 1 once the post-delivery change evidence was processed
        'nc_done': _i(1),       # 1 once the post-delivery no-change evidence was processed
    }


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    pr, pc = self_pos_from_channel(obs, CH_GOAL, 2, 7)
    face = self_facing(obs)
    inv = obs[2, 2, 5:10]
    hold = item_code(inv)                  # 0 none,1 i0,2 i1,3 d0,4 d1,5 other
    hold_plate = (inv[0] == 1) & (inv[1] == 0)
    hold_dish = inv[1] == 1
    hold_ingr = (inv[0] == 0) & (inv[1] == 0) & jnp.any(inv[2:5] > 0)
    dish_type = jnp.where(hold == 3, 0, jnp.where(hold == 4, 1, -1)).astype(I32)
    ingr_type = jnp.where(inv[2] > 0, 0, jnp.where(inv[3] > 0, 1, 2)).astype(I32)

    # pot
    pot, _ = cell(obs, pr, pc, 2, 4)
    pc0, pc1, pc2 = pot[CH_ITEM + 2], pot[CH_ITEM + 3], pot[CH_ITEM + 4]
    n_pot = pc0 + pc1 + pc2
    cooking = pot[CH_TIMER] > 0
    ready = pot[CH_ITEM + 1] > 0
    pot_busy = cooking | ready
    ptype = jnp.where(n_pot == 0, -1, jnp.where(pc0 >= pc1, 0, 1)).astype(I32)

    # ---- raw signal (col-3 counters visible only from col 5) ----
    at_col5 = pc == 5
    c13, _ = cell(obs, pr, pc, 1, 3)
    c23, _ = cell(obs, pr, pc, 2, 3)
    c33, _ = cell(obs, pr, pc, 3, 3)
    o1 = jnp.any(c13[CH_ITEM:CH_ITEM + 5] > 0).astype(I32)
    o2 = jnp.any(c23[CH_ITEM:CH_ITEM + 5] > 0).astype(I32)
    o3 = jnp.any(c33[CH_ITEM:CH_ITEM + 5] > 0).astype(I32)
    code = o1 + 2 * o2 + 4 * o3
    same = code == mem['raw']
    stable = jnp.where(at_col5, jnp.where(same, mem['stable'] + 1, 1), mem['stable']).astype(I32)
    raw = jnp.where(at_col5, code, mem['raw']).astype(I32)
    default = jnp.asarray(DEFAULT_DECODE_T, dtype=I32)
    idx8 = jnp.arange(8)

    # ---- convention learning from a persistent post-delivery change ----
    L = mem['L']
    nmap = mem['nmap']
    d_del = mem['d_del']
    raw_del = mem['raw_del']
    persist_ok = jnp.where(raw == 0, stable >= NOTHING_LEARN, stable >= PERSIST)
    evidence = at_col5 & persist_ok & (raw != raw_del) & (mem['learned'] == 0) & (d_del >= 0)

    # recipe(raw_del) = d_del ; recipe(raw) = 1 - d_del
    def confirm(Lv, state, rec):
        wrong_default = default[state] != rec
        val = jnp.where(wrong_default, L_BIG, -L_BIG)
        return jnp.where((idx8 == state) & (state != 0), val, Lv)
    L_new = confirm(confirm(L, raw_del, d_del), raw, 1 - d_del)
    nmap_new = jnp.where(raw_del == 0, d_del, jnp.where(raw == 0, 1 - d_del, nmap))
    L = jnp.where(evidence, L_new, L).astype(I32)
    nmap = jnp.where(evidence, nmap_new, nmap).astype(I32)
    learned = jnp.where(evidence, 1, mem['learned']).astype(I32)

    # ---- no-change evidence at the freshness deadline (before the next dish is committed) ----
    tsd = mem['tsd']
    changed = jnp.where(at_col5 & (code != raw_del), 1, mem['changed']).astype(I32)
    nc_now = at_col5 & (tsd >= W_FRESH) & (changed == 0) & (mem['nc_done'] == 0) & (d_del >= 0) \
        & (raw == raw_del) & (raw_del != 0)
    step = jnp.where(default[raw_del] == d_del, 1, -PROBE_FAIL_STEP)
    L_nc = jnp.clip(L + jnp.where(idx8 == raw_del, step, 0), -L_BIG, L_BIG)
    L = jnp.where(nc_now, L_nc, L).astype(I32)
    nc_done = jnp.where(nc_now, 1, mem['nc_done']).astype(I32)

    # ---- decode ----
    flip = L[raw] > 0
    dec_item = jnp.where(flip, 1 - default[raw], default[raw])
    nothing_ok = (nmap >= 0) & (stable >= NOTHING_PERSIST)
    sig = jnp.where(raw == 0, jnp.where(nothing_ok, nmap, -1), dec_item).astype(I32)
    sig_ok = sig >= 0

    fresh = (tsd >= W_FRESH) | (changed == 1)
    want = jnp.where(sig_ok, sig, jnp.where(tsd >= W_LONG, mem['alt'], -1)).astype(I32)
    target = jnp.where(n_pot > 0, ptype, want).astype(I32)

    SC_R = jnp.asarray(SC_R_T, dtype=I32)
    SC_C = jnp.asarray(SC_C_T, dtype=I32)
    AP_R = jnp.asarray(AP_R_T, dtype=I32)
    AP_C = jnp.asarray(AP_C_T, dtype=I32)
    AP_D = jnp.asarray(AP_D_T, dtype=I32)

    # side counters memory update
    def read_counter(k):
        v, ok = cell(obs, pr, pc, SC_R[k], SC_C[k])
        return jnp.where(ok, item_code(v[CH_ITEM:CH_ITEM + 5]), mem['counters'][k])
    counters = jnp.stack([read_counter(k) for k in range(4)]).astype(I32)
    dists = jnp.abs(AP_R - pr) + jnp.abs(AP_C - pc)

    def nearest(mask):
        d = jnp.where(mask, dists, 99)
        k = jnp.argmin(d)
        return k, d[k] < 99

    def counter_action(k):
        return approach(pr, pc, face, AP_R[k], AP_C[k], AP_D[k])

    k_free, has_free = nearest(counters == 0)
    act_dump = jnp.where(has_free, counter_action(k_free), STAY)

    act_pot = approach(pr, pc, face, 2, 5, LEFT)
    act_wait = approach(pr, pc, face, 2, 5, LEFT, final=STAY)
    act_deliver = approach(pr, pc, face, 2, 6, RIGHT)
    # plate: nearest pile
    plate_row = jnp.where(pr <= 2, 1, 3)
    act_plate = approach(pr, pc, face, plate_row, 5, LEFT)

    # fetch ingredient of type `target` from supply or matching cached counter
    sup_row = jnp.where(target == 0, 1, 3)
    d_sup = jnp.abs(sup_row - pr) + jnp.abs(6 - pc)
    k_cache, has_cache = nearest(counters == (target + 1))
    use_cache = has_cache & (dists[k_cache] < d_sup)
    act_fetch = jnp.where(use_cache, counter_action(k_cache), approach(pr, pc, face, sup_row, 6, RIGHT))

    # cached dish matching signal
    k_dish, has_dish = nearest(counters == (want + 3))
    act_take_dish = counter_action(k_dish)

    # ---- decision ----
    # holding dish
    mismatch = sig_ok & (dish_type != sig)
    a_dish = jnp.where(mismatch & has_free, act_dump, act_deliver)
    # holding plate
    a_plate = jnp.where(ready, act_pot, jnp.where(cooking, act_wait, act_dump))
    # holding ingredient
    can_insert = (~pot_busy) & (n_pot < 3)
    ingr_match = ingr_type == target
    commit = (n_pot > 0) | (fresh & (target >= 0))
    a_ingr = jnp.where(can_insert & ingr_match & commit, act_pot,
             jnp.where(can_insert & ((target < 0) | (ingr_match & ~fresh)), act_wait,
             act_dump))
    # empty hands
    a_empty = jnp.where(pot_busy, act_plate,
              jnp.where(n_pot > 0, act_fetch,
              jnp.where(fresh & sig_ok & has_dish, act_take_dish,
              jnp.where(want >= 0, act_fetch, act_wait))))

    action = jnp.where(hold_dish, a_dish,
             jnp.where(hold_plate, a_plate,
             jnp.where(hold_ingr, a_ingr, a_empty))).astype(I32)

    # ---- delivery bookkeeping ----
    delivering = hold_dish & (pr == 2) & (pc == 6) & (face == 2) & (action == INTERACT)

    tsd_next = jnp.where(delivering, 0, jnp.minimum(tsd + 1, 10000)).astype(I32)
    raw_del_next = jnp.where(delivering, raw, raw_del).astype(I32)
    changed_next = jnp.where(delivering, 0, changed).astype(I32)
    alt_next = jnp.where(delivering, 1 - mem['alt'], mem['alt']).astype(I32)
    d_del_next = jnp.where(delivering, dish_type, d_del).astype(I32)
    learned_next = jnp.where(delivering, 0, learned).astype(I32)
    nc_done_next = jnp.where(delivering, 0, nc_done).astype(I32)

    new_mem = {'raw': raw, 'stable': stable, 'tsd': tsd_next, 'counters': counters,
               'raw_del': raw_del_next, 'changed': changed_next, 'alt': alt_next,
               'L': L, 'nmap': nmap, 'd_del': d_del_next, 'learned': learned_next,
               'nc_done': nc_done_next}
    trace = {'raw': raw, 'sig': sig, 'want': want, 'target': target, 'tsd': tsd, 'hold': hold,
             'n_pot': _i(n_pot), 'pr': _i(pr), 'pc': _i(pc), 'counters': counters,
             'changed': changed, 'L': L, 'nmap': nmap, 'stable': stable}
    return action, new_mem, trace

