"""Overcooked V2 demo_cook_wide joint policy.

Roles (derived from geometry, not private conventions):
  agent_0 (top room, 3 cells): plates cooked pots and delivers; while the pot
    is being filled it fetches ingredients that match what is already visibly
    in the pot (it cannot see the recipe indicator).
  agent_1 (bottom corridor): reads the always-on recipe indicator, fetches the
    recipe ingredient and loads the pot from below; during cooking it caches
    extra ingredients on the counters next to the pot.
All decisions use only the local observation plus small private memory.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_role_split', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

# ----------------------------------------------------------------------------
# Static layout knowledge
# ----------------------------------------------------------------------------
LAYOUT = [
    "WWWWBXBWWWW",
    "WWW0 A 1WWW",
    "WWWWWPWWWWW",
    "W    A    W",
    "0  W3R3W  0",
    "W1WWWWWWW1W",
]
GH, GW = 6, 11
V = 5
RAD = 2

# actions
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# facing ids
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
MODE_INT, MODE_WAIT = 0, 1
BIG = 999
# agent_1 cache-stocking thresholds (pot timer): start a supply trip / stash spare then trip
TRIP_T = 6
STASH_T = 9

A0_CELLS = [(1, 4), (1, 5), (1, 6)]
A1_CELLS = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
            (4, 1), (4, 2), (4, 8), (4, 9)]

# agent_1 counter cache slots: cell, standing cell, facing, preferred type (-1 any)
SLOT_CELL_L = [(2, 4), (2, 3), (2, 2), (2, 6), (2, 7), (2, 8), (4, 3), (4, 7)]
SLOT_STAND_L = [(3, 4), (3, 3), (3, 2), (3, 6), (3, 7), (3, 8), (3, 3), (3, 7)]
SLOT_DIR_L = [F_UP] * 6 + [F_DOWN] * 2
SLOT_PREF_L = [0, 0, 0, 1, 1, 1, -1, -1]
NSLOT = 8

# agent_1 supply access: [type, side] -> (stand_r, stand_c, facing); valid mask
SUP_STAND_L = [
    [(4, 1, F_LEFT), (4, 9, F_RIGHT)],   # type 0
    [(4, 1, F_DOWN), (4, 9, F_DOWN)],    # type 1
    [(3, 5, F_UP), (3, 5, F_UP)],        # type 2 (none)
    [(3, 4, F_DOWN), (3, 6, F_DOWN)],    # type 3
]
SUP_VALID_L = [[1, 1], [1, 1], [0, 0], [1, 1]]


def _static_map():
    m = [[[0] * 10 for _ in range(GW)] for _ in range(GH)]
    for r, row in enumerate(LAYOUT):
        for c, ch in enumerate(row):
            if ch == 'W':
                m[r][c][0] = 1
            elif ch == 'X':
                m[r][c][1] = 1
            elif ch == 'P':
                m[r][c][2] = 1
            elif ch == 'R':
                m[r][c][3] = 1
            elif ch == 'L':
                m[r][c][4] = 1
            elif ch == 'B':
                m[r][c][5] = 1
            elif ch.isdigit():
                m[r][c][6 + int(ch)] = 1
    return m


def _templates(cells):
    m = _static_map()
    t = []
    for (r, c) in cells:
        crop = []
        for i in range(V):
            rowv = []
            for j in range(V):
                gr, gc = r + i - RAD, c + j - RAD
                if 0 <= gr < GH and 0 <= gc < GW:
                    rowv.append(m[gr][gc])
                else:
                    rowv.append([0] * 10)
            crop.append(rowv)
        t.append(crop)
    return jnp.asarray(t, dtype=jnp.int32), jnp.asarray(cells, dtype=jnp.int32)


def _consts():
    """Build constant arrays at trace time (module-scope calls are disallowed)."""
    k = {}
    k["FACE2ACT"] = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)
    k["A0_T"], k["A0_C"] = _templates(A0_CELLS)
    k["A1_T"], k["A1_C"] = _templates(A1_CELLS)
    k["SLOT_CELL"] = jnp.asarray(SLOT_CELL_L, dtype=jnp.int32)
    k["SLOT_STAND"] = jnp.asarray(SLOT_STAND_L, dtype=jnp.int32)
    k["SLOT_DIR"] = jnp.asarray(SLOT_DIR_L, dtype=jnp.int32)
    k["SLOT_PREF"] = jnp.asarray(SLOT_PREF_L, dtype=jnp.int32)
    k["SUP_STAND"] = jnp.asarray(SUP_STAND_L, dtype=jnp.int32)
    k["SUP_VALID"] = jnp.asarray(SUP_VALID_L, dtype=jnp.int32)
    return k


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def localize(obs, templates, cells):
    static = obs[:, :, 22:32]
    d = jnp.abs(static[None] - templates).sum(axis=(1, 2, 3))
    idx = jnp.argmin(d)
    return cells[idx, 0], cells[idx, 1]


def item_kind(block):
    """-1 empty, 0..3 raw ingredient type, 4 empty plate, 5 cooked dish."""
    plate = block[..., 0]
    cooked = block[..., 1]
    cnt = block[..., 2:6]
    tot = cnt.sum(-1)
    typ = jnp.argmax(cnt, axis=-1).astype(jnp.int32)
    return jnp.where(cooked > 0, 5, jnp.where(plate > 0, 4, jnp.where(tot > 0, typ, -1))).astype(jnp.int32)


def read_cell(obs, r, c, gr, gc):
    lr = gr - r + 2
    lc = gc - c + 2
    vis = (lr >= 0) & (lr < V) & (lc >= 0) & (lc < V)
    cell = obs[jnp.clip(lr, 0, V - 1), jnp.clip(lc, 0, V - 1)]
    return vis, cell


def partner_info(obs):
    ppos = obs[:, :, 11]
    vis = ppos.sum() > 0
    idx = jnp.argmax(ppos.reshape(-1))
    pr = idx // V
    pc = idx % V
    block = obs[pr, pc, 16:22]
    kind = jnp.where(vis, item_kind(block), -1)
    return vis, pr, pc, kind


def sel(cond, a, b):
    return tuple(jnp.where(cond, x, y) for x, y in zip(a, b))


def plan(tr, tc, tdir, mode):
    return (i32(tr), i32(tc), i32(tdir), i32(mode))


def execute(K, r, c, facing, p, nav):
    tr, tc, tdir, mode = p
    at = (r == tr) & (c == tc)
    move = nav(r, c, tr, tc)
    act_at = jnp.where(facing != tdir, K["FACE2ACT"][tdir],
                       jnp.where(mode == MODE_INT, INTERACT, STAY))
    return jnp.where(at, act_at, move).astype(jnp.int32)


def nav0(r, c, tr, tc):
    return jnp.where(c < tc, RIGHT, jnp.where(c > tc, LEFT, STAY)).astype(jnp.int32)


def nav1(r, c, tr, tc):
    at = (r == tr) & (c == tc)
    horiz = jnp.where(c < tc, RIGHT, LEFT)
    adj4 = (r == 4) & (tr == 4) & (jnp.abs(tc - c) == 1)
    act = jnp.where(at, STAY,
          jnp.where(adj4, horiz,
          jnp.where(r == 4, UP,
          jnp.where(c != tc, horiz,
          jnp.where(tr == 4, DOWN, STAY)))))
    return act.astype(jnp.int32)


def dist1(r, c, tr, tc):
    return jnp.abs(c - tc) + (r != tr).astype(jnp.int32)


def pot_state(block, timer):
    cnt = block[2:6]
    pcount = cnt.sum()
    ptype = jnp.argmax(cnt).astype(jnp.int32)
    pcooked = block[1] > 0
    idle = (timer == 0) & (~pcooked)
    return pcount, ptype, pcooked, idle, cnt


# ----------------------------------------------------------------------------
# agent_0: top room.
# memory = [empty_steps, hold_steps, prev_held, suspect, suspect_steps, wrong, dish_wrong, prev_pcount]
# ----------------------------------------------------------------------------
def init_agent_0():
    return jnp.asarray([0, 0, -1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, init_agent_0(), memory)
    empty_steps, hold_steps = mem[0], mem[1]
    prev_held, suspect, s_steps, wrong, dish_wrong, prev_pcount = mem[2], mem[3], mem[4], mem[5], mem[6], mem[7]
    self_loaded = mem[8]

    K = _consts()
    r, c = localize(obs, K["A0_T"], K["A0_C"])
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    held = item_kind(obs[2, 2, 5:11])
    held_ing = (held >= 0) & (held <= 3)

    _, potcell = read_cell(obs, r, c, 2, 5)
    pcount, ptype, pcooked, idle, pcnt = pot_state(potcell[32:38], potcell[44])
    busy = ~idle
    needs = idle & (pcount > 0) & (pcount < 3)
    pot_empty = idle & (pcount == 0)
    mixed = (pcnt > 0).sum() >= 2

    _, c4 = read_cell(obs, r, c, 2, 4)
    _, c6 = read_cell(obs, r, c, 2, 6)
    k4 = item_kind(c4[32:38])
    k6 = item_kind(c6[32:38])
    free4 = k4 == -1
    free6 = k6 == -1

    pvis, _, _, pkind = partner_info(obs)
    partner_ing = pvis & (pkind >= 0) & (pkind <= 3)

    # --- suspicion tracking: pot contents present right after our delivery were loaded
    # under the previous recipe (partner loaded between plating and delivery).
    delivered_now = (prev_held == 5) & (held == -1)
    plated_now = (prev_held == 4) & (held == 5)
    partner_other = partner_ing & (pkind != ptype) & (pcount > 0)
    same_added = (suspect == 1) & (pcount > prev_pcount) & (~mixed) & (~delivered_now) & (self_loaded == 0)
    suspect_n = jnp.where(delivered_now & (pcount > 0), 1, suspect)
    suspect_n = jnp.where(pot_empty | same_added, 0, suspect_n)
    s_steps_n = jnp.where(suspect_n == 1, s_steps + 1, 0)
    suspect_n = jnp.where(s_steps_n > 25, 0, suspect_n)
    wrong_n = jnp.where(pot_empty, 0, jnp.where(mixed | ((suspect_n == 1) & partner_other), 1, wrong))
    is_wrong = wrong_n == 1
    is_suspect = (suspect_n == 1) & (~is_wrong)
    dish_wrong_n = jnp.where(plated_now, wrong, jnp.where(held == 5, dish_wrong, 0))

    wait_pot = plan(1, 5, F_DOWN, MODE_WAIT)
    deliver = plan(1, 5, F_UP, MODE_INT)
    load = plan(1, 5, F_DOWN, MODE_INT)
    plate_plan = sel(c <= 5, plan(1, 4, F_UP, MODE_INT), plan(1, 6, F_UP, MODE_INT))

    def acquire(X):
        # options: counter (2,4), counter (2,6), supply
        both_full = (~free4) & (~free6)
        cpen = jnp.where(both_full, 0, 2)
        ca = 2 * jnp.abs(c - 4) + cpen
        cb = 2 * jnp.abs(c - 6) + cpen
        cs = jnp.where(X == 0, 2 * jnp.abs(c - 4) + 1, 2 * jnp.abs(c - 6) + 1)
        va = k4 == X
        vb = k6 == X
        vs = (X == 0) | (X == 1)
        costs = jnp.stack([jnp.where(va, ca, BIG), jnp.where(vb, cb, BIG), jnp.where(vs, cs, BIG)])
        best = jnp.argmin(costs)
        valid = costs[best] < BIG
        pa = plan(1, 4, F_DOWN, MODE_INT)
        pb = plan(1, 6, F_DOWN, MODE_INT)
        ps = sel(X == 0, plan(1, 4, F_LEFT, MODE_INT), plan(1, 6, F_RIGHT, MODE_INT))
        p = sel(best == 0, pa, sel(best == 1, pb, ps))
        return p, valid

    def stash(X):
        ca = 2 * jnp.abs(c - 4) + jnp.where(X == 0, 0, 1)
        cb = 2 * jnp.abs(c - 6) + jnp.where(X == 1, 0, 1)
        costs = jnp.stack([jnp.where(free4, ca, BIG), jnp.where(free6, cb, BIG)])
        best = jnp.argmin(costs)
        valid = costs[best] < BIG
        p = sel(best == 0, plan(1, 4, F_DOWN, MODE_INT), plan(1, 6, F_DOWN, MODE_INT))
        return p, valid

    # --- held ingredient X
    X = held
    stash_p, stash_ok = stash(X)
    stash_or_wait = sel(stash_ok, stash_p, wait_pot)
    spec_ok = (empty_steps >= 40)
    keep_holding = pot_empty & ((partner_ing & (pkind == X)) | (hold_steps < 10))
    ing_plan = sel(needs & ((ptype == X) | is_wrong), load,
               sel(pot_empty & spec_ok, load,
               sel(keep_holding, wait_pot, stash_or_wait)))

    # --- empty hands
    acq_p, acq_ok = acquire(ptype)
    guard = (~free4) & (~free6) & (k4 != ptype) & (k6 != ptype) & (pcount >= 2)
    acq_plan = sel(acq_ok & (~guard), acq_p, wait_pot)
    # speculative prefetch of the partner's carried type (hold, do not load)
    pre_p, pre_ok = acquire(pkind)
    pre_plan = sel(pre_ok & (free4 | free6), pre_p, wait_pot)
    # stalled-pot fallback: load from counters if exactly one ingredient type is cached
    k4i = (k4 >= 0) & (k4 <= 3)
    k6i = (k6 >= 0) & (k6 <= 3)
    one_type = (k4i & ((~k6i) | (k6 == k4))) | (k6i & (~k4i))
    spec_type = jnp.where(k4i, k4, k6)
    spec_p = sel(k4 == spec_type, plan(1, 4, F_DOWN, MODE_INT), plan(1, 6, F_DOWN, MODE_INT))
    spec_plan = sel(spec_ok & one_type, spec_p, wait_pot)
    # recycle a stashed wrong dish: pot already spoiled and partner's carried type reveals recipe
    k4t = jnp.argmax(c4[34:38]).astype(jnp.int32)
    k6t = jnp.argmax(c6[34:38]).astype(jnp.int32)
    r4 = (k4 == 5) & (k4t == pkind)
    r6 = (k6 == 5) & (k6t == pkind)
    recycle_ok = (is_wrong | is_suspect) & partner_other & (r4 | r6)
    recycle_plan = sel(r4, plan(1, 4, F_DOWN, MODE_INT), plan(1, 6, F_DOWN, MODE_INT))
    empty_plan = sel(recycle_ok, recycle_plan,
                 sel(busy, plate_plan,
                 sel(needs, acq_plan,
                 sel(pot_empty & partner_ing, pre_plan, spec_plan))))

    # --- plate
    plate_held = sel(pcooked, load, wait_pot)

    dish_stash_p, dish_stash_ok = stash(held)
    dish_plan = sel((dish_wrong_n == 1) & dish_stash_ok, dish_stash_p, deliver)
    p = sel(held == 5, dish_plan,
        sel(held == 4, plate_held,
        sel(held_ing, ing_plan, empty_plan)))

    action = execute(K, r, c, facing, p, nav0)

    new_empty = jnp.where(pot_empty, empty_steps + 1, 0)
    new_hold = jnp.where(held_ing & pot_empty, hold_steps + 1, 0)
    loading_now = (action == INTERACT) & held_ing & (p[0] == 1) & (p[1] == 5) & (p[2] == F_DOWN) & (c == 5)
    new_mem = jnp.stack([new_empty, new_hold, held, suspect_n, s_steps_n, wrong_n, dish_wrong_n, pcount,
                         jnp.where(loading_now, 1, 0)]).astype(jnp.int32)
    trace = {"r": i32(r), "c": i32(c), "held": i32(held), "tr": p[0], "tc": p[1], "tdir": p[2],
             "mode": p[3], "pcount": i32(pcount), "ptype": i32(ptype), "k4": i32(k4), "k6": i32(k6),
             "susp": i32(suspect_n), "wrong": i32(wrong_n), "dwrong": i32(dish_wrong_n)}
    return action, new_mem, trace


# ----------------------------------------------------------------------------
# agent_1: bottom corridor.
# memory: dict(recipe, pot(6), timer, slots(8), wait, mm)
# ----------------------------------------------------------------------------
def init_agent_1():
    return {
        "recipe": i32(-1),
        "pot": jnp.zeros((6,), dtype=jnp.int32),
        "timer": i32(0),
        "slots": -jnp.ones((NSLOT,), dtype=jnp.int32),
        "wait": i32(0),
        "mm": i32(0),
        "task": i32(-1),
        "lastwrong": i32(0),
        "pstuck": i32(0),
        "pcol": i32(5),
    }


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    K = _consts()
    SLOT_CELL, SLOT_STAND, SLOT_DIR, SLOT_PREF = K["SLOT_CELL"], K["SLOT_STAND"], K["SLOT_DIR"], K["SLOT_PREF"]
    SUP_STAND, SUP_VALID = K["SUP_STAND"], K["SUP_VALID"]
    r, c = localize(obs, K["A1_T"], K["A1_C"])
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    held = item_kind(obs[2, 2, 5:11])
    held_ing = (held >= 0) & (held <= 3)

    # recipe
    ind_vis, indcell = read_cell(obs, r, c, 4, 5)
    rblock = indcell[38:44]
    rvalid = ind_vis & (rblock[2:6].sum() > 0)
    R = jnp.where(rvalid, jnp.argmax(rblock[2:6]).astype(jnp.int32), mem["recipe"])
    R_known = R >= 0

    # pot (memory when out of view)
    pot_vis, potcell = read_cell(obs, r, c, 2, 5)
    mtimer = mem["timer"]
    mpot = mem["pot"]
    mpot_dec = mpot.at[1].set(jnp.where(mtimer == 1, 1, mpot[1]))
    mtimer_dec = jnp.maximum(mtimer - 1, 0)
    pblock = jnp.where(pot_vis, potcell[32:38], mpot_dec)
    ptimer = jnp.where(pot_vis, potcell[44], mtimer_dec)
    pcount, ptype, pcooked, idle, pcnt = pot_state(pblock, ptimer)
    busy = ~idle
    mismatch = R_known & (pcount > 0) & ((pcnt.sum() - jnp.where(R >= 0, pcnt[jnp.clip(R, 0, 3)], 0)) > 0)

    # counter slots
    lr = SLOT_CELL[:, 0] - r + 2
    lc = SLOT_CELL[:, 1] - c + 2
    svis = (lr >= 0) & (lr < V) & (lc >= 0) & (lc < V)
    sblocks = obs[jnp.clip(lr, 0, V - 1), jnp.clip(lc, 0, V - 1), 32:38]
    skind = jnp.where(svis, item_kind(sblocks), mem["slots"])

    # partner
    pvis, ppr, ppc, pkind = partner_info(obs)
    dp = pvis & (pkind == 5)
    wait_del = dp & (pcount == 0) & (mem["wait"] < 8)

    loadable = idle & (pcount < 3) & R_known & ((~mismatch) | (mem["mm"] < 12))
    # was the last non-empty pot content wrong for the recipe? (its delivery cannot resample)
    lastwrong = jnp.where(pcount > 0, jnp.where(mismatch, 1, 0), mem["lastwrong"])
    wait_del = wait_del & (lastwrong == 0)
    # a wrong dish is cooking/cooked: prepare the correct ingredient at the pot for immediate load
    prep = mismatch & busy & R_known
    N = jnp.where(pcount > 0, ptype, R)

    pot_int = plan(3, 5, F_UP, MODE_INT)
    pot_wait = plan(3, 5, F_UP, MODE_WAIT)

    sdist = dist1(r, c, SLOT_STAND[:, 0], SLOT_STAND[:, 1])
    side = jnp.where(c <= 5, 0, 1)

    shared = (jnp.arange(NSLOT) == 0) | (jnp.arange(NSLOT) == 3)
    partner_empty = pvis & (pkind == -1)
    fetch_pen = jnp.where(shared & (pcount > 0) & partner_empty, 2, 0)

    def fetch(T):
        Tc = jnp.clip(T, 0, 3)
        sc = jnp.where(skind == T, sdist + fetch_pen, BIG)
        bs = jnp.argmin(sc)
        slot_cost = sc[bs]
        st = SUP_STAND[Tc, side]
        sup_cost = jnp.where(SUP_VALID[Tc, side] > 0, dist1(r, c, st[0], st[1]) + 1, BIG)
        slot_p = plan(SLOT_STAND[bs, 0], SLOT_STAND[bs, 1], SLOT_DIR[bs], MODE_INT)
        sup_p = plan(st[0], st[1], st[2], MODE_INT)
        p = sel(slot_cost <= sup_cost, slot_p, sup_p)
        ok = jnp.minimum(slot_cost, sup_cost) < BIG
        return p, ok

    def stash(T, allow_generic, shared_pen=5, pref_pen=2):
        pref_ok = (SLOT_PREF >= 0) | (allow_generic & (SLOT_PREF == -1))
        cost = jnp.where((skind == -1) & pref_ok,
                         sdist + jnp.where(SLOT_PREF == T, 0, pref_pen) + jnp.where(shared, shared_pen, 0), BIG)
        bs = jnp.argmin(cost)
        p = plan(SLOT_STAND[bs, 0], SLOT_STAND[bs, 1], SLOT_DIR[bs], MODE_INT)
        return p, cost[bs] < BIG

    # ---- committed task memory: -1 none, 0/1 prefetch trip for that type, 2 stash spare
    task = mem["task"]
    in_trip = (task == 0) | (task == 1)

    # ---- held ingredient X
    X = held
    stash_g, stash_g_ok = stash(X, True, 2, 3)
    stash_p, stash_p_ok = stash(X, False, 5, 2)
    start_stash = busy & (ptimer >= STASH_T) & (task != 2) & (task != 3)
    do_stash = (~loadable) & stash_p_ok & ((task == 2) | start_stash)
    ing_plan = sel(loadable & (N == X), sel(wait_del, pot_wait, pot_int),
               sel(loadable, sel(stash_g_ok, stash_g, pot_wait),
               sel(do_stash, stash_p, pot_wait)))

    # ---- held plate/dish: park it anywhere free
    park_p, park_ok = stash(-1, True, 5, 0)
    park_plan = sel(park_ok, park_p, pot_wait)

    # ---- empty hands
    fetch_p, fetch_ok = fetch(N)
    fetch_plan = sel(fetch_ok, fetch_p, pot_wait)
    n0 = (skind == 0).sum()
    n1 = (skind == 1).sum()
    free0 = ((skind == -1) & (SLOT_PREF == 0)).any()
    free1 = ((skind == -1) & (SLOT_PREF == 1)).any()
    pre_t = jnp.where(n1 < n0, 1, 0)
    pre_t = jnp.where(jnp.where(pre_t == 0, free0, free1), pre_t, 1 - pre_t)
    pre_t = jnp.where(in_trip, task, pre_t)
    has_free = jnp.where(pre_t == 0, free0, free1)
    start_trip = busy & has_free & ((task == 2) | (ptimer >= TRIP_T))
    pre_ok = in_trip | start_trip
    st = SUP_STAND[pre_t, side]
    pre_plan = sel(pre_ok, plan(st[0], st[1], st[2], MODE_INT), pot_wait)
    empty_plan = sel(~R_known, pot_wait, sel(loadable, fetch_plan, pre_plan))

    # ---- unblock partner: it holds an ingredient, pot busy, both shared counters hold ingredients
    k4s, k6s = skind[0], skind[3]
    k4i = (k4s >= 0) & (k4s <= 3)
    k6i = (k6s >= 0) & (k6s <= 3)
    partner_ing = pvis & (pkind >= 0) & (pkind <= 3)
    priv_free_for = lambda T: ((skind == -1) & (~shared) & ((SLOT_PREF == T) | (SLOT_PREF == -1))).any()
    pstuck = jnp.where(~busy, 0, jnp.where(pvis, jnp.where(partner_ing, mem["pstuck"] + 1, 0), mem["pstuck"]))
    pcol = jnp.where(pvis, c + ppc - 2, mem["pcol"])
    unblock = busy & R_known & (k4i | k6i) & ((partner_ing & k4i & k6i) | (pstuck >= 6))
    pick4 = k4i & ((~k6i) | (pcol <= 5))
    unblock_take = sel(pick4, plan(3, 4, F_UP, MODE_INT), plan(3, 6, F_UP, MODE_INT))
    unblock_plan = sel(held_ing, sel(stash_p_ok, stash_p, pot_wait), unblock_take)

    # ---- prepare correct ingredient while a wrong dish cooks
    fetchR_p, fetchR_ok = fetch(R)
    ing_plan = sel(prep, sel(X == R, pot_wait, sel(stash_p_ok, stash_p, sel(stash_g_ok, stash_g, pot_wait))), ing_plan)
    empty_plan = sel(prep, sel(fetchR_ok, fetchR_p, pot_wait), empty_plan)
    ing_plan = sel(unblock, unblock_plan, ing_plan)
    empty_plan = sel(unblock, unblock_plan, empty_plan)

    # ---- while the partner carries a cooked dish the recipe may be about to change: wait at pot
    stale_wait = dp & (pcount == 0) & (mem["wait"] < 8) & R_known & (lastwrong == 0)
    p = sel(held_ing, ing_plan, sel(held >= 4, park_plan, empty_plan))
    p = sel(stale_wait, pot_wait, p)
    action = execute(K, r, c, facing, p, nav1)

    new_task = jnp.where(loadable | (~R_known) | stale_wait | prep, -1,
               jnp.where(unblock & (~held_ing), 3,
               jnp.where(held_ing, jnp.where(do_stash, 2, jnp.where(task == 3, 3, -1)),
               jnp.where(held >= 4, -1,
               jnp.where(pre_ok, pre_t, -1))))).astype(jnp.int32)
    at_pot_int = (action == INTERACT) & (p[0] == 3) & (p[1] == 5) & (p[2] == F_UP) & (p[3] == MODE_INT)
    new_mm = mem["mm"] + jnp.where(at_pot_int & mismatch, 1, 0)
    new_wait = jnp.where(dp & (pcount == 0), mem["wait"] + 1, 0)
    new_mem = {
        "recipe": i32(R),
        "pot": pblock.astype(jnp.int32),
        "timer": i32(ptimer),
        "slots": skind.astype(jnp.int32),
        "wait": i32(new_wait),
        "mm": i32(new_mm),
        "task": i32(new_task),
        "lastwrong": i32(lastwrong),
        "pstuck": i32(pstuck),
        "pcol": i32(pcol),
    }
    trace = {"r": i32(r), "c": i32(c), "held": i32(held), "tr": p[0], "tc": p[1], "tdir": p[2],
             "mode": p[3], "recipe": i32(R), "pcount": i32(pcount), "ptype": i32(ptype),
             "ptimer": i32(ptimer), "loadable": i32(loadable), "n0": i32(n0), "n1": i32(n1), "task": i32(task)}
    return action, new_mem, trace
