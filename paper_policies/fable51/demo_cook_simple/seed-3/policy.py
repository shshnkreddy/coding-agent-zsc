"""Overcooked V2 demo_cook_simple: convention-light two-room coordination policy.

Agent 0 (left room) can read the always-on recipe indicator and reach the pot.
It fetches the recipe ingredient and adds it to the pot whenever the pot can
accept it, pre-fetching the next unit while the pot cooks.

Agent 1 (right room) cannot see the recipe.  It treats the pot contents as the
only recipe evidence: it completes a partially filled pot with the same
ingredient, fetches a plate while the pot cooks, collects and delivers the
dish, and dumps obviously wrong (mixed / ingredient-2) dishes on counters.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'pot_driven_rooms',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# ----------------------------------------------------------------------------
# Static layout knowledge (plain Python literals; arrays are built lazily)
# ----------------------------------------------------------------------------
H, W = 5, 11
GRID = [
    "WWWWWR2W0WW",
    "0      W  B",
    "W     APA X",
    "1      W  B",
    "WWWWWR2W1WW",
]
STATIC_INDEX = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'L': 4, 'B': 5, '0': 6, '1': 7, '2': 8}

# Actions
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Facing direction ids (observation encoding)
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3

POT_RC = (2, 7)
IND_RC = ((0, 5), (4, 5))

# Agent 0 interaction targets: (stand_r, stand_c, dir)
A0_POT = (2, 6, D_RIGHT)
A0_SUPPLY = [
    (1, 1, D_LEFT),  # ingredient 0 at (1,0)
    (3, 1, D_LEFT),  # ingredient 1 at (3,0)
]
A0_COUNTERS = [  # (stand_r, stand_c, dir, cell_r, cell_c, shared_penalty)
    (1, 1, D_UP, 0, 1, 0), (1, 2, D_UP, 0, 2, 0), (1, 3, D_UP, 0, 3, 0), (1, 4, D_UP, 0, 4, 0),
    (3, 1, D_DOWN, 4, 1, 0), (3, 2, D_DOWN, 4, 2, 0), (3, 3, D_DOWN, 4, 3, 0), (3, 4, D_DOWN, 4, 4, 0),
    (2, 1, D_LEFT, 2, 0, 0),
    (1, 6, D_RIGHT, 1, 7, 100), (3, 6, D_RIGHT, 3, 7, 100),
]
A0_LOOK = (2, 4)

# Agent 1 targets
A1_POT = (2, 8, D_LEFT)
A1_SUPPLY = [
    (1, 8, D_UP),    # ingredient 0 at (0,8)
    (3, 8, D_DOWN),  # ingredient 1 at (4,8)
]
A1_PLATES = [(1, 9, D_RIGHT), (3, 9, D_RIGHT)]
A1_DELIVER = (2, 9, D_RIGHT)
A1_COUNTERS = [
    (1, 9, D_UP, 0, 9, 0), (3, 9, D_DOWN, 4, 9, 0),
    (1, 8, D_LEFT, 1, 7, 100), (3, 8, D_LEFT, 3, 7, 100),
]

PENDING_MAX = 6
A0_NEAR_IDX = [2, 3, 6, 7]  # (0,3), (0,4), (4,3), (4,4): private counters near the pot
COUNTER_EVIDENCE_WAIT = 24
BIG = 10000


def _static_map():
    m = [[[0] * 9 for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            idx = STATIC_INDEX.get(GRID[r][c])
            if idx is not None:
                m[r][c][idx] = 1
    return m


def _floor_cells():
    return [(r, c) for r in range(H) for c in range(W) if GRID[r][c] in ' A']


def _templates():
    m = _static_map()
    out = []
    for (r, c) in _floor_cells():
        t = [[[0] * 9 for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                rr, cc = r + i - 2, c + j - 2
                if 0 <= rr < H and 0 <= cc < W:
                    t[i][j] = list(m[rr][cc])
        out.append(t)
    return out


def _arr(x):
    return jnp.asarray(x, dtype=jnp.int32)


def consts():
    k = {}
    k['TEMPL'] = _arr(_templates())            # [24,5,5,9]
    k['FLOOR_RC'] = _arr(_floor_cells())        # [24,2]
    k['DIR_ACT'] = _arr([UP, DOWN, RIGHT, LEFT])
    k['A0_C_STAND'] = _arr([[c[0], c[1]] for c in A0_COUNTERS])
    k['A0_C_DIR'] = _arr([c[2] for c in A0_COUNTERS])
    k['A0_C_CELL'] = _arr([[c[3], c[4]] for c in A0_COUNTERS])
    k['A0_C_PEN'] = _arr([c[5] for c in A0_COUNTERS])
    k['A0_S_STAND'] = _arr([[s[0], s[1]] for s in A0_SUPPLY])
    k['A0_S_DIR'] = _arr([s[2] for s in A0_SUPPLY])
    k['A1_C_STAND'] = _arr([[c[0], c[1]] for c in A1_COUNTERS])
    k['A1_C_DIR'] = _arr([c[2] for c in A1_COUNTERS])
    k['A1_C_CELL'] = _arr([[c[3], c[4]] for c in A1_COUNTERS])
    k['A1_C_PEN'] = _arr([c[5] for c in A1_COUNTERS])
    k['A1_S_STAND'] = _arr([[s[0], s[1]] for s in A1_SUPPLY])
    k['A1_S_DIR'] = _arr([s[2] for s in A1_SUPPLY])
    k['A1_P_STAND'] = _arr([[p[0], p[1]] for p in A1_PLATES])
    k['A1_P_DIR'] = _arr([p[2] for p in A1_PLATES])
    return k


# ----------------------------------------------------------------------------
# Observation helpers
# ----------------------------------------------------------------------------
def localize(obs, K):
    static = obs[:, :, 20:29]
    diff = jnp.abs(static[None] - K['TEMPL']).sum(axis=(1, 2, 3))
    idx = jnp.argmin(diff)
    return K['FLOOR_RC'][idx, 0], K['FLOOR_RC'][idx, 1]


def cell(obs, sr, sc, r, c):
    """Return (40-vector, visible flag) for global cell (r, c)."""
    li = r - sr + 2
    lj = c - sc + 2
    vis = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    v = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4)]
    return v * vis.astype(v.dtype), vis


def item_code(v):
    """0 empty, 1..3 raw ingredient k+1, 4 plate, 5 cooked dish."""
    plate, cooked = v[29], v[30]
    counts = v[31:34]
    n = counts.sum()
    raw = (cooked == 0) & (plate == 0) & (n > 0)
    code = jnp.where(cooked > 0, 5, jnp.where(plate > 0, 4, jnp.where(raw, 1 + jnp.argmax(counts), 0)))
    return code.astype(jnp.int32)


def update_counters(obs, sr, sc, cells, mem_codes):
    def one(rc, old):
        v, vis = cell(obs, sr, sc, rc[0], rc[1])
        return jnp.where(vis, item_code(v), old)
    return jax.vmap(one)(cells, mem_codes)


def move_toward(sr, sc, tr, tc):
    dr = tr - sr
    dc = tc - sc
    a_row = jnp.where(dr < 0, UP, DOWN)
    a_col = jnp.where(dc < 0, LEFT, RIGHT)
    return jnp.where(dr != 0, a_row, jnp.where(dc != 0, a_col, STAY)).astype(jnp.int32)


def do_task(K, sr, sc, facing, tr, tc, tdir, interact_ok):
    """Walk to (tr, tc), face tdir, then interact (if allowed) else stay."""
    at = (sr == tr) & (sc == tc)
    turn = K['DIR_ACT'][tdir]
    at_act = jnp.where(facing == tdir, jnp.where(interact_ok, INTERACT, STAY), turn)
    return jnp.where(at, at_act, move_toward(sr, sc, tr, tc)).astype(jnp.int32)


def nearest(sr, sc, stands, valid, penalty=None):
    d = jnp.abs(stands[:, 0] - sr) + jnp.abs(stands[:, 1] - sc)
    if penalty is not None:
        d = d + penalty
    d = jnp.where(valid, d, BIG)
    i = jnp.argmin(d)
    return i, d[i] < BIG


def decode_self(obs):
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    held_plate = inv[0] > 0
    held_cooked = inv[1] > 0
    counts = inv[2:5]
    held_raw = (~held_cooked) & (~held_plate) & (counts.sum() > 0)
    held_k = jnp.argmax(counts)
    held_any = inv.sum() > 0
    return facing, held_plate, held_cooked, counts, held_raw, held_k, held_any


def decode_pot(obs, sr, sc):
    pv, pvis = cell(obs, sr, sc, POT_RC[0], POT_RC[1])
    counts = pv[31:34]
    n = counts.sum()
    cooked = pv[30] > 0
    cooking = pv[39] > 0
    return pvis, counts, n, cooked, cooking


def wrong_dish(counts):
    return (counts[2] > 0) | ((counts[0] > 0) & (counts[1] > 0))


# ----------------------------------------------------------------------------
# Agent 0: recipe reader / ingredient supplier (left room)
# ----------------------------------------------------------------------------
def init_agent_0():
    return {
        'recipe': jnp.asarray(-1, dtype=jnp.int32),
        'pending': jnp.asarray(0, dtype=jnp.int32),   # delivery may be in flight
        'pend_age': jnp.asarray(0, dtype=jnp.int32),
        'prev_pot': jnp.asarray(-1, dtype=jnp.int32),  # -1 unseen, 0 not busy, 1 cooking/ready
        'counters': jnp.zeros((len(A0_COUNTERS),), dtype=jnp.int32),
        'stock_trip': jnp.asarray(0, dtype=jnp.int32),
    }


def partner_info(obs):
    ppos = obs[:, :, 10]
    vis = ppos.sum() > 0
    inv = (obs[:, :, 15:20] * ppos[:, :, None]).sum(axis=(0, 1))
    plate = inv[0] > 0
    cooked = inv[1] > 0
    counts = inv[2:5]
    raw = vis & (~plate) & (~cooked) & (counts.sum() > 0)
    k = jnp.argmax(counts).astype(jnp.int32)
    return vis, vis & cooked, raw, k


def partner_global(obs, sr, sc):
    """Partner's global (row, col) if visible (else (-9, -9))."""
    ppos = obs[:, :, 10]
    vis = ppos.sum() > 0
    flat = jnp.argmax(ppos.reshape(-1))
    li = flat // 5
    lj = flat % 5
    pr = jnp.where(vis, sr + li - 2, -9).astype(jnp.int32)
    pc = jnp.where(vis, sc + lj - 2, -9).astype(jnp.int32)
    return pr, pc


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(jnp.int32)
    K = consts()
    sr, sc = localize(obs, K)
    facing, held_plate, held_cooked, hcounts, held_raw, held_k, held_any = decode_self(obs)

    # Recipe from indicators.
    r1, vis1 = cell(obs, sr, sc, IND_RC[0][0], IND_RC[0][1])
    r2, vis2 = cell(obs, sr, sc, IND_RC[1][0], IND_RC[1][1])
    rc1 = r1[36:39]
    rc2 = r2[36:39]
    rec_counts = jnp.where(vis1 & (rc1.sum() > 0), rc1, rc2)
    rec_vis = (vis1 & (rc1.sum() > 0)) | (vis2 & (rc2.sum() > 0))
    recipe = jnp.where(rec_vis, jnp.argmax(rec_counts), mem['recipe']).astype(jnp.int32)
    recipe_known = recipe >= 0
    recipe_changed = (mem['recipe'] >= 0) & (recipe != mem['recipe'])
    rk = jnp.clip(recipe, 0, 2)
    alt = jnp.where(rk == 0, 1, 0).astype(jnp.int32)

    # Pot state.
    pvis, pcounts, pn, pcooked, pcooking = decode_pot(obs, sr, sc)
    pot_busy = pvis & (pcooked | pcooking)
    pot_empty = pvis & (pn == 0)
    other_types = pn - pcounts[rk]
    pot_accepts = pvis & (~pot_busy) & (pn < 3) & (other_types == 0)

    # Delivery-in-flight tracking: after the pot is emptied (or the partner is
    # seen carrying a dish) the recipe may be resampled; wait until the partner
    # is seen again without a dish, or a step cap expires.
    p_vis, p_dish, p_raw, p_k = partner_info(obs)
    transition = (mem['prev_pot'] == 1) & pot_empty
    start = transition | p_dish
    release = p_vis & (~p_dish)
    pend_age = jnp.where(start, 0, mem['pend_age'] + 1).astype(jnp.int32)
    pending = ((mem['pending'] > 0) | start) & (~release) & (pend_age < PENDING_MAX)
    pending = pending.astype(jnp.int32)
    prev_pot = jnp.where(pvis, jnp.where(pot_busy, 1, 0), mem['prev_pot']).astype(jnp.int32)

    counters = update_counters(obs, sr, sc, K['A0_C_CELL'], mem['counters'])
    near_mask = jnp.zeros((len(A0_COUNTERS),), dtype=bool).at[jnp.asarray(A0_NEAR_IDX)].set(True)
    have_alt_near = ((counters == (alt + 1)) & near_mask).any()

    # Stock trip: while the pot is being filled/cooked, fetch the alternate
    # ingredient once and park it on a private counter near the pot, so a
    # recipe change costs a short swap instead of a full supply round trip.
    trip_start = (~held_any) & recipe_known & pvis & (pn >= 1) & (~have_alt_near)
    stock_trip = ((mem['stock_trip'] > 0) | trip_start) & recipe_known & (~have_alt_near) & (~recipe_changed)
    stock_trip = stock_trip.astype(jnp.int32)

    # --- candidate actions -------------------------------------------------
    # 1) add / wait at pot
    can_add = pot_accepts & (pending == 0) & held_raw & (held_k == recipe)
    act_pot = do_task(K, sr, sc, facing, A0_POT[0], A0_POT[1], A0_POT[2], can_add)

    # 2) dump held item on nearest empty counter (private preferred)
    ci, c_ok = nearest(sr, sc, K['A0_C_STAND'], counters == 0, K['A0_C_PEN'])
    act_dump = do_task(K, sr, sc, facing, K['A0_C_STAND'][ci, 0], K['A0_C_STAND'][ci, 1], K['A0_C_DIR'][ci], True)
    act_dump = jnp.where(c_ok, act_dump, STAY)

    # 2b) place alternate on nearest empty near counter
    ni, n_ok = nearest(sr, sc, K['A0_C_STAND'], (counters == 0) & near_mask)
    act_place = do_task(K, sr, sc, facing, K['A0_C_STAND'][ni, 0], K['A0_C_STAND'][ni, 1], K['A0_C_DIR'][ni], True)
    act_place = jnp.where(n_ok, act_place, act_dump)

    # 3) fetch ingredient `want` from nearest source (supply or counter holding it)
    want = jnp.where(stock_trip > 0, alt, rk)
    src_stand = jnp.concatenate([K['A0_S_STAND'], K['A0_C_STAND']], axis=0)
    src_dir = jnp.concatenate([K['A0_S_DIR'], K['A0_C_DIR']], axis=0)
    src_valid = jnp.concatenate([jnp.arange(2) == want, counters == (want + 1)], axis=0)
    si, s_ok = nearest(sr, sc, src_stand, src_valid)
    act_fetch = do_task(K, sr, sc, facing, src_stand[si, 0], src_stand[si, 1], src_dir[si], True)
    act_fetch = jnp.where(s_ok, act_fetch, STAY)

    # 4) look for recipe
    act_look = move_toward(sr, sc, A0_LOOK[0], A0_LOOK[1])

    hold_good = held_raw & recipe_known & (held_k == recipe)
    hold_alt = held_raw & recipe_known & (held_k == alt) & (stock_trip > 0)
    action = jnp.where(
        held_any,
        jnp.where(hold_good, act_pot, jnp.where(hold_alt, act_place, act_dump)),
        jnp.where(recipe_known, act_fetch, act_look),
    ).astype(jnp.int32)
    mode = jnp.where(held_any, jnp.where(hold_good, 1, jnp.where(hold_alt, 5, 2)),
                     jnp.where(recipe_known, jnp.where(stock_trip > 0, 6, 3), 4))

    new_mem = {
        'recipe': recipe,
        'pending': pending,
        'pend_age': pend_age,
        'prev_pot': prev_pot,
        'counters': counters.astype(jnp.int32),
        'stock_trip': stock_trip,
    }
    trace = {
        'row': sr.astype(jnp.int32), 'col': sc.astype(jnp.int32),
        'recipe': recipe, 'mode': mode.astype(jnp.int32),
        'pending': pending, 'pot_n': pn.astype(jnp.int32),
        'stock_trip': stock_trip,
    }
    return action, new_mem, trace


# ----------------------------------------------------------------------------
# Agent 1: pot completer / plater / deliverer (right room)
# ----------------------------------------------------------------------------
def init_agent_1():
    return {
        'counters': jnp.zeros((len(A1_COUNTERS),), dtype=jnp.int32),
        'empty_steps': jnp.asarray(0, dtype=jnp.int32),
        'suspect': jnp.asarray(0, dtype=jnp.int32),
        'dish_suspect': jnp.asarray(0, dtype=jnp.int32),
        'prev_cooked': jnp.asarray(0, dtype=jnp.int32),
        'pk_seen': jnp.asarray(-1, dtype=jnp.int32),   # raw type partner held at pot stand with empty pot
        'infer': jnp.asarray(-1, dtype=jnp.int32),     # inferred recipe after partner walked away
        'prev_pvis': jnp.asarray(0, dtype=jnp.int32),
        'prev_sr': jnp.asarray(-1, dtype=jnp.int32),
        'prev_sc': jnp.asarray(-1, dtype=jnp.int32),
        'c_age': jnp.zeros((len(A1_COUNTERS),), dtype=jnp.int32),  # steps since counter content changed
        'since_deliv': jnp.asarray(999, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(jnp.int32)
    K = consts()
    sr, sc = localize(obs, K)
    facing, held_plate, held_cooked, hcounts, held_raw, held_k, held_any = decode_self(obs)

    pvis, pcounts, pn, pcooked, pcooking = decode_pot(obs, sr, sc)
    pot_busy = pcooked | pcooking
    pot_empty = pn == 0
    pot_partial = (pn > 0) & (pn < 3) & (~pot_busy)
    usable = (pcounts[0] + pcounts[1]) > 0
    dominant = jnp.argmax(pcounts[0:2]).astype(jnp.int32)
    pot_pure = (pn > 0) & (pcounts[2] == 0) & ((pcounts[0] == 0) | (pcounts[1] == 0))

    counters = update_counters(obs, sr, sc, K['A1_C_CELL'], mem['counters'])
    c_age = jnp.where(counters != mem['counters'], 0, jnp.minimum(mem['c_age'] + 1, 999)).astype(jnp.int32)
    just_delivered = (mem['prev_cooked'] > 0) & (~held_cooked) & (sr == A1_DELIVER[0]) & (sc == A1_DELIVER[1])
    since_deliv = jnp.where(just_delivered, 0, jnp.minimum(mem['since_deliv'] + 1, 999)).astype(jnp.int32)
    empty_steps = jnp.where(pot_empty, mem['empty_steps'] + 1, 0).astype(jnp.int32)

    # Partner evidence: a partner visibly holding a raw ingredient that differs
    # from a pure pot's ingredient suggests the pot does not match the recipe.
    p_vis, p_dish, p_raw, p_k = partner_info(obs)
    p_raw01 = p_raw & (p_k <= 1)
    conflict = pot_pure & p_raw01 & (p_k != dominant)
    suspect = jnp.where(pot_empty, 0, jnp.maximum(mem['suspect'], conflict.astype(jnp.int32))).astype(jnp.int32)
    just_collected = held_cooked & (mem['prev_cooked'] == 0)
    dish_suspect = jnp.where(just_collected, mem['suspect'],
                             jnp.where(held_cooked, mem['dish_suspect'], 0)).astype(jnp.int32)

    # Fallback evidence when the pot stays empty: shared counters (indices 2,3)
    # holding raw 0/1, or the partner visibly holding raw 0/1.
    # A counter item only counts as evidence if it appeared after our last
    # delivery (older items may be stale leftovers from a previous recipe).
    shared_codes = counters[2:4]
    shared_age = c_age[2:4]
    shared_has = ((shared_codes == 1) | (shared_codes == 2)) & (shared_age < since_deliv)
    shared_any = shared_has.any()
    shared_pick = jnp.argmin(jnp.where(shared_has, shared_age, 9999))
    shared_k = (shared_codes[shared_pick] - 1).astype(jnp.int32)
    fb_k = jnp.where(p_raw01, p_k, shared_k)
    use_fallback = pot_empty & (empty_steps >= COUNTER_EVIDENCE_WAIT) & (shared_any | p_raw01)

    # Walk-away inference: a partner that stood at the pot stand holding raw k
    # while the pot was empty, then left still holding it, evidently could not
    # use k; with two recipes the target must be the other ingredient.
    pr, pc = partner_global(obs, sr, sc)
    at_stand = p_vis & (pr == A0_POT[0]) & (pc == A0_POT[1])
    armed = mem['pk_seen'] >= 0
    pk_seen = jnp.where(
        ~pot_empty, -1,
        jnp.where(p_vis & (~p_raw01), -1,
                  jnp.where(p_raw01 & (at_stand | armed), p_k, mem['pk_seen']))).astype(jnp.int32)
    self_still = (sr == mem['prev_sr']) & (sc == mem['prev_sc'])
    walked_away = (~p_vis) & (mem['prev_pvis'] > 0) & armed & pot_empty & self_still
    infer = jnp.where(~pot_empty, -1,
                      jnp.where(walked_away, 1 - mem['pk_seen'], mem['infer'])).astype(jnp.int32)
    pk_seen = jnp.where(walked_away, -1, pk_seen).astype(jnp.int32)
    use_infer = pot_empty & (infer >= 0)

    # Any idle partial pot gets completed with its dominant 0/1 ingredient: a
    # matching pot becomes a correct dish, a doomed (mixed / ingredient-2) pot
    # gets cooked so it can be plated and dumped, freeing the pot.
    need = jnp.where(pot_partial, dominant,
                     jnp.where(use_infer, infer,
                               jnp.where(use_fallback, fb_k, -1))).astype(jnp.int32)
    need_known = need >= 0
    nk = jnp.clip(need, 0, 1)

    # --- candidate actions -------------------------------------------------
    act_deliver = do_task(K, sr, sc, facing, A1_DELIVER[0], A1_DELIVER[1], A1_DELIVER[2], True)

    ci, c_ok = nearest(sr, sc, K['A1_C_STAND'], counters == 0, K['A1_C_PEN'])
    act_dump = do_task(K, sr, sc, facing, K['A1_C_STAND'][ci, 0], K['A1_C_STAND'][ci, 1], K['A1_C_DIR'][ci], True)

    # collect when ready, else wait facing the pot
    act_pot_collect = do_task(K, sr, sc, facing, A1_POT[0], A1_POT[1], A1_POT[2], pcooked)
    # add held ingredient
    held_other = hcounts.sum() - hcounts[jnp.clip(held_k, 0, 2)]
    pot_doomed = (pn > 0) & (~pot_pure)
    pot_accepts_held = (~pot_busy) & (pn < 3) & (((pn - pcounts[jnp.clip(held_k, 0, 2)]) == 0) | pot_doomed)
    act_pot_add = do_task(K, sr, sc, facing, A1_POT[0], A1_POT[1], A1_POT[2], True)
    act_wait = do_task(K, sr, sc, facing, A1_POT[0], A1_POT[1], A1_POT[2], False)

    # fetch plate: nearest pile or counter holding a plate
    pl_stand = jnp.concatenate([K['A1_P_STAND'], K['A1_C_STAND']], axis=0)
    pl_dir = jnp.concatenate([K['A1_P_DIR'], K['A1_C_DIR']], axis=0)
    pl_valid = jnp.concatenate([jnp.ones((2,), dtype=bool), counters == 4], axis=0)
    pi, _ = nearest(sr, sc, pl_stand, pl_valid)
    act_plate = do_task(K, sr, sc, facing, pl_stand[pi, 0], pl_stand[pi, 1], pl_dir[pi], True)

    # fetch ingredient `need`
    src_stand = jnp.concatenate([K['A1_S_STAND'], K['A1_C_STAND']], axis=0)
    src_dir = jnp.concatenate([K['A1_S_DIR'], K['A1_C_DIR']], axis=0)
    src_valid = jnp.concatenate([jnp.arange(2) == nk, counters == (nk + 1)], axis=0)
    # Prefer consuming a passed counter item over the supply at equal distance.
    src_pen = jnp.concatenate([jnp.ones((2,), dtype=jnp.int32), jnp.zeros((len(A1_COUNTERS),), dtype=jnp.int32)], axis=0)
    si, _ = nearest(sr, sc, src_stand, src_valid, src_pen)
    act_fetch = do_task(K, sr, sc, facing, src_stand[si, 0], src_stand[si, 1], src_dir[si], True)

    # --- decision ----------------------------------------------------------
    dish_bad = wrong_dish(hcounts) | (dish_suspect > 0)
    act_if_dish = jnp.where(dish_bad & c_ok, act_dump, act_deliver)
    act_if_plate = jnp.where(pot_busy, act_pot_collect, jnp.where(c_ok, act_dump, act_wait))
    act_if_raw = jnp.where(pot_accepts_held, act_pot_add, jnp.where(c_ok, act_dump, act_wait))
    act_if_empty = jnp.where(
        pot_busy, act_plate,
        jnp.where(need_known, act_fetch, act_wait),
    )
    action = jnp.where(
        held_cooked, act_if_dish,
        jnp.where(held_plate, act_if_plate,
                  jnp.where(held_raw, act_if_raw, act_if_empty)),
    ).astype(jnp.int32)
    mode = jnp.where(held_cooked, 1, jnp.where(held_plate, 2, jnp.where(held_raw, 3, jnp.where(pot_busy, 4, jnp.where(need_known, 5, 6)))))

    new_mem = {
        'counters': counters.astype(jnp.int32),
        'empty_steps': empty_steps,
        'suspect': suspect,
        'dish_suspect': dish_suspect,
        'prev_cooked': held_cooked.astype(jnp.int32),
        'pk_seen': pk_seen,
        'infer': infer,
        'prev_pvis': p_vis.astype(jnp.int32),
        'prev_sr': sr.astype(jnp.int32),
        'prev_sc': sc.astype(jnp.int32),
        'c_age': c_age,
        'since_deliv': since_deliv,
    }
    trace = {
        'row': sr.astype(jnp.int32), 'col': sc.astype(jnp.int32),
        'mode': mode.astype(jnp.int32), 'need': need,
        'pot_n': pn.astype(jnp.int32), 'empty_steps': empty_steps,
        'suspect': suspect, 'dish_suspect': dish_suspect,
        'infer': infer,
    }
    return action, new_mem, trace
