"""Overcooked V2 test_time_simple: counter-signal recipe protocol.

Geometry facts used (from context/environment):
  * agent_0 lives in the left room (rows 1-3, cols 1-2). It can see the
    always-on recipe indicator R at (2,0), the shared counters (1,3),(2,3),(3,3)
    and the pot at (2,4). It can only obtain ingredient 2 (piles at (0,2),(4,2)).
  * agent_1 lives in the right room (rows 1-3, cols 5-6). It has the pot (2,4),
    plate piles (1,4),(3,4), ingredient piles 0 at (1,7), 1 at (3,7) and the
    delivery station (2,7). It can never see R, but it sees the shared
    counters (column 3) whenever it stands in column 5.
  * The seats never see each other. The only agent_0 -> agent_1 channel is the
    occupancy of the three shared counters.

Protocol (agent_0 always displays the current recipe):
  recipe [0,0,0] -> one item on the top counter (1,3)
  recipe [1,1,1] -> items on the middle (2,3) and bottom (3,3) counters
This is readable under top/bottom, counter-index, middle-bit and item-count
conventions. agent_1 decodes leniently (see _DECODE), waits a grace period
after its own deliveries before committing to a fresh pot, re-verifies at the
pot, and parks mismatched ingredients / dishes on right-room counters.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'counter_signal_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Action ids
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids (observation encoding)
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3

# Channels
CH_POT = 22
CH_RECIPE_IND = 23
CH_ITEM = 29  # item block [29, 34): plate, cooked, ing0, ing1, ing2
CH_RECIPE = 34  # recipe block [34, 39)
CH_TIMER = 39

# agent_1 tuning
GRACE_AFTER_DELIVERY = 10  # steps before trusting the signal for a commitment
STABLE_NEEDED = 2          # consecutive identical signal observations
EMPTY_TIMEOUT = 20         # consecutive empty-signal observations before guessing recipe 0

# agent_0 adaptive-signal tuning
NUM_VARIANTS = 5
MISMATCH_COMPLETE_STEPS = 15   # signal must have been complete this long before a wrong load counts
MISMATCH_COMPLETE_CONF = 25    # stricter once the variant has been confirmed
CONFIRM_COMPLETE_STEPS = 5
IDLE_TIMEOUT_R0 = 40           # empty pot + complete signal this long (unconfirmed) -> next variant
IDLE_TIMEOUT_R1 = 30           # r1 fallback chain ({m,b}->{b}->{m}) is harmless to lenient decoders

I32 = jnp.int32


def _i(x):
    return jnp.asarray(x, dtype=I32)


def _dir_to_action(d):
    table = jnp.array([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=I32)
    return table[d]


def _task(tr, tc, tf, interact):
    return jnp.array([tr, tc, tf, interact], dtype=I32)


def _nav(r, c, f, task):
    """Move to task cell, face task direction, then INTERACT (or STAY)."""
    tr, tc, tf, inter = task[0], task[1], task[2], task[3]
    move_v = jnp.where(tr < r, A_UP, A_DOWN)
    move_h = jnp.where(tc < c, A_LEFT, A_RIGHT)
    final = jnp.where(inter > 0, A_INTERACT, A_STAY)
    act = jnp.where(r != tr, move_v,
                    jnp.where(c != tc, move_h,
                              jnp.where(f == tf, final, _dir_to_action(tf))))
    return act.astype(I32)


def _cell(obs, gr, gc, sr, sc):
    """Return (visible, 40-channel block) for global cell (gr, gc)."""
    i = gr - sr + 2
    j = gc - sc + 2
    vis = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    blk = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), :]
    return vis, blk


def _occupied(blk):
    return blk[CH_ITEM:CH_ITEM + 5].sum() > 0


def _item_code(blk):
    """0 empty, 1 ing0, 2 ing1, 3 dish0, 4 dish1, 5 other."""
    p = blk[CH_ITEM]
    ck = blk[CH_ITEM + 1]
    c0 = blk[CH_ITEM + 2]
    c1 = blk[CH_ITEM + 3]
    c2 = blk[CH_ITEM + 4]
    empty = (p == 0) & (ck == 0) & (c0 + c1 + c2 == 0)
    raw = (p == 0) & (ck == 0)
    ing0 = raw & (c0 > 0) & (c1 == 0) & (c2 == 0)
    ing1 = raw & (c1 > 0) & (c0 == 0) & (c2 == 0)
    dish0 = (ck == 1) & (c0 == 3) & (c1 == 0) & (c2 == 0)
    dish1 = (ck == 1) & (c1 == 3) & (c0 == 0) & (c2 == 0)
    return jnp.where(empty, 0,
                     jnp.where(ing0, 1,
                               jnp.where(ing1, 2,
                                         jnp.where(dish0, 3,
                                                   jnp.where(dish1, 4, 5))))).astype(I32)


def _select(cond, a, b):
    return jnp.where(cond, a, b)


# --------------------------------------------------------------------------
# agent_0: recipe signaller (left room)
# --------------------------------------------------------------------------

# agent_0 memory slots
Z_STEP = 0
Z_V0 = 1          # variant index for recipe 0
Z_V1 = 2          # variant index for recipe 1
Z_CONF0 = 3       # variant confirmed by a matching partner load
Z_CONF1 = 4
Z_COMPLETE = 5    # consecutive steps with displayed pattern == target pattern
Z_POT_N = 6       # last known pot ingredient count
Z_POT_T = 7       # last known pot ingredient type
Z_EMPTY = 8       # consecutive steps with (last known) empty pot
Z_SWITCHES = 9    # number of variant switches (diagnostic)
Z_SIZE = 12


def _variant_table():
    # pattern code = t*4 + m*2 + b
    # recipe 0: {t}, {}, {b}, {m}, {t,m}
    # recipe 1: {m,b}, {b}, {m}, {t}, {t,m}
    return jnp.array([[4, 0, 1, 2, 6],
                      [3, 1, 2, 4, 6]], dtype=I32)


def init_agent_0():
    return jnp.zeros((Z_SIZE,), dtype=I32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(I32)
    mem = jnp.where(reset, init_agent_0(), memory)

    # Localise via the recipe indicator at global (2, 0).
    ind = obs[:, :, CH_RECIPE_IND].reshape(-1)
    idx = jnp.argmax(ind)
    ri = idx // 5
    rj = idx % 5
    r = 4 - ri
    c = 2 - rj
    ind_blk = obs[ri, rj, :]
    recipe = jnp.where(ind_blk[CH_RECIPE + 3] > 0, 1, 0).astype(I32)

    self_blk = obs[2, 2, :]
    f = jnp.argmax(self_blk[1:5]).astype(I32)
    holding = self_blk[5:10].sum() > 0

    # Shared counters.
    _, blk_t = _cell(obs, 1, 3, r, c)
    _, blk_m = _cell(obs, 2, 3, r, c)
    _, blk_b = _cell(obs, 3, 3, r, c)
    occ = jnp.stack([_occupied(blk_t), _occupied(blk_m), _occupied(blk_b)])
    occ_pat = (occ[0].astype(I32) * 4 + occ[1].astype(I32) * 2 + occ[2].astype(I32))

    # ---- partner feedback via the pot (visible only from column 2) ----
    pot_vis, pot_blk = _cell(obs, 2, 4, r, c)
    pcnt = pot_blk[CH_ITEM + 2:CH_ITEM + 5]
    pn = pcnt.sum()
    ptype = jnp.argmax(pcnt).astype(I32)
    prev_pn = mem[Z_POT_N]
    first_insert = pot_vis & (pn > 0) & ((prev_pn == 0) | (pn < prev_pn))
    known_pn = jnp.where(pot_vis, pn, prev_pn)
    known_pt = jnp.where(pot_vis, ptype, mem[Z_POT_T])

    variants = _variant_table()
    v_cur = jnp.where(recipe == 0, mem[Z_V0], mem[Z_V1])
    conf_cur = jnp.where(recipe == 0, mem[Z_CONF0], mem[Z_CONF1])
    target_pat = variants[recipe, v_cur]
    complete_prev = mem[Z_COMPLETE]

    need_steps = jnp.where(conf_cur == 1, MISMATCH_COMPLETE_CONF, MISMATCH_COMPLETE_STEPS)
    mismatch = first_insert & (ptype != recipe) & (complete_prev >= need_steps)
    confirm = first_insert & (ptype == recipe) & (complete_prev >= CONFIRM_COMPLETE_STEPS)
    empty_prev = mem[Z_EMPTY]
    idle_limit = jnp.where(recipe == 0, IDLE_TIMEOUT_R0, IDLE_TIMEOUT_R1)
    timeout = (empty_prev >= idle_limit) & (complete_prev >= idle_limit) & (conf_cur == 0)
    switch = mismatch | timeout

    v_new = (v_cur + 1) % NUM_VARIANTS
    v0 = jnp.where(switch & (recipe == 0), v_new, mem[Z_V0])
    v1 = jnp.where(switch & (recipe == 1), v_new, mem[Z_V1])
    conf0 = jnp.where(switch & (recipe == 0), 0, jnp.where(confirm & (recipe == 0), 1, mem[Z_CONF0]))
    conf1 = jnp.where(switch & (recipe == 1), 0, jnp.where(confirm & (recipe == 1), 1, mem[Z_CONF1]))
    v_use = jnp.where(recipe == 0, v0, v1)
    target_pat = variants[recipe, v_use]
    target = jnp.stack([(target_pat // 4) % 2 == 1, (target_pat // 2) % 2 == 1, target_pat % 2 == 1])
    complete_now = occ_pat == target_pat
    complete_steps = jnp.where(switch, 0, jnp.where(complete_now, complete_prev + 1, 0))
    complete_steps = jnp.minimum(complete_steps, 100000)
    empty_steps = jnp.where(switch, 0, jnp.where(known_pn == 0, empty_prev + 1, 0))
    empty_steps = jnp.minimum(empty_steps, 100000)

    need_place = target & (~occ)
    need_pick = occ & (~target)
    rows = jnp.array([1, 2, 3], dtype=I32)
    dist = jnp.abs(rows - r)
    place_k = jnp.argmin(jnp.where(need_place, dist, 99))
    pick_k = jnp.argmin(jnp.where(need_pick, dist, 99))
    any_place = need_place.any()
    any_pick = need_pick.any()
    target_count = target.astype(I32).sum()
    want_hand = target_count <= 1

    task_place = _task(rows[place_k], 2, D_RIGHT, 1)
    task_pick = _task(rows[pick_k], 2, D_RIGHT, 1)
    task_pile = _select(r < 2, _task(1, 2, D_UP, 1), _task(3, 2, D_DOWN, 1))

    # Far-left counters (never visible to agent_1): spare storage.
    v01, b01 = _cell(obs, 0, 1, r, c)
    v10, b10 = _cell(obs, 1, 0, r, c)
    v41, b41 = _cell(obs, 4, 1, r, c)
    v30, b30 = _cell(obs, 3, 0, r, c)
    occ01 = v01 & _occupied(b01)
    occ10 = v10 & _occupied(b10)
    occ41 = v41 & _occupied(b41)
    occ30 = v30 & _occupied(b30)
    free01 = v01 & (~_occupied(b01))
    free10 = v10 & (~_occupied(b10))
    free41 = v41 & (~_occupied(b41))
    free30 = v30 & (~_occupied(b30))
    t01 = _task(1, 1, D_UP, 1)
    t10 = _task(1, 1, D_LEFT, 1)
    t41 = _task(3, 1, D_DOWN, 1)
    t30 = _task(3, 1, D_LEFT, 1)
    top_drop = _select(free01, t01, _select(free10, t10, _select(free41, t41, _select(free30, t30, _task(1, 1, D_UP, 0)))))
    bot_drop = _select(free41, t41, _select(free30, t30, _select(free01, t01, _select(free10, t10, _task(3, 1, D_DOWN, 0)))))
    task_drop = _select(r <= 2, top_drop, bot_drop)
    # Fetch: recycle a stored spare if one is visible, otherwise use a pile.
    top_fetch = _select(occ01, t01, _select(occ10, t10, task_pile))
    bot_fetch = _select(occ41, t41, _select(occ30, t30, task_pile))
    task_fetch = _select(r <= 2, top_fetch, bot_fetch)

    task_idle = _task(2, 2, D_RIGHT, 0)

    task = _select(holding & any_place, task_place,
            _select((~holding) & any_pick, task_pick,
             _select(holding & any_pick, task_drop,
              _select((~holding) & any_place, task_fetch,
               _select((~holding) & want_hand, task_fetch,
                _select(holding & (~want_hand), task_drop, task_idle))))))

    action = _nav(r, c, f, task)
    step = jnp.minimum(mem[Z_STEP] + 1, 100000)
    switches = jnp.minimum(mem[Z_SWITCHES] + switch.astype(I32), 100000)
    new_mem = jnp.stack([step, v0, v1, conf0, conf1, complete_steps, known_pn, known_pt,
                         empty_steps, switches, _i(0), _i(0)]).astype(I32)
    trace = {
        'recipe': recipe,
        'pattern': occ_pat.astype(I32),
        'target': target_pat.astype(I32),
        'variant': v_use.astype(I32),
        'switches': switches.astype(I32),
        'complete': complete_steps.astype(I32),
        'holding': holding.astype(I32),
        'pot_n': known_pn.astype(I32),
    }
    return action, new_mem, trace


# --------------------------------------------------------------------------
# agent_1: cook / deliver (right room)
# --------------------------------------------------------------------------

# memory slots
M_LAST_PAT = 0
M_STABLE = 1
M_SINCE_DEL = 2
M_DEL_ATTEMPT = 3
M_EMPTY_CNT = 4
M_S05 = 5
M_S06 = 6
M_S45 = 7
M_S46 = 8
M_STEP = 9
M_SIZE = 12

def _decode_table():
    # pattern bits: t*4 + m*2 + b  ->  recipe (-1 unknown)
    return jnp.array([-1, 1, 1, 1, 0, -1, 1, -1], dtype=I32)


def init_agent_1():
    return jnp.zeros((M_SIZE,), dtype=I32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(I32)
    mem = jnp.where(reset, init_agent_1(), memory)

    # Localise via the pot at global (2, 4); always inside the view.
    pot_map = obs[:, :, CH_POT].reshape(-1)
    idx = jnp.argmax(pot_map)
    pi = idx // 5
    pj = idx % 5
    r = 4 - pi
    c = 6 - pj

    self_blk = obs[2, 2, :]
    f = jnp.argmax(self_blk[1:5]).astype(I32)
    hp = self_blk[5]
    hc = self_blk[6]
    hcnt = self_blk[7:10]
    hsum = hcnt.sum()
    held_none = (hp == 0) & (hc == 0) & (hsum == 0)
    held_ing = (hp == 0) & (hc == 0) & (hsum > 0)
    held_plate = (hp == 1) & (hc == 0)
    held_dish = hc == 1
    ing_type = jnp.argmax(hcnt).astype(I32)
    dish_type = jnp.where((hcnt[0] == 3) & (hcnt[1] == 0), 0,
                          jnp.where((hcnt[1] == 3) & (hcnt[0] == 0), 1, 2)).astype(I32)

    pot_blk = obs[pi, pj, :]
    pcnt = pot_blk[CH_ITEM + 2:CH_ITEM + 5]
    pcooked = pot_blk[CH_ITEM + 1]
    ptimer = pot_blk[CH_TIMER]
    pn = pcnt.sum()
    pot_type = jnp.argmax(pcnt).astype(I32)
    pot_cooking = ptimer > 0
    pot_ready = (pcooked == 1) & (ptimer == 0)
    pot_empty = (pn == 0) & (pcooked == 0) & (ptimer == 0)
    pot_partial = (pn > 0) & (pn < 3) & (~pot_cooking) & (pcooked == 0)

    # ---- delivery detection & counters ----
    delivered = (mem[M_DEL_ATTEMPT] == 1) & held_none
    since_del = jnp.where(delivered, 0, jnp.minimum(mem[M_SINCE_DEL] + 1, 100000))
    stable0 = jnp.where(delivered, 0, mem[M_STABLE])
    empty0 = jnp.where(delivered, 0, mem[M_EMPTY_CNT])

    # ---- signal counters (1,3),(2,3),(3,3) ----
    vt, blk_t = _cell(obs, 1, 3, r, c)
    vm, blk_m = _cell(obs, 2, 3, r, c)
    vb, blk_b = _cell(obs, 3, 3, r, c)
    sig_vis = vt & vm & vb
    pattern = (_occupied(blk_t).astype(I32) * 4 + _occupied(blk_m).astype(I32) * 2
               + _occupied(blk_b).astype(I32))
    same = pattern == mem[M_LAST_PAT]
    stable = jnp.where(sig_vis, jnp.where(same, stable0 + 1, 1), stable0)
    stable = jnp.minimum(stable, 100000)
    last_pat = jnp.where(sig_vis, pattern, mem[M_LAST_PAT])
    empty_cnt = jnp.where(sig_vis, jnp.where(pattern == 0, empty0 + 1, 0), empty0)
    empty_cnt = jnp.minimum(empty_cnt, 100000)

    r_dec = _decode_table()[last_pat]
    seen = stable >= 1
    guess = (empty_cnt >= EMPTY_TIMEOUT) & (since_del >= GRACE_AFTER_DELIVERY)
    r_opt = jnp.where(seen, r_dec, -1)
    r_opt = jnp.where((r_opt < 0) & guess, 0, r_opt)
    trusted = (stable >= STABLE_NEEDED) & (since_del >= GRACE_AFTER_DELIVERY)
    r_com = jnp.where(trusted, r_dec, -1)
    r_com = jnp.where((r_com < 0) & guess, 0, r_com)

    # ---- stash cells (0,5),(0,6),(4,5),(4,6) ----
    v05, b05 = _cell(obs, 0, 5, r, c)
    v06, b06 = _cell(obs, 0, 6, r, c)
    v45, b45 = _cell(obs, 4, 5, r, c)
    v46, b46 = _cell(obs, 4, 6, r, c)
    s05 = jnp.where(v05, _item_code(b05), mem[M_S05])
    s06 = jnp.where(v06, _item_code(b06), mem[M_S06])
    s45 = jnp.where(v45, _item_code(b45), mem[M_S45])
    s46 = jnp.where(v46, _item_code(b46), mem[M_S46])

    # ---- tasks ----
    T_POT = _task(2, 5, D_LEFT, 1)
    T_WAIT = _task(2, 5, D_LEFT, 0)
    T_DELIVER = _task(2, 6, D_RIGHT, 1)
    T_PLATE_T = _task(1, 5, D_LEFT, 1)
    T_PLATE_B = _task(3, 5, D_LEFT, 1)
    T_ING0 = _task(1, 6, D_RIGHT, 1)
    T_ING1 = _task(3, 6, D_RIGHT, 1)
    T_C05 = _task(1, 5, D_UP, 1)
    T_C06 = _task(1, 6, D_UP, 1)
    T_C45 = _task(3, 5, D_DOWN, 1)
    T_C46 = _task(3, 6, D_DOWN, 1)

    task_plate = _select(r <= 2, T_PLATE_T, T_PLATE_B)

    fetch0 = _select(s05 == 1, T_C05, _select(s06 == 1, T_C06, T_ING0))
    fetch1 = _select(s45 == 2, T_C45, _select(s46 == 2, T_C46, T_ING1))

    def fetch(i):
        return _select(i == 0, fetch0, fetch1)

    stash_ing0 = _select(s05 == 0, T_C05, _select(s06 == 0, T_C06, _select(s45 == 0, T_C45, _select(s46 == 0, T_C46, T_WAIT))))
    stash_ing1 = _select(s45 == 0, T_C45, _select(s46 == 0, T_C46, _select(s05 == 0, T_C05, _select(s06 == 0, T_C06, T_WAIT))))
    stash_ing = _select(ing_type == 0, stash_ing0, stash_ing1)

    stash_dish0 = _select(s06 == 0, T_C06, _select(s05 == 0, T_C05, _select(s46 == 0, T_C46, _select(s45 == 0, T_C45, T_DELIVER))))
    stash_dish1 = _select(s46 == 0, T_C46, _select(s45 == 0, T_C45, _select(s06 == 0, T_C06, _select(s05 == 0, T_C05, T_DELIVER))))
    stash_dish = _select(dish_type == 1, stash_dish1, stash_dish0)

    stash_plate = _select(r <= 2,
                          _select(s05 == 0, T_C05, _select(s06 == 0, T_C06, _select(s45 == 0, T_C45, _select(s46 == 0, T_C46, T_WAIT)))),
                          _select(s45 == 0, T_C45, _select(s46 == 0, T_C46, _select(s05 == 0, T_C05, _select(s06 == 0, T_C06, T_WAIT)))))

    want_dish_code = jnp.where(r_com == 0, 3, jnp.where(r_com == 1, 4, -5))
    has_stash_dish = (s05 == want_dish_code) | (s06 == want_dish_code) | (s45 == want_dish_code) | (s46 == want_dish_code)
    pick_dish = _select(s06 == want_dish_code, T_C06,
                _select(s05 == want_dish_code, T_C05,
                 _select(s46 == want_dish_code, T_C46, T_C45)))

    # Case A: holding a cooked dish
    match = (dish_type == r_com) & (r_com >= 0)
    task_A = _select(match, T_DELIVER, _select(r_com < 0, T_WAIT, stash_dish))

    # Case B: holding an empty plate
    task_B = _select(pot_ready, T_POT, _select(pot_cooking, T_WAIT, stash_plate))

    # Case C: holding a raw ingredient
    can_add_partial = pot_partial & (pot_type == ing_type)
    can_add_empty = pot_empty & (r_com == ing_type)
    task_C = _select(can_add_partial | can_add_empty, T_POT,
                     _select(pot_empty & (r_com < 0), T_WAIT, stash_ing))

    # Case D: empty hands
    task_D = _select(pot_ready, task_plate,
             _select(has_stash_dish, pick_dish,
              _select(pot_cooking, task_plate,
               _select(pot_partial, fetch(pot_type),
                _select(r_opt >= 0, fetch(r_opt), T_WAIT)))))

    task = _select(held_dish, task_A,
            _select(held_plate, task_B,
             _select(held_ing, task_C, task_D)))

    action = _nav(r, c, f, task)

    del_attempt = ((action == A_INTERACT) & held_dish & (r == 2) & (c == 6) & (f == D_RIGHT)).astype(I32)
    step = jnp.minimum(mem[M_STEP] + 1, 100000)
    new_mem = jnp.stack([last_pat, stable, since_del, del_attempt, empty_cnt,
                         s05, s06, s45, s46, step, _i(0), _i(0)]).astype(I32)
    trace = {
        'r_opt': r_opt.astype(I32),
        'r_com': r_com.astype(I32),
        'pattern': jnp.where(sig_vis, pattern, -1).astype(I32),
        'since_del': since_del.astype(I32),
        'task_row': task[0],
        'task_col': task[1],
        'held': jnp.where(held_dish, 3, jnp.where(held_plate, 2, jnp.where(held_ing, 1, 0))).astype(I32),
        'pot_n': pn.astype(I32),
    }
    return action, new_mem, trace
