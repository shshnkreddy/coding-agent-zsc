"""Overcooked V2 test_time_simple: counter-marker recipe signalling + solo cook.

Seat 0 (left room) sees the always-on recipe indicator but cannot reach the pot.
Seat 1 (right room) can run the whole kitchen loop but cannot see the recipe.
The only shared observable is the three counters at column 3 ((1,3),(2,3),(3,3)),
which seat 0 can load with ingredient-2 markers and seat 1 can see from column 5.

Encoding used by seat 0 (chosen to be legible to several natural decoders):
  recipe [0,0,0] -> marker on top counter (1,3) only            (row of ingredient-0 supply)
  recipe [1,1,1] -> markers on middle (2,3) and bottom (3,3)     (row of ingredient-1 supply)
Seat 1 decodes: {top} -> 0 ; {bottom}, {middle}, {middle,bottom} -> 1 ; anything else is
treated as "in transition" and not trusted until it settles.  Mismatched dishes are stored
on the outer counters and delivered later when the recipe matches.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'marker_signal_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

I32 = jnp.int32
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

POST_DELIVERY_WAIT = 1      # steps to linger at the pot after a delivery before fetching
FLIP_WINDOW = 8             # steps after own delivery during which a counter change implies recipe flip
BLIND_COOK_STEP = 20        # start cooking type 0 if no readable signal by then
SIG_IDLE_STEPS = 20         # signaller: pot idle this long with a complete recipe-1 signal -> cook cannot read it
SIG_WRONG_STABLE = 10       # signaller: signal complete this long before cook adds ingredient 0 -> cook misreads it
SIG_MAX_SWITCH = 4          # signaller: maximum encoding switches per episode
FALLBACK_MAX_DELIVERIES = 2 # cook: max deliveries on the empty-counter fallback without any signal change
EMPTY_FALLBACK = 40         # steps of persistently empty counters before assuming recipe 0

def _tables():
    """Static lookup tables (built inside a function; module-scope calls are disallowed)."""
    return {
        # Cook (right room) outer counters: cell, stand cell, facing action
        'OUT_R': jnp.asarray([0, 0, 4, 4], I32),
        'OUT_C': jnp.asarray([5, 6, 5, 6], I32),
        'OUT_SR': jnp.asarray([1, 1, 3, 3], I32),
        'OUT_SC': jnp.asarray([5, 6, 5, 6], I32),
        'OUT_F': jnp.asarray([UP, UP, DOWN, DOWN], I32),
        # Signaller (left room) outer walls: cell, stand cell, facing action
        'LOUT_R': jnp.asarray([0, 1, 3, 4], I32),
        'LOUT_C': jnp.asarray([1, 0, 0, 1], I32),
        'LOUT_SR': jnp.asarray([1, 1, 3, 3], I32),
        'LOUT_SC': jnp.asarray([1, 1, 1, 1], I32),
        'LOUT_F': jnp.asarray([UP, LEFT, LEFT, DOWN], I32),
    }


def _init():
    return {
        't': jnp.asarray(0, I32),
        'est': jnp.asarray(-1, I32),
        'since': jnp.asarray(0, I32),
        'bits': jnp.zeros((3,), I32),
        'seen': jnp.asarray(0, I32),
        'fresh': jnp.asarray(0, I32),
        'empty': jnp.asarray(0, I32),
        'flipw': jnp.asarray(0, I32),
        'outer': jnp.zeros((4,), I32),
        'pd': jnp.asarray(0, I32),
        'fbmode': jnp.asarray(0, I32),
        'fbdeliv': jnp.asarray(0, I32),
    }


def _i(x):
    return jnp.asarray(x, I32)


def _read(obs, r, c, R, C):
    """Return the 40-channel vector of global cell (R, C) as seen from (r, c) and a validity flag."""
    lr = R - r + 2
    lc = C - c + 2
    valid = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    cell = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), :]
    return jnp.where(valid, cell, 0), valid


def _item_code(block):
    """0 empty, 1 raw0, 2 raw1, 3 raw2, 4 plate, 5 dish0, 6 dish1, 7 other dish."""
    plate, cooked, n0, n1, n2 = block[0], block[1], block[2], block[3], block[4]
    tot = n0 + n1 + n2
    is_dish = cooked > 0
    raw = (~is_dish) & (plate == 0)
    code = jnp.where(raw & (tot == 0), 0,
           jnp.where(raw & (n0 > 0), 1,
           jnp.where(raw & (n1 > 0), 2,
           jnp.where(raw & (n2 > 0), 3,
           jnp.where((~is_dish) & (plate > 0), 4,
           jnp.where(is_dish & (n0 == 3) & (n1 == 0) & (n2 == 0), 5,
           jnp.where(is_dish & (n1 == 3) & (n0 == 0) & (n2 == 0), 6, 7)))))))
    return code.astype(I32)


def _facing(me):
    up, down, right, left = me[1], me[2], me[3], me[4]
    return jnp.where(up > 0, UP, jnp.where(down > 0, DOWN, jnp.where(right > 0, RIGHT, LEFT))).astype(I32)


def _locate(obs, channel):
    flat = obs[:, :, channel].reshape(-1)
    idx = jnp.argmax(flat)
    found = flat[idx] > 0
    return (idx // 5).astype(I32), (idx % 5).astype(I32), found


def _sel(cond, a, b):
    return tuple(jnp.where(cond, x, y).astype(I32) for x, y in zip(a, b))


def _goal(r, c, f, mode):
    return (_i(r), _i(c), _i(f), _i(mode))


def _navigate(r, c, face, goal):
    gr, gc, gf, mode = goal
    dr = gr - r
    dc = gc - c
    at = (dr == 0) & (dc == 0)
    move_v = jnp.where(dr > 0, DOWN, UP)
    move_h = jnp.where(dc > 0, RIGHT, LEFT)
    move = jnp.where(dr != 0, move_v, move_h)
    final = jnp.where(mode == 0, INTERACT, STAY)
    return jnp.where(~at, move, jnp.where(face != gf, gf, final)).astype(I32)


def _nearest(r, c, mask, stand_r, stand_c):
    dist = jnp.abs(stand_r - r) + jnp.abs(stand_c - c)
    dist = jnp.where(mask, dist, 99)
    idx = jnp.argmin(dist)
    ok = dist[idx] < 99
    return idx.astype(I32), ok


def _occupied(obs, r, c, R, C):
    cell, valid = _read(obs, r, c, R, C)
    return (jnp.sum(cell[29:34]) > 0) & valid


def _update_outer(obs, r, c, outer, OR, OC):
    new = []
    for k in range(4):
        cell, valid = _read(obs, r, c, OR[k], OC[k])
        code = _item_code(cell[29:34])
        new.append(jnp.where(valid, code, outer[k]))
    return jnp.stack(new).astype(I32)


# --------------------------------------------------------------------------------------
# Seat 1: cook (right room)
# --------------------------------------------------------------------------------------

def _cook(obs, mem, r, c, face):
    T = _tables()
    OUT_R, OUT_C, OUT_SR, OUT_SC, OUT_F = T['OUT_R'], T['OUT_C'], T['OUT_SR'], T['OUT_SC'], T['OUT_F']
    me = obs[2, 2, :]
    hold = _item_code(me[5:10])
    t = mem['t']

    pot, _ = _read(obs, r, c, 2, 4)
    pot_cooked = pot[30] > 0
    pot_timer = pot[39]
    pn0, pn1, pn2 = pot[31], pot[32], pot[33]
    pot_tot = pn0 + pn1 + pn2
    pot_cooking = pot_timer > 0
    pot_idle = (~pot_cooked) & (~pot_cooking) & (pot_tot < 3)
    pot_empty = pot_tot == 0
    pot_type = jnp.where(pot_empty, -1, jnp.where(pn0 == pot_tot, 0, jnp.where(pn1 == pot_tot, 1, jnp.where(pn2 == pot_tot, 2, -2))))

    # ---- signal decoding ----
    visible = c == 5
    bits_now = jnp.stack([_occupied(obs, r, c, 1, 3), _occupied(obs, r, c, 2, 3), _occupied(obs, r, c, 3, 3)]).astype(I32)
    bits_prev = mem['bits']
    seen_prev = mem['seen'] > 0
    bits = jnp.where(visible, bits_now, bits_prev)
    changed = visible & seen_prev & jnp.any(bits != bits_prev)
    newly = visible & seen_prev & (bits > bits_prev)
    any_new = jnp.any(newly)
    new_meaning = jnp.where(newly[0], 0, 1).astype(I32)
    since = jnp.where(changed, 0, jnp.minimum(mem['since'] + 1, 1000)).astype(I32)
    seen = jnp.maximum(mem['seen'], visible.astype(I32))
    tb, mb, bb = bits[0] > 0, bits[1] > 0, bits[2] > 0
    pat_est = jnp.where(tb & ~mb & ~bb, 0, jnp.where(~tb & (mb | bb), 1, -2))
    is_empty = ~tb & ~mb & ~bb
    est_prev = mem['est']
    fresh_prev = mem['fresh'] > 0
    flipw = mem['flipw']
    # any counter change shortly after our own delivery (that is not a new placement) implies the recipe flipped
    do_flip = changed & (~any_new) & (flipw > 0) & (est_prev >= 0)
    settle_need = jnp.where(fresh_prev, 8, 3)
    settled = (pat_est >= 0) & (since >= settle_need)
    empty_cnt = jnp.where(is_empty & (seen > 0), jnp.minimum(mem['empty'] + 1, 1000), 0).astype(I32)
    empty_fb = is_empty & (empty_cnt >= EMPTY_FALLBACK) & (est_prev == -1)
    est = jnp.where(any_new, new_meaning,
          jnp.where(do_flip, 1 - est_prev,
          jnp.where(settled, pat_est,
          jnp.where(empty_fb, 0, est_prev)))).astype(I32)
    # a change that leaves an unambiguous pattern agreeing with the estimate keeps trust
    fresh = jnp.where(any_new | do_flip | empty_fb, True, jnp.where(changed, pat_est == est, fresh_prev))
    fresh = jnp.where(settled & (~any_new) & (~do_flip), False, fresh)
    # fallback mode: estimate came from the empty-counter fallback and nothing has changed since.
    fbmode = jnp.where(empty_fb, True, jnp.where(changed, False, mem['fbmode'] > 0))
    fb_deliv = jnp.where(changed, 0, mem['fbdeliv']).astype(I32)
    trusted = (est >= 0) & (fresh | (since >= 3)) & (~(fbmode & (fb_deliv >= FALLBACK_MAX_DELIVERIES)))
    # the flip window closes on the first counter change after a delivery, whatever it was
    flipw_dec = jnp.where(changed, 0, jnp.maximum(flipw - 1, 0)).astype(I32)

    outer = _update_outer(obs, r, c, mem['outer'], OUT_R, OUT_C)
    pd = mem['pd']

    def outer_goal(idx):
        return (OUT_SR[idx], OUT_SC[idx], OUT_F[idx], _i(0))

    def nearest_outer(mask):
        return _nearest(r, c, mask, OUT_SR, OUT_SC)

    def supply_goal(x):
        return _sel(x == 0, _goal(1, 6, RIGHT, 0), _goal(3, 6, RIGHT, 0))

    def fetch_goal(x):
        idx, ok = nearest_outer(outer == (1 + x))
        return _sel(ok, outer_goal(idx), supply_goal(x))

    def store_goal():
        idx, ok = nearest_outer(outer == 0)
        return _sel(ok, outer_goal(idx), _goal(2, 5, LEFT, 1))

    G_POT_I = _goal(2, 5, LEFT, 0)
    G_POT_W = _goal(2, 5, LEFT, 1)
    G_DELIVER = _goal(2, 6, RIGHT, 0)
    G_PLATE = _sel(r <= 2, _goal(1, 5, LEFT, 0), _goal(3, 5, LEFT, 0))

    have_dish0 = jnp.any(outer == 5)
    have_dish1 = jnp.any(outer == 6)
    blind = (est == -1) & (t >= BLIND_COOK_STEP)
    blind_type = jnp.where(~have_dish0, 0, jnp.where(~have_dish1, 1, -1))
    cook_type = jnp.where(trusted, est, jnp.where(blind, blind_type, -1)).astype(I32)

    # ---- empty hands ----
    goal = G_POT_W
    # d. pot empty -> fetch cook type after post-delivery grace
    fetch_ok = (cook_type >= 0) & (pd == 0)
    goal = _sel(pot_empty & pot_idle & fetch_ok, fetch_goal(cook_type), goal)
    # c. pot partially filled -> complete it with its own type
    fill_type = jnp.where((pot_type == 0) | (pot_type == 1), pot_type, 0)
    goal = _sel(pot_idle & ~pot_empty, fetch_goal(fill_type), goal)
    # b. pot cooking or cooked -> get a plate
    goal = _sel(pot_cooking | pot_cooked, G_PLATE, goal)
    # a. stored dish matching trusted recipe -> retrieve it
    sidx, sok = nearest_outer(outer == (5 + est))
    goal = _sel(trusted & sok & ((~pot_cooked) | (pot_type != est)), outer_goal(sidx), goal)
    goal_empty = goal

    # ---- holding raw ingredient X (codes 1,2,3) ----
    x = hold - 1
    pot_has_x = pot_idle & (~pot_empty) & ((pot_type == x) | (pot_type == -2))
    pot_other = pot_idle & (~pot_empty) & ~pot_has_x
    can_pot_new = pot_idle & pot_empty & (cook_type == x)
    wait_new = pot_idle & pot_empty & (cook_type == -1) & (~trusted)
    goal_raw = _sel(pot_has_x | can_pot_new, G_POT_I, store_goal())
    goal_raw = _sel(wait_new, G_POT_W, goal_raw)
    goal_raw = _sel(hold == 3, store_goal(), goal_raw)

    # ---- holding plate ----
    goal_plate = _sel(pot_cooked, G_POT_I, _sel(pot_cooking, G_POT_W, store_goal()))

    # ---- holding dish ----
    dish_type = hold - 5
    deliver_now = (hold == 5) | (hold == 6)
    goal_dish = _sel(trusted & (est == dish_type), G_DELIVER,
                     _sel(trusted | (hold == 7), store_goal(), G_POT_W))

    goal = _sel(hold == 0, goal_empty,
           _sel((hold >= 1) & (hold <= 3), goal_raw,
           _sel(hold == 4, goal_plate, goal_dish)))

    action = _navigate(r, c, face, goal)
    delivering = deliver_now & (action == INTERACT) & (goal[0] == 2) & (goal[1] == 6) & (goal[2] == RIGHT)
    pd_next = jnp.where(delivering, POST_DELIVERY_WAIT, jnp.maximum(pd - 1, 0)).astype(I32)

    new_mem = {
        't': (t + 1).astype(I32),
        'est': est,
        'since': since,
        'bits': bits,
        'seen': seen.astype(I32),
        'fresh': fresh.astype(I32),
        'empty': empty_cnt,
        'flipw': jnp.where(delivering, FLIP_WINDOW, flipw_dec).astype(I32),
        'outer': outer,
        'pd': pd_next,
        'fbmode': fbmode.astype(I32),
        'fbdeliv': (fb_deliv + (delivering & fbmode).astype(I32)).astype(I32),
    }
    trace = {
        'role': _i(1), 'r': r, 'c': c, 'hold': hold, 'est': est, 'trusted': trusted.astype(I32),
        'goal_r': goal[0], 'goal_c': goal[1], 'pot_tot': pot_tot.astype(I32), 'pot_cooked': pot_cooked.astype(I32),
        'bits': bits, 'outer': outer, 'action': action,
    }
    return action, new_mem, trace


# --------------------------------------------------------------------------------------
# Seat 0: signaller (left room)
# --------------------------------------------------------------------------------------

def _signaller(obs, mem, r, c, face):
    T = _tables()
    LOUT_R, LOUT_C, LOUT_SR, LOUT_SC, LOUT_F = T['LOUT_R'], T['LOUT_C'], T['LOUT_SR'], T['LOUT_SC'], T['LOUT_F']
    me = obs[2, 2, :]
    hold = _item_code(me[5:10])
    holding = hold != 0
    t = mem['t']

    rcell, _ = _read(obs, r, c, 2, 0)
    rn0, rn1 = rcell[36], rcell[37]
    rec = jnp.where(rn0 == 3, 0, jnp.where(rn1 == 3, 1, -1))
    # Adaptive recipe-1 encoding.  mode 0: {middle,bottom}; mode 1: {bottom}; mode 2: {middle}.
    # Recipe 0 is always {top}.  Switch mode when the cook clearly cannot read the signal
    # (pot stays empty for a long time while the signal is complete) or misreads it
    # (adds ingredient 0 to an empty pot while recipe 1 has been signalled for a while).
    mode = mem['pd']
    desired = jnp.stack([rec == 0,
                         (rec == 1) & (mode != 1),
                         (rec == 1) & (mode != 2)])
    cur = jnp.stack([_occupied(obs, r, c, 1, 3), _occupied(obs, r, c, 2, 3), _occupied(obs, r, c, 3, 3)])
    missing = desired & ~cur
    extra = cur & ~desired
    total = jnp.sum(cur.astype(I32)) + holding.astype(I32)

    outer = _update_outer(obs, r, c, mem['outer'], LOUT_R, LOUT_C)

    pot, pot_vis = _read(obs, r, c, 2, 4)
    pot_n0 = jnp.where(pot_vis, pot[31], mem['seen'] % 4)
    pot_tot = jnp.where(pot_vis, pot[31] + pot[32] + pot[33], mem['seen'] // 4)
    pot_busy = (pot_tot > 0) | (pot[30] > 0) | (pot[39] > 0)
    prev_n0 = mem['seen'] % 4
    complete = jnp.all(cur == desired)
    stable = jnp.where(complete, jnp.minimum(mem['since'] + 1, 1000), 0).astype(I32)
    pot_idle_steps = jnp.where(pot_vis & (~pot_busy), jnp.minimum(mem['empty'] + 1, 1000),
                               jnp.where(pot_vis, 0, mem['empty'])).astype(I32)
    idle_evidence = (rec == 1) & complete & (stable >= SIG_IDLE_STEPS) & (pot_idle_steps >= SIG_IDLE_STEPS)
    prev_tot = mem['seen'] // 4
    wrong_evidence = (rec == 1) & pot_vis & (prev_tot == 0) & (pot_n0 > prev_n0) & (mem['since'] >= SIG_WRONG_STABLE)
    nswitch = mem['flipw']
    switch = (idle_evidence | wrong_evidence) & (nswitch < SIG_MAX_SWITCH)
    mode_next = jnp.where(switch, (mode + 1) % 3, mode).astype(I32)
    stable = jnp.where(switch, 0, stable)
    pot_idle_steps = jnp.where(switch, 0, pot_idle_steps)

    ctr_rows = jnp.asarray([1, 2, 3], I32)
    ctr_cols = jnp.asarray([2, 2, 2], I32)

    def counter_goal(mask):
        idx, ok = _nearest(r, c, mask, ctr_rows, ctr_cols)
        return (ctr_rows[idx], _i(2), _i(RIGHT), _i(0)), ok

    def outer_goal(idx):
        return (LOUT_SR[idx], LOUT_SC[idx], LOUT_F[idx], _i(0))

    G_IDLE = _goal(2, 2, RIGHT, 1)
    # sources: supplies (0,2) via (1,2) UP and (4,2) via (3,2) DOWN, or outer walls holding an item
    sup_goal = _sel(r <= 2, _goal(1, 2, UP, 0), _goal(3, 2, DOWN, 0))
    oidx, ook = _nearest(r, c, outer != 0, LOUT_SR, LOUT_SC)
    fetch_goal = _sel(ook, outer_goal(oidx), sup_goal)
    didx, dok = _nearest(r, c, outer == 0, LOUT_SR, LOUT_SC)
    dump_goal = _sel(dok, outer_goal(didx), G_IDLE)

    place_goal, any_missing = counter_goal(missing)
    pick_goal, any_extra = counter_goal(extra)

    goal_hold = _sel(any_missing, place_goal, _sel(any_extra | (total > 2), dump_goal, G_IDLE))
    goal_empty = _sel(any_extra, pick_goal, _sel(any_missing | (total < 2), fetch_goal, G_IDLE))
    goal = _sel(holding, goal_hold, goal_empty)

    action = _navigate(r, c, face, goal)
    new_mem = dict(mem)
    new_mem['t'] = (t + 1).astype(I32)
    new_mem['outer'] = outer
    new_mem['est'] = rec.astype(I32)
    new_mem['pd'] = mode_next
    new_mem['since'] = stable
    new_mem['empty'] = pot_idle_steps
    new_mem['flipw'] = (nswitch + switch.astype(I32)).astype(I32)
    new_mem['seen'] = (pot_n0 + 4 * pot_tot).astype(I32)
    trace = {
        'role': _i(0), 'r': r, 'c': c, 'hold': hold, 'est': rec.astype(I32), 'trusted': _i(0),
        'goal_r': goal[0], 'goal_c': goal[1], 'pot_tot': mode_next, 'pot_cooked': stable,
        'bits': cur.astype(I32), 'outer': outer, 'action': action,
    }
    return action, new_mem, trace


# --------------------------------------------------------------------------------------
# Unified act
# --------------------------------------------------------------------------------------

def _act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, I32)
    init = _init()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    me = obs[2, 2, :]
    face = _facing(me)

    lr_r, lc_r, found_r = _locate(obs, 23)     # recipe indicator at global (2,0)
    lr_p, lc_p, found_p = _locate(obs, 22)     # pot at global (2,4)
    is_left = found_r
    r = jnp.where(is_left, 4 - lr_r, 4 - lr_p).astype(I32)
    c = jnp.where(is_left, 2 - lc_r, 6 - lc_p).astype(I32)

    a0, m0, tr0 = _signaller(obs, mem, r, c, face)
    a1, m1, tr1 = _cook(obs, mem, r, c, face)
    action = jnp.where(is_left, a0, a1).astype(I32)
    new_mem = jax.tree_util.tree_map(lambda a, b: jnp.where(is_left, a, b).astype(I32), m0, m1)
    trace = {k: jnp.where(is_left, tr0[k], tr1[k]).astype(I32) for k in tr1}
    return action, new_mem, trace


def init_agent_0():
    return _init()


def agent_0_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)


def init_agent_1():
    return _init()


def agent_1_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)
