"""Overcooked V2 test_time_wide policy.

Layout facts used (zero-based (row, col)):
  top room rows 1-2, cols 1-4 (agent_0); bottom room rows 4-5, cols 1-4 (agent_1).
  goal (0,2), plate pile (0,3), pots (3,1) and (3,3), ingredient 0 at (1,0)/(1,5),
  ingredient 1 at (2,0)/(2,5), recipe indicator (6,2) visible only from the bottom room.

Roles:
  agent_0 (top) cooks: keeps pot L=(3,1) filled with ingredient 0 and pot R=(3,3) with
  ingredient 1, scoops the pot matching its recipe estimate, delivers.  The recipe estimate
  comes from a small Bayesian reader over binary partner features (column side, facing,
  shared-counter items, held item) whose mapping is learned from whether the partner's
  signal changes after a delivery (the recipe only changes on a correct delivery).
  agent_1 (bottom) reads the always-on indicator and stands under pot L (4,1) for recipe 0
  or pot R (4,3) for recipe 1, facing the pot.  If it sees the partner scoop a dish that
  does not match the recipe, it inverts its signal polarity.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'signal_reader_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

# actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# facing ids
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
NF = 8          # partner features
NH = 2 * NF     # hypotheses: (feature, mapping)
# prior log weights, per feature (value0 -> recipe 0, value0 -> recipe 1):
# pos (left/right), facing LEFT (no/yes), facing RIGHT, facing UP, facing DOWN,
# item on counter (3,2) (absent/present), item on counter (3,4), partner holding item (no/yes)
PRIOR_LIST = [2.0, 0.0, 0.8, 0.5, 0.5, 0.8, 0.7, 0.7, 0.7, 0.7, 1.0, 0.5, 0.7, 0.3, 0.7, 0.3]
LOG_HALF = -0.6931472
SWITCH_MARGIN = 0.5      # needed to leave the currently followed hypothesis
RETURN_SLACK = 0.3       # default hypothesis regains control when within this of the current one (< SWITCH_MARGIN)


def _face_to_act():
    return jnp.array([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)


def _prior():
    return jnp.array(PRIOR_LIST, dtype=jnp.float32)
LOG_RHO = -12.0
LOG_1MRHO = 0.0


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _find_one(chan):
    flat = chan.reshape(-1)
    idx = jnp.argmax(flat)
    found = flat[idx] > 0
    return idx // 5, idx % 5, found


def _cell(obs, r, c, gr, gc):
    """Observation block of global cell (gr, gc) as seen from (r, c); zeros if not in view."""
    i = gr - r + 2
    j = gc - c + 2
    vis = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    ic = jnp.clip(i, 0, 4)
    jc = jnp.clip(j, 0, 4)
    return jnp.where(vis, obs[ic, jc], 0), vis


def _goto(r, c, facing, tr, tc, fdir, interact):
    """Move to (tr, tc), then face fdir (all targets are blocked in fdir), then act."""
    dr = tr - r
    dc = tc - c
    move_v = jnp.where(dr > 0, A_DOWN, A_UP)
    move_h = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    face_act = _face_to_act()[fdir]
    final = jnp.where(interact, A_INTERACT, A_STAY)
    act = jnp.where(dr != 0, move_v,
                    jnp.where(dc != 0, move_h,
                              jnp.where(facing != fdir, face_act, final)))
    return _i32(act)


# ----------------------------------------------------------------------------- agent 0

def init_agent_0():
    return {
        'logw': _prior(),
        'sig': jnp.zeros((NF,), jnp.float32),
        'seen': jnp.zeros((NF, 2), jnp.int32),
        'cur_h': _i32(0),
        'fresh': _i32(0),
        'feat_last': -jnp.ones((NF,), jnp.int32),
        'feat_del': -jnp.ones((NF,), jnp.int32),
        'd_del': _i32(-1),
        'pending': _i32(0),
        'n_upd': _i32(0),
        'n_credit': _i32(0),
        'pot': jnp.zeros((2, 6), jnp.int32),      # n0 n1 n2 n3 cooked timer
        'bank': jnp.zeros((2, 6), jnp.int32),     # counters (0,1), (0,4): plate cooked n0..n3
        'since_del': _i32(100),
        'last_seen': _i32(100),
        't': _i32(0),
    }


def _pot_state(vec):
    return jnp.stack([vec[34], vec[35], vec[36], vec[37], vec[33], vec[44]]).astype(jnp.int32)


def _advance_pot(p):
    timer = p[5]
    cooked = jnp.where(timer == 1, 1, p[4])
    return p.at[4].set(cooked).at[5].set(jnp.maximum(timer - 1, 0))


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(jnp.int32)

    gi, gj, _ = _find_one(obs[:, :, 23])
    r = 2 - gi
    c = 4 - gj
    me = obs[2, 2]
    facing = jnp.argmax(me[1:5])
    held_plate = me[5] > 0
    held_cooked = me[6] > 0
    hcounts = me[7:11]
    hsum = hcounts.sum()
    held_ing = (~held_plate) & (hsum > 0)
    held_ing_type = jnp.clip(jnp.argmax(hcounts), 0, 1)
    held_dish = held_plate & held_cooked
    held_plate_only = held_plate & (~held_cooked)
    empty = (~held_plate) & (hsum == 0)
    dish_type = jnp.where(hcounts[0] == 3, 0, jnp.where(hcounts[1] == 3, 1, -1))
    dish_good = dish_type >= 0

    # pots (index 0 = L (3,1) <-> ingredient 0, index 1 = R (3,3) <-> ingredient 1)
    vL, visL = _cell(obs, r, c, 3, 1)
    vR, visR = _cell(obs, r, c, 3, 3)
    potL = jnp.where(visL, _pot_state(vL), _advance_pot(mem['pot'][0]))
    potR = jnp.where(visR, _pot_state(vR), _advance_pot(mem['pot'][1]))
    pots = jnp.stack([potL, potR])
    n = pots[:, 0] + pots[:, 1] + pots[:, 2] + pots[:, 3]
    cooked = pots[:, 4] > 0
    cooking = pots[:, 5] > 0
    idle = (~cooked) & (~cooking)
    needs_fill = idle & (n < 3)
    ready_type = jnp.where(pots[:, 0] == 3, 0, jnp.where(pots[:, 1] == 3, 1, -1))
    ready_good = cooked & (ready_type >= 0)
    ready_bad = cooked & (ready_type < 0)

    # dish bank on counters (0,1) [via (1,1) UP] and (0,4) [via (1,4) UP]
    b0, vb0 = _cell(obs, r, c, 0, 1)
    b1, vb1 = _cell(obs, r, c, 0, 4)
    bank0 = jnp.where(vb0, b0[32:38], mem['bank'][0])
    bank1 = jnp.where(vb1, b1[32:38], mem['bank'][1])
    bank = jnp.stack([bank0, bank1])
    bank_empty = bank.sum(axis=1) == 0
    bank_dish_type = jnp.where((bank[:, 0] > 0) & (bank[:, 1] > 0),
                               jnp.where(bank[:, 2] == 3, 0, jnp.where(bank[:, 3] == 3, 1, -1)), -1)

    # partner features
    pi, pj, pvis = _find_one(obs[:, :, 11])
    pcell = obs[pi, pj]
    pcol = c + pj - 2
    f_pos = jnp.where(pcol >= 3, 1, 0)
    f_fl = jnp.where(pcell[15] > 0, 1, 0)
    f_fr = jnp.where(pcell[14] > 0, 1, 0)
    f_fu = jnp.where(pcell[12] > 0, 1, 0)
    f_fd = jnp.where(pcell[13] > 0, 1, 0)
    f_hold = jnp.where(pcell[18:22].sum() > 0, 1, 0)
    c32, v32 = _cell(obs, r, c, 3, 2)
    c34, v34 = _cell(obs, r, c, 3, 4)
    f_c32 = jnp.where((c32[34:38].sum() > 0) & (c32[32] == 0), 1, 0)
    f_c34 = jnp.where((c34[34:38].sum() > 0) & (c34[32] == 0), 1, 0)
    feat_new = jnp.stack([f_pos, f_fl, f_fr, f_fu, f_fd, f_c32, f_c34, f_hold])
    feat_vis = jnp.stack([pvis, pvis, pvis, pvis, pvis, v32, v34, pvis])
    feat_prev = mem['feat_last']
    feat_last = jnp.where(feat_vis, feat_new, feat_prev)
    seen = mem['seen']
    fidx = jnp.arange(NF)
    seen = seen.at[fidx, jnp.clip(feat_new, 0, 1)].max(feat_vis.astype(jnp.int32))

    # timing evidence: a signal feature changes only shortly after a delivery
    since_del = mem['since_del']
    d_del = mem['d_del']
    chg_step = (feat_prev >= 0) & (feat_last >= 0) & (feat_prev != feat_last) & (d_del >= 0)
    in_win = since_del <= 14
    loss = jnp.where(chg_step & (~in_win), -1.5, 0.0)
    sig_pre = jnp.clip(mem['sig'] + loss, -30.0, 30.0)
    # net change between consecutive deliveries is credited at the delivery update below
    sig = sig_pre
    last_seen = jnp.where(pvis, 0, mem['last_seen'] + 1)
    fresh = ((mem['fresh'] > 0) | (pvis & (since_del >= 4)) | (since_del >= 14)) \
        & ((last_seen <= 6) | (last_seen >= 20))

    # recipe estimate: hypothesis with the best weight among features that have shown a
    # post-delivery change (the default position hypotheses are always eligible), with
    # hysteresis for leaving the default and none for returning to it.
    logw = mem['logw']
    hf = jnp.arange(NH) // 2
    hm = jnp.arange(NH) % 2
    hidx = jnp.arange(NH)
    elig = ((seen[hf, 0] > 0) & (seen[hf, 1] > 0) & (sig[hf] > 0.0)) | (hidx <= 1)
    score = jnp.where(elig, logw + sig[hf], -1000.0)
    best = jnp.argmax(score)
    cur = mem['cur_h']
    leave_ok = score[best] > score[cur] + SWITCH_MARGIN
    return_ok = (cur != 0) & (score[0] >= score[cur] - RETURN_SLACK)
    hstar = jnp.where(leave_ok, best, jnp.where(return_ok, 0, cur))
    hstar = jnp.where(elig[hstar], hstar, 0)
    fstar = hstar // 2
    mstar = hstar % 2
    vstar = feat_last[fstar]
    rho = jnp.where(vstar < 0, 0, vstar) ^ mstar
    # no feature has ever changed after several evaluated deliveries: alternate dish types
    nosig = (mem['n_upd'] >= 4) & (mem['n_credit'] == 0) & (d_del >= 0)
    rho = jnp.where(nosig, 1 - d_del, rho)
    other = 1 - rho

    # ------------------------------------------------------------------ decision
    # targets: (row, col, facing, interact)
    T_ING = jnp.array([[1, 1, F_LEFT], [2, 4, F_RIGHT]], dtype=jnp.int32)
    T_POT = jnp.array([[2, 1, F_DOWN], [2, 3, F_DOWN]], dtype=jnp.int32)
    T_BANK = jnp.array([[1, 1, F_UP], [1, 4, F_UP]], dtype=jnp.int32)
    T_PLATE = jnp.array([1, 3, F_UP], dtype=jnp.int32)
    T_DELIVER = jnp.array([1, 2, F_UP], dtype=jnp.int32)
    T_WAIT = jnp.array([2, 2, F_DOWN], dtype=jnp.int32)

    # ingredient pickup spots per (pot, type): pot L uses the left supplies, pot R the right ones
    T_TAKE = jnp.array([[[1, 1, F_LEFT], [2, 1, F_LEFT]],
                        [[1, 4, F_RIGHT], [2, 4, F_RIGHT]]], dtype=jnp.int32)

    # pot content type: 0/1 pure, -1 empty, -2 garbage
    ctype = jnp.where(n == 0, -1,
                      jnp.where((pots[:, 0] > 0) & (pots[:, 1] == 0) & (pots[:, 2] + pots[:, 3] == 0), 0,
                                jnp.where((pots[:, 1] > 0) & (pots[:, 0] == 0) & (pots[:, 2] + pots[:, 3] == 0), 1, -2)))
    garbage = ctype == -2
    bank_has_rho = (bank_dish_type == rho)
    bank_rho_idx = jnp.argmax(bank_has_rho)
    free_slot = bank_empty.any()
    free_idx = jnp.argmax(bank_empty)
    stock0 = (bank_dish_type == 0).sum() + (ctype == 0).sum()
    stock1 = (bank_dish_type == 1).sum() + (ctype == 1).sum()
    stock = jnp.stack([stock0, stock1])
    # desired type for an empty pot: keep both types in stock, otherwise the current recipe
    # (pot R prefers type 1 when both are stocked because its route for type 0 is long)
    want_rho_first = stock[rho] == 0
    want_other = (~want_rho_first) & (stock[other] == 0)
    desired = jnp.where(want_rho_first, rho, jnp.where(want_other, other, jnp.array([rho, 1])))
    fill_type = jnp.where(ctype >= 0, ctype, desired)
    fill_type = jnp.clip(fill_type, 0, 1)
    # fill priority: partially filled current-type first, then by count
    fill_score = jnp.where(needs_fill, 10 + 5 * (fill_type == rho) + n, -1)
    any_needs = needs_fill.any()
    fill_idx = jnp.argmax(fill_score)
    T_FILL = T_TAKE[fill_idx, fill_type[fill_idx]]

    rho_cooked = cooked & (ctype == rho)
    rho_cooking = cooking & (ctype == rho)
    any_cooking = cooking.any()
    timer_masked = jnp.where(rho_cooking, pots[:, 5], 999)
    rho_cook_idx = jnp.argmin(timer_masked)
    rho_cook_timer = timer_masked[rho_cook_idx]
    cooking_idx = jnp.argmin(jnp.where(cooking, pots[:, 5], 999))
    # a ready pot whose dish we can park: other-type good dish with a free bank slot, or bad dish
    parkable = cooked & (garbage | ((ctype != rho) & free_slot))

    # --- empty hands
    e_conds = [
        (~fresh) & any_needs,
        ~fresh,
        bank_has_rho.any(),
        rho_cooked.any(),
        any_needs,
        parkable.any(),
        rho_cooking.any() & (rho_cook_timer <= 7),
        any_cooking,
    ]
    e_targets = [
        T_FILL,
        T_WAIT,
        T_BANK[bank_rho_idx],
        T_PLATE,
        T_FILL,
        T_PLATE,
        T_PLATE,
        T_PLATE,
    ]
    e_target = jnp.select(e_conds, e_targets, T_WAIT)
    e_inter = jnp.select(e_conds, [True, False] + [True] * 6, False)

    # --- holding plate only
    p_conds = [
        ~fresh,
        rho_cooked.any(),
        parkable.any(),
        rho_cooking.any(),
        any_cooking,
        free_slot,
    ]
    p_targets = [
        T_WAIT,
        T_POT[jnp.argmax(rho_cooked)],
        T_POT[jnp.argmax(parkable)],
        T_POT[rho_cook_idx],
        T_POT[cooking_idx],
        T_BANK[free_idx],
    ]
    p_inter_l = [False, True, True, False, False, True]
    p_target = jnp.select(p_conds, p_targets, T_WAIT)
    p_inter = jnp.select(p_conds, p_inter_l, False)

    # --- holding dish
    T_DUMP = jnp.array([[2, 4, F_DOWN], [2, 2, F_DOWN]], dtype=jnp.int32)
    dump34_free = c34[32:38].sum() == 0
    dump32_free = c32[32:38].sum() == 0
    d_conds = [
        ~fresh,
        dish_good & (dish_type == rho),
        free_slot,
        (~dish_good) & dump34_free,
        (~dish_good) & dump32_free,
        ~dish_good,
    ]
    d_targets = [T_WAIT, T_DELIVER, T_BANK[free_idx], T_DUMP[0], T_DUMP[1], T_DELIVER]
    d_target = jnp.select(d_conds, d_targets, T_WAIT)
    d_inter = jnp.select(d_conds, [False] + [True] * 5, False)

    # --- holding ingredient: put it in a pot that takes it (same type, empty, or garbage)
    x = held_ing_type
    acc = needs_fill & ((ctype == x) | (ctype == -1) | garbage)
    acc_score = jnp.where(acc, 10 + 5 * (ctype == x) + 2 * (desired == x) + n, -1)
    i_pot = jnp.argmax(acc_score)
    i_accepts = acc.any()
    i_target = jnp.where(i_accepts, T_POT[i_pot], T_WAIT)
    i_inter = i_accepts

    target = jnp.where(empty, e_target,
                       jnp.where(held_plate_only, p_target,
                                 jnp.where(held_dish, d_target, i_target)))
    inter = jnp.where(empty, e_inter,
                      jnp.where(held_plate_only, p_inter,
                                jnp.where(held_dish, d_inter, i_inter)))
    action = _goto(r, c, facing, target[0], target[1], target[2], inter)

    # ------------------------------------------------------------------ learning
    delivering = held_dish & dish_good & (r == 1) & (c == 2) & (facing == F_UP) & (action == A_INTERACT) \
        & (target[0] == 1) & (target[1] == 2)
    feat_del = mem['feat_del']
    pending = mem['pending'] > 0
    settled = fresh & (since_del >= 8)
    do_update = pending & (d_del >= 0) & (settled | delivering)
    known = (feat_del >= 0) & (feat_last >= 0)
    changed = known & (feat_del != feat_last)
    vprev = feat_del[hf]
    pred = jnp.where(vprev < 0, 0, vprev) ^ hm
    correct = pred == d_del
    own = jnp.where(~known[hf], 0.0,
                    jnp.where(correct, LOG_HALF, jnp.where(changed[hf], LOG_RHO, LOG_1MRHO)))
    logw_new = jnp.where(do_update, logw + own, logw)
    logw_new = jnp.maximum(logw_new - jnp.max(logw_new), -60.0)
    sig_new = jnp.clip(jnp.where(do_update & changed, sig + 2.5, sig), -30.0, 30.0)
    feat_del_new = jnp.where(delivering, feat_last, feat_del)
    d_del_new = jnp.where(delivering, dish_type, d_del)

    new_mem = {
        'logw': logw_new.astype(jnp.float32),
        'sig': sig_new.astype(jnp.float32),
        'seen': seen.astype(jnp.int32),
        'cur_h': _i32(hstar),
        'fresh': _i32(jnp.where(delivering, 0, fresh.astype(jnp.int32))),
        'feat_last': feat_last.astype(jnp.int32),
        'feat_del': feat_del_new.astype(jnp.int32),
        'd_del': _i32(d_del_new),
        'pending': _i32(jnp.where(delivering, 1, jnp.where(do_update, 0, mem['pending']))),
        'n_upd': _i32(mem['n_upd'] + do_update.astype(jnp.int32)),
        'n_credit': _i32(mem['n_credit'] + (do_update & changed.any()).astype(jnp.int32)),
        'pot': pots.astype(jnp.int32),
        'bank': bank.astype(jnp.int32),
        'since_del': _i32(jnp.where(delivering, 0, jnp.minimum(mem['since_del'] + 1, 1000))),
        'last_seen': _i32(jnp.minimum(last_seen, 1000)),
        't': _i32(mem['t'] + 1),
    }
    trace = {
        'rho': _i32(rho),
        'hstar': _i32(jnp.where(nosig, 99, hstar)),
        'row': _i32(r), 'col': _i32(c),
        'pvis': pvis,
        'fresh': fresh,
        'sig0': sig[0],
        'f_pos': _i32(feat_last[0]),
        'delivering': delivering,
        'target_r': _i32(target[0]), 'target_c': _i32(target[1]),
        'held': _i32(jnp.where(held_dish, 3 + dish_type, jnp.where(held_plate, 2, jnp.where(held_ing, 10 + held_ing_type, 0)))),
        'potL': _i32(ctype[0] * 100 + pots[0, 4] * 50 + pots[0, 5]),
        'potR': _i32(ctype[1] * 100 + pots[1, 4] * 50 + pots[1, 5]),
        'bank': _i32(bank_dish_type[0] * 10 + bank_dish_type[1]),
    }
    return action, new_mem, trace


# ----------------------------------------------------------------------------- agent 1

def init_agent_1():
    return {
        'polarity': _i32(0),
        'had_dish': _i32(0),
        'settled': _i32(0),
        'cooldown': _i32(0),
        'last_target': _i32(-1),
        'flips': _i32(0),
        'wcount': _i32(0),
        'last_recipe': _i32(-1),
    }


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(jnp.int32)

    ri, rj, rfound = _find_one(obs[:, :, 25])
    r = 8 - ri
    c = 4 - rj
    rcell = obs[ri, rj]
    recipe = jnp.where(rcell[41] == 3, 1, 0)
    me = obs[2, 2]
    facing = jnp.argmax(me[1:5])

    sig = recipe ^ mem['polarity']
    tcol = jnp.where(sig == 0, 1, 3)
    action = _goto(r, c, facing, 4, tcol, F_UP, False)

    at_target = (r == 4) & (c == tcol) & (facing == F_UP)
    settled = jnp.where(at_target & (mem['last_target'] == tcol), mem['settled'] + 1, 0)

    # wrong-scoop detection
    pi, pj, pvis = _find_one(obs[:, :, 11])
    pcell = obs[pi, pj]
    has_dish = pvis & (pcell[16] > 0) & (pcell[17] > 0)
    d = jnp.where(pcell[18] == 3, 0, jnp.where(pcell[19] == 3, 1, -1))
    event = has_dish & (mem['had_dish'] == 0)
    recipe_changed = (mem['last_recipe'] >= 0) & (mem['last_recipe'] != recipe)
    wrong_scoop = event & (d >= 0) & (d != recipe) & (settled >= 3)
    right_scoop = event & (d == recipe)
    wcount = jnp.where(right_scoop | recipe_changed, 0, mem['wcount'] + wrong_scoop.astype(jnp.int32))
    wrong = (wcount >= 3) & (mem['cooldown'] == 0) & (mem['flips'] < 2)
    wcount = jnp.where(wrong, 0, wcount)
    polarity = jnp.where(wrong, 1 - mem['polarity'], mem['polarity'])
    cooldown = jnp.where(wrong, 25, jnp.maximum(mem['cooldown'] - 1, 0))

    new_mem = {
        'polarity': _i32(polarity),
        'had_dish': _i32(has_dish),
        'settled': _i32(jnp.minimum(settled, 1000)),
        'cooldown': _i32(cooldown),
        'last_target': _i32(tcol),
        'flips': _i32(mem['flips'] + wrong.astype(jnp.int32)),
        'wcount': _i32(wcount),
        'last_recipe': _i32(recipe),
    }
    trace = {
        'recipe': _i32(recipe), 'polarity': _i32(polarity), 'row': _i32(r), 'col': _i32(c),
        'wrong': wrong, 'rfound': rfound,
    }
    return action, new_mem, trace
