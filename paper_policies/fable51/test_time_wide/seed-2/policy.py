"""Overcooked V2 test_time_wide policy.

Layout facts (see context/environment/README.md):
  * agent_0 lives in the top room (rows 1-2, cols 1-4) and is the only seat that
    can reach ingredients 0/1, plates, pots (from above) and the delivery station.
  * agent_1 lives in the bottom room (rows 4-5, cols 1-4) and always sees the
    always-on recipe indicator at (6,2); agent_0 never can.
  * Recipes are [0,0,0] or [1,1,1]; ingredient 3 (bottom room) is never needed.

Roles:
  * agent_0: solo cook.  Pot (3,1) <- ingredient 0, pot (3,3) <- ingredient 1,
    keep both pots loaded, fetch plate, choose which dish to deliver from a
    belief over the recipe, deliver, refill.  The recipe belief is a small
    Bayesian filter over hypotheses "partner feature phi (with polarity) encodes
    the recipe" plus "no information".  With an uninformative partner it
    degrades to the info-free alternating strategy (which has positive
    expected value because a wrong delivery does not resample the recipe).
  * agent_1: pure signaller.  Stands in row 4 (visible from agent_0's row 2):
    recipe 0 -> cell (4,1) facing LEFT, recipe 1 -> cell (4,3) facing RIGHT.
    This satisfies column-threshold decoders, "below pot k" decoders and
    left/right facing decoders simultaneously.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'solo_cook_signal_decoder', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

# facing ids (observation)
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
# action ids
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5


def _dir_to_act():
    return jnp.array([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)


# ---- hypothesis space for the recipe-signal decoder (agent_0) ----
# features: 0 col>=3, 1 col>=2, 2 facing LEFT, 3 facing RIGHT, 4 facing UP,
#           5 facing DOWN, 6 item on counter (3,2), 7 partner holding something
# polarity 0: feature==1 <-> recipe 1;  polarity 1: feature==1 <-> recipe 0
NF = 8
NH = 1 + 2 * NF


def _hyp():
    h_feat = jnp.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], dtype=jnp.int32)
    h_pol = jnp.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    h_prior = jnp.array([0.06,
                         0.24, 0.015,
                         0.16, 0.01,
                         0.01, 0.06,
                         0.06, 0.01,
                         0.02, 0.02,
                         0.02, 0.02,
                         0.02, 0.01,
                         0.015, 0.015], dtype=jnp.float32)
    # predicted feature value for recipe r under hypothesis h: r XOR pol
    r_idx = jnp.array([0, 1], dtype=jnp.int32)
    h_pred = (r_idx[None, :] ^ h_pol[:, None]).astype(jnp.int32)  # [NH,2]
    return h_feat, h_pol, h_prior, h_pred


LIKE_OK = 1.8    # likelihood ratio vs. noise when observation matches
LIKE_BAD = 0.2   # when it contradicts
GRACE = 10       # steps after a delivery before the signal is re-anchored


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _local(gr, gc, my_r, my_c):
    i = gr - my_r + 2
    j = gc - my_c + 2
    valid = (i >= 0) & (i <= 4) & (j >= 0) & (j <= 4)
    return jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), valid


def _goto(my_r, my_c, my_dir, tr, tc, tdir):
    """Move to (tr,tc) (horizontal first), then turn to tdir.

    Returns (action_before_arrival_or_turn, arrived_and_facing).
    All turn actions used here are toward blocked cells so they never move.
    """
    dc = tc - my_c
    dr = tr - my_r
    move_h = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    move_v = jnp.where(dr > 0, A_DOWN, A_UP)
    at = (dc == 0) & (dr == 0)
    facing = my_dir == tdir
    act = jnp.where(dc != 0, move_h, jnp.where(dr != 0, move_v, _dir_to_act()[tdir]))
    return _i32(act), at & facing


def _goto_interact(my_r, my_c, my_dir, tr, tc, tdir):
    act, done = _goto(my_r, my_c, my_dir, tr, tc, tdir)
    return jnp.where(done, _i32(A_INTERACT), act)


def _goto_wait(my_r, my_c, my_dir, tr, tc, tdir):
    act, done = _goto(my_r, my_c, my_dir, tr, tc, tdir)
    return jnp.where(done, _i32(A_STAY), act)


# ---------------------------------------------------------------- agent 0

def init_agent_0():
    _, _, h_prior, _ = _hyp()
    return {
        't': _i32(0),
        't_deliv': _i32(0),
        'ndeliv': _i32(0),
        'd_last': _i32(0),
        'W': h_prior[:, None] * jnp.full((NH, 2), 0.5, dtype=jnp.float32),
        'anchor': jnp.full((NF,), -1, dtype=jnp.int32),
        'prev_anchor': jnp.full((NF,), -1, dtype=jnp.int32),
        'spurious': jnp.zeros((NF,), dtype=jnp.int32),
        'nchg': jnp.zeros((NF,), dtype=jnp.int32),
        'prev_cooked_type': _i32(-1),
        'pot0_mem': jnp.array([0, 0, 0, 0, 1], dtype=jnp.int32),  # total, cooking, ready, bad, accepts
    }


def _pot_state(obs, gr, gc, my_r, my_c):
    i, j, valid = _local(gr, gc, my_r, my_c)
    cell = obs[i, j]
    cnt = cell[34:38]
    total = cnt.sum()
    cooked = cell[33] > 0
    cooking = cell[44] > 0
    accepts = (total < 3) & (~cooking) & (~cooked)
    bad = (cnt[2] + cnt[3] > 0) | ((cnt[0] > 0) & (cnt[1] > 0))
    return valid, _i32(total), cooking, cooked, bad, accepts


def _counter_empty(obs, gr, gc, my_r, my_c):
    """True if counter (gr,gc) is visible and empty, or not visible (assume empty)."""
    i, j, valid = _local(gr, gc, my_r, my_c)
    occupied = obs[i, j, 32:38].sum() > 0
    return (~valid) | (~occupied)


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(jnp.int32)
    H_FEAT, H_POL, _, H_PRED = _hyp()

    # ---- self ----
    gidx = jnp.argmax(obs[:, :, 23].reshape(-1))
    gi, gj = gidx // 5, gidx % 5
    my_r = _i32(2 - gi)
    my_c = _i32(4 - gj)
    my_dir = _i32(jnp.argmax(obs[2, 2, 1:5]))
    inv = obs[2, 2, 5:11]
    has_plate = inv[0] > 0
    is_cooked = inv[1] > 0
    counts = inv[2:6]
    empty_plate = has_plate & (~is_cooked)
    hold_raw = (~has_plate) & (~is_cooked) & (counts.sum() > 0)
    raw_type = _i32(jnp.argmax(counts))
    good0 = (counts[0] == 3) & (counts[1] == 0) & (counts[2] == 0) & (counts[3] == 0)
    good1 = (counts[1] == 3) & (counts[0] == 0) & (counts[2] == 0) & (counts[3] == 0)
    dish_type = jnp.where(good0, 0, jnp.where(good1, 1, 2))
    cooked_type = jnp.where(is_cooked, dish_type, -1)
    empty_hands = inv.sum() == 0

    # ---- partner ----
    pm = obs[:, :, 11]
    p_vis = pm.sum() > 0
    pidx = jnp.argmax(pm.reshape(-1))
    pi, pj = pidx // 5, pidx % 5
    p_c = my_c + pj - 2
    p_dir = _i32(jnp.argmax(obs[pi, pj, 12:16]))
    p_hold = obs[pi, pj, 16:22].sum() > 0

    # ---- pots ----
    v0, tot0, cooking0, ready0, bad0, acc0 = _pot_state(obs, 3, 1, my_r, my_c)
    v1, tot1, cooking1, ready1, bad1, acc1 = _pot_state(obs, 3, 3, my_r, my_c)
    pm0 = mem['pot0_mem']
    tot0 = jnp.where(v0, tot0, pm0[0])
    cooking0 = jnp.where(v0, cooking0, pm0[1] > 0)
    ready0 = jnp.where(v0, ready0, pm0[2] > 0)
    bad0 = jnp.where(v0, bad0, pm0[3] > 0)
    acc0 = jnp.where(v0, acc0, pm0[4] > 0)
    pot0_mem = jnp.stack([tot0, cooking0.astype(jnp.int32), ready0.astype(jnp.int32),
                          bad0.astype(jnp.int32), acc0.astype(jnp.int32)]).astype(jnp.int32)

    # ---- counters ----
    c32_i, c32_j, c32_valid = _local(3, 2, my_r, my_c)
    c32_item = obs[c32_i, c32_j, 32:38].sum() > 0

    # ---- features ----
    f_val = jnp.stack([
        (p_c >= 3).astype(jnp.int32),
        (p_c >= 2).astype(jnp.int32),
        (p_dir == LEFT).astype(jnp.int32),
        (p_dir == RIGHT).astype(jnp.int32),
        (p_dir == UP).astype(jnp.int32),
        (p_dir == DOWN).astype(jnp.int32),
        c32_item.astype(jnp.int32),
        p_hold.astype(jnp.int32),
    ])
    f_def = jnp.stack([
        p_vis, p_vis, p_vis, p_vis, p_vis, p_vis,
        c32_valid,
        p_vis,
    ])

    # ---- belief: delivery event (cycle end) ----
    t = mem['t']
    W = mem['W']
    anchor = mem['anchor']
    prev_anchor = mem['prev_anchor']
    spurious = mem['spurious']
    nchg = mem['nchg']
    d_prev = mem['prev_cooked_type']
    delivered = (d_prev >= 0) & (d_prev <= 1) & (~is_cooked)
    W0, W1 = W[:, 0], W[:, 1]
    Wd0 = jnp.stack([W0 * 0.5, W1 + W0 * 0.5], axis=1)
    Wd1 = jnp.stack([W0 + W1 * 0.5, W1 * 0.5], axis=1)
    Wt = jnp.where(delivered, jnp.where(d_prev == 0, Wd0, Wd1), W)
    # a feature that has never changed across a delivery carries no evidence,
    # so only ungated (nchg>=1) features earn the "clean cycle" bonus.
    sp_f = jnp.where(anchor >= 0,
                     jnp.where(spurious > 0, LIKE_BAD, jnp.where(nchg >= 1, LIKE_OK, 1.0)),
                     1.0)
    sp_h = jnp.where(H_FEAT >= 0, sp_f[jnp.clip(H_FEAT, 0, NF - 1)], 1.0)
    Wt = jnp.where(delivered, Wt * sp_h[:, None], Wt)
    W = Wt / jnp.maximum(Wt.sum(), 1e-9)
    prev_anchor = jnp.where(delivered & (anchor >= 0), anchor, prev_anchor)
    anchor = jnp.where(delivered, -1, anchor)
    spurious = jnp.where(delivered, 0, spurious)
    t_deliv = jnp.where(delivered, t, mem['t_deliv'])
    ndeliv = mem['ndeliv'] + delivered.astype(jnp.int32)
    d_last = jnp.where(delivered, d_prev, mem['d_last'])

    # ---- belief: signal observation (anchor / re-anchor) ----
    in_window = (t - t_deliv) >= GRACE
    new_evt = f_def & in_window & (anchor < 0)
    chg_evt = f_def & (anchor >= 0) & (f_val != anchor)
    apply_evt = new_evt | chg_evt
    nchg = nchg + (new_evt & (prev_anchor >= 0) & (f_val != prev_anchor)).astype(jnp.int32)
    spurious = spurious | chg_evt.astype(jnp.int32)
    anchor = jnp.where(apply_evt, f_val, anchor)
    hf = jnp.clip(H_FEAT, 0, NF - 1)
    evt_h = apply_evt[hf] & (H_FEAT >= 0)
    val_h = f_val[hf]
    factor = jnp.where(evt_h[:, None], jnp.where(val_h[:, None] == H_PRED, LIKE_OK, LIKE_BAD), 1.0)
    # boundary change: the recipe changed right after my delivery d_last, so
    # that delivery was correct; this pins the polarity of feature phi.
    bchg = new_evt & (prev_anchor >= 0) & (f_val != prev_anchor)
    s_star = (prev_anchor ^ jnp.clip(mem['d_last'], 0, 1)) & 1
    kill = bchg[hf] & (H_FEAT >= 0) & (H_POL != s_star[hf])
    factor = jnp.where(kill[:, None], 0.05, factor)
    W2 = W * factor
    # features that never changed across a delivery carry no evidence: keep
    # their hypothesis mass frozen (only the pinned recipe belief updates).
    gated = (nchg[hf] == 0) & (H_FEAT >= 0)
    mass_old = W.sum(1)
    mass_new = jnp.maximum(W2.sum(1), 1e-9)
    W3 = jnp.where(gated[:, None], W2 * (mass_old / mass_new)[:, None], W2)
    W = W3 / jnp.maximum(W3.sum(), 1e-9)

    # ---- decision: which dish is wanted ----
    kk = jnp.maximum(ndeliv, t // 45)
    decay = jnp.where(kk <= 1, 1.0, jnp.where(kk == 2, 0.6, jnp.where(kk == 3, 0.2, 0.05)))
    # once some feature has demonstrably tracked a recipe change, features that
    # never changed are much less likely to be the partner's signal.
    any_ungated = (nchg >= 1).any()
    tau_f = jnp.where(nchg >= 1, 1.0, jnp.where(any_ungated, jnp.minimum(decay, 0.2), decay))
    tau_h = jnp.where(H_FEAT >= 0, tau_f[hf], 1.0)
    p0 = (tau_h * W[:, 0]).sum() / jnp.maximum((tau_h * W.sum(1)).sum(), 1e-9)
    d_choice = jnp.where(p0 >= 0.5, 0, 1)

    # ---- controller ----
    ready = jnp.stack([ready0, ready1])
    cooking = jnp.stack([cooking0, cooking1])
    accepts = jnp.stack([acc0, acc1])
    totals = jnp.stack([tot0, tot1])
    pot_col = jnp.array([1, 3], dtype=jnp.int32)

    # dump target: first free counter among (0,4),(0,1),(3,4),(3,2)
    e04 = _counter_empty(obs, 0, 4, my_r, my_c)
    e01 = _counter_empty(obs, 0, 1, my_r, my_c)
    e34 = _counter_empty(obs, 3, 4, my_r, my_c)
    dump_04 = _goto_interact(my_r, my_c, my_dir, 1, 4, UP)
    dump_01 = _goto_interact(my_r, my_c, my_dir, 1, 1, UP)
    dump_34 = _goto_interact(my_r, my_c, my_dir, 2, 4, DOWN)
    dump_32 = _goto_interact(my_r, my_c, my_dir, 2, 2, DOWN)
    e32 = ~c32_item
    act_deliver = _goto_interact(my_r, my_c, my_dir, 1, 2, UP)
    # if every dump slot is full, deliver the bad dish anyway to free the hands
    act_dump = jnp.where(e04, dump_04, jnp.where(e01, dump_01, jnp.where(e34, dump_34, jnp.where(e32, dump_32, act_deliver))))

    # A: holding cooked dish
    act_A = jnp.where(dish_type <= 1, act_deliver, act_dump)

    # B: holding empty plate
    d = d_choice
    other = 1 - d
    avail = ready | cooking
    bad = jnp.stack([bad0, bad1])
    good_avail = avail & (~bad)
    bad_ready = ready & bad
    uncertain = jnp.abs(p0 - 0.5) < 0.15
    d_eff = jnp.where(good_avail[d], d,
                      jnp.where(good_avail[other] & uncertain, other,
                                jnp.where(bad_ready[d], d, jnp.where(bad_ready[other], other, d))))
    pot_ok = good_avail[d_eff] | bad_ready[d_eff]
    scoop = _goto_interact(my_r, my_c, my_dir, 2, pot_col[d_eff], DOWN)
    wait = _goto_wait(my_r, my_c, my_dir, 2, pot_col[d_eff], DOWN)
    wait_center = _goto_wait(my_r, my_c, my_dir, 2, 2, DOWN)
    act_B = jnp.where(pot_ok, jnp.where(ready[d_eff], scoop, wait),
                      jnp.where(cooking[d], wait, act_dump))

    # C: holding raw ingredient (never feed a contaminated pot)
    good_acc = accepts & (~bad)
    k = jnp.clip(raw_type, 0, 1)
    tgt = jnp.where(good_acc[k] & (raw_type <= 1), k, jnp.where(good_acc[1 - k] & (raw_type <= 1), 1 - k, -1))
    put = _goto_interact(my_r, my_c, my_dir, 2, pot_col[jnp.clip(tgt, 0, 1)], DOWN)
    act_C = jnp.where(tgt >= 0, put, act_dump)

    # D: empty hands
    partial0 = good_acc[0] & (tot0 > 0)
    partial1 = good_acc[1] & (tot1 > 0)
    # partially filled pot first; otherwise the pot of the currently believed recipe
    fill_k = jnp.where(partial0, 0, jnp.where(partial1, 1,
                       jnp.where(good_acc[d], d, jnp.where(good_acc[other], other, -1))))
    get_ing0 = _goto_interact(my_r, my_c, my_dir, 1, 1, LEFT)
    get_ing1 = _goto_interact(my_r, my_c, my_dir, 2, 4, RIGHT)
    get_plate = _goto_interact(my_r, my_c, my_dir, 1, 3, UP)
    plate_useful = good_avail[d] | (good_avail.any() & uncertain) | bad_ready.any()
    act_D = jnp.where(fill_k == 0, get_ing0, jnp.where(fill_k == 1, get_ing1,
                      jnp.where(plate_useful, get_plate, wait_center)))

    action = jnp.where(is_cooked, act_A,
                       jnp.where(empty_plate, act_B,
                                 jnp.where(hold_raw, act_C, act_D)))
    action = _i32(action)

    new_mem = {
        't': t + 1,
        't_deliv': t_deliv,
        'ndeliv': ndeliv,
        'd_last': _i32(d_last),
        'W': W.astype(jnp.float32),
        'anchor': anchor.astype(jnp.int32),
        'prev_anchor': prev_anchor.astype(jnp.int32),
        'spurious': spurious.astype(jnp.int32),
        'nchg': nchg.astype(jnp.int32),
        'prev_cooked_type': _i32(cooked_type),
        'pot0_mem': pot0_mem,
    }
    trace = {
        'p0': p0.astype(jnp.float32),
        'd_choice': _i32(d_choice),
        'ndeliv': ndeliv,
        'nchg': nchg.astype(jnp.int32),
        'wmass': W.sum(1).astype(jnp.float32),
        'p_vis': p_vis,
        'fill_k': _i32(fill_k),
        'hold': _i32(jnp.where(is_cooked, 3, jnp.where(empty_plate, 2, jnp.where(hold_raw, 1, 0)))),
        'deliv_d': _i32(jnp.where(delivered, d_prev, -1)),
        'p_c': _i32(jnp.where(p_vis, p_c, -1)),
        'anchor': anchor.astype(jnp.int32),
        'spur': spurious.astype(jnp.int32),
        'my_rc': jnp.stack([my_r, my_c]).astype(jnp.int32),
    }
    return action, new_mem, trace


# ---------------------------------------------------------------- agent 1

def init_agent_1():
    return {'recipe': _i32(0), 't': _i32(0)}


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(jnp.int32)

    ridx = jnp.argmax(obs[:, :, 25].reshape(-1))
    ri, rj = ridx // 5, ridx % 5
    r_vis = obs[ri, rj, 25] > 0
    my_r = _i32(8 - ri)
    my_c = _i32(4 - rj)
    my_dir = _i32(jnp.argmax(obs[2, 2, 1:5]))
    rc = obs[ri, rj, 40:44]
    known = r_vis & (rc.sum() > 0)
    recipe = jnp.where(known, jnp.where(rc[1] > rc[0], 1, 0), mem['recipe'])
    recipe = _i32(recipe)

    # recipe 0: stand at (4,1) facing LEFT (turn is blocked by supply at (4,0))
    at0 = (my_r == 4) & (my_c == 1)
    act0 = jnp.where(at0 & (my_dir == LEFT), A_STAY,
                     jnp.where(at0, A_LEFT,
                               jnp.where(my_r == 5, A_UP, A_LEFT)))
    # recipe 1: stand at (4,3) facing RIGHT (arrive from (4,2) moving right)
    at1 = (my_r == 4) & (my_c == 3)
    act1 = jnp.where(at1 & (my_dir == RIGHT), A_STAY,
                     jnp.where(my_r == 5, A_UP,
                               jnp.where(my_c < 3, A_RIGHT, A_LEFT)))
    action = _i32(jnp.where(recipe == 1, act1, act0))

    new_mem = {'recipe': recipe, 't': mem['t'] + 1}
    trace = {'recipe': recipe, 'my_r': my_r, 'my_c': my_c}
    return action, new_mem, trace
