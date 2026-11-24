"""Asymmetric two-room courier for the test_time_wide Overcooked-V2 layout.

Layout facts that drive the design (see context/environment/README.md):

  WWXBWW      (0,2) delivery, (0,3) plate pile
  0 A  0      (1,0)/(1,5) ingredient 0 piles
  1    1      (2,0)/(2,5) ingredient 1 piles
  WPWPWW      (3,1)/(3,3) pots, (3,2)/(3,4) counters shared by both rooms
  3 A  3      (4,0)/(4,5) ingredient 3 piles
  W    W
  WWRWWW      (6,2) always-on recipe indicator

The two floor regions are disconnected 2x4 rectangles.  Seat 0 lives in rows
1-2 and owns the delivery station, the plate pile and the only useful
ingredient piles (0 and 1).  Seat 1 lives in rows 4-5 and is the only seat that
can ever see the recipe indicator.  Recipes are [0,0,0] or [1,1,1], so exactly
one bit has to travel from the lower room to the upper room every time the
recipe is resampled.

Conventions used here are deliberately "left means ingredient 0, right means
ingredient 1", applied redundantly to every channel a partner might plausibly
use: which pot the partner stands in front of, which side of its room it
occupies, which shared counter carries a raw marker, and - completely
convention free - the ingredient content of any plated dish that is handed
across the shared counters.  Seat 0 never delivers unless its accumulated
evidence is decisive, so an unreadable partner costs 0 rather than -20 a time.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'wide_room_courier',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5

# facing id (UP=0, DOWN=1, RIGHT=2, LEFT=3) produced by each movement action
FACE_OF_ACTION = (2, 1, 3, 0)

I32 = jnp.int32
MEM_LEN = 40

# ---------------------------------------------------------------- stations --
# (row, col, facing movement action); every facing target is a blocked cell so
# the movement action degenerates into a pure turn once the seat has arrived.
S0_DELIVER, S0_PLATE = 0, 1
S0_ING0, S0_ING1 = 2, 3
S0_POT_L, S0_POT_R = 4, 5
S0_CTR_L, S0_CTR_R = 6, 7
S0_STASH_A, S0_STASH_B = 8, 9
S0_IDLE = 10
A0_ST = (
    (1, 2, A_UP),     # deliver at (0,2)
    (1, 3, A_UP),     # plate pile (0,3)
    (1, 1, A_LEFT),   # ingredient 0 pile (1,0)
    (2, 4, A_RIGHT),  # ingredient 1 pile (2,5)
    (2, 1, A_DOWN),   # pot (3,1)
    (2, 3, A_DOWN),   # pot (3,3)
    (2, 2, A_DOWN),   # shared counter (3,2)
    (2, 4, A_DOWN),   # shared counter (3,4)
    (1, 1, A_UP),     # spare counter (0,1)
    (1, 4, A_UP),     # spare counter (0,4)
    (2, 2, A_DOWN),   # idle / watch post
)

S1_POT_L, S1_POT_R = 0, 1
S1_CTR_L, S1_CTR_R = 2, 3
S1_ING3_L, S1_ING3_R = 4, 5
S1_STASH_A, S1_STASH_B = 6, 7
S1_PEEK = 8
A1_ST = (
    (4, 1, A_UP),     # pot (3,1)
    (4, 3, A_UP),     # pot (3,3)
    (4, 2, A_UP),     # shared counter (3,2)
    (4, 4, A_UP),     # shared counter (3,4)
    (4, 1, A_LEFT),   # ingredient 3 pile (4,0)
    (4, 4, A_RIGHT),  # ingredient 3 pile (4,5)
    (5, 1, A_DOWN),   # spare counter (6,1)
    (5, 3, A_DOWN),   # spare counter (6,3)
    (4, 4, A_RIGHT),  # peek post: the only place counter (3,4) is visible from
)


# ----------------------------------------------------------------- helpers --
def _cell(obs, r, c, tr, tc):
    """Feature vector of global cell (tr, tc); zeros when outside the crop."""
    lr = tr - r + 2
    lc = tc - c + 2
    vis = (lr >= 0) & (lr <= 4) & (lc >= 0) & (lc <= 4)
    v = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), :].astype(I32)
    return vis, jnp.where(vis, v, jnp.zeros_like(v))


def _locate(obs, channel, gr, gc):
    """Exact self position from a landmark that is always inside the crop."""
    plane = obs[:, :, channel]
    idx = jnp.argmax(plane.reshape(-1))
    lr = (idx // 5).astype(I32)
    lc = (idx % 5).astype(I32)
    seen = plane.sum() > 0
    r = jnp.where(seen, gr - lr + 2, gr)
    c = jnp.where(seen, gc - lc + 2, gc)
    return r.astype(I32), c.astype(I32)


def _pot(obs, r, c, pr, pc, cache):
    """[i0, i1, i2, i3, cooked, timer]; cached (and ticked) while unseen."""
    vis, v = _cell(obs, r, c, pr, pc)
    fresh = jnp.concatenate([v[34:38], v[33:34], v[44:45]]).astype(I32)
    tick = jnp.maximum(cache[5] - 1, 0)
    done = jnp.maximum(cache[4], (cache[5] == 1).astype(I32))
    stale = jnp.concatenate([cache[0:4], done[None], tick[None]]).astype(I32)
    return jnp.where(vis, fresh, stale)


def _item(obs, r, c, tr, tc, cache):
    """[plate, cooked, i0, i1, i2, i3] of whatever sits on a counter/pot."""
    vis, v = _cell(obs, r, c, tr, tc)
    return jnp.where(vis, v[32:38].astype(I32), cache)


def _goto(r, c, st):
    """Vertical-then-horizontal move inside an open 2x4 room, else turn."""
    tr, tc, face = st[0], st[1], st[2]
    return jnp.where(r < tr, A_DOWN,
                     jnp.where(r > tr, A_UP,
                               jnp.where(c < tc, A_RIGHT,
                                         jnp.where(c > tc, A_LEFT, face))))


def _emit(r, c, facing, table, st_idx, do_int):
    st = jnp.asarray(table, dtype=I32)[st_idx]
    faces = jnp.asarray(FACE_OF_ACTION, dtype=I32)
    arrived = (r == st[0]) & (c == st[1]) & (facing == faces[st[2]])
    return jnp.where(arrived & do_int, A_INT, _goto(r, c, st)).astype(I32), arrived


def _inventory(obs):
    inv = obs[2, 2, 5:11].astype(I32)
    raw = inv[2] + inv[3] + inv[4] + inv[5]
    dish = inv[1] == 1
    plate = (inv[0] == 1) & (inv[1] == 0)
    ing = (inv[0] == 0) & (inv[1] == 0) & (raw > 0)
    return inv, plate, dish, ing


def _dish_kind(block):
    """(is a clean 3-stack of ingredient k, k) for an inventory/item block."""
    total = block[2] + block[3] + block[4] + block[5]
    pure0 = (block[2] == 3) & (total == 3)
    pure1 = (block[3] == 3) & (total == 3)
    return pure0, pure1


# ------------------------------------------------------------------ seat 0 --
def init_agent_0():
    return jnp.zeros((MEM_LEN,), dtype=I32)


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, jnp.zeros((MEM_LEN,), dtype=I32), memory.astype(I32))
    obs = obs.astype(I32)

    # memory layout
    ev = mem[0]
    step = mem[1]
    blackout = mem[14]
    last_offer = mem[15]

    r, c = _locate(obs, 23, 0, 2)            # delivery station (0,2)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(I32)
    inv, has_plate, has_dish, has_ing = _inventory(obs)

    pot_l = _pot(obs, r, c, 3, 1, mem[2:8])
    pot_r = _pot(obs, r, c, 3, 3, mem[8:14])
    ctr_l = _item(obs, r, c, 3, 2, mem[16:22])
    ctr_r = _item(obs, r, c, 3, 4, mem[22:28])
    _, sa_v = _cell(obs, r, c, 0, 1)
    _, sb_v = _cell(obs, r, c, 0, 4)
    stash_a = sa_v[32:38]
    stash_b = sb_v[32:38]

    # ------------------------------------------------------------- evidence --
    # Two separate accumulators.  ``evf`` only holds cues that name an
    # ingredient outright (a plated dish, or a partner pointing at a pot whose
    # contents are visible), so it needs no shared convention at all.  ``evc``
    # holds left/right cues whose polarity is a guess; that guess is calibrated
    # against evf when both are decisive and is flipped when repeated
    # deliveries fail to make the partner's signal move.
    evc = mem[28]
    pol_bit = mem[29]
    dcount = mem[30]
    last_sign = mem[31]
    moved = mem[32]
    prev_seen, prev_pr, prev_pc = mem[33], mem[34], mem[35]

    marker_l = ((ctr_l[5] > 0) & (ctr_l[1] == 0)).astype(I32)
    marker_r = ((ctr_r[5] > 0) & (ctr_r[1] == 0)).astype(I32)

    pplane = obs[:, :, 11]
    p_seen = pplane.sum() > 0
    pidx = jnp.argmax(pplane.reshape(-1))
    plr = (pidx // 5).astype(I32)
    plc = (pidx % 5).astype(I32)
    p_row = r + plr - 2
    p_col = c + plc - 2
    p_up = (obs[plr, plc, 12] == 1) & p_seen
    p_inv = obs[plr, plc, 16:22].astype(I32)
    moved_now = (prev_seen == 1) & p_seen & ((p_row != prev_pr) | (p_col != prev_pc))
    moved = jnp.maximum(moved, moved_now.astype(I32))

    def pot_ing(p):
        tot = p[0] + p[1] + p[2] + p[3]
        return (p[0] > 0) & (p[0] == tot), (p[1] > 0) & (p[1] == tot)

    l_is0, l_is1 = pot_ing(pot_l)
    r_is0, r_is1 = pot_ing(pot_r)
    # A partner that has never moved is not communicating; its spawn posture
    # must not be mistaken for a deliberate pointing gesture.
    active = moved == 1
    at_pot_l = p_up & active & (p_row == 4) & (p_col == 1)
    at_pot_r = p_up & active & (p_row == 4) & (p_col == 3)
    at_ctr_l = p_up & active & (p_row == 4) & (p_col == 2)
    at_ctr_r = p_up & active & (p_row == 4) & (p_col == 4)

    def dish_vote(block):
        p0, p1 = _dish_kind(block)
        return jnp.where(block[1] == 1,
                         jnp.where(p1, 6, jnp.where(p0, -6, 0)), 0).astype(I32)

    # convention-free: the pot the partner has posted itself at names its own
    # ingredient, so no shared left/right agreement is needed.  Facing the pot
    # counts for more than merely standing next to it.
    near_l = active & p_seen & (p_row == 4) & (p_col == 1)
    near_r = active & p_seen & (p_row == 4) & (p_col == 3)
    w_l = jnp.where(at_pot_l, 3, jnp.where(near_l, 2, 0))
    w_r = jnp.where(at_pot_r, 3, jnp.where(near_r, 2, 0))
    df = w_l * jnp.where(l_is1, 1, jnp.where(l_is0, -1, 0))
    df = df + w_r * jnp.where(r_is1, 1, jnp.where(r_is0, -1, 0))
    df = df + dish_vote(ctr_l) + dish_vote(ctr_r)
    df = df + jnp.where(p_seen, dish_vote(p_inv), 0)

    # convention-bound: pure left/right cues
    dc = -2 * marker_l + 2 * marker_r
    dc = dc - w_l * jnp.where((~l_is0) & (~l_is1), 1, 0)
    dc = dc + w_r * jnp.where((~r_is0) & (~r_is1), 1, 0)
    dc = dc + jnp.where(at_ctr_l, -2, 0) + jnp.where(at_ctr_r, 2, 0)
    dc = dc + jnp.where(active & p_seen & (p_row >= 3),
                        jnp.where(p_col <= 2, -1, 1), 0)

    quiet = blackout > 0
    evf = jnp.where(quiet, 0, jnp.clip(ev + df, -12, 12))
    evc = jnp.where(quiet, 0, jnp.clip(evc + dc, -12, 12))
    pol = jnp.where(pol_bit == 1, -1, 1).astype(I32)
    # calibrate the left/right polarity whenever an unambiguous cue disagrees
    both = (jnp.abs(evf) >= 5) & (jnp.abs(evc) >= 5)
    clash = both & (((evf > 0) & (evc * pol < 0)) | ((evf < 0) & (evc * pol > 0)))
    pol_bit = jnp.where(clash, 1 - pol_bit, pol_bit)
    pol = jnp.where(pol_bit == 1, -1, 1).astype(I32)

    gflip = mem[36]
    ev = jnp.clip(evf + pol * evc, -18, 18)
    ev = jnp.where(gflip == 1, -ev, ev)
    tgt = (ev > 0).astype(I32)
    sure = jnp.abs(ev) >= 5

    # ---------------------------------------------------------- pot summary --
    def pot_flags(p):
        total = p[0] + p[1] + p[2] + p[3]
        ready = p[4] == 1
        cooking = p[5] > 0
        fillable = (~ready) & (~cooking) & (total < 3)
        pure0 = (p[0] == 3) & (total == 3)
        pure1 = (p[1] == 3) & (total == 3)
        junk = ready & (~pure0) & (~pure1)
        return total, ready, cooking, fillable, junk

    lt, l_ready, l_cook, l_fill, l_junk = pot_flags(pot_l)
    rt, r_ready, r_cook, r_fill, r_junk = pot_flags(pot_r)

    # Plating is only ever worth doing once the recipe bit is decided: a dish
    # cooked for the wrong recipe has to be dumped and costs a whole pot cycle.
    l_match = sure & l_ready & (pot_l[tgt] == 3) & (lt == 3)
    r_match = sure & r_ready & (pot_r[tgt] == 3) & (rt == 3)
    match_any = l_match | r_match
    match_st = jnp.where(l_match, S0_POT_L, S0_POT_R)

    l_soon = sure & l_cook & (pot_l[tgt] == 3)
    r_soon = sure & r_cook & (pot_r[tgt] == 3)
    soon_any = l_soon | r_soon
    soon_st = jnp.where(l_soon, S0_POT_L, S0_POT_R)
    soon_t = jnp.where(l_soon, pot_l[5], pot_r[5])

    # keep pot (3,1) a pure ingredient-0 pot and pot (3,3) a pure ingredient-1
    # pot, but follow whatever is already inside so mixtures are never created.
    l_add = jnp.where(pot_l[1] > pot_l[0], 1, 0).astype(I32)
    r_add = jnp.where(pot_r[0] > pot_r[1], 0, 1).astype(I32)
    want_l = l_fill & (l_add == tgt)
    want_r = r_fill & (r_add == tgt)
    fill_pot = jnp.where(want_l, 0, jnp.where(want_r, 1,
                         jnp.where(l_fill, 0, jnp.where(r_fill, 1, -1))))
    fill_any = fill_pot >= 0
    fill_ing = jnp.where(fill_pot == 1, r_add, l_add)
    fill_pot_st = jnp.where(fill_pot == 1, S0_POT_R, S0_POT_L)
    ing_st = jnp.where(fill_ing == 1, S0_ING1, S0_ING0)

    # ------------------------------------------------------------- counters --
    cl_free = (ctr_l[0] + ctr_l[1] + ctr_l[2] + ctr_l[3] + ctr_l[4] + ctr_l[5]) == 0
    cr_free = (ctr_r[0] + ctr_r[1] + ctr_r[2] + ctr_r[3] + ctr_r[4] + ctr_r[5]) == 0
    cl_dish = ctr_l[1] == 1
    cr_dish = ctr_r[1] == 1
    ctr_plate = ((ctr_l[0] == 1) & (ctr_l[1] == 0)) | ((ctr_r[0] == 1) & (ctr_r[1] == 0))
    sa_free = (stash_a[0] + stash_a[1] + stash_a[2] + stash_a[3] + stash_a[4] + stash_a[5]) == 0
    sb_free = (stash_b[0] + stash_b[1] + stash_b[2] + stash_b[3] + stash_b[4] + stash_b[5]) == 0
    stash_st = jnp.where(sa_free, S0_STASH_A, S0_STASH_B)

    offer_want = ((~sure) & (~ctr_plate) & (~fill_any)
                  & (cl_free | cr_free) & ((step - last_offer) > 20))
    # Somewhere to put a plate down that is not needed right now.  Shared
    # counters come first, so a parked plate doubles as an offer to the partner.
    park_ok = cl_free | cr_free | sa_free | sb_free
    park_st = jnp.where(cl_free, S0_CTR_L,
                        jnp.where(cr_free, S0_CTR_R,
                                  jnp.where(sa_free, S0_STASH_A, S0_STASH_B)))

    # ------------------------------------------------------------ behaviour --
    grab_ctr = cl_dish | cr_dish
    grab_st = jnp.where(cl_dish, S0_CTR_L, S0_CTR_R)
    d_p0, d_p1 = _dish_kind(inv)
    d_pure = d_p0 | d_p1
    d_kind = d_p1.astype(I32)
    deliver_ok = has_dish & d_pure & sure & (d_kind == tgt)
    dump_dish = has_dish & ((~d_pure) | (sure & d_pure & (d_kind != tgt)))
    st_dish = jnp.where(dump_dish, stash_st, S0_DELIVER)
    int_dish = deliver_ok | dump_dish

    # Holding a plate must never become an absorbing state: if there is nothing
    # to scoop right now the plate is parked so the hands are free for stocking
    # pots or for collecting a dish the partner left on a shared counter.
    wait_now = soon_any & (soon_t <= 8)
    drop_need = (grab_ctr | fill_any) & park_ok
    plate_st = jnp.where(match_any, match_st,
                         jnp.where(l_junk, S0_POT_L,
                                   jnp.where(r_junk, S0_POT_R,
                                             jnp.where(wait_now, soon_st,
                                                       jnp.where(drop_need, park_st,
                                                                 jnp.where(soon_any, soon_st,
                                                                           jnp.where(offer_want, park_st, S0_IDLE)))))))
    int_plate = jnp.where(match_any | l_junk | r_junk, True,
                          jnp.where(wait_now, False,
                                    jnp.where(drop_need, True,
                                              jnp.where(soon_any, False, offer_want))))

    k = jnp.argmax(inv[2:6]).astype(I32)
    put_l = (k == 0) & l_fill & (l_add == 0)
    put_r = (k == 1) & r_fill & (r_add == 1)
    st_ing = jnp.where(put_l, S0_POT_L, jnp.where(put_r, S0_POT_R, stash_st))

    rec_a = (stash_a[1] == 1) & (stash_a[2 + tgt] == 3) & sure
    rec_b = (stash_b[1] == 1) & (stash_b[2 + tgt] == 3) & sure
    recycle = rec_a | rec_b
    rec_st = jnp.where(rec_a, S0_STASH_A, S0_STASH_B)
    # A ready dish is worth collecting straight away; otherwise stocking pots
    # comes before idling next to one that is still cooking.
    later_plate = soon_any | l_junk | r_junk | offer_want
    need_plate = match_any | later_plate
    cl_p = (ctr_l[0] == 1) & (ctr_l[1] == 0)
    cr_p = (ctr_r[0] == 1) & (ctr_r[1] == 0)
    plate_src = jnp.where(match_any & cl_p, S0_CTR_L,
                          jnp.where(match_any & cr_p, S0_CTR_R, S0_PLATE))
    st_empty = jnp.where(grab_ctr, grab_st,
                         jnp.where(recycle, rec_st,
                                   jnp.where(match_any, plate_src,
                                             jnp.where(fill_any, ing_st,
                                                       jnp.where(later_plate, S0_PLATE, S0_IDLE)))))
    int_empty = grab_ctr | recycle | need_plate | fill_any

    st_idx = jnp.where(has_dish, st_dish,
                       jnp.where(has_plate, plate_st,
                                 jnp.where(has_ing, st_ing, st_empty)))
    do_int = jnp.where(has_dish, int_dish,
                       jnp.where(has_plate, int_plate,
                                 jnp.where(has_ing, True, int_empty)))

    action, arrived = _emit(r, c, facing, A0_ST, st_idx, do_int)

    # ------------------------------------------------------------- bookkeep --
    acted = (action == A_INT) & arrived
    delivered = acted & (st_idx == S0_DELIVER) & has_dish
    offered = acted & has_plate & ((st_idx == S0_CTR_L) | (st_idx == S0_CTR_R))
    took_ctr = acted & (~has_dish) & (~has_plate) & (~has_ing) & grab_ctr
    taken = jnp.where(cl_dish, ctr_l, ctr_r)
    t_p0, t_p1 = _dish_kind(taken)
    evf_next = jnp.where(took_ctr, jnp.where(t_p1, 12, jnp.where(t_p0, -12, evf)), evf)
    evf_next = jnp.where(delivered, 0, evf_next)
    evc_next = jnp.where(delivered, 0, evc)
    blackout_next = jnp.where(delivered, 3, jnp.maximum(blackout - 1, 0))
    last_offer_next = jnp.where(offered, step, last_offer)

    # An incorrect delivery never resamples the recipe, so a partner signal that
    # refuses to move across several deliveries means the left/right polarity is
    # almost certainly backwards.  One observed change proves it is not.
    sgn = jnp.where(ev >= 5, 1, jnp.where(ev <= -5, -1, 0)).astype(I32)
    changed = (sgn != 0) & (last_sign != 0) & (sgn != last_sign)
    dcount_n = jnp.where(delivered, dcount + 1, dcount)
    dcount_n = jnp.where(changed, 0, dcount_n)
    do_flip = dcount_n >= 7
    gflip_n = jnp.where(do_flip, 1 - gflip, gflip)
    pol_bit_n = pol_bit
    dcount_n = jnp.where(do_flip, 0, dcount_n)
    last_sign_n = jnp.where(sgn != 0, sgn, last_sign)
    last_sign_n = jnp.where(do_flip, 0, last_sign_n)

    nxt = jnp.zeros((MEM_LEN,), dtype=I32)
    nxt = nxt.at[0].set(evf_next)
    nxt = nxt.at[1].set(step + 1)
    nxt = nxt.at[2:8].set(pot_l)
    nxt = nxt.at[8:14].set(pot_r)
    nxt = nxt.at[14].set(blackout_next)
    nxt = nxt.at[15].set(last_offer_next)
    nxt = nxt.at[16:22].set(ctr_l)
    nxt = nxt.at[22:28].set(ctr_r)
    nxt = nxt.at[28].set(evc_next)
    nxt = nxt.at[29].set(pol_bit_n)
    nxt = nxt.at[30].set(dcount_n)
    nxt = nxt.at[31].set(last_sign_n)
    nxt = nxt.at[32].set(moved)
    nxt = nxt.at[33].set(p_seen.astype(I32))
    nxt = nxt.at[34].set(p_row)
    nxt = nxt.at[35].set(p_col)
    nxt = nxt.at[36].set(gflip_n)

    trace = {
        'pos': jnp.stack([r, c]).astype(I32),
        'ev': ev.astype(I32),
        'evc': evc_next.astype(I32),
        'pol': pol_bit_n.astype(I32),
        'gflip': gflip_n.astype(I32),
        'dcount': dcount_n.astype(I32),
        'target': tgt.astype(I32),
        'sure': sure.astype(I32),
        'station': st_idx.astype(I32),
        'delivered': delivered.astype(I32),
    }
    return action.astype(I32), nxt, trace


# ------------------------------------------------------------------ seat 1 --
def init_agent_1():
    return jnp.zeros((MEM_LEN,), dtype=I32)


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, jnp.zeros((MEM_LEN,), dtype=I32), memory.astype(I32))
    obs = obs.astype(I32)

    step = mem[1]

    r, c = _locate(obs, 25, 6, 2)            # recipe indicator (6,2)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(I32)
    inv, has_plate, has_dish, has_ing = _inventory(obs)

    _, rec_v = _cell(obs, r, c, 6, 2)
    recipe = rec_v[38:44]
    tgt = (recipe[3] > 0).astype(I32)

    pot_l = _pot(obs, r, c, 3, 1, mem[2:8])
    pot_r = _pot(obs, r, c, 3, 3, mem[8:14])
    ctr_l = _item(obs, r, c, 3, 2, mem[16:22])
    ctr_r = _item(obs, r, c, 3, 4, mem[22:28])

    def pot_flags(p):
        total = p[0] + p[1] + p[2] + p[3]
        ready = p[4] == 1
        pure0 = (p[0] == 3) & (total == 3)
        pure1 = (p[1] == 3) & (total == 3)
        return ready, ready & (~pure0) & (~pure1)

    l_ready, l_junk = pot_flags(pot_l)
    r_ready, r_junk = pot_flags(pot_r)
    l_match = l_ready & (pot_l[tgt] == 3) & ((pot_l[0] + pot_l[1] + pot_l[2] + pot_l[3]) == 3)
    r_match = r_ready & (pot_r[tgt] == 3) & ((pot_r[0] + pot_r[1] + pot_r[2] + pot_r[3]) == 3)
    match_any = l_match | r_match
    match_st = jnp.where(l_match, S1_POT_L, S1_POT_R)
    purge_any = l_junk | r_junk
    purge_st = jnp.where(l_junk, S1_POT_L, S1_POT_R)
    l_soon = (pot_l[5] > 0) & (pot_l[tgt] == 3)
    r_soon = (pot_r[5] > 0) & (pot_r[tgt] == 3)
    soon_any = l_soon | r_soon
    soon_st = jnp.where(l_soon, S1_POT_L, S1_POT_R)

    def blk(x):
        return x[0] + x[1] + x[2] + x[3] + x[4] + x[5]

    cl_free = blk(ctr_l) == 0
    cr_free = blk(ctr_r) == 0
    cl_plate = (ctr_l[0] == 1) & (ctr_l[1] == 0)
    cr_plate = (ctr_r[0] == 1) & (ctr_r[1] == 0)
    cl_mark = (ctr_l[5] > 0) & (ctr_l[1] == 0)
    cr_mark = (ctr_r[5] > 0) & (ctr_r[1] == 0)

    # Pointing at the pot whose contents match the recipe needs no convention:
    # the partner can read the ingredient straight off the pot it is told about.
    def pot_ing(p):
        tot = p[0] + p[1] + p[2] + p[3]
        return (p[0] > 0) & (p[0] == tot), (p[1] > 0) & (p[1] == tot)

    l_is0, l_is1 = pot_ing(pot_l)
    r_is0, r_is1 = pot_ing(pot_r)
    l_says = jnp.where(tgt == 1, l_is1, l_is0)
    r_says = jnp.where(tgt == 1, r_is1, r_is0)
    signal_st = jnp.where(l_says, S1_POT_L,
                          jnp.where(r_says, S1_POT_R,
                                    jnp.where(tgt == 1, S1_POT_R, S1_POT_L)))
    # Counter (3,4) is invisible from column 1, so a plate offered there while
    # this seat points at the left pot would never be noticed: peek at it
    # whenever the cached view has gone stale.
    cr_vis = c >= 2
    last_cr = jnp.where(cr_vis, step, mem[15])
    patrol = (step - last_cr) > 25
    # No raw marker is ever left on a shared counter: both of them have to stay
    # clear for plates and finished dishes, which are the convention-free
    # channel that actually carries the recipe across the wall.

    # holding a plated dish: hand it across a shared counter, preferring (3,2)
    # which is visible from every cell of the upper room.
    hand_st = jnp.where(cl_free, S1_CTR_L, S1_CTR_R)
    hand_ok = cl_free | cr_free
    st_dish = jnp.where(hand_ok, hand_st, signal_st)
    int_dish = hand_ok

    st_plate = jnp.where(match_any, match_st,
                         jnp.where(purge_any, purge_st,
                                   jnp.where(soon_any, soon_st, signal_st)))
    int_plate = match_any | purge_any

    st_ing = S1_STASH_A
    int_ing = True

    take_plate = (cl_plate | cr_plate) & (match_any | purge_any | soon_any)
    take_st = jnp.where(cl_plate, S1_CTR_L, S1_CTR_R)
    st_empty = jnp.where(take_plate, take_st,
                         jnp.where(patrol, S1_PEEK, signal_st))
    int_empty = take_plate

    st_idx = jnp.where(has_dish, st_dish,
                       jnp.where(has_plate, st_plate,
                                 jnp.where(has_ing, st_ing, st_empty)))
    do_int = jnp.where(has_dish, int_dish,
                       jnp.where(has_plate, int_plate,
                                 jnp.where(has_ing, int_ing, int_empty)))

    action, arrived = _emit(r, c, facing, A1_ST, st_idx, do_int)

    nxt = jnp.zeros((MEM_LEN,), dtype=I32)
    nxt = nxt.at[1].set(step + 1)
    nxt = nxt.at[2:8].set(pot_l)
    nxt = nxt.at[8:14].set(pot_r)
    nxt = nxt.at[15].set(last_cr)
    nxt = nxt.at[16:22].set(ctr_l)
    nxt = nxt.at[22:28].set(ctr_r)

    trace = {
        'pos': jnp.stack([r, c]).astype(I32),
        'target': tgt.astype(I32),
        'station': st_idx.astype(I32),
        'patrol': patrol.astype(I32),
    }
    return action.astype(I32), nxt, trace
