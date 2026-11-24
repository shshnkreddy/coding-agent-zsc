"""Overcooked V2 demo_cook_wide policy.

Seat roles (grounded in geometry, not in a private convention):
  agent_0 (top room, cells (1,4)-(1,6)): has both ingredient supplies, the pot,
      plate piles and the delivery station adjacent, but can never see the recipe
      indicator.  It infers the (homogeneous) recipe from what is in the pot or
      what the partner hands over on the shared counters (2,4)/(2,6), completes
      the pot, plates the cooked dish and delivers it.  Evidence produced before
      its last delivery (recipe resample) is treated as stale: a pot seeded from
      stale evidence is delivered only once fresh evidence (partner's next
      ingredient) confirms the type, and disposed of when it contradicts.
      While idle it stages ingredients on the shared counters (one, keeping a
      counter free, until the partner is seen using them; then one of each type).
  agent_1 (bottom corridor): sees the recipe indicator and the pot from (3,5).
      It seeds the pot with the recipe ingredient (held ready, taken from the
      shared counters, a private counter cache, or the far supplies), keeps
      loading if the partner does not help, and never seeds a fresh pot while
      the partner still carries a cooked dish (the recipe may change on delivery).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'pot_seed_handoff', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

# ---------------------------------------------------------------- constants
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# observation channel holding the "facing" one-hot for each movement action id
FACE_CH_OF_ACTION_L = [3, 2, 4, 1]

ASCII = [
    "WWWWBXBWWWW",
    "WWW0 A 1WWW",
    "WWWWWPWWWWW",
    "W    A    W",
    "0  W3R3W  0",
    "W1WWWWWWW1W",
]
H, W = 6, 11


A0_CELLS_L = [(1, 4), (1, 5), (1, 6)]
A1_CELLS_L = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
              (4, 1), (4, 2), (4, 8), (4, 9)]

# agent-1 counter slots: 8 private cache counters (bottom room only) + 2 shared counters
SLOT_R_L = [2, 4, 2, 2, 2, 4, 2, 2, 2, 2]
SLOT_C_L = [3, 3, 2, 1, 7, 7, 8, 9, 4, 6]
SLOT_FACE_L = [A_UP, A_DOWN, A_UP, A_UP, A_UP, A_DOWN, A_UP, A_UP, A_UP, A_UP]
SLOT_SIDE_L = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
SLOT_ROW4_L = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
SLOT_SHARED_L = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
NSLOT = 10


def _static_channels(r, c):
    """10 static channels for global cell (r, c); zeros when off-map."""
    v = [0] * 10
    if 0 <= r < H and 0 <= c < W:
        ch = ASCII[r][c]
        if ch == 'W':
            v[0] = 1
        elif ch == 'X':
            v[1] = 1
        elif ch == 'P':
            v[2] = 1
        elif ch == 'R':
            v[3] = 1
        elif ch == 'L':
            v[4] = 1
        elif ch == 'B':
            v[5] = 1
        elif ch in '0123':
            v[6 + int(ch)] = 1
    return v


def _templates(cells):
    t = [[[_static_channels(r + i - 2, c + j - 2) for j in range(5)] for i in range(5)]
         for (r, c) in cells]
    return jnp.asarray(t, dtype=jnp.int32)


# ---------------------------------------------------------------- helpers
def _locate(obs, templates, cells):
    stat = obs[:, :, 22:32]
    d = jnp.abs(stat[None] - templates).sum(axis=(1, 2, 3))
    k = jnp.argmin(d)
    return cells[k, 0], cells[k, 1]


def _read(obs, sr, sc, gr, gc, c0, n):
    """Read channels [c0, c0+n) of global cell (gr, gc); zero + invalid if out of view."""
    li = gr - sr + 2
    lj = gc - sc + 2
    ok = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    v = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), c0:c0 + n]
    return v * ok.astype(v.dtype), ok


def _item(block):
    """Decode a 6-entry item block -> (empty, junk, is_raw_ingredient, type, total)."""
    plate, cooked, counts = block[0], block[1], block[2:6]
    total = counts.sum()
    typ = jnp.argmax(counts)
    empty = (plate == 0) & (cooked == 0) & (total == 0)
    junk = (plate > 0) | (cooked > 0)
    is_ing = (~junk) & (total > 0)
    return empty, junk, is_ing, typ, total


def _select(conds, goals, default):
    out = default
    for c, g in zip(reversed(conds), reversed(goals)):
        out = jnp.where(c, g, out)
    return out


def _execute(obs, sr, sc, goal, nav):
    tr, tc, fa, do_int = goal[0], goal[1], goal[2], goal[3]
    at = (sr == tr) & (sc == tc)
    facing = obs[2, 2, jnp.asarray(FACE_CH_OF_ACTION_L, jnp.int32)[fa]] > 0
    act = jnp.where(~at, nav, jnp.where(~facing, fa, jnp.where(do_int > 0, A_INTERACT, A_STAY)))
    return act.astype(jnp.int32)


def _goal(tr, tc, fa, do_int):
    return jnp.stack([jnp.asarray(tr, jnp.int32), jnp.asarray(tc, jnp.int32),
                      jnp.asarray(fa, jnp.int32), jnp.asarray(do_int, jnp.int32)])


def _partner(obs, sr, sc):
    """Partner visibility, global position, facing one-hot (UP, DOWN, RIGHT, LEFT) and held item."""
    ppos = obs[:, :, 11]
    vis = ppos.sum() > 0
    pflat = jnp.argmax(ppos.reshape(-1))
    pr = sr + pflat // 5 - 2
    pc = sc + pflat % 5 - 2
    pface = obs[:, :, 12:16].sum(axis=(0, 1))
    pinv = obs[:, :, 16:22].sum(axis=(0, 1))
    return vis, pr, pc, pface, pinv


def _b(x):
    return jnp.asarray(x).astype(jnp.int32)


# ================================================================ agent 0
# goal ids
G0_DELIVER, G0_DISPOSE_L, G0_DISPOSE_R, G0_SCOOP, G0_WAITPOT, G0_INSERT = 0, 1, 2, 3, 4, 5
G0_PLATE_L, G0_PLATE_R, G0_ING0, G0_ING1, G0_TAKE_L, G0_TAKE_R = 6, 7, 8, 9, 10, 11
G0_TABLE_L = [
    [1, 5, A_UP, 1],      # deliver
    [1, 4, A_DOWN, 1],    # dispose on (2,4)
    [1, 6, A_DOWN, 1],    # dispose on (2,6)
    [1, 5, A_DOWN, 1],    # scoop
    [1, 5, A_DOWN, 0],    # wait at pot
    [1, 5, A_DOWN, 1],    # insert
    [1, 4, A_UP, 1],      # plate from (0,4)
    [1, 6, A_UP, 1],      # plate from (0,6)
    [1, 4, A_LEFT, 1],    # ingredient 0 from (1,3)
    [1, 6, A_RIGHT, 1],   # ingredient 1 from (1,7)
    [1, 4, A_DOWN, 1],    # take from (2,4)
    [1, 6, A_DOWN, 1],    # take from (2,6)
]
STAGE_MIN_TIMER = 11
HOLD_PATIENCE = 12
PLATE_PATIENCE = 6
DISH_PATIENCE = 20

# memory layout (all int32)
M0_MINE_L, M0_MINE_R = 0, 1          # item on shared counter (2,4)/(2,6) was placed by me
M0_PLATE_WAIT = 2                    # steps holding a plate while the pot is idle
M0_PREV_PC = 3                       # partner column last step (row 3) or -1
M0_HOLD_WAIT = 4                     # steps holding an ingredient while the pot is empty
M0_SEEDABLE = 5                      # held ingredient came from a partner hand-off
M0_STALE_SEED = 6                    # current pot was seeded from pre-resample evidence
M0_DISH_VERDICT = 7                  # 0 unknown / 1 matches recipe / 2 does not
M0_DISH_WAIT = 8                     # steps holding an unverified dish
M0_PARTNER_TAKES = 9                 # partner has removed one of my staged items
M0_TOOK_L, M0_TOOK_R = 10, 11        # I took from (2,4)/(2,6) last step
M0_PARTNER_FRESH = 12                # partner seen empty-handed since my last delivery
M0_FRESH_L, M0_FRESH_R = 13, 14      # counter item appeared after fresh evidence was possible
M0_HELD_FRESH = 15                   # held ingredient is fresh evidence for the recipe
M0_PREV_TOTAL = 16                   # pot ingredient count last step
M0_I_INSERTED = 17                   # I inserted into the pot last step
M0_PREV_H_DISH = 18                  # I held a dish last step
M0_POT_VERDICT = 19                  # verdict for the (stale-seeded) pot contents
M0_STAGING = 20                      # held ingredient was picked up to stage on a counter
M0_SCOOPED = 21                      # I scooped last step
M0_PREV_L_EMPTY, M0_PREV_R_EMPTY = 22, 23
M0_EMPTY_SEEN = 24                   # partner seen empty-handed since my last scoop
M0_STALE_EVER = 25                   # a stale seeding has been observed this episode
M0_SIZE = 26


def init_agent_0():
    m = [0] * M0_SIZE
    m[M0_PREV_PC] = -1
    m[M0_PARTNER_FRESH] = 1
    m[M0_PREV_L_EMPTY] = 1
    m[M0_PREV_R_EMPTY] = 1
    return jnp.asarray(m, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_0(), memory)
    sr, sc = _locate(obs, _templates(A0_CELLS_L), jnp.asarray(A0_CELLS_L, jnp.int32))

    inv = obs[2, 2, 5:11]
    h_empty, h_junk, h_ing, h_type, h_total = _item(inv)
    h_plate = (inv[0] > 0) & (inv[1] == 0)
    h_dish = inv[1] > 0
    dish_valid = h_dish & (inv[2:6].max() == 3) & (h_total == 3) & (h_type <= 1)

    partner_vis, pr, pc, pface, pinv = _partner(obs, sr, sc)
    _, _, pt_ing, pt_type, _ = _item(pinv)
    pt_ing = pt_ing & (pt_type <= 1)
    partner_empty = partner_vis & (pinv.sum() == 0)
    p_toward = ((pc >= 4) & (pc <= 6)) | ((pc == 3) & (pface[2] > 0)) | ((pc == 7) & (pface[3] > 0))
    partner_row3 = partner_vis & (pr == 3)
    partner_coming = partner_row3 & p_toward
    partner_at_pot = partner_row3 & (pc == 5) & (pface[0] > 0)
    partner_fresh = mem[M0_PARTNER_FRESH] > 0
    # partner just arrived carrying a fresh ingredient (was not standing at the pot last step)
    arriving = partner_coming & pt_ing & (mem[M0_PREV_PC] != 5) & partner_fresh

    pot, _ = _read(obs, sr, sc, 2, 5, 32, 6)
    timer = _read(obs, sr, sc, 2, 5, 44, 1)[0][0]
    _, _, _, p_type, p_total = _item(pot)
    p_ready = pot[1] > 0
    p_cooking = timer > 0
    p_idle = (~p_ready) & (~p_cooking)
    p_empty = p_idle & (p_total == 0)
    p_partial = p_idle & (p_total > 0) & (p_total < 3)
    p_good = (pot[2:6].max() == p_total) & (p_type <= 1)

    cl, _ = _read(obs, sr, sc, 2, 4, 32, 6)
    cr, _ = _read(obs, sr, sc, 2, 6, 32, 6)
    l_empty, l_junk, l_ing, l_type, _ = _item(cl)
    r_empty, r_junk, r_ing, r_type, _ = _item(cr)
    l_plate = (cl[0] > 0) & (cl[1] == 0)
    r_plate = (cr[0] > 0) & (cr[1] == 0)
    # stored cooked dishes (a wrong homogeneous dish becomes right once the recipe flips)
    l_dish = cl[1] > 0
    r_dish = cr[1] > 0
    l_dish_ok = l_dish & (cl[2:6].max() == 3) & (cl[2:6].sum() == 3) & (l_type <= 1)
    r_dish_ok = r_dish & (cr[2:6].max() == 3) & (cr[2:6].sum() == 3) & (r_type <= 1)
    # my staged item vanished without me taking it -> the partner uses staged items
    partner_takes = jnp.where(
        ((mem[M0_MINE_L] > 0) & l_empty & (mem[M0_TOOK_L] == 0)) |
        ((mem[M0_MINE_R] > 0) & r_empty & (mem[M0_TOOK_R] == 0)), 1, mem[M0_PARTNER_TAKES])
    mine_l = jnp.where(l_empty, 0, mem[M0_MINE_L])
    mine_r = jnp.where(r_empty, 0, mem[M0_MINE_R])
    sig_l = l_ing & (l_type <= 1) & (mine_l == 0)
    sig_r = r_ing & (r_type <= 1) & (mine_r == 0)
    # freshness of counter items: appeared while the partner could have fetched post-resample
    fresh_l = jnp.where(l_empty, 0, jnp.where((mem[M0_PREV_L_EMPTY] > 0) & l_ing, _b(partner_fresh), mem[M0_FRESH_L]))
    fresh_r = jnp.where(r_empty, 0, jnp.where((mem[M0_PREV_R_EMPTY] > 0) & r_ing, _b(partner_fresh), mem[M0_FRESH_R]))

    # pot seeding detection and staleness
    scooped_last = mem[M0_SCOOPED] > 0
    seeded_now = p_idle & (p_total > 0) & ((mem[M0_PREV_TOTAL] == 0) | scooped_last)
    my_seed = seeded_now & (mem[M0_I_INSERTED] > 0)
    stale_by_partner = (~partner_fresh) | (mem[M0_PREV_H_DISH] > 0) | scooped_last
    stale_seed = jnp.where(seeded_now,
                           jnp.where(my_seed, mem[M0_HELD_FRESH] == 0, stale_by_partner),
                           jnp.where(p_empty, False, mem[M0_STALE_SEED] > 0))

    # fresh recipe evidence: partner's held ingredient after being seen empty-handed post
    # delivery, or a counter item that appeared under the same condition
    ev_partner = partner_row3 & pt_ing & partner_fresh
    ev_l = sig_l & (fresh_l > 0)
    ev_r = sig_r & (fresh_r > 0)
    ref_type = jnp.where(h_dish, h_type, p_type)
    ev_ok = (ev_partner & (pt_type == ref_type)) | (ev_l & (l_type == ref_type)) | (ev_r & (r_type == ref_type))
    ev_bad = (ev_partner & (pt_type != ref_type)) | (ev_l & (l_type != ref_type)) | (ev_r & (r_type != ref_type))
    pot_verdict = jnp.where(p_empty | seeded_now, 0, mem[M0_POT_VERDICT])
    pot_verdict = jnp.where((pot_verdict == 0) & stale_seed & (p_total > 0) & (~h_dish),
                            jnp.where(ev_ok, 1, jnp.where(ev_bad, 2, 0)), pot_verdict)
    dish_verdict = jnp.where(h_dish, mem[M0_DISH_VERDICT], 0)
    dish_verdict = jnp.where(h_dish & (dish_verdict == 0), jnp.where(ev_ok, 1, jnp.where(ev_bad, 2, 0)), dish_verdict)
    dish_wait = jnp.where(h_dish & (dish_verdict == 0), mem[M0_DISH_WAIT] + 1, 0)

    prefer_l = sc <= 5
    any_free = l_empty | r_empty
    # disposal / staging side: type 0 -> (2,4), type 1 -> (2,6); otherwise nearest free
    want_l = jnp.where(h_ing, h_type == 0, prefer_l)
    disp_goal = jnp.where(want_l,
                          jnp.where(l_empty, G0_DISPOSE_L, G0_DISPOSE_R),
                          jnp.where(r_empty, G0_DISPOSE_R, G0_DISPOSE_L))
    need = jnp.where(p_good, p_type, 0)
    supply_goal = jnp.where(need == 0, G0_ING0, G0_ING1)
    l_has = l_ing & (l_type == need)
    r_has = r_ing & (r_type == need)
    hand_l = l_has & (mine_l == 0)
    hand_r = r_has & (mine_r == 0)
    take_goal = jnp.where(l_has, G0_TAKE_L, jnp.where(r_has, G0_TAKE_R, supply_goal))
    # partner hand-offs are taken; the supply is used when the pot cannot fill before I am
    # back (one ingredient in) or a counter is free to put a redundant one down; otherwise
    # take my own staged item (which frees the counter it came from)
    safe_supply = (p_total == 1) | any_free
    # while the partner is not known to use staged items, consume my probe as a pot fill
    probe_l = l_has & (mine_l > 0)
    probe_r = r_has & (mine_r > 0)
    use_probe = (partner_takes == 0) & (probe_l | probe_r)
    probe_goal = jnp.where(probe_l, G0_TAKE_L, G0_TAKE_R)
    fetch_goal = jnp.where(hand_l, G0_TAKE_L,
                           jnp.where(hand_r, G0_TAKE_R,
                                     jnp.where(use_probe, probe_goal,
                                               jnp.where(safe_supply, supply_goal, take_goal))))
    # a stored dish whose type is confirmed by fresh evidence: fetch it and deliver
    fresh_type = jnp.where(ev_partner, pt_type, jnp.where(ev_l, l_type, jnp.where(ev_r, r_type, -1)))
    stored_ok_l = l_dish_ok & (fresh_type == l_type)
    stored_ok_r = r_dish_ok & (fresh_type == r_type)
    stored_goal = jnp.where(stored_ok_l, G0_TAKE_L, G0_TAKE_R)
    plate_goal = jnp.where(l_plate, G0_TAKE_L,
                           jnp.where(r_plate, G0_TAKE_R,
                                     jnp.where(prefer_l, G0_PLATE_L, G0_PLATE_R)))
    sig_conflict = sig_l & sig_r & (l_type != r_type)
    sig_any = (sig_l | sig_r) & (~sig_conflict)
    sig_goal = jnp.where(prefer_l,
                         jnp.where(sig_l, G0_TAKE_L, G0_TAKE_R),
                         jnp.where(sig_r, G0_TAKE_R, G0_TAKE_L))

    # staging: put ingredients on the shared counters while idle.  Until the partner has been
    # seen taking staged items, stage at most one so a counter always stays free for
    # partners that hand ingredients over via the counters; afterwards stage one of each type.
    first_type = jnp.where((p_total > 0) & p_good, p_type, jnp.where(prefer_l, 0, 1))
    stage_goal = jnp.where(l_empty & r_empty,
                           jnp.where(first_type == 0, G0_ING0, G0_ING1),
                           jnp.where(l_empty, G0_ING0, G0_ING1))
    stale_ever = (mem[M0_STALE_EVER] > 0) | stale_seed
    stage_time = jnp.where(partner_takes > 0,
                           (p_cooking & (timer >= STAGE_MIN_TIMER)) | (p_empty & (~sig_any)),
                           ((p_cooking & (timer >= STAGE_MIN_TIMER)) | (p_empty & (~sig_any)))
                           & (~stale_ever))
    stage_room = jnp.where(partner_takes > 0, any_free, l_empty & r_empty)
    stage_ok = h_empty & stage_time & stage_room
    stage_or_plate = jnp.where(stage_room & (partner_takes > 0), stage_goal, plate_goal)

    # partner stands at the pot with the last ingredient: stage/plate instead of a redundant
    # fetch (partners heading for a counter instead would leave me stuck holding a plate)
    partner_brings = partner_row3 & (pc == 5) & pt_ing & (pt_type == need)
    skip_third = p_partial & (p_total == 2) & partner_brings

    # holding a plate while the pot still needs ingredients: after a grace period park it
    plate_stuck = h_plate & p_idle & (p_total < 3)
    plate_wait = jnp.where(plate_stuck, mem[M0_PLATE_WAIT] + 1, 0)
    park_plate = plate_stuck & (plate_wait > PLATE_PATIENCE) & any_free

    # holding an ingredient while the pot is empty: only seed with evidence for the recipe
    seed_ev = (mem[M0_SEEDABLE] > 0) | (sig_l & (l_type == h_type)) | (sig_r & (r_type == h_type))
    can_insert = h_ing & p_idle & (p_total < 3) & (
        ((p_total > 0) & ((p_type == h_type) | (~p_good))) | (p_empty & seed_ev))
    hold_wait = jnp.where(h_ing & p_empty, mem[M0_HOLD_WAIT] + 1, 0)
    conflict = (arriving & (pt_type != h_type)) | (sig_l & (l_type != h_type)) | (sig_r & (r_type != h_type))
    staging = mem[M0_STAGING] > 0
    keep_holding = h_ing & p_empty & (~conflict) & (hold_wait <= HOLD_PATIENCE) & (~staging)
    prefetch = h_empty & p_empty & arriving & (~sig_any)
    prefetch_goal = jnp.where(pt_type == 0, G0_ING0, G0_ING1)

    # dish from a stale-seeded pot: deliver once confirmed, dispose if contradicted, deliver
    # anyway after a patience limit (a partner that is never seen empty-handed gives no evidence)
    verdict = jnp.where(mem[M0_DISH_VERDICT] > 0, mem[M0_DISH_VERDICT], dish_verdict)
    deliver_ok = h_dish & dish_valid & ((verdict == 1) | (dish_wait > DISH_PATIENCE))
    dish_bad = h_dish & ((~dish_valid) | (verdict == 2))

    conds = [
        deliver_ok,                                          # 0
        dish_bad & any_free,                                 # 1
        dish_bad,                                            # 2
        h_dish,                                              # 3 unverified: wait
        h_plate & p_ready,                                   # 4
        park_plate,                                          # 5
        h_plate,                                             # 6
        can_insert,                                          # 7
        keep_holding,                                        # 8
        h_ing & any_free,                                    # 9
        h_ing,                                               # 10
        h_junk & any_free,                                   # 11
        h_junk,                                              # 12
        h_empty & (stored_ok_l | stored_ok_r),               # 13 stored dish now matches
        h_empty & p_ready,                                   # 14
        h_empty & p_cooking & (timer < STAGE_MIN_TIMER),     # 15
        h_empty & skip_third,                                # 16
        h_empty & p_partial,                                 # 17
        h_empty & p_empty & sig_any,                         # 18
        prefetch,                                            # 19
        stage_ok,                                            # 20
        h_empty & p_cooking,                                 # 21
    ]
    goals = [
        G0_DELIVER, disp_goal, G0_DELIVER, G0_WAITPOT, G0_SCOOP, disp_goal, G0_WAITPOT, G0_INSERT,
        G0_WAITPOT, disp_goal, G0_WAITPOT, disp_goal, G0_WAITPOT, stored_goal, plate_goal, plate_goal,
        stage_or_plate, fetch_goal, sig_goal, prefetch_goal, stage_goal, plate_goal,
    ]
    idx = [jnp.asarray(i, jnp.int32) for i in range(len(conds))]
    sel = _select(conds, idx, jnp.asarray(99, jnp.int32))
    gid = _select(conds, goals, jnp.asarray(G0_WAITPOT, jnp.int32))
    goal = jnp.asarray(G0_TABLE_L, jnp.int32)[gid]
    nav = jnp.where(sc < goal[1], A_RIGHT, A_LEFT)
    action = _execute(obs, sr, sc, goal, nav)

    interacting = action == A_INTERACT
    placing = interacting & (~h_empty)
    mine_l = jnp.where(placing & (gid == G0_DISPOSE_L) & l_empty, 1, mine_l)
    mine_r = jnp.where(placing & (gid == G0_DISPOSE_R) & r_empty, 1, mine_r)
    took_l = interacting & h_empty & (gid == G0_TAKE_L)
    took_r = interacting & h_empty & (gid == G0_TAKE_R)
    taking_sig = (took_l & sig_l) | (took_r & sig_r)
    picking = interacting & h_empty & ((gid == G0_ING0) | (gid == G0_ING1) | took_l | took_r)
    stage_pick = picking & ((sel == 20) | ((sel == 16) & stage_room & (partner_takes > 0)))
    took_stored = interacting & h_empty & (sel == 13)
    seedable = jnp.where(h_empty, _b(taking_sig), mem[M0_SEEDABLE])
    held_fresh_new = (took_l & (fresh_l > 0)) | (took_r & (fresh_r > 0)) | \
        (interacting & h_empty & (sel == 19)) | (picking & (sel == 17))
    held_fresh = jnp.where(h_empty, _b(held_fresh_new), mem[M0_HELD_FRESH])
    staging_next = jnp.where(h_empty, _b(stage_pick), mem[M0_STAGING])
    prev_pc = jnp.where(partner_row3, pc, -1)
    scooping = interacting & h_plate & p_ready & (gid == G0_SCOOP)
    delivering = interacting & h_dish & (gid == G0_DELIVER)
    inserting = interacting & h_ing & (gid == G0_INSERT) & p_empty
    empty_seen = jnp.where(scooping, 0, jnp.where(partner_empty, 1, mem[M0_EMPTY_SEEN]))
    partner_fresh_next = jnp.where(delivering, _b(partner_empty | (empty_seen > 0)),
                                   jnp.where(partner_empty, 1, _b(partner_fresh)))
    fresh_l_next = jnp.where(delivering, 0, fresh_l)
    fresh_r_next = jnp.where(delivering, 0, fresh_r)
    dish_verdict_next = jnp.where(scooping, jnp.where(stale_seed, pot_verdict, 1),
                                  jnp.where(took_stored, 1, jnp.where(h_dish, verdict, 0)))
    new_mem = jnp.stack([
        mine_l, mine_r, plate_wait, prev_pc, hold_wait, seedable, _b(stale_seed),
        dish_verdict_next, dish_wait, partner_takes, _b(took_l), _b(took_r), partner_fresh_next,
        fresh_l_next, fresh_r_next, held_fresh, p_total, _b(inserting), _b(h_dish), pot_verdict,
        staging_next, _b(scooping), _b(l_empty), _b(r_empty), empty_seen, _b(stale_ever),
    ]).astype(jnp.int32)
    trace = {'goal': gid.astype(jnp.int32), 'col': sc.astype(jnp.int32),
             'pot_total': p_total.astype(jnp.int32), 'stale': _b(stale_seed),
             'verdict': verdict.astype(jnp.int32), 'fresh': _b(partner_fresh)}
    return action, new_mem, trace


# ================================================================ agent 1
# memory layout: [0] recipe (-1 unknown), [1:11] slot contents, [11:16] pot memory,
#   [16] steps the pot has held a homogeneous wrong-type partial load,
#   [17] steps the pot has been partial without its content changing, [18] last pot total,
#   [19] partner has been seen staging ingredients on the shared counters
#   pot memory = [total, type, cooking, ready, good]
M1_RECIPE = 0
M1_SLOT = 1
M1_POT = 11
M1_WRONG = 16
M1_STALE = 17
M1_LAST_TOTAL = 18
M1_PARTNER_STAGES = 19
M1_SIZE = 20
WRONG_POT_PATIENCE = 30
HELP_PATIENCE = 8

# goal ids for trace only
G1_READ, G1_INSERT, G1_WAITPOT, G1_FAR, G1_GET, G1_DUMP_ING, G1_DUMP_JUNK, G1_CLEAN = 0, 1, 2, 3, 4, 5, 6, 7


def init_agent_1():
    m = [-1] + [-1] * NSLOT + [0] * 5 + [0, 0, 0, 0]
    return jnp.asarray(m, dtype=jnp.int32)


def _far_goal(side, t):
    """Far supply on the given side (0 = left cell (4,1), 1 = right cell (4,9))."""
    left = side == 0
    face = jnp.where(left, jnp.where(t == 0, A_LEFT, A_DOWN), jnp.where(t == 0, A_RIGHT, A_DOWN))
    return _goal(4, jnp.where(left, 1, 9), face, 1)


def _slot_goal(k):
    return _goal(3, jnp.asarray(SLOT_C_L, jnp.int32)[k], jnp.asarray(SLOT_FACE_L, jnp.int32)[k], 1)


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_1(), memory)
    sr, sc = _locate(obs, _templates(A1_CELLS_L), jnp.asarray(A1_CELLS_L, jnp.int32))
    near_side = jnp.where(sc <= 5, 0, 1)

    inv = obs[2, 2, 5:11]
    h_empty, _, h_is_ing, h_type, _ = _item(inv)
    h_ing = h_is_ing & (h_type <= 1)
    h_junk = (~h_empty) & (~h_ing)

    # recipe
    rc = obs[:, :, 40:44].sum(axis=(0, 1))
    rec_vis = rc.sum() > 0
    rec = jnp.where(rec_vis, jnp.argmax(rc), mem[M1_RECIPE]).astype(jnp.int32)
    rec01 = jnp.clip(rec, 0, 1)

    # pot (observed or remembered)
    pot, pot_vis = _read(obs, sr, sc, 2, 5, 32, 6)
    timer = _read(obs, sr, sc, 2, 5, 44, 1)[0][0]
    _, _, _, po_type, po_total = _item(pot)
    po_ready = (pot[1] > 0).astype(jnp.int32)
    po_cooking = (timer > 0).astype(jnp.int32)
    po_good = ((pot[2:6].max() == po_total) & (po_type <= 1)).astype(jnp.int32)
    pot_obs = jnp.stack([po_total, po_type, po_cooking, po_ready, po_good]).astype(jnp.int32)
    pot_mem = jnp.where(pot_vis, pot_obs, mem[M1_POT:M1_POT + 5])
    p_total, p_type = pot_mem[0], pot_mem[1]
    p_cooking, p_ready, p_good = pot_mem[2] > 0, pot_mem[3] > 0, pot_mem[4] > 0
    p_idle = (~p_cooking) & (~p_ready)
    p_empty = p_idle & (p_total == 0)
    p_partial = p_idle & (p_total > 0) & (p_total < 3)
    pot_accepting = p_idle & (p_total < 3)
    pot_stale = jnp.where(p_partial & (p_total == mem[M1_LAST_TOTAL]), mem[M1_STALE] + 1, 0)
    pot_stale = pot_stale.astype(jnp.int32)

    # partner
    partner_vis, pr, pc, pface, pinv = _partner(obs, sr, sc)
    partner_cooked = pinv[1] > 0

    # counter slots (8 private + 2 shared)
    slots = []
    for k in range(NSLOT):
        blk, vis = _read(obs, sr, sc, SLOT_R_L[k], SLOT_C_L[k], 32, 6)
        e, j, ing, t, _ = _item(blk)
        content = jnp.where(e, -1, jnp.where(ing & (t <= 1), t, 9))
        slots.append(jnp.where(vis, content, mem[M1_SLOT + k]).astype(jnp.int32))
    slots = jnp.stack(slots)
    slot_c = jnp.asarray(SLOT_C_L, jnp.int32)
    slot_side = jnp.asarray(SLOT_SIDE_L, jnp.int32)
    slot_row4 = jnp.asarray(SLOT_ROW4_L, jnp.int32)
    slot_shared = jnp.asarray(SLOT_SHARED_L, jnp.int32) > 0
    private = ~slot_shared
    cache0 = ((slots == 0) & private).sum()
    cache1 = ((slots == 1) & private).sum()
    cache_rec = jnp.where(rec == 0, cache0, cache1)
    shared_ing = ((slots >= 0) & (slots <= 1) & slot_shared).any()
    partner_stages = jnp.where(shared_ing, 1, mem[M1_PARTNER_STAGES]).astype(jnp.int32)
    slot_free = (slots == -1) & private
    dist_pot = jnp.abs(slot_c - 5)
    dist_me = jnp.abs(slot_c - sc)
    BIG = 1000
    # restock dump: keep type-0 on the left, type-1 on the right, row-2 slots first
    restock_cost = dist_pot + 3 * (slot_side != jnp.clip(h_type, 0, 1)) + 4 * slot_row4
    k_dump_restock = jnp.argmin(jnp.where(slot_free, restock_cost, BIG))
    # flip dump: minimise walk me -> dump slot -> slot holding the new recipe type -> pot
    y_ok = slots == rec01
    pair_cost = (jnp.abs(slot_c - sc)[:, None] + jnp.abs(slot_c[:, None] - slot_c[None, :])
                 + jnp.abs(slot_c - 5)[None, :])
    pair_cost = jnp.where(slot_free[:, None] & y_ok[None, :], pair_cost, BIG)
    k_dump_flip = jnp.argmin(pair_cost.min(axis=1))
    any_rec_source = y_ok.any()
    flip_case = h_ing & (rec >= 0) & (rec != h_type) & any_rec_source & p_empty & (~partner_cooked)
    k_dump_ing = jnp.where(flip_case, k_dump_flip, k_dump_restock)
    k_dump_junk = jnp.argmin(jnp.where(slot_free, 10 - dist_pot, BIG))
    any_free = slot_free.any()

    # homogeneous wrong-type partial pot: do not complete it until patience runs out
    pot_wrong = p_partial & p_good & (rec >= 0) & (p_type != rec)
    wrong_timer = jnp.where(pot_wrong, mem[M1_WRONG] + 1, 0).astype(jnp.int32)
    wrong_timed_out = wrong_timer > WRONG_POT_PATIENCE

    need = jnp.where(p_partial & p_good, p_type, rec)
    need = jnp.clip(need, 0, 1)
    need_cost = dist_me + dist_pot
    k_get_need = jnp.argmin(jnp.where(slots == need, need_cost, BIG))
    need_cache = jnp.where(need == 0, cache0, cache1)
    shared_has_need = ((slots == need) & slot_shared).any()
    has_need_any = (slots == need).any()
    k_get_rec = jnp.argmin(jnp.where(y_ok & private, need_cost, BIG))

    # shared counters junk (visible only)
    l_junk = slots[8] == 9
    r_junk = slots[9] == 9

    g_read = _goal(3, jnp.clip(sc, 3, 7), A_UP, 0)
    g_insert = _goal(3, 5, A_UP, 1)
    g_wait = _goal(3, 5, A_UP, 0)
    g_dump_ing = _slot_goal(k_dump_ing)
    g_dump_junk = _slot_goal(k_dump_junk)
    g_need = jnp.where(has_need_any, _slot_goal(k_get_need), _far_goal(near_side, need))
    g_clean = jnp.where(l_junk, _goal(3, 4, A_UP, 1), _goal(3, 6, A_UP, 1))
    g_far0 = _far_goal(jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32))
    g_far1 = _far_goal(jnp.asarray(1, jnp.int32), jnp.asarray(1, jnp.int32))
    # the held seed for the next round comes from a private spare or the far supply so the
    # partner's staged shared-counter items stay available for the second ingredient
    g_hold_rec = jnp.where(cache_rec >= 2, _slot_goal(k_get_rec), _far_goal(near_side, rec01))

    seed_ok = p_empty & (rec == h_type) & (~partner_cooked)
    complete_ok = p_partial & ((~p_good) | (p_type == h_type))
    insert_ok = h_ing & pot_accepting & (seed_ok | complete_ok)
    # wrong-type pot timed out: drop what I hold so I can fetch the pot's type
    unblock_dump = h_ing & pot_wrong & wrong_timed_out & (h_type != p_type) & any_free
    wait_ok = h_ing & (rec == h_type)
    # seeding an empty pot, or helping to fill a partial one when a cheap source exists
    # (shared counter, or a private cache with a spare) or the partner is not progressing
    seed_load = p_empty & (~partner_cooked)
    help_cheap = shared_has_need | (need_cache >= 2)
    help_load = p_partial & (help_cheap | (pot_stale > HELP_PATIENCE))
    load_ok = h_empty & pot_accepting & (seed_load | help_load) & \
        (~(pot_wrong & (~wrong_timed_out)))
    stock_needed = h_empty & (partner_stages == 0)

    conds = [
        rec < 0,
        h_junk & any_free,
        h_junk,
        insert_ok,
        unblock_dump,
        wait_ok,
        h_ing & any_free,
        h_ing,
        load_ok,
        h_empty & (l_junk | r_junk),
        stock_needed & (cache0 == 0),
        stock_needed & (cache1 == 0),
        h_empty,
    ]
    goals = [
        g_read, g_dump_junk, g_wait, g_insert, g_dump_ing, g_wait, g_dump_ing, g_wait,
        g_need, g_clean, g_far0, g_far1, g_hold_rec,
    ]
    gids = [G1_READ, G1_DUMP_JUNK, G1_WAITPOT, G1_INSERT, G1_DUMP_ING, G1_WAITPOT, G1_DUMP_ING,
            G1_WAITPOT, G1_GET, G1_CLEAN, G1_FAR, G1_FAR, G1_FAR]
    goal = _select(conds, goals, g_wait)
    gid = _select(conds, [jnp.asarray(g, jnp.int32) for g in gids], jnp.asarray(G1_WAITPOT, jnp.int32))

    tr, tc = goal[0], goal[1]
    horiz = jnp.where(sc < tc, A_RIGHT, A_LEFT)
    vert = jnp.where(sr < tr, A_DOWN, A_UP)
    nav = jnp.where(sc != tc, jnp.where(sr == 4, A_UP, horiz), vert)
    action = _execute(obs, sr, sc, goal, nav)

    new_mem = jnp.concatenate([rec[None], slots, pot_mem, wrong_timer[None], pot_stale[None],
                               p_total[None], partner_stages[None]]).astype(jnp.int32)
    trace = {'goal': gid.astype(jnp.int32), 'row': sr.astype(jnp.int32), 'col': sc.astype(jnp.int32),
             'rec': rec, 'cache0': cache0.astype(jnp.int32), 'cache1': cache1.astype(jnp.int32)}
    return action, new_mem, trace
