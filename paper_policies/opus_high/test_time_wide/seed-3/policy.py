"""Overcooked-V2 `test_time_wide` joint policy.

Layout facts that drive the design (see context/environment/README.md):

* The board is split into two disconnected rooms.  agent_0's room (rows 1-2)
  owns every resource needed to cook and deliver: ingredient 0 / 1 supplies,
  the plate pile, the delivery station and both pot access cells.  agent_1's
  room (rows 4-5) only touches ingredient 3 (never part of a recipe) and the
  always-on recipe indicator at (6, 2).
* The recipe indicator is visible from *every* cell of the bottom room and from
  *no* cell of the top room.  So the cook (top seat) cannot see the order and
  the bottom seat cannot cook.  The only cooperative channel is what the bottom
  seat does with its body, seen by the cook whenever the cook stands in row 2.

Design:

* agent_1 is a pure, harmless signaller.  It never picks anything up (so it can
  never spoil a pot) and parks under the pot whose side encodes the requested
  ingredient: left pot (3, 1) for ingredient 0, right pot (3, 3) for
  ingredient 1, facing the pot.  "Left = first ingredient, right = second" is
  the most natural reading of a two-pot / two-recipe board, and the pose is
  also legible to a partner that only looks at column, at side-of-room, or at
  which pot the partner is pointing at.
* agent_0 is a complete solo cook running a *two-pot pipeline*.  Pot (3, 1) is
  dedicated to ingredient 0 (loaded from supply (1, 0) via cell (1, 1)) and pot
  (3, 3) to ingredient 1 (loaded from supply (2, 5) via cell (2, 4)).  Cooked
  food never spoils, so keeping one standing dish of each recipe means a
  delivery usually costs only plate + scoop + deliver, and the 20-step cook of
  the pot just emptied overlaps with the next delivery instead of stalling it.
  This is a pure throughput change: it needs no agreement with the partner and
  therefore helps against every partner type.
* agent_0 never delivers a dish that does not match its own committed recipe
  belief, so wrong-dish penalties can only come from a wrong belief, never from
  a mixed pot.  A pot that a partner contaminated is filled up, cooked, scooped
  and dumped on a counter, which clears it for reuse.
* agent_0's belief is learnt, not assumed.  A recipe is resampled only after a
  *correct* delivery, so if the partner's signal changes after we delivered
  ingredient X, that delivery was correct: the recipe was X and the signal
  value shown before the change was this partner's name for X.  One such event
  fixes the whole convention, including a partner whose left/right sense is the
  opposite of ours.  Until such an event we alternate the ingredient between
  deliveries, which is the optimal uninformed play here (a wrong delivery does
  not resample the recipe, so the next, flipped, dish is guaranteed correct).
  Repeated deliveries with no signal change are Bayesian evidence that the
  learnt polarity is wrong, and after four of them the polarity is discarded
  and relearnt.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'two_room_pipelined_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

# ---------------------------------------------------------------- constants --

A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5

# obs channel carrying "self is facing <action direction>"
_FACE_CH = (3, 2, 4, 1)  # right, down, left, up

_H, _W = 7, 6
_WALLS = ((0, 0), (0, 1), (0, 4), (0, 5), (3, 0), (3, 2), (3, 4), (3, 5),
          (5, 0), (5, 5), (6, 0), (6, 1), (6, 3), (6, 4), (6, 5))


def _static_map():
    grid = [[[0] * 10 for _ in range(_W)] for _ in range(_H)]
    for (r, c) in _WALLS:
        grid[r][c][0] = 1
    grid[0][2][1] = 1                      # delivery / goal
    for (r, c) in ((3, 1), (3, 3)):
        grid[r][c][2] = 1                  # pots
    grid[6][2][3] = 1                      # recipe indicator
    grid[0][3][5] = 1                      # plate pile
    for (r, c) in ((1, 0), (1, 5)):
        grid[r][c][6] = 1                  # ingredient 0 supply
    for (r, c) in ((2, 0), (2, 5)):
        grid[r][c][7] = 1                  # ingredient 1 supply
    for (r, c) in ((4, 0), (4, 5)):
        grid[r][c][9] = 1                  # ingredient 3 supply
    return grid


def _padded_crops(cells):
    grid = _static_map()
    pad = [[[0] * 10 for _ in range(_W + 4)] for _ in range(_H + 4)]
    for r in range(_H):
        for c in range(_W):
            pad[r + 2][c + 2] = grid[r][c]
    crops = []
    for (r, c) in cells:
        crops.append([[pad[r + i][c + j] for j in range(5)] for i in range(5)])
    return crops


_TOP_CELLS = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4))
_BOT_CELLS = ((4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4))

# ---- agent_0 work stations -------------------------------------------------
# pot p lives at (3, _POT_COL[p]) and is served from (2, _POT_COL[p]) facing
# DOWN; pot p is dedicated to ingredient p.
_POT_COL = (1, 3)
# ingredient k is picked up from _PICK cell facing _PICK_F
_PICK_R = (1, 2)
_PICK_C = (1, 4)
_PICK_F = (A_LEFT, A_RIGHT)
# free counters, as (stand row, stand col, facing, counter row, counter col).
# (3, 2) comes first: it is in view from every cell of the room and sits one
# step from both pot access cells, so it doubles as the plate parking spot.
_CNT_SR = (2, 1, 2, 1)
_CNT_SC = (2, 1, 4, 4)
_CNT_SF = (A_DOWN, A_UP, A_DOWN, A_UP)
_CNT_GR = (3, 0, 3, 0)
_CNT_GC = (2, 1, 4, 4)

_STABLE_MIN = 3           # observed steps on one side before it is a signal
_LMAX = 6                 # clip for the polarity log-odds (base-2)
_L_PRIOR = 1              # mild prior that the partner's left side means "0"
_MAX_CLEAR = 3            # spoiled pots emptied onto counters before giving up
_BLIND_CAP = 12           # steps spent waiting for a signal we expect to return

# ------------------------------------------------------------------ helpers --


def _localize(obs, cells):
    """Recover own global (row, col) from the static observation channels."""
    crops = jnp.asarray(_padded_crops(cells), dtype=jnp.int32)
    rows = jnp.asarray([p[0] for p in cells], dtype=jnp.int32)
    cols = jnp.asarray([p[1] for p in cells], dtype=jnp.int32)
    static = obs[:, :, 22:32]
    diff = jnp.sum(jnp.abs(crops - static[None, :, :, :]), axis=(1, 2, 3))
    idx = jnp.argmin(diff)
    return rows[idx], cols[idx]


def _read_cell(obs, r, c, gr, gc):
    """Observation vector of global cell (gr, gc); zeros when out of view."""
    lr = gr - r + 2
    lc = gc - c + 2
    valid = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    vec = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)]
    return valid, jnp.where(valid, vec, jnp.zeros_like(vec))


def _nav(r, c, tr, tc):
    """Horizontal-then-vertical step inside an open rectangular room."""
    return jnp.where(c < tc, A_RIGHT,
                     jnp.where(c > tc, A_LEFT,
                               jnp.where(r < tr, A_DOWN,
                                         jnp.where(r > tr, A_UP, A_STAY))))


def _goto(obs, r, c, tr, tc, face, check_face):
    at = (r == tr) & (c == tc)
    face_ch = jnp.asarray(_FACE_CH, dtype=jnp.int32)[face]
    facing_ok = jnp.logical_or(obs[2, 2, face_ch] > 0,
                               jnp.logical_not(jnp.asarray(check_face)))
    ready = at & facing_ok
    action = jnp.where(at, jnp.where(facing_ok, A_STAY, face),
                       _nav(r, c, tr, tc))
    return action, ready


def _partner_side(obs, r, c):
    """Which side of its room the partner occupies: -1 unseen/neutral, 0, 1.

    Position dominates; the facing direction only breaks a central tie.
    """
    ppos = obs[:, :, 11]
    visible = jnp.sum(ppos) > 0
    flat = jnp.argmax(ppos.reshape(-1))
    pi = flat // 5
    pj = flat % 5
    pgc = c + pj - 2
    pgr = r + pi - 2
    shown = visible & (pgr >= 3)
    face_left = obs[pi, pj, 15] > 0
    face_right = obs[pi, pj, 14] > 0
    by_face = jnp.where(face_left, 0, jnp.where(face_right, 1, -1))
    side = jnp.where(pgc <= 1, 0, jnp.where(pgc >= 3, 1, by_face))
    return jnp.where(shown, side, -1).astype(jnp.int32)


# ------------------------------------------------------------------ agent_0 --

# memory layout
_M_WANT, _M_LASTING, _M_LASTSIG, _M_L = 0, 1, 2, 3
_M_UPD, _M_SPARE, _M_SIDE, _M_STABLE = 4, 5, 6, 7
_M_A0, _M_B0, _M_N0, _M_R0, _M_T0 = 8, 9, 10, 11, 12
_M_A1, _M_B1, _M_N1, _M_R1, _M_T1 = 13, 14, 15, 16, 17
_M_NDUMP, _M_NDELIV, _M_TICK, _M_BLIND = 18, 19, 20, 21
_M_EVER = 22
_MEM0 = 23


def init_agent_0():
    mem = [0] * _MEM0
    mem[_M_LASTING] = -1
    mem[_M_LASTSIG] = -1
    mem[_M_L] = _L_PRIOR
    mem[_M_SIDE] = -1
    return jnp.asarray(mem, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, init_agent_0(), memory)

    r, c = _localize(obs, _TOP_CELLS)

    inv_plate = obs[2, 2, 5]
    inv_cooked = obs[2, 2, 6]
    held = obs[2, 2, 7:11]
    held_sum = jnp.sum(held)
    holding_dish = inv_cooked > 0
    holding_plate = (inv_plate > 0) & (inv_cooked == 0)
    holding_ing = (inv_cooked == 0) & (inv_plate == 0) & (held_sum > 0)
    hands_empty = ~(holding_dish | holding_plate | holding_ing)

    # ---- partner signal ---------------------------------------------------
    # A side counts as a signal only once the partner has held it for a few
    # consecutive observed steps, so that walking around reads as "no signal"
    # and a partner repositioning after a recipe change is never half-read.
    side_obs = _partner_side(obs, r, c)
    seen_now = side_obs >= 0
    same_side = seen_now & (side_obs == mem[_M_SIDE])
    stable = jnp.where(~seen_now, mem[_M_STABLE],
                       jnp.where(same_side, mem[_M_STABLE] + 1, 1))
    side = jnp.where(seen_now, side_obs, mem[_M_SIDE])
    sig = jnp.where((stable >= _STABLE_MIN) & (side >= 0), side, -1)

    # ---- pot bookkeeping ---------------------------------------------------
    # Pot (3, 3) is visible from every cell of the room; pot (3, 1) is hidden
    # only while we stand in column 4, so a decremented timer carries us over.
    def pot_state(p, ma, mb, mn, mr, mt):
        ok, v = _read_cell(obs, r, c, 3, _POT_COL[p])
        tmr_m = jnp.maximum(mem[mt] - 1, 0)
        rdy_m = jnp.maximum(mem[mr],
                            ((mem[mt] > 0) & (tmr_m == 0)).astype(jnp.int32))
        return (jnp.where(ok, v[34], mem[ma]),
                jnp.where(ok, v[35], mem[mb]),
                jnp.where(ok, jnp.sum(v[34:38]), mem[mn]),
                jnp.where(ok, (v[33] > 0).astype(jnp.int32), rdy_m),
                jnp.where(ok, v[44], tmr_m))

    a0, b0, n0, rd0, tm0 = pot_state(0, _M_A0, _M_B0, _M_N0, _M_R0, _M_T0)
    a1, b1, n1, rd1, tm1 = pot_state(1, _M_A1, _M_B1, _M_N1, _M_R1, _M_T1)

    def cnt_of(p, k):
        return jnp.where(k == 0, jnp.where(p == 0, a0, a1),
                         jnp.where(p == 0, b0, b1))

    zero = jnp.asarray(0, dtype=jnp.int32)
    one_i = jnp.asarray(1, dtype=jnp.int32)

    # a pot is usable for ingredient k only while everything already in it is
    # ingredient k; anything the partner dropped in makes it unusable for both,
    # which is how a spoiled pot is abandoned instead of ground on
    def loadable(p, k):
        rd = jnp.where(p == 0, rd0, rd1)
        tm = jnp.where(p == 0, tm0, tm1)
        n = jnp.where(p == 0, n0, n1)
        return (rd == 0) & (tm == 0) & (n < 3) & (cnt_of(p, k) == n)

    def committed(p, k):
        n = jnp.where(p == 0, n0, n1)
        return (n > 0) & (cnt_of(p, k) == n)

    pure0 = (n0 == 3) & ((a0 == 3) | (b0 == 3))
    pure1 = (n1 == 3) & ((a1 == 3) | (b1 == 3))
    badr0 = (rd0 > 0) & ~pure0
    badr1 = (rd1 > 0) & ~pure1

    # ---- recipe belief -----------------------------------------------------
    # `L` is the base-2 log-odds that the partner's signal value *is* the
    # requested ingredient id (polarity 0) rather than its complement.  Exactly
    # one observation per delivery interval carries information:
    #
    #   * the signal changed  -> the recipe was resampled -> the last delivery
    #     was correct -> the polarity is proven to be last_ing XOR last_sig;
    #   * the signal did not change -> either the recipe resampled to the same
    #     value (probability 1/2 if the polarity we used was right) or the last
    #     delivery was wrong (probability 1 if it was not), which is exactly one
    #     bit of evidence against the polarity that made that delivery correct.
    #
    # With a flat L this reproduces blind alternation, which is the optimal
    # uninformed play here, and a single change event pins the convention down
    # whatever the partner's left/right sense is.
    have_prev = (mem[_M_LASTING] >= 0) & (mem[_M_LASTSIG] >= 0) & (sig >= 0)
    do_upd = have_prev & (mem[_M_UPD] == 0)
    changed = sig != mem[_M_LASTSIG]
    implied = jnp.bitwise_xor(mem[_M_LASTING], mem[_M_LASTSIG])
    proven = jnp.where(implied == 0, _LMAX, -_LMAX)
    step_l = jnp.where(changed, proven - mem[_M_L],
                       jnp.where(implied == 1, 1, -1))
    lodds = jnp.where(do_upd,
                      jnp.clip(mem[_M_L] + step_l, -_LMAX, _LMAX), mem[_M_L])
    upd = jnp.where(do_upd, 1, mem[_M_UPD])

    pol_eff = jnp.where(lodds > 0, 0, 1)
    # with no signal at all, flip the ingredient after every delivery: a wrong
    # delivery leaves the recipe in place, so the flipped dish is then certain
    want_new = jnp.where(sig >= 0, jnp.bitwise_xor(sig, pol_eff),
                         jnp.where(mem[_M_LASTING] >= 0,
                                   1 - mem[_M_LASTING], 0))
    want = jnp.where(holding_dish, mem[_M_WANT], want_new)

    # ---- counters ----------------------------------------------------------
    cgr = jnp.asarray(_CNT_GR, dtype=jnp.int32)
    cgc = jnp.asarray(_CNT_GC, dtype=jnp.int32)
    c_taken = []
    c_plate = []
    c_dish = [[], []]
    for i in range(4):
        ok_i, v_i = _read_cell(obs, r, c, cgr[i], cgc[i])
        tot_i = jnp.sum(v_i[34:38])
        c_taken.append(ok_i & (jnp.sum(v_i[32:38]) > 0))
        c_plate.append(ok_i & (v_i[32] > 0) & (v_i[33] == 0) & (tot_i == 0))
        for k in range(2):
            c_dish[k].append(ok_i & (v_i[32] > 0) & (v_i[33] > 0)
                             & (tot_i == 3) & (v_i[34 + k] == 3))
    c_taken = jnp.stack(c_taken)
    c_plate = jnp.stack(c_plate)
    c_dish = jnp.stack([jnp.stack(c_dish[0]), jnp.stack(c_dish[1])])
    # "free" is optimistic for a counter we cannot see; (3, 2) is always in view
    stash_i = jnp.argmin(c_taken.astype(jnp.int32))
    any_free = ~jnp.all(c_taken)
    have_stash_plate = jnp.any(c_plate)
    plate_i = jnp.argmax(c_plate.astype(jnp.int32))

    # Right after a delivery the partner needs a few steps to re-pose, and the
    # cook cannot see into the other room from row 1 at all.  Committing to a
    # dish while the signal is missing throws away the information, so instead
    # do the recipe-independent half of the work and, failing that, stand where
    # the partner is visible.  A partner that never shows a stable pose never
    # sets _M_EVER, so this never costs anything against a silent one.
    ever = jnp.maximum(mem[_M_EVER], (sig >= 0).astype(jnp.int32))
    blind = jnp.where(sig >= 0, 0, mem[_M_BLIND] + 1)
    unsure = (sig < 0) & (ever > 0) & (blind < _BLIND_CAP)

    # ---- job selection -----------------------------------------------------
    other = 1 - want
    serve0 = (rd0 > 0) & (n0 == 3) & (cnt_of(zero, want) == 3)
    serve1 = (rd1 > 0) & (n1 == 3) & (cnt_of(one_i, want) == 3)
    good_w = serve0 | serve1
    c_serve = c_dish[want]
    ctr_serve = jnp.any(c_serve)
    ctr_i = jnp.argmax(c_serve.astype(jnp.int32))
    serve_raw = good_w | ctr_serve
    # empty-handed errand: a buffered dish of the wanted recipe beats a plate
    take_ctr = ctr_serve & ~unsure

    # A half-filled pot is dead weight, so always finish one before opening the
    # other; otherwise keep exactly one pot per recipe alive and start the pot
    # whose ingredient is not covered yet.
    p0k = jnp.where(a0 == n0, 0, jnp.where(b0 == n0, 1, -1))
    p1k = jnp.where(a1 == n1, 0, jnp.where(b1 == n1, 1, -1))
    fin0 = (rd0 == 0) & (tm0 == 0) & (n0 > 0) & (n0 < 3) & (p0k >= 0)
    fin1 = (rd1 == 0) & (tm1 == 0) & (n1 > 0) & (n1 < 3) & (p1k >= 0)
    have_w = committed(zero, want) | committed(one_i, want)
    have_o = committed(zero, other) | committed(one_i, other)
    fresh_k = jnp.where(~have_w, want, jnp.where(~have_o, other, want))
    target_k = jnp.where(fin0, p0k, jnp.where(fin1, p1k, fresh_k))
    lk0 = loadable(zero, target_k)
    lk1 = loadable(one_i, target_k)
    any_load = lk0 | lk1
    # pot p is cheapest to load with ingredient p, so prefer that pairing
    load_pot = jnp.where(fin0, 0,
                         jnp.where(fin1, 1,
                                   jnp.where(target_k == 0,
                                             jnp.where(lk0, 0, 1),
                                             jnp.where(lk1, 1, 0))))

    # a ready pot that is not a clean dish of one ingredient blocks its pot;
    # clear it onto a free counter, but only a few times
    any_badr = badr0 | badr1
    can_clear = any_badr & any_free & (mem[_M_NDUMP] < _MAX_CLEAR)
    clear_p = jnp.where(badr0, 0, 1)

    #   0 deliver | 1 dump | 2 scoop | 3 add | 4 pick | 5 plate | 6 wait
    #   7 park the plate on a counter | 8 watch the other room
    dish_ok = holding_dish & (jnp.take(held, want) == 3) & (held_sum == 3)
    do_deliver = holding_dish & dish_ok

    held_k = jnp.clip(jnp.argmax(held).astype(jnp.int32), 0, 3)
    hk = jnp.minimum(held_k, 1)
    ah0 = loadable(zero, held_k)
    ah1 = loadable(one_i, held_k)
    addable = holding_ing & (held_k < 2) & (ah0 | ah1)
    add_pot = jnp.where(fin0 & (p0k == held_k) & ah0, 0,
                        jnp.where(fin1 & (p1k == held_k) & ah1, 1,
                                  jnp.where(hk == 0, jnp.where(ah0, 0, 1),
                                            jnp.where(ah1, 1, 0))))

    # Proactively emptying the idle pot onto a counter was measured to be a
    # wash: it removes the cook-wait but costs the same in park/unpark labour.
    # A dish that did end up on a counter is still worth collecting, which is
    # what `take_ctr` below does.
    serve_now = good_w & ~unsure
    scoop_p = jnp.where(serve_now, jnp.where(serve0, 0, 1), clear_p)
    can_scoop = serve_now | can_clear

    pick_k = target_k
    prefer_load = any_load & (~serve_raw | unsure)
    # while there is nothing to do, stand at the pot that will serve `want`
    wait_pot = jnp.where(serve0 | committed(zero, want), 0,
                         jnp.where(serve1 | committed(one_i, want), 1, want))

    plate_mode = jnp.where(
        can_scoop, 2,
        jnp.where((any_load | take_ctr) & any_free, 7,
                  jnp.where(unsure, 8, 6)))
    mode = jnp.where(
        holding_dish, jnp.where(do_deliver, 0, 1),
        jnp.where(holding_plate, plate_mode,
                  jnp.where(holding_ing, jnp.where(addable, 3, 1),
                            jnp.where(prefer_load, 4, 5))))

    # ---- targets -----------------------------------------------------------
    csr = jnp.asarray(_CNT_SR, dtype=jnp.int32)
    csc = jnp.asarray(_CNT_SC, dtype=jnp.int32)
    csf = jnp.asarray(_CNT_SF, dtype=jnp.int32)
    dump_i = stash_i          # always aim at the first counter that looks free

    pot_cols = jnp.asarray(_POT_COL, dtype=jnp.int32)
    pick_r = jnp.asarray(_PICK_R, dtype=jnp.int32)[pick_k]
    pick_c = jnp.asarray(_PICK_C, dtype=jnp.int32)[pick_k]
    pick_f = jnp.asarray(_PICK_F, dtype=jnp.int32)[pick_k]
    src_i = jnp.where(take_ctr, ctr_i, plate_i)
    use_ctr = take_ctr | have_stash_plate
    plate_r = jnp.where(use_ctr, csr[src_i], 1)
    plate_c = jnp.where(use_ctr, csc[src_i], 3)
    plate_f = jnp.where(use_ctr, csf[src_i], A_UP)
    one = jnp.asarray(1, dtype=jnp.int32)

    # (2, 2) sees every cell the partner can stand on in row 4
    tr = jnp.stack([one, csr[dump_i], one * 2, one * 2, pick_r, plate_r,
                    one * 2, csr[stash_i], one * 2])[mode]
    tc = jnp.stack([one * 2, csc[dump_i], pot_cols[scoop_p],
                    pot_cols[add_pot], pick_c, plate_c, pot_cols[wait_pot],
                    csc[stash_i], one * 2])[mode]
    tf = jnp.stack([one * A_UP, csf[dump_i], one * A_DOWN, one * A_DOWN,
                    pick_f, plate_f, one * A_DOWN, csf[stash_i],
                    one * A_DOWN])[mode]
    want_int = jnp.asarray([1, 1, 1, 1, 1, 1, 0, 1, 0],
                           dtype=jnp.int32)[mode] > 0

    action, ready = _goto(obs, r, c, tr, tc, tf, True)
    action = jnp.where(ready & want_int, A_INT, action)

    # ---- memory update -----------------------------------------------------
    acted = ready & want_int
    did_deliver = acted & (mode == 0)
    did_dump = acted & (mode == 1)
    did_scoop = acted & (mode == 2)
    did_add = acted & (mode == 3)

    mem = mem.at[_M_WANT].set(want)
    mem = mem.at[_M_L].set(lodds)
    mem = mem.at[_M_UPD].set(jnp.where(did_deliver, 0, upd))
    mem = mem.at[_M_LASTING].set(jnp.where(did_deliver, want, mem[_M_LASTING]))
    mem = mem.at[_M_LASTSIG].set(jnp.where(did_deliver, sig, mem[_M_LASTSIG]))
    mem = mem.at[_M_SIDE].set(jnp.where(did_deliver, -1, side))
    mem = mem.at[_M_STABLE].set(jnp.where(did_deliver, 0, stable))
    mem = mem.at[_M_BLIND].set(jnp.where(did_deliver, 0, blind))
    mem = mem.at[_M_EVER].set(ever)

    # our own interactions change a pot before the next observation arrives
    add0 = did_add & (add_pot == 0)
    add1 = did_add & (add_pot == 1)
    sc0 = did_scoop & (scoop_p == 0)
    sc1 = did_scoop & (scoop_p == 1)
    inc_a = (held_k == 0).astype(jnp.int32)
    inc_b = (held_k == 1).astype(jnp.int32)
    a0n = jnp.where(add0, a0 + inc_a, jnp.where(sc0, 0, a0))
    b0n = jnp.where(add0, b0 + inc_b, jnp.where(sc0, 0, b0))
    n0n = jnp.where(add0, n0 + 1, jnp.where(sc0, 0, n0))
    a1n = jnp.where(add1, a1 + inc_a, jnp.where(sc1, 0, a1))
    b1n = jnp.where(add1, b1 + inc_b, jnp.where(sc1, 0, b1))
    n1n = jnp.where(add1, n1 + 1, jnp.where(sc1, 0, n1))
    t0n = jnp.where(add0 & (n0 == 2), 19, tm0)
    t1n = jnp.where(add1 & (n1 == 2), 19, tm1)
    r0n = jnp.where(sc0, 0, rd0)
    r1n = jnp.where(sc1, 0, rd1)

    mem = mem.at[_M_A0].set(a0n)
    mem = mem.at[_M_B0].set(b0n)
    mem = mem.at[_M_N0].set(n0n)
    mem = mem.at[_M_R0].set(r0n)
    mem = mem.at[_M_T0].set(t0n)
    mem = mem.at[_M_A1].set(a1n)
    mem = mem.at[_M_B1].set(b1n)
    mem = mem.at[_M_N1].set(n1n)
    mem = mem.at[_M_R1].set(r1n)
    mem = mem.at[_M_T1].set(t1n)

    # only a genuinely unservable dish counts against the clearing budget;
    # parking a clean dish on a counter is ordinary pipeline work
    bad_dish = ~((held_sum == 3) & ((held[0] == 3) | (held[1] == 3)))
    mem = mem.at[_M_NDELIV].set(mem[_M_NDELIV] + did_deliver.astype(jnp.int32))
    mem = mem.at[_M_NDUMP].set(
        mem[_M_NDUMP] + (did_dump & bad_dish).astype(jnp.int32))
    mem = mem.at[_M_TICK].set(mem[_M_TICK] + 1)

    trace = {
        'pos': jnp.stack([r, c]).astype(jnp.int32),
        'mode': mode.astype(jnp.int32),
        'want': want.astype(jnp.int32),
        'sig': sig.astype(jnp.int32),
        'pol': pol_eff.astype(jnp.int32),
        'stale': lodds.astype(jnp.int32),
        'ndeliv': mem[_M_NDELIV].astype(jnp.int32),
        'pot0': jnp.stack([a0n, b0n, n0n, r0n, t0n]).astype(jnp.int32),
        'pot1': jnp.stack([a1n, b1n, n1n, r1n, t1n]).astype(jnp.int32),
    }
    return jnp.asarray(action, dtype=jnp.int32), mem, trace


# ------------------------------------------------------------------ agent_1 --


def init_agent_1():
    return jnp.asarray([0, 0], dtype=jnp.int32)  # [step, last recipe side]


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, init_agent_1(), memory)

    r, c = _localize(obs, _BOT_CELLS)

    # the always-on recipe indicator at (6, 2) is visible from the whole room
    ind_ok, ind = _read_cell(obs, r, c, 6, 2)
    counts = ind[40:44]
    shown = ind_ok & (jnp.sum(counts) > 0)
    side = jnp.where(counts[1] > counts[0], 1, 0)
    side = jnp.where(shown, side, mem[1])

    # park under the pot that encodes the requested ingredient, facing it
    tr = 4
    tc = jnp.where(side == 0, 1, 3)
    action, ready = _goto(obs, r, c, tr, tc, jnp.asarray(A_UP, jnp.int32),
                          True)
    action = jnp.where(ready, A_STAY, action)

    mem = mem.at[0].set(mem[0] + 1)
    mem = mem.at[1].set(side.astype(jnp.int32))
    trace = {
        'pos': jnp.stack([r, c]).astype(jnp.int32),
        'side': side.astype(jnp.int32),
        'shown': shown.astype(jnp.int32),
    }
    return jnp.asarray(action, dtype=jnp.int32), mem, trace
