"""Overcooked-V2 demo_cook_simple joint policy.

Layout facts that drive the design (both seats know the layout up front):

    WWWWWR2W0WW
    0      W  B
    W     APA X
    1      W  B
    WWWWWR2W1WW

The two rooms are disconnected.  Seat 0 lives in the left room (rows 1-3,
cols 1-6) and is the *only* seat that can ever see a recipe indicator
(R at (0,5)/(4,5)).  Seat 1 lives in the right room (rows 1-3, cols 8-9)
and is the *only* seat that can reach plates (1,10)/(3,10) and the delivery
station (2,10).  The single pot (2,7) is reachable from both rooms and is
visible from every right-room cell.

Coordination therefore uses no private convention at all: seat 0 reads the
recipe and deposits one unit of the correct raw ingredient into the shared
pot.  The pot contents are the message - any observer can read them.  Seat 1
never *initiates* the pot (it cannot see a recipe indicator, so guessing is a
zero-expectation gamble); it only amplifies a pot that already holds a pure
raw stack, then plates the cooked result and delivers it when the dish is a
pure triple of ingredient 0 or 1 (the only shapes any recipe can take here).
Anything else is shelved on a counter instead of delivered, so a mis-cooked
pot costs time but never -20.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'pot_signal_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# action ids and facing ids (they use different encodings)
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3

GRID = ["WWWWWR2W0WW",
        "0      W  B",
        "W     APA X",
        "1      W  B",
        "WWWWWR2W1WW"]
# order matches observation channels 20..28
SYMBOLS = "WXPRLB012"

BIG = 10000

# reachable cells per seat, as a 5x11 localisation penalty mask
PEN_0_LIST = [[BIG] * 11,
              [BIG, 0, 0, 0, 0, 0, 0, BIG, BIG, BIG, BIG],
              [BIG, 0, 0, 0, 0, 0, 0, BIG, BIG, BIG, BIG],
              [BIG, 0, 0, 0, 0, 0, 0, BIG, BIG, BIG, BIG],
              [BIG] * 11]
PEN_1_LIST = [[BIG] * 11,
              [BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, 0, 0, BIG],
              [BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, 0, 0, BIG],
              [BIG, BIG, BIG, BIG, BIG, BIG, BIG, BIG, 0, 0, BIG],
              [BIG] * 11]

# ---- seat 0 larder counters (unreachable and invisible from the right room)
S0_CELL = [[0, 4], [4, 4], [0, 1], [4, 1]]
S0_STAND_R = [1, 3, 1, 3]
S0_STAND_C = [4, 4, 1, 1]
S0_TURN = [UP, DOWN, UP, DOWN]
S0_FDIR = [F_UP, F_DOWN, F_UP, F_DOWN]
S0_TYPE = [0, 1, 0, 1]
S0_FAR = [0, 0, 200, 200]
POT0_R, POT0_C = 2, 6

# ---- seat 1 shelves.  Only the two private counters (0,9)/(4,9) are used:
# (1,7)/(3,7) are the shared hand-off counters and are left untouched so that
# anything seen there is unambiguously the partner's doing.
S1_CELL = [[0, 9], [4, 9]]
S1_STAND_R = [1, 3]
S1_STAND_C = [9, 9]
S1_TURN = [UP, DOWN]
S1_FDIR = [F_UP, F_DOWN]
S1_FAR = [0, 0]
# retrieving a shelved unit is preferred to walking to a supply pile, so that
# shelves drain instead of silently filling up over an episode
S1_SHELF_BONUS = [-6, -6]
POT1_R, POT1_C = 2, 8

# how long the pot must sit empty and idle before seat 1 will believe an
# ingredient parked on a shared hand-off counter is a recipe signal.  The bar
# is high the first time, because a lone ingredient on a counter is much more
# likely to be a partner's spare than a deliberate message; once the channel
# has demonstrably been used, a short wait is enough.
DESPERATE_STEPS = 40
DESPERATE_STEPS_SEEN = 6
# steps seat 0 waits after the pot is scooped, so that the recipe resample
# triggered by the imminent delivery is observed before the next deposit
HOLD_OFF_STEPS = 3
TAINT_STEPS = 12

INIT_0_LIST = [-1, 0, -1, 0, -1, -1, -1, -1, 0]
# shelf (0,9), shelf (4,9), delivery-attempt flag, taint, taint timer,
# empty run, hand-off-channel-confirmed flag
INIT_1_LIST = [-1, -1, 0, 0, 0, 0, 0]


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _padded_static():
    """Zero-padded static-object map, shape (9, 15, 9), channels 20..28."""
    pad = [[[0] * 9 for _ in range(15)] for _ in range(9)]
    for r in range(5):
        for c in range(11):
            k = SYMBOLS.find(GRID[r][c])
            if k >= 0:
                pad[r + 2][c + 2][k] = 1
    return jnp.asarray(pad, dtype=jnp.int32)


def _localize(obs, penalty_list):
    """Recover the global cell by matching the 5x5 static crop.

    The static crop is an exact fingerprint of position for every reachable
    cell (verified against the live environment), so this never drifts.
    """
    pad = _padded_static()
    view = obs[:, :, 20:29]
    dist = jnp.asarray(penalty_list, dtype=jnp.int32)
    for i in range(5):
        for j in range(5):
            dist = dist + jnp.abs(
                pad[i:i + 5, j:j + 11, :] - view[i, j][None, None, :]).sum(-1)
    k = jnp.argmin(dist.reshape(-1)).astype(jnp.int32)
    return k // 11, k % 11


def _cell(obs, row, col, gr, gc):
    """Channel vector at global (gr, gc) plus a visibility flag."""
    i = gr - row + 2
    j = gc - col + 2
    inside = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    return inside, obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4)]


def _item_code(vec):
    """-1 empty, 0/1/2 a single raw ingredient, 3 anything else."""
    plate, cooked = vec[29], vec[30]
    cnt = vec[31:34]
    total = cnt.sum()
    bare = (plate == 0) & (cooked == 0)
    single = ((cnt > 0).sum() == 1) & (total == 1)
    return jnp.where(bare & (total == 0), -1,
                     jnp.where(bare & single, jnp.argmax(cnt), 3)).astype(jnp.int32)


def _pot_state(vec):
    cnt = vec[31:34]
    total = cnt.sum().astype(jnp.int32)
    kind = jnp.where(total == 0, -1,
                     jnp.where((cnt > 0).sum() == 1, jnp.argmax(cnt), 3)).astype(jnp.int32)
    cooked = vec[30].astype(jnp.int32)
    busy = ((vec[39] > 0) | (cooked > 0)).astype(jnp.int32)
    return total, kind, busy, cooked


def _held(obs):
    vec = obs[2, 2]
    plate, cooked = vec[5], vec[6]
    cnt = vec[7:10]
    total = cnt.sum()
    bare = (plate == 0) & (cooked == 0)
    code = jnp.where(bare & (total == 0), -1,
                     jnp.where(bare & (total > 0), jnp.argmax(cnt), 3)).astype(jnp.int32)
    return code, plate.astype(jnp.int32), cooked.astype(jnp.int32), cnt.astype(jnp.int32)


def _step_action(row, col, face, sr, sc, turn, fdir, do_interact):
    """Walk to (sr, sc), turn to ``fdir``, then INTERACT (or STAY).

    Every stand cell used here has an object in direction ``fdir``, so the
    turn action is a blocked move: it re-orients without displacing us.  Both
    rooms are open rectangles, so axis-by-axis movement is a shortest path and
    never collides -- the two seats can never share a room.
    """
    dr = sr - row
    dc = sc - col
    mv = jnp.where(dr > 0, DOWN, UP)
    mh = jnp.where(dc > 0, RIGHT, LEFT)
    vertical = (fdir == F_UP) | (fdir == F_DOWN)
    # finish on the axis that already leaves us facing the target
    first_a = jnp.where(vertical, mh, mv)
    first_n = jnp.where(vertical, dc != 0, dr != 0)
    second_a = jnp.where(vertical, mv, mh)
    second_n = jnp.where(vertical, dr != 0, dc != 0)
    settled = jnp.where(face == fdir, jnp.where(do_interact, INTERACT, STAY), turn)
    return jnp.where(first_n, first_a, jnp.where(second_n, second_a, settled)).astype(jnp.int32)


def _shelf_choice(stand_r, stand_c, far, codes, row, col, want_type, types):
    """Cheapest free shelf, preferring one already dedicated to ``want_type``."""
    cost = (jnp.abs(stand_r - row) + jnp.abs(stand_c - col) + far
            + (codes != -1).astype(jnp.int32) * 5000)
    if types is not None:
        cost = cost + (types != want_type).astype(jnp.int32) * 60
    idx = jnp.argmin(cost).astype(jnp.int32)
    return idx, cost[idx] < 5000


# ==========================================================================
# seat 0 : left room -- recipe reader and pot loader
# ==========================================================================
def init_agent_0():
    return jnp.asarray(INIT_0_LIST, dtype=jnp.int32)


def _source_0(kind):
    """Stand cell / turn action / facing for the left-room supply of ``kind``."""
    sr = jnp.where(kind == 1, 3, 1).astype(jnp.int32)
    sc = jnp.where(kind == 2, 6, 1).astype(jnp.int32)
    turn = jnp.where(kind == 2, UP, LEFT).astype(jnp.int32)
    fdir = jnp.where(kind == 2, F_UP, F_LEFT).astype(jnp.int32)
    return sr, sc, turn, fdir


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, jnp.asarray(INIT_0_LIST, dtype=jnp.int32), memory)
    obs = obs.astype(jnp.int32)
    row, col = _localize(obs, PEN_0_LIST)
    face = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    stand_r = jnp.asarray(S0_STAND_R, dtype=jnp.int32)
    stand_c = jnp.asarray(S0_STAND_C, dtype=jnp.int32)
    turns = jnp.asarray(S0_TURN, dtype=jnp.int32)
    fdirs = jnp.asarray(S0_FDIR, dtype=jnp.int32)
    types = jnp.asarray(S0_TYPE, dtype=jnp.int32)
    far = jnp.asarray(S0_FAR, dtype=jnp.int32)

    # ---- belief update -------------------------------------------------
    target = mem[0]
    for gr in (0, 4):
        seen, vec = _cell(obs, row, col, gr, 5)
        rcnt = vec[36:39]
        target = jnp.where(seen & (rcnt.sum() > 0),
                           jnp.argmax(rcnt).astype(jnp.int32), target)

    seen_pot, pvec = _cell(obs, row, col, 2, 7)
    p_n, p_kind, p_busy, _ = _pot_state(pvec)
    pot_n = jnp.where(seen_pot, p_n, mem[1])
    pot_kind = jnp.where(seen_pot, p_kind, mem[2])
    pot_busy = jnp.where(seen_pot, p_busy, mem[3])

    # A pot that has just gone from full/cooking to empty means somebody is
    # carrying a dish to the delivery station.  A correct delivery resamples
    # the recipe, and the new one only becomes visible a step later, so
    # anything deposited right now would encode a stale recipe.  Pause.
    emptied = ((mem[3] == 1) | (mem[1] == 3)) & (pot_busy == 0) & (pot_n == 0)
    hold_off = jnp.where(emptied, HOLD_OFF_STEPS,
                         jnp.maximum(mem[8] - 1, 0)).astype(jnp.int32)

    shelves = []
    for i in range(4):
        seen_i, vec_i = _cell(obs, row, col, S0_CELL[i][0], S0_CELL[i][1])
        shelves.append(jnp.where(seen_i, _item_code(vec_i), mem[4 + i]))
    shelves = jnp.stack(shelves).astype(jnp.int32)

    new_mem = jnp.concatenate(
        [jnp.stack([target, pot_n, pot_kind, pot_busy]).astype(jnp.int32),
         shelves, hold_off[None]])

    # ---- situation -----------------------------------------------------
    hold, _, _, _ = _held(obs)
    holding = hold >= 0
    known = target >= 0
    pot_open = (pot_busy == 0) & (pot_n < 3)
    deposit_ok = pot_open & (hold_off == 0)

    # A: fetch ``target`` from the cheapest stocked shelf, else from the supply
    src_r, src_c, src_turn, src_fdir = _source_0(jnp.maximum(target, 0))
    acq_r = jnp.concatenate([stand_r, src_r[None]])
    acq_c = jnp.concatenate([stand_c, src_c[None]])
    acq_turn = jnp.concatenate([turns, src_turn[None]])
    acq_fdir = jnp.concatenate([fdirs, src_fdir[None]])
    acq_bad = jnp.concatenate([(shelves != target).astype(jnp.int32) * 1000,
                               jnp.zeros((1,), dtype=jnp.int32)])
    acq_cost = (jnp.abs(acq_r - row) + jnp.abs(acq_c - col)
                + jnp.abs(acq_r - POT0_R) + jnp.abs(acq_c - POT0_C) + acq_bad)
    ai = jnp.argmin(acq_cost).astype(jnp.int32)

    # B: shelve whatever we are carrying
    si, shelf_ok = _shelf_choice(stand_r, stand_c,
                                 far + jnp.abs(stand_r - POT0_R) + jnp.abs(stand_c - POT0_C),
                                 shelves, row, col, hold, types)

    # C: restock one of the two near shelves while the pot is unavailable
    restock = jnp.where(shelves[0] == -1, 0,
                        jnp.where(shelves[1] == -1, 1, -1)).astype(jnp.int32)
    rs_r, rs_c, rs_turn, rs_fdir = _source_0(jnp.maximum(restock, 0))

    # ---- priority chain, lowest priority applied first ------------------
    # default: park facing the pot.  From (2,6) the pot, both recipe
    # indicators and both near shelves are all inside the 5x5 view.  Idling
    # one cell west to shorten the walk to the larder was measured to be
    # slightly worse, so the post stays adjacent to the pot.
    sr = jnp.asarray(POT0_R, dtype=jnp.int32)
    sc = jnp.asarray(POT0_C, dtype=jnp.int32)
    turn = jnp.asarray(RIGHT, dtype=jnp.int32)
    fdir = jnp.asarray(F_RIGHT, dtype=jnp.int32)
    grab = jnp.asarray(0, dtype=jnp.int32)
    mode = jnp.asarray(0, dtype=jnp.int32)

    put_in_pot = holding & known & (hold == target) & deposit_ok
    ready_but_waiting = holding & known & (hold == target) & pot_open & (hold_off > 0)
    options = [
        ((~holding) & (restock >= 0) & ~(known & pot_open),
         rs_r, rs_c, rs_turn, rs_fdir, 1, 1),
        ((~holding) & known & pot_open,
         acq_r[ai], acq_c[ai], acq_turn[ai], acq_fdir[ai], 1, 2),
        (holding & (~put_in_pot) & (~ready_but_waiting) & shelf_ok,
         stand_r[si], stand_c[si], turns[si], fdirs[si], 1, 3),
        # no shelf free: tip it into an already-dirty pot so it can cook and be
        # cleared.  Never into an empty pot -- that would fake a recipe signal.
        (holding & (~put_in_pot) & (~ready_but_waiting) & (~shelf_ok)
         & deposit_ok & (pot_n > 0),
         POT0_R, POT0_C, RIGHT, F_RIGHT, 1, 4),
        # stand over the pot with the ingredient ready, but do not commit it
        # until the post-delivery recipe has been seen
        (ready_but_waiting, POT0_R, POT0_C, RIGHT, F_RIGHT, 0, 6),
        # the deposit that tells seat 1 which recipe is up
        (put_in_pot, POT0_R, POT0_C, RIGHT, F_RIGHT, 1, 5),
    ]
    for cond, nr, nc, nt, nf, ng, nm in options:
        sr = jnp.where(cond, nr, sr)
        sc = jnp.where(cond, nc, sc)
        turn = jnp.where(cond, nt, turn)
        fdir = jnp.where(cond, nf, fdir)
        grab = jnp.where(cond, ng, grab)
        mode = jnp.where(cond, nm, mode)

    action = _step_action(row, col, face, sr, sc, turn, fdir, grab == 1)
    trace = {
        'pos': jnp.stack([row, col]),
        'target': target,
        'pot': jnp.stack([pot_n, pot_kind, pot_busy, hold_off]),
        'shelves': shelves,
        'hold': hold,
        'mode': mode,
    }
    return action, new_mem, trace


# ==========================================================================
# seat 1 : right room -- pot amplifier, plater and deliverer
# ==========================================================================
def init_agent_1():
    return jnp.asarray(INIT_1_LIST, dtype=jnp.int32)


def _source_1(kind):
    sr = jnp.where(kind == 1, 3, 1).astype(jnp.int32)
    turn = jnp.where(kind == 1, DOWN, UP).astype(jnp.int32)
    fdir = jnp.where(kind == 1, F_DOWN, F_UP).astype(jnp.int32)
    return sr, jnp.asarray(8, dtype=jnp.int32), turn, fdir


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, jnp.asarray(INIT_1_LIST, dtype=jnp.int32), memory)
    obs = obs.astype(jnp.int32)
    row, col = _localize(obs, PEN_1_LIST)
    face = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    stand_r = jnp.asarray(S1_STAND_R, dtype=jnp.int32)
    stand_c = jnp.asarray(S1_STAND_C, dtype=jnp.int32)
    turns = jnp.asarray(S1_TURN, dtype=jnp.int32)
    fdirs = jnp.asarray(S1_FDIR, dtype=jnp.int32)
    far = jnp.asarray(S1_FAR, dtype=jnp.int32)

    # the pot lies inside the crop from every right-room cell, so it never
    # needs remembering; only (0,9)/(4,9) can leave the view.
    _, pvec = _cell(obs, row, col, 2, 7)
    pot_n, pot_kind, pot_busy, pot_cooked = _pot_state(pvec)

    shelves = []
    for i in range(2):
        seen_i, vec_i = _cell(obs, row, col, S1_CELL[i][0], S1_CELL[i][1])
        shelves.append(jnp.where(seen_i, _item_code(vec_i), mem[i]))
    shelves = jnp.stack(shelves).astype(jnp.int32)

    hold, h_plate, h_cooked, h_cnt = _held(obs)
    is_dish = (h_plate == 1) & (h_cooked == 1)
    is_plate = (h_plate == 1) & (h_cooked == 0)
    is_raw = (h_plate == 0) & (h_cooked == 0) & (h_cnt.sum() > 0)
    raw_kind = jnp.argmax(h_cnt).astype(jnp.int32)
    empty = (~is_dish) & (~is_plate) & (~is_raw)
    # the only recipes this layout can request are [3,0,0] and [0,3,0]
    dish_ok = ((h_cnt[0] == 3) | (h_cnt[1] == 3)) & (h_cnt[2] == 0) & (h_cnt.sum() == 3)

    # ---- is the pot's content trustworthy? ------------------------------
    # A correct delivery resamples the recipe, and the new one is only visible
    # one step later.  So anything already sitting in the pot on the very first
    # observation after our own delivery was committed against the *previous*
    # recipe.  Amplifying it would turn a stale guess into a confident wrong
    # dish, so leave it to the partner to correct instead.
    delivered = (mem[2] == 1) & (~is_dish)
    taint = jnp.where(delivered, (pot_n > 0).astype(jnp.int32), mem[3])
    taint = jnp.where(pot_n == 0, 0, taint).astype(jnp.int32)
    taint_timer = jnp.where(delivered & (pot_n > 0), TAINT_STEPS,
                            jnp.maximum(mem[4] - 1, 0)).astype(jnp.int32)
    tainted = (taint == 1) & (taint_timer > 0)

    empty_run = jnp.where((pot_n == 0) & (pot_busy == 0),
                          jnp.minimum(mem[5] + 1, 999), 0).astype(jnp.int32)
    handoff_seen = mem[6]
    desperate = empty_run >= jnp.where(handoff_seen == 1,
                                       DESPERATE_STEPS_SEEN, DESPERATE_STEPS)

    # ---- what the pot needs --------------------------------------------
    # pure raw stack of 0/1 -> that is the recipe, match it.  Anything else
    # (mixed, or ingredient 2 which no recipe can ask for) is unsalvageable:
    # fill it so that it cooks and can be scooped out and thrown away, rather
    # than leaving the only pot blocked for the rest of the episode.
    readable = (pot_kind >= 0) & (pot_kind <= 1)
    spoiled = (pot_kind == 2) | (pot_kind == 3)
    want = jnp.where(readable, pot_kind, 0).astype(jnp.int32)
    helpable = ((pot_busy == 0) & (pot_n > 0) & (pot_n < 3)
                & ((readable & (~tainted)) | spoiled))

    # last resort only: if the pot has stood empty for a long time the partner
    # is clearly not using it, and a lone ingredient parked on a shared
    # hand-off counter is the only recipe evidence available.
    _, v17 = _cell(obs, row, col, 1, 7)
    _, v37 = _cell(obs, row, col, 3, 7)
    c17 = _item_code(v17)
    c37 = _item_code(v37)
    sig17 = (c17 >= 0) & (c17 <= 1)
    sig37 = (c37 >= 0) & (c37 <= 1)
    lone_signal = sig17 ^ sig37
    signal_r = jnp.where(sig17, 1, 3).astype(jnp.int32)

    src_r, src_c, src_turn, src_fdir = _source_1(want)
    acq_r = jnp.concatenate([stand_r, src_r[None]])
    acq_c = jnp.concatenate([stand_c, src_c[None]])
    acq_turn = jnp.concatenate([turns, src_turn[None]])
    acq_fdir = jnp.concatenate([fdirs, src_fdir[None]])
    acq_bad = jnp.concatenate([(shelves != want).astype(jnp.int32) * 1000
                               + jnp.asarray(S1_SHELF_BONUS, dtype=jnp.int32),
                               jnp.zeros((1,), dtype=jnp.int32)])
    acq_cost = (jnp.abs(acq_r - row) + jnp.abs(acq_c - col)
                + jnp.abs(acq_r - POT1_R) + jnp.abs(acq_c - POT1_C) + acq_bad)
    ai = jnp.argmin(acq_cost).astype(jnp.int32)

    di, shelf_ok = _shelf_choice(stand_r, stand_c, far, shelves, row, col, None, None)

    plate_r = jnp.where(jnp.abs(row - 1) <= jnp.abs(row - 3), 1, 3).astype(jnp.int32)

    sr = jnp.asarray(POT1_R, dtype=jnp.int32)
    sc = jnp.asarray(POT1_C, dtype=jnp.int32)
    turn = jnp.asarray(LEFT, dtype=jnp.int32)
    fdir = jnp.asarray(F_LEFT, dtype=jnp.int32)
    grab = jnp.asarray(0, dtype=jnp.int32)
    mode = jnp.asarray(0, dtype=jnp.int32)

    add_raw = (is_raw & (pot_busy == 0) & (pot_n < 3)
               & (((pot_n > 0) & ((pot_kind == raw_kind) | spoiled))
                  | ((pot_n == 0) & desperate)))
    options = [
        # help fill a pot the partner has already committed to
        (empty & helpable, acq_r[ai], acq_c[ai], acq_turn[ai], acq_fdir[ai], 1, 1),
        # pot cooking or ready -> pre-fetch a plate
        (empty & (pot_busy == 1), plate_r, 9, RIGHT, F_RIGHT, 1, 2),
        # long-dead pot plus a lone ingredient on a hand-off counter
        (empty & desperate & lone_signal, signal_r, 8, LEFT, F_LEFT, 1, 8),
        (is_raw & (~add_raw) & shelf_ok,
         stand_r[di], stand_c[di], turns[di], fdirs[di], 1, 3),
        # both shelves full: tip the spare into an already-dirty pot so our
        # hands come free; the result is provably wrong and gets thrown away
        (is_raw & (~add_raw) & (~shelf_ok) & (pot_busy == 0) & (pot_n > 0) & (pot_n < 3),
         POT1_R, POT1_C, LEFT, F_LEFT, 1, 9),
        (add_raw, POT1_R, POT1_C, LEFT, F_LEFT, 1, 4),
        # holding a plate: wait at the pot, scoop the instant it is cooked
        (is_plate & (pot_cooked == 1), POT1_R, POT1_C, LEFT, F_LEFT, 1, 5),
        # a dish that cannot match any recipe is shelved, never delivered
        (is_dish & (~dish_ok) & shelf_ok,
         stand_r[di], stand_c[di], turns[di], fdirs[di], 1, 6),
        (is_dish & (dish_ok | (~shelf_ok)), 2, 9, RIGHT, F_RIGHT, 1, 7),
    ]
    for cond, nr, nc, nt, nf, ng, nm in options:
        sr = jnp.where(cond, nr, sr)
        sc = jnp.where(cond, nc, sc)
        turn = jnp.where(cond, nt, turn)
        fdir = jnp.where(cond, nf, fdir)
        grab = jnp.where(cond, ng, grab)
        mode = jnp.where(cond, nm, mode)

    action = _step_action(row, col, face, sr, sc, turn, fdir, grab == 1)
    # remember whether this step was an actual delivery attempt, and whether
    # the hand-off counter channel has ever actually been used
    delivering = ((mode == 7) & (action == INTERACT)).astype(jnp.int32)
    took_handoff = (mode == 8) & (action == INTERACT)
    handoff_seen = jnp.where(took_handoff, 1, handoff_seen).astype(jnp.int32)
    new_mem = jnp.concatenate(
        [shelves, jnp.stack([delivering, taint, taint_timer, empty_run,
                             handoff_seen]).astype(jnp.int32)])
    trace = {
        'pos': jnp.stack([row, col]),
        'pot': jnp.stack([pot_n, pot_kind, pot_busy, pot_cooked]),
        'shelves': shelves,
        'flags': jnp.stack([taint, taint_timer, empty_run, hold]),
        'mode': mode,
    }
    return action, new_mem, trace
