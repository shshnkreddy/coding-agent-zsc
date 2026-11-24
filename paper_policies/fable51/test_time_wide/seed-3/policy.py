"""Overcooked V2 test_time_wide policy.

Agent 0 (top room) runs the whole cook/plate/deliver loop alone: the left pot
(3,1) always holds ingredient 0, the right pot (3,3) always holds ingredient 1,
so both possible dishes are kept ready. It cannot see the recipe indicator, so
it decodes the recipe from the partner's observable state on row 4
(column position first; a held item or an item on the shared right counter
are weak extra cues for recipe 1). Because the partner's convention is
unknown, agent 0 keeps a small Bayesian posterior over
(signal polarity in {correct, reversed, uninformative}) x (current recipe),
updated once per delivery cycle, and uses the recipe dynamics (a wrong
delivery keeps the recipe, a correct one resamples it) to self-correct.

Agent 1 (bottom room) reads the always-on recipe indicator at (6,2) and
signals by standing on row 4 under the left pot (col 1) for recipe [0,0,0]
and under the right pot (col 3) for recipe [1,1,1], facing up.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'cook_bayes_column_signal', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

# Actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing direction ids (observation)
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3
DIR_TO_ACTION_LIST = [A_UP, A_DOWN, A_RIGHT, A_LEFT]


def _dir_to_action():
    return jnp.asarray(DIR_TO_ACTION_LIST, dtype=jnp.int32)


# Channels
C_SELF, C_PARTNER = 0, 11
C_WALL, C_GOAL, C_POT, C_RECIPE_IND, C_BUTTON, C_PLATE_PILE = 22, 23, 24, 25, 26, 27
C_ITEM_PLATE, C_ITEM_COOKED, C_ITEM_ING0 = 32, 33, 34
C_RECIPE_ING0 = 40
C_POT_TIMER = 44

# Agent-0 tasks: (target_row, target_col, facing, interact)
T_DELIVER, T_PLATE, T_POT_L, T_POT_R, T_ING0, T_ING1, T_LOOK, T_DUMP_A, T_DUMP_B = 0, 1, 2, 3, 4, 5, 6, 7, 8
T_DUMP_C, T_DUMP_D = 9, 10
TASKS_LIST = [
    [1, 2, D_UP, 1],     # deliver at (0,2)
    [1, 3, D_UP, 1],     # plate pile at (0,3)
    [2, 1, D_DOWN, 1],   # left pot (3,1)
    [2, 3, D_DOWN, 1],   # right pot (3,3)
    [1, 1, D_LEFT, 1],   # ingredient 0 supply (1,0)
    [2, 4, D_RIGHT, 1],  # ingredient 1 supply (2,5)
    [2, 2, D_DOWN, 0],   # look at partner from centre
    [1, 1, D_UP, 1],     # dump on counter (0,1)
    [1, 4, D_UP, 1],     # dump on counter (0,4)
    [2, 2, D_DOWN, 1],   # dump on shared counter (3,2)
    [2, 4, D_DOWN, 1],   # dump on shared counter (3,4)
]


def _tasks():
    return jnp.asarray(TASKS_LIST, dtype=jnp.int32)


# Memory layout for agent 0: dict(i=int32 vector, p=float32 (4,2) posterior)
M_T, M_LASTVAL, M_LASTSTEP, M_STABLE, M_DELIVSTEP = 0, 1, 2, 3, 4
M_FRESHVAL, M_EVERREAD, M_COMMITTED, M_TARGET, M_PREVREAD = 5, 6, 7, 8, 9
M_POTL, M_POTR = 10, 15   # 5 fields each: n0, n1, nother, cooked, timer
M_CYCLEVAL, M_CYCLESTEP, M_MINCOL, M_MAXCOL, M_MOVED, M_FACEMASK = 20, 21, 22, 23, 24, 25
M_CLEAN, M_DUMPS = 26, 27   # cleanup target pot (0 none, 1 left, 2 right); dumps done so far
# Four-cue vectors: [column reading, partner holds item, item on (3,2), item on (3,4)]
NCUE = 4
M_LAST4, M_FRESH4, M_PREV4, M_CYCLE4 = 28, 32, 36, 40
M_SEEN0, M_SEEN1 = 44, 45   # bitmasks: cue k observed with value 0 / value 1
MEM0_SIZE = 46

# Column vote toward recipe 1 (index = partner global column 0..5)
COL_VOTE_LIST = [0, -4, 1, 4, 4, 0]

# Signal-model hypotheses (rows): for each cue k in [column, held, counter(3,2),
# counter(3,4)] a "cue k tracks recipe" (2k) and "cue k tracks reversed recipe"
# (2k+1) hypothesis, then "everything static" (8) and "random" (9).
NH = 10
H_STATIC, H_RANDOM = 8, 9
PRIOR_H = [0.88, 0.005, 0.006, 0.002, 0.004, 0.002, 0.004, 0.002, 0.075, 0.02]
LIVE_BOOST = 10.0      # applied to a secondary cue's hypotheses once that cue is seen to vary
SIG_EPS = 0.05
STICKY = 0.97
MOVED_FACTORS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0]   # partner seen at two columns/facings
MIDCYCLE_OWN, MIDCYCLE_OTHER = 0.5, 0.3     # factors when a cue changes with no delivery in between
MIDCYCLE_MIN_GAP = 20                       # only count mid-cycle changes this long after a delivery
PREFETCH_TIMER = 10     # fetch the plate when the target pot has at most this many cook steps left
SIGNAL_DWELL = 8        # agent 1: steps spent at each recipe-1 signal column
FILL_FIRST_UNTIL = 360  # before this step, refill pots before fetching a plate
READ_DELAY = 8          # steps after a delivery before a reading counts
NO_READ_FALLBACK = 60   # steps without any reading before deciding on prior alone
LOST_FALLBACK = 40      # steps after delivery without fresh reading before deciding on prior


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _locate(chan):
    idx = jnp.argmax(chan.reshape(-1))
    return _i32(idx // 5), _i32(idx % 5)


def _cell(obs, li, lj):
    li = jnp.clip(li, 0, 4)
    lj = jnp.clip(lj, 0, 4)
    return obs[li, lj]


def _move_toward(r, c, tr, tc, tdir):
    dc = tc - c
    dr = tr - r
    horiz = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    vert = jnp.where(dr > 0, A_DOWN, A_UP)
    vertical_face = (tdir == D_UP) | (tdir == D_DOWN)
    act_hfirst = jnp.where(dc != 0, horiz, vert)
    act_vfirst = jnp.where(dr != 0, vert, horiz)
    return _i32(jnp.where(vertical_face, act_hfirst, act_vfirst))


def _do_task(r, c, facing, task):
    spec = _tasks()[task]
    tr, tc, tdir, inter = spec[0], spec[1], spec[2], spec[3]
    at = (r == tr) & (c == tc)
    faced = facing == tdir
    move = _move_toward(r, c, tr, tc, tdir)
    face_act = _dir_to_action()[tdir]
    final = jnp.where(inter == 1, A_INTERACT, A_STAY)
    return _i32(jnp.where(~at, move, jnp.where(~faced, face_act, final))), at & faced


def _pot_update(obs, st, li, lj, visible):
    cell = _cell(obs, li, lj)
    seen = jnp.stack([
        cell[C_ITEM_ING0], cell[C_ITEM_ING0 + 1],
        cell[C_ITEM_ING0 + 2] + cell[C_ITEM_ING0 + 3],
        cell[C_ITEM_COOKED], cell[C_POT_TIMER]]).astype(jnp.int32)
    timer = st[4]
    new_timer = jnp.maximum(timer - 1, 0)
    cooked = jnp.maximum(st[3], _i32(timer == 1))
    pred = jnp.stack([st[0], st[1], st[2], cooked, new_timer]).astype(jnp.int32)
    return jnp.where(visible, seen, pred)


def _cue_likelihood(k, s, prev):
    """(NH, 2) likelihood factor of cue k reading s (prev = previous committed value or -1).

    Hypotheses about cue k predict s from the recipe; every other hypothesis
    treats cue k as sticky (except random, which treats it as a coin flip).
    Invalid readings (s < 0) contribute no evidence.
    """
    rr = jnp.arange(2)
    plus = jnp.where(rr == s, 1.0 - SIG_EPS, SIG_EPS)
    minus = 1.0 - plus
    sticky = jnp.where(prev < 0, 0.5, jnp.where(prev == s, STICKY, 1.0 - STICKY))
    stat = jnp.full((2,), 1.0) * sticky
    half = jnp.full((2,), 0.5)
    hidx = jnp.arange(NH)
    f = jnp.broadcast_to(stat[None, :], (NH, 2))
    f = jnp.where((hidx == 2 * k)[:, None], plus[None, :], f)
    f = jnp.where((hidx == 2 * k + 1)[:, None], minus[None, :], f)
    f = jnp.where((hidx == H_RANDOM)[:, None], half[None, :], f)
    return jnp.where(s >= 0, f, jnp.ones((NH, 2))).astype(jnp.float32)


def _signal_update(post, cues, prev):
    """Multiply posterior by the joint likelihood of the committed cue vector."""
    like = jnp.ones((NH, 2), dtype=jnp.float32)
    for k in range(NCUE):
        like = like * _cue_likelihood(k, cues[k], prev[k])
    new = post * like
    return new / jnp.maximum(jnp.sum(new), 1e-9)


def _live_factors(k):
    """Boost for cue k's hypotheses once cue k has been seen with both values."""
    hidx = jnp.arange(NH)
    own = (hidx == 2 * k) | (hidx == 2 * k + 1)
    return jnp.where(own, LIVE_BOOST, 1.0).astype(jnp.float32)


def _midcycle_factors(k):
    """Hypothesis factors when cue k changes with no delivery in between."""
    hidx = jnp.arange(NH)
    own = (hidx == 2 * k) | (hidx == 2 * k + 1)
    f = jnp.where(own, MIDCYCLE_OWN, MIDCYCLE_OTHER)
    return jnp.where(hidx == H_RANDOM, 1.0, f).astype(jnp.float32)


def _event_update(post, factors):
    """Multiply each hypothesis row by a likelihood factor and renormalise."""
    f = jnp.asarray(factors, dtype=jnp.float32)
    new = post * f[:, None]
    return new / jnp.maximum(jnp.sum(new), 1e-9)


def _delivery_transition(post, d):
    """Recipe dynamics: correct delivery resamples uniformly, wrong keeps recipe."""
    rr = jnp.arange(2)
    is_d = (rr == d)[None, :]
    pd = jnp.sum(post * is_d, axis=1)       # (num_hypotheses,)
    po = jnp.sum(post * (~is_d), axis=1)
    col_d = pd * 0.5
    col_o = po + pd * 0.5
    new = jnp.where(is_d, col_d[:, None], col_o[:, None])
    return new.astype(jnp.float32)


def _init_post():
    h = jnp.asarray(PRIOR_H, dtype=jnp.float32)
    return jnp.stack([h * 0.5, h * 0.5], axis=1)


def init_agent_0():
    mem = jnp.zeros((MEM0_SIZE,), dtype=jnp.int32)
    mem = mem.at[M_LASTSTEP].set(-100)
    mem = mem.at[M_DELIVSTEP].set(-1000)
    mem = mem.at[M_TARGET].set(-1)
    mem = mem.at[M_CYCLESTEP].set(-1000)
    mem = mem.at[M_MINCOL].set(9)
    mem = mem.at[M_MAXCOL].set(-1)
    mem = mem.at[M_LAST4:M_LAST4 + NCUE].set(-1)
    mem = mem.at[M_FRESH4:M_FRESH4 + NCUE].set(-1)
    mem = mem.at[M_PREV4:M_PREV4 + NCUE].set(-1)
    mem = mem.at[M_CYCLE4:M_CYCLE4 + NCUE].set(-1)
    return {'i': mem, 'p': _init_post()}


def agent_0_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs).astype(jnp.int32)
    init = init_agent_0()
    mem = jnp.where(reset, init['i'], memory['i'])
    post = jnp.where(reset, init['p'], memory['p'])
    t = mem[M_T]

    # Localise from the delivery goal at (0,2) (always in view from top room).
    gi, gj = _locate(obs[:, :, C_GOAL])
    r = 0 - gi + 2
    c = 2 - gj + 2
    facing = _i32(jnp.argmax(obs[2, 2, 1:5]))
    me = obs[2, 2]
    hold_plate = (me[5] == 1) & (me[6] == 0)
    hold_dish = me[6] == 1
    ing_counts = me[7:11]
    has_ing = jnp.sum(ing_counts) > 0
    hold_ing = _i32(jnp.argmax(ing_counts))

    # Pot state (with prediction while out of view).
    potL = _pot_update(obs, mem[M_POTL:M_POTL + 5], 5 - r, 3 - c, c <= 3)
    potR = _pot_update(obs, mem[M_POTR:M_POTR + 5], 5 - r, 5 - c, c >= 1)

    def pot_flags(p, want):
        other = jnp.where(want == 0, p[1], p[0]) + p[2]
        good = other == 0
        n = p[0] + p[1] + p[2]
        ready = (p[3] == 1) & good
        accepts = (p[4] == 0) & (p[3] == 0) & (n < 3) & good
        return ready, accepts, n

    readyL, accL, nL = pot_flags(potL, 0)
    readyR, accR, nR = pot_flags(potR, 1)
    ready_of = jnp.stack([readyL, readyR])
    acc_of = jnp.stack([accL, accR])
    goodL = (potL[1] + potL[2]) == 0
    goodR = (potR[0] + potR[2]) == 0
    # Contaminated pots (partner interference): cook them out, then scoop and dump.
    nLtot = potL[0] + potL[1] + potL[2]
    nRtot = potR[0] + potR[1] + potR[2]
    badL_idle = (~goodL) & (potL[4] == 0) & (potL[3] == 0) & (nLtot < 3)
    badR_idle = (~goodR) & (potR[4] == 0) & (potR[3] == 0) & (nRtot < 3)
    badL_cooked = (~goodL) & (potL[3] == 1)
    badR_cooked = (~goodR) & (potR[3] == 1)
    clean = mem[M_CLEAN]
    dumps = mem[M_DUMPS]
    can_dump = dumps < 4
    dump_task = jnp.where(dumps == 0, T_DUMP_A, jnp.where(dumps == 1, T_DUMP_B,
                          jnp.where(dumps == 2, T_DUMP_C, T_DUMP_D)))
    soon_of = jnp.stack([goodL & (potL[4] > 0) & (potL[4] <= PREFETCH_TIMER),
                         goodR & (potR[4] > 0) & (potR[4] <= PREFETCH_TIMER)])
    cooking_of = jnp.stack([goodL & (potL[4] > 0), goodR & (potR[4] > 0)])

    # Partner reading (instantaneous cue decode).
    pvis = jnp.sum(obs[:, :, C_PARTNER]) > 0
    pi, pj = _locate(obs[:, :, C_PARTNER])
    pc = jnp.clip(c + pj - 2, 0, 5)
    pcell = _cell(obs, pi, pj)
    pheld = jnp.sum(pcell[16:22]) > 0
    # Shared counters (3,2) and (3,4): local (5-r, 4-c) and (5-r, 6-c).
    ca = _cell(obs, 5 - r, 4 - c)
    cb = _cell(obs, 5 - r, 6 - c)
    ctr_a = (jnp.sum(ca[32:38]) > 0)
    ctr_b = (c >= 2) & (jnp.sum(cb[32:38]) > 0)
    score = jnp.asarray(COL_VOTE_LIST, dtype=jnp.int32)[pc]
    reading = jnp.where(score > 0, 1, jnp.where(score < 0, 0, -1))
    valid_read = pvis & (reading >= 0)
    cues = jnp.stack([reading, _i32(pheld), _i32(ctr_a), _i32(ctr_b)]).astype(jnp.int32)
    cues = jnp.where(valid_read, cues, -1)

    last4, last_step, stable = mem[M_LAST4:M_LAST4 + NCUE], mem[M_LASTSTEP], mem[M_STABLE]
    consecutive = jnp.all(cues == last4) & (last_step == t - 1)
    new_stable = jnp.where(valid_read, jnp.where(consecutive, stable + 1, 1), stable)
    new_last4 = jnp.where(valid_read, cues, last4)
    new_last_step = jnp.where(valid_read, t, last_step)
    delivstep = mem[M_DELIVSTEP]
    fresh = t >= delivstep + READ_DELAY
    have_fresh = valid_read & fresh & (new_stable >= 2)
    fresh4 = jnp.where(have_fresh, cues, mem[M_FRESH4:M_FRESH4 + NCUE])
    freshval = fresh4[0]
    ever_read = jnp.maximum(mem[M_EVERREAD], _i32(have_fresh))

    # Evidence about the partner's signalling behaviour (independent of polarity):
    # (a) seen at two different columns -> not a static/idle partner;
    # (b) a stable reading that changes with no delivery in between -> not a
    #     recipe-driven signal (random movement).
    mincol = jnp.where(pvis, jnp.minimum(mem[M_MINCOL], pc), mem[M_MINCOL])
    maxcol = jnp.where(pvis, jnp.maximum(mem[M_MAXCOL], pc), mem[M_MAXCOL])
    pface = _i32(jnp.argmax(pcell[12:16]))
    facemask = jnp.where(pvis, mem[M_FACEMASK] | (1 << pface), mem[M_FACEMASK])
    nfaces = ((facemask >> 0) & 1) + ((facemask >> 1) & 1) + ((facemask >> 2) & 1) + ((facemask >> 3) & 1)
    moved_now = (mem[M_MOVED] == 0) & (((maxcol > mincol) & (mincol >= 1)) | (nfaces >= 2))
    cycle4 = mem[M_CYCLE4:M_CYCLE4 + NCUE]
    mid_ok = have_fresh & (t >= delivstep + MIDCYCLE_MIN_GAP)
    post = jnp.where(moved_now, _event_update(post, MOVED_FACTORS), post)
    # Secondary cues only become credible once observed to vary.
    bits = jnp.left_shift(1, jnp.arange(NCUE))
    seen0 = mem[M_SEEN0] | jnp.where(valid_read, jnp.sum(jnp.where(cues == 0, bits, 0)), 0)
    seen1 = mem[M_SEEN1] | jnp.where(valid_read, jnp.sum(jnp.where(cues == 1, bits, 0)), 0)
    live_old = mem[M_SEEN0] & mem[M_SEEN1]
    live_new = (seen0 & seen1) & (~live_old)
    for k in range(1, NCUE):
        post = jnp.where((live_new >> k) & 1 == 1, _event_update(post, _live_factors(k)), post)
    midcycle = jnp.zeros((), dtype=jnp.bool_)
    for k in range(NCUE):
        changed_k = mid_ok & (cycle4[k] >= 0) & (cues[k] != cycle4[k])
        post = jnp.where(changed_k, _event_update(post, _midcycle_factors(k)), post)
        midcycle = midcycle | changed_k
    new_cycle4 = jnp.where(have_fresh, cues, cycle4)

    # Tentative posterior including this cycle's reading (not yet committed).
    prev4 = mem[M_PREV4:M_PREV4 + NCUE]
    tent = jnp.where(freshval >= 0, _signal_update(post, fresh4, prev4), post)
    pR = jnp.sum(tent, axis=0)
    dstar = _i32(jnp.argmax(pR))
    committed = mem[M_COMMITTED] == 1
    target = jnp.where(committed, mem[M_TARGET], dstar)
    can_commit = (freshval >= 0) | ((ever_read == 0) & (t >= NO_READ_FALLBACK)) | ((delivstep >= 0) & (t >= delivstep + LOST_FALLBACK))

    # Task selection.
    tclip = jnp.clip(target, 0, 1)
    target_ready = ready_of[tclip]
    scoop_task = jnp.where(target == 0, T_POT_L, T_POT_R)
    # Which pot to fill when hands are empty: prefer the target's pot.
    prefer_t = acc_of[tclip]
    fillL = jnp.where(prefer_t, target == 0,
                      jnp.where(accL & accR, nL >= nR, accL))
    fill_task = jnp.where(fillL, T_ING0, T_ING1)
    any_fill = accL | accR

    target_soon = soon_of[tclip] & (~any_fill)
    # Keep both pots busy: refill an empty/partial pot before delivering, unless
    # the episode is about to end.
    fill_first = any_fill & (t < FILL_FIRST_UNTIL)
    go_plate = (target_ready | target_soon) & (committed | can_commit) & (~fill_first)
    # Cleanup of contaminated pots (lowest priority among empty-hand tasks).
    clean_plate = (clean == 0) & can_dump & (badL_cooked | badR_cooked)
    clean_pot_id = jnp.where(badL_cooked, 1, 2)
    bad_fill = badL_idle | badR_idle
    bad_fill_task = jnp.where(badL_idle, T_ING0, T_ING1)
    task_idle = jnp.where(clean_plate | (clean > 0), T_PLATE, jnp.where(bad_fill, bad_fill_task, T_LOOK))
    task_empty = jnp.where(go_plate, T_PLATE, jnp.where(any_fill, fill_task, task_idle))
    ing_pot_acc = jnp.where(hold_ing == 0, accL, jnp.where(hold_ing == 1, accR, False))
    bad_add_task = jnp.where(badL_idle, T_POT_L, jnp.where(badR_idle, T_POT_R, dump_task))
    task_ing = jnp.where(ing_pot_acc, jnp.where(hold_ing == 0, T_POT_L, T_POT_R), bad_add_task)
    # With a plate: scoop when ready; while still cooking, wait at the pot (interact is a no-op).
    task_plate_normal = jnp.where(target_ready | cooking_of[tclip], scoop_task, T_LOOK)
    task_plate = jnp.where(clean > 0, jnp.where(clean == 1, T_POT_L, T_POT_R), task_plate_normal)
    task_dish = jnp.where(clean > 0, dump_task, T_DELIVER)
    task = jnp.where(hold_dish, task_dish,
                     jnp.where(hold_plate, task_plate,
                               jnp.where(has_ing, task_ing, task_empty)))
    task = _i32(task)
    action, executing = _do_task(r, c, facing, task)
    delivered = (task == T_DELIVER) & executing
    empty_hands = (~hold_dish) & (~hold_plate) & (~has_ing)
    start_clean = empty_hands & (task == T_PLATE) & (~go_plate) & clean_plate
    dumped_any = (task == dump_task) & executing & (~empty_hands)
    dumped = dumped_any & hold_dish & (clean > 0)
    new_clean = jnp.where(dumped, 0, jnp.where(start_clean, clean_pot_id, clean))
    new_dumps = jnp.where(dumped_any, dumps + 1, dumps)
    commit_now = (~committed) & (task == T_PLATE) & empty_hands & (~start_clean) & (clean == 0)

    # Posterior bookkeeping.
    post_c = jnp.where(commit_now, tent, post)
    post_d = jnp.where(delivered, _delivery_transition(post_c, tclip), post_c)
    new_prev4 = jnp.where(commit_now & (freshval >= 0), fresh4, prev4)
    neg4 = jnp.full((NCUE,), -1, dtype=jnp.int32)

    new_mem = mem
    new_mem = new_mem.at[M_T].set(t + 1)
    new_mem = new_mem.at[M_LAST4:M_LAST4 + NCUE].set(new_last4)
    new_mem = new_mem.at[M_LASTSTEP].set(new_last_step)
    new_mem = new_mem.at[M_STABLE].set(jnp.where(delivered, 0, new_stable))
    new_mem = new_mem.at[M_DELIVSTEP].set(jnp.where(delivered, t, delivstep))
    new_mem = new_mem.at[M_FRESH4:M_FRESH4 + NCUE].set(jnp.where(delivered | commit_now, neg4, fresh4))
    new_mem = new_mem.at[M_EVERREAD].set(ever_read)
    new_mem = new_mem.at[M_COMMITTED].set(jnp.where(delivered, 0, jnp.where(commit_now, 1, mem[M_COMMITTED])))
    new_mem = new_mem.at[M_TARGET].set(jnp.where(delivered, -1, jnp.where(commit_now, dstar, mem[M_TARGET])))
    new_mem = new_mem.at[M_PREV4:M_PREV4 + NCUE].set(new_prev4)
    new_mem = new_mem.at[M_POTL:M_POTL + 5].set(potL)
    new_mem = new_mem.at[M_POTR:M_POTR + 5].set(potR)
    new_mem = new_mem.at[M_CYCLE4:M_CYCLE4 + NCUE].set(jnp.where(delivered, neg4, new_cycle4))
    new_mem = new_mem.at[M_MINCOL].set(mincol)
    new_mem = new_mem.at[M_MAXCOL].set(maxcol)
    new_mem = new_mem.at[M_MOVED].set(jnp.maximum(mem[M_MOVED], _i32(moved_now)))
    new_mem = new_mem.at[M_FACEMASK].set(facemask)
    new_mem = new_mem.at[M_CLEAN].set(new_clean)
    new_mem = new_mem.at[M_DUMPS].set(new_dumps)
    new_mem = new_mem.at[M_SEEN0].set(seen0)
    new_mem = new_mem.at[M_SEEN1].set(seen1)
    trace = {
        'target': target, 'task': task, 'score': _i32(score), 'row': r, 'col': c,
        'readyL': _i32(readyL), 'readyR': _i32(readyR), 'nL': _i32(nL), 'nR': _i32(nR),
        'pvis': _i32(pvis), 'pc': pc, 'freshval': freshval, 'p1': pR[1],
        'ph_correct': jnp.sum(tent[0]), 'ph_rev': jnp.sum(tent[1]),
        'ph_uninf': jnp.sum(tent[H_STATIC]) + jnp.sum(tent[H_RANDOM]),
        'ph_held': jnp.sum(tent[2]) + jnp.sum(tent[3]), 'ph_ctr': jnp.sum(tent[4:8]),
        'held': cues[1], 'ctr_a': cues[2], 'ctr_b': cues[3],
        'midcycle': _i32(midcycle), 'moved': _i32(moved_now), 'clean': new_clean, 'dumps': new_dumps,
    }
    return action, {'i': new_mem, 'p': post_d}, trace


# ---------------------------------------------------------------- agent 1

def init_agent_1():
    return jnp.zeros((4,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs).astype(jnp.int32)
    mem = jnp.where(reset, init_agent_1(), memory)
    # Localise from the recipe indicator at (6,2), always in view from bottom room.
    ii, ij = _locate(obs[:, :, C_RECIPE_IND])
    r = 6 - ii + 2
    c = 2 - ij + 2
    facing = _i32(jnp.argmax(obs[2, 2, 1:5]))
    rcell = _cell(obs, ii, ij)
    counts = rcell[C_RECIPE_ING0:C_RECIPE_ING0 + 4]
    have_recipe = jnp.sum(counts) > 0
    recipe = _i32(jnp.argmax(counts))
    recipe = jnp.where(have_recipe, recipe, mem[0])
    # Recipe 1: alternate between the right-pot column (3) and the right wall
    # column (4) with fixed dwells so exact col-3 and exact col-4 decoders both
    # get a readable signal; threshold decoders (col >= 3) see no change.
    # Recipe 0: stay at col 1 but alternate facing UP / LEFT (both blocked, so
    # the position is stable) as a liveness cue that distinguishes a signaller
    # from an idle partner.
    phase = (mem[1] // SIGNAL_DWELL) % 2
    target_col = jnp.where(recipe == 0, 1, jnp.where(phase == 0, 3, 4))
    target_dir = jnp.where((recipe == 0) & (phase == 1), D_LEFT, D_UP)
    tr, tc, tdir = _i32(4), _i32(target_col), _i32(target_dir)
    at = (r == tr) & (c == tc)
    faced = facing == tdir
    move = _move_toward(r, c, tr, tc, tdir)
    action = _i32(jnp.where(~at, move, jnp.where(~faced, _dir_to_action()[tdir], A_STAY)))
    new_mem = mem.at[0].set(recipe).at[1].set(mem[1] + 1)
    trace = {'recipe': recipe, 'row': r, 'col': c}
    return action, new_mem, trace
