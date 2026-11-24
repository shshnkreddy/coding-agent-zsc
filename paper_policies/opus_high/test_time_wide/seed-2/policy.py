"""Overcooked-V2 `test_time_wide` joint policy.

Layout facts that drive the design (see context/environment/README.md):

    WWXBWW      rows 1-2 (cols 1-4) form the *top* room  -> seat agent_0
    0 A  0      rows 4-5 (cols 1-4) form the *bottom* room -> seat agent_1
    1    1      the rooms are disconnected: row 3 is wall/pot/wall/pot/wall
    WPWPWW
    3 A  3
    W    W
    WWRWWW

* Ingredient supplies 0 and 1, the plate pile (0,3) and the delivery station
  (0,2) are reachable only from the top room, so agent_0 must cook and deliver
  every dish by itself.
* The always-on recipe indicator sits at (6,2) and is inside the 5x5 crop of
  every bottom-room cell and of *no* top-room cell.  Hence agent_1 is the only
  seat that can ever read the target recipe, and agent_0 is the only seat that
  can act on it.  The episode therefore has exactly one useful coordination
  channel: agent_1 -> agent_0 recipe signalling.
* The only cells both seats can touch are the counters (3,2) and (3,4).

Signalling (speaker = agent_1) is emitted redundantly on one shared left/right
axis: a marker ingredient parked on counter (3,2) vs (3,4), plus an extreme
column (4,1) vs (4,4) and a LEFT vs RIGHT facing.  Listening (agent_0) reads
the same axis, counters first, partner pose second.

The left/right *polarity* is an arbitrary convention that an independently
generated partner need not share, so agent_0 never trusts it: it calibrates the
polarity online.  A recipe is resampled only after a *correct* delivery, so a
change of the partner's signal across one of our deliveries proves that
delivery was correct, while a signal that never changes is evidence that we are
decoding with the wrong polarity.  With no usable evidence the accumulator
oscillates, which reproduces the information-free optimum of alternating the
delivered dish (2/3 of deliveries correct) instead of repeating a dish that the
environment would never resample.

Throughput: agent_0 runs a two-pot pipeline.  Pot (3,1) is nominally dedicated
to ingredient 0 (supply (1,0), reached from (1,1)) and pot (3,3) to ingredient 1
(supply (2,5), reached from (2,4)), so the pot just emptied by a delivery can be
refilled before the new recipe is even known, the 20-step cook overlaps useful
work, and the counters (0,1) and (0,4) act as stash slots for a spare plate or a
cooked dish whose turn has not come yet.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'signal_worker_xp',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

_I32 = jnp.int32

# action ids
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5
# facing ids (observation channels 1..4)
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# facing id -> the movement action that produces that facing
_FACE_ACT = (A_UP, A_DOWN, A_RIGHT, A_LEFT)


def _face_action(want_face):
    return jnp.asarray(_FACE_ACT, dtype=_I32)[want_face]

# channel groups
C_SELF_FACE = 1          # 1..4
C_SELF_PLATE = 5
C_SELF_COOKED = 6
C_SELF_ING = 7           # 7..10
C_PART_POS = 11
C_PART_FACE = 12         # 12..15
C_GOAL = 23
C_RECIPE_CELL = 25
C_ITEM_PLATE = 32
C_ITEM_COOKED = 33
C_ITEM_ING = 34          # 34..37
C_DISPLAY_ING = 40       # 40..43
C_POT_TIMER = 44

# how long agent_0 waits after a delivery before trusting the partner signal
_SETTLE_STEPS = 10
_SETTLE_STABLE = 2
_SETTLE_TIMEOUT = 26
# the first commit of an episode also seeds the polarity baseline, and no dish
# can be ready before step 35 anyway, so it is worth waiting for a good read
_SETTLE_FIRST = 18
# how long a pose has to be held before it counts as a deliberate signal
_POSE_STABLE = 4
_POSE_HOLD = 5
# a real signal flips at most once per delivery; this much more is noise
_FLIP_SLACK = 4
# a single-counter presence convention needs the "absent" state to persist
_ABSENT_RUN = 11
# a decoded flip this long after our last delivery cannot be a real recipe
# change, so it is evidence that the channel we are reading is just noise
_NOISE_GAP = 28
_NOISE_MAX = 3
# Polarity evidence, in bits of log-odds favouring the *inverted* reading.
# A delivery that does not move the partner's signal is one bit against the
# polarity we just used (a correct delivery redraws the recipe, and the redraw
# differs half of the time, whereas a wrong delivery can never redraw it).  A
# signal change proves the polarity we just used was right.
_EV_PRIOR = 1            # mild prior for "left/low column means ingredient 0"
_EV_PROOF = 5
_EV_CAP = 6

# stash-slot contents
S_EMPTY, S_PLATE, S_DISH0, S_DISH1, S_JUNK = 0, 1, 2, 3, 4


def _tree_reset(init, memory, reset):
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(reset, a, b).astype(_I32), init, memory)


def _argmax2d(plane):
    flat = jnp.argmax(plane.reshape(-1))
    return (flat // 5).astype(_I32), (flat % 5).astype(_I32)


def _cell(obs, my_r, my_c, gr, gc):
    """Channel vector of global cell (gr, gc) plus a visibility flag."""
    i = gr - my_r + 2
    j = gc - my_c + 2
    vis = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    vec = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), :]
    return vec * vis.astype(vec.dtype), vis


def _goto(my_r, my_c, facing, tr, tc, want_face, arrive):
    """Walk to (tr, tc) inside an open rectangular room, then face and act.

    Every target facing used here points at a blocked cell, so the facing move
    only turns the player.
    """
    move = jnp.where(
        my_c < tc, A_RIGHT,
        jnp.where(my_c > tc, A_LEFT,
                  jnp.where(my_r < tr, A_DOWN, A_UP)))
    at = (my_r == tr) & (my_c == tc)
    return jnp.where(
        at,
        jnp.where(facing == want_face, arrive, _face_action(want_face)),
        move).astype(_I32)


def _has_item(vec, vis):
    return (jnp.sum(vec[C_ITEM_PLATE:C_ITEM_ING + 4]) > 0) & vis


# --------------------------------------------------------------------------
# agent_0 : top room worker + listener
# --------------------------------------------------------------------------

def init_agent_0():
    z = jnp.asarray(0, dtype=_I32)
    zc = jnp.zeros((4,), dtype=_I32)
    return {
        # pot (3,1) -- nominal ingredient 0
        'pcL': zc, 'ckL': z, 'tmL': z,
        # pot (3,3) -- nominal ingredient 1
        'pcR': zc, 'ckR': z, 'tmR': z,
        # private counters used as stash slots
        'st1': z,                                   # counter (0,1)
        'st4': z,                                   # counter (0,4)
        # recipe channel
        'b_last': z,                                # last decoded signal bit
        'stable': z,                                # consecutive stable reads
        'seen_l': z, 'seen_r': z,                   # counters the partner uses
        'run_l': z, 'run_r': z,                     # steps a counter stayed bare
        'p_col': z, 'p_face': z, 'p_hold': z,       # how still the partner is
        'fresh': z,                                 # good read since last delivery
        'flips': z, 'dels': z,                      # flip rate vs delivery rate
        'noise': z,                                 # implausible flips seen
        # polarity calibration
        'pol': z,                                   # 0: b==0 means ingredient 0
        'ev': jnp.asarray(-_EV_PRIOR, dtype=_I32),  # log-odds, >0 means invert
        'b_commit': z,                              # bit used at the last commit
        'q_commit': z,                              # was that read trustworthy
        'have_prev': z,                             # a previous commit exists
        'since_del': z,                             # steps since our last delivery
        'tgt': z,                                   # committed ingredient id
        'armed': jnp.asarray(1, dtype=_I32),        # a new commit is due
    }


def _read_pot(obs, my_r, my_c, gr, gc, pc_prev, ck_prev, tm_prev):
    """Pot contents/cooked/timer, from sight when possible and memory when not.

    Pot (3,1) leaves the 5x5 crop whenever agent_0 works at column 4, so its
    cook timer has to be carried forward privately.
    """
    vec, vis = _cell(obs, my_r, my_c, gr, gc)
    tm_pred = jnp.maximum(tm_prev - 1, 0)
    ck_pred = jnp.where((jnp.sum(pc_prev) >= 3) & (tm_pred == 0), 1, ck_prev)
    pc = jnp.where(vis, vec[C_ITEM_ING:C_ITEM_ING + 4], pc_prev).astype(_I32)
    ck = jnp.where(vis, (vec[C_ITEM_COOKED] > 0).astype(_I32), ck_pred)
    tm = jnp.where(vis, vec[C_POT_TIMER], tm_pred).astype(_I32)
    return pc, ck.astype(_I32), tm


def _read_slot(obs, my_r, my_c, gr, gc, prev):
    """Classify what sits on one of agent_0's two private counters."""
    vec, vis = _cell(obs, my_r, my_c, gr, gc)
    counts = vec[C_ITEM_ING:C_ITEM_ING + 4]
    any_item = jnp.sum(vec[C_ITEM_PLATE:C_ITEM_ING + 4]) > 0
    cooked = vec[C_ITEM_COOKED] > 0
    now = jnp.where(
        ~any_item, S_EMPTY,
        jnp.where(cooked & (counts[0] == 3), S_DISH0,
                  jnp.where(cooked & (counts[1] == 3), S_DISH1,
                            jnp.where(cooked, S_JUNK,
                                      jnp.where(vec[C_ITEM_PLATE] > 0,
                                                S_PLATE, S_JUNK)))))
    return jnp.where(vis, now, prev).astype(_I32)


def _pure_count(pc, k):
    """1 when this pot is stocked with ingredient `k` and nothing else."""
    n = jnp.sum(pc)
    return ((n > 0) & (pc[k] == n)).astype(_I32)


def _pot_score(pc, ck, tm, nominal, k):
    """How good this pot is as the source of the next dish of ingredient `k`."""
    n = jnp.sum(pc)
    pure = pc[k] == n
    idle = (n == 0) & (ck == 0) & (tm == 0)
    return jnp.where(
        (ck > 0) & pure & (n == 3), 8,
        jnp.where((tm > 0) & pure, 6,
                  jnp.where((n > 0) & (n < 3) & (tm == 0) & (ck == 0) & pure, 5,
                            jnp.where(idle & (nominal == k), 4,
                                      jnp.where(idle, 3, 0))))).astype(_I32)


def agent_0_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=_I32)
    mem = _tree_reset(init_agent_0(), memory, reset)

    # ---- self localisation: the delivery station (0,2) is in view from every
    # reachable top-room cell, which pins down our global position.
    gi, gj = _argmax2d(obs[:, :, C_GOAL])
    my_r = jnp.clip(2 - gi, 1, 2)
    my_c = jnp.clip(4 - gj, 1, 4)

    me = obs[2, 2, :]
    facing = jnp.argmax(me[C_SELF_FACE:C_SELF_FACE + 4]).astype(_I32)
    hi = me[C_SELF_ING:C_SELF_ING + 4]
    held_cooked = me[C_SELF_COOKED] > 0
    held_plate = (me[C_SELF_PLATE] > 0) & (~held_cooked)
    held_raw = (jnp.sum(hi) > 0) & (me[C_SELF_PLATE] == 0)

    # ---- recipe channel ---------------------------------------------------
    lvec, lvis = _cell(obs, my_r, my_c, 3, 2)
    rvec, rvis = _cell(obs, my_r, my_c, 3, 4)
    l_item = _has_item(lvec, lvis)
    r_item = _has_item(rvec, rvis)
    seen_l = jnp.maximum(mem['seen_l'], l_item.astype(_I32))
    seen_r = jnp.maximum(mem['seen_r'], r_item.astype(_I32))
    run_l = jnp.where(~lvis, mem['run_l'],
                      jnp.where(l_item, 0,
                                jnp.minimum(mem['run_l'] + 1, 99))).astype(_I32)
    run_r = jnp.where(~rvis, mem['run_r'],
                      jnp.where(r_item, 0,
                                jnp.minimum(mem['run_r'] + 1, 99))).astype(_I32)

    both_used = (seen_l > 0) & (seen_r > 0)
    # two shared counters in use: the marked side is the bit, an empty board is
    # just the partner carrying the marker across.
    two_ok = both_used & lvis & rvis & (l_item != r_item)
    two_b = r_item.astype(_I32)
    # a single counter in use: presence vs a *sustained* absence is the bit.
    one_l = (seen_l > 0) & (seen_r == 0) & lvis
    one_r = (seen_r > 0) & (seen_l == 0) & rvis
    one_ok = ((one_l & (l_item | (run_l >= _ABSENT_RUN)))
              | (one_r & (r_item | (run_r >= _ABSENT_RUN))))
    one_b = jnp.where(one_l, (~l_item).astype(_I32), r_item.astype(_I32))
    cnt_ok = two_ok | one_ok
    cnt_b = jnp.where(two_ok, two_b, one_b).astype(_I32)

    ppl = obs[:, :, C_PART_POS]
    pi, pj = _argmax2d(ppl)
    p_vis = jnp.max(ppl) > 0
    p_col = my_c + (pj - 2)
    p_face = jnp.argmax(obs[pi, pj, C_PART_FACE:C_PART_FACE + 4]).astype(_I32)
    p_face_r = obs[pi, pj, C_PART_FACE + 2] > 0
    p_face_l = obs[pi, pj, C_PART_FACE + 3] > 0
    # A pose only means something if the partner is holding it.  Someone who is
    # walking around doing its own job would otherwise be read as a signal that
    # flips every few steps.
    p_same = (p_col == mem['p_col']) & (p_face == mem['p_face'])
    p_hold = jnp.where(~p_vis, mem['p_hold'],
                       jnp.where(p_same, jnp.minimum(mem['p_hold'] + 1, 99),
                                 1)).astype(_I32)
    p_col_mem = jnp.where(p_vis, p_col, mem['p_col']).astype(_I32)
    p_face_mem = jnp.where(p_vis, p_face, mem['p_face']).astype(_I32)
    pose_b = jnp.where(p_col <= 1, 0,
                       jnp.where(p_col >= 3, 1,
                                 jnp.where(p_face_r, 1, 0))).astype(_I32)
    pose_ok = (p_vis & (p_hold >= _POSE_HOLD)
               & ((p_col != 2) | p_face_r | p_face_l))

    ev_valid = cnt_ok | pose_ok
    b_obs = jnp.where(cnt_ok, cnt_b, pose_b).astype(_I32)
    frozen = mem['noise'] >= _NOISE_MAX
    b_now = jnp.where(ev_valid & (~frozen), b_obs, mem['b_last']).astype(_I32)
    # A pose only reads while the partner is inside our 5x5 crop, which it often
    # is not, so remember that a trustworthy read has happened since the last
    # delivery instead of demanding one at the exact step we commit.
    fresh = jnp.maximum(mem['fresh'], ev_valid.astype(_I32)).astype(_I32)
    stable = jnp.where(
        ev_valid & (~frozen),
        jnp.where(b_obs == mem['b_last'], jnp.minimum(mem['stable'] + 1, 99), 1),
        jnp.minimum(mem['stable'] + 1, 99)).astype(_I32)
    since_del = jnp.minimum(mem['since_del'] + 1, 999).astype(_I32)
    # The recipe only ever changes right after one of our own correct
    # deliveries, so a real signal flips at most once per delivery.  Flips long
    # after the last delivery, or simply far more flips than deliveries, mean we
    # are reading somebody's ordinary movement; freezing the bit then leaves the
    # polarity accumulator to alternate, which beats chasing noise.
    flipped = (b_now != mem['b_last']) & (mem['have_prev'] > 0)
    implausible = flipped & (since_del >= _NOISE_GAP)
    flips = jnp.minimum(mem['flips'] + flipped.astype(_I32), 99).astype(_I32)
    over = flips >= mem['dels'] + _FLIP_SLACK
    noise = jnp.minimum(mem['noise'] + implausible.astype(_I32),
                        _NOISE_MAX).astype(_I32)
    noise = jnp.where(over, _NOISE_MAX, noise).astype(_I32)
    # A read is trustworthy when it comes off the shared counters, from a pose
    # the partner has held for a while, or from a channel we have already
    # written off as noise (in which case the bit is a constant).
    # Once the partner has shown it uses the shared counters, only a counter
    # read counts: an empty board means the marker is in transit, not a new
    # recipe, and mixing channels is how a spurious "change" becomes a false
    # proof.  A partner that never touches a counter is read by pose instead.
    cnt_user = (seen_l + seen_r) > 0
    qual = cnt_ok | frozen | (pose_ok & (~cnt_user))
    first = mem['have_prev'] == 0
    s_wait = jnp.where(first, _SETTLE_FIRST, _SETTLE_STEPS)
    s_stop = jnp.where(first, _SETTLE_FIRST + _SETTLE_TIMEOUT, _SETTLE_TIMEOUT)
    settled = (((since_del >= s_wait) & (stable >= _SETTLE_STABLE)
                & (fresh > 0) & (cnt_ok | (~cnt_user)))
               | (since_del >= s_stop))

    # ---- polarity calibration, committed once per delivery cycle ----------
    do_commit = (mem['armed'] > 0) & settled
    changed = (b_now != mem['b_commit']) & (mem['have_prev'] > 0)
    nochange = (mem['have_prev'] > 0) & (~changed)
    # Treating a change as proof is near-irreversible, so only do it when both
    # ends of the comparison were trustworthy reads; a change between two shaky
    # reads says nothing.  A no-change is safe to act on either way: at worst it
    # makes the accumulator oscillate, which is the information-free optimum.
    proof = changed & qual & (mem['q_commit'] > 0)
    step = jnp.where(mem['pol'] == 0, 1, -1).astype(_I32)
    ev_new = jnp.where(
        proof, -_EV_PROOF * step,
        jnp.where(changed, mem['ev'],
                  jnp.where(nochange,
                            jnp.clip(mem['ev'] + step, -_EV_CAP, _EV_CAP),
                            mem['ev']))).astype(_I32)
    pol_new = (ev_new > 0).astype(_I32)

    pol = jnp.where(do_commit, pol_new, mem['pol']).astype(_I32)
    ev = jnp.where(do_commit, ev_new, mem['ev']).astype(_I32)
    b_commit = jnp.where(do_commit, b_now, mem['b_commit']).astype(_I32)
    q_commit = jnp.where(do_commit, qual.astype(_I32),
                         mem['q_commit']).astype(_I32)
    have_prev = jnp.maximum(mem['have_prev'], do_commit.astype(_I32))
    armed = jnp.where(do_commit, 0, mem['armed']).astype(_I32)
    tgt_mem = jnp.where(do_commit, jnp.bitwise_xor(b_now, pol_new),
                        mem['tgt']).astype(_I32)
    committed = armed == 0
    # before the commit the raw reading only steers pot refills, never a dish
    tgt = jnp.where(committed, tgt_mem,
                    jnp.bitwise_xor(b_now, pol)).astype(_I32)

    # ---- pots and stash slots --------------------------------------------
    pcL, ckL, tmL = _read_pot(obs, my_r, my_c, 3, 1,
                              mem['pcL'], mem['ckL'], mem['tmL'])
    pcR, ckR, tmR = _read_pot(obs, my_r, my_c, 3, 3,
                              mem['pcR'], mem['ckR'], mem['tmR'])
    st1 = _read_slot(obs, my_r, my_c, 0, 1, mem['st1'])
    st4 = _read_slot(obs, my_r, my_c, 0, 4, mem['st4'])
    free1 = st1 == S_EMPTY
    free4 = st4 == S_EMPTY
    any_free = free1 | free4

    nL = jnp.sum(pcL)
    nR = jnp.sum(pcR)
    fillL = (nL < 3) & (ckL == 0) & (tmL == 0)
    fillR = (nR < 3) & (ckR == 0) & (tmR == 0)

    sL = _pot_score(pcL, ckL, tmL, 0, tgt)
    sR = _pot_score(pcR, ckR, tmR, 1, tgt)
    tp_is_L = sL >= sR
    tp_c = jnp.where(tp_is_L, 1, 3).astype(_I32)
    op_c = jnp.where(tp_is_L, 3, 1).astype(_I32)
    tp_ck = jnp.where(tp_is_L, ckL, ckR) > 0
    op_ck = jnp.where(tp_is_L, ckR, ckL) > 0
    tp_n = jnp.where(tp_is_L, nL, nR)
    op_pc = jnp.where(tp_is_L, pcR, pcL)
    op_pure = (op_pc[0] == 3) | (op_pc[1] == 3)
    tp_tm = jnp.where(tp_is_L, tmL, tmR)
    tp_cooking = (tp_tm > 0) | (tp_n >= 3)

    # which pot to top up, and with what: finish a partly filled pot first,
    # then the pot the current recipe wants, then the other one.  A pot a
    # partner has spoiled is worth topping up only to cook it out and bin it,
    # so it never gets priority over a clean one.
    badL = (pcL[2] + pcL[3] > 0) | ((pcL[0] > 0) & (pcL[1] > 0))
    badR = (pcR[2] + pcR[3] > 0) | ((pcR[0] > 0) & (pcR[1] > 0))
    cleanL = fillL & (~badL)
    cleanR = fillR & (~badR)
    fillL = fillL & ((~badL) | any_free)
    fillR = fillR & ((~badR) | any_free)
    any_clean = cleanL | cleanR
    fillL = jnp.where(any_clean, cleanL, fillL)
    fillR = jnp.where(any_clean, cleanR, fillR)
    tp_bad = jnp.where(tp_is_L, badL, badR)
    part_L = fillL & (nL > 0) & (~badL)
    part_R = fillR & (nR > 0) & (~badR)
    fill_L = jnp.where(
        part_L, True,
        jnp.where(part_R, False,
                  jnp.where(tp_is_L & fillL, True,
                            jnp.where(fillR, False, fillL))))
    fill_any = fillL | fillR
    f_pc = jnp.where(fill_L, pcL, pcR)
    f_n = jnp.where(fill_L, nL, nR)
    # Stock of each dish kind already cooked, cooking or started.  Two servings
    # of one kind are worthless while the other kind is missing, so an empty pot
    # is started on whichever kind is in shorter supply, the current recipe
    # breaking ties.  Long runs of a repeated recipe then use both pots.
    av0 = (_pure_count(pcL, 0) + _pure_count(pcR, 0)
           + (st1 == S_DISH0).astype(_I32) + (st4 == S_DISH0).astype(_I32))
    av1 = (_pure_count(pcL, 1) + _pure_count(pcR, 1)
           + (st1 == S_DISH1).astype(_I32) + (st4 == S_DISH1).astype(_I32))
    f_nominal = jnp.where(fill_L, 0, 1).astype(_I32)
    want = jnp.where(jnp.where(fill_L, tp_is_L, ~tp_is_L),
                     tgt, f_nominal).astype(_I32)
    # keeping one serving of each kind standing by is worth more than raw
    # throughput, so only override the dedication when this kind is already
    # saturated and the other one is missing entirely
    av_want = jnp.where(want == 0, av0, av1)
    av_other = jnp.where(want == 0, av1, av0)
    bal_ing = jnp.where((av_want >= 2) & (av_other == 0), 1 - want,
                        want).astype(_I32)
    fill_ing = jnp.where(f_n > 0, (f_pc[1] > f_pc[0]).astype(_I32),
                         bal_ing).astype(_I32)

    # ---- candidate moves --------------------------------------------------
    a_deliver = _goto(my_r, my_c, facing, 1, 2, F_UP, A_INT)
    a_plate = _goto(my_r, my_c, facing, 1, 3, F_UP, A_INT)
    a_s1 = _goto(my_r, my_c, facing, 1, 1, F_UP, A_INT)
    a_s4 = _goto(my_r, my_c, facing, 1, 4, F_UP, A_INT)
    a_tp = _goto(my_r, my_c, facing, 2, tp_c, F_DOWN, A_INT)
    # Idle steps are free, but a partner that signals only with its body sits in
    # row 4 and is outside our crop from most of the room.  When we are waiting
    # anyway and still owe a commit, wait at (2,2), the one cell that sees every
    # column of the far room.
    seek = (armed > 0) & (fresh == 0) & (~cnt_user)
    a_tp_wait = jnp.where(seek, _goto(my_r, my_c, facing, 2, 2, F_DOWN, A_STAY),
                          _goto(my_r, my_c, facing, 2, tp_c, F_DOWN, A_STAY))
    a_op = _goto(my_r, my_c, facing, 2, op_c, F_DOWN, A_INT)
    src_r = jnp.where(fill_ing == 0, 1, 2).astype(_I32)
    src_c = jnp.where(fill_L, 1, 4).astype(_I32)
    src_f = jnp.where(fill_L, F_LEFT, F_RIGHT).astype(_I32)
    a_src = _goto(my_r, my_c, facing, src_r, src_c, src_f, A_INT)

    # holding a raw ingredient: into a pot that already agrees with it
    raw_k = (hi[1] > hi[0]).astype(_I32)
    okL_raw = fillL & ((nL == 0) | (pcL[raw_k] == nL) | badL)
    okR_raw = fillR & ((nR == 0) | (pcR[raw_k] == nR) | badR)
    dep_L = jnp.where(raw_k == 0,
                      jnp.where(okL_raw, True, ~okR_raw),
                      jnp.where(okR_raw, False, okL_raw))
    a_dep = _goto(my_r, my_c, facing, 2, jnp.where(dep_L, 1, 3), F_DOWN, A_INT)
    act_raw = jnp.where(okL_raw | okR_raw, a_dep, a_tp_wait)

    # holding a cooked dish: deliver it when it is what the recipe asks for,
    # park a sound dish of the other kind, bin anything unservable
    dish_tgt = hi[tgt] == 3
    dish_pure = (hi[0] == 3) | (hi[1] == 3)
    a_park = jnp.where(free1, a_s1, jnp.where(free4, a_s4, a_deliver))
    a_bin = jnp.where(free4, a_s4, jnp.where(free1, a_s1, a_deliver))
    act_dish = jnp.where(dish_tgt & committed, a_deliver,
                         jnp.where(dish_pure, a_park, a_bin))

    # holding a plate: scoop the pot we are waiting on, or free the other pot
    tp_pc = jnp.where(tp_is_L, pcL, pcR)
    tp_pure = (tp_pc[0] == 3) | (tp_pc[1] == 3)
    act_plate = jnp.where(
        tp_ck & (tp_pure | any_free),
        jnp.where(committed, a_tp, a_tp_wait),
        jnp.where(committed & op_ck & any_free & (op_pure | tp_bad), a_op,
                  jnp.where(tp_cooking, a_tp_wait,
                            jnp.where(free1, a_s1,
                                      jnp.where(free4, a_s4, a_tp_wait)))))

    # empty handed: fetch a stashed dish, a plate for a ready pot, or refill
    want1 = committed & (st1 == S_DISH0 + tgt)
    want4 = committed & (st4 == S_DISH0 + tgt)
    a_getplate = jnp.where(st1 == S_PLATE, a_s1,
                           jnp.where(st4 == S_PLATE, a_s4, a_plate))
    act_empty = jnp.where(
        want1, a_s1,
        jnp.where(want4, a_s4,
                  jnp.where(committed & tp_ck, a_getplate,
                            jnp.where(fill_any, a_src, a_getplate))))

    action = jnp.where(
        held_cooked, act_dish,
        jnp.where(held_plate, act_plate,
                  jnp.where(held_raw, act_raw, act_empty))).astype(_I32)

    delivered = (held_cooked & dish_tgt & committed & (my_r == 1) & (my_c == 2)
                 & (facing == F_UP) & (action == A_INT))
    next_mem = {
        'pcL': pcL, 'ckL': ckL, 'tmL': tmL,
        'pcR': pcR, 'ckR': ckR, 'tmR': tmR,
        'st1': st1, 'st4': st4,
        'b_last': b_now,
        'stable': stable,
        'seen_l': seen_l, 'seen_r': seen_r,
        'run_l': run_l, 'run_r': run_r,
        'p_col': p_col_mem, 'p_face': p_face_mem, 'p_hold': p_hold,
        'fresh': jnp.where(delivered, 0, fresh).astype(_I32),
        'flips': flips,
        'dels': jnp.minimum(mem['dels'] + delivered.astype(_I32), 99).astype(_I32),
        'noise': noise,
        'pol': pol,
        'ev': ev,
        'b_commit': b_commit,
        'q_commit': q_commit,
        'have_prev': have_prev,
        'since_del': jnp.where(delivered, 0, since_del).astype(_I32),
        'tgt': tgt_mem,
        'armed': jnp.where(delivered, 1, armed).astype(_I32),
    }
    trace = {
        'row': my_r,
        'col': my_c,
        'bit': b_now,
        'pol': pol,
        'ev': ev,
        'tgt': tgt,
        'nL': nL.astype(_I32),
        'nR': nR.astype(_I32),
        'ckL': ckL, 'ckR': ckR,
        'st1': st1, 'st4': st4,
        'noise': noise,
        'delivered': delivered.astype(_I32),
    }
    return action, next_mem, trace


# --------------------------------------------------------------------------
# agent_1 : bottom room recipe broadcaster
# --------------------------------------------------------------------------

def init_agent_1():
    z = jnp.asarray(0, dtype=_I32)
    return {
        'bit': z,      # last seen recipe bit
        'lo': z,       # counter (3,2): 0 empty, 1 our marker, 2 foreign item
        'ro': z,       # counter (3,4)
    }


def _counter_state(vec, vis, prev):
    """0 empty / 1 our own ingredient marker / 2 somebody else's item."""
    ing = vec[C_ITEM_ING:C_ITEM_ING + 4]
    any_item = jnp.sum(vec[C_ITEM_PLATE:C_ITEM_ING + 4]) > 0
    mine = (ing[3] > 0) & (vec[C_ITEM_PLATE] == 0) & (vec[C_ITEM_COOKED] == 0)
    now = jnp.where(any_item, jnp.where(mine, 1, 2), 0).astype(_I32)
    return jnp.where(vis, now, prev).astype(_I32)


def agent_1_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=_I32)
    mem = _tree_reset(init_agent_1(), memory, reset)

    # ---- localisation via the always-on recipe indicator at (6,2)
    plane = obs[:, :, C_RECIPE_CELL]
    ri, rj = _argmax2d(plane)
    r_vis = jnp.max(plane) > 0
    my_r = jnp.clip(8 - ri, 4, 5)
    my_c = jnp.clip(4 - rj, 1, 4)

    display = obs[ri, rj, C_DISPLAY_ING:C_DISPLAY_ING + 4]
    shown = (jnp.sum(display) > 0) & r_vis
    bit = jnp.where(shown, (display[1] > display[0]).astype(_I32), mem['bit'])

    me = obs[2, 2, :]
    facing = jnp.argmax(me[C_SELF_FACE:C_SELF_FACE + 4]).astype(_I32)
    held_ing = me[C_SELF_ING:C_SELF_ING + 4]
    holds_marker = (held_ing[3] > 0) & (me[C_SELF_PLATE] == 0)
    holds_any = (jnp.sum(held_ing) > 0) | (me[C_SELF_PLATE] > 0)
    holds_other = holds_any & (~holds_marker)

    lvec, lvis = _cell(obs, my_r, my_c, 3, 2)
    rvec, rvis = _cell(obs, my_r, my_c, 3, 4)
    lo = _counter_state(lvec, lvis, mem['lo'])
    ro = _counter_state(rvec, rvis, mem['ro'])

    want_state = jnp.where(bit == 1, ro, lo)
    other_state = jnp.where(bit == 1, lo, ro)
    want_stand_c = jnp.where(bit == 1, 4, 2).astype(_I32)
    other_stand_c = jnp.where(bit == 1, 2, 4).astype(_I32)

    # spill counters used only inside the bottom room, invisible to agent_0
    d1vec, d1vis = _cell(obs, my_r, my_c, 5, 0)
    dump_free_1 = (jnp.sum(d1vec[C_ITEM_PLATE:C_ITEM_ING + 4]) == 0) | (~d1vis)
    a_dump1 = _goto(my_r, my_c, facing, 5, 1, F_LEFT, A_INT)
    a_dump2 = _goto(my_r, my_c, facing, 5, 1, F_DOWN, A_INT)
    a_dump = jnp.where(dump_free_1, a_dump1, a_dump2)

    a_place = _goto(my_r, my_c, facing, 4, want_stand_c, F_UP, A_INT)
    a_clear = _goto(my_r, my_c, facing, 4, other_stand_c, F_UP, A_INT)
    # nearest ingredient-3 supply, used purely as a marker source
    near_right = my_c >= 3
    a_take = jnp.where(
        near_right,
        _goto(my_r, my_c, facing, 4, 4, F_RIGHT, A_INT),
        _goto(my_r, my_c, facing, 4, 1, F_LEFT, A_INT))
    # parked broadcast pose: extreme column + matching facing
    a_park = jnp.where(
        bit == 1,
        _goto(my_r, my_c, facing, 4, 4, F_RIGHT, A_STAY),
        _goto(my_r, my_c, facing, 4, 1, F_LEFT, A_STAY))

    action = jnp.where(
        holds_marker,
        jnp.where(want_state == 0, a_place, a_dump),
        jnp.where(
            holds_other, a_dump,
            jnp.where(other_state == 1, a_clear,
                      jnp.where(want_state == 0, a_take, a_park)))).astype(_I32)

    # keep the counter memory consistent with the interaction we are about to do
    place_now = holds_marker & (want_state == 0) & (action == A_INT)
    clear_now = ((~holds_any) & (other_state == 1) & (action == A_INT)
                 & (my_r == 4) & (my_c == other_stand_c) & (facing == F_UP))
    lo_next = jnp.where(place_now & (bit == 0), 1,
                        jnp.where(clear_now & (bit == 1), 0, lo)).astype(_I32)
    ro_next = jnp.where(place_now & (bit == 1), 1,
                        jnp.where(clear_now & (bit == 0), 0, ro)).astype(_I32)

    next_mem = {'bit': bit, 'lo': lo_next, 'ro': ro_next}
    trace = {
        'row': my_r,
        'col': my_c,
        'bit': bit,
        'lo': lo_next,
        'ro': ro_next,
        'holds': holds_marker.astype(_I32),
    }
    return action, next_mem, trace
