"""Overcooked-V2 `test_time_simple`: split kitchen with a counter-row recipe signal.

Geometry facts that drive everything below:

* The two rooms are physically disconnected.  agent_0 only reaches the recipe
  indicator (2,0), the ingredient-2 piles (0,2)/(4,2) and the wall counters
  (1,3),(2,3),(3,3).  agent_1 alone reaches the pot (2,4), the plate piles
  (1,4)/(3,4), the ingredient piles (1,7)/(3,7) and the delivery station (2,7).
* With view radius 2 the seats never see each other, and agent_1 never sees the
  recipe indicator.  The only cells both seats perceive are the column-3
  counters: agent_0 can put items there, and agent_1 sees them from column 5.

Protocol (chosen to be as literal and geometry-driven as possible, so that an
independently written partner has a good chance of reading or producing it):
the requested ingredient is marked by leaving one item on the counter whose row
matches that ingredient's supply row - (1,3) for ingredient 0, whose pile is at
(1,7), and (3,3) for ingredient 1, whose pile is at (3,7).

Neither seat assumes the other honours that convention.

agent_1 treats that convention as a starting hypothesis about which display asks
for ingredient 0, not as an assumption.  It moves the hypothesis after a run of
deliveries that leave the display unchanged - the signature of a kitchen stuck
on a misread, since only a correct delivery can resample the recipe - and fixes
it outright when the display does change, which proves that the newly shown
marker asks for the ingredient that was not just delivered.  With no marker in
sight it alternates its guess, which strictly beats repeating one for the same
reason: a wrong delivery leaves the request standing.

agent_0 sees the pot from column 2, so it knows exactly what the partner cooked.
When the partner fills the pot with an ingredient the current display could not
have requested, and the display had been correct long enough for the partner to
have seen it, agent_0 moves on to the next of four encodings.  A partner that
agrees with the convention never triggers this, so it costs nothing to agree.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'counter_row_signal_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids as they appear in observation channels 1..4.
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
_FACE_ACTION = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# Counters agent_1 can use as a scratch surface, as (cell, stand, facing).
_STASH = (
    ((0, 5), (1, 5), F_UP),
    ((4, 5), (3, 5), F_DOWN),
    ((0, 6), (1, 6), F_UP),
    ((4, 6), (3, 6), F_DOWN),
)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _locate(obs, channel, gr, gc, r_lo, r_hi, c_lo, c_hi):
    """Self position from a landmark that is visible from every room cell."""
    idx = jnp.arange(5, dtype=jnp.int32)
    mask = (obs[:, :, channel] > 0).astype(jnp.int32)
    total = mask.sum()
    i = (mask * idx[:, None]).sum() // jnp.maximum(total, 1)
    j = (mask * idx[None, :]).sum() // jnp.maximum(total, 1)
    i = jnp.where(total > 0, i, 2)
    j = jnp.where(total > 0, j, 2)
    return (jnp.clip(gr - i + 2, r_lo, r_hi).astype(jnp.int32),
            jnp.clip(gc - j + 2, c_lo, c_hi).astype(jnp.int32))


def _read(obs, sr, sc, gr, gc):
    """Feature vector of global cell (gr,gc); zeros (and ok=False) if unseen."""
    i = gr - sr + 2
    j = gc - sc + 2
    ok = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    vec = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), :]
    return vec * ok.astype(jnp.int32), ok


def _has_item(vec):
    return vec[29:34].sum() > 0


def _step_toward(sr, sc, tr, tc, face, facing, want_interact):
    """Walk (vertically first) to (tr,tc), turn to `face`, then act.

    Every cell this policy turns toward is impassable, so the turning move can
    never displace the agent.
    """
    dr = tr - sr
    dc = tc - sc
    move = jnp.where(dr < 0, A_UP,
                     jnp.where(dr > 0, A_DOWN,
                               jnp.where(dc < 0, A_LEFT, A_RIGHT)))
    arrived = (dr == 0) & (dc == 0)
    turn = jnp.asarray(_FACE_ACTION, dtype=jnp.int32)[face]
    here = jnp.where(facing == face,
                     jnp.where(want_interact, A_INTERACT, A_STAY),
                     turn)
    return jnp.where(arrived, here, move).astype(jnp.int32)


# --------------------------------------------------------------------------
# agent_0: reads the recipe, maintains the counter signal.
# --------------------------------------------------------------------------

_Z_RECIPE = 0     # last recipe seen on the indicator
_Z_ROW = 1        # row currently used for the signal (diagnostic)
_Z_ENC = 2        # which of the four encodings is currently in use
_Z_POTN = 3       # pot fill seen on the previous step
_Z_STABLE = 4     # steps the displayed signal has been exactly correct
_Z_SEARCH = 5     # a misread has been proven, so keep searching quickly
_M0 = 6

# A partner that inserts the wrong ingredient is only taken as proof of a
# misread when the signal has been correct for longer than a cook cycle's head
# start, i.e. when the request did not change at the previous delivery.  A
# partner that simply read the marker before it moved must not flip us, because
# two seats adapting to each other's stale evidence would thrash.
_STABLE_PROOF = 16
# Once one misread is proven the partner is known to use another convention,
# so further evidence needs only to rule out a marker the partner had not yet
# had time to see.
_STABLE_QUICK = 6


def init_agent_0():
    return jnp.zeros((_M0,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, jnp.zeros((_M0,), dtype=jnp.int32), memory)

    sr, sc = _locate(obs, 23, 2, 0, 1, 3, 1, 2)
    facing = _facing(obs)
    held = obs[2, 2, 5:10].sum() > 0

    recipe, _ = _read(obs, sr, sc, 2, 0)
    shown = recipe[36:39].sum() > 0
    want_true = jnp.where(recipe[37] > recipe[36], 1, 0)
    want_true = jnp.where(shown, want_true, mem[_Z_RECIPE]).astype(jnp.int32)

    # From column 2 the pot is in view, so the partner's ingredient choice is
    # observable.  A pot that starts filling with the wrong ingredient while the
    # signal has been displayed correctly for a while means the partner reads
    # the convention the other way round; encode it their way from now on.
    pot, pot_ok = _read(obs, sr, sc, 2, 4)
    pot_n = pot[31:34].sum()
    pot_type = jnp.argmax(pot[31:34]).astype(jnp.int32)
    new_fill = pot_ok & (mem[_Z_POTN] == 0) & (pot_n > 0)
    proof = jnp.where(mem[_Z_SEARCH] > 0, _STABLE_QUICK, _STABLE_PROOF)
    misread = new_fill & (pot_type != want_true) & (mem[_Z_STABLE] >= proof)
    searching = jnp.maximum(mem[_Z_SEARCH], misread.astype(jnp.int32))
    # Proof that the partner reads the marker differently, so move on to the
    # next encoding: row of the matching supply, its mirror, counter row 1+k,
    # and that mirror.  A partner reading any of them is eventually matched,
    # and a partner that agrees is never disturbed because it never misreads.
    enc = jnp.where(misread, (mem[_Z_ENC] + 1) % 4, mem[_Z_ENC]).astype(jnp.int32)

    want = jnp.bitwise_xor(want_true, enc % 2)
    sig_row = jnp.where(enc < 2,
                        jnp.where(want == 0, 1, 3),
                        jnp.where(want == 0, 1, 2))

    c1, _ = _read(obs, sr, sc, 1, 3)
    c2, _ = _read(obs, sr, sc, 2, 3)
    c3, _ = _read(obs, sr, sc, 3, 3)
    o1, o2, o3 = _has_item(c1), _has_item(c2), _has_item(c3)
    desired_occupied = jnp.where(sig_row == 1, o1, jnp.where(sig_row == 2, o2, o3))
    stale_row = jnp.where(o1 & (sig_row != 1), 1,
                          jnp.where(o2 & (sig_row != 2), 2,
                                    jnp.where(o3 & (sig_row != 3), 3, 0)))

    spare_a, ok_a = _read(obs, sr, sc, 1, 0)
    spare_b, ok_b = _read(obs, sr, sc, 3, 0)
    free_a = ok_a & jnp.logical_not(_has_item(spare_a))
    free_b = ok_b & jnp.logical_not(_has_item(spare_b))

    place = _step_toward(sr, sc, sig_row, 2, F_RIGHT, facing, True)
    clear = _step_toward(sr, sc, jnp.maximum(stale_row, 1), 2, F_RIGHT, facing, True)
    guard = _step_toward(sr, sc, sig_row, 2, F_RIGHT, facing, False)
    take = jnp.where(sr <= 2,
                     _step_toward(sr, sc, 1, 2, F_UP, facing, True),
                     _step_toward(sr, sc, 3, 2, F_DOWN, facing, True))
    drop = jnp.where(free_a, _step_toward(sr, sc, 1, 1, F_LEFT, facing, True),
                     jnp.where(free_b, _step_toward(sr, sc, 3, 1, F_LEFT, facing, True),
                               jnp.asarray(A_STAY, dtype=jnp.int32)))

    action = jnp.where(
        held,
        jnp.where(desired_occupied, drop, place),
        jnp.where(stale_row > 0, clear,
                  jnp.where(desired_occupied, guard, take)),
    ).astype(jnp.int32)

    display_ok = desired_occupied & (stale_row == 0)
    stable = jnp.where(display_ok, jnp.minimum(mem[_Z_STABLE] + 1, 99), 0)

    mem = mem.at[_Z_RECIPE].set(want_true)
    mem = mem.at[_Z_ROW].set(sig_row.astype(jnp.int32))
    mem = mem.at[_Z_ENC].set(enc)
    mem = mem.at[_Z_SEARCH].set(searching.astype(jnp.int32))
    mem = mem.at[_Z_POTN].set(jnp.where(pot_ok, pot_n, mem[_Z_POTN]).astype(jnp.int32))
    mem = mem.at[_Z_STABLE].set(stable.astype(jnp.int32))
    trace = {
        'recipe': want_true,
        'signal_row': sig_row.astype(jnp.int32),
        'encoding': enc,
        'holding': held.astype(jnp.int32),
        'row': sr,
        'col': sc,
    }
    return action, mem, trace


# --------------------------------------------------------------------------
# agent_1: the whole kitchen, plus a permissive reader of the signal.
# --------------------------------------------------------------------------

# The display is classified into four patterns: only the top counter, only the
# middle one, only the bottom one, or several at once.  The reading is carried
# by one hypothesis - which pattern asks for ingredient 0, everything else
# asking for ingredient 1 - so it is always self-consistent.  Starting at "the
# top counter" reproduces the convention above and also reads a partner that
# numbers the counters 1+k, or one that marks two counters for ingredient 1.
_P_TOP, _P_MID, _P_BOTTOM, _P_MANY = 0, 1, 2, 3

# memory slots
_S_SIG = 0        # pattern last seen on the counters (-1 unknown)
_S_STREAK = 1     # deliveries in a row that left the pattern unchanged
_S_COMMIT = 2     # pattern read for the previous cook
_S_LAST = 3       # ingredient of the previous delivery (-1 none)
_S_DISH = 4       # held a plated dish on the previous step
_S_SEEN = 5       # a pattern was read at least once
_S_CLOCK = 6      # steps since episode start or last delivery
_S_COOK = 7       # ingredient currently in the pot / just delivered
_S_NDEL = 8       # number of deliveries
_S_PEND = 9       # a delivery is waiting for its pattern-change check
_S_COIN = 10      # stable random first guess
_S_ZERO = 11      # pattern currently believed to ask for ingredient 0
_S_MOVED = 12     # a delivery has been seen to change the pattern
_S_MOVES = 13     # number of times the hypothesis has been moved
_M1 = 14

_WAIT_LIMIT = 16
_CHECK_DELAY = 5
# A delivery that leaves the display unchanged is weak evidence that the reading
# is wrong and the kitchen is stuck, because only a correct delivery resamples
# the recipe and a resample repeats half the time.  Act on three such deliveries
# while nothing is known, on five after the hypothesis has already been moved,
# and hardly ever once a display change has shown the reading to be right.
_MOVE_AFTER_COLD = 3
_MOVE_AFTER_AGAIN = 5
_MOVE_AFTER_WARM = 8


def init_agent_1():
    mem = jnp.zeros((_M1,), dtype=jnp.int32)
    mem = mem.at[_S_SIG].set(-1)
    mem = mem.at[_S_COMMIT].set(-1)
    mem = mem.at[_S_LAST].set(-1)
    mem = mem.at[_S_COOK].set(-1)
    mem = mem.at[_S_PEND].set(1)
    mem = mem.at[_S_ZERO].set(_P_TOP)
    return mem


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, init_agent_1(), memory)

    sr, sc = _locate(obs, 22, 2, 4, 1, 3, 5, 6)
    facing = _facing(obs)

    inv = obs[2, 2, 5:10]
    hold_plate = inv[0] > 0
    hold_cooked = inv[1] > 0
    hold_dish = hold_plate & hold_cooked
    hold_ing = inv[2:5].sum() > 0
    hold_type = jnp.argmax(inv[2:5]).astype(jnp.int32)
    hold_any = inv.sum() > 0

    pot, _ = _read(obs, sr, sc, 2, 4)
    pot_n = pot[31:34].sum()
    pot_ready = pot[30] > 0
    pot_cooking = jnp.logical_not(pot_ready) & (pot[39] > 0)

    # --- delivery bookkeeping -------------------------------------------------
    delivered = (mem[_S_DISH] > 0) & jnp.logical_not(hold_any)
    last_delivered = jnp.where(delivered, mem[_S_COOK], mem[_S_LAST])
    n_delivered = mem[_S_NDEL] + delivered.astype(jnp.int32)
    pending = jnp.where(delivered, 1, mem[_S_PEND])
    clock = jnp.where(delivered, 0, jnp.minimum(mem[_S_CLOCK] + 1, 999))
    sig_mem = jnp.where(delivered, -1, mem[_S_SIG])

    # --- read the signal ------------------------------------------------------
    can_read = sc == 5
    k1, _ = _read(obs, sr, sc, 1, 3)
    k2, _ = _read(obs, sr, sc, 2, 3)
    k3, _ = _read(obs, sr, sc, 3, 3)
    b1, b2, b3 = _has_item(k1), _has_item(k2), _has_item(k3)
    single = jnp.stack([b1 & jnp.logical_not(b2) & jnp.logical_not(b3),
                        b2 & jnp.logical_not(b1) & jnp.logical_not(b3),
                        b3 & jnp.logical_not(b1) & jnp.logical_not(b2)])
    pattern = jnp.where(single[0], _P_TOP,
                        jnp.where(single[1], _P_MID,
                                  jnp.where(single[2], _P_BOTTOM, _P_MANY)))
    shown = can_read & (b1 | b2 | b3)
    pattern = jnp.where(shown, pattern, -1).astype(jnp.int32)
    nothing_shown = can_read & jnp.logical_not(b1 | b2 | b3)
    fresh = pattern >= 0
    sig_mem = jnp.where(fresh, pattern, jnp.where(nothing_shown, -1, sig_mem))
    seen_ever = jnp.maximum(mem[_S_SEEN], fresh.astype(jnp.int32))

    # Evidence arrives once per delivery, once the partner has had time to move
    # its marker.  A display that changed proves the delivery was correct, since
    # only a correct delivery resamples the recipe; a display that did not
    # change is weak evidence that the reading is wrong.  When the evidence
    # mounts, the display we were stuck on becomes the one taken to ask for
    # ingredient 0, unless it already was, in which case no display does.
    prev = mem[_S_COMMIT]
    check = (pending > 0) & fresh & (prev >= 0) & (clock >= _CHECK_DELAY)
    same = check & (pattern == prev)
    streak = jnp.where(same, mem[_S_STREAK] + 1,
                       jnp.where(check, 0, mem[_S_STREAK]))
    moved = jnp.maximum(mem[_S_MOVED],
                        (check & (pattern != prev)).astype(jnp.int32))
    limit = jnp.where(moved > 0, _MOVE_AFTER_WARM,
                      jnp.where(mem[_S_MOVES] > 0, _MOVE_AFTER_AGAIN,
                                _MOVE_AFTER_COLD))
    correcting = same & (streak >= limit)
    streak = jnp.where(correcting, 0, streak)
    # A changed display carries certainty rather than evidence: the request now
    # shown differs from the one just cooked, either because a correct delivery
    # resampled it or because the partner re-marked it after a wrong one.  So
    # the new display asks for the ingredient that was not just delivered.
    proof = check & (pattern != prev) & (last_delivered >= 0)
    zero_pat = jnp.where(correcting,
                         jnp.where(prev == mem[_S_ZERO], -1, prev),
                         jnp.where(proof & (last_delivered == 1), pattern,
                                   mem[_S_ZERO])).astype(jnp.int32)
    moves = mem[_S_MOVES] + correcting.astype(jnp.int32)
    settled = (pending > 0) & fresh & (clock >= _CHECK_DELAY)
    commit = jnp.where(settled, pattern, prev)
    pending = jnp.where(settled, 0, pending)

    # --- what to cook ---------------------------------------------------------
    coin = jnp.where(reset, jax.random.randint(rng, (), 0, 2).astype(jnp.int32),
                     mem[_S_COIN])
    decoded = jnp.where(sig_mem >= 0, jnp.where(sig_mem == zero_pat, 0, 1), -1)
    # Without a signal, alternate: a wrong delivery does not resample the recipe.
    fallback = jnp.where(last_delivered >= 0, 1 - last_delivered, coin)
    target = jnp.where(decoded >= 0, decoded, fallback)
    target = jnp.where(pot_n > 0, jnp.where(pot[32] > pot[31], 1, 0), target)
    target = target.astype(jnp.int32)

    wait_limit = jnp.where((seen_ever > 0) | (n_delivered == 0), _WAIT_LIMIT, 0)
    waiting = (pot_n == 0) & (sig_mem < 0) & (clock <= wait_limit)

    # --- scratch counters -----------------------------------------------------
    free_here = []
    holds_target = []
    stash_step = []
    take_step = []
    for (cell, stand, face) in _STASH:
        vec, ok = _read(obs, sr, sc, cell[0], cell[1])
        free_here.append(ok & jnp.logical_not(_has_item(vec)))
        wanted = jnp.take(vec[31:34], target) > 0
        holds_target.append(ok & wanted & (vec[29] == 0) & (vec[30] == 0))
        act = _step_toward(sr, sc, stand[0], stand[1], face, facing, True)
        stash_step.append(act)
        take_step.append(act)

    drop_act = stash_step[3]
    for k in (2, 1, 0):
        drop_act = jnp.where(free_here[k], stash_step[k], drop_act)
    any_free = free_here[0] | free_here[1] | free_here[2] | free_here[3]

    pot_act = _step_toward(sr, sc, 2, 5, F_LEFT, facing, True)
    pot_hold = _step_toward(sr, sc, 2, 5, F_LEFT, facing, False)
    deliver_act = _step_toward(sr, sc, 2, 6, F_RIGHT, facing, True)
    plate_act = _step_toward(sr, sc, jnp.where(sr >= 3, 3, 1), 5, F_LEFT, facing, True)
    pile_act = _step_toward(sr, sc, jnp.where(target == 0, 1, 3), 6, F_RIGHT, facing, True)

    fetch_act = pile_act
    for k in (3, 2, 1, 0):
        fetch_act = jnp.where(holds_target[k], take_step[k], fetch_act)

    # Dropping is only refused when every scratch counter is full; then adding
    # the held item to the pot at least keeps the kitchen moving.
    drop_act = jnp.where(any_free, drop_act, pot_act)

    ready_act = jnp.where(hold_plate & jnp.logical_not(hold_cooked), pot_act,
                          jnp.where(hold_ing, drop_act, plate_act))
    cook_act = jnp.where(hold_plate & jnp.logical_not(hold_cooked), pot_hold,
                         jnp.where(hold_ing, drop_act, plate_act))
    idle_act = jnp.where(hold_ing,
                         jnp.where(hold_type == target, pot_act, drop_act),
                         jnp.where(hold_plate, drop_act,
                                   jnp.where(waiting, pot_hold, fetch_act)))

    action = jnp.where(hold_dish, deliver_act,
                       jnp.where(pot_ready, ready_act,
                                 jnp.where(pot_cooking, cook_act, idle_act)))
    action = action.astype(jnp.int32)

    cooking_type = jnp.where(pot_n > 0, jnp.where(pot[32] > pot[31], 1, 0), mem[_S_COOK])

    mem = mem.at[_S_SIG].set(sig_mem)
    mem = mem.at[_S_STREAK].set(streak)
    mem = mem.at[_S_COMMIT].set(commit)
    mem = mem.at[_S_LAST].set(last_delivered)
    mem = mem.at[_S_DISH].set(hold_dish.astype(jnp.int32))
    mem = mem.at[_S_SEEN].set(seen_ever)
    mem = mem.at[_S_CLOCK].set(clock)
    mem = mem.at[_S_COOK].set(cooking_type.astype(jnp.int32))
    mem = mem.at[_S_NDEL].set(n_delivered)
    mem = mem.at[_S_PEND].set(pending)
    mem = mem.at[_S_COIN].set(coin)
    mem = mem.at[_S_ZERO].set(zero_pat)
    mem = mem.at[_S_MOVED].set(moved)
    mem = mem.at[_S_MOVES].set(jnp.minimum(moves, 99))

    trace = {
        'target': target,
        'signal': sig_mem,
        'zero_pattern': zero_pat,
        'pot_n': pot_n.astype(jnp.int32),
        'deliveries': n_delivered,
        'row': sr,
        'col': sc,
        'waiting': waiting.astype(jnp.int32),
    }
    return action, mem, trace
