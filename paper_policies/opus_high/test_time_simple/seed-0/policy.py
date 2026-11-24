"""Two-room Overcooked-V2 policy for layout ``test_time_simple``.

Geometry (row, column), 5x8 board::

    WW2WWWWW
    W  WB  0
    R AWPA X
    W  WB  1
    WW2WWWWW

The two walkable regions are *completely disconnected*:

* left room  = rows 1..3, cols 1..2   (seat 0 spawns here)
* right room = rows 1..3, cols 5..6   (seat 1 spawns here)

Seat 0 can always see the always-on recipe indicator ``R`` at (2, 0) but has
no pot, no plate pile, no delivery station and only ingredient-2 supplies.
Seat 1 owns the whole kitchen (pot, both plate piles, ingredient 0 and 1
supplies, delivery station) but can never see ``R``.

The only shared channel is the wall column 3: cells (1,3), (2,3), (3,3) are
counters reachable *only* from the left room, yet they are inside the view of
any right-room cell in column 5.  Seat 0 therefore writes and seat 1 reads.

Signalling convention (chosen as the natural spatial Schelling point):

* an item on the **top** counter (1,3) -> cook ingredient 0, whose supply is
  at (1,7), i.e. the same row;
* an item on the **bottom** counter (3,3) -> cook ingredient 1, whose supply
  is at (3,7), again the same row.

The decoder is deliberately tolerant: it uses the vertical centroid of the
occupied signal counters, so ``{top}``, ``{top,middle}``, ``{bottom}`` and
``{bottom,middle}`` all decode, and every other pattern decodes as UNKNOWN.
Seat 1 never delivers a dish unless a stable, fresh signal agrees with what it
is holding, so a partner using an unreadable convention costs time but never
incurs the -20 wrong-delivery penalty.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'row_signal_kitchen',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
A_RIGHT = 0
A_DOWN = 1
A_LEFT = 2
A_UP = 3
A_STAY = 4
A_INTERACT = 5

# Item-block channels on a grid cell (plate, cooked, ing0, ing1, ing2).
_ITEM_CH = (29, 30, 31, 32, 33)

# Post-delivery freshness window, in steps.  It adapts to the partner's own
# measured re-signal latency; these are the prior and the clamps.  Once we have
# watched the partner rewrite the counters, waiting exactly that many steps is
# enough: at that point a reading equal to the previous recipe can only mean the
# resampled recipe really is unchanged.
_BLACKOUT_PRIOR = 7
_BLACKOUT_MIN = 5
_BLACKOUT_MAX = 30
# Consecutive stable readings required before committing to a signal.  A single
# item can never move between two decodable counters without passing through an
# undecodable pattern, so one confirming reading is enough.
_STABLE = 1
# Idle steps without any usable signal before seat 1 speculatively cooks.
_SPECULATE_AFTER = 30
# Pot timer still needed to be able to restock the depot and fetch a plate.
_DEPOT_SLACK = 12
# Consecutive deliveries after which an unchanging recipe display is taken as
# evidence that our reading of the counter code is inverted.  Only a *correct*
# delivery resamples the recipe, so seeing the display change is outright proof
# that the current reading is right; never seeing it change is evidence at odds
# of 2:1 per delivery that it is wrong.
_POLARITY_EVIDENCE = 6


def _cell(obs, i, j, ch):
    """Value of ``ch`` at local cell (i, j); 0 when the cell is out of view."""
    inside = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    ii = jnp.clip(i, 0, 4)
    jj = jnp.clip(j, 0, 4)
    return jnp.where(inside, obs[ii, jj, ch], 0)


def _gcell(obs, row, col, grow, gcol, ch):
    """Value of ``ch`` at global cell (grow, gcol) for a viewer at (row, col)."""
    return _cell(obs, grow - row + 2, gcol - col + 2, ch)


def _occupied(obs, row, col, grow, gcol):
    """True when a grid item sits on the global cell (grow, gcol)."""
    total = jnp.zeros((), dtype=jnp.int32)
    for ch in _ITEM_CH:
        total = total + _gcell(obs, row, col, grow, gcol, ch)
    return total > 0


def _locate(obs, ch, grow, gcol):
    """Self global (row, col) from a unique static landmark at (grow, gcol)."""
    plane = obs[:, :, ch]
    flat = plane.reshape(-1)
    idx = jnp.argmax(flat)
    seen = jnp.max(flat) > 0
    i = idx // 5
    j = idx - 5 * i
    return grow - i + 2, gcol - j + 2, seen


def _move_step(row, col, trow, tcol, face):
    """One movement action towards (trow, tcol), ending with facing ``face``.

    Both rooms are obstacle-free 3x2 rectangles holding a single player, so a
    greedy axis-by-axis walk is always collision free.  When the requested
    final facing is horizontal we clear the row difference first (and vice
    versa) so the last translation already leaves us facing the right way.
    """
    drow = trow - row
    dcol = tcol - col
    vert = jnp.where(drow > 0, A_DOWN, A_UP)
    horiz = jnp.where(dcol > 0, A_RIGHT, A_LEFT)
    face_is_horizontal = (face == A_RIGHT) | (face == A_LEFT)
    act = jnp.where(
        (drow != 0) & ((dcol == 0) | face_is_horizontal),
        vert,
        jnp.where(dcol != 0, horiz, face),
    )
    return jnp.asarray(act, dtype=jnp.int32)


def _facing(obs, face):
    """True when the observer currently faces the direction of action ``face``.

    Movement actions are RIGHT=0, DOWN=1, LEFT=2, UP=3 while the observation
    one-hot facing channels are UP=1, DOWN=2, RIGHT=3, LEFT=4.
    """
    channel = jnp.where(face == A_RIGHT, 3,
                        jnp.where(face == A_DOWN, 2,
                                  jnp.where(face == A_LEFT, 4, 1)))
    return obs[2, 2, channel] > 0


def _goto_act(obs, row, col, trow, tcol, face):
    """Walk to (trow, tcol), turn to ``face``, then INTERACT."""
    arrived = (row == trow) & (col == tcol) & _facing(obs, face)
    return jnp.where(arrived, A_INTERACT, _move_step(row, col, trow, tcol, face))


def _goto_wait(obs, row, col, trow, tcol, face):
    """Walk to (trow, tcol), turn to ``face``, then hold position."""
    arrived = (row == trow) & (col == tcol) & _facing(obs, face)
    return jnp.where(arrived, A_STAY, _move_step(row, col, trow, tcol, face))


# --------------------------------------------------------------------------
# Seat 0: recipe reader / counter signaller (left room, rows 1..3, cols 1..2)
# --------------------------------------------------------------------------


def init_agent_0():
    return jnp.zeros((2,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    memory = jnp.where(reset, init_agent_0(), memory)

    row, col, seen = _locate(obs, 23, 2, 0)

    # Target recipe as displayed by the always-on indicator at (2, 0).
    rec0 = _gcell(obs, row, col, 2, 0, 36)
    rec1 = _gcell(obs, row, col, 2, 0, 37)
    rec2 = _gcell(obs, row, col, 2, 0, 38)
    # Row of the signal counter that encodes the wanted ingredient.
    want_row = jnp.where(rec1 > rec0, 3, jnp.where(rec2 > rec0, 2, 1))

    occ = jnp.stack([
        _occupied(obs, row, col, 1, 3),
        _occupied(obs, row, col, 2, 3),
        _occupied(obs, row, col, 3, 3),
    ])
    rows = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    occ_want = jnp.sum(jnp.where(rows == want_row, occ, False)) > 0
    stale = occ & (rows != want_row)
    stale_row = jnp.where(jnp.any(stale), jnp.argmax(stale) + 1, 0)

    held = (obs[2, 2, 5] + obs[2, 2, 6] + obs[2, 2, 7]
            + obs[2, 2, 8] + obs[2, 2, 9]) > 0

    # Ingredient-2 supplies sit at (0,2) and (4,2); pick the one on the way to
    # the counter we are about to write.
    supply_row = jnp.where(want_row == 3, 3,
                           jnp.where(want_row == 1, 1,
                                     jnp.where(row <= 2, 1, 3)))
    supply_face = jnp.where(supply_row == 1, A_UP, A_DOWN)

    # Private dump counters, invisible from the right room.
    free10 = ~_occupied(obs, row, col, 1, 0)
    free30 = ~_occupied(obs, row, col, 3, 0)
    free01 = ~_occupied(obs, row, col, 0, 1)
    dump_row = jnp.where(free10, 1, jnp.where(free30, 3, jnp.where(free01, 1, 0)))
    dump_face = jnp.where(free10, A_LEFT, jnp.where(free30, A_LEFT, A_UP))
    a_dump = jnp.where(
        dump_row > 0,
        _goto_act(obs, row, col, jnp.maximum(dump_row, 1), 1, dump_face),
        jnp.asarray(A_STAY, dtype=jnp.int32),
    )

    a_take_stale = _goto_act(obs, row, col, jnp.maximum(stale_row, 1), 2, A_RIGHT)
    a_take_supply = _goto_act(obs, row, col, supply_row, 2, supply_face)
    a_place = _goto_act(obs, row, col, want_row, 2, A_RIGHT)
    # Park next to the live counter, facing it, so a recipe flip can be
    # answered by a single INTERACT followed by a two-cell walk.
    a_park = _goto_wait(obs, row, col, want_row, 2, A_RIGHT)

    act_empty = jnp.where(
        stale_row > 0, a_take_stale,
        jnp.where(occ_want, a_park, a_take_supply))
    act_held = jnp.where(occ_want, a_dump, a_place)

    action = jnp.where(held, act_held, act_empty)
    action = jnp.where(seen, action, jnp.asarray(A_STAY, dtype=jnp.int32))
    action = jnp.asarray(action, dtype=jnp.int32)

    trace = {
        'want_row': jnp.asarray(want_row, dtype=jnp.int32),
        'stale_row': jnp.asarray(stale_row, dtype=jnp.int32),
        'signal_up': jnp.asarray(occ_want, dtype=jnp.int32),
        'held': jnp.asarray(held, dtype=jnp.int32),
    }
    return action, memory, trace


# --------------------------------------------------------------------------
# Seat 1: kitchen (right room, rows 1..3, cols 5..6)
# --------------------------------------------------------------------------

# memory layout
_M_SIG = 0        # last decoded signal while column 3 was visible (-1 unknown)
_M_RUN = 1        # consecutive stable readings of that signal
_M_PENDING = 2    # signal value that may still be stale after our delivery
_M_SINCE = 3      # steps since our last delivery
_M_SAWUNK = 4     # signal went UNKNOWN since our last delivery
_M_TGT = 5        # committed ingredient for the batch in progress (-1 none)
_M_ST0 = 6        # a cooked ingredient-0 dish is banked on counter (0,6)
_M_ST1 = 7        # a cooked ingredient-1 dish is banked on counter (4,6)
_M_IDLE = 8       # consecutive idle steps without a usable signal
_M_LAT = 9        # measured partner re-signal latency (-1 = never observed)
_M_DEP0 = 10      # a raw ingredient-0 unit sits in the depot at (0,5)
_M_DEP1 = 11      # a raw ingredient-1 unit sits in the depot at (4,5)
_M_LATLOCK = 12   # the current post-delivery window already yielded a sample
_M_POL = 13       # 1 once we believe the counter code is inverted for us
_M_STREAK = 14    # deliveries in a row that did not resample the recipe
_M_PROVEN = 15    # we have seen the recipe resample, so the reading is correct


def init_agent_1():
    return jnp.asarray([-1, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                       dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    mem = jnp.where(reset, init_agent_1(), memory)

    row, col, seen = _locate(obs, 22, 2, 4)

    # ---------------- signal decoding ----------------
    # Only column-5 cells have the signal counters inside their 5x5 crop.
    visible = col == 5
    top = _occupied(obs, row, col, 1, 3).astype(jnp.int32)
    mid = _occupied(obs, row, col, 2, 3).astype(jnp.int32)
    bot = _occupied(obs, row, col, 3, 3).astype(jnp.int32)
    centroid = top - bot
    coded = jnp.where(centroid > 0, 0, jnp.where(centroid < 0, 1, -1))
    polarity = mem[_M_POL]
    raw_sig = jnp.where((coded >= 0) & (polarity > 0), 1 - coded, coded)

    prev_sig = mem[_M_SIG]
    prev_run = mem[_M_RUN]
    sig = jnp.where(visible, raw_sig, prev_sig)
    run = jnp.where(
        visible,
        jnp.where((raw_sig >= 0) & (raw_sig == prev_sig), prev_run + 1,
                  jnp.where(raw_sig >= 0, 1, 0)),
        prev_run,
    )
    saw_unknown = jnp.maximum(
        mem[_M_SAWUNK], jnp.where(visible & (raw_sig < 0), 1, 0))
    since = jnp.minimum(mem[_M_SINCE] + 1, 1000)
    pending = mem[_M_PENDING]

    # How long does this particular partner take to re-write the counters after
    # a delivery resamples the recipe?  Whenever we see the signal actually flip
    # to a different value we learn that latency directly; until then we assume
    # a deliberately pessimistic window.  Waiting `latency + 1` is exactly what
    # it takes to know that an *unchanged* reading is a genuine confirmation
    # rather than a leftover of the previous recipe.
    # Latch the measurement on the *first* post-delivery step that shows a
    # different value; the signal stays different afterwards, so without the
    # latch the estimate would grow without bound.
    flip_seen = (visible & (pending >= 0) & (raw_sig >= 0)
                 & (raw_sig != pending) & (mem[_M_LATLOCK] == 0))
    measured = jnp.where(flip_seen, jnp.maximum(mem[_M_LAT], since),
                         mem[_M_LAT])
    blackout = jnp.where(measured < 0, _BLACKOUT_PRIOR,
                         jnp.clip(measured, _BLACKOUT_MIN, _BLACKOUT_MAX))
    # A signal is fresh if it differs from the value we already acted on, if we
    # watched the counters go empty since delivering, or if enough time passed.
    fresh = (sig != pending) | (saw_unknown > 0) | (since >= blackout)
    sig_ok = (sig >= 0) & (run >= _STABLE) & fresh

    # ---------------- pot ----------------
    pot_cooked = _gcell(obs, row, col, 2, 4, 30) > 0
    pot0 = _gcell(obs, row, col, 2, 4, 31)
    pot1 = _gcell(obs, row, col, 2, 4, 32)
    pot2 = _gcell(obs, row, col, 2, 4, 33)
    pot_count = pot0 + pot1 + pot2
    pot_timer = _gcell(obs, row, col, 2, 4, 39)
    pot_type = jnp.argmax(jnp.stack([pot0, pot1, pot2])).astype(jnp.int32)
    pot_open = (pot_count < 3) & (pot_timer <= 0) & (~pot_cooked)

    # ---------------- inventory ----------------
    has_plate = obs[2, 2, 5] > 0
    has_cooked = obs[2, 2, 6] > 0
    inv0 = obs[2, 2, 7]
    inv1 = obs[2, 2, 8]
    inv2 = obs[2, 2, 9]
    inv_total = inv0 + inv1 + inv2
    inv_type = jnp.argmax(jnp.stack([inv0, inv1, inv2])).astype(jnp.int32)
    holds_dish = has_plate & has_cooked
    holds_raw = (~has_cooked) & (inv_total > 0)
    holds_plate = has_plate & (~has_cooked)
    hands_free = (~has_plate) & (~has_cooked) & (inv_total == 0)

    # ---------------- depot and dish bank ----------------
    # (0,5) and (4,5) are the only counters adjacent to a cell that neighbours
    # the pot, so a raw unit parked there is one step cheaper to load than a
    # trip to the wall supply.  Cooked dishes we cannot deliver yet are banked
    # one column further out on (0,6) and (4,6).
    sees_top = row <= 2
    sees_bot = row >= 2
    dep0 = ((_gcell(obs, row, col, 0, 5, 31) > 0)
            & (_gcell(obs, row, col, 0, 5, 30) == 0)).astype(jnp.int32)
    dep1 = ((_gcell(obs, row, col, 4, 5, 32) > 0)
            & (_gcell(obs, row, col, 4, 5, 30) == 0)).astype(jnp.int32)
    depot0 = jnp.where(sees_top, dep0, mem[_M_DEP0])
    depot1 = jnp.where(sees_bot, dep1, mem[_M_DEP1])
    bank_0 = ((_gcell(obs, row, col, 0, 6, 30) > 0)
              & (_gcell(obs, row, col, 0, 6, 31) > 0)).astype(jnp.int32)
    bank_1 = ((_gcell(obs, row, col, 4, 6, 30) > 0)
              & (_gcell(obs, row, col, 4, 6, 32) > 0)).astype(jnp.int32)
    stash0 = jnp.where(sees_top, bank_0, mem[_M_ST0])
    stash1 = jnp.where(sees_bot, bank_1, mem[_M_ST1])
    free06 = ~_occupied(obs, row, col, 0, 6)
    free46 = ~_occupied(obs, row, col, 4, 6)

    # ---------------- which ingredient are we cooking ----------------
    idle_prev = mem[_M_IDLE]
    speculative = jnp.where(stash0 == 0, 0, jnp.where(stash1 == 0, 1, -1))
    target = jnp.where(
        pot_count > 0, pot_type,
        jnp.where(sig_ok, sig,
                  jnp.where(mem[_M_TGT] >= 0, mem[_M_TGT],
                            jnp.where(idle_prev >= _SPECULATE_AFTER,
                                      speculative, -1))))

    # ---------------- candidate moves ----------------
    a_pot = _goto_act(obs, row, col, 2, 5, A_LEFT)
    a_hold = _goto_wait(obs, row, col, 2, 5, A_LEFT)
    a_deliver = _goto_act(obs, row, col, 2, 6, A_RIGHT)
    a_plate = _goto_act(obs, row, col, 1, 5, A_LEFT)

    # Load the batch from the depot when it is stocked, otherwise from the wall.
    tgt_row = jnp.where(target == 1, 3, 1)
    from_depot = jnp.where(target == 1, depot1, depot0) > 0
    a_fetch = _goto_act(
        obs, row, col, tgt_row,
        jnp.where(from_depot, 5, 6),
        jnp.where(from_depot, jnp.where(target == 1, A_DOWN, A_UP), A_RIGHT))

    # Restocking trip: collect the missing depot ingredient from its supply.
    stock_type = jnp.where(depot0 == 0, 0, 1)
    a_stock = _goto_act(obs, row, col,
                        jnp.where(stock_type == 1, 3, 1), 6, A_RIGHT)

    inv_row = jnp.where(inv_type == 0, 1, 3)
    inv_face = jnp.where(inv_type == 0, A_UP, A_DOWN)
    a_bank = _goto_act(obs, row, col, inv_row, 6, inv_face)
    sig_row = jnp.where(sig == 0, 1, 3)
    sig_face = jnp.where(sig == 0, A_UP, A_DOWN)
    a_unbank = _goto_act(obs, row, col, sig_row, 6, sig_face)
    # A raw unit we cannot use goes to its depot slot, falling back to the bank
    # counters so that we can never be stuck holding it.
    depot_free = jnp.where(inv_type == 0, depot0, depot1) == 0
    drop_col = jnp.where(depot_free, 5, 6)
    drop_row = jnp.where(depot_free, inv_row,
                         jnp.where(free06, 1, jnp.where(free46, 3, inv_row)))
    drop_face = jnp.where(drop_row == 1, A_UP, A_DOWN)
    a_drop_raw = _goto_act(obs, row, col, drop_row, drop_col, drop_face)

    # ---------------- behaviour ----------------
    dish_matches = sig_ok & (sig == inv_type)
    dish_wrong = sig_ok & (sig != inv_type)
    bank_free = jnp.where(inv_type == 0, free06, free46)
    act_dish = jnp.where(dish_matches, a_deliver,
                         jnp.where(dish_wrong & bank_free, a_bank, a_hold))

    act_plate = jnp.where(pot_cooked, a_pot, a_hold)

    # Only commit a raw unit to the pot if it still matches what we believe the
    # batch should be; anything else is parked in the depot for later use.
    right_ing = jnp.where(pot_count > 0, inv_type == pot_type,
                          jnp.where(sig_ok, inv_type == sig, True))
    can_add = pot_open & right_ing
    act_raw = jnp.where(can_add, a_pot, a_drop_raw)

    can_take = sig_ok & (jnp.where(sig == 0, stash0, stash1) > 0)
    restock = ((depot0 == 0) | (depot1 == 0)) & (pot_timer >= _DEPOT_SLACK)
    act_free = jnp.where(
        pot_cooked, a_plate,
        jnp.where(can_take, a_unbank,
                  jnp.where(pot_count >= 3,
                            jnp.where(restock, a_stock, a_plate),
                            jnp.where(pot_open & (target >= 0), a_fetch,
                                      a_hold))))

    action = jnp.where(
        holds_dish, act_dish,
        jnp.where(holds_plate, act_plate,
                  jnp.where(holds_raw, act_raw, act_free)))
    action = jnp.where(seen, action, jnp.asarray(A_STAY, dtype=jnp.int32))
    action = jnp.asarray(action, dtype=jnp.int32)

    # ---------------- memory update ----------------
    interacting = action == A_INTERACT
    delivered = (interacting & holds_dish & (row == 2) & (col == 6)
                 & _facing(obs, A_RIGHT))
    banked = (interacting & holds_dish & (col == 6) & (row == inv_row)
              & _facing(obs, inv_face))
    unbanked = (interacting & hands_free & can_take & (col == 6)
                & (row == sig_row) & _facing(obs, sig_face))

    # Polarity evidence.  A couple of steps past the point where the partner
    # would have finished rewriting the counters, an unmoved display means our
    # previous delivery did not resample the recipe, i.e. it scored -20.
    mature = (since == blackout + 2) & (pending >= 0) & (~delivered)
    no_resample = mature & (mem[_M_LATLOCK] == 0)
    proven = jnp.maximum(mem[_M_PROVEN], flip_seen.astype(jnp.int32))
    # If we have just learned the partner is slower than we assumed, the earlier
    # deliveries we are counting as evidence were plausibly spoiled by acting on
    # a stale reading rather than by an inverted code, so the tally restarts.
    lat_grew = measured > mem[_M_LAT]
    streak = jnp.where(flip_seen | lat_grew, 0,
                       jnp.where(no_resample, mem[_M_STREAK] + 1,
                                 mem[_M_STREAK]))
    flip_code = no_resample & (proven == 0) & (streak >= _POLARITY_EVIDENCE)
    new_pol = jnp.where(flip_code, 1 - polarity, polarity)
    new_streak = jnp.where(flip_code, 0, streak)

    # After inverting the reading, the very same counters now mean the opposite,
    # so the cached reading and the value we last acted on flip with them.
    new_pending = jnp.where(delivered, inv_type,
                            jnp.where(flip_code, 1 - pending, pending))
    new_sig = jnp.where(flip_code & (sig >= 0), 1 - sig, sig)
    new_since = jnp.where(delivered, 0, since)
    new_saw = jnp.where(delivered, 0, saw_unknown)
    new_run = jnp.where(delivered, 0, run)
    new_tgt = jnp.where(delivered | banked, -1, target)
    new_st0 = jnp.where(banked & (inv_type == 0), 1,
                        jnp.where(unbanked & (sig == 0), 0, stash0))
    new_st1 = jnp.where(banked & (inv_type == 1), 1,
                        jnp.where(unbanked & (sig == 1), 0, stash1))
    idling = hands_free & (pot_count == 0) & pot_open & (~sig_ok)
    new_idle = jnp.where(idling, jnp.minimum(idle_prev + 1, 1000), 0)

    new_latlock = jnp.where(
        delivered, 0,
        jnp.maximum(mem[_M_LATLOCK], flip_seen.astype(jnp.int32)))

    next_mem = jnp.stack([
        new_sig, new_run, new_pending, new_since, new_saw,
        new_tgt, new_st0, new_st1, new_idle, measured, depot0, depot1,
        new_latlock, new_pol, new_streak, proven,
    ]).astype(jnp.int32)

    trace = {
        'sig': jnp.asarray(sig, dtype=jnp.int32),
        'sig_ok': jnp.asarray(sig_ok, dtype=jnp.int32),
        'target': jnp.asarray(target, dtype=jnp.int32),
        'pot_count': jnp.asarray(pot_count, dtype=jnp.int32),
        'pot_timer': jnp.asarray(pot_timer, dtype=jnp.int32),
        'delivered': jnp.asarray(delivered, dtype=jnp.int32),
        'bank': jnp.asarray(new_st0 + 2 * new_st1, dtype=jnp.int32),
        'depot': jnp.asarray(depot0 + 2 * depot1, dtype=jnp.int32),
        'blackout': jnp.asarray(blackout, dtype=jnp.int32),
        'polarity': jnp.asarray(new_pol, dtype=jnp.int32),
        'streak': jnp.asarray(new_streak, dtype=jnp.int32),
        'proven': jnp.asarray(proven, dtype=jnp.int32),
    }
    return action, next_mem, trace
