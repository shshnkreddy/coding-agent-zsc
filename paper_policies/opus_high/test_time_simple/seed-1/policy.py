"""Overcooked-V2 `test_time_simple` cooperative policy.

Layout facts this policy relies on (all from context/environment/):

    WW2WWWWW
    W  WB  0
    R AWPA X
    W  WB  1
    WW2WWWWW

The board is split by the wall column 3 into two private rooms that share no
floor.  Seat 0 lives in rows 1-3 / columns 1-2 and can only reach the
ingredient-2 piles and the always-on recipe indicator `R`.  Seat 1 lives in
rows 1-3 / columns 5-6 and owns everything that scores: the ingredient 0/1
piles, both plate piles, the pot and the delivery station.  The agents never
see each other (their 5x5 crops never overlap on a player cell).

Consequences:

* Only seat 0 can read the recipe; only seat 1 can cook it.  The single
  bidirectional channel is the wall column 3: seat 0 can put items on
  (1,3), (2,3) and (3,3), and seat 1 can see those three cells whenever it
  stands on column 5.  Seat 0 can in turn watch the pot (2,4) from column 2.
* Guessing is worthless: a wrong delivery is -20 and a right one +20.

Convention used here (kept as close to the layout's own geometry as possible
so that an independently written partner has a fair chance of sharing it):
a single ingredient-2 item parked on the *top* counter (1,3) means "cook
ingredient 0" (whose pile sits on the top row of the other room, (1,7)), and
an item on the *bottom* counter (3,3) means "cook ingredient 1" (pile at
(3,7)).  Seat 1 decodes defensively: exactly the top cell occupied -> 0, any
occupancy that is not "top cell alone" -> 1, nothing visible -> unknown, and
it refuses to deliver while the target is unknown.

Both seats carry a one-bit polarity that can flip if the evidence says the
partner reads the counters the other way round:

* seat 0 flips after it watches the pot get filled with three units of the
  wrong ingredient while its own display had been stable for a long time
  (a stale-signal race cannot explain that);
* seat 1 flips after three deliveries in a row that were followed by no
  change of the partner's display (a correct delivery re-samples the recipe
  and therefore changes the display half of the time; an incorrect one never
  does), and locks its polarity for good once it has seen such a change.

A display that stays empty far longer than any re-signalling pause is itself
read as a code, so a partner that encodes one of the two recipes by leaving
the counters bare is still understood.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'counter_signal_relay',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5
# Facing ids as they appear in observation channels 1..4.
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# Movement action that produces each facing (index by facing id).
_FACE_MOVE = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

_FRESH_STEPS = 6  # initial post-delivery window for the partner to re-signal
_FLIP_AFTER = 3   # deliveries with no display echo before seat 1 flips polarity
_SILENCE_STEPS = 20  # readable blank steps before a blank display is itself a code


def _sel(conds, vals, default):
    """Priority select: first true condition wins."""
    out = jnp.asarray(default, dtype=jnp.int32)
    for cond, val in zip(conds[::-1], vals[::-1]):
        out = jnp.where(cond, jnp.asarray(val, dtype=jnp.int32), out)
    return out


def _cell(obs, i, j, lo, hi):
    """Channels [lo, hi) of local cell (i, j); zeros when out of the crop."""
    idx = jnp.arange(5)
    mask = ((idx[:, None] == i) & (idx[None, :] == j)).astype(jnp.int32)
    return jnp.sum(mask[:, :, None] * obs[:, :, lo:hi], axis=(0, 1))


def _in_view(i, j):
    return (i >= 0) & (i < 5) & (j >= 0) & (j < 5)


def _item(obs, i, j):
    """Grid-item block (plate, cooked, counts) of a local cell."""
    return _cell(obs, i, j, 29, 34)


def _occupied(obs, i, j):
    return jnp.sum(_item(obs, i, j)) > 0


def _free_counter(obs, i, j):
    """True only when the cell is visible *and* empty."""
    return _in_view(i, j) & (jnp.sum(_item(obs, i, j)) == 0)


def _locate(obs, channel, gr, gc):
    """Global self position from a unique static landmark at (gr, gc)."""
    flat = jnp.argmax(obs[:, :, channel].reshape(-1))
    li = (flat // 5).astype(jnp.int32)
    lj = (flat % 5).astype(jnp.int32)
    return gr - li + 2, gc - lj + 2, li, lj


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _nav(r, c, tr, tc, face, want_face, at_act):
    """Walk to (tr, tc) (rows first), then turn to `want_face` and act.

    Both rooms are obstacle-free rectangles, so fixing the row and then the
    column always works, and a turn is issued as a movement into the facing
    object, which is blocked and therefore only changes orientation.
    """
    row_act = jnp.where(tr > r, A_DOWN, A_UP)
    col_act = jnp.where(tc > c, A_RIGHT, A_LEFT)
    at_row = r == tr
    at_cell = at_row & (c == tc)
    final = jnp.where(face == want_face, at_act, jnp.array(_FACE_MOVE, dtype=jnp.int32)[want_face])
    return jnp.where(at_cell, final, jnp.where(at_row, col_act, row_act)).astype(jnp.int32)


# --------------------------------------------------------------------------
# Seat 0: recipe reader / counter signaller.
# --------------------------------------------------------------------------
# memory = [polarity, flips_used, flip_cooldown, display_stable_steps, step]

def init_agent_0():
    return jnp.zeros((5,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, jnp.zeros((5,), jnp.int32), memory)
    pol, flips, cool, stable, step = mem[0], mem[1], mem[2], mem[3], mem[4]

    # The recipe indicator (2,0) is in view from every cell of the left room.
    r, c, li, lj = _locate(obs, 23, 2, 0)
    recipe = _cell(obs, li, lj, 36, 39)
    target = jnp.where(recipe[1] > recipe[0], 1, 0).astype(jnp.int32)

    sj = 5 - c  # local column of map column 3
    occ_top = _occupied(obs, 3 - r, sj)  # (1,3)
    occ_mid = _occupied(obs, 4 - r, sj)  # (2,3)
    occ_bot = _occupied(obs, 5 - r, sj)  # (3,3)

    # Has the display already been correct (for the current polarity) for a
    # long time?  Only then can a wrong pot be blamed on a decoding mismatch.
    e_old = jnp.bitwise_xor(target, pol)
    d_old = jnp.where(e_old == 0, occ_top, occ_bot)
    o_old = jnp.where(e_old == 0, occ_bot, occ_top)
    display_ok = d_old & (~o_old) & (~occ_mid)

    # Pot feedback: the pot (2,4) is in the crop only from column 2.
    pot = _item(obs, 4 - r, 6 - c)
    pot_counts = pot[2:5]
    # Only a pot filled with three units of a single *wrong* ingredient is
    # evidence of a decoding mismatch; a mixed pot means partner confusion
    # that flipping the display would not repair.
    wrong_pot = ((c == 2) & (jnp.sum(pot_counts) == 3)
                 & (jnp.sum(pot_counts * (jnp.arange(3) == target)) == 0))
    do_flip = wrong_pot & (stable >= 25) & (cool == 0) & (flips < 2)

    pol = jnp.bitwise_xor(pol, do_flip.astype(jnp.int32))
    flips = flips + do_flip.astype(jnp.int32)
    cool = jnp.where(do_flip, 60, jnp.maximum(cool - 1, 0))
    stable = jnp.where(do_flip, 0,
                       jnp.where(display_ok, jnp.minimum(stable + 1, 4000), 0))

    emit = jnp.bitwise_xor(target, pol)
    want_row = jnp.where(emit == 0, 1, 3)   # counter that must hold the item
    stale_row = jnp.where(emit == 0, 3, 1)  # counter that must be cleared
    occ_want = jnp.where(emit == 0, occ_top, occ_bot)
    occ_stale = jnp.where(emit == 0, occ_bot, occ_top)

    held = jnp.sum(obs[2, 2, 5:10]) > 0
    face = _facing(obs)

    # Clearing the stale cell happens before placing, so a transition shows
    # "nothing" (decoded as unknown) instead of a stale target.
    conds = [
        held & (~occ_want),   # place the carried unit on the wanted counter
        held,                 # wanted counter already full -> park at (4,1)
        (~held) & occ_stale,  # recycle the unit from the stale counter
        (~held) & occ_mid,    # remove a stray unit from (2,3)
        (~held) & (~occ_want),  # fetch a fresh ingredient-2 unit
    ]
    tr = _sel(conds, [want_row, 3, stale_row, 2, jnp.where(emit == 0, 1, 3)], want_row)
    tc = _sel(conds, [2, 1, 2, 2, 2], 2)
    wf = _sel(conds, [F_RIGHT, F_DOWN, F_RIGHT, F_RIGHT,
                      jnp.where(emit == 0, F_UP, F_DOWN)], F_RIGHT)
    aa = _sel(conds, [A_INT, A_INT, A_INT, A_INT, A_INT], A_STAY)
    action = _nav(r, c, tr, tc, face, wf, aa)

    step = jnp.minimum(step + 1, 4000)
    new_mem = jnp.stack([pol, flips, cool, stable, step]).astype(jnp.int32)
    trace = {
        'target': target,
        'emit': emit,
        'polarity': pol,
        'stable': stable,
        'pos': jnp.stack([r, c]).astype(jnp.int32),
    }
    return action, new_mem, trace


# --------------------------------------------------------------------------
# Seat 1: signal reader / cook / deliverer.
# --------------------------------------------------------------------------
# memory = [polarity, last_valid_signal, empty_streak, signal_at_delivery,
#           changed_since_delivery, no_change_count, locked, has_delivered,
#           steps_since_delivery, step, adaptive_gate, commit_offset]

def init_agent_1():
    return jnp.array([0, -1, 0, -1, 0, 0, 0, 0, 1000, 0, _FRESH_STEPS, -1],
                     dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, init_agent_1(), memory)
    (pol, last_sig, empty_streak, sig_del, changed, no_change,
     locked, has_del, since_del, step, gate, commit_off) = [mem[i] for i in range(12)]

    # The pot (2,4) is in the crop from every cell of the right room.
    r, c, li, lj = _locate(obs, 22, 2, 4)
    face = _facing(obs)

    # ---- decode the partner's counter display (only readable at column 5)
    readable = c == 5
    sj = 5 - c
    top = _occupied(obs, 3 - r, sj) & readable
    mid = _occupied(obs, 4 - r, sj) & readable
    bot = _occupied(obs, 5 - r, sj) & readable
    total = top.astype(jnp.int32) + mid.astype(jnp.int32) + bot.astype(jnp.int32)
    raw = jnp.where(total == 0, -1, jnp.where((total == 1) & top, 0, 1)).astype(jnp.int32)
    raw_valid = readable & (raw >= 0)
    last_sig = jnp.where(raw_valid, raw, last_sig)
    empty_streak = jnp.where(readable,
                             jnp.where(raw < 0, jnp.minimum(empty_streak + 1, 2000), 0),
                             empty_streak)
    # A partner that signals only one of the two recipes leaves the counters
    # empty for the other one; our own emitter is never empty for long.
    silent = (empty_streak >= _SILENCE_STEPS) & (last_sig >= 0)
    effective = jnp.where(raw_valid, raw, jnp.where(silent, 1 - last_sig, last_sig))
    goal = jnp.where(effective >= 0, jnp.bitwise_xor(effective, pol), -1).astype(jnp.int32)
    # The echo test runs on the effective code, not the raw occupancy, so a
    # partner that encodes one recipe by an empty display still produces a
    # detectable change; our own blank windows are far too short to register.
    change_now = ((effective >= 0) & (has_del > 0) & (sig_del >= 0)
                  & (effective != sig_del) & (changed == 0))
    changed = jnp.maximum(changed, change_now.astype(jnp.int32))
    # If the partner's new display only shows up *after* this batch was already
    # committed, the gate was too short for that partner: lengthen it.
    # A change seen much later than that is re-signalling after the partner
    # changed its own mind, not latency, so the window stays bounded.
    late = (change_now & (commit_off >= 0) & (since_del > commit_off)
            & (since_del <= 16))
    gate = jnp.where(late, jnp.clip(since_del + 2, gate, 18), gate)
    # A blank display means the partner is part-way through re-signalling, so
    # the first unit of a batch is never committed on a stale reading.
    readable_now = raw_valid | silent
    fresh = ((since_del >= gate) & readable_now) | (changed > 0)

    # ---- pot, inventory, spare counters
    pot = _item(obs, li, lj)
    pot_counts = pot[2:5]
    pot_n = jnp.sum(pot_counts)
    pot_cooked = pot[1] > 0
    pot_timer = _cell(obs, li, lj, 39, 40)[0]
    pot_busy = (pot_timer > 0) | pot_cooked | (pot_n >= 3)
    committed = (~pot_busy) & (pot_n >= 1)
    pot_ing = jnp.argmax(pot_counts).astype(jnp.int32)

    inv = obs[2, 2, 5:10]
    inv_counts = inv[2:5]
    held_n = jnp.sum(inv_counts)
    held_ing = jnp.argmax(inv_counts).astype(jnp.int32)
    has_dish = (inv[0] > 0) & (inv[1] > 0)
    has_plate = (inv[0] > 0) & (inv[1] == 0)
    has_raw = (inv[0] == 0) & (held_n > 0)
    empty_hands = (inv[0] == 0) & (held_n == 0)

    need = jnp.where(committed, pot_ing, goal)
    safe_need = jnp.maximum(need, 0)
    safe_goal = jnp.maximum(goal, 0)

    # Spare counters of the right room, used as a parking area:
    # (0,5) and (0,6) are reached from row 1 facing UP, (4,5) and (4,6) from
    # row 3 facing DOWN.  They are visited in the order that keeps the walk
    # towards the pile we need next as short as possible.
    park_row = jnp.array([1, 1, 3, 3], dtype=jnp.int32)
    park_col = jnp.array([5, 6, 5, 6], dtype=jnp.int32)
    park_face = jnp.array([F_UP, F_UP, F_DOWN, F_DOWN], dtype=jnp.int32)
    park_li = jnp.array([2, 2, 6, 6], dtype=jnp.int32) - r
    park_lj = jnp.array([7, 8, 7, 8], dtype=jnp.int32) - c
    order = jnp.where(need == 1, jnp.array([2, 3, 0, 1]), jnp.array([0, 1, 2, 3]))

    items = jnp.stack([_item(obs, park_li[k], park_lj[k]) for k in range(4)])
    seen = jnp.stack([_in_view(park_li[k], park_lj[k]) for k in range(4)])
    counts = items[:, 2:5]
    one_goal = (jnp.arange(3) == safe_goal).astype(jnp.int32)
    one_need = (jnp.arange(3) == safe_need).astype(jnp.int32)

    free_c = seen & (jnp.sum(items, axis=1) == 0)
    dish_c = ((items[:, 0] > 0) & (items[:, 1] > 0)
              & (jnp.sum(counts * one_goal, axis=1) == 3) & (goal >= 0))
    raw_c = ((items[:, 0] == 0) & (items[:, 1] == 0)
             & (jnp.sum(counts, axis=1) == 1)
             & (jnp.sum(counts * one_need, axis=1) == 1) & (need >= 0))

    def first_of(mask):
        idx = order[jnp.argmax(mask[order])]
        return idx, jnp.any(mask)

    park_k, park_ok = first_of(free_c)
    dish_k, take_dish = first_of(dish_c)
    raw_k, take_raw = first_of(raw_c)

    dish_ok = (goal >= 0) & (jnp.sum(inv_counts * (jnp.arange(3) == safe_goal)) == 3)
    insert_ok = ((~pot_busy) & has_raw & (need >= 0) & (held_ing == need)
                 & (committed | fresh))

    supply_row = jnp.where(safe_need == 0, 1, 3)

    conds = [
        has_dish & dish_ok,                                  # deliver
        has_dish & (goal >= 0) & park_ok,                    # park a wrong dish
        has_dish,                                            # hold and wait
        has_plate & pot_cooked,                              # plate the dish
        has_plate & (pot_timer > 0),                         # wait by the pot
        has_plate & park_ok,                                 # park a stray plate
        has_plate,
        insert_ok,                                           # load the pot
        has_raw & (~pot_busy) & (need >= 0) & (held_ing == need),  # await fresh signal
        has_raw & park_ok,                                   # park a wrong unit
        has_raw,
        empty_hands & (pot_cooked | (pot_timer > 0)),        # fetch a plate
        empty_hands & (pot_n == 0) & take_dish,              # re-use a parked dish
        empty_hands & (need >= 0) & take_raw,                # re-use a parked unit
        empty_hands & (need >= 0),                           # fetch from the pile
    ]
    pk_r, pk_c, pk_f = park_row[park_k], park_col[park_k], park_face[park_k]
    tr = _sel(conds, [2, pk_r, 2, 2, 2, pk_r, 2, 2, 2, pk_r, 2, 1,
                      park_row[dish_k], park_row[raw_k], supply_row], 2)
    tc = _sel(conds, [6, pk_c, 5, 5, 5, pk_c, 5, 5, 5, pk_c, 5, 5,
                      park_col[dish_k], park_col[raw_k], 6], 5)
    wf = _sel(conds, [F_RIGHT, pk_f, F_LEFT, F_LEFT, F_LEFT, pk_f, F_LEFT,
                      F_LEFT, F_LEFT, pk_f, F_LEFT, F_LEFT,
                      park_face[dish_k], park_face[raw_k], F_RIGHT], F_LEFT)
    aa = _sel(conds, [A_INT, A_INT, A_STAY, A_INT, A_STAY, A_INT, A_STAY,
                      A_INT, A_STAY, A_INT, A_STAY, A_INT, A_INT, A_INT,
                      A_INT], A_STAY)
    action = _nav(r, c, tr, tc, face, wf, aa)

    # ---- delivery bookkeeping and polarity adaptation
    delivered = ((action == A_INT) & (r == 2) & (c == 6)
                 & (face == F_RIGHT) & has_dish)
    judge = delivered & (has_del > 0)
    saw_change = changed > 0
    no_change = jnp.where(judge, jnp.where(saw_change, 0, no_change + 1), no_change)
    locked = jnp.where(judge & saw_change, 1, locked)
    do_flip = judge & (~saw_change) & (no_change >= _FLIP_AFTER) & (locked == 0)
    pol = jnp.bitwise_xor(pol, do_flip.astype(jnp.int32))
    no_change = jnp.where(do_flip, 0, no_change)
    sig_del = jnp.where(delivered, effective, sig_del)
    changed = jnp.where(delivered, 0, changed)
    has_del = jnp.maximum(has_del, delivered.astype(jnp.int32))
    # Offset at which this batch became irreversible (first unit in the pot).
    first_insert = ((action == A_INT) & (r == 2) & (c == 5) & (face == F_LEFT)
                    & has_raw & (~pot_busy) & (pot_n == 0))
    commit_off = jnp.where(delivered, -1,
                           jnp.where(first_insert & (commit_off < 0), since_del, commit_off))
    since_del = jnp.where(delivered, 0, jnp.minimum(since_del + 1, 2000))
    step = jnp.minimum(step + 1, 4000)

    new_mem = jnp.stack([pol, last_sig, empty_streak, sig_del, changed,
                         no_change, locked, has_del, since_del, step,
                         gate, commit_off]).astype(jnp.int32)
    trace = {
        'goal': goal,
        'raw': raw,
        'polarity': pol,
        'gate': gate,
        'pot_n': pot_n.astype(jnp.int32),
        'delivered': delivered.astype(jnp.int32),
        'pos': jnp.stack([r, c]).astype(jnp.int32),
    }
    return action, new_mem, trace
