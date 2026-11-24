"""Overcooked V2 (test_time_simple) joint policy.

Layout facts that drive the design (see context/environment/README.md):

    WW2WWWWW
    W  WB  0
    R AWPA X
    W  WB  1
    WW2WWWWW

The two rooms are disjoint.  agent_0 only reaches the ingredient-2 piles, the
always-on recipe indicator ``R`` at (2, 0) and a few counters; agent_1 owns the
ingredient 0/1 piles, both plate piles, the pot and the delivery station but can
never see ``R`` and never sees its partner (the column gap is 3 > view radius).
Recipes are [0,0,0] or [1,1,1], so only agent_1 can cook and only agent_0 knows
what to cook.  The one usable channel is the column-3 counter strip: agent_0 can
drop an ingredient on (1, 3) / (2, 3) / (3, 3) and agent_1 sees that column from
its column-5 cells.

Convention (chosen to be the natural, non-private reading of the board):
    item on (1, 3)  ->  cook ingredient 0   (its pile sits at (1, 7), same row)
    item on (3, 3)  ->  cook ingredient 1   (its pile sits at (3, 7), same row)

agent_0 idles beside the *other* counter holding a spare unit, so the new marker
appears one step after the recipe changes and the stale one is collected over
the next four; a partner that reads the strip immediately after delivering still
sees the fresh marker.

agent_1 does not blindly trust the convention.  It decodes a bit from the strip
(which counter, or whether the single counter this partner uses is loaded, or
the middle counter as a last resort) and applies a polarity learned from
delivery feedback: an incorrect delivery never resamples the recipe, so a signal
that moved proves the last delivery was correct, while a signal that stayed put
is one bit of evidence that the convention is being read backwards.  The prior
in that log-odds test is the strength of the convention, and with no readable
signal at all it degrades into strict alternation, which still earns 2/3 correct
deliveries because a wrong delivery leaves the recipe in place.  Commit timing
adapts to the partner's measured rewrite latency, and the ingredient in hand is
re-checked against the strip before the irreversible first deposit.
"""

from __future__ import annotations

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'counter_signal_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5
# Facing ids used by observation channels 1..4.
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# Movement action that produces each facing id.
_FACE_TO_MOVE = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

_I32 = jnp.int32

# agent_1 timing constants.
_POST_DELIVERY_WAIT = 10   # steps to let the partner rewrite the strip
_FIRST_WAIT = 5            # steps to let the partner post the first signal
_EXTEND_WAIT = 2           # extra steps while the strip reads "in flight"
_EXTEND_BUDGET = 4         # per cycle
_MIN_WAIT = 3              # never commit sooner than this after a delivery
_MAX_WAIT = 14             # ceiling once a slow partner has been measured
_GRACE_BUDGET = 6          # whole-episode budget for "no signal posted yet"
_LOCK = 8                  # log-odds asserted by a confirmed correct delivery
_PRIOR_BOTH = 3            # prior for the row convention on two counters
_PRIOR_ONE = 2             # prior when the partner only ever uses one counter


def _landmark(obs, channel):
    """Local (row, col) of the unique static object in ``channel``."""
    flat = jnp.argmax(obs[:, :, channel].reshape(-1))
    return (flat // 5).astype(_I32), (flat % 5).astype(_I32)


def _cell(obs, row, col, grow, gcol):
    """Feature vector of global cell ``(grow, gcol)`` plus an in-view flag."""
    li = grow - row + 2
    lj = gcol - col + 2
    inb = (li >= 0) & (li <= 4) & (lj >= 0) & (lj <= 4)
    vec = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4)]
    return jnp.where(inb, vec, jnp.zeros_like(vec)), inb


def _has_item(vec):
    return jnp.sum(vec[29:34]) > 0


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(_I32)


def _first_true(conds, rows, cols, faces, acts):
    """Select the goal tuple of the first satisfied condition (last is default)."""
    idx = jnp.argmax(jnp.stack(conds))
    pick = lambda values: jnp.stack([jnp.asarray(v) for v in values])[idx]
    return (pick(rows).astype(_I32), pick(cols).astype(_I32),
            pick(faces).astype(_I32), pick(acts))


def _step_toward(row, col, facing, trow, tcol, face, interact):
    """One action of a go-to-cell / turn / act plan inside an open 3x2 room.

    The axis order is chosen so the final translation already sets the wanted
    facing whenever possible; turning in place is done with a movement action
    into the object being faced, which is always blocked here.
    """
    vertical_first = (face == F_RIGHT) | (face == F_LEFT)
    dv = jnp.where(trow > row, A_DOWN, A_UP)
    dh = jnp.where(tcol > col, A_RIGHT, A_LEFT)
    need_v = trow != row
    need_h = tcol != col
    arrived = jnp.where(facing == face,
                        jnp.where(interact, A_INT, A_STAY),
                        jnp.asarray(_FACE_TO_MOVE, dtype=_I32)[face])
    action = jnp.where(
        vertical_first,
        jnp.where(need_v, dv, jnp.where(need_h, dh, arrived)),
        jnp.where(need_h, dh, jnp.where(need_v, dv, arrived)),
    )
    return action.astype(_I32)


# --------------------------------------------------------------------------- #
# agent_0: recipe reader / signaller (left room)
# --------------------------------------------------------------------------- #

def init_agent_0():
    return jnp.asarray(0, dtype=_I32)


def agent_0_act(obs, reset, memory, rng):
    ri, rj = _landmark(obs, 23)          # recipe indicator at global (2, 0)
    row = 4 - ri
    col = 2 - rj
    facing = _facing(obs)
    holding = jnp.sum(obs[2, 2, 5:10]) > 0

    want = jnp.argmax(obs[ri, rj][36:39]).astype(_I32)   # 0 or 1
    want1 = want == 1

    top, _ = _cell(obs, row, col, 1, 3)
    bot, _ = _cell(obs, row, col, 3, 3)
    top_occ = _has_item(top)
    bot_occ = _has_item(bot)

    tgt_row = jnp.where(want1, 3, 1).astype(_I32)
    oth_row = jnp.where(want1, 1, 3).astype(_I32)
    tgt_occ = jnp.where(want1, bot_occ, top_occ)
    oth_occ = jnp.where(want1, top_occ, bot_occ)

    # Idling next to the *other* counter with a spare unit in hand makes the new
    # marker appear one step after the recipe changes instead of five: with two
    # recipes the other counter is always the next one to be written.
    tgt_face = jnp.where(want1, F_DOWN, F_UP).astype(_I32)   # pile beside stand
    oth_face = jnp.where(want1, F_UP, F_DOWN).astype(_I32)

    place = holding & (~tgt_occ)          # write the new marker at once
    clear = (~holding) & oth_occ          # then take the stale marker away
    fetch = (~holding) & (~tgt_occ)       # nothing written yet: get a unit
    spare = (~holding) & tgt_occ          # keep a spare ready at the other cell

    trow, tcol, face, interact = _first_true(
        [place, clear, fetch, spare, jnp.asarray(True)],
        [tgt_row, oth_row, tgt_row, oth_row, oth_row],
        [2, 2, 2, 2, 2],
        [F_RIGHT, F_RIGHT, tgt_face, oth_face, F_RIGHT],
        [True, True, True, True, False],
    )
    action = _step_toward(row, col, facing, trow, tcol, face, interact)
    trace = {
        'want': want,
        'top_occ': top_occ.astype(_I32),
        'bot_occ': bot_occ.astype(_I32),
        'holding': holding.astype(_I32),
    }
    return action, jnp.where(reset, jnp.asarray(0, dtype=_I32), memory), trace


# --------------------------------------------------------------------------- #
# agent_1: cook / plate / deliver (right room)
# --------------------------------------------------------------------------- #

def init_agent_1():
    return {
        'last_bit': jnp.asarray(1, dtype=_I32),      # last readable strip bit
        'ever_top': jnp.asarray(0, dtype=_I32),      # counter (1, 3) ever loaded
        'ever_bot': jnp.asarray(0, dtype=_I32),      # counter (3, 3) ever loaded
        'flight': jnp.asarray(0, dtype=_I32),        # strip seen mid-rewrite
        'committed': jnp.asarray(0, dtype=_I32),     # bit used for current dish
        'pol': jnp.asarray(0, dtype=_I32),           # learned polarity xor
        'level': jnp.asarray(0, dtype=_I32),      # log-odds for the default read
        'wait': jnp.asarray(_FIRST_WAIT, dtype=_I32),
        'target': jnp.asarray(0, dtype=_I32),
        'have_target': jnp.asarray(0, dtype=_I32),
        'delivered': jnp.asarray(0, dtype=_I32),
        'grace': jnp.asarray(_GRACE_BUDGET, dtype=_I32),
        'extend': jnp.asarray(_EXTEND_BUDGET, dtype=_I32),
        'prev_valid': jnp.asarray(0, dtype=_I32),    # strip readable last look
        'since': jnp.asarray(0, dtype=_I32),         # steps since last delivery
        'lat': jnp.asarray(_POST_DELIVERY_WAIT, dtype=_I32),   # partner latency
        'lat_ok': jnp.asarray(0, dtype=_I32),        # latency actually measured
    }


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    mem = {k: jnp.where(reset, init[k], memory[k]).astype(_I32) for k in init}

    pi, pj = _landmark(obs, 22)          # pot at global (2, 4)
    row = 4 - pi
    col = 6 - pj
    facing = _facing(obs)

    me = obs[2, 2]
    held_plate = me[5] > 0
    held_cooked = me[6] > 0
    held_ing = jnp.sum(me[7:10]) > 0
    has_dish = held_plate & held_cooked
    has_plate = held_plate & (~held_cooked)
    empty_hands = (~held_plate) & (~held_cooked) & (~held_ing)

    pot = obs[pi, pj]
    pot_counts = pot[31:34]
    pot_total = jnp.sum(pot_counts)
    pot_cooked = (pot[30] > 0) & (pot[39] == 0)
    pot_cooking = pot[39] > 0
    pot_open = (~pot_cooked) & (~pot_cooking) & (pot_total < 3)

    # ---- read the column-3 signal strip (only visible from column 5) ----
    top, top_in = _cell(obs, row, col, 1, 3)
    mid, mid_in = _cell(obs, row, col, 2, 3)
    bot, bot_in = _cell(obs, row, col, 3, 3)
    view_ok = top_in & mid_in & bot_in
    top_occ = _has_item(top)
    mid_occ = _has_item(mid)
    bot_occ = _has_item(bot)

    # Which cells this partner has ever used decides how the strip is read.  A
    # partner that only ever loads one side counter is encoding "cell occupied"
    # against "cell empty"; one that uses both is encoding "which counter".
    ever_top = jnp.where(view_ok & top_occ, 1, mem['ever_top']).astype(_I32)
    ever_bot = jnp.where(view_ok & bot_occ, 1, mem['ever_bot']).astype(_I32)
    seen = jnp.maximum(ever_top, ever_bot).astype(_I32)
    both = (ever_top > 0) & (ever_bot > 0)
    only_top = (ever_top > 0) & (ever_bot == 0)
    only_bot = (ever_bot > 0) & (ever_top == 0)

    raw_bit = jnp.where(
        both, jnp.where(bot_occ, 1, 0),
        jnp.where(only_top, jnp.where(top_occ, 0, 1),
                  jnp.where(only_bot, jnp.where(bot_occ, 1, 0),
                            jnp.where(mid_occ, 0, 1)))).astype(_I32)
    # In the two-counter regime an empty or doubly loaded strip is a partner
    # mid-rewrite, not a reading; everywhere else every state is meaningful.
    raw_valid = view_ok & ((~both) | (top_occ != bot_occ))
    eff_bit = jnp.where(raw_valid, raw_bit, mem['last_bit']).astype(_I32)

    wait = jnp.maximum(mem['wait'] - 1, 0).astype(_I32)

    # ---- decide the ingredient for a fresh pot ----
    # A strip seen mid-rewrite and then settled on a different value is a
    # finished update: commit at once instead of burning the rest of the wait.
    since = (mem['since'] + 1).astype(_I32)
    fed_back = mem['delivered'] > 0
    # Only a readable strip that *became* unreadable is a partner rewrite; a
    # strip that never reads cleanly carries no timing information at all.
    prev_valid = jnp.where(view_ok, raw_valid.astype(_I32),
                           mem['prev_valid']).astype(_I32)
    rewriting = view_ok & (~raw_valid) & fed_back & (mem['prev_valid'] > 0)
    in_flight = jnp.where(rewriting, 1, mem['flight']).astype(_I32)
    settled = fed_back & raw_valid & (eff_bit != mem['committed'])
    # Only a correct delivery resamples the recipe, and there are two recipes,
    # so any strip movement after a delivery means "the other one".  That can be
    # acted on before the partner has finished writing the new value.
    infer_flip = fed_back & (in_flight > 0) & (~raw_valid)
    decode_bit = jnp.where(infer_flip, 1 - eff_bit, eff_bit).astype(_I32)
    # How long this particular partner takes to touch the strip after a
    # delivery.  Once measured, a strip still untouched that far past a delivery
    # means the recipe did not change, so the next dish can start earlier.
    disturbed = rewriting | (view_ok & fed_back & (eff_bit != mem['committed']))
    first_flight = disturbed & (mem['flight'] == 0)
    lat = jnp.where(first_flight,
                    jnp.where(mem['lat_ok'] > 0, jnp.maximum(mem['lat'], since), since),
                    mem['lat']).astype(_I32)
    lat_ok = jnp.maximum(mem['lat_ok'], first_flight.astype(_I32)).astype(_I32)
    in_flight = jnp.maximum(in_flight, disturbed.astype(_I32)).astype(_I32)

    ready = (mem['have_target'] == 0) & view_ok & (
        (wait == 0) | settled | infer_flip)
    stall_signal = ready & (seen == 0) & (mem['grace'] > 0)
    stall_flight = ready & (seen > 0) & (~raw_valid) & (~infer_flip) & (
        mem['extend'] > 0)
    commit = ready & (~stall_signal) & (~stall_flight)
    # Nothing is irreversible until the first ingredient goes into the pot, so a
    # late partner update is still taken into account until then.
    revise = (mem['have_target'] > 0) & (pot_total == 0) & raw_valid & (
        empty_hands | held_ing)

    # Delivery feedback as log-odds (base 2) in favour of reading the strip the
    # way the board suggests.  A wrong dish never resamples the recipe, so a
    # signal that moved proves the last delivery was correct and locks the
    # current reading in; a signal that stayed put is exactly one bit of
    # evidence against it.  The prior is the strength of the convention: strong
    # when the partner uses both row counters, weaker when it uses only one,
    # none at all when the strip carries nothing, where the rule degenerates
    # into strict alternation and still wins two deliveries out of three.
    prior = jnp.where(both, _PRIOR_BOTH,
                      jnp.where(only_top | only_bot, _PRIOR_ONE, 0)).astype(_I32)
    changed = fed_back & ((eff_bit != mem['committed']) | (in_flight > 0))
    stale = fed_back & (~changed)
    step = jnp.where(mem['pol'] == 1, 1, -1).astype(_I32)
    level = jnp.where(changed, jnp.where(mem['pol'] == 0, _LOCK, -_LOCK),
                      jnp.where(stale,
                                jnp.clip(mem['level'] + step, -_LOCK - 4, _LOCK + 4),
                                mem['level'])).astype(_I32)
    pol = jnp.where(level + prior >= 0, 0, 1).astype(_I32)

    new_target = jnp.bitwise_xor(decode_bit, pol).astype(_I32)
    target = jnp.where(commit | revise, new_target, mem['target']).astype(_I32)
    have_target = jnp.where(commit, 1, mem['have_target']).astype(_I32)

    # An already-loaded pot dictates the ingredient, whatever the strip says.
    pot_ing = jnp.argmax(pot_counts).astype(_I32)
    fetch_ing = jnp.where(pot_total > 0, pot_ing, target).astype(_I32)
    fetch_row = jnp.where(fetch_ing == 0, 1, 3).astype(_I32)

    # ---- pick the current sub-goal ----
    # Free wall counters of the right room, used to park an item that must not
    # go into the pot.  Standing cell and facing per counter.
    free_a = ~_has_item(_cell(obs, row, col, 0, 5)[0])
    free_b = ~_has_item(_cell(obs, row, col, 0, 6)[0])
    free_c = ~_has_item(_cell(obs, row, col, 4, 5)[0])
    park_row, park_col, park_face, park_any = _first_true(
        [free_a, free_b, free_c, jnp.asarray(True)],
        [1, 1, 3, 3],
        [5, 6, 5, 6],
        [F_UP, F_UP, F_DOWN, F_DOWN],
        [True, True, True, free_a | free_b | free_c],
    )

    # Verification before the irreversible first deposit: the strip may have
    # moved on while this ingredient was being fetched.
    held_ing_id = jnp.argmax(me[7:10]).astype(_I32)
    first_load = held_ing & (pot_total == 0)
    mismatch = first_load & raw_valid & (held_ing_id != target) & park_any
    hold_on = first_load & view_ok & (~raw_valid) & (mem['extend'] > 0)

    waiting = empty_hands & (pot_total == 0) & (have_target == 0)
    trow, tcol, face, interact = _first_true(
        [has_dish,                        # deliver the plated dish
         has_plate & pot_cooked,          # scoop the cooked pot
         has_plate & pot_cooking,         # stand by the pot until it is done
         has_plate,                       # stray plate: park it on a counter
         mismatch,                        # wrong ingredient in hand: park it
         hold_on,                         # strip mid-rewrite: hold the item
         held_ing & pot_open,             # load the pot
         held_ing,                        # stray ingredient: park it
         empty_hands & (pot_cooked | pot_cooking),   # fetch the plate
         waiting,                         # no recipe committed yet
         jnp.asarray(True)],              # fetch the next raw ingredient
        [2, 2, 2, park_row, park_row, 2, 2, park_row, 1, 2, fetch_row],
        [6, 5, 5, park_col, park_col, 5, 5, park_col, 5, 5, 6],
        [F_RIGHT, F_LEFT, F_LEFT, park_face, park_face, F_LEFT, F_LEFT,
         park_face, F_LEFT, F_LEFT, F_RIGHT],
        [True, True, False, True, True, False, True, True, True, False, True],
    )

    action = _step_toward(row, col, facing, trow, tcol, face, interact)

    delivering = has_dish & (action == A_INT)
    # A mismatch proves the partner was slower than measured: back off again.
    lat = jnp.where(mismatch,
                    jnp.minimum(jnp.maximum(lat, since + 1), _MAX_WAIT),
                    lat).astype(_I32)
    lat_ok = jnp.maximum(lat_ok, mismatch.astype(_I32)).astype(_I32)
    next_wait = jnp.where(lat_ok > 0,
                          jnp.clip(lat + 1, _MIN_WAIT, _MAX_WAIT),
                          _POST_DELIVERY_WAIT).astype(_I32)
    new_wait = jnp.where(
        delivering, next_wait,
        jnp.where(stall_signal | stall_flight, _EXTEND_WAIT, wait)).astype(_I32)

    next_mem = {
        'last_bit': eff_bit,
        'ever_top': ever_top,
        'ever_bot': ever_bot,
        'flight': jnp.where(commit | delivering, 0, in_flight).astype(_I32),
        'committed': jnp.where(commit | revise, decode_bit,
                               mem['committed']).astype(_I32),
        'pol': jnp.where(commit, pol, mem['pol']).astype(_I32),
        'level': jnp.where(commit, level, mem['level']).astype(_I32),
        'wait': new_wait,
        'target': target,
        'have_target': jnp.where(delivering, 0, have_target).astype(_I32),
        'delivered': jnp.where(delivering, 1,
                               jnp.where(commit, 0, mem['delivered'])).astype(_I32),
        'grace': jnp.where(stall_signal, mem['grace'] - 1, mem['grace']).astype(_I32),
        'extend': jnp.where(delivering, _EXTEND_BUDGET,
                            jnp.where(stall_flight | hold_on, mem['extend'] - 1,
                                      mem['extend'])).astype(_I32),
        'prev_valid': prev_valid,
        'since': jnp.where(delivering, 0, since).astype(_I32),
        'lat': lat,
        'lat_ok': lat_ok,
    }

    trace = {
        'target': target,
        'pol': next_mem['pol'],
        'level': next_mem['level'],
        'eff_bit': eff_bit,
        'seen': seen,
        'wait': new_wait,
        'pot_total': pot_total.astype(_I32),
        'commit': commit.astype(_I32),
        'since': next_mem['since'],
        'lat': lat,
        'mismatch': mismatch.astype(_I32),
    }
    return action, next_mem, trace
