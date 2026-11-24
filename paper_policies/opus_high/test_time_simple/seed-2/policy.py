"""Overcooked V2 ``test_time_simple``: split-room cook-and-signal policy.

Layout facts that drive every decision below (rows down, cols right)::

    WW2WWWWW      (0,2)/(4,2) ingredient-2 supply   (1,4)/(3,4) plate piles
    W  WB  0      (1,7) ingredient-0 supply         (2,4) pot
    R AWPA X      (3,7) ingredient-1 supply         (2,7) delivery
    W  WB  1      (2,0) always-on recipe indicator
    WW2WWWWW      (1,3)/(2,3)/(3,3) counters

The two walkable regions are disconnected.  The left seat reaches only the
recipe indicator, the ingredient-2 supplies and the column-3 counters; the
right seat reaches the pot, both plate piles, the ingredient-0/1 supplies and
the delivery station but can never see the recipe indicator (it is five
columns away and the view radius is two).  Recipes are three units of
ingredient 0 or three units of ingredient 1, so ingredient 2 is never edible
and the left seat can never contribute food.

Consequences:

* The right seat has to run the whole cook cycle alone: three ingredient
  trips (6 steps each), plate fetched while the pot cooks, scoop, deliver.
* The left seat's only useful act is telling the right seat which ingredient
  to use.  The only channel is an item left on a column-3 counter, which the
  right seat can see whenever it stands in column 5.
* The marker row mirrors the supply row: a marker on the *top* counter (1,3)
  means the *top* ingredient supply (1,7), i.e. ingredient 0; a marker on the
  *bottom* counter (3,3) means the *bottom* supply (3,7), i.e. ingredient 1.
  That geometric correspondence is the only non-arbitrary mapping available,
  so it is the convention most likely to be shared by an independently
  generated partner.  The middle counter is deliberately left unused.

Robustness choices for cross-play with an unknown partner:

* The right seat treats "no marker"/"both markers" as *no information* and
  waits a bounded number of steps rather than guessing, because the left seat
  removes the stale marker before placing the new one.
* With no marker ever seen (a partner that does not signal) the right seat
  alternates the cooked ingredient between deliveries.  A wrong delivery does
  not resample the recipe, so alternating converges to ~2/3 success instead of
  the 1/2 of a constant choice.
* A cooked dish that contradicts the currently displayed marker is dumped on a
  free counter instead of delivered: that costs 0 instead of -20 and frees the
  pot just as fast.
* The ingredient choice is only locked once something is in the pot, so a late
  marker update is still honoured during the walk to a supply.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'split_room_signaller',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids (different encoding from the action ids).
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# Action that turns towards a given facing id (movement into a blocked cell
# still updates facing, which every turn below relies on).
FACE_ACT = (UP, DOWN, RIGHT, LEFT)

I32 = jnp.int32


def _iarr(value):
    return jnp.asarray(value, dtype=I32)


def _facing(obs):
    """Facing id of the observer (one-hot over channels 1..4 at the centre)."""
    return jnp.argmax(obs[2, 2, 1:5]).astype(I32)


def _locate(obs, channel, glob_row, glob_col):
    """Global position of the observer from a unique static landmark."""
    plane = obs[:, :, channel].reshape(-1)
    idx = jnp.argmax(plane)
    row = glob_row - (idx // 5) + 2
    col = glob_col - (idx % 5) + 2
    return row.astype(I32), col.astype(I32)


def _cell(obs, row, col, grow, gcol):
    """Channel vector of global cell ``(grow, gcol)`` plus a visibility flag."""
    i = jnp.clip(grow - row + 2, 0, 4)
    j = jnp.clip(gcol - col + 2, 0, 4)
    visible = (jnp.abs(grow - row) <= 2) & (jnp.abs(gcol - col) <= 2)
    return obs[i, j, :], visible


def _has_item(cell):
    """True when a counter/pot cell holds any object (plate, food or dish)."""
    return jnp.sum(cell[29:34]) > 0


def _nav(row, col, trow, tcol, facing, need_face, at_target_action):
    """One step towards ``(trow, tcol)``, ending up facing ``need_face``.

    The axis aligned with ``need_face`` is travelled last, so the final move
    already sets the required facing and no separate turn step is needed in
    the common cases.  Standing on the target cell, the policy either turns or
    performs ``at_target_action``.
    """
    row_last = (need_face == F_UP) | (need_face == F_DOWN)
    step_row = jnp.where(trow > row, DOWN, UP)
    step_col = jnp.where(tcol > col, RIGHT, LEFT)
    need_row = row != trow
    need_col = col != tcol
    move = jnp.where(
        row_last,
        jnp.where(need_col, step_col, step_row),
        jnp.where(need_row, step_row, step_col),
    )
    at = (~need_row) & (~need_col)
    turn = jnp.asarray(FACE_ACT, dtype=I32)[need_face]
    action = jnp.where(at, jnp.where(facing == need_face, at_target_action, turn), move)
    return action.astype(I32)


# --------------------------------------------------------------------------
# agent_0 -- left room: read the recipe, keep exactly one counter marker.
# --------------------------------------------------------------------------

def init_agent_0():
    return {'t': _iarr(0)}


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(I32)
    step = jnp.where(reset, _iarr(0), memory['t']) + 1

    # The recipe indicator at (2,0) is visible from every cell of the left
    # room, so it localises the seat exactly.
    row, col = _locate(obs, 23, 2, 0)
    facing = _facing(obs)
    ind, _ = _cell(obs, row, col, 2, 0)
    # Recipes are [3,0,0] or [0,3,0]; channels 36/37 are the displayed counts.
    want1 = ind[37] > ind[36]
    good_row = jnp.where(want1, _iarr(3), _iarr(1))
    bad_row = jnp.where(want1, _iarr(1), _iarr(3))
    # Ingredient-2 supply used to create a marker sits above (1,2) or below (3,2).
    fetch_face = jnp.where(want1, _iarr(F_DOWN), _iarr(F_UP))

    good_cell, _ = _cell(obs, row, col, good_row, 3)
    bad_cell, _ = _cell(obs, row, col, bad_row, 3)
    good_set = _has_item(good_cell)
    bad_set = _has_item(bad_cell)

    me = obs[2, 2, :]
    holding = (me[5] + me[7] + me[8] + me[9]) > 0

    # Priorities: clear a stale marker first (so the ambiguous transient is
    # "nothing displayed" rather than "both displayed"), then place the right
    # one, then idle next to it ready for the next switch.
    branch = jnp.where(
        bad_set & (~holding),
        _iarr(0),  # pick the stale marker back up
        jnp.where(
            ~good_set,
            jnp.where(holding, _iarr(1), _iarr(2)),  # place / fetch
            jnp.where(holding, _iarr(3), _iarr(4)),  # stow spare / idle
        ),
    )

    one = _iarr(1)
    two = _iarr(2)
    trow = jnp.stack([bad_row, good_row, good_row, one, good_row])[branch]
    tcol = jnp.stack([two, two, two, one, two])[branch]
    nface = jnp.stack([
        _iarr(F_RIGHT), _iarr(F_RIGHT), fetch_face, _iarr(F_UP), _iarr(F_RIGHT),
    ])[branch]
    at_act = jnp.stack([
        _iarr(INTERACT), _iarr(INTERACT), _iarr(INTERACT), _iarr(INTERACT), _iarr(STAY),
    ])[branch]

    action = _nav(row, col, trow, tcol, facing, nface, at_act)

    trace = {
        'row': row,
        'col': col,
        'recipe': want1.astype(I32),
        'branch': branch,
        'good_set': good_set.astype(I32),
        'bad_set': bad_set.astype(I32),
    }
    return action, {'t': step}, trace


# --------------------------------------------------------------------------
# agent_1 -- right room: read the marker and run the whole cook cycle.
# --------------------------------------------------------------------------

# Branch table: target cell, required facing, action once in place.
#  0 deliver | 1 pot interact | 2 wait at pot | 3 plate pile
#  4 supply 0 | 5 supply 1 | 6 dump on (0,5) | 7 dump on (4,5)
_B_ROW = (2, 2, 2, 1, 1, 3, 1, 3)
_B_COL = (6, 5, 5, 5, 6, 6, 5, 5)
_B_FACE = (F_RIGHT, F_LEFT, F_LEFT, F_LEFT, F_RIGHT, F_RIGHT, F_UP, F_DOWN)
_B_ACT = (INTERACT, INTERACT, STAY, INTERACT, INTERACT, INTERACT, INTERACT, INTERACT)

# How long to keep waiting for a readable marker, and how long to wait for a
# first marker at the very start of an episode.
AMBIGUOUS_BUDGET = 12
OPENING_BUDGET = 12
# Consecutive deliveries after which an unchanged marker is taken as evidence
# that our reading of it is wrong.  A partner that tracks the recipe moves its
# marker after about half of our *correct* deliveries and never after a wrong
# one, so 3 unchanged deliveries in a row have probability 1/8 when we read the
# marker correctly but probability 1 when we do not.  An episode affords only
# about nine deliveries, so this is the least patient setting that still leaves
# an aligned partner nearly intact: measured against probe partners, 3 scores
# 173/-2/+6 (aligned/inverted/frozen) where 4 scores 176/-20/-32 and 2 scores
# 141/+8/+40.
POLARITY_PATIENCE = 3


def init_agent_1():
    z = _iarr(0)
    return {
        'belief': z,        # last readable marker: 0 or 1
        'has_sig': z,       # a marker has been read at least once
        'alt': z,           # alternating fallback choice
        'ambig': z,         # consecutive readable-position steps without a marker
        'since_d': _iarr(99),  # steps since our last delivery
        'dump': z,          # which counter to dump a bad dish on (0/1/2=give up)
        'ever': z,          # a marker has been seen on some column-3 counter
        'last': z,          # meaning of the most recent unambiguous marker
        'inv': z,           # read the marker inverted?
        'nochg': z,         # deliveries in a row that left the marker unchanged
        'trust': z,         # a marker change has been seen after a delivery
        'sig_d': _iarr(2),  # marker as of our previous delivery (2 = none)
        't': z,
    }


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(I32)
    mem = {k: jnp.where(reset, init_agent_1()[k], v) for k, v in memory.items()}
    step = mem['t'] + 1
    since_d = jnp.minimum(mem['since_d'] + 1, _iarr(99))

    # The pot at (2,4) is visible from every cell of the right room.
    row, col = _locate(obs, 22, 2, 4)
    facing = _facing(obs)

    me = obs[2, 2, :]
    plate = me[5] > 0
    cooked = me[6] > 0
    ing0, ing1 = me[7], me[8]
    holds_ing = (ing0 + ing1 + me[9]) > 0
    hold_dish = plate & cooked
    hold_plate = plate & (~cooked)

    pot, _ = _cell(obs, row, col, 2, 4)
    pot_n = pot[31] + pot[32] + pot[33]
    pot_ready = pot[30] > 0
    pot_cooking = pot[39] > 0
    pot_open = (pot_n < 3) & (~pot_cooking) & (~pot_ready)

    # --- read the marker -------------------------------------------------
    # Column 3 is only inside the 5x5 crop while standing in column 5.
    #
    # The reading is deliberately tolerant of several plausible partner
    # conventions.  A marker on the top counter means the top supply
    # (ingredient 0) and one on the bottom counter means the bottom supply
    # (ingredient 1); a marker on the unmotivated middle counter is read as
    # ingredient 1 and left to the polarity check below.  Once any marker has
    # been seen, an *empty* counter row is informative too: a partner that
    # clears the stale marker before showing the new one, or that encodes the
    # recipe as presence/absence on a single counter, says "the other
    # ingredient" by showing nothing.  Two simultaneous top/bottom markers are
    # the one genuinely ambiguous state and are treated as no reading.
    readable = col == 5
    top_cell, _ = _cell(obs, row, col, 1, 3)
    mid_cell, _ = _cell(obs, row, col, 2, 3)
    bot_cell, _ = _cell(obs, row, col, 3, 3)
    top = _has_item(top_cell) & readable
    mid = _has_item(mid_cell) & readable
    bot = _has_item(bot_cell) & readable

    shown = top | mid | bot
    ambiguous = top & bot
    ever = mem['ever'] | (shown & (~ambiguous)).astype(I32)
    sig_shown = jnp.where(top, _iarr(0), _iarr(1))
    sig = jnp.where(shown, sig_shown, 1 - mem['last'])
    sig_valid = readable & (~ambiguous) & (shown | (ever > 0))
    last = jnp.where(readable & shown & (~ambiguous), sig_shown, mem['last'])

    belief = jnp.where(sig_valid, sig, mem['belief'])
    has_sig = mem['has_sig'] | sig_valid.astype(I32)
    ambig = jnp.where(sig_valid, _iarr(0),
                      jnp.where(readable, mem['ambig'] + 1, mem['ambig']))

    # A marker that has been gone for a long time is treated as no channel at
    # all; freezing on a stale belief could repeat a -20 delivery forever.
    sig_dead = ambig > AMBIGUOUS_BUDGET
    use_sig = (has_sig > 0) & (~sig_dead)

    # Wait rather than commit while the marker is mid-update, and give the
    # partner a short grace period at the start of the episode.
    # ``opening`` also pulls the seat back to column 5 before it has ever seen
    # a marker; afterwards waiting only happens where the marker is readable,
    # otherwise the seat would bounce between the supply and the pot.
    opening = (has_sig == 0) & (step <= OPENING_BUDGET)
    stale = (~sig_valid) & ((has_sig > 0) | opening) & (~sig_dead)
    need_wait = (stale & (readable | opening)) | (since_d <= 1)

    # --- choose the ingredient ------------------------------------------
    # ``inv`` records that the delivery history has contradicted the plain
    # reading of the marker (see the polarity check below).
    eff = jnp.where(mem['inv'] > 0, 1 - belief, belief)
    want = jnp.where(use_sig, eff, mem['alt'])
    # Once the pot holds something the choice is locked: mixing guarantees a
    # dish that matches no recipe.
    want = jnp.where(pot_n > 0, jnp.where(pot[32] > pot[31], _iarr(1), _iarr(0)), want)

    # --- a dish that contradicts the marker is dumped, not delivered -----
    dish_ok = (~use_sig) | jnp.where(eff > 0, ing1 >= 3, ing0 >= 3)

    c05, v05 = _cell(obs, row, col, 0, 5)
    c45, v45 = _cell(obs, row, col, 4, 5)
    dump = jnp.where((mem['dump'] == 0) & v05 & _has_item(c05), _iarr(1), mem['dump'])
    dump = jnp.where((dump == 1) & v45 & _has_item(c45), _iarr(2), dump)
    dump_branch = jnp.where(dump == 0, _iarr(6), _iarr(7))

    branch = jnp.where(
        hold_dish,
        jnp.where(dish_ok | (dump >= 2), _iarr(0), dump_branch),
        jnp.where(
            hold_plate,
            jnp.where(pot_ready, _iarr(1), _iarr(2)),
            jnp.where(
                holds_ing,
                jnp.where(pot_open, _iarr(1), dump_branch),
                jnp.where(
                    pot_ready | pot_cooking,
                    _iarr(3),
                    jnp.where(need_wait, _iarr(2),
                              jnp.where(want > 0, _iarr(5), _iarr(4))),
                ),
            ),
        ),
    )

    action = _nav(row, col,
                  jnp.asarray(_B_ROW, dtype=I32)[branch],
                  jnp.asarray(_B_COL, dtype=I32)[branch],
                  facing,
                  jnp.asarray(_B_FACE, dtype=I32)[branch],
                  jnp.asarray(_B_ACT, dtype=I32)[branch])

    delivered = (branch == 0) & (action == INTERACT)
    since_d = jnp.where(delivered, _iarr(0), since_d)
    alt = jnp.where(delivered, 1 - mem['alt'], mem['alt'])

    # --- polarity check -------------------------------------------------
    # A correct delivery resamples the recipe, a wrong one does not.  So a
    # marker that never moves across several of our deliveries says our
    # deliveries are wrong, i.e. we are reading the marker backwards.  One
    # observed change proves the reading is right and disables the check.
    cur = jnp.where(use_sig, belief, _iarr(2))
    comparable = delivered & (cur < 2) & (mem['sig_d'] < 2)
    changed = comparable & (cur != mem['sig_d'])
    trust = mem['trust'] | changed.astype(I32)
    nochg = jnp.where(changed, _iarr(0),
                      jnp.where(comparable, mem['nochg'] + 1, mem['nochg']))
    flip = (nochg >= POLARITY_PATIENCE) & (trust == 0)
    inv = jnp.where(flip, 1 - mem['inv'], mem['inv'])
    nochg = jnp.where(flip, _iarr(0), nochg)
    sig_d = jnp.where(delivered & (cur < 2), cur, mem['sig_d'])

    next_mem = {
        'belief': belief.astype(I32),
        'has_sig': has_sig.astype(I32),
        'alt': alt.astype(I32),
        'ambig': ambig.astype(I32),
        'since_d': since_d.astype(I32),
        'dump': dump.astype(I32),
        'ever': ever.astype(I32),
        'last': last.astype(I32),
        'inv': inv.astype(I32),
        'nochg': nochg.astype(I32),
        'trust': trust.astype(I32),
        'sig_d': sig_d.astype(I32),
        't': step.astype(I32),
    }
    trace = {
        'row': row,
        'col': col,
        'branch': branch,
        'want': want.astype(I32),
        'belief': belief.astype(I32),
        'sig_valid': sig_valid.astype(I32),
        'pot_n': pot_n.astype(I32),
        'delivered': delivered.astype(I32),
        'inv': inv.astype(I32),
        'trust': trust.astype(I32),
        'nochg': nochg.astype(I32),
    }
    return action, next_mem, trace
