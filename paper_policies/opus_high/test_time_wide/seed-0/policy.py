"""Overcooked V2 `test_time_wide`: room-split cook + learned recipe signal.

Layout facts used below (public, from context/environment):

    row 0:  W W X B W W      X = delivery (0,2), B = plate pile (0,3)
    row 1:  0 . . . . 0      ingredient 0 supplies at (1,0) and (1,5)
    row 2:  1 . . . . 1      ingredient 1 supplies at (2,0) and (2,5)
    row 3:  W P W P W W      pots at (3,1) and (3,3)
    row 4:  3 . . . . 3      ingredient 3 supplies at (4,0) and (4,5)
    row 5:  W . . . . W
    row 6:  W W R W W W      always-on recipe indicator at (6,2)

Row 3 is solid, so the board is two disconnected rooms.  The *top* room
(rows 1-2) is the only place that can reach ingredients 0/1, plates and the
delivery station, and it can never see the recipe indicator.  The *bottom*
room (rows 4-5) can always see the recipe indicator and can do nothing else
useful.  The only viable division of labour is therefore: top seat cooks,
bottom seat tells it which of the two recipes ([0,0,0] or [1,1,1]) is live.
Seat is inferred from geometry, never from the seat index, so both act
functions run the same code and either seat can be paired with a stranger.

Signalling (bottom seat).  Stand still in the column half that matches the
ingredient id: (4,1), directly under the left pot, for ingredient 0; (4,3),
directly under the right pot, for ingredient 1; always facing the indicated
pot.  "Left/low is 0, right/high is 1" is the most focal reading of a binary
positional cue, both cells sit under a pot, and both are inside the view of
a top-seat partner standing anywhere in row 2 at columns 1-3, which is where
the cooking work happens.  The signaller never touches a pot or a counter,
so it cannot spoil a stranger's cook and cannot leave misleading objects
lying around.

Decoding (top seat) assumes no shared convention.  Seven binary properties of
the partner are tracked in parallel - its column half, whether it holds
anything, whether it is hidden from the lookout cell, two facing splits and
two further column thresholds - and each is treated as a cue of unknown
polarity.  The evidence is free and comes from the delivery itself: a correct
delivery resamples the recipe while a wrong one leaves it untouched, so a cue
that changes across a delivery proves the delivery was correct, which pins
that cue's polarity (old value against the ingredient just delivered) and
confirms it.  A cue that does not change is one bit of evidence against its
current polarity.  Cues that change *within* a batch, or that go unreadable
while the partner is in plain view, are not recipe-locked and are refused as
evidence.  The focal column cue decides by default; another dimension takes
over only after enough agreeing confirmations.  With no usable cue the seat
alternates its guess, which is the best blind rule available here (two thirds
correct, because a wrong delivery freezes the recipe and makes the opposite
guess certain next time).

Cooking (top seat) uses both pots.  One pot is filled with the believed
ingredient; while it cooks, the otherwise idle steps go into stocking the
other pot with the opposite ingredient.  That store is wanted with probability
one half after a correct delivery, because the recipe is resampled, and with
certainty after a wrong one, because a wrong delivery leaves the recipe
standing - so a misread recipe costs about twelve steps to correct instead of
a whole cycle.  Pot contents are remembered while out of view, since the five
by five window does not cover a pot and its ingredient supply at once.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'room_split_signal',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5
# Facing ids as they appear in the observation.
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# Movement action that produces each facing (indexed by facing id).
TURN_ACTION = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# Cue dimensions: column half, holds-anything, hidden from the lookout, faces
# right, faces down, column above 1, column above 3.
NCUE = 7

# Scalar memory slots.
M_G = 0              # ingredient the top seat is cooking this batch (0/1)
M_AWAIT = 1          # 1 while peeking at the partner before committing
M_PEEK = 2           # steps spent peeking
M_DELIV = 3          # 1 if the previous action was a delivery interaction
M_POT = 4            # column of the pot in use (1 or 3)
M_RECIPE = 5         # bottom seat: last recipe read from the indicator
M_SEEN = 6           # 1 once the partner has ever been inside the view
M_FIRST = 7          # 1 until the first batch has been committed
M_SEL = 8            # cue dimension the current batch was decided from
M_BLIND = 9          # 1 if the current batch was decided without a cue
M_AGE = 10           # steps since the current batch was committed
# Per-cue blocks of NCUE entries each.
S_VAL = 11                   # value the batch was committed on (-1 = unread)
S_PEEK = S_VAL + NCUE        # value seen during the current peek (-1 = none)
S_SIGN = S_PEEK + NCUE       # polarity: ingredient = value xor sign
S_CONF = S_SIGN + NCUE       # confidence in that polarity
S_CHURN = S_CONF + NCUE      # mid-batch changes seen (a signal has none)
S_PREV = S_CHURN + NCUE      # value at the previous sighting (-1 = none)
S_DIRTY = S_PREV + NCUE      # 1 if the cue moved away from its committed value
S_NCONF = S_DIRTY + NCUE     # consistent confirmations of this dimension
# Believed contents of each pot: [ingredient 0, ingredient 1, foreign, cooked,
# timer].  The view radius is only 2, so a pot leaves the field of view while
# its own ingredient supply is being visited; without this belief the pot would
# read as empty from there and the seat would oscillate between two plans.
P_W = 5
P_LEFT = S_NCONF + NCUE
P_RIGHT = P_LEFT + P_W
MEM_SIZE = P_RIGHT + P_W

PEEK_STEPS = 2       # observations to wait after a delivery before committing
PEEK_MAX = 6         # hard limit on that wait
# Confidence granted by a confirmed polarity, per cue dimension.  The
# positional cue is the one worth trusting for long; a wrong lock on a
# secondary dimension is capped so it can cost only a few deliveries.
CONF_LOCK = (6, 3, 3, 3, 3, 3, 3)
CONF_START = 1       # prior confidence in the focal polarity
# Confirmations a dimension needs before it may overrule the focal one.  Held
# items and presence in the near row are states a partner stays in, so one
# confirmation is enough; a facing direction or a column threshold flips in
# passing while a partner merely walks, so those need a second, agreeing one.
CONF_NEED = (0, 1, 1, 2, 2, 2, 2)
CHURN_LIMIT = 6      # mid-batch changes before a cue is dropped as noise
# A partner that signals by moving, or by picking something up, needs a few
# steps after the recipe changes to reach its new signalling state.  Cue
# movement inside this window after a commit is repositioning, not noise.
GRACE = 6
# Lowest pot countdown at which another hedge unit is worth fetching.  A unit
# round trip is six or seven steps and the plate run back to the primary pot is
# about six, so a margin of eight fits two hedge units into one 19-step cook
# window and still has the plate in hand when the dish comes up.
HEDGE_MARGIN = 8


def _init_memory():
    mem = jnp.zeros((MEM_SIZE,), dtype=jnp.int32)
    mem = mem.at[M_AWAIT].set(1)
    mem = mem.at[M_POT].set(1)
    mem = mem.at[M_FIRST].set(1)
    mem = mem.at[S_VAL:S_VAL + NCUE].set(-1)
    mem = mem.at[S_PEEK:S_PEEK + NCUE].set(-1)
    mem = mem.at[S_CONF:S_CONF + NCUE].set(CONF_START)
    mem = mem.at[S_PREV:S_PREV + NCUE].set(-1)
    return mem


def _act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    mem = jnp.where(reset, _init_memory(), memory)

    rows = jnp.arange(5, dtype=jnp.int32).reshape(5, 1)
    cols = jnp.arange(5, dtype=jnp.int32).reshape(1, 5)

    # ---- seat detection and self localisation -------------------------------
    # The delivery station (0,2) is inside the view from every cell of the top
    # room and from no cell of the bottom room; the recipe indicator (6,2) is
    # the mirror image.  Either landmark pins down the global self position.
    goal = obs[:, :, 23]
    indicator = obs[:, :, 25]
    is_top = jnp.sum(goal) > 0
    self_row = jnp.where(is_top, 2 - jnp.sum(goal * rows),
                         8 - jnp.sum(indicator * rows))
    self_col = jnp.where(is_top, 4 - jnp.sum(goal * cols),
                         4 - jnp.sum(indicator * cols))
    self_row = jnp.clip(self_row, 0, 6)
    self_col = jnp.clip(self_col, 0, 5)

    def cell(grow, gcol):
        """Channel vector of a global cell, zeros when outside the view."""
        i = grow - self_row + 2
        j = gcol - self_col + 2
        inside = (i >= 0) & (i <= 4) & (j >= 0) & (j <= 4)
        vec = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), :]
        return jnp.where(inside, vec, 0)

    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    turn_action = jnp.asarray(TURN_ACTION, dtype=jnp.int32)

    def navigate(trow, tcol, want_face, interact):
        """Walk to a cell of the (obstacle free) room, turn, then act."""
        dc = tcol - self_col
        dr = trow - self_row
        move = jnp.where(dc > 0, A_RIGHT,
                jnp.where(dc < 0, A_LEFT,
                 jnp.where(dr > 0, A_DOWN, A_UP)))
        arrived = (dc == 0) & (dr == 0)
        settled = jnp.where(facing != want_face, turn_action[want_face],
                            jnp.where(interact, A_INT, A_STAY))
        return jnp.where(arrived, settled, move)

    # =========================== bottom seat =================================
    ind_counts = cell(6, 2)[40:44]
    have_recipe = jnp.sum(ind_counts) > 0
    recipe_one = jnp.where(have_recipe,
                           (ind_counts[1] > ind_counts[0]).astype(jnp.int32),
                           mem[M_RECIPE])
    bottom_action = navigate(4, jnp.where(recipe_one == 1, 3, 1), F_UP, False)

    # ======================= top seat: cue observations ======================
    # A stranger in the bottom room has the same problem our signaller has but
    # may carry the recipe bit on a different observable.  Seven binary
    # properties of the partner are tracked in parallel; each one is only
    # "read" when it is actually observable, so an absent or ambiguous partner
    # never produces a default reading.  Dimension 0 is the focal one and wins
    # every tie; the rest can only take over the decision after a confirmed
    # change across a delivery.
    partner = obs[:, :, 11]
    partner_seen = jnp.sum(partner) > 0
    partner_col = self_col + jnp.sum(partner * cols) - 2
    partner_holds = jnp.sum(obs[:, :, 16:22]) > 0
    face_up = jnp.sum(partner * obs[:, :, 12]) > 0
    face_down = jnp.sum(partner * obs[:, :, 13]) > 0
    face_right = jnp.sum(partner * obs[:, :, 14]) > 0
    face_left = jnp.sum(partner * obs[:, :, 15]) > 0
    # Standing in row 2 at column 2 or 3, the whole of the bottom room's row 4
    # is inside the view, so "partner not visible" is a reliable reading there.
    at_lookout = (self_row == 2) & (self_col >= 2) & (self_col <= 3)

    raw = jnp.stack([(partner_col >= 3).astype(jnp.int32),
                     partner_holds.astype(jnp.int32),
                     (~partner_seen).astype(jnp.int32),
                     face_right.astype(jnp.int32),
                     face_down.astype(jnp.int32),
                     (partner_col >= 2).astype(jnp.int32),
                     (partner_col >= 4).astype(jnp.int32)])
    valid = jnp.stack([partner_seen,
                       partner_seen,
                       partner_seen | at_lookout,
                       partner_seen & (face_right | face_left),
                       partner_seen & (face_up | face_down),
                       partner_seen,
                       partner_seen])

    seen_ever = jnp.maximum(mem[M_SEEN], partner_seen.astype(jnp.int32))
    cue_val = mem[S_VAL:S_VAL + NCUE]
    cue_sign = mem[S_SIGN:S_SIGN + NCUE]
    cue_conf = mem[S_CONF:S_CONF + NCUE]
    cue_churn = mem[S_CHURN:S_CHURN + NCUE]
    cue_prev = mem[S_PREV:S_PREV + NCUE]
    cue_dirty = mem[S_DIRTY:S_DIRTY + NCUE]
    cue_nconf = mem[S_NCONF:S_NCONF + NCUE]

    # ---------------------------- inventory ---------------------------------
    held_plate = obs[2, 2, 5]
    held_cooked = obs[2, 2, 6]
    held = obs[2, 2, 7:11]
    n_held = jnp.sum(held)
    holding_dish = (held_plate == 1) & (held_cooked == 1)
    holding_plate = (held_plate == 1) & (held_cooked == 0)
    holding_ingredient = (held_plate == 0) & (n_held > 0)
    empty_handed = (held_plate == 0) & (n_held == 0)

    # --- post-delivery bookkeeping ------------------------------------------
    delivered = (mem[M_DELIV] == 1) & empty_handed
    await_flag = jnp.where(delivered, 1, mem[M_AWAIT])
    peek = jnp.where(delivered, 0, mem[M_PEEK])
    peek_val = jnp.where(delivered, -1, mem[S_PEEK:S_PEEK + NCUE])
    peek_val = jnp.where((await_flag == 1) & valid, raw, peek_val)

    # --- commit to the next batch -------------------------------------------
    # Never commit on a stale reading while a partner that is known to be
    # observable has not been looked at yet; if it is simply never visible,
    # commit at once and play blind rather than waste the horizon.
    # "Never seen" only counts as evidence of an absent signaller once the
    # lookout cell, from which the whole of row 4 is in view, has been
    # reached; otherwise the seat has simply not looked yet.
    commit = (await_flag == 1) & (peek >= PEEK_STEPS) & (
        jnp.any(peek_val >= 0) | ((seen_ever == 0) & at_lookout)
        | (peek >= PEEK_MAX))
    first = mem[M_FIRST] == 1
    g_prev = mem[M_G]

    new_val = jnp.where(peek_val >= 0, peek_val, cue_val)
    known = new_val >= 0
    changed = known & (cue_val >= 0) & (new_val != cue_val) & (~first)
    # For the secondary dimensions, only a cue that held still for the whole
    # batch and then changed across the delivery counts as evidence; one that
    # wandered mid-batch is an artefact and must not be allowed to take over
    # the decision.  The positional cue is the default reading anyway, so it
    # is not held to that standard: refusing its evidence only prolongs the
    # polarity search that the evidence is there to end.
    lock = jnp.asarray(CONF_LOCK, dtype=jnp.int32)
    primary = jnp.arange(NCUE) == 0
    evidence = changed & ((cue_dirty == 0) | primary)

    # A change proves the last delivery was correct, so the recipe in force
    # then was exactly the ingredient just delivered: that pins the polarity.
    sign_pin = jnp.bitwise_xor(jnp.maximum(cue_val, 0), g_prev)
    conf_dec = cue_conf - 1
    flip = (~changed) & (conf_dec < 0) & (~first) & known
    sign_next = jnp.where(evidence, sign_pin,
                          jnp.where(flip, 1 - cue_sign, cue_sign))
    conf_next = jnp.where(evidence, lock,
                          jnp.where(flip, 0, jnp.clip(conf_dec, 0, lock)))
    stale = first | (~known) | (changed & (~evidence))
    sign_next = jnp.where(stale, cue_sign, sign_next)
    conf_next = jnp.where(stale, cue_conf, conf_next)

    # How often a dimension has been confirmed, and consistently so: a pin that
    # contradicts the polarity an earlier pin established restarts the count.
    # A partner walking past produces the odd accidental confirmation, so a
    # secondary dimension only earns the right to overrule the focal one after
    # two agreeing confirmations.
    nconf_next = jnp.where(
        evidence,
        jnp.where((cue_nconf > 0) & (sign_pin != cue_sign), 1,
                  jnp.minimum(cue_nconf + 1, 3)),
        cue_nconf)
    nconf_next = jnp.where(stale, cue_nconf, nconf_next)

    # Use the most-confirmed cue that still looks like a signal; earlier
    # dimensions win ties, so the positional cue is the default reading.
    usable = (known & (cue_churn < CHURN_LIMIT)
              & (primary | (nconf_next >= jnp.asarray(CONF_NEED))))
    score = jnp.where(usable, conf_next * 4 - jnp.arange(NCUE), -100)
    sel = jnp.argmax(score)
    blind = (score[sel] < 0) | (seen_ever == 0)
    g_cue = jnp.bitwise_xor(new_val[sel], sign_next[sel])
    g_next = jnp.where(blind, jnp.where(first, g_prev, 1 - g_prev), g_cue)

    g = jnp.where(commit, g_next, mem[M_G])
    cue_sign = jnp.where(commit, sign_next, cue_sign)
    cue_conf = jnp.where(commit, conf_next, cue_conf)
    cue_val = jnp.where(commit, new_val, cue_val)
    cue_dirty = jnp.where(commit, 0, cue_dirty)
    cue_nconf = jnp.where(commit, nconf_next, cue_nconf)
    sel = jnp.where(commit, sel, mem[M_SEL])
    blind_flag = jnp.where(commit, blind.astype(jnp.int32), mem[M_BLIND])
    first = jnp.where(commit, 0, mem[M_FIRST])
    await_flag = jnp.where(commit, 0, await_flag)

    # --- late refinement of the reading --------------------------------------
    # Nothing is committed until the first ingredient of the batch is in a pot,
    # so a partner that is still walking to its signalling cell can still be
    # read.  This also rescues a batch that had to be committed blind because
    # the partner had not shown itself yet by the end of the peek.  A pot that
    # already holds a *cooked* dish is a store built in earlier cycles and does
    # not count as an investment in the current one.
    def in_view(grow, gcol):
        i = grow - self_row + 2
        j = gcol - self_col + 2
        return (i >= 0) & (i <= 4) & (j >= 0) & (j <= 4)

    def pot_belief(gcol, slot):
        """[n0, n1, foreign, cooked, timer]; remembered while out of view."""
        vec = cell(3, gcol)
        fresh = jnp.stack([vec[34], vec[35], vec[36] + vec[37], vec[33], vec[44]])
        old = mem[slot:slot + P_W]
        stale = jnp.stack([old[0], old[1], old[2],
                           jnp.where(old[4] == 1, 1, old[3]),
                           jnp.maximum(old[4] - 1, 0)])
        return jnp.where(in_view(3, gcol), fresh, stale)

    potL_vec = pot_belief(1, P_LEFT)
    potR_vec = pot_belief(3, P_RIGHT)

    def part_raw(desc):
        return (desc[4] == 0) & (desc[3] == 0) & (jnp.sum(desc[0:3]) > 0)

    invested = part_raw(potL_vec) | part_raw(potR_vec)
    refining = (await_flag == 0) & empty_handed & (~invested)
    cue_val = jnp.where(refining & valid, raw, cue_val)
    g = jnp.where(refining & valid[sel] & (cue_churn[sel] < CHURN_LIMIT),
                  jnp.bitwise_xor(raw[sel], cue_sign[sel]), g)
    hedge_ing = 1 - g

    # A recipe-locked cue never changes in the middle of a batch; one that
    # keeps flipping is not a signal.  Count transitions between successive
    # sightings, not steps of disagreement, so that a single stale reading
    # cannot condemn a partner that is standing still.
    age = jnp.where(commit, 0, jnp.minimum(mem[M_AGE] + 1, 99))
    settled = valid & (await_flag == 0) & (~refining) & (age >= GRACE)
    churn_hit = settled & (cue_prev >= 0) & (raw != cue_prev)
    cue_churn = cue_churn + churn_hit.astype(jnp.int32)
    cue_prev = jnp.where(valid, raw, cue_prev)
    # A dimension that stops being readable while the partner is in plain view
    # is not the channel the partner is signalling on - a facing cue read off a
    # signaller that was merely walking past looks recipe-locked otherwise,
    # because it goes quiet instead of contradicting itself.
    unreadable = ((~valid) & partner_seen & (await_flag == 0) & (~refining)
                  & (age >= GRACE))
    cue_dirty = jnp.maximum(
        cue_dirty, ((settled & (cue_val >= 0) & (raw != cue_val))
                    | unreadable).astype(jnp.int32))

    # --- where to dump something unusable ------------------------------------
    # Several reachable counters, so partner interference can never leave the
    # loop stuck holding something it cannot put down.
    a_free = jnp.sum(cell(0, 1)[32:38]) == 0
    b_free = jnp.sum(cell(0, 4)[32:38]) == 0
    c_free = jnp.sum(cell(3, 2)[32:38]) == 0
    dump_row = jnp.where(a_free | b_free, 1, 2)
    dump_col = jnp.where(a_free, 1, jnp.where(b_free, 4,
                         jnp.where(c_free, 2, 4)))
    dump_face = jnp.where(a_free | b_free, F_UP, F_DOWN)

    # --- pot choice ----------------------------------------------------------
    # Both pots are ranked for the believed ingredient every step, not just at
    # commit: a partner can drop a foreign ingredient into the pot being filled
    # at any moment, and a batch that cannot notice that would stall for the
    # rest of the episode.  The ranking is a pure function of the two pots, so
    # it cannot thrash, and it prefers the left pot on ties because (2,1) is
    # adjacent to both the left pot and an ingredient supply, which makes the
    # left pot two steps per unit cheaper to fill than the right one.
    def pot_rank(desc, target):
        total = jnp.sum(desc[0:3])
        mine = desc[target]
        cooked = desc[3] == 1
        timer = desc[4]
        addable = (~cooked) & (timer == 0) & (total < 3)
        return jnp.where(cooked & (mine == 3), 6,          # dish ready to plate
                jnp.where((timer > 0) & (mine == 3), 5,    # cooking, ours
                 jnp.where(addable & (mine == total) & (total > 0), 4,
                  jnp.where(total == 0, 3,                 # empty
                   jnp.where(addable, 1, 0)))))            # fouled / blocked

    left_first = pot_rank(potL_vec, g) >= pot_rank(potR_vec, g)
    pot_col = jnp.where(left_first, 1, 3)
    hedge_col = jnp.where(left_first, 3, 1)
    pot = jnp.where(left_first, potL_vec, potR_vec)
    hedge_pot = jnp.where(left_first, potR_vec, potL_vec)

    pot_total = jnp.sum(pot[0:3])
    pot_cooked = pot[3] == 1
    pot_ready = pot_cooked & (pot[g] == 3)
    pot_timer = pot[4]
    pot_cooking = pot_timer > 0
    pot_fillable = (~pot_cooked) & (~pot_cooking) & (pot_total < 3)

    # The hedge pot is stocked with the *other* ingredient while the primary
    # cooks.  After the next correct delivery the recipe is resampled, so the
    # opposite ingredient is wanted with probability one half; after a wrong
    # delivery it is wanted with certainty.  A cooked hedge pot is therefore a
    # dish in store that can be plated and delivered in about eight steps.
    hedge_total = jnp.sum(hedge_pot[0:3])
    hedge_fillable = ((hedge_pot[3] == 0) & (hedge_pot[4] == 0)
                      & (hedge_total < 3)
                      & (hedge_pot[hedge_ing] == hedge_total))

    # --- reactive cook loop ---------------------------------------------------
    dish_ok = held[g] == 3

    def supply_loc(ing, pcol):
        """Cell to stand on, and facing, to draw `ing` for the pot in `pcol`."""
        return (jnp.where(ing == 0, 1, 2),
                jnp.where(pcol == 1, 1, 4),
                jnp.where(pcol == 1, F_LEFT, F_RIGHT))

    g_row, g_col, g_face = supply_loc(g, pot_col)
    h_row, h_col, h_face = supply_loc(hedge_ing, hedge_col)

    # default: fetch the next raw ingredient for the primary pot
    t_row, t_col, t_face, t_int = g_row, g_col, g_face, True
    # the primary is cooking and there is time to spare -> stock the hedge pot
    to_hedge = (empty_handed & pot_cooking & hedge_fillable
                & (pot_timer >= HEDGE_MARGIN))
    t_row = jnp.where(to_hedge, h_row, t_row)
    t_col = jnp.where(to_hedge, h_col, t_col)
    t_face = jnp.where(to_hedge, h_face, t_face)
    # empty handed and the primary is busy or finished -> fetch a plate
    need_plate = empty_handed & (pot_cooked | pot_cooking) & (~to_hedge)
    t_row = jnp.where(need_plate, 1, t_row)
    t_col = jnp.where(need_plate, 3, t_col)
    t_face = jnp.where(need_plate, F_UP, t_face)
    # carrying the wanted ingredient -> put it in the primary pot
    to_pot = holding_ingredient & (held[g] == 1) & pot_fillable
    t_row = jnp.where(to_pot, 2, t_row)
    t_col = jnp.where(to_pot, pot_col, t_col)
    t_face = jnp.where(to_pot, F_DOWN, t_face)
    # carrying the opposite ingredient -> put it in the hedge pot
    to_hedge_pot = (holding_ingredient & (held[hedge_ing] == 1)
                    & hedge_fillable & (~to_pot))
    t_row = jnp.where(to_hedge_pot, 2, t_row)
    t_col = jnp.where(to_hedge_pot, hedge_col, t_col)
    t_face = jnp.where(to_hedge_pot, F_DOWN, t_face)
    # carrying a plate -> wait at the primary pot, collect as soon as it cooks
    plate_at_pot = holding_plate & (pot_cooked | pot_cooking)
    t_row = jnp.where(plate_at_pot, 2, t_row)
    t_col = jnp.where(plate_at_pot, pot_col, t_col)
    t_face = jnp.where(plate_at_pot, F_DOWN, t_face)
    t_int = jnp.where(plate_at_pot, pot_cooked, t_int)
    # carrying a finished dish -> deliver it
    to_deliver = holding_dish & dish_ok
    t_row = jnp.where(to_deliver, 1, t_row)
    t_col = jnp.where(to_deliver, 2, t_col)
    t_face = jnp.where(to_deliver, F_UP, t_face)
    # anything unusable in hand -> park it on a free counter
    junk = ((holding_dish & (~dish_ok))
            | (holding_ingredient & (~to_pot) & (~to_hedge_pot))
            | (holding_plate & (~pot_cooked) & (~pot_cooking)))
    t_row = jnp.where(junk, dump_row, t_row)
    t_col = jnp.where(junk, dump_col, t_col)
    t_face = jnp.where(junk, dump_face, t_face)
    t_int = jnp.where(junk, True, t_int)
    # peeking at the partner before the batch is committed
    peeking = (await_flag == 1) & empty_handed
    t_row = jnp.where(peeking, 2, t_row)
    t_col = jnp.where(peeking, 2, t_col)
    t_face = jnp.where(peeking, F_DOWN, t_face)
    t_int = jnp.where(peeking, False, t_int)

    top_action = navigate(t_row, t_col, t_face, t_int)

    # --- memory writeback -----------------------------------------------------
    delivering = (top_action == A_INT) & to_deliver & (self_row == 1) \
        & (self_col == 2) & (facing == F_UP)
    peek = jnp.where(await_flag == 1, peek + 1, 0)

    action = jnp.where(is_top, top_action, bottom_action)

    new_mem = mem
    new_mem = new_mem.at[M_G].set(g)
    new_mem = new_mem.at[M_AWAIT].set(await_flag)
    new_mem = new_mem.at[M_PEEK].set(peek)
    new_mem = new_mem.at[M_DELIV].set(
        jnp.where(is_top, delivering.astype(jnp.int32), 0))
    new_mem = new_mem.at[M_POT].set(pot_col)
    new_mem = new_mem.at[M_RECIPE].set(recipe_one)
    new_mem = new_mem.at[M_SEEN].set(seen_ever)
    new_mem = new_mem.at[M_FIRST].set(first)
    new_mem = new_mem.at[M_SEL].set(sel)
    new_mem = new_mem.at[M_BLIND].set(blind_flag)
    new_mem = new_mem.at[M_AGE].set(age)
    new_mem = new_mem.at[S_VAL:S_VAL + NCUE].set(cue_val)
    new_mem = new_mem.at[S_PEEK:S_PEEK + NCUE].set(peek_val)
    new_mem = new_mem.at[S_SIGN:S_SIGN + NCUE].set(cue_sign)
    new_mem = new_mem.at[S_CONF:S_CONF + NCUE].set(cue_conf)
    new_mem = new_mem.at[S_CHURN:S_CHURN + NCUE].set(cue_churn)
    new_mem = new_mem.at[S_PREV:S_PREV + NCUE].set(cue_prev)
    new_mem = new_mem.at[S_DIRTY:S_DIRTY + NCUE].set(cue_dirty)
    new_mem = new_mem.at[S_NCONF:S_NCONF + NCUE].set(cue_nconf)
    new_mem = new_mem.at[P_LEFT:P_LEFT + P_W].set(potL_vec)
    new_mem = new_mem.at[P_RIGHT:P_RIGHT + P_W].set(potR_vec)

    trace = {
        'is_top': is_top.astype(jnp.int32),
        'self_row': self_row.astype(jnp.int32),
        'self_col': self_col.astype(jnp.int32),
        'target_ingredient': g.astype(jnp.int32),
        'recipe_one': recipe_one.astype(jnp.int32),
        'cue_selected': sel.astype(jnp.int32),
        'sign': cue_sign[sel].astype(jnp.int32),
        'confidence': cue_conf[sel].astype(jnp.int32),
        'churn': jnp.sum(cue_churn).astype(jnp.int32),
        'blind': blind_flag.astype(jnp.int32),
        'nconf': jnp.sum(cue_nconf * (4 ** jnp.arange(NCUE))).astype(jnp.int32),
        'dirty': jnp.sum(cue_dirty * (2 ** jnp.arange(NCUE))).astype(jnp.int32),
        'churnb': jnp.sum(jnp.minimum(cue_churn, 3)
                          * (4 ** jnp.arange(NCUE))).astype(jnp.int32),
        'pot_col': pot_col.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'pot_ready': pot_ready.astype(jnp.int32),
        'hedge_total': hedge_total.astype(jnp.int32),
        'to_hedge': to_hedge.astype(jnp.int32),
        'hedge_fillable': hedge_fillable.astype(jnp.int32),
        'pot_timer': pot_timer.astype(jnp.int32),
        'empty_handed': empty_handed.astype(jnp.int32),
        'hedge_cooked': hedge_pot[3].astype(jnp.int32),
        'delivering': delivering.astype(jnp.int32),
    }
    return action.astype(jnp.int32), new_mem.astype(jnp.int32), trace


def init_agent_0():
    return _init_memory()


def agent_0_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)


def init_agent_1():
    return _init_memory()


def agent_1_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)
