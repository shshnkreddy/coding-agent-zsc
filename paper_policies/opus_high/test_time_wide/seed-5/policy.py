"""Overcooked V2 `test_time_wide` joint policy.

Layout facts that drive the design (rows x cols = 7 x 6)::

    WWXBWW      (0,2) delivery station, (0,3) plate pile
    0 A  0      row 1: ingredient-0 piles at (1,0)/(1,5)
    1    1      row 2: ingredient-1 piles at (2,0)/(2,5)
    WPWPWW      pots at (3,1)/(3,3); (3,2) and (3,4) are shared counters
    3 A  3      row 4: ingredient-3 piles at (4,0)/(4,5)
    W    W
    WWRWWW      (6,2) always-on recipe indicator

Row 3 is impassable, so agent_0 is confined to the top room (rows 1-2, cols
1-4) and agent_1 to the bottom room (rows 4-5, cols 1-4).  Only agent_0 can
reach ingredients 0/1, the plate pile and the delivery station, so agent_0 has
to cook and deliver on its own.  Only agent_1 can see the recipe indicator, so
agent_0 never observes the requested recipe directly.

agent_0 therefore combines two information sources.

Both are folded into one log-odds statistic L over two hypotheses about the
occupancy code of the shared counter (3,2), which is visible from every floor
cell of the top room:

    A: the request equals the code,   B: the request equals 1 - the code.

Only agent_0's own correct delivery can resample the recipe, so after a
delivery of ingredient d observed at code s:

* if the code changes soon afterwards, that delivery was correct, which
  supports the hypothesis that called it correct (log-odds +2.3, softened from
  the exact +infinity so a noisy partner can be recovered from);
* if it does not change, the delivery was either wrong (probability 1 of no
  change) or correct with a repeat draw (probability 1/2), which supports the
  other hypothesis by log 2.

With an inert partner the code is constant, the two hypotheses degenerate into
"the request is / is not what I just served", and the rule reproduces exactly
the optimal open-loop alternation that converges to a 2/3 hit rate.  With a
partner whose display tracks the recipe, a single observed change pins the
mapping down and every later delivery is correct.  No agreement about the
*direction* of the code is needed, only that the display is some consistent
function of the recipe.

agent_1 supplies such a display with one-step latency: it parks at (4,2) facing
the counter while holding an ingredient-3 unit, and toggles the counter so that
an item is present exactly when the request is ingredient 1, the direction a
reader with no shared convention is most likely to assume.  Mirroring the same
mark on the other shared counter was tried and abandoned: one hand cannot keep
two cells four steps apart in step, and the resulting five-to-eleven step lag
left a partner reading that counter worse off than with no display at all.
Nothing agent_0 does depends on that particular convention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "bare_candidate",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 7,
    "max_width": 6,
    "num_ingredients": 4,
}

# actions
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# facing ids used by the observation (UP, DOWN, RIGHT, LEFT) -> channels 1..4
FACE_UP, FACE_DOWN, FACE_RIGHT, FACE_LEFT = 0, 1, 2, 3

# goal table rows: stand_row, stand_col, turn action, facing id, interact flag
GOAL_TABLE_0 = (
    (1, 2, UP, FACE_UP, 1),      # 0 deliver at (0,2)
    (1, 3, UP, FACE_UP, 1),      # 1 plate pile (0,3)
    (2, 1, DOWN, FACE_DOWN, 1),  # 2 left pot (3,1)
    (2, 3, DOWN, FACE_DOWN, 1),  # 3 right pot (3,3)
    (1, 1, LEFT, FACE_LEFT, 1),   # 4 ingredient-0 pile (1,0), serves pot L
    (2, 4, RIGHT, FACE_RIGHT, 1),  # 5 ingredient-1 pile (2,5), serves pot R
    (1, 1, UP, FACE_UP, 1),      # 6 private counter (0,1)
    (1, 4, UP, FACE_UP, 1),      # 7 private counter (0,4)
    (-1, -1, STAY, FACE_UP, 0),  # 8 wait in place
    (2, 4, DOWN, FACE_DOWN, 1),   # 9 shared counter (3,4), emergency storage
    (2, 2, DOWN, FACE_DOWN, 1),   # 10 shared counter (3,2), emergency storage
    (1, 4, RIGHT, FACE_RIGHT, 1),  # 11 ingredient-0 pile (1,5), serves pot R
    (2, 1, LEFT, FACE_LEFT, 1),    # 12 ingredient-1 pile (2,0), serves pot L
)
G_DELIVER, G_PLATE, G_POTL, G_POTR = 0, 1, 2, 3
G_ING0, G_ING1, G_CNT1, G_CNT4, G_WAIT = 4, 5, 6, 7, 8
G_SH4, G_SH2, G_ING0R, G_ING1L = 9, 10, 11, 12

GOAL_TABLE_1 = (
    (4, 2, UP, FACE_UP, 1),      # 0 toggle the shared counter (3,2)
    (4, 2, UP, FACE_UP, 0),      # 1 hold the station, no interaction
    (4, 1, LEFT, FACE_LEFT, 1),  # 2 ingredient-3 pile (4,0)
    (5, 1, DOWN, FACE_DOWN, 1),  # 3 private counter (6,1)
)
S_TOGGLE, S_HOLD, S_FETCH, S_DROP = 0, 1, 2, 3

I32 = jnp.int32

# agent_0 integer memory slots
M_T = 0          # step counter
M_CODE = 1       # last observed signal code (0/1)
M_STABLE = 2     # steps since the code last changed
M_PEND = 3       # a delivery is waiting for a code change
M_DCODE = 4      # code observed at that delivery
M_DING = 5       # ingredient delivered
M_DOK = 6        # that delivery is usable for learning
M_DT = 7         # step at which that delivery happened
M_NCAP = 8       # the "no change" evidence of that delivery was applied
M_SEEN2 = 9      # shared counter (3,2) has been occupied at least once
M_SEEN4 = 10     # shared counter (3,4) has been occupied at least once
M_C4 = 11        # last observed occupancy of (3,4), which is not always in view
M_NOISE = 24     # display changes that no delivery of ours can explain
M_POT = 12       # 2 pots x 6 slots: n0 n1 n2 n3 cooked timer
M_PCODE = 25     # last known half of the room the partner stood in (0 left)
M_PDC = 26       # that half at the time of the pending delivery
M_PPEND = 27     # a delivery is waiting for the partner to move
M_PA = 28        # post-delivery moves supporting "request == partner half"
M_PB = 29        # post-delivery moves supporting the opposite reading
M_PSTAB = 30     # sightings during which the partner kept to the same half
M_PSEEN = 31     # step at which the partner was last in view
M_SIZE = 32

W_CHANGE = 2.3     # log-odds gain from an observed post-delivery code change
W_NOCHANGE = 0.69  # log-odds gain from no change (log 2)
L_CLAMP = 6.0      # keeps every conclusion revisable
L_INIT = 0.4       # mild prior that "counter occupied" means ingredient 1
SETTLE = 2         # steps after a delivery before "no change" is concluded
NOISE_WINDOW = 12  # steps after a delivery in which a change is still credible
NOISE_LIMIT = 3    # unexplainable display changes before the display is ignored
POS_WINDOW = 20    # the partner is only intermittently in view, so allow longer
POS_CONFIRM = 2    # unanimous post-delivery moves needed to read the position
POS_SETTLE = 4     # a restless partner's position is not treated as a message
POS_STALE = 8      # the delivery cell cannot see row 4, so allow a short gap


# --------------------------------------------------------------------------
# shared observation helpers
# --------------------------------------------------------------------------


def _column_from_pots(obs, pot_row):
    """Column of the observer, read off the visible pot cells of row 3."""
    row = obs[jnp.clip(pot_row, 0, 4), :, 24]
    j = jnp.max(jnp.where(row > 0, jnp.arange(5, dtype=I32), -1))
    return (5 - j).astype(I32)


def _locate_top(obs):
    """Global cell of agent_0 (rows 1-2, cols 1-4)."""
    row_zero_wall = jnp.sum(obs[0, :, 22]) > 0
    r = jnp.where(row_zero_wall, 2, 1).astype(I32)
    return r, _column_from_pots(obs, 5 - r)


def _locate_bottom(obs):
    """Global cell of agent_1 (rows 4-5, cols 1-4)."""
    rows = jnp.sum(obs[:, :, 25], axis=1)
    r = (8 - jnp.argmax(rows)).astype(I32)
    return r, _column_from_pots(obs, 5 - r)


def _cell(obs, r, c, gr, gc):
    """Channel vector of global cell (gr, gc); zeros plus a flag when unseen."""
    li = gr - r + 2
    lj = gc - c + 2
    vis = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    vec = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4)]
    return vec * vis.astype(I32), vis


def _item(vec):
    """(plate flag, cooked flag, counts, total) of a grid item block."""
    return vec[32], vec[33], vec[34:38], jnp.sum(vec[34:38])


def _goal_action(table, gidx, r, c, facing):
    """Walk to the goal cell, turn to its facing, then interact if requested."""
    row = jnp.asarray(table, dtype=I32)[gidx]
    tr, tc, turn, face_id, do_int = row[0], row[1], row[2], row[3], row[4]
    at = (r == tr) & (c == tc)
    facing_ok = facing[jnp.clip(face_id, 0, 3)] > 0
    settled = jnp.where(facing_ok, jnp.where(do_int > 0, INTERACT, STAY), turn)
    horiz = jnp.where(tc > c, RIGHT, LEFT)
    vert = jnp.where(tr > r, DOWN, UP)
    # Walk the axis that is not the one we must end up facing first, so that the
    # last step of the walk already sets the required facing and no separate
    # turn is needed on arrival.
    want_vertical = (face_id == FACE_UP) | (face_id == FACE_DOWN)
    step = jnp.where(want_vertical,
                     jnp.where(c != tc, horiz, vert),
                     jnp.where(r != tr, vert, horiz))
    action = jnp.where(at, settled, step)
    return jnp.where(tr < 0, jnp.asarray(STAY, I32), action).astype(I32)


# --------------------------------------------------------------------------
# agent_0 : chef in the top room
# --------------------------------------------------------------------------


def init_agent_0():
    ints = jnp.zeros((M_SIZE,), dtype=I32)
    return {"i": ints, "b": jnp.asarray(L_INIT, dtype=jnp.float32)}


def _pot_view(obs, r, c, gr, gc, remembered):
    """Fuse the observed pot cell with a dead-reckoned memory of it."""
    vec, vis = _cell(obs, r, c, gr, gc)
    _, cooked, counts, _ = _item(vec)
    timer = vec[44]
    seen = jnp.concatenate([counts, cooked[None], timer[None]]).astype(I32)
    old_timer = remembered[5]
    pred_timer = jnp.maximum(old_timer - 1, 0)
    pred_cooked = jnp.where(old_timer == 1, 1, remembered[4])
    pred = jnp.concatenate(
        [remembered[0:4], pred_cooked[None], pred_timer[None]]).astype(I32)
    return jnp.where(vis, seen, pred)


def _pot_facts(state):
    counts = state[0:4]
    cooked = state[4] > 0
    timer = state[5]
    total = jnp.sum(counts)
    return {"counts": counts, "cooked": cooked, "timer": timer,
            "cooking": timer > 0, "total": total,
            "idle": (~cooked) & (timer == 0)}


def _fill_score(pot, ing, preferred, allow_cross):
    """How willing we are to put `ing` into this pot; 0 means never."""
    clean = pot["counts"][ing] == pot["total"]  # empty, or only this ingredient
    ok = pot["idle"] & (pot["total"] < 3) & clean
    return jnp.where(ok & (pot["total"] > 0), 3,
                     jnp.where(ok & preferred, 2,
                               jnp.where(ok & allow_cross, 1, 0))).astype(I32)


def _poisoned(pot, ing):
    """Idle pot holding something that is not `ing`: never worth topping up."""
    return pot["idle"] & (pot["total"] > 0) & (pot["counts"][ing] != pot["total"])


def _pile_goal(ing, use_right_pot):
    """Ingredient pile nearest to the pot the ingredient is destined for."""
    return jnp.where(ing == 0,
                     jnp.where(use_right_pot, G_ING0R, G_ING0),
                     jnp.where(use_right_pot, G_ING1, G_ING1L)).astype(I32)


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    ints = jnp.where(reset, init["i"], memory["i"])
    logodds = jnp.where(reset, init["b"], memory["b"])

    r, c = _locate_top(obs)
    facing = obs[2, 2, 1:5]
    inv_plate = obs[2, 2, 5] > 0
    inv_cooked = obs[2, 2, 6] > 0
    inv_counts = obs[2, 2, 7:11]
    inv_total = jnp.sum(inv_counts)
    inv_id = jnp.argmax(inv_counts).astype(I32)
    held_dish = inv_cooked
    held_plate = inv_plate & (~inv_cooked)
    held_ing = (~inv_plate) & (~inv_cooked) & (inv_total > 0)

    # ---- signal counters (3,2) and (3,4) ----------------------------------
    # (3,2) is in view from every floor cell of the room, (3,4) only from
    # columns 2-4, so the latter is remembered between sightings.  Whichever
    # counter a partner has actually used carries the code; if neither has ever
    # been used the code is a constant and the inference below degenerates to
    # open-loop alternation.
    sig_vec, sig_vis = _cell(obs, r, c, 3, 2)
    s_plate, s_cooked, _, s_total = _item(sig_vec)
    raw2 = ((s_total + s_plate + s_cooked) > 0).astype(I32)
    alt_vec, alt_vis = _cell(obs, r, c, 3, 4)
    a_plate, a_cooked, _, a_total = _item(alt_vec)
    raw4 = jnp.where(alt_vis, ((a_total + a_plate + a_cooked) > 0).astype(I32),
                     ints[M_C4]).astype(I32)
    seen2 = jnp.maximum(ints[M_SEEN2], jnp.where(sig_vis, raw2, 0)).astype(I32)
    seen4 = jnp.maximum(ints[M_SEEN4], jnp.where(alt_vis, raw4, 0)).astype(I32)
    use_alt = (seen2 == 0) & (seen4 > 0)
    raw_code = jnp.where(use_alt, raw4,
                         jnp.where(sig_vis, raw2, ints[M_CODE])).astype(I32)
    # Only our own correct delivery can change the request, so a display that
    # changes long after our last delivery is not tracking the recipe.  After a
    # few such changes the display is ignored and the inference falls back to
    # pure alternation, which is the best a silent room allows.
    trusted = ints[M_NOISE] < NOISE_LIMIT
    spurious = trusted & (raw_code != ints[M_CODE]) & (ints[M_DT] > 0) \
        & (ints[M_PEND] == 0) & (ints[M_T] - ints[M_DT] > NOISE_WINDOW)
    noise = (ints[M_NOISE] + spurious.astype(I32)).astype(I32)
    trusted = noise < NOISE_LIMIT
    code = jnp.where(trusted, raw_code, 0).astype(I32)
    # switching which counter is believed to carry the signal invalidates the
    # evidence collected about the old one
    swapped = (use_alt != (ints[M_SEEN2] == 0) & (ints[M_SEEN4] > 0)) \
        | (noise == NOISE_LIMIT) & (ints[M_NOISE] < NOISE_LIMIT)
    logodds = jnp.where(swapped, jnp.float32(L_INIT), logodds)
    changed = code != ints[M_CODE]
    stable = jnp.where(changed, 0, ints[M_STABLE] + 1).astype(I32)

    # ---- Bayesian update of the code -> recipe mapping ---------------------
    # sign(+1) if hypothesis A ("request == code") called the pending delivery
    # correct, sign(-1) if hypothesis B did.
    pend = jnp.where(swapped, 0, ints[M_PEND])
    ncap = ints[M_NCAP]
    sgn = jnp.where(ints[M_DING] == ints[M_DCODE], 1.0, -1.0).astype(jnp.float32)
    moved = (pend > 0) & (code != ints[M_DCODE])
    # an observed change refutes the hypothesis that called the delivery wrong
    # and retracts any "no change" evidence already booked for it
    logodds = logodds + jnp.where(
        moved, sgn * (W_CHANGE + ncap.astype(jnp.float32) * W_NOCHANGE), 0.0)
    pend = jnp.where(moved, 0, pend)
    ncap = jnp.where(moved, 0, ncap)
    fresh = jnp.where(use_alt, alt_vis, sig_vis)
    settled = (pend > 0) & (ncap == 0) & fresh \
        & (ints[M_T] - ints[M_DT] >= SETTLE)
    logodds = logodds - jnp.where(settled, sgn * W_NOCHANGE, 0.0)
    ncap = jnp.where(settled, 1, ncap)
    logodds = jnp.clip(logodds, -L_CLAMP, L_CLAMP).astype(jnp.float32)

    # ---- second candidate code: the half of the room the partner stands in --
    # A partner may encode the request in where it stands rather than in a
    # counter.  Its cell is only intermittently in view, so the last sighting is
    # remembered.  The same "only our own correct delivery resamples the recipe"
    # argument applies: a move seen soon after a delivery proves that delivery
    # correct and therefore pins one reading of the position.  Unlike the
    # counter this code moves for reasons of its own, so it is believed only
    # after POS_CONFIRM unanimous confirmations and is dropped for good as soon
    # as one confirmation contradicts another.
    pmask = obs[:, :, 11]
    pvis = jnp.sum(pmask) > 0
    pj = jnp.argmax(jnp.sum(pmask, axis=0)).astype(I32)
    praw = ((c + pj - 2) >= 3).astype(I32)
    pcode = jnp.where(pvis, praw, ints[M_PCODE]).astype(I32)
    # Stability is counted over sightings only: while the partner is out of view
    # its remembered half proves nothing about how restless it is.
    pstab = jnp.where(~pvis, ints[M_PSTAB],
                      jnp.where(praw != ints[M_PCODE], 0,
                                ints[M_PSTAB] + 1)).astype(I32)
    pseen = jnp.where(pvis, ints[M_T], ints[M_PSEEN]).astype(I32)
    pfresh = ints[M_T] - ints[M_DT] <= POS_WINDOW
    pmoved = (ints[M_PPEND] > 0) & pvis & pfresh & (pcode != ints[M_PDC])
    psame = ints[M_DING] == ints[M_PDC]
    pa = (ints[M_PA] + (pmoved & psame).astype(I32)).astype(I32)
    pb = (ints[M_PB] + (pmoved & (~psame)).astype(I32)).astype(I32)
    ppend = jnp.where(pmoved | (~pfresh), 0, ints[M_PPEND]).astype(I32)
    pos_trust = (jnp.maximum(pa, pb) >= POS_CONFIRM) & (jnp.minimum(pa, pb) == 0)
    pos_guess = jnp.where(pa > pb, pcode, 1 - pcode).astype(I32)

    # ---- current best guess of the requested ingredient -------------------
    # The counters carry the request whenever a partner has ever used one; only
    # when both have stayed empty all episode is the position code consulted, so
    # every counter-signalling partner sees exactly the old behaviour.
    guess = jnp.where(logodds > 0.0, code, 1 - code).astype(I32)
    use_pos = (seen2 == 0) & (seen4 == 0) & pos_trust
    guess = jnp.where(use_pos, pos_guess, guess).astype(I32)

    # ---- pots --------------------------------------------------------------
    pot_mem = ints[M_POT:M_POT + 12].reshape(2, 6)
    potl = _pot_view(obs, r, c, 3, 1, pot_mem[0])
    potr = _pot_view(obs, r, c, 3, 3, pot_mem[1])
    other = 1 - guess
    fl = _pot_facts(potl)
    fr = _pot_facts(potr)
    # Default division of labour: the left pot cooks ingredient 0 next to the
    # left piles, the right pot ingredient 1 next to the right piles.  It is
    # only a preference: a pot holding anything else (a partner's ingredient 3,
    # say) is never topped up, so poisoned pots are abandoned rather than turned
    # into dishes that cost 20 to hand in.
    cross_g = jnp.where(guess == 0, _poisoned(fl, guess), _poisoned(fr, guess))
    cross_y = jnp.where(other == 0, _poisoned(fl, other), _poisoned(fr, other))
    gl = _fill_score(fl, guess, guess == 0, cross_g)
    gr = _fill_score(fr, guess, guess == 1, cross_g)
    yl = _fill_score(fl, other, other == 0, cross_y)
    yr = _fill_score(fr, other, other == 1, cross_y)
    g_use_r = gr > gl
    g_fill = jnp.maximum(gl, gr) > 0
    yl = jnp.where(g_fill & (~g_use_r), 0, yl)
    yr = jnp.where(g_fill & g_use_r, 0, yr)
    y_use_r = yr > yl
    y_fill = jnp.maximum(yl, yr) > 0

    gl_ready = fl["cooked"] & (fl["counts"][guess] == 3)
    gr_ready = fr["cooked"] & (fr["counts"][guess] == 3)
    g_ready = gl_ready | gr_ready
    yl_ready = fl["cooked"] & (fl["counts"][other] == 3)
    yr_ready = fr["cooked"] & (fr["counts"][other] == 3)
    y_ready = yl_ready | yr_ready
    gl_cook = fl["cooking"] & (fl["counts"][guess] == 3)
    gr_cook = fr["cooking"] & (fr["counts"][guess] == 3)
    g_cooking = gl_cook | gr_cook
    g_timer = jnp.where(gl_cook, fl["timer"], fr["timer"])
    junk_l = fl["cooked"] & (fl["counts"][0] != 3) & (fl["counts"][1] != 3)
    junk_r = fr["cooked"] & (fr["counts"][0] != 3) & (fr["counts"][1] != 3)
    junk_any = junk_l | junk_r
    junk_goal = jnp.where(junk_l, G_POTL, G_POTR)

    serve_r = jnp.where(g_ready, ~gl_ready, gr_cook)
    g_pot_goal = jnp.where(serve_r, G_POTR, G_POTL)
    y_pot_goal = jnp.where(yl_ready, G_POTL, G_POTR)
    g_ing_goal = _pile_goal(guess, g_use_r)
    y_ing_goal = _pile_goal(other, y_use_r)
    # steps needed to collect a plate and stand at the pot being served again
    plate_lead = jnp.abs(r - 1) + jnp.abs(c - 3) + 1 + jnp.where(serve_r, 1, 3)

    # ---- private counters (0,1) and (0,4) ---------------------------------
    c1_vec, c1_vis = _cell(obs, r, c, 0, 1)
    c4_vec, c4_vis = _cell(obs, r, c, 0, 4)

    def counter_facts(vec, vis):
        plate, cooked, counts, total = _item(vec)
        free = vis & ((total + plate + cooked) == 0)
        top = jnp.argmax(counts).astype(I32)
        dish = vis & (cooked > 0) & (total == 3) & (jnp.max(counts) == 3) \
            & (top == guess)
        bare = vis & (plate > 0) & (cooked == 0)
        raw = vis & (plate == 0) & (cooked == 0) & (total > 0)
        return free, dish, bare, raw, top

    c1_free, c1_dish, c1_plate, c1_raw, c1_top = counter_facts(c1_vec, c1_vis)
    c4_free, c4_dish, c4_plate, c4_raw, c4_top = counter_facts(c4_vec, c4_vis)
    dump_goal = jnp.where(c1_free, G_CNT1, jnp.where(c4_free, G_CNT4, G_WAIT))

    # ---- holding a finished dish ------------------------------------------
    # A dish that does not match the request is never handed in: it is stored
    # instead, on a shared counter if both private ones are taken, because a
    # wrong delivery costs 20 and a stored dish may still be wanted later.
    s2_free = sig_vis & (raw2 == 0)
    s4_free = alt_vis & (raw4 == 0)
    store_goal = jnp.where(
        c1_free, G_CNT1,
        jnp.where(c4_free, G_CNT4,
                  jnp.where(s4_free, G_SH4,
                            jnp.where(s2_free, G_SH2, G_WAIT))))
    dish_ok = (inv_total == 3) & (jnp.max(inv_counts) == 3) & (inv_id == guess)
    goal_dish = jnp.where(dish_ok, G_DELIVER, store_goal)

    # ---- holding an empty plate -------------------------------------------
    # An off-recipe pot that is already cooked is emptied onto a free counter:
    # the dish keeps indefinitely and the pot can start its next batch, which
    # turns waiting time into stock.
    bank_room = c1_free | c4_free
    store_room = store_goal != G_WAIT
    goal_plate = jnp.where(
        g_ready, g_pot_goal,
        jnp.where(y_ready & bank_room, y_pot_goal,
                  jnp.where(junk_any & store_room, junk_goal,
                            jnp.where(g_cooking, g_pot_goal, dump_goal))))

    # ---- holding a raw ingredient -----------------------------------------
    # It goes to the pot that was picked for it when the trip started; the same
    # scores are recomputed here, so the two decisions agree.
    is_g = inv_id == guess
    own_ok = jnp.where(is_g, g_fill, y_fill)
    own_goal = jnp.where(jnp.where(is_g, g_use_r, y_use_r), G_POTR, G_POTL)
    goal_raw = jnp.where((inv_id <= 1) & own_ok, own_goal, dump_goal)

    # ---- empty handed ------------------------------------------------------
    take_plate = jnp.where(c1_plate, G_CNT1,
                           jnp.where(c4_plate, G_CNT4, G_PLATE))
    take_g = jnp.where(c1_raw & (c1_top == guess), G_CNT1,
                       jnp.where(c4_raw & (c4_top == guess), G_CNT4, g_ing_goal))
    take_y = jnp.where(c1_raw & (c1_top == other), G_CNT1,
                       jnp.where(c4_raw & (c4_top == other), G_CNT4, y_ing_goal))
    # A display on (3,4) is only readable from columns 2-4, so when that is the
    # signal source, hold position for a moment after delivering rather than
    # walking out of sight and committing to a stale reading.
    blind_wait = use_alt & (ints[M_T] - ints[M_DT] <= 1) & (ints[M_DT] > 0)
    goal_empty = jnp.where(
        blind_wait, G_WAIT,
        jnp.where(
        c1_dish, G_CNT1,
        jnp.where(
            c4_dish, G_CNT4,
            jnp.where(
                g_ready, take_plate,
                jnp.where(
                    g_fill, take_g,
                    jnp.where(
                        g_cooking & (g_timer <= plate_lead), take_plate,
                        jnp.where(y_fill, take_y,
                                  jnp.where((junk_any & store_room)
                                            | (y_ready & bank_room),
                                            take_plate, G_WAIT))))))))

    goal = jnp.where(held_dish, goal_dish,
                     jnp.where(held_plate, goal_plate,
                               jnp.where(held_ing, goal_raw,
                                         goal_empty))).astype(I32)
    action = _goal_action(GOAL_TABLE_0, goal, r, c, facing)

    # ---- delivery bookkeeping ----------------------------------------------
    delivering = (goal == G_DELIVER) & (action == INTERACT) & held_dish
    pure_dish = (inv_total == 3) & (jnp.max(inv_counts) == 3) & (inv_id <= 1)
    usable = delivering & (stable >= 3) & pure_dish
    dcode = jnp.where(delivering, code, ints[M_DCODE]).astype(I32)
    dingn = jnp.where(delivering, inv_id, ints[M_DING]).astype(I32)
    dok = jnp.where(delivering, usable.astype(I32), ints[M_DOK]).astype(I32)
    pend = jnp.where(delivering, usable.astype(I32), pend).astype(I32)
    ncap = jnp.where(delivering, 0, ncap).astype(I32)
    dt = jnp.where(delivering, ints[M_T], ints[M_DT]).astype(I32)
    pdc = jnp.where(delivering, pcode, ints[M_PDC]).astype(I32)
    pusable = delivering & pure_dish & (pstab >= POS_SETTLE) \
        & (ints[M_T] - pseen <= POS_STALE)
    ppend = jnp.where(delivering, pusable.astype(I32), ppend).astype(I32)

    out = ints.at[M_T].set(ints[M_T] + 1)
    out = out.at[M_CODE].set(code)
    out = out.at[M_STABLE].set(stable)
    out = out.at[M_PEND].set(pend)
    out = out.at[M_DCODE].set(dcode)
    out = out.at[M_DING].set(dingn)
    out = out.at[M_DOK].set(dok)
    out = out.at[M_DT].set(dt)
    out = out.at[M_NCAP].set(ncap)
    out = out.at[M_SEEN2].set(seen2)
    out = out.at[M_SEEN4].set(seen4)
    out = out.at[M_C4].set(raw4)
    out = out.at[M_NOISE].set(noise)
    out = out.at[M_PCODE].set(pcode)
    out = out.at[M_PDC].set(pdc)
    out = out.at[M_PPEND].set(ppend)
    out = out.at[M_PA].set(pa)
    out = out.at[M_PB].set(pb)
    out = out.at[M_PSTAB].set(pstab)
    out = out.at[M_PSEEN].set(pseen)
    out = out.at[M_POT:M_POT + 12].set(jnp.concatenate([potl, potr]))

    trace = {
        "pos": jnp.stack([r, c]).astype(I32),
        "goal": goal,
        "guess": guess,
        "code": code,
        "tab": jnp.stack([pend, ncap]).astype(I32),
        "pos_code": jnp.stack([pcode, pa, pb, use_pos.astype(I32)]).astype(I32),
        "belief": logodds,
        "pot_l": potl.astype(I32),
        "pot_r": potr.astype(I32),
        "deliver": delivering.astype(I32),
        "inv": jnp.stack([inv_plate.astype(I32), inv_cooked.astype(I32),
                          inv_total.astype(I32), inv_id]),
    }
    return action.astype(I32), {"i": out.astype(I32), "b": logodds}, trace


# --------------------------------------------------------------------------
# agent_1 : recipe signaller in the bottom room
# --------------------------------------------------------------------------


def init_agent_1():
    return jnp.zeros((4,), dtype=I32)


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_1(), memory)
    r, c = _locate_bottom(obs)
    facing = obs[2, 2, 1:5]

    ind_vec, ind_vis = _cell(obs, r, c, 6, 2)
    want_on = ind_vis & (ind_vec[41] > 0)

    sig_vec, _ = _cell(obs, r, c, 3, 2)
    s_plate, s_cooked, _, s_total = _item(sig_vec)
    is_on = (s_total + s_plate + s_cooked) > 0

    inv_plate = obs[2, 2, 5] > 0
    inv_cooked = obs[2, 2, 6] > 0
    inv_counts = obs[2, 2, 7:11]
    holding = inv_plate | inv_cooked | (jnp.sum(inv_counts) > 0)
    has_marker = (~inv_plate) & (~inv_cooked) & (inv_counts[3] > 0)

    turn_on = want_on & (~is_on)
    turn_off = (~want_on) & is_on
    goal = jnp.where(
        turn_on,
        jnp.where(holding, S_TOGGLE, S_FETCH),
        jnp.where(
            turn_off,
            jnp.where(holding, S_DROP, S_TOGGLE),
            jnp.where(want_on | has_marker, S_HOLD,
                      jnp.where(holding, S_DROP, S_FETCH)),
        ),
    ).astype(I32)

    action = _goal_action(GOAL_TABLE_1, goal, r, c, facing)
    trace = {
        "pos": jnp.stack([r, c]).astype(I32),
        "goal": goal,
        "want_on": want_on.astype(I32),
        "is_on": is_on.astype(I32),
    }
    return action.astype(I32), mem.at[0].set(mem[0] + 1), trace
