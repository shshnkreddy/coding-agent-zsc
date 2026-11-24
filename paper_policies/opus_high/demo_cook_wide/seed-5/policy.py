"""Cooperative policy for the ``demo_cook_wide`` Overcooked-V2 layout.

Geometry forces a fixed division of labour that both seats can infer from the
static map alone, without any private code book:

* ``agent_0`` is locked inside the three-cell alcove ``(1,4) (1,5) (1,6)``.  It
  is the only seat that can reach the plate piles ``(0,4) (0,6)`` and the
  delivery station ``(0,5)``, and it reaches both ingredient piles and the pot
  in a couple of steps.  It cannot see the recipe indicator at ``(4,5)``.
* ``agent_1`` roams the lower room.  It is the only seat that can read the
  recipe indicator, and it can also load the pot at ``(2,5)`` from ``(3,5)``.

So the recipe has to travel upwards.  The channel used here is deliberately
self-describing rather than conventional: ``agent_1`` carries a unit of the
requested ingredient into the pot.  Pot contents, and the ingredient a visible
partner is carrying, are exactly what a partner policy would have to look at
anyway, so ``agent_0`` reads both, plus items left on the shared counters
``(2,4)``/``(2,6)``.

Two rules keep this honest against a partner whose conventions we do not share.
Evidence only counts if it post-dates our own last delivery, because that
delivery is what resampled the recipe; and a dish is delivered only if it is a
clean three-of-a-kind, since a wrong dish costs exactly what a right one earns.
If nothing decodable ever arrives, ``agent_0`` falls back to guessing and then
*alternating*: a wrong delivery does not resample the recipe, so the opposite
guess that follows it is certain to be right, which is worth about +6.7 a
delivery instead of the zero that waiting forever scores.  Every wait and every
recovery in here is bounded, because in cross-play unbounded patience is
indistinguishable from being broken.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'alcove_relay',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# --------------------------------------------------------------------------
# static constants derived from the published layout
# --------------------------------------------------------------------------

_GRID = (
    "WWWWBXBWWWW",
    "WWW0 A 1WWW",
    "WWWWWPWWWWW",
    "W    A    W",
    "0  W3R3W  0",
    "W1WWWWWWW1W",
)
# index into the ten static observation channels 22..31
_STATIC_SLOT = {"W": 0, "X": 1, "P": 2, "R": 3, "B": 5, "0": 6, "1": 7, "3": 9}
_H, _W, _NS = 6, 11, 10
_PAD = 2

_ACT_RIGHT, _ACT_DOWN, _ACT_LEFT, _ACT_UP, _ACT_STAY, _ACT_INT = 0, 1, 2, 3, 4, 5
_UP, _DOWN, _RIGHT, _LEFT = 0, 1, 2, 3

_FACE_TO_ACTION = (_ACT_UP, _ACT_DOWN, _ACT_RIGHT, _ACT_LEFT)

_A0_POS = ((1, 4), (1, 5), (1, 6))
_A1_POS = (
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 1), (4, 2), (4, 8), (4, 9),
)


# --------------------------------------------------------------------------
# small observation helpers
# --------------------------------------------------------------------------


def _templates(positions):
    """One-hot static-channel patches expected at each candidate cell.

    Built from the published layout on every trace; the values are compile-time
    constants, so this costs nothing at run time.
    """
    codes = []
    for row in range(-_PAD, _H + _PAD):
        line = []
        for col in range(-_PAD, _W + _PAD):
            inside = 0 <= row < _H and 0 <= col < _W
            symbol = _GRID[row][col] if inside else " "
            line.append(_STATIC_SLOT.get(symbol, -1))
        codes.append(line)
    board = jnp.asarray(codes, jnp.int32)
    slots = jnp.arange(_NS, dtype=jnp.int32)
    one_hot = (board[:, :, None] == slots[None, None, :]).astype(jnp.int32)
    return jnp.stack(
        [one_hot[row:row + 5, col:col + 5] for row, col in positions]
    )


def _static(obs):
    return obs[:, :, 22:32]


def _localise(obs, positions):
    # padded index of the window's top-left corner is (row + _PAD - 2, ...)
    templates = _templates([(row + _PAD - 2, col + _PAD - 2) for row, col in positions])
    match = jnp.sum(templates == _static(obs)[None], axis=(1, 2, 3))
    idx = jnp.argmax(match)
    rows = jnp.asarray([row for row, _ in positions], jnp.int32)
    cols = jnp.asarray([col for _, col in positions], jnp.int32)
    return rows[idx], cols[idx]


def _facing(obs):
    """Facing id of the observing agent (UP=0, DOWN=1, RIGHT=2, LEFT=3)."""
    one_hot = obs[2, 2, 1:5]
    return jnp.asarray(jnp.argmax(one_hot), jnp.int32)


def _cell(obs, self_row, self_col, row, col):
    dr = row - self_row + 2
    dc = col - self_col + 2
    visible = (dr >= 0) & (dr <= 4) & (dc >= 0) & (dc <= 4)
    vec = obs[jnp.clip(dr, 0, 4), jnp.clip(dc, 0, 4)]
    return vec, visible


def _pure_type(counts):
    """Ingredient id when ``counts`` is a single recipe ingredient, else -1."""
    total = jnp.sum(counts)
    only0 = (total > 0) & (counts[0] == total)
    only1 = (total > 0) & (counts[1] == total)
    return jnp.asarray(jnp.where(only0, 0, jnp.where(only1, 1, -1)), jnp.int32)


def _item_signal(vec):
    """Recipe ingredient advertised by a grid cell holding a bare ingredient."""
    bare = (vec[32] == 0) & (vec[33] == 0)
    return jnp.asarray(jnp.where(bare, _pure_type(vec[34:38]), -1), jnp.int32)


def _face_action(face):
    return jnp.take(jnp.asarray(_FACE_TO_ACTION, jnp.int32), face)


# --------------------------------------------------------------------------
# agent_0 : the alcove cook
# --------------------------------------------------------------------------

# belief, mine4, mine6, candidate, confirmations, partner-reloaded,
# stale left-counter signal, stale right-counter signal, steps partner unseen,
# steps spent yielding the last pot slot to the partner, starvation counter,
# partner-is-uninformative latch, next blind guess, spoiled-pot clear-outs used,
# previous pot load, whether we ourselves loaded it last step
_A0_INIT = (-1, 0, 0, -1, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0)
_A0_TRIP_STEPS = 3
_A0_YIELD_LIMIT = 8
_A0_PATIENCE = 45
# One clear-out per shared counter: each one ends with an inedible dish parked
# there, and a partner that keeps spoiling the pot must not be able to drag us
# through that cycle over and over.
_A0_MAX_CLEAROUTS = 2


def init_agent_0():
    return jnp.asarray(_A0_INIT, jnp.int32)


def _nav0(self_col, face, target_col, target_face, interact):
    """Walk the alcove to ``target_col``, turn, then interact (or wait)."""
    arrived = self_col == target_col
    step = jnp.where(target_col > self_col, _ACT_RIGHT, _ACT_LEFT)
    here = jnp.where(
        face == target_face,
        jnp.where(interact, _ACT_INT, _ACT_STAY),
        _face_action(target_face),
    )
    return jnp.asarray(jnp.where(arrived, here, step), jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    (
        belief, mine4, mine6, cand, conf, reloaded, stale_left, stale_right,
        unseen, yielded, starve, silent, guess, clearouts, prev_pot, self_fed,
    ) = (memory[i] for i in range(16))

    obs = jnp.asarray(obs, jnp.int32)
    self_col = _localise(obs, _A0_POS)[1]
    face = _facing(obs)

    self_vec = obs[2, 2]
    held_plate = self_vec[5]
    held_cooked = self_vec[6]
    held_counts = self_vec[7:11]
    held_total = jnp.sum(held_counts)
    held_type = jnp.asarray(jnp.argmax(held_counts), jnp.int32)
    has_dish = (held_plate > 0) & (held_cooked > 0)
    has_plate = (held_plate > 0) & (held_cooked == 0)
    has_ingredient = held_total > 0
    holding_any = (held_plate > 0) | (held_total > 0)

    pot_vec, _ = _cell(obs, 1, self_col, 2, 5)
    pot_counts = pot_vec[34:38]
    pot_total = jnp.sum(pot_counts)
    pot_ready = pot_vec[33] > 0
    pot_cooking = pot_vec[44] > 0
    pot_idle = (~pot_ready) & (~pot_cooking)
    pot_room = pot_idle & (pot_total < 3)
    pot_type = _pure_type(pot_counts)
    pot_spoiled = (pot_total > 0) & (pot_type < 0)

    left_vec, _ = _cell(obs, 1, self_col, 2, 4)
    right_vec, _ = _cell(obs, 1, self_col, 2, 6)
    left_free = (left_vec[32] == 0) & (left_vec[33] == 0) & (jnp.sum(left_vec[34:38]) == 0)
    right_free = (right_vec[32] == 0) & (right_vec[33] == 0) & (jnp.sum(right_vec[34:38]) == 0)
    mine4 = jnp.where(left_free, 0, mine4)
    mine6 = jnp.where(right_free, 0, mine6)
    left_signal = jnp.where(mine4 > 0, -1, _item_signal(left_vec))
    right_signal = jnp.where(mine6 > 0, -1, _item_signal(right_vec))

    partner_seen = jnp.sum(obs[:, :, 11]) > 0
    partner_counts = jnp.sum(obs[:, :, 18:22], axis=(0, 1))
    partner_bare = (jnp.sum(obs[:, :, 16]) == 0) & (jnp.sum(obs[:, :, 17]) == 0)
    partner_signal = jnp.where(partner_seen & partner_bare, _pure_type(partner_counts), -1)

    # Soft evidence has to post-date our last delivery, because that delivery is
    # what resampled the recipe.  An ingredient the partner was already carrying
    # then, or one that has sat untouched on a counter since then, says nothing
    # about the new target, so it only counts once the partner has been seen
    # empty-handed again or the counter has actually changed.
    # "Post-dates the delivery" means either we watched the partner's hands come
    # up empty, or it was away long enough to have collected something new; a
    # partner that simply stood there holding the same unit is not evidence.
    unseen = jnp.asarray(jnp.where(partner_seen, 0, unseen + 1), jnp.int32)
    reloaded = jnp.where(partner_seen & (partner_signal < 0), 1, reloaded)
    reloaded = jnp.where(unseen >= _A0_TRIP_STEPS, 1, reloaded)
    stale_left = jnp.where(left_signal < 0, -1, stale_left)
    stale_right = jnp.where(right_signal < 0, -1, stale_right)
    partner_fresh = jnp.where(reloaded > 0, partner_signal, -1)
    left_fresh = jnp.where(left_signal == stale_left, -1, left_signal)
    right_fresh = jnp.where(right_signal == stale_right, -1, right_signal)
    # Two consecutive frames are still required: a partner merely walking past
    # with an ingredient it is about to put away must not flip our belief.
    soft = jnp.where(
        partner_fresh >= 0,
        partner_fresh,
        jnp.where(left_fresh >= 0, left_fresh, right_fresh),
    )
    conf = jnp.where(soft < 0, 0, jnp.where(soft == cand, conf + 1, 1))
    cand = soft
    pot_evidence = (pot_total > 0) & (pot_type >= 0)
    informed = (soft >= 0) & (conf >= 2)
    belief = jnp.where(
        pot_evidence,
        pot_type,
        jnp.where((belief < 0) & informed, soft, belief),
    )

    # If nothing usable ever arrives, waiting scores zero for the rest of the
    # episode.  Guessing and then *alternating* does better: a wrong delivery
    # does not resample the recipe, so the opposite guess that follows it is
    # certain to be right.  That is worth about +6.7 a delivery against a
    # partner that never says anything, and it only unlocks after a long
    # silence, so a partner that does communicate never triggers it.
    starve = jnp.asarray(
        jnp.where(informed | (belief >= 0), 0, starve + 1), jnp.int32
    )
    silent = jnp.where(informed, 0, jnp.where(starve >= _A0_PATIENCE, 1, silent))
    goal = jnp.where(belief >= 0, belief, jnp.where(silent > 0, guess, -1))

    plate_col = jnp.where(self_col <= 5, 4, 6)
    drop_col = jnp.where(left_free, 4, 6)
    can_drop = left_free | right_free

    # A pot holding a mixture, or anything that is not a recipe ingredient, can
    # only be freed by cooking it and parking the result on a counter.  That is
    # worth doing a couple of times, never in a loop: delivering the result
    # would simply pay -20 per attempt.
    clearing = pot_spoiled & (clearouts < _A0_MAX_CLEAROUTS) & can_drop
    pot_worth_cooking = (pot_type >= 0) | clearing

    # holding a finished dish -----------------------------------------------
    dish_type = _pure_type(held_counts)
    deliver_ok = has_dish & (held_total == 3) & (dish_type >= 0)
    act_dish = jnp.where(
        deliver_ok,
        _nav0(self_col, face, 5, _UP, True),
        jnp.where(
            can_drop,
            _nav0(self_col, face, drop_col, _DOWN, True),
            _nav0(self_col, face, 5, _DOWN, False),
        ),
    )

    # holding an empty plate -------------------------------------------------
    pot_wants_food = pot_room & ((goal >= 0) | clearing)
    act_plate = jnp.where(
        pot_ready & pot_worth_cooking,
        _nav0(self_col, face, 5, _DOWN, True),
        jnp.where(
            pot_wants_food & can_drop,
            _nav0(self_col, face, drop_col, _DOWN, True),
            _nav0(self_col, face, 5, _DOWN, False),
        ),
    )

    # holding a raw ingredient ----------------------------------------------
    # Only commit an ingredient to an empty pot once we actually have a target;
    # an ingredient picked up merely to unblock a counter must be held, not
    # dropped straight back down, or the two would shuffle forever.
    fits_pot = pot_room & (
        (pot_type == held_type)
        | ((pot_total == 0) & (goal == held_type))
        | clearing
    )
    need_hands = (pot_ready | pot_cooking | (pot_total >= 3)) & pot_worth_cooking
    wrong_ingredient = (goal >= 0) & (goal != held_type) & (~clearing)
    act_carry = jnp.where(
        fits_pot,
        _nav0(self_col, face, 5, _DOWN, True),
        jnp.where(
            (wrong_ingredient | need_hands) & can_drop,
            _nav0(self_col, face, drop_col, _DOWN, True),
            _nav0(self_col, face, 5, _DOWN, False),
        ),
    )

    # empty handed -----------------------------------------------------------
    fill_type = jnp.where(goal >= 0, goal, 0)
    want_fill = pot_room & (
        clearing | ((goal >= 0) & ((pot_total == 0) | (pot_type == goal)))
    )
    # never pick an ingredient up that we might be unable to put down again
    want_fill = want_fill & (can_drop | (pot_total <= 1))
    # The partner may be about to top the pot up, so briefly leave the last slot
    # for it rather than fetch a spare we would have to dump.  Strictly bounded:
    # a partner that simply stands there holding an ingredient must not be able
    # to stall the pot for the rest of the episode.
    # The wait is short and it is refunded every time the partner is actually
    # seen loading the pot, so a partner that uses the slot keeps being offered
    # it while one stuck holding a spare costs only a few steps a cycle.
    partner_fed = (pot_total > prev_pot) & (self_fed == 0)
    partner_inbound = partner_seen & (partner_signal >= 0) & (pot_total == 2)
    yielded = jnp.asarray(
        jnp.where(
            partner_fed | (pot_total < 2),
            0,
            jnp.where(partner_inbound, yielded + 1, yielded),
        ),
        jnp.int32,
    )
    partner_inbound = partner_inbound & (yielded <= _A0_YIELD_LIMIT)
    # Taking a matching ingredient off a shared counter costs one extra step and
    # clears it, so a counter-signalling partner has to re-post a fresh signal
    # instead of leaving a stale one behind after the recipe is resampled.
    reclaim4 = _item_signal(left_vec) == fill_type
    reclaim6 = _item_signal(right_vec) == fill_type
    pile_col = jnp.where(fill_type == 0, 4, 6)
    pile_face = jnp.where(fill_type == 0, _LEFT, _RIGHT)
    act_fetch = jnp.where(
        reclaim4,
        _nav0(self_col, face, 4, _DOWN, True),
        jnp.where(
            reclaim6,
            _nav0(self_col, face, 6, _DOWN, True),
            _nav0(self_col, face, pile_col, pile_face, True),
        ),
    )
    # A counter-signalling partner that only re-posts when its counter is empty
    # would otherwise be stuck behind its own outdated signal.  Lifting the
    # untrusted item off makes room for a fresh one; we only do it when the
    # other counter is free, so the item we are now holding can always be put
    # back down again.
    blocked_left = (left_signal >= 0) & (left_fresh < 0) & right_free
    blocked_right = (right_signal >= 0) & (right_fresh < 0) & left_free
    unblock = (belief < 0) & pot_idle & (pot_total == 0) & (
        blocked_left | blocked_right
    )
    act_unblock = _nav0(
        self_col, face, jnp.where(blocked_left, 4, 6), _DOWN, True
    )
    act_empty = jnp.where(
        (pot_ready | pot_cooking | (pot_total >= 3)) & pot_worth_cooking,
        _nav0(self_col, face, plate_col, _UP, True),
        jnp.where(
            want_fill & (~partner_inbound),
            act_fetch,
            jnp.where(
                unblock, act_unblock, _nav0(self_col, face, 5, _DOWN, False)
            ),
        ),
    )

    action = jnp.where(
        has_dish,
        act_dish,
        jnp.where(has_plate, act_plate, jnp.where(has_ingredient, act_carry, act_empty)),
    )
    action = jnp.asarray(action, jnp.int32)

    delivered = deliver_ok & (action == _ACT_INT) & (self_col == 5) & (face == _UP)
    dropping = (action == _ACT_INT) & (face == _DOWN) & (self_col != 5) & holding_any
    belief = jnp.where(delivered, -1, belief)
    mine4 = jnp.where(dropping & (self_col == 4), 1, mine4)
    mine6 = jnp.where(dropping & (self_col == 6), 1, mine6)
    reloaded = jnp.where(delivered, 0, reloaded)
    stale_left = jnp.where(delivered, left_signal, stale_left)
    stale_right = jnp.where(delivered, right_signal, stale_right)
    # alternate the blind guess after every delivery (see the starvation note)
    guess = jnp.where(delivered, 1 - guess, guess)
    clearouts = jnp.asarray(
        jnp.where(dropping & has_dish, clearouts + 1, clearouts), jnp.int32
    )
    self_fed = jnp.asarray(
        (action == _ACT_INT) & (face == _DOWN) & (self_col == 5) & has_ingredient,
        jnp.int32,
    )
    prev_pot = jnp.asarray(pot_total, jnp.int32)

    next_memory = jnp.stack(
        [
            belief, mine4, mine6, cand, conf, reloaded, stale_left, stale_right,
            unseen, yielded, starve, silent, guess, clearouts, prev_pot, self_fed,
        ]
    ).astype(jnp.int32)
    trace = {
        "belief": belief,
        "goal": jnp.asarray(goal, jnp.int32),
        "self_col": jnp.asarray(self_col, jnp.int32),
        "pot_total": jnp.asarray(pot_total, jnp.int32),
        "pot_type": pot_type,
        "silent": jnp.asarray(silent, jnp.int32),
        "delivered": jnp.asarray(delivered, jnp.int32),
    }
    return action, next_memory, trace


# --------------------------------------------------------------------------
# agent_1 : the recipe reader / pot loader
# --------------------------------------------------------------------------

# recipe, slot0, slot1, fetching, wanted ingredient, blocked-pot counter,
# staging flag, hold-off counter, forced deposits used
_A1_INIT = (-1, -1, -1, 0, -1, 0, 0, 0, 0)
_A1_STALL_LIMIT = 15
_A1_STAGE_MARGIN = 15
_A1_HOLDOFF_LIMIT = 6
# Enough forced deposits to cook one blocked pot out, and no more: a partner
# that keeps re-blocking the pot must not be able to farm -20 deliveries off us.
_A1_MAX_FORCED = 3


def init_agent_1():
    return jnp.asarray(_A1_INIT, jnp.int32)


def _nav1(row, col, face, target_row, target_col, target_face, interact):
    """Row 3 is a corridor; the four row-4 stubs hang off it."""
    arrived = (row == target_row) & (col == target_col)
    climb = (row == 4) & (~((target_row == 4) & (target_col == col)))
    sideways = jnp.where(target_col > col, _ACT_RIGHT, _ACT_LEFT)
    along_row = jnp.where(col != target_col, sideways, _ACT_DOWN)
    step = jnp.where(climb, _ACT_UP, along_row)
    here = jnp.where(
        face == target_face,
        jnp.where(interact, _ACT_INT, _ACT_STAY),
        _face_action(target_face),
    )
    return jnp.asarray(jnp.where(arrived, here, step), jnp.int32)


def _slot_state(obs, row, col, slot_row, slot_col, remembered):
    vec, visible = _cell(obs, row, col, slot_row, slot_col)
    counts = vec[34:38]
    total = jnp.sum(counts)
    busy = (total > 0) | (vec[32] > 0) | (vec[33] > 0)
    single = (total == 1) & (vec[32] == 0) & (vec[33] == 0)
    seen = jnp.where(busy, jnp.where(single, jnp.argmax(counts), 8), -1)
    return jnp.asarray(jnp.where(visible, seen, remembered), jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    recipe, slot0, slot1, fetching, want, stall, staging, holdoff, forces = (
        memory[i] for i in range(9)
    )

    obs = jnp.asarray(obs, jnp.int32)
    row, col = _localise(obs, _A1_POS)
    face = _facing(obs)

    shown = jnp.sum(obs[:, :, 40:44], axis=(0, 1))
    shown_type = jnp.where(shown[0] > 0, 0, jnp.where(shown[1] > 0, 1, -1))
    recipe = jnp.asarray(jnp.where(shown_type >= 0, shown_type, recipe), jnp.int32)

    self_vec = obs[2, 2]
    held_counts = self_vec[7:11]
    held_total = jnp.sum(held_counts)
    held_type = jnp.asarray(jnp.argmax(held_counts), jnp.int32)
    carrying = held_total > 0

    pot_vec, pot_visible = _cell(obs, row, col, 2, 5)
    pot_counts = pot_vec[34:38]
    pot_total = jnp.sum(pot_counts)
    pot_idle = (pot_vec[33] == 0) & (pot_vec[44] == 0)
    pot_type = _pure_type(pot_counts)

    slot0 = _slot_state(obs, row, col, 4, 3, slot0)
    slot1 = _slot_state(obs, row, col, 4, 7, slot1)

    # A partner already holding a finished dish is about to deliver it, which
    # resamples the recipe.  Anything we put in the pot during those few steps
    # would silently become the previous target, which is exactly the trap that
    # makes an otherwise-sound partner mislead its cook.  The counter keeps a
    # partner that never delivers from freezing us forever.
    partner_dish = (
        (jnp.sum(obs[:, :, 11]) > 0)
        & (jnp.sum(obs[:, :, 16]) > 0)
        & (jnp.sum(obs[:, :, 17]) > 0)
    )
    holdoff = jnp.asarray(jnp.where(partner_dish, holdoff + 1, 0), jnp.int32)
    imminent = partner_dish & (holdoff <= _A1_HOLDOFF_LIMIT)

    accepts = pot_visible & pot_idle & (pot_total < 3) & (~imminent)

    # A partial pot holding something we would never match is a deadlock if the
    # partner is also waiting for someone else to move.  After watching it sit
    # still long enough, top it up regardless so it can cook and be emptied.
    obstructed = accepts & (pot_total > 0) & (recipe >= 0) & (pot_type != recipe)
    stall = jnp.asarray(
        jnp.where(pot_visible, jnp.where(obstructed, stall + 1, 0), stall), jnp.int32
    )
    forced = (stall >= _A1_STALL_LIMIT) & (forces < _A1_MAX_FORCED)

    # ---- carrying an ingredient -------------------------------------------
    deposit_ok = (
        carrying
        & accepts
        & (held_type == recipe)
        & ((pot_total == 0) | (pot_type == held_type) | forced)
    )
    prefer = jnp.where(held_type == 1, 1, 0)
    other = 1 - prefer
    prefer_free = jnp.where(prefer == 0, slot0 < 0, slot1 < 0)
    other_free = jnp.where(other == 0, slot0 < 0, slot1 < 0)
    stash = jnp.where(prefer_free, prefer, jnp.where(other_free, other, -1))
    # While a delivery is imminent, simply wait by the pot: the recipe is about
    # to be redrawn, so whether this unit is still wanted is not yet decided and
    # walking it to a cache slot would throw away eight steps either way.
    park = pot_visible & (~deposit_ok) & ((stash < 0) | imminent)
    go_stash = pot_visible & (~deposit_ok) & (stash >= 0) & (~imminent)
    carry_row = jnp.asarray(3, jnp.int32)
    carry_col = jnp.where(go_stash, jnp.where(stash == 1, 7, 3), 5)
    carry_face = jnp.where(go_stash, _DOWN, _UP)
    act_carry = _nav1(row, col, face, carry_row, carry_col, carry_face, ~park)

    # ---- empty handed ------------------------------------------------------
    want_fill = (
        accepts
        & (recipe >= 0)
        & ((pot_total == 0) | (pot_type == recipe) | forced)
    )
    # While the pot cooks there is nothing else to do, and the timer says
    # exactly how much slack is left, so top up an empty cache slot.  A full
    # cache turns the next post-delivery relay trip from thirteen steps into
    # about eight, which is the main throughput bottleneck.
    stage_ing = jnp.where(slot0 < 0, 0, jnp.where(slot1 < 0, 1, -1))
    can_stage = (
        pot_visible
        & (pot_vec[44] >= _A1_STAGE_MARGIN)
        & (stage_ing >= 0)
        & (~imminent)
    )
    continuing = fetching > 0
    # Only a staging trip is locked to its ingredient; a trip for the pot always
    # re-aims at the freshest recipe reading, so a resample mid-walk is free.
    mode_stage = jnp.where(continuing, staging > 0, (~want_fill) & can_stage)
    active = (continuing | want_fill | can_stage)
    target_ing = jnp.where(
        mode_stage, jnp.where(continuing, want, stage_ing), recipe
    )
    active = active & (target_ing >= 0)
    take_slot0 = slot0 == target_ing
    take_slot1 = slot1 == target_ing
    use_right = col >= 6
    pile_col = jnp.where(use_right, 9, 1)
    pile_face = jnp.where(
        target_ing == 0, jnp.where(use_right, _RIGHT, _LEFT), _DOWN
    )
    src_row = jnp.where(take_slot0 | take_slot1, 3, 4)
    src_col = jnp.where(take_slot0, 3, jnp.where(take_slot1, 7, pile_col))
    src_face = jnp.where(take_slot0 | take_slot1, _DOWN, pile_face)
    act_seek = jnp.where(
        active,
        _nav1(row, col, face, src_row, src_col, src_face, True),
        _nav1(row, col, face, 3, 5, _UP, False),
    )

    action = jnp.asarray(jnp.where(carrying, act_carry, act_seek), jnp.int32)

    forcing_now = (
        (action == _ACT_INT)
        & carrying
        & (row == 3)
        & (col == 5)
        & (face == _UP)
        & forced
        & (pot_total > 0)
        & (pot_type != held_type)
    )
    forces = jnp.asarray(jnp.where(forcing_now, forces + 1, forces), jnp.int32)
    fetching = jnp.where(carrying, 0, jnp.where(active, 1, 0))
    want = jnp.where(fetching > 0, target_ing, -1)
    staging = jnp.where(fetching > 0, mode_stage.astype(jnp.int32), 0)

    next_memory = jnp.stack(
        [recipe, slot0, slot1, fetching, want, stall, staging, holdoff, forces]
    ).astype(jnp.int32)
    trace = {
        "recipe": recipe,
        "row": jnp.asarray(row, jnp.int32),
        "col": jnp.asarray(col, jnp.int32),
        "held": jnp.asarray(jnp.where(carrying, held_type, -1), jnp.int32),
        "pot_total": jnp.asarray(jnp.where(pot_visible, pot_total, -1), jnp.int32),
        "stall": stall,
        "slot0": slot0,
        "slot1": slot1,
        "target": jnp.asarray(target_ing, jnp.int32),
    }
    return action, next_memory, trace
