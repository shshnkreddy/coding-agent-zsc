"""Cooperative policy for Overcooked-V2 layout ``demo_cook_wide``.

Layout facts that drive the design (all taken from the environment README):

    WWWWBXBWWWW
    WWW0 A 1WWW
    WWWWWPWWWWW
    W    A    W
    0  W3R3W  0
    W1WWWWWWW1W

* Seat 0 lives in the three-cell top corridor (1,4)-(1,6).  It is the only
  agent that can reach the plate piles (0,4)/(0,6) and the delivery station
  (0,5), and it can reach an ingredient-0 supply at (1,3) and an ingredient-1
  supply at (1,7).  It can therefore cook and deliver on its own -- but the
  recipe indicator at (4,5) is permanently outside its 5x5 view, so it never
  sees the target recipe directly.
* Seat 1 lives in the bottom room.  From (3,5) it sees the recipe indicator,
  sees the pot, and can put ingredients into the pot from below.  It can never
  reach a plate or the delivery station.

So the only real coordination problem is telling the top agent which of the two
possible recipes ([0,0,0] or [1,1,1]) is currently requested.  This policy uses
*physical* evidence only, never a private code:

  * the bottom agent puts the correct raw ingredient into the shared pot, and
    carries it in plain sight of the top agent while doing so;
  * the top agent believes, in priority order, the pot contents, a raw
    ingredient left on the shared counters (2,4)/(2,6), and the ingredient its
    partner is visibly holding.  Those are exactly the signals any competent
    partner produces as a side effect of doing its job.

If no evidence at all arrives the top agent still cooks, guessing and flipping
its guess after every delivery: a wrong delivery does not resample the recipe,
so the flipped follow-up guess is then guaranteed correct.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'cook_wide_evidence',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# --------------------------------------------------------------------------
# Static map constants.  Module scope holds literals only; the JAX tables are
# materialised inside ``_tables`` so they are folded away at trace time.
# --------------------------------------------------------------------------

_H, _W = 6, 11

# Plane index of every board cell in observation channel order
# (wall, goal, pot, recipe, button, plate pile, ingredient supplies 0..3);
# -1 marks traversable floor.
_MAP_PLANES = (
    (0, 0, 0, 0, 5, 1, 5, 0, 0, 0, 0),
    (0, 0, 0, 6, -1, -1, -1, 7, 0, 0, 0),
    (0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
    (0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0),
    (6, -1, -1, 0, 9, 3, 9, 0, -1, -1, 6),
    (0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0),
)

# The sixteen reachable floor cells, in index order.
_FLOOR = ((1, 4), (1, 5), (1, 6), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
          (3, 6), (3, 7), (3, 8), (3, 9), (4, 1), (4, 2), (4, 8), (4, 9))
_NCELL = 16

# Movement actions: 0 RIGHT, 1 DOWN, 2 LEFT, 3 UP, 4 STAY, 5 INTERACT.
# _NEXT[src][goal] is the first movement action of a shortest path (4 when the
# goal is the current cell or lies in the other, unreachable room).
_NEXT = (
    (4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (2, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (4, 4, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
    (4, 4, 4, 2, 2, 4, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 4, 0, 0, 0, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 2, 4, 0, 0, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 4, 0, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 0, 0, 2, 2, 0, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 0, 2, 2, 1, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 1, 1),
    (4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0),
    (4, 4, 4, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 3, 3),
    (4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 4, 0),
    (4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 4),
)
_DIST = (
    (0, 1, 2, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99),
    (1, 0, 1, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99),
    (2, 1, 0, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 8, 9),
    (99, 99, 99, 1, 0, 1, 2, 3, 4, 5, 6, 7, 2, 1, 7, 8),
    (99, 99, 99, 2, 1, 0, 1, 2, 3, 4, 5, 6, 3, 2, 6, 7),
    (99, 99, 99, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 5, 6),
    (99, 99, 99, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 4, 5),
    (99, 99, 99, 5, 4, 3, 2, 1, 0, 1, 2, 3, 6, 5, 3, 4),
    (99, 99, 99, 6, 5, 4, 3, 2, 1, 0, 1, 2, 7, 6, 2, 3),
    (99, 99, 99, 7, 6, 5, 4, 3, 2, 1, 0, 1, 8, 7, 1, 2),
    (99, 99, 99, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 2, 1),
    (99, 99, 99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 9, 10),
    (99, 99, 99, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 0, 8, 9),
    (99, 99, 99, 8, 7, 6, 5, 4, 3, 2, 1, 2, 9, 8, 0, 1),
    (99, 99, 99, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 1, 0),
)

# Facing ids in the observation: UP=0, DOWN=1, RIGHT=2, LEFT=3.
_ACT_TO_FACE = (2, 1, 3, 0)

_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5

# Cell indices used by the two roles.
C14, C15, C16 = 0, 1, 2
C31, C33, C35, C37 = 3, 5, 7, 9
C41, C42, C48, C49 = 12, 13, 14, 15

# Padded-world coordinates (global + 2).
_POT_W = (2 + 2, 5 + 2)
_CNT_L_W = (2 + 2, 4 + 2)      # counter (2,4), shared with the bottom room
_CNT_R_W = (2 + 2, 6 + 2)      # counter (2,6)
_STASH0_W = (4 + 2, 3 + 2)     # counter (4,3): bottom agent's ingredient-0 cache
_STASH1_W = (4 + 2, 7 + 2)     # counter (4,7): bottom agent's ingredient-1 cache

_I32 = jnp.int32


def _sel(cond, first, second):
    return tuple(jnp.where(cond, a, b) for a, b in zip(first, second))


def _static_crops():
    """The expected 5x5 static-channel crop seen from each floor cell."""
    planes = jnp.asarray(_MAP_PLANES, dtype=_I32)
    onehot = (planes[:, :, None] == jnp.arange(10, dtype=_I32)[None, None, :])
    padded = jnp.pad(onehot.astype(_I32), ((2, 2), (2, 2), (0, 0)))
    return jnp.stack([padded[r:r + 5, c:c + 5] for (r, c) in _FLOOR])


def _decode(obs):
    """Locate ourselves on the static map and scatter the crop into a world."""
    static = obs[:, :, 22:32]
    mismatch = jnp.abs(_static_crops() - static[None, :, :, :]).sum(axis=(1, 2, 3))
    pos = jnp.argmin(mismatch).astype(_I32)
    floor_r = jnp.asarray([cell[0] for cell in _FLOOR], dtype=_I32)
    floor_c = jnp.asarray([cell[1] for cell in _FLOOR], dtype=_I32)
    row = floor_r[pos]
    col = floor_c[pos]
    world = lax.dynamic_update_slice(
        jnp.zeros((_H + 4, _W + 4, 45), dtype=_I32), obs.astype(_I32), (row, col, 0))
    seen = lax.dynamic_update_slice(
        jnp.zeros((_H + 4, _W + 4), dtype=_I32), jnp.ones((5, 5), dtype=_I32), (row, col))
    return pos, row, col, world, seen


def _item(world, cell):
    return world[cell[0], cell[1], 32:38]


def _raw_kind(block):
    """(is a single raw ingredient, its type) for a 6-wide item block."""
    counts = block[2:6]
    kinds = (counts > 0).sum()
    kind = jnp.argmax(counts).astype(_I32)
    raw = (block[0] == 0) & (block[1] == 0) & (kinds == 1)
    return raw, kind


def _act_for(pos, facing, stand, face_act, final_act):
    move = jnp.asarray(_NEXT, dtype=_I32)[pos, stand]
    want = jnp.asarray(_ACT_TO_FACE, dtype=_I32)[face_act]
    return jnp.where(pos != stand, move,
                     jnp.where(facing != want, face_act, final_act)).astype(_I32)


def _dist(pos, goal):
    return jnp.asarray(_DIST, dtype=_I32)[pos, goal]


# --------------------------------------------------------------------------
# Seat 0 - the top corridor: cook, plate and deliver.
# --------------------------------------------------------------------------

_M0 = 8  # t, belief, guess, deliver_pending, own_left, own_right, blind_at, taught

WAIT_FOR_EVIDENCE = 25   # steps we give the partner to reveal the first recipe
WAIT_IF_SILENT = 6       # once the partner proved silent, stop waiting


HOLD_LIMIT = 4           # how long to hold an ingredient on weak evidence alone
JUNK_LIMIT = 1           # how many un-dumpable spoiled dishes we will eat
# Measured A/B against a partner that keeps an ingredient in hand while the pot
# cooks and drops it in the instant the pot empties: treating those pre-delivery
# contents as suspect and cooking them off scored 70, taking them at face value
# scored 106.  A wrong delivery does not resample the recipe, so it is largely
# self-correcting -- the partner re-reads the unchanged recipe and the next
# batch is right -- whereas cooking off a pot burns a whole cycle for nothing.
TRUST_OLD_POT = False


def init_agent_0():
    return jnp.asarray([0, -1, 0, 0, -1, -1, WAIT_FOR_EVIDENCE, 0, 0, 0, 0, -1],
                       dtype=_I32)


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_0(), memory)
    pos, _row, col, world, _seen = _decode(obs)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(_I32)

    inv = obs[2, 2, 5:11].astype(_I32)
    inv_counts = inv[2:6]
    inv_total = inv_counts.sum()
    hands_empty = (inv[0] == 0) & (inv[1] == 0) & (inv_total == 0)
    has_dish = (inv[0] > 0) & (inv[1] > 0)
    has_plate = (inv[0] > 0) & (inv[1] == 0)
    has_ing = (inv[0] == 0) & (inv_total > 0)
    held_kind = jnp.argmax(inv_counts).astype(_I32)

    # --- a delivery resamples the recipe, so forget what we believed ---
    delivered = (mem[3] > 0) & hands_empty
    belief = jnp.where(delivered, -1, mem[1])
    guess = jnp.where(delivered, 1 - mem[2], mem[2])
    taught = jnp.where(delivered, 0, mem[7])
    step = jnp.where(delivered, 0, mem[0])
    # Stay patient with a partner that told us something last cycle; stop
    # waiting on one that did not.  (Waiting two silent cycles before giving up
    # measured worse against both the silent and the erratic mock partners.)
    blind_at = jnp.where(delivered,
                         jnp.where(mem[7] > 0, WAIT_FOR_EVIDENCE, WAIT_IF_SILENT),
                         mem[6])

    # --- pot ---
    pot = _item(world, _POT_W)
    pot_timer = world[_POT_W[0], _POT_W[1], 44]
    pot_counts = pot[2:6]
    pot_total = pot_counts.sum()
    pot_kinds = (pot_counts > 0).sum()
    pot_kind = jnp.argmax(pot_counts).astype(_I32)
    pot_ready = pot[1] > 0
    pot_open = (~pot_ready) & (pot_timer == 0) & (pot_total < 3)
    spoiled = (pot_kinds > 1) | ((pot_kinds == 1) & (pot_kind > 1))

    # Anything already in the pot when we delivered was chosen for the recipe
    # that our delivery has just replaced, so it is not evidence about the new
    # one.  A partner that keeps one ingredient in hand while the pot cooks
    # loads the pot in exactly this way.  The mark clears as soon as the
    # contents change again, which means somebody acted on the new recipe.
    mark = jnp.where(delivered, jnp.where(pot_total > 0, pot_total, -1), mem[11])
    pot_stale = (mark >= 0) & (pot_total > 0) & (pot_total == mark) & TRUST_OLD_POT
    mark = jnp.where(pot_stale, mark, -1)

    # --- shared counters: what is on them, and did we put it there ---
    left = _item(world, _CNT_L_W)
    right = _item(world, _CNT_R_W)
    left_raw, left_kind = _raw_kind(left)
    right_raw, right_kind = _raw_kind(right)
    left_free = (left.sum() == 0)
    right_free = (right.sum() == 0)
    own_left = jnp.where(left_raw & (left_kind == mem[4]), mem[4], -1)
    own_right = jnp.where(right_raw & (right_kind == mem[5]), mem[5], -1)

    # --- evidence about the current recipe ---
    ev_pot = (pot_kinds == 1) & (pot_kind < 2) & (~pot_stale)
    ev_l = left_raw & (left_kind < 2) & (left_kind != own_left)
    ev_r = right_raw & (right_kind < 2) & (right_kind != own_right)
    clash = ev_l & ev_r & (left_kind != right_kind)
    ev_cnt = (ev_l | ev_r) & (~clash)
    cnt_kind = jnp.where(ev_l, left_kind, right_kind)

    pinv = obs[:, :, 16:22].sum(axis=(0, 1)).astype(_I32)
    p_raw, p_kind = _raw_kind(pinv)
    ev_partner = p_raw & (p_kind < 2)

    # Left-over contents are a fair guess when nothing else is known, but they
    # must not outrank anything the partner does after the recipe changed.
    ev_stale = pot_stale & (pot_kinds == 1) & (pot_kind < 2)
    learned = ev_pot | ev_cnt | ev_partner
    target = jnp.where(ev_pot, pot_kind,
                       jnp.where(ev_cnt, cnt_kind,
                                 jnp.where(ev_partner, p_kind,
                                           jnp.where(belief >= 0, belief,
                                                     jnp.where(ev_stale, pot_kind,
                                                               belief)))))
    taught = jnp.where((belief < 0) & learned, 1, taught)
    blind = (target < 0) & (step >= blind_at)
    target = jnp.where(blind, guess, target)
    known = target >= 0

    # A partner seen carrying an ingredient is only weak evidence: it may be
    # restocking, not cooking.  Act on it, but do not commit the pot to it
    # until the pot itself confirms (or we have waited long enough that the
    # partner is plainly not going to add anything).
    # Counter items are only weak evidence too: an ingredient left there before
    # the last delivery says nothing about the recipe that replaced it.
    guessing = (jnp.where(delivered, 0, mem[8]) > 0) | blind
    strong = ev_pot | guessing
    waited = jnp.where(has_ing & (~strong) & (pot_total == 0), mem[9] + 1, 0)
    may_fill = strong | ((pot_total > 0) & (~pot_stale)) | (waited >= HOLD_LIMIT)
    # left-over contents that a fresh signal contradicts spoil the pot as surely
    # as a mixture does: they have to be cooked off before anything can be earned
    pot_dirty = spoiled | (pot_stale & (ev_cnt | ev_partner | guessing)
                           & known & (pot_kind != target))

    # --- candidate tasks ---
    near_plate = jnp.where(pos == C16, C16, C14)
    supply_stand = jnp.where(target == 1, C16, C14)
    supply_face = jnp.where(target == 1, _RIGHT, _LEFT)
    near_stand = jnp.where(pos == C16, C16, C14)
    near_face = jnp.where(pos == C16, _RIGHT, _LEFT)
    # prefer collecting a matching raw ingredient already sitting on a counter
    take_l = left_raw & (left_kind == target)
    take_r = right_raw & (right_kind == target)
    fetch_stand = jnp.where(take_l, C14, jnp.where(take_r, C16, supply_stand))
    fetch_face = jnp.where(take_l | take_r, _DOWN, supply_face)
    drop_stand = jnp.where(left_free, C14, C16)
    drop_ok = left_free | right_free

    # A finished dish was cooked on the evidence available at the time and the
    # recipe cannot have changed since (only a correct delivery resamples it).
    # So deliver any pure dish, and only hold it back when the pot itself --
    # the one strong signal -- now says the recipe is the other ingredient.
    dish_kinds = (inv_counts > 0).sum()
    pure_dish = (dish_kinds == 1) & (inv_total == 3) & (held_kind < 2)
    good_dish = pure_dish & ((~ev_pot) | (pot_kind == held_kind))
    # A spoiled pot (mixed, or holding an ingredient no recipe can use) can only
    # be cleared by cooking it and getting rid of the result.  Parking it on a
    # counter is free; delivering it costs 20, so do that at most once -- a
    # partner that keeps spoiling the pot would otherwise bleed the team dry.
    junk_left = mem[10] < JUNK_LIMIT
    # Parking a dish must never take the last counter: an ingredient left in
    # hand while the pot is busy can only be put down on a counter, so running
    # out of counters would strand us holding it with no way to fetch a plate.
    park_dish = left_free & right_free
    junk_ok = park_dish | junk_left

    # --- priority cascade (later assignments win) ---
    stand, face, fin = C15, jnp.asarray(_DOWN, _I32), jnp.asarray(_STAY, _I32)

    # pot has room: fill it
    fill_clean = pot_open & (~pot_dirty)
    stand, face, fin = _sel(fill_clean & known & hands_empty,
                            (fetch_stand, fetch_face, jnp.asarray(_INTERACT, _I32)),
                            (stand, face, fin))
    stand, face, fin = _sel(fill_clean & known & has_ing & (held_kind == target)
                            & may_fill,
                            (C15, _DOWN, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(fill_clean & known & has_ing & (held_kind == target)
                            & (~may_fill),
                            (C15, _DOWN, _STAY), (stand, face, fin))
    stand, face, fin = _sel(fill_clean & has_ing & ((held_kind != target) | (~known))
                            & drop_ok,
                            (drop_stand, _DOWN, _INTERACT), (stand, face, fin))
    # nowhere to park it: better a flushable pot than a permanently full hand
    stand, face, fin = _sel(fill_clean & has_ing & ((held_kind != target) | (~known))
                            & (~drop_ok),
                            (C15, _DOWN, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(fill_clean & has_plate & drop_ok,
                            (drop_stand, _DOWN, _INTERACT), (stand, face, fin))
    # pot has room but holds something that cannot match any recipe: flush it
    stand, face, fin = _sel(pot_open & pot_dirty & junk_ok & hands_empty,
                            (near_stand, near_face, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(pot_open & pot_dirty & junk_ok & has_ing,
                            (C15, _DOWN, _INTERACT), (stand, face, fin))

    # pot busy: get a plate and stand ready
    busy = (~pot_open) & (~pot_ready)
    stand, face, fin = _sel(busy & hands_empty,
                            (near_plate, _UP, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(busy & has_plate, (C15, _DOWN, _STAY), (stand, face, fin))
    stand, face, fin = _sel(busy & has_ing & drop_ok,
                            (drop_stand, _DOWN, _INTERACT), (stand, face, fin))

    # pot ready: plate it, unless its contents are spoiled beyond our means
    collect = pot_ready & ((~pot_dirty) | junk_ok)
    stand, face, fin = _sel(collect & hands_empty,
                            (near_plate, _UP, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(collect & has_plate, (C15, _DOWN, _INTERACT),
                            (stand, face, fin))
    stand, face, fin = _sel(pot_ready & has_ing & drop_ok,
                            (drop_stand, _DOWN, _INTERACT), (stand, face, fin))

    # holding a finished dish
    keep_dish = has_dish & (~good_dish) & (~park_dish) & (~pure_dish) & (~junk_left)
    stand, face, fin = _sel(has_dish & (~good_dish) & park_dish,
                            (drop_stand, _DOWN, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(has_dish & (good_dish | (~park_dish)) & (~keep_dish),
                            (C15, _UP, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(keep_dish, (C15, _DOWN, _STAY), (stand, face, fin))

    action = _act_for(pos, facing, stand, face, fin)

    delivering = has_dish & (action == _INTERACT) & (stand == C15) & (face == _UP)
    putting = (action == _INTERACT) & (stand != C15) & (face == _DOWN) & has_ing
    new_left = jnp.where(putting & (stand == C14), held_kind, own_left)
    new_right = jnp.where(putting & (stand == C16), held_kind, own_right)
    taking = (action == _INTERACT) & (face == _DOWN) & hands_empty
    new_left = jnp.where(taking & (stand == C14), -1, new_left)
    new_right = jnp.where(taking & (stand == C16), -1, new_right)

    new_mem = jnp.stack([
        step + 1,
        target,
        guess,
        delivering.astype(_I32),
        new_left,
        new_right,
        blind_at,
        taught,
        guessing.astype(_I32),
        waited,
        mem[10] + (delivering & (~pure_dish)).astype(_I32),
        mark,
    ]).astype(_I32)
    hold = jnp.where(has_dish, 2, jnp.where(has_plate, 1,
                     jnp.where(has_ing, 3 + held_kind, 0))).astype(_I32)
    trace = {
        'pos': pos,
        'target': target,
        'pot_total': pot_total.astype(_I32),
        'action': action,
        'blind': blind.astype(_I32),
        'hold': hold,
        'pot_kind': jnp.where(pot_kinds == 1, pot_kind, -1).astype(_I32),
        'stand': stand.astype(_I32),
    }
    return action, new_mem, trace


# --------------------------------------------------------------------------
# Seat 1 - the bottom room: read the recipe and feed the pot.
# --------------------------------------------------------------------------

_M1 = 7  # t, belief, stash0, stash1, stall, pot_open, pot_kind

STALL_LIMIT = 45


def init_agent_1():
    return jnp.asarray([0, -1, 0, 0, 0, 1, -1], dtype=_I32)


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_1(), memory)
    pos, _row, _col, world, seen = _decode(obs)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(_I32)

    inv = obs[2, 2, 5:11].astype(_I32)
    inv_counts = inv[2:6]
    inv_total = inv_counts.sum()
    hands_empty = (inv[0] == 0) & (inv[1] == 0) & (inv_total == 0)
    has_ing = (inv[0] == 0) & (inv_total > 0) & ((inv_counts > 0).sum() == 1)
    junk = (~hands_empty) & (~(has_ing & (jnp.argmax(inv_counts) < 2)))
    held_kind = jnp.argmax(inv_counts).astype(_I32)

    # the always-on indicator is readable from (3,3)..(3,7)
    recipe = obs[:, :, 40:44].sum(axis=(0, 1)).astype(_I32)
    recipe_seen = recipe.sum() > 0
    target = jnp.where(recipe_seen, jnp.argmax(recipe).astype(_I32), mem[1])
    known = target >= 0

    pot_vis = seen[_POT_W[0], _POT_W[1]] > 0
    pot = _item(world, _POT_W)
    pot_timer = world[_POT_W[0], _POT_W[1], 44]
    pot_counts = pot[2:6]
    pot_total = pot_counts.sum()
    pot_kinds = (pot_counts > 0).sum()
    pot_kind = jnp.argmax(pot_counts).astype(_I32)
    # The pot leaves our view on every supply run, so remember its state:
    # guessing that an unseen pot is free makes the agent flip-flop on the way.
    seen_open = (pot[1] == 0) & (pot_timer == 0) & (pot_total < 3)
    seen_kind = jnp.where(pot_kinds == 0, -1, jnp.where(pot_kinds == 1, pot_kind, -2))
    pot_open = jnp.where(pot_vis, seen_open, mem[5] > 0)
    held_pot_kind = jnp.where(pot_vis, seen_kind, mem[6])
    pot_matches = (held_pot_kind == -1) | (held_pot_kind == target)

    # our two private caches, remembered when they are out of sight
    s0_raw, s0_kind = _raw_kind(_item(world, _STASH0_W))
    s1_raw, s1_kind = _raw_kind(_item(world, _STASH1_W))
    s0 = jnp.where(seen[_STASH0_W[0], _STASH0_W[1]] > 0,
                   (s0_raw & (s0_kind == 0)).astype(_I32), mem[2])
    s1 = jnp.where(seen[_STASH1_W[0], _STASH1_W[1]] > 0,
                   (s1_raw & (s1_kind == 1)).astype(_I32), mem[3])
    have_t = jnp.where(target == 1, s1, s0) > 0
    have_o = jnp.where(target == 1, s0, s1) > 0
    # Caches are collected from row 3, where the top agent can see us pick the
    # requested ingredient up -- that sighting is a true signal.  They are
    # *stocked* from row 4 instead, which the top agent can never see, so that
    # carrying the other ingredient around never looks like a signal.
    stash_stand = jnp.where(target == 1, C37, C33)
    held_put_stand = jnp.where(held_kind == 1, C48, C42)
    held_put_face = jnp.where(held_kind == 1, _LEFT, _RIGHT)
    held_free = jnp.where(held_kind == 1, s1, s0) == 0

    # a pot left holding the wrong ingredient is worth unblocking eventually
    blocked = (~pot_open) | (~pot_matches)
    stall = jnp.where(pot_open & (~pot_matches) & has_ing & (held_kind == target),
                      mem[4] + 1, 0)
    unblock = stall > STALL_LIMIT

    # Fetching for immediate use: nearest supply, carried past the top agent in
    # plain sight.  Fetching to restock a cache: the supply on that cache's own
    # side, so the whole trip stays in the bottom row and out of sight.
    left_near = _dist(pos, C41) <= _dist(pos, C49)
    depot_stand = jnp.where(left_near, C41, C49)
    depot_face = jnp.where(target == 1, _DOWN,
                           jnp.where(left_near, _LEFT, _RIGHT))
    other = 1 - target
    restock_t_stand = jnp.where(target == 1, C49, C41)
    restock_t_face = jnp.where(target == 1, _DOWN, _LEFT)
    restock_o_stand = jnp.where(other == 1, C49, C41)
    restock_o_face = jnp.where(other == 1, _DOWN, _LEFT)

    # idle: stand at the pot; if the recipe has never been seen, go and read it
    look = jnp.where(_dist(pos, C33) <= _dist(pos, C37), C33, C37)
    stand = jnp.where(known, C35, look)
    face = jnp.asarray(_UP, _I32)
    fin = jnp.asarray(_STAY, _I32)

    # empty hands: acquire what the kitchen needs next
    need_now = pot_open & pot_matches & known
    stand, face, fin = _sel(hands_empty & need_now & have_t,
                            (stash_stand, _DOWN, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(hands_empty & need_now & (~have_t),
                            (depot_stand, depot_face, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(hands_empty & (~need_now) & known & (~have_t),
                            (restock_t_stand, restock_t_face, _INTERACT),
                            (stand, face, fin))
    stand, face, fin = _sel(hands_empty & (~need_now) & known & have_t & (~have_o),
                            (restock_o_stand, restock_o_face, _INTERACT),
                            (stand, face, fin))

    # carrying the wanted ingredient
    stand, face, fin = _sel(has_ing & (held_kind == target) & known
                            & ((pot_open & pot_matches) | unblock),
                            (C35, _UP, _INTERACT), (stand, face, fin))
    stand, face, fin = _sel(has_ing & (held_kind == target) & known & blocked
                            & (~unblock) & held_free,
                            (held_put_stand, held_put_face, _INTERACT),
                            (stand, face, fin))
    # carrying the other ingredient: park it in its own cache
    stand, face, fin = _sel(has_ing & (held_kind < 2) & ((held_kind != target) | (~known))
                            & held_free,
                            (held_put_stand, held_put_face, _INTERACT),
                            (stand, face, fin))
    # never expected: dump anything else out of the way
    stand, face, fin = _sel(junk, (C31, _UP, _INTERACT), (stand, face, fin))

    action = _act_for(pos, facing, stand, face, fin)

    new_mem = jnp.stack([mem[0] + 1, target, s0, s1, stall,
                         pot_open.astype(_I32), held_pot_kind]).astype(_I32)
    hold = jnp.where(hands_empty, 0, jnp.where(has_ing, 3 + held_kind, 9)).astype(_I32)
    trace = {
        'pos': pos,
        'target': target,
        'pot_total': pot_total.astype(_I32),
        'action': action,
        'stash': (s0 + 2 * s1).astype(_I32),
        'hold': hold,
        'stand': stand.astype(_I32),
    }
    return action, new_mem, trace
