"""Hand-written cooperative policy for Overcooked-V2 layout ``demo_cook_wide``.

Layout facts this policy is built on (all public, from the layout itself)::

    WWWWBXBWWWW
    WWW0 A 1WWW
    WWWWWPWWWWW
    W    A    W
    0  W3R3W  0
    W1WWWWWWW1W

* The two rooms are movement-disconnected.  Seat 0 only ever stands on
  ``(1,4)``, ``(1,5)`` or ``(1,6)``; it is the only seat that can reach the
  plate piles ``(0,4)/(0,6)`` and the delivery station ``(0,5)``, and it can
  never see the always-on recipe indicator ``(4,5)``.
* Seat 1 lives in the lower room, is the only seat that can read the recipe,
  and reaches the ingredient piles ``(4,0)/(4,10)/(5,1)/(5,9)``.
* Both seats reach the single pot ``(2,5)``: seat 0 from ``(1,5)`` facing down,
  seat 1 from ``(3,5)`` facing up.

Division of labour, justified only by locally observable evidence:

* Seat 1 reads the recipe and seeds the pot with one correct raw ingredient.
  That deposit is an ordinary productive action which also happens to be the
  only recipe evidence seat 0 can ever obtain - it is not a private code.
* Seat 0 reads the pot contents, tops the pot up with the *same* ingredient it
  already sees in the pot, fetches a plate while the pot cooks, plates the
  dish and delivers it.  It never guesses an ingredient on its own, because a
  guess is worth 0 in expectation (+20/-20) while waiting costs only time.
* Every branch is checked against what is visible now, so neither seat relies
  on the partner running this same code.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "evidence_split_cook",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 6,
    "max_width": 11,
    "num_ingredients": 4,
}

# action ids
_RIGHT = 0
_DOWN = 1
_LEFT = 2
_UP = 3
_STAY = 4
_INTERACT = 5

# action id -> facing id (observation uses UP=0 DOWN=1 RIGHT=2 LEFT=3)
_ACT2FACE = (2, 1, 3, 0)

# Static code of every column of map row 4 ("0  W3R3W  0"), padded with two
# out-of-map columns on each side.  1 = wall, 2 = recipe indicator,
# 4 = ingredient-0 pile, 8 = ingredient-3 pile.  Row 4 is visible from every
# cell of the lower room, and its 5-wide window is unique per column.
_ROW4_PAD = (0, 0, 4, 0, 0, 1, 8, 2, 8, 1, 0, 0, 4, 0, 0)

_INIT0 = (-1, 0, 0, -1, 0)
_INIT1 = (-1, 0, 0, 0, 1, 0, -1)


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------


def _face_of(action):
    return jnp.asarray(_ACT2FACE, dtype=jnp.int32)[action]


def _read(obs, pr, pc, gr, gc):
    """Feature vector of global cell (gr, gc) plus its visibility flag."""
    i = gr - pr + 2
    j = gc - pc + 2
    ok = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    vec = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4)]
    return jnp.where(ok, vec, jnp.zeros_like(vec)), ok


def _pot_state(vec):
    """(total, cooking, ready, idle, kind) for a pot feature vector."""
    counts = vec[34:38]
    total = jnp.sum(counts)
    timer = vec[44]
    cooked = vec[33]
    cooking = timer > 0
    ready = (cooked > 0) & (timer == 0)
    idle = jnp.logical_not(cooking | ready)
    best = jnp.argmax(counts)
    pure = (counts[best] == total) & (total > 0) & (best < 2)
    # -1 empty, 0/1 a single reachable ingredient, -2 mixed or unreachable
    kind = jnp.where(total == 0, -1, jnp.where(pure, best, -2)).astype(jnp.int32)
    return total, cooking, ready, idle, kind


def _inventory(obs):
    plate = obs[2, 2, 5]
    cooked = obs[2, 2, 6]
    counts = obs[2, 2, 7:11]
    total = jnp.sum(counts)
    kind = jnp.argmax(counts).astype(jnp.int32)
    dish = cooked > 0
    empty_plate = (plate > 0) & (cooked == 0)
    raw = (total > 0) & (cooked == 0)
    return counts, total, kind, dish, empty_plate, raw


def _partner(obs):
    """Partner held ingredient counts / dish flag, summed over the crop."""
    counts = jnp.sum(obs[:, :, 18:22], axis=(0, 1))
    total = jnp.sum(counts)
    kind = jnp.argmax(counts).astype(jnp.int32)
    dish = jnp.sum(obs[:, :, 17]) > 0
    pure_raw = ((counts[kind] == total) & (total > 0) & (kind < 2)
                & jnp.logical_not(dish))
    return kind, dish, pure_raw


# --------------------------------------------------------------------------
# seat 0 : upper room (three cells), completes the pot, plates and delivers
# --------------------------------------------------------------------------


def _station0(pc, facing, sc, face_act, do_interact):
    """Walk along row 1 to column ``sc``, turn to ``face_act``, then act."""
    step = jnp.where(sc > pc, _RIGHT, jnp.where(sc < pc, _LEFT, _STAY))
    faced = facing == _face_of(face_act)
    here = jnp.where(faced, jnp.where(do_interact, _INTERACT, _STAY), face_act)
    return jnp.where(sc == pc, here, step).astype(jnp.int32)


def init_agent_0():
    return jnp.asarray(_INIT0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, jnp.asarray(_INIT0, dtype=jnp.int32), memory)
    belief = mem[0]
    empty_ct = mem[1]
    stuck_ct = mem[2]
    par_seen = mem[3]
    hold_ct = mem[4]

    # localisation: the pot (2,5) is visible from all three reachable cells,
    # so its local column fixes our own column.
    pot_col = jnp.argmax(jnp.sum(obs[:, :, 24], axis=0))
    pc = (7 - pot_col).astype(jnp.int32)
    pr = jnp.asarray(1, dtype=jnp.int32)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    inv, inv_tot, inv_kind, hold_dish, hold_plate, hold_raw = _inventory(obs)
    pot_vec, _ = _read(obs, pr, pc, 2, 5)
    p_total, p_cooking, p_ready, p_idle, p_kind = _pot_state(pot_vec)
    par_kind, par_dish, par_pure = _partner(obs)

    counter_l, _ = _read(obs, pr, pc, 2, 4)
    counter_r, _ = _read(obs, pr, pc, 2, 6)
    free_l = jnp.sum(counter_l[32:38]) == 0
    free_r = jnp.sum(counter_r[32:38]) == 0

    empty_ct_n = jnp.where(p_total == 0, empty_ct + 1, 0)
    blocked = p_idle & (p_kind == -2) & (p_total > 0) & (p_total < 3)
    stuck_n = jnp.where(blocked, stuck_ct + 1, 0)
    force = stuck_n > 12

    # Recipe belief.  Pot contents are committed evidence.  A partner that only
    # carries an ingredient is used solely as a deadlock breaker: the pot must
    # have stayed untouched for a long stretch *and* the partner must have sat
    # in view holding that same ingredient, which rules out a partner merely in
    # transit with a stale item.
    par_same = par_pure & (par_kind == par_seen)
    hold_ct_n = jnp.where(par_pure, jnp.where(par_same, hold_ct + 1, 1), 0)
    par_seen_n = jnp.where(par_pure, par_kind, -1)
    fallback = (empty_ct_n > 40) & par_pure & (hold_ct_n > 12)
    belief_n = jnp.where(p_kind >= 0, p_kind, belief)
    belief_n = jnp.where((p_kind < 0) & fallback, par_kind, belief_n)

    need_known = ((p_kind >= 0) | force
                  | ((p_total == 0) & fallback & (belief_n >= 0)))
    need_kind = jnp.where(p_kind >= 0, p_kind,
                          jnp.where(force, 0, jnp.maximum(belief_n, 0)))

    dump_col = jnp.where(free_l & free_r, jnp.where(pc == 6, 6, 4),
                         jnp.where(free_l, 4, 6))
    plate_col = jnp.where(pc == 6, 6, 4)
    ing_col = jnp.where(need_kind == 1, 6, 4)
    ing_face = jnp.where(need_kind == 1, _RIGHT, _LEFT)

    a_deliver = _station0(pc, facing, 5, _UP, True)
    a_pot = _station0(pc, facing, 5, _DOWN, True)
    a_wait = _station0(pc, facing, 5, _DOWN, False)
    a_dump = _station0(pc, facing, dump_col, _DOWN, True)
    a_plate = _station0(pc, facing, plate_col, _UP, True)
    a_ing = _station0(pc, facing, ing_col, ing_face, True)

    hands_empty = jnp.logical_not(hold_dish | hold_plate | hold_raw)
    fillable = p_idle & (p_total < 3)

    action = a_wait
    action = jnp.where(hands_empty & fillable & need_known, a_ing, action)
    action = jnp.where(hands_empty & (p_cooking | p_ready), a_plate, action)

    deposit_ok = fillable & ((p_kind == inv_kind)
                             | ((p_total == 0) & (belief_n == inv_kind)
                                & (belief_n >= 0))
                             | (force & (p_kind == -2)))
    action = jnp.where(hold_raw, jnp.where(deposit_ok, a_pot, a_dump), action)
    action = jnp.where(
        hold_plate,
        jnp.where(p_ready, a_pot,
                  jnp.where(fillable & need_known, a_dump, a_wait)),
        action)

    # A dish is deliverable only if it is exactly three units of one reachable
    # ingredient; anything else is provably wrong, so park it on a counter
    # (0 reward) instead of delivering it (-20).
    dish_ok = (inv[inv_kind] == inv_tot) & (inv_tot == 3) & (inv_kind < 2)
    action = jnp.where(hold_dish, jnp.where(dish_ok, a_deliver, a_dump), action)

    delivered = hold_dish & dish_ok & (action == _INTERACT)
    belief_n = jnp.where(delivered, -1, belief_n)

    mem_n = jnp.stack([belief_n.astype(jnp.int32),
                       empty_ct_n.astype(jnp.int32),
                       stuck_n.astype(jnp.int32),
                       par_seen_n.astype(jnp.int32),
                       hold_ct_n.astype(jnp.int32)])
    trace = {
        "col": pc.astype(jnp.int32),
        "belief": belief_n.astype(jnp.int32),
        "pot_kind": p_kind.astype(jnp.int32),
        "pot_total": p_total.astype(jnp.int32),
    }
    return action.astype(jnp.int32), mem_n, trace


# --------------------------------------------------------------------------
# seat 1 : lower room, reads the recipe and seeds the pot
# --------------------------------------------------------------------------


def _locate1(obs):
    """Own cell in the lower room from the static crop alone."""
    # The cell above is a wall/pot in row 3 and plain floor in row 4.
    above_static = jnp.sum(obs[1, 2, 22:32])
    in_row3 = above_static > 0
    pr = jnp.where(in_row3, 3, 4).astype(jnp.int32)
    lrow = jnp.where(in_row3, 3, 2)
    band = obs[lrow]
    codes = (band[:, 22] + 2 * band[:, 25] + 4 * band[:, 28] + 8 * band[:, 31])
    table = jnp.asarray(_ROW4_PAD, dtype=codes.dtype)
    scores = jnp.stack([jnp.sum(table[c:c + 5] == codes) for c in range(1, 10)])
    pc = (jnp.argmax(scores) + 1).astype(jnp.int32)
    return pr, pc


def _nav1(pr, pc, tr, tc):
    """One step of a shortest path inside the lower room."""
    same_stub = ((pr == 4) & (tr == 4)
                 & (((pc <= 2) & (tc <= 2)) | ((pc >= 8) & (tc >= 8))))
    horiz = jnp.where(tc > pc, _RIGHT, jnp.where(tc < pc, _LEFT, _STAY))
    row3 = jnp.where(pc == tc, jnp.where(tr == 4, _DOWN, _STAY), horiz)
    return jnp.where(same_stub, horiz,
                     jnp.where(pr == 4, _UP, row3)).astype(jnp.int32)


def _dist1(pr, pc, tc):
    """Distance to the row-4 pile stand at column ``tc``."""
    same_stub = ((pr == 4)
                 & (((pc <= 2) & (tc <= 2)) | ((pc >= 8) & (tc >= 8))))
    base = jnp.abs(pc - tc)
    return jnp.where(pr == 3, base + 1, jnp.where(same_stub, base, base + 2))


def _station1(pr, pc, facing, sr, sc, face_act, do_interact):
    at = (pr == sr) & (pc == sc)
    faced = facing == _face_of(face_act)
    here = jnp.where(faced, jnp.where(do_interact, _INTERACT, _STAY), face_act)
    return jnp.where(at, here, _nav1(pr, pc, sr, sc)).astype(jnp.int32)


def init_agent_1():
    return jnp.asarray(_INIT1, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, jnp.asarray(_INIT1, dtype=jnp.int32), memory)
    belief = mem[0]
    slot0_mem = mem[1]
    slot1_mem = mem[2]
    guard_ct = mem[3]
    pot_mem = mem[4]
    block_ct = mem[5]

    pr, pc = _locate1(obs)
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    inv, inv_tot, inv_kind, hold_dish, hold_plate, hold_raw = _inventory(obs)

    rec_vec, _ = _read(obs, pr, pc, 4, 5)
    rec_counts = rec_vec[40:44]
    rec_sum = jnp.sum(rec_counts)
    rec_kind = jnp.argmax(rec_counts).astype(jnp.int32)
    belief_n = jnp.where(rec_sum > 0, rec_kind, belief)
    known = belief_n >= 0

    pot_vec, pot_vis = _read(obs, pr, pc, 2, 5)
    p_total, p_cooking, p_ready, p_idle, p_kind = _pot_state(pot_vec)

    want = jnp.maximum(belief_n, 0)
    other = 1 - want

    # Deadlock recovery: a partner may seed the pot with contents that can no
    # longer match the recipe.  A part-filled pot cannot be emptied by anyone,
    # so after a long stall fill it up regardless of type.  Topping it up with
    # a different ingredient is deliberate: the dish then cooks, is visibly
    # mixed - so a partner that checks its dish can bin it for 0 instead of
    # delivering it for -20 - and the pot is free for a correct dish again.
    matched = (p_kind >= 0) & (p_kind == want)
    mismatch = (pot_vis & p_idle & (p_total > 0) & (p_total < 3) & known
                & jnp.logical_not(matched))
    block_ct_n = jnp.where(mismatch, block_ct + 1,
                           jnp.where(pot_vis, 0, block_ct))
    clear_mode = block_ct_n > 30

    pot_free_now = (p_idle & (p_total < 3)
                    & ((p_total == 0) | (p_kind == want) | clear_mode))
    pot_free = jnp.where(pot_vis, pot_free_now, pot_mem > 0)
    pot_mem_n = jnp.where(pot_vis, pot_free_now.astype(jnp.int32), pot_mem)

    # Staging counters reachable from (3,3) and out of seat 0's reach:
    # ingredient 0 on (2,3) above, ingredient 1 on (4,3) below.
    slot0_vec, slot0_vis = _read(obs, pr, pc, 2, 3)
    slot1_vec, slot1_vis = _read(obs, pr, pc, 4, 3)
    slot0_now = (slot0_vec[34] > 0) & (jnp.sum(slot0_vec[32:38]) == slot0_vec[34])
    slot1_now = (slot1_vec[35] > 0) & (jnp.sum(slot1_vec[32:38]) == slot1_vec[35])
    slot0 = jnp.where(slot0_vis, slot0_now, slot0_mem > 0)
    slot1 = jnp.where(slot1_vis, slot1_now, slot1_mem > 0)
    slot0_n = jnp.where(slot0_vis, slot0_now.astype(jnp.int32), slot0_mem)
    slot1_n = jnp.where(slot1_vis, slot1_now.astype(jnp.int32), slot1_mem)
    have_want = jnp.where(want == 0, slot0, slot1)
    have_other = jnp.where(other == 0, slot0, slot1)

    # A partner carrying a cooked dish is about to deliver, which resamples the
    # recipe: seeding the pot now would very likely contaminate it.
    par_kind, par_dish, par_pure = _partner(obs)
    guard_ct_n = jnp.where(par_dish, guard_ct + 1, 0)
    guard = par_dish & (guard_ct_n < 25)

    spare_vec, _ = _read(obs, pr, pc, 2, 2)
    spare_free = jnp.sum(spare_vec[32:38]) == 0

    right_side = (_dist1(pr, pc, 9) + 7) < (_dist1(pr, pc, 1) + 3)
    pile_col = jnp.where(right_side, 9, 1)
    fetch_kind = jnp.where(pot_free | guard, want,
                           jnp.where(jnp.logical_not(have_other), other, want))
    pile_face = jnp.where(fetch_kind == 1, _DOWN,
                          jnp.where(right_side, _RIGHT, _LEFT))
    slot_face = jnp.asarray((_UP, _DOWN), dtype=jnp.int32)

    a_pot = _station1(pr, pc, facing, 3, 5, _UP, True)
    a_hub = _station1(pr, pc, facing, 3, 5, _UP, False)
    a_pile = _station1(pr, pc, facing, 4, pile_col, pile_face, True)
    a_take = _station1(pr, pc, facing, 3, 3, slot_face[want], True)
    a_put = _station1(pr, pc, facing, 3, 3,
                      slot_face[jnp.clip(inv_kind, 0, 1)], True)
    a_spare = _station1(pr, pc, facing, 3, 2, _UP, True)
    # nearest cell of row 3 from which the recipe indicator (4,5) is visible
    a_seek = _station1(pr, pc, facing, 3, jnp.clip(pc, 3, 7), _UP, False)

    deposit_ok = (pot_free & jnp.logical_not(guard) & known
                  & ((inv_kind == want) | (clear_mode & (inv_kind < 2))))
    slot_free_held = jnp.where(inv_kind == 0, jnp.logical_not(slot0),
                               jnp.logical_not(slot1))
    stash = jnp.where((inv_kind < 2) & slot_free_held, a_put,
                      jnp.where(spare_free, a_spare, a_hub))

    soon = pot_free | guard
    a_empty = jnp.where(soon,
                        jnp.where(have_want, a_take, a_pile),
                        jnp.where(jnp.logical_not(have_other), a_pile,
                                  jnp.where(have_want, a_take, a_pile)))

    keep = (inv_kind == want) & (inv_kind < 2)
    action = a_empty
    action = jnp.where(hold_raw,
                       jnp.where(deposit_ok, a_pot,
                                 jnp.where(keep, a_hub, stash)),
                       action)
    action = jnp.where(hold_plate | hold_dish,
                       jnp.where(spare_free, a_spare, a_hub), action)
    # Without a recipe reading nothing can be decided, so close in on the
    # always-on indicator's view radius first.
    action = jnp.where(known, action, a_seek)

    mem_n = jnp.stack([belief_n.astype(jnp.int32),
                       slot0_n.astype(jnp.int32),
                       slot1_n.astype(jnp.int32),
                       guard_ct_n.astype(jnp.int32),
                       pot_mem_n.astype(jnp.int32),
                       block_ct_n.astype(jnp.int32),
                       clear_mode.astype(jnp.int32)])
    trace = {
        "row": pr.astype(jnp.int32),
        "col": pc.astype(jnp.int32),
        "recipe": belief_n.astype(jnp.int32),
        "slots": jnp.stack([slot0_n, slot1_n]).astype(jnp.int32),
    }
    return action.astype(jnp.int32), mem_n, trace
