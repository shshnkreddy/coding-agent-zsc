"""Overcooked-V2 `demo_cook_wide` joint policy.

Layout facts that drive the design (see context/environment/README.md):

    WWWWBXBWWWW      row 0: plate piles (0,4)/(0,6), delivery (0,5)
    WWW0 A 1WWW      row 1: seat-0 room -> only (1,4),(1,5),(1,6)
    WWWWWPWWWWW      row 2: pot (2,5), the rest are counters
    W    A    W      row 3: seat-1 corridor
    0  W3R3W  0      row 4: recipe indicator (4,5), ingredient piles
    W1WWWWWWW1W

Seat 0 lives in a three-cell room.  From there it can reach both plate piles,
the delivery station, the pot, an ingredient-0 pile at (1,3) and an
ingredient-1 pile at (1,7), so it can execute a whole dish alone -- but row 4
is outside its 5x5 view, so it can *never* read the recipe indicator.

Seat 1 can read the recipe indicator and reach the pot from (3,5), but it can
reach neither a plate pile nor the delivery station.

The only information channel used here is physical and self-explanatory: seat 1
carries, and then deposits into the pot, a unit of the requested ingredient.
Seat 0 simply mirrors the ingredient type it can *see* -- first the pot's
contents, otherwise whatever raw ingredient the partner is holding.  No
arbitrary code, position or timing convention is involved, so the behaviour
degrades gracefully with an unknown partner.  If no evidence at all shows up
for a long while, seat 0 falls back to alternating its guess between the two
possible recipes, which is the best a blind cook can do (a wrong delivery
leaves the recipe unchanged, so the next opposite guess is certain to match).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'pot_signal_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# --------------------------------------------------------------------------
# action / facing encodings
# --------------------------------------------------------------------------
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# movement action that produces a given facing (indexed by facing id)
_FACE_TO_ACT = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# observation channel offsets
C_SELF_DIR = 1          # 1..4   up/down/right/left
C_SELF_PLATE = 5
C_SELF_COOKED = 6
C_SELF_ING = 7          # 7..10
C_PARTNER_ING = 18      # 18..21
C_STATIC = 22           # 22..31  (6 static objects + 4 ingredient piles)
C_ITEM_PLATE = 32
C_ITEM_COOKED = 33
C_ITEM_ING = 34         # 34..37
C_RECIPE_ING = 40       # 40..43
C_POT_TIMER = 44

# --------------------------------------------------------------------------
# static map constants (used for template-matching self localization)
# --------------------------------------------------------------------------
_GRID = (
    "WWWWBXBWWWW",
    "WWW0 A 1WWW",
    "WWWWWPWWWWW",
    "W    A    W",
    "0  W3R3W  0",
    "W1WWWWWWW1W",
)
_H, _W, _NST = 6, 11, 10
_OBJ_CH = {"W": 0, "X": 1, "P": 2, "R": 3, "L": 4, "B": 5}

# key cells
POT_R, POT_C = 2, 5
REC_R, REC_C = 4, 5
# seat-1 private parking counters (unreachable for seat 0)
S1_SLOT_C = (3, 7)      # (2,3) holds ingredient 0, (2,7) holds ingredient 1

# steps a blind seat 0 waits for any evidence before guessing
BLIND_AFTER = 35

_I32 = jnp.int32


def _static_tables():
    """Per-floor-cell 5x5 static templates plus their global coordinates.

    Every walkable cell of this layout has a distinct static crop, so a single
    template match recovers the seat's global position from its own view.
    """
    padded = [[[0] * _NST for _ in range(_W + 4)] for _ in range(_H + 4)]
    for r in range(_H):
        for c in range(_W):
            sym = _GRID[r][c]
            idx = _OBJ_CH.get(sym)
            if idx is None and sym.isdigit():
                idx = 6 + int(sym)
            if idx is not None:
                padded[r + 2][c + 2][idx] = 1
    floor = [(r, c) for r in range(_H) for c in range(_W) if _GRID[r][c] in " A"]
    crops = [[[padded[r + i][c + j] for j in range(5)] for i in range(5)]
             for (r, c) in floor]
    return floor, crops


def _localize(obs, top_room):
    """Return (row, col) of self from the static channels of the 5x5 crop."""
    floor, crops = _static_tables()
    allowed = [(p[0] == 1) == top_room for p in floor]
    templates = jnp.asarray(crops, dtype=_I32)
    penalty = jnp.asarray([0 if ok else 1000 for ok in allowed], dtype=_I32)
    rows = jnp.asarray([p[0] for p in floor], dtype=_I32)
    cols = jnp.asarray([p[1] for p in floor], dtype=_I32)
    static = obs[:, :, C_STATIC:C_STATIC + _NST].astype(_I32)
    cost = jnp.sum(jnp.abs(templates - static[None]), axis=(1, 2, 3)) + penalty
    idx = jnp.argmin(cost)
    return rows[idx], cols[idx]


def _cell(obs, sr, sc, r, c):
    """Channels of global cell (r, c); zeros (and visible=False) if off-view."""
    i = r - sr + 2
    j = c - sc + 2
    vis = (i >= 0) & (i <= 4) & (j >= 0) & (j <= 4)
    vec = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4)].astype(_I32)
    return vec * vis.astype(_I32), vis


def _turn_or_use(face, want, do_int):
    """Face `want`; once facing it, INTERACT (or STAY)."""
    aligned = face == want
    turn = jnp.asarray(_FACE_TO_ACT, dtype=_I32)[want]
    return jnp.where(
        aligned,
        jnp.where(do_int, _I32(A_INT), _I32(A_STAY)),
        turn,
    ).astype(_I32)


def _self_facing(obs):
    return jnp.argmax(obs[2, 2, C_SELF_DIR:C_SELF_DIR + 4]).astype(_I32)


# ==========================================================================
# seat 0 -- top room: fills the pot, plates and delivers
# ==========================================================================
# memory:
#   0 steps since this dish cycle began
#   1 ingredient last seen in the partner's hands this cycle (-1 = none)
#   2 parity of the blind alternating guess
#   3 whether a finished dish was held on the previous step
#   4 blind mode: the partner has not been stocking the pot
#   5 the partner has stocked the pot at least once this episode
#   6 we already put an ingredient in the pot during this cycle
#   7 how many unmatchable dishes we have written off at the hatch
_MEM0 = (0, -1, 0, 0, 0, 0, 0, 0)

# how many certain -20 deliveries we accept to keep the kitchen running
WRITE_OFF_LIMIT = 1

# how long seat 0 waits for the partner to reveal the recipe before guessing
WAIT_UNKNOWN = 50     # nothing known about the partner yet
WAIT_BLIND = 16       # the partner has repeatedly failed to stock the pot
WAIT_COOP = 70        # the partner has stocked the pot before, just slowly


def init_agent_0():
    return jnp.asarray(_MEM0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(_I32)
    mem = jnp.where(reset, init_agent_0(), memory)
    t = mem[0]
    hint = mem[1]
    parity = mem[2]
    prev_dish = mem[3]
    blind_mode = mem[4]
    coop = mem[5]
    stocked = mem[6]
    write_offs = mem[7]

    _, sc = _localize(obs, True)
    face = _self_facing(obs)

    plate = obs[2, 2, C_SELF_PLATE]
    cooked = obs[2, 2, C_SELF_COOKED]
    held = obs[2, 2, C_SELF_ING:C_SELF_ING + 4]
    has_dish = cooked > 0
    has_plate = (plate > 0) & (cooked == 0)
    has_ing = (jnp.sum(held) > 0) & (~has_dish) & (~has_plate)
    held_id = jnp.argmax(held).astype(_I32)

    # a dish that disappeared from our hands was handed to the delivery station
    delivered = (prev_dish > 0) & (~has_dish)
    t = jnp.where(delivered, _I32(0), t + 1)
    hint = jnp.where(delivered, _I32(-1), hint)
    parity = jnp.where(delivered, 1 - parity, parity)
    stocked = jnp.where(delivered, _I32(0), stocked)

    pot, _ = _cell(obs, 1, sc, POT_R, POT_C)
    pot_counts = pot[C_ITEM_ING:C_ITEM_ING + 4]
    pot_n = jnp.sum(pot_counts)
    pot_cooked = pot[C_ITEM_COOKED] > 0
    pot_timer = pot[C_POT_TIMER]
    pot_accepts = (pot_n < 3) & (pot_timer == 0) & (~pot_cooked)
    pot_busy = (pot_timer > 0) | pot_cooked

    # the two counters this seat can reach, (2,4) and (2,6)
    cnt4, _ = _cell(obs, 1, sc, 2, 4)
    cnt6, _ = _cell(obs, 1, sc, 2, 6)
    c4_used = jnp.sum(cnt4[C_ITEM_PLATE:C_ITEM_ING + 4]) > 0
    c6_used = jnp.sum(cnt6[C_ITEM_PLATE:C_ITEM_ING + 4]) > 0
    free_counter = (~c4_used) | (~c6_used)

    # Every recipe here is three of one ingredient, so a pot that mixes types
    # or holds ingredient 2/3 can never yield a match.  Cooking it out still
    # costs a full cycle, and is only worth doing when the resulting dish can
    # be set down rather than handed in for a certain -20.
    pot_spoiled = ((pot_counts[2] + pot_counts[3]) > 0) | (
        (pot_counts[0] > 0) & (pot_counts[1] > 0)
    )
    worth_clearing = free_counter | (write_offs < WRITE_OFF_LIMIT)
    pot_usable = (~pot_spoiled) | worth_clearing

    # partner-held raw ingredient (non-zero only on the partner's cell)
    p_ing = jnp.sum(obs[:, :, C_PARTNER_ING:C_PARTNER_ING + 4], axis=(0, 1))
    p_says = (p_ing[0] + p_ing[1]) > 0
    p_k = jnp.where(p_ing[0] >= p_ing[1], _I32(0), _I32(1))

    pot_says = (pot_counts[0] + pot_counts[1]) > 0
    pot_k = jnp.where(pot_counts[0] >= pot_counts[1], _I32(0), _I32(1))
    hint = jnp.where(p_says, p_k, hint)

    # What the partner put in the pot is the only *trustworthy* recipe report:
    # an ingredient merely held may be a leftover from an earlier recipe, and
    # copying such a stale item locks the team into repeating a wrong dish.
    partner_stocked = pot_says & (stocked == 0)
    coop = jnp.maximum(coop, partner_stocked.astype(_I32))
    blind_mode = jnp.where(partner_stocked, _I32(0), blind_mode)
    patience = jnp.where(
        blind_mode > 0, _I32(WAIT_BLIND),
        jnp.where(coop > 0, _I32(WAIT_COOP), _I32(WAIT_UNKNOWN)),
    )
    blind = (t >= patience) & (~pot_says)
    blind_mode = jnp.where(blind, _I32(1), blind_mode)

    # what we may commit to the pot, and what we may pick up in advance
    put_k = jnp.where(pot_says, pot_k, jnp.where(blind, parity, _I32(-1)))
    # once the partner is known not to stock the pot, carry the guess in hand
    # while still waiting, so the patience window costs almost no time.
    # Speculative pick-ups are only taken when a counter is free to undo them.
    early = jnp.where(blind_mode > 0, parity,
                      jnp.where(free_counter, hint, _I32(-1)))
    get_k = jnp.where(put_k >= 0, put_k, early)
    k = put_k

    # ---- choose a target column / facing / whether to interact -----------
    tcol = _I32(5)
    want = _I32(F_DOWN)
    do_int = jnp.asarray(False)

    # empty hands and the pot still wants ingredients
    # Reclaim a previously parked unit from either counter before walking to a
    # pile; leaving one stranded can eventually occupy both counters and leave
    # nowhere to put a wrong ingredient down.
    slot = C_ITEM_ING + jnp.clip(get_k, 0, 3)
    c4_has = cnt4[slot] > 0
    c6_has = cnt6[slot] > 0
    pile_col = jnp.where(get_k == 0, _I32(4), _I32(6))
    fetch_col = jnp.where(c4_has, _I32(4), jnp.where(c6_has, _I32(6), pile_col))
    fetch_face = jnp.where(
        c4_has | c6_has,
        _I32(F_DOWN),
        jnp.where(get_k == 0, _I32(F_LEFT), _I32(F_RIGHT)),
    )
    do_fetch = ((~has_ing) & (~has_plate) & (~has_dish) & pot_accepts
                & (get_k >= 0) & pot_usable)
    tcol = jnp.where(do_fetch, fetch_col, tcol)
    want = jnp.where(do_fetch, fetch_face, want)
    do_int = jnp.where(do_fetch, True, do_int)

    # empty hands while the pot cooks: fetch a plate and stand by
    do_plate = (~has_ing) & (~has_plate) & (~has_dish) & pot_busy & pot_usable
    tcol = jnp.where(do_plate, jnp.where(sc == 6, _I32(6), _I32(4)), tcol)
    want = jnp.where(do_plate, _I32(F_UP), want)
    do_int = jnp.where(do_plate, True, do_int)

    # holding a raw ingredient: commit it only when the target type is settled,
    # otherwise stand on the pot cell holding it, ready to drop it in at once
    park_pref = jnp.where(held_id == 0, _I32(4), _I32(6))
    pref_used = jnp.where(held_id == 0, c4_used, c6_used)
    other_used = jnp.where(held_id == 0, c6_used, c4_used)
    park_col = jnp.where(pref_used, 10 - park_pref, park_pref)
    can_park = (~pref_used) | (~other_used)
    fits = (put_k >= 0) & (held_id == put_k)
    wrong = (put_k >= 0) & (held_id != put_k)
    # with both counters occupied there is nowhere to put a wrong ingredient
    # down, so drop it in rather than stall the kitchen for the whole episode
    # into a pot that is already spoiled anything may go: that frees our hands
    # without tying up a counter we will need to set the doomed dish down on
    do_drop = has_ing & pot_accepts & pot_usable & (
        fits | pot_spoiled | (wrong & (~can_park))
    )
    do_park = has_ing & (~do_drop) & can_park & (
        (~pot_accepts) | wrong | (~pot_usable)
    )
    tcol = jnp.where(do_drop, _I32(5), tcol)
    want = jnp.where(do_drop, _I32(F_DOWN), want)
    do_int = jnp.where(do_drop, True, do_int)
    tcol = jnp.where(do_park, park_col, tcol)
    want = jnp.where(do_park, _I32(F_DOWN), want)
    do_int = jnp.where(do_park, True, do_int)

    # holding an empty plate
    plate_wait = has_plate & pot_busy
    tcol = jnp.where(plate_wait, _I32(5), tcol)
    want = jnp.where(plate_wait, _I32(F_DOWN), want)
    do_int = jnp.where(plate_wait, pot_cooked & pot_usable, do_int)
    # (defensive) a plate with nothing cooking would deadlock: put it down
    plate_stuck = has_plate & (~pot_busy)
    stuck_col = jnp.where(c4_used, _I32(6), _I32(4))
    tcol = jnp.where(plate_stuck, stuck_col, tcol)
    want = jnp.where(plate_stuck, _I32(F_DOWN), want)
    do_int = jnp.where(plate_stuck, (~c4_used) | (~c6_used), do_int)

    # holding the finished dish
    tcol = jnp.where(has_dish, _I32(5), tcol)
    want = jnp.where(has_dish, _I32(F_UP), want)
    do_int = jnp.where(has_dish, True, do_int)

    # A dish that mixes types, or holds ingredient 2 or 3, matches no recipe
    # now or after any resample, so handing it in is a certain -20.  Set it
    # down on a free counter instead; only when no counter is free do we write
    # one off at the hatch, and only a couple of times -- past that a partner
    # is spoiling every pot and standing still beats paying for it.  Anything
    # unexpected in the inventory encoding falls through to a normal delivery.
    kinds = jnp.sum((held > 0).astype(_I32))
    hopeless = has_dish & ((kinds > 1) | (held[2] > 0) | (held[3] > 0))
    ditch = hopeless & free_counter
    write_off = hopeless & (~free_counter) & (write_offs < WRITE_OFF_LIMIT)
    sit_out = hopeless & (~free_counter) & (write_offs >= WRITE_OFF_LIMIT)
    tcol = jnp.where(ditch, jnp.where(c4_used, _I32(6), _I32(4)), tcol)
    want = jnp.where(ditch, _I32(F_DOWN), want)
    do_int = jnp.where(ditch, True, do_int)
    tcol = jnp.where(sit_out, _I32(5), tcol)
    want = jnp.where(sit_out, _I32(F_DOWN), want)
    do_int = jnp.where(sit_out, False, do_int)

    arrived = sc == tcol
    action = jnp.where(
        arrived,
        _turn_or_use(face, want, do_int),
        jnp.where(sc < tcol, _I32(A_RIGHT), _I32(A_LEFT)),
    ).astype(_I32)

    stocked = jnp.maximum(stocked, (do_drop & (action == A_INT)).astype(_I32))
    write_offs = write_offs + (write_off & (action == A_INT)).astype(_I32)
    new_mem = jnp.stack([
        t, hint, parity, has_dish.astype(_I32), blind_mode, coop, stocked,
        write_offs,
    ]).astype(jnp.int32)
    trace = {
        "col": sc.astype(jnp.int32),
        "k": k.astype(jnp.int32),
        "pot_n": pot_n.astype(jnp.int32),
        "blind": blind_mode.astype(jnp.int32),
        "spoiled": pot_spoiled.astype(jnp.int32),
    }
    return action, new_mem, trace


# ==========================================================================
# seat 1 -- bottom corridor: reads the recipe, stocks and loads the pot
# ==========================================================================
# memory: [recipe_ingredient, pot_accepts_belief, pot_contents_belief,
#          slot0_state, slot1_state, steps_held_back_for_a_pending_delivery]
#   pot contents belief: -1 empty, otherwise the dominant ingredient id
#   slot state: 0 empty, 1 holds its own ingredient, 2 holds something else
_MEM1 = (-1, 1, -1, 0, 0, 0)

# how long to hold back an ingredient while the partner carries a finished dish
HOLD_FOR_DELIVERY = 25


def init_agent_1():
    return jnp.asarray(_MEM1, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(_I32)
    mem = jnp.where(reset, init_agent_1(), memory)
    k_mem = mem[0]
    pot_mem = mem[1]
    dom_mem = mem[2]
    slot0 = mem[3]
    slot1 = mem[4]
    dish_wait = mem[5]

    sr, sc = _localize(obs, False)
    face = _self_facing(obs)

    plate = obs[2, 2, C_SELF_PLATE]
    cooked = obs[2, 2, C_SELF_COOKED]
    held = obs[2, 2, C_SELF_ING:C_SELF_ING + 4]
    has_ing = jnp.sum(held) > 0
    held_id = jnp.argmax(held).astype(_I32)
    has_other = (plate > 0) | (cooked > 0)

    # recipe indicator at (4,5) is readable from (3,3)..(3,7)
    rec, rec_vis = _cell(obs, sr, sc, REC_R, REC_C)
    rec_counts = rec[C_RECIPE_ING:C_RECIPE_ING + 4]
    rec_ok = rec_vis & (jnp.sum(rec_counts) > 0)
    rec_k = jnp.where(rec_counts[0] >= rec_counts[1], _I32(0), _I32(1))
    k_mem = jnp.where(rec_ok, rec_k, k_mem)
    k = k_mem
    k_known = k >= 0
    k_safe = jnp.clip(k, 0, 1)

    pot, pot_vis = _cell(obs, sr, sc, POT_R, POT_C)
    pot_counts = pot[C_ITEM_ING:C_ITEM_ING + 4]
    pot_n = jnp.sum(pot_counts)
    pot_acc_obs = (pot_n < 3) & (pot[C_POT_TIMER] == 0) & (pot[C_ITEM_COOKED] == 0)
    pot_mem = jnp.where(pot_vis, pot_acc_obs.astype(_I32), pot_mem)
    dom_obs = jnp.where(pot_n > 0, jnp.argmax(pot_counts).astype(_I32), _I32(-1))
    dom_mem = jnp.where(pot_vis, dom_obs, dom_mem)
    # never mix: a pot already holding a different ingredient is a lost dish,
    # and adding to it only guarantees the failure instead of leaving the
    # partner's own (possibly deliberate) pure pot alone
    compatible = (dom_mem < 0) | (dom_mem == k_safe)
    # A delivery resamples the recipe, so the ingredient that decides an empty
    # pot's dish must not be chosen while the partner is still walking a
    # finished dish to the hatch: it would already be out of date.
    p_dish = jnp.sum(obs[:, :, 17]) > 0
    stall = p_dish & (dom_mem < 0) & (dish_wait < HOLD_FOR_DELIVERY)
    dish_wait = jnp.where(stall, dish_wait + 1, _I32(0))
    pot_open = (pot_mem > 0) & compatible      # the pot could take an ingredient
    pot_ready_now = pot_open & (~stall)        # ...and now is the moment to add

    # private parking counters (2,3) for ingredient 0 and (2,7) for ingredient 1
    s0, s0_vis = _cell(obs, sr, sc, 2, 3)
    s1, s1_vis = _cell(obs, sr, sc, 2, 7)
    s0_val = jnp.where(
        jnp.sum(s0[C_ITEM_PLATE:C_ITEM_ING + 4]) == 0,
        _I32(0),
        jnp.where(s0[C_ITEM_ING + 0] > 0, _I32(1), _I32(2)),
    )
    s1_val = jnp.where(
        jnp.sum(s1[C_ITEM_PLATE:C_ITEM_ING + 4]) == 0,
        _I32(0),
        jnp.where(s1[C_ITEM_ING + 1] > 0, _I32(1), _I32(2)),
    )
    slot0 = jnp.where(s0_vis, s0_val, slot0)
    slot1 = jnp.where(s1_vis, s1_val, slot1)

    slot_k = jnp.where(k_safe == 0, slot0, slot1)
    slot_held = jnp.where(held_id == 0, slot0, slot1)
    slot_col_k = jnp.where(k_safe == 0, _I32(S1_SLOT_C[0]), _I32(S1_SLOT_C[1]))
    slot_col_h = jnp.where(held_id == 0, _I32(S1_SLOT_C[0]), _I32(S1_SLOT_C[1]))

    # which depot to use: (4,1) on the left, (4,9) on the right
    use_left = sc <= 5
    depot_c = jnp.where(use_left, _I32(1), _I32(9))

    # ---- pick a target cell / facing / interaction ------------------------
    trow = _I32(3)
    tcol = _I32(5)
    want = _I32(F_UP)
    do_int = jnp.asarray(False)

    empty = (~has_ing) & (~has_other)

    # empty hands, pot busy: keep one unit of each ingredient parked nearby
    restock_type = jnp.where(slot0 == 0, _I32(0), _I32(1))
    need_restock = empty & (~pot_open) & ((slot0 == 0) | (slot1 == 0))
    # empty hands, pot hungry: take the parked unit, else walk to a depot
    take_parked = empty & pot_ready_now & k_known & (slot_k == 1)
    go_depot_k = empty & pot_ready_now & k_known & (slot_k != 1)

    fetch_type = jnp.where(need_restock, restock_type, k_safe)
    do_depot = need_restock | go_depot_k
    depot_face = jnp.where(
        fetch_type == 0,
        jnp.where(use_left, _I32(F_LEFT), _I32(F_RIGHT)),
        _I32(F_DOWN),
    )
    trow = jnp.where(do_depot, _I32(4), trow)
    tcol = jnp.where(do_depot, depot_c, tcol)
    want = jnp.where(do_depot, depot_face, want)
    do_int = jnp.where(do_depot, True, do_int)

    trow = jnp.where(take_parked, _I32(3), trow)
    tcol = jnp.where(take_parked, slot_col_k, tcol)
    want = jnp.where(take_parked, _I32(F_UP), want)
    do_int = jnp.where(take_parked, True, do_int)

    # carrying a raw ingredient
    wrong_type = k_known & (held_id != k_safe)
    do_park = has_ing & (wrong_type | (~pot_open)) & (slot_held == 0)
    do_load = has_ing & (~do_park) & pot_ready_now & (~wrong_type)
    trow = jnp.where(do_load, _I32(3), trow)
    tcol = jnp.where(do_load, _I32(5), tcol)
    want = jnp.where(do_load, _I32(F_UP), want)
    do_int = jnp.where(do_load, True, do_int)
    trow = jnp.where(do_park, _I32(3), trow)
    tcol = jnp.where(do_park, slot_col_h, tcol)
    want = jnp.where(do_park, _I32(F_UP), want)
    do_int = jnp.where(do_park, True, do_int)

    # never seen the recipe yet: walk to the middle where the indicator shows
    trow = jnp.where(~k_known, _I32(3), trow)
    tcol = jnp.where(~k_known, _I32(5), tcol)
    want = jnp.where(~k_known, _I32(F_UP), want)
    do_int = jnp.where(~k_known, False, do_int)

    # ---- navigate ---------------------------------------------------------
    row_move = jnp.where(sc < tcol, _I32(A_RIGHT), _I32(A_LEFT))
    to_row3 = jnp.where(sr == 4, _I32(A_UP), row_move)
    same_wing = ((sc <= 2) & (tcol <= 2)) | ((sc >= 8) & (tcol >= 8))
    to_row4 = jnp.where(
        sr == 4,
        jnp.where(same_wing, row_move, _I32(A_UP)),
        jnp.where(sc == tcol, _I32(A_DOWN), row_move),
    )
    nav = jnp.where(trow == 3, to_row3, to_row4)
    arrived = (sr == trow) & (sc == tcol)
    action = jnp.where(arrived, _turn_or_use(face, want, do_int), nav).astype(_I32)

    new_mem = jnp.stack(
        [k_mem, pot_mem, dom_mem, slot0, slot1, dish_wait]).astype(jnp.int32)
    trace = {
        "row": sr.astype(jnp.int32),
        "col": sc.astype(jnp.int32),
        "k": k.astype(jnp.int32),
        "pot_dom": dom_mem.astype(jnp.int32),
        "slots": (slot0 * 10 + slot1).astype(jnp.int32),
    }
    return action, new_mem, trace
