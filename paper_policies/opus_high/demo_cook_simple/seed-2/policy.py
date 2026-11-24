"""Cooperative policy for the Overcooked-V2 layout ``demo_cook_simple``.

Layout facts that drive the design (all taken from the environment spec):

    WWWWWR2W0WW
    0      W  B
    W     APA X
    1      W  B
    WWWWWR2W1WW

* The two walkable regions -- left ``rows 1..3 x cols 1..6`` and right
  ``rows 1..3 x cols 8..9`` -- are disconnected, so each seat is alone in its
  own rectangular room and the seats can never collide or swap jobs.
* Only the left room can see the always-on recipe indicators at (0,5)/(4,5).
* Only the right room can reach the plate piles (1,10)/(3,10) and the delivery
  station (2,10).
* The pot (2,7) is the single shared object: reachable from (2,6) facing right
  and from (2,8) facing left, and visible from every cell of the right room.

The roles are therefore forced by geometry rather than by a private
convention: the left seat is the only possible source of recipe information
and the right seat is the only possible deliverer.  The left seat publishes the
recipe by putting a matching raw ingredient into the shared pot; the right seat
simply mirrors whatever is already in the pot, fills it up, plates it and
delivers.  That signal is ordinary game evidence, so it stays meaningful with
an independently generated partner.

Because a partner may publish the recipe differently, the right seat reads, in
order of trust: raw contents of the pot, a raw ingredient that freshly appeared
on one of the shared counters (1,7)/(3,7), and -- only after the pot has stood
empty for a while -- a raw ingredient the partner is visibly carrying.  It
never guesses: a wrong guess spoils the single pot and costs a delivery.  It
also picks up a plated dish or plate left on a counter, and finishes and
delivers a pot it did not fill, so a partner that cooks differently still gets
its dish out.  Symmetrically the left seat keeps supplying and, if the pot has
been spoiled by anyone, fills it up so that it can be emptied again.

Everything else is bookkeeping: exact self-localisation from the static
observation channels, a remembered recipe for the trip to the far supplies, and
a private per-ingredient stash on the left-room counters (never on the shared
ones, where a stale ingredient could mislead a partner).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "room_split_cook",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 11,
    "num_ingredients": 3,
}

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# facing ids are UP=0, DOWN=1, RIGHT=2, LEFT=3, indexed by the move action id
_FACE_OF_ACTION = (2, 1, 3, 0)

# static-object code per cell, matching observation channels 20..28 in order
# (1 wall, 2 goal, 3 pot, 4 recipe indicator, 5 button, 6 plate pile,
#  7 ingredient-0 supply, 8 ingredient-1 supply, 9 ingredient-2 supply).
_MAP = (
    (1, 1, 1, 1, 1, 4, 9, 1, 7, 1, 1),
    (7, 0, 0, 0, 0, 0, 0, 1, 0, 0, 6),
    (1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2),
    (8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 6),
    (1, 1, 1, 1, 1, 4, 9, 1, 8, 1, 1),
)
_HEIGHT, _WIDTH = 5, 11
_VIEW = 2

_LEFT_CELLS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
)
_RIGHT_CELLS = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))


def _crops(cells):
    """Static-code 5x5 crop of every candidate cell (zero padding off-map)."""
    pad = [[0] * (_WIDTH + 2 * _VIEW) for _ in range(_HEIGHT + 2 * _VIEW)]
    for r in range(_HEIGHT):
        for c in range(_WIDTH):
            pad[r + _VIEW][c + _VIEW] = _MAP[r][c]
    out = []
    for (r, c) in cells:
        out.append([[pad[r + i][c + j] for j in range(5)] for i in range(5)])
    return jnp.asarray(out, dtype=jnp.int32), jnp.asarray(cells, dtype=jnp.int32)


# Station = (stand row, stand col, facing action towards the object).
_POT_L = (2, 6, RIGHT)          # left-side pot access
_POT_R = (2, 8, LEFT)           # right-side pot access
_SUPPLY_L = ((1, 1, LEFT), (3, 1, LEFT), (1, 6, UP))   # ingredients 0,1,2 (left)
_SUPPLY_R = ((1, 8, UP), (3, 8, DOWN), (1, 8, UP))     # ingredients 0,1 (right)
_SHARE_L = ((1, 6, RIGHT), (3, 6, RIGHT))  # counters (1,7) / (3,7) from the left
# Left-room parking counters, none of them visible from the other room.  The
# first two act as a per-ingredient stash: an ingredient that the resampled
# recipe no longer wants is left there and picked up again when it comes back.
_CNT_L = (
    (0, 4, 1, 4, UP),
    (4, 4, 3, 4, DOWN),
    (0, 3, 1, 3, UP),
    (4, 3, 3, 3, DOWN),
)
_DELIVER = (2, 9, RIGHT)
_PLATE_R = ((1, 9, RIGHT), (3, 9, RIGHT))
_SEEK_L = (2, 5, RIGHT)         # a cell from which both recipe indicators show

# right-room counters: (cell row, cell col, stand row, stand col, facing)
_COUNTERS_R = (
    (0, 9, 1, 9, UP),
    (4, 9, 3, 9, DOWN),
    (1, 7, 1, 8, LEFT),
    (3, 7, 3, 8, LEFT),
)
_SHARED_IDX = (2, 3)

_EMPTY_WAIT = 10     # fallback wait before trusting an empty pot after a scoop
_DISH_WAIT = 25      # give up on a partner that carries a dish around forever
_HINT_WAIT = 10      # empty-pot steps before a partner-held ingredient is a hint
_RECOVER_UNTIL = 330  # last step at which clearing a spoiled pot still pays


# --------------------------------------------------------------------------
# observation helpers
# --------------------------------------------------------------------------


def _static_code(obs):
    weights = jnp.arange(1, 10, dtype=jnp.int32)
    return (obs[:, :, 20:29] * weights).sum(axis=-1)


def _localize(obs, cells):
    crops, table = _crops(cells)
    code = _static_code(obs)
    scores = (crops == code[None, :, :]).sum(axis=(1, 2))
    rc = table[jnp.argmax(scores)]
    return rc[0], rc[1]


def _read_cell(obs, r, c, gr, gc):
    """Feature vector of global cell (gr, gc) plus a visibility flag."""
    lr = gr - r + _VIEW
    lc = gc - c + _VIEW
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    vec = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)]
    return vec, visible


def _go(r, c, facing, station):
    """Walk to the station cell, turn to it, then INTERACT (rooms are convex)."""
    tr, tc, face_action = station
    want = _FACE_OF_ACTION[face_action]
    move = jnp.where(c < tc, RIGHT, jnp.where(c > tc, LEFT, jnp.where(r < tr, DOWN, UP)))
    arrived = (r == tr) & (c == tc)
    turn_or_act = jnp.where(facing == want, INTERACT, face_action)
    return jnp.asarray(jnp.where(arrived, turn_or_act, move), dtype=jnp.int32)


def _go_wait(r, c, facing, station):
    """Same as :func:`_go` but idles instead of interacting once in place."""
    action = _go(r, c, facing, station)
    return jnp.where(action == INTERACT, STAY, action).astype(jnp.int32)


def _self_facts(obs):
    facing = jnp.argmax(obs[_VIEW, _VIEW, 1:5]).astype(jnp.int32)
    held_plate = obs[_VIEW, _VIEW, 5] > 0
    held_cooked = obs[_VIEW, _VIEW, 6] > 0
    held_counts = obs[_VIEW, _VIEW, 7:10]
    has_ing = held_counts.sum() > 0
    held_ing = jnp.argmax(held_counts).astype(jnp.int32)
    return facing, held_plate, held_cooked, has_ing, held_ing


def _pot_facts(obs, r, c, remembered):
    vec, visible = _read_cell(obs, r, c, 2, 7)
    fresh = jnp.concatenate([vec[31:34], vec[30:31], vec[39:40]]).astype(jnp.int32)
    state = jnp.where(visible, fresh, remembered)
    counts = state[0:3]
    return state, counts, state[3], state[4]


def _counters(obs, r, c, stations, mem_busy, mem_kind):
    """Belief about nearby counters: busy flag and raw-ingredient id (-1: none).

    Cells that are out of view keep their remembered value; every counter is
    visible from the cell we stand on to use it, so a stale belief can only
    cost a walk, never a loop.
    """
    busy, kind, seen, dish, plate = [], [], [], [], []
    for (gr, gc, _sr, _sc, _face) in stations:
        vec, visible = _read_cell(obs, r, c, gr, gc)
        raw = (vec[29] == 0) & (vec[30] == 0) & (vec[31:34].sum() > 0)
        busy.append((vec[29:34].sum() > 0).astype(jnp.int32))
        kind.append(jnp.where(raw, jnp.argmax(vec[31:34]), -1).astype(jnp.int32))
        dish.append(visible & (vec[29] > 0) & (vec[30] > 0))
        plate.append(visible & (vec[29] > 0) & (vec[30] == 0) & (vec[31:34].sum() == 0))
        seen.append(visible)
    seen = jnp.stack(seen)
    busy = jnp.where(seen, jnp.stack(busy), mem_busy)
    kind = jnp.where(seen, jnp.stack(kind), mem_kind)
    return busy, kind, seen, jnp.stack(dish), jnp.stack(plate)


def _partner_facts(obs):
    """What the partner carries, when the partner is inside our crop."""
    here = obs[:, :, 10]
    visible = here.sum() > 0
    inventory = (obs[:, :, 15:20] * here[:, :, None]).sum(axis=(0, 1))
    raw = visible & (inventory[0] == 0) & (inventory[1] == 0) & (inventory[2:5].sum() > 0)
    kind = jnp.argmax(inventory[2:5]).astype(jnp.int32)
    dish = visible & (inventory[1] > 0)
    return visible, raw, kind, dish


def _pick_station(rank, stations, r, c, facing):
    """Action towards the highest-ranked station (ties: lowest index)."""
    actions = jnp.stack(
        [_go(r, c, facing, (sr, sc, face)) for (_gr, _gc, sr, sc, face) in stations]
    )
    index = jnp.argmax(rank)
    return actions[index], index, rank[index]


def _select(actions, flags, fallback):
    """First action whose flag is set, else ``fallback`` (fixed-shape)."""
    out = jnp.asarray(fallback, dtype=jnp.int32)
    for action, flag in zip(reversed(actions), reversed(flags)):
        out = jnp.where(flag, jnp.asarray(action, dtype=jnp.int32), out)
    return out


def _reset_memory(memory, init, reset):
    return jax.tree_util.tree_map(
        lambda old, start: jnp.where(reset, start, old), memory, init
    )


# --------------------------------------------------------------------------
# agent 0 -- left room: reads the recipe, supplies the pot
# --------------------------------------------------------------------------


def init_agent_0():
    return {
        "step": jnp.asarray(0, dtype=jnp.int32),
        "pot": jnp.zeros((5,), dtype=jnp.int32),
        "target": jnp.asarray(-1, dtype=jnp.int32),
        "mark": jnp.asarray(-1, dtype=jnp.int32),
        "ticks": jnp.asarray(99, dtype=jnp.int32),
        "busy": jnp.zeros((4,), dtype=jnp.int32),
        "kind": -jnp.ones((4,), dtype=jnp.int32),
        "helper": jnp.asarray(0, dtype=jnp.int32),
        "fed": jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = _reset_memory(memory, init_agent_0(), reset)

    r, c = _localize(obs, _LEFT_CELLS)
    facing, held_plate, held_cooked, has_ing, held_ing = _self_facts(obs)

    # --- recipe, from the always-on indicators (or from memory while away) ---
    code = _static_code(obs)
    shown = ((code == 4) | (code == 5))[:, :, None] * obs[:, :, 36:39]
    shown = shown.sum(axis=(0, 1))
    recipe_visible = shown.sum() > 0
    target = jnp.where(recipe_visible, jnp.argmax(shown).astype(jnp.int32), mem["target"])

    # --- pot ---
    pot_state, pot_counts, pot_cooked, pot_timer = _pot_facts(obs, r, c, mem["pot"])
    pot_n = pot_counts.sum()
    pot_major = jnp.argmax(pot_counts).astype(jnp.int32)
    pot_mixed = (pot_counts > 0).sum() > 1
    pot_empty = pot_n == 0
    pot_space = (pot_n < 3) & (pot_timer == 0) & (pot_cooked == 0)

    # An empty pot is only safe to seed once the delivery that emptied it has
    # certainly happened, otherwise the recipe may still resample afterwards.
    ticks = jnp.where(pot_empty, jnp.minimum(mem["ticks"] + 1, 999), 0)
    mark = jnp.where(pot_empty, mem["mark"], target)
    # Seeing the partner empty-handed next to an empty pot is direct evidence
    # that no dish is on its way to the station any more; the step count is
    # only a fallback for a partner we cannot see.
    partner_seen, _praw, _pkind, partner_dish = _partner_facts(obs)
    settled = (mark < 0) | (mark != target) | (partner_seen & ~partner_dish)
    seed_safe = (settled | (ticks >= _EMPTY_WAIT)) & ~(
        partner_dish & (ticks < _DISH_WAIT)
    )

    known = target >= 0
    pot_wrong = (pot_n > 0) & (pot_mixed | (pot_major != target))
    # A spoiled pot blocks everything, so fill it up and let it be dumped.
    recover = pot_wrong & pot_space & (mem["step"] < _RECOVER_UNTIL)

    matches = jnp.where(pot_empty, seed_safe, (pot_major == held_ing) & ~pot_mixed)
    insert = has_ing & pot_space & known & (((held_ing == target) & matches) | recover)
    dump = has_ing & known & ~insert & (held_ing != target)
    hold = has_ing & ~insert & ~dump

    # --- parking counters -------------------------------------------------
    busy, kind, _seen, _dish, _plate = _counters(
        obs, r, c, _CNT_L, mem["busy"], mem["kind"]
    )
    base = jnp.asarray([4, 3, 2, 1], dtype=jnp.int32)
    favourite = jnp.clip(held_ing, 0, 1)
    free_rank = jnp.where(busy == 0, base + 10 * (jnp.arange(4) == favourite), -1)
    drop, _drop_idx, _drop_rank = _pick_station(free_rank, _CNT_L, r, c, facing)

    # While the pot needs nothing from us, put the ingredient the recipe does
    # *not* ask for on a counter: after the next delivery the recipe flips half
    # of the time and the stash then replaces a long supply trip.
    other = jnp.where(target == 1, 0, 1)
    # A pot count that grew without an insert of ours means the partner tops
    # the pot up, so we are free to spend the time on the stash instead.
    helper = (mem["helper"] > 0) | ((pot_n > mem["pot"][0:3].sum()) & (mem["fed"] == 0))
    relieved = ~pot_space | (helper & (pot_n > 0))
    prearm = ~has_ing & known & relieved & ~jnp.any(kind == other)
    wanted = jnp.where(prearm, other, target)

    take_rank = jnp.where((kind == wanted) & (busy > 0) & ~prearm, base, -1)
    take, _take_idx, take_rank_best = _pick_station(take_rank, _CNT_L, r, c, facing)
    stashed = take_rank_best > 0

    supply = _select(
        [_go(r, c, facing, _SUPPLY_L[0]), _go(r, c, facing, _SUPPLY_L[1]),
         _go(r, c, facing, _SUPPLY_L[2])],
        [wanted == 0, wanted == 1, wanted == 2],
        _go(r, c, facing, _SUPPLY_L[0]),
    )
    fetch = jnp.where(stashed, take, supply)

    # Stray plates/dishes are useless on this side: park them where the other
    # seat can pick them up.
    share_free = []
    for (gr, gc) in ((1, 7), (3, 7)):
        vec, vis = _read_cell(obs, r, c, gr, gc)
        share_free.append(vis & (vec[29:34].sum() == 0))
    share = _select(
        [_go(r, c, facing, _SHARE_L[0]), _go(r, c, facing, _SHARE_L[1])],
        share_free,
        drop,
    )

    action = _select(
        [share, _go(r, c, facing, _POT_L), drop,
         _go_wait(r, c, facing, _POT_L), _go_wait(r, c, facing, _SEEK_L), fetch],
        [held_plate | held_cooked, insert, dump, hold, ~known, known],
        _go_wait(r, c, facing, _SEEK_L),
    )

    mode = _select(
        [1, 2, 3, 4, 5, 6],
        [held_plate | held_cooked, insert, dump, hold, ~known, known],
        0,
    )

    new_mem = {
        "step": mem["step"] + 1,
        "pot": pot_state,
        "target": target,
        "mark": mark,
        "ticks": ticks,
        "busy": busy.astype(jnp.int32),
        "kind": kind.astype(jnp.int32),
        "helper": helper.astype(jnp.int32),
        "fed": (insert & (action == INTERACT)).astype(jnp.int32),
    }
    trace = {
        "pos": jnp.stack([r, c]).astype(jnp.int32),
        "target": target.astype(jnp.int32),
        "pot_n": pot_n.astype(jnp.int32),
        "mode": mode.astype(jnp.int32),
        "action": action.astype(jnp.int32),
    }
    return action.astype(jnp.int32), new_mem, trace


# --------------------------------------------------------------------------
# agent 1 -- right room: mirrors the pot, plates and delivers
# --------------------------------------------------------------------------


def init_agent_1():
    return {
        "step": jnp.asarray(0, dtype=jnp.int32),
        "pot": jnp.zeros((5,), dtype=jnp.int32),
        "ticks": jnp.asarray(0, dtype=jnp.int32),
        "occupied": jnp.zeros((4,), dtype=jnp.int32),
        "kind": -jnp.ones((4,), dtype=jnp.int32),
        "fresh": jnp.zeros((4,), dtype=jnp.int32),
        "mine": jnp.zeros((4,), dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = _reset_memory(memory, init_agent_1(), reset)

    r, c = _localize(obs, _RIGHT_CELLS)
    facing, held_plate, held_cooked, has_ing, held_ing = _self_facts(obs)

    pot_state, pot_counts, pot_cooked, pot_timer = _pot_facts(obs, r, c, mem["pot"])
    pot_n = pot_counts.sum()
    pot_major = jnp.argmax(pot_counts).astype(jnp.int32)
    pot_mixed = (pot_counts > 0).sum() > 1
    pot_empty = pot_n == 0
    pot_space = (pot_n < 3) & (pot_timer == 0) & (pot_cooked == 0)
    pot_ready = pot_cooked > 0
    pot_busy = ~pot_space & ~pot_ready

    empty_ticks = jnp.where(pot_empty & pot_space, jnp.minimum(mem["ticks"] + 1, 999), 0)

    # --- neighbouring counters -------------------------------------------
    occ_arr, kind, seen, cnt_dish, cnt_plate = _counters(
        obs, r, c, _COUNTERS_R, mem["occupied"], mem["kind"]
    )
    raw_arr = kind >= 0
    # "fresh" = the counter was empty the last time we looked at it, so the item
    # on it now was placed deliberately and reflects the current recipe.
    fresh = jnp.where(
        seen,
        jnp.where(raw_arr, jnp.where(mem["occupied"] == 0, 1, mem["fresh"]), 0),
        mem["fresh"],
    )
    mine = jnp.where(occ_arr > 0, mem["mine"], 0)

    # --- parking spot for an item we cannot use right now ------------------
    park_rank = jnp.where(occ_arr == 0, jnp.arange(4, 0, -1), -1)
    park, park_idx, park_best = _pick_station(park_rank, _COUNTERS_R, r, c, facing)
    park_actions = jnp.stack(
        [_go(r, c, facing, (sr, sc, face)) for (_gr, _gc, sr, sc, face) in _COUNTERS_R]
    )

    # --- a partner hand-off on a shared counter ---------------------------
    usable = [
        raw_arr[k] & (fresh[k] > 0) & (mine[k] == 0) & pot_empty & pot_space
        for k in _SHARED_IDX
    ]
    take = _select(
        [park_actions[_SHARED_IDX[0]], park_actions[_SHARED_IDX[1]]],
        usable,
        park_actions[_SHARED_IDX[0]],
    )
    handoff = usable[0] | usable[1]

    # --- a plated dish somebody left on a counter --------------------------
    dish_rank = jnp.where(cnt_dish, jnp.arange(4, 0, -1), -1)
    dish_take, _dish_idx, dish_best = _pick_station(dish_rank, _COUNTERS_R, r, c, facing)
    dish_waiting = dish_best > 0

    # --- ingredient the pot asks for --------------------------------------
    # With an empty pot the only recipe evidence we can have comes from the
    # partner; a partner that has been carrying a raw ingredient while the pot
    # stayed empty is telling us which one the recipe wants.
    _pvis, partner_raw, partner_kind, _pdish = _partner_facts(obs)
    hint = partner_raw & pot_empty & pot_space & (empty_ticks >= _HINT_WAIT)
    want = jnp.where(pot_n > 0, pot_major, partner_kind)
    want = jnp.where(want <= 1, want, 0)
    fetch_ing = jnp.where(
        want == 1,
        _go(r, c, facing, _SUPPLY_R[1]),
        _go(r, c, facing, _SUPPLY_R[0]),
    )
    # An ingredient of the wanted kind already lying on a counter is taken from
    # there: it costs a few steps more than the pile but it keeps the counters
    # usable and consumes what a partner handed over.
    reclaim_rank = jnp.where((kind == want) & (pot_n > 0), jnp.arange(4, 0, -1), -1)
    reclaim, _reclaim_idx, reclaim_best = _pick_station(
        reclaim_rank, _COUNTERS_R, r, c, facing
    )
    fetch_ing = jnp.where(reclaim_best > 0, reclaim, fetch_ing)
    fetch_plate = jnp.where(
        r <= 2, _go(r, c, facing, _PLATE_R[0]), _go(r, c, facing, _PLATE_R[1])
    )
    # A plate already standing on a counter is exactly as close as the pile.
    plate_rank = jnp.where(cnt_plate, jnp.arange(4, 0, -1), -1)
    plate_take, _plate_idx, plate_best = _pick_station(
        plate_rank, _COUNTERS_R, r, c, facing
    )
    fetch_plate = jnp.where(plate_best > 0, plate_take, fetch_plate)

    idle = ~has_ing & ~held_plate & ~held_cooked
    free_counter = park_best > 0
    insert_ok = has_ing & pot_space & (pot_empty | (pot_major == held_ing) | pot_mixed)
    park_ing = has_ing & ~insert_ok & free_counter
    scoop = held_plate & pot_ready
    park_plate = (held_plate & ~pot_ready & pot_space & free_counter
                  & ((pot_n > 0) | handoff | hint))
    wait_plate = held_plate & ~scoop & ~park_plate
    rescue = idle & dish_waiting
    need_plate = idle & ~rescue & (pot_ready | pot_busy)
    feed = idle & ~rescue & pot_space & ((pot_n > 0) | hint)
    collect = idle & ~rescue & handoff & ~feed

    action = _select(
        [_go(r, c, facing, _DELIVER), _go(r, c, facing, _POT_R), park,
         _go(r, c, facing, _POT_R), park, _go_wait(r, c, facing, _POT_R),
         dish_take, fetch_plate, fetch_ing, take],
        [held_cooked, scoop, park_plate, insert_ok, park_ing, wait_plate,
         rescue, need_plate, feed, collect],
        _go_wait(r, c, facing, _POT_R),
    )

    parking = (park_plate | park_ing) & (park_best > 0)
    mine = jnp.where(
        parking & (jnp.arange(4) == park_idx) & (action == INTERACT), 1, mine
    )
    # Delivering resamples the recipe, so every piece of counter evidence
    # gathered before it becomes stale; picking an item up clears it too.
    delivered = held_cooked & (action == INTERACT)
    fresh = jnp.where((collect | delivered) & (action == INTERACT), 0, fresh)

    mode = _select(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [held_cooked, scoop, park_plate, insert_ok, park_ing, wait_plate,
         rescue, need_plate, feed, collect],
        0,
    )

    new_mem = {
        "step": mem["step"] + 1,
        "pot": pot_state,
        "ticks": empty_ticks.astype(jnp.int32),
        "occupied": occ_arr.astype(jnp.int32),
        "kind": kind.astype(jnp.int32),
        "fresh": fresh.astype(jnp.int32),
        "mine": mine.astype(jnp.int32),
    }
    trace = {
        "pos": jnp.stack([r, c]).astype(jnp.int32),
        "want": want.astype(jnp.int32),
        "pot_n": pot_n.astype(jnp.int32),
        "mode": mode.astype(jnp.int32),
        "action": action.astype(jnp.int32),
    }
    return action.astype(jnp.int32), new_mem, trace
