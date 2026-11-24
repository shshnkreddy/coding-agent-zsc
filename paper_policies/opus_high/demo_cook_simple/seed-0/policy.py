"""Joint cook-and-deliver policy for Overcooked V2 ``demo_cook_simple``.

The board is split by column 7, whose only non-wall cell is the pot itself, so
each seat is permanently confined to one room:

    left room  (rows 1-3, cols 1-6): recipe indicators, ingredient piles 0/1/2,
                                     pot access from (2, 6)
    right room (rows 1-3, cols 8-9): pot access from (2, 8), ingredient piles
                                     0/1, plate piles, delivery station

The role split is therefore forced by geometry rather than by a private
convention: only the right seat can plate and deliver, and only the left seat
can see a recipe indicator.  The left seat loads the first ingredient, which
publishes the target ingredient inside the pot where anyone can read it; the
right seat only ever completes the type the pot already holds -- the single
completion that can still be correct -- and never commits an empty pot to a
guess.  Each seat decides from its own crop alone, so the two halves work with
unknown partners as well as with each other.
"""

from __future__ import annotations

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'room_split_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# actions
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# facing ids: UP=0, DOWN=1, RIGHT=2, LEFT=3
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
_FACE_TO_ACTION = (UP, DOWN, RIGHT, LEFT)

# static channels
C_WALL, C_GOAL, C_POT, C_RECIPE, C_BUTTON, C_PLATE = 20, 21, 22, 23, 24, 25
C_PILE0, C_PILE1, C_PILE2 = 26, 27, 28

# stations: (stand_row, stand_col, facing, interact_on_arrival)
_LEFT_STATIONS = (
    (2, 6, F_RIGHT, 0),   # 0 observation post, already facing the pot
    (2, 6, F_RIGHT, 1),   # 1 pot (2, 7)
    (1, 1, F_LEFT, 1),    # 2 ingredient-0 pile (1, 0)
    (3, 1, F_LEFT, 1),    # 3 ingredient-1 pile (3, 0)
    (1, 6, F_UP, 1),      # 4 ingredient-2 pile (0, 6)
    (1, 4, F_UP, 1),      # 5 counter (0, 4): home of a spare ingredient 0
    (3, 4, F_DOWN, 1),    # 6 counter (4, 4): home of a spare ingredient 1
    (1, 2, F_UP, 1),      # 7 counter (0, 2): spare / junk
    (3, 2, F_DOWN, 1),    # 8 counter (4, 2): spare / junk
)
_RIGHT_STATIONS = (
    (2, 8, F_LEFT, 0),    # 0 park beside the pot
    (2, 8, F_LEFT, 1),    # 1 pot (2, 7)
    (1, 8, F_UP, 1),      # 2 ingredient-0 pile (0, 8)
    (3, 8, F_DOWN, 1),    # 3 ingredient-1 pile (4, 8)
    (1, 9, F_RIGHT, 1),   # 4 plate pile (1, 10)
    (3, 9, F_RIGHT, 1),   # 5 plate pile (3, 10)
    (2, 9, F_RIGHT, 1),   # 6 delivery station (2, 10)
    (1, 9, F_UP, 1),      # 7 counter (0, 9)
    (3, 9, F_DOWN, 1),    # 8 counter (4, 9)
    (1, 8, F_LEFT, 1),    # 9 counter (1, 7)
    (3, 8, F_LEFT, 1),    # 10 counter (3, 7)
)
_LEFT_SLOTS = ((0, 4), (4, 4), (0, 2), (4, 2))
_RIGHT_SLOTS = ((0, 9), (4, 9), (1, 7), (3, 7))
_LEFT_HOME = (5, 6, 7)    # station that stores a spare unit of ingredient k
_LEFT_PILE = (2, 3, 4)    # station of the supply pile of ingredient k

# memory layout (each seat keeps its own; index 10 differs per room)
M_T, M_REC, M_POTN, M_POTT, M_POTS, M_WAIT, M_SLOT0 = 0, 1, 2, 3, 4, 5, 6
M_PDISH, M_STALL = 10, 11
M_HANDOFF = 10
_INIT_MEMORY = (0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0)

# steps allowed for a scooped dish to be delivered before the recipe display is
# trusted again for an empty pot
RECIPE_SETTLE = 5
# steps a partially filled pot may stay unchanged before the left seat tops it
# up itself instead of leaving the remaining units to the other room
STALL_LIMIT = 8
# how long the left seat keeps deferring to a cooked dish it saw leave the pot
# but has not yet seen put down anywhere
DELIVERY_GRACE = 16
# how long the right seat waits on a dead pot before treating an ingredient
# left on the shared counters (1, 7) / (3, 7) as a hand-off meant for the pot
HANDOFF_WAIT = 15


# ------------------------------------------------------------------- utilities


def _landmark(obs, channel):
    """Column of a static landmark inside the crop, plus a visibility flag."""
    columns = jnp.max(obs[:, :, channel], axis=0)
    return jnp.max(columns) > 0, jnp.argmax(columns).astype(jnp.int32)


def _localize(obs):
    """Recover the global cell from static landmarks; the map is fully known."""
    above = jnp.sum(obs[1, 2, C_WALL:C_PILE2 + 1]) > 0
    below = jnp.sum(obs[3, 2, C_WALL:C_PILE2 + 1]) > 0
    row = jnp.where(above, 1, jnp.where(below, 3, 2)).astype(jnp.int32)

    goal_seen, goal_col = _landmark(obs, C_GOAL)      # unique cell (2, 10)
    pot_seen, pot_col = _landmark(obs, C_POT)         # unique cell (2, 7)
    rec_seen, rec_col = _landmark(obs, C_RECIPE)      # cells (0, 5), (4, 5)
    p0_seen, p0_col = _landmark(obs, C_PILE0)         # (1, 0) when near column 0
    p1_seen, p1_col = _landmark(obs, C_PILE1)         # (3, 0) when near column 0
    edge_seen = p0_seen | p1_seen
    edge_col = jnp.where(p0_seen, p0_col, p1_col)

    left_col = jnp.where(
        pot_seen,
        9 - pot_col,
        jnp.where(rec_seen, 7 - rec_col, jnp.where(edge_seen, 2 - edge_col, 1)),
    )
    col = jnp.where(goal_seen, 12 - goal_col, left_col).astype(jnp.int32)
    return row, jnp.clip(col, 1, 9)


def _read(obs, row, col, gr, gc):
    """Return (visible, channel vector) for a global cell."""
    i = gr - row + 2
    j = gc - col + 2
    visible = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    vector = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), :]
    return visible, vector


def _occupied(obs, row, col, gr, gc, remembered):
    visible, vector = _read(obs, row, col, gr, gc)
    here = (jnp.sum(vector[29:34]) > 0).astype(jnp.int32)
    return jnp.where(visible, here, remembered).astype(jnp.int32)


def _slot_flags(obs, row, col, memory, cells):
    flags = [
        _occupied(obs, row, col, gr, gc, memory[M_SLOT0 + k])
        for k, (gr, gc) in enumerate(cells)
    ]
    return jnp.stack(flags).astype(jnp.int32)


def _counter_offer(obs, row, col, gr, gc):
    """True when a counter holds exactly one raw unit of a usable ingredient."""
    visible, vector = _read(obs, row, col, gr, gc)
    counts = vector[31:34]
    kind = jnp.argmax(counts).astype(jnp.int32)
    single = (vector[29] == 0) & (vector[30] == 0) & (jnp.sum(counts) == 1)
    return visible & single & (kind < 2), kind


def _navigate(row, col, facing, target):
    """Walk an obstacle-free room rectangle, then turn and optionally interact."""
    face_to_action = jnp.asarray(_FACE_TO_ACTION, dtype=jnp.int32)
    trow, tcol, tface, tinteract = target[0], target[1], target[2], target[3]
    drow = trow - row
    dcol = tcol - col
    step_v = jnp.where(drow > 0, DOWN, UP)
    step_h = jnp.where(dcol > 0, RIGHT, LEFT)
    arrive = jnp.where(
        facing == tface,
        jnp.where(tinteract > 0, INTERACT, STAY),
        face_to_action[tface],
    )
    # close along the axis of the required facing so the final move also turns
    vertical_first = (tface == F_RIGHT) | (tface == F_LEFT)
    plan_v = jnp.where(drow != 0, step_v, jnp.where(dcol != 0, step_h, arrive))
    plan_h = jnp.where(dcol != 0, step_h, jnp.where(drow != 0, step_v, arrive))
    return jnp.where(vertical_first, plan_v, plan_h).astype(jnp.int32)


def _pot_view(obs, row, col, memory):
    visible, vector = _read(obs, row, col, 2, 7)
    raw = vector[31:34]
    count = jnp.sum(raw)
    kind = jnp.where(count > 0, jnp.argmax(raw), -1).astype(jnp.int32)
    state = jnp.where(vector[30] > 0, 2, jnp.where(vector[39] > 0, 1, 0)).astype(jnp.int32)
    pot_n = jnp.where(visible, count, memory[M_POTN]).astype(jnp.int32)
    pot_t = jnp.where(visible, kind, memory[M_POTT]).astype(jnp.int32)
    pot_s = jnp.where(visible, state, memory[M_POTS]).astype(jnp.int32)
    # a cooked pot that became empty means somebody collected the dish
    scooped = visible & (memory[M_POTS] == 2) & (state == 0) & (count == 0)
    return pot_n, pot_t, pot_s, scooped


def _inventory(obs):
    plate = obs[2, 2, 5]
    cooked = obs[2, 2, 6]
    counts = obs[2, 2, 7:10]
    total = jnp.sum(counts)
    kind = jnp.argmax(counts).astype(jnp.int32)
    return plate, cooked, counts, total, kind


def _store(memory, recipe, pot_n, pot_t, pot_s, wait, slots, pdish, stall):
    memory = memory.at[M_REC].set(recipe)
    memory = memory.at[M_POTN].set(pot_n)
    memory = memory.at[M_POTT].set(pot_t)
    memory = memory.at[M_POTS].set(pot_s)
    memory = memory.at[M_WAIT].set(wait)
    memory = memory.at[M_SLOT0:M_SLOT0 + 4].set(slots)
    memory = memory.at[M_PDISH].set(pdish)
    memory = memory.at[M_STALL].set(stall)
    return memory.astype(jnp.int32)


# ------------------------------------------------------------------- left seat


def _left_seat(obs, memory, row, col):
    table = jnp.asarray(_LEFT_STATIONS, dtype=jnp.int32)
    home = jnp.asarray(_LEFT_HOME, dtype=jnp.int32)
    pile = jnp.asarray(_LEFT_PILE, dtype=jnp.int32)

    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    plate, _cooked, _counts, held_total, held_kind = _inventory(obs)
    has_ingredient = (held_total > 0) & (plate == 0)
    has_object = plate > 0

    # recipe indicators at (0, 5) and (4, 5)
    vis_a, cell_a = _read(obs, row, col, 0, 5)
    vis_b, cell_b = _read(obs, row, col, 4, 5)
    disp_a = cell_a[36:39] * vis_a.astype(jnp.int32)
    disp_b = cell_b[36:39] * vis_b.astype(jnp.int32)
    display = jnp.where(jnp.sum(disp_a) > 0, disp_a, disp_b)
    seen = jnp.sum(display) > 0
    shown = jnp.argmax(display).astype(jnp.int32)
    previous = memory[M_REC]
    recipe = jnp.where(seen, shown, previous).astype(jnp.int32)
    changed = seen & (previous >= 0) & (shown != previous)

    pot_n, pot_t, pot_s, scooped = _pot_view(obs, row, col, memory)
    pot_moved = (pot_n != memory[M_POTN]) | (pot_s != memory[M_POTS])
    stall = jnp.where(pot_moved, 0, jnp.minimum(memory[M_STALL] + 1, 999)).astype(jnp.int32)

    # A partner seen putting down the cooked dish proves the recipe display is
    # up to date again: either it delivered (and the recipe resampled before
    # this observation) or it did not (and the recipe never changed at all).
    partner_seen = jnp.max(obs[:, :, 10]) > 0
    partner_dish = jnp.max(obs[:, :, 16]) > 0
    delivered = (memory[M_PDISH] > 0) & partner_seen & jnp.logical_not(partner_dish)
    # an emptied cooked pot means a dish exists even if its carrier is out of view
    pdish = jnp.where(
        partner_dish | scooped, 1, jnp.where(delivered, 0, memory[M_PDISH])
    ).astype(jnp.int32)

    wait = jnp.maximum(memory[M_WAIT] - 1, 0)
    wait = jnp.where(scooped, RECIPE_SETTLE, wait)
    wait = jnp.where(changed | delivered, 0, wait).astype(jnp.int32)
    # a dish is still in flight: the display may not have been resampled yet
    hold_off = (pdish > 0) & (stall < DELIVERY_GRACE) & jnp.logical_not(changed)

    slots = _slot_flags(obs, row, col, memory, _LEFT_SLOTS)

    addable = (pot_s == 0) & (pot_n < 3)
    committed = pot_n > 0
    target = jnp.where(recipe >= 0, recipe, pot_t).astype(jnp.int32)
    # A pot already holding the wrong type can never win, so it is completed
    # with whatever is in hand: that clears it fastest and leaves visibly mixed
    # contents, which any partner that checks can park instead of delivering.
    doomed = committed & (recipe >= 0) & (pot_t != recipe)
    # committing the empty pot is this seat's job; topping up an already
    # committed pot is left to the other room unless nothing happens there
    gate = jnp.where(
        committed, stall >= STALL_LIMIT, (wait == 0) & jnp.logical_not(hold_off)
    )
    can_add = addable & (target >= 0) & gate
    add_now = addable & gate & ((held_kind == target) | doomed)

    safe_t = jnp.clip(target, 0, 2)
    cached = jnp.where(safe_t < 2, slots[safe_t], 0)
    fetch_target = jnp.where(cached > 0, home[safe_t], pile[safe_t])

    junk = jnp.where(slots[2] == 0, 7, 8)
    home_free = jnp.where(held_kind < 2, slots[jnp.clip(held_kind, 0, 1)], 1) == 0
    drop_here = jnp.where(home_free, home[held_kind], junk)

    carry_plan = jnp.where(
        add_now, 1, jnp.where(held_kind == target, 0, drop_here)
    )
    idle_plan = jnp.where(recipe >= 0, fetch_target, 0)
    station = jnp.where(
        has_ingredient,
        carry_plan,
        jnp.where(has_object, junk, idle_plan),
    ).astype(jnp.int32)

    action = _navigate(row, col, facing, table[station])
    # an unknown recipe only happens in columns 1-2 at reset: step into view
    action = jnp.where(recipe < 0, jnp.asarray(RIGHT, dtype=jnp.int32), action)
    return action.astype(jnp.int32), _store(
        memory, recipe, pot_n, pot_t, pot_s, wait, slots, pdish, stall
    )


# ------------------------------------------------------------------ right seat


def _right_seat(obs, memory, row, col):
    table = jnp.asarray(_RIGHT_STATIONS, dtype=jnp.int32)

    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    plate, cooked_flag, counts, held_total, held_kind = _inventory(obs)
    has_dish = (plate > 0) & (cooked_flag > 0)
    has_plate = (plate > 0) & (cooked_flag == 0)
    has_ingredient = (held_total > 0) & (plate == 0)
    # the only possible recipes are [0,0,0] and [1,1,1]; anything else is a
    # certain -20, so it is parked on a counter instead of being delivered
    deliverable = (counts[0] == 3) | (counts[1] == 3)

    pot_n, pot_t, pot_s, _scooped = _pot_view(obs, row, col, memory)
    pot_moved = (pot_n != memory[M_POTN]) | (pot_s != memory[M_POTS])
    stall = jnp.where(pot_moved, 0, jnp.minimum(memory[M_STALL] + 1, 999)).astype(jnp.int32)
    slots = _slot_flags(obs, row, col, memory, _RIGHT_SLOTS)

    ready = pot_s == 2
    busy = (pot_s == 1) | (pot_n >= 3) | ready
    filling = (pot_s == 0) & (pot_n > 0) & (pot_n < 3)
    empty_pot = (pot_s == 0) & (pot_n == 0)

    # (1, 7) and (3, 7) are reachable from the other room, so plates and dishes
    # are only ever parked on (0, 9) / (4, 9) where nobody can strand them
    stash_near = jnp.where(slots[0] == 0, 7, jnp.where(slots[1] == 0, 8, 0))
    stash_any = jnp.where(
        slots[0] == 0,
        7,
        jnp.where(
            slots[1] == 0,
            8,
            jnp.where(slots[2] == 0, 9, jnp.where(slots[3] == 0, 10, 0)),
        ),
    )
    plate_pile = jnp.where(row <= 2, 4, 5)

    # A single raw unit left on a counter that both rooms can reach is the only
    # way the other room can offer material without using the pot.  It is taken
    # to the pot only once the pot has been dead for a while, so it can never
    # pre-empt a partner that is still working.
    offer_a, kind_a = _counter_offer(obs, row, col, 1, 7)
    offer_b, kind_b = _counter_offer(obs, row, col, 3, 7)
    offered = offer_a | offer_b
    offer_station = jnp.where(offer_a, 9, 10)
    dead_pot = empty_pot & (stall >= HANDOFF_WAIT)
    carrying_offer = memory[M_HANDOFF] > 0

    dish_plan = jnp.where(deliverable | (stash_any == 0), 6, stash_any)
    plate_plan = jnp.where(ready, 1, jnp.where(busy, 0, stash_near))
    carry_plan = jnp.where(
        (filling & (held_kind == pot_t)) | (empty_pot & carrying_offer),
        1,
        jnp.where(empty_pot, 0, stash_any),
    )
    idle_plan = jnp.where(
        busy,
        plate_pile,
        jnp.where(
            filling,
            jnp.where(pot_t == 0, 2, jnp.where(pot_t == 1, 3, 0)),
            jnp.where(dead_pot & offered, offer_station, 0),
        ),
    )
    station = jnp.where(
        has_dish,
        dish_plan,
        jnp.where(has_plate, plate_plan, jnp.where(has_ingredient, carry_plan, idle_plan)),
    ).astype(jnp.int32)

    action = _navigate(row, col, facing, table[station])
    empty_handed = jnp.logical_not(has_dish | has_plate | has_ingredient)
    took_offer = empty_handed & (station >= 9) & (action == INTERACT)
    handoff = jnp.where(
        took_offer, 1, jnp.where(has_ingredient, memory[M_HANDOFF], 0)
    ).astype(jnp.int32)
    return action.astype(jnp.int32), _store(
        memory,
        memory[M_REC],
        pot_n,
        pot_t,
        pot_s,
        memory[M_WAIT],
        slots,
        handoff,
        stall,
    )


# ----------------------------------------------------------------- entry points


def _act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    fresh = jnp.asarray(_INIT_MEMORY, dtype=jnp.int32)
    memory = jnp.where(jnp.asarray(reset, dtype=bool), fresh, memory)
    row, col = _localize(obs)
    right_room = col >= 8

    left_action, left_memory = _left_seat(obs, memory, row, col)
    right_action, right_memory = _right_seat(obs, memory, row, col)

    action = jnp.where(right_room, right_action, left_action).astype(jnp.int32)
    memory = jnp.where(right_room, right_memory, left_memory).astype(jnp.int32)
    memory = memory.at[M_T].set(memory[M_T] + 1)

    trace = {
        "state": jnp.stack(
            [
                row,
                col,
                right_room.astype(jnp.int32),
                action,
                memory[M_REC],
                memory[M_POTN],
                memory[M_POTT],
                memory[M_POTS],
                memory[M_WAIT],
            ]
        ).astype(jnp.int32)
    }
    return action, memory, trace


def init_agent_0():
    return jnp.asarray(_INIT_MEMORY, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)


def init_agent_1():
    return jnp.asarray(_INIT_MEMORY, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)
