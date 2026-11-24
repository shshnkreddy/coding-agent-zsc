"""Cooperative Overcooked-V2 policy for layout ``demo_cook_simple``.

The geometry forces a hard role split that any competent pair of policies has to
discover:

* Only the **left** room (seat ``agent_0``) can see the recipe indicators at
  (0, 5) / (4, 5), and only it can reach the ingredient piles at (1, 0) / (3, 0).
* Only the **right** room (seat ``agent_1``) can reach the plate piles at
  (1, 10) / (3, 10) and the delivery station at (2, 10).  It has its own
  ingredient-0 pile at (0, 8) and ingredient-1 pile at (4, 8) one step from the
  pot, but it can never observe the target recipe.

The shared pot at (2, 7) is the only object both seats can touch, so it carries
the message: ``agent_0`` puts the *first* correct raw ingredient into the pot,
and ``agent_1`` simply mirrors whatever is already in the pot while topping it
up.  That needs no shared convention beyond "the pot shows what we are cooking",
which is the reading an independently written partner is most likely to share.
``agent_1`` additionally accepts a single unambiguous raw ingredient left on the
shared counters (1, 7) / (3, 7) as a hand-off, for partners that prefer counter
passing, and mirrors the pot in preference to any private signal.

``agent_0`` keeps a private ingredient cache on counters (0, 4) / (4, 4) (with
(0, 2) / (4, 2) as overflow).  Those cells sit in columns the right room can
never observe, so the cache shortens the post-delivery signal from ~15 steps to
~7 without ever showing an ambiguous item to a partner.

Both rooms are obstacle-free rectangles (rows 1-3 x cols 1-6 and rows 1-3 x
cols 8-9), so shortest paths are plain Manhattan routes; the approach axis is
ordered so the final step already leaves the agent facing its target.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "pot_signal_role_split",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 11,
    "num_ingredients": 3,
}

# --------------------------------------------------------------------------
# Layout and observation constants (literals only; no module-scope calls).
# --------------------------------------------------------------------------

W = 11
_ASCII = ("WWWWWR2W0WW", "0      W  B", "W     APA X", "1      W  B", "WWWWWR2W1WW")
# Order of static channels 20..28.
_CLASS_ORDER = "WXPRLB012"

C_SELF_DIR = 1          # UP, DOWN, RIGHT, LEFT one-hot at own cell
C_SELF_INV = 5          # plate, cooked, ing0, ing1, ing2
C_STATIC = 20
C_ITEM = 29             # plate, cooked, ing0, ing1, ing2 on the grid cell
C_RECIPE = 34           # plate, cooked, ing0, ing1, ing2 of the displayed recipe
C_POT_TIMER = 39

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Observation facing ids (UP, DOWN, RIGHT, LEFT) -> movement action ids.
_DIR_TO_ACT = [UP, DOWN, RIGHT, LEFT]

_LEFT_CELLS = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
               (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
               (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6))
_RIGHT_CELLS = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))

# --------------------------------------------------------------------------
# Table construction (traced once per jit; all calls live inside functions).
# --------------------------------------------------------------------------


def _crop_bank(cells):
    """Expected static 5x5x9 crop, and flat cell index, for each candidate."""
    grid = jnp.asarray(
        [[[1 if _ASCII[r][c] == k else 0 for k in _CLASS_ORDER] for c in range(W)]
         for r in range(5)], dtype=jnp.int32)
    pad = jnp.zeros((9, W + 4, 9), dtype=jnp.int32).at[2:7, 2:2 + W, :].set(grid)
    crops = jnp.stack([pad[r:r + 5, c:c + 5, :] for (r, c) in cells])
    ids = jnp.asarray([r * W + c for (r, c) in cells], dtype=jnp.int32)
    return crops, ids


def _localize(obs, cells):
    """Exact grid position: the candidate whose static crop matches best."""
    crops, ids = _crop_bank(cells)
    static = obs[:, :, C_STATIC:C_STATIC + 9]
    score = jnp.sum(crops == static[None], axis=(1, 2, 3))
    return ids[jnp.argmax(score)]


def _read_cell(obs, pos, gr, gc):
    """Observation vector at global (gr, gc), plus a visibility flag."""
    lr = gr - pos // W + 2
    lc = gc - pos % W + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    return obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)], visible


def _item_code(cell):
    """-1 empty, 0..2 raw ingredient id, 3 plate or plated dish."""
    counts = cell[C_ITEM + 2:C_ITEM + 5]
    plateish = (cell[C_ITEM] > 0) | (cell[C_ITEM + 1] > 0)
    return jnp.where(plateish, 3,
                     jnp.where(jnp.sum(counts) > 0, jnp.argmax(counts), -1)).astype(jnp.int32)


def _decode_self(obs):
    face = jnp.asarray(_DIR_TO_ACT)[jnp.argmax(obs[2, 2, C_SELF_DIR:C_SELF_DIR + 4])]
    return face.astype(jnp.int32), obs[2, 2, C_SELF_INV:C_SELF_INV + 5]


def _manhattan(pos, goal):
    return (jnp.abs(pos // W - goal // W) + jnp.abs(pos % W - goal % W)).astype(jnp.int32)


def _go(pos, face, stand, need, interact):
    """Route to ``stand``, end up facing ``need``, then INTERACT (or STAY).

    Rooms have no interior obstacles, so the route only has to pick an axis
    order.  Resolving the axis *orthogonal* to ``need`` first means the last
    translation already sets the required facing, saving a turn.
    """
    row, col = pos // W, pos % W
    goal_row, goal_col = stand // W, stand % W
    off_row, off_col = row != goal_row, col != goal_col
    vert = jnp.where(goal_row > row, DOWN, UP)
    horiz = jnp.where(goal_col > col, RIGHT, LEFT)
    need_horiz = (need == RIGHT) | (need == LEFT)
    row_first = jnp.where(off_row, vert, horiz)
    col_first = jnp.where(off_col, horiz, vert)
    move = jnp.where(need_horiz, row_first, col_first)
    final = jnp.where(face == need, jnp.where(interact, INTERACT, STAY), need)
    return jnp.where(off_row | off_col, move, final).astype(jnp.int32)


def _pot_state(obs, pos):
    cell, visible = _read_cell(obs, pos, 2, 7)
    counts = jnp.where(visible, cell[C_ITEM + 2:C_ITEM + 5], 0)
    return counts, jnp.where(visible, cell[C_ITEM + 1], 0), jnp.where(visible, cell[C_POT_TIMER], 0), visible


# --------------------------------------------------------------------------
# agent_0: left room -- reads the recipe, seeds the pot, keeps a cache.
# --------------------------------------------------------------------------

A0_IDLE = 27            # (2, 5): sees both recipe indicators, the pot and both cache slots
A0_POT_STAND = 28       # (2, 6), facing RIGHT
A0_SUP_STAND = (12, 34, 17)     # (1,1) ing0, (3,1) ing1, (1,6) ing2
A0_SUP_NEED = (LEFT, LEFT, UP)
# Cache slots: standing cell + facing.  Counters (0,4) (4,4) (0,2) (4,2) all lie
# in columns the right room can never see.
A0_SLOT_CELLS = ((0, 4), (4, 4), (0, 2), (4, 2))
A0_SLOT_STAND = (15, 37, 13, 35)
A0_SLOT_NEED = (UP, DOWN, UP, DOWN)
# Preferred drop order per held-item code 0..3.
A0_DROP_ORDER = ((0, 2, 3, 1), (1, 3, 2, 0), (2, 3, 0, 1), (2, 3, 0, 1))

# memory: [recipe_seen, recipe_id, pot_has_room, pot_spoiled, slot0..slot3,
#          pot_signature, stagnant_steps]
_A0_INIT_LIST = [0, 0, 1, 0, -1, -1, -1, -1, -1, 0]
# How long a doomed pot must sit untouched before we conclude the partner has
# given up on it and fill it ourselves just to force a cook and free it.
_STAGNANT_STEPS = 15


def init_agent_0():
    return jnp.asarray(_A0_INIT_LIST, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, jnp.asarray(_A0_INIT_LIST, dtype=jnp.int32), memory)
    pos = _localize(obs, _LEFT_CELLS)
    face, inv = _decode_self(obs)

    # --- beliefs -----------------------------------------------------------
    top, vis_top = _read_cell(obs, pos, 0, 5)
    bottom, vis_bottom = _read_cell(obs, pos, 4, 5)
    rec_counts = (jnp.where(vis_top, top[C_RECIPE + 2:C_RECIPE + 5], 0)
                  + jnp.where(vis_bottom, bottom[C_RECIPE + 2:C_RECIPE + 5], 0))
    rec_seen = jnp.sum(rec_counts) > 0
    recipe_known = jnp.where(rec_seen, 1, mem[0]).astype(jnp.int32)
    recipe = jnp.where(rec_seen, jnp.argmax(rec_counts), mem[1]).astype(jnp.int32)

    pot_counts, pot_cooked, pot_timer, pot_visible = _pot_state(obs, pos)
    pot_total = jnp.sum(pot_counts)
    room_now = (pot_cooked == 0) & (pot_timer == 0) & (pot_total < 3)
    spoiled_now = jnp.sum(jnp.where(jnp.arange(3) == recipe, 0, pot_counts)) > 0
    pot_room = jnp.where(pot_visible, room_now.astype(jnp.int32), mem[2]).astype(jnp.int32)
    pot_spoiled = jnp.where(pot_visible, spoiled_now.astype(jnp.int32), mem[3]).astype(jnp.int32)

    # Track how long the pot has been untouched, so a doomed pot can be told
    # apart from one a partner is actively (if wrongly) working on.
    signature = (pot_counts[0] + 5 * pot_counts[1] + 25 * pot_counts[2]
                 + 125 * pot_cooked + 250 * (pot_timer > 0)).astype(jnp.int32)
    pot_sig = jnp.where(pot_visible, signature, mem[8]).astype(jnp.int32)
    stagnant = jnp.where(pot_visible & (signature != mem[8]), 0,
                         jnp.minimum(mem[9] + 1, 1000)).astype(jnp.int32)

    slots = mem[4:8]
    for s in range(4):
        cell, visible = _read_cell(obs, pos, A0_SLOT_CELLS[s][0], A0_SLOT_CELLS[s][1])
        slots = slots.at[s].set(jnp.where(visible, _item_code(cell), slots[s]))
    slot_stand = jnp.asarray(A0_SLOT_STAND, dtype=jnp.int32)
    slot_need = jnp.asarray(A0_SLOT_NEED, dtype=jnp.int32)

    # --- inventory ---------------------------------------------------------
    held_counts = inv[2:5]
    holds_ing = jnp.sum(held_counts) > 0
    holds_other = (inv[0] > 0) | (inv[1] > 0)
    held_id = jnp.where(holds_ing, jnp.argmax(held_counts), 3).astype(jnp.int32)

    known = recipe_known > 0
    has_room = pot_room > 0
    want = jnp.clip(recipe, 0, 2)

    # A pot that already holds the wrong ingredient cannot be rescued: we know
    # the recipe, so every completion of it is a -20 delivery.  Adding to it
    # only makes a pure-wrong dish into a certainly-wrong mixed one and wastes
    # our time, so leave it to whoever is working on it.  Only once it has sat
    # untouched long enough to look abandoned do we fill it, purely to force a
    # cook so the pot can be emptied again instead of deadlocking the episode.
    doomed = pot_spoiled > 0
    force_clear = doomed & (stagnant >= _STAGNANT_STEPS)
    should_feed = has_room & ((~doomed) | force_clear)

    # --- candidate actions -------------------------------------------------
    idle = _go(pos, face, A0_IDLE, face, False)

    cached = slots == want
    cache_slot = jnp.argmax(cached)
    acquire = jnp.where(
        jnp.any(cached),
        _go(pos, face, slot_stand[cache_slot], slot_need[cache_slot], True),
        _go(pos, face, jnp.asarray(A0_SUP_STAND)[want], jnp.asarray(A0_SUP_NEED)[want], True))

    # Refill an empty cache slot while the pot is busy cooking.
    restock_slot = jnp.where(slots[0] < 0, 0, 1)
    restock_needed = (slots[0] < 0) | (slots[1] < 0)
    restock = _go(pos, face, jnp.asarray(A0_SUP_STAND)[restock_slot],
                  jnp.asarray(A0_SUP_NEED)[restock_slot], True)

    order = jnp.asarray(A0_DROP_ORDER, dtype=jnp.int32)[held_id]
    free = slots[order] < 0
    drop_slot = order[jnp.argmax(free)]
    stow = jnp.where(jnp.any(free),
                     _go(pos, face, slot_stand[drop_slot], slot_need[drop_slot], True),
                     idle)

    to_pot = _go(pos, face, A0_POT_STAND, RIGHT, True)

    # --- priority ladder (later assignments win) ---------------------------
    empty_handed = (~holds_ing) & (~holds_other)
    action = idle
    action = jnp.where(known & empty_handed & (~should_feed) & restock_needed, restock, action)
    action = jnp.where(known & empty_handed & should_feed, acquire, action)
    action = jnp.where(holds_ing | holds_other, stow, action)
    action = jnp.where(holds_ing & should_feed & ((held_id == want) | force_clear),
                       to_pot, action)

    new_mem = jnp.concatenate(
        [jnp.stack([recipe_known, recipe, pot_room, pot_spoiled]), slots.astype(jnp.int32),
         jnp.stack([pot_sig, stagnant])])
    trace = {
        "pos": pos.astype(jnp.int32),
        "recipe": recipe.astype(jnp.int32),
        "pot_total": pot_total.astype(jnp.int32),
        "held": held_id.astype(jnp.int32),
        "action": action.astype(jnp.int32),
    }
    return action.astype(jnp.int32), new_mem, trace


# --------------------------------------------------------------------------
# agent_1: right room -- mirrors the pot, plates and delivers.
# --------------------------------------------------------------------------

A1_IDLE = 30            # (2, 8), facing LEFT: adjacent to the pot
A1_SUP_STAND = (19, 41, 19)     # (1,8) ing0, (3,8) ing1
A1_SUP_NEED = (UP, DOWN, UP)
A1_DELIVER_STAND = 31   # (2, 9), facing RIGHT
A1_PLATE_STAND = (20, 42)       # (1,9) and (3,9), facing RIGHT
# Shared with the left room: read-only for us, so that leaving something here
# can never jam a partner that is using these counters itself.
A1_CNT_CELLS = ((1, 7), (3, 7))
A1_CNT_STAND = (19, 41)         # (1,8) and (3,8), facing LEFT
# Private storage: the walls at (0,9) and (4,9) are reachable only from the
# right room, so anything parked there is out of the partner's way entirely.
A1_PARK_CELLS = ((0, 9), (4, 9))
A1_PARK_STAND = (20, 42)        # (1,9) and (3,9)
A1_PARK_NEED = (UP, DOWN)

# memory: [empty_pot_streak, seen_empty_upper, seen_empty_lower,
#          presumed_good, known_bad, previous_pot_total, deposited_last_step,
#          partner_uses_pot]
_A1_INIT_LIST = [0, 0, 0, 0, 0, 0, 0, 0]
_HANDOFF_PATIENCE = 6
_STALE_PATIENCE = 15
_PLATE_PATIENCE = 20


def init_agent_1():
    return jnp.asarray(_A1_INIT_LIST, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, jnp.asarray(_A1_INIT_LIST, dtype=jnp.int32), memory)
    pos = _localize(obs, _RIGHT_CELLS)
    face, inv = _decode_self(obs)

    pot_counts, pot_cooked, pot_timer, _ = _pot_state(obs, pos)
    pot_total = jnp.sum(pot_counts)
    pot_busy = (pot_timer > 0) | (pot_cooked > 0) | (pot_total >= 3)
    pot_room = (pot_cooked == 0) & (pot_timer == 0) & (pot_total < 3)
    # Every recipe is three of a single ingredient drawn from {0, 1}, so a pot
    # holding a mixture, or any ingredient 2, can never be served.
    pot_unservable = ((jnp.sum((pot_counts > 0).astype(jnp.int32)) > 1)
                      | (pot_counts[2] > 0))

    upper, _ = _read_cell(obs, pos, A1_CNT_CELLS[0][0], A1_CNT_CELLS[0][1])
    lower, _ = _read_cell(obs, pos, A1_CNT_CELLS[1][0], A1_CNT_CELLS[1][1])
    cnt_code = jnp.stack([_item_code(upper), _item_code(lower)])
    cnt_free = cnt_code < 0
    cnt_raw = (cnt_code >= 0) & (cnt_code <= 2)
    cnt_stand = jnp.asarray(A1_CNT_STAND, dtype=jnp.int32)

    park_up, _ = _read_cell(obs, pos, A1_PARK_CELLS[0][0], A1_PARK_CELLS[0][1])
    park_down, _ = _read_cell(obs, pos, A1_PARK_CELLS[1][0], A1_PARK_CELLS[1][1])
    park_cells = jnp.stack([park_up, park_down])
    park_code = jnp.stack([_item_code(park_up), _item_code(park_down)])
    park_free = park_code < 0
    park_raw = (park_code >= 0) & (park_code <= 2)
    park_plate = (park_cells[:, C_ITEM] > 0) & (park_cells[:, C_ITEM + 1] == 0)
    park_stand = jnp.asarray(A1_PARK_STAND, dtype=jnp.int32)
    park_need = jnp.asarray(A1_PARK_NEED, dtype=jnp.int32)

    # "Seen empty since our own last delivery": an item that appeared on a
    # shared counter after that moment cannot predate the current recipe.
    seen_empty = jnp.where(cnt_free, 1, mem[1:3])

    # Clearing a doomed pot is only worth a -20 delivery once it has already
    # paid for itself; otherwise a partner that keeps spoiling the pot would
    # turn us into a -20 pump.  Parking privately always costs nothing.
    can_park = jnp.any(park_free)
    disposal_ok = can_park | (mem[3] > mem[4])
    pot_workable = (~pot_unservable) | disposal_ok

    # Has the partner ever put an ingredient in the pot itself?  If so the pot
    # is demonstrably its channel, and an item sitting on a shared counter is
    # far more likely to be that partner's private cache than a hand-off meant
    # for us -- cooking it would be a coin flip.  Trust only the pot from then on.
    partner_added = (pot_total - mem[5] - mem[6]) > 0
    partner_uses_pot = jnp.maximum(mem[7], partner_added.astype(jnp.int32))

    held_counts = inv[2:5]
    holds_ing = jnp.sum(held_counts) > 0
    holds_plate = (inv[0] > 0) & (inv[1] == 0)
    holds_dish = (inv[0] > 0) & (inv[1] > 0)
    held_id = jnp.where(holds_ing, jnp.argmax(held_counts), 3).astype(jnp.int32)

    # What is the pot asking for?  Pure contents name the recipe.  Anything
    # mixed is already unservable, so fill it from the nearest pile to force a
    # cook and free the pot again.
    pure1 = (pot_counts[1] > 0) & (pot_counts[0] == 0) & (pot_counts[2] == 0)
    target = jnp.where(pure1, 1, 0).astype(jnp.int32)

    # --- candidate actions -------------------------------------------------
    idle = _go(pos, face, A1_IDLE, LEFT, False)
    to_pot = _go(pos, face, A1_IDLE, LEFT, True)
    to_deliver = _go(pos, face, A1_DELIVER_STAND, RIGHT, True)

    plate_stand = jnp.asarray(A1_PLATE_STAND, dtype=jnp.int32)
    nearest_pile = plate_stand[jnp.argmin(jnp.stack(
        [_manhattan(pos, plate_stand[0]), _manhattan(pos, plate_stand[1])]))]
    plate_slot = jnp.argmax(park_plate)
    to_plate = jnp.where(jnp.any(park_plate),
                         _go(pos, face, park_stand[plate_slot], park_need[plate_slot], True),
                         _go(pos, face, nearest_pile, RIGHT, True))

    park_slot = jnp.argmax(park_free)
    to_park = _go(pos, face, park_stand[park_slot], park_need[park_slot], True)

    # Reclaim a previously parked ingredient before walking to a pile, so the
    # private slots stay available.
    own_match = park_raw & (park_code == target)
    shared_match = cnt_raw & (cnt_code == target)
    from_pile = _go(pos, face, jnp.asarray(A1_SUP_STAND)[target],
                    jnp.asarray(A1_SUP_NEED)[target], True)
    take_match = jnp.where(
        jnp.any(own_match),
        _go(pos, face, park_stand[jnp.argmax(own_match)],
            park_need[jnp.argmax(own_match)], True),
        jnp.where(jnp.any(shared_match),
                  _go(pos, face, cnt_stand[jnp.argmax(shared_match)], LEFT, True),
                  from_pile))
    match = jnp.any(own_match) | jnp.any(shared_match)

    # Blind hand-off: exactly one shared counter holds a raw ingredient (we never
    # put anything there ourselves), it appeared after our own last delivery so
    # it cannot have been chosen under the previous recipe, and the pot has
    # stayed empty long enough that this is a deliberate offer, not a transient.
    offer = cnt_raw
    offer_slot = jnp.argmax(offer)
    # A stale offer (placed before our last delivery, so possibly chosen under
    # the previous recipe) is still taken eventually: for a partner that only
    # passes over counters, refusing it forever deadlocks the one channel it has.
    fresh = seen_empty[offer_slot] > 0
    patient = mem[0] >= jnp.where(fresh, _HANDOFF_PATIENCE, _STALE_PATIENCE)
    lone_offer = ((jnp.sum(offer.astype(jnp.int32)) == 1) & patient
                  & (partner_uses_pot == 0))
    take_offer = _go(pos, face, cnt_stand[offer_slot], LEFT, True)

    # A dish that cannot match any recipe (mixed, or holding ingredient 2) is
    # parked rather than delivered: parking is free, delivering costs 20.
    bad_dish = (jnp.sum((held_counts > 0).astype(jnp.int32)) > 1) | (held_counts[2] > 0)
    # A plate is only useful once something is cooking.
    stale_plate = (pot_total == 0) & (~pot_busy) & (mem[0] >= _PLATE_PATIENCE) & can_park

    # --- priority ladder (later assignments win) ---------------------------
    empty_handed = (~holds_ing) & (~holds_plate) & (~holds_dish)
    action = idle
    action = jnp.where(empty_handed & (pot_total == 0) & lone_offer, take_offer, action)
    action = jnp.where(empty_handed & (pot_total > 0) & pot_room & pot_workable,
                       jnp.where(match, take_match, from_pile), action)
    action = jnp.where(empty_handed & pot_busy & pot_workable, to_plate, action)
    action = jnp.where(holds_ing,
                       jnp.where(pot_room & pot_workable, to_pot,
                                 jnp.where(can_park, to_park, idle)), action)
    action = jnp.where(holds_plate,
                       jnp.where(stale_plate | (pot_busy & (~pot_workable) & can_park),
                                 to_park, to_pot),
                       action)
    action = jnp.where(holds_dish, jnp.where(bad_dish & can_park, to_park, to_deliver), action)

    # --- memory ------------------------------------------------------------
    streak = jnp.where((pot_total == 0) & (pot_cooked == 0) & (pot_timer == 0),
                       jnp.minimum(mem[0] + 1, 1000), 0)
    delivering = ((action == INTERACT) & (pos == A1_DELIVER_STAND) & (face == RIGHT)
                  & holds_dish)
    good = jnp.minimum(mem[3] + (delivering & (~bad_dish)).astype(jnp.int32), 1000)
    bad = jnp.minimum(mem[4] + (delivering & bad_dish).astype(jnp.int32), 1000)
    seen_empty = jnp.where(delivering, 0, seen_empty)
    deposited = ((action == INTERACT) & (pos == A1_IDLE) & (face == LEFT)
                 & holds_ing & pot_room)

    new_mem = jnp.concatenate([jnp.stack([streak.astype(jnp.int32)]),
                               seen_empty.astype(jnp.int32),
                               jnp.stack([good.astype(jnp.int32), bad.astype(jnp.int32),
                                          pot_total.astype(jnp.int32),
                                          deposited.astype(jnp.int32),
                                          partner_uses_pot.astype(jnp.int32)])])
    trace = {
        "pos": pos.astype(jnp.int32),
        "target": target.astype(jnp.int32),
        "pot_total": pot_total.astype(jnp.int32),
        "held": held_id.astype(jnp.int32),
        "action": action.astype(jnp.int32),
    }
    return action.astype(jnp.int32), new_mem, trace
