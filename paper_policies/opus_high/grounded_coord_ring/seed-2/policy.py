"""grounded_coord_ring policy: the inner seat cooks, the outer seat is a recipe beacon.

Layout facts that drive the design (context/environment/layout.md):

* agent_1 is locked inside the 8-cell ring around the pot.  From there it can
  reach ingredient supply 0, ingredient supply 1, both plate piles, the pot,
  both delivery stations and both recipe buttons, so it can run the entire
  cook-and-deliver loop alone.  It can never see an always-on recipe indicator
  ``R`` (all four sit on the outer border), so the only ground truth it can get
  about the target recipe is a currently active button display.
* agent_0 is locked in the outer corridor and can never reach the pot, so it
  cannot cook at all.  It does permanently see an ``R`` indicator.  Its one
  useful contribution is to act as a recipe beacon: pay the 5-point button
  charge when the recipe changes so the inner seat need not spend time on it.

Neither seat depends on the partner following a private convention.  The inner
seat presses the button itself whenever nothing is displayed, and the outer
seat withholds a charge whenever it can see that the recipe is already
displayed or that the partner is already carrying something.
"""

from __future__ import annotations

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'ring_cook_beacon',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}

# Actions and the facing id each movement action leaves behind
# (facing ids: UP=0, DOWN=1, RIGHT=2, LEFT=3).
_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5
_ACT_FACING = (2, 1, 3, 0)

# ---------------------------------------------------------------------------
# Inner ring (agent_1).  Cell ids:
#   0=(3,3) 1=(3,4) 2=(3,5) 3=(4,3) 4=(4,5) 5=(5,3) 6=(5,4) 7=(5,5)
# The pot at (4,4) is inside the 5x5 view from every one of them, so the pot's
# local position localizes the agent exactly.
# ---------------------------------------------------------------------------
_INNER_LIN = (0, 1, 2, 3, 3, 4, 5, 6, 7)  # (row-3)*3+(col-3) -> cell id
_INNER_STEP = (
    (4, 0, 0, 1, 0, 1, 1, 0),
    (2, 4, 0, 2, 0, 2, 0, 0),
    (2, 2, 4, 2, 1, 1, 1, 1),
    (3, 3, 3, 4, 1, 1, 1, 1),
    (3, 3, 3, 1, 4, 1, 1, 1),
    (3, 3, 0, 3, 0, 4, 0, 0),
    (2, 0, 0, 2, 0, 2, 4, 0),
    (2, 3, 3, 2, 3, 2, 2, 4),
)
_DIST_TOP_BUTTON = (1, 0, 1, 2, 2, 3, 4, 3)
_DIST_BOTTOM_BUTTON = (3, 4, 3, 2, 2, 1, 0, 1)

_HUB = 3              # (4,3): touches the pot, delivery (4,2), and sees both buttons
_HUB_POT_FACE = 0     # RIGHT, towards the pot (4,4)
_DELIVER_CELL = 3
_DELIVER_FACE = 2     # LEFT, towards delivery station (4,2)
_TOP_PRESS = 1        # (3,4), faces button (2,4)
_BOTTOM_PRESS = 6     # (5,4), faces button (6,4)
# Lane 0 works ingredient 0 in the top half, lane 1 works ingredient 1 below.
_SUPPLY_CELL = (0, 5)
_SUPPLY_FACE = (2, 2)
_POT_CELL = (1, 6)
_POT_FACE = (1, 3)
_PLATE_CELL = (2, 7)
_PLATE_FACE = (0, 0)
_SPARE_CELL = (0, 5)
_SPARE_FACE = (3, 1)

# ---------------------------------------------------------------------------
# Outer corridor (agent_0).  Cells in table order:
#   (1,1..7), (2,1),(2,7),(3,1),(3,7),(4,1),(4,7),(5,1),(5,7),(6,1),(6,7),
#   (7,1..7).  Index 3 is the camp (1,4), which is adjacent to button (2,4) and
#   sees the always-on indicator (0,4).
# Localization uses per-static-channel count / row moment / column moment of the
# 5x5 crop, which is unique across those 24 cells.
# ---------------------------------------------------------------------------
_CAMP = 3
_CAMP_FACE = 1  # DOWN, towards button (2,4)
_OUTER_TO_CAMP = (0, 0, 0, 4, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0, 0, 3)
# Mirror camp (7,4) beside button (6,4); it also sees indicator (8,4).
_CAMP2 = 20
_CAMP2_FACE = 3  # UP, towards button (6,4)
_OUTER_TO_CAMP2 = (1, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 4, 2, 2, 2)
_OUTER_SIG = (
    (7, 0, 0, 0, 0, 0, 1, 0, 2, 14, 0, 0, 0, 0, 0, 4, 0, 5, 15, 0, 0, 0, 0, 0, 3, 0, 5),
    (7, 0, 0, 1, 1, 0, 1, 0, 2, 14, 0, 0, 1, 3, 0, 4, 0, 5, 8, 0, 0, 4, 4, 0, 2, 0, 3),
    (5, 0, 0, 1, 1, 0, 1, 0, 2, 11, 0, 0, 1, 3, 0, 4, 0, 2, 8, 0, 0, 3, 3, 0, 1, 0, 6),
    (6, 0, 0, 1, 1, 1, 1, 0, 2, 14, 0, 0, 1, 3, 4, 4, 0, 2, 12, 0, 0, 2, 2, 4, 0, 0, 4),
    (5, 0, 0, 1, 1, 1, 0, 0, 2, 11, 0, 0, 1, 3, 4, 0, 0, 2, 12, 0, 0, 1, 1, 3, 0, 0, 2),
    (7, 0, 0, 1, 1, 1, 0, 0, 2, 14, 0, 0, 1, 3, 4, 0, 0, 5, 20, 0, 0, 0, 0, 2, 0, 0, 5),
    (7, 0, 0, 0, 0, 1, 0, 0, 2, 14, 0, 0, 0, 0, 4, 0, 0, 5, 13, 0, 0, 0, 0, 1, 0, 0, 3),
    (7, 1, 0, 1, 0, 0, 1, 0, 2, 7, 4, 0, 4, 0, 0, 3, 0, 3, 15, 3, 0, 1, 0, 0, 3, 0, 5),
    (7, 1, 0, 1, 0, 1, 0, 0, 2, 7, 4, 0, 4, 0, 3, 0, 0, 3, 13, 1, 0, 3, 0, 1, 0, 0, 3),
    (4, 1, 0, 1, 0, 0, 1, 1, 2, 3, 3, 0, 3, 0, 0, 2, 4, 6, 9, 3, 0, 1, 0, 0, 3, 3, 2),
    (4, 1, 0, 1, 0, 2, 0, 0, 2, 3, 3, 0, 3, 0, 6, 0, 0, 6, 7, 1, 0, 3, 0, 2, 0, 0, 6),
    (6, 1, 0, 1, 0, 0, 1, 1, 2, 12, 2, 0, 2, 0, 0, 1, 3, 4, 16, 3, 0, 1, 0, 0, 3, 3, 2),
    (6, 1, 0, 1, 0, 2, 0, 0, 2, 12, 2, 0, 2, 0, 4, 0, 0, 4, 8, 1, 0, 3, 0, 2, 0, 0, 6),
    (4, 1, 0, 1, 0, 0, 1, 1, 2, 13, 1, 0, 1, 0, 0, 0, 2, 2, 9, 3, 0, 1, 0, 0, 3, 3, 2),
    (4, 1, 0, 1, 0, 2, 0, 0, 2, 13, 1, 0, 1, 0, 2, 0, 0, 2, 7, 1, 0, 3, 0, 2, 0, 0, 6),
    (7, 1, 0, 1, 0, 0, 0, 1, 2, 21, 0, 0, 0, 0, 0, 0, 1, 5, 15, 3, 0, 1, 0, 0, 0, 3, 5),
    (7, 1, 0, 1, 0, 1, 0, 0, 2, 21, 0, 0, 0, 0, 1, 0, 0, 5, 13, 1, 0, 3, 0, 1, 0, 0, 3),
    (7, 0, 0, 0, 0, 0, 0, 1, 2, 14, 0, 0, 0, 0, 0, 0, 0, 3, 15, 0, 0, 0, 0, 0, 0, 3, 5),
    (7, 0, 0, 1, 1, 0, 0, 1, 2, 14, 0, 0, 3, 1, 0, 0, 0, 3, 8, 0, 0, 4, 4, 0, 0, 2, 3),
    (5, 0, 0, 1, 1, 0, 0, 1, 2, 9, 0, 0, 3, 1, 0, 0, 0, 6, 8, 0, 0, 3, 3, 0, 0, 1, 6),
    (6, 0, 0, 1, 1, 1, 0, 1, 2, 10, 0, 0, 3, 1, 0, 0, 0, 6, 12, 0, 0, 2, 2, 4, 0, 0, 4),
    (5, 0, 0, 1, 1, 1, 0, 0, 2, 9, 0, 0, 3, 1, 0, 0, 0, 6, 12, 0, 0, 1, 1, 3, 0, 0, 2),
    (7, 0, 0, 1, 1, 1, 0, 0, 2, 14, 0, 0, 3, 1, 0, 0, 0, 3, 20, 0, 0, 0, 0, 2, 0, 0, 5),
    (7, 0, 0, 0, 0, 1, 0, 0, 2, 14, 0, 0, 0, 0, 0, 0, 0, 3, 13, 0, 0, 0, 0, 1, 0, 0, 3),
)

_INNER_HUB_WAIT = 0       # steps spent at the hub hoping a partner lights a button
_OUTER_RELIGHT = 28     # dark steps before re-lighting for a known-fast partner
_OUTER_FAST_CYCLE = 46    # longest measured cycle that the relight timer suits
_MIN_CYCLE = 28           # shorter intervals cannot be a whole cook cycle
_OUTER_RESCUE = 45        # dark steps before an insurance charge for a silent partner
_OUTER_MAX_FRUITLESS = 1  # unanswered charges before trying the other button
_OUTER_MAX_RESCUES = 5    # total insurance charges, so an inert partner costs 25
# A charge can only pay for itself if a full cook cycle (15 loading + 20
# cooking + delivery) still fits in the 400 step episode.
_LAST_USEFUL_PRESS = 361


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------

def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _displayed_recipe(obs, mask):
    """Ingredient counts of any recipe display sitting on a ``mask`` cell."""
    shown = (obs[:, :, 34:39].sum(axis=-1) * mask > 0)[..., None]
    return jnp.max(jnp.where(shown, obs[:, :, 36:39], 0), axis=(0, 1))


def _goto(cell, facing, target_cell, face_action, step_table):
    """Walk to ``target_cell``; once there face the object and INTERACT."""
    move = step_table[cell, target_cell]
    want = jnp.asarray(_ACT_FACING, dtype=jnp.int32)[face_action]
    return jnp.where(
        cell == target_cell,
        jnp.where(facing == want, _INTERACT, face_action),
        move,
    )


# ---------------------------------------------------------------------------
# agent_0 - outer corridor recipe beacon.
# ---------------------------------------------------------------------------

_OUTER_INIT = (-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0)


def init_agent_0():
    # [recipe, armed, dark, since change, shortest gap, seen change, step,
    #  unanswered charges at this camp, camp side, insurance charges spent,
    #  steps since the display last came on, display was on, lightings seen]
    return jnp.asarray(_OUTER_INIT, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    memory = jnp.where(reset, jnp.asarray(_OUTER_INIT, dtype=jnp.int32), memory)
    last_recipe, armed, since_lit = memory[0], memory[1], memory[2]
    since_change, min_gap, seen_change, step = memory[3], memory[4], memory[5], memory[6]
    fruitless, side, spent = memory[7], memory[8], memory[9]
    since_rise, was_lit, rises_seen = memory[10], memory[11], memory[12]

    static = obs[:, :, 20:29]
    axis = jnp.arange(5, dtype=jnp.int32)
    feature = jnp.concatenate([
        static.sum(axis=(0, 1)),
        (static * axis[:, None, None]).sum(axis=(0, 1)),
        (static * axis[None, :, None]).sum(axis=(0, 1)),
    ])
    table = jnp.asarray(_OUTER_SIG, dtype=jnp.int32)
    cell = jnp.argmin(jnp.abs(table - feature[None]).sum(axis=1)).astype(jnp.int32)
    facing = _facing(obs)

    # Ground truth from the always-on indicator, visible from the corridor only.
    board = _displayed_recipe(obs, obs[:, :, 23])
    board_seen = board.sum() > 0
    board_target = jnp.where(board_seen, jnp.argmax(board).astype(jnp.int32), -1)

    # A lit button already carries the current recipe, so no charge is needed.
    button_lit = _displayed_recipe(obs, obs[:, :, 24]).sum() > 0
    partner_carrying_raw = obs[:, :, 17:20].sum() > 0   # partner is still loading the pot
    partner_busy = obs[:, :, 15:20].sum() > 0

    # A changed indicator means the partner just completed a correct delivery,
    # i.e. it is starting a fresh cycle and needs the new recipe right now.
    # This is the only cycle boundary the corridor can actually observe, and a
    # charge paid here is always perfectly timed.
    changed = board_seen & (board_target != last_recipe) & (last_recipe >= 0)
    last_recipe = jnp.where(board_seen, board_target, last_recipe)

    # Intervals between changes are whole numbers of the partner's cook cycle,
    # so the shortest one bounds that cycle from above.  Intervals between
    # button lightings measure the same thing and are available much sooner,
    # because a cycle needs its recipe whether or not the recipe changed.
    measurable = changed & (seen_change > 0) & (since_change >= _MIN_CYCLE)
    min_gap = jnp.where(
        measurable,
        jnp.where(min_gap > 0, jnp.minimum(min_gap, since_change), since_change),
        min_gap,
    )
    seen_change = jnp.maximum(seen_change, changed.astype(jnp.int32))
    since_change = jnp.where(changed, 0, since_change + 1)

    rising = button_lit & (was_lit == 0)
    lit_gap = rising & (rises_seen > 0) & (since_rise >= _MIN_CYCLE) & (since_rise <= 140)
    min_gap = jnp.where(
        lit_gap,
        jnp.where(min_gap > 0, jnp.minimum(min_gap, since_rise), since_rise),
        min_gap,
    )
    rises_seen = jnp.maximum(rises_seen, rising.astype(jnp.int32))
    since_rise = jnp.where(rising, 0, since_rise + 1)
    was_lit = button_lit.astype(jnp.int32)

    # A partner known to deliver about every 40 steps needs the recipe again
    # roughly a display-lifetime after the last one went dark; re-lighting then
    # keeps it from paying for a charge itself.  Skip it while the partner is
    # visibly still loading the pot, since it cannot need a new recipe yet.
    fast_partner = (min_gap > 0) & (min_gap <= _OUTER_FAST_CYCLE)
    relight = (fast_partner & (since_lit >= _OUTER_RELIGHT)
               & jnp.logical_not(partner_carrying_raw))
    # Insurance only: if nobody has lit a button for a long time the partner is
    # probably waiting for information it cannot get by itself.  Give up after a
    # few charges that produce neither a delivery nor a partner carrying
    # anything, so an inert partner cannot drain the team score.
    progress = changed | partner_busy
    fruitless = jnp.where(progress, 0, fruitless)
    rescue = ((since_lit >= _OUTER_RESCUE) & jnp.logical_not(partner_busy) & board_seen
              & (spent < _OUTER_MAX_RESCUES))
    armed = jnp.where(changed | relight | rescue, 1, armed)
    armed = jnp.where(button_lit, 0, armed)
    since_lit = jnp.where(button_lit, 0, since_lit + 1)

    # Only one of the two buttons is reachable from a camp, and a partner parked
    # in the far half of the ring cannot see the one being lit.  Before spending
    # another charge that the previous one has not been answered, move to the
    # mirror camp and light the other button instead.
    swap = rescue & (fruitless >= _OUTER_MAX_FRUITLESS)
    side = jnp.where(swap, 1 - side, side)
    fruitless = jnp.where(swap, 0, fruitless)

    at_bottom = side > 0
    camp = jnp.where(at_bottom, _CAMP2, _CAMP)
    camp_face = jnp.where(at_bottom, _CAMP2_FACE, _CAMP_FACE)
    route = jnp.where(
        at_bottom,
        jnp.asarray(_OUTER_TO_CAMP2, dtype=jnp.int32)[cell],
        jnp.asarray(_OUTER_TO_CAMP, dtype=jnp.int32)[cell],
    )

    at_camp = cell == camp
    aimed = facing == jnp.asarray(_ACT_FACING, dtype=jnp.int32)[camp_face]
    press = (armed > 0) & at_camp & aimed & (step <= _LAST_USEFUL_PRESS)

    action = jnp.where(
        at_camp,
        jnp.where(aimed, jnp.where(press, _INTERACT, _STAY), camp_face),
        route,
    )

    armed = jnp.where(press, 0, armed)
    since_lit = jnp.where(press, 0, since_lit)
    fruitless = jnp.where(press & rescue, fruitless + 1, fruitless)
    spent = jnp.where(press & rescue, spent + 1, spent)
    next_memory = jnp.stack([
        last_recipe.astype(jnp.int32),
        armed.astype(jnp.int32),
        since_lit.astype(jnp.int32),
        since_change.astype(jnp.int32),
        min_gap.astype(jnp.int32),
        seen_change.astype(jnp.int32),
        (step + 1).astype(jnp.int32),
        fruitless.astype(jnp.int32),
        side.astype(jnp.int32),
        spent.astype(jnp.int32),
        since_rise.astype(jnp.int32),
        was_lit.astype(jnp.int32),
        rises_seen.astype(jnp.int32),
    ])
    trace = {
        'cell': cell,
        'recipe': last_recipe.astype(jnp.int32),
        'min_gap': min_gap.astype(jnp.int32),
        'side': side.astype(jnp.int32),
        'spent': spent.astype(jnp.int32),
        'press': press.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


# ---------------------------------------------------------------------------
# agent_1 - inner ring cook-and-deliver loop.
# ---------------------------------------------------------------------------

def init_agent_1():
    # [target ingredient id (-1 unknown), steps waited for a display, step]
    return jnp.asarray([-1, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    memory = jnp.where(reset, jnp.asarray([-1, 0, 0], dtype=jnp.int32), memory)
    target, hub_wait, step = memory[0], memory[1], memory[2]

    # The pot is always inside the inner view, so it pins down the global cell.
    pot_mask = obs[:, :, 22]
    axis = jnp.arange(5, dtype=jnp.int32)
    pot_row = (pot_mask * axis[:, None]).sum()
    pot_col = (pot_mask * axis[None, :]).sum()
    row, col = 6 - pot_row, 6 - pot_col
    linear = jnp.clip((row - 3) * 3 + (col - 3), 0, 8)
    cell = jnp.asarray(_INNER_LIN, dtype=jnp.int32)[linear]
    facing = _facing(obs)
    step_table = jnp.asarray(_INNER_STEP, dtype=jnp.int32)

    # Only an active button can display a recipe in here, and it always shows
    # the current one, so any display is ground truth.
    shown = _displayed_recipe(obs, jnp.ones((5, 5), dtype=jnp.int32))
    target = jnp.where(shown.sum() > 0, jnp.argmax(shown).astype(jnp.int32), target)
    known = target >= 0
    lane = jnp.where(target == 1, 1, 0)

    inventory = obs[2, 2, 5:10]
    holding_dish = (inventory[0] > 0) & (inventory[1] > 0)
    holding_plate = (inventory[0] > 0) & (inventory[1] == 0)
    hands_empty = inventory.sum() == 0
    holds_target = known & (inventory[2:5][jnp.clip(target, 0, 2)] > 0)

    pot_ready = (obs[:, :, 30] * pot_mask).sum() > 0
    pot_timer = (obs[:, :, 39] * pot_mask).sum()
    pot_busy = pot_ready | (pot_timer > 0)

    press_cell = jnp.where(
        jnp.asarray(_DIST_TOP_BUTTON, dtype=jnp.int32)[cell]
        <= jnp.asarray(_DIST_BOTTOM_BUTTON, dtype=jnp.int32)[cell],
        _TOP_PRESS,
        _BOTTOM_PRESS,
    )
    press_face = jnp.where(press_cell == _TOP_PRESS, _UP, _DOWN)

    supply_cell = jnp.asarray(_SUPPLY_CELL, dtype=jnp.int32)[lane]
    supply_face = jnp.asarray(_SUPPLY_FACE, dtype=jnp.int32)[lane]
    pot_cell = jnp.asarray(_POT_CELL, dtype=jnp.int32)[lane]
    pot_face = jnp.asarray(_POT_FACE, dtype=jnp.int32)[lane]
    plate_cell = jnp.asarray(_PLATE_CELL, dtype=jnp.int32)[lane]
    plate_face = jnp.asarray(_PLATE_FACE, dtype=jnp.int32)[lane]
    spare_cell = jnp.asarray(_SPARE_CELL, dtype=jnp.int32)[lane]
    spare_face = jnp.asarray(_SPARE_FACE, dtype=jnp.int32)[lane]

    to_deliver = _goto(cell, facing, _DELIVER_CELL, _DELIVER_FACE, step_table)
    to_pot = _goto(cell, facing, pot_cell, pot_face, step_table)
    # The hub touches both the pot and a delivery station, so scooping there
    # saves two steps per cycle compared with scooping beside the supply.
    to_scoop = _goto(cell, facing, _HUB, _HUB_POT_FACE, step_table)
    to_plate = _goto(cell, facing, plate_cell, plate_face, step_table)
    to_supply = _goto(cell, facing, supply_cell, supply_face, step_table)
    to_spare = _goto(cell, facing, spare_cell, spare_face, step_table)
    to_button = _goto(cell, facing, press_cell, press_face, step_table)

    # Leave a beacon partner one step to light a button before paying for it.
    hold_at_hub = (cell == _HUB) & (hub_wait < _INNER_HUB_WAIT)
    seek_recipe = jnp.where(hands_empty, jnp.where(hold_at_hub, _STAY, to_button), to_spare)
    # Too late in the episode for another cycle to pay for a button charge.
    seek_recipe = jnp.where(step > _LAST_USEFUL_PRESS, _STAY, seek_recipe)
    # Pot loaded: fetch a plate, then wait at the hub and keep trying to scoop.
    while_cooking = jnp.where(holding_plate, to_scoop, jnp.where(hands_empty, to_plate, to_spare))
    # Pot idle: load three units of the target ingredient.
    while_loading = jnp.where(holds_target, to_pot, jnp.where(hands_empty, to_supply, to_spare))

    action = jnp.where(
        holding_dish,
        to_deliver,
        jnp.where(pot_busy, while_cooking, jnp.where(known, while_loading, seek_recipe)),
    )

    delivering = holding_dish & (action == _INTERACT) & (cell == _DELIVER_CELL)
    next_memory = jnp.stack([
        jnp.where(delivering, -1, target).astype(jnp.int32),
        jnp.where(known, 0, hub_wait + 1).astype(jnp.int32),
        (step + 1).astype(jnp.int32),
    ])
    trace = {
        'cell': cell,
        'target': target.astype(jnp.int32),
        'pot_timer': pot_timer.astype(jnp.int32),
        'action': action.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
