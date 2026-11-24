"""Grounded-coordination policy for the `grounded_coord_simple` Overcooked-V2 layout.

Board (row, column), rows increase downward::

    WW2WWWWW
    W  WB  0
    R ALPA X
    W  WB  1
    WW2WWWWW

The two floor regions are disconnected.  The left region (rows 1-3, columns 1-2)
only touches the always-on recipe indicator ``R`` at (2, 0), the recipe button
``L`` at (2, 3) and the useless ingredient-2 piles.  The right region (rows 1-3,
columns 5-6) touches the pot (2, 4), both plate piles, ingredient 0 at (1, 7),
ingredient 1 at (3, 7) and the delivery station (2, 7), but can never see ``R``.

So the only viable division of labour is: the left seat reads the recipe and
publishes it by activating the button, the right seat cooks and delivers.  The
button is a *grounded* channel -- while active it displays the literal counts of
the current target recipe -- so this protocol needs no private convention and
works with partners that were generated independently.

Both seats run the same controller; the room is detected from the observation.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_button_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# --- action ids -------------------------------------------------------------
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5

# Facing order used below: 0=up, 1=down, 2=right, 3=left.  Observation channel
# for facing f is 1 + f; the movement action that produces facing f is
# _FACE_ACTION[f] (built inside functions; module-scope calls are not allowed).
_FACE_ACTION_IDS = (A_UP, A_DOWN, A_RIGHT, A_LEFT)
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3

# --- channels ---------------------------------------------------------------
C_SELF_PLATE, C_SELF_COOKED, C_SELF_ING = 5, 6, 7
C_POT, C_RECIPE_IND = 22, 23
C_ITEM_PLATE, C_ITEM_ING = 29, 31
C_DISP_ING = 36
C_POT_TIMER = 39

# --- tuning -----------------------------------------------------------------
PRESS_BUDGET = 16       # hard cap on button activations (-5 reward each)
PRESSES_PER_GAP = 1     # activations per idle-pot stretch, before backing off
RETRY_AFTER = 30        # ... unless the pot stays empty this long afterwards
IDLE_PRESS_GAP = 50     # re-press gap when the partner never uses the pot
SETTLE_STEPS = 4        # let a pending delivery resample the recipe first
LAST_USEFUL_STEP = 360  # no cook-and-deliver cycle can finish after this
WAIT_STEPS = 22         # steps to wait in column 5 for a signal before guessing
WAIT_STEPS_SHORT = 5    # ... once the partner looks like a non-signaller


def _zero():
    return jnp.zeros((), dtype=jnp.int32)


def _init_memory():
    return {
        'known': _zero(),        # 1 once the button has shown us the recipe
        'rtype': _zero(),        # ingredient id of the known recipe
        'alt': _zero(),          # guess to use when uninformed (alternates)
        'wait': _zero(),         # steps spent waiting for a signal this cycle
        'streak': _zero(),       # consecutive uninformed cycles
        'presses': _zero(),      # button activations issued (left seat)
        'since': _zero(),        # steps since the last activation
        'gap': _zero(),          # activations since the pot was last in use
        'lastr': -jnp.ones((), dtype=jnp.int32),  # recipe last shown at R
        'confirm': _zero(),      # R changed, so a delivery has already landed
        'empty': jnp.asarray(SETTLE_STEPS, dtype=jnp.int32),  # steps pot seen empty
        'step': _zero(),         # steps elapsed in the episode
    }


def _reset_memory(memory, reset):
    fresh = _init_memory()
    flag = jnp.asarray(reset, dtype=jnp.bool_).reshape(())
    return {k: jnp.where(flag, fresh[k], memory[k].astype(jnp.int32))
            for k in fresh}


def _locate(plane):
    """Local (row, col) of the unique cell flagged in `plane`, plus a found bit."""
    flat = plane.reshape(-1)
    idx = jnp.argmax(flat)
    return idx // 5, idx % 5, flat[idx] > 0


def _cell(obs, row, col, self_row, self_col):
    """Observation vector of global cell (row, col), and whether it is in view."""
    i = row - self_row + 2
    j = col - self_col + 2
    visible = (i >= 0) & (i <= 4) & (j >= 0) & (j <= 4)
    return obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4)] * visible, visible


def _approach(obs, row, col, target_row, target_col, facing, arrived_action):
    """Walk to (target_row, target_col), turn to `facing`, then act.

    Both floor regions are solid rectangles, so fixing the row first and the
    column second never crosses an obstacle.  Every interaction target is a
    blocking object, so the turning move only changes facing.
    """
    row_move = jnp.where(target_row > row, A_DOWN, A_UP)
    col_move = jnp.where(target_col > col, A_RIGHT, A_LEFT)
    face_actions = jnp.asarray(_FACE_ACTION_IDS, dtype=jnp.int32)
    facing_index = jnp.asarray(facing, dtype=jnp.int32)
    facing_ok = obs[2, 2, 1 + facing_index] > 0
    on_cell = jnp.where(facing_ok, arrived_action, face_actions[facing_index])
    return jnp.where(row != target_row, row_move,
                     jnp.where(col != target_col, col_move, on_cell))


# ---------------------------------------------------------------------------
# Left region: read the recipe, publish it on the button.
# ---------------------------------------------------------------------------
def _left_seat(obs, mem):
    ri, rj, _ = _locate(obs[:, :, C_RECIPE_IND])
    row = 4 - ri
    col = 2 - rj

    pot, pot_seen = _cell(obs, 2, 4, row, col)
    pot_empty = (pot[C_ITEM_ING:C_ITEM_ING + 3].sum() == 0) & (pot[C_ITEM_PLATE + 1] == 0) & (pot[C_POT_TIMER] == 0)
    button, _ = _cell(obs, 2, 3, row, col)
    button_active = button[C_DISP_ING:C_DISP_ING + 3].sum() > 0

    at_post = (row == 2) & (col == 2)
    facing_button = obs[2, 2, 1 + F_RIGHT] > 0

    # The always-on indicator is always in view from this side of the map, and
    # the recipe only resamples on a correct delivery, so a change of what it
    # displays is proof that the partner's delivery has already landed.
    indicator, _ = _cell(obs, 2, 0, row, col)
    shown = jnp.argmax(indicator[C_DISP_ING:C_DISP_ING + 3]).astype(jnp.int32)
    changed = (mem['lastr'] >= 0) & (shown != mem['lastr'])
    confirmed = jnp.maximum(mem['confirm'], changed.astype(jnp.int32)) > 0

    # Without that proof, wait a few steps after the pot empties: the partner is
    # probably still walking to the delivery station, and a display published
    # before the delivery lands can be cached by the partner and go stale.
    settled = (mem['empty'] >= SETTLE_STEPS) | confirmed
    # One activation per idle stretch is enough for a partner that watches the
    # button at all; a second one is only worth its 5 reward once the pot has
    # stayed empty long enough that the partner looks stuck or inattentive.
    retry = (mem['gap'] < PRESSES_PER_GAP + 1) & (mem['since'] >= RETRY_AFTER)
    fresh_gap = mem['gap'] < PRESSES_PER_GAP
    stale = mem['since'] >= IDLE_PRESS_GAP
    in_time = mem['step'] < LAST_USEFUL_STEP
    press = (at_post & facing_button & pot_seen & pot_empty & settled
             & (~button_active) & in_time
             & (mem['presses'] < PRESS_BUDGET) & (fresh_gap | retry | stale))

    action = _approach(obs, row, col, 2, 2, F_RIGHT,
                       jnp.where(press, A_INTERACT, A_STAY))

    saw_use = pot_seen & (~pot_empty)
    new = dict(mem)
    new['presses'] = mem['presses'] + press.astype(jnp.int32)
    new['since'] = jnp.where(press, 0, jnp.minimum(mem['since'] + 1, 10000))
    new['gap'] = jnp.where(saw_use, 0, mem['gap'] + press.astype(jnp.int32))
    new['empty'] = jnp.where(pot_seen,
                             jnp.where(pot_empty, jnp.minimum(mem['empty'] + 1, 100), 0),
                             mem['empty'])
    new['lastr'] = shown
    new['confirm'] = jnp.where(press, 0, confirmed.astype(jnp.int32))
    new['step'] = mem['step'] + 1
    trace = {
        'room': jnp.asarray(0, dtype=jnp.int32),
        'row': row.astype(jnp.int32),
        'col': col.astype(jnp.int32),
        'act_now': press.astype(jnp.int32),       # left: button activation
        'counter': new['presses'].astype(jnp.int32),  # left: activations so far
        'known': mem['known'].astype(jnp.int32),
        'rtype': mem['rtype'].astype(jnp.int32),
        'wait': mem['wait'].astype(jnp.int32),
    }
    return action.astype(jnp.int32), new, trace


# ---------------------------------------------------------------------------
# Right region: gather, cook, plate, deliver.
# ---------------------------------------------------------------------------
# counters reachable from the right region, with the cell to stand on and the
# direction to face: counter cell, standing cell, facing.
_DUMP_SPOTS = (((0, 5), (1, 5), F_UP),
               ((4, 5), (3, 5), F_DOWN),
               ((0, 6), (1, 6), F_UP),
               ((4, 6), (3, 6), F_DOWN))


def _right_seat(obs, mem):
    pi, pj, _ = _locate(obs[:, :, C_POT])
    row = 4 - pi
    col = 6 - pj

    # --- inventory ---------------------------------------------------------
    held_plate = obs[2, 2, C_SELF_PLATE] > 0
    held_cooked = obs[2, 2, C_SELF_COOKED] > 0
    held_ing = obs[2, 2, C_SELF_ING:C_SELF_ING + 3]
    held_any_ing = held_ing.sum() > 0
    holding_dish = held_plate & held_cooked
    holding_plate = held_plate & (~held_cooked)
    holding_ing = (~held_plate) & held_any_ing
    empty_hands = (~held_plate) & (~held_any_ing)

    # --- pot ---------------------------------------------------------------
    pot = obs[pi, pj]
    pot_counts = pot[C_ITEM_ING:C_ITEM_ING + 3]
    pot_sum = pot_counts.sum()
    pot_ready = pot[C_ITEM_PLATE + 1] > 0
    pot_cooking = pot[C_POT_TIMER] > 0
    pot_idle = (~pot_ready) & (~pot_cooking)

    # --- the button is visible from column 5 only --------------------------
    button, button_seen = _cell(obs, 2, 3, row, col)
    display = button[C_DISP_ING:C_DISP_ING + 3]
    signal = button_seen & (display.sum() > 0)
    known = jnp.where(signal, 1, mem['known'])
    rtype = jnp.where(signal, jnp.argmax(display).astype(jnp.int32), mem['rtype'])

    # --- which ingredient are we making? -----------------------------------
    # Pot contents are the commitment: once something is in the pot it cannot be
    # taken out, so we always finish a pure batch of whatever is already there.
    guess = mem['alt']
    wanted = jnp.where(known > 0, rtype, guess)
    wanted = jnp.where((pot_sum > 0) & pot_idle, jnp.argmax(pot_counts).astype(jnp.int32), wanted)
    # Ingredient 2's piles are walled off from this room, so a target of 2 could
    # never be satisfied: fall back to 0, cook a mismatched batch and dump it,
    # rather than looping between the supply and a counter forever.
    wanted = jnp.where(wanted > 1, 0, wanted)

    fresh_batch = empty_hands & pot_idle & (pot_sum == 0)
    budget = jnp.where(mem['streak'] >= 2, WAIT_STEPS_SHORT, WAIT_STEPS)
    waiting = fresh_batch & (known == 0) & (mem['wait'] < budget)

    # --- candidate actions -------------------------------------------------
    # free counter to drop something we must not use
    occupied = jnp.zeros((4,), dtype=jnp.bool_)
    seen = jnp.zeros((4,), dtype=jnp.bool_)
    for k in range(4):
        cell, vis = _cell(obs, _DUMP_SPOTS[k][0][0], _DUMP_SPOTS[k][0][1], row, col)
        busy = cell[C_ITEM_PLATE:C_ITEM_PLATE + 5].sum() > 0
        occupied = occupied.at[k].set(busy)
        seen = seen.at[k].set(vis)
    usable = seen & (~occupied)
    choice = jnp.asarray(-1, dtype=jnp.int32)
    for k in range(3, -1, -1):
        choice = jnp.where(usable[k], k, choice)
    have_dump = choice >= 0
    ck = jnp.clip(choice, 0, 3)
    stand_rows = jnp.asarray([spot[1][0] for spot in _DUMP_SPOTS], dtype=jnp.int32)
    stand_cols = jnp.asarray([spot[1][1] for spot in _DUMP_SPOTS], dtype=jnp.int32)
    stand_face = jnp.asarray([spot[2] for spot in _DUMP_SPOTS], dtype=jnp.int32)
    a_dump = _approach(obs, row, col, stand_rows[ck], stand_cols[ck],
                       stand_face[ck], A_INTERACT)

    a_pot = _approach(obs, row, col, 2, 5, F_LEFT, A_INTERACT)
    a_hold = _approach(obs, row, col, 2, 5, F_LEFT, A_STAY)
    a_deliver = _approach(obs, row, col, 2, 6, F_RIGHT, A_INTERACT)
    plate_row = jnp.where(row >= 3, 3, 1)
    a_plate = _approach(obs, row, col, plate_row, 5, F_LEFT, A_INTERACT)
    a_fetch = _approach(obs, row, col, jnp.where(wanted == 1, 3, 1), 6, F_RIGHT, A_INTERACT)

    # --- choose ------------------------------------------------------------
    # Every recipe here is three units of a single ingredient, so "the dish
    # holds three of the required ingredient" is an exact match test: it also
    # rejects a mixed dish, which would otherwise cost 20 at the delivery
    # station.  Dumping such a dish on a counter costs nothing and leaves the
    # (unchanged) recipe to be cooked correctly on the next cycle.
    dish_ok = held_ing[jnp.clip(rtype, 0, 2)] == 3
    wrong_dish = holding_dish & (known > 0) & (~dish_ok)
    dish_action = jnp.where(wrong_dish & have_dump, a_dump, a_deliver)

    ing_type = jnp.argmax(held_ing).astype(jnp.int32)
    keep_ing = (ing_type == wanted) & pot_idle & (pot_sum < 3)
    ing_action = jnp.where(keep_ing | (~have_dump), a_pot, a_dump)

    plate_action = jnp.where(pot_ready, a_pot,
                             jnp.where(pot_cooking, a_hold,
                                       jnp.where(have_dump, a_dump, a_hold)))

    empty_action = jnp.where(pot_ready | pot_cooking, a_plate,
                             jnp.where(waiting, a_hold, a_fetch))

    action = jnp.where(holding_dish, dish_action,
                       jnp.where(holding_plate, plate_action,
                                 jnp.where(holding_ing, ing_action, empty_action)))

    delivered = holding_dish & (~wrong_dish) & (action == A_INTERACT) & (row == 2) & (col == 6)

    new = dict(mem)
    new['step'] = mem['step'] + 1
    new['known'] = jnp.where(delivered, 0, known)
    new['rtype'] = rtype
    new['alt'] = jnp.where(delivered, 1 - mem['alt'], mem['alt'])
    new['wait'] = jnp.where(delivered, 0,
                            jnp.where(waiting & button_seen, mem['wait'] + 1, mem['wait']))
    new['streak'] = jnp.where(signal, 0,
                              jnp.where(delivered & (known == 0), mem['streak'] + 1,
                                        jnp.where(delivered, 0, mem['streak'])))
    trace = {
        'room': jnp.asarray(1, dtype=jnp.int32),
        'row': row.astype(jnp.int32),
        'col': col.astype(jnp.int32),
        'act_now': delivered.astype(jnp.int32),   # right: delivery issued
        'counter': wanted.astype(jnp.int32),      # right: ingredient being made
        'known': known.astype(jnp.int32),
        'rtype': rtype.astype(jnp.int32),
        'wait': new['wait'].astype(jnp.int32),
    }
    return action.astype(jnp.int32), new, trace


def _act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    mem = _reset_memory(memory, reset)
    in_left_room = obs[:, :, C_RECIPE_IND].sum() > 0

    left_action, left_mem, left_trace = _left_seat(obs, mem)
    right_action, right_mem, right_trace = _right_seat(obs, mem)

    action = jnp.where(in_left_room, left_action, right_action)
    new_mem = {k: jnp.where(in_left_room, left_mem[k], right_mem[k]).astype(jnp.int32)
               for k in mem}
    trace = {k: jnp.where(in_left_room, left_trace[k], right_trace[k]).astype(jnp.int32)
             for k in left_trace}
    return action.astype(jnp.int32), new_mem, trace


def init_agent_0():
    return _init_memory()


def agent_0_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)


def init_agent_1():
    return _init_memory()


def agent_1_act(obs, reset, memory, rng):
    return _act(obs, reset, memory, rng)
