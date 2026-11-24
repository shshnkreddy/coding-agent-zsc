"""Two-room grounded-coordination policy for the grounded_coord_simple layout.

Layout recap (row, column), rows downward:

    WW2WWWWW
    W  WB  0
    R ALPA X
    W  WB  1
    WW2WWWWW

The two rooms are disjoint.  Seat A (left) can only reach the always-on recipe
indicator R at (2, 0) and the recipe button L at (2, 3); it can never touch the
pot, plates, supplies of ingredient 0/1 or the delivery station.  Seat B (right)
can do the entire cook-plate-deliver cycle alone but can never see the recipe
unless the button at (2, 3) is active (visible only from column 5).

Therefore the only useful division of labour is physically forced:

  * agent_0 parks at (2, 2) facing the button and activates it when the team
    plainly needs the recipe (pot empty, or pot contents contradicting R),
    rate limited because every activation costs 5 reward.
  * agent_1 runs the cooking loop, reads the recipe off any visible recipe
    display (grounded environment semantics, no private code), and keeps a
    Bayesian belief over the two possible recipes so that it still performs
    above chance when no partner signal ever arrives.

Both seats localise themselves from a unique static landmark inside the 5x5
crop (the pot for agent_1, the button for agent_0), so no dead reckoning or
step counting is required.
"""

from __future__ import annotations

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

# Action ids.
A_RIGHT = 0
A_DOWN = 1
A_LEFT = 2
A_UP = 3
A_STAY = 4
A_INTERACT = 5

# Facing channel produced by each movement action (UP=1, DOWN=2, RIGHT=3, LEFT=4).
_FACE_CH = (3, 2, 4, 1)

# Counter squares agent_1 can deposit on: (counter row, counter col, stand row,
# stand col, facing move).  Ordered by preference.
_DUMP_SPOTS = (
    (0, 5, 1, 5, A_UP),
    (0, 6, 1, 6, A_UP),
    (4, 5, 3, 5, A_DOWN),
    (4, 6, 3, 6, A_DOWN),
)

# Steps agent_1 is willing to idle next to the pot waiting for a recipe signal
# before falling back on its own belief.  Patience is worth a lot while it is
# still unknown whether the partner signals at all (one long listen), a lot once
# a partner has been seen to display a recipe, and nothing at all once listening
# has demonstrably failed: guessing then costs nothing, because the belief
# update still earns 20/3 per delivery on average.
WAIT_FIRST = 35
WAIT_SEEN = 32
WAIT_GAVE_UP = 2
SIGNAL_MEMORY = 70

# agent_0 activation rate limit and per-need-episode budget.  The gap widens to
# the partner's own observed response latency, so a cook that is simply slow to
# start does not collect a redundant (and costly) reminder every cycle.
PRESS_GAP = 13
PRESS_GAP_MAX = 30
LATENCY_MARGIN = 4
PRESS_MAX = 3
# One delivery cycle costs about 41 steps, so a later activation cannot pay for
# itself before the horizon.
PRESS_DEADLINE = 358
# Hard episode budget: ten cycles need ten activations, anything far beyond
# that means the signal is not being used.
PRESS_BUDGET = 13
# Activations that were plainly ignored (partner filled the pot with something
# the indicator contradicts while the display was live) before giving up.
STRIKE_LIMIT = 3
STRIKE_WINDOW = 30
# An unchanging, un-cooking pot for this long means the cook is stalled.
STALL_LIMIT = 25
# Silence after which an active partner gets another reminder.
PRESS_RETRY = 40
# Silence after which even a partner that has never touched the pot gets one:
# cheap insurance against a cook that simply missed the opening activations.
PRESS_RETRY_SILENT = 80


def _int(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _locate(obs, channel):
    """Local (row, col) of the first cell carrying a unique static channel."""
    mask = obs[:, :, channel] > 0
    flat = mask.reshape(-1)
    index = jnp.argmax(flat)
    return (index // 5).astype(jnp.int32), (index % 5).astype(jnp.int32), jnp.any(flat)


def _masked_cell(obs, mask):
    """Full channel vector of the (unique) masked cell, zeros when absent."""
    return jnp.sum(jnp.where(mask[:, :, None], obs, 0), axis=(0, 1))


def _recipe_reading(obs):
    """Ingredient counts shown by any visible recipe display, and a seen flag.

    Both the always-on indicator and an active button always show the *current*
    target recipe, so any reading is fresh by construction.
    """
    display = (obs[:, :, 23] > 0) | (obs[:, :, 24] > 0)
    counts = obs[:, :, 36:39]
    valid = display & (jnp.sum(counts, axis=-1) > 0)
    reading = jnp.max(jnp.where(valid[:, :, None], counts, 0), axis=(0, 1))
    return reading, jnp.any(valid)


def _facing_ok(obs, move):
    channel = jnp.asarray(_FACE_CH, dtype=jnp.int32)[move]
    return obs[2, 2, channel] > 0


def _approach(obs, row, col, target_row, target_col, face):
    """Walk to a cell then turn to face an adjacent blocking object.

    Both rooms are open rectangles, so moving rows first and columns second is
    always collision free.  Turning is done with the movement action itself:
    the target object blocks the step, leaving only the facing change.
    """
    vertical = jnp.where(row > target_row, _int(A_UP), _int(A_DOWN))
    horizontal = jnp.where(col > target_col, _int(A_LEFT), _int(A_RIGHT))
    move = jnp.where(row != target_row, vertical, horizontal)
    at_cell = (row == target_row) & (col == target_col)
    aimed = _facing_ok(obs, face)
    action = jnp.where(at_cell, face, move)
    return action.astype(jnp.int32), at_cell & aimed


def _dump_target(obs, row, col):
    """Pick a visible, empty counter to drop an unwanted item on."""
    oks = []
    for counter_row, counter_col, _, _, _ in _DUMP_SPOTS:
        local_row = counter_row - row + 2
        local_col = counter_col - col + 2
        in_view = (local_row >= 0) & (local_row < 5) & (local_col >= 0) & (local_col < 5)
        cell = obs[jnp.clip(local_row, 0, 4), jnp.clip(local_col, 0, 4)]
        free = (jnp.sum(cell[29:34]) == 0) & (cell[20] > 0)
        oks.append(in_view & free)
    oks = jnp.stack(oks)
    index = jnp.argmax(oks)
    stand_rows = jnp.asarray([spot[2] for spot in _DUMP_SPOTS], dtype=jnp.int32)
    stand_cols = jnp.asarray([spot[3] for spot in _DUMP_SPOTS], dtype=jnp.int32)
    faces = jnp.asarray([spot[4] for spot in _DUMP_SPOTS], dtype=jnp.int32)
    return stand_rows[index], stand_cols[index], faces[index], jnp.any(oks)


# --------------------------------------------------------------------------
# agent_0: recipe signaller in the left room
# --------------------------------------------------------------------------


def init_agent_0():
    return {
        'since': jnp.asarray(99, dtype=jnp.int32),
        'presses': jnp.asarray(0, dtype=jnp.int32),
        'need': jnp.asarray(0, dtype=jnp.int32),
        'clock': jnp.asarray(0, dtype=jnp.int32),
        'spent': jnp.asarray(0, dtype=jnp.int32),
        'strikes': jnp.asarray(0, dtype=jnp.int32),
        'filled': jnp.asarray(0, dtype=jnp.int32),
        'potsig': jnp.asarray(-1, dtype=jnp.int32),
        'stale': jnp.asarray(0, dtype=jnp.int32),
        'active': jnp.asarray(0, dtype=jnp.int32),
        'latency': jnp.asarray(0, dtype=jnp.int32),
        'phase': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    start = init_agent_0()
    since = jnp.where(reset, start['since'], memory['since'])
    presses = jnp.where(reset, start['presses'], memory['presses'])
    prev_need = jnp.where(reset, start['need'], memory['need'])
    clock = jnp.where(reset, start['clock'], memory['clock'])
    spent = jnp.where(reset, start['spent'], memory['spent'])
    strikes = jnp.where(reset, start['strikes'], memory['strikes'])
    prev_filled = jnp.where(reset, start['filled'], memory['filled'])
    prev_sig = jnp.where(reset, start['potsig'], memory['potsig'])
    stale = jnp.where(reset, start['stale'], memory['stale'])
    active = jnp.where(reset, start['active'], memory['active'])
    latency = jnp.where(reset, start['latency'], memory['latency'])
    phase_age = jnp.where(reset, start['phase'], memory['phase'])

    # Self position from the button landmark, which is visible from every cell
    # of the left room.
    button_row, button_col, button_seen = _locate(obs, 24)
    row = jnp.where(button_seen, jnp.clip(4 - button_row, 1, 3), _int(2))
    col = jnp.where(button_seen, jnp.clip(5 - button_col, 1, 2), _int(2))

    pot_mask = obs[:, :, 22] > 0
    pot_seen = jnp.any(pot_mask)
    pot_cell = _masked_cell(obs, pot_mask)
    pot_counts = pot_cell[31:34]
    pot_cooked = pot_cell[30] > 0
    pot_total = jnp.sum(pot_counts)
    pot_timer = pot_cell[39]
    pot_empty = (pot_total == 0) & (~pot_cooked)

    # A pot whose coarse state has not moved for a long time suggests a partner
    # stalled for want of information, even if it is not empty.
    signature = jnp.where(
        pot_seen, pot_total * 4 + pot_cooked.astype(jnp.int32) * 2 + (pot_timer > 0).astype(jnp.int32),
        _int(-1),
    )
    stale_next = jnp.where(signature == prev_sig, jnp.minimum(stale + 1, _int(999)), _int(0))
    stuck = (stale >= STALL_LIMIT) & (~pot_cooked) & (pot_timer == 0) & (pot_total < 3)

    recipe, recipe_seen = _recipe_reading(obs)
    contradicts = recipe_seen & jnp.any(pot_counts > recipe)

    # Did the partner start a dish that the live display should have prevented?
    filled = pot_seen & (jnp.sum(pot_counts) > 0)
    started = filled & (prev_filled == 0)
    # Wide enough that every fill following a reminder is judged, but each
    # judgement is reversible: a partner that dumps a mistaken dish and refills
    # correctly scores one strike and immediately takes it back.
    display_was_live = since <= STRIKE_WINDOW
    ignored = started & display_was_live & contradicts
    heeded = started & display_was_live & (~contradicts)
    strikes_next = jnp.clip(strikes + ignored.astype(jnp.int32) - heeded.astype(jnp.int32), 0, 5)
    muted = strikes >= STRIKE_LIMIT

    # The team needs the recipe displayed when the cook is about to start a new
    # dish, or when what is in the pot cannot become the target dish.
    need = jnp.where(pot_seen, pot_empty | contradicts | stuck, jnp.asarray(True))
    affordable = (clock <= PRESS_DEADLINE) & (spent < PRESS_BUDGET) & (~muted)

    action, ready = _approach(obs, row, col, _int(2), _int(2), _int(A_RIGHT))

    new_episode = need & (prev_need == 0)
    budget = jnp.where(new_episode, _int(0), presses)
    age = jnp.where(new_episode, _int(0), phase_age)
    # A reminder is only worth paying for once the partner is later than it has
    # ever been after an activation it demonstrably heeded.
    overdue = (budget == 0) | (age >= jnp.minimum(latency + LATENCY_MARGIN, _int(PRESS_GAP_MAX)))
    # A partner that has demonstrably used the pot at least once is worth one
    # more reminder after a long silence, even inside the same need episode.
    retry = ((active > 0) & (since >= PRESS_RETRY)) | (since >= PRESS_RETRY_SILENT)
    press = need & affordable & ready & (since >= PRESS_GAP) & (
        ((budget < PRESS_MAX) & overdue) | retry
    )

    action = jnp.where(ready, jnp.where(press, _int(A_INTERACT), _int(A_STAY)), action)

    next_memory = {
        'since': jnp.where(press, _int(0), jnp.minimum(since + 1, _int(999))),
        'presses': (budget + press.astype(jnp.int32)).astype(jnp.int32),
        'need': need.astype(jnp.int32),
        'clock': jnp.minimum(clock + 1, _int(9999)).astype(jnp.int32),
        'spent': (spent + press.astype(jnp.int32)).astype(jnp.int32),
        'strikes': strikes_next.astype(jnp.int32),
        'filled': filled.astype(jnp.int32),
        'potsig': signature.astype(jnp.int32),
        'stale': stale_next.astype(jnp.int32),
        'active': jnp.maximum(active, filled.astype(jnp.int32)).astype(jnp.int32),
        # How long after the need arose the partner acted on an activation it
        # heeded; measured from the need onset, not from the last activation,
        # because reminders themselves reset the latter.
        'latency': jnp.where(heeded, jnp.maximum(latency, jnp.clip(age, 0, 40)), latency).astype(jnp.int32),
        'phase': jnp.minimum(age + 1, _int(999)).astype(jnp.int32),
    }
    trace = {
        'need': need.astype(jnp.int32),
        'press': press.astype(jnp.int32),
        'strikes': strikes_next.astype(jnp.int32),
        'spent': next_memory['spent'],
    }
    return action.astype(jnp.int32), next_memory, trace


# --------------------------------------------------------------------------
# agent_1: cook in the right room
# --------------------------------------------------------------------------


def init_agent_1():
    return {
        'p0': jnp.asarray(0.5, dtype=jnp.float32),
        'wait': jnp.asarray(0, dtype=jnp.int32),
        'quiet': jnp.asarray(999, dtype=jnp.int32),
        'gave_up': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    start = init_agent_1()
    p0 = jnp.where(reset, start['p0'], memory['p0'])
    wait = jnp.where(reset, start['wait'], memory['wait'])
    quiet = jnp.where(reset, start['quiet'], memory['quiet'])
    gave_up = jnp.where(reset, start['gave_up'], memory['gave_up'])

    # Self position from the pot landmark, visible from every cell of the room.
    pot_mask = obs[:, :, 22] > 0
    pot_row, pot_col, pot_seen = _locate(obs, 22)
    row = jnp.where(pot_seen, jnp.clip(4 - pot_row, 1, 3), _int(2))
    col = jnp.where(pot_seen, jnp.clip(6 - pot_col, 5, 6), _int(5))

    pot_cell = _masked_cell(obs, pot_mask)
    pot_counts = pot_cell[31:34]
    pot_total = jnp.sum(pot_counts)
    pot_cooked = pot_cell[30] > 0
    pot_timer = pot_cell[39]
    pot_busy = pot_cooked | (pot_timer > 0)

    # Any visible recipe display gives certainty about the current target.
    reading, reading_seen = _recipe_reading(obs)
    p0 = jnp.where(reading_seen, jnp.where(reading[0] >= reading[1], 1.0, 0.0), p0)

    me = obs[2, 2]
    held_plate = me[5] > 0
    held_cooked = me[6] > 0
    held_ing = me[7:10]
    held_total = jnp.sum(held_ing)
    holding_dish = held_plate & held_cooked
    holding_plate = held_plate & (~held_cooked)
    holding_raw = (~held_plate) & (held_total > 0)

    guess = jnp.where(p0 >= 0.5, _int(0), _int(1))
    pot_type = jnp.argmax(pot_counts).astype(jnp.int32)
    wanted = jnp.where(pot_total > 0, pot_type, guess)

    dump_row, dump_col, dump_face, dump_ok = _dump_target(obs, row, col)

    # --- empty hands -----------------------------------------------------
    plate_row = jnp.where(row >= 3, _int(3), _int(1))
    ingredient_row = jnp.where(wanted == 1, _int(3), _int(1))
    confident = (jnp.maximum(p0, 1.0 - p0) >= 0.9) | (pot_total > 0)
    # Be patient when some partner has recently displayed a recipe, spend one
    # long listen before concluding that nobody signals, and stop paying for
    # silence after that.
    heard = quiet <= SIGNAL_MEMORY
    patience = jnp.where(
        heard, _int(WAIT_SEEN), jnp.where(gave_up > 0, _int(WAIT_GAVE_UP), _int(WAIT_FIRST))
    )
    timed_out = wait >= patience
    collect = confident | timed_out
    e_row = jnp.where(pot_busy, plate_row, jnp.where(collect, ingredient_row, _int(2)))
    e_col = jnp.where(pot_busy, _int(5), jnp.where(collect, _int(6), _int(5)))
    e_face = jnp.where(pot_busy, _int(A_LEFT), jnp.where(collect, _int(A_RIGHT), _int(A_LEFT)))
    e_use = pot_busy | collect
    # Count only while genuinely stalled for want of a recipe.  The counter must
    # keep its value once the timeout fires, otherwise the fallback would reset
    # itself every step and never commit to a dish.
    idle_hands = (~holding_dish) & (~holding_plate) & (~holding_raw)
    stalled = idle_hands & (~pot_busy) & (pot_total == 0) & (~confident)

    # --- holding a raw ingredient ---------------------------------------
    raw_type = jnp.argmax(held_ing).astype(jnp.int32)
    space = (pot_timer == 0) & (~pot_cooked) & (pot_total < 3)
    fits = jnp.where(pot_total > 0, pot_counts[raw_type] > 0, raw_type == guess)
    # Last resort: with no free counter in sight, put an unwanted unit into the
    # pot anyway rather than freeze holding it forever.  The junk dish is dumped
    # or delivered later, but the cycle keeps running.
    deposit = space & (fits | (~dump_ok))
    r_row = jnp.where(deposit, _int(2), dump_row)
    r_col = jnp.where(deposit, _int(5), dump_col)
    r_face = jnp.where(deposit, _int(A_LEFT), dump_face)
    r_use = deposit | dump_ok

    # --- holding an empty plate ------------------------------------------
    plate_keep = pot_busy
    p_row = jnp.where(plate_keep, _int(2), dump_row)
    p_col = jnp.where(plate_keep, _int(5), dump_col)
    p_face = jnp.where(plate_keep, _int(A_LEFT), dump_face)
    p_use = jnp.where(plate_keep, pot_cooked, dump_ok)

    # --- holding a cooked dish -------------------------------------------
    correct = jnp.where(
        held_ing[0] >= 3, p0, jnp.where(held_ing[1] >= 3, 1.0 - p0, jnp.float32(0.0))
    )
    deliver = (correct >= 0.5) | (~dump_ok)
    d_row = jnp.where(deliver, _int(2), dump_row)
    d_col = jnp.where(deliver, _int(6), dump_col)
    d_face = jnp.where(deliver, _int(A_RIGHT), dump_face)

    # --- pick the active branch ------------------------------------------
    target_row = jnp.where(
        holding_dish, d_row, jnp.where(holding_plate, p_row, jnp.where(holding_raw, r_row, e_row))
    )
    target_col = jnp.where(
        holding_dish, d_col, jnp.where(holding_plate, p_col, jnp.where(holding_raw, r_col, e_col))
    )
    target_face = jnp.where(
        holding_dish, d_face,
        jnp.where(holding_plate, p_face, jnp.where(holding_raw, r_face, e_face)),
    )
    interacts = jnp.where(
        holding_dish, jnp.asarray(True),
        jnp.where(holding_plate, p_use, jnp.where(holding_raw, r_use, e_use)),
    )

    action, ready = _approach(obs, row, col, target_row, target_col, target_face)
    action = jnp.where(ready, jnp.where(interacts, _int(A_INTERACT), _int(A_STAY)), action)

    # Delivering consumes the dish and, when correct, resamples the recipe, so
    # the belief has to be pushed forward without ever observing the reward.
    delivering = holding_dish & deliver & ready
    pure_a = held_ing[0] >= 3
    pure_b = held_ing[1] >= 3
    after = jnp.where(pure_a, p0 * 0.5, jnp.where(pure_b, p0 + (1.0 - p0) * 0.5, p0))
    p0 = jnp.where(delivering & (pure_a | pure_b), after, p0)

    fresh_cycle = reading_seen | delivering
    held_wait = jnp.where(stalled, jnp.minimum(wait + 1, _int(999)), wait)
    next_memory = {
        'p0': p0.astype(jnp.float32),
        'wait': jnp.where(fresh_cycle, _int(0), held_wait).astype(jnp.int32),
        'quiet': jnp.where(reading_seen, _int(0), jnp.minimum(quiet + 1, _int(999))).astype(jnp.int32),
        'gave_up': jnp.maximum(
            gave_up, (stalled & timed_out & (~heard)).astype(jnp.int32)
        ).astype(jnp.int32),
    }
    trace = {
        'p0': p0.astype(jnp.float32),
        'wait': next_memory['wait'],
        'row': row.astype(jnp.int32),
        'col': col.astype(jnp.int32),
        'pot': pot_total.astype(jnp.int32),
        'deliver': delivering.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
