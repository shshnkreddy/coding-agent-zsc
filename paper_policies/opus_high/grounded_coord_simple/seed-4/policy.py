"""Grounded-coordination policy for overcooked_v2 ``grounded_coord_simple``.

Layout facts that drive the design (see context/environment/README.md)::

    WW2WWWWW
    W  WB  0
    R ALPA X
    W  WB  1
    WW2WWWWW

The two rooms are disjoint.  The left room (seat 0) touches only the always-on
recipe indicator ``R``, the recipe button ``L`` and the useless ingredient 2
piles.  The right room (seat 1) touches the pot, both plate piles, ingredient
piles 0 and 1 and the delivery station.  Therefore seat 1 alone can cook and
deliver, but it can never see the true recipe: the only channel between the
rooms is the button display at (2, 3), which seat 1 can read from column 5.

Seat 0 therefore parks next to the button and announces the recipe at the start
of each dish cycle, which it recognises from the pot emptying out again.  An
activation costs 5 reward, so an unchanged recipe is left unannounced unless the
partner has shown that it needs one anyway - it stalled with an empty pot, or it
cooked the wrong ingredient - and never for more than two cycles running.

Seat 1 runs a reactive cook -> plate -> deliver loop over a calibrated belief
about the recipe: any display sets it with certainty, and silence is read
through an explicit model of the partner, since a partner that announces changes
makes silence mean "unchanged" while a mute one makes it mean nothing.  In the
latter case an incorrect delivery leaves the recipe standing, so the ingredient
just turned in becomes the less likely one and the seat switches.

Nothing here is a private code: both seats trade only in the environment's own
recipe display and in what the shared pot reveals about the other room.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_button_coord',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Movement/interaction action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INT = 0, 1, 2, 3, 4, 5
# Facing ids as they appear in observation channels 1..4.
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# Action that yields each facing (also usable as a pure turn when blocked).
_TURN = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# Seat 0 waits this many steps between repeat activations of the button.
_PRESS_GAP = 15
# Steps seat 0 waits after the pot empties before judging whether the coming
# dish needs to be announced: the partner's delivery, and with it any change of
# the recipe, lands within a couple of steps of it collecting the cooked food.
_DECIDE_DELAY = 3
# Silent cycles in a row before the recipe is re-shown anyway, so that a
# partner discounting silence never has to stretch a reading too far.
_SKIP_CAP = 2
# An empty pot this long into a cycle means the partner is waiting to be told.
_STALL = 16
# Re-activation period when no partner activity is visible at all, and the cap
# on such speculative activations (each costs 5 team reward).
_IDLE_PERIOD = 50
_MAX_PRESSES = 5
# Steps a freshly activated button stays visible in observations.
_LIT_STEPS = 9
# Wrong pot commitments against a lit display before seat 0 stops paying.
_IGNORE_STRIKES = 2

# How long seat 1 waits for a fresh display after a delivery before falling
# back on its own inference.
_WAIT_MAX = 4
# Prior that a partner signaller is one that speaks up whenever the recipe
# changes, so that its silence carries the meaning "still the same recipe".
_SPEAKER_PRIOR = 0.8
# Steps an unconfirmed dish is held at the pot, in view of the button, hoping
# for a confirming display before it is turned in anyway.
_HOLD_MAX = 8
# Belief strength at which waiting for confirmation is not worth the delay.
_SURE_ENOUGH = 0.6


def _static_grid():
    """Global static map as [row][col][9] flags matching obs channels 20..28."""
    rows = ("WW2WWWWW", "W  WB  0", "R ALPA X", "W  WB  1", "WW2WWWWW")
    symbol = {"W": 0, "X": 1, "P": 2, "R": 3, "L": 4, "B": 5, "0": 6, "1": 7, "2": 8}
    grid = [[[0] * 9 for _ in range(8)] for _ in range(5)]
    for r, row in enumerate(rows):
        for c, sym in enumerate(row):
            channel = symbol.get(sym)
            if channel is not None:
                grid[r][c][channel] = 1
    return grid


def _crops(cells):
    """Expected 5x5x9 static crop for each candidate self position."""
    grid = _static_grid()
    out = []
    for (r, c) in cells:
        crop = []
        for i in range(5):
            line = []
            for j in range(5):
                gr, gc = r + i - 2, c + j - 2
                inside = 0 <= gr < 5 and 0 <= gc < 8
                line.append(list(grid[gr][gc]) if inside else [0] * 9)
            crop.append(line)
        out.append(crop)
    return jnp.asarray(out, dtype=jnp.int32)


_CELLS_0 = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
_CELLS_1 = ((1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6))


def _locate(obs, cells):
    """Self position from the static channels; the six room cells are unique.

    The wall channel is deliberately excluded so that localization only relies
    on object channels whose semantics are unambiguous.
    """
    crops = _crops(cells)
    positions = jnp.asarray(cells, dtype=jnp.int32)
    seen = obs[None, :, :, 21:29]
    cost = jnp.abs(crops[:, :, :, 1:9] - seen).sum(axis=(1, 2, 3))
    best = jnp.argmin(cost)
    return positions[best, 0], positions[best, 1]


def _pot(obs):
    """Pot contents/timer read from wherever the pot is inside the crop."""
    flat = obs.reshape(25, 40)
    mask = flat[:, 22]
    cell = flat[jnp.argmax(mask)]
    seen = mask.sum() > 0
    counts = cell[31:34]
    total = counts.sum()
    cooked = cell[30] > 0
    timer = cell[39]
    return {
        'seen': seen,
        'counts': counts,
        'n': total,
        'cooked': seen & cooked,
        'timer': timer,
        'ingredient': jnp.argmax(counts),
        'empty': seen & (total == 0) & (~cooked),
        'filling': seen & (total > 0) & (total < 3) & (timer == 0) & (~cooked),
        'cooking': seen & (timer > 0),
        'ready': seen & cooked & (timer == 0),
        'accepts': seen & (total < 3) & (timer == 0) & (~cooked),
    }


def _display(obs):
    """Recipe shown by any visible indicator (always-on R or an active L)."""
    shown = obs.reshape(25, 40)[:, 36:39].max(axis=0)
    return shown.sum() > 0, jnp.argmax(shown)


def _held(obs):
    inventory = obs[2, 2, 5:10]
    counts = inventory[2:5]
    total = counts.sum()
    plate = inventory[0] > 0
    cooked = inventory[1] > 0
    return {
        'dish': cooked,
        'plate': plate & (~cooked),
        'ingredient': (~plate) & (~cooked) & (total > 0),
        'kind': jnp.argmax(counts),
        'empty': (~plate) & (~cooked) & (total == 0),
    }


def _counter_state(obs, r, c, gr, gc):
    """0 = visibly free, 1 = out of view, 2 = visibly occupied."""
    li, lj = gr - r + 2, gc - c + 2
    visible = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    item = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), 29:34].sum()
    return jnp.where(~visible, 1, jnp.where(item > 0, 2, 0))


def _step_towards(r, c, facing, tr, tc, want_dir, interact):
    """Walk to (tr, tc), turn to ``want_dir``, then INTERACT or STAY.

    Both rooms are open rectangles, so moving in row first and then in column
    never runs into an obstacle, and a turn is a blocked move into the object
    being addressed, which costs a step only when the approach did not already
    leave the right facing.
    """
    drow, dcol = tr - r, tc - c
    move = jnp.where(drow < 0, A_UP,
                     jnp.where(drow > 0, A_DOWN,
                               jnp.where(dcol > 0, A_RIGHT,
                                         jnp.where(dcol < 0, A_LEFT, -1))))
    arrived = move < 0
    turn = jnp.asarray(_TURN, dtype=jnp.int32)[want_dir]
    final = jnp.where(facing != want_dir, turn,
                      jnp.where(interact, A_INT, A_STAY))
    return jnp.where(arrived, final, move).astype(jnp.int32)


def _reset_memory(memory, fresh, reset):
    return jax.tree_util.tree_map(
        lambda old, new: jnp.where(reset, new, old), memory, fresh)


# --------------------------------------------------------------------------
# Seat 0: recipe signaller in the left room.
# --------------------------------------------------------------------------

def init_agent_0():
    zero = jnp.asarray(0, dtype=jnp.int32)
    return {
        'since': zero,          # steps since the last activation
        'presses': zero,        # activations so far this episode
        'recipe': jnp.asarray(-1, dtype=jnp.int32),  # recipe at last activation
        'nonempty': zero,       # pot was seen non-empty since last activation
        'pressed': zero,        # any activation yet
        'ignored': zero,        # times the partner committed against a live display
        'pot_n': zero,          # pot fill level last step
        'armed': zero,          # a new dish cycle is waiting to be judged
        'age': zero,            # steps since that cycle began
        'skips': zero,          # cycles left unannounced in a row
        'needs': zero,          # partner has shown it must be told every cycle
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)
    mem = _reset_memory(memory, init_agent_0(), reset)

    r, c = _locate(obs, _CELLS_0)
    facing = jnp.argmax(obs[2, 2, 1:5])
    pot = _pot(obs)
    shown, recipe_now = _display(obs)

    since = mem['since'] + 1
    nonempty = jnp.maximum(mem['nonempty'],
                           (pot['seen'] & (~pot['empty'])).astype(jnp.int32))
    # While the button is still lit it displays the *current* recipe by itself,
    # so the recorded recipe keeps tracking it and no re-activation is needed.
    live = (mem['pressed'] > 0) & (since <= _LIT_STEPS) & shown
    recipe_mem = jnp.where(live, recipe_now, mem['recipe'])

    first = mem['pressed'] == 0
    # A pot that just became empty again means the partner started a new dish;
    # that cycle then waits to be judged rather than being announced at once.
    new_cycle = pot['empty'] & (nonempty > 0) & (since >= _PRESS_GAP)
    armed = jnp.maximum(mem['armed'], new_cycle.astype(jnp.int32))
    age = jnp.where(armed > mem['armed'], 0, mem['age'] + 1)

    # The indicator only changes after a correct delivery: fresh news to relay.
    changed = (mem['pressed'] > 0) & shown & (recipe_now != recipe_mem)
    # An unchanged recipe is worth 5 reward only if this partner needs to hear
    # it again, or if it has already gone unannounced for too many cycles.
    due = (armed > 0) & (age >= _DECIDE_DELAY) & ((mem['needs'] > 0)
                                                  | (mem['skips'] >= _SKIP_CAP))
    # Left unannounced and the pot never started: the partner is waiting to be
    # told, so tell it, and from now on tell it every cycle.
    stalled_cycle = (armed > 0) & (age >= _STALL) & pot['empty']
    # Nothing observable at all for a long time: re-offer the recipe in case the
    # partner missed the first activation entirely.
    stalled = (pot['empty'] & (nonempty == 0) & (since >= _IDLE_PERIOD)
               & (mem['presses'] < _MAX_PRESSES))

    # What the partner does with the pot says whether the channel is working.
    committed = pot['seen'] & (mem['pot_n'] == 0) & (pot['n'] > 0)
    wrong = committed & shown & (pot['ingredient'] != recipe_now)
    lit = (mem['pressed'] > 0) & (since <= _LIT_STEPS + 1)
    # Cooking the wrong thing with the recipe in plain sight means the partner
    # ignores the channel; cooking it after a silent cycle means the opposite,
    # that this partner cannot carry a reading across a delivery.
    ignored = mem['ignored'] + (wrong & lit).astype(jnp.int32)
    needs = jnp.maximum(mem['needs'], (wrong & (~lit)).astype(jnp.int32))
    needs = jnp.maximum(needs, stalled_cycle.astype(jnp.int32))
    useless = ignored >= _IGNORE_STRIKES

    want = (first | changed | due | stalled_cycle
            | (stalled & (since >= _PRESS_GAP))) & (~useless)

    action = _step_towards(r, c, facing, 2, 2, F_RIGHT, want)
    pressed_now = action == A_INT
    # The cycle is settled once it is announced or once the partner gets going.
    skipped = (armed > 0) & (~pressed_now) & committed
    armed = jnp.where(pressed_now | committed, 0, armed)

    next_mem = {
        'since': jnp.where(pressed_now, 0, since).astype(jnp.int32),
        'presses': (mem['presses'] + pressed_now.astype(jnp.int32)).astype(jnp.int32),
        'recipe': jnp.where(pressed_now, recipe_now, recipe_mem).astype(jnp.int32),
        'nonempty': jnp.where(pressed_now, 0, nonempty).astype(jnp.int32),
        'pressed': jnp.maximum(mem['pressed'], pressed_now.astype(jnp.int32)),
        'ignored': ignored.astype(jnp.int32),
        'pot_n': jnp.where(pot['seen'], pot['n'], mem['pot_n']).astype(jnp.int32),
        'armed': armed.astype(jnp.int32),
        'age': age.astype(jnp.int32),
        'skips': jnp.where(pressed_now, 0,
                           mem['skips'] + skipped.astype(jnp.int32)).astype(jnp.int32),
        'needs': needs.astype(jnp.int32),
    }
    trace = {
        'row': r.astype(jnp.int32),
        'col': c.astype(jnp.int32),
        'press': pressed_now.astype(jnp.int32),
        'presses': next_mem['presses'],
        'recipe': recipe_now.astype(jnp.int32),
        'pot_n': pot['n'].astype(jnp.int32),
        'ignored': ignored.astype(jnp.int32),
        'needs': needs.astype(jnp.int32),
        'skips': mem['skips'].astype(jnp.int32),
    }
    return action, next_mem, trace


# --------------------------------------------------------------------------
# Seat 1: cook and deliverer in the right room.
# --------------------------------------------------------------------------

# mode -> (row, col, facing, interact)
#  0 idle at the pot          1 use the pot (fill / plate up)
#  2 take ingredient 0        3 take ingredient 1
#  4 take a plate             5 deliver
#  6..9 put an unwanted item on a free counter
_M_ROW = (2, 2, 1, 3, 1, 2, 1, 3, 1, 3)
_M_COL = (5, 5, 6, 6, 5, 6, 5, 5, 6, 6)
_M_DIR = (F_LEFT, F_LEFT, F_RIGHT, F_RIGHT, F_LEFT, F_RIGHT,
          F_UP, F_DOWN, F_UP, F_DOWN)
_M_INT = (0, 1, 1, 1, 1, 1, 1, 1, 1, 1)
_DUMP_CELLS = ((0, 5), (4, 5), (0, 6), (4, 6))


def init_agent_1():
    """Belief state: ``belief`` is the ingredient believed to be the recipe and
    ``q`` the probability that this is right (1.0 right after a display)."""
    zero = jnp.asarray(0, dtype=jnp.int32)
    return {
        'belief': zero,                              # believed recipe ingredient
        'q': jnp.asarray(0.5, dtype=jnp.float32),    # P(recipe == belief)
        'w': jnp.asarray(_SPEAKER_PRIOR, dtype=jnp.float32),  # P(partner speaks)
        'ever': zero,                                # any delivery made yet
        'seen': zero,                                # any display ever read
        'since_del': zero,                           # steps since the delivery
        'pending': jnp.asarray(1, dtype=jnp.int32),  # awaiting a post-delivery read
        'dkind': zero,                               # ingredient last turned in
        'dq': jnp.asarray(0.5, dtype=jnp.float32),   # its probability of matching
        'had_dish': zero,                            # held a dish last step
        'turning_in': zero,                          # last action was a delivery
        'hold': zero,                                # steps an unsure dish was held
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)
    mem = _reset_memory(memory, init_agent_1(), reset)

    r, c = _locate(obs, _CELLS_1)
    facing = jnp.argmax(obs[2, 2, 1:5])
    pot = _pot(obs)
    held = _held(obs)
    shown, recipe_now = _display(obs)
    one = jnp.asarray(1.0, dtype=jnp.float32)
    half = jnp.asarray(0.5, dtype=jnp.float32)

    # --- what the last delivery implies ----------------------------------
    # A dish only leaves the hands through the delivery station or a counter;
    # only the former can have resampled the recipe.
    delivered = (mem['had_dish'] > 0) & (~held['dish']) & (mem['turning_in'] > 0)
    since_del = jnp.where(delivered, 0, jnp.minimum(mem['since_del'] + 1, 999))
    pending = jnp.maximum(mem['pending'], delivered.astype(jnp.int32))

    belief = mem['belief']
    q = mem['q']
    speaker = mem['w']

    # A display is ground truth about the current recipe.
    belief = jnp.where(shown, jnp.clip(recipe_now, 0, 1), belief)
    q = jnp.where(shown, one, q)
    # A display arriving right after a delivery is a partner that keeps this
    # seat informed when it matters, which is what gives silence its meaning.
    speaker = jnp.where(shown & (since_del <= _WAIT_MAX + 2),
                        jnp.asarray(_SPEAKER_PRIOR, dtype=jnp.float32), speaker)
    seen = jnp.maximum(mem['seen'], shown.astype(jnp.int32))
    pending = jnp.where(shown, 0, pending)

    # No display arrived in time: fall back on inference about the dish of
    # kind ``dkind`` that was turned in with probability ``dq`` of matching.
    #  * If the partner speaks whenever the recipe changes, silence says the
    #    recipe is whatever it was, so P(dkind) = .5 dq / (1 - .5 dq): the
    #    delivery was right (dq) and the resample repeated it (.5), against
    #    every history in which nothing would have been announced.
    #  * If the partner never speaks, silence says nothing: a correct delivery
    #    (dq) resamples to dkind half the time and a wrong one leaves the other
    #    ingredient in place, so P(dkind) = dq / 2.
    # Silence itself is evidence about which partner this is, since a speaker
    # would have announced a change: its likelihood of silence is 1 - .5 dq.
    settle = (pending > 0) & (since_del >= _WAIT_MAX) & (mem['ever'] > 0)
    quiet = one - half * mem['dq']
    p_speaker = (half * mem['dq']) / quiet
    p_mute = half * mem['dq']
    p_same = speaker * p_speaker + (one - speaker) * p_mute
    posted = speaker * quiet
    speaker_next = posted / (posted + (one - speaker))
    same_kind = p_same >= half
    belief = jnp.where(settle, jnp.where(same_kind, mem['dkind'], 1 - mem['dkind']),
                       belief)
    q = jnp.where(settle, jnp.maximum(p_same, one - p_same), q)
    speaker = jnp.where(settle, speaker_next, speaker)
    pending = jnp.where(settle | (since_del >= _WAIT_MAX), 0, pending)

    belief = jnp.clip(belief, 0, 1).astype(jnp.int32)

    # --- what to do about it ---------------------------------------------
    # A partly filled pot dictates what must be finished, whatever the belief.
    target = jnp.where(pot['filling'], pot['ingredient'], belief)
    fetch = jnp.where(target == 0, 2, 3)

    scores = jnp.stack([_counter_state(obs, r, c, gr, gc) for gr, gc in _DUMP_CELLS])
    slot = jnp.argmin(scores)
    dump = (6 + slot).astype(jnp.int32)
    can_dump = scores[slot] < 2

    # Never commit the pot while a post-delivery reading may still arrive.
    may_start = pending == 0

    # Probability that the dish in hand is the one being asked for.
    dish_kind = held['kind']
    q_dish = jnp.where(dish_kind == belief, q, one - q)
    hold = jnp.where(held['dish'], mem['hold'] + 1, 0)
    sure = q_dish >= _SURE_ENOUGH
    # An unsure dish waits at the pot, where the button stays in view, but only
    # while a display is something this partner has ever produced, and never
    # long enough to let a mute partner stall the kitchen.
    patient = (hold >= _HOLD_MAX) | (seen == 0)
    turn_in = sure | ((q_dish >= half) & patient)
    discard = q_dish < half

    empty_mode = jnp.where(pot['ready'] | pot['cooking'], 4,
                           jnp.where(pot['filling'] | may_start, fetch, 0))
    ing_mode = jnp.where((held['kind'] == target) & pot['accepts'], 1,
                         jnp.where(can_dump, dump, 0))
    plate_mode = jnp.where(pot['ready'], 1,
                           jnp.where(pot['cooking'], 0,
                                     jnp.where(can_dump, dump, 0)))
    dish_mode = jnp.where(turn_in, 5, jnp.where(discard & can_dump, dump, 0))

    mode = jnp.where(held['dish'], dish_mode,
                     jnp.where(held['plate'], plate_mode,
                               jnp.where(held['ingredient'], ing_mode, empty_mode)))

    table = jnp.asarray([_M_ROW, _M_COL, _M_DIR, _M_INT], dtype=jnp.int32)
    action = _step_towards(r, c, facing, table[0, mode], table[1, mode],
                           table[2, mode], table[3, mode] > 0)
    turning_in = (mode == 5) & (action == A_INT)

    next_mem = {
        'belief': belief,
        'q': q.astype(jnp.float32),
        'w': speaker.astype(jnp.float32),
        'ever': jnp.maximum(mem['ever'], turning_in.astype(jnp.int32)).astype(jnp.int32),
        'seen': seen.astype(jnp.int32),
        'since_del': since_del.astype(jnp.int32),
        'pending': pending.astype(jnp.int32),
        'dkind': jnp.where(turning_in, dish_kind, mem['dkind']).astype(jnp.int32),
        'dq': jnp.where(turning_in, q_dish, mem['dq']).astype(jnp.float32),
        'had_dish': held['dish'].astype(jnp.int32),
        'turning_in': turning_in.astype(jnp.int32),
        'hold': hold.astype(jnp.int32),
    }
    trace = {
        'row': r.astype(jnp.int32),
        'col': c.astype(jnp.int32),
        'mode': mode.astype(jnp.int32),
        'belief': belief,
        'q': q.astype(jnp.float32),
        'w': speaker.astype(jnp.float32),
        'pot_n': pot['n'].astype(jnp.int32),
        'pot_timer': pot['timer'].astype(jnp.int32),
    }
    return action, next_mem, trace
