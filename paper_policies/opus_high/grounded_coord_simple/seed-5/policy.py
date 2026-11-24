"""Grounded-coordination policy for overcooked_v2 ``grounded_coord_simple``.

Layout facts that drive the design (see context/environment/README.md):

* The two floor regions are disconnected.  The left region (rows 1-3, cols 1-2)
  can only reach the always-on recipe indicator ``R`` at (2, 0), the recipe
  button ``L`` at (2, 3) and the ingredient-2 piles.  The right region
  (rows 1-3, cols 5-6) is the only one that can reach the pot (2, 4), both
  plate piles, the ingredient-0/1 piles and the delivery station (2, 7).
* Therefore the right seat does all cooking and the left seat can only help by
  pressing the button, whose display at (2, 3) is readable from column 5 of the
  right room.  An active display always shows the *current* recipe, so a
  visible display is authoritative.

Each act function decides its own role from its observation (only the left
region ever sees the recipe-indicator channel), so the policy behaves correctly
whichever seat it is placed in.
"""

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

# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids as they appear in observation channels 1..4.
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# Movement action that produces each facing (also used as a pure turn when the
# target cell is blocked, which is true for every station used below).
_TURN = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# Cook stations: task id -> (stand row, stand col, facing, interact?).
# 0 WAIT@pot, 1 POT, 2 SUPPLY0, 3 SUPPLY1, 4 PLATE, 5 DELIVER,
# 6 COUNTER(0,5), 7 COUNTER(0,6), 8 COUNTER(4,5), 9 COUNTER(4,6)
_T_R = (2, 2, 1, 3, 1, 2, 1, 1, 3, 3)
_T_C = (5, 5, 6, 6, 5, 6, 5, 6, 5, 6)
_T_F = (F_LEFT, F_LEFT, F_RIGHT, F_RIGHT, F_LEFT, F_RIGHT,
        F_UP, F_UP, F_DOWN, F_DOWN)
_T_I = (0, 1, 1, 1, 1, 1, 1, 1, 1, 1)

# Tunables.
_REFRESH = 34    # signaller: dark steps before a refresh press (period ~43)
_WAIT = 8        # cook: steps to wait for a post-delivery signal before committing
_TRUST = 2       # cook: minimum silent deliveries treated as "recipe unchanged"
_TRUST_ADD = 1   # cook: offset applied to the observed activation count
_MAX_TRUST = 8   # cook: cap on that trust however talkative the partner is
_BOOT = 50       # cook: step at which to start cooking blind if never signalled


def _init():
    return {
        't': jnp.asarray(0, dtype=jnp.int32),
        'rec': jnp.zeros((3,), dtype=jnp.int32),      # last displayed recipe
        'known': jnp.asarray(0, dtype=jnp.int32),
        'seen': jnp.asarray(0, dtype=jnp.int32),      # display read since my last delivery
        'silent': jnp.asarray(0, dtype=jnp.int32),    # deliveries since last display read
        'bursts': jnp.asarray(0, dtype=jnp.int32),    # cook: distinct activations seen
        'lit_t': jnp.asarray(-100, dtype=jnp.int32),  # cook: last step a display was read
        'deliv_t': jnp.asarray(-100, dtype=jnp.int32),
        'prev': jnp.full((3,), -1, dtype=jnp.int32),  # signaller: last recipe seen on R
        'vis': jnp.asarray(0, dtype=jnp.int32),       # signaller: lit steps since that change
        'act_t': jnp.asarray(-100, dtype=jnp.int32),  # signaller: last step display was on
    }


def _find(obs, channel):
    """Local (row, col) of a unique static object, plus a presence flag."""
    flat = obs[:, :, channel].reshape(-1)
    idx = jnp.argmax(flat)
    return idx // 5, idx % 5, flat[idx] > 0


def _read(obs, row, col, grow, gcol, lo, hi):
    """Channel slice of global cell (grow, gcol) given my global position."""
    i = grow - row + 2
    j = gcol - col + 2
    ok = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    values = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), lo:hi]
    return values * ok.astype(jnp.int32), ok


def _nav(row, col, face, trow, tcol, tface, interact):
    """Move to a station (rows first), turn to face it, then act."""
    at = (row == trow) & (col == tcol)
    arrived = jnp.where(face == tface,
                        jnp.where(interact > 0, A_INTERACT, A_STAY),
                        jnp.asarray(_TURN, dtype=jnp.int32)[tface])
    move = jnp.where(row > trow, A_UP,
                     jnp.where(row < trow, A_DOWN,
                               jnp.where(col > tcol, A_LEFT, A_RIGHT)))
    return jnp.where(at, arrived, move).astype(jnp.int32)


def _signal(obs, mem):
    """Left room: watch R(2, 0) and keep the button display at (2, 3) useful."""
    t = mem['t']
    li, lj, ok = _find(obs, 23)
    row = 4 - li
    col = 2 - lj
    face = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    current, _ = _read(obs, row, col, 2, 0, 36, 39)
    shown, _ = _read(obs, row, col, 2, 3, 36, 39)
    active = jnp.sum(shown) > 0

    # An active display always shows the *live* recipe, so a recipe change that
    # happens while the display is on needs no press -- provided the display
    # then stays on long enough for the cook to look at it.  A cook walking the
    # right room is away from column 5 for at most two steps in a row, so three
    # consecutive lit steps since the change count as delivered.
    changed = jnp.any(current != mem['prev'])
    seen_by_cook = jnp.where(changed,
                             active.astype(jnp.int32),
                             jnp.where(active, mem['vis'] + 1, mem['vis']))
    stale = (t - mem['act_t']) >= _REFRESH
    press = jnp.logical_and(jnp.logical_not(active),
                            (seen_by_cook < 3) | stale)

    action = _nav(row, col, face, 2, 2, F_RIGHT, press.astype(jnp.int32))
    action = jnp.where(ok, action, jnp.asarray(A_STAY, dtype=jnp.int32))

    out = dict(mem)
    out['prev'] = jnp.where(ok, current, mem['prev'])
    out['vis'] = seen_by_cook.astype(jnp.int32)
    out['act_t'] = jnp.where(active, t, mem['act_t'])
    out['t'] = t + 1
    return action.astype(jnp.int32), out, press.astype(jnp.int32)


def _cook(obs, mem):
    """Right room: fill the pot, plate the dish and deliver it."""
    t = mem['t']
    pi, pj, ok = _find(obs, 22)
    row = 4 - pi
    col = 6 - pj
    face = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    held_plate = obs[2, 2, 5]
    held_cooked = obs[2, 2, 6]
    held_ing = obs[2, 2, 7:10]
    held_n = jnp.sum(held_ing)
    hands_free = (held_plate == 0) & (held_cooked == 0) & (held_n == 0)

    pot_item = obs[pi, pj, 29:34]
    pot_counts = pot_item[2:5]
    pot_total = jnp.sum(pot_counts)
    pot_ready = pot_item[1] > 0
    pot_timer = obs[pi, pj, 39]
    cooking = pot_timer > 0
    pot_idle = jnp.logical_not(cooking | pot_ready)

    # --- recipe belief -----------------------------------------------------
    shown, shown_ok = _read(obs, row, col, 2, 3, 36, 39)
    fresh = shown_ok & (jnp.sum(shown) > 0)
    rec = jnp.where(fresh, shown, mem['rec'])
    known = jnp.where(fresh, 1, mem['known'])
    seen = jnp.where(fresh, 1, mem['seen'])
    silent = jnp.where(fresh, 0, mem['silent'])

    # Count distinct activations (a single activation stays lit for at most 9
    # observations).  A partner that keeps signalling is evidence that silence
    # really means "recipe unchanged", so trust silence in proportion to how
    # talkative the partner has proved to be -- but never unboundedly, so that
    # a partner which stops signalling costs a few dishes rather than the rest
    # of the episode.
    bursts = mem['bursts'] + (fresh & ((t - mem['lit_t']) > 12)).astype(jnp.int32)
    lit_t = jnp.where(fresh, t, mem['lit_t'])
    trust = jnp.clip(bursts + _TRUST_ADD, _TRUST, _MAX_TRUST)

    # Never signalled: start cooking blind, alternating the guess after every
    # delivery.  A wrong dish leaves the recipe unchanged, so alternation is
    # right on the next attempt and is worth +20/3 per delivery in expectation.
    boot = (known == 0) & (t >= _BOOT)
    rec = jnp.where(boot, jnp.asarray([3, 0, 0], dtype=jnp.int32), rec)
    silent = jnp.where(boot, trust + 1, silent)
    known = jnp.where(boot, 1, known)

    swap = jnp.stack([rec[1], rec[0], rec[2]])
    shift = jnp.maximum(silent - trust, 0)
    belief = jnp.where((shift % 2) == 1, swap, rec)

    # --- counter bookkeeping ----------------------------------------------
    top_a, ok_a = _read(obs, row, col, 0, 5, 29, 34)
    top_b, ok_b = _read(obs, row, col, 0, 6, 29, 34)
    bot_a, ok_ja = _read(obs, row, col, 4, 5, 29, 34)
    bot_b, ok_jb = _read(obs, row, col, 4, 6, 29, 34)
    free_a = ok_a & (jnp.sum(top_a) == 0)
    free_b = ok_b & (jnp.sum(top_b) == 0)
    free_ja = ok_ja & (jnp.sum(bot_a) == 0)
    free_jb = ok_jb & (jnp.sum(bot_b) == 0)
    keep_a = ok_a & (top_a[1] > 0) & jnp.all(top_a[2:5] == belief)
    keep_b = ok_b & (top_b[1] > 0) & jnp.all(top_b[2:5] == belief)
    stash_ready = keep_a | keep_b
    stash_task = jnp.where(keep_a, 6, 7)
    dish_task = jnp.where(free_a, 6, jnp.where(free_b, 7, 5))
    junk_task = jnp.where(free_ja, 8,
                          jnp.where(free_jb, 9,
                                    jnp.where(free_a, 6, jnp.where(free_b, 7, 0))))

    # --- which ingredient the pot wants next -------------------------------
    deficit = jnp.maximum(belief - pot_counts, 0)
    consistent = jnp.all(pot_counts <= belief)
    # If the pot already disagrees with the belief, finish it as three of a
    # kind: such a dish still matches one of the possible recipes later on.
    fill = jnp.where(consistent,
                     jnp.argmax(deficit[:2]),
                     jnp.argmax(pot_counts[:2])).astype(jnp.int32)
    hold_idx = jnp.argmax(held_ing).astype(jnp.int32)

    commit = (known > 0) & ((seen > 0) | ((t - mem['deliv_t']) >= _WAIT))
    dish_ok = (jnp.all(held_ing == belief) & (held_n > 0)) | (known == 0)
    deposit_ok = pot_idle & (pot_total < 3) & (held_n > 0) & (hold_idx == fill)

    # --- priority cascade (later assignments win) --------------------------
    task = jnp.zeros((), dtype=jnp.int32)
    task = jnp.where(hands_free & pot_idle & (pot_total < 3) & commit,
                     jnp.where(fill == 0, 2, 3), task)
    task = jnp.where(hands_free & pot_idle & stash_ready, stash_task, task)
    task = jnp.where(hands_free & jnp.logical_not(pot_idle), 4, task)
    task = jnp.where((held_n > 0) & jnp.logical_not(deposit_ok), junk_task, task)
    task = jnp.where(deposit_ok, 1, task)
    plate_only = (held_plate > 0) & (held_cooked == 0)
    task = jnp.where(plate_only & cooking, 0, task)
    task = jnp.where(plate_only & pot_ready, 1, task)
    task = jnp.where(plate_only & pot_idle, junk_task, task)
    task = jnp.where((held_cooked > 0) & dish_ok, 5, task)
    task = jnp.where((held_cooked > 0) & jnp.logical_not(dish_ok), dish_task, task)

    action = _nav(row, col, face,
                  jnp.asarray(_T_R, dtype=jnp.int32)[task],
                  jnp.asarray(_T_C, dtype=jnp.int32)[task],
                  jnp.asarray(_T_F, dtype=jnp.int32)[task],
                  jnp.asarray(_T_I, dtype=jnp.int32)[task])
    action = jnp.where(ok, action, jnp.asarray(A_STAY, dtype=jnp.int32))

    delivered = (action == A_INTERACT) & (task == 5)
    out = dict(mem)
    out['rec'] = rec
    out['known'] = known
    out['seen'] = jnp.where(delivered, 0, seen)
    out['silent'] = jnp.where(delivered, silent + 1, silent)
    out['bursts'] = bursts
    out['lit_t'] = lit_t
    out['deliv_t'] = jnp.where(delivered, t, mem['deliv_t'])
    out['t'] = t + 1
    return action.astype(jnp.int32), out, task


def _act(obs, reset, memory):
    obs = obs.astype(jnp.int32)
    start = _init()
    mem = {key: jnp.where(reset, start[key], memory[key]) for key in start}

    left = jnp.max(obs[:, :, 23]) > 0
    sig_action, sig_mem, press = _signal(obs, mem)
    cook_action, cook_mem, task = _cook(obs, mem)

    action = jnp.where(left, sig_action, cook_action).astype(jnp.int32)
    out = {key: jnp.where(left, sig_mem[key], cook_mem[key]) for key in start}
    trace = {
        'role_left': left.astype(jnp.int32),
        'task': jnp.where(left, press + 20, task).astype(jnp.int32),
        'belief': out['rec'],
        'known': out['known'],
        'silent': out['silent'],
    }
    return action, out, trace


def init_agent_0():
    return _init()


def agent_0_act(obs, reset, memory, rng):
    return _act(obs, reset, memory)


def init_agent_1():
    return _init()


def agent_1_act(obs, reset, memory, rng):
    return _act(obs, reset, memory)
