"""Grounded-coordination policy for Overcooked V2 ``grounded_coord_simple``.

Roles (dictated by the layout, not by convention):
  * agent_0 lives in the left room. It can see the always-on recipe indicator
    and the pot, and it is the only seat that can reach the recipe button.
    It stands at (2,2) facing the button and presses it whenever the pot is
    empty and the display is dark, so the partner can read the *grounded*
    recipe display. Presses are rate limited with exponential backoff so a
    non-responsive partner cannot drain the team return.
  * agent_1 lives in the right room and can reach the pot, plate piles, the
    ingredient supplies and the delivery station. It reads the recipe from the
    button display (visible from column 5), then runs the full
    fetch -> pot -> plate -> scoop -> deliver cycle with facing-preserving paths.
    During the otherwise idle cook it stashes two ingredients of the current
    dish type on the counters (0,5)/(4,5); if the resampled recipe is unchanged
    the next fill is two steps shorter.
    Fallbacks: reuse the last recipe after a wait, guess after a long wait,
    and dump dishes that mismatch the known recipe onto a counter instead of
    delivering them for -20.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_button_cook_v2', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3

# Channels
CH_POT = 22
CH_BUTTON = 24
CH_TIMER = 39

# Counter slots reachable by agent_1: (counter_r, counter_c, stand_r, stand_c, facing)
# 0,1 are the "good" stash slots (5-step round trip to the pot); 2,3 are dump slots.
SLOTS = ((0, 5, 1, 5, F_UP), (4, 5, 3, 5, F_DOWN), (0, 6, 1, 6, F_UP), (4, 6, 3, 6, F_DOWN))

# Tuning
A0_EMPTY_OBS_BEFORE_PRESS = 3     # wait for delivery to have happened
A0_LAST_USEFUL_PRESS_STEP = 360   # a fresh cycle needs ~40 steps
A1_WAIT_REUSE_LAST = 10           # steps of silence before falling back on a prior
A1_WAIT_MAX = 30                  # cap for the adaptive silence wait
A1_WAIT_GUESS = 25                # steps of silence before guessing (no prior)
A1_STASH_MIN_TIMER = 12           # only start a stash run with this much cook time left


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _locate(mask):
    idx = jnp.argmax(mask.reshape(-1))
    return _i32(idx // 5), _i32(idx % 5)


def _cell(obs, r, c, lo, hi):
    """Return obs[r, c, lo:hi] (zeros if the local cell is outside the crop) and a visibility flag."""
    ok = (r >= 0) & (r <= 4) & (c >= 0) & (c <= 4)
    rc = jnp.clip(r, 0, 4)
    cc = jnp.clip(c, 0, 4)
    vals = obs[rc, cc, lo:hi]
    return jnp.where(ok, vals, 0), ok


def _nav(r, c, f, tr, tc, tf, final_action):
    """Vertical-first navigation on an obstacle-free room, then turn, then act."""
    vert = jnp.where(tr > r, A_DOWN, A_UP)
    horz = jnp.where(tc > c, A_RIGHT, A_LEFT)
    turn = jnp.asarray([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)[tf]
    act = jnp.where(r != tr, vert, jnp.where(c != tc, horz, jnp.where(f != tf, turn, final_action)))
    return _i32(act)


def _facing(obs):
    return _i32(jnp.argmax(obs[2, 2, 1:5]))


# --------------------------------------------------------------------------
# agent_0: signaller
# --------------------------------------------------------------------------
# memory: [step, empty_count, presses_in_phase, steps_since_press]

def init_agent_0():
    return jnp.asarray([0, 0, 0, 1000], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    mem = jnp.where(reset, init_agent_0(), memory)
    step, empty_count, presses, since_press = mem[0], mem[1], mem[2], mem[3]

    # Localise from the button (global (2,3)).
    br, bc = _locate(obs[:, :, CH_BUTTON])
    r = 4 - br
    c = 5 - bc
    f = _facing(obs)

    # Pot at global (2,4).
    pot_item, pot_vis = _cell(obs, 4 - r, 6 - c, 29, 34)
    pot_empty = pot_vis & (jnp.sum(pot_item) == 0)
    pot_full = pot_vis & (jnp.sum(pot_item) > 0)

    # Display at the button.
    disp, _ = _cell(obs, br, bc, 34, 39)
    disp_on = jnp.sum(disp[2:]) > 0

    # Always-on indicator at global (2,0) (diagnostic only).
    rec, rec_vis = _cell(obs, 4 - r, 2 - c, 34, 39)
    recipe = jnp.where(rec_vis & (jnp.sum(rec[2:]) > 0), jnp.argmax(rec[2:]), -1)

    empty_count = jnp.where(pot_empty, empty_count + 1, jnp.where(pot_full, 0, empty_count))
    presses = jnp.where(pot_full, 0, presses)

    in_place = (r == 2) & (c == 2) & (f == F_RIGHT)
    extra = jnp.where(presses < 3, 0, jnp.minimum(160, 10 * (2 ** jnp.clip(presses - 3, 0, 4))))
    gap_ok = since_press >= 10 + extra
    settled = (empty_count >= A0_EMPTY_OBS_BEFORE_PRESS) | (step < A0_EMPTY_OBS_BEFORE_PRESS)
    want_press = in_place & settled & (~disp_on) & gap_ok & (step <= A0_LAST_USEFUL_PRESS_STEP)

    nav_action = _nav(r, c, f, _i32(2), _i32(2), _i32(F_RIGHT), _i32(A_STAY))
    action = jnp.where(in_place, jnp.where(want_press, A_INTERACT, A_STAY), nav_action)
    action = _i32(action)

    presses = jnp.where(want_press, presses + 1, presses)
    since_press = jnp.where(want_press, 0, jnp.minimum(since_press + 1, 1000))
    new_mem = jnp.stack([step + 1, empty_count, presses, since_press]).astype(jnp.int32)
    trace = {
        'pos_r': _i32(r), 'pos_c': _i32(c), 'recipe': _i32(recipe),
        'disp_on': _i32(disp_on), 'pot_empty': _i32(pot_empty), 'press': _i32(want_press),
        'presses': _i32(presses),
    }
    return action, new_mem, trace


# --------------------------------------------------------------------------
# agent_1: cook
# --------------------------------------------------------------------------
# memory: [known, last_known, wait, expect_delivery, step, guessed,
#          slot0, slot1, slot2, slot3, n_disp, last_seen_on, last_guessed, lat_max]
# slot memory: -2 empty, -1 plate/other, 0..2 single raw ingredient of type k, 10+k cooked dish of type k.
# n_disp counts distinct display activations seen; last_guessed says whether the recipe used for
# the most recent delivery was unconfirmed (guessed / fallen back) rather than read from a display.

def init_agent_1():
    return jnp.asarray([-1, -1, 0, 0, 0, 0, -2, -2, -2, -2, 0, -1000, 0, 0], dtype=jnp.int32)


def _slot_info(obs, r, c, slot_mem):
    """Update remembered counter contents from whatever slots are currently visible."""
    new = []
    for i, (sr, sc, _, _, _) in enumerate(SLOTS):
        item, vis = _cell(obs, sr - r + 2, sc - c + 2, 29, 34)
        total = jnp.sum(item[2:])
        is_raw = (item[0] == 0) & (item[1] == 0) & (total == 1)
        is_dish = item[1] == 1
        seen = jnp.where(jnp.sum(item) == 0, -2,
                         jnp.where(is_raw, jnp.argmax(item[2:]),
                                   jnp.where(is_dish, 10 + jnp.argmax(item[2:]), -1)))
        new.append(jnp.where(vis, seen, slot_mem[i]))
    slot_mem = jnp.stack(new).astype(jnp.int32)
    occ = slot_mem != -2
    raw = jnp.where((slot_mem >= 0) & (slot_mem < 10), slot_mem, -1)
    dish = jnp.where(slot_mem >= 10, slot_mem - 10, -1)
    return occ, raw, dish, slot_mem


def _first_true(order, flags):
    """Index (into SLOTS) of the first slot in `order` whose flag is set, and whether any is."""
    flags_ord = flags[order]
    k = jnp.argmax(flags_ord)
    return order[k], jnp.any(flags_ord)


def agent_1_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    mem = jnp.where(reset, init_agent_1(), memory)
    known, last_known, wait, expect_del, step, guessed = mem[0], mem[1], mem[2], mem[3], mem[4], mem[5]
    slot_mem = mem[6:10]
    n_disp, last_seen_on, last_guessed, lat_max = mem[10], mem[11], mem[12], mem[13]

    # Localise from the pot (global (2,4)).
    pr, pc = _locate(obs[:, :, CH_POT])
    r = 4 - pr
    c = 6 - pc
    f = _facing(obs)

    # Inventory
    inv = obs[2, 2, 5:10]
    held_counts = inv[2:]
    held_n = jnp.sum(held_counts)
    holding_dish = inv[1] == 1
    holding_plate = (inv[0] == 1) & (~holding_dish)
    holding_ing = (held_n > 0) & (~holding_dish)
    held_type = _i32(jnp.argmax(held_counts))
    hands_empty = (~holding_dish) & (~holding_plate) & (~holding_ing)

    # Pot state
    pot_item = obs[pr, pc, 29:34]
    timer = obs[pr, pc, CH_TIMER]
    pot_counts = pot_item[2:]
    pot_count = jnp.sum(pot_counts)
    pot_type = _i32(jnp.argmax(pot_counts))
    pot_ready = pot_item[1] == 1
    pot_cooking = (timer > 0) & (~pot_ready)
    pot_idle = (~pot_ready) & (~pot_cooking)
    pot_empty = pot_idle & (pot_count == 0)

    # Button display at global (2,3)
    disp, disp_vis = _cell(obs, 4 - r, 5 - c, 34, 39)
    disp_on = disp_vis & (jnp.sum(disp[2:]) > 0)
    disp_recipe = _i32(jnp.argmax(disp[2:]))

    # Counter slots
    slot_occ, slot_raw, slot_dish, slot_mem = _slot_info(obs, r, c, slot_mem)
    slot_tab = jnp.asarray(SLOTS, dtype=jnp.int32)
    lower = r == 3
    good_order = jnp.where(lower, jnp.asarray([1, 0]), jnp.asarray([0, 1]))
    raw_order = jnp.where(lower, jnp.asarray([1, 0, 3, 2]), jnp.asarray([0, 1, 2, 3]))
    dump_order = jnp.where(lower, jnp.asarray([3, 2, 1, 0]), jnp.asarray([2, 3, 0, 1]))

    # ---- knowledge update ----
    delivered = (expect_del == 1) & hands_empty
    last_known = jnp.where(delivered & (known >= 0), known, last_known)
    last_guessed = jnp.where(delivered, guessed, last_guessed)
    known = jnp.where(delivered, -1, known)
    wait = jnp.where(delivered, 0, wait)
    guessed = jnp.where(delivered, 0, guessed)

    # A single activation is visible for 9 observations, so a display seen on after >= 10 steps
    # without seeing it on must be a new activation (moving out of view does not count).
    n_disp = jnp.where(disp_on & (step - last_seen_on >= 10), jnp.minimum(n_disp + 1, 100), n_disp)
    last_seen_on = jnp.where(disp_on, step, last_seen_on)
    # Observed partner latency: idle steps we had waited when a display finally appeared.
    lat_max = jnp.where(disp_on & (known < 0), jnp.maximum(lat_max, wait), lat_max)
    known = jnp.where(disp_on, disp_recipe, known)
    guessed = jnp.where(disp_on, 0, guessed)
    wait = jnp.where(disp_on, 0, wait)

    idle_unknown = (known < 0) & hands_empty & pot_empty
    wait = jnp.where(idle_unknown, wait + 1, wait)
    # Silence after a delivery. If the partner has shown >= 2 displays it is communicative, so
    # silence most plausibly means "recipe unchanged": reuse. Likewise reuse once after a
    # display-confirmed recipe. Otherwise (unconfirmed prior, partner mostly silent) a wrong
    # delivery leaves the recipe unchanged, so the other recipe is more likely: flip.
    wait_thresh = jnp.clip(jnp.maximum(A1_WAIT_REUSE_LAST, lat_max + 5), A1_WAIT_REUSE_LAST, A1_WAIT_MAX)
    fallback = idle_unknown & (wait > wait_thresh) & (last_known >= 0)
    reuse_ok = (n_disp >= 2) | (last_guessed == 0)
    known = jnp.where(fallback, jnp.where(reuse_ok, last_known, 1 - last_known), known)
    guess_bit = _i32(jax.random.randint(rng, (), 0, 2))
    do_guess = idle_unknown & (wait > A1_WAIT_GUESS) & (last_known < 0)
    known = jnp.where(do_guess, guess_bit, known)
    guessed = jnp.where(do_guess | fallback, 1, guessed)
    wait = jnp.where(fallback | do_guess, 0, wait)

    # ---- targets (default: wait at (2,5) facing the pot) ----
    tr, tc, tf, fin = _i32(2), _i32(5), _i32(F_LEFT), _i32(A_STAY)
    mode = _i32(0)

    def set_target(sel, ntr, ntc, ntf, nfin, nmode, tr, tc, tf, fin, mode):
        return (jnp.where(sel, ntr, tr), jnp.where(sel, ntc, tc), jnp.where(sel, ntf, tf),
                jnp.where(sel, nfin, fin), jnp.where(sel, nmode, mode))

    # D. empty hands ---------------------------------------------------
    need_pot_type = hands_empty & pot_idle & (pot_count > 0) & (pot_type <= 1)
    need_known = hands_empty & pot_empty & (known >= 0) & (known <= 1)
    need_type = jnp.where(need_pot_type, pot_type, known)
    need = need_pot_type | need_known
    # Stash pickup if a good slot holds the needed raw type.
    has_need = slot_raw == need_type
    good_has = jnp.stack([has_need[0], has_need[1]])
    pick_idx, pick_any = _first_true(good_order, good_has)
    use_stash = need & pick_any
    itr = jnp.where(need_type == 0, 1, 3)
    tr, tc, tf, fin, mode = set_target(need, itr, 6, F_RIGHT, A_INTERACT, 1, tr, tc, tf, fin, mode)
    tr, tc, tf, fin, mode = set_target(use_stash, slot_tab[pick_idx, 2], slot_tab[pick_idx, 3],
                                       slot_tab[pick_idx, 4], A_INTERACT, 7, tr, tc, tf, fin, mode)
    # During cook: stash run if time permits and a good slot is free; otherwise fetch plate.
    good_free = jnp.stack([~slot_occ[0], ~slot_occ[1]])
    stash_type = jnp.where((known >= 0) & (known <= 1), known, pot_type)
    n_free = jnp.sum(~slot_occ)
    stash_run = hands_empty & pot_cooking & (timer >= A1_STASH_MIN_TIMER) & jnp.any(good_free) \
        & (stash_type <= 1) & (n_free >= 2)
    str_ = jnp.where(stash_type == 0, 1, 3)
    tr, tc, tf, fin, mode = set_target(stash_run, str_, 6, F_RIGHT, A_INTERACT, 8, tr, tc, tf, fin, mode)
    fetch_plate = hands_empty & (pot_ready | pot_cooking) & (~stash_run)
    plate_r = jnp.where(lower, 3, 1)
    tr, tc, tf, fin, mode = set_target(fetch_plate, plate_r, 5, F_LEFT, A_INTERACT, 2, tr, tc, tf, fin, mode)
    # A counter dish matching the known recipe is a 7-step +20: reclaim it before anything else,
    # unless the pot is about to finish (keep the plate schedule).
    dish_match = (slot_dish == known) & (known >= 0)
    rd_idx, rd_any = _first_true(dump_order, dish_match)
    reclaim = hands_empty & rd_any & (~(pot_cooking & (timer < A1_STASH_MIN_TIMER)))
    tr, tc, tf, fin, mode = set_target(reclaim, slot_tab[rd_idx, 2], slot_tab[rd_idx, 3],
                                       slot_tab[rd_idx, 4], A_INTERACT, 9, tr, tc, tf, fin, mode)

    # C. holding ingredient ------------------------------------------
    compat = pot_idle & (pot_count < 3) & jnp.where(
        pot_count == 0, (known < 0) | (known == held_type), pot_type == held_type)
    deposit = holding_ing & compat
    place_ing = holding_ing & (~compat)
    tr, tc, tf, fin, mode = set_target(deposit, 2, 5, F_LEFT, A_INTERACT, 3, tr, tc, tf, fin, mode)
    ing_idx, ing_any = _first_true(raw_order, ~slot_occ)
    tr, tc, tf, fin, mode = set_target(place_ing & ing_any, slot_tab[ing_idx, 2], slot_tab[ing_idx, 3],
                                       slot_tab[ing_idx, 4], A_INTERACT, 6, tr, tc, tf, fin, mode)

    # B. holding plate --------------------------------------------------
    scoop = holding_plate & pot_ready
    wait_cook = holding_plate & pot_cooking
    tr, tc, tf, fin, mode = set_target(scoop | wait_cook, 2, 5, F_LEFT,
                                       jnp.where(scoop, A_INTERACT, A_STAY), 4, tr, tc, tf, fin, mode)
    dump_plate = holding_plate & pot_idle

    # A. holding dish ---------------------------------------------------
    mismatch = holding_dish & (known >= 0) & (held_type != known)
    deliver = holding_dish & (~mismatch)
    tr, tc, tf, fin, mode = set_target(deliver, 2, 6, F_RIGHT, A_INTERACT, 5, tr, tc, tf, fin, mode)
    dump_idx, dump_any = _first_true(dump_order, ~slot_occ)
    dump = (dump_plate | mismatch) & dump_any
    tr, tc, tf, fin, mode = set_target(dump, slot_tab[dump_idx, 2], slot_tab[dump_idx, 3],
                                       slot_tab[dump_idx, 4], A_INTERACT, 6, tr, tc, tf, fin, mode)
    # If no counter is free for a mismatched dish, deliver it anyway (expected value 0 vs. stall).
    tr, tc, tf, fin, mode = set_target(mismatch & (~dump_any), 2, 6, F_RIGHT, A_INTERACT, 5, tr, tc, tf, fin, mode)

    action = _nav(r, c, f, tr, tc, tf, fin)

    delivering = holding_dish & (mode == 5)
    expect_del_new = _i32(delivering & (r == 2) & (c == 6) & (f == F_RIGHT) & (action == A_INTERACT))

    new_mem = jnp.concatenate([
        jnp.stack([known, last_known, wait, expect_del_new, step + 1, guessed]).astype(jnp.int32),
        slot_mem,
        jnp.stack([n_disp, last_seen_on, last_guessed, lat_max]).astype(jnp.int32)]).astype(jnp.int32)
    trace = {
        'pos_r': _i32(r), 'pos_c': _i32(c), 'known': _i32(known), 'mode': _i32(mode),
        'disp_on': _i32(disp_on), 'pot_count': _i32(pot_count), 'timer': _i32(timer),
        'guessed': _i32(guessed), 'wait': _i32(wait), 'n_disp': _i32(n_disp), 'lat_max': _i32(lat_max),
        'stashA': _i32(slot_raw[0]), 'stashB': _i32(slot_raw[1]),
    }
    return action, new_mem, trace
