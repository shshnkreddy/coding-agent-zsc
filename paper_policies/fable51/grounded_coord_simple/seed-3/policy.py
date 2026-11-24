"""Grounded coordination policy for Overcooked V2 grounded_coord_simple.

Layout facts (see context/environment/README.md):

    WW2WWWWW
    W  WB  0
    R ALPA X
    W  WB  1
    WW2WWWWW

* agent_0 lives in the left room (rows 1-3, cols 1-2).  It always sees the
  always-on recipe indicator R at (2,0), sees the button L at (2,3) and, from
  column 2, sees the pot at (2,4).  It is the only player able to press the
  button (from (2,2) facing right).  It can reach nothing else useful.
* agent_1 lives in the right room (rows 1-3, cols 5-6).  It is the only player
  able to use the pot (from (2,5) facing left), plate piles, ingredient
  supplies 0/1 and the delivery station.  It never sees R; it sees the button
  display only from column 5.

Protocol (grounded, not a convention): agent_0 presses the button when the pot
is empty and nothing is displayed, so the true current recipe is shown to the
cook.  agent_1 reads any displayed recipe, cooks it, delivers it, stores a
dish that turns out to be wrong on a counter for later, and falls back to
guessing only when no recipe information arrives for a while.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_button_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Actions
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids: UP=0, DOWN=1, RIGHT=2, LEFT=3 ; facing produced by each move action
FACE_OF_ACTION_LIST = (2, 1, 3, 0)

# Observation channels
CH_POT, CH_RECIPE_IND = 22, 23
CH_ITEM_LO, CH_ITEM_HI = 29, 34      # plate, cooked, ing0, ing1, ing2 on grid cell
CH_RECIPE_LO, CH_RECIPE_HI = 36, 39  # displayed recipe ingredient counts
CH_TIMER = 39

BIG = 1000


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _cell(obs, i, j):
    """Return the 40-channel vector at local (i, j), zeros if outside the crop."""
    valid = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    ic = jnp.clip(i, 0, 4)
    jc = jnp.clip(j, 0, 4)
    return jnp.where(valid, obs[ic, jc, :], 0), valid


def _locate(obs, ch):
    flat = obs[:, :, ch].reshape(-1)
    idx = jnp.argmax(flat)
    return idx // 5, idx % 5


def _go_interact(r, c, facing, tr, tc, face_act):
    """Move to floor cell (tr, tc), turn to face_act, then INTERACT."""
    dr = tr - r
    dc = tc - c
    vert = jnp.where(dr > 0, DOWN, UP)
    horiz = jnp.where(dc > 0, RIGHT, LEFT)
    need_face = facing != jnp.asarray(FACE_OF_ACTION_LIST, dtype=jnp.int32)[face_act]
    return _i32(jnp.where(dr != 0, vert,
                jnp.where(dc != 0, horiz,
                jnp.where(need_face, face_act, INTERACT))))


def _wait_at(r, c, facing, tr, tc, face_act):
    nav = _go_interact(r, c, facing, tr, tc, face_act)
    return _i32(jnp.where(nav == INTERACT, STAY, nav))


def _nearest(r, c, cand_r, cand_c, cand_act, valid):
    dist = jnp.abs(cand_r - r) + jnp.abs(cand_c - c) + jnp.where(valid, 0, BIG)
    k = jnp.argmin(dist)
    return cand_r[k], cand_c[k], cand_act[k], valid[k]


def _reset_mem(reset, init, memory):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)


# ---------------------------------------------------------------------------
# agent_0: recipe signaller
# ---------------------------------------------------------------------------

def init_agent_0():
    return {
        'since_press': _i32(BIG),   # steps since our last button press
        'n_press': _i32(0),         # presses since pot last had contents
        'empty_run': _i32(5),       # consecutive observations of an empty pot
    }


def agent_0_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    mem = _reset_mem(reset, init_agent_0(), memory)

    ri, rj = _locate(obs, CH_RECIPE_IND)       # R at global (2, 0)
    r = 4 - ri
    c = 2 - rj
    facing = jnp.argmax(obs[2, 2, 1:5])

    button, _ = _cell(obs, 4 - r, 5 - c)        # L at global (2, 3)
    display_active = button[CH_RECIPE_LO:CH_RECIPE_HI].sum() > 0

    pot, pot_visible = _cell(obs, 4 - r, 6 - c)  # P at global (2, 4)
    pot_cnt = pot[31:34].sum()
    pot_empty = pot_visible & (pot_cnt == 0)
    pot_nonempty = pot_visible & (pot_cnt > 0)

    empty_run = jnp.where(pot_nonempty, 0,
                          jnp.where(pot_empty, mem['empty_run'] + 1, mem['empty_run']))
    n_press = jnp.where(pot_nonempty, 0, mem['n_press'])
    cooldown = jnp.minimum(20 + 20 * n_press, 90)

    at_spot = (r == 2) & (c == 2) & (facing == 2)
    want_press = (at_spot & pot_empty & (~display_active) & (empty_run >= 4)
                  & (mem['since_press'] >= cooldown))

    nav = _go_interact(r, c, facing, 2, 2, RIGHT)
    action = _i32(jnp.where(at_spot, jnp.where(want_press, INTERACT, STAY), nav))

    new_mem = {
        'since_press': _i32(jnp.where(want_press, 0, jnp.minimum(mem['since_press'] + 1, BIG))),
        'n_press': _i32(jnp.where(want_press, n_press + 1, n_press)),
        'empty_run': _i32(jnp.minimum(empty_run, BIG)),
    }
    trace = {
        'press': want_press,
        'pot_cnt': _i32(pot_cnt),
        'display_active': display_active,
        'row': _i32(r), 'col': _i32(c),
    }
    return action, new_mem, trace


# ---------------------------------------------------------------------------
# agent_1: cook / deliverer
# ---------------------------------------------------------------------------

# Storage counters reachable by agent_1 and their access cell / facing action
STORE_R_LIST = (0, 0, 4, 4)
STORE_C_LIST = (5, 6, 5, 6)
STORE_AR_LIST = (1, 1, 3, 3)
STORE_AC_LIST = (5, 6, 5, 6)
STORE_ACT_LIST = (UP, UP, DOWN, DOWN)


def _store_tables():
    return (jnp.asarray(STORE_R_LIST, dtype=jnp.int32),
            jnp.asarray(STORE_C_LIST, dtype=jnp.int32),
            jnp.asarray(STORE_AR_LIST, dtype=jnp.int32),
            jnp.asarray(STORE_AC_LIST, dtype=jnp.int32),
            jnp.asarray(STORE_ACT_LIST, dtype=jnp.int32))

# Item codes stored in memory for counters
CODE_EMPTY, CODE_DISH0, CODE_DISH1, CODE_DISH2, CODE_RAW0, CODE_RAW1, CODE_RAW2, CODE_PLATE = 0, 1, 2, 3, 4, 5, 6, 7

HOLD_WAIT_LONG = 8     # hold a guessed dish at (2,5) hoping for a display (reactive partner)
HOLD_WAIT_SHORT = 2    # ... when the partner has shown no sign of re-signalling
W_FIRST = 14           # wait for a display before the first guess (no display ever seen)
W_ONCE = 16            # first wait after the partner has displayed exactly once
W_ONCE_AGAIN = 3       # later waits when the partner displayed once and never re-signalled
W_REPEAT = 1           # wait after a guessed delivery when no display was ever seen
W_MULTI = 30           # wait when the partner has re-activated the display at least once
SPEC_DELAY = 4         # unknown steps before speculatively fetching the guessed ingredient
W_MULTI_AGAIN = 20     # ... after one expired wait
W_MULTI_LATER = 8      # ... after two or more expired waits


def init_agent_1():
    return {
        'known': _i32(0),        # 0 unknown, 1 recipe0, 2 recipe1
        'guess': _i32(0),        # fallback guess type (0/1)
        'unk_steps': _i32(0),    # steps since recipe became unknown
        'seen_disp': _i32(0),    # ever saw a display this episode
        'n_events': _i32(0),     # distinct display activations observed
        'since_disp': _i32(BIG), # steps since a display was last seen
        'n_fallback': _i32(0),   # fallback waits that expired without a display
        'n_guess': _i32(0),      # guessed deliveries so far
        'hold_wait': _i32(0),    # steps holding a cooked dish with unknown recipe
        'store': jnp.zeros((4,), dtype=jnp.int32),
    }


def _classify_item(block):
    plate, cooked = block[0] > 0, block[1] > 0
    ing = block[2:5]
    cnt = ing.sum()
    t = jnp.argmax(ing)
    return _i32(jnp.where(cooked, CODE_DISH0 + t,
                jnp.where(cnt > 0, CODE_RAW0 + t,
                jnp.where(plate, CODE_PLATE, CODE_EMPTY))))


def agent_1_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs, dtype=jnp.int32)
    mem = _reset_mem(reset, init_agent_1(), memory)
    STORE_R, STORE_C, STORE_AR, STORE_AC, STORE_ACT = _store_tables()

    pi, pj = _locate(obs, CH_POT)              # pot at global (2, 4)
    r = 4 - pi
    c = 6 - pj
    me = obs[2, 2]
    facing = jnp.argmax(me[1:5])

    h_plate = me[5] > 0
    h_cooked = me[6] > 0
    h_ing = me[7:10]
    h_cnt = h_ing.sum()
    holding_cooked = h_cooked
    holding_plate = h_plate & (~h_cooked)
    holding_raw = (h_cnt > 0) & (~h_cooked)
    hands_empty = ~(holding_cooked | holding_plate | holding_raw)
    held_type = jnp.argmax(h_ing)

    pot = obs[pi, pj]
    pot_cnt = pot[31:34].sum()
    pot_cooked = pot[30] > 0
    pot_timer = pot[CH_TIMER]
    pot_cooking = (pot_timer > 0) & (~pot_cooked)
    pot_idle = (~pot_cooked) & (~pot_cooking)
    pot_type = jnp.argmax(pot[31:34])

    # Button display (visible only from column 5)
    button, _ = _cell(obs, 4 - r, 5 - c)
    disp = button[CH_RECIPE_LO:CH_RECIPE_HI]
    disp_active = disp.sum() > 0
    disp_type = jnp.argmax(disp)
    known = jnp.where(disp_active, disp_type + 1, mem['known'])
    seen_disp = jnp.maximum(mem['seen_disp'], disp_active.astype(jnp.int32))
    unk_steps = jnp.where(disp_active, 0, mem['unk_steps'] + 1)
    onset = disp_active & (mem['since_disp'] >= 10)
    n_events = jnp.minimum(mem['n_events'] + onset.astype(jnp.int32), BIG)
    since_disp = jnp.where(disp_active, 0, jnp.minimum(mem['since_disp'] + 1, BIG))

    # Counter storage memory, refreshed from the observation where visible
    def _upd(k, prev):
        blk, vis = _cell(obs, STORE_R[k] - r + 2, STORE_C[k] - c + 2)
        return jnp.where(vis, _classify_item(blk[CH_ITEM_LO:CH_ITEM_HI]), prev)
    store = jnp.stack([_upd(k, mem['store'][k]) for k in range(4)])

    target = jnp.where(known > 0, known - 1, mem['guess'])      # 0/1
    W = jnp.where(n_events >= 2, jnp.where(mem['n_fallback'] == 0, W_MULTI,
                                  jnp.where(mem['n_fallback'] == 1, W_MULTI_AGAIN, W_MULTI_LATER)),
        jnp.where(n_events == 1, jnp.where(mem['n_fallback'] == 0, W_ONCE, W_ONCE_AGAIN),
        jnp.where(mem['n_guess'] > 0, W_REPEAT, W_FIRST)))
    in_fallback = (known == 0) & (unk_steps >= W)
    fallback_start = in_fallback & (unk_steps == W)
    n_fallback = jnp.minimum(mem['n_fallback'] + fallback_start.astype(jnp.int32), BIG)
    hold_limit = jnp.where((n_events >= 2) | ((n_events == 1) & (mem['n_fallback'] == 0)),
                           HOLD_WAIT_LONG, HOLD_WAIT_SHORT)
    work_type = jnp.where(pot_cnt > 0, pot_type, target)
    hold_wait = jnp.where(holding_cooked & (known == 0), mem['hold_wait'] + 1, 0)

    # --- candidate actions -------------------------------------------------
    act_deliver = _go_interact(r, c, facing, 2, 6, RIGHT)
    act_pot = _go_interact(r, c, facing, 2, 5, LEFT)
    act_wait = _wait_at(r, c, facing, 2, 5, LEFT)

    # fetch ingredient of type work_type: supply or a counter holding it
    sup_r = jnp.where(work_type == 0, 1, 3)
    cand_r = jnp.concatenate([jnp.asarray([sup_r]), STORE_AR])
    cand_c = jnp.concatenate([jnp.asarray([6]), STORE_AC])
    cand_a = jnp.concatenate([jnp.asarray([RIGHT]), STORE_ACT])
    cand_v = jnp.concatenate([jnp.asarray([work_type < 2]), store == (CODE_RAW0 + work_type)])
    fr, fc, fa, fv = _nearest(r, c, cand_r, cand_c, cand_a, cand_v)
    act_fetch_ing = jnp.where(fv, _go_interact(r, c, facing, fr, fc, fa), act_wait)

    # fetch plate: piles at (1,4)/(3,4) from (1,5)/(3,5) facing left, or a stored plate
    cand_r = jnp.concatenate([jnp.asarray([1, 3]), STORE_AR])
    cand_c = jnp.concatenate([jnp.asarray([5, 5]), STORE_AC])
    cand_a = jnp.concatenate([jnp.asarray([LEFT, LEFT]), STORE_ACT])
    cand_v = jnp.concatenate([jnp.asarray([True, True]), store == CODE_PLATE])
    pr, pc, pa, _ = _nearest(r, c, cand_r, cand_c, cand_a, cand_v)
    act_fetch_plate = _go_interact(r, c, facing, pr, pc, pa)

    # drop held item on the nearest empty counter
    dr_, dc_, da_, dv_ = _nearest(r, c, STORE_AR, STORE_AC, STORE_ACT, store == CODE_EMPTY)
    act_drop = jnp.where(dv_, _go_interact(r, c, facing, dr_, dc_, da_), act_wait)

    # pick up a stored cooked dish matching the known recipe
    have_known_dish = jnp.any(store == (CODE_DISH0 + target))
    kr, kc, ka, kv = _nearest(r, c, STORE_AR, STORE_AC, STORE_ACT, store == (CODE_DISH0 + target))
    act_pick_known = _go_interact(r, c, facing, kr, kc, ka)

    # pick up any stored cooked dish (fallback guess)
    is_dish = (store >= CODE_DISH0) & (store <= CODE_DISH2)
    have_any_dish = jnp.any(is_dish)
    ar_, ac_, aa_, _ = _nearest(r, c, STORE_AR, STORE_AC, STORE_ACT, is_dish)
    act_pick_any = _go_interact(r, c, facing, ar_, ac_, aa_)

    # --- decision ----------------------------------------------------------
    # empty hands
    e_fetch_plate = pot_cooked | pot_cooking
    e_fill = pot_idle & (pot_cnt > 0)
    e_known = pot_idle & (pot_cnt == 0) & (known > 0)
    e_fallback = pot_idle & (pot_cnt == 0) & (known == 0) & in_fallback
    # unknown recipe, still waiting: speculatively fetch the guessed ingredient (never committed
    # to the pot until the recipe is known or the wait expires)
    e_spec = (pot_idle & (pot_cnt == 0) & (known == 0) & (~in_fallback) & (~have_any_dish)
              & (unk_steps >= SPEC_DELAY))
    act_empty = jnp.where(e_fetch_plate, act_fetch_plate,
               jnp.where(e_fill, act_fetch_ing,
               jnp.where(e_known, jnp.where(have_known_dish, act_pick_known, act_fetch_ing),
               jnp.where(e_fallback, jnp.where(have_any_dish, act_pick_any, act_fetch_ing),
               jnp.where(e_spec, act_fetch_ing,
               act_wait)))))

    # holding cooked dish
    match = (known > 0) & (held_type == known - 1)
    mismatch = (known > 0) & (held_type != known - 1)
    guess_deliver = (known == 0) & (hold_wait > hold_limit)
    act_cooked = jnp.where(match | guess_deliver, act_deliver,
                jnp.where(mismatch, act_drop, act_wait))

    # holding plate
    act_plate = jnp.where(pot_cooked, act_pot,
               jnp.where(pot_cooking, act_wait, act_drop))

    # holding raw ingredient
    can_add = pot_idle & (pot_cnt < 3) & (
        ((pot_cnt > 0) & (held_type == pot_type))
        | ((pot_cnt == 0) & (known > 0) & (held_type == known - 1))
        | ((pot_cnt == 0) & (known == 0) & in_fallback))
    hold_raw = pot_idle & (pot_cnt == 0) & (known == 0) & (~in_fallback)
    act_raw = jnp.where(can_add, act_pot, jnp.where(hold_raw, act_wait, act_drop))

    action = _i32(jnp.where(holding_cooked, act_cooked,
                  jnp.where(holding_plate, act_plate,
                  jnp.where(holding_raw, act_raw, act_empty))))

    # --- bookkeeping -------------------------------------------------------
    delivering = holding_cooked & (match | guess_deliver) & (action == INTERACT) & (r == 2) & (c == 6)
    known_deliver = delivering & match
    guess_delivered = delivering & (~match)
    new_known = jnp.where(known_deliver, 0, known)
    commit_guess = holding_raw & can_add & (pot_cnt == 0) & (known == 0) & (action == INTERACT)
    # after a guessed delivery, alternate; after a known delivery, guess that the recipe repeats
    # (neutral for most partners, right for partners that only signal on recipe change)
    new_guess = jnp.where(guess_delivered, 1 - held_type,
                jnp.where(known_deliver, held_type,
                jnp.where(commit_guess, held_type, mem['guess'])))
    new_unk = jnp.where(delivering, 0, unk_steps)
    new_n_guess = mem['n_guess'] + guess_delivered.astype(jnp.int32)

    new_mem = {
        'known': _i32(new_known),
        'guess': _i32(new_guess),
        'unk_steps': _i32(jnp.minimum(new_unk, BIG)),
        'seen_disp': _i32(seen_disp),
        'n_events': _i32(n_events),
        'since_disp': _i32(since_disp),
        'n_fallback': _i32(n_fallback),
        'n_guess': _i32(jnp.minimum(new_n_guess, BIG)),
        'hold_wait': _i32(jnp.minimum(hold_wait, BIG)),
        'store': store.astype(jnp.int32),
    }
    trace = {
        'known': _i32(known),
        'row': _i32(r), 'col': _i32(c),
        'pot_cnt': _i32(pot_cnt),
        'in_fallback': in_fallback,
        'delivering': delivering,
    }
    return action, new_mem, trace
