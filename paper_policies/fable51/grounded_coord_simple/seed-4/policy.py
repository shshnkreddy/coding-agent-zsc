"""grounded_coord_simple policy.

Agent 0 (left room) can reach nothing but the recipe button, so it acts as the
button operator: it reads the always-on indicator, watches the pot, and presses
the button when the cook is likely to need the recipe (episode start, right
after the pot is emptied, and with escalating backoff while the pot stays empty).

Agent 1 (right room) does every cooking step alone. It tracks a Bayesian belief
over the two recipes from button displays and its own deliveries, cooks the most
likely dish, stashes dishes it believes are wrong on counters for later, and
falls back to alternating guesses if no partner information ever arrives.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'button_operator_belief_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
MOVE_FOR_FACING_T = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# Channels
CH_WALL, CH_GOAL, CH_POT, CH_RECIPE_IND, CH_BUTTON, CH_PLATE_PILE = 20, 21, 22, 23, 24, 25
CH_ITEM_PLATE, CH_ITEM_COOKED, CH_ITEM_I0, CH_ITEM_I1, CH_ITEM_I2 = 29, 30, 31, 32, 33
CH_REC_I0, CH_REC_I1 = 36, 37
CH_TIMER = 39

# Global object coordinates
POT_R, POT_C = 2, 4
RIND_R, RIND_C = 2, 0
BTN_R, BTN_C = 2, 3

# Agent-1 stash counters: (0,5),(0,6),(4,5),(4,6); access cell and facing.
SLOT_R_T = (0, 0, 4, 4)
SLOT_C_T = (5, 6, 5, 6)
SLOT_AR_T = (1, 1, 3, 3)
SLOT_AC_T = (5, 6, 5, 6)
SLOT_AF_T = (F_UP, F_UP, F_DOWN, F_DOWN)
SLOT_GROUP_T = (0, 0, 1, 1)


def _slot_arrays():
    mk = lambda t: jnp.asarray(t, dtype=jnp.int32)
    return (mk(SLOT_R_T), mk(SLOT_C_T), mk(SLOT_AR_T), mk(SLOT_AC_T), mk(SLOT_AF_T), mk(SLOT_GROUP_T))

START_WAIT = 8      # steps to wait for a recipe display at episode start
POST_WAIT = 4       # steps to wait after a delivery for the partner's signal
LAST_USEFUL_PRESS = 358  # agent 0 never presses at or after this step
# Probability that a silent partner would have signalled a recipe change, by
# observed partner style: displays after deliveries only ever showed changed
# recipes (change-only) vs. also showed unchanged ones (mixed).
SILENCE_WEIGHT_CHANGE_ONLY = 0.9
SILENCE_WEIGHT_MIXED = 0.5
SILENCE_WEIGHT_SEEN_ONLY = 0.7  # partner pressed, but never yet after a delivery


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _localize(obs, channel, gr, gc):
    """Global (row, col) of self using a unique static object at (gr, gc)."""
    flat = obs[:, :, channel].reshape(-1)
    idx = jnp.argmax(flat)
    lr = idx // 5
    lc = idx % 5
    return _i32(gr - (lr - 2)), _i32(gc - (lc - 2))


def _cell(obs, sr, sc, gr, gc):
    """Channel vector of global cell (gr, gc) and whether it is in view."""
    lr = gr - sr + 2
    lc = gc - sc + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    lr_c = jnp.clip(lr, 0, 4)
    lc_c = jnp.clip(lc, 0, 4)
    return obs[lr_c, lc_c, :], visible


def _facing(obs):
    return _i32(jnp.argmax(obs[2, 2, 1:5]))


def _goto(r, c, f, tr, tc, tf, final_action):
    """Move to (tr, tc), face tf, then emit final_action."""
    dr = tr - r
    dc = tc - c
    at = (dr == 0) & (dc == 0)
    vert = jnp.where(dr > 0, A_DOWN, A_UP)
    horiz = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    move = jnp.where(dr != 0, vert, horiz)
    turn = jnp.asarray(MOVE_FOR_FACING_T, dtype=jnp.int32)[tf]
    return _i32(jnp.where(at, jnp.where(f == tf, final_action, turn), move))


def _reset_tree(reset, init, mem):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, mem)


# --------------------------------------------------------------------------
# Agent 0: button operator
# --------------------------------------------------------------------------

def init_agent_0():
    return {
        't': _i32(0),
        'since_press': _i32(1000),
        'presses': _i32(0),
        'prev_recipe': _i32(-1),
        'prev_ready': _i32(0),
        'clear_age': _i32(1000),
        'backoff': _i32(14),
        'warned': _i32(0),
        'empty_age': _i32(0),
        'prev_cnt': _i32(0),
        'wrong_deposits': _i32(0),
    }


def agent_0_act(obs, reset, memory, rng):
    mem = _reset_tree(reset, init_agent_0(), memory)
    obs = jnp.asarray(obs, dtype=jnp.int32)
    r, c = _localize(obs, CH_RECIPE_IND, RIND_R, RIND_C)
    f = _facing(obs)

    rcell, _ = _cell(obs, r, c, RIND_R, RIND_C)
    cur_recipe = _i32(jnp.where(rcell[CH_REC_I0] > 0, 0, jnp.where(rcell[CH_REC_I1] > 0, 1, -1)))

    pcell, pot_vis = _cell(obs, r, c, POT_R, POT_C)
    pot_cnt = pcell[CH_ITEM_I0] + pcell[CH_ITEM_I1] + pcell[CH_ITEM_I2]
    pot_ready = pcell[CH_ITEM_COOKED] > 0
    pot_timer = pcell[CH_TIMER]
    pot_type = _i32(jnp.argmax(pcell[CH_ITEM_I0:CH_ITEM_I2 + 1]))
    pot_empty = pot_vis & (pot_cnt == 0) & (~pot_ready)
    pot_partial = pot_vis & (pot_cnt > 0) & (pot_timer == 0) & (~pot_ready)
    pot_active = pot_vis & ((pot_cnt > 0) | pot_ready)

    bcell, btn_vis = _cell(obs, r, c, BTN_R, BTN_C)
    display_on = btn_vis & ((bcell[CH_REC_I0] + bcell[CH_REC_I1]) > 0)

    in_position = (r == 2) & (c == 2) & (f == F_RIGHT)

    cleared = pot_vis & (mem['prev_ready'] > 0) & (~pot_ready) & (pot_cnt == 0)
    clear_age = _i32(jnp.where(cleared, 0, jnp.minimum(mem['clear_age'] + 1, 1000)))
    recipe_changed = (mem['prev_recipe'] >= 0) & (cur_recipe >= 0) & (cur_recipe != mem['prev_recipe'])
    since = mem['since_press']

    backoff = _i32(jnp.where(pot_active, 14, mem['backoff']))
    empty_age = _i32(jnp.where(pot_vis, jnp.where(pot_empty, jnp.minimum(mem['empty_age'] + 1, 1000), 0),
                               jnp.minimum(mem['empty_age'] + 1, 1000)))
    warned = _i32(jnp.where(pot_empty | pot_ready, 0, mem['warned']))

    reason_initial = mem['presses'] == 0
    reason_clear = clear_age == 2
    reason_recipe = recipe_changed & (since > 5)
    reason_backoff = pot_empty & (since >= 9 + backoff) & (empty_age >= 9 + backoff)
    reason_wrong = pot_partial & (pot_type != cur_recipe) & (cur_recipe >= 0) & (since >= 9) & (warned == 0)

    # Partner ignoring the button: it started a wrong-type pot shortly after a display.
    became_partial = pot_partial & (mem['prev_cnt'] == 0)
    wrong_now = became_partial & (cur_recipe >= 0) & (pot_type != cur_recipe) & (since <= 15)
    wrong_deposits = _i32(jnp.minimum(mem['wrong_deposits'] + _i32(wrong_now), 100))
    suppressed = wrong_deposits >= 2

    want_press = in_position & (~suppressed) & (mem['t'] < LAST_USEFUL_PRESS) & (~display_on | recipe_changed) & (
        reason_initial | reason_clear | reason_recipe | reason_backoff | reason_wrong)

    action = _goto(r, c, f, _i32(2), _i32(2), _i32(F_RIGHT), _i32(A_STAY))
    action = _i32(jnp.where(want_press, A_INTERACT, action))

    new_mem = {
        't': mem['t'] + 1,
        'since_press': _i32(jnp.where(want_press, 0, jnp.minimum(since + 1, 1000))),
        'presses': mem['presses'] + _i32(want_press),
        'prev_recipe': _i32(jnp.where(cur_recipe >= 0, cur_recipe, mem['prev_recipe'])),
        'prev_ready': _i32(jnp.where(pot_vis, pot_ready, mem['prev_ready'])),
        'clear_age': clear_age,
        'backoff': _i32(jnp.where(want_press & reason_backoff, jnp.minimum(backoff * 2, 120), backoff)),
        'warned': _i32(jnp.where(want_press & reason_wrong, 1, warned)),
        'empty_age': empty_age,
        'prev_cnt': _i32(jnp.where(pot_vis, pot_cnt, mem['prev_cnt'])),
        'wrong_deposits': wrong_deposits,
    }
    trace = {
        'r': r, 'c': c, 'recipe': cur_recipe, 'press': _i32(want_press),
        'pot_cnt': _i32(pot_cnt), 'pot_ready': _i32(pot_ready), 'display': _i32(display_on),
        'reason': _i32(reason_initial) + 2 * _i32(reason_clear) + 4 * _i32(reason_recipe)
                  + 8 * _i32(reason_backoff) + 16 * _i32(reason_wrong),
    }
    return action, new_mem, trace


# --------------------------------------------------------------------------
# Agent 1: solo cook with recipe belief
# --------------------------------------------------------------------------

def init_agent_1():
    return {
        't': _i32(0),
        'p0': jnp.asarray(0.5, dtype=jnp.float32),  # P(recipe == ingredient 0)
        'wait': _i32(0),
        'seen_button': _i32(0),
        'delivered': _i32(0),
        'last_type': _i32(-1),  # type of the last delivered dish
        'n_same': _i32(0),      # post-delivery displays equal to last delivered type
        'n_change': _i32(0),    # post-delivery displays different from last delivered type
        'counted': _i32(0),     # a display has been counted in this delivery cycle
        'stash': jnp.zeros((4,), dtype=jnp.int32),  # 0 empty, 1 dish0, 2 dish1, 3 raw0, 4 raw1, 5 other
    }


def _slot_code(cell):
    item = (cell[CH_ITEM_PLATE] + cell[CH_ITEM_COOKED] + cell[CH_ITEM_I0]
            + cell[CH_ITEM_I1] + cell[CH_ITEM_I2]) > 0
    cooked = cell[CH_ITEM_COOKED] > 0
    d0 = (cell[CH_ITEM_I0] > 0) & (cell[CH_ITEM_I1] == 0) & (cell[CH_ITEM_I2] == 0)
    d1 = (cell[CH_ITEM_I1] > 0) & (cell[CH_ITEM_I0] == 0) & (cell[CH_ITEM_I2] == 0)
    raw = (~cooked) & (cell[CH_ITEM_PLATE] == 0)
    return _i32(jnp.where(~item, 0,
                          jnp.where(cooked & d0, 1,
                                    jnp.where(cooked & d1, 2,
                                              jnp.where(raw & d0, 3, jnp.where(raw & d1, 4, 5))))))


def agent_1_act(obs, reset, memory, rng):
    mem = _reset_tree(reset, init_agent_1(), memory)
    obs = jnp.asarray(obs, dtype=jnp.int32)
    r, c = _localize(obs, CH_POT, POT_R, POT_C)
    f = _facing(obs)

    # Inventory
    me = obs[2, 2, :]
    inv_plate = me[5] > 0
    inv_cooked = me[6] > 0
    inv_ing = me[7:10]
    inv_ing_sum = inv_ing[0] + inv_ing[1] + inv_ing[2]
    hold_dish = inv_cooked
    hold_plate = inv_plate & (~inv_cooked)
    hold_ing = (~inv_plate) & (~inv_cooked) & (inv_ing_sum > 0)
    hold_none = (~inv_plate) & (~inv_cooked) & (inv_ing_sum == 0)
    held_type = _i32(jnp.argmax(inv_ing))  # 0/1 (2 never appears in this room)

    # Pot
    pcell, _ = _cell(obs, r, c, POT_R, POT_C)
    pot_cnt = pcell[CH_ITEM_I0] + pcell[CH_ITEM_I1] + pcell[CH_ITEM_I2]
    pot_ready = pcell[CH_ITEM_COOKED] > 0
    pot_timer = pcell[CH_TIMER]
    pot_type = _i32(jnp.argmax(pcell[CH_ITEM_I0:CH_ITEM_I2 + 1]))
    pot_cooking = pot_timer > 0
    pot_partial = (pot_cnt > 0) & (~pot_cooking) & (~pot_ready)
    pot_empty = (pot_cnt == 0) & (~pot_ready)

    # Button display (visible only from column 5)
    bcell, btn_vis = _cell(obs, r, c, BTN_R, BTN_C)
    disp0 = btn_vis & (bcell[CH_REC_I0] > 0)
    disp1 = btn_vis & (bcell[CH_REC_I1] > 0)
    display = disp0 | disp1
    p0 = jnp.where(disp0, 1.0, jnp.where(disp1, 0.0, mem['p0'])).astype(jnp.float32)
    seen_button = _i32(jnp.maximum(mem['seen_button'], _i32(display)))
    # Partner signalling style, learned from the first display seen in each
    # post-delivery cycle: does it also show recipes equal to what I just delivered?
    last0 = mem['last_type']
    disp_type = _i32(jnp.where(disp0, 0, 1))
    new_event = display & (mem['delivered'] > 0) & (mem['counted'] == 0) & (last0 >= 0)
    n_same = _i32(jnp.minimum(mem['n_same'] + _i32(new_event & (disp_type == last0)), 100))
    n_change = _i32(jnp.minimum(mem['n_change'] + _i32(new_event & (disp_type != last0)), 100))
    counted = _i32(jnp.maximum(mem['counted'], _i32(new_event)))
    # Weight that a silent partner would have signalled a change.
    w = jnp.where(n_same + n_change == 0, jnp.where(seen_button > 0, SILENCE_WEIGHT_SEEN_ONLY, 0.0),
                  jnp.where(n_same == 0, SILENCE_WEIGHT_CHANGE_ONLY, SILENCE_WEIGHT_MIXED)).astype(jnp.float32)

    # Silence for POST_WAIT steps after a delivery: soft evidence that the recipe
    # did not change. Applied to the post-delivery belief q of the delivered type.
    silence_now = (mem['wait'] == POST_WAIT) & (mem['delivered'] > 0) & (last0 >= 0) & (w > 0) & (counted == 0) & (~display)
    q = jnp.where(last0 == 0, p0, 1.0 - p0)
    q_new = q / jnp.maximum(1.0 - q * w, 1e-3)
    p0 = jnp.where(silence_now, jnp.where(last0 == 0, q_new, 1.0 - q_new), p0).astype(jnp.float32)

    # Stash slots (refresh from observation where visible)
    SLOT_R, SLOT_C, SLOT_AR, SLOT_AC, SLOT_AF, SLOT_GROUP = _slot_arrays()

    def refresh(i, prev):
        cell, vis = _cell(obs, r, c, SLOT_R[i], SLOT_C[i])
        return _i32(jnp.where(vis, _slot_code(cell), prev))
    stash = jnp.stack([refresh(i, mem['stash'][i]) for i in range(4)])
    free = stash == 0
    has_free = jnp.any(free)

    has_stash0 = jnp.any(stash == 1)
    has_stash1 = jnp.any(stash == 2)
    # Preferred type
    # Tie-break with no information: assume the recipe did not change (a partner
    # that only signals changes is then always right; otherwise it is a coin flip).
    last = mem['last_type']
    tie_T = _i32(jnp.where(last >= 0, last, jnp.where(has_stash1 & (~has_stash0), 1, 0)))
    T = _i32(jnp.where(p0 > 0.5, 0, jnp.where(p0 < 0.5, 1, tie_T)))
    pT = jnp.where(T == 0, p0, 1.0 - p0)
    has_stash_T = jnp.where(T == 0, has_stash0, has_stash1)
    match_T = stash == (T + 1)
    slot_T = _i32(jnp.argmax(match_T))
    act_retrieve = _goto(r, c, f, SLOT_AR[slot_T], SLOT_AC[slot_T], SLOT_AF[slot_T], _i32(A_INTERACT))

    # Common actions
    act_pot_interact = _goto(r, c, f, _i32(2), _i32(5), _i32(F_LEFT), _i32(A_INTERACT))
    act_pot_wait = _goto(r, c, f, _i32(2), _i32(5), _i32(F_LEFT), _i32(A_STAY))
    act_deliver = _goto(r, c, f, _i32(2), _i32(6), _i32(F_RIGHT), _i32(A_INTERACT))
    plate_row = _i32(jnp.where(r <= 2, 1, 3))
    act_plate = _goto(r, c, f, plate_row, _i32(5), _i32(F_LEFT), _i32(A_INTERACT))

    def act_fetch(typ):
        row = _i32(jnp.where(typ == 0, 1, 3))
        from_supply = _goto(r, c, f, row, _i32(6), _i32(F_RIGHT), _i32(A_INTERACT))
        raw_match = stash == (typ + 3)
        has_raw = jnp.any(raw_match)
        rslot = _i32(jnp.argmax(raw_match))
        from_slot = _goto(r, c, f, SLOT_AR[rslot], SLOT_AC[rslot], SLOT_AF[rslot], _i32(A_INTERACT))
        return jnp.where(has_raw, from_slot, from_supply)

    # Drop slot for an item of "group" g (prefer matching group, else any free)
    def drop_slot(g):
        pref = free & (SLOT_GROUP == g)
        has_pref = jnp.any(pref)
        return _i32(jnp.where(has_pref, jnp.argmax(pref), jnp.argmax(free)))

    # 1. Holding a dish
    pA = jnp.where(held_type == 0, p0, 1.0 - p0)
    dslot = drop_slot(held_type)
    act_stash = _goto(r, c, f, SLOT_AR[dslot], SLOT_AC[dslot], SLOT_AF[dslot], _i32(A_INTERACT))
    deliver_now = (pA >= 0.5) | (~has_free)
    act_dish = jnp.where(deliver_now, act_deliver, act_stash)

    # 2. Holding a plate
    aslot = drop_slot(_i32(jnp.where(r <= 2, 0, 1)))
    act_drop_any = _goto(r, c, f, SLOT_AR[aslot], SLOT_AC[aslot], SLOT_AF[aslot], _i32(A_INTERACT))
    act_plate_held = jnp.where(pot_ready, act_pot_interact,
                               jnp.where(pot_cooking | (~has_free), act_pot_wait, act_drop_any))

    # 3. Holding an ingredient
    # Commit to an empty pot only if the held type is still believed correct;
    # otherwise park the ingredient on a counter and fetch the right one.
    commit_ok = (pA >= 0.5) | (~has_free)
    can_add = (pot_empty & commit_ok) | (pot_partial & (pot_type == held_type))
    act_ing_held = jnp.where(can_add, act_pot_interact, jnp.where(has_free, act_drop_any, act_pot_wait))

    # 4. Empty hands
    wait_limit = jnp.where(mem['delivered'] > 0, POST_WAIT, START_WAIT)
    # Wait for information at a tie, or (with a partner that has signalled after
    # deliveries before) until the silence evidence has been applied.
    waiting = ((p0 == 0.5) & (mem['wait'] < wait_limit)) | (
        (mem['delivered'] > 0) & (w > 0) & (counted == 0) & (mem['wait'] < POST_WAIT))
    act_empty_pot = jnp.where(waiting, act_pot_wait,
                              jnp.where(has_stash_T, act_retrieve, act_fetch(T)))
    act_cooking = jnp.where(has_stash_T & (pT >= 0.5) & (pot_timer >= 10), act_retrieve, act_plate)
    act_none = jnp.where(pot_ready, act_plate,
                         jnp.where(pot_cooking, act_cooking,
                                   jnp.where(pot_partial, act_fetch(pot_type), act_empty_pot)))

    action = _i32(jnp.where(hold_dish, act_dish,
                            jnp.where(hold_plate, act_plate_held,
                                      jnp.where(hold_ing, act_ing_held, act_none))))

    # Belief update on delivery (deterministic when in place holding a dish)
    delivering = hold_dish & (r == 2) & (c == 6) & (f == F_RIGHT) & (action == A_INTERACT)
    p0_after = jnp.where(held_type == 0, 0.5 * p0, 0.5 + 0.5 * p0)
    p0_new = jnp.where(delivering, p0_after, p0).astype(jnp.float32)
    wait_new = _i32(jnp.where(delivering, 0, jnp.minimum(mem['wait'] + 1, 1000)))

    new_mem = {
        't': mem['t'] + 1,
        'p0': p0_new,
        'wait': wait_new,
        'seen_button': seen_button,
        'delivered': _i32(jnp.maximum(mem['delivered'], _i32(delivering))),
        'last_type': _i32(jnp.where(delivering, held_type, mem['last_type'])),
        'n_same': n_same,
        'n_change': n_change,
        'counted': _i32(jnp.where(delivering, 0, counted)),
        'stash': stash,
    }
    mode = _i32(jnp.where(hold_dish, 1, jnp.where(hold_plate, 2, jnp.where(hold_ing, 3, 0))))
    trace = {
        'r': r, 'c': c, 'p0': p0, 'mode': mode, 'T': T, 'pot_cnt': _i32(pot_cnt),
        'pot_timer': _i32(pot_timer), 'display': _i32(display), 'deliver': _i32(delivering),
        'stash': stash, 'waiting': _i32(waiting),
    }
    return action, new_mem, trace
