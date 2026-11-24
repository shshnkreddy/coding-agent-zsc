"""Overcooked V2 grounded_coord_simple policy.

Agent 0 (left room) can see the always-on recipe indicator and the pot, and can
press the button recipe indicator.  It parks at (2,2) facing right and presses
the button once per delivery period (with bounded re-presses) so the cook can
read the recipe.

Agent 1 (right room) does the whole cook cycle.  It reads the recipe from the
button display whenever visible, keeps a belief with a staleness flag after
each delivery, stores mismatching dishes on counters instead of delivering
them, and cooks speculatively when no information is available.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'button_informer_solo_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing directions (observation encoding)
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3
DIR_TO_ACT_LIST = [A_UP, A_DOWN, A_RIGHT, A_LEFT]

# Channels
CH_SELF_DIR = 1
CH_SELF_INV = 5
CH_STATIC_GOAL = 21
CH_STATIC_RECIPE = 23
CH_ITEM = 29
CH_RECIPE = 34
CH_TIMER = 39

# Tunables
W_STALE = 6        # steps to wait for a fresh display after a delivery before assuming unchanged
W_HOLD = 6         # steps to hold an undecided dish before storing it
W_INIT = 10        # steps to wait for first information before speculative cooking
K_ALT = 3          # consecutive unconfirmed deliveries before alternating the assumed recipe
T_BLIND = 30       # with no information at all by this step, start blind alternating deliveries
PLATE_LEAD = 5     # fetch plate when pot timer <= this
PRESS_DELAY = 3    # agent 0: steps after pot emptied before pressing (unless recipe changed)
PRESS_CAP = 12     # agent 0: max presses per episode
REPRESS_GAP_1 = 16
REPRESS_GAP_2 = 30
LAST_PRESS_STEP = 356  # no period/re-presses after this: a fresh cook cycle (~41 steps) cannot finish


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _cell(obs, r, c, gr, gc):
    """Feature vector of global cell (gr, gc) as seen from (r, c); zeros if not visible."""
    i = gr - r + 2
    j = gc - c + 2
    vis = (i >= 0) & (i <= 4) & (j >= 0) & (j <= 4)
    feat = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), :]
    feat = jnp.where(vis, feat, jnp.zeros_like(feat))
    return vis, feat


def _locate(obs, channel, gr, gc):
    flat = jnp.argmax(obs[:, :, channel].reshape(-1))
    i = flat // 5
    j = flat % 5
    return _i32(gr - i + 2), _i32(gc - j + 2)


def _self_dir(obs):
    return jnp.argmax(obs[2, 2, CH_SELF_DIR:CH_SELF_DIR + 4]).astype(jnp.int32)


def _item_type(block):
    """block = [plate, cooked, n0, n1, n2]. -1 empty, 0/1 pure dish type, 2 other."""
    p, k, n0, n1, n2 = block[0], block[1], block[2], block[3], block[4]
    empty = (p == 0) & (k == 0) & (n0 + n1 + n2 == 0)
    dish0 = (k == 1) & (n0 >= 1) & (n1 == 0) & (n2 == 0)
    dish1 = (k == 1) & (n1 >= 1) & (n0 == 0) & (n2 == 0)
    return jnp.where(empty, -1, jnp.where(dish0, 0, jnp.where(dish1, 1, 2))).astype(jnp.int32)


def _nav(r, c, d, tr, tc, td, interact):
    at = (r == tr) & (c == tc)
    move = jnp.where(r < tr, A_DOWN, jnp.where(r > tr, A_UP, jnp.where(c < tc, A_RIGHT, A_LEFT)))
    turn = jnp.asarray(DIR_TO_ACT_LIST, dtype=jnp.int32)[td]
    act_at = jnp.where(d == td, jnp.where(interact, A_INTERACT, A_STAY), turn)
    return jnp.where(at, act_at, move).astype(jnp.int32)


def _reset_mem(reset, init, mem):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, mem)


# ---------------------------------------------------------------- agent 0 --

def init_agent_0():
    return {
        'step': _i32(0),
        'period_start': _i32(0),
        'period_pressed': _i32(0),
        'period_presses': _i32(0),
        'since_press': _i32(99),
        'touched': _i32(0),
        'total': _i32(0),
        'prev_nonempty': _i32(0),
        'prev_rec': _i32(-1),
        'cook_pressed': _i32(0),
        'pot': jnp.zeros((5,), dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    mem = _reset_mem(reset, init_agent_0(), memory)
    step = mem['step']
    r, c = _locate(obs, CH_STATIC_RECIPE, 2, 0)
    d = _self_dir(obs)

    # always-on recipe indicator at (2,0)
    _, rf = _cell(obs, r, c, 2, 0)
    rec_now = jnp.where(rf[CH_RECIPE + 3] > 0, 1, 0).astype(jnp.int32)

    # pot at (2,4)
    pvis, pf = _cell(obs, r, c, 2, 4)
    pot_obs = jnp.stack([pf[CH_ITEM + 2], pf[CH_ITEM + 3], pf[CH_ITEM + 4], pf[CH_ITEM + 1], pf[CH_TIMER]]).astype(jnp.int32)
    pot = jnp.where(pvis, pot_obs, mem['pot'])
    pot_cnt = pot[0] + pot[1] + pot[2]
    nonempty = pot_cnt > 0
    pot_cooked = pot[3] > 0
    pot_type = _item_type(jnp.stack([_i32(0), _i32(1), pot[0], pot[1], pot[2]]))
    pot_type = jnp.where(pot_cnt == 0, -1, pot_type)

    empty_event = (mem['prev_nonempty'] > 0) & (~nonempty)
    rec_changed = (mem['prev_rec'] >= 0) & (rec_now != mem['prev_rec'])

    period_start = jnp.where(empty_event, step, mem['period_start'])
    period_pressed = jnp.where(empty_event, 0, mem['period_pressed'])
    period_presses = jnp.where(empty_event, 0, mem['period_presses'])
    touched = jnp.where(empty_event, 0, jnp.maximum(mem['touched'], nonempty.astype(jnp.int32)))
    cook_pressed = jnp.where(pot_cooked, mem['cook_pressed'], 0)
    since_press = mem['since_press']

    delay = jnp.where(period_start == 0, 0, PRESS_DELAY)
    gap = jnp.where(period_presses == 1, REPRESS_GAP_1, jnp.where(period_presses == 2, REPRESS_GAP_2, 10 ** 6))
    mismatch = pot_cooked & (pot_type != rec_now)

    want_change = rec_changed & (since_press >= 2)
    want_first = (period_pressed == 0) & (step - period_start >= delay)
    want_re = (period_pressed == 1) & (touched == 0) & (~nonempty) & (since_press >= gap)
    want_cook = mismatch & (cook_pressed == 0)
    in_time = step <= LAST_PRESS_STEP
    want = (mem['total'] < PRESS_CAP) & (((want_change | want_first | want_re) & in_time) | want_cook)

    # inventory (agent 0 only ever holds ingredient 2 used as a counter marker)
    inv = obs[2, 2, CH_SELF_INV:CH_SELF_INV + 5]
    holding = inv.sum() > 0

    # counter markers at (1,3) [recipe 0] and (3,3) [recipe 1]
    _, tf = _cell(obs, r, c, 1, 3)
    _, bf = _cell(obs, r, c, 3, 3)
    top_item = tf[CH_ITEM:CH_ITEM + 5].sum() > 0
    bot_item = bf[CH_ITEM:CH_ITEM + 5].sum() > 0
    want_top = rec_now == 0
    desired_present = jnp.where(want_top, top_item, bot_item)
    wrong_present = jnp.where(want_top, bot_item, top_item)
    # goal rows for marker work: desired counter reached from (1,2)/(3,2) facing right
    m_row = jnp.where(want_top, 1, 3).astype(jnp.int32)
    m_supply_dir = jnp.where(want_top, D_UP, D_DOWN).astype(jnp.int32)
    w_row = jnp.where(want_top, 3, 1).astype(jnp.int32)

    # marker goal (row, col, dir, interact)
    # holding: place on desired counter if empty, else dump on spare counter (0,1) from (1,1) facing up
    g_hold = jnp.where(desired_present,
                       jnp.asarray([1, 1, D_UP, 1], dtype=jnp.int32),
                       jnp.stack([m_row, _i32(2), _i32(D_RIGHT), _i32(1)]))
    # empty hands: remove wrong marker, else fetch ingredient 2 near desired row, else park
    g_empty = jnp.where(wrong_present,
                        jnp.stack([w_row, _i32(2), _i32(D_RIGHT), _i32(1)]),
                        jnp.where(~desired_present,
                                  jnp.stack([m_row, _i32(2), m_supply_dir, _i32(1)]),
                                  jnp.asarray([2, 2, D_RIGHT, 0], dtype=jnp.int32)))
    g_press = jnp.asarray([2, 2, D_RIGHT, 1], dtype=jnp.int32)
    g_park = jnp.asarray([2, 2, D_RIGHT, 0], dtype=jnp.int32)
    # press has priority when hands are empty; when holding, first put the item down
    goal = jnp.where(holding, g_hold, jnp.where(want, g_press, g_empty))
    action = _nav(r, c, d, goal[0], goal[1], goal[2], goal[3])

    in_pos = (r == 2) & (c == 2) & (d == D_RIGHT)
    press = want & in_pos & (~holding) & (action == A_INTERACT)

    pi = press.astype(jnp.int32)
    new_mem = {
        'step': step + 1,
        'period_start': period_start,
        'period_pressed': jnp.maximum(period_pressed, pi),
        'period_presses': period_presses + pi,
        'since_press': jnp.where(press, 0, jnp.minimum(since_press + 1, 10 ** 6)),
        'touched': touched,
        'total': mem['total'] + pi,
        'prev_nonempty': nonempty.astype(jnp.int32),
        'prev_rec': rec_now,
        'cook_pressed': jnp.maximum(cook_pressed, (press & mismatch).astype(jnp.int32)),
        'pot': pot,
    }
    new_mem = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.int32), new_mem)
    trace = {
        'press': pi,
        'rec': rec_now,
        'pot_cnt': pot_cnt.astype(jnp.int32),
        'pot_type': pot_type,
        'cooked': pot_cooked.astype(jnp.int32),
        'w_ch': want_change.astype(jnp.int32),
        'w_first': want_first.astype(jnp.int32),
        'w_re': want_re.astype(jnp.int32),
        'w_cook': want_cook.astype(jnp.int32),
        'top': top_item.astype(jnp.int32),
        'bot': bot_item.astype(jnp.int32),
        'hold': holding.astype(jnp.int32),
        'total': new_mem['total'],
        'r': r, 'c': c,
    }
    return action, new_mem, trace


# ---------------------------------------------------------------- agent 1 --

# goal table: (row, col, dir, interact)
G_WAIT, G_POT, G_ING0, G_ING1, G_PLATE_T, G_PLATE_B, G_DELIVER, G_CNT0, G_CNT1, G_CNT2, G_CNT3 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
GOALS_LIST = [
    [2, 5, D_LEFT, 0],   # wait facing pot / button
    [2, 5, D_LEFT, 1],   # pot
    [1, 6, D_RIGHT, 1],  # ingredient 0 supply (1,7)
    [3, 6, D_RIGHT, 1],  # ingredient 1 supply (3,7)
    [1, 5, D_LEFT, 1],   # plate pile (1,4)
    [3, 5, D_LEFT, 1],   # plate pile (3,4)
    [2, 6, D_RIGHT, 1],  # delivery (2,7)
    [1, 5, D_UP, 1],     # counter (0,5)
    [1, 6, D_UP, 1],     # counter (0,6)
    [3, 5, D_DOWN, 1],   # counter (4,5)
    [3, 6, D_DOWN, 1],   # counter (4,6)
]
SLOT_CELLS_LIST = [[0, 5], [0, 6], [4, 5], [4, 6]]
SLOT_STAND_LIST = [[1, 5], [1, 6], [3, 5], [3, 6]]


def init_agent_1():
    return {
        'step': _i32(0),
        'rec': _i32(-1),
        'conf': _i32(0),
        'stale_t': _i32(0),
        'hold_t': _i32(0),
        'button_seen': _i32(0),
        'ucount': _i32(0),
        'pot': jnp.zeros((5,), dtype=jnp.int32),
        'slots': -jnp.ones((4,), dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    mem = _reset_mem(reset, init_agent_1(), memory)
    GOALS = jnp.asarray(GOALS_LIST, dtype=jnp.int32)
    SLOT_STAND = jnp.asarray(SLOT_STAND_LIST, dtype=jnp.int32)
    step = mem['step']
    r, c = _locate(obs, CH_STATIC_GOAL, 2, 7)
    d = _self_dir(obs)

    # inventory
    inv = obs[2, 2, CH_SELF_INV:CH_SELF_INV + 5]
    h_plate, h_cooked = inv[0], inv[1]
    h_n = inv[2:5]
    h_raw = h_n.sum() > 0
    held_kind = jnp.where(h_cooked > 0, 3, jnp.where(h_plate > 0, 2, jnp.where(h_raw, 1, 0))).astype(jnp.int32)
    raw_type = jnp.argmax(h_n).astype(jnp.int32)
    dish_type = _item_type(inv)

    # pot
    pvis, pf = _cell(obs, r, c, 2, 4)
    pot_obs = jnp.stack([pf[CH_ITEM + 2], pf[CH_ITEM + 3], pf[CH_ITEM + 4], pf[CH_ITEM + 1], pf[CH_TIMER]]).astype(jnp.int32)
    mp = mem['pot']
    m_timer = jnp.maximum(mp[4] - 1, 0)
    m_cooked = jnp.maximum(mp[3], ((mp[4] == 1)).astype(jnp.int32))
    pot_mem = jnp.stack([mp[0], mp[1], mp[2], m_cooked, m_timer]).astype(jnp.int32)
    pot = jnp.where(pvis, pot_obs, pot_mem)
    pot_cnt = pot[0] + pot[1] + pot[2]
    pot_cooked = pot[3] > 0
    pot_cooking = (pot[4] > 0) & (~pot_cooked)
    pot_idle = (~pot_cooked) & (~pot_cooking)
    pot_type = _item_type(jnp.stack([_i32(0), _i32(1), pot[0], pot[1], pot[2]]))  # treat as dish for typing
    pot_type = jnp.where(pot_cnt == 0, -1, pot_type)

    # button display at (2,3)
    bvis, bf = _cell(obs, r, c, 2, 3)
    disp_on = bvis & ((bf[CH_RECIPE + 2] + bf[CH_RECIPE + 3]) > 0)
    disp_rec = jnp.where(bf[CH_RECIPE + 3] > 0, 1, 0).astype(jnp.int32)
    rec = jnp.where(disp_on, disp_rec, mem['rec'])
    conf = jnp.where(disp_on, 2, mem['conf'])
    button_seen = jnp.maximum(mem['button_seen'], disp_on.astype(jnp.int32))
    ucount = jnp.where(disp_on, 0, mem['ucount'])

    # counter marker fallback (1,3)/(3,3): only if the button was never seen
    tvis, tf = _cell(obs, r, c, 1, 3)
    bovis, bof = _cell(obs, r, c, 3, 3)
    top_item = tvis & (tf[CH_ITEM:CH_ITEM + 5].sum() > 0)
    bot_item = bovis & (bof[CH_ITEM:CH_ITEM + 5].sum() > 0)
    marker = (button_seen == 0) & tvis & bovis & (top_item ^ bot_item)
    marker_rec = jnp.where(top_item, 0, 1).astype(jnp.int32)
    rec = jnp.where(marker, marker_rec, rec)
    stale_t = mem['stale_t']
    stale_t = jnp.where(marker & (conf == 0), W_STALE, stale_t)
    conf = jnp.where(marker & (conf == 0), 1, conf)
    blind = (conf == 0) & (step >= T_BLIND)
    rec = jnp.where(blind, 0, rec)
    stale_t = jnp.where(blind, W_STALE, stale_t)
    ucount = jnp.where(blind, K_ALT, ucount)
    conf = jnp.where(blind, 1, conf)

    # counter slots
    slots = mem['slots']
    for s in range(4):
        svis, sf = _cell(obs, r, c, SLOT_CELLS_LIST[s][0], SLOT_CELLS_LIST[s][1])
        st = _item_type(sf[CH_ITEM:CH_ITEM + 5])
        slots = slots.at[s].set(jnp.where(svis, st, slots[s]))

    known = (conf == 2) | ((conf == 1) & (stale_t >= W_STALE))
    B = rec
    dist = jnp.abs(SLOT_STAND[:, 0] - r) + jnp.abs(SLOT_STAND[:, 1] - c)
    free_mask = slots == -1
    any_free = free_mask.any()
    free_slot = jnp.argmin(jnp.where(free_mask, dist, 1000)).astype(jnp.int32)
    b_mask = slots == B
    any_b = b_mask.any() & (conf >= 1)
    b_slot = jnp.argmin(jnp.where(b_mask, dist, 1000)).astype(jnp.int32)
    plate_goal = jnp.where(r <= 2, G_PLATE_T, G_PLATE_B)
    ing_goal = lambda t: jnp.where(t == 1, G_ING1, G_ING0)
    cnt_goal = lambda s: G_CNT0 + s

    # --- held dish
    match = known & (dish_type == B)
    mismatch = known & (dish_type != B)
    store_goal = jnp.where(any_free, cnt_goal(free_slot), G_WAIT)
    g_dish = jnp.where(match, G_DELIVER,
             jnp.where(mismatch, store_goal,
             jnp.where(mem['hold_t'] >= W_HOLD, store_goal, G_WAIT)))
    # --- held plate
    g_plate = jnp.where(pot_cooked, G_POT, jnp.where(pot_cooking, G_WAIT, store_goal))
    # --- held raw
    ok_insert = pot_idle & (pot_cnt < 3) & ((pot_cnt == 0) | (pot_type == raw_type))
    g_raw = jnp.where(ok_insert, G_POT, store_goal)
    # --- empty hands
    first = jnp.where(conf >= 1, B, 0)
    second = 1 - first
    has_first = (slots == first).any()
    has_second = (slots == second).any()
    g_cook = jnp.where((conf == 0) & (step < W_INIT), G_WAIT,
             jnp.where((conf == 1) & (stale_t < W_STALE), G_WAIT,
             jnp.where(~has_first, ing_goal(first),
             jnp.where(~has_second, ing_goal(second), G_WAIT))))
    fill_type = jnp.where(pot_type == 1, 1, 0)
    g_empty = jnp.where(known & any_b, cnt_goal(b_slot),
              jnp.where(pot_cooked | (pot_cooking & (pot[4] <= PLATE_LEAD)), plate_goal,
              jnp.where(pot_cooking, G_WAIT,
              jnp.where(pot_cnt > 0, ing_goal(fill_type), g_cook))))

    goal = jnp.where(held_kind == 3, g_dish,
           jnp.where(held_kind == 2, g_plate,
           jnp.where(held_kind == 1, g_raw, g_empty))).astype(jnp.int32)
    grow = GOALS[goal]
    action = _nav(r, c, d, grow[0], grow[1], grow[2], grow[3])

    delivering = (goal == G_DELIVER) & (action == A_INTERACT) & (held_kind == 3)
    ucount_next = jnp.where(delivering, jnp.where(conf == 2, 0, ucount + 1), ucount)
    flip = delivering & (ucount_next >= K_ALT)
    rec_next = jnp.where(flip, 1 - rec, rec)
    conf_next = jnp.where(delivering, 1, conf)
    stale_next = jnp.where(delivering, 0, jnp.minimum(stale_t + 1, 10 ** 6))
    hold_next = jnp.where(held_kind == 3, jnp.minimum(mem['hold_t'] + 1, 10 ** 6), 0)

    new_mem = {
        'step': step + 1,
        'rec': rec_next,
        'conf': conf_next,
        'stale_t': stale_next,
        'hold_t': hold_next,
        'button_seen': button_seen,
        'ucount': ucount_next,
        'pot': pot,
        'slots': slots,
    }
    new_mem = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.int32), new_mem)
    trace = {
        'goal': goal,
        'rec': rec,
        'conf': conf,
        'held': held_kind,
        'ucount': ucount_next,
        'pot_cnt': pot_cnt.astype(jnp.int32),
        'r': r, 'c': c,
    }
    return action, new_mem, trace
