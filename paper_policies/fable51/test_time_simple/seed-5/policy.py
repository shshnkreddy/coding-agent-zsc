"""Counter-signal cooperative policy for the test_time_simple layout.

Geometry facts (from context/environment):
  * agent_0 lives in the left room (rows 1-3, cols 1-2). It can see the
    always-on recipe indicator at (2,0), reach ingredient-2 supplies at (0,2)
    and (4,2), and place items on counters (1,3), (2,3), (3,3).  It can never
    reach the pot, plates, cooking ingredients, or the delivery station.
  * agent_1 lives in the right room (rows 1-3, cols 5-6).  It can reach the
    plate piles (1,4)/(3,4), the pot (2,4), ingredient 0 (1,7), ingredient 1
    (3,7), the delivery station (2,7), and private counters (0,5),(0,6),(4,5),
    (4,6).  It can never see the recipe indicator or its partner.
  * The counters at column 3 are visible to agent_1 whenever it stands in
    column 5.  That is the only channel from agent_0 to agent_1.

Protocol (agent_0 -> agent_1): one ingredient-2 unit on counter (1,3) means
"recipe [0,0,0]" (row of ingredient-0 supply); one unit on counter (3,3) means
"recipe [1,1,1]" (row of ingredient-1 supply).  Agent_1 additionally accepts an
item on (2,3) as "recipe 1" (index-style conventions) and uses the most
recently placed item when several are present.  Agent_1 validates the
convention through recipe changes observed after its own deliveries, flips the
interpretation after repeated unchanged signals, and alternates dish types when
no signal is ever offered.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'counter_signal_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Actions
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids: UP=0, DOWN=1, RIGHT=2, LEFT=3
FACE_ACT_LIST = (UP, DOWN, RIGHT, LEFT)

# Channels
CH_SELF_DIR = 1
CH_SELF_INV = 5
CH_STATIC_POT = 22
CH_STATIC_RECIPE = 23
CH_ITEM = 29
CH_RECIPE = 34
CH_TIMER = 39

# Signal cells (column 3 counters) and their decoded recipe type.
SIG_MAP_LIST = (0, 1, 1, 0)   # TOP, MID, BOT, ABSENT

# agent_1 private counters, stand cells, and facing direction.
PRIV_R_LIST = (0, 0, 4, 4)
PRIV_C_LIST = (5, 6, 5, 6)
PRIV_STAND_R_LIST = (1, 1, 3, 3)
PRIV_FACE_LIST = (0, 0, 1, 1)

# Decoder tuning
FLIP_K_INITIAL = 6     # unchanged signals after deliveries before first flip
FLIP_K_AFTER = 1       # threshold after a flip that has not been validated
NO_SIGNAL_WAIT = 20    # steps to wait for a signal before guessing
VALIDATE_DELAY = 6     # steps after a delivery before checking the signal
ABSENT_STEPS = 12      # visible steps without any signal before ABSENT class applies
LATENCY_CAP = 6        # max learned partner delay used for the post-delivery quarantine (free within the normal 7-step fetch)

# agent_0 adaptation tuning
WRONG_FILL_K = 2       # consecutive wrong pot fills before inverting encoding
IDLE_TOGGLE_STEPS = 16 # pot idle steps under a recipe-1 signal before trying (2,3)


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _anchor(obs, ch):
    flat = jnp.argmax(obs[:, :, ch].reshape(-1))
    return _i32(flat // 5), _i32(flat % 5)


def _cell(obs, self_r, self_c, r, c, ch):
    """Value of channel ch at global (r,c); 0 when outside the local view."""
    i = r - self_r + 2
    j = c - self_c + 2
    valid = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    ic = jnp.clip(i, 0, 4)
    jc = jnp.clip(j, 0, 4)
    return jnp.where(valid, obs[ic, jc, ch], 0)


def _cell_has_item(obs, self_r, self_c, r, c):
    total = _i32(0)
    for k in range(5):
        total = total + _cell(obs, self_r, self_c, r, c, CH_ITEM + k)
    return total > 0


def _item_type(obs, self_r, self_c, r, c):
    """-1 empty, 0/1 raw ingredient 0/1, 2 anything else (plate, dish, ingredient 2)."""
    plate = _cell(obs, self_r, self_c, r, c, CH_ITEM) > 0
    cooked = _cell(obs, self_r, self_c, r, c, CH_ITEM + 1) > 0
    c0 = _cell(obs, self_r, self_c, r, c, CH_ITEM + 2)
    c1 = _cell(obs, self_r, self_c, r, c, CH_ITEM + 3)
    c2 = _cell(obs, self_r, self_c, r, c, CH_ITEM + 4)
    empty = ~plate & ~cooked & (c0 + c1 + c2 == 0)
    raw0 = ~plate & ~cooked & (c0 > 0) & (c1 == 0) & (c2 == 0)
    raw1 = ~plate & ~cooked & (c1 > 0) & (c0 == 0) & (c2 == 0)
    return _i32(jnp.where(empty, -1, jnp.where(raw0, 0, jnp.where(raw1, 1, 2))))


def _in_view(self_r, self_c, r, c):
    return (jnp.abs(r - self_r) <= 2) & (jnp.abs(c - self_c) <= 2)


def _facing(obs):
    return _i32(jnp.argmax(obs[2, 2, CH_SELF_DIR:CH_SELF_DIR + 4]))


def _nav(self_r, self_c, facing, sr, sc, fd, final_action):
    """Move to stand cell (sr,sc) (vertical first), face fd, then act."""
    dr = sr - self_r
    dc = sc - self_c
    move_v = jnp.where(dr > 0, DOWN, UP)
    move_h = jnp.where(dc > 0, RIGHT, LEFT)
    move = jnp.where(dr != 0, move_v, move_h)
    at = (dr == 0) & (dc == 0)
    face_act = jnp.asarray(FACE_ACT_LIST, dtype=jnp.int32)[fd]
    return _i32(jnp.where(at, jnp.where(facing == fd, final_action, face_act), move))


# ---------------------------------------------------------------------------
# agent_0: recipe signaller
# ---------------------------------------------------------------------------

def init_agent_0():
    return {
        't': _i32(0),
        'invert': _i32(0),          # 1 -> swapped encoding
        'wrong_fills': _i32(0),     # consecutive wrong first-ingredient fills
        'prev_pot_n': _i32(0),
        'sig_since': _i32(-1),      # step at which current target was first seen present
        'sig_recipe': _i32(-1),     # recipe the placed signal corresponds to
        'enc1': _i32(0),            # recipe-1 signal cell: 0 -> (3,3), 1 -> (2,3)
        'idle_count': _i32(0),      # steps the pot stayed empty while a signal was up
    }


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    t = mem['t']

    ai, aj = _anchor(obs, CH_STATIC_RECIPE)
    self_r = 4 - ai
    self_c = 2 - aj
    facing = _facing(obs)

    r0 = _cell(obs, self_r, self_c, 2, 0, CH_RECIPE + 2) > 0
    r1 = _cell(obs, self_r, self_c, 2, 0, CH_RECIPE + 3) > 0
    recipe = _i32(jnp.where(r1 & ~r0, 1, 0))

    holding = jnp.sum(obs[2, 2, CH_SELF_INV:CH_SELF_INV + 5]) > 0

    top = _cell_has_item(obs, self_r, self_c, 1, 3)
    mid = _cell_has_item(obs, self_r, self_c, 2, 3)
    bot = _cell_has_item(obs, self_r, self_c, 3, 3)

    # Encoded recipe -> target row: type 0 -> row 1; type 1 -> row 3 (or row 2
    # after the fallback toggle).  `invert` swaps the type mapping.
    enc = recipe ^ mem['invert']
    tr = jnp.where(enc == 0, 1, jnp.where(mem['enc1'] == 0, 3, 2))
    present = jnp.stack([top, mid, bot])
    rows = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    present_target = present[tr - 1]
    wrong_mask = present & (rows != tr)
    wrong_any = jnp.any(wrong_mask)
    wrong_row = rows[jnp.argmax(wrong_mask)]

    # Pot watching (adaptation): the pot at (2,4) is visible from column 2 only.
    pot_visible = _in_view(self_r, self_c, 2, 4)
    pot_c0 = _cell(obs, self_r, self_c, 2, 4, CH_ITEM + 2)
    pot_c1 = _cell(obs, self_r, self_c, 2, 4, CH_ITEM + 3)
    pot_c2 = _cell(obs, self_r, self_c, 2, 4, CH_ITEM + 4)
    pot_n = pot_c0 + pot_c1 + pot_c2
    pot_type = _i32(jnp.where(pot_c1 > pot_c0, 1, 0))
    new_fill = pot_visible & (pot_n > 0) & (mem['prev_pot_n'] == 0)
    signal_age = jnp.where(mem['sig_since'] >= 0, t - mem['sig_since'], -1)
    signal_fresh = (signal_age >= 8) & (mem['sig_recipe'] == recipe)
    wrong_fill = new_fill & signal_fresh & (pot_type != recipe)
    right_fill = new_fill & signal_fresh & (pot_type == recipe)
    wrong_fills = jnp.where(wrong_fill, mem['wrong_fills'] + 1,
                            jnp.where(right_fill, 0, mem['wrong_fills']))
    do_invert = wrong_fills >= WRONG_FILL_K
    invert = jnp.where(do_invert, 1 - mem['invert'], mem['invert'])
    wrong_fills = jnp.where(do_invert, 0, wrong_fills)

    # Recipe-1 encoding fallback: if the signal has been up for a long time and
    # the pot never receives anything, the partner may not understand this cell.
    pot_timer = _cell(obs, self_r, self_c, 2, 4, CH_TIMER)
    pot_cooked = _cell(obs, self_r, self_c, 2, 4, CH_ITEM + 1) > 0
    pot_idle_empty = pot_visible & (pot_n == 0) & (pot_timer == 0) & ~pot_cooked
    clean = present_target & ~wrong_any
    idle_count = jnp.where(clean & pot_idle_empty & (enc == 1), mem['idle_count'] + 1, 0)
    toggle = idle_count >= IDLE_TOGGLE_STEPS
    enc1 = jnp.where(toggle, 1 - mem['enc1'], mem['enc1'])
    idle_count = jnp.where(toggle, 0, idle_count)

    # Signal bookkeeping for freshness.
    sig_since = jnp.where(clean, jnp.where(mem['sig_since'] >= 0, mem['sig_since'], t), -1)
    sig_since = jnp.where(clean & (mem['sig_recipe'] != recipe), t, sig_since)
    sig_recipe = jnp.where(clean, recipe, -1)

    # Actions
    place = _nav(self_r, self_c, facing, tr, 2, 2, INTERACT)
    # drop a spare item on a private counter: (0,1) via (1,1) facing UP, else (4,1)
    top_priv = _cell_has_item(obs, self_r, self_c, 0, 1)
    drop = jnp.where(top_priv,
                     _nav(self_r, self_c, facing, 3, 1, 1, INTERACT),
                     _nav(self_r, self_c, facing, 1, 1, 0, INTERACT))
    pick_wrong = _nav(self_r, self_c, facing, wrong_row, 2, 2, INTERACT)
    fetch = jnp.where(tr == 1,
                      _nav(self_r, self_c, facing, 1, 2, 0, INTERACT),
                      _nav(self_r, self_c, facing, 3, 2, 1, INTERACT))
    idle = _nav(self_r, self_c, facing, tr, 2, 2, STAY)

    action = jnp.where(
        holding,
        jnp.where(present_target, drop, place),
        jnp.where(wrong_any, pick_wrong, jnp.where(present_target, idle, fetch)),
    )

    next_mem = {
        't': t + 1,
        'invert': _i32(invert),
        'wrong_fills': _i32(wrong_fills),
        'prev_pot_n': _i32(jnp.where(pot_visible, pot_n, mem['prev_pot_n'])),
        'sig_since': _i32(sig_since),
        'sig_recipe': _i32(sig_recipe),
        'enc1': _i32(enc1),
        'idle_count': _i32(idle_count),
    }
    trace = {
        'recipe': recipe,
        'invert': _i32(invert),
        'enc1': _i32(enc1),
        'self_r': _i32(self_r),
        'self_c': _i32(self_c),
    }
    return _i32(action), next_mem, trace


# ---------------------------------------------------------------------------
# agent_1: cook and deliver, decoding the partner's counter signal
# ---------------------------------------------------------------------------

def init_agent_1():
    return {
        't': _i32(0),
        'app': jnp.full((3,), -1, dtype=jnp.int32),   # appearance step per signal cell
        'prev_present': jnp.zeros((3,), dtype=jnp.int32),
        'reading': _i32(-1),        # last visible signal class (-1 none, 0 TOP, 1 MID, 2 BOT)
        'absent_count': _i32(0),
        'inv': jnp.zeros((4,), dtype=jnp.int32),   # per-class interpretation flips
        'flip_cls': _i32(-1),
        'flip_k': _i32(FLIP_K_INITIAL),
        'same_count': _i32(0),
        'last_type': _i32(-1),      # type of last delivered dish
        'del_time': _i32(-1),
        'del_sig': _i32(-1),
        'pending': _i32(0),
        'checked_sig': _i32(-1),    # reading used by the last post-delivery check
        'flipped_on_check': _i32(0),
        'latency': _i32(0),         # learned partner signal-update delay after a delivery
        'signal_ever': _i32(0),     # any signal ever decoded
        'guessed_before': _i32(0),  # a no-signal guess was already made
        'wait_count': _i32(0),
        'guess': _i32(-1),
        'occ': jnp.full((4,), -1, dtype=jnp.int32),   # item type on private counters (-1 empty, 0/1 raw, 2 other)
        'ndeliv': _i32(0),
        'ndump': _i32(0),
    }


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    t = mem['t']

    pi, pj = _anchor(obs, CH_STATIC_POT)
    self_r = 4 - pi
    self_c = 6 - pj
    facing = _facing(obs)

    # Inventory
    inv = obs[2, 2, CH_SELF_INV:CH_SELF_INV + 5]
    inv_plate = inv[0] > 0
    inv_cooked = inv[1] > 0
    inv_counts = inv[2:5]
    holding_raw = (jnp.sum(inv_counts) > 0) & ~inv_plate & ~inv_cooked
    holding_plate = inv_plate & ~inv_cooked
    holding_dish = inv_cooked
    held_type = _i32(jnp.where(inv_counts[1] > inv_counts[0], 1, 0))
    empty = ~inv_plate & ~inv_cooked & (jnp.sum(inv_counts) == 0)

    # Pot
    pot_cooked = obs[pi, pj, CH_ITEM + 1] > 0
    pot_counts = obs[pi, pj, CH_ITEM + 2:CH_ITEM + 5]
    pot_n = jnp.sum(pot_counts)
    pot_timer = obs[pi, pj, CH_TIMER]
    pot_cooking = pot_timer > 0
    pot_ready = pot_cooked & ~pot_cooking
    pot_idle = ~pot_cooking & ~pot_cooked
    pot_type = _i32(jnp.where(pot_counts[1] > pot_counts[0], 1, 0))

    # Signal decoding (only when column 3 is visible, i.e. self_c == 5)
    visible = self_c <= 5
    present = jnp.stack([
        _cell_has_item(obs, self_r, self_c, 1, 3),
        _cell_has_item(obs, self_r, self_c, 2, 3),
        _cell_has_item(obs, self_r, self_c, 3, 3),
    ]).astype(jnp.int32)
    appeared = (present > 0) & (mem['prev_present'] == 0)
    app_new = jnp.where(appeared, t, mem['app'])
    app_new = jnp.where(present > 0, app_new, -1)
    app = jnp.where(visible, app_new, mem['app'])
    prev_present = jnp.where(visible, present, mem['prev_present'])
    any_present = jnp.any(present > 0)
    newest = jnp.argmax(jnp.where(present > 0, app_new, -1))
    cls_now = _i32(jnp.where(any_present, newest, -1))          # 0 TOP, 1 MID, 2 BOT
    absent_count = jnp.where(visible, jnp.where(any_present, 0, mem['absent_count'] + 1), mem['absent_count'])
    signal_ever = jnp.where(visible & any_present, 1, mem['signal_ever'])
    cls_raw = _i32(jnp.where(visible, cls_now, mem['reading']))
    # Class 3 = ABSENT: a signal was seen before but nothing has been displayed
    # for a while (presence-style conventions encode one recipe by absence).
    reading = _i32(jnp.where(cls_raw >= 0, cls_raw,
                             jnp.where((signal_ever > 0) & (absent_count >= ABSENT_STEPS), 3, -1)))
    base_map = jnp.asarray(SIG_MAP_LIST, dtype=jnp.int32)
    idx4 = jnp.arange(4, dtype=jnp.int32)

    def _flip_mask(c):
        # flip the class used for the delivery; TOP and BOT are flipped together
        return (idx4 == c) | ((c == 0) & (idx4 == 2)) | ((c == 2) & (idx4 == 0))

    # Delivery validation: did the signal change after my last delivery?
    inv = mem['inv']
    flip_k = mem['flip_k']
    same_count = mem['same_count']
    pending = mem['pending']
    since_del = t - mem['del_time']
    check_delay = jnp.maximum(VALIDATE_DELAY, mem['latency'] + 1)
    can_check = (pending > 0) & visible & (since_del >= check_delay) & (reading >= 0) & (mem['del_sig'] >= 0)
    changed = reading != mem['del_sig']
    validated = can_check & changed
    unchanged = can_check & ~changed
    same_count = jnp.where(validated, 0, jnp.where(unchanged, same_count + 1, same_count))
    flip_k = jnp.where(validated, FLIP_K_INITIAL, flip_k)
    do_flip = unchanged & (same_count >= flip_k)
    inv = jnp.where(do_flip & _flip_mask(mem['del_sig']), 1 - inv, inv)
    same_count = jnp.where(do_flip, 0, same_count)
    flip_k = jnp.where(do_flip, FLIP_K_AFTER, flip_k)
    pending = jnp.where(can_check | (since_del >= 60) | (mem['del_sig'] < 0), 0, pending)
    checked_sig = jnp.where(can_check, reading, mem['checked_sig'])
    flipped_on_check = jnp.where(can_check, jnp.where(do_flip, 1, 0), mem['flipped_on_check'])
    flip_cls = jnp.where(do_flip, mem['del_sig'], mem['flip_cls'])
    # Late change: the partner updated its signal after our check.  The recipe
    # only changes on our own correct deliveries, so this validates the
    # interpretation used for that delivery; undo a flip made on the stale check.
    late_change = (pending == 0) & visible & (reading >= 0) & (checked_sig >= 0) & (reading != checked_sig)
    undo = late_change & (flipped_on_check > 0)
    inv = jnp.where(undo & _flip_mask(flip_cls), 1 - inv, inv)
    same_count = jnp.where(late_change, 0, same_count)
    flip_k = jnp.where(late_change, FLIP_K_INITIAL, flip_k)
    checked_sig = jnp.where(late_change, -1, checked_sig)
    flipped_on_check = jnp.where(late_change, 0, flipped_on_check)
    latency = jnp.where(late_change & (mem['del_time'] >= 0),
                        jnp.maximum(mem['latency'], jnp.minimum(since_del, LATENCY_CAP)), mem['latency'])

    rd = jnp.clip(reading, 0, 3)
    eff = _i32(jnp.where(reading >= 0, base_map[rd] ^ inv[rd], -1))

    # Private counter occupancy tracking
    occ_vis = jnp.stack([_in_view(self_r, self_c, PRIV_R_LIST[k], PRIV_C_LIST[k]) for k in range(4)])
    occ_now = jnp.stack([_item_type(obs, self_r, self_c, PRIV_R_LIST[k], PRIV_C_LIST[k]) for k in range(4)]).astype(jnp.int32)
    priv_stand_r = jnp.asarray(PRIV_STAND_R_LIST, dtype=jnp.int32)
    priv_c = jnp.asarray(PRIV_C_LIST, dtype=jnp.int32)
    priv_face = jnp.asarray(PRIV_FACE_LIST, dtype=jnp.int32)
    occ = jnp.where(occ_vis, occ_now, mem['occ'])
    free_idx = _i32(jnp.argmax(occ < 0))
    all_full = jnp.all(occ >= 0)
    drop_action = _nav(self_r, self_c, facing, priv_stand_r[free_idx], priv_c[free_idx],
                       priv_face[free_idx], INTERACT)

    # Fallback guess when no signal for a long time
    waiting = empty & (pot_n == 0) & (eff < 0)
    wait_count = jnp.where(waiting, mem['wait_count'] + 1, 0)
    guess = mem['guess']
    guess = jnp.where((pot_n > 0) | (eff >= 0), -1, guess)
    fallback_type = _i32(jnp.where(mem['last_type'] >= 0, 1 - mem['last_type'], 0))
    wait_limit = jnp.where((signal_ever > 0) | (mem['guessed_before'] == 0), NO_SIGNAL_WAIT, 1)
    make_guess = waiting & (wait_count >= wait_limit) & (guess < 0)
    guess = jnp.where(make_guess, fallback_type, guess)
    guessed_before = jnp.where(make_guess, 1, mem['guessed_before'])

    want = _i32(jnp.where(pot_n > 0, pot_type, jnp.where(eff >= 0, eff, guess)))
    pot_doomed = (pot_n > 0) & (eff >= 0) & (pot_type != eff)

    # --- action selection --------------------------------------------------
    deliver = _nav(self_r, self_c, facing, 2, 6, 2, INTERACT)
    dish_wrong = holding_dish & (eff >= 0) & (eff != held_type)
    act_dish = jnp.where(dish_wrong & ~all_full, drop_action, deliver)

    scoop = _nav(self_r, self_c, facing, 2, 5, 3, INTERACT)
    wait_pot = _nav(self_r, self_c, facing, 2, 5, 3, STAY)
    act_plate = jnp.where(pot_ready, scoop, jnp.where(pot_cooking, wait_pot, drop_action))

    insert = _nav(self_r, self_c, facing, 2, 5, 3, INTERACT)
    can_insert = pot_idle & (pot_n < 3) & ((pot_n == 0) | (pot_type == held_type) | pot_doomed)
    raw_ok = (pot_n > 0) | (eff < 0) | (eff == held_type)
    # Quarantine: after a delivery, hold the first ingredient at the pot until
    # the partner has had time (learned latency) to update its signal.
    quarantine = (pot_n == 0) & (mem['del_time'] >= 0) & (since_del <= mem['latency'])
    act_raw = jnp.where(quarantine & raw_ok, wait_pot,
                        jnp.where(can_insert & (raw_ok | all_full), insert, drop_action))

    plate_row = jnp.where(self_r <= 2, 1, 3)
    get_plate = _nav(self_r, self_c, facing, plate_row, 5, 3, INTERACT)
    supply_row = jnp.where(want == 0, 1, 3)
    get_ing_supply = _nav(self_r, self_c, facing, supply_row, 6, 2, INTERACT)
    # Recycle a matching raw ingredient parked on a private counter if any.
    stash_match = (occ == want) & (want >= 0)
    stash_idx = _i32(jnp.argmax(stash_match))
    get_ing_stash = _nav(self_r, self_c, facing, priv_stand_r[stash_idx], priv_c[stash_idx],
                         priv_face[stash_idx], INTERACT)
    get_ing = jnp.where(jnp.any(stash_match), get_ing_stash, get_ing_supply)
    act_empty = jnp.where(pot_cooking | pot_ready, get_plate,
                          jnp.where(want >= 0, get_ing, wait_pot))

    action = jnp.where(holding_dish, act_dish,
                       jnp.where(holding_plate, act_plate,
                                 jnp.where(holding_raw, act_raw, act_empty)))

    # Delivery bookkeeping (action is a delivery interact right now)
    delivering = holding_dish & ~dish_wrong & (self_r == 2) & (self_c == 6) & (facing == 2) & (action == INTERACT)
    dumping = holding_dish & dish_wrong & (action == INTERACT)
    del_time = jnp.where(delivering, t, mem['del_time'])
    del_sig = jnp.where(delivering, reading, mem['del_sig'])
    # After delivering, the recipe may have changed: forget the stale reading so
    # the next fetch decision waits for a fresh look at the signal counters.
    cls_raw = _i32(jnp.where(delivering, -1, cls_raw))
    last_type = jnp.where(delivering, held_type, mem['last_type'])
    pending = jnp.where(delivering, 1, pending)
    ndeliv = mem['ndeliv'] + jnp.where(delivering, 1, 0)
    ndump = mem['ndump'] + jnp.where(dumping, 1, 0)

    next_mem = {
        't': t + 1,
        'app': app.astype(jnp.int32),
        'prev_present': prev_present.astype(jnp.int32),
        'reading': _i32(cls_raw),
        'absent_count': _i32(absent_count),
        'inv': inv.astype(jnp.int32),
        'flip_cls': _i32(flip_cls),
        'flip_k': _i32(flip_k),
        'same_count': _i32(same_count),
        'last_type': _i32(last_type),
        'del_time': _i32(del_time),
        'del_sig': _i32(del_sig),
        'pending': _i32(pending),
        'checked_sig': _i32(checked_sig),
        'flipped_on_check': _i32(flipped_on_check),
        'latency': _i32(latency),
        'signal_ever': _i32(signal_ever),
        'guessed_before': _i32(guessed_before),
        'wait_count': _i32(wait_count),
        'guess': _i32(guess),
        'occ': occ.astype(jnp.int32),
        'ndeliv': _i32(ndeliv),
        'ndump': _i32(ndump),
    }
    trace = {
        'self_r': _i32(self_r),
        'self_c': _i32(self_c),
        'reading': _i32(reading),
        'eff': _i32(eff),
        'want': _i32(want),
        'invert': _i32(jnp.sum(inv)),
        'same_count': _i32(same_count),
        'ndeliv': _i32(ndeliv),
        'ndump': _i32(ndump),
        'pot_n': _i32(pot_n),
        'latency': _i32(latency),
    }
    return _i32(action), next_mem, trace
