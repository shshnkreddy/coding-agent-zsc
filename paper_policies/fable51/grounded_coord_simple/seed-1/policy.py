"""Overcooked V2 grounded_coord_simple policy.

Layout facts (from the environment spec):
  * agent_0 lives in the left room (rows 1-3, cols 1-2). It can always see the
    always-on recipe indicator R at (2,0), can see the pot at (2,4) from column 2,
    and can press the recipe button L at (2,3) from (2,2) facing right.
    It cannot reach the pot, plates, useful ingredients or the delivery station.
  * agent_1 lives in the right room (rows 1-3, cols 5-6). It can reach the pot
    (from (2,5) facing left), plate piles (from (1,5)/(3,5) facing left),
    ingredient 0 (from (1,6) facing right), ingredient 1 (from (3,6) facing
    right), the delivery station (from (2,6) facing right) and four counters
    at (0,5),(0,6),(4,5),(4,6). It can see the button display only from column 5.
  * The agents never see each other.

Protocol (grounded, no private convention): agent_0 presses the button when the
partner plausibly needs the recipe (episode start, after a dish leaves the pot,
after the displayed recipe changes, or when the pot is being filled with a
mismatching ingredient) with a bounded re-press schedule. agent_1 cooks alone,
reads the button whenever it is displayed, stores mismatching dishes on counters
for later reuse, and falls back to speculative cooking / gamble delivery if no
information arrives.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'button_signaller_solo_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids in the observation
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
_FACE_TO_ACTION_LIST = [A_UP, A_DOWN, A_RIGHT, A_LEFT]

# Channels
CH_SELF_FACE = 1      # 1..4
CH_SELF_INV = 5       # 5..9 : plate, cooked, ing0, ing1, ing2
CH_GOAL = 21
CH_RECIPE_IND = 23
CH_ITEM = 29          # 29..33 : plate, cooked, ing0, ing1, ing2
CH_RECIPE = 34        # 34..38 : plate, cooked, ing0, ing1, ing2
CH_TIMER = 39

# agent_1 tuning
W1_DEFAULT = 20       # commit threshold before any partner signal has been observed
W1_DEAD = 8           # commit threshold once a blind delivery happened with no signal ever seen
W1_MIN, W1_MAX = 8, 32
W1_MARGIN = 8         # commit threshold = max observed post-delivery signal delay + margin
W0_MIN, W0_MAX = 3, 6  # fetch speculative ingredient this many unknown steps before commit window
G_GAMBLE = 30         # unknown steps before delivering an unverified dish
BLIND_STREAK_FLIP_RESIGNALLER = 4  # blind streak before flipping guess, partner has re-signalled before
BLIND_STREAK_FLIP_SILENT = 2       # blind streak before flipping guess, partner never re-signalled
BLIND_STREAK_SHORT_WAIT = 2        # blind streak after which a never-re-signalling partner gets short waits

# agent_0 tuning
REPRESS_STEPS = 14    # re-press if pot still empty this long after a press
CYCLE_MAX_PRESSES = 3
DORMANT_STEPS = 40
DORMANT_STEPS_NEVER_ACTIVE = 100
EMPTIED_DELAY = 1
LAST_USEFUL_PRESS_STEP = 359   # a press later than this cannot lead to a fresh cooked delivery
IGNORE_LIMIT = 2      # first-ingredient mismatches while display visible before going quiet
SILENT_STEPS = 60     # quiet-mode trickle press interval
READY_STALL_STEPS = 10  # cooked pot untouched this long -> partner may be waiting for the recipe

I32 = jnp.int32


def _i(x):
    return jnp.asarray(x, dtype=I32)


def _self_pose(obs, anchor_channel, ar, ac):
    """Global (row, col) of self and facing id, using a static anchor cell."""
    flat = obs[:, :, anchor_channel].reshape(-1)
    idx = jnp.argmax(flat)
    lr = idx // 5
    lc = idx % 5
    r = _i(ar) - (lr - 2)
    c = _i(ac) - (lc - 2)
    face = jnp.argmax(obs[2, 2, CH_SELF_FACE:CH_SELF_FACE + 4]).astype(I32)
    return r.astype(I32), c.astype(I32), face


def _cell(obs, r, c, gr, gc):
    """Observation vector at global cell (gr, gc); zeros if out of view."""
    lr = _i(gr) - r + 2
    lc = _i(gc) - c + 2
    valid = (lr >= 0) & (lr <= 4) & (lc >= 0) & (lc <= 4)
    v = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), :]
    return jnp.where(valid, v, 0)


def _step_to(r, c, f, plan):
    """plan = [target_row, target_col, target_face, interact_flag]."""
    tr, tc, tf, inter = plan[0], plan[1], plan[2], plan[3]
    at = (r == tr) & (c == tc)
    faced = f == tf
    move = jnp.where(r < tr, A_DOWN, jnp.where(r > tr, A_UP, jnp.where(c < tc, A_RIGHT, A_LEFT)))
    turn = jnp.asarray(_FACE_TO_ACTION_LIST, dtype=I32)[jnp.clip(tf, 0, 3)]
    here = jnp.where(faced, jnp.where(inter > 0, A_INTERACT, A_STAY), turn)
    action = jnp.where(at, here, move).astype(I32)
    will_interact = at & faced & (inter > 0)
    return action, will_interact


def _plan(tr, tc, tf, inter):
    return jnp.stack([_i(tr), _i(tc), _i(tf), _i(inter)])


def _sel(cond, a, b):
    return jnp.where(cond, a, b)


# --------------------------------------------------------------------------
# agent_0 : recipe signaller
# --------------------------------------------------------------------------
# memory layout
M0_RECIPE_PREV = 0
M0_NEED = 1
M0_DELAY = 2
M0_SINCE_PRESS = 3
M0_CYCLE_PRESSES = 4
M0_TOTAL_PRESSES = 5
M0_PREV_CNT = 6
M0_PREV_MISMATCH = 7
M0_STEP = 8
M0_IGNORE = 9
M0_EVER_ACTIVE = 10
M0_READY_STEPS = 11
M0_SIZE = 12


def init_agent_0():
    m = jnp.zeros((M0_SIZE,), dtype=I32)
    m = m.at[M0_RECIPE_PREV].set(-1)
    m = m.at[M0_NEED].set(1)
    m = m.at[M0_SINCE_PRESS].set(999)
    m = m.at[M0_PREV_CNT].set(-1)
    return m


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_0(), memory)
    r, c, f = _self_pose(obs, CH_RECIPE_IND, 2, 0)

    # Always-on recipe indicator at (2,0)
    rcell = _cell(obs, r, c, 2, 0)
    rec_counts = rcell[CH_RECIPE + 2:CH_RECIPE + 5]
    recipe_now = jnp.argmax(rec_counts).astype(I32)
    recipe_prev = mem[M0_RECIPE_PREV]
    first = recipe_prev < 0
    recipe_changed = (~first) & (recipe_now != recipe_prev)

    # Pot at (2,4) visible from column 2
    pot_visible = c == 2
    pot = _cell(obs, r, c, 2, 4)
    pot_counts = pot[CH_ITEM + 2:CH_ITEM + 5]
    pot_cnt = jnp.sum(pot_counts).astype(I32)
    pot_cooked = pot[CH_ITEM + 1] > 0
    other_cnt = pot_cnt - pot_counts[jnp.clip(recipe_now, 0, 2)]
    mismatch = pot_visible & (pot_cnt > 0) & (other_cnt > 0)
    prev_cnt = mem[M0_PREV_CNT]
    prev_mismatch = mem[M0_PREV_MISMATCH] > 0
    pot_emptied = pot_visible & (prev_cnt > 0) & (pot_cnt == 0)
    pot_progress = pot_visible & (prev_cnt >= 0) & (pot_cnt > prev_cnt)
    new_mismatch = mismatch & (~prev_mismatch)

    need = mem[M0_NEED] > 0
    delay = mem[M0_DELAY]
    since = mem[M0_SINCE_PRESS]
    cyc = mem[M0_CYCLE_PRESSES]
    total = mem[M0_TOTAL_PRESSES]
    step = mem[M0_STEP]
    ignore = mem[M0_IGNORE]
    ever_active = (mem[M0_EVER_ACTIVE] > 0) | (pot_visible & (pot_cnt > 0))
    ready_steps = jnp.where(pot_visible & pot_cooked, jnp.minimum(mem[M0_READY_STEPS] + 1, 1000), 0)

    # Partner started an empty pot with a wrong ingredient while our display was
    # visible to it: evidence that it does not read the button.
    first_wrong_visible = pot_progress & (prev_cnt == 0) & mismatch & (since <= 10)
    ignore = jnp.where(first_wrong_visible, ignore + 1, ignore)
    quiet = ignore >= IGNORE_LIMIT

    # Partner shows it knows the recipe: an ingredient of the right type went in.
    need = jnp.where(pot_progress & (~new_mismatch), False, need)

    # Display still active and auto-updates to the new recipe -> no new press.
    display_fresh = since <= 5
    trig_now = first | (recipe_changed & (~display_fresh)) | new_mismatch
    need = need | trig_now
    delay = jnp.where(trig_now, 0, delay)
    # Dish left the pot: partner will deliver and then want the (possibly new) recipe.
    need_from_empty = pot_emptied & (~need)
    need = need | pot_emptied
    delay = jnp.where(need_from_empty, EMPTIED_DELAY, delay)
    cyc = jnp.where(pot_emptied | recipe_changed | pot_progress, 0, cyc)

    # Re-press schedule while the pot stays empty after a press.
    pot_idle_empty = pot_visible & (pot_cnt == 0) & (~pot_cooked)
    dormant_steps = jnp.where(ever_active, DORMANT_STEPS, DORMANT_STEPS_NEVER_ACTIVE)
    repress_ok = pot_idle_empty & (total > 0) & (~need) & (
        ((cyc < CYCLE_MAX_PRESSES) & (since >= REPRESS_STEPS)) | (since >= dormant_steps))
    need = need | repress_ok
    delay = jnp.where(repress_ok, 0, delay)

    # Cooked pot left untouched: partner may be waiting to confirm the recipe.
    ready_stall = (ready_steps >= READY_STALL_STEPS) & (since >= REPRESS_STEPS) & (~need)
    need = need | ready_stall
    delay = jnp.where(ready_stall, 0, delay)

    # Quiet mode: partner ignores the button; only a slow trickle of presses.
    quiet_ok = pot_idle_empty & (since >= SILENT_STEPS)
    allowed = jnp.where(quiet, quiet_ok, True) & (step <= LAST_USEFUL_PRESS_STEP)

    station = _plan(2, 2, F_RIGHT, 1)
    action, at_station = _step_to(r, c, f, station)
    press = at_station & need & (delay <= 0) & allowed
    action = jnp.where(at_station & (~press), _i(A_STAY), action)

    since_new = jnp.where(press, 0, jnp.minimum(since + 1, 999))
    need_new = jnp.where(press, False, need)
    delay_new = jnp.maximum(delay - 1, 0)
    cyc_new = jnp.where(press, cyc + 1, cyc)
    total_new = jnp.where(press, total + 1, total)

    new_mem = jnp.stack([
        recipe_now,
        need_new.astype(I32),
        delay_new.astype(I32),
        since_new.astype(I32),
        cyc_new.astype(I32),
        total_new.astype(I32),
        jnp.where(pot_visible, pot_cnt, -1).astype(I32),
        mismatch.astype(I32),
        jnp.minimum(step + 1, 10000).astype(I32),
        jnp.minimum(ignore, 100).astype(I32),
        ever_active.astype(I32),
        ready_steps.astype(I32),
    ])
    trace = {
        'press': press.astype(I32),
        'recipe': recipe_now,
        'pot_cnt': jnp.where(pot_visible, pot_cnt, -1).astype(I32),
        'total_presses': total_new.astype(I32),
        'quiet': quiet.astype(I32),
    }
    return action.astype(I32), new_mem, trace


# --------------------------------------------------------------------------
# agent_1 : solo cook
# --------------------------------------------------------------------------
M1_KNOWN = 0
M1_UNK_STEPS = 1
M1_LAST_DEL = 2
M1_LAST_INFORMED = 3
M1_MAX_DELAY = 4      # largest post-delivery unknown-step count at which a display appeared (-1: never)
M1_N_BLIND = 5        # consecutive deliveries made without recipe knowledge
M1_SEEN_ANY = 6       # partner has displayed the recipe at least once (including at episode start)
M1_SIZE = 7

# counters (cell) and where to stand / face to use them
_CNT_CELLS_LIST = [[0, 5], [0, 6], [4, 5], [4, 6]]
_CNT_STAND_LIST = [[1, 5, F_UP], [1, 6, F_UP], [3, 5, F_DOWN], [3, 6, F_DOWN]]


def init_agent_1():
    m = jnp.zeros((M1_SIZE,), dtype=I32)
    m = m.at[M1_KNOWN].set(-1)
    m = m.at[M1_LAST_DEL].set(-1)
    m = m.at[M1_MAX_DELAY].set(-1)
    return m


def _nearest_counter(r, c, f, mask):
    stand = jnp.asarray(_CNT_STAND_LIST, dtype=I32)
    dist = jnp.abs(stand[:, 0] - r) + jnp.abs(stand[:, 1] - c) + (stand[:, 2] != f).astype(I32)
    idx = jnp.argmin(jnp.where(mask, dist, 999))
    plan = jnp.stack([stand[idx, 0], stand[idx, 1], stand[idx, 2], _i(1)])
    return plan, jnp.any(mask)


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_1(), memory)
    known = mem[M1_KNOWN]
    unk = mem[M1_UNK_STEPS]
    last_del = mem[M1_LAST_DEL]
    last_informed = mem[M1_LAST_INFORMED] > 0
    max_delay = mem[M1_MAX_DELAY]
    n_blind = mem[M1_N_BLIND]
    seen_any = mem[M1_SEEN_ANY] > 0

    r, c, f = _self_pose(obs, CH_GOAL, 2, 7)

    inv = obs[2, 2, CH_SELF_INV:CH_SELF_INV + 5]
    h_plate = inv[0] > 0
    h_cooked = inv[1] > 0
    h_counts = inv[2:5]
    h_any_ing = jnp.sum(h_counts) > 0
    held_kind = jnp.where(h_cooked, 3, jnp.where(h_plate, 2, jnp.where(h_any_ing, 1, 0)))
    held_type = jnp.argmax(h_counts).astype(I32)

    # Pot
    pot = _cell(obs, r, c, 2, 4)
    p_counts = pot[CH_ITEM + 2:CH_ITEM + 5]
    p_cnt = jnp.sum(p_counts).astype(I32)
    p_ready = pot[CH_ITEM + 1] > 0
    p_cooking = pot[CH_TIMER] > 0
    p_idle = (~p_ready) & (~p_cooking)
    p_empty = p_cnt == 0
    p_type = jnp.argmax(p_counts).astype(I32)
    p_mixed = jnp.sum((p_counts > 0).astype(I32)) > 1

    # Button display (visible only from column 5)
    btn = _cell(obs, r, c, 2, 3)
    b_counts = btn[CH_RECIPE + 2:CH_RECIPE + 5]
    b_active = jnp.sum(b_counts) > 0
    b_type = jnp.argmax(b_counts).astype(I32)
    # Learn how long after a delivery this partner's signal shows up. The signal
    # at episode start is not informative about post-delivery behaviour, so it
    # only counts as evidence that the partner signals at all.
    signal_arrived = b_active & (known < 0)
    seen_any = seen_any | b_active
    max_delay = jnp.where(signal_arrived & (last_del >= 0), jnp.maximum(max_delay, unk), max_delay).astype(I32)
    seen = seen_any
    delay_known = max_delay >= 0
    known = jnp.where(b_active, b_type, known)
    K = known >= 0
    R = jnp.clip(known, 0, 2)

    # Counters
    items = jnp.stack([_cell(obs, r, c, cc[0], cc[1]) for cc in _CNT_CELLS_LIST])
    it_plate = items[:, CH_ITEM] > 0
    it_cooked = items[:, CH_ITEM + 1] > 0
    it_counts = items[:, CH_ITEM + 2:CH_ITEM + 5]
    it_has = it_plate | (jnp.sum(it_counts, axis=1) > 0)
    it_dish = it_cooked
    it_type = jnp.argmax(it_counts, axis=1).astype(I32)
    it_raw = it_has & (~it_plate) & (~it_cooked)
    it_free = ~it_has
    n_free = jnp.sum(it_free.astype(I32))
    stored = jnp.stack([jnp.any(it_dish & (it_type == 0)), jnp.any(it_dish & (it_type == 1)), jnp.any(it_dish & (it_type == 2))])
    stored0 = stored[0]
    stored1 = stored[1]

    # Which type to cook when the recipe is unknown:
    #  * a type we do not already have stored, else
    #  * if this partner has ever signalled, assume the recipe is unchanged
    #    (a partner that only signals changes would have told us), but flip
    #    after a long streak of blind deliveries in case it stopped signalling;
    #  * if the partner has never signalled, use the posterior after a blind
    #    gamble: a wrong delivery leaves the other recipe active.
    last_c = jnp.clip(last_del, 0, 1)
    other_last = (1 - last_c).astype(I32)
    flip_at = jnp.where(delay_known, BLIND_STREAK_FLIP_RESIGNALLER, BLIND_STREAK_FLIP_SILENT)
    guess_seen = jnp.where(n_blind >= flip_at, other_last, last_c)
    guess = jnp.where(last_del < 0, 0, jnp.where(seen, guess_seen, other_last)).astype(I32)
    spec_type = jnp.where(stored0 & (~stored1), 1, jnp.where(stored1 & (~stored0), 0, guess)).astype(I32)
    T = jnp.where(K, R, spec_type).astype(I32)
    gamble = (~K) & (unk >= G_GAMBLE)
    # Commit window adapts to the partner's observed signalling delay.
    short_wait = ((~seen) & (n_blind >= 1)) | (n_blind >= BLIND_STREAK_SHORT_WAIT)
    w1 = jnp.where(delay_known, jnp.clip(max_delay + W1_MARGIN, W1_MIN, W1_MAX),
                   jnp.where(short_wait, W1_DEAD, W1_DEFAULT))
    w0 = jnp.clip(w1 - W1_MARGIN, W0_MIN, W0_MAX)
    speculate_ok = (~K) & (unk >= w0)
    commit_ok = K | (unk >= w1)
    g_type = jnp.where(stored[jnp.clip(guess, 0, 1)], guess, 1 - guess).astype(I32)

    # Plans
    P_POT = _plan(2, 5, F_LEFT, 1)
    P_WAIT = _plan(2, 5, F_LEFT, 0)
    P_DELIVER = _plan(2, 6, F_RIGHT, 1)
    P_PLATE = _sel(r == 3, _plan(3, 5, F_LEFT, 1), _plan(1, 5, F_LEFT, 1))
    P_STORE, can_store = _nearest_counter(r, c, f, it_free)
    can_store2 = n_free >= 2

    def ing_plan(k):
        supply = _sel(k == 1, _plan(3, 6, F_RIGHT, 1), _plan(1, 6, F_RIGHT, 1))
        from_cnt, has_cnt = _nearest_counter(r, c, f, it_raw & (it_type == k))
        return _sel(has_cnt, from_cnt, supply)

    def dish_plan(k):
        return _nearest_counter(r, c, f, it_dish & (it_type == k))

    # ---- held dish
    d = held_type
    dish_match = K & (R == d)
    dish_wrong = K & (R != d)
    plan_dish = _sel(dish_match, P_DELIVER,
                _sel(dish_wrong, _sel(can_store, P_STORE, P_DELIVER),
                _sel(gamble, P_DELIVER,
                _sel(p_idle & p_empty & can_store, P_STORE, P_WAIT))))

    # ---- held plate
    plan_plate = _sel(p_ready, P_POT,
                 _sel(p_cooking, P_WAIT,
                 _sel(can_store2, P_STORE, P_WAIT)))

    # ---- held ingredient
    k = held_type
    accepts = p_idle & (p_cnt < 3) & (p_empty | ((p_type == k) & (~p_mixed)))
    plan_ing = _sel(accepts & (~p_empty), P_POT,
               _sel(accepts & p_empty & (k == T), _sel(commit_ok, P_POT, P_WAIT),
               _sel(accepts & p_empty, _sel(can_store2, P_STORE, P_POT),
               _sel(can_store2, P_STORE, P_WAIT))))

    # ---- empty hands
    take_match_plan, take_match_ok = dish_plan(R)
    take_match = K & take_match_ok
    take_gamble_plan, take_gamble_ok = dish_plan(g_type)
    take_gamble = gamble & take_gamble_ok
    both_stored = stored0 & stored1
    plan_empty = _sel(take_match, take_match_plan,
                 _sel(take_gamble, take_gamble_plan,
                 _sel(p_ready | p_cooking, P_PLATE,
                 _sel(p_idle & (~p_empty), ing_plan(p_type),
                 _sel(K, ing_plan(R),
                 _sel(speculate_ok & (~both_stored), ing_plan(spec_type), P_WAIT))))))

    plan = _sel(held_kind == 3, plan_dish,
           _sel(held_kind == 2, plan_plate,
           _sel(held_kind == 1, plan_ing, plan_empty)))

    action, will_interact = _step_to(r, c, f, plan)

    # Delivery bookkeeping
    delivering = will_interact & (held_kind == 3) & (plan[0] == 2) & (plan[1] == 6) & (plan[2] == F_RIGHT)
    known_new = jnp.where(delivering, -1, known)
    last_del_new = jnp.where(delivering, d, last_del)
    last_informed_new = jnp.where(delivering, K, last_informed)
    n_blind_new = jnp.where(delivering, jnp.where(K, 0, jnp.minimum(n_blind + 1, 100)), n_blind)
    unk_new = jnp.where(delivering, 0, jnp.where(known_new < 0, jnp.minimum(unk + 1, 999), 0))

    new_mem = jnp.stack([known_new.astype(I32), unk_new.astype(I32), last_del_new.astype(I32),
                         last_informed_new.astype(I32), max_delay.astype(I32), n_blind_new.astype(I32),
                         seen_any.astype(I32)])
    trace = {
        'known': known_new.astype(I32),
        'held_kind': _i(held_kind),
        'pot_cnt': p_cnt,
        'plan_r': plan[0],
        'plan_c': plan[1],
        'deliver': delivering.astype(I32),
        'gamble': gamble.astype(I32),
        'unk': unk_new.astype(I32),
        'w1': w1.astype(I32),
        'n_blind': n_blind_new.astype(I32),
    }
    return action.astype(I32), new_mem, trace
