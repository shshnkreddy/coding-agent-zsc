"""Grounded-coordination policy for overcooked_v2 grounded_coord_simple.

Roles (fixed by the layout, the two rooms are disconnected):
  agent_0 (left room)  : reads the always-on recipe indicator and relays it
                         through the button display (the only channel the
                         right room can observe). Presses at episode start,
                         after every pot pickup (delivery imminent), when the
                         recipe changes while no display is active, and as a
                         backoff "nudge" when the pot stays empty.
  agent_1 (right room) : waits at the pot for the button display, then runs
                         fetch x3 -> cook -> plate -> deliver. Falls back to
                         the last-known / inferred recipe after a bounded wait
                         so it never deadlocks with a silent partner.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'button_relay_cook_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# ---------------------------------------------------------------- constants
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
MAX_STEPS = 400
I32 = jnp.int32

# agent_1 tuning
W_START = 30       # steps to wait for a display at episode start before guessing
W_POST = 10        # steps to wait for a display after a delivery before guessing
W_POST_SILENT = 4  # post-delivery wait once the partner has stayed silent after a delivery
END_MARGIN = 45    # do not deliver a known-wrong dish if fewer steps remain
CACHE0_MIN_TIMER = 11   # pot timer needed to start caching ingredient 0 (6 steps + 4 for plate)
CACHE1_MIN_TIMER = 12   # pot timer needed to start caching ingredient 1 (7 steps + 4 for plate)
P1_READ_DISPLAY = True   # variant switch: ignore the display entirely (guessing partner)
PRIOR_LOGIT = 1.2          # prior log-odds that partner silence means "recipe unchanged"
POST_PRESS_LOGIT = 4.0     # evidence: partner re-signalled after a delivery
SILENT_WINDOW_LOGIT = -0.69  # evidence: silent post-delivery window (P=0.5 under unchanged model)
START_SILENT_LOGIT = -3.0  # evidence: partner never signalled at episode start
ODD_PRESS_LOGIT = -4.0     # evidence: display onset mid-cycle while certain (periodic partner)
W_PERIODIC_CAP = 34        # max post-delivery wait for a detected periodic partner

# agent_0 tuning
MIN_CYCLE = 40      # minimum steps for fill (17) + cook (19) + pickup/deliver (3) + margin
NUDGE_INIT = 18
NUDGE_CAP = 120
P0_PRESS_ON_CHANGE = True   # variant switches for partner emulation
P0_PRESS_ON_PICKUP = True
P0_NUDGE = True
P0_PRESS_FIRST = True


def _i(x):
    return jnp.asarray(x, dtype=I32)


# ---------------------------------------------------------------- decoding
def _locate(obs, ch, gr, gc):
    """Global (row, col) of self from a static landmark channel at global (gr, gc)."""
    m = obs[:, :, ch].reshape(-1)
    idx = jnp.argmax(m)
    found = m[idx] > 0
    lr = idx // 5
    lc = idx % 5
    return _i(gr - (lr - 2)), _i(gc - (lc - 2)), found


def _cell(obs, my_r, my_c, gr, gc, start, n):
    """Channel block [start, start+n) at global cell (gr, gc); zeros if out of view."""
    lr = gr - my_r + 2
    lc = gc - my_c + 2
    inview = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    blk = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), start:start + n]
    return jnp.where(inview, blk, 0), inview


def _facing(obs):
    return _i(jnp.argmax(obs[2, 2, 1:5]))


# ---------------------------------------------------------------- navigation
def _nav(my_r, my_c, facing, tr, tc, tf, final_action):
    """Move to (tr, tc), face tf, then emit final_action. Rooms are obstacle-free
    rectangles so axis-ordered moves always work. The move along the facing
    axis is done last so that arrival usually already sets the right facing."""
    dr = tr - my_r
    dc = tc - my_c
    vert = jnp.where(dr > 0, A_DOWN, A_UP)
    horiz = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    need_v = dr != 0
    need_h = dc != 0
    tf_h = (tf == F_RIGHT) | (tf == F_LEFT)
    first = jnp.where(tf_h, vert, horiz)
    first_need = jnp.where(tf_h, need_v, need_h)
    second = jnp.where(tf_h, horiz, vert)
    second_need = jnp.where(tf_h, need_h, need_v)
    turn = jnp.where(tf == F_UP, A_UP,
                     jnp.where(tf == F_DOWN, A_DOWN,
                               jnp.where(tf == F_RIGHT, A_RIGHT, A_LEFT)))
    act = jnp.where(first_need, first,
                    jnp.where(second_need, second,
                              jnp.where(facing != tf, turn, final_action)))
    return _i(act)


# ================================================================ agent_0
# memory: t, last_press_t, press_recipe, got_ing_since_press, nudge_interval,
#         pending_press, prev_pot_cooked, prev_pot_count, n_press
def init_agent_0():
    return jnp.array([0, -100, -1, 0, NUDGE_INIT, 0, 0, 0, 0], dtype=I32)


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_0(), memory)
    t, lp, pr, gi, ni, pp, pc, pn, npress = (mem[k] for k in range(9))

    my_r, my_c, _ = _locate(obs, 23, 2, 0)          # always-on indicator at (2,0)
    facing = _facing(obs)
    inv = obs[2, 2, 5:10]
    empty = inv.sum() == 0

    rblk, _ = _cell(obs, my_r, my_c, 2, 0, 36, 3)
    recipe = _i(jnp.argmax(rblk))
    dblk, _ = _cell(obs, my_r, my_c, 2, 3, 36, 3)
    disp_active = dblk.sum() > 0
    potblk, pot_in = _cell(obs, my_r, my_c, 2, 4, 29, 5)
    pot_cooked = potblk[1] > 0
    pot_n = potblk[2:5].sum()

    pickup = pot_in & (pc == 1) & (~pot_cooked) & (pot_n == 0)
    added = pot_in & (pot_n > pn)
    pp1 = jnp.where(pickup, 1, pp)
    gi1 = jnp.where(added, 1, gi)
    ni1 = jnp.where(added, NUDGE_INIT, ni)
    pr1 = jnp.where(disp_active, recipe, pr)       # active display tracks current recipe

    since = t - lp
    rule_first = (lp < 0) & P0_PRESS_FIRST
    rule_change = (recipe != pr1) & (since >= 6) & P0_PRESS_ON_CHANGE
    rule_pickup = (pp1 == 1) & P0_PRESS_ON_PICKUP
    rule_nudge = pot_in & (pot_n == 0) & (~pot_cooked) & (gi1 == 0) & (since >= ni1) & P0_NUDGE
    base = rule_first | rule_change | rule_pickup
    # a press only pays off if a full fill-cook-deliver cycle can still finish
    time_ok = t <= MAX_STEPS - MIN_CYCLE
    want = (base | rule_nudge) & (since >= 2) & empty & time_ok
    is_nudge = rule_nudge & (~base)

    a_press = _nav(my_r, my_c, facing, 2, 2, F_RIGHT, A_INTERACT)
    a_idle = _nav(my_r, my_c, facing, 2, 2, F_RIGHT, A_STAY)
    action = jnp.where(want, a_press, a_idle)
    pressing = want & (action == A_INTERACT)

    lp2 = jnp.where(pressing, t, lp)
    pr2 = jnp.where(pressing, recipe, pr1)
    gi2 = jnp.where(pressing, 0, gi1)
    pp2 = jnp.where(pressing, 0, pp1)
    ni2 = jnp.where(pressing & is_nudge, jnp.minimum(ni1 * 2, NUDGE_CAP), ni1)
    pc2 = jnp.where(pot_in, _i(pot_cooked), pc)
    pn2 = jnp.where(pot_in, _i(pot_n), pn)
    np2 = npress + _i(pressing)

    new_mem = jnp.stack([t + 1, lp2, pr2, gi2, ni2, pp2, pc2, pn2, np2]).astype(I32)
    trace = {
        "press": _i(pressing),
        "recipe": recipe,
        "rule_first": _i(rule_first),
        "rule_change": _i(rule_change),
        "rule_pickup": _i(rule_pickup),
        "rule_nudge": _i(is_nudge),
        "n_press": np2,
    }
    return _i(action), new_mem, trace


# ================================================================ agent_1
# memory: ints  = t, belief, certain, wait, prev_dish, prev_dish_ing, n_deliv,
#                 stage, silent, ever_certain, slot(0,5) state, slot(4,5) state,
#                 cache task (slot state: 0 empty, 1 intended ingredient, 2 other item),
#                 prev display visible, last display onset step, onset period estimate,
#                 count of "odd" onsets (mid-cycle while certain -> periodic partner)
#         floats = logit (log-odds that partner silence means "recipe unchanged"),
#                  p0 (P[recipe == 0] under the "silence is uninformative" model)
def init_agent_1():
    return (jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=I32),
            jnp.array([PRIOR_LOGIT, 0.5], dtype=jnp.float32))


def agent_1_act(obs, reset, memory, rng):
    init_i, init_f = init_agent_1()
    mem = jnp.where(reset, init_i, memory[0])
    memf = jnp.where(reset, init_f, memory[1])
    (t, b, c, w, prev_dish, prev_dish_ing, ndeliv, stage, silent, ever_c,
     s05, s45, ctask, prev_disp, last_edge, period, odd_n) = (mem[k] for k in range(17))
    logit, p0 = memf[0], memf[1]

    my_r, my_c, _ = _locate(obs, 21, 2, 7)          # delivery goal at (2,7)
    facing = _facing(obs)
    inv = obs[2, 2, 5:10]
    has_dish = inv[1] > 0
    has_plate = (inv[0] > 0) & (~has_dish)
    ingc = inv[2:5]
    has_ing = (ingc.sum() > 0) & (~has_dish)
    ing_id = _i(jnp.argmax(ingc))
    empty = inv.sum() == 0

    potblk, _ = _cell(obs, my_r, my_c, 2, 4, 29, 5)
    timer, _ = _cell(obs, my_r, my_c, 2, 4, 39, 1)
    pot_cooked = potblk[1] > 0
    pot_counts = potblk[2:5]
    pot_n = pot_counts.sum()
    pot_cooking = timer[0] > 0
    pot_ing = _i(jnp.argmax(pot_counts))

    disp, disp_in = _cell(obs, my_r, my_c, 2, 3, 36, 3)
    disp_active = (disp.sum() > 0) & P1_READ_DISPLAY
    disp_recipe = _i(jnp.argmax(disp))

    c13, _ = _cell(obs, my_r, my_c, 1, 3, 29, 5)
    c33, _ = _cell(obs, my_r, my_c, 3, 3, 29, 5)
    h13 = c13.sum() > 0
    h33 = c33.sum() > 0
    hint_avail = (h13 != h33) & disp_in
    hint = _i(jnp.where(h13, 0, 1))

    # --- belief update
    delivered = (prev_dish == 1) & empty
    d = prev_dish_ing
    # "unchanged unless told" model: b only changes via displays; certainty
    # survives a known-wrong delivery (recipe is not resampled then).
    c1 = jnp.where(delivered, jnp.where((c == 1) & (d != b), 1, 0), c)
    b1 = jnp.where(disp_active, disp_recipe, b)
    c1 = jnp.where(disp_active, 1, c1)
    # "uninformative silence" model: P[recipe == d] halves on delivery of d
    # (correct -> resampled 50/50, wrong -> recipe is the other one).
    pd = jnp.where(d == 0, p0, 1.0 - p0)
    pd_new = pd * 0.5
    p0_1 = jnp.where(delivered, jnp.where(d == 0, pd_new, 1.0 - pd_new), p0)
    p0_1 = jnp.where(disp_active, jnp.where(disp_recipe == 0, 1.0, 0.0), p0_1)
    stage1 = jnp.where(delivered, 1, stage)
    w1 = jnp.where(delivered | disp_active, 0, w)
    # periodic-press partner detection: a display onset while we are already
    # certain (mid-cycle, not at a delivery) cannot come from an on-change or
    # pickup-triggered signaller.
    c_mid = jnp.where(delivered, jnp.where((c == 1) & (d != b), 1, 0), c)
    # onset = display visible now and the button cell was in view with no
    # display last step (walking out of view and back must not count)
    edge = disp_active & (prev_disp == 0)
    seen_or_unseen = disp_active | (~disp_in)
    odd = edge & (c_mid == 1) & (~delivered)
    period1 = jnp.where(edge & (last_edge >= 0), t - last_edge, period)
    last_edge1 = jnp.where(edge, t, last_edge)
    odd_n1 = odd_n + _i(odd)
    periodic = odd_n1 >= 2
    w_periodic = jnp.clip(period1 + 3, 4, W_PERIODIC_CAP)
    w_post = jnp.where(periodic, w_periodic, jnp.where(silent == 1, W_POST_SILENT, W_POST))
    wait_limit = jnp.where(stage1 == 1, w_post, W_START)
    expire = (c1 == 0) & (w1 == wait_limit)
    post_press = disp_active & (stage == 1) & (c == 0)   # partner re-signalled after a delivery
    logit1 = logit + jnp.where(post_press, POST_PRESS_LOGIT, 0.0)
    logit1 = logit1 + jnp.where(odd, ODD_PRESS_LOGIT, 0.0)
    logit1 = logit1 + jnp.where(expire & (stage1 == 1), SILENT_WINDOW_LOGIT, 0.0)
    logit1 = logit1 + jnp.where(expire & (stage1 == 0), START_SILENT_LOGIT, 0.0)
    ever_c1 = jnp.where(disp_active, 1, ever_c)
    s = jax.nn.sigmoid(logit1)
    p_zero = jnp.where(ever_c1 == 1, s * jnp.where(b1 == 0, 1.0, 0.0) + (1.0 - s) * p0_1, p0_1)
    guess = _i(jnp.where(p_zero >= 0.5, 0, 1))
    blind = (ever_c1 == 0) & (stage1 == 0)
    b_eff = jnp.where(c1 == 1, b1, jnp.where(blind & hint_avail, hint, guess))
    ready = (c1 == 1) | (w1 >= wait_limit)
    # partner stayed silent through a full post-delivery window: shorten later waits
    silent1 = jnp.where((stage1 == 1) & (c1 == 0) & (w1 >= wait_limit), 1, silent)

    target_ing = jnp.where(pot_n > 0, pot_ing, b_eff)
    remaining = MAX_STEPS - t
    known_wrong = has_dish & (c1 == 1) & (ing_id != b1)
    hold = known_wrong & (remaining < END_MARGIN)

    # --- candidate actions
    a_goal = _nav(my_r, my_c, facing, 2, 6, F_RIGHT, A_INTERACT)
    a_pot = _nav(my_r, my_c, facing, 2, 5, F_LEFT, A_INTERACT)
    a_waitpot = _nav(my_r, my_c, facing, 2, 5, F_LEFT, A_STAY)
    plate_r = jnp.where(my_r <= 2, 1, 3)
    a_plate = _nav(my_r, my_c, facing, plate_r, 5, F_LEFT, A_INTERACT)
    ing_r = jnp.where(target_ing == 0, 1, 3)
    a_ing = _nav(my_r, my_c, facing, ing_r, 6, F_RIGHT, A_INTERACT)

    # --- counter cache slots: (0,5) holds one ingredient-0 unit, (4,5) one ingredient-1 unit
    blk05, in05 = _cell(obs, my_r, my_c, 0, 5, 29, 5)
    blk45, in45 = _cell(obs, my_r, my_c, 4, 5, 29, 5)
    st05_obs = jnp.where(blk05.sum() == 0, 0,
                         jnp.where((blk05[2] > 0) & (blk05[0] == 0) & (blk05[1] == 0), 1, 2))
    st45_obs = jnp.where(blk45.sum() == 0, 0,
                         jnp.where((blk45[3] > 0) & (blk45[0] == 0) & (blk45[1] == 0), 1, 2))
    s05_1 = jnp.where(in05, st05_obs, s05)
    s45_1 = jnp.where(in45, st45_obs, s45)
    cache0_ok = s05_1 == 1
    cache1_ok = s45_1 == 1
    e05 = s05_1 == 0
    e45 = s45_1 == 0
    e06 = _cell(obs, my_r, my_c, 0, 6, 29, 5)[0].sum() == 0
    e46 = _cell(obs, my_r, my_c, 4, 6, 29, 5)[0].sum() == 0

    put05 = _nav(my_r, my_c, facing, 1, 5, F_UP, A_INTERACT)
    put45 = _nav(my_r, my_c, facing, 3, 5, F_DOWN, A_INTERACT)
    put06 = _nav(my_r, my_c, facing, 1, 6, F_UP, A_INTERACT)
    put46 = _nav(my_r, my_c, facing, 3, 6, F_DOWN, A_INTERACT)
    grab0 = _nav(my_r, my_c, facing, 1, 6, F_RIGHT, A_INTERACT)
    grab1 = _nav(my_r, my_c, facing, 3, 6, F_RIGHT, A_INTERACT)

    ing0_held = has_ing & (ing_id == 0)
    ing1_held = has_ing & (ing_id == 1)
    top_first = my_r <= 2
    a_gen = jnp.where(top_first,
                      jnp.where(e06, put06, jnp.where(e46, put46, jnp.where(e05, put05, put45))),
                      jnp.where(e46, put46, jnp.where(e06, put06, jnp.where(e45, put45, put05))))
    a_drop = jnp.where(ing0_held & e05, put05, jnp.where(ing1_held & e45, put45, a_gen))

    # fetch for the pot: matching cache slot (5 steps from the pot) beats the supply (6)
    use_cache = jnp.where(target_ing == 0, cache0_ok, cache1_ok)
    a_fetch = jnp.where(use_cache, jnp.where(target_ing == 0, put05, put45), a_ing)

    waiting = (pot_n == 0) & (~ready)
    a_idle = jnp.where(waiting,
                       jnp.where(empty, a_waitpot, a_drop),
                       jnp.where(has_ing & (ing_id == target_ing), a_pot,
                                 jnp.where(empty, a_fetch, a_drop)))
    timer_v = timer[0]
    # cache task: 0 none, 1 caching ingredient 0, 2 caching ingredient 1.
    # The timer threshold gates only the start; a started task runs to completion.
    ctask1 = jnp.where(~pot_cooking, 0, ctask)
    ctask1 = jnp.where((ctask1 == 1) & (~e05), 0, ctask1)
    ctask1 = jnp.where((ctask1 == 2) & (~e45), 0, ctask1)
    can_start = pot_cooking & empty & (ctask1 == 0)
    want_c0 = e05 & (timer_v >= CACHE0_MIN_TIMER)
    want_c1 = e45 & (timer_v >= CACHE1_MIN_TIMER)
    ctask1 = jnp.where(can_start & want_c0, 1, jnp.where(can_start & want_c1, 2, ctask1))
    a_cook_empty = jnp.where(ctask1 == 1, grab0, jnp.where(ctask1 == 2, grab1, a_plate))
    a_cooking = jnp.where(has_plate, a_waitpot, jnp.where(empty, a_cook_empty, a_drop))
    caching = pot_cooking & (~has_plate) & ((ctask1 > 0) | has_ing)
    a_cooked = jnp.where(has_plate, a_pot, jnp.where(empty, a_plate, a_drop))
    a_dish = jnp.where(hold, A_STAY, a_goal)
    action = jnp.where(has_dish, a_dish,
                       jnp.where(pot_cooked, a_cooked,
                                 jnp.where(pot_cooking, a_cooking, a_idle)))

    phase = jnp.where(has_dish, 4, jnp.where(pot_cooked, 3, jnp.where(pot_cooking, jnp.where(caching, 5, 2), jnp.where(waiting, 0, 1))))

    w2 = jnp.where(c1 == 1, 0, w1 + 1)
    new_mem_i = jnp.stack([
        t + 1, b1, c1, w2, _i(has_dish),
        jnp.where(has_dish, ing_id, prev_dish_ing),
        ndeliv + _i(delivered), stage1, silent1, ever_c1, s05_1, s45_1, ctask1,
        _i(seen_or_unseen), last_edge1, period1, odd_n1,
    ]).astype(I32)
    new_mem_f = jnp.stack([logit1, p0_1]).astype(jnp.float32)
    trace = {
        "phase": _i(phase),
        "belief": b1,
        "certain": c1,
        "wait": w2,
        "disp_active": _i(disp_active),
        "n_deliv": ndeliv + _i(delivered),
        "pot_n": _i(pot_n),
        "row": my_r,
        "col": my_c,
        "silent": silent1,
        "guess": guess,
        "s05": s05_1,
        "s45": s45_1,
        "odd_n": odd_n1,
        "period": period1,
        "logit": logit1,
        "p0": p0_1,
    }
    return _i(action), (new_mem_i, new_mem_f), trace
