"""Overcooked V2 ``test_time_wide`` joint policy.

Layout facts that drive the design (see context/environment/README.md):

* agent_0 lives in the top room (rows 1-2).  It can reach ingredient 0/1
  supplies, the plate pile, the delivery station and both pots, but it can
  never see the recipe indicator at (6, 2).
* agent_1 lives in the bottom room (rows 4-5).  It always sees the recipe
  indicator but can only touch ingredient 3 (never part of a recipe), a few
  counters and the two pots.  Its only useful contribution is to *tell*
  agent_0 the recipe through locally visible posture.

agent_1 (signaller): recipe [0,0,0] -> stand at (4,1) under the left pot,
facing it, empty handed.  recipe [1,1,1] -> stand at (4,3) under the right
pot, facing it, holding one unit of ingredient 3.  It never interacts with
pots or with the shared counters (3,2)/(3,4).

agent_0 (cook): keeps the left pot stocked with ingredient 0 and the right
pot with ingredient 1, plates the dish that matches the decoded recipe and
delivers it.  Dishes that stop matching the recipe are parked on the top
counters (0,1)/(0,4) and reused later.  The recipe decoder reads partner
column, held-item flag, facing and shared-counter items with prior weights,
and learns the partner's real convention online: a stable change of the
partner's posture after one of agent_0's deliveries implies that delivery was
correct (only correct deliveries resample the recipe).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'signal_cook_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1,
                   'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

# Facing direction ids (observation encoding).
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3
# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
DIR_TO_ACT = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# Prior decoder weights (negative -> recipe 0, positive -> recipe 1).
COL_W = (-1.0, -0.3, 1.0, 1.0)      # partner column 1..4
HOLD_W = (-0.4, 0.8)                # partner holds nothing / something
FACE_W = (0.0, 0.0, 0.15, -0.15)    # UP, DOWN, RIGHT, LEFT
C32_W = (0.0, 0.3)                  # item on shared counter (3,2)
C34_W = (0.0, 0.3)                  # item on shared counter (3,4)
TAB_OFF = (0, 4, 6, 10, 12)           # learned-table offsets per channel
LEARNED_W = 3.0
STABLE_FRAMES = 2
GRACE_STEPS = 7          # steps after a delivery before the partner signal is trusted again
UNCHANGED_STEPS = 9      # steps after a delivery before "signal unchanged" is concluded
CHANGE_WINDOW = 40       # window after a delivery in which a signal change is attributed to it
PEEK_STEPS = 8           # how long an empty-handed cook waits to read an unknown signal
TAB_RESET_RUN = 7        # consecutive unchanged outcomes under learned tables before dropping them
DISH_WAIT_STEPS = 40     # how long a cook holding a dish waits for an unknown signal


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _cell(obs, r, c, gr, gc):
    """Channel vector of global cell (gr, gc) seen from (r, c); zeros when out of view."""
    i = gr - r + 2
    j = gc - c + 2
    vis = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    v = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4)]
    return jnp.where(vis, v, 0), vis


def _locate(obs, channel, anchor_r, anchor_c):
    """Own global position from a static landmark that is always in view."""
    idx = jnp.argmax(obs[:, :, channel].reshape(-1))
    i = idx // 5
    j = idx % 5
    return anchor_r - (i - 2), anchor_c - (j - 2)


def _step_toward(r, c, tr, tc):
    return jnp.where(tr < r, A_UP, jnp.where(tr > r, A_DOWN,
                     jnp.where(tc < c, A_LEFT, A_RIGHT)))


def _go(r, c, f, tr, tc, tdir, interact):
    """Walk to (tr, tc) (open rooms, no obstacles), face tdir, then interact or stay."""
    at = (r == tr) & (c == tc)
    faced = f == tdir
    return jnp.where(~at, _step_toward(r, c, tr, tc),
                     jnp.where(~faced, jnp.asarray(DIR_TO_ACT, dtype=jnp.int32)[tdir],
                               jnp.where(interact, A_INTERACT, A_STAY)))


def _self_state(obs):
    me = obs[2, 2]
    f = jnp.argmax(me[1:5])
    plate = me[5] > 0
    cooked = me[6] > 0
    ing = me[7:11]
    return f, plate, cooked, ing


def _select(rules, default):
    """First rule whose condition holds wins; rules are (cond, (tr, tc, tdir, interact))."""
    tr, tc, td, it = default
    tr, tc, td = _i32(tr), _i32(tc), _i32(td)
    it = jnp.asarray(it, dtype=bool)
    which = _i32(len(rules))
    for k in range(len(rules) - 1, -1, -1):
        cond, (a, b, d, e) = rules[k]
        tr = jnp.where(cond, a, tr)
        tc = jnp.where(cond, b, tc)
        td = jnp.where(cond, d, td)
        it = jnp.where(cond, e, it)
        which = jnp.where(cond, k, which)
    return tr, tc, td, it, which


# ---------------------------------------------------------------------------
# agent_1: signaller in the bottom room
# ---------------------------------------------------------------------------

def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    r, c = _locate(obs, 25, 6, 2)
    f, plate, cooked, ing = _self_state(obs)
    holding = plate | cooked | (ing.sum() > 0)
    rc, _ = _cell(obs, r, c, 6, 2)
    recipe1 = rc[41] > 0
    want_hold = recipe1
    sig_c = jnp.where(recipe1, 3, 1)

    def item(gr, gc):
        v, vis = _cell(obs, r, c, gr, gc)
        return (v[32:38].sum() > 0), vis

    it63, _ = item(6, 3)
    it61, vis61 = item(6, 1)
    it64, vis64 = item(6, 4)
    it50, vis50 = item(5, 0)

    src_rules = [
        (it63, (5, 3, D_DOWN, True)),
        (vis61 & it61, (5, 1, D_DOWN, True)),
        (vis64 & it64, (5, 4, D_DOWN, True)),
        (c <= 2, (4, 1, D_LEFT, True)),
    ]
    dst_rules = [
        (~it63, (5, 3, D_DOWN, True)),
        (vis61 & ~it61, (5, 1, D_DOWN, True)),
        (vis64 & ~it64, (5, 4, D_DOWN, True)),
        (vis50 & ~it50, (5, 1, D_LEFT, True)),
    ]
    s_tr, s_tc, s_td, _, _ = _select(src_rules, (4, 4, D_RIGHT, True))
    d_tr, d_tc, d_td, _, _ = _select(dst_rules, (5, 4, D_RIGHT, True))

    need_take = want_hold & ~holding
    need_drop = (~want_hold) & holding
    tr = jnp.where(need_take, s_tr, jnp.where(need_drop, d_tr, 4))
    tc = jnp.where(need_take, s_tc, jnp.where(need_drop, d_tc, sig_c))
    td = jnp.where(need_take, s_td, jnp.where(need_drop, d_td, D_UP))
    interact = need_take | need_drop
    action = _go(r, c, f, tr, tc, td, interact)
    zero = jnp.asarray(0, dtype=jnp.int32)
    trace = {'recipe1': recipe1.astype(jnp.int32), 'row': _i32(r), 'col': _i32(c),
             'holding': holding.astype(jnp.int32)}
    return _i32(action), jnp.where(reset, zero, memory), trace


# ---------------------------------------------------------------------------
# agent_0: cook in the top room
# ---------------------------------------------------------------------------

def _split_code(code):
    col = code % 4
    hold = (code // 4) % 2
    face = (code // 8) % 4
    c32 = (code // 32) % 2
    c34 = (code // 64) % 2
    return jnp.stack([col, hold, face, c32, c34]).astype(jnp.int32)


def _item_sig(code):
    """Item-based part of a posture code (held item, shared counters)."""
    return (code // 4) % 2 + 2 * ((code // 32) % 2) + 4 * ((code // 64) % 2)


def _prior_score(code, pos_ok=True):
    v = _split_code(code)
    f32 = lambda w: jnp.asarray(w, dtype=jnp.float32)
    pos_part = f32(COL_W)[v[0]] + f32(FACE_W)[v[2]]
    return (jnp.where(pos_ok, pos_part, 0.0) + f32(HOLD_W)[v[1]]
            + f32(C32_W)[v[3]] + f32(C34_W)[v[4]])


def _norm_code(code, pos_ok):
    """Posture code with column/facing removed when position is known to be noise."""
    return jnp.where(pos_ok, code, code - code % 4 - 8 * ((code // 8) % 4))


def _decode(code, tab, bad, belief, pos_ok):
    v = _split_code(code)
    entries = tab[jnp.asarray(TAB_OFF, dtype=jnp.int32) + v]
    known = (entries >= 0) & (bad == 0)
    learned = jnp.sum(jnp.where(known, jnp.where(entries == 1, LEARNED_W, -LEARNED_W), 0.0))
    sign = jnp.where(belief >= 0.5, 1.0, -1.0)
    score = sign * _prior_score(code, pos_ok) + learned
    return (score > 0).astype(jnp.int32), jnp.any(known).astype(jnp.int32)


def _learn(tab, bad, old_code, new_code, old_dish, mask):
    """Old posture maps to the delivered dish, new posture to the other dish.

    Conflicts mark the channel bad.  ``mask`` selects which channels
    (col, hold, face, c32, c34) are updated.
    """
    vo = _split_code(old_code)
    vn = _split_code(new_code)
    new_dish = 1 - old_dish
    for k in range(5):
        if not mask[k]:
            continue
        io = TAB_OFF[k] + vo[k]
        e = tab[io]
        bad = bad.at[k].set(jnp.where((e >= 0) & (e != old_dish), 1, bad[k]))
        tab = tab.at[io].set(jnp.where(e < 0, old_dish, e))
        inn = TAB_OFF[k] + vn[k]
        e2 = tab[inn]
        bad = bad.at[k].set(jnp.where((e2 >= 0) & (e2 != new_dish), 1, bad[k]))
        tab = tab.at[inn].set(jnp.where(e2 < 0, new_dish, e2))
    return tab, bad


def init_agent_0():
    return {
        't': _i32(0),
        'pots': jnp.zeros((2, 6), dtype=jnp.int32),   # [n0,n1,n2,n3,cooked,timer] pot L, pot R
        'cnt': jnp.zeros((2, 6), dtype=jnp.int32),    # [plate,cooked,n0..n3] counters (0,1),(0,4)
        'recipe': _i32(0),
        'known': _i32(0),
        'last_code': _i32(-1),
        'stable_n': _i32(0),
        'stable_code': _i32(-1),
        'tab': -jnp.ones((14,), dtype=jnp.int32),
        'bad': jnp.zeros((5,), dtype=jnp.int32),
        'pending': _i32(0),
        'deliv_code': _i32(-1),
        'deliv_dish': _i32(-1),
        'deliv_t': _i32(-1000),
        'belief': jnp.asarray(0.85, dtype=jnp.float32),
        'unch_done': _i32(0),
        'tab_used': _i32(0),
        'deliv_tab': _i32(0),
        'tab_run': _i32(0),
        'wait': _i32(0),
        'c34': _i32(0),
        'spont': _i32(0),
    }


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    m = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    t = m['t']
    r, c = _locate(obs, 23, 0, 2)
    f, plate, cooked, ing = _self_state(obs)
    n_ing = ing.sum()
    holding_dish = cooked
    holding_plate = plate & ~cooked
    has_ing = (n_ing > 0) & ~cooked & ~plate
    empty = ~plate & ~cooked & (n_ing == 0)
    ing_id = jnp.argmax(ing)
    dish_type = jnp.where(cooked & (ing[0] == 3), 0, jnp.where(cooked & (ing[1] == 3), 1, -1))

    # ---- pots (memory with extrapolated timers when out of view) ----
    def pot_state(p, old):
        v, vis = _cell(obs, r, c, 3, 1 + 2 * p)
        seen = jnp.stack([v[34], v[35], v[36], v[37], v[33], v[44]])
        old_timer = old[5]
        extrap = old.at[5].set(jnp.maximum(old_timer - 1, 0))
        extrap = extrap.at[4].set(jnp.maximum(old[4], (old_timer == 1).astype(jnp.int32)))
        return jnp.where(vis, seen, extrap)

    pots = jnp.stack([pot_state(0, m['pots'][0]), pot_state(1, m['pots'][1])])

    # ---- parking counters (0,1) and (0,4) ----
    def cnt_state(gc, old):
        v, vis = _cell(obs, r, c, 0, gc)
        seen = jnp.stack([v[32], v[33], v[34], v[35], v[36], v[37]])
        return jnp.where(vis, seen, old)

    cnt = jnp.stack([cnt_state(1, m['cnt'][0]), cnt_state(4, m['cnt'][1])])
    cnt_empty = cnt.sum(axis=1) == 0
    cnt_dish = jnp.where((cnt[:, 1] > 0) & (cnt[:, 2] == 3), 0,
                         jnp.where((cnt[:, 1] > 0) & (cnt[:, 3] == 3), 1, -1))

    # ---- partner posture code ----
    pmask = obs[:, :, 11]
    vis_p = pmask.sum() > 0
    pidx = jnp.argmax(pmask.reshape(-1))
    pv = obs[pidx // 5, pidx % 5]
    pc = c + (pidx % 5) - 2
    col_code = jnp.clip(pc - 1, 0, 3)
    hold_code = (pv[16:22].sum() > 0).astype(jnp.int32)
    face_code = jnp.argmax(pv[12:16]).astype(jnp.int32)
    v32, _ = _cell(obs, r, c, 3, 2)
    c32 = (v32[32:38].sum() > 0).astype(jnp.int32)
    v34, vis34 = _cell(obs, r, c, 3, 4)
    c34 = jnp.where(vis34, (v34[32:38].sum() > 0).astype(jnp.int32), m['c34'])
    code = col_code + 4 * hold_code + 8 * face_code + 32 * c32 + 64 * c34
    same = vis_p & (code == m['last_code'])
    stable_n = jnp.where(vis_p, jnp.where(same, m['stable_n'] + 1, 1), 0)
    last_code = jnp.where(vis_p, code, m['last_code'])
    is_stable = stable_n >= STABLE_FRAMES
    fresh = t >= m['deliv_t'] + GRACE_STEPS
    use = is_stable & fresh

    # ---- partner moving on its own (not after one of my deliveries) => position is noise ----
    pending = m['pending']
    prev_stable = m['stable_code']
    col_changed_stable = use & (prev_stable >= 0) & ((code % 4) != (prev_stable % 4))
    spont = m['spont'] + jnp.where(col_changed_stable & (pending == 0), 1, 0)
    pos_ok = spont < 2
    tab, bad, belief = m['tab'], m['bad'], m['belief']
    bad = jnp.where(pos_ok, bad, bad.at[0].set(1).at[2].set(1))

    # ---- online convention learning ----
    ncode = _norm_code(code, pos_ok)
    ndeliv = _norm_code(m['deliv_code'], pos_ok)
    in_window = (pending == 1) & (t <= m['deliv_t'] + CHANGE_WINDOW)
    item_changed = use & in_window & (_item_sig(code) != _item_sig(m['deliv_code']))
    moved = use & in_window & ~item_changed & (ncode != ndeliv)
    changed = item_changed | moved
    unchanged = (use & (pending == 1) & (m['unch_done'] == 0) & (ncode == ndeliv)
                 & (t >= m['deliv_t'] + UNCHANGED_STEPS))
    prior_driven = m['deliv_tab'] == 0
    ps = _prior_score(m['deliv_code'], pos_ok)
    raw_dec = (ps > 0).astype(jnp.int32)
    informative = prior_driven
    agree = raw_dec == m['deliv_dish']
    odds = belief / (1.0 - belief)
    b_item = jnp.where(agree, jnp.maximum(belief, 0.995), jnp.minimum(belief, 0.005))
    odds_moved = jnp.where(agree, odds * 4.0, odds / 4.0)
    odds_unch = jnp.where(agree, odds / 2.0, odds * 2.0)
    b_moved = odds_moved / (1.0 + odds_moved)
    b_unch = odds_unch / (1.0 + odds_unch)
    belief = jnp.where(item_changed & informative, b_item,
                       jnp.where(moved & informative, b_moved,
                                 jnp.where(unchanged & informative, b_unch, belief)))
    belief = jnp.clip(belief, 0.002, 0.998)
    # Learned tables that keep predicting a recipe which never changes are probably garbage.
    tab_run = jnp.where(changed, 0, jnp.where(unchanged & ~prior_driven, m['tab_run'] + 1, m['tab_run']))
    reset_tab = tab_run >= TAB_RESET_RUN
    tab = jnp.where(reset_tab, -jnp.ones_like(tab), tab)
    bad = jnp.where(reset_tab, jnp.zeros_like(bad), bad)
    tab_run = jnp.where(reset_tab, 0, tab_run)
    belief = jnp.where(reset_tab, jnp.asarray(0.85, dtype=jnp.float32), belief)
    dish_c = jnp.clip(m['deliv_dish'], 0, 1)
    tab_all, bad_all = _learn(tab, bad, m['deliv_code'], code, dish_c, (1, 1, 1, 1, 1))
    tab_pos, bad_pos = _learn(tab, bad, m['deliv_code'], code, dish_c, (1, 0, 1, 0, 0))
    tab = jnp.where(item_changed, tab_all, jnp.where(moved, tab_pos, tab))
    bad = jnp.where(item_changed, bad_all, jnp.where(moved, bad_pos, bad))
    unch_done = jnp.where(unchanged, 1, m['unch_done'])
    pending = jnp.where(changed | (t > m['deliv_t'] + CHANGE_WINDOW), 0, pending)

    dec_recipe, dec_tab = _decode(code, tab, bad, belief, pos_ok)
    recipe = jnp.where(use, dec_recipe, m['recipe'])
    tab_used = jnp.where(use, dec_tab, m['tab_used'])
    known = jnp.where(use, 1, m['known'])
    stable_code = jnp.where(use, code, m['stable_code'])
    wait = jnp.where(known == 1, 0, m['wait'] + 1)
    R = recipe
    O = 1 - R

    # ---- pot predicates ----
    tot = pots[:, :4].sum(axis=1)
    timer = pots[:, 5]
    cookedp = pots[:, 4] > 0
    cooking = timer > 0
    idle = ~cooking & ~cookedp

    def usable(p, i):
        return idle[p] & (tot[p] < 3) & (pots[p, i] == tot[p])

    def ready(p, i):
        return cookedp[p] & (pots[p, i] == 3)

    def cookg(p, i):
        return cooking[p] & (pots[p, i] == 3)

    def choose(pred, i):
        d = i          # designated pot index equals the ingredient id (0 -> left, 1 -> right)
        o = 1 - i
        return jnp.where(pred(d, i), d, jnp.where(pred(o, i), o, -1))

    def choose_strict(pred, i):
        return jnp.where(pred(i, i), i, -1)

    rp = choose(ready, R)
    cp = choose(cookg, R)
    fp = choose_strict(usable, R)
    fo = choose_strict(usable, O)
    ing_ok = ing_id <= 1
    fi = choose_strict(usable, jnp.clip(ing_id, 0, 1))
    any_cooked = cookedp[0] | cookedp[1]
    # recipe-independent work: an idle pot that can take more of its designated ingredient
    fd = jnp.where(usable(0, 0) & (tot[0] > 0), 0, jnp.where(usable(1, 1) & (tot[1] > 0), 1,
                   jnp.where(usable(0, 0), 0, jnp.where(usable(1, 1), 1, -1))))
    any_cooking = cooking[0] | cooking[1]
    cooked_p = jnp.where(cookedp[0], 0, 1)

    # ---- targets ----
    def pot_t(p, interact):
        return (2, 1 + 2 * jnp.maximum(p, 0), D_DOWN, interact)

    def sup_t(i):
        row = 1 + i
        dl = jnp.abs(r - row) + jnp.abs(c - 1)
        dr = jnp.abs(r - row) + jnp.abs(c - 4)
        left = dl <= dr
        return (row, jnp.where(left, 1, 4), jnp.where(left, D_LEFT, D_RIGHT), True)

    def cnt_t(k):
        return (1, jnp.where(k == 0, 1, 4), D_UP, True)

    free_k = jnp.where(cnt_empty[0] & cnt_empty[1], jnp.where(c <= 2, 0, 1),
                       jnp.where(cnt_empty[0], 0, jnp.where(cnt_empty[1], 1, -1)))
    has_free = free_k >= 0
    match_k = jnp.where(cnt_dish[0] == R, 0, jnp.where(cnt_dish[1] == R, 1, -1))

    WAIT = (2, 2, D_DOWN, False)
    PLATE = (1, 3, D_UP, True)
    DELIVER = (1, 2, D_UP, True)
    unknown = known == 0
    known1 = known == 1
    dish_wait = unknown & (wait < DISH_WAIT_STEPS)
    start_peek = unknown & (t < PEEK_STEPS)
    near_done = (cp >= 0) & (timer[jnp.maximum(cp, 0)] <= 8)

    # Buffered production: pots are always refilled with their designated
    # ingredient; finished dishes that are not needed right now are parked on
    # the top counters so that any recipe can be served quickly later.
    rules = [
        # holding a cooked dish
        (holding_dish & known1 & (dish_type == R), DELIVER),
        (holding_dish & dish_wait & has_free, cnt_t(free_k)),
        (holding_dish & dish_wait, WAIT),
        (holding_dish & known1 & has_free, cnt_t(free_k)),
        (holding_dish, DELIVER),
        # holding an empty plate
        (holding_plate & known1 & (rp >= 0), pot_t(rp, True)),
        (holding_plate & known1 & near_done, pot_t(cp, False)),
        (holding_plate & any_cooked & has_free, pot_t(cooked_p, True)),
        (holding_plate & (cp >= 0), pot_t(cp, False)),
        (holding_plate & has_free, cnt_t(free_k)),
        (holding_plate, WAIT),
        # holding a raw ingredient
        (has_ing & ing_ok & (fi >= 0), pot_t(fi, True)),
        (has_ing & has_free, cnt_t(free_k)),
        (has_ing, WAIT),
        # empty handed
        (empty & known1 & (match_k >= 0), cnt_t(match_k)),
        (empty & known1 & (rp >= 0), PLATE),
        (empty & start_peek, WAIT),
        (empty & (fp >= 0), sup_t(R)),
        (empty & known1 & near_done, PLATE),
        (empty & any_cooked & has_free, PLATE),
        (empty & (fo >= 0), sup_t(O)),
        (empty & known1 & (cp >= 0), PLATE),
    ]
    tr, tc, td, it, which = _select(rules, WAIT)
    action = _go(r, c, f, tr, tc, td, it)

    # ---- delivery bookkeeping ----
    delivering = holding_dish & (r == 1) & (c == 2) & (f == D_UP) & (action == A_INTERACT)
    pending = jnp.where(delivering, 1, pending)
    deliv_code = jnp.where(delivering, stable_code, m['deliv_code'])
    deliv_dish = jnp.where(delivering, dish_type, m['deliv_dish'])
    deliv_t = jnp.where(delivering, t, m['deliv_t'])
    deliv_tab = jnp.where(delivering, tab_used, m['deliv_tab'])
    known = jnp.where(delivering, 0, known)
    wait = jnp.where(delivering, 0, wait)
    unch_done = jnp.where(delivering, 0, unch_done)

    new_m = {
        't': t + 1,
        'pots': pots,
        'cnt': cnt,
        'recipe': _i32(recipe),
        'known': _i32(known),
        'last_code': _i32(last_code),
        'stable_n': _i32(stable_n),
        'stable_code': _i32(stable_code),
        'tab': tab.astype(jnp.int32),
        'bad': bad.astype(jnp.int32),
        'pending': _i32(pending),
        'deliv_code': _i32(deliv_code),
        'deliv_dish': _i32(deliv_dish),
        'deliv_t': _i32(deliv_t),
        'belief': belief.astype(jnp.float32),
        'unch_done': _i32(unch_done),
        'tab_used': _i32(tab_used),
        'deliv_tab': _i32(deliv_tab),
        'tab_run': _i32(tab_run),
        'wait': _i32(wait),
        'c34': _i32(c34),
        'spont': _i32(spont),
    }
    trace = {'recipe': _i32(recipe), 'known': _i32(known), 'code': _i32(code),
             'rule': _i32(which), 'belief': belief.astype(jnp.float32), 'spont': _i32(spont),
             'row': _i32(r), 'col': _i32(c), 'deliver': delivering.astype(jnp.int32)}
    return _i32(action), new_m, trace
