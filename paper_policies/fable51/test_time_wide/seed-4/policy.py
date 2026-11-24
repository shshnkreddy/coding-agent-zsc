"""Overcooked V2 test_time_wide joint policy.

Layout facts (see context/environment/README.md):
  * agent_0 lives in the top room (rows 1-2, cols 1-4): has both usable
    ingredient supplies, both pots (from above), plates and delivery, but can
    NEVER see the always-on recipe indicator at (6, 2).
  * agent_1 lives in the bottom room (rows 4-5, cols 1-4): always sees the
    recipe indicator, can touch both pots from below and the pass-through
    counters (3, 2) / (3, 4), but has no useful supplies.

Protocol (agent_0 = top cook, agent_1 = bottom signaller/helper):
  * Top cooks ingredient 0 in the left pot (3, 1) and ingredient 1 in the
    right pot (3, 3) continuously, so a correct dish is always available
    once the recipe is known.
  * Bottom stands directly below the pot whose index equals the recipe
    ingredient (col 1 -> recipe 0, col 3 -> recipe 1), facing up.
  * Top reads the partner's column when the partner is stationary on row 4
    (col <= 2 -> recipe 0, col >= 3 -> recipe 1) and feeds the readings into
    a small Bayesian filter over (partner model) x (recipe).  Partner models:
    live (signal == recipe), reversed (signal == 1 - recipe) and
    uninformative (constant/random).  A correct delivery resamples the
    recipe while a wrong one leaves it unchanged, so the filter makes the
    top alternate dish types when the partner is uninformative (positive
    expected value) instead of repeating the same wrong dish.
  * Robustness fallbacks for unknown partners: top delivers any valid cooked
    dish a partner leaves on a pass-through counter; if the partner has
    never signalled, top offers a plate on counter (3, 2) so the partner can
    plate the correct pot.  Bottom, if it finds a plate on a pass-through
    counter, plates the pot matching the recipe and returns the dish.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'typed_dual_pot_signal', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

# Actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3

I32 = jnp.int32
F32 = jnp.float32


def _face_action():
    return jnp.array([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=I32)


def _i(x):
    return jnp.asarray(x, dtype=I32)


# ---------------------------------------------------------------- helpers
def _argmax2d(ch):
    idx = jnp.argmax(ch.reshape(-1))
    return (idx // 5).astype(I32), (idx % 5).astype(I32)


def _cell(obs, gr, gc, sr, sc):
    """Return the 45-channel vector at global (gr, gc) and a visibility flag."""
    i = gr - sr + 2
    j = gc - sc + 2
    vis = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    v = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4)]
    return v, vis


def _update_cells(obs, sr, sc, coords, cells_mem):
    def one(coord, mem_row):
        v, vis = _cell(obs, coord[0], coord[1], sr, sc)
        new = jnp.concatenate([v[32:38], v[44:45]]).astype(I32)
        return jnp.where(vis, new, mem_row)
    return jax.vmap(one)(coords, cells_mem)


def _nav(sr, sc, facing, tr, tc, tdir, do_interact):
    """Move to (tr, tc) inside an obstacle-free room, face tdir, then interact/stay."""
    a_h = jnp.where(sc < tc, A_RIGHT, A_LEFT)
    a_v = jnp.where(sr < tr, A_DOWN, A_UP)
    move = jnp.where(sc != tc, a_h, a_v)
    at = (sr == tr) & (sc == tc)
    face_ok = facing == tdir
    act_at = jnp.where(face_ok, jnp.where(do_interact > 0, A_INTERACT, A_STAY), _face_action()[tdir])
    return jnp.where(at, act_at, move).astype(I32)


def _block_info(b):
    """b: 7-vector [plate, cooked, c0, c1, c2, c3, timer]."""
    plate = b[0]
    cooked = b[1]
    counts = b[2:6]
    total = counts.sum()
    timer = b[6]
    empty = (total == 0) & (plate == 0) & (cooked == 0)
    plate_only = (plate == 1) & (cooked == 0)
    dish = (plate == 1) & (cooked == 1)
    type0 = (counts[0] == 3) & (total == 3)
    type1 = (counts[1] == 3) & (total == 3)
    dtype = jnp.where(type0, 0, jnp.where(type1, 1, -1)).astype(I32)
    pure0 = (counts[0] == total) & (total > 0)
    pure1 = (counts[1] == total) & (total > 0)
    garbage = (total > 0) & (~pure0) & (~pure1)
    idle = (timer == 0) & (cooked == 0)
    ready = cooked == 1
    return dict(plate=plate, cooked=cooked, counts=counts, total=total, timer=timer,
                empty=empty, plate_only=plate_only, dish=dish, dtype=dtype,
                pure0=pure0, pure1=pure1, garbage=garbage, idle=idle, ready=ready)


def _pot_accepts(p, k):
    pure_k = jnp.where(k == 0, p['pure0'], p['pure1'])
    return p['idle'] & (p['total'] < 3) & ((p['total'] == 0) | pure_k | p['garbage'])


def _decode(obs, anchor_ch, ar, ac):
    i, j = _argmax2d(obs[:, :, anchor_ch])
    sr = _i(ar) - i + 2
    sc = _i(ac) - j + 2
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(I32)
    held = obs[2, 2, 5:11].astype(I32)
    pvis = obs[:, :, 11].sum() > 0
    pi, pj = _argmax2d(obs[:, :, 11])
    pr = sr + pi - 2
    pc = sc + pj - 2
    return sr, sc, facing, held, pvis, pr, pc


def _partner_extra(obs, pvis):
    """Partner facing id (UP=0, DOWN=1, RIGHT=2, LEFT=3) and held-item flag."""
    pi, pj = _argmax2d(obs[:, :, 11])
    pface = jnp.argmax(obs[pi, pj, 12:16]).astype(I32)
    pitem = (obs[pi, pj, 16:22].sum() > 0).astype(I32)
    return jnp.where(pvis, pface, -1).astype(I32), jnp.where(pvis, pitem, -1).astype(I32)


def _held_info(held):
    hp = held[0]
    hc = held[1]
    counts = held[2:6]
    total = counts.sum()
    hd = (hp == 1) & (hc == 1)
    hpl = (hp == 1) & (hc == 0)
    hraw = (hp == 0) & (hc == 0) & (total > 0)
    he = (hp == 0) & (hc == 0) & (total == 0)
    rawk = jnp.argmax(counts).astype(I32)
    type0 = (counts[0] == 3) & (total == 3)
    type1 = (counts[1] == 3) & (total == 3)
    dtype = jnp.where(type0, 0, jnp.where(type1, 1, -1)).astype(I32)
    return hd, hpl, hraw, he, rawk, dtype


# ================================================================ agent 0
# tracked cells: 0 potL(3,1) 1 potR(3,3) 2 c32 3 c34 4 c01 5 c04
def _cells0():
    return jnp.array([[3, 1], [3, 3], [3, 2], [3, 4], [0, 1], [0, 4]], dtype=I32)


# tasks: tr, tc, dir, interact
T_GET0, T_GET1, T_POTL, T_POTR, T_PLATE, T_DELIVER, T_C32, T_C34, T_C01, T_C04, T_WAIT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10


def _tasks0():
    return jnp.array([
        [1, 1, D_LEFT, 1],   # GET0 from (1,0)
        [2, 4, D_RIGHT, 1],  # GET1 from (2,5)
        [2, 1, D_DOWN, 1],   # POTL
        [2, 3, D_DOWN, 1],   # POTR
        [1, 3, D_UP, 1],     # PLATE
        [1, 2, D_UP, 1],     # DELIVER
        [2, 2, D_DOWN, 1],   # C32
        [2, 4, D_DOWN, 1],   # C34
        [1, 1, D_UP, 1],     # C01
        [1, 4, D_UP, 1],     # C04
        [2, 2, D_DOWN, 0],   # WAIT at (2,2)
    ], dtype=I32)


WAIT_BEFORE_OFFER = 10
OFFER_TIMEOUT = 40
OFFER_TAKEN_TIMEOUT = 100
FRESH_DELAY = 4
CONF_THRESH = 0.55
FORCE_WAIT = 12

# Three partner signal channels, each with four partner models:
#   live (signal == recipe), reversed (signal == 1 - recipe),
#   constant (signal never changes), random (signal independent of recipe).
# channel 0: column on row 4 (relative/absolute mapping)
# channel 1: facing LEFT -> 0 / RIGHT -> 1
# channel 2: holding any item -> 1
PRIOR_COL = (0.84, 0.02, 0.04, 0.10)
PRIOR_FACE = (0.40, 0.15, 0.20, 0.25)
PRIOR_ITEM = (0.08, 0.08, 0.42, 0.42)
LIK_LIVE = 0.95
LOOK_STEPS = 8


def _prior():
    a = jnp.array(PRIOR_COL, dtype=F32)
    b = jnp.array(PRIOR_FACE, dtype=F32)
    c = jnp.array(PRIOR_ITEM, dtype=F32)
    joint = a[:, None, None] * b[None, :, None] * c[None, None, :]
    return jnp.stack([joint * 0.5, joint * 0.5], axis=-1)   # [4,4,4,2]


def _channel_lik(s_val, last_read):
    """[4,2] likelihood of reading s_val under live/rev/const/rand for R=0/1."""
    ar2 = jnp.arange(2)
    lik_live = jnp.where(ar2 == s_val, LIK_LIVE, 1.0 - LIK_LIVE).astype(F32)
    lik_rev = jnp.where(ar2 == s_val, 1.0 - LIK_LIVE, LIK_LIVE).astype(F32)
    lik_const_scalar = jnp.where(last_read < 0, 0.5, jnp.where(s_val == last_read, 0.98, 0.02)).astype(F32)
    lik_const = jnp.ones((2,), dtype=F32) * lik_const_scalar
    lik_rand = jnp.full((2,), 0.5, dtype=F32)
    return jnp.stack([lik_live, lik_rev, lik_const, lik_rand])


def init_agent_0():
    return {
        't': _i(0),
        'post': _prior(),          # [col_model, face_model, item_model, recipe]
        'last_read': jnp.full((3,), -1, dtype=I32),   # last stable reading per channel
        'pend_ch': jnp.zeros((3,), dtype=I32),        # per-channel post-delivery re-read pending
        'nread': _i(0),            # number of reading events (any channel)
        'pending': _i(0),          # post-delivery reading pending (any channel)
        'p_row': _i(-1),
        'p_col': _i(-1),
        'p_face': _i(-1),
        'p_item': _i(-1),
        'p_stable': _i(0),
        'col_min': _i(99),
        'col_max': _i(-99),
        'last_del_t': _i(-1000),
        'last_del_type': _i(-1),
        'mode': _i(0),      # 0 normal, 1 plate offered, 2 offer finished
        'stage': _i(0),     # offer stage: 0 need to place, 1 placed
        'offer_t': _i(0),
        'wait_unc': _i(0),  # steps with a dish available but uncertain recipe
        'wait_t': _i(0),    # steps with a dish available and no reading ever
        'trusted': _i(0),
        'prev_task': _i(T_WAIT),
        'prev_hd': _i(0),
        'prev_hpl': _i(0),
        'prev_he': _i(1),
        'prev_dtype': _i(-1),
        'cells': jnp.zeros((6, 7), dtype=I32),
    }


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    m = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(I32)
    t = m['t']

    sr, sc, facing, held, pvis, pr, pc = _decode(obs, 23, 0, 2)
    hd, hpl, hraw, he, rawk, dtype = _held_info(held)
    cells = _update_cells(obs, sr, sc, _cells0(), m['cells'])
    potL = _block_info(cells[0])
    potR = _block_info(cells[1])
    c32 = _block_info(cells[2])
    c34 = _block_info(cells[3])
    c01 = _block_info(cells[4])
    c04 = _block_info(cells[5])

    # ---- event detection from previous step
    delivered = (m['prev_task'] == T_DELIVER) & (m['prev_hd'] == 1) & he
    del_type = m['prev_dtype']
    last_del_t = jnp.where(delivered, t, m['last_del_t'])
    picked_counter = ((m['prev_task'] == T_C32) | (m['prev_task'] == T_C34)) & (m['prev_he'] == 1) & hd
    trusted = jnp.where(hd, jnp.where(picked_counter, 1, m['trusted']), 0)
    placed_offer = ((m['prev_task'] == T_C32) | (m['prev_task'] == T_C34)) & (m['prev_hpl'] == 1) & he & (m['mode'] == 1) & (m['stage'] == 0)

    # ---- posterior: delivery transition (recipe resamples iff delivery was correct)
    post = m['post']
    dk_del = jnp.clip(del_type, 0, 1)
    p_x = post[..., dk_del]                     # mass on recipe == delivered type
    trans = post.at[..., dk_del].set(0.5 * p_x)
    trans = trans.at[..., 1 - dk_del].add(0.5 * p_x)
    post = jnp.where(delivered & (del_type >= 0), trans, post)
    pending = jnp.where(delivered, 1, m['pending'])
    pend_ch = jnp.where(delivered, jnp.ones((3,), dtype=I32), m['pend_ch'])

    # ---- stable signal readings from the partner (row 4 => signal)
    pface, pitem = _partner_extra(obs, pvis)
    on_row4 = pvis & (pr == 4)
    same_cell = pvis & (pr == m['p_row']) & (pc == m['p_col'])
    p_stable = jnp.where(same_cell, m['p_stable'] + 1, jnp.where(pvis, 1, 0))
    p_row = jnp.where(pvis, pr, m['p_row'])
    p_col = jnp.where(pvis, pc, m['p_col'])
    same_face = pvis & (pface == m['p_face'])
    same_item = pvis & (pitem == m['p_item'])
    p_face = jnp.where(pvis, pface, m['p_face'])
    p_item = jnp.where(pvis, pitem, m['p_item'])
    # relative column mapping: once the partner has parked at two different
    # columns, the smaller one means recipe 0 and the larger recipe 1;
    # before that, use the absolute rule (col <= 2 -> 0).
    stable_pos = on_row4 & (p_stable >= 2)
    col_min = jnp.where(stable_pos, jnp.minimum(m['col_min'], pc), m['col_min'])
    col_max = jnp.where(stable_pos, jnp.maximum(m['col_max'], pc), m['col_max'])
    two_cols = col_max > col_min
    s_rel = jnp.where(2 * pc <= col_min + col_max, 0, 1)
    s_abs = jnp.where(pc <= 2, 0, 1)
    s_col = jnp.where(two_cols, s_rel, s_abs).astype(I32)
    s_face = jnp.where(pface == D_LEFT, 0, 1).astype(I32)
    s_item = jnp.clip(pitem, 0, 1).astype(I32)
    fresh_ok = t >= last_del_t + FRESH_DELAY
    valid = jnp.stack([
        stable_pos & fresh_ok,
        stable_pos & fresh_ok & same_face & ((pface == D_LEFT) | (pface == D_RIGHT)),
        stable_pos & fresh_ok & same_item,
    ])
    s_vals = jnp.stack([s_col, s_face, s_item])
    events = valid & ((pend_ch == 1) | (s_vals != m['last_read']))
    lik0 = _channel_lik(s_vals[0], m['last_read'][0])
    lik1 = _channel_lik(s_vals[1], m['last_read'][1])
    lik2 = _channel_lik(s_vals[2], m['last_read'][2])
    one = jnp.ones_like(lik0)
    l0 = jnp.where(events[0], lik0, one)[:, None, None, :]
    l1 = jnp.where(events[1], lik1, one)[None, :, None, :]
    l2 = jnp.where(events[2], lik2, one)[None, None, :, :]
    post = post * l0 * l1 * l2
    post = post / jnp.maximum(post.sum(), 1e-6)
    read_event = events.any()
    last_read = jnp.where(events, s_vals, m['last_read']).astype(I32)
    pend_ch = jnp.where(events, 0, pend_ch).astype(I32)
    nread = m['nread'] + read_event.astype(I32)
    pending = jnp.where(read_event, 0, pending)

    pR = post.sum(axis=(0, 1, 2))                 # P(recipe == 0), P(recipe == 1)
    argmax_r = jnp.argmax(pR).astype(I32)
    maxp = pR[argmax_r]

    # ---- mode machine (plate offer when the partner has never signalled)
    any_ready = potL['ready'] | potR['ready'] | c01['dish'] | c04['dish'] | hd
    # 'looked' counts steps spent at a vantage cell (row 2, col 2/3: whole row 4
    # visible) with a dish available and no usable reading; forced decisions
    # only happen after actually looking for the partner.
    at_vantage = (sr == 2) & ((sc == 2) | (sc == 3))
    raw_conf = (maxp >= CONF_THRESH) & (pending == 0)
    looked = jnp.where(read_event | delivered, 0,
                       jnp.where(at_vantage & any_ready & (~raw_conf), m['wait_unc'] + 1, m['wait_unc']))
    settled = (pending == 0) | (looked >= LOOK_STEPS)
    confident = (maxp >= CONF_THRESH) & settled
    wait_unc = looked
    wait_t = jnp.where(nread > 0, 0, jnp.where(any_ready, m['wait_t'] + 1, m['wait_t']))
    mode = m['mode']
    stage = m['stage']
    offer_t = m['offer_t']
    mode = jnp.where(read_event & (mode == 1), 0, mode)
    # the partner answered a plate offer with a dish we just delivered: offer again
    reoffer = delivered & (m['trusted'] == 1)
    start_offer = ((mode == 0) & (nread == 0) & (wait_t > WAIT_BEFORE_OFFER)) | reoffer
    mode = jnp.where(start_offer, 1, mode)
    stage = jnp.where(start_offer, 0, stage)
    stage = jnp.where(placed_offer, 1, stage)
    offer_t = jnp.where(placed_offer, t, offer_t)
    plate_on_counter = c32['plate_only'] | c34['plate_only']
    timeout = jnp.where(plate_on_counter, OFFER_TIMEOUT, OFFER_TAKEN_TIMEOUT)
    offer_done = (mode == 1) & (stage == 1) & (t - offer_t > timeout)
    mode = jnp.where(offer_done, 2, mode)
    offering = (mode == 1) & (stage == 1)

    forced = (~offering) & (looked >= LOOK_STEPS)
    b_eff = jnp.where(confident | forced, argmax_r, -1).astype(I32)
    known = b_eff >= 0
    bk = jnp.clip(b_eff, 0, 1)
    potB_ready = jnp.where(bk == 0, potL['ready'], potR['ready'])
    potB_timer = jnp.where(bk == 0, potL['timer'], potR['timer'])
    potB_pure = jnp.where(bk == 0, potL['pure0'], potR['pure1'])
    potB_dtype = jnp.where(bk == 0, potL['dtype'], potR['dtype'])
    T_POTB = jnp.where(bk == 0, T_POTL, T_POTR).astype(I32)
    slotB = jnp.where(bk == 0, c01['dish'] & (c01['dtype'] == 0), c04['dish'] & (c04['dtype'] == 1))
    T_SLOTB = jnp.where(bk == 0, T_C01, T_C04).astype(I32)
    # pot to fill first: the more likely recipe once settled; while the
    # post-delivery reading is pending, refill the pot we just emptied (its
    # stand cell also gives a view of the partner on row 4).
    last_del_type = jnp.where(delivered & (del_type >= 0), del_type, m['last_del_type'])
    ak = jnp.where(settled | (last_del_type < 0), argmax_r, jnp.clip(last_del_type, 0, 1)).astype(I32)
    potA_accepts = jnp.where(ak == 0, _pot_accepts(potL, 0), _pot_accepts(potR, 1))
    T_GETA = jnp.where(ak == 0, T_GET0, T_GET1).astype(I32)

    def dump_task():
        return jnp.where(c01['empty'], T_C01, jnp.where(c04['empty'], T_C04, jnp.where(c34['empty'], T_C34, T_WAIT))).astype(I32)

    # store slot for held dish type
    dk = jnp.clip(dtype, 0, 1)
    slot_empty = jnp.where(dk == 0, c01['empty'], c04['empty'])
    T_STORE = jnp.where(slot_empty, jnp.where(dk == 0, T_C01, T_C04), T_WAIT).astype(I32)

    # counter dishes acceptable to pick up (partner-made unless we are confident otherwise)
    def ok_dish(c):
        return c['dish'] & (c['dtype'] >= 0) & ((~confident) | (c['dtype'] == argmax_r))
    c32_ok = ok_dish(c32)
    c34_ok = ok_dish(c34)

    raw_ok = hraw & (rawk <= 1)
    raw_accept = jnp.where(rawk == 0, _pot_accepts(potL, 0), _pot_accepts(potR, 1))
    T_POTRAW = jnp.where(rawk == 0, T_POTL, T_POTR).astype(I32)

    conds = [
        hd & (dtype < 0),
        hd & ((trusted == 1) | (known & (b_eff == dtype))),
        hd & known & (b_eff != dtype),
        hd,
        hpl & known & potB_ready & (potB_dtype == bk),
        hpl & (mode == 1) & (stage == 0),
        hpl,
        raw_ok & raw_accept,
        hraw,
        he & known & slotB,
        he & c32_ok,
        he & c34_ok,
        he & known & ((potB_ready & (potB_dtype == bk)) | ((potB_timer > 0) & (potB_timer <= 8) & potB_pure)),
        he & (mode == 1) & (stage == 0) & (~plate_on_counter),
        he & (mode == 2) & c32['plate_only'],
        he & (mode == 2) & c34['plate_only'],
        he & potA_accepts,
        he & _pot_accepts(potL, 0),
        he & _pot_accepts(potR, 1),
    ]
    choices = [
        dump_task(),
        _i(T_DELIVER),
        T_STORE,
        _i(T_WAIT),
        T_POTB,
        jnp.where(c32['empty'], T_C32, jnp.where(c34['empty'], T_C34, T_WAIT)).astype(I32),
        _i(T_WAIT),
        T_POTRAW,
        dump_task(),
        T_SLOTB,
        _i(T_C32),
        _i(T_C34),
        _i(T_PLATE),
        _i(T_PLATE),
        _i(T_C32),
        _i(T_C34),
        T_GETA,
        _i(T_GET0),
        _i(T_GET1),
    ]
    task = jnp.select(conds, choices, _i(T_WAIT)).astype(I32)
    row = _tasks0()[task]
    tdir = jnp.where(task == T_WAIT, facing, row[2])
    action = _nav(sr, sc, facing, row[0], row[1], tdir, row[3])

    new_mem = {
        't': t + 1,
        'post': post.astype(F32),
        'last_read': last_read.astype(I32),
        'pend_ch': pend_ch.astype(I32),
        'nread': nread.astype(I32),
        'pending': pending.astype(I32),
        'p_row': p_row.astype(I32),
        'p_col': p_col.astype(I32),
        'p_face': p_face.astype(I32),
        'p_item': p_item.astype(I32),
        'p_stable': p_stable.astype(I32),
        'col_min': col_min.astype(I32),
        'col_max': col_max.astype(I32),
        'last_del_t': last_del_t.astype(I32),
        'last_del_type': last_del_type.astype(I32),
        'mode': mode.astype(I32),
        'stage': stage.astype(I32),
        'offer_t': offer_t.astype(I32),
        'wait_unc': wait_unc.astype(I32),
        'wait_t': wait_t.astype(I32),
        'trusted': trusted.astype(I32),
        'prev_task': task,
        'prev_hd': hd.astype(I32),
        'prev_hpl': hpl.astype(I32),
        'prev_he': he.astype(I32),
        'prev_dtype': dtype.astype(I32),
        'cells': cells,
    }
    trace = {
        'task': task,
        'belief': b_eff.astype(I32),
        'p1': (pR[1] * 100).astype(I32),
        'plive': (post[0].sum() * 100).astype(I32),
        'pconst': (post[2].sum() * 100).astype(I32),
        'pface_live': (post[:, 0].sum() * 100).astype(I32),
        'pitem_live': (post[:, :, 0].sum() * 100).astype(I32),
        'mode': mode.astype(I32),
        'sr': sr, 'sc': sc,
    }
    return action, new_mem, trace


# ================================================================ agent 1
def _cells1():
    return jnp.array([[3, 1], [3, 3], [3, 2], [3, 4]], dtype=I32)


B_SIG0, B_SIG1, B_POTL, B_POTR, B_C32, B_C34 = 0, 1, 2, 3, 4, 5


def _tasks1():
    return jnp.array([
        [4, 1, D_UP, 0],  # signal recipe 0 (under left pot, facing it)
        [4, 3, D_UP, 0],  # signal recipe 1 (under right pot, facing it)
        [4, 1, D_UP, 1],  # plate left pot
        [4, 3, D_UP, 1],  # plate right pot
        [4, 2, D_UP, 1],  # counter (3,2)
        [4, 4, D_UP, 1],  # counter (3,4)
    ], dtype=I32)


HOLD_TIMEOUT = 60


def init_agent_1():
    return {
        't': _i(0),
        'recipe': _i(0),
        'hold_t': _i(0),
        'cells': jnp.zeros((4, 7), dtype=I32),
    }


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    m = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(I32)
    t = m['t']

    sr, sc, facing, held, pvis, pr, pc = _decode(obs, 25, 6, 2)
    hd, hpl, hraw, he, rawk, dtype = _held_info(held)
    ind, ind_vis = _cell(obs, 6, 2, sr, sc)
    recipe = jnp.where(ind_vis & (ind[40] == 3), 0,
                       jnp.where(ind_vis & (ind[41] == 3), 1, m['recipe'])).astype(I32)
    k = recipe
    cells = _update_cells(obs, sr, sc, _cells1(), m['cells'])
    potL = _block_info(cells[0])
    potR = _block_info(cells[1])
    c32 = _block_info(cells[2])
    c34 = _block_info(cells[3])

    potK_ready = jnp.where(k == 0, potL['ready'], potR['ready'])
    potK_dtype = jnp.where(k == 0, potL['dtype'], potR['dtype'])
    potK_timer = jnp.where(k == 0, potL['timer'], potR['timer'])
    potK_pure = jnp.where(k == 0, potL['pure0'], potR['pure1'])
    potK_good = potK_ready & (potK_dtype == k)
    potK_cooking = (potK_timer > 0) & potK_pure
    T_SIG = jnp.where(k == 0, B_SIG0, B_SIG1).astype(I32)
    T_POTK = jnp.where(k == 0, B_POTL, B_POTR).astype(I32)
    T_PUT = jnp.where(c32['empty'], B_C32, jnp.where(c34['empty'], B_C34, T_SIG)).astype(I32)
    T_TAKE = jnp.where(c32['plate_only'], B_C32, B_C34).astype(I32)

    hold_t = jnp.where(hpl & (~potK_good) & (~potK_cooking), m['hold_t'] + 1, 0)

    conds = [
        hd,
        hpl & potK_good,
        hpl & (hold_t > HOLD_TIMEOUT),
        hpl,
        he & (c32['plate_only'] | c34['plate_only']) & (potK_good | potK_cooking),
    ]
    choices = [
        T_PUT,
        T_POTK,
        T_PUT,
        T_SIG,
        T_TAKE,
    ]
    task = jnp.select(conds, choices, T_SIG).astype(I32)
    row = _tasks1()[task]
    action = _nav(sr, sc, facing, row[0], row[1], row[2], row[3])

    new_mem = {
        't': t + 1,
        'recipe': recipe,
        'hold_t': hold_t.astype(I32),
        'cells': cells,
    }
    trace = {'task': task, 'recipe': recipe, 'sr': sr, 'sc': sc}
    return action, new_mem, trace
