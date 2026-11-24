"""Overcooked V2 demo_cook_simple: hand-coded cooperative policy.

Layout facts (rows x cols = 5 x 11):
  agent_0 lives in the left room (rows 1-3, cols 1-6), sees the recipe
  indicator from cols 3-6, and can insert ingredients into the pot (2,7)
  from (2,6).  Supplies: ingredient 0 at (1,0), ingredient 1 at (3,0).
  agent_1 lives in the right room (rows 1-3, cols 8-9), never sees the
  recipe indicator, but has local supplies (ingredient 0 at (0,8),
  ingredient 1 at (4,8)), plates at (1,10)/(3,10) and delivery at (2,10).
Protocol grounded in observable evidence:
  agent_0 inserts the recipe ingredient into the pot; agent_1 reads the
  pot contents, fills the remainder, plates and delivers.  Counters are
  used as an ingredient stash by agent_0 and for leftovers by agent_1.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'recipe_relay_stash', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

# ---------------------------------------------------------------- constants
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3

PENDING_MAX = 9      # steps after a scoop before trusting the displayed recipe
RESTOCK_MIN_TIMER = 8
CUE_N = 3            # consecutive stationary steps partner must hold an ingredient
COUNTER_N = 25       # pot-empty steps before a lone shared-counter ingredient is used
CUE_SEQ_MAX = 2      # consecutive same-type cue fills before flipping type

_GRID = ["WWWWWR2W0WW", "0      W  B", "W     APA X", "1      W  B", "WWWWWR2W1WW"]
_CH = {"W": 0, "X": 1, "P": 2, "R": 3, "L": 4, "B": 5, "0": 6, "1": 7, "2": 8}

# cells of each room (localization candidates)
CELLS0 = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
          (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
          (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]
CELLS1 = [(1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9)]

# agent 0 tables: stash counters (cell, stand cell, facing); first two are shared with the right room
STASH_CELL_L = [(1, 7), (3, 7), (0, 4), (4, 4), (0, 3), (4, 3), (2, 0), (0, 2), (4, 2)]
STASH_STAND_L = [(1, 6), (3, 6), (1, 4), (3, 4), (1, 3), (3, 3), (2, 1), (1, 2), (3, 2)]
STASH_FACE_L = [D_RIGHT, D_RIGHT, D_UP, D_DOWN, D_UP, D_DOWN, D_LEFT, D_UP, D_DOWN]
STASH_SHARED_L = [True, True, False, False, False, False, False, False, False]
STASH_BY_TYPE_L = [[0, 2, 4], [1, 3, 5], [6, 7, 8]]
SUP0_STAND_L = [(1, 1), (3, 1), (1, 6)]
SUP0_FACE_L = [D_LEFT, D_LEFT, D_UP]
POT0_STAND_L = (2, 6)
WAIT0_STAND_L = (2, 5)

# agent 1 tables: four counters reachable from the right room
C1_CELL_L = [(1, 7), (3, 7), (0, 9), (4, 9)]
C1_STAND_L = [(1, 8), (3, 8), (1, 9), (3, 9)]
C1_FACE_L = [D_LEFT, D_LEFT, D_UP, D_DOWN]
C1_PREF_L = [[2, 0, 3, 1], [3, 1, 2, 0], [2, 3, 0, 1]]   # private counters first
DUMP_PREF_L = [2, 3, 0, 1]
SUP1_STAND_L = [(1, 8), (3, 8), (2, 8)]
SUP1_FACE_L = [D_UP, D_DOWN, D_LEFT]
SUP1_EXISTS_L = [True, True, False]
POT1_STAND_L = (2, 8)
DELIV_STAND_L = (2, 9)
PLATE_STAND_L = [(1, 9), (3, 9)]


def _build_static():
    static = [[[0] * 9 for _ in range(11)] for _ in range(5)]
    for r, row in enumerate(_GRID):
        for c, ch in enumerate(row):
            if ch in _CH:
                static[r][c][_CH[ch]] = 1
    return static


def _templates(cells):
    static = _build_static()
    out = []
    for (r, c) in cells:
        crop = []
        for i in range(5):
            rowl = []
            for j in range(5):
                gr, gc = r + i - 2, c + j - 2
                if 0 <= gr < 5 and 0 <= gc < 11:
                    rowl.append(list(static[gr][gc]))
                else:
                    rowl.append([0] * 9)
            crop.append(rowl)
        out.append(crop)
    return jnp.array(out, dtype=jnp.int32), jnp.array(cells, dtype=jnp.int32)


def _consts():
    """Build all constant tables (called inside traced functions)."""
    a = lambda x: jnp.array(x, dtype=jnp.int32)
    t0, c0 = _templates(CELLS0)
    t1, c1 = _templates(CELLS1)
    return dict(
        TEMPL0=t0, CELLARR0=c0, TEMPL1=t1, CELLARR1=c1,
        STASH_CELL=a(STASH_CELL_L), STASH_STAND=a(STASH_STAND_L), STASH_FACE=a(STASH_FACE_L),
        STASH_SHARED=jnp.array(STASH_SHARED_L), STASH_BY_TYPE=a(STASH_BY_TYPE_L),
        SUP0_STAND=a(SUP0_STAND_L), SUP0_FACE=a(SUP0_FACE_L),
        POT0_STAND=a(POT0_STAND_L), WAIT0_STAND=a(WAIT0_STAND_L),
        C1_CELL=a(C1_CELL_L), C1_STAND=a(C1_STAND_L), C1_FACE=a(C1_FACE_L),
        C1_PREF=a(C1_PREF_L), DUMP_PREF=a(DUMP_PREF_L),
        SUP1_STAND=a(SUP1_STAND_L), SUP1_FACE=a(SUP1_FACE_L),
        SUP1_EXISTS=jnp.array(SUP1_EXISTS_L),
        POT1_STAND=a(POT1_STAND_L), DELIV_STAND=a(DELIV_STAND_L), PLATE_STAND=a(PLATE_STAND_L),
    )


# ---------------------------------------------------------------- helpers
def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _item_kind(block):
    """-1 empty, 0..2 raw ingredient type, 3 empty plate, 4 cooked dish."""
    plate, cooked = block[0], block[1]
    counts = block[2:5]
    n = counts.sum()
    kind = jnp.where(cooked > 0, 4, jnp.where(plate > 0, 3, jnp.where(n > 0, jnp.argmax(counts), -1)))
    return kind.astype(jnp.int32)


def _decode(obs, templ, cellarr):
    obs = obs.astype(jnp.int32)
    static = obs[:, :, 20:29]
    score = (static[None] == templ).sum(axis=(1, 2, 3))
    pos = cellarr[jnp.argmax(score)]
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    self_block = obs[2, 2, 5:10]
    held = _item_kind(self_block)
    pmask = obs[:, :, 10]
    pvis = pmask.sum() > 0
    pblock = (obs[:, :, 15:20] * pmask[:, :, None]).sum(axis=(0, 1))
    pheld = jnp.where(pvis, _item_kind(pblock), -1)
    potmask = obs[:, :, 22]
    pot_vis = potmask.sum() > 0
    pot_block = (obs[:, :, 29:34] * potmask[:, :, None]).sum(axis=(0, 1))
    pot_timer = (obs[:, :, 39] * potmask).sum()
    counts = pot_block[2:5]
    n = counts.sum()
    cooked = pot_block[1] > 0
    cooking = pot_timer > 0
    rmask = obs[:, :, 23]
    rec = (obs[:, :, 36:39] * rmask[:, :, None]).max(axis=(0, 1))
    rvis = (rmask.sum() > 0) & (rec.sum() > 0)
    rtype = jnp.argmax(rec).astype(jnp.int32)
    return dict(
        obs=obs, pos=pos, facing=facing, held=held, self_block=self_block,
        pvis=pvis, pheld=pheld,
        pot_vis=pot_vis, pot_n=n, pot_cooked=cooked, pot_cooking=cooking,
        pot_empty=pot_vis & (n == 0) & ~cooked & ~cooking,
        pot_idle=pot_vis & (n > 0) & ~cooked & ~cooking,
        pot_pure=counts.max() == n, pot_type=jnp.argmax(counts).astype(jnp.int32),
        pot_timer=pot_timer, rvis=rvis, rtype=rtype,
    )


def _cell_kind(obs, pos, cell):
    """Kind of the item on a global cell; -2 when the cell is not in view."""
    lr = cell[0] - pos[0] + 2
    lc = cell[1] - pos[1] + 2
    vis = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    block = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), 29:34]
    return vis, jnp.where(vis, _item_kind(block), -2)


def _move_toward(pos, stand):
    dr = stand[0] - pos[0]
    dc = stand[1] - pos[1]
    vert = jnp.where(dr > 0, A_DOWN, A_UP)
    horz = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    return _i32(jnp.where(dr != 0, vert, horz))


def _goto(pos, facing, stand, face, interact):
    at = (pos[0] == stand[0]) & (pos[1] == stand[1])
    turn = jnp.array([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)[face]
    act = jnp.where(~at, _move_toward(pos, stand),
                    jnp.where(facing != face, turn, jnp.where(interact, A_INTERACT, A_STAY)))
    return _i32(act)


def _dist(pos, stand):
    return jnp.abs(stand[0] - pos[0]) + jnp.abs(stand[1] - pos[1])


# ---------------------------------------------------------------- agent 0
def init_agent_0():
    return {
        'recipe': _i32(-1),     # last seen recipe type
        'pot_last': _i32(0),    # 0 empty, 1 idle nonempty, 2 cooking, 3 cooked
        'pending': _i32(-1),    # -1 none, else steps since pot was seen scooped
        'task': _i32(-1),       # -1 none, else ingredient type to fetch from supply
        'stash': jnp.full((9,), -1, dtype=jnp.int32),  # last seen kinds on stash counters
        'shared_ok': _i32(1),   # 0 once the partner is seen taking from shared counters
        'prev_held': _i32(-1),
    }


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    C = _consts()
    d = _decode(obs, C['TEMPL0'], C['CELLARR0'])
    o, pos, facing, held = d['obs'], d['pos'], d['facing'], d['held']

    # --- recipe memory
    recipe = jnp.where(d['rvis'], d['rtype'], mem['recipe'])
    recipe_changed = d['rvis'] & (mem['recipe'] >= 0) & (d['rtype'] != mem['recipe'])
    k = jnp.clip(recipe, 0, 2)

    # --- pot memory and delivery-pending guard
    pot_vis = d['pot_vis']
    pot_code = jnp.where(pot_vis,
                         jnp.where(d['pot_empty'], 0, jnp.where(d['pot_idle'], 1, jnp.where(d['pot_cooking'], 2, 3))),
                         mem['pot_last'])
    scooped = pot_vis & d['pot_empty'] & (mem['pot_last'] != 0)
    pending = mem['pending']
    pending = jnp.where(scooped & (pending < 0), 0, pending)
    pending = jnp.where(pending >= 0, pending + 1, pending)
    partner_no_dish = d['pvis'] & (d['pheld'] != 4)
    resolve = (pending >= 0) & (recipe_changed | (partner_no_dish & (pending >= 2)) | (pending >= PENDING_MAX)
                                | (pot_vis & ~d['pot_empty']))
    pending = jnp.where(resolve, -1, pending)
    is_pending = pending >= 0

    # --- stash memory and partner-take detection
    svis, skind = jax.vmap(lambda c: _cell_kind(o, pos, c))(C['STASH_CELL'])
    old = mem['stash']
    just_picked = (held >= 0) & (held <= 2) & (mem['prev_held'] == -1)
    taken = svis & (old >= 0) & (old <= 2) & (skind != old) & ~(just_picked & (held == old))
    shared_ok = jnp.where((taken & C['STASH_SHARED']).any(), 0, mem['shared_ok'])
    stash = jnp.where(svis, skind, old).astype(jnp.int32)
    stash_free = (stash == -1) & ((shared_ok == 1) | ~C['STASH_SHARED'])
    sdist = jax.vmap(lambda s: _dist(pos, s))(C['STASH_STAND'])

    # --- pot usability
    pot_ok = jnp.where(pot_vis,
                       (d['pot_empty'] & ~is_pending) | (d['pot_idle'] & (d['pot_n'] < 3)),
                       (mem['pot_last'] <= 1) & ~is_pending)
    need = jnp.where(pot_vis & d['pot_idle'], d['pot_type'], k)

    def slots(t):
        idx = C['STASH_BY_TYPE'][t]
        return idx, stash[idx], stash_free[idx], sdist[idx]

    # pick a stashed item of the needed type (nearest)
    idx_n, kinds_n, _, dists_n = slots(need)
    has_need = kinds_n == need
    pick_i = idx_n[jnp.argmin(jnp.where(has_need, dists_n, 999))]
    any_pick = has_need.any()
    act_pick = _goto(pos, facing, C['STASH_STAND'][pick_i], C['STASH_FACE'][pick_i], True)

    task = mem['task']
    act_task = _goto(pos, facing, C['SUP0_STAND'][jnp.clip(task, 0, 2)], C['SUP0_FACE'][jnp.clip(task, 0, 2)], True)

    # restock decision while pot busy: keep one of each recipe type in stock
    other = jnp.where(k == 0, 1, 0)
    _, kinds_k, free_k, _ = slots(k)
    _, kinds_o, free_o, _ = slots(other)
    missing_k = ~(kinds_k == k).any() & free_k.any()
    missing_o = ~(kinds_o == other).any() & free_o.any()
    restock_type = jnp.where(missing_k, k, other)
    can_restock = (missing_k | missing_o) & (k <= 1)
    pot_busy_long = pot_vis & d['pot_cooking'] & (d['pot_timer'] >= RESTOCK_MIN_TIMER)
    do_restock = ~is_pending & ~pot_ok & pot_busy_long & can_restock

    act_wait6 = _goto(pos, facing, C['POT0_STAND'], D_RIGHT, False)
    act_wait5 = _goto(pos, facing, C['WAIT0_STAND'], D_RIGHT, False)
    act_see = _goto(pos, facing, C['WAIT0_STAND'], D_RIGHT, False)
    act_insert = _goto(pos, facing, C['POT0_STAND'], D_RIGHT, True)

    # stash the held item at the first free slot of its type
    ht = jnp.clip(jnp.where((held >= 0) & (held <= 2), held, 2), 0, 2)
    idx_h, _, free_h, _ = slots(ht)
    free_i = idx_h[jnp.argmax(free_h)]
    any_free = free_h.any()
    act_stash = _goto(pos, facing, C['STASH_STAND'][free_i], C['STASH_FACE'][free_i], True)

    empty_hands = held == -1
    fetch_now = pot_ok & (recipe >= 0)
    new_task = jnp.where(fetch_now & ~any_pick, need, jnp.where(do_restock, restock_type, -1))
    task_active = task >= 0
    act_new_task = _goto(pos, facing, C['SUP0_STAND'][jnp.clip(new_task, 0, 2)], C['SUP0_FACE'][jnp.clip(new_task, 0, 2)], True)
    act_empty = jnp.where(recipe < 0, act_see,
                jnp.where(task_active, act_task,
                jnp.where(fetch_now & any_pick, act_pick,
                jnp.where(new_task >= 0, act_new_task, act_wait6))))
    task_next = jnp.where(empty_hands, jnp.where(task_active, task, new_task), -1)

    can_insert = pot_ok & (need == held) & (recipe >= 0)
    act_hold = jnp.where(recipe < 0, act_see,
               jnp.where(can_insert, act_insert,
               jnp.where(any_free, act_stash, act_wait5)))

    action = jnp.where(empty_hands, act_empty, act_hold)

    next_mem = {
        'recipe': _i32(recipe),
        'pot_last': _i32(pot_code),
        'pending': _i32(pending),
        'task': _i32(task_next),
        'stash': stash,
        'shared_ok': _i32(shared_ok),
        'prev_held': _i32(held),
    }
    trace = {
        'pos_r': pos[0], 'pos_c': pos[1], 'held': held, 'recipe': _i32(recipe),
        'pending': _i32(pending), 'pot_code': _i32(pot_code), 'task': _i32(task_next),
        'need': _i32(need), 'shared_ok': _i32(shared_ok), 'action': _i32(action),
    }
    return _i32(action), next_mem, trace


# ---------------------------------------------------------------- agent 1
def init_agent_1():
    return {
        'cue_type': _i32(-1),
        'cue_count': _i32(0),
        'cue_r': _i32(-1),
        'cue_c': _i32(-1),
        'seq_type': _i32(-1),     # type of consecutive cue-driven fills
        'seq_count': _i32(0),
        'reason': _i32(0),        # why we are fetching: 1 pot, 2 cue, 3 cue-flip, 4 counter
        'empty_steps': _i32(0),
        'prev_pot_n': _i32(0),
        'just_inserted': _i32(0),
    }


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    C = _consts()
    d = _decode(obs, C['TEMPL1'], C['CELLARR1'])
    o, pos, facing, held = d['obs'], d['pos'], d['facing'], d['held']
    pot_empty, pot_idle = d['pot_empty'], d['pot_idle']
    pot_busy = d['pot_cooking'] | d['pot_cooked']
    n = d['pot_n']

    # --- partner hand cue: stationary partner holding the same raw type for CUE_N steps
    pflat = jnp.argmax(o[:, :, 10].reshape(-1))
    pgr = pos[0] + pflat // 5 - 2
    pgc = pos[1] + pflat % 5 - 2
    praw = d['pvis'] & (d['pheld'] >= 0) & (d['pheld'] <= 2)
    same = praw & (d['pheld'] == mem['cue_type']) & (pgr == mem['cue_r']) & (pgc == mem['cue_c'])
    cue_count = jnp.where(same, mem['cue_count'] + 1, jnp.where(praw, 1, 0))
    cue_type = jnp.where(praw, d['pheld'], -1)
    k_cue = jnp.where(cue_count >= CUE_N, cue_type, -1)

    # --- partner inserted into the pot -> reset cue-fill sequence
    partner_inserted = (mem['prev_pot_n'] == 0) & (n > 0) & (mem['just_inserted'] == 0)
    seq_type = jnp.where(partner_inserted, -1, mem['seq_type'])
    seq_count = jnp.where(partner_inserted, 0, mem['seq_count'])
    flip = (k_cue >= 0) & (seq_type == k_cue) & (seq_count >= CUE_SEQ_MAX)
    k_cue_eff = jnp.where(flip, jnp.where(k_cue == 0, 1, 0), k_cue)

    # --- counters
    cvis, ckind = jax.vmap(lambda c: _cell_kind(o, pos, c))(C['C1_CELL'])
    cfree = (ckind == -1) | (ckind == -2)     # unseen counters assumed free
    cdist = jax.vmap(lambda s: _dist(pos, s))(C['C1_STAND'])
    shared = ckind[:2]
    raw_present = jnp.array([(shared == 0).any(), (shared == 1).any(), (shared == 2).any()])
    lone = raw_present.sum() == 1
    lone_type = jnp.argmax(raw_present).astype(jnp.int32)
    empty_steps = jnp.where(pot_empty & (k_cue < 0), mem['empty_steps'] + 1, 0)
    k_counter = jnp.where(lone & (empty_steps >= COUNTER_N), lone_type, -1)

    act_wait = _goto(pos, facing, C['POT1_STAND'], D_LEFT, False)
    act_scoop = _goto(pos, facing, C['POT1_STAND'], D_LEFT, True)
    act_insert = act_scoop
    act_deliver = _goto(pos, facing, C['DELIV_STAND'], D_RIGHT, True)
    plate_i = jnp.where(pos[0] <= 2, 0, 1)
    act_plate = _goto(pos, facing, C['PLATE_STAND'][plate_i], D_RIGHT, True)

    def first_free(pref):
        order_free = cfree[pref]
        idx = pref[jnp.argmax(order_free)]
        return order_free.any(), _goto(pos, facing, C['C1_STAND'][idx], C['C1_FACE'][idx], True)

    # --- holding a cooked dish
    sb = d['self_block']
    dish_pure = sb[2:5].max() == sb[2:5].sum()
    any_dump, act_dump = first_free(C['DUMP_PREF'])
    act_dish = jnp.where(dish_pure | ~any_dump, act_deliver, act_dump)

    # --- holding a plate
    act_platehold = jnp.where(d['pot_cooked'], act_scoop,
                    jnp.where(d['pot_cooking'], act_wait,
                    jnp.where(pot_idle & any_dump, act_dump, act_wait)))

    # --- holding a raw ingredient
    ht = jnp.clip(held, 0, 2)
    compat = pot_empty | (pot_idle & (n < 3) & ((d['pot_pure'] & (d['pot_type'] == held)) | ~d['pot_pure']))
    any_sfree, act_stash = first_free(C['C1_PREF'][ht])
    act_raw = jnp.where(compat, act_insert, jnp.where(any_sfree, act_stash, act_wait))

    # --- empty hands
    reason_now = jnp.where(pot_idle & (n < 3), 1,
                 jnp.where(pot_empty & (k_cue >= 0), jnp.where(flip, 3, 2),
                 jnp.where(pot_empty & (k_counter >= 0), 4, 0)))
    need = jnp.where(pot_idle, d['pot_type'], jnp.where(k_cue >= 0, k_cue_eff, k_counter))
    want_fetch = reason_now > 0
    nd = jnp.clip(need, 0, 2)
    has_need = (ckind == need) & (need >= 0)
    pick_idx = jnp.argmin(jnp.where(has_need, cdist, 999))
    dsup = _dist(pos, C['SUP1_STAND'][nd])
    any_pick = has_need.any() & ((cdist[pick_idx] < dsup) | ~C['SUP1_EXISTS'][nd])
    act_pick = _goto(pos, facing, C['C1_STAND'][pick_idx], C['C1_FACE'][pick_idx], True)
    act_supply = _goto(pos, facing, C['SUP1_STAND'][nd], C['SUP1_FACE'][nd], C['SUP1_EXISTS'][nd])
    act_fetch = jnp.where(any_pick, act_pick, act_supply)
    act_empty = jnp.where(pot_busy, act_plate, jnp.where(want_fetch, act_fetch, act_wait))

    action = jnp.where(held == 4, act_dish,
             jnp.where(held == 3, act_platehold,
             jnp.where(held >= 0, act_raw, act_empty)))

    # --- bookkeeping of cue-driven fills
    reason = jnp.where(held == -1, reason_now, mem['reason'])
    at_pot = (pos[0] == C['POT1_STAND'][0]) & (pos[1] == C['POT1_STAND'][1]) & (facing == D_LEFT)
    inserting = (held >= 0) & (held <= 2) & pot_empty & at_pot & (action == A_INTERACT)
    cue_fill = inserting & (reason == 2)
    seq_type = jnp.where(cue_fill, held, jnp.where(inserting, -1, seq_type))
    seq_count = jnp.where(cue_fill, jnp.where(mem['seq_type'] == held, mem['seq_count'] + 1, 1),
                          jnp.where(inserting, 0, seq_count))

    next_mem = {
        'cue_type': _i32(cue_type), 'cue_count': _i32(cue_count),
        'cue_r': _i32(pgr), 'cue_c': _i32(pgc),
        'seq_type': _i32(seq_type), 'seq_count': _i32(seq_count),
        'reason': _i32(reason), 'empty_steps': _i32(empty_steps),
        'prev_pot_n': _i32(n), 'just_inserted': _i32(inserting),
    }
    trace = {
        'pos_r': pos[0], 'pos_c': pos[1], 'held': held, 'pot_n': _i32(n),
        'pot_cooking': d['pot_cooking'], 'pot_cooked': d['pot_cooked'],
        'need': _i32(need), 'reason': _i32(reason), 'seq_count': _i32(seq_count), 'action': _i32(action),
    }
    return _i32(action), next_mem, trace
