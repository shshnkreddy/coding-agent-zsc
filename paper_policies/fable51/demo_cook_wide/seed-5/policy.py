"""Overcooked V2 demo_cook_wide: observation-driven role state machines.

Seat 0 (top room, row 1): can reach plates, delivery, ingredient 0/1 supplies and
the pot from above, but never sees the recipe indicator.  It infers the recipe
from pot contents (or a partner visibly carrying an ingredient), tops up the
pot, plates the cooked dish and delivers.

Seat 1 (bottom room, rows 3-4): sees the recipe indicator and the pot, can
insert into the pot from below and fetch ingredients from the far corners.  It
inserts the required ingredient, keeps filling while the pot is open, and while
the pot is busy pre-stashes one unit of each recipe type on nearby counters so
the next recipe can be started quickly after delivery.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'role_state_machines', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3

MEM_LEN = 16
H, W = 6, 11

_ASCII = [
    "WWWWBXBWWWW",
    "WWW0 A 1WWW",
    "WWWWWPWWWWW",
    "W    A    W",
    "0  W3R3W  0",
    "W1WWWWWWW1W",
]

CELLS_0 = [(1, 4), (1, 5), (1, 6)]
CELLS_1 = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
           (4, 1), (4, 2), (4, 8), (4, 9)]

# Counters usable by seat 1: (counter cell, access cell, facing).  Preference:
# keep (2,4)/(2,6) (the only counters seat 0 can reach) as last resort.
COUNTERS_1 = [
    ((2, 3), (3, 3), F_UP),
    ((2, 7), (3, 7), F_UP),
    ((4, 3), (3, 3), F_DOWN),
    ((4, 7), (3, 7), F_DOWN),
    ((2, 4), (3, 4), F_UP),
    ((2, 6), (3, 6), F_UP),
]
C1_PEN_LIST = [0, 0, 0, 0, 2, 2]


def _static_value(r, c):
    # channel order 22..31: wall, goal, pot, recipe, button, plate, ing0..ing3
    v = [0] * 10
    if 0 <= r < H and 0 <= c < W:
        ch = _ASCII[r][c]
        if ch == 'W':
            v[0] = 1
        elif ch == 'X':
            v[1] = 1
        elif ch == 'P':
            v[2] = 1
        elif ch == 'R':
            v[3] = 1
        elif ch == 'L':
            v[4] = 1
        elif ch == 'B':
            v[5] = 1
        elif ch in '0123':
            v[6 + int(ch)] = 1
    return v


def _templates(cells):
    out = [[[_static_value(r + i - 2, c + j - 2) for j in range(5)] for i in range(5)]
           for (r, c) in cells]
    return jnp.asarray(out, dtype=jnp.int32)


def _face_to_action():
    return jnp.asarray([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)


def _cells_arrays(cells):
    return (jnp.asarray([r for r, _ in cells], dtype=jnp.int32),
            jnp.asarray([c for _, c in cells], dtype=jnp.int32))


def _counter_arrays():
    ar = jnp.asarray([a[0] for _, a, _ in COUNTERS_1], dtype=jnp.int32)
    ac = jnp.asarray([a[1] for _, a, _ in COUNTERS_1], dtype=jnp.int32)
    fd = jnp.asarray([f for _, _, f in COUNTERS_1], dtype=jnp.int32)
    pen = jnp.asarray(C1_PEN_LIST, dtype=jnp.int32)
    return ar, ac, fd, pen


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _localize(obs, templ, cells_r, cells_c):
    static = obs[:, :, 22:32]
    diff = jnp.sum(jnp.abs(templ - static[None]), axis=(1, 2, 3))
    k = jnp.argmin(diff)
    return cells_r[k], cells_c[k]


def _cell(obs, r, c, gr, gc):
    """Channel vector of global cell (gr, gc) as seen from (r, c); zeros if unseen."""
    li = gr - r + 2
    lj = gc - c + 2
    vis = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    vec = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4)] * vis.astype(obs.dtype)
    return vec, vis


def _item(block):
    """Decode a 6-entry item block -> dict of scalars."""
    plate = block[0] > 0
    cooked = block[1] > 0
    counts = block[2:6]
    total = jnp.sum(counts)
    present = counts > 0
    ntypes = jnp.sum(present)
    pure = jnp.where(ntypes == 1, jnp.argmax(counts), -1).astype(jnp.int32)
    spoiled = (ntypes >= 2) | present[2] | present[3]
    has = plate | cooked | (total > 0)
    raw = has & ~plate & ~cooked
    return dict(plate=plate, cooked=cooked, counts=counts, total=total,
                ntypes=ntypes, pure=pure, spoiled=spoiled, has=has, raw=raw)


def _counter_type(block):
    """-1 empty, 0..3 single raw ingredient unit type, 9 anything else."""
    it = _item(block)
    single = it['raw'] & (it['ntypes'] == 1) & (it['total'] == 1)
    return jnp.where(~it['has'], -1, jnp.where(single, it['pure'], 9)).astype(jnp.int32)


def _pot(block, timer):
    it = _item(block)
    cooking = timer > 0
    cooked = it['cooked']
    idle = ~cooking & ~cooked
    full = it['total'] >= 3
    return dict(total=it['total'], pure=it['pure'], spoiled=it['spoiled'],
                cooking=cooking, cooked=cooked, idle=idle, full=full,
                open_empty=idle & (it['total'] == 0))


def _pot_open_for(p, t):
    return p['idle'] & ~p['full'] & ((p['total'] == 0) | (~p['spoiled'] & (p['pure'] == t)))


def _self(obs):
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    held = _item(obs[2, 2, 5:11])
    return facing, held


def _partner(obs):
    pos = obs[:, :, 11]
    vis = jnp.any(pos > 0)
    idx = jnp.argmax(pos.reshape(-1))
    pi, pj = idx // 5, idx % 5
    vec = obs[pi, pj]
    held = _item(vec[16:22])
    facing = jnp.argmax(vec[12:16]).astype(jnp.int32)
    return vis, pi - 2, pj - 2, facing, held


def _sel(cond, a, b):
    """Select between two goal tuples (tr, tc, fd, interact)."""
    return tuple(jnp.where(cond, x, y) for x, y in zip(a, b))


def _goal(tr, tc, fd, inter):
    return (_i32(tr), _i32(tc), _i32(fd), _i32(inter))


# ----------------------------------------------------------------------------
# Seat 0: top room
# ----------------------------------------------------------------------------
def init_agent_0():
    mem = jnp.zeros((MEM_LEN,), dtype=jnp.int32)
    mem = mem.at[0].set(-1)      # known recipe type K
    mem = mem.at[1].set(100)     # steps since last delivery
    # 2 deliver_flag, 3 prev_held_dish, 4 suspect, 5 suspect_age, 6 flush,
    # 7 prev_pot_total, 8 seen_partner_empty_since_delivery, 9 wait_count
    return mem


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_0(), memory)
    K = mem[0]
    ssd = mem[1]
    deliver_flag = mem[2]
    prev_dish = mem[3] > 0
    suspect = mem[4] > 0
    suspect_age = mem[5]
    flush = mem[6] > 0
    prev_total = mem[7]
    seen_empty = mem[8] > 0
    wait_count = mem[9]
    my_insert_prev = mem[10]
    suspect_type = mem[11]
    step = mem[13]

    FACE_TO_ACTION = _face_to_action()
    cr, cc = _cells_arrays(CELLS_0)
    r, c = _localize(obs, _templates(CELLS_0), cr, cc)
    facing, held = _self(obs)
    h_empty = ~held['has']
    h_dish = held['cooked']
    h_plate = held['plate'] & ~held['cooked']
    h_ing = held['raw']
    h_type = held['pure']

    pot_vec, _ = _cell(obs, r, c, 2, 5)
    p = _pot(pot_vec[32:38], pot_vec[44])

    pv, pdr, pdc, pfacing, pheld = _partner(obs)
    partner_ing = pv & pheld['raw'] & (pheld['ntypes'] == 1) & (pheld['pure'] <= 1)
    partner_type = pheld['pure']
    partner_empty = pv & ~pheld['has']
    partner_near_pot = pv & (pdr == 2) & (jnp.abs(c + pdc - 5) <= 1)

    c4, _ = _cell(obs, r, c, 2, 4)
    c6, _ = _cell(obs, r, c, 2, 6)
    t4 = _counter_type(c4[32:38])
    t6 = _counter_type(c6[32:38])
    free4 = t4 == -1
    free6 = t6 == -1
    d4 = _item(c4[32:38])
    d6 = _item(c6[32:38])
    dish4 = jnp.where(d4['cooked'] & ~d4['spoiled'], d4['pure'], -2)   # parked dish type
    dish6 = jnp.where(d6['cooked'] & ~d6['spoiled'], d6['pure'], -2)

    # --- event detection
    delivered = (deliver_flag == 1) & h_empty
    dish_gone = prev_dish & h_empty
    flush = jnp.where(dish_gone, False, flush)
    K = jnp.where(delivered, -1, K)
    ssd = jnp.where(delivered, 0, jnp.minimum(ssd + 1, 1000))
    seen_empty = jnp.where(delivered, False, seen_empty) | partner_empty
    fresh = (ssd >= 4) | seen_empty

    # contents already in the pot when the recipe was resampled are suspect
    new_suspect = delivered & (p['total'] > 0)
    suspect = jnp.where(new_suspect, True, suspect)
    suspect_type = jnp.where(new_suspect, p['pure'], suspect_type)
    suspect_age = jnp.where(delivered, 0, jnp.minimum(suspect_age + 1, 1000))
    grew = p['total'] > prev_total + my_insert_prev   # partner added something
    same_type = partner_ing & (partner_type == suspect_type)
    diff_type = partner_ing & (partner_type != suspect_type) & fresh
    grew_same = grew & ~p['spoiled'] & (p['pure'] == suspect_type)
    grew_diff = grew & (p['spoiled'] | (p['pure'] != suspect_type))
    resolved_ok = suspect & (grew_same | same_type | (suspect_age > 40))
    resolved_bad = suspect & ~resolved_ok & (diff_type | grew_diff | (p['spoiled'] & ~h_dish))
    flush = flush | resolved_bad
    suspect = suspect & ~resolved_ok & ~resolved_bad

    # --- recipe inference
    pot_pure01 = (p['total'] > 0) & ~p['spoiled'] & (p['pure'] >= 0) & (p['pure'] <= 1)
    K = jnp.where(pot_pure01 & ~suspect & ~flush, p['pure'], K)
    K = jnp.where((K < 0) & ~suspect & ~flush & partner_ing & (p['total'] == 0) & fresh, partner_type, K)

    sp = p['spoiled'] | flush          # pot contents must be flushed
    fill_needed = sp | suspect         # complete the pot with its own type regardless
    fill_type = jnp.where(p['pure'] >= 0, p['pure'], 0)

    # goals: (row, col, facing, interact)
    G_DELIVER = _goal(1, 5, F_UP, 1)
    G_POT = _goal(1, 5, F_DOWN, 1)
    G_WAIT = _goal(1, 5, F_DOWN, 0)
    G_HOLD_UP = _goal(1, 5, F_UP, 0)
    G_PLATE = _sel(c <= 5, _goal(1, 4, F_UP, 1), _goal(1, 6, F_UP, 1))
    G_ING0 = _goal(1, 4, F_LEFT, 1)
    G_ING1 = _goal(1, 6, F_RIGHT, 1)
    G_C4 = _goal(1, 4, F_DOWN, 1)
    G_C6 = _goal(1, 6, F_DOWN, 1)
    any_free = free4 | free6
    G_FREE = _sel(free4 & (~free6 | (c <= 5)), G_C4, G_C6)

    # --- holding a cooked dish
    dish_ok = ~held['spoiled'] & ~flush
    # a wrong-but-pure dish is delivered at -20 only when no counter is free and time remains to recoup;
    # a truly spoiled dish (mixed / unusable ingredient) is never delivered.
    dump_or_hold = _sel(any_free, G_FREE,
                        _sel(~held['spoiled'] & (step < 360), G_DELIVER, G_HOLD_UP))
    g_dish = _sel(suspect & ~held['spoiled'], G_HOLD_UP, _sel(dish_ok, G_DELIVER, dump_or_hold))

    # --- holding a plate
    need_fill = p['idle'] & ~p['full'] & ((K >= 0) | fill_needed)
    g_plate = _sel(p['cooked'], G_POT,
                   _sel(p['cooking'], G_WAIT,
                        _sel(need_fill & any_free, G_FREE, G_WAIT)))

    # --- holding a raw ingredient
    insert_ok = p['idle'] & ~p['full'] & (
        (fill_needed & ((p['total'] == 0) | (p['pure'] == h_type) | p['spoiled']))
        | (~fill_needed & (p['total'] > 0) & (p['pure'] == h_type))
        | (~fill_needed & (p['total'] == 0) & (h_type == K)))
    g_ing = _sel(insert_ok, G_POT, _sel(any_free, G_FREE, G_WAIT))

    # --- empty hands
    completes_cond = partner_ing & (partner_type == K) & (p['total'] == 2) & partner_near_pot
    partner_completes = completes_cond & (wait_count < 4)
    wait_count = jnp.where(completes_cond, jnp.minimum(wait_count + 1, 100), 0)
    tgt = jnp.where(fill_needed, fill_type, K)
    # supplies are cheaper than counters unless already standing at the counter's access cell
    from_c4 = (t4 == tgt) & (c == 4) & (facing != F_LEFT)
    from_c6 = (t6 == tgt) & (c == 6) & (facing != F_RIGHT)
    G_FETCH = _sel(from_c4 & (~from_c6 | (c <= 5)), G_C4,
                   _sel(from_c6, G_C6, _sel(tgt == 0, G_ING0, G_ING1)))
    fill_now = p['idle'] & ~p['full'] & ((K >= 0) | fill_needed) & ~partner_completes
    # a parked dish whose type matches the current recipe can be delivered while the pot cooks
    parked_ok = (K >= 0) & ~suspect & ~flush & p['cooking'] & ~p['spoiled'] & (p['pure'] == K)
    take4 = parked_ok & (dish4 == K)
    take6 = parked_ok & (dish6 == K)
    G_TAKE = _sel(take4 & (~take6 | (c <= 5)), G_C4, G_C6)
    g_empty = _sel(take4 | take6, G_TAKE,
                   _sel(p['cooked'] | p['cooking'], G_PLATE,
                        _sel(fill_now, G_FETCH, G_WAIT)))

    goal = _sel(h_dish, g_dish, _sel(h_plate, g_plate, _sel(h_ing, g_ing, g_empty)))
    tr, tc, fd, inter = goal

    # navigation on row 1
    move = jnp.where(tc > c, A_RIGHT, A_LEFT)
    turn = FACE_TO_ACTION[fd]
    act_here = jnp.where(facing != fd, turn, jnp.where(inter > 0, A_INTERACT, A_STAY))
    action = jnp.where(c != tc, move, act_here).astype(jnp.int32)

    deliver_next = (action == A_INTERACT) & h_dish & (c == 5) & (facing == F_UP)
    my_insert_next = ((action == A_INTERACT) & h_ing & (c == 5) & (facing == F_DOWN) & p['idle'] & ~p['full']).astype(jnp.int32)
    new_mem = mem.at[0].set(K).at[1].set(ssd).at[2].set(deliver_next.astype(jnp.int32))
    new_mem = new_mem.at[3].set(h_dish.astype(jnp.int32)).at[4].set(suspect.astype(jnp.int32))
    new_mem = new_mem.at[5].set(suspect_age).at[6].set(flush.astype(jnp.int32))
    new_mem = new_mem.at[7].set(p['total']).at[8].set(seen_empty.astype(jnp.int32))
    new_mem = new_mem.at[9].set(wait_count).at[10].set(my_insert_next).at[11].set(suspect_type)
    new_mem = new_mem.at[13].set(jnp.minimum(step + 1, 10000))
    trace = {"K": _i32(K), "col": _i32(c), "goal_col": _i32(tc), "goal_face": _i32(fd),
             "pot_total": _i32(p['total']), "suspect": _i32(suspect), "flush": _i32(flush),
             "held": _i32(jnp.where(h_dish, 3, jnp.where(h_plate, 2, jnp.where(h_ing, 1, 0))))}
    return action.astype(jnp.int32), new_mem, trace


# ----------------------------------------------------------------------------
# Seat 1: bottom room
# ----------------------------------------------------------------------------
def init_agent_1():
    mem = jnp.zeros((MEM_LEN,), dtype=jnp.int32)
    mem = mem.at[0].set(-1)                  # recipe R
    mem = mem.at[1:7].set(-1)                # counter contents (empty)
    mem = mem.at[8].set(-1)                  # pot pure type
    # 12 steps partner has held a dish
    return mem


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_1(), memory)

    FACE_TO_ACTION = _face_to_action()
    C1_AR, C1_AC, C1_FD, C1_PEN = _counter_arrays()
    cr, cc = _cells_arrays(CELLS_1)
    r, c = _localize(obs, _templates(CELLS_1), cr, cc)
    facing, held = _self(obs)
    h_empty = ~held['has']
    h_ing = held['raw'] & (held['ntypes'] == 1)
    h_type = jnp.where(h_ing, held['pure'], 9)

    # recipe
    ind_vec, ind_vis = _cell(obs, r, c, 4, 5)
    R_obs = jnp.where(ind_vec[41] > 0, 1, jnp.where(ind_vec[40] > 0, 0, -1))
    R = jnp.where(ind_vis & (R_obs >= 0), R_obs, mem[0]).astype(jnp.int32)
    other = 1 - R

    # pot (memory when unseen)
    pot_vec, pot_vis = _cell(obs, r, c, 2, 5)
    p_obs = _pot(pot_vec[32:38], pot_vec[44])
    p_total = jnp.where(pot_vis, p_obs['total'], mem[7])
    p_pure = jnp.where(pot_vis, p_obs['pure'], mem[8])
    p_cooking = jnp.where(pot_vis, p_obs['cooking'], mem[9] > 0)
    p_cooked = jnp.where(pot_vis, p_obs['cooked'], mem[10] > 0)
    p_spoiled = jnp.where(pot_vis, p_obs['spoiled'], mem[11] > 0)
    p_idle = ~p_cooking & ~p_cooked
    p_full = p_total >= 3
    p = dict(total=p_total, pure=p_pure, spoiled=p_spoiled, cooking=p_cooking,
             cooked=p_cooked, idle=p_idle, full=p_full)

    # counters
    cm = []
    for k, (cell, _, _) in enumerate(COUNTERS_1):
        vec, vis = _cell(obs, r, c, cell[0], cell[1])
        t = _counter_type(vec[32:38])
        cm.append(jnp.where(vis, t, mem[1 + k]))
    cm = jnp.stack(cm).astype(jnp.int32)
    dist = jnp.abs(C1_AR - r) + jnp.abs(C1_AC - c)
    free = cm == -1
    any_free = jnp.any(free)
    k_free = jnp.argmin(jnp.where(free, dist + C1_PEN, 1000))
    hasR = jnp.any(cm == R)
    hasO = jnp.any(cm == other)
    k_R = jnp.argmin(jnp.where(cm == R, dist, 1000))

    G_INSERT = _goal(3, 5, F_UP, 1)
    G_WAIT = _goal(3, 5, F_UP, 0)
    G_STASH = (C1_AR[k_free], C1_AC[k_free], C1_FD[k_free], _i32(1))
    G_PICK = (C1_AR[k_R], C1_AC[k_R], C1_FD[k_R], _i32(1))

    def fetch_goal(t):
        left = c <= 5
        g_left = _sel(t == 0, _goal(4, 1, F_LEFT, 1), _goal(4, 1, F_DOWN, 1))
        g_right = _sel(t == 0, _goal(4, 9, F_RIGHT, 1), _goal(4, 9, F_DOWN, 1))
        return _sel(left, g_left, g_right)

    # partner holding a dish: the recipe may be about to change, hold off inserting
    pv, _, _, _, pheld = _partner(obs)
    dish_steps = jnp.where(pv & pheld['cooked'], mem[12] + 1, 0)
    hold_off = (dish_steps > 0) & (dish_steps <= 16)

    # pot idle with contents of a wrong or spoiled type: cannot be completed for R
    wrong_pot = p_idle & ~p_full & (p_total > 0) & ((p_pure != R) | p_spoiled)

    # holding an ingredient
    insert_ok = (R >= 0) & (h_type == R) & _pot_open_for(p, h_type) & ~hold_off
    hold_here = (R >= 0) & (h_type == R) & (wrong_pot | hold_off)
    g_hold = _sel(insert_ok, G_INSERT, _sel(hold_here, G_WAIT, _sel(any_free, G_STASH, G_WAIT)))

    # empty hands
    pot_open_R = _pot_open_for(p, R)
    g_open = _sel(hasR, G_PICK, fetch_goal(R))
    g_busy = _sel(~hasR, fetch_goal(R), _sel(~hasO, fetch_goal(other), G_WAIT))
    g_empty = _sel(R < 0, G_WAIT, _sel(pot_open_R | wrong_pot, g_open, g_busy))

    goal = _sel(h_empty, g_empty, g_hold)
    tr, tc, fd, inter = goal

    # navigation in the bottom room
    at = (r == tr) & (c == tc)
    horiz = jnp.where(tc > c, A_RIGHT, A_LEFT)
    move_row3 = jnp.where(c != tc, horiz, A_DOWN)
    move_row4 = jnp.where((tr == 4) & (jnp.abs(tc - c) == 1), horiz, A_UP)
    move = jnp.where(r == 3, move_row3, move_row4)
    turn = FACE_TO_ACTION[fd]
    act_here = jnp.where(facing != fd, turn, jnp.where(inter > 0, A_INTERACT, A_STAY))
    action = jnp.where(at, act_here, move).astype(jnp.int32)

    new_mem = mem.at[0].set(R)
    new_mem = new_mem.at[1:7].set(cm)
    new_mem = new_mem.at[7].set(p_total).at[8].set(p_pure)
    new_mem = new_mem.at[9].set(p_cooking.astype(jnp.int32)).at[10].set(p_cooked.astype(jnp.int32))
    new_mem = new_mem.at[11].set(p_spoiled.astype(jnp.int32)).at[12].set(dish_steps)
    trace = {"R": _i32(R), "row": _i32(r), "col": _i32(c), "goal_row": _i32(tr), "goal_col": _i32(tc),
             "held": _i32(h_type), "pot_total": _i32(p_total)}
    return action, new_mem, trace
