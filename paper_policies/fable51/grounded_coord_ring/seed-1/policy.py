"""grounded_coord_ring policy.

Seat roles
----------
agent_1 (inner room, only seat that can reach the pot): fully self-sufficient
cook.  It learns the recipe from the button indicator, loads the pot, and
presses the button again shortly before delivery so the still-active display
reveals the next recipe at no extra travel cost.  It opportunistically uses a
matching ingredient left on the shared counter next to its pickup cell.

agent_0 (outer ring): cannot reach the pot.  It reads the free recipe
indicator and keeps the current recipe ingredient on the two shared counters
(2,3)/(6,3) that the inner seat can reach, removes stale items from those
counters after the recipe changes, and delivers any plated dish the partner
leaves on a shared counter.  This grounds the recipe for partners that rely on
counter evidence while staying harmless for partners that do not.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'ring_cook_signaller', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}

# ---------------------------------------------------------------------------
# Static layout data (literals only at module scope)
# ---------------------------------------------------------------------------
LAYOUT = [
    "WWW2R2WWW",
    "W       W",
    "W WWLWW W",
    "2 0   B 2",
    "RAXAP X R",
    "2 1   B 2",
    "W WWLWW W",
    "W       W",
    "WWW2R2WWW",
]
H = 9
W = 9
# static channel order: wall, goal, pot, recipe_indicator, button, plate_pile, ing0, ing1, ing2
CH = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'L': 4, 'B': 5, '0': 6, '1': 7, '2': 8}

RING_CELLS = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 1), (2, 7), (3, 1), (3, 7),
              (4, 1), (4, 7), (5, 1), (5, 7), (6, 1), (6, 7), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
INNER_CELLS = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5)]

# action ids
A_RIGHT = 0
A_DOWN = 1
A_LEFT = 2
A_UP = 3
A_STAY = 4
A_INTERACT = 5
# facing ids (observation)
F_UP = 0
F_DOWN = 1
F_RIGHT = 2
F_LEFT = 3
TURN_ACTION = [A_UP, A_DOWN, A_RIGHT, A_LEFT]  # facing id -> action that faces it
MOVES = [(0, 1, A_RIGHT), (1, 0, A_DOWN), (0, -1, A_LEFT), (-1, 0, A_UP)]

# Item type codes
IT_NONE = 0
IT_ING = 1
IT_PLATE = 2
IT_DISH = 3
BIG = 10000

# agent_0 tables: shared counters (cell, access cell, facing); first two are stocking counters
S_LIST = [((2, 3), (1, 3), F_DOWN), ((6, 3), (7, 3), F_UP), ((2, 5), (1, 5), F_DOWN), ((6, 5), (7, 5), F_UP)]
K_COUNT = 2
# outer-only dump counters (cell, access cell, facing)
D_LIST = [((0, 1), (1, 1), F_UP), ((0, 2), (1, 2), F_UP), ((0, 6), (1, 6), F_UP), ((0, 7), (1, 7), F_UP),
          ((1, 0), (1, 1), F_LEFT), ((2, 0), (2, 1), F_LEFT), ((6, 0), (6, 1), F_LEFT), ((7, 0), (7, 1), F_LEFT),
          ((8, 1), (7, 1), F_DOWN), ((8, 2), (7, 2), F_DOWN), ((8, 6), (7, 6), F_DOWN), ((8, 7), (7, 7), F_DOWN),
          ((1, 8), (1, 7), F_RIGHT), ((2, 8), (2, 7), F_RIGHT), ((6, 8), (6, 7), F_RIGHT), ((7, 8), (7, 7), F_RIGHT),
          ((2, 2), (1, 2), F_DOWN), ((2, 6), (1, 6), F_DOWN), ((6, 2), (7, 2), F_UP), ((6, 6), (7, 6), F_UP)]
SPARE_TOP = 16  # D_LIST index of (2,2), spare slot for the top watch post
SPARE_BOT = 18  # D_LIST index of (6,2), spare slot for the bottom watch post
SUP_LIST = [((3, 1), F_RIGHT), ((5, 1), F_RIGHT), ((1, 3), F_UP)]  # access for ingredient 0, 1, 2
X_LIST = [((4, 1), F_RIGHT), ((4, 7), F_LEFT)]
NONCORNER = [(1, 2), (1, 6), (2, 1), (2, 7), (6, 1), (6, 7), (7, 2), (7, 6)]
PRESS_AT = 6  # observed pot timer at which agent_1 presses the button before plating
PRESS_LATE_T = 352  # after this step a pre-delivery press cannot serve another full cycle
FIND_LATE_T = 361  # after this step a fresh recipe lookup cannot lead to a delivery


def _build_room(cells):
    """Pure-Python construction of templates and BFS routing for one room."""
    n = len(cells)
    idx = {}
    for i in range(n):
        idx[cells[i]] = i
    static = [[[0] * 9 for _ in range(W + 4)] for _ in range(H + 4)]
    for r in range(H):
        for c in range(W):
            ch = LAYOUT[r][c]
            if ch in CH:
                static[r + 2][c + 2][CH[ch]] = 1
    templ = []
    for (r, c) in cells:
        templ.append([[static[r + i][c + j] for j in range(5)] for i in range(5)])
    dist = [[999] * n for _ in range(n)]
    nxt = [[A_STAY] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        frontier = [s]
        first = {s: A_STAY}
        for _ in range(n):
            new = []
            for u in frontier:
                ur, uc = cells[u]
                for (dr, dc, act) in MOVES:
                    v = idx.get((ur + dr, uc + dc))
                    if v is not None and dist[s][v] == 999:
                        dist[s][v] = dist[s][u] + 1
                        first[v] = act if u == s else first[u]
                        new.append(v)
            frontier = new
        for t in range(n):
            if t != s:
                nxt[s][t] = first[t]
    xy = [[r, c] for (r, c) in cells]
    return {
        'templ': jnp.asarray(templ, dtype=jnp.int32),
        'dist': jnp.asarray(dist, dtype=jnp.int32),
        'next': jnp.asarray(nxt, dtype=jnp.int32),
        'xy': jnp.asarray(xy, dtype=jnp.int32),
        'idx': idx,
    }


def item_type(block):
    """block: [..., 5] = plate, cooked, c0, c1, c2 -> (type, ingredient id)."""
    plate = block[..., 0] > 0
    cooked = block[..., 1] > 0
    counts = block[..., 2:5]
    ing = jnp.argmax(counts, axis=-1)
    has_ing = jnp.sum(counts, axis=-1) > 0
    t = jnp.where(plate & cooked, IT_DISH,
                  jnp.where(plate, IT_PLATE, jnp.where(has_ing | cooked, IT_ING, IT_NONE)))
    return t.astype(jnp.int32), ing.astype(jnp.int32)


def localize(obs, templ):
    static = obs[:, :, 20:29]
    match = jnp.sum((templ == static[None]).astype(jnp.int32), axis=(1, 2, 3))
    return jnp.argmax(match).astype(jnp.int32)


def facing_of(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def cell_block(obs, my_r, my_c, gr, gc, start, width):
    """Channel block [start:start+width] at global (gr, gc); zeros if not in view."""
    li = gr - my_r + 2
    lj = gc - my_c + 2
    vis = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    lic = jnp.clip(li, 0, 4)
    ljc = jnp.clip(lj, 0, 4)
    blk = jax.lax.dynamic_slice(obs, (lic, ljc, start), (1, 1, width))[0, 0]
    return jnp.where(vis, blk, jnp.zeros_like(blk)), vis


def recipe_from_view(obs):
    """Return displayed recipe ingredient id or -1 if no display is in view."""
    counts = obs[:, :, 36:39]
    present = jnp.sum(counts, axis=-1) > 0
    any_present = jnp.any(present)
    flat = counts.reshape(-1, 3)
    pf = present.reshape(-1)
    first = jnp.argmax(pf)
    r = jnp.argmax(flat[first])
    return jnp.where(any_present, r, -1).astype(jnp.int32)


def go_or_act(pos, target, facing, req_face, nxt_table, act_when_there):
    """Move toward target cell index; when there, turn; when facing, do act."""
    at = pos == target
    move = nxt_table[pos, target]
    turn = jnp.asarray(TURN_ACTION, dtype=jnp.int32)[req_face]
    return jnp.where(at, jnp.where(facing == req_face, act_when_there, turn), move).astype(jnp.int32)


# ---------------------------------------------------------------------------
# agent_1: inner cook
# ---------------------------------------------------------------------------
def init_agent_1():
    return {
        'r': jnp.asarray(-1, dtype=jnp.int32),
        'pressed': jnp.asarray(0, dtype=jnp.int32),
        'deliver': jnp.asarray(0, dtype=jnp.int32),
        't': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    room = _build_room(INNER_CELLS)
    NEXT = room['next']
    DIST = room['dist']
    XY = room['xy']
    IDX = room['idx']
    Q_TOP, Q_BOT = IDX[(3, 4)], IDX[(5, 4)]
    P_TOP, P_BOT = IDX[(3, 3)], IDX[(5, 3)]
    B_TOP, B_BOT = IDX[(3, 5)], IDX[(5, 5)]
    D_CELL = IDX[(4, 5)]

    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(jnp.int32)

    pos = localize(obs, room['templ'])
    my_r, my_c = XY[pos, 0], XY[pos, 1]
    facing = facing_of(obs)
    held_t, held_k = item_type(obs[2, 2, 5:10])
    held_none = held_t == IT_NONE

    # pot state (always in view)
    pot_blk, _ = cell_block(obs, my_r, my_c, 4, 4, 29, 5)
    pot_timer, _ = cell_block(obs, my_r, my_c, 4, 4, 39, 1)
    pot_timer = pot_timer[0]
    pot_n = jnp.sum(pot_blk[2:5])
    pot_cooked = pot_blk[1] > 0
    pot_cooking = pot_timer > 0
    pot_idle = (~pot_cooked) & (~pot_cooking)

    # delivery bookkeeping
    t = mem['t']
    delivered = (mem['deliver'] == 1) & held_none
    r = jnp.where(delivered, -1, mem['r'])
    # the pre-delivery press flag only matters while the pot is cooking/cooked
    pressed = jnp.where(delivered | pot_idle, 0, mem['pressed'])
    # late in the episode a press cannot pay for itself: skip it
    press_late = t > PRESS_LATE_T
    find_late = t > FIND_LATE_T
    seen = recipe_from_view(obs)
    r = jnp.where(seen >= 0, seen, r)
    r_known = r >= 0
    side_bot = r == 1
    Q = jnp.where(side_bot, Q_BOT, Q_TOP)
    P = jnp.where(side_bot, P_BOT, P_TOP)
    B = jnp.where(side_bot, B_BOT, B_TOP)
    btn_face_at_q = jnp.where(side_bot, F_DOWN, F_UP)
    k_face_at_p = jnp.where(side_bot, F_DOWN, F_UP)
    k_row = jnp.where(side_bot, 6, 2)

    # counter next to pickup cell
    k_blk, _ = cell_block(obs, my_r, my_c, k_row, 3, 29, 5)
    k_t, k_k = item_type(k_blk)
    # use the partner-stocked counter only when already facing it (no extra turn)
    k_match = (k_t == IT_ING) & (k_k == r) & (k_blk[1] == 0) & (facing == k_face_at_p)

    # --- candidate actions ---
    a_deliver = go_or_act(pos, D_CELL, facing, F_RIGHT, NEXT, A_INTERACT)
    a_take = go_or_act(pos, D_CELL, facing, F_LEFT, NEXT, jnp.where(pot_cooked, A_INTERACT, A_STAY))
    a_plate_dump = go_or_act(pos, B_TOP, facing, F_UP, NEXT, A_INTERACT)
    a_plate = jnp.where(pot_idle, a_plate_dump, a_take)
    held_side_bot = held_k == 1
    Qh = jnp.where(held_side_bot, Q_BOT, Q_TOP)
    pot_face_h = jnp.where(held_side_bot, F_UP, F_DOWN)
    a_insert = go_or_act(pos, Qh, facing, pot_face_h, NEXT, A_INTERACT)
    can_insert = pot_idle & (pot_n < 3) & ((held_k == r) | (~r_known) | (pot_n > 0))
    a_ing = jnp.where(can_insert, a_insert, a_plate_dump)
    nearest_q = jnp.where(DIST[pos, Q_BOT] < DIST[pos, Q_TOP], Q_BOT, Q_TOP)
    nq_face = jnp.where(nearest_q == Q_BOT, F_DOWN, F_UP)
    a_find = go_or_act(pos, nearest_q, facing, nq_face, NEXT, A_INTERACT)
    a_press = go_or_act(pos, Q, facing, btn_face_at_q, NEXT, A_INTERACT)
    a_wait_press = go_or_act(pos, Q, facing, btn_face_at_q, NEXT,
                             jnp.where(pot_timer <= PRESS_AT, A_INTERACT, A_STAY))
    a_getplate = go_or_act(pos, B, facing, F_RIGHT, NEXT, A_INTERACT)
    a_load = go_or_act(pos, P, facing, jnp.where(k_match, k_face_at_p, F_LEFT), NEXT, A_INTERACT)

    skip_press = (pressed == 1) | press_late
    a_empty_known = jnp.where(
        pot_cooked, jnp.where(skip_press, a_getplate, a_press),
        jnp.where(pot_cooking, jnp.where(skip_press, a_getplate, a_wait_press), a_load))
    a_empty = jnp.where(r_known, a_empty_known, jnp.where(find_late, A_STAY, a_find))

    action = jnp.where(held_t == IT_DISH, a_deliver,
                       jnp.where(held_t == IT_PLATE, a_plate,
                                 jnp.where(held_t == IT_ING, a_ing, a_empty)))
    action = action.astype(jnp.int32)

    # memory updates
    at_q_btn = (pos == Q) & (facing == btn_face_at_q)
    at_nq_btn = (pos == nearest_q) & (facing == nq_face)
    pressing = (action == A_INTERACT) & held_none & (at_q_btn | at_nq_btn)
    pressed = jnp.where(pressing, 1, pressed)
    delivering = (action == A_INTERACT) & (held_t == IT_DISH) & (pos == D_CELL) & (facing == F_RIGHT)

    new_mem = {
        'r': r.astype(jnp.int32),
        'pressed': pressed.astype(jnp.int32),
        'deliver': jnp.where(delivering, 1, 0).astype(jnp.int32),
        't': (t + 1).astype(jnp.int32),
    }
    trace = {'r': r, 'pressed': pressed, 'pos': pos, 'pot_n': pot_n, 'pot_timer': pot_timer,
             'held': held_t, 'cooked': pot_cooked}
    return action, new_mem, trace


# ---------------------------------------------------------------------------
# agent_0: outer ring signaller / courier
# ---------------------------------------------------------------------------
def init_agent_0():
    return {
        'r': jnp.asarray(-1, dtype=jnp.int32),
        't': jnp.asarray(0, dtype=jnp.int32),
        'items': jnp.zeros((H, W, 5), dtype=jnp.int32),
        'seen': jnp.zeros((H, W), dtype=jnp.int32),
        # partner activity tracking (for stuck-partner rescue)
        'p_cell': jnp.asarray(-1, dtype=jnp.int32),
        'p_face': jnp.asarray(-1, dtype=jnp.int32),
        'p_held': jnp.asarray(-1, dtype=jnp.int32),
        'p_streak': jnp.asarray(0, dtype=jnp.int32),
        'p_last_t': jnp.asarray(-1000, dtype=jnp.int32),
        'last_rescue_t': jnp.asarray(-1000, dtype=jnp.int32),
        'rescue_fail': jnp.asarray(0, dtype=jnp.int32),
        'post': jnp.asarray(0, dtype=jnp.int32),
        'last_refresh_t': jnp.asarray(-1000, dtype=jnp.int32),
        'last_act': jnp.asarray(4, dtype=jnp.int32),
        'cons_t': jnp.asarray(-1000, dtype=jnp.int32),
        'cons_n': jnp.asarray(0, dtype=jnp.int32),
        'courier': jnp.asarray(0, dtype=jnp.int32),
    }


COURIER_WINDOW = 20   # consecutive partner consumptions of counter items at most this far
COURIER_COUNT = 3     # apart, this many times, mark the partner as courier-dependent
                      # (enables the spare slots)
REFRESH_PERIOD = 8    # at the watch post, lift and replace the counter item this often so
                      # partners that only trust newly appeared items see a fresh signal
IDLE_STEPS = 24       # identical partner observations spanning this many steps count as "stuck"
                      # (longer than a full 20-step cook wait, so cook-waiting partners never trigger)
SIGHT_GAP = 12        # max steps between partner sightings for the idle streak to continue
RESCUE_COOLDOWN = 40  # minimum steps between rescue presses
RESCUE_CHECK = 12     # steps after a rescue press to check whether the partner moved
RESCUE_MAX_FAIL = 2   # give up rescuing after this many ineffective presses
PATROL_AGE = 40       # switch watch post when the other stocking counter is this stale
PARTNER_LOST = 30     # steps without a partner sighting before checking the other side


def _pick(costs, mask):
    c = jnp.where(mask, costs, BIG)
    i = jnp.argmin(c)
    return i, mask[i]


def agent_0_act(obs, reset, memory, rng):
    room = _build_room(RING_CELLS)
    NEXT = room['next']
    DIST = room['dist']
    XY = room['xy']
    IDX = room['idx']
    S_RC = jnp.asarray([[c[0], c[1]] for (c, a, f) in S_LIST], dtype=jnp.int32)
    S_ACC = jnp.asarray([IDX[a] for (c, a, f) in S_LIST], dtype=jnp.int32)
    S_FACE = jnp.asarray([f for (c, a, f) in S_LIST], dtype=jnp.int32)
    D_RC = jnp.asarray([[c[0], c[1]] for (c, a, f) in D_LIST], dtype=jnp.int32)
    D_ACC = jnp.asarray([IDX[a] for (c, a, f) in D_LIST], dtype=jnp.int32)
    D_FACE = jnp.asarray([f for (c, a, f) in D_LIST], dtype=jnp.int32)
    SUP_ACC = jnp.asarray([IDX[a] for (a, f) in SUP_LIST], dtype=jnp.int32)
    SUP_FACE = jnp.asarray([f for (a, f) in SUP_LIST], dtype=jnp.int32)
    X_ACC = jnp.asarray([IDX[a] for (a, f) in X_LIST], dtype=jnp.int32)
    X_FACE = jnp.asarray([f for (a, f) in X_LIST], dtype=jnp.int32)
    NC = jnp.asarray([IDX[a] for a in NONCORNER], dtype=jnp.int32)
    n_s = len(S_LIST)
    n_d = len(D_LIST)

    init = init_agent_0()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    obs = obs.astype(jnp.int32)
    t = mem['t']

    pos = localize(obs, room['templ'])
    my_r, my_c = XY[pos, 0], XY[pos, 1]
    facing = facing_of(obs)
    held_t, held_k = item_type(obs[2, 2, 5:10])
    held_none = held_t == IT_NONE

    # update item memory for visible in-map cells
    rows = jnp.arange(H)[:, None]
    cols = jnp.arange(W)[None, :]
    li = rows - my_r + 2
    lj = cols - my_c + 2
    vis = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    lic = jnp.clip(li, 0, 4)
    ljc = jnp.clip(lj, 0, 4)
    obs_items = obs[lic, ljc, 29:34]
    items = jnp.where(vis[:, :, None], obs_items, mem['items'])
    seen = jnp.where(vis, t, mem['seen'])

    seen_r = recipe_from_view(obs)
    r = jnp.where(seen_r >= 0, seen_r, mem['r'])
    r_known = r >= 0
    r_safe = jnp.clip(r, 0, 2)

    def cost(acc, face):
        return DIST[pos, acc] + jnp.where(facing == face, 0, 1)

    # classify counters from memory
    s_blk = items[S_RC[:, 0], S_RC[:, 1]]
    s_t, s_k = item_type(s_blk)
    # detect the partner consuming a stocked item (counter was good, now seen empty,
    # and I did not just take it myself)
    prev_blk = mem['items'][S_RC[:, 0], S_RC[:, 1]]
    prev_t, prev_k = item_type(prev_blk)
    prev_good = (prev_t == IT_ING) & (prev_k == r) & (prev_blk[:, 1] == 0)
    s_vis = vis[S_RC[:, 0], S_RC[:, 1]]
    mine = (pos == S_ACC) & (facing == S_FACE) & (mem['last_act'] == A_INTERACT)
    consumed = jnp.any(prev_good & s_vis & (s_t == IT_NONE) & (~mine))
    cons_n = jnp.where(consumed,
                       jnp.where(t - mem['cons_t'] <= COURIER_WINDOW, mem['cons_n'] + 1, 1),
                       mem['cons_n'])
    courier = jnp.where(cons_n >= COURIER_COUNT, 1, mem['courier'])
    cons_t = jnp.where(consumed, t, mem['cons_t'])
    d_blk = items[D_RC[:, 0], D_RC[:, 1]]
    d_t, d_k = item_type(d_blk)
    s_cost = cost(S_ACC, S_FACE)
    d_cost = cost(D_ACC, D_FACE)

    dish_i, has_dish = _pick(s_cost, s_t == IT_DISH)
    k_mask = jnp.arange(n_s) < K_COUNT
    k_good = (s_t == IT_ING) & (s_k == r) & (s_blk[:, 1] == 0)
    stale_i, has_stale = _pick(s_cost, k_mask & (s_t != IT_NONE) & (~k_good))
    empty_i, has_empty = _pick(s_cost, k_mask & (s_t == IT_NONE))

    # ---- partner tracking ----
    p_pos_map = obs[:, :, 10]
    p_vis = jnp.any(p_pos_map > 0)
    p_flat = jnp.argmax(p_pos_map.reshape(-1))
    p_li = p_flat // 5
    p_lj = p_flat % 5
    p_gr = my_r + p_li - 2
    p_gc = my_c + p_lj - 2
    p_cell = jnp.where(p_vis, p_gr * W + p_gc, -1).astype(jnp.int32)
    p_face = jnp.where(p_vis, jnp.argmax(obs[p_li, p_lj, 11:15]), -1).astype(jnp.int32)
    p_ht, p_hk = item_type(obs[p_li, p_lj, 15:20])
    p_held = jnp.where(p_vis, p_ht * 4 + p_hk, -1).astype(jnp.int32)
    gap = t - mem['p_last_t']
    same = p_vis & (p_cell == mem['p_cell']) & (p_face == mem['p_face']) & \
        (p_held == mem['p_held']) & (gap <= SIGHT_GAP)
    p_streak = jnp.where(p_vis, jnp.where(same, mem['p_streak'] + gap, 1),
                         jnp.where(gap <= SIGHT_GAP, mem['p_streak'], 0)).astype(jnp.int32)
    p_last_t = jnp.where(p_vis, t, mem['p_last_t'])
    p_cell_m = jnp.where(p_vis, p_cell, mem['p_cell'])
    p_face_m = jnp.where(p_vis, p_face, mem['p_face'])
    p_held_m = jnp.where(p_vis, p_held, mem['p_held'])
    p_idle = p_vis & (p_streak >= IDLE_STEPS) & (p_ht == IT_NONE)
    rescue_fail = mem['rescue_fail'] + jnp.where(
        (t == mem['last_rescue_t'] + RESCUE_CHECK) & p_vis & (p_streak >= RESCUE_CHECK), 1, 0)
    rescue_ok = (rescue_fail < RESCUE_MAX_FAIL) & (t - mem['last_rescue_t'] >= RESCUE_COOLDOWN)
    want_rescue = p_idle & rescue_ok & r_known
    p_row_top = (p_cell_m // W) <= 4
    resc_acc = jnp.where(p_row_top, IDX[(1, 4)], IDX[(7, 4)])
    resc_face = jnp.where(p_row_top, F_DOWN, F_UP)
    # spare slots (2,2)/(6,2) next to the watch posts are reserved for a spare recipe
    # ingredient so a consumed counter item can be replaced in a few steps
    is_spare = ((jnp.arange(n_d) == SPARE_TOP) | (jnp.arange(n_d) == SPARE_BOT)) & (courier == 1)
    dump_i, has_dump = _pick(d_cost, (d_t == IT_NONE) & (~is_spare))
    sup_cost = cost(SUP_ACC[r_safe], SUP_FACE[r_safe])
    src_costs = jnp.concatenate([sup_cost[None], d_cost])
    src_mask = jnp.concatenate([jnp.asarray([True]),
                                (d_t == IT_ING) & (d_k == r) & (d_blk[:, 1] == 0)])
    src_i, _ = _pick(src_costs, src_mask)
    d_src = jnp.clip(src_i - 1, 0, n_d - 1)
    src_acc = jnp.where(src_i == 0, SUP_ACC[r_safe], D_ACC[d_src])
    src_face = jnp.where(src_i == 0, SUP_FACE[r_safe], D_FACE[d_src])
    x_i = jnp.argmin(cost(X_ACC, X_FACE))
    # watch post (stateful): switch only when standing at the post and either the
    # other stocking counter is stale or the partner has not been sighted lately
    k_seen = seen[S_RC[:K_COUNT, 0], S_RC[:K_COUNT, 1]]
    post = mem['post']
    other = 1 - post
    at_post = pos == S_ACC[post]
    p_side = jnp.where((p_cell_m // W) <= 4, 0, 1)
    lost = t - p_last_t >= PARTNER_LOST
    switch = at_post & (~p_vis) & ((t - k_seen[other] >= PATROL_AGE) | lost)
    # if the partner was sighted recently on the other side, head there
    goto_side = (~lost) & (~p_vis) & (p_side != post) & (mem['p_cell'] >= 0)
    post = jnp.where(switch | goto_side, other, post)
    patrol_i = post
    nc_i = jnp.argmin(DIST[pos, NC])
    sp = jnp.where(post == 0, SPARE_TOP, SPARE_BOT)
    spare_good = (d_t[sp] == IT_ING) & (d_k[sp] == r) & (d_blk[sp, 1] == 0)
    spare_empty = (d_t[sp] == IT_NONE) & (courier == 1)
    spare_stale = (~spare_good) & (d_t[sp] != IT_NONE) & (courier == 1)

    a_deliver = go_or_act(pos, X_ACC[x_i], facing, X_FACE[x_i], NEXT, A_INTERACT)
    a_dump = go_or_act(pos, D_ACC[dump_i], facing, D_FACE[dump_i], NEXT, A_INTERACT)
    a_dump = jnp.where(has_dump, a_dump, A_STAY)
    a_getdish = go_or_act(pos, S_ACC[dish_i], facing, S_FACE[dish_i], NEXT, A_INTERACT)
    a_clear = go_or_act(pos, S_ACC[stale_i], facing, S_FACE[stale_i], NEXT, A_INTERACT)
    a_stock = go_or_act(pos, S_ACC[empty_i], facing, S_FACE[empty_i], NEXT, A_INTERACT)
    a_fetch = go_or_act(pos, src_acc, facing, src_face, NEXT, A_INTERACT)
    # patrol: stand at the post facing the stocking counter; every REFRESH_PERIOD steps
    # lift the (correct) item and put it back next step so it counts as newly placed
    refresh_due = (t - mem['last_refresh_t'] >= REFRESH_PERIOD) & k_good[patrol_i]
    a_patrol = go_or_act(pos, S_ACC[patrol_i], facing, S_FACE[patrol_i], NEXT,
                         jnp.where(refresh_due, A_INTERACT, A_STAY))
    a_offcorner = go_or_act(pos, NC[nc_i], facing, facing, NEXT, A_STAY)
    a_rescue = go_or_act(pos, resc_acc, facing, resc_face, NEXT, A_INTERACT)
    a_spare = go_or_act(pos, D_ACC[sp], facing, D_FACE[sp], NEXT, A_INTERACT)

    held_good = (held_t == IT_ING) & (held_k == r)

    # priority cascade
    a_hold_good = jnp.where(has_dish | want_rescue, a_dump,
                            jnp.where(has_empty, a_stock,
                                      jnp.where(spare_empty, a_spare, a_dump)))
    a_idle = jnp.where(spare_stale, a_spare, jnp.where(spare_empty, a_fetch, a_patrol))
    a_empty_hands = jnp.where(has_dish, a_getdish,
                              jnp.where(want_rescue, a_rescue,
                                        jnp.where(has_stale, a_clear,
                                                  jnp.where(has_empty, a_fetch, a_idle))))
    a_known = jnp.where(held_t == IT_DISH, a_deliver,
                        jnp.where(held_good, a_hold_good,
                                  jnp.where(held_none, a_empty_hands, a_dump)))
    a_unknown = jnp.where(held_t == IT_DISH, a_deliver, a_offcorner)
    action = jnp.where(r_known, a_known, a_unknown).astype(jnp.int32)

    rescuing = want_rescue & held_none & (~has_dish) & (action == A_INTERACT) & \
        (pos == resc_acc) & (facing == resc_face)
    last_rescue_t = jnp.where(rescuing, t, mem['last_rescue_t'])
    refreshing = r_known & held_none & (~has_dish) & (~want_rescue) & (~has_stale) & \
        (~has_empty) & (~spare_stale) & (~spare_empty) & refresh_due & \
        (pos == S_ACC[patrol_i]) & (facing == S_FACE[patrol_i])
    last_refresh_t = jnp.where(refreshing, t, mem['last_refresh_t'])

    new_mem = {
        'r': r.astype(jnp.int32),
        't': (t + 1).astype(jnp.int32),
        'items': items.astype(jnp.int32),
        'seen': seen.astype(jnp.int32),
        'p_cell': p_cell_m.astype(jnp.int32),
        'p_face': p_face_m.astype(jnp.int32),
        'p_held': p_held_m.astype(jnp.int32),
        'p_streak': p_streak.astype(jnp.int32),
        'p_last_t': p_last_t.astype(jnp.int32),
        'last_rescue_t': last_rescue_t.astype(jnp.int32),
        'rescue_fail': rescue_fail.astype(jnp.int32),
        'post': post.astype(jnp.int32),
        'last_refresh_t': last_refresh_t.astype(jnp.int32),
        'last_act': action.astype(jnp.int32),
        'cons_t': cons_t.astype(jnp.int32),
        'cons_n': cons_n.astype(jnp.int32),
        'courier': courier.astype(jnp.int32),
    }
    trace = {'r': r, 'pos': pos, 'held': held_t, 'has_dish': has_dish, 'has_stale': has_stale,
             'has_empty': has_empty, 'p_streak': p_streak, 'want_rescue': want_rescue,
             'rescue_fail': rescue_fail}
    return action, new_mem, trace
