"""grounded_coord_ring policy: outer seat ferries/signals the recipe ingredient, inner seat cooks.

agent_0 (outer ring): reads the always-on recipe indicator, keeps the top shared
counters (2,3)/(2,5) stocked with the current recipe ingredient, removes stale
items, periodically re-places the counter item as a "fresh" heartbeat signal,
stages the other ingredient nearby for quick recipe flips, and delivers any
matching cooked dish left on a shared counter.

agent_1 (inner room): infers the recipe from the button display or from a raw
ingredient that newly appears on a shared counter; after a bounded wait with no
signal it presses the button itself. It loads the pot, plates and delivers.
"""

import jax
import jax.numpy as jnp

from custom_primitives import (
    N_CELLS, CELL_R, CELL_C, INNER, DR, DC, DIR_TO_ACT, STATIC_FLAT, TEMPL_FLAT,
    DIST_FLAT, NEXT_FLAT,
    C11, C12, C13, C14, C15, C16, C17, C21, C27, C31, C33, C34, C35, C37, C41, C43, C45, C47,
    C51, C53, C54, C55, C57, C61, C67, C71, C72, C73, C75, C76, C77,
)

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'ring_ferry_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}

BIG = 999
ACT_STAY = 4
ACT_INTERACT = 5
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

# Shared counters between rooms: (2,3), (2,5), (6,3), (6,5)
SHARED = [(2, 3), (2, 5), (6, 3), (6, 5)]
R_CELLS = [(0, 4), (4, 0), (4, 8), (8, 4)]
STAGES = [(2, 2), (0, 2)]
DUMPS = [  # (wall r, wall c, access cell, dir)
    (0, 1, C11, UP), (1, 0, C11, LEFT), (2, 0, C21, LEFT),
    (0, 6, C16, UP), (2, 6, C16, DOWN), (0, 7, C17, UP), (1, 8, C17, RIGHT), (2, 8, C27, RIGHT),
    (6, 0, C61, LEFT), (7, 0, C71, LEFT), (8, 1, C71, DOWN), (8, 2, C72, DOWN), (6, 2, C72, UP),
    (6, 6, C76, UP), (8, 6, C76, DOWN), (8, 7, C77, DOWN), (7, 8, C77, RIGHT), (6, 8, C67, RIGHT),
]
ND = 18

# agent_1 memory layout
M1_RECIPE = 0
M1_WAIT = 1
M1_TRUST = 2
M1_PREV_HELD = 3
M1_CODES = 4        # 4..7 last-seen shared counter codes
M1_TOUCH = 8        # 8..11 self touched counter k last step
M1_COOLDOWN = 12
M1_STEP = 13
M1_NDELIV = 14
M1_DELIV_ATTEMPT = 15
M1_LAST_ACT = 16
M1_SRC_COMMIT = 17   # 1 = fetch from supply only (counter take failed / item lifted)
M1_SRC_K = 18        # counter index targeted as ingredient source last step, else -1
M1_SIZE = 20
INIT1_LIST = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0]

W_TRUST = 20
W_START = 28
W_UNTRUST = 6

# agent_0 memory layout
M0_RECIPE = 0
M0_CODES = 1     # 1..4 shared counter codes
M0_STAGE = 5     # 5..6 stage codes
M0_IDLE = 7
M0_DUMP = 8      # 8..25 dump occupancy flags
M0_STEP = 26
M0_PREV_HELD = 27
M0_LAST_EVID = 28      # step of last partner-activity evidence
M0_LAST_PRESS = 29     # step of our last button press
M0_SELF_SUFF = 30      # partner shown to know the recipe without our button press
M0_PRESS_COUNT = 31
M0_PRESS_USEFUL = 32
M0_PRESS_PENDING = 33
M0_PREV_TOUCH = 34     # counter index we interacted with last step (-1 none)
M0_SIZE = 36
INIT0_LIST = [-1] + [0] * 28 + [-1000, 0, 0, 0, 0, -1, 0]
HEARTBEAT = 2
K_STOCK = 2
EVID_GAP = 30          # steps without activity evidence before considering a press
PRESS_GAP = 30         # minimum steps between our presses
SELF_SUFF_GAP = 40     # holding items this long after our last press => partner knows recipe itself
MAX_USELESS_PRESSES = 2


class Tables:
    pass


def tables():
    """Build JAX constant tables (called inside traced functions only)."""
    T = Tables()
    T.templ = jnp.asarray(TEMPL_FLAT, dtype=jnp.int32).reshape(N_CELLS, 5, 5, 9)
    T.dist = jnp.asarray(DIST_FLAT, dtype=jnp.int32).reshape(N_CELLS, N_CELLS)
    T.next = jnp.asarray(NEXT_FLAT, dtype=jnp.int32).reshape(N_CELLS, N_CELLS)
    T.cell_r = jnp.asarray(CELL_R, dtype=jnp.int32)
    T.cell_c = jnp.asarray(CELL_C, dtype=jnp.int32)
    T.inner = jnp.asarray(INNER, dtype=jnp.int32)
    T.dr = jnp.asarray(DR, dtype=jnp.int32)
    T.dc = jnp.asarray(DC, dtype=jnp.int32)
    T.dir_to_act = jnp.asarray(DIR_TO_ACT, dtype=jnp.int32)
    T.static = jnp.asarray(STATIC_FLAT, dtype=jnp.int32).reshape(9, 9, 9)
    return T


# ---------------------------------------------------------------------------
# Generic helpers (JAX)
# ---------------------------------------------------------------------------
def localize(T, obs, inner):
    """Return the floor-cell index of self via static-channel template matching."""
    static = obs[:, :, 20:29]
    diff = jnp.sum(jnp.abs(static[None] - T.templ), axis=(1, 2, 3))
    penalty = jnp.where(T.inner == inner, 0, BIG)
    return jnp.argmin(diff + penalty).astype(jnp.int32)


def cell_vec(obs, pr, pc, r, c):
    """Feature vector (40,) of global cell (r,c) and a visibility flag."""
    lr = r - pr + 2
    lc = c - pc + 2
    vis = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    vec = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)]
    return vec * vis.astype(jnp.int32), vis


def item_code(block):
    """Item block [plate, cooked, n0, n1, n2] -> code.

    0 empty, 1/2/3 raw ingredient 0/1/2, 4 empty plate, 5/6 cooked dish of
    pure ingredient 0/1, 7 other cooked dish.
    """
    plate, cooked = block[0], block[1]
    n = block[2:5]
    total = jnp.sum(n)
    raw = (cooked == 0) & (plate == 0) & (total > 0)
    raw_code = 1 + jnp.argmax(n).astype(jnp.int32)
    pure0 = (n[0] == 3) & (n[1] == 0) & (n[2] == 0)
    pure1 = (n[1] == 3) & (n[0] == 0) & (n[2] == 0)
    dish_code = jnp.where(pure0, 5, jnp.where(pure1, 6, 7))
    code = jnp.where(cooked > 0, dish_code, jnp.where(plate > 0, 4, jnp.where(raw, raw_code, 0)))
    return code.astype(jnp.int32)


def recipe_type(block):
    """Recipe block [plate, cooked, n0, n1, n2] -> (valid, type in {0,1})."""
    n = block[2:5]
    valid = jnp.sum(n) > 0
    t = jnp.where(n[0] > 0, 0, 1).astype(jnp.int32)
    return valid, t


def steer(T, pos, facing, tcell, tdir):
    """Action to reach cell tcell, face tdir, and interact."""
    at = pos == tcell
    move = T.next[pos, tcell]
    turn = T.dir_to_act[tdir]
    return jnp.where(at, jnp.where(facing == tdir, ACT_INTERACT, turn), move).astype(jnp.int32)


def goto(T, pos, tcell):
    return jnp.where(pos == tcell, ACT_STAY, T.next[pos, tcell]).astype(jnp.int32)


def nearest(T, pos, cells, valid, penalty=None, facing=None, dirs=None):
    """Index of the valid candidate cell nearest to pos, plus any-valid flag.

    If facing/dirs are given, a target at the current cell costs one extra step
    unless it is already faced (a turn is needed before interacting).
    """
    d = T.dist[pos, cells]
    if penalty is not None:
        d = d + penalty
    if facing is not None:
        d = d + ((cells == pos) & (dirs != facing)).astype(jnp.int32)
    d = jnp.where(valid, d, BIG)
    k = jnp.argmin(d).astype(jnp.int32)
    return k, d[k] < BIG


def front_cell(T, pos, facing):
    fr = T.cell_r[pos] + T.dr[facing]
    fc = T.cell_c[pos] + T.dc[facing]
    return fr, fc


def ia(values):
    return jnp.asarray(values, dtype=jnp.int32)


def ba(values):
    return jnp.asarray(values, dtype=bool)


# ---------------------------------------------------------------------------
# agent_1: inner room cook
# ---------------------------------------------------------------------------
def init_agent_1():
    return jnp.asarray(INIT1_LIST, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    T = tables()
    mem = jnp.where(reset, jnp.asarray(INIT1_LIST, dtype=jnp.int32), memory)
    obs = obs.astype(jnp.int32)
    pos = localize(T, obs, 1)
    pr = T.cell_r[pos]
    pc = T.cell_c[pos]
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    held = item_code(obs[2, 2, 5:10])

    # order matters for ties: from (3,3)/(5,3) prefer (3,4)/(5,4) so the return move faces the supply
    in_pot_cells = ia([C34, C54, C43, C45])
    in_pot_dirs = ia([DOWN, UP, RIGHT, LEFT])
    in_row4_pot_cells = ia([C43, C45])
    in_row4_pot_dirs = ia([RIGHT, LEFT])
    in_plate_cells = ia([C35, C55])
    in_plate_dirs = ia([RIGHT, RIGHT])
    in_goal_cells = ia([C43, C45])
    in_goal_dirs = ia([LEFT, RIGHT])
    in_btn_cells = ia([C34, C54])
    in_btn_dirs = ia([UP, DOWN])
    in_cnt_cells = ia([C33, C35, C53, C55])
    in_cnt_dirs = ia([UP, UP, DOWN, DOWN])
    in_sup_cells = ia([C33, C53])
    in_sup_dirs = ia([LEFT, LEFT])
    in_wait_cells = ia([C43, C45])
    top_penalty = ia([0, 0, 2, 2])
    ones2 = ba([True, True])
    ones4 = ba([True, True, True, True])

    # pot state
    potv, _ = cell_vec(obs, pr, pc, 4, 4)
    pot_counts = potv[31:34]
    pot_n = jnp.sum(pot_counts)
    pot_cooked = potv[30] > 0
    pot_timer = potv[39]
    pot_cooking = pot_timer > 0
    pot_idle = (~pot_cooking) & (~pot_cooked)
    pot_type = jnp.argmax(pot_counts).astype(jnp.int32)

    # button displays
    b1, v1 = cell_vec(obs, pr, pc, 2, 4)
    b2, v2 = cell_vec(obs, pr, pc, 6, 4)
    d1v, d1t = recipe_type(b1[34:39])
    d2v, d2t = recipe_type(b2[34:39])
    disp = (v1 & d1v) | (v2 & d2v)
    disp_type = jnp.where(v1 & d1v, d1t, d2t)

    # shared counters
    prev_codes = mem[M1_CODES:M1_CODES + 4]
    touched = mem[M1_TOUCH:M1_TOUCH + 4]
    codes_now = []
    vis_l = []
    for k in range(4):
        cv, vis = cell_vec(obs, pr, pc, SHARED[k][0], SHARED[k][1])
        codes_now.append(item_code(cv[29:34]))
        vis_l.append(vis)
    codes_now = jnp.stack(codes_now)
    vis = jnp.stack(vis_l)
    known_codes = jnp.where(vis, codes_now, prev_codes)
    changed = vis & (codes_now != prev_codes) & (touched == 0)
    fresh0 = changed & (codes_now == 1)
    fresh1 = changed & (codes_now == 2)
    any_fresh = jnp.any(fresh0) | jnp.any(fresh1)
    fresh_type = jnp.where(jnp.any(fresh0), 0, 1).astype(jnp.int32)
    partner_took_dish = jnp.any(changed & (prev_codes >= 5) & (codes_now == 0))

    # recipe belief update
    recipe = mem[M1_RECIPE]
    wait = mem[M1_WAIT]
    trusted = mem[M1_TRUST]
    cooldown = mem[M1_COOLDOWN]
    ndeliv = mem[M1_NDELIV]
    delivered = (mem[M1_DELIV_ATTEMPT] > 0) & (held == 0)
    ndeliv = ndeliv + delivered.astype(jnp.int32)
    became_unknown = delivered | partner_took_dish
    recipe = jnp.where(became_unknown, -1, recipe)
    wait = jnp.where(became_unknown, 0, wait)
    recipe = jnp.where(any_fresh, fresh_type, recipe)
    recipe = jnp.where(disp, disp_type, recipe)
    signal_from_partner = any_fresh | (disp & (cooldown == 0))
    trusted = jnp.where(signal_from_partner, 1, trusted)
    known = recipe >= 0
    wait = jnp.where(known, 0, wait + 1)
    wait_limit = jnp.where(trusted > 0, W_TRUST, jnp.where(ndeliv == 0, W_START, W_UNTRUST))
    wait_over = wait >= wait_limit

    t = jnp.maximum(recipe, 0)

    # candidate actions
    k_pot, _ = nearest(T, pos, in_pot_cells, ones4, None, facing, in_pot_dirs)
    act_pot = steer(T, pos, facing, in_pot_cells[k_pot], in_pot_dirs[k_pot])
    k_pot4, _ = nearest(T, pos, in_row4_pot_cells, ones2, None, facing, in_row4_pot_dirs)
    act_pot4 = steer(T, pos, facing, in_row4_pot_cells[k_pot4], in_row4_pot_dirs[k_pot4])
    k_plate, _ = nearest(T, pos, in_plate_cells, ones2, None, facing, in_plate_dirs)
    act_plate = steer(T, pos, facing, in_plate_cells[k_plate], in_plate_dirs[k_plate])
    k_goal, _ = nearest(T, pos, in_goal_cells, ones2, None, facing, in_goal_dirs)
    act_goal = steer(T, pos, facing, in_goal_cells[k_goal], in_goal_dirs[k_goal])
    k_btn, _ = nearest(T, pos, in_btn_cells, ones2, None, facing, in_btn_dirs)
    act_btn = steer(T, pos, facing, in_btn_cells[k_btn], in_btn_dirs[k_btn])
    k_wait, _ = nearest(T, pos, in_wait_cells, ones2)
    act_wait = goto(T, pos, in_wait_cells[k_wait])

    empty_cnt = known_codes == 0
    k_empty, any_empty = nearest(T, pos, in_cnt_cells, empty_cnt, top_penalty, facing, in_cnt_dirs)
    act_dump = steer(T, pos, facing, in_cnt_cells[k_empty], in_cnt_dirs[k_empty])
    act_dump = jnp.where(any_empty, act_dump, act_wait)

    match_dish = known & (known_codes == 5 + t)
    k_md, any_md = nearest(T, pos, in_cnt_cells, match_dish, None, facing, in_cnt_dirs)
    act_take_dish = steer(T, pos, facing, in_cnt_cells[k_md], in_cnt_dirs[k_md])

    # ingredient source for the needed type: counters holding it, else supply.
    # If a targeted counter item vanished before we took it (partner lifted it),
    # commit to the supply to avoid periodic livelocks with the partner.
    need = jnp.where(pot_n > 0, pot_type, t)
    need = jnp.clip(need, 0, 1)
    prev_k = mem[M1_SRC_K]
    prev_k_c = jnp.clip(prev_k, 0, 3)
    take_failed = (prev_k >= 0) & (held == 0) & (known_codes[prev_k_c] != 1 + need)
    src_commit = jnp.where(held != 0, 0, jnp.where(take_failed, 1, mem[M1_SRC_COMMIT]))
    src_cells = jnp.concatenate([in_sup_cells[need][None], in_cnt_cells])
    src_dirs = jnp.concatenate([in_sup_dirs[need][None], in_cnt_dirs])
    src_valid = jnp.concatenate([ba([True]), (known_codes == 1 + need) & (src_commit == 0)])
    k_src, _ = nearest(T, pos, src_cells, src_valid, None, facing, src_dirs)
    act_fetch = steer(T, pos, facing, src_cells[k_src], src_dirs[k_src])
    fetch_k = jnp.where(k_src >= 1, k_src - 1, -1).astype(jnp.int32)

    can_press = wait_over & (cooldown == 0) & (~disp)
    act_unknown = jnp.where(can_press, act_btn, act_wait)

    is_dish = held >= 5
    is_plate = held == 4
    is_raw = (held >= 1) & (held <= 3)
    u = held - 1

    # --- holding a dish
    dish_match = known & (held == 5 + t)
    act_dish = jnp.where(dish_match, act_goal,
                         jnp.where(known, act_dump,
                                   jnp.where(wait_over, act_dump, act_wait)))
    # --- holding a plate
    act_plate_h = jnp.where(pot_cooked, act_pot4,
                            jnp.where(pot_cooking, jnp.where(act_pot4 == ACT_INTERACT, ACT_STAY, act_pot4),
                                      act_dump))
    # --- holding a raw ingredient
    can_add = pot_idle & (pot_n < 3) & ((pot_n == 0) | (pot_type == u))
    add_ok = can_add & ((pot_n > 0) | (~known) | (u == t))
    act_raw = jnp.where(add_ok, act_pot, act_dump)
    # --- empty hands
    act_empty = jnp.where(any_md, act_take_dish,
                jnp.where(pot_cooked | pot_cooking, jnp.where(known, act_plate, act_unknown),
                jnp.where(pot_n > 0, act_fetch,
                jnp.where(known, act_fetch, act_unknown))))

    action = jnp.where(is_dish, act_dish,
             jnp.where(is_plate, act_plate_h,
             jnp.where(is_raw, act_raw, act_empty))).astype(jnp.int32)

    # memory update
    fr, fc = front_cell(T, pos, facing)
    interacting = action == ACT_INTERACT
    front_goal = T.static[jnp.clip(fr, 0, 8), jnp.clip(fc, 0, 8), 1] > 0
    front_btn = T.static[jnp.clip(fr, 0, 8), jnp.clip(fc, 0, 8), 4] > 0
    touch_new = jnp.stack([interacting & (fr == SHARED[k][0]) & (fc == SHARED[k][1]) for k in range(4)]).astype(jnp.int32)
    press = interacting & front_btn & (held == 0)
    cooldown = jnp.where(press, 10, jnp.maximum(cooldown - 1, 0))
    # a timeout press means the partner did not signal this cycle: stop trusting
    trusted = jnp.where(press, 0, trusted)
    deliv_attempt = (interacting & front_goal & is_dish).astype(jnp.int32)
    fetching = (action == act_fetch) & (held == 0) & (~any_md) & (~(pot_cooked | pot_cooking)) & ((pot_n > 0) | known)
    src_k_new = jnp.where(fetching, fetch_k, -1).astype(jnp.int32)

    new = mem
    new = new.at[M1_RECIPE].set(recipe)
    new = new.at[M1_WAIT].set(wait)
    new = new.at[M1_TRUST].set(trusted)
    new = new.at[M1_PREV_HELD].set(held)
    new = new.at[M1_CODES:M1_CODES + 4].set(known_codes)
    new = new.at[M1_TOUCH:M1_TOUCH + 4].set(touch_new)
    new = new.at[M1_COOLDOWN].set(cooldown)
    new = new.at[M1_STEP].set(mem[M1_STEP] + 1)
    new = new.at[M1_NDELIV].set(ndeliv)
    new = new.at[M1_DELIV_ATTEMPT].set(deliv_attempt)
    new = new.at[M1_LAST_ACT].set(action)
    new = new.at[M1_SRC_COMMIT].set(src_commit)
    new = new.at[M1_SRC_K].set(src_k_new)
    new = new.astype(jnp.int32)

    trace = {
        'pos': pos, 'held': held, 'recipe': recipe, 'wait': wait, 'trusted': trusted,
        'pot_n': pot_n, 'pot_timer': pot_timer, 'ndeliv': ndeliv, 'codes': known_codes,
        'src_commit': src_commit,
    }
    return action, new, trace


# ---------------------------------------------------------------------------
# agent_0: outer ring ferry / signaller
# ---------------------------------------------------------------------------
def init_agent_0():
    return jnp.asarray(INIT0_LIST, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    T = tables()
    mem = jnp.where(reset, jnp.asarray(INIT0_LIST, dtype=jnp.int32), memory)
    obs = obs.astype(jnp.int32)
    pos = localize(T, obs, 0)
    pr = T.cell_r[pos]
    pc = T.cell_c[pos]
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    held = item_code(obs[2, 2, 5:10])

    out_home = C13
    out_cnt_cells = ia([C13, C15, C73, C75])
    out_cnt_dirs = ia([DOWN, DOWN, UP, UP])
    out_sup_cells = ia([C31, C51])
    out_sup_dirs = ia([RIGHT, RIGHT])
    out_goal_cells = ia([C41, C47])
    out_goal_dirs = ia([RIGHT, LEFT])
    out_stage_cells = ia([C12, C12])
    out_stage_dirs = ia([DOWN, UP])
    dump_cells = ia([d[2] for d in DUMPS])
    dump_dirs = ia([d[3] for d in DUMPS])
    top_valid = ba([True, True, False, False])
    ones2 = ba([True, True])

    # recipe from any visible always-on indicator
    prev_recipe = mem[M0_RECIPE]
    recipe = prev_recipe
    for (rr, rc) in R_CELLS:
        rv, rvis = cell_vec(obs, pr, pc, rr, rc)
        valid, rt = recipe_type(rv[34:39])
        recipe = jnp.where(rvis & valid, rt, recipe)
    known = recipe >= 0
    t = jnp.maximum(recipe, 0)
    flip = (prev_recipe >= 0) & known & (recipe != prev_recipe)

    # shared counters (last seen)
    old_codes = mem[M0_CODES:M0_CODES + 4]
    new_codes = []
    vis_l = []
    for k in range(4):
        cv, vis = cell_vec(obs, pr, pc, SHARED[k][0], SHARED[k][1])
        new_codes.append(jnp.where(vis, item_code(cv[29:34]), old_codes[k]))
        vis_l.append(vis)
    codes = jnp.stack(new_codes)
    cvis = jnp.stack(vis_l)
    prev_touch = mem[M0_PREV_TOUCH]
    consumed = jnp.any(cvis & (old_codes >= 1) & (old_codes <= 3) & (codes == 0)
                       & (jnp.arange(4) != prev_touch))

    # partner activity evidence
    step = mem[M0_STEP]
    partner_visible = jnp.any(obs[:, :, 10] > 0)
    partner_holding = jnp.any(obs[:, :, 15:20] > 0)
    since_press = step - mem[M0_LAST_PRESS]
    b_top, b_top_vis = cell_vec(obs, pr, pc, 2, 4)
    b_bot, b_bot_vis = cell_vec(obs, pr, pc, 6, 4)
    d1v, _ = recipe_type(b_top[34:39])
    d2v, _ = recipe_type(b_bot[34:39])
    display_on = (b_top_vis & d1v) | (b_bot_vis & d2v)
    evidence = partner_holding | flip | consumed
    last_evid = jnp.where(evidence, step, mem[M0_LAST_EVID])
    self_suff = mem[M0_SELF_SUFF]
    self_suff = jnp.where((consumed | partner_holding) & (since_press > SELF_SUFF_GAP), 1, self_suff)
    pending = mem[M0_PRESS_PENDING]
    useful = mem[M0_PRESS_USEFUL] + ((pending > 0) & evidence).astype(jnp.int32)
    pending = jnp.where(evidence | (since_press > 45), 0, pending)
    inert = (mem[M0_PRESS_COUNT] >= MAX_USELESS_PRESSES) & (useful == 0)
    press_needed = known & (self_suff == 0) & (~inert) & (step - last_evid >= EVID_GAP) \
        & (since_press >= PRESS_GAP) & (~display_on) & (held == 0)
    act_press = steer(T, pos, facing, C14, DOWN)
    # stages
    st = mem[M0_STAGE:M0_STAGE + 2]
    new_st = []
    for k in range(2):
        cv, vis = cell_vec(obs, pr, pc, STAGES[k][0], STAGES[k][1])
        new_st.append(jnp.where(vis, item_code(cv[29:34]), st[k]))
    st = jnp.stack(new_st)
    # dumps
    dumps = mem[M0_DUMP:M0_DUMP + ND]
    new_d = []
    for k in range(ND):
        cv, vis = cell_vec(obs, pr, pc, DUMPS[k][0], DUMPS[k][1])
        occ = jnp.sum(cv[29:34]) > 0
        new_d.append(jnp.where(vis, occ.astype(jnp.int32), dumps[k]))
    dumps = jnp.stack(new_d)

    # candidate actions
    act_home = goto(T, pos, out_home)
    act_home = jnp.where((pos == out_home) & (facing != DOWN), T.dir_to_act[DOWN], act_home)
    k_goal, _ = nearest(T, pos, out_goal_cells, ones2)
    act_goal = steer(T, pos, facing, out_goal_cells[k_goal], out_goal_dirs[k_goal])
    k_dump, any_dump = nearest(T, pos, dump_cells, dumps == 0)
    act_dump = steer(T, pos, facing, dump_cells[k_dump], dump_dirs[k_dump])
    act_dump = jnp.where(any_dump, act_dump, act_home)
    k_stage_empty, any_stage_empty = nearest(T, pos, out_stage_cells, st == 0)
    act_stage = steer(T, pos, facing, out_stage_cells[k_stage_empty], out_stage_dirs[k_stage_empty])
    act_stage_or_dump = jnp.where(any_stage_empty, act_stage, act_dump)

    top_empty = (codes == 0) & top_valid
    k_te, any_top_empty = nearest(T, pos, out_cnt_cells, top_empty)
    act_place = steer(T, pos, facing, out_cnt_cells[k_te], out_cnt_dirs[k_te])

    match_dish = known & (codes == 5 + t)
    k_md, any_md = nearest(T, pos, out_cnt_cells, match_dish)
    act_take_dish = steer(T, pos, facing, out_cnt_cells[k_md], out_cnt_dirs[k_md])

    stale = top_valid & (codes >= 1) & (codes <= 4) & (codes != 1 + t)
    k_st, any_stale = nearest(T, pos, out_cnt_cells, stale)
    act_take_stale = steer(T, pos, facing, out_cnt_cells[k_st], out_cnt_dirs[k_st])

    stocked = jnp.sum(((codes == 1 + t) & top_valid).astype(jnp.int32))
    need_stock = known & (stocked < K_STOCK) & any_top_empty
    # source of ingredient t: stage holding it, else supply
    src_cells = jnp.concatenate([out_stage_cells, out_sup_cells[t][None]])
    src_dirs = jnp.concatenate([out_stage_dirs, out_sup_dirs[t][None]])
    src_valid = jnp.concatenate([st == 1 + t, ba([True])])
    k_src, _ = nearest(T, pos, src_cells, src_valid)
    act_fetch_t = steer(T, pos, facing, src_cells[k_src], src_dirs[k_src])

    other = 1 - t
    have_other = jnp.any(st == 1 + other)
    need_stage_other = known & (~have_other) & any_stage_empty
    act_fetch_other = steer(T, pos, facing, out_sup_cells[other], out_sup_dirs[other])

    # heartbeat: periodically lift and re-place the counter item at home, but never
    # while the partner is visible (it may be about to take the item)
    at_home = (pos == out_home) & (facing == DOWN)
    home_stocked = codes[0] == 1 + t
    idle = mem[M0_IDLE]
    idle = jnp.where(at_home & home_stocked, idle + 1, 0)
    beat = at_home & home_stocked & (idle >= HEARTBEAT) & (~partner_visible)
    act_idle = jnp.where(beat, ACT_INTERACT, act_home)
    idle = jnp.where(beat, 0, idle)

    is_dish = held >= 5
    is_plate = held == 4
    is_raw = (held >= 1) & (held <= 3)
    u = held - 1

    act_dish = jnp.where(known & (held == 5 + t), act_goal, jnp.where(known, act_dump, act_home))
    act_plate_h = act_dump
    act_raw = jnp.where(known & (u == t),
                        jnp.where(any_top_empty, act_place, act_stage_or_dump),
                        jnp.where(known, act_stage_or_dump, act_home))
    # signal first (stock an empty top counter), then clear stale, then stage the other type
    act_empty = jnp.where(~known, act_home,
                jnp.where(any_md, act_take_dish,
                jnp.where(need_stock, act_fetch_t,
                jnp.where(press_needed, act_press,
                jnp.where(any_stale, act_take_stale,
                jnp.where(need_stage_other, act_fetch_other, act_idle))))))

    action = jnp.where(is_dish, act_dish,
             jnp.where(is_plate, act_plate_h,
             jnp.where(is_raw, act_raw, act_empty))).astype(jnp.int32)

    fr, fc = front_cell(T, pos, facing)
    interacting = action == ACT_INTERACT
    pressed = interacting & press_needed & (fr == 2) & (fc == 4)
    touch_k = -1
    for k in range(4):
        touch_k = jnp.where(interacting & (fr == SHARED[k][0]) & (fc == SHARED[k][1]), k, touch_k)

    new = mem
    new = new.at[M0_RECIPE].set(recipe)
    new = new.at[M0_CODES:M0_CODES + 4].set(codes)
    new = new.at[M0_STAGE:M0_STAGE + 2].set(st)
    new = new.at[M0_IDLE].set(idle)
    new = new.at[M0_DUMP:M0_DUMP + ND].set(dumps)
    new = new.at[M0_STEP].set(step + 1)
    new = new.at[M0_PREV_HELD].set(held)
    new = new.at[M0_LAST_EVID].set(jnp.where(pressed, step, last_evid))
    new = new.at[M0_LAST_PRESS].set(jnp.where(pressed, step, mem[M0_LAST_PRESS]))
    new = new.at[M0_SELF_SUFF].set(self_suff)
    new = new.at[M0_PRESS_COUNT].set(mem[M0_PRESS_COUNT] + pressed.astype(jnp.int32))
    new = new.at[M0_PRESS_USEFUL].set(useful)
    new = new.at[M0_PRESS_PENDING].set(jnp.where(pressed, 1, pending))
    new = new.at[M0_PREV_TOUCH].set(jnp.asarray(touch_k, dtype=jnp.int32))
    new = new.astype(jnp.int32)

    trace = {'pos': pos, 'held': held, 'recipe': recipe, 'codes': codes, 'stage': st, 'idle': idle,
             'self_suff': self_suff, 'press_needed': press_needed, 'last_evid': last_evid,
             'presses': new[M0_PRESS_COUNT]}
    return action, new, trace
