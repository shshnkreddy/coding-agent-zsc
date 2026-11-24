"""Overcooked V2 test_time_simple: counter-signal recipe relay + solo cook.

Layout facts (see context/environment/README.md):
  * agent_0 lives in the left room (cols 1-2). It can see the always-on
    recipe indicator at (2,0) and can only pick up ingredient 2 from (0,2)/(4,2)
    and place it on the shared counters (1,3), (2,3), (3,3).
  * agent_1 lives in the right room (cols 5-6). It reaches pot (2,4), plate
    piles (1,4)/(3,4), ingredient piles 0 (1,7) / 1 (3,7) and delivery (2,7),
    but never sees the recipe indicator nor agent_0.

Protocol (row-matched, the natural Schelling convention):
  * recipe [3,0,0]  ->  one item on the TOP counter (1,3)   (ingredient-0 pile is on row 1)
  * recipe [0,3,0]  ->  one item on the BOTTOM counter (3,3) (ingredient-1 pile is on row 3)
agent_1 decodes tolerantly (top-only=0, bottom-only=1, middle-only=1, else
ambiguous), gates pot insertion / delivery on a definitive current signal,
and stages wrong items on its own counters (0,5),(0,6),(4,5),(4,6).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'row_signal_solo_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# facing ids (observation encoding)
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
# action ids
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5

# item codes
C_EMPTY, C_ING0, C_ING1, C_ING2, C_PLATE, C_DISH0, C_DISH1, C_OTHER = 0, 1, 2, 3, 4, 5, 6, 7

HINT_MAX_AGE = 12
WAIT_TIMEOUT = 12
BLIND_AFTER_STEPS = 60  # no signal ever seen by then -> alternate dish types blindly

MEM1_LEN = 16
MEM0_LEN = 4


# ----------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------

def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _locate(obs, ch, gr, gc):
    """Global (row, col) of self from a static anchor channel always in view."""
    flat = obs[:, :, ch].reshape(-1)
    idx = jnp.argmax(flat)
    i = idx // 5
    j = idx % 5
    return _i32(gr - (i - 2)), _i32(gc - (j - 2))


def _cell(obs, r, c, gr, gc):
    """Feature vector of global cell (gr, gc) as seen from self at (r, c)."""
    i = gr - r + 2
    j = gc - c + 2
    vis = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    ic = jnp.clip(i, 0, 4)
    jc = jnp.clip(j, 0, 4)
    return jnp.where(vis, obs[ic, jc, :], 0), vis


def _item_code(vec):
    """vec = [plate, cooked, c0, c1, c2] -> item code."""
    plate, cooked, c0, c1, c2 = vec[0], vec[1], vec[2], vec[3], vec[4]
    total = c0 + c1 + c2
    empty = (plate == 0) & (cooked == 0) & (total == 0)
    raw = (plate == 0) & (cooked == 0)
    ing0 = raw & (c0 > 0) & (c1 == 0) & (c2 == 0)
    ing1 = raw & (c1 > 0) & (c0 == 0) & (c2 == 0)
    ing2 = raw & (c2 > 0) & (c0 == 0) & (c1 == 0)
    plate_only = (plate == 1) & (cooked == 0) & (total == 0)
    dish0 = (cooked == 1) & (c0 == 3) & (c1 == 0) & (c2 == 0)
    dish1 = (cooked == 1) & (c1 == 3) & (c0 == 0) & (c2 == 0)
    code = _i32(C_OTHER)
    code = jnp.where(dish1, C_DISH1, code)
    code = jnp.where(dish0, C_DISH0, code)
    code = jnp.where(plate_only, C_PLATE, code)
    code = jnp.where(ing2, C_ING2, code)
    code = jnp.where(ing1, C_ING1, code)
    code = jnp.where(ing0, C_ING0, code)
    code = jnp.where(empty, C_EMPTY, code)
    return _i32(code)


def _task(r, c, f, w=0):
    return jnp.stack([_i32(r), _i32(c), _i32(f), _i32(w)])


def _sel(cond, a, b):
    return jnp.where(cond, a, b)


def _nav(r, c, f, task):
    """Action to reach task cell, face task direction, then INTERACT (or STAY)."""
    tr, tc, tf, w = task[0], task[1], task[2], task[3]
    dr = tr - r
    dc = tc - c
    mv_r = jnp.where(dr > 0, A_DOWN, A_UP)
    mv_c = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    tf_horiz = (tf == RIGHT) | (tf == LEFT)
    # last move should align with the target facing when possible
    move = jnp.where(tf_horiz,
                     jnp.where(dr != 0, mv_r, mv_c),
                     jnp.where(dc != 0, mv_c, mv_r))
    at = (dr == 0) & (dc == 0)
    dir_to_act = jnp.array([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)
    turn = dir_to_act[jnp.clip(tf, 0, 3)]
    final = jnp.where(f == tf, jnp.where(w > 0, A_STAY, A_INTERACT), turn)
    return _i32(jnp.where(at, final, move))


def _facing(obs):
    return _i32(jnp.argmax(obs[2, 2, 1:5]))


def _first_true(pred, order):
    """Index (into original array) of the first true entry following `order`, else -1."""
    po = pred[order]
    k = jnp.argmax(po)
    return jnp.where(jnp.any(po), order[k], -1)


def _present(vec):
    return (vec[29] + vec[30] + vec[31] + vec[32] + vec[33]) > 0


# ----------------------------------------------------------------------------
# agent 1: cook in the right room
# ----------------------------------------------------------------------------

def init_agent_1():
    mem = jnp.zeros((MEM1_LEN,), dtype=jnp.int32)
    mem = mem.at[1].set(-1)  # L: last definitive signal
    mem = mem.at[3].set(-1)  # G: committed fetch target
    mem = mem.at[13].set(1)  # X: last delivered dish type (blind mode alternates from it)
    return mem


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_1(), memory)
    obs = obs.astype(jnp.int32)
    r, c = _locate(obs, 22, 2, 4)
    f = _facing(obs)
    held = _item_code(obs[2, 2, 5:10])

    # pot state
    pot, _ = _cell(obs, r, c, 2, 4)
    pc0, pc1, pc2 = pot[31], pot[32], pot[33]
    ptotal = pc0 + pc1 + pc2
    timer = pot[39]
    cooked = pot[30] == 1
    cooking = timer > 0
    pot_empty = (ptotal == 0) & (~cooked) & (~cooking)
    partial = (ptotal > 0) & (ptotal < 3) & (~cooked) & (~cooking)
    pot_type = jnp.where(pc0 == ptotal, 0, jnp.where(pc1 == ptotal, 1, -1))

    # signal counters (visible only from column 5)
    top_v, vis_t = _cell(obs, r, c, 1, 3)
    mid_v, vis_m = _cell(obs, r, c, 2, 3)
    bot_v, vis_b = _cell(obs, r, c, 3, 3)
    sig_vis = vis_t & vis_m & vis_b
    top = _present(top_v)
    mid = _present(mid_v)
    bot = _present(bot_v)
    s_dec = jnp.where(top & (~bot), 0,
            jnp.where(bot & (~top), 1,
            jnp.where(mid & (~top) & (~bot), 1, -1)))
    S = _i32(jnp.where(sig_vis, s_dec, -2))
    s_def = S >= 0

    # memory decode
    L = mem[1]
    age = mem[2]
    G = mem[3]
    W = mem[4]
    K = mem[5:9]

    # update counter knowledge for (0,5),(0,6),(4,5),(4,6)
    cells = ((0, 5), (0, 6), (4, 5), (4, 6))
    newK = []
    for idx, (gr, gc) in enumerate(cells):
        v, vis = _cell(obs, r, c, gr, gc)
        code = _item_code(v[29:34])
        newK.append(jnp.where(vis, code, K[idx]))
    K = jnp.stack(newK)

    # partner-signal tracking (raw reading, before any override)
    L_new = jnp.where(s_def, S, L)
    age_new = jnp.where(s_def, 0, jnp.minimum(age + 1, 999))

    # blind mode: the partner has never shown any signal.  The recipe is only
    # resampled after a correct delivery, so alternating dish types is right
    # two thirds of the time in steady state; strictly better than idling.
    X = mem[13]
    blind = ((L_new < 0) | (age_new > BLIND_AFTER_STEPS)) & (mem[0] >= BLIND_AFTER_STEPS)
    S = _i32(jnp.where(blind, 1 - X, S))
    s_def = S >= 0

    # belief / hint
    hint_ok = (L >= 0) & (age_new <= HINT_MAX_AGE) & (~blind)
    B = jnp.where(s_def, S,
        jnp.where(S == -1, jnp.where(hint_ok, 1 - L, -1), G))
    B = _i32(B)
    W_new = jnp.where(s_def, 0, jnp.where(S == -1, jnp.minimum(W + 1, 999), W))
    # effective signal: current definitive reading, else the committed belief
    # while the counters are out of view (column 6), else unknown.
    E = _i32(jnp.where(s_def, S, jnp.where(S == -2, G, -1)))
    e_def = E >= 0

    # tasks
    T_HUB = _task(2, 5, LEFT, 1)
    T_POT = _task(2, 5, LEFT, 0)
    T_DELIVER = _task(2, 6, RIGHT, 0)
    T_PILE = jnp.stack([_task(1, 6, RIGHT, 0), _task(3, 6, RIGHT, 0)])
    T_PLATE = _sel(r <= 2, _task(1, 5, LEFT, 0), _task(3, 5, LEFT, 0))
    TC = jnp.stack([_task(1, 5, UP, 0), _task(1, 6, UP, 0), _task(3, 5, DOWN, 0), _task(3, 6, DOWN, 0)])

    order_ing0 = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    order_ing1 = jnp.array([2, 3, 0, 1], dtype=jnp.int32)
    order_other = jnp.array([1, 3, 0, 2], dtype=jnp.int32)

    def stage_task(order, fallback):
        idx = _first_true(K == C_EMPTY, order)
        return _sel(idx >= 0, TC[jnp.clip(idx, 0, 3)], fallback)

    def fetch_ing_task(k):
        k = jnp.clip(k, 0, 1)
        order = _sel(k == 0, order_ing0, order_ing1)
        idx = _first_true(K == (k + 1), order)
        return _sel(idx >= 0, TC[jnp.clip(idx, 0, 3)], T_PILE[k])

    def fetch_dish_task(k):
        idx = _first_true(K == (C_DISH0 + jnp.clip(k, 0, 1)), order_other)
        return idx, TC[jnp.clip(idx, 0, 3)]

    # --- holding a dish ---
    is_dish = (held == C_DISH0) | (held == C_DISH1) | (held == C_OTHER)
    dish_k = jnp.where(held == C_DISH0, 0, jnp.where(held == C_DISH1, 1, -1))
    dish_match = e_def & (E == dish_k)
    dish_wrong = e_def & (E != dish_k)
    t_dish = _sel(dish_match, T_DELIVER,
             _sel(dish_wrong, stage_task(order_other, T_DELIVER),
             _sel(W_new > WAIT_TIMEOUT, T_DELIVER, T_HUB)))

    # --- holding a plate ---
    t_plate = _sel(cooked, T_POT,
              _sel(cooking, T_HUB, stage_task(order_other, T_HUB)))

    # --- holding an ingredient ---
    is_ing = (held == C_ING0) | (held == C_ING1) | (held == C_ING2)
    ing_k = held - 1
    stage_ing = stage_task(_sel(ing_k == 0, order_ing0, _sel(ing_k == 1, order_ing1, order_other)), T_POT)
    t_ing_busy = stage_task(_sel(ing_k == 0, order_ing0, _sel(ing_k == 1, order_ing1, order_other)), T_HUB)
    t_ing_partial = _sel(pot_type == ing_k, T_POT, stage_ing)
    # signal in transition (absent/ambiguous shortly after a steady reading):
    # the recipe most likely flipped, so stage the held ingredient right away.
    hint_wrong = (S == -1) & hint_ok & ((1 - L) != ing_k)
    t_ing_empty = _sel(e_def & (E == ing_k), T_POT,
                  _sel(e_def | hint_wrong, stage_ing,
                  _sel(W_new > WAIT_TIMEOUT, T_POT, T_HUB)))
    t_ing = _sel(cooking | cooked, t_ing_busy,
            _sel(partial, t_ing_partial, t_ing_empty))
    t_ing = _sel(ing_k == 2, t_ing_busy, t_ing)

    # --- empty hands ---
    d_idx, d_task = fetch_dish_task(B)
    staged_dish = (B >= 0) & (d_idx >= 0)
    fill_k = jnp.where(pot_type >= 0, pot_type, jnp.where(B >= 0, B, 0))
    t_empty = _sel(cooked, T_PLATE,
              _sel(cooking, _sel(staged_dish, d_task, T_PLATE),
              _sel(partial, fetch_ing_task(fill_k),
              _sel(staged_dish, d_task,
              _sel(B >= 0, fetch_ing_task(B), T_HUB)))))

    task = _sel(is_dish, t_dish,
           _sel(held == C_PLATE, t_plate,
           _sel(is_ing, t_ing, t_empty)))

    action = _nav(r, c, f, task)

    delivered = (action == A_INTERACT) & (r == 2) & (c == 6) & (f == RIGHT) & is_dish
    mem = mem.at[0].set(mem[0] + 1)
    mem = mem.at[1].set(L_new)
    mem = mem.at[2].set(age_new)
    # a delivery may resample the recipe: drop the committed belief so the
    # next fetch is decided only after re-reading the counters from column 5.
    mem = mem.at[3].set(jnp.where(delivered, -1, B))
    mem = mem.at[4].set(W_new)
    mem = mem.at[5:9].set(K)
    mem = mem.at[9].set(mem[9] + delivered.astype(jnp.int32))
    mem = mem.at[10].set(S)
    mem = mem.at[11].set(held)
    mem = mem.at[12].set(blind.astype(jnp.int32))
    mem = mem.at[13].set(jnp.where(delivered & (dish_k >= 0), dish_k, X))
    trace = {"pos": jnp.stack([r, c]), "held": held, "signal": S, "belief": B,
             "task": task, "pot_total": ptotal, "delivered": mem[9]}
    return _i32(action), mem, trace


# ----------------------------------------------------------------------------
# agent 0: recipe relay in the left room
# ----------------------------------------------------------------------------

def init_agent_0():
    return jnp.zeros((MEM0_LEN,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_0(), memory)
    obs = obs.astype(jnp.int32)
    r, c = _locate(obs, 23, 2, 0)
    f = _facing(obs)
    held = _item_code(obs[2, 2, 5:10])

    ind, _ = _cell(obs, r, c, 2, 0)
    rc0, rc1 = ind[36], ind[37]
    k = _i32(jnp.where(rc1 > rc0, 1, 0))
    des_top = k == 0

    top_v, _ = _cell(obs, r, c, 1, 3)
    mid_v, _ = _cell(obs, r, c, 2, 3)
    bot_v, _ = _cell(obs, r, c, 3, 3)
    top = _present(top_v)
    mid = _present(mid_v)
    bot = _present(bot_v)
    desired_occ = jnp.where(des_top, top, bot)
    wrong_occ = jnp.where(des_top, bot, top)

    T_TOP = _task(1, 2, RIGHT, 0)
    T_BOT = _task(3, 2, RIGHT, 0)
    T_MID = _task(2, 2, RIGHT, 0)
    T_IDLE = _sel(des_top, _task(1, 2, RIGHT, 1), _task(3, 2, RIGHT, 1))
    T_DES = _sel(des_top, T_TOP, T_BOT)
    T_WRONG = _sel(des_top, T_BOT, T_TOP)
    T_PILE = _sel(des_top, _task(1, 2, UP, 0), _task(3, 2, DOWN, 0))

    # dump counters: (0,1) via (1,1) UP ; (4,1) via (3,1) DOWN ; (1,0) via (1,1) LEFT ; (3,0) via (3,1) LEFT
    d01, v01 = _cell(obs, r, c, 0, 1)
    d41, v41 = _cell(obs, r, c, 4, 1)
    d10, v10 = _cell(obs, r, c, 1, 0)
    d30, v30 = _cell(obs, r, c, 3, 0)
    free = jnp.stack([v01 & (~_present(d01)), v41 & (~_present(d41)),
                      v10 & (~_present(d10)), v30 & (~_present(d30))])
    TD = jnp.stack([_task(1, 1, UP, 0), _task(3, 1, DOWN, 0), _task(1, 1, LEFT, 0), _task(3, 1, LEFT, 0)])
    order = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    didx = _first_true(free, order)
    T_DUMP = TD[jnp.clip(jnp.where(didx >= 0, didx, 0), 0, 3)]

    t_hold = _sel(~desired_occ, T_DES, T_DUMP)
    t_free = _sel(wrong_occ, T_WRONG,
             _sel(mid, T_MID,
             _sel(desired_occ, T_IDLE, T_PILE)))
    task = _sel(held != C_EMPTY, t_hold, t_free)
    action = _nav(r, c, f, task)

    mem = mem.at[0].set(mem[0] + 1)
    mem = mem.at[1].set(k)
    trace = {"pos": jnp.stack([r, c]), "recipe": k, "held": held, "task": task,
             "sig": jnp.stack([top.astype(jnp.int32), mid.astype(jnp.int32), bot.astype(jnp.int32)])}
    return _i32(action), mem, trace
