"""Overcooked V2 ``test_time_wide`` policy.

Layout facts (rows x cols = 7 x 6):

    WWXBWW      row 0: goal (0,2), plate pile (0,3)
    0 A  0      row 1: ingredient-0 supplies (1,0) (1,5)
    1    1      row 2: ingredient-1 supplies (2,0) (2,5)
    WPWPWW      row 3: pots (3,1) (3,3); counters (3,2) (3,4)
    3 A  3      row 4: ingredient-3 supplies (useless)
    W    W      row 5
    WWRWWW      row 6: always-on recipe indicator (6,2)

The two rooms are disconnected.  agent_0 (top room) has every resource
needed to cook and deliver but can never see the recipe indicator.
agent_1 (bottom room) always sees the indicator but cannot cook or
deliver.  Therefore:

* agent_0 is the chef.  It keeps both pots busy (finishing partial pots
  first, otherwise loading the dish type with the lower inventory), keeps a
  stock of cooked dishes on the top counters (0,1)/(0,4) next to the
  delivery station, and delivers only the dish type it currently wants.
  The wanted type comes from the partner's column when the partner is
  visible in row 4 (column <= 2 -> 0, column >= 3 -> 1).  Because a wrong
  delivery does not resample the recipe, the mapping is verified from the
  environment's own dynamics: a change of the partner's side shortly after
  a delivery proves that delivery was correct and calibrates the mapping.
  Until calibrated, the default mapping is trusted for one delivery without
  a flip (two if the partner was seen standing under a pot facing it), then
  dish types are alternated, which succeeds ~2/3 of the time regardless of
  the partner.  Two side changes between deliveries mark the partner as
  uninformative (blind alternation for the rest of the episode).
* agent_1 is the signaller.  It stands directly below the pot whose
  ingredient matches the current recipe, facing the pot: (4,1) for recipe 0
  and (4,3) for recipe 1.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'chef_top_signal_bottom', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

I32 = jnp.int32

# actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# facing ids (observation encoding)
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
FACE_TO_ACTION_T = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# observation channels
CH_PARTNER = 11
CH_GOAL = 23
CH_RECIPE_IND = 25
CH_ITEM = 32       # 32..38: plate, cooked, n0, n1, n2, n3
CH_RECIPE = 38     # 38..44
CH_TIMER = 44

# agent_0 memory layout
M_SIDE_LAST = 0    # last observed partner side (0/1), -1 if never seen
M_SEEN = 1
M_NEED_LOOK = 2
M_LOOK_T = 3
M_STEP = 4
M_LAST_D = 5       # type of the last delivered dish, -1 if none
M_PRE_SIDE = 6     # partner side at the time of the last delivery, -1 if unknown
M_FLIPPED = 7      # a side flip (vs pre_side) has been observed since the last delivery
M_CHANGES = 8      # number of side changes observed since the last delivery
M_CALIB = 9        # mapping calibrated by a post-delivery flip
M_INVERT = 10      # mapping bit: recipe = side XOR invert
M_BLIND = 11       # partner signal judged uninformative: blind alternation
M_WANT = 12
M_NOFLIP = 13      # consecutive deliveries without an observed post-delivery flip
M_CANON = 14       # partner has been seen at a canonical signalling post facing the pot
M_POT = 16         # 2 pots x 6 (n0, n1, n2, n3, cooked, timer)
M_CTR = 28         # 4 counters x 6 (plate, cooked, n0, n1, n2, n3)
MEM0_SIZE = 52

POT_POS = ((3, 1), (3, 3))
POT_ACCESS_T = ((2, 1, F_DOWN), (2, 3, F_DOWN))
CTR_POS = ((0, 1), (0, 4), (3, 2), (3, 4))
CTR_ACCESS_T = ((1, 1, F_UP), (1, 4, F_UP), (2, 2, F_DOWN), (2, 4, F_DOWN))

LOOK_STEPS = 4


def i32(x):
    return jnp.asarray(x, dtype=I32)


# ----------------------------------------------------------------------------
# observation helpers
# ----------------------------------------------------------------------------

def locate(obs, channel, anchor_r, anchor_c):
    """Global self position from a unique static landmark inside the view."""
    mask = obs[:, :, channel] > 0
    idx = jnp.argmax(mask.reshape(-1))
    i = idx // 5
    j = idx % 5
    return i32(anchor_r - i + 2), i32(anchor_c - j + 2)


def read_cell(obs, r, c, gr, gc):
    """Return (channel vector, visible flag) of global cell (gr, gc)."""
    i = gr - r + 2
    j = gc - c + 2
    vis = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    ic = jnp.clip(i, 0, 4)
    jc = jnp.clip(j, 0, 4)
    return obs[ic, jc, :], vis


def item_kind(v):
    """Classify a 6-vector item block.

    kind: 0 none, 1 raw ingredient, 2 empty plate, 3 cooked dish.
    ing_type: argmax ingredient id.
    dish_type: 0 -> [3,0,0,0], 1 -> [0,3,0,0], 2 -> anything else.
    """
    plate = v[0]
    cooked = v[1]
    total = v[2] + v[3] + v[4] + v[5]
    none = (plate == 0) & (cooked == 0) & (total == 0)
    kind = jnp.where(none, 0, jnp.where(cooked > 0, 3, jnp.where(plate > 0, 2, 1)))
    ing_type = jnp.argmax(v[2:6]).astype(I32)
    dish_type = jnp.where((v[2] == 3) & (total == 3), 0,
                          jnp.where((v[3] == 3) & (total == 3), 1, 2))
    return i32(kind), ing_type, i32(dish_type)


def pot_info(st):
    """st = (n0, n1, n2, n3, cooked, timer)."""
    total = st[0] + st[1] + st[2] + st[3]
    ptype = jnp.where(total == 0, -1,
                      jnp.where(st[0] == total, 0,
                                jnp.where(st[1] == total, 1, 2)))
    cooking = st[5] > 0
    ready = st[4] > 0
    idle = (~cooking) & (~ready)
    return i32(total), i32(ptype), cooking, ready, idle


def navigate(r, c, f, tr, tc, tf, do):
    """Move inside an obstacle-free rectangular room towards (tr, tc), turn to
    face tf, then perform ``do`` (STAY or INTERACT)."""
    at = (r == tr) & (c == tc)
    dr = tr - r
    dc = tc - c
    horiz_first = (tf == F_UP) | (tf == F_DOWN)
    hmove = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    vmove = jnp.where(dr > 0, A_DOWN, A_UP)
    move = jnp.where(horiz_first,
                     jnp.where(dc != 0, hmove, vmove),
                     jnp.where(dr != 0, vmove, hmove))
    turn = jnp.asarray(FACE_TO_ACTION_T, dtype=I32)[jnp.clip(tf, 0, 3)]
    return i32(jnp.where(at, jnp.where(f == tf, do, turn), move))


def select_pot(flags, prefer):
    other = 1 - prefer
    return i32(jnp.where(flags[prefer], prefer, jnp.where(flags[other], other, -1)))


def first_index(flags):
    idx = i32(-1)
    for k in reversed(range(len(flags))):
        idx = jnp.where(flags[k], i32(k), idx)
    return idx


def pick(cond, new, old):
    return tuple(jnp.where(cond, i32(n), i32(o)) for n, o in zip(new, old))


# ----------------------------------------------------------------------------
# agent 0: chef in the top room
# ----------------------------------------------------------------------------

def init_agent_0():
    mem = jnp.zeros((MEM0_SIZE,), dtype=I32)
    return mem.at[M_SIDE_LAST].set(-1).at[M_LAST_D].set(-1).at[M_PRE_SIDE].set(-1)


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_0(), memory)
    obs = obs.astype(I32)
    POT_ACCESS = jnp.asarray(POT_ACCESS_T, dtype=I32)
    CTR_ACCESS = jnp.asarray(CTR_ACCESS_T, dtype=I32)

    r, c = locate(obs, CH_GOAL, 0, 2)
    r = jnp.clip(r, 1, 2)
    c = jnp.clip(c, 1, 4)
    f = jnp.argmax(obs[2, 2, 1:5]).astype(I32)
    held = obs[2, 2, 5:11]
    h_kind, h_ing, h_dish = item_kind(held)

    # ---- partner signal bookkeeping ---------------------------------------
    pmask = obs[:, :, CH_PARTNER] > 0
    pvis = jnp.any(pmask)
    pidx = jnp.argmax(pmask.reshape(-1))
    pc = c + (pidx % 5) - 2
    side_obs = i32(jnp.where(pc >= 3, 1, 0))
    pr = r + (pidx // 5) - 2
    p_face_up = obs[jnp.clip(pidx // 5, 0, 4), jnp.clip(pidx % 5, 0, 4), 12] > 0
    canon_now = pvis & (pr == 4) & ((pc == 1) | (pc == 3)) & p_face_up
    canon = i32(jnp.where(canon_now, 1, memory[M_CANON]))
    noflip = memory[M_NOFLIP]

    side_last_old = memory[M_SIDE_LAST]
    last_d = memory[M_LAST_D]
    pre_side = memory[M_PRE_SIDE]
    flipped = memory[M_FLIPPED]
    changes = memory[M_CHANGES]
    calib = memory[M_CALIB]
    invert = memory[M_INVERT]
    blind = memory[M_BLIND]

    changed = pvis & (side_last_old >= 0) & (side_obs != side_last_old)
    changes = i32(changes + changed)
    blind = i32(jnp.where(changes >= 2, 1, blind))
    # reference side for flip detection: the side known at the last delivery,
    # or the first sighting if none was known
    pre_side_known = pre_side >= 0
    pre_side = i32(jnp.where(pvis & (~pre_side_known), side_obs, pre_side))
    is_flip = pvis & pre_side_known & (side_obs != pre_side)
    do_calib = is_flip & (flipped == 0) & (last_d >= 0) & (blind == 0)
    invert = i32(jnp.where(do_calib, side_obs ^ (1 - jnp.clip(last_d, 0, 1)), invert))
    calib = i32(jnp.where(do_calib, 1, calib))
    noflip = i32(jnp.where(is_flip, 0, noflip))
    flipped = i32(jnp.where(is_flip, 1, flipped))
    side_last = i32(jnp.where(pvis, side_obs, side_last_old))
    seen = i32(jnp.where(pvis, 1, memory[M_SEEN]))
    need_look = i32(jnp.where(pvis, 0, memory[M_NEED_LOOK]))
    look_t = i32(jnp.maximum(memory[M_LOOK_T] - 1, 0))

    # wanted dish type
    alt = i32(jnp.where(last_d < 0, 0, 1 - jnp.clip(last_d, 0, 1)))
    side_c = jnp.clip(side_last, 0, 1)
    want_signal = i32(side_c ^ invert)
    # uncalibrated: trust the default mapping while the number of deliveries
    # without an observed flip stays below the trust limit (2 if the partner
    # has been seen at a canonical post facing the pot, else 1), then alternate
    noflip_eff = noflip + jnp.where(flipped == 0, 1, 0)
    trust_limit = jnp.where(canon == 1, 2, 1)
    want_uncal = jnp.where(last_d < 0, side_c, jnp.where(noflip_eff < trust_limit, side_c, alt))
    want = i32(jnp.where((blind == 1) | (seen == 0), alt,
                         jnp.where(calib == 1, want_signal, want_uncal)))
    belief = want
    other = 1 - belief
    use_look = (seen == 1) & (blind == 0)

    # ---- pot memory --------------------------------------------------------
    pot_states = []
    for p, (gr, gc) in enumerate(POT_POS):
        cell, vis = read_cell(obs, r, c, gr, gc)
        seen_state = jnp.concatenate([cell[34:38], cell[33:34], cell[44:45]])
        old = memory[M_POT + 6 * p:M_POT + 6 * p + 6]
        pred = old.at[5].set(jnp.maximum(old[5] - 1, 0))
        pred = pred.at[4].set(jnp.where(old[5] == 1, 1, old[4]))
        pot_states.append(jnp.where(vis, seen_state, pred).astype(I32))

    totals, ptypes, cookings, readys, idles = [], [], [], [], []
    for st in pot_states:
        t, pt, ck, rd, idl = pot_info(st)
        totals.append(t)
        ptypes.append(pt)
        cookings.append(ck)
        readys.append(rd)
        idles.append(idl)
    totals = jnp.stack(totals)
    ptypes = jnp.stack(ptypes)
    cookings = jnp.stack(cookings)
    readys = jnp.stack(readys)
    idles = jnp.stack(idles)

    # ---- counter memory ----------------------------------------------------
    ctr_states = []
    for k, (gr, gc) in enumerate(CTR_POS):
        cell, vis = read_cell(obs, r, c, gr, gc)
        old = memory[M_CTR + 6 * k:M_CTR + 6 * k + 6]
        ctr_states.append(jnp.where(vis, cell[CH_ITEM:CH_ITEM + 6], old).astype(I32))
    ctr_kind, ctr_dish = [], []
    for st in ctr_states:
        kd, _, dt = item_kind(st)
        ctr_kind.append(kd)
        ctr_dish.append(dt)
    ctr_kind = jnp.stack(ctr_kind)
    ctr_dish = jnp.stack(ctr_dish)
    ctr_free = ctr_kind == 0
    top_free = [ctr_free[0], ctr_free[1]]
    free_top = first_index(top_free)
    has_free_top = free_top >= 0
    free_any = first_index([ctr_free[k] for k in range(4)])
    has_free_any = free_any >= 0

    def ctr_target(idx):
        ic = jnp.clip(idx, 0, 3)
        return (CTR_ACCESS[ic, 0], CTR_ACCESS[ic, 1], CTR_ACCESS[ic, 2], i32(A_INTERACT))

    stash_top_target = ctr_target(free_top)
    stash_any_target = ctr_target(free_any)

    # ---- derived pot selections -------------------------------------------
    def pot_of(flags, prefer):
        idx = select_pot(flags, prefer)
        ok = idx >= 0
        ic = jnp.clip(idx, 0, 1)
        return ok, idx, (POT_ACCESS[ic, 0], POT_ACCESS[ic, 1], POT_ACCESS[ic, 2])

    valid_type = (ptypes == 0) | (ptypes == 1)
    ready_b_ok, ready_b_idx, ready_b_tgt = pot_of(readys & (ptypes == belief), belief)
    ready_o_ok, ready_o_idx, ready_o_tgt = pot_of(readys & (ptypes == other), other)
    cook_b_ok, cook_b_idx, cook_b_tgt = pot_of(cookings & (ptypes == belief), belief)
    cook_any_ok, cook_any_idx, cook_any_tgt = pot_of(cookings, belief)
    garb_ok, garb_idx, garb_tgt = pot_of(readys & (ptypes == 2), belief)
    # wait at the cooking pot with the smallest timer (prefer the wanted type)
    timers = jnp.stack([pot_states[0][5], pot_states[1][5]])
    t_adj = jnp.where(cookings, timers, 999) - jnp.where(ptypes == belief, 1, 0)
    soon_idx = i32(jnp.argmin(t_adj))
    soon_tgt = (POT_ACCESS[soon_idx, 0], POT_ACCESS[soon_idx, 1], POT_ACCESS[soon_idx, 2])

    # inventory per type: pots holding that type (any state) + counter dishes
    def inventory(t):
        return (jnp.sum((ptypes == t).astype(I32))
                + jnp.sum(((ctr_kind == 3) & (ctr_dish == t)).astype(I32)))

    inv_b = inventory(belief)
    inv_o = inventory(other)

    partial_any = idles & (totals < 3) & (totals > 0) & valid_type
    partial_idx = select_pot(partial_any, belief)
    partial_type = jnp.clip(ptypes[jnp.clip(partial_idx, 0, 1)], 0, 1)
    balanced_type = i32(jnp.where(inv_b <= inv_o, belief, other))
    # finish a partial pot first, otherwise load the type with the lower inventory
    load_type = i32(jnp.where(partial_idx >= 0, partial_type, balanced_type))

    def load_pot(k):
        partial = idles & (totals < 3) & (ptypes == k)
        empty = idles & (totals == 0)
        idx_p = select_pot(partial, k)
        idx_e = select_pot(empty, k)
        idx = jnp.where(idx_p >= 0, idx_p, idx_e)
        ok = idx >= 0
        ic = jnp.clip(idx, 0, 1)
        return ok, idx, (POT_ACCESS[ic, 0], POT_ACCESS[ic, 1], POT_ACCESS[ic, 2])

    load_ok, load_idx, load_tgt = load_pot(load_type)
    held_ing = jnp.clip(h_ing, 0, 1)
    load_h_ok, load_h_idx, load_h_tgt = load_pot(held_ing)

    def fetch_target(k, pot_idx):
        pot_col = POT_ACCESS[jnp.clip(pot_idx, 0, 1), 1]
        ar = jnp.where(k == 0, 1, 2)
        cost_l = jnp.abs(r - ar) + jnp.abs(c - 1) + jnp.abs(1 - pot_col)
        cost_r = jnp.abs(r - ar) + jnp.abs(c - 4) + jnp.abs(4 - pot_col)
        use_left = cost_l <= cost_r
        return (i32(ar), i32(jnp.where(use_left, 1, 4)),
                i32(jnp.where(use_left, F_LEFT, F_RIGHT)), i32(A_INTERACT))

    plate_target = (i32(1), i32(3), i32(F_UP), i32(A_INTERACT))
    deliver_target = (i32(1), i32(2), i32(F_UP), i32(A_INTERACT))
    wait_center = (i32(2), i32(2), f, i32(A_STAY))

    # ---- decision: empty hands --------------------------------------------
    ctr_dish_b = (ctr_kind == 3) & (ctr_dish == belief)
    dish_ctr = first_index([ctr_dish_b[k] for k in range(4)])
    dish_ctr_tgt = ctr_target(dish_ctr)

    # lowest priority first, overridden by higher priorities
    t_empty = wait_center
    t_empty = pick(garb_ok, plate_target, t_empty)
    t_empty = pick(cook_any_ok & (cook_b_ok | has_free_top), plate_target, t_empty)
    t_empty = pick(ready_o_ok & has_free_top, plate_target, t_empty)
    t_empty = pick(load_ok, fetch_target(load_type, load_idx), t_empty)
    t_empty = pick(ready_b_ok, plate_target, t_empty)
    t_empty = pick(dish_ctr >= 0, dish_ctr_tgt, t_empty)
    t_empty = pick((need_look == 1) & (look_t > 0) & use_look, wait_center, t_empty)

    # ---- decision: holding an empty plate ---------------------------------
    t_plate = pick(has_free_any, stash_any_target, wait_center)
    t_plate = pick(cook_any_ok, soon_tgt + (i32(A_STAY),), t_plate)
    t_plate = pick(garb_ok, garb_tgt + (i32(A_INTERACT),), t_plate)
    t_plate = pick(ready_o_ok & has_free_top, ready_o_tgt + (i32(A_INTERACT),), t_plate)
    t_plate = pick(ready_b_ok, ready_b_tgt + (i32(A_INTERACT),), t_plate)

    # ---- decision: holding a raw ingredient --------------------------------
    t_ing = pick(has_free_any, stash_any_target, wait_center)
    t_ing = pick(load_h_ok & (h_ing <= 1), load_h_tgt + (i32(A_INTERACT),), t_ing)

    # ---- decision: holding a cooked dish -----------------------------------
    deliver_ok = h_dish == belief
    t_dish = pick(has_free_any, stash_any_target, deliver_target)
    t_dish = pick(has_free_top, stash_top_target, t_dish)
    t_dish = pick(deliver_ok, deliver_target, t_dish)

    target = pick(h_kind == 3, t_dish,
                  pick(h_kind == 2, t_plate,
                       pick(h_kind == 1, t_ing, t_empty)))
    tr, tc, tf, do = target
    action = navigate(r, c, f, tr, tc, tf, do)

    delivered = (action == A_INTERACT) & (r == 1) & (c == 2) & (f == F_UP) & (h_kind == 3)
    need_look = i32(jnp.where(delivered, 1, need_look))
    look_t = i32(jnp.where(delivered, LOOK_STEPS, look_t))
    # count this inter-delivery period as a no-flip only if a previous delivery exists
    noflip = i32(jnp.where(delivered & (last_d >= 0) & (flipped == 0), noflip + 1, noflip))
    last_d = i32(jnp.where(delivered & (h_dish <= 1), h_dish, last_d))
    pre_side = i32(jnp.where(delivered, side_last, pre_side))
    flipped = i32(jnp.where(delivered, 0, flipped))
    changes = i32(jnp.where(delivered, 0, changes))

    head = jnp.stack([side_last, seen, need_look, look_t, memory[M_STEP] + 1,
                      last_d, pre_side, flipped, changes, calib, invert, blind, want,
                      noflip, canon, i32(0)])
    new_memory = jnp.concatenate([head] + pot_states + ctr_states)
    new_memory = new_memory.astype(I32)

    trace = {
        "pos": jnp.stack([r, c, f]),
        "want": want,
        "mode": jnp.stack([seen, calib, invert, blind, last_d, changes, noflip, canon, flipped]),
        "held": jnp.stack([h_kind, h_ing, h_dish]),
        "target": jnp.stack([tr, tc, tf, do]),
        "pots": jnp.stack(pot_states),
    }
    return action, new_memory, trace


# ----------------------------------------------------------------------------
# agent 1: recipe signaller in the bottom room
# ----------------------------------------------------------------------------

def init_agent_1():
    return jnp.zeros((4,), dtype=I32)


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, init_agent_1(), memory)
    obs = obs.astype(I32)

    r, c = locate(obs, CH_RECIPE_IND, 6, 2)
    r = jnp.clip(r, 4, 5)
    c = jnp.clip(c, 1, 4)
    f = jnp.argmax(obs[2, 2, 1:5]).astype(I32)

    cell, vis = read_cell(obs, r, c, 6, 2)
    recipe_seen = jnp.where(cell[CH_RECIPE + 2] > 0, 0, 1)
    recipe = i32(jnp.where(vis, recipe_seen, memory[0]))

    tc = i32(jnp.where(recipe == 0, 1, 3))
    action = navigate(r, c, f, i32(4), tc, i32(F_UP), i32(A_STAY))

    new_memory = memory.at[0].set(recipe).at[1].set(memory[1] + 1)
    trace = {"pos": jnp.stack([r, c, f]), "recipe": recipe}
    return action, new_memory, trace
