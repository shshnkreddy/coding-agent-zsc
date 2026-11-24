"""Scripted cooperative policy for Overcooked V2 demo_cook_simple.

Geometry (fixed 5x11 map):

    WWWWWR2W0WW
    0      W  B
    W     APA X
    1      W  B
    WWWWWR2W1WW

agent_0 lives in the left room (cols 1-6).  It is the only seat that can see the
always-on recipe indicators (0,5)/(4,5), and it can reach ingredient supplies
0 (1,0), 1 (3,0) and the shared pot (2,7) from (2,6).
agent_1 lives in the right room (cols 8-9).  It can reach supplies 0 (0,8),
1 (4,8), the pot from (2,8), plates (1,10)/(3,10) and delivery (2,10).

Protocol (evidence based, no private convention):
  * agent_0 reads the recipe and puts the recipe ingredient into the pot.  The pot
    contents are visible to agent_1 from anywhere in the right room.
  * agent_1 infers the target ingredient from the pot contents (or, if the pot is
    empty, from an ingredient left on the pass-through counters (1,7)/(3,7) or the
    ingredient the partner is visibly carrying), fills the pot, fetches a plate
    while the pot cooks, plates the dish and delivers it.
  * agent_0 keeps contributing ingredients until the pot is full, so a passive
    partner that only plates/delivers still gets cooked dishes.
  * agent_1 sets provably wrong dishes (mixed, or containing ingredient 2) aside on
    a counter instead of delivering them (-20).
"""

import jax
import jax.numpy as jnp

from custom_primitives import ROOM0, ROOM1, SIG0, SIG1, DIST, NEXT

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'pot_signal_roles_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3
DIR_TO_ACTION = [A_UP, A_DOWN, A_RIGHT, A_LEFT]
BIG = 99

# Goal encoding: (stand_row, stand_col, facing_dir).
G0_POT = (2, 6, D_RIGHT)
G0_SUP = [(1, 1, D_LEFT), (3, 1, D_LEFT), (1, 6, D_UP)]  # supply for ingredient 0, 1, 2
G0_LOOK = (2, 5, D_RIGHT)
# storage counters for agent_0: (counter cell, stand cell + facing)
STASH0_CELLS = [(0, 4), (0, 3), (0, 2), (0, 1), (4, 4), (4, 3), (4, 2), (4, 1), (2, 0), (1, 7), (3, 7)]
STASH0_GOALS = [(1, 4, D_UP), (1, 3, D_UP), (1, 2, D_UP), (1, 1, D_UP),
                (3, 4, D_DOWN), (3, 3, D_DOWN), (3, 2, D_DOWN), (3, 1, D_DOWN),
                (2, 1, D_LEFT), (1, 6, D_RIGHT), (3, 6, D_RIGHT)]
G1_POT = (2, 8, D_LEFT)
G1_DELIVER = (2, 9, D_RIGHT)
G1_PLATE = [(1, 9, D_RIGHT), (3, 9, D_RIGHT)]
G1_SUP = [(1, 8, D_UP), (3, 8, D_DOWN)]  # supply for ingredient 0, 1
STASH1_CELLS = [(0, 9), (4, 9), (1, 7), (3, 7)]
STASH1_GOALS = [(1, 9, D_UP), (3, 9, D_DOWN), (1, 8, D_LEFT), (3, 8, D_LEFT)]

ITEM_EMPTY, ITEM_PLATE, ITEM_DISH = -1, 3, 4


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def consts():
    """Build all constant JAX arrays (called inside traced functions)."""
    return dict(
        sig0=_i32(SIG0), sig1=_i32(SIG1), cells0=_i32(ROOM0), cells1=_i32(ROOM1),
        dist=_i32(DIST), nxt=_i32(NEXT), dir2act=_i32(DIR_TO_ACTION),
        pot=_i32([2, 7]),
        g0_pot=_i32(G0_POT), g0_sup=_i32(G0_SUP), g0_look=_i32(G0_LOOK),
        stash0_cells=_i32(STASH0_CELLS), stash0_goals=_i32(STASH0_GOALS),
        g1_pot=_i32(G1_POT), g1_deliver=_i32(G1_DELIVER), g1_plate=_i32(G1_PLATE),
        g1_sup=_i32(G1_SUP), stash1_cells=_i32(STASH1_CELLS), stash1_goals=_i32(STASH1_GOALS),
    )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------
def localize(obs, sigs, cells):
    static = obs[:, :, 20:29]
    match = jnp.all(sigs == static[None], axis=(1, 2, 3))
    idx = jnp.argmax(match)
    return cells[idx], jnp.any(match)


def cell_at(obs, pos, g):
    """Return the 40-channel vector of global cell g (zeros if out of view) and visibility."""
    idx = g - pos + 2
    valid = jnp.all(idx >= 0) & jnp.all(idx < 5)
    ci = jnp.clip(idx, 0, 4)
    vec = obs[ci[0], ci[1]]
    return jnp.where(valid, vec, 0), valid


def item_code(block):
    """block = 5 channels [plate, cooked, c0, c1, c2] -> item code."""
    counts = block[2:5]
    has_ing = jnp.sum(counts) > 0
    code = jnp.where(block[1] > 0, ITEM_DISH,
                     jnp.where(block[0] > 0, ITEM_PLATE,
                               jnp.where(has_ing, jnp.argmax(counts), ITEM_EMPTY)))
    return _i32(code)


def self_state(obs):
    me = obs[2, 2]
    facing = jnp.argmax(me[1:5])
    held = item_code(me[5:10])
    return _i32(facing), held


def partner_state(obs):
    mask = obs[:, :, 10]
    vis = jnp.any(mask > 0)
    feats = jnp.sum(obs[:, :, 10:20] * mask[:, :, None], axis=(0, 1))
    held = item_code(feats[5:10])
    return vis, jnp.where(vis, held, ITEM_EMPTY)


def partner_pose(obs, pos):
    """Partner global (row, col) and facing dir (valid only when partner visible)."""
    mask = obs[:, :, 10]
    flat = jnp.argmax(mask.reshape(-1))
    li, lj = flat // 5, flat % 5
    gpos = jnp.stack([pos[0] + li - 2, pos[1] + lj - 2])
    feats = jnp.sum(obs[:, :, 11:15] * mask[:, :, None], axis=(0, 1))
    return _i32(gpos), _i32(jnp.argmax(feats))


def recipe_state(obs):
    ind = obs[:, :, 23]
    vis = jnp.any(ind > 0)
    counts = jnp.sum(obs[:, :, 36:39] * ind[:, :, None], axis=(0, 1))
    return vis, _i32(jnp.argmax(counts))


def pot_state(obs, pos, C):
    vec, vis = cell_at(obs, pos, C['pot'])
    counts = vec[31:34]
    n = jnp.sum(counts)
    cooked = vec[30] > 0
    timer = vec[39]
    ntypes = jnp.sum(counts > 0)
    ptype = _i32(jnp.argmax(counts))
    return dict(vis=vis, counts=counts, n=_i32(n), cooked=cooked, timer=_i32(timer),
                idle=(timer == 0) & (~cooked), pure=(ntypes == 1), ptype=ptype,
                bad=(ntypes > 1) | (counts[2] > 0))


def update_counter_memory(obs, pos, cells, mem):
    """Refresh remembered item codes for the given counter cells when visible."""
    def one(cell, old):
        vec, vis = cell_at(obs, pos, cell)
        return jnp.where(vis, item_code(vec[29:34]), old)
    return jax.vmap(one)(cells, mem)


def nav_action(C, pos, facing, goal, interact):
    """Move toward goal stand cell, then turn to goal facing, then interact / stay."""
    tr, tc, tf = goal[0], goal[1], goal[2]
    at = (pos[0] == tr) & (pos[1] == tc)
    move = C['nxt'][pos[0], pos[1], tr, tc]
    turn = C['dir2act'][tf]
    act = jnp.where(~at, move,
                    jnp.where(facing != tf, turn,
                              jnp.where(interact, A_INTERACT, A_STAY)))
    return _i32(act)


def goal_dist(C, pos, goal):
    return C['dist'][pos[0], pos[1], goal[0], goal[1]]


def pick_nearest(C, pos, goals, valid):
    """goals: [K,3]; valid: [K] bool -> (goal[3], any_valid)."""
    d = jax.vmap(lambda g: goal_dist(C, pos, g))(goals)
    d = jnp.where(valid, d, BIG + 1)
    k = jnp.argmin(d)
    return goals[k], jnp.any(valid)


def pick_round_trip(C, pos, goals, valid, home):
    """Choose the goal minimising dist(pos, goal) + dist(goal, home) (ties -> lowest index)."""
    d = jax.vmap(lambda g: goal_dist(C, pos, g) + goal_dist(C, g, home))(goals)
    d = jnp.where(valid, d, 4 * BIG)
    k = jnp.argmin(d)
    return goals[k], jnp.any(valid), d[k]


def sel_goal(cond, a, b):
    return jnp.where(cond, a, b)


# ---------------------------------------------------------------------------
# agent_0 : left room, recipe reader / pot signaller / ingredient helper
# ---------------------------------------------------------------------------
def init_agent_0():
    return {
        'recipe': _i32(-1),
        'since_dish': _i32(100),
        'empty_since': _i32(100),
        'stuck': _i32(0),
        'build': _i32(0),
        'stash': jnp.full((len(STASH0_CELLS),), ITEM_EMPTY, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    C = consts()
    init = init_agent_0()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    pos, _ = localize(obs, C['sig0'], C['cells0'])
    facing, held = self_state(obs)
    p_vis, p_held = partner_state(obs)
    rec_vis, rec_type = recipe_state(obs)
    pot = pot_state(obs, pos, C)

    recipe = jnp.where(rec_vis, rec_type, mem['recipe'])
    stash = update_counter_memory(obs, pos, C['stash0_cells'], mem['stash'])

    # Delivery-in-flight tracking: a dish was seen (partner holding it, or pot ready).
    saw_dish = (p_vis & (p_held == ITEM_DISH)) | (pot['vis'] & pot['cooked'])
    since_dish = jnp.where(saw_dish, 0, jnp.minimum(mem['since_dish'] + 1, 100))
    empty_since = jnp.where(pot['vis'],
                            jnp.where(pot['n'] == 0, jnp.minimum(mem['empty_since'] + 1, 100), 0),
                            mem['empty_since'])
    safe_empty = (p_vis & (p_held != ITEM_DISH)) | ((since_dish >= 4) & (empty_since >= 4))

    holding_ing = (held >= 0) & (held <= 2)
    holding_other = (held == ITEM_PLATE) | (held == ITEM_DISH)
    recipe_known = recipe >= 0

    # pot stuck with a pure wrong ingredient (partner guessed wrong): flush after a while
    wrong_pure = pot['vis'] & pot['idle'] & (pot['n'] > 0) & pot['pure'] & (pot['ptype'] != recipe)
    stuck = jnp.where(wrong_pure, jnp.minimum(mem['stuck'] + 1, 100), 0)

    # indicator (0,5)/(4,5) is visible from any cell in columns 3-7: walk right along this row
    goal_look = jnp.stack([pos[0], _i32(3), _i32(D_RIGHT)])
    goal_pot = C['g0_pot']

    # fetch recipe ingredient: supply or a stash counter holding it, whichever gives the
    # shorter round trip back to the pot (ties -> stash, which also clears counters)
    r_idx = jnp.clip(recipe, 0, 2)
    sup_goal = C['g0_sup'][r_idx]
    sup_cost = goal_dist(C, pos, sup_goal) + goal_dist(C, sup_goal, goal_pot)
    stash_has = (stash == recipe) & recipe_known
    stash_goal, stash_any, stash_cost = pick_round_trip(C, pos, C['stash0_goals'], stash_has, goal_pot)
    use_stash = stash_any & (stash_cost <= sup_cost)
    goal_fetch = sel_goal(use_stash, stash_goal, sup_goal)

    # drop held (wrong) item: empty stash counter with the shortest detour on the way to the
    # pot (so stored items sit close to the pot); avoid pass-through cells if possible
    stash_empty = stash == ITEM_EMPTY
    pref = stash_empty & jnp.asarray([True] * 9 + [False, False])
    drop_goal_a, drop_any_a, _ = pick_round_trip(C, pos, C['stash0_goals'], pref, goal_pot)
    drop_goal_b, drop_any_b, _ = pick_round_trip(C, pos, C['stash0_goals'], stash_empty, goal_pot)
    drop_goal = sel_goal(drop_any_a, drop_goal_a, sel_goal(drop_any_b, drop_goal_b, goal_pot))
    drop_any = drop_any_a | drop_any_b

    # insertion allowed?
    pot_accepts = pot['vis'] & pot['idle'] & (pot['n'] < 3)
    same_type = pot['pure'] & (pot['ptype'] == held)
    allow_insert = pot_accepts & (((pot['n'] == 0) & safe_empty) | same_type | (stuck >= 25))

    # --- build mode: while the pot cooks, stash the alternate ingredient near the pot ---
    other = jnp.where(recipe == 0, 1, 0)
    stash_has_other = jnp.any(stash == other) & recipe_known
    pot_busy_vis = pot['vis'] & (pot['timer'] > 0)
    pot_right = pot['pure'] & (pot['ptype'] == recipe)
    build_start = pot_busy_vis & pot_right & (pot['timer'] >= 12) & recipe_known & (~stash_has_other) & \
        ((held == recipe) | (held == ITEM_EMPTY))
    build = jnp.where((~recipe_known) | stash_has_other | (pot['vis'] & pot['idle']), 0,
                      jnp.where(build_start, 1, mem['build']))
    in_build = build > 0
    goal_fetch_other = C['g0_sup'][jnp.clip(other, 0, 2)]

    #  case A: holding nothing
    a_goal = sel_goal(recipe_known, sel_goal(in_build, goal_fetch_other, goal_fetch), goal_look)
    a_int = recipe_known
    #  case B: holding recipe ingredient -> go to pot, insert when allowed (or stash it in build mode)
    b_goal = sel_goal(in_build & drop_any, drop_goal, goal_pot)
    b_int = jnp.where(in_build & drop_any, True, allow_insert)
    #  case C: holding a wrong ingredient or a plate/dish -> stash it
    c_goal = drop_goal
    c_int = drop_any

    is_b = holding_ing & recipe_known & (held == recipe)
    is_c = (holding_ing & recipe_known & (held != recipe)) | holding_other
    goal = sel_goal(is_b, b_goal, sel_goal(is_c, c_goal, a_goal))
    interact = jnp.where(is_b, b_int, jnp.where(is_c, c_int, a_int))
    unknown_hold = holding_ing & (~recipe_known)
    goal = sel_goal(unknown_hold, goal_look, goal)
    interact = jnp.where(unknown_hold, False, interact)

    action = nav_action(C, pos, facing, goal, interact)

    new_mem = {
        'recipe': _i32(recipe),
        'since_dish': _i32(since_dish),
        'empty_since': _i32(empty_since),
        'stuck': _i32(stuck),
        'build': _i32(build),
        'stash': _i32(stash),
    }
    trace = {
        'pos': pos, 'held': held, 'recipe': _i32(recipe), 'pot_n': pot['n'],
        'mode': _i32(jnp.where(is_b, 1, jnp.where(is_c, 2, 0))), 'goal': goal,
        'allow_insert': allow_insert, 'since_dish': _i32(since_dish), 'build': _i32(build),
        'stash': _i32(stash),
    }
    return action, new_mem, trace


# ---------------------------------------------------------------------------
# agent_1 : right room, pot filler / plater / deliverer
# ---------------------------------------------------------------------------
def init_agent_1():
    return {
        'dish_bad': _i32(0),
        'suspect': _i32(0),
        'tgt_mem': _i32(-1),
        'counters': jnp.full((len(STASH1_CELLS),), ITEM_EMPTY, dtype=jnp.int32),
        'cnt_stale': jnp.zeros((len(STASH1_CELLS),), dtype=jnp.int32),
        'pot_n_prev': _i32(0),
        'last_insert': _i32(0),
        'partner_inserts': _i32(0),
    }


def agent_1_act(obs, reset, memory, rng):
    C = consts()
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    pos, _ = localize(obs, C['sig1'], C['cells1'])
    facing, held = self_state(obs)
    p_vis, p_held = partner_state(obs)
    p_pos, p_dir = partner_pose(obs, pos)
    pot = pot_state(obs, pos, C)
    counters = update_counter_memory(obs, pos, C['stash1_cells'], mem['counters'])

    # Partner seen carrying a (recipe-eligible) ingredient that does not match the pot
    # contents while the pot is non-empty is evidence the pot is wrong: accumulate sightings
    # until the pot is emptied.
    mismatch = p_vis & (p_held >= 0) & (p_held <= 1) & (pot['n'] > 0) & pot['pure'] & \
        (p_held != pot['ptype'])
    suspect = jnp.where(pot['n'] == 0, 0,
                        jnp.where(mismatch, jnp.minimum(mem['suspect'] + 1, 100), mem['suspect']))
    p_at_pot = p_vis & (p_pos[0] == 2) & (p_pos[1] == 6)

    holding_ing = (held >= 0) & (held <= 2)
    holding_plate = held == ITEM_PLATE
    holding_dish = held == ITEM_DISH
    holding_none = held == ITEM_EMPTY

    # --- target ingredient inference (evidence only) ---
    # Counter items that were already present when we last delivered predate the recipe
    # resample and are not evidence (they may still be used as a source once the target is known).
    stale = mem['cnt_stale'] > 0
    pot_has = pot['n'] > 0
    pot_t = jnp.where(pot['ptype'] == 2, 0, pot['ptype'])
    # Has the partner ever put an ingredient into the pot directly?  If so, items it leaves on
    # the pass-through counters are its own stash (possibly the alternate type), not hand-offs.
    partner_inserted_now = (pot['n'] - mem['pot_n_prev'] - mem['last_insert']) > 0
    partner_inserts = jnp.maximum(mem['partner_inserts'], jnp.where(partner_inserted_now, 1, 0))
    pass_items = counters[2:4]
    pass_valid = (pass_items >= 0) & (pass_items <= 1) & (~stale[2:4]) & (partner_inserts == 0)
    pass_t = jnp.where(pass_valid[0], pass_items[0], pass_items[1])
    pass_any = jnp.any(pass_valid)
    partner_t_valid = p_at_pot & (p_dir == D_RIGHT) & (p_held >= 0) & (p_held <= 1)
    target_ev = jnp.where(pot_has, pot_t,
                          jnp.where(pass_any, pass_t,
                                    jnp.where(partner_t_valid, p_held, -1)))
    # Remember a fresh counter item's type until the pot has contents (we may have picked it up).
    target = _i32(jnp.where(target_ev >= 0, target_ev, mem['tgt_mem']))
    target_known = target >= 0
    # stale recipe-eligible item on a pass-through counter blocking a counter-passing partner
    stale_pass = (pass_items >= 0) & (pass_items <= 1) & stale[2:4]
    stale_pass_any = jnp.any(stale_pass)

    goal_pot = C['g1_pot']
    goal_deliver = C['g1_deliver']

    plate_goal, _ = pick_nearest(C, pos, C['g1_plate'], jnp.asarray([True, True]))
    t_idx = jnp.clip(target, 0, 1)
    sup_goal = C['g1_sup'][t_idx]
    cnt_has = (counters == target) & target_known
    cnt_goal, cnt_any, cnt_cost = pick_round_trip(C, pos, C['stash1_goals'], cnt_has, goal_pot)
    sup_cost = goal_dist(C, pos, sup_goal) + goal_dist(C, sup_goal, goal_pot)
    # consume counter items (partner hand-offs, parked leftovers) unless clearly more expensive
    use_cnt = cnt_any & (cnt_cost <= sup_cost + 1)
    goal_fetch = sel_goal(use_cnt, cnt_goal, sup_goal)
    cnt_empty = counters == ITEM_EMPTY
    drop_pref = cnt_empty & jnp.asarray([True, True, False, False])
    drop_goal_a, drop_any_a = pick_nearest(C, pos, C['stash1_goals'], drop_pref)
    drop_goal_b, drop_any_b = pick_nearest(C, pos, C['stash1_goals'], cnt_empty)
    drop_goal = sel_goal(drop_any_a, drop_goal_a, drop_goal_b)
    drop_any = drop_any_a | drop_any_b

    dish_bad = mem['dish_bad'] > 0

    # 1. holding dish
    d1_goal = sel_goal(dish_bad & drop_any, drop_goal, goal_deliver)
    d1_int = jnp.asarray(True)
    # 2. pot cooked or cooking
    busy = pot['cooked'] | (pot['timer'] > 0)
    d2_goal = sel_goal(holding_plate, goal_pot, sel_goal(holding_none, plate_goal, drop_goal))
    d2_int = jnp.where(holding_plate, pot['cooked'], jnp.where(holding_none, True, drop_any))
    # 3. pot idle (n < 3)
    match = target_known & (held == target)
    # holding an unidentified ingredient: park it on a right-room counter if one is free, else
    # wait at the pot; holding a known-wrong ingredient: drop it on any free counter
    park_goal = sel_goal(drop_any_a, drop_goal_a, goal_pot)
    park_int = drop_any_a
    wrong_goal = sel_goal(drop_any, drop_goal, goal_pot)
    d3_ing_goal = sel_goal(match, goal_pot, sel_goal(target_known, wrong_goal, park_goal))
    d3_ing_int = jnp.where(match, True, jnp.where(target_known, drop_any, park_int))
    # holding nothing: fetch the target; if unknown, clear a stale pass-through item (so a
    # counter-passing partner can place a fresh one); else wait
    stale_goal, _ = pick_nearest(C, pos, C['stash1_goals'],
                                 jnp.concatenate([jnp.zeros((2,), dtype=bool), stale_pass]))
    clear_ok = stale_pass_any & (partner_inserts == 0)
    d3_none_goal = sel_goal(target_known, goal_fetch, sel_goal(clear_ok, stale_goal, goal_pot))
    d3_none_int = target_known | clear_ok
    d3_goal = sel_goal(holding_ing, d3_ing_goal, sel_goal(holding_none, d3_none_goal, drop_goal))
    d3_int = jnp.where(holding_ing, d3_ing_int, jnp.where(holding_none, d3_none_int, drop_any))

    goal = sel_goal(holding_dish, d1_goal, sel_goal(busy, d2_goal, d3_goal))
    interact = jnp.where(holding_dish, d1_int, jnp.where(busy, d2_int, d3_int))

    action = nav_action(C, pos, facing, goal, interact)

    about_to_plate = pot['cooked'] & holding_plate & (action == A_INTERACT) & \
        (pos[0] == 2) & (pos[1] == 8) & (facing == D_LEFT)
    pot_bad = pot['bad'] | (suspect >= 2)
    new_bad = jnp.where(about_to_plate, jnp.where(pot_bad, 1, 0),
                        jnp.where(holding_dish, mem['dish_bad'], 0))
    delivering = holding_dish & (~dish_bad) & (action == A_INTERACT) & \
        (pos[0] == 2) & (pos[1] == 9) & (facing == D_RIGHT)
    cnt_stale = jnp.where(delivering, (counters != ITEM_EMPTY).astype(jnp.int32),
                          jnp.where(counters == ITEM_EMPTY, 0, mem['cnt_stale']))
    tgt_mem = jnp.where(delivering | pot_has, -1, jnp.where(pass_any, pass_t, mem['tgt_mem']))
    inserting_now = holding_ing & (action == A_INTERACT) & (pos[0] == 2) & (pos[1] == 8) & \
        (facing == D_LEFT) & pot['idle'] & (pot['n'] < 3)

    new_mem = {
        'dish_bad': _i32(new_bad),
        'suspect': _i32(suspect),
        'tgt_mem': _i32(tgt_mem),
        'counters': _i32(counters),
        'cnt_stale': _i32(cnt_stale),
        'pot_n_prev': _i32(pot['n']),
        'last_insert': _i32(jnp.where(inserting_now, 1, 0)),
        'partner_inserts': _i32(partner_inserts),
    }
    trace = {
        'pos': pos, 'held': held, 'target': target, 'pot_n': pot['n'], 'pot_timer': pot['timer'],
        'pot_cooked': pot['cooked'], 'goal': goal, 'dish_bad': _i32(new_bad), 'suspect': _i32(suspect),
    }
    return action, new_mem, trace
