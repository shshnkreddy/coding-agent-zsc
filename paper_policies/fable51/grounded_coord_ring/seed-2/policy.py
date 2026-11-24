"""grounded_coord_ring policy.

agent_1 (inner room, only seat that can reach the pot) is a self-sufficient
cook: it infers the recipe from any grounded evidence it can see (active button
display, partner-held ingredient, fresh ingredient on a shared counter) and
presses the button itself when nothing arrives.  agent_0 (outer ring) reads
the free always-on indicators, stocks the shared counters nearest the pot room
with the correct ingredient, clears stale wrong ingredients, shows the correct
ingredient in hand where the inner seat can see it, and delivers/parks cooked
dishes found on shared counters.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'ring_cook_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1,
                   'max_height': 9, 'max_width': 9, 'num_ingredients': 3}

# ---------------------------------------------------------------- map tables
GRID = [
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
N = 9
NC = N * N
STATIC_CH = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'L': 4, 'B': 5, '0': 6, '1': 7, '2': 8}


def _idx(r, c):
    return r * N + c


def _floor(r, c):
    return 0 <= r < N and 0 <= c < N and GRID[r][c] in " A"



DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
BIG = 1000
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
OUTER_CELLS = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 1), (2, 7), (3, 1), (3, 7),
               (4, 1), (4, 7), (5, 1), (5, 7), (6, 1), (6, 7), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
               (7, 6), (7, 7)]
INNER_CELLS = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5)]
SHARED = [((2, 3), (3, 3), UP, (1, 3), DOWN),
          ((2, 5), (3, 5), UP, (1, 5), DOWN),
          ((6, 3), (5, 3), DOWN, (7, 3), UP),
          ((6, 5), (5, 5), DOWN, (7, 5), UP)]
SHARED_CELLS = [(2, 3), (2, 5), (6, 3), (6, 5)]
R_CELLS = [(0, 4), (4, 0), (4, 8), (8, 4)]
PARK_CELLS = [(1, 0), (7, 0)]
RESHOW_CELLS = [(2, 0), (6, 0)]
DUMPS = [((1, 2), DOWN, (2, 2)), ((1, 6), DOWN, (2, 6)), ((7, 2), UP, (6, 2)), ((7, 6), UP, (6, 6)),
         ((1, 2), UP, (0, 2)), ((7, 2), DOWN, (8, 2)), ((1, 1), UP, (0, 1)), ((7, 1), DOWN, (8, 1)),
         ((1, 6), UP, (0, 6)), ((7, 6), DOWN, (8, 6)), ((1, 7), UP, (0, 7)), ((7, 7), DOWN, (8, 7)),
         ((2, 7), RIGHT, (2, 8)), ((6, 7), RIGHT, (6, 8)), ((1, 7), RIGHT, (1, 8)), ((7, 7), RIGHT, (7, 8))]
NDUMP = 16
W_FIRST = 3      # steps to wait for evidence in the first cycle before pressing the button
W_LATER = 12     # steps to wait after a delivery when partner evidence has been arriving
W_SHORT = 3      # steps to wait after a delivery when partner evidence has not been arriving
D_COUNTER = 7    # counter evidence accepted only this long after a delivery
BUTTON_LEAD = 3  # start walking to the button this many steps before the deadline
PENDING_STEPS = 30   # outer: how long to keep showing the ingredient after a trigger
STOCK_WINDOW = 26    # outer: only start stocking/clearing trips this soon after a trigger
RESIGNAL_STEPS = 45  # outer: re-signal if no delivery/recipe change observed for this long
STUCK_STEPS = 40     # outer: partner never seen holding anything for this long -> press button
MAX_FAILED_PRESSES = 1

I32 = jnp.int32


def build_tables():
    """Build static templates and BFS tables (pure Python, bounded loops)."""
    templates = []
    for r in range(N):
        for c in range(N):
            crop = [[[0] * 9 for _ in range(5)] for _ in range(5)]
            for i in range(5):
                for j in range(5):
                    gr, gc = r + i - 2, c + j - 2
                    if 0 <= gr < N and 0 <= gc < N:
                        ch = GRID[gr][gc]
                        if ch in STATIC_CH:
                            crop[i][j][STATIC_CH[ch]] = 1
            templates.append(crop)
    dist = [[BIG] * NC for _ in range(NC)]
    nxt = [[4] * NC for _ in range(NC)]
    last = [[-1] * NC for _ in range(NC)]
    act2face = [2, 1, 3, 0]  # action RIGHT->facing RIGHT(2), DOWN->1, LEFT->3, UP->0
    for s in range(NC):
        sr, sc = divmod(s, N)
        if not _floor(sr, sc):
            continue
        d = {s: 0}
        first = {s: 4}
        lastd = {s: -1}
        frontier = [s]
        for _depth in range(40):
            new_frontier = []
            for u in frontier:
                ur, uc = divmod(u, N)
                for a in range(4):
                    dr, dc = DELTAS[a]
                    vr, vc = ur + dr, uc + dc
                    if _floor(vr, vc):
                        v = vr * N + vc
                        if v not in d:
                            d[v] = d[u] + 1
                            first[v] = a if u == s else first[u]
                            lastd[v] = act2face[a]
                            new_frontier.append(v)
            frontier = new_frontier
        for v in d:
            dist[s][v] = d[v]
            nxt[s][v] = first[v]
            last[s][v] = lastd[v]
    return (jnp.asarray(templates, dtype=I32), jnp.asarray(dist, dtype=I32), jnp.asarray(nxt, dtype=I32),
            jnp.asarray(last, dtype=I32))




def i32(x):
    return jnp.asarray(x, dtype=I32)



def consts():
    """All JAX constant tables, built at trace time (no module-scope calls)."""
    templates, dist, nxt, last = build_tables()
    K = {}
    K['TEMPLATES'] = templates
    K['DIST'] = dist
    K['NEXT'] = nxt
    K['LAST'] = last
    K['FACE2ACT'] = jnp.asarray([3, 1, 0, 2], dtype=I32)
    om = [False] * NC
    for r, c in OUTER_CELLS:
        om[r * N + c] = True
    im = [False] * NC
    for r, c in INNER_CELLS:
        im[r * N + c] = True
    K['OUTER_MASK'] = jnp.asarray(om)
    K['INNER_MASK'] = jnp.asarray(im)
    K['SHARED_IN_ACC'] = jnp.asarray([s[1][0] * N + s[1][1] for s in SHARED], dtype=I32)
    K['SHARED_IN_FACE'] = jnp.asarray([s[2] for s in SHARED], dtype=I32)
    K['SHARED_OUT_ACC'] = jnp.asarray([s[3][0] * N + s[3][1] for s in SHARED], dtype=I32)
    K['SHARED_OUT_FACE'] = jnp.asarray([s[4] for s in SHARED], dtype=I32)
    # inner
    K['IN_SUPPLY_ACC'] = jnp.asarray([_idx(3, 3), _idx(5, 3), _idx(3, 3)], dtype=I32)
    K['IN_SUPPLY_FACE'] = jnp.asarray([LEFT, LEFT, LEFT], dtype=I32)
    K['IN_POT_ACC'] = jnp.asarray([_idx(4, 3), _idx(3, 4), _idx(5, 4), _idx(4, 5)], dtype=I32)
    K['IN_POT_FACE'] = jnp.asarray([RIGHT, DOWN, UP, LEFT], dtype=I32)
    K['IN_PLATE_ACC'] = jnp.asarray([_idx(3, 5), _idx(5, 5)], dtype=I32)
    K['IN_PLATE_FACE'] = jnp.asarray([RIGHT, RIGHT], dtype=I32)
    K['IN_DELIV_ACC'] = jnp.asarray([_idx(4, 3), _idx(4, 5)], dtype=I32)
    K['IN_DELIV_FACE'] = jnp.asarray([LEFT, RIGHT], dtype=I32)
    K['IN_BUTTON_ACC'] = jnp.asarray([_idx(3, 4), _idx(5, 4)], dtype=I32)
    K['IN_BUTTON_FACE'] = jnp.asarray([UP, DOWN], dtype=I32)
    K['IN_WAIT'] = i32(_idx(4, 3))
    # outer
    K['OUT_SUPPLY_ACC'] = jnp.asarray([_idx(3, 1), _idx(5, 1), _idx(3, 1)], dtype=I32)
    K['OUT_SUPPLY_FACE'] = jnp.asarray([RIGHT, RIGHT, RIGHT], dtype=I32)
    K['OUT_SHOW'] = jnp.asarray([_idx(3, 1), _idx(5, 1), _idx(3, 1)], dtype=I32)
    K['OUT_DELIV_ACC'] = jnp.asarray([_idx(4, 1), _idx(4, 7)], dtype=I32)
    K['OUT_DELIV_FACE'] = jnp.asarray([RIGHT, LEFT], dtype=I32)
    K['OUT_RVIEW'] = i32(_idx(4, 1))
    K['OUT_BUTTON'] = i32(_idx(1, 4))
    K['PARK_ACC'] = jnp.asarray([_idx(1, 1), _idx(7, 1), _idx(1, 1)], dtype=I32)
    K['RESHOW_ACC'] = jnp.asarray([_idx(2, 1), _idx(6, 1), _idx(2, 1)], dtype=I32)
    K['PARK_FACE'] = jnp.asarray([LEFT, LEFT, LEFT], dtype=I32)
    K['DUMP_ACC'] = jnp.asarray([d[0][0] * N + d[0][1] for d in DUMPS], dtype=I32)
    K['DUMP_FACE'] = jnp.asarray([d[1] for d in DUMPS], dtype=I32)
    K['PRIMARY'] = jnp.asarray([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=jnp.bool_)
    return K


# ---------------------------------------------------------------- obs helpers
def read_cell(obs, r, c, gr, gc):
    """Full 40-channel vector at global (gr, gc); zeros if outside the view."""
    li = gr - r + 2
    lj = gc - c + 2
    valid = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    v = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4)]
    return jnp.where(valid, v, 0), valid


def item_code(v5):
    """0 empty, 1+j raw ingredient j, 4 plate, 5+j cooked dish dominated by j."""
    plate, cooked = v5[0], v5[1]
    counts = v5[2:5]
    j = jnp.argmax(counts).astype(I32)
    has_ing = jnp.sum(counts) > 0
    return jnp.where(cooked > 0, 5 + j, jnp.where(plate > 0, 4, jnp.where(has_ing, 1 + j, 0))).astype(I32)


def recipe_of(v5):
    """Recipe id (0/1/2) from a recipe display block, -1 if nothing displayed."""
    counts = v5[2:5]
    return jnp.where(jnp.sum(counts) > 0, jnp.argmax(counts), -1).astype(I32)


def localize(K, obs, mask):
    static = obs[:, :, 20:29]
    diff = jnp.sum(jnp.abs(K['TEMPLATES'] - static[None]), axis=(1, 2, 3))
    diff = diff + jnp.where(mask, 0, 100000)
    pos = jnp.argmin(diff).astype(I32)
    return pos, pos // N, pos % N


def act_toward(K, pos, facing, access, face_dir, interact):
    """Move to access cell, turn to face_dir, then INTERACT (or STAY if not interact)."""
    at = pos == access
    move = K['NEXT'][pos, access]
    turn = K['FACE2ACT'][face_dir]
    a = jnp.where(at,
                  jnp.where(interact > 0, jnp.where(facing == face_dir, 5, turn), 4),
                  move)
    return a.astype(I32)


def nearest(K, pos, access_arr, face_arr, valid, facing=None):
    d = K['DIST'][pos, access_arr] + jnp.where(valid, 0, 10000)
    if facing is not None:
        arrive = K['LAST'][pos, access_arr]
        arrive = jnp.where(arrive < 0, facing, arrive)
        d = d + jnp.where(arrive != face_arr, 1, 0)
    i = jnp.argmin(d)
    return access_arr[i], face_arr[i], d[i] < 10000


def choose(default, pairs):
    """pairs: list of (cond, goal) in priority order; goal = (access, face, interact)."""
    goal = default
    for cond, g in reversed(pairs):
        goal = tuple(jnp.where(cond, a, b) for a, b in zip(g, goal))
    return goal


def apply_reset(init, memory, reset):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)


# ================================================================ agent_1: inner cook
def init_agent_1():
    return {
        't': i32(0),
        'belief': i32(-1),
        'conf': i32(0),
        'cyc': i32(0),
        'ncyc': i32(0),
        'snap_c': jnp.full((4,), -1, dtype=I32),
        'last_p': i32(-2),
        'prev_dish': i32(0),
        'prev_deliver': i32(0),
        'hits': i32(0),
        'miss': i32(0),
        'pressed': i32(0),
        'ev_belief': i32(-1),
        'ev_counted': i32(0),
        'disp_seen': i32(-1),
        'prev_recipe': i32(-1),
        'verified': i32(0),
        'contra': i32(0),
        'changed_seen': i32(0),
    }


def agent_1_act(obs, reset, memory, rng):
    K = consts()
    mem = apply_reset(init_agent_1(), memory, reset)
    t = mem['t']
    pos, r, c = localize(K, obs, K['INNER_MASK'])
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(I32)
    self_item = obs[2, 2, 5:10]
    held = item_code(self_item)
    held_plate = held == 4
    held_dish = held >= 5
    held_ing = (held >= 1) & (held <= 3)
    held_type = jnp.where(held_ing, held - 1, jnp.where(held_dish, held - 5, -1)).astype(I32)

    partner_vis = jnp.sum(obs[:, :, 10]) > 0
    p_item = jnp.sum(obs[:, :, 15:20], axis=(0, 1))
    p_code = item_code(p_item)
    p_type = jnp.where(partner_vis, jnp.where((p_code >= 1) & (p_code <= 3), p_code - 1, -1), -2).astype(I32)

    pot_v, _ = read_cell(obs, r, c, 4, 4)
    pot_counts = pot_v[31:34]
    pot_n = jnp.sum(pot_counts)
    pot_cooked = pot_v[30] > 0
    pot_timer = pot_v[39]
    pot_type = jnp.argmax(pot_counts).astype(I32)
    pot_cooking = pot_timer > 0
    pot_idle = (~pot_cooking) & (~pot_cooked)
    pot_ready = pot_cooked & (~pot_cooking)

    # shared counters
    c_codes = []
    c_vis = []
    for (gr, gc) in SHARED_CELLS:
        v, vis = read_cell(obs, r, c, gr, gc)
        c_codes.append(item_code(v[29:34]))
        c_vis.append(vis)
    c_codes = jnp.stack(c_codes)
    c_vis = jnp.stack(c_vis)
    c_ing = jnp.where((c_codes >= 1) & (c_codes <= 3), c_codes - 1, -1)

    # button displays
    b1, _ = read_cell(obs, r, c, 2, 4)
    b2, _ = read_cell(obs, r, c, 6, 4)
    disp = jnp.maximum(recipe_of(b1[34:39]), recipe_of(b2[34:39]))

    # ---- cycle bookkeeping: detect own successful delivery
    delivered = (mem['prev_dish'] > 0) & (mem['prev_deliver'] > 0) & (~held_dish)
    prev_recipe = jnp.where(delivered, mem['belief'], mem['prev_recipe'])
    belief = jnp.where(delivered, -1, mem['belief'])
    conf = jnp.where(delivered, 0, mem['conf'])
    ev_belief = jnp.where(delivered, -1, mem['ev_belief'])
    ev_counted = jnp.where(delivered, 0, mem['ev_counted'])
    disp_seen = jnp.where(delivered, -1, mem['disp_seen'])
    cyc = jnp.where(delivered, t, mem['cyc'])
    ncyc = mem['ncyc'] + jnp.where(delivered, 1, 0)
    snap_c = jnp.where(delivered, c_ing, mem['snap_c'])
    since = t - cyc
    verified = mem['verified']
    contra = mem['contra']
    changed_seen = mem['changed_seen']

    # ---- evidence
    # partner held ingredient 0/1: fresh in first cycle, or a transition from a seen non-holding state,
    # or seen sufficiently long after the delivery
    p_ok = (p_type >= 0) & (p_type <= 1)
    p_fresh = p_ok & ((ncyc == 0) | ((mem['last_p'] != p_type) & (mem['last_p'] != -2)))
    # counters
    c_ok = (c_ing >= 0) & (c_ing <= 1) & c_vis
    c_fresh = c_ok & ((ncyc == 0) | ((c_ing != snap_c) & (since >= D_COUNTER)))
    any_c = jnp.any(c_fresh)
    c_val = c_ing[jnp.argmax(c_fresh)]
    ev_val = jnp.where(p_fresh, p_type, jnp.where(any_c, c_val, -1))
    new_ev = (ev_val >= 0) & (ev_belief < 0)
    ev_belief = jnp.where(new_ev, ev_val, ev_belief)
    # trust evidence without a button check only once the partner has been verified and has
    # demonstrated tracking of recipe changes (or this evidence itself differs from last recipe)
    trusted = (contra == 0) & (verified >= 1) & (changed_seen | (ev_val != prev_recipe))
    accept = new_ev & trusted & (conf < 2)
    changed_seen = jnp.where(accept & (ev_val != prev_recipe), 1, changed_seen)
    belief = jnp.where(accept, ev_val, belief)
    conf = jnp.where(accept, 1, conf)
    need_verify = (ev_belief >= 0) & (conf < 1)
    # button display: authoritative; also scores the evidence source
    disp_seen = jnp.where(disp >= 0, disp, disp_seen)
    score_now = (disp_seen >= 0) & (ev_belief >= 0) & (ev_counted == 0)
    verified = verified + jnp.where(score_now & (disp_seen == ev_belief), 1, 0)
    contra = contra + jnp.where(score_now & (disp_seen != ev_belief), 1, 0)
    ev_counted = jnp.where(score_now, 1, ev_counted)
    belief = jnp.where(disp >= 0, disp, belief)
    conf = jnp.where(disp >= 0, 2, conf)
    # pot contents dictate while a dish is in progress
    known = belief >= 0

    # ---- goals
    pot_acc, pot_face, _ = nearest(K, pos, K['IN_POT_ACC'], K['IN_POT_FACE'], jnp.ones((4,), dtype=bool), facing)
    del_acc, del_face, _ = nearest(K, pos, K['IN_DELIV_ACC'], K['IN_DELIV_FACE'], jnp.ones((2,), dtype=bool), facing)
    btn_acc, btn_face, _ = nearest(K, pos, K['IN_BUTTON_ACC'], K['IN_BUTTON_FACE'], jnp.ones((2,), dtype=bool), facing)
    empty_c = c_codes == 0
    dump_acc, dump_face, dump_ok = nearest(K, pos, K['SHARED_IN_ACC'], K['SHARED_IN_FACE'], empty_c)
    # plate sources: piles plus counters holding a plate
    plate_acc_all = jnp.concatenate([K['IN_PLATE_ACC'], K['SHARED_IN_ACC']])
    plate_face_all = jnp.concatenate([K['IN_PLATE_FACE'], K['SHARED_IN_FACE']])
    plate_valid = jnp.concatenate([jnp.ones((2,), dtype=bool), c_codes == 4])
    plate_acc, plate_face, _ = nearest(K, pos, plate_acc_all, plate_face_all, plate_valid, facing)

    # ingredient to fetch
    want = jnp.where(pot_n > 0, pot_type, jnp.where(known, belief, 0)).astype(I32)
    want = jnp.clip(want, 0, 2)
    src_acc_all = jnp.concatenate([K['IN_SUPPLY_ACC'][want][None], K['SHARED_IN_ACC']])
    src_face_all = jnp.concatenate([K['IN_SUPPLY_FACE'][want][None], K['SHARED_IN_FACE']])
    src_valid = jnp.concatenate([jnp.ones((1,), dtype=bool), c_ing == want])
    src_acc, src_face, _ = nearest(K, pos, src_acc_all, src_face_all, src_valid, facing)

    # counter dish matching belief
    dish_match_c = known & (c_codes == 5 + belief)
    cd_acc, cd_face, cd_ok = nearest(K, pos, K['SHARED_IN_ACC'], K['SHARED_IN_FACE'], dish_match_c)

    dish_ok = (~known) | ((self_item[2 + jnp.clip(belief, 0, 2)] == 3))
    ing_ok = pot_idle & (pot_n < 3) & ((pot_n == 0) | (pot_type == held_type)) & \
        ((pot_n > 0) | (~known) | (belief == held_type))

    later_w = jnp.where((mem['hits'] > 0) & (mem['hits'] >= mem['miss']), W_LATER, W_SHORT)
    deadline = jnp.where(ncyc == 0, W_FIRST, later_w)
    go_button = (since >= (deadline - BUTTON_LEAD)) | need_verify
    # adaptivity bookkeeping: evidence arriving in a cycle is a hit; a button press without any
    # evidence this cycle is a miss
    # a display appearing without our own press is a partner-provided signal: count it as a hit
    partner_disp = (disp >= 0) & (mem['pressed'] == 0) & (mem['conf'] < 2) & (~delivered)
    hits = mem['hits'] + jnp.where(new_ev | partner_disp, 1, 0)
    own_press = (disp >= 0) & (mem['pressed'] > 0) & (mem['conf'] == 0)
    miss = mem['miss'] + jnp.where(own_press & (ev_belief < 0), 1, 0)

    one = i32(1)
    zero = i32(0)
    default = (K['IN_WAIT'], i32(RIGHT), zero)
    pairs = [
        # holding a cooked dish
        (held_dish & dish_ok, (del_acc, del_face, one)),
        (held_dish & dump_ok, (dump_acc, dump_face, one)),
        (held_dish, (del_acc, del_face, one)),
        # holding a plate
        (held_plate & pot_ready, (pot_acc, pot_face, one)),
        (held_plate & pot_cooking, (K['IN_WAIT'], i32(RIGHT), zero)),
        (held_plate & dump_ok, (dump_acc, dump_face, one)),
        (held_plate, (pot_acc, pot_face, zero)),
        # holding an ingredient
        (held_ing & ing_ok, (pot_acc, pot_face, one)),
        (held_ing & dump_ok, (dump_acc, dump_face, one)),
        (held_ing, (K['IN_WAIT'], i32(RIGHT), zero)),
        # empty hands
        (pot_cooking | pot_ready, (plate_acc, plate_face, one)),
        (cd_ok, (cd_acc, cd_face, one)),
        (pot_idle & (pot_n > 0), (src_acc, src_face, one)),
        (known, (src_acc, src_face, one)),
        (go_button, (btn_acc, btn_face, one)),
    ]
    goal = choose(default, pairs)
    action = act_toward(K, pos, facing, goal[0], goal[1], goal[2])

    is_deliver = held_dish & (goal[0] == del_acc) & (goal[1] == del_face) & (action == 5)
    is_press = (~held_dish) & (~held_plate) & (~held_ing) & (goal[0] == btn_acc) & (goal[1] == btn_face) & (action == 5)
    new_mem = {
        't': t + 1,
        'belief': belief.astype(I32),
        'conf': conf.astype(I32),
        'cyc': cyc.astype(I32),
        'ncyc': ncyc.astype(I32),
        'snap_c': snap_c.astype(I32),
        'last_p': p_type.astype(I32),
        'prev_dish': jnp.where(held_dish, 1, 0).astype(I32),
        'prev_deliver': jnp.where(is_deliver, 1, 0).astype(I32),
        'hits': hits.astype(I32),
        'miss': miss.astype(I32),
        'pressed': jnp.where(is_press, 1, 0).astype(I32),
        'ev_belief': ev_belief.astype(I32),
        'ev_counted': ev_counted.astype(I32),
        'disp_seen': disp_seen.astype(I32),
        'prev_recipe': prev_recipe.astype(I32),
        'verified': verified.astype(I32),
        'contra': contra.astype(I32),
        'changed_seen': changed_seen.astype(I32),
    }
    trace = {'pos': pos, 'belief': belief.astype(I32), 'conf': conf.astype(I32), 'held': held,
             'pot_n': pot_n.astype(I32), 'goal': goal[0].astype(I32), 'ncyc': ncyc.astype(I32),
             'since': since.astype(I32), 'hits': hits.astype(I32), 'miss': miss.astype(I32),
             'ev': ev_belief.astype(I32), 'ver': verified.astype(I32), 'contra': contra.astype(I32)}
    return action, new_mem, trace


# ================================================================ agent_0: outer supplier
def init_agent_0():
    return {
        't': i32(0),
        'k': i32(-1),
        'cnt': jnp.zeros((4,), dtype=I32),
        'park': jnp.zeros((2,), dtype=I32),
        'phase': i32(0),
        'p_dish': i32(0),
        'walls': jnp.zeros((NDUMP,), dtype=I32),
        'pending': i32(0),
        'trig_t': i32(0),
        'hold_t': i32(0),
        'seen_t': i32(0),
        'carry_t': i32(-100),
        'press_t': i32(-100),
        'failed': i32(0),
        'pressing': i32(0),
    }


def agent_0_act(obs, reset, memory, rng):
    K = consts()
    mem = apply_reset(init_agent_0(), memory, reset)
    t = mem['t']
    pos, r, c = localize(K, obs, K['OUTER_MASK'])
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(I32)
    self_item = obs[2, 2, 5:10]
    held = item_code(self_item)
    held_plate = held == 4
    held_dish = held >= 5
    held_ing = (held >= 1) & (held <= 3)
    held_type = jnp.where(held_ing, held - 1, jnp.where(held_dish, held - 5, -1)).astype(I32)

    # recipe from any visible always-on indicator
    k = mem['k']
    for (gr, gc) in R_CELLS:
        v, vis = read_cell(obs, r, c, gr, gc)
        rk = recipe_of(v[34:39])
        k = jnp.where(vis & (rk >= 0), rk, k)
    known = k >= 0
    kk = jnp.clip(k, 0, 2)

    # shared counters (memory of last seen)
    cnt = mem['cnt']
    c_now = []
    for i, (gr, gc) in enumerate(SHARED_CELLS):
        v, vis = read_cell(obs, r, c, gr, gc)
        c_now.append(jnp.where(vis, item_code(v[29:34]), cnt[i]))
    cnt = jnp.stack(c_now).astype(I32)

    # parked dishes memory (walls (2,0) and (6,0))
    park = mem['park']
    p_now = []
    for i, (gr, gc) in enumerate(PARK_CELLS):
        v, vis = read_cell(obs, r, c, gr, gc)
        p_now.append(jnp.where(vis, item_code(v[29:34]), park[i]))
    park = jnp.stack(p_now).astype(I32)

    # partner delivery observed: partner visible holding a dish last step, visible empty-handed now
    partner_vis = jnp.sum(obs[:, :, 10]) > 0
    p_item = jnp.sum(obs[:, :, 15:20], axis=(0, 1))
    p_dish_now = partner_vis & (p_item[1] > 0)
    p_empty_now = partner_vis & (jnp.sum(p_item) == 0)
    saw_delivery = (mem['p_dish'] > 0) & p_empty_now
    p_ing_seen = partner_vis & (item_code(p_item) >= 1) & (item_code(p_item) <= 3)
    k_changed = known & (mem['k'] >= 0) & (k != mem['k'])
    first_known = known & (mem['k'] < 0)
    seen_t = jnp.where(partner_vis, t, mem['seen_t'])
    unseen_long = (t - seen_t) >= 30
    # far-side delivery: partner seen with a plate/dish since the last trigger, now seen empty-handed
    p_carry = partner_vis & ((p_item[0] > 0) | (p_item[1] > 0))
    carry_t = jnp.where(p_carry, t, mem['carry_t'])
    far_delivery = p_empty_now & (carry_t > mem['trig_t']) & (t - carry_t >= 2) & (t - carry_t <= 12)
    trigger = saw_delivery | far_delivery | k_changed | first_known | \
        (known & unseen_long & (t - mem['trig_t'] >= RESIGNAL_STEPS))
    pending = jnp.maximum(mem['pending'] - 1, 0)
    pending = jnp.where(trigger, PENDING_STEPS, pending)
    trig_t = jnp.where(trigger, t, mem['trig_t'])
    since_trig = t - trig_t
    # message received: partner seen holding an ingredient a few steps after the trigger
    pending = jnp.where(p_ing_seen & (since_trig >= 3), 0, pending)
    showing = pending > 0
    trip_ok = since_trig <= STOCK_WINDOW
    # stuck partner detection: never seen holding anything for a long time -> press the button once
    hold_t = jnp.where(p_ing_seen | p_dish_now | (partner_vis & (p_item[0] > 0)), t, mem['hold_t'])
    press_t = mem['press_t']
    failed = mem['failed']
    pressing = mem['pressing']
    # evaluate previous press: success if partner held something within 20 steps after it
    eval_now = (pressing == 2) & (t - press_t == 20)
    failed = jnp.where(eval_now, jnp.where(hold_t > press_t, 0, failed + 1), failed)
    pressing = jnp.where(eval_now, 0, pressing)
    stuck = known & (t - hold_t >= STUCK_STEPS) & (since_trig >= 20) & (failed < MAX_FAILED_PRESSES) & \
        (pressing == 0) & (t - press_t >= STUCK_STEPS)
    pressing = jnp.where(stuck, 1, pressing)
    # re-show state machine: 1 = drop held ingredient on the re-show wall, 2 = pick it back up
    phase = mem['phase']
    rs_wall = []
    for (gr, gc) in RESHOW_CELLS:
        v, vis = read_cell(obs, r, c, gr, gc)
        rs_wall.append(jnp.where(vis, item_code(v[29:34]), -1))
    rs_wall = jnp.stack(rs_wall)
    rs_i = jnp.clip(k, 0, 1)
    rs_acc = K['RESHOW_ACC'][rs_i]
    rs_empty = rs_wall[rs_i] == 0
    # dump spot validity: remembered wall occupancy (only this seat can reach these walls)
    walls = mem['walls']
    w_now = []
    for i, (acc, fd, wall) in enumerate(DUMPS):
        v, vis = read_cell(obs, r, c, wall[0], wall[1])
        w_now.append(jnp.where(vis, item_code(v[29:34]), walls[i]))
    walls = jnp.stack(w_now).astype(I32)
    d_valid = walls == 0
    dump_acc, dump_face, dump_ok = nearest(K, pos, K['DUMP_ACC'], K['DUMP_FACE'], d_valid)

    primary = K['PRIMARY'][kk]
    c_empty = cnt == 0
    need_stock = known & jnp.any(primary & c_empty)
    stock_acc, stock_face, _ = nearest(K, pos, K['SHARED_OUT_ACC'], K['SHARED_OUT_FACE'], primary & c_empty)
    c_is_ing = (cnt >= 1) & (cnt <= 3)
    stale = known & c_is_ing & (cnt != 1 + kk)
    any_stale = jnp.any(stale)
    stale_acc, stale_face, _ = nearest(K, pos, K['SHARED_OUT_ACC'], K['SHARED_OUT_FACE'], stale)
    match_dish = known & (cnt == 5 + kk)
    md_acc, md_face, md_ok = nearest(K, pos, K['SHARED_OUT_ACC'], K['SHARED_OUT_FACE'], match_dish)
    parked_match = known & (park == 5 + kk)
    pm_acc, pm_face, pm_ok = nearest(K, pos, K['PARK_ACC'][:2], K['PARK_FACE'][:2], parked_match)
    del_acc, del_face, _ = nearest(K, pos, K['OUT_DELIV_ACC'], K['OUT_DELIV_FACE'], jnp.ones((2,), dtype=bool))

    one = i32(1)
    zero = i32(0)
    dish_match = known & (self_item[2 + jnp.clip(held_type, 0, 2)] == 3)
    ing_match = known & (held_type == kk)
    start_reshow = saw_delivery & held_ing & ing_match & (phase == 0)
    phase = jnp.where(start_reshow, 1, phase)
    # phase 1 -> 2 once hands are empty; phase 2 -> 0 once holding again
    phase = jnp.where((phase == 1) & (~held_ing) & (~held_dish) & (~held_plate), 2, phase)
    phase = jnp.where((phase == 2) & held_ing, 0, phase)
    # abort re-show if recipe changed away from what we hold, or the wall is unusable
    phase = jnp.where((phase == 1) & (~ing_match), 0, phase)
    phase = jnp.where((phase == 1) & (~rs_empty) & (rs_wall[rs_i] >= 0), 0, phase)
    phase = jnp.where((phase == 2) & (rs_wall[rs_i] == 0), 0, phase)
    default = (K['OUT_RVIEW'], i32(RIGHT), zero)
    pairs = [
        # re-show in progress
        ((phase == 1) & held_ing, (rs_acc, i32(LEFT), one)),
        ((phase == 2) & (~held_ing) & (~held_dish) & (~held_plate), (rs_acc, i32(LEFT), one)),
        # holding a dish
        (held_dish & dish_match, (del_acc, del_face, one)),
        (held_dish & (~known), (K['OUT_RVIEW'], i32(RIGHT), zero)),
        (held_dish, (K['PARK_ACC'][jnp.clip(held_type, 0, 2)], i32(LEFT), one)),
        # holding a plate
        (held_plate & dump_ok, (dump_acc, dump_face, one)),
        (held_plate, (K['OUT_RVIEW'], i32(RIGHT), zero)),
        # holding an ingredient
        (held_ing & known & (~ing_match) & dump_ok, (dump_acc, dump_face, one)),
        (held_ing & ing_match & showing, (K['OUT_SHOW'][kk], i32(RIGHT), zero)),
        (held_ing & ing_match & need_stock, (stock_acc, stock_face, one)),
        (held_ing & ing_match & dump_ok, (dump_acc, dump_face, one)),
        (held_ing & ing_match, (K['OUT_SHOW'][kk], i32(RIGHT), zero)),
        (held_ing, (K['OUT_RVIEW'], i32(RIGHT), zero)),
        # empty hands
        ((pressing == 1), (K['OUT_BUTTON'], i32(DOWN), one)),
        (md_ok, (md_acc, md_face, one)),
        (pm_ok, (pm_acc, pm_face, one)),
        (known & showing, (K['OUT_SUPPLY_ACC'][kk], K['OUT_SUPPLY_FACE'][kk], one)),
        (need_stock & trip_ok, (K['OUT_SUPPLY_ACC'][kk], K['OUT_SUPPLY_FACE'][kk], one)),
        (any_stale & trip_ok, (stale_acc, stale_face, one)),
    ]
    goal = choose(default, pairs)
    action = act_toward(K, pos, facing, goal[0], goal[1], goal[2])
    did_press = (pressing == 1) & (goal[0] == K['OUT_BUTTON']) & (action == 5) & \
        (~held_ing) & (~held_dish) & (~held_plate)
    press_t = jnp.where(did_press, t, press_t)
    pressing = jnp.where(did_press, 2, pressing)

    new_mem = {
        't': t + 1,
        'k': k.astype(I32),
        'cnt': cnt.astype(I32),
        'park': park.astype(I32),
        'phase': phase.astype(I32),
        'p_dish': jnp.where(p_dish_now, 1, 0).astype(I32),
        'walls': walls.astype(I32),
        'pending': pending.astype(I32),
        'trig_t': trig_t.astype(I32),
        'hold_t': hold_t.astype(I32),
        'seen_t': seen_t.astype(I32),
        'carry_t': carry_t.astype(I32),
        'press_t': press_t.astype(I32),
        'failed': failed.astype(I32),
        'pressing': pressing.astype(I32),
    }
    trace = {'pos': pos, 'k': k.astype(I32), 'held': held, 'goal': goal[0].astype(I32),
             'cnt': cnt.astype(I32), 'phase': phase.astype(I32), 'pending': pending.astype(I32)}
    return action, new_mem, trace
