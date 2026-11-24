"""grounded_coord_ring policy.

agent_1 (inner room) is a self-sufficient cook: it reads the recipe from any
visible display (button or free indicator), pressing the top button itself
when no display is visible, then fills the pot, plates and delivers.
agent_0 (outer ring) parks at (1,4) next to the top button, watches the free
indicator at (0,4) and presses the button only when it has evidence the
partner needs the recipe (recipe change, episode start, or an idle partner).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'ring_cook_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1,
                   'max_height': 9, 'max_width': 9, 'num_ingredients': 3}

# ----------------------------------------------------------------------------
# Literal tables (generated offline from context/environment/layout.md)
# ----------------------------------------------------------------------------
# actions / facing ids
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
FACE_ACT = [A_UP, A_DOWN, A_RIGHT, A_LEFT]  # facing id -> movement action that sets it

# Ring cells (agent_0 region), in this order:
# (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,1),(2,7),(3,1),(3,7),(4,1),(4,7),
# (5,1),(5,7),(6,1),(6,7),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7)
# For each ring cell, the local 5x5 view of static objects encoded as
# (static channel index + 1), 0 = floor/off-map. Channel order:
# wall, goal, pot, R, L, plate, ing0, ing1, ing2.
RING_CODES = [[[0, 0, 0, 0, 0], [0, 1, 1, 1, 9], [0, 1, 0, 0, 0], [0, 1, 0, 1, 1], [0, 9, 0, 7, 0]], [[0, 0, 0, 0, 0], [1, 1, 1, 9, 4], [1, 0, 0, 0, 0], [1, 0, 1, 1, 5], [9, 0, 7, 0, 0]], [[0, 0, 0, 0, 0], [1, 1, 9, 4, 9], [0, 0, 0, 0, 0], [0, 1, 1, 5, 1], [0, 7, 0, 0, 0]], [[0, 0, 0, 0, 0], [1, 9, 4, 9, 1], [0, 0, 0, 0, 0], [1, 1, 5, 1, 1], [7, 0, 0, 0, 6]], [[0, 0, 0, 0, 0], [9, 4, 9, 1, 1], [0, 0, 0, 0, 0], [1, 5, 1, 1, 0], [0, 0, 0, 6, 0]], [[0, 0, 0, 0, 0], [4, 9, 1, 1, 1], [0, 0, 0, 0, 1], [5, 1, 1, 0, 1], [0, 0, 6, 0, 9]], [[0, 0, 0, 0, 0], [9, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 0], [0, 6, 0, 9, 0]], [[0, 1, 1, 1, 9], [0, 1, 0, 0, 0], [0, 1, 0, 1, 1], [0, 9, 0, 7, 0], [0, 4, 0, 2, 0]], [[9, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 0], [0, 6, 0, 9, 0], [0, 2, 0, 4, 0]], [[0, 1, 0, 0, 0], [0, 1, 0, 1, 1], [0, 9, 0, 7, 0], [0, 4, 0, 2, 0], [0, 9, 0, 8, 0]], [[0, 0, 0, 1, 0], [1, 1, 0, 1, 0], [0, 6, 0, 9, 0], [0, 2, 0, 4, 0], [0, 6, 0, 9, 0]], [[0, 1, 0, 1, 1], [0, 9, 0, 7, 0], [0, 4, 0, 2, 0], [0, 9, 0, 8, 0], [0, 1, 0, 1, 1]], [[1, 1, 0, 1, 0], [0, 6, 0, 9, 0], [0, 2, 0, 4, 0], [0, 6, 0, 9, 0], [1, 1, 0, 1, 0]], [[0, 9, 0, 7, 0], [0, 4, 0, 2, 0], [0, 9, 0, 8, 0], [0, 1, 0, 1, 1], [0, 1, 0, 0, 0]], [[0, 6, 0, 9, 0], [0, 2, 0, 4, 0], [0, 6, 0, 9, 0], [1, 1, 0, 1, 0], [0, 0, 0, 1, 0]], [[0, 4, 0, 2, 0], [0, 9, 0, 8, 0], [0, 1, 0, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 1, 9]], [[0, 2, 0, 4, 0], [0, 6, 0, 9, 0], [1, 1, 0, 1, 0], [0, 0, 0, 1, 0], [9, 1, 1, 1, 0]], [[0, 9, 0, 8, 0], [0, 1, 0, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 1, 9], [0, 0, 0, 0, 0]], [[9, 0, 8, 0, 0], [1, 0, 1, 1, 5], [1, 0, 0, 0, 0], [1, 1, 1, 9, 4], [0, 0, 0, 0, 0]], [[0, 8, 0, 0, 0], [0, 1, 1, 5, 1], [0, 0, 0, 0, 0], [1, 1, 9, 4, 9], [0, 0, 0, 0, 0]], [[8, 0, 0, 0, 6], [1, 1, 5, 1, 1], [0, 0, 0, 0, 0], [1, 9, 4, 9, 1], [0, 0, 0, 0, 0]], [[0, 0, 0, 6, 0], [1, 5, 1, 1, 0], [0, 0, 0, 0, 0], [9, 4, 9, 1, 1], [0, 0, 0, 0, 0]], [[0, 0, 6, 0, 9], [5, 1, 1, 0, 1], [0, 0, 0, 0, 1], [4, 9, 1, 1, 1], [0, 0, 0, 0, 0]], [[0, 6, 0, 9, 0], [1, 1, 0, 1, 0], [0, 0, 0, 1, 0], [9, 1, 1, 1, 0], [0, 0, 0, 0, 0]]]
# BFS next action on the ring: NEXT0[from][to]
NEXT0 = [[4, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0], [2, 4, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0], [2, 2, 4, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 0, 0, 0], [2, 2, 2, 4, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0, 0, 0, 0], [2, 2, 2, 2, 4, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 4, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 4, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 4, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 4, 1, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 4, 0, 0, 0, 0, 0], [2, 2, 2, 2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 4, 0, 0, 0, 0], [2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 4, 0, 0, 0], [2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 4, 0, 0], [2, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 4, 0], [2, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 4]]
TARGET0 = 3  # ring index of (1,4)

# Inner cells (agent_1 region): 0:(3,3) 1:(3,4) 2:(3,5) 3:(4,3) 4:(4,5) 5:(5,3) 6:(5,4) 7:(5,5)
NEXT1 = [[4, 0, 0, 1, 0, 1, 1, 0], [2, 4, 0, 2, 0, 2, 0, 0], [2, 2, 4, 2, 1, 1, 1, 1], [3, 3, 3, 4, 1, 1, 1, 1], [3, 3, 3, 1, 4, 1, 1, 1], [3, 3, 0, 3, 0, 4, 0, 0], [2, 0, 0, 2, 0, 2, 4, 0], [2, 3, 3, 2, 3, 2, 2, 4]]
INNER_LOOKUP = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, 0, 0], [0, 0, 0, 3, 0, 4, 0, 0, 0], [0, 0, 0, 5, 6, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
C33, C34, C35, C43, C45, C53, C54, C55 = 0, 1, 2, 3, 4, 5, 6, 7
# nearest pot-adjacent cell & facing, indexed by current inner cell
POT_CELL = [C34, C34, C34, C43, C45, C54, C54, C54]
POT_DIR = [F_DOWN, F_DOWN, F_DOWN, F_RIGHT, F_LEFT, F_UP, F_UP, F_UP]
PLATE_CELL = [C35, C35, C35, C35, C35, C55, C55, C55]  # face RIGHT there
CTR_CELL = [C33, C33, C35, C33, C35, C53, C53, C55]     # shared counters
CTR_DIR = [F_UP, F_UP, F_UP, F_UP, F_UP, F_DOWN, F_DOWN, F_DOWN]
ALT_CTR = [C35, C35, C33, C35, C33, C55, C55, C53]      # other counter on the same side
SUPPLY_CELL = [C33, C53, C33]  # by ingredient id (ing2 unreachable -> fallback); face LEFT
BUTTON_CELL, BUTTON_DIR = C34, F_UP
# (4,5) touches both the pot (to its left) and the right delivery station (4,6):
# plate the dish there, then turn right and deliver in two steps.
PICKUP_CELL, PICKUP_DIR = C45, F_LEFT
DELIV_CELL, DELIV_DIR = C45, F_RIGHT
# Pre-press: the button display lasts 9 observations and is updated on delivery, so pressing
# it at pot timer <= 6 (then plate via (3,5), pick at (4,5), turn, deliver) shows the NEW
# recipe in the observation right after delivery: press -> deliver at press+8.
PREPRESS_T = 6
LAST_USEFUL_PREPRESS_STEP = 352  # press+8 (delivery) + 39 (next cycle) must fit in 399
# facing id -> (drow, dcol) of the faced cell
FACE_DR = [-1, 1, 0, 0]
FACE_DC = [0, 0, 1, -1]

PRESS_COOLDOWN = 10
IDLE_COOLDOWN = 20
IDLE_STEPS = 5
IDLE_NO_ING_STEPS = 25
INIT_STEP = 8
STALL_STEPS = 45  # no recipe change / display / press seen for this long -> refresh display
LAST_USEFUL_PRESS_STEP = 360  # a press after this cannot yield a delivery before step 399


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _reset_tree(reset, init, memory):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _display(obs, mask):
    """Return (active, recipe(3,)) from the cells selected by mask (5,5)."""
    counts = obs[:, :, 36:39]
    tot = counts.sum(-1) * mask
    active = tot.max() > 0
    cell = jnp.argmax(tot.reshape(-1))
    rec = counts.reshape(25, 3)[cell]
    return active, rec.astype(jnp.int32)


def _nav(pos, facing, goal, gdir, do_interact, next_table):
    at = pos == goal
    faced = facing == gdir
    move = next_table[pos, goal]
    turn = _i32(FACE_ACT)[gdir]
    return jnp.where(~at, move, jnp.where(~faced, turn, jnp.where(do_interact, A_INTERACT, A_STAY)))


# ============================================================================
# agent_1: inner-room cook
# ============================================================================
def init_agent_1():
    return {'known': _i32(0), 'recipe': jnp.zeros((3,), jnp.int32), 'prev_dish': _i32(0),
            'step': _i32(0), 'prepressed': _i32(0)}


def agent_1_act(obs, reset, memory, rng):
    mem = _reset_tree(reset, init_agent_1(), memory)
    obs = obs.astype(jnp.int32)
    next1 = _i32(NEXT1)
    # --- localisation from the pot (always in view inside the inner room)
    pot_flat = jnp.argmax(obs[:, :, 22].reshape(-1))
    pi, pj = pot_flat // 5, pot_flat % 5
    gr, gc = 6 - pi, 6 - pj
    pos = _i32(INNER_LOOKUP)[jnp.clip(gr, 0, 8), jnp.clip(gc, 0, 8)]
    facing = _facing(obs)
    # --- inventory
    inv = obs[2, 2, 5:10]
    has_dish = inv[1] > 0
    has_plate = (inv[0] > 0) & ~has_dish
    cnt = inv[2:5]
    has_ing = (cnt.sum() > 0) & (inv[0] == 0) & (inv[1] == 0)
    ing_id = jnp.argmax(cnt).astype(jnp.int32)
    # --- pot state
    pot_cooked = obs[pi, pj, 30] > 0
    pot_cnt = obs[pi, pj, 31:34]
    pot_timer = obs[pi, pj, 39]
    pot_cooking = pot_timer > 0
    pot_idle = ~pot_cooked & ~pot_cooking
    pot_total = pot_cnt.sum()
    # --- recipe knowledge from any visible display (button or free indicator)
    mask = ((obs[:, :, 23] + obs[:, :, 24]) > 0).astype(jnp.int32)
    active, disp = _display(obs, mask)
    delivered = (mem['prev_dish'] > 0) & ~has_dish
    known = jnp.where(active, 1, jnp.where(delivered, 0, mem['known'])).astype(jnp.int32)
    recipe = jnp.where(active, disp, mem['recipe'])
    need_mask = (recipe - pot_cnt) > 0
    need = jnp.argmax(need_mask).astype(jnp.int32)
    # --- branch selection
    step = mem['step']
    can_put = pot_idle & (pot_total < 3) & ((known == 0) | (recipe[ing_id] > pot_cnt[ing_id]))
    drop = (has_ing & ~can_put) | (has_plate & pot_idle)
    press_useful = step <= LAST_USEFUL_PRESS_STEP
    prepress_wanted = ((mem['prepressed'] == 0) & (step <= LAST_USEFUL_PREPRESS_STEP))
    # branches: 0 deliver, 1 plate->pot (wait at pickup cell), 2 put ingredient,
    #           3 drop held item on a counter, 4 get plate, 5 press button, 6 get ingredient,
    #           7 idle (recipe unknown too late in the episode to matter),
    #           8 pre-press the button while the pot cooks
    branch = jnp.where(has_dish, 0,
             jnp.where(drop, 3,
             jnp.where(has_plate, 1,
             jnp.where(has_ing, 2,
             jnp.where(pot_cooked | pot_cooking, jnp.where(prepress_wanted, 8, 4),
             jnp.where(known == 0, jnp.where(press_useful, 5, 7), 6)))))).astype(jnp.int32)
    prepress_now = (pot_timer <= PREPRESS_T) | pot_cooked
    # while waiting for the press moment, face the pot (down) so that a partner watching from
    # the ring does not mistake the wait for "needs the recipe"; turn up one step before.
    prepress_dir = jnp.where((pot_timer <= PREPRESS_T + 1) | pot_cooked, F_UP, F_DOWN).astype(jnp.int32)
    pot_cell = _i32(POT_CELL)[pos]
    pot_dir = _i32(POT_DIR)[pos]
    # counter drop: if the faced primary counter already holds an item, use the alternate one
    dr = _i32(FACE_DR)[facing]
    dc = _i32(FACE_DC)[facing]
    front_item = obs[2 + dr, 2 + dc, 29:34].sum() > 0
    ctr_primary = _i32(CTR_CELL)[pos]
    ctr_dir = _i32(CTR_DIR)[pos]
    at_primary_faced = (pos == ctr_primary) & (facing == ctr_dir)
    ctr_cell = jnp.where(at_primary_faced & front_item, _i32(ALT_CTR)[pos], ctr_primary)
    goals = jnp.stack([_i32(DELIV_CELL), _i32(PICKUP_CELL), pot_cell, ctr_cell,
                       _i32(PLATE_CELL)[pos], _i32(BUTTON_CELL), _i32(SUPPLY_CELL)[need], pos,
                       _i32(BUTTON_CELL)])
    dirs = jnp.stack([_i32(DELIV_DIR), _i32(PICKUP_DIR), pot_dir, ctr_dir,
                      _i32(F_RIGHT), _i32(BUTTON_DIR), _i32(F_LEFT), facing, prepress_dir])
    goal = goals[branch]
    gdir = dirs[branch]
    do_interact = jnp.where(branch == 1, pot_cooked,
                  jnp.where(branch == 8, prepress_now, branch != 7))
    action = _nav(pos, facing, goal, gdir, do_interact, next1).astype(jnp.int32)
    pressed_now = (branch == 8) & (action == A_INTERACT)
    prepressed = jnp.where(pot_idle, 0, jnp.where(pressed_now, 1, mem['prepressed'])).astype(jnp.int32)
    new_mem = {'known': known, 'recipe': recipe.astype(jnp.int32),
               'prev_dish': has_dish.astype(jnp.int32), 'step': (step + 1).astype(jnp.int32),
               'prepressed': prepressed}
    trace = {'pos': pos.astype(jnp.int32), 'branch': branch, 'known': known,
             'need': need, 'pot_total': pot_total.astype(jnp.int32)}
    return action, new_mem, trace


# ============================================================================
# agent_0: ring signaller
# ============================================================================
def init_agent_0():
    return {'step': _i32(0), 'last_press': _i32(-100), 'have_recipe': _i32(0),
            'last_recipe': jnp.zeros((3,), jnp.int32), 'idle': _i32(0), 'pcell': _i32(-1),
            'pface': _i32(-1), 'seen_holding': _i32(0), 'pressed_initial': _i32(0),
            'pending': _i32(0), 'last_evidence': _i32(0), 'self_suff': _i32(0),
            'top_seen': _i32(0), 'last_ing_seen': _i32(-100)}


def agent_0_act(obs, reset, memory, rng):
    mem = _reset_tree(reset, init_agent_0(), memory)
    obs = obs.astype(jnp.int32)
    step = mem['step']
    # --- localisation by static template matching
    static = obs[:, :, 20:29]
    code = jnp.where(static.sum(-1) > 0, jnp.argmax(static, axis=-1) + 1, 0).astype(jnp.int32)
    mism = (_i32(RING_CODES) != code[None]).sum(axis=(1, 2))
    pos = jnp.argmin(mism).astype(jnp.int32)
    facing = _facing(obs)
    at_target = pos == TARGET0
    # --- recipe on free indicator
    maskR = (obs[:, :, 23] > 0).astype(jnp.int32)
    activeR, recR = _display(obs, maskR)
    changed = activeR & (mem['have_recipe'] > 0) & jnp.any(recR != mem['last_recipe'])
    have_recipe = (mem['have_recipe'] > 0) | activeR
    last_recipe = jnp.where(activeR, recR, mem['last_recipe'])
    # --- button display state
    maskL = (obs[:, :, 24] > 0).astype(jnp.int32)
    button_active, _ = _display(obs, maskL)
    # --- partner
    flat = obs.reshape(25, 40)
    pvis = obs[:, :, 10].max() > 0
    pcell = jnp.where(pvis, jnp.argmax(obs[:, :, 10].reshape(-1)), -1).astype(jnp.int32)
    pinv = flat[jnp.clip(pcell, 0, 24), 15:20]
    p_empty = pvis & (pinv.sum() == 0)
    p_holding_any = pvis & (pinv.sum() > 0)
    p_holding_ing = pvis & (pinv[2:5].sum() > 0) & (pinv[0] == 0) & (pinv[1] == 0)
    pface = jnp.where(pvis, jnp.argmax(flat[jnp.clip(pcell, 0, 24), 11:15]), -1).astype(jnp.int32)
    same = pvis & (pcell == mem['pcell']) & (pface == mem['pface'])
    idle = jnp.where(p_empty & same, mem['idle'] + 1, 0).astype(jnp.int32)
    seen_holding = jnp.where(changed, 0, jnp.maximum(mem['seen_holding'], p_holding_ing.astype(jnp.int32)))
    pending = jnp.where(button_active, 0, jnp.maximum(mem['pending'], changed.astype(jnp.int32)))
    pressed_initial = jnp.maximum(mem['pressed_initial'], button_active.astype(jnp.int32))
    since = step - mem['last_press']
    # --- is the partner self-sufficient (learns the recipe without our help)?
    # (a) the top display is active outside the 9-observation window of our own press
    partner_pressed = button_active & (since >= PRESS_COOLDOWN)
    # (b) the partner carries an ingredient although no top display has shown the current recipe
    top_seen = jnp.where(changed, 0, jnp.maximum(mem['top_seen'], button_active.astype(jnp.int32)))
    learned_alone = p_holding_ing & (top_seen == 0) & ~button_active
    self_suff = jnp.maximum(mem['self_suff'], (partner_pressed | learned_alone).astype(jnp.int32))
    helping = self_suff == 0
    # from (1,4) the button (2,4) lies one row below, so face DOWN to interact
    can_press = (at_target & (facing == F_DOWN) & ~button_active & (since >= PRESS_COOLDOWN)
                 & (step <= LAST_USEFUL_PRESS_STEP))
    want_change = (pending > 0) & helping
    want_init = (step >= INIT_STEP) & (pressed_initial == 0) & (seen_holding == 0) & helping
    # idle partner: empty-handed, motionless, not facing the pot, and not seen carrying an
    # ingredient recently (a cook waiting for the pot is not waiting for the recipe)
    last_ing_seen = jnp.where(p_holding_ing, step, mem['last_ing_seen']).astype(jnp.int32)
    want_idle = ((idle >= IDLE_STEPS) & (pface != F_DOWN) & (since >= IDLE_COOLDOWN) & helping
                 & ((step - last_ing_seen) >= IDLE_NO_ING_STEPS))
    # evidence that the partner knows the recipe: a recipe change was just handled, a display
    # was active, the partner carries something, or we pressed. Without any evidence for
    # STALL_STEPS, refresh the display (only while we still believe the partner needs help).
    evidence = changed | button_active | p_holding_any
    last_evidence = jnp.where(evidence, step, mem['last_evidence'])
    want_stall = ((step - last_evidence) >= STALL_STEPS) & helping
    press = can_press & (want_change | want_init | want_idle | want_stall)
    last_evidence = jnp.where(press, step, last_evidence).astype(jnp.int32)
    move = _i32(NEXT0)[pos, TARGET0]
    action = jnp.where(press, A_INTERACT,
             jnp.where(at_target, jnp.where(facing != F_DOWN, A_DOWN, A_STAY), move)).astype(jnp.int32)
    new_mem = {
        'step': (step + 1).astype(jnp.int32),
        'last_press': jnp.where(press, step, mem['last_press']).astype(jnp.int32),
        'have_recipe': have_recipe.astype(jnp.int32),
        'last_recipe': last_recipe.astype(jnp.int32),
        'idle': idle,
        'pcell': pcell,
        'pface': pface,
        'seen_holding': seen_holding.astype(jnp.int32),
        'pressed_initial': jnp.maximum(pressed_initial, press.astype(jnp.int32)).astype(jnp.int32),
        'pending': jnp.where(press, 0, pending).astype(jnp.int32),
        'last_evidence': last_evidence,
        'self_suff': self_suff.astype(jnp.int32),
        'top_seen': top_seen.astype(jnp.int32),
        'last_ing_seen': last_ing_seen,
    }
    trace = {'pos': pos, 'press': press.astype(jnp.int32), 'changed': changed.astype(jnp.int32),
             'button_active': button_active.astype(jnp.int32), 'idle': idle,
             'want_init': want_init.astype(jnp.int32), 'self_suff': self_suff.astype(jnp.int32)}
    return action, new_mem, trace
