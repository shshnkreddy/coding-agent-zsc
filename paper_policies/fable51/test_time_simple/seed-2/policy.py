"""Overcooked V2 test_time_simple policy.

Geometry facts (from context/environment):
  * Left room (cols 1-2) and right room (cols 5-6) are separated by the wall
    column 3.  Only agent_1 (right room) can reach the pot (2,4), plate piles,
    ingredient 0 (1,7), ingredient 1 (3,7) and the delivery cell (2,7).
  * Only agent_0 (left room) can see the always-on recipe indicator (2,0).
  * Agent_0 can pick up ingredient 2 from (0,2)/(4,2) and drop it on the wall
    counters (1,3), (2,3), (3,3).  Agent_1 can see those counters from column 5.

Protocol:
  * agent_0 keeps one ingredient-2 unit on counter (1,3) when the recipe is
    [0,0,0] (same row as the ingredient-0 supply) and on (3,3) when the recipe
    is [1,1,1] (same row as the ingredient-1 supply), swapping it on change.
  * agent_1 cooks alone.  It decodes the counter signal tolerantly (row match,
    both occupied -> most recently placed, middle-only -> recipe 1, nothing for
    a long time -> alternating guess) and re-verifies the signal before
    committing an ingredient into an empty pot.  Wrong ingredients are parked
    on the right-room side counters and reused later.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'row_signal_solo_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Actions
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# Channels
CH_WALL, CH_GOAL, CH_POT, CH_RECIPE_IND = 20, 21, 22, 23
CH_REC0, CH_REC1 = 36, 37
CH_TIMER = 39

SIGNAL_TIMEOUT = 20        # steps with no signal before falling back to a guess
SIGNAL_TIMEOUT_NOSIG = 6   # shorter timeout once the partner has never signalled
MEM_SIZE = 16


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _move_for_facing(tf):
    """Move action that produces facing id tf (UP=0, DOWN=1, RIGHT=2, LEFT=3)."""
    table = jnp.asarray([A_UP, A_DOWN, A_RIGHT, A_LEFT], dtype=jnp.int32)
    return table[tf]


def _cell(obs, r, c, gr, gc, ch):
    """Value of channel ch at global cell (gr, gc); 0 when outside the view."""
    li = gr - r + 2
    lj = gc - c + 2
    valid = (li >= 0) & (li <= 4) & (lj >= 0) & (lj <= 4)
    v = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), ch]
    return jnp.where(valid, v, 0)


def _occupied(obs, r, c, gr, gc):
    li = gr - r + 2
    lj = gc - c + 2
    valid = (li >= 0) & (li <= 4) & (lj >= 0) & (lj <= 4)
    v = jnp.sum(obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), 29:34])
    return jnp.where(valid, v > 0, False)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _go_face_act(r, c, facing, tr, tc, tf, ready_action):
    """Walk to (tr,tc) inside an obstacle-free room, face tf, then ready_action.

    The last move is chosen along the target facing so that the facing step is
    usually free.
    """
    dr = tr - r
    dc = tc - c
    move_v = jnp.where(dr > 0, A_DOWN, A_UP)
    move_h = jnp.where(dc > 0, A_RIGHT, A_LEFT)
    tf_horizontal = (tf == F_RIGHT) | (tf == F_LEFT)
    both = (dr != 0) & (dc != 0)
    first = jnp.where(tf_horizontal, move_v, move_h)
    move = jnp.where(both, first, jnp.where(dr != 0, move_v, move_h))
    at = (dr == 0) & (dc == 0)
    faced = facing == tf
    out = jnp.where(~at, move, jnp.where(~faced, _move_for_facing(tf), ready_action))
    return _i32(out)


# ---------------------------------------------------------------------------
# agent_0: recipe signaller in the left room
# ---------------------------------------------------------------------------

def init_agent_0():
    return jnp.zeros((MEM_SIZE,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs).astype(jnp.int32)
    mem = jnp.where(reset, init_agent_0(), memory)

    # localise from the always-on recipe indicator at global (2,0)
    flat = jnp.argmax(obs[:, :, CH_RECIPE_IND].reshape(-1))
    iR = flat // 5
    jR = flat % 5
    r = 4 - iR
    c = 2 - jR
    facing = _facing(obs)

    rec0 = obs[iR, jR, CH_REC0]
    rec1 = obs[iR, jR, CH_REC1]
    rec = jnp.where(rec1 > rec0, 1, 0)
    tr = jnp.where(rec == 0, 1, 3)                 # target counter row
    sup_face = jnp.where(rec == 0, F_UP, F_DOWN)   # supply (0,2) or (4,2)

    holding = jnp.sum(obs[2, 2, 5:10]) > 0
    occ1 = _occupied(obs, r, c, 1, 3)
    occ2 = _occupied(obs, r, c, 2, 3)
    occ3 = _occupied(obs, r, c, 3, 3)
    occ = jnp.stack([occ1, occ2, occ3])
    rows = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    target_occ = jnp.where(rec == 0, occ1, occ3)
    wrong = occ & (rows != tr)
    any_wrong = jnp.any(wrong)
    wrong_row = rows[jnp.argmax(wrong)]

    # candidate actions
    a_place = _go_face_act(r, c, facing, tr, 2, F_RIGHT, A_INTERACT)
    a_pick_wrong = _go_face_act(r, c, facing, wrong_row, 2, F_RIGHT, A_INTERACT)
    a_get = _go_face_act(r, c, facing, tr, 2, sup_face, A_INTERACT)
    a_hold_pos = _go_face_act(r, c, facing, tr, 2, F_RIGHT, A_STAY)
    dump0_occ = _occupied(obs, r, c, tr, 0)
    a_dump = jnp.where(
        ~dump0_occ,
        _go_face_act(r, c, facing, tr, 1, F_LEFT, A_INTERACT),
        _go_face_act(r, c, facing, tr, 1, sup_face, A_INTERACT),
    )

    action = jnp.where(
        holding,
        jnp.where(~target_occ, a_place, a_dump),
        jnp.where(any_wrong, a_pick_wrong, jnp.where(~target_occ, a_get, a_hold_pos)),
    )
    branch = jnp.where(
        holding,
        jnp.where(~target_occ, 0, 1),
        jnp.where(any_wrong, 2, jnp.where(~target_occ, 3, 4)),
    )

    new_mem = mem.at[0].set(jnp.minimum(mem[0] + 1, 100000))
    new_mem = new_mem.at[1].set(rec)
    trace = {
        'row': _i32(r), 'col': _i32(c), 'recipe': _i32(rec),
        'branch': _i32(branch), 'occ1': _i32(occ1), 'occ3': _i32(occ3),
    }
    return _i32(action), new_mem, trace


# ---------------------------------------------------------------------------
# agent_1: solo cook in the right room
# ---------------------------------------------------------------------------

def init_agent_1():
    return jnp.zeros((MEM_SIZE,), dtype=jnp.int32)


# side counters of the right room: (counter row, counter col, access row, access col, facing)
SIDE_ROW = (0, 4, 0, 4)
SIDE_COL = (5, 5, 6, 6)
SIDE_ACC_ROW = (1, 3, 1, 3)
SIDE_FACE = (F_UP, F_DOWN, F_UP, F_DOWN)


def _side_actions(r, c, facing):
    return [_go_face_act(r, c, facing, SIDE_ACC_ROW[i], SIDE_COL[i], SIDE_FACE[i], A_INTERACT)
            for i in range(4)]


def _side_code(obs, r, c, i, remembered):
    """Item code on side counter i: 0 empty, 1 ing0, 2 ing1, 3 ing2, 4 plate/other.

    Uses the current observation when the cell is visible, else the remembered code.
    """
    gr, gc = SIDE_ROW[i], SIDE_COL[i]
    li = gr - r + 2
    lj = gc - c + 2
    valid = (li >= 0) & (li <= 4) & (lj >= 0) & (lj <= 4)
    item = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), 29:34]
    is_plate = item[0] > 0
    counts = item[2:5]
    code = jnp.where(is_plate, 4, jnp.where(counts[0] > 0, 1, jnp.where(counts[1] > 0, 2, jnp.where(counts[2] > 0, 3, 0))))
    return jnp.where(valid, code, remembered)


def _dump_action(r, c, facing, codes):
    """Drop the held item on the first free side counter; STAY if all are full."""
    acts = _side_actions(r, c, facing)
    out = _i32(A_STAY)
    for i in (3, 2, 1, 0):
        out = jnp.where(codes[i] == 0, acts[i], out)
    return out


def _reuse_action(r, c, facing, codes, k, default):
    """Pick ingredient k from a side counter that holds it, else default."""
    acts = _side_actions(r, c, facing)
    out = default
    for i in (3, 2, 1, 0):
        out = jnp.where(codes[i] == k + 1, acts[i], out)
    return out


def agent_1_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs).astype(jnp.int32)
    mem = jnp.where(reset, init_agent_1(), memory)
    since_del = mem[0]
    last_sig = mem[1]
    last_change = mem[2]
    prev_t = mem[3]
    prev_b = mem[4]
    target = mem[6]        # 0 = none, else ingredient id + 1 committed for fetching
    guess = mem[7]         # fallback recipe guess (0 or 1)
    ever_sig = mem[8]      # 1 once any signal has been seen
    fb_used = mem[9]       # 1 once a fallback guess has been used
    cycle_fb = mem[10]     # 1 if the current cook cycle was started by a fallback guess

    # localise from the pot at global (2,4)
    flat = jnp.argmax(obs[:, :, CH_POT].reshape(-1))
    iP = flat // 5
    jP = flat % 5
    r = 4 - iP
    c = 6 - jP
    facing = _facing(obs)

    inv = obs[2, 2, 5:10]
    has_dish = inv[1] > 0
    has_plate = (inv[0] > 0) & ~has_dish
    ing_counts = inv[2:5]
    has_ing = (jnp.sum(ing_counts) > 0) & ~has_dish & ~has_plate
    held_k = jnp.argmax(ing_counts).astype(jnp.int32)
    hands_empty = ~has_dish & ~has_plate & ~has_ing

    pot_cooked = obs[iP, jP, 30] > 0
    pot_counts = obs[iP, jP, 31:34]
    pot_count = jnp.sum(pot_counts)
    pot_timer = obs[iP, jP, CH_TIMER]
    pot_cooking = (pot_timer > 0) & ~pot_cooked
    pot_idle = ~pot_cooking & ~pot_cooked
    pot_empty = pot_idle & (pot_count == 0)
    pot_k = jnp.argmax(pot_counts).astype(jnp.int32)

    # signal decoding (counters in column 3 are visible only from column 5)
    vis = c == 5
    occ_t = _occupied(obs, r, c, 1, 3)
    occ_m = _occupied(obs, r, c, 2, 3)
    occ_b = _occupied(obs, r, c, 3, 3)
    new_t = vis & occ_t & (prev_t == 0)
    new_b = vis & occ_b & (prev_b == 0)
    last_change = jnp.where(new_b, 2, jnp.where(new_t, 1, last_change))
    both_val = jnp.where(last_change == 0, 2, last_change)
    dec = jnp.where(
        occ_t & ~occ_b, 1,
        jnp.where(occ_b & ~occ_t, 2,
                  jnp.where(occ_t & occ_b, both_val, jnp.where(occ_m, 2, 0))))
    sig = jnp.where(vis, dec, 0)
    have_sig = sig > 0
    sig_k = sig - 1
    ever_sig = jnp.where(have_sig, 1, ever_sig)
    timeout = jnp.where((ever_sig == 0) & (fb_used == 1), SIGNAL_TIMEOUT_NOSIG, SIGNAL_TIMEOUT)
    fallback = since_del >= timeout

    # candidate actions
    a_deliver = _go_face_act(r, c, facing, 2, 6, F_RIGHT, A_INTERACT)
    a_pot_interact = _go_face_act(r, c, facing, 2, 5, F_LEFT, A_INTERACT)
    a_wait_pot = _go_face_act(r, c, facing, 2, 5, F_LEFT, A_STAY)
    plate_row = jnp.where(r <= 2, 1, 3)
    a_plate = _go_face_act(r, c, facing, plate_row, 5, F_LEFT, A_INTERACT)
    codes = [_side_code(obs, r, c, i, mem[11 + i]) for i in range(4)]
    a_fetch0 = _reuse_action(r, c, facing, codes, 0, _go_face_act(r, c, facing, 1, 6, F_RIGHT, A_INTERACT))
    a_fetch1 = _reuse_action(r, c, facing, codes, 1, _go_face_act(r, c, facing, 3, 6, F_RIGHT, A_INTERACT))
    a_dump = _dump_action(r, c, facing, codes)

    def fetch(k):
        return jnp.where(k == 0, a_fetch0, jnp.where(k == 1, a_fetch1, a_wait_pot))

    # holding an ingredient
    pot_accepts = pot_idle & (pot_count < 3) & ((pot_count == 0) | (pot_k == held_k))
    match = have_sig & (sig_k == held_k)
    conflict = have_sig & (sig_k != held_k)
    fb_ok = fallback & (guess == held_k)
    a_hold_empty = jnp.where(
        ~vis, a_wait_pot,
        jnp.where(match, a_pot_interact,
                  jnp.where(conflict, a_dump,
                            jnp.where(fb_ok, a_pot_interact, a_wait_pot))))
    a_holding = jnp.where(
        pot_accepts,
        jnp.where(pot_count > 0, a_pot_interact, a_hold_empty),
        a_dump,
    )

    # empty hands: commit to a fetch target while the counters are visible and
    # keep it (in memory) while walking to the supply in column 6.
    target_vis = jnp.where(have_sig, sig_k + 1, jnp.where(fallback, guess + 1, 0))
    target_now = jnp.where(vis, target_vis, target)
    a_empty_pot = jnp.where(
        target_now > 0, fetch(target_now - 1),
        jnp.where(~vis, _i32(A_LEFT), a_wait_pot))
    a_empty = jnp.where(
        pot_cooking | pot_cooked, a_plate,
        jnp.where(pot_idle & (pot_count > 0), fetch(pot_k), a_empty_pot))

    # plate in hand
    a_with_plate = jnp.where(pot_cooked, a_pot_interact,
                             jnp.where(pot_cooking, a_wait_pot, a_dump))

    action = jnp.where(
        has_dish, a_deliver,
        jnp.where(has_plate, a_with_plate,
                  jnp.where(has_ing, a_holding, a_empty)))
    action = _i32(action)

    delivering = has_dish & (r == 2) & (c == 6) & (facing == F_RIGHT) & (action == A_INTERACT)
    since_del_next = jnp.where(delivering, 0, jnp.minimum(since_del + 1, 100000))

    # fallback bookkeeping: an empty-pot insertion under fallback marks the cycle
    inserting_empty = has_ing & pot_empty & (action == A_INTERACT) & (r == 2) & (c == 5) & (facing == F_LEFT)
    fb_insert = inserting_empty & ~match & fb_ok
    cycle_fb_next = jnp.where(delivering, 0, jnp.where(fb_insert, 1, cycle_fb))
    fb_used_next = jnp.where(fb_insert, 1, fb_used)
    # Alternate the guess after a fallback delivery only while the partner has
    # never signalled (a wrong delivery keeps the recipe, so alternating cannot
    # spiral).  Once any signal has been seen, "nothing" most plausibly means
    # recipe 0, so keep the guess at 0.
    guess_next = jnp.where(ever_sig == 1, 0, jnp.where(delivering & (cycle_fb == 1), 1 - guess, guess))

    target_next = jnp.where(hands_empty & pot_empty, target_now, 0)

    new_mem = mem.at[0].set(since_del_next)
    new_mem = new_mem.at[1].set(jnp.where(vis, dec, last_sig))
    new_mem = new_mem.at[2].set(last_change)
    new_mem = new_mem.at[3].set(jnp.where(vis, occ_t.astype(jnp.int32), prev_t))
    new_mem = new_mem.at[4].set(jnp.where(vis, occ_b.astype(jnp.int32), prev_b))
    new_mem = new_mem.at[5].set(jnp.minimum(mem[5] + 1, 100000))
    new_mem = new_mem.at[6].set(_i32(target_next))
    new_mem = new_mem.at[7].set(_i32(guess_next))
    new_mem = new_mem.at[8].set(_i32(ever_sig))
    new_mem = new_mem.at[9].set(_i32(fb_used_next))
    new_mem = new_mem.at[10].set(_i32(cycle_fb_next))
    for i in range(4):
        new_mem = new_mem.at[11 + i].set(_i32(codes[i]))

    trace = {
        'row': _i32(r), 'col': _i32(c), 'sig': _i32(sig), 'since_del': _i32(since_del),
        'pot_count': _i32(pot_count), 'pot_timer': _i32(pot_timer),
        'has_ing': _i32(has_ing), 'has_plate': _i32(has_plate), 'has_dish': _i32(has_dish),
        'delivering': _i32(delivering), 'target': _i32(target_now),
        'guess': _i32(guess), 'fallback': _i32(fallback), 'cycle_fb': _i32(cycle_fb),
    }
    return action, new_mem, trace
