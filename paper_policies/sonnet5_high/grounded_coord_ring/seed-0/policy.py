"""Grounded-coordination policy for the grounded_coord_ring OCV2 layout.

Geometry (fixed, known at design time from the layout, not learned at runtime):
  * agent_0 always spawns in the 24-cell OUTER ring corridor and can never
    reach the pot directly.
  * agent_1 always spawns in the 8-cell INNER ring around the pot and is the
    only seat that can insert ingredients / scoop / (also) deliver.
  * Both delivery stations, both plate piles and ingredient supplies 0 and 1
    are reachable from *both* rings (they sit on the shared boundary), so
    each seat can act on those points on its own.
  * The always-on recipe indicator (R) is only ever visible from the outer
    ring; the inner ring never sees it, so agent_1 must pay the button cost
    to learn the recipe if it wants to act without relying on a partner.

Both seats are therefore given self-sufficient closed loops: agent_1 checks
the recipe via the button and completes full cook/plate/deliver cycles on
its own; agent_0 checks the recipe for free via the always-on indicator and
stages a matching ingredient on the shared hand-off counter for whichever
partner wants to use it. Neither seat depends on the other's cooperation to
make progress, which keeps behaviour robust to independently generated
partners while still being able to help a partner that reads the counters.

All per-step logic uses fixed-shape jnp ops (`jnp.where`, `jax.lax.switch`)
driven by re-deriving the exact global position each step from the
static-channel crop (channels 20-28), which is invariant to inventory /
recipe / cooking state and therefore gives an exact, stateless
self-localization within each seat's small reachable ring.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}

# ---------------------------------------------------------------------------
# Static map codes, rows 0-8 / cols 0-8 (context/environment/specification.json)
# 0 floor, 1 wall, 2 ingredient0, 3 ingredient1, 4 ingredient2,
# 5 recipe indicator, 6 button, 7 plate pile, 8 delivery, 9 pot.
# ---------------------------------------------------------------------------
_MAP_CODES = [
    [1, 1, 1, 4, 5, 4, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 6, 1, 1, 0, 1],
    [4, 0, 2, 0, 0, 0, 7, 0, 4],
    [5, 0, 8, 0, 9, 0, 8, 0, 5],
    [4, 0, 3, 0, 0, 0, 7, 0, 4],
    [1, 0, 1, 1, 6, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 4, 5, 4, 1, 1, 1],
]

# Static-channel (20..28) one-hot vector per map code (index 0..9).
_STATIC_VEC = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 floor
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 wall
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2 ingredient0
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 3 ingredient1
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 4 ingredient2
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5 recipe indicator
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6 button
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7 plate pile
    [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 8 delivery
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 9 pot
]

# Action ids.
_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5

# Ring cell lists, walked clockwise, verified adjacent and crop-unique offline
# (scripts/verify_layout.py).
_OUTER_RING = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
    (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
]
_INNER_RING = [
    (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (4, 3),
]

# Per-index clockwise / counter-clockwise action, verified offline.
_OUTER_CW = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
_OUTER_CCW = [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_INNER_CW = [0, 0, 1, 1, 2, 2, 3, 3]
_INNER_CCW = [1, 2, 2, 3, 3, 0, 0, 1]
_OUTER_N = 24
_INNER_N = 8

# Named ring indices (from _OUTER_RING / _INNER_RING above).
_IN_BUTTON = 1   # (3, 4)
_IN_ING0 = 0     # (3, 3)
_IN_ING1 = 6     # (5, 3)
_IN_POT = 7      # (4, 3)
_IN_PLATE = 2    # (3, 5)

_OUT_READSPOT0 = 2   # (1, 3)
_OUT_READSPOT1 = 16  # (7, 3)
_OUT_ING0 = 22       # (3, 1)
_OUT_ING1 = 20       # (5, 1)

# Local (row,col) offsets within the 5x5 crop of the object each phase reads,
# derived analytically from the fixed self-cell -> object-cell relationship.
_IN_BUTTON_LOCAL = (1, 2)   # self (3,4) reading L at (2,4)
_IN_POT_LOCAL = (2, 3)      # self (4,3) reading P at (4,4)
_OUT_R0_LOCAL = (1, 3)      # self (1,3) reading R at (0,4)
_OUT_DROP0_LOCAL = (3, 2)   # self (1,3) reading counter at (2,3)
_OUT_DROP1_LOCAL = (1, 2)   # self (7,3) reading counter at (6,3)

# facing one-hot index (order UP,DOWN,RIGHT,LEFT) -> action producing that facing
_FACE_TO_ACTION = [_UP, _DOWN, _RIGHT, _LEFT]


def _crop_at(row, col):
    out = []
    for i in range(5):
        row_out = []
        for j in range(5):
            gr, gc = row + i - 2, col + j - 2
            if 0 <= gr < 9 and 0 <= gc < 9:
                row_out.append(_STATIC_VEC[_MAP_CODES[gr][gc]])
            else:
                row_out.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        out.append(row_out)
    return out


def _ring_crops(ring):
    return [_crop_at(r, c) for (r, c) in ring]


def _localize(obs, ring):
    ref_crops = jnp.asarray(_ring_crops(ring), dtype=jnp.int32)
    static_crop = obs[:, :, 20:29]
    matches = jnp.all(jnp.equal(ref_crops, static_crop[None]), axis=(1, 2, 3))
    return jnp.argmax(matches).astype(jnp.int32)


def _current_facing_action(obs):
    onehot = obs[2, 2, 1:5]
    face_table = jnp.asarray(_FACE_TO_ACTION, dtype=jnp.int32)
    return face_table[jnp.argmax(onehot)]


def _nav_action(idx, target_idx, cw, ccw, n):
    cw_table = jnp.asarray(cw, dtype=jnp.int32)
    ccw_table = jnp.asarray(ccw, dtype=jnp.int32)
    d_cw = jnp.mod(target_idx - idx, n)
    d_ccw = jnp.mod(idx - target_idx, n)
    return jnp.where(d_cw <= d_ccw, cw_table[idx], ccw_table[idx])


# ---------------------------------------------------------------------------
# agent_1: inner ring, sole pot operator. Fully self-sufficient loop:
#   0 GOTO_BUTTON -> 1 READ_RECIPE -> 2 GOTO_INGREDIENT -> 3 GOTO_POT_INSERT
#   (loops 2<->3 three times) -> 4 GOTO_PLATE -> 5 WAIT_AND_SCOOP
#   -> 6 GOTO_DELIVER -> back to 0
# ---------------------------------------------------------------------------

def init_agent_0():
    return {
        'phase': jnp.asarray(0, dtype=jnp.int32),
        'recipe_id': jnp.asarray(0, dtype=jnp.int32),
    }


def init_agent_1():
    return {
        'phase': jnp.asarray(0, dtype=jnp.int32),
        'recipe_id': jnp.asarray(0, dtype=jnp.int32),
        'inserted': jnp.asarray(0, dtype=jnp.int32),
        'step': jnp.asarray(0, dtype=jnp.int32),
    }


# A fresh cycle (button press -> 3 inserts -> plate -> 20-step cook -> scoop ->
# deliver) needs roughly 40-50 steps in the worst case. Refusing to start (pay
# the button cost for) a new cycle once fewer than this many steps remain
# avoids paying -5 for a cycle that provably cannot be completed and
# delivered before the episode ends.
_MAX_STEPS = 400
_MIN_CYCLE_BUDGET = 45


def _agent1_targets(phase, recipe_id):
    target_idx = jax.lax.switch(
        phase,
        [
            lambda: jnp.asarray(_IN_BUTTON, jnp.int32),
            lambda: jnp.asarray(_IN_BUTTON, jnp.int32),
            lambda: jnp.where(recipe_id == 0, _IN_ING0, _IN_ING1).astype(jnp.int32),
            lambda: jnp.asarray(_IN_POT, jnp.int32),
            lambda: jnp.asarray(_IN_PLATE, jnp.int32),
            lambda: jnp.asarray(_IN_POT, jnp.int32),
            lambda: jnp.asarray(_IN_POT, jnp.int32),
        ],
    )
    desired_face = jax.lax.switch(
        phase,
        [
            lambda: jnp.asarray(_UP, jnp.int32),
            lambda: jnp.asarray(_UP, jnp.int32),
            lambda: jnp.asarray(_LEFT, jnp.int32),
            lambda: jnp.asarray(_RIGHT, jnp.int32),
            lambda: jnp.asarray(_RIGHT, jnp.int32),
            lambda: jnp.asarray(_RIGHT, jnp.int32),
            lambda: jnp.asarray(_LEFT, jnp.int32),
        ],
    )
    return target_idx, desired_face


def _agent1_arrival(phase, obs, recipe_id, inserted):
    def p0_press():
        return _INTERACT, jnp.asarray(1, jnp.int32), recipe_id, inserted

    def p1_read():
        new_recipe = jnp.where(obs[_IN_BUTTON_LOCAL[0], _IN_BUTTON_LOCAL[1], 36] > 0,
                                0, 1).astype(jnp.int32)
        return _STAY, jnp.asarray(2, jnp.int32), new_recipe, jnp.asarray(0, jnp.int32)

    def p2_pick():
        return _INTERACT, jnp.asarray(3, jnp.int32), recipe_id, inserted

    def p3_insert():
        new_inserted = inserted + 1
        next_phase = jnp.where(new_inserted < 3, 2, 4).astype(jnp.int32)
        return _INTERACT, next_phase, recipe_id, new_inserted

    def p4_plate():
        return _INTERACT, jnp.asarray(5, jnp.int32), recipe_id, inserted

    def p5_wait():
        ready = obs[_IN_POT_LOCAL[0], _IN_POT_LOCAL[1], 30] > 0
        action = jnp.where(ready, _INTERACT, _STAY).astype(jnp.int32)
        next_phase = jnp.where(ready, 6, 5).astype(jnp.int32)
        return action, next_phase, recipe_id, inserted

    def p6_deliver():
        return _INTERACT, jnp.asarray(0, jnp.int32), recipe_id, jnp.asarray(0, jnp.int32)

    branches = [p0_press, p1_read, p2_pick, p3_insert, p4_plate, p5_wait, p6_deliver]
    return jax.lax.switch(phase, branches)


def agent_1_act(obs, reset, memory, rng):
    del rng
    phase = memory['phase']
    recipe_id = memory['recipe_id']
    inserted = memory['inserted']
    step = memory['step']

    idx = _localize(obs, _INNER_RING)
    cur_face = _current_facing_action(obs)
    target_idx, desired_face = _agent1_targets(phase, recipe_id)

    at_target = idx == target_idx
    facing_ok = cur_face == desired_face

    nav_action = _nav_action(idx, target_idx, _INNER_CW, _INNER_CCW, _INNER_N)
    turn_action = desired_face

    arr_action, arr_phase, arr_recipe, arr_inserted = _agent1_arrival(
        phase, obs, recipe_id, inserted)

    action = jnp.where(~at_target, nav_action, jnp.where(~facing_ok, turn_action, arr_action))
    next_phase = jnp.where(at_target & facing_ok, arr_phase, phase)
    next_recipe = jnp.where(at_target & facing_ok, arr_recipe, recipe_id)
    next_inserted = jnp.where(at_target & facing_ok, arr_inserted, inserted)

    # Never start (pay the button cost for) a fresh cycle once too little of
    # the episode remains to plausibly finish and deliver it.
    out_of_budget = (phase == 0) & (step >= _MAX_STEPS - _MIN_CYCLE_BUDGET)
    action = jnp.where(out_of_budget, _STAY, action)
    next_phase = jnp.where(out_of_budget, phase, next_phase)
    next_recipe = jnp.where(out_of_budget, recipe_id, next_recipe)
    next_inserted = jnp.where(out_of_budget, inserted, next_inserted)

    action = jnp.asarray(action, dtype=jnp.int32)
    next_memory = {
        'phase': jnp.where(reset, jnp.asarray(0, jnp.int32), next_phase),
        'recipe_id': jnp.where(reset, jnp.asarray(0, jnp.int32), next_recipe),
        'inserted': jnp.where(reset, jnp.asarray(0, jnp.int32), next_inserted),
        'step': jnp.where(reset, jnp.asarray(0, jnp.int32), step + 1),
    }
    trace = {'phase': next_memory['phase'], 'ring_idx': idx}
    return action, next_memory, trace


# ---------------------------------------------------------------------------
# agent_0: outer ring, cannot reach the pot. Self-sufficient helper loop that
# reads the free always-on recipe indicator and stages a matching ingredient
# on the nearest hand-off counter for whichever partner wants to use it.
#   0 READ_RECIPE -> 1 GOTO_INGREDIENT -> 2 GOTO_DROP -> back to 0
# ---------------------------------------------------------------------------

def _agent0_targets(phase, recipe_id):
    target_idx = jax.lax.switch(
        phase,
        [
            lambda: jnp.asarray(_OUT_READSPOT0, jnp.int32),
            lambda: jnp.where(recipe_id == 0, _OUT_ING0, _OUT_ING1).astype(jnp.int32),
            lambda: jnp.where(recipe_id == 0, _OUT_READSPOT0, _OUT_READSPOT1).astype(jnp.int32),
        ],
    )
    desired_face = jax.lax.switch(
        phase,
        [
            lambda: jnp.asarray(_RIGHT, jnp.int32),  # unused (requires_facing=False)
            lambda: jnp.asarray(_RIGHT, jnp.int32),
            lambda: jnp.where(recipe_id == 0, _DOWN, _UP).astype(jnp.int32),
        ],
    )
    requires_facing = jax.lax.switch(
        phase,
        [lambda: jnp.asarray(False), lambda: jnp.asarray(True), lambda: jnp.asarray(True)],
    )
    return target_idx, desired_face, requires_facing


def _agent0_arrival(phase, obs, recipe_id):
    def p0_read():
        new_recipe = jnp.where(obs[_OUT_R0_LOCAL[0], _OUT_R0_LOCAL[1], 36] > 0,
                                0, 1).astype(jnp.int32)
        return _STAY, jnp.asarray(1, jnp.int32), new_recipe

    def p1_pick():
        return _INTERACT, jnp.asarray(2, jnp.int32), recipe_id

    def p2_drop():
        occ0 = jnp.sum(obs[_OUT_DROP0_LOCAL[0], _OUT_DROP0_LOCAL[1], 29:34]) > 0
        occ1 = jnp.sum(obs[_OUT_DROP1_LOCAL[0], _OUT_DROP1_LOCAL[1], 29:34]) > 0
        occupied = jnp.where(recipe_id == 0, occ0, occ1)
        action = jnp.where(occupied, _STAY, _INTERACT).astype(jnp.int32)
        next_phase = jnp.where(occupied, 2, 0).astype(jnp.int32)
        return action, next_phase, recipe_id

    branches = [p0_read, p1_pick, p2_drop]
    return jax.lax.switch(phase, branches)


def agent_0_act(obs, reset, memory, rng):
    del rng
    phase = memory['phase']
    recipe_id = memory['recipe_id']

    idx = _localize(obs, _OUTER_RING)
    cur_face = _current_facing_action(obs)
    target_idx, desired_face, requires_facing = _agent0_targets(phase, recipe_id)

    at_target = idx == target_idx
    facing_ok = jnp.where(requires_facing, cur_face == desired_face, True)

    nav_action = _nav_action(idx, target_idx, _OUTER_CW, _OUTER_CCW, _OUTER_N)
    turn_action = desired_face

    arr_action, arr_phase, arr_recipe = _agent0_arrival(phase, obs, recipe_id)

    action = jnp.where(~at_target, nav_action, jnp.where(~facing_ok, turn_action, arr_action))
    next_phase = jnp.where(at_target & facing_ok, arr_phase, phase)
    next_recipe = jnp.where(at_target & facing_ok, arr_recipe, recipe_id)

    action = jnp.asarray(action, dtype=jnp.int32)
    next_memory = {
        'phase': jnp.where(reset, jnp.asarray(0, jnp.int32), next_phase),
        'recipe_id': jnp.where(reset, jnp.asarray(0, jnp.int32), next_recipe),
    }
    trace = {'phase': next_memory['phase'], 'ring_idx': idx}
    return action, next_memory, trace
