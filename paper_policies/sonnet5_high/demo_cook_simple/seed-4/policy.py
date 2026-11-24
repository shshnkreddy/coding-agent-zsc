"""Cooperative Overcooked-V2 (demo_cook_simple) policy.

Design summary
--------------
The board splits into two disjoint rooms connected only through a shared
pot cell. agent_0's room holds all three ingredient supplies and the
always-on recipe indicator; agent_1's room holds both plate piles, the
delivery station, and (redundantly) ingredient supplies 0 and 1. Neither
seat receives global coordinates, so each act call first *localizes*
itself by matching the observed static-object channels (walls, pot,
recipe indicator, plate piles, ingredient piles) against a precomputed
template for every reachable cell in its room, then runs a small
deterministic waypoint state machine:

  * agent_0 ("chef"): reads the recipe indicator (ground truth, always
    visible from its pot-adjacent waypoint) and repeatedly fetches the
    matching ingredient, depositing it in the pot. It never deposits an
    ingredient unless the pot is idle, uncooked, below capacity and free
    of a different ingredient type, so it can never introduce a mismatch.
  * agent_1 ("server"): watches the pot from its own pot-adjacent
    waypoint. It prefetches a plate as soon as cooking starts, scoops
    once the pot reports the cooked flag, and delivers. It only helps
    fill the pot with a raw ingredient when the pot's *observed*
    contents already reveal the needed type (never guesses blindly, so
    it cannot corrupt the recipe even when paired with an unfamiliar
    partner).

Both roles are therefore safe to pair with an independently produced
policy: agent_0 never needs information from a partner, and agent_1 only
acts on ingredient type once it has direct observational evidence.

Note: all lookup tables below are built with pure module-scope literals
and comprehensions (no module-scope function calls), since the runtime
source validator forbids calls at module/class scope. jnp conversions of
these plain-Python constants happen inside the act functions.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'localizing_chef_server',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# ---------------------------------------------------------------------------
# Static map knowledge (public layout facts, not runtime observations).
# ---------------------------------------------------------------------------

_ROWS = (0, 1, 2, 3, 4)
_COLS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
_OFFSETS = (-2, -1, 0, 1, 2)
_ZERO9 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

_WALLS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10),
    (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (4, 7), (4, 9), (4, 10),
)
_GOAL = ((2, 10),)
_POT = ((2, 7),)
_RECIPE_INDICATOR = ((0, 5), (4, 5))
_BUTTON = ()
_PLATE_PILE = ((1, 10), (3, 10))
_ING0_SUPPLY = ((0, 8), (1, 0))
_ING1_SUPPLY = ((3, 0), (4, 8))
_ING2_SUPPLY = ((0, 6), (4, 6))

# Static channel layout used for localization templates (matches absolute
# obs channels 20..28): wall, goal, pot, recipe_indicator, button,
# plate_pile, ing0_supply, ing1_supply, ing2_supply.
_GLOBAL_STATIC = [
    [
        [
            1 if (r, c) in _WALLS else 0,
            1 if (r, c) in _GOAL else 0,
            1 if (r, c) in _POT else 0,
            1 if (r, c) in _RECIPE_INDICATOR else 0,
            1 if (r, c) in _BUTTON else 0,
            1 if (r, c) in _PLATE_PILE else 0,
            1 if (r, c) in _ING0_SUPPLY else 0,
            1 if (r, c) in _ING1_SUPPLY else 0,
            1 if (r, c) in _ING2_SUPPLY else 0,
        ]
        for c in _COLS
    ]
    for r in _ROWS
]

_ROOM0_CANDIDATES = [(r, c) for r in (1, 2, 3) for c in (1, 2, 3, 4, 5, 6)]
_ROOM1_CANDIDATES = [(r, c) for r in (1, 2, 3) for c in (8, 9)]

_ROOM0_ROWS = [r for (r, c) in _ROOM0_CANDIDATES]
_ROOM0_COLS = [c for (r, c) in _ROOM0_CANDIDATES]
_ROOM1_ROWS = [r for (r, c) in _ROOM1_CANDIDATES]
_ROOM1_COLS = [c for (r, c) in _ROOM1_CANDIDATES]

_TEMPLATES0 = [
    [
        [
            (_GLOBAL_STATIC[sr + di][sc + dj] if (0 <= sr + di <= 4 and 0 <= sc + dj <= 10) else _ZERO9)
            for dj in _OFFSETS
        ]
        for di in _OFFSETS
    ]
    for (sr, sc) in _ROOM0_CANDIDATES
]

_TEMPLATES1 = [
    [
        [
            (_GLOBAL_STATIC[sr + di][sc + dj] if (0 <= sr + di <= 4 and 0 <= sc + dj <= 10) else _ZERO9)
            for dj in _OFFSETS
        ]
        for di in _OFFSETS
    ]
    for (sr, sc) in _ROOM1_CANDIDATES
]


def _localize(obs_static, templates_literal, rows_literal, cols_literal):
    templates = jnp.asarray(templates_literal, dtype=jnp.int32)
    rows = jnp.asarray(rows_literal, dtype=jnp.int32)
    cols = jnp.asarray(cols_literal, dtype=jnp.int32)
    diffs = templates - obs_static[None]
    sq = jnp.sum(diffs * diffs, axis=(1, 2, 3))
    idx = jnp.argmin(sq)
    return rows[idx], cols[idx]


# ---------------------------------------------------------------------------
# Action / direction constants (plain Python ints; wrapped by jnp inside
# functions as needed).
# ---------------------------------------------------------------------------

_ACT_RIGHT = 0
_ACT_DOWN = 1
_ACT_LEFT = 2
_ACT_UP = 3
_ACT_STAY = 4
_ACT_INTERACT = 5

_DIR_UP = 0
_DIR_DOWN = 1
_DIR_RIGHT = 2
_DIR_LEFT = 3


def _action_for_facing(target_face):
    return jnp.where(
        target_face == _DIR_UP, _ACT_UP,
        jnp.where(
            target_face == _DIR_DOWN, _ACT_DOWN,
            jnp.where(target_face == _DIR_RIGHT, _ACT_RIGHT, _ACT_LEFT),
        ),
    )


def _nav_action(pos_row, pos_col, target_row, target_col):
    dr = target_row - pos_row
    dc = target_col - pos_col
    return jnp.where(
        dr < 0, _ACT_UP,
        jnp.where(
            dr > 0, _ACT_DOWN,
            jnp.where(
                dc < 0, _ACT_LEFT,
                jnp.where(dc > 0, _ACT_RIGHT, _ACT_STAY),
            ),
        ),
    )


def _resolve_action(pos_row, pos_col, facing_id, target_row, target_col, target_face, interact_flag):
    at_target = (pos_row == target_row) & (pos_col == target_col)
    navigate_act = _nav_action(pos_row, pos_col, target_row, target_col)
    facing_ok = facing_id == target_face
    at_target_action = jnp.where(
        interact_flag,
        jnp.where(facing_ok, _ACT_INTERACT, _action_for_facing(target_face)),
        _ACT_STAY,
    )
    return jnp.where(at_target, at_target_action, navigate_act).astype(jnp.int32)


# ---------------------------------------------------------------------------
# agent_0: chef. Reads the recipe indicator, fetches matching ingredient,
# fills the pot.
# ---------------------------------------------------------------------------

_HOME0_R, _HOME0_C = 2, 6
_SUPPLY0_R, _SUPPLY0_C = 1, 1  # ingredient 0 pickup, face LEFT
_SUPPLY1_R, _SUPPLY1_C = 3, 1  # ingredient 1 pickup, face LEFT

# A small pool of counter cells for discarding a stale ingredient, rotated
# through so that one already-occupied counter from an earlier mismatch
# never permanently blocks future discards.
_DISCARD_ROWS = (1, 1, 1, 1, 3, 3, 3, 3)
_DISCARD_COLS = (4, 3, 2, 1, 4, 3, 2, 1)
_DISCARD_FACES = (_DIR_UP, _DIR_UP, _DIR_UP, _DIR_UP, _DIR_DOWN, _DIR_DOWN, _DIR_DOWN, _DIR_DOWN)
_NUM_DISCARD_SLOTS = 8

# Local coordinates valid only when standing at HOME0 (row range is always
# 0..4 with zero padding there, since self row is always 2, and columns
# 4..8 lie fully inside the map when self col is 6).
_IND_LOCAL = (0, 1)
_POT0_LOCAL = (2, 3)


_SETTLE_THRESHOLD = 8  # steps to wait at home with an empty pot before
# starting a fresh batch, to let an in-flight delivery (and any recipe
# resample it triggers) resolve before locking in an ingredient type.


def init_agent_0():
    return {
        "belief": jnp.asarray(-1, dtype=jnp.int32),  # recipe belief: -1 unknown, 0/1 known
        "settle": jnp.asarray(0, dtype=jnp.int32),
        "prev_stale": jnp.asarray(0, dtype=jnp.int32),
        "discard_slot": jnp.asarray(0, dtype=jnp.int32),
        "next_slot": jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)

    belief_in = jnp.where(reset, jnp.int32(-1), memory["belief"])
    settle_in = jnp.where(reset, jnp.int32(0), memory["settle"])
    prev_stale_in = jnp.where(reset, jnp.int32(0), memory["prev_stale"])
    discard_slot_in = jnp.where(reset, jnp.int32(0), memory["discard_slot"])
    next_slot_in = jnp.where(reset, jnp.int32(0), memory["next_slot"])

    pos_row, pos_col = _localize(obs[:, :, 20:29], _TEMPLATES0, _ROOM0_ROWS, _ROOM0_COLS)
    facing_id = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    self_inv = obs[2, 2, 5:10]
    ing0, ing1, ing2 = self_inv[2], self_inv[3], self_inv[4]
    holding_any = (ing0 + ing1 + ing2) > 0
    held_type = jnp.where(ing0 > 0, 0, jnp.where(ing1 > 0, 1, 2))

    at_home = (pos_row == _HOME0_R) & (pos_col == _HOME0_C)

    ind_ing0 = obs[_IND_LOCAL[0], _IND_LOCAL[1], 36]
    ind_ing1 = obs[_IND_LOCAL[0], _IND_LOCAL[1], 37]
    belief = jnp.where(
        at_home,
        jnp.where(ind_ing0 > 0, 0, jnp.where(ind_ing1 > 0, 1, belief_in)),
        belief_in,
    ).astype(jnp.int32)
    belief_known = belief >= 0

    pot_cooked = obs[_POT0_LOCAL[0], _POT0_LOCAL[1], 30]
    pot_ing0 = obs[_POT0_LOCAL[0], _POT0_LOCAL[1], 31]
    pot_ing1 = obs[_POT0_LOCAL[0], _POT0_LOCAL[1], 32]
    pot_ing2 = obs[_POT0_LOCAL[0], _POT0_LOCAL[1], 33]
    pot_timer = obs[_POT0_LOCAL[0], _POT0_LOCAL[1], 39]

    own_for_belief = jnp.where(belief == 0, pot_ing0, pot_ing1)
    other_for_belief = (pot_ing0 + pot_ing1 + pot_ing2) - own_for_belief
    pot_empty = (pot_ing0 + pot_ing1 + pot_ing2 == 0) & (pot_timer == 0) & (pot_cooked == 0)

    # Free-running counter of steps elapsed since we last *witnessed* (while
    # at home) the pot occupied/cooking. It only resets when we can confirm
    # a new batch is actually underway; a brief errand away from home (e.g.
    # discarding a stale ingredient) does not reset it, since the pot can
    # only go from empty to occupied through a deposit, not on its own. This
    # gives an in-flight delivery (and any recipe resample it triggers) time
    # to land before we lock in a fresh batch's first ingredient.
    settle = jnp.where(at_home & (~pot_empty), jnp.int32(0), jnp.minimum(settle_in + 1, _SETTLE_THRESHOLD))
    settled = settle >= _SETTLE_THRESHOLD
    is_fresh_batch = (own_for_belief == 0) & (other_for_belief == 0)

    pot_ok_to_fetch = (
        belief_known & (pot_timer == 0) & (pot_cooked == 0) & (own_for_belief < 3) & (other_for_belief == 0)
        & (~is_fresh_batch | settled)
    )

    # A held ingredient can go stale if the recipe indicator changes belief
    # while the ingredient is already in hand (e.g. fetched under an old
    # belief, or picked up as a surplus while a partner finishes filling the
    # pot). Only deposit when the held type still matches the *current*
    # belief; otherwise discard it on the nearby counter instead of risking
    # a mismatched pot.
    type_matches_belief = belief_known & (held_type == belief)
    stale_hold = holding_any & (~type_matches_belief)

    # Rotate through a pool of discard counters: a slot used by an earlier
    # mismatch stays occupied forever (no trash action exists), so reusing
    # the same single counter would eventually deadlock every future
    # mismatch. Assign a fresh slot only when a stale hold newly begins.
    rising_stale = stale_hold & (prev_stale_in == 0)
    discard_slot = jnp.where(rising_stale, next_slot_in, discard_slot_in)
    next_slot = jnp.where(rising_stale, (next_slot_in + 1) % _NUM_DISCARD_SLOTS, next_slot_in)
    discard_rows = jnp.asarray(_DISCARD_ROWS, dtype=jnp.int32)
    discard_cols = jnp.asarray(_DISCARD_COLS, dtype=jnp.int32)
    discard_faces = jnp.asarray(_DISCARD_FACES, dtype=jnp.int32)
    discard_row = discard_rows[discard_slot]
    discard_col = discard_cols[discard_slot]
    discard_face = discard_faces[discard_slot]

    own_for_held = jnp.where(held_type == 0, pot_ing0, jnp.where(held_type == 1, pot_ing1, pot_ing2))
    other_for_held = (pot_ing0 + pot_ing1 + pot_ing2) - own_for_held
    # The settle gate also protects the very first deposit into a freshly
    # emptied pot: an already-held ingredient (e.g. a surplus fetched while
    # a partner topped off the previous batch) must not be dropped in until
    # the pot has stayed empty long enough for an in-flight delivery's
    # resample to land, or it can silently start the next batch with a
    # stale ingredient type.
    deposit_ok = (
        type_matches_belief & (pot_timer == 0) & (pot_cooked == 0) & (own_for_held < 3) & (other_for_held == 0)
        & (~pot_empty | settled)
    )

    supply_row = jnp.where(belief == 0, _SUPPLY0_R, _SUPPLY1_R)
    supply_col = jnp.where(belief == 0, _SUPPLY0_C, _SUPPLY1_C)

    target_row = jnp.where(
        stale_hold, discard_row,
        jnp.where(holding_any, _HOME0_R,
                  jnp.where(at_home, jnp.where(pot_ok_to_fetch, supply_row, _HOME0_R),
                            jnp.where(belief_known, supply_row, _HOME0_R))),
    )
    target_col = jnp.where(
        stale_hold, discard_col,
        jnp.where(holding_any, _HOME0_C,
                  jnp.where(at_home, jnp.where(pot_ok_to_fetch, supply_col, _HOME0_C),
                            jnp.where(belief_known, supply_col, _HOME0_C))),
    )
    target_face = jnp.where(
        stale_hold, discard_face,
        jnp.where(holding_any, _DIR_RIGHT,
                  jnp.where(at_home, jnp.where(pot_ok_to_fetch, _DIR_LEFT, _DIR_RIGHT),
                            jnp.where(belief_known, _DIR_LEFT, _DIR_RIGHT))),
    )
    interact_flag = jnp.where(
        stale_hold, True,
        jnp.where(holding_any, deposit_ok,
                  jnp.where(at_home, pot_ok_to_fetch, belief_known)),
    )

    action = _resolve_action(pos_row, pos_col, facing_id, target_row, target_col, target_face, interact_flag)

    next_memory = {
        "belief": jnp.where(reset, jnp.int32(-1), belief).astype(jnp.int32),
        "settle": jnp.where(reset, jnp.int32(0), settle).astype(jnp.int32),
        "prev_stale": jnp.where(reset, jnp.int32(0), stale_hold.astype(jnp.int32)).astype(jnp.int32),
        "discard_slot": jnp.where(reset, jnp.int32(0), discard_slot).astype(jnp.int32),
        "next_slot": jnp.where(reset, jnp.int32(0), next_slot).astype(jnp.int32),
    }
    trace = {
        "pos_row": pos_row.astype(jnp.int32),
        "pos_col": pos_col.astype(jnp.int32),
        "facing": facing_id,
        "belief": belief,
        "holding_any": holding_any,
        "held_type": held_type.astype(jnp.int32),
        "at_home": at_home,
        "pot_cooked": pot_cooked,
        "pot_ing0": pot_ing0,
        "pot_ing1": pot_ing1,
        "pot_timer": pot_timer,
        "deposit_ok": deposit_ok,
        "settle": settle,
        "stale_hold": stale_hold,
        "discard_slot": discard_slot,
        "target_row": target_row,
        "target_col": target_col,
    }
    return action, next_memory, trace


# ---------------------------------------------------------------------------
# agent_1: server. Scoops and delivers; helps fill the pot only when the
# observed pot contents already reveal the ingredient type.
# ---------------------------------------------------------------------------

_HOME1_R, _HOME1_C = 2, 8
_PLATE1_R, _PLATE1_C = 1, 9   # face RIGHT
_ING0_1_R, _ING0_1_C = 1, 8   # face UP
_ING1_1_R, _ING1_1_C = 3, 8   # face DOWN
_DELIVER1_R, _DELIVER1_C = 2, 9  # face RIGHT

_POT1_LOCAL = (2, 1)

_TASK_NONE = 0
_TASK_PLATE = 1
_TASK_ING0 = 2
_TASK_ING1 = 3


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)  # pending task: 0 none, 1 plate, 2 ing0, 3 ing1


def agent_1_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)

    task_in = jnp.where(reset, jnp.int32(0), memory)

    pos_row, pos_col = _localize(obs[:, :, 20:29], _TEMPLATES1, _ROOM1_ROWS, _ROOM1_COLS)
    facing_id = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    self_inv = obs[2, 2, 5:10]
    plate_flag, cooked_flag = self_inv[0], self_inv[1]
    ing0, ing1, ing2 = self_inv[2], self_inv[3], self_inv[4]
    holding_dish = cooked_flag > 0
    holding_plate = (plate_flag > 0) & (cooked_flag == 0)
    holding_ing = (ing0 + ing1 + ing2) > 0
    held_type = jnp.where(ing0 > 0, 0, jnp.where(ing1 > 0, 1, 2))

    at_home = (pos_row == _HOME1_R) & (pos_col == _HOME1_C)

    pot_cooked = obs[_POT1_LOCAL[0], _POT1_LOCAL[1], 30]
    pot_ing0 = obs[_POT1_LOCAL[0], _POT1_LOCAL[1], 31]
    pot_ing1 = obs[_POT1_LOCAL[0], _POT1_LOCAL[1], 32]
    pot_ing2 = obs[_POT1_LOCAL[0], _POT1_LOCAL[1], 33]
    pot_timer = obs[_POT1_LOCAL[0], _POT1_LOCAL[1], 39]

    need_plate = (pot_cooked > 0) | (pot_timer > 0)
    need_ing0 = (~need_plate) & (pot_timer == 0) & (pot_ing0 > 0) & (pot_ing0 < 3) & (pot_ing1 == 0) & (pot_ing2 == 0)
    need_ing1 = (~need_plate) & (pot_timer == 0) & (pot_ing1 > 0) & (pot_ing1 < 3) & (pot_ing0 == 0) & (pot_ing2 == 0)
    fetch_type = jnp.where(need_plate, _TASK_PLATE, jnp.where(need_ing0, _TASK_ING0, jnp.where(need_ing1, _TASK_ING1, _TASK_NONE)))

    task = jnp.where(at_home, fetch_type, task_in).astype(jnp.int32)

    own_for_held = jnp.where(held_type == 0, pot_ing0, jnp.where(held_type == 1, pot_ing1, pot_ing2))
    other_for_held = (pot_ing0 + pot_ing1 + pot_ing2) - own_for_held
    deposit_ok = (pot_timer == 0) & (pot_cooked == 0) & (own_for_held < 3) & (other_for_held == 0)

    # Fetch-phase target selection (only used when empty-handed).
    fetch_target_row = jnp.where(
        task == _TASK_PLATE, _PLATE1_R,
        jnp.where(task == _TASK_ING0, _ING0_1_R,
                  jnp.where(task == _TASK_ING1, _ING1_1_R, _HOME1_R)),
    )
    fetch_target_col = jnp.where(
        task == _TASK_PLATE, _PLATE1_C,
        jnp.where(task == _TASK_ING0, _ING0_1_C,
                  jnp.where(task == _TASK_ING1, _ING1_1_C, _HOME1_C)),
    )
    fetch_target_face = jnp.where(
        task == _TASK_PLATE, _DIR_RIGHT,
        jnp.where(task == _TASK_ING0, _DIR_UP,
                  jnp.where(task == _TASK_ING1, _DIR_DOWN, _DIR_LEFT)),
    )
    fetch_interact_flag = task != _TASK_NONE

    target_row = jnp.where(
        holding_dish, _DELIVER1_R,
        jnp.where(holding_plate, _HOME1_R,
                  jnp.where(holding_ing, _HOME1_R, fetch_target_row)),
    )
    target_col = jnp.where(
        holding_dish, _DELIVER1_C,
        jnp.where(holding_plate, _HOME1_C,
                  jnp.where(holding_ing, _HOME1_C, fetch_target_col)),
    )
    target_face = jnp.where(
        holding_dish, _DIR_RIGHT,
        jnp.where(holding_plate, _DIR_LEFT,
                  jnp.where(holding_ing, _DIR_LEFT, fetch_target_face)),
    )
    interact_flag = jnp.where(
        holding_dish, True,
        jnp.where(holding_plate, pot_cooked > 0,
                  jnp.where(holding_ing, deposit_ok, fetch_interact_flag)),
    )

    action = _resolve_action(pos_row, pos_col, facing_id, target_row, target_col, target_face, interact_flag)

    next_memory = jnp.where(reset, jnp.int32(0), task).astype(jnp.int32)
    trace = {
        "pos_row": pos_row.astype(jnp.int32),
        "pos_col": pos_col.astype(jnp.int32),
        "facing": facing_id,
        "task": task,
        "holding_dish": holding_dish,
        "holding_plate": holding_plate,
        "holding_ing": holding_ing,
        "held_type": held_type.astype(jnp.int32),
        "pot_cooked": pot_cooked,
        "pot_ing0": pot_ing0,
        "pot_ing1": pot_ing1,
        "pot_timer": pot_timer,
        "deposit_ok": deposit_ok,
        "target_row": target_row,
        "target_col": target_col,
    }
    return action, next_memory, trace
