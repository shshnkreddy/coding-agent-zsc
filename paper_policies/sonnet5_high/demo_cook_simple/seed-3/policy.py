"""Geometry-aware Overcooked V2 policy.

Layout facts (fixed, from context/environment/README.md + specification.json,
layout demo_cook_simple, 5x11 board):

  WWWWWR2W0WW
  0      W  B
  W     APA X
  1      W  B
  WWWWWR2W1WW

The two players occupy two floor rooms that share no floor connection at all
(the only shared cell is the pot, reachable from either side). agent_0's room
(rows 1-3, cols 1-6) can see the always-on recipe indicators (0,5)/(4,5) and
has access to ingredient supplies 0 and 1 (the only ingredients that ever
appear in a recipe). agent_1's room (rows 1-3, cols 8-9) can never see a
recipe indicator (out of its view-radius-2 reach) but is the only room with
access to the plate piles and the delivery station. This structurally forces
a division of labor that does not depend on a private convention: agent_0 is
the only seat that can know the recipe and load the pot; agent_1 is the only
seat that can plate and deliver. Both seats act purely from locally decodable
information (self-localization via the known static map, current recipe
display, current pot contents/timer, and own inventory) so the policy stays
sensible against an independently generated partner following the same
seat-specific affordances.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'seat_role_division',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# ---------------------------------------------------------------------------
# Static map facts (plain literals only; no calls at module scope).
# ---------------------------------------------------------------------------

_GRID = (
    "WWWWWR2W0WW",
    "0      W  B",
    "W     APA X",
    "1      W  B",
    "WWWWWR2W1WW",
)
_CHAR_TO_STATIC = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'B': 5, '0': 6, '1': 7, '2': 8}
_NUM_STATIC = 9
_PAD = 2

_ROOM0 = [(r, c) for r in (1, 2, 3) for c in (1, 2, 3, 4, 5, 6)]
_ROOM1 = [(r, c) for r in (1, 2, 3) for c in (8, 9)]

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3

POT_RC = (2, 7)
IND_A_RC = (0, 5)
IND_B_RC = (4, 5)

# Standing cell + required facing + facing action to reach it, per goal.
G_POT0 = (2, 6, F_RIGHT, RIGHT)
G_SUP0 = (1, 1, F_LEFT, LEFT)
G_SUP1 = (3, 1, F_LEFT, LEFT)

G_POT1 = (2, 8, F_LEFT, LEFT)
G_PLATE = (1, 9, F_RIGHT, RIGHT)
G_DELIVERY = (2, 9, F_RIGHT, RIGHT)

# Counters immediately beside the pot on agent_0's side, used to stage a
# fetched ingredient ahead of time while the pot is still busy (cooking or
# holding an uncollected dish), so the next batch starts faster.
STAGE_A_RC = (1, 7)
STAGE_B_RC = (3, 7)
G_STAGE_A = (1, 6, F_RIGHT, RIGHT)
G_STAGE_B = (3, 6, F_RIGHT, RIGHT)

UNKNOWN_TARGET = 2
STAGE_EMPTY, STAGE_ING0, STAGE_ING1 = 0, 1, 2


# ---------------------------------------------------------------------------
# Helpers (all calls happen inside function bodies, never at module scope).
# ---------------------------------------------------------------------------

def _padded_static_grid():
    height = len(_GRID)
    width = len(_GRID[0])
    padded_h = height + 2 * _PAD
    padded_w = width + 2 * _PAD
    padded = [[[0] * _NUM_STATIC for _ in range(padded_w)] for _ in range(padded_h)]
    for row in range(height):
        for col in range(width):
            ch = _GRID[row][col]
            if ch in _CHAR_TO_STATIC:
                padded[_PAD + row][_PAD + col][_CHAR_TO_STATIC[ch]] = 1
    return padded


def _room_tables(positions):
    padded = _padded_static_grid()
    crops = []
    for (r, c) in positions:
        crop = [[padded[r + i][c + j] for j in range(5)] for i in range(5)]
        crops.append(crop)
    pos_array = jnp.asarray(positions, dtype=jnp.int32)
    crop_array = jnp.asarray(crops, dtype=jnp.int32)
    return pos_array, crop_array


def _localize(obs, room_pos, room_crops):
    obs_static = obs[:, :, 20:29].astype(jnp.int32)
    diff = jnp.sum(jnp.abs(room_crops - obs_static[None]), axis=(1, 2, 3))
    idx = jnp.argmin(diff)
    pos = room_pos[idx]
    return pos[0], pos[1]


def _local_coord_and_valid(r, c, gr, gc):
    li = 2 + (gr - r)
    lj = 2 + (gc - c)
    valid = (li >= 0) & (li <= 4) & (lj >= 0) & (lj <= 4)
    li_c = jnp.clip(li, 0, 4)
    lj_c = jnp.clip(lj, 0, 4)
    return li_c, lj_c, valid


def _navigate(r, c, facing_id, target_r, target_c, face_id, face_action, interact_ok):
    at_target = (r == target_r) & (c == target_c)
    move_action = jnp.where(
        r < target_r, DOWN,
        jnp.where(r > target_r, UP,
                  jnp.where(c < target_c, RIGHT,
                            jnp.where(c > target_c, LEFT, STAY))))
    facing_ok = facing_id == face_id
    at_action = jnp.where(
        interact_ok & facing_ok, INTERACT,
        jnp.where(facing_ok, STAY, face_action))
    action = jnp.where(at_target, at_action, move_action)
    return action.astype(jnp.int32)


# ---------------------------------------------------------------------------
# agent_0: sees the recipe, fetches matching ingredient x3, loads the pot.
# ---------------------------------------------------------------------------

def init_agent_0():
    return {
        'target': jnp.asarray(UNKNOWN_TARGET, dtype=jnp.int32),
        'pot_count': jnp.asarray(0, dtype=jnp.int32),
        'stage_a': jnp.asarray(STAGE_EMPTY, dtype=jnp.int32),
        'stage_b': jnp.asarray(STAGE_EMPTY, dtype=jnp.int32),
    }


def _select(cond, a, b):
    return tuple(jnp.where(cond, ai, bi) for ai, bi in zip(a, b))


def agent_0_act(obs, reset, memory, rng):
    del rng
    room_pos, room_crops = _room_tables(_ROOM0)

    target_in = jnp.where(reset, jnp.int32(UNKNOWN_TARGET), memory['target']).astype(jnp.int32)
    pot_count_in = jnp.where(reset, jnp.int32(0), memory['pot_count']).astype(jnp.int32)
    stage_a_in = jnp.where(reset, jnp.int32(STAGE_EMPTY), memory['stage_a']).astype(jnp.int32)
    stage_b_in = jnp.where(reset, jnp.int32(STAGE_EMPTY), memory['stage_b']).astype(jnp.int32)

    r, c = _localize(obs, room_pos, room_crops)
    facing_id = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    li, lj, pot_valid = _local_coord_and_valid(r, c, POT_RC[0], POT_RC[1])
    pot_read = obs[li, lj, 31] + obs[li, lj, 32] + obs[li, lj, 33]
    pot_count = jnp.where(pot_valid, pot_read, pot_count_in).astype(jnp.int32)

    lia, lja, ind_a_valid = _local_coord_and_valid(r, c, IND_A_RC[0], IND_A_RC[1])
    lib, ljb, ind_b_valid = _local_coord_and_valid(r, c, IND_B_RC[0], IND_B_RC[1])
    ing0_a = obs[lia, lja, 36]
    ing1_a = obs[lia, lja, 37]
    ing0_b = obs[lib, ljb, 36]
    ing1_b = obs[lib, ljb, 37]
    ing0 = jnp.where(ind_a_valid, ing0_a, ing0_b)
    ing1 = jnp.where(ind_a_valid, ing1_a, ing1_b)
    ind_valid = ind_a_valid | ind_b_valid
    read_target = jnp.where(ing0 > 0, jnp.int32(0), jnp.where(ing1 > 0, jnp.int32(1), target_in))
    target = jnp.where(ind_valid, read_target, target_in).astype(jnp.int32)

    lsa, lsa2, stage_a_valid = _local_coord_and_valid(r, c, STAGE_A_RC[0], STAGE_A_RC[1])
    stage_a_ing0 = obs[lsa, lsa2, 31]
    stage_a_ing1 = obs[lsa, lsa2, 32]
    stage_a_read = jnp.where(stage_a_ing0 > 0, jnp.int32(STAGE_ING0),
                              jnp.where(stage_a_ing1 > 0, jnp.int32(STAGE_ING1), jnp.int32(STAGE_EMPTY)))
    stage_a = jnp.where(stage_a_valid, stage_a_read, stage_a_in).astype(jnp.int32)

    lsb, lsb2, stage_b_valid = _local_coord_and_valid(r, c, STAGE_B_RC[0], STAGE_B_RC[1])
    stage_b_ing0 = obs[lsb, lsb2, 31]
    stage_b_ing1 = obs[lsb, lsb2, 32]
    stage_b_read = jnp.where(stage_b_ing0 > 0, jnp.int32(STAGE_ING0),
                              jnp.where(stage_b_ing1 > 0, jnp.int32(STAGE_ING1), jnp.int32(STAGE_EMPTY)))
    stage_b = jnp.where(stage_b_valid, stage_b_read, stage_b_in).astype(jnp.int32)

    held0 = obs[2, 2, 7]
    held1 = obs[2, 2, 8]
    holding_any = (held0 > 0) | (held1 > 0)
    holding_target_ing = jnp.where(target == 0, held0 > 0, jnp.where(target == 1, held1 > 0, False))
    target_unknown = target == UNKNOWN_TARGET

    stage_a_empty = stage_a == STAGE_EMPTY
    stage_b_empty = stage_b == STAGE_EMPTY
    stage_slot_available = stage_a_empty | stage_b_empty
    stage_a_matches = jnp.where(target == 0, stage_a == STAGE_ING0, jnp.where(target == 1, stage_a == STAGE_ING1, False))
    stage_b_matches = jnp.where(target == 0, stage_b == STAGE_ING0, jnp.where(target == 1, stage_b == STAGE_ING1, False))

    # 5-tuples throughout: (goal_r, goal_c, face_id, face_action, interact_ok)
    default_result = (G_POT0[0], G_POT0[1], G_POT0[2], G_POT0[3], False)

    sup_r = jnp.where(target == 0, G_SUP0[0], G_SUP1[0])
    sup_c = jnp.where(target == 0, G_SUP0[1], G_SUP1[1])
    fetch_result = (sup_r, sup_c, G_SUP0[2], G_SUP0[3], True)

    pick_r = jnp.where(stage_a_matches, G_STAGE_A[0], G_STAGE_B[0])
    pick_c = jnp.where(stage_a_matches, G_STAGE_A[1], G_STAGE_B[1])
    pickup_result = (pick_r, pick_c, G_STAGE_A[2], G_STAGE_A[3], True)

    pickup_cond = (pot_count < 3) & (stage_a_matches | stage_b_matches)
    fetch_cond = (pot_count < 3) | stage_slot_available
    hands_empty_result = _select(
        target_unknown, default_result,
        _select(pickup_cond, pickup_result,
                _select(fetch_cond, fetch_result, default_result)))

    stage_a_res = (G_STAGE_A[0], G_STAGE_A[1], G_STAGE_A[2], G_STAGE_A[3], True)
    stage_b_res = (G_STAGE_B[0], G_STAGE_B[1], G_STAGE_B[2], G_STAGE_B[3], True)
    insert_res = (G_POT0[0], G_POT0[1], G_POT0[2], G_POT0[3], True)
    holding_correct_result = _select(
        pot_count < 3, insert_res,
        _select(stage_a_empty, stage_a_res,
                _select(stage_b_empty, stage_b_res, default_result)))

    mismatched_result = _select(
        stage_a_empty, stage_a_res,
        _select(stage_b_empty, stage_b_res, default_result))

    holding_result = _select(holding_target_ing, holding_correct_result, mismatched_result)

    final = _select(holding_any, holding_result, hands_empty_result)
    goal_r, goal_c, goal_face, goal_face_action, interact_ok = final
    goal_r = goal_r.astype(jnp.int32)
    goal_c = goal_c.astype(jnp.int32)
    goal_face = goal_face.astype(jnp.int32)
    goal_face_action = goal_face_action.astype(jnp.int32)

    action = _navigate(r, c, facing_id, goal_r, goal_c, goal_face, goal_face_action, interact_ok)

    next_memory = {
        'target': target,
        'pot_count': pot_count,
        'stage_a': stage_a,
        'stage_b': stage_b,
    }
    return action, next_memory, {}


# ---------------------------------------------------------------------------
# agent_1: never sees the recipe; fetches a plate, waits for the pot to be
# ready, collects the cooked dish, and delivers it.
# ---------------------------------------------------------------------------

def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng, memory
    room_pos, room_crops = _room_tables(_ROOM1)

    r, c = _localize(obs, room_pos, room_crops)
    facing_id = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    li, lj, pot_valid = _local_coord_and_valid(r, c, POT_RC[0], POT_RC[1])
    pot_cooked = obs[li, lj, 30]
    pot_timer = obs[li, lj, 39]
    pot_ready = pot_valid & (pot_cooked > 0) & (pot_timer == 0)

    plate_flag = obs[2, 2, 5]
    cooked_flag = obs[2, 2, 6]
    holding_dish = cooked_flag > 0
    holding_plate_only = (plate_flag > 0) & (cooked_flag == 0)

    goal_r = jnp.where(holding_dish, G_DELIVERY[0],
                        jnp.where(holding_plate_only, G_POT1[0], G_PLATE[0])).astype(jnp.int32)
    goal_c = jnp.where(holding_dish, G_DELIVERY[1],
                        jnp.where(holding_plate_only, G_POT1[1], G_PLATE[1])).astype(jnp.int32)
    goal_face = jnp.where(holding_dish, G_DELIVERY[2],
                           jnp.where(holding_plate_only, G_POT1[2], G_PLATE[2])).astype(jnp.int32)
    goal_face_action = jnp.where(holding_dish, G_DELIVERY[3],
                                  jnp.where(holding_plate_only, G_POT1[3], G_PLATE[3])).astype(jnp.int32)

    interact_ok = jnp.where(holding_dish, True, jnp.where(holding_plate_only, pot_ready, True))

    action = _navigate(r, c, facing_id, goal_r, goal_c, goal_face, goal_face_action, interact_ok)

    next_memory = jnp.asarray(0, dtype=jnp.int32)
    return action, next_memory, {}
