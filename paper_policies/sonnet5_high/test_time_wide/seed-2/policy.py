"""Overcooked V2 (test_time_wide) cooperative policy.

Map facts (fixed, from context/environment/README.md + specification.json):
  - agent_0's connected floor is rows 1-2, cols 1-4. It can reach ingredient
    supplies 0/1, the plate pile, and the delivery station, but its view
    (radius 2) never reaches the recipe indicator R at (6,2) -- max visible
    row from row 2 is row 4.
  - agent_1's connected floor is rows 4-5, cols 1-4. It always sees R (row
    distance <=2 for any reachable cell) but cannot reach ingredient 0/1
    supplies, the plate pile, or delivery -- it structurally cannot execute
    any part of the cook/plate/deliver loop.
  - The only cells reachable from BOTH rooms are the two pots (3,1) and
    (3,3): agent_0 reaches them facing DOWN from row 2, agent_1 facing UP
    from row 4.
  - agent_0 never receives reward as an act-function input and never
    observes the recipe, so it cannot get feedback on whether a delivery
    was correct. The best *feedback-free* strategy against a recipe that
    resamples uniformly only after a correct delivery is a deterministic
    period-2 alternation between the two possible recipes: this guarantees
    a long-run 2/3 delivery-match rate (Markov chain over match/mismatch:
    mismatch always forces a match next step; match is followed by a match
    with probability 1/2), which strictly beats any fixed single-recipe
    guess (which can never recover once wrong).
  - Using both pots lets agent_0 fill/cook one recipe guess while the other
    cooks, roughly doubling delivery throughput versus a single-pot cycle,
    with no dependency on the partner.
  - agent_1 cannot contribute to throughput, so it broadcasts the live
    recipe by standing at the pot column that matches the recipe's list
    index (index0 -> pot (3,1), index1 -> pot (3,3)) and facing it. This
    is a zero-cost, self-sufficient behavior: it never touches ingredients
    or plates (so it can never corrupt a pot), and agent_0's correctness
    never depends on any partner reading this signal.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'dual_pot_alternating_chef',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

# ---------------------------------------------------------------------------
# Static map facts
# ---------------------------------------------------------------------------
H, W = 7, 6

# Action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids (observation channels 1-4 / 12-15 order).
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3

POT_A = (3, 1)
POT_B = (3, 3)
RECIPE_INDICATOR = (6, 2)

# Stage cells: where to stand + which way to face to interact with a target.
STAGE_ING0 = (1, 1, F_LEFT)     # -> ingredient 0 supply (1, 0)
STAGE_ING1 = (2, 1, F_LEFT)     # -> ingredient 1 supply (2, 0)
STAGE_POTA = (2, 1, F_DOWN)     # -> pot A (3, 1)
STAGE_POTB = (2, 3, F_DOWN)     # -> pot B (3, 3)
STAGE_PLATE = (1, 3, F_UP)      # -> plate pile (0, 3)
STAGE_DELIVER = (1, 2, F_UP)    # -> delivery station (0, 2)

# agent_1 "pointing" stages (broadcast which pot matches the live recipe).
STAGE_POINT_A = (4, 1, F_UP)
STAGE_POINT_B = (4, 3, F_UP)

AGENT0_REGION = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]
AGENT1_REGION = [(4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4)]

# Static-channel full board (channels 22-31: wall, goal, pot, recipe,
# button, plate_pile, ingredient0..3) used for self-localization.
_STATIC_POS = {
    0: [(0, 0), (0, 1), (0, 4), (0, 5), (3, 0), (3, 2), (3, 4), (3, 5),
        (5, 0), (5, 5), (6, 0), (6, 1), (6, 3), (6, 4), (6, 5)],  # wall
    1: [(0, 2)],       # goal
    2: [(3, 1), (3, 3)],  # pot
    3: [(6, 2)],       # recipe indicator
    4: [],             # button indicator (none in this layout)
    5: [(0, 3)],       # plate pile
    6: [(1, 0), (1, 5)],  # ingredient 0
    7: [(2, 0), (2, 5)],  # ingredient 1
    8: [],                 # ingredient 2 (unused)
    9: [(4, 0), (4, 5)],  # ingredient 3
}


def _build_full_board_list():
    board = [[[0.0] * 10 for _ in range(W)] for _ in range(H)]
    for chan, cells in _STATIC_POS.items():
        for (r, c) in cells:
            board[r][c][chan] = 1.0
    return board


def _build_crops_and_positions(region):
    board = _build_full_board_list()
    crops = []
    for (r, c) in region:
        crop = [[[0.0] * 10 for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                gr, gc = r + i - 2, c + j - 2
                if 0 <= gr < H and 0 <= gc < W:
                    crop[i][j] = board[gr][gc]
        crops.append(crop)
    return jnp.asarray(crops, dtype=jnp.float32), jnp.asarray(region, dtype=jnp.int32)


def _localize(obs, crops, positions):
    obs_static = obs[:, :, 22:32].astype(jnp.float32)
    diffs = jnp.sum((crops - obs_static[None, ...]) ** 2, axis=(1, 2, 3))
    idx = jnp.argmin(diffs)
    pos = positions[idx]
    return pos[0], pos[1]


def _read_cell(obs, self_row, self_col, global_row, global_col):
    li = jnp.clip(global_row - self_row + 2, 0, 4)
    lj = jnp.clip(global_col - self_col + 2, 0, 4)
    return obs[li, lj, :]


def _approach(self_row, self_col, self_facing, target_row, target_col, target_facing):
    at_pos = (self_row == target_row) & (self_col == target_col)
    row_diff = target_row - self_row
    col_diff = target_col - self_col
    move_action = jnp.where(
        row_diff > 0, DOWN,
        jnp.where(
            row_diff < 0, UP,
            jnp.where(
                col_diff > 0, RIGHT,
                jnp.where(col_diff < 0, LEFT, STAY),
            ),
        ),
    )
    turn_lookup = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)
    turn_action = turn_lookup[target_facing]
    pre_action = jnp.where(at_pos, turn_action, move_action)
    arrived = at_pos & (self_facing == target_facing)
    return pre_action.astype(jnp.int32), arrived


# ---------------------------------------------------------------------------
# agent_0: the chef. Memory = single int32 "next guess" (0 or 1), flipped
# exactly once per freshly-started pot fill (period-2 alternation over the
# temporal sequence of fill commits, independent of which physical pot).
# ---------------------------------------------------------------------------

def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)
    crops_0, pos_0 = _build_crops_and_positions(AGENT0_REGION)
    self_row, self_col = _localize(obs, crops_0, pos_0)
    self_facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    inv = obs[2, 2, 5:11]
    plate_flag, cooked_flag = inv[0], inv[1]
    ing_counts = inv[2:6]
    holding_ing = jnp.sum(ing_counts) >= 1
    held_type = jnp.argmax(ing_counts).astype(jnp.int32)

    holding_cooked = cooked_flag == 1
    holding_plate_only = (plate_flag == 1) & (cooked_flag == 0)
    holding_nothing = (~holding_cooked) & (~holding_plate_only) & (~holding_ing)

    pot_a_vals = _read_cell(obs, self_row, self_col, POT_A[0], POT_A[1])
    pot_b_vals = _read_cell(obs, self_row, self_col, POT_B[0], POT_B[1])
    potA_count = jnp.sum(pot_a_vals[34:38])
    potB_count = jnp.sum(pot_b_vals[34:38])
    potA_cooked = pot_a_vals[33] == 1
    potB_cooked = pot_b_vals[33] == 1
    potA_timer = pot_a_vals[44]
    potB_timer = pot_b_vals[44]
    potA_type = jnp.argmax(pot_a_vals[34:38]).astype(jnp.int32)
    potB_type = jnp.argmax(pot_b_vals[34:38]).astype(jnp.int32)

    both_full = (potA_count >= 3) & (potB_count >= 3)

    guess = memory

    case1 = holding_cooked
    case2 = holding_plate_only & (~case1)
    case3 = holding_nothing & both_full
    case5 = holding_nothing & (~both_full)
    case4 = holding_ing & (~case1)

    # --- case2: which pot to collect from ---
    use_potA_collect = potA_cooked | ((~potB_cooked) & (potA_timer <= potB_timer))
    collect_row = jnp.where(use_potA_collect, STAGE_POTA[0], STAGE_POTB[0])
    collect_col = jnp.where(use_potA_collect, STAGE_POTA[1], STAGE_POTB[1])
    collect_ready = jnp.where(use_potA_collect, potA_cooked, potB_cooked)

    # --- case4/case5: which pot to fill ---
    fill_use_potA = potA_count < 3
    fill_row = jnp.where(fill_use_potA, STAGE_POTA[0], STAGE_POTB[0])
    fill_col = jnp.where(fill_use_potA, STAGE_POTA[1], STAGE_POTB[1])
    fill_count = jnp.where(fill_use_potA, potA_count, potB_count)
    fill_existing_type = jnp.where(fill_use_potA, potA_type, potB_type)

    fetch_type = jnp.where(fill_count > 0, fill_existing_type, guess)
    fetch_row = jnp.where(fetch_type == 0, STAGE_ING0[0], STAGE_ING1[0])
    fetch_col = jnp.where(fetch_type == 0, STAGE_ING0[1], STAGE_ING1[1])

    conds = [case1, case2, case3, case4, case5]
    target_rows = [
        STAGE_DELIVER[0], collect_row, STAGE_PLATE[0], fill_row, fetch_row,
    ]
    target_cols = [
        STAGE_DELIVER[1], collect_col, STAGE_PLATE[1], fill_col, fetch_col,
    ]
    target_facings = [
        STAGE_DELIVER[2], STAGE_POTA[2], STAGE_PLATE[2], STAGE_POTA[2], STAGE_ING0[2],
    ]

    target_row = jnp.select(conds, target_rows).astype(jnp.int32)
    target_col = jnp.select(conds, target_cols).astype(jnp.int32)
    target_facing = jnp.select(conds, target_facings).astype(jnp.int32)

    pre_action, arrived = _approach(self_row, self_col, self_facing, target_row, target_col, target_facing)

    ready_actions = [
        INTERACT,
        jnp.where(collect_ready, INTERACT, STAY),
        INTERACT,
        INTERACT,
        INTERACT,
    ]
    ready_action = jnp.select(conds, ready_actions).astype(jnp.int32)

    action = jnp.where(arrived, ready_action, pre_action)

    commit_fetch = case5 & arrived & (fill_count == 0) & (action == INTERACT)
    new_guess = jnp.where(commit_fetch, 1 - guess, guess)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), new_guess.astype(jnp.int32))
    case_idx = jnp.select(conds, [1, 2, 3, 4, 5]).astype(jnp.int32)
    trace = {
        'row': self_row.astype(jnp.int32), 'col': self_col.astype(jnp.int32),
        'case': case_idx, 'guess': guess,
        'potA_count': potA_count.astype(jnp.int32), 'potB_count': potB_count.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


# ---------------------------------------------------------------------------
# agent_1: cannot reach ingredients/plates/delivery. It always sees the
# recipe indicator, so it stands at the pot column matching the live
# recipe's index, facing that pot. This never touches any item, so it can
# never corrupt a pot, and it costs nothing if a partner ignores it.
# ---------------------------------------------------------------------------

def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)
    crops_1, pos_1 = _build_crops_and_positions(AGENT1_REGION)
    self_row, self_col = _localize(obs, crops_1, pos_1)
    self_facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    r_vals = _read_cell(obs, self_row, self_col, RECIPE_INDICATOR[0], RECIPE_INDICATOR[1])
    recipe_type = jnp.where(r_vals[40] >= 3, 0, 1)

    target_row = jnp.where(recipe_type == 0, STAGE_POINT_A[0], STAGE_POINT_B[0]).astype(jnp.int32)
    target_col = jnp.where(recipe_type == 0, STAGE_POINT_A[1], STAGE_POINT_B[1]).astype(jnp.int32)
    target_facing = jnp.asarray(STAGE_POINT_A[2], dtype=jnp.int32)

    pre_action, arrived = _approach(self_row, self_col, self_facing, target_row, target_col, target_facing)
    action = jnp.where(arrived, STAY, pre_action)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    return action.astype(jnp.int32), next_memory, {}
