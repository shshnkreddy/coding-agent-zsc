"""Cooperative Overcooked-v2 policy for the test_time_wide layout.

Map facts (from context/environment/README.md and specification.json):
  * Two disjoint rooms are separated by the pot row. agent_0 (upper room,
    rows 1-2) can reach both ingredient-0/1 supplies, both pots, the plate
    pile and the delivery station, but its view radius (2) never reaches the
    recipe indicator at row 6. agent_1 (lower room, rows 4-5) always sees the
    recipe indicator (row 6 is always within its view) but can never reach
    ingredient-0/1 supplies, the plate pile, or the delivery station.
  * Consequently the recipe must be communicated from agent_1 to agent_0
    through locally observable state. The signal used here is agent_1's
    position: it stands adjacent to the pot whose left-to-right order matches
    the needed ingredient id (ingredient 0 -> left pot, column 1; ingredient
    1 -> right pot, column 3), facing that pot. This is the most direct,
    ordinally-natural mapping available given the map's only two pots and two
    recipes, so it is a reasonable Schelling point for independently
    generated partners rather than a bespoke private code.
  * agent_0 self-localizes every step via the delivery station (always
    visible anywhere in its room); agent_1 self-localizes via the recipe
    indicator (always visible anywhere in its room). Both then navigate
    reactively (Manhattan, obstacle-free rooms) toward fixed global targets.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'pot_column_signal',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

# Action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# Observation channel ids (see context/environment/specification.json).
SELF_FACE_UP, SELF_FACE_DOWN, SELF_FACE_RIGHT, SELF_FACE_LEFT = 1, 2, 3, 4
SELF_PLATE, SELF_COOKED = 5, 6
SELF_ING0, SELF_ING1 = 7, 8
PARTNER_POS = 11
PARTNER_FACE_UP = 12
GOAL_CH = 23
RECIPE_IND_CH = 25
RECIPE_ING0, RECIPE_ING1 = 40, 41

# Fixed global landmarks.
GOAL_ROW, GOAL_COL = 0, 2
RECIPE_ROW, RECIPE_COL = 6, 2


def _find_local(obs, channel):
    """Local (row, col) of the arg-max cell of a channel, plus presence flag."""
    flat = obs[:, :, channel].reshape(-1)
    idx = jnp.argmax(flat)
    row = idx // 5
    col = idx % 5
    present = jnp.max(flat) > 0
    return row, col, present


def _localize(obs, channel, land_row, land_col):
    lr, lc, present = _find_local(obs, channel)
    self_row = land_row - (lr - 2)
    self_col = land_col - (lc - 2)
    return self_row, self_col, present


def _current_face(obs):
    up = obs[2, 2, SELF_FACE_UP] > 0
    down = obs[2, 2, SELF_FACE_DOWN] > 0
    right = obs[2, 2, SELF_FACE_RIGHT] > 0
    return jnp.select(
        [up, down, right],
        [jnp.asarray(UP), jnp.asarray(DOWN), jnp.asarray(RIGHT)],
        default=jnp.asarray(LEFT),
    )


def _navigate_and_act(self_row, self_col, target_row, target_col, desired_face, obs):
    """Move toward target; once there, turn to face desired_face, else interact."""
    delta_row = target_row - self_row
    delta_col = target_col - self_col
    move_vert = jnp.where(delta_row > 0, DOWN, UP)
    move_horiz = jnp.where(delta_col > 0, RIGHT, LEFT)
    current_face = _current_face(obs)
    final_action = jnp.where(current_face == desired_face, INTERACT, desired_face)
    action = jnp.where(
        delta_row != 0, move_vert, jnp.where(delta_col != 0, move_horiz, final_action)
    )
    arrived = (delta_row == 0) & (delta_col == 0)
    return action, arrived


def _navigate_only(self_row, self_col, target_row, target_col, arrived_action):
    delta_row = target_row - self_row
    delta_col = target_col - self_col
    move_vert = jnp.where(delta_row > 0, DOWN, UP)
    move_horiz = jnp.where(delta_col > 0, RIGHT, LEFT)
    action = jnp.where(
        delta_row != 0, move_vert, jnp.where(delta_col != 0, move_horiz, arrived_action)
    )
    return action


# ---------------------------------------------------------------------------
# agent_0: upper room. Stage machine: 0=SIGNAL,1=GET,2=DROP,3=PLATE,4=WAIT,5=DELIVER
# memory = [stage, branch, cnt] int32
# ---------------------------------------------------------------------------


def init_agent_0():
    return jnp.zeros((3,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    stage, branch, cnt = memory[0], memory[1], memory[2]

    self_row, self_col, _ = _localize(obs, GOAL_CH, GOAL_ROW, GOAL_COL)

    # --- SIGNAL stage: read partner's column via partner position channel ---
    # Also require the partner to be facing up (toward the pot it stands next
    # to), matching the natural "point at the pot" convention. A genuine
    # signaling partner always satisfies this, while a partner whose position
    # only coincidentally matches a signal column (e.g. random spawn) has
    # only a 1-in-4 chance of also facing up, cutting false positives.
    p_row_l, p_col_l, p_present = _find_local(obs, PARTNER_POS)
    partner_row = self_row + (p_row_l - 2)
    partner_col = self_col + (p_col_l - 2)
    partner_facing_up = obs[p_row_l, p_col_l, PARTNER_FACE_UP] > 0
    signal_col1 = (partner_row == 4) & (partner_col == 1) & p_present & partner_facing_up
    signal_col3 = (partner_row == 4) & (partner_col == 3) & p_present & partner_facing_up
    signal_valid = signal_col1 | signal_col3
    new_branch = jnp.where(signal_col1, 0, 1).astype(jnp.int32)

    check_action = _navigate_only(self_row, self_col, 2, 2, STAY)

    # --- branch-specific targets ---
    sup_row = jnp.where(branch == 0, 1, 2)
    sup_col = jnp.where(branch == 0, 1, 4)
    sup_face = jnp.where(branch == 0, LEFT, RIGHT)
    pot_row = 2
    pot_col = jnp.where(branch == 0, 1, 3)
    pot_face = DOWN
    ing_channel = jnp.where(branch == 0, SELF_ING0, SELF_ING1)

    get_action, _ = _navigate_and_act(self_row, self_col, sup_row, sup_col, sup_face, obs)
    holding_ing = obs[2, 2, ing_channel] > 0

    drop_action, _ = _navigate_and_act(self_row, self_col, pot_row, pot_col, pot_face, obs)
    hands_empty_of_ing = obs[2, 2, ing_channel] == 0

    plate_action, _ = _navigate_and_act(self_row, self_col, 1, 3, UP, obs)
    holding_plate = obs[2, 2, SELF_PLATE] > 0

    wait_action, _ = _navigate_and_act(self_row, self_col, pot_row, pot_col, pot_face, obs)
    holding_cooked = obs[2, 2, SELF_COOKED] > 0

    deliver_action, _ = _navigate_and_act(self_row, self_col, 1, 2, UP, obs)
    hands_fully_empty = (obs[2, 2, SELF_PLATE] == 0) & (obs[2, 2, SELF_COOKED] == 0)

    action = jnp.select(
        [stage == 0, stage == 1, stage == 2, stage == 3, stage == 4, stage == 5],
        [check_action, get_action, drop_action, plate_action, wait_action, deliver_action],
        default=jnp.asarray(STAY),
    ).astype(jnp.int32)

    next_stage_from0 = jnp.where(signal_valid, 1, 0)
    next_branch_from0 = jnp.where(signal_valid, new_branch, branch)
    next_cnt_from0 = jnp.where(signal_valid, 0, cnt)

    next_stage_from1 = jnp.where(holding_ing, 2, 1)

    done3 = cnt >= 2
    next_stage_from2 = jnp.where(hands_empty_of_ing, jnp.where(done3, 3, 1), 2)
    next_cnt_from2 = jnp.where(hands_empty_of_ing & jnp.logical_not(done3), cnt + 1, cnt)

    next_stage_from3 = jnp.where(holding_plate, 4, 3)
    next_stage_from4 = jnp.where(holding_cooked, 5, 4)
    next_stage_from5 = jnp.where(hands_fully_empty, 0, 5)
    next_cnt_from5 = jnp.where(hands_fully_empty, 0, cnt)

    next_stage = jnp.select(
        [stage == 0, stage == 1, stage == 2, stage == 3, stage == 4, stage == 5],
        [
            next_stage_from0,
            next_stage_from1,
            next_stage_from2,
            next_stage_from3,
            next_stage_from4,
            next_stage_from5,
        ],
        default=jnp.asarray(0),
    ).astype(jnp.int32)

    next_branch = jnp.where(stage == 0, next_branch_from0, branch).astype(jnp.int32)

    next_cnt = jnp.select(
        [stage == 0, stage == 2, stage == 5],
        [next_cnt_from0, next_cnt_from2, next_cnt_from5],
        default=cnt,
    ).astype(jnp.int32)

    next_memory = jnp.stack([next_stage, next_branch, next_cnt]).astype(jnp.int32)
    next_memory = jnp.where(reset, jnp.zeros((3,), dtype=jnp.int32), next_memory)

    trace = {
        'stage': stage.astype(jnp.int32),
        'branch': branch.astype(jnp.int32),
        'cnt': cnt.astype(jnp.int32),
        'self_row': self_row.astype(jnp.int32),
        'self_col': self_col.astype(jnp.int32),
    }
    return action, next_memory, trace


# ---------------------------------------------------------------------------
# agent_1: lower room. Stateless signaling: read recipe, walk to matching pot
# column, face up (toward the pot). Never interacts (never touches items).
# ---------------------------------------------------------------------------


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    self_row, self_col, _ = _localize(obs, RECIPE_IND_CH, RECIPE_ROW, RECIPE_COL)

    lr, lc, _ = _find_local(obs, RECIPE_IND_CH)
    ing0_count = obs[lr, lc, RECIPE_ING0]
    target_ing = jnp.where(ing0_count > 0, 0, 1)
    target_col = jnp.where(target_ing == 0, 1, 3)

    action = _navigate_only(self_row, self_col, 4, target_col, UP).astype(jnp.int32)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)

    trace = {
        'target_ing': target_ing.astype(jnp.int32),
        'self_row': self_row.astype(jnp.int32),
        'self_col': self_col.astype(jnp.int32),
    }
    return action, next_memory, trace
