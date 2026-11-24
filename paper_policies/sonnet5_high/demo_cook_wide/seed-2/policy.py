"""Overcooked V2 (demo_cook_wide) policy.

Map geometry (see context/environment/README.md): the two seats occupy
disjoint rooms that only share access to a single pot. agent_0's room has
the plate piles and the delivery station but the recipe indicator is
always out of its view range (too far south). agent_1's room has the
recipe indicator and ingredient piles but no plates or delivery station.

Because this asymmetry is a fixed property of the layout (not a private
convention), any competent independently-generated partner controlling
agent_1 has to read the recipe to be useful, and any competent partner
controlling agent_0 cannot verify ingredient choice at all. So each seat
here plays the role forced on it by what it can observe:

- agent_1 reads the recipe indicator, fetches the matching ingredient, and
  feeds the shared pot. It never guesses when the recipe is not visible;
  it keeps its last known target and searches for it.
- agent_0 never gambles on ingredient choice (it cannot see the recipe at
  all), and instead focuses on plate fetch -> scoop -> deliver, which is
  always safe and always useful regardless of who fills the pot.

agent_0's room fits entirely inside its 5x5 view, so it just navigates
greedily toward whichever target (plate pile / pot / delivery goal) it
currently wants. agent_1's room is larger than its view, so it localizes
itself from known landmarks and routes deterministically instead of
wandering; see the comment above `_LANDMARKS` below.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'role_from_geometry',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# Action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# Direction enum used throughout (matches the observation's facing-channel
# order UP, DOWN, RIGHT, LEFT).
_DIR_UP, _DIR_DOWN, _DIR_RIGHT, _DIR_LEFT = 0, 1, 2, 3
_MOVE_FOR_DIR_LIST = [UP, DOWN, RIGHT, LEFT]


def _move_for_dir(dir_idx):
    return jnp.asarray(_MOVE_FOR_DIR_LIST, dtype=jnp.int32)[dir_idx]

# Observation channel indices (see context/environment/README.md).
CH_SELF_PLATE, CH_SELF_COOKED = 5, 6
CH_SELF_ING0, CH_SELF_ING1 = 7, 8
CH_GOAL, CH_POT, CH_PLATE_PILE = 23, 24, 27
CH_RECIPE_IND = 25
CH_ING_SUPPLY0, CH_ING_SUPPLY1, CH_ING_SUPPLY3 = 28, 29, 31
CH_ITEM_COOKED = 33
CH_RECIPE_ING0, CH_RECIPE_ING1 = 40, 41


def _facing_idx(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _find_local(channel2d):
    flat = channel2d.reshape(-1)
    idx = jnp.argmax(flat)
    row = (idx // 5).astype(jnp.int32)
    col = (idx % 5).astype(jnp.int32)
    visible = flat[idx] > 0
    return row, col, visible


# Static/object channels whose presence at a cell blocks movement into it
# (everything except plain floor and the transient spawn marker).
_STATIC_BLOCK_LO, _STATIC_BLOCK_HI = 22, 32


def _blocked(obs, row, col):
    return jnp.any(obs[row, col, _STATIC_BLOCK_LO:_STATIC_BLOCK_HI] > 0)


def _navigate(obs, dr, dc, facing_idx):
    """Greedy, obstacle-aware step toward a local target offset (dr, dc).

    Returns INTERACT when already adjacent and correctly facing the
    target; otherwise a movement/turn action that reduces the offset,
    preferring the horizontal axis but falling back to the vertical axis
    (or vice versa) when the preferred axis is blocked by a wall/object,
    since this map's rooms are corridor-shaped rather than open.
    """
    dr = jnp.asarray(dr, jnp.int32)
    dc = jnp.asarray(dc, jnp.int32)
    is_adjacent = (jnp.abs(dr) + jnp.abs(dc)) == 1
    dir_adj = jnp.where(
        dr == -1, _DIR_UP,
        jnp.where(dr == 1, _DIR_DOWN, jnp.where(dc == 1, _DIR_RIGHT, _DIR_LEFT)),
    )
    move_action_adj = _move_for_dir(dir_adj)
    action_if_adjacent = jnp.where(facing_idx == dir_adj, INTERACT, move_action_adj)

    want_horizontal = dc != 0
    want_vertical = dr != 0
    horiz_dir = jnp.where(dc > 0, _DIR_RIGHT, _DIR_LEFT)
    vert_dir = jnp.where(dr > 0, _DIR_DOWN, _DIR_UP)
    horiz_dc = jnp.where(dc > 0, 1, -1)
    vert_dr = jnp.where(dr > 0, 1, -1)
    horiz_blocked = _blocked(obs, 2, 2 + horiz_dc)
    vert_blocked = _blocked(obs, 2 + vert_dr, 2)

    use_horizontal = want_horizontal & jnp.logical_not(horiz_blocked)
    use_vertical = jnp.logical_not(use_horizontal) & want_vertical & jnp.logical_not(vert_blocked)
    fallback_dir = jnp.where(want_horizontal, horiz_dir, vert_dir)
    dir_general = jnp.where(use_horizontal, horiz_dir, jnp.where(use_vertical, vert_dir, fallback_dir))
    move_action_general = _move_for_dir(dir_general)

    action = jnp.where(is_adjacent, action_if_adjacent, move_action_general)
    return action.astype(jnp.int32)


# --------------------------------------------------------------------------
# agent_0: plate fetch -> scoop -> deliver. Never touches raw ingredients
# because it structurally cannot see the recipe indicator.
# --------------------------------------------------------------------------

def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    facing = _facing_idx(obs)

    held_cooked = obs[2, 2, CH_SELF_COOKED] > 0
    held_plate = (obs[2, 2, CH_SELF_PLATE] > 0) & jnp.logical_not(held_cooked)

    goal_r, goal_c, _ = _find_local(obs[:, :, CH_GOAL])
    pot_r, pot_c, _ = _find_local(obs[:, :, CH_POT])
    plate_r, plate_c, _ = _find_local(obs[:, :, CH_PLATE_PILE])

    pot_ready = obs[pot_r, pot_c, CH_ITEM_COOKED] > 0

    action_deliver = _navigate(obs, goal_r - 2, goal_c - 2, facing)
    action_scoop_raw = _navigate(obs, pot_r - 2, pot_c - 2, facing)
    action_scoop = jnp.where(
        (action_scoop_raw == INTERACT) & jnp.logical_not(pot_ready), STAY, action_scoop_raw
    )
    action_getplate = _navigate(obs, plate_r - 2, plate_c - 2, facing)

    action = jnp.where(
        held_cooked, action_deliver, jnp.where(held_plate, action_scoop, action_getplate)
    )
    action = jnp.asarray(action, jnp.int32)

    next_memory = jnp.where(reset, init_agent_0(), memory)
    return action, next_memory, {}


# --------------------------------------------------------------------------
# agent_1: read the recipe, fetch the matching ingredient, feed the pot.
#
# agent_1's room (row 3 corridor plus the row-4 alcoves at columns 1,2,8,9)
# is bigger than the 5x5 view, so instead of blindly wandering we localize
# self (global row/col) from whichever known landmark is currently visible
# -- the pot, the always-on recipe indicator, or any ingredient pile -- and
# then route deterministically along the map's known corridor topology
# (align column on row 3, then step down into the row-4 alcove, or vice
# versa). Landmark coverage is complete across every reachable cell, so
# this needs no fallback search phase.
# --------------------------------------------------------------------------

# Global coordinates of landmarks, most reliable/central first.
_LANDMARKS = [
    (CH_POT, 2, 5),
    (CH_RECIPE_IND, 4, 5),
    (CH_ING_SUPPLY3, 4, 4),
    (CH_ING_SUPPLY3, 4, 6),
    (CH_ING_SUPPLY0, 4, 0),
    (CH_ING_SUPPLY0, 4, 10),
    (CH_ING_SUPPLY1, 5, 1),
    (CH_ING_SUPPLY1, 5, 9),
]


def _landmark_candidate(obs, channel, glob_row, glob_col):
    r, c, vis = _find_local(obs[:, :, channel])
    self_row = glob_row - (r - 2)
    self_col = glob_col - (c - 2)
    plausible = vis & (self_row >= 3) & (self_row <= 4) & (self_col >= 1) & (self_col <= 9)
    return self_row.astype(jnp.int32), self_col.astype(jnp.int32), plausible


def _localize_agent_1(obs):
    row = jnp.asarray(3, dtype=jnp.int32)
    col = jnp.asarray(5, dtype=jnp.int32)
    for channel, glob_row, glob_col in reversed(_LANDMARKS):
        cand_row, cand_col, plausible = _landmark_candidate(obs, channel, glob_row, glob_col)
        row = jnp.where(plausible, cand_row, row)
        col = jnp.where(plausible, cand_col, col)
    return row, col


def _route(self_row, self_col, stand_row, stand_col, facing, object_dir):
    """Deterministic step toward (stand_row, stand_col), then face+interact.

    Row 4 only connects to row 3 at the same column, so from row 4 always
    climb to row 3 first unless already at the destination cell.
    """
    at_dest = (self_row == stand_row) & (self_col == stand_col)
    turn_or_interact = jnp.where(facing == object_dir, INTERACT, _move_for_dir(object_dir))

    need_col_move = self_col != stand_col
    horiz_action = jnp.where(self_col < stand_col, RIGHT, LEFT)
    vert_action = jnp.where(self_row < stand_row, DOWN, UP)
    move_action = jnp.where(
        self_row == 4,
        UP,
        jnp.where(need_col_move, horiz_action, vert_action),
    )
    action = jnp.where(at_dest, turn_or_interact, move_action)
    return action.astype(jnp.int32)


def init_agent_1():
    return {
        'target': jnp.asarray(0, dtype=jnp.int32),
        # Column of the last wall counter a mismatched ingredient was
        # deposited onto, or -1 if none is pending. Used to detect a
        # failed deposit (counter already occupied) and move to a
        # different counter instead of retrying the same spot forever.
        'deposit_col': jnp.asarray(-1, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    facing = _facing_idx(obs)

    ing0 = obs[2, 2, CH_SELF_ING0]
    ing1 = obs[2, 2, CH_SELF_ING1]
    holding_nothing = (ing0 == 0) & (ing1 == 0)
    held_type = jnp.where(ing0 > 0, 0, 1)

    recipe0_visible = jnp.max(obs[:, :, CH_RECIPE_ING0]) > 0
    recipe1_visible = jnp.max(obs[:, :, CH_RECIPE_ING1]) > 0
    target = jnp.where(recipe0_visible, 0, jnp.where(recipe1_visible, 1, memory['target']))
    target = target.astype(jnp.int32)

    self_row, self_col = _localize_agent_1(obs)

    # Fetch phase: pick whichever alcove (left col1, right col9) is closer,
    # recomputed each step -- stable because the router always moves the
    # chosen side further into range, never back across the midline.
    side_left = self_col <= 5
    fetch_col = jnp.where(side_left, 1, 9)
    fetch_dir = jnp.where(target == 0, jnp.where(side_left, _DIR_LEFT, _DIR_RIGHT), _DIR_DOWN)
    action_fetch = _route(self_row, self_col, 4, fetch_col, facing, fetch_dir)

    # Deliver-to-pot phase, only taken when the held ingredient still
    # matches the current target (never place a mismatched ingredient).
    action_to_pot = _route(self_row, self_col, 3, 5, facing, _DIR_UP)

    # Mismatch phase: the recipe changed while an ingredient was in hand.
    # Never risk contaminating the pot -- detour to deposit it on the wall
    # counter directly above any row-3 cell other than the pot column. If
    # the previous attempt at this same column already failed (the counter
    # there is occupied by an earlier deposit), move to a different column
    # instead of retrying the same blocked counter forever.
    at_pot_column = self_col == 5
    failed_here_before = memory['deposit_col'] == self_col
    move_to_new_column = jnp.where(self_col > 1, LEFT, RIGHT)
    fresh_attempt_action = jnp.where(facing == _DIR_UP, INTERACT, _move_for_dir(_DIR_UP))
    mismatch_action = jnp.where(
        self_row == 4, UP,
        jnp.where(at_pot_column, LEFT,
                  jnp.where(failed_here_before, move_to_new_column, fresh_attempt_action)),
    )
    mismatch_deposit_col_next = jnp.where(
        self_row == 4, memory['deposit_col'],
        jnp.where(at_pot_column, memory['deposit_col'],
                  jnp.where(failed_here_before, jnp.asarray(-1, jnp.int32),
                            jnp.where(facing == _DIR_UP, self_col, memory['deposit_col']))),
    ).astype(jnp.int32)

    mismatch = jnp.logical_not(holding_nothing) & (held_type != target)
    action = jnp.where(
        holding_nothing,
        action_fetch,
        jnp.where(mismatch, mismatch_action, action_to_pot),
    )
    action = jnp.asarray(action, jnp.int32)

    next_deposit_col = jnp.where(mismatch, mismatch_deposit_col_next, jnp.asarray(-1, jnp.int32))
    candidate_memory = {'target': target, 'deposit_col': next_deposit_col}
    next_memory = jax.tree_util.tree_map(
        lambda init_leaf, cand_leaf: jnp.where(reset, init_leaf, cand_leaf),
        init_agent_1(),
        candidate_memory,
    )
    return action, next_memory, {}
