"""Overcooked V2 (test_time_wide) policy.

Layout facts (fixed, from context/environment/README.md + specification.json):
  Board 7x6. agent_0 room = rows{1,2} x cols{1..4}, fully open rectangle.
  agent_1 room = rows{4,5} x cols{1..4}, fully open rectangle, physically
  disconnected from agent_0's room (no path exists between them).

  agent_0's room gives access to: ingredient-0 supply (1,0)/(1,5),
  ingredient-1 supply (2,0)/(2,5), both pots (3,1)/(3,3) [north side],
  plate supply (0,3), delivery (0,2). agent_0 can solo the entire
  cook-and-deliver cycle.

  agent_1's room gives access to: ingredient-3 supply (unused by any
  recipe), both pots (3,1)/(3,3) [south side, only useful for the unused
  ingredient 3], and it is the ONLY seat that can ever see the always-on
  recipe indicator R at (6,2) (agent_0's room is too far away to ever see
  R). agent_1 cannot reach a plate or the delivery station, so it cannot
  causally contribute to any delivery in this layout.

  Because agent_0 never observes the recipe and cross-play partners are
  independently generated (no shared private convention can be assumed),
  agent_0 hedges: it dedicates pot-left (3,1) to recipe [0,0,0] and
  pot-right (3,3) to recipe [1,1,1], cooks both continuously, and
  delivers every finished dish regardless of which recipe is currently
  requested. Since the two dishes cover both possible recipes, this
  guarantees frequent correct deliveries and can never get stuck
  permanently on a wrong recipe (unlike committing to a single guess).

  agent_1 cannot help cook/plate/deliver, so it never interacts with any
  object (zero risk of contaminating a pot). It self-localizes with the
  always-visible recipe indicator and drifts to the column mirroring the
  current recipe at row 4 (the only row from which agent_0 could ever
  possibly observe it, since the two rooms are exactly 2 rows apart),
  facing up. This is a low-risk, non-interactive, purely positional
  signal that costs nothing if unread by an independently generated
  partner and is fully compatible with agent_0's self-sufficient plan.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

# ---- shared helpers -------------------------------------------------------

# Action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5


def _find_channel_local(obs, channel):
    """Locate the (row, col) in the 5x5 view where `channel` is nonzero."""
    grid = obs[:, :, channel]
    flat = grid.reshape(-1)
    idx = jnp.argmax(flat)
    found = flat[idx] > 0
    di = (idx // 5).astype(jnp.int32)
    dj = (idx % 5).astype(jnp.int32)
    return di, dj, found


def _local_of(global_row, global_col, self_row, self_col):
    li = global_row - self_row + 2
    lj = global_col - self_col + 2
    li = jnp.clip(li, 0, 4).astype(jnp.int32)
    lj = jnp.clip(lj, 0, 4).astype(jnp.int32)
    return li, lj


def _navigate_action(r, c, tr, tc):
    return jnp.where(
        r < tr, DOWN,
        jnp.where(
            r > tr, UP,
            jnp.where(
                c < tc, RIGHT,
                jnp.where(c > tc, LEFT, STAY),
            ),
        ),
    )


# ---- agent_0: solo cook-and-deliver hedge ----------------------------------

# Global (row, col) of key objects in agent_0's room.
_POT_LEFT = (3, 1)
_POT_RIGHT = (3, 3)

# Goal codes.
_G_WAIT, _G_FETCH0, _G_FETCH1, _G_ADD0, _G_ADD1, _G_PLATE, _G_SCOOP_L, _G_SCOOP_R, _G_DELIVER = (
    0, 1, 2, 3, 4, 5, 6, 7, 8,
)

_HOME_ROW_LIST = [1, 1, 2, 2, 2, 1, 2, 2, 1]
_HOME_COL_LIST = [1, 1, 1, 1, 3, 3, 1, 3, 2]
_FACE_ACT_LIST = [STAY, LEFT, LEFT, DOWN, DOWN, UP, DOWN, DOWN, UP]


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)

    di, dj, found = _find_channel_local(obs, 23)  # goal / delivery channel
    self_row = jnp.where(found, 2 - di, 1)
    self_col = jnp.where(found, 4 - dj, 2)

    pl_i, pl_j = _local_of(_POT_LEFT[0], _POT_LEFT[1], self_row, self_col)
    pr_i, pr_j = _local_of(_POT_RIGHT[0], _POT_RIGHT[1], self_row, self_col)

    pl_cooked = obs[pl_i, pl_j, 33] > 0
    pl_ing0 = obs[pl_i, pl_j, 34]
    pl_ing1 = obs[pl_i, pl_j, 35]
    pl_timer = obs[pl_i, pl_j, 44]

    pl_ing3 = obs[pl_i, pl_j, 37]

    pr_cooked = obs[pr_i, pr_j, 33] > 0
    pr_ing0 = obs[pr_i, pr_j, 34]
    pr_ing1 = obs[pr_i, pr_j, 35]
    pr_timer = obs[pr_i, pr_j, 44]
    pr_ing3 = obs[pr_i, pr_j, 37]

    # agent_1 can only ever reach ingredient-3 supply, so ingredient-3 is the
    # only contamination risk a partner could introduce into a shared pot.
    # Guard fill/scoop decisions against it so we never cook or deliver a
    # mixed (wrong) dish even if a partner interferes with a pot.
    pl_idle_need0 = (~pl_cooked) & (pl_timer == 0) & (pl_ing1 == 0) & (pl_ing3 == 0) & (pl_ing0 < 3)
    pr_idle_need1 = (~pr_cooked) & (pr_timer == 0) & (pr_ing0 == 0) & (pr_ing3 == 0) & (pr_ing1 < 3)
    pl_safe_ready = pl_cooked & (pl_ing1 == 0) & (pl_ing3 == 0) & (pl_ing0 == 3)
    pr_safe_ready = pr_cooked & (pr_ing0 == 0) & (pr_ing3 == 0) & (pr_ing1 == 3)
    any_ready = pl_safe_ready | pr_safe_ready

    plate_flag = obs[2, 2, 5]
    cooked_flag = obs[2, 2, 6]
    ing0 = obs[2, 2, 7]
    ing1 = obs[2, 2, 8]
    ing2 = obs[2, 2, 9]
    ing3 = obs[2, 2, 10]

    raw_hands = (plate_flag == 0) & (cooked_flag == 0)
    holding_ing0 = (ing0 > 0) & raw_hands
    holding_ing1 = (ing1 > 0) & raw_hands
    holding_plate_empty = (plate_flag > 0) & (cooked_flag == 0)
    holding_dish = (plate_flag > 0) & (cooked_flag > 0)

    # Collect a ready dish before starting a fresh fill, so a finished dish
    # does not sit uncollected near the end of the episode.
    goal_empty = jnp.where(
        any_ready, _G_PLATE,
        jnp.where(
            pl_idle_need0, _G_FETCH0,
            jnp.where(pr_idle_need1, _G_FETCH1, _G_WAIT),
        ),
    )
    goal_plate = jnp.where(pl_safe_ready, _G_SCOOP_L, _G_SCOOP_R)

    other_held = (ing2 > 0) | (ing3 > 0)
    goal = jnp.where(
        holding_ing0, _G_ADD0,
        jnp.where(
            holding_ing1, _G_ADD1,
            jnp.where(
                holding_plate_empty, goal_plate,
                jnp.where(
                    holding_dish, _G_DELIVER,
                    jnp.where(other_held, _G_WAIT, goal_empty),
                ),
            ),
        ),
    )

    home_row = jnp.array(_HOME_ROW_LIST, dtype=jnp.int32)[goal]
    home_col = jnp.array(_HOME_COL_LIST, dtype=jnp.int32)[goal]
    face_act = jnp.array(_FACE_ACT_LIST, dtype=jnp.int32)[goal]

    at_home = (self_row == home_row) & (self_col == home_col)

    facing_up = obs[2, 2, 1] > 0
    facing_down = obs[2, 2, 2] > 0
    facing_right = obs[2, 2, 3] > 0
    cur_face_act = jnp.where(facing_up, UP, jnp.where(facing_down, DOWN, jnp.where(facing_right, RIGHT, LEFT)))

    nav_act = _navigate_action(self_row, self_col, home_row, home_col)
    act_at_home = jnp.where(cur_face_act == face_act, INTERACT, face_act)

    action = jnp.where(goal == _G_WAIT, STAY, jnp.where(~at_home, nav_act, act_at_home))
    action = action.astype(jnp.int32)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    trace = {
        'goal': goal.astype(jnp.int32),
        'self_row': self_row.astype(jnp.int32),
        'self_col': self_col.astype(jnp.int32),
    }
    return action, next_memory, trace


# ---- agent_1: recipe-observing, non-interactive, position-signalling ------


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)

    di, dj, found = _find_channel_local(obs, 25)  # recipe indicator channel
    self_row = jnp.where(found, 8 - di, 4)
    self_col = jnp.where(found, 4 - dj, 2)

    recipe_ing0 = obs[di, dj, 40]
    recipe_ing1 = obs[di, dj, 41]

    target_col = jnp.where(
        found & (recipe_ing0 > 0), 1,
        jnp.where(found & (recipe_ing1 > 0), 3, 2),
    ).astype(jnp.int32)
    target_row = jnp.asarray(4, dtype=jnp.int32)

    action = jnp.where(
        self_row != target_row, UP,
        jnp.where(self_col < target_col, RIGHT, jnp.where(self_col > target_col, LEFT, UP)),
    )
    action = action.astype(jnp.int32)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    return action, next_memory, {}
