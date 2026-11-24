"""Policy for Overcooked V2 test_time_simple.

Layout analysis (see context/environment/README.md + specification.json):
  WW2WWWWW
  W  WB  0
  R AWPA X
  W  WB  1
  WW2WWWWW
Column 3 is a solid wall in every row, so the map splits into two fully
disconnected rooms: agent_0 is confined to cols 0-2 (only reachable objects:
the recipe indicator R and a never-used ingredient-2 supply), agent_1 is
confined to cols 4-7 (pot, both plate piles, ingredient-0/1 supplies, and the
delivery station). Neither seat's 5x5 view radius ever reaches into the other
room, so there is no observable channel by which agent_0 could inform agent_1
of the sampled recipe, and no way for agent_0 to physically contribute to
cooking/delivery. Consequently:
  * agent_0 cannot affect team reward in this layout and simply stays put.
  * agent_1 must act "blind" with respect to the sampled recipe.

Blind exploit: a correct delivery resamples the recipe uniformly; an
incorrect delivery leaves it unchanged. Modeling belief p_t = P(recipe==A)
under any state-independent guess sequence g_t shows the following exact
recursion for the belief after playing guess q_t (1 for "guess A", 0 for
"guess B"):
    p_{t+1} = 0.5 * (1 + p_t - q_t)
with immediate expected reward 40*match_prob - 20, match_prob = p_t*q_t +
(1-p_t)*(1-q_t). Backward induction (the Bellman value is convex piecewise
linear in p, so bang-bang control is optimal) shows the optimal open-loop
policy is simply to alternate guesses every cycle: A, B, A, B, .... This
converges to a steady-state match probability of 2/3 (vs 1/2 for a constant
guess or i.i.d. random guessing), i.e. strictly positive expected reward per
cycle, achievable with zero information about the true recipe and zero
coordination with agent_0.

agent_1 therefore runs a fixed scripted cycle: gather 3 units of an
alternating ingredient type into the pot, fetch a plate, wait out the fixed
20-step cook timer, plate, deliver, repeat -- alternating the target
ingredient every completed cycle. The very first two steps of the episode
self-localize (using the pot, which is always visible from every reachable
agent_1 cell) to walk from the random spawn cell to a canonical reference
cell before the open-loop script begins.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# Static channel indices (see specification.json observation_channels).
CH_POT = 22

# ---------------------------------------------------------------------------
# agent_0: causally irrelevant in this layout (fully isolated room with no
# usable ingredient and no way to signal agent_1). Always stay put.
# ---------------------------------------------------------------------------


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del obs, rng
    action = jnp.asarray(STAY, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


# ---------------------------------------------------------------------------
# agent_1: scripted cook/deliver cycle with alternating recipe guess.
# ---------------------------------------------------------------------------


def _self_position_from_pot(obs):
    """Locate self (global row, col) using the always-visible pot landmark."""
    pot_slice = obs[:, :, CH_POT].reshape(-1)
    idx = jnp.argmax(pot_slice)
    i = idx // 5
    j = idx % 5
    self_row = 4 - i
    self_col = 6 - j
    return self_row, self_col


# Per-cycle scripted action lists (module scope must not contain calls, so
# these are built with only literal lists and the +/* operators).
# Each rep does the vertical move first (while still in column 5/6 so it
# never needs a wasted reorientation), then the horizontal move that lands
# already facing the right way for the following INTERACT.
# target 0: ingredient row 1 (v1=UP, v2=DOWN). target 1: row 3 (v1=DOWN, v2=UP).
# Each rep's leading vertical move works from either starting column (5 or
# 6); the RIGHT that follows either performs the col5->col6 move or, if
# already at col6, is a harmless blocked reorientation -- either way it ends
# at col6 facing right. So the exact same rep script also serves as the
# first rep of the next cycle right after a delivery (which leaves the
# player at col6), with no separate reposition step needed.
_REP_0 = [UP, RIGHT, INTERACT, DOWN, LEFT, INTERACT]
_REP_1 = [DOWN, RIGHT, INTERACT, UP, LEFT, INTERACT]
_TAIL = [UP, LEFT, INTERACT, DOWN, LEFT] + [INTERACT] * 15 + [RIGHT, INTERACT]
_SCRIPT_0_LIST = _REP_0 + _REP_0 + _REP_0 + _TAIL
_SCRIPT_1_LIST = _REP_1 + _REP_1 + _REP_1 + _TAIL
_CYCLE_LEN = 40


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    zero = jnp.asarray(0, dtype=jnp.int32)
    t = jnp.where(reset, zero, memory)

    script0 = jnp.asarray(_SCRIPT_0_LIST, dtype=jnp.int32)
    script1 = jnp.asarray(_SCRIPT_1_LIST, dtype=jnp.int32)

    self_row, self_col = _self_position_from_pot(obs)

    align_row_action = jnp.where(self_row == 1, DOWN, jnp.where(self_row == 3, UP, STAY))
    align_col_action = jnp.where(self_col == 6, LEFT, STAY)

    cycle_step = (t - 2) % _CYCLE_LEN
    cycle_index = (t - 2) // _CYCLE_LEN
    target = cycle_index % 2

    scripted_action = jnp.where(target == 0, script0[cycle_step], script1[cycle_step])

    action = jnp.where(
        t == 0, align_row_action,
        jnp.where(t == 1, align_col_action, scripted_action),
    )
    action = jnp.asarray(action, dtype=jnp.int32)

    next_memory = t + 1
    trace = {
        "t": jnp.asarray(t, dtype=jnp.int32),
        "self_row": jnp.asarray(self_row, dtype=jnp.int32),
        "self_col": jnp.asarray(self_col, dtype=jnp.int32),
        "cycle_step": jnp.asarray(cycle_step, dtype=jnp.int32),
        "target": jnp.asarray(target, dtype=jnp.int32),
    }
    return action, next_memory, trace
