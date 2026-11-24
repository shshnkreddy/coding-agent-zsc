"""Overcooked V2 (demo_cook_wide) cooperative policy.

Both seats are confined to small, disjoint, fully-enumerable floor regions
(agent_0: 3 cells in the upper corridor; agent_1: 13 cells in the lower
room). Only agent_1 can ever see the always-on recipe indicator; only
agent_0 can ever reach the plate supply and the delivery station. The
single shared pot is reachable from both sides. Given this geometry, the
only structurally available coordination signal for agent_0 is the shared,
observable pot content (plus its own reactive plate/deliver logic), while
agent_1's only useful contribution is reading the recipe indicator and
feeding matching ingredients into the pot. This is not a private
convention: it is the unique way either seat can act usefully at all, so
it should generalize to any competent independently generated partner.

Both seats identify their exact global cell by matching the observed
static-object crop (channels 22:32) against precomputed per-cell
signatures (literal constants below, derived offline from the fixed
layout), then use a precomputed shortest-path table to navigate.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# Facing ids (per README): UP=0, DOWN=1, RIGHT=2, LEFT=3
FACING_UP, FACING_DOWN, FACING_RIGHT, FACING_LEFT = 0, 1, 2, 3
# Action ids (per README / specification.json)
ACT_RIGHT, ACT_DOWN, ACT_LEFT, ACT_UP, ACT_STAY, ACT_INTERACT = 0, 1, 2, 3, 4, 5

# turn-in-place action needed to face a given direction (safe here: the
# neighboring cell in every use below is a non-floor object, so the
# movement attempt never actually relocates the agent, only its facing)
FACE_TO_ACTION_LIST = [ACT_UP, ACT_DOWN, ACT_RIGHT, ACT_LEFT]

# ---------------------------------------------------------------------------
# Agent_0 region: 3 collinear cells (1,4) (1,5) (1,6).
# Literal constants precomputed offline from the fixed demo_cook_wide layout.
# ---------------------------------------------------------------------------

A0_LEFT, A0_CENTER, A0_RIGHT = 0, 1, 2

AGENT0_SIG_LIST = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]

AGENT0_POT_LOCAL_LIST = [[3, 3], [3, 2], [3, 1]]
AGENT0_MOVE_LIST = [[4, 0, 0], [2, 4, 0], [2, 2, 4]]

# ---------------------------------------------------------------------------
# Agent_1 region: 13 cells (row 3 cols 1..9, plus (4,1) (4,2) (4,8) (4,9)),
# indices 0..8 = (3,1)..(3,9); 9=(4,1); 10=(4,2); 11=(4,8); 12=(4,9).
# ---------------------------------------------------------------------------

POT_IDX = 4
SUPPLY_IDX = 9
# Two alternate dump targets, each an interior wall cell reachable by facing
# DOWN from a row-3 cell: (3,3)->wall(4,3), and (3,7)->wall(4,7). A single
# shared dump slot can be left permanently occupied by a stale leftover of
# the *other* ingredient type (each wall counter holds only one item), which
# would otherwise deadlock all future mismatch handling; two slots plus
# failure-triggered alternation avoid that.
DUMP_A_IDX = 2
DUMP_B_IDX = 6

AGENT1_SIG_LIST = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]

AGENT1_NEXT_ACTION_LIST = [[4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [2, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [2, 2, 4, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0], [2, 2, 2, 4, 0, 0, 0, 0, 0, 2, 2, 0, 0], [2, 2, 2, 2, 4, 0, 0, 0, 0, 2, 2, 0, 0], [2, 2, 2, 2, 2, 4, 0, 0, 0, 2, 2, 0, 0], [2, 2, 2, 2, 2, 2, 4, 0, 0, 2, 2, 0, 0], [2, 2, 2, 2, 2, 2, 2, 4, 0, 2, 2, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4]]

# fixed local offsets, valid only when the agent is actually at POT_IDX == (3,5)
POT_LOCAL_AT_POTIDX = (1, 2)
RECIPE_LOCAL_AT_POTIDX = (3, 2)
SELF_LOCAL = (2, 2)
# offsets of the wall below each dump cell, as seen from that dump cell
DUMP_WALL_LOCAL = (3, 2)  # wall (4,3) as seen from (3,3)


# ---------------------------------------------------------------------------
# Shared decoding helpers (all JAX calls here run inside act(), never at
# module scope)
# ---------------------------------------------------------------------------

def _identify_cell(obs, sig_table):
    obs_static = obs[:, :, 22:32]
    diffs = jnp.sum(jnp.abs(obs_static[None, :, :, :] - sig_table), axis=(1, 2, 3))
    return jnp.argmin(diffs).astype(jnp.int32)


def _self_facing(obs):
    r, c = SELF_LOCAL
    return jnp.argmax(obs[r, c, 1:5]).astype(jnp.int32)


def _self_inventory(obs):
    r, c = SELF_LOCAL
    return obs[r, c, 5], obs[r, c, 6], obs[r, c, 7], obs[r, c, 8], obs[r, c, 9], obs[r, c, 10]


# ---------------------------------------------------------------------------
# Agent_0: fully reactive (the pot is visible from every reachable cell)
# ---------------------------------------------------------------------------

def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    face_to_action = jnp.asarray(FACE_TO_ACTION_LIST, dtype=jnp.int32)
    a0_sig = jnp.asarray(AGENT0_SIG_LIST, dtype=jnp.int32)
    a0_pot_local = jnp.asarray(AGENT0_POT_LOCAL_LIST, dtype=jnp.int32)
    a0_move = jnp.asarray(AGENT0_MOVE_LIST, dtype=jnp.int32)

    cell_idx = _identify_cell(obs, a0_sig)
    facing = _self_facing(obs)
    plate, cooked, ing0, ing1, _ing2, _ing3 = _self_inventory(obs)

    pot_local = a0_pot_local[cell_idx]
    pot_row, pot_col = pot_local[0], pot_local[1]
    pot_ing0 = obs[pot_row, pot_col, 34]
    pot_ing1 = obs[pot_row, pot_col, 35]
    pot_cooked = obs[pot_row, pot_col, 33]
    pot_timer = obs[pot_row, pot_col, 44]

    pot_ready = pot_cooked > 0
    pot_cooking = pot_timer > 0
    pot_busy = pot_ready | pot_cooking

    room_type0 = (pot_ing0 < 3) & (pot_ing1 == 0) & (~pot_busy)
    room_type1 = (pot_ing1 < 3) & (pot_ing0 == 0) & (~pot_busy)

    holding_dish = cooked > 0
    holding_plate = (plate > 0) & (cooked == 0)
    holding_ing0 = (plate == 0) & (cooked == 0) & (ing0 > 0)
    holding_ing1 = (plate == 0) & (cooked == 0) & (ing1 > 0)

    partial0 = (pot_ing0 > 0) & (pot_ing0 < 3) & (pot_ing1 == 0) & (~pot_busy)
    partial1 = (pot_ing1 > 0) & (pot_ing1 < 3) & (pot_ing0 == 0) & (~pot_busy)

    empty_target = jnp.where(
        pot_ready, A0_LEFT,
        jnp.where(
            pot_cooking, A0_LEFT,
            jnp.where(partial0, A0_LEFT, jnp.where(partial1, A0_RIGHT, A0_CENTER)),
        ),
    )
    empty_facing = jnp.where(
        pot_ready, FACING_UP,
        jnp.where(
            pot_cooking, FACING_UP,
            jnp.where(partial0, FACING_LEFT, jnp.where(partial1, FACING_RIGHT, facing)),
        ),
    )
    empty_interact = pot_ready | pot_cooking | partial0 | partial1

    target_idx = jnp.where(
        holding_dish, A0_CENTER,
        jnp.where(
            holding_plate, A0_CENTER,
            jnp.where(holding_ing0, A0_CENTER, jnp.where(holding_ing1, A0_CENTER, empty_target)),
        ),
    )
    target_facing = jnp.where(
        holding_dish, FACING_UP,
        jnp.where(
            holding_plate, FACING_DOWN,
            jnp.where(holding_ing0, FACING_DOWN, jnp.where(holding_ing1, FACING_DOWN, empty_facing)),
        ),
    )
    should_interact = jnp.where(
        holding_dish, True,
        jnp.where(
            holding_plate, pot_ready,
            jnp.where(holding_ing0, room_type0, jnp.where(holding_ing1, room_type1, empty_interact)),
        ),
    )

    at_target = cell_idx == target_idx
    need_turn = at_target & (facing != target_facing)
    move_action = a0_move[cell_idx, target_idx]
    turn_action = face_to_action[target_facing]
    action = jnp.where(
        ~at_target, move_action,
        jnp.where(need_turn, turn_action, jnp.where(should_interact, ACT_INTERACT, ACT_STAY)),
    ).astype(jnp.int32)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    trace = {'cell_idx': cell_idx, 'action': action}
    return action, next_memory, trace


# ---------------------------------------------------------------------------
# Agent_1: reads the recipe indicator near the pot, fetches matching
# ingredients, and feeds the pot; dumps mismatched leftovers on a counter.
# ---------------------------------------------------------------------------

RECIPE_DEBOUNCE_STEPS = 3  # steps of a stable reading required before trusting an empty pot's recipe
PARTIAL_STALL_STEPS = 10   # steps a partial pot must sit unchanged before agent_1 self-helps


def init_agent_1():
    return {
        'state': jnp.asarray(0, dtype=jnp.int32),          # 0=seek pot, 1=seek supply
        'target_type': jnp.asarray(-1, dtype=jnp.int32),   # -1 unknown, else 0/1
        'dump_alt': jnp.asarray(0, dtype=jnp.int32),       # which dump slot to try next
        'dump_pending': jnp.asarray(0, dtype=jnp.int32),   # 1 right after an attempted drop
        'empty_wait': jnp.asarray(0, dtype=jnp.int32),     # consecutive stable empty-pot recipe reads
        'partial_wait': jnp.asarray(0, dtype=jnp.int32),   # consecutive steps a partial pot sat unchanged
        'last_pot_count': jnp.asarray(0, dtype=jnp.int32),  # last-seen total ingredient count at the pot
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    face_to_action = jnp.asarray(FACE_TO_ACTION_LIST, dtype=jnp.int32)
    a1_sig = jnp.asarray(AGENT1_SIG_LIST, dtype=jnp.int32)
    a1_next = jnp.asarray(AGENT1_NEXT_ACTION_LIST, dtype=jnp.int32)

    init_mem = init_agent_1()
    memory = jax.tree_util.tree_map(lambda i, m: jnp.where(reset, i, m), init_mem, memory)

    cell_idx = _identify_cell(obs, a1_sig)
    facing = _self_facing(obs)
    plate, cooked, ing0, ing1, ing2, ing3 = _self_inventory(obs)

    at_pot = cell_idx == POT_IDX
    rr, rc = RECIPE_LOCAL_AT_POTIDX
    recipe_ing0 = obs[rr, rc, 40]
    recipe_ing1 = obs[rr, rc, 41]
    observed_target = jnp.where(recipe_ing0 > 0, 0, jnp.where(recipe_ing1 > 0, 1, -1))
    target_type = jnp.where(at_pot & (observed_target >= 0), observed_target, memory['target_type'])

    pr, pc = POT_LOCAL_AT_POTIDX
    pot_ing0 = obs[pr, pc, 34]
    pot_ing1 = obs[pr, pc, 35]
    pot_cooked = obs[pr, pc, 33]
    pot_timer = obs[pr, pc, 44]
    pot_busy = (pot_cooked > 0) | (pot_timer > 0)

    # A pot reading ingredient counts of 0 does not mean the *displayed
    # recipe* is fresh: the environment scoops a cooked pot (clearing its
    # ingredient counts) one or more steps *before* the later delivery
    # interaction that actually resamples the recipe. Agent_1 typically sits
    # parked at the pot, so it often observes "pot empty" together with the
    # stale, about-to-change recipe display in that narrow window. Fetching
    # immediately on that stale read causes a coin-flip chance of grabbing
    # the wrong ingredient type once the recipe resamples mid-transit.
    # Require several consecutive steps of a stable reading before trusting
    # it enough to commit to a fetch.
    pot_fully_empty = (pot_ing0 == 0) & (pot_ing1 == 0) & (~pot_busy)
    same_target_as_before = target_type == memory['target_type']
    empty_wait = jnp.where(
        at_pot & pot_fully_empty,
        jnp.where(same_target_as_before,
                  jnp.minimum(memory['empty_wait'] + 1, RECIPE_DEBOUNCE_STEPS),
                  jnp.asarray(1, dtype=jnp.int32)),
        jnp.asarray(0, dtype=jnp.int32),
    ).astype(jnp.int32)
    recipe_confirmed = empty_wait >= RECIPE_DEBOUNCE_STEPS

    # agent_0 (this same policy bundle, in self-play) tops off a partially
    # filled pot itself very quickly using its own adjacent supply access, so
    # agent_1 must not race it there every cycle (that redundant race is what
    # produced chronic surplus/mismatch dumps, see need_fetch below). But an
    # independently generated partner agent_0 may not offer that help at all,
    # so agent_1 must remain able to finish the recipe alone: if a partial
    # pot sits at the same ingredient count for many consecutive steps while
    # agent_1 is parked there, nobody else is completing it and agent_1
    # should step in itself.
    pot_count = (pot_ing0 + pot_ing1).astype(jnp.int32)
    pot_is_partial = (pot_count > 0) & (pot_count < 3) & (~pot_busy)
    same_pot_count = pot_is_partial & (pot_count == memory['last_pot_count'])
    partial_wait = jnp.where(
        at_pot & pot_is_partial,
        jnp.where(same_pot_count,
                  jnp.minimum(memory['partial_wait'] + 1, PARTIAL_STALL_STEPS),
                  jnp.asarray(1, dtype=jnp.int32)),
        jnp.asarray(0, dtype=jnp.int32),
    ).astype(jnp.int32)
    partial_stalled = partial_wait >= PARTIAL_STALL_STEPS

    holding_ing0 = (plate == 0) & (cooked == 0) & (ing0 > 0)
    holding_ing1 = (plate == 0) & (cooked == 0) & (ing1 > 0)
    holding_other = (plate == 0) & (cooked == 0) & ((ing2 > 0) | (ing3 > 0))
    holding_any_ing = holding_ing0 | holding_ing1 | holding_other
    holding_empty = ~(holding_any_ing | (plate > 0) | (cooked > 0))

    is_match0 = holding_ing0 & (target_type == 0)
    is_match1 = holding_ing1 & (target_type == 1)
    holding_match = is_match0 | is_match1
    holding_mismatch = holding_any_ing & (~holding_match) & (target_type >= 0)
    holding_mismatch = holding_mismatch | (holding_other & (target_type < 0))

    # Primarily fetch only a *seed* ingredient for a fully empty pot: a
    # same-policy partner agent_0 has its own much faster local supply access
    # and independently tops off any partially filled pot via its own
    # partial0/partial1 logic, so agent_1 racing it there every cycle just
    # loses, returns holding a now-redundant/mismatched unit, and is forced
    # to dump it on one of only two wall-counter slots; repeated redundant
    # dumps can saturate both slots with stale off-type leftovers and block
    # all future dumps forever. But an independently generated partner
    # agent_0 may offer no such help at all, so agent_1 must still be able to
    # finish the recipe alone: `partial_stalled` (many consecutive steps of
    # an unchanging partial fill) is the signal that nobody else is helping,
    # and only then does agent_1 step in for the remaining units itself.
    need_fetch0 = (target_type == 0) & (
        (pot_fully_empty & recipe_confirmed)
        | ((pot_ing0 > 0) & (pot_ing0 < 3) & (pot_ing1 == 0) & (~pot_busy) & partial_stalled)
    )
    need_fetch1 = (target_type == 1) & (
        (pot_fully_empty & recipe_confirmed)
        | ((pot_ing1 > 0) & (pot_ing1 < 3) & (pot_ing0 == 0) & (~pot_busy) & partial_stalled)
    )
    need_fetch = need_fetch0 | need_fetch1

    can_insert0 = is_match0 & (pot_ing0 < 3) & (pot_ing1 == 0) & (~pot_busy)
    can_insert1 = is_match1 & (pot_ing1 < 3) & (pot_ing0 == 0) & (~pot_busy)
    can_insert = can_insert0 | can_insert1

    MODE_TOPOT, MODE_TOSUPPLY, MODE_RETURN, MODE_DUMP = 0, 1, 2, 3

    want_supply = holding_empty & ((memory['state'] == 1) | (at_pot & need_fetch))
    action_mode = jnp.where(
        holding_mismatch, MODE_DUMP,
        jnp.where(holding_match, MODE_RETURN, jnp.where(want_supply, MODE_TOSUPPLY, MODE_TOPOT)),
    )
    next_state = jnp.where(action_mode == MODE_TOSUPPLY, 1, 0).astype(jnp.int32)

    move_to_pot = jnp.where(at_pot, ACT_STAY, a1_next[cell_idx, POT_IDX])

    at_supply = cell_idx == SUPPLY_IDX
    desired_supply_facing = jnp.where(target_type == 0, FACING_LEFT, FACING_DOWN)
    supply_ready_action = jnp.where(
        facing != desired_supply_facing, face_to_action[desired_supply_facing], ACT_INTERACT
    )
    move_to_supply = jnp.where(at_supply, supply_ready_action, a1_next[cell_idx, SUPPLY_IDX])

    return_ready_action = jnp.where(
        facing != FACING_UP, face_to_action[FACING_UP], jnp.where(can_insert, ACT_INTERACT, ACT_STAY)
    )
    move_return = jnp.where(at_pot, return_ready_action, a1_next[cell_idx, POT_IDX])

    # --- dump handling with two alternating slots ---
    # If the previous step attempted a drop at the slot we were then using
    # and we are *still* holding a mismatched ingredient now, that attempt
    # must have failed (the counter was already occupied); switch slots.
    prev_slot_idx = jnp.where(memory['dump_alt'] == 0, DUMP_A_IDX, DUMP_B_IDX)
    attempt_failed = (memory['dump_pending'] > 0) & holding_mismatch & (cell_idx == prev_slot_idx)
    dump_alt = jnp.where(attempt_failed, 1 - memory['dump_alt'], memory['dump_alt']).astype(jnp.int32)
    dump_slot_idx = jnp.where(dump_alt == 0, DUMP_A_IDX, DUMP_B_IDX)

    at_dump = cell_idx == dump_slot_idx
    dump_ready_action = jnp.where(facing != FACING_DOWN, face_to_action[FACING_DOWN], ACT_INTERACT)
    move_dump = jnp.where(at_dump, dump_ready_action, a1_next[cell_idx, dump_slot_idx])
    issuing_dump_interact = holding_mismatch & at_dump & (facing == FACING_DOWN)
    next_dump_pending = jnp.where(action_mode == MODE_DUMP, issuing_dump_interact.astype(jnp.int32), 0)

    base_action = jnp.select(
        [action_mode == MODE_TOPOT, action_mode == MODE_TOSUPPLY,
         action_mode == MODE_RETURN, action_mode == MODE_DUMP],
        [move_to_pot, move_to_supply, move_return, move_dump],
        default=ACT_STAY,
    )

    # General opportunistic reclaim: whenever empty-handed and standing at
    # either dump slot with a leftover matching the current target, grab it
    # (this both fetches for free and clears the slot for future dumps),
    # regardless of which mode the rest of the logic chose.
    dwr, dwc = DUMP_WALL_LOCAL
    dump_wall_ing0 = obs[dwr, dwc, 34]
    dump_wall_ing1 = obs[dwr, dwc, 35]
    reclaim_match = ((target_type == 0) & (dump_wall_ing0 > 0)) | ((target_type == 1) & (dump_wall_ing1 > 0))
    at_a_dump_slot = (cell_idx == DUMP_A_IDX) | (cell_idx == DUMP_B_IDX)
    reclaim_ready_action = jnp.where(facing != FACING_DOWN, face_to_action[FACING_DOWN], ACT_INTERACT)
    can_reclaim_here = holding_empty & at_a_dump_slot & reclaim_match

    action = jnp.where(can_reclaim_here, reclaim_ready_action, base_action).astype(jnp.int32)

    next_memory = {
        'state': next_state,
        'target_type': target_type.astype(jnp.int32),
        'dump_alt': dump_alt,
        'dump_pending': next_dump_pending,
        'empty_wait': empty_wait,
        'partial_wait': partial_wait,
        'last_pot_count': pot_count,
    }
    trace = {
        'cell_idx': cell_idx, 'action_mode': action_mode, 'target_type': target_type,
        'pot_ing0': pot_ing0, 'pot_ing1': pot_ing1, 'pot_busy': pot_busy.astype(jnp.int32),
        'holding_ing0': holding_ing0.astype(jnp.int32), 'holding_ing1': holding_ing1.astype(jnp.int32),
        'holding_match': holding_match.astype(jnp.int32), 'holding_mismatch': holding_mismatch.astype(jnp.int32),
        'dump_alt': memory['dump_alt'], 'dump_pending': memory['dump_pending'], 'empty_wait': empty_wait,
        'partial_wait': partial_wait,
    }
    return action, next_memory, trace
