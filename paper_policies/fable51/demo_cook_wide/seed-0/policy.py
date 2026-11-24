"""Overcooked V2 demo_cook_wide heuristic policy (room split).

agent_0 (top room, row 1 cols 4..6) plates and delivers; it never guesses the
recipe and only adds an ingredient matching what the pot already holds (one unit
when the pot idles at count 1, or a stall-breaker unit after a long idle).
agent_1 (bottom room) reads the always-on recipe indicator, fetches the recipe
ingredient from the side supplies, fills the pot from (3,5), pre-fetches during
cooking, and drops mismatched ingredients on counters (later used as a cache).

Navigation tables below are literals generated offline (gen_tables.py) from the
fixed layout; runtime code uses only fixed-shape jnp ops.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'room_split_heuristic', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

A0_CELLS = [[1, 4], [1, 5], [1, 6]]
A0_GOAL = [4, 5, 0, 8, 3, 10, 1, 9]
A0_DIST = [[2, 2, 0, 3, 1, 2, 1, 3], [2, 2, 1, 3, 1, 2, 0, 3], [2, 2, 1, 3, 1, 2, 1, 3], [2, 2, 1, 3, 0, 2, 1, 3], [0, 1, 2, 2, 1, 1, 2, 2], [1, 0, 2, 2, 1, 1, 2, 2], [1, 1, 2, 2, 1, 1, 2, 2], [1, 1, 2, 2, 1, 1, 2, 2], [2, 2, 3, 0, 2, 1, 3, 1], [2, 2, 3, 1, 2, 1, 3, 0], [2, 2, 3, 1, 2, 0, 3, 1], [2, 2, 3, 1, 2, 1, 3, 1]]
A0_NEXT = [[0, 0, 4, 0, 2, 0, 1, 0], [0, 0, 3, 0, 2, 0, 4, 0], [0, 0, 3, 0, 2, 0, 1, 0], [0, 0, 3, 0, 4, 0, 1, 0], [4, 1, 2, 0, 2, 0, 2, 0], [3, 4, 2, 0, 2, 0, 2, 0], [3, 1, 2, 0, 2, 0, 2, 0], [3, 1, 2, 0, 2, 0, 2, 0], [2, 2, 2, 4, 2, 0, 2, 1], [2, 2, 2, 3, 2, 0, 2, 4], [2, 2, 2, 3, 2, 4, 2, 1], [2, 2, 2, 3, 2, 0, 2, 1]]
A0_TEMPLATES = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
A1_CELLS = [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [4, 1], [4, 2], [4, 8], [4, 9]]
A1_GOAL = [16, 39, 50, 37, 49, 13, 21, 8, 24, 9, 25, 4, 28, 0, 32, 3, 34, 41, 45, 12, 20]
A1_DIST = [[5, 2, 9, 1, 9, 4, 6, 3, 7, 3, 7, 2, 8, 0, 9, 1, 8, 2, 8, 4, 6], [5, 2, 9, 1, 9, 4, 6, 3, 7, 3, 7, 2, 8, 1, 9, 1, 8, 2, 8, 4, 6], [5, 2, 9, 1, 9, 4, 6, 3, 7, 3, 7, 2, 8, 1, 9, 1, 8, 2, 8, 4, 6], [5, 2, 9, 1, 9, 4, 6, 3, 7, 3, 7, 2, 8, 1, 9, 0, 8, 2, 8, 4, 6], [4, 2, 8, 2, 8, 3, 5, 2, 6, 2, 6, 0, 7, 2, 8, 1, 7, 1, 7, 3, 5], [4, 2, 8, 2, 8, 3, 5, 2, 6, 2, 6, 1, 7, 2, 8, 1, 7, 1, 7, 3, 5], [4, 2, 8, 2, 8, 3, 5, 2, 6, 2, 6, 1, 7, 2, 8, 1, 7, 1, 7, 3, 5], [4, 2, 8, 2, 8, 3, 5, 2, 6, 2, 6, 1, 7, 2, 8, 1, 7, 1, 7, 3, 5], [3, 3, 7, 3, 7, 2, 4, 0, 5, 1, 5, 2, 6, 3, 7, 2, 6, 2, 6, 2, 4], [3, 3, 7, 3, 7, 2, 4, 1, 5, 0, 5, 2, 6, 3, 7, 2, 6, 2, 6, 2, 4], [3, 3, 7, 3, 7, 2, 4, 1, 5, 1, 5, 2, 6, 3, 7, 2, 6, 2, 6, 2, 4], [3, 3, 7, 3, 7, 2, 4, 1, 5, 1, 5, 2, 6, 3, 7, 2, 6, 2, 6, 2, 4], [2, 4, 6, 4, 6, 1, 3, 2, 4, 2, 4, 3, 5, 4, 6, 3, 5, 3, 5, 0, 3], [2, 4, 6, 4, 6, 0, 3, 2, 4, 2, 4, 3, 5, 4, 6, 3, 5, 3, 5, 1, 3], [2, 4, 6, 4, 6, 1, 3, 2, 4, 2, 4, 3, 5, 4, 6, 3, 5, 3, 5, 1, 3], [2, 4, 6, 4, 6, 1, 3, 2, 4, 2, 4, 3, 5, 4, 6, 3, 5, 3, 5, 1, 3], [0, 5, 5, 5, 5, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 4, 4, 4, 4, 2, 2], [1, 5, 5, 5, 5, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 4, 4, 4, 4, 2, 2], [1, 5, 5, 5, 5, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 4, 4, 4, 4, 2, 2], [1, 5, 5, 5, 5, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 4, 4, 4, 4, 2, 2], [2, 6, 4, 6, 4, 3, 1, 4, 2, 4, 2, 5, 3, 6, 4, 5, 3, 5, 3, 3, 0], [2, 6, 4, 6, 4, 3, 0, 4, 2, 4, 2, 5, 3, 6, 4, 5, 3, 5, 3, 3, 1], [2, 6, 4, 6, 4, 3, 1, 4, 2, 4, 2, 5, 3, 6, 4, 5, 3, 5, 3, 3, 1], [2, 6, 4, 6, 4, 3, 1, 4, 2, 4, 2, 5, 3, 6, 4, 5, 3, 5, 3, 3, 1], [3, 7, 3, 7, 3, 4, 2, 5, 0, 5, 1, 6, 2, 7, 3, 6, 2, 6, 2, 4, 2], [3, 7, 3, 7, 3, 4, 2, 5, 1, 5, 0, 6, 2, 7, 3, 6, 2, 6, 2, 4, 2], [3, 7, 3, 7, 3, 4, 2, 5, 1, 5, 1, 6, 2, 7, 3, 6, 2, 6, 2, 4, 2], [3, 7, 3, 7, 3, 4, 2, 5, 1, 5, 1, 6, 2, 7, 3, 6, 2, 6, 2, 4, 2], [4, 8, 2, 8, 2, 5, 3, 6, 2, 6, 2, 7, 0, 8, 2, 7, 1, 7, 1, 5, 3], [4, 8, 2, 8, 2, 5, 3, 6, 2, 6, 2, 7, 1, 8, 2, 7, 1, 7, 1, 5, 3], [4, 8, 2, 8, 2, 5, 3, 6, 2, 6, 2, 7, 1, 8, 2, 7, 1, 7, 1, 5, 3], [4, 8, 2, 8, 2, 5, 3, 6, 2, 6, 2, 7, 1, 8, 2, 7, 1, 7, 1, 5, 3], [5, 9, 2, 9, 1, 6, 4, 7, 3, 7, 3, 8, 2, 9, 0, 8, 1, 8, 2, 6, 4], [5, 9, 2, 9, 1, 6, 4, 7, 3, 7, 3, 8, 2, 9, 1, 8, 1, 8, 2, 6, 4], [5, 9, 2, 9, 1, 6, 4, 7, 3, 7, 3, 8, 2, 9, 1, 8, 0, 8, 2, 6, 4], [5, 9, 2, 9, 1, 6, 4, 7, 3, 7, 3, 8, 2, 9, 1, 8, 1, 8, 2, 6, 4], [6, 1, 10, 1, 10, 5, 7, 4, 8, 4, 8, 2, 9, 1, 10, 2, 9, 2, 9, 5, 7], [6, 1, 10, 0, 10, 5, 7, 4, 8, 4, 8, 2, 9, 1, 10, 2, 9, 2, 9, 5, 7], [6, 1, 10, 1, 10, 5, 7, 4, 8, 4, 8, 2, 9, 1, 10, 2, 9, 2, 9, 5, 7], [6, 0, 10, 1, 10, 5, 7, 4, 8, 4, 8, 2, 9, 1, 10, 2, 9, 2, 9, 5, 7], [5, 1, 9, 2, 9, 4, 6, 3, 7, 3, 7, 1, 8, 2, 9, 2, 8, 1, 8, 4, 6], [5, 1, 9, 2, 9, 4, 6, 3, 7, 3, 7, 1, 8, 2, 9, 2, 8, 0, 8, 4, 6], [5, 1, 9, 2, 9, 4, 6, 3, 7, 3, 7, 1, 8, 2, 9, 2, 8, 1, 8, 4, 6], [5, 1, 9, 2, 9, 4, 6, 3, 7, 3, 7, 1, 8, 2, 9, 2, 8, 1, 8, 4, 6], [5, 9, 1, 9, 2, 6, 4, 7, 3, 7, 3, 8, 1, 9, 2, 8, 2, 8, 1, 6, 4], [5, 9, 1, 9, 2, 6, 4, 7, 3, 7, 3, 8, 1, 9, 2, 8, 2, 8, 0, 6, 4], [5, 9, 1, 9, 2, 6, 4, 7, 3, 7, 3, 8, 1, 9, 2, 8, 2, 8, 1, 6, 4], [5, 9, 1, 9, 2, 6, 4, 7, 3, 7, 3, 8, 1, 9, 2, 8, 2, 8, 1, 6, 4], [6, 10, 1, 10, 1, 7, 5, 8, 4, 8, 4, 9, 2, 10, 1, 9, 2, 9, 2, 7, 5], [6, 10, 1, 10, 0, 7, 5, 8, 4, 8, 4, 9, 2, 10, 1, 9, 2, 9, 2, 7, 5], [6, 10, 0, 10, 1, 7, 5, 8, 4, 8, 4, 9, 2, 10, 1, 9, 2, 9, 2, 7, 5], [6, 10, 1, 10, 1, 7, 5, 8, 4, 8, 4, 9, 2, 10, 1, 9, 2, 9, 2, 7, 5]]
A1_NEXT = [[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0], [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0], [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 2, 0, 1, 0, 0, 0], [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 2, 0, 1, 0, 0, 0], [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 2, 0, 1, 0, 0, 0], [0, 2, 0, 2, 0, 0, 0, 4, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0], [0, 2, 0, 2, 0, 0, 0, 3, 0, 4, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0], [0, 2, 0, 2, 0, 0, 0, 3, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0], [0, 2, 0, 2, 0, 0, 0, 3, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0], [0, 2, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 4, 0], [0, 2, 0, 2, 0, 4, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 0], [0, 2, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 0], [0, 2, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 0], [4, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0], [3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0], [3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0], [3, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0], [2, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 4], [2, 2, 0, 2, 0, 2, 4, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 3], [2, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 3], [2, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 3], [2, 2, 0, 2, 0, 2, 2, 2, 4, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2], [2, 2, 0, 2, 0, 2, 2, 2, 3, 2, 4, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2], [2, 2, 0, 2, 0, 2, 2, 2, 3, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2], [2, 2, 0, 2, 0, 2, 2, 2, 3, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2], [2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 4, 2, 0, 2, 0, 2, 1, 2, 2], [2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 3, 2, 0, 2, 0, 2, 1, 2, 2], [2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 3, 2, 0, 2, 0, 2, 1, 2, 2], [2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 3, 2, 0, 2, 0, 2, 1, 2, 2], [2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 0, 2, 2, 2, 2], [2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 0, 2, 2, 2, 2], [2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 4, 2, 2, 2, 2], [2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 0, 2, 2, 2, 2], [3, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3], [3, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3], [3, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3], [3, 4, 3, 1, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3], [3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 3, 3], [3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 4, 3, 3, 3], [3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 3, 3], [3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 3, 3], [3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 1, 3, 3], [3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 4, 3, 3], [3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 1, 3, 3], [3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 1, 3, 3], [3, 3, 0, 3, 1, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3], [3, 3, 0, 3, 4, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3], [3, 3, 4, 3, 1, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3], [3, 3, 0, 3, 1, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3]]
A1_TEMPLATES = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
A1_COUNTER_CELLS = [[2, 3], [2, 7], [4, 3], [4, 7], [2, 2], [2, 8], [2, 1], [2, 9], [3, 0], [3, 10], [5, 2], [5, 8], [2, 4], [2, 6]]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
VIEW = 5
NUM_ING = 4
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
A_STAY, A_INTERACT = 4, 5
INF = 999

CH_SELF_DIR = 1
CH_SELF_INV = 5
CH_PART_POS = 11
CH_PART_INV = 16
CH_STATIC = 22
CH_ITEM = 32
CH_RECIPE = 38
CH_TIMER = 44

POT_R, POT_C = 2, 5
R_R, R_C = 4, 5

# agent_0 targets: 0 deliver (1,5)UP, 1 pot (1,5)DOWN, 2 plate (1,4)UP, 3 plate (1,6)UP,
# 4 ing0 (1,4)LEFT, 5 ing1 (1,6)RIGHT, 6 counter(2,4) (1,4)DOWN, 7 counter(2,6) (1,6)DOWN
T0_DELIVER, T0_POT, T0_PLATE_L, T0_PLATE_R, T0_ING0, T0_ING1, T0_C24, T0_C26 = 0, 1, 2, 3, 4, 5, 6, 7
STALL_STEPS = 15
# behaviour flags (also used to generate synthetic cross-play partners)
A0_HELP_MODE = 1      # 0: never add ingredients, 1: add at count 1 (+stall-breaker), 2: add at count 1 or 2
A1_PREFETCH = 1       # fetch the recipe ingredient while the pot is cooking
A1_NAIVE_INSERT = 0   # 1: insert held ingredient whenever the pot is idle and not full (no checks)
A1_SIGNAL_ONLY = 0    # 1: only ever insert into an empty pot, then wait

# agent_1 targets: 0 pot (3,5)UP, 1 ing0 (4,1)LEFT, 2 ing0 (4,9)RIGHT, 3 ing1 (4,1)DOWN,
# 4 ing1 (4,9)DOWN, 5 ing3 (3,4)DOWN, 6 ing3 (3,6)DOWN, 7.. counters (see A1_COUNTER_CELLS)
T1_POT = 0
T1_COUNTER0 = 7
N_COUNTERS = 14
N_T1 = 21
# supply target mask per ingredient type (rows: ingredient 0..3)
SUPPLY_MASK_LIT = [
    [0, 1, 1, 0, 0, 0, 0] + [0] * 14,
    [0, 0, 0, 1, 1, 0, 0] + [0] * 14,
    [0] * 21,
    [0, 0, 0, 0, 0, 1, 1] + [0] * 14,
]
# counters (2,4) and (2,6) are shared with the top room: never used as drop targets
DROP_ALLOWED_LIT = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
PARTNER_HOLD_LIMIT = 10


# --------------------------------------------------------------------------
# JAX helpers (all fixed shape)
# --------------------------------------------------------------------------

def _localize(templates, obs):
    static = obs[:, :, CH_STATIC:CH_STATIC + 10]
    diff = jnp.sum(jnp.abs(templates - static[None]), axis=(1, 2, 3))
    return jnp.argmin(diff).astype(jnp.int32)


def _facing(obs):
    return jnp.argmax(obs[2, 2, CH_SELF_DIR:CH_SELF_DIR + 4]).astype(jnp.int32)


def _read_cell(obs, my_r, my_c, cell_r, cell_c):
    """Return (feature vector [45], valid flag) of a global cell inside the local crop."""
    lr = cell_r - my_r + 2
    lc = cell_c - my_c + 2
    valid = (lr >= 0) & (lr < VIEW) & (lc >= 0) & (lc < VIEW)
    lr_c = jnp.clip(lr, 0, VIEW - 1)
    lc_c = jnp.clip(lc, 0, VIEW - 1)
    feat = obs[lr_c, lc_c, :] * valid.astype(obs.dtype)
    return feat, valid


def _item_type(feat, base):
    """Classify an item block: -1 empty, 0..3 raw ingredient, 4 plate/dish."""
    plate = feat[base]
    cooked = feat[base + 1]
    counts = feat[base + 2:base + 2 + NUM_ING]
    total = jnp.sum(counts)
    other = (plate + cooked) > 0
    ing = jnp.argmax(counts).astype(jnp.int32)
    return jnp.where(other, 4, jnp.where(total > 0, ing, -1)).astype(jnp.int32)


def _pot_state(feat):
    counts = feat[CH_ITEM + 2:CH_ITEM + 2 + NUM_ING]
    count = jnp.sum(counts).astype(jnp.int32)
    ptype = jnp.where(count > 0, jnp.argmax(counts), -1).astype(jnp.int32)
    timer = feat[CH_TIMER].astype(jnp.int32)
    cooked = feat[CH_ITEM + 1].astype(jnp.int32)
    idle = (timer == 0) & (cooked == 0)
    return count, ptype, timer, cooked, idle


def _partner(obs):
    pos = obs[:, :, CH_PART_POS]
    visible = jnp.sum(pos) > 0
    inv = jnp.sum(obs[:, :, CH_PART_INV:CH_PART_INV + 2 + NUM_ING] * pos[:, :, None], axis=(0, 1))
    return visible, inv


def _self_inventory(obs):
    inv = obs[2, 2, CH_SELF_INV:CH_SELF_INV + 2 + NUM_ING]
    plate = inv[0] > 0
    cooked = inv[1] > 0
    counts = inv[2:]
    held_k = jnp.where(jnp.sum(counts) > 0, jnp.argmax(counts), -1).astype(jnp.int32)
    return plate, cooked, held_k


def _nav_action(goal_tbl, next_tbl, state, target, interact):
    """Move toward target; at the goal state INTERACT (if requested) else STAY."""
    goal = goal_tbl[target]
    at_goal = state == goal
    move = next_tbl[state, target]
    return jnp.where(at_goal, jnp.where(interact, A_INTERACT, A_STAY), move).astype(jnp.int32)


def _nearest(dist_tbl, state, mask):
    d = dist_tbl[state]
    d = jnp.where(mask, d, INF)
    return jnp.argmin(d).astype(jnp.int32), jnp.min(d)


# --------------------------------------------------------------------------
# agent_0: top room (plating / delivery / cautious helping)
# --------------------------------------------------------------------------

PARTNER_IDLE_STEPS = 4


def init_agent_0():
    return {
        "pot_sig": jnp.zeros((4,), dtype=jnp.int32),
        "pot_unchanged": jnp.asarray(0, dtype=jnp.int32),
        "partner_idle": jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs).astype(jnp.int32)
    reset = jnp.asarray(reset)
    init = init_agent_0()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    templates = jnp.asarray(A0_TEMPLATES, dtype=jnp.int32)
    cells = jnp.asarray(A0_CELLS, dtype=jnp.int32)
    goal_tbl = jnp.asarray(A0_GOAL, dtype=jnp.int32)
    next_tbl = jnp.asarray(A0_NEXT, dtype=jnp.int32)

    cell = _localize(templates, obs)
    my_r = cells[cell, 0]
    my_c = cells[cell, 1]
    facing = _facing(obs)
    state = cell * 4 + facing
    held_plate, held_cooked, held_k = _self_inventory(obs)

    pot_feat, _ = _read_cell(obs, my_r, my_c, POT_R, POT_C)
    count, ptype, timer, cooked, idle = _pot_state(pot_feat)
    c24 = _item_type(_read_cell(obs, my_r, my_c, 2, 4)[0], CH_ITEM)
    c26 = _item_type(_read_cell(obs, my_r, my_c, 2, 6)[0], CH_ITEM)
    free24 = c24 == -1
    free26 = c26 == -1
    any_free = free24 | free26

    sig = jnp.stack([count, ptype, timer, cooked]).astype(jnp.int32)
    same = jnp.all(sig == memory["pot_sig"])
    unchanged = jnp.where(same, memory["pot_unchanged"] + 1, 0).astype(jnp.int32)
    stall = unchanged >= STALL_STEPS

    # partner visibly idle (in view, empty-handed) for several steps: it is not
    # fetching, so completing the pot ourselves cannot cause an overflow.
    p_vis, p_inv = _partner(obs)
    p_empty = p_vis & (jnp.sum(p_inv) == 0)
    partner_idle = jnp.where(p_empty, memory["partner_idle"] + 1, 0).astype(jnp.int32)
    partner_waiting = partner_idle >= PARTNER_IDLE_STEPS

    # a held dish that cannot match any recipe (mixed types, or ingredient 2/3)
    inv_counts = obs[2, 2, CH_SELF_INV + 2:CH_SELF_INV + 2 + NUM_ING]
    dish_types = jnp.sum((inv_counts > 0).astype(jnp.int32))
    dish_bad = held_cooked & ((dish_types >= 2) | (inv_counts[2] > 0) | (inv_counts[3] > 0))

    left_side = my_c <= 5

    def nav(target, interact):
        return _nav_action(goal_tbl, next_tbl, state, target, interact)

    # holding cooked dish -> deliver (or park a certainly-wrong dish on a free counter)
    dump_target = jnp.where(free24 & (left_side | ~free26), T0_C24, T0_C26)
    act_dump = nav(dump_target, True)
    act_deliver = jnp.where(dish_bad & any_free, act_dump, nav(T0_DELIVER, True))
    # holding plate -> scoop when cooked else wait facing the pot
    act_plate = nav(T0_POT, cooked > 0)
    # holding raw ingredient
    type_ok = ptype == held_k
    help_count_ok = jnp.where(A0_HELP_MODE == 2, (count == 1) | (count == 2),
                              (count == 1) | ((count == 2) & (stall | partner_waiting)))
    insert_ok = idle & type_ok & help_count_ok
    insert_fallback = idle & type_ok & (count > 0) & (count < 3) & (~any_free)
    act_wait = nav(T0_POT, False)
    act_insert = nav(T0_POT, True)
    act_ing = jnp.where(insert_ok | insert_fallback, act_insert,
                        jnp.where(any_free, act_dump, act_wait))
    # empty hands
    busy = (cooked > 0) | (timer > 0)
    plate_target = jnp.where(left_side, T0_PLATE_L, T0_PLATE_R)
    act_getplate = nav(plate_target, True)
    helpable = idle & (count > 0) & (count < 3) & ((ptype == 0) | (ptype == 1)) & \
        help_count_ok & (A0_HELP_MODE > 0)
    from_c24 = c24 == ptype
    from_c26 = c26 == ptype
    src = jnp.where(from_c24 & (left_side | ~from_c26), T0_C24,
                    jnp.where(from_c26, T0_C26,
                              jnp.where(ptype == 0, T0_ING0, T0_ING1)))
    src_is_counter = from_c24 | from_c26
    can_help = helpable & (any_free | src_is_counter)
    act_fetch = nav(src, True)
    act_empty = jnp.where(busy, act_getplate, jnp.where(can_help, act_fetch, act_wait))

    action = jnp.where(held_cooked, act_deliver,
                       jnp.where(held_plate, act_plate,
                                 jnp.where(held_k >= 0, act_ing, act_empty)))
    mode = jnp.where(held_cooked, 1,
                     jnp.where(held_plate, 2,
                               jnp.where(held_k >= 0, 3,
                                         jnp.where(busy, 4, jnp.where(can_help, 5, 0))))).astype(jnp.int32)

    next_memory = {"pot_sig": sig, "pot_unchanged": unchanged, "partner_idle": partner_idle}
    trace = {"cell": cell, "mode": mode, "count": count, "ptype": ptype, "timer": timer,
             "unchanged": unchanged, "partner_idle": partner_idle, "dish_bad": dish_bad}
    return action.astype(jnp.int32), next_memory, trace


# --------------------------------------------------------------------------
# agent_1: bottom room (recipe reading / ingredient fetching)
# --------------------------------------------------------------------------

def init_agent_1():
    return {
        "recipe": jnp.asarray(-1, dtype=jnp.int32),
        "counters": jnp.full((N_COUNTERS,), -1, dtype=jnp.int32),
        "partner_cooked_steps": jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs).astype(jnp.int32)
    reset = jnp.asarray(reset)
    init = init_agent_1()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    templates = jnp.asarray(A1_TEMPLATES, dtype=jnp.int32)
    cells = jnp.asarray(A1_CELLS, dtype=jnp.int32)
    goal_tbl = jnp.asarray(A1_GOAL, dtype=jnp.int32)
    next_tbl = jnp.asarray(A1_NEXT, dtype=jnp.int32)
    dist_tbl = jnp.asarray(A1_DIST, dtype=jnp.int32)
    counter_cells = jnp.asarray(A1_COUNTER_CELLS, dtype=jnp.int32)
    supply_mask = jnp.asarray(SUPPLY_MASK_LIT, dtype=jnp.int32) > 0
    drop_allowed = jnp.asarray(DROP_ALLOWED_LIT, dtype=jnp.int32) > 0

    cell = _localize(templates, obs)
    my_r = cells[cell, 0]
    my_c = cells[cell, 1]
    facing = _facing(obs)
    state = cell * 4 + facing
    held_plate, held_cooked, held_k = _self_inventory(obs)
    held_other = held_plate | held_cooked

    # recipe (always-on indicator)
    r_feat, r_valid = _read_cell(obs, my_r, my_c, R_R, R_C)
    r_counts = r_feat[CH_RECIPE + 2:CH_RECIPE + 2 + NUM_ING]
    r_seen = r_valid & (jnp.sum(r_counts) > 0)
    recipe = jnp.where(r_seen, jnp.argmax(r_counts), memory["recipe"]).astype(jnp.int32)

    # pot
    pot_feat, pot_vis = _read_cell(obs, my_r, my_c, POT_R, POT_C)
    count, ptype, timer, cooked, idle = _pot_state(pot_feat)

    # partner (top room) - is it carrying a cooked dish right now?
    p_vis, p_inv = _partner(obs)
    p_cooked = p_vis & (p_inv[1] > 0)
    p_steps = jnp.where(p_cooked, memory["partner_cooked_steps"] + 1,
                        jnp.where(p_vis, 0, memory["partner_cooked_steps"])).astype(jnp.int32)
    partner_busy = p_cooked & (p_steps <= PARTNER_HOLD_LIMIT)

    # counter contents memory (updated when in view)
    def upd(i, mem_val):
        feat, valid = _read_cell(obs, my_r, my_c, counter_cells[i, 0], counter_cells[i, 1])
        t = _item_type(feat, CH_ITEM)
        return jnp.where(valid, t, mem_val)
    counters = jax.vmap(upd)(jnp.arange(N_COUNTERS), memory["counters"]).astype(jnp.int32)

    zeros_head = jnp.zeros((T1_COUNTER0,), dtype=bool)

    def nav(target, interact):
        return _nav_action(goal_tbl, next_tbl, state, target, interact)

    # drop target: nearest allowed counter known to be empty
    drop_mask = jnp.concatenate([zeros_head, (counters == -1) & drop_allowed])
    drop_t, drop_d = _nearest(dist_tbl, state, drop_mask)
    act_drop = nav(drop_t, True)
    act_wait = nav(T1_POT, False)
    act_insert = nav(T1_POT, True)

    # holding raw ingredient k
    k = held_k
    flush_ok = pot_vis & idle & (count > 0) & (count < 3) & (ptype == k) & (A1_SIGNAL_ONLY == 0)
    fresh_ok = pot_vis & idle & (count == 0) & (recipe == k) & (~partner_busy)
    naive_ok = pot_vis & idle & (count < 3) & (A1_NAIVE_INSERT == 1)
    flush_ok = flush_ok | naive_ok
    fresh_ok = fresh_ok | naive_ok
    wait_ok = pot_vis & (recipe == k) & (((count == 0) & idle) | (~idle))
    act_hold = jnp.where(~pot_vis, act_wait,
                         jnp.where(flush_ok | fresh_ok, act_insert,
                                   jnp.where(wait_ok, act_wait,
                                             jnp.where(drop_d < INF, act_drop, act_wait))))

    # empty hands: fetch what the pot needs (pot type if partially filled, else recipe)
    need = jnp.where(pot_vis & idle & (count > 0), ptype, recipe).astype(jnp.int32)
    need_c = jnp.clip(need, 0, NUM_ING - 1)
    src_mask = supply_mask[need_c] | jnp.concatenate([zeros_head, counters == need])
    src_t, src_d = _nearest(dist_tbl, state, src_mask)
    act_fetch = nav(src_t, True)
    hold_off = pot_vis & idle & (count == 0) & partner_busy
    no_prefetch_block = (A1_PREFETCH == 0) & pot_vis & (~idle)
    signal_block = (A1_SIGNAL_ONLY == 1) & pot_vis & (count > 0)
    act_empty = jnp.where((recipe < 0) | (need < 0) | (src_d >= INF) | hold_off
                          | no_prefetch_block | signal_block, act_wait, act_fetch)

    act_other = jnp.where(drop_d < INF, act_drop, act_wait)

    action = jnp.where(held_other, act_other,
                       jnp.where(held_k >= 0, act_hold, act_empty))
    mode = jnp.where(held_other, 1, jnp.where(held_k >= 0, 2, 3)).astype(jnp.int32)

    next_memory = {"recipe": recipe, "counters": counters, "partner_cooked_steps": p_steps}
    trace = {"cell": cell, "mode": mode, "recipe": recipe, "need": need, "src": src_t,
             "count": count, "ptype": ptype, "timer": timer}
    return action.astype(jnp.int32), next_memory, trace
