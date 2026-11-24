"""Overcooked V2 demo_cook_wide joint policy (convention-light, evidence-driven).

agent_0 (top room; plates, delivery, pot, ingredient piles; cannot see the
recipe indicator). Infers the recipe from the shared pot contents (strongest),
then raw ingredients left by the partner on the shared counters (2,4)/(2,6),
then the partner's visibly held ingredient. Fills the pot, fetches a plate while
cooking, delivers pure dishes. While a dish cooks it watches for contrary
evidence (partner holding, or freshly placing, a *different* ingredient type):
that means the pot was seeded with a stale ingredient, so the dish is dumped on
a shared counter instead of delivered (avoids the -20). Mixed dishes are always
dumped. After a long silence it falls back to alternating guesses.

agent_1 (bottom room; indicator, ingredient piles, pot). Reads the always-on
indicator, fetches the recipe ingredient and inserts it whenever the pot is
idle, compatible, and the partner is not holding a cooked dish (delivery is
imminent and the recipe may change). Holds a speculative ingredient for the
next round, stashes mismatched ingredients in off-view caches at (4,3)/(4,7),
and clears dumped dishes from the shared counters to off-view counters.

Lookup tables below were generated offline (gen_tables.py) from the layout:
CELLS*: candidate floor cells per seat; EXPECTED*: static-channel crops used for
localization; TABLE*[target, cell, facing] -> action from BFS over (cell, facing).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'pot_follow_v2', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

A_RIGHT = 0
A_DOWN = 1
A_LEFT = 2
A_UP = 3
A_STAY = 4
A_INTERACT = 5

CELLS0 = [[1, 4], [1, 5], [1, 6]]
EXPECTED0 = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
TABLE0 = [[[2, 2, 2, 5], [2, 2, 2, 2], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 5, 0]], [[5, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [5, 3, 3, 3]], [[0, 0, 0, 0], [1, 5, 1, 1], [2, 2, 2, 2]], [[0, 0, 0, 0], [5, 3, 3, 3], [2, 2, 2, 2]], [[1, 5, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [1, 5, 1, 1]], [[0, 0, 0, 0], [1, 4, 1, 1], [2, 2, 2, 2]]]
CELLS1 = [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [4, 1], [4, 2], [4, 8], [4, 9]]
EXPECTED1 = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
TABLE1 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [5, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2]], [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 5], [2, 2, 2, 2], [3, 3, 3, 3], [2, 2, 2, 2]], [[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [1, 5, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [3, 3, 3, 3], [0, 0, 0, 0], [0, 0, 5, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [3, 3, 3, 3], [0, 0, 0, 0], [1, 5, 1, 1]], [[0, 0, 0, 0], [0, 0, 0, 0], [1, 5, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 5, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2]], [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [1, 5, 1, 1], [3, 3, 3, 3], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [0, 0, 0, 0], [3, 3, 3, 3], [1, 5, 1, 1], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [5, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [5, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2]], [[5, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 3, 3], [2, 2, 2, 2], [3, 3, 3, 3], [2, 2, 2, 2]], [[2, 2, 2, 5], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [5, 3, 3, 3], [0, 0, 0, 0], [3, 3, 3, 3], [0, 0, 0, 0], [3, 3, 3, 3]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 0], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]]

# agent_0 targets
T0_SUPPLY0 = 0   # (1,4) face LEFT  -> ingredient 0 pile (1,3)
T0_SUPPLY1 = 1   # (1,6) face RIGHT -> ingredient 1 pile (1,7)
T0_PLATE_L = 2   # (1,4) face UP    -> plate pile (0,4)
T0_PLATE_R = 3   # (1,6) face UP    -> plate pile (0,6)
T0_POT = 4       # (1,5) face DOWN  -> pot (2,5)
T0_DELIVER = 5   # (1,5) face UP    -> delivery (0,5)
T0_CTR_L = 6     # (1,4) face DOWN  -> shared counter (2,4)
T0_CTR_R = 7     # (1,6) face DOWN  -> shared counter (2,6)
T0_WAIT = 8      # (1,5) face DOWN, no interact

# agent_1 targets
T1_POT = 0       # (3,5) face UP    -> pot (2,5)
T1_WAIT = 1      # (3,5) face UP, no interact
T1_SUP0_L = 2    # (4,1) face LEFT  -> ingredient 0 pile (4,0)
T1_SUP1_L = 3    # (4,1) face DOWN  -> ingredient 1 pile (5,1)
T1_SUP0_R = 4    # (4,9) face RIGHT -> ingredient 0 pile (4,10)
T1_SUP1_R = 5    # (4,9) face DOWN  -> ingredient 1 pile (5,9)
T1_SLOT0 = 6     # (3,3) face DOWN  -> cache (4,3) for type 0
T1_SLOT1 = 7     # (3,7) face DOWN  -> cache (4,7) for type 1
T1_SLOT0B = 8    # (4,2) face DOWN  -> secondary cache (5,2)
T1_SLOT1B = 9    # (4,8) face DOWN  -> secondary cache (5,8)
T1_CTR_L = 10    # (3,4) face UP    -> shared counter (2,4)
T1_CTR_R = 11    # (3,6) face UP    -> shared counter (2,6)
T1_JUNK_A = 12   # (3,1) face UP    -> junk counter (2,1)
T1_JUNK_B = 13   # (3,1) face LEFT  -> junk counter (3,0)
T1_JUNK_C = 14   # (3,9) face UP    -> junk counter (2,9)
T1_JUNK_D = 15   # (3,9) face RIGHT -> junk counter (3,10)

# agent_0 memory slots
M0_SSD = 0        # steps since own delivery (or dump)
M0_IDLE = 1       # consecutive steps without any recipe signal
M0_FALLBACK = 2   # guessing fallback engaged
M0_GUESS = 3      # next guess type
M0_CL_STATE = 4   # counter (2,4): 0 empty, 1 foreign item, 2 own item
M0_CR_STATE = 5
M0_CL_AGE = 6     # steps since the item appeared
M0_CR_AGE = 7
M0_CL_EARLY = 8   # item appeared within POST_DELIVERY_GUARD of own delivery
M0_CR_EARLY = 9
M0_POT_AGE = 10   # steps since the pot became non-empty
M0_CONTRA = 11    # contrary evidence against the current pot/dish type
M0_DROP = 12      # own drop performed last step: 0 none, 1 left, 2 right
M0_PHELD = 13     # partner's held raw type last step (-1 none / not visible)
M0_POTPREV = 14   # pot ingredient count last step
M0_INFER = 15     # recipe type inferred from partner discarding its ingredient (-1 none)
M0_PFRESH = 16    # partner seen empty-handed since my last delivery
M0_PWAIT = 17     # steps spent waiting for the partner to complete the pot
M0_T = 18
IDLE_FALLBACK_STEPS = 50
POST_DELIVERY_GUARD = 10
SUSPECT_EXPIRY = 30
DISCARD_WINDOW = 10

# agent_1 memory slots
M1_R = 0          # last seen recipe type (-1 unknown)
M1_S0 = 1         # cache (4,3) holds type 0
M1_S1 = 2         # cache (4,7) holds type 1
M1_S0B = 3
M1_S1B = 4
M1_MISMATCH = 5   # steps pot has held idle contents incompatible with recipe
M1_PDISH = 6      # steps partner has been holding a cooked dish
M1_JA = 7         # junk counters occupied
M1_JB = 8
M1_JC = 9
M1_JD = 10
M1_T = 11
MISMATCH_FILL_STEPS = 40
PARTNER_DISH_TIMEOUT = 25


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _localize(obs, expected, cells):
    static = obs[:, :, 22:32]
    mism = jnp.sum(jnp.abs(static[None] - expected), axis=(1, 2, 3))
    ci = jnp.argmin(mism).astype(jnp.int32)
    rc = cells[ci]
    return rc[0], rc[1], ci


def _cell(obs, r, c, gr, gc):
    """Observation vector at global cell (gr, gc); zeros if outside the view."""
    lr = gr - r + 2
    lc = gc - c + 2
    vis = (lr >= 0) & (lr <= 4) & (lc >= 0) & (lc <= 4)
    vals = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)]
    return vals * vis.astype(jnp.int32), vis


def _self(obs):
    me = obs[2, 2]
    facing = jnp.argmax(me[1:5]).astype(jnp.int32)
    inv = me[5:11]
    return facing, inv


def _partner(obs):
    pmask = obs[:, :, 11]
    pvis = jnp.sum(pmask) > 0
    pinv = jnp.sum(obs[:, :, 16:22] * pmask[:, :, None], axis=(0, 1))
    return pvis, pinv


def _pot_state(vals):
    counts = vals[34:38]
    cnt = jnp.sum(counts)
    cooked = vals[33] > 0
    cooking = vals[44] > 0
    idle = (~cooked) & (~cooking)
    n_types = jnp.sum((counts > 0).astype(jnp.int32))
    ptype = jnp.argmax(counts).astype(jnp.int32)
    return counts, cnt, cooked, cooking, idle, n_types, ptype


def _raw_item(vals):
    """Counter item: (has_any_item, is_single_raw_ingredient_0_or_1, type, has_plate)."""
    items = vals[32:38]
    has = jnp.sum(items) > 0
    raw_good = (items[0] == 0) & (items[1] == 0) & (jnp.sum(items[2:4]) == 1) & (jnp.sum(items[4:6]) == 0)
    typ = jnp.argmax(items[2:6]).astype(jnp.int32)
    plated = items[0] > 0
    return has, raw_good, typ, plated


def _init0():
    return jnp.asarray([100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 1, 0, 0], dtype=jnp.int32)


def _init1():
    return jnp.asarray([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def init_agent_0():
    return _init0()


def init_agent_1():
    return _init1()


def _counter_update(has, prev_state, prev_age, prev_early, own_drop, ssd):
    appeared = has & (prev_state == 0)
    state = jnp.where(~has, 0, jnp.where(appeared, jnp.where(own_drop, 2, 1), prev_state))
    age = jnp.where(~has, 0, jnp.where(appeared, 0, prev_age + 1))
    early = jnp.where(~has, 0, jnp.where(appeared, (ssd < POST_DELIVERY_GUARD).astype(jnp.int32), prev_early))
    return state.astype(jnp.int32), age.astype(jnp.int32), early.astype(jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs).astype(jnp.int32)
    init = _init0()
    mem = jnp.where(jnp.asarray(reset, dtype=bool), init, memory)
    table = jnp.asarray(TABLE0, dtype=jnp.int32)
    expected = jnp.asarray(EXPECTED0, dtype=jnp.int32)
    cells = jnp.asarray(CELLS0, dtype=jnp.int32)

    r, c, ci = _localize(obs, expected, cells)
    facing, inv = _self(obs)
    held_plate = inv[0] > 0
    held_cooked = inv[1] > 0
    held_counts = inv[2:6]
    held_n = jnp.sum(held_counts)
    held_type = jnp.argmax(held_counts).astype(jnp.int32)
    holding_ing = held_n > 0
    holding_dish = held_cooked
    holding_plate = held_plate & (~held_cooked)

    pot_vals, _ = _cell(obs, r, c, 2, 5)
    pot_counts, pot_cnt, pot_cooked, pot_cooking, pot_idle, pot_ntypes, pot_type = _pot_state(pot_vals)
    pot_pure_good = (pot_cnt > 0) & (pot_ntypes == 1) & (pot_type <= 1)
    pot_bad = (pot_cnt > 0) & (~pot_pure_good)
    pot_age = jnp.where(pot_cnt == 0, 0, mem[M0_POT_AGE] + 1)

    ctrL, _ = _cell(obs, r, c, 2, 4)
    ctrR, _ = _cell(obs, r, c, 2, 6)
    hasL, rawL, typL, _pl = _raw_item(ctrL)
    hasR, rawR, typR, _pr = _raw_item(ctrR)

    ssd = mem[M0_SSD]
    stL, ageL, earlyL = _counter_update(hasL, mem[M0_CL_STATE], mem[M0_CL_AGE], mem[M0_CL_EARLY], mem[M0_DROP] == 1, ssd)
    stR, ageR, earlyR = _counter_update(hasR, mem[M0_CR_STATE], mem[M0_CR_AGE], mem[M0_CR_EARLY], mem[M0_DROP] == 2, ssd)
    foreignL = hasL & rawL & (stL == 1)
    foreignR = hasR & rawR & (stR == 1)
    # an item that was already there at my last delivery (or appeared right after) was
    # fetched under the previous recipe: 50/50 stale. Trust it only after a long quiet spell.
    # sinceX: steps between my last delivery and the item's appearance (negative = pre-delivery)
    sinceL = ssd - ageL
    sinceR = ssd - ageR
    preL = foreignL & (sinceL < 0)                                   # fetched under an unknown recipe
    preR = foreignR & (sinceR < 0)
    discardL = foreignL & (sinceL >= 0) & (sinceL <= DISCARD_WINDOW)  # dropped right after delivery
    discardR = foreignR & (sinceR >= 0) & (sinceR <= DISCARD_WINDOW)
    freshL = foreignL & (sinceL > DISCARD_WINDOW)                     # fetched under the current recipe
    freshR = foreignR & (sinceR > DISCARD_WINDOW)
    sigL = freshL | (preL & (ssd >= SUSPECT_EXPIRY))
    sigR = freshR | (preR & (ssd >= SUSPECT_EXPIRY))

    pvis, pinv = _partner(obs)
    p_counts = pinv[2:6]
    p_type = jnp.argmax(p_counts).astype(jnp.int32)
    p_holding_ing = (jnp.sum(p_counts) > 0) & (pinv[0] == 0) & (pinv[1] == 0) & (p_type <= 1)
    # partner-held item is trusted only if the partner was seen empty-handed since my delivery
    # (so the item was fetched under the current recipe), or after a long quiet spell
    p_fresh = jnp.maximum(mem[M0_PFRESH], (pvis & (~p_holding_ing)).astype(jnp.int32))
    psig = pvis & p_holding_ing & (p_fresh > 0)

    # contrary evidence: reference type is the pot's pure type, or the held dish's type
    dish_types = jnp.sum((held_counts > 0).astype(jnp.int32))
    dish_good = holding_dish & (dish_types == 1) & (held_type <= 1)
    ref_valid = jnp.where(holding_dish, dish_good, pot_pure_good)
    ref_type = jnp.where(holding_dish, held_type, pot_type)
    ev_partner = pvis & p_holding_ing & (p_type != ref_type) & (p_fresh > 0)
    ev_ctrL = (ageL < pot_age) & ((discardL & (typL == ref_type)) | (freshL & (typL != ref_type)))
    ev_ctrR = (ageR < pot_age) & ((discardR & (typR == ref_type)) | (freshR & (typR != ref_type)))
    evidence = ref_valid & (ev_partner | ev_ctrL | ev_ctrR)
    contra = jnp.where((pot_cnt == 0) & (~holding_dish), 0, jnp.maximum(mem[M0_CONTRA], evidence.astype(jnp.int32)))

    # discard cue: shortly after my delivery the partner went from holding X to empty hands
    # while the pot stayed empty -> it discarded X because the recipe changed -> other type.
    p_held_now = jnp.where(pvis & p_holding_ing, p_type, -1).astype(jnp.int32)
    p_held_prev = mem[M0_PHELD]
    discard_cue = ((ssd <= DISCARD_WINDOW) & pvis & (~p_holding_ing) & (p_held_prev >= 0)
                   & (pot_cnt == 0) & (mem[M0_POTPREV] == 0))
    ctr_discard_type = jnp.where(discardL, 1 - typL, jnp.where(discardR, 1 - typR, -1))
    inferred = jnp.where(pot_cnt > 0, -1,
               jnp.where(discard_cue, 1 - p_held_prev,
               jnp.where(ctr_discard_type >= 0, ctr_discard_type, mem[M0_INFER]))).astype(jnp.int32)
    isig = (inferred >= 0) & (pot_cnt == 0)

    # partner about to complete the pot: do not fetch a surplus third ingredient (bounded wait)
    partner_completes = (pot_cnt == 2) & pvis & p_holding_ing & (p_type == pot_type)
    pwait = jnp.where(partner_completes, mem[M0_PWAIT] + 1, 0)
    skip_fetch = partner_completes & (pwait <= 8)

    any_signal = (pot_cnt > 0) | sigL | sigR | psig | isig
    idle_steps = jnp.where(any_signal, 0, mem[M0_IDLE] + 1)
    fallback = (mem[M0_FALLBACK] > 0) | (idle_steps > IDLE_FALLBACK_STEPS)
    guess = mem[M0_GUESS]

    neg = _i32(-1)
    X = jnp.where(pot_pure_good, pot_type,
        jnp.where(sigL, typL,
        jnp.where(sigR, typR,
        jnp.where(psig, p_type,
        jnp.where(isig, inferred,
        jnp.where(fallback, guess, neg))))))

    freeL = ~hasL
    freeR = ~hasR
    left_side = c <= 5
    drop_t = jnp.where(freeL & freeR, jnp.where(left_side, T0_CTR_L, T0_CTR_R),
             jnp.where(freeL, T0_CTR_L, jnp.where(freeR, T0_CTR_R, neg)))
    has_drop = drop_t >= 0

    def drop_or(alt):
        return jnp.where(has_drop, drop_t, alt)

    plate_t = jnp.where(left_side, T0_PLATE_L, T0_PLATE_R)

    def supply_t(x):
        return jnp.where(x == 0, T0_SUPPLY0, T0_SUPPLY1)

    def fetch_t(x):
        return jnp.where(hasL & rawL & (typL == x), T0_CTR_L,
               jnp.where(hasR & rawR & (typR == x), T0_CTR_R, supply_t(x)))

    # holding cooked dish: deliver pure 0/1 dishes unless contradicted; dump otherwise
    t_dish = jnp.where(dish_good & (contra == 0), T0_DELIVER, drop_or(T0_DELIVER))

    # holding empty plate
    t_plate = jnp.where(pot_cooked, T0_POT,
              jnp.where(pot_cooking, T0_WAIT, drop_or(T0_WAIT)))

    # holding raw ingredient h
    h = held_type
    insert_ok_empty = (X == h) | ((X == -1) & (ssd >= POST_DELIVERY_GUARD))
    t_ing_empty_pot = jnp.where(insert_ok_empty, T0_POT,
                      jnp.where((X != -1) & (X != h), drop_or(T0_POT), T0_WAIT))
    t_ing_nonempty = jnp.where(pot_bad, T0_POT,
                     jnp.where(pot_type == h, T0_POT, drop_or(T0_WAIT)))
    t_ing = jnp.where(pot_cooking | pot_cooked, drop_or(T0_WAIT),
            jnp.where(pot_cnt == 0, t_ing_empty_pot, t_ing_nonempty))

    # empty hands
    nearest_supply = jnp.where(left_side, T0_SUPPLY0, T0_SUPPLY1)
    t_empty = jnp.where(pot_cooking | pot_cooked, plate_t,
              jnp.where(pot_bad, nearest_supply,
              jnp.where(pot_cnt > 0, jnp.where(skip_fetch, T0_WAIT, fetch_t(pot_type)),
              jnp.where(X >= 0, fetch_t(X), T0_WAIT))))

    target = jnp.where(holding_dish, t_dish,
             jnp.where(holding_plate, t_plate,
             jnp.where(holding_ing, t_ing, t_empty))).astype(jnp.int32)
    action = table[target, ci, facing].astype(jnp.int32)

    interacting = action == A_INTERACT
    delivering = holding_dish & (target == T0_DELIVER) & interacting
    dumping = holding_dish & ((target == T0_CTR_L) | (target == T0_CTR_R)) & interacting
    dropping = (holding_ing | holding_plate | holding_dish) & interacting
    drop_flag = jnp.where(dropping & (target == T0_CTR_L), 1, jnp.where(dropping & (target == T0_CTR_R), 2, 0))
    cycle_end = delivering | dumping
    ssd_new = jnp.where(cycle_end, 0, jnp.minimum(ssd + 1, 1000))
    guess_new = jnp.where(delivering, 1 - guess, guess)
    contra_mem = jnp.where(cycle_end, 0, contra)
    infer_mem = jnp.where(cycle_end, -1, inferred)
    pfresh_mem = jnp.where(cycle_end, 0, p_fresh)

    new_mem = jnp.stack([
        ssd_new.astype(jnp.int32),
        idle_steps.astype(jnp.int32),
        fallback.astype(jnp.int32),
        guess_new.astype(jnp.int32),
        stL, stR, ageL, ageR, earlyL, earlyR,
        pot_age.astype(jnp.int32),
        contra_mem.astype(jnp.int32),
        drop_flag.astype(jnp.int32),
        p_held_now.astype(jnp.int32),
        pot_cnt.astype(jnp.int32),
        infer_mem.astype(jnp.int32),
        pfresh_mem.astype(jnp.int32),
        pwait.astype(jnp.int32),
        (mem[M0_T] + 1).astype(jnp.int32),
    ]).astype(jnp.int32)
    trace = {
        'target': target,
        'x': X.astype(jnp.int32),
        'pos': jnp.stack([r, c]).astype(jnp.int32),
        'fallback': fallback.astype(jnp.int32),
        'contra': contra.astype(jnp.int32),
        'infer': inferred.astype(jnp.int32),
        'pfresh': p_fresh.astype(jnp.int32),
    }
    return action, new_mem, trace


def _slot_has(obs, r, c, gr, gc, typ, mem_val):
    vals, vis = _cell(obs, r, c, gr, gc)
    has_obs = (vals[34 + typ] > 0).astype(jnp.int32)
    return jnp.where(vis, has_obs, mem_val)


def _occupied(obs, r, c, gr, gc, mem_val):
    vals, vis = _cell(obs, r, c, gr, gc)
    has_obs = (jnp.sum(vals[32:38]) > 0).astype(jnp.int32)
    return jnp.where(vis, has_obs, mem_val)


def agent_1_act(obs, reset, memory, rng):
    obs = jnp.asarray(obs).astype(jnp.int32)
    init = _init1()
    mem = jnp.where(jnp.asarray(reset, dtype=bool), init, memory)
    table = jnp.asarray(TABLE1, dtype=jnp.int32)
    expected = jnp.asarray(EXPECTED1, dtype=jnp.int32)
    cells = jnp.asarray(CELLS1, dtype=jnp.int32)

    r, c, ci = _localize(obs, expected, cells)
    facing, inv = _self(obs)
    held_counts = inv[2:6]
    holding_ing = jnp.sum(held_counts) > 0
    holding_plated = inv[0] > 0
    h = jnp.argmax(held_counts).astype(jnp.int32)

    # recipe indicator at (4,5)
    rvals, rvis = _cell(obs, r, c, 4, 5)
    rcounts = rvals[40:44]
    r_present = rvis & (jnp.sum(rcounts) > 0)
    R = jnp.where(r_present, jnp.argmax(rcounts).astype(jnp.int32), mem[M1_R])
    R_known = R >= 0
    R_idx = jnp.clip(R, 0, 3)

    # pot at (2,5)
    pot_vals, pot_vis = _cell(obs, r, c, 2, 5)
    pot_counts, pot_cnt, pot_cooked, pot_cooking, pot_idle, pot_ntypes, pot_type = _pot_state(pot_vals)

    # partner holding a cooked dish -> delivery imminent, recipe may change
    pvis, pinv = _partner(obs)
    p_dish = pvis & (pinv[1] > 0)
    p_dish_steps = jnp.where(p_dish, mem[M1_PDISH] + 1, 0)
    partner_block = p_dish & (p_dish_steps < PARTNER_DISH_TIMEOUT)

    others_zero = (pot_cnt - pot_counts[jnp.clip(h, 0, 3)]) == 0
    insertable = (pot_vis & R_known & holding_ing & (h == R) & pot_idle
                  & (pot_cnt < 3) & others_zero & (~partner_block))

    pot_matches_R = (pot_cnt == 0) | ((pot_ntypes == 1) & (pot_type == R))
    mismatch = pot_vis & pot_idle & (pot_cnt > 0) & (~pot_matches_R)
    mismatch_steps = jnp.where(mismatch, mem[M1_MISMATCH] + 1,
                     jnp.where(pot_vis, 0, mem[M1_MISMATCH]))
    fill_mode = mismatch_steps > MISMATCH_FILL_STEPS
    fill_insert = fill_mode & pot_vis & pot_idle & (pot_cnt < 3) & holding_ing

    # cache slots (out of the top room's view)
    s0 = _slot_has(obs, r, c, 4, 3, 0, mem[M1_S0])
    s1 = _slot_has(obs, r, c, 4, 7, 1, mem[M1_S1])
    s0b = _slot_has(obs, r, c, 5, 2, 0, mem[M1_S0B])
    s1b = _slot_has(obs, r, c, 5, 8, 1, mem[M1_S1B])

    # junk counters for dumped dishes
    ja = _occupied(obs, r, c, 2, 1, mem[M1_JA])
    jb = _occupied(obs, r, c, 3, 0, mem[M1_JB])
    jc = _occupied(obs, r, c, 2, 9, mem[M1_JC])
    jd = _occupied(obs, r, c, 3, 10, mem[M1_JD])

    # plated items on the shared counters (dumped dishes / plates) -> janitor duty
    ctrL, _ = _cell(obs, r, c, 2, 4)
    ctrR, _ = _cell(obs, r, c, 2, 6)
    platedL = (ctrL[32] > 0) & (ctrL[33] > 0)
    platedR = (ctrR[32] > 0) & (ctrR[33] > 0)

    left_side = c <= 5
    junk_left = jnp.where(ja == 0, T1_JUNK_A, jnp.where(jb == 0, T1_JUNK_B, jnp.where(jc == 0, T1_JUNK_C, T1_JUNK_D)))
    junk_right = jnp.where(jc == 0, T1_JUNK_C, jnp.where(jd == 0, T1_JUNK_D, jnp.where(ja == 0, T1_JUNK_A, T1_JUNK_B)))
    junk_t = jnp.where(left_side, junk_left, junk_right)

    stash_t = jnp.where(h == 0, jnp.where(s0 == 0, T1_SLOT0, T1_SLOT0B),
              jnp.where(s1 == 0, T1_SLOT1, T1_SLOT1B))
    t_hold = jnp.where(insertable | fill_insert, T1_POT,
             jnp.where(R_known & (h != R), stash_t, T1_WAIT))

    supply_t = jnp.where(left_side, jnp.where(R_idx == 0, T1_SUP0_L, T1_SUP1_L),
                         jnp.where(R_idx == 0, T1_SUP0_R, T1_SUP1_R))
    slot_has_R = jnp.where(R_idx == 0, s0 > 0, s1 > 0)
    slot_t_R = jnp.where(R_idx == 0, T1_SLOT0, T1_SLOT1)
    t_fetch = jnp.where(~R_known, T1_WAIT, jnp.where(slot_has_R, slot_t_R, supply_t))
    t_empty = jnp.where(platedL, T1_CTR_L, jnp.where(platedR, T1_CTR_R, t_fetch))

    target = jnp.where(holding_plated, junk_t,
             jnp.where(holding_ing, t_hold, t_empty)).astype(jnp.int32)
    action = table[target, ci, facing].astype(jnp.int32)

    # remember junk drops performed now (observation overrides when visible)
    junk_drop = holding_plated & (action == A_INTERACT)
    ja_n = jnp.maximum(ja, (junk_drop & (target == T1_JUNK_A)).astype(jnp.int32))
    jb_n = jnp.maximum(jb, (junk_drop & (target == T1_JUNK_B)).astype(jnp.int32))
    jc_n = jnp.maximum(jc, (junk_drop & (target == T1_JUNK_C)).astype(jnp.int32))
    jd_n = jnp.maximum(jd, (junk_drop & (target == T1_JUNK_D)).astype(jnp.int32))

    new_mem = jnp.stack([
        R.astype(jnp.int32), s0.astype(jnp.int32), s1.astype(jnp.int32),
        s0b.astype(jnp.int32), s1b.astype(jnp.int32),
        mismatch_steps.astype(jnp.int32), p_dish_steps.astype(jnp.int32),
        ja_n, jb_n, jc_n, jd_n,
        (mem[M1_T] + 1).astype(jnp.int32),
    ]).astype(jnp.int32)
    trace = {
        'target': target,
        'recipe': R.astype(jnp.int32),
        'pos': jnp.stack([r, c]).astype(jnp.int32),
        'insertable': insertable.astype(jnp.int32),
    }
    return action, new_mem, trace
