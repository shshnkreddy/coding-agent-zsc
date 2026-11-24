"""JAX-compatible cooperative Overcooked-v2 policy for the test_time_simple layout.

Layout analysis (from context/environment/specification.json object_positions
and wall list): the two seats occupy fully disconnected 3x2 rooms. Column 3 is
solid wall for every row, and column 4 holds a plate pile / pot / plate pile
(all non-floor), so agent_0's room (rows 1-3, cols 1-2) and agent_1's room
(rows 1-3, cols 5-6) share no reachable cell and are never within each
other's 5x5 view (column distance >= 3). The only recipe indicator (R) sits
at (2, 0), inside agent_0's room and far outside agent_1's view. agent_1 alone
can reach the pot, both plate piles, both recipe-relevant ingredient supplies
(0 and 1) and the delivery station; agent_0's only reachable ingredient (id 2)
is never part of any recipe ([3,0,0] or [0,3,0]). No button exists in this
layout, so there is no way to bring the recipe into agent_1's view, and no
counter is reachable from both rooms, so no information can pass between
seats. Consequently agent_0 cannot affect team reward under this fixed
layout, and agent_1 must act without ever observing the target recipe or a
delivery outcome (the observation carries no reward/success signal, and a
delivered dish leaves the hand regardless of correctness).

Since a wrong delivery does *not* resample the recipe, the optimal
feedback-free policy is to alternate the batched ingredient every delivery:
whatever the previous (unobserved) outcome, switching type guarantees a match
on the very next attempt whenever the previous one was wrong (only two
recipes exist), and gives a fresh 50/50 draw immediately after a correct one.
agent_1 runs a small reactive state machine driven purely by its own
observation (inventory + pot contents/timer as seen through its own crop),
toggling its target ingredient once per completed delivery. agent_0 has no
reachable object that changes the outcome, so it is a no-op.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Action ids
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Facing codes as encoded by self-facing one-hot channels 1..4 (UP, DOWN, RIGHT, LEFT)
FACE_UP, FACE_DOWN, FACE_RIGHT, FACE_LEFT = 0, 1, 2, 3

# Observation channel indices (see context/environment/specification.json)
CH_GOAL = 21
CH_POT = 22
CH_PLATE = 25
CH_ING0 = 26
CH_ING1 = 27


def _find_local(mask):
    """Return (row, col) in the 5x5 crop of the (assumed unique) 1-valued cell."""
    flat = mask.reshape(-1)
    idx = jnp.argmax(flat).astype(jnp.int32)
    return idx // 5, idx % 5


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    # agent_0's room is physically disconnected from every task-relevant
    # object (pot, plate piles, ingredients 0/1, delivery station); its only
    # reachable ingredient (id 2) is never part of any recipe, and it can
    # never see or be seen by agent_1. It cannot influence team reward, so it
    # simply stays put.
    action = jnp.asarray(STAY, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


def init_agent_1():
    return {'target_ingredient': jnp.asarray(0, dtype=jnp.int32)}


def agent_1_act(obs, reset, memory, rng):
    target_ingredient = memory['target_ingredient']

    plate_self = obs[2, 2, 5]
    cooked_self = obs[2, 2, 6]
    ing0_self = obs[2, 2, 7]
    ing1_self = obs[2, 2, 8]

    holding_dish = (plate_self == 1) & (cooked_self == 1)
    holding_plate_empty = (plate_self == 1) & (cooked_self == 0)
    ing_target_self = jnp.where(target_ingredient == 0, ing0_self, ing1_self)
    holding_ing_target = ing_target_self > 0

    pot_row, pot_col = _find_local(obs[:, :, CH_POT])
    pot_ing0 = obs[pot_row, pot_col, 31]
    pot_ing1 = obs[pot_row, pot_col, 32]
    pot_ing2 = obs[pot_row, pot_col, 33]
    pot_count = pot_ing0 + pot_ing1 + pot_ing2

    # mode: 0 = deliver, 1 = interact with pot (deposit or collect),
    # 2 = fetch matching ingredient, 3 = fetch an empty plate.
    mode = jnp.where(
        holding_dish, 0,
        jnp.where(
            holding_plate_empty | holding_ing_target, 1,
            jnp.where(pot_count < 3, 2, 3),
        ),
    )

    goal_row, goal_col = _find_local(obs[:, :, CH_GOAL])

    ing_mask = jnp.where(target_ingredient == 0, obs[:, :, CH_ING0], obs[:, :, CH_ING1])
    ing_row, ing_col = _find_local(ing_mask)

    # Two plate piles can be simultaneously visible; pick the one sharing the
    # current target ingredient's row (both are on the same side of the room).
    row_iota = jnp.broadcast_to(jnp.arange(5)[:, None], (5, 5))
    plate_mask = jnp.where(row_iota == ing_row, obs[:, :, CH_PLATE], 0)
    plate_row, plate_col = _find_local(plate_mask)

    target_row = jnp.where(
        mode == 0, goal_row,
        jnp.where(mode == 1, pot_row, jnp.where(mode == 2, ing_row, plate_row)),
    )
    target_col = jnp.where(
        mode == 0, goal_col,
        jnp.where(mode == 1, pot_col, jnp.where(mode == 2, ing_col, plate_col)),
    )

    # Every reachable target's only floor neighbor is one column away: the
    # goal and both ingredient supplies are approached from the west (agent
    # faces RIGHT, dc = +1); the pot and both plate piles are approached from
    # the east (agent faces LEFT, dc = -1).
    desired_dc = jnp.where((mode == 0) | (mode == 2), 1, -1)
    face_action = jnp.where((mode == 0) | (mode == 2), RIGHT, LEFT)
    desired_face_code = jnp.where(face_action == RIGHT, FACE_RIGHT, FACE_LEFT)

    dr = target_row - 2
    dc = target_col - 2

    facing_code = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    at_row = dr == 0
    at_col = dc == desired_dc
    at_target = at_row & at_col
    facing_ok = facing_code == desired_face_code

    action = jnp.where(
        at_target,
        jnp.where(facing_ok, INTERACT, face_action),
        jnp.where(
            dr > 0, DOWN,
            jnp.where(dr < 0, UP, jnp.where(dc > desired_dc, RIGHT, LEFT)),
        ),
    )
    action = jnp.asarray(action, dtype=jnp.int32)

    # A delivery attempt is committed the instant INTERACT is issued while
    # holding a plated dish, regardless of whether it will turn out correct
    # (no feedback is observable either way); alternate for the next attempt.
    will_deliver = (mode == 0) & at_target & facing_ok
    next_target_ingredient = jnp.where(will_deliver, 1 - target_ingredient, target_ingredient)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = {
        'target_ingredient': jnp.where(reset, zero, next_target_ingredient).astype(jnp.int32),
    }

    trace = {
        'mode': mode.astype(jnp.int32),
        'target_ingredient': target_ingredient,
        'pot_count': pot_count.astype(jnp.int32),
    }
    return action, next_memory, trace
