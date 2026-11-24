"""Policy for overcooked_v2 test_time_simple layout.

Map topology (fixed for this task instance): a solid wall column (col 3)
fully separates agent_0's room (cols 1-2) from agent_1's room (cols 5-6).
No cell or object is reachable by both seats, and neither seat's view radius
(2) can ever see across the wall. Consequently agent_0 cannot influence any
state agent_1 can perceive or use (and vice versa): zero bits of information
or material can cross, regardless of either seat's policy or its partner's.

agent_0's room contains the only recipe indicator (R) plus a decoy
ingredient-2 supply that is never used by any configured recipe. Since
agent_0 cannot reach the pot, plates, ingredient 0/1 supplies or delivery,
and cannot signal anything to agent_1, its actions provably cannot affect
team reward under any partner. It stays put.

agent_1's room has full access to the pot, both plate piles, ingredient 0/1
supplies and delivery, but can never observe the recipe indicator. Since an
incorrect delivery does not resample the hidden recipe while a correct one
does (uniformly, independently of history), and there are only two possible
recipes (3x ingredient 0 or 3x ingredient 1), a fixed open-loop alternation
of guesses across successive delivery attempts guarantees a correct delivery
within at most two consecutive attempts, independent of any partner's
behavior in agent_0's seat. agent_1 runs this alternation via a small
observation-reactive controller: every quantity it needs (self position,
pot contents/timer, own inventory) is recovered each step from the local
5x5x40 observation using the pot's static-channel position as a fixed
reference point, so no positional memory is required -- only the single
alternating recipe-guess bit persists across steps.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Action ids (README / specification.json).
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

# Fixed global object positions (row, col) for this layout.
POT_ROW, POT_COL = 2, 4
DELIVERY_STAND = (2, 6)      # face RIGHT into (2,7)
ING0_STAND = (1, 6)         # face RIGHT into (1,7)
ING1_STAND = (3, 6)         # face RIGHT into (3,7)
PLATE_STAND_ROW_A, PLATE_STAND_COL = 1, 5   # face LEFT into (1,4)
PLATE_STAND_ROW_B = 3                        # face LEFT into (3,4)
POT_STAND = (2, 5)          # face LEFT into (2,4)


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    # agent_0's room is fully disconnected from every reachable object used
    # by any recipe and from agent_1's view; no action can affect reward.
    action = jnp.asarray(STAY, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


def init_agent_1():
    # Single persistent bit: which recipe we are currently guessing (0 or 1).
    return jnp.asarray(0, dtype=jnp.int32)


def _navigate(sr, sc, tr, tc, want_right, face_right, face_left):
    """Greedy row-then-column navigation to (tr, tc), then turn to face.

    Returns (action_if_not_ready, arrived_and_facing_correctly).
    """
    at_row = sr == tr
    at_col = sc == tc
    row_action = jnp.where(sr < tr, DOWN, UP)
    col_action = jnp.where(sc < tc, RIGHT, LEFT)
    turn_action = jnp.where(want_right, RIGHT, LEFT)
    facing_ok = jnp.where(want_right, face_right == 1, face_left == 1)
    nav_action = jnp.where(
        jnp.logical_not(at_row), row_action,
        jnp.where(jnp.logical_not(at_col), col_action, turn_action),
    )
    arrived = jnp.logical_and(jnp.logical_and(at_row, at_col), facing_ok)
    return nav_action, arrived


def agent_1_act(obs, reset, memory, rng):
    guess = memory

    # Locate the (always-visible) pot in the local crop; use it as a fixed
    # reference point to recover this seat's own global position without
    # any positional memory.
    pot_static = obs[:, :, 22]
    pot_idx = jnp.argmax(pot_static.reshape(-1))
    pot_r = (pot_idx // 5).astype(jnp.int32)
    pot_c = (pot_idx % 5).astype(jnp.int32)
    self_r = (POT_ROW - (pot_r - 2)).astype(jnp.int32)
    self_c = (POT_COL - (pot_c - 2)).astype(jnp.int32)

    pot_ing0 = obs[pot_r, pot_c, 31]
    pot_ing1 = obs[pot_r, pot_c, 32]
    pot_cooked = obs[pot_r, pot_c, 30]
    pot_timer = obs[pot_r, pot_c, 39]

    plate_flag = obs[2, 2, 5]
    cooked_flag = obs[2, 2, 6]
    held_ing0 = obs[2, 2, 7]
    held_ing1 = obs[2, 2, 8]
    face_right = obs[2, 2, 3]
    face_left = obs[2, 2, 4]

    holding_cooked = cooked_flag == 1
    holding_plate_only = jnp.logical_and(plate_flag == 1, cooked_flag == 0)
    holding_ingredient = jnp.logical_or(held_ing0 > 0, held_ing1 > 0)

    # --- Case A: holding a plated cooked dish -> deliver it. ---
    nav_deliver, arrived_deliver = _navigate(
        self_r, self_c, DELIVERY_STAND[0], DELIVERY_STAND[1], True, face_right, face_left
    )
    action_a = jnp.where(arrived_deliver, INTERACT, nav_deliver)

    # --- Case B: holding an empty plate -> go to pot, scoop when cooked. ---
    nav_pot_b, arrived_pot_b = _navigate(
        self_r, self_c, POT_STAND[0], POT_STAND[1], False, face_right, face_left
    )
    action_b = jnp.where(arrived_pot_b, jnp.where(pot_cooked == 1, INTERACT, STAY), nav_pot_b)

    # --- Case C: holding a raw ingredient -> deposit it into the pot. ---
    nav_pot_c, arrived_pot_c = _navigate(
        self_r, self_c, POT_STAND[0], POT_STAND[1], False, face_right, face_left
    )
    action_c = jnp.where(arrived_pot_c, INTERACT, nav_pot_c)

    # --- Case D: empty hands -> fetch next ingredient unit, or a plate
    # once the guessed recipe's ingredient count is already satisfied /
    # cooking / cooked. ---
    guess_count = jnp.where(guess == 0, pot_ing0, pot_ing1)
    need_ingredient = jnp.logical_and(guess_count < 3, jnp.logical_and(pot_timer == 0, pot_cooked == 0))

    ing_target_row = jnp.where(guess == 0, ING0_STAND[0], ING1_STAND[0])
    ing_target_col = jnp.where(guess == 0, ING0_STAND[1], ING1_STAND[1])
    nav_ing, arrived_ing = _navigate(self_r, self_c, ing_target_row, ing_target_col, True, face_right, face_left)
    action_d_ing = jnp.where(arrived_ing, INTERACT, nav_ing)

    plate_target_row = jnp.where(self_r <= 2, PLATE_STAND_ROW_A, PLATE_STAND_ROW_B)
    nav_plate, arrived_plate = _navigate(self_r, self_c, plate_target_row, PLATE_STAND_COL, False, face_right, face_left)
    action_d_plate = jnp.where(arrived_plate, INTERACT, nav_plate)

    action_d = jnp.where(need_ingredient, action_d_ing, action_d_plate)

    action = jnp.where(
        holding_cooked, action_a,
        jnp.where(
            holding_plate_only, action_b,
            jnp.where(holding_ingredient, action_c, action_d),
        ),
    )

    delivered_now = jnp.logical_and(holding_cooked, arrived_deliver)
    new_guess = jnp.where(delivered_now, 1 - guess, guess)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, new_guess).astype(jnp.int32)

    return action.astype(jnp.int32), next_memory, {}
