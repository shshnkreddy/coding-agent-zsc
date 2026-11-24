"""Overcooked V2 (test_time_simple) policy.

Layout structural fact (fixed by the task's board geometry, invariant to any
partner policy): column 3 is a solid wall the full height of the board, so
agent_0's spawn room (rows 1-3, cols 1-2, plus the recipe indicator at
(2,0)) and agent_1's spawn room (rows 1-3, cols 5-6, plus pot/plates/
ingredient supplies 0 and 1/delivery at col 4 and col 7) are disjoint and
never mutually visible (room gap 3 > view radius 2). agent_0 can reach only
the always-on recipe indicator R and ingredient supply 2, which is never
part of any recipe; it cannot reach the pot, a plate pile, or the delivery
station, and cannot hand items to agent_1 (no shared counter). Its action
therefore cannot influence team reward, independent of the partner policy.
agent_1 can reach every task-relevant object but never observes the recipe
indicator, so it must act as a blind solo cook who infers the hidden
binary recipe (three of ingredient 0 or three of ingredient 1) purely from
delivery outcomes it cannot directly observe (no delivery-success channel).
Because a wrong delivery never resamples the recipe while a correct one
resamples uniformly at random, always switching the targeted ingredient
after every delivery attempt is the reward-maximizing blind strategy
(steady-state accuracy 2/3, versus 1/2 for a fixed guess).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'blind_solo_alternator',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids (match environment's runtime_contract).
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# Observation channel indices.
SELF_FACE_UP, SELF_FACE_DOWN, SELF_FACE_RIGHT, SELF_FACE_LEFT = 1, 2, 3, 4
SELF_ING0, SELF_ING1 = 7, 8
SELF_PLATE, SELF_COOKED = 5, 6
CH_POT = 22
CH_GOAL = 21
CH_PLATE_PILE = 25
CH_ING0_SUPPLY = 26
CH_ING1_SUPPLY = 27
GRID_COOKED = 30
GRID_ING0 = 31
GRID_ING1 = 32
POT_TIMER = 39


def _nearest(channel_slice):
    """Return (dr, dc, found) offset (relative to self) of the nearest cell
    where channel_slice > 0, using Manhattan distance; found=False if none."""
    rows = jnp.arange(5).reshape(5, 1)
    cols = jnp.arange(5).reshape(1, 5)
    dist = jnp.abs(rows - 2) + jnp.abs(cols - 2)
    mask = channel_slice > 0
    masked_dist = jnp.where(mask, dist, 999)
    flat_idx = jnp.argmin(masked_dist.reshape(-1))
    row = flat_idx // 5
    col = flat_idx % 5
    found = jnp.any(mask)
    return row - 2, col - 2, found


def _facing_matches(obs, desired):
    up = obs[2, 2, SELF_FACE_UP] > 0
    down = obs[2, 2, SELF_FACE_DOWN] > 0
    right = obs[2, 2, SELF_FACE_RIGHT] > 0
    left = obs[2, 2, SELF_FACE_LEFT] > 0
    return jnp.where(desired == UP, up,
           jnp.where(desired == DOWN, down,
           jnp.where(desired == RIGHT, right, left)))


def _go_interact(obs, dr, dc, found):
    """Navigate toward / turn to face / interact with a target offset
    (dr, dc) from self. INTERACT is only emitted once the observed facing
    one-hot actually matches the direction toward the target; otherwise the
    agent turns (if adjacent) or steps closer (row axis first, then column
    axis, which is always collision-free within either room here)."""
    adjacent = (jnp.abs(dr) + jnp.abs(dc)) == 1
    desired = jnp.where(dc == 1, RIGHT,
               jnp.where(dr == 1, DOWN,
               jnp.where(dc == -1, LEFT, UP)))
    facing_ok = _facing_matches(obs, desired)
    step_action = jnp.where(dr != 0, jnp.where(dr > 0, DOWN, UP),
                             jnp.where(dc > 0, RIGHT, LEFT))
    action = jnp.where(adjacent & facing_ok, INTERACT,
              jnp.where(adjacent, desired, step_action))
    action = jnp.where(found, action, STAY)
    return action


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    # agent_0's room (cols 1-2) is structurally cut off from the pot,
    # plates, ingredient supplies 0/1 and delivery station (col 3 is a
    # solid wall the full board height); it can never affect team reward
    # here regardless of the partner policy, so STAY is payoff-neutral
    # and safe.
    action = jnp.asarray(STAY, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


def init_agent_1():
    # memory: target ingredient id to cook next (0 or 1).
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    target = jnp.where(reset, jax.random.bernoulli(rng).astype(jnp.int32), memory)

    self_plate = obs[2, 2, SELF_PLATE] > 0
    self_cooked = obs[2, 2, SELF_COOKED] > 0
    self_ing0 = obs[2, 2, SELF_ING0] > 0
    self_ing1 = obs[2, 2, SELF_ING1] > 0
    holding_raw = self_ing0 | self_ing1
    holding_dish = self_plate & self_cooked
    holding_empty_plate = self_plate & (~self_cooked)

    pot_dr, pot_dc, pot_found = _nearest(obs[:, :, CH_POT])
    pot_row = jnp.clip(2 + pot_dr, 0, 4)
    pot_col = jnp.clip(2 + pot_dc, 0, 4)
    pot_cooked_flag = obs[pot_row, pot_col, GRID_COOKED] > 0
    pot_timer = obs[pot_row, pot_col, POT_TIMER]
    pot_ing_target = jnp.where(target == 0, obs[pot_row, pot_col, GRID_ING0],
                                obs[pot_row, pot_col, GRID_ING1])

    goal_dr, goal_dc, goal_found = _nearest(obs[:, :, CH_GOAL])
    plate_dr, plate_dc, plate_found = _nearest(obs[:, :, CH_PLATE_PILE])
    ing0_dr, ing0_dc, ing0_found = _nearest(obs[:, :, CH_ING0_SUPPLY])
    ing1_dr, ing1_dc, ing1_found = _nearest(obs[:, :, CH_ING1_SUPPLY])
    ing_dr = jnp.where(target == 0, ing0_dr, ing1_dr)
    ing_dc = jnp.where(target == 0, ing0_dc, ing1_dc)
    ing_found = jnp.where(target == 0, ing0_found, ing1_found)

    need_more = pot_ing_target < 3
    pot_idle_with_room = need_more & (pot_timer == 0) & (~pot_cooked_flag)

    # Empty-hands sub-target: collect a ready dish (plate), pre-fetch a
    # plate while the pot cooks, or top up the pot with the target
    # ingredient.
    empty_target_dr = jnp.where(pot_cooked_flag, plate_dr,
                        jnp.where(pot_idle_with_room, ing_dr, plate_dr))
    empty_target_dc = jnp.where(pot_cooked_flag, plate_dc,
                        jnp.where(pot_idle_with_room, ing_dc, plate_dc))
    empty_target_found = jnp.where(pot_cooked_flag, plate_found,
                          jnp.where(pot_idle_with_room, ing_found, plate_found))

    dr = jnp.where(holding_dish, goal_dr,
          jnp.where(holding_empty_plate, pot_dr,
          jnp.where(holding_raw, pot_dr,
          empty_target_dr)))
    dc = jnp.where(holding_dish, goal_dc,
          jnp.where(holding_empty_plate, pot_dc,
          jnp.where(holding_raw, pot_dc,
          empty_target_dc)))
    found = jnp.where(holding_dish, goal_found,
             jnp.where(holding_empty_plate, pot_found,
             jnp.where(holding_raw, pot_found,
             empty_target_found)))

    action = _go_interact(obs, dr, dc, found)

    # Flip the targeted ingredient exactly when we commit to a delivery
    # interaction (we never observe whether that delivery was correct).
    delivering_now = holding_dish & (action == INTERACT)
    next_target = jnp.where(delivering_now, 1 - target, target)
    next_memory = jnp.where(reset, jax.random.bernoulli(rng).astype(jnp.int32), next_target)

    action = jnp.asarray(action, dtype=jnp.int32)
    return action, next_memory, {}
