"""Reactive coordination policy for grounded_coord_simple.

Map facts baked into this logic (see context/environment/README.md):
- The board splits into two floor-disconnected rooms. agent_0's room
  (cols 1-2) only borders the always-on recipe indicator R, the (useless,
  never-requested) ingredient-2 piles, and the recipe button L. agent_1's
  room (cols 5-6) borders the pot, both real ingredient piles (0 and 1),
  both plate piles, and the delivery station -- agent_1 does all cooking.
- agent_0 can therefore never cook or deliver; its only lever is L, whose
  display always reflects the *current* recipe while active, regardless of
  who pressed it or when. agent_1 can only learn the recipe by seeing L,
  which is visible only when standing in column 5.
- Both rooms are small, open (no interior obstacles) rectangles, and each
  room holds exactly one agent, so greedy Manhattan navigation toward a
  target found in the local 5x5 crop never needs collision handling.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

ACTION_RIGHT = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_UP = 3
ACTION_STAY = 4
ACTION_INTERACT = 5

# obs one-hot facing order (channels 1..4): UP=0, DOWN=1, RIGHT=2, LEFT=3.
# Movement action IDs: RIGHT=0, DOWN=1, LEFT=2, UP=3.
_OBS_DIR_TO_ACTION_LIST = [3, 1, 0, 2]

HEARTBEAT_STEPS = 120
MAX_STEPS = 400
# A full fetch-cook-deliver cycle needs roughly 40+ steps; broadcasting a
# recipe with less runway left than that cannot lead to another delivery,
# so skip paying the activation cost that close to the episode end.
END_GUARD_STEPS = 40


def _locate(mask5x5):
    flat = mask5x5.reshape(-1).astype(jnp.int32)
    idx = jnp.argmax(flat)
    found = jnp.max(flat) > 0
    di = idx // 5 - 2
    dj = idx % 5 - 2
    return found, di, dj, idx


def _nav_step(found, di, dj, facing_obs, want_interact):
    """Greedily approach a target found at relative offset (di, dj); interact
    once adjacent and correctly facing, if requested."""
    adjacent = (jnp.abs(di) + jnp.abs(dj)) == 1
    desired_dir_adj = jnp.where(
        dj == 1, 2, jnp.where(dj == -1, 3, jnp.where(di == 1, 1, 0))
    )
    dir_if_vert = jnp.where(di > 0, 1, 0)
    dir_if_horiz = jnp.where(dj > 0, 2, 3)
    # Every interactable target in this room layout sits outside the agent's
    # own 2-wide room, so the final approach step is always horizontal.
    # Resolving the row first means that last horizontal step also leaves
    # the agent already facing the target, needing no extra turn-in-place.
    desired_dir_far = jnp.where(di != 0, dir_if_vert, dir_if_horiz)
    desired_dir = jnp.where(adjacent, desired_dir_adj, desired_dir_far)
    facing_match = facing_obs == desired_dir
    obs_dir_to_action = jnp.asarray(_OBS_DIR_TO_ACTION_LIST, dtype=jnp.int32)
    move_action = obs_dir_to_action[desired_dir]
    action_when_adjacent = jnp.where(want_interact, ACTION_INTERACT, ACTION_STAY)
    action = jnp.where(adjacent & facing_match, action_when_adjacent, move_action)
    action = jnp.where(found, action, ACTION_STAY)
    did_interact = found & adjacent & facing_match & want_interact
    return action.astype(jnp.int32), did_interact


def init_agent_0():
    return {
        'last_recipe': jnp.asarray(-1, dtype=jnp.int32),
        'steps_since_press': jnp.asarray(0, dtype=jnp.int32),
        'elapsed': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    eff_last = jnp.where(reset, jnp.asarray(-1, jnp.int32), memory['last_recipe'])
    eff_steps = jnp.where(reset, jnp.asarray(0, jnp.int32), memory['steps_since_press'])
    eff_elapsed = jnp.where(reset, jnp.asarray(0, jnp.int32), memory['elapsed'])

    facing_obs = jnp.argmax(obs[2, 2, 1:5])
    flat_obs = obs.reshape(25, 40)

    mask_r = obs[:, :, 23] > 0
    found_r, di_r, dj_r, idx_r = _locate(mask_r)
    ing0 = flat_obs[idx_r, 36]
    ing1 = flat_obs[idx_r, 37]
    current_recipe_raw = jnp.where(ing0 > 0, 0, jnp.where(ing1 > 0, 1, -1))
    current_recipe = jnp.where(found_r, current_recipe_raw, eff_last).astype(jnp.int32)

    changed = current_recipe != eff_last
    steps_inc = eff_steps + 1
    near_end = eff_elapsed >= (MAX_STEPS - END_GUARD_STEPS)
    want_press = (changed | (steps_inc >= HEARTBEAT_STEPS)) & (~near_end)

    mask_l = obs[:, :, 24] > 0
    found_l, di_l, dj_l, idx_l = _locate(mask_l)
    action, did_interact = _nav_step(found_l, di_l, dj_l, facing_obs, want_press)

    new_last = jnp.where(did_interact, current_recipe, eff_last)
    new_steps = jnp.where(did_interact, jnp.asarray(0, jnp.int32), steps_inc)

    next_memory = {
        'last_recipe': new_last.astype(jnp.int32),
        'steps_since_press': new_steps.astype(jnp.int32),
        'elapsed': (eff_elapsed + 1).astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, {}


CONFIRM_WINDOW_STEPS = 12


def init_agent_1():
    return {
        'known_recipe': jnp.asarray(-1, dtype=jnp.int32),
        'confirm_left': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    eff_known = jnp.where(reset, jnp.asarray(-1, jnp.int32), memory['known_recipe'])
    eff_confirm = jnp.where(reset, jnp.asarray(0, jnp.int32), memory['confirm_left'])

    facing_obs = jnp.argmax(obs[2, 2, 1:5])
    flat_obs = obs.reshape(25, 40)

    held_plate = obs[2, 2, 5]
    held_cooked = obs[2, 2, 6]
    held_ing_sum = obs[2, 2, 7] + obs[2, 2, 8] + obs[2, 2, 9]

    mask_pot = obs[:, :, 22] > 0
    found_pot, di_p, dj_p, idx_p = _locate(mask_pot)
    pot_cooked = flat_obs[idx_p, 30]
    pot_timer = flat_obs[idx_p, 39]
    pot_total = flat_obs[idx_p, 31] + flat_obs[idx_p, 32] + flat_obs[idx_p, 33]

    mask_goal = obs[:, :, 21] > 0
    found_goal, di_g, dj_g, idx_g = _locate(mask_goal)

    mask_plate = obs[:, :, 25] > 0
    found_plate, di_pl, dj_pl, idx_pl = _locate(mask_plate)

    mask_button = obs[:, :, 24] > 0
    found_button, di_b, dj_b, idx_b = _locate(mask_button)
    btn_ing0 = flat_obs[idx_b, 36]
    btn_ing1 = flat_obs[idx_b, 37]
    seen_recipe = jnp.where(btn_ing0 > 0, 0, jnp.where(btn_ing1 > 0, 1, -1))

    target_is_1 = eff_known == 1
    mask_ing_target = jnp.where(target_is_1, obs[:, :, 27] > 0, obs[:, :, 26] > 0)
    found_ing, di_i, dj_i, idx_i = _locate(mask_ing_target)

    cond_deliver = (held_plate > 0) & (held_cooked > 0)
    cond_scoop = (held_plate > 0) & (held_cooked == 0) & (held_ing_sum == 0)
    cond_insert = held_ing_sum > 0
    empty_handed = (held_plate == 0) & (held_cooked == 0) & (held_ing_sum == 0)

    # A correct delivery may resample the *same* recipe (50/50), which
    # agent_0 cannot distinguish from "no change" via the always-on
    # indicator alone. Rather than blocking on a fresh broadcast every
    # cycle, only demand one for a short bounded window right after each
    # delivery; if none arrives, fall back to the previous (still-valid,
    # in the repeat case) belief instead of stalling for a heartbeat.
    in_confirm = (eff_confirm > 0) | (eff_known == -1)

    pot_ready_for_more = found_pot & (pot_cooked == 0) & (pot_timer == 0) & (pot_total < 3)
    pot_cooking = found_pot & (pot_timer > 0)
    pot_ready_done = found_pot & (pot_cooked > 0)

    action_deliver, did_deliver_raw = _nav_step(found_goal, di_g, dj_g, facing_obs, jnp.asarray(True))
    action_scoop, _ = _nav_step(found_pot, di_p, dj_p, facing_obs, pot_ready_done)
    action_insert, _ = _nav_step(found_pot, di_p, dj_p, facing_obs, pot_ready_for_more)

    action_seek = jnp.where(found_button, ACTION_STAY, ACTION_LEFT)

    action_get_plate, _ = _nav_step(found_plate, di_pl, dj_pl, facing_obs, jnp.asarray(True))
    action_get_ing, _ = _nav_step(found_ing, di_i, dj_i, facing_obs, jnp.asarray(True))

    action_empty_know = jnp.where(
        pot_cooking | pot_ready_done, action_get_plate,
        jnp.where(pot_ready_for_more, action_get_ing, ACTION_STAY),
    )
    action_empty = jnp.where(in_confirm, action_seek, action_empty_know)

    action = jnp.where(
        cond_deliver, action_deliver,
        jnp.where(
            cond_scoop, action_scoop,
            jnp.where(cond_insert, action_insert, jnp.where(empty_handed, action_empty, ACTION_STAY)),
        ),
    )

    did_deliver = cond_deliver & did_deliver_raw
    fresh_read = empty_handed & in_confirm & found_button & (seen_recipe >= 0)
    checking_no_read = empty_handed & in_confirm & (~fresh_read)

    known_after_read = jnp.where(fresh_read, seen_recipe, eff_known)
    confirm_after_read = jnp.where(
        fresh_read, jnp.asarray(0, jnp.int32),
        jnp.where(checking_no_read, jnp.maximum(eff_confirm - 1, 0), eff_confirm),
    )

    next_known = jnp.where(did_deliver, eff_known, known_after_read)
    next_confirm = jnp.where(did_deliver, jnp.asarray(CONFIRM_WINDOW_STEPS, jnp.int32), confirm_after_read)

    next_memory = {
        'known_recipe': next_known.astype(jnp.int32),
        'confirm_left': next_confirm.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, {}
