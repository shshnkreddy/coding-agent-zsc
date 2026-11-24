"""Grounded button-signal cooperative policy for grounded_coord_simple.

Map facts (fixed layout, verified against specification.json):
  - Two rooms are fully disconnected; no item can ever cross between them.
  - agent_0's room only borders the always-on recipe indicator R (2,0) and
    the button indicator L (2,3). agent_0 can never reach the pot, plates,
    ingredients 0/1, or delivery.
  - agent_1's room borders the pot P (2,4), both plate piles B, ingredient
    supplies 0 and 1, and the delivery station X. agent_1 can never see R.
  - The button L is the only channel by which agent_1 can learn the recipe;
    it is a literal, environment-defined display of the true recipe, not an
    arbitrary private code, so it is the grounded choice for both roles.

Both act functions are purely reactive: they relocate the target object
within the local 5x5 view every step and only track a small amount of
persistent memory to decide when to act (agent_0: last broadcast recipe;
agent_1: last confirmed recipe, the ingredient type committed to the pot
batch currently being assembled, and a rotating discard-counter index).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

# Static channel indices (from specification.json static_channel_order).
CH_WALL = 20
CH_GOAL = 21
CH_POT = 22
CH_RECIPE_IND = 23
CH_BUTTON_IND = 24
CH_PLATE_PILE = 25
CH_ING_SUPPLY0 = 26  # ingredient supplies are 26, 27, 28

# Self inventory channels.
CH_SELF_PLATE = 5
CH_SELF_COOKED = 6
CH_SELF_ING0 = 7  # 7, 8, 9

# Recipe/display block (5 wide): plate flag, cooked flag, ing0, ing1, ing2.
CH_RECIPE_ING0 = 36  # 36, 37, 38

# Grid-item block (pot/counter contents): plate flag, cooked flag, ing0..2.
CH_ITEM_COOKED = 30
CH_ITEM_ING0 = 31  # 31, 32, 33
CH_POT_TIMER = 39

# Facing channels: self-facing UP=1, DOWN=2, RIGHT=3, LEFT=4.
CH_FACE_UP = 1
CH_FACE_DOWN = 2
CH_FACE_RIGHT = 3
CH_FACE_LEFT = 4


def _find_object(channel_grid):
    """Locate a static/dynamic marker within the 5x5 local view.

    Returns local (row, col, found) for the argmax cell; found is False
    when every cell is zero (object absent from the current view).
    """
    flat = channel_grid.reshape(-1)
    idx = jnp.argmax(flat)
    found = flat[idx] > 0
    li = idx // 5
    lj = idx % 5
    return li, lj, found


def _face_for(target_dr, target_dc):
    """Facing action/channel to look at an adjacent cell offset by (dr, dc)."""
    face_action = jnp.where(
        target_dr > 0, DOWN, jnp.where(target_dr < 0, UP, jnp.where(target_dc > 0, RIGHT, LEFT))
    )
    facing_channel = jnp.where(
        target_dr > 0,
        CH_FACE_DOWN,
        jnp.where(target_dr < 0, CH_FACE_UP, jnp.where(target_dc > 0, CH_FACE_RIGHT, CH_FACE_LEFT)),
    )
    return face_action, facing_channel


def _nav_and_interact(obs, channel, target_dr, target_dc, face_action, facing_channel, do_interact):
    """Greedy Manhattan navigation to stand at offset (target_dr, target_dc)
    relative to the located `channel` marker, then face `face_action` and
    interact. `channel` is a landmark used purely for localization: the
    thing actually interacted with may be a different, geometrically fixed
    neighbor of that landmark (e.g. a counter next to the pot).
    Self is always at local (2, 2).
    """
    grid = jnp.take(obs, channel, axis=2)
    li, lj, found = _find_object(grid)
    delta_row = li - 2
    delta_col = lj - 2

    row_diff = delta_row - target_dr
    col_diff = delta_col - target_dc
    row_ok = row_diff == 0
    at_target = row_ok & (col_diff == 0)

    move_down = row_diff > 0
    move_up = row_diff < 0
    move_right = col_diff > 0
    move_left = col_diff < 0

    row_action = jnp.where(move_down, DOWN, UP)
    col_action = jnp.where(move_left, LEFT, jnp.where(move_right, RIGHT, STAY))
    nav_action = jnp.where(~row_ok, row_action, col_action)

    is_facing = obs[2, 2, facing_channel] > 0
    aligned_action = jnp.where(
        is_facing,
        jnp.where(do_interact, INTERACT, STAY),
        face_action,
    )
    action = jnp.where(at_target, aligned_action, nav_action)
    action = jnp.where(found, action, STAY)
    return action.astype(jnp.int32), (at_target & found), is_facing


def init_agent_0():
    # Sentinel -1 forces an initial broadcast the first time R is read.
    return {'last_broadcast': jnp.asarray([-1, -1, -1], dtype=jnp.int32)}


def agent_0_act(obs, reset, memory, rng):
    del rng
    r_li, r_lj, r_found = _find_object(obs[:, :, CH_RECIPE_IND])
    recipe = obs[r_li, r_lj, CH_RECIPE_ING0:CH_RECIPE_ING0 + 3].astype(jnp.int32)
    recipe = jnp.where(r_found, recipe, jnp.zeros(3, dtype=jnp.int32))

    changed = jnp.any(recipe != memory['last_broadcast'])

    face_action, facing_channel = _face_for(0, 1)
    action, at_target, is_facing = _nav_and_interact(
        obs, CH_BUTTON_IND, 0, 1, face_action, facing_channel, changed
    )

    will_press = at_target & is_facing & changed
    new_last = jnp.where(will_press, recipe, memory['last_broadcast'])
    new_memory = {'last_broadcast': new_last}

    init_mem = init_agent_0()
    new_memory = jax.tree_util.tree_map(
        lambda r, n: jnp.where(reset, r, n), init_mem, new_memory
    )
    return action, new_memory, {}


FALLBACK_GUESS_DELAY = 60  # steps to wait for a button signal before guessing


def init_agent_1():
    return {
        'known_recipe': jnp.zeros(3, dtype=jnp.int32),
        'committed_target': jnp.asarray(-1, dtype=jnp.int32),
        'discard_idx': jnp.asarray(0, dtype=jnp.int32),
        'step_count': jnp.asarray(0, dtype=jnp.int32),
        'guess_id': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    held_ing_vec = obs[2, 2, CH_SELF_ING0:CH_SELF_ING0 + 3]
    held_plate = obs[2, 2, CH_SELF_PLATE] > 0
    held_cooked = obs[2, 2, CH_SELF_COOKED] > 0
    held_ing_total = jnp.sum(held_ing_vec)

    holding_dish = held_plate & held_cooked
    holding_plate_only = held_plate & (~held_cooked)
    holding_ingredient = held_ing_total > 0
    held_type = jnp.argmax(held_ing_vec).astype(jnp.int32)

    p_li, p_lj, p_found = _find_object(obs[:, :, CH_POT])
    pot_cooked = (obs[p_li, p_lj, CH_ITEM_COOKED] > 0) & p_found
    pot_timer = jnp.where(p_found, obs[p_li, p_lj, CH_POT_TIMER], 0)
    pot_count = jnp.where(
        p_found, jnp.sum(obs[p_li, p_lj, CH_ITEM_ING0:CH_ITEM_ING0 + 3]), 0
    )
    pot_cooking = (~pot_cooked) & (pot_timer > 0)
    pot_idle_open = (~pot_cooked) & (pot_timer == 0) & (pot_count < 3)
    pot_empty = pot_idle_open & (pot_count == 0)

    l_li, l_lj, l_found = _find_object(obs[:, :, CH_BUTTON_IND])
    displayed = obs[l_li, l_lj, CH_RECIPE_ING0:CH_RECIPE_ING0 + 3].astype(jnp.int32)
    button_active = l_found & (jnp.sum(displayed) > 0)
    known_recipe = jnp.where(button_active, displayed, memory['known_recipe'])
    recipe_valid = jnp.sum(known_recipe) > 0
    target_id_live = jnp.where(known_recipe[0] > 0, 0, 1).astype(jnp.int32)

    # If no button signal has ever arrived after a generous grace period
    # (e.g. an independently generated agent_0 partner that never presses
    # the button, or is very slow to find it), fall back to a deterministic
    # alternating guess instead of idling for the whole episode. A wrong
    # guess never triggers a recipe resample, so the very next alternated
    # guess is then guaranteed correct; in steady state this alternation
    # nets a positive expected return (~2/3 hit rate) with no observation
    # of correctness required. Any real button confirmation always takes
    # priority over the guess, and switches are only ever locked in at a
    # batch boundary (pot empty), so this cannot corrupt a pot batch.
    step_count = memory['step_count']
    fallback_active = (~recipe_valid) & (step_count > FALLBACK_GUESS_DELAY)
    have_target_signal = recipe_valid | fallback_active
    target_id_choice = jnp.where(recipe_valid, target_id_live, memory['guess_id'])

    # Lock the ingredient target for an entire pot batch at the moment the
    # batch starts (pot empty), so a recipe change announced mid-fetch can
    # never mix two ingredient types into the same pot. Re-derive freely
    # while the pot is still empty (nothing committed yet); freeze once the
    # first ingredient lands.
    committed_target = jnp.where(
        pot_empty,
        jnp.where(have_target_signal, target_id_choice, memory['committed_target']),
        memory['committed_target'],
    )
    has_target = committed_target >= 0

    match = holding_ingredient & has_target & (held_type == committed_target) & pot_idle_open

    cond0 = holding_dish
    cond1 = (~cond0) & holding_plate_only
    cond2a = (~cond0) & (~cond1) & holding_ingredient & match
    cond2b = (~cond0) & (~cond1) & holding_ingredient & (~match)
    not_holding = (~cond0) & (~cond1) & (~cond2a) & (~cond2b)
    cond3 = not_holding & (pot_cooked | pot_cooking)
    cond4 = not_holding & (~cond3) & pot_idle_open & has_target
    # else: idle-watch at the pot (also keeps the button in view from there).

    channel = CH_POT
    channel = jnp.where(cond3, CH_PLATE_PILE, channel)
    channel = jnp.where(cond4, CH_ING_SUPPLY0 + committed_target, channel)
    channel = jnp.where(cond2a, CH_POT, channel)
    channel = jnp.where(cond2b, CH_POT, channel)
    channel = jnp.where(cond1, CH_POT, channel)
    channel = jnp.where(cond0, CH_GOAL, channel)

    target_dc_simple = jnp.asarray(-1, dtype=jnp.int32)
    target_dc_simple = jnp.where(cond3, -1, target_dc_simple)
    target_dc_simple = jnp.where(cond4, 1, target_dc_simple)
    target_dc_simple = jnp.where(cond0, 1, target_dc_simple)

    # Discard slots: stand next to the pot (always visible) at a fixed
    # offset, then face a neighboring wall/counter cell and place the held
    # item there. Rotating across 4 slots avoids ever reusing an
    # already-occupied counter within the same episode after repeated
    # discards.
    discard_dr_opts = jnp.asarray([1, 1, -1, -1], dtype=jnp.int32)
    discard_dc_opts = jnp.asarray([-1, -2, -1, -2], dtype=jnp.int32)
    discard_face_opts = jnp.asarray([UP, UP, DOWN, DOWN], dtype=jnp.int32)
    discard_face_ch_opts = jnp.asarray(
        [CH_FACE_UP, CH_FACE_UP, CH_FACE_DOWN, CH_FACE_DOWN], dtype=jnp.int32
    )
    slot = memory['discard_idx'] % 4
    discard_dr = discard_dr_opts[slot]
    discard_dc = discard_dc_opts[slot]
    discard_face = discard_face_opts[slot]
    discard_face_ch = discard_face_ch_opts[slot]

    target_dr = jnp.where(cond2b, discard_dr, jnp.asarray(0, dtype=jnp.int32))
    target_dc = jnp.where(cond2b, discard_dc, target_dc_simple)

    simple_face_action, simple_facing_channel = _face_for(jnp.asarray(0, dtype=jnp.int32), target_dc_simple)
    face_action = jnp.where(cond2b, discard_face, simple_face_action)
    facing_channel = jnp.where(cond2b, discard_face_ch, simple_facing_channel)

    action, at_target, is_facing = _nav_and_interact(
        obs, channel, target_dr, target_dc, face_action, facing_channel, True
    )

    about_to_discard = cond2b & at_target & is_facing
    new_discard_idx = jnp.where(about_to_discard, (memory['discard_idx'] + 1) % 4, memory['discard_idx'])

    about_to_deliver_on_guess = cond0 & at_target & is_facing & (~recipe_valid)
    new_guess_id = jnp.where(about_to_deliver_on_guess, 1 - memory['guess_id'], memory['guess_id'])

    new_memory = {
        'known_recipe': known_recipe,
        'committed_target': committed_target,
        'discard_idx': new_discard_idx,
        'step_count': jnp.minimum(step_count + 1, FALLBACK_GUESS_DELAY + 1),
        'guess_id': new_guess_id,
    }
    init_mem = init_agent_1()
    new_memory = jax.tree_util.tree_map(
        lambda r, n: jnp.where(reset, r, n), init_mem, new_memory
    )
    return action, new_memory, {}
