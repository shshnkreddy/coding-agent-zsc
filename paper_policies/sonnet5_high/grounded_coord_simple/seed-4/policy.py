"""Grounded button-relay coordination policy for grounded_coord_simple.

Layout facts baked in (see context/environment/README.md, layout.md):
  - Two rooms are physically disconnected; agents never share floor cells.
  - agent_0's room only reaches the always-on recipe indicator R (global
    (2,0)) and the button indicator L (global (2,3)); it can never reach the
    pot, plates, ingredient 0/1 piles, or the delivery station.
  - agent_1's room reaches the pot (2,4), both plate piles (1,4)/(3,4),
    ingredient piles 0 (1,7) and 1 (3,7), and delivery (2,7); it can only see
    the button cell when standing in column 5 of its room, and never sees R.
  - Only recipes [3,0,0] and [0,3,0] (ingredient 0 or ingredient 1) occur;
    ingredient 2 is a decoy agent_0 can reach but which is never useful.

Because of this fixed geometry, the only viable communication channel for any
competent partner is: agent_0 relays the true recipe (read from R) through
button activations, and agent_1 reads the button's displayed recipe and
remembers it. Both act functions rely only on this physically-grounded,
non-arbitrary signal (the button literally displays true ingredient counts),
so an independently generated partner that also uses the button in the
obviously-intended way should interoperate without a private convention.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# Facing ids as encoded in the observation channels (README: UP=0 DOWN=1
# RIGHT=2 LEFT=3), distinct from the action ids above.
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3
# Movement action that reorients a player to face a given facing id.
_ACT_FROM_FACING = (UP, DOWN, RIGHT, LEFT)

# Static-object channel indices.
CH_WALL, CH_GOAL, CH_POT, CH_RECIPE_IND, CH_BUTTON_IND, CH_PLATE_PILE = 20, 21, 22, 23, 24, 25
# Self inventory channels.
CH_SELF_PLATE, CH_SELF_COOKED = 5, 6
CH_SELF_ING0, CH_SELF_ING1, CH_SELF_ING2 = 7, 8, 9
# Self facing channels (order UP, DOWN, RIGHT, LEFT).
CH_SELF_FACE0 = 1
# Grid-item (counter/pot) channels.
CH_ITEM_PLATE, CH_ITEM_COOKED = 29, 30
CH_ITEM_ING0, CH_ITEM_ING1, CH_ITEM_ING2 = 31, 32, 33
CH_POT_TIMER = 39
# Displayed-recipe channels (valid at R and active L cells).
CH_REC_ING0, CH_REC_ING1 = 36, 37

HEARTBEAT_STEPS = 80


def _argmax_rc(mask):
    flat = jnp.argmax(mask.reshape(-1))
    i = flat // 5
    j = flat % 5
    return i, j


def _self_facing_id(obs):
    face = obs[2, 2, CH_SELF_FACE0:CH_SELF_FACE0 + 4]
    return jnp.argmax(face).astype(jnp.int32)


def _nav_and_act(self_row, self_col, facing_id, tr, tc, desired_facing, arrived_action):
    row_diff = tr - self_row
    col_diff = tc - self_col
    at_target = jnp.logical_and(row_diff == 0, col_diff == 0)
    move_action = jnp.where(
        row_diff > 0, DOWN,
        jnp.where(row_diff < 0, UP,
                  jnp.where(col_diff > 0, RIGHT, LEFT)),
    )
    facing_ok = facing_id == desired_facing
    reorient_action = jnp.asarray(_ACT_FROM_FACING, dtype=jnp.int32)[desired_facing]
    at_target_action = jnp.where(facing_ok, arrived_action, reorient_action)
    return jnp.where(at_target, at_target_action, move_action).astype(jnp.int32)


# ---------------------------------------------------------------------------
# agent_0: sees the true recipe on R at all times; relays it via the button.
# ---------------------------------------------------------------------------

def init_agent_0():
    # [last_seen_ing0, last_seen_ing1, steps_since_change, steps_since_broadcast]
    return jnp.asarray([-1, -1, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    init = jnp.asarray([-1, -1, 0, 0], dtype=jnp.int32)
    mem = jnp.where(reset, init, memory)

    recipe_mask = obs[:, :, CH_RECIPE_IND].astype(jnp.int32)
    ing0 = jnp.sum(recipe_mask * obs[:, :, CH_REC_ING0])
    ing1 = jnp.sum(recipe_mask * obs[:, :, CH_REC_ING1])

    i, j = _argmax_rc(recipe_mask)
    # R is at global (2, 0): self_row + i - 2 = 2, self_col + j - 2 = 0.
    self_row = 4 - i
    self_col = 2 - j
    facing_id = _self_facing_id(obs)

    recipe_changed = jnp.logical_or(ing0 != mem[0], ing1 != mem[1])
    steps_since_change = jnp.where(recipe_changed, 0, mem[2] + 1)
    # Press immediately on a recipe change, once more shortly after (in case
    # the partner wasn't looking during the first activation window), and
    # otherwise re-press on a long-period heartbeat as a robustness backstop
    # against a partner that only samples the button rarely.
    should_broadcast = jnp.logical_or(
        jnp.logical_or(steps_since_change == 0, steps_since_change == 12),
        mem[3] >= HEARTBEAT_STEPS,
    )
    arrived_action = jnp.where(should_broadcast, INTERACT, STAY).astype(jnp.int32)

    action = _nav_and_act(self_row, self_col, facing_id, 2, 2, F_RIGHT, arrived_action)

    did_broadcast = action == INTERACT
    new_mem3 = jnp.where(did_broadcast, 0, mem[3] + 1)
    new_mem = jnp.stack([ing0, ing1, steps_since_change, new_mem3]).astype(jnp.int32)

    trace = {
        'ing0': ing0.astype(jnp.int32),
        'ing1': ing1.astype(jnp.int32),
        'did_broadcast': did_broadcast.astype(jnp.int32),
    }
    return action, new_mem, trace


# ---------------------------------------------------------------------------
# agent_1: never sees R directly; reads the button (visible only from column
# 5 of its room) and remembers the target ingredient, then executes the full
# fetch -> cook -> plate -> deliver pipeline.
# ---------------------------------------------------------------------------

POST_DELIVERY_WAIT_STEPS = 6


def init_agent_1():
    # [known_target_ingredient (-1 = unknown), drop_slot (round-robin 0..3),
    #  steps_since_own_delivery (large sentinel when not awaiting)].
    return jnp.asarray([-1, 0, POST_DELIVERY_WAIT_STEPS], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    init = jnp.asarray([-1, 0, POST_DELIVERY_WAIT_STEPS], dtype=jnp.int32)
    mem = jnp.where(reset, init, memory)

    pot_mask = obs[:, :, CH_POT].astype(jnp.int32)
    i, j = _argmax_rc(pot_mask)
    # Pot is at global (2, 4): self_row + i - 2 = 2, self_col + j - 2 = 4.
    self_row = 4 - i
    self_col = 6 - j
    facing_id = _self_facing_id(obs)

    button_mask = obs[:, :, CH_BUTTON_IND].astype(jnp.int32)
    button_visible = jnp.sum(button_mask) > 0
    button_ing0 = jnp.sum(button_mask * obs[:, :, CH_REC_ING0])
    button_ing1 = jnp.sum(button_mask * obs[:, :, CH_REC_ING1])
    button_active = jnp.logical_or(button_ing0 > 0, button_ing1 > 0)
    saw_recipe = jnp.logical_and(button_visible, button_active)
    observed_target = jnp.where(button_ing0 > 0, 0, 1).astype(jnp.int32)
    known_target = jnp.where(saw_recipe, observed_target, mem[0])
    # A fresh button read is always current (the button shows the live
    # recipe every step it is active, not a value frozen at activation).
    # Right after our own delivery the recipe may have resampled, so hold
    # off committing to a fetch for a short, bounded window -- long enough
    # for the partner to notice and relay a change, but short enough not to
    # waste time when the recipe happened to repeat (which a change-only
    # relay never re-announces).
    steps_since_delivery = mem[2]
    awaiting_confirmation = jnp.logical_and(
        steps_since_delivery < POST_DELIVERY_WAIT_STEPS,
        jnp.logical_not(saw_recipe),
    )
    known = jnp.logical_and(known_target >= 0, jnp.logical_not(awaiting_confirmation))
    target_is_ing0 = known_target == 0

    inv_plate = obs[2, 2, CH_SELF_PLATE]
    inv_cooked = obs[2, 2, CH_SELF_COOKED]
    inv_ing0 = obs[2, 2, CH_SELF_ING0]
    inv_ing1 = obs[2, 2, CH_SELF_ING1]
    inv_ing2 = obs[2, 2, CH_SELF_ING2]
    holding_nothing = jnp.logical_and(
        jnp.logical_and(inv_plate == 0, inv_cooked == 0),
        jnp.logical_and(jnp.logical_and(inv_ing0 == 0, inv_ing1 == 0), inv_ing2 == 0),
    )
    holding_plated_dish = jnp.logical_and(inv_plate == 1, inv_cooked == 1)
    holding_empty_plate = jnp.logical_and(inv_plate == 1, inv_cooked == 0)
    holding_ing0 = inv_ing0 > 0
    holding_ing1 = inv_ing1 > 0
    holding_ing2 = inv_ing2 > 0
    holding_any_ing = jnp.logical_or(jnp.logical_or(holding_ing0, holding_ing1), holding_ing2)
    holding_target_ing = jnp.where(target_is_ing0, holding_ing0, holding_ing1)
    # An ingredient can go stale in hand if the recipe changes (button relay
    # arrives) between fetching it and depositing it; drop it on the nearest
    # counter instead of getting stuck holding a now-useless item forever.
    holding_correct_ing = jnp.logical_and(known, holding_target_ing)
    holding_wrong_ing = jnp.logical_and(holding_any_ing, jnp.logical_not(holding_correct_ing))

    pot_ing0 = jnp.sum(pot_mask * obs[:, :, CH_ITEM_ING0])
    pot_ing1 = jnp.sum(pot_mask * obs[:, :, CH_ITEM_ING1])
    pot_ing2 = jnp.sum(pot_mask * obs[:, :, CH_ITEM_ING2])
    pot_cooked = jnp.sum(pot_mask * obs[:, :, CH_ITEM_COOKED])
    pot_timer = jnp.sum(pot_mask * obs[:, :, CH_POT_TIMER])
    pot_total = pot_ing0 + pot_ing1 + pot_ing2

    pot_ready = pot_cooked > 0
    pot_cooking_not_ready = jnp.logical_and(pot_timer > 0, jnp.logical_not(pot_ready))
    pot_needs_fill = jnp.logical_and(pot_total < 3, jnp.logical_and(pot_cooked == 0, pot_timer == 0))

    # Round-robin across the four reachable empty counters for stale-item
    # drops so a single previously-occupied counter can't jam this forever.
    drop_slot = mem[1]
    drop_tr = jnp.where(drop_slot < 2, 1, 3)
    drop_tc = jnp.where(jnp.logical_or(drop_slot == 0, drop_slot == 2), 5, 6)
    drop_face = jnp.where(drop_slot < 2, F_UP, F_DOWN)

    c1 = holding_plated_dish
    c2 = jnp.logical_and(pot_ready, holding_empty_plate)
    c3 = jnp.logical_and(pot_ready, holding_nothing)
    c4 = jnp.logical_and(pot_cooking_not_ready, holding_empty_plate)
    c5 = jnp.logical_and(pot_cooking_not_ready, holding_nothing)
    c_drop = holding_wrong_ing
    c6 = jnp.logical_and(pot_needs_fill, jnp.logical_and(known, holding_target_ing))
    c7 = jnp.logical_and(pot_needs_fill, jnp.logical_and(known, holding_nothing))

    conds = [c1, c2, c3, c4, c5, c_drop, c6, c7]
    tr7 = jnp.where(target_is_ing0, 1, 3)
    tr_list = [2, 2, 1, 2, 1, drop_tr, 2, tr7]
    tc_list = [6, 5, 5, 5, 5, drop_tc, 5, 6]
    face_list = [F_RIGHT, F_LEFT, F_LEFT, F_LEFT, F_LEFT, drop_face, F_LEFT, F_RIGHT]
    act_list = [INTERACT, INTERACT, INTERACT, STAY, INTERACT, INTERACT, INTERACT, INTERACT]

    default_tr, default_tc, default_face, default_act = 2, 5, F_LEFT, STAY

    tr = jnp.select(conds, tr_list, default=default_tr)
    tc = jnp.select(conds, tc_list, default=default_tc)
    face = jnp.select(conds, face_list, default=default_face).astype(jnp.int32)
    arrived_act = jnp.select(conds, act_list, default=default_act).astype(jnp.int32)

    action = _nav_and_act(self_row, self_col, facing_id, tr, tc, face, arrived_act)

    did_drop = jnp.logical_and(c_drop, jnp.logical_and(self_row == tr, jnp.logical_and(self_col == tc, facing_id == face)))
    new_drop_slot = jnp.where(did_drop, (drop_slot + 1) % 4, drop_slot)
    did_deliver = jnp.logical_and(c1, action == INTERACT)
    new_steps_since_delivery = jnp.where(
        did_deliver, 0,
        jnp.where(saw_recipe, POST_DELIVERY_WAIT_STEPS,
                  jnp.minimum(steps_since_delivery + 1, POST_DELIVERY_WAIT_STEPS)),
    )
    new_mem = jnp.asarray([known_target, new_drop_slot, new_steps_since_delivery], dtype=jnp.int32)

    trace = {
        'known_target': known_target,
        'pot_total': pot_total.astype(jnp.int32),
        'pot_ready': pot_ready.astype(jnp.int32),
        'holding_wrong_ing': holding_wrong_ing.astype(jnp.int32),
        'awaiting_confirmation': awaiting_confirmation.astype(jnp.int32),
    }
    return action, new_mem, trace
