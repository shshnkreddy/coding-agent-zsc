"""Button-broadcast coordination policy for grounded_coord_simple.

Map facts (fixed, from context/environment): agent_0's room (rows 1-3,
cols 1-2) only reaches the always-on recipe indicator R (2,0) and the
button indicator L (2,3); it can never reach the pot, plates, ingredient
supplies 0/1, or delivery. agent_1's room (rows 1-3, cols 5-6) reaches the
pot (2,4), both plate piles (1,4)/(3,4), ingredient supplies 0 (1,7) and 1
(3,7), and delivery (2,7); it can never reach R or L, and the only usable
ingredient in its own reachable pile (ingredient 2) never appears in any
recipe. The two rooms never share a view of each other. The button L is
therefore the only in-game channel for agent_1 to learn the target recipe,
and it is visible to agent_1 whenever it stands in column 5 (which is also
where it deposits ingredients and collects plates). agent_0 broadcasts the
recipe it reads from R by pressing L whenever the recipe changes (plus a
periodic safety refresh); agent_1 waits for a valid reading before ever
committing ingredients, and pauses briefly near the pot after each delivery
to catch a possible post-delivery update before trusting stale memory.
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

# Channel indices (see context/environment/specification.json).
CH_SELF_DIR_START = 1              # UP, DOWN, RIGHT, LEFT occupy [1, 5)
CH_SELF_DIR_STOP = 5
CH_SELF_PLATE = 5
CH_SELF_COOKED = 6
CH_SELF_ING0 = 7
CH_SELF_ING1 = 8
CH_STATIC_POT = 22
CH_STATIC_RECIPE_IND = 23
CH_STATIC_BUTTON_IND = 24
CH_ITEM_COOKED = 30
CH_ITEM_ING0 = 31
CH_ITEM_ING1 = 32
CH_RECIPE_ING0 = 36
CH_RECIPE_ING1 = 37
CH_POT_TIMER = 39

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# Facing-channel offset (within [1, 5)) required to face each movement
# direction: RIGHT->2, DOWN->1, LEFT->3, UP->0.
_FACING_LOOKUP = (2, 1, 3, 0)


def _facing_lookup():
    return jnp.array(_FACING_LOOKUP, dtype=jnp.int32)


def _locate_flat(obs, channel):
    flat = obs[:, :, channel].reshape(-1)
    idx = jnp.argmax(flat)
    row = (idx // 5).astype(jnp.int32)
    col = (idx % 5).astype(jnp.int32)
    found = jnp.max(flat) > 0
    return row, col, found


def _navigate(self_r, self_c, target_r, target_c, desired_action, facing_onehot):
    """Greedy row-then-column walk inside an obstacle-free 3x2 room, then
    face the desired direction and interact once aligned."""
    needed_idx = _facing_lookup()[desired_action]
    facing_ok = facing_onehot[needed_idx] == 1
    at_target = (self_r == target_r) & (self_c == target_c)
    move_action = jnp.where(
        self_r < target_r, DOWN,
        jnp.where(
            self_r > target_r, UP,
            jnp.where(
                self_c < target_c, RIGHT,
                jnp.where(self_c > target_c, LEFT, desired_action),
            ),
        ),
    )
    action = jnp.where(at_target & facing_ok, INTERACT, move_action)
    return action.astype(jnp.int32)


def init_agent_0():
    return {
        'last_sent': jnp.asarray(-1, dtype=jnp.int32),
        'steps_since_press': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    facing_onehot = obs[2, 2, CH_SELF_DIR_START:CH_SELF_DIR_STOP]

    # Self-localize via the button cell, which is always in view from
    # anywhere in agent_0's room (button at global (2, 3)).
    lr, lc, _ = _locate_flat(obs, CH_STATIC_BUTTON_IND)
    self_r = 4 - lr
    self_c = 5 - lc

    # The always-on recipe indicator R is always in view from this room.
    rr, rc, _ = _locate_flat(obs, CH_STATIC_RECIPE_IND)
    ing0 = obs[rr, rc, CH_RECIPE_ING0]
    ing1 = obs[rr, rc, CH_RECIPE_ING1]
    current_recipe = jnp.where(ing0 == 3, 0, jnp.where(ing1 == 3, 1, -1)).astype(jnp.int32)

    last_sent = memory['last_sent']
    steps_since_press = memory['steps_since_press']

    refresh_interval = 200
    should_press = (current_recipe >= 0) & (
        (current_recipe != last_sent) | (steps_since_press >= refresh_interval)
    )

    nav_action = _navigate(self_r, self_c, 2, 2, RIGHT, facing_onehot)
    would_interact = nav_action == INTERACT
    action = jnp.where(should_press, nav_action, jnp.where(would_interact, STAY, nav_action))

    pressed = should_press & would_interact
    new_last_sent = jnp.where(pressed, current_recipe, last_sent)
    new_steps = jnp.where(pressed, 0, steps_since_press + 1)

    new_memory = {
        'last_sent': jnp.where(reset, jnp.asarray(-1, jnp.int32), new_last_sent).astype(jnp.int32),
        'steps_since_press': jnp.where(reset, jnp.asarray(0, jnp.int32), new_steps).astype(jnp.int32),
    }
    return action.astype(jnp.int32), new_memory, {}


def init_agent_1():
    return {
        'known_ingredient': jnp.asarray(-1, dtype=jnp.int32),
        'sync_wait': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    facing_onehot = obs[2, 2, CH_SELF_DIR_START:CH_SELF_DIR_STOP]
    inv_plate = obs[2, 2, CH_SELF_PLATE]
    inv_cooked = obs[2, 2, CH_SELF_COOKED]
    inv_ing0 = obs[2, 2, CH_SELF_ING0]
    inv_ing1 = obs[2, 2, CH_SELF_ING1]

    # Self-localize via the pot cell, always in view from anywhere in
    # agent_1's room (pot at global (2, 4)).
    lr, lc, _ = _locate_flat(obs, CH_STATIC_POT)
    self_r = 4 - lr
    self_c = 6 - lc

    pot_ing0 = obs[lr, lc, CH_ITEM_ING0]
    pot_ing1 = obs[lr, lc, CH_ITEM_ING1]
    pot_cooked = obs[lr, lc, CH_ITEM_COOKED]
    pot_timer = obs[lr, lc, CH_POT_TIMER]

    # The button L is only visible from column 5, which agent_1 passes
    # through whenever it deposits ingredients or collects plates.
    br, bc, found = _locate_flat(obs, CH_STATIC_BUTTON_IND)
    b_ing0 = obs[br, bc, CH_RECIPE_ING0]
    b_ing1 = obs[br, bc, CH_RECIPE_ING1]
    reading = jnp.where(found & (b_ing0 == 3), 0, jnp.where(found & (b_ing1 == 3), 1, -1)).astype(jnp.int32)

    known = memory['known_ingredient']
    sync_wait = memory['sync_wait']
    effective_known = jnp.where(reading >= 0, reading, known)

    nav_pot = _navigate(self_r, self_c, 2, 5, LEFT, facing_onehot)
    nav_deliver = _navigate(self_r, self_c, 2, 6, RIGHT, facing_onehot)
    plate_row = jnp.where(self_r <= 2, 1, 3)
    nav_plate = _navigate(self_r, self_c, plate_row, 5, LEFT, facing_onehot)
    ing_row = jnp.where(effective_known == 0, 1, 3)
    nav_ing = _navigate(self_r, self_c, ing_row, 6, RIGHT, facing_onehot)

    holding_dish = (inv_plate == 1) & (inv_cooked == 1)
    holding_empty_plate = (inv_plate == 1) & (inv_cooked == 0)
    holding_ingredient = (inv_ing0 > 0) | (inv_ing1 > 0)
    holding_nothing = (inv_plate == 0) & (inv_cooked == 0) & (inv_ing0 == 0) & (inv_ing1 == 0)

    pot_target_count = jnp.where(effective_known == 0, pot_ing0, pot_ing1)
    pot_idle_fillable = (pot_timer == 0) & (pot_cooked == 0) & (pot_target_count < 3)
    in_sync_wait = sync_wait > 0

    action = nav_pot
    action = jnp.where(holding_nothing & (effective_known < 0), nav_pot, action)
    action = jnp.where(holding_nothing & in_sync_wait, nav_pot, action)
    action = jnp.where(
        holding_nothing & (effective_known >= 0) & (~in_sync_wait) & pot_idle_fillable,
        nav_ing, action,
    )
    pot_full_or_cooking = (~pot_idle_fillable) & (effective_known >= 0)
    action = jnp.where(
        holding_nothing & (~in_sync_wait) & pot_full_or_cooking,
        nav_plate, action,
    )
    action = jnp.where(holding_ingredient, nav_pot, action)
    action = jnp.where(holding_empty_plate, nav_pot, action)
    action = jnp.where(holding_dish, nav_deliver, action)
    action = action.astype(jnp.int32)

    delivered_now = holding_dish & (action == INTERACT)
    grace = 6
    new_sync_wait = jnp.where(
        delivered_now, grace,
        jnp.where(
            in_sync_wait & (reading >= 0), 0,
            jnp.where(in_sync_wait, sync_wait - 1, sync_wait),
        ),
    )

    new_memory = {
        'known_ingredient': jnp.where(reset, jnp.asarray(-1, jnp.int32), effective_known).astype(jnp.int32),
        'sync_wait': jnp.where(reset, jnp.asarray(0, jnp.int32), new_sync_wait).astype(jnp.int32),
    }
    return action, new_memory, {}
