"""Coordination policy for grounded_coord_simple (Overcooked V2).

Layout facts (fixed, from environment/README.md + layout.md):
  R  (always-on recipe indicator) at (2,0), only reachable/viewable by agent_0.
  L  (button recipe indicator)    at (2,3), only agent_0 can stand adjacent
     to press it; it is visible to agent_1 only when agent_1 is at column 5.
  P  (pot)                        at (2,4), reachable only by agent_1 (from
     column 5).
  B  (plate piles)                at (1,4) and (3,4), reachable only by
     agent_1 (from column 5).
  Ingredient supply 0 at (1,7), supply 1 at (3,7): reachable only by
     agent_1 (from column 6). Ingredient supply 2 is only reachable by
     agent_0 and is never used by either configured recipe.
  Delivery X at (2,7): reachable only by agent_1 (from column 6).

Because of this partition, agent_1 is the only seat that can physically
execute the full ingredient -> pot -> plate -> deliver cycle, but only
agent_0 can observe the always-on recipe indicator and only agent_0 can
press the button that broadcasts the recipe to agent_1. So agent_0's role
is: read the true recipe (always visible to it) and press the button
whenever the recipe changes (and periodically thereafter, to raise the
odds agent_1 catches the broadcast window), while agent_1's role is: read
the button whenever visible, remember the last recipe it saw, and run the
cook/deliver cycle, waiting near the pot (which doubles as the button's
viewing spot) whenever it has nothing else useful to do.

Both seats self-localize every step from a landmark that is always inside
their own view (R for agent_0, the pot for agent_1), rather than tracking
position via dead reckoning, so blocked moves never desync navigation.
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

# Action ids (plain Python ints; cast to jnp arrays inside functions only).
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

# Facing ids as encoded in the observation (channels 1..4 / 11..14): UP=0, DOWN=1, RIGHT=2, LEFT=3.
FACE_UP = 0
FACE_DOWN = 1
FACE_RIGHT = 2
FACE_LEFT = 3

PRESS_INTERVAL = 60  # steps between defensive re-broadcasts of an unchanged recipe.


def _facing_id(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _find_marker(obs, channel):
    flat = obs[:, :, channel].reshape(-1)
    idx = jnp.argmax(flat)
    found = flat[idx] > 0
    r = (idx // 5).astype(jnp.int32)
    c = (idx % 5).astype(jnp.int32)
    return r, c, found


def _local_coords(self_row, self_col, gr, gc):
    lr = gr - self_row + 2
    lc = gc - self_col + 2
    visible = (lr >= 0) & (lr <= 4) & (lc >= 0) & (lc <= 4)
    return jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), visible


def _read(obs, self_row, self_col, gr, gc, channel):
    lr, lc, visible = _local_coords(self_row, self_col, gr, gc)
    val = obs[lr, lc, channel]
    return jnp.where(visible, val, 0)


def _move_toward(dr, dc):
    return jnp.where(
        dr > 0, DOWN,
        jnp.where(
            dr < 0, UP,
            jnp.where(dc > 0, RIGHT, jnp.where(dc < 0, LEFT, STAY)),
        ),
    ).astype(jnp.int32)


def _face_to_action(desired_facing):
    face_to_action = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)
    return face_to_action[desired_facing]


def _step_to(self_row, self_col, facing_id, tr, tc, desired_facing, interact_flag):
    at_target = (self_row == tr) & (self_col == tc)
    move_action = _move_toward(tr - self_row, tc - self_col)
    turn_action = _face_to_action(desired_facing)
    action = jnp.where(
        ~at_target, move_action,
        jnp.where(
            facing_id != desired_facing, turn_action,
            jnp.where(interact_flag, INTERACT, STAY),
        ),
    )
    return action.astype(jnp.int32)


# --------------------------------------------------------------------------
# agent_0: watches the always-on recipe indicator R at (2,0) and presses the
# button L at (2,3) to broadcast the recipe to agent_1. It never touches any
# ingredient, plate, or the pot (it cannot reach them).
# --------------------------------------------------------------------------

def init_agent_0():
    return {
        'last_recipe': jnp.asarray(-1, dtype=jnp.int32),
        'steps_since_press': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    r_idx, c_idx, found = _find_marker(obs, 23)  # recipe_indicator channel
    self_row = jnp.where(found, 4 - r_idx, jnp.asarray(2, dtype=jnp.int32))
    self_col = jnp.where(found, 2 - c_idx, jnp.asarray(2, dtype=jnp.int32))
    facing_id = _facing_id(obs)

    ch0 = obs[r_idx, c_idx, 36]
    ch1 = obs[r_idx, c_idx, 37]
    recipe_id = jnp.where(~found, -1, jnp.where(ch0 > 0, 0, jnp.where(ch1 > 0, 1, -1))).astype(jnp.int32)

    need_press = (recipe_id >= 0) & (
        (recipe_id != memory['last_recipe']) | (memory['steps_since_press'] >= PRESS_INTERVAL)
    )

    target_row = jnp.asarray(2, dtype=jnp.int32)
    target_col = jnp.asarray(2, dtype=jnp.int32)
    action = _step_to(self_row, self_col, facing_id, target_row, target_col, FACE_RIGHT, need_press)

    at_target = (self_row == target_row) & (self_col == target_col)
    pressed_now = need_press & at_target & (facing_id == FACE_RIGHT)

    next_memory = {
        'last_recipe': jnp.where(pressed_now, recipe_id, memory['last_recipe']),
        'steps_since_press': jnp.where(pressed_now, jnp.asarray(0, dtype=jnp.int32), memory['steps_since_press'] + 1),
    }
    next_memory = {k: jnp.where(reset, init_agent_0()[k], v) for k, v in next_memory.items()}

    trace = {'recipe': recipe_id, 'pressed': pressed_now.astype(jnp.int32)}
    return action, next_memory, trace


# --------------------------------------------------------------------------
# agent_1: runs the full ingredient -> pot -> plate -> deliver cycle. It
# reads the current recipe off the button (only visible from column 5) and
# remembers the last value observed; when it has nothing else useful to do
# it waits at the pot / button viewing spot at column 5.
# --------------------------------------------------------------------------

def init_agent_1():
    # batch_recipe: the recipe locked in for the pot batch currently being
    # assembled/cooked/delivered, -1 until confirmed by at least one look at
    # the button this cycle. It is cleared back to -1 right after a delivery,
    # forcing a fresh look (a detour through column 5) before the next
    # ingredient is ever picked up -- this is what prevents fetching an
    # ingredient for a recipe that has gone stale.
    # last_seen_recipe: the most recent recipe actually read off the button,
    # persisted across cycles so an inactive button (because the recipe did
    # not change) doesn't force a long wait for a defensive re-broadcast.
    # steps_since_fresh: steps since last_seen_recipe was actually confirmed
    # by a live reading. Trusting it forever would be unsafe against a
    # partner that stops broadcasting updates -- past STALE_LIMIT steps
    # without any fresh confirmation, treat the recipe as unknown again and
    # go back to waiting for a live signal instead of silently gambling with
    # a guess that may no longer be true.
    return {
        'batch_recipe': jnp.asarray(-1, dtype=jnp.int32),
        'last_seen_recipe': jnp.asarray(-1, dtype=jnp.int32),
        'steps_since_fresh': jnp.asarray(0, dtype=jnp.int32),
    }


STALE_LIMIT = 150  # steps without a fresh button reading before distrusting the persisted guess.


def agent_1_act(obs, reset, memory, rng):
    del rng
    r_idx, c_idx, found = _find_marker(obs, 22)  # pot channel
    self_row = jnp.where(found, 4 - r_idx, jnp.asarray(2, dtype=jnp.int32))
    self_col = jnp.where(found, 6 - c_idx, jnp.asarray(5, dtype=jnp.int32))
    facing_id = _facing_id(obs)

    # Own inventory.
    plate_flag = obs[2, 2, 5] > 0
    cooked_flag = obs[2, 2, 6] > 0
    held_ing0 = obs[2, 2, 7] > 0
    held_ing1 = obs[2, 2, 8] > 0
    held_ing2 = obs[2, 2, 9] > 0
    holding_dish = plate_flag & cooked_flag
    holding_plate_only = plate_flag & (~cooked_flag)
    holding_any_ingredient = held_ing0 | held_ing1 | held_ing2
    holding_nothing = (~plate_flag) & (~cooked_flag) & (~holding_any_ingredient)

    # Pot state (pot cell == the marker cell used for self-localization).
    pot_cooked = jnp.where(found, obs[r_idx, c_idx, 30], 0) > 0
    pot_ing0 = jnp.where(found, obs[r_idx, c_idx, 31], 0)
    pot_ing1 = jnp.where(found, obs[r_idx, c_idx, 32], 0)
    pot_ing2 = jnp.where(found, obs[r_idx, c_idx, 33], 0)
    pot_timer = jnp.where(found, obs[r_idx, c_idx, 39], 0)
    pot_total = pot_ing0 + pot_ing1 + pot_ing2
    pot_ready = pot_cooked
    pot_cooking = (pot_timer > 0) & (~pot_ready)
    pot_has_room = (pot_total < 3) & (pot_timer == 0) & (~pot_ready)

    # Recipe display via the button, only visible from column 5. This is a
    # *fresh* per-step observation, not carried state. -1 means either not
    # currently visible (not at column 5) or visible but inactive.
    btn_ing0 = _read(obs, self_row, self_col, 2, 3, 36)
    btn_ing1 = _read(obs, self_row, self_col, 2, 3, 37)
    observed_recipe = jnp.where(btn_ing0 > 0, 0, jnp.where(btn_ing1 > 0, 1, -1)).astype(jnp.int32)
    at_button_column = self_col == 5

    # Best current guess of the true recipe: a fresh reading if we have one
    # this step, else whatever we last saw (persisted across cycles), but
    # only if that memory is not too stale to trust.
    guess_is_fresh = observed_recipe >= 0
    persisted_is_trustworthy = memory['steps_since_fresh'] < STALE_LIMIT
    persisted_guess = jnp.where(persisted_is_trustworthy, memory['last_seen_recipe'], jnp.asarray(-1, dtype=jnp.int32))
    best_guess = jnp.where(guess_is_fresh, observed_recipe, persisted_guess)

    # The recipe committed to for the pot batch in progress. Once locked in
    # (memory['batch_recipe'] >= 0) it stays fixed regardless of what the
    # button shows later (the true recipe cannot change mid-batch, since it
    # only changes on a correct delivery, which this agent itself performs
    # and which resets batch_recipe to -1). If not yet locked in, it can only
    # become known once we've actually stood at the button's column at least
    # once this cycle -- an inactive button there means "unchanged" (trust
    # the persisted guess), a fresh reading means "just changed" (trust it).
    # This forces one real look at the button after every delivery, so a
    # stale guess is never used without at least checking it first.
    batch_recipe = jnp.where(
        memory['batch_recipe'] >= 0, memory['batch_recipe'],
        jnp.where(at_button_column, best_guess, jnp.asarray(-1, dtype=jnp.int32)),
    )
    recipe_known = batch_recipe >= 0

    holding_matches_recipe = jnp.where(batch_recipe == 0, held_ing0, jnp.where(batch_recipe == 1, held_ing1, False))
    holding_wrong_for_recipe = holding_any_ingredient & (~holding_matches_recipe)

    plate_target_row = jnp.where(self_row <= 2, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    ing_target_row = jnp.where(batch_recipe == 0, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))

    # Priority-ordered task selection: (target_row, target_col, desired_facing, interact_flag).
    tr = self_row
    tc = self_col
    fc = facing_id
    ia = jnp.asarray(False)

    # c8: recipe unknown -> wait at column 5, watching the button.
    cond = ~recipe_known
    tr = jnp.where(cond, jnp.asarray(2, dtype=jnp.int32), tr)
    tc = jnp.where(cond, jnp.asarray(5, dtype=jnp.int32), tc)
    fc = jnp.where(cond, facing_id, fc)
    ia = jnp.where(cond, jnp.asarray(False), ia)

    # c7: holding a plate with nothing else to do -> wait at column 5.
    cond = holding_plate_only & (~pot_ready) & (~pot_cooking)
    tr = jnp.where(cond, jnp.asarray(2, dtype=jnp.int32), tr)
    tc = jnp.where(cond, jnp.asarray(5, dtype=jnp.int32), tc)
    fc = jnp.where(cond, facing_id, fc)
    ia = jnp.where(cond, jnp.asarray(False), ia)

    # c6: fetch the ingredient matching the known recipe.
    cond = pot_has_room & recipe_known & holding_nothing
    tr = jnp.where(cond, ing_target_row, tr)
    tc = jnp.where(cond, jnp.asarray(6, dtype=jnp.int32), tc)
    fc = jnp.where(cond, FACE_RIGHT, fc)
    ia = jnp.where(cond, jnp.asarray(True), ia)

    # c5: add held matching ingredient to the pot.
    cond = pot_has_room & recipe_known & holding_matches_recipe
    tr = jnp.where(cond, jnp.asarray(2, dtype=jnp.int32), tr)
    tc = jnp.where(cond, jnp.asarray(5, dtype=jnp.int32), tc)
    fc = jnp.where(cond, FACE_LEFT, fc)
    ia = jnp.where(cond, jnp.asarray(True), ia)

    # c4: pot cooking or ready and hands empty -> fetch a plate ahead of time.
    cond = (pot_ready | pot_cooking) & holding_nothing
    tr = jnp.where(cond, plate_target_row, tr)
    tc = jnp.where(cond, jnp.asarray(5, dtype=jnp.int32), tc)
    fc = jnp.where(cond, FACE_LEFT, fc)
    ia = jnp.where(cond, jnp.asarray(True), ia)

    # c3: holding something unusable right now -> dump it on whichever of the
    # two overhead counters ((0,5) or (0,6)) is currently empty, so a single
    # occupied counter can never permanently strand this fallback.
    cell_05_occupied = sum(_read(obs, self_row, self_col, 0, 5, ch) for ch in (29, 30, 31, 32, 33)) > 0
    cell_06_occupied = sum(_read(obs, self_row, self_col, 0, 6, ch) for ch in (29, 30, 31, 32, 33)) > 0
    dump_col = jnp.where(cell_05_occupied & (~cell_06_occupied), jnp.asarray(6, dtype=jnp.int32), jnp.asarray(5, dtype=jnp.int32))
    cond = holding_any_ingredient & (pot_ready | holding_wrong_for_recipe)
    tr = jnp.where(cond, jnp.asarray(1, dtype=jnp.int32), tr)
    tc = jnp.where(cond, dump_col, tc)
    fc = jnp.where(cond, FACE_UP, fc)
    ia = jnp.where(cond, jnp.asarray(True), ia)

    # c2: pot ready and holding an empty plate -> collect the dish.
    cond = pot_ready & holding_plate_only
    tr = jnp.where(cond, jnp.asarray(2, dtype=jnp.int32), tr)
    tc = jnp.where(cond, jnp.asarray(5, dtype=jnp.int32), tc)
    fc = jnp.where(cond, FACE_LEFT, fc)
    ia = jnp.where(cond, jnp.asarray(True), ia)

    # c1: holding the plated cooked dish -> deliver.
    cond = holding_dish
    tr = jnp.where(cond, jnp.asarray(2, dtype=jnp.int32), tr)
    tc = jnp.where(cond, jnp.asarray(6, dtype=jnp.int32), tc)
    fc = jnp.where(cond, FACE_RIGHT, fc)
    ia = jnp.where(cond, jnp.asarray(True), ia)

    action = _step_to(self_row, self_col, facing_id, tr, tc, fc, ia)

    # c1 (holding_dish) is the highest-priority branch above, so whenever
    # holding_dish is true, (tr, tc, fc, ia) are exactly its delivery target;
    # a delivery actually happens this step once we're there, facing it.
    delivered_now = holding_dish & (self_row == 2) & (self_col == 6) & (facing_id == FACE_RIGHT)
    next_batch_recipe = jnp.where(delivered_now, jnp.asarray(-1, dtype=jnp.int32), batch_recipe)
    next_steps_since_fresh = jnp.where(guess_is_fresh, jnp.asarray(0, dtype=jnp.int32), memory['steps_since_fresh'] + 1)
    next_memory = {
        'batch_recipe': next_batch_recipe,
        'last_seen_recipe': jnp.where(guess_is_fresh, observed_recipe, memory['last_seen_recipe']),
        'steps_since_fresh': next_steps_since_fresh,
    }
    next_memory = {k: jnp.where(reset, init_agent_1()[k], v) for k, v in next_memory.items()}

    trace = {
        'batch_recipe': batch_recipe,
        'pot_total': pot_total.astype(jnp.int32),
        'pot_ready': pot_ready.astype(jnp.int32),
        'self_row': self_row,
        'self_col': self_col,
        'holding_wrong_for_recipe': holding_wrong_for_recipe.astype(jnp.int32),
    }
    return action, next_memory, trace
