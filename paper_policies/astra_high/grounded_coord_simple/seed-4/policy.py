"""Grounded coordination for two disconnected kitchen rooms."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_cook_v8', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def _position(obs, channel, global_row, global_col):
    k = jnp.argmax(obs[:, :, channel])
    return global_row + 2 - k // 5, global_col + 2 - k % 5


def _goto(r, c, facing, tr, tc, direction):
    """Move to an approach cell, then face and interact with its object."""
    turn = jnp.asarray([3, 1, 0, 2], jnp.int32)[direction]
    action = jnp.where(facing == direction, 5, turn)
    action = jnp.where(c != tc, jnp.where(c < tc, 0, 2), action)
    action = jnp.where(r != tr, jnp.where(r < tr, 1, 3), action)
    return action.astype(jnp.int32)


def init_agent_0():
    # time, previous cooked flag, pickup age, reveal age, has revealed, previous recipe, attempts
    return jnp.asarray([0, 0, -1, 100, 0, -1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    m = jnp.where(reset, init_agent_0(), memory)
    r, c = _position(obs, 23, 2, 0)
    facing = jnp.argmax(obs[2, 2, 1:5])
    pot_mask = obs[:, :, 22]
    cooked = jnp.sum(pot_mask * obs[:, :, 30])
    contents = jnp.sum(pot_mask[:, :, None] * obs[:, :, 31:34])
    seen = jnp.any(pot_mask)
    pickup = seen & (m[1] == 1) & (cooked == 0) & (contents == 0)
    age = jnp.where(pickup, 0, jnp.where(m[2] >= 0, m[2] + 1, -1))
    # A dish can be delivered two actions after plating. Reveal just afterward.
    current_recipe = jnp.argmax(jnp.sum(obs[:, :, 36:39] * obs[:, :, 23:24], axis=(0, 1)))
    changed = (m[5] >= 0) & (current_recipe != m[5])
    lit = jnp.sum(obs[:, :, 36:39] * obs[:, :, 24:25]) > 0
    attempts = jnp.where(contents > 0, 0, m[6])
    retry = seen & (age < 0) & (contents == 0) & (m[3] >= 18) & (attempts < 3)
    due = ((m[4] == 0) | (age >= 2) | (changed & ~lit) | retry) & (m[0] <= 360)
    action = _goto(r, c, facing, 2, 2, 2)
    at_button = (r == 2) & (c == 2)
    action = jnp.where(at_button & ~due, 4, action).astype(jnp.int32)
    activated = at_button & (facing == 2) & due
    age = jnp.where(activated, -1, age)
    nm = jnp.asarray([m[0] + 1, jnp.where(seen, cooked, m[1]), age,
                      jnp.where(activated, 0, m[3] + 1), m[4] | activated, current_recipe, attempts + activated], jnp.int32)
    return action, nm, {'pickup': pickup, 'reveal': activated, 'age': age}


def init_agent_1():
    # t, recipe, held-dish flag, caches, preparation, dish type, wait, confirmed, spare counters
    state = jnp.asarray([0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0], jnp.int32)
    # Rows: responsive-on-change helper, unresponsive helper; columns: target ingredient.
    belief = jnp.asarray([[0.0, 0.0], [0.5, 0.5]], jnp.float32)
    pending_change = jnp.zeros((2,), jnp.float32)
    return state, belief, pending_change


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    m, belief, pending = tuple(jnp.where(reset, z, x) for z, x in zip(initial, memory))
    r, c = _position(obs, 22, 2, 4)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    plate, dish = inv[0] > 0, inv[1] > 0
    raw = (jnp.sum(inv[2:]) > 0) & ~dish
    raw_type = jnp.argmax(inv[2:])
    dish_type = jnp.where(dish, raw_type, m[6])
    delivered = (m[2] > 0) & ~dish & (r == 2) & (c == 6)

    # A matching delivery resamples the target; an incorrect one leaves it intact.
    half_match = 0.5 * belief[:, m[6]]
    transitioned = belief.at[:, m[6]].set(half_match)
    transitioned = transitioned.at[:, 1 - m[6]].add(half_match)
    changed_mass = jnp.eye(2, dtype=jnp.float32)[1 - m[6]] * half_match[0]
    belief = jnp.where(delivered, transitioned, belief)
    pending = jnp.where(delivered, changed_mass, pending)
    recipe = jnp.where(delivered, -1, m[1])
    confirmed = (m[8] > 0) & ~delivered
    wait = jnp.where(delivered, 0, m[7] + 1)

    # Actual displays take precedence over all guesses or inferred signaling habits.
    display = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    fresh = jnp.sum(display) > 0
    response_age = jnp.where(delivered, 0, jnp.where(m[11] >= 0, m[11] + 1, -1))
    response_delay = jnp.where(fresh & (response_age >= 0), jnp.maximum(m[12], response_age), m[12])
    response_age = jnp.where(fresh, -1, response_age)
    timeout = jnp.clip(response_delay + 2, 24, 60)
    shown = jnp.argmax(display)
    observed_belief = jnp.asarray([0.75, 0.25], jnp.float32)[:, None] * jnp.eye(2, dtype=jnp.float32)[jnp.clip(shown, 0, 1)]
    belief = jnp.where(fresh, observed_belief, belief)
    pending = jnp.where(fresh, jnp.zeros_like(pending), pending)
    recipe = jnp.where(fresh, shown, recipe)
    confirmed = confirmed | fresh

    # Silence may mean an unchanged recipe, but repeated silence reduces that confidence.
    fallback = (recipe < 0) & (wait >= timeout)
    silent_belief = belief.at[0].set(jnp.maximum(belief[0] - pending, 0.0))
    silent_belief = silent_belief / jnp.maximum(jnp.sum(silent_belief), 1e-8)
    belief = jnp.where(fallback, silent_belief, belief)
    pending = jnp.where(fallback, jnp.zeros_like(pending), pending)
    recipe = jnp.where(fallback, jnp.argmax(jnp.sum(belief, axis=0) + 1e-5 * raw * jnp.eye(2)[jnp.clip(raw_type, 0, 1)]), recipe)
    wait = jnp.where(recipe >= 0, 0, wait)

    pm = obs[:, :, 22]
    pot_counts = jnp.sum(pm[:, :, None] * obs[:, :, 31:34], axis=(0, 1))
    count = jnp.sum(pot_counts)
    full = count == 3
    timer = jnp.sum(pm * obs[:, :, 39])
    ready = jnp.any(pm * obs[:, :, 30])
    cache_rows = jnp.asarray([0, 4]) + 2 - r
    cache_seen = (cache_rows >= 0) & (cache_rows < 5)
    cache_items = obs[jnp.clip(cache_rows, 0, 4), 7 - c, 31:34]
    cache_available = jnp.asarray([cache_items[0, 0] > 0, cache_items[1, 1] > 0])
    caches = jnp.where(cache_seen, cache_available, m[3:5] > 0)
    spare_items = obs[jnp.clip(cache_rows, 0, 4), 8 - c, 29:34]
    # Spare codes: 0 empty, 1 ingredient-0 dish, 2 ingredient-1 dish, 3 other item.
    spare_cooked = (spare_items[:, 0] > 0) & (spare_items[:, 1] > 0)
    observed_spares = jnp.where(spare_cooked, 1 + jnp.argmax(spare_items[:, 2:], axis=1),
                                jnp.where(jnp.sum(spare_items, axis=1) > 0, 3, 0))
    spare_types = jnp.where(cache_seen, observed_spares, m[9:11])
    occupied_spares = spare_types > 0

    # If a late display corrects a guess after loading started, finish that batch and
    # set it aside. A partially filled pot cannot otherwise be cleared.
    target_type = jnp.where((count > 0) & ~full, jnp.argmax(pot_counts), recipe)
    target_row = jnp.where(target_type == 1, 3, 1)
    wait_recipe = jnp.where(c == 5, 4, 2).astype(jnp.int32)
    ingredient_action = _goto(r, c, facing, target_row, 6, 2)
    cache_direction = jnp.where(target_type == 1, 1, 0)
    cache_action = _goto(r, c, facing, target_row, 5, cache_direction)
    ingredient_action = jnp.where(caches[jnp.clip(target_type, 0, 1)], cache_action, ingredient_action)
    pot_action = _goto(r, c, facing, 2, 5, 3)
    plate_row = jnp.where(r == 3, 3, 1)
    plate_action = _goto(r, c, facing, plate_row, 5, 3)
    delivery_action = _goto(r, c, facing, 2, 6, 2)
    action = jnp.where(recipe >= 0, ingredient_action, wait_recipe)
    matching_spares = (spare_types == recipe + 1) & (recipe >= 0)
    reuse_type = jnp.where(matching_spares[1] & ((r >= 2) | ~matching_spares[0]), 1, 0)
    reuse = confirmed & jnp.any(matching_spares) & (count == 0) & ~raw & ~plate
    reuse_action = _goto(r, c, facing, jnp.where(reuse_type == 0, 1, 3), 6, jnp.where(reuse_type == 0, 0, 1))
    action = jnp.where(reuse, reuse_action, action)
    opening_prefetch = (recipe < 0) & (m[0] < 3) & ~raw & ~plate & (count == 0)
    opening_action = _goto(r, c, facing, jnp.where(r == 3, 3, 1), 6, 2)
    action = jnp.where(opening_prefetch, opening_action, action)
    action = jnp.where(full | ready, plate_action, action)

    previous_prep = jnp.where((m[5] >= 0) & ~caches[jnp.clip(m[5], 0, 1)], m[5], -1)
    start_prep = full & ~jnp.all(caches) & (timer >= 12) & ~raw & ~plate
    prep_type = jnp.where(previous_prep >= 0, previous_prep, jnp.where(start_prep, jnp.where(~caches[0], 0, 1), -1))
    prep_type = jnp.where(full, prep_type, -1)
    prep_row = jnp.where(prep_type == 0, 1, 3)
    prep = full & (prep_type >= 0) & ~raw & ~plate
    prep_action = _goto(r, c, facing, prep_row, 6, 2)
    action = jnp.where(prep, prep_action, action)
    drop_row = jnp.where(raw_type == 0, 1, 3)
    drop_dir = jnp.where(raw_type == 0, 0, 1)
    drop_action = _goto(r, c, facing, drop_row, 5, drop_dir)
    guarded_pot_action = jnp.where(raw & (recipe < 0) & (pot_action == 5), 4, pot_action)
    action = jnp.where(raw | plate, guarded_pot_action, action)
    mismatch_raw = raw & (target_type >= 0) & (raw_type != target_type)
    action = jnp.where(raw & (full | mismatch_raw), drop_action, action)
    action = jnp.where(dish, delivery_action, action)

    wrong_dish = dish & confirmed & (raw_type != recipe)
    spare_type = jnp.where(~occupied_spares[0], 0, 1)
    spare_action = _goto(r, c, facing, jnp.where(spare_type == 0, 1, 3), 6, jnp.where(spare_type == 0, 0, 1))
    action = jnp.where(wrong_dish & ~jnp.all(occupied_spares), spare_action, action).astype(jnp.int32)
    nm = jnp.asarray([m[0] + 1, recipe, dish, caches[0], caches[1], prep_type, dish_type,
                      wait, confirmed, spare_types[0], spare_types[1], response_age, response_delay], jnp.int32)
    return action, (nm, belief, pending), {'recipe': recipe, 'confirmed': confirmed, 'belief': jnp.sum(belief, axis=0),
              'pot_counts': pot_counts, 'position': jnp.asarray([r, c]), 'delivered': delivered, 'fallback': fallback, 'wrong_dish': wrong_dish, 'timeout': timeout, 'reuse': reuse}
