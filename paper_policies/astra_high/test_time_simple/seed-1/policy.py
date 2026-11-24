"""Observable spatial recipe signalling and continuous cooking."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'dish_cache_v8', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def init_agent_0():
    return {'response': jnp.full(8, -1, dtype=jnp.int32),
            'code': jnp.int32(1), 'batchcode': jnp.int32(1),
            'count': jnp.int32(0), 'recipe': jnp.int32(-1), 'age': jnp.int32(0),
            'ready': jnp.bool_(False), 'batchvalid': jnp.bool_(False)}


def init_agent_1():
    # Joint belief over a partner's two counter codes and current target.
    codes = jnp.arange(8, dtype=jnp.int32)
    upper, middle, lower = (codes & 1) > 0, (codes & 2) > 0, (codes & 4) > 0
    centroid = (lower.astype(jnp.float32) - upper) / jnp.maximum(upper + middle.astype(jnp.int32) + lower, 1)
    weight = jnp.exp(0.35 * (centroid[None, :] - centroid[:, None]))
    single = (codes == 1) | (codes == 2) | (codes == 4)
    weight *= jnp.where(single[:, None] & single[None, :], 3.0, 1.0)
    weight *= jnp.where(codes[:, None] == codes[None, :], 0.03, 1.0)
    weight = weight.at[0, 0].set(1.0)
    belief = jnp.stack([weight, weight], axis=-1)
    belief /= jnp.sum(belief)
    return {'time': jnp.int32(0), 'cooldown': jnp.int32(10),
            'recipe': jnp.int32(0), 'dish': jnp.bool_(False),
            'decide': jnp.bool_(True), 'belief': belief,
            'code': jnp.int32(0), 'p0': jnp.float32(0.5),
            'visible_code': jnp.int32(-1), 'stable': jnp.int32(0),
            'cycle_prior': belief, 'since_delivery': jnp.int32(0), 'refresh_wait': jnp.int32(9),
            'cache': jnp.zeros(2, dtype=jnp.bool_), 'goal': jnp.bool_(False),
            'store': jnp.bool_(False), 'held_recipe': jnp.int32(0), 'revised_cycle': jnp.bool_(False)}


def position(obs, channel, object_row, object_col):
    idx = jnp.argmax(obs[:, :, channel])
    return object_row - idx // 5 + 2, object_col - idx % 5 + 2


def cell(obs, row, col, target_row, target_col, channel):
    lr, lc = target_row - row + 2, target_col - col + 2
    valid = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    return jnp.where(valid, obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), channel], 0)


def interact_at(row, col, direction, target_row, target_col, facing):
    face_action = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[facing]
    on_target = jnp.where(direction == facing, 5, face_action)
    horizontal = jnp.where(col < target_col, 0, 2)
    vertical = jnp.where(row < target_row, 1, 3)
    return jnp.where(row != target_row, vertical,
                     jnp.where(col != target_col, horizontal, on_target)).astype(jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    import jax
    memory = jax.tree.map(lambda new, old: jnp.where(reset, new, old), init_agent_0(), memory)
    row, col = position(obs, 23, 2, 0)
    facing = jnp.argmax(obs[2, 2, 1:5])
    recipe = jnp.argmax(jnp.sum(obs[:, :, 36:39], axis=(0, 1)))
    ingredients = jnp.asarray([cell(obs, row, col, 2, 4, ch) for ch in (31, 32, 33)])
    count = jnp.where(col == 2, jnp.sum(ingredients), memory['count'])
    batchcode = jnp.where((count > 0) & (memory['count'] == 0), memory['code'], memory['batchcode'])
    batchvalid = jnp.where((count > 0) & (memory['count'] == 0), memory['ready'] & (count == 1), memory['batchvalid'])
    full = (col == 2) & (count == 3) & (memory['count'] < 3)
    cooked_recipe = jnp.argmax(ingredients)
    pure = jnp.max(ingredients) == 3
    response = memory['response']
    # Previously successful signals are retained when a learning partner probes.
    learn = full & batchvalid & pure & (cooked_recipe < 2) & ((response[batchcode] != recipe) | (cooked_recipe == recipe))
    response = response.at[batchcode].set(jnp.where(learn, cooked_recipe, response[batchcode]))
    age = memory['age'] + 1
    stalled = (age > 65) & (count == 0) & (response[memory['code']] < 0)
    response = response.at[memory['code']].set(jnp.where(stalled, -2, response[memory['code']]))
    default_code = jnp.where(recipe == 0, 1, 4)
    other_code = 5 - default_code
    preferences = jnp.asarray([0.0, 0.0, 5.0, 3.0, 0.0, 2.0, 3.0, 1.0])
    preferences = preferences.at[default_code].set(7.0).at[other_code].set(6.0)
    both_wrong = (response[1] == 1 - recipe) & (response[4] == 1 - recipe)
    composite = default_code | 2
    preferences = preferences.at[composite].set(jnp.where(both_wrong, 5.5, 4.0))
    preferences = preferences.at[0].set(jnp.where(both_wrong, 5.25, 0.0))
    scores = preferences + jnp.where(response == recipe, 20.0, jnp.where(response == -1, 0.0, -20.0))
    choice = jnp.argmax(scores).astype(jnp.int32)
    reconsider = (recipe != memory['recipe']) | full | stalled
    code = jnp.where(reconsider, choice, memory['code'])
    age = jnp.where(reconsider, 0, age)
    rows = jnp.arange(1, 4, dtype=jnp.int32)
    wanted = (jnp.asarray([1, 2, 4], dtype=jnp.int32) & code) > 0
    present = jnp.asarray([cell(obs, row, col, r, 3, 33) > 0 for r in (1, 2, 3)])
    unwanted = present & ~wanted
    missing = wanted & ~present
    holding = obs[2, 2, 9] > 0
    clear_row = 1 + jnp.argmax(unwanted)
    put_row = 1 + jnp.argmax(missing)
    source_row = jnp.where(row <= 2, 1, 3)
    source_face = jnp.where(source_row == 1, 0, 1)
    acquire = interact_at(row, col, facing, source_row, 2, source_face)
    stash = jnp.asarray([cell(obs, row, col, r, 0, 33) > 0 for r in (1, 3)])
    take_stash_row = 1 + 2 * jnp.argmax(stash)
    free_stash_row = 1 + 2 * jnp.argmax(~stash)
    take_stash = interact_at(row, col, facing, take_stash_row, 1, 3)
    drop_stash = interact_at(row, col, facing, free_stash_row, 1, 3)
    acquire = jnp.where(jnp.any(stash), take_stash, acquire)
    put = interact_at(row, col, facing, put_row, 2, 2)
    remove = interact_at(row, col, facing, clear_row, 2, 2)
    idle = jnp.where(col == 2, 4, 0)
    action = jnp.where(holding, jnp.where(jnp.any(missing), put, jnp.where(jnp.any(unwanted) & jnp.any(~stash), drop_stash, idle)),
                jnp.where(jnp.any(unwanted), remove, jnp.where(jnp.any(missing), acquire, idle)))
    next_memory = {'response': response, 'code': code, 'batchcode': batchcode,
                   'count': count, 'recipe': recipe, 'age': age,
                   'ready': jnp.all(present == wanted), 'batchvalid': batchvalid}
    return action.astype(jnp.int32), next_memory, {
        'position': jnp.asarray([row, col]), 'recipe': recipe, 'code': code,
        'pot_count': count, 'response': response, 'learn': learn}


def agent_1_act(obs, reset, memory, rng):
    import jax
    memory = jax.tree.map(lambda new, old: jnp.where(reset, new, old), init_agent_1(), memory)
    row, col = position(obs, 22, 2, 4)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    plate, dish = inv[0] > 0, inv[1] > 0
    raw = jnp.sum(inv[2:]) > 0
    empty = jnp.sum(inv) == 0
    pot_count = sum(cell(obs, row, col, 2, 4, ch) for ch in (31, 32, 33))
    upper = cell(obs, row, col, 1, 3, 33) > 0
    middle = cell(obs, row, col, 2, 3, 33) > 0
    lower = cell(obs, row, col, 3, 3, 33) > 0
    code = upper.astype(jnp.int32) + 2 * middle + 4 * lower
    held_recipe = jnp.argmax(inv[2:4]).astype(jnp.int32)
    cache_visible = jnp.abs(jnp.asarray([0, 4]) - row) <= 2
    cache_seen = jnp.asarray([cell(obs, row, col, r, 5, 30) > 0 for r in (0, 4)])
    cache = jnp.where(cache_visible, cache_seen, memory['cache'])
    delivered = memory['goal'] & memory['dish'] & ~dish
    stored = memory['store'] & memory['dish'] & ~dish
    since_delivery = jnp.where(delivered, 0, memory['since_delivery'] + 1)
    cooldown = jnp.where(delivered, memory['refresh_wait'], jnp.maximum(memory['cooldown'] - 1, 0))
    decide = memory['decide'] | delivered
    stable = jnp.where((col == 5) & (code == memory['visible_code']) & ~delivered, memory['stable'] + 1, 0)
    belief = memory['belief']
    matched = belief[:, :, memory['held_recipe']] * 0.5
    updated = belief.at[:, :, memory['held_recipe']].set(0)
    updated += matched[:, :, None]
    belief = jnp.where(delivered, updated, belief)
    codes = jnp.arange(8, dtype=jnp.int32)
    likelihood = jnp.stack([jnp.broadcast_to(codes[:, None] == code, (8, 8)),
                            jnp.broadcast_to(codes[None, :] == code, (8, 8))], axis=-1)
    pair_belief = jnp.sum(belief, axis=-1)
    best_pair = jnp.argmax(pair_belief)
    first_code, second_code = best_pair // 8, best_pair % 8
    single_first = (first_code == 1) | (first_code == 2) | (first_code == 4)
    single_second = (second_code == 1) | (second_code == 2) | (second_code == 4)
    known_spatial = (jnp.max(pair_belief) > 0.98) & single_first & single_second & (first_code != second_code)
    recognized = (code == first_code) | (code == second_code)
    early = known_spatial & recognized & (stable >= 1) & (cooldown <= 7) & (memory['refresh_wait'] == 9)
    commit = decide & ((cooldown == 0) | early) & (col == 5) & empty & (pot_count == 0)
    posterior = belief * jnp.where(likelihood, 1.0, 0.0001)
    posterior /= jnp.maximum(jnp.sum(posterior), 1e-20)
    cycle_prior = jnp.where(commit, belief, memory['cycle_prior'])
    belief = jnp.where(commit, posterior, belief)
    # A target cannot change during a dish's preparation. A later message is
    # therefore a revision of this cycle's observation, not a new recipe.
    late = ~decide & (col == 5) & (stable >= 2) & (code != memory['code'])
    revised = cycle_prior * jnp.where(likelihood, 1.0, 0.0001)
    revised /= jnp.maximum(jnp.sum(revised), 1e-20)
    belief = jnp.where(late, revised, belief)
    observed_delay = jnp.minimum(since_delivery - stable + 3, 24)
    refresh_wait = jnp.where(late, jnp.maximum(memory['refresh_wait'], observed_delay), memory['refresh_wait'])
    p0 = jnp.sum(belief[:, :, 0])
    recipe = jnp.where(commit | stored | (late & empty & (pot_count == 0)), (p0 < 0.5).astype(jnp.int32), memory['recipe'])
    supply_row = jnp.where(recipe == 0, 1, 3)
    plate_row = jnp.where(row <= 2, 1, 3)
    supply_action = interact_at(row, col, facing, supply_row, 6, 2)
    cache_face = jnp.where(recipe == 0, 0, 1)
    take_cache = interact_at(row, col, facing, supply_row, 5, cache_face)
    supply_action = jnp.where(cache[recipe] & (pot_count == 0), take_cache, supply_action)
    pot_action = interact_at(row, col, facing, 2, 5, 3)
    plate_action = interact_at(row, col, facing, plate_row, 5, 3)
    deliver_action = interact_at(row, col, facing, 2, 6, 2)
    revised_cycle = jnp.where(commit, False, memory['revised_cycle']) | late
    dish_match_probability = jnp.where(held_recipe == 0, p0, 1.0 - p0)
    stash_dish = dish & revised_cycle & (dish_match_probability < 0.35) & ~cache[held_recipe] & (memory['time'] < 340)
    stash_row = jnp.where(held_recipe == 0, 1, 3)
    stash_face = jnp.where(held_recipe == 0, 0, 1)
    stash_action = interact_at(row, col, facing, stash_row, 5, stash_face)
    dish_action = jnp.where(stash_dish, stash_action, deliver_action)
    provisional = (jnp.sum(posterior[:, :, 0]) < 0.5).astype(jnp.int32)
    wait_row = jnp.where(provisional == 0, 1, 3)
    wait_action = jnp.where(col != 5, 2, jnp.where(row != wait_row, jnp.where(row < wait_row, 1, 3), 4))
    action = jnp.where(dish, dish_action,
                jnp.where(plate, pot_action,
                jnp.where(raw, pot_action,
                jnp.where(pot_count >= 3, plate_action,
                jnp.where(decide & ~commit, wait_action, supply_action)))))
    next_memory = {'time': memory['time'] + 1, 'cooldown': cooldown,
                   'recipe': recipe, 'dish': dish, 'decide': decide & ~commit,
                   'belief': belief, 'code': jnp.where(commit | late, code, memory['code']), 'p0': p0,
                   'visible_code': jnp.where(col == 5, code, memory['visible_code']), 'stable': stable,
                   'cycle_prior': cycle_prior, 'since_delivery': since_delivery, 'refresh_wait': refresh_wait,
                   'cache': cache, 'held_recipe': jnp.where(dish, held_recipe, memory['held_recipe']),
                   'goal': dish & ~stash_dish & (action == 5) & (row == 2) & (col == 6) & (facing == 2),
                   'store': dish & stash_dish & (action == 5) & (row == stash_row) & (col == 5) & (facing == stash_face),
                   'revised_cycle': revised_cycle}
    return action.astype(jnp.int32), next_memory, {
        'position': jnp.asarray([row, col]), 'recipe': recipe,
        'code': code, 'pot_count': pot_count, 'p0': p0,
        'cooldown': cooldown, 'delivered': delivered, 'commit': commit, 'known_spatial': known_spatial, 'late': late, 'refresh_wait': refresh_wait, 'cache': cache, 'stored': stored, 'stash_dish': stash_dish}
