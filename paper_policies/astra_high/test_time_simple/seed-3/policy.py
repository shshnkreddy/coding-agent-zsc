"""Local evidence driven recipe signalling and continuing kitchen control."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'adaptive_observable_coordination', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def locate(obs, channel, global_row, global_col):
    idx = jnp.argmax(obs[:, :, channel].reshape(-1))
    return jnp.asarray(global_row + 2, jnp.int32) - idx // 5, jnp.asarray(global_col + 2, jnp.int32) - idx % 5


def cell(obs, row, col, target_row, target_col):
    r, c = target_row - row + 2, target_col - col + 2
    valid = (r >= 0) & (r < 5) & (c >= 0) & (c < 5)
    return jnp.where(valid, obs[jnp.clip(r, 0, 4), jnp.clip(c, 0, 4)], jnp.zeros((40,), jnp.int32))


def use(row, col, facing, stand_row, stand_col, direction):
    """Navigate a rectangular room to an interaction position."""
    turn = jnp.asarray([3, 1, 0, 2], jnp.int32)[direction]
    act = jnp.where(facing == direction, 5, turn)
    act = jnp.where(col != stand_col, jnp.where(col < stand_col, 0, 2), act)
    return jnp.where(row != stand_row, jnp.where(row < stand_row, 1, 3), act).astype(jnp.int32)


def init_agent_0():
    z = jnp.asarray(0, jnp.int32)
    return {'time': z, 'responses': jnp.zeros((8, 2), jnp.int32),
            'timeouts': jnp.zeros((8,), jnp.int32), 'prev_count': z,
            'pattern': z, 'stable': z, 'empty_age': z, 'code': jnp.asarray(1, jnp.int32)}


def agent_0_act(obs, reset, memory, rng):
    mem = {k: jnp.where(reset, v, memory[k]) for k, v in init_agent_0().items()}
    row, col = locate(obs, 23, 2, 0)
    face = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    recipe = jnp.argmax(jnp.sum(obs[:, :, 36:38], axis=(0, 1)))
    occupied = jnp.array([jnp.any(cell(obs, row, col, r, 3)[29:34] > 0) for r in (1, 2, 3)])
    pattern = jnp.sum(occupied.astype(jnp.int32) * jnp.array([1, 2, 4]))
    stable = jnp.where(pattern == mem['pattern'], mem['stable'] + 1, 1)
    pot = cell(obs, row, col, 2, 4)
    visible = col == 2
    count = jnp.where(visible, jnp.sum(pot[31:34]), mem['prev_count'])
    observed_kind = jnp.argmax(pot[31:33])
    sample = visible & (count > 0) & (mem['prev_count'] == 0) & (stable >= 7)
    responses = mem['responses'].at[pattern, observed_kind].add(sample.astype(jnp.int32))
    empty_age = jnp.where((count == 0) & (pattern == mem['code']), mem['empty_age'] + 1, 0)
    timeout = (empty_age >= 24) & (stable >= 10)
    timeouts = mem['timeouts'].at[pattern].add(timeout.astype(jnp.int32))
    # Keep a demonstrated successful encoding; search unknown encodings when the
    # partner's first ingredient shows that the current code was misunderstood.
    priorities = jnp.asarray([[2, 8, 6, 5, 7, 3, 4, 1], [2, 7, 6, 5, 8, 3, 4, 1]], jnp.int32)[recipe]
    good, bad = responses[:, recipe], responses[:, 1 - recipe]
    score = priorities + jnp.where(good > bad, 100, jnp.where(bad > good, -100, 0)) - 20 * timeouts
    score = score.at[2].add(jnp.where(jnp.any(timeouts > 0), 3, 0))
    code = jnp.argmax(score)
    wanted = (code & jnp.array([1, 2, 4])) > 0
    missing, extra = wanted & ~occupied, ~wanted & occupied
    missing_row, extra_row = jnp.argmax(missing) + 1, jnp.argmax(extra) + 1
    stash = jnp.array([jnp.any(cell(obs, row, col, r, 0)[29:34] > 0) for r in (1, 3)])
    pickup_stash_row = 1 + 2 * jnp.argmax(stash)
    drop_stash_row = 1 + 2 * jnp.argmax(~stash)
    fetchrow = jnp.where(missing_row <= 2, 1, 3)
    fetch = use(row, col, face, fetchrow, 2, jnp.where(fetchrow == 1, 0, 1))
    fetch = jnp.where(jnp.any(stash), use(row, col, face, pickup_stash_row, 1, jnp.asarray(3)), fetch)
    place = use(row, col, face, missing_row, 2, jnp.asarray(2))
    clear = use(row, col, face, extra_row, 2, jnp.asarray(2))
    store = use(row, col, face, drop_stash_row, 1, jnp.asarray(3))
    watch = jnp.where(col == 2, 4, 0)
    holding = jnp.sum(inv) > 0
    act = jnp.where(holding,
                    jnp.where(jnp.any(missing), place, jnp.where(jnp.any(extra), store, watch)),
                    jnp.where(jnp.any(extra), clear, jnp.where(jnp.any(missing), fetch, watch)))
    next_mem = {'time': mem['time'] + 1, 'responses': responses, 'timeouts': timeouts,
                'prev_count': count, 'pattern': pattern, 'stable': stable,
                'empty_age': jnp.where(timeout | (code != mem['code']), 0, empty_age), 'code': code}
    return act.astype(jnp.int32), next_mem, {'recipe': recipe, 'pattern': pattern, 'code': code, 'sample': sample, 'responses': responses}


def init_agent_1():
    z = jnp.asarray(0, jnp.int32)
    return {'time': z, 'recipe': jnp.asarray(-1, jnp.int32), 'cooldown': z,
            'logits': jnp.asarray([0, -2, 0, 1, 2, 1, 2, 1], jnp.float32),
            'pattern': z, 'stable': z, 'meal_pattern': z, 'meal_clean': jnp.asarray(True),
            'last_pattern': z, 'last_kind': z, 'last_clean': jnp.asarray(True),
            'pending': jnp.asarray(False), 'delivered_at': jnp.asarray(-100, jnp.int32),
            'latency': jnp.asarray(-1, jnp.int32), 'change_seen': jnp.asarray(False),
            'certain_meal': jnp.asarray(False), 'finalized': jnp.asarray(False),
            'pulsing': jnp.asarray(False), 'blank_seen': jnp.asarray(False), 'other_seen': jnp.asarray(False)}


def agent_1_act(obs, reset, memory, rng):
    mem = {k: jnp.where(reset, v, memory[k]) for k, v in init_agent_1().items()}
    row, col = locate(obs, 22, 2, 4)
    face = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    pot = cell(obs, row, col, 2, 4)
    markers = jnp.array([jnp.any(cell(obs, row, col, r, 3)[29:34] > 0) for r in (1, 2, 3)])
    visible = col == 5
    pattern = jnp.where(visible, jnp.sum(markers.astype(jnp.int32) * jnp.array([1, 2, 4])), mem['pattern'])
    stable = jnp.where(visible, jnp.where(pattern == mem['pattern'], mem['stable'] + 1, 1), mem['stable'])
    count = jnp.sum(pot[31:34])
    tracking = (mem['recipe'] >= 0) & ~(mem['certain_meal'] & ~mem['finalized'])
    blank_seen = mem['blank_seen'] | (tracking & visible & (pattern == 0) & (mem['meal_pattern'] != 0))
    other_seen = mem['other_seen'] | (tracking & visible & (pattern != 0) & (pattern != mem['meal_pattern']))
    returned = tracking & mem['blank_seen'] & visible & (pattern == mem['meal_pattern']) & (pattern != 0)
    pulsing = mem['pulsing'] | returned
    age = mem['time'] - mem['delivered_at']
    changed = visible & (pattern != mem['last_pattern']) & mem['last_clean'] & (age >= 2) & (age <= 20) & (~pulsing | (pattern != 0))
    reaction = changed & ~mem['change_seen']
    latency = jnp.where(reaction & ~pulsing, jnp.maximum(mem['latency'], age), mem['latency'])
    grace = jnp.where(latency >= 0, jnp.minimum(20, latency + 3), 8)
    grace = jnp.where(pulsing, 8, grace)
    available = ~pulsing | (pattern != 0) | (age >= 24)
    early = (mem['recipe'] < 0) & mem['pending'] & changed
    choose = (mem['recipe'] < 0) & visible & available & (early | ((age >= grace) & (stable >= 2) & ((pattern != 0) | (mem['time'] >= 10))))
    same = pattern == mem['last_pattern']
    update = choose & mem['pending'] & mem['last_clean']
    logits = mem['logits']
    old_belief = logits[mem['last_pattern']]
    adjusted = jnp.where(same, old_belief + jnp.where(mem['last_kind'] == 0, 1., -1.), jnp.where(mem['last_kind'] == 1, 20., -20.))
    logits = logits.at[mem['last_pattern']].set(jnp.where(update, adjusted, old_belief))
    logits = logits.at[pattern].set(jnp.where(update & ~same & ~early, jnp.where(mem['last_kind'] == 0, 20., -20.), logits[pattern]))
    tie = jnp.where(mem['pending'], 1 - mem['last_kind'], 0)
    inferred = jnp.where(logits[pattern] == 0, tie, (logits[pattern] > 0).astype(jnp.int32))
    inferred = jnp.where(mem['pending'] & ~mem['last_clean'], 1 - mem['last_kind'], inferred)
    inferred = jnp.where(early, 1 - mem['last_kind'], inferred)
    recipe = jnp.where(choose, inferred, mem['recipe'])
    certain_meal = jnp.where(choose, early, mem['certain_meal'])
    finalize = certain_meal & ~mem['finalized'] & (pot[39] > 0)
    settled = visible & (stable >= 4)
    meal_pattern = jnp.where(choose | finalize, pattern, mem['meal_pattern'])
    meal_clean = jnp.where(choose | (certain_meal & ~mem['finalized']), True, mem['meal_clean'] & (~visible | (pattern == meal_pattern) | (pulsing & (pattern == 0))))
    meal_clean = jnp.where(returned & ~other_seen, True, meal_clean)
    meal_clean = jnp.where(finalize, settled, meal_clean)
    logits = logits.at[pattern].set(jnp.where(finalize & settled, jnp.where(recipe == 1, 20., -20.), logits[pattern]))
    kind = jnp.where(count > 0, jnp.argmax(pot[31:33]), jnp.maximum(recipe, 0))
    supply = use(row, col, face, 1 + 2 * kind, 6, jnp.asarray(2))
    add = use(row, col, face, 2, 5, jnp.asarray(3))
    plate = use(row, col, face, jnp.where(row > 2, 3, 1), 5, jnp.asarray(3))
    deliver = use(row, col, face, 2, 6, jnp.asarray(2))
    wait_at_pot = jnp.where((row == 2) & (col == 5) & (face == 3), 4, add)
    ready, cooking = pot[30] > 0, pot[39] > 0
    empty_act = jnp.where(ready | cooking, plate, jnp.where(recipe >= 0, supply, wait_at_pot))
    act = jnp.where(inv[1] > 0, deliver,
          jnp.where(inv[0] > 0, jnp.where(ready, add, wait_at_pot),
          jnp.where(jnp.sum(inv[2:]) > 0, add, empty_act)))
    did_deliver = (inv[1] > 0) & (row == 2) & (col == 6) & (face == 2) & (act == 5)
    next_mem = {'time': mem['time'] + 1,
                'recipe': jnp.where(did_deliver, -1, recipe),
                'cooldown': jnp.where(did_deliver, 7, jnp.maximum(0, mem['cooldown'] - 1)),
                'logits': logits, 'pattern': pattern, 'stable': stable,
                'meal_pattern': meal_pattern, 'meal_clean': meal_clean,
                'last_pattern': jnp.where(did_deliver, meal_pattern, mem['last_pattern']),
                'last_kind': jnp.where(did_deliver, kind, mem['last_kind']),
                'last_clean': jnp.where(did_deliver, meal_clean, mem['last_clean']),
                'pending': jnp.where(did_deliver, True, jnp.where(choose, False, mem['pending'])),
                'delivered_at': jnp.where(did_deliver, mem['time'], mem['delivered_at']),
                'latency': latency, 'change_seen': jnp.where(did_deliver, False, mem['change_seen'] | reaction),
                'certain_meal': certain_meal, 'finalized': jnp.where(choose, False, mem['finalized'] | finalize),
                'pulsing': pulsing, 'blank_seen': jnp.where(choose | finalize, False, blank_seen),
                'other_seen': jnp.where(choose | finalize, False, other_seen)}
    return act.astype(jnp.int32), next_mem, {'recipe': recipe, 'position': jnp.array([row, col]), 'pattern': pattern, 'logits': logits, 'choose': choose, 'pot_count': count, 'clean': meal_clean, 'early': early, 'latency': latency, 'age': age, 'pulsing': pulsing}
