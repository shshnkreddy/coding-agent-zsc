"""Adaptive cooperation using only local counter and pot observations.

The recipe-side seat learns which public marker elicits each ingredient.
The kitchen-side seat learns cue meanings and response times, continues
cooking after delivery, and recovers stored dishes when they become useful.
All episode state is private, fixed-shape JAX memory.
"""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'adaptive_public_cues', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def init_agent_0():
    return {'state': jnp.zeros((5,), dtype=jnp.int32),
            'response': jnp.full((4,), -1, dtype=jnp.int32),
            'trials': jnp.zeros((4,), dtype=jnp.int32)}


def init_agent_1():
    # Clock, dish, cooldown, previous action/inventory, signal tracking,
    # whether any marker has been seen, and the delivered dish's cue.
    return {'state': jnp.asarray([0, 0, 0, 4, 0, -1, 0, -1, 0, -1, -1, 0, 100, 0, -1, 0], jnp.int32),
            'meaning': jnp.asarray([-1, -3, 1, 1, 3, 1, 1, 1], jnp.int32),
            'latency': jnp.zeros((8,), jnp.int32),
            'stash': jnp.zeros((4,), jnp.int32), 'discard': jnp.asarray(0, jnp.int32),
            'from_cache': jnp.asarray(0, jnp.int32)}


def position(obs, seat):
    channel = 23 if seat == 0 else 22
    index = jnp.argmax(obs[:, :, channel].reshape(-1))
    anchor_col = 0 if seat == 0 else 4
    return 4 - index // 5, anchor_col + 2 - index % 5


def at(obs, row, col, r, c, channel):
    i, j = row - r + 2, col - c + 2
    valid = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    return jnp.where(valid, obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4), channel], 0)


def navigate(r, c, direction, tr, tc, face, interact=True):
    turn = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)[face]
    finish = jnp.where(direction == face, 5 if interact else 4, turn)
    return jnp.where(r < tr, 1, jnp.where(r > tr, 3,
           jnp.where(c < tc, 0, jnp.where(c > tc, 2, finish)))).astype(jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    initial = init_agent_0()
    state = jnp.where(reset, initial['state'], memory['state'])
    response = jnp.where(reset, initial['response'], memory['response'])
    trials = jnp.where(reset, initial['trials'], memory['trials'])
    t, previous_total, age, last_symbol, idle = state
    r, c = position(obs, 0)
    direction = jnp.argmax(obs[2, 2, 1:5])
    recipe = jnp.argmax(jnp.sum(obs[:, :, 36:39], axis=(0, 1)))
    markers = jnp.asarray([at(obs, row, 3, r, c, 33) > 0 for row in (1, 2, 3)])
    present = jnp.any(markers)
    symbol = jnp.where(present, jnp.argmax(markers) + 1, 0)
    age = jnp.where(symbol == last_symbol, age + 1, 0)
    have = obs[2, 2, 9] > 0
    counts = jnp.asarray([at(obs, 2, 4, r, c, x) for x in (31, 32, 33)])
    total = jnp.sum(counts)
    visible = c == 2
    new_pot = visible & (total > 0) & (previous_total == 0)
    learn = new_pot & (age >= 4) & (counts[2] == 0)
    reversal = learn & (response[symbol] >= 0) & (response[symbol] != jnp.argmax(counts))
    response = jnp.where(reversal, jnp.full_like(response, -1), response)
    trials = jnp.where(reversal, jnp.zeros_like(trials), trials)
    response = response.at[symbol].set(jnp.where(learn, jnp.argmax(counts), response[symbol]))
    trials = trials.at[symbol].add(learn.astype(jnp.int32))
    previous_total = jnp.where(visible, total, previous_total)
    idle = jnp.where((previous_total == 0) & (symbol == last_symbol), idle + 1, 0)
    stalled = (idle >= 32) & (age >= 20)
    response = response.at[symbol].set(jnp.where(stalled, -1, response[symbol]))
    trials = trials.at[symbol].add(stalled.astype(jnp.int32))
    idle = jnp.where(stalled, 0, idle)

    # Start with the spatially aligned code, but prefer a symbol whose
    # observed effect is correct over an untested convention.
    rank = jnp.where(recipe == 0, jnp.asarray([3, 0, 2, 1]), jnp.asarray([3, 1, 2, 0]))
    costs = jnp.where(response == recipe, rank,
                     jnp.where(response < 0, 10 + 4 * trials + rank, 100 + 4 * trials + rank))
    desired = jnp.argmin(costs)
    wrong = present & (symbol != desired)
    get_wrong = navigate(r, c, direction, jnp.maximum(symbol, 1), 2, 2)
    put_right = navigate(r, c, direction, jnp.maximum(desired, 1), 2, 2)
    source_row = jnp.where(desired == 3, 3, 1)
    source_face = jnp.where(desired == 3, 1, 0)
    get_source = navigate(r, c, direction, source_row, 2, source_face)
    observe = navigate(r, c, direction, jnp.where(symbol > 0, symbol, 2), 2, 2, False)
    action = jnp.where(wrong, get_wrong,
             jnp.where(desired == 0, observe,
             jnp.where(symbol == desired, observe, jnp.where(have, put_right, get_source))))
    next_state = jnp.asarray([t + 1, previous_total, age, symbol, idle], jnp.int32)
    return action, {'state': next_state, 'response': response, 'trials': trials}, {
           'recipe': recipe, 'desired': desired, 'response': response,
           'position': jnp.asarray([r, c]), 'signal': markers}


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    state = jnp.where(reset, initial['state'], memory['state'])
    meaning = jnp.where(reset, initial['meaning'], memory['meaning'])
    stash = jnp.where(reset, initial['stash'], memory['stash'])
    discard = jnp.where(reset, initial['discard'], memory['discard'])
    latency = jnp.where(reset, initial['latency'], memory['latency'])
    from_cache = jnp.where(reset, initial['from_cache'], memory['from_cache'])
    (t, chosen, delay, previous_action, previous_cooked, last_sig,
     streak, stable, seen, cycle_sig, delivered_sig, delivered_dish, since_delivery, changed_seen, pending_target, unchanged_run) = state
    r, c = position(obs, 1)
    direction = jnp.argmax(obs[2, 2, 1:5])
    plate, cooked = obs[2, 2, 5], obs[2, 2, 6]
    from_cache = jnp.where(cooked > 0, from_cache, 0)
    held = obs[2, 2, 7:10]
    raw = (jnp.sum(held) > 0) & (cooked == 0)
    empty = (plate == 0) & (jnp.sum(held) == 0)
    counts = jnp.asarray([at(obs, 2, 4, r, c, x) for x in (31, 32, 33)])
    total = jnp.sum(counts)
    ready = at(obs, 2, 4, r, c, 30) > 0
    timer = at(obs, 2, 4, r, c, 39)

    # All three writable divider cells are visible from column five.
    marks = jnp.asarray([at(obs, row, 3, r, c, 33) > 0 for row in (1, 2, 3)])
    sig = jnp.sum(marks.astype(jnp.int32) * jnp.asarray([1, 2, 4], jnp.int32))
    visible = c == 5
    streak = jnp.where(visible, jnp.where(sig == last_sig, streak + 1, 1), streak)
    last_sig = jnp.where(visible, sig, last_sig)
    seen = jnp.maximum(seen, (visible & (sig > 0)).astype(jnp.int32))
    settled = visible & (streak >= jnp.where(sig == 0, 5, 2))
    stable = jnp.where(settled, sig, stable)

    delivered = (previous_cooked > 0) & (cooked == 0) & (previous_action == 5) & (r == 2) & (c == 6)
    delivered_sig = jnp.where(delivered, stable, delivered_sig)
    delivered_dish = jnp.where(delivered, chosen, delivered_dish)
    since_delivery = jnp.where(delivered, 0, jnp.minimum(since_delivery + 1, 100))
    changed_seen = jnp.where(delivered, 0, changed_seen)
    pending_target = jnp.where(delivered, -1, pending_target)
    old_index = jnp.clip(delivered_sig, 0, 7)
    wait_budget = jnp.where(latency[old_index] > 0, latency[old_index], 8)
    wait_budget = jnp.where(seen > 0, wait_budget, 1)

    # Before any new ingredient reaches the pot, a change in the public
    # recipe cue is evidence of a correct delivery and the opposite target.
    hint = visible & (delivered_sig >= 0) & (sig != delivered_sig) & (changed_seen == 0)
    hint = hint & (since_delivery > 0) & (since_delivery <= 24) & (total == 0) & (plate == 0)
    pending_target = jnp.where(hint, 1 - delivered_dish, pending_target)
    changed_seen = jnp.where(hint, 1, changed_seen)
    observed_latency = jnp.clip(since_delivery + 1, 2, 16)
    latency = latency.at[old_index].set(jnp.where(hint,
                 jnp.maximum(latency[old_index], observed_latency), latency[old_index]))
    process = (delay == 1) & ~delivered & ~hint & (changed_seen == 0)
    delay = jnp.where(delivered, wait_budget, jnp.where(hint, 0, jnp.maximum(delay - 1, 0)))
    chosen = jnp.where(delivered, 1 - chosen, chosen)

    # Unchanged cues weaken an unconfirmed interpretation. Confirmed labels
    # are retained until a long uninformative run suggests a stale signal.
    evidence = process & (delivered_sig >= 0) & (seen > 0)
    old_meaning = meaning[old_index]
    unchanged_run = jnp.where(hint, 0, unchanged_run + evidence.astype(jnp.int32))
    stale = evidence & (unchanged_run >= 4) & (jnp.abs(old_meaning) >= 32)
    old_meaning = jnp.where(stale, jnp.where(delivered_dish == 0, -1, 1), old_meaning)
    unchanged_run = jnp.where(stale, 0, unchanged_run)
    no_change = jnp.clip(old_meaning + jnp.where(delivered_dish == 0, 2, -2), -8, 8)
    no_change = jnp.where(jnp.abs(old_meaning) >= 32, old_meaning, no_change)
    old_meaning = jnp.where(evidence, no_change, old_meaning)
    old_meaning = jnp.where(hint, jnp.where(delivered_dish == 1, 64, -64), old_meaning)
    meaning = meaning.at[old_index].set(old_meaning)
    opposite = jnp.where(old_index == 1, 4, 1)
    spatial = (evidence | hint) & ((old_index == 1) | (old_index == 4))
    meaning = meaning.at[opposite].set(jnp.where(spatial, -old_meaning, meaning[opposite]))
    # The marker can be in transit while ingredients are collected. Assign
    # the new label only to a settled observed symbol.
    new_index = jnp.clip(stable, 0, 7)
    teach_new = (pending_target >= 0) & settled & (stable != delivered_sig) & (stable >= 0)
    meaning = meaning.at[new_index].set(jnp.where(teach_new,
                    jnp.where(pending_target == 1, 64, -64), meaning[new_index]))

    # A partner can revise its cue after observing our first pot ingredient.
    # Beyond the ordinary post-delivery window, a settled replacement is
    # treated as a correction of this committed dish, not a new recipe draw.
    committed_type = jnp.argmax(counts)
    correction = settled & (total > 0) & (counts[2] == 0) & (pending_target < 0)
    correction = correction & (since_delivery > 24) & (stable != cycle_sig) & (cycle_sig >= 0)
    correction = correction & ((cycle_sig > 0) | (jnp.abs(meaning[0]) >= 32))
    correction = correction & ((stable > 0) | (streak >= 10))
    pending_target = jnp.where(correction, 1 - committed_type, pending_target)
    unchanged_run = jnp.where(correction, 0, unchanged_run)
    correction_value = jnp.where(committed_type == 0, 64, -64)
    meaning = meaning.at[new_index].set(jnp.where(correction, correction_value, meaning[new_index]))
    correction_opposite = jnp.where(new_index == 1, 4, 1)
    correction_spatial = correction & ((new_index == 1) | (new_index == 4))
    meaning = meaning.at[correction_opposite].set(jnp.where(correction_spatial,
                     -correction_value, meaning[correction_opposite]))

    selecting = (total == 0) & empty & (delay == 0)
    has_cue = (seen > 0) & (stable >= 0)
    decoded = (meaning[jnp.clip(stable, 0, 7)] > 0).astype(jnp.int32)
    chosen = jnp.where(selecting & has_cue, decoded, chosen)
    chosen = jnp.where(selecting & (pending_target >= 0), pending_target, chosen)
    cycle_sig = jnp.where(selecting & visible, stable, cycle_sig)
    chosen = jnp.where((total > 0) & (total < 3), jnp.argmax(counts), chosen)

    supply = navigate(r, c, direction, 1 + 2 * chosen, 6, 2)
    pot = navigate(r, c, direction, 2, 5, 3)
    plate_row = jnp.where(r <= 2, 1, 3)
    get_plate = navigate(r, c, direction, plate_row, 5, 3)
    deliver = navigate(r, c, direction, 2, 6, 2)
    # A later stable cue can expose a mistake after the pot is committed.
    # Store that dish on an unused outer counter instead of delivering it.
    expected = jnp.where(pending_target >= 0, pending_target, decoded)
    contradiction = (stable != cycle_sig) & has_cue & (expected != jnp.argmax(held))
    contradiction = contradiction & ((stable > 0) | (jnp.abs(meaning[0]) >= 32))
    discard = jnp.where(cooked > 0, jnp.maximum(discard, contradiction.astype(jnp.int32)), 0)
    stash_rows = jnp.asarray([1, 3, 1, 3], jnp.int32)
    stash_cols = jnp.asarray([5, 5, 6, 6], jnp.int32)
    stash_faces = jnp.asarray([0, 1, 0, 1], jnp.int32)
    distances = jnp.abs(stash_rows - r) + jnp.abs(stash_cols - c)
    dump_index = jnp.argmin(jnp.where(stash == 0, distances, 100))
    dump = navigate(r, c, direction, stash_rows[dump_index], stash_cols[dump_index], stash_faces[dump_index])
    discarding = (discard > 0) & jnp.any(stash == 0)
    deliver = jnp.where(discarding, dump, deliver)
    watch = navigate(r, c, direction, 2, 5, 3, False)
    matching_cache = stash == (chosen + 1)
    cache_index = jnp.argmin(jnp.where(matching_cache, distances, 100))
    get_cache = navigate(r, c, direction, stash_rows[cache_index], stash_cols[cache_index], stash_faces[cache_index])
    use_cache = selecting & jnp.any(matching_cache)
    # An already-cooked dish can finish much sooner than a fresh cook. Give
    # the sender time to finish its current cue before resampling again.
    cue_finished = (stable > 0) & (stable != delivered_sig) & (decoded == pending_target)
    cache_wait = (from_cache > 0) & ((since_delivery < 10) |
                 ((pending_target >= 0) & ~cue_finished & (since_delivery < 16)))
    deliver = jnp.where(cache_wait & ~discarding, watch, deliver)
    waiting = (delay > 0) | ((t < 12) & ~has_cue)
    action = jnp.where(cooked > 0, deliver, jnp.where(raw, pot,
             jnp.where(plate > 0, jnp.where(ready, pot, watch),
             jnp.where((timer > 0) | ready, get_plate,
             jnp.where(waiting, watch, jnp.where(use_cache, get_cache, supply))))))
    dropped = discarding & (cooked > 0) & (action == 5)
    stash = stash.at[dump_index].set(jnp.where(dropped, jnp.argmax(held) + 1, stash[dump_index]))
    collected = use_cache & ~waiting & (action == 5)
    stash = stash.at[cache_index].set(jnp.where(collected, 0, stash[cache_index]))
    from_cache = jnp.where(collected, 1, from_cache)
    next_state = jnp.asarray([t + 1, chosen, delay, action, cooked, last_sig,
                            streak, stable, seen, cycle_sig, delivered_sig, delivered_dish, since_delivery, changed_seen, pending_target, unchanged_run], jnp.int32)
    return action, {'state': next_state, 'meaning': meaning, 'stash': stash, 'discard': discard, 'latency': latency, 'from_cache': from_cache}, {
        'chosen': chosen, 'position': jnp.asarray([r, c]), 'signal': sig,
        'stable': stable, 'meaning': meaning, 'pot': counts, 'delay': delay, 'discard': discard, 'pending': pending_target, 'latency': latency, 'correction': correction, 'unchanged': unchanged_run, 'stash': stash, 'from_cache': from_cache}
