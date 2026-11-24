"""Locally observed recipe signaling and repeated cooking."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'spatial_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def position(obs, channel, landmark_row, landmark_col):
    flat = jnp.argmax(obs[:, :, channel])
    return jnp.asarray([landmark_row + 2 - flat // 5,
                        landmark_col + 2 - flat % 5], jnp.int32)


def cell(obs, pos, row, col):
    r, c = row - pos[0] + 2, col - pos[1] + 2
    valid = (r >= 0) & (r < 5) & (c >= 0) & (c < 5)
    return jnp.where(valid, obs[jnp.clip(r, 0, 4), jnp.clip(c, 0, 4)], jnp.zeros((40,), jnp.int32))


def approach(obs, pos, row, col, direction, interact=True):
    facing = jnp.argmax(obs[2, 2, 1:5])
    turn = jnp.asarray([3, 1, 0, 2], jnp.int32)[direction]
    arrival = jnp.where(facing == direction, 5 if interact else 4, turn)
    action = jnp.where(pos[1] < col, 0, jnp.where(pos[1] > col, 2, arrival))
    return jnp.where(pos[0] < row, 1, jnp.where(pos[0] > row, 3, action)).astype(jnp.int32)


def init_agent_0():
    return {'state': jnp.zeros((4,), jnp.int32),
            'responses': jnp.zeros((4, 2), jnp.int32),
            'unresponsive': jnp.zeros((4,), jnp.bool_)}


def agent_0_act(obs, reset, memory, rng):
    initial = init_agent_0()
    memory = {k: jnp.where(reset, initial[k], memory[k]) for k in initial}
    last_n, last_marker, marker_age, idle_age = memory['state']
    responses, unresponsive = memory['responses'], memory['unresponsive']
    pos = position(obs, 23, 2, 0)
    recipe = jnp.argmax(jnp.sum(obs[:, :, 36:39], axis=(0, 1)))
    occupied = jnp.asarray([jnp.sum(cell(obs, pos, r, 3)[29:34]) > 0 for r in (1,2,3)])
    marker = jnp.where(jnp.any(occupied), jnp.argmax(occupied)+1, 0)
    marker_age = jnp.where(marker == last_marker, marker_age + 1, 0)
    pot = cell(obs, pos, 2, 4)
    n = jnp.sum(pot[31:34])
    observed = pot[22] > 0
    first_ingredient = observed & (last_n == 0) & (n > 0) & (marker_age >= 6)
    response = jnp.argmax(pot[31:33])
    responses = responses.at[marker, response].add(first_ingredient.astype(jnp.int32))
    last_n = jnp.where(observed, n, last_n)
    idle_age = jnp.where(observed, jnp.where(n == 0, idle_age+1, 0), idle_age)
    stalled = observed & (idle_age >= 35) & (marker_age >= 20)
    unresponsive = unresponsive.at[marker].set(unresponsive[marker] | stalled)
    established = jnp.abs(responses[:,0]-responses[:,1]) >= 1
    predicted = jnp.argmax(responses, axis=1)
    priority = jnp.where(recipe == 0, jnp.asarray([3,20,7,10]), jnp.asarray([3,10,7,20]))
    priority = priority.at[2].add(jnp.where(jnp.any(unresponsive), 20, 0))
    scores = priority + jnp.where(established, jnp.where(predicted == recipe, 100, -100), 0) - 200*unresponsive
    target = jnp.argmax(scores)
    held = jnp.sum(obs[2, 2, 5:10]) > 0
    drop = approach(obs, pos, jnp.maximum(target, 1), 2, 2)
    reclaim = approach(obs, pos, jnp.maximum(marker, 1), 2, 2)
    supply_row = jnp.where(target <= 2, 1, 3)
    supply = approach(obs, pos, supply_row, 2, jnp.where(supply_row == 1, 0, 1))
    correct = marker == target
    action = jnp.where(correct, 4, jnp.where(held, jnp.where(target > 0, drop, 4), jnp.where(marker > 0, reclaim, supply)))
    next_memory = {'state': jnp.asarray([last_n, marker, marker_age, idle_age], jnp.int32), 'responses': responses, 'unresponsive': unresponsive}
    return action.astype(jnp.int32), next_memory, {'recipe': recipe, 'pos': pos, 'marker': marker, 'target': target, 'responses': responses}


def init_agent_1():
    # stage: 0 settle/read, 1 collect and fill, 2 plate, 3 deliver
    return {'state': jnp.asarray([0, 0, 0, 0, 0, 0, 0], jnp.int32),
            'belief': jnp.asarray([0.5, 0.8, 0.5, 0.7, 0.2, 0.5, 0.3, 0.5], jnp.float32),
            'known': jnp.zeros((8,), jnp.bool_)}


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    memory = {k: jnp.where(reset, initial[k], memory[k]) for k in initial}
    stage, recipe, age, last_signal, deliveries, seen_signal, signal_age = memory['state']
    belief, known = memory['belief'], memory['known']
    pos = position(obs, 22, 2, 4)
    upper = jnp.sum(cell(obs, pos, 1, 3)[29:34]) > 0
    lower = jnp.sum(cell(obs, pos, 3, 3)[29:34]) > 0
    middle = jnp.sum(cell(obs, pos, 2, 3)[29:34]) > 0
    signal = upper.astype(jnp.int32) + 2 * middle.astype(jnp.int32) + 4 * lower.astype(jnp.int32)
    visible = pos[1] == 5
    signal_age = jnp.where(visible, jnp.where(signal == seen_signal, signal_age+1, 0), signal_age)
    seen_signal = jnp.where(visible, signal, seen_signal)
    pot = cell(obs, pos, 2, 4)
    counts = pot[31:34]
    n = jnp.sum(counts)
    inv = obs[2, 2, 5:10]
    held_raw = (jnp.sum(inv[2:]) > 0) & (inv[1] == 0)
    plate = inv[0] > 0
    dish = inv[1] > 0
    # Select the next batch after allowing publicly visible signals to settle.
    early = (signal > 0) & (signal_age >= 1) & (((deliveries == 0) & (age >= 4)) | (known[signal] & (age >= 3)))
    ready = (stage == 0) & ((age >= 7) | early) & visible
    # A changed stable signal after delivery proves the old recipe was correct.
    # For an unchanged signal, a correct dish would preserve the signal with
    # probability one half; an incorrect dish necessarily preserves it.
    changed = signal != last_signal
    update = ready & (deliveries > 0)
    old_p = belief[last_signal]
    p_dish = jnp.where(recipe == 0, old_p, 1.0-old_p)
    posterior = p_dish / (2.0-p_dish)
    unchanged_p = jnp.where(recipe == 0, posterior, 1.0-posterior)
    learned_old = jnp.where(changed, (recipe == 0).astype(jnp.float32),
                            jnp.where(known[last_signal], old_p, unchanged_p))
    belief = belief.at[last_signal].set(jnp.where(update, learned_old, old_p))
    belief = belief.at[signal].set(jnp.where(update & changed, (recipe == 1).astype(jnp.float32), belief[signal]))
    known = known.at[last_signal].set(known[last_signal] | (update & changed))
    known = known.at[signal].set(known[signal] | (update & changed))
    tie_choice = jnp.where(deliveries > 0, 1-recipe, 0)
    choice = jnp.where(belief[signal] > 0.50001, 0, jnp.where(belief[signal] < 0.49999, 1, tie_choice))
    recipe = jnp.where(ready, choice, recipe)
    last_signal = jnp.where(ready, signal, last_signal)
    stage = jnp.where(ready, 1, stage)
    stage = jnp.where((stage == 1) & (n >= 3), 2, stage)
    stage = jnp.where(dish, 3, stage)
    last_signal = jnp.where((stage == 2) & (pos[1] == 5), signal, last_signal)
    row = 1 + 2 * recipe
    fetch = approach(obs, pos, row, 6, 2)
    insert = approach(obs, pos, 2, 5, 3)
    fill_action = jnp.where(held_raw, insert, fetch)
    # Nearest plate supply, then wait in front of the visible pot.
    plate_row = jnp.where(pos[0] <= 2, 1, 3)
    get_plate = approach(obs, pos, plate_row, 5, 3)
    scoop = approach(obs, pos, 2, 5, 3)
    at_pot = (pos[0] == 2) & (pos[1] == 5) & (obs[2, 2, 4] == 1)
    scoop = jnp.where(at_pot & (pot[30] == 0), 4, scoop)
    plate_action = jnp.where(plate, scoop, get_plate)
    deliver = approach(obs, pos, 2, 6, 2)
    wait = approach(obs, pos, 2, 5, 3, interact=False)
    action = jnp.where(stage == 0, wait, jnp.where(stage == 1, fill_action, jnp.where(stage == 2, plate_action, deliver)))
    delivered = (stage == 3) & (action == 5) & (pos[0] == 2) & (pos[1] == 6)
    next_stage = jnp.where(delivered, 0, stage)
    next_age = jnp.where(delivered, 0, age + 1)
    next_memory = {'state': jnp.asarray([next_stage, recipe, next_age, last_signal, deliveries + delivered, seen_signal, signal_age], jnp.int32), 'belief': belief, 'known': known}
    return action.astype(jnp.int32), next_memory, {'stage': stage, 'recipe': recipe, 'signal': signal, 'pot_n': n, 'pos': pos, 'deliveries': deliveries, 'belief': belief, 'known': known}
