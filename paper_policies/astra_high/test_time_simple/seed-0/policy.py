"""Local-observation cook and recipe-counter signaler."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'timed_stocking_adaptive_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def init_agent_0():
    # clock, most recently requested pattern, observed-full-pot latch, stable age
    return {'state': jnp.array([0, 0, 0, 0], jnp.int32),
            'responses': jnp.full((8,), -1, jnp.int32)}


def convention_prior():
    # Eight possible occupancy patterns on the three dividing counters.
    prior = jnp.full((8, 8), 0.0001, jnp.float32)
    prior = prior.at[1, 4].set(0.64).at[4, 1].set(0.06)
    prior = prior.at[1, 2].set(0.045).at[2, 4].set(0.025)
    prior = prior.at[2, 1].set(0.01).at[4, 2].set(0.01)
    prior = prior.at[0, 2].set(0.04).at[2, 0].set(0.01)
    prior = prior.at[1, 0].set(0.025).at[0, 1].set(0.015)
    prior = prior.at[0, 4].set(0.025).at[4, 0].set(0.015)
    prior = prior.at[jnp.arange(8), jnp.arange(8)].set(0.008)
    return jnp.repeat(prior[:, :, None], 2, axis=2) / (2 * jnp.sum(prior))


def init_agent_1():
    # clock, recipe, since delivery, committed, last mask, stability,
    # ever saw a nonempty signal, last dish, current cycle mask.
    return {'state': jnp.array([0, 0, 100, 0, -1, 0, 0, 1, 0, 0], jnp.int32),
            'belief': convention_prior(), 'counters': jnp.zeros((4, 5), jnp.int32)}


def locate(obs, left):
    channel = 23 if left else 22
    index = jnp.argmax(obs[:, :, channel])
    row = 4 - index // 5
    col = (2 if left else 6) - index % 5
    return row, col


def cell(obs, row, col, gr, gc):
    lr, lc = gr - row + 2, gc - col + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    return jnp.where(visible, obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)], jnp.zeros((40,), jnp.int32))


def move_to(row, col, tr, tc):
    return jnp.where(row < tr, 1, jnp.where(row > tr, 3,
           jnp.where(col < tc, 0, jnp.where(col > tc, 2, 4))))


def use(obs, row, col, sr, sc, direction):
    face = jnp.argmax(obs[2, 2, 1:5])
    turn = jnp.array([3, 1, 0, 2], dtype=jnp.int32)[direction]
    return jnp.where((row == sr) & (col == sc), jnp.where(face == direction, 5, turn), move_to(row, col, sr, sc))


def agent_0_act(obs, reset, memory, rng):
    initial = init_agent_0()
    state = jnp.where(reset, initial['state'], memory['state'])
    responses = jnp.where(reset, initial['responses'], memory['responses'])
    row, col = locate(obs, True)
    recipe = jnp.argmax(jnp.sum(obs[:, :, 36:39], axis=(0, 1)))
    bits = jnp.array([1, 2, 4], jnp.int32)
    have = jnp.array([jnp.sum(cell(obs, row, col, r, 3)[29:34]) > 0 for r in (1, 2, 3)])
    mask = jnp.sum(have.astype(jnp.int32) * bits)
    stable = jnp.where(mask == state[1], state[3] + 1, 0)
    pot = cell(obs, row, col, 2, 4)
    count = jnp.sum(pot[31:34])
    full = count >= 3
    pure = jnp.max(pot[31:33]) == 3
    learn = full & pure & (state[2] == 0) & (stable >= 4)
    cooked_type = jnp.argmax(pot[31:34])
    responses = responses.at[state[1]].set(jnp.where(learn, cooked_type, responses[state[1]]))
    # Prefer a demonstrated response, otherwise try the natural spatial signals.
    priorities = jnp.where(recipe == 0,
        jnp.array([4, 8, 6, 3, 7, 2, 1, 0]),
        jnp.array([4, 7, 6, 1, 8, 2, 3, 0]))
    allowed = (jnp.arange(8) == 0) | (jnp.arange(8) == 1) | (jnp.arange(8) == 2) | (jnp.arange(8) == 4)
    exhausted = ~jnp.any(allowed & ((responses == recipe) | (responses < 0)))
    responses = jnp.where(exhausted, jnp.full_like(responses, -1), responses)
    scores = jnp.where(allowed, priorities + jnp.where(responses == recipe, 100, jnp.where(responses < 0, 50, 0)), -1000)
    desired = jnp.argmax(scores)
    want = (desired & bits) > 0
    bad, missing = have & ~want, want & ~have
    badrow, newrow = 1 + jnp.argmax(bad), 1 + jnp.argmax(missing)
    held = jnp.sum(obs[2, 2, 5:10]) > 0
    supplyrow = jnp.where(newrow == 1, 1, jnp.where(newrow == 3, 3, jnp.where(row <= 2, 1, 3)))
    acquire = use(obs, row, col, supplyrow, 2, jnp.where(supplyrow == 1, 0, 1))
    clean = use(obs, row, col, badrow, 2, 2)
    place = use(obs, row, col, newrow, 2, 2)
    action = jnp.where(held, jnp.where(jnp.any(missing), place, 4),
                       jnp.where(jnp.any(bad), clean, jnp.where(jnp.any(missing), acquire, 4)))
    next_state = jnp.array([state[0]+1, desired, jnp.where(count == 0, 0, state[2] | learn),
                            jnp.where(desired == state[1], stable, 0)], jnp.int32)
    return action.astype(jnp.int32), {'state': next_state, 'responses': responses}, {
        'recipe': recipe, 'signal': desired, 'learn': learn,
        'responses': responses, 'position': jnp.array([row, col])}


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    state = jnp.where(reset, initial['state'], memory['state'])
    belief = jnp.where(reset, initial['belief'], memory['belief'])
    remembered_counters = jnp.where(reset, initial['counters'], memory['counters'])
    row, col = locate(obs, False)
    counters = jnp.array([[0, 5], [0, 6], [4, 5], [4, 6]], jnp.int32)
    counter_visible = (jnp.abs(counters[:, 0] - row) <= 2) & (jnp.abs(counters[:, 1] - col) <= 2)
    observed_counters = jnp.stack([cell(obs, row, col, rr, cc)[29:34] for rr, cc in ((0, 5), (0, 6), (4, 5), (4, 6))])
    remembered_counters = jnp.where(counter_visible[:, None], observed_counters, remembered_counters)
    pot = cell(obs, row, col, 2, 4)
    inv = obs[2, 2, 5:10]
    occupied = jnp.array([jnp.sum(cell(obs, row, col, r, 3)[29:34]) > 0 for r in (1, 2, 3)])
    mask = jnp.sum(occupied.astype(jnp.int32) * jnp.array([1, 2, 4], jnp.int32))
    visible = col == 5
    stable = jnp.where(visible, jnp.where(mask == state[4], state[5] + 1, 1), 0)
    seen = state[6] | (visible & (mask > 0))
    count = jnp.sum(pot[31:34])
    has_raw = jnp.sum(inv[2:]) > 0
    has_plate = inv[0] > 0
    has_dish = inv[1] > 0
    # A blank board may be either an intentional symbol or a transfer in progress.
    ready_signal = (stable >= jnp.where(jnp.sum(occupied) == 1, 1, 2)) & (state[2] >= 3) & ((mask > 0) | ((state[2] >= 8) & (state[0] >= 9)))
    decide = (state[3] == 0) & visible & ready_signal & (count == 0) & ~has_raw & ~has_plate
    patterns0 = jnp.arange(8)[:, None]
    patterns1 = jnp.arange(8)[None, :]
    likelihood = jnp.stack((jnp.broadcast_to(patterns0 == mask, (8, 8)), jnp.broadcast_to(patterns1 == mask, (8, 8))), axis=2)
    posterior = (belief + convention_prior() * 0.0001) * likelihood
    posterior = posterior / jnp.maximum(jnp.sum(posterior), 1e-12)
    belief = jnp.where(decide, posterior, belief)
    # The target cannot resample while this sole pot is cooking. A stable new
    # symbol here is evidence of a delayed request or corrective feedback.
    cooking_type = jnp.argmax(pot[31:34])
    correction = visible & (count >= 3) & (state[3] > 0) & (state[9] == 0) & (mask != state[8]) & (stable >= jnp.where(mask == 0, 8, 3))
    corrected_target = 1 - cooking_type
    code0 = jnp.where(corrected_target == 0, mask, state[8])
    code1 = jnp.where(corrected_target == 1, mask, state[8])
    corrected_belief = jnp.zeros_like(belief).at[code0, code1, corrected_target].set(1.0)
    belief = jnp.where(correction, corrected_belief, belief)
    rejected = (state[9] > 0) | correction
    p1 = jnp.sum(belief[:, :, 1])
    chosen = jnp.where(seen, (p1 > 0.5).astype(jnp.int32), 1 - state[7])
    recipe = jnp.where(decide, chosen, state[1])
    committed = (state[3] > 0) | decide
    ingredient = jnp.where(count > 0, jnp.argmax(pot[31:34]), recipe)
    supply_row = jnp.where(ingredient == 0, 1, 3)
    get_ingredient = use(obs, row, col, supply_row, 6, 2)
    stock_index = jnp.where(ingredient == 0, 0, 2)
    stocked = (remembered_counters[stock_index, 1] == 0) & (remembered_counters[stock_index, 2 + ingredient] == 1)
    get_stock = use(obs, row, col, supply_row, 5, jnp.where(ingredient == 0, 0, 1))
    get_ingredient = jnp.where(stocked, get_stock, get_ingredient)
    at_pot = use(obs, row, col, 2, 5, 3)
    plate_row = jnp.where(row <= 2, 1, 3)
    get_plate = use(obs, row, col, plate_row, 5, 3)
    deliver = use(obs, row, col, 2, 6, 2)
    counters = jnp.array([[0, 5], [0, 6], [4, 5], [4, 6]], jnp.int32)
    stands = jnp.array([[1, 5], [1, 6], [3, 5], [3, 6]], jnp.int32)
    empty_counter = jnp.sum(remembered_counters, axis=1) == 0
    costs = jnp.abs(stands[:, 0] - row) + jnp.abs(stands[:, 1] - col) + jnp.where(empty_counter, 0, 100)
    storage = jnp.argmin(costs)
    storage_dir = jnp.where(storage < 2, 0, 1)
    stash = use(obs, row, col, stands[storage, 0], stands[storage, 1], storage_dir)
    can_stash = jnp.any(empty_counter)
    dish_action = jnp.where(rejected & can_stash, stash, deliver)
    # Reserve the closest upper/lower counters for one future ingredient each.
    missing_stock = jnp.array([empty_counter[0], empty_counter[2]])
    stock_type = jnp.argmax(missing_stock)
    stock_row = jnp.where(stock_type == 0, 1, 3)
    collect_stock = use(obs, row, col, stock_row, 6, 2)
    stock_budget = jnp.abs(row - stock_row) + jnp.abs(col - 6) + 9
    stock_job = jnp.where(jnp.any(missing_stock) & (pot[39] >= stock_budget), collect_stock, get_plate)
    held_type = jnp.argmax(inv[2:])
    put_stock = use(obs, row, col, jnp.where(held_type == 0, 1, 3), 5, jnp.where(held_type == 0, 0, 1))
    raw_job = jnp.where(count >= 3, put_stock, at_pot)
    next_job = jnp.where(count >= 3, stock_job, jnp.where((count == 0) & ~committed, move_to(row, col, 2, 5), get_ingredient))
    action = jnp.where(has_dish, dish_action, jnp.where(has_plate, at_pot, jnp.where(has_raw, raw_job, next_job)))
    delivered = has_dish & (row == 2) & (col == 6) & (action == 5) & (jnp.argmax(obs[2, 2, 1:5]) == 2)
    stashed = has_dish & rejected & can_stash & (row == stands[storage, 0]) & (col == stands[storage, 1]) & (action == 5) & (jnp.argmax(obs[2, 2, 1:5]) == storage_dir)
    finished = delivered | stashed
    dish = jnp.argmax(inv[2:])
    # A correct delivery resamples uniformly; an incorrect one preserves target.
    mass0, mass1 = belief[:, :, 0], belief[:, :, 1]
    updated0 = jnp.where(dish == 0, mass0 * 0.5, mass0 + mass1 * 0.5)
    updated1 = jnp.where(dish == 0, mass1 + mass0 * 0.5, mass1 * 0.5)
    belief = jnp.where(delivered, jnp.stack((updated0, updated1), axis=2), belief)
    next_state = jnp.array([state[0]+1, recipe, jnp.where(delivered, 0, jnp.where(stashed, 100, state[2]+1)),
                           jnp.where(finished, 0, committed), jnp.where(visible, mask, state[4]),
                           stable, seen, jnp.where(delivered, dish, state[7]), jnp.where(decide, mask, state[8]), jnp.where(finished | decide, 0, rejected)], jnp.int32)
    return action.astype(jnp.int32), {'state': next_state, 'belief': belief, 'counters': remembered_counters}, {
        'recipe': recipe, 'signal': mask, 'position': jnp.array([row, col]),
        'pot_count': count, 'pot_timer': pot[39], 'delivered': delivered,
        'p1': p1, 'decide': decide, 'dish': dish, 'correction': correction, 'stashed': stashed, 'stocked': stocked, 'counters': remembered_counters}
