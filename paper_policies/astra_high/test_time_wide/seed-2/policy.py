"""Local-observation cooking and partner guidance."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'adaptive_physical_cooperation', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}


def init_agent_0():
    return {'t': jnp.int32(0), 'world': jnp.zeros((7, 6, 7), jnp.int32),
            'recipe': jnp.int32(0), 'phase': jnp.int32(0), 'wait': jnp.int32(0),
            'offer': jnp.int32(0), 'job': jnp.int32(0),
            'cuebelief': jnp.array([2, 2, -2, -2, -2], jnp.int32), 'belief': jnp.int32(2), 'lastcue': jnp.int32(0), 'lastempty': jnp.int32(0),
            'deliverycue': jnp.int32(0), 'deliveryempty': jnp.int32(0),
            'delivered': jnp.int32(0), 'pending': jnp.int32(0),
            'courier_patience': jnp.int32(0), 'mode': jnp.int32(0), 'courier_clock': jnp.int32(0), 'partnerinv': jnp.zeros(6, jnp.int32),
            'trusted': jnp.int32(0), 'deliverytrusted': jnp.int32(0), 'idle': jnp.int32(0),
            'orientation': jnp.int32(0), 'signal': jnp.int32(0), 'stable': jnp.int32(0)}


def init_agent_1():
    return init_agent_0()


def locate(obs, seat):
    board = ('WWXBWW', '0    0', '1    1', 'WPWPWW', '3    3', 'W    W', 'WWRWWW')
    types = 'WXPRLB0123'
    candidates = [(r, c) for r in ((1, 2) if seat == 0 else (4, 5)) for c in range(1, 5)]
    templates = []
    for r, c in candidates:
        templates.append([[[int(0 <= r+i-2 < 7 and 0 <= c+j-2 < 6 and board[r+i-2][c+j-2] == ch)
                            for ch in types] for j in range(5)] for i in range(5)])
    error = jnp.sum(jnp.abs(jnp.asarray(templates, jnp.int32) - obs[None, :, :, 22:32]), axis=(1, 2, 3))
    return jnp.asarray(candidates, jnp.int32)[jnp.argmin(error)]


def observe(obs, pos, old):
    rr, cc = jnp.meshgrid(jnp.arange(7), jnp.arange(6), indexing='ij')
    lr, lc = rr-pos[0]+2, cc-pos[1]+2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    data = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)]
    old = old.at[:, :, 1].set(jnp.where(old[:, :, 6] == 1, 1, old[:, :, 1]))
    old = old.at[:, :, 6].set(jnp.maximum(0, old[:, :, 6]-1))
    fresh = jnp.concatenate((data[:, :, 32:38], data[:, :, 44:45]), axis=-1)
    return jnp.where(visible[:, :, None], fresh, old)


def move(pos, target):
    delta = target-pos
    return jnp.where(delta[1] > 0, 0, jnp.where(delta[1] < 0, 2,
           jnp.where(delta[0] > 0, 1, jnp.where(delta[0] < 0, 3, 4)))).astype(jnp.int32)


def interact(pos, facing, stance, direction):
    at = jnp.all(pos == stance)
    turn = jnp.asarray([3, 1, 0, 2], jnp.int32)[direction]
    # A horizontal final leg gives the needed facing for side supplies.
    # Both rooms are unobstructed rectangles, so either shortest axis order
    # is valid for these stance targets.
    delta = stance-pos
    vertical_first = (direction >= 2) & (delta[1] != 0) & (delta[0] != 0)
    navigation = jnp.where(vertical_first, jnp.where(delta[0] > 0, 1, 3), move(pos, stance))
    return jnp.where(at, jnp.where(facing == direction, 5, turn), navigation).astype(jnp.int32)


def pot_choice(world, ingredient, pos, seat):
    pots = world[3, jnp.array([1, 3])]
    totals = jnp.sum(pots[:, 2:6], axis=1)
    compatible = pots[:, ingredient+2] == totals
    score = totals*10 + pots[:, 1]*20 - jnp.abs(jnp.array([1, 3])-pos[1])
    # Break empty-pot ties by the natural left/right allocation.
    score = score + 10*(jnp.arange(2) == ingredient).astype(jnp.int32)
    idx = jnp.argmax(jnp.where(compatible, score, -1000))
    return idx, pots[idx], jnp.any(compatible)


def counter_action(pos, facing, col, seat):
    return interact(pos, facing, jnp.array([2 if seat == 0 else 4, col]), 1 if seat == 0 else 0)


def raw_action(pos, facing, ingredient, potcol):
    # Choose the supply by full round-trip cost, rather than current column alone.
    cols = jnp.array([1, 4])
    costs = jnp.abs(cols-pos[1]) + jnp.abs(cols-potcol)
    side = jnp.argmin(costs)
    return interact(pos, facing, jnp.array([ingredient+1, cols[side]]), jnp.where(side == 0, 3, 2))


def private_storage(world, pos, facing, seat):
    if seat == 0:
        cells = jnp.array([[0, 1], [0, 4]], jnp.int32)
        stances = jnp.array([[1, 1], [1, 4]], jnp.int32)
        directions = jnp.array([0, 0], jnp.int32)
    else:
        cells = jnp.array([[5, 0], [6, 1], [6, 3], [6, 4], [5, 5]], jnp.int32)
        stances = jnp.array([[5, 1], [5, 1], [5, 3], [5, 4], [5, 4]], jnp.int32)
        directions = jnp.array([3, 1, 1, 1, 2], jnp.int32)
    items = world[cells[:, 0], cells[:, 1], :6]
    free = jnp.sum(items, axis=1) == 0
    distance = jnp.sum(jnp.abs(stances-pos), axis=1)*2 + (directions != facing).astype(jnp.int32)
    index = jnp.argmin(jnp.where(free, distance, 1000))
    store_action = interact(pos, facing, stances[index], directions[index])
    return items, stances, directions, distance, store_action, jnp.any(free)


def fetch_raw(world, pos, facing, ingredient, potcol):
    items, stances, directions, distance, _, _ = private_storage(world, pos, facing, 0)
    wanted = (items[:, ingredient+2] > 0) & (items[:, 1] == 0)
    index = jnp.argmin(jnp.where(wanted, distance, 1000))
    return jnp.where(jnp.any(wanted), interact(pos, facing, stances[index], directions[index]),
                     raw_action(pos, facing, ingredient, potcol))


def agent_0_act(obs, reset, memory, rng):
    m = jax.tree.map(lambda x, y: jnp.where(reset, x, y), init_agent_0(), memory)
    pos = locate(obs, 0)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:11]
    empty = jnp.sum(inv) == 0
    raw = (jnp.sum(inv[2:]) > 0) & (inv[1] == 0)
    world = observe(obs, pos, m['world'])
    saved, saved_stances, saved_directions, saved_distance, store_action, store_space = private_storage(world, pos, facing, 0)
    saved_plates = (saved[:, 0] > 0) & (saved[:, 1] == 0)
    plate_index = jnp.argmin(jnp.where(saved_plates, saved_distance, 1000))
    plate_fetch = jnp.where(jnp.any(saved_plates), interact(pos, facing, saved_stances[plate_index], saved_directions[plate_index]),
                            interact(pos, facing, jnp.array([1, 3]), 0))
    partner_seen = jnp.any(obs[:, :, 11] > 0)
    pi = jnp.argmax(obs[:, :, 11])
    pc = pi % 5 + pos[1]-2
    pinv = obs.reshape(25, 45)[pi, 16:22]
    partnerinv = jnp.where(partner_seen, pinv, m['partnerinv'])
    mode = (m['mode'] > 0) & (m['courier_clock'] < 80)
    # Keep separate meanings for each observable side and for a settled
    # absence. No shared codebook is required once a signal change confirms
    # the recipe of a preceding delivery.
    missing_ready = ~partner_seen & (m['wait'] >= 8) & jnp.all(pos == jnp.array([2, 2]))
    cue = jnp.where(pc <= 2, 0, 2).astype(jnp.int32)
    cue = jnp.where(partner_seen, cue, jnp.where(missing_ready, 4, m['lastcue']))
    partner_empty = jnp.sum(pinv) == 0
    idle = jnp.where(partner_seen, jnp.where(partner_empty & (cue == m['lastcue']) & (pc != 2), m['idle']+1, 0), m['idle'])
    settling = (m['phase'] == 0) & (m['wait'] >= 4) & (partner_seen | missing_ready)
    calibrate = settling & (m['pending'] > 0) & partner_empty & (m['deliveryempty'] > 0)
    changed = cue != m['deliverycue']
    delivered_zero = m['delivered'] == 0
    grounded = settling & (m['pending'] > 0) & (m['deliverytrusted'] > 0) & (m['deliveryempty'] > 0)
    oldbelief = m['cuebelief'][m['deliverycue']]
    positive = jnp.where(delivered_zero, 16, -16)
    evidence = jnp.where(changed | grounded, positive,
                         jnp.clip(oldbelief + jnp.where(delivered_zero, -1, 1), -16, 16))
    cuebelief = m['cuebelief'].at[m['deliverycue']].set(jnp.where(calibrate | grounded, evidence, oldbelief))
    cuebelief = cuebelief.at[cue].set(jnp.where((calibrate | grounded) & changed, -positive, cuebelief[cue]))
    belief = cuebelief[cue]
    prior = jnp.array([0, 0, 1, 1, 1], jnp.int32)[cue]
    inferred = jnp.where(belief == 0, prior, (belief < 0).astype(jnp.int32))
    explicit = (jnp.sum(pinv[2:4]) > 0) & partner_seen
    inferred = jnp.where(explicit, jnp.argmax(pinv[2:4]), inferred)
    indicated_pot = world[3, jnp.clip(pc, 0, 5)]
    plate_signal = (obs.reshape(25, 45)[pi, 12] > 0) & partner_seen & (pinv[0] > 0) & (pinv[1] == 0) & ((pc == 1) | (pc == 3)) & (jnp.sum(indicated_pot[2:4]) > 0)
    inferred = jnp.where(plate_signal, jnp.argmax(indicated_pot[2:4]), inferred)
    recipe = jnp.where((m['phase'] == 0) & (partner_seen | missing_ready), inferred, m['recipe'])
    phase = jnp.where(settling, 1, m['phase'])
    idx, pot, compatible = pot_choice(world, recipe, pos, 0)
    potcol = jnp.array([1, 3])[idx]
    count = jnp.sum(pot[2:6])
    otheridx, other, otherok = pot_choice(world, 1-recipe, pos, 0)
    othercol = jnp.array([1, 3])[otheridx]
    othercount = jnp.sum(other[2:6])
    shared = world[3, jnp.array([2, 4])]
    dish_mask = (shared[:, 1] > 0) & ((shared[:, 2] == 3) | (shared[:, 3] == 3))
    dish_idx = jnp.argmax(dish_mask)
    dish_col = jnp.array([2, 4])[dish_idx]
    available = jnp.sum(shared[:, :6], axis=1) == 0
    free_col = jnp.array([2, 4])[jnp.argmax(available)]
    release = jnp.where(jnp.any(available), counter_action(pos, facing, free_col, 0),
                         jnp.where(store_space, store_action, move(pos, jnp.array([2, 2]))))
    partner_plate = partner_seen & (pinv[0] > 0)
    # Once per episode offer an ordinary plate to enable a recipe-aware courier.
    offering = (m['offer'] == 0) & (count == 3) & (pot[6] > 8) & jnp.any(available) & ~jnp.any(shared[:, 0] > 0) & ~partner_plate
    offering = offering | (m['job'] == 1)
    # During cooking, prepare the other pure recipe. A held raw item commits only
    # that item; visible pot contents determine its safe destination.
    prep_other = (count == 3) & (pot[6] > 5) & otherok & (othercount < 3) & ~jnp.any(dish_mask)
    work_recipe = jnp.where(prep_other, 1-recipe, recipe)
    work_col = jnp.where(prep_other, othercol, potcol)
    fill = fetch_raw(world, pos, facing, work_recipe, work_col)
    action = jnp.where(count < 3, fill,
        jnp.where(prep_other, fill, plate_fetch))
    courier_patience = m['courier_patience'] + ((partnerinv[0] > 0) & (pot[1] > 0)).astype(jnp.int32)
    defer_plate = empty & (count == 3) & (partnerinv[0] > 0) & (courier_patience < 25)
    defer_action = jnp.where(otherok & (othercount < 3), fetch_raw(world, pos, facing, 1-recipe, othercol), move(pos, jnp.array([2, 2])))
    action = jnp.where(defer_plate, defer_action, action)
    action = jnp.where(empty & jnp.any(dish_mask), counter_action(pos, facing, dish_col, 0), action)
    action = jnp.where(empty & partner_seen & (pinv[1] > 0) & ~jnp.any(dish_mask),
                       move(pos, jnp.array([2, 2])), action)
    # Plate already in hand: reserve it for a ready requested soup, or finish
    # the explicit offer. If the partner scooped first, release it on a counter.
    plate = (inv[0] > 0) & (inv[1] == 0)
    scoop = jnp.where(pot[1] > 0, interact(pos, facing, jnp.array([2, potcol]), 1), move(pos, jnp.array([2, potcol])))
    plate_act = jnp.where(count < 3, release, scoop)
    action = jnp.where(plate, plate_act, action)
    offer_act = jnp.where(plate, release, plate_fetch)
    action = jnp.where(offering & ~raw & (inv[1] == 0), offer_act, action)
    rawtype = jnp.argmax(inv[2:6])
    ridx, rpot, rok = pot_choice(world, rawtype, pos, 0)
    rawcol = jnp.array([1, 3])[ridx]
    can_insert = rok & (jnp.sum(rpot[2:6]) < 3) & (rpot[6] == 0) & (rpot[1] == 0)
    rawact = jnp.where(can_insert, interact(pos, facing, jnp.array([2, rawcol]), 1),
                       release)
    action = jnp.where(raw, rawact, action)
    action = jnp.where(inv[1] > 0, interact(pos, facing, jnp.array([1, 2]), 0), action)
    action = jnp.where((phase == 0) & empty, move(pos, jnp.array([2, 2])), action)
    # A demonstrated courier knows the recipe directly. Keep both recipes
    # available and supply plates, leaving dish choice to that partner.
    worker_other = (count == 3) & otherok & (othercount < 3)
    worker_recipe = jnp.where(worker_other, 1-recipe, recipe)
    worker_col = jnp.where(worker_other, othercol, potcol)
    worker_can_fill = (compatible & (count < 3)) | worker_other
    worker = jnp.where(worker_can_fill, fetch_raw(world, pos, facing, worker_recipe, worker_col), move(pos, jnp.array([2, 2])))
    need_plate = ~jnp.any((shared[:, 0] > 0) & (shared[:, 1] == 0)) & (partnerinv[0] == 0) & jnp.any(available) & ~jnp.any(dish_mask)
    worker = jnp.where(need_plate & ((count == 3) | ~worker_can_fill), plate_fetch, worker)
    worker = jnp.where(partnerinv[1] > 0, move(pos, jnp.array([2, 2])), worker)
    worker = jnp.where(jnp.any(dish_mask), counter_action(pos, facing, dish_col, 0), worker)
    worker = jnp.where(plate, release, worker)
    worker = jnp.where(raw, rawact, worker)
    worker = jnp.where(inv[1] > 0, interact(pos, facing, jnp.array([1, 2]), 0), worker)
    action = jnp.where(mode, worker, action)
    delivery = (inv[1] > 0) & (action == 5) & jnp.all(pos == jnp.array([1, 2])) & (facing == 0)
    offered = offering & plate & (action == 5) & (pos[0] == 2) & (pos[1] == free_col) & (facing == 1)
    received = empty & jnp.any(dish_mask) & (action == 5) & (pos[0] == 2) & (pos[1] == dish_col) & (facing == 1)
    trusted = jnp.where(received, 1, jnp.where(delivery, 0, m['trusted']))
    offer = jnp.where(offered, 1, m['offer'])
    job = jnp.where(offered, 0, jnp.where(offering, 1, m['job']))
    phase = jnp.where(delivery, 0, phase)
    wait = jnp.where(delivery, 0, m['wait']+1)
    out = {**m, 't': m['t']+1, 'world': world, 'recipe': recipe, 'phase': phase, 'wait': wait, 'offer': offer, 'job': job,
           'cuebelief': cuebelief, 'belief': belief, 'lastcue': cue, 'idle': idle, 'trusted': trusted,
           'mode': jnp.where(received, 1, mode.astype(jnp.int32)),
           'courier_patience': jnp.where(received, 0, courier_patience),
           'courier_clock': jnp.where(received, 0, m['courier_clock']+1), 'partnerinv': partnerinv,
           'deliverytrusted': jnp.where(delivery, m['trusted'], m['deliverytrusted']),
           'lastempty': jnp.where(partner_seen, (partner_empty & (pc != 2) & ((idle >= 3) | (m['trusted'] > 0))).astype(jnp.int32), m['lastempty']),
           'deliverycue': jnp.where(delivery, cue, m['deliverycue']),
           'deliveryempty': jnp.where(delivery, m['lastempty'], m['deliveryempty']),
           'delivered': jnp.where(delivery, jnp.argmax(inv[2:4]), m['delivered']).astype(jnp.int32),
           'pending': jnp.where(delivery, 1, jnp.where(settling, 0, m['pending']))}
    return action, out, {'pos': pos, 'recipe': recipe, 'phase': phase, 'pot': pot, 'inventory': inv, 'job': job, 'belief': belief, 'mode': mode}


def agent_1_act(obs, reset, memory, rng):
    m = jax.tree.map(lambda x, y: jnp.where(reset, x, y), init_agent_1(), memory)
    pos = locate(obs, 1)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:11]
    empty = jnp.sum(inv) == 0
    raw = (jnp.sum(inv[2:]) > 0) & (inv[1] == 0)
    displayed = jnp.sum(obs[:, :, 40:44], axis=(0, 1))
    recipe = jnp.where(jnp.sum(displayed) > 0, jnp.argmax(displayed), m['recipe']).astype(jnp.int32)
    world = observe(obs, pos, m['world'])
    saved, saved_stances, saved_directions, saved_distance, store_action, store_space = private_storage(world, pos, facing, 1)
    shared = world[3, jnp.array([2, 4])]
    available = jnp.sum(shared[:, :6], axis=1) == 0
    free_col = jnp.array([2, 4])[jnp.argmax(available)]
    idx, pot, compatible = pot_choice(world, recipe, pos, 1)
    potcol = jnp.array([1, 3])[idx]
    count = jnp.sum(pot[2:6])
    pots = world[3, jnp.array([1, 3])]
    oldpots = m['world'][3, jnp.array([1, 3])]
    totals = jnp.sum(pots[:, 2:6], axis=1)
    oldtotals = jnp.sum(oldpots[:, 2:6], axis=1)
    # If both pots block the current recipe, rescue one finished batch into
    # storage instead of forcing the upper player to deliver it incorrectly.
    clearing = ~compatible & jnp.any(totals == 3) & store_space
    clear_score = pots[:, 1]*100 - pots[:, 6] - jnp.abs(jnp.array([1, 3])-pos[1])
    clear_index = jnp.argmax(jnp.where(totals == 3, clear_score, -1000))
    service_pot = jnp.where(clearing, pots[clear_index], pot)
    service_col = jnp.where(clearing, jnp.array([1, 3])[clear_index], potcol)
    need_plate = (compatible & (count == 3)) | clearing
    starts = (totals == 1) & (oldtotals == 0) & (jnp.sum(totals) == 1)
    first_idx = jnp.argmax(starts)
    first_type = jnp.argmax(pots[first_idx, 2:4]).astype(jnp.int32)
    learn = empty & jnp.any(starts) & (m['stable'] >= 3) & (m['t'] >= 8)
    orientation = jnp.where(learn, first_type ^ m['signal'], m['orientation'])
    signal = recipe ^ orientation
    target = jnp.array([4, jnp.where(signal == 0, 1, 3)])
    stable = jnp.where(jnp.all(pos == target) & (signal == m['signal']), m['stable']+1, 0)
    action = move(pos, target)
    action = jnp.where(jnp.all(pos == target), jnp.where(facing == 0, 4, 3), action)
    # The far shared counter is outside the left signaling stance. Refresh
    # that view periodically once a cooked dish needs a plate.
    unseen_time = jnp.where(pos[1] >= 2, 0, m['wait']+1)
    physical_seen = jnp.maximum(m['offer'], (jnp.any(shared[:, :6] > 0) | ~empty).astype(jnp.int32))
    scan = empty & ((pot[1] > 0) | (physical_seen > 0)) & (unseen_time >= 6)
    action = jnp.where(scan, move(pos, jnp.array([4, 2])), action)
    correct_raw = (shared[:, recipe+2] > 0) & (shared[:, 1] == 0)
    can_fill = compatible & (count < 3)
    raw_col = jnp.array([2, 4])[jnp.argmax(correct_raw)]
    plates = (shared[:, 0] > 0) & (shared[:, 1] == 0)
    plate_col = jnp.array([2, 4])[jnp.argmax(plates)]
    # Shared ingredients are useful only for a matching, still-idle pot.
    action = jnp.where(empty & can_fill & jnp.any(correct_raw), counter_action(pos, facing, raw_col, 1), action)
    action = jnp.where(empty & need_plate & jnp.any(plates), counter_action(pos, facing, plate_col, 1), action)
    plate = (inv[0] > 0) & (inv[1] == 0)
    scoop = jnp.where(service_pot[1] > 0, interact(pos, facing, jnp.array([4, service_col]), 0), move(pos, jnp.array([4, service_col])))
    idle_plate = action
    at_pot = jnp.all(pos == jnp.array([4, service_col]))
    scoop = jnp.where((service_pot[1] == 0) & at_pot, jnp.where(facing == 0, 4, 3), scoop)
    action = jnp.where(plate, jnp.where((compatible & (count > 0)) | clearing, scoop, idle_plate), action)
    matching_raw = inv[recipe+2] > 0
    stash = jnp.where(store_space, store_action, move(pos, target))
    saved_dish = (saved[:, 1] > 0) & (saved[:, recipe+2] == 3) & jnp.any(available)
    saved_raw = (saved[:, 1] == 0) & (saved[:, recipe+2] > 0) & can_fill
    saved_plate = (saved[:, 0] > 0) & (saved[:, 1] == 0) & need_plate
    useful = saved_dish | saved_raw | saved_plate
    saved_score = saved_dish.astype(jnp.int32)*100 + saved_raw.astype(jnp.int32)*40 + saved_plate.astype(jnp.int32)*20 - saved_distance
    saved_index = jnp.argmax(jnp.where(useful, saved_score, -1000))
    recover = interact(pos, facing, saved_stances[saved_index], saved_directions[saved_index])
    fresh_available = (can_fill & jnp.any(correct_raw)) | (need_plate & jnp.any(plates))
    action = jnp.where(empty & jnp.any(useful) & (~fresh_available | jnp.any(saved_dish)), recover, action)
    action = jnp.where(raw, jnp.where(matching_raw & can_fill,
        interact(pos, facing, jnp.array([4, potcol]), 0), stash), action)
    correct_dish = (inv[recipe+2] == 3)
    give = jnp.where(jnp.any(available), counter_action(pos, facing, free_col, 1), move(pos, target))
    action = jnp.where(inv[1] > 0, jnp.where(correct_dish, give, stash), action)
    out = {**m, 't': m['t']+1, 'recipe': recipe, 'world': world, 'orientation': orientation, 'signal': signal, 'stable': stable, 'wait': unseen_time, 'offer': physical_seen}
    return action.astype(jnp.int32), out, {'pos': pos, 'recipe': recipe, 'inventory': inv, 'orientation': orientation}
