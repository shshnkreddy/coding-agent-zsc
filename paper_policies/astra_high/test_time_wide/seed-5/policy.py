"""Recipe-aware physical cooperation across the kitchen partition."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'physical_handoffs', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}


def _init():
    return {'items': jnp.zeros((9, 7), jnp.int32),
            'flight': jnp.asarray(0, jnp.int32),
            'no_cue': jnp.asarray(0, jnp.int32),
            'next_kind': jnp.asarray(0, jnp.int32),
            'time': jnp.asarray(0, jnp.int32),
            'unseen_right': jnp.asarray(0, jnp.int32),
            'raw_seen': jnp.asarray(0, jnp.int32),
            'return_counter': jnp.asarray(0, jnp.int32),
            'cue': jnp.asarray(-1, jnp.int32),
            'cue_kind': jnp.asarray(-1, jnp.int32),
            'verified': jnp.asarray(0, jnp.int32),
            'map_logit': jnp.asarray(2, jnp.int32),
            'map_known': jnp.asarray(-1, jnp.int32),
            'pending_cue': jnp.asarray(-1, jnp.int32),
            'pending_kind': jnp.asarray(0, jnp.int32),
            'served_pot': jnp.asarray(0, jnp.int32),
            'pending_age': jnp.asarray(0, jnp.int32),
            'pending_scored': jnp.asarray(0, jnp.int32),
            'cue_age': jnp.asarray(0, jnp.int32),
            'direct': jnp.asarray(0, jnp.int32)}


def init_agent_0():
    return _init()


def init_agent_1():
    return _init()


def _observe(obs, reset, memory, lower):
    memory = jax.tree.map(lambda a, b: jnp.where(reset, a, b), _init(), memory)
    landmark = 25 if lower else 27
    flat = jnp.argmax(obs[:, :, landmark])
    col = (4 if lower else 5) - flat % 5
    row = (8 if lower else 2) - flat // 5
    direction = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:11]
    cols = jnp.asarray([1, 3, 2, 4, 0, 5, 1, 3, 4], jnp.int32)
    rows = jnp.asarray([3, 3, 3, 3, 5, 5, 6, 6, 6], jnp.int32)
    rr = rows - row + 2
    cc = cols - col + 2
    visible = (rr >= 0) & (rr < 5) & (cc >= 0) & (cc < 5)
    local = obs[jnp.clip(rr, 0, 4), jnp.clip(cc, 0, 4)]
    current = jnp.concatenate((local[:, 32:38], local[:, 44:45]), axis=1)
    items = jnp.where(visible[:, None], current, memory['items'])
    return row, col, direction, inv, items, memory


def _at(row, col, direction, tr, tc, face, interact=True):
    # Floor in either room is an unobstructed rectangle.
    move = jnp.where(tc > col, 0, 2)
    move = jnp.where(col == tc, jnp.where(tr > row, 1, 3), move)
    face_action = jnp.asarray([3, 1, 0, 2], jnp.int32)[face]
    there = (row == tr) & (col == tc)
    finish = jnp.where(direction == face, 5 if interact else 4, face_action)
    return jnp.where(there, finish, move).astype(jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    row, col, direction, inv, items, memory = _observe(obs, reset, memory, False)
    pots, counters = items[:2], items[2:4]
    partner_flat = jnp.argmax(obs[:, :, 11])
    partner_visible = jnp.any(obs[:, :, 11] > 0)
    partner_row = row + partner_flat // 5 - 2
    partner_col = col + partner_flat % 5 - 2
    partner = obs[partner_flat // 5, partner_flat % 5]
    pointed = partner_visible & (partner_row == 4) & ((partner_col == 1) | (partner_col == 3)) & (partner[12] > 0)
    cue_now = (partner_col - 1) // 2
    cue = jnp.where(pointed, cue_now, memory['cue'])
    cue_age = jnp.where(pointed, jnp.where(cue_now == memory['cue'], jnp.minimum(memory['cue_age'] + 1, 8), 1), jnp.maximum(0, memory['cue_age'] - 1))
    pending_age = jnp.where(memory['pending_cue'] >= 0, memory['pending_age'] + 1, 0)
    stable_response = pointed & (cue_age >= 2) & (memory['pending_cue'] >= 0) & (pending_age <= 24)
    changed_response = stable_response & (cue != memory['pending_cue'])
    unchanged_response = stable_response & (cue == memory['pending_cue']) & (pending_age >= 6) & (memory['pending_scored'] == 0)
    mapping_evidence = memory['pending_kind'] ^ jnp.maximum(memory['pending_cue'], 0)
    map_known = jnp.where(changed_response & (memory['map_known'] < 0), mapping_evidence, memory['map_known'])
    map_known = jnp.where(memory['verified'] == 3, 0, map_known)
    map_logit = jnp.clip(memory['map_logit'] + jnp.where(unchanged_response & (map_known < 0), jnp.where(mapping_evidence == 0, -1, 1), 0), -4, 4)
    pending_scored = jnp.where(unchanged_response | changed_response, 1, memory['pending_scored'])
    reversed_mapping = jnp.where(map_known >= 0, map_known, (map_logit < 0).astype(jnp.int32))
    mapped_cue = jnp.maximum(cue, 0) ^ reversed_mapping
    no_cue = jnp.where(pointed, 0, memory['no_cue'] + 1)
    sums = jnp.sum(pots[:, 2:6], axis=1)
    pointed_pot = jnp.maximum(cue, 0)
    pointed_kind = jnp.argmax(pots[pointed_pot, 2:6])
    grounded = pointed & (sums[pointed_pot] > 0) & (pots[pointed_pot, 2 + pointed_kind] == sums[pointed_pot]) & (pointed_kind < 2)
    cue_kind = jnp.where(grounded, pointed_kind, memory['cue_kind'])
    free = jnp.sum(counters[:, :6], axis=1) == 0
    counter = jnp.where(free[0], 0, 1)
    counter_col = 2 + 2 * counter
    soup = counters[:, 1] > 0
    soup_index = jnp.argmax(soup)
    plate_out = jnp.any((counters[:, 0] > 0) & (counters[:, 1] == 0))
    flight = jnp.maximum(0, memory['flight'] - 1)
    flight = jnp.where(inv[1] > 0, 0, flight)
    # Give the informed player ample time to accept the physical handoff.
    ignored = plate_out & (flight < 75) & (partner[16] == 0)
    direct = jnp.where(ignored & (cue_age >= 4), 1, memory['direct'])
    direct = jnp.where(ignored & (flight < 60) & (cue_age < 2), 2, direct)
    direct = jnp.where((direct > 0) & (no_cue > 20), 2, direct)
    direct = jnp.where((direct > 0) & (cue_age >= 4), 1, direct)
    direct = jnp.where((partner_visible & (partner[16] > 0)) | jnp.any(soup), 0, direct)
    learned_direct = (memory['verified'] == 3) & (cue_age >= 4) & (flight == 0) & ~plate_out & (jnp.sum(inv) == 0) & ~jnp.any(soup)
    direct = jnp.where(learned_direct, 1, direct)
    blind_match = (sums > 0) & (pots[:, 2 + memory['next_kind']] == sums)
    blind_pot = jnp.where(jnp.any(blind_match), jnp.argmax(blind_match), memory['next_kind'])
    direct_pot = jnp.where(direct == 2, blind_pot, mapped_cue)
    valid_direct = (direct == 2) | (cue_age >= 2)
    direct_ready = (direct > 0) & valid_direct & (pots[direct_pot, 1] > 0)
    need_plate = (flight == 0) & ~plate_out & jnp.any(free) & jnp.any(sums == 3)
    need_plate = need_plate & ((direct == 0) | (valid_direct & (sums[direct_pot] == 3)))
    need_plate = need_plate & ((cue_age < 2) | (sums[jnp.maximum(cue, 0)] == 3) | (direct > 0))
    available = (sums < 3) & (pots[:, 1] == 0) & (pots[:, 6] == 0)
    need_plate = need_plate & ((direct == 0) | (pots[direct_pot, 6] <= 6) | ~jnp.any(available))
    pot_index = jnp.argmax(jnp.where(available, 10 + sums + 4 * ((jnp.arange(2) == jnp.where(direct > 0, mapped_cue, cue)) & (cue_age >= 2)), -100))
    kind = jnp.where(sums[pot_index] > 0, jnp.argmax(pots[pot_index, 2:6]), pot_index)
    pot_col = 1 + 2 * pot_index
    source_row = 1 + kind
    # Pick the closest source including the trip back to the selected pot.
    left_cost = jnp.abs(col - 1) + jnp.abs(pot_col - 1)
    right_cost = jnp.abs(col - 4) + jnp.abs(pot_col - 4)
    right = right_cost < left_cost
    source_col = jnp.where(right, 4, 1)
    source_face = jnp.where(right, 2, 3)
    action = _at(row, col, direction, 2, 2, 1, False)
    task = jnp.asarray(0, jnp.int32)
    action = jnp.where(jnp.any(available), _at(row, col, direction, source_row, source_col, source_face), action)
    task = jnp.where(jnp.any(available), 1, task)
    action = jnp.where(need_plate, _at(row, col, direction, 1, 3, 0), action)
    task = jnp.where(need_plate, 2, task)
    reclaim = (direct > 0) & plate_out & direct_ready
    plate_index = jnp.argmax((counters[:, 0] > 0) & (counters[:, 1] == 0))
    action = jnp.where(reclaim, _at(row, col, direction, 2, 2 + 2 * plate_index, 1), action)
    task = jnp.where(reclaim, 7, task)
    action = jnp.where(jnp.any(soup), _at(row, col, direction, 2, 2 + 2 * soup_index, 1), action)
    task = jnp.where(jnp.any(soup), 3, task)
    raw = (inv[0] == 0) & (jnp.sum(inv[2:]) > 0)
    held_kind = jnp.argmax(inv[2:])
    compatible = available & ((sums == 0) | (pots[:, 2 + held_kind] == sums))
    dest = jnp.argmax(jnp.where(compatible, 10 + 4 * sums + (jnp.arange(2) == held_kind), -100))
    raw_action = _at(row, col, direction, 2, 1 + 2 * dest, 1)
    # An unusable ingredient can be parked on an upper-room wall.
    raw_action = jnp.where(jnp.any(compatible), raw_action, _at(row, col, direction, 1, 1, 0))
    action = jnp.where(raw, raw_action, action)
    task = jnp.where(raw, 4, task)
    empty_plate = (inv[0] > 0) & (inv[1] == 0)
    action = jnp.where(empty_plate, _at(row, col, direction, 2, counter_col, 1), action)
    task = jnp.where(empty_plate, 5, task)
    direct_action = _at(row, col, direction, 2, 1 + 2 * direct_pot, 1)
    direct_action = jnp.where(direct_ready, direct_action, _at(row, col, direction, 2, 2, 1, False))
    action = jnp.where(empty_plate & (direct > 0), direct_action, action)
    task = jnp.where(empty_plate & (direct > 0), 8, task)
    flight = jnp.where(empty_plate & (action == 5) & (row == 2) & (col == counter_col) & jnp.any(free), 100, flight)
    action = jnp.where(inv[1] > 0, _at(row, col, direction, 1, 2, 0), action)
    task = jnp.where(inv[1] > 0, 6, task)
    delivered = (inv[1] > 0) & (row == 1) & (col == 2) & (direction == 0) & (action == 5)
    scooping = empty_plate & (task == 8) & (action == 5) & (row == 2) & ((col == 1) | (col == 3))
    served_pot = jnp.where(scooping, (col - 1) // 2, memory['served_pot'])
    capture_delivery = delivered & (direct > 0) & (cue >= 0) & (no_cue <= 8)
    pending_cue = jnp.where(delivered, jnp.where(capture_delivery, cue, -1), memory['pending_cue'])
    pending_kind = jnp.where(capture_delivery, served_pot, memory['pending_kind'])
    pending_age = jnp.where(delivered, 0, pending_age)
    pending_scored = jnp.where(delivered, 0, pending_scored)
    next_kind = jnp.where(delivered, 1 - jnp.argmax(inv[2:4]), memory['next_kind'])
    returned_kind = jnp.argmax(counters[soup_index, 2:6])
    returning = (task == 3) & (action == 5) & (jnp.sum(inv) == 0)
    confirmed = returning & (cue_kind == returned_kind) & (returned_kind < 2) & (counters[soup_index, 2 + returned_kind] == 3)
    verified = jnp.where(confirmed, memory['verified'] | (1 << returned_kind), memory['verified'])
    verified = jnp.where(returning & (cue_kind >= 0) & (cue_kind != returned_kind), 0, verified)
    next_memory = {**memory, 'items': items, 'flight': flight, 'time': memory['time'] + 1,
                   'cue': cue, 'cue_age': cue_age, 'direct': direct, 'cue_kind': cue_kind, 'verified': verified,
                   'no_cue': no_cue, 'next_kind': next_kind,
                   'map_known': map_known, 'map_logit': map_logit, 'pending_cue': pending_cue,
                   'pending_kind': pending_kind, 'served_pot': served_pot, 'pending_age': pending_age, 'pending_scored': pending_scored}
    return action, next_memory, {'map_known': map_known, 'map_logit': map_logit, 'mapped_cue': mapped_cue, 'verified': verified, 'cue_kind': cue_kind, 'cue': cue, 'cue_age': cue_age, 'direct': direct, 'task': task,
                                'pos': jnp.asarray([row,col]), 'items': items, 'flight': flight,
                                'next_kind': next_kind}



def agent_1_act(obs, reset, memory, rng):
    row, col, direction, inv, items, memory = _observe(obs, reset, memory, True)
    pots, storage = items[:2], items[2:]
    # Interaction positions for the two shared and five private counters.
    access_row = jnp.asarray([4, 4, 5, 5, 5, 5, 5], jnp.int32)
    access_col = jnp.asarray([2, 4, 1, 4, 1, 3, 4], jnp.int32)
    access_face = jnp.asarray([0, 0, 3, 2, 1, 1, 1], jnp.int32)
    distance = jnp.abs(access_row - row) + jnp.abs(access_col - col)
    private = jnp.arange(7) >= 2
    free = jnp.sum(storage[:, :6], axis=1) == 0
    free_shared = jnp.any(free[:2])
    free_private = jnp.any(free & private)
    stash_index = jnp.argmin(jnp.where(free & private, distance, 100))
    drop_counter = jnp.where(free[memory['return_counter']], memory['return_counter'], 1 - memory['return_counter'])
    stash_action = _at(row, col, direction, access_row[stash_index], access_col[stash_index], access_face[stash_index])
    return_action = _at(row, col, direction, 4, 2 + 2 * drop_counter, 0)
    recipe = jnp.max(obs[:, :, 40:44], axis=(0, 1))
    recipe_kind = jnp.argmax(recipe)
    sizes = jnp.sum(pots[:, 2:6], axis=1)
    match = jnp.all(pots[:, 2:6] == recipe, axis=1)
    partial_match = (sizes > 0) & jnp.all(pots[:, 2:6] <= recipe, axis=1)
    waiting_pot = jnp.where(jnp.any(partial_match), jnp.argmax(jnp.where(partial_match, 10 + sizes, -100)), recipe_kind)
    correct_pot = jnp.argmax(jnp.where(match, 100 - pots[:, 6], -100))
    available = (sizes < 3) & (pots[:, 1] == 0) & (pots[:, 6] == 0)
    raw = (storage[:, 0] == 0) & (storage[:, 1] == 0) & (jnp.sum(storage[:, 2:4], axis=1) > 0)
    raw_kind = jnp.argmax(storage[:, 2:6], axis=1)
    # Do not consume the other empty pot with surplus of an already stocked recipe.
    existing = pots[:, 2 + raw_kind].T
    existing_any = jnp.any(existing > 0, axis=1)
    fits = available[None, :] & (((sizes[None, :] > 0) & (existing == sizes[None, :])) | ((sizes[None, :] == 0) & ~existing_any[:, None]))
    useful_raw = raw & jnp.any(fits, axis=1)
    clear_raw = raw & ~private & free_private
    take_raw = jnp.any(useful_raw | clear_raw)
    raw_index = jnp.argmax(jnp.where(useful_raw | clear_raw, 20 * useful_raw + 5 * (raw_kind == recipe_kind) - distance, -100))
    raw_seen = memory['raw_seen'] | jnp.any(raw & ~private).astype(jnp.int32)
    plates = (storage[:, 0] > 0) & (storage[:, 1] == 0)
    plate_index = jnp.argmin(jnp.where(plates, distance, 100))
    take_plate = jnp.any(plates) & ((raw_seen == 0) | (jnp.any(match) & free_shared & ~jnp.any(raw & ~private)))
    stored_dish = private & (storage[:, 1] > 0) & jnp.all(storage[:, 2:6] == recipe, axis=1)
    dish_index = jnp.argmin(jnp.where(stored_dish, distance, 100))
    unseen_right = jnp.where(col >= 2, 0, memory['unseen_right'] + 1)
    waiting_col = jnp.where(unseen_right >= 8, 2, 1 + 2 * waiting_pot)
    action = _at(row, col, direction, 4, waiting_col, 0, False)
    task = jnp.asarray(0, jnp.int32)
    action = jnp.where(take_raw, _at(row, col, direction, access_row[raw_index], access_col[raw_index], access_face[raw_index]), action)
    task = jnp.where(take_raw, 4, task)
    action = jnp.where(take_plate, _at(row, col, direction, access_row[plate_index], access_col[plate_index], access_face[plate_index]), action)
    task = jnp.where(take_plate, 1, task)
    action = jnp.where(jnp.any(stored_dish) & free_shared, _at(row, col, direction, access_row[dish_index], access_col[dish_index], access_face[dish_index]), action)
    task = jnp.where(jnp.any(stored_dish) & free_shared, 6, task)
    empty_plate = (inv[0] > 0) & (inv[1] == 0)
    plate_action = _at(row, col, direction, 4, 1 + 2 * correct_pot, 0)
    plate_action = jnp.where(jnp.any(match), plate_action, _at(row, col, direction, 4, 1 + 2 * waiting_pot, 0, False))
    park_plate = (raw_seen > 0) & (~jnp.any(match) | jnp.any(raw[:2])) & free_private
    plate_action = jnp.where(park_plate, stash_action, plate_action)
    action = jnp.where(empty_plate, plate_action, action)
    task = jnp.where(empty_plate, 2, task)
    held_raw = (inv[0] == 0) & jnp.any(inv[2:] > 0)
    held_kind = jnp.argmax(inv[2:])
    held_existing = pots[:, 2 + held_kind]
    compatible = available & (((sizes > 0) & (held_existing == sizes)) | ((sizes == 0) & ~jnp.any(held_existing > 0))) & (held_kind < 2)
    raw_pot = jnp.argmax(jnp.where(compatible, 10 + 2 * sizes + (jnp.arange(2) == held_kind), -100))
    raw_action = _at(row, col, direction, 4, 1 + 2 * raw_pot, 0)
    raw_action = jnp.where(jnp.any(compatible), raw_action, jnp.where(free_private, stash_action, return_action))
    action = jnp.where(held_raw, raw_action, action)
    task = jnp.where(held_raw, 5, task)
    correct_held = jnp.all(inv[2:6] == recipe)
    soup_action = jnp.where(free_shared & correct_held, return_action, jnp.where(free_private, stash_action, return_action))
    action = jnp.where(inv[1] > 0, soup_action, action)
    task = jnp.where(inv[1] > 0, 3, task)
    plate_pickup = (jnp.sum(inv) == 0) & (action == 5) & (row == 4) & (col == 2 + 2 * plate_index) & (plate_index < 2) & plates[plate_index]
    return_counter = jnp.where(plate_pickup, plate_index, memory['return_counter'])
    next_memory = {**memory, 'items': items, 'time': memory['time'] + 1,
                   'unseen_right': unseen_right, 'raw_seen': raw_seen, 'return_counter': return_counter}
    return action, next_memory, {'task': task, 'pos': jnp.asarray([row,col]), 'recipe': recipe, 'items': items}
