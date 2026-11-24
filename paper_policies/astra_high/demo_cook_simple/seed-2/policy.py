"""Observation-grounded cooperative cooks for the split kitchen."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'shared_pot_cooks', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

_LAYOUT = ('WWWWWR2W0WW', '0      W  B', 'W      P  X', '1      W  B', 'WWWWWR2W1WW')
_CHANNEL = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'L': 4, 'B': 5, '0': 6, '1': 7, '2': 8}
def _static():
    return jnp.asarray([[[int(_CHANNEL.get(c, -1) == k) for k in range(9)] for c in row] for row in _LAYOUT], jnp.int32)
def _coords():
    return jnp.asarray([(r, c) for r in range(5) for c in range(11)], jnp.int32)
def _floor():
    return jnp.asarray([_LAYOUT[r][c] == ' ' for r in range(5) for c in range(11)])
def _templates():
    return jnp.asarray([[[[int(_CHANNEL.get(_LAYOUT[r+i-2][c+j-2], -1) == k) if 0 <= r+i-2 < 5 and 0 <= c+j-2 < 11 else 0 for k in range(9)] for j in range(5)] for i in range(5)] for r in range(5) for c in range(11)], jnp.int32)


def _init():
    return {'recipe': jnp.asarray(-1, jnp.int32), 'items': jnp.zeros((55, 5), jnp.int32),
            'pot_count': jnp.asarray(0, jnp.int32), 'pending': jnp.asarray(False),
            'unseen_steps': jnp.asarray(0, jnp.int32), 'handoff_mode': jnp.asarray(False), 'counter_wait': jnp.asarray(0, jnp.int32),
            'counter_baseline': jnp.zeros((2, 5), jnp.int32), 'cue_index': jnp.asarray(-1, jnp.int32),
            'empty_steps': jnp.asarray(0, jnp.int32), 'signal_kind': jnp.asarray(-1, jnp.int32),
            'signal_steps': jnp.asarray(0, jnp.int32), 'ready_age': jnp.asarray(0, jnp.int32),
            'handoff_pending': jnp.asarray(False), 'intent': jnp.asarray(-1, jnp.int32), 'delivery_age': jnp.asarray(100, jnp.int32),
            'was_cooked': jnp.asarray(False), 'age': jnp.asarray(0, jnp.int32), 'steps': jnp.asarray(0, jnp.int32)}


def init_agent_0():
    return _init()


def init_agent_1():
    return _init()


def _decode(obs, reset, memory, seat):
    memory = {k: jnp.where(reset, _init()[k], v) for k, v in memory.items()}
    room = (_coords()[:, 1] < 7) if seat == 0 else (_coords()[:, 1] > 7)
    error = jnp.sum(jnp.abs(_templates() - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    index = jnp.argmin(jnp.where(_floor() & room, error, 100000))
    pos = _coords()[index]
    local = _coords() - pos + 2
    visible = jnp.all((local >= 0) & (local < 5), axis=1)
    li = jnp.clip(local, 0, 4)
    items = jnp.where(visible[:, None], obs[li[:, 0], li[:, 1], 29:34], memory['items'])
    recipe_counts = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    recipe = jnp.where(jnp.sum(recipe_counts) > 0, jnp.argmax(recipe_counts), memory['recipe']).astype(jnp.int32)
    pot = items[29]
    count = jnp.sum(pot[2:])
    partner = jnp.sum(obs[:, :, 10]) > 0
    partner_cooked = jnp.sum(obs[:, :, 16]) > 0
    unseen_steps = jnp.where(partner, 0, memory['unseen_steps'] + 1)
    cleared = visible[29] & (count == 0) & (memory['pot_count'] > 0)
    pending = memory['pending'] | cleared
    age = jnp.where(cleared, 0, memory['age'] + 1)
    resolved = (recipe != memory['recipe']) | (partner & ~partner_cooked & (age > 0)) | (~partner & (unseen_steps >= 24))
    pending = pending & ~resolved
    memory = {**memory, 'unseen_steps': unseen_steps.astype(jnp.int32), 'recipe': recipe, 'items': items, 'pot_count': count.astype(jnp.int32),
              'pending': pending, 'age': age.astype(jnp.int32), 'steps': memory['steps'] + 1}
    return pos, memory, pot, visible, jnp.argmax(obs[2, 2, 1:5]), obs[2, 2, 5:10]


def _navigate(pos, facing, target, seat, interact=True):
    room = (_coords()[:, 1] < 7) if seat == 0 else (_coords()[:, 1] > 7)
    adjacent = jnp.sum(jnp.abs(_coords() - target), axis=1) == 1
    dist = jnp.sum(jnp.abs(_coords() - pos), axis=1)
    cost = jnp.where(_floor() & room & adjacent, dist, 1000)
    stand = _coords()[jnp.argmin(cost)]
    diff = target - pos
    direction = jnp.where(diff[0] < 0, 0, jnp.where(diff[0] > 0, 1, jnp.where(diff[1] > 0, 2, 3)))
    face_action = jnp.asarray([3, 1, 0, 2], jnp.int32)[direction]
    final = jnp.where(facing == direction, 5 if interact else 4, face_action)
    walk = jnp.where(pos[0] < stand[0], 1, jnp.where(pos[0] > stand[0], 3, jnp.where(pos[1] < stand[1], 0, 2)))
    return jnp.where(jnp.all(pos == stand), final, walk).astype(jnp.int32)


def _counter_target(pos, items, seat, kind=-1, private=False):
    room = (_coords()[:, 1] < 7) if seat == 0 else (_coords()[:, 1] > 7)
    floor = _floor() & room
    adj = jnp.sum(jnp.abs(_coords()[:, None, :] - _coords()[None, :, :]), axis=2) == 1
    distance = jnp.sum(jnp.abs(_coords() - pos), axis=1)
    cost = jnp.min(jnp.where(adj & floor[None, :], distance[None, :] + 1, 1000), axis=1)
    empty = jnp.sum(items, axis=1) == 0
    raw_match = (items[:, 0] == 0) & (items[:, 1] == 0) & (items[:, 2 + jnp.clip(kind, 0, 2)] > 0)
    match = jnp.where(kind == -3, items[:, 1] > 0, jnp.where(kind == -2, (items[:, 0] > 0) & (items[:, 1] == 0), jnp.where(kind < 0, empty, raw_match)))
    valid = (_static().reshape(55, 9)[:, 0] > 0) & match & (cost < 1000) & (~jnp.asarray(private) | (_coords()[:, 1] != 7))
    target_index = jnp.argmin(jnp.where(valid, cost, 10000))
    return _coords()[target_index], jnp.any(valid), jnp.min(jnp.where(valid, cost, 10000))


def agent_0_act(obs, reset, memory, rng):
    pos, mem, pot, visible, facing, inv = _decode(obs, reset, memory, 0)
    recipe = mem['recipe']
    count = jnp.sum(pot[2:])
    raw = jnp.sum(inv[2:]) > 0
    held_type = jnp.argmax(inv[2:])
    ready_age = jnp.where(pot[1] > 0, mem['ready_age'] + 1, 0)
    shared_cooked = jnp.any(mem['items'][jnp.array([18, 40]), 1] > 0)
    partner_present = jnp.sum(obs[:, :, 10]) > 0
    partner_cooked = jnp.sum(obs[:, :, 16]) > 0
    handoff_pending = mem['handoff_pending'] | (inv[1] > 0)
    handoff_pending = handoff_pending & ~((inv[1] == 0) & ~shared_cooked & ((partner_present & ~partner_cooked) | (~partner_present & (mem['unseen_steps'] >= 24))))
    handoff_mode = mem['handoff_mode'] | (inv[0] > 0) | jnp.any(mem['items'][jnp.array([18, 40]), 0] > 0)
    mem = {**mem, 'ready_age': ready_age.astype(jnp.int32), 'handoff_pending': handoff_pending, 'handoff_mode': handoff_mode}
    busy = (count >= 3) | (pot[1] > 0) | mem['pending'] | handoff_pending
    supply = jnp.array([jnp.where(recipe == 1, 3, 1), 0], jnp.int32)
    stored, available, storage_cost = _counter_target(pos, mem['items'], 0, recipe)
    source = jnp.where((available & (storage_cost < jnp.sum(jnp.abs(supply - pos)))) , stored, supply)
    discard, space, _ = _counter_target(pos, mem['items'], 0, private=mem['handoff_mode'])
    go_pot = _navigate(pos, facing, jnp.array([2, 7]), 0)
    wait_pot = _navigate(pos, facing, jnp.array([2, 7]), 0, False)
    get_raw = _navigate(pos, facing, source, 0)
    drop = _navigate(pos, facing, discard, 0)
    action = jnp.where(raw, jnp.where(held_type != recipe, drop, jnp.where(busy, wait_pot, go_pot)), get_raw)
    action = jnp.where(recipe < 0, wait_pot, action)
    # A supplied plate is a natural request to return a plated dish.
    plate_target, plate_available, _ = _counter_target(pos, mem['items'], 0, -2)
    helping = (ready_age >= 4) & plate_available
    action = jnp.where(helping, jnp.where(raw, drop, _navigate(pos, facing, plate_target, 0)), action)
    action = jnp.where(inv[0] > 0, jnp.where(count > 0, jnp.where(pot[1] > 0, go_pot, wait_pot), drop), action)
    shared = jnp.array([[1, 7], [3, 7]], jnp.int32)
    shared_empty = jnp.sum(mem['items'][jnp.array([18, 40])], axis=1) == 0
    handoff_cost = jnp.sum(jnp.abs(shared - pos), axis=1)
    handoff = shared[jnp.argmin(jnp.where(shared_empty, handoff_cost, 1000))]
    action = jnp.where(inv[1] > 0, jnp.where(jnp.any(shared_empty), _navigate(pos, facing, handoff, 0), wait_pot), action)
    trace = {'pos': pos, 'recipe': recipe, 'pot': pot, 'pending': mem['pending'], 'inventory': inv}
    return action.astype(jnp.int32), mem, trace


def agent_1_act(obs, reset, memory, rng):
    pos, mem, pot, visible, facing, inv = _decode(obs, reset, memory, 1)
    count = jnp.sum(pot[2:])
    inventory = obs[2, 2, 5:10]
    delivered = mem['was_cooked'] & (inventory[1] == 0)
    delivery_age = jnp.where(delivered, 0, mem['delivery_age'] + 1)
    partner_scooped = (count == 0) & (memory['pot_count'] > 0) & (inventory[1] == 0) & ~reset
    intent = jnp.where(delivered | partner_scooped, -1, mem['intent'])
    empty_steps = jnp.where((count > 0) | delivered, 0, mem['empty_steps'] + 1)
    # A partner carrying a raw ingredient while facing the pot gives a
    # physically meaningful cue, even if they expect us to do all loading.
    partner_inv = jnp.sum(obs[:, :, 15:20], axis=(0, 1))
    partner_local = jnp.asarray([2, 6]) - pos + 2
    pl = jnp.clip(partner_local, 0, 4)
    partner_at_pot = jnp.all((partner_local >= 0) & (partner_local < 5)) & (obs[pl[0], pl[1], 10] > 0)
    partner_facing_pot = obs[pl[0], pl[1], 13] > 0
    signal = partner_at_pot & partner_facing_pot & (partner_inv[0] == 0) & (partner_inv[1] == 0) & (jnp.sum(partner_inv[2:4]) == 1) & (delivery_age >= 2)
    displayed_kind = jnp.argmax(partner_inv[2:])
    raw_display = (partner_inv[0] == 0) & (partner_inv[1] == 0) & (jnp.sum(partner_inv[2:4]) == 1)
    stable_steps = jnp.where(raw_display & (displayed_kind == mem['signal_kind']) & ~delivered, mem['signal_steps'] + 1, 0)
    stable_signal = raw_display & (stable_steps >= 4) & (empty_steps >= 8) & (delivery_age >= 6)
    intent = jnp.where((count == 0) & (signal | stable_signal), displayed_kind, intent)
    # After a prolonged empty-pot stall, an unambiguous raw counter item
    # can be a physical handoff; never interpret counter position as a code.
    shared_items = mem['items'][jnp.array([18, 40])]
    baseline = jnp.where(delivered, shared_items, mem['counter_baseline'])
    baseline = jnp.where((jnp.sum(shared_items, axis=1) == 0)[:, None], 0, baseline)
    fresh = jnp.any(shared_items != baseline, axis=1)
    shared_raw = (shared_items[:, 0] == 0) & (shared_items[:, 1] == 0) & fresh
    shared_counts = jnp.sum(jnp.where(shared_raw[:, None], shared_items[:, 2:], 0), axis=0)
    partner_cells = jnp.asarray([[1, 6], [3, 6]]) - pos + 2
    pc = jnp.clip(partner_cells, 0, 4)
    partner_waits = jnp.any(jnp.all((partner_cells >= 0) & (partner_cells < 5), axis=1) & (obs[pc[:, 0], pc[:, 1], 10] > 0) & (obs[pc[:, 0], pc[:, 1], 13] > 0)) & (jnp.sum(partner_inv) == 0)
    counter_wait = jnp.where(partner_waits & ~delivered, mem['counter_wait'] + 1, 0)
    counter_signal = (jnp.sum(shared_counts > 0) == 1) & (shared_counts[2] == 0) & ((empty_steps >= 24) | ((counter_wait >= 4) & (delivery_age >= 4)))
    mem = {**mem, 'counter_wait': counter_wait.astype(jnp.int32)}
    accept_counter = (intent < 0) & counter_signal
    intent = jnp.where(accept_counter, jnp.argmax(shared_counts), intent)
    cue_index = jnp.where(delivered | (jnp.sum(inventory[2:]) > 0), -1, mem['cue_index'])
    cue_index = jnp.where(accept_counter, jnp.asarray([18, 40])[jnp.argmax(jnp.sum(jnp.where(shared_raw[:, None], shared_items[:, 2:], 0), axis=1))], cue_index)
    cue_item = mem['items'][jnp.maximum(cue_index, 0)]
    withdrawn = (cue_index >= 0) & (jnp.sum(cue_item[2:]) == 0) & (jnp.sum(inventory) == 0)
    intent = jnp.where(withdrawn & (count == 0), -1, intent)
    cue_index = jnp.where(withdrawn, -1, cue_index)
    mem = {**mem, 'counter_baseline': baseline, 'cue_index': cue_index.astype(jnp.int32)}
    mem = {**mem, 'empty_steps': empty_steps.astype(jnp.int32), 'signal_steps': stable_steps.astype(jnp.int32), 'signal_kind': jnp.where(raw_display, displayed_kind, -1).astype(jnp.int32)}
    intent = jnp.where(count > 0, jnp.argmax(pot[2:]), intent).astype(jnp.int32)
    mem = {**mem, 'intent': intent, 'delivery_age': delivery_age.astype(jnp.int32), 'was_cooked': inventory[1] > 0}
    kind = jnp.maximum(intent, 0)
    raw = jnp.sum(inv[2:]) > 0
    held_kind = jnp.argmax(inv[2:])
    supply = jnp.array([jnp.where(kind == 1, 4, 0), 8], jnp.int32)
    plate_supply = jnp.array([jnp.where(pos[0] > 2, 3, 1), 10], jnp.int32)
    discard, space, _ = _counter_target(pos, mem['items'], 1)
    stored, available, storage_cost = _counter_target(pos, mem['items'], 1, kind)
    source = jnp.where(available & (storage_cost <= jnp.sum(jnp.abs(supply - pos))), stored, supply)
    source = jnp.where(mem['cue_index'] >= 0, _coords()[jnp.maximum(mem['cue_index'], 0)], source)
    go_pot = _navigate(pos, facing, jnp.array([2, 7]), 1)
    wait_pot = _navigate(pos, facing, jnp.array([2, 7]), 1, False)
    deliver = _navigate(pos, facing, jnp.array([2, 10]), 1)
    drop = _navigate(pos, facing, discard, 1)
    get_raw = _navigate(pos, facing, source, 1)
    get_plate = _navigate(pos, facing, plate_supply, 1)
    action = jnp.where(count >= 3, get_plate, jnp.where(intent >= 0, get_raw, wait_pot))
    action = jnp.where(raw, jnp.where((intent >= 0) & (count < 3) & (held_kind == kind), go_pot, drop), action)
    action = jnp.where(inv[0] > 0, jnp.where(count >= 3, jnp.where(pot[1] > 0, go_pot, wait_pot), drop), action)
    cooked_target, cooked_available, _ = _counter_target(pos, mem['items'], 1, -3)
    retrieve_dish = jnp.where(jnp.sum(inv) > 0, drop, _navigate(pos, facing, cooked_target, 1))
    action = jnp.where(cooked_available, retrieve_dish, action)
    action = jnp.where(inv[1] > 0, deliver, action)
    return action.astype(jnp.int32), mem, {'pos': pos, 'pot': pot, 'inventory': inv, 'intent': intent, 'delivery_age': delivery_age}
