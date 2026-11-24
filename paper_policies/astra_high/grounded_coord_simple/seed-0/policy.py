"""Recipe relay and observation-driven cook for separated kitchens."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'pace_adaptive_relay', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def position(obs, seat):
    channel, ar, ac = (24, 2, 3) if seat == 0 else (22, 2, 4)
    index = jnp.argmax(obs[:, :, channel])
    return ar + 2 - index // 5, ac + 2 - index % 5


def navigate(r, c, facing, tr, tc, final_direction):
    turn = jnp.asarray([3, 1, 0, 2], jnp.int32)[final_direction]
    at_target = jnp.where(facing == final_direction, 5, turn)
    return jnp.where(r != tr, jnp.where(r < tr, 1, 3),
        jnp.where(c != tc, jnp.where(c < tc, 0, 2), at_target)).astype(jnp.int32)


def init_agent_0():
    # Previous count, empty age, display age, clock, retries, last addition, addition interval.
    return jnp.asarray([0, 0, 100, 0, 0, 0, 6], jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    m = jnp.where(reset, init_agent_0(), memory)
    r, c = position(obs, 0)
    facing = jnp.argmax(obs[2, 2, 1:5])
    count = jnp.sum(obs[:, :, 31:34] * obs[:, :, 22, None])
    cleared = (m[0] > 0) & (count == 0)
    empty_age = jnp.where(cleared, 0, jnp.where(count == 0, m[1] + 1, 0))
    retries = jnp.where(count > 0, 0, m[4])
    added = count > m[0]
    addition_interval = jnp.where(added & (m[0] > 0), jnp.clip(m[3] - m[5], 6, 60), m[6])
    last_addition = jnp.where(added, m[3], m[5])
    relay_delay = jnp.clip((addition_interval + 2) // 3, 2, 20)
    base_interval = jnp.maximum(20, 2 * addition_interval)
    interval = base_interval * (2 ** jnp.clip(retries - 1, 0, 4))
    requested = (count == 0) & ((empty_age >= relay_delay) | (m[3] < 10)) & (m[2] >= interval) & (m[3] < 365)
    button_action = navigate(r, c, facing, 2, 2, 2)
    action = jnp.where((r != 2) | (c != 2) | requested, button_action, 4).astype(jnp.int32)
    pressed = action == 5
    next_m = jnp.asarray([count, empty_age, jnp.where(pressed, 0, m[2] + 1), m[3] + 1, retries + pressed, last_addition, addition_interval], jnp.int32)
    return action, next_m, {'pot_count': count, 'display': pressed, 'empty_age': empty_age, 'retries': retries, 'pace': addition_interval, 'relay_delay': relay_delay}


def init_agent_1():
    # Recipe, waiting age, previous delivered type, clock, confirmed, signal age,
    # remembered occupancy of the four reachable counters.
    return jnp.asarray([-1, 0, 1, 0, 0, 1000, 0, 0, 0, 0], jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    m = jnp.where(reset, init_agent_1(), memory)
    r, c = position(obs, 1)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inventory = obs[2, 2, 5:10]
    plate, cooked = inventory[0] > 0, inventory[1] > 0
    held_kind = jnp.argmax(inventory[2:])
    raw = (jnp.sum(inventory[2:]) > 0) & ~cooked
    pot = jnp.sum(obs[:, :, 29:34] * obs[:, :, 22, None], axis=(0, 1))
    count = jnp.sum(pot[2:])
    timer = jnp.sum(obs[:, :, 39] * obs[:, :, 22])
    displayed = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    visible = jnp.sum(displayed) > 0
    recipe = jnp.where(visible, jnp.argmax(displayed), m[0])
    confirmed = jnp.where(visible, 1, m[4])
    signal_age = jnp.where(visible, 0, m[5] + 1)
    wait_limit = jnp.where((m[3] > 60) & (signal_age > 60), 3, 16)
    fallback = (recipe < 0) & (m[1] >= wait_limit)
    recipe = jnp.where(fallback, 1 - m[2], recipe)
    confirmed = jnp.where(fallback, 0, confirmed)
    kind = jnp.where(count > 0, jnp.argmax(pot[2:]), jnp.maximum(recipe, 0))
    pot_action = navigate(r, c, facing, 2, 5, 3)
    ingredient_action = navigate(r, c, facing, jnp.where(kind == 0, 1, 3), 6, 2)
    plate_row = jnp.where(r <= 2, 1, 3)
    plate_action = navigate(r, c, facing, plate_row, 5, 3)
    delivery_action = navigate(r, c, facing, 2, 6, 2)
    wait_recipe = jnp.where(c != 5, 2, 4)
    ready = pot[1] > 0
    plate_to_pot = jnp.where((r == 2) & (c == 5) & (facing == 3) & ~ready, 4, pot_action)
    empty_action = jnp.where((count >= 3) | (timer > 0) | ready, plate_action,
        jnp.where((recipe >= 0) | (count > 0), ingredient_action, wait_recipe))
    action = jnp.where(cooked, delivery_action, jnp.where(raw, pot_action, jnp.where(plate, plate_to_pot, empty_action)))

    # Discard a known-wrong item only when an empty reachable counter is known.
    cr = jnp.asarray([0, 0, 4, 4], jnp.int32)
    cc = jnp.asarray([5, 6, 5, 6], jnp.int32)
    lr, lc = cr - r + 2, cc - c + 2
    seen = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    occupied = jnp.sum(obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), 29:34], axis=1) > 0
    occupied = jnp.where(seen, occupied, m[6:10] > 0)
    dr = jnp.where(cr == 0, 1, 3)
    costs = jnp.abs(dr - r) + jnp.abs(cc - c) + jnp.where(occupied, 100, 0)
    dest = jnp.argmin(costs)
    discard_action = navigate(r, c, facing, dr[dest], cc[dest], jnp.where(cr[dest] == 0, 0, 1))
    discard = (confirmed > 0) & (recipe >= 0) & (held_kind != recipe) & (cooked | (raw & (count == 0))) & jnp.any(~occupied)
    action = jnp.where(discard, discard_action, action).astype(jnp.int32)
    delivering = cooked & ~discard & (action == 5) & (r == 2) & (c == 6) & (facing == 2)
    recipe = jnp.where(delivering, -1, recipe)
    confirmed = jnp.where(delivering, 0, confirmed)
    next_m = jnp.concatenate((jnp.asarray([recipe, jnp.where(recipe < 0, m[1] + 1, 0), jnp.where(delivering, held_kind, m[2]), m[3] + 1, confirmed, signal_age], jnp.int32), occupied.astype(jnp.int32)))
    return action, next_m, {'recipe': recipe, 'confirmed': confirmed, 'pot_count': count, 'timer': timer, 'delivering': delivering, 'discard': discard, 'fallback': fallback}
