"""Grounded recipe display outside; autonomous repeated cooking inside."""
import jax.numpy as jnp
from custom_primitives import FLOOR, TEMPLATES, ACTIONS, COSTS

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}
_LAYOUT = ('WWW2R2WWW', 'W       W', 'W WWLWW W', '2 0   B 2',
           'RAXAP X R', '2 1   B 2', 'W WWLWW W', 'W       W', 'WWW2R2WWW')

def _decode(obs):
    errors = jnp.sum(jnp.abs(jnp.asarray(TEMPLATES, jnp.int32) - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    pos = jnp.asarray(FLOOR, jnp.int32)[jnp.argmin(errors)]
    facing = jnp.asarray([3, 1, 0, 2], jnp.int32)[jnp.argmax(obs[2, 2, 1:5])]
    return pos, facing, obs[2, 2, 5:10]


def _go(target, pos, facing):
    return jnp.asarray(ACTIONS, jnp.int32)[target, pos, facing]


def _costs(pos, facing):
    return jnp.asarray(COSTS, jnp.int32)[:, pos, facing]


def _global_observation(obs, pos):
    rr = jnp.arange(81) // 9 - pos // 9 + 2
    cc = jnp.arange(81) % 9 - pos % 9 + 2
    visible = (rr >= 0) & (rr < 5) & (cc >= 0) & (cc < 5)
    values = obs[jnp.clip(rr, 0, 4), jnp.clip(cc, 0, 4)]
    return jnp.where(visible[:, None], values, 0), visible


def init_agent_0():
    # Recipe, last known counter contents; all inventory checks remain observed.
    return {'recipe': jnp.asarray(-1, jnp.int32), 'items': jnp.zeros((81, 5), jnp.int32),
            'activity': jnp.asarray(0, jnp.int32), 'patrol': jnp.asarray(0, jnp.int32),
            'partner': jnp.zeros((10,), jnp.int32),
            'mute': jnp.asarray(0, jnp.int32), 'plate_seen': jnp.asarray(0, jnp.int32), 'loads': jnp.asarray(0, jnp.int32)}


def agent_0_act(obs, reset, memory, rng):
    memory = {k: jnp.where(reset, init_agent_0()[k], v) for k, v in memory.items()}
    pos, facing, held = _decode(obs)
    world, visible = _global_observation(obs, pos)
    items = jnp.where(visible[:, None], world[:, 29:34], memory['items'])
    recipe_counts = jnp.max(obs[:, :, 36:39], axis=(0, 1))
    recipe = jnp.where(jnp.sum(recipe_counts) > 0, jnp.argmax(recipe_counts), memory['recipe']).astype(jnp.int32)
    partner_pos = jnp.argmax(world[:, 10]).astype(jnp.int32)
    partner_visible = jnp.sum(world[:, 10]) > 0
    signature = jnp.concatenate((partner_pos[None], world[partner_pos, 11:20]))
    changed = partner_visible & jnp.any(signature != memory['partner'])
    activity = jnp.where(changed | (recipe != memory['recipe']), 0, memory['activity']+1).astype(jnp.int32)
    partner = jnp.where(partner_visible, signature, memory['partner'])
    plate_here = partner_visible & (partner_pos == 39) & (signature[5] > 0) & (signature[6] == 0)
    raw_now = partner_visible & (signature[5] == 0) & (signature[6] == 0) & (jnp.sum(signature[7:]) == 1)
    raw_before = (memory['partner'][5] == 0) & (memory['partner'][6] == 0) & (jnp.sum(memory['partner'][7:]) == 1)
    loads = memory['loads'] + (raw_now & ~raw_before).astype(jnp.int32)
    third_load = loads >= 3
    mute = jnp.where((plate_here & (memory['plate_seen'] == 0)) | third_load, 32, jnp.maximum(memory['mute']-1, 0)).astype(jnp.int32)
    plate_seen = jnp.where(partner_visible, plate_here.astype(jnp.int32), memory['plate_seen'])
    completed = partner_visible & (partner_pos == 39) & (memory['partner'][0] == 39) & (memory['partner'][6] > 0) & (memory['partner'][4] > 0) & (jnp.sum(signature[5:]) == 0)
    mute = jnp.where(completed | (recipe != memory['recipe']), 0, mute).astype(jnp.int32)
    loads = jnp.where(third_load | completed | (recipe != memory['recipe']), 0, loads).astype(jnp.int32)
    patrol = jnp.where(activity >= 45, 1, memory['patrol']).astype(jnp.int32)
    activity = jnp.where((pos == 43) & (patrol > 0), 0, activity).astype(jnp.int32)
    patrol = jnp.where(pos == 43, 0, patrol).astype(jnp.int32)
    costs = _costs(pos, facing)
    indicators = jnp.asarray([4, 36, 44, 76], jnp.int32)
    indicator = indicators[jnp.argmin(costs[indicators])]
    # Present the literal ingredient from the central left cell. Both shared
    # left counters and the free recipe display remain in view.
    stand = jnp.where(patrol > 0, 43, 37).astype(jnp.int32)
    supply = jnp.where(recipe == 0, 29, 47)
    correct = (held[0] == 0) & (held[1] == 0) & (held[2+jnp.maximum(recipe, 0)] == 1)
    empty = jnp.sum(held) == 0
    walls = jnp.asarray([ch == 'W' for row in _LAYOUT for ch in row])
    free = walls & (jnp.sum(items, axis=-1) == 0)
    cells = jnp.arange(81)
    shared = (cells == 21) | (cells == 23) | (cells == 57) | (cells == 59)
    next_distance = jnp.abs(cells//9 - supply//9) + jnp.abs(cells%9 - supply%9)
    dump = jnp.argmin(jnp.where(free, 10*costs + next_distance + 10000*shared, 99999))
    stored = walls & (items[:, 0] == 0) & (items[:, 1] == 0) & (items[:, 2+jnp.maximum(recipe, 0)] == 1)
    stored_target = jnp.argmin(jnp.where(stored, costs, 9999))
    get_target = jnp.where(jnp.min(jnp.where(stored, costs, 9999)) < costs[supply], stored_target, supply)
    target = jnp.where(empty, jnp.where(mute > 0, stand, get_target), jnp.where(correct & (mute == 0), stand, dump))
    matching_dish = walls & (items[:, 0] > 0) & (items[:, 1] > 0) & (items[:, 2+jnp.maximum(recipe, 0)] == 3)
    offered = jnp.any(matching_dish & (costs < 999)) & (recipe >= 0)
    dish_target = jnp.argmin(jnp.where(matching_dish, costs, 9999))
    target = jnp.where(offered, jnp.where(empty, dish_target, dump), target)
    goals = jnp.asarray([38, 42], jnp.int32)
    goal = goals[jnp.argmin(costs[goals])]
    held_match = (held[1] > 0) & (held[2+jnp.maximum(recipe, 0)] == 3) & (recipe >= 0)
    target = jnp.where(held_match, goal, target)
    patrol = jnp.where(offered | held_match, 0, patrol).astype(jnp.int32)
    target = jnp.where(recipe < 0, indicator, target)
    action = _go(target, pos, facing)
    # Once close enough to see a free indicator, memory refreshes every step.
    return action, {'recipe': recipe, 'items': items, 'activity': activity, 'patrol': patrol, 'partner': partner, 'mute': mute, 'plate_seen': plate_seen, 'loads': loads}, {'position': pos, 'recipe': recipe, 'target': target, 'activity': activity, 'patrol': patrol}


def init_agent_1():
    return {'recipe': jnp.asarray(-1, jnp.int32), 'step': jnp.asarray(0, jnp.int32), 'armed': jnp.asarray(0, jnp.int32)}


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    memory = {k: jnp.where(reset, initial[k], v) for k, v in memory.items()}
    pos, facing, held = _decode(obs)
    costs = _costs(pos, facing)
    counts = jnp.max(obs[:, :, 36:39], axis=(0, 1))
    recipe = jnp.where(jnp.sum(counts) > 0, jnp.argmax(counts), memory['recipe']).astype(jnp.int32)
    pot = jnp.sum(obs[:, :, 29:34] * obs[:, :, 22:23], axis=(0, 1))
    timer = jnp.max(obs[:, :, 39])
    amount = jnp.sum(pot[2:])
    goals = jnp.asarray([38, 42], jnp.int32)
    goal = goals[jnp.argmin(costs[goals])]
    plates = jnp.asarray([33, 51], jnp.int32)
    plate = plates[jnp.argmin(costs[plates])]
    buttons = jnp.asarray([22, 58], jnp.int32)
    button = buttons[jnp.argmin(costs[buttons])]
    supply = jnp.where(recipe == 0, 29, 47)
    empty = jnp.sum(held) == 0
    prepay = memory['step'] + timer + 42 <= 400
    arm = empty & (amount >= 3) & (memory['armed'] == 0) & prepay
    target = jnp.where(empty, jnp.where(amount >= 3, jnp.where(arm, button, plate), supply), 40)
    target = jnp.where(empty & (recipe < 0), button, target)
    target = jnp.where(held[1] > 0, goal, target)
    scoop_position = jnp.where(memory['armed'] > 0, 41, 39)
    target = jnp.where((held[0] > 0) & (held[1] == 0) & (pos != scoop_position), scoop_position, target)
    action = _go(target, pos, facing)
    action = jnp.where(arm & (action == 5) & (timer > 5), 4, action)
    armed = jnp.where(arm & (action == 5), 1, memory['armed']).astype(jnp.int32)
    action = jnp.where(empty & (recipe < 0) & (memory['step'] > 362), 4, action)
    delivered = (held[1] > 0) & (action == 5)
    recipe = jnp.where(delivered, -1, recipe).astype(jnp.int32)
    armed = jnp.where(delivered, 0, armed).astype(jnp.int32)
    return action, {'recipe': recipe, 'step': memory['step']+1, 'armed': armed}, {'position': pos, 'recipe': recipe, 'target': target, 'pot_amount': amount, 'timer': timer}
