"""Observation-driven cooking with autonomous recipe acquisition."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'autonomous_grounded_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}

from custom_primitives import CROPS, ROUTES, COSTS, FLOOR_MASK, COUNTERS

_POT = 40
_SOURCES = (29, 47, 3)
_BUTTONS = (22, 58)
_PLATES = (33, 51)
_GOALS = (38, 42)


def _initial():
    return {'t': jnp.int32(0), 'recipe': jnp.zeros(3, jnp.int32),
            'last_cooked': jnp.bool_(False), 'last_action': jnp.int32(4),
            'last_target': jnp.int32(0), 'armed': jnp.bool_(False)}


def _initial_outer():
    return {'t': jnp.int32(0), 'recipe': jnp.zeros(3, jnp.int32),
            'quiet': jnp.int32(0), 'helping': jnp.bool_(False),
            'eligible': jnp.bool_(True), 'help_button': jnp.int32(22),
            'checked': jnp.bool_(False), 'help_time': jnp.int32(-1000),
            'responsive': jnp.bool_(False)}


def init_agent_0():
    return _initial_outer()


def init_agent_1():
    return _initial()


def _decode(obs):
    error = jnp.sum(jnp.abs(jnp.asarray(CROPS, jnp.int32) - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    pos = jnp.argmin(jnp.where(jnp.asarray(FLOOR_MASK, jnp.bool_), error, 10000)).astype(jnp.int32)
    face = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    state = pos * 4 + face
    return pos, state, obs[2, 2, 5:10]


def _nearest(targets, state):
    targets = jnp.asarray(targets, jnp.int32)
    return targets[jnp.argmin(jnp.asarray(COSTS, jnp.int32)[targets, state])]


def _cell(obs, pos, target):
    r = target // 9 - pos // 9 + 2
    c = target % 9 - pos % 9 + 2
    visible = (r >= 0) & (r < 5) & (c >= 0) & (c < 5)
    return jnp.where(visible, obs[jnp.clip(r, 0, 4), jnp.clip(c, 0, 4)], 0)


def agent_1_act(obs, reset, memory, rng):
    m = jax.tree.map(lambda x, y: jnp.where(reset, y, x), memory, _initial())
    pos, state, inv = _decode(obs)
    pot = _cell(obs, pos, _POT)
    shown = jnp.max(obs[:, :, 36:39], axis=(0, 1))
    delivered = m['last_cooked'] & (inv[1] == 0) & (m['last_action'] == 5) & ((m['last_target'] == 38) | (m['last_target'] == 42))
    recipe = jnp.where(delivered, jnp.zeros(3, jnp.int32), m['recipe'])
    recipe = jnp.where(jnp.sum(shown) > 0, shown, recipe)
    armed = m['armed'] & ~delivered
    known = jnp.sum(recipe) > 0
    ingredient = jnp.argmax(recipe - pot[31:34])
    full = (jnp.sum(pot[31:34]) >= 3) | (pot[30] > 0) | (pot[39] > 0)
    target = jnp.where(known, jnp.asarray(_SOURCES)[ingredient], _nearest(_BUTTONS, state))
    target = jnp.where(full, _nearest(_PLATES, state), target)
    prearm = full & ~armed & (jnp.sum(inv) == 0) & (m['t'] < 354)
    target = jnp.where(prearm, _nearest(_BUTTONS, state), target)
    target = jnp.where(jnp.sum(inv[2:]) > 0, _POT, target)
    target = jnp.where(inv[0] > 0, _POT, target)
    target = jnp.where(inv[1] > 0, _nearest(_GOALS, state), target)
    action = jnp.asarray(ROUTES, jnp.int32)[target, state]
    action = jnp.where(prearm & (action == 5) & (pot[39] > 6), 4, action)
    armed = armed | (prearm & (action == 5))
    # Use the pot face beside the chosen supply. Returning to that supply
    # then needs one move and one interaction, without an extra turn.
    raw_kind = jnp.argmax(inv[2:])
    raw_row = jnp.where(raw_kind == 0, 3, 5)
    raw_pose = raw_row * 9 + 4
    raw_face = jnp.where(raw_kind == 0, 1, 0)
    raw_turn = jnp.where(raw_kind == 0, 1, 3)
    raw_route = jnp.where(pos // 9 == raw_row,
                          jnp.where(pos % 9 < 4, 0, 2),
                          jnp.where(pos // 9 < raw_row, 1, 3))
    raw_route = jnp.where(pos == raw_pose, jnp.where(state % 4 == raw_face, 5, raw_turn), raw_route)
    action = jnp.where((jnp.sum(inv[2:]) > 0) & (inv[0] == 0), raw_route, action)
    # Route the empty plate to the east pot face: two delivery actions follow
    # the scoop, while the equally short north approach needs four.
    scoop_route = jnp.where(pos == 41, jnp.where(state % 4 == 3, 5, 2),
                            jnp.asarray(ROUTES, jnp.int32)[42, state])
    action = jnp.where((inv[0] > 0) & (inv[1] == 0), scoop_route, action)
    # A recipe read without time for another full dish has negative value.
    reading = (target == 22) | (target == 58)
    action = jnp.where(reading & (m['t'] >= 362), 4, action).astype(jnp.int32)
    next_m = {'t': m['t'] + 1, 'recipe': recipe, 'last_cooked': inv[1] > 0,
              'last_action': action, 'last_target': target, 'armed': armed}
    return action, next_m, {'pos': pos, 'target': target, 'recipe': recipe,
                            'pot_count': jnp.sum(pot[31:34]), 'timer': pot[39], 'delivered': delivered, 'armed': armed}


def agent_0_act(obs, reset, memory, rng):
    m = jax.tree.map(lambda x, y: jnp.where(reset, y, x), memory, _initial_outer())
    pos, state, inv = _decode(obs)
    shown = jnp.max(obs[:, :, 36:39], axis=(0, 1))
    recipe = jnp.where(jnp.sum(shown) > 0, shown, m['recipe'])
    kind = jnp.argmax(recipe)
    target = jnp.asarray(_SOURCES)[kind]
    # Get within view of the western free indicator and acquire the actual
    # requested raw ingredient. It is readable as an ordinary held object.
    wrong = (jnp.sum(inv) > 0) & (inv[2 + kind] == 0)
    empty_counter = jax.vmap(lambda p: jnp.sum(_cell(obs, pos, p)[29:34]) == 0)(jnp.asarray(COUNTERS, jnp.int32))
    counter = jnp.asarray(COUNTERS)[jnp.argmin(jnp.where(empty_counter, jnp.asarray(COSTS)[jnp.asarray(COUNTERS), state], 10000))]
    target = jnp.where(wrong, counter, target)
    target = jnp.where(jnp.sum(recipe) == 0, 36, target)
    action = jnp.asarray(ROUTES, jnp.int32)[target, state]
    # While displaying an ingredient, remain beside its supply. This keeps the
    # always-on recipe display visible, so a recipe change is noticed promptly.
    correct = (inv[2 + kind] > 0) & (jnp.sum(recipe) > 0)
    cue_route = jnp.asarray(ROUTES, jnp.int32)[jnp.asarray(_SOURCES)[kind], state]
    cue_route = jnp.where(cue_route == 5, 4, cue_route)
    action = jnp.where(correct, cue_route, action).astype(jnp.int32)
    # A semantic cue may not be understood by a separately generated cook.
    # Offer a direct public recipe after a prolonged lack of visible work.
    activity = jnp.any(obs[:, :, 15:20] > 0)
    quiet = jnp.where(activity, 0, m['quiet'] + 1).astype(jnp.int32)
    eligible = m['eligible'] | activity
    responsive = m['responsive'] | (activity & (m['t'] - m['help_time'] <= 16))
    patience = jnp.where(responsive, 28, 40)
    enter = (quiet >= patience) & eligible & (m['t'] < 350) & ~m['helping']
    helping = (m['helping'] | enter) & ~activity
    preferred_button = jnp.where(responsive, m['help_button'], _nearest(_BUTTONS, state))
    button = jnp.where(enter, preferred_button, m['help_button'])
    checked = m['checked'] & ~enter
    at_button = (jnp.sum(inv) == 0) & (jnp.asarray(ROUTES, jnp.int32)[button, state] == 5)
    # At the north approach all north-row inner positions are visible; if
    # nobody is seen, the south button covers all possible remaining rows.
    switch = helping & at_button & ~jnp.any(obs[:, :, 10] > 0) & ~checked
    button = jnp.where(switch, 80 - button, button)
    checked = checked | switch
    help_target = jnp.where(jnp.sum(inv) > 0, counter, button)
    help_action = jnp.asarray(ROUTES, jnp.int32)[help_target, state]
    served = helping & (jnp.sum(inv) == 0) & (help_action == 5)
    active = jnp.sum(_cell(obs, pos, button)[36:39]) > 0
    activated = served & ~active
    help_action = jnp.where(served & active, 4, help_action)
    target = jnp.where(helping, help_target, target)
    action = jnp.where(helping, help_action, action)
    eligible = eligible & ~served
    helping = helping & ~served
    quiet = jnp.where(served, 0, quiet)
    next_m = {'t': m['t'] + 1, 'recipe': recipe, 'quiet': quiet,
              'helping': helping, 'eligible': eligible, 'help_button': button,
              'checked': checked, 'help_time': jnp.where(served, m['t'], m['help_time']),
              'responsive': responsive}
    return action, next_m, {'pos': pos, 'target': target, 'recipe': recipe,
                            'quiet': quiet, 'helping': helping, 'activated': activated, 'responsive': responsive}
