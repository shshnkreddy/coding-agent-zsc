"""Cooperative policy for Overcooked V2 layout `grounded_coord_ring`.

Map geometry (fixed, from context/environment): agent_0 spawns on the
24-cell outer ring and can reach ingredient supply 2 (unused by any
recipe), both plate piles, both delivery stations and the always-on
recipe indicators, but it can never reach the pot, ingredient supply 0,
or ingredient supply 1 -- those cells are adjacent only to the 8-cell
inner ring where agent_1 spawns. Only agent_1 borders the pot. agent_1's
inner ring is also adjacent to both ingredient supplies 0/1, both plate
piles, both delivery stations and both recipe buttons, so agent_1 alone
can complete an entire cook-and-deliver cycle. Since agent_0 has no
physical way to hand off anything useful across the two disjoint regions
under this fixed layout, agent_1 is built to be fully self-sufficient
(including self-activating the recipe button to learn the target
recipe) rather than depending on any signalling convention with an
independently generated partner. agent_0 stays out of the way.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}


# ---- agent_0: outer ring; cannot reach the pot or ingredients 0/1 here ----
#
# agent_0 cannot reach the pot or ingredient supplies 0/1, so it has no way
# to help our own self-sufficient agent_1 in self-play. But four wall cells
# -- (2,3), (2,5), (6,3), (6,5) -- are physically shared counters between
# the outer ring (agent_0) and the inner ring (agent_1): a real, grounded
# hand-off surface, not a coded convention. agent_0 is memoryless and purely
# reactive: it patrols its 24-cell ring by wall-following, and whenever a
# plate pile or a hand-off counter is an immediate neighbor it opportunistically
# picks up / drops a plate there. None of this carries any direct reward, so
# it cannot change agent_0's own self-play contribution (always 0); it only
# gives an independently generated partner agent_1 a real object to use if
# its own strategy looks for one, without agent_0 assuming any convention
# about what a partner will do with it.

_DIR_DR = (-1, 1, 0, 0)   # UP, DOWN, RIGHT, LEFT
_DIR_DC = (0, 0, 1, -1)
_ACTION_FOR_DIR = (3, 1, 0, 2)
_CW = (2, 3, 1, 0)
_CCW = (3, 2, 0, 1)
_OPP = (1, 0, 3, 2)


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def _static_sum(obs, r, c):
    return jnp.sum(obs[r, c, 20:29])


def _is_blocked(obs, dr, dc):
    return _static_sum(obs, 2 + dr, 2 + dc) > 0


def _is_real_beyond(obs, dr, dc):
    tr, tc = 2 + 2 * dr, 2 + 2 * dc
    pr, pc = (0, 1) if dr != 0 else (1, 0)
    total = (
        _static_sum(obs, tr, tc)
        + _static_sum(obs, tr + pr, tc + pc)
        + _static_sum(obs, tr - pr, tc - pc)
    )
    return total > 0


def agent_0_act(obs, reset, memory, rng):
    current_dir = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    cw = jnp.asarray(_CW, dtype=jnp.int32)
    ccw = jnp.asarray(_CCW, dtype=jnp.int32)
    opp = jnp.asarray(_OPP, dtype=jnp.int32)
    action_for_dir = jnp.asarray(_ACTION_FOR_DIR, dtype=jnp.int32)

    candidate_dirs = [cw[current_dir], current_dir, ccw[current_dir], opp[current_dir]]
    wallfollow_dir = candidate_dirs[-1]
    for cand in reversed(candidate_dirs[:-1]):
        dr = jnp.asarray(_DIR_DR, dtype=jnp.int32)[cand]
        dc = jnp.asarray(_DIR_DC, dtype=jnp.int32)[cand]
        wallfollow_dir = jnp.where(_is_blocked(obs, dr, dc), wallfollow_dir, cand)
    wallfollow_action = action_for_dir[wallfollow_dir]

    holding_nothing = jnp.sum(obs[2, 2, 5:10]) == 0
    holding_plate_only = (obs[2, 2, 5] == 1) & (obs[2, 2, 6] == 0)

    target_dir = jnp.asarray(3, dtype=jnp.int32)
    found = jnp.asarray(False)
    for d in (3, 2, 1, 0):
        dr, dc = _DIR_DR[d], _DIR_DC[d]
        plate_here = obs[2 + dr, 2 + dc, 25] == 1
        wall_here = obs[2 + dr, 2 + dc, 20] == 1
        wall_item_free = jnp.sum(obs[2 + dr, 2 + dc, 29:34]) == 0
        pickup_ok = holding_nothing & plate_here
        drop_ok = holding_plate_only & wall_here & wall_item_free & _is_real_beyond(obs, dr, dc)
        active = pickup_ok | drop_ok
        target_dir = jnp.where(active, d, target_dir)
        found = found | active

    reactive_action = jnp.where(current_dir == target_dir, _INTERACT, action_for_dir[target_dir])
    action = jnp.where(found, reactive_action, wallfollow_action)

    zero = jnp.asarray(0, dtype=jnp.int32)
    return action.astype(jnp.int32), jnp.where(reset, zero, memory), {}


# ---- agent_1: inner ring; sole access to the pot and to ingredients 0/1 ----
#
# The inner ring is a cycle of 8 cells around the pot:
#   node 0 (3,3)  ingredient-0 supply is one step LEFT
#   node 1 (3,4)  the (2,4) recipe button is one step UP
#   node 2 (3,5)  a plate pile is one step RIGHT
#   node 3 (4,5)  pot is one step LEFT / delivery (4,6) is one step RIGHT
#   node 4 (5,5)  a plate pile is one step RIGHT
#   node 5 (5,4)  the (6,4) recipe button is one step DOWN
#   node 6 (5,3)  ingredient-1 supply is one step LEFT
#   node 7 (4,3)  pot is one step RIGHT / delivery (4,2) is one step LEFT
#
# The pot is visible from every ring cell (max distance 2), and its offset
# relative to self uniquely identifies the current node, so no absolute
# position memory is required to navigate.

_NODE_DR = (1, 1, 1, 0, -1, -1, -1, 0)
_NODE_DC = (1, 0, -1, -1, -1, 0, 1, 1)
_EDGE_FWD = (0, 0, 1, 1, 2, 2, 3, 3)
_EDGE_BWD = (1, 2, 2, 3, 3, 0, 0, 1)

_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5
_CH_UP, _CH_DOWN, _CH_RIGHT, _CH_LEFT = 1, 2, 3, 4


def _localize(obs):
    node_dr = jnp.asarray(_NODE_DR, dtype=jnp.int32)
    node_dc = jnp.asarray(_NODE_DC, dtype=jnp.int32)
    pot_channel = obs[:, :, 22]
    flat = jnp.argmax(pot_channel.reshape(-1))
    i = flat // 5
    j = flat % 5
    dr = (i - 2).astype(jnp.int32)
    dc = (j - 2).astype(jnp.int32)
    matches = (node_dr == dr) & (node_dc == dc)
    return jnp.argmax(matches).astype(jnp.int32)


def _nav_action(cur, target):
    edge_fwd = jnp.asarray(_EDGE_FWD, dtype=jnp.int32)
    edge_bwd = jnp.asarray(_EDGE_BWD, dtype=jnp.int32)
    fdist = jnp.mod(target - cur, 8)
    bdist = jnp.mod(cur - target, 8)
    return jnp.where(fdist <= bdist, edge_fwd[cur], edge_bwd[cur])


def _facing_ok(obs, channel):
    return obs[2, 2, channel] == 1


def _branch_goto_button(obs, mem):
    cur = _localize(obs)
    at_node = cur == 1
    face_good = _facing_ok(obs, _CH_UP)
    action = jnp.where(at_node, jnp.where(face_good, _INTERACT, _UP), _nav_action(cur, 1))
    ready = at_node & face_good
    next_phase = jnp.where(ready, 1, 0)
    return action.astype(jnp.int32), next_phase, mem['target_ing'], mem['deposited']


def _branch_read_recipe(obs, mem):
    ing0 = obs[1, 2, 36]
    ing1 = obs[1, 2, 37]
    visible = (ing0 + ing1) > 0
    target_ing = jnp.where(ing0 > 0, 0, 1).astype(jnp.int32)
    next_node = jnp.where(target_ing == 0, 0, 6)
    move_action = _nav_action(_localize(obs), next_node)
    action = jnp.where(visible, move_action, _STAY)
    next_phase = jnp.where(visible, 2, 1)
    next_target_ing = jnp.where(visible, target_ing, mem['target_ing'])
    next_deposited = jnp.where(visible, jnp.asarray(0, jnp.int32), mem['deposited'])
    return action.astype(jnp.int32), next_phase, next_target_ing, next_deposited


def _branch_fetch_ingredient(obs, mem):
    target_node = jnp.where(mem['target_ing'] == 0, 0, 6)
    cur = _localize(obs)
    at_node = cur == target_node
    face_good = _facing_ok(obs, _CH_LEFT)
    action = jnp.where(at_node, jnp.where(face_good, _INTERACT, _LEFT), _nav_action(cur, target_node))
    ready = at_node & face_good
    next_phase = jnp.where(ready, 3, 2)
    return action.astype(jnp.int32), next_phase, mem['target_ing'], mem['deposited']


def _branch_deposit(obs, mem):
    cur = _localize(obs)
    at_node = cur == 7
    face_good = _facing_ok(obs, _CH_RIGHT)
    action = jnp.where(at_node, jnp.where(face_good, _INTERACT, _RIGHT), _nav_action(cur, 7))
    ready = at_node & face_good
    new_deposited = jnp.where(ready, mem['deposited'] + 1, mem['deposited'])
    next_phase = jnp.where(ready, jnp.where(new_deposited >= 3, 4, 2), 3)
    return action.astype(jnp.int32), next_phase, mem['target_ing'], new_deposited


def _branch_get_plate(obs, mem):
    cur = _localize(obs)
    at_node = cur == 2
    face_good = _facing_ok(obs, _CH_RIGHT)
    action = jnp.where(at_node, jnp.where(face_good, _INTERACT, _RIGHT), _nav_action(cur, 2))
    ready = at_node & face_good
    next_phase = jnp.where(ready, 5, 4)
    return action.astype(jnp.int32), next_phase, mem['target_ing'], mem['deposited']


def _branch_wait_cook(obs, mem):
    cur = _localize(obs)
    at_node = cur == 7
    face_good = _facing_ok(obs, _CH_RIGHT)
    cooked = obs[2, 3, 30] == 1
    action_when_facing = jnp.where(cooked, _INTERACT, _STAY)
    action = jnp.where(at_node, jnp.where(face_good, action_when_facing, _RIGHT), _nav_action(cur, 7))
    ready = at_node & face_good & cooked
    next_phase = jnp.where(ready, 6, 5)
    return action.astype(jnp.int32), next_phase, mem['target_ing'], mem['deposited']


def _branch_deliver(obs, mem):
    cur = _localize(obs)
    at_node = cur == 7
    face_good = _facing_ok(obs, _CH_LEFT)
    action = jnp.where(at_node, jnp.where(face_good, _INTERACT, _LEFT), _nav_action(cur, 7))
    ready = at_node & face_good
    next_phase = jnp.where(ready, 0, 6)
    next_deposited = jnp.where(ready, jnp.asarray(0, jnp.int32), mem['deposited'])
    return action.astype(jnp.int32), next_phase, mem['target_ing'], next_deposited


_BRANCHES = [
    _branch_goto_button,
    _branch_read_recipe,
    _branch_fetch_ingredient,
    _branch_deposit,
    _branch_get_plate,
    _branch_wait_cook,
    _branch_deliver,
]


# A fresh cycle (button press -> 3 ingredients -> cook -> plate -> deliver)
# takes roughly 40-50 steps in this layout. Once too few steps remain in the
# fixed 400-step episode to plausibly finish a new cycle, self-activating the
# button would only pay its -5 cost with no chance of an offsetting +20
# delivery, so a new cycle is not started that late.
_MAX_STEPS = 400
_LATE_CUTOFF = _MAX_STEPS - 40


def init_agent_1():
    return {
        'phase': jnp.asarray(0, dtype=jnp.int32),
        'target_ing': jnp.asarray(0, dtype=jnp.int32),
        'deposited': jnp.asarray(0, dtype=jnp.int32),
        'steps': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    init_mem = init_agent_1()
    mem = jax.tree_util.tree_map(lambda i, m: jnp.where(reset, i, m), init_mem, memory)
    action, next_phase, next_target_ing, next_deposited = jax.lax.switch(mem['phase'], _BRANCHES, obs, mem)

    too_late = mem['steps'] >= _LATE_CUTOFF
    would_press = (mem['phase'] == 0) & (action == _INTERACT)
    hold_off = too_late & would_press
    action = jnp.where(hold_off, _STAY, action)
    next_phase = jnp.where(hold_off, 0, next_phase)

    next_memory = {
        'phase': next_phase.astype(jnp.int32),
        'target_ing': next_target_ing.astype(jnp.int32),
        'deposited': next_deposited.astype(jnp.int32),
        'steps': jnp.where(reset, jnp.asarray(0, jnp.int32), mem['steps'] + 1).astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, {}
