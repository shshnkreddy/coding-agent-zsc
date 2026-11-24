"""Grounded-coordination signaler/cook policy for the overcooked_v2 layout.

Map facts (fixed, known before play): agent_0's room (cols 1-2) and agent_1's
room (cols 5-6) are disjoint floor regions with no path between them. Agent_0's
room borders the free always-on recipe indicator R (col 0) and the button L
(col 3). Agent_1's room borders both ingredient supplies, the plate supply,
the pot, and the delivery station. The only channel by which agent_1 can learn
the target recipe is the button L, which is inside agent_1's view only when
agent_1 stands in column 5 (row 2, the cell adjacent to the pot). Agent_0
therefore acts as a signaler (read R for free, press L on change/timeout) and
agent_1 acts as a reactive cook/deliverer that waits for a button confirmation
before committing to fetching ingredients.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_coord_signaler_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Movement action ids.
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

# Self-facing channel indices (channels 1-4: UP, DOWN, RIGHT, LEFT).
FACE_UP = 1
FACE_DOWN = 2
FACE_RIGHT = 3
FACE_LEFT = 4

# Static / semantic channel indices from the observation schema.
CH_WALL = 20
CH_GOAL = 21
CH_POT = 22
CH_RECIPE_INDICATOR = 23
CH_BUTTON = 24
CH_PLATE_PILE = 25
CH_ING0_SUPPLY = 26
CH_ING1_SUPPLY = 27
CH_ING2_SUPPLY = 28
CH_ITEM_PLATE = 29
CH_ITEM_COOKED = 30
CH_ITEM_ING0 = 31
CH_ITEM_ING1 = 32
CH_ITEM_ING2 = 33
CH_RECIPE_PLATE = 34
CH_RECIPE_COOKED = 35
CH_RECIPE_ING0 = 36
CH_RECIPE_ING1 = 37
CH_RECIPE_ING2 = 38
CH_POT_TIMER = 39

CH_SELF_PLATE = 5
CH_SELF_COOKED = 6
CH_SELF_ING0 = 7
CH_SELF_ING1 = 8
CH_SELF_ING2 = 9

BROADCAST_TIMEOUT = 50
CONFIRM_WAIT_TIMEOUT = 20


def _find_first(mask):
    """Return (row, col, found) of the first nonzero cell of a 5x5 mask."""
    flat = mask.reshape(-1)
    idx = jnp.argmax(flat)
    found = flat[idx] > 0
    row = idx // 5
    col = idx % 5
    return row, col, found


def _nav_action(obs, mask, target_dr, target_dc, arrive_move_action, facing_channel):
    """Greedy relative-offset navigation toward an object, then interact.

    ``target_dr``/``target_dc`` describe the desired (object_row - self_row,
    object_col - self_col) offset, i.e. the adjacency cell from which the
    object can be interacted with. ``arrive_move_action`` is the movement
    action id whose resulting facing matches that adjacency direction.
    """
    row, col, found = _find_first(mask)
    offset_row = row - 2
    offset_col = col - 2
    arrived = jnp.logical_and(offset_row == target_dr, offset_col == target_dc)
    facing_ok = obs[2, 2, facing_channel] == 1
    ready = jnp.logical_and(arrived, facing_ok)

    # offset = object_pos - self_pos. To shrink offset toward target, move
    # self *toward* the object: offset too large -> step in the increasing
    # direction (DOWN/RIGHT); offset too small -> step the other way.
    move_row = jnp.where(offset_row > target_dr, DOWN,
               jnp.where(offset_row < target_dr, UP, jnp.int32(-1)))
    move_col = jnp.where(offset_col > target_dc, RIGHT,
               jnp.where(offset_col < target_dc, LEFT, jnp.int32(-1)))
    nav = jnp.where(move_row >= 0, move_row, jnp.where(move_col >= 0, move_col, STAY))

    action = jnp.where(ready, INTERACT, jnp.where(arrived, arrive_move_action, nav))
    action = jnp.where(found, action, STAY)
    return action.astype(jnp.int32), ready, found


def _decode_recipe_at(obs, row, col):
    ing0 = obs[row, col, CH_RECIPE_ING0]
    ing1 = obs[row, col, CH_RECIPE_ING1]
    return jnp.where(ing0 > 0, jnp.int32(0), jnp.where(ing1 > 0, jnp.int32(1), jnp.int32(-1)))


def init_agent_0():
    return {
        'last_broadcast': jnp.asarray(-1, dtype=jnp.int32),
        'steps_since': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    init = init_agent_0()
    memory = jax.tree_util.tree_map(lambda i, m: jnp.where(reset, i, m), init, memory)
    last_broadcast = memory['last_broadcast']
    steps_since = memory['steps_since']

    r_row, r_col, r_found = _find_first(obs[:, :, CH_RECIPE_INDICATOR])
    true_recipe = jnp.where(r_found, _decode_recipe_at(obs, r_row, r_col), jnp.int32(-1))

    need_broadcast = jnp.logical_and(
        true_recipe >= 0,
        jnp.logical_or(true_recipe != last_broadcast, steps_since >= BROADCAST_TIMEOUT),
    )

    approach_action, _ready, _found = _nav_action(
        obs, obs[:, :, CH_BUTTON], 0, 1, RIGHT, FACE_RIGHT
    )
    action = jnp.where(need_broadcast, approach_action, STAY).astype(jnp.int32)

    did_press = action == INTERACT
    new_last_broadcast = jnp.where(did_press, true_recipe, last_broadcast).astype(jnp.int32)
    new_steps_since = jnp.where(did_press, jnp.int32(0), steps_since + 1).astype(jnp.int32)

    new_memory = {'last_broadcast': new_last_broadcast, 'steps_since': new_steps_since}
    return action, new_memory, {}


def init_agent_1():
    return {
        'known_recipe': jnp.asarray(-1, dtype=jnp.int32),
        'confirmed': jnp.asarray(0, dtype=jnp.int32),
        'wait_counter': jnp.asarray(0, dtype=jnp.int32),
        'prev_holding_dish': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    init = init_agent_1()
    memory = jax.tree_util.tree_map(lambda i, m: jnp.where(reset, i, m), init, memory)
    known_recipe = memory['known_recipe']
    confirmed = memory['confirmed']
    wait_counter = memory['wait_counter']
    prev_holding_dish = memory['prev_holding_dish']

    # Opportunistically decode the button signal whenever it is in view.
    l_row, l_col, l_found = _find_first(obs[:, :, CH_BUTTON])
    signal_recipe = _decode_recipe_at(obs, l_row, l_col)
    signal_valid = jnp.logical_and(l_found, signal_recipe >= 0)
    known_recipe = jnp.where(signal_valid, signal_recipe, known_recipe).astype(jnp.int32)
    confirmed = jnp.where(signal_valid, jnp.int32(1), confirmed)

    # Own holdings, derived purely from the current observation.
    hold_plate = obs[2, 2, CH_SELF_PLATE]
    hold_cooked = obs[2, 2, CH_SELF_COOKED]
    hold_ingr_sum = obs[2, 2, CH_SELF_ING0] + obs[2, 2, CH_SELF_ING1] + obs[2, 2, CH_SELF_ING2]
    holding_dish = jnp.logical_and(hold_plate > 0, hold_cooked > 0)
    holding_plate_empty = jnp.logical_and(hold_plate > 0, hold_cooked == 0)
    holding_ingr = hold_ingr_sum > 0
    holding_nothing = jnp.logical_and(
        jnp.logical_and(hold_plate == 0, hold_cooked == 0), hold_ingr_sum == 0
    )

    # Pot contents, always visible from anywhere in this room.
    p_row, p_col, _p_found = _find_first(obs[:, :, CH_POT])
    pot_total = (
        obs[p_row, p_col, CH_ITEM_ING0]
        + obs[p_row, p_col, CH_ITEM_ING1]
        + obs[p_row, p_col, CH_ITEM_ING2]
    )
    pot_cooked = obs[p_row, p_col, CH_ITEM_COOKED] > 0

    # Right after delivering a dish, force a fresh confirmation window before
    # trusting (possibly stale) known_recipe again.
    just_delivered = jnp.logical_and(prev_holding_dish == 1, jnp.logical_not(holding_dish))
    confirmed = jnp.where(just_delivered, jnp.int32(0), confirmed)
    wait_counter = jnp.where(just_delivered, jnp.int32(0), wait_counter)

    cycle_start = jnp.logical_and(pot_total == 0, holding_nothing)
    can_fetch = jnp.logical_and(
        known_recipe >= 0, jnp.logical_or(confirmed == 1, wait_counter >= CONFIRM_WAIT_TIMEOUT)
    )
    waiting_phase = jnp.logical_and(cycle_start, jnp.logical_not(can_fetch))
    wait_counter = jnp.where(waiting_phase, wait_counter + 1, wait_counter).astype(jnp.int32)

    pot_left_act, _r1, _f1 = _nav_action(obs, obs[:, :, CH_POT], 0, -1, LEFT, FACE_LEFT)
    deliver_act, _r2, _f2 = _nav_action(obs, obs[:, :, CH_GOAL], 0, 1, RIGHT, FACE_RIGHT)
    plate_act, _r3, _f3 = _nav_action(obs, obs[:, :, CH_PLATE_PILE], 0, -1, LEFT, FACE_LEFT)
    ingr_mask = jnp.where(known_recipe == 0, obs[:, :, CH_ING0_SUPPLY], obs[:, :, CH_ING1_SUPPLY])
    ingr_act, _r4, _f4 = _nav_action(obs, ingr_mask, 0, 1, RIGHT, FACE_RIGHT)

    action = jnp.where(
        waiting_phase, pot_left_act,
        jnp.where(
            holding_dish, deliver_act,
            jnp.where(
                jnp.logical_and(pot_cooked, holding_plate_empty), pot_left_act,
                jnp.where(
                    jnp.logical_and(pot_total == 3, jnp.logical_and(
                        jnp.logical_not(holding_plate_empty), jnp.logical_not(holding_ingr))),
                    plate_act,
                    jnp.where(
                        holding_ingr, pot_left_act,
                        jnp.where(pot_total < 3, ingr_act, pot_left_act),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    new_memory = {
        'known_recipe': known_recipe,
        'confirmed': confirmed.astype(jnp.int32),
        'wait_counter': wait_counter,
        'prev_holding_dish': holding_dish.astype(jnp.int32),
    }
    return action, new_memory, {}
