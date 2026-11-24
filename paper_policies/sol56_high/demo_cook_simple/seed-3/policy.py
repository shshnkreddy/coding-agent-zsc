"""Reactive asymmetric policy for the demo_cook_simple layout.

Both seats infer their global location by matching the static part of their
local observation against the known layout. The left seat produces the
currently displayed ingredient; the right seat plates ready pots and serves.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'reactive_roles_v12_inventory_guard',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}


# Static channel index (within observation channels 20:29) at every object.
_OBJECTS = {
    (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0, (0, 4): 0,
    (0, 5): 3, (0, 6): 8, (0, 7): 0, (0, 8): 6,
    (0, 9): 0, (0, 10): 0,
    (1, 0): 6, (1, 7): 0, (1, 10): 5,
    (2, 0): 0, (2, 7): 2, (2, 10): 1,
    (3, 0): 7, (3, 7): 0, (3, 10): 5,
    (4, 0): 0, (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0,
    (4, 5): 3, (4, 6): 8, (4, 7): 0, (4, 8): 7,
    (4, 9): 0, (4, 10): 0,
}

_POS0_LIST = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
)
_POS1_LIST = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))


def _static_patch(row, col):
    patch = [[[0] * 9 for _ in range(5)] for _ in range(5)]
    for lr in range(5):
        for lc in range(5):
            key = (row + lr - 2, col + lc - 2)
            if key in _OBJECTS:
                patch[lr][lc][_OBJECTS[key]] = 1
    return patch


def _locate(obs, position_list):
    positions = jnp.asarray(position_list, dtype=jnp.int32)
    patches = jnp.asarray(
        [_static_patch(r, c) for r, c in position_list], dtype=jnp.int32)
    difference = patches - obs[None, :, :, 20:29]
    score = jnp.sum(difference * difference, axis=(1, 2, 3))
    return positions[jnp.argmin(score)]


def _move_to(pos, target_row, target_col):
    """Shortest unobstructed move in either rectangular room."""
    vertical = jnp.where(pos[0] < target_row, 1, 3)
    horizontal = jnp.where(pos[1] < target_col, 0, 2)
    return jnp.where(pos[0] != target_row, vertical,
                     jnp.where(pos[1] != target_col, horizontal, 4))


def _at_object_action(obs, pos, stand_row, stand_col, face_channel, turn_action):
    move = _move_to(pos, stand_row, stand_col)
    at_stand = (pos[0] == stand_row) & (pos[1] == stand_col)
    facing = obs[2, 2, face_channel] > 0
    use = jnp.where(facing, 5, turn_action)
    return jnp.where(at_stand, use, move)


def _global_item_present(obs, pos, row, col):
    """Read a known global counter when it lies in the local crop."""
    local_row = row - pos[0] + 2
    local_col = col - pos[1] + 2
    visible = ((local_row >= 0) & (local_row < 5) &
               (local_col >= 0) & (local_col < 5))
    local_row = jnp.clip(local_row, 0, 4)
    local_col = jnp.clip(local_col, 0, 4)
    present = jnp.sum(obs[local_row, local_col, 29:34]) > 0
    return visible, present


def init_agent_0():
    # recipe, remembered pot state (0 empty/1 partial/2 busy), cache flags,
    # staging phase (0 pot-bound, 1 stocking type 0, 2 stocking type 1),
    # and the duration of an observable partial-pot assistance wait.
    # Final fields: delivery-sync phase and consecutive pot-unseen steps.
    return jnp.asarray([-1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_0()
    memory = jnp.where(reset, initial, memory)
    pos = _locate(obs, _POS0_LIST)

    displayed_0 = jnp.sum(obs[:, :, 36])
    displayed_1 = jnp.sum(obs[:, :, 37])
    sees_recipe = (displayed_0 + displayed_1) > 0
    observed_recipe = jnp.where(displayed_1 > displayed_0, 1, 0)
    recipe = jnp.where(sees_recipe, observed_recipe, memory[0])

    held_0 = obs[2, 2, 7]
    held_1 = obs[2, 2, 8]
    holding_raw = (held_0 + held_1) > 0

    pot_visible = jnp.sum(obs[:, :, 22]) > 0
    pot_count = jnp.sum(obs[:, :, 22] *
                        (obs[:, :, 31] + obs[:, :, 32] + obs[:, :, 33]))
    pot_cooked = jnp.sum(obs[:, :, 22] * obs[:, :, 30]) > 0
    pot_timer = jnp.sum(obs[:, :, 22] * obs[:, :, 39])
    pot_busy = (pot_count >= 3) | pot_cooked | (pot_timer > 0)

    observed_pot_state = jnp.where(pot_busy, 2, jnp.where(pot_count > 0, 1, 0))
    just_collected = (pot_visible & (memory[1] == 2) &
                      (observed_pot_state == 0) & (memory[7] == 0))
    pot_state = jnp.where(pot_visible, observed_pot_state, memory[1])
    pot_unseen = jnp.where(
        pot_visible, 0, jnp.minimum(memory[7] + 1, 100))

    cache0_visible, cache0_observed = _global_item_present(obs, pos, 0, 4)
    cache1_visible, cache1_observed = _global_item_present(obs, pos, 4, 4)
    cache0 = jnp.where(cache0_visible, cache0_observed, memory[2]).astype(jnp.int32)
    cache1 = jnp.where(cache1_visible, cache1_observed, memory[3]).astype(jnp.int32)

    phase = memory[4]
    stage_finished = (~holding_raw) & (
        ((phase == 1) & (cache0 > 0)) | ((phase == 2) & (cache1 > 0)))
    phase = jnp.where(stage_finished, 0, phase)
    begin_stage0 = (~holding_raw) & (phase == 0) & (pot_state == 2) & (cache0 == 0)
    begin_stage1 = ((~holding_raw) & (phase == 0) & (pot_state == 2) &
                    (cache0 > 0) & (cache1 == 0))
    phase = jnp.where(begin_stage0, 1, jnp.where(begin_stage1, 2, phase))

    partial_waiting = ((phase == 0) & (~holding_raw) & pot_visible &
                       (pot_count > 0) & (pot_count < 3) & (~pot_busy))
    wait_count = jnp.where(
        pot_state != 1, 0,
        jnp.where(memory[5] >= 6, 6,
                  jnp.where(partial_waiting,
                            jnp.minimum(memory[5] + 1, 6), memory[5])))

    # Unknown recipes are resolved for free from position (2,5), which sees
    # both always-on indicators. A raw ingredient always takes priority.
    seek_recipe = _move_to(pos, 2, 5)
    add_to_pot = _at_object_action(obs, pos, 2, 6, 3, 0)  # face RIGHT
    supply_row = jnp.where(recipe == 1, 3, 1)
    fetch = _at_object_action(obs, pos, supply_row, 1, 4, 2)  # face LEFT
    fetch0 = _at_object_action(obs, pos, 1, 1, 4, 2)
    fetch1 = _at_object_action(obs, pos, 3, 1, 4, 2)
    counter0 = _at_object_action(obs, pos, 1, 4, 1, 3)  # face UP
    counter1 = _at_object_action(obs, pos, 3, 4, 2, 1)  # face DOWN

    selected_cache = jnp.where(recipe == 1, cache1, cache0)
    retrieve_cache = jnp.where(recipe == 1, counter1, counter0)
    normal_empty = jnp.where(
        recipe < 0, seek_recipe,
        jnp.where(pot_state == 2, _move_to(pos, 2, 5),
                  jnp.where((pot_state == 1) & (wait_count < 6),
                            _move_to(pos, 2, 6),
                            jnp.where(selected_cache > 0,
                                      retrieve_cache, fetch))))
    empty_action = jnp.where(
        phase == 1, fetch0, jnp.where(phase == 2, fetch1, normal_empty))

    stage_held = jnp.where(phase == 1, counter0, counter1)
    # If a partner unexpectedly fills the pot while a pot-bound ingredient is
    # in hand, preserve it in its type-specific cache instead of deadlocking.
    held_type = jnp.where(held_1 > held_0, 1, 0)
    overflow_cache = jnp.where(held_type == 1, counter1, counter0)
    pot_bound = jnp.where(pot_state == 2, overflow_cache, add_to_pot)
    held_action = jnp.where(phase > 0, stage_held, pot_bound)
    action = jnp.where(holding_raw, held_action, empty_action)

    # A promptly observed pot pickup precedes delivery by two actions. Move to
    # the cache midpoint, wait through that transition, then let the freshly
    # observed recipe select a vertical cache route. If the pot was out of view
    # while staging, do not apply this synchronization delay retroactively.
    delivery_phase = jnp.where(just_collected, 1, memory[6])
    syncing = delivery_phase == 1
    sync_action = _move_to(pos, 2, 4)
    action = jnp.where((~holding_raw) & syncing, sync_action, action)
    at_midpoint = (pos[0] == 2) & (pos[1] == 4)
    next_delivery_phase = jnp.where(
        (delivery_phase == 1) & at_midpoint, 2,
        jnp.where(delivery_phase == 2, 0, delivery_phase))

    next_memory = jnp.asarray(
        [recipe, pot_state, cache0, cache1, phase, wait_count,
         next_delivery_phase, pot_unseen],
        dtype=jnp.int32)
    trace = {
        'position': pos,
        'recipe': recipe,
        'pot_count': pot_count,
        'pot_state': pot_state,
        'cache0': cache0,
        'cache1': cache1,
        'phase': phase,
        'partial_wait': wait_count,
        'delivery_sync': delivery_phase,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # 0 ready, 1 first contribution in progress, 2 first complete,
    # 3 second contribution in progress, 4 second complete.
    return jnp.asarray([0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    pos = _locate(obs, _POS1_LIST)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = (obs[2, 2, 7] + obs[2, 2, 8]) > 0
    pot_count_0 = jnp.sum(obs[:, :, 22] * obs[:, :, 31])
    pot_count_1 = jnp.sum(obs[:, :, 22] * obs[:, :, 32])
    pot_count = pot_count_0 + pot_count_1 + jnp.sum(
        obs[:, :, 22] * obs[:, :, 33])
    pot_ready = jnp.sum(obs[:, :, 22] * obs[:, :, 30]) > 0
    pot_timer = jnp.sum(obs[:, :, 22] * obs[:, :, 39])
    pot_busy = (pot_count >= 3) | pot_ready | (pot_timer > 0)
    assist_stage = memory[0]
    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_empty = jnp.sum(obs[:, :, 15:20]) == 0
    partner_raw = ((jnp.sum(obs[:, :, 17:20]) > 0) &
                   (jnp.sum(obs[:, :, 15:17]) == 0))

    get_plate = _at_object_action(obs, pos, 1, 9, 3, 0)  # face RIGHT
    wait_at_pot = _move_to(pos, 2, 8)
    collect = _at_object_action(obs, pos, 2, 8, 4, 2)  # face LEFT
    serve = _at_object_action(obs, pos, 2, 9, 3, 0)  # face RIGHT

    # Once the producer reveals the homogeneous recipe by placing the first
    # ingredient in the pot, contribute matching ingredients from the adjacent
    # top/bottom supply. Visible producer inventory and stalled-pot evidence
    # decide whether one or two contributions are appropriate.
    ingredient_row = jnp.where(pot_count_1 > pot_count_0, 3, 1)
    fetch_ingredient = _at_object_action(
        obs, pos, ingredient_row, 8,
        jnp.where(ingredient_row == 1, 1, 2),
        jnp.where(ingredient_row == 1, 3, 1))
    add_ingredient = _at_object_action(obs, pos, 2, 8, 4, 2)
    discard_ingredient = _at_object_action(obs, pos, 1, 9, 1, 3)

    plate_action = jnp.where(pot_ready, collect, wait_at_pot)
    raw_action = jnp.where(pot_busy, discard_ingredient, add_ingredient)
    finished_first = ((assist_stage == 1) & (~held_raw) &
                      (pot_count >= 2))
    finished_second = ((assist_stage == 3) & (~held_raw) &
                       (pot_count >= 3))
    assist_stage = jnp.where(
        finished_first, 2, jnp.where(finished_second, 4, assist_stage))
    stalled_two = ((assist_stage == 2) & (~held_raw) & (~held_plate) &
                   (pot_count == 2) & (~pot_busy))
    stall_count = jnp.where(
        stalled_two, jnp.minimum(memory[1] + 1, 10), 0)
    first_assist = ((assist_stage == 0) & (pot_count == 1) & (~pot_busy) &
                    (~(partner_visible & partner_raw)))
    second_assist = ((assist_stage == 2) & (pot_count == 2) &
                     (~pot_busy) &
                     ((partner_visible & partner_empty) | (stall_count >= 10)))
    assist_stage = jnp.where(
        first_assist, 1, jnp.where(second_assist, 3, assist_stage))
    assist_in_progress = (assist_stage == 1) | (assist_stage == 3)
    waiting_for_stall = (assist_stage == 2) & (pot_count == 2)
    empty_action = jnp.where(
        pot_count == 0, wait_at_pot,
        jnp.where(assist_in_progress, fetch_ingredient,
                  jnp.where(waiting_for_stall, wait_at_pot, get_plate)))
    action = jnp.where(
        held_cooked, serve,
        jnp.where(held_raw, raw_action,
                  jnp.where(held_plate, plate_action, empty_action)))

    next_stage = jnp.where(pot_count == 0, 0, assist_stage)
    next_memory = jnp.asarray([next_stage, stall_count], dtype=jnp.int32)
    trace = {
        'position': pos,
        'pot_ready': pot_ready.astype(jnp.int32),
        'inventory_mode': (
            held_plate.astype(jnp.int32) + 2 * held_cooked.astype(jnp.int32)),
        'pot_count': pot_count,
        'assisted': next_stage,
        'stall_count': stall_count,
    }
    return action.astype(jnp.int32), next_memory, trace
