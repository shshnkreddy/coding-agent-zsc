"""Reactive, convention-free policy for the asymmetric demo kitchen."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "reactive_asymmetric",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 5,
    "max_width": 11,
    "num_ingredients": 3,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


# Static map channels 20:29. Exact known-layout crops let each player recover
# its global coordinate from the untranslated local observation.
_WALLS = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9),
          (0, 10), (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2),
          (4, 3), (4, 4), (4, 7), (4, 9), (4, 10))
_OBJECTS = {
    0: _WALLS,
    1: ((2, 10),),
    2: ((2, 7),),
    3: ((0, 5), (4, 5)),
    4: (),
    5: ((1, 10), (3, 10)),
    6: ((0, 8), (1, 0)),
    7: ((3, 0), (4, 8)),
    8: ((0, 6), (4, 6)),
}
_POS0_PY = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6))
_POS1_PY = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))


def _locate(obs, seat):
    grid = jnp.zeros((5, 11, 9), dtype=jnp.int32)
    for channel in range(9):
        for row, col in _OBJECTS[channel]:
            grid = grid.at[row, col, channel].set(1)
    padded = jnp.pad(grid, ((2, 2), (2, 2), (0, 0)))
    if seat == 0:
        positions_py = _POS0_PY
    else:
        positions_py = _POS1_PY
    positions = jnp.asarray(positions_py, dtype=jnp.int32)
    crops = jnp.stack(tuple(padded[r:r + 5, c:c + 5, :]
                             for r, c in positions_py))
    delta = jnp.abs(crops - obs[None, :, :, 20:29])
    score = jnp.sum(delta, axis=(1, 2, 3))
    return positions[jnp.argmin(score)]


def _facing(obs):
    # Observation direction order is UP, DOWN, RIGHT, LEFT.
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _navigate(obs, pos, target, final_move):
    """Move in an obstacle-free room, then face and interact at target."""
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    movement = jnp.where(
        dr < 0, UP,
        jnp.where(dr > 0, DOWN,
                  jnp.where(dc < 0, LEFT, jnp.where(dc > 0, RIGHT, STAY))))
    at_target = (dr == 0) & (dc == 0)
    desired_facing = jnp.where(
        final_move == UP, 0,
        jnp.where(final_move == DOWN, 1,
                  jnp.where(final_move == RIGHT, 2, 3))).astype(jnp.int32)
    target_action = jnp.where(_facing(obs) == desired_facing,
                              INTERACT, final_move)
    return jnp.where(at_target, target_action, movement).astype(jnp.int32)


def _visible_recipe(obs, old_recipe):
    counts = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    seen = jnp.sum(counts) > 0
    recipe = jnp.argmax(counts).astype(jnp.int32)
    return jnp.where(seen, recipe, old_recipe), seen


def _pot_state(obs):
    mask = obs[:, :, 22]
    present = jnp.sum(mask) > 0
    total = jnp.sum(obs[:, :, 31:34] * mask[:, :, None])
    cooked = jnp.sum(obs[:, :, 30] * mask) > 0
    timer = jnp.sum(obs[:, :, 39] * mask)
    return present, total.astype(jnp.int32), cooked, timer.astype(jnp.int32)


def init_agent_0():
    # recipe, phase (0 monitor, 1 fetch, 2 insert, 3 cache-drop,
    # 4 cache-retrieve, 5 stage-fetch, 6 stage-drop), cache type, elapsed,
    # insertion guard, cache0, cache1, staged-this-cycle
    return jnp.asarray((-1, 0, 0, 0, 0, 0, 0, 0), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_0()
    mem = jnp.where(reset, initial, memory)
    recipe, recipe_seen = _visible_recipe(obs, mem[0])
    pos = _locate(obs, 0)
    phase = mem[1]
    cache_type = mem[2]
    cache0 = mem[5]
    cache1 = mem[6]
    stage_done = mem[7]

    inv_counts = obs[2, 2, 7:10]
    has_ingredient = jnp.sum(inv_counts) > 0
    held_type = jnp.argmax(inv_counts).astype(jnp.int32)
    pot_present, pot_total, pot_cooked, pot_timer = _pot_state(obs)
    pot_mask = obs[:, :, 22]
    pot_counts = jnp.sum(obs[:, :, 31:34] * pot_mask[:, :, None], axis=(0, 1))
    target_pot_count = jnp.where(recipe == 0, pot_counts[0],
                                 jnp.where(recipe == 1, pot_counts[1], 0))
    pot_compatible = (recipe >= 0) & (pot_total == target_pot_count)

    monitor_row = jnp.where(recipe == 1, 3, 1).astype(jnp.int32)
    monitor = jnp.asarray((monitor_row, 5), dtype=jnp.int32)
    at_monitor = jnp.all(pos == monitor)
    pot_needs_ingredient = (pot_present & (pot_total < 3) &
                            (~pot_cooked) & (pot_timer == 0) & pot_compatible)

    # Inventory observations are authoritative for completion of transfers.
    completed_cache = (~has_ingredient) & ((phase == 3) | (phase == 6))
    retrieved_cache = has_ingredient & (phase == 4)
    cache0 = cache0 + jnp.where(completed_cache & (cache_type == 0), 1, 0)
    cache1 = cache1 + jnp.where(completed_cache & (cache_type == 1), 1, 0)
    cache0 = cache0 - jnp.where(retrieved_cache & (cache_type == 0), 1, 0)
    cache1 = cache1 - jnp.where(retrieved_cache & (cache_type == 1), 1, 0)
    phase = jnp.where(has_ingredient & (phase == 5), 6, phase)
    phase = jnp.where(has_ingredient & (phase != 3) & (phase != 6), 2, phase)
    finished_transfer = ((~has_ingredient) &
                         ((phase == 2) | (phase == 3) | (phase == 6)))
    phase = jnp.where(finished_transfer, 0, phase)
    stage_done = jnp.where(completed_cache & (mem[1] == 6),
                           jnp.minimum(stage_done + 1, 2), stage_done)
    stage_done = jnp.where(pot_needs_ingredient, 0, stage_done)
    recipe_cache = jnp.where(recipe == 1, cache1, cache0)
    begin_retrieve = ((phase == 0) & (recipe >= 0) & at_monitor &
                      pot_needs_ingredient & (recipe_cache > 0))
    reset_fast_path = (mem[3] == 0) & recipe_seen
    begin_fetch = ((phase == 0) & (recipe >= 0) & (recipe_cache == 0) &
                   ((at_monitor & pot_needs_ingredient) | reset_fast_path))
    phase = jnp.where(begin_retrieve, 4, phase)
    cache_type = jnp.where(begin_retrieve, recipe, cache_type)
    phase = jnp.where(begin_fetch, 1, phase)
    incompatible_loaded_pot = (has_ingredient & pot_present &
                               (recipe >= 0) & (~pot_compatible))
    mismatch = (has_ingredient & (recipe >= 0) & (held_type != recipe)) | \
        incompatible_loaded_pot
    phase = jnp.where(mismatch, 3, phase)
    cache_type = jnp.where(mismatch, held_type, cache_type)
    pot_active = pot_present & ((pot_total >= 3) | pot_cooked | (pot_timer > 0))
    planned_stage_type = jnp.where(
        cache0 < cache1, 0,
        jnp.where(cache1 < cache0, 1, recipe)).astype(jnp.int32)
    stage_cache_count = jnp.where(planned_stage_type == 1, cache1, cache0)
    filled_now = (~has_ingredient) & (mem[1] == 2) & pot_active
    chained_stage = (completed_cache & (mem[1] == 6) & (stage_done < 2))
    stage_opportunity = (at_monitor & pot_active) | filled_now | chained_stage
    begin_stage = ((phase == 0) & (recipe >= 0) & stage_opportunity &
                   (stage_done < 2) & (stage_cache_count < 2))
    phase = jnp.where(begin_stage, 5, phase)
    cache_type = jnp.where(begin_stage, planned_stage_type, cache_type)

    action_monitor = _navigate(obs, pos, monitor, UP)
    action_monitor = jnp.where(at_monitor, STAY, action_monitor)

    # Both useful supplies are immediately left of column 1.
    supply_row = jnp.where(recipe == 1, 3, 1).astype(jnp.int32)
    supply = jnp.asarray((supply_row, 1), dtype=jnp.int32)
    action_fetch = _navigate(obs, pos, supply, LEFT)

    # Only interact with the pot while it can accept a raw ingredient.
    pot_target = jnp.asarray((2, 6), dtype=jnp.int32)
    at_pot = jnp.all(pos == pot_target)
    pot_accepts = (pot_present & (pot_total < 3) & (~pot_cooked) &
                   (pot_timer == 0) & pot_compatible)
    blocked_at_pot = (phase == 2) & at_pot & (~pot_accepts)
    insert_guard = jnp.where(blocked_at_pot, 3,
                             jnp.maximum(mem[4] - 1, 0)).astype(jnp.int32)
    action_insert = _navigate(obs, pos, pot_target, RIGHT)
    action_insert = jnp.where(at_pot & ((~pot_accepts) | (insert_guard > 0)),
                              STAY, action_insert)

    # Type-specific private counters form a small transparent ingredient cache.
    # Type 0 uses the top wall; type 1 uses the bottom wall.
    drop_count = jnp.where(cache_type == 1, cache1, cache0)
    drop_col = 4 - jnp.minimum(drop_count, 3)
    drop_row = jnp.where(cache_type == 1, 3, 1).astype(jnp.int32)
    drop_move = jnp.where(cache_type == 1, DOWN, UP).astype(jnp.int32)
    drop_target = jnp.asarray((drop_row, drop_col), dtype=jnp.int32)
    action_dump = _navigate(obs, pos, drop_target, drop_move)

    retrieve_count = jnp.where(cache_type == 1, cache1, cache0)
    retrieve_col = 5 - jnp.maximum(retrieve_count, 1)
    retrieve_row = jnp.where(cache_type == 1, 3, 1).astype(jnp.int32)
    retrieve_move = jnp.where(cache_type == 1, DOWN, UP).astype(jnp.int32)
    retrieve_target = jnp.asarray((retrieve_row, retrieve_col), dtype=jnp.int32)
    action_retrieve = _navigate(obs, pos, retrieve_target, retrieve_move)

    stage_supply_row = jnp.where(cache_type == 1, 3, 1).astype(jnp.int32)
    stage_supply = jnp.asarray((stage_supply_row, 1), dtype=jnp.int32)
    action_stage_fetch = _navigate(obs, pos, stage_supply, LEFT)

    action = jnp.where(
        phase == 1, action_fetch,
        jnp.where(phase == 2, action_insert,
                  jnp.where(phase == 3, action_dump,
                            jnp.where(phase == 4, action_retrieve,
                                      jnp.where(phase == 5, action_stage_fetch,
                                                jnp.where(phase == 6, action_dump,
                                                          action_monitor))))))
    next_memory = jnp.asarray((recipe, phase, cache_type, mem[3] + 1,
                               insert_guard, cache0, cache1, stage_done),
                              dtype=jnp.int32)
    trace = {
        "pos": pos,
        "phase": phase,
        "recipe": recipe,
        "recipe_seen": recipe_seen,
        "pot_total": pot_total,
        "pot_compatible": pot_compatible,
        "held_type": jnp.where(has_ingredient, held_type, -1),
        "insert_guard": insert_guard,
        "cache": jnp.asarray((cache0, cache1), dtype=jnp.int32),
        "stage_done": stage_done,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # elapsed, phase, assists-this-cycle, inferred ingredient, grace clock
    return jnp.asarray((0, 0, 0, 0, 0), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    initial = init_agent_1()
    mem = jnp.where(reset, initial, memory)
    pos = _locate(obs, 1)
    has_plate = obs[2, 2, 5] > 0
    has_cooked = obs[2, 2, 6] > 0
    inv_counts = obs[2, 2, 7:10]
    has_ingredient = jnp.sum(inv_counts) > 0
    pot_present, pot_total, pot_cooked, pot_timer = _pot_state(obs)
    pot_mask = obs[:, :, 22]
    pot_counts = jnp.sum(obs[:, :, 31:34] * pot_mask[:, :, None], axis=(0, 1))
    pure_pot = jnp.sum(pot_counts > 0) == 1

    phase = mem[1]
    helped = mem[2]
    fetch_type = mem[3]
    assist_clock = mem[4]
    hands_empty = (~has_plate) & (~has_cooked) & (~has_ingredient)
    # A delivery is recognizable locally as the transition to an empty pot
    # followed by empty hands. Re-arm one-shot assistance for the next cycle.
    cycle_clear = hands_empty & pot_present & (pot_total == 0)
    helped = jnp.where(cycle_clear, 0, helped)
    phase = jnp.where(cycle_clear, 0, phase)
    assist_clock = jnp.where(cycle_clear, 0, assist_clock)

    # Complete state transitions from observed inventory, never assumed timing.
    phase = jnp.where((phase == 3) & has_ingredient, 4, phase)
    transfer_done = ((phase == 4) | (phase == 5)) & (~has_ingredient)
    phase = jnp.where(transfer_done, 0, phase)
    partial_idle = (pot_present & (pot_total > 0) & (pot_total < 3) &
                    (~pot_cooked) & (pot_timer == 0) & pure_pot)
    clock_active = hands_empty & (phase == 0) & (helped == 1) & partial_idle
    assist_clock = jnp.where(clock_active, assist_clock + 1, 0)
    partner_has_ingredient = jnp.sum(obs[:, :, 17:20]) > 0
    help_eligible = ((helped == 0) |
                     ((helped == 1) & (assist_clock >= 20) &
                      (~partner_has_ingredient)))
    begin_help = (hands_empty & (phase == 0) & (helped < 2) &
                  partial_idle & help_eligible)
    observed_type = jnp.argmax(jnp.sum(obs[:, :, 31:34], axis=(0, 1)))
    fetch_type = jnp.where(begin_help, observed_type, fetch_type).astype(jnp.int32)
    phase = jnp.where(begin_help, 3, phase)
    helped = jnp.where(begin_help, helped + 1, helped)

    delivery_target = jnp.asarray((2, 9), dtype=jnp.int32)
    action_deliver = _navigate(obs, pos, delivery_target, RIGHT)

    pot_target = jnp.asarray((2, 8), dtype=jnp.int32)
    at_pot = jnp.all(pos == pot_target)
    action_plate = _navigate(obs, pos, pot_target, LEFT)
    action_plate = jnp.where(at_pot & (~pot_cooked), STAY, action_plate)

    plate_row = jnp.where(pos[0] <= 2, 1, 3).astype(jnp.int32)
    plate_target = jnp.asarray((plate_row, 9), dtype=jnp.int32)
    action_get_plate = _navigate(obs, pos, plate_target, RIGHT)

    # Before committing the inventory slot to a plate, contribute exactly one
    # matching ingredient inferred from an already-partial pot.
    ingredient_row = jnp.where(fetch_type == 0, 1, 3).astype(jnp.int32)
    ingredient_move = jnp.where(fetch_type == 0, UP, DOWN).astype(jnp.int32)
    ingredient_target = jnp.asarray((ingredient_row, 8), dtype=jnp.int32)
    action_fetch = _navigate(obs, pos, ingredient_target, ingredient_move)
    pot_matches_fetch = pot_total == pot_counts[fetch_type]
    pot_accepts = (pot_present & (pot_total < 3) & (~pot_cooked) &
                   (pot_timer == 0) & pot_matches_fetch)
    action_insert = _navigate(obs, pos, pot_target, LEFT)
    must_dump = (phase == 4) & at_pot & (~pot_accepts)
    phase = jnp.where(must_dump, 5, phase)
    # Shared side counters make a redundant ingredient visible and recoverable
    # to an independently generated left-side partner.
    dump_target = jnp.asarray((ingredient_row, 8), dtype=jnp.int32)
    action_dump = _navigate(obs, pos, dump_target, LEFT)

    # With empty hands, wait at the pot until either a partial pot invites the
    # one-shot assist or a full/cooking pot makes it time to fetch a plate.
    action_wait = _navigate(obs, pos, pot_target, LEFT)
    action_wait = jnp.where(at_pot, STAY, action_wait)
    should_plate = (helped >= 2) | (pot_total >= 3) | pot_cooked | (pot_timer > 0)
    action_empty = jnp.where(should_plate, action_get_plate, action_wait)

    action = jnp.where(has_cooked, action_deliver,
                       jnp.where(has_plate, action_plate,
                                 jnp.where(phase == 3, action_fetch,
                                           jnp.where(phase == 4, action_insert,
                                                     jnp.where(phase == 5, action_dump,
                                                               action_empty)))))
    mode = jnp.where(has_cooked, 2,
                     jnp.where(has_plate, 1, phase)).astype(jnp.int32)
    next_memory = jnp.asarray((mem[0] + 1, phase, helped, fetch_type,
                               assist_clock),
                              dtype=jnp.int32)
    trace = {
        "pos": pos,
        "mode": mode,
        "pot_total": pot_total,
        "pot_cooked": pot_cooked,
        "pot_timer": pot_timer,
        "pot_present": pot_present,
        "pure_pot": pure_pot,
        "helped": helped,
        "fetch_type": fetch_type,
        "assist_clock": assist_clock,
        "partner_has_ingredient": partner_has_ingredient,
    }
    return action.astype(jnp.int32), next_memory, trace
