"""Partition-aligned producer/server policy for demo_cook_simple."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "overflow_buffer_v11",
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


# Templates for observation channels 20:29 let each seat infer its global
# floor cell from the unrotated local crop.
_OBJECTS = (
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10),
     (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
     (4, 7), (4, 9), (4, 10)),
    ((2, 10),),                 # goal
    ((2, 7),),                  # pot
    ((0, 5), (4, 5)),           # always-on recipe indicators
    (),                         # button indicators
    ((1, 10), (3, 10)),         # plate piles
    ((0, 8), (1, 0)),           # ingredient 0 supplies
    ((3, 0), (4, 8)),           # ingredient 1 supplies
    ((0, 6), (4, 6)),           # ingredient 2 supplies
)

_LEFT_POSITIONS = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                   (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                   (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6))
_RIGHT_POSITIONS = ((1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9))


def _template(row, col):
    return [[[1 if (row + i - 2, col + j - 2) in _OBJECTS[k] else 0
              for k in range(9)] for j in range(5)] for i in range(5)]


def _locate(obs, positions):
    templates = jnp.asarray([_template(r, c) for r, c in positions],
                            dtype=jnp.int32)
    rows = jnp.asarray([p[0] for p in positions], dtype=jnp.int32)
    cols = jnp.asarray([p[1] for p in positions], dtype=jnp.int32)
    visible = obs[:, :, 20:29]
    errors = jnp.sum(jnp.abs(templates - visible[None, :, :, :]), axis=(1, 2, 3))
    index = jnp.argmin(errors)
    return rows[index], cols[index]


def _move_to(row, col, target_row, target_col):
    return jnp.where(
        row < target_row, DOWN,
        jnp.where(row > target_row, UP,
                  jnp.where(col < target_col, RIGHT,
                            jnp.where(col > target_col, LEFT, STAY))))


def _face_and_interact(obs, row, col, target_row, target_col,
                       facing_channel, turn_action):
    at_target = (row == target_row) & (col == target_col)
    facing = obs[2, 2, facing_channel] > 0
    return jnp.where(at_target,
                     jnp.where(facing, INTERACT, turn_action),
                     _move_to(row, col, target_row, target_col))


def init_agent_0():
    # [recipe, steps, prior pot count, stable, working, cooldown,
    #  staging phase, staging id, held last, top stash, bottom stash]
    return jnp.asarray([-1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
                       dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    base = jnp.where(
        reset,
        jnp.asarray([-1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0], dtype=jnp.int32),
        memory)
    row, col = _locate(obs, _LEFT_POSITIONS)

    shown_0 = jnp.sum(obs[:, :, 36]) > 0
    shown_1 = jnp.sum(obs[:, :, 37]) > 0
    recipe = jnp.where(shown_0, 0, jnp.where(shown_1, 1, base[0]))

    held_0 = obs[2, 2, 7] > 0
    held_1 = obs[2, 2, 8] > 0
    held_2 = obs[2, 2, 9] > 0
    holding_ingredient = held_0 | held_1 | held_2
    held_id = jnp.where(held_0, 0, jnp.where(held_1, 1, 2))
    at_home = (row == 2) & (col == 6)
    stash_0 = jnp.where(at_home, obs[0, 0, 31] > 0, base[9] > 0)
    stash_1 = jnp.where(at_home, obs[4, 0, 32] > 0, base[10] > 0)

    pot_0 = jnp.sum(obs[:, :, 22] * obs[:, :, 31])
    pot_1 = jnp.sum(obs[:, :, 22] * obs[:, :, 32])
    pot_2 = jnp.sum(obs[:, :, 22] * obs[:, :, 33])
    pot_count = pot_0 + pot_1 + pot_2
    pot_cooked = jnp.sum(obs[:, :, 22] * obs[:, :, 30]) > 0
    pot_timer = jnp.sum(obs[:, :, 22] * obs[:, :, 39])
    pot_busy = (pot_count >= 3) | pot_cooked | (pot_timer > 0)
    pot_visible = jnp.sum(obs[:, :, 22]) > 0
    count_changed = pot_visible & (pot_count != base[2])
    stable = jnp.where(pot_visible,
                       jnp.where(count_changed, 0, base[3] + 1), base[3])
    pot_just_scooped = pot_visible & (pot_count == 0) & (base[2] >= 3)
    cooldown = jnp.where(pot_just_scooped, 2, jnp.maximum(base[5] - 1, 0))
    # At home, the partner's two ingredient pickup stances appear at local
    # cells (1,4) and (3,4).  Pause once after a pot change, then yield only
    # to direct evidence that the partner is fetching/carrying an ingredient.
    partner_at_supply = ((row == 2) & (col == 6) &
                         ((obs[1, 4, 10] > 0) | (obs[3, 4, 10] > 0)))
    partner_raw = jnp.sum(obs[:, :, 17:20]) > 0
    helper_intent = partner_at_supply | partner_raw
    yield_for_partner = ((pot_count > 0) & (pot_count < 3) &
                         ((stable < 1) | helper_intent))

    go_home = _move_to(row, col, 2, 6)
    put_in_pot = _face_and_interact(obs, row, col, 2, 6, 3, RIGHT)
    # A partial pot is irreversible.  If a partner started it with a
    # non-target ingredient, complete that same type promptly so the bad dish
    # can be delivered/cleared instead of blocking the sole pot forever.
    observed_working = jnp.where(
        pot_0 > 0, 0, jnp.where(pot_1 > 0, 1, jnp.where(pot_2 > 0, 2, recipe)))
    retained_working = jnp.where(base[4] >= 0, base[4], recipe)
    working_id = jnp.where(pot_visible, observed_working, retained_working)

    finished_staging = ((base[6] == 1) & (base[8] > 0) &
                        (~holding_ingredient))
    phase = jnp.where(finished_staging, 2, base[6])
    phase = jnp.where((phase == 2) & at_home, 0, phase)
    pot_matches_recipe = (((recipe == 0) & (pot_0 == pot_count)) |
                          ((recipe == 1) & (pot_1 == pot_count)))
    recipe_stashed = jnp.where(recipe == 0, stash_0, stash_1)
    start_staging = ((phase == 0) & at_home & pot_busy & (pot_count >= 3) &
                     pot_matches_recipe & (~holding_ingredient) &
                     (~recipe_stashed))
    held_stashed = (((held_id == 0) & stash_0) |
                    ((held_id == 1) & stash_1) | (held_id == 2))
    start_overflow = ((phase == 0) & pot_visible & pot_busy &
                      holding_ingredient & (~held_stashed))
    start_any_staging = start_staging | start_overflow
    active_phase = jnp.where(start_any_staging, 1, phase)
    staging_id = jnp.where(start_staging, recipe,
                           jnp.where(start_overflow, held_id, base[7]))

    requested_id = jnp.where(active_phase == 1, staging_id, working_id)
    supply_row = jnp.where(requested_id == 0, 1,
                           jnp.where(requested_id == 1, 3, 1))
    supply_col = jnp.where(requested_id == 2, 6, 1)
    supply_facing = jnp.where(requested_id == 2, 1, 4)
    supply_turn = jnp.where(requested_id == 2, UP, LEFT)
    get_from_supply = _face_and_interact(
        obs, row, col, supply_row, supply_col, supply_facing, supply_turn)
    counter_row = jnp.where(requested_id == 0, 1, 3)
    counter_facing = jnp.where(requested_id == 0, 1, 2)
    counter_turn = jnp.where(requested_id == 0, UP, DOWN)
    use_counter = _face_and_interact(
        obs, row, col, counter_row, 4, counter_facing, counter_turn)
    working_stashed = (((working_id == 0) & stash_0) |
                       ((working_id == 1) & stash_1))
    get_normal = jnp.where(working_stashed, use_counter, get_from_supply)
    staging_action = jnp.where(holding_ingredient, use_counter, get_from_supply)

    action = jnp.where(
        recipe < 0, go_home,
        jnp.where(active_phase == 1, staging_action,
                  jnp.where(active_phase == 2, go_home,
                            jnp.where(cooldown > 0, go_home,
                                      jnp.where(holding_ingredient, put_in_pot,
                                                jnp.where(pot_busy |
                                                          yield_for_partner,
                                                          go_home,
                                                          get_normal))))))
    next_memory = jnp.asarray(
        [recipe, base[1] + 1,
         jnp.where(pot_visible, pot_count, base[2]), stable, working_id,
         cooldown, active_phase, staging_id,
         holding_ingredient.astype(jnp.int32),
         stash_0.astype(jnp.int32), stash_1.astype(jnp.int32)],
        dtype=jnp.int32)
    trace = {
        "position": jnp.asarray([row, col], dtype=jnp.int32),
        "recipe": recipe,
        "pot_count": pot_count.astype(jnp.int32),
        "phase": active_phase,
        "stashes": jnp.asarray([stash_0, stash_1], dtype=jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    # [steps, inferred id, assists, prior count, stable steps, held-raw last,
    #  empty steps, alternating guess, solo mode, cooked last, watch, early]
    return jnp.asarray([0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    base = jnp.where(
        reset,
        jnp.asarray([0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
        memory)
    row, col = _locate(obs, _RIGHT_POSITIONS)

    has_plate = obs[2, 2, 5] > 0
    has_cooked = obs[2, 2, 6] > 0
    held_0 = obs[2, 2, 7] > 0
    held_1 = obs[2, 2, 8] > 0
    holding_ingredient = (held_0 | held_1) & (~has_cooked)
    pot_0 = jnp.sum(obs[:, :, 22] * obs[:, :, 31])
    pot_1 = jnp.sum(obs[:, :, 22] * obs[:, :, 32])
    pot_count = pot_0 + pot_1 + jnp.sum(obs[:, :, 22] * obs[:, :, 33])
    pot_cooked = jnp.sum(obs[:, :, 22] * obs[:, :, 30]) > 0
    pot_timer = jnp.sum(obs[:, :, 22] * obs[:, :, 39])

    delivery_attempt = (base[9] > 0) & (~has_cooked)
    watch = jnp.where(delivery_attempt, 6, jnp.maximum(base[10] - 1, 0))
    partner_inventory_early = jnp.sum(obs[:, :, 15:20])
    cue_window = (watch > 0) & (row == 2) & (col == 8) & (pot_count == 0)
    motion_0 = cue_window & (obs[1, 0, 10] > 0) & (partner_inventory_early == 0)
    motion_1 = cue_window & (obs[3, 0, 10] > 0) & (partner_inventory_early == 0)
    motion_signal = motion_0 | motion_1
    early = jnp.where(delivery_attempt, 0,
                      jnp.where(motion_signal, 1, base[11]))
    guess = jnp.where(delivery_attempt, 1 - base[7], base[7])
    empty_idle = ((pot_count == 0) & (~holding_ingredient) & (~has_plate) &
                  (~has_cooked) & (~pot_cooked) & (pot_timer == 0))
    empty_steps = jnp.where(empty_idle, base[6] + 1, 0)
    start_solo = empty_steps >= 40
    solo = jnp.where(delivery_attempt, 0,
                     jnp.where(start_solo, 1, base[8]))
    inferred_from_pot = jnp.where(
        pot_0 > 0, 0, jnp.where(pot_1 > 0, 1, base[1]))
    inferred = jnp.where(
        motion_0, 0,
        jnp.where(motion_1, 1,
                  jnp.where((solo > 0) & (pot_count == 0), guess,
                            inferred_from_pot)))
    new_cycle = (pot_count == 0) & (base[3] >= 3)
    newly_holding = holding_ingredient & (base[5] == 0)
    assisted = jnp.where(
        new_cycle, 0,
        jnp.where(newly_holding, jnp.minimum(base[2] + 1, 2), base[2]))
    count_changed = pot_count != base[3]
    stable = jnp.where(count_changed, 0, base[4] + 1)
    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_inventory = jnp.sum(obs[:, :, 15:20])
    partner_waiting_empty = partner_visible & (partner_inventory == 0)
    partner_cooked = jnp.sum(obs[:, :, 16]) > 0
    partner_raw_seen = (jnp.sum(obs[:, :, 17:20]) > 0) & (~partner_cooked)

    get_plate = _face_and_interact(obs, row, col, 1, 9, 3, RIGHT)
    use_pot = _face_and_interact(obs, row, col, 2, 8, 4, LEFT)
    deliver = _face_and_interact(obs, row, col, 2, 9, 3, RIGHT)
    wait_at_pot = _move_to(row, col, 2, 8)
    supply_row = jnp.where(inferred == 0, 1, 3)
    supply_facing = jnp.where(inferred == 0, 1, 2)
    supply_turn = jnp.where(inferred == 0, UP, DOWN)
    get_ingredient = _face_and_interact(
        obs, row, col, supply_row, 8, supply_facing, supply_turn)
    discard_ingredient = _face_and_interact(obs, row, col, 1, 9, 1, UP)
    put_or_discard = jnp.where(
        (pot_count < 3) & (pot_timer == 0) & (~pot_cooked),
        use_pot, discard_ingredient)
    first_assist = assisted == 0
    stalled_out_of_view = (stable >= 12) & (~partner_raw_seen)
    second_assist = ((assisted == 1) & (pot_count == 2) &
                     (partner_waiting_empty | stalled_out_of_view))
    should_assist = ((inferred >= 0) & (pot_count > 0) & (pot_count < 3) &
                     (first_assist | second_assist))
    should_solo_fill = ((solo > 0) & (pot_count < 3) &
                        (~pot_cooked) & (pot_timer == 0))
    should_early_fill = ((early > 0) & (assisted < 2) & (pot_count < 3) &
                         (~pot_cooked) & (pot_timer == 0))

    action = jnp.where(
        has_cooked, deliver,
        jnp.where(has_plate, use_pot,
                  jnp.where(holding_ingredient, put_or_discard,
                            jnp.where(pot_cooked | (pot_timer > 0), get_plate,
                                      jnp.where(should_assist | should_solo_fill |
                                                should_early_fill,
                                                get_ingredient,
                                                wait_at_pot)))))
    next_memory = jnp.asarray(
        [base[0] + 1, inferred, assisted, pot_count, stable,
         holding_ingredient.astype(jnp.int32), empty_steps, guess, solo,
         has_cooked.astype(jnp.int32), watch, early], dtype=jnp.int32)
    trace = {
        "position": jnp.asarray([row, col], dtype=jnp.int32),
        "pot_timer": pot_timer.astype(jnp.int32),
        "pot_count": pot_count.astype(jnp.int32),
        "inferred": inferred,
        "assisted": assisted,
        "solo": solo,
        "early": early,
        "has_plate": has_plate,
        "has_cooked": has_cooked,
    }
    return action.astype(jnp.int32), next_memory, trace
