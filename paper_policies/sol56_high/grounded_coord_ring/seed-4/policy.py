"""Cross-play-oriented policy for the grounded coordination ring.

The inner player is a self-sufficient cook. The outer player continually reads
a free recipe display and places matching ingredients on a shared counter.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "reactive_ring_cook_v8",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}


_FACE_ACTION_RAW = (3, 1, 0, 2)


# Inner-room floor cells, with the central pot omitted.
_INNER_INDEX_RAW = (
    (-1, -1, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1,  0,  1,  2, -1, -1, -1),
    (-1, -1, -1,  3, -1,  4, -1, -1, -1),
    (-1, -1, -1,  5,  6,  7, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1),
)

# First action on a shortest path. Rows and columns use inner index order
# (3,3),(3,4),(3,5),(4,3),(4,5),(5,3),(5,4),(5,5).
_INNER_NAV_RAW = (
    (4, 2, 2, 3, 3, 3, 2, 2),
    (0, 4, 2, 3, 3, 3, 0, 3),
    (0, 0, 4, 3, 3, 0, 0, 3),
    (1, 2, 2, 4, 1, 3, 2, 2),
    (0, 0, 1, 1, 4, 0, 0, 3),
    (1, 2, 1, 1, 1, 4, 2, 2),
    (1, 0, 1, 1, 1, 0, 4, 2),
    (0, 0, 1, 1, 1, 0, 0, 4),
)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _inner_location(obs):
    pot_flat = jnp.argmax(obs[:, :, 22])
    pot_r = pot_flat // 5
    pot_c = pot_flat % 5
    self_r = 6 - pot_r
    self_c = 6 - pot_c
    inner_index = jnp.asarray(_INNER_INDEX_RAW, dtype=jnp.int32)
    return self_r, self_c, inner_index[self_r, self_c]


def _inner_use(obs, pos_index, target_index, target_facing):
    face_action = jnp.asarray(_FACE_ACTION_RAW, dtype=jnp.int32)
    inner_nav = jnp.asarray(_INNER_NAV_RAW, dtype=jnp.int32)
    at_target = pos_index == target_index
    orient_or_use = jnp.where(
        _facing(obs) == target_facing,
        jnp.asarray(5, dtype=jnp.int32),
        face_action[target_facing],
    )
    return jnp.where(at_target, orient_or_use,
                     inner_nav[target_index, pos_index])


def _counter_ingredient(obs, self_r, self_c, counter_r, counter_c,
                        ingredient_channel):
    local_r = counter_r - self_r + 2
    local_c = counter_c - self_c + 2
    visible = ((local_r >= 0) & (local_r < 5) &
               (local_c >= 0) & (local_c < 5))
    value = obs[jnp.clip(local_r, 0, 4), jnp.clip(local_c, 0, 4),
                ingredient_channel]
    return jnp.where(visible, value, 0)


def init_agent_1():
    # desired ingredient (-1 unknown), elapsed steps, delivery intents,
    # recipe-query-in-progress flag, late-cook button-preactivated flag
    return jnp.asarray([-1, 0, 0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    desired = memory[0]

    shown_0 = jnp.sum(obs[:, :, 36])
    shown_1 = jnp.sum(obs[:, :, 37])
    recipe_visible = (shown_0 + shown_1) > 0
    desired = jnp.where(recipe_visible,
                        jnp.where(shown_1 > shown_0, 1, 0), desired)
    self_r, self_c, pos_index = _inner_location(obs)
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:10]) > 0
    hands_empty = ~(held_plate | held_cooked | held_raw)

    pot_cooked = jnp.sum(obs[:, :, 30]) > 0
    pot_timer = jnp.sum(obs[:, :, 39])

    supply_target = jnp.where(desired == 1, 5, 0)
    acquire_target = supply_target
    acquire_face = jnp.asarray(3, dtype=jnp.int32)

    raw_query_action = _inner_use(obs, pos_index, 1, 0)
    cycle_can_start = (memory[1] <= 361) & (memory[2] < 10)
    query_committed = (memory[3] > 0) | cycle_can_start
    query_action = jnp.where(query_committed, raw_query_action, 4)
    # A raw item acquired on the north/south edge is loaded from that same
    # side of the pot. The repeated supply loop is five actions rather than
    # six through the west pot interaction cell.
    pot_ingredient_target = jnp.where(
        self_r == 3, 1,
        jnp.where(self_r == 5, 6, jnp.where(desired == 1, 6, 1)),
    )
    pot_ingredient_facing = jnp.where(pot_ingredient_target == 1, 1, 0)
    ingredient_action = _inner_use(obs, pos_index,
                                   pot_ingredient_target,
                                   pot_ingredient_facing)
    plate_target = jnp.where(desired == 1, 7, 2)
    plate_action = _inner_use(obs, pos_index, plate_target, 2)
    plate_pot_action = _inner_use(obs, pos_index, 4, 3)
    deliver_action = _inner_use(obs, pos_index, 4, 2)
    acquire_action = _inner_use(obs, pos_index,
                                acquire_target, acquire_face)

    # Use otherwise idle cooking time to leave the button active across the
    # delivery transition. At timer 6 the plate-and-deliver route consumes at
    # most nine ticks, so the resampled recipe remains visible afterward.
    should_prepare = (memory[4] == 0) & (memory[2] < 9)
    preparation_target = jnp.where(desired == 1, 6, 1)
    preparation_facing = jnp.where(desired == 1, 1, 0)
    at_preparation_pose = ((pos_index == preparation_target) &
                           (_facing(obs) == preparation_facing))
    preparation_use = _inner_use(obs, pos_index, preparation_target,
                                 preparation_facing)
    preparation_action = jnp.where(
        at_preparation_pose & (pot_timer > 6), 4, preparation_use)
    cooking_action = jnp.where(should_prepare,
                               preparation_action, plate_action)

    empty_action = jnp.where(pot_cooked, plate_action,
                             jnp.where(pot_timer > 0, cooking_action,
                                       acquire_action))
    known_action = jnp.where(
        held_cooked,
        deliver_action,
        jnp.where(held_plate, plate_pot_action,
                  jnp.where(held_raw, ingredient_action, empty_action)),
    )
    action = jnp.where(desired < 0, query_action, known_action)

    delivery_intent = (held_cooked & (pos_index == 4) &
                       (_facing(obs) == 2))
    preparation_intent = (hands_empty & should_prepare & (pot_timer > 0) &
                          (pot_timer <= 6) & at_preparation_pose)
    next_desired = jnp.where(delivery_intent, -1, desired)
    next_query_flag = jnp.where(
        recipe_visible | delivery_intent,
        0,
        jnp.where((desired < 0) & query_committed, 1, 0),
    )
    next_prepared = jnp.where(
        delivery_intent,
        0,
        jnp.where(preparation_intent, 1, memory[4]),
    )
    next_memory = jnp.asarray([
        next_desired,
        memory[1] + 1,
        memory[2] + delivery_intent.astype(jnp.int32),
        next_query_flag,
        next_prepared,
    ], dtype=jnp.int32)
    trace = {
        "desired": desired,
        "position": pos_index,
        "pot_timer": pot_timer.astype(jnp.int32),
        "deliveries": next_memory[2],
    }
    return action.astype(jnp.int32), next_memory, trace


# Outer-ring positions form a cycle. Static templates identify the randomized
# spawn from observation channels 20:29.
_GRID_CODES_RAW = (
    (1, 1, 1, 9, 4, 9, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 1, 1, 5, 1, 1, 0, 1),
    (9, 0, 7, 0, 0, 0, 6, 0, 9),
    (4, 0, 2, 0, 3, 0, 2, 0, 4),
    (9, 0, 8, 0, 0, 0, 6, 0, 9),
    (1, 0, 1, 1, 5, 1, 1, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 1, 1, 9, 4, 9, 1, 1, 1),
)
_OUTER_POS_RAW = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
    (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
    (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
)


def _outer_location(obs):
    grid = jnp.asarray(_GRID_CODES_RAW, dtype=jnp.int32)
    positions = jnp.asarray(_OUTER_POS_RAW, dtype=jnp.int32)
    offsets = jnp.arange(-2, 3, dtype=jnp.int32)
    rows = positions[:, 0, None, None] + offsets[None, :, None]
    cols = positions[:, 1, None, None] + offsets[None, None, :]
    visible = ((rows >= 0) & (rows < 9) & (cols >= 0) & (cols < 9))
    templates = jnp.where(
        visible,
        grid[jnp.clip(rows, 0, 8), jnp.clip(cols, 0, 8)],
        0,
    )
    weights = jnp.arange(1, 10, dtype=jnp.int32)
    observed = jnp.sum(obs[:, :, 20:29] * weights[None, None, :], axis=2)
    difference = jnp.sum(jnp.abs(templates - observed[None, :, :]),
                         axis=(1, 2))
    return jnp.argmin(difference).astype(jnp.int32)


def _outer_move(index, target):
    positions = jnp.asarray(_OUTER_POS_RAW, dtype=jnp.int32)
    clockwise_steps = (target - index) % 24
    counter_steps = (index - target) % 24
    here = positions[index]
    next_pos = positions[(index + 1) % 24]
    prior_pos = positions[(index - 1) % 24]
    next_delta = next_pos - here
    prior_delta = prior_pos - here
    cw_action = jnp.where(next_delta[1] == 1, 0,
                          jnp.where(next_delta[0] == 1, 1,
                                    jnp.where(next_delta[1] == -1, 2, 3)))
    ccw_action = jnp.where(prior_delta[1] == 1, 0,
                           jnp.where(prior_delta[0] == 1, 1,
                                     jnp.where(prior_delta[1] == -1, 2, 3)))
    move = jnp.where(clockwise_steps <= counter_steps,
                     cw_action, ccw_action)
    return jnp.where(index == target, 4, move)


def _outer_use(obs, index, target, target_facing):
    face_action = jnp.asarray(_FACE_ACTION_RAW, dtype=jnp.int32)
    orient_or_use = jnp.where(_facing(obs) == target_facing,
                              jnp.asarray(5, dtype=jnp.int32),
                              face_action[target_facing])
    return jnp.where(index == target, orient_or_use,
                     _outer_move(index, target))


def init_agent_0():
    # desired ingredient, elapsed steps, number of observed recipe changes
    # and occupied-quarantine skip offset
    return jnp.asarray([-1, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    old_desired = memory[0]
    shown_0 = jnp.sum(obs[:, :, 36])
    shown_1 = jnp.sum(obs[:, :, 37])
    visible = (shown_0 + shown_1) > 0
    shown_choice = jnp.where(shown_1 > shown_0, 1, 0)
    changed = visible & (old_desired >= 0) & (shown_choice != old_desired)
    change_count = memory[2] + changed.astype(jnp.int32)
    quarantine_skip = jnp.where(changed, 0, memory[3])
    desired = old_desired
    desired = jnp.where(visible,
                        shown_choice, desired)

    index = _outer_location(obs)
    held_0 = obs[2, 2, 7] > 0
    held_1 = obs[2, 2, 8] > 0
    held_relevant = held_0 | held_1
    held_kind = jnp.where(held_1, 1, 0)
    mismatched = held_relevant & (held_kind != desired) & (desired >= 0)

    # Ring indices: top hub (1,3)=2, bottom hub (7,3)=16, ingredient access
    # at (3,1)=22 and (5,1)=20.  Each ingredient uses its nearest hub.
    pickup_target = jnp.where(desired == 1, 20, 22)
    north_distance = jnp.minimum((2 - index) % 24, (index - 2) % 24)
    south_distance = jnp.minimum((16 - index) % 24, (index - 16) % 24)
    unknown_target = jnp.where(south_distance < north_distance, 16, 2)
    unknown_action = _outer_move(index, unknown_target)
    pickup_action = _outer_use(obs, index, pickup_target,
                               jnp.asarray(2, dtype=jnp.int32))
    stage_target = jnp.where(desired == 1, 16, 2)
    stage_facing = jnp.where(desired == 1, 0, 1)
    stage_action = _outer_use(obs, index, stage_target, stage_facing)

    # Four north and four south perimeter walls are unreachable from the inner
    # room. Keep each stale ingredient near its source side. If a chosen slot
    # is unexpectedly occupied, advance the private skip offset immediately.
    north_dumps = jnp.asarray((0, 1, 5, 6), dtype=jnp.int32)
    south_dumps = jnp.asarray((18, 17, 13, 12), dtype=jnp.int32)
    base_dump_slot = ((change_count - 1) // 2 + quarantine_skip) % 4
    dump_target = jnp.where(held_kind == 1,
                            south_dumps[base_dump_slot],
                            north_dumps[base_dump_slot])
    dump_facing = jnp.where(held_kind == 1, 1, 0)
    front_row = jnp.where(dump_facing == 0, 1, 3)
    front_occupied = jnp.sum(obs[front_row, 2, 29:34]) > 0
    dump_blocked = mismatched & (index == dump_target) & front_occupied
    quarantine_skip = quarantine_skip + dump_blocked.astype(jnp.int32)
    effective_slot = ((change_count - 1) // 2 + quarantine_skip) % 4
    dump_target = jnp.where(held_kind == 1,
                            south_dumps[effective_slot],
                            north_dumps[effective_slot])
    dump_action = _outer_use(obs, index, dump_target, dump_facing)
    known_action = jnp.where(
        held_relevant,
        jnp.where(mismatched, dump_action, stage_action),
        pickup_action,
    )
    action = jnp.where(desired < 0, unknown_action, known_action)
    next_memory = jnp.asarray([desired, memory[1] + 1, change_count,
                               quarantine_skip],
                              dtype=jnp.int32)
    trace = {
        "desired": desired,
        "position": index,
        "held_kind": jnp.where(held_relevant, held_kind, -1),
    }
    return action.astype(jnp.int32), next_memory, trace
