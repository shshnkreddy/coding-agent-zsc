"""Convention-light policy for the grounded coordination ring.

The outer player reads the free recipe display and puts requested ingredients on
the upper pass counter. The inner player accepts visible ingredient evidence,
but can activate a recipe button and run the complete cooking loop on its own.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_reactive_cooks", "adapter": "overcooked_v2",
    "adapter_version": 1, "max_height": 9, "max_width": 9,
    "num_ingredients": 3,
}

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

_GRID = (
    "WWW2R2WWW", "W       W", "W WWLWW W", "2 0   B 2",
    "R X P X R", "2 1   B 2", "W WWLWW W", "W       W",
    "WWW2R2WWW",
)
_OUTER_CELLS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 1), (2, 7), (3, 1), (3, 7), (4, 1), (4, 7),
    (5, 1), (5, 7), (6, 1), (6, 7), (7, 1), (7, 2), (7, 3),
    (7, 4), (7, 5), (7, 6), (7, 7),
)
_INNER_CELLS = (
    (3, 3), (3, 4), (3, 5), (4, 3),
    (4, 5), (5, 3), (5, 4), (5, 5),
)


def _feature(ch):
    vals = [0] * 9
    if ch == "W": vals[0] = 1
    elif ch == "X": vals[1] = 1
    elif ch == "P": vals[2] = 1
    elif ch == "R": vals[3] = 1
    elif ch == "L": vals[4] = 1
    elif ch == "B": vals[5] = 1
    elif ch == "0": vals[6] = 1
    elif ch == "1": vals[7] = 1
    elif ch == "2": vals[8] = 1
    return vals


def _templates(cells):
    result = []
    for row, col in cells:
        crop = []
        for di in range(-2, 3):
            line = []
            for dj in range(-2, 3):
                rr, cc = row + di, col + dj
                line.append(_feature(_GRID[rr][cc]) if 0 <= rr < 9 and 0 <= cc < 9 else [0] * 9)
            crop.append(line)
        result.append(crop)
    return jnp.asarray(result, dtype=jnp.int32)


def _next_table(cells):
    table = [[[[STAY for _ in range(9)] for _ in range(9)] for _ in range(9)] for _ in range(9)]
    allowed = set(cells)
    moves = ((0, 1, RIGHT), (1, 0, DOWN), (0, -1, LEFT), (-1, 0, UP))
    for source in cells:
        queue = [source]
        first = {source: STAY}
        for cur in queue:
            for dr, dc, action in moves:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in allowed and nxt not in first:
                    first[nxt] = action if cur == source else first[cur]
                    queue.append(nxt)
        for dest in cells:
            table[source[0]][source[1]][dest[0]][dest[1]] = first[dest]
    return jnp.asarray(table, dtype=jnp.int32)


def _locate(obs, positions, templates):
    static = obs[:, :, 20:29]
    errors = jnp.sum(jnp.abs(templates - static[None]), axis=(1, 2, 3))
    return positions[jnp.argmin(errors)]


def _navigate(pos, target, table):
    return table[pos[0], pos[1], target[0], target[1]]


def _face_and_interact(obs, pos, stand, direction, table):
    move_for_direction = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)
    at_stand = jnp.all(pos == stand)
    facing = jnp.argmax(obs[2, 2, 1:5])
    terminal = jnp.where(facing == direction, INTERACT, move_for_direction[direction])
    return jnp.where(at_stand, terminal, _navigate(pos, stand, table))


def _global_value(obs, pos, row, col, channel):
    lr, lc = row - pos[0] + 2, col - pos[1] + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    return obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4), channel] * visible.astype(jnp.int32)


def _shown_recipe(obs):
    counts = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    return jnp.sum(counts) > 0, jnp.argmax(counts).astype(jnp.int32)


def init_agent_0():
    # step, phase (0 read, 1 fetch, 2 verify, 3 pass, 4 discard), target,
    # pass wait, rotating discard-counter index
    return jnp.asarray([0, 0, -1, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, init_agent_0(), memory)
    step, phase, target, wait, discard_index = mem
    positions = jnp.asarray(_OUTER_CELLS, dtype=jnp.int32)
    templates = _templates(_OUTER_CELLS)
    next_table = _next_table(_OUTER_CELLS)
    pos = _locate(obs, positions, templates)
    shown, shown_target = _shown_recipe(obs)
    target = jnp.where(shown, shown_target, target)

    raw_counts = obs[2, 2, 7:10]
    has_raw = jnp.sum(raw_counts) > 0
    held_kind = jnp.argmax(raw_counts).astype(jnp.int32)
    read_stand = jnp.asarray([4, 1], dtype=jnp.int32)
    at_read = jnp.all(pos == read_stand)

    pick0 = _face_and_interact(obs, pos, jnp.asarray([3, 1]), 2, next_table)
    pick1 = _face_and_interact(obs, pos, jnp.asarray([5, 1]), 2, next_table)
    pick_action = jnp.where(target == 0, pick0, pick1)
    read_action = _navigate(pos, read_stand, next_table)
    verify_action = _navigate(pos, read_stand, next_table)
    verified = at_read & shown & has_raw & (held_kind == shown_target)
    mismatched = at_read & shown & has_raw & (held_kind != shown_target)

    pass_stand = jnp.where(target == 0, jnp.asarray([1, 3]),
                           jnp.asarray([7, 3]))
    upper_pass_action = _face_and_interact(
        obs, pos, jnp.asarray([1, 3]), 1, next_table)
    lower_pass_action = _face_and_interact(
        obs, pos, jnp.asarray([7, 3]), 0, next_table)
    pass_action = jnp.where(target == 0, upper_pass_action,
                            lower_pass_action)
    upper_item_count = sum(_global_value(obs, pos, 2, 3, ch) for ch in range(29, 34))
    lower_item_count = sum(_global_value(obs, pos, 6, 3, ch) for ch in range(29, 34))
    pass_item_count = jnp.where(target == 0, upper_item_count,
                                lower_item_count)
    pass_free = pass_item_count == 0
    at_pass = jnp.all(pos == pass_stand)
    pass_action = jnp.where(at_pass & (~pass_free), STAY, pass_action)
    discard_stands = jnp.asarray([
        [2, 1], [1, 1], [1, 2], [1, 5], [1, 6], [1, 7],
        [6, 1], [7, 1], [7, 2], [7, 5], [7, 6], [7, 7],
    ], dtype=jnp.int32)
    discard_directions = jnp.asarray(
        [2, 0, 0, 1, 0, 0, 2, 1, 1, 0, 1, 1], dtype=jnp.int32)
    discard_stand = discard_stands[discard_index]
    discard_direction = discard_directions[discard_index]
    discard_action = _face_and_interact(
        obs, pos, discard_stand, discard_direction, next_table)
    at_discard = jnp.all(pos == discard_stand)

    action = jnp.where(phase == 0, read_action,
             jnp.where(phase == 1, pick_action,
             jnp.where(phase == 2, verify_action,
             jnp.where(phase == 3, pass_action, discard_action))))
    action = jnp.where((phase == 0) & shown, pick_action, action)
    action = jnp.where((phase == 2) & verified, pass_action, action)
    action = jnp.where(mismatched, discard_action, action)

    next_phase = phase
    next_phase = jnp.where((phase == 0) & shown, 1, next_phase)
    picked_now = (phase == 1) & (~has_raw) & (action == INTERACT)
    next_phase = jnp.where(picked_now, 2, next_phase)
    next_phase = jnp.where((phase == 2) & verified, 3, next_phase)
    next_phase = jnp.where(mismatched, 4, next_phase)
    discarded_now = (phase == 4) & at_discard & (action == INTERACT)
    next_phase = jnp.where(discarded_now, 0, next_phase)
    passed_now = (phase == 3) & at_pass & pass_free & (action == INTERACT)
    next_phase = jnp.where(passed_now, 0, next_phase)
    next_target = jnp.where(passed_now | discarded_now, -1, target)
    next_wait = jnp.where(at_pass & (~pass_free), wait + 1, 0)
    next_discard = jnp.where(discarded_now, (discard_index + 1) % 12,
                             discard_index)
    next_mem = jnp.asarray(
        [step + 1, next_phase, next_target, next_wait, next_discard],
        dtype=jnp.int32)
    trace = {"pos": pos, "phase": next_phase, "target": next_target}
    return action.astype(jnp.int32), next_mem, trace


def init_agent_1():
    # step, recipe, unsupported wait, require refresh, initial button side
    return jnp.asarray([0, -1, 0, 0, -1], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    mem = jnp.where(reset, init_agent_1(), memory)
    step, target, unknown_wait, must_refresh, button_side = mem
    positions = jnp.asarray(_INNER_CELLS, dtype=jnp.int32)
    templates = _templates(_INNER_CELLS)
    next_table = _next_table(_INNER_CELLS)
    pos = _locate(obs, positions, templates)

    shown, shown_target = _shown_recipe(obs)
    partner_raw = jnp.sum(obs[:, :, 17:20], axis=(0, 1))
    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_signal = jnp.sum(partner_raw) > 0
    partner_kind = jnp.argmax(partner_raw).astype(jnp.int32)
    partner_valid = partner_signal & (partner_kind < 2)
    intent0 = ((step > 0) &
               (_global_value(obs, pos, 3, 1, 10) > 0) &
               (_global_value(obs, pos, 3, 1, 13) > 0))
    intent1 = ((step > 0) &
               (_global_value(obs, pos, 5, 1, 10) > 0) &
               (_global_value(obs, pos, 5, 1, 13) > 0))
    intent_signal = intent0 | intent1
    intent_kind = jnp.where(intent0, 0, 1)
    upper_pass_raw = jnp.asarray([
        _global_value(obs, pos, 2, 3, 31),
        _global_value(obs, pos, 2, 3, 32),
        _global_value(obs, pos, 2, 3, 33),
    ])
    lower_pass_raw = jnp.asarray([
        _global_value(obs, pos, 6, 3, 31),
        _global_value(obs, pos, 6, 3, 32),
        _global_value(obs, pos, 6, 3, 33),
    ])
    upper_signal = jnp.sum(upper_pass_raw) > 0
    lower_signal = jnp.sum(lower_pass_raw) > 0
    pass_signal = upper_signal | lower_signal
    upper_kind = jnp.argmax(upper_pass_raw).astype(jnp.int32)
    lower_kind = jnp.argmax(lower_pass_raw).astype(jnp.int32)
    upper_valid = upper_signal & (upper_kind < 2)
    lower_valid = lower_signal & (lower_kind < 2)
    pass_kind = jnp.where(upper_valid, upper_kind, lower_kind)
    pass_valid = upper_valid | lower_valid
    inferred = jnp.where(partner_valid, partner_kind,
                         jnp.where(intent_signal, intent_kind, pass_kind))
    evidence = partner_valid | intent_signal | pass_valid
    target = jnp.where(shown, shown_target, target)
    target = jnp.where((target < 0) & evidence & (must_refresh == 0), inferred, target)
    must_refresh = jnp.where(shown, 0, must_refresh)
    known = target >= 0

    inv = obs[2, 2, 5:10]
    has_plate, has_dish = inv[0] > 0, inv[1] > 0
    raw = inv[2:5]
    has_raw = jnp.sum(raw) > 0
    pot_count = sum(jnp.sum(obs[:, :, ch] * obs[:, :, 22]) for ch in range(31, 34))
    pot_cooked = jnp.sum(obs[:, :, 30] * obs[:, :, 22]) > 0
    pot_timer = jnp.sum(obs[:, :, 39] * obs[:, :, 22])

    wait_stand = jnp.asarray([4, 3], dtype=jnp.int32)
    wait_action = _navigate(pos, wait_stand, next_table)
    button_action = _face_and_interact(obs, pos, jnp.asarray([3, 4]), 0, next_table)
    lower_button_action = _face_and_interact(
        obs, pos, jnp.asarray([5, 4]), 1, next_table)
    random_side = jnp.bitwise_and(rng[0], jnp.asarray(1, dtype=rng.dtype)).astype(jnp.int32)
    start_early_button = ((unknown_wait >= 8) & (~partner_visible) &
                          (button_side < 0) & (must_refresh == 0))
    active_side = jnp.where(start_early_button, random_side, button_side)
    early_button_action = jnp.where(active_side == 0, button_action,
                                    lower_button_action)
    unknown_action = jnp.where(
        step > 353, STAY,
        jnp.where(active_side >= 0, early_button_action,
        jnp.where((unknown_wait >= 18) | (must_refresh > 0),
                  button_action, wait_action)))

    pot_top = _face_and_interact(obs, pos, jnp.asarray([3, 4]), 1, next_table)
    pot_bottom = _face_and_interact(obs, pos, jnp.asarray([5, 4]), 0, next_table)
    raw_kind = jnp.argmax(raw).astype(jnp.int32)
    raw_action = jnp.where(raw_kind == 0, pot_top, pot_bottom)

    plate_top = _face_and_interact(obs, pos, jnp.asarray([3, 5]), 2, next_table)
    plate_bottom = _face_and_interact(obs, pos, jnp.asarray([5, 5]), 2, next_table)
    get_plate = jnp.where(target == 0, plate_top, plate_bottom)
    plate_wait = _face_and_interact(obs, pos, jnp.asarray([4, 5]), 3, next_table)
    # Interacting early is harmless; it also keeps the carrier facing the pot
    # so the first interaction after cooking completes collects immediately.
    plate_action = plate_wait
    deliver_action = _face_and_interact(obs, pos, jnp.asarray([4, 5]), 2, next_table)

    matching_pass = pass_signal & (pass_kind == target)
    take_upper_pass = _face_and_interact(
        obs, pos, jnp.asarray([3, 3]), 0, next_table)
    take_lower_pass = _face_and_interact(
        obs, pos, jnp.asarray([5, 3]), 1, next_table)
    take_pass = jnp.where(upper_signal, take_upper_pass, take_lower_pass)
    take0 = _face_and_interact(obs, pos, jnp.asarray([3, 3]), 3, next_table)
    take1 = _face_and_interact(obs, pos, jnp.asarray([5, 3]), 3, next_table)
    take_supply = jnp.where(target == 0, take0, take1)
    need_raw = (pot_count < 3) & (~pot_cooked) & (pot_timer == 0)
    side_button_action = jnp.where(target == 0, button_action,
                                   lower_button_action)
    button_stand = jnp.where(target == 0, jnp.asarray([3, 4]),
                             jnp.asarray([5, 4]))
    button_direction = jnp.where(target == 0, 0, 1)
    self_facing = jnp.argmax(obs[2, 2, 1:5])
    face_move = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)[button_direction]
    at_button_stand = jnp.all(pos == button_stand)
    stage_for_refresh = jnp.where(
        at_button_stand,
        jnp.where(self_facing == button_direction, STAY, face_move),
        _navigate(pos, button_stand, next_table))
    refresh_or_plate = jnp.where((pot_timer == 5) | (~shown),
                                 side_button_action, get_plate)
    cooking_action = jnp.where(
        step > 353, get_plate,
        jnp.where(pot_timer > 5, stage_for_refresh, refresh_or_plate))
    empty_action = jnp.where(need_raw, take_supply,
                    jnp.where(pot_timer > 0, cooking_action, get_plate))
    known_action = jnp.where(has_dish, deliver_action,
                   jnp.where(has_raw, raw_action,
                   jnp.where(has_plate, plate_action, empty_action)))
    action = jnp.where(known, known_action, unknown_action)

    delivered_now = known & has_dish & (action == INTERACT)
    next_target = jnp.where(delivered_now, -1, target)
    next_unknown = jnp.where(next_target >= 0, 0, unknown_wait + 1)
    next_refresh = jnp.where(delivered_now, 1, must_refresh)
    next_side = jnp.where(start_early_button, random_side, button_side)
    next_side = jnp.where(next_target >= 0, -1, next_side)
    next_mem = jnp.asarray(
        [step + 1, next_target, next_unknown, next_refresh, next_side],
        dtype=jnp.int32)
    trace = {"pos": pos, "target": next_target,
             "pot_count": pot_count.astype(jnp.int32)}
    return action.astype(jnp.int32), next_mem, trace
