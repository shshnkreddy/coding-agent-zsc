"""Cross-play-oriented policy for the two-room wide kitchen."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {"name": "wide_natural_signal", "adapter": "overcooked_v2",
                   "adapter_version": 1, "max_height": 7, "max_width": 6,
                   "num_ingredients": 4}


def _locate_upper(obs):
    # Delivery X at global (0, 2) is visible from every upper spawn cell.
    index = jnp.argmax(obs[:, :, 23])
    local_r, local_c = index // 5, index % 5
    return jnp.asarray([2 - local_r, 4 - local_c], dtype=jnp.int32)


def _locate_lower(obs):
    # Recipe R at global (6, 2) is visible from every lower spawn cell.
    index = jnp.argmax(obs[:, :, 25])
    local_r, local_c = index // 5, index % 5
    return jnp.asarray([8 - local_r, 4 - local_c], dtype=jnp.int32)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _navigate(pos, facing, target_r, target_c, object_action):
    """Move row-first to an access square, face its object, then interact."""
    r, c = pos[0], pos[1]
    vertical = jnp.where(r < target_r, 1, 3)
    horizontal = jnp.where(c < target_c, 0, 2)
    # Movement actions are R,D,L,U = 0,1,2,3 while observed facings are
    # U,D,R,L = 0,1,2,3.
    desired_facing = jnp.where(object_action == 0, 2,
                     jnp.where(object_action == 1, 1,
                     jnp.where(object_action == 2, 3, 0)))
    return jnp.where(r != target_r, vertical,
                     jnp.where(c != target_c, horizontal,
                               jnp.where(facing == desired_facing, 5, object_action))).astype(jnp.int32)


def _grid_value(obs, self_pos, global_r, global_c, channel):
    lr = jnp.clip(global_r - self_pos[0] + 2, 0, 4)
    lc = jnp.clip(global_c - self_pos[1] + 2, 0, 4)
    visible = ((jnp.abs(global_r - self_pos[0]) <= 2) &
               (jnp.abs(global_c - self_pos[1]) <= 2))
    return jnp.where(visible, obs[lr, lc, channel], 0)


def init_agent_0():
    # target, valid, elapsed, signal candidate/run, fill (2=none), signal col
    return jnp.asarray([0, 0, 0, 0, 0, 2, -1], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_0(), memory)
    pos = _locate_upper(obs)
    facing = _facing(obs)

    flat_partner = jnp.argmax(obs[:, :, 11])
    pr_local, pc_local = flat_partner // 5, flat_partner % 5
    partner_visible = jnp.max(obs[:, :, 11]) > 0
    partner_r = pos[0] + pr_local - 2
    partner_c = pos[1] + pc_local - 2
    partner_right = jnp.max(obs[:, :, 14]) > 0
    partner_left = jnp.max(obs[:, :, 15]) > 0
    horizontal = partner_left | partner_right
    signal0 = (partner_visible & (partner_r == 4) &
               (partner_left | ((~horizontal) & (partner_c <= 2))))
    signal1 = (partner_visible & (partner_r == 4) &
               (partner_right | ((~horizontal) & (partner_c >= 3))))
    got_signal = signal0 | signal1
    signal_value = jnp.where(signal1, 1, 0)
    # Require an unchanged physical stance, rather than merely two positions on
    # the same side, so spawn-to-target transit cannot masquerade as a signal.
    same_candidate = (got_signal & (signal_value == memory[3]) &
                      (partner_c == memory[6]))
    signal_run = jnp.where(got_signal,
                           jnp.where(same_candidate, memory[4] + 1, 1), 0)
    candidate = jnp.where(got_signal, signal_value, memory[3])
    directional = got_signal & horizontal & (memory[2] > 0)
    committed = (signal_run >= 2) | directional
    target = jnp.where(committed, candidate, memory[0])
    valid = jnp.where(committed, 1, memory[1])

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:11]) > 0
    raw_type = jnp.argmax(obs[2, 2, 7:11])
    pot_c = jnp.where(target == 0, 1, 3)
    count0 = (_grid_value(obs, pos, 3, 1, 34) +
              _grid_value(obs, pos, 3, 1, 35) +
              _grid_value(obs, pos, 3, 1, 36) +
              _grid_value(obs, pos, 3, 1, 37))
    count1 = (_grid_value(obs, pos, 3, 3, 34) +
              _grid_value(obs, pos, 3, 3, 35) +
              _grid_value(obs, pos, 3, 3, 36) +
              _grid_value(obs, pos, 3, 3, 37))
    cooked0 = _grid_value(obs, pos, 3, 1, 33) > 0
    cooked1 = _grid_value(obs, pos, 3, 3, 33) > 0
    timer0 = _grid_value(obs, pos, 3, 1, 44)
    timer1 = _grid_value(obs, pos, 3, 3, 44)
    pot_count = jnp.where(target == 0, count0, count1)
    other_count = jnp.where(target == 0, count1, count0)
    pot_cooked = jnp.where(target == 0, cooked0, cooked1)
    pot_timer = jnp.where(target == 0, timer0, timer1)

    wait_action = _navigate(pos, facing, 2, 2, 4)
    wait_action = jnp.where((pos[0] == 2) & (pos[1] == 2), 4, wait_action)
    raw_pot_c = jnp.where(raw_type == 1, 3, 1)
    raw_action = _navigate(pos, facing, 2, raw_pot_c, 1)
    plate_action = _navigate(pos, facing, 2, pot_c, 1)
    dish_action = _navigate(pos, facing, 1, 2, 3)
    target_active = (pot_count >= 3) | pot_cooked | (pot_timer > 0)
    # While the requested soup cooks, build a reserve of the opposite recipe.
    # Latch the selected ingredient until it reaches its pot because the target
    # pot falls outside the crop near the opposite supply.
    proposed_fill = jnp.where(target_active & (other_count < 3), 1 - target, target)
    fill_type = jnp.where(memory[5] < 2, memory[5], proposed_fill)
    ingredient_action = jnp.where(
        fill_type == 0, _navigate(pos, facing, 1, 1, 2),
        _navigate(pos, facing, 2, 4, 0))
    get_plate_action = _navigate(pos, facing, 1, 3, 3)
    # Do not start a reserve trip when the requested soup is nearly ready: a
    # cross-kitchen round trip can otherwise delay delivery by a full refill.
    should_plate = pot_cooked | (target_active &
                   ((other_count >= 3) |
                    ((pot_timer > 0) & (pot_timer <= 5))))
    empty_action = jnp.where(should_plate, get_plate_action, ingredient_action)

    action = jnp.where(valid == 0, wait_action,
                       jnp.where(held_cooked, dish_action,
                                 jnp.where(held_plate, plate_action,
                                           jnp.where(held_raw, raw_action, empty_action))))
    delivering = (held_cooked & (pos[0] == 1) & (pos[1] == 2) &
                  (facing == 0) & (action == 5))
    valid = jnp.where(delivering, 0, valid)
    signal_run = jnp.where(delivering, 0, signal_run)
    depositing = held_raw & (action == 5) & (pos[0] == 2) & (pos[1] == raw_pot_c)
    begin_fill = (valid > 0) & (~held_plate) & (~held_cooked) & (~held_raw) & (memory[5] >= 2) & (
        ~should_plate)
    fill_plan = jnp.where(begin_fill, proposed_fill, memory[5])
    fill_plan = jnp.where(depositing | held_plate | held_cooked, 2, fill_plan)
    next_memory = jnp.asarray([target, valid, memory[2] + 1,
                               candidate, signal_run, fill_plan,
                               jnp.where(got_signal, partner_c, memory[6])], dtype=jnp.int32)
    trace = {"target": target, "signal_valid": valid, "pot_count": pot_count,
             "pot_timer": pot_timer, "self_row": pos[0], "self_col": pos[1],
             "count0": count0, "count1": count1, "fill_plan": fill_plan,
             "held_plate": held_plate, "held_raw": held_raw,
             "held_cooked": held_cooked}
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    return jnp.asarray([0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, init_agent_1(), memory)
    pos = _locate_lower(obs)
    facing = _facing(obs)
    recipe0 = jnp.sum(obs[:, :, 40])
    recipe1 = jnp.sum(obs[:, :, 41])
    observed = (recipe0 + recipe1) > 0
    target = jnp.where(observed, jnp.where(recipe1 > recipe0, 1, 0), memory[0])
    target_c = jnp.where(target == 0, 1, 3)
    counter_c = jnp.where(target == 0, 2, 4)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_raw = jnp.sum(obs[2, 2, 7:11]) > 0
    raw_type = jnp.argmax(obs[2, 2, 7:11])
    pot_ready = _grid_value(obs, pos, 3, target_c, 33) > 0
    pot_count = (_grid_value(obs, pos, 3, target_c, 34) +
                 _grid_value(obs, pos, 3, target_c, 35) +
                 _grid_value(obs, pos, 3, target_c, 36) +
                 _grid_value(obs, pos, 3, target_c, 37))
    pot_timer = _grid_value(obs, pos, 3, target_c, 44)
    pot_accepts = (~pot_ready) & (pot_timer == 0) & (pot_count < 3)
    counter_plate = ((_grid_value(obs, pos, 3, counter_c, 32) > 0) &
                     (_grid_value(obs, pos, 3, counter_c, 33) == 0))
    raw2_target = jnp.where(target == 0,
                            _grid_value(obs, pos, 3, 2, 34),
                            _grid_value(obs, pos, 3, 2, 35))
    raw4_target = jnp.where(target == 0,
                            _grid_value(obs, pos, 3, 4, 34),
                            _grid_value(obs, pos, 3, 4, 35))
    raw_counter_c = jnp.where(raw2_target > 0, 2, 4)
    matching_raw = ((raw2_target > 0) | (raw4_target > 0)) & pot_accepts

    r, c = pos[0], pos[1]
    move = jnp.where(r < 4, 1, jnp.where(r > 4, 3,
                     jnp.where(c < target_c, 0, jnp.where(c > target_c, 2, 4))))
    at_target = (r == 4) & (c == target_c)
    signal_action = jnp.where(at_target, jnp.where(facing == 0, 4, 3), move)

    # An independent upper chef may stage a plate on the shared counter beside
    # the requested pot.  Only take it when the matching soup is already ready,
    # then return the plated dish to that same counter for delivery upstairs.
    pickup_plate_action = _navigate(pos, facing, 4, counter_c, 3)
    scoop_action = _navigate(pos, facing, 4, target_c, 3)
    return_dish_action = _navigate(pos, facing, 4, counter_c, 3)
    raw_pot_c = jnp.where(raw_type == 0, 1, 3)
    deposit_raw_action = _navigate(pos, facing, 4, raw_pot_c, 3)
    pickup_raw_action = _navigate(pos, facing, 4, raw_counter_c, 3)
    action = jnp.where(held_cooked, return_dish_action,
             jnp.where(held_plate, scoop_action,
             jnp.where(held_raw, deposit_raw_action,
             jnp.where(counter_plate & pot_ready, pickup_plate_action,
             jnp.where(matching_raw, pickup_raw_action,
                       signal_action)))))
    next_memory = jnp.asarray([target, memory[1] + 1], dtype=jnp.int32)
    trace = {"target": target, "self_row": r, "self_col": c}
    return action.astype(jnp.int32), next_memory, trace
