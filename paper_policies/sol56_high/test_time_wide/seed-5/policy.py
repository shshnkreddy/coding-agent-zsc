"""Convention-robust policy for the two-room Overcooked layout.

The lower player can always see the recipe but cannot obtain either useful
ingredient. It therefore communicates recipe 0/1 by waiting below the
left/right pot. The upper player confirms that signal from the partner's
position and performs the whole production cycle.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'dual_recipe_cache_v5',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}


def _upper_position(obs):
    """Recover the global upper-room position from the visible type-0 pile."""
    pile = obs[:, :, 28]
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    lr = jnp.sum(pile * rows)
    lc = jnp.sum(pile * cols)
    row = 3 - lr
    col = jnp.where(lc <= 2, 2 - lc, 7 - lc)
    return row.astype(jnp.int32), col.astype(jnp.int32)


def _lower_position(obs):
    """Recover the global lower-room position from the visible type-3 pile."""
    pile = obs[:, :, 31]
    rows = jnp.arange(5, dtype=jnp.int32)[:, None]
    cols = jnp.arange(5, dtype=jnp.int32)[None, :]
    lr = jnp.sum(pile * rows)
    lc = jnp.sum(pile * cols)
    row = 6 - lr
    col = jnp.where(lc <= 2, 2 - lc, 7 - lc)
    return row.astype(jnp.int32), col.astype(jnp.int32)


def _facing(obs, direction):
    # Observation direction order is up, down, right, left.
    return obs[2, 2, 1 + direction] > 0


def _go_interact(obs, row, col, target_row, target_col, face_action, face_dir):
    """Navigate on the rectangular floor, then face and interact."""
    vertical = jnp.where(row < target_row, 1, 3)
    horizontal = jnp.where(col < target_col, 0, 2)
    move = jnp.where(row != target_row, vertical, horizontal)
    at_target = (row == target_row) & (col == target_col)
    finish = jnp.where(_facing(obs, face_dir), 5, face_action)
    return jnp.where(at_target, finish, move).astype(jnp.int32)


def _global_value(obs, self_row, self_col, global_row, global_col, channel):
    lr = global_row - self_row + 2
    lc = global_col - self_col + 2
    visible = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    lr = jnp.clip(lr, 0, 4)
    lc = jnp.clip(lc, 0, 4)
    return jnp.where(visible, obs[lr, lc, channel], 0)


def init_agent_0():
    # elapsed, recipe, selected pot, prior cooked, fetch mode,
    # alternate-signal candidate, alternate-signal visible streak
    # fetch mode: 0 none, 1 ingredient, 2 plate for pot
    return jnp.asarray([0, -1, 0, 0, 0, -1, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    initial = init_agent_0()
    mem = jnp.where(reset, initial, memory)
    step = mem[0]
    believed = mem[1]
    chosen_pot = mem[2]
    previous_cooked = mem[3] > 0
    fetch_mode = mem[4]
    signal_candidate = mem[5]
    signal_streak = mem[6]

    row, col = _upper_position(obs)
    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_i0 = obs[2, 2, 7] > 0
    held_i1 = obs[2, 2, 8] > 0
    hands_empty = ~(held_plate | held_cooked | held_i0 | held_i1)

    # Accept only the conspicuous, settled signal: partner directly beneath a
    # pot, facing upward toward it. Random initial positions do not qualify.
    partner = obs[:, :, 11]
    prows = jnp.arange(5, dtype=jnp.int32)[:, None]
    pcols = jnp.arange(5, dtype=jnp.int32)[None, :]
    seen = jnp.sum(partner) > 0
    plr = jnp.sum(partner * prows)
    plc = jnp.sum(partner * pcols)
    partner_row = row + plr - 2
    partner_col = col + plc - 2
    partner_up = jnp.sum(obs[:, :, 12]) > 0
    partner_right = jnp.sum(obs[:, :, 14]) > 0
    partner_left = jnp.sum(obs[:, :, 15]) > 0
    exact0 = seen & partner_up & (partner_row == 4) & (partner_col == 1)
    exact1 = seen & partner_up & (partner_row == 4) & (partner_col == 3)
    exact_signal = exact0 | exact1
    exact_value = jnp.where(exact1, 1, 0)

    # Independently generated lower policies may choose a nearby edge pose or
    # literal LEFT/RIGHT facing instead of the exact pot-facing convention.
    # Require three repeated visible observations before trusting these weaker
    # cues, and ignore them while the partner is carrying an object.
    partner_empty = jnp.sum(obs[:, :, 16:22]) == 0
    alternate0 = (partner_col == 1) | (
        (partner_col == 2) & partner_left
    )
    alternate1 = (partner_col >= 3) | (
        (partner_col == 2) & partner_right
    )
    alternate_valid = (
        seen & partner_empty & (partner_row == 4) & (alternate0 | alternate1)
    )
    alternate_value = jnp.where(alternate1, 1, 0)
    same_candidate = alternate_valid & (alternate_value == signal_candidate)
    next_signal_streak = jnp.where(
        alternate_valid,
        jnp.where(same_candidate, jnp.minimum(signal_streak + 1, 3), 1),
        0,
    )
    next_signal_candidate = jnp.where(alternate_valid, alternate_value, -1)
    confirmed_alternate = alternate_valid & (next_signal_streak >= 3)
    signaled = exact_signal | confirmed_alternate
    signal_value = jnp.where(exact_signal, exact_value, alternate_value)

    # A disappearing cooked inventory at the delivery station means that a
    # delivery just occurred; force a fresh signal for the resampled recipe.
    just_delivered = previous_cooked & hands_empty
    believed = jnp.where(just_delivered, -1, believed)
    believed = jnp.where(signaled, signal_value, believed)

    # Read both pots. At every decision point used below they are in view.
    p0_cooked = _global_value(obs, row, col, 3, 1, 33)
    p0_i0 = _global_value(obs, row, col, 3, 1, 34)
    p0_i1 = _global_value(obs, row, col, 3, 1, 35)
    p0_timer = _global_value(obs, row, col, 3, 1, 44)
    p1_cooked = _global_value(obs, row, col, 3, 3, 33)
    p1_i0 = _global_value(obs, row, col, 3, 3, 34)
    p1_i1 = _global_value(obs, row, col, 3, 3, 35)
    p1_timer = _global_value(obs, row, col, 3, 3, 44)
    counter_plate = (
        (_global_value(obs, row, col, 3, 2, 32) > 0)
        & (_global_value(obs, row, col, 3, 2, 33) == 0)
    )

    p0_other = (
        p0_i1
        + _global_value(obs, row, col, 3, 1, 36)
        + _global_value(obs, row, col, 3, 1, 37)
    )
    p1_other = (
        p1_i0
        + _global_value(obs, row, col, 3, 3, 36)
        + _global_value(obs, row, col, 3, 3, 37)
    )
    # Each pot is a permanent cache for one of the only two possible recipes.
    # This food remains useful across every future recipe resampling.
    desired_count0 = p0_i0
    wrong_count0 = p0_other
    desired_count1 = p1_i1
    wrong_count1 = p1_other
    designated = jnp.maximum(believed, 0)
    usable0 = (
        (p0_cooked == 0)
        & (p0_timer == 0)
        & (desired_count0 < 3)
        & (wrong_count0 == 0)
    )
    usable1 = (
        (p1_cooked == 0)
        & (p1_timer == 0)
        & (desired_count1 < 3)
        & (wrong_count1 == 0)
    )
    partial0 = usable0 & (desired_count0 > 0)
    partial1 = usable1 & (desired_count1 > 0)
    empty0 = usable0 & (desired_count0 == 0)
    empty1 = usable1 & (desired_count1 == 0)
    ready0 = (p0_cooked > 0) & (desired_count0 == 3) & (wrong_count0 == 0)
    ready1 = (p1_cooked > 0) & (desired_count1 == 3) & (wrong_count1 == 0)

    # Finish a partial batch first. Otherwise fill an empty pot, preferring the
    # currently requested cache when known.
    partial_pot = jnp.where(partial0, 0, 1)
    preferred_empty = jnp.where(
        designated == 0,
        jnp.where(empty0, 0, 1),
        jnp.where(empty1, 1, 0),
    )
    any_partial = partial0 | partial1
    any_empty = empty0 | empty1
    background_pot = jnp.where(any_partial, partial_pot, preferred_empty)
    target_ready = ((believed == 0) & ready0) | ((believed == 1) & ready1)
    ready_pot = designated
    target_timer = jnp.where(designated == 0, p0_timer, p1_timer)
    target_can_take = jnp.where(designated == 0, usable0, usable1)
    known_recipe = believed >= 0
    # The target cache has priority. While it cooks, maintain whichever other
    # cache needs work; once the target is ready, deliver immediately.
    fill_pot = jnp.where(known_recipe & target_can_take, designated, background_pot)
    background_work = any_partial | any_empty
    can_fill = jnp.where(
        known_recipe,
        target_can_take | (~target_ready & background_work),
        background_work,
    )
    pot_ready = target_ready

    # Inventory-specific objectives.
    delivery_action = _go_interact(obs, row, col, 1, 2, 3, 0)
    plate_target_col = jnp.where(chosen_pot == 0, 1, 3)
    scoop_action = _go_interact(obs, row, col, 2, plate_target_col, 1, 1)
    counter_action = _go_interact(obs, row, col, 2, 2, 1, 1)
    ingredient_target_col = jnp.where(chosen_pot == 0, 1, 3)
    deposit_action = _go_interact(obs, row, col, 2, ingredient_target_col, 1, 1)

    # Empty-handed objectives. Waiting at (2,2) maximizes visibility of either
    # signal position. Ingredient supplies and plate/goal are fixed.
    observe_action = _go_interact(obs, row, col, 2, 2, 4, 0)
    get_i0_left = _go_interact(obs, row, col, 1, 1, 2, 3)
    get_i0_right = _go_interact(obs, row, col, 1, 4, 0, 2)
    get_i1_left = _go_interact(obs, row, col, 2, 1, 2, 3)
    get_i1_right = _go_interact(obs, row, col, 2, 4, 0, 2)
    pickup_pot = jnp.where(fetch_mode == 1, chosen_pot, fill_pot)
    # Pot identity now also determines recipe identity. Use the nearest mirrored
    # supply for that fixed pot.
    get_ingredient = jnp.where(pickup_pot == 0, get_i0_left, get_i1_right)
    get_plate_pile = _go_interact(obs, row, col, 1, 3, 3, 0)
    get_plate = jnp.where(counter_plate, counter_action, get_plate_pile)
    # Completing a partial pot outranks plating, preventing a recipe-changing
    # delivery from stranding an unfinished old batch. Empty-pot pipelining is
    # also done before plating because cooked food never burns.
    idle_action = jnp.where(believed >= 0, 4, observe_action)
    empty_action = jnp.where(
        can_fill, get_ingredient, jnp.where(pot_ready, get_plate, idle_action)
    )
    empty_action = jnp.where(fetch_mode == 1, get_ingredient, empty_action)
    empty_action = jnp.where(fetch_mode == 2, get_plate, empty_action)
    prefetch_plate = (
        hands_empty
        & (fetch_mode == 0)
        & (believed >= 0)
        & ~can_fill
        & ~pot_ready
        & (target_timer > 0)
    )
    empty_action = jnp.where(prefetch_plate, get_plate, empty_action)

    action = empty_action
    action = jnp.where(held_i0 | held_i1, deposit_action, action)
    action = jnp.where(held_plate & ~held_cooked, scoop_action, action)
    action = jnp.where(held_cooked, delivery_action, action)

    # Lock the designated pot when leaving to fetch an item.
    fetching_ingredient = (
        hands_empty & (fetch_mode == 0) & can_fill
    )
    fetching_plate = (
        hands_empty
        & (fetch_mode == 0)
        & ~can_fill
        & (pot_ready | prefetch_plate)
    )
    chosen_pot = jnp.where(fetching_ingredient, fill_pot, chosen_pot)
    chosen_pot = jnp.where(fetching_plate, ready_pot, chosen_pot)
    next_fetch_mode = jnp.where(fetching_ingredient, 1, fetch_mode)
    next_fetch_mode = jnp.where(fetching_plate, 2, next_fetch_mode)
    ingredient_acquired = (fetch_mode == 1) & (held_i0 | held_i1)
    scoop_plate_acquired = (fetch_mode == 2) & held_plate
    next_fetch_mode = jnp.where(
        ingredient_acquired | scoop_plate_acquired,
        0,
        next_fetch_mode,
    )
    next_memory = jnp.asarray(
        [
            step + 1,
            believed,
            chosen_pot,
            held_cooked.astype(jnp.int32),
            next_fetch_mode,
            next_signal_candidate,
            next_signal_streak,
        ],
        dtype=jnp.int32,
    )
    trace = {
        'belief': believed,
        'global_pos': jnp.asarray([row, col], dtype=jnp.int32),
        'signal': jnp.where(signaled, signal_value, -1),
        'pot_ready': pot_ready.astype(jnp.int32),
        'fetch_mode': fetch_mode,
        'signal_streak': next_signal_streak,
    }
    return action, next_memory, trace


def init_agent_1():
    return jnp.asarray([0, -1], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    mem = jnp.where(reset, initial, memory)
    row, col = _lower_position(obs)
    recipe0 = jnp.sum(obs[:, :, 40])
    recipe1 = jnp.sum(obs[:, :, 41])
    visible = (recipe0 + recipe1) > 0
    recipe = jnp.where(recipe1 > recipe0, 1, 0)
    recipe = jnp.where(visible, recipe, mem[1])
    target_col = jnp.where(recipe == 0, 1, 3)

    # Stand directly below the corresponding pot and face it. Once settled,
    # STAY preserves the clearest possible positional/facing signal.
    nav = _go_interact(obs, row, col, 4, target_col, 3, 0)
    settled = (row == 4) & (col == target_col) & _facing(obs, 0)
    action = jnp.where(settled, 4, nav).astype(jnp.int32)
    next_memory = jnp.asarray([mem[0] + 1, recipe], dtype=jnp.int32)
    trace = {
        'recipe': recipe,
        'global_pos': jnp.asarray([row, col], dtype=jnp.int32),
    }
    return action, next_memory, trace
