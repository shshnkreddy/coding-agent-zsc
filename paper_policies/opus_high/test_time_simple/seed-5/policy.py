"""Two-room Overcooked coordination policy for ``test_time_simple``.

The 5x8 board is split by the wall column 3 into two disconnected rooms, so the
roles are forced by geometry rather than by any private convention:

* agent_0's room (rows 1-3, cols 1-2) touches only the ingredient-2 supplies at
  (0,2)/(4,2), the always-on recipe indicator at (2,0) and the counters
  (1,3),(2,3),(3,3).  Ingredient 2 appears in no recipe, so agent_0 can never
  cook or deliver anything.
* agent_1's room (rows 1-3, cols 5-6) touches the pot (2,4), both plate piles,
  the ingredient-0 supply (1,7), the ingredient-1 supply (3,7) and the delivery
  station (2,7).  It can complete a whole dish alone, but with a view radius of
  2 it can never see the recipe indicator at column 0.

The only information channel is therefore item presence on the column-3
counters: agent_0 can place/remove from column 2, and agent_1 can see column 3
whenever it stands in column 5.  The signalling convention mirrors the supply
rows, which is the natural reading of the layout for either side: the upper
counter (1,3) means ingredient 0 (whose supply is the upper one, at (1,7)) and
the lower counter (3,3) means ingredient 1 (supply at (3,7)).  agent_1 decodes
relationally where it can, so a partner that uses the middle counter as one of
its two markers is still understood once both markers have been seen.

Wrong deliveries cost -20 *and* do not resample the recipe, so blindly repeating
one recipe is an absorbing loss.  agent_1 therefore never cooks on a stale or
ambiguous reading; and if a partner never produces any readable marker at all it
falls back on guesses that alternate after every delivery, which turns that same
no-resample rule into a guarantee that a miss is followed by a hit.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'counter_signal_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
A_RIGHT = 0
A_DOWN = 1
A_LEFT = 2
A_UP = 3
A_STAY = 4
A_INTERACT = 5

# Facing ids as they appear in observation channels 1..4.
F_UP = 0
F_DOWN = 1
F_RIGHT = 2
F_LEFT = 3

# Steps agent_1 will wait after one of its own deliveries before trusting a
# signal that still reads as the recipe it just delivered.  This is the default
# used until the partner's own clearing latency has been measured; thereafter
# the wait is that measured latency plus CONFIRM_MARGIN, bounded by the two
# limits below.  A prompt signaller is answered in about 5 steps, which is worth
# roughly one extra delivery per episode, while a sluggish one is measured as
# sluggish and still gets a wait long enough to avoid a stale reading.
CONFIRM_STEPS = 8
CONFIRM_MARGIN = 2
CONFIRM_MIN = 4
CONFIRM_MAX = 12

# Steps agent_1 will wait for any usable signal before cooking on a blind guess.
# Well beyond what a working signaller needs, so it only fires against a partner
# that never produces a readable marker.
FALLBACK_STEPS = 30


def _turn_action(want_face):
    """Movement action that produces ``want_face``.

    Every cell we ever turn towards is blocked (wall, supply, pile, pot or
    delivery station), so the turn never translates the player.
    """
    return jnp.where(
        want_face == F_UP, A_UP,
        jnp.where(want_face == F_DOWN, A_DOWN,
                  jnp.where(want_face == F_RIGHT, A_RIGHT, A_LEFT)))


def _facing(obs):
    """Own facing id; the observer is always at local cell (2, 2)."""
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _locate(obs, channel, row_of, col_of):
    """Global (row, col) of self from the unique static cell in ``channel``."""
    plane = obs[:, :, channel]
    flat = jnp.argmax(plane.reshape(-1))
    local_row = (flat // 5).astype(jnp.int32)
    local_col = (flat % 5).astype(jnp.int32)
    visible = jnp.sum(plane) > 0
    return row_of(local_row), col_of(local_col), visible, local_row, local_col


def _occupied(obs, local_row, local_col):
    """True when the grid cell carries any item (counter or pot contents)."""
    cell = obs[local_row, local_col]
    return jnp.sum(cell[29:34]) > 0


def _step_action(row, col, goal_row, goal_col, want_face, interact, facing):
    """Walk to the goal cell, turn to ``want_face``, then interact or wait.

    Both rooms are obstacle-free 3x2 rectangles containing a single player, so
    moving vertically and then horizontally always reaches the goal.
    """
    d_row = goal_row - row
    d_col = goal_col - col
    move = jnp.where(
        d_row < 0, A_UP,
        jnp.where(d_row > 0, A_DOWN, jnp.where(d_col < 0, A_LEFT, A_RIGHT)))
    arrived = (d_row == 0) & (d_col == 0)
    settled = jnp.where(
        facing == want_face,
        jnp.where(interact, A_INTERACT, A_STAY),
        _turn_action(want_face))
    return jnp.where(arrived, settled, move).astype(jnp.int32)


# ---------------------------------------------------------------- agent_0 ---


def init_agent_0():
    return {
        'last_recipe': jnp.asarray(0, dtype=jnp.int32),
        'step': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    last_recipe = jnp.where(reset, 0, memory['last_recipe']).astype(jnp.int32)
    step = jnp.where(reset, 0, memory['step']).astype(jnp.int32)

    # The recipe indicator at (2,0) is inside the view from every cell of the
    # left room, which pins down our own global position as well.
    row, col, visible, ind_row, ind_col = _locate(
        obs, 23, lambda lr: 4 - lr, lambda lc: 2 - lc)
    row = jnp.where(visible, jnp.clip(row, 1, 3), 2).astype(jnp.int32)
    col = jnp.where(visible, jnp.clip(col, 1, 2), 2).astype(jnp.int32)

    indicator = obs[ind_row, ind_col]
    count_0 = indicator[36]
    count_1 = indicator[37]
    shown = visible & ((count_0 + count_1) > 0)
    recipe = jnp.where(shown, jnp.where(count_0 >= count_1, 0, 1),
                       last_recipe).astype(jnp.int32)

    # Counters (1,3), (2,3), (3,3) are always inside the view from the room.
    signal_col = 5 - col
    occ_top = _occupied(obs, 1 - row + 2, signal_col)
    occ_mid = _occupied(obs, 2 - row + 2, signal_col)
    occ_bot = _occupied(obs, 3 - row + 2, signal_col)

    holding = jnp.sum(obs[2, 2, 5:10]) > 0

    target_row = jnp.where(recipe == 0, 1, 3).astype(jnp.int32)
    other_row = jnp.where(recipe == 0, 3, 1).astype(jnp.int32)
    occ_target = jnp.where(recipe == 0, occ_top, occ_bot)
    occ_other = jnp.where(recipe == 0, occ_bot, occ_top)

    # Clear stale markers before placing the new one: a momentarily empty
    # channel reads as "no information", whereas two occupied counters could be
    # misread as a definite recipe by a partner that only checks one cell.
    clear_other = (~holding) & occ_other
    clear_mid = (~holding) & occ_mid & (~occ_other)
    place = holding & (~occ_target)
    fetch = (~holding) & (~occ_target) & (~occ_other) & (~occ_mid)

    goal_row = jnp.where(clear_other, other_row,
                         jnp.where(clear_mid, 2, target_row)).astype(jnp.int32)
    goal_col = jnp.asarray(2, dtype=jnp.int32)
    # Fetching faces the ingredient-2 supply above (0,2) or below (4,2) us;
    # everything else faces the counter column.
    want_face = jnp.where(
        fetch, jnp.where(target_row == 1, F_UP, F_DOWN), F_RIGHT).astype(jnp.int32)
    interact = clear_other | clear_mid | place | fetch

    action = _step_action(row, col, goal_row, goal_col, want_face, interact,
                          _facing(obs))

    next_memory = {
        'last_recipe': recipe,
        'step': step + 1,
    }
    trace = {
        'recipe': recipe,
        'row': row,
        'col': col,
        'occupancy': jnp.stack([occ_top.astype(jnp.int32),
                                occ_mid.astype(jnp.int32),
                                occ_bot.astype(jnp.int32)]),
        'holding': holding.astype(jnp.int32),
    }
    return action, next_memory, trace


# ---------------------------------------------------------------- agent_1 ---


def init_agent_1():
    return {
        'belief': jnp.asarray(-1, dtype=jnp.int32),
        'prev_belief': jnp.asarray(-1, dtype=jnp.int32),
        'wait': jnp.asarray(0, dtype=jnp.int32),
        'broke': jnp.asarray(0, dtype=jnp.int32),
        'had_dish': jnp.asarray(0, dtype=jnp.int32),
        'guess': jnp.asarray(0, dtype=jnp.int32),
        'latency': jnp.asarray(-1, dtype=jnp.int32),
        'seen_top': jnp.asarray(0, dtype=jnp.int32),
        'seen_bot': jnp.asarray(0, dtype=jnp.int32),
        'step': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    fresh_start = reset
    belief = jnp.where(fresh_start, -1, memory['belief']).astype(jnp.int32)
    prev_belief = jnp.where(fresh_start, -1, memory['prev_belief']).astype(jnp.int32)
    wait = jnp.where(fresh_start, 0, memory['wait']).astype(jnp.int32)
    broke = jnp.where(fresh_start, 0, memory['broke']).astype(jnp.int32)
    had_dish = jnp.where(fresh_start, 0, memory['had_dish']).astype(jnp.int32)
    guess = jnp.where(fresh_start, 0, memory['guess']).astype(jnp.int32)
    latency = jnp.where(fresh_start, -1, memory['latency']).astype(jnp.int32)
    seen_top = jnp.where(fresh_start, 0, memory['seen_top']).astype(jnp.int32)
    seen_bot = jnp.where(fresh_start, 0, memory['seen_bot']).astype(jnp.int32)
    step = jnp.where(fresh_start, 0, memory['step']).astype(jnp.int32)

    # The pot at (2,4) is inside the view from every cell of the right room.
    row, col, pot_visible, pot_row, pot_col = _locate(
        obs, 22, lambda lr: 4 - lr, lambda lc: 6 - lc)
    row = jnp.where(pot_visible, jnp.clip(row, 1, 3), 2).astype(jnp.int32)
    col = jnp.where(pot_visible, jnp.clip(col, 5, 6), 5).astype(jnp.int32)

    pot = obs[pot_row, pot_col]
    pot_count = pot[31] + pot[32] + pot[33]
    pot_cooked = pot[30] > 0
    pot_busy = (pot_count >= 3) | pot_cooked | (pot[39] > 0)

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ingredient = jnp.sum(obs[2, 2, 7:10]) > 0
    held_dish = held_plate & held_cooked
    held_empty_plate = held_plate & (~held_cooked)
    held_anything = held_plate | held_cooked | held_ingredient

    # Losing a dish can only mean we handed it to the delivery station.  A
    # correct delivery resamples the recipe, so the belief must be re-earned.
    delivered = (had_dish > 0) & (~held_dish)
    # Blind fallback bookkeeping.  A wrong delivery does not resample the
    # recipe and there are only two of them, so flipping the guess after every
    # delivery means a miss is always followed by a certain hit: that is worth
    # about +6.7 per delivery against a partner that never signals, whereas
    # repeating one recipe blindly is an absorbing loss.
    guess = jnp.where(delivered,
                      jnp.where(belief >= 0, 1 - belief, 1 - guess),
                      guess).astype(jnp.int32)
    prev_belief = jnp.where(delivered, belief, prev_belief).astype(jnp.int32)
    belief = jnp.where(delivered, -1, belief).astype(jnp.int32)
    wait = jnp.where(delivered, 0, wait).astype(jnp.int32)
    broke = jnp.where(delivered, 0, broke).astype(jnp.int32)

    # Column 3 is only inside the view from column 5.
    readable = col == 5
    signal_col = 5 - col
    occ_top = _occupied(obs, 3 - row, signal_col) & readable
    occ_mid = _occupied(obs, 4 - row, signal_col) & readable
    occ_bot = _occupied(obs, 5 - row, signal_col) & readable
    unique = (occ_top.astype(jnp.int32) + occ_mid.astype(jnp.int32)
              + occ_bot.astype(jnp.int32)) == 1

    seen_top = jnp.where(unique & occ_top, 1, seen_top).astype(jnp.int32)
    seen_bot = jnp.where(unique & occ_bot, 1, seen_bot).astype(jnp.int32)
    # A partner whose two markers are {middle, bottom} means ingredient 0 by the
    # middle one; with {top, middle} the middle one is the lower marker.
    mid_meaning = jnp.where(seen_bot > 0, 0,
                            jnp.where(seen_top > 0, 1, -1)).astype(jnp.int32)
    signal = jnp.where(
        unique & occ_top, 0,
        jnp.where(unique & occ_bot, 1,
                  jnp.where(unique & occ_mid, mid_meaning, -1))).astype(jnp.int32)
    signal = jnp.where(readable, signal, -2).astype(jnp.int32)

    # The belief is free to move only while nothing of this cycle is committed;
    # once the first ingredient is in hand it is frozen until the delivery.
    open_cycle = (pot_count == 0) & (~pot_cooked) & (~held_ingredient) & (~held_dish)
    wait = jnp.where(open_cycle, wait + 1, wait).astype(jnp.int32)

    # How long the partner takes to retire a marker after one of our deliveries.
    # Only the first break of each post-delivery await is a measurement, and
    # only once a delivery has actually happened; the running maximum keeps the
    # estimate on the conservative side.  A partner that never clears is never
    # measured, so it keeps the cautious default.
    first_break = ((belief < 0) & (prev_belief >= 0) & readable
                   & (signal < 0) & (broke == 0))
    latency = jnp.where(first_break, jnp.maximum(latency, wait), latency).astype(jnp.int32)
    confirm = jnp.where(
        latency < 0, CONFIRM_STEPS,
        jnp.clip(latency + CONFIRM_MARGIN, CONFIRM_MIN, CONFIRM_MAX)).astype(jnp.int32)

    broke = jnp.where(readable & (signal < 0), 1, broke).astype(jnp.int32)
    trustworthy = (signal >= 0) & (
        (signal != prev_belief) | (broke > 0) | (wait >= confirm))
    belief = jnp.where(open_cycle & trustworthy, signal, belief).astype(jnp.int32)
    # Nothing readable for a long time: fall back on the alternating guess.
    belief = jnp.where(open_cycle & (belief < 0) & (wait >= FALLBACK_STEPS),
                       guess, belief).astype(jnp.int32)

    side_row = jnp.where(belief == 1, 3, 1).astype(jnp.int32)

    # Lowest priority: fetch the next ingredient of the believed recipe.
    goal_row = side_row
    goal_col = jnp.asarray(6, dtype=jnp.int32)
    want_face = jnp.asarray(F_RIGHT, dtype=jnp.int32)
    interact = True

    def prefer(selected, g_row, g_col, face, do_interact):
        return (
            jnp.where(selected, g_row, goal_row).astype(jnp.int32),
            jnp.where(selected, g_col, goal_col).astype(jnp.int32),
            jnp.where(selected, face, want_face).astype(jnp.int32),
            jnp.where(selected, do_interact, interact),
        )

    # Safety only: park a stray plate on the counter beside the pot.
    goal_row, goal_col, want_face, interact = prefer(
        held_empty_plate & (~pot_busy), side_row, 5,
        jnp.where(belief == 1, F_DOWN, F_UP), True)
    # No trusted recipe yet: stand in column 5, where the signal is visible.
    goal_row, goal_col, want_face, interact = prefer(
        (belief < 0) & (~pot_busy), 2, 5, F_LEFT, False)
    # Carrying an ingredient for a pot that still has room: put it in.
    goal_row, goal_col, want_face, interact = prefer(
        held_ingredient & (~pot_busy), 2, 5, F_LEFT, True)
    # Safety only: a spare ingredient once the pot is already going.
    goal_row, goal_col, want_face, interact = prefer(
        held_ingredient & pot_busy, 1, 5, F_UP, True)
    # Pot is cooking or ready and our hands are free: collect a plate.
    goal_row, goal_col, want_face, interact = prefer(
        (~held_anything) & pot_busy, side_row, 5, F_LEFT, True)
    # Waiting at the pot with a plate; the interact is a no-op until it is
    # cooked, so the dish is scooped on the very first step it is available.
    goal_row, goal_col, want_face, interact = prefer(
        held_empty_plate & pot_busy, 2, 5, F_LEFT, True)
    # Highest priority: deliver.
    goal_row, goal_col, want_face, interact = prefer(
        held_dish, 2, 6, F_RIGHT, True)

    action = _step_action(row, col, goal_row, goal_col, want_face, interact,
                          _facing(obs))

    next_memory = {
        'belief': belief,
        'prev_belief': prev_belief,
        'wait': wait,
        'broke': broke,
        'had_dish': held_dish.astype(jnp.int32),
        'guess': guess,
        'latency': latency,
        'seen_top': seen_top,
        'seen_bot': seen_bot,
        'step': step + 1,
    }
    trace = {
        'belief': belief,
        'signal': signal,
        'row': row,
        'col': col,
        'pot_count': pot_count.astype(jnp.int32),
        'pot_cooked': pot_cooked.astype(jnp.int32),
        'wait': wait,
        'confirm': confirm,
        'delivered': delivered.astype(jnp.int32),
    }
    return action, next_memory, trace
