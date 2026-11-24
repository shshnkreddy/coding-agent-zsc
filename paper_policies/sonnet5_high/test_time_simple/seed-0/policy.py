"""Counter-signal chef policy for the isolated-rooms Overcooked V2 layout.

Layout facts (see context/environment/README.md and specification.json):
  - agent_0's room (cols 1-2, rows 1-3) only contains the always-on recipe
    indicator R and the (recipe-irrelevant) ingredient-2 supply. It cannot
    reach the pot, plate piles, ingredient-0/1 supplies, or the delivery
    station.
  - agent_1's room (cols 5-6, rows 1-3) contains the pot, both plate piles,
    ingredient-0/1 supplies, and the delivery station, but never sees the
    recipe indicator.
  - The two rooms share no floor cell; items cannot be passed directly. The
    only cross-room information channel is that the three dividing counters
    at column 3, rows 1/2/3 are always visible to agent_0, and visible to
    agent_1 only while agent_1 stands in column 5 (since agent_1's crop is
    self_col +/- 2 and 3 is within that range only when self_col == 5).

Strategy:
  - agent_0 continuously mirrors the current recipe by holding/placing one
    unit of ingredient 2 (its only reachable, recipe-irrelevant item) on the
    row-1 counter for recipe [0,0,0] or the row-3 counter for recipe
    [1,1,1], removing/relocating a stale token when the recipe changes.
  - agent_1 self-localizes every step from the always-visible pot channel,
    opportunistically reads the counter signal whenever it is in column 5,
    and commits to a belief either as soon as an unambiguous signal appears
    or after a bounded wait (so it never stalls forever if a cross-play
    partner does not implement this convention; it then falls back to the
    previous/default belief, which defaults to ingredient 0 -- a natural
    Schelling point). It then fetches three units of the believed
    ingredient, deposits them into the pot, prefetches a plate while the pot
    cooks, collects the cooked dish, and delivers it, looping for the whole
    episode.

Both act functions are pure reflex controllers: the action each step is
recomputed entirely from the current local observation (self position is
recovered via a known static-object anchor) plus a small amount of private
memory, using only jnp.where-based fixed-shape control flow.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'counter_signal_chef',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
ACT_RIGHT = 0
ACT_DOWN = 1
ACT_LEFT = 2
ACT_UP = 3
ACT_STAY = 4
ACT_INTERACT = 5

# Observation channel indices (see context/environment/README.md).
CH_SELF_FACE_UP = 1
CH_SELF_FACE_DOWN = 2
CH_SELF_FACE_RIGHT = 3
CH_SELF_FACE_LEFT = 4
CH_SELF_PLATE = 5
CH_SELF_COOKED = 6
CH_SELF_ING0 = 7
CH_SELF_ING1 = 8
CH_SELF_ING2 = 9
CH_STATIC_POT = 22
CH_STATIC_RECIPE_IND = 23
CH_ITEM_COOKED = 30
CH_ITEM_ING0 = 31
CH_ITEM_ING1 = 32
CH_ITEM_ING2 = 33
CH_RECIPE_ING0 = 36
CH_RECIPE_ING1 = 37


def _argmax2d(channel_slice):
    """Return (row, col) local indices of the (assumed unique) max cell."""
    flat = channel_slice.reshape(-1)
    idx = jnp.argmax(flat).astype(jnp.int32)
    return idx // 5, idx % 5


def _facing_action_id(obs):
    """Current facing expressed as the movement-action id that would face it."""
    up = obs[2, 2, CH_SELF_FACE_UP]
    down = obs[2, 2, CH_SELF_FACE_DOWN]
    right = obs[2, 2, CH_SELF_FACE_RIGHT]
    left = obs[2, 2, CH_SELF_FACE_LEFT]
    return (up * ACT_UP + down * ACT_DOWN + right * ACT_RIGHT + left * ACT_LEFT).astype(jnp.int32)


def _navigate(self_row, self_col, facing_id, target_row, target_col, need_face, face_action, do_interact):
    """Generic single-step controller: walk to a cell, face a direction, interact."""
    return jnp.where(
        self_row != target_row,
        jnp.where(target_row > self_row, ACT_DOWN, ACT_UP),
        jnp.where(
            self_col != target_col,
            jnp.where(target_col > self_col, ACT_RIGHT, ACT_LEFT),
            jnp.where(
                (need_face > 0) & (facing_id != face_action),
                face_action,
                jnp.where(do_interact > 0, ACT_INTERACT, ACT_STAY),
            ),
        ),
    ).astype(jnp.int32)


# ---------------------------------------------------------------------------
# agent_0: recipe spotter / signaler. Stateless reflex; memory is a dummy
# scalar kept only to satisfy the interface.
# ---------------------------------------------------------------------------

def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)

    i, j = _argmax2d(obs[:, :, CH_STATIC_RECIPE_IND])
    # Recipe indicator R is at global (2, 0); local (i,j) <-> global
    # (self_row + i - 2, self_col + j - 2).
    self_row = 4 - i
    self_col = 2 - j

    recipe0 = obs[i, j, CH_RECIPE_ING0] > 0
    desired_row = jnp.where(recipe0, jnp.int32(1), jnp.int32(3))
    other_row = jnp.where(recipe0, jnp.int32(3), jnp.int32(1))

    holding_ing2 = obs[2, 2, CH_SELF_ING2] > 0

    def counter_present(row):
        li = jnp.clip(row - self_row + 2, 0, 4)
        lj = jnp.clip(3 - self_col + 2, 0, 4)
        return obs[li, lj, CH_ITEM_ING2] > 0

    desired_present = counter_present(desired_row)
    other_present = counter_present(other_row)

    # Goal codes: 0 stay, 1 fetch-top-supply, 2 fetch-bottom-supply,
    # 3 interact-counter-row1, 4 interact-counter-row2 (unused), 5
    # interact-counter-row3. Counter interaction places a held token onto an
    # empty counter, or picks one up from an occupied counter with empty
    # hands -- the environment resolves which based on state.
    goal_fetch = jnp.where(desired_row == 1, jnp.int32(1), jnp.int32(2))
    goal_counter_other = jnp.where(other_row == 1, jnp.int32(3), jnp.int32(5))
    goal_counter_desired = jnp.where(desired_row == 1, jnp.int32(3), jnp.int32(5))

    goal = jnp.int32(0)
    goal = jnp.where(~holding_ing2 & ~other_present & ~desired_present, goal_fetch, goal)
    goal = jnp.where(~holding_ing2 & ~other_present & desired_present, jnp.int32(0), goal)
    goal = jnp.where(~holding_ing2 & other_present, goal_counter_other, goal)
    goal = jnp.where(holding_ing2 & ~desired_present, goal_counter_desired, goal)
    goal = jnp.where(holding_ing2 & desired_present, jnp.int32(0), goal)

    target_row_table = jnp.array([0, 1, 3, 1, 2, 3], dtype=jnp.int32)
    target_col_table = jnp.array([0, 2, 2, 2, 2, 2], dtype=jnp.int32)
    face_table = jnp.array([0, 3, 1, 0, 0, 0], dtype=jnp.int32)
    do_interact_table = jnp.array([0, 1, 1, 1, 1, 1], dtype=jnp.int32)

    tr = target_row_table[goal]
    tc = target_col_table[goal]
    face = face_table[goal]
    do_int = do_interact_table[goal]

    facing_id = _facing_action_id(obs)
    nav_action = _navigate(self_row, self_col, facing_id, tr, tc, jnp.int32(1), face, do_int)
    action = jnp.where(goal == 0, ACT_STAY, nav_action).astype(jnp.int32)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    return action, next_memory, {}


# ---------------------------------------------------------------------------
# agent_1: chef. Memory is [belief, checked_this_cycle, cycle_wait].
# ---------------------------------------------------------------------------

WAIT_LIMIT = 7


def init_agent_1():
    # [belief, checked_this_cycle, cycle_wait, fallback_toggle]
    return jnp.zeros((4,), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    belief = memory[0]
    checked = memory[1]
    cycle_wait = memory[2]
    fallback_toggle = memory[3]

    ip, jp = _argmax2d(obs[:, :, CH_STATIC_POT])
    # Pot is at global (2, 4).
    self_row = 4 - ip
    self_col = 6 - jp

    pot_ing0 = obs[ip, jp, CH_ITEM_ING0]
    pot_ing1 = obs[ip, jp, CH_ITEM_ING1]
    pot_ing2 = obs[ip, jp, CH_ITEM_ING2]
    pot_count = pot_ing0 + pot_ing1 + pot_ing2
    pot_cooked = obs[ip, jp, CH_ITEM_COOKED] > 0

    h_plate = obs[2, 2, CH_SELF_PLATE] > 0
    h_cooked = obs[2, 2, CH_SELF_COOKED] > 0
    h_ing0 = obs[2, 2, CH_SELF_ING0] > 0
    h_ing1 = obs[2, 2, CH_SELF_ING1] > 0

    holding_dish = h_plate & h_cooked
    holding_empty_plate = h_plate & (~h_cooked)
    holding_raw = (h_ing0 | h_ing1) & (~h_plate)
    holding_nothing = (~h_plate) & (~h_cooked) & (~h_ing0) & (~h_ing1)

    mask_col5 = self_col == 5
    li1 = jnp.clip(1 - self_row + 2, 0, 4)
    li3 = jnp.clip(3 - self_row + 2, 0, 4)
    lj = jnp.clip(3 - self_col + 2, 0, 4)
    top_present = mask_col5 & (obs[li1, lj, CH_ITEM_ING2] > 0)
    bottom_present = mask_col5 & (obs[li3, lj, CH_ITEM_ING2] > 0)
    clear_signal = jnp.logical_xor(top_present, bottom_present)
    signal_belief = jnp.where(top_present, jnp.int32(0), jnp.int32(1))

    checked_bool = checked > 0
    trigger = (~checked_bool) & (clear_signal | (cycle_wait >= WAIT_LIMIT))
    # A clear signal is always trusted. Absent that, falling back to a fixed
    # guess is dangerous: wrong deliveries never resample the recipe, so a
    # frozen guess that misses once is stuck missing for the rest of the
    # episode. Alternating the fallback guess each time the wait times out
    # instead guarantees a hit on the very next attempt after any miss
    # (whichever of the two recipes is "stuck" is exactly the one the
    # alternation lands on next), bounding the damage of any single miss.
    #
    # A partner that signals with the mirror-image row<->recipe mapping
    # would make trusting the signal catastrophic instead (confidently wrong
    # forever). That risk is judged low relative to its cost: rescuing it
    # would require distrusting the signal on the large majority of cycles
    # (empirically, escaping the lock needs a >=~75% distrust rate), which
    # would destroy matched-convention throughput far more than the small
    # probability of facing a deliberately inverted, less-natural mapping
    # justifies. The chosen row<->recipe mapping mirrors shared geometry
    # (top counter row <-> top ingredient-0 supply row), which is a low-
    # complexity Schelling point that independently reasoning partners
    # solving the same layout are more likely to converge on than to invert.
    used_fallback_now = trigger & (~clear_signal)
    new_belief = jnp.where(
        trigger & clear_signal, signal_belief, jnp.where(used_fallback_now, fallback_toggle, belief)
    )
    new_fallback_toggle = jnp.where(used_fallback_now, 1 - fallback_toggle, fallback_toggle)
    new_checked_bool = checked_bool | trigger
    new_cycle_wait = jnp.where(checked_bool, cycle_wait, cycle_wait + 1)

    pot_lt3 = pot_count < 3

    # Goal codes: 0 stay, 1 deliver, 2 collect, 3 fetch-plate, 4 deposit,
    # 5 seek-signal, 6 fetch-ingredient-0, 7 fetch-ingredient-1.
    goal = jnp.int32(0)
    goal = jnp.where(pot_lt3 & holding_empty_plate, jnp.int32(0), goal)
    goal = jnp.where(
        pot_lt3 & holding_nothing & new_checked_bool,
        jnp.where(new_belief == 0, jnp.int32(6), jnp.int32(7)),
        goal,
    )
    goal = jnp.where(pot_lt3 & holding_nothing & (~new_checked_bool), jnp.int32(5), goal)
    goal = jnp.where(pot_lt3 & holding_raw, jnp.int32(4), goal)
    goal = jnp.where((~pot_lt3) & (~pot_cooked) & holding_raw, jnp.int32(0), goal)
    goal = jnp.where((~pot_lt3) & (~pot_cooked) & holding_empty_plate, jnp.int32(0), goal)
    goal = jnp.where((~pot_lt3) & (~pot_cooked) & holding_nothing, jnp.int32(3), goal)
    goal = jnp.where(pot_cooked & holding_raw, jnp.int32(0), goal)
    goal = jnp.where(pot_cooked & holding_nothing, jnp.int32(3), goal)
    goal = jnp.where(pot_cooked & holding_empty_plate, jnp.int32(2), goal)
    goal = jnp.where(holding_dish, jnp.int32(1), goal)

    is_delivering_goal = goal == 1
    final_checked = jnp.where(is_delivering_goal, jnp.int32(0), new_checked_bool.astype(jnp.int32))
    final_cycle_wait = jnp.where(is_delivering_goal, jnp.int32(0), new_cycle_wait)

    target_row_table = jnp.array([0, 2, 2, 1, 2, 2, 1, 3], dtype=jnp.int32)
    target_col_table = jnp.array([0, 6, 5, 5, 5, 5, 6, 6], dtype=jnp.int32)
    face_table = jnp.array([0, 0, 2, 2, 2, 0, 0, 0], dtype=jnp.int32)
    need_face_table = jnp.array([0, 1, 1, 1, 1, 0, 1, 1], dtype=jnp.int32)
    do_interact_table = jnp.array([0, 1, 1, 1, 1, 0, 1, 1], dtype=jnp.int32)

    tr = target_row_table[goal]
    tc = target_col_table[goal]
    face = face_table[goal]
    need_face = need_face_table[goal]
    do_int = do_interact_table[goal]

    facing_id = _facing_action_id(obs)
    nav_action = _navigate(self_row, self_col, facing_id, tr, tc, need_face, face, do_int)
    action = jnp.where(goal == 0, ACT_STAY, nav_action).astype(jnp.int32)

    next_memory = jnp.stack([
        new_belief.astype(jnp.int32),
        final_checked.astype(jnp.int32),
        final_cycle_wait.astype(jnp.int32),
        new_fallback_toggle.astype(jnp.int32),
    ])
    next_memory = jnp.where(reset, jnp.zeros((4,), dtype=jnp.int32), next_memory)

    return action, next_memory, {}
