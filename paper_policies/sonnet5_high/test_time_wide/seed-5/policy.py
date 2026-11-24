"""Overcooked V2 (test_time_wide) cooperative policy.

Board (fixed, 7x6):
    WWXBWW
    0 A  0
    1    1
    WPWPWW
    3 A  3
    W    W
    WWRWWW

agent_0 spawns in the top room (rows 1-2), an open 2x4 rectangle with no
internal obstacles, which has access to ingredient supplies 0 and 1 (both
used by recipes), the plate pile and the delivery station (all reachable by
facing up from row 1), and both pots (facing down from row 2). agent_0 can
never see the recipe indicator (row 6), which lives in the bottom room, and
row 3 (all wall/pot, no floor) makes the two rooms mutually unreachable by
movement.

agent_1 spawns in the bottom room (rows 4-5), also an open 2x4 rectangle,
which always has the recipe indicator in view, plus both pots (facing up
from row 4), but no useful ingredient supply (only ingredient 3, unused by
any recipe), no plate pile and no delivery station.

So agent_1 is the only seat that ever sees the target recipe, and agent_0 is
the only seat that can actually cook and deliver it. Since there is no
explicit message channel, agent_1 signals the target ingredient by standing
below the pot associated with that ingredient (column 1 for ingredient 0,
column 3 for ingredient 1 -- the ordinal id-to-column mapping); agent_0 reads
its partner's room-relative column from its own local observation to pick a
target each cook cycle. Column 1 reads as a clear signal for ingredient 0;
columns 3 or 4 read as a clear signal for ingredient 1 (broadened past our
own agent_1's exact column 3 to also catch a partner using a related but not
identical right-pot convention); anything else (partner absent, or column 2,
the tied/ambiguous spawn column) falls back to an unbiased coin flip. Because
a wrong delivery never
resamples the recipe, a partner whose reported column is uninformative but
happens to be static would otherwise look like a permanently confident
signal and cause a compounding run of wrong deliveries; agent_0 guards
against this with a Bayesian trust decay that treats a signal repeating the
same value cycle after cycle as increasing evidence of exactly that failure
mode, and decays trust in it toward the unbiased coin flip.

Both seats localize themselves every step from a single always-visible
landmark (the delivery/goal cell for agent_0, the recipe indicator for
agent_1) via simple arithmetic, and navigate with straight Manhattan moves
(each room is an open rectangle, so no pathfinding table is needed). This
means no positional memory is required and behaviour self-corrects after
every reset.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'cook_and_signal',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

# --------------------------------------------------------------------------
# Movement / facing ids (per spec).
# --------------------------------------------------------------------------
_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5
_F_UP, _F_DOWN, _F_RIGHT, _F_LEFT = 0, 1, 2, 3

# --------------------------------------------------------------------------
# Observation channel layout constants (see environment/README.md).
# --------------------------------------------------------------------------
_C_SELF_FACING = 1     # channels 1..4 = self facing UP,DOWN,RIGHT,LEFT
_C_SELF_PLATE = 5
_C_SELF_COOKED = 6
_C_SELF_ING0 = 7
_C_SELF_ING1 = 8
_C_PARTNER_POS = 11
_C_GOAL = 23
_C_RECIPE_INDICATOR = 25
_C_ITEM_START = 32     # plate, cooked, ing0, ing1, ing2, ing3 -> item ing0 = 34, ing1 = 35
_C_RECIPE_START = 38   # plate, cooked, ing0, ing1, ing2, ing3 -> recipe ing0 = 40, ing1 = 41

# Known global coordinates used as always-visible landmarks / targets.
_GOAL_ROW, _GOAL_COL = 0, 2
_RECIPE_ROW, _RECIPE_COL = 6, 2


def _facing_to_action(desired_facing):
    return jnp.where(
        desired_facing == _F_UP, _UP,
        jnp.where(desired_facing == _F_DOWN, _DOWN,
                  jnp.where(desired_facing == _F_RIGHT, _RIGHT, _LEFT)))


def _locate_via_landmark(obs, channel, target_row, target_col):
    """Self (row, col) from a channel with exactly one always-visible 1."""
    layer = obs[:, :, channel]
    flat_idx = jnp.argmax(layer.reshape(-1))
    li = flat_idx // 5
    lj = flat_idx % 5
    self_row = target_row - li + 2
    self_col = target_col - lj + 2
    return self_row, self_col, li, lj


def _nav_and_interact(self_row, self_col, target_row, target_col,
                       desired_facing, facing, arrived_action, require_facing=True):
    """One open-rectangle room: Manhattan move (row first) then face then act.

    ``require_facing`` must only be True when ``desired_facing`` points at a
    cell that blocks passage (a supply/pot/wall/plate-pile/delivery cell):
    the "turn in place" step is implemented by issuing the corresponding
    movement action, which would otherwise actually walk the agent onto an
    open floor cell instead of merely turning to face it.
    """
    row_diff = target_row - self_row
    col_diff = target_col - self_col
    moving = (row_diff != 0) | (col_diff != 0)
    move_action = jnp.where(
        row_diff > 0, _DOWN,
        jnp.where(row_diff < 0, _UP,
                  jnp.where(col_diff > 0, _RIGHT, _LEFT)))
    if require_facing:
        face_ok = facing == desired_facing
        turn_action = _facing_to_action(desired_facing)
        action = jnp.where(moving, move_action, jnp.where(face_ok, arrived_action, turn_action))
        arrived = jnp.logical_not(moving) & face_ok
    else:
        action = jnp.where(moving, move_action, arrived_action)
        arrived = jnp.logical_not(moving)
    return action.astype(jnp.int32), arrived


# ==========================================================================
# agent_0: the only seat that can cook/plate/deliver. Blind to the recipe;
# infers the target ingredient from agent_1's observed position each cycle.
# ==========================================================================

_S_DECIDE = 0
_S_FETCH = 1
_S_PUT_POT = 2
_S_GET_PLATE = 3
_S_COLLECT = 4
_S_DELIVER = 5

# Target-side-independent waypoints (row, col) in the top room.
_ING0_R, _ING0_C = 1, 1   # facing LEFT -> ingredient-0 supply (1,0)
_ING1_R, _ING1_C = 2, 1   # facing LEFT -> ingredient-1 supply (2,0)
_POT0_R, _POT0_C = 2, 1   # facing DOWN -> pot (3,1)
_POT1_R, _POT1_C = 2, 3   # facing DOWN -> pot (3,3)
_PLATE_R, _PLATE_C = 1, 3  # facing UP -> plate pile (0,3)
_DELIVER_R, _DELIVER_C = 1, 2  # facing UP -> delivery station (0,2)
_VANTAGE_R, _VANTAGE_C = 2, 2  # sees partner room fully


def init_agent_0():
    return {
        "state": jnp.asarray(_S_DECIDE, dtype=jnp.int32),
        "target_side": jnp.asarray(0, dtype=jnp.int32),
        "same_streak": jnp.asarray(0, dtype=jnp.int32),
        "last_clear_side": jnp.asarray(-1, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    self_row, self_col, _li, _lj = _locate_via_landmark(obs, _C_GOAL, _GOAL_ROW, _GOAL_COL)
    facing = jnp.argmax(obs[2, 2, _C_SELF_FACING:_C_SELF_FACING + 4]).astype(jnp.int32)

    state = memory["state"]
    target_side = memory["target_side"]

    fetch_r = jnp.where(target_side == 0, _ING0_R, _ING1_R)
    fetch_c = jnp.where(target_side == 0, _ING0_C, _ING1_C)
    pot_r = jnp.where(target_side == 0, _POT0_R, _POT1_R)
    pot_c = jnp.where(target_side == 0, _POT0_C, _POT1_C)

    self_ing_target = jnp.where(target_side == 0, obs[2, 2, _C_SELF_ING0], obs[2, 2, _C_SELF_ING1])
    self_plate = obs[2, 2, _C_SELF_PLATE]
    self_cooked = obs[2, 2, _C_SELF_COOKED]
    # The pot is always directly below the top-of-pot waypoint (row+1, same col).
    pot_ing_count = obs[3, 2, _C_ITEM_START + 2 + target_side]

    # ---- DECIDE: go to vantage point, read partner column, pick a side ----
    # No facing requirement: the vantage cell is only for observing, not
    # interacting, and its neighbours are open floor (turning would walk).
    decide_action, decide_arrived = _nav_and_interact(
        self_row, self_col, _VANTAGE_R, _VANTAGE_C, _F_UP, facing, _STAY, require_facing=False)

    partner_layer = obs[:, :, _C_PARTNER_POS]
    partner_present = jnp.sum(partner_layer) > 0
    flat_idx = jnp.argmax(partner_layer.reshape(-1))
    p_j = flat_idx % 5
    partner_col = self_col + p_j - 2
    # Only the exact columns our own agent_1 ever settles on (1 or 3) count as
    # a clear signal; anything else (including an idle partner parked at its
    # spawn column, or column 4 which our protocol never uses) falls back to
    # an unbiased coin flip rather than a fixed default. Any deterministic
    # default would, against an uninformative partner, become a fixed guess
    # that is wrong half the time, and since wrong deliveries never resample
    # the recipe, a biased default would compound into long negative streaks
    # instead of averaging out.
    # Column 4 (the room's rightmost column) is strictly closer to the right
    # pot (column 3) than the left pot (column 1) and is never the ambiguous
    # spawn column (2, tied between both pots), so it is treated as an
    # additional clear reading for side 1. This captures partners whose
    # convention places them near, but not exactly on, the right pot column.
    clear_signal = partner_present & ((partner_col == 1) | (partner_col == 3) | (partner_col == 4))
    signal_side = jnp.where(partner_col == 1, 0, 1)

    # A partner that keeps reporting the *same* clear side cycle after cycle
    # is, from agent_0's blind perspective, indistinguishable from "we keep
    # guessing wrong and the frozen (never-resampled) recipe never changes"
    # (that lock-in is guaranteed to reproduce an unchanging signal if our
    # partner is genuinely recipe-tracking) versus "we keep guessing right
    # and recipe resampling just happened to repeat the same value" (which
    # only has probability 1/2 per extra repeat). So each extra repeat of an
    # identical clear reading is Bayesian evidence of being stuck, and we
    # progressively discount trust in the signal as the streak grows,
    # decaying toward an unbiased coin flip. This also protects against a
    # non-cooperating partner whose position is clear but simply static and
    # therefore uncorrelated with the recipe.
    same_streak = memory["same_streak"]
    last_clear_side = memory["last_clear_side"]
    repeats = jnp.where(
        clear_signal,
        jnp.where(signal_side == last_clear_side, jnp.minimum(same_streak + 1, 7), 1),
        same_streak)
    new_last_clear_side = jnp.where(clear_signal, signal_side, last_clear_side)
    # Prior odds 15:1 in favour of trusting a first-time clear reading: a
    # sweep over 6/9/15/25/60 against synthetic silent, random, inverted, and
    # shifted-convention partners (see the ledger) found the population
    # expected value roughly flat across 15-60 and clearly worse below 9;
    # 15 keeps most of that plateau's self-play/informative-partner upside
    # while bounding the downside against a static/adversarial partner more
    # tightly than the higher end of the plateau.
    trust_prob = 15.0 / (15.0 + jnp.exp2((repeats - 1).astype(jnp.float32)))

    rng_signal, rng_explore = jax.random.split(rng)
    trust_roll = jax.random.uniform(rng_explore) < trust_prob
    random_side = jax.random.randint(rng_signal, (), 0, 2)
    trust_signal = clear_signal & trust_roll
    chosen_side = jnp.where(trust_signal, signal_side, random_side)

    new_target_side = jnp.where(decide_arrived, chosen_side, target_side)
    new_same_streak = jnp.where(decide_arrived, repeats, same_streak)
    new_last_clear_side = jnp.where(decide_arrived, new_last_clear_side, last_clear_side)
    state_after_decide = jnp.where(decide_arrived, _S_FETCH, _S_DECIDE)

    # ---- FETCH: get one unit of the target ingredient ----
    fetch_action, _ = _nav_and_interact(self_row, self_col, fetch_r, fetch_c, _F_LEFT, facing, _INTERACT)
    have_ingredient = self_ing_target >= 1
    state_after_fetch = jnp.where(have_ingredient, _S_PUT_POT, _S_FETCH)

    # ---- PUT_POT: drop the ingredient into the correct pot ----
    put_action, _ = _nav_and_interact(self_row, self_col, pot_r, pot_c, _F_DOWN, facing, _INTERACT)
    dropped = self_ing_target == 0
    pot_full = pot_ing_count >= 3
    state_after_put = jnp.where(dropped, jnp.where(pot_full, _S_GET_PLATE, _S_FETCH), _S_PUT_POT)

    # ---- GET_PLATE ----
    plate_action, _ = _nav_and_interact(self_row, self_col, _PLATE_R, _PLATE_C, _F_UP, facing, _INTERACT)
    state_after_plate = jnp.where(self_plate >= 1, _S_COLLECT, _S_GET_PLATE)

    # ---- COLLECT: wait at the pot until the dish is plated ----
    collect_action, _ = _nav_and_interact(self_row, self_col, pot_r, pot_c, _F_DOWN, facing, _INTERACT)
    state_after_collect = jnp.where(self_cooked >= 1, _S_DELIVER, _S_COLLECT)

    # ---- DELIVER ----
    deliver_action, _ = _nav_and_interact(self_row, self_col, _DELIVER_R, _DELIVER_C, _F_UP, facing, _INTERACT)
    state_after_deliver = jnp.where(self_cooked == 0, _S_DECIDE, _S_DELIVER)

    action = jax.lax.switch(
        state,
        [lambda: decide_action, lambda: fetch_action, lambda: put_action,
         lambda: plate_action, lambda: collect_action, lambda: deliver_action],
    )
    next_state = jax.lax.switch(
        state,
        [lambda: state_after_decide, lambda: state_after_fetch, lambda: state_after_put,
         lambda: state_after_plate, lambda: state_after_collect, lambda: state_after_deliver],
    )
    next_target_side = jnp.where(state == _S_DECIDE, new_target_side, target_side)
    next_same_streak = jnp.where(state == _S_DECIDE, new_same_streak, same_streak)
    next_last_clear_side = jnp.where(state == _S_DECIDE, new_last_clear_side, last_clear_side)

    init = init_agent_0()
    next_memory = {
        "state": jnp.where(reset, init["state"], next_state.astype(jnp.int32)),
        "target_side": jnp.where(reset, init["target_side"], next_target_side.astype(jnp.int32)),
        "same_streak": jnp.where(reset, init["same_streak"], next_same_streak.astype(jnp.int32)),
        "last_clear_side": jnp.where(
            reset, init["last_clear_side"], next_last_clear_side.astype(jnp.int32)),
    }
    return action.astype(jnp.int32), next_memory, {}


# ==========================================================================
# agent_1: always sees the recipe indicator but has no way to cook it.
# Signals the target ingredient by standing below the associated pot.
# ==========================================================================

_POT0_BOT_R, _POT0_BOT_C = 4, 1  # facing UP -> pot (3,1)
_POT1_BOT_R, _POT1_BOT_C = 4, 3  # facing UP -> pot (3,3)


def init_agent_1():
    return {"unused": jnp.asarray(0, dtype=jnp.int32)}


def agent_1_act(obs, reset, memory, rng):
    self_row, self_col, li, lj = _locate_via_landmark(
        obs, _C_RECIPE_INDICATOR, _RECIPE_ROW, _RECIPE_COL)
    facing = jnp.argmax(obs[2, 2, _C_SELF_FACING:_C_SELF_FACING + 4]).astype(jnp.int32)

    recipe_ing0 = obs[li, lj, _C_RECIPE_START + 2]
    recipe_ing1 = obs[li, lj, _C_RECIPE_START + 3]
    target_side = jnp.where(recipe_ing0 > 0, 0, jnp.where(recipe_ing1 > 0, 1, 0))

    dest_r = jnp.where(target_side == 0, _POT0_BOT_R, _POT1_BOT_R)
    dest_c = jnp.where(target_side == 0, _POT0_BOT_C, _POT1_BOT_C)
    # No facing requirement: standing here is the whole signal, no interact.
    action, _ = _nav_and_interact(
        self_row, self_col, dest_r, dest_c, _F_UP, facing, _STAY, require_facing=False)

    next_memory = {"unused": jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory["unused"])}
    return action.astype(jnp.int32), next_memory, {}
