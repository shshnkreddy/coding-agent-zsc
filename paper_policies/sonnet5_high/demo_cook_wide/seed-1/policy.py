"""Cooperative Overcooked-V2 (demo_cook_wide) policy.

Map facts baked into this policy (see context/environment/README.md and
specification.json for derivation):

- agent_0 is confined for its whole life to the 3-cell corridor (1,4)-(1,6).
  That corridor alone gives adjacent access to both ingredient supplies used
  by any recipe (ids 0 and 1), both plate piles, the delivery station and the
  pot (facing down from (1,5)). agent_0 can therefore cook and deliver solo,
  but the always-on recipe indicator at (4,5) is never inside agent_0's crop
  (max visible row from (1,*) is row 3), so agent_0 can never read the target
  recipe directly.
- agent_1 lives in the disconnected south room (row 3-4) and can see the
  recipe indicator from (3,5) (and can also reach the pot from below at
  (3,5)), but has no access to a plate pile or the delivery station, so it
  can never finish a dish by itself.

Because only two recipes exist (3x ingredient 0, or 3x ingredient 1), the
simplest convention-free coordination signal is the *physical pot contents*:
whichever raw ingredient type is already sitting in the pot should be topped
up to 3 units by whoever gets there, without needing to agree on any private
protocol. agent_1 additionally has privileged recipe-indicator access, so it
seeds the pot with the right type; agent_0 reactively finishes whatever type
it finds (or guesses, after a short wait, if the pot is still empty) so that
throughput does not collapse to zero even if paired with a partner that does
not cooperate the same way.

XP robustness notes (from cross-play testing against independently-coded
stand-in partners, not just self-play): agent_1's confirm/hold windows
before departing/inserting are kept short (CONFIRM_THRESHOLD=2,
HOLD_CONFIRM_THRESHOLD=1) because a long wait mainly widens the window in
which an unknown partner's own agent_0 can self-guess and seed the pot
before this agent arrives; the discard-on-mismatch guard also checks pot
contents (not just its own held item) against the live recipe, so it
refuses to compound contamination left by a partner's own guess. agent_0's
WAIT_THRESHOLD=50 solo-guess fallback was empirically confirmed necessary
at that length: shortening it caused agent_0 to occasionally race ahead of
even its own (slower) agent_1 in self-play, which is strictly worse.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

# Action ids (see specification.json "actions").
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

# Facing one-hot channel indices at a player's own cell.
FACE_UP_CH = 1
FACE_DOWN_CH = 2
FACE_RIGHT_CH = 3
FACE_LEFT_CH = 4

WAIT_THRESHOLD = 50

# Static-object type codes per board cell, matching (channel_index - 21) for
# channels 22..31 (wall=1, goal=2, pot=3, recipe_indicator=4, button=5,
# plate_pile=6, ingredient0..3=7..10); 0 = no static object (floor/off-map).
# Derived directly from specification.json's ascii_grid / object_positions.
_FULL_MAP = [
    [1, 1, 1, 1, 6, 2, 6, 1, 1, 1, 1],
    [1, 1, 1, 7, 0, 0, 0, 8, 1, 1, 1],
    [1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [7, 0, 0, 1, 10, 4, 10, 1, 0, 0, 7],
    [1, 8, 1, 1, 1, 1, 1, 1, 1, 8, 1],
]
_PAD = [[0] * 15, [0] * 15]
_PADDED_MAP = _PAD + [[0, 0] + row + [0, 0] for row in _FULL_MAP] + _PAD

# agent_1's reachable region (row 3 cols 1-9, plus the two south spurs).
_AGENT_1_CANDIDATES = [
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 1), (4, 2), (4, 8), (4, 9),
]


# --------------------------------------------------------------------------
# agent_0: reactive controller confined to corridor columns {4,5,6} (idx 0,1,2)
# --------------------------------------------------------------------------

def init_agent_0():
    return {
        'wait_counter': jnp.asarray(0, dtype=jnp.int32),
        'park_alt': jnp.asarray(0, dtype=jnp.int32),
        'guess_type': jnp.asarray(-1, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    coin = jax.random.bernoulli(rng).astype(jnp.int32)
    memory = {
        'wait_counter': jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['wait_counter']),
        'park_alt': jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['park_alt']),
        'guess_type': jnp.where(reset, jnp.asarray(-1, dtype=jnp.int32), memory['guess_type']),
    }

    # Locate self within the corridor via the pot's local column (row index 3
    # is constant because agent_0 never leaves row 1).
    local_col_pot = jnp.argmax(obs[3, :, 24]).astype(jnp.int32)
    col_idx = 3 - local_col_pot  # 0 -> col4, 1 -> col5, 2 -> col6

    pot_cooked = obs[3, local_col_pot, 33]
    pot_timer = obs[3, local_col_pot, 44]
    pot_ingr0 = obs[3, local_col_pot, 34]
    pot_ingr1 = obs[3, local_col_pot, 35]
    pot_ingr2 = obs[3, local_col_pot, 36]
    pot_ingr3 = obs[3, local_col_pot, 37]
    pot_sum = pot_ingr0 + pot_ingr1 + pot_ingr2 + pot_ingr3
    pot_needs = (pot_timer == 0) & (pot_cooked == 0) & (pot_sum < 3)
    pot_ready = pot_cooked == 1
    pot_cooking_or_ready = (pot_timer > 0) | pot_ready

    held_plate = obs[2, 2, 5]
    held_cooked = obs[2, 2, 6]
    held_ingr0 = obs[2, 2, 7]
    held_ingr1 = obs[2, 2, 8]

    holding_cooked = held_cooked == 1
    holding_plate_only = (held_plate == 1) & (held_cooked == 0)
    holding_ingr = (held_ingr0 == 1) | (held_ingr1 == 1)

    facing_up = obs[2, 2, FACE_UP_CH] == 1
    facing_down = obs[2, 2, FACE_DOWN_CH] == 1
    facing_left = obs[2, 2, FACE_LEFT_CH] == 1
    facing_right = obs[2, 2, FACE_RIGHT_CH] == 1

    existing0 = pot_ingr0 > 0
    existing1 = pot_ingr1 > 0
    pot_empty_of_01 = jnp.logical_not(existing0 | existing1)

    def nav_vertical(target_col, want_face_down, on_arrival_action):
        at_target = col_idx == target_col
        facing_ok = jnp.where(want_face_down, facing_down, facing_up)
        turn = jnp.where(want_face_down, DOWN, UP)
        move = jnp.where(col_idx < target_col, RIGHT, LEFT)
        at_target_action = jnp.where(facing_ok, on_arrival_action, turn)
        return jnp.where(at_target, at_target_action, move)

    # Case A: holding the finished plated dish -> deliver at col_idx 1, face up.
    action_a = nav_vertical(1, False, INTERACT)

    # Case B: holding an empty plate -> go to col_idx 1, face down, scoop once ready.
    scoop_action = jnp.where(pot_ready, INTERACT, STAY)
    action_b = nav_vertical(1, True, scoop_action)

    # Case C: holding a raw ingredient -> insert if the pot still needs it,
    # otherwise park it on a counter so hands are free for plate/scoop duty.
    action_c_insert = nav_vertical(1, True, INTERACT)
    park_target = jnp.where(memory['park_alt'] == 0, 0, 2)
    action_c_park = nav_vertical(park_target, True, INTERACT)
    action_c = jnp.where(pot_needs, action_c_insert, action_c_park)
    did_park_this_step = (
        holding_ingr & jnp.logical_not(pot_needs)
        & (col_idx == park_target) & facing_down
    )

    # Case D: empty-handed.
    target_plate_col = jnp.where(col_idx == 2, 2, 0)
    action_d1_fetch_plate = nav_vertical(target_plate_col, False, INTERACT)

    in_wait_context = pot_needs & pot_empty_of_01
    should_wait = in_wait_context & (memory['wait_counter'] < WAIT_THRESHOLD)
    # Lock the coin-flip guess into memory the moment we commit to fetching
    # (rather than re-drawing a fresh coin every step): re-sampling a random
    # target column each step would make the walk toward the supply
    # incoherently flip-flop direction every step and never actually arrive.
    locked_guess = jnp.where(memory['guess_type'] >= 0, memory['guess_type'], coin)
    target_type = jnp.where(existing0, 0, jnp.where(existing1, 1, locked_guess))
    target_ingr_col = jnp.where(target_type == 0, 0, 2)
    facing_needed = jnp.where(target_type == 0, facing_left, facing_right)
    turn_needed = jnp.where(target_type == 0, LEFT, RIGHT)
    at_ingr = col_idx == target_ingr_col
    move_ingr = jnp.where(col_idx < target_ingr_col, RIGHT, LEFT)
    action_d2_fetch_ingr = jnp.where(
        at_ingr, jnp.where(facing_needed, INTERACT, turn_needed), move_ingr
    )
    action_d2 = jnp.where(should_wait, STAY, action_d2_fetch_ingr)

    action_d = jnp.where(pot_cooking_or_ready, action_d1_fetch_plate,
                          jnp.where(pot_needs, action_d2, STAY))

    action = jnp.where(
        holding_cooked, action_a,
        jnp.where(
            holding_plate_only, action_b,
            jnp.where(holding_ingr, action_c, action_d)
        ),
    ).astype(jnp.int32)

    next_wait_counter = jnp.where(
        should_wait, memory['wait_counter'] + 1,
        jnp.where(in_wait_context, memory['wait_counter'], jnp.asarray(0, dtype=jnp.int32)),
    ).astype(jnp.int32)
    next_park_alt = jnp.where(
        did_park_this_step, 1 - memory['park_alt'], memory['park_alt']
    ).astype(jnp.int32)
    next_guess_type = jnp.where(
        in_wait_context, locked_guess, jnp.asarray(-1, dtype=jnp.int32)
    ).astype(jnp.int32)

    next_memory = {
        'wait_counter': next_wait_counter, 'park_alt': next_park_alt,
        'guess_type': next_guess_type,
    }
    return action, next_memory, {}


# --------------------------------------------------------------------------
# agent_1: fully reactive controller over the whole south room. Spawn is
# randomized across 13 reachable cells, so position is re-derived every step
# by matching the locally visible static-object footprint against all 13
# candidate cells on the known, fixed map (self-correcting, no dead
# reckoning/drift risk).
# --------------------------------------------------------------------------

# The pot is cleared at scoop time, which happens a couple of steps *before*
# the plated dish actually reaches the delivery station and the recipe
# resamples. Without a buffer, agent_1 can start prepping (and inserting
# into) the next batch using the still-stale pre-resample recipe reading.
# Shortened from an earlier 6: a long wait here only widens the window in
# which an independently generated partner playing agent_0 (unknown
# cadence) can self-guess and seed the pot before this agent arrives; any
# residual staleness is still caught by the discard-on-mismatch logic below,
# which does not depend on this threshold.
CONFIRM_THRESHOLD = 2

# Because interacts resolve in seat order within one timestep, an already
# in-hand ingredient can slip into the pot the very same step agent_0 scoops
# -- one step *before* this agent's own observation would ever show
# pot_needs=True (observations always lag same-step effects by one step).
# Requiring this many *consecutive observed* pot_needs=True reads before
# actually inserting a held ingredient means the insert can never land in
# that same-step race window. Shortened from an earlier 4 for the same
# latency reason as CONFIRM_THRESHOLD above.
HOLD_CONFIRM_THRESHOLD = 1


def init_agent_1():
    return {
        'mode': jnp.asarray(0, dtype=jnp.int32),  # 0=seek pot/recipe at home, 1=seek ingredient
        'target_type': jnp.asarray(0, dtype=jnp.int32),
        'confirm_counter': jnp.asarray(0, dtype=jnp.int32),
        'hold_confirm': jnp.asarray(0, dtype=jnp.int32),
    }


def _localize_agent_1(obs):
    padded_map = jnp.array(_PADDED_MAP, dtype=jnp.int32)
    seg = obs[:, :, 22:32]
    seg_sum = jnp.sum(seg, axis=-1)
    seg_argmax = jnp.argmax(seg, axis=-1)
    obs_type = jnp.where(seg_sum > 0, seg_argmax + 1, 0).astype(jnp.int32)

    crops = jnp.stack(
        [padded_map[r:r + 5, c:c + 5] for (r, c) in _AGENT_1_CANDIDATES]
    )
    mismatch = jnp.sum((crops != obs_type[None]).astype(jnp.int32), axis=(1, 2))
    best = jnp.argmin(mismatch)
    candidates_arr = jnp.asarray(_AGENT_1_CANDIDATES, dtype=jnp.int32)
    return candidates_arr[best, 0], candidates_arr[best, 1]


def agent_1_act(obs, reset, memory, rng):
    del rng
    memory = {
        'mode': jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['mode']),
        'target_type': jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['target_type']),
        'confirm_counter': jnp.where(
            reset, jnp.asarray(0, dtype=jnp.int32), memory['confirm_counter']
        ),
        'hold_confirm': jnp.where(
            reset, jnp.asarray(0, dtype=jnp.int32), memory['hold_confirm']
        ),
    }
    mode = memory['mode']
    target_type = memory['target_type']
    hold_confirm = memory['hold_confirm']
    confirm_counter = memory['confirm_counter']

    self_row, self_col = _localize_agent_1(obs)

    held_ingr0 = obs[2, 2, 7]
    held_ingr1 = obs[2, 2, 8]
    holding_ingr = (held_ingr0 == 1) | (held_ingr1 == 1)
    facing_up = obs[2, 2, FACE_UP_CH] == 1
    facing_down = obs[2, 2, FACE_DOWN_CH] == 1
    facing_left = obs[2, 2, FACE_LEFT_CH] == 1

    # Pot (2,5) and recipe indicator (4,5) are both visible whenever
    # self_col is within [3,7] (both are directly below/at column 5).
    visible = (self_col >= 3) & (self_col <= 7)
    local_row_pot = jnp.clip(4 - self_row, 0, 4)
    local_col_pot = jnp.clip(7 - self_col, 0, 4)
    local_row_recipe = jnp.clip(6 - self_row, 0, 4)
    local_col_recipe = local_col_pot

    pot_cooked = jnp.where(visible, obs[local_row_pot, local_col_pot, 33], 0)
    pot_timer = jnp.where(visible, obs[local_row_pot, local_col_pot, 44], 0)
    pot_ingr0 = jnp.where(visible, obs[local_row_pot, local_col_pot, 34], 0)
    pot_ingr1 = jnp.where(visible, obs[local_row_pot, local_col_pot, 35], 0)
    pot_ingr2 = jnp.where(visible, obs[local_row_pot, local_col_pot, 36], 0)
    pot_ingr3 = jnp.where(visible, obs[local_row_pot, local_col_pot, 37], 0)
    pot_sum = pot_ingr0 + pot_ingr1 + pot_ingr2 + pot_ingr3
    pot_needs = visible & (pot_timer == 0) & (pot_cooked == 0) & (pot_sum < 3)

    recipe_ingr0 = jnp.where(visible, obs[local_row_recipe, local_col_recipe, 40], 0)
    recipe_ingr1 = jnp.where(visible, obs[local_row_recipe, local_col_recipe, 41], 0)
    recipe_target = jnp.where(recipe_ingr0 > 0, 0, jnp.where(recipe_ingr1 > 0, 1, 0))
    # agent_1 has privileged, always-current recipe-indicator access, so its
    # own reading is authoritative and must not be overridden by whatever
    # (possibly uninformed) contents already happen to sit in the pot.
    target_type_candidate = recipe_target

    at_home = (self_row == 3) & (self_col == 5)
    toward_home_move = jnp.where(
        self_row == 4, UP,
        jnp.where(self_col < 5, RIGHT, jnp.where(self_col > 5, LEFT, STAY)),
    )

    # Deliver/insert path: navigate to (3,5), then face up and wait for
    # HOLD_CONFIRM_THRESHOLD consecutive *observed* pot_needs=True reads
    # before inserting (see HOLD_CONFIRM_THRESHOLD comment above).
    ready_to_insert = at_home & facing_up & pot_needs & (hold_confirm >= HOLD_CONFIRM_THRESHOLD)
    deliver_action = jnp.where(
        at_home, jnp.where(facing_up, jnp.where(ready_to_insert, INTERACT, STAY), UP),
        toward_home_move,
    )
    in_hold_confirm_context = at_home & facing_up & pot_needs
    next_hold_confirm = jnp.where(
        in_hold_confirm_context,
        jnp.minimum(hold_confirm + 1, HOLD_CONFIRM_THRESHOLD),
        jnp.asarray(0, dtype=jnp.int32),
    ).astype(jnp.int32)

    # A pot clears at scoop time, a step or two *before* the actual delivery
    # (and thus the recipe resample) happens. Because interacts resolve in
    # seat order within one timestep, a pending held ingredient can slip into
    # a just-cleared pot the instant agent_0 scoops -- before this agent gets
    # any chance to notice a stale commitment. Guard the final insert step by
    # re-checking the held type against the live recipe right at home, and
    # discard (rather than insert) a stale ingredient onto a nearby counter.
    # Checked every step while waiting at home (not gated on pot_needs):
    # pot_needs reflects the observation from *before* this step's actions
    # resolve, so a same-step scoop-then-insert race would otherwise slip a
    # stale ingredient in right as pot_needs first flips true. Continuously
    # discarding a stale holding as soon as the recipe changes, even while
    # still waiting out the cook, closes that window entirely.
    held_type = jnp.where(held_ingr0 == 1, 0, 1)
    is_discarding = mode == 2
    # Besides checking our own held type against the live recipe, also guard
    # against a pot that some *other* seat (e.g. an independently generated
    # agent_0 with its own solo guess-fallback) has already seeded with the
    # opposite ingredient type before we arrived: inserting our own
    # correct-type unit on top would not undo that contamination, so treat
    # it the same as a stale mismatch and discard instead of compounding it.
    pot_type_conflict = jnp.where(
        held_type == 0, pot_ingr1 > 0, pot_ingr0 > 0
    )
    fresh_mismatch = (
        jnp.logical_not(is_discarding) & at_home & visible
        & ((recipe_target != held_type) | pot_type_conflict)
    )
    should_discard = is_discarding | fresh_mismatch

    at_discard_spot = (self_row == 3) & (self_col == 4)
    discard_move = jnp.where(self_col > 4, LEFT, jnp.where(self_col < 4, RIGHT, STAY))
    discard_action = jnp.where(
        at_discard_spot, jnp.where(facing_up, INTERACT, UP), discard_move
    )
    # Only leave discard mode once the interact has actually been attempted
    # while facing up -- resetting as soon as we merely *arrive* would drop
    # back to the at_home-gated mismatch check (false at col 4), causing an
    # infinite ping-pong between home and the discard spot.
    next_mode_from_discard = jnp.where(at_discard_spot & facing_up, 0, 2).astype(jnp.int32)

    holding_action = jnp.where(should_discard, discard_action, deliver_action)

    # Seek-home path (mode 0): walk home, then wait out a short confirmation
    # window (re-reading the recipe every step) before committing to a trip,
    # so a stale pre-resample reading never gets carried into the next batch.
    ready_to_depart = at_home & pot_needs & (confirm_counter >= CONFIRM_THRESHOLD)
    seek_home_action = jnp.where(
        at_home, jnp.where(ready_to_depart, LEFT, STAY), toward_home_move
    )
    next_mode_from_seek = jnp.where(ready_to_depart, 1, 0).astype(jnp.int32)
    next_target_from_seek = jnp.where(
        ready_to_depart, target_type_candidate, target_type
    ).astype(jnp.int32)
    next_confirm_from_seek = jnp.where(
        at_home & pot_needs, jnp.minimum(confirm_counter + 1, CONFIRM_THRESHOLD),
        jnp.asarray(0, dtype=jnp.int32),
    ).astype(jnp.int32)

    # Seek-ingredient path (mode 1): walk to (4,1), then fetch the committed type.
    at_corner = (self_row == 4) & (self_col == 1)
    corner_move = jnp.where(
        self_row == 3,
        jnp.where(self_col == 1, DOWN, jnp.where(self_col > 1, LEFT, RIGHT)),
        STAY,
    )
    facing_needed = jnp.where(target_type == 0, facing_left, facing_down)
    turn_needed = jnp.where(target_type == 0, LEFT, DOWN)
    at_corner_action = jnp.where(facing_needed, INTERACT, turn_needed)
    seek_ingr_action = jnp.where(at_corner, at_corner_action, corner_move)

    action = jnp.where(
        holding_ingr, holding_action,
        jnp.where(mode == 0, seek_home_action, seek_ingr_action),
    ).astype(jnp.int32)

    next_mode = jnp.where(
        holding_ingr,
        jnp.where(should_discard, next_mode_from_discard, jnp.asarray(0, dtype=jnp.int32)),
        jnp.where(mode == 0, next_mode_from_seek, jnp.asarray(1, dtype=jnp.int32)),
    )
    next_target_type = jnp.where(
        holding_ingr, target_type,
        jnp.where(mode == 0, next_target_from_seek, target_type),
    )
    next_confirm_counter = jnp.where(
        holding_ingr, jnp.asarray(0, dtype=jnp.int32),
        jnp.where(mode == 0, next_confirm_from_seek, jnp.asarray(0, dtype=jnp.int32)),
    ).astype(jnp.int32)
    next_hold_confirm = jnp.where(
        holding_ingr, next_hold_confirm, jnp.asarray(0, dtype=jnp.int32)
    ).astype(jnp.int32)

    next_memory = {
        'mode': next_mode, 'target_type': next_target_type,
        'confirm_counter': next_confirm_counter, 'hold_confirm': next_hold_confirm,
    }
    return action, next_memory, {}
