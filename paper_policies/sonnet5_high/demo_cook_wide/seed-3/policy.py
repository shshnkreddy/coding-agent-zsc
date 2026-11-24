"""Geometry-grounded coordination policy for demo_cook_wide.

Design rationale (robust to independently generated partners, XP objective):

- agent_0 spawns in a 3-cell row (cols 4-6 of row 1) with direct access to
  the pot (facing down from col 5), the delivery station and both plate
  piles (facing up), and both ingredient-0/1 supplies (facing left/right at
  the row ends). It can never see the recipe indicator (always > view
  radius away), so it never guesses ingredients; it only fetches a plate,
  collects whatever cooks, and delivers it. This is the best it can do
  given its information constraints and is helpful for *any* competent
  partner who fills the pot, regardless of how that partner decides to do
  so.
- agent_1 spawns in a corridor (row 3, cols 1-9, plus two small alcoves at
  row 4) that includes a direct line of sight to the always-on recipe
  indicator and direct pot access from the opposite side (facing up from
  (3,5)). It reads the recipe from the indicator, fetches matching raw
  ingredient units from whichever supply corner is closer, and deposits
  them into the pot, gated on the pot actually being able to accept more
  (idle and under capacity) to avoid depositing a stale ingredient into a
  freshly resampled recipe.

Neither seat depends on any private signalling convention: both act purely
on affordances and state that are determined by the fixed, known map
geometry and by locally observable game state (own inventory, the pot's
visible contents, the recipe indicator), so the same logic is reasonable
for any competent, independently generated partner occupying the
complementary seat.

Self-localization (since the environment does not expose global
coordinates) is done by exact template matching of the local 5x5 static
channel crop (walls/objects/supplies, channels 22-31, which never change
during an episode) against the one known, fixed global map.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'geometry_coordinator',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 6,
    'max_width': 11,
    'num_ingredients': 4,
}

# ---------------------------------------------------------------------------
# Static map knowledge (fixed, known before the episode; see environment/
# README.md and specification.json). Channel order matches obs[:, :, 22:32]:
# wall, goal(delivery), pot, recipe_indicator, button_indicator, plate_pile,
# ingredient0, ingredient1, ingredient2, ingredient3.
# ---------------------------------------------------------------------------

_ROWS, _COLS, _STATIC_CH = 6, 11, 10

_WALLS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10),
    (1, 0), (1, 1), (1, 2), (1, 8), (1, 9), (1, 10),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
    (3, 0), (3, 10),
    (4, 3), (4, 7),
    (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10),
)
_GOAL = ((0, 5),)
_POT = ((2, 5),)
_RECIPE = ((4, 5),)
_PLATE = ((0, 4), (0, 6))
_ING0 = ((1, 3), (4, 0), (4, 10))
_ING1 = ((1, 7), (5, 1), (5, 9))
_ING3 = ((4, 4), (4, 6))

# Action ids (specification.json): RIGHT=0, DOWN=1, LEFT=2, UP=3, STAY=4,
# INTERACT=5. Facing one-hot channel order (README): 1=UP, 2=DOWN, 3=RIGHT,
# 4=LEFT. Plain Python ints at module scope (no calls allowed there); cast
# to jnp inside functions where used.
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


def _build_static_windows():
    """Builds, for every one of the 66 global cells, the 5x5x10 static
    channel crop a self-centered observation would show if self stood
    there. All-JAX, no numpy, built fresh inside a function body (no
    module-scope calls are permitted)."""
    m = jnp.zeros((_ROWS, _COLS, _STATIC_CH), dtype=jnp.int32)
    for (r, c) in _WALLS:
        m = m.at[r, c, 0].set(1)
    for (r, c) in _GOAL:
        m = m.at[r, c, 1].set(1)
    for (r, c) in _POT:
        m = m.at[r, c, 2].set(1)
    for (r, c) in _RECIPE:
        m = m.at[r, c, 3].set(1)
    for (r, c) in _PLATE:
        m = m.at[r, c, 5].set(1)
    for (r, c) in _ING0:
        m = m.at[r, c, 6].set(1)
    for (r, c) in _ING1:
        m = m.at[r, c, 7].set(1)
    for (r, c) in _ING3:
        m = m.at[r, c, 9].set(1)

    padded = jnp.zeros((_ROWS + 4, _COLS + 4, _STATIC_CH), dtype=jnp.int32)
    padded = padded.at[2:2 + _ROWS, 2:2 + _COLS, :].set(m)

    windows = jnp.zeros((_ROWS * _COLS, 5, 5, _STATIC_CH), dtype=jnp.int32)
    for r in range(_ROWS):
        for c in range(_COLS):
            idx = r * _COLS + c
            windows = windows.at[idx].set(padded[r:r + 5, c:c + 5, :])
    return windows


# facing one-hot index produced by each directional action.
def _action_to_facing_idx():
    return jnp.array([2, 1, 3, 0], dtype=jnp.int32)


def _localize(obs):
    windows = _build_static_windows()
    obs_static = obs[:, :, 22:32].astype(jnp.int32)
    diff = windows - obs_static[None, :, :, :]
    score = jnp.sum(diff * diff, axis=(1, 2, 3))
    idx = jnp.argmin(score).astype(jnp.int32)
    self_row = idx // _COLS
    self_col = idx % _COLS
    return self_row, self_col


def _move_towards(self_row, self_col, target_row, target_col):
    """Greedy nav respecting the corridor topology (row3 open corridor,
    row4 reachable only via cols {1,2,8,9})."""
    want_up_from_row4 = self_row == 4
    col_mismatch = self_col != target_col
    go_right = target_col > self_col
    row3_action = jnp.where(
        col_mismatch,
        jnp.where(go_right, RIGHT, LEFT),
        jnp.where(target_row == 4, DOWN, STAY),
    )
    return jnp.where(want_up_from_row4, UP, row3_action)


def _face_and_interact(obs, desired_action):
    current_facing_idx = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    desired_facing_idx = _action_to_facing_idx()[desired_action]
    return jnp.where(current_facing_idx == desired_facing_idx, INTERACT, desired_action)


# ---------------------------------------------------------------------------
# agent_0: cook/deliver seat, blind to the recipe indicator.
# ---------------------------------------------------------------------------

def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    self_row, self_col = _localize(obs)

    plate = obs[2, 2, 5]
    cooked = obs[2, 2, 6]
    ing0 = obs[2, 2, 7]
    ing1 = obs[2, 2, 8]
    holding_nothing = (plate == 0) & (cooked == 0) & (ing0 == 0) & (ing1 == 0)
    holding_plate_only = (plate == 1) & (cooked == 0)
    # else: holding a plated cooked dish (cooked == 1).

    dist4 = jnp.abs(self_col - 4)
    dist6 = jnp.abs(self_col - 6)
    plate_col = jnp.where(dist4 <= dist6, 4, 6)

    target_row = jnp.asarray(1, dtype=jnp.int32)
    target_col = jnp.where(holding_nothing, plate_col, jnp.int32(5))

    desired_face = jnp.where(
        holding_nothing, UP,
        jnp.where(holding_plate_only, DOWN, UP),
    )

    at_target = (self_row == target_row) & (self_col == target_col)
    nav_action = _move_towards(self_row, self_col, target_row, target_col)
    interact_action = _face_and_interact(obs, desired_face)
    action = jnp.where(at_target, interact_action, nav_action).astype(jnp.int32)

    next_memory = jnp.asarray(0, dtype=jnp.int32)
    return action, next_memory, {}


# ---------------------------------------------------------------------------
# agent_1: recipe-visible seat, ingredient runner.
# Memory layout: [target_ingredient (0/1), recipe_known (0/1),
#                 empty_pot_wait_streak (0..EMPTY_WAIT_THRESHOLD)].
# ---------------------------------------------------------------------------

EMPTY_WAIT_THRESHOLD = 4


def init_agent_1():
    return jnp.array([0, 0, 0], dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    target_prev = memory[0]
    known_prev = memory[1]
    wait_prev = memory[2]
    self_row, self_col = _localize(obs)
    at_hub = (self_row == 3) & (self_col == 5)

    # Hub cell (3,5): recipe indicator is at fixed local offset (3,2) and
    # the pot at fixed local offset (1,2) whenever we are exactly at hub.
    recipe_disp0 = obs[3, 2, 40]
    recipe_disp1 = obs[3, 2, 41]
    new_target = jnp.where(at_hub, jnp.where(recipe_disp0 > 0, 0, 1), target_prev)
    new_known = jnp.where(at_hub, jnp.int32(1), known_prev)

    pot_timer = obs[1, 2, 44]
    pot_cooked = obs[1, 2, 33]
    pot_total_items = obs[1, 2, 34] + obs[1, 2, 35] + obs[1, 2, 36] + obs[1, 2, 37]
    pot_ready_uncollected = pot_cooked == 1
    pot_idle_empty = (pot_timer == 0) & (pot_cooked == 0) & (pot_total_items == 0)
    # Interactions resolve agent_0 first, then agent_1, within the same
    # step. If agent_1 is camped at the hub already holding a unit while
    # the pot still shows cooked/ready, and agent_0's collect happens that
    # same step, agent_1's interact would deposit into the freshly-emptied
    # pot within that very same step -- before the delivery (and any
    # recipe resample) that follows a step or two later. That first unit
    # would then use the *old* target while units 2-3 (fetched afterward)
    # use the resampled *new* target, corrupting the dish. So: never
    # interact with the pot while it still shows cooked/ready (just wait);
    # only once the pot is observed genuinely idle-and-empty (i.e. already
    # collected) do we start a short confirmation streak before committing
    # the first unit of the next batch, giving the following delivery's
    # resample time to resolve. Once the pot holds >=1 item no further
    # resample can occur until this batch is delivered, so later units
    # deposit immediately.
    new_wait = jnp.where(
        at_hub,
        jnp.where(pot_idle_empty, jnp.minimum(wait_prev + 1, EMPTY_WAIT_THRESHOLD), jnp.int32(0)),
        wait_prev,
    )
    deposit_ready = (~pot_ready_uncollected) & (
        (pot_total_items > 0) | (new_wait >= EMPTY_WAIT_THRESHOLD)
    )

    ing0 = obs[2, 2, 7]
    ing1 = obs[2, 2, 8]
    holding_ingredient = (ing0 == 1) | (ing1 == 1)
    held_ingredient_type = jnp.where(ing0 == 1, 0, 1)
    # A held raw ingredient was fetched under a possibly-stale target: the
    # recipe can resample (after a correct delivery elsewhere) while this
    # unit is still in transit to the pot. Depositing a stale unit would
    # corrupt the next batch's composition and cause a wrong (-20) dish, so
    # such units must be discarded onto a counter instead of deposited.
    stale_ingredient = holding_ingredient & (new_known == 1) & (held_ingredient_type != new_target)

    left_corner_closer = self_col <= 5
    fetch_col = jnp.where(left_corner_closer, jnp.int32(1), jnp.int32(9))
    fetch_row = jnp.asarray(4, dtype=jnp.int32)
    fetch_face_ing0 = jnp.where(left_corner_closer, LEFT, RIGHT)
    fetch_face = jnp.where(new_target == 0, fetch_face_ing0, DOWN)

    mode_drop = stale_ingredient
    mode_deposit = holding_ingredient & (~stale_ingredient)
    # Fetch proactively even while the pot is still cooking a prior batch:
    # only one unit can ever be held at a time, and any unit that goes
    # stale by the time it could be deposited is caught by mode_drop above,
    # so pre-fetching cannot corrupt a delivery; it only removes idle time
    # from the critical path between cook-finish and the next deposit.
    mode_fetch = (~holding_ingredient) & (new_known == 1)

    # Drop target: (3,3) facing DOWN reaches the (4,3) counter, a cell on
    # the normal path between the hub and the left fetch corner.
    target_row = jnp.where(
        mode_drop, jnp.int32(3),
        jnp.where(mode_deposit, jnp.int32(3),
                  jnp.where(mode_fetch, fetch_row, jnp.int32(3))),
    )
    target_col = jnp.where(
        mode_drop, jnp.int32(3),
        jnp.where(mode_deposit, jnp.int32(5),
                  jnp.where(mode_fetch, fetch_col, jnp.int32(5))),
    )
    desired_face = jnp.where(
        mode_drop, DOWN,
        jnp.where(mode_deposit, UP,
                  jnp.where(mode_fetch, fetch_face, UP)),
    )
    should_interact_mode = mode_drop | (mode_deposit & deposit_ready) | mode_fetch

    at_target = (self_row == target_row) & (self_col == target_col)
    nav_action = _move_towards(self_row, self_col, target_row, target_col)
    interact_action = _face_and_interact(obs, desired_face)
    arrived_action = jnp.where(should_interact_mode, interact_action, STAY)
    action = jnp.where(at_target, arrived_action, nav_action).astype(jnp.int32)

    new_memory = jnp.stack([new_target, new_known, new_wait]).astype(jnp.int32)
    init_mem = jnp.array([0, 0, 0], dtype=jnp.int32)
    next_memory = jnp.where(reset, init_mem, new_memory).astype(jnp.int32)

    return action, next_memory, {}
