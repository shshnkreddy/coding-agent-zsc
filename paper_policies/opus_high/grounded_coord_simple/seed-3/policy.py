"""Grounded-coordination policy for overcooked_v2 `grounded_coord_simple`.

Layout facts the policy is built on (all from context/environment):

    WW2WWWWW
    W  WB  0
    R ALPA X
    W  WB  1
    WW2WWWWW

The two rooms are disconnected and share no counter, so seat 0 can never touch
food and seat 1 can never see the always-on recipe indicator `R` at (2, 0).
The only information channel between them is the recipe button `L` at (2, 3):
seat 0 activates it and its displayed recipe counts appear in observation
channels 36-38 for anybody whose 5x5 crop covers that cell.  That signal is
grounded in the environment's own recipe encoding, not in a private convention,
so it is readable by an independently written partner and our cook can read a
button pressed by an independently written messenger.

Seat 0 (messenger): park at (2, 2) facing the button, read the true recipe off
`R`, and press only when the displayed recipe differs from the last broadcast.
Each press costs 5 reward, so presses are spent only when they carry news; one
extra retry per broadcast covers a partner that missed the nine-step window,
and a partner that never reacts is not charged for more than that one retry.

Seat 1 (cook): read the button whenever it is inside the crop (any cell of
column 5, which is where the pot, plate pile and waiting spot all are), then
run the fetch/cook/plate/deliver loop for the believed ingredient.  A dish that
contradicts the current belief is dumped on a counter rather than delivered,
and a dumped ingredient is picked up again when it becomes wanted.

Seat 1 never assumes the partner speaks.  It waits a while for a first
broadcast, then guesses; because a wrong delivery does not resample the recipe
while a correct one does, an unconfirmed guess should alternate after every
delivery, which is what it does.  Silence from a partner that has broadcast
after a previous delivery is read as "recipe unchanged"; silence from a partner
never heard from twice is distrusted much sooner.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_button_coord',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 8,
    'num_ingredients': 3,
}

# Action ids.
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Direction ids (observation order) and the move action that produces them.
D_UP, D_DOWN, D_RIGHT, D_LEFT = 0, 1, 2, 3
_FACE_ACTION = (A_UP, A_DOWN, A_RIGHT, A_LEFT)

# Channels.
CH_SELF_DIR = 1          # 1..4, one-hot over D_*
CH_SELF_PLATE = 5
CH_SELF_COOKED = 6
CH_SELF_ING = 7          # 7..9
CH_POT = 22
CH_RECIPE_IND = 23
CH_BUTTON = 24
CH_ITEM = 29             # 29..33, plate/cooked/counts on the grid
CH_ITEM_COOKED = 30
CH_ITEM_ING = 31         # 31..33
CH_RECIPE = 34           # 34..38, displayed recipe block
CH_RECIPE_ING = 36       # 36..38
CH_POT_TIMER = 39

# Static geometry (global row, column).
POT_RC = (2, 4)
RECIPE_IND_RC = (2, 0)
BUTTON_CELL = (2, 2)     # seat-0 floor cell adjacent to the button at (2, 3)
POT_CELL = (2, 5)        # seat-1 floor cell adjacent to the pot
PLATE_CELL = (1, 5)      # adjacent to the plate pile at (1, 4)
DELIVER_CELL = (2, 6)    # adjacent to the delivery station at (2, 7)
ING0_CELL = (1, 6)       # adjacent to the ingredient-0 supply at (1, 7)
ING1_CELL = (3, 6)       # adjacent to the ingredient-1 supply at (3, 7)
# (access row, access column, counter row, counter column, facing) for every
# counter the cook can use, preferred order first.
COUNTERS = (
    (1, 5, 0, 5, D_UP),
    (3, 5, 4, 5, D_DOWN),
    (1, 6, 0, 6, D_UP),
    (3, 6, 4, 6, D_DOWN),
)

# Seat-0 tuning.
PRESS_TIMEOUT = 90       # steps of silence before one retry of a broadcast

# Seat-1 tuning.
WAIT_FIRST = 40          # steps to wait for a first broadcast before guessing
WAIT_AFTER_DELIVERY = 4  # steps to wait for a post-delivery update
UNRELIABLE_AFTER = 5     # silent deliveries before distrusting a stale belief
UNRELIABLE_UNCONFIRMED = 3   # same, for a partner never heard from twice

# [last_broadcast, steps_since_press, retry_available, t]
_INIT_0 = (-1, 0, 0, 0)
# [belief, known, checking, wait, misses, t, ever_delivered, unreliable,
#  confirmed, dump_index]
_INIT_1 = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)


def _locate(obs, channel, land_row, land_col, default_row, default_col):
    """Self global position from a unique static landmark in the crop."""
    mask = obs[:, :, channel]
    flat = jnp.argmax(mask.reshape(-1))
    local_row = (flat // 5).astype(jnp.int32)
    local_col = (flat % 5).astype(jnp.int32)
    found = jnp.sum(mask) > 0
    row = jnp.where(found, land_row - local_row + 2, default_row)
    col = jnp.where(found, land_col - local_col + 2, default_col)
    return row.astype(jnp.int32), col.astype(jnp.int32), local_row, local_col


def _masked(obs, mask, channel):
    return jnp.sum(obs[:, :, channel] * mask)


def _counter_state(obs, row, col, belief):
    """Visibility, item load and wanted-ingredient flag for every counter."""
    visible = []
    contents = []
    matching = []
    for _, _, counter_row, counter_col, _ in COUNTERS:
        local_row = counter_row - row + 2
        local_col = counter_col - col + 2
        inside = (local_row >= 0) & (local_row < 5) & (local_col >= 0) & (local_col < 5)
        block = obs[jnp.clip(local_row, 0, 4), jnp.clip(local_col, 0, 4), CH_ITEM:CH_RECIPE]
        total = jnp.where(inside, jnp.sum(block), 0)
        wanted = jnp.take(block[2:], belief)
        visible.append(inside)
        contents.append(total)
        matching.append(inside & (total == 1) & (wanted == 1))
    return jnp.stack(visible), jnp.stack(contents), jnp.stack(matching)


def _facing(obs):
    return obs[2, 2, CH_SELF_DIR:CH_SELF_DIR + 4]


def _navigate(row, col, target_row, target_col, face_dir, facing, at_action):
    """Vertical-then-horizontal walk; turn on arrival, then act.

    Both rooms are full 3x2 floor rectangles and each room holds exactly one
    player, so this never needs obstacle avoidance.  Vertical-first ordering is
    what makes every approach arrive already facing its work cell.
    """
    vertical = jnp.where(target_row > row, A_DOWN, A_UP)
    horizontal = jnp.where(target_col > col, A_RIGHT, A_LEFT)
    face_ok = jnp.take(facing, face_dir) > 0
    turn = jnp.take(jnp.asarray(_FACE_ACTION, dtype=jnp.int32), face_dir)
    return jnp.where(
        row != target_row,
        vertical,
        jnp.where(col != target_col, horizontal, jnp.where(face_ok, at_action, turn)),
    ).astype(jnp.int32)


def _displayed_recipe(obs, mask):
    """Ingredient id and validity of a recipe display under `mask`."""
    count_0 = _masked(obs, mask, CH_RECIPE_ING)
    count_1 = _masked(obs, mask, CH_RECIPE_ING + 1)
    count_2 = _masked(obs, mask, CH_RECIPE_ING + 2)
    valid = (count_0 + count_1 + count_2) > 0
    # Only [0,0,0] and [1,1,1] are configured; ingredient 2 has no supply in the
    # cook's room, so a stray display is resolved to the nearest reachable type.
    ingredient = jnp.where(count_1 > count_0, 1, 0).astype(jnp.int32)
    return ingredient, valid


def init_agent_0():
    return jnp.asarray(_INIT_0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset, jnp.asarray(_INIT_0, dtype=jnp.int32), memory)
    last_sent = memory[0]
    since_press = memory[1]
    retry_available = memory[2]
    clock = memory[3]

    row, col, _, _ = _locate(obs, CH_RECIPE_IND, RECIPE_IND_RC[0], RECIPE_IND_RC[1],
                             BUTTON_CELL[0], BUTTON_CELL[1])
    recipe, recipe_valid = _displayed_recipe(obs, obs[:, :, CH_RECIPE_IND])

    parked = (row == BUTTON_CELL[0]) & (col == BUTTON_CELL[1])
    news = recipe_valid & ((last_sent < 0) | (recipe != last_sent))
    # One retry per broadcast: a partner that missed the window gets a second
    # chance, but a partner that is ignoring the button (no recipe change ever
    # follows) does not keep being charged 5 reward for nothing.
    insurance = recipe_valid & (since_press >= PRESS_TIMEOUT) & (retry_available > 0)
    want_press = news | insurance

    facing = _facing(obs)
    action = _navigate(row, col, BUTTON_CELL[0], BUTTON_CELL[1], D_RIGHT, facing,
                       jnp.where(want_press, A_INTERACT, A_STAY).astype(jnp.int32))
    pressed = parked & want_press & (action == A_INTERACT)

    next_memory = jnp.stack([
        jnp.where(pressed, recipe, last_sent),
        jnp.where(pressed, 0, jnp.minimum(since_press + 1, 10000)),
        jnp.where(pressed, jnp.where(news, 1, 0), retry_available),
        clock + 1,
    ]).astype(jnp.int32)

    trace = {
        'recipe': recipe,
        'recipe_valid': recipe_valid.astype(jnp.int32),
        'pressed': pressed.astype(jnp.int32),
        'row': row,
        'col': col,
    }
    return action.astype(jnp.int32), next_memory, trace


def init_agent_1():
    return jnp.asarray(_INIT_1, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset, jnp.asarray(_INIT_1, dtype=jnp.int32), memory)
    belief = memory[0]
    known = memory[1]
    checking = memory[2]
    wait = memory[3]
    misses = memory[4]
    clock = memory[5]
    ever_delivered = memory[6]
    unreliable = memory[7]
    confirmed = memory[8]
    dump_index = memory[9]

    row, col, _, _ = _locate(obs, CH_POT, POT_RC[0], POT_RC[1], POT_CELL[0], POT_CELL[1])

    # --- grounded recipe signal -------------------------------------------
    # Any visible recipe display: the button at (2, 3) in practice, and the
    # always-on indicator too if a differently shaped map ever showed it here.
    signal_mask = obs[:, :, CH_BUTTON] + obs[:, :, CH_RECIPE_IND]
    signal_recipe, signal_valid = _displayed_recipe(obs, signal_mask)
    belief = jnp.where(signal_valid, signal_recipe, belief)
    known = jnp.where(signal_valid, 1, known)
    checking = jnp.where(signal_valid, 0, checking)
    wait = jnp.where(signal_valid, 0, wait)
    misses = jnp.where(signal_valid, 0, misses)
    unreliable = jnp.where(signal_valid, 0, unreliable)
    confirmed = jnp.where(signal_valid & (ever_delivered > 0), 1, confirmed)

    # --- pot state ---------------------------------------------------------
    pot_mask = obs[:, :, CH_POT]
    pot_counts = jnp.stack([_masked(obs, pot_mask, CH_ITEM_ING + k) for k in range(3)])
    pot_total = jnp.sum(pot_counts)
    pot_ready = _masked(obs, pot_mask, CH_ITEM_COOKED) > 0
    pot_cooking = _masked(obs, pot_mask, CH_POT_TIMER) > 0
    pot_open = jnp.logical_not(pot_ready | pot_cooking) & (pot_total < 3)

    # --- inventory ---------------------------------------------------------
    held_plate = obs[2, 2, CH_SELF_PLATE]
    held_cooked = obs[2, 2, CH_SELF_COOKED]
    held_counts = obs[2, 2, CH_SELF_ING:CH_SELF_ING + 3]
    held_total = jnp.sum(held_counts)
    held_belief = jnp.take(held_counts, belief)
    holding_dish = (held_cooked > 0) & (held_plate > 0)
    holding_plate = (held_plate > 0) & (held_cooked == 0)
    holding_raw = (held_plate == 0) & (held_total > 0)
    hands_free = (held_plate == 0) & (held_total == 0)

    dish_matches = (held_belief == 3) & (held_total == 3)
    raw_matches = held_belief > 0

    # --- counters ----------------------------------------------------------
    # The four counters reachable from this room, in preference order.  Column
    # 5 comes first so the recipe button stays inside the crop while working.
    visible, contents, matching = _counter_state(obs, row, col, belief)
    access_row = jnp.asarray([c[0] for c in COUNTERS], dtype=jnp.int32)
    access_col = jnp.asarray([c[1] for c in COUNTERS], dtype=jnp.int32)
    access_dir = jnp.asarray([c[4] for c in COUNTERS], dtype=jnp.int32)

    # Dump target: keep the current choice while it is not known to be taken,
    # and only ever move down the list, so a full counter cannot cause a loop.
    known_taken = visible & (contents > 0)
    fresh_index = jnp.argmax(jnp.logical_not(known_taken)).astype(jnp.int32)
    blocked = jnp.take(known_taken, dump_index)
    dump_row = jnp.take(access_row, dump_index)
    dump_col = jnp.take(access_col, dump_index)
    dump_dir = jnp.take(access_dir, dump_index)

    # A counter holding a single unit of the wanted ingredient is a closer
    # supply than the pile, and picking it up frees the counter again.
    recycle = jnp.any(matching)
    recycle_index = jnp.argmax(matching).astype(jnp.int32)

    # --- goal selection ----------------------------------------------------
    supply_row = jnp.where(belief == 1, ING1_CELL[0], ING0_CELL[0])
    fetch_row = jnp.where(recycle, jnp.take(access_row, recycle_index), supply_row)
    fetch_col = jnp.where(recycle, jnp.take(access_col, recycle_index), ING0_CELL[1])
    fetch_dir = jnp.where(recycle, jnp.take(access_dir, recycle_index), D_RIGHT)

    # Default (hands free, pot idle and empty): fetch, or hold still while
    # waiting for a broadcast that may be on its way.
    waiting = checking > 0
    goal_row = jnp.where(waiting, POT_CELL[0], fetch_row)
    goal_col = jnp.where(waiting, POT_CELL[1], fetch_col)
    goal_dir = jnp.where(waiting, D_LEFT, fetch_dir)
    goal_act = jnp.where(waiting, A_STAY, A_INTERACT)

    # Hands free with a busy pot: fetch the plate ahead of time.
    pot_busy = pot_ready | pot_cooking
    goal_row = jnp.where(hands_free & pot_busy, PLATE_CELL[0], goal_row)
    goal_col = jnp.where(hands_free & pot_busy, PLATE_CELL[1], goal_col)
    goal_dir = jnp.where(hands_free & pot_busy, D_LEFT, goal_dir)
    goal_act = jnp.where(hands_free & pot_busy, A_INTERACT, goal_act)

    # Hands free with a partly filled idle pot: keep filling regardless of the
    # wait, because the pot must reach 3 before it can be emptied again.
    topping_up = hands_free & jnp.logical_not(pot_busy) & (pot_total > 0)
    goal_row = jnp.where(topping_up, fetch_row, goal_row)
    goal_col = jnp.where(topping_up, fetch_col, goal_col)
    goal_dir = jnp.where(topping_up, fetch_dir, goal_dir)
    goal_act = jnp.where(topping_up, A_INTERACT, goal_act)

    # Carrying a raw ingredient.
    can_insert = holding_raw & raw_matches & pot_open
    goal_row = jnp.where(holding_raw, jnp.where(can_insert, POT_CELL[0], dump_row), goal_row)
    goal_col = jnp.where(holding_raw, jnp.where(can_insert, POT_CELL[1], dump_col), goal_col)
    goal_dir = jnp.where(holding_raw, jnp.where(can_insert, D_LEFT, dump_dir), goal_dir)
    goal_act = jnp.where(holding_raw, A_INTERACT, goal_act)

    # Carrying an empty plate: stand at the pot and scoop the moment it is done.
    goal_row = jnp.where(holding_plate, POT_CELL[0], goal_row)
    goal_col = jnp.where(holding_plate, POT_CELL[1], goal_col)
    goal_dir = jnp.where(holding_plate, D_LEFT, goal_dir)
    goal_act = jnp.where(holding_plate, jnp.where(pot_ready, A_INTERACT, A_STAY), goal_act)

    # Carrying a plated dish: deliver it only if it still matches the belief.
    goal_row = jnp.where(holding_dish, jnp.where(dish_matches, DELIVER_CELL[0], dump_row), goal_row)
    goal_col = jnp.where(holding_dish, jnp.where(dish_matches, DELIVER_CELL[1], dump_col), goal_col)
    goal_dir = jnp.where(holding_dish, jnp.where(dish_matches, D_RIGHT, dump_dir), goal_dir)
    goal_act = jnp.where(holding_dish, A_INTERACT, goal_act)

    facing = _facing(obs)
    action = _navigate(row, col, goal_row, goal_col, goal_dir, facing, goal_act)

    # --- belief bookkeeping ------------------------------------------------
    delivering = (holding_dish & dish_matches & (action == A_INTERACT)
                  & (row == DELIVER_CELL[0]) & (col == DELIVER_CELL[1]))

    # A correct delivery resamples the recipe, so afterwards pause at the pot
    # and look for a fresh broadcast before committing to ingredients.
    limit = jnp.where(ever_delivered > 0, WAIT_AFTER_DELIVERY, WAIT_FIRST)
    timed_out = (checking > 0) & (wait >= limit)
    silent = (timed_out & (ever_delivered > 0) & (known > 0)
              & jnp.logical_not(signal_valid))
    misses = jnp.where(silent, misses + 1, misses)
    # A partner that has broadcast at least once after a delivery has shown it
    # re-broadcasts, so its silence really does mean "recipe unchanged" and the
    # belief can be trusted much longer.  A partner that has only ever spoken
    # once is far more likely to leave the belief silently stale.
    patience = jnp.where(confirmed > 0, UNRELIABLE_AFTER, UNRELIABLE_UNCONFIRMED)
    unreliable = jnp.where(misses >= patience, 1, unreliable)
    checking = jnp.where(timed_out, 0, checking)
    wait = jnp.where(checking > 0, wait + 1, 0)

    # Without a trustworthy broadcast, the posterior after an unconfirmed
    # delivery favours the other recipe (a wrong dish never resamples), so
    # alternate instead of repeating a guess that may be stuck wrong.
    flip = delivering & ((known == 0) | (unreliable > 0))
    belief = jnp.where(flip, 1 - belief, belief)
    checking = jnp.where(delivering, 1, checking)
    wait = jnp.where(delivering, 0, wait)
    ever_delivered = jnp.where(delivering, 1, ever_delivered)

    dumping = (holding_raw & jnp.logical_not(can_insert)) | (holding_dish & jnp.logical_not(dish_matches))
    dump_index = jnp.where(
        dumping,
        jnp.where(blocked, jnp.minimum(dump_index + 1, len(COUNTERS) - 1), dump_index),
        fresh_index,
    )

    next_memory = jnp.stack([
        belief, known, checking, wait, misses, clock + 1, ever_delivered, unreliable,
        confirmed, dump_index,
    ]).astype(jnp.int32)

    trace = {
        'belief': belief.astype(jnp.int32),
        'known': known.astype(jnp.int32),
        'checking': checking.astype(jnp.int32),
        'misses': misses.astype(jnp.int32),
        'unreliable': unreliable.astype(jnp.int32),
        'confirmed': confirmed.astype(jnp.int32),
        'recycle': recycle.astype(jnp.int32),
        'dump_index': dump_index.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
        'delivering': delivering.astype(jnp.int32),
        'row': row,
        'col': col,
    }
    return action.astype(jnp.int32), next_memory, trace
