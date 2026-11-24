"""Overcooked V2 (test_time_wide) policy.

Layout facts (see context/environment/README.md):
  - agent_0's reachable room is rows {1,2} x cols {1..4}. It is walled off from
    the recipe indicator R (row 6): R is never inside agent_0's 5x5 crop.
  - agent_1's reachable room is rows {4,5} x cols {1..4}. It can always see R
    (row distance <=2 for both rows), but cannot reach ingredient supplies 0/1,
    the plate pile, or the delivery station -- only ingredient 3 (unused by any
    recipe), the two pots (from below) and R.
  - Consequently agent_0 can execute a full cook/deliver cycle alone but never
    observes the target recipe; agent_1 observes the recipe but cannot act on
    it. There is no reward signal fed into act(), and delivering clears the
    hand whether or not the dish matched, so agent_0 gets no direct feedback
    about success either.

Strategy (robust to an unknown, independently generated partner):
  - agent_0 ignores the partner entirely and deterministically alternates
    which of the two recipes it cooks, using only its own local observation
    (self position via the always-visible goal marker, held-item channels,
    and pot-content channels) to drive a small reactive controller. With
    exactly two possible recipes and no feedback, alternating the guess every
    delivery is the best achievable open-loop policy: it can never be wrong
    twice in a row (a miss forces the unique other recipe to be the true
    one), giving a steady-state hit rate of 2/3 regardless of what the
    partner does. It pipelines the two pots (one "consumer" being serviced --
    plate fetched, waited on, scooped, delivered -- while the other
    "producer" is filled with the complementary guess) so the fill phase
    overlaps the other pot's 20-step cook timer instead of idling; roles
    swap on every delivery, which still makes the delivered-guess sequence
    alternate 0,1,0,1,... exactly, preserving the 2/3 hit rate while roughly
    doubling throughput versus a single-pot design.
  - agent_1 cannot contribute physically, so it avoids any harmful action
    (it never touches ingredient 3 or a pot, and never issues INTERACT at
    all) and instead emits a costless, honestly-grounded position/facing
    signal: it walks to the pot-access cell below whichever pot corresponds
    to the currently displayed recipe and faces it. This costs nothing,
    cannot desynchronize a partner that ignores it, and may help a partner
    that happens to look for such a cue.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'oc2_alternating_guess_pointer',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}

# Movement/interact action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Facing ids as encoded in the observation channels (UP=0, DOWN=1, RIGHT=2, LEFT=3).
FACE_UP, FACE_DOWN, FACE_RIGHT, FACE_LEFT = 0, 1, 2, 3


def _goto_action(self_row, self_col, target_row, target_col):
    dr = target_row - self_row
    dc = target_col - self_col
    row_act = jnp.where(dr > 0, DOWN, UP)
    col_act = jnp.where(dc > 0, RIGHT, LEFT)
    return jnp.where(dr != 0, row_act, jnp.where(dc != 0, col_act, STAY)).astype(jnp.int32)


def _self_facing(obs):
    vals = obs[2, 2, 1:5]
    return jnp.argmax(vals).astype(jnp.int32)


# ---------------------------------------------------------------------------
# agent_0: executor. Self-sufficient; never depends on partner behaviour.
# ---------------------------------------------------------------------------

# Room cells for agent_0: rows {1,2} x cols {1..4}.
# Two pots, both accessed from row 2 (facing DOWN): pot 0 at column 1
# (co-located with the ingredient-1 supply, so pot 0 pairs cheaply with
# ingredient 1), pot 1 at column 3 (closer to the right-hand supplies).
# Ingredient access: row 1 for ingredient 0, row 2 for ingredient 1; column 1
# (facing LEFT) for the side paired with pot 0, column 4 (facing RIGHT) for
# the side paired with pot 1.
# Plate access (facing UP toward plate pile (0,3)): (1,3).
# Delivery access (facing UP toward delivery (0,2)): (1,2).
POT_ROW = 2
PLA_ROW, PLA_COL = 1, 3
DA_ROW, DA_COL = 1, 2


def init_agent_0():
    return {
        # Which physical pot (0 or 1) is currently being serviced (plate
        # fetched, scooped, delivered) and the recipe guess its contents
        # correspond to. The *other* pot is always the "producer": it gets
        # filled with the complementary guess while the consumer cooks, so
        # the fill phase overlaps the other pot's cook timer instead of
        # idling. Swapping consumer<->producer on every delivery still makes
        # the delivered-guess sequence alternate 0,1,0,1,... exactly as the
        # single-pot design did, preserving the 2/3 steady-state hit rate.
        'consumer_pot': jnp.asarray(0, dtype=jnp.int32),
        'guess_consumer': jnp.asarray(0, dtype=jnp.int32),
        # True only before the very first pot has ever been filled. At
        # episode start both pots are empty, so there is nothing yet for the
        # producer-pipelining logic to overlap with: the fill target must be
        # the *consumer* pot itself (matching the plain single-pot design)
        # until that first fill completes, at which point this flips off
        # permanently and the fill target becomes the producer pot as usual.
        'bootstrap': jnp.asarray(1, dtype=jnp.int32),
        # Sticky bit: true from the moment the current fill target (producer
        # pot normally, or the consumer pot itself during bootstrap) is
        # observed full/cooking (while standing at it) until the next
        # delivery swaps roles. Needed because a couple of routes from a pot
        # to the plate pile pass through that pot's own ingredient-access
        # cell, and without this the pass-through would look like "time to
        # pick another ingredient" even though the pot is already full (the
        # same class of bug found and fixed for the single-pot design in
        # A01).
        'fill_full': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    # Self-localize every step from the goal marker (always inside agent_0's
    # crop anywhere in its 8-cell room), so the controller needs no travel
    # memory and is robust to the random per-episode spawn position/facing.
    goal_ch = obs[:, :, 23]
    flat_idx = jnp.argmax(goal_ch.reshape(-1)).astype(jnp.int32)
    local_row = flat_idx // 5
    local_col = flat_idx % 5
    self_row = 2 - local_row
    self_col = 4 - local_col

    facing_id = _self_facing(obs)
    facing_down = facing_id == FACE_DOWN
    facing_up = facing_id == FACE_UP

    plate_flag = obs[2, 2, 5]
    cooked_flag = obs[2, 2, 6]
    ing0 = obs[2, 2, 7]
    ing1 = obs[2, 2, 8]
    holding_plate_only = (plate_flag == 1) & (cooked_flag == 0)
    holding_dish = (plate_flag == 1) & (cooked_flag == 1)

    consumer_pot = memory['consumer_pot']
    guess_consumer = memory['guess_consumer']
    bootstrap = memory['bootstrap']
    fill_full = memory['fill_full']
    producer_pot = 1 - consumer_pot
    guess_producer = 1 - guess_consumer

    # During bootstrap (nothing ever filled yet) the fill target is the
    # consumer pot itself; afterward it is always the producer pot. The
    # switch must take effect within the very step the consumer pot is
    # observed reaching 3 units, not just on the following step -- otherwise
    # this step would still aim at the (now-full) consumer pot's ingredient
    # cell and try to fetch a 4th, unwanted unit.
    consumer_col_for_check = jnp.where(consumer_pot == 0, 1, 3)
    at_consumer_for_check = (self_row == POT_ROW) & (self_col == consumer_col_for_check)
    consumer_full_live = at_consumer_for_check & ((obs[3, 2, 34] + obs[3, 2, 35]) >= 3)
    effective_bootstrap = (bootstrap > 0) & (~consumer_full_live)
    fill_pot = jnp.where(effective_bootstrap, consumer_pot, producer_pot)
    fill_guess = jnp.where(effective_bootstrap, guess_consumer, guess_producer)

    holding_ing = jnp.where(fill_guess == 0, ing0, ing1) > 0
    holding_none = (~holding_plate_only) & (~holding_dish) & (~holding_ing)

    consumer_col = jnp.where(consumer_pot == 0, 1, 3)
    fill_col = jnp.where(fill_pot == 0, 1, 3)
    at_consumer = (self_row == POT_ROW) & (self_col == consumer_col)
    at_fill = (self_row == POT_ROW) & (self_col == fill_col)

    ia_row = jnp.where(fill_guess == 0, 1, 2)
    ia_col = jnp.where(fill_pot == 0, 1, 4)
    at_ia = (self_row == ia_row) & (self_col == ia_col)
    pick_turn_action = jnp.where(fill_pot == 0, LEFT, RIGHT)
    pick_face_id = jnp.where(fill_pot == 0, FACE_LEFT, FACE_RIGHT)
    facing_pick = facing_id == pick_face_id

    at_plate = (self_row == PLA_ROW) & (self_col == PLA_COL)
    at_deliver = (self_row == DA_ROW) & (self_col == DA_COL)

    # Whatever pot sits directly below self (row 2, either pot's column) has
    # local coords (3, 2). Only meaningful when actually standing at a pot;
    # only read in branches gated on at_consumer / at_fill.
    pot_total_here = obs[3, 2, 34] + obs[3, 2, 35]
    pot_cooked_here = obs[3, 2, 33]

    # R1: holding a cooked dish -> deliver it, then swap consumer/producer.
    r1_action = jnp.where(
        ~at_deliver, _goto_action(self_row, self_col, DA_ROW, DA_COL),
        jnp.where(~facing_up, UP, INTERACT))
    is_delivering = at_deliver & facing_up & holding_dish

    # R2: holding an empty plate -> go scoop the consumer pot once cooked.
    r2_action = jnp.where(
        ~at_consumer, _goto_action(self_row, self_col, POT_ROW, consumer_col),
        jnp.where(~facing_down, DOWN,
                  jnp.where(pot_cooked_here == 1, INTERACT, STAY)))

    # R3: holding the fill target's guessed raw ingredient -> deposit it.
    r3_action = jnp.where(
        ~at_fill, _goto_action(self_row, self_col, POT_ROW, fill_col),
        jnp.where(~facing_down, DOWN, INTERACT))

    # R4: holding nothing. If the fill-target pot already holds 3 units this
    # cycle (live reading, or remembered via the sticky bit once we've left
    # it), go fetch a plate for the consumer; otherwise seek the fill
    # target's ingredient-access cell and pick. Some pot/ingredient pairings
    # route through that pot's own ingredient-access cell en route to the
    # plate pile, so the sticky/live "fill target already full" check must
    # take priority over "am I at the ingredient cell" (see A01's fixed bug).
    fill_full_live = at_fill & (pot_total_here >= 3)
    fill_full_now = (fill_full > 0) | fill_full_live
    r4_goto_plate = _goto_action(self_row, self_col, PLA_ROW, PLA_COL)
    r4_pick = jnp.where(~facing_pick, pick_turn_action, INTERACT)
    r4_seek = _goto_action(self_row, self_col, ia_row, ia_col)
    r4_action = jnp.where(fill_full_now, r4_goto_plate, jnp.where(at_ia, r4_pick, r4_seek))

    # R5: at the plate pile with empty hands and the fill target is (or was,
    # per the sticky bit) actually full -> pick up a plate.
    r5_action = jnp.where(~facing_up, UP, INTERACT)
    at_plate_empty = holding_none & at_plate & (fill_full > 0)

    action = jnp.where(
        holding_dish, r1_action,
        jnp.where(holding_plate_only, r2_action,
                  jnp.where(holding_ing, r3_action,
                            jnp.where(at_plate_empty, r5_action, r4_action)))
    ).astype(jnp.int32)

    # Bootstrap ends the moment the consumer pot is first observed full;
    # from then on the fill target is (permanently) the producer pot. Uses
    # consumer_full_live (not fill_full_live, which by now already reflects
    # whichever pot effective_bootstrap selected for this step).
    next_bootstrap = jnp.where((bootstrap > 0) & consumer_full_live, 0, bootstrap).astype(jnp.int32)
    next_consumer_pot = jnp.where(is_delivering, producer_pot, consumer_pot).astype(jnp.int32)
    next_guess_consumer = jnp.where(is_delivering, guess_producer, guess_consumer).astype(jnp.int32)
    next_fill_full = jnp.where(
        is_delivering, 0, jnp.where(fill_full_live, 1, fill_full)).astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    one = jnp.asarray(1, dtype=jnp.int32)
    next_memory = {
        'consumer_pot': jnp.where(reset, zero, next_consumer_pot),
        'guess_consumer': jnp.where(reset, zero, next_guess_consumer),
        'bootstrap': jnp.where(reset, one, next_bootstrap),
        'fill_full': jnp.where(reset, zero, next_fill_full),
    }
    action = jnp.where(reset, jnp.asarray(STAY, dtype=jnp.int32), action)
    return action, next_memory, {}


# ---------------------------------------------------------------------------
# agent_1: cannot reach ingredients 0/1, plates, or delivery. It stays
# harmless (never touches ingredient 3 or a pot) and points at whichever pot
# access cell corresponds to the currently displayed recipe, as a costless,
# honestly-grounded signal that a partner may or may not use.
# ---------------------------------------------------------------------------

POT1_ROW, POT1_COL = 4, 1  # below pot (3,1)
POT2_ROW, POT2_COL = 4, 3  # below pot (3,3)


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    recipe_ch = obs[:, :, 25]
    flat_idx = jnp.argmax(recipe_ch.reshape(-1)).astype(jnp.int32)
    local_row = flat_idx // 5
    local_col = flat_idx % 5
    self_row = 8 - local_row
    self_col = 4 - local_col

    ing0_disp = obs[local_row, local_col, 40]
    target_row = POT1_ROW
    target_col = jnp.where(ing0_disp > 0, POT1_COL, POT2_COL)

    facing_id = _self_facing(obs)
    at_target = (self_row == target_row) & (self_col == target_col)
    action = jnp.where(
        at_target,
        jnp.where(facing_id == FACE_UP, STAY, UP),
        _goto_action(self_row, self_col, target_row, target_col),
    ).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)
    action = jnp.where(reset, jnp.asarray(STAY, dtype=jnp.int32), action)
    return action, next_memory, {}
