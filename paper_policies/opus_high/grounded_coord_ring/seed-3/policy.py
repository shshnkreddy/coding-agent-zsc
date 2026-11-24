"""Grounded-communication policy for the ``grounded_coord_ring`` layout.

Layout facts that drive the design (all read off the environment spec, never
assumed about the partner):

* The two seats live in *disconnected* floor regions.  ``agent_1`` occupies the
  inner ring around the pot and can reach the pot, both ingredient supplies,
  both plate piles and both delivery stations, so it can complete a dish alone.
  ``agent_0`` occupies the outer ring and can reach no inner cell and no pot,
  so it can never cook.  Roles are therefore fixed by geometry, not convention.
* ``agent_1`` can never see an always-on recipe indicator: every ``R`` cell is
  further than the view radius from every inner floor cell.  The only way a
  recipe can enter ``agent_1``'s observation is a *button* indicator at (2,4)
  or (6,4), reachable only from the outer-ring cells (1,4) and (7,4).

The signal is grounded: an active button renders the live recipe straight into
the recipe channels of the cook's observation, so no shared code word or
private convention is required for it to be understood.

``agent_1`` (cook):
  * acts on *any* visible recipe display, button or indicator, either seat's;
  * keeps that belief until it delivers, because the recipe can only be
    resampled by a correct delivery;
  * refuses to deliver a dish a known recipe contradicts, stashing it on a
    counter rather than taking the -20;
  * if no signal ever arrives, falls back to an *alternating* guess, which is
    the best open-loop rule available given that a wrong delivery leaves the
    recipe unchanged (so the next, flipped, guess is certain to be right).

``agent_0`` (broadcaster) posts beside a button, reads the always-on indicator
and presses about once per delivery cycle -- on a recipe change, on seeing the
partner carry a dish, or on a keep-alive timer -- so a partner that expects a
fresh press every cycle never stalls waiting for one.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_ring_coord', 'adapter': 'overcooked_v2',
                   'adapter_version': 1, 'max_height': 9, 'max_width': 9,
                   'num_ingredients': 3}

# --------------------------------------------------------------------------
# Static map knowledge (context/environment/specification.json)
# 0 floor, 1 wall, 2 delivery, 3 pot, 4 indicator, 5 button, 6 plate pile,
# 7 ingredient-0 supply, 8 ingredient-1 supply, 9 ingredient-2 supply.
# The code is one plus the offset of the matching static channel (20..28).
# --------------------------------------------------------------------------
MAP_CODE = [[1, 1, 1, 9, 4, 9, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 5, 1, 1, 0, 1],
            [9, 0, 7, 0, 0, 0, 6, 0, 9],
            [4, 0, 2, 0, 3, 0, 2, 0, 4],
            [9, 0, 8, 0, 0, 0, 6, 0, 9],
            [1, 0, 1, 1, 5, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 9, 4, 9, 1, 1, 1]]

# Reachable floor of each seat, used both as the localisation hypothesis set
# and as the index space of the navigation tables.
REGION_0 = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 7],
            [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [7, 6], [7, 5], [7, 4],
            [7, 3], [7, 2], [7, 1], [6, 1], [5, 1], [4, 1], [3, 1], [2, 1]]
REGION_1 = [[3, 3], [3, 4], [3, 5], [4, 3], [4, 5], [5, 3], [5, 4], [5, 5]]

# Shortest-path tables (breadth first over the floor graph, precomputed).
# action ids: RIGHT=0 DOWN=1 LEFT=2 UP=3 STAY=4 INTERACT=5
DIST_1 = [[0, 1, 2, 1, 3, 2, 3, 4],
          [1, 0, 1, 2, 2, 3, 4, 3],
          [2, 1, 0, 3, 1, 4, 3, 2],
          [1, 2, 3, 0, 4, 1, 2, 3],
          [3, 2, 1, 4, 0, 3, 2, 1],
          [2, 3, 4, 1, 3, 0, 1, 2],
          [3, 4, 3, 2, 2, 1, 0, 1],
          [4, 3, 2, 3, 1, 2, 1, 0]]
NEXT_1 = [[4, 0, 0, 1, 0, 1, 1, 0],
          [2, 4, 0, 2, 0, 2, 0, 0],
          [2, 2, 4, 2, 1, 1, 1, 1],
          [3, 3, 3, 4, 1, 1, 1, 1],
          [3, 3, 3, 1, 4, 1, 1, 1],
          [3, 3, 0, 3, 0, 4, 0, 0],
          [2, 0, 0, 2, 0, 2, 4, 0],
          [2, 3, 3, 2, 3, 2, 2, 4]]
# agent_0 only ever walks to one of its two button posts (1,4) / (7,4)
DIST_0 = [[3, 9], [2, 10], [1, 11], [0, 12], [1, 11], [2, 10], [3, 9],
          [4, 8], [5, 7], [6, 6], [7, 5], [8, 4], [9, 3], [10, 2], [11, 1],
          [12, 0], [11, 1], [10, 2], [9, 3], [8, 4], [7, 5], [6, 6], [5, 7],
          [4, 8]]
NEXT_0 = [[0, 1], [0, 2], [0, 2], [4, 0], [2, 0], [2, 0], [2, 1], [3, 1],
          [3, 1], [3, 1], [3, 1], [3, 1], [3, 2], [0, 2], [0, 2], [0, 4],
          [2, 0], [2, 0], [3, 0], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]]

# facing ids: UP=0 DOWN=1 RIGHT=2 LEFT=3
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
# movement action that merely turns to a given facing
FACE_ACT = [3, 1, 0, 2]

# agent_1 work stations, as indices into REGION_1, with the facing they need
POT_STANDS = [1, 6, 3, 4]              # (3,4) (5,4) (4,3) (4,5)
POT_DIRS = [DOWN, UP, RIGHT, LEFT]
# (4,3) is the one cell that touches the pot, a delivery station, and -- via
# (3,3) and (5,3) -- both ingredient supply stands.  Waiting out the cook there
# means two steps from scoop to delivery and three more to the next pickup,
# whichever ingredient is next.  Both buttons are visible from it as well, so
# every press is read while we wait.
SCOOP_STANDS = [3]                     # (4,3)
SCOOP_DIRS = [RIGHT]
DELIV_STANDS = [3, 4]
DELIV_DIRS = [LEFT, RIGHT]
PLATE_STANDS = [2, 7]                  # (3,5) (5,5)
PLATE_DIRS = [RIGHT, RIGHT]
CNT_STANDS = [0, 2, 5, 7]              # counters (2,3) (2,5) (6,3) (6,5)
CNT_DIRS = [UP, UP, DOWN, DOWN]
CNT_CELLS = [[2, 3], [2, 5], [6, 3], [6, 5]]
# Where to stand while waiting for a recipe signal.  (4,3) sees both buttons
# but neither button post can see it; (3,4) and (5,4) each see one button and
# each sit inside one post's view, so a partner that presses on noticing an
# idle teammate can notice us there.  Sweeping the three covers both cases.
LISTEN_SWEEP = [3, 1, 6]               # (4,3) (3,4) (5,4)
ING_STANDS = [0, 5]                    # ingredient 0 at (3,3), 1 at (5,3)

BUTTON_CELLS = [[2, 4], [6, 4]]
R_CELLS = [[0, 4], [4, 0], [4, 8], [8, 4]]
DISPLAY_CELLS = [[0, 4], [4, 0], [4, 8], [8, 4], [2, 4], [6, 4]]

# tunables
# KEEPALIVE 37 means a press every 38 steps (the counter resets on the press
# step itself), which is exactly the length of a cook cycle on this layout, so
# a press lands one step before each delivery and covers both the hand-in and
# the read of the recipe resampled by it.
KEEPALIVE = 37       # agent_0: force a press this long after the previous one
CUED_PERIOD = 70     # agent_0: slack period while the partner cues us itself
SLOW_PERIOD = 90     # agent_0: slack period while nothing seems to happen
STALE_LIMIT = 150    # agent_0: no change and no partner movement for this long
DISH_COOLDOWN = 28   # agent_0: gap for the "partner carries a dish" trigger
DISH_MODE = 60       # agent_0: how recent a sighted dish keeps us cue-driven
IDLE_COOLDOWN = 20   # agent_0: gap for the "partner is stuck waiting" trigger
IDLE_STEPS = 5       # agent_0: how long a partner must sit still to count
IDLE_BUDGET = 3      # agent_0: stuck-partner presses between productivity
WAIT_LIMIT = 26      # agent_1: steps spent waiting before committing the pot
DISH_WAIT = 30       # agent_1: steps spent waiting before delivering unsure
SWEEP_AFTER = 18     # agent_1: when to start making itself visible instead
BLIND_GIVEUP = 45    # agent_1: total fruitless waiting before going open loop
SIGNALS_TRUSTED = 2  # agent_1: distinct displays before silence means anything
SILENCE_TRUSTED = 160  # agent_1: and only if one was seen this recently
POST_SWITCH = 70     # agent_0: abandon a post after this many blind steps

# memory layouts
INIT_0 = [0, -1, -1, -1, 0, 0, 0, 99, 0, 0, 0, -1, 0, 900, IDLE_BUDGET, 0]
INIT_1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 900, 0, 0]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _localize(obs, region):
    """Region index of self, by matching static channels against the map."""
    padded = jnp.pad(jnp.asarray(MAP_CODE, jnp.int32), 2)
    code = (obs[:, :, 20:29] * jnp.arange(1, 10, dtype=jnp.int32)).sum(-1)
    templates = jnp.stack([padded[r:r + 5, c:c + 5] for r, c in region])
    return jnp.argmin((templates != code[None]).sum(axis=(1, 2)))


def _global_view(obs, index, region):
    """Scatter the crop onto a padded board; cells outside the view stay 0."""
    rows = jnp.asarray([p[0] for p in region], jnp.int32)
    cols = jnp.asarray([p[1] for p in region], jnp.int32)
    board = jnp.zeros((13, 13, 40), jnp.int32)
    return jax.lax.dynamic_update_slice(board, obs, (rows[index],
                                                     cols[index], 0))


def _cell(board, rc):
    return board[rc[0] + 2, rc[1] + 2]


def _display(board, cells):
    """Live recipe counts rendered by any visible indicator or button."""
    return jnp.stack([_cell(board, rc)[36:39] for rc in cells]).max(axis=0)


def _approach(pos, stand, want_dir, facing):
    """Walk to ``stand``, turn to ``want_dir``, then INTERACT."""
    move = jnp.asarray(NEXT_1, jnp.int32)[pos, stand]
    turn = jnp.asarray(FACE_ACT, jnp.int32)[want_dir]
    at = pos == stand
    return jnp.where(at, jnp.where(facing == want_dir, 5, turn), move)


def _approach_best(pos, stands, dirs, facing, penalty=None):
    """Same, for the cheapest of several equivalent work stations."""
    stands = jnp.asarray(stands, jnp.int32)
    dirs = jnp.asarray(dirs, jnp.int32)
    cost = jnp.asarray(DIST_1, jnp.int32)[pos][stands]
    if penalty is not None:
        cost = cost + penalty
    k = jnp.argmin(cost)
    return _approach(pos, stands[k], dirs[k], facing)


def _walk_to(pos, stand):
    """Walk to ``stand`` and STAY once standing on it."""
    return jnp.where(pos == stand, jnp.int32(4),
                     jnp.asarray(NEXT_1, jnp.int32)[pos, stand])


def _sweep_stand(waited, hunting):
    """Where to stand while waiting for a recipe signal.

    (4,3) is the default because both buttons are visible from it.  Only when
    no display has ever been seen -- so a partner might be waiting to notice us
    rather than broadcasting on its own -- do we start alternating onto the two
    pot cells that a button post can see, accepting one button of visibility
    each in exchange for being visible ourselves.
    """
    sweep = jnp.asarray(LISTEN_SWEEP, jnp.int32)
    late = jnp.maximum(waited - SWEEP_AFTER, 0) // 10
    roam = hunting & (waited >= SWEEP_AFTER)
    return sweep[jnp.where(roam, 1 + late % 2, 0)]


# --------------------------------------------------------------------------
# agent_0 -- outer ring broadcaster
# memory: [post, bcast0..2, seen0..2, since_press, blind, presses, override,
#          partner_cell, still, since_dish, idle_budget, stale]
# --------------------------------------------------------------------------

def init_agent_0():
    return jnp.asarray(INIT_0, jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, jnp.asarray(INIT_0, jnp.int32), memory)

    pos = _localize(obs, REGION_0)
    board = _global_view(obs, pos, REGION_0)
    facing = jnp.argmax(obs[2, 2, 1:5])

    # ---- which recipe is currently requested ------------------------------
    recipe_now = _display(board, R_CELLS)
    recipe = jnp.where(recipe_now.sum() > 0, recipe_now, mem[4:7])
    have_recipe = recipe.sum() > 0

    # ---- which post to hold ------------------------------------------------
    # The ingredient-0 supply (3,2) only serves the top half of the inner ring
    # and the ingredient-1 supply (5,2) only the bottom half, so the requested
    # ingredient predicts which half the cook will be working in.  An override
    # set by the blind counter wins, for a partner that ignores that geometry.
    override = mem[10]
    prior = jnp.where(recipe[0] > 0, jnp.int32(0),
                      jnp.where(recipe[1] > 0, jnp.int32(1), mem[0]))
    wanted_post = jnp.where(override == 0, prior, override - 1)

    # ---- is a button already showing it -----------------------------------
    # Any lit button counts: it renders the live recipe, so a second press
    # while one is still counting down would buy nothing.  Checking only the
    # button of the post we are walking towards made us re-press the recipe
    # the partner had already read from the other one.
    buttons = jnp.stack([_cell(board, rc)[36:39] for rc in BUTTON_CELLS])
    active = buttons.sum() > 0

    broadcast = mem[1:4]
    changed = have_recipe & jnp.any(recipe != broadcast)
    partner_dish = (obs[:, :, 15] * obs[:, :, 16]).sum() > 0
    since = mem[7]
    at_held = jnp.asarray(DIST_0, jnp.int32)[pos, mem[0]] == 0
    at_post = at_held

    # ---- is the partner visibly stuck? ------------------------------------
    # Only an *empty-handed* teammate sitting still counts.  One that is
    # holding something is mid-task -- notably, standing at the pot with a
    # plate waiting out the twenty-step cook looks identical to idling, and
    # treating it as stuck bought a badly timed press every cycle.  Tracked
    # only while we hold the post, so our own walking cannot shift the frame.
    partner_here = obs[:, :, 10].sum() > 0
    partner_cell = jnp.argmax(obs[:, :, 10].reshape(-1))
    partner_empty = (obs[:, :, 15:20].sum() == 0) & partner_here
    tracked = partner_here & at_post & (mem[11] >= 0)
    same_cell = tracked & (partner_cell == mem[11])
    moved = tracked & (partner_cell != mem[11])
    still = jnp.where(same_cell, jnp.minimum(mem[12] + 1, 900), jnp.int32(0))
    partner_stuck = partner_empty & (still >= IDLE_STEPS)

    # ---- how much of the timing can the partner itself tell us? -----------
    # A partner that plates up where we can see it announces its own delivery
    # about four steps ahead, which is a better clock than ours: one press then
    # covers the hand-in and the read of the recipe it resamples.  While that
    # cue keeps arriving, stretch the fixed period right out so a mistimed
    # periodic press cannot leave a button lit and suppress the good one.
    # Our own cook plates up at (4,3), out of both posts' view, so self-play
    # never enters this mode.
    since_dish = jnp.where(partner_dish, jnp.int32(0),
                           jnp.minimum(mem[13] + 1, 900))
    # Only the moment a dish *appears* is news.  A partner that walks around
    # holding one, or drifts in and out of view with it, would otherwise buy a
    # fresh press every cooldown, which cost 100 a game against one probe.
    new_dish = partner_dish & (mem[13] > 0)
    # Evidence the partner is actually doing something: a recipe change means
    # a correct delivery, and seeing it move means it is at least playing.
    stale = jnp.where(changed | moved, jnp.int32(0),
                      jnp.minimum(mem[15] + 1, 900))
    period = jnp.where(stale > STALE_LIMIT, jnp.int32(SLOW_PERIOD),
                       jnp.where(since_dish < DISH_MODE, jnp.int32(CUED_PERIOD),
                                 jnp.int32(KEEPALIVE)))

    # A stuck partner may simply be one that never acts; a few presses are
    # worth trying, but only evidence of productivity buys more.
    idle_budget = jnp.where(changed, jnp.int32(IDLE_BUDGET), mem[14])
    trig_idle = partner_stuck & (since >= IDLE_COOLDOWN) & (idle_budget > 0)
    trig_other = (changed
                  | (new_dish & (since >= DISH_COOLDOWN))
                  | (since >= period))
    want_press = (~active) & have_recipe & (trig_other | trig_idle)

    # Relocating between posts costs twelve steps, during which nothing can be
    # broadcast.  Never start that walk with a press outstanding or nearly due:
    # send the recipe first, then move.  Skipping this made every recipe change
    # cost a stall, because the change flips the wanted post on the very step
    # the change becomes visible.
    hold_post = at_held & (want_press | (since >= period - 14))
    # With a press outstanding and no post under our feet, the recipe matters
    # more than which button carries it: head for whichever post is closer.
    nearest = jnp.argmin(jnp.asarray(DIST_0, jnp.int32)[pos]).astype(jnp.int32)
    post = jnp.where(hold_post, mem[0],
                     jnp.where(want_press, nearest, wanted_post))
    button_dir = jnp.asarray([DOWN, UP], jnp.int32)[post]
    at_post = jnp.asarray(DIST_0, jnp.int32)[pos, post] == 0

    move = jnp.asarray(NEXT_0, jnp.int32)[pos, post]
    turn = jnp.asarray(FACE_ACT, jnp.int32)[button_dir]
    aimed = at_post & (facing == button_dir)
    action = jnp.where(at_post,
                       jnp.where(facing == button_dir,
                                 jnp.where(want_press, 5, 4), turn),
                       move)

    pressed = aimed & want_press
    # an active button renders the live recipe, so the cook already has it
    broadcast = jnp.where(pressed | active, recipe, broadcast)
    since = jnp.where(pressed, jnp.int32(0), jnp.minimum(since + 1, 900))
    idle_budget = jnp.where(pressed & trig_idle & (~trig_other),
                            jnp.maximum(idle_budget - 1, 0), idle_budget)

    # ---- has this post gone blind? ----------------------------------------
    blind = jnp.where(partner_here | changed, jnp.int32(0),
                      jnp.minimum(mem[8] + 1, 900))
    switch = blind > POST_SWITCH
    override = jnp.where(switch, 2 - post, override)
    blind = jnp.where(switch, jnp.int32(0), blind)

    nxt = jnp.concatenate([
        post.reshape(1), broadcast, recipe, since.reshape(1),
        blind.reshape(1),
        (mem[9] + pressed.astype(jnp.int32)).reshape(1),
        override.reshape(1),
        jnp.where(partner_here & at_post, partner_cell, -1).reshape(1),
        still.reshape(1), since_dish.reshape(1), idle_budget.reshape(1),
        stale.reshape(1)]).astype(jnp.int32)

    trace = {'pos': pos.astype(jnp.int32),
             'recipe': recipe.astype(jnp.int32),
             'pressed': pressed.astype(jnp.int32),
             'presses': nxt[9]}
    return action.astype(jnp.int32), nxt, trace


# --------------------------------------------------------------------------
# agent_1 -- inner ring cook
# memory: [target, known, last_delivered, wait, ever_seen, mute, delivered,
#          dish_wait, blind_wait, signals, since_display, redundant, stashed]
# --------------------------------------------------------------------------

def init_agent_1():
    return jnp.asarray(INIT_1, jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    del rng
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, jnp.asarray(INIT_1, jnp.int32), memory)

    pos = _localize(obs, REGION_1)
    board = _global_view(obs, pos, REGION_1)
    facing = jnp.argmax(obs[2, 2, 1:5])

    target = mem[0]
    known = mem[1]
    last_delivered = mem[2]
    wait = mem[3]
    ever_seen = mem[4]
    mute = mem[5]
    dish_wait = mem[7]
    blind_wait = mem[8]
    signals = mem[9]

    # ---- own inventory ----------------------------------------------------
    held_plate = obs[2, 2, 5]
    held_cooked = obs[2, 2, 6]
    held_counts = obs[2, 2, 7:10]
    held_total = held_counts.sum()
    has_dish = (held_plate == 1) & (held_cooked == 1)
    has_plate = (held_plate == 1) & (held_cooked == 0)
    has_ing = (held_plate == 0) & (held_total > 0)
    dish_type = jnp.argmax(held_counts)

    # ---- pot --------------------------------------------------------------
    pot = _cell(board, [4, 4])
    pot_counts = pot[31:34]
    pot_total = pot_counts.sum()
    pot_cooked = pot[30]
    pot_timer = pot[39]
    pot_type = jnp.argmax(pot_counts)
    pot_busy = (pot_timer > 0) | (pot_cooked == 1)
    pot_idle = ~pot_busy

    # ---- recipe evidence --------------------------------------------------
    display = _display(board, DISPLAY_CELLS)
    display_any = display.sum() > 0
    # How talkative is this partner?  Counting distinct activations, and how
    # long ago the last one was, is what tells us later whether its silence
    # carries information or merely means it is not signalling at all.
    edge = display_any & (mem[10] > 0)
    signals = jnp.where(edge, jnp.minimum(signals + 1, 9), signals)
    since_display = jnp.where(display_any, jnp.int32(0),
                              jnp.minimum(mem[10] + 1, 900))
    # An activation that tells us something we already knew means the partner
    # signals on its own schedule rather than only when the recipe moves; two
    # of those and we stop reading anything into its silence.
    redundant = jnp.where(edge & (known == 1) & (jnp.argmax(display) == target),
                          jnp.minimum(mem[11] + 1, 9), mem[11])
    target = jnp.where(display_any, jnp.argmax(display), target)
    known = jnp.where(display_any, jnp.int32(1), known)
    ever_seen = jnp.where(display_any, jnp.int32(1), ever_seen)
    mute = jnp.where(display_any, jnp.int32(0), mute)
    hunting = ever_seen == 0

    # Weaker evidence: a raw ingredient a partner left on a counter, which is
    # a plausible way for a partner to point at a recipe without paying the
    # button cost.  Only trusted while no display has ever appeared, and only
    # while we have not stashed a rejected ingredient ourselves -- otherwise we
    # would read our own discard back as the partner's advice.
    counter_item = ((obs[:, :, 20] == 1) & (obs[:, :, 29] == 0)
                    & (obs[:, :, 31:34].sum(axis=-1) > 0))
    hint = jnp.max(jnp.where(counter_item[..., None], obs[:, :, 31:34], 0),
                   axis=(0, 1))
    use_hint = (hunting & (mem[12] == 0) & (known == 0)
                & ((hint[0] + hint[1]) > 0))
    target = jnp.where(use_hint, jnp.argmax(hint[:2]), target)

    # ---- last resort: alternate, because a wrong delivery does not resample
    # Waiting costs one step per step; committing to the wrong ingredient
    # costs a whole 38-step cycle plus the walk to stash the useless dish, so
    # waiting a couple of dozen steps for evidence is the cheaper gamble.  Only
    # a partner that has stayed silent for BLIND_GIVEUP cumulative steps of
    # waiting is written off, after which we play the open-loop alternation.
    trust_silence = ((signals >= SIGNALS_TRUSTED) & (redundant < 2)
                     & (since_display <= SILENCE_TRUSTED))
    # Silence is only ever used to pick *which* way to guess.  Shortening the
    # wait on the strength of it, or treating the conclusion as firm enough to
    # deliver on unchecked, cost 115 against one probe broadcaster whose
    # presses merely happened not to look redundant: the inference is a
    # tie-breaker, not evidence.
    limit = jnp.where(mute == 1, jnp.int32(0), jnp.int32(WAIT_LIMIT))
    listening = ((held_total == 0) & (held_plate == 0) & (pot_total == 0)
                 & pot_idle & (known == 0))
    timed_out = listening & (wait >= limit)
    # What does the partner's silence mean?  If it has activated the display
    # several times, recently, and never merely to repeat something we already
    # knew, it signals when the recipe moves, so nothing appearing after our
    # delivery is evidence of a resample to the same recipe: repeat.  If it has
    # barely signalled, or signals on its own clock, silence says nothing, and
    # alternating is the better open-loop play -- a wrong delivery leaves the
    # recipe alone, so the flipped guess is then certain to be right.
    guess = jnp.where(trust_silence, last_delivered, 1 - last_delivered)
    target = jnp.where(timed_out, guess, target)
    wait = jnp.where(listening, jnp.minimum(wait + 1, 900), wait)
    blind_wait = jnp.where(hunting & (listening | (has_dish & (known == 0))),
                           jnp.minimum(blind_wait + 1, 900), blind_wait)
    mute = jnp.where(hunting & (blind_wait >= BLIND_GIVEUP), jnp.int32(1), mute)

    # ---- free counters for stashing ---------------------------------------
    occupied = jnp.stack([(_cell(board, rc)[29:34].sum() > 0).astype(jnp.int32)
                          for rc in CNT_CELLS])
    act_stash = _approach_best(pos, CNT_STANDS, CNT_DIRS, facing,
                               penalty=occupied * 50)

    # ---- carrying a plated dish -------------------------------------------
    # A known recipe stays valid until we ourselves deliver, because only a
    # correct delivery resamples it.  So a dish backed by a reading is safe to
    # hand in.  With no reading at all, waiting beside the delivery station --
    # from where both buttons are visible -- costs a few steps, whereas a bad
    # guess costs 40 relative to a good one.  A partner that never signals is
    # detected by the mute flag, and then we stop waiting and play the
    # open-loop alternation instead.
    unsure = known == 0
    still_waiting = unsure & (mute == 0) & (dish_wait < DISH_WAIT)
    deliverable = unsure | (target == dish_type)
    act_deliver = _approach_best(pos, DELIV_STANDS, DELIV_DIRS, facing)
    act_dish = jnp.where(
        still_waiting,
        _walk_to(pos, _sweep_stand(blind_wait, hunting)),
        jnp.where(deliverable, act_deliver, act_stash))
    dish_wait = jnp.where(has_dish & unsure, jnp.minimum(dish_wait + 1, 900),
                          jnp.int32(0))

    # ---- carrying an empty plate ------------------------------------------
    act_scoop = _approach_best(pos, SCOOP_STANDS, SCOOP_DIRS, facing)
    act_plate = jnp.where(pot_cooked == 1, act_scoop,
                          jnp.where(act_scoop == 5, jnp.int32(4), act_scoop))
    act_plate = jnp.where(pot_busy, act_plate, act_stash)

    # ---- carrying a raw ingredient ----------------------------------------
    fits = jnp.where(pot_total == 0, dish_type == target, dish_type == pot_type)
    can_add = pot_idle & (pot_total < 3) & fits
    act_pot = _approach_best(pos, POT_STANDS, POT_DIRS, facing)
    act_ing = jnp.where(can_add, act_pot, act_stash)
    rejecting = has_ing & (~can_add)

    # ---- empty handed -----------------------------------------------------
    want = jnp.where(pot_total > 0, pot_type, target)
    supply = jnp.asarray(ING_STANDS, jnp.int32)[jnp.clip(want, 0, 1)]
    act_fetch_ing = _approach(pos, supply, LEFT, facing)
    act_fetch_plate = _approach_best(pos, PLATE_STANDS, PLATE_DIRS, facing)
    act_listen = _walk_to(pos, _sweep_stand(blind_wait, hunting))
    act_empty = jnp.where(pot_busy, act_fetch_plate,
                          jnp.where(listening & (~timed_out),
                                    act_listen, act_fetch_ing))

    action = jnp.where(has_dish, act_dish,
                       jnp.where(has_plate, act_plate,
                                 jnp.where(has_ing, act_ing, act_empty)))

    # ---- a delivery invalidates the recipe belief -------------------------
    delivering = has_dish & (~still_waiting) & deliverable & (action == 5)
    known = jnp.where(delivering, jnp.int32(0), known)
    last_delivered = jnp.where(delivering, dish_type, last_delivered)
    wait = jnp.where(delivering, jnp.int32(0), wait)
    dish_wait = jnp.where(delivering, jnp.int32(0), dish_wait)
    delivered = mem[6] + delivering.astype(jnp.int32)
    stashed = jnp.maximum(mem[12],
                          (rejecting & (action == 5)).astype(jnp.int32))

    nxt = jnp.stack([target, known, last_delivered, wait, ever_seen, mute,
                     delivered, dish_wait, blind_wait, signals,
                     since_display, redundant, stashed]).astype(jnp.int32)

    trace = {'pos': pos.astype(jnp.int32),
             'target': target.astype(jnp.int32),
             'known': known.astype(jnp.int32),
             'pot': pot_total.astype(jnp.int32),
             'delivered': delivered.astype(jnp.int32)}
    return action.astype(jnp.int32), nxt, trace
