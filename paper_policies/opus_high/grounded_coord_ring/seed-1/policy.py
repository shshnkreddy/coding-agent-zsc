"""Grounded coordination policy for overcooked_v2 / grounded_coord_ring.

Layout facts that drive the whole design (see context/environment):

    WWW2R2WWW      agent_0 floor: the 24-cell outer corridor ring
    W       W      agent_1 floor: the 8-cell ring around the pot
    W WWLWW W
    2 0   B 2      The two floor regions are DISCONNECTED, so the seats can
    RAXAP X R      never collide, never hand items over directly, and only
    2 1   B 2      interact through the four shared counters (2,3) (2,5)
    W WWLWW W      (6,3) (6,5), the two recipe buttons (2,4) (6,4) and the
    W       W      recipe itself.
    WWW2R2WWW

* agent_1 (inner ring) reaches both recipe ingredient piles, both plate
  piles, both delivery stations, the pot and both buttons.  It is the only
  seat that can cook, so it runs a complete self-sufficient cook loop.  It
  can never see a free always-on recipe indicator (`R`), so every new recipe
  must come either from an active button display (free if somebody else
  pressed it) or from its own button press (-5 team reward).
* agent_0 (outer corridor) reads `R` for free from 20 of its 24 cells but can
  never reach the pot.  Its only useful job is publishing the current recipe
  on button (2,4), which any partner that reads the recipe channels can
  consume.  It presses only when the recipe it reads has changed and no
  display is already active, so the team pays at most one press per delivery
  cycle, whichever seat happens to pay it.

Nothing here depends on the partner running this same policy: each seat acts
on its own local evidence (localized static layout, pot contents, visible
recipe displays, visible counter items) and falls back to self-sufficient
behaviour when the partner does nothing.  Both agents read the environment's
own recipe channels rather than any private signalling convention.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_ring_cook_publish',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 9,
    'max_width': 9,
    'num_ingredients': 3,
}

# ---------------------------------------------------------------- constants

A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
F_UP, F_DOWN, F_RIGHT, F_LEFT = 0, 1, 2, 3

# action that turns the agent towards facing id f (UP, DOWN, RIGHT, LEFT)
_ACT_FOR_FACE = [A_UP, A_DOWN, A_RIGHT, A_LEFT]

# observation channel offsets
CH_STATIC0 = 20          # wall, goal, pot, recipe_ind, button, plate_pile
CH_STATIC1 = 29          # end of the static block (piles 26..28 included)
CH_ITEM0 = 29            # grid item: plate, cooked, ing0, ing1, ing2
CH_REC0 = 34             # displayed recipe: plate, cooked, ing0, ing1, ing2
CH_TIMER = 39
CH_SEEN = 40             # appended by _globalize

_GRID = (
    "WWW2R2WWW",
    "W       W",
    "W WWLWW W",
    "2 0   B 2",
    "RAXAP X R",
    "2 1   B 2",
    "W WWLWW W",
    "W       W",
    "WWW2R2WWW",
)
_SYMBOLS = "WXPRLB012"

# agent_1 ring, clockwise, index 0..7
_INNER_CELLS = ((3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (4, 3))
# agent_0 corridor ring, clockwise, index 0..23
_OUTER_CELLS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
    (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
)

# ring step actions: forward moves to index+1, backward to index-1
_I_FWD = [A_RIGHT, A_RIGHT, A_DOWN, A_DOWN, A_LEFT, A_LEFT, A_UP, A_UP]
_I_BWD = [A_DOWN, A_LEFT, A_LEFT, A_UP, A_UP, A_RIGHT, A_RIGHT, A_DOWN]
_O_FWD = [A_RIGHT] * 6 + [A_DOWN] * 6 + [A_LEFT] * 6 + [A_UP] * 6
_O_BWD = [A_DOWN] + [A_LEFT] * 6 + [A_UP] * 6 + [A_RIGHT] * 6 + [A_DOWN] * 5

# agent_1 work stations: (ring index, facing that points at the object)
_DELIVER_I = [3, 7]                                  # (4,6) / (4,2)
_DELIVER_F = [F_RIGHT, F_LEFT]
_POTFILL_I = [3, 7, 1, 5]                            # prefer delivery-side cells
_POTFILL_F = [F_LEFT, F_RIGHT, F_DOWN, F_UP]
_POTINS_I = [1, 5, 3, 7]                             # prefer supply-wing cells
_POTINS_F = [F_DOWN, F_UP, F_LEFT, F_RIGHT]
_PLATE_I = [2, 4]                                    # (3,6) / (5,6)
_PLATE_F = [F_RIGHT, F_RIGHT]
_PRESS_I = [1, 5]                                    # buttons (2,4) / (6,4)
_PRESS_F = [F_UP, F_DOWN]
_CNT_I = [0, 2, 4, 6]                                # shared counter access
_CNT_F = [F_UP, F_UP, F_DOWN, F_DOWN]
_CNT_CELLS = ((2, 3), (2, 5), (6, 5), (6, 3))
_IND_CELLS = ((0, 4), (4, 0), (4, 8), (8, 4))        # always-on indicators

_OUTER_STATION = 3                                   # (1,4): reads R, presses (2,4)
_HINT_STATION = 2                                    # (1,3): reads R, serves (2,3)
_HINT_CELL = (2, 3)                                  # shared counter used for hints
_PARK_I = [0, 1]                                     # (1,1) / (1,2) face up
_PARK_F = [F_UP, F_UP]
_PARK_CELLS = ((0, 1), (0, 2))                       # invisible from the inner ring
_PILE_I = [22, 20]                                   # (3,1) / (5,1) face right
_PILE_F = [F_RIGHT, F_RIGHT]

HINT_SAFE_AGE = 2         # steps after own delivery before a stale hint is ruled out
HINT_FRESH_AGE = 1        # a hint that differs from the old recipe cannot be stale
HINT_CLASH_AGE = 3        # a disagreement this old is the partner's fault, not lag
HINT_CONFIRM_MAX = 2      # hint-only deliveries tolerated before paying to re-check
OUTER_STALE = 60          # steps without delivery evidence before helping again
OUTER_BLIND_TRIES = 2     # speculative activations before giving a mute partner up

_FILL_I = 3               # (4,5): pot on the left, delivery (4,6) on the right
_FILL_F = F_LEFT
_DELIV_F = F_RIGHT
_WATCH_I = 7              # (4,3): sees both buttons, one step from either pile

CYCLE_STEPS = 39          # measured cook-and-deliver cycle on this layout
MAX_STEPS = 400

INNER_PATIENCE = 4        # steps of listening before paying for a press
INNER_PATIENCE_T0 = 9     # longer at episode start: partner may still be walking
PRESS_LEAD = 5            # pot timer at which the in-cook press is made
DISPLAY_LIFE = 9          # observations an activation stays visible for
OUTER_WATCHDOG = 25       # display-quiet steps before helping a silent partner

# ---------------------------------------------------------------- helpers


def _padded_static():
    """13x13x9 static one-hot planes with canvas[g + 2] == global cell g."""
    plane = []
    for row in range(-2, 11):
        line = []
        for col in range(-2, 11):
            vec = [0] * 9
            if 0 <= row < 9 and 0 <= col < 9:
                index = _SYMBOLS.find(_GRID[row][col])
                if index >= 0:
                    vec[index] = 1
            line.append(vec)
        plane.append(line)
    return plane


def _crops(cells):
    """Static 5x5x9 crops of each candidate cell, off-map padded with zeros."""
    plane = _padded_static()
    stack = []
    for row, col in cells:
        stack.append([[plane[row + i][col + j] for j in range(5)] for i in range(5)])
    return jnp.asarray(stack, dtype=jnp.int32)


def _localize(obs, cells):
    """Unique ring index from the static channels (all 32 crops differ)."""
    static = obs[:, :, CH_STATIC0:CH_STATIC1]
    cost = jnp.abs(_crops(cells) - static[None]).sum(axis=(1, 2, 3))
    return jnp.argmin(cost).astype(jnp.int32)


def _globalize(obs, cells, pos):
    """Place the local crop on a global canvas: canvas[g + 2] == global cell g."""
    rows = jnp.asarray([r for r, _ in cells], dtype=jnp.int32)
    cols = jnp.asarray([c for _, c in cells], dtype=jnp.int32)
    seen = jnp.ones(obs.shape[:2] + (1,), dtype=obs.dtype)
    ext = jnp.concatenate([obs, seen], axis=-1)
    canvas = jnp.zeros((13, 13, ext.shape[-1]), dtype=obs.dtype)
    return jax.lax.dynamic_update_slice(
        canvas, ext, (rows[pos], cols[pos], jnp.asarray(0, dtype=jnp.int32)))


def _at(canvas, row, col):
    return canvas[row + 2, col + 2]


def _display(cell):
    """(active, ingredient id) of a recipe display at this cell, if visible."""
    counts = cell[CH_REC0 + 2:CH_REC0 + 5]
    active = (cell[CH_SEEN] > 0) & (counts.sum() > 0)
    return active, jnp.argmax(counts).astype(jnp.int32)


def _pick(pos, idxs, faces, penalty, size):
    """Nearest station on the ring, biased by an integer penalty."""
    targets = jnp.asarray(idxs, dtype=jnp.int32)
    delta = jnp.mod(targets - pos, size)
    dist = jnp.minimum(delta, size - delta)
    choice = jnp.argmin(penalty * 1000 + dist)
    return targets[choice], jnp.asarray(faces, dtype=jnp.int32)[choice]


def _nav(pos, face, goal_idx, goal_face, goal_act, fwd, bwd, size):
    """Walk around the ring; once in place, turn and then run goal_act."""
    delta = jnp.mod(goal_idx - pos, size)
    move = jnp.where(delta <= size // 2,
                     jnp.asarray(fwd, dtype=jnp.int32)[pos],
                     jnp.asarray(bwd, dtype=jnp.int32)[pos])
    turn = jnp.asarray(_ACT_FOR_FACE, dtype=jnp.int32)[goal_face]
    return jnp.where(delta == 0,
                     jnp.where(face == goal_face, goal_act, turn),
                     move).astype(jnp.int32)


def _sel(cond, a, b):
    return (jnp.where(cond, a[0], b[0]),
            jnp.where(cond, a[1], b[1]),
            jnp.where(cond, a[2], b[2]))


def _ing_station(kind):
    """Access cell for ingredient pile 0 at (3,2) / pile 1 at (5,2)."""
    return (jnp.where(kind == 1, jnp.asarray(6, dtype=jnp.int32),
                      jnp.asarray(0, dtype=jnp.int32)),
            jnp.asarray(F_LEFT, dtype=jnp.int32))


def _init_memory():
    # [0] recipe (-1 unknown)        [6] free-information credit
    # [1] step counter               [7] steps since my last delivery
    # [2] wait (or display-quiet)    [8] counter-hint credibility
    # [3] press cooldown             [9] recipe of the previous cycle
    # [4] paid inside this cook      [10] unconfirmed hint-based deliveries
    # [5] life left of a display I caused / published-this-cycle latch
    # [11] a display was seen during this cycle
    # [12] free information was available at some point in this cycle
    return jnp.asarray([-1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], dtype=jnp.int32)


def _facing(me):
    return jnp.argmax(me[1:5]).astype(jnp.int32)


def _int(value):
    return jnp.asarray(value, dtype=jnp.int32)


# ---------------------------------------------------------------- agent_1


def init_agent_1():
    return _init_memory()


def agent_1_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, _init_memory(), memory)
    rec_mem, step_t, wait, cooldown = mem[0], mem[1], mem[2], mem[3]
    prepressed, mine_left, free_credit, age = mem[4], mem[5], mem[6], mem[7]
    hint_cred, prev_rec, unconfirmed, cycle_disp = mem[8], mem[9], mem[10], mem[11]

    pos = _localize(obs, _INNER_CELLS)
    canvas = _globalize(obs, _INNER_CELLS, pos)
    me = obs[2, 2]
    face = _facing(me)

    # the pot at (4,4) is inside the view from every cell of the inner ring
    pot = _at(canvas, 4, 4)
    pot_raw = pot[CH_ITEM0 + 2:CH_ITEM0 + 5]
    n_pot = pot_raw.sum()
    pot_kind = jnp.argmax(pot_raw).astype(jnp.int32)
    pot_cooked = pot[CH_ITEM0 + 1] > 0
    pot_timer = pot[CH_TIMER]
    pot_busy = pot_cooked | (pot_timer > 0)

    # recipe: an active button display always shows the *current* recipe
    on_top, kind_top = _display(_at(canvas, 2, 4))
    on_bot, kind_bot = _display(_at(canvas, 6, 4))
    disp_on = on_top | on_bot
    disp_kind = jnp.where(on_top, kind_top, kind_bot)

    # own inventory
    inv_plate = me[5] > 0
    inv_cooked = me[6] > 0
    inv_raw = me[7:10]
    n_inv = inv_raw.sum()
    inv_kind = jnp.argmax(inv_raw).astype(jnp.int32)
    has_dish = inv_plate & inv_cooked
    has_plate = inv_plate & jnp.logical_not(inv_cooked)
    has_ing = jnp.logical_not(inv_plate) & (n_inv > 0)

    # shared counters
    cnt = jnp.stack([_at(canvas, r, c) for r, c in _CNT_CELLS])
    cnt_seen = cnt[:, CH_SEEN] > 0
    cnt_used = cnt[:, CH_ITEM0:CH_ITEM0 + 5].sum(axis=1) > 0
    cnt_free = cnt_seen & jnp.logical_not(cnt_used)
    cnt_plate = (cnt_seen & (cnt[:, CH_ITEM0] > 0) & (cnt[:, CH_ITEM0 + 1] == 0)
                 & (cnt[:, CH_ITEM0 + 2:CH_ITEM0 + 5].sum(axis=1) == 0))

    # ---- free grounded channel: a raw ingredient left on a shared counter.
    # A partner in the corridor can read the recipe for nothing, so an item it
    # puts within our reach is evidence about the recipe.  Only ids 0 and 1 can
    # ever be a recipe here, and two different ids at once carry no message.
    cnt_raw = cnt[:, CH_ITEM0 + 2:CH_ITEM0 + 5]
    cnt_is_raw = (cnt_seen & (cnt[:, CH_ITEM0] == 0) & (cnt[:, CH_ITEM0 + 1] == 0)
                  & (cnt_raw.sum(axis=1) > 0))
    cnt_kind = jnp.argmax(cnt_raw, axis=1)
    hint_0 = jnp.any(cnt_is_raw & (cnt_kind == 0))
    hint_1 = jnp.any(cnt_is_raw & (cnt_kind == 1))
    hint_any = hint_0 | hint_1
    hint_clash = hint_0 & hint_1
    hint_kind = jnp.where(hint_0, _int(0), _int(1))
    hint_usable = hint_any & jnp.logical_not(hint_clash)

    # Credibility: a display is ground truth, so agreement earns trust and a
    # disagreement that survives the partner's removal window destroys it.
    agree = disp_on & hint_usable & (hint_kind == disp_kind)
    clash = disp_on & hint_usable & (hint_kind != disp_kind) & (age >= HINT_CLASH_AGE)
    cred_next = jnp.where(clash, _int(2),
                          jnp.where(agree & (hint_cred != 2), _int(1), hint_cred))
    # A hint equal to the recipe we just delivered might simply be left over, so
    # it is only believed once the partner has had a step to withdraw it.  One
    # that differs cannot be a leftover at all.
    needed_age = jnp.where((prev_rec >= 0) & (hint_kind != prev_rec),
                           _int(HINT_FRESH_AGE), _int(HINT_SAFE_AGE))
    # After a few deliveries decided by hints alone, ignore them for one cycle
    # and buy a display instead: it re-checks the hint for a whole 9 steps and
    # bounds how long a misleading partner can cost us.
    must_confirm = unconfirmed >= HINT_CONFIRM_MAX
    trust_hint = (hint_usable & (hint_cred == 1) & (age >= needed_age)
                  & jnp.logical_not(must_confirm))

    rec_known = disp_on | trust_hint | (rec_mem >= 0)
    rec_kind = jnp.where(disp_on, disp_kind,
                         jnp.where(rec_mem >= 0, jnp.maximum(rec_mem, 0), hint_kind))

    # candidate stations.  Filling the plate and delivering both happen at
    # (4,5), so the cook always ends facing the pot and then the hatch.
    zero2 = jnp.zeros(2, dtype=jnp.int32)
    zero4 = jnp.zeros(4, dtype=jnp.int32)
    fill_i, fill_f = _int(_FILL_I), _int(_FILL_F)
    dv_i, dv_f = _int(_FILL_I), _int(_DELIV_F)
    ins_i, ins_f = _pick(pos, _POTINS_I, _POTINS_F, zero4, 8)
    pile_i, pile_f = _pick(pos, _PLATE_I, _PLATE_F, zero2, 8)
    # Two button choices.  While waiting for the next recipe, walk to the
    # button of the wing the recipe probably needs (a visible counter hint, or
    # failing that the previous recipe): both buttons are the same distance
    # from the delivery cell, but the piles are not, so guessing right saves
    # three steps and guessing wrong costs nothing but the walk.  During the
    # cook, use whichever button is nearest so the detour stays inside the
    # idle window.
    # Always use the top button (2,4).  Both buttons are the same distance from
    # the delivery cell, but (2,4) is the only one a corridor partner can see
    # (so it learns that we inform ourselves and stops paying too), and the
    # whole route to it keeps the shared counters in view, so a free hint is
    # still picked up on the way.  The detour from the lower wing fits inside
    # the idle part of the cook.
    press_i, press_f = _int(_PRESS_I[0]), _int(_PRESS_F[0])
    wait_i, wait_f = press_i, press_f
    # free counter: prefer a seen-empty one, then an unseen one, avoid occupied
    drop_pen = jnp.where(cnt_free, 0, jnp.where(cnt_seen, 5, 1))
    drop_i, drop_f = _pick(pos, _CNT_I, _CNT_F, drop_pen, 8)
    take_i, take_f = _pick(pos, _CNT_I, _CNT_F, jnp.where(cnt_plate, 0, 9), 8)
    staged_plate = jnp.any(cnt_plate)

    # ---- empty hands: get information, then ingredients, then a plate
    # While the recipe is unknown, walk to the button and listen on the way:
    # the whole route keeps both the button cell and the shared counters in
    # view, so a partner activation or a counter hint is still caught for free
    # and nothing is lost if it never comes.  Information is worthless if no
    # further cycle fits in the horizon.
    useful = step_t + CYCLE_STEPS <= MAX_STEPS
    patience = jnp.where(step_t < 14, INNER_PATIENCE_T0, INNER_PATIENCE)
    ready = (wait >= patience) & (cooldown == 0) & useful
    goal = (wait_i, wait_f, jnp.where(ready, A_INTERACT, A_STAY))

    fetch_i, fetch_f = _ing_station(rec_kind)
    goal = _sel(rec_known & (rec_kind <= 1) & (n_pot == 0)
                & jnp.logical_not(pot_busy),
                (fetch_i, fetch_f, _int(A_INTERACT)), goal)
    # a partially filled pot dictates what to fetch next, recipe or not
    more_i, more_f = _ing_station(pot_kind)
    goal = _sel((n_pot > 0) & jnp.logical_not(pot_busy) & (pot_kind <= 1),
                (more_i, more_f, _int(A_INTERACT)), goal)
    # cooking or cooked: fetch a plate (a staged one from a counter if present)
    plate_goal = _sel(staged_plate,
                      (take_i, take_f, _int(A_INTERACT)),
                      (pile_i, pile_f, _int(A_INTERACT)))
    goal = _sel(pot_busy, plate_goal, goal)

    # In-cook press.  A display always shows the *current* recipe and survives
    # DISPLAY_LIFE observations, so pressing with ~PRESS_LEAD timer left makes
    # the recipe that gets resampled by this delivery readable immediately
    # afterwards, at no cost in steps.  Skipped while the partner is supplying
    # that information for free.
    # the served cycle ends about PRESS_LEAD + 2 + CYCLE_STEPS steps from here
    prepress_useful = step_t + PRESS_LEAD + 2 + CYCLE_STEPS <= MAX_STEPS
    want_prepress = ((free_credit <= 0) & (prepressed == 0)
                     & (pot_timer > 0) & prepress_useful)
    prepress_act = jnp.where((pot_timer <= PRESS_LEAD) & (cooldown == 0),
                             A_INTERACT, A_STAY)
    goal = _sel(want_prepress, (press_i, press_f, prepress_act), goal)

    # ---- holding a raw ingredient
    can_insert = (jnp.logical_not(pot_busy) & (n_pot < 3)
                  & ((n_pot == 0) | (pot_kind == inv_kind)))
    want_insert = can_insert & (jnp.logical_not(rec_known) | (rec_kind == inv_kind)
                                | (n_pot > 0))
    goal_ing = _sel(want_insert,
                    (ins_i, ins_f, _int(A_INTERACT)),
                    (drop_i, drop_f, _int(A_INTERACT)))

    # ---- holding an empty plate
    goal_plate = _sel(pot_cooked,
                      (fill_i, fill_f, _int(A_INTERACT)),
                      _sel(pot_timer > 0,
                           (fill_i, fill_f, _int(A_STAY)),
                           (drop_i, drop_f, _int(A_INTERACT))))

    # ---- holding a plated dish: deliver only what matches, else park it
    dish_ok = jnp.logical_not(rec_known) | (inv_raw[rec_kind] >= 3)
    goal_dish = _sel(dish_ok,
                     (dv_i, dv_f, _int(A_INTERACT)),
                     (drop_i, drop_f, _int(A_INTERACT)))

    goal = _sel(has_ing, goal_ing, goal)
    goal = _sel(has_plate, goal_plate, goal)
    goal = _sel(has_dish, goal_dish, goal)

    action = _nav(pos, face, goal[0], goal[1], goal[2], _I_FWD, _I_BWD, 8)

    at_delivery = ((pos == 3) & (face == F_RIGHT)) | ((pos == 7) & (face == F_LEFT))
    at_button = (pos == _PRESS_I[0]) & (face == _PRESS_F[0])
    delivered = (action == A_INTERACT) & has_dish & at_delivery
    pressed = ((action == A_INTERACT) & at_button
               & jnp.logical_not(has_dish | has_plate | has_ing))

    # a correct delivery resamples the recipe, so the cached value expires
    rec_next = jnp.where(
        delivered, _int(-1),
        jnp.where(disp_on, disp_kind,
                  jnp.where((rec_mem < 0) & trust_hint, hint_kind, rec_mem)))

    # Did the recipe for this new cycle arrive without us paying for it?  A
    # display we lit ourselves does not count, which is what mine_left tracks.
    # Does a free channel exist at all?  Judge it over the whole cycle, not at
    # one instant: after a change the partner needs a few steps to replace its
    # hint, and that lag does not mean the channel is dead.
    free_now = (disp_on & (mine_left == 0)) | trust_hint
    saw_free = jnp.maximum(mem[12], free_now.astype(jnp.int32))
    credit_next = jnp.where(
        delivered,
        jnp.clip(free_credit + jnp.where(saw_free > 0, 1, -1), 0, 2),
        free_credit)
    # a cycle whose recipe rested on a hint that no display ever corroborated
    seen_disp = jnp.maximum(cycle_disp, (disp_on & (age >= 1)).astype(jnp.int32))
    unconf_next = jnp.where(delivered,
                            jnp.where(seen_disp > 0, _int(0),
                                      jnp.minimum(unconfirmed + 1, 9)),
                            unconfirmed)

    mem_next = jnp.stack([
        rec_next,
        step_t + 1,
        jnp.where(rec_next >= 0, _int(0), wait + 1),
        jnp.where(pressed, _int(4), jnp.maximum(cooldown - 1, 0)),
        jnp.where(delivered, _int(0),
                  jnp.maximum(prepressed, (pressed & (pot_timer > 0)).astype(jnp.int32))),
        jnp.where(pressed, _int(DISPLAY_LIFE), jnp.maximum(mine_left - 1, 0)),
        credit_next,
        jnp.where(delivered, _int(0), jnp.minimum(age + 1, 60)),
        cred_next,
        jnp.where(delivered, rec_kind, prev_rec),
        unconf_next,
        jnp.where(delivered, _int(0), seen_disp),
        jnp.where(delivered, _int(0), saw_free),
    ]).astype(jnp.int32)

    trace = {
        'pos': pos,
        'recipe': rec_kind.astype(jnp.int32),
        'recipe_known': rec_known.astype(jnp.int32),
        'pot_n': n_pot.astype(jnp.int32),
        'pot_timer': pot_timer.astype(jnp.int32),
        'delivered': delivered.astype(jnp.int32),
        'pressed': pressed.astype(jnp.int32),
        'display': disp_on.astype(jnp.int32),
        'credit': credit_next.astype(jnp.int32),
        'hint': jnp.where(hint_usable, hint_kind, _int(-1)).astype(jnp.int32),
        'hint_cred': cred_next.astype(jnp.int32),
        'trust': trust_hint.astype(jnp.int32),
        'unconfirmed': unconf_next.astype(jnp.int32),
    }
    return action.astype(jnp.int32), mem_next, trace


# ---------------------------------------------------------------- agent_0


def init_agent_0():
    return _init_memory()


def agent_0_act(obs, reset, memory, rng):
    del rng
    mem = jnp.where(reset, _init_memory(), memory)
    rec_mem, step_t, need, cooldown = mem[0], mem[1], mem[2], mem[3]
    quiet, mine_left, partner_informs = mem[4], mem[5], mem[6]

    pos = _localize(obs, _OUTER_CELLS)
    canvas = _globalize(obs, _OUTER_CELLS, pos)
    me = obs[2, 2]
    face = _facing(me)

    # free ground truth: any visible always-on recipe indicator
    ind = jnp.stack([_at(canvas, r, c) for r, c in _IND_CELLS])
    ind_counts = ind[:, CH_REC0 + 2:CH_REC0 + 5]
    ind_on = (ind[:, CH_SEEN] > 0) & (ind_counts.sum(axis=1) > 0)
    ind_any = jnp.any(ind_on)
    ind_kind = jnp.argmax(ind_counts, axis=1).astype(jnp.int32)
    ind_pick = ind_kind[jnp.argmax(ind_on.astype(jnp.int32))]

    # has the recipe already been published on a button display?
    on_top, kind_top = _display(_at(canvas, 2, 4))
    on_bot, kind_bot = _display(_at(canvas, 6, 4))
    disp_on = on_top | on_bot
    disp_kind = jnp.where(on_top, kind_top, kind_bot)

    rec_kind = jnp.where(ind_any, ind_pick,
                         jnp.where(disp_on, disp_kind, jnp.maximum(rec_mem, 0)))
    rec_known = ind_any | disp_on | (rec_mem >= 0)
    changed = ind_any & ((rec_mem < 0) | (ind_pick != rec_mem))
    # a transition between two *known* recipes is the fingerprint of a delivery
    real_change = ind_any & (rec_mem >= 0) & (ind_pick != rec_mem)

    # A display that our own press did not cause proves the partner publishes
    # for itself; paying again would only duplicate an unavoidable cost.  A
    # recipe change also proves it: the indicator only changes after a correct
    # delivery, so if we published nothing during that cycle the partner must
    # have obtained the recipe on its own.
    published = mem[7]
    informs_next = jnp.maximum(
        partner_informs,
        ((disp_on & (mine_left == 0)) | (real_change & (published == 0))
         ).astype(jnp.int32))

    # Publish when the recipe changed (a delivery just happened) and nobody has
    # an active display.  The watchdog covers deliveries that resample the same
    # recipe, where the indicator alone gives no evidence at all.
    # Deliveries, not displays, are the evidence that the partner is coping: a
    # long silence with no recipe change at all means it is probably waiting for
    # information it cannot get, which is worth paying for again.
    stale_next = jnp.where(real_change, _int(0), jnp.minimum(mem[8] + 1, 999))
    quiet_next = jnp.where(disp_on, _int(0), quiet + 1)
    # Speculative help is only worth paying for a couple of times: if no
    # delivery has ever happened, publishing the recipe again is not what the
    # partner is missing, and each attempt costs the team 5.
    ever_deliv = jnp.maximum(mem[9], real_change.astype(jnp.int32))
    blind_tries = mem[10]
    watchdog = ((stale_next >= OUTER_STALE)
                & ((ever_deliv > 0) | (blind_tries < OUTER_BLIND_TRIES)))
    need_now = jnp.logical_not(disp_on) & ((need > 0) | changed | watchdog)
    horizon_ok = step_t + CYCLE_STEPS <= MAX_STEPS
    want_press = (need_now & rec_known & (cooldown == 0) & (informs_next == 0)
                  & horizon_ok)

    # ---- free channel: keep one raw unit of the current recipe on the shared
    # counter (2,3) and take it back the moment the indicator changes, so its
    # presence always means "the recipe is this ingredient".  The unused type
    # waits on (0,1)/(0,2), which no cell of the inner ring can see, so it
    # never muddles the message.
    hint = _at(canvas, _HINT_CELL[0], _HINT_CELL[1])
    h_seen = hint[CH_SEEN] > 0
    h_total = hint[CH_ITEM0:CH_ITEM0 + 5].sum()
    h_raw = hint[CH_ITEM0 + 2:CH_ITEM0 + 5]
    h_is_raw = (hint[CH_ITEM0] == 0) & (hint[CH_ITEM0 + 1] == 0) & (h_raw.sum() > 0)
    h_kind = jnp.argmax(h_raw).astype(jnp.int32)
    h_free = h_seen & (h_total == 0)
    h_blocked = h_seen & (h_total > 0) & jnp.logical_not(h_is_raw)
    h_right = h_seen & h_is_raw & rec_known & (h_kind == rec_kind)
    h_wrong = h_seen & h_is_raw & rec_known & (h_kind != rec_kind)

    inv_raw = me[7:10]
    holding = (inv_raw.sum() > 0) & (me[5] == 0)
    hold_kind = jnp.argmax(inv_raw).astype(jnp.int32)
    hands_free = (me[5] == 0) & (me[6] == 0) & (inv_raw.sum() == 0)

    park = jnp.stack([_at(canvas, r, c) for r, c in _PARK_CELLS])
    park_raw = park[:, CH_ITEM0 + 2:CH_ITEM0 + 5]
    park_seen = park[:, CH_SEEN] > 0
    park_is_raw = ((park[:, CH_ITEM0] == 0) & (park[:, CH_ITEM0 + 1] == 0)
                   & (park_raw.sum(axis=1) > 0))
    park_kind = jnp.argmax(park_raw, axis=1)
    park_open = park_seen & (park[:, CH_ITEM0:CH_ITEM0 + 5].sum(axis=1) == 0)
    park_has = park_is_raw & park_seen & (park_kind == rec_kind)
    put_i, put_f = _pick(pos, _PARK_I, _PARK_F,
                         jnp.where(park_open, 0, jnp.where(park_seen, 9, 1)), 24)
    take_i, take_f = _pick(pos, _PARK_I, _PARK_F, jnp.where(park_has, 0, 9), 24)
    pile_i = jnp.where(rec_kind == 1, _int(_PILE_I[1]), _int(_PILE_I[0]))
    get_i = jnp.where(jnp.any(park_has), take_i, pile_i)
    get_f = _int(F_RIGHT)
    get_f = jnp.where(jnp.any(park_has), take_f, get_f)

    station = _int(_OUTER_STATION)
    hint_post = _int(_HINT_STATION)
    publishable = holding & (hold_kind == rec_kind) & (h_free | jnp.logical_not(h_seen))

    goal = (hint_post, _int(F_DOWN), _int(A_STAY))
    goal = _sel(hands_free & rec_known & jnp.logical_not(h_blocked | h_right),
                (get_i, get_f, _int(A_INTERACT)), goal)
    goal = _sel(publishable, (hint_post, _int(F_DOWN), _int(A_INTERACT)), goal)
    goal = _sel(holding & jnp.logical_not(publishable),
                (put_i, put_f, _int(A_INTERACT)), goal)
    goal = _sel(hands_free & h_wrong, (hint_post, _int(F_DOWN), _int(A_INTERACT)), goal)
    goal = _sel(want_press & hands_free,
                (station, _int(F_DOWN), _int(A_INTERACT)), goal)

    action = _nav(pos, face, goal[0], goal[1], goal[2], _O_FWD, _O_BWD, 24)
    pressed = (action == A_INTERACT) & (pos == station) & (face == F_DOWN)

    mem_next = jnp.stack([
        jnp.where(ind_any, ind_pick, jnp.where(disp_on, disp_kind, rec_mem)),
        step_t + 1,
        jnp.where(pressed | disp_on, _int(0), need_now.astype(jnp.int32)),
        jnp.where(pressed, _int(4), jnp.maximum(cooldown - 1, 0)),
        jnp.where(pressed, _int(0), quiet_next),
        jnp.where(pressed, _int(DISPLAY_LIFE), jnp.maximum(mine_left - 1, 0)),
        informs_next,
        jnp.where(pressed, _int(1), jnp.where(real_change, _int(0), published)),
        jnp.where(pressed, _int(0), stale_next),
        ever_deliv,
        jnp.where(pressed & (ever_deliv == 0), blind_tries + 1, blind_tries),
        mem[11], mem[12],
    ]).astype(jnp.int32)

    trace = {
        'pos': pos,
        'recipe': rec_kind.astype(jnp.int32),
        'recipe_known': rec_known.astype(jnp.int32),
        'changed': changed.astype(jnp.int32),
        'pressed': pressed.astype(jnp.int32),
        'display': disp_on.astype(jnp.int32),
        'quiet': quiet_next.astype(jnp.int32),
        'informs': informs_next.astype(jnp.int32),
        'hint_up': h_right.astype(jnp.int32),
        'holding': holding.astype(jnp.int32),
    }
    return action.astype(jnp.int32), mem_next, trace
