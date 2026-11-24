"""Cooperative Overcooked-V2 policy for the disconnected two-room layout.

The board is split into two rooms that share only the pot (2,7) and the two
counter cells (1,7)/(3,7):

    WWWWWR2W0WW
    0      W  B
    W     APA X
    1      W  B
    WWWWWR2W1WW

Only the left room can see a recipe indicator (R at (0,5)/(4,5)); only the
right room can reach the plate piles (1,10)/(3,10) and the delivery station
(2,10).  Both rooms can reach ingredient supplies 0 and 1 and can interact with
the pot.  The two floor regions are disconnected, so seats never collide.

The roles are therefore forced by geometry rather than by any private
convention:

* left-room seat  : reads the displayed recipe and puts the first ingredient of
  the required type into the pot.  That insertion is also the only
  convention-free way to tell the right room which recipe is displayed.  It
  then keeps the pot topped up and restocks two private staging counters
  (0,4)/(4,4) so the next cycle can start quickly after a delivery.
* right-room seat : never starts a pot, because it cannot know the recipe.  It
  copies the ingredient type already in the pot, fetches a plate while the pot
  cooks, scoops, delivers dishes that are a pure stack of three units of one
  edible ingredient, and dumps anything mixed on a private counter.

The role is derived from self-localization, so either act function works in
either seat and neither seat depends on its partner running this same policy.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'room_division_cook',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# --------------------------------------------------------------------------
# static map knowledge (published layout geometry only)
# codes: 0 floor, 1 wall, 2 goal, 3 pot, 4 recipe indicator, 5 button,
#        6 plate pile, 7 ingredient 0, 8 ingredient 1, 9 ingredient 2
# --------------------------------------------------------------------------
_CODES = [
    [1, 1, 1, 1, 1, 4, 9, 1, 7, 1, 1],
    [7, 0, 0, 0, 0, 0, 0, 1, 0, 0, 6],
    [1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2],
    [8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 6],
    [1, 1, 1, 1, 1, 4, 9, 1, 8, 1, 1],
]
_FLOOR = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9),
]

# action ids
_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5

# ---- behaviour modes: (target cell, facing action, interact?) -------------
M_IDLE_L = 0    # wait at the pot cell, watching recipe / pot / partner
M_POT_L = 1     # insert the held ingredient into the pot
M_TAKE_S0 = 2   # take staged ingredient 0 from counter (0,4)
M_TAKE_S1 = 3   # take staged ingredient 1 from counter (4,4)
M_PUT_S0 = 4    # stage ingredient 0 on counter (0,4)
M_PUT_S1 = 5    # stage ingredient 1 on counter (4,4)
M_SUP0_L = 6    # take ingredient 0 from supply (1,0)
M_SUP1_L = 7    # take ingredient 1 from supply (3,0)
M_SUP2_L = 8    # take ingredient 2 from supply (0,6)
M_DUMP_L0 = 9   # drop an unusable item on private counter (0,3)
M_DUMP_L1 = 10  # drop an unusable item on private counter (0,2)
M_IDLE_R = 11   # wait beside the pot, facing it
M_POT_R = 12    # insert into / scoop out of the pot
M_ING0_R = 13   # take ingredient 0 from supply (0,8)
M_ING1_R = 14   # take ingredient 1 from supply (4,8)
M_PLATE_R = 15  # take a plate from pile (1,10)
M_DELIVER = 16  # deliver at the station (2,10)
M_DUMP_R0 = 17  # drop an unusable item on private counter (0,9)
M_DUMP_R1 = 18  # drop an unusable item on private counter (4,9)
M_SCOUT_L = 19  # walk into sight of a recipe indicator (start of episode)
M_DUMP_R2 = 20  # last-resort drop on shared counter (1,7)
M_DUMP_R3 = 21  # last-resort drop on shared counter (3,7)

# target cell, facing, and what to do on arrival (0 stay, 1 interact, 2 stay
# without turning).  Waiting posts use 2 so that they never step off target.
_MODE_ROW = [2, 2, 1, 3, 1, 3, 1, 3, 1, 1, 1, 2, 2, 1, 3, 1, 2, 1, 3, 2, 1, 3]
_MODE_COL = [5, 6, 4, 4, 4, 4, 1, 1, 6, 3, 2, 8, 8, 8, 8, 9, 9, 9, 9, 3, 8, 8]
_MODE_FACE = [_RIGHT, _RIGHT, _UP, _DOWN, _UP, _DOWN, _LEFT, _LEFT, _UP, _UP,
              _UP, _LEFT, _LEFT, _UP, _DOWN, _RIGHT, _RIGHT, _UP, _DOWN,
              _LEFT, _LEFT, _LEFT]
_MODE_ACT = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]

_POT_R, _POT_C = 2, 7
_SLOT0_R, _SLOT0_C = 0, 4   # staging counter for ingredient 0
_SLOT1_R, _SLOT1_C = 4, 4   # staging counter for ingredient 1

# memory: 0 step, 1 recipe(3=unknown), 2 pot_n, 3 pot_type(3=none), 4 pot_mixed,
# 5 pot_cooked, 6 pot_timer, 7 steps since the pot was emptied, 8 slot0,
# 9 slot1, 10 recoveries, 11 restock intent, 12 steps with a dead pot,
# 13 next solo ingredient, 14 solo latch, 15 observed delivery lag,
# 16 steps since the pot contents last changed,
# 17 steps since the partner was seen carrying something
_MEM0 = [0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 199]
_MAX_RECOVERY_FILLS = 4  # bounded clean-up of a mis-typed pot started elsewhere

# The right room cannot read the recipe, so it never starts a pot while the
# partner might still do it.  If the pot stays dead this long the partner is
# not loading at all, and cooking alternating single-ingredient dishes becomes
# the best available option: a wrong delivery leaves the recipe unchanged, so
# the following dish of the other type is certain to be correct.  That gives
# +20 on two deliveries out of three in expectation instead of nothing.
_SOLO_AFTER = 45
_SOLO_QUIET = 60


def _i32(x):
    return jnp.asarray(x, jnp.int32)


def _static_windows():
    """Reference 5x5 static crops for every reachable floor cell."""
    codes = _i32(_CODES)
    planes = jnp.stack([_i32(codes == k) for k in range(1, 10)], axis=-1)
    pad = jnp.zeros((9, 15, 9), jnp.int32).at[2:7, 2:13].set(planes)
    windows = jnp.stack([pad[r:r + 5, c:c + 5] for (r, c) in _FLOOR])
    return _i32(_FLOOR), windows


def _localize(obs):
    """Exact position: all 24 floor crops are distinct (min distance 4)."""
    cells, windows = _static_windows()
    static = obs[:, :, 20:29]
    cost = jnp.abs(windows - static[None]).sum(axis=(1, 2, 3))
    return cells[jnp.argmin(cost)]


def _cell(obs, pos, grow, gcol):
    """Channel vector of a global cell plus a visibility flag."""
    rr = grow - pos[0] + 2
    cc = gcol - pos[1] + 2
    ok = (rr >= 0) & (rr < 5) & (cc >= 0) & (cc < 5)
    return obs[jnp.clip(rr, 0, 4), jnp.clip(cc, 0, 4)], ok


def _navigate(pos, face_act, mode):
    """Walk to the mode's cell, turn to its facing, then act there."""
    trow = _i32(_MODE_ROW)[mode]
    tcol = _i32(_MODE_COL)[mode]
    tface = _i32(_MODE_FACE)[mode]
    want_interact = _i32(_MODE_ACT)[mode]

    drow = trow - pos[0]
    dcol = tcol - pos[1]
    vert = jnp.where(drow > 0, _DOWN, _UP)
    horiz = jnp.where(dcol > 0, _RIGHT, _LEFT)

    # every interaction target faces a blocking object, so a movement action
    # there only turns the agent.  Waiting posts (code 2) never turn, because
    # their facing may be free floor.
    arrived = jnp.where(
        want_interact == 2,
        _STAY,
        jnp.where(face_act != tface, tface,
                  jnp.where(want_interact > 0, _INTERACT, _STAY)),
    )
    # approach so that the final move already sets the wanted facing
    face_horizontal = (tface == _RIGHT) | (tface == _LEFT)
    vert_first = jnp.where(drow != 0, vert, jnp.where(dcol != 0, horiz, arrived))
    horiz_first = jnp.where(dcol != 0, horiz, jnp.where(drow != 0, vert, arrived))
    return _i32(jnp.where(face_horizontal, vert_first, horiz_first))


def _agent(obs, reset, memory, rng):
    obs = _i32(obs)
    mem = _i32(jnp.where(reset, _i32(_MEM0), memory))

    pos = _localize(obs)
    me = obs[2, 2]
    face_act = _i32([_UP, _DOWN, _RIGHT, _LEFT])[jnp.argmax(me[1:5])]

    held_plate = me[5] > 0
    held_cooked = me[6] > 0
    held_counts = me[7:10]
    held_total = held_counts.sum()
    held_type = _i32(jnp.argmax(held_counts))
    holding_dish = held_plate & held_cooked
    holding_plate = held_plate & (~held_cooked)
    holding_ing = (~held_plate) & (held_total > 0)

    # ---- displayed recipe (visible from the left room only) ----
    rec_counts = obs[:, :, 36:39].max(axis=0).max(axis=0)
    rec_visible = rec_counts.sum() > 0
    recipe = _i32(jnp.where(rec_visible, jnp.argmax(rec_counts), mem[1]))

    # ---- pot state; remembered and aged while out of view ----
    pot_vec, pot_visible = _cell(obs, pos, _POT_R, _POT_C)
    live_counts = pot_vec[31:34]
    live_n = live_counts.sum()
    live_type = _i32(jnp.where(live_n > 0, jnp.argmax(live_counts), 3))
    live_mixed = _i32((live_counts > 0).sum() > 1)
    live_cooked = _i32(pot_vec[30] > 0)
    live_timer = _i32(pot_vec[39])

    aged_timer = jnp.maximum(mem[6] - 1, 0)
    aged_cooked = jnp.where((mem[6] > 0) & (aged_timer == 0), 1, mem[5])

    pot_n = _i32(jnp.where(pot_visible, live_n, mem[2]))
    pot_type = _i32(jnp.where(pot_visible, live_type, mem[3]))
    pot_mixed = _i32(jnp.where(pot_visible, live_mixed, mem[4]))
    pot_cooked = _i32(jnp.where(pot_visible, live_cooked, aged_cooked))
    pot_timer = _i32(jnp.where(pot_visible, live_timer, aged_timer))

    # ---- private staging counters of the left room ----
    s0_vec, s0_ok = _cell(obs, pos, _SLOT0_R, _SLOT0_C)
    s1_vec, s1_ok = _cell(obs, pos, _SLOT1_R, _SLOT1_C)
    slot0 = _i32(jnp.where(s0_ok, s0_vec[31] > 0, mem[8] > 0))
    slot1 = _i32(jnp.where(s1_ok, s1_vec[32] > 0, mem[9] > 0))

    # ---- never start a new pot while a delivery is still pending: the recipe
    #      is resampled by that delivery ----
    partner_has_dish = obs[:, :, 16].max() > 0
    scoop_seen = pot_visible & (live_n == 0) & ((mem[5] > 0) | (mem[2] > 0))
    since_empty = _i32(jnp.where(scoop_seen | partner_has_dish, 0,
                                 jnp.minimum(mem[7] + 1, 99)))
    # A delivery resamples the recipe, so the gap between the scoop and the new
    # display is learned rather than assumed: it is two steps for a partner
    # that walks straight to the station, more for a slower one.
    recipe_changed = rec_visible & (mem[1] < 3) & (recipe != mem[1])
    deliver_lag = _i32(jnp.where(recipe_changed & (since_empty <= 12),
                                 jnp.maximum(mem[15], since_empty),
                                 mem[15]))
    fresh_recipe = since_empty >= deliver_lag

    pot_idle = (pot_cooked == 0) & (pot_timer == 0) & (pot_n < 3)
    recipe_known = recipe < 3
    wrong_batch = (pot_n > 0) & ((pot_mixed > 0) | (recipe_known & (pot_type != recipe)))
    pot_stall = _i32(jnp.where(
        pot_visible,
        jnp.where(pot_n != mem[2], 0, jnp.minimum(mem[16] + 1, 99)),
        mem[16]))
    # A mis-typed batch is only completed to unblock the pot when the partner
    # has clearly abandoned it; a partner that is still loading the wrong
    # ingredient must never be helped to finish faster.
    may_recover = (mem[10] < _MAX_RECOVERY_FILLS) & (pot_stall >= 15)

    # ======================= left-room seat =============================
    want = _i32(jnp.where(pot_n > 0, pot_type, recipe))
    pot_open = (
        pot_idle
        & (want < 3)
        & ((pot_n > 0) | fresh_recipe)
        & ((~wrong_batch) | may_recover)
    )
    d3_vec, d3_ok = _cell(obs, pos, 0, 3)
    dump_l = jnp.where(d3_ok & (d3_vec[29:34].sum() > 0), M_DUMP_L1, M_DUMP_L0)

    take_want = jnp.where(
        want == 0,
        jnp.where(slot0 > 0, M_TAKE_S0, M_SUP0_L),
        jnp.where(want == 1,
                  jnp.where(slot1 > 0, M_TAKE_S1, M_SUP1_L),
                  M_SUP2_L),
    )
    # Restocking a staging counter is a ~13 step round trip, so it is only
    # started with enough cooking time left, and it is kept sticky in memory so
    # that the trip is never abandoned half way.
    holding_nothing = (~held_plate) & (held_total == 0)
    restock_time = (pot_cooked == 0) & (pot_timer >= 12)
    restock_need = jnp.where(slot0 == 0, 1, jnp.where(slot1 == 0, 2, 0))
    prev_restock = mem[11]
    still_valid = (
        (prev_restock > 0)
        & jnp.where(prev_restock == 1, slot0 == 0, slot1 == 0)
        & (~pot_open)
        & holding_nothing
    )
    restock_active = _i32(jnp.where(
        still_valid, prev_restock,
        jnp.where((~pot_open) & holding_nothing & restock_time, restock_need, 0)))
    restock = jnp.where(restock_active == 1, M_SUP0_L,
                        jnp.where(restock_active == 2, M_SUP1_L, M_IDLE_L))
    stage_held = jnp.where(
        (held_type == 0) & (slot0 == 0), M_PUT_S0,
        jnp.where((held_type == 1) & (slot1 == 0), M_PUT_S1,
                  jnp.where(held_type == 2, dump_l, M_IDLE_L)),
    )
    left_mode = jnp.where(
        holding_dish | holding_plate,
        dump_l,
        jnp.where(
            holding_ing,
            jnp.where(pot_open & (held_type == want), M_POT_L, stage_held),
            jnp.where(
                ~recipe_known,
                M_SCOUT_L,
                jnp.where(pot_open, take_want, restock),
            ),
        ),
    )

    # ======================= right-room seat ============================
    # four drop cells, so a useless item can always be put down somewhere
    def _busy(grow, gcol):
        vec, ok = _cell(obs, pos, grow, gcol)
        return ok & (vec[29:34].sum() > 0)

    dump_r = jnp.where(
        _busy(0, 9), jnp.where(
            _busy(4, 9), jnp.where(_busy(1, 7), M_DUMP_R3, M_DUMP_R2),
            M_DUMP_R1),
        M_DUMP_R0)

    # only pure three-unit stacks of an edible ingredient can be a valid recipe
    dish_pure = (held_total == 3) & ((held_counts > 0).sum() == 1) & (held_type < 2)
    reachable_in_pot = live_counts[0:2]
    fill_type = _i32(jnp.where(
        pot_mixed > 0,
        jnp.where(reachable_in_pot.sum() > 0, jnp.argmax(reachable_in_pot), 2),
        pot_type))
    fill_possible = pot_idle & (pot_n > 0) & (pot_n < 3) & (fill_type < 2)
    take_fill = jnp.where(fill_type == 0, M_ING0_R, M_ING1_R)

    # last-resort solo cooking when the partner never loads the pot
    pot_dead = (pot_n == 0) & (pot_cooked == 0) & (pot_timer == 0)
    dead_streak = _i32(jnp.where(pot_dead, jnp.minimum(mem[12] + 1, 999), 0))
    # A partner that carries ingredients to the pot is loading it, only slowly;
    # taking the pot over would waste its work, so solo cooking waits until
    # there has been no sign of the partner for a long time.
    partner_carries = obs[:, :, 17:20].max() > 0
    since_carry = _i32(jnp.where(partner_carries, 0,
                                 jnp.minimum(mem[17] + 1, 199)))
    solo_latch = _i32(((mem[14] > 0) | (dead_streak >= _SOLO_AFTER))
                      & (since_carry >= _SOLO_QUIET))
    solo_type = _i32(mem[13])
    solo_start = pot_dead & (solo_latch > 0)
    take_solo = jnp.where(solo_type == 0, M_ING0_R, M_ING1_R)
    insert_ok = ((pot_n > 0) & (pot_type == held_type)) | (
        solo_start & (held_type == solo_type))

    right_mode = jnp.where(
        holding_dish,
        jnp.where(dish_pure, M_DELIVER, dump_r),
        jnp.where(
            holding_plate,
            jnp.where(pot_cooked > 0, M_POT_R,
                      jnp.where(fill_possible | solo_start, dump_r, M_IDLE_R)),
            jnp.where(
                holding_ing,
                jnp.where(pot_idle & (pot_n < 3) & insert_ok, M_POT_R, dump_r),
                jnp.where(
                    pot_cooked > 0,
                    M_PLATE_R,
                    jnp.where(
                        fill_possible, take_fill,
                        jnp.where(solo_start, take_solo,
                                  jnp.where((pot_timer > 0) | (pot_n >= 3),
                                            M_PLATE_R, M_IDLE_R))),
                ),
            ),
        ),
    )

    is_right = pos[1] > _POT_C
    mode = _i32(jnp.where(is_right, right_mode, left_mode))
    action = _navigate(pos, face_act, mode)

    recovered = mem[10] + jnp.where(
        (~is_right) & (mode == M_POT_L) & (action == _INTERACT) & wrong_batch, 1, 0)
    # alternate the solo ingredient after every delivery this seat makes
    delivered = is_right & (mode == M_DELIVER) & (action == _INTERACT)
    next_solo = _i32(jnp.where(delivered, 1 - solo_type, solo_type))

    new_mem = _i32(jnp.stack([
        mem[0] + 1,
        recipe,
        pot_n,
        pot_type,
        pot_mixed,
        pot_cooked,
        pot_timer,
        since_empty,
        slot0,
        slot1,
        _i32(recovered),
        _i32(jnp.where(is_right, 0, restock_active)),
        dead_streak,
        next_solo,
        _i32(jnp.where(is_right, solo_latch, 0)),
        deliver_lag,
        pot_stall,
        since_carry,
    ]))

    trace = {
        'pos': _i32(pos),
        'mode': mode,
        'recipe': recipe,
        'pot_n': pot_n,
        'pot_timer': pot_timer,
        'act': action,
    }
    return action, new_mem, trace


def init_agent_0():
    return jnp.asarray(_MEM0, jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    return _agent(obs, reset, memory, rng)


def init_agent_1():
    return jnp.asarray(_MEM0, jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    return _agent(obs, reset, memory, rng)
