"""Policy for overcooked_v2 / grounded_coord_ring.

Layout facts that drive the design (all taken from context/environment):

    WWW2R2WWW      inner room = {(3,3),(3,4),(3,5),(4,3),(4,5),(5,3),(5,4),(5,5)}
    W       W      outer ring = the corridor around it
    W WWLWW W
    2 0   B 2      The two rooms are disconnected floor regions.
    RAXAP X R      Only the inner seat can reach the pot (4,4); it can also
    2 1   B 2      reach both ingredient piles, both plate piles and both
    W WWLWW W      delivery stations, so it is self-sufficient *except* that
    W       W      it can never see an always-on recipe indicator (R).
    WWW2R2WWW      Only the outer seat can see R; both seats can activate a
                   button indicator L at (2,4)/(6,4) (cost 5), which makes the
                   current recipe visible inside the inner room.

Therefore:
  * inner seat -> full solo cook loop; it learns the target from a visible
    button display and activates a button itself when no display is up.
  * outer seat -> grounded informer: it parks next to a button, watches R and
    activates the button when the recipe changed (i.e. a delivery just
    happened) or when its display has been down for a while.

No private signalling convention is used: the only channel is the button
display, which is part of the environment's own grounded mechanics, and the
inner seat never depends on the partner doing anything.
"""

from __future__ import annotations

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "ring_cook_informer",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}

# ---------------------------------------------------------------- literals --
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
_CHAR_CHANNEL = {"W": 0, "X": 1, "P": 2, "R": 3, "L": 4, "B": 5, "0": 6, "1": 7, "2": 8}

# action ids
_RIGHT, _DOWN, _LEFT, _UP, _STAY, _INTERACT = 0, 1, 2, 3, 4, 5
# facing ids as encoded in the observation
_D_UP, _D_DOWN, _D_RIGHT, _D_LEFT = 0, 1, 2, 3

# task table: (cell to stand on, direction to face).  Interacting from that
# pose touches the intended object.
_TASKS = (
    ((3, 4), _D_UP),      # 0  activate top button (2,4)
    ((5, 4), _D_DOWN),    # 1  activate bottom button (6,4)
    ((3, 3), _D_LEFT),    # 2  take ingredient 0 from (3,2)
    ((5, 3), _D_LEFT),    # 3  take ingredient 1 from (5,2)
    ((3, 4), _D_DOWN),    # 4  load pot from the north
    ((5, 4), _D_UP),      # 5  load pot from the south
    ((3, 5), _D_RIGHT),   # 6  take a plate from (3,6)
    ((5, 5), _D_RIGHT),   # 7  take a plate from (5,6)
    ((4, 3), _D_RIGHT),   # 8  scoop the pot from the west station
    ((4, 5), _D_LEFT),    # 9  scoop the pot from the east station
    ((4, 3), _D_LEFT),    # 10 deliver at (4,2)
    ((4, 5), _D_RIGHT),   # 11 deliver at (4,6)
    ((3, 3), _D_UP),      # 12 dump onto counter (2,3)
    ((5, 3), _D_DOWN),    # 13 dump onto counter (6,3)
    ((3, 5), _D_UP),      # 14 dump onto counter (2,5)
    ((5, 5), _D_DOWN),    # 15 dump onto counter (6,5)
    ((1, 4), _D_DOWN),    # 16 outer seat: activate the top button
    ((7, 4), _D_UP),      # 17 outer seat: activate the bottom button
)
# dump task id -> counter cell it targets
_DUMP = ((12, (2, 3)), (13, (6, 3)), (14, (2, 5)), (15, (6, 5)))

_POT = (4, 4)
_R_TOP = (0, 4)
_R_BOT = (8, 4)
_B_TOP = (2, 4)
_B_BOT = (6, 4)

# memory slots:
# cook:  0 t, 1 recipe, 2 own last activation, 4 last delivery, 5 trust,
#        7 was holding a dish, 9 pre-activation done
# outer: 0 t, 1 recipe at last activation, 2 last activation, 3 speculative
#        activations, 4 last observed R recipe, 5 evidence, 6 station,
#        8 last station switch, 9 partner is self-sufficient,
#        10 last sign of life at this station, 11 serving clock
_INIT_MEM = (0, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0)

_PRESS_DEADLINE = 358          # a cook activation later than this cannot pay off
_PREPRESS_DEADLINE = 350       # ... same for the pre-delivery activation
_PREPRESS_TIMER = 6            # pot countdown at which the pre-activation fires
_TRUST_WAIT = 9                # steps to wait for an informing partner's display
_ANNOUNCE_WINDOW = 6           # a display this soon after a delivery is an announcement
_OUTER_DEADLINE = 380
_OUTER_PERIOD = 41             # refresh period, a little under a cook cycle
_OUTER_FIRST = 10              # let the partner light the button first
_OUTER_BLIND_LIMIT = 6         # activations to spend before any delivery is seen
_SWITCH_AT = 100               # serving steps without a sign of life or a delivery


# ------------------------------------------------------------------ tables --
def _tables():
    """Derive every constant table from the literal layout (traced once)."""
    cells = []
    for r in range(9):
        for c in range(9):
            if _GRID[r][c] in " A":
                cells.append((r, c))
    index_of = {}
    for i in range(len(cells)):
        index_of[cells[i]] = i
    n = len(cells)

    smap = []
    for r in range(9):
        row = []
        for c in range(9):
            vec = [0] * 9
            ch = _GRID[r][c]
            if ch in _CHAR_CHANNEL:
                vec[_CHAR_CHANNEL[ch]] = 1
            row.append(vec)
        smap.append(row)
    padded = jnp.pad(jnp.asarray(smap, dtype=jnp.int32), ((2, 2), (2, 2), (0, 0)))
    crops = jnp.stack([padded[r:r + 5, c:c + 5, :] for (r, c) in cells])

    # neighbour table ordered like the movement actions (right, down, left, up)
    deltas = ((0, 1), (1, 0), (0, -1), (-1, 0))
    nb = []
    for i in range(n):
        r, c = cells[i]
        row = []
        for a in range(4):
            row.append(index_of.get((r + deltas[a][0], c + deltas[a][1]), i))
        nb.append(row)
    neighbours = jnp.asarray(nb, dtype=jnp.int32)

    big = 999
    dist0 = []
    for i in range(n):
        row = [big] * n
        row[i] = 0
        for a in range(4):
            if nb[i][a] != i:
                row[nb[i][a]] = 1
        dist0.append(row)
    dist = jnp.asarray(dist0, dtype=jnp.int32)
    for _ in range(6):  # min-plus squaring: 2**6 > ring diameter
        dist = jnp.minimum(dist, jnp.min(dist[:, :, None] + dist[None, :, :], axis=1))
    # greedy shortest-path step: the neighbour index is exactly the action id
    step = jnp.argmin(dist[neighbours, :], axis=1).astype(jnp.int32)

    stand = jnp.asarray([index_of[t[0]] for t in _TASKS], dtype=jnp.int32)
    face = jnp.asarray([t[1] for t in _TASKS], dtype=jnp.int32)
    rows = jnp.asarray([rc[0] for rc in cells], dtype=jnp.int32)
    cols = jnp.asarray([rc[1] for rc in cells], dtype=jnp.int32)
    inner = jnp.asarray(
        [1 if (abs(rc[0] - _POT[0]) <= 2 and abs(rc[1] - _POT[1]) <= 2) else 0 for rc in cells],
        dtype=jnp.int32,
    )
    move_for_dir = jnp.asarray([_UP, _DOWN, _RIGHT, _LEFT], dtype=jnp.int32)
    return crops, rows, cols, inner, dist, step, stand, face, move_for_dir


# ------------------------------------------------------------------ helpers --
def _cell_vec(obs, r, c, gr, gc):
    """All 40 channels of global cell (gr, gc); zeros when outside the view."""
    li = gr - r + 2
    lj = gc - c + 2
    ok = (li >= 0) & (li < 5) & (lj >= 0) & (lj < 5)
    vec = obs[jnp.clip(li, 0, 4), jnp.clip(lj, 0, 4), :]
    return jnp.where(ok, vec, jnp.zeros_like(vec))


def _navigate(tab, cell, facing, task, do_interact):
    _, _, _, _, _, step, stand, face, move_for_dir = tab
    goal = stand[task]
    want_dir = face[task]
    arrived = cell == goal
    aimed = facing == want_dir
    return jnp.where(
        arrived,
        jnp.where(aimed, jnp.where(do_interact, _INTERACT, _STAY), move_for_dir[want_dir]),
        step[cell, goal],
    ).astype(jnp.int32)


# --------------------------------------------------- inner seat: solo cook --
def _cook(tab, obs, mem, cell, r, c, t):
    _, _, _, _, dist, _, stand, _, _ = tab
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    plate = obs[2, 2, 5]
    cooked = obs[2, 2, 6]
    h0, h1, h2 = obs[2, 2, 7], obs[2, 2, 8], obs[2, 2, 9]
    hold_dish = cooked > 0
    hold_plate = (plate > 0) & (cooked == 0)
    hands_empty = (plate + cooked + h0 + h1 + h2) == 0

    pot = _cell_vec(obs, r, c, _POT[0], _POT[1])
    pot_ready = pot[30] > 0
    pot_timer = pot[39]
    pot_cooking = pot_timer > 0

    # Recipe knowledge: a visible display overrides memory, and a delivery
    # invalidates memory because the target is resampled afterwards.
    just_delivered = (mem[7] > 0) & (cooked == 0)
    remembered = jnp.where(just_delivered, -1, mem[1])
    disp0 = jnp.max(obs[:, :, 36])
    disp1 = jnp.max(obs[:, :, 37])
    rec = jnp.where(disp0 > 0, 0, jnp.where(disp1 > 0, 1, remembered)).astype(jnp.int32)

    # Bookkeeping for "is the partner informing me?".  A display that is up
    # more than nine steps after our own activation was lit by the partner; if
    # it appeared right after our last delivery, the partner reliably announces
    # each new target and we can stop paying for displays ourselves.
    own_press_t = mem[2]
    display_on = (disp0 > 0) | (disp1 > 0)
    partner_lit = display_on & ((own_press_t < 0) | ((t - own_press_t) > 9))
    last_delivery_t = jnp.where(just_delivered, t, mem[4])
    # A partner that lights a button right after a delivery is announcing the
    # fresh target, which makes our own activation pure waste.  Trust such a
    # partner: skip the pre-activation and give it a few steps to speak up.
    # Trust is dropped again as soon as we have to pay for a display ourselves.
    announced = (partner_lit & (last_delivery_t >= 0)
                 & ((t - last_delivery_t) <= _ANNOUNCE_WINDOW))
    partner_informs = (mem[5] > 0) | announced

    # pick the nearer variant of each symmetric task
    east = dist[cell, stand[9]] <= dist[cell, stand[8]]
    serve_task = jnp.where(east, 9, 8)
    deliver_task = jnp.where(east, 11, 10)
    plate_task = jnp.where(dist[cell, stand[7]] < dist[cell, stand[6]], 7, 6)
    # Always use the north button (2,4): it is reachable well inside the pot
    # countdown from either lane, so it costs nothing, and it keeps this seat's
    # activations visible to a partner watching that button.
    press_task = jnp.asarray(0, dtype=jnp.int32)
    get_task = jnp.where(rec == 1, 3, 2)
    load_task = jnp.where(rec == 1, 5, 4)

    dump_task = jnp.asarray(_DUMP[0][0], dtype=jnp.int32)
    dump_cost = jnp.asarray(9999, dtype=jnp.int32)
    for task_id, counter in _DUMP:
        occupied = jnp.sum(_cell_vec(obs, r, c, counter[0], counter[1])[29:34]) > 0
        cost = dist[cell, stand[task_id]] + jnp.where(occupied, 50, 0)
        better = cost < dump_cost
        dump_task = jnp.where(better, task_id, dump_task)
        dump_cost = jnp.where(better, cost, dump_cost)

    matches = (((rec == 0) & (h0 == 3) & (h1 == 0) & (h2 == 0))
               | ((rec == 1) & (h1 == 3) & (h0 == 0) & (h2 == 0)))
    holds_needed = ((rec == 0) & (h0 == 1)) | ((rec == 1) & (h1 == 1))

    # While the pot cooks there are idle steps.  Spend them on activating a
    # button exactly late enough that its display is still alive one step after
    # the coming delivery: the recipe is resampled then, so the next target is
    # read for free and no post-delivery button trip is needed.  The return
    # path (plate, scoop, turn, deliver) is 8 steps, matching the 9-step
    # display lifetime, and the countdown of the pot gives the schedule.
    pre_due = pot_timer <= _PREPRESS_TIMER
    pre_wait = ((mem[9] == 0) & hands_empty & (t <= _PREPRESS_DEADLINE)
                & (~partner_informs))

    # When the partner has been informing us, walk to the button but hold the
    # activation for a few steps: if it announces the new target we save the 5.
    press_ok = ((t <= _PRESS_DEADLINE)
                & ((~partner_informs) | (last_delivery_t < 0)
                   | ((t - last_delivery_t) >= _TRUST_WAIT)))

    task_a = jnp.where(matches, deliver_task, dump_task)          # holding a dish
    task_b = jnp.where(hold_plate, serve_task,                    # pot is ready
                       jnp.where(hands_empty, plate_task, dump_task))
    task_c = jnp.where(pre_wait, press_task, task_b)              # pot is cooking
    # While waiting for a trusted partner to announce, wait on the delivery
    # station: it is where the last delivery left us, it is three steps from
    # either ingredient pile, and it is one of the only two cells from which
    # *both* buttons are visible, so an announcement cannot be missed.
    task_d = jnp.where(                                           # pot is idle
        rec < 0,
        jnp.where(hands_empty, jnp.where(press_ok, press_task, serve_task), dump_task),
        jnp.where(holds_needed, load_task,
                  jnp.where(hands_empty, get_task, dump_task)),
    )
    act_d = jnp.where((rec < 0) & hands_empty, press_ok, True)

    cooking_phase = pot_cooking & (~hold_dish) & (~pot_ready)
    task = jnp.where(hold_dish, task_a,
                     jnp.where(pot_ready, task_b,
                               jnp.where(pot_cooking, task_c, task_d)))
    do_interact = jnp.where(cooking_phase,
                            jnp.where(pre_wait, pre_due, ~hold_plate),
                            jnp.where(hold_dish | pot_ready, True, act_d))

    action = _navigate(tab, cell, facing, task, do_interact)
    # never pay for a display that can no longer produce a delivery
    idle_late = ((~hold_dish) & (~pot_ready) & (~pot_cooking)
                 & (rec < 0) & (t > _PRESS_DEADLINE))
    action = jnp.where(idle_late, jnp.asarray(_STAY, dtype=jnp.int32), action)

    pre_done = jnp.where(pot_cooking,
                         jnp.maximum(mem[9],
                                     jnp.where(cooking_phase & pre_wait
                                               & (action == _INTERACT), 1, 0)),
                         0).astype(jnp.int32)
    pressed_now = (task == 0) & (action == _INTERACT)
    trust = jnp.where(announced, 1,
                      jnp.where(pressed_now & (~cooking_phase), 0, mem[5]))
    new_mem = (mem.at[1].set(rec)
               .at[2].set(jnp.where(pressed_now, t, own_press_t).astype(jnp.int32))
               .at[4].set(last_delivery_t.astype(jnp.int32))
               .at[5].set(trust.astype(jnp.int32))
               .at[7].set(jnp.where(hold_dish, 1, 0).astype(jnp.int32))
               .at[9].set(pre_done))
    return action, new_mem, task, rec


# --------------------------------------------- outer seat: recipe informer --
def _informer(tab, obs, mem, cell, r, c, t):
    """Park beside a button and light it when the cook plausibly needs it.

    Three behaviours, in order of authority: stay SILENT once the partner has
    shown that it lights buttons itself; SWITCH to the other button if nothing
    observable ever happens on this side; otherwise SERVE on a schedule.
    """
    _, _, _, _, _, _, stand, face, _ = tab
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    station = mem[6]
    top = station == 0
    task = jnp.where(top, 16, 17)
    at_station = (cell == stand[task]) & (facing == face[task])

    # the current recipe from any visible always-on indicator
    rec0 = jnp.zeros((), dtype=jnp.int32)
    rec1 = jnp.zeros((), dtype=jnp.int32)
    for cellrc in ((0, 4), (8, 4), (4, 0), (4, 8)):
        vec = _cell_vec(obs, r, c, cellrc[0], cellrc[1])
        rec0 = jnp.maximum(rec0, vec[36])
        rec1 = jnp.maximum(rec1, vec[37])
    rec_now = jnp.where(rec0 > 0, 0, jnp.where(rec1 > 0, 1, -1)).astype(jnp.int32)

    bvec = jnp.where(top,
                     _cell_vec(obs, r, c, _B_TOP[0], _B_TOP[1]),
                     _cell_vec(obs, r, c, _B_BOT[0], _B_BOT[1]))
    display_on = (bvec[36] + bvec[37]) > 0

    seen = rec_now >= 0
    changed = seen & (mem[4] >= 0) & (rec_now != mem[4])  # a delivery happened
    partner_busy = jnp.max(obs[:, :, 15:20]) > 0          # partner carrying something
    partner_here = jnp.max(obs[:, :, 10]) > 0             # partner visible at all
    # "is anybody actually delivering?" versus "is anybody on this side?"
    # Only a completed delivery proves that informing is worth its price; a
    # partner merely carrying something may just be flailing.
    evidence = mem[5] + jnp.where(changed, 1, 0)

    pressed_before = mem[2] >= 0
    # A display we did not light means the partner activates buttons itself;
    # informing such a partner is pure cost, so stay quiet from then on.
    ours_fresh = pressed_before & ((t - mem[2]) <= 9)
    self_sufficient = jnp.maximum(mem[9], jnp.where(display_on & (~ours_fresh), 1, 0))

    serve_clock = mem[11]
    station_seen_t = jnp.where((changed | partner_here) & at_station, t, mem[10])

    # Nothing observable ever happens on this side?  The partner must work out
    # of sight of this button, so move to the other one; both stations see an
    # always-on indicator and a button equally well.
    # Only move when nothing is being delivered at all: if deliveries happen,
    # this station is either serving the partner or the partner needs nothing,
    # and a partner can easily be at work in a cell neither station can see.
    switch = ((self_sufficient == 0) & (evidence == 0) & (serve_clock >= _SWITCH_AT)
              & ((t - station_seen_t) >= _SWITCH_AT))

    quiet = self_sufficient > 0
    blind = evidence == 0            # nothing yet suggests the partner works
    due_first = (~pressed_before) & (t >= _OUTER_FIRST)
    due_change = pressed_before & seen & (rec_now != mem[1])
    due_timer = pressed_before & ((t - mem[2]) >= _OUTER_PERIOD)
    want = (seen & (~display_on) & (t <= _OUTER_DEADLINE) & (~quiet)
            & ((~blind) | (mem[3] < _OUTER_BLIND_LIMIT))
            & (due_first | due_change | due_timer))

    action = _navigate(tab, cell, facing, task, want)
    pressing = want & at_station
    # the serve clock only runs while we are parked and free to inform
    serving = at_station & (~quiet) & (~switch)
    serve_next = jnp.where(switch, 0, serve_clock + jnp.where(serving, 1, 0))

    new_mem = (
        mem.at[1].set(jnp.where(pressing, rec_now, mem[1]))
        .at[2].set(jnp.where(pressing, t, mem[2]))
        .at[3].set(jnp.where(pressing & blind, mem[3] + 1, mem[3]))
        .at[4].set(jnp.where(seen, rec_now, mem[4]))
        .at[5].set(evidence)
        .at[6].set(jnp.where(switch, 1 - station, station))
        .at[8].set(jnp.where(switch, t, mem[8]))
        .at[9].set(self_sufficient)
        .at[10].set(jnp.where(switch, t, station_seen_t).astype(jnp.int32))
        .at[11].set(serve_next.astype(jnp.int32))
    )
    return action, new_mem, task, rec_now


# --------------------------------------------------------- shared entry pt --
def _policy(obs, reset, memory, rng):
    tab = _tables()
    crops, rows, cols, inner_flag, _, _, _, _, _ = tab

    obs = jnp.asarray(obs, dtype=jnp.int32)
    mem = jnp.where(reset, jnp.asarray(_INIT_MEM, dtype=jnp.int32), memory)
    t = mem[0]

    # Localise by matching the static channels against every legal stand cell;
    # the 32 crops are pairwise distinct, so this is exact.
    diffs = jnp.sum(jnp.abs(crops - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    cell = jnp.argmin(diffs).astype(jnp.int32)
    r = rows[cell]
    c = cols[cell]
    inner = inner_flag[cell] > 0  # only the inner room is near the pot

    cook_act, cook_mem, cook_task, cook_rec = _cook(tab, obs, mem, cell, r, c, t)
    out_act, out_mem, out_task, out_rec = _informer(tab, obs, mem, cell, r, c, t)

    action = jnp.where(inner, cook_act, out_act).astype(jnp.int32)
    new_mem = jnp.where(inner, cook_mem, out_mem).at[0].set(t + 1).astype(jnp.int32)
    trace = {
        "cell": cell,
        "inner": inner.astype(jnp.int32),
        "task": jnp.where(inner, cook_task, out_task).astype(jnp.int32),
        "recipe": jnp.where(inner, cook_rec, out_rec).astype(jnp.int32),
        "action": action,
    }
    return action, new_mem, trace


def init_agent_0():
    return jnp.asarray(_INIT_MEM, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    return _policy(obs, reset, memory, rng)


def init_agent_1():
    return jnp.asarray(_INIT_MEM, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    return _policy(obs, reset, memory, rng)
