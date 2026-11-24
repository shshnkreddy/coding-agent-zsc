"""Hand-coded cooperative policy for demo_cook_simple (Overcooked V2).

Geometry forces the role split: the pot at (2,7) is the only opening between a
left room (agent_0) and a right room (agent_1).  Only agent_0 can see the
recipe indicators at (0,5)/(4,5); only agent_1 can reach plates and the
delivery station.  The single coordination signal both seats can read straight
out of the environment is the *pot contents*, so agent_0 (who knows the recipe)
seeds the pot with the target ingredient and the right-room seat simply copies
whatever pure ingredient the pot already holds.  No private convention is
required, so either seat also behaves sensibly beside an unknown partner.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'pot_seed_coop', 'adapter': 'overcooked_v2', 'adapter_version': 1,
                   'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

# --------------------------------------------------------------------------
# Constants describing the fixed layout (literals only at module scope).
# Static channel order: wall, goal, pot, recipe, button, plate, ing0/1/2.
# Board:
#   WWWWWR2W0WW
#   0      W  B
#   W     APA X
#   1      W  B
#   WWWWWR2W1WW
# --------------------------------------------------------------------------
_MAP_CODES = (
    (0, 0, 0, 0, 0, 3, 8, 0, 6, 0, 0),
    (6, -1, -1, -1, -1, -1, -1, 0, -1, -1, 5),
    (0, -1, -1, -1, -1, -1, -1, 2, -1, -1, 1),
    (7, -1, -1, -1, -1, -1, -1, 0, -1, -1, 5),
    (0, 0, 0, 0, 0, 3, 8, 0, 7, 0, 0),
)
_WIDTH = 11

# facing ids UP=0 DOWN=1 RIGHT=2 LEFT=3 -> movement action that produces them
_DIR_ACTION = (3, 1, 0, 2)
_UP, _DOWN, _RIGHT, _LEFT = 0, 1, 2, 3
_STAY, _INTERACT = 4, 5

_ROWS_0 = (1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3)
_COLS_0 = (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)
_ROWS_1 = (1, 1, 2, 2, 3, 3)
_COLS_1 = (8, 9, 8, 9, 8, 9)

# Shortest-path action from every board cell towards each goal standing cell.
# Ties are broken so the last step of a route already sets the facing needed
# for the interaction waiting there, which saves one turn action per trip.
# agent_0 goals: 0 (1,1) ing0 supply, 1 (3,1) ing1 supply, 2 (2,6) pot,
#   3 (1,4) cache0, 4 (3,4) cache1, 5 (1,6) ing2 supply, 6 (1,3) cache2,
#   7 (2,5) idle post: sees the pot, both recipe indicators and both caches,
#   and is one step closer to the caches than the pot tile is.
_NAV_0 = (
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4,
     4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 4, 4,
     4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 4, 4,
     4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 2, 2, 4, 4, 4, 4, 4, 0, 0, 0, 3, 2, 2, 4, 4,
     4, 4, 4, 0, 0, 0, 3, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 2, 2, 4, 4, 4, 4, 4, 0, 0, 0, 1, 2, 2, 4, 4,
     4, 4, 4, 0, 0, 0, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 3, 4, 4,
     4, 4, 4, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 2, 2, 2, 4, 4, 4, 4, 4, 0, 0, 3, 2, 2, 2, 4, 4,
     4, 4, 4, 0, 0, 3, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 2, 4, 4,
     4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
)
# agent_1 goals: 0 (1,8) ing0 supply, 1 (3,8) ing1 supply, 2 (2,8) pot,
#   3 (1,9) plate pile / top counter, 4 (2,9) delivery, 5 (3,9) bottom counter
_NAV_1 = (
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3,
     2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1,
     2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
     2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3,
     3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0,
     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0,
     1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
)

# agent_0: per ingredient type, where to buy it and where its private cache is
_SUP_GOAL_0 = (0, 1, 5)
_SUP_DIR_0 = (_LEFT, _LEFT, _UP)
_CACHE_GOAL_0 = (3, 4, 6)
_CACHE_DIR_0 = (_UP, _DOWN, _UP)
_CACHE_R = (0, 4, 0)
_CACHE_C = (4, 4, 3)
# agent_1: per ingredient type, where to buy it (ingredient 2 is unreachable)
_SUP_GOAL_1 = (0, 1, 2)
_SUP_DIR_1 = (_UP, _DOWN, _LEFT)

_POT_R, _POT_C = 2, 7
_MEM_0 = (0, 1, 0, 0, 0, 9)  # target, pot-accepts flag, 3 cache flags, empty-pot streak
_SETTLE_STEPS = 4  # observations of an empty pot before its order is trusted


# --------------------------------------------------------------------------
# Observation helpers.
# --------------------------------------------------------------------------
def _localize(obs, rows, cols):
    """Identify our cell by matching the static crop against every room cell."""
    codes = jnp.asarray(_MAP_CODES, dtype=jnp.int32)
    static_map = (codes[:, :, None] == jnp.arange(9, dtype=jnp.int32)[None, None, :])
    padded = jnp.pad(static_map.astype(jnp.int32), ((2, 2), (2, 2), (0, 0)))
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    span = jnp.arange(5, dtype=jnp.int32)
    patches = padded[rows[:, None, None] + span[None, :, None],
                     cols[:, None, None] + span[None, None, :]]
    mismatch = jnp.sum(jnp.abs(patches - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    idx = jnp.argmin(mismatch)
    return rows[idx], cols[idx], rows[idx] * _WIDTH + cols[idx]


def _read(obs, row, col, target_r, target_c):
    """Features of a known global cell, plus whether it is inside the crop."""
    i = jnp.asarray(target_r, dtype=jnp.int32) - row + 2
    j = jnp.asarray(target_c, dtype=jnp.int32) - col + 2
    visible = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    cell = obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4)]
    return visible, cell


def _self_state(obs):
    """Facing, and what is in our hand (a dish also carries ingredient counts)."""
    facing = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)
    plate = obs[2, 2, 5] > 0
    cooked = obs[2, 2, 6] > 0
    counts = obs[2, 2, 7:10]
    has_raw = (jnp.sum(counts) > 0) & jnp.logical_not(plate) & jnp.logical_not(cooked)
    ing = jnp.argmax(counts).astype(jnp.int32)
    return facing, plate, cooked, has_raw, ing


def _pot_state(cell):
    counts = cell[31:34]
    total = jnp.sum(counts)
    timer = cell[39]
    cooked = cell[30] > 0
    kind = jnp.argmax(counts).astype(jnp.int32)
    pure = (counts[kind] == total) & (total > 0)
    addable = (total < 3) & (timer == 0) & jnp.logical_not(cooked)
    ready = cooked & (timer == 0)
    return total, kind, pure, addable, ready, timer


def _execute(nav, cell_idx, facing, goal, want_dir, interact):
    """Walk towards `goal`; on arrival face `want_dir` and maybe INTERACT."""
    table = jnp.asarray(nav, dtype=jnp.int32)
    step = table[goal, cell_idx]
    turn = jnp.asarray(_DIR_ACTION, dtype=jnp.int32)[want_dir]
    aligned = facing == want_dir
    arrived = jnp.where(aligned,
                        jnp.where(interact, _INTERACT, _STAY),
                        turn)
    return jnp.where(step == _STAY, arrived, step).astype(jnp.int32)


def _task(goal, want_dir, interact):
    return (jnp.asarray(goal, dtype=jnp.int32),
            jnp.asarray(want_dir, dtype=jnp.int32),
            jnp.asarray(interact, dtype=jnp.bool_))


def _pick(condition, current, candidate):
    """Choose between whole (goal, facing, interact) task tuples."""
    return (jnp.where(condition, candidate[0], current[0]),
            jnp.where(condition, candidate[1], current[1]),
            jnp.where(condition, candidate[2], current[2]))


# --------------------------------------------------------------------------
# agent_0 : left room.  Reads the recipe, seeds and refills the pot, and keeps
# one spare unit of each ingredient on the private counters (0,4) and (4,4)
# so the next order can be started within a few steps.
# --------------------------------------------------------------------------
def init_agent_0():
    return jnp.asarray(_MEM_0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, jnp.asarray(_MEM_0, dtype=jnp.int32), memory)

    row, col, cell_idx = _localize(obs, _ROWS_0, _COLS_0)
    facing, held_plate, held_cooked, has_raw, ing = _self_state(obs)

    # Recipe: either always-on indicator carries the live target while in view.
    vis_a, cell_a = _read(obs, row, col, 0, 5)
    vis_b, cell_b = _read(obs, row, col, 4, 5)
    shown = (jnp.where(vis_a, cell_a[36:39], 0) + jnp.where(vis_b, cell_b[36:39], 0))
    target = jnp.where(jnp.sum(shown) > 0,
                       jnp.argmax(shown).astype(jnp.int32), mem[0])

    pot_vis, pot_cell = _read(obs, row, col, _POT_R, _POT_C)
    pot_total, _, _, pot_addable, _, _ = _pot_state(pot_cell)
    pot_ok = jnp.where(pot_vis, pot_addable, mem[1] > 0)
    # The order is resampled when the other seat delivers, a step or two after
    # it empties the pot.  Seeding an empty pot immediately would commit to the
    # previous order, so wait until the indicator can no longer be stale.  This
    # costs nothing: fetching an ingredient already takes longer than that.
    empty_now = pot_vis & (pot_total == 0) & pot_addable
    empty_streak = jnp.where(pot_vis,
                             jnp.where(empty_now, jnp.minimum(mem[5] + 1, 9), 0),
                             mem[5])
    may_seed = jnp.logical_not(empty_now) | (empty_streak >= _SETTLE_STEPS)

    # Private ingredient caches: counters no other seat can reach.
    vis0, cell0 = _read(obs, row, col, _CACHE_R[0], _CACHE_C[0])
    vis1, cell1 = _read(obs, row, col, _CACHE_R[1], _CACHE_C[1])
    vis2, cell2 = _read(obs, row, col, _CACHE_R[2], _CACHE_C[2])
    cache0 = jnp.where(vis0, cell0[31] > 0, mem[2] > 0)
    cache1 = jnp.where(vis1, cell1[32] > 0, mem[3] > 0)
    cache2 = jnp.where(vis2, cell2[33] > 0, mem[4] > 0)
    cache_flags = jnp.stack([cache0, cache1, cache2])

    sup_goal = jnp.asarray(_SUP_GOAL_0, dtype=jnp.int32)
    sup_dir = jnp.asarray(_SUP_DIR_0, dtype=jnp.int32)
    cache_goal = jnp.asarray(_CACHE_GOAL_0, dtype=jnp.int32)
    cache_dir = jnp.asarray(_CACHE_DIR_0, dtype=jnp.int32)

    idle = _task(7, _RIGHT, False)
    feed_pot = (jnp.asarray(2, dtype=jnp.int32), jnp.asarray(_RIGHT, dtype=jnp.int32),
                may_seed)

    # Idle time is spent staging one spare unit of each ingredient.
    prefetch = _pick(jnp.logical_not(cache1), idle,
                     (sup_goal[1], sup_dir[1], jnp.asarray(True)))
    prefetch = _pick(jnp.logical_not(cache0), prefetch,
                     (sup_goal[0], sup_dir[0], jnp.asarray(True)))

    # Pot wants ingredients: take the staged unit when there is one.
    from_cache = cache_flags[target]
    acquire = _pick(from_cache,
                    (sup_goal[target], sup_dir[target], jnp.asarray(True)),
                    (cache_goal[target], cache_dir[target], jnp.asarray(True)))

    task = _pick(pot_ok, prefetch, acquire)

    # Carrying something: feed the pot when it matches, else stage the item.
    slot = jnp.where(has_raw, ing, jnp.asarray(0, dtype=jnp.int32))
    stash = _pick(jnp.logical_not(cache_flags[slot]), idle,
                  (cache_goal[slot], cache_dir[slot], jnp.asarray(True)))
    carry = _pick((ing == target) & pot_ok, stash, feed_pot)
    task = _pick(has_raw, task, carry)
    # A stray plate or dish would block every job: park it on the spare counter
    # at (0,3), which no ingredient this layout ever uses is staged on.
    task = _pick(held_plate | held_cooked, task,
                 (cache_goal[2], cache_dir[2], jnp.asarray(True)))

    action = _execute(_NAV_0, cell_idx, facing, task[0], task[1], task[2])
    next_mem = jnp.stack([target,
                          pot_ok.astype(jnp.int32),
                          cache0.astype(jnp.int32),
                          cache1.astype(jnp.int32),
                          cache2.astype(jnp.int32),
                          empty_streak]).astype(jnp.int32)
    trace = {"pos": (row * _WIDTH + col).astype(jnp.int32),
             "target": target.astype(jnp.int32),
             "goal": task[0].astype(jnp.int32),
             "action": action}
    return action, next_mem, trace


# --------------------------------------------------------------------------
# agent_1 : right room.  It cannot see the recipe, so it never guesses one: it
# only ever copies the pure ingredient already sitting in the pot, then plates
# the finished dish and delivers it.
# --------------------------------------------------------------------------
_MEM_1 = (0, 0, 0)  # hold wait, hint failures, steps the pot has stood empty
_HOLD_LIMIT = 30
_HINT_LIMIT = 2
_STALL_LIMIT = 25  # after this the pot is clearly not being seeded by the partner


def init_agent_1():
    return jnp.asarray(_MEM_1, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)
    mem = jnp.where(reset, jnp.asarray(_MEM_1, dtype=jnp.int32), memory)

    row, col, cell_idx = _localize(obs, _ROWS_1, _COLS_1)
    facing, held_plate, held_cooked, has_raw, ing = _self_state(obs)

    _, pot_cell = _read(obs, row, col, _POT_R, _POT_C)
    total, kind, pure, addable, ready, timer = _pot_state(pot_cell)
    busy = (timer > 0) | ready | (total >= 3)
    # Only a pot holding a single raw ingredient type is safe to copy.  (Waiting
    # instead for the partner to confirm a seed that was dropped in before our
    # own delivery resampled the order was measured to cost far more throughput
    # than the occasional wrong dish it avoids, so we simply help at once.)
    needs_load = addable & pure & (kind < 2)

    # A partner carrying a raw ingredient towards an empty pot is telling us,
    # by plain evidence rather than by convention, which dish is being made.
    partner_counts = jnp.sum(obs[:, :, 17:20], axis=(0, 1))
    partner_kind = jnp.argmax(partner_counts).astype(jnp.int32)
    partner_raw = ((jnp.sum(partner_counts) > 0)
                   & (jnp.sum(obs[:, :, 15]) == 0) & (jnp.sum(obs[:, :, 16]) == 0))
    hint = (partner_raw & (total == 0) & jnp.logical_not(busy)
            & (partner_kind < 2) & (mem[1] < _HINT_LIMIT))
    # While the pot is empty it is worth holding an ingredient for a while: the
    # partner is probably about to seed the pot with that same type.
    empty_pot = (total == 0) & jnp.logical_not(busy)
    patient = empty_pot & (mem[0] < _HOLD_LIMIT)
    # A pot left empty this long means the partner is not going to seed it, so
    # act on whatever it has shown us about the wanted ingredient.
    stalled = empty_pot & (mem[2] >= _STALL_LIMIT)

    # Counters available for parking an item we cannot use.
    vis_a, cell_a = _read(obs, row, col, 0, 9)
    vis_b, cell_b = _read(obs, row, col, 4, 9)
    free_a = jnp.logical_not(vis_a & (jnp.sum(cell_a[29:34]) > 0))
    free_b = jnp.logical_not(vis_b & (jnp.sum(cell_b[29:34]) > 0))

    # Raw ingredients the partner may have left on the counters between rooms.
    _, cell_top = _read(obs, row, col, 1, 7)
    _, cell_bottom = _read(obs, row, col, 3, 7)
    top_kind = jnp.argmax(cell_top[31:34]).astype(jnp.int32)
    bottom_kind = jnp.argmax(cell_bottom[31:34]).astype(jnp.int32)
    top_raw = (jnp.sum(cell_top[31:34]) > 0) & (jnp.sum(cell_top[29:31]) == 0) & (top_kind < 2)
    bottom_raw = ((jnp.sum(cell_bottom[31:34]) > 0) & (jnp.sum(cell_bottom[29:31]) == 0)
                  & (bottom_kind < 2))

    wait_at_pot = _task(2, _LEFT, False)
    collect = _task(2, _LEFT, True)
    sup_goal = jnp.asarray(_SUP_GOAL_1, dtype=jnp.int32)
    sup_dir = jnp.asarray(_SUP_DIR_1, dtype=jnp.int32)

    # Four parking slots keep us from ever being stuck with a useless item.
    dump = _pick(jnp.sum(cell_bottom[29:34]) == 0, wait_at_pot, _task(1, _LEFT, True))
    dump = _pick(jnp.sum(cell_top[29:34]) == 0, dump, _task(0, _LEFT, True))
    dump = _pick(free_b, dump, _task(5, _DOWN, True))
    dump = _pick(free_a, dump, _task(3, _UP, True))

    # Empty handed: help fill the pot, otherwise fetch the plate in advance.
    fetch_kind = jnp.where(needs_load, kind, partner_kind)
    task = _pick(busy, wait_at_pot, _task(3, _RIGHT, True))
    task = _pick(needs_load | hint,
                 task, (sup_goal[fetch_kind], sup_dir[fetch_kind], jnp.asarray(True)))
    # Stand-off: take an ingredient the partner parked on a shared counter.
    counter_take = _pick(top_raw, _task(1, _LEFT, True), _task(0, _LEFT, True))
    task = _pick(stalled & (top_raw | bottom_raw), task, counter_take)

    # Carrying an ingredient: add it only if the pot already shows that type,
    # or if the pot has stood empty long enough that nobody else will seed it.
    carry = _pick(patient, dump, wait_at_pot)
    carry = _pick((needs_load & (ing == kind)) | (stalled & addable), carry, collect)
    task = _pick(has_raw, task, carry)

    # Carrying an empty plate: collect the dish, or free the hands to load.
    plate_carry = _pick(ready, wait_at_pot, collect)
    plate_carry = _pick(needs_load & jnp.logical_not(ready), plate_carry, dump)
    task = _pick(held_plate & jnp.logical_not(held_cooked), task, plate_carry)

    # Carrying the finished dish.  Every recipe this layout can ask for is three
    # units of one ingredient, so a mixed dish is certainly wrong: park it on a
    # counter instead of paying the failed-delivery penalty for it.
    held_counts = obs[2, 2, 7:10]
    deliverable = (jnp.sum(held_counts) == 3) & (held_counts[ing] == 3) & (ing < 2)
    deliver = _task(4, _RIGHT, True)
    park = _pick(free_a | free_b, deliver, dump)
    task = _pick(held_cooked, task, _pick(deliverable, park, deliver))

    action = _execute(_NAV_1, cell_idx, facing, task[0], task[1], task[2])
    waiting = has_raw & (total == 0) & jnp.logical_not(busy)
    gave_up = waiting & (mem[0] == _HOLD_LIMIT)  # counted once per abandoned wait
    next_mem = jnp.stack([jnp.where(waiting, mem[0] + 1, 0),
                          mem[1] + gave_up.astype(jnp.int32),
                          jnp.where(empty_pot, jnp.minimum(mem[2] + 1, 99), 0)
                          ]).astype(jnp.int32)
    trace = {"pos": (row * _WIDTH + col).astype(jnp.int32),
             "pot_n": total.astype(jnp.int32),
             "goal": task[0].astype(jnp.int32),
             "action": action}
    return action, next_mem, trace
