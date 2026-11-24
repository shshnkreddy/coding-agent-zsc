"""Overcooked V2 (grounded_coord_ring) policy.

Design summary
--------------
The map is fixed (random_reset=false): only spawn position/facing and the
target recipe are randomized. agent_0 lives on a 24-cell outer ring that
never touches the pot; agent_1 lives on an 8-cell inner ring that
surrounds the pot on all four sides. Both rings are literal graph cycles,
so once we know *where on the ring* we are, navigation is a 1-D problem
(walk clockwise or counter-clockwise to a target ring index). The two
rings are also topologically disconnected: every boundary cell between
them is an impassable object (supply/counter/button/plate/pot), never
floor, so neither seat's policy can ever physically enter the other
seat's territory, no matter what either policy does.

Self-localization: every one of the 32 floor cells has a unique 5x5 crop
of the static-object channels (walls/goal/pot/recipe/button/plate/
ingredient piles), which never change during an episode. We precompute
the expected crop for every cell of each agent's own ring and match the
live observation against that table (nearest neighbour) to recover an
exact ring index every step. No global coordinates are ever received or
required as input; this is derived purely from the local observation.

agent_1 (pot owner) is fully self-sufficient: it cannot see the
always-on recipe indicators from inside its ring, so it briefly
activates the adjacent button (small one-time cost) to read the target
recipe, then fetches three matching ingredients, cooks, plates and
delivers, entirely within its own reachable cells. This does not depend
on any convention with the partner, so it is robust to an independently
generated agent_0.

agent_0 (outer ring) can never reach the pot and therefore can never
complete a delivery by itself. It can, however, see an always-on recipe
indicator for free and reach the shared ingredient supplies / shared
counters that sit on the boundary between the two rings. It uses this
free information to continuously ferry the recipe-matching ingredient
onto the nearest shared counter next to the pot room, which is a
best-effort, no-downside way to help an unknown partner that might look
for staged ingredients there (agent_0 can never lose team reward: it
never presses a button and blocked/ineffective actions cost nothing).

All lookup tables below are built from pure Python constants inside
plain functions (no module-scope calls, no numpy), then converted to
JAX arrays; this only runs during tracing/compilation, not per step.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}


_ASCII_GRID = (
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

_CHAR_TO_CHANNEL = {
    'W': 0,  # wall/counter
    'X': 1,  # goal / delivery
    'P': 2,  # pot
    'R': 3,  # always-on recipe indicator
    'L': 4,  # button recipe indicator
    'B': 5,  # plate pile
    '0': 6,  # ingredient 0 supply
    '1': 7,  # ingredient 1 supply
    '2': 8,  # ingredient 2 supply
}

# Outer ring (agent_0): 24 cells, cyclic order, never adjacent to the pot.
_AGENT0_CELLS = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
    (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
)

# Inner ring (agent_1): 8 cells around the pot, cyclic order.
# 0:(3,3) ing0 corner, 1:(3,4) pot-N/button-N, 2:(3,5) plate corner,
# 3:(4,5) pot-E/deliver-E, 4:(5,5) plate corner, 5:(5,4) pot-S/button-S,
# 6:(5,3) ing1 corner, 7:(4,3) pot-W/deliver-W.
_AGENT1_CELLS = (
    (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (4, 3),
)

# Movement action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
# Facing codes as they appear in the observation (channels 1..4 / 11..14).
FACING_UP, FACING_DOWN, FACING_RIGHT, FACING_LEFT = 0, 1, 2, 3

# Ring indices used by agent_1 (see _AGENT1_CELLS ordering above).
_ING0_CORNER, _POT_N, _PLATE_NE, _POT_E, _PLATE_SE, _POT_S, _ING1_CORNER, _POT_W = 0, 1, 2, 3, 4, 5, 6, 7

# Ring indices used by agent_0 waypoints (see _AGENT0_CELLS ordering above).
_CHECK_IDX = 21   # (4, 1): adjacent to the always-on recipe indicator at (4, 0).
_SUPPLY0_IDX = 22  # (3, 1): faces RIGHT into ingredient-0 supply at (3, 2).
_SUPPLY1_IDX = 20  # (5, 1): faces RIGHT into ingredient-1 supply at (5, 2).
_COUNTER0_IDX = 2  # (1, 3): faces DOWN onto shared counter (2, 3).
_COUNTER1_IDX = 16  # (7, 3): faces UP onto shared counter (6, 3).


def _build_static_grid():
    height = len(_ASCII_GRID)
    width = len(_ASCII_GRID[0])
    grid = [[[0] * 9 for _ in range(width)] for _ in range(height)]
    for r, row in enumerate(_ASCII_GRID):
        for c, ch in enumerate(row):
            channel = _CHAR_TO_CHANNEL.get(ch)
            if channel is not None:
                grid[r][c][channel] = 1
    return grid, height, width


def _crop_flat(grid, height, width, row, col):
    out = []
    for i in range(5):
        for j in range(5):
            gr = row + i - 2
            gc = col + j - 2
            if 0 <= gr < height and 0 <= gc < width:
                out.extend(grid[gr][gc])
            else:
                out.extend([0] * 9)
    return out


def _opposite(action):
    return {RIGHT: LEFT, DOWN: UP, LEFT: RIGHT, UP: DOWN}[action]


def _forward_actions(cells):
    n = len(cells)
    fw = []
    for i in range(n):
        r1, c1 = cells[i]
        r2, c2 = cells[(i + 1) % n]
        if c2 - c1 == 1:
            fw.append(RIGHT)
        elif c2 - c1 == -1:
            fw.append(LEFT)
        elif r2 - r1 == 1:
            fw.append(DOWN)
        else:
            fw.append(UP)
    return fw


def _backward_actions(fw):
    n = len(fw)
    return [_opposite(fw[(i - 1) % n]) for i in range(n)]


def _build_tables():
    grid, height, width = _build_static_grid()
    table0 = [_crop_flat(grid, height, width, r, c) for (r, c) in _AGENT0_CELLS]
    table1 = [_crop_flat(grid, height, width, r, c) for (r, c) in _AGENT1_CELLS]
    fw0 = _forward_actions(_AGENT0_CELLS)
    bw0 = _backward_actions(fw0)
    fw1 = _forward_actions(_AGENT1_CELLS)
    bw1 = _backward_actions(fw1)
    return (
        jnp.asarray(table0, dtype=jnp.int32),
        jnp.asarray(table1, dtype=jnp.int32),
        jnp.asarray(fw0, dtype=jnp.int32),
        jnp.asarray(bw0, dtype=jnp.int32),
        jnp.asarray(fw1, dtype=jnp.int32),
        jnp.asarray(bw1, dtype=jnp.int32),
    )


def _localize(obs, table):
    local_static = obs[:, :, 20:29].reshape(-1).astype(jnp.int32)
    diffs = table - local_static[None, :]
    scores = jnp.sum(diffs * diffs, axis=1)
    return jnp.argmin(scores).astype(jnp.int32)


def _self_facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _action_for_facing(desired_facing):
    table = jnp.asarray([UP, DOWN, RIGHT, LEFT], dtype=jnp.int32)
    return table[desired_facing]


def _goto(self_idx, self_facing, target_idx, desired_facing, ring_size, forward, backward, ready):
    fwd = jnp.mod(target_idx - self_idx, ring_size)
    bwd = ring_size - fwd
    move_action = jnp.where(fwd <= bwd, forward[self_idx], backward[self_idx])
    at_target = self_idx == target_idx
    facing_ok = self_facing == desired_facing
    turn_action = _action_for_facing(desired_facing)
    action = jnp.where(
        jnp.logical_not(at_target),
        move_action,
        jnp.where(jnp.logical_not(facing_ok), turn_action,
                  jnp.where(ready, INTERACT, STAY)),
    )
    return action.astype(jnp.int32), jnp.logical_and(at_target, facing_ok)


def _goto_no_facing(self_idx, target_idx, ring_size, forward, backward):
    fwd = jnp.mod(target_idx - self_idx, ring_size)
    bwd = ring_size - fwd
    move_action = jnp.where(fwd <= bwd, forward[self_idx], backward[self_idx])
    at_target = self_idx == target_idx
    action = jnp.where(jnp.logical_not(at_target), move_action, STAY)
    return action.astype(jnp.int32), at_target


def _phase_array(mapping):
    return jnp.asarray([mapping.get(p, 0) for p in range(7)], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# agent_1: self-sufficient pot owner.
# ---------------------------------------------------------------------------

def init_agent_1():
    return {
        'phase': jnp.asarray(0, dtype=jnp.int32),
        'fill_count': jnp.asarray(0, dtype=jnp.int32),
        'target_ing': jnp.asarray(0, dtype=jnp.int32),
        'started': jnp.asarray(False),
    }


def agent_1_act(obs, reset, memory, rng):
    del rng
    _, table1, _, _, fw1, bw1 = _build_tables()
    obs = obs.astype(jnp.int32)
    self_idx = _localize(obs, table1)
    self_facing = _self_facing(obs)

    phase = memory['phase']
    fill_count = memory['fill_count']
    target_ing = memory['target_ing']
    started = memory['started']

    # Per-phase (0,2,3,4,5,6) target ring index / desired facing, split by
    # which ingredient (A=ingredient 0, B=ingredient 1) the recipe needs.
    # Phase 1 has no navigation of its own; it is folded into phase 2 below.
    target_a = _phase_array({0: _POT_N, 2: _ING0_CORNER, 3: _POT_N, 4: _PLATE_NE, 5: _POT_N, 6: _POT_W})
    target_b = _phase_array({0: _POT_N, 2: _ING1_CORNER, 3: _POT_S, 4: _PLATE_SE, 5: _POT_S, 6: _POT_W})
    facing_a = _phase_array({0: FACING_UP, 2: FACING_LEFT, 3: FACING_DOWN, 4: FACING_RIGHT, 5: FACING_DOWN, 6: FACING_LEFT})
    facing_b = _phase_array({0: FACING_UP, 2: FACING_LEFT, 3: FACING_UP, 4: FACING_RIGHT, 5: FACING_UP, 6: FACING_LEFT})

    # Read the recipe from the button cell, valid only right after phase 0's
    # activation (self is at _POT_N, button sits one cell north -> local (1,2)).
    recipe_ing0 = obs[1, 2, 36]
    recipe_ing1 = obs[1, 2, 37]
    read_target_ing = jnp.where(recipe_ing0 > recipe_ing1, 0, 1).astype(jnp.int32)
    target_ing = jnp.where(phase == 1, read_target_ing, target_ing)

    # Navigation phase actually used this step: phase 1 immediately behaves
    # like phase 2 (walk toward the matching ingredient corner).
    working_phase = jnp.where(phase == 1, 2, phase)

    target_idx = jnp.where(target_ing == 0, target_a[working_phase], target_b[working_phase])
    desired_facing = jnp.where(target_ing == 0, facing_a[working_phase], facing_b[working_phase])

    # Pot cooked-flag gate, only meaningful (and only checked) in phase 5.
    pot_row = jnp.where(target_ing == 0, 3, 1)
    pot_col = 2
    cooked = obs[pot_row, pot_col, 30] > 0
    ready = jnp.where(working_phase == 5, cooked, True)
    # Guard against a spawn coincidence where the agent is already at the
    # button, correctly facing it, on the very first decision of the
    # episode: without this, a same-cell activation could be immediately
    # followed by an indistinguishable reactivation, wasting an extra -5.
    ready = jnp.where(jnp.logical_and(working_phase == 0, jnp.logical_not(started)), False, ready)

    action, at_and_facing_ok = _goto(
        self_idx, self_facing, target_idx, desired_facing, 8, fw1, bw1, ready)
    did_interact = jnp.logical_and(at_and_facing_ok, action == INTERACT)

    new_phase = phase
    new_phase = jnp.where(jnp.logical_and(phase == 0, did_interact), 1, new_phase)
    new_phase = jnp.where(phase == 1, 2, new_phase)
    new_phase = jnp.where(jnp.logical_and(phase == 2, did_interact), 3, new_phase)
    filled_enough = (fill_count + 1) >= 3
    new_phase = jnp.where(
        jnp.logical_and(phase == 3, did_interact),
        jnp.where(filled_enough, 4, 2),
        new_phase,
    )
    new_phase = jnp.where(jnp.logical_and(phase == 4, did_interact), 5, new_phase)
    new_phase = jnp.where(jnp.logical_and(phase == 5, did_interact), 6, new_phase)
    new_phase = jnp.where(jnp.logical_and(phase == 6, did_interact), 0, new_phase)

    new_fill_count = jnp.where(jnp.logical_and(phase == 3, did_interact), fill_count + 1, fill_count)
    new_fill_count = jnp.where(jnp.logical_and(phase == 6, did_interact), 0, new_fill_count)

    init = init_agent_1()
    next_memory = {
        'phase': jnp.where(reset, init['phase'], new_phase.astype(jnp.int32)),
        'fill_count': jnp.where(reset, init['fill_count'], new_fill_count.astype(jnp.int32)),
        'target_ing': jnp.where(reset, init['target_ing'], target_ing.astype(jnp.int32)),
        'started': jnp.asarray(True),
    }
    return action.astype(jnp.int32), next_memory, {}


# ---------------------------------------------------------------------------
# agent_0: cannot reach the pot; ferries the recipe-matching ingredient to a
# shared counter as a free, no-downside assist for an unknown partner.
# ---------------------------------------------------------------------------

def init_agent_0():
    return {
        'phase': jnp.asarray(0, dtype=jnp.int32),
        'target_ing': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    table0, _, fw0, bw0, _, _ = _build_tables()
    obs = obs.astype(jnp.int32)
    self_idx = _localize(obs, table0)
    self_facing = _self_facing(obs)

    phase = memory['phase']
    target_ing = memory['target_ing']

    # Phase 0: walk to the checkpoint next to a free, always-on recipe
    # indicator; no facing requirement to view it.
    action0, arrived0 = _goto_no_facing(self_idx, _CHECK_IDX, 24, fw0, bw0)
    recipe_ing0 = obs[2, 1, 36]
    recipe_ing1 = obs[2, 1, 37]
    read_target_ing = jnp.where(recipe_ing0 > recipe_ing1, 0, 1).astype(jnp.int32)
    target_ing_after0 = jnp.where(arrived0, read_target_ing, target_ing)

    supply_idx = jnp.where(target_ing_after0 == 0, _SUPPLY0_IDX, _SUPPLY1_IDX)
    action1, at1 = _goto(self_idx, self_facing, supply_idx, FACING_RIGHT, 24, fw0, bw0, True)

    counter_idx = jnp.where(target_ing == 0, _COUNTER0_IDX, _COUNTER1_IDX)
    counter_facing = jnp.where(target_ing == 0, FACING_DOWN, FACING_UP)
    action2, at2 = _goto(self_idx, self_facing, counter_idx, counter_facing, 24, fw0, bw0, True)

    action = jnp.where(
        phase == 0, jnp.where(arrived0, action1, action0),
        jnp.where(phase == 1, action1, action2),
    ).astype(jnp.int32)

    did_interact1 = jnp.logical_and(phase == 1, jnp.logical_and(at1, action == INTERACT))
    did_interact2 = jnp.logical_and(phase == 2, jnp.logical_and(at2, action == INTERACT))

    new_phase = phase
    new_phase = jnp.where(jnp.logical_and(phase == 0, arrived0), 1, new_phase)
    new_phase = jnp.where(did_interact1, 2, new_phase)
    new_phase = jnp.where(did_interact2, 0, new_phase)

    new_target_ing = jnp.where(phase == 0, target_ing_after0, target_ing)

    init = init_agent_0()
    next_memory = {
        'phase': jnp.where(reset, init['phase'], new_phase.astype(jnp.int32)),
        'target_ing': jnp.where(reset, init['target_ing'], new_target_ing.astype(jnp.int32)),
    }
    return action.astype(jnp.int32), next_memory, {}
