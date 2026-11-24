"""Role-specialized coordination policy for demo_cook_simple (Overcooked V2).

Map facts (fixed, from context/environment/README.md and specification.json):
  - agent_0's room (rows 1-3, cols 1-6) is a fully open floor rectangle with
    access to all three ingredient supplies and to both always-on recipe
    indicators, but no access to any plate pile or the delivery station.
  - agent_1's room (rows 1-3, cols 8-9) is a fully open floor rectangle with
    access to ingredient supplies 0 and 1, the plate piles, and the delivery
    station, but the recipe indicators are always outside its view radius.
  - The two rooms are physically disconnected (no floor path between them);
    the only shared cell is the pot at (2, 7), reachable from each room's own
    adjacent access cell. This asymmetry is a fixed environment fact, not a
    private convention, so each seat is given role logic matched to what only
    that seat can ever observe/do.

Agent_0 therefore reads the recipe and keeps the pot topped up with the
correct ingredient. Agent_1 cannot see the recipe, so it never guesses one
blindly; it only mirrors whatever ingredient type is already progressing in
the pot (topping off partially filled correct batches), and otherwise handles
plate fetching, scooping, and delivery. Both seats self-localize every step
via template matching of the static-object channels against the known fixed
map, so no dead-reckoning/drift is possible.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'room_role_specialist',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 5,
    'max_width': 11,
    'num_ingredients': 3,
}

# ---------------------------------------------------------------------------
# Static map facts (fixed layout; embedding known geometry is not host I/O).
# ---------------------------------------------------------------------------

POT_POS = (2, 7)

# Direction ids in the observation / facing channels: UP=0, DOWN=1, RIGHT=2, LEFT=3
# Movement action ids: RIGHT=0, DOWN=1, LEFT=2, UP=3, STAY=4, INTERACT=5
STAY = 4
INTERACT = 5


def _padded_static_map():
    """Build the (9, 15, 9) zero-padded static-object map as a JAX constant.

    Built from plain Python literals inside a function (no module-scope calls,
    no numpy/host I/O): channel order matches obs channels 20..28, i.e.
    wall, goal, pot, recipe_indicator, button_indicator, plate_pile,
    ingredient0, ingredient1, ingredient2 supply.
    """
    walls = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 10),
        (1, 7), (2, 0), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
        (4, 7), (4, 9), (4, 10),
    ]
    goal = [(2, 10)]
    pot = [(2, 7)]
    recipe_ind = [(0, 5), (4, 5)]
    button_ind = []
    plate_pile = [(1, 10), (3, 10)]
    ing0 = [(0, 8), (1, 0)]
    ing1 = [(3, 0), (4, 8)]
    ing2 = [(0, 6), (4, 6)]
    layers = [walls, goal, pot, recipe_ind, button_ind, plate_pile, ing0, ing1, ing2]

    rows = []
    for rr in range(5):
        row = []
        for cc in range(11):
            cell = [1 if (rr, cc) in layer else 0 for layer in layers]
            row.append(cell)
        rows.append(row)
    static_map = jnp.asarray(rows, dtype=jnp.int32)
    return jnp.pad(static_map, ((2, 2), (2, 2), (0, 0)))


def _agent0_cells():
    return jnp.asarray(
        [[rr, cc] for rr in (1, 2, 3) for cc in (1, 2, 3, 4, 5, 6)], dtype=jnp.int32
    )


def _agent1_cells():
    return jnp.asarray([[rr, cc] for rr in (1, 2, 3) for cc in (8, 9)], dtype=jnp.int32)


def _face_to_action():
    return jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)


def _localize(obs_static, candidate_cells):
    """Exact self-localization via template matching of static channels."""
    padded = _padded_static_map()

    def crop_at(rc):
        return jax.lax.dynamic_slice(padded, (rc[0], rc[1], 0), (5, 5, 9))

    crops = jax.vmap(crop_at)(candidate_cells)
    diffs = jnp.sum(jnp.abs(crops - obs_static[None]), axis=(1, 2, 3))
    idx = jnp.argmin(diffs)
    chosen = candidate_cells[idx]
    return chosen[0], chosen[1]


def _nav_action(cur_r, cur_c, target_r, target_c, want_face, cur_facing):
    """Greedy movement toward (target_r, target_c); face+interact once there.

    want_face < 0 means "no facing requirement, just arrive and stay put".
    Both rooms are fully open rectangles, so simple axis-priority movement
    never gets stuck on an obstacle.
    """
    at_target = (cur_r == target_r) & (cur_c == target_c)
    move_row = jnp.where(target_r > cur_r, 1, 3)  # DOWN else UP
    move_col = jnp.where(target_c > cur_c, 0, 2)  # RIGHT else LEFT
    action_travel = jnp.where(cur_r != target_r, move_row, move_col)

    want_face_idx = jnp.clip(want_face, 0, 3)
    face_action = _face_to_action()[want_face_idx]
    action_if_face_needed = jnp.where(cur_facing == want_face, INTERACT, face_action)
    action_at_target = jnp.where(want_face >= 0, action_if_face_needed, STAY)

    action = jnp.where(at_target, action_at_target, action_travel)
    return action.astype(jnp.int32)


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Agent 0: recipe-aware ingredient gatherer (fills the pot).
# ---------------------------------------------------------------------------

_A0_POT_R, _A0_POT_C, _A0_POT_FACE = 2, 6, 2       # face RIGHT toward pot (2,7)
_A0_ING0_R, _A0_ING0_C, _A0_ING_FACE = 1, 1, 3      # face LEFT toward (1,0)
_A0_ING1_R, _A0_ING1_C = 3, 1                       # face LEFT toward (3,0)


def init_agent_0():
    return {
        'target_ing': jnp.asarray(0, dtype=jnp.int32),
        'pot_ing0': jnp.asarray(0, dtype=jnp.int32),
        'pot_ing1': jnp.asarray(0, dtype=jnp.int32),
        'pot_busy': jnp.asarray(0, dtype=jnp.int32),
        'recipe_known': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_0_act(obs, reset, memory, rng):
    del rng
    init = init_agent_0()
    mem = jax.tree_util.tree_map(lambda i, m: jnp.where(reset, i, m), init, memory)

    static_slice = obs[:, :, 20:29]
    r, c = _localize(static_slice, _agent0_cells())

    # The recipe is guaranteed constant for the whole lifetime of a batch
    # (it only resamples on a correct delivery, which necessarily ends the
    # previous batch), so it is always safe to fetch using the latest
    # opportunistic reading rather than whatever type currently sits in the
    # pot: using the pot's own contents to pick the fetch target instead
    # would let one stale pre-fetched ingredient perpetuate the wrong type
    # forever across batch boundaries.
    rec0_vis = jnp.max(obs[:, :, 36]) > 0
    rec1_vis = jnp.max(obs[:, :, 37]) > 0
    recipe_visible = rec0_vis | rec1_vis
    new_target = jnp.where(rec0_vis, 0, 1).astype(jnp.int32)
    target_ing = jnp.where(recipe_visible, new_target, mem['target_ing']).astype(jnp.int32)
    recipe_known = jnp.where(recipe_visible, 1, mem['recipe_known']).astype(jnp.int32)

    pot_r_local = POT_POS[0] - r + 2
    pot_c_local = POT_POS[1] - c + 2
    pot_visible = (
        (pot_r_local >= 0) & (pot_r_local < 5) & (pot_c_local >= 0) & (pot_c_local < 5)
    )
    pr = jnp.clip(pot_r_local, 0, 4)
    pc = jnp.clip(pot_c_local, 0, 4)
    ing0_at_pot = obs[pr, pc, 31]
    ing1_at_pot = obs[pr, pc, 32]
    cooked_at_pot = obs[pr, pc, 30]
    timer_at_pot = obs[pr, pc, 39]
    obs_pot_busy = ((timer_at_pot > 0) | (cooked_at_pot > 0)).astype(jnp.int32)

    pot_ing0 = jnp.where(pot_visible, ing0_at_pot, mem['pot_ing0']).astype(jnp.int32)
    pot_ing1 = jnp.where(pot_visible, ing1_at_pot, mem['pot_ing1']).astype(jnp.int32)
    pot_busy = jnp.where(pot_visible, obs_pot_busy, mem['pot_busy']).astype(jnp.int32)

    held_ing0 = obs[2, 2, 7]
    held_ing1 = obs[2, 2, 8]
    holding = (held_ing0 + held_ing1) > 0

    cur_facing = _facing(obs)

    action_deposit = _nav_action(r, c, _A0_POT_R, _A0_POT_C, _A0_POT_FACE, cur_facing)
    action_fetch0 = _nav_action(r, c, _A0_ING0_R, _A0_ING0_C, _A0_ING_FACE, cur_facing)
    action_fetch1 = _nav_action(r, c, _A0_ING1_R, _A0_ING1_C, _A0_ING_FACE, cur_facing)
    action_wait = _nav_action(r, c, _A0_POT_R, _A0_POT_C, -1, cur_facing)

    action_fetch = jnp.where(target_ing == 0, action_fetch0, action_fetch1)
    # Never fetch until the recipe has actually been observed at least once:
    # fetching on the arbitrary init-time default guess can pick up the wrong
    # ingredient before ever seeing the real recipe, and since that held item
    # cannot change type, it would still get deposited (wrong) once true
    # target_ing updates mid-transit. Waiting at the pot (which also sees the
    # recipe indicator) always resolves this within a step or two.
    # Never fetch a "spare" ingredient while the pot is cooking/ready either:
    # doing so ahead of time is what let one stale pre-fetch survive across a
    # recipe change (which only happens on delivery) and contaminate the next
    # batch. Only fetch once the pot is confirmed idle and the recipe known.
    action_empty = jnp.where((pot_busy > 0) | (recipe_known == 0), action_wait, action_fetch)

    # In self-play our own agent_1 never touches raw ingredients, so the pot's
    # contents always match whatever agent_0 itself put there and this branch
    # never fires. But in cross-play an independently generated partner may
    # also have access to the pot (its room includes ingredient supplies 0/1)
    # and could deposit a mismatched type; the interact rule for a pot only
    # requires "room in an idle, uncooked pot" (no type check), so blindly
    # depositing onto an already-different-type pot would compound the
    # contamination. Wait (do not deposit, do not discard) instead: the pot
    # can only be cleared by cooking + scooping, which always eventually
    # happens (scoop only needs an empty plate on a cooked pot), so this is a
    # bounded wait, not a deadlock -- once the pot empties, deposit resumes.
    pot_mismatch = ((target_ing == 0) & (pot_ing1 > 0)) | ((target_ing == 1) & (pot_ing0 > 0))
    action_deposit_or_wait = jnp.where(pot_mismatch, action_wait, action_deposit)
    action = jnp.where(holding, action_deposit_or_wait, action_empty).astype(jnp.int32)

    next_mem = {
        'target_ing': target_ing,
        'pot_ing0': pot_ing0,
        'pot_ing1': pot_ing1,
        'pot_busy': pot_busy,
        'recipe_known': recipe_known,
    }
    return action, next_mem, {}


# ---------------------------------------------------------------------------
# Agent 1: plate/scoop/delivery handler. Never touches raw ingredients (it can
# never see the recipe indicator, so it cannot safely choose or verify an
# ingredient type); this also keeps it deadlock-free, since its inventory can
# only ever be empty, an empty plate, or a finished dish, each of which has an
# always-available next action.
# ---------------------------------------------------------------------------

_A1_POT_R, _A1_POT_C, _A1_POT_FACE = 2, 8, 3         # face LEFT toward pot (2,7)
_A1_DELIVER_R, _A1_DELIVER_C, _A1_DELIVER_FACE = 2, 9, 2  # face RIGHT toward (2,10)
_A1_PLATE_R, _A1_PLATE_C, _A1_PLATE_FACE = 1, 9, 2   # face RIGHT toward (1,10)


def init_agent_1():
    return {'dummy': jnp.asarray(0, dtype=jnp.int32)}


def agent_1_act(obs, reset, memory, rng):
    del rng
    init = init_agent_1()
    mem = jax.tree_util.tree_map(lambda i, m: jnp.where(reset, i, m), init, memory)

    static_slice = obs[:, :, 20:29]
    r, c = _localize(static_slice, _agent1_cells())

    plate = obs[2, 2, 5]
    cooked = obs[2, 2, 6]
    holding_dish = (plate > 0) & (cooked > 0)
    holding_plate = (plate > 0) & (cooked == 0)

    cur_facing = _facing(obs)

    action_deliver = _nav_action(
        r, c, _A1_DELIVER_R, _A1_DELIVER_C, _A1_DELIVER_FACE, cur_facing
    )
    action_to_pot = _nav_action(r, c, _A1_POT_R, _A1_POT_C, _A1_POT_FACE, cur_facing)
    action_to_plate = _nav_action(
        r, c, _A1_PLATE_R, _A1_PLATE_C, _A1_PLATE_FACE, cur_facing
    )

    action = jnp.where(
        holding_dish,
        action_deliver,
        jnp.where(holding_plate, action_to_pot, action_to_plate),
    ).astype(jnp.int32)

    return action, mem, {}
