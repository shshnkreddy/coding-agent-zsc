"""A geometry-driven first cooperative candidate.

The two rooms are disconnected for walking but share the two pots in the
boundary wall.  The lower seat can see the recipe indicator and uses one
ingredient-3 unit as a temporary, unambiguous marker.  The upper seat reads
that marker and prepares the other pot.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'pot_marker_v1',
    'adapter': 'overcooked_v2',
    'adapter_version': 1,
    'max_height': 7,
    'max_width': 6,
    'num_ingredients': 4,
}


# Static channels 22:32, in their observation order.  Keeping the map in the
# policy makes position recovery independent of the random reset location.
_STATIC = (
    [[1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0,0]],
    [[0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0,0]],
    [[1,0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,0]],
    [[1,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]],
    [[1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]],
)


def _position(obs):
    """Match the observed static crop against the known padded board."""
    static = jnp.asarray(_STATIC, dtype=jnp.int32)
    padded = jnp.pad(static, ((2, 2), (2, 2), (0, 0)))
    cand_r = jnp.repeat(jnp.arange(7, dtype=jnp.int32), 6)
    cand_c = jnp.tile(jnp.arange(6, dtype=jnp.int32), 7)
    dr = jnp.arange(5, dtype=jnp.int32) - 2
    dc = jnp.arange(5, dtype=jnp.int32) - 2
    rows = cand_r[:, None, None] + dr[None, :, None] + 2
    cols = cand_c[:, None, None] + dc[None, None, :] + 2
    crops = padded[rows, cols, :]
    error = jnp.sum(jnp.abs(crops - obs[None, :, :, 22:32]), axis=(1, 2, 3))
    idx = jnp.argmin(error)
    return cand_r[idx], cand_c[idx]


def _self_cell(obs, start, stop):
    return obs[2, 2, start:stop]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5])


def _grid_cell(obs, pr, pc, row, col, start, stop):
    rr = jnp.clip(row - pr + 2, 0, 4)
    cc = jnp.clip(col - pc + 2, 0, 4)
    return obs[rr, cc, start:stop]


def _move_or_face(pr, pc, tr, tc, facing, desired_face):
    horizontal = jnp.where(tc > pc, jnp.asarray(0, jnp.int32), jnp.asarray(2, jnp.int32))
    vertical = jnp.where(tr > pr, jnp.asarray(1, jnp.int32), jnp.asarray(3, jnp.int32))
    move = jnp.where(pr != tr, vertical, jnp.where(pc != tc, horizontal, jnp.asarray(4, jnp.int32)))
    # Facing ids are UP=0, DOWN=1, RIGHT=2, LEFT=3, while action ids use
    # RIGHT=0, DOWN=1, LEFT=2, UP=3.
    face_action = jnp.where(
        desired_face == 0, 3,
        jnp.where(desired_face == 1, 1, jnp.where(desired_face == 2, 0, 2)),
    )
    return jnp.where(
        (pr == tr) & (pc == tc) & (facing != desired_face),
        face_action,
        move,
    )


def _top_act(obs, reset, memory):
    pr, pc = _position(obs)
    facing = _facing(obs)
    phase = jnp.where(reset, 0, memory[0])
    recipe = memory[1]

    left_marker = _grid_cell(obs, pr, pc, 3, 1, 34, 38)[3]
    right_marker = _grid_cell(obs, pr, pc, 3, 3, 34, 38)[3]
    marker_seen = (left_marker > 0) | (right_marker > 0)
    discovered_marker_col = jnp.where(left_marker > 0, 1, 3)
    marker_amount = jnp.where(left_marker > 0, left_marker, right_marker)
    # Count-coded markers are the primary convention.  A right-side single
    # marker remains a fallback for the earlier canonical partner convention.
    discovered_recipe = jnp.where(marker_amount >= 2, 1, jnp.where(right_marker > 0, 1, 0))
    at_scan = (pr == 2) & (pc == 2)
    wait_count = jnp.where(reset, 0, memory[3])
    left_single = (left_marker == 1) & (right_marker == 0)
    signal_ready = (marker_amount >= 2) | (right_marker > 0) | (left_single & (wait_count >= 4))
    discover = (phase == 0) & at_scan & signal_ready
    recipe = jnp.where(discover, discovered_recipe, recipe)
    marker_col = memory[2]
    marker_col = jnp.where(discover, discovered_marker_col, marker_col)
    phase = jnp.where(discover, 1, phase)
    wait_count = jnp.where(
        (phase == 0) & at_scan & left_single & (~signal_ready),
        jnp.minimum(wait_count + 1, 4),
        jnp.where(discover, 0, wait_count),
    )

    production_col = jnp.where(marker_col == 1, 3, 1)
    ingredient_row = jnp.where(recipe == 0, 1, 2)
    held = _self_cell(obs, 7, 11)
    held_any = jnp.sum(held) > 0
    pot_item = _grid_cell(obs, pr, pc, 3, production_col, 34, 38)
    pot_count = pot_item[recipe]
    pot_cooked = _grid_cell(obs, pr, pc, 3, production_col, 33, 34)[0] > 0
    pot_timer = _grid_cell(obs, pr, pc, 3, production_col, 44, 45)[0]

    # Phase 0: rendezvous at the central top-room cell and wait for a marker.
    a0 = _move_or_face(pr, pc, 2, 2, facing, 0)
    a0 = jnp.where((phase == 0) & at_scan & marker_seen, 4, a0)

    # Phase 1: fetch the selected ingredient and add it to the other pot.
    at_ing = (pr == ingredient_row) & (pc == 1)
    at_pot = (pr == 2) & (pc == production_col)
    need_fill = pot_count < 3
    fetch_action = _move_or_face(pr, pc, ingredient_row, 1, facing, 3)
    # The top player stands above the boundary pot, so the pot is DOWN.
    add_action = _move_or_face(pr, pc, 2, production_col, facing, 1)
    a1 = jnp.where(held_any, add_action, fetch_action)
    a1 = jnp.where((~held_any) & at_ing & (facing == 3), 5, a1)
    a1 = jnp.where(held_any & at_pot & (facing == 1), 5, a1)
    phase_after_fill = jnp.where((phase == 1) & (~need_fill), 2, phase)

    # Phase 2: walk to the plate station and wait for cooking to finish.
    at_plate = (pr == 1) & (pc == 3)
    wait_action = _move_or_face(pr, pc, 1, 3, facing, 0)
    ready = pot_cooked & (pot_timer == 0)
    phase_after_wait = jnp.where((phase_after_fill == 2) & at_plate & ready, 3, phase_after_fill)

    # Phase 3: acquire a plate; phase 4: collect the cooked pot.
    plate = _self_cell(obs, 5, 6)[0] > 0
    a3 = jnp.where(at_plate & (facing == 0), 5, wait_action)
    a3 = jnp.where(plate, 4, a3)
    a4 = _move_or_face(pr, pc, 2, production_col, facing, 1)
    at_pot_now = (pr == 2) & (pc == production_col)
    a4 = jnp.where(at_pot_now & plate & (facing == 1), 5, a4)
    phase_after_plate = jnp.where((phase_after_wait == 3) & plate, 4, phase_after_wait)
    cooked = _self_cell(obs, 6, 7)[0] > 0
    phase_after_collect = jnp.where((phase_after_plate == 4) & cooked, 5, phase_after_plate)

    # Phase 5: deliver at the goal.
    at_goal = (pr == 1) & (pc == 2)
    a5 = _move_or_face(pr, pc, 1, 2, facing, 0)
    a5 = jnp.where(at_goal & cooked & (facing == 0), 5, a5)
    phase_next = jnp.where((phase_after_collect == 5) & at_goal & cooked & (facing == 0), 6, phase_after_collect)

    action = jnp.where(phase == 0, a0, jnp.where(phase == 1, a1,
        jnp.where(phase == 2, wait_action, jnp.where(phase == 3, a3,
        jnp.where(phase == 4, a4, jnp.where(phase == 5, a5, 4))))))
    next_phase = jnp.where(phase == 0, phase, jnp.where(phase == 1, phase_after_fill,
        jnp.where(phase == 2, phase_after_wait, jnp.where(phase == 3, phase_after_plate,
        jnp.where(phase == 4, phase_after_collect, phase_next)))))
    next_memory = memory.at[0].set(next_phase).at[1].set(recipe).at[2].set(marker_col).at[3].set(wait_count)
    return action.astype(jnp.int32), next_memory


def _bottom_act(obs, reset, memory):
    pr, pc = _position(obs)
    facing = _facing(obs)
    phase = jnp.where(reset, 0, memory[0])
    recipe = (jnp.sum(obs[:, :, 41]) > 0).astype(jnp.int32)
    source_c = 1
    marker_c = 1
    marker = _grid_cell(obs, pr, pc, 3, marker_c, 34, 38)[3]
    target_amount = recipe + 1
    held3 = _self_cell(obs, 10, 11)[0] > 0
    at_source = (pr == 4) & (pc == source_c)
    at_marker = (pr == 4) & (pc == marker_c)
    source_face = 3
    go_source = _move_or_face(pr, pc, 4, source_c, facing, source_face)
    go_marker = _move_or_face(pr, pc, 4, marker_c, facing, 0)
    marker_done = marker >= target_amount
    action = jnp.where((phase == 1) | marker_done, 4, go_source)
    action = jnp.where((phase == 0) & held3, go_marker, action)
    action = jnp.where((phase == 0) & (~held3) & at_source & (facing == source_face), 5, action)
    action = jnp.where((phase == 0) & held3 & at_marker & (facing == 0), 5, action)
    next_phase = jnp.where((phase == 0) & marker_done, 1, phase)
    next_phase = jnp.where((phase == 0) & held3 & at_marker & (facing == 0) & (marker + 1 >= target_amount), 1, next_phase)
    next_memory = memory.at[0].set(next_phase).at[1].set(recipe)
    return action.astype(jnp.int32), next_memory


def _act(obs, reset, memory):
    pr, _ = _position(obs)
    top_action, top_memory = _top_act(obs, reset, memory)
    bottom_action, bottom_memory = _bottom_act(obs, reset, memory)
    is_top = pr < 3
    return jnp.where(is_top, top_action, bottom_action), jnp.where(is_top, top_memory, bottom_memory)


def init_agent_0():
    return jnp.zeros((8,), dtype=jnp.int32)


def init_agent_1():
    return jnp.zeros((8,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    action, next_memory = _act(obs, reset, memory)
    pr, pc = _position(obs)
    left = _grid_cell(obs, pr, pc, 3, 1, 34, 38)
    right = _grid_cell(obs, pr, pc, 3, 3, 34, 38)
    return action, next_memory, {
        'row': pr.astype(jnp.int32),
        'col': pc.astype(jnp.int32),
        'phase': memory[0].astype(jnp.int32),
        'facing': _facing(obs).astype(jnp.int32),
        'held0': obs[2, 2, 7].astype(jnp.int32),
        'held1': obs[2, 2, 8].astype(jnp.int32),
        'pot0': left[0].astype(jnp.int32),
        'pot1': left[1].astype(jnp.int32),
        'pot3': left[3].astype(jnp.int32),
        'timer': _grid_cell(obs, pr, pc, 3, 1, 44, 45)[0].astype(jnp.int32),
        'memory_recipe': memory[1].astype(jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    action, next_memory = _act(obs, reset, memory)
    pr, pc = _position(obs)
    left = _grid_cell(obs, pr, pc, 3, 1, 34, 38)
    return action, next_memory, {
        'row': pr.astype(jnp.int32),
        'col': pc.astype(jnp.int32),
        'phase': memory[0].astype(jnp.int32),
        'facing': _facing(obs).astype(jnp.int32),
        'held0': obs[2, 2, 7].astype(jnp.int32),
        'held1': obs[2, 2, 8].astype(jnp.int32),
        'pot3': left[3].astype(jnp.int32),
        'memory_recipe': memory[1].astype(jnp.int32),
    }
