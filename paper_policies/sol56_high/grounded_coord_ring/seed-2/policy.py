"""Convention-robust policy for the grounded coordination ring.

The outer player advertises the recipe by placing matching ingredients on
ordinary interface counters. The inner player can use those ingredients, but
does not depend on that convention: it reads the recipe button and can collect
all ingredients, plates, and deliver by itself.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "grounded_public_handoff",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": 9,
    "max_width": 9,
    "num_ingredients": 3,
}


# Movement action IDs (the facing IDs in observations use a different order).
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5


# The two walkable components are cycles. Their order is clockwise.
_OUTER = (
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
        (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
        (6, 1), (5, 1), (4, 1), (3, 1), (2, 1),
)
_INNER = (
    (3, 3), (3, 4), (3, 5), (4, 5),
    (5, 5), (5, 4), (5, 3), (4, 3),
)


# Static observation template, used to recover global position from the local
# crop. Every legal position has a unique 5x5 signature on this layout.
_WALLS = (
    (0, 0), (0, 1), (0, 2), (0, 6), (0, 7), (0, 8),
     (1, 0), (1, 8), (2, 0), (2, 2), (2, 3), (2, 5), (2, 6), (2, 8),
     (6, 0), (6, 2), (6, 3), (6, 5), (6, 6), (6, 8),
     (7, 0), (7, 8), (8, 0), (8, 1), (8, 2), (8, 6), (8, 7), (8, 8),
)
_GOALS = ((4, 2), (4, 6))
_POTS = ((4, 4),)
_RECIPE = ((0, 4), (4, 0), (4, 8), (8, 4))
_BUTTON = ((2, 4), (6, 4))
_PLATES = ((3, 6), (5, 6))
_ING0 = ((3, 2),)
_ING1 = ((5, 2),)
_ING2 = (
    (0, 3), (0, 5), (3, 0), (3, 8),
    (5, 0), (5, 8), (8, 3), (8, 5),
)


def _set_static(grid, positions, channel):
    points = jnp.asarray(positions, dtype=jnp.int32)
    return grid.at[points[:, 0], points[:, 1], channel].set(1)


def _coords_and_signatures(coords_tuple):
    static = jnp.zeros((9, 9, 9), dtype=jnp.int32)
    static = _set_static(static, _WALLS, 0)
    static = _set_static(static, _GOALS, 1)
    static = _set_static(static, _POTS, 2)
    static = _set_static(static, _RECIPE, 3)
    static = _set_static(static, _BUTTON, 4)
    static = _set_static(static, _PLATES, 5)
    static = _set_static(static, _ING0, 6)
    static = _set_static(static, _ING1, 7)
    static = _set_static(static, _ING2, 8)
    padded = jnp.pad(static, ((2, 2), (2, 2), (0, 0)))
    signatures = jnp.stack(
        [padded[r:r + 5, c:c + 5] for r, c in coords_tuple]
    )
    return jnp.asarray(coords_tuple, dtype=jnp.int32), signatures


def _locate(obs, signatures):
    mismatch = jnp.sum(jnp.abs(signatures - obs[None, :, :, 20:29]), axis=(1, 2, 3))
    return jnp.argmin(mismatch).astype(jnp.int32)


def _move_for_delta(dr, dc):
    action = jnp.where(dr < 0, UP, DOWN)
    action = jnp.where(dc < 0, LEFT, action)
    action = jnp.where(dc > 0, RIGHT, action)
    return action.astype(jnp.int32)


def _facing_id(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _navigate(index, target, coords, desired_facing, arrived_action, obs):
    """Shortest-path action on a cycle, then turn toward the serviced object."""
    n = jnp.int32(coords.shape[0])
    forward = jnp.mod(target - index, n)
    backward = jnp.mod(index - target, n)
    next_index = jnp.where(forward <= backward, jnp.mod(index + 1, n), jnp.mod(index - 1, n))
    delta = coords[next_index] - coords[index]
    move = _move_for_delta(delta[0], delta[1])
    face_action = jnp.where(desired_facing == 0, UP,
                  jnp.where(desired_facing == 1, DOWN,
                  jnp.where(desired_facing == 2, RIGHT, LEFT)))
    at_target = index == target
    return jnp.where(
        at_target,
        jnp.where(_facing_id(obs) == desired_facing, arrived_action, face_action),
        move,
    ).astype(jnp.int32)


def _displayed_recipe(obs, old_recipe):
    shown = jnp.sum(obs[:, :, 36:39], axis=(0, 1))
    visible = jnp.sum(shown) > 0
    recipe = jnp.argmax(shown).astype(jnp.int32)
    return jnp.where(visible, recipe, old_recipe), visible


def init_agent_0():
    # [steps, recipe, offer_active, saw_cooked_partner, wait,
    #  deposit_pending, saw_plate_or_cooked_progress]
    return jnp.asarray([0, -1, 0, 0, 0, 0, 0], dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, init_agent_0(), memory)
    recipe, recipe_visible = _displayed_recipe(obs, base[1])
    outer, signatures = _coords_and_signatures(_OUTER)
    index = _locate(obs, signatures)

    held_counts = obs[2, 2, 7:10]
    holding_raw = jnp.sum(held_counts) > 0
    held_id = jnp.argmax(held_counts).astype(jnp.int32)

    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_plate = jnp.sum(obs[:, :, 15]) > 0
    partner_cooked = jnp.sum(obs[:, :, 16]) > 0
    matching_channel = jnp.int32(31) + jnp.maximum(recipe, 0)
    matching_counter_occupied = jnp.sum(obs[:, :, 20] * obs[:, :, matching_channel]) > 0
    deposited = (base[5] > 0) & (~holding_raw)
    armed = (base[3] > 0) | partner_cooked
    recipe_changed = (base[2] > 0) & recipe_visible & (base[1] >= 0) & (recipe != base[1])
    completed = (armed & partner_visible & (~partner_cooked)) | recipe_changed
    offered = jnp.where(completed, jnp.int32(0), base[2] + deposited.astype(jnp.int32))
    wait = jnp.where(offered > 0, base[4] + 1, jnp.int32(0))
    progressed = jnp.where(
        completed, False,
        (base[6] > 0) | ((offered > 0) & (partner_plate | partner_cooked)),
    )
    early_refill = (offered > 0) & (wait > 6) & (~progressed) & (~matching_counter_occupied)
    # A timeout keeps the outer role useful with partners that deliver outside
    # its view or store a plated dish instead of delivering immediately.
    timed_out = wait > 40
    rearm = timed_out | early_refill
    offered = jnp.where(rearm, jnp.int32(0), offered)
    armed = jnp.where(completed | rearm, False, armed)
    wait = jnp.where(rearm, jnp.int32(0), wait)
    progressed = jnp.where(rearm, False, progressed)

    # Public convention: ingredient 0 is offered through the upper-left
    # interface counter and ingredient 1 through the lower-left one.
    offer_target = jnp.where(held_id == 0, jnp.int32(2), jnp.int32(16))
    offer_face = jnp.where(held_id == 0, jnp.int32(1), jnp.int32(0))
    offer_action = _navigate(index, offer_target, outer, offer_face, INTERACT, obs)

    # With empty hands, read an always-on display if necessary, otherwise get
    # exactly the requested ingredient from the left-side supplies.
    pickup_target = jnp.where(recipe == 0, jnp.int32(22), jnp.int32(20))
    pickup_action = _navigate(index, pickup_target, outer, jnp.int32(2), INTERACT, obs)
    seek_recipe = _navigate(index, jnp.int32(21), outer, jnp.int32(3), STAY, obs)
    empty_action = jnp.where(recipe >= 0, pickup_action, seek_recipe)
    wait_action = _navigate(index, jnp.int32(21), outer, jnp.int32(3), STAY, obs)
    empty_action = jnp.where(offered > 0, wait_action, empty_action)
    # Never advertise an ingredient acquired for an old recipe. With no trash
    # action available, quarantine it in hand at an always-on display until it
    # becomes relevant again; the inner fallback remains fully autonomous.
    stale_held = (recipe >= 0) & (held_id != recipe)
    holding_action = jnp.where(stale_held, seek_recipe, offer_action)
    action = jnp.where(holding_raw, holding_action, empty_action).astype(jnp.int32)

    deposit_pending = holding_raw & (action == INTERACT)
    next_memory = jnp.asarray(
        [base[0] + 1, recipe, offered, armed.astype(jnp.int32), wait,
         deposit_pending.astype(jnp.int32), progressed.astype(jnp.int32)],
        dtype=jnp.int32,
    )
    trace = {
        "position_index": index,
        "recipe": recipe,
        "recipe_visible": recipe_visible,
        "holding_raw": holding_raw,
        "offers_this_dish": offered,
        "delivery_armed": armed,
        "progressed": progressed,
        "early_refill": early_refill,
    }
    return action, next_memory, trace


def init_agent_1():
    # [steps, recipe, delivery_pending, unknown_age,
    #  observed_empty_handoff, observed_visible_empty_partner]
    return jnp.asarray([0, -1, 0, 0, 0, 0], dtype=jnp.int32)


def _global_cell(obs, position, row, col, channel):
    i = jnp.int32(row) - position[0] + 2
    j = jnp.int32(col) - position[1] + 2
    visible = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    ii = jnp.clip(i, 0, 4)
    jj = jnp.clip(j, 0, 4)
    return jnp.where(visible, obs[ii, jj, channel], jnp.int32(0))


def agent_1_act(obs, reset, memory, rng):
    del rng
    base = jnp.where(reset, init_agent_1(), memory)
    inner, signatures = _coords_and_signatures(_INNER)
    index = _locate(obs, signatures)
    position = inner[index]

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_counts = obs[2, 2, 7:10]
    holding_raw = jnp.sum(held_counts) > 0
    empty = ~(held_plate | holding_raw)

    # A delivery consumes the held dish whether correct or not, so the next
    # dish always begins by grounding the new target again.
    delivered = (base[2] > 0) & empty
    old_recipe = jnp.where(delivered, jnp.int32(-1), base[1])

    # Ingredient semantics are sufficient recipe evidence here because both
    # possible recipes are homogeneous. Require an empty observation first so
    # an old item left over from the previous dish is never reinterpreted.
    counter_raw0 = jnp.sum(obs[:, :, 20] * obs[:, :, 31])
    counter_raw1 = jnp.sum(obs[:, :, 20] * obs[:, :, 32])
    counter_present = (counter_raw0 + counter_raw1) > 0
    partner_visible = jnp.sum(obs[:, :, 10]) > 0
    partner_raw0 = jnp.sum(obs[:, :, 17])
    partner_raw1 = jnp.sum(obs[:, :, 18])
    partner_present = (partner_raw0 + partner_raw1) > 0
    new_unknown = delivered | reset
    saw_empty = jnp.where(new_unknown, ~counter_present,
                          (base[4] > 0) | (~counter_present))
    visible_partner_empty = partner_visible & (~partner_present)
    saw_partner_empty = jnp.where(
        new_unknown, visible_partner_empty,
        (base[5] > 0) | visible_partner_empty,
    )
    fresh_counter = (old_recipe < 0) & (base[4] > 0) & counter_present
    fresh_partner = (old_recipe < 0) & (base[5] > 0) & partner_present
    fresh_offer = fresh_counter | fresh_partner
    counter_recipe = jnp.argmax(jnp.asarray([counter_raw0, counter_raw1])).astype(jnp.int32)
    partner_recipe = jnp.argmax(jnp.asarray([partner_raw0, partner_raw1])).astype(jnp.int32)
    offered_recipe = jnp.where(fresh_partner, partner_recipe, counter_recipe)
    evidenced_recipe = jnp.where(fresh_offer, offered_recipe, old_recipe)
    recipe, recipe_visible = _displayed_recipe(obs, evidenced_recipe)
    unknown_age = jnp.where(recipe >= 0, jnp.int32(0),
                  jnp.where(new_unknown, jnp.int32(0), base[3] + 1))

    pot_counts = jnp.sum(obs[:, :, 31:34] * obs[:, :, 22:23], axis=(0, 1))
    pot_total = jnp.sum(pot_counts)
    pot_cooked = jnp.sum(obs[:, :, 30] * obs[:, :, 22]) > 0
    pot_timer = jnp.sum(obs[:, :, 39] * obs[:, :, 22])

    # Scan all four shared counters for a raw item matching the known recipe.
    # This accepts common partner handoffs without requiring a particular side.
    item_channel = jnp.int32(31) + jnp.maximum(recipe, 0)
    c0 = _global_cell(obs, position, 2, 3, item_channel)
    c1 = _global_cell(obs, position, 2, 5, item_channel)
    c2 = _global_cell(obs, position, 6, 3, item_channel)
    c3 = _global_cell(obs, position, 6, 5, item_channel)
    counter_values = jnp.asarray([c0, c1, c2, c3])
    has_offer = (recipe >= 0) & (jnp.max(counter_values) > 0)
    offer_choice = jnp.argmax(counter_values).astype(jnp.int32)
    offer_targets = jnp.asarray([0, 2, 6, 4], dtype=jnp.int32)
    offer_faces = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    offer_target = offer_targets[offer_choice]
    offer_face = offer_faces[offer_choice]

    # Navigation primitives for each possible immediate job.
    to_button = _navigate(index, jnp.int32(1), inner, jnp.int32(0), INTERACT, obs)
    supply_target = jnp.where(recipe == 0, jnp.int32(0), jnp.int32(6))
    to_supply = _navigate(index, supply_target, inner, jnp.int32(3), INTERACT, obs)
    to_offer = _navigate(index, offer_target, inner, offer_face, INTERACT, obs)
    to_pot_interact = _navigate(index, jnp.int32(7), inner, jnp.int32(2), INTERACT, obs)
    to_pot_wait = _navigate(index, jnp.int32(7), inner, jnp.int32(2), STAY, obs)
    to_plate = _navigate(index, jnp.int32(2), inner, jnp.int32(2), INTERACT, obs)
    to_goal = _navigate(index, jnp.int32(7), inner, jnp.int32(3), INTERACT, obs)

    need_recipe = recipe < 0
    need_ingredient = (pot_total < 3) & (~pot_cooked) & (pot_timer == 0)
    empty_known_action = jnp.where(
        need_ingredient,
        jnp.where(has_offer, to_offer, to_supply),
        to_plate,
    )
    # Wait at the left pot/goal service square, which sees both recipe-relevant
    # handoff counters. If no partner evidence arrives, use the paid button and
    # continue autonomously rather than depending on a convention.
    wait_for_offer = _navigate(index, jnp.int32(7), inner, jnp.int32(2), STAY, obs)
    evidence_limit = jnp.where(base[0] < 40, jnp.int32(22), jnp.int32(9))
    grounded_start = jnp.where(unknown_age < evidence_limit, wait_for_offer, to_button)
    # Avoid all new work when there is too little horizon even for a fast cycle.
    start_cycle = jnp.where(base[0] < 358, grounded_start, jnp.int32(STAY))
    empty_action = jnp.where(need_recipe, start_cycle, empty_known_action)

    # Empty plate: approach the pot immediately, overlap travel with cooking,
    # and wait at the service square until the dish is actually scoopable.
    plate_action = jnp.where(pot_cooked, to_pot_interact, to_pot_wait)
    action = jnp.where(
        held_cooked,
        to_goal,
        jnp.where(held_plate, plate_action,
        jnp.where(holding_raw, to_pot_interact, empty_action)),
    ).astype(jnp.int32)

    attempting_delivery = held_cooked & (action == INTERACT)
    next_memory = jnp.asarray(
        [base[0] + 1, recipe, attempting_delivery.astype(jnp.int32),
         unknown_age, jnp.where(recipe >= 0, False, saw_empty).astype(jnp.int32),
         jnp.where(recipe >= 0, False, saw_partner_empty).astype(jnp.int32)],
        dtype=jnp.int32,
    )
    trace = {
        "position_index": index,
        "recipe": recipe,
        "recipe_visible": recipe_visible,
        "pot_total": pot_total,
        "pot_timer": pot_timer,
        "pot_cooked": pot_cooked,
        "has_offer": has_offer,
        "fresh_offer": fresh_offer,
        "fresh_partner": fresh_partner,
        "unknown_age": unknown_age,
    }
    return action, next_memory, trace
