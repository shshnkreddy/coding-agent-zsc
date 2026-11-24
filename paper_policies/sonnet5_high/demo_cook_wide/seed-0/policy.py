"""Geometry-aware coordination policy for demo_cook_wide.

Layout facts baked into this policy (fixed, known in advance):
  - agent_0 is permanently confined to row 1, columns {4,5,6}. It can never
    see the recipe indicator (row 4). It CAN always see the pot (row 2,
    col 5), both plate piles (row 0, cols 4/6), the delivery station
    (row 0, col 5), and ingredient supplies 0/1 (row 1, cols 3/7) when close
    enough to the relevant side.
  - agent_1 is confined to row 3 (cols 1-9) plus four row-4 nubs
    (cols 1, 2, 8, 9). It can see the recipe indicator whenever within two
    cells of (4, 5), which includes the pot cell (3, 5) it must visit to add
    ingredients. It can reach ingredient supplies 0 and 1 via (4, 1) (facing
    left / down) or symmetric (4, 9).

Coordination protocol (convention forced by the geometry, not a private
handshake): agent_1 is the only seat that can ever observe the recipe, so it
seeds the pot with one unit of the correct ingredient type. agent_0 never
guesses the recipe; it only ever adds ingredients that already match the
pot's existing (visible) contents, and handles plate fetch, collection and
delivery, which agent_1 cannot reach at all. This should remain sound
against any independently generated partner that discovers the same
geometry-forced division of labor.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

# Action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5


def _find_nearest(plane):
    """Nearest (Manhattan) true cell in a 5x5 int/bool plane, relative to (2,2)."""
    row_idx = jnp.arange(5, dtype=jnp.int32).reshape(5, 1)
    col_idx = jnp.arange(5, dtype=jnp.int32).reshape(1, 5)
    dist = jnp.abs(row_idx - 2) + jnp.abs(col_idx - 2)
    big = jnp.asarray(999, dtype=jnp.int32)
    masked = jnp.where(plane > 0, dist, big)
    flat = jnp.argmin(masked)
    i = (flat // 5).astype(jnp.int32)
    j = (flat % 5).astype(jnp.int32)
    found = masked[i, j] < big
    return found, i - 2, j - 2


# --------------------------------------------------------------------------
# agent_0: bounded to row 1, columns {4,5,6}. Mirrors pot contents; owns
# plate fetch, pot collection and delivery.
# --------------------------------------------------------------------------

def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)

    face_up = obs[2, 2, 1] > 0
    face_down = obs[2, 2, 2] > 0
    face_left = obs[2, 2, 4] > 0

    held_plate = obs[2, 2, 5] > 0
    held_cooked = obs[2, 2, 6] > 0
    held_ing = obs[2, 2, 7:11]
    held_any_ing = jnp.sum(held_ing) > 0
    held_type0 = held_ing[0] > 0
    held_type1 = held_ing[1] > 0

    found_pot, dip, djp = _find_nearest(obs[:, :, 24])
    pi = jnp.clip(2 + dip, 0, 4)
    pj = jnp.clip(2 + djp, 0, 4)
    self_col = 5 - djp  # pot is always visible from this room; global col 5.

    pot_counts = obs[pi, pj, 34:38]
    pot_cooked = obs[pi, pj, 33] > 0
    pot_total = jnp.sum(pot_counts)
    pot_type0 = pot_counts[0] > 0
    pot_type1 = pot_counts[1] > 0

    target_type0 = pot_type0
    target_type1 = jnp.logical_and(jnp.logical_not(pot_type0), pot_type1)
    has_target = jnp.logical_or(target_type0, target_type1)

    at_col5 = self_col == 5

    # Priority 1: holding a plated cooked dish -> deliver at (0,5) from col5.
    deliver_action = jnp.select(
        [self_col < 5, self_col > 5,
         jnp.logical_and(at_col5, jnp.logical_not(face_up)),
         jnp.logical_and(at_col5, face_up)],
        [RIGHT, LEFT, UP, INTERACT], default=STAY)
    deliver_case = jnp.logical_and(held_plate, held_cooked)

    # Priority 2: holding an empty plate -> go collect from the pot.
    collect_action = jnp.select(
        [self_col < 5, self_col > 5,
         jnp.logical_and(at_col5, jnp.logical_not(face_down)),
         jnp.logical_and(jnp.logical_and(at_col5, face_down), pot_cooked)],
        [RIGHT, LEFT, DOWN, INTERACT], default=STAY)
    collect_case = jnp.logical_and(held_plate, jnp.logical_not(held_cooked))

    # Priority 3: holding a raw ingredient -> add to the pot if it matches.
    can_add = jnp.logical_and(
        pot_total < 3,
        jnp.logical_or(pot_total == 0,
                        jnp.logical_or(jnp.logical_and(held_type0, pot_type0),
                                       jnp.logical_and(held_type1, pot_type1))))
    add_action = jnp.select(
        [self_col < 5, self_col > 5,
         jnp.logical_and(at_col5, jnp.logical_not(face_down)),
         jnp.logical_and(jnp.logical_and(at_col5, face_down), can_add)],
        [RIGHT, LEFT, DOWN, INTERACT], default=STAY)
    ing_case = held_any_ing

    empty_case = jnp.logical_not(jnp.logical_or(held_plate, jnp.logical_or(held_cooked, held_any_ing)))

    # Priority 4a: empty-handed and pot is cooked -> fetch a plate (left pile).
    plate_action = jnp.select(
        [self_col > 4,
         jnp.logical_and(self_col == 4, jnp.logical_not(face_up)),
         jnp.logical_and(self_col == 4, face_up)],
        [LEFT, UP, INTERACT], default=STAY)
    # Fetch the plate as soon as the pot reaches capacity (no more mirroring
    # needed), rather than waiting for the cooked flag, so the plate trip
    # overlaps with the cook-time wait instead of following it.
    need_plate_case = jnp.logical_and(empty_case, pot_total >= 3)

    # Priority 4b: empty-handed, pot has a visible partial ingredient below
    # capacity -> fetch a matching unit ourselves (mirror, never guess).
    fetch0_action = jnp.where(jnp.logical_and(self_col == 4, face_left), INTERACT, LEFT)
    fetch1_action = jnp.where(jnp.logical_and(self_col == 6, obs[2, 2, 3] > 0), INTERACT, RIGHT)
    fetch_action = jnp.where(target_type0, fetch0_action, fetch1_action)
    fetch_case = jnp.logical_and(empty_case, jnp.logical_and(
        jnp.logical_not(pot_cooked), jnp.logical_and(has_target, pot_total < 3)))

    action = jnp.select(
        [deliver_case, collect_case, ing_case, need_plate_case, fetch_case],
        [deliver_action, collect_action, add_action, plate_action, fetch_action],
        default=STAY)

    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    trace = {
        'self_col': self_col.astype(jnp.int32),
        'pot_total': pot_total.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace


# --------------------------------------------------------------------------
# agent_1: bounded to row 3 (cols 1-9) plus row-4 nubs at cols {1,2,8,9}.
# Reads the recipe near the pot, seeds one matching ingredient, returns.
# --------------------------------------------------------------------------

def init_agent_1():
    return {
        'phase': jnp.asarray(0, dtype=jnp.int32),        # 0 monitor, 1 to-supply, 2 to-pot
        'target_type': jnp.asarray(-1, dtype=jnp.int32),
        'recipe_type': jnp.asarray(-1, dtype=jnp.int32),
    }


def agent_1_act(obs, reset, memory, rng):
    obs = obs.astype(jnp.int32)

    zero_i = jnp.asarray(0, dtype=jnp.int32)
    neg1 = jnp.asarray(-1, dtype=jnp.int32)
    phase = jnp.where(reset, zero_i, memory['phase'])
    target_type = jnp.where(reset, neg1, memory['target_type'])
    recipe_type = jnp.where(reset, neg1, memory['recipe_type'])

    face_up = obs[2, 2, 1] > 0
    face_down = obs[2, 2, 2] > 0
    face_left = obs[2, 2, 4] > 0
    held_ing = obs[2, 2, 7:11]
    held_any_ing = jnp.sum(held_ing) > 0

    # --- Self localization via unique / sign-disambiguated landmarks. ---
    found_pot, dip, djp = _find_nearest(obs[:, :, 24])
    row_pot, col_pot = 2 - dip, 5 - djp

    found_rec, dir_, djr = _find_nearest(obs[:, :, 25])
    row_rec, col_rec = 4 - dir_, 5 - djr

    found_i3, di3, dj3 = _find_nearest(obs[:, :, 31])
    row_i3 = 4 - di3
    col_i3 = jnp.where(dj3 > 0, 4 - dj3, 6 - dj3)

    found_i0, di0, dj0 = _find_nearest(obs[:, :, 28])
    row_i0 = 4 - di0
    col_i0 = jnp.where(dj0 < 0, -dj0, 10 - dj0)

    self_row = jnp.select([found_pot, found_rec, found_i3, found_i0],
                           [row_pot, row_rec, row_i3, row_i0], default=3)
    self_col = jnp.select([found_pot, found_rec, found_i3, found_i0],
                           [col_pot, col_rec, col_i3, col_i0], default=5)

    # --- Recipe reading (only trustworthy when the indicator is in view). ---
    ri = jnp.clip(2 + dir_, 0, 4)
    rj = jnp.clip(2 + djr, 0, 4)
    recipe_now = jnp.where(
        found_rec,
        jnp.where(obs[ri, rj, 40] > 0, 0, jnp.where(obs[ri, rj, 41] > 0, 1, -1)),
        -1)
    new_recipe_type = jnp.where(found_rec, recipe_now, recipe_type)

    # --- Pot reading. ---
    pi = jnp.clip(2 + dip, 0, 4)
    pj = jnp.clip(2 + djp, 0, 4)
    pot_counts = jnp.where(found_pot, obs[pi, pj, 34:38], jnp.zeros((4,), dtype=jnp.int32))
    pot_total = jnp.sum(pot_counts)
    pot_type0 = pot_counts[0] > 0
    pot_type1 = pot_counts[1] > 0

    # Partner (agent_0) is always visible from the pot station (row offset 2,
    # col offset 0, within the view radius). Use this to avoid a one-step
    # race: right after agent_0 *collects* a cooked dish the pot looks empty,
    # but the recipe only resamples on the later *delivery* interact. Do not
    # seed a fresh batch while the partner is still carrying a plated dish,
    # since the recipe is about to change out from under us.
    found_partner, dipn, djpn = _find_nearest(obs[:, :, 11])
    pni = jnp.clip(2 + dipn, 0, 4)
    pnj = jnp.clip(2 + djpn, 0, 4)
    partner_has_dish = jnp.logical_and(found_partner,
                                        jnp.logical_and(obs[pni, pnj, 16] > 0, obs[pni, pnj, 17] > 0))

    at_pot_station = jnp.logical_and(self_row == 3, self_col == 5)
    at_supply_station = jnp.logical_and(self_row == 4, self_col == 1)

    # Phase 0 -> 1: seed decision (pot empty, recipe known, at pot station,
    # and the partner isn't mid-delivery of the previous dish).
    cond_seed = jnp.logical_and(phase == 0, jnp.logical_and(
        at_pot_station, jnp.logical_and(pot_total == 0, jnp.logical_and(
            new_recipe_type >= 0, jnp.logical_not(partner_has_dish)))))
    new_phase = jnp.where(cond_seed, 1, phase)
    new_target_type = jnp.where(cond_seed, new_recipe_type, target_type)

    # Phase 1 -> 2: pickup ready (at supply station, correct facing).
    pickup_ready0 = jnp.logical_and(new_phase == 1, jnp.logical_and(
        new_target_type == 0, jnp.logical_and(at_supply_station, face_left)))
    pickup_ready1 = jnp.logical_and(new_phase == 1, jnp.logical_and(
        new_target_type == 1, jnp.logical_and(at_supply_station, face_down)))
    pickup_ready = jnp.logical_or(pickup_ready0, pickup_ready1)
    new_phase = jnp.where(pickup_ready, 2, new_phase)

    # Phase 2 -> 0: add ready (at pot station, facing up, pot has room / matches).
    can_add = jnp.logical_and(
        pot_total < 3,
        jnp.logical_or(pot_total == 0,
                        jnp.logical_or(jnp.logical_and(new_target_type == 0, pot_type0),
                                       jnp.logical_and(new_target_type == 1, pot_type1))))
    add_ready = jnp.logical_and(new_phase == 2, jnp.logical_and(
        at_pot_station, jnp.logical_and(face_up, can_add)))
    new_phase_final = jnp.where(add_ready, 0, new_phase)
    new_target_final = jnp.where(add_ready, neg1, new_target_type)

    # --- Navigation primitives. ---
    nav_to_pot = jnp.where(
        self_row == 4, UP,
        jnp.where(self_col < 5, RIGHT, jnp.where(self_col > 5, LEFT, STAY)))

    travel_to_supply = jnp.where(
        self_row == 4,
        jnp.where(self_col == 1, STAY, UP),
        jnp.where(self_col > 1, LEFT, DOWN))
    turn_at_supply = jnp.where(new_target_type == 0, LEFT, DOWN)
    phase1_nav = jnp.where(at_supply_station, turn_at_supply, travel_to_supply)

    at_pot_wrong_facing = jnp.logical_and(at_pot_station, jnp.logical_not(face_up))
    phase2_nav = jnp.where(at_pot_wrong_facing, UP, nav_to_pot)

    action = jnp.select(
        [pickup_ready, add_ready, new_phase_final == 1, new_phase_final == 2],
        [INTERACT, INTERACT, phase1_nav, phase2_nav],
        default=nav_to_pot)  # phase 0: monitor / travel to pot station.

    next_memory = {
        'phase': new_phase_final.astype(jnp.int32),
        'target_type': new_target_final.astype(jnp.int32),
        'recipe_type': new_recipe_type.astype(jnp.int32),
    }
    trace = {
        'self_row': self_row.astype(jnp.int32),
        'self_col': self_col.astype(jnp.int32),
        'phase': new_phase_final.astype(jnp.int32),
        'recipe_type': new_recipe_type.astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
