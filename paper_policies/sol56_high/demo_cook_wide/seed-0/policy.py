"""Map-aware, convention-friendly policy for demo_cook_wide."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'room_roles_v1', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STAY = 4
INTERACT = 5

# Immutable fixed-map feature tensor corresponding to observation channels
# 22:32. It lets each seat localize from its unrotated local crop.
_WALLS = (
    (0,0),(0,1),(0,2),(0,3),(0,7),(0,8),(0,9),(0,10),
    (1,0),(1,1),(1,2),(1,8),(1,9),(1,10),
    (2,0),(2,1),(2,2),(2,3),(2,4),(2,6),(2,7),(2,8),(2,9),(2,10),
    (3,0),(3,10),(4,3),(4,7),
    (5,0),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,10),
)
_UPPER_POS = ((1,4),(1,5),(1,6))
_LOWER_POS = (
    ((3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),
     (4,1),(4,2),(4,8),(4,9)))


def _locate(obs, candidates):
    candidates = jnp.asarray(candidates, dtype=jnp.int32)
    off = jnp.arange(-2, 3, dtype=jnp.int32)
    static = jnp.zeros((6, 11, 10), dtype=jnp.int32)
    walls = jnp.asarray(_WALLS, dtype=jnp.int32)
    static = static.at[walls[:, 0], walls[:, 1], 0].set(1)
    static = static.at[0, 5, 1].set(1)
    static = static.at[2, 5, 2].set(1)
    static = static.at[4, 5, 3].set(1)
    static = static.at[jnp.asarray((0,0)), jnp.asarray((4,6)), 5].set(1)
    static = static.at[jnp.asarray((1,4,4)), jnp.asarray((3,0,10)), 6].set(1)
    static = static.at[jnp.asarray((1,5,5)), jnp.asarray((7,1,9)), 7].set(1)
    static = static.at[jnp.asarray((4,4)), jnp.asarray((4,6)), 9].set(1)
    rr = candidates[:, 0, None, None] + off[None, :, None]
    cc = candidates[:, 1, None, None] + off[None, None, :]
    valid = (rr >= 0) & (rr < 6) & (cc >= 0) & (cc < 11)
    crops = static[jnp.clip(rr, 0, 5), jnp.clip(cc, 0, 10), :]
    crops = crops * valid[..., None]
    error = jnp.sum(jnp.abs(crops - obs[None, :, :, 22:32]), axis=(1,2,3))
    return candidates[jnp.argmin(error)]


def _facing(obs):
    return jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)


def _face_or_interact(facing, wanted_direction, turn_action):
    return jnp.where(facing == wanted_direction, INTERACT, turn_action)


def init_agent_0():
    return jnp.zeros((1,), dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    """Upper seat assists from visible pot state, then plates and delivers."""
    pos = _locate(obs, _UPPER_POS)
    col = pos[1]
    facing = _facing(obs)
    plate = obs[2, 2, 5] > 0
    cooked = obs[2, 2, 6] > 0
    raw = jnp.sum(obs[2, 2, 7:11]) > 0

    pot_col = jnp.int32(2) + (jnp.int32(5) - col)
    pot_counts = obs[3, pot_col, 34:38]
    pot_total = jnp.sum(pot_counts)
    pot_ready = obs[3, pot_col, 33] > 0
    pot_busy = (pot_total >= 3) | pot_ready | (obs[3, pot_col, 44] > 0)

    deliver_action = jnp.where(
        col < 5, RIGHT,
        jnp.where(col > 5, LEFT, _face_or_interact(facing, jnp.int32(0), UP)))
    pot_action = jnp.where(
        col < 5, RIGHT,
        jnp.where(col > 5, LEFT, _face_or_interact(facing, jnp.int32(1), DOWN)))
    plate_action = jnp.where(
        col > 4, LEFT, _face_or_interact(facing, jnp.int32(0), UP))

    # Once the lower cook has publicly established the recipe by placing one
    # ingredient, fetch exactly one matching unit from the close upper supply.
    # At total two, wait for the lower cook's already-in-flight next unit.
    partner_counts = jnp.sum(obs[:, :, 18:22], axis=(0, 1))
    signal_counts = pot_counts + partner_counts
    signal_total = jnp.sum(signal_counts)
    partner_inventory_total = jnp.sum(obs[:, :, 16:22])
    partner_at_interface = obs[4, pot_col, 11] > 0
    partner_yielding = partner_at_interface & (partner_inventory_total == 0)
    inferred_kind = jnp.argmax(signal_counts).astype(jnp.int32)
    supply_col = jnp.where(inferred_kind == 1, jnp.int32(6), jnp.int32(4))
    supply_dir = jnp.where(inferred_kind == 1, jnp.int32(2), jnp.int32(3))
    supply_turn = jnp.where(inferred_kind == 1, RIGHT, LEFT)
    fetch_action = jnp.where(
        col < supply_col, RIGHT,
        jnp.where(col > supply_col, LEFT,
                  _face_or_interact(facing, supply_dir, supply_turn)))
    wait_center = jnp.where(col < 5, RIGHT,
                  jnp.where(col > 5, LEFT, STAY))
    empty_action = jnp.where(
        pot_busy, plate_action,
        jnp.where((signal_total > 0) &
                  ((pot_total < 2) | ((pot_total == 2) & partner_yielding)),
                  fetch_action, wait_center))

    action = jnp.where(cooked, deliver_action,
             jnp.where(plate | raw, pot_action, empty_action))
    old = jnp.where(reset, jnp.zeros_like(memory), memory)
    nxt = old.at[0].set(old[0] + jnp.int32(1))
    trace = {"phase": jnp.where(cooked, 2, jnp.where(plate, 1, 0)).astype(jnp.int32)}
    return action.astype(jnp.int32), nxt, trace


def _to_center(row, col):
    return jnp.where(row == 4, UP,
        jnp.where(col < 5, RIGHT, jnp.where(col > 5, LEFT, STAY)))


def _to_left_supply(row, col):
    from_row3 = jnp.where(col > 1, LEFT, DOWN)
    from_row4 = jnp.where(col == 1, STAY,
                 jnp.where(col == 2, LEFT, UP))
    return jnp.where(row == 3, from_row3, from_row4)


def _to_left_supply_zero(row, col):
    # Enter (4,1) from (4,2), so the last move already faces ingredient 0.
    from_row3 = jnp.where(col > 2, LEFT,
                jnp.where(col < 2, RIGHT, DOWN))
    from_row4 = jnp.where(col == 1, STAY,
                 jnp.where(col == 2, LEFT, UP))
    return jnp.where(row == 3, from_row3, from_row4)


def _to_right_supply(row, col):
    from_row3 = jnp.where(col < 9, RIGHT, DOWN)
    from_row4 = jnp.where(col == 9, STAY,
                 jnp.where(col == 8, RIGHT, UP))
    return jnp.where(row == 3, from_row3, from_row4)


def _to_right_supply_zero(row, col):
    # Enter (4,9) from (4,8), already facing ingredient 0 at (4,10).
    from_row3 = jnp.where(col < 8, RIGHT, DOWN)
    from_row4 = jnp.where(col == 9, STAY,
                 jnp.where(col == 8, RIGHT, UP))
    return jnp.where(row == 3, from_row3, from_row4)


# The short wait at each new partial count is an adaptive
# public-state protocol: helpers get first opportunity, but non-helping partners
# time out to the ordinary self-sufficient ingredient loop.
def init_agent_1():
    # [remembered recipe, elapsed steps, last observed pot count, wait age]
    return jnp.asarray((-1, 0, -1, 0), dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    base = jnp.where(reset, init_agent_1(), memory)
    pos = _locate(obs, _LOWER_POS)
    row, col = pos[0], pos[1]
    facing = _facing(obs)

    recipe0 = jnp.sum(obs[:, :, 40])
    recipe1 = jnp.sum(obs[:, :, 41])
    recipe_seen = (recipe0 + recipe1) > 0
    observed_kind = jnp.where(recipe1 > recipe0, jnp.int32(1), jnp.int32(0))
    desired = jnp.where(recipe_seen, observed_kind, base[0])

    holding_raw = jnp.sum(obs[2, 2, 7:11]) > 0
    at_center = (row == 3) & (col == 5)
    at_supply = (row == 4) & (col == 1)
    pot_total = jnp.sum(obs[1, 2, 34:38])
    pot_cooked = obs[1, 2, 33] > 0
    pot_busy = (pot_total >= 3) | pot_cooked | (obs[1, 2, 44] > 0)
    pot_partial = (pot_total > 0) & (pot_total < 3) & (~pot_busy)

    same_count = base[2] == pot_total
    wait_age = jnp.where(
        at_center & pot_partial,
        jnp.where(same_count, base[3] + jnp.int32(1), jnp.int32(0)),
        jnp.int32(0))

    add_action = jnp.where(
        at_center, _face_or_interact(facing, jnp.int32(0), UP),
        _to_center(row, col))
    supply_direction = jnp.where(desired == 1, jnp.int32(1), jnp.int32(3))
    supply_turn = jnp.where(desired == 1, DOWN, LEFT)
    left_fetch_action = jnp.where(
        at_supply, _face_or_interact(facing, supply_direction, supply_turn),
        jnp.where(desired == 0, _to_left_supply_zero(row, col),
                  _to_left_supply(row, col)))
    at_right_supply = (row == 4) & (col == 9)
    right_direction = jnp.where(desired == 1, jnp.int32(1), jnp.int32(2))
    right_turn = jnp.where(desired == 1, DOWN, RIGHT)
    right_fetch_action = jnp.where(
        at_right_supply, _face_or_interact(facing, right_direction, right_turn),
        jnp.where(desired == 0, _to_right_supply_zero(row, col),
                  _to_right_supply(row, col)))
    fetch_action = jnp.where(col > 5, right_fetch_action, left_fetch_action)

    partner_serving = (jnp.sum(obs[:, :, 16]) > 0) | (jnp.sum(obs[:, :, 17]) > 0)
    partner_raw = jnp.sum(obs[:, :, 18:22]) > 0
    partner_visible = jnp.sum(obs[:, :, 11]) > 0
    partner_at_interface = obs[0, 2, 11] > 0
    helper_evidence = partner_raw | (partner_visible & (~partner_at_interface))
    yield_to_helper = (at_center & pot_partial & (wait_age < 6) &
                       (~partner_serving) &
                       ((wait_age == 0) | helper_evidence))
    empty_action = jnp.where(
        desired < 0, _to_center(row, col),
        jnp.where((at_center & pot_busy) | yield_to_helper, STAY,
                  fetch_action))
    action = jnp.where(holding_raw, add_action, empty_action)

    next_last = jnp.where(at_center, pot_total, base[2])
    nxt = jnp.asarray((desired, base[1] + jnp.int32(1), next_last, wait_age),
                      dtype=jnp.int32)
    trace = {"desired": desired.astype(jnp.int32),
             "pot_total": pot_total.astype(jnp.int32),
             "wait": wait_age.astype(jnp.int32)}
    return action.astype(jnp.int32), nxt, trace
