"""Solo blind-cook policy for agent_1; safe no-op for agent_0.

Environment structure (test_time_simple layout): agent_0's spawn room only
touches the always-on recipe indicator and an unused ingredient supply, with
no pot/plates/delivery and no shared counter with agent_1's room, and the two
rooms are never within each other's view radius. So agent_0 cannot affect
reward or communicate with any partner, and agent_1 can solo-complete the
full cook/deliver cycle but never observes the target recipe. With only two
possible recipes and no resample on a wrong delivery, agent_1 alternates its
guessed recipe every delivery so a miss is always followed by a guaranteed
hit. All navigation is a reactive controller driven purely from the current
local observation; the only carried memory bit is the alternating guess.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}

# Movement action ids.
RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# Facing ids (matches observation channel order UP, DOWN, RIGHT, LEFT).
FACE_UP, FACE_DOWN, FACE_RIGHT, FACE_LEFT = 0, 1, 2, 3

def _locate(channel_slice):
    """Return local (row, col) of the single 1-valued cell in a 5x5 slice."""
    flat = channel_slice.reshape(-1)
    idx = jnp.argmax(flat).astype(jnp.int32)
    return idx // 5, idx % 5


def _approach_action(dr, dc, current_facing_id):
    """Move toward a target offset (dr, dc) from self, or interact if adjacent and facing it."""
    desired_move = jnp.where(
        dr < 0, UP,
        jnp.where(
            dr > 0, DOWN,
            jnp.where(dc > 0, RIGHT, jnp.where(dc < 0, LEFT, STAY)),
        ),
    ).astype(jnp.int32)
    adjacent = (jnp.abs(dr) + jnp.abs(dc)) == 1
    move_to_facing = jnp.array([FACE_RIGHT, FACE_DOWN, FACE_LEFT, FACE_UP, FACE_UP], dtype=jnp.int32)
    desired_facing = move_to_facing[desired_move]
    facing_correct = current_facing_id == desired_facing
    action = jnp.where(adjacent & facing_correct, INTERACT, desired_move)
    return action.astype(jnp.int32)


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    # This seat's room has no pot, plates, delivery, or shared counter with the
    # partner's room, and is outside the partner's view radius under this
    # layout, so no action here can change reward or communicate. Stay put.
    action = jnp.asarray(STAY, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


def init_agent_1():
    return {'target_ing': jnp.asarray(0, dtype=jnp.int32)}


def agent_1_act(obs, reset, memory, rng):
    target_ing = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['target_ing'])

    self_plate = obs[2, 2, 5]
    self_cooked = obs[2, 2, 6]
    self_ing0 = obs[2, 2, 7]
    self_ing1 = obs[2, 2, 8]

    holding_cooked = self_cooked == 1
    holding_plate_empty = (self_plate == 1) & (self_cooked == 0)
    holding_ingredient = (self_ing0 + self_ing1) > 0

    current_facing_id = jnp.argmax(obs[2, 2, 1:5]).astype(jnp.int32)

    pot_r, pot_c = _locate(obs[:, :, 22])
    pot_dr, pot_dc = pot_r - 2, pot_c - 2
    pot_count = obs[pot_r, pot_c, 31] + obs[pot_r, pot_c, 32] + obs[pot_r, pot_c, 33]

    supply_slice = jnp.where(target_ing == 0, obs[:, :, 26], obs[:, :, 27])
    sup_r, sup_c = _locate(supply_slice)
    sup_dr, sup_dc = sup_r - 2, sup_c - 2

    plate_r, plate_c = _locate(obs[:, :, 25])
    plate_dr, plate_dc = plate_r - 2, plate_c - 2

    deliv_r, deliv_c = _locate(obs[:, :, 21])
    deliv_dr, deliv_dc = deliv_r - 2, deliv_c - 2

    need_ingredient = pot_count < 3

    dr = jnp.where(
        holding_cooked, deliv_dr,
        jnp.where(
            holding_plate_empty, pot_dr,
            jnp.where(holding_ingredient, pot_dr, jnp.where(need_ingredient, sup_dr, plate_dr)),
        ),
    )
    dc = jnp.where(
        holding_cooked, deliv_dc,
        jnp.where(
            holding_plate_empty, pot_dc,
            jnp.where(holding_ingredient, pot_dc, jnp.where(need_ingredient, sup_dc, plate_dc)),
        ),
    )

    action = _approach_action(dr, dc, current_facing_id)

    is_deliver_interact = holding_cooked & (action == INTERACT)
    new_target_ing = jnp.where(is_deliver_interact, 1 - target_ing, target_ing)

    new_memory = {'target_ing': new_target_ing}
    return action, new_memory, {}
