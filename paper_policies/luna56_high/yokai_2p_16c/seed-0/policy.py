"""Convention-robust local policy for the two-player Yokai board."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'quadrant_exchange', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    # [0:16] known colors (-1 unknown), [16:48] last public positions,
    # [48] role, [49] turn counter, [50:66] private-observation flags.
    return jnp.zeros((66,), dtype=jnp.int32).at[:48].set(-1).at[49].set(0)


def _quadrant(x, y):
    return (x >= 5).astype(jnp.int32) * 2 + (y >= 5).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    nf = obs.node_features
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    active = nf[0, 12] == 1
    substep = nf[0, 13].astype(jnp.int32)

    init = jnp.zeros((66,), dtype=jnp.int32).at[:48].set(-1)
    init = init.at[16:48].set(pos.reshape(-1))
    init = init.at[48].set(1 - nf[0, 12].astype(jnp.int32))
    init = init.at[49].set(0)
    mem = jnp.where(jnp.asarray(reset), init, memory)

    colors = mem[:16]
    private = mem[50:66] > 0
    visible = nf[:16, :4].sum(axis=1) > 0
    observed_color = jnp.argmax(nf[:16, :4], axis=1).astype(jnp.int32)
    colors = jnp.where(visible, observed_color, colors)
    private = jnp.where(visible, True, private)

    old_pos = mem[16:48].reshape((16, 2))
    moved = jnp.any(pos != old_pos, axis=1) & (~jnp.asarray(reset))
    inside = (pos[:, 0] >= 3) & (pos[:, 0] <= 6) & (pos[:, 1] >= 3) & (pos[:, 1] <= 6)
    public_color = _quadrant(pos[:, 0], pos[:, 1])
    colors = jnp.where(moved & inside & (~private), public_color, colors)

    role = mem[48]
    ids = jnp.arange(16, dtype=jnp.int32)
    owned = (ids % 2) == role
    unlocked = nf[:16, 10] == 0
    unknown = colors < 0
    partner_seen = nf[:16, 11] == 1

    # Observation turns are deliberately disjoint between the two seats.
    observe_actions = jnp.arange(1, 17, dtype=jnp.int32)
    observe_legal = legal[1:17]
    observe_card = observe_actions - 1
    assigned = unknown[observe_card] & unlocked[observe_card] & owned[observe_card] & (~partner_seen[observe_card])
    learnable = unknown[observe_card] & unlocked[observe_card]
    obs_score = (
        observe_legal.astype(jnp.int32)
        + assigned.astype(jnp.int32) * 10000
        + learnable.astype(jnp.int32) * 100
        - observe_card
    )
    chosen_obs = jnp.argmax(jnp.where(observe_legal, obs_score, -10000000)).astype(jnp.int32) + 1

    # Prefer a legal relocation of a known card into its color quadrant.
    actions = jnp.arange(2325, dtype=jnp.int32)
    is_move = (actions >= 17) & (actions < 1617)
    move_card = jnp.clip((actions - 17) // 100, 0, 15)
    rem = jnp.maximum(actions - 17, 0) % 100
    dx = rem // 10
    dy = rem % 10
    dest_q = _quadrant(dx, dy)
    card_known = colors[move_card] >= 0
    card_wrong = _quadrant(pos[move_card, 0], pos[move_card, 1]) != colors[move_card]
    desired = card_known & card_wrong & (dest_q == colors[move_card])
    near = (jnp.abs(pos[:, 0, None] - dx[None, :]) + jnp.abs(pos[:, 1, None] - dy[None, :])) == 1
    same_color = (colors[:, None] == colors[move_card][None, :]) & (colors[:, None] >= 0)
    same_neighbors = (near & same_color).sum(axis=0)
    move_score = (
        is_move.astype(jnp.int32)
        + desired.astype(jnp.int32) * 5000
        + same_neighbors.astype(jnp.int32) * 1000
        + (card_known & card_wrong).astype(jnp.int32) * 100
        - actions
    )
    move_score = jnp.where(legal, move_score, -10000000)
    chosen_move = jnp.argmax(move_score).astype(jnp.int32)

    hints = nf[16:23, 4:8]
    reveal_legal = legal[1617:1624]
    reveal_actions = jnp.arange(1617, 1624, dtype=jnp.int32)
    reveal_score = jnp.where(reveal_legal, 1000 - reveal_actions, -10000000)
    reveal_choice = jnp.argmax(reveal_score).astype(jnp.int32) + 1617

    place_actions = jnp.arange(1624, 2324, dtype=jnp.int32)
    place_legal = legal[1624:2324]
    p_rem = (place_actions - 1624) % 100
    ph = (place_actions - 1624) // 100
    px = p_rem // 10
    py = p_rem % 10
    same = (pos[:, 0, None] == px[None, :]) & (pos[:, 1, None] == py[None, :])
    card_at = jnp.argmax(same.astype(jnp.int32), axis=0)
    placement_color = colors[card_at]
    correct = hints[ph, placement_color.clip(0, 3)] > 0
    place_score = correct.astype(jnp.int32) * 10000 - place_actions
    place_score = jnp.where(place_legal, place_score, -10000000)
    place_choice = jnp.argmax(place_score).astype(jnp.int32) + 1624
    hint_action = jnp.where(jnp.any(reveal_legal), reveal_choice, place_choice)

    all_known = jnp.all(colors >= 0)
    same_adj = (
        (obs.adjacency_matrix[:16, :16] > 0)
        & (colors[:, None] == colors[None, :])
        & (colors[:, None] >= 0)
        & (colors[None, :] >= 0)
    )
    reach = jax.lax.fori_loop(
        0, 4, lambda _, value: (value @ value > 0).astype(jnp.int32), same_adj.astype(jnp.int32)
    )
    color_ok = jnp.stack([
        jnp.all(jnp.where((colors[:, None] == c) & (colors[None, :] == c), reach > 0, True))
        for c in range(4)
    ])
    connected = jnp.all(color_ok)
    can_end = all_known & connected & legal[2324]

    action = jnp.where((substep == 0) | (substep == 1), chosen_obs, 0)
    action = jnp.where(substep == 2, chosen_move, action)
    action = jnp.where(substep == 3, hint_action, action)
    action = jnp.where((substep == 0) & can_end, 2324, action)
    action = jnp.where(active, action, 0).astype(jnp.int32)

    next_mem = mem.at[:16].set(colors)
    next_mem = next_mem.at[50:66].set(private.astype(jnp.int32))
    next_mem = next_mem.at[16:48].set(pos.reshape(-1))
    next_mem = next_mem.at[49].set(mem[49] + ((active & (substep == 0) & (~jnp.asarray(reset))).astype(jnp.int32)))
    return action, next_mem, {}
