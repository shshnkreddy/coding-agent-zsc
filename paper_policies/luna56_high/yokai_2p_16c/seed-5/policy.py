"""Shared local policy for the two-player Yokai rearrangement game."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    legal = obs.legal_actions
    features = obs.node_features
    active = features[0, 12] > 0
    substep = features[0, 13]
    call_step = jnp.where(reset, jnp.asarray(0, jnp.int32), memory)
    turn = call_step // 4

    legal_i = legal.astype(jnp.int32)
    fallback = jnp.argmax(legal_i).astype(jnp.int32)

    # The partner-observed bit gives a public, role-free division of the deck:
    # on each turn a seat claims cards that the partner has not claimed yet.
    private_known = jnp.any(features[:16, :4] > 0, axis=1)
    private_colors = jnp.argmax(features[:16, :4].astype(jnp.int32), axis=1)
    hint_sets = features[16:23, 4:8] > 0
    hint_is_singleton = jnp.sum(hint_sets.astype(jnp.int32), axis=1) == 1
    exact_hint = hint_is_singleton & (features[16:23, 10] > 0)
    hint_colors = jnp.argmax(hint_sets.astype(jnp.int32), axis=1)
    card_hint_link = obs.adjacency_matrix[:16, 16:23] > 0
    public_known = jnp.any(card_hint_link & exact_hint[None, :], axis=1)
    public_colors = jnp.sum(
        card_hint_link.astype(jnp.int32) * exact_hint[None, :].astype(jnp.int32)
        * hint_colors[None, :], axis=1
    )
    known = private_known | public_known
    partner_seen = features[:16, 11] > 0
    observe_candidates = legal[1:17] & (~known) & (~partner_seen)
    observe_any = jnp.any(observe_candidates)
    observe_action = 1 + jnp.argmax(observe_candidates.astype(jnp.int32))
    observe_action = jnp.where(observe_any, observe_action, fallback)

    # Fixed four-cell regions are connected and form a connected 4x4 board.
    # The block is shifted one cell to y=4..7, leaving a single empty row at
    # its far edge while retaining 12 initially occupied target cells.
    # The policy uses the same public color-to-region convention in both seats.
    pos = obs.positions
    colors = jnp.where(private_known, private_colors, public_colors)
    in_target_card = (
        (pos[:, 0] >= 3) & (pos[:, 0] <= 6) &
        (pos[:, 1] >= 4) & (pos[:, 1] <= 7)
    )
    current_region = jnp.where(
        in_target_card,
        ((pos[:, 0] - 3) // 2) * 2 + ((pos[:, 1] - 4) // 2),
        -1,
    )
    wrong_region = known & in_target_card & (current_region != colors)

    # Enumerate all card/destination moves in a fixed-size tensor.  The legal
    # mask remains authoritative for connectivity, occupancy and locks.
    grid_x = jnp.repeat(jnp.arange(10, dtype=jnp.int32), 10)
    grid_y = jnp.tile(jnp.arange(10, dtype=jnp.int32), 10)
    card_id = jnp.repeat(jnp.arange(16, dtype=jnp.int32), 100)
    dest_x = jnp.tile(grid_x, 16)
    dest_y = jnp.tile(grid_y, 16)
    move_actions = 17 + card_id * 100 + dest_x * 10 + dest_y
    move_legal = legal[move_actions]
    dest_in_target = (
        (dest_x >= 3) & (dest_x <= 6) &
        (dest_y >= 4) & (dest_y <= 7)
    )
    dest_region = jnp.where(
        dest_in_target,
        ((dest_x - 3) // 2) * 2 + ((dest_y - 4) // 2),
        -1,
    )
    card_known = known[card_id]
    card_wrong = wrong_region[card_id]
    card_color = colors[card_id]
    card_misplaced = card_known & ((~in_target_card[card_id]) | card_wrong)
    direct = card_misplaced & dest_in_target & (dest_region == card_color)
    clear_wrong = card_wrong & (~dest_in_target)
    move_score = move_legal.astype(jnp.int32) * (
        direct.astype(jnp.int32) * 10000 +
        clear_wrong.astype(jnp.int32) * 1000 +
        card_known.astype(jnp.int32)
    )
    move_flat = jnp.argmax(move_score).astype(jnp.int32)
    move_action = move_actions[move_flat]
    move_any = jnp.any(move_score > 0)
    move_action = jnp.where(move_any, move_action, fallback)

    # Reveal hints while possible.  Once all are public, prefer a legal
    # placement whose revealed set contains a privately known card color.
    reveal_candidates = legal[1617:1624]
    reveal_action = 1617 + jnp.argmax(reveal_candidates.astype(jnp.int32))
    reveal_any = jnp.any(reveal_candidates)
    place_x = jnp.tile(grid_x, 7)
    place_y = jnp.tile(grid_y, 7)
    hint_id = jnp.repeat(jnp.arange(7, dtype=jnp.int32), 100)
    place_actions = 1624 + hint_id * 100 + place_x * 10 + place_y
    place_legal = legal[place_actions]
    card_at_cell = jnp.sum(
        (pos[:, None, 0] == place_x[None, :]) &
        (pos[:, None, 1] == place_y[None, :]), axis=0
    )
    # Reconstruct the color of the card at each placement cell from the
    # public position and the active seat's private observations.
    same_cell = (
        (pos[:, None, 0] == place_x[None, :]) &
        (pos[:, None, 1] == place_y[None, :])
    )
    cell_known = jnp.any(same_cell & known[:, None], axis=0)
    cell_color = jnp.sum(
        same_cell.astype(jnp.int32) * colors[:, None], axis=0
    )
    cell_in_target = (
        (place_x >= 3) & (place_x <= 6) &
        (place_y >= 4) & (place_y <= 7)
    )
    cell_region = jnp.where(
        cell_in_target,
        ((place_x - 3) // 2) * 2 + ((place_y - 4) // 2),
        -1,
    )
    cell_correct = cell_known & cell_in_target & (cell_region == cell_color)
    hint_match = hint_sets[hint_id, cell_color] & cell_correct
    place_score = place_legal.astype(jnp.int32) * (
        hint_match.astype(jnp.int32) * 1000 +
        cell_correct.astype(jnp.int32) * 100 +
        cell_in_target.astype(jnp.int32) * 10
    )
    place_flat = jnp.argmax(place_score).astype(jnp.int32)
    place_action = place_actions[place_flat]
    place_any = jnp.any(place_score > 0)
    place_action = jnp.where(place_any, place_action, fallback)
    hint_action = jnp.where(reveal_any, reveal_action, place_action)

    # At turn 13, substep 0 is the last safe opportunity to end before the
    # 56-substep cap.  Otherwise choose the phase-specific action.
    should_end = (turn >= 13) & legal[2324]
    active_action = jnp.where(
        substep == 0, jnp.where(should_end, jnp.asarray(2324, jnp.int32), observe_action),
        jnp.where(substep == 1, observe_action,
                  jnp.where(substep == 2, move_action, hint_action)),
    )
    action = jnp.where(active, active_action, jnp.asarray(0, jnp.int32))
    next_memory = call_step + 1
    return action.astype(jnp.int32), next_memory.astype(jnp.int32), {}
