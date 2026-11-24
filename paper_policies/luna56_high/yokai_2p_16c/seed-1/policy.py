"""A convention-light information and color-clustering policy."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    # own/received colors, previous public positions, and active-turn count.
    memory = jnp.full((81,), -1, dtype=jnp.int32)
    return memory.at[80].set(0)


def _first_true(mask):
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _connected_by_color(obs, colors, all_known):
    cards = obs.adjacency_matrix[:16, :16].astype(jnp.int32)
    equal = colors[:, None] == colors[None, :]
    connected = []
    node_ids = jnp.arange(16, dtype=jnp.int32)
    for color in range(4):
        in_color = colors == color
        graph = cards * equal * in_color[:, None] * in_color[None, :]
        representative = _first_true(in_color)
        reached = in_color & (node_ids == representative)
        for _ in range(16):
            reached = reached | (graph @ reached.astype(jnp.int32) > 0)
        connected.append(jnp.sum(reached & in_color) == 4)
    return all_known & jnp.all(jnp.stack(connected))


def _safe_certificate(obs, colors, known):
    """Sufficient connectivity test despite up to four unobserved cards."""
    cards = obs.adjacency_matrix[:16, :16].astype(jnp.int32)
    node_ids = jnp.arange(16, dtype=jnp.int32)
    board_reached = node_ids == 0
    for _ in range(16):
        board_reached = board_reached | (cards @ board_reached.astype(jnp.int32) > 0)
    board_connected = jnp.all(board_reached)

    counts = jnp.sum(
        known[:, None] & (colors[:, None] == jnp.arange(4, dtype=jnp.int32)[None, :]),
        axis=0,
    )
    missing = 4 - counts
    known_group_ok = []
    attach_ok = jnp.ones((), dtype=jnp.bool_)
    for color in range(4):
        group = known & (colors == color)
        group_graph = cards * group[:, None] * group[None, :]
        representative = _first_true(group)
        reached = group & (node_ids == representative)
        for _ in range(16):
            reached = reached | (group_graph @ reached.astype(jnp.int32) > 0)
        known_group_ok.append(jnp.sum(reached & group) == counts[color])
        color_neighbors = cards @ group.astype(jnp.int32) > 0
        needs_attachment = (counts[color] > 0) & (missing[color] > 0)
        attach_ok = attach_ok & jnp.all(
            jnp.where(known, True, (~needs_attachment) | color_neighbors)
        )

    unknown = ~known
    unknown_graph = cards * unknown[:, None] * unknown[None, :]
    unknown_representative = _first_true(unknown)
    unknown_reached = unknown & (node_ids == unknown_representative)
    for _ in range(16):
        unknown_reached = unknown_reached | (
            unknown_graph @ unknown_reached.astype(jnp.int32) > 0
        )
    unknown_connected = jnp.sum(unknown_reached & unknown) == jnp.sum(unknown)
    enough_known = jnp.sum(known) >= 12
    # With four unknown cards, a count of four identifies a unique all-unknown
    # color.  Multiple count-four entries (the initial no-information state)
    # are ambiguous and must not certify anything.
    all_unknown_color = jnp.sum(missing == 4) == 1
    return (
        enough_known
        & board_connected
        & jnp.all(jnp.stack(known_group_ok))
        & attach_ok
        & ((~all_unknown_color) | unknown_connected)
    )


def _canonical_target_cells(colors, known):
    # Four 2x2 blocks in the original 4x4 footprint, one block per color.
    safe_colors = jnp.where(known, colors, 0)
    equal = (safe_colors[:, None] == safe_colors[None, :]) & known[:, None] & known[None, :]
    rank = jnp.sum(jnp.tril(equal), axis=1).astype(jnp.int32) - 1
    base_x = 3 + 2 * (safe_colors % 2)
    base_y = 3 + 2 * (safe_colors // 2)
    return base_x + rank.clip(0, 3) % 2, base_y + rank.clip(0, 3) // 2


def _robust_target_cells(colors, known):
    counts = jnp.sum(
        known[:, None] & (colors[:, None] == jnp.arange(4, dtype=jnp.int32)[None, :]),
        axis=0,
    )
    missing = 4 - counts
    color_ids = jnp.arange(4, dtype=jnp.int32)
    missing_mask = missing > 0
    first_missing = _first_true(missing_mask)
    second_missing = _first_true(missing_mask & (color_ids != first_missing))
    remaining = ~(color_ids == first_missing) & ~(color_ids == second_missing)
    first_remaining = _first_true(remaining)
    second_remaining = _first_true(remaining & (color_ids != first_remaining))
    same_other_1 = _first_true(color_ids != first_missing)
    same_other_2 = _first_true((color_ids != first_missing) & (color_ids != same_other_1))
    same_other_3 = _first_true(
        (color_ids != first_missing)
        & (color_ids != same_other_1)
        & (color_ids != same_other_2)
    )
    same_roles = jnp.stack(
        (first_missing, same_other_1, same_other_2, same_other_3)
    )
    diff_roles = jnp.stack(
        (first_missing, second_missing, first_remaining, second_remaining)
    )
    role_colors = jnp.where((jnp.sum(missing_mask) == 1), same_roles, diff_roles)
    roles = jnp.sum(
        (role_colors[None, :] == jnp.where(known, colors, 0)[:, None])
        * jnp.arange(4, dtype=jnp.int32)[None, :],
        axis=1,
    )
    equal_known = (colors[:, None] == colors[None, :]) & known[:, None] & known[None, :]
    ranks = jnp.sum(jnp.tril(equal_known), axis=1).astype(jnp.int32) - 1

    same_x = jnp.asarray([[3, 4, 0, 0], [2, 2, 2, 2], [1, 1, 1, 1], [3, 4, 5, 5]], jnp.int32)
    same_y = jnp.asarray([[4, 4, 0, 0], [3, 4, 5, 6], [3, 4, 5, 6], [6, 6, 6, 5]], jnp.int32)
    diff_x = jnp.asarray([[3, 4, 4, 0], [2, 2, 2, 0], [1, 1, 1, 1], [2, 3, 4, 5]], jnp.int32)
    diff_y = jnp.asarray([[4, 4, 5, 0], [3, 4, 5, 0], [3, 4, 5, 6], [6, 6, 6, 6]], jnp.int32)
    use_same = jnp.sum(missing_mask) == 1
    slot_x = jnp.where(use_same, same_x, diff_x)
    slot_y = jnp.where(use_same, same_y, diff_y)
    known_rank = ranks.clip(0, 3)
    known_x = slot_x[roles, known_rank]
    known_y = slot_y[roles, known_rank]
    unknown_rank = (jnp.cumsum((~known).astype(jnp.int32)) - 1).clip(0, 1)
    hole_x = jnp.asarray([3, 3], jnp.int32)
    hole_y = jnp.asarray([3, 5], jnp.int32)
    target_x = jnp.where(known, known_x, hole_x[unknown_rank])
    target_y = jnp.where(known, known_y, hole_y[unknown_rank])
    return target_x, target_y


def _hint_action(obs, colors, known):
    legal = obs.legal_actions
    hint_rows = obs.node_features[16:23, 4:8]
    unrevealed = jnp.sum(hint_rows, axis=1) == 0
    reveal_legal = legal[1617:1624]
    reveal = _first_true(reveal_legal)
    reveal_action = 1617 + reveal

    # If all hints have been exposed, prefer a legal placement whose revealed
    # set contains our privately known card color.  Usually the episode ends
    # before this branch is needed.
    hint_ids = jnp.arange(7, dtype=jnp.int32)[:, None]
    card_ids = jnp.arange(16, dtype=jnp.int32)[None, :]
    px = obs.positions[:, 0][None, :]
    py = obs.positions[:, 1][None, :]
    place_ids = 1624 + hint_ids * 100 + px * 10 + py
    place_legal = legal[place_ids]
    sets = jnp.broadcast_to(hint_rows[:, None, :], (7, 16, 4))
    card_colors = jnp.broadcast_to(colors[None, :, None].clip(0, 3), (7, 16, 1))
    correct = (
        (known[None, :, None] >= 0)
        & (jnp.take_along_axis(sets, card_colors, axis=2) > 0)
    )[:, :, 0]
    correct_place = place_legal & correct
    place_flat = _first_true(correct_place.reshape(-1))
    any_correct = jnp.any(correct_place)
    any_place = jnp.any(place_legal)
    fallback_place = _first_true(place_legal.reshape(-1))
    flat_place = jnp.where(any_correct, place_flat, fallback_place)
    place_action = place_ids.reshape(-1)[flat_place]
    place_action = jnp.where(any_place, place_action, jnp.asarray(0, jnp.int32))

    any_reveal = jnp.any(reveal_legal & unrevealed)
    # legal reveal entries already encode unrevealed status, but the explicit
    # mask makes the intent robust to adapter changes.
    reveal_action = 1617 + _first_true(reveal_legal & unrevealed)
    return jnp.where(any_reveal, reveal_action, place_action)


def agent_act(obs, reset, memory, rng):
    prior_colors = jnp.where(reset, jnp.full((16,), -1, jnp.int32), memory[:16])
    received = jnp.where(reset, jnp.full((16,), -1, jnp.int32), memory[16:32])
    positions = obs.positions
    prior_positions = jnp.where(
        reset, positions.reshape(-1), memory[32:64]
    ).reshape((16, 2))
    current_partner_seen = obs.node_features[:16, 11] > 0
    prior_partner_seen = jnp.where(
        reset, current_partner_seen, memory[64:80] > 0
    )
    visible = jnp.sum(obs.node_features[:16, :4], axis=1) > 0
    visible_colors = jnp.argmax(obs.node_features[:16, :4], axis=1).astype(jnp.int32)
    colors = jnp.where(visible, visible_colors, prior_colors)

    # A public movement made at substep 2 is visible to both seats at
    # substep 3.  The destination cell carries a pair of colors, while the
    # public partner-seen flags identify the two card IDs in that pair.
    changed = jnp.any(positions != prior_positions, axis=1)
    changed_card = _first_true(changed)
    new_partner = current_partner_seen & (~prior_partner_seen)
    pair_first = _first_true(new_partner)
    pair_second = _first_true(new_partner & (jnp.arange(16) != pair_first))
    code = (positions[changed_card, 0] * 10 + positions[changed_card, 1]) % 16
    decoded_first = code // 4
    decoded_second = code % 4
    message_valid = (
        (jnp.sum(changed) == 1)
        & (jnp.sum(new_partner) == 2)
        & (obs.node_features[0, 13] == 3)
    )
    ids = jnp.arange(16, dtype=jnp.int32)
    received = jnp.where(message_valid & (ids == pair_first), decoded_first, received)
    received = jnp.where(message_valid & (ids == pair_second), decoded_second, received)
    known = colors >= 0
    all_known = jnp.all(known)

    substep = obs.node_features[0, 13].astype(jnp.int32)
    active = obs.node_features[0, 12] > 0
    legal = obs.legal_actions

    # Public partner-seen flags let identical local policies partition the
    # inspection work without requiring a seat-specific convention.
    partner_seen = obs.node_features[:16, 11] > 0
    observe_legal = legal[1:17]
    preferred = observe_legal & (~known) & (~partner_seen)
    secondary = observe_legal & (~known)
    observe_index = jnp.where(
        jnp.any(preferred),
        _first_true(preferred),
        jnp.where(jnp.any(secondary), _first_true(secondary), _first_true(observe_legal)),
    )
    observe_action = 1 + observe_index

    known_count = jnp.sum(known)
    canonical_x, canonical_y = _canonical_target_cells(colors, known)
    robust_x, robust_y = _robust_target_cells(colors, known)
    robust_ready = known_count == 14
    target_x = jnp.where(robust_ready, robust_x, canonical_x)
    target_y = jnp.where(robust_ready, robust_y, canonical_y)
    # Before the robust two-unknown layout is available, leave unknown cards
    # in place and use the move slot for known cards only.
    target_x = jnp.where(known | robust_ready | all_known, target_x, positions[:, 0])
    target_y = jnp.where(known | robust_ready | all_known, target_y, positions[:, 1])
    at_target = (positions[:, 0] == target_x) & (positions[:, 1] == target_y)
    empty_target = jnp.all(
        (positions[:, 0][None, :] != target_x[:, None])
        | (positions[:, 1][None, :] != target_y[:, None]),
        axis=1,
    )
    card_ids = jnp.arange(16, dtype=jnp.int32)
    desired_ids = 17 + card_ids * 100 + target_x * 10 + target_y
    desired_legal = legal[desired_ids]
    targetable = known | robust_ready | all_known
    desired = targetable & (~at_target) & empty_target & desired_legal
    desired_index = _first_true(desired)
    desired_action = desired_ids[desired_index]

    move_grid = legal[17:1617].reshape((16, 100))
    wrong_card = targetable & (~at_target)
    wrong_moves = move_grid & wrong_card[:, None]
    any_wrong_move = jnp.any(wrong_moves)
    wrong_flat = _first_true(wrong_moves.reshape(-1))
    fallback_flat = _first_true(move_grid.reshape(-1))
    fallback_action = 17 + jnp.where(any_wrong_move, wrong_flat, fallback_flat)
    move_action = jnp.where(jnp.any(desired), desired_action, fallback_action)

    # Prefer a legal destination whose cell index carries both colors of the
    # two cards newly exposed this turn. If no such destination is legal,
    # retain the ordinary target/fallback move.
    new_card = visible & (prior_colors < 0)
    message_card = _first_true(new_card)
    second_new_card = _first_true(new_card & (jnp.arange(16) != message_card))
    message_code = colors[message_card].clip(0, 3) * 4 + colors[second_new_card].clip(0, 3)
    destination_index = jnp.arange(100, dtype=jnp.int32)
    message_moves = move_grid[message_card] & (destination_index % 16 == message_code)
    message_any = (jnp.sum(new_card) == 2) & jnp.any(message_moves)
    message_flat = _first_true(message_moves)
    message_action = 17 + message_card * 100 + message_flat

    # A color-component merge is a more useful post-information move than a
    # permutation step: choose a card in an incomplete known component and a
    # legal destination orthogonally adjacent to another card of its color.
    card_graph = obs.adjacency_matrix[:16, :16].astype(jnp.int32)
    same_graph = card_graph * (
        known[:, None]
        & known[None, :]
        & (colors[:, None] == colors[None, :])
    )
    reach = jnp.eye(16, dtype=jnp.bool_)
    for _ in range(16):
        reach = reach | (reach.astype(jnp.int32) @ same_graph > 0)
    component_size = jnp.sum(reach, axis=1)
    incomplete = known & (component_size < 4)
    destination_x = jnp.arange(100, dtype=jnp.int32) // 10
    destination_y = jnp.arange(100, dtype=jnp.int32) % 10
    destination_adjacent = (
        (destination_x[None, :, None] == positions[None, None, :, 0])
        & (destination_y[None, :, None] == positions[None, None, :, 1])
    )
    not_self = ~jnp.eye(16, dtype=jnp.bool_)
    same_other = (
        known[:, None]
        & known[None, :]
        & (colors[:, None] == colors[None, :])
        & not_self
    )
    merge_destination = jnp.any(
        destination_adjacent & same_other[:, None, :], axis=2
    )
    merge_moves = move_grid & merge_destination & incomplete[:, None]
    merge_any = jnp.any(merge_moves)
    merge_flat = _first_true(merge_moves.reshape(-1))
    merge_action = 17 + merge_flat
    move_action = jnp.where(merge_any, merge_action, move_action)

    hint_action = _hint_action(obs, colors, known)
    base_action = jnp.where(
        (substep == 0) | (substep == 1),
        observe_action,
        jnp.where(substep == 2, move_action, hint_action),
    )
    layout_done = (robust_ready | all_known) & jnp.all(at_target)
    solved_known = _connected_by_color(obs, colors, all_known)
    certificate = _safe_certificate(obs, colors, known)
    end_ok = active & (substep == 0) & (layout_done | solved_known | certificate) & legal[2324]
    action = jnp.where(end_ok, jnp.asarray(2324, jnp.int32), base_action)
    action = jnp.where(active, action, jnp.asarray(0, jnp.int32))

    next_partner_seen = jnp.where(
        (substep == 0), current_partner_seen, prior_partner_seen
    )
    turns = jnp.where(reset, jnp.asarray(0, jnp.int32), memory[80])
    turns = turns + (active & (substep == 3)).astype(jnp.int32)
    next_memory = jnp.concatenate(
        (colors, received, positions.reshape(-1), next_partner_seen.astype(jnp.int32), turns[None])
    )
    trace = {
        "known_count": known_count.astype(jnp.int32),
        "solved": layout_done | solved_known | certificate,
        "received_count": jnp.sum(received >= 0).astype(jnp.int32),
        "turns": turns,
    }
    return action.astype(jnp.int32), next_memory, trace
