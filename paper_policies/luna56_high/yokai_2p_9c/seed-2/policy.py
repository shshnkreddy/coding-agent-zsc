"""A small convention-oriented policy for the two-player Yokai board."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'stable_id_grouping', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    # [step, color(card 0), ..., color(card 8)], with colors 0..2 encoded 1..3.
    return jnp.zeros((10,), dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    nf = obs.node_features
    legal = obs.legal_actions
    step = memory[0]
    active = nf[0, 10] > 0
    substep = nf[0, 11]
    turn = step // 4

    # Perfect memory is visible only while this seat is active, so copy every
    # newly visible one-hot color into the private fixed-shape memory.
    one_hot = nf[:9, :3]
    visible = jnp.sum(one_hot, axis=1) > 0
    visible_color = jnp.argmax(one_hot, axis=1).astype(jnp.int32) + 1
    old_colors = memory[1:]
    colors = jnp.where(active & visible, visible_color, old_colors)

    # If the shared observation schedule has seen eight cards, the ninth color
    # follows from the fixed three-of-each deck.
    counts = jnp.asarray([jnp.sum(colors == (k + 1)) for k in range(3)])
    unknown = jnp.sum(colors == 0)
    missing_color = jnp.where(counts[0] < 3, 1, jnp.where(counts[1] < 3, 2, 3))
    complete_colors = jnp.where((unknown == 1) & (colors == 0), missing_color, colors)

    # Both seats inspect IDs 0..7 in four pairs. Stable IDs make this useful
    # even when paired with another policy.
    role_one = (step % 8) >= 4
    local_turn = jnp.where(role_one, (step - 4) // 8, step // 8)
    pair_id = jnp.clip(local_turn, 0, 3) * 2 + (substep == 1)
    observe_action = 1 + pair_id

    # Prefer a legal relocation that makes a known card adjacent to another
    # known card of the same color. This uses every move substep for gradual
    # grouping, while the legal mask enforces occupancy and whole-board
    # connectivity.
    card_ids = jnp.arange(9, dtype=jnp.int32)
    color = complete_colors
    grid_id = jnp.arange(81, dtype=jnp.int32)
    cand_card = jnp.repeat(card_ids, 81)
    cand_x = jnp.tile(grid_id // 9, 9)
    cand_y = jnp.tile(grid_id % 9, 9)
    cand_actions = 10 + cand_card * 81 + cand_x * 9 + cand_y
    cand_legal = legal[cand_actions]
    cand_known = color[cand_card] > 0
    cand_unlocked = nf[cand_card, 8] == 0
    cand_not_here = (nf[cand_card, 6] != cand_x) | (nf[cand_card, 7] != cand_y)
    target = jnp.stack((cand_x, cand_y), axis=1)
    distance = jnp.abs(target[:, None, 0] - obs.positions[None, :, 0]) + jnp.abs(target[:, None, 1] - obs.positions[None, :, 1])
    same_color = (color[cand_card, None] == color[None, :]) & (color[cand_card, None] > 0)
    not_self = card_ids[None, :] != cand_card[:, None]
    target_same = jnp.any((distance == 1) & same_color & not_self, axis=1)
    current_distance = jnp.abs(obs.positions[:, None, 0] - obs.positions[None, :, 0]) + jnp.abs(obs.positions[:, None, 1] - obs.positions[None, :, 1])
    current_same = jnp.any((current_distance == 1) & (color[:, None] == color[None, :]) & (color[:, None] > 0) & (card_ids[None, :] != card_ids[:, None]), axis=1)
    group_mask = cand_legal & cand_known & cand_unlocked & cand_not_here & target_same & ~current_same[cand_card]
    group_mask = jnp.where(jnp.any(group_mask), group_mask, cand_legal & cand_known & cand_unlocked & cand_not_here & target_same)
    group_index = jnp.argmax(group_mask.astype(jnp.int32))
    group_action = cand_actions[group_index]
    group_any = jnp.any(group_mask)
    move_action = jnp.where(group_any, group_action, jnp.argmax(legal.astype(jnp.int32)))

    # Reveal one hint per early turn. Thereafter place the largest remaining
    # revealed set first, on a locally known compatible unlocked card.
    hint_rows = nf[9:13]
    hint_sizes = jnp.sum(hint_rows[:, 3:6], axis=1)
    unplaced = hint_rows[:, 8] == 0
    hint_priority = jnp.where(unplaced, hint_sizes * 10 - jnp.arange(4), -100)
    hint_id = jnp.argmax(hint_priority).astype(jnp.int32)
    chosen_set = hint_rows[hint_id, 3:6]
    compatible = ((complete_colors > 0) & (chosen_set[complete_colors - 1] > 0) & (nf[:9, 8] == 0))
    card_adjacency = obs.adjacency_matrix[:9, :9]
    same_degree = jnp.sum(card_adjacency * (complete_colors[:, None] == complete_colors[None, :]) * (card_ids[:, None] != card_ids[None, :]), axis=1)
    target_priority = jnp.where(compatible, same_degree * 10 - card_ids, -100)
    target_card = jnp.argmax(target_priority.astype(jnp.int32))
    any_compatible = jnp.any(compatible)
    unlocked_card = nf[:9, 8] == 0
    fallback_card = jnp.argmax(unlocked_card.astype(jnp.int32))
    target_card = jnp.where(any_compatible, target_card, fallback_card)
    target_pos = obs.positions[target_card]
    place_action = 743 + hint_id * 81 + target_pos[0] * 9 + target_pos[1]
    reveal_action = 739 + jnp.clip(turn, 0, 3)
    hint_action = jnp.where(turn < 4, reveal_action, place_action)

    desired = jnp.where(substep < 2, observe_action, jnp.where(substep == 2, move_action, hint_action))
    initial_positions = jnp.asarray([[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5]], dtype=jnp.int32)
    board_untouched = jnp.all(obs.positions == initial_positions) & jnp.all(nf[:9, 8] == 0)
    early_end = jnp.asarray(False)
    desired = jnp.where(early_end, 1067, desired)
    desired = jnp.where(active, desired, 0)
    desired_legal = legal[desired]
    first_legal = jnp.argmax(legal.astype(jnp.int32)).astype(jnp.int32)
    action = jnp.where(desired_legal, desired, first_legal).astype(jnp.int32)

    next_memory = jnp.concatenate((jnp.asarray([step + 1], dtype=jnp.int32), colors))
    next_memory = jnp.where(reset, jnp.concatenate((jnp.asarray([1], dtype=jnp.int32), colors)), next_memory)
    return action, next_memory, {}
