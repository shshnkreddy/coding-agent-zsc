"""Shared local policy for the two-player Yokai rearrangement task."""

import jax.numpy as jnp
from jax import lax


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'memory_snake_candidate',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 3,
    'deck_size': 9,
    'grid_size': 9,
    'num_hints': 4,
    'num_nodes': 13,
    'node_features': 13,
    'num_actions': 1068,
    'policy_structure': 'shared_local',
}


def init_agent():
    return (
        jnp.full((9,), -1, dtype=jnp.int8),
        jnp.full((9,), -1, dtype=jnp.int8),
        jnp.zeros((3,), dtype=jnp.int8),
        jnp.asarray(-1, dtype=jnp.int8),
    )


def agent_act(obs, reset, memory, rng):
    del rng
    old_known, old_assigned, old_counts, old_last_move = memory
    initial_known = jnp.full((9,), -1, dtype=jnp.int8)
    initial_assigned = jnp.full((9,), -1, dtype=jnp.int8)
    initial_counts = jnp.zeros((3,), dtype=jnp.int8)
    initial_last_move = jnp.asarray(-1, dtype=jnp.int8)
    known = jnp.where(reset, initial_known, old_known)
    assigned = jnp.where(reset, initial_assigned, old_assigned)
    counts = jnp.where(reset, initial_counts, old_counts)
    last_move = jnp.where(reset, initial_last_move, old_last_move)

    card_features = obs.node_features[:9]
    visible = jnp.sum(card_features[:, :3], axis=1) > 0
    visible_color = jnp.argmax(card_features[:, :3], axis=1).astype(jnp.int8)
    known = jnp.where(visible, visible_color, known)
    color_ids = jnp.arange(3, dtype=jnp.int8)
    known_counts = jnp.sum((known[:, None] == color_ids[None, :]).astype(jnp.int32), axis=0)
    num_known = jnp.sum((known >= 0).astype(jnp.int32))
    can_infer_eighth = num_known == 8
    missing_color = jnp.argmin(known_counts).astype(jnp.int8)
    effective_known = jnp.where((known < 0) & can_infer_eighth, missing_color, known)

    # Give newly seen cards stable path slots in observation order.  The
    # color order is anchored by card 0, so the assignment is shared without
    # needing a player identity or a private convention.
    root_color = known[0]

    def assign_one(i, carry):
        cur_assigned, cur_counts = carry
        c = visible_color[i]
        new = visible[i] & (cur_assigned[i] < 0)
        base = jnp.where(c == root_color, 0, jnp.where(c == ((root_color + 1) % 3), 3, 6))
        slot = jnp.clip(base + cur_counts[c], 0, 8).astype(jnp.int8)
        one = jnp.asarray(1, dtype=jnp.int8)
        updated_assigned = cur_assigned.at[i].set(jnp.where(new, slot, cur_assigned[i]))
        updated_counts = cur_counts.at[c].add(jnp.where(new, one, jnp.asarray(0, dtype=jnp.int8)))
        return updated_assigned, updated_counts

    assigned, counts = lax.fori_loop(0, 9, assign_one, (assigned, counts))

    substep = obs.node_features[0, 11].astype(jnp.int32)
    active = obs.node_features[0, 10] > 0
    legal = obs.legal_actions

    # A connected snake starts at card 0's initial corner and uses three-cell
    # runs for each color. It can be reached with eight relocations.
    path_x = jnp.asarray([3, 2, 1, 0, 0, 0, 0, 1, 2], dtype=jnp.int32)
    path_y = jnp.asarray([3, 3, 3, 3, 2, 1, 0, 0, 0], dtype=jnp.int32)
    all_known = jnp.all(known >= 0)
    slot = jnp.clip(assigned.astype(jnp.int32), 0, 8)
    target_x = path_x[slot]
    target_y = path_y[slot]

    current_x = obs.positions[:, 0]
    current_y = obs.positions[:, 1]
    at_target = (current_x == target_x) & (current_y == target_y)

    # Score every legal relocation by the remembered same-color neighbors it
    # would create, then by total neighbors.  The arrays are fixed [9, 81].
    cell_x = jnp.repeat(jnp.arange(9, dtype=jnp.int32), 9)
    cell_y = jnp.tile(jnp.arange(9, dtype=jnp.int32), 9)
    candidate_actions = 10 + jnp.arange(9, dtype=jnp.int32)[:, None] * 81
    candidate_actions = candidate_actions + cell_x[None, :] * 9 + cell_y[None, :]
    candidate_legal = legal[candidate_actions]
    near = (
        (jnp.abs(cell_x[None, :, None] - current_x[None, None, :])
         + jnp.abs(cell_y[None, :, None] - current_y[None, None, :])) == 1
    ).astype(jnp.int32)
    not_self = 1 - jnp.eye(9, dtype=jnp.int32)[:, None, :]
    near = near * not_self
    known_pair = (effective_known[:, None] == effective_known[None, :]) & (effective_known[:, None] >= 0)
    same_neighbors = jnp.sum(near * known_pair[:, None, :].astype(jnp.int32), axis=2)
    total_neighbors = jnp.sum(near, axis=2)
    current_adj = obs.adjacency_matrix[:9, :9].astype(jnp.int32) * (1 - jnp.eye(9, dtype=jnp.int32))
    current_same = jnp.sum(current_adj * known_pair.astype(jnp.int32), axis=1)
    current_total = jnp.sum(current_adj, axis=1)
    eligible = candidate_legal & (jnp.arange(9, dtype=jnp.int32)[:, None] > 0) & (effective_known[:, None] >= 0)
    cooldown_eligible = eligible & (jnp.arange(9, dtype=jnp.int8)[:, None] != last_move)
    move_score = (same_neighbors - current_same[:, None]) * 100 + total_neighbors - current_total[:, None]
    use_cooldown = jnp.any(cooldown_eligible)
    scored_eligible = jnp.where(use_cooldown, cooldown_eligible, eligible)
    scored_moves = jnp.where(scored_eligible, move_score, -jnp.ones_like(move_score) * 1000000)
    selected_move = jnp.argmax(scored_moves.reshape(-1))
    has_scored_move = jnp.any(scored_eligible)
    scored_action = candidate_actions.reshape(-1)[selected_move]
    fallback_move = 10 + jnp.argmax(legal[10:739].astype(jnp.int32))
    chosen_move = jnp.where(has_scored_move, scored_action, fallback_move)

    observe_legal = legal[1:10]
    unknown = known < 0
    observe_candidates = observe_legal & unknown
    observe_action = 1 + jnp.argmax(observe_candidates.astype(jnp.int32))
    observe_fallback = 1 + jnp.argmax(observe_legal.astype(jnp.int32))
    chosen_observe = jnp.where(jnp.any(observe_candidates), observe_action, observe_fallback)

    # Check the color groups using the public card adjacency graph.
    card_adj = obs.adjacency_matrix[:9, :9].astype(jnp.int32)
    offdiag = 1 - jnp.eye(9, dtype=jnp.int32)
    left = (effective_known[:, None, None] == color_ids[None, None, :]).astype(jnp.int32)
    right = (effective_known[None, :, None] == color_ids[None, None, :]).astype(jnp.int32)
    connected_edges = jnp.sum(card_adj[:, :, None] * left * right * offdiag[:, :, None], axis=(0, 1))
    clustered = (all_known | can_infer_eighth) & jnp.all(connected_edges >= 4)
    end_action = jnp.asarray(1067, dtype=jnp.int32)

    hint_features = obs.node_features[9:13]
    hint_sets = hint_features[:, 3:6].astype(jnp.int32)
    hint_revealed = jnp.sum(hint_sets, axis=1) > 0
    hint_placed = hint_features[:, 8] > 0
    reveal_candidates = legal[739:743] & (~hint_revealed)
    reveal_action = 739 + jnp.argmax(reveal_candidates.astype(jnp.int32))
    any_reveal = jnp.any(reveal_candidates)

    card_unlocked = card_features[:, 8] == 0
    matching = hint_sets[:, None, :] * (effective_known[None, :, None] == color_ids[None, None, :])
    matching = jnp.sum(matching, axis=2) > 0
    place_actions = 743 + jnp.arange(4, dtype=jnp.int32)[:, None] * 81
    place_actions = place_actions + current_x[None, :] * 9 + current_y[None, :]
    place_legal = legal[place_actions]
    place_candidates = place_legal & matching & card_unlocked[None, :] & (~hint_placed[:, None])
    preferred_place = place_candidates & at_target[None, :]
    any_preferred = jnp.any(preferred_place)
    any_place = jnp.any(place_candidates)
    place_flat = jnp.argmax(preferred_place.reshape(-1).astype(jnp.int32))
    place_any_flat = jnp.argmax(place_candidates.reshape(-1).astype(jnp.int32))
    preferred_hint = place_flat // 9
    preferred_card = place_flat % 9
    any_hint = place_any_flat // 9
    any_card = place_any_flat % 9
    preferred_action = 743 + preferred_hint * 81 + current_x[preferred_card] * 9 + current_y[preferred_card]
    any_place_action = 743 + any_hint * 81 + current_x[any_card] * 9 + current_y[any_card]
    place_fallback = 743 + jnp.argmax(legal[743:1067].astype(jnp.int32))
    chosen_place = jnp.where(any_preferred, preferred_action, jnp.where(any_place, any_place_action, place_fallback))
    hint_action = jnp.where(any_reveal, reveal_action, chosen_place)

    action = jnp.where(
        ~active,
        0,
        jnp.where(
            substep == 0,
            jnp.where(clustered & legal[1067], end_action, chosen_observe),
            jnp.where(substep == 1, chosen_observe, jnp.where(substep == 2, chosen_move, hint_action)),
        ),
    ).astype(jnp.int32)
    chosen_card = jnp.clip((chosen_move - 10) // 81, 0, 8).astype(jnp.int8)
    moved_now = active & (substep == 2)
    next_last_move = jnp.where(moved_now, chosen_card, last_move)
    return action, (known, assigned, counts, next_last_move), {}
