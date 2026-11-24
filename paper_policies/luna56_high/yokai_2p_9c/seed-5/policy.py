"""A local, information-seeking heuristic for the two-player Yokai board."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'local_adjacency_learner',
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
    return {
        'known': jnp.zeros((9, 3), dtype=jnp.int32),
        'turn': jnp.asarray(0, dtype=jnp.int32),
        'cooldown': jnp.zeros((9,), dtype=jnp.int32),
        'prev_positions': jnp.zeros((9, 2), dtype=jnp.int32),
    }


def _legal_first(mask):
    return jnp.argmax(mask).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    active = features[0, 10] > 0
    substep = features[0, 11]

    old_known = jnp.where(reset, jnp.zeros((9, 3), dtype=jnp.int32), memory['known'])
    visible = features[:9, :3]
    visible_now = jnp.any(visible > 0, axis=1) & active
    known = jnp.where(visible_now[:, None], visible, old_known)
    old_turn = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['turn'])
    turn = old_turn + (active & (substep == 3)).astype(jnp.int32)
    positions = obs.positions.astype(jnp.int32)
    old_positions = jnp.where(reset, positions, memory['prev_positions'])
    changed = jnp.any(positions != old_positions, axis=1) & (~reset)
    old_cooldown = jnp.where(reset, jnp.zeros((9,), dtype=jnp.int32), memory['cooldown'])
    cooldown = jnp.maximum(old_cooldown - 1, 0)
    cooldown = jnp.where(changed, jnp.asarray(4, dtype=jnp.int32), cooldown)

    # At observation substeps, prefer a currently legal card whose color is
    # not in private memory. The legal mask also excludes the first card at
    # substep 1, so the two observations in a turn are distinct.
    unknown = ~jnp.any(known > 0, axis=1)
    observe_legal = legal[1:10]
    observe_preferred = observe_legal & unknown
    observe_has_preferred = jnp.any(observe_preferred)
    observe_idx = jnp.where(
        observe_has_preferred,
        _legal_first(observe_preferred),
        _legal_first(observe_legal),
    )
    observe_action = observe_idx + 1

    # If eight card colors are known, infer the ninth from the fixed 3x3
    # multiset. This is useful both for a correct hint placement and for
    # recognizing a solved board without any communication convention.
    known_rows = jnp.any(known > 0, axis=1)
    counts = known.sum(axis=0)
    missing = 3 - counts
    infer = (known_rows.sum() == 8) & jnp.all(missing >= 0) & (missing.sum() == 1)
    inferred_color = jnp.where(
        (~known_rows)[:, None] & infer,
        (missing > 0)[None, :],
        known > 0,
    ).astype(jnp.int32)

    # Score every legal relocation by same-color edges among cards whose
    # colors are known locally. Once all colors are available, a relocation
    # producing three connected color groups receives top priority. A small
    # distance tie-break keeps early, under-informed moves near the board.
    move_idx = jnp.arange(729, dtype=jnp.int32)
    move_card = move_idx // 81
    move_cell = move_idx % 81
    move_x = move_cell // 9
    move_y = move_cell % 9
    destination = jnp.stack((move_x, move_y), axis=1)
    card_ids = jnp.arange(9, dtype=jnp.int32)[None, :]
    moved = card_ids == move_card[:, None]
    new_positions = jnp.where(moved[:, :, None], destination[:, None, :], positions[None, :, :])
    delta = jnp.abs(new_positions[:, :, None, :] - new_positions[:, None, :, :]).sum(axis=-1)
    board_edges = delta == 1
    color_edges = (inferred_color @ inferred_color.T) > 0
    same_edges = (board_edges & color_edges[None, :, :]).sum(axis=(1, 2))
    same_color = inferred_color[:, None, :] * inferred_color[None, :, :]
    candidate_color_edges = (board_edges[:, :, :, None] * same_color[None, :, :, :]).sum(axis=(1, 2)) // 2
    candidate_solved = infer & jnp.all(candidate_color_edges >= 2, axis=1)
    travel = jnp.abs(destination - positions[move_card]).sum(axis=1)
    move_score = (candidate_solved.astype(jnp.int32) * 100000
                  + same_edges.astype(jnp.int32) * 100
                  - travel - cooldown[move_card] * 30)
    move_legal = legal[10:739]
    move_choice = jnp.argmax(jnp.where(move_legal, move_score, -1000000)).astype(jnp.int32) + 10

    current_edges = obs.adjacency_matrix[:9, :9].astype(jnp.int32)
    current_edges = current_edges * (1 - jnp.eye(9, dtype=jnp.int32))
    current_color_edges = current_edges[:, :, None] * same_color
    current_color_edge_counts = current_color_edges.sum(axis=(0, 1)) // 2
    current_solved = infer & jnp.all(current_color_edge_counts >= 2)
    current_color_degree = (current_edges * color_edges).sum(axis=1)

    # Reveal each still-hidden hint before placing. Once all are public,
    # choose a legal placement whose hint set contains a privately known
    # compatible card, preferring lower stable IDs for deterministic behavior.
    hint_sets = features[9:13, 3:6]
    hint_revealed = jnp.any(hint_sets > 0, axis=1)
    reveal_legal = legal[739:743]
    has_reveal = jnp.any(reveal_legal)
    reveal_action = _legal_first(reveal_legal) + 739

    hint_ids = jnp.repeat(jnp.arange(4, dtype=jnp.int32), 9)
    card_ids_flat = jnp.tile(jnp.arange(9, dtype=jnp.int32), 4)
    target_pos = positions[card_ids_flat]
    placement_actions = 743 + hint_ids * 81 + target_pos[:, 0] * 9 + target_pos[:, 1]
    placement_legal = legal[placement_actions]
    target_unlocked = features[card_ids_flat, 8] == 0
    hint_available = hint_revealed[hint_ids] & (features[9 + hint_ids, 8] == 0)
    compatible = (hint_sets[hint_ids] * inferred_color[card_ids_flat]).sum(axis=1) > 0
    placement_good = placement_legal & target_unlocked & hint_available & compatible
    target_degree = current_color_degree[card_ids_flat]
    placement_score = jnp.where(
        placement_good,
        target_degree * 1000 - hint_ids * 10 - card_ids_flat,
        -1000000,
    )
    placement_choice = placement_actions[jnp.argmax(placement_score)].astype(jnp.int32)
    any_good = jnp.any(placement_good)
    placement_fallback = _legal_first(legal)
    place_action = jnp.where(any_good, placement_choice, placement_fallback)
    hint_action = jnp.where(has_reveal, reveal_action, place_action)

    stage_action = jnp.where((substep == 0) | (substep == 1), observe_action, move_choice)
    stage_action = jnp.where(substep == 3, hint_action, stage_action)
    end_allowed = active & (substep == 0) & current_solved & legal[1067]
    stage_action = jnp.where(end_allowed, jnp.asarray(1067, dtype=jnp.int32), stage_action)
    action = jnp.where(active, stage_action, jnp.asarray(0, dtype=jnp.int32))
    return action.astype(jnp.int32), {
        'known': known,
        'turn': turn,
        'cooldown': cooldown,
        'prev_positions': positions,
    }, {}
