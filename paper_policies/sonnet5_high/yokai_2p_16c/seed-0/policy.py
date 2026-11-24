"""Cooperative Yokai policy.

Strategy: privately track each card's true color from this seat's own
observations (perfect memory persists them while active). If at least three
of the four colors are fully identified (4 cards each), the fourth color's
membership is deduced by elimination without further observation. Moves
greedily connect pairs of cards known (directly or by elimination) to share
a color. Hints are placed in preference to revealing new ones (this keeps
already-revealed hints from sitting idle while preserving the higher-value
unrevealed count), and are placed on a card whose true color is confirmed to
lie in the hint's color set whenever one is available. The game is ended
only once elimination gives full-board color certainty and that certainty
shows every color already forming one connected group.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'yokai_deduce_cluster',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 4,
    'deck_size': 16,
    'grid_size': 10,
    'num_hints': 7,
    'num_nodes': 23,
    'node_features': 15,
    'num_actions': 2325,
    'policy_structure': 'shared_local',
}

NUM_CARDS = 16
NUM_COLORS = 4
NUM_HINTS = 7
CARDS_PER_COLOR = 4

HINT_SIZES_LIST = [1, 1, 2, 2, 2, 3, 3]
DELTAS_LIST = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def init_agent():
    return jnp.full((NUM_CARDS,), -1, dtype=jnp.int8)


def _connected_by_color(colors, adjacency16):
    same = colors[:, None] == colors[None, :]
    adj = adjacency16 > 0
    reach = (same & adj) | jnp.eye(NUM_CARDS, dtype=bool)
    for _ in range(4):
        reach = reach | (jnp.matmul(reach.astype(jnp.int32), reach.astype(jnp.int32)) > 0)
    ok = jnp.asarray(True)
    for c in range(NUM_COLORS):
        mask_c = colors == c
        pair_mask = mask_c[:, None] & mask_c[None, :]
        pair_ok = jnp.all(jnp.where(pair_mask, reach, True))
        ok = ok & pair_ok
    return ok


def agent_act(obs, reset, memory, rng):
    del rng
    known_color = jnp.where(reset, init_agent(), memory)

    node_features = obs.node_features
    card_feats = node_features[:NUM_CARDS]
    hint_feats = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS]

    own_color_onehot = card_feats[:, 0:4].astype(jnp.int32)
    hot_mask = jnp.sum(own_color_onehot, axis=-1) > 0
    color_from_obs = jnp.argmax(own_color_onehot, axis=-1).astype(jnp.int8)
    known_color = jnp.where(hot_mask, color_from_obs, known_color)

    locked = card_feats[:, 10] != 0
    unlocked = jnp.logical_not(locked)
    positions = obs.positions.astype(jnp.int32)

    active = node_features[0, 12] > 0
    substep = node_features[0, 13].astype(jnp.int32)

    legal_actions = obs.legal_actions

    # --- Deduce the fourth color by elimination once three are fully known.
    counts = jnp.sum(known_color[:, None].astype(jnp.int32) == jnp.arange(NUM_COLORS)[None, :], axis=0)
    full_mask = counts == CARDS_PER_COLOR
    num_full = jnp.sum(full_mask.astype(jnp.int32))
    can_deduce = num_full >= 3
    missing_color = jnp.argmax(jnp.logical_not(full_mask).astype(jnp.int32))
    effective_known = jnp.where(
        can_deduce,
        jnp.where(known_color == -1, missing_color, known_color.astype(jnp.int32)),
        known_color.astype(jnp.int32),
    )

    adjacency16 = obs.adjacency_matrix[:NUM_CARDS, :NUM_CARDS]
    eye16 = jnp.eye(NUM_CARDS, dtype=bool)
    connectivity_ok = _connected_by_color(effective_known, adjacency16)
    certain_end_ok = can_deduce & connectivity_ok

    # When exactly two colors are fully known and exactly two cards remain
    # unknown, those two colors must each be missing exactly one member and
    # the two unknown cards must be exactly those two members -- just not
    # which is which. Connectivity can still be certified without resolving
    # that ambiguity if it holds under both possible assignments.
    unknown_count = jnp.sum(known_color == -1)
    ambiguous_case = (num_full == 2) & (unknown_count == 2)
    missing_rank = jnp.argsort(jnp.where(jnp.logical_not(full_mask), jnp.arange(NUM_COLORS), 100))
    missing_a, missing_b = missing_rank[0], missing_rank[1]
    unknown_rank = jnp.argsort(jnp.where(known_color == -1, jnp.arange(NUM_CARDS), 100))
    card_u1, card_u2 = unknown_rank[0], unknown_rank[1]
    base_known = known_color.astype(jnp.int32)
    hyp_a = base_known.at[card_u1].set(missing_a).at[card_u2].set(missing_b)
    hyp_b = base_known.at[card_u1].set(missing_b).at[card_u2].set(missing_a)
    ambiguous_ok = (
        ambiguous_case
        & _connected_by_color(hyp_a, adjacency16)
        & _connected_by_color(hyp_b, adjacency16)
    )

    end_condition = legal_actions[2324] & (certain_end_ok | ambiguous_ok)

    same_eff = (effective_known[:, None] == effective_known[None, :]) & (effective_known[:, None] != -1)

    # Per-card "outstanding" fragmentation score: how many other known
    # same-color cards are not yet in this card's connected component.
    # Used to prioritize merges (below) and, here, to bias observation
    # toward the single most-fragmented color's known members.
    known_eff_mask = effective_known != -1
    same_c_eff = (effective_known[:, None] == effective_known[None, :]) & known_eff_mask[:, None] & known_eff_mask[None, :]
    reach_c = (adjacency16 > 0) & same_c_eff | eye16
    for _ in range(4):
        reach_c = reach_c | (jnp.matmul(reach_c.astype(jnp.int32), reach_c.astype(jnp.int32)) > 0)
    component_size = jnp.sum(reach_c & same_c_eff, axis=1)
    color_known_count = jnp.sum(effective_known[:, None] == jnp.arange(NUM_COLORS)[None, :], axis=0)
    outstanding = color_known_count[jnp.clip(effective_known, 0, 3)] - component_size

    # --- Observe target: an actually-unknown, unlocked card (skip once
    # elimination already certifies every card's color). Prefer a card
    # adjacent to a known member of the single most-fragmented color (a
    # match there directly attacks the bottleneck); otherwise lowest-ID
    # first for broad, unbiased board coverage (helps elimination too).
    unknown_unlocked = (known_color == -1) & unlocked & jnp.logical_not(can_deduce)
    observe_legal_mask = legal_actions[1:17]
    bottleneck_card = jnp.argmax(jnp.where(known_eff_mask, outstanding, -1))
    has_bottleneck = jnp.any(known_eff_mask & (outstanding > 0))
    adjacent_to_bottleneck = (adjacency16 > 0)[:, bottleneck_card] & jnp.logical_not(eye16)[:, bottleneck_card]
    bottleneck_legal = unknown_unlocked & adjacent_to_bottleneck & has_bottleneck & observe_legal_mask
    plain_legal = unknown_unlocked & observe_legal_mask
    has_bottleneck_target = jnp.any(bottleneck_legal)
    has_plain_target = jnp.any(plain_legal)
    bottleneck_target_card = jnp.argmax(bottleneck_legal.astype(jnp.int32))
    plain_card = jnp.argmax(plain_legal.astype(jnp.int32))
    fallback_observe = jnp.argmax(observe_legal_mask.astype(jnp.int32))
    observe_card = jnp.where(
        has_bottleneck_target, bottleneck_target_card,
        jnp.where(has_plain_target, plain_card, fallback_observe),
    )
    observe_action = 1 + observe_card

    action0 = jnp.where(end_condition, 2324, observe_action)
    action1 = observe_action

    # --- Move: connect a pair of cards known (directly or by elimination)
    # to share a color but not yet adjacent. Never blindly relocate a card
    # that is already anchoring an established same-color adjacency.
    not_adj16 = adjacency16 == 0

    # A locked card can still serve as a fixed anchor (its own-color
    # partner moves to it); only the mover needs to be unlocked, which the
    # environment's own legality mask already enforces below.
    candidate_mask = same_eff & not_adj16 & jnp.logical_not(eye16) & (unlocked[:, None] | unlocked[None, :])
    candidate_mask = jnp.triu(candidate_mask, k=1)
    has_candidate = jnp.any(candidate_mask)

    # For every (mover, anchor) ordered pair, find whether some empty
    # neighbor cell of the anchor is a legal destination for the mover, and
    # if so which one (lowest-index direction).
    deltas = jnp.asarray(DELTAS_LIST, dtype=jnp.int32)
    neighbors_of_anchor = positions[:, None, :] + deltas[None, :, :]
    valid_nb = jnp.all((neighbors_of_anchor >= 0) & (neighbors_of_anchor <= 9), axis=-1)
    clipped_nb = jnp.clip(neighbors_of_anchor, 0, 9)
    base_cell_id = clipped_nb[:, :, 0] * 10 + clipped_nb[:, :, 1]
    action_ids_tensor = 17 + jnp.arange(NUM_CARDS)[:, None, None] * 100 + base_cell_id[None, :, :]
    legal_tensor = legal_actions[action_ids_tensor] & valid_nb[None, :, :]
    has_direction = jnp.any(legal_tensor, axis=-1)
    best_delta_idx = jnp.argmax(legal_tensor.astype(jnp.int32), axis=-1)
    best_action = jnp.take_along_axis(action_ids_tensor, best_delta_idx[:, :, None], axis=-1)[:, :, 0]

    # Among all candidate pairs, use any that is geometrically achievable
    # (not just the first-indexed one, which might be locally boxed in), and
    # prefer the pair whose color currently has the most outstanding known
    # members still outside its component: since every color must end up
    # connected for any reward, spreading effort to the neediest color
    # first is better than repeatedly polishing one that is already ahead.
    feasible_pair = candidate_mask & (has_direction | has_direction.T)
    has_feasible = jnp.any(feasible_pair)

    neediness = outstanding[:, None] + outstanding[None, :]
    priority_masked = jnp.where(feasible_pair, neediness, -10_000_000)
    idx_flat = jnp.argmax(priority_masked.reshape(-1))
    i_card = idx_flat // NUM_CARDS
    j_card = idx_flat % NUM_CARDS

    # Prefer moving whichever of the pair is more fragmented (higher
    # outstanding) onto the more established side, among the feasible
    # directions.
    pref_j_mover = outstanding[j_card] >= outstanding[i_card]
    dir_j_ok = has_direction[j_card, i_card]
    dir_i_ok = has_direction[i_card, j_card]
    use_j_as_mover = jnp.where(dir_j_ok & dir_i_ok, pref_j_mover, dir_j_ok)
    final_mover = jnp.where(use_j_as_mover, j_card, i_card)
    final_anchor = jnp.where(use_j_as_mover, i_card, j_card)
    pair_action = best_action[final_mover, final_anchor]

    # Fallback (no achievable known pair to connect): any legal move, but
    # avoid disturbing a color that is already fully known and merged.
    completed = known_eff_mask & (outstanding == 0) & (color_known_count[jnp.clip(effective_known, 0, 3)] == 4)
    move_legal_mask = legal_actions[17:1617]
    move_legal_2d = move_legal_mask.reshape(NUM_CARDS, 100)
    uncompleted_moves = move_legal_2d & jnp.logical_not(completed[:, None])
    has_uncompleted_fallback = jnp.any(uncompleted_moves)
    uncompleted_fallback = 17 + jnp.argmax(uncompleted_moves.reshape(-1).astype(jnp.int32))
    any_fallback = 17 + jnp.argmax(move_legal_mask.astype(jnp.int32))
    fallback_move = jnp.where(has_uncompleted_fallback, uncompleted_fallback, any_fallback)

    action2 = jnp.where(has_feasible, pair_action, fallback_move)

    # --- Hint: prefer placing an already-revealed hint (preserves the more
    # valuable unrevealed count); reveal only when nothing is placeable.
    hint_colorsets = hint_feats[:, 4:8].astype(jnp.int32)
    reveal_legal = legal_actions[1617:1624]
    place_legal_full = legal_actions[1624:2324].reshape(NUM_HINTS, 10, 10)
    exists_place_h = jnp.any(place_legal_full, axis=(1, 2))
    has_any_place = jnp.any(exists_place_h)
    chosen_h = jnp.argmax(exists_place_h.astype(jnp.int32))

    hint_sizes = jnp.asarray(HINT_SIZES_LIST, dtype=jnp.int32)
    reveal_priority = jnp.where(reveal_legal, hint_sizes * 10 - jnp.arange(NUM_HINTS), -1_000_000)
    reveal_choice = jnp.argmax(reveal_priority)
    reveal_action = 1617 + reveal_choice
    has_reveal = jnp.any(reveal_legal)

    # Placement target priority: a known color matching the hint's set
    # (guaranteed correct) beats any other known color (guaranteed wrong,
    # but preserves an unlocked card's chance of being observed/deduced
    # later) which beats an unknown card (locks away potential knowledge
    # that clustering and future deduction still need).
    hint_set_for_h = hint_colorsets[chosen_h]
    card_color_clipped = jnp.clip(effective_known, 0, 3)
    known_unlocked = (effective_known != -1) & unlocked
    matches_known = known_unlocked & (hint_set_for_h[card_color_clipped] == 1)
    cell_action_ids_for_h = 1624 + chosen_h * 100 + positions[:, 0] * 10 + positions[:, 1]
    legal_for_h = legal_actions[cell_action_ids_for_h]
    placement_priority = jnp.where(matches_known, 2, jnp.where(known_unlocked, 1, 0))
    placement_priority = jnp.where(legal_for_h, placement_priority, -1)
    best_place_card = jnp.argmax(placement_priority)
    place_action = cell_action_ids_for_h[best_place_card]
    action3 = jnp.where(has_any_place, place_action, jnp.where(has_reveal, reveal_action, 1617))

    action = jnp.select(
        [substep == 0, substep == 1, substep == 2, substep == 3],
        [action0, action1, action2, action3],
        default=0,
    )
    action = jnp.where(active, action, 0).astype(jnp.int32)

    next_memory = known_color.astype(jnp.int8)
    trace = {
        'known_count': jnp.sum(known_color != -1).astype(jnp.int32),
        'can_deduce': can_deduce.astype(jnp.int32),
        'end_condition': end_condition.astype(jnp.int32),
        'substep': substep,
        'has_candidate': has_candidate.astype(jnp.int32),
        'action': action,
        'ambiguous_case': ambiguous_case.astype(jnp.int32),
        'ambiguous_ok': ambiguous_ok.astype(jnp.int32),
        'num_full': num_full.astype(jnp.int32),
    }
    return action, next_memory, trace
