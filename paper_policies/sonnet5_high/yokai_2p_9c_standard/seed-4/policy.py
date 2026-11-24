"""Heuristic yokai policy: cluster known-color cards, reveal-first hints,
end only once locally known evidence shows no adjacency conflicts.

All decisions use only this seat's own observation history (kept in private
memory) and publicly visible board state. No cross-seat convention is
assumed: the policy never relies on the partner interpreting a signal in a
specific way, it only ever takes locally-justifiable, board-improving
actions (or safe/neutral ones when uninformed).
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'known_color_cluster_v1',
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

NUM_CARDS = 9
GRID = 9
NUM_HINTS = 4

MOVE_WEIGHT = 10.0
SATISFIED_MOVE_WEIGHT = 2.0
RAW_WEIGHT = 1.0
NEG = -1.0e6


def init_agent():
    return {
        'known_color': jnp.full((NUM_CARDS,), -1, dtype=jnp.int8),
        'partner_seen': jnp.zeros((NUM_CARDS,), dtype=jnp.bool_),
        'my_turns': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_act(obs, reset, memory, rng):
    del rng

    node_features = obs.node_features
    positions = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions

    active = node_features[0, 10] > 0
    substep = node_features[0, 11].astype(jnp.int32)

    base_known = jnp.where(reset, jnp.full((NUM_CARDS,), -1, dtype=jnp.int8), memory['known_color'])
    base_partner_seen = jnp.where(reset, jnp.zeros((NUM_CARDS,), dtype=jnp.bool_), memory['partner_seen'])
    base_my_turns = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['my_turns'])

    card_rows = node_features[0:NUM_CARDS]
    color_onehot = card_rows[:, 0:3]
    visible_now = jnp.sum(color_onehot, axis=-1) > 0
    color_now = jnp.argmax(color_onehot, axis=-1).astype(jnp.int8)
    known_color = jnp.where(visible_now, color_now, base_known)

    partner_flag_now = card_rows[:, 9] > 0
    partner_seen = base_partner_seen | partner_flag_now

    # ---- deterministic elimination: if two colors already have all 3 of
    # their cards identified, every remaining unknown card must be the
    # third color (pure counting deduction, not a guess). ----
    counts = jnp.stack([jnp.sum((known_color == c).astype(jnp.int32)) for c in range(3)])
    saturated = counts == 3
    num_saturated = jnp.sum(saturated.astype(jnp.int32))
    remaining_color = jnp.argmin(saturated.astype(jnp.int32))  # the one non-saturated color
    infer = (num_saturated == 2) & (known_color == -1)
    known_color = jnp.where(infer, remaining_color.astype(jnp.int8), known_color)

    # ---- adjacency among current card positions ----
    diff = positions[:, None, :] - positions[None, :, :]
    dist = jnp.abs(diff).sum(-1)
    adj_matrix = dist == 1  # (9,9) symmetric, no self-loops (dist 0 excluded)

    known_i = known_color.astype(jnp.int32)
    same_color_pairs = (known_i[:, None] == known_i[None, :]) & (known_i[:, None] >= 0)
    same_color_pairs = same_color_pairs & (~jnp.eye(NUM_CARDS, dtype=jnp.bool_))

    # ---- confidence check: any known color group with a missing adjacency ----
    problem_count = jnp.asarray(0, dtype=jnp.int32)
    color_needs_help = jnp.zeros((3,), dtype=jnp.bool_)
    for c in range(3):
        mask_c = known_i == c
        count_c = jnp.sum(mask_c.astype(jnp.int32))
        edges_c = jnp.sum((adj_matrix & same_color_pairs & mask_c[:, None] & mask_c[None, :]).astype(jnp.int32)) // 2
        required = jnp.where(count_c == 3, 2, jnp.where(count_c == 2, 1, 0))
        problem_c = (edges_c < required) & (count_c >= 2)
        problem_count = problem_count + problem_c.astype(jnp.int32)
        color_needs_help = color_needs_help.at[c].set(problem_c)

    # ---- deterministic (zero-risk) ambiguous-completion certainty check ----
    # Every turn forces a hint reveal/place, converting the highest-value
    # unrevealed hints (worth 5 each) into lower-value revealed/placed ones
    # (worth 2 / +-1), so ending sooner is valuable whenever it can be done
    # with full certainty. Counting-deduction (above) already resolves the
    # case of exactly one truly unknown card. The next-hardest fully
    # resolvable case is: exactly one color is fully known (saturated) and
    # the other two each have exactly 2 known members with their single
    # missing member being one of the two remaining unknown cards, in either
    # order. If, for BOTH candidate colors, EITHER unknown card would
    # complete that color's group into a connected triple (i.e. the color's
    # own known-pair edge, plus the candidate's adjacency to that pair,
    # reaches the 2-edges-needed threshold for every currently-unknown
    # card), then the true (unobserved) label assignment cannot matter and
    # connectivity is certain without ever learning which unknown card is
    # which color.
    unknown_mask = known_i == -1
    unknown_count = jnp.sum(unknown_mask.astype(jnp.int32))
    ambiguous_shape = (unknown_count == 2) & (num_saturated == 1)

    color_ok = jnp.zeros((3,), dtype=jnp.bool_)
    for c in range(3):
        mask_c = known_i == c
        count_c = jnp.sum(mask_c.astype(jnp.int32))
        edge_exists_c = jnp.any(adj_matrix & mask_c[:, None] & mask_c[None, :])
        edges_c_full = jnp.sum((adj_matrix & mask_c[:, None] & mask_c[None, :]).astype(jnp.int32)) // 2
        connect_count_c = jnp.sum((adj_matrix & mask_c[None, :]).astype(jnp.int32), axis=1)  # (9,)
        valid_as_third = (edge_exists_c.astype(jnp.int32) + connect_count_c) >= 2  # (9,)
        pair_ok_c = jnp.all(jnp.where(unknown_mask, valid_as_third, True))
        ok_c = jnp.where(count_c == 3, edges_c_full >= 2, jnp.where(count_c == 2, pair_ok_c, False))
        color_ok = color_ok.at[c].set(ok_c)
    ambiguous_certain = ambiguous_shape & jnp.all(color_ok)

    # ---- observe scoring (actions 1..9) ----
    unknown = known_color == -1
    obs_score = jnp.where(unknown, 100.0, -50.0) + jnp.where(unknown & (~partner_seen), 20.0, 0.0)

    # ---- move scoring (actions 10..738, block of 9*81) ----
    cur_same_neighbors = jnp.sum((adj_matrix & same_color_pairs).astype(jnp.int32), axis=1)

    cell_x = jnp.repeat(jnp.arange(GRID, dtype=jnp.int32), GRID)
    cell_y = jnp.tile(jnp.arange(GRID, dtype=jnp.int32), GRID)
    cell_xy = jnp.stack([cell_x, cell_y], axis=-1)  # (81,2)

    dest_diff = cell_xy[:, None, :] - positions[None, :, :]
    dest_dist = jnp.abs(dest_diff).sum(-1)
    dest_adj = (dest_dist == 1).astype(jnp.float32)  # (81,9)

    sm = same_color_pairs.astype(jnp.float32)  # (9,9)
    neighbor_count_new = dest_adj @ sm  # (81,9): [k, i]

    delta = neighbor_count_new - cur_same_neighbors[None, :].astype(jnp.float32)  # (81,9)

    # Prioritize cards whose own color group still has an unmet adjacency
    # requirement over cards whose group already satisfies it: with only one
    # forced move per turn, spending it on a redundant extra same-color edge
    # for an already-connected group instead of the first/second edge a
    # still-broken group needs is a wasted move given the tight move budget.
    card_needs_help = color_needs_help[jnp.clip(known_i, 0, 2)]  # (9,)
    per_card_move_weight = jnp.where(card_needs_help, MOVE_WEIGHT, SATISFIED_MOVE_WEIGHT)

    # Secondary, low-weight tiebreaker: among moves that don't help a known
    # color, avoid needlessly shrinking total (any-color) adjacency, since an
    # unknown card's current neighbor could turn out to already be a correct
    # same-color pairing we should not carelessly break.
    cur_raw_neighbors = jnp.sum(adj_matrix.astype(jnp.float32), axis=1)  # (9,)
    total_adj_new = dest_adj.sum(axis=1)  # (81,)
    raw_neighbor_new = total_adj_new[:, None] - dest_adj  # (81,9): exclude self-position term
    raw_delta = raw_neighbor_new - cur_raw_neighbors[None, :]  # (81,9)

    move_score = ((delta.T * per_card_move_weight[:, None]) + raw_delta.T * RAW_WEIGHT).reshape(-1)  # (9*81,) matches i*81+k layout

    # ---- reveal scoring (actions 739..742) ----
    reveal_score = jnp.full((NUM_HINTS,), 100.0, dtype=jnp.float32)

    # ---- place scoring (actions 743..1066, block of 4*81) ----
    hint_rows = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS]
    hint_set = hint_rows[:, 3:6] > 0  # (4,3) bool

    flat_cells = positions[:, 0] * GRID + positions[:, 1]
    occ_card_id = jnp.full((GRID * GRID,), -1, dtype=jnp.int32).at[flat_cells].set(jnp.arange(NUM_CARDS, dtype=jnp.int32))
    occupied = occ_card_id >= 0
    known_at_cell = known_i[jnp.clip(occ_card_id, 0, NUM_CARDS - 1)]
    known_at_cell = jnp.where(occupied, known_at_cell, -1)

    clipped_known = jnp.clip(known_at_cell, 0, 2)
    membership = hint_set[:, clipped_known]  # (4, 81) via broadcasting hint dim x cell dim
    known_valid = known_at_cell >= 0
    # When forced to place on a still-unknown card (no known-color cell is a
    # safe/better option), prefer locking a low-degree (peripheral) card over
    # a high-degree (hub/bridge) one: locking permanently removes it from
    # future moves, and a well-connected hub card is more likely to still be
    # needed to help complete a color group later.
    degree_by_card = cur_raw_neighbors  # (9,), computed above for move scoring
    degree_at_cell = jnp.where(occupied, degree_by_card[jnp.clip(occ_card_id, 0, NUM_CARDS - 1)], 0.0)
    unknown_fallback = -10.0 - 3.0 * degree_at_cell
    place_val = jnp.where(known_valid[None, :], jnp.where(membership, 60.0, -1000.0), unknown_fallback[None, :])
    place_score = place_val.reshape(-1)  # (4*81,) matches h*81+k layout

    # ---- end-game scoring (action 1067) ----
    # Every turn forces a hint reveal/place regardless of our choice, and the
    # terminal score formula (5*unrevealed + 2*revealed + correct - incorrect
    # if connected, else 0) makes unrevealed hints far more valuable than
    # revealed or placed ones. So this seat only ever volunteers to end once
    # it has full deterministic certainty of connectivity from its own
    # knowledge (direct observation, exact counting-deduction, or the
    # zero-risk ambiguous-completion check above); it never gambles on an
    # under-informed guess, but it does end as soon as certainty is reached
    # rather than waiting for exhaustive full disclosure of every card.
    all_known = jnp.all(known_color != -1)
    can_end = ((problem_count == 0) & all_known) | ambiguous_certain
    end_score = jnp.where(can_end, jnp.asarray(500.0), jnp.asarray(NEG))

    score = jnp.concatenate([
        jnp.asarray([NEG], dtype=jnp.float32),   # no-op (0)
        obs_score.astype(jnp.float32),           # 1..9
        move_score.astype(jnp.float32),          # 10..738
        reveal_score,                            # 739..742
        place_score.astype(jnp.float32),         # 743..1066
        jnp.reshape(end_score, (1,)).astype(jnp.float32),  # 1067
    ])

    legal_score = jnp.where(legal, score, NEG * 10.0)
    chosen = jnp.argmax(legal_score).astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    action = jnp.where(active, chosen, zero)

    turn_completing = active & (substep == 3)
    my_turns_next = base_my_turns + turn_completing.astype(jnp.int32)

    next_memory = {
        'known_color': known_color,
        'partner_seen': partner_seen,
        'my_turns': my_turns_next,
    }

    trace = {
        'action': action,
        'problem_count': problem_count,
        'my_turns': base_my_turns,
        'substep': substep,
        'unknown_count': unknown_count,
        'num_saturated': num_saturated,
        'ambiguous_shape': ambiguous_shape,
        'ambiguous_certain': ambiguous_certain,
    }

    return action, next_memory, trace
