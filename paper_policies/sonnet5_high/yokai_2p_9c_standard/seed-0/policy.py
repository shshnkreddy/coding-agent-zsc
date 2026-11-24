"""Self-sufficient, convention-robust heuristic for two-player 9-card Yokai.

Design summary (see ledger A01 for rationale):
  - Each seat keeps its own permanent memory of colors it has personally
    observed (the environment clears "own color" observation features at
    every turn boundary, so this must be remembered locally).
  - Colors are inferred beyond direct observation using the fixed 3-per-color
    deck: if at most one color still has undiscovered copies, every
    still-unknown card must be that color.
  - Moves are scored by how much they increase orthogonal adjacency between
    cards *this seat believes* share a color, and decrease adjacency between
    cards it believes differ; cards of unknown color contribute zero, so a
    seat never fights a partner's placement of cards it knows nothing about.
  - If the seat already believes the whole board is connected, move choices
    get a large bonus for preserving that state (using only legal moves),
    to survive until it can safely end the game.
  - Hint reveals are preferred over placements whenever a choice exists,
    since revealing costs less potential score than placing, and delaying
    placement lets more information accumulate before a card gets locked.
  - The episode is ended (substep 0, action 1067) only when this seat is
    fully certain (via direct observation + elimination) that every color
    is connected -- never a probabilistic gamble.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
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

_NUM_CARDS = 9
_NUM_COLORS = 3
_NUM_HINTS = 4
_GRID = 9

_NEG_INF = -1e9


def init_agent():
    return jnp.full((_NUM_CARDS,), -1, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    node_features = obs.node_features.astype(jnp.int32)  # [13,13]
    positions = obs.positions.astype(jnp.int32)  # [9,2]
    legal = obs.legal_actions  # [1068] bool

    reset = jnp.asarray(reset, dtype=jnp.bool_)
    init_memory = jnp.full((_NUM_CARDS,), -1, dtype=jnp.int32)
    known_color_prev = jnp.where(reset, init_memory, memory)

    # ---- update permanent color memory from this call's observation ----
    card_color_onehot = node_features[0:_NUM_CARDS, 0:_NUM_COLORS]  # [9,3]
    has_color = jnp.sum(card_color_onehot, axis=1) > 0  # [9]
    raw_color = jnp.argmax(card_color_onehot, axis=1).astype(jnp.int32)
    known_color = jnp.where(has_color, raw_color, known_color_prev)

    # ---- elimination inference: fixed 3-per-color deck ----
    counts = jnp.stack(
        [jnp.sum(known_color == c) for c in range(_NUM_COLORS)]
    ).astype(jnp.int32)
    deficiency = _NUM_COLORS - counts  # copies of each color not yet located
    deficiency = jnp.maximum(deficiency, 0)
    num_deficient = jnp.sum(deficiency > 0)
    deficient_color = jnp.argmax(deficiency).astype(jnp.int32)
    believed_color = jnp.where(
        known_color != -1,
        known_color,
        jnp.where(num_deficient <= 1, deficient_color, -1),
    )
    fully_known = jnp.all(believed_color != -1)

    # ---- pairwise adjacency / same-known-color structure (current board) ----
    diff = positions[:, None, :] - positions[None, :, :]  # [9,9,2]
    dist = jnp.abs(diff[..., 0]) + jnp.abs(diff[..., 1])  # [9,9]
    adjacent = dist == 1  # [9,9] bool
    same_known = (believed_color[:, None] == believed_color[None, :]) & (
        believed_color[:, None] != -1
    )
    edge_matrix = adjacent & same_known  # [9,9] bool, symmetric, no self-loops

    def color_edge_count(c):
        cmask = believed_color == c
        sub = edge_matrix & cmask[:, None] & cmask[None, :]
        return jnp.sum(sub) // 2

    color_edges = jnp.stack([color_edge_count(c) for c in range(_NUM_COLORS)])
    all_connected = jnp.all(color_edges >= 2)  # trusted only when fully_known

    pre_i_edges = jnp.sum(edge_matrix, axis=1)  # [9]
    other_pair_adjacent = jnp.clip(2 - pre_i_edges, 0, 2)  # [9]

    # Edges, among currently known cards, between the *other* known members of
    # card i's own believed color (i.e. excluding edges that involve i
    # itself). This is exact (not the "assume exactly 2 total" shortcut used
    # for the preserve-bonus above) so it is valid even when a color group is
    # only partially known.
    safe_believed = jnp.clip(believed_color, 0, _NUM_COLORS - 1)
    other_pair_adjacent_general = jnp.where(
        believed_color != -1, color_edges[safe_believed] - pre_i_edges, 0
    )

    # Current (pre-move) contribution of every card to the pairwise score, so
    # that moving a card away from an already-good position is charged the
    # opportunity cost of breaking that pair, not just credited for any new
    # pairs it forms. Without this term, argmax cannot tell "leave a correct
    # pair alone" apart from "break a correct pair for a neutral spot".
    diff_known = (
        (believed_color[:, None] != believed_color[None, :])
        & (believed_color[:, None] != -1)
        & (believed_color[None, :] != -1)
    )
    current_pair_value = jnp.where(
        adjacent & same_known, 1.0, jnp.where(adjacent & diff_known, -1.0, 0.0)
    )
    current_score_per_card = jnp.sum(current_pair_value, axis=1)  # [9]

    card_locked = node_features[0:_NUM_CARDS, 8] == 1  # [9]

    rng_move, rng_place, rng_obs = jax.random.split(rng, 3)

    # ---- score every (card, x, y) move candidate ----
    idx_range = jnp.arange(_GRID)
    ii, xi, yi = jnp.meshgrid(
        jnp.arange(_NUM_CARDS), idx_range, idx_range, indexing='ij'
    )
    ii_f = ii.reshape(-1)
    xi_f = xi.reshape(-1)
    yi_f = yi.reshape(-1)
    n_move = ii_f.shape[0]

    bc_i = believed_color[ii_f]

    move_score = jnp.zeros((n_move,), dtype=jnp.float32)
    match_edges = jnp.zeros((n_move,), dtype=jnp.float32)
    for j in range(_NUM_CARDS):
        valid = (ii_f != j) & (bc_i != -1) & (believed_color[j] != -1)
        same = bc_i == believed_color[j]
        dxj = jnp.abs(xi_f - positions[j, 0])
        dyj = jnp.abs(yi_f - positions[j, 1])
        adj_j = (dxj + dyj) == 1
        contrib = jnp.where(valid & adj_j, jnp.where(same, 1.0, -1.0), 0.0)
        move_score = move_score + contrib
        match_edges = match_edges + jnp.where(valid & adj_j & same, 1.0, 0.0)

    group_edges_post = match_edges + other_pair_adjacent[ii_f].astype(jnp.float32)
    connected_post = group_edges_post >= 2.0
    preserve_bonus = jnp.where(
        fully_known & all_connected & connected_post, 1000.0, 0.0
    )
    # Reward permanently finishing a color group (using the exact, not the
    # "assume 2 total edges", count) as soon as all three of its members are
    # known -- once a whole group is confirmed connected it will never need
    # to move again, which banks progress early rather than leaving several
    # groups half-finished until information arrives too late to fix them.
    group_edges_post_general = match_edges + other_pair_adjacent_general[
        ii_f
    ].astype(jnp.float32)
    completion_bonus = jnp.where(group_edges_post_general >= 2.0, 10.0, 0.0)
    # Charge the opportunity cost of leaving the card's current position.
    move_score = move_score - current_score_per_card[ii_f]
    move_score = move_score + completion_bonus
    move_score = move_score + preserve_bonus

    dist_moved = (
        jnp.abs(xi_f - positions[ii_f, 0]) + jnp.abs(yi_f - positions[ii_f, 1])
    ).astype(jnp.float32)
    move_score = move_score - 0.001 * dist_moved
    move_score = move_score + jax.random.uniform(rng_move, (n_move,)) * 1e-4

    move_action_ids = 10 + ii_f * 81 + xi_f * 9 + yi_f
    legal_move = legal[move_action_ids]
    move_score = jnp.where(legal_move, move_score, _NEG_INF)
    best_move_idx = jnp.argmax(move_score)
    move_action = move_action_ids[best_move_idx]
    move_fallback = jnp.argmax(legal[10:739]) + 10
    move_action = jnp.where(
        move_score[best_move_idx] > _NEG_INF / 2, move_action, move_fallback
    )

    # ---- hint reveal / placement scoring ----
    hint_color_sets = node_features[9:13, 3:6] > 0  # [4,3] bool
    hint_revealed = jnp.any(hint_color_sets, axis=1)  # [4]
    hint_placed = node_features[9:13, 8] == 1  # [4]
    hint_unplaced_revealed = hint_revealed & (~hint_placed)
    set_size = jnp.sum(hint_color_sets, axis=1).astype(jnp.float32)

    settled_card = jnp.any(edge_matrix, axis=1)  # [9]

    hh, cc = jnp.meshgrid(
        jnp.arange(_NUM_HINTS), jnp.arange(_NUM_CARDS), indexing='ij'
    )
    hh_f = hh.reshape(-1)
    cc_f = cc.reshape(-1)
    n_place = hh_f.shape[0]

    bc_c = believed_color[cc_f]
    safe_bc_c = jnp.clip(bc_c, 0, _NUM_COLORS - 1)
    in_set = hint_color_sets[hh_f, safe_bc_c]
    known_correct = (bc_c != -1) & in_set
    known_wrong = (bc_c != -1) & (~in_set)

    place_desir = jnp.where(
        known_correct,
        10.0,
        jnp.where(known_wrong, -10.0, 0.5 * set_size[hh_f]),
    )
    place_desir = place_desir + jnp.where(settled_card[cc_f], 1.0, 0.0)
    place_desir = place_desir - 0.001 * cc_f.astype(jnp.float32)
    place_desir = place_desir - 0.0001 * hh_f.astype(jnp.float32)
    place_desir = place_desir + jax.random.uniform(rng_place, (n_place,)) * 1e-4

    place_action_ids = 743 + hh_f * 81 + positions[cc_f, 0] * 9 + positions[cc_f, 1]
    legal_place_action = legal[place_action_ids]
    legal_place = (
        legal_place_action & hint_unplaced_revealed[hh_f] & (~card_locked[cc_f])
    )
    place_desir = jnp.where(legal_place, place_desir, _NEG_INF)
    best_place_idx = jnp.argmax(place_desir)
    place_action = place_action_ids[best_place_idx]
    place_fallback = jnp.argmax(legal[743:1067]) + 743
    place_action = jnp.where(
        place_desir[best_place_idx] > _NEG_INF / 2, place_action, place_fallback
    )

    reveal_action_ids = 739 + jnp.arange(_NUM_HINTS)
    legal_reveal = legal[reveal_action_ids]
    reveal_score = jnp.where(
        legal_reveal & (~hint_revealed),
        1.0 - 0.01 * jnp.arange(_NUM_HINTS, dtype=jnp.float32),
        _NEG_INF,
    )
    best_reveal_idx = jnp.argmax(reveal_score)
    reveal_action = reveal_action_ids[best_reveal_idx]
    any_unrevealed_legal = jnp.any(legal_reveal & (~hint_revealed))

    hint_action = jnp.where(any_unrevealed_legal, reveal_action, place_action)

    # ---- observation choice (substeps 0 and 1) ----
    # Prefer unknown cards that currently sit next to an already-known card
    # whose own color group is not yet complete: whatever the neighbor's
    # color turns out to be, that answer is immediately actionable (confirms
    # a pair to keep, or a mismatch to fix) and can help finish an
    # in-progress group, so it yields more decision-relevant information per
    # observation than an arbitrary unknown card far from everything we know
    # (a neighbor of an already-*complete* group carries little of that
    # extra value, since the complete group needs no further matches).
    group_incomplete = believed_color != -1
    incomplete_known = group_incomplete & (
        color_edges[safe_believed] < 2
    )
    adjacent_to_incomplete_known = jnp.any(
        adjacent & incomplete_known[None, :], axis=1
    )
    observe_action_ids = 1 + jnp.arange(_NUM_CARDS)
    legal_observe = legal[observe_action_ids]
    unknown_bonus = jnp.where(
        believed_color == -1,
        jnp.where(adjacent_to_incomplete_known, 1.5, 1.0),
        0.0,
    )
    obs_score = jnp.where(
        legal_observe,
        unknown_bonus
        - 0.001 * jnp.arange(_NUM_CARDS, dtype=jnp.float32)
        + jax.random.uniform(rng_obs, (_NUM_CARDS,)) * 1e-4,
        _NEG_INF,
    )
    best_obs_idx = jnp.argmax(obs_score)
    observe_action = observe_action_ids[best_obs_idx]
    observe_fallback = jnp.argmax(legal[1:10]) + 1
    observe_action = jnp.where(
        obs_score[best_obs_idx] > _NEG_INF / 2, observe_action, observe_fallback
    )

    # ---- end-game decision (substep 0 only) ----
    end_legal = legal[1067]
    should_end = fully_known & all_connected & end_legal
    substep0_action = jnp.where(should_end, 1067, observe_action)

    active = node_features[0, 10] != 0
    substep = node_features[0, 11].astype(jnp.int32)

    active_action = jnp.select(
        [substep == 0, substep == 1, substep == 2],
        [substep0_action, observe_action, move_action],
        default=hint_action,
    ).astype(jnp.int32)

    action = jnp.where(active, active_action, 0).astype(jnp.int32)

    next_memory = known_color

    return action, next_memory, {}
