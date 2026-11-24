"""Coordination-aware Yokai policy.

Strategy (convention-free, relies only on public game state and each seat's
own private, perfect-memory observations -- never a shared secret protocol):

1. Deduce card colors from own observations plus set-elimination (3 cards per
   color): once the unresolved color-count is concentrated on a single color,
   every remaining unknown card must be that color.
2. On the forced move each turn, greedily move a card toward same-known-color
   adjacency (only a card's own edges change when it alone moves, so the
   score decomposes cleanly per candidate move).
3. End the game the moment this seat's own knowledge *certifies* that every
   color is fully connected. Ending before any hint is revealed is strictly
   the highest-value path (5 pts per unrevealed hint beats any hint-based
   score), so we never reveal/place hints if we can avoid it.
4. When forced to act at substep 3 (reveal or place), only prefer placing
   over revealing a fresh hint when the target card's color cluster is
   *both* provably correct for the hint (via set-elimination, not just
   direct observation) *and* already fully connected (edge_count >= 2 for
   every color the card could possibly be). Placing locks the card for the
   rest of the episode, which can permanently block that color's
   connectivity if the cluster isn't finished yet -- so an unsafe-to-lock
   guaranteed-correct card falls back to the same low-priority probability
   scoring as a genuinely ambiguous card, keeping reveal preferred until a
   lock is truly safe. This last-resort fallback (used only once no
   unrevealed hint remains) ranks options by the probability the card's
   still-ambiguous color falls inside the hint's revealed set.

The policy is stateless: with perfect memory, obs already exposes this
seat's full observation history whenever it is active, so there is nothing
useful to carry in private memory. Memory is a dummy scalar to satisfy the
fixed-shape PyTree contract.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
NUM_HINTS = 4
GRID = 9
NUM_ACTIONS = 1068

_NEG = -1e9


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    # (x, y) grid in the same flattening order as the action formula
    # (cell = x * GRID + y).
    grid_xy = jnp.stack(
        jnp.meshgrid(jnp.arange(GRID), jnp.arange(GRID), indexing='ij'), axis=-1
    ).reshape(GRID * GRID, 2)  # [81, 2]
    eye9 = jnp.eye(NUM_CARDS, dtype=bool)

    node_features = obs.node_features.astype(jnp.int32)  # [13, 13]
    positions = obs.positions.astype(jnp.int32)  # [9, 2]
    legal = obs.legal_actions  # [1068] bool

    own_onehot = node_features[:NUM_CARDS, 0:3]  # [9, 3]
    hint_colorset = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 3:6]  # [4, 3]

    known = jnp.sum(own_onehot, axis=-1) > 0  # [9]

    counts_known = jnp.sum(own_onehot, axis=0)  # [3]
    remaining = NUM_COLORS - counts_known  # [3]
    unknown_mask = ~known  # [9]
    num_unknown = jnp.sum(unknown_mask.astype(jnp.int32))
    num_colors_with_remaining = jnp.sum((remaining > 0).astype(jnp.int32))
    can_deduce_all_unknown = (num_colors_with_remaining == 1) & (num_unknown > 0)
    deduced_color = jnp.argmax(remaining)
    deduced_row = jax.nn.one_hot(deduced_color, NUM_COLORS, dtype=own_onehot.dtype)

    effective_onehot = jnp.where(
        (unknown_mask & can_deduce_all_unknown)[:, None],
        jnp.broadcast_to(deduced_row, (NUM_CARDS, NUM_COLORS)),
        own_onehot,
    )
    effective_known = known | (unknown_mask & can_deduce_all_unknown)  # [9]
    effective_color_idx = jnp.argmax(effective_onehot, axis=-1)  # [9]
    full_deduction = jnp.all(effective_known)

    # Orthogonal adjacency between current card positions.
    diff = positions[:, None, :] - positions[None, :, :]  # [9, 9, 2]
    manhattan = jnp.sum(jnp.abs(diff), axis=-1)  # [9, 9]
    adjacent = (manhattan == 1)  # [9, 9] bool
    adjacent_f = adjacent.astype(jnp.float32)

    # Per-color connectivity (only trustworthy when full_deduction holds).
    color_member = effective_onehot.astype(jnp.float32).T  # [3, 9]
    pair = color_member @ adjacent_f @ color_member.T  # [3, 3]
    edge_count = jnp.diagonal(pair) / 2.0  # [3]
    all_connected = jnp.all(edge_count >= 2.0)

    end_score = jnp.where(full_deduction & all_connected, 1e7, _NEG)

    # A known card whose own color group isn't yet certified connected
    # (edge_count < 2): learning about its unknown neighbors is the most
    # directly actionable information available.
    edge_count_per_card = edge_count[effective_color_idx]  # [9]
    needy_known = effective_known & (edge_count_per_card < 2.0)  # [9]
    remaining_f = remaining.astype(jnp.float32)  # [3]
    prob_match_per_card = remaining_f[effective_color_idx] / jnp.maximum(
        num_unknown.astype(jnp.float32), 1.0
    )  # [9]
    needy_value = jnp.where(needy_known, prob_match_per_card, 0.0)  # [9]

    # --- Observe scores (actions 1..9, id = 1 + i) ---
    partner_seen = node_features[:NUM_CARDS, 9] > 0  # [9]
    large_dist = jnp.asarray(1000.0, dtype=jnp.float32)
    dist_to_needy = jnp.where(needy_known[None, :], manhattan.astype(jnp.float32), large_dist)  # [9, 9]
    min_dist_to_needy = jnp.min(dist_to_needy, axis=1)  # [9]
    proximity_bonus = jnp.clip(3.0 - min_dist_to_needy, 0.0, 3.0)  # [9], 2.0 at adjacency, tapers to 0 by distance 3
    observe_base = jnp.where(effective_known, 0.0, 10.0)
    observe_coverage = jnp.where(effective_known, 0.0, jnp.where(partner_seen, 0.0, 1.0))
    observe_adjacency = jnp.where(effective_known, 0.0, proximity_bonus)
    observe_scores = observe_base + observe_coverage + observe_adjacency  # [9]

    # --- Move scores (actions 10..738, id = 10 + i*81 + x*9 + y) ---
    color_eq = effective_color_idx[:, None] == effective_color_idx[None, :]  # [9, 9]
    known_pair = effective_known[:, None] & effective_known[None, :]
    same_color_partner = (color_eq & known_pair & ~eye9).astype(jnp.float32)  # [9, 9]

    old_adjacent_count = jnp.sum(same_color_partner * adjacent_f, axis=1)  # [9]

    cell_diff = grid_xy[:, None, :] - positions[None, :, :]  # [81, 9, 2]
    cell_manhattan = jnp.sum(jnp.abs(cell_diff), axis=-1)  # [81, 9]
    cell_adjacent = (cell_manhattan == 1).astype(jnp.float32)  # [81, 9]

    new_adjacent_count = same_color_partner @ cell_adjacent.T  # [9, 81]
    move_score = new_adjacent_count - old_adjacent_count[:, None]  # [9, 81]

    # Distance-gradient bonus: the board starts as a solid 3x3 block, so a
    # card deep inside it may have no free cell adjacent to a same-color
    # partner this turn. Reward shrinking the Manhattan distance to the
    # nearest known same-color partner so multi-move migrations make
    # progress even when a move cannot achieve adjacency outright. Weighted
    # well below a real adjacency gain (+1) so it never overrides one.
    same_color_partner_bool = color_eq & known_pair & ~eye9  # [9, 9]
    large = jnp.asarray(1000.0, dtype=jnp.float32)
    old_dist_masked = jnp.where(same_color_partner_bool, manhattan.astype(jnp.float32), large)  # [9, 9]
    old_min_dist = jnp.min(old_dist_masked, axis=1)  # [9]
    has_partner = old_min_dist < large  # [9]

    partner_b = same_color_partner_bool[:, None, :]  # [9, 1, 9]
    cell_manhattan_b = cell_manhattan.astype(jnp.float32)[None, :, :]  # [1, 81, 9]
    new_dist_masked = jnp.where(partner_b, cell_manhattan_b, large)  # [9, 81, 9]
    new_min_dist = jnp.min(new_dist_masked, axis=-1)  # [9, 81]

    distance_bonus = jnp.where(
        has_partner[:, None],
        (old_min_dist[:, None] - new_min_dist) * 0.05,
        0.0,
    )  # [9, 81]
    move_score = move_score + distance_bonus

    # Probabilistic nudge for still-unknown movers: every currently-unknown
    # card is a priori exchangeable, so we cannot tell *which* one matches a
    # given needy known-color cluster, but placing *some* unknown card next
    # to a needy cluster has strictly non-negative expected value versus an
    # uninformed neutral move (an unknown mover otherwise scores exactly 0).
    # Weighted well below a real adjacency gain (+1) and below the maximum
    # possible probability (<1, since certainty would already have been
    # captured by set-elimination above).
    soft_target_value = cell_adjacent @ needy_value  # [81]

    unknown_mover = ~effective_known  # [9]
    move_score = move_score + jnp.where(
        unknown_mover[:, None], soft_target_value[None, :] * 0.3, 0.0
    )

    # Mirror case: a known needy mover approaching an unknown card also has
    # positive expected value (same probability reasoning, opposite roles).
    # needy_value is already 0 for unknown cards, so this only affects known
    # needy movers.
    unknown_adjacent_count = cell_adjacent @ (~effective_known).astype(jnp.float32)  # [81]
    move_score = move_score + jnp.outer(needy_value, unknown_adjacent_count) * 0.3

    move_key, place_key = jax.random.split(rng)
    move_noise = jax.random.uniform(move_key, (NUM_CARDS, GRID * GRID), minval=0.0, maxval=1e-3)
    move_score = move_score + move_noise
    move_scores_flat = move_score.reshape(NUM_CARDS * GRID * GRID)  # [729]

    # --- Reveal scores (actions 739..742, id = 739 + h) ---
    hint_revealed = jnp.sum(hint_colorset, axis=-1) > 0  # [4]
    reveal_scores = jnp.where(hint_revealed, _NEG, 50.0)  # [4]

    # --- Place scores (actions 743..1066, id = 743 + h*81 + x*9 + y) ---
    # A card's *possible* color set: for a known/deduced card, exactly its
    # one true color; for a still-ambiguous card, every color that still has
    # remaining slots among the unknowns (all such cards are a-priori
    # exchangeable, so they share the same possible set). Comparing this
    # possible set against a hint's revealed set -- rather than only exact
    # per-card certainty -- lets set-elimination prove a placement is
    # guaranteed correct (possible set is a subset of the hint's set) or
    # guaranteed wrong (no overlap at all) even before the card's exact
    # color is individually known.
    possible_mask = jnp.where(
        effective_known[:, None],
        effective_onehot.astype(bool),
        jnp.broadcast_to((remaining > 0)[None, :], (NUM_CARDS, NUM_COLORS)),
    )  # [9, 3]
    possible_f = possible_mask.astype(jnp.float32)
    possible_count = jnp.sum(possible_f, axis=-1)  # [9]
    overlap = hint_colorset.astype(jnp.float32) @ possible_f.T  # [4, 9]

    guaranteed_correct = (possible_count[None, :] > 0) & (overlap >= possible_count[None, :])  # [4, 9]
    guaranteed_wrong = overlap <= 0  # [4, 9]
    ambiguous_prob = overlap / jnp.maximum(possible_count[None, :], 1.0)  # [4, 9]

    # Even a *guaranteed*-correct placement locks the card in its current
    # cell, which can pre-empt a clustering move it still needs. Only treat
    # it as safe to prioritize over revealing a fresh hint once every color
    # the card could possibly be already has its cluster resolved
    # (edge_count >= 2); otherwise fall through to the same probability
    # scoring as a genuinely ambiguous card (guaranteed correct implies
    # ambiguous_prob == 1.0 there, so it is still ranked at the top of that
    # tier, just below revealing).
    large_edge = jnp.asarray(1000.0, dtype=jnp.float32)
    masked_edge = jnp.where(possible_mask, edge_count[None, :], large_edge)  # [9, 3]
    min_possible_edge = jnp.min(masked_edge, axis=-1)  # [9]
    safe_to_lock = min_possible_edge >= 2.0  # [9]

    place_score_hk = jnp.where(
        guaranteed_correct & safe_to_lock[None, :],
        100.0 + old_adjacent_count[None, :] * 10.0,
        jnp.where(
            guaranteed_wrong,
            -1e6,
            ambiguous_prob * 2.0 + jnp.where(safe_to_lock[None, :], 0.1, 0.0),
        ),
    )
    place_noise = jax.random.uniform(place_key, (NUM_HINTS, NUM_CARDS), minval=0.0, maxval=1e-3)
    place_score_hk = place_score_hk + place_noise

    cell_idx = positions[:, 0] * GRID + positions[:, 1]  # [9]
    place_grid = jnp.full((NUM_HINTS, GRID * GRID), _NEG)
    place_grid = place_grid.at[:, cell_idx].set(place_score_hk)
    place_scores_flat = place_grid.reshape(NUM_HINTS * GRID * GRID)  # [324]

    # --- Assemble the full 1068-length score vector ---
    score = jnp.full((NUM_ACTIONS,), _NEG)
    score = score.at[0].set(0.0)
    score = jax.lax.dynamic_update_slice(score, observe_scores, (1,))
    score = jax.lax.dynamic_update_slice(score, move_scores_flat, (10,))
    score = jax.lax.dynamic_update_slice(score, reveal_scores, (739,))
    score = jax.lax.dynamic_update_slice(score, place_scores_flat, (743,))
    score = score.at[1067].set(end_score)

    score = jnp.where(legal, score, _NEG)
    action = jnp.argmax(score).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)

    trace = {
        'full_deduction': full_deduction.astype(jnp.int32),
        'all_connected': all_connected.astype(jnp.int32),
        'chosen_action': action,
        'edge_count': edge_count,
        'known_count': jnp.sum(known.astype(jnp.int32)),
        'effective_known_count': jnp.sum(effective_known.astype(jnp.int32)),
    }
    return action, next_memory, trace
