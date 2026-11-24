"""Cluster-and-end cooperative policy for two-player 9-card Yokai.

Strategy summary (grounded in public rules + own private observations only,
no shared convention with the partner):
  - Persistently remember own-observed card colors across turns in memory.
  - Deduce remaining unknown card colors once only one color still has cards
    unaccounted for (simple, always-correct elimination).
  - Only choose to end the game once colors are fully determined (observed or
    deduced) AND a computed adjacency closure shows all 3 colors connected.
  - When observing, prefer unlocked cards whose color we do not yet know.
  - When moving, prefer relocating a known-color card next to other known
    same-color cards (vectorized score over all legal move actions).
  - Hint substep is forced every turn; revealing keeps a hint's value at R=2,
    while placing turns it into C=1 (correct) or I=-1 (incorrect), both worse
    than leaving it revealed. So always prefer reveal over place, and when
    forced to place (no unrevealed hint remains), prefer a card whose known
    color is guaranteed inside the hint's revealed color set.
  - A single unified preference score over all 1068 actions is masked by
    obs.legal_actions and argmax-selected; the environment's own legality
    mask naturally restricts the action set to whichever substep is active,
    so no explicit substep branching is required.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'cluster_and_end_v2', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
NUM_HINTS = 4
NUM_ACTIONS = 1068


def init_agent():
    return jnp.full((NUM_CARDS,), -1, dtype=jnp.int8)


def _update_known_colors(known_color, node_features):
    card_rows = node_features[:NUM_CARDS, 0:3]
    has_color = jnp.sum(card_rows, axis=-1) > 0
    observed_color = jnp.argmax(card_rows, axis=-1).astype(jnp.int8)
    return jnp.where(has_color, observed_color, known_color)


def _deduce(known_color):
    color_ids = jnp.arange(NUM_COLORS, dtype=jnp.int8)
    onehot_known = (known_color[:, None] == color_ids[None, :]).astype(jnp.int32)
    counts = jnp.sum(onehot_known, axis=0)
    remaining = NUM_CARDS // NUM_COLORS - counts
    colors_with_remaining = remaining > 0
    n_colors_remaining = jnp.sum(colors_with_remaining)
    deduce_color = jnp.argmax(remaining).astype(jnp.int8)
    unknown_mask = known_color == -1
    effective = jnp.where(
        unknown_mask,
        jnp.where(n_colors_remaining == 1, deduce_color, jnp.int8(-1)),
        known_color,
    )
    fully_determined = jnp.all(effective != -1)
    return effective, fully_determined


def _adjacency_bool(positions):
    dx = jnp.abs(positions[:, None, 0] - positions[None, :, 0])
    dy = jnp.abs(positions[:, None, 1] - positions[None, :, 1])
    return (dx + dy) == 1


def _color_connected(effective_color, positions):
    adj = _adjacency_bool(positions)

    def one_color(c):
        members = effective_color == c
        same_color_adj = adj & members[:, None] & members[None, :]
        r = (jnp.eye(NUM_CARDS, dtype=jnp.int32) + same_color_adj.astype(jnp.int32)) > 0

        def body(_, r):
            r_int = r.astype(jnp.int32)
            return (r | ((r_int @ r_int) > 0))

        r = jax.lax.fori_loop(0, 4, body, r)
        first_idx = jnp.argmax(members)
        reachable = r[first_idx]
        return jnp.all(jnp.where(members, reachable, True))

    flags = jnp.stack([one_color(c) for c in range(NUM_COLORS)])
    return jnp.all(flags)


def _observe_scores(known_color):
    observe_idx = 1 + jnp.arange(NUM_CARDS)
    pref = jnp.where(known_color == -1, 2.0, 0.5)
    scores = jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32)
    return scores.at[observe_idx].set(pref)


def _current_same_color_neighbors(effective_color, positions):
    adj = _adjacency_bool(positions)
    same = (effective_color[:, None] == effective_color[None, :]) & (effective_color[:, None] != -1)
    return jnp.sum(adj & same, axis=-1).astype(jnp.float32)


def _pair_adjacent(effective_color, positions):
    """For each card i, whether its two other same-color members (if both
    known) are already adjacent to each other, independent of i's position.
    Lets a move that brings i in touch with just one of them still be
    recognized as completing full 3-card connectivity for that color."""
    adj = _adjacency_bool(positions)
    same_color_mask = (
        (effective_color[:, None] == effective_color[None, :])
        & (effective_color[:, None] != -1)
        & (~jnp.eye(NUM_CARDS, dtype=bool))
    )
    mask3 = same_color_mask[:, :, None] & same_color_mask[:, None, :]
    return jnp.any(mask3 & adj[None, :, :], axis=(1, 2))


def _move_scores(effective_color, positions):
    current = _current_same_color_neighbors(effective_color, positions)
    pair_adjacent = _pair_adjacent(effective_color, positions)
    move_idx = jnp.arange(10, 739)
    rel = move_idx - 10
    i = rel // 81
    rem = rel % 81
    x = rem // 9
    y = rem % 9
    color_i = effective_color[i]
    dx = jnp.abs(positions[None, :, 0] - x[:, None])
    dy = jnp.abs(positions[None, :, 1] - y[:, None])
    adj_to_j = (dx + dy) == 1
    same_color_j = (effective_color[None, :] == color_i[:, None]) & (color_i[:, None] != -1)
    not_self = jnp.arange(NUM_CARDS)[None, :] != i[:, None]
    new_neighbors = jnp.sum(adj_to_j & same_color_j & not_self, axis=-1).astype(jnp.float32)
    # Net improvement: reward gains, penalize disturbing an already well-placed
    # known card when no better spot is found for it.
    improvement = new_neighbors - current[i]
    # Extra bonus for a move that completes full 3-card connectivity for its
    # color right now (either by touching both other members, or touching one
    # while the other pair is already linked), prioritizing "finishing" moves
    # over merely "improving" ones when both are otherwise tied.
    completes_group = (new_neighbors >= 2) | ((new_neighbors >= 1) & pair_adjacent[i])
    improvement = improvement + jnp.where(completes_group, 0.5, 0.0)
    # When no informative (color-known) improvement is available, minimize
    # disruption risk to cards whose true color we cannot see: prefer a small
    # relocation over a large jump, since a nearby move is less likely to tear
    # a card away from a same-color cluster we simply do not know about.
    src_x = positions[i, 0]
    src_y = positions[i, 1]
    distance = (jnp.abs(x - src_x) + jnp.abs(y - src_y)).astype(jnp.float32)
    score = improvement * 100.0 - distance * 0.01
    scores = jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32)
    return scores.at[10:739].set(score)


def _hint_scores(effective_color, positions, node_features, current_same_color_neighbors):
    hint_rows = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 3:6]
    unrevealed = jnp.sum(hint_rows, axis=-1) == 0
    reveal_idx = 739 + jnp.arange(NUM_HINTS)
    reveal_scores = jnp.where(unrevealed, 5.0, -1e9)
    scores = jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32)
    scores = scores.at[reveal_idx].set(reveal_scores)

    place_idx = jnp.arange(743, 1067)
    rel = place_idx - 743
    h = rel // 81
    rem = rel % 81
    x = rem // 9
    y = rem % 9
    cell_id = positions[:, 0] * 9 + positions[:, 1]
    card_at_cell = jnp.full((81,), -1, dtype=jnp.int32)
    card_at_cell = card_at_cell.at[cell_id].set(jnp.arange(NUM_CARDS, dtype=jnp.int32))
    cell = x * 9 + y
    k = card_at_cell[cell]
    k_safe = jnp.maximum(k, 0)
    card_color = effective_color[k_safe]
    hint_color_set = hint_rows[h]
    card_color_safe = jnp.clip(card_color, 0, NUM_COLORS - 1).astype(jnp.int32)
    in_set = jnp.take_along_axis(hint_color_set, card_color_safe[:, None], axis=-1)[:, 0]
    known_mask = card_color != -1
    set_size = jnp.sum(hint_color_set, axis=-1).astype(jnp.float32)
    prior = set_size / 3.0
    # Prefer locking a card that is already well-clustered (safe to freeze)
    # over one that might still need to move.
    lock_safety = current_same_color_neighbors[k_safe] * 0.1
    # Among unknown-color cards, prefer locking a peripheral (low total
    # degree) one: it is less likely to be a still-needed connector card.
    total_degree = jnp.sum(_adjacency_bool(positions), axis=-1).astype(jnp.float32)
    peripheral_bonus = -total_degree[k_safe] * 0.05
    place_score = (
        jnp.where(known_mask, jnp.where(in_set > 0, 3.0, -10.0), prior + peripheral_bonus)
        + lock_safety
    )
    place_score = jnp.where(k >= 0, place_score, -1e9)
    return scores.at[place_idx].set(place_score)


def _end_score(fully_determined, connected):
    safe_to_end = fully_determined & connected
    val = jnp.where(safe_to_end, 1000.0, -1e9)
    scores = jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32)
    return scores.at[1067].set(val)


def agent_act(obs, reset, memory, rng):
    del rng
    known_color = jnp.where(reset, jnp.full((NUM_CARDS,), -1, dtype=jnp.int8), memory)
    known_color = _update_known_colors(known_color, obs.node_features)
    effective_color, fully_determined = _deduce(known_color)
    connected = _color_connected(effective_color, obs.positions)

    current_same_color_neighbors = _current_same_color_neighbors(effective_color, obs.positions)
    score = (
        _observe_scores(known_color)
        + _move_scores(effective_color, obs.positions)
        + _hint_scores(effective_color, obs.positions, obs.node_features, current_same_color_neighbors)
        + _end_score(fully_determined, connected)
    )

    masked_score = jnp.where(obs.legal_actions, score, -1e12)
    action = jnp.argmax(masked_score).astype(jnp.int32)

    trace = {
        'fully_determined': fully_determined,
        'connected': connected,
    }
    return action, known_color, trace
