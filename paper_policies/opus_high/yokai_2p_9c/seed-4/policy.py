"""Belief-based cooperative Yokai policy.

The seat keeps an exact uniform posterior over every color assignment that is
consistent with the cards it has personally observed, and acts greedily on that
posterior: it moves cards to maximise expected color connectivity, spends the
mandatory substep-3 action on the cheapest hint action it can justify, and ends
the episode as soon as the expected terminal reward beats a continuation
estimate.  No signalling convention with the partner is assumed: every decision
uses only public board facts and this seat's own private observations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_solver', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_ASSIGN = 1680
MOVE_BASE = 10
NUM_MOVES = 729
REVEAL_BASE = 739
PLACE_BASE = 743
END_ACTION = 1067
NEG = -1.0e9

# Terminal value of ending after k hint actions (k = 0..8) when every placement
# so far is correct is 20 - 3*ceil(k/2) - floor(k/2).  Ending is worthwhile when
# the expected terminal reward beats what later turns are realistically worth;
# the table below is that continuation estimate, biased low because an unknown
# partner makes continuation less valuable.
END_THRESHOLD = (9.0, 9.0, 6.5, 5.5, 4.5, 3.5, 3.0, 2.0, 0.0)

# An early end is only taken when the seat's own observations alone give the
# solved board at least this probability.  Behavioural inference about the
# partner can be wrong, but this floor never depends on the partner.
HARD_MIN = 0.15

# Minimum posterior quality before a hint is placed rather than revealed.
PLACE_MIN = 0.45

# Log-odds credited to assignments in which a card the partner just moved
# shares its color with one of the partner-seen cards it landed next to.  Any
# competent partner clusters the colors it knows, so this is evidence from
# observed behaviour rather than an agreed signalling convention.
PARTNER_EVIDENCE = 1.5

# Log-odds credited to assignments in which a placed hint's color set really
# does contain its card's color.  A placement is worth +1 when right and -1 when
# wrong, so any competent partner places hints it believes to be correct.
PLACEMENT_EVIDENCE = 1.0

# Weight of "every color connected" relative to the count of connected colors.
# Large values make the search prefer arrangements that stay solved whatever
# the unobserved cards turn out to be, which is what makes an early end safe.
SOLVED_BONUS = 4.0

# Credit for a color with two of its three cards already touching.  A smooth
# value function gives the one-ply move search a gradient to follow while the
# colors are still scattered.
PARTIAL_CREDIT = 0.25


def _tables():
    """Every deal of 3 cards per color to the 9 card ids, plus pair indices.

    ``assign[a, i]`` is the color of card ``i`` under assignment ``a``.
    ``pairs[a]`` holds, for each color, the three flat [9, 9] adjacency indices
    of that color's card pairs; three same-colored cards are connected exactly
    when at least two of those three pairs are orthogonally adjacent.
    """
    assign = []
    pairs = []
    for a0 in range(9):
        for b0 in range(a0 + 1, 9):
            for c0 in range(b0 + 1, 9):
                rest = [i for i in range(9) if i not in (a0, b0, c0)]
                for u in range(6):
                    for v in range(u + 1, 6):
                        for w in range(v + 1, 6):
                            group1 = (rest[u], rest[v], rest[w])
                            group2 = tuple(i for i in rest if i not in group1)
                            row = [0] * 9
                            flat = []
                            for color, group in enumerate(((a0, b0, c0), group1, group2)):
                                p, q, r = group
                                row[p] = color
                                row[q] = color
                                row[r] = color
                                flat += [p * 9 + q, p * 9 + r, q * 9 + r]
                            assign.append(row)
                            pairs.append(flat)
    return (jnp.asarray(assign, dtype=jnp.int8),
            jnp.asarray(pairs, dtype=jnp.int32).reshape(-1))


def _adjacency(pos):
    """Orthogonal card adjacency for positions [..., 9, 2]."""
    delta = jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]).sum(-1)
    return (delta == 1).astype(jnp.int8)


def _color_edges(pos, pair_flat):
    """Adjacent same-color pairs per assignment and color, for positions."""
    lead = pos.shape[:-2]
    flat = _adjacency(pos).reshape(lead + (81,))
    edges = jnp.take(flat, pair_flat, axis=-1)
    return edges.reshape(lead + (NUM_ASSIGN, 3, 3)).sum(-1), flat


def _color_connected(pos, pair_flat):
    """Per-assignment per-color connectivity for positions [..., 9, 2]."""
    edges, flat = _color_edges(pos, pair_flat)
    return edges >= 2, flat


def _normalise(weights):
    return weights / jnp.maximum(weights.sum(), 1e-6)


def _board_value(pos, pair_flat):
    """Per-assignment board value: connected colors, with a solved bonus."""
    edges, flat = _color_edges(pos, pair_flat)
    conn = edges >= 2
    ncon = conn.sum(-1).astype(jnp.float32)
    partial = jnp.sum((edges == 1).astype(jnp.float32), axis=-1)
    value = ncon + PARTIAL_CREDIT * partial + SOLVED_BONUS * (ncon >= 3.0)
    return value, conn, flat


def _observe_action(known, partner_seen, legal, rng, pos, prob, assign,
                    pair_flat):
    """Inspect the card whose color would tell us most about the board.

    Pivotality is the posterior-weighted spread of the board value across the
    card's possible colors; a card whose color is already known scores zero, so
    unknown and decision-relevant cards come first.
    """
    value, _, _ = _board_value(pos, pair_flat)
    mean_value = (value * prob).sum()
    weighted = prob * value
    pivot = jnp.zeros((9,), dtype=jnp.float32)
    for color in range(3):
        mask = (assign == color).astype(jnp.float32)
        pivot = pivot + jnp.abs(weighted @ mask - mean_value * (prob @ mask))
    score = (10.0 * (known < 0).astype(jnp.float32)
             + pivot
             + 0.3 * (1.0 - partner_seen.astype(jnp.float32))
             + 0.01 * jax.random.uniform(rng, (9,)))
    score = jnp.where(legal[1:10], score, NEG)
    return (1 + jnp.argmax(score)).astype(jnp.int32)


def _move_action(pos, prob, legal, assign, pair_flat):
    """Greedy move maximising expected color connectivity of the board."""
    card_of = jnp.arange(NUM_MOVES, dtype=jnp.int32) // 81
    x_of = (jnp.arange(NUM_MOVES, dtype=jnp.int32) % 81) // 9
    y_of = jnp.arange(NUM_MOVES, dtype=jnp.int32) % 9
    new_xy = jnp.stack([x_of, y_of], axis=-1)
    moved = (jnp.arange(9, dtype=jnp.int32)[None, :] == card_of[:, None])
    boards = jnp.where(moved[..., None], new_xy[:, None, :], pos[None, :, :])
    value, _, flat = _board_value(boards, pair_flat)
    expected = (value * prob[None, :]).sum(-1)
    compact = flat.sum(-1).astype(jnp.float32) * 0.01
    score = jnp.where(legal[MOVE_BASE:MOVE_BASE + NUM_MOVES],
                      expected + compact, NEG)
    return (MOVE_BASE + jnp.argmax(score)).astype(jnp.int32)


def _terminal_value(pos, prob, hard_prob, hint_sets, revealed, placed, targets,
                    assign, pair_flat):
    """Expected reward of ending the episode right now."""
    conn, _ = _color_connected(pos, pair_flat)
    solved = jnp.all(conn, axis=-1).astype(jnp.float32)
    unrevealed = jnp.sum(1.0 - revealed.astype(jnp.float32))
    waiting = jnp.sum(revealed.astype(jnp.float32)
                      * (1.0 - placed.astype(jnp.float32)))
    base = 5.0 * unrevealed + 2.0 * waiting
    correct = _hint_colors(assign, targets, hint_sets)
    delta = (placed.astype(jnp.float32)[None, :] * (2.0 * correct - 1.0)).sum(-1)
    reward = solved * (base + delta)
    return (reward * prob).sum(), (solved * hard_prob).sum()


def _hint_action(pos, prob, hint_sets, legal, assign, pair_flat):
    """Place a hint whose set almost surely covers the card, else reveal one."""
    conn, _ = _color_connected(pos, pair_flat)
    columns = []
    for color in range(3):
        columns.append(((assign == color).astype(jnp.float32)
                        * prob[:, None]).sum(0))
    card_color = jnp.stack(columns, axis=-1)               # [9, 3]
    match = hint_sets @ card_color.T                       # [4, 9]
    card_conn = jnp.take_along_axis(conn.astype(jnp.float32),
                                    assign.astype(jnp.int32), axis=1)
    settled = (card_conn * prob[:, None]).sum(0)           # [9]
    quality = (match > 0.99).astype(jnp.float32) * settled[None, :]
    cell = pos[:, 0] * 9 + pos[:, 1]
    place_id = (PLACE_BASE + jnp.arange(4, dtype=jnp.int32)[:, None] * 81
                + cell[None, :])
    quality = jnp.where(legal[place_id], quality, NEG).reshape(-1)
    best = jnp.argmax(quality)
    place_action = place_id.reshape(-1)[best].astype(jnp.int32)

    # Reveal the two-color hints (ids 1..3) before the single-color hint 0.
    reveal_score = jnp.asarray([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    reveal_legal = legal[REVEAL_BASE:REVEAL_BASE + 4]
    reveal_score = jnp.where(reveal_legal, reveal_score, NEG)
    reveal_action = (REVEAL_BASE + jnp.argmax(reveal_score)).astype(jnp.int32)

    can_reveal = jnp.any(reveal_legal)
    can_place = quality[best] > NEG / 2.0
    take_place = can_place & ((quality[best] >= PLACE_MIN)
                              | jnp.logical_not(can_reveal))
    return jnp.where(take_place, place_action, reveal_action)


def _hint_colors(assign, targets, hint_sets):
    """P-style indicator that each placed hint's set covers its card's color."""
    target_color = (targets[None, :, :]
                    * assign[:, None, :].astype(jnp.float32)).sum(-1)
    correct = jnp.zeros((NUM_ASSIGN, 4), dtype=jnp.float32)
    for color in range(3):
        correct = correct + ((target_color == color).astype(jnp.float32)
                             * hint_sets[None, :, color])
    return correct


def _partner_evidence(pos, prev_pos, was_active, partner_seen, assign):
    """Log-weight bonus from a single card the partner moved since last call."""
    changed = jnp.any(pos != prev_pos, axis=1)
    single = jnp.sum(changed.astype(jnp.int32)) == 1
    moved = changed.astype(jnp.float32)
    neighbours = (moved[None, :] @ _adjacency(pos).astype(jnp.float32))[0]
    witness = (neighbours > 0) & partner_seen & jnp.logical_not(changed)
    gate = (single & jnp.logical_not(was_active)
            & jnp.any(partner_seen & changed) & jnp.any(witness))
    moved_color = (assign.astype(jnp.float32) * moved[None, :]).sum(1)
    same = jnp.any((assign.astype(jnp.float32) == moved_color[:, None])
                   & witness[None, :], axis=1)
    return jnp.where(gate, PARTNER_EVIDENCE * same.astype(jnp.float32),
                     jnp.zeros((NUM_ASSIGN,), dtype=jnp.float32))


def init_agent():
    return (jnp.full((9,), -1, dtype=jnp.int8),
            jnp.zeros((NUM_ASSIGN,), dtype=jnp.float32),
            jnp.zeros((9, 2), dtype=jnp.int8),
            jnp.zeros((), dtype=jnp.bool_))


def agent_act(obs, reset, memory, rng):
    assign, pair_flat = _tables()
    features = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions

    old_known, old_logw, old_pos, old_active = memory
    fresh = reset | jnp.all(old_pos == 0)
    old_known = jnp.where(fresh, jnp.full((9,), -1, dtype=jnp.int8), old_known)
    old_logw = jnp.where(fresh, jnp.zeros((NUM_ASSIGN,), dtype=jnp.float32),
                         old_logw)
    old_pos = jnp.where(fresh, pos.astype(jnp.int8), old_pos)
    old_active = old_active & jnp.logical_not(fresh)

    own_onehot = features[0:9, 0:3]
    observed_now = own_onehot.sum(1) > 0
    observed_color = jnp.argmax(own_onehot, axis=1).astype(jnp.int8)
    known = jnp.where(observed_now, observed_color, old_known)

    partner_seen = features[0:9, 9] > 0
    active = features[0, 10] > 0
    substep = features[0, 11]

    hint_sets = features[9:13, 3:6].astype(jnp.float32)
    revealed = hint_sets.sum(1) > 0
    placed = features[9:13, 8] > 0
    targets = obs.adjacency_matrix[9:13, 0:9].astype(jnp.float32)

    logw = old_logw + _partner_evidence(pos, old_pos.astype(jnp.int32),
                                        old_active, partner_seen, assign)
    hint_belief = PLACEMENT_EVIDENCE * (
        _hint_colors(assign, targets, hint_sets)
        * placed.astype(jnp.float32)[None, :]).sum(-1)
    consistent = jnp.all((assign == known[None, :]) | (known[None, :] < 0),
                         axis=1).astype(jnp.float32)
    total_logw = logw + hint_belief
    weights = consistent * jnp.exp(total_logw - jnp.max(total_logw))
    prob = _normalise(jnp.where(jnp.sum(consistent) > 0, weights, consistent))
    hard_prob = _normalise(consistent)
    rng_obs, _ = jax.random.split(rng)

    def branch_end_or_observe():
        value, hard_solved = _terminal_value(pos, prob, hard_prob, hint_sets,
                                             revealed, placed, targets, assign,
                                             pair_flat)
        used = (jnp.sum(revealed.astype(jnp.int32))
                + jnp.sum(placed.astype(jnp.int32)))
        threshold = jnp.asarray(END_THRESHOLD, dtype=jnp.float32)[
            jnp.clip(used, 0, 8)]
        stop = (value >= threshold) & (hard_solved >= HARD_MIN) & legal[END_ACTION]
        action = jnp.where(stop, jnp.int32(END_ACTION),
                           _observe_action(known, partner_seen, legal, rng_obs,
                                           pos, prob, assign, pair_flat))
        return action, value, hard_solved

    def branch_observe():
        return (_observe_action(known, partner_seen, legal, rng_obs, pos,
                                prob, assign, pair_flat),
                jnp.float32(0.0), jnp.float32(0.0))

    def branch_move():
        return (_move_action(pos, prob, legal, assign, pair_flat),
                jnp.float32(0.0), jnp.float32(0.0))

    def branch_hint():
        return (_hint_action(pos, prob, hint_sets, legal, assign, pair_flat),
                jnp.float32(0.0), jnp.float32(0.0))

    def act():
        return jax.lax.switch(jnp.clip(substep, 0, 3),
                              [branch_end_or_observe, branch_observe,
                               branch_move, branch_hint])

    def idle():
        return jnp.int32(0), jnp.float32(0.0), jnp.float32(0.0)

    action, value, hard_solved = jax.lax.cond(active, act, idle)
    action = jnp.where(legal[action], action,
                       jnp.argmax(legal).astype(jnp.int32)).astype(jnp.int32)

    next_memory = (known, logw, pos.astype(jnp.int8), active)
    trace = {
        'end_value': value,
        'p_solved_hard': hard_solved,
        'num_known': jnp.sum((known >= 0).astype(jnp.int32)),
        'action': action,
    }
    return action, next_memory, trace
