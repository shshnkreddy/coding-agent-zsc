"""Convention-light greedy policy for the two-player 16-card Yokai game."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'observable_clusterer', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    # The environment supplies perfect private recall. A counter is retained as
    # fixed-shape memory but is deliberately not used as a private convention.
    return jnp.asarray(0, dtype=jnp.int32)


def _first_best(score, legal):
    masked = jnp.where(legal, score, jnp.asarray(-1.0e9, score.dtype))
    return jnp.argmax(masked).astype(jnp.int32)


def _observe_action(obs):
    colors = obs.node_features[:16, :4]
    unknown = jnp.sum(colors, axis=1) == 0
    partner_unseen = obs.node_features[:16, 11] == 0
    unlocked = obs.node_features[:16, 10] == 0
    ids = jnp.arange(16, dtype=jnp.float32)
    score = 100.0 * unknown + 10.0 * partner_unseen + unlocked - 0.001 * ids
    return 1 + _first_best(score, obs.legal_actions[1:17])


def _effective_colors(obs):
    private = obs.node_features[:16, :4].astype(jnp.int32)
    hint_sets = obs.node_features[16:23, 4:8].astype(jnp.int32)
    links = obs.adjacency_matrix[:16, 16:23].astype(jnp.int32)
    attached = links @ hint_sets
    public_singleton = jnp.sum(attached, axis=1) == 1
    privately_known = jnp.sum(private, axis=1) == 1
    exact = jnp.where(privately_known[:, None], private,
                      jnp.where(public_singleton[:, None], attached, 0))

    # Propagate literal hint constraints through the fixed four-of-each
    # multiset.  A multi-color tile becomes exact once all but one of its colors
    # have already reached count four.
    has_hint = jnp.sum(attached, axis=1) > 0
    for _ in range(7):
        remaining_positive = (4 - jnp.sum(exact, axis=0)) > 0
        unresolved = jnp.sum(exact, axis=1) == 0
        allowed = remaining_positive[None, :] & jnp.where(
            has_hint[:, None], attached > 0, jnp.ones((16, 4), dtype=jnp.bool_))
        unique = jnp.sum(allowed, axis=1) == 1
        exact = jnp.where((unresolved & unique)[:, None], allowed.astype(jnp.int32), exact)
    return exact


def _assignment_options(colors, option_count, max_unknown):
    """Enumerate colors of the first max_unknown unresolved cards."""
    unresolved = jnp.sum(colors, axis=1) == 0
    ranks = jnp.cumsum(unresolved.astype(jnp.int32))
    num_unknown = jnp.sum(unresolved)
    options = jnp.arange(option_count, dtype=jnp.int32)
    assigned = jnp.broadcast_to(colors[None, :, :], (option_count, 16, 4))
    card_ids = jnp.arange(16)
    for rank in range(max_unknown):
        card_id = jnp.argmax((ranks == rank + 1).astype(jnp.int32))
        digit = (options // (4 ** rank)) % 4
        choice = jnp.eye(4, dtype=jnp.int32)[digit]
        replace = (card_ids == card_id) & (num_unknown > rank)
        assigned = jnp.where(replace[None, :, None], choice[:, None, :], assigned)
    counts_ok = jnp.all(jnp.sum(assigned, axis=1) == 4, axis=1)
    valid = counts_ok & (num_unknown <= max_unknown)
    return assigned, valid


def _solved_assignments(assignments, adjacency):
    same = (assignments @ jnp.swapaxes(assignments, 1, 2)) > 0
    eye = jnp.eye(16, dtype=jnp.bool_)
    reach = (adjacency[:, None, :, :] & same[None, :, :, :]) | eye[None, None, :, :]
    for k in range(16):
        reach = reach | (reach[:, :, :, k, None] & reach[:, :, None, k, :])
    return jnp.all(~same[None, :, :, :] | reach, axis=(2, 3))


def _solved_one(assignment, adjacency):
    same = (assignment @ assignment.T) > 0
    reach = adjacency & same[None, :, :]
    reach = reach | jnp.eye(16, dtype=jnp.bool_)[None, :, :]
    for k in range(16):
        reach = reach | (reach[:, :, k, None] & reach[:, None, k, :])
    return jnp.all(~same[None, :, :] | reach, axis=(1, 2))


def _safe_to_end(obs):
    colors = _effective_colors(obs)
    assignments, valid = _assignment_options(colors, 256, 4)
    adjacency = obs.adjacency_matrix[:16, :16].astype(jnp.bool_)[None, :, :]
    solved = _solved_assignments(assignments, adjacency)[0]
    return jnp.any(valid) & jnp.all(~valid | solved)


def _move_action(obs):
    pos = obs.positions.astype(jnp.int32)
    colors = _effective_colors(obs).astype(jnp.float32)
    known = jnp.sum(colors, axis=1) > 0

    a = jnp.arange(1600, dtype=jnp.int32)
    moved = a // 100
    cell = a % 100
    dest = jnp.stack((cell // 10, cell % 10), axis=1)
    candidate_pos = jnp.where(
        (jnp.arange(16)[None, :, None] == moved[:, None, None]),
        dest[:, None, :],
        pos[None, :, :],
    )
    delta = jnp.abs(candidate_pos[:, :, None, :] - candidate_pos[:, None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    same = (colors @ colors.T) > 0
    eye = jnp.eye(16, dtype=jnp.bool_)
    reach = (adjacent & same[None, :, :]) | eye[None, :, :]
    # Static Floyd-Warshall closure: joining two fragments is worth the full
    # cross-product of their sizes, unlike merely adding another cycle edge.
    for k in range(16):
        reach = reach | (reach[:, :, k, None] & reach[:, None, k, :])
    component_pairs = jnp.sum(reach & same[None, :, :] & ~eye[None, :, :], axis=(1, 2)) * 0.5
    same_edges = jnp.sum(adjacent & same[None, :, :] & ~eye[None, :, :], axis=(1, 2)) * 0.5
    remaining = jnp.maximum(4.0 - jnp.sum(colors, axis=0), 0.0)
    unknown_count = jnp.maximum(jnp.sum(~known), 1)
    hint_sets = obs.node_features[16:23, 4:8].astype(jnp.float32)
    links = obs.adjacency_matrix[:16, 16:23].astype(jnp.float32)
    attached = links @ hint_sets
    has_hint = jnp.sum(attached, axis=1) > 0
    allowed = jnp.where(
        has_hint[:, None], attached > 0, jnp.ones((16, 4), dtype=jnp.bool_))
    belief_weight = remaining[None, :] * allowed
    belief_total = jnp.sum(belief_weight, axis=1, keepdims=True)
    fallback = remaining[None, :] / unknown_count
    unknown_belief = jnp.where(
        belief_total > 0, belief_weight / jnp.maximum(belief_total, 1.0), fallback)
    belief = jnp.where(
        known[:, None], colors, unknown_belief)
    expected_same = belief @ belief.T
    uncertain_pair = (~known[:, None]) | (~known[None, :])
    expected_uncertain_edges = jnp.sum(
        adjacent * expected_same[None, :, :]
        * uncertain_pair[None, :, :] * ~eye[None, :, :], axis=(1, 2)) * 0.5
    center_cost = jnp.sum(jnp.abs(candidate_pos.astype(jnp.float32) - 4.5), axis=(1, 2))
    mover_known = known[moved].astype(jnp.float32)
    partner_seen_private = ((obs.node_features[:16, 11] > 0) & ~known)[moved].astype(jnp.float32)
    score = (50.0 * component_pairs + 0.0 * same_edges
             + 2.0 * expected_uncertain_edges * mover_known
             + 0.1 * mover_known - partner_seen_private - 0.002 * center_cost)

    # With at most two unresolved cards, directly optimize the fraction of
    # multiset-consistent worlds in which this relocation completes the game.
    assignments, valid = _assignment_options(colors.astype(jnp.int32), 16, 2)
    def add_world(option, total):
        solved = _solved_one(assignments[option], adjacent)
        return total + (solved & valid[option]).astype(jnp.int32)
    solved_count = jax.lax.fori_loop(
        0, 16, add_world, jnp.zeros(1600, dtype=jnp.int32))
    valid_count = jnp.maximum(jnp.sum(valid), 1)
    score = score + 10000.0 * solved_count.astype(jnp.float32) / valid_count
    idx = _first_best(score, obs.legal_actions[17:1617])
    return 17 + idx


def _hint_action(obs):
    # Reveal all tiles first, delaying irreversible card locks.
    reveal_legal = obs.legal_actions[1617:1624]
    reveal = 1617 + jnp.argmax(reveal_legal).astype(jnp.int32)

    colors = _effective_colors(obs)
    hints = obs.node_features[16:23, 4:8].astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    delta = jnp.abs(pos[:, None, :] - pos[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    same = (colors @ colors.T) > 0
    reach = (adjacent & same) | jnp.eye(16, dtype=jnp.bool_)
    for k in range(16):
        reach = reach | (reach[:, k, None] & reach[None, k, :])
    component_size = jnp.sum(reach & same, axis=1).astype(jnp.float32)

    a = jnp.arange(700, dtype=jnp.int32)
    hint_id = a // 100
    cell = a % 100
    xy = jnp.stack((cell // 10, cell % 10), axis=1)
    at_card = jnp.all(xy[:, None, :] == pos[None, :, :], axis=-1)
    card_id = jnp.argmax(at_card, axis=1)
    card_color = colors[card_id]
    compatible = jnp.sum(card_color * hints[hint_id], axis=1) > 0
    known_card = jnp.sum(card_color, axis=1) > 0
    locked_by_color = colors.T @ obs.node_features[:16, 10].astype(jnp.int32)
    prior_color_locks = card_color @ locked_by_color
    score = (100.0 * compatible + 10.0 * known_card
             + 10.0 * component_size[card_id]
             - 0.25 * prior_color_locks.astype(jnp.float32)
             - 0.001 * a.astype(jnp.float32))
    place = 1624 + _first_best(score, obs.legal_actions[1624:2324])
    return jnp.where(jnp.any(reveal_legal), reveal, place)


def agent_act(obs, reset, memory, rng):
    del rng
    active = obs.node_features[0, 12] == 1
    substep = obs.node_features[0, 13].astype(jnp.int32)
    actions = jnp.stack((
        _observe_action(obs),
        _observe_action(obs),
        _move_action(obs),
        _hint_action(obs),
    ))
    planned = jnp.where((substep == 0) & _safe_to_end(obs),
                        jnp.asarray(2324, jnp.int32), actions[substep])
    action = jnp.where(active, planned, jnp.asarray(0, jnp.int32))
    old_memory = jnp.where(reset, jnp.asarray(0, jnp.int32), memory)
    next_memory = old_memory + active.astype(jnp.int32)
    return action.astype(jnp.int32), next_memory, {}
