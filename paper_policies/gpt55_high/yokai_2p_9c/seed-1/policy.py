"""Partner-compatible greedy policy for the 9-card Yokai task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'greedy_known_cluster', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _first_legal(legal_actions):
    return jnp.argmax(legal_actions).astype(jnp.int32)


def _card_belief(node_features):
    color_oh = node_features[:9, :3].astype(jnp.int32)
    known = jnp.sum(color_oh, axis=1) > 0
    known_color = jnp.argmax(color_oh, axis=1).astype(jnp.int32)
    known_counts = jnp.sum(color_oh, axis=0)
    remaining = 3 - known_counts
    unknown_count = 9 - jnp.sum(known.astype(jnp.int32))
    possible = remaining > 0
    single_remaining_color = jnp.sum(possible.astype(jnp.int32)) == 1
    can_infer_unknowns = (unknown_count > 0) & single_remaining_color
    inferred_color = jnp.argmax(remaining).astype(jnp.int32)
    confident = known | (can_infer_unknowns & (~known))
    colors = jnp.where(known, known_color, inferred_color)
    return colors, confident


def _same_color_adjacency(positions, colors, confident):
    card_ids = jnp.arange(9, dtype=jnp.int32)
    pair_mask = card_ids[:, None] < card_ids[None, :]
    delta = positions[:, None, :] - positions[None, :, :]
    dist = jnp.sum(jnp.abs(delta), axis=-1)
    same = colors[:, None] == colors[None, :]
    pair = pair_mask & confident[:, None] & confident[None, :] & same
    return pair & (dist == 1), pair, dist


def _groups_connected(positions, colors, confident):
    adjacent, _, _ = _same_color_adjacency(positions, colors, confident)
    connected = []
    for color in range(3):
        in_color = confident & (colors == color)
        count = jnp.sum(in_color.astype(jnp.int32))
        adj_count = jnp.sum((adjacent & in_color[:, None] & in_color[None, :]).astype(jnp.int32))
        connected.append((count == 3) & (adj_count >= 2))
    return jnp.all(jnp.stack(connected))


def _choose_observe(obs, colors, confident):
    del colors
    card_ids = jnp.arange(9, dtype=jnp.int32)
    legal = obs.legal_actions[1:10]
    partner_seen = obs.node_features[:9, 9].astype(jnp.int32) > 0
    unknown = ~confident
    score = (
        unknown.astype(jnp.int32) * 100
        + (~partner_seen).astype(jnp.int32) * 10
        - card_ids
    )
    masked = jnp.where(legal, score, -1000000)
    return (1 + jnp.argmax(masked)).astype(jnp.int32)


def _board_score(positions, colors, confident):
    adjacent, pair, dist = _same_color_adjacency(positions, colors, confident)
    same_adj = jnp.sum(adjacent.astype(jnp.int32))
    distance_penalty = jnp.sum(jnp.where(pair, dist, 0).astype(jnp.int32))
    center_penalty = jnp.sum(jnp.abs(positions - 4).astype(jnp.int32))
    connected_count = []
    for color in range(3):
        in_color = confident & (colors == color)
        count = jnp.sum(in_color.astype(jnp.int32))
        adj_count = jnp.sum((adjacent & in_color[:, None] & in_color[None, :]).astype(jnp.int32))
        connected_count.append((count == 3) & (adj_count >= 2))
    full_groups = jnp.sum(jnp.stack(connected_count).astype(jnp.int32))
    return full_groups * 1000 + same_adj * 60 - distance_penalty * 4 - center_penalty


def _choose_move(obs, colors, confident):
    card_ids = jnp.arange(9, dtype=jnp.int32)
    move_offsets = jnp.arange(729, dtype=jnp.int32)
    move_cards = move_offsets // 81
    move_cells = move_offsets % 81
    move_dests = jnp.stack((move_cells // 9, move_cells % 9), axis=1)
    legal = obs.legal_actions[10:739]
    positions = obs.positions.astype(jnp.int32)
    moved = positions[None, :, :]
    dests = move_dests[:, None, :]
    is_moved_card = card_ids[None, :, None] == move_cards[:, None, None]
    candidate_positions = jnp.where(is_moved_card, dests, moved)

    def score_one(pos):
        return _board_score(pos, colors, confident)

    scores = jax.vmap(score_one)(candidate_positions)
    mover_known = confident[move_cards].astype(jnp.int32)
    current_adj, _, _ = _same_color_adjacency(positions, colors, confident)
    current_card_adj = jnp.sum((current_adj | current_adj.T).astype(jnp.int32), axis=1)
    isolation_bonus = jnp.maximum(0, 2 - current_card_adj[move_cards])
    scores = scores + mover_known * (3 + isolation_bonus * 4)
    masked = jnp.where(legal, scores, -1000000000)
    return (10 + jnp.argmax(masked)).astype(jnp.int32)


def _choose_hint(obs, colors, confident):
    place_offsets = jnp.arange(324, dtype=jnp.int32)
    place_hints = place_offsets // 81
    place_cells = place_offsets % 81
    place_dests = jnp.stack((place_cells // 9, place_cells % 9), axis=1)
    reveal_legal = obs.legal_actions[739:743]
    reveal_action = (739 + jnp.argmax(reveal_legal)).astype(jnp.int32)

    place_legal = obs.legal_actions[743:1067]
    positions = obs.positions.astype(jnp.int32)
    same_cell = jnp.all(positions[None, :, :] == place_dests[:, None, :], axis=-1)
    occupied = jnp.any(same_cell, axis=1)
    card_at_cell = jnp.argmax(same_cell.astype(jnp.int32), axis=1).astype(jnp.int32)
    hint_colors = obs.node_features[9:13, 3:6].astype(jnp.int32)
    card_colors = colors[card_at_cell]
    match = hint_colors[place_hints, card_colors] > 0
    adjacent, _, _ = _same_color_adjacency(positions, colors, confident)
    card_adj = jnp.sum((adjacent | adjacent.T).astype(jnp.int32), axis=1)
    score = (
        (occupied & confident[card_at_cell] & match).astype(jnp.int32) * 1000
        + confident[card_at_cell].astype(jnp.int32) * 25
        + card_adj[card_at_cell] * 10
        - card_at_cell
        - place_hints * 2
    )
    masked_place = jnp.where(place_legal, score, -1000000)
    place_action = (743 + jnp.argmax(masked_place)).astype(jnp.int32)
    return jnp.where(jnp.any(reveal_legal), reveal_action, place_action).astype(jnp.int32)


def _active_action(obs):
    node_features = obs.node_features
    colors, confident = _card_belief(node_features)
    substep = node_features[0, 11].astype(jnp.int32)
    fallback = _first_legal(obs.legal_actions)

    def substep0(_):
        solved = _groups_connected(obs.positions.astype(jnp.int32), colors, confident)
        can_end = obs.legal_actions[1067] & solved
        return jnp.where(can_end, 1067, _choose_observe(obs, colors, confident)).astype(jnp.int32)

    action = jax.lax.switch(
        substep,
        (
            substep0,
            lambda _: _choose_observe(obs, colors, confident),
            lambda _: _choose_move(obs, colors, confident),
            lambda _: _choose_hint(obs, colors, confident),
        ),
        operand=None,
    )
    return jnp.where(obs.legal_actions[action], action, fallback).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    zero = jnp.asarray(0, dtype=jnp.int32)
    memory = jnp.where(reset, zero, memory)
    active = obs.node_features[0, 10] > 0
    action = jnp.where(active, _active_action(obs), zero).astype(jnp.int32)
    return action, memory, {}
