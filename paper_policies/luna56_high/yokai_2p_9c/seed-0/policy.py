"""A convention-oriented policy for the two-player Yokai board."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'canonical_color_regions',
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
    # -1 means that this seat has not personally observed the card.
    return (
        jnp.full((9,), -1, dtype=jnp.int32),
        jnp.full((9, 2), -1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _remember(obs, memory):
    features = obs.node_features[:9, :3]
    visible = jnp.sum(features, axis=1) > 0
    colors = jnp.argmax(features, axis=1).astype(jnp.int32)
    return jnp.where(visible, colors, memory)


def _singleton_hint_colors(obs):
    hint_sets = obs.node_features[9:13, 3:6]
    placed = obs.node_features[9:13, 8] > 0
    singleton = placed & (jnp.sum(hint_sets, axis=1) == 1)
    singleton_color = jnp.argmax(hint_sets, axis=1).astype(jnp.int32)
    card_at_hint = jnp.all(
        obs.positions[:, None, :] == obs.node_features[9:13, 6:8][None, :, :], axis=2
    )
    by_color = []
    for color in range(3):
        by_color.append(
            jnp.any(
                card_at_hint
                & singleton[None, :]
                & (singleton_color[None, :] == color),
                axis=1,
            )
        )
    inferred = jnp.stack(by_color, axis=1)
    return jnp.argmax(inferred, axis=1).astype(jnp.int32), jnp.any(inferred, axis=1)


def _connected_by_known_colors(obs, colors):
    known = jnp.all(colors >= 0)
    adjacency = obs.adjacency_matrix[:9, :9].astype(jnp.int32)
    results = []
    for color in range(3):
        members = colors == color
        seed_index = jnp.argmax(members.astype(jnp.int32))
        reach = jnp.arange(9) == seed_index
        same = (colors[:, None] == colors[None, :]) & members[:, None]
        for _ in range(9):
            reach = reach | jnp.any((adjacency > 0) & same & reach[None, :], axis=1)
        results.append(jnp.all(jnp.where(members, reach, True)))
    return known & jnp.all(jnp.asarray(results))


def _observe_action(obs, colors):
    legal = obs.legal_actions[1:10]
    unseen = colors < 0
    partner_seen = obs.node_features[:9, 9] > 0
    score = legal.astype(jnp.int32) * 10
    score = score + unseen.astype(jnp.int32) * 100
    score = score + (~partner_seen).astype(jnp.int32) * 20
    score = jnp.where(legal, score, jnp.asarray(-1000000, dtype=jnp.int32))
    chosen = jnp.argmax(score).astype(jnp.int32) + 1
    fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    return jnp.where(jnp.any(legal), chosen, fallback)


def _move_action(obs, colors):
    legal = obs.legal_actions[10:739].reshape((9, 9, 9))
    known = colors >= 0
    target_x = 3 + jnp.where(known, colors, 0)
    xs = jnp.arange(9)[None, :, None]
    ys = jnp.arange(9)[None, None, :]
    current_x = obs.positions[:, 0][:, None, None]
    current_y = obs.positions[:, 1][:, None, None]
    target = target_x[:, None, None]
    needs = (current_x != target) | (current_y < 3) | (current_y > 5)
    card_ids = jnp.arange(9)
    same_known = (
        (colors[:, None] == colors[None, :])
        & known[None, :]
        & (card_ids[:, None] != card_ids[None, :])
    )
    dest_x = jnp.arange(9)[None, None, :, None]
    dest_y = jnp.arange(9)[None, None, None, :]
    other_x = obs.positions[None, :, 0, None, None]
    other_y = obs.positions[None, :, 1, None, None]
    adjacent_same = jnp.sum(
        same_known[:, :, None, None]
        & ((jnp.abs(dest_x - other_x) + jnp.abs(dest_y - other_y)) == 1),
        axis=1,
    )
    score = legal.astype(jnp.int32) * 1
    score = score + known[:, None, None].astype(jnp.int32) * 1000
    score = score + needs.astype(jnp.int32) * 500
    score = score + (xs == target).astype(jnp.int32) * 300
    score = score + ((ys >= 3) & (ys <= 5)).astype(jnp.int32) * 100
    score = score + adjacent_same.astype(jnp.int32) * 120
    score = jnp.where(legal, score, jnp.asarray(-1000000, dtype=jnp.int32))
    flat = jnp.argmax(score.reshape((729,)))
    chosen = flat.astype(jnp.int32) + 10
    fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    return jnp.where(jnp.any(legal), chosen, fallback)


def _hint_action(obs, colors):
    place_legal = obs.legal_actions[743:1067].reshape((4, 9, 9))
    hint_sets = obs.node_features[9:13, 3:6] > 0
    revealed = jnp.sum(hint_sets, axis=1) > 0
    unplaced = obs.node_features[9:13, 8] == 0
    known = colors >= 0
    clipped_colors = jnp.clip(colors, 0, 2)
    card_matches = hint_sets[:, clipped_colors] & known[None, :]
    xgrid = jnp.arange(9)[None, :, None]
    ygrid = jnp.arange(9)[None, None, :]
    at_cell = (
        (obs.positions[:, 0, None, None] == xgrid)
        & (obs.positions[:, 1, None, None] == ygrid)
    )
    correct = jnp.any(
        at_cell[None, :, :, :] & card_matches[:, :, None, None], axis=1
    )
    central = (xgrid >= 3) & (xgrid <= 5) & (ygrid >= 3) & (ygrid <= 5)
    correct = correct & revealed[:, None, None] & unplaced[:, None, None]
    already_placed = jnp.any(obs.node_features[9:13, 8] > 0)
    has_reveal = jnp.any(obs.legal_actions[739:743])
    correct = correct
    place_score = correct.astype(jnp.int32) * 1000 + central.astype(jnp.int32) * 100
    place_score = jnp.where(place_legal, place_score, jnp.asarray(-1000000, dtype=jnp.int32))
    place_chosen = jnp.argmax(place_score.reshape((324,))).astype(jnp.int32) + 743
    any_correct = jnp.any(correct & place_legal)
    legal = obs.legal_actions[739:743]
    score = legal.astype(jnp.int32) * 10 - jnp.arange(4, dtype=jnp.int32)
    score = jnp.where(legal, score, jnp.asarray(-1000000, dtype=jnp.int32))
    chosen = jnp.argmax(score).astype(jnp.int32) + 739
    fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    reveal_or_fallback = jnp.where(jnp.any(legal), chosen, fallback)
    return jnp.where(any_correct, place_chosen, reveal_or_fallback)


def _structurally_solved(obs, move_count):
    positions = obs.positions
    central = jnp.all(
        (positions[:, 0] >= 3) & (positions[:, 0] <= 5)
        & (positions[:, 1] >= 3) & (positions[:, 1] <= 5)
    )
    return central & (move_count >= 4)


def agent_act(obs, reset, memory, rng):
    del rng
    old_colors, old_positions, old_move_count = memory
    init_colors, init_positions, init_move_count = init_agent()
    prior_colors = jnp.where(reset, init_colors, old_colors)
    prior_positions = jnp.where(reset, init_positions, old_positions)
    prior_move_count = jnp.where(reset, init_move_count, old_move_count)
    current_positions = obs.positions.astype(jnp.int32)
    changed = (prior_positions[:, 0] >= 0) & jnp.any(current_positions != prior_positions, axis=1)
    move_count = jnp.minimum(prior_move_count + jnp.sum(changed.astype(jnp.int32)), 8)
    active = obs.node_features[0, 10] > 0
    colors = _remember(obs, prior_colors)
    public_colors, public_known = _singleton_hint_colors(obs)
    colors = jnp.where((colors < 0) & public_known, public_colors, colors)
    partner_move = (~active) & changed
    moved_column_known = partner_move & (current_positions[:, 0] >= 3) & (current_positions[:, 0] <= 5)
    moved_colors = current_positions[:, 0] - 3
    colors = jnp.where((colors < 0) & moved_column_known, moved_colors, colors)
    substep = obs.node_features[0, 11].astype(jnp.int32)
    solved = _connected_by_known_colors(obs, colors) | _structurally_solved(obs, move_count)
    end_legal = obs.legal_actions[1067]
    end_or_observe = jnp.where(
        solved & end_legal, jnp.asarray(1067, dtype=jnp.int32), _observe_action(obs, colors)
    )
    observe = _observe_action(obs, colors)
    move = _move_action(obs, colors)
    hint = _hint_action(obs, colors)
    fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    action = jnp.where(substep == 0, end_or_observe, fallback)
    action = jnp.where(substep == 1, observe, action)
    action = jnp.where(substep == 2, move, action)
    action = jnp.where(substep == 3, hint, action)
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32))
    return action.astype(jnp.int32), (colors, current_positions, move_count), {}
