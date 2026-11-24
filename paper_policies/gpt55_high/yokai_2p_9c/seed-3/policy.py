"""Valid interface-only starter; implement all policy logic here."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _card_colors(node_features):
    color_bits = node_features[:9, :3] > 0
    known = jnp.any(color_bits, axis=1)
    colors = jnp.argmax(color_bits, axis=1).astype(jnp.int32)
    return known, colors


def _observe_action(obs, known):
    legal = obs.legal_actions[1:10]
    partner_seen = obs.node_features[:9, 9] > 0
    card_ids = jnp.arange(9, dtype=jnp.int32)
    score = jnp.where(legal & (~known) & (~partner_seen), card_ids, 1000 + card_ids)
    score = jnp.where(legal & (~known) & partner_seen, 20 + card_ids, score)
    score = jnp.where(legal & known, 40 + card_ids, score)
    return (1 + jnp.argmin(score)).astype(jnp.int32)


def _band_distance(x, y, color):
    target_y = color + jnp.asarray(2, dtype=jnp.int32)
    x_under = jnp.maximum(jnp.asarray(3, dtype=jnp.int32) - x, 0)
    x_over = jnp.maximum(x - jnp.asarray(5, dtype=jnp.int32), 0)
    return jnp.abs(y - target_y) + x_under + x_over


def _block_distance(x, y):
    x_under = jnp.maximum(jnp.asarray(3, dtype=jnp.int32) - x, 0)
    x_over = jnp.maximum(x - jnp.asarray(5, dtype=jnp.int32), 0)
    y_under = jnp.maximum(jnp.asarray(2, dtype=jnp.int32) - y, 0)
    y_over = jnp.maximum(y - jnp.asarray(4, dtype=jnp.int32), 0)
    return x_under + x_over + y_under + y_over


def _move_action(obs, known, colors):
    move_legal = obs.legal_actions[10:739]
    action_offsets = jnp.arange(729, dtype=jnp.int32)
    card = action_offsets // 81
    cell = action_offsets % 81
    dest_x = cell // 9
    dest_y = cell % 9

    pos = obs.positions
    card_color = colors[card]
    cur_dist = _band_distance(pos[card, 0], pos[card, 1], card_color)
    dest_dist = _band_distance(dest_x, dest_y, card_color)
    cur_block_dist = _block_distance(pos[card, 0], pos[card, 1])
    dest_block_dist = _block_distance(dest_x, dest_y)
    improvement = cur_dist - dest_dist
    legal_known = move_legal & known[card]
    improving = legal_known & (improvement > 0)
    unsettled_known = legal_known & (cur_dist > 0)
    unknown_outside = move_legal & (~known[card]) & (cur_block_dist > 0)
    unknown_inside = move_legal & (~known[card]) & (cur_block_dist == 0)

    # Prefer useful moves of known-color cards; if none exist, still make the
    # least-bad legal known-card move required by the substep.
    score = jnp.where(improving, dest_dist * 20 + card, 100000 + action_offsets)
    score = jnp.where((~improving) & unsettled_known, 2000 + dest_dist * 20 + card, score)
    score = jnp.where(unknown_outside, 6000 + dest_block_dist * 20 + action_offsets, score)
    score = jnp.where(unknown_inside, 8000 + dest_block_dist * 20 + action_offsets, score)
    score = jnp.where((~improving) & legal_known & (~unsettled_known), 9000 + dest_dist * 20 + card, score)
    return (10 + jnp.argmin(score)).astype(jnp.int32)


def _hint_action(obs, known, colors):
    reveal_legal = obs.legal_actions[739:743]
    hint_ids = jnp.arange(4, dtype=jnp.int32)
    reveal_scores = jnp.where(reveal_legal, hint_ids, 1000 + hint_ids)
    reveal_action = (739 + jnp.argmin(reveal_scores)).astype(jnp.int32)

    place_legal = obs.legal_actions[743:1067].reshape((4, 9, 9))
    hint_colors = obs.node_features[9:13, 3:6] > 0
    card_pos = obs.positions
    card_dist = _band_distance(card_pos[:, 0], card_pos[:, 1], colors)

    h = jnp.arange(4, dtype=jnp.int32)[:, None]
    c = jnp.arange(9, dtype=jnp.int32)[None, :]
    x = card_pos[c, 0]
    y = card_pos[c, 1]
    legal_at_card = place_legal[h, x, y]
    matches = hint_colors[h, colors[c]] & known[c]
    settled = card_dist[c] == 0
    flat_action = h * 81 + x * 9 + y
    place_score = jnp.where(legal_at_card & matches & settled, flat_action, 100000 + flat_action)
    place_score = jnp.where(legal_at_card & matches & (~settled), 2000 + flat_action, place_score)
    place_score = jnp.where(legal_at_card & (~matches), 9000 + flat_action, place_score)
    any_place = jnp.any(legal_at_card)
    place_choice = jnp.argmin(place_score.reshape(-1))
    place_offset = flat_action.reshape(-1)[place_choice]
    place_action = (743 + place_offset).astype(jnp.int32)
    place_action = jnp.where(any_place, place_action, jnp.asarray(0, dtype=jnp.int32))

    any_reveal = jnp.any(reveal_legal)
    return jnp.where(any_reveal, reveal_action, place_action).astype(jnp.int32)


def _end_ready(obs, known, colors):
    pos = obs.positions
    in_block = (pos[:, 0] >= 3) & (pos[:, 0] <= 5) & (pos[:, 1] >= 2) & (pos[:, 1] <= 4)
    own_correct = (~known) | (pos[:, 1] == (colors + 2))
    seen_by_someone = known | (obs.node_features[:9, 9] > 0)
    return jnp.all(in_block & own_correct & seen_by_someone)


def agent_act(obs, reset, memory, rng):
    fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    active = obs.node_features[0, 10] > 0
    known, colors = _card_colors(obs.node_features)
    substep = obs.node_features[0, 11].astype(jnp.int32)

    observe_action = _observe_action(obs, known)
    move_action = _move_action(obs, known, colors)
    hint_action = _hint_action(obs, known, colors)
    end_action = jnp.asarray(1067, dtype=jnp.int32)

    step0 = jnp.where(obs.legal_actions[1067] & _end_ready(obs, known, colors), end_action, observe_action)
    action = jnp.where(substep == 0, step0, fallback)
    action = jnp.where(substep == 1, observe_action, action)
    action = jnp.where(substep == 2, move_action, action)
    action = jnp.where(substep == 3, hint_action, action)
    action = jnp.where(active, action, fallback)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}
