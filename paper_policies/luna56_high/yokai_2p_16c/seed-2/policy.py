"""Fixed-shape public-state controller for the Yokai card board."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'public_arm_controller', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return (jnp.full((16,), -1, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros((16,), dtype=jnp.bool_))


def _targets():
    return jnp.asarray([
        [[2, 3], [2, 4], [2, 5], [2, 6]],
        [[3, 7], [4, 7], [5, 7], [6, 7]],
        [[7, 6], [7, 5], [7, 4], [7, 3]],
        [[6, 2], [5, 2], [4, 2], [3, 2]],
    ], dtype=jnp.int32)


def _first_legal(mask, fallback=0):
    return jnp.where(jnp.any(mask), jnp.argmax(mask), jnp.asarray(fallback, dtype=jnp.int32))


def _one_hot(values):
    return values[:, None] == jnp.arange(4, dtype=jnp.int32)[None, :]


def agent_act(obs, reset, memory, rng):
    del rng
    colors, role, turn, moved = memory
    nf = obs.node_features
    legal = obs.legal_actions
    active = nf[0, 12] == 1
    substep = nf[0, 13]
    initial_role = jnp.where(active, jnp.asarray(0, jnp.int32), jnp.asarray(1, jnp.int32))
    role = jnp.where(reset, initial_role, role)
    colors = jnp.where(reset, jnp.full((16,), -1, dtype=jnp.int32), colors)
    turn = jnp.where(reset, jnp.asarray(0, jnp.int32), turn)
    moved = jnp.where(reset, jnp.zeros((16,), dtype=jnp.bool_), moved)

    seen_now = jnp.any(nf[:16, :4] != 0, axis=1)
    observed_color = jnp.argmax(nf[:16, :4], axis=1).astype(jnp.int32)
    colors = jnp.where(seen_now, observed_color, colors)

    # A placed singleton hint is a public, locally verifiable color label:
    # the hint node's adjacency row identifies its locked target card.
    hint_set = nf[16:23, 4:8] != 0
    hint_placed = nf[16:23, 10] == 1
    singleton = hint_placed & (jnp.sum(hint_set, axis=1) == 1)
    singleton_color = jnp.argmax(hint_set, axis=1).astype(jnp.int32)
    hint_edges = obs.adjacency_matrix[16:23, :16] != 0
    public_label = jnp.any(hint_edges.T & singleton[None, :], axis=1)
    public_color = jnp.sum((hint_edges.T & singleton[None, :]) * singleton_color[None, :], axis=1).astype(jnp.int32)
    colors = jnp.where(public_label, public_color, colors)
    known = colors >= 0

    initial_ids = jnp.arange(16, dtype=jnp.int32)
    initial_xy = jnp.stack((3 + initial_ids // 4, 3 + initial_ids % 4), axis=1)
    public_moved = jnp.any(obs.positions != initial_xy, axis=1)
    has_moved = moved | public_moved

    partner_seen = nf[:16, 11] == 1
    unlocked = nf[:16, 10] == 0
    observe_mask = legal[1:17] & unlocked & (~known) & (~partner_seen)
    observe_action = 1 + _first_legal(observe_mask, 0)
    observe_fallback = 1 + _first_legal(legal[1:17] & unlocked, 0)
    observe_action = jnp.where(jnp.any(observe_mask), observe_action, observe_fallback)

    targets = _targets()
    move_ids = jnp.arange(1600, dtype=jnp.int32)
    card_id = move_ids // 100
    x = (move_ids % 100) // 10
    y = move_ids % 10
    move_legal = legal[17:1617]
    card_color = jnp.clip(colors[card_id], 0, 3)
    t = targets[card_color]
    destination_is_target = jnp.any((t[:, :, 0] == x[:, None]) & (t[:, :, 1] == y[:, None]), axis=1)
    occupied = jnp.any((obs.positions[:, 0, None] == x[None, :]) & (obs.positions[:, 1, None] == y[None, :]), axis=0)
    target_empty = ~occupied
    current_is_target = jnp.any((t[:, :, 0] == obs.positions[card_id, 0, None]) & (t[:, :, 1] == obs.positions[card_id, 1, None]), axis=1)
    useful_move = move_legal & known[card_id] & (~has_moved[card_id]) & destination_is_target & target_empty & (~current_is_target)
    other_ids = jnp.arange(16, dtype=jnp.int32)[None, :]
    same_known = (known[None, :] & (colors[None, :] == card_color[:, None]) & (other_ids != card_id[:, None]))
    distance = jnp.abs(obs.positions[None, :, 0] - x[:, None]) + jnp.abs(obs.positions[None, :, 1] - y[:, None])
    near_same = jnp.any(same_known & (distance == 1), axis=1)
    distance_to_same = jnp.min(jnp.where(same_known, distance, 99), axis=1)
    other_target_xy = targets[jnp.clip(colors, 0, 3)]
    other_is_target = jnp.any((other_target_xy[:, :, 0] == obs.positions[:, 0, None]) & (other_target_xy[:, :, 1] == obs.positions[:, 1, None]), axis=1)
    near_stable = jnp.any((same_known & other_is_target[None, :]) & (distance == 1), axis=1)
    stable_useful = useful_move & near_stable
    connected_useful = useful_move & near_same
    chosen_move = jnp.where(jnp.any(stable_useful), stable_useful,
                            jnp.where(jnp.any(connected_useful), connected_useful, useful_move))
    useful_action = 17 + _first_legal(chosen_move, 0)
    fallback_stable = move_legal & known[card_id] & (~has_moved[card_id]) & unlocked[card_id] & near_stable & (~current_is_target)
    fallback_local = move_legal & known[card_id] & (~has_moved[card_id]) & unlocked[card_id] & near_same & (~current_is_target)
    preferred_fallback = move_legal & known[card_id] & (~has_moved[card_id]) & unlocked[card_id]
    fallback_unmoved = move_legal & (~has_moved[card_id]) & unlocked[card_id]
    fallback_mask = jnp.where(jnp.any(fallback_stable), fallback_stable,
                              jnp.where(jnp.any(fallback_local), fallback_local, preferred_fallback))
    fallback_mask = jnp.where(jnp.any(fallback_mask), fallback_mask, fallback_unmoved)
    fallback_mask = jnp.where(jnp.any(fallback_mask), fallback_mask, move_legal)
    fallback_score = jnp.where(fallback_mask, 100 * known[card_id].astype(jnp.int32) - distance_to_same, -10000)
    fallback_move = 17 + jnp.argmax(fallback_score)
    move_action = jnp.where(jnp.any(chosen_move), useful_action, fallback_move)

    hint_revealed = jnp.any(hint_set, axis=1)
    hint_unplaced = nf[16:23, 10] == 0
    hh = jnp.repeat(jnp.arange(7, dtype=jnp.int32), 16)
    cc = jnp.tile(jnp.arange(16, dtype=jnp.int32), 7)
    px = obs.positions[cc, 0]
    py = obs.positions[cc, 1]
    place_ids = hh * 100 + px * 10 + py
    place_index = hh * 100 + px * 10 + py - 0
    place_legal = legal[1624:2324]
    legal_for_pair = place_legal[place_index]
    matches = jnp.any(hint_set[hh] & _one_hot(jnp.clip(colors[cc], 0, 3)), axis=1)
    place_good = legal_for_pair & hint_revealed[hh] & hint_unplaced[hh] & known[cc] & unlocked[cc] & matches
    place_any = legal_for_pair & hint_revealed[hh] & hint_unplaced[hh] & unlocked[cc]
    # Lock an arm card first, so hint placement does not remove a card that
    # the geometric phase still needs to relocate.
    settled_pair = current_is_target[cc]
    good_settled = place_good & settled_pair
    chosen_good = jnp.where(jnp.any(good_settled), good_settled, place_good)
    place_good_action = 1624 + place_ids[_first_legal(chosen_good, 0)]
    place_any_action = 1624 + place_ids[_first_legal(place_any, 0)]
    place_action = jnp.where(jnp.any(place_good), place_good_action, place_any_action)

    reveal_mask = legal[1617:1624]
    reveal_action = 1617 + _first_legal(reveal_mask, 0)
    action = jnp.where(substep < 2, observe_action,
                       jnp.where(substep == 2, move_action,
                                 jnp.where(jnp.any(reveal_mask), reveal_action, place_action)))
    action = jnp.where((substep == 0) & (turn >= 14), jnp.asarray(2324, jnp.int32), action)
    action = jnp.where(active, action, jnp.asarray(0, jnp.int32))
    next_turn = jnp.where(active & (substep == 3), turn + 1, turn)
    is_move = (action >= 17) & (action < 1617)
    selected_card = jnp.clip((action - 17) // 100, 0, 15)
    marked = jnp.arange(16, dtype=jnp.int32) == selected_card
    next_moved = moved | (marked & is_move)
    return action.astype(jnp.int32), (colors, role, next_turn, next_moved), {}
