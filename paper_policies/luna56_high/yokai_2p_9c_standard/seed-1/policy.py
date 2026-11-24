"""A memory-based, convention-free controller for the two-player Yōkai game."""

import jax
import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'color_memory_cluster', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    # Colors are one-hot rows; known says which stable card IDs have been
    # learned by this seat. The final unknown row can be inferred because the
    # deck contains exactly three cards of each color.
    return (jnp.zeros((9, 3), dtype=jnp.int8),
            jnp.zeros((9,), dtype=jnp.bool_),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((9, 2), dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32))


def _choose(mask, values, offset):
    """Choose the highest-valued legal local action, or no-op if none exist."""
    safe_values = jnp.where(mask, values, jnp.asarray(-1000000, dtype=values.dtype))
    local = jnp.argmax(safe_values).astype(jnp.int32)
    return jnp.where(jnp.any(mask), local + jnp.asarray(offset, dtype=jnp.int32),
                     jnp.asarray(0, dtype=jnp.int32))


def _update_memory(obs, memory):
    old_colors, old_known, last_card, last_destination, previous_positions, recent_card = memory
    features = obs.node_features
    visible = jnp.sum(features[:9, :3], axis=1) > 0
    colors = jnp.where(visible[:, None], features[:9, :3], old_colors)
    known = old_known | visible

    # Once eight stable IDs are known, the remaining row is determined by the
    # fixed 3-of-each color multiset. This is independent of the partner's
    # private observation convention.
    raw_remaining = 3 - jnp.sum(colors.astype(jnp.int32), axis=0)
    enough = (jnp.sum(known.astype(jnp.int32)) >= 8) | (jnp.sum(raw_remaining > 0) == 1)
    remaining = jnp.clip(raw_remaining, 0, 1)
    missing = ~known
    inferred_colors = remaining.astype(jnp.int8)[None, :]
    colors = jnp.where((enough & missing)[:, None], inferred_colors, colors)
    known = known | (enough & missing)
    changed = jnp.any(obs.positions != previous_positions, axis=1)
    changed_card = jnp.argmax(changed.astype(jnp.int32)).astype(jnp.int32)
    recent_card = jnp.where(jnp.any(changed), changed_card, recent_card)
    return colors, known, last_card, last_destination, obs.positions, recent_card


def _is_solved(obs, colors, known):
    adjacency = obs.adjacency_matrix[:9, :9].astype(jnp.int32)
    adjacency = adjacency * (1 - jnp.eye(9, dtype=jnp.int32))
    edges_by_color = jnp.sum(adjacency[:, :, None] *
                              (colors[:, None, :] * colors[None, :, :]), axis=(0, 1)) / 2
    connected = edges_by_color >= 2
    return jnp.all(known) & jnp.all(connected)


def _observe_action(obs, known, rng):
    card_ids = jnp.arange(9, dtype=jnp.int32)
    # Unknown cards are preferred; stable IDs provide a shared deterministic
    # inspection order that remains legible to independently generated seats.
    values = (~known).astype(jnp.int32) * 100 + (8 - card_ids)
    return _choose(obs.legal_actions[1:10], values, 1)


def _move_action(obs, colors, known, last_card, recent_card, rng):
    move_ids = jnp.arange(729, dtype=jnp.int32)
    card_ids = move_ids // 81
    cells = move_ids % 81
    destinations = jnp.stack((cells // 9, cells % 9), axis=1)
    positions = obs.positions

    at_destination = jnp.all(destinations[:, None, :] == positions[None, :, :], axis=2)
    destination_adj = at_destination.astype(jnp.int32)
    current_adj = obs.adjacency_matrix[card_ids, :9].astype(jnp.int32)
    pair_same = jnp.sum(colors[card_ids, None, :] * colors[None, :, :], axis=2)
    pair_known = known[card_ids, None] & known[None, :]
    same = pair_same * pair_known.astype(jnp.int32)
    different = (1 - pair_same) * pair_known.astype(jnp.int32)

    new_same = jnp.sum(destination_adj * same, axis=1)
    old_same = jnp.sum(current_adj * same, axis=1)
    new_different = jnp.sum(destination_adj * different, axis=1)
    old_different = jnp.sum(current_adj * different, axis=1)
    delta = (new_same - old_same) * 20 - (new_different - old_different) * 3
    # Prefer moving a card whose color is known, while still allowing a legal
    # move when this seat has not yet observed anything.
    creates_same_edge = new_same > old_same
    # Retain the broad delta objective.  The small random term only resolves
    # exact integer ties, desynchronizing seats without overriding a useful
    # color-adjacency improvement.
    values = delta + known[card_ids].astype(jnp.int32) * 30
    values = values.astype(jnp.float32) + jax.random.uniform(rng, (729,), minval=0.0, maxval=1.9)
    return _choose(obs.legal_actions[10:739], values, 10)


def _hint_action(obs, colors, known):
    legal = obs.legal_actions
    reveal_values = -jnp.arange(4, dtype=jnp.int32)
    reveal = _choose(legal[739:743], reveal_values, 739)

    place_ids = jnp.arange(324, dtype=jnp.int32)
    hints = place_ids // 81
    cells = place_ids % 81
    coordinates = jnp.stack((cells // 9, cells % 9), axis=1)
    matches = jnp.all(coordinates[:, None, :] == obs.positions[None, :, :], axis=2)
    card_ids = jnp.argmax(matches.astype(jnp.int32), axis=1)
    card_colors = colors[card_ids]
    hint_colors = obs.node_features[9:13, 3:6][hints]
    compatible = jnp.sum(card_colors * hint_colors, axis=1)
    pair_same = jnp.sum(card_colors[:, None, :] * colors[None, :, :], axis=2)
    pair_known = known[card_ids, None] & known[None, :]
    same_neighbors = jnp.sum(obs.adjacency_matrix[card_ids, :9].astype(jnp.int32) *
                              pair_same * pair_known.astype(jnp.int32), axis=1)
    values = compatible * 100 + same_neighbors * 10 + known[card_ids].astype(jnp.int32) * 2 - hints
    place = _choose(legal[743:1067], values, 743)
    return jnp.where(jnp.any(legal[739:743]), reveal, place)


def agent_act(obs, reset, memory, rng):
    updated = _update_memory(obs, memory)
    colors, known, last_card, last_destination, previous_positions, recent_card = updated
    substep = obs.node_features[0, 11].astype(jnp.int32)

    observe = _observe_action(obs, known, rng)
    move = _move_action(obs, colors, known, last_card, recent_card, rng)
    hint = _hint_action(obs, colors, known)
    end = jnp.where(obs.legal_actions[1067], jnp.asarray(1067, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32))
    action0 = jnp.where(_is_solved(obs, colors, known) & (end != 0), end, observe)
    action = jnp.where(substep == 0, action0,
                       jnp.where(substep == 1, observe,
                                  jnp.where(substep == 2, move, hint)))
    initial = init_agent()
    is_move = (substep == 2) & (action >= 10) & (action < 739)
    move_card = (action - 10) // 81
    move_cell = (action - 10) % 81
    move_destination = jnp.stack((move_cell // 9, move_cell % 9))
    carried_card = jnp.where(is_move, move_card, last_card)
    carried_destination = jnp.where(is_move, move_destination, last_destination)
    next_memory = (jnp.where(reset, initial[0], colors),
                   jnp.where(reset, initial[1], known),
                   jnp.where(reset, initial[2], carried_card),
                   jnp.where(reset, initial[3], carried_destination),
                   obs.positions.astype(jnp.int32),
                   jnp.where(reset, initial[5], recent_card))
    return action.astype(jnp.int32), next_memory, {}
