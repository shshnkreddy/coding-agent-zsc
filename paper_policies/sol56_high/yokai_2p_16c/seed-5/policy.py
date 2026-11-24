"""Locally legible greedy Yōkai policy.

The policy remembers exact observations and cautious inferences from a partner's
move. It clusters known equal-colour cards without a private role assignment,
and only ends when its accumulated view proves the board solved.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "colors": jnp.zeros((16, 4), dtype=jnp.int8),
        "exact": jnp.zeros((16,), dtype=jnp.bool_),
        "positions": jnp.zeros((16, 2), dtype=jnp.int32),
        "turns": jnp.asarray(0, dtype=jnp.int32),
        "last_substep": jnp.asarray(3, dtype=jnp.int32),
    }


def _solved(colors, known, adjacency):
    """Whether four fully-known colour classes are each connected."""
    all_connected = jnp.asarray(True)
    labels = jnp.argmax(colors, axis=1)
    for color in range(4):
        mask = known & (labels == color)
        start = jax.nn.one_hot(jnp.argmax(mask), 16, dtype=jnp.bool_) & mask
        reached = start
        for _ in range(15):
            reached = reached | (jnp.any(adjacency & reached[None, :], axis=1) & mask)
        all_connected = all_connected & (jnp.sum(mask) == 4) & jnp.all(reached | ~mask)
    return jnp.all(known) & all_connected


def _constrained_beliefs(colors, exact, adjacency, allowed):
    """Choose the most connectivity-consistent completion of <=4 unknowns."""
    codes = jnp.arange(256, dtype=jnp.int32)
    powers = jnp.asarray([1, 4, 16, 64], dtype=jnp.int32)
    digits = (codes[:, None] // powers[None, :]) % 4
    unknown = ~exact
    rank = jnp.clip(jnp.cumsum(unknown.astype(jnp.int32)) - 1, 0, 3)
    exact_labels = jnp.argmax(colors, axis=1)
    labels = jnp.where(exact[None, :], exact_labels[None, :], digits[:, rank])

    counts = jnp.stack([jnp.sum(labels == c, axis=1) for c in range(4)], axis=1)
    card_ids = jnp.arange(16, dtype=jnp.int32)[None, :]
    compatible = allowed[card_ids, labels]
    valid = (jnp.all(counts == 4, axis=1)
             & jnp.all(compatible | exact[None, :], axis=1)
             & (jnp.sum(exact) >= 12))
    same = labels[:, :, None] == labels[:, None, :]
    graph = adjacency[None, :, :] & same
    reach = graph | jnp.eye(16, dtype=jnp.bool_)[None, :, :]
    for _ in range(15):
        reach = reach | (jnp.matmul(reach.astype(jnp.int8),
                                    reach.astype(jnp.int8)) > 0)
    connected_pairs = jnp.sum(
        reach & ~jnp.eye(16, dtype=jnp.bool_)[None, :, :], axis=(1, 2))
    same_edges = jnp.sum(graph, axis=(1, 2))
    prior_labels = jnp.argmax(colors, axis=1)
    had_prior = jnp.sum(colors, axis=1) > 0
    agrees = jnp.sum((labels == prior_labels[None, :])
                     & unknown[None, :] & had_prior[None, :], axis=1)
    score = connected_pairs * 100 + same_edges * 5 + agrees
    best = jnp.argmax(jnp.where(valid, score, -100000))
    completion = jax.nn.one_hot(labels[best], 4, dtype=jnp.int8)
    use = (jnp.sum(exact) >= 12) & jnp.any(valid)
    beliefs = jnp.where((use & unknown)[:, None], completion, colors)
    all_valid_solved = (jnp.any(valid)
                        & jnp.all((connected_pairs == 48) | ~valid))
    return beliefs, all_valid_solved


def _observe_action(obs, known, colors):
    legal = obs.legal_actions[1:17]
    partner_seen = obs.node_features[:16, 11] > 0
    belief_known = jnp.sum(colors, axis=1) > 0
    priority = ((~known).astype(jnp.int32) * 100
                + (~belief_known).astype(jnp.int32) * 20
                + (~partner_seen).astype(jnp.int32) * 5
                + (15 - jnp.arange(16, dtype=jnp.int32)))
    return (1 + jnp.argmax(jnp.where(legal, priority, -100000))).astype(jnp.int32)


def _move_action(obs, colors, exact):
    positions = obs.positions
    known = jnp.sum(colors, axis=1) > 0
    labels = jnp.argmax(colors, axis=1)
    same = ((labels[:, None] == labels[None, :])
            & known[:, None] & known[None, :]
            & ~jnp.eye(16, dtype=jnp.bool_))

    current_dist = jnp.sum(jnp.abs(positions[:, None, :] - positions[None, :, :]), axis=2)
    current_adj = current_dist == 1
    base_pair_distance = jnp.sum(jnp.where(same, current_dist, 0)) // 2
    old_edges = jnp.sum(current_adj & same, axis=1)
    old_distance = jnp.sum(jnp.where(same, current_dist, 0), axis=1)

    # For each possible removed card, compute connected components among the
    # other same-colour cards.  This rewards joining two components much more
    # than adding a redundant edge inside one component.
    remove = jnp.arange(16)[:, None]
    keep_pair = ((jnp.arange(16)[None, :] != remove)[:, :, None]
                 & (jnp.arange(16)[None, None, :] != remove[:, None, :]))
    graph = current_adj[None, :, :] & same[None, :, :] & keep_pair
    reach = graph | jnp.eye(16, dtype=jnp.bool_)[None, :, :]
    for _ in range(15):
        reach = reach | (jnp.matmul(reach.astype(jnp.int8),
                                    reach.astype(jnp.int8)) > 0)
    connected_pairs_without = (jnp.sum(reach & ~jnp.eye(16, dtype=jnp.bool_)[None, :, :],
                                       axis=(1, 2)) // 2)

    move_index = jnp.arange(1600, dtype=jnp.int32)
    card = move_index // 100
    cell = move_index % 100
    destination = jnp.stack((cell // 10, cell % 10), axis=1)
    distance = jnp.sum(jnp.abs(destination[:, None, :] - positions[None, :, :]), axis=2)
    same_for_move = same[card]
    new_edges = jnp.sum((distance == 1) & same_for_move, axis=1)
    new_distance = jnp.sum(jnp.where(same_for_move, distance, 0), axis=1)
    new_neighbours = (distance == 1) & same_for_move
    joined = jnp.any(reach[card] & new_neighbours[:, None, :], axis=2)
    joined_count = jnp.sum(joined, axis=1)
    connected_pairs = connected_pairs_without[card] + joined_count
    total_pair_distance = base_pair_distance - old_distance[card] + new_distance
    other_card = jnp.arange(16)[None, :] != card[:, None]
    board_contacts = jnp.sum((distance == 1) & other_card, axis=1)
    center_distance = (jnp.abs(destination[:, 0] * 2 - 9)
                       + jnp.abs(destination[:, 1] * 2 - 9))

    score = (connected_pairs * 1200 + new_edges * 80 - total_pair_distance * 4
             + known[card].astype(jnp.int32) * 100
             + exact[card].astype(jnp.int32) * 300
             - old_edges[card] * 3
             - center_distance)
    legal = obs.legal_actions[17:1617]
    return (17 + jnp.argmax(jnp.where(legal, score, -10000000))).astype(jnp.int32)


def _hint_action(obs, colors, exact):
    reveal_legal = obs.legal_actions[1617:1624]
    any_reveal = jnp.any(reveal_legal)
    reveal = (1617 + jnp.argmax(reveal_legal)).astype(jnp.int32)

    place_index = jnp.arange(700, dtype=jnp.int32)
    hint = place_index // 100
    cell = place_index % 100
    destination = jnp.stack((cell // 10, cell % 10), axis=1)
    at_card = jnp.all(destination[:, None, :] == obs.positions[None, :, :], axis=2)
    card = jnp.argmax(at_card, axis=1)
    card_known = exact[card]
    hint_colors = obs.node_features[16:23, 4:8] > 0
    matches = jnp.any(hint_colors[hint] & (colors[card] > 0), axis=1) & card_known
    hint_size = jnp.sum(hint_colors[hint], axis=1)

    dist = jnp.sum(jnp.abs(obs.positions[:, None, :] - obs.positions[None, :, :]), axis=2)
    labels = jnp.argmax(colors, axis=1)
    same_adj = jnp.sum((dist == 1) & (labels[:, None] == labels[None, :])
                       & exact[:, None] & exact[None, :], axis=1)
    partner_unseen = ~(obs.node_features[:16, 11] > 0)
    place_score = (matches.astype(jnp.int32) * 1000
                   + card_known.astype(jnp.int32) * 100
                   + same_adj[card] * 20
                   + partner_unseen[card].astype(jnp.int32) * 100
                   + (partner_unseen[card].astype(jnp.int32)
                      * (4 - hint_size) * 50)
                   - place_index // 20)
    place_legal = obs.legal_actions[1624:2324]
    place = (1624 + jnp.argmax(jnp.where(place_legal, place_score, -10000000))).astype(jnp.int32)
    safe = place_legal & matches & (same_adj[card] > 0)
    safe_place = (1624 + jnp.argmax(jnp.where(safe, place_score, -10000000))).astype(jnp.int32)
    return jnp.where(jnp.any(safe), safe_place, jnp.where(any_reveal, reveal, place))


def agent_act(obs, reset, memory, rng):
    del rng
    initial = init_agent()
    colors = jnp.where(reset, initial["colors"], memory["colors"])
    exact = jnp.where(reset, initial["exact"], memory["exact"])
    old_positions = jnp.where(reset, obs.positions, memory["positions"])
    turns = jnp.where(reset, initial["turns"], memory["turns"])
    last_substep = jnp.where(reset, initial["last_substep"], memory["last_substep"])

    features = obs.node_features[:16, :4]
    visible = jnp.sum(features, axis=1) > 0
    colors = jnp.where(visible[:, None], features, colors)
    exact = exact | visible

    # A placed one-colour hint is public, exact color information when players
    # use hints truthfully.  This is useful with independently written partners
    # and does not require decoding a private signal.
    hint_features = obs.node_features[16:23]
    hint_colors = hint_features[:, 4:8]
    singleton = jnp.sum(hint_colors, axis=1) == 1
    placed = hint_features[:, 10] > 0
    same_position = jnp.all(obs.positions[:, None, :] == hint_features[None, :, 8:10], axis=2)
    placed_on_card = same_position & placed[None, :]
    has_placed_hint = jnp.any(placed_on_card, axis=1)
    hinted_colors = jnp.any(
        placed_on_card[:, :, None] & (hint_colors > 0)[None, :, :], axis=1)
    allowed = jnp.where(has_placed_hint[:, None], hinted_colors,
                        jnp.ones((16, 4), dtype=jnp.bool_))
    public_signal = jnp.sum(
        jnp.where((same_position & singleton[None, :] & placed[None, :])[:, :, None],
                  hint_colors[None, :, :], 0), axis=1).astype(jnp.int8)
    has_public_signal = jnp.sum(public_signal, axis=1) == 1
    colors = jnp.where(has_public_signal[:, None], public_signal, colors)
    exact = exact | has_public_signal

    moved = jnp.any(obs.positions != old_positions, axis=1)
    moved_card = jnp.argmax(moved)
    dist = jnp.sum(jnp.abs(obs.positions[moved_card] - obs.positions), axis=1)
    neighbour_votes = jnp.sum(jnp.where((dist == 1)[:, None], colors, 0), axis=0)
    inferred_color = jnp.argmax(neighbour_votes)
    unambiguous = (jnp.sum(neighbour_votes > 0) == 1) & (jnp.max(neighbour_votes) > 0)
    exact_neighbour_votes = jnp.sum(
        jnp.where(((dist == 1) & exact)[:, None], colors, 0), axis=0)
    exact_support = exact_neighbour_votes[inferred_color] > 0
    inferred = jax.nn.one_hot(inferred_color, 4, dtype=jnp.int8)
    can_infer = (jnp.any(moved) & ~exact[moved_card]
                 & unambiguous & exact_support)
    colors = colors.at[moved_card].set(jnp.where(can_infer, inferred, colors[moved_card]))

    # The deck contains exactly four of every colour.  Once all still-unknown
    # cards must share the sole deficient colour, their identities are exact.
    exact_counts = jnp.sum(jnp.where(exact[:, None], colors, 0), axis=0)
    incomplete = exact_counts < 4
    sole_deficit = jnp.sum(incomplete) == 1
    missing_color = jnp.argmax(incomplete)
    forced = jax.nn.one_hot(missing_color, 4, dtype=jnp.int8)
    force_mask = ~exact & sole_deficit
    colors = jnp.where(force_mask[:, None], forced, colors)
    exact = exact | force_mask

    colors, all_valid_solved = _constrained_beliefs(
        colors, exact, obs.adjacency_matrix[:16, :16] > 0, allowed)

    substep = obs.node_features[0, 13].astype(jnp.int32)
    active = obs.node_features[0, 12] > 0
    new_turn = active & (substep == 0) & ((last_substep != 0) | reset)
    turns = turns + new_turn.astype(jnp.int32)
    solved = _solved(colors, jnp.sum(colors, axis=1) > 0,
                     obs.adjacency_matrix[:16, :16] > 0)
    placed_count = jnp.sum(obs.node_features[16:23, 10] > 0)

    observe = _observe_action(obs, exact, colors)
    confident_time = ((placed_count >= 6)
                      | ((placed_count >= 5) & all_valid_solved))
    end_now = solved & confident_time & obs.legal_actions[2324]
    first_action = jnp.where(end_now, jnp.asarray(2324, jnp.int32), observe)
    move = _move_action(obs, colors, exact)
    hint = _hint_action(obs, colors, exact)
    proposed = jax.lax.switch(substep, (lambda: first_action, lambda: observe,
                                        lambda: move, lambda: hint))
    fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    action = jnp.where(active & obs.legal_actions[proposed], proposed, fallback)

    next_memory = {
        "colors": colors,
        "exact": exact,
        "positions": obs.positions,
        "turns": turns,
        "last_substep": substep,
    }
    trace = {
        "known": jnp.sum(jnp.sum(colors, axis=1) > 0).astype(jnp.int32),
        "exact": jnp.sum(exact).astype(jnp.int32),
        "solved": solved,
        "turns": turns,
    }
    return action, next_memory, trace
