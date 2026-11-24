"""Locally informed, convention-light policy for the 9-card Yokai game."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'local_clusterer', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _colors(obs):
    """Known colors, augmented when the final unknown is forced by counts."""
    onehot = obs.node_features[:9, :3].astype(jnp.int32)
    known = jnp.sum(onehot, axis=1) > 0
    colors = jnp.argmax(onehot, axis=1).astype(jnp.int32)

    # A correctly placed singleton is an unambiguous public disclosure.  Good
    # partners are strongly incentivized to place it truthfully; using it also
    # recovers colors of cards that were locked before this seat could inspect.
    hint_sets = obs.node_features[9:13, 3:6].astype(jnp.int32)
    singleton = (jnp.sum(hint_sets, axis=1) == 1) & (obs.node_features[9:13, 8] > 0)
    links = (obs.adjacency_matrix[:9, 9:13] > 0) & singleton[None, :]
    hinted = jnp.any(links, axis=1)
    public_color_scores = jnp.einsum('ih,hc->ic', links.astype(jnp.int32), hint_sets)
    public_colors = jnp.argmax(public_color_scores, axis=1).astype(jnp.int32)
    colors = jnp.where((~known) & hinted, public_colors, colors)
    known = known | hinted

    counts = jnp.stack([jnp.sum(known & (colors == c)) for c in range(3)])
    missing_color = jnp.argmax(3 - counts).astype(jnp.int32)
    infer = jnp.sum(~known) == 1
    colors = jnp.where((~known) & infer, missing_color, colors)
    known = known | ((~known) & infer)
    return colors, known


def _observe_action(obs, colors, known):
    del colors
    legal_cards = obs.legal_actions[1:10]
    # Learn peripheral cards first: they are generally the safest relocation
    # candidates under the whole-board connectivity constraint.
    degree = jnp.sum(obs.adjacency_matrix[:9, :9] > 0, axis=1) - 1
    partner_unseen = obs.node_features[:9, 9] == 0
    value = (~known).astype(jnp.int32) * 1000 - degree * 100
    value = value + partner_unseen.astype(jnp.int32) * 20
    value = value - jnp.arange(9, dtype=jnp.int32)
    value = jnp.where(legal_cards, value, -10000)
    return (1 + jnp.argmax(value)).astype(jnp.int32)


def _success_probability(obs, colors, known):
    """Uniform posterior success fraction over locally feasible completions."""
    # There are only 3**9 assignments.  XLA constant-folds the radix decode,
    # and the fixed batch avoids data-dependent loops or host callbacks.
    code = jnp.arange(3 ** 9, dtype=jnp.int32)[:, None]
    powers = (3 ** jnp.arange(9, dtype=jnp.int32))[None, :]
    assignments = (code // powers) % 3
    matches = jnp.all((assignments == colors[None, :]) | (~known[None, :]), axis=1)
    hint_sets = obs.node_features[9:13, 3:6].astype(jnp.int32)
    placed = obs.node_features[9:13, 8] > 0
    links = (obs.adjacency_matrix[:9, 9:13] > 0) & placed[None, :]
    attached_sets = jnp.einsum('ih,hc->ic', links.astype(jnp.int32), hint_sets)
    has_hint = jnp.any(links, axis=1)
    assignment_allowed = jnp.take_along_axis(
        attached_sets[None, :, :], assignments[:, :, None], axis=2)[..., 0] > 0
    matches = matches & jnp.all(assignment_allowed | (~has_hint[None, :]), axis=1)
    balanced = jnp.all(jnp.stack([
        jnp.sum(assignments == c, axis=1) == 3 for c in range(3)
    ], axis=1), axis=1)
    feasible = matches & balanced

    edges = jnp.triu(obs.adjacency_matrix[:9, :9] > 0, 1)
    color_edges = jnp.stack([
        jnp.sum(edges[None, :, :] &
                (assignments[:, :, None] == c) &
                (assignments[:, None, :] == c), axis=(1, 2))
        for c in range(3)
    ], axis=1)
    success = jnp.all(color_edges >= 2, axis=1)
    feasible_count = jnp.sum(feasible)
    successful_count = jnp.sum(feasible & success)
    return successful_count.astype(jnp.float32) / jnp.maximum(feasible_count, 1)


def _move_action(obs, colors, known):
    legal = obs.legal_actions[10:739].reshape(9, 9, 9)
    card = jnp.arange(9, dtype=jnp.int32)[:, None, None]
    dx = jnp.arange(9, dtype=jnp.int32)[None, :, None]
    dy = jnp.arange(9, dtype=jnp.int32)[None, None, :]

    old = obs.positions.astype(jnp.int32)
    # Candidate positions: [moving card, destination x, destination y,
    # resulting card id, coordinate].
    base = jnp.broadcast_to(old[None, None, None, :, :], (9, 9, 9, 9, 2))
    moved = jnp.stack((jnp.broadcast_to(dx, (9, 9, 9)),
                       jnp.broadcast_to(dy, (9, 9, 9))), axis=-1)
    mask = jnp.arange(9, dtype=jnp.int32)[None, None, None, :] == card[..., None]
    pos = jnp.where(mask[..., None], moved[..., None, :], base)

    delta = jnp.abs(pos[..., :, None, :] - pos[..., None, :, :])
    dist = jnp.sum(delta, axis=-1)
    pair = jnp.triu(jnp.ones((9, 9), dtype=jnp.bool_), 1)

    # Posterior P(same color) under the fixed 3/3/3 multiset.  This preserves
    # valuable unknown adjacency rather than treating masked cards as noise.
    known_counts = jnp.stack([jnp.sum(known & (colors == c)) for c in range(3)])
    remaining = 3 - known_counts
    n_unknown = jnp.sum(~known)
    safe_n = jnp.maximum(n_unknown, 1)
    p_ku = remaining[colors].astype(jnp.float32) / safe_n.astype(jnp.float32)
    uu_num = jnp.sum(remaining * (remaining - 1)).astype(jnp.float32)
    uu_den = jnp.maximum(n_unknown * (n_unknown - 1), 1).astype(jnp.float32)
    p_uu = uu_num / uu_den
    both_known = known[:, None] & known[None, :]
    one_known = known[:, None] ^ known[None, :]
    known_end_color = jnp.where(known[:, None], colors[:, None], colors[None, :])
    prob = jnp.where(both_known,
                     (colors[:, None] == colors[None, :]).astype(jnp.float32),
                     jnp.where(one_known, remaining[known_end_color] / safe_n, p_uu))
    prob = prob * pair
    adjacent = jnp.sum((dist == 1) * prob, axis=(-2, -1))

    # Pull known equal colors together even before a legal adjacency gain is
    # available. The small canonical x-band term only resolves ties.
    close = jnp.sum(jnp.maximum(0, 7 - dist) * prob, axis=(-2, -1))
    band_error = jnp.sum(jnp.abs(pos[..., :, 0] - (3 + colors)) * known,
                         axis=-1).astype(jnp.int32)
    compact = jnp.sum(dist * pair, axis=(-2, -1)).astype(jnp.int32)
    moving_known = known[:, None, None].astype(jnp.int32)
    travel = jnp.abs(dx - old[:, 0, None, None]) + jnp.abs(dy - old[:, 1, None, None])
    score = adjacent * 10000.0 + close * 30.0 - band_error * 4.0 - compact
    score = score + moving_known * 3 - travel
    score = jnp.where(legal, score, -100000000)
    return (10 + jnp.argmax(score)).astype(jnp.int32)


def _hint_action(obs, colors, known):
    hint_sets = obs.node_features[9:13, 3:6].astype(jnp.int32)
    revealed = jnp.sum(hint_sets, axis=1) > 0

    reveal_legal = obs.legal_actions[739:743]
    reveal_value = (~revealed).astype(jnp.int32) * 10 - jnp.arange(4, dtype=jnp.int32)
    reveal_value = jnp.where(reveal_legal, reveal_value, -10000)
    reveal_action = (739 + jnp.argmax(reveal_value)).astype(jnp.int32)

    place_legal = obs.legal_actions[743:1067].reshape(4, 9, 9)
    xs = jnp.arange(9, dtype=jnp.int32)[None, :, None]
    ys = jnp.arange(9, dtype=jnp.int32)[None, None, :]
    at = ((obs.positions[:, 0, None, None] == xs) &
          (obs.positions[:, 1, None, None] == ys))
    target = jnp.argmax(at.astype(jnp.int32), axis=0)
    occupied = jnp.any(at, axis=0)
    target_known = known[target]
    compatible = hint_sets[:, colors[target]] > 0

    card_dist = jnp.sum(jnp.abs(obs.positions[:, None, :] - obs.positions[None, :, :]), axis=-1)
    same_neighbor = jnp.sum((card_dist == 1) &
                            (colors[:, None] == colors[None, :]) &
                            known[:, None] & known[None, :], axis=1)
    stable = same_neighbor[target]
    color_counts = jnp.stack([jnp.sum(known & (colors == c)) for c in range(3)])
    color_edges = jnp.stack([
        jnp.sum(jnp.triu((card_dist == 1) &
                         (colors[:, None] == c) & (colors[None, :] == c) &
                         known[:, None] & known[None, :], 1))
        for c in range(3)
    ])
    complete_color = (color_counts == 3) & (color_edges >= 2)
    # Compatibility dominates; lock cards already contributing to a cluster.
    score = compatible.astype(jnp.int32) * target_known.astype(jnp.int32) * 1000
    score = score + complete_color[colors[target]].astype(jnp.int32) * 300
    score = score + stable * 30 + target_known.astype(jnp.int32) * 10
    score = score - jnp.arange(4, dtype=jnp.int32)[:, None, None]
    score = score - target
    score = jnp.where(place_legal & occupied[None, :, :], score, -100000)
    place_action = (743 + jnp.argmax(score)).astype(jnp.int32)

    any_reveal = jnp.any(reveal_legal)
    return jnp.where(any_reveal, reveal_action, place_action).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    active = obs.node_features[0, 10] > 0
    substep = obs.node_features[0, 11].astype(jnp.int32)
    colors, known = _colors(obs)

    observe_card = _observe_action(obs, colors, known)
    success_probability = _success_probability(obs, colors, known)
    placed_count = jnp.sum(obs.node_features[9:13, 8] > 0)
    threshold = jnp.where(placed_count == 2, 0.49, 1.0)
    value_end = (success_probability >= threshold) & obs.legal_actions[1067]
    observe = jnp.where(value_end, jnp.asarray(1067, dtype=jnp.int32), observe_card)
    move = _move_action(obs, colors, known)
    hint = _hint_action(obs, colors, known)
    choices = jnp.stack((observe, observe, move, hint))
    chosen = choices[substep]
    action = jnp.where(active, chosen, jnp.asarray(0, dtype=jnp.int32))
    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    trace = {
        'known_count': jnp.sum(known).astype(jnp.int32),
        'substep': substep,
    }
    return action.astype(jnp.int32), next_memory, trace
