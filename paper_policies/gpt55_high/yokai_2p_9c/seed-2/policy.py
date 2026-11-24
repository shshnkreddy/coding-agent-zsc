"""Partner-robust heuristic policy for the 9-card Yokai task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


N_CARDS = 9
N_COLORS = 3


def init_agent():
    return {
        "known": jnp.zeros((N_CARDS, N_COLORS), dtype=jnp.bool_),
    }


def _first_scored(mask, score):
    legal_score = jnp.where(mask, score, -1_000_000)
    return jnp.argmax(legal_score).astype(jnp.int32)


def _target_cost(positions, known, colors):
    x = positions[:, 0]
    y = positions[:, 1]
    target_x = colors + 3
    y_low = jnp.maximum(3 - y, 0)
    y_high = jnp.maximum(y - 5, 0)
    dist = jnp.abs(x - target_x) + y_low + y_high
    return jnp.sum(jnp.where(known, dist, 0))


def _known_connectivity(positions, known, colors):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((N_CARDS, N_CARDS), dtype=jnp.bool_), 1)
    both_known = known[:, None] & known[None, :]
    same = colors[:, None] == colors[None, :]
    same_edges = jnp.sum(adjacent & upper & both_known & same)
    diff_edges = jnp.sum(adjacent & upper & both_known & (~same))
    return same_edges.astype(jnp.int32), diff_edges.astype(jnp.int32)


def _board_score(positions, known_oh):
    known = jnp.any(known_oh, axis=1)
    colors = jnp.argmax(known_oh.astype(jnp.int32), axis=1)
    same_edges, diff_edges = _known_connectivity(positions, known, colors)
    target_cost = _target_cost(positions, known, colors)
    center_cost = jnp.sum(jnp.abs(positions[:, 0] - 4) + jnp.abs(positions[:, 1] - 4))
    return 56 * same_edges - 28 * diff_edges - 3 * target_cost - center_cost


def _all_known_colors_connected(positions, known_oh):
    known = jnp.any(known_oh, axis=1)
    colors = jnp.argmax(known_oh.astype(jnp.int32), axis=1)
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1

    def color_ok(c):
        members = known & (colors == c)
        count = jnp.sum(members.astype(jnp.int32))
        member_adj = adjacent & members[:, None] & members[None, :]
        edges = jnp.sum(jnp.triu(member_adj, 1).astype(jnp.int32))
        return edges >= jnp.maximum(count - 1, 0)

    return color_ok(0) & color_ok(1) & color_ok(2)


def _observe_action(obs, known_oh):
    legal = obs.legal_actions[1:10]
    known = jnp.any(known_oh, axis=1)
    partner_seen = obs.node_features[:N_CARDS, 9] > 0
    ids = jnp.arange(N_CARDS, dtype=jnp.int32)
    primary = legal & (~known) & (~partner_seen)
    secondary = legal & (~known)
    score = jnp.where(primary, 300 - ids, jnp.where(secondary, 200 - ids, 100 - ids))
    return (_first_scored(legal, score) + 1).astype(jnp.int32)


def _move_action(obs, known_oh):
    legal = obs.legal_actions[10:739]
    action_ix = jnp.arange(729, dtype=jnp.int32)
    move_card = action_ix // 81
    cell = action_ix % 81
    dest = jnp.stack([cell // 9, cell % 9], axis=1).astype(jnp.int32)

    positions = obs.positions.astype(jnp.int32)
    cards = jnp.arange(N_CARDS, dtype=jnp.int32)
    trial_positions = jnp.where(
        cards[None, :, None] == move_card[:, None, None],
        dest[:, None, :],
        positions[None, :, :],
    )

    known = jnp.any(known_oh, axis=1)
    colors = jnp.argmax(known_oh.astype(jnp.int32), axis=1)
    moving_known = known[move_card]
    moving_color = colors[move_card]
    destination_x = dest[:, 0]
    canonical_push = jnp.where(moving_known, -1 * jnp.abs(destination_x - (moving_color + 3)), 0)
    leaving_wrong_column = known[move_card] & (positions[move_card, 0] != (moving_color + 3))

    scores = jax.vmap(lambda p: _board_score(p, known_oh))(trial_positions)
    scores = scores + canonical_push + jnp.where(leaving_wrong_column, 4, 0)
    chosen = _first_scored(legal, scores)
    return (chosen + 10).astype(jnp.int32)


def _hint_action(obs, known_oh):
    reveal_legal = obs.legal_actions[739:743]
    reveal_ids = jnp.arange(4, dtype=jnp.int32)
    reveal = (_first_scored(reveal_legal, 20 - reveal_ids) + 739).astype(jnp.int32)

    place_legal = obs.legal_actions[743:1067]
    pix = jnp.arange(324, dtype=jnp.int32)
    hint = pix // 81
    cell = pix % 81
    xy = jnp.stack([cell // 9, cell % 9], axis=1).astype(jnp.int32)
    positions = obs.positions.astype(jnp.int32)
    at_cell = jnp.all(positions[None, :, :] == xy[:, None, :], axis=-1)
    card = jnp.argmax(at_cell.astype(jnp.int32), axis=1)
    known = jnp.any(known_oh, axis=1)
    colors = jnp.argmax(known_oh.astype(jnp.int32), axis=1)
    hint_colors = obs.node_features[9:13, 3:6] > 0
    matches = hint_colors[hint, colors[card]]
    card_known = known[card]
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    same_known = known[:, None] & known[None, :] & (colors[:, None] == colors[None, :])
    same_neighbor_count = jnp.sum((adjacent & same_known).astype(jnp.int32), axis=1)
    score = (
        jnp.where(card_known & matches, 200, 0)
        + jnp.where(card_known & matches, 30 * same_neighbor_count[card], 0)
        + jnp.where(~card_known, 20, 0)
        - pix
    )
    place = (_first_scored(place_legal, score) + 743).astype(jnp.int32)
    return jnp.where(jnp.any(reveal_legal), reveal, place).astype(jnp.int32)


def _end_or_observe(obs, known_oh):
    known = jnp.any(known_oh, axis=1)
    known_count = jnp.sum(known.astype(jnp.int32))
    locally_connected = _all_known_colors_connected(obs.positions.astype(jnp.int32), known_oh)
    confident = (known_count >= 7) & locally_connected
    can_end = obs.legal_actions[1067]
    observe = _observe_action(obs, known_oh)
    return jnp.where(confident & can_end, jnp.asarray(1067, dtype=jnp.int32), observe)


def agent_act(obs, reset, memory, rng):
    del rng
    zero_known = jnp.zeros((N_CARDS, N_COLORS), dtype=jnp.bool_)
    old_known = jnp.where(reset, zero_known, memory["known"])
    active = obs.node_features[0, 10] > 0
    observed_now = obs.node_features[:N_CARDS, :N_COLORS] > 0
    known = jnp.where(active, old_known | observed_now, old_known)
    next_memory = {"known": known}

    substep = obs.node_features[0, 11].astype(jnp.int32)
    active_action = jax.lax.switch(
        substep,
        (
            lambda _: _end_or_observe(obs, known),
            lambda _: _observe_action(obs, known),
            lambda _: _move_action(obs, known),
            lambda _: _hint_action(obs, known),
        ),
        operand=None,
    )
    action = jnp.where(active, active_action, jnp.asarray(0, dtype=jnp.int32))
    return action.astype(jnp.int32), next_memory, {}
