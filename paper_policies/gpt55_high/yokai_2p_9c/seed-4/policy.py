"""Deterministic Yokai policy built from local observations only."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'canonical_cluster_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
NUM_HINTS = 4
UNKNOWN = -1


def init_agent():
    return jnp.full((NUM_CARDS,), UNKNOWN, dtype=jnp.int32)


def _first_legal(mask, offset):
    return (jnp.argmax(mask).astype(jnp.int32) + jnp.asarray(offset, dtype=jnp.int32))


def _update_memory(obs, memory, reset):
    base = jnp.where(reset, init_agent(), memory)
    color_bits = obs.node_features[:NUM_CARDS, 0:NUM_COLORS] > 0
    has_color = jnp.any(color_bits, axis=1)
    seen_color = jnp.argmax(color_bits.astype(jnp.int32), axis=1).astype(jnp.int32)
    return jnp.where(has_color, seen_color, base)


def _inferred_colors(known):
    counts = jnp.sum(known[:, None] == jnp.arange(NUM_COLORS, dtype=jnp.int32)[None, :], axis=0)
    missing = 3 - counts
    unknown = known < 0
    missing_kinds = jnp.sum(missing > 0)
    fill_ok = (jnp.sum(unknown) > 0) & (missing_kinds == 1)
    fill_color = jnp.argmax(missing).astype(jnp.int32)
    return jnp.where(unknown & fill_ok, fill_color, known)


def _color_connected(positions, colors, color):
    mask = colors == color
    count = jnp.sum(mask)
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((NUM_CARDS, NUM_CARDS), dtype=bool), k=1)
    pairs = jnp.sum(adjacent & upper & mask[:, None] & mask[None, :])
    return (count == 3) & (pairs >= 2)


def _all_connected(positions, colors):
    complete = jnp.all(colors >= 0)
    c0 = _color_connected(positions, colors, 0)
    c1 = _color_connected(positions, colors, 1)
    c2 = _color_connected(positions, colors, 2)
    return complete & c0 & c1 & c2


def _target_distance_for_positions(positions, colors):
    known = colors >= 0
    target_x = colors + 3
    y = positions[:, 1]
    y_outside = jnp.maximum(jnp.asarray(3, dtype=jnp.int32) - y, y - jnp.asarray(5, dtype=jnp.int32))
    y_dist = jnp.maximum(y_outside, jnp.asarray(0, dtype=jnp.int32))
    dist = jnp.abs(positions[:, 0] - target_x) + y_dist
    return jnp.sum(jnp.where(known, dist, 0))


def _choose_observe(obs, known):
    legal = obs.legal_actions[1:10]
    unlocked = obs.node_features[:NUM_CARDS, 8] == 0
    unknown = known < 0
    partner_unseen = obs.node_features[:NUM_CARDS, 9] == 0
    diversified = legal & unlocked & unknown & partner_unseen
    preferred = legal & unlocked & unknown
    use_diversified = jnp.any(diversified)
    use_preferred = jnp.any(preferred)
    mask = jnp.where(use_diversified, diversified, jnp.where(use_preferred, preferred, legal))
    return _first_legal(mask, 1)


def _choose_move(obs, known):
    colors = _inferred_colors(known)
    positions = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions[10:739]

    action_ids = jnp.arange(729, dtype=jnp.int32)
    card = action_ids // 81
    rem = action_ids - card * 81
    dest = jnp.stack([rem // 9, rem % 9], axis=1).astype(jnp.int32)
    moving_known = colors[card] >= 0
    use_known = jnp.any(legal & moving_known)
    usable = legal & (moving_known | (~use_known))

    card_axis = jnp.arange(NUM_CARDS, dtype=jnp.int32)[None, :, None]
    candidate_positions = jnp.where(
        card_axis == card[:, None, None],
        dest[:, None, :],
        positions[None, :, :],
    )

    delta = jnp.abs(candidate_positions[:, :, None, :] - candidate_positions[:, None, :, :])
    adjacent = jnp.sum(delta, axis=-1) == 1
    upper = jnp.triu(jnp.ones((NUM_CARDS, NUM_CARDS), dtype=bool), k=1)
    same_known = (colors[:, None] == colors[None, :]) & (colors[:, None] >= 0)
    same_pairs = jnp.sum(adjacent & upper[None, :, :] & same_known[None, :, :], axis=(1, 2))

    connected_bonus = jnp.zeros((729,), dtype=jnp.int32)
    for color in range(NUM_COLORS):
        mask = colors == color
        pairs = jnp.sum(adjacent & upper[None, :, :] & mask[None, :, None] & mask[None, None, :], axis=(1, 2))
        complete = jnp.sum(mask) == 3
        connected_bonus = connected_bonus + jnp.where(complete & (pairs >= 2), 1, 0)

    target_x = colors + 3
    known_mask = colors >= 0
    xs = candidate_positions[:, :, 0]
    ys = candidate_positions[:, :, 1]
    y_outside = jnp.maximum(3 - ys, ys - 5)
    y_dist = jnp.maximum(y_outside, 0)
    dist = jnp.abs(xs - target_x[None, :]) + y_dist
    target_dist = jnp.sum(jnp.where(known_mask[None, :], dist, 0), axis=1)

    moved_card_was_settled = colors[card] >= 0
    current_dist = _target_distance_for_positions(positions, colors)
    score = same_pairs * 90 + connected_bonus * 240 - target_dist * 5
    score = score + jnp.where(target_dist < current_dist, 8, 0)
    score = score + jnp.where(moved_card_was_settled, 1, 0)
    score = jnp.where(usable, score, jnp.full((729,), -1000000, dtype=jnp.int32))
    return jnp.argmax(score).astype(jnp.int32) + 10


def _choose_hint(obs, known):
    reveal_legal = obs.legal_actions[739:743]

    def reveal():
        return _first_legal(reveal_legal, 739)

    def place():
        colors = _inferred_colors(known)
        legal = obs.legal_actions[743:1067]
        action_ids = jnp.arange(324, dtype=jnp.int32)
        hint = action_ids // 81
        rem = action_ids - hint * 81
        xy = jnp.stack([rem // 9, rem % 9], axis=1).astype(jnp.int32)
        at_cell = jnp.all(obs.positions[None, :, :] == xy[:, None, :], axis=-1)
        card = jnp.argmax(at_cell.astype(jnp.int32), axis=1)
        card_color = colors[card]
        known_card = card_color >= 0
        hint_bits = obs.node_features[NUM_CARDS + hint, 3:6] > 0
        hint_size = jnp.sum(hint_bits, axis=1).astype(jnp.int32)
        matches = known_card & hint_bits[jnp.arange(324), jnp.maximum(card_color, 0)]

        pos = obs.positions[card]
        target_x = card_color + 3
        y_outside = jnp.maximum(3 - pos[:, 1], pos[:, 1] - 5)
        dist = jnp.abs(pos[:, 0] - target_x) + jnp.maximum(y_outside, 0)
        placement_score = (
            jnp.where(matches, 300, 0)
            + jnp.where(~known_card, hint_size * 12, 0)
            - jnp.where(known_card & (~matches), 300, 0)
            - jnp.where(known_card, dist * 4, 0)
        )
        placement_score = jnp.where(legal, placement_score, jnp.full((324,), -1000000, dtype=jnp.int32))
        return jnp.argmax(placement_score).astype(jnp.int32) + 743

    return jax.lax.cond(jnp.any(reveal_legal), reveal, place)


def agent_act(obs, reset, memory, rng):
    del rng
    next_memory = _update_memory(obs, memory, reset)
    active = obs.node_features[0, 10] > 0
    substep = obs.node_features[0, 11].astype(jnp.int32)
    colors = _inferred_colors(next_memory)
    can_end = obs.legal_actions[1067] & _all_connected(obs.positions.astype(jnp.int32), colors)

    def inactive_action():
        return jnp.asarray(0, dtype=jnp.int32)

    def active_action():
        def step0():
            return jax.lax.cond(can_end, lambda: jnp.asarray(1067, dtype=jnp.int32), lambda: _choose_observe(obs, next_memory))

        branches = (
            step0,
            lambda: _choose_observe(obs, next_memory),
            lambda: _choose_move(obs, next_memory),
            lambda: _choose_hint(obs, next_memory),
        )
        return jax.lax.switch(substep, branches)

    action = jax.lax.cond(active, active_action, inactive_action)
    return action.astype(jnp.int32), next_memory, {}
