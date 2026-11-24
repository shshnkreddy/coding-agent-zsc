"""Partner-readable deterministic policy for the 16-card Yokai task."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'canonical_zone_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


NUM_ACTIONS = 2325
END_GAME = 2324
NEG = -1_000_000

X_MIN_VALUES = (2, 4, 2, 4)
X_MAX_VALUES = (3, 5, 3, 5)
Y_MIN_VALUES = (3, 3, 5, 5)
Y_MAX_VALUES = (4, 4, 6, 6)


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _masked_argmax(scores, legal):
    masked = jnp.where(legal, scores, NEG)
    return jnp.argmax(masked).astype(jnp.int32)


def _target_distance(color, x, y):
    x_min = jnp.asarray(X_MIN_VALUES, dtype=jnp.int32)
    x_max = jnp.asarray(X_MAX_VALUES, dtype=jnp.int32)
    y_min = jnp.asarray(Y_MIN_VALUES, dtype=jnp.int32)
    y_max = jnp.asarray(Y_MAX_VALUES, dtype=jnp.int32)
    xmin = x_min[color]
    xmax = x_max[color]
    ymin = y_min[color]
    ymax = y_max[color]
    dx = jnp.maximum(jnp.maximum(xmin - x, x - xmax), 0)
    dy = jnp.maximum(jnp.maximum(ymin - y, y - ymax), 0)
    return dx + dy


def _in_target_for_color(color, x, y):
    return _target_distance(color, x, y) == 0


def _in_canonical_layout(x, y):
    return (x >= 2) & (x <= 5) & (y >= 3) & (y <= 6)


def _zone_color_at(x, y):
    top = y <= 4
    left = x <= 3
    return jnp.where(top & left, 0, jnp.where(top, 1, jnp.where(left, 2, 3))).astype(jnp.int32)


def _card_state(obs):
    nf = obs.node_features
    colors = nf[:16, :4].astype(jnp.int32)
    known = jnp.sum(colors, axis=1) > 0
    color = jnp.argmax(colors, axis=1).astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    locked = nf[:16, 10].astype(jnp.int32) > 0
    partner_seen = nf[:16, 11].astype(jnp.int32) > 0
    return colors, known, color, pos, locked, partner_seen


def _should_end(obs):
    _, known, color, pos, _, _ = _card_state(obs)
    x = pos[:, 0]
    y = pos[:, 1]
    in_layout = _in_canonical_layout(x, y)
    zone_color = _zone_color_at(x, y)
    known_agrees = (~known) | (color == zone_color)
    return jnp.all(in_layout & known_agrees) & obs.legal_actions[END_GAME]


def _choose_observe(obs):
    _, known, _, _, locked, partner_seen = _card_state(obs)
    card_ids = jnp.arange(16, dtype=jnp.int32)
    action_ids = 1 + card_ids
    legal = obs.legal_actions[action_ids]
    unknown = ~known
    score = (
        unknown.astype(jnp.int32) * 1000
        + (~partner_seen).astype(jnp.int32) * 100
        + (~locked).astype(jnp.int32) * 20
        - card_ids
    )
    best_card = _masked_argmax(score, legal)
    return (best_card + 1).astype(jnp.int32)


def _choose_move(obs):
    _, known, color, pos, _, _ = _card_state(obs)
    actions = jnp.arange(NUM_ACTIONS, dtype=jnp.int32)
    move = actions - 17
    card = jnp.clip(move // 100, 0, 15)
    rem = move - card * 100
    dest_x = jnp.clip(rem // 10, 0, 9)
    dest_y = jnp.clip(rem - dest_x * 10, 0, 9)

    c_color = color[card]
    c_known = known[card]
    cur_x = pos[card, 0]
    cur_y = pos[card, 1]

    cur_dist = _target_distance(c_color, cur_x, cur_y)
    dest_dist = _target_distance(c_color, dest_x, dest_y)
    improves = cur_dist - dest_dist
    dest_own_zone = _in_target_for_color(c_color, dest_x, dest_y)
    cur_own_zone = _in_target_for_color(c_color, cur_x, cur_y)

    cur_in_layout = _in_canonical_layout(cur_x, cur_y)
    dest_in_layout = _in_canonical_layout(dest_x, dest_y)

    known_score = (
        20_000
        + improves * 700
        + dest_own_zone.astype(jnp.int32) * 4_000
        - dest_dist * 250
        - cur_own_zone.astype(jnp.int32) * 30_000
    )
    unknown_score = (
        dest_in_layout.astype(jnp.int32) * 500
        - cur_in_layout.astype(jnp.int32) * 250
        - jnp.abs(dest_x - 3) * 5
        - jnp.abs(dest_y - 4) * 5
    )
    score = jnp.where(c_known, known_score, unknown_score) - card

    legal = obs.legal_actions & (actions >= 17) & (actions < 1617)
    return _masked_argmax(score, legal)


def _choose_hint(obs):
    nf = obs.node_features
    actions = jnp.arange(NUM_ACTIONS, dtype=jnp.int32)
    hint_sets = nf[16:23, 4:8].astype(jnp.int32)
    revealed = jnp.sum(hint_sets, axis=1) > 0
    placed = nf[16:23, 10].astype(jnp.int32) > 0

    reveal_h = jnp.arange(7, dtype=jnp.int32)
    reveal_actions = 1617 + reveal_h
    reveal_legal = obs.legal_actions[reveal_actions]
    hint_size_priority = jnp.asarray([1, 1, 2, 2, 2, 3, 3], dtype=jnp.int32)
    reveal_score = hint_size_priority * 100 - reveal_h
    best_reveal_h = _masked_argmax(reveal_score, reveal_legal)
    best_reveal = best_reveal_h + 1617
    any_reveal = jnp.any(reveal_legal & (~revealed))

    _, known, color, pos, _, _ = _card_state(obs)
    place = actions - 1624
    hint = jnp.clip(place // 100, 0, 6)
    rem = place - hint * 100
    x = jnp.clip(rem // 10, 0, 9)
    y = jnp.clip(rem - x * 10, 0, 9)

    matches = (pos[:, 0][None, :] == x[:, None]) & (pos[:, 1][None, :] == y[:, None])
    card = jnp.argmax(matches.astype(jnp.int32), axis=1).astype(jnp.int32)
    c_known = known[card]
    c_color = color[card]
    contains = hint_sets[hint, c_color] > 0
    card_in_target = _in_target_for_color(c_color, pos[card, 0], pos[card, 1])
    h_size = hint_size_priority[hint]

    place_score = (
        contains.astype(jnp.int32) * 8_000
        + c_known.astype(jnp.int32) * 2_000
        + card_in_target.astype(jnp.int32) * 500
        + h_size * 50
        - hint
        - card
    )
    place_legal = obs.legal_actions & (actions >= 1624) & (actions < 2324) & revealed[hint] & (~placed[hint])
    best_place = _masked_argmax(place_score, place_legal)
    return jnp.where(any_reveal, best_reveal, best_place).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    substep = obs.node_features[0, 13].astype(jnp.int32)
    active = obs.node_features[0, 12].astype(jnp.int32) > 0

    def step0(_):
        return jnp.where(_should_end(obs), jnp.asarray(END_GAME, dtype=jnp.int32), _choose_observe(obs))

    branches = (
        step0,
        lambda _: _choose_observe(obs),
        lambda _: _choose_move(obs),
        lambda _: _choose_hint(obs),
    )
    chosen = jax.lax.switch(substep, branches, operand=None)
    action = jnp.where(active, chosen, jnp.asarray(0, dtype=jnp.int32)).astype(jnp.int32)
    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    trace = {
        "substep": substep,
        "known_count": jnp.sum(obs.node_features[:16, :4]).astype(jnp.int32),
        "should_end": _should_end(obs),
    }
    return action, next_memory, trace
