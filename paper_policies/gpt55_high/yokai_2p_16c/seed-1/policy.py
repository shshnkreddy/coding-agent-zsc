"""Convention-light Yokai policy for the 16-card graph adapter."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'canonical_known_clusterer', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _neg():
    return jnp.asarray(-1_000_000_000, dtype=jnp.int32)


def _card_ids():
    return jnp.arange(16, dtype=jnp.int32)


def _hint_ids():
    return jnp.arange(7, dtype=jnp.int32)


def _grid_xy():
    grid = jnp.arange(10, dtype=jnp.int32)
    return grid[:, None], grid[None, :]


def _zone_masks():
    xx, yy = _grid_xy()
    return jnp.stack([
        ((xx >= 3) & (xx <= 4) & (yy >= 3) & (yy <= 4)),
        ((xx >= 5) & (xx <= 6) & (yy >= 3) & (yy <= 4)),
        ((xx >= 3) & (xx <= 4) & (yy >= 5) & (yy <= 6)),
        ((xx >= 5) & (xx <= 6) & (yy >= 5) & (yy <= 6)),
    ]).astype(jnp.bool_)


def _zone_centers():
    return jnp.asarray([[3, 3], [6, 3], [3, 6], [6, 6]], dtype=jnp.int32)


def _card_facts(obs):
    features = obs.node_features
    colors_oh = features[:16, 0:4] > 0
    known = jnp.any(colors_oh, axis=1)
    color = jnp.argmax(colors_oh.astype(jnp.int32), axis=1).astype(jnp.int32)
    locked = features[:16, 10] > 0
    partner_seen = features[:16, 11] > 0
    pos = obs.positions.astype(jnp.int32)
    x = pos[:, 0]
    y = pos[:, 1]
    zone = jnp.where(
        (x >= 3) & (x <= 4) & (y >= 3) & (y <= 4),
        0,
        jnp.where(
            (x >= 5) & (x <= 6) & (y >= 3) & (y <= 4),
            1,
            jnp.where(
                (x >= 3) & (x <= 4) & (y >= 5) & (y <= 6),
                2,
                jnp.where((x >= 5) & (x <= 6) & (y >= 5) & (y <= 6), 3, -1),
            ),
        ),
    ).astype(jnp.int32)
    return colors_oh, known, color, locked, partner_seen, pos, zone


def _choose_observe(obs):
    _, known, _, locked, partner_seen, _, _ = _card_facts(obs)
    legal = obs.legal_actions[1:17]
    scores = jnp.where(legal, jnp.asarray(1000, jnp.int32), _neg())
    scores = scores + jnp.where(~known, 400, 0)
    scores = scores + jnp.where(~partner_seen, 40, 0)
    scores = scores + jnp.where(~locked, 10, 0)
    scores = scores - _card_ids()
    return (1 + jnp.argmax(scores)).astype(jnp.int32)


def _can_end(obs):
    _, known, color, _, _, _, zone = _card_facts(obs)
    known_count = jnp.sum(known.astype(jnp.int32))
    all_in_zones = jnp.all(zone >= 0)
    color_axis = jnp.arange(4, dtype=jnp.int32)[:, None]
    zone_counts = jnp.sum((zone[None, :] == color_axis).astype(jnp.int32), axis=1)
    color_counts = jnp.sum((known[None, :] & (color[None, :] == color_axis)).astype(jnp.int32), axis=1)
    unknown_by_zone = jnp.sum(((~known)[None, :] & (zone[None, :] == color_axis)).astype(jnp.int32), axis=1)
    known_correct = (~known) | (zone == color)
    missing_by_color = 4 - color_counts
    inferred_complete = jnp.all(zone_counts == 4) & jnp.all(unknown_by_zone == missing_by_color)
    return (
        obs.legal_actions[2324]
        & (known_count >= 12)
        & all_in_zones
        & jnp.all(known_correct)
        & inferred_complete
    )


def _choose_step0(obs):
    return jnp.where(_can_end(obs), jnp.asarray(2324, dtype=jnp.int32), _choose_observe(obs))


def _choose_move(obs):
    _, known, color, locked, _, pos, zone = _card_facts(obs)
    del pos
    in_own_zone = known & (zone == color)
    should_sort = known & (~locked) & (~in_own_zone)
    should_stage = should_sort & (zone >= 0)

    move_legal = obs.legal_actions[17:1617].reshape((16, 10, 10))
    zone_masks = _zone_masks()
    target_masks = zone_masks[color]
    non_zone = ~jnp.any(zone_masks, axis=0)

    xx, yy = _grid_xy()
    center = _zone_centers()[color]
    dist_to_center = jnp.abs(xx[None, :, :] - center[:, 0, None, None]) + jnp.abs(
        yy[None, :, :] - center[:, 1, None, None]
    )
    card_penalty = _card_ids()[:, None, None] * 1000
    xy_penalty = xx[None, :, :] * 10 + yy[None, :, :]

    target_scores = jnp.where(
        move_legal & should_sort[:, None, None] & target_masks,
        200_000 - card_penalty - dist_to_center * 20 - xy_penalty,
        _neg(),
    )

    staging_scores = jnp.where(
        move_legal & should_stage[:, None, None] & non_zone[None, :, :],
        150_000 - card_penalty - dist_to_center * 20 - xy_penalty,
        _neg(),
    )

    blocker = (~locked) & (~in_own_zone) & (zone >= 0)
    blocker_scores = jnp.where(
        move_legal & blocker[:, None, None] & non_zone[None, :, :],
        100_000 - card_penalty - xy_penalty,
        _neg(),
    )

    fallback_scores = jnp.where(
        move_legal,
        1_000 + jnp.where(~in_own_zone[:, None, None], 500, 0) - card_penalty - xy_penalty,
        _neg(),
    )

    scores = jnp.maximum(jnp.maximum(target_scores, staging_scores), jnp.maximum(blocker_scores, fallback_scores))
    flat = jnp.argmax(scores.reshape((1600,))).astype(jnp.int32)
    return (17 + flat).astype(jnp.int32)


def _choose_hint(obs):
    _, known, color, locked, _, pos, zone = _card_facts(obs)
    reveal_legal = obs.legal_actions[1617:1624]
    hint_ids = _hint_ids()
    reveal_scores = jnp.where(reveal_legal, 1000 - hint_ids, _neg())
    reveal_action = (1617 + jnp.argmax(reveal_scores)).astype(jnp.int32)
    can_reveal = jnp.any(reveal_legal)

    hint_colors = obs.node_features[16:23, 4:8] > 0
    place_legal = obs.legal_actions[1624:2324].reshape((7, 10, 10))
    xx, yy = _grid_xy()
    card_at_cell = (xx[None, :, :] == pos[:, 0, None, None]) & (
        yy[None, :, :] == pos[:, 1, None, None]
    )
    hint_matches_card = jnp.take_along_axis(
        hint_colors, color[None, :], axis=1
    )
    correct_known_cell = jnp.any(
        card_at_cell[None, :, :, :]
        & known[None, :, None, None]
        & (~locked)[None, :, None, None]
        & hint_matches_card[:, :, None, None],
        axis=1,
    )
    zone_hint_cell = jnp.any(
        _zone_masks()[None, :, :, :]
        & hint_colors[:, :, None, None],
        axis=1,
    )
    stable_known_cell = jnp.any(
        card_at_cell[None, :, :, :]
        & known[None, :, None, None]
        & (zone == color)[None, :, None, None],
        axis=1,
    )

    h_penalty = hint_ids[:, None, None] * 1000
    xy_penalty = xx[None, :, :] * 10 + yy[None, :, :]
    place_scores = jnp.where(
        place_legal & correct_known_cell & stable_known_cell,
        50_000 - h_penalty - xy_penalty,
        _neg(),
    )
    place_scores = jnp.maximum(
        place_scores,
        jnp.where(place_legal & correct_known_cell, 30_000 - h_penalty - xy_penalty, _neg()),
    )
    place_scores = jnp.maximum(
        place_scores,
        jnp.where(place_legal & zone_hint_cell, 10_000 - h_penalty - xy_penalty, _neg()),
    )
    place_scores = jnp.maximum(
        place_scores,
        jnp.where(place_legal, 1_000 - h_penalty - xy_penalty, _neg()),
    )
    place_flat = jnp.argmax(place_scores.reshape((700,))).astype(jnp.int32)
    place_action = (1624 + place_flat).astype(jnp.int32)
    return jnp.where(can_reveal, reveal_action, place_action)


def agent_act(obs, reset, memory, rng):
    del rng
    memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    active = obs.node_features[0, 12] > 0
    substep = obs.node_features[0, 13].astype(jnp.int32)
    action = jnp.where(
        substep == 0,
        _choose_step0(obs),
        jnp.where(
            substep == 1,
            _choose_observe(obs),
            jnp.where(substep == 2, _choose_move(obs), _choose_hint(obs)),
        ),
    )
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32)).astype(jnp.int32)
    trace = {
        "substep": substep,
        "known_count": jnp.sum(jnp.any(obs.node_features[:16, 0:4] > 0, axis=1).astype(jnp.int32)),
        "can_end": _can_end(obs),
    }
    return action, memory, trace
