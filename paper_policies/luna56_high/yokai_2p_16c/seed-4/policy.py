"""A public-geometry protocol for the two-player Yokai board."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'canonical_zone_protocol', 'adapter': 'yokai', 'adapter_version': 1,
    'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10,
    'num_hints': 7, 'num_nodes': 23, 'node_features': 15,
    'num_actions': 2325, 'policy_structure': 'shared_local',
}


def init_agent():
    return (-jnp.ones((16,), dtype=jnp.int8),
            jnp.zeros((16,), dtype=jnp.bool_),
            jnp.asarray(0, dtype=jnp.int32))


def _zone_distance(x, y, color):
    d0 = jnp.abs(x - 2) + jnp.maximum(0, 3 - y) + jnp.maximum(0, y - 6)
    d1 = jnp.abs(x - 7) + jnp.maximum(0, 3 - y) + jnp.maximum(0, y - 6)
    d2 = jnp.abs(y - 2) + jnp.maximum(0, 3 - x) + jnp.maximum(0, x - 6)
    d3 = jnp.abs(y - 7) + jnp.maximum(0, 3 - x) + jnp.maximum(0, x - 6)
    return jnp.where(color == 0, d0, jnp.where(color == 1, d1,
        jnp.where(color == 2, d2, d3)))


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.node_features
    legal = obs.legal_actions
    active = features[0, 12] > 0
    substep = features[0, 13]
    old_known = jnp.where(reset, -jnp.ones((16,), dtype=jnp.int8), memory[0])
    old_moved = jnp.where(reset, jnp.zeros((16,), dtype=jnp.bool_), memory[1])
    old_turn = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory[2])

    color_bits = features[:16, :4]
    visible = jnp.sum(color_bits, axis=1) > 0
    seen_color = jnp.argmax(color_bits, axis=1).astype(jnp.int8)
    known = jnp.where(visible, seen_color, old_known)

    # A private observation overrides the public convention.  A card in an
    # exact side/top/bottom zone carries a color label for the other seat.
    x0, y0 = obs.positions[:, 0], obs.positions[:, 1]
    in0 = (x0 == 2) & (y0 >= 3) & (y0 <= 6)
    in1 = (x0 == 7) & (y0 >= 3) & (y0 <= 6)
    in2 = (y0 == 2) & (x0 >= 3) & (x0 <= 6)
    in3 = (y0 == 7) & (x0 >= 3) & (x0 <= 6)
    inferred = jnp.where(in0, 0, jnp.where(in1, 1, jnp.where(in2, 2, 3)))
    in_zone = in0 | in1 | in2 | in3
    effective = jnp.where(known >= 0, known, jnp.where(in_zone, inferred, -1))
    wrong_zone = in_zone & (known >= 0) & (known != inferred)
    processed = in_zone & (~wrong_zone)

    other_seen = features[:16, 11] > 0
    unlocked = features[:16, 10] == 0
    obs_legal = legal[1:17]
    fresh = (~visible) & (~other_seen) & (~in_zone) & unlocked & obs_legal
    fallback_fresh = (~visible) & (~in_zone) & unlocked & obs_legal
    zoned_unknown = (~visible) & in_zone & unlocked & obs_legal
    obs_score = (fresh.astype(jnp.int32) * 1000 +
                 fallback_fresh.astype(jnp.int32) * 100 +
                 zoned_unknown.astype(jnp.int32) * 10 +
                 obs_legal.astype(jnp.int32))
    observe_action = 1 + jnp.argmax(obs_score).astype(jnp.int32)

    move_ids = jnp.arange(1600, dtype=jnp.int32)
    move_card = move_ids // 100
    encoded_cell = move_ids % 100
    move_x, move_y = encoded_cell // 10, encoded_cell % 10
    move_legal = legal[17:1617]
    move_color = effective[move_card]
    # Zone occupancy is public, so a partner can recognize that this card has
    # already consumed its protocol move even though moved is seat-private.
    correction_needed = wrong_zone[move_card]
    ready = (move_color >= 0) & ((~old_moved[move_card]) | correction_needed) & (~processed[move_card])
    move_dist = _zone_distance(move_x, move_y, jnp.maximum(move_color, 0))
    source_dist = (jnp.abs(move_x - x0[move_card]) +
                   jnp.abs(move_y - y0[move_card]))
    desired_dist = jnp.where(move_color >= 0, move_dist, source_dist)
    touch0 = (x0[move_card] == 3) & (y0[move_card] >= 3) & (y0[move_card] <= 5)
    touch1 = (x0[move_card] == 6) & (y0[move_card] >= 3) & (y0[move_card] <= 5)
    touch2 = (y0[move_card] == 3) & (x0[move_card] >= 3) & (x0[move_card] <= 5)
    touch3 = (y0[move_card] == 6) & (x0[move_card] >= 3) & (x0[move_card] <= 5)
    touches_zone = jnp.where(move_color == 0, touch0, jnp.where(move_color == 1, touch1,
        jnp.where(move_color == 2, touch2, touch3))) & (move_color >= 0)
    # If the currently exposed cards are already zoned, revisit a known card
    # that failed to reach an exact zone before ever guessing a color for an
    # unknown card.  Unknown fallback moves stay close to the central board.
    reposition = (move_color >= 0) & (~processed[move_card])
    ready_nonboundary = ready & (~touches_zone)
    move_score = (move_legal.astype(jnp.int32) * 1000000 +
                  (ready_nonboundary & move_legal).astype(jnp.int32) * 120000 +
                  (ready & move_legal).astype(jnp.int32) * 100000 +
                  (reposition & move_legal).astype(jnp.int32) * 10000 -
                  desired_dist * move_legal.astype(jnp.int32) * 10)
    move_action = 17 + jnp.argmax(move_score).astype(jnp.int32)
    chosen_move_card = (move_action - 17) // 100
    is_move = substep == 2
    moved = old_moved | ((jnp.arange(16) == chosen_move_card) & is_move & active)

    hint_features = features[16:23, 4:8]
    revealed = jnp.sum(hint_features, axis=1) > 0
    reveal_legal = legal[1617:1624]
    reveal_score = reveal_legal.astype(jnp.int32) * (1 + (~revealed).astype(jnp.int32) * 1000)
    reveal_action = 1617 + jnp.argmax(reveal_score).astype(jnp.int32)

    place_ids = jnp.arange(700, dtype=jnp.int32)
    place_hint = place_ids // 100
    place_cell = place_ids % 100
    place_x, place_y = place_cell // 10, place_cell % 10
    place_legal = legal[1624:2324]
    same_xy = ((obs.positions[:, 0, None] == place_x[None, :]) &
               (obs.positions[:, 1, None] == place_y[None, :]))
    placed_card = jnp.argmax(same_xy.astype(jnp.int32), axis=0)
    card_present = jnp.max(same_xy, axis=0)
    place_color = effective[placed_card]
    hint_set = hint_features[place_hint]
    compatible = ((place_color >= 0) &
                  (hint_set[jnp.arange(700), jnp.maximum(place_color, 0)] > 0))
    hint_size = jnp.sum(hint_set, axis=1)[place_hint]
    place_score = (place_legal.astype(jnp.int32) * 1000000 +
                   (hint_size * place_legal).astype(jnp.int32) * 10 +
                   (compatible & place_legal).astype(jnp.int32) * 100 +
                   (card_present & place_legal).astype(jnp.int32) * 5)
    place_action = 1624 + jnp.argmax(place_score).astype(jnp.int32)
    hint_action = jnp.where(jnp.any(reveal_legal & (~revealed)), reveal_action, place_action)

    end_now = (substep == 0) & (old_turn >= 14) & legal[2324]
    base_action = jnp.where(substep == 0, jnp.where(end_now, 2324, observe_action),
        jnp.where(substep == 1, observe_action,
        jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, base_action, jnp.asarray(0, dtype=jnp.int32))
    turn = old_turn + (active & (substep == 0)).astype(jnp.int32)
    return action.astype(jnp.int32), (known, moved, turn), {}
