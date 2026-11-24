"""A conservative, belief-guided Hanabi policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'belief_guided_safe', 'adapter': 'hanabi', 'adapter_version': 1,
    'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5,
    'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local',
}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _discard_counts(features):
    # Discard subblocks have lengths 3, 2, 2, 2, 1 for ranks 0..4.
    pieces = []
    for color in range(5):
        start = 203 + 10 * color
        pieces.extend((jnp.sum(features[start:start + 3]),
                       jnp.sum(features[start + 3:start + 5]),
                       jnp.sum(features[start + 5:start + 7]),
                       jnp.sum(features[start + 7:start + 9]),
                       jnp.sum(features[start + 9:start + 10])))
    return jnp.stack(pieces, axis=0)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions
    colors = jnp.repeat(jnp.arange(5, dtype=jnp.int32), 5)
    ranks = jnp.tile(jnp.arange(5, dtype=jnp.int32), 5)

    fireworks = features[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks * (jnp.arange(5, dtype=jnp.float32) + 1.0), axis=1)

    # The other hand is public and can be used to select a safe hint.
    partner = features[0:125].reshape((5, 5, 5))
    partner_occupied = jnp.sum(partner, axis=(1, 2)) > 0.5
    partner_color = jnp.argmax(jnp.sum(partner, axis=2), axis=1)
    partner_rank = jnp.argmax(jnp.sum(partner, axis=1), axis=1)
    values = jnp.arange(5)
    color_matches = partner_occupied[:, None] & (partner_color[:, None] == values[None, :])
    rank_matches = partner_occupied[:, None] & (partner_rank[:, None] == values[None, :])
    color_counts = jnp.sum(color_matches, axis=0)
    rank_counts = jnp.sum(rank_matches, axis=0)
    partner_native = features[308:658].reshape((2, 5, 35))[1]
    partner_belief = partner_native[:, 0:25]
    playable_cards = ranks == heights[colors]
    color_card_masks = colors[None, :] == values[:, None]
    rank_card_masks = ranks[None, :] == values[:, None]
    color_mass = jnp.sum(partner_belief[:, None, :] * color_card_masks[None, :, :], axis=2)
    rank_mass = jnp.sum(partner_belief[:, None, :] * rank_card_masks[None, :, :], axis=2)
    color_safe_mass = jnp.sum(partner_belief[:, None, :] * color_card_masks[None, :, :]
                              * playable_cards[None, None, :], axis=2)
    rank_safe_mass = jnp.sum(partner_belief[:, None, :] * rank_card_masks[None, :, :]
                             * playable_cards[None, None, :], axis=2)
    color_posterior_safe = jnp.where(color_mass > 0.0, color_safe_mass / color_mass, 0.0)
    rank_posterior_safe = jnp.where(rank_mass > 0.0, rank_safe_mass / rank_mass, 0.0)
    color_actionable = color_matches & (color_posterior_safe >= 0.999)
    rank_actionable = rank_matches & (rank_posterior_safe >= 0.999)
    color_actionable_counts = jnp.sum(color_actionable, axis=0)
    rank_actionable_counts = jnp.sum(rank_actionable, axis=0)
    color_posterior_max = jnp.max(jnp.where(color_matches, color_posterior_safe, 0.0), axis=0)
    rank_posterior_max = jnp.max(jnp.where(rank_matches, rank_posterior_safe, 0.0), axis=0)
    known_color = partner_native[:, 25:30] > 0.5
    known_rank = partner_native[:, 30:35] > 0.5
    new_color_counts = jnp.sum(color_matches & ~known_color, axis=0)
    new_rank_counts = jnp.sum(rank_matches & ~known_rank, axis=0)
    color_good = color_actionable_counts > 0
    rank_good = rank_actionable_counts > 0
    # Safe hints dominate; within that class prefer a hint that is not already
    # recorded on the recipient's native knowledge block.
    color_score = jnp.where(color_good, 1000 * color_actionable_counts
                            + 10 * new_color_counts + color_counts,
                            -jnp.ones_like(color_counts))
    rank_score = jnp.where(rank_good, 1000 * rank_actionable_counts
                           + 10 * new_rank_counts + rank_counts,
                          -jnp.ones_like(rank_counts))
    safe_color = jnp.argmax(color_score)
    safe_rank = jnp.argmax(rank_score)
    has_safe_hint = jnp.any(color_good | rank_good)
    safe_hint = jnp.where(rank_score[safe_rank] >= color_score[safe_color],
                          15 + safe_rank, 10 + safe_color)
    general_color_score = (100 * color_posterior_max + 200 * new_color_counts
                           + color_counts)
    general_rank_score = (100 * rank_posterior_max + 200 * new_rank_counts
                          + rank_counts)
    general_color = jnp.argmax(general_color_score)
    general_rank = jnp.argmax(general_rank_score)
    general_hint = jnp.where(general_rank_score[general_rank] >= general_color_score[general_color],
                             15 + general_rank, 10 + general_color)
    hint_action = jnp.where(has_safe_hint, safe_hint, general_hint).astype(jnp.int32)
    has_partner_card = jnp.any(partner_occupied)

    # Native V0 probabilities already reflect public hint evidence.
    native = features[308:658].reshape((2, 5, 35))[0]
    belief = native[:, 0:25]
    own_occupied = jnp.sum(belief, axis=1) > 0.0
    dead_counts = _discard_counts(features)
    multiplicities = jnp.asarray([3, 2, 2, 2, 1] * 5, dtype=jnp.float32)
    played_cards = (ranks < heights[colors]).astype(jnp.float32)
    remaining = multiplicities - dead_counts - played_cards
    dead_cards = (ranks < heights[colors]) | (remaining <= 0.0)
    safe_probability = jnp.sum(belief * playable_cards.astype(jnp.float32), axis=1)
    dead_probability = jnp.sum(belief * dead_cards.astype(jnp.float32), axis=1)
    safe_probability = jnp.where(own_occupied, safe_probability, 0.0)
    dead_probability = jnp.where(own_occupied, dead_probability, 0.0)
    play_slot = jnp.argmax(safe_probability)
    discard_slot = jnp.argmax(dead_probability)
    fallback_discard_slot = jnp.argmax(jnp.where(own_occupied,
                                                  dead_probability - safe_probability,
                                                  -2.0))
    play_action = (5 + play_slot).astype(jnp.int32)
    discard_action = discard_slot.astype(jnp.int32)
    fallback_discard_action = fallback_discard_slot.astype(jnp.int32)
    lives = jnp.sum(features[200:203])
    deck_remaining = jnp.sum(features[127:167])
    play_threshold = jnp.where((lives >= 2.999) & (deck_remaining <= 5.0), 0.50, 0.999)
    has_safe_play = jnp.max(safe_probability) >= play_threshold
    has_dead_discard = jnp.max(dead_probability) >= 0.999

    info_tokens = jnp.sum(features[192:200])
    has_tokens = info_tokens > 0.0
    chosen_discard = jnp.where(has_dead_discard, discard_action,
                               fallback_discard_action)
    candidate_hint = jnp.where(has_partner_card, hint_action,
                               jnp.asarray(20, dtype=jnp.int32))
    hint_legal = (candidate_hint >= 0) & (candidate_hint < 21) & legal[candidate_hint]
    play_legal = legal[play_action]
    discard_legal = legal[chosen_discard]
    safe_hint_legal = has_safe_hint & has_tokens & hint_legal
    dead_discard_priority = has_dead_discard & discard_legal & ~safe_hint_legal
    forced_fallback = jnp.where(play_legal, play_action,
                                jnp.argmax(legal.astype(jnp.int32)))
    active_choice = jnp.where(
        has_safe_play & play_legal,
        play_action,
        jnp.where(safe_hint_legal, candidate_hint,
                  jnp.where(dead_discard_priority, chosen_discard,
                            jnp.where(has_tokens & hint_legal, candidate_hint,
                                      jnp.where(discard_legal, chosen_discard,
                                                forced_fallback)))),
    ).astype(jnp.int32)
    inactive = legal[20]
    action = jnp.where(inactive, jnp.asarray(20, dtype=jnp.int32), active_choice)
    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)
    return action, next_memory, {}
