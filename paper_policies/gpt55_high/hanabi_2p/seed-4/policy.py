"""Convention-light Hanabi policy using literal public evidence."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_safe_hints', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NUM_COLORS = 5
NUM_RANKS = 5
HAND_SIZE = 5
NUM_CARD_TYPES = NUM_COLORS * NUM_RANKS
EPS = 1.0e-6


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _first_with_score(scores, valid):
    masked = jnp.where(valid, scores, -1.0e9)
    index = jnp.argmax(masked).astype(jnp.int32)
    any_valid = jnp.any(valid)
    return index, any_valid


def _firework_heights(features):
    ranks = jnp.arange(NUM_RANKS, dtype=jnp.float32)
    fw = features[167:192].reshape((NUM_COLORS, NUM_RANKS))
    return jnp.sum(fw * (ranks + 1.0), axis=1).astype(jnp.int32)


def _discard_counts(features):
    disc = features[203:253].reshape((NUM_COLORS, 10))
    return jnp.stack(
        [
            jnp.sum(disc[:, 0:3], axis=1),
            jnp.sum(disc[:, 3:5], axis=1),
            jnp.sum(disc[:, 5:7], axis=1),
            jnp.sum(disc[:, 7:9], axis=1),
            jnp.sum(disc[:, 9:10], axis=1),
        ],
        axis=1,
    )


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions

    heights = _firework_heights(features)
    tokens = jnp.sum(features[192:200])
    lives = jnp.sum(features[200:203])
    discards = _discard_counts(features)
    colors = jnp.arange(NUM_COLORS, dtype=jnp.int32)
    ranks = jnp.arange(NUM_RANKS, dtype=jnp.int32)
    card_indices = jnp.arange(NUM_CARD_TYPES, dtype=jnp.int32)
    card_colors = card_indices // NUM_RANKS
    card_ranks = card_indices % NUM_RANKS
    multiplicities = jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0], dtype=jnp.float32)
    played = (ranks[None, :] < heights[:, None]).astype(jnp.float32)
    remaining = multiplicities[None, :] - discards - played

    playable_type = (card_ranks == heights[card_colors]) & (heights[card_colors] < NUM_RANKS)
    useless_type = card_ranks < heights[card_colors]
    critical_type = (card_ranks >= heights[card_colors]) & (remaining.reshape((NUM_CARD_TYPES,)) <= 1.0 + EPS)

    partner = features[0:125].reshape((HAND_SIZE, NUM_COLORS, NUM_RANKS))
    partner_occ = jnp.sum(partner, axis=(1, 2)) > EPS
    partner_flat = partner.reshape((HAND_SIZE, NUM_CARD_TYPES))
    partner_counts = jnp.sum(partner_flat, axis=0)
    adjusted_counts = jnp.maximum(remaining.reshape((NUM_CARD_TYPES,)) - partner_counts, 0.0)

    knowledge = features[308:658].reshape((2, HAND_SIZE, 35))
    own_v0 = knowledge[0, :, 0:25]
    own_hinted = jnp.sum(knowledge[0, :, 25:35], axis=1) > 0.5
    own_occ = jnp.sum(own_v0, axis=1) > EPS
    own_possible = own_v0 > EPS
    adjusted_weights = own_possible.astype(jnp.float32) * adjusted_counts[None, :]
    adjusted_weight_sum = jnp.sum(adjusted_weights, axis=1, keepdims=True)
    adjusted_v0 = jnp.where(
        adjusted_weight_sum > EPS,
        adjusted_weights / jnp.maximum(adjusted_weight_sum, EPS),
        own_v0,
    )
    play_prob = jnp.sum(adjusted_v0 * playable_type.astype(jnp.float32)[None, :], axis=1)
    useless_prob = jnp.sum(adjusted_v0 * useless_type.astype(jnp.float32)[None, :], axis=1)
    critical_prob = jnp.sum(adjusted_v0 * critical_type.astype(jnp.float32)[None, :], axis=1)

    slot_order = (HAND_SIZE - jnp.arange(HAND_SIZE, dtype=jnp.float32)) * 0.001
    strong_hinted_play = own_hinted & (play_prob >= 0.90) & (lives > 2.5)
    certain_play = own_occ & ((play_prob >= 1.0 - EPS) | strong_hinted_play) & legal[5:10]
    play_slot, has_play = _first_with_score(play_prob + slot_order, certain_play)
    play_action = play_slot + jnp.asarray(5, dtype=jnp.int32)

    partner_type = jnp.argmax(partner_flat, axis=1).astype(jnp.int32)
    partner_color = partner_type // NUM_RANKS
    partner_rank = partner_type % NUM_RANKS
    partner_playable = (
        partner_occ
        & (partner_rank == heights[partner_color])
        & (heights[partner_color] < NUM_RANKS)
    )
    partner_know = knowledge[1]
    partner_color_known = partner_know[jnp.arange(HAND_SIZE), 25 + partner_color] > 0.5
    partner_rank_known = partner_know[jnp.arange(HAND_SIZE), 30 + partner_rank] > 0.5

    slot_colors = partner_color[:, None] == colors[None, :]
    slot_ranks = partner_rank[:, None] == ranks[None, :]
    complete_color_slots = partner_playable & partner_rank_known & (~partner_color_known)
    complete_rank_slots = partner_playable & partner_color_known & (~partner_rank_known)
    initial_slots = partner_playable & (~partner_color_known) & (~partner_rank_known)

    color_complete = jnp.sum(slot_colors * complete_color_slots[:, None], axis=0)
    rank_complete = jnp.sum(slot_ranks * complete_rank_slots[:, None], axis=0)

    rank_public_remaining = remaining > EPS
    rank_globally_playable = jnp.all(
        (heights[:, None] == ranks[None, :]) | (~rank_public_remaining),
        axis=0,
    )
    rank_direct = jnp.sum(slot_ranks * initial_slots[:, None], axis=0) * rank_globally_playable.astype(jnp.float32)
    color_initial = jnp.sum(slot_colors * initial_slots[:, None], axis=0)

    hint_scores = jnp.concatenate(
        [
            jnp.where(color_complete > 0.0, 400.0 + color_complete, 0.0)
            + jnp.where(color_initial > 0.0, 40.0 + color_initial, 0.0),
            jnp.where(rank_complete > 0.0, 500.0 + rank_complete, 0.0)
            + jnp.where(rank_direct > 0.0, 300.0 + rank_direct, 0.0),
        ],
        axis=0,
    )
    hint_legal = legal[10:20] & (tokens > 0.5) & (hint_scores > 0.0)
    hint_index, has_hint = _first_with_score(hint_scores, hint_legal)
    hint_action = hint_index + jnp.asarray(10, dtype=jnp.int32)

    certain_discard = own_occ & (useless_prob >= 1.0 - EPS) & legal[0:5]
    discard_slot, has_certain_discard = _first_with_score(useless_prob + slot_order, certain_discard)
    discard_action = discard_slot.astype(jnp.int32)

    discard_scores = useless_prob - 0.75 * critical_prob + slot_order
    fallback_discard_slot, has_fallback_discard = _first_with_score(discard_scores, legal[0:5] & own_occ)
    fallback_discard_action = fallback_discard_slot.astype(jnp.int32)

    fallback_hint_index, has_fallback_hint = _first_with_score(jnp.ones((10,), dtype=jnp.float32), legal[10:20] & (tokens > 0.5))
    fallback_hint_action = fallback_hint_index + jnp.asarray(10, dtype=jnp.int32)
    first_legal = jnp.argmax(legal).astype(jnp.int32)

    active_action = jnp.where(has_fallback_discard, fallback_discard_action, first_legal)
    active_action = jnp.where(has_fallback_hint & (~has_fallback_discard), fallback_hint_action, active_action)
    active_action = jnp.where(has_certain_discard, discard_action, active_action)
    active_action = jnp.where(has_hint, hint_action, active_action)
    active_action = jnp.where(has_play, play_action, active_action)

    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), active_action)
    zero = jnp.asarray(0, dtype=jnp.int32)
    trace = {
        'play_prob_max': jnp.max(play_prob),
        'useless_prob_max': jnp.max(useless_prob),
        'hint_score_max': jnp.max(hint_scores),
        'chosen_action': action,
    }
    return action, jnp.where(reset, zero, memory), trace
