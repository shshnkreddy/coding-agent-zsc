"""Literal, convention-light Hanabi policy for the XP objective."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_xp_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NUM_COLORS = 5
NUM_RANKS = 5
HAND_SIZE = 5
NUM_ACTIONS = 21


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _decode(features):
    partner = features[0:125].reshape((HAND_SIZE, NUM_COLORS, NUM_RANKS))
    fireworks = features[167:192].reshape((NUM_COLORS, NUM_RANKS))
    info = jnp.sum(features[192:200])
    lives = jnp.sum(features[200:203])
    discards_raw = features[203:253].reshape((NUM_COLORS, 10))
    knowledge = features[308:658].reshape((2, HAND_SIZE, 35))
    return partner, fireworks, info, lives, discards_raw, knowledge


def _firework_heights(fireworks):
    rank_values = jnp.arange(1, NUM_RANKS + 1, dtype=jnp.float32)
    return jnp.sum(fireworks * rank_values[None, :], axis=1)


def _discard_counts(discards_raw):
    return jnp.stack(
        [
            jnp.sum(discards_raw[:, 0:3], axis=1),
            jnp.sum(discards_raw[:, 3:5], axis=1),
            jnp.sum(discards_raw[:, 5:7], axis=1),
            jnp.sum(discards_raw[:, 7:9], axis=1),
            discards_raw[:, 9],
        ],
        axis=1,
    )


def _with_scores(scores, actions, action_scores):
    return scores.at[actions].max(action_scores)


def _best_legal_action(scores, legal_actions):
    legal_scores = jnp.where(legal_actions, scores, -1.0e9)
    default_scores = jnp.where(legal_actions, -1.0e8, -1.0e9)
    legal_scores = jnp.maximum(legal_scores, default_scores)
    return jnp.argmax(legal_scores).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions
    partner, fireworks, info, lives, discards_raw, knowledge = _decode(features)

    heights = _firework_heights(fireworks)
    ranks = jnp.arange(NUM_RANKS, dtype=jnp.float32)[None, :]
    playable_type = ranks == heights[:, None]
    dead_type = ranks < heights[:, None]

    own_v0 = knowledge[0, :, 0:25].reshape((HAND_SIZE, NUM_COLORS, NUM_RANKS))
    own_hint_flags = knowledge[0, :, 25:35]
    own_occupied = jnp.sum(own_v0, axis=(1, 2)) > 0.0
    own_protected = jnp.sum(own_hint_flags, axis=1) > 0.0

    discard_counts = _discard_counts(discards_raw)
    multiplicities = jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0], dtype=jnp.float32)
    remaining_public = multiplicities[None, :] - discard_counts - fireworks
    partner_visible_counts = jnp.sum(partner, axis=0)
    adjusted_counts = jnp.maximum(remaining_public - partner_visible_counts, 0.0)
    own_possible = own_v0 > 0.0
    own_weights = own_possible.astype(jnp.float32) * adjusted_counts[None, :, :]
    own_weight_sum = jnp.sum(own_weights, axis=(1, 2), keepdims=True)
    own_belief = jnp.where(own_weight_sum > 0.0, own_weights / own_weight_sum, own_v0)
    own_p_play = jnp.sum(own_belief * playable_type[None, :, :], axis=(1, 2))
    own_p_dead = jnp.sum(own_belief * dead_type[None, :, :], axis=(1, 2))
    critical_type = (remaining_public <= 1.01) & (~dead_type)
    own_p_critical = jnp.sum(own_belief * critical_type[None, :, :], axis=(1, 2))
    own_expected_rank = jnp.sum(
        own_belief * jnp.arange(NUM_RANKS, dtype=jnp.float32)[None, None, :],
        axis=(1, 2),
    )

    partner_slot_has = jnp.sum(partner, axis=(1, 2)) > 0.0
    partner_playable_slot = jnp.sum(partner * playable_type[None, :, :], axis=(1, 2)) > 0.5
    partner_rank = jnp.sum(partner, axis=1) > 0.5
    partner_color = jnp.sum(partner, axis=2) > 0.5
    partner_color_flags = knowledge[1, :, 25:30] > 0.5
    partner_rank_flags = knowledge[1, :, 30:35] > 0.5

    rank_card_count = jnp.sum(partner_rank.astype(jnp.float32), axis=0)
    playable_rank_count = jnp.sum(
        (partner_rank & partner_playable_slot[:, None]).astype(jnp.float32), axis=0
    )
    all_rank_cards_playable = (rank_card_count > 0.0) & (playable_rank_count == rank_card_count)
    new_playable_rank_count = jnp.sum(
        (
            partner_rank
            & partner_playable_slot[:, None]
            & (~partner_rank_flags)
        ).astype(jnp.float32),
        axis=0,
    )

    slot_rank_is_needed = jnp.sum(partner * playable_type[None, :, :], axis=(1, 2)) > 0.5
    rank_complement_targets = (
        partner_rank
        & slot_rank_is_needed[:, None]
        & (~partner_rank_flags)
        & jnp.any(partner_color_flags & partner_color, axis=1)[:, None]
    )
    rank_complement_count = jnp.sum(rank_complement_targets.astype(jnp.float32), axis=0)

    playable_color_targets = partner_color & partner_playable_slot[:, None]
    actual_rank_flags_for_needed = jnp.sum(
        partner * partner_rank_flags[:, None, :].astype(jnp.float32), axis=(1, 2)
    ) > 0.5
    color_complement_targets = (
        playable_color_targets
        & actual_rank_flags_for_needed[:, None]
        & (~partner_color_flags)
    )
    color_complement_count = jnp.sum(color_complement_targets.astype(jnp.float32), axis=0)

    partner_critical = jnp.sum(partner * critical_type[None, :, :], axis=(1, 2)) > 0.5
    unprotected_partner = ~(jnp.any(partner_color_flags, axis=1) | jnp.any(partner_rank_flags, axis=1))
    save_rank_count = jnp.sum((partner_rank & partner_critical[:, None] & unprotected_partner[:, None]).astype(jnp.float32), axis=0)

    scores = jnp.full((NUM_ACTIONS,), -1.0e9, dtype=jnp.float32)

    slot_ids = jnp.arange(HAND_SIZE)
    play_actions = 5 + slot_ids
    discard_actions = slot_ids
    rank_actions = 15 + jnp.arange(NUM_RANKS)
    color_actions = 10 + jnp.arange(NUM_COLORS)

    sure_play = (own_p_play > 0.999) & own_occupied
    scores = _with_scores(scores, play_actions, jnp.where(sure_play, 1000.0 + own_p_play, -1.0e9))

    rank_hint_ok = all_rank_cards_playable & (new_playable_rank_count > 0.0)
    scores = _with_scores(scores, rank_actions, jnp.where(rank_hint_ok, 900.0 + new_playable_rank_count, -1.0e9))

    color_hint_ok = color_complement_count > 0.0
    scores = _with_scores(scores, color_actions, jnp.where(color_hint_ok, 850.0 + color_complement_count, -1.0e9))

    rank_comp_ok = rank_complement_count > 0.0
    scores = _with_scores(scores, rank_actions, jnp.where(rank_comp_ok, 840.0 + rank_complement_count, -1.0e9))

    safe_discard = (own_p_dead > 0.999) & own_occupied
    scores = _with_scores(scores, discard_actions, jnp.where(safe_discard, 700.0 + own_p_dead, -1.0e9))

    high_conf_play = (own_p_play > 0.90) & (lives > 2.5) & own_occupied
    scores = _with_scores(scores, play_actions, jnp.where(high_conf_play, 760.0 + own_p_play, -1.0e9))

    save_hint_ok = save_rank_count > 0.0
    scores = _with_scores(scores, rank_actions, jnp.where(save_hint_ok, 500.0 + save_rank_count, -1.0e9))

    fallback_base = jnp.where(info > 6.0, 90.0, 300.0)
    fallback_discard_score = (
        fallback_base
        + 10.0 * own_p_dead
        - 10.0 * own_p_play
        - 14.0 * own_p_critical
        - 0.15 * own_expected_rank
        - 20.0 * own_protected.astype(jnp.float32)
    )
    scores = _with_scores(
        scores,
        discard_actions,
        jnp.where(own_occupied, fallback_discard_score, -1.0e9),
    )

    any_partner_slot = jnp.sum(partner_slot_has.astype(jnp.float32)) > 0.0
    arbitrary_rank_hint = jnp.where(rank_card_count > 0.0, 110.0 + (4.0 - jnp.arange(NUM_RANKS, dtype=jnp.float32)), -1.0e9)
    arbitrary_color_hint = jnp.where(
        jnp.sum(partner_color.astype(jnp.float32), axis=0) > 0.0,
        105.0,
        -1.0e9,
    )
    scores = _with_scores(scores, rank_actions, jnp.where(any_partner_slot, arbitrary_rank_hint, -1.0e9))
    scores = _with_scores(scores, color_actions, jnp.where(any_partner_slot, arbitrary_color_hint, -1.0e9))

    emergency_play = (lives > 1.5) & own_occupied
    scores = _with_scores(scores, play_actions, jnp.where(emergency_play, 50.0 + own_p_play, -1.0e9))

    chosen = _best_legal_action(scores, legal)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), chosen)
    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory + jnp.asarray(1, dtype=jnp.int32))
    trace = {
        "info": info.astype(jnp.float32),
        "lives": lives.astype(jnp.float32),
        "own_p_play_max": jnp.max(own_p_play).astype(jnp.float32),
        "own_p_dead_max": jnp.max(own_p_dead).astype(jnp.float32),
        "score_chosen": jnp.max(jnp.where(legal, scores, -1.0e9)).astype(jnp.float32),
    }
    return action, next_memory, trace
