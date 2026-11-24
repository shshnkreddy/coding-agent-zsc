"""Convention-light Hanabi policy using public beliefs and literal hints."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _firework_heights(features):
    fireworks = features[167:192].reshape((5, 5))
    rank_numbers = jnp.arange(1, 6, dtype=jnp.float32)
    return jnp.sum(fireworks * rank_numbers[None, :], axis=1).astype(jnp.int32)


def _type_tables(heights):
    colors = jnp.arange(5, dtype=jnp.int32)[:, None]
    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    playable = (ranks == heights[:, None]) & (heights[:, None] < 5)
    dead = ranks < heights[:, None]
    return playable.astype(jnp.float32), dead.astype(jnp.float32), colors, ranks


def _discard_counts_by_type(features):
    discards = features[203:253].reshape((5, 10))
    return jnp.stack(
        [
            jnp.sum(discards[:, 0:3], axis=1),
            jnp.sum(discards[:, 3:5], axis=1),
            jnp.sum(discards[:, 5:7], axis=1),
            jnp.sum(discards[:, 7:9], axis=1),
            discards[:, 9],
        ],
        axis=1,
    )


def _normalize(weights):
    total = jnp.sum(weights, axis=(1, 2), keepdims=True)
    return jnp.where(total > 0.0, weights / jnp.maximum(total, 1.0e-6), weights)


def _hinted_belief(v0, attr_mask, slot_matches):
    keep = jnp.where(slot_matches[:, None, None], attr_mask[None, :, :], 1.0 - attr_mask[None, :, :])
    return _normalize(v0 * keep)


def _best_hint(legal, partner_hand, partner_v0, playable_by_type):
    slot_playable = jnp.sum(partner_hand * playable_by_type[None, :, :], axis=(1, 2)) > 0.5
    current_p = jnp.sum(partner_v0 * playable_by_type[None, :, :], axis=(1, 2))

    color_masks = jnp.eye(5, dtype=jnp.float32)[:, :, None] * jnp.ones((5, 1, 5), dtype=jnp.float32)
    rank_masks = jnp.eye(5, dtype=jnp.float32)[:, None, :] * jnp.ones((5, 5, 1), dtype=jnp.float32)
    attr_masks = jnp.concatenate([color_masks, rank_masks], axis=0)

    color_matches = jnp.sum(partner_hand, axis=2).T > 0.5
    rank_matches = jnp.sum(partner_hand, axis=1).T > 0.5
    matches = jnp.concatenate([color_matches, rank_matches], axis=0)

    def score_one(i):
        v_new = _hinted_belief(partner_v0, attr_masks[i], matches[i])
        p_new = jnp.sum(v_new * playable_by_type[None, :, :], axis=(1, 2))
        good = matches[i] & slot_playable
        bad = matches[i] & (~slot_playable)
        best_good = jnp.max(jnp.where(good, p_new, -1.0))
        best_bad = jnp.max(jnp.where(bad, p_new, 0.0))
        improvement = jnp.max(jnp.where(good, p_new - current_p, 0.0))
        num_good = jnp.sum(good.astype(jnp.float32))
        num_bad = jnp.sum(bad.astype(jnp.float32))
        has_new_value = improvement > 0.02
        return (
            120.0 * best_good
            + 18.0 * improvement
            + 2.0 * num_good
            - 45.0 * best_bad
            - 4.0 * num_bad
            - 200.0 * (1.0 - has_new_value.astype(jnp.float32))
        )

    raw_scores = jnp.array([score_one(i) for i in range(10)])
    hint_legal = jnp.concatenate([legal[10:15], legal[15:20]])
    scores = jnp.where(hint_legal, raw_scores, -1.0e9)
    best_i = jnp.argmax(scores)
    best_action = jnp.where(best_i < 5, 10 + best_i, 15 + (best_i - 5))
    return best_action.astype(jnp.int32), scores[best_i]


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions

    heights = _firework_heights(features)
    playable_by_type, dead_by_type, _, ranks = _type_tables(heights)
    lives = jnp.sum(features[200:203])
    deck_count = jnp.sum(features[127:167])
    discard_counts = _discard_counts_by_type(features)
    multiplicities = jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0], dtype=jnp.float32)
    already_played = (ranks < heights[:, None]).astype(jnp.float32)
    remaining_by_type = multiplicities[None, :] - discard_counts - already_played
    future_needed = ranks >= heights[:, None]
    critical_by_type = ((remaining_by_type <= 1.01) & future_needed).astype(jnp.float32)

    knowledge = features[308:658].reshape((2, 5, 35))
    self_v0 = knowledge[0, :, :25].reshape((5, 5, 5))
    partner_v0 = knowledge[1, :, :25].reshape((5, 5, 5))
    partner_hand = features[:125].reshape((5, 5, 5))

    p_play = jnp.sum(self_v0 * playable_by_type[None, :, :], axis=(1, 2))
    p_dead = jnp.sum(self_v0 * dead_by_type[None, :, :], axis=(1, 2))
    p_critical = jnp.sum(self_v0 * critical_by_type[None, :, :], axis=(1, 2))
    occupied = jnp.sum(self_v0, axis=(1, 2)) > 0.0
    slot_age = (5.0 - jnp.arange(5, dtype=jnp.float32)) * 0.001

    play_scores = jnp.where(legal[5:10] & occupied, p_play + slot_age, -1.0e9)
    best_play_slot = jnp.argmax(play_scores)
    best_play_p = p_play[best_play_slot]
    play_action = (5 + best_play_slot).astype(jnp.int32)

    hint_action, hint_score = _best_hint(legal, partner_hand, partner_v0, playable_by_type)

    discard_scores = 3.0 * p_dead - 2.5 * p_play + slot_age
    discard_scores = jnp.where(legal[:5] & occupied, discard_scores, -1.0e9)
    discard_action = jnp.argmax(discard_scores).astype(jnp.int32)

    fallback_scores = jnp.where(legal, jnp.arange(21, dtype=jnp.float32) * -0.001, -1.0e9)
    fallback_action = jnp.argmax(fallback_scores).astype(jnp.int32)

    base_play_threshold = jnp.where(lives >= 2.5, 0.76, jnp.where(lives >= 1.5, 0.86, 0.985))
    late_play_threshold = jnp.where(lives >= 2.5, 0.62, jnp.where(lives >= 1.5, 0.70, 0.985))
    play_threshold = jnp.where(deck_count <= 8.0, late_play_threshold, base_play_threshold)
    confident_play = best_play_p >= play_threshold
    useful_hint = hint_score >= 90.0
    any_hint = hint_score > -1.0e8
    can_discard = jnp.max(discard_scores) > -1.0e8
    emergency_play = (best_play_p >= 0.55) & (~can_discard)

    action = jnp.where(
        legal[20],
        20,
        jnp.where(
            confident_play,
            play_action,
            jnp.where(
                useful_hint,
                hint_action,
                jnp.where(
                    can_discard,
                    discard_action,
                    jnp.where(any_hint, hint_action, jnp.where(emergency_play, play_action, fallback_action)),
                ),
            ),
        ),
    ).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    trace = {
        "p_play": p_play,
        "p_dead": p_dead,
        "p_critical": p_critical,
        "hint_score": jnp.asarray(hint_score, dtype=jnp.float32),
    }
    return action, jnp.where(reset, zero, memory), trace
