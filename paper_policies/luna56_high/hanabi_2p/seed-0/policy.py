"""A conservative, convention-light Hanabi policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'belief_safe_hints',
    'adapter': 'hanabi',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 5,
    'num_ranks': 5,
    'hand_size': 5,
    'observation_size': 658,
    'num_actions': 21,
    'policy_structure': 'shared_local',
}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions

    fireworks = f[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks * jnp.arange(1, 6, dtype=f.dtype), axis=1)
    colors = jnp.repeat(jnp.arange(5), 5)
    ranks = jnp.tile(jnp.arange(5), 5)
    playable_types = ranks == heights[colors]
    live_types = ranks >= heights[colors]

    knowledge = f[308:658].reshape((2, 5, 35))
    own_weights = knowledge[0, :, :25]
    partner_weights = knowledge[1, :, :25]
    own_total = jnp.sum(own_weights, axis=1)
    own_safe_mass = jnp.sum(own_weights * playable_types, axis=1)
    own_live_mass = jnp.sum(own_weights * live_types, axis=1)
    own_present = own_total > 0.0
    own_safe = own_present & (own_safe_mass >= own_total - 1e-5)
    own_dead = own_present & (own_live_mass <= 1e-5)

    own_hint_strength = (
        jnp.sum(knowledge[0, :, 25:30], axis=1)
        + jnp.sum(knowledge[0, :, 30:35], axis=1)
    )
    safe_play_scores = jnp.where(
        own_safe,
        100.0 + own_hint_strength,
        -1.0e6,
    )

    partner_hand = f[0:125].reshape((5, 25))
    partner_present = jnp.sum(partner_hand, axis=1) > 0.0
    partner_types = partner_hand > 0.0
    partner_total = jnp.sum(partner_weights, axis=1)
    partner_current_safe = (
        jnp.sum(partner_weights * playable_types, axis=1)
        >= partner_total - 1e-5
    ) & (partner_total > 0.0)

    # These flags are public evidence of what the partner already knows.
    # They let the sender choose the complementary attribute instead of
    # repeatedly naming the same color or rank.
    partner_color_flags = knowledge[1, :, 25:30] > 0.0
    partner_rank_flags = knowledge[1, :, 30:35] > 0.0

    color_match = colors[None, :] == jnp.arange(5)[:, None]
    rank_match = ranks[None, :] == jnp.arange(5)[:, None]

    def hint_quality(match):
        matching = partner_types[:, None, :] & match[None, :, :]
        new_weights = jnp.where(
            matching,
            partner_weights[:, None, :] * jnp.where(match[None, :, :], 1.0, 0.0),
            partner_weights[:, None, :] * jnp.where(match[None, :, :], 0.0, 1.0),
        )
        totals = jnp.sum(new_weights, axis=2)
        safe_mass = jnp.sum(new_weights * playable_types[None, None, :], axis=2)
        after_safe = (safe_mass >= totals - 1e-5) & (totals > 0.0)
        newly_safe = after_safe & (~partner_current_safe[:, None])
        singleton = (totals <= 1.00001) & (totals > 0.0)
        useful = jnp.sum(newly_safe & partner_present[:, None], axis=0)
        exact = jnp.sum(singleton & partner_present[:, None], axis=0)
        safe = jnp.sum(after_safe & partner_present[:, None], axis=0)
        coverage = jnp.sum(matching & partner_present[:, None, None], axis=(0, 2))
        # The sender sees the partner's identity exactly.  This term avoids
        # spending an early token on a merely informative attribute when a
        # rank/color hint can directly identify a card that is playable now.
        actual_playable = partner_types & playable_types[None, :]
        target = jnp.sum(
            actual_playable[:, None, :] & match[None, :, :], axis=(0, 2)
        )
        return 100000.0 * target + 1000.0 * useful + 20.0 * safe + 3.0 * exact + coverage

    color_quality = hint_quality(color_match)
    rank_quality = hint_quality(rank_match)

    actual_color_match = jnp.sum(
        partner_types[:, None, :] & color_match[None, :, :], axis=2
    ) > 0
    actual_rank_match = jnp.sum(
        partner_types[:, None, :] & rank_match[None, :, :], axis=2
    ) > 0
    target_card = jnp.sum(
        partner_types * playable_types[None, :], axis=1
    ) > 0
    actual_rank_zero = actual_rank_match[:, 0]
    color_known = jnp.sum(actual_color_match & partner_color_flags, axis=1) > 0
    rank_known = jnp.sum(actual_rank_match & partner_rank_flags, axis=1) > 0
    current_complete = target_card & (
        (actual_rank_zero & rank_known) | (color_known & rank_known)
    )

    color_after_known = color_known[:, None] | actual_color_match
    rank_after_color = rank_known[:, None]
    color_after_safe = target_card[:, None] & rank_after_color & (
        actual_rank_zero[:, None] | color_after_known
    )
    color_new_safe = color_after_safe & (~current_complete[:, None])
    color_progress = target_card[:, None] & (~color_known[:, None]) & actual_color_match

    rank_after_known = rank_known[:, None] | actual_rank_match
    color_after_rank = color_known[:, None]
    rank_after_safe = target_card[:, None] & rank_after_known & (
        actual_rank_zero[:, None] | color_after_rank
    )
    rank_new_safe = rank_after_safe & (~current_complete[:, None])
    rank_progress = target_card[:, None] & (~rank_known[:, None]) & actual_rank_match

    color_completion = 1000000.0 * jnp.sum(color_new_safe, axis=0)
    color_completion = color_completion + 10000.0 * jnp.sum(color_progress, axis=0)
    rank_completion = 1000000.0 * jnp.sum(rank_new_safe, axis=0)
    rank_completion = rank_completion + 10000.0 * jnp.sum(rank_progress, axis=0)
    hint_scores = jnp.concatenate(
        (color_quality + color_completion, rank_quality + rank_completion), axis=0
    )
    partner_has_card = jnp.any(partner_present)
    any_info = jnp.sum(f[192:200]) > 0.0
    hint_available = any_info & partner_has_card & (~jnp.any(partner_current_safe))
    hint_scores = jnp.where(hint_available, hint_scores, -1.0e6)
    useful_hint = hint_scores >= 1000.0
    hint_priority = jnp.where(useful_hint, 500000.0, 300000.0) + hint_scores

    dead_discard_scores = jnp.where(own_dead, 400000.0, -1.0e6)
    risk_play_scores = 200000.0 + 100000.0 * own_safe_mass / jnp.maximum(own_total, 1e-6)
    risk_play_scores = jnp.where(own_present, risk_play_scores, -1.0e6)
    fallback_discard_scores = jnp.where(own_present, 100000.0, -1.0e6)

    scores = jnp.full((21,), -1.0e6, dtype=f.dtype)
    scores = scores.at[5:10].set(safe_play_scores + 1000000.0)
    scores = scores.at[10:20].set(hint_priority)
    scores = scores.at[0:5].set(dead_discard_scores)

    best_safe = jnp.max(safe_play_scores)
    best_hint = jnp.max(hint_priority)
    best_dead = jnp.max(dead_discard_scores)
    have_primary = (best_safe > 0.0) | (best_hint > 0.0) | (best_dead > 0.0)
    best_risk = jnp.max(risk_play_scores)
    # With bomb-zero scoring, a plausible play is not worth a life unless it
    # is close to certain.  Discarding keeps the episode alive and lets later
    # safe draws or hints contribute to the final score.
    risk_ok = best_risk > 200000.0 + 70000.0
    fallback_play = jnp.where(risk_ok, risk_play_scores + 150000.0, -1.0e6)
    scores = scores.at[5:10].set(
        jnp.where(have_primary, scores[5:10], fallback_play)
    )
    scores = scores.at[0:5].set(
        jnp.where(have_primary, scores[0:5], fallback_discard_scores)
    )
    scores = jnp.where(legal, scores, -1.0e9)
    action = jnp.argmax(scores).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)
    trace = {
        'safe_play': jnp.asarray(best_safe > 0.0),
        'hint_score': jnp.asarray(best_hint),
        'dead_card': jnp.asarray(best_dead > 0.0),
        'risk': jnp.asarray(best_risk - 200000.0),
    }
    return action, next_memory, trace
