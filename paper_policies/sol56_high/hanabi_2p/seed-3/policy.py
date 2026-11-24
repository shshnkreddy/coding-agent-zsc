"""A conservative, convention-legible two-player Hanabi policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'conservative_direct_hint',
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
    return {
        'turn': jnp.asarray(0, dtype=jnp.int32),
        'positive': jnp.zeros((5, 10), dtype=jnp.bool_),
        'signal': jnp.zeros((5,), dtype=jnp.bool_),
    }


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    last = f[253:308]

    partner_cards = f[0:125].reshape(5, 5, 5)
    partner_flat = partner_cards.reshape(5, 25)
    partner_occupied = jnp.sum(partner_flat, axis=1) > 0.5
    partner_color = jnp.argmax(jnp.sum(partner_cards, axis=2), axis=1)
    partner_rank = jnp.argmax(jnp.sum(partner_cards, axis=1), axis=1)

    fireworks = f[167:192].reshape(5, 5)
    heights = jnp.sum(fireworks * jnp.arange(1, 6, dtype=f.dtype)[None, :], axis=1).astype(jnp.int32)
    info = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    deck_empty = jnp.sum(f[127:167]) < 0.5
    any_missing_card = jnp.any(f[125:127] > 0.5)

    knowledge = f[308:658].reshape(2, 5, 35)
    own_v = knowledge[0, :, 0:25].reshape(5, 5, 5)
    partner_v = knowledge[1, :, 0:25].reshape(5, 5, 5)
    own_positive = jnp.any(knowledge[0, :, 25:35] > 0.5, axis=1)
    partner_positive = jnp.any(knowledge[1, :, 25:35] > 0.5, axis=1)
    own_positive_detail = knowledge[0, :, 25:35] > 0.5
    partner_positive_detail = knowledge[1, :, 25:35] > 0.5

    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    playable_types = ranks == heights[:, None]
    next_playable_types = ranks == (heights[:, None] + 1)
    obsolete_types = ranks < heights[:, None]

    discard_block = f[203:253].reshape(5, 10)
    discarded_counts = jnp.stack((
        jnp.sum(discard_block[:, 0:3], axis=1),
        jnp.sum(discard_block[:, 3:5], axis=1),
        jnp.sum(discard_block[:, 5:7], axis=1),
        jnp.sum(discard_block[:, 7:9], axis=1),
        jnp.sum(discard_block[:, 9:10], axis=1),
    ), axis=1)
    multiplicities = jnp.asarray([3, 2, 2, 2, 1], dtype=f.dtype)[None, :]
    already_played = ranks < heights[:, None]
    remaining_public = multiplicities - discarded_counts - already_played.astype(f.dtype)
    needed_or_later = ranks >= heights[:, None]
    missing_needed = needed_or_later & (remaining_public <= 0.01)
    first_missing = jnp.min(
        jnp.where(missing_needed, ranks, jnp.asarray(5, dtype=jnp.int32)), axis=1
    )
    unreachable_types = ranks >= first_missing[:, None]
    expendable_types = obsolete_types | unreachable_types
    critical_types = (remaining_public <= 1.01) & (~expendable_types)

    # Native V0 deliberately does not remove the cards visible in the partner's
    # hand. Removing them here is an ordinary card-count deduction available to
    # any partner policy, and can turn some own-slot beliefs into certainties.
    partner_counts = jnp.sum(partner_cards, axis=0)
    available_to_self = jnp.maximum(remaining_public - partner_counts, 0.0)
    own_mask = own_v > 1.0e-8
    slot_available = jnp.broadcast_to(available_to_self[None, :, :], own_mask.shape)
    for _ in range(4):
        effective_mask = own_mask & (slot_available > 0.0)
        singleton = effective_mask & (
            jnp.sum(effective_mask, axis=(1, 2), keepdims=True) == 1
        )
        assigned_counts = jnp.sum(singleton.astype(f.dtype), axis=0)
        slot_available = jnp.maximum(
            available_to_self[None, :, :]
            - assigned_counts[None, :, :]
            + singleton.astype(f.dtype),
            0.0,
        )
    effective_mask = own_mask & (slot_available > 0.0)
    own_refined = effective_mask.astype(f.dtype) * slot_available

    own_mass = jnp.sum(own_refined, axis=(1, 2))
    own_play_prob = jnp.sum(own_refined * playable_types[None, :, :], axis=(1, 2)) / jnp.maximum(own_mass, 1.0e-8)
    own_obsolete_prob = jnp.sum(own_refined * expendable_types[None, :, :], axis=(1, 2)) / jnp.maximum(own_mass, 1.0e-8)
    own_critical_prob = jnp.sum(own_refined * critical_types[None, :, :], axis=(1, 2)) / jnp.maximum(own_mass, 1.0e-8)
    native_own_mass = jnp.sum(own_v, axis=(1, 2))
    native_own_play_prob = jnp.sum(own_v * playable_types[None, :, :], axis=(1, 2)) / jnp.maximum(native_own_mass, 1.0e-8)
    own_occupied = own_mass > 1.0e-8
    certain_play = own_occupied & (own_play_prob > 0.999)

    partner_mass = jnp.sum(partner_v, axis=(1, 2))
    partner_play_prob = jnp.sum(partner_v * playable_types[None, :, :], axis=(1, 2)) / jnp.maximum(partner_mass, 1.0e-8)
    partner_certain_play = partner_occupied & (partner_play_prob > 0.999)
    partner_actual_play = partner_occupied & (partner_rank == heights[partner_color])

    # A direct clue points at its newest (rightmost) matching card. We only
    # send it if that target is truly playable and literal evidence supports it.
    slots = jnp.arange(5, dtype=jnp.int32)
    colors = jnp.arange(5, dtype=jnp.int32)
    rank_ids = jnp.arange(5, dtype=jnp.int32)
    color_matches = partner_color[None, :] == colors[:, None]
    rank_matches = partner_rank[None, :] == rank_ids[:, None]
    clue_matches = jnp.concatenate((color_matches, rank_matches), axis=0) & partner_occupied[None, :]
    partner_attr_marked = partner_positive_detail.T
    fresh_clue_matches = clue_matches & (~partner_attr_marked)
    has_fresh_match = jnp.any(fresh_clue_matches, axis=1)
    targeted_matches = jnp.where(has_fresh_match[:, None], fresh_clue_matches, clue_matches)
    clue_target = jnp.argmax(jnp.where(targeted_matches, slots[None, :], -1), axis=1)

    target_cards = partner_v[clue_target]
    color_type_masks = jnp.eye(5, dtype=f.dtype)[:, :, None] * jnp.ones((1, 1, 5), dtype=f.dtype)
    rank_type_masks = jnp.ones((5, 5, 1), dtype=f.dtype) * jnp.eye(5, dtype=f.dtype)[:, None, :]
    clue_type_masks = jnp.concatenate((color_type_masks, rank_type_masks), axis=0)
    all_posteriors = partner_v[None, :, :, :] * jnp.where(
        clue_matches[:, :, None, None],
        clue_type_masks[:, None, :, :],
        1.0 - clue_type_masks[:, None, :, :],
    )
    all_posterior_mass = jnp.sum(all_posteriors, axis=(2, 3))
    all_play_prob = jnp.sum(all_posteriors * playable_types[None, None, :, :], axis=(2, 3)) / jnp.maximum(all_posterior_mass, 1.0e-8)
    new_safe_plays = jnp.sum(((all_play_prob > 0.999) & (~partner_certain_play[None, :])).astype(f.dtype), axis=1)
    partner_next_prob = jnp.sum(
        partner_v * next_playable_types[None, :, :], axis=(1, 2)
    ) / jnp.maximum(partner_mass, 1.0e-8)
    all_next_prob = jnp.sum(
        all_posteriors * next_playable_types[None, None, :, :], axis=(2, 3)
    ) / jnp.maximum(all_posterior_mass, 1.0e-8)
    new_future_plays = jnp.sum(
        (
            (all_next_prob > 0.999)
            & (partner_next_prob[None, :] <= 0.999)
        ).astype(f.dtype),
        axis=1,
    )
    partner_expendable_prob = jnp.sum(
        partner_v * expendable_types[None, :, :], axis=(1, 2)
    ) / jnp.maximum(partner_mass, 1.0e-8)
    all_expendable_prob = jnp.sum(
        all_posteriors * expendable_types[None, None, :, :], axis=(2, 3)
    ) / jnp.maximum(all_posterior_mass, 1.0e-8)
    new_safe_discards = jnp.sum(
        (
            (all_expendable_prob > 0.999)
            & (partner_expendable_prob[None, :] <= 0.999)
        ).astype(f.dtype),
        axis=1,
    )
    prior_prob = partner_v / jnp.maximum(partner_mass[:, None, None], 1.0e-8)
    prior_entropy = -jnp.sum(
        prior_prob * jnp.log(jnp.maximum(prior_prob, 1.0e-8))
    )
    posterior_prob_all = all_posteriors / jnp.maximum(
        all_posterior_mass[:, :, None, None], 1.0e-8
    )
    posterior_entropy = -jnp.sum(
        posterior_prob_all * jnp.log(jnp.maximum(posterior_prob_all, 1.0e-8)),
        axis=(1, 2, 3),
    )
    clue_information_gain = prior_entropy - posterior_entropy
    posterior = target_cards * clue_type_masks
    posterior_mass = jnp.sum(posterior, axis=(1, 2))
    posterior_play_prob = jnp.sum(posterior * playable_types[None, :, :], axis=(1, 2)) / jnp.maximum(posterior_mass, 1.0e-8)
    target_actual_play = partner_actual_play[clue_target]
    target_already_safe = partner_certain_play[clue_target]
    clue_legal = legal[10:20]
    convention_threshold = jnp.where(
        lives <= 1,
        0.999,
        jnp.where(lives == 2, 0.34, 0.19),
    )
    convention_direct_clue = (
        clue_legal
        & has_fresh_match
        & target_actual_play
        & (~target_already_safe)
        & (posterior_play_prob >= convention_threshold)
    )
    literal_clue = clue_legal & (new_safe_plays > 0.5)
    direct_clue = convention_direct_clue | literal_clue
    newly_informative = has_fresh_match.astype(f.dtype)
    target_color = partner_color[clue_target]
    target_rank = partner_rank[clue_target]
    target_chain_unlock = jnp.sum(
        (
            (partner_color[None, :] == target_color[:, None])
            & (partner_rank[None, :] == (target_rank[:, None] + 1))
            & partner_occupied[None, :]
        ).astype(f.dtype),
        axis=1,
    )
    target_five_token = ((target_rank == 4) & (info < 8)).astype(f.dtype)
    playable_matches_count = jnp.sum(
        (clue_matches & partner_actual_play[None, :]).astype(f.dtype), axis=1
    )
    direct_score = jnp.where(
        direct_clue,
        10.0
        + 2.0 * new_safe_plays
        + posterior_play_prob
        + 0.25 * newly_informative
        + 0.1 * clue_information_gain
        + 0.5 * playable_matches_count,
        -100.0,
    )
    direct_score = jnp.where(
        direct_clue,
        direct_score
        + convention_direct_clue.astype(f.dtype)
        * (1.5 * target_chain_unlock + target_five_token),
        -100.0,
    )
    direct_hint_action = (10 + jnp.argmax(direct_score)).astype(jnp.int32)
    has_direct_hint = jnp.any(direct_clue)

    # Interpret only a hint made by the partner on the immediately preceding
    # transition. Literal certainty still takes priority over convention.
    received_hint = (last[1] > 0.5) & ((last[4] > 0.5) | (last[5] > 0.5)) & (last[6] > 0.5)
    received_matches = last[18:23] > 0.5
    prior_positive = jnp.where(reset, jnp.zeros((5, 10), dtype=jnp.bool_), memory['positive'])
    newly_marked = jnp.any(own_positive_detail & (~prior_positive), axis=1)
    fresh_received = received_matches & newly_marked
    target_pool = jnp.where(jnp.any(fresh_received), fresh_received, received_matches)
    received_target = jnp.argmax(jnp.where(target_pool, slots, -1))
    received_prob = native_own_play_prob[received_target]
    new_signal = received_hint & jnp.any(fresh_received) & own_occupied[received_target] & (received_prob >= convention_threshold)

    # Recommendations are intentionally one-turn only: after any intervening
    # play, the fireworks can change and make the inferred target obsolete.
    signal = jnp.where(
        new_signal,
        jnp.eye(5, dtype=jnp.bool_)[received_target],
        jnp.zeros((5,), dtype=jnp.bool_),
    ) & own_occupied
    convention_slot = jnp.argmax(signal).astype(jnp.int32)
    signal_still_plausible = native_own_play_prob[convention_slot] >= convention_threshold
    signal = signal & signal_still_plausible
    convention_play = jnp.any(signal) & own_occupied[convention_slot]

    # Prefer certainly obsolete cards. Otherwise discard the oldest unprotected
    # card, with expected obsolescence as a tie-breaker.
    certain_obsolete = own_occupied & (own_obsolete_prob > 0.999)
    obsolete_score = jnp.where(certain_obsolete & legal[0:5], 20.0 + own_obsolete_prob - 0.01 * slots, -100.0)
    obsolete_slot = jnp.argmax(obsolete_score).astype(jnp.int32)
    has_obsolete = jnp.any(certain_obsolete & legal[0:5])

    unprotected = own_occupied & (~own_positive)
    discard_score = (
        3.0 * own_obsolete_prob
        - 2.5 * own_play_prob
        - 7.0 * own_critical_prob
        + 0.20 * (4 - slots).astype(f.dtype)
        + 0.50 * unprotected.astype(f.dtype)
        - 1.00 * own_positive.astype(f.dtype)
    )
    discard_score = jnp.where(legal[0:5], discard_score, -100.0)
    weighted_discard_slot = jnp.argmax(discard_score).astype(jnp.int32)
    legal_unprotected = legal[0:5] & unprotected
    conventional_chop_slot = jnp.argmax(legal_unprotected).astype(jnp.int32)
    discard_slot = jnp.where(
        jnp.any(legal_unprotected), conventional_chop_slot, weighted_discard_slot
    ).astype(jnp.int32)

    # Protect an unmarked critical chop card (all rank fives, or the last public
    # copy of a still-needed identity) before it can be discarded.
    partner_unprotected = partner_occupied & (~partner_positive)
    chop_slot = jnp.argmax(partner_unprotected).astype(jnp.int32)
    chop_color = partner_color[chop_slot]
    chop_rank = partner_rank[chop_slot]
    chop_needed = (chop_rank >= heights[chop_color]) & (chop_rank < first_missing[chop_color])
    chop_critical = jnp.any(partner_unprotected) & chop_needed & (
        (chop_rank == 4)
        | (remaining_public[chop_color, chop_rank] <= 1.01)
    )
    clue_safe = (
        target_actual_play
        | (posterior_play_prob < convention_threshold)
        | (new_safe_plays > 0.5)
    )
    marks_chop = clue_matches[:, chop_slot] & (~partner_attr_marked[:, chop_slot])
    save_clue = clue_legal & clue_safe & marks_chop
    save_score = jnp.where(
        save_clue,
        2.0 * new_safe_plays
        + 2.0 * newly_informative
        + 0.1 * clue_information_gain
        + 0.1 * (jnp.arange(10) >= 5),
        -100.0,
    )
    save_hint_action = (10 + jnp.argmax(save_score)).astype(jnp.int32)
    partner_has_safe_discard = jnp.any(
        partner_occupied & (partner_expendable_prob > 0.999)
    )
    literal_progress_score = jnp.where(literal_clue, direct_score, -100.0)
    literal_progress_action = (
        10 + jnp.argmax(literal_progress_score)
    ).astype(jnp.int32)
    has_token_backed_literal_progress = (
        chop_critical
        & (~partner_has_safe_discard)
        & (info >= 2)
        & jnp.any(literal_clue)
        & (~deck_empty)
    )
    has_save_hint = (
        chop_critical
        & jnp.any(save_clue)
        & (~partner_has_safe_discard)
        & (~has_token_backed_literal_progress)
        & (~deck_empty)
    )

    discard_clue = clue_legal & clue_safe & (new_safe_discards > 0.5)
    discard_clue_score = jnp.where(
        discard_clue,
        3.0 * new_safe_discards + clue_information_gain,
        -100.0,
    )
    discard_hint_action = (10 + jnp.argmax(discard_clue_score)).astype(jnp.int32)
    has_discard_hint = (
        ((~jnp.any(partner_unprotected)) | (info == 8))
        & jnp.any(discard_clue)
        & (~deck_empty)
    )
    emergency_discard_clue = (
        clue_legal
        & (new_safe_discards > 0.5)
        & (posterior_play_prob < convention_threshold)
        & (new_safe_plays < 0.5)
    )
    emergency_discard_score = jnp.where(
        emergency_discard_clue,
        3.0 * new_safe_discards + clue_information_gain,
        -100.0,
    )
    emergency_discard_action = (
        10 + jnp.argmax(emergency_discard_score)
    ).astype(jnp.int32)
    has_emergency_discard_hint = (
        chop_critical
        & (~jnp.any(save_clue))
        & jnp.any(emergency_discard_clue)
        & (~deck_empty)
    )

    # If our own conventional chop is likely to contain a last needed copy,
    # spend a token establishing a literal safe discard in the partner hand.
    # This changes whose hand supplies the next token without asking the
    # partner to understand any private convention.
    own_chop_slot = jnp.argmax(unprotected).astype(jnp.int32)
    own_chop_at_risk = (
        jnp.any(legal_unprotected)
        & (own_critical_prob[own_chop_slot] >= 0.35)
    )
    has_risk_shift_hint = (
        own_chop_at_risk
        & jnp.any(emergency_discard_clue)
        & (~deck_empty)
    )

    # At eight tokens discarding is illegal, so give a useful legal clue. It is
    # not interpreted as a play unless its literal posterior crosses threshold.
    clue_size = jnp.sum(clue_matches, axis=1).astype(f.dtype)
    filler_ok = clue_legal & clue_safe
    filler_score = jnp.where(
        clue_legal,
        20.0 * clue_safe.astype(f.dtype)
        + clue_information_gain
        + 0.1 * newly_informative
        - 0.1 * posterior_play_prob,
        -100.0,
    )
    filler_hint_action = (10 + jnp.argmax(filler_score)).astype(jnp.int32)

    setup_clue = clue_legal & clue_safe & (new_future_plays > 0.5)
    setup_score = jnp.where(
        setup_clue,
        3.0 * new_future_plays + clue_information_gain,
        -100.0,
    )
    setup_hint_action = (10 + jnp.argmax(setup_score)).astype(jnp.int32)
    has_setup_hint = (
        (info >= 2) & jnp.any(setup_clue) & (~deck_empty)
    )

    gamble_score = jnp.where(legal[5:10], own_play_prob, -1.0)
    gamble_action = (5 + jnp.argmax(gamble_score)).astype(jnp.int32)
    has_endgame_gamble = deck_empty & (lives >= 2) & (jnp.max(gamble_score) > 0.0)

    partner_next_after_color = jnp.sum(
        (
            (partner_color[None, :] == colors[:, None])
            & (partner_rank[None, :] == (heights[:, None] + 1))
            & partner_occupied[None, :]
        ).astype(f.dtype),
        axis=1,
    )
    own_color_mass = jnp.sum(own_refined, axis=2)
    expected_unlock = jnp.sum(
        own_color_mass * partner_next_after_color[None, :], axis=1
    ) / jnp.maximum(own_mass, 1.0e-8)
    rank_five_prob = jnp.sum(own_refined[:, :, 4], axis=1) / jnp.maximum(own_mass, 1.0e-8)
    rank_progress = jnp.sum(
        own_refined * jnp.arange(1, 6, dtype=f.dtype)[None, None, :], axis=(1, 2)
    ) / jnp.maximum(own_mass, 1.0e-8)
    safe_play_score = jnp.where(
        certain_play,
        4.0 * expected_unlock
        + 2.0 * (info < 8).astype(f.dtype) * rank_five_prob
        + 0.01 * rank_progress
        - 0.001 * slots.astype(f.dtype),
        -100.0,
    )
    safe_play_action = (5 + jnp.argmax(safe_play_score)).astype(jnp.int32)
    convention_action = (5 + convention_slot).astype(jnp.int32)
    fallback_action = jnp.argmax(legal).astype(jnp.int32)

    action = fallback_action
    action = jnp.where(jnp.any(clue_legal), filler_hint_action, action)
    action = jnp.where(jnp.any(legal[0:5]), discard_slot, action)
    action = jnp.where(has_obsolete, obsolete_slot, action)
    action = jnp.where(has_endgame_gamble, gamble_action, action)
    action = jnp.where(has_setup_hint, setup_hint_action, action)
    action = jnp.where(has_risk_shift_hint, emergency_discard_action, action)
    action = jnp.where(has_discard_hint, discard_hint_action, action)
    action = jnp.where(has_direct_hint, direct_hint_action, action)
    action = jnp.where(
        has_emergency_discard_hint, emergency_discard_action, action
    )
    action = jnp.where(has_save_hint, save_hint_action, action)
    action = jnp.where(convention_play, convention_action, action)
    action = jnp.where(jnp.any(certain_play), safe_play_action, action)
    action = jnp.where(
        has_token_backed_literal_progress, literal_progress_action, action
    )
    action = jnp.where(
        has_emergency_discard_hint, emergency_discard_action, action
    )
    action = jnp.where(has_save_hint, save_hint_action, action)
    is_last_final_action = (
        deck_empty
        & any_missing_card
        & (~legal[20])
        & (lives >= 2)
        & (jnp.max(gamble_score) > 0.0)
    )
    action = jnp.where(is_last_final_action, gamble_action, action)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), action)

    old_turn = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['turn'])
    # A signal is consumed whenever its recommended slot is selected to play.
    played_signal = (~legal[20]) & (action == (5 + convention_slot))
    signal_after_action = signal & (~(jnp.eye(5, dtype=jnp.bool_)[convention_slot] & played_signal))
    next_memory = {
        'turn': old_turn + (~legal[20]).astype(jnp.int32),
        'positive': own_positive_detail,
        'signal': signal_after_action,
    }
    trace = {
        'certain_play': jnp.any(certain_play),
        'convention_play': convention_play,
        'direct_hint': has_direct_hint,
        'info': info,
        'lives': lives,
    }
    return action, next_memory, trace
