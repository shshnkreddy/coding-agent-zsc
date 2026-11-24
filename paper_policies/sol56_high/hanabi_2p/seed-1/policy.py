"""Literal-first two-player Hanabi policy with cautious clue interpretation."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'literal_safe_v1',
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
        'intent': jnp.zeros((5,), dtype=jnp.bool_),
        'clued': jnp.zeros((5,), dtype=jnp.bool_),
        'empty_steps': jnp.asarray(0, dtype=jnp.int32),
        'focus_trust': jnp.asarray(True, dtype=jnp.bool_),
        'pending_focus': jnp.asarray(False, dtype=jnp.bool_),
    }


def _shift_after_remove(values, slot):
    """Shift slot-indexed memory exactly as the environment shifts a hand."""
    idx = jnp.arange(5, dtype=jnp.int32)
    source = jnp.minimum(idx + (idx >= slot).astype(jnp.int32), 4)
    shifted = values[source]
    return shifted.at[4].set(False)


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions

    partner_cards = f[0:125].reshape((5, 5, 5))
    partner_occupied = jnp.sum(partner_cards, axis=(1, 2)) > 0.5
    partner_color = jnp.argmax(jnp.sum(partner_cards, axis=2), axis=1)
    partner_rank = jnp.argmax(jnp.sum(partner_cards, axis=1), axis=1)

    fireworks = f[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks * jnp.arange(1, 6, dtype=jnp.float32), axis=1).astype(jnp.int32)
    info = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)
    prior_empty_steps = jnp.where(reset, 0, memory['empty_steps'])
    empty_steps = jnp.where(deck == 0, jnp.minimum(prior_empty_steps + 1, 2), 0).astype(jnp.int32)
    final_actor_turn = empty_steps >= 2

    discards_raw = f[203:253].reshape((5, 10))
    discard_counts = jnp.stack(
        [
            jnp.sum(discards_raw[:, 0:3], axis=1),
            jnp.sum(discards_raw[:, 3:5], axis=1),
            jnp.sum(discards_raw[:, 5:7], axis=1),
            jnp.sum(discards_raw[:, 7:9], axis=1),
            discards_raw[:, 9],
        ],
        axis=1,
    )
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.float32)[None, :]
    rank_grid = jnp.arange(5, dtype=jnp.int32)[None, :]
    played = rank_grid < heights[:, None]
    remaining = multiplicity - discard_counts - played.astype(jnp.float32)

    knowledge = f[308:658].reshape((2, 5, 35))
    native_own_belief = knowledge[0, :, 0:25].reshape((5, 5, 5))
    partner_belief = knowledge[1, :, 0:25].reshape((5, 5, 5))
    own_positive = knowledge[0, :, 25:35]
    partner_positive = knowledge[1, :, 25:35]
    own_occupied = jnp.sum(native_own_belief, axis=(1, 2)) > 0.0
    own_clued_now = jnp.sum(own_positive, axis=1) > 0.0
    partner_clued = jnp.sum(partner_positive, axis=1) > 0.0

    # V0 deliberately ignores cards visible in the partner hand.  Removing
    # those exact cards gives a strictly sharper local marginal while using no
    # hidden information or convention assumptions.
    visible_counts = jnp.sum(partner_cards, axis=0)
    available_to_self = jnp.maximum(remaining - visible_counts, 0.0)
    own_weights = (native_own_belief > 0.0).astype(jnp.float32) * available_to_self[None, :, :]
    own_weight_sum = jnp.sum(own_weights, axis=(1, 2), keepdims=True)
    own_belief = own_weights / jnp.maximum(own_weight_sum, 1.0e-8)
    own_literal_mask = native_own_belief > 0.0
    for _ in range(4):
        total_expected = jnp.sum(own_belief, axis=0)
        other_expected = total_expected[None, :, :] - own_belief
        effective_counts = jnp.maximum(available_to_self[None, :, :] - other_expected, 0.0)
        own_weights = own_literal_mask.astype(jnp.float32) * effective_counts
        own_weight_sum = jnp.sum(own_weights, axis=(1, 2), keepdims=True)
        own_belief = own_weights / jnp.maximum(own_weight_sum, 1.0e-8)

    playable_types = rank_grid == heights[:, None]
    obsolete_types = rank_grid < heights[:, None]
    critical_types = (remaining <= 1.01) & (remaining > 0.0) & (~obsolete_types)

    own_play_prob = jnp.sum(own_belief * playable_types[None, :, :], axis=(1, 2))
    own_obsolete_prob = jnp.sum(own_belief * obsolete_types[None, :, :], axis=(1, 2))
    own_critical_prob = jnp.sum(own_belief * critical_types[None, :, :], axis=(1, 2))
    own_support = own_belief > 0.0
    support_n = jnp.sum(own_support, axis=(1, 2))
    support_play_n = jnp.sum(own_support & playable_types[None, :, :], axis=(1, 2))
    certain_play = own_occupied & (support_n > 0) & (support_play_n == support_n)

    last_actor_self = f[253] > 0.5
    last_actor_partner = f[254] > 0.5
    last_play_or_discard = jnp.sum(f[255:257]) > 0.5
    removed_slot = jnp.argmax(f[276:281]).astype(jnp.int32)
    did_own_remove = last_actor_self & last_play_or_discard
    focus_miss = (
        (~reset)
        & memory['pending_focus']
        & last_actor_self
        & (f[255] > 0.5)
        & (~(f[306] > 0.5))
    )
    focus_trust = jnp.where(
        reset, True, memory['focus_trust'] & (~focus_miss)
    )
    base_intent = jnp.where(
        did_own_remove,
        _shift_after_remove(memory['intent'], removed_slot),
        memory['intent'],
    )
    prior_clued = jnp.where(
        did_own_remove,
        _shift_after_remove(memory['clued'], removed_slot),
        memory['clued'],
    )
    base_intent = jnp.where(reset, jnp.zeros((5,), dtype=jnp.bool_), base_intent)
    prior_clued = jnp.where(reset, jnp.zeros((5,), dtype=jnp.bool_), prior_clued)

    received_hint = last_actor_partner & (jnp.sum(f[257:259]) > 0.5) & (f[259] > 0.5)
    hint_matches = f[271:276] > 0.5
    newly_touched = hint_matches & (~prior_clued) & own_occupied
    has_new = jnp.any(newly_touched)
    focus_pool = jnp.where(has_new, newly_touched, hint_matches & own_occupied)
    focus_slot = jnp.max(jnp.where(focus_pool, jnp.arange(5), -1))
    focus_onehot = jnp.arange(5) == focus_slot
    added_intent = received_hint & has_new & jnp.any(focus_pool) & focus_onehot
    intent = base_intent | added_intent
    focused_literal_play = jnp.any(focus_onehot & certain_play)
    received_rank_five_save = received_hint & (f[270] > 0.5) & (~focused_literal_play)

    clue_ids = jnp.arange(10, dtype=jnp.int32)
    is_color_clue = clue_ids < 5
    clue_value = jnp.where(is_color_clue, clue_ids, clue_ids - 5)
    actual_match = jnp.where(
        is_color_clue[:, None],
        partner_color[None, :] == clue_value[:, None],
        partner_rank[None, :] == clue_value[:, None],
    ) & partner_occupied[None, :]

    colors = jnp.arange(5, dtype=jnp.int32)[None, None, :, None]
    ranks = jnp.arange(5, dtype=jnp.int32)[None, None, None, :]
    attr_equal = jnp.where(
        is_color_clue[:, None, None, None],
        colors == clue_value[:, None, None, None],
        ranks == clue_value[:, None, None, None],
    )
    restriction = jnp.where(actual_match[:, :, None, None], attr_equal, ~attr_equal)
    simulated_support = (partner_belief[None, :, :, :] > 0.0) & restriction
    sim_n = jnp.sum(simulated_support, axis=(2, 3))
    sim_play_n = jnp.sum(simulated_support & playable_types[None, None, :, :], axis=(2, 3))
    sim_certain = partner_occupied[None, :] & (sim_n > 0) & (sim_play_n == sim_n)

    current_partner_support = partner_belief > 0.0
    current_n = jnp.sum(current_partner_support, axis=(1, 2))
    current_play_n = jnp.sum(current_partner_support & playable_types[None, :, :], axis=(1, 2))
    current_partner_certain = partner_occupied & (current_n > 0) & (current_play_n == current_n)
    newly_certain_count = jnp.sum(sim_certain & (~current_partner_certain[None, :]), axis=1)

    clue_new_touch = actual_match & (~partner_clued[None, :])
    clue_has_new = jnp.any(clue_new_touch, axis=1)
    clue_focus_pool = jnp.where(clue_has_new[:, None], clue_new_touch, actual_match)
    clue_focus_slot = jnp.max(jnp.where(clue_focus_pool, jnp.arange(5)[None, :], -1), axis=1)
    clue_focus_valid = clue_focus_slot >= 0
    safe_focus_slot = jnp.maximum(clue_focus_slot, 0)
    focus_c = partner_color[safe_focus_slot]
    focus_r = partner_rank[safe_focus_slot]
    clue_focus_playable = clue_focus_valid & (focus_r == heights[focus_c])
    clue_index = jnp.arange(10, dtype=jnp.int32)
    focus_sim_n = sim_n[clue_index, safe_focus_slot]
    focus_sim_play_n = sim_play_n[clue_index, safe_focus_slot]
    focus_sim_certain = clue_focus_valid & (focus_sim_n > 0) & (focus_sim_play_n == focus_sim_n)

    support_reduction = jnp.sum(current_partner_support[None, :, :, :], axis=(1, 2, 3)) - jnp.sum(
        simulated_support, axis=(1, 2, 3)
    )
    matching_playable = actual_match & (partner_rank[None, :] == heights[partner_color][None, :])
    playable_match_count = jnp.sum(matching_playable, axis=1)
    clue_legal = legal[10:20]
    clue_score = (
        40.0 * newly_certain_count.astype(jnp.float32)
        + 15.0 * clue_focus_playable.astype(jnp.float32)
        + 3.0 * playable_match_count.astype(jnp.float32)
        + 0.01 * support_reduction.astype(jnp.float32)
        + 0.5 * clue_has_new.astype(jnp.float32)
    )
    clue_score = jnp.where(clue_legal, clue_score, -1.0e6)
    best_clue = jnp.argmax(clue_score).astype(jnp.int32)
    best_clue_score = clue_score[best_clue]
    rank_five_clue_can_play = (clue_ids != 9) | focus_sim_certain
    play_communicates = (clue_has_new | focus_sim_certain) & rank_five_clue_can_play
    play_clue_score = jnp.where(
        clue_legal & clue_focus_playable & play_communicates, clue_score, -1.0e6
    )
    best_play_clue = jnp.argmax(play_clue_score).astype(jnp.int32)
    best_play_clue_score = play_clue_score[best_play_clue]
    safe_stall = clue_legal & ((focus_sim_play_n == 0) | (~clue_has_new))
    stall_clue_score = jnp.where(safe_stall, clue_score, -1.0e6)
    best_stall_clue = jnp.argmax(stall_clue_score).astype(jnp.int32)
    has_safe_stall = stall_clue_score[best_stall_clue] > -1.0e5
    focus_distance = focus_r - heights[focus_c]
    setup_distance_ok = ((focus_distance == 1) & (info >= 3)) | (
        (focus_distance == 2) & (info >= 6)
    ) | ((focus_distance == 3) & (info >= 8))
    partner_live = partner_occupied & (partner_rank >= heights[partner_color])
    partner_clued_live = partner_clued & partner_live
    safe_setup = clue_legal & clue_has_new & setup_distance_ok & (focus_sim_play_n == 0)
    setup_focus_age = (4 - safe_focus_slot).astype(jnp.float32)
    setup_focus_critical = critical_types[focus_c, focus_r]
    setup_clue_score = jnp.where(
        safe_setup,
        50.0
        + clue_score
        - 0.05 * focus_sim_n.astype(jnp.float32)
        + setup_focus_age
        + 20.0 * setup_focus_critical.astype(jnp.float32),
        -1.0e6,
    )
    best_setup_clue = jnp.argmax(setup_clue_score).astype(jnp.int32)
    should_setup = (
        (deck > 0)
        & (jnp.sum(partner_clued_live) < 4)
        & (setup_clue_score[best_setup_clue] > 0.0)
    )

    actual_partner_critical = critical_types[partner_color, partner_rank]
    older_slot = jnp.arange(5) <= 2
    save_targets = (
        partner_occupied
        & actual_partner_critical
        & (~partner_clued)
        & older_slot
        & (partner_rank != heights[partner_color])
    )
    saves_target = jnp.any(actual_match & save_targets[None, :], axis=1)
    unambiguous_save = clue_legal & saves_target & (focus_sim_play_n == 0)
    save_value = jnp.sum(
        actual_match * save_targets[None, :] * jnp.asarray([5.0, 3.0, 1.0, 0.0, 0.0])[None, :],
        axis=1,
    )
    save_clue_score = jnp.where(unambiguous_save, 100.0 * save_value + clue_score, -1.0e6)
    best_save_clue = jnp.argmax(save_clue_score).astype(jnp.int32)
    should_save = (info > 0) & (deck > 0) & (save_clue_score[best_save_clue] > 0.0)

    certain_priority = certain_play.astype(jnp.float32) * (100.0 + own_play_prob)
    certain_slot = jnp.argmax(certain_priority).astype(jnp.int32)
    has_certain_play = jnp.any(certain_play)
    fresh_intent = added_intent & (own_play_prob > 0.0) & (~received_rank_five_save)
    intent_candidate = (
        fresh_intent
        & (lives >= 2)
        & (focus_trust | (own_play_prob >= 0.90))
    )
    intent_slot = jnp.argmax(jnp.where(intent_candidate, own_play_prob + 0.01 * jnp.arange(5), -1.0))
    has_intent_play = jnp.any(intent_candidate)
    probability_slot = jnp.argmax(own_play_prob + 0.001 * jnp.arange(5)).astype(jnp.int32)
    best_play_probability = own_play_prob[probability_slot]
    high_confidence_play = ((best_play_probability >= 0.90) & (lives >= 2)) | (
        (best_play_probability >= 0.80) & (lives >= 3)
    )
    endgame_probability_play = (deck == 0) & (best_play_probability > 0.0) & (lives >= 2)

    partner_is_playable = partner_occupied & (partner_rank == heights[partner_color])
    partner_needs_play_clue = jnp.any(partner_is_playable & (~current_partner_certain))
    can_give_play_clue = (info > 0) & (best_play_clue_score > 0.0)
    slot_age_bonus = (4.0 - jnp.arange(5, dtype=jnp.float32)) * 0.02
    protected_live_card = own_clued_now & (own_obsolete_prob < 0.999)
    protected_penalty = jnp.where(
        (jnp.sum(protected_live_card) >= 4) & (deck > 10), 20.0, 100.0
    )
    discard_score = (
        80.0 * own_obsolete_prob
        - 35.0 * own_critical_prob
        - 8.0 * own_play_prob
        - protected_penalty * protected_live_card.astype(jnp.float32)
        - 5.0 * intent.astype(jnp.float32)
        + slot_age_bonus
    )
    discard_score = jnp.where(legal[0:5], discard_score, -1.0e6)
    discard_slot = jnp.argmax(discard_score).astype(jnp.int32)
    can_discard = jnp.any(legal[0:5])

    should_clue = can_give_play_clue & (partner_needs_play_clue | (info >= 8)) & (~final_actor_turn)
    fallback_clue = (info > 0) & jnp.any(clue_legal)
    fallback_clue_id = jnp.where(has_safe_stall, best_stall_clue, best_clue)

    action = jnp.where(
        has_intent_play,
        5 + intent_slot,
        jnp.where(
            has_certain_play,
            5 + certain_slot,
            jnp.where(
                high_confidence_play,
                5 + probability_slot,
                jnp.where(
                    should_save,
                    10 + best_save_clue,
                    jnp.where(
                        should_clue,
                        10 + best_play_clue,
                        jnp.where(
                            should_setup,
                            10 + best_setup_clue,
                            jnp.where(
                                endgame_probability_play,
                                5 + probability_slot,
                                jnp.where(
                                    can_discard,
                                    discard_slot,
                                    jnp.where(fallback_clue, 10 + fallback_clue_id, jnp.argmax(legal)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), action)

    next_memory = {
        'intent': intent,
        'clued': own_clued_now,
        'empty_steps': empty_steps,
        'focus_trust': focus_trust,
        'pending_focus': (~legal[20]) & has_intent_play,
    }
    trace = {
        'play_prob': own_play_prob,
        'intent': intent,
        'certain_play': certain_play,
        'best_clue': best_clue,
        'best_clue_score': best_clue_score,
        'best_play_clue': best_play_clue,
        'best_stall_clue': best_stall_clue,
        'best_save_clue': best_save_clue,
        'best_setup_clue': best_setup_clue,
        'should_setup': should_setup,
        'should_save': should_save,
        'discard_score': discard_score,
        'heights': heights,
        'remaining': remaining,
        'partner_color': partner_color,
        'partner_rank': partner_rank,
        'partner_occupied': partner_occupied,
        'info': info,
        'lives': lives,
        'deck': deck,
        'best_play_probability': best_play_probability,
        'reset': reset,
        'empty_steps': empty_steps,
        'focus_trust': focus_trust,
        'pending_focus': memory['pending_focus'],
    }
    return action, next_memory, trace
