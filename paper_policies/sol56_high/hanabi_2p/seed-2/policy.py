"""Convention-light literal-belief policy for two-player Hanabi."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'reachable_critical_protection', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        'intent': jnp.zeros(5, dtype=jnp.bool_),
        'positive': jnp.zeros((5, 10), dtype=jnp.bool_),
        'zero_steps': jnp.asarray(0, dtype=jnp.int32),
        'rank_trust': jnp.asarray(1, dtype=jnp.int32),
        'out_pending': jnp.asarray(False, dtype=jnp.bool_),
        'out_focus': jnp.asarray(0, dtype=jnp.int32),
        'out_trust': jnp.asarray(2, dtype=jnp.int32),
        'protected': jnp.zeros(5, dtype=jnp.bool_),
    }


def _discard_counts(features):
    d = features[203:253].reshape(5, 10)
    return jnp.stack(
        [jnp.sum(d[:, 0:3], axis=1),
         jnp.sum(d[:, 3:5], axis=1),
         jnp.sum(d[:, 5:7], axis=1),
         jnp.sum(d[:, 7:9], axis=1),
         d[:, 9]], axis=1)


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    deck_count = jnp.sum(f[127:167])
    zero_steps = jnp.where(
        reset | (deck_count > 0.5),
        jnp.asarray(0, jnp.int32),
        jnp.minimum(memory['zero_steps'] + 1, 2))
    hand = f[0:125].reshape(5, 5, 5)
    occupied_partner = jnp.sum(hand, axis=(1, 2)) > 0.5
    actual_flat = hand.reshape(5, 25)
    actual_color = jnp.argmax(jnp.sum(hand, axis=2), axis=1)
    actual_rank = jnp.argmax(jnp.sum(hand, axis=1), axis=1)

    fw = f[167:192].reshape(5, 5)
    heights = jnp.sum(fw * jnp.arange(1, 6, dtype=f.dtype)[None, :], axis=1).astype(jnp.int32)
    type_color = jnp.repeat(jnp.arange(5), 5)
    type_rank = jnp.tile(jnp.arange(5), 5)
    playable_flat = type_rank == heights[type_color]
    obsolete_flat = type_rank < heights[type_color]

    discards = _discard_counts(f)
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=f.dtype)
    already_played = type_rank < heights[type_color]
    public_remaining = multiplicity[type_rank] - discards.reshape(25) - already_played.astype(f.dtype)

    knowledge = f[308:658].reshape(2, 5, 35)
    weights = knowledge[:, :, :25]
    possible = weights > 1.0e-7
    occupied = jnp.sum(weights, axis=2) > 1.0e-7

    # Native V0 omits cards visible in the partner's hand. Refine only our own
    # beliefs with those counts, then iterate exact singleton eliminations.
    visible_counts = jnp.sum(actual_flat, axis=0)
    self_counts = jnp.maximum(public_remaining - visible_counts, 0.0)
    self_possible = possible[0] & (self_counts[None, :] > 0.0)
    available_for_slot = jnp.broadcast_to(self_counts[None, :], (5, 25))
    subset_ids = jnp.arange(1, 32, dtype=jnp.int32)
    subset_bits = ((subset_ids[:, None] >> jnp.arange(5, dtype=jnp.int32)[None, :]) & 1) > 0
    for _ in range(3):
        singleton = self_possible & (jnp.sum(self_possible, axis=1, keepdims=True) == 1)
        assigned = jnp.sum(singleton.astype(f.dtype), axis=0)
        available_for_slot = self_counts[None, :] - assigned[None, :] + singleton.astype(f.dtype)
        self_possible = self_possible & (available_for_slot > 0.0)
        selected = subset_bits & occupied[0][None, :]
        subset_union = jnp.any(selected[:, :, None] & self_possible[None, :, :], axis=1)
        subset_size = jnp.sum(selected, axis=1)
        subset_capacity = jnp.sum(subset_union.astype(f.dtype) * self_counts[None, :], axis=1)
        saturated = (subset_size > 0) & (subset_capacity <= subset_size.astype(f.dtype) + 0.01)
        hall_blocked = jnp.any(
            saturated[:, None, None] & (~selected[:, :, None]) & subset_union[:, None, :],
            axis=0)
        self_possible = self_possible & (~hall_blocked)
    possible = possible.at[0].set(self_possible)
    own_raw_weights = self_possible.astype(f.dtype) * jnp.maximum(available_for_slot, 0.0)
    own_weight_sum = jnp.maximum(jnp.sum(own_raw_weights, axis=1, keepdims=True), 1.0e-8)
    weights = weights.at[0].set(own_raw_weights / own_weight_sum)
    support_n = jnp.sum(possible, axis=2)
    certain_play = occupied & (jnp.sum(possible & (~playable_flat[None, None, :]), axis=2) == 0)
    certain_obsolete = occupied & (jnp.sum(possible & (~obsolete_flat[None, None, :]), axis=2) == 0)

    # Update private focused-play intentions from the last public transition.
    last = f[253:308]
    last_actor_self = last[0] > 0.5
    last_actor_partner = last[1] > 0.5
    last_removed = (last[2] > 0.5) | (last[3] > 0.5)
    last_hint = (last[4] > 0.5) | (last[5] > 0.5)
    last_target_self = last[6] > 0.5
    removed_slot = jnp.argmax(last[23:28]).astype(jnp.int32)
    response_due = memory['out_pending'] & last_actor_partner & (~reset)
    response_followed = ((last[2] > 0.5)
                         & (removed_slot == memory['out_focus'])
                         & (last[53] > 0.5))
    out_trust = jnp.where(
        response_due,
        jnp.where(response_followed,
                  jnp.minimum(memory['out_trust'] + 1, 3),
                  jnp.maximum(memory['out_trust'] - 1, 0)),
        memory['out_trust'])
    acted_on_intent = last_actor_self & (last[2] > 0.5) & memory['intent'][removed_slot] & (~reset)
    intended_play_succeeded = last[53] > 0.5
    rank_trust = jnp.where(
        acted_on_intent & intended_play_succeeded,
        jnp.minimum(memory['rank_trust'] + 1, 3),
        jnp.where(acted_on_intent, jnp.asarray(0, jnp.int32), memory['rank_trust']))
    idx = jnp.arange(5)
    shifted_idx = jnp.minimum(idx + (idx >= removed_slot).astype(jnp.int32), 4)
    shifted_intent = jnp.where(idx < 4, memory['intent'][shifted_idx], False)
    intent = jnp.where(last_actor_self & last_removed & (~reset), shifted_intent, memory['intent'])
    shifted_protected = jnp.where(idx < 4, memory['protected'][shifted_idx], False)
    protected = jnp.where(last_actor_self & last_removed & (~reset),
                          shifted_protected, memory['protected'])

    prior_positive = memory['positive']
    named_attr = jnp.argmax(jnp.concatenate([last[8:13], last[13:18]])).astype(jnp.int32)
    matches = last[18:23] > 0.5
    was_known_attr = prior_positive[:, named_attr]
    newly_touched = matches & (~was_known_attr)
    focus_pool = jnp.where(jnp.any(newly_touched), newly_touched, matches)
    focus = jnp.argmax(jnp.where(focus_pool, idx + 1, 0)).astype(jnp.int32)
    prior_any_hint = jnp.any(prior_positive, axis=1)
    prior_chop = jnp.argmax(jnp.where(occupied[0] & (~prior_any_hint), 5 - idx, -100)).astype(jnp.int32)
    received_hint = last_actor_partner & last_hint & last_target_self & (~reset)
    clue_is_save = matches[prior_chop] & (~certain_play[0, prior_chop])
    received_color_clue = received_hint & (last[4] > 0.5)
    protected = protected & occupied[0]
    # Rank clues use focus as play intent. Color clues remain literal and are
    # also available for saves and forced information clues.
    focus_play_probability = jnp.sum(weights[0, focus] * playable_flat.astype(f.dtype))
    required_play_probability = jnp.where(
        rank_trust >= 2, 0.10, jnp.where(rank_trust == 1, 0.14, 0.5))
    received_rank_clue = received_hint & (last[5] > 0.5)
    rank_clue_compatible = focus_play_probability > required_play_probability
    add_intent = received_rank_clue & rank_clue_compatible
    rank_trust = jnp.where(received_rank_clue & (~rank_clue_compatible), 0, rank_trust)
    intent = intent | (idx == focus) & add_intent
    intent = intent & occupied[0] & (~certain_obsolete[0])

    age = (5 - jnp.arange(5)).astype(f.dtype)
    playable_by_policy = certain_play[0] | intent
    play_scores = jnp.where(playable_by_policy, 100.0 + 20.0 * intent.astype(f.dtype) + age, -1.0e6)
    play_slot = jnp.argmax(play_scores).astype(jnp.int32)
    have_play = jnp.any(playable_by_policy)

    # Simulate literal native knowledge after each possible color/rank hint.
    cand = jnp.arange(10)
    is_color_hint = cand < 5
    value = cand % 5
    tc = type_color[None, None, :]
    tr = type_rank[None, None, :]
    v = value[:, None, None]
    attr_type_match = jnp.where(is_color_hint[:, None, None], tc == v, tr == v)
    actual_match = jnp.where(
        is_color_hint[:, None],
        actual_color[None, :] == value[:, None],
        actual_rank[None, :] == value[:, None])
    literal_keep = jnp.where(actual_match[:, :, None], attr_type_match, ~attr_type_match)
    post_possible = possible[1][None, :, :] & literal_keep
    post_n = jnp.sum(post_possible, axis=2)
    post_certain_play = occupied_partner[None, :] & (
        jnp.sum(post_possible & (~playable_flat[None, None, :]), axis=2) == 0)
    new_play = post_certain_play & (~certain_play[1][None, :])
    actual_playable = jnp.sum(actual_flat * playable_flat[None, :], axis=1) > 0.5
    new_actual_play = new_play & actual_playable[None, :]
    new_play_count = jnp.sum(new_actual_play, axis=1)
    earliest_new = jnp.max(jnp.where(new_actual_play, age[None, :], 0.0), axis=1)

    hint_legal = legal[10:20]
    info_gain = jnp.sum(jnp.maximum(support_n[1][None, :] - post_n, 0), axis=1).astype(f.dtype)
    useful_hint_score = 100.0 * new_play_count + 3.0 * earliest_new + 0.01 * info_gain

    # Focus is the rightmost card newly touched by an attribute; if a repeated
    # clue adds no positive flag, it is the rightmost matching card.
    partner_positive = knowledge[1, :, 25:35] > 0.5
    candidate_prior_attr = jnp.transpose(partner_positive)[cand, :]
    candidate_newtouch = actual_match & (~candidate_prior_attr)
    focus_hint_pool = jnp.where(jnp.any(candidate_newtouch, axis=1)[:, None], candidate_newtouch, actual_match)
    hint_focus = jnp.argmax(jnp.where(focus_hint_pool, idx[None, :] + 1, 0), axis=1)
    focus_is_playable = actual_playable[hint_focus]

    # Protect a visibly unique, still-needed card if it is the partner's likely
    # oldest-unhinted discard.
    remaining = public_remaining
    needed = type_rank >= heights[type_color]
    missing = remaining.reshape(5, 5) < 0.01
    prerequisite_rank = jnp.arange(5)[None, None, :]
    target_rank = jnp.arange(5)[None, :, None]
    in_unbuilt_prefix = ((prerequisite_rank >= heights[:, None, None])
                         & (prerequisite_rank < target_rank))
    blocked = jnp.any(in_unbuilt_prefix & missing[:, None, :], axis=2)
    dead_type = obsolete_flat | blocked.reshape(25)
    critical_type = (remaining <= 1.01) & needed & (~dead_type)
    certain_dead = occupied & (jnp.sum(possible & (~dead_type[None, None, :]), axis=2) == 0)
    critical_prob = jnp.sum(weights * critical_type[None, None, :], axis=2)
    downstream_value = (5 - type_rank).astype(f.dtype)
    critical_loss_prob = jnp.sum(
        weights * (critical_type * downstream_value)[None, None, :], axis=2)
    chop_critical_prob = critical_prob[0, prior_chop]
    color_save_focus = jnp.argmax(jnp.where(matches, idx + 1, 0)).astype(jnp.int32)
    new_save_slots = (received_color_clue & (idx == color_save_focus)
                      & (~certain_play[0]) & (critical_prob[0] > 0.05))
    new_save = jnp.any(new_save_slots)
    protected = protected | new_save_slots
    useful_prob = jnp.sum(weights * needed[None, None, :], axis=2)
    positively_hinted = jnp.sum(knowledge[:, :, 25:35], axis=2) > 0.5
    visible_unprotected_type = jnp.sum(
        actual_flat * (~positively_hinted[1])[:, None].astype(f.dtype), axis=0) > 0.5
    two_copy_type = (remaining > 1.01) & (remaining <= 2.01) & needed
    exposure_type = two_copy_type & visible_unprotected_type

    discard_score = (1000.0 * certain_dead.astype(f.dtype)
                     + 20.0 * (~positively_hinted).astype(f.dtype)
                     + age[None, :]
                     - 42.0 * critical_loss_prob
                     - 5.0 * useful_prob)
    discard_score = discard_score.at[0].add(
        -15.0 * (protected & (~certain_dead[0])).astype(f.dtype))
    discard_score = jnp.where(occupied, discard_score, -1.0e6)
    own_discard_slot = jnp.argmax(discard_score[0]).astype(jnp.int32)
    partner_discard_slot = jnp.argmax(discard_score[1]).astype(jnp.int32)
    partner_actual_critical = jnp.sum(actual_flat * critical_type[None, :], axis=1) > 0.5
    partner_chop = jnp.argmax(jnp.where(occupied[1] & (~positively_hinted[1]), age, -100.0)).astype(jnp.int32)
    unprotected_critical = partner_actual_critical & (~positively_hinted[1])
    threatened_type = exposure_type & (weights[0, own_discard_slot] > 0.5)
    preexposure_card = (jnp.sum(actual_flat * threatened_type[None, :], axis=1) > 0.5) & (~positively_hinted[1])
    protection_targets = unprotected_critical | preexposure_card
    needs_save = jnp.any(protection_targets)

    saved_count = jnp.sum(actual_match & protection_targets[None, :], axis=1)
    save_score = 100.0 * saved_count + 0.01 * info_gain

    direct_valid = focus_is_playable & (~is_color_hint)
    direct_score = (100.0 + 10.0 * candidate_newtouch[cand, hint_focus].astype(f.dtype)
                    + 2.0 * age[hint_focus] + 0.01 * info_gain)

    general_hint_score = jnp.where(is_color_hint, info_gain, -1.0e6)
    useful_safe = is_color_hint | focus_is_playable
    useful_hint_score = jnp.where(hint_legal & useful_safe, useful_hint_score, -1.0e6)
    save_score = jnp.where(hint_legal & is_color_hint, save_score, -1.0e6)
    direct_score = jnp.where(hint_legal & direct_valid, direct_score, -1.0e6)
    general_hint_score = jnp.where(hint_legal, general_hint_score, -1.0e6)
    best_useful = jnp.argmax(useful_hint_score).astype(jnp.int32)
    best_save = jnp.argmax(save_score).astype(jnp.int32)
    best_direct = jnp.argmax(direct_score).astype(jnp.int32)
    best_general = jnp.argmax(general_hint_score).astype(jnp.int32)
    have_useful_hint = jnp.any(hint_legal & useful_safe & (new_play_count > 0))
    have_direct_hint = jnp.any(hint_legal & direct_valid)
    have_save_hint = jnp.any(hint_legal & is_color_hint & (saved_count > 0))
    have_hint = jnp.any(hint_legal)
    partner_ready = jnp.any(certain_play[1])
    discard_legal = legal[own_discard_slot]
    surplus_info = ((jnp.sum(f[192:200]) >= 6.0) & (~partner_ready)
                    & (info_gain[best_general] > 0.0))

    fallback = jnp.argmax(legal).astype(jnp.int32)
    action = jnp.where(have_hint, 10 + best_general, fallback)
    action = jnp.where(discard_legal, own_discard_slot, action)
    action = jnp.where(have_hint & surplus_info, 10 + best_general, action)
    action = jnp.where(have_save_hint & needs_save & (~partner_ready), 10 + best_save, action)
    action = jnp.where(have_direct_hint & (~partner_ready) & (rank_trust > 0)
                       & (out_trust > 0),
                       10 + best_direct, action)
    action = jnp.where(have_useful_hint & (~partner_ready), 10 + best_useful, action)
    action = jnp.where(have_play, 5 + play_slot, action)

    play_probability = jnp.sum(weights[0] * playable_flat[None, :], axis=1)
    probable_slot = jnp.argmax(jnp.where(occupied[0], play_probability, -1.0)).astype(jnp.int32)
    lives = jnp.sum(f[200:203])
    current_score = jnp.sum(heights.astype(f.dtype))
    break_even = current_score / (current_score + 1.0)
    last_turn_speculation = ((zero_steps >= 2) & (~have_play) & jnp.any(occupied[0])
                             & ((lives >= 2.0) | (play_probability[probable_slot] > break_even)))
    penultimate_speculation = ((zero_steps == 1) & (~have_play) & (lives >= 2.0)
                               & jnp.any(occupied[0])
                               & (partner_ready | (~jnp.any(actual_playable))))
    final_speculation = last_turn_speculation | penultimate_speculation
    action = jnp.where(final_speculation, 5 + probable_slot, action)
    action = jnp.where(legal[20], jnp.asarray(20, jnp.int32), action)

    sent_direct = ((action == 10 + best_direct) & have_direct_hint
                   & (~have_useful_hint) & (~partner_ready)
                   & (rank_trust > 0) & (out_trust > 0))
    out_pending = jnp.where(response_due, False, memory['out_pending']) | sent_direct
    out_focus = jnp.where(sent_direct, hint_focus[best_direct], memory['out_focus'])

    current_positive = knowledge[0, :, 25:35] > 0.5
    next_memory = {
        'intent': jnp.where(reset, jnp.zeros(5, dtype=jnp.bool_), intent),
        'positive': jnp.where(reset, jnp.zeros((5, 10), dtype=jnp.bool_), current_positive),
        'zero_steps': jnp.where(reset, jnp.asarray(0, jnp.int32), zero_steps),
        'rank_trust': jnp.where(reset, jnp.asarray(1, jnp.int32), rank_trust),
        'out_pending': out_pending,
        'out_focus': jnp.where(reset & (~sent_direct), jnp.asarray(0, jnp.int32), out_focus),
        'out_trust': jnp.where(reset, jnp.asarray(2, jnp.int32), out_trust),
        'protected': jnp.where(reset, jnp.zeros(5, dtype=jnp.bool_), protected),
    }
    trace = {
        'certain_play': certain_play[0],
        'partner_ready': partner_ready,
        'useful_hint': have_useful_hint,
        'direct_hint': have_direct_hint,
        'intent': intent,
        'rank_trust': rank_trust,
        'out_pending': out_pending,
        'out_trust': out_trust,
        'response_due': response_due,
        'response_followed': response_followed,
        'protected': protected,
        'new_save': new_save,
        'new_save_slots': new_save_slots,
        'chop_critical_prob': chop_critical_prob,
        'critical_loss_prob': critical_loss_prob[0],
        'save': needs_save,
        'surplus_info': surplus_info,
        'discard_slot': own_discard_slot,
        'reset_seen': reset,
        'heights': heights,
        'deck': jnp.sum(f[127:167]),
        'tokens': jnp.sum(f[192:200]),
        'lives': lives,
        'last_type': jnp.argmax(last[2:6]).astype(jnp.int32),
        'last_card': jnp.argmax(last[28:53]).astype(jnp.int32),
        'last_success': last[53] > 0.5,
        'final_speculation': final_speculation,
    }
    return action.astype(jnp.int32), next_memory, trace
