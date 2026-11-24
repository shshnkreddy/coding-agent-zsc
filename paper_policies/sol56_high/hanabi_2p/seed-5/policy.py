"""Convention-robust two-player Hanabi policy.

The policy uses only public literal beliefs. Fresh clues are interpreted as
play clues when the touched card is plausibly playable; rank-five and other
low-playability clues naturally behave as save/information clues.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "public_literal_v1",
    "adapter": "hanabi",
    "adapter_version": 1,
    "num_agents": 2,
    "num_colors": 5,
    "num_ranks": 5,
    "hand_size": 5,
    "observation_size": 658,
    "num_actions": 21,
    "policy_structure": "shared_local",
}


def init_agent():
    return {
        "turn": jnp.asarray(0, dtype=jnp.int32),
        "hint_flags": jnp.zeros((5, 10), dtype=jnp.bool_),
    }


def _argmax_masked(values, mask):
    return jnp.argmax(jnp.where(mask, values, -1.0e9)).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions

    partner = f[0:125].reshape((5, 25))
    partner_occ = jnp.sum(partner, axis=1) > 0.5
    partner_color = jnp.argmax(partner.reshape((5, 5, 5)).sum(axis=2), axis=1)
    partner_rank = jnp.argmax(partner.reshape((5, 5, 5)).sum(axis=1), axis=1)

    fw_rows = f[167:192].reshape((5, 5))
    ranks1 = jnp.arange(1, 6, dtype=jnp.float32)
    heights = jnp.sum(fw_rows * ranks1[None, :], axis=1).astype(jnp.int32)
    info = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)

    disc = f[203:253].reshape((5, 10))
    discard_counts = jnp.stack(
        (disc[:, 0:3].sum(1), disc[:, 3:5].sum(1),
         disc[:, 5:7].sum(1), disc[:, 7:9].sum(1), disc[:, 9]), axis=1
    )
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.float32)
    played = jnp.arange(5)[None, :] < heights[:, None]
    remaining = multiplicity[None, :] - discard_counts - played.astype(jnp.float32)

    colors = jnp.repeat(jnp.arange(5), 5)
    ranks = jnp.tile(jnp.arange(5), 5)
    playable_type = ranks == heights[colors]
    already_done = ranks < heights[colors]
    needed = jnp.arange(5)[None, :] >= heights[:, None]
    exhausted_needed = needed & (remaining <= 0.0)
    dead_from = jnp.min(
        jnp.where(exhausted_needed, jnp.arange(5)[None, :], 5), axis=1
    )
    dead_type = already_done | (ranks >= dead_from[colors])
    critical_type = (remaining[colors, ranks] <= 1.0) & (~dead_type)

    knowledge = f[308:658].reshape((2, 5, 35))
    own_belief = knowledge[0, :, 0:25]
    other_belief = knowledge[1, :, 0:25]
    own_hint_flags = knowledge[0, :, 25:35] > 0.5
    prior_hint_flags = jnp.where(
        reset, jnp.zeros((5, 10), dtype=jnp.bool_), memory["hint_flags"]
    )
    own_occ = jnp.sum(own_belief, axis=1) > 0.0
    own_pplay = jnp.sum(own_belief * playable_type[None, :], axis=1)
    visible_counts = jnp.sum(partner, axis=0)
    own_available = jnp.maximum(remaining.reshape((25,)) - visible_counts, 0.0)
    propagated_support = own_belief > 0.0
    slot_available = jnp.broadcast_to(own_available[None, :], (5, 25))
    # Singleton identities elsewhere in our hand consume public copies too.
    # Five fixed passes reach closure for a five-card hand.
    for _ in range(5):
        singleton = (propagated_support
                     & (jnp.sum(propagated_support, axis=1, keepdims=True) == 1))
        singleton_counts = jnp.sum(singleton.astype(jnp.float32), axis=0)
        slot_available = jnp.maximum(
            own_available[None, :] - singleton_counts[None, :]
            + singleton.astype(jnp.float32), 0.0
        )
        propagated_support = propagated_support & (slot_available > 0.0)
    counted_raw = propagated_support.astype(jnp.float32) * slot_available
    counted_belief = counted_raw / jnp.maximum(
        jnp.sum(counted_raw, axis=1, keepdims=True), 1.0e-9
    )
    counted_pplay = jnp.sum(counted_belief * playable_type[None, :], axis=1)
    own_pdead = jnp.sum(own_belief * dead_type[None, :], axis=1)
    own_pcritical = jnp.sum(own_belief * critical_type[None, :], axis=1)
    own_positive = jnp.sum(knowledge[0, :, 25:35], axis=1) > 0.0

    certain_score = jnp.maximum(own_pplay, counted_pplay)
    certain_play = own_occ & (certain_score > 0.999)
    certain_slot = _argmax_masked(certain_score, certain_play)

    received_hint = ((f[254] > 0.5) & ((f[257] > 0.5) | (f[258] > 0.5))
                     & (f[259] > 0.5))
    touched = f[271:276] > 0.5
    newly_touched = touched & jnp.any(own_hint_flags & (~prior_hint_flags), axis=1)
    # V0 is recomputed in each observer's local feature vector.  Tiny floating
    # differences can otherwise reverse an argmax tie between touched cards.
    # Quantization makes clue focus a stable public rule (earliest tied slot).
    clue_focus_score = jnp.round(own_pplay * 10000.0)
    clue_slot = _argmax_masked(clue_focus_score, newly_touched & own_occ)
    clue_pplay = own_pplay[clue_slot]
    last_was_hint = (f[257] > 0.5) | (f[258] > 0.5)
    final_last_turn = (deck == 0) & (last_was_hint | (f[126] > 0.5))
    base_trust_threshold = jnp.where(lives >= 3, 0.24,
                                     jnp.where(lives == 2, 0.38, 0.999))
    unique_trust_threshold = jnp.where(lives >= 3, 0.075,
                                       jnp.where(lives == 2, 0.18, 0.999))
    trust_threshold = jnp.where(jnp.sum(newly_touched) == 1,
                                unique_trust_threshold, base_trust_threshold)
    trust_threshold = jnp.maximum(
        trust_threshold,
        jnp.where(received_hint & (info == 7), 0.34, 0.0),
    )
    trust_threshold = jnp.where(
        received_hint & final_last_turn, 0.0, trust_threshold
    )
    clue_play = (received_hint & jnp.any(newly_touched)
                 & newly_touched[clue_slot] & own_occ[clue_slot]
                 & (clue_pplay >= trust_threshold))

    clue_ids = jnp.arange(10)
    clue_is_color = clue_ids < 5
    clue_value = jnp.where(clue_is_color, clue_ids, clue_ids - 5)
    type_match = jnp.where(
        clue_is_color[:, None],
        colors[None, :] == clue_value[:, None],
        ranks[None, :] == clue_value[:, None],
    )
    actual_match = jnp.where(
        clue_is_color[:, None],
        partner_color[None, :] == clue_value[:, None],
        partner_rank[None, :] == clue_value[:, None],
    ) & partner_occ[None, :]
    partner_hint_flags = knowledge[1, :, 25:35] > 0.5
    candidate_new = actual_match & (~partner_hint_flags.T)
    candidate_has_new = jnp.any(candidate_new, axis=1)
    candidate_focus = jnp.where(candidate_has_new[:, None], candidate_new, actual_match)
    restricted = other_belief[None, :, :] * type_match[:, None, :]
    denom = jnp.sum(restricted, axis=2)
    post_pplay = (jnp.sum(restricted * playable_type[None, None, :], axis=2)
                  / jnp.maximum(denom, 1.0e-9))
    post_focus_score = jnp.round(post_pplay * 10000.0)
    predicted_slot = jnp.argmax(
        jnp.where(candidate_focus, post_focus_score, -1.0), axis=1
    )
    qidx = jnp.arange(10)
    predicted_p = post_pplay[qidx, predicted_slot]
    true_playable = partner_rank == heights[partner_color]
    predicted_true_play = true_playable[predicted_slot]
    true_critical = critical_type.reshape((5, 5))[partner_color, partner_rank]
    predicted_true_critical = true_critical[predicted_slot]
    other_pre_pplay = jnp.sum(other_belief * playable_type[None, :], axis=1)
    novelty = 1.0 - other_pre_pplay[predicted_slot]
    nonplay_touches = jnp.sum(candidate_focus & (~true_playable[None, :]), axis=1)
    touch_count = jnp.sum(candidate_focus, axis=1)
    all_touched_playable = nonplay_touches == 0
    universal_clue = (touch_count == 1) | all_touched_playable
    candidate_threshold = jnp.where(touch_count == 1,
                                    unique_trust_threshold, base_trust_threshold)
    candidate_threshold = jnp.maximum(
        candidate_threshold, jnp.where(info == 8, 0.34, 0.0)
    )
    candidate_threshold = jnp.where(deck == 0, 0.0, candidate_threshold)
    rank_bonus = (~clue_is_color).astype(jnp.float32) * 0.075
    clue_score = (predicted_p + 0.20 * novelty + rank_bonus
                  + 0.50 * (touch_count == 1).astype(jnp.float32)
                  + 0.30 * all_touched_playable.astype(jnp.float32)
                  + 0.20 * predicted_true_critical.astype(jnp.float32)
                  - 0.075 * nonplay_touches)
    clue_legal = legal[10:20]
    actionable_clue = (clue_legal & candidate_has_new & predicted_true_play
                       & (predicted_p >= candidate_threshold)
                       & (novelty > 0.001))
    best_clue = _argmax_masked(clue_score, actionable_clue)
    have_actionable_clue = jnp.any(actionable_clue)

    partner_positive = jnp.any(partner_hint_flags, axis=1)
    unprotected_critical = partner_occ & true_critical & (~partner_positive)
    save_targets = candidate_new & unprotected_critical[None, :]
    save_candidate = (clue_legal & jnp.any(save_targets, axis=1)
                      & (predicted_p < candidate_threshold))
    save_score = (jnp.sum(save_targets, axis=1).astype(jnp.float32)
                  + (~clue_is_color).astype(jnp.float32) * 0.10
                  - 0.15 * jnp.sum(candidate_new & (~save_targets), axis=1)
                  - predicted_p)
    best_save = _argmax_masked(save_score, save_candidate)
    have_save = jnp.any(save_candidate) & (info > 0) & (deck > 0)

    age_bonus = (4 - jnp.arange(5)).astype(jnp.float32) * 0.40
    discard_score = (5.0 * own_pdead + 15.0 * (own_pdead > 0.999)
                     - 7.0 * own_pcritical
                     - 2.0 * own_positive.astype(jnp.float32)
                     - 0.5 * own_pplay + age_bonus)
    discard_slot = _argmax_masked(discard_score, legal[0:5] & own_occ)

    repeated_clue = clue_legal & (~candidate_has_new)
    fallback_mask = jnp.where(jnp.any(repeated_clue), repeated_clue, clue_legal)
    fallback_info_score = (-predicted_p + 0.03 * (~clue_is_color)
                           - 0.01 * jnp.sum(actual_match, axis=1))
    fallback_clue = _argmax_masked(fallback_info_score, fallback_mask)

    risky_threshold = jnp.where(lives >= 3, 0.25,
                                jnp.where(lives == 2, 0.40, 0.999))
    risky_slot = _argmax_masked(own_pplay, legal[5:10] & own_occ)
    endgame_risky_play = ((deck <= 1) & (own_pplay[risky_slot] >= risky_threshold))
    active = ~legal[20]
    final_attempt = ((deck == 0) & (lives > 1) & (own_pplay[risky_slot] > 0.0)
                     & (final_last_turn | (~have_actionable_clue)))

    action = jnp.asarray(20, dtype=jnp.int32)
    action = jnp.where(endgame_risky_play, 5 + risky_slot, action)
    action = jnp.where(info == 8, 10 + fallback_clue, action)
    can_discard = jnp.any(legal[0:5])
    action = jnp.where(can_discard, discard_slot, action)
    action = jnp.where(have_save, 10 + best_save, action)
    action = jnp.where(have_actionable_clue, 10 + best_clue, action)
    action = jnp.where(final_attempt, 5 + risky_slot, action)
    action = jnp.where(clue_play, 5 + clue_slot, action)
    action = jnp.where(jnp.any(certain_play), 5 + certain_slot, action)
    action = jnp.where(active, action, 20).astype(jnp.int32)

    prior_count = jnp.where(reset, jnp.asarray(0, jnp.int32), memory["turn"])
    shifted_hint_flags = jnp.concatenate(
        (own_hint_flags[1:, :], jnp.zeros((1, 10), dtype=jnp.bool_)), axis=0
    )
    removes_own_card = active & (action < 10)
    next_memory = {
        "turn": prior_count + active.astype(jnp.int32),
        "hint_flags": jnp.where(removes_own_card, shifted_hint_flags, own_hint_flags),
    }
    trace = {
        "best_clue": best_clue,
        "chosen_hint_p": predicted_p[jnp.clip(action - 10, 0, 9)],
        "chosen_hint_slot": predicted_slot[jnp.clip(action - 10, 0, 9)],
        "chosen_hint_true": predicted_true_play[jnp.clip(action - 10, 0, 9)],
        "clue_slot": clue_slot,
        "deck": deck,
        "have_actionable": have_actionable_clue,
        "have_save": have_save,
        "info": info,
        "lives": lives,
        "pplay": own_pplay.astype(jnp.float32),
        "pdead": own_pdead.astype(jnp.float32),
        "received_hint": received_hint.astype(jnp.bool_),
        "partner_playable": true_playable,
        "partner_rank": partner_rank,
        "score": jnp.sum(heights).astype(jnp.int32),
        "heights": heights,
        "turn": prior_count,
        "chosen": action,
    }
    return action, next_memory, trace
