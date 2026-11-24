"""Literal-belief Hanabi policy.

The policy deliberately bases its strongest actions on public, literal card
knowledge.  This makes its clues and plays useful with partners that do not
share a private convention.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions
    turn = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)

    # Public board and hands.
    partner = f[0:125].reshape(5, 5, 5)
    fireworks_oh = f[167:192].reshape(5, 5)
    fireworks = jnp.sum(
        fireworks_oh * jnp.arange(1, 6, dtype=jnp.float32)[None, :], axis=1
    ).astype(jnp.int32)
    info = jnp.sum(f[192:200]).astype(jnp.int32)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)
    knowledge = f[308:658].reshape(2, 5, 35)
    native_own_belief = knowledge[0, :, :25].reshape(5, 5, 5)
    partner_belief = knowledge[1, :, :25].reshape(5, 5, 5)

    colors = jnp.arange(5, dtype=jnp.int32)[:, None]
    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    playable_type = ranks == fireworks[:, None]
    obsolete_type = ranks < fireworks[:, None]

    # Exact public remaining-copy counts (successful lower stack cards and all
    # discards are public). These distinguish expendable duplicates from the
    # last copy of a still-needed card.
    discard_raw = f[203:253].reshape(5, 10)
    discard_counts = jnp.stack([
        jnp.sum(discard_raw[:, 0:3], axis=1),
        jnp.sum(discard_raw[:, 3:5], axis=1),
        jnp.sum(discard_raw[:, 5:7], axis=1),
        jnp.sum(discard_raw[:, 7:9], axis=1),
        jnp.sum(discard_raw[:, 9:10], axis=1),
    ], axis=1)
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.float32)[None, :]
    remaining = multiplicity - discard_counts - obsolete_type.astype(jnp.float32)
    critical_type = (~obsolete_type) & (remaining <= 1.0)
    surplus_type = (~obsolete_type) & (remaining >= 2.0)

    # Native V0 intentionally omits the five cards visible to this observer.
    # Preserve its exact literal support, but condition multiplicity weights on
    # those visible cards before making private play/discard judgments.
    visible_partner_counts = jnp.sum(partner, axis=0)
    available_to_own = jnp.maximum(remaining - visible_partner_counts, 0.0)
    own_raw = ((native_own_belief > 0.0).astype(jnp.float32)
               * available_to_own[None, :, :])
    own_norm = jnp.sum(own_raw, axis=(1, 2), keepdims=True)
    own_belief = own_raw / jnp.maximum(own_norm, 1e-6)

    own_mass = jnp.sum(own_belief, axis=(1, 2))
    own_play_prob = jnp.sum(
        own_belief * playable_type[None, :, :], axis=(1, 2)
    )
    own_obsolete_prob = jnp.sum(
        own_belief * obsolete_type[None, :, :], axis=(1, 2)
    )
    discard_type_value = (3.0 * obsolete_type.astype(jnp.float32)
                          + 1.45 * surplus_type.astype(jnp.float32)
                          - 5.0 * critical_type.astype(jnp.float32))
    native_own_mass = jnp.sum(native_own_belief, axis=(1, 2))
    native_discard_value = jnp.sum(
        native_own_belief * discard_type_value[None, :, :], axis=(1, 2)
    ) / jnp.maximum(native_own_mass, 1e-6)
    native_own_critical_prob = jnp.sum(
        native_own_belief * critical_type[None, :, :], axis=(1, 2)
    ) / jnp.maximum(native_own_mass, 1e-6)
    zero_token_type_value = (3.0 * obsolete_type.astype(jnp.float32)
                             + 1.45 * surplus_type.astype(jnp.float32)
                             - 5.0 * critical_type.astype(jnp.float32))
    refined_discard_value = jnp.sum(
        own_belief * zero_token_type_value[None, :, :], axis=(1, 2)
    ) / jnp.maximum(own_mass, 1e-6)
    own_discard_value = jnp.where(
        info == 0, refined_discard_value, native_discard_value
    )
    own_hinted = jnp.any(knowledge[0, :, 25:35] > 0.5, axis=1)
    own_discard_value = (own_discard_value
                         - 1.5 * own_hinted.astype(jnp.float32)
                         - 0.25 * own_hinted.astype(jnp.float32)
                         * native_own_critical_prob
                         - 0.01 * jnp.arange(5, dtype=jnp.float32))
    certain_play = (own_mass > 0.5) & (own_play_prob > own_mass - 1e-5)
    certain_obsolete = (own_mass > 0.5) & (own_obsolete_prob > own_mass - 1e-5)

    # Evaluate each literal hint by applying its positive and negative evidence
    # to the recipient's current public belief.  Score newly certain playable
    # cards most highly, with information gain as a small tie breaker.
    partner_present = jnp.sum(partner, axis=(1, 2)) > 0.5
    partner_c = jnp.argmax(jnp.sum(partner, axis=2), axis=1)
    partner_r = jnp.argmax(jnp.sum(partner, axis=1), axis=1)
    partner_mass = jnp.sum(partner_belief, axis=(1, 2))
    partner_play_mass = jnp.sum(
        partner_belief * playable_type[None, :, :], axis=(1, 2)
    )
    partner_certain_play = ((partner_mass > 0.0)
                            & (partner_play_mass > partner_mass - 1e-5))
    partner_discard_value = jnp.sum(
        partner_belief * discard_type_value[None, :, :], axis=(1, 2)
    ) / jnp.maximum(partner_mass, 1e-6)
    partner_critical_prob = jnp.sum(
        partner_belief * critical_type[None, :, :], axis=(1, 2)
    ) / jnp.maximum(partner_mass, 1e-6)
    partner_hinted = jnp.any(knowledge[1, :, 25:35] > 0.5, axis=1)
    partner_discard_value = (partner_discard_value
                             - 1.5 * partner_hinted.astype(jnp.float32)
                             - 0.25 * partner_hinted.astype(jnp.float32)
                             * partner_critical_prob
                             - 0.01 * jnp.arange(5, dtype=jnp.float32))
    predicted_partner_discard = jnp.argmax(
        jnp.where(partner_present, partner_discard_value, -1e9)
    )
    save_c = partner_c[predicted_partner_discard]
    save_r = partner_r[predicted_partner_discard]
    partner_actual_playable = playable_type[partner_c, partner_r] & partner_present
    partner_actual_critical = critical_type[partner_c, partner_r] & partner_present
    save_is_critical = critical_type[save_c, save_r]
    at_risk_critical = save_is_critical
    save_color_known = knowledge[1, predicted_partner_discard, 25 + save_c] > 0.5
    save_rank_known = knowledge[1, predicted_partner_discard, 30 + save_r] > 0.5
    rightmost_same_rank = jnp.argmax(jnp.where(
        (partner_r == save_r) & partner_present,
        jnp.arange(5, dtype=jnp.int32), -1,
    ))
    completion_focus_belief = (partner_belief[rightmost_same_rank]
                               * (ranks == save_r))
    completion_focus_mass = jnp.sum(completion_focus_belief)
    completion_focus_play = jnp.sum(completion_focus_belief * playable_type)
    completion_focus_prob = completion_focus_play / jnp.maximum(
        completion_focus_mass, 1e-6
    )
    completion_threshold = jnp.where(
        jnp.sum(f[200:203]) <= 1.0, 0.60, 0.10
    )
    can_complete_rank = (save_color_known & (~save_rank_known)
                         & (predicted_partner_discard == rightmost_same_rank))
    can_complete_color = save_rank_known & (~save_color_known)
    needs_save_information = ((~partner_hinted[predicted_partner_discard])
                              | can_complete_rank | can_complete_color)
    unambiguous_rank_save = (save_r == 4) & (jnp.max(fireworks) < 4)
    initial_save_action = jnp.where(
        unambiguous_rank_save, 15 + save_r, 10 + save_c
    ).astype(jnp.int32)
    completion_action = jnp.where(
        can_complete_color, 10 + save_c, 15 + save_r
    ).astype(jnp.int32)
    save_action = jnp.where(
        partner_hinted[predicted_partner_discard],
        completion_action, initial_save_action,
    ).astype(jnp.int32)
    need_save = (at_risk_critical & needs_save_information
                 & (~jnp.any(partner_actual_playable)) & legal[save_action])

    slot_index = jnp.arange(5, dtype=jnp.int32)

    def color_hint_score(c):
        matches = partner_c == c
        mask = jnp.where(
            matches[:, None, None],
            colors[None, :, :] == c,
            colors[None, :, :] != c,
        )
        post = partner_belief * mask
        mass = jnp.sum(post, axis=(1, 2))
        play_mass = jnp.sum(post * playable_type[None, :, :], axis=(1, 2))
        safe = matches & partner_present & (mass > 0.0) & (play_mass > mass - 1e-5)
        removed = jnp.sum((partner_belief > 0.0) & (~mask))
        # Color clues are literal/informational, never an implicit play order.
        new_color = ~(knowledge[1, :, 25 + c] > 0.5)
        setup = matches & partner_actual_playable & new_color
        return (1000.0 * jnp.sum(safe)
                + 100.0 * jnp.sum(safe & partner_actual_critical)
                + 40.0 * jnp.sum(setup)
                - jnp.sum(matches) + 0.01 * removed)

    def rank_hint_score(r):
        matches = partner_r == r
        mask = jnp.where(
            matches[:, None, None],
            ranks[None, :, :] == r,
            ranks[None, :, :] != r,
        )
        post = partner_belief * mask
        mass = jnp.sum(post, axis=(1, 2))
        play_mass = jnp.sum(post * playable_type[None, :, :], axis=(1, 2))
        safe = matches & partner_present & (mass > 0.0) & (play_mass > mass - 1e-5)
        removed = jnp.sum((partner_belief > 0.0) & (~mask))
        focus = jnp.argmax(jnp.where(matches & partner_present, slot_index, -1))
        focus_playable = playable_type[partner_c[focus], partner_r[focus]]
        focus_critical = critical_type[partner_c[focus], partner_r[focus]]
        focus_confidence = play_mass[focus] / jnp.maximum(mass[focus], 1e-6)
        send_threshold = jnp.where(jnp.sum(f[200:203]) <= 1.0, 0.60, 0.10)
        return (1000.0 * jnp.sum(safe)
                + 100.0 * jnp.sum(safe & partner_actual_critical)
                + 150.0 * focus_playable * (focus_confidence >= send_threshold)
                + 100.0 * focus_playable * focus_critical
                * (focus_confidence >= send_threshold)
                - jnp.sum(matches) + 0.01 * removed)

    color_scores = jnp.stack([color_hint_score(c) for c in range(5)])
    rank_scores = jnp.stack([rank_hint_score(r) for r in range(5)])
    hint_scores = jnp.concatenate([color_scores, rank_scores])
    hint_legal = legal[10:20]
    best_hint_i = jnp.argmax(jnp.where(hint_legal, hint_scores, -1e9))
    best_hint = (10 + best_hint_i).astype(jnp.int32)
    best_hint_score = jnp.max(jnp.where(hint_legal, hint_scores, -1e9))
    useful_hint = best_hint_score >= 30.0
    last_token_hint_ok = ((best_hint_i >= 5) | (best_hint_score >= 70.0))
    color_legal = legal[10:15]
    safe_stall_hint = (10 + jnp.argmax(
        jnp.where(color_legal, color_scores, -1e9)
    )).astype(jnp.int32)

    # Interpret the immediately preceding partner clue. A conventional play is
    # accepted only when the public posterior itself assigns meaningful chance
    # to playability (and more conservatively on the last life). This rejects
    # premature five-save clues from unfamiliar partners.
    last = f[253:308]
    # Rank hints are the only conventional direct-play channel. Color clues
    # can safely carry general information or spend a token when discarding is
    # illegal, without prompting a speculative play.
    was_hint = last[5] > 0.5
    to_self = (last[1] > 0.5) & (last[6] > 0.5)
    touched = last[18:23] > 0.5
    focus_slot = jnp.argmax(jnp.where(touched, slot_index, -1))
    focus_prob = own_play_prob[focus_slot] / jnp.maximum(own_mass[focus_slot], 1e-6)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    convention_threshold = jnp.where(lives <= 1, 0.60, 0.10)
    convention_play = (was_hint & to_self & jnp.any(touched)
                       & (focus_prob >= convention_threshold)
                       & legal[5 + focus_slot])
    convention_action = (5 + focus_slot).astype(jnp.int32)
    own_play_posterior = own_play_prob / jnp.maximum(own_mass, 1e-6)
    belief_play_slot = jnp.argmax(
        jnp.where(legal[5:10], own_play_posterior, -1.0)
    )
    belief_play_threshold = jnp.where(lives >= 3, 0.85, 0.95)
    belief_play = ((lives > 1)
                   & (own_play_posterior[belief_play_slot] >= belief_play_threshold))
    belief_play_action = (5 + belief_play_slot).astype(jnp.int32)

    # Deterministic tie-breaking keeps behavior legible to an unfamiliar
    # partner. Prefer oldest/lowest-index cards when equally justified.
    safe_play_slot = jnp.argmax(certain_play)
    safe_play_action = (5 + safe_play_slot).astype(jnp.int32)
    obsolete_slot = jnp.argmax(certain_obsolete)
    obsolete_action = obsolete_slot.astype(jnp.int32)
    fallback_discard_slot = jnp.argmax(
        jnp.where(legal[0:5], own_discard_value, -1e9)
    )
    fallback_discard = fallback_discard_slot.astype(jnp.int32)

    any_play = jnp.any(certain_play & legal[5:10])
    any_obsolete = jnp.any(certain_obsolete & legal[0:5])
    can_discard = jnp.any(legal[0:5])
    can_hint = jnp.any(hint_legal)
    urgent_redirect = (at_risk_critical & jnp.any(partner_actual_playable)
                       & useful_hint & can_hint)
    # At full information, spend a clue before discarding. At lower token
    # counts, recycle a certainly obsolete card before giving a low-value clue.
    chosen_active = jnp.where(
        need_save,
        save_action,
        jnp.where(
            urgent_redirect,
            best_hint,
            jnp.where(
                any_play,
                safe_play_action,
                jnp.where(
                    convention_play,
                    convention_action,
                    jnp.where(
                        belief_play,
                        belief_play_action,
                        jnp.where(
                            useful_hint & can_hint
                            & ((info >= 2) | last_token_hint_ok),
                            best_hint,
                            jnp.where(
                                any_obsolete,
                                obsolete_action,
                                jnp.where(can_discard, fallback_discard, safe_stall_hint),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    action = jnp.where(legal[20], 20, chosen_active).astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    trace = {
        "fireworks": fireworks,
        "deck": deck,
        "info": info,
        "turn": turn,
        "remaining": remaining,
        "discard_value": own_discard_value,
        "need_save": need_save,
    }
    return action, turn + jnp.asarray(1, dtype=jnp.int32), trace
