"""Rule-based Hanabi policy using only public, deducible information.

Strategy: sharpens the environment-provided V0 belief with elimination
against the partner's visible hand and against our own already-identified
cards, then plays/discards when confident and otherwise hints a partner
card that is actually playable, critical (last copy), or unidentified.
All reasoning is grounded in observable public facts and directly visible
cards, not a private convention shared only between two policy instances.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
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

_EPS = 1e-3
_PLAY_CONF_THRESH = 0.82
_DEAD_CONF_THRESH = 0.9


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions

    partner_hand = features[0:125].reshape(5, 5, 5)          # slot, color, rank
    fireworks = features[167:192].reshape(5, 5)               # color, rank one-hot
    ranks = jnp.arange(5)
    height = jnp.sum(fireworks * (ranks + 1)[None, :], axis=1)  # (5,) in 0..5

    knowledge = features[308:658].reshape(2, 5, 35)
    self_v0 = knowledge[0, :, 0:25].reshape(5, 5, 5)   # slot, color, rank
    partner_v0 = knowledge[1, :, 0:25].reshape(5, 5, 5)

    playable_mask = (ranks[None, :] == height[:, None]).astype(jnp.float32)  # color, rank
    already_played_mask = ranks[None, :] < height[:, None]

    # Extend "dead" beyond already-played duplicates to ranks that can never
    # be reached because all copies of some required lower rank are gone
    # (fully public: derived from discard counts and firework height only).
    discards = features[203:253].reshape(5, 10)
    discard_count = jnp.stack(
        [
            jnp.sum(discards[:, 0:3], axis=1),
            jnp.sum(discards[:, 3:5], axis=1),
            jnp.sum(discards[:, 5:7], axis=1),
            jnp.sum(discards[:, 7:9], axis=1),
            jnp.sum(discards[:, 9:10], axis=1),
        ],
        axis=1,
    )  # (5 colors, 5 ranks)
    multiplicities = jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0])
    remaining_count = multiplicities[None, :] - already_played_mask.astype(jnp.float32) - discard_count
    blocked_cell = (remaining_count <= 0.5) & (~already_played_mask)
    blocked_exists = jnp.any(blocked_cell, axis=1)
    first_blocked_rank = jnp.where(blocked_exists, jnp.argmax(blocked_cell, axis=1), 5)
    dead_mask = (already_played_mask | (ranks[None, :] >= first_blocked_rank[:, None])).astype(jnp.float32)

    # Cross-hand elimination: V0 does not subtract cards visible in the
    # partner's hand from remaining public counts. If every remaining copy
    # of a card type is already visible in the partner's hand, that type
    # cannot be in our own hand either, so we can sharpen our own belief
    # using only publicly available and directly visible information.
    partner_visible_count = jnp.sum(partner_hand, axis=0)  # (color, rank)
    fully_accounted = remaining_count <= partner_visible_count + 0.5
    self_v0_zeroed = jnp.where(fully_accounted[None], 0.0, self_v0)
    corrected_total = jnp.sum(self_v0_zeroed, axis=(1, 2), keepdims=True)
    self_v0 = jnp.where(
        corrected_total > 1e-6, self_v0_zeroed / jnp.maximum(corrected_total, 1e-6), self_v0
    )

    # Second-order elimination: also subtract copies already confidently
    # identified in OTHER own slots (not just partner-visible ones) from
    # each slot's effective remaining count.
    self_occupied_pre = jnp.sum(self_v0, axis=(1, 2)) > 0.5
    self_confident_identity = self_occupied_pre[:, None, None] & (self_v0 > 0.95)
    self_confident_identity_count = jnp.sum(self_confident_identity.astype(jnp.float32), axis=0)
    other_confident_excl_self = (
        self_confident_identity_count[None, :, :] - self_confident_identity.astype(jnp.float32)
    )
    adjusted_remaining = (remaining_count - partner_visible_count)[None, :, :] - other_confident_excl_self
    fully_accounted2 = adjusted_remaining <= 0.5
    self_v0_zeroed2 = jnp.where(fully_accounted2, 0.0, self_v0)
    corrected_total2 = jnp.sum(self_v0_zeroed2, axis=(1, 2), keepdims=True)
    self_v0 = jnp.where(
        corrected_total2 > 1e-6, self_v0_zeroed2 / jnp.maximum(corrected_total2, 1e-6), self_v0
    )

    self_mass_total = jnp.sum(self_v0, axis=(1, 2))
    self_occupied = self_mass_total > 0.5
    self_mass_playable = jnp.sum(self_v0 * playable_mask[None], axis=(1, 2))
    self_mass_dead = jnp.sum(self_v0 * dead_mask[None], axis=(1, 2))
    self_certain_play = self_occupied & (self_mass_playable > _PLAY_CONF_THRESH * self_mass_total)
    self_certain_dead = self_occupied & (self_mass_dead > _DEAD_CONF_THRESH * self_mass_total)

    partner_mass_total = jnp.sum(partner_v0, axis=(1, 2))
    partner_occupied = partner_mass_total > 0.5
    partner_mass_playable = jnp.sum(partner_v0 * playable_mask[None], axis=(1, 2))
    partner_certain_play = partner_occupied & (partner_mass_playable > partner_mass_total - _EPS)

    partner_hand_occupied = jnp.sum(partner_hand, axis=(1, 2)) > 0.5
    partner_color = jnp.argmax(jnp.sum(partner_hand, axis=2), axis=1)  # (5,)
    partner_rank = jnp.argmax(jnp.sum(partner_hand, axis=1), axis=1)   # (5,)
    partner_actual_playable = partner_hand_occupied & (partner_rank == height[partner_color])

    partner_color_flags = knowledge[1, :, 25:30]
    partner_rank_flags = knowledge[1, :, 30:35]
    partner_has_color_info = jnp.any(partner_color_flags > 0.5, axis=1)
    partner_has_rank_info = jnp.any(partner_rank_flags > 0.5, axis=1)

    # Cards that are the last surviving copy of a still-needed rank deserve
    # a protective hint before they might be carelessly discarded: this is
    # a broadly useful heuristic regardless of the partner's convention,
    # since most reasonable partners avoid discarding already-hinted cards.
    card_remaining = remaining_count[partner_color, partner_rank]
    still_needed = partner_rank >= height[partner_color]
    fully_known = partner_has_color_info & partner_has_rank_info
    critical_mask = partner_hand_occupied & (card_remaining <= 1.5) & still_needed & (~fully_known)
    has_critical = jnp.any(critical_mask)
    critical_slot = jnp.argmax(critical_mask)
    critical_color = partner_color[critical_slot]
    critical_rank = partner_rank[critical_slot]
    critical_hint_action = jnp.where(
        ~partner_has_color_info[critical_slot], 10 + critical_color, 15 + critical_rank
    ).astype(jnp.int32)

    hint_target = partner_actual_playable & (~partner_certain_play)
    has_hint_target = jnp.any(hint_target)
    target_slot = jnp.argmax(hint_target)
    target_color = partner_color[target_slot]
    target_rank = partner_rank[target_slot]

    slot_v0 = partner_v0[target_slot]  # (color, rank)
    mass_color_total = jnp.sum(slot_v0[target_color, :])
    mass_color_playable = jnp.sum(slot_v0[target_color, :] * playable_mask[target_color, :])
    mass_rank_total = jnp.sum(slot_v0[:, target_rank])
    mass_rank_playable = jnp.sum(slot_v0[:, target_rank] * playable_mask[:, target_rank])
    color_ratio = mass_color_playable / jnp.maximum(mass_color_total, _EPS)
    rank_ratio = mass_rank_playable / jnp.maximum(mass_rank_total, _EPS)
    use_color_hint = color_ratio >= rank_ratio
    hint_action = jnp.where(use_color_hint, 10 + target_color, 15 + target_rank).astype(jnp.int32)

    self_play_idx = jnp.argmax(self_certain_play)
    cert_play_action = (5 + self_play_idx).astype(jnp.int32)

    self_dead_idx = jnp.argmax(self_certain_dead)
    cert_dead_action = self_dead_idx.astype(jnp.int32)

    # Prefer discarding an occupied self slot that has never received a
    # hint: this policy only hints cards it deems important (playable
    # targets or fresh information), so an already-hinted card is more
    # likely to matter than an untouched one.
    self_color_flags = knowledge[0, :, 25:30]
    self_rank_flags = knowledge[0, :, 30:35]
    self_has_color_info = jnp.any(self_color_flags > 0.5, axis=1)
    self_has_rank_info = jnp.any(self_rank_flags > 0.5, axis=1)
    self_unhinted = self_occupied & (~self_has_color_info) & (~self_has_rank_info)
    has_self_unhinted = jnp.any(self_unhinted)
    self_unhinted_slot = jnp.argmax(self_unhinted)
    discard_oldest_action = jnp.where(
        has_self_unhinted, self_unhinted_slot, jnp.asarray(0, dtype=jnp.int32)
    ).astype(jnp.int32)

    self_best_play_idx = jnp.argmax(self_mass_playable)
    risky_play_action = (5 + self_best_play_idx).astype(jnp.int32)

    # Prefer completing a partially-identified partner card (only color or
    # only rank known) over starting a brand-new unknown one, so that cards
    # reach full identification quickly instead of many cards staying
    # half-known forever. Only start a fresh unknown slot if none are
    # partially known.
    partial_missing_rank = partner_hand_occupied & partner_has_color_info & (~partner_has_rank_info)
    partial_missing_color = partner_hand_occupied & (~partner_has_color_info) & partner_has_rank_info
    has_partial = jnp.any(partial_missing_rank) | jnp.any(partial_missing_color)
    missing_rank_slot = jnp.argmax(partial_missing_rank)
    missing_color_slot = jnp.argmax(partial_missing_color)
    use_missing_rank = jnp.any(partial_missing_rank)
    partial_slot = jnp.where(use_missing_rank, missing_rank_slot, missing_color_slot)
    partial_hint_action = jnp.where(
        use_missing_rank, 15 + partner_rank[partial_slot], 10 + partner_color[partial_slot]
    ).astype(jnp.int32)

    uninformed = partner_hand_occupied & (~partner_has_color_info) & (~partner_has_rank_info)
    uninformed_slot = jnp.argmax(uninformed)
    new_hint_action = (10 + partner_color[uninformed_slot]).astype(jnp.int32)

    fallback_hint_action = jnp.where(has_partial, partial_hint_action, new_hint_action).astype(jnp.int32)

    final_action = jnp.argmax(legal).astype(jnp.int32)

    def is_legal(a):
        return legal[a]

    action = final_action
    branch = jnp.asarray(0, dtype=jnp.int32)
    take = is_legal(risky_play_action)
    action = jnp.where(take, risky_play_action, action)
    branch = jnp.where(take, 1, branch)
    take = is_legal(fallback_hint_action)
    action = jnp.where(take, fallback_hint_action, action)
    branch = jnp.where(take, 3, branch)
    take = has_critical & is_legal(critical_hint_action)
    action = jnp.where(take, critical_hint_action, action)
    branch = jnp.where(take, 5, branch)
    take = is_legal(discard_oldest_action)
    action = jnp.where(take, discard_oldest_action, action)
    branch = jnp.where(take, 2, branch)
    take = jnp.any(self_certain_dead) & is_legal(cert_dead_action)
    action = jnp.where(take, cert_dead_action, action)
    branch = jnp.where(take, 4, branch)
    take = has_hint_target & is_legal(hint_action)
    action = jnp.where(take, hint_action, action)
    branch = jnp.where(take, 6, branch)
    take = jnp.any(self_certain_play) & is_legal(cert_play_action)
    action = jnp.where(take, cert_play_action, action)
    branch = jnp.where(take, 7, branch)

    action = action.astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)
    lives = jnp.sum(features[200:203]).astype(jnp.float32)
    tokens = jnp.sum(features[192:200]).astype(jnp.float32)
    trace = {"branch": branch, "lives": lives, "tokens": tokens}
    return action, next_memory, trace
