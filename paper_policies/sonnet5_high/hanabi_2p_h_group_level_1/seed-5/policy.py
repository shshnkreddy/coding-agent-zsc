"""H-group Level 1 rule-based Hanabi policy.

Both seats run this same code with independent local observations, private
memory and PRNG. Own-hand identity is never observed; only the native V0
possibility weights and positive hint flags for own slots are available.
Partner-hand identities are fully observed, which lets the clue-giving side
of the logic reason exactly (no belief modelling needed) while the
clue-receiving side must use V0 confidence plus a small amount of private
memory to reconstruct clue *focus* (H-group Level 1) for the own hand.

Native slot order: index 0 is oldest / H-group "rightmost", the highest
occupied index is newest / H-group "leftmost" (see runtime_mapping.md).
Occupied slots always form the contiguous prefix ``0 .. k-1``.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hgroup_level1_v1',
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

NUM_COLORS = 5
NUM_RANKS = 5
HAND = 5
MULT_LIST = (3, 2, 2, 2, 1)  # copies per rank id 0..4

# Discard-count thermometer subblock boundaries per color (sizes match MULT_LIST).
_RANK_STARTS = (0, 3, 5, 7, 9)
_RANK_ENDS = (3, 5, 7, 9, 10)


def init_agent():
    return {
        'play_focus': jnp.zeros((HAND,), dtype=jnp.bool_),
        'prev_own_clued': jnp.zeros((HAND,), dtype=jnp.bool_),
    }


def _shift_remove(arr, idx):
    """Remove native index idx: entries after it shift left; slot 4 -> False."""
    slot_idx = jnp.arange(HAND, dtype=jnp.int32)
    shifted_tail = jnp.concatenate([arr[1:], jnp.zeros((1,), dtype=arr.dtype)])
    return jnp.where(slot_idx < idx, arr, shifted_tail)


def _focus_index(matched, clued_before, occupied):
    """H-group Level 1 clue-focus rule. matched/clued_before/occupied: bool[5]."""
    slot_idx = jnp.arange(HAND, dtype=jnp.int32)
    neg = jnp.asarray(-1, dtype=jnp.int32)

    occ_matched = matched & occupied
    newly = occ_matched & (~clued_before)
    any_new = jnp.any(newly)

    chop_candidates = jnp.where(occupied & (~clued_before), slot_idx, HAND)
    chop_idx = jnp.min(chop_candidates)
    chop_valid = chop_idx < HAND
    chop_safe = jnp.clip(chop_idx, 0, HAND - 1)
    chop_touched = chop_valid & matched[chop_safe]

    leftmost_new = jnp.max(jnp.where(newly, slot_idx, neg))
    leftmost_retouched = jnp.max(jnp.where(occ_matched, slot_idx, neg))

    focus = jnp.where(
        ~any_new,
        leftmost_retouched,
        jnp.where(chop_touched, chop_idx, leftmost_new),
    )
    return focus.astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions

    slot_idx = jnp.arange(HAND, dtype=jnp.int32)
    mult = jnp.asarray(MULT_LIST, dtype=jnp.float32)

    is_active = jnp.logical_not(legal[20])

    # ---- decode observation ----
    partner_hand = features[0:125].reshape(HAND, NUM_COLORS, NUM_RANKS)
    occupied_partner = partner_hand.sum(axis=(1, 2)) > 0.5

    fireworks = features[167:192].reshape(NUM_COLORS, NUM_RANKS)
    height = (fireworks * jnp.arange(1, NUM_RANKS + 1, dtype=jnp.float32)).sum(axis=1)

    info_tokens = features[192:200].sum()

    discards = features[203:253].reshape(NUM_COLORS, 10)
    discard_count = jnp.stack(
        [discards[:, s:e].sum(axis=1) for s, e in zip(_RANK_STARTS, _RANK_ENDS)],
        axis=1,
    )  # [color, rank]

    last = features[253:308]
    acted_self = last[0] > 0.5
    act_type = last[2:6]  # play, discard, color-hint, rank-hint
    is_play_act = act_type[0] > 0.5
    is_discard_act = act_type[1] > 0.5
    is_hint_act = (act_type[2] > 0.5) | (act_type[3] > 0.5)
    recipient_self = last[6] > 0.5
    match_flags = (last[18:23] > 0.5)
    removed_onehot = last[23:28]
    removed_idx = jnp.argmax(removed_onehot)

    native = features[308:658].reshape(2, HAND, 35)
    self_native = native[0]
    partner_native = native[1]
    v0_self = self_native[:, 0:25].reshape(HAND, NUM_COLORS, NUM_RANKS)
    color_hint_self = self_native[:, 25:30]
    rank_hint_self = self_native[:, 30:35]
    occupied_self = v0_self.sum(axis=(1, 2)) > 0.5
    own_clued = (color_hint_self.sum(axis=1) > 0.5) | (rank_hint_self.sum(axis=1) > 0.5)

    color_hint_partner = partner_native[:, 25:30]
    rank_hint_partner = partner_native[:, 30:35]
    partner_clued = (color_hint_partner.sum(axis=1) > 0.5) | (rank_hint_partner.sum(axis=1) > 0.5)

    # ---- memory bookkeeping (runs on every transition, both seats) ----
    play_focus = memory['play_focus']
    prev_own_clued = memory['prev_own_clued']

    self_played_or_discarded = acted_self & (is_play_act | is_discard_act)
    play_focus = jnp.where(self_played_or_discarded, _shift_remove(play_focus, removed_idx), play_focus)
    prev_own_clued = jnp.where(
        self_played_or_discarded, _shift_remove(prev_own_clued, removed_idx), prev_own_clued
    )

    # A clue whose focus lands on the (pre-clue) chop is ambiguous between a
    # Save Clue and a Play Clue (H-group Level 1); only a non-chop focus is an
    # unambiguous Play Clue that the receiver should trust and play on faith,
    # even when V0 alone cannot pin the color. Chop-focus (and merely
    # touched, non-focus) cards fall back to a strict V0-elimination check.
    hint_to_self = is_hint_act & recipient_self
    focus_of_received_hint = _focus_index(match_flags, prev_own_clued, occupied_self)
    chop_before_candidates = jnp.where(occupied_self & (~prev_own_clued), slot_idx, HAND)
    chop_before_idx = jnp.min(chop_before_candidates)
    focus_is_chop = focus_of_received_hint == chop_before_idx
    focus_onehot = (slot_idx == focus_of_received_hint)
    play_focus = jnp.where(
        hint_to_self & (~focus_is_chop), play_focus | focus_onehot, play_focus
    )

    new_prev_own_clued = own_clued

    default_memory = init_agent()
    next_memory = {
        'play_focus': jnp.where(reset, default_memory['play_focus'], play_focus),
        'prev_own_clued': jnp.where(reset, default_memory['prev_own_clued'], new_prev_own_clued),
    }

    # ---- sharpen V0 using directly-observed partner-hand copies ----
    # The native V0 already applies the literal hint mask, but its "remaining
    # public count" factor does not subtract cards visible in the partner's
    # hand (per the spec). We can see the partner's hand exactly, so recover
    # the 0/1 literal mask from V0 and re-weight it with a tighter count that
    # excludes those visible copies -- a strictly more accurate own-hand
    # posterior, without ever looking at our own hidden cards.
    already_played_cr = (
        jnp.arange(NUM_RANKS, dtype=jnp.float32)[None, :] < height[:, None]
    ).astype(jnp.float32)  # [color, rank]
    pub_count = mult[None, :] - discard_count - already_played_cr  # [color, rank]
    partner_visible_count = jnp.sum(
        jnp.where(occupied_partner[:, None, None], partner_hand, 0.0), axis=0
    )  # [color, rank]
    pub_count_sharp = jnp.maximum(pub_count - partner_visible_count, 0.0)  # [color, rank]
    mask_self = (v0_self > 1e-6).astype(jnp.float32)  # [slot, color, rank]
    # If several of our own slots could all still be the same scarce
    # identity, naively giving each of them the full remaining count
    # double-(or triple-)counts that identity's probability mass across
    # slots. Split each (color, rank) count across the own slots that still
    # find it possible, since at most that many of our own cards can
    # actually be it.
    own_claimants = jnp.maximum(mask_self.sum(axis=0), 1.0)  # [color, rank]
    pub_count_split = pub_count_sharp[None, :, :] / own_claimants[None, :, :]
    sharp_unnorm = mask_self * pub_count_split
    sharp_sum = sharp_unnorm.sum(axis=(1, 2), keepdims=True)
    v0_self_sharp = jnp.where(sharp_sum > 1e-6, sharp_unnorm / jnp.maximum(sharp_sum, 1e-6), v0_self)

    # ---- own-hand play evaluation ----
    is_playable_grid = (
        jnp.arange(NUM_RANKS, dtype=jnp.float32)[None, :] == height[:, None]
    ).astype(jnp.float32)  # [color, rank]
    playable_mass = jnp.einsum('scr,cr->s', v0_self_sharp, is_playable_grid)

    # "playable_mass" is our own literal-information estimate of the
    # probability that playing this slot right now would succeed; it is not
    # a certainty even for a definite (non-chop) play-clue focus, since the
    # giver's promise only pins down what the *receiver* cannot directly
    # verify (the color, when only a rank hint was given). Gambling on a
    # play with mass well below 1 risks a life for a coin flip, and a bomb
    # on the final life zeroes the entire score. So even for a trusted
    # focus we still require a fairly concentrated posterior before acting,
    # just a lower bar than an unfocused, merely-touched or chop-focus card
    # (which needs near-certain elimination). On the final life demand
    # much higher confidence for both.
    lives_remaining = features[200:203].sum()
    on_last_life = lives_remaining <= 1.5
    on_middle_life = (lives_remaining > 1.5) & (lives_remaining <= 2.5)
    play_threshold = jnp.where(
        on_last_life,
        jnp.where(play_focus, 0.6, 0.995),
        jnp.where(
            on_middle_life,
            jnp.where(play_focus, 0.2, 0.8),
            jnp.where(play_focus, 0.1, 0.8),
        ),
    )
    play_eligible = occupied_self & own_clued & (playable_mass >= play_threshold)
    play_score = jnp.where(play_eligible, playable_mass + jnp.where(play_focus, 10.0, 0.0), -1.0)
    best_play_slot = jnp.argmax(play_score)
    valid_a = jnp.any(play_eligible)
    action_a = (5 + best_play_slot).astype(jnp.int32)

    # ---- partner-hand exact reasoning (clue giving) ----
    color_idx_partner = jnp.argmax(partner_hand.sum(axis=2), axis=1)  # [slot]
    rank_idx_partner = jnp.argmax(partner_hand.sum(axis=1), axis=1)  # [slot]

    height_of_color = height[color_idx_partner]
    is_playable_partner = (rank_idx_partner.astype(jnp.float32) == height_of_color)
    needed_partner = rank_idx_partner.astype(jnp.float32) >= height_of_color

    already_played = jnp.where(height_of_color > rank_idx_partner.astype(jnp.float32), 1.0, 0.0)
    remaining_count_partner = (
        mult[rank_idx_partner] - discard_count[color_idx_partner, rank_idx_partner] - already_played
    )

    chop_partner_candidates = jnp.where(occupied_partner & (~partner_clued), slot_idx, HAND)
    chop_partner = jnp.min(chop_partner_candidates)
    chop_partner_valid = chop_partner < HAND
    chop_safe_p = jnp.clip(chop_partner, 0, HAND - 1)

    chop_color = color_idx_partner[chop_safe_p]
    chop_rank = rank_idx_partner[chop_safe_p]
    chop_needed = needed_partner[chop_safe_p]
    chop_remaining = remaining_count_partner[chop_safe_p]

    same_identity = (
        (color_idx_partner == chop_color) & (rank_idx_partner == chop_rank) & occupied_partner
    )
    other_copy_in_hand = jnp.sum(same_identity.astype(jnp.int32)) - 1  # exclude chop itself
    other_copy_in_hand = jnp.maximum(other_copy_in_hand, 0)

    # A unique *playable* card is defined by H-group as a Play Clue, not a
    # Save Clue (priority C below already covers it); reserve the Save Clue
    # path for cards that are critical but not yet playable now.
    chop_is_playable = is_playable_partner[chop_safe_p]
    is_critical = chop_needed & (chop_remaining <= 1.0) & (~chop_is_playable)
    is_2save = (
        (chop_rank == 1)
        & chop_needed
        & (chop_remaining >= 1.9)
        & (chop_remaining <= 2.1)
        & (other_copy_in_hand == 0)
        & (~chop_is_playable)
    )
    should_save = chop_partner_valid & (is_critical | is_2save)
    use_rank_save = (chop_rank == 4) | is_2save
    save_color_action = (10 + chop_color).astype(jnp.int32)
    save_rank_action = (15 + chop_rank).astype(jnp.int32)
    action_b = jnp.where(use_rank_save, save_rank_action, save_color_action)
    valid_b = should_save & (info_tokens > 0.5)

    # ---- play-clue target search over partner's unclued playable cards ----
    # Consider every candidate playable target (not just the first occupied
    # one) and prefer whichever can be cleanly isolated (a hint that
    # correctly resolves H-group focus onto it) with the least collateral
    # touching, since that gives the receiver the sharpest posterior.
    play_targets = occupied_partner & (~partner_clued) & is_playable_partner

    per_j_valid = []
    per_j_action = []
    per_j_collateral = []
    for j in range(HAND):
        j_color = color_idx_partner[j]
        j_rank = rank_idx_partner[j]
        matched_by_color_j = occupied_partner & (color_idx_partner == j_color)
        matched_by_rank_j = occupied_partner & (rank_idx_partner == j_rank)
        focus_color_j = _focus_index(matched_by_color_j, partner_clued, occupied_partner)
        focus_rank_j = _focus_index(matched_by_rank_j, partner_clued, occupied_partner)
        color_ok_j = focus_color_j == j
        rank_ok_j = focus_rank_j == j
        new_touch_color_j = jnp.sum((matched_by_color_j & (~partner_clued)).astype(jnp.int32))
        new_touch_rank_j = jnp.sum((matched_by_rank_j & (~partner_clued)).astype(jnp.int32))
        prefer_color_j = color_ok_j & (~rank_ok_j | (new_touch_color_j <= new_touch_rank_j))
        use_color_j = color_ok_j & (prefer_color_j | ~rank_ok_j)
        action_j = jnp.where(
            use_color_j, (10 + j_color).astype(jnp.int32), (15 + j_rank).astype(jnp.int32)
        )
        collateral_j = jnp.where(use_color_j, new_touch_color_j, new_touch_rank_j)
        valid_j = play_targets[j] & (color_ok_j | rank_ok_j)
        per_j_valid.append(valid_j)
        per_j_action.append(action_j)
        per_j_collateral.append(collateral_j)

    per_j_valid = jnp.stack(per_j_valid)
    per_j_action = jnp.stack(per_j_action)
    per_j_collateral = jnp.stack(per_j_collateral)

    has_target = jnp.any(per_j_valid)
    selection_score = jnp.where(per_j_valid, -per_j_collateral.astype(jnp.float32), -1e9)
    best_j = jnp.argmax(selection_score)
    action_c = per_j_action[best_j]
    valid_c = has_target & (info_tokens > 0.5)

    # ---- discard fallback ----
    # Default convention target is chop (lowest unclued index). But we can
    # also use our own elimination-based belief (never our hidden identity
    # directly, only public counts) to avoid chopping a card that is almost
    # certainly the last remaining copy of a needed identity -- the guide
    # itself allows discarding a known-useless clued card ahead of chop, so
    # preferring a safer unclued slot when our own belief strongly warns
    # against chop is in the same spirit, not a private signal.
    already_played_self_grid = (
        jnp.arange(NUM_RANKS, dtype=jnp.float32)[None, :] < height[:, None]
    ).astype(jnp.float32)
    trash_mass = jnp.einsum('scr,cr->s', v0_self_sharp, already_played_self_grid)

    needed_grid = 1.0 - already_played_self_grid  # [color, rank], 1 if rank>=height
    critical_grid = (pub_count <= 1.0).astype(jnp.float32) * needed_grid  # [color, rank]
    critical_risk = jnp.einsum('scr,cr->s', v0_self_sharp, critical_grid)

    unclued_self = occupied_self & (~own_clued)
    safe_unclued_score = jnp.where(unclued_self, -critical_risk, -1e9)
    safest_unclued = jnp.argmax(safe_unclued_score)

    chop_self_candidates = jnp.where(unclued_self, slot_idx, HAND)
    chop_self = jnp.min(chop_self_candidates)
    chop_self_valid = chop_self < HAND
    chop_risk = critical_risk[jnp.clip(chop_self, 0, HAND - 1)]
    safest_risk = critical_risk[safest_unclued]
    use_safer_unclued = chop_risk >= 0.05

    trash_score = jnp.where(occupied_self, trash_mass, -1.0)
    fallback_discard = jnp.argmax(trash_score)
    unclued_discard = jnp.where(use_safer_unclued, safest_unclued, jnp.clip(chop_self, 0, HAND - 1))
    discard_target = jnp.where(chop_self_valid, unclued_discard, fallback_discard)
    action_d = discard_target.astype(jnp.int32)
    valid_d = (info_tokens < 7.5) & jnp.any(occupied_self)

    # ---- generic stall hint fallback (tokens capped at 8, cannot discard) ----
    first_partner_candidates = jnp.where(occupied_partner, slot_idx, HAND)
    first_partner = jnp.clip(jnp.min(first_partner_candidates), 0, HAND - 1)
    action_e = (10 + color_idx_partner[first_partner]).astype(jnp.int32)
    valid_e = (info_tokens > 0.5) & jnp.any(occupied_partner)

    fallback_global = jnp.argmax(legal.astype(jnp.int32)).astype(jnp.int32)

    def apply(cur, valid, act):
        act = act.astype(jnp.int32)
        ok = valid & legal[jnp.clip(act, 0, 20)]
        return jnp.where(ok, act, cur)

    action = fallback_global
    action = apply(action, valid_e, action_e)
    action = apply(action, valid_d, action_d)
    action = apply(action, valid_c, action_c)
    action = apply(action, valid_b, action_b)
    action = apply(action, valid_a, action_a)

    final_action = jnp.where(is_active, action, jnp.asarray(20, dtype=jnp.int32))

    priority_used = jnp.asarray(0, dtype=jnp.int32)
    priority_used = jnp.where(valid_e & legal[jnp.clip(action_e.astype(jnp.int32), 0, 20)], 1, priority_used)
    priority_used = jnp.where(valid_d & legal[jnp.clip(action_d.astype(jnp.int32), 0, 20)], 2, priority_used)
    priority_used = jnp.where(valid_c & legal[jnp.clip(action_c.astype(jnp.int32), 0, 20)], 3, priority_used)
    priority_used = jnp.where(valid_b & legal[jnp.clip(action_b.astype(jnp.int32), 0, 20)], 4, priority_used)
    priority_used = jnp.where(valid_a & legal[jnp.clip(action_a.astype(jnp.int32), 0, 20)], 5, priority_used)
    trace = {
        'lives': features[200:203].sum().astype(jnp.float32),
        'tokens': info_tokens.astype(jnp.float32),
        'height_sum': height.sum().astype(jnp.float32),
        'priority_used': jnp.where(is_active, priority_used, jnp.asarray(-1, dtype=jnp.int32)),
        'action': final_action,
        'played_via_focus': play_focus[best_play_slot].astype(jnp.float32),
        'played_mass': playable_mass[best_play_slot],
    }

    return final_action, next_memory, trace
