"""H-group Level 1 rule-based Hanabi policy.

Most reasoning is recomputed fresh each turn from the observation's public
fields (partner hand, discard/firework counts, native possibility masks and
hint flags). One thing is NOT recoverable from a single observation: the
H-group "focus" of the clue we most recently received, and the convention
that a *color* play-clue promises the specific rank that is currently
playable for that color (trusted, not merely narrowed by elimination). We
track that promise in a small private memory, keyed by our own native slot
index and kept aligned across slot shifts using the public last-action
fields (which reveal exactly which of our slots was removed on our own
play/discard).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'h_group_level1_rule_based',
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

BIG = 999


def init_agent():
    return {
        # own clued status as of the end of the previous call (pre-transition
        # snapshot used to determine which of our slots are "new" when we
        # are the target of a hint).
        'prev_clued': jnp.zeros((5,), dtype=jnp.bool_),
        # per own-slot trusted play-clue promise: a color clue whose focus is
        # this slot promises it is (delayed-)playable at exactly the color's
        # then-current firework height.
        'frozen_valid': jnp.zeros((5,), dtype=jnp.bool_),
        'frozen_target_rank': jnp.zeros((5,), dtype=jnp.int32),
    }


def _discard_counts(discard_block):
    # discard_block: [5 colors, 10] thermometer subblocks per rank:
    # [0:3],[3:5],[5:7],[7:9],[9:10]
    starts = [0, 3, 5, 7, 9]
    lens = [3, 2, 2, 2, 1]
    per_rank = []
    for start, length in zip(starts, lens):
        per_rank.append(discard_block[:, start:start + length].sum(axis=-1))
    return jnp.stack(per_rank, axis=1)  # [5 colors, 5 ranks]


def _shift_after_removal(arr, removed_slot, default_last):
    idx5 = jnp.arange(5)
    src_idx = jnp.clip(jnp.where(idx5 < removed_slot, idx5, idx5 + 1), 0, 4)
    shifted = arr[src_idx]
    shifted = shifted.at[4].set(default_last)
    return shifted


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    mult = jnp.array([3.0, 2.0, 2.0, 2.0, 1.0])  # copies per rank id 0..4
    idx5 = jnp.arange(5)

    partner_hand = features[0:125].reshape(5, 5, 5)  # [slot, color, rank]
    fireworks = features[167:192].reshape(5, 5)  # [color, rank] one-hot
    info_tokens = features[192:200]
    discards = features[203:253].reshape(5, 10)
    last_action = features[253:308]
    native = features[308:658].reshape(2, 5, 35)  # [rel player, slot, 35]

    self_v0 = native[0, :, 0:25].reshape(5, 5, 5)  # [slot, color, rank]
    self_color_flags = native[0, :, 25:30]
    self_rank_flags = native[0, :, 30:35]

    partner_color_flags = native[1, :, 25:30]
    partner_rank_flags = native[1, :, 30:35]

    # ---- public board state ----
    firework_has = fireworks.sum(axis=-1) > 0.5
    firework_height_i = jnp.where(firework_has, jnp.argmax(fireworks, axis=-1) + 1, 0)
    firework_height = firework_height_i.astype(jnp.float32)

    discard_count = _discard_counts(discards)  # [color, rank]
    played_mask = idx5[None, :] < firework_height[:, None]  # [color, rank]
    remaining_count = mult[None, :] - discard_count - played_mask.astype(jnp.float32)

    partner_count = partner_hand.sum(axis=0)  # [color, rank]
    hidden_count = jnp.maximum(remaining_count - partner_count, 0.0)

    playable_grid = (idx5[None, :] == firework_height_i[:, None])  # [color, rank] bool
    trash_grid = played_mask  # already-played identities

    # ---- own hand: raw public elimination + Good Touch trash exclusion ----
    self_occupied = self_v0.sum(axis=(1, 2)) > 1e-6
    has_color_flag_self = jnp.any(self_color_flags > 0.5, axis=-1)
    has_rank_flag_self = jnp.any(self_rank_flags > 0.5, axis=-1)
    self_clued = has_color_flag_self | has_rank_flag_self
    color_of_self_slot = jnp.argmax(self_color_flags, axis=-1)  # meaningful iff has_color_flag_self

    # NOTE: we deliberately do NOT blanket-exclude "already played" (trash)
    # identities here using the *current* firework height. That promise
    # ("this clued card isn't trash") only holds at clue time; as the game
    # advances and other copies get played, re-applying it with the current
    # height would wrongly delete the true (now-stale) identity from the
    # candidate set. The validated per-slot frozen-promise mechanism below
    # captures the clue-time promise correctly instead.
    raw_combined_self = (self_v0 > 1e-9) & (hidden_count[None, :, :] > 0.5)  # [slot, color, rank]
    combined_self = raw_combined_self
    combined_sum = combined_self.sum(axis=(1, 2))
    has_possibility = combined_sum > 0.5
    determined_self = jnp.isclose(combined_sum, 1.0)

    violation_self = jnp.any(combined_self & (~playable_grid[None, :, :]), axis=(1, 2))
    elimination_safe_self = self_occupied & has_possibility & (~violation_self)

    # Safe to discard whenever every remaining consistent identity is
    # already played (trash), not only when the identity is fully singled
    # out: no possible true identity would ever be needed on the stacks.
    trash_self = self_occupied & has_possibility & jnp.all(
        jnp.where(combined_self, trash_grid[None, :, :], True), axis=(1, 2))
    is_five_self = self_occupied & determined_self & jnp.any(
        combined_self & (idx5[None, None, :] == 4), axis=(1, 2))

    # ---- last-action decoding (offsets local to the 253:308 block) ----
    act_player_onehot = last_action[0:2]  # [self, partner]
    action_type_onehot = last_action[2:6]  # [play, discard, color hint, rank hint]
    hint_target_onehot = last_action[6:8]  # [self, partner]
    color_hint_onehot = last_action[8:13]
    rank_hint_onehot = last_action[13:18]
    hint_match = last_action[18:23] > 0.5
    removed_onehot = last_action[23:28]

    actor_is_partner = act_player_onehot[1] > 0.5
    actor_is_self = act_player_onehot[0] > 0.5
    is_color_hint_action = action_type_onehot[2] > 0.5
    is_rank_hint_action = action_type_onehot[3] > 0.5
    is_play_or_discard_action = (action_type_onehot[0] > 0.5) | (action_type_onehot[1] > 0.5)
    hint_target_is_self = hint_target_onehot[0] > 0.5

    hint_to_me = actor_is_partner & (is_color_hint_action | is_rank_hint_action) & hint_target_is_self
    removed_by_me = actor_is_self & is_play_or_discard_action
    clue_value_color = jnp.argmax(color_hint_onehot)

    # ---- interpret a clue just received: determine its focus, and whether
    # it establishes a trusted color play-clue promise (vs a chop save) ----
    prev_clued = memory['prev_clued']
    chop_before_mask = self_occupied & (~prev_clued)
    chop_before_exists = jnp.any(chop_before_mask)
    chop_before_idx = jnp.argmin(jnp.where(chop_before_mask, idx5, BIG))

    new_mask = hint_match & (~prev_clued)
    any_new = jnp.any(new_mask)
    retouch_focus = jnp.argmax(jnp.where(hint_match, idx5, -1))
    newcard_focus = jnp.argmax(jnp.where(new_mask, idx5, -1))
    chop_touched = chop_before_exists & jnp.take(hint_match, chop_before_idx)
    focus_idx = jnp.where(~any_new, retouch_focus, jnp.where(chop_touched, chop_before_idx, newcard_focus))
    is_chop_focus = chop_touched & (focus_idx == chop_before_idx)

    focus_has_possibility = jnp.any(combined_self[focus_idx])
    focus_all_playable = focus_has_possibility & jnp.all(
        jnp.where(combined_self[focus_idx], playable_grid, True))
    # Non-chop focus is unambiguously a play clue. Chop-focus is only a play
    # clue (rather than a save) when the resulting elimination already shows
    # every consistent identity as currently playable.
    treat_as_play = (~is_chop_focus) | focus_all_playable
    set_frozen = hint_to_me & treat_as_play & is_color_hint_action

    frozen_valid_upd = memory['frozen_valid'].at[focus_idx].set(
        jnp.where(set_frozen, True, memory['frozen_valid'][focus_idx]))
    frozen_target_upd = memory['frozen_target_rank'].at[focus_idx].set(
        jnp.where(set_frozen, firework_height_i[clue_value_color], memory['frozen_target_rank'][focus_idx]))

    # ---- re-align our own belief slots if we just played/discarded ----
    removed_slot = jnp.argmax(removed_onehot)
    frozen_valid_shifted = _shift_after_removal(frozen_valid_upd, removed_slot, jnp.asarray(False))
    frozen_target_shifted = _shift_after_removal(frozen_target_upd, removed_slot, jnp.asarray(0, jnp.int32))
    frozen_valid_final = jnp.where(removed_by_me, frozen_valid_shifted, frozen_valid_upd)
    frozen_target_final = jnp.where(removed_by_me, frozen_target_shifted, frozen_target_upd)

    # A frozen promise self-invalidates once it is no longer literally
    # consistent with public elimination (e.g. later hints ruled it out).
    frozen_literal_ok = raw_combined_self[idx5, color_of_self_slot, jnp.clip(frozen_target_final, 0, 4)]
    frozen_valid_final = frozen_valid_final & self_occupied & frozen_literal_ok

    frozen_play_self = frozen_valid_final & (firework_height_i[color_of_self_slot] == frozen_target_final)
    is_safe_play_self = elimination_safe_self | frozen_play_self

    # own chop = lowest native index among unclued occupied slots
    self_unclued = self_occupied & (~self_clued)
    self_chop_exists = jnp.any(self_unclued)
    self_chop_idx = jnp.argmin(jnp.where(self_unclued, idx5, BIG))

    # ---- own play decision ----
    priority_score = jnp.where(is_safe_play_self, jnp.where(is_five_self, 2, 1), 0)
    any_safe_play = jnp.any(is_safe_play_self)
    play_slot = jnp.argmax(priority_score)
    play_action = 5 + play_slot

    # ---- own discard decision ----
    any_trash = jnp.any(trash_self)
    trash_idx = jnp.argmin(jnp.where(trash_self, idx5, BIG))
    # If the whole hand is clued (no chop), fall back to the oldest
    # occupied slot (lowest native index), matching chop semantics rather
    # than discarding the newest (and often still-ambiguous) card.
    fallback_discard_idx = jnp.argmin(jnp.where(self_occupied, idx5, BIG))
    discard_slot = jnp.where(
        any_trash, trash_idx,
        jnp.where(self_chop_exists, self_chop_idx, fallback_discard_idx))
    discard_action = discard_slot

    info_token_count = info_tokens.sum()
    tokens_available = info_token_count > 0.5
    discard_legal = info_token_count < 7.5

    # ---- partner hand reasoning (fully known identities) ----
    partner_occupied = partner_hand.sum(axis=(1, 2)) > 0.5
    partner_clued = jnp.any(partner_color_flags > 0.5, axis=-1) | jnp.any(partner_rank_flags > 0.5, axis=-1)

    partner_color_onehot = partner_hand.sum(axis=2)  # [slot, color]
    partner_rank_onehot = partner_hand.sum(axis=1)  # [slot, rank]
    partner_color_idx = jnp.argmax(partner_color_onehot, axis=-1)
    partner_rank_idx = jnp.argmax(partner_rank_onehot, axis=-1)

    height_at_slot = jnp.take(firework_height_i, partner_color_idx)
    partner_trash_slot = partner_occupied & (partner_rank_idx < height_at_slot)

    partner_unclued = partner_occupied & (~partner_clued)
    partner_chop_exists = jnp.any(partner_unclued)
    partner_chop_idx = jnp.argmin(jnp.where(partner_unclued, idx5, BIG))

    chop_color = partner_color_idx[partner_chop_idx]
    chop_rank = partner_rank_idx[partner_chop_idx]
    chop_remaining = remaining_count[chop_color, chop_rank]
    chop_other_in_hand = partner_count[chop_color, chop_rank] - 1.0
    chop_discarded = discard_count[chop_color, chop_rank]
    visible_other_2 = chop_discarded + chop_other_in_hand

    need_5save = partner_chop_exists & (chop_rank == 4)
    need_2save = partner_chop_exists & (chop_rank == 1) & (visible_other_2 < 0.5)
    need_2save_critical = (partner_chop_exists & (chop_rank == 1) & (visible_other_2 >= 0.5)
                           & (chop_discarded >= 0.5) & (chop_remaining < 1.5))
    need_critical_generic = (partner_chop_exists & (chop_rank != 4) & (chop_rank != 1)
                             & (chop_remaining < 1.5))
    need_any_save = need_5save | need_2save | need_2save_critical | need_critical_generic

    def bad_touch(matched, target_idx):
        new = matched & (~partner_clued) & partner_occupied
        return jnp.sum(new & (idx5 != target_idx) & partner_trash_slot)

    matched_color_chop = partner_occupied & (partner_color_idx == chop_color)
    matched_rank_chop = partner_occupied & (partner_rank_idx == chop_rank)
    bad_color_chop = bad_touch(matched_color_chop, partner_chop_idx)
    bad_rank_chop = bad_touch(matched_rank_chop, partner_chop_idx)

    save_use_rank = need_5save | need_2save | (
        (need_2save_critical | need_critical_generic) & (bad_rank_chop < bad_color_chop))
    save_value = jnp.where(
        need_5save, 4,
        jnp.where(need_2save, 1, jnp.where(save_use_rank, chop_rank, chop_color)))
    save_action = jnp.where(save_use_rank, 15 + save_value, 10 + save_value)

    # ---- partner play-clue search (evaluate every candidate target) ----
    playable_partner = partner_occupied & (~partner_clued) & (partner_rank_idx == height_at_slot)

    def eval_candidate(t):
        target_color = partner_color_idx[t]
        target_rank = partner_rank_idx[t]

        matched_color_t = partner_occupied & (partner_color_idx == target_color)
        matched_rank_t = partner_occupied & (partner_rank_idx == target_rank)
        new_color_t = matched_color_t & (~partner_clued)
        new_rank_t = matched_rank_t & (~partner_clued)

        focus_color_t = jnp.argmax(jnp.where(new_color_t, idx5, -1))
        focus_rank_t = jnp.argmax(jnp.where(new_rank_t, idx5, -1))

        chop_conflict_color = partner_chop_exists & matched_color_t[partner_chop_idx] & (partner_chop_idx != t)
        chop_conflict_rank = partner_chop_exists & matched_rank_t[partner_chop_idx] & (partner_chop_idx != t)

        valid_color = (~chop_conflict_color) & (focus_color_t == t)
        valid_rank = (~chop_conflict_rank) & (focus_rank_t == t)

        bad_color = bad_touch(matched_color_t, t)
        bad_rank = bad_touch(matched_rank_t, t)

        use_rank = valid_rank & (~valid_color | (bad_rank < bad_color))
        valid = valid_color | valid_rank
        value = jnp.where(use_rank, target_rank, target_color)
        return valid, use_rank, value

    valid_c, use_rank_c, value_c = jax.vmap(eval_candidate)(idx5)
    eligible = playable_partner & valid_c

    chop_eligible = partner_chop_exists & jnp.take(eligible, partner_chop_idx)
    target_idx = jnp.where(
        chop_eligible, partner_chop_idx,
        jnp.argmin(jnp.where(eligible, idx5, BIG)))

    any_playable_partner = jnp.any(eligible)
    use_rank_for_play = jnp.take(use_rank_c, target_idx)
    play_clue_value = jnp.take(value_c, target_idx)
    play_clue_action = jnp.where(use_rank_for_play, 15 + play_clue_value, 10 + play_clue_value)

    valid_play_clue = any_playable_partner & tokens_available

    # ---- forced fallback hint (tokens==8, nothing else to do) ----
    # This clue is not backed by a genuine play/save justification, so it
    # must not manufacture a false "next playable rank" promise. Prefer
    # retouching an already-clued card (a real tempo clue, no new focus);
    # otherwise use a RANK clue (never triggers the color-clue height
    # promise) on the lowest-index occupied slot as a last resort.
    fallback_retouch_exists = jnp.any(partner_clued)
    fallback_retouch_target = jnp.argmin(jnp.where(partner_clued, idx5, BIG))
    fallback_new_target = jnp.argmin(jnp.where(partner_occupied, idx5, BIG))
    fallback_target = jnp.where(fallback_retouch_exists, fallback_retouch_target, fallback_new_target)
    fallback_value = partner_rank_idx[fallback_target]
    fallback_action = 15 + fallback_value

    # ---- combine branches by priority ----
    save_ready = tokens_available & need_any_save

    action = jnp.where(
        any_safe_play, play_action,
        jnp.where(
            save_ready, save_action,
            jnp.where(
                valid_play_clue, play_clue_action,
                jnp.where(discard_legal, discard_action, fallback_action))))
    branch = jnp.where(
        any_safe_play, 1,
        jnp.where(save_ready, 2, jnp.where(valid_play_clue, 3, jnp.where(discard_legal, 4, 5))))

    is_active = jnp.logical_not(obs.legal_actions[20])
    action = jnp.where(is_active, action, 20).astype(jnp.int32)
    branch = jnp.where(is_active, branch, 0).astype(jnp.int32)

    legal_fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    action_is_legal = jnp.take(obs.legal_actions, action)
    action = jnp.where(action_is_legal, action, legal_fallback)

    next_memory = {
        'prev_clued': jnp.where(reset, jnp.zeros((5,), dtype=jnp.bool_), self_clued),
        'frozen_valid': jnp.where(reset, jnp.zeros((5,), dtype=jnp.bool_), frozen_valid_final),
        'frozen_target_rank': jnp.where(reset, jnp.zeros((5,), dtype=jnp.int32), frozen_target_final),
    }
    trace = {'branch': branch}
    return action, next_memory, trace
