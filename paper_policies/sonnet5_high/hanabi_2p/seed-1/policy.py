"""Grounded rule-based Hanabi policy.

Strategy summary (convention-free, grounded in publicly derivable beliefs),
in decision priority order:
  1. Play a card from own hand only when the corrected public belief (V0,
     refined by subtracting cards visible in the partner's hand) puts
     near-all probability mass on the card being currently playable.
  2. Give a hint when it would make a partner's *actually playable* card
     become (near-)certainly playable under the same literal V0 update the
     partner can independently compute; this does not depend on any shared
     "hint means play" convention, only on shared literal-information math.
  3. Discard a card whose corrected belief says it is dead: already played
     past, or permanently unreachable because a needed lower rank has zero
     copies left anywhere.
  4. Give a hint that reveals a partner's sole remaining copy of a card it
     has received no hint about yet (protects against a blind discard),
     or otherwise any hint that gives a touched slot new information.
  5. Otherwise discard: prefer the oldest occupied slot (the common
     convention-free Hanabi default) weighted down by belief-estimated
     uniqueness risk and playability.

All quantities are computed from `obs.features` alone (no hidden state), so
both seats reach mutually consistent beliefs even when generated
independently, since the underlying math (not any convention) is shared.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_hint_play_v1',
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

_MULT_PER_RANK_LIST = [3.0, 2.0, 2.0, 2.0, 1.0]
_RANK_PLUS1_LIST = [1.0, 2.0, 3.0, 4.0, 5.0]

_PLAY_CERTAIN_THRESHOLD = 0.95
_TRASH_CERTAIN_THRESHOLD = 0.95
_HINT_USEFUL_THRESHOLD = 0.05
_HINT_NEWINFO_THRESHOLD = 0.05


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _constants():
    mult_per_rank = jnp.asarray(_MULT_PER_RANK_LIST, dtype=jnp.float32)
    rank_plus1 = jnp.asarray(_RANK_PLUS1_LIST, dtype=jnp.float32)

    # Discard sub-block mapping: 10 thermometer entries -> 5 ranks, per color.
    # rank 0 uses entries [0:3], rank 1 [3:5], rank 2 [5:7], rank 3 [7:9], rank 4 [9:10].
    discard_offsets = (0, 3, 5, 7, 9)
    discard_lens = (3, 2, 2, 2, 1)
    discard_map_t_rows = []
    for r in range(5):
        row = [0.0] * 10
        for i in range(discard_offsets[r], discard_offsets[r] + discard_lens[r]):
            row[i] = 1.0
        discard_map_t_rows.append(row)
    discard_map = jnp.asarray(discard_map_t_rows, dtype=jnp.float32).T  # (10, 5)

    eye5_rows = [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
    eye5 = jnp.asarray(eye5_rows, dtype=jnp.float32)
    return mult_per_rank, rank_plus1, discard_map, eye5


def _safe_normalize(unnorm):
    total = jnp.sum(unnorm, axis=(-2, -1), keepdims=True)
    return unnorm / jnp.maximum(total, 1e-9)


def agent_act(obs, reset, memory, rng):
    del rng
    mult_per_rank, rank_plus1, discard_map, eye5 = _constants()
    features = obs.features

    partner_hand = features[0:125].reshape(5, 5, 5)  # [slot, color, rank]
    fireworks = features[167:192].reshape(5, 5)  # [color, rank]
    discards_raw = features[203:253].reshape(5, 10)  # [color, 10]
    native = features[308:658].reshape(2, 5, 35)
    self_v0 = native[0, :, 0:25].reshape(5, 5, 5)  # [slot, color, rank]
    partner_v0 = native[1, :, 0:25].reshape(5, 5, 5)

    height = jnp.sum(fireworks * rank_plus1[None, :], axis=1)  # (5,) per color
    rank_ids = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    playable_mask = (rank_ids[None, :] == height[:, None]).astype(jnp.float32)  # (5c,5r)
    trash_mask = (rank_ids[None, :] < height[:, None]).astype(jnp.float32)

    discard_count = discards_raw @ discard_map  # (5 colors, 5 ranks)
    mult_grid = jnp.broadcast_to(mult_per_rank[None, :], (5, 5))
    played_count = trash_mask  # exactly one played copy per already-completed rank
    remaining_count = jnp.clip(mult_grid - discard_count - played_count, 0.0, None)  # (5,5)

    # A rank is permanently dead (its color can never reach or pass it) once
    # any not-yet-played predecessor rank (or the rank itself) has zero
    # remaining copies anywhere in the game (all discarded). This is a strict,
    # publicly-derivable extension of "already played" trash.
    avail = (remaining_count > 0.5).astype(jnp.float32)  # (5c,5r)
    avail_or_played = jnp.where(trash_mask > 0.5, 1.0, avail)  # (5c,5r)
    reachable_cumprod = jnp.cumprod(avail_or_played, axis=1)  # (5c,5r)
    extra_dead_mask = (rank_ids[None, :] >= height[:, None]).astype(jnp.float32) * (
        1.0 - (reachable_cumprod > 0.5).astype(jnp.float32)
    )
    dead_mask = jnp.maximum(trash_mask, extra_dead_mask)  # (5c,5r)

    partner_count = jnp.sum(partner_hand, axis=0)  # (5c,5r): counts in partner's actual hand

    occupied_self = obs.legal_actions[5:10].astype(jnp.float32)  # (5,)
    partner_occupied = (jnp.sum(partner_hand, axis=(1, 2)) > 0.5).astype(jnp.float32)  # (5,)

    # --- Own hand: correct the native V0 by removing cards visible in the
    # partner's hand from the remaining-count multiplier (V0 itself omits
    # this correction; we can safely apply it since we directly see the
    # partner's hand). ---
    own_support = (self_v0 > 1e-9).astype(jnp.float32)  # (5s,5c,5r)
    own_remaining = jnp.clip(remaining_count[None, :, :] - partner_count[None, :, :], 0.0, None)
    own_unnorm = own_support * own_remaining
    own_sum = jnp.sum(own_unnorm, axis=(1, 2))
    own_v0c = _safe_normalize(own_unnorm)
    own_v0_final = jnp.where(own_sum[:, None, None] > 1e-9, own_v0c, self_v0)

    playability_self = jnp.sum(own_v0_final * playable_mask[None, :, :], axis=(1, 2))  # (5,)
    trash_self = jnp.sum(own_v0_final * dead_mask[None, :, :], axis=(1, 2))  # (5,)
    uniqueness = (remaining_count <= 1.0).astype(jnp.float32)  # (5c,5r)
    unique_risk_self = jnp.sum(own_v0_final * uniqueness[None, :, :], axis=(1, 2))  # (5,)

    # --- Play values ---
    play_certain = playability_self >= _PLAY_CERTAIN_THRESHOLD
    play_tier = jnp.where(play_certain, 4.0, -1.0)
    play_value = play_tier * 1000.0 + playability_self
    play_value = jnp.where(occupied_self > 0.5, play_value, -1e9)

    # --- Discard values ---
    discard_trash = trash_self >= _TRASH_CERTAIN_THRESHOLD
    slot_age_bias = jnp.asarray([1.28, 0.96, 0.64, 0.32, 0.0], dtype=jnp.float32)  # prefer older slots
    discard_fallback_score = (1.0 - unique_risk_self) - 0.1 * playability_self + slot_age_bias
    discard_tier = jnp.where(discard_trash, 2.0, 0.0)
    discard_score = jnp.where(discard_trash, trash_self, discard_fallback_score)
    discard_value = discard_tier * 1000.0 + discard_score
    discard_value = jnp.where(occupied_self > 0.5, discard_value, -1e9)

    # --- Hints: simulate the literal post-hint V0 update for the partner
    # using only publicly derivable quantities (remaining_count, partner's
    # own native V0 support mask), and reward hints that make an actually
    # playable partner card become certain. ---
    partner_real_playable = jnp.sum(partner_hand * playable_mask[None, :, :], axis=(1, 2))  # (5s,)
    partner_real_playable = partner_real_playable * partner_occupied
    old_playability_partner = jnp.sum(partner_v0 * playable_mask[None, :, :], axis=(1, 2))  # (5s,)
    partner_support = (partner_v0 > 1e-9).astype(jnp.float32)  # (5s,5c,5r)

    partner_color_onehot = jnp.sum(partner_hand, axis=2)  # (5s,5c)
    partner_rank_onehot = jnp.sum(partner_hand, axis=1)  # (5s,5r)

    denom_occupied = jnp.maximum(jnp.sum(partner_occupied), 1.0)

    # Cards that are the sole remaining copy and have received no positive
    # hint flag yet are at risk of being blindly discarded by any reasonable
    # partner; protecting them (via any hint touching that slot) is grounded
    # in the same shared, convention-free information math.
    partner_color_flags = native[1, :, 25:30]  # (5s,5c) positive color-hint flags
    partner_rank_flags = native[1, :, 30:35]  # (5s,5r) positive rank-hint flags
    partner_no_info = (
        jnp.sum(partner_color_flags, axis=1) + jnp.sum(partner_rank_flags, axis=1)
    ) < 0.5  # (5s,)
    partner_real_unique = jnp.sum(partner_hand * uniqueness[None, :, :], axis=(1, 2))  # (5s,)
    critical_partner = partner_occupied * partner_real_unique * partner_no_info.astype(jnp.float32)

    # Color hints: h indexes hinted color 0..4.
    touched_color = (partner_color_onehot.T > 0.5)  # (5h,5s)
    pos_mask_c = partner_support[None, :, :, :] * eye5[:, None, :, None]
    neg_mask_c = partner_support[None, :, :, :] * (1.0 - eye5)[:, None, :, None]
    new_mask_c = jnp.where(touched_color[:, :, None, None], pos_mask_c, neg_mask_c)
    new_unnorm_c = new_mask_c * remaining_count[None, None, :, :]
    new_v0_c = _safe_normalize(new_unnorm_c)
    new_playability_c = jnp.sum(new_v0_c * playable_mask[None, None, :, :], axis=(2, 3))  # (5h,5s)
    gain_c = jnp.sum(
        partner_real_playable[None, :]
        * jnp.clip(new_playability_c - old_playability_partner[None, :], 0.0, None),
        axis=1,
    )
    already_color = (partner_color_flags.T > 0.5)  # (5h,5s)
    new_info_c = jnp.sum(
        touched_color.astype(jnp.float32)
        * (1.0 - already_color.astype(jnp.float32))
        * partner_occupied[None, :],
        axis=1,
    ) / denom_occupied
    save_gain_c = jnp.sum(critical_partner[None, :] * touched_color.astype(jnp.float32), axis=1)

    # Rank hints: h indexes hinted rank 0..4.
    touched_rank = (partner_rank_onehot.T > 0.5)  # (5h,5s)
    pos_mask_r = partner_support[None, :, :, :] * eye5[:, None, None, :]
    neg_mask_r = partner_support[None, :, :, :] * (1.0 - eye5)[:, None, None, :]
    new_mask_r = jnp.where(touched_rank[:, :, None, None], pos_mask_r, neg_mask_r)
    new_unnorm_r = new_mask_r * remaining_count[None, None, :, :]
    new_v0_r = _safe_normalize(new_unnorm_r)
    new_playability_r = jnp.sum(new_v0_r * playable_mask[None, None, :, :], axis=(2, 3))
    gain_r = jnp.sum(
        partner_real_playable[None, :]
        * jnp.clip(new_playability_r - old_playability_partner[None, :], 0.0, None),
        axis=1,
    )
    already_rank = (partner_rank_flags.T > 0.5)  # (5h,5s)
    new_info_r = jnp.sum(
        touched_rank.astype(jnp.float32)
        * (1.0 - already_rank.astype(jnp.float32))
        * partner_occupied[None, :],
        axis=1,
    ) / denom_occupied
    save_gain_r = jnp.sum(critical_partner[None, :] * touched_rank.astype(jnp.float32), axis=1)

    gain = jnp.concatenate([gain_c, gain_r])  # (10,) [color0..4, rank0..4]
    new_info = jnp.concatenate([new_info_c, new_info_r])  # (10,)
    save_gain = jnp.concatenate([save_gain_c, save_gain_r])  # (10,)

    hint_useful = gain >= _HINT_USEFUL_THRESHOLD
    hint_saves = save_gain > 0.5
    hint_has_new_info = new_info >= _HINT_NEWINFO_THRESHOLD

    hint_tier = jnp.where(
        hint_useful, 3.0, jnp.where(hint_saves, 1.5, jnp.where(hint_has_new_info, 1.0, -2.0))
    )
    hint_score = jnp.where(
        hint_useful,
        jnp.clip(gain, 0.0, 1.0),
        jnp.where(
            hint_saves,
            jnp.clip(save_gain, 0.0, 5.0) / 5.0,
            jnp.where(hint_has_new_info, new_info, 0.0),
        ),
    )
    hint_value = hint_tier * 1000.0 + hint_score  # (10,)

    noop_value = jnp.asarray([1.0e6], dtype=jnp.float32)  # dominates only when it is the sole legal action

    values = jnp.concatenate([discard_value, play_value, hint_value, noop_value])  # (21,)
    values = jnp.where(obs.legal_actions, values, -1e9)

    action = jnp.argmax(values).astype(jnp.int32)
    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    return action, next_memory, {}
