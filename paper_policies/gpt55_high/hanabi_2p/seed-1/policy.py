"""Conservative Hanabi policy for XP-oriented coordination."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'xp_conservative_hints_v7', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.zeros((5,), dtype=jnp.int32)


def _first_true(mask):
    idx = jnp.arange(5, dtype=jnp.int32)
    return jnp.min(jnp.where(mask, idx, jnp.asarray(5, dtype=jnp.int32)))


def _shift_after_removal(values, removed_onehot):
    idx = jnp.arange(5)
    removed = jnp.argmax(removed_onehot).astype(jnp.int32)
    shifted_left = jnp.concatenate([values[1:], jnp.asarray([0], dtype=values.dtype)])
    return jnp.where(idx < removed, values, shifted_left)


def _choose_hint(legal, partner_cards, partner_occ, partner_playable):
    slot_idx = jnp.arange(5)
    colors = jnp.arange(5)
    ranks = jnp.arange(5)
    partner_color = jnp.argmax(jnp.sum(partner_cards, axis=2), axis=1)
    partner_rank = jnp.argmax(jnp.sum(partner_cards, axis=1), axis=1)

    color_match = (partner_color[:, None] == colors[None, :]) & partner_occ[:, None]
    rank_match = (partner_rank[:, None] == ranks[None, :]) & partner_occ[:, None]

    color_play_count = jnp.sum(color_match & partner_playable[:, None], axis=0)
    rank_play_count = jnp.sum(rank_match & partner_playable[:, None], axis=0)
    color_total = jnp.sum(color_match, axis=0)
    rank_total = jnp.sum(rank_match, axis=0)

    color_first = jnp.min(jnp.where(color_match, slot_idx[:, None], 5), axis=0)
    rank_first = jnp.min(jnp.where(rank_match, slot_idx[:, None], 5), axis=0)
    color_first_playable = jnp.any(
        color_match & partner_playable[:, None] & (slot_idx[:, None] == color_first[None, :]),
        axis=0,
    )
    rank_first_playable = jnp.any(
        rank_match & partner_playable[:, None] & (slot_idx[:, None] == rank_first[None, :]),
        axis=0,
    )

    legal_color = legal[10:15]
    legal_rank = legal[15:20]
    color_ok = legal_color & (color_play_count > 0) & color_first_playable
    rank_ok = legal_rank & (rank_play_count > 0) & rank_first_playable

    color_score = (
        900.0
        + 35.0 * color_play_count.astype(jnp.float32)
        - 9.0 * (color_total - color_play_count).astype(jnp.float32)
        - color_first.astype(jnp.float32)
    )
    rank_score = (
        1000.0
        + 35.0 * rank_play_count.astype(jnp.float32)
        - 7.0 * (rank_total - rank_play_count).astype(jnp.float32)
        - rank_first.astype(jnp.float32)
    )
    color_score = jnp.where(color_ok, color_score, -1.0e9)
    rank_score = jnp.where(rank_ok, rank_score, -1.0e9)
    all_scores = jnp.concatenate([color_score, rank_score])
    best = jnp.argmax(all_scores).astype(jnp.int32)
    return 10 + best, jnp.max(all_scores)


def _choose_stall_hint(legal, partner_cards, partner_occ):
    partner_color = jnp.argmax(jnp.sum(partner_cards, axis=2), axis=1)
    colors = jnp.arange(5)
    color_match = (partner_color[:, None] == colors[None, :]) & partner_occ[:, None]
    color_count = jnp.sum(color_match, axis=0)
    score = jnp.where(legal[10:15] & (color_count > 0), color_count.astype(jnp.float32), -1.0e9)
    return 10 + jnp.argmax(score).astype(jnp.int32), jnp.max(score)


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions

    last = f[253:308]
    acted_self = last[0] > 0.5
    acted_partner = last[1] > 0.5
    was_play_or_discard = (last[2] > 0.5) | (last[3] > 0.5)
    was_color_hint = last[4] > 0.5
    was_rank_hint = last[5] > 0.5
    hinted_self = last[6] > 0.5
    hint_matches = last[18:23] > 0.5

    shifted = _shift_after_removal(memory, last[23:28])
    remembered = jnp.where(acted_self & was_play_or_discard, shifted, memory)
    hint_slot = _first_true(hint_matches)
    hint_values = jnp.where(jnp.arange(5) == hint_slot, jnp.where(was_rank_hint, 2, 1), remembered)
    got_hint = acted_partner & hinted_self & (was_color_hint | was_rank_hint) & jnp.any(hint_matches)
    remembered = jnp.where(got_hint, hint_values, remembered)
    remembered = jnp.where(reset, jnp.zeros((5,), dtype=jnp.int32), remembered)

    partner_cards = f[0:125].reshape((5, 5, 5))
    partner_occ = jnp.sum(partner_cards, axis=(1, 2)) > 0.5
    fireworks = f[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks * jnp.arange(1, 6, dtype=jnp.float32)[None, :], axis=1).astype(jnp.int32)
    rank_ids = jnp.arange(5, dtype=jnp.int32)
    playable_type = (rank_ids[None, :] == heights[:, None]) & (heights[:, None] < 5)
    obsolete_type = rank_ids[None, :] < heights[:, None]

    partner_playable = jnp.sum(partner_cards * playable_type.astype(jnp.float32)[None, :, :], axis=(1, 2)) > 0.5

    know = f[308:658].reshape((2, 5, 35))[0]
    own_v0 = know[:, 0:25].reshape((5, 5, 5))
    discard_raw = f[203:253].reshape((5, 10))
    discard_counts = jnp.stack(
        [
            jnp.sum(discard_raw[:, 0:3], axis=1),
            jnp.sum(discard_raw[:, 3:5], axis=1),
            jnp.sum(discard_raw[:, 5:7], axis=1),
            jnp.sum(discard_raw[:, 7:9], axis=1),
            discard_raw[:, 9],
        ],
        axis=1,
    )
    multiplicities = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.float32)
    firework_counts = obsolete_type.astype(jnp.float32)
    public_remaining = jnp.maximum(multiplicities[None, :] - discard_counts - firework_counts, 0.0)
    partner_counts = jnp.sum(partner_cards, axis=0)
    adjusted_remaining = jnp.maximum(public_remaining - partner_counts, 0.0)
    visibility_factor = adjusted_remaining / jnp.maximum(public_remaining, 1.0)
    adjusted_v0 = own_v0 * visibility_factor[None, :, :]
    adjusted_norm = jnp.sum(adjusted_v0, axis=(1, 2), keepdims=True)
    own_v0 = jnp.where(adjusted_norm > 0.0, adjusted_v0 / adjusted_norm, own_v0)
    own_occ = jnp.sum(own_v0, axis=(1, 2)) > 0.0
    play_prob = jnp.sum(own_v0 * playable_type.astype(jnp.float32)[None, :, :], axis=(1, 2))
    safe_prob = jnp.sum(own_v0 * obsolete_type.astype(jnp.float32)[None, :, :], axis=(1, 2))
    rank4_prob = jnp.sum(own_v0[:, :, 4], axis=1)

    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    info = jnp.sum(f[192:200]).astype(jnp.int32)
    deck_count = jnp.sum(f[127:167]).astype(jnp.int32)
    active = ~legal[20]

    play_allowed = legal[5:10] & own_occ
    convention_play = ((remembered >= 2) & (lives > 1)) | ((remembered >= 1) & (play_prob > 0.45) & (lives > 1))
    endgame_play = ((deck_count <= 4) & (play_prob > 0.62) & (lives > 1)) | ((deck_count == 0) & (play_prob > 0.45) & (lives > 1))
    prob_play = (play_prob > 0.985) | ((play_prob > 0.78) & (lives > 2)) | endgame_play
    play_ok = play_allowed & (prob_play | convention_play)
    play_score = jnp.where(
        play_ok,
        play_prob + 0.30 * (remembered >= 2).astype(jnp.float32) + 0.08 * (remembered == 1).astype(jnp.float32) - 0.002 * jnp.arange(5, dtype=jnp.float32),
        -1.0e9,
    )
    play_slot = jnp.argmax(play_score).astype(jnp.int32)
    play_action = 5 + play_slot
    has_play = jnp.max(play_score) > -1.0e8

    hint_action, hint_score = _choose_hint(legal, partner_cards, partner_occ, partner_playable)
    has_hint = hint_score > -1.0e8

    discard_allowed = legal[0:5] & own_occ
    discard_score = safe_prob - 0.35 * play_prob - 0.18 * rank4_prob - 0.006 * jnp.arange(5, dtype=jnp.float32)
    discard_score = jnp.where(discard_allowed, discard_score, -1.0e9)
    discard_slot = jnp.argmax(discard_score).astype(jnp.int32)
    discard_action = discard_slot
    best_safe_discard = jnp.max(safe_prob * discard_allowed.astype(jnp.float32))
    has_safe_discard = best_safe_discard > 0.50
    must_discard = (info <= 5) & jnp.any(discard_allowed)

    stall_action, stall_score = _choose_stall_hint(legal, partner_cards, partner_occ)
    has_stall = stall_score > -1.0e8
    best_legal = jnp.argmax(legal).astype(jnp.int32)

    action = jnp.where(has_play, play_action, best_legal)
    action = jnp.where((~has_play) & has_hint, hint_action, action)
    action = jnp.where((~has_play) & (~has_hint) & (has_safe_discard | must_discard), discard_action, action)
    action = jnp.where((~has_play) & (~has_hint) & (~(has_safe_discard | must_discard)) & has_stall, stall_action, action)
    action = jnp.where(active, action, jnp.asarray(20, dtype=jnp.int32)).astype(jnp.int32)

    remembered = jnp.where(own_occ, remembered, jnp.zeros((5,), dtype=jnp.int32))
    trace = {
        'play_p': jnp.max(play_prob),
        'safe_p': jnp.max(safe_prob),
        'hint_score': hint_score,
        'memory_sum': jnp.sum(remembered),
    }
    return action, remembered, trace
