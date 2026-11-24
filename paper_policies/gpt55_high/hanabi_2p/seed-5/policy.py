"""Convention-oriented Hanabi policy for the local observation API."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NEG = -1.0e9
COPIES_TUPLE = (3.0, 2.0, 2.0, 2.0, 1.0)


def _ranks():
    return jnp.arange(5, dtype=jnp.int32)


def _slots():
    return jnp.arange(5, dtype=jnp.int32)


def init_agent():
    return jnp.zeros((5,), dtype=jnp.bool_)


def _first_true(mask):
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _best_action(scores, offset):
    return (jnp.argmax(scores).astype(jnp.int32) + jnp.asarray(offset, dtype=jnp.int32))


def _decode(features):
    partner = features[0:125].reshape((5, 5, 5))
    fireworks_raw = features[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks_raw * jnp.arange(1, 6, dtype=jnp.float32), axis=1).astype(jnp.int32)
    info = jnp.sum(features[192:200]).astype(jnp.int32)
    lives = jnp.sum(features[200:203]).astype(jnp.int32)
    disc_raw = features[203:253].reshape((5, 10))
    discards = jnp.stack(
        [
            jnp.sum(disc_raw[:, 0:3], axis=1),
            jnp.sum(disc_raw[:, 3:5], axis=1),
            jnp.sum(disc_raw[:, 5:7], axis=1),
            jnp.sum(disc_raw[:, 7:9], axis=1),
            disc_raw[:, 9],
        ],
        axis=1,
    )
    last = features[253:308]
    knowledge = features[308:658].reshape((2, 5, 35))
    return partner, heights, info, lives, discards, last, knowledge


def _playable_grid(heights):
    return (_ranks()[None, :] == heights[:, None]).astype(jnp.float32)


def _play_prob(belief, heights):
    return jnp.sum(belief * _playable_grid(heights)[None, :, :], axis=(1, 2))


def _occupied_from_belief(belief):
    return jnp.sum(belief, axis=(1, 2)) > 0.5


def _known_hinted(knowledge_player):
    return jnp.sum(knowledge_player[:, 25:35], axis=1) > 0.5


def _shift_after_own_remove(flags, remove_slot):
    idx = _slots()
    src = jnp.minimum(idx + (idx >= remove_slot).astype(jnp.int32), 4)
    shifted = flags[src]
    return jnp.where(idx == 4, False, shifted)


def _update_memory(memory, reset, last, own_play_prob, lives):
    flags = memory
    actor_self = last[0] > 0.5
    was_play_or_discard = (last[2] + last[3]) > 0.5
    removed_slot = _first_true(last[23:28] > 0.5)
    shifted = _shift_after_own_remove(flags, removed_slot)
    flags = jnp.where(actor_self & was_play_or_discard, shifted, flags)

    hint_to_self = (last[6] > 0.5) & ((last[4] + last[5]) > 0.5)
    matches = last[18:23] > 0.5
    single_match = jnp.sum(matches.astype(jnp.int32)) == 1
    color_hint = last[4] > 0.5
    rank_hint = last[5] > 0.5
    direct_rank = rank_hint & (own_play_prob > 0.10)
    direct_color = color_hint & (own_play_prob > 0.10)
    direct = matches & single_match & (direct_rank | direct_color) & (lives > 1)
    flags = jnp.where(hint_to_self, flags | direct, flags)
    return jnp.where(reset, jnp.zeros_like(flags), flags)


def _hint_scores(partner, partner_belief, heights, legal):
    playable = _playable_grid(heights)
    actual_playable = jnp.sum(partner * playable[None, :, :], axis=(1, 2)) > 0.5

    color_masks = jax.nn.one_hot(_ranks(), 5, dtype=jnp.float32)
    color_masks = color_masks[:, :, None] * jnp.ones((5, 5), dtype=jnp.float32)[None, :, :]
    rank_masks = jnp.ones((5, 5), dtype=jnp.float32)[None, :, :] * jax.nn.one_hot(_ranks(), 5, dtype=jnp.float32)[:, None, :]

    def score_one(mask, is_legal):
        slot_match = jnp.sum(partner * mask[None, :, :], axis=(1, 2)) > 0.5
        keep = jnp.where(slot_match[:, None, None], mask[None, :, :], 1.0 - mask[None, :, :])
        filt = partner_belief * keep
        norm = jnp.sum(filt, axis=(1, 2), keepdims=True)
        new_belief = filt / jnp.maximum(norm, 1.0e-6)
        cert = _play_prob(new_belief, heights) > 0.999
        cert_count = jnp.sum((cert & slot_match).astype(jnp.float32))
        good_count = jnp.sum((actual_playable & slot_match).astype(jnp.float32))
        bad_count = jnp.sum(((~actual_playable) & slot_match).astype(jnp.float32))
        raw = 100.0 * cert_count + 10.0 * good_count - 35.0 * bad_count
        return jnp.where(is_legal, raw, NEG)

    color_scores = jax.vmap(score_one)(color_masks, legal[10:15])
    rank_scores = jax.vmap(score_one)(rank_masks, legal[15:20]) + 3.0
    return color_scores, rank_scores


def _fallback_hint_scores(partner, partner_knowledge, heights, legal):
    playable = _playable_grid(heights)
    actual_playable = jnp.sum(partner * playable[None, :, :], axis=(1, 2)) > 0.5
    partner_color = jnp.sum(partner, axis=2) > 0.5
    partner_rank = jnp.sum(partner, axis=1) > 0.5
    color_good = partner_color.T @ actual_playable.astype(jnp.float32)
    rank_good = partner_rank.T @ actual_playable.astype(jnp.float32)
    rank_five_hinted = partner_knowledge[:, 34] > 0.5
    unhinted_five = jnp.any(partner_rank[:, 4] & (~rank_five_hinted)).astype(jnp.float32)
    color_scores = jnp.where(legal[10:15], color_good, NEG)
    rank_scores = jnp.where(
        legal[15:20],
        rank_good + 30.0 * jax.nn.one_hot(4, 5, dtype=jnp.float32) * unhinted_five,
        NEG,
    )
    return color_scores, rank_scores


def _choose_hint(partner, partner_knowledge, partner_belief, heights, legal):
    color_scores, rank_scores = _hint_scores(partner, partner_belief, heights, legal)
    best_color = jnp.max(color_scores)
    best_rank = jnp.max(rank_scores)
    cert_action = jnp.where(best_rank >= best_color, _best_action(rank_scores, 15), _best_action(color_scores, 10))
    cert_score = jnp.maximum(best_color, best_rank)

    fb_color, fb_rank = _fallback_hint_scores(partner, partner_knowledge, heights, legal)
    best_fb_color = jnp.max(fb_color)
    best_fb_rank = jnp.max(fb_rank)
    fb_action = jnp.where(best_fb_rank >= best_fb_color, _best_action(fb_rank, 15), _best_action(fb_color, 10))
    fb_score = jnp.maximum(best_fb_color, best_fb_rank)
    action = jnp.where(cert_score > 0.0, cert_action, fb_action)
    score = jnp.where(cert_score > 0.0, cert_score, fb_score)
    return action.astype(jnp.int32), score


def _discard_scores(own_belief, own_hint_flags, heights, discards, legal):
    occupied = _occupied_from_belief(own_belief)
    ranks = _ranks()
    dead_grid = (ranks[None, :] < heights[:, None]).astype(jnp.float32)
    dead_prob = jnp.sum(own_belief * dead_grid[None, :, :], axis=(1, 2))
    play_prob = _play_prob(own_belief, heights)
    copies = jnp.asarray(COPIES_TUPLE, dtype=jnp.float32)
    remaining = copies[None, :] - discards
    critical_grid = ((remaining <= 1.0) & (ranks[None, :] >= heights[:, None])).astype(jnp.float32)
    critical_prob = jnp.sum(own_belief * critical_grid[None, :, :], axis=(1, 2))
    unhinted = (~own_hint_flags).astype(jnp.float32)
    age_bonus = (5 - _slots()).astype(jnp.float32)
    raw = 120.0 * dead_prob + 12.0 * unhinted + age_bonus - 85.0 * critical_prob - 55.0 * play_prob
    return jnp.where(legal[0:5] & occupied, raw, NEG), dead_prob


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions
    partner, heights, info, lives, discards, last, knowledge = _decode(features)
    own_belief = knowledge[0, :, 0:25].reshape((5, 5, 5))
    partner_belief = knowledge[1, :, 0:25].reshape((5, 5, 5))

    own_play_prob = _play_prob(own_belief, heights)
    next_memory = _update_memory(memory, reset, last, own_play_prob, lives)

    occupied = _occupied_from_belief(own_belief)
    certain_play = (own_play_prob > 0.999) & occupied & legal[5:10]
    clue_play = next_memory & (own_play_prob > 0.05) & (lives > 1) & legal[5:10]
    play_mask = certain_play | clue_play
    play_scores = jnp.where(play_mask, 100.0 - _slots().astype(jnp.float32), NEG)
    play_action = _best_action(play_scores, 5)
    has_play = jnp.max(play_scores) > 0.0

    hint_action, hint_score = _choose_hint(partner, knowledge[1], partner_belief, heights, legal)
    has_hint = hint_score > NEG / 2.0
    has_good_hint = hint_score > 0.0

    own_hint_flags = _known_hinted(knowledge[0])
    discard_scores, dead_prob = _discard_scores(own_belief, own_hint_flags, heights, discards, legal)
    discard_action = _best_action(discard_scores, 0)
    has_discard = jnp.max(discard_scores) > NEG / 2.0
    has_safe_discard = jnp.max(jnp.where(dead_prob > 0.999, discard_scores, NEG)) > NEG / 2.0
    safe_discard_action = _best_action(jnp.where(dead_prob > 0.999, discard_scores, NEG), 0)

    lowest_play = _best_action(jnp.where(legal[5:10], 1.0, NEG), 5)
    lowest_legal = _first_true(legal)

    active_action = jnp.where(has_discard, discard_action, lowest_legal)
    active_action = jnp.where(has_good_hint, hint_action, active_action)
    active_action = jnp.where(has_safe_discard & (info <= 2), safe_discard_action, active_action)
    active_action = jnp.where(has_play, play_action, active_action)
    active_action = jnp.where((~has_play) & (~has_good_hint) & (~has_discard) & has_hint, hint_action, active_action)
    active_action = jnp.where((~has_play) & (~has_good_hint) & (~has_discard) & (~has_hint), lowest_play, active_action)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), active_action.astype(jnp.int32))

    trace = {
        "play_prob_max": jnp.max(own_play_prob),
        "hint_score": hint_score.astype(jnp.float32),
        "discard_score": jnp.max(discard_scores),
        "memory_flags": jnp.sum(next_memory.astype(jnp.float32)),
    }
    return action, next_memory, trace
