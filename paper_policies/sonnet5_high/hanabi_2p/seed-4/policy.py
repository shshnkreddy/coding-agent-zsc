"""Rule-based Hanabi policy built purely on public, literal observation features.

Design notes (why this is convention-free):
  - The environment already exposes, per relative player and slot, a public
    "V0" belief distribution over card identity derived only from remaining
    public counts and the public hint history (see native_knowledge block).
    Both seats can compute this identically, and an independent partner can
    plausibly derive similar beliefs about its own hand, since it is a
    function of shared public information, not a private signal invented by
    this policy.
  - We play a card only when our own public belief says it is (near) certain
    to be playable, hint a partner card that is genuinely playable but not
    yet publicly obvious to the partner, and discard cards our belief marks
    as certain trash. This uses literal, truthful signals (real hints and
    real card safety) rather than an arbitrary shared code, so it should
    degrade gracefully against differently-designed partners.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NUM_COLORS = 5
NUM_RANKS = 5
NUM_SLOTS = 5

# Play-confidence threshold, made stricter as remaining lives shrink so a
# single miscalibrated guess cannot as easily end the episode at bomb-zero.
PLAY_THRESHOLD_BY_LIVES = (1.0, 0.99, 0.75, 0.45)  # index by life count 0..3
HINT_THRESHOLD = 0.0


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions

    partner_hand = features[0:125].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    fireworks = features[167:192].reshape(NUM_COLORS, NUM_RANKS)
    info_tokens = features[192:200]
    lives = features[200:203]
    native = features[308:658].reshape(2, NUM_SLOTS, 35)

    self_v0 = native[0, :, 0:25].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    partner_v0 = native[1, :, 0:25].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)

    rank_idx = jnp.arange(NUM_RANKS, dtype=jnp.float32)
    height = jnp.sum(fireworks * (rank_idx + 1.0)[None, :], axis=-1)  # (colors,) in [0, 5]
    needed_onehot = jax.nn.one_hot(height.astype(jnp.int32), NUM_RANKS)  # (colors, ranks)
    trash_indicator = (rank_idx[None, :] < height[:, None]).astype(jnp.float32)  # (colors, ranks)

    info_available = jnp.sum(info_tokens) > 0.5

    self_occupied = jnp.sum(self_v0, axis=(1, 2)) > 0.5
    partner_occupied = jnp.sum(partner_hand, axis=(1, 2)) > 0.5

    prob_playable_self = jnp.sum(self_v0 * needed_onehot[None], axis=(1, 2))
    prob_trash_self = jnp.sum(self_v0 * trash_indicator[None], axis=(1, 2))

    # Tier 1: play a self card we are (near) certain is playable. The bar
    # rises as remaining lives shrink, so a wrong guess is less likely to
    # ever reach the catastrophic all-lives-lost (bomb-zero) outcome.
    play_score = jnp.where(self_occupied, prob_playable_self, -1.0)
    best_play_slot = jnp.argmax(play_score)
    best_play_value = play_score[best_play_slot]
    play_action = 5 + best_play_slot
    life_count = jnp.sum(lives).astype(jnp.int32)
    play_threshold = jnp.asarray(PLAY_THRESHOLD_BY_LIVES)[life_count]
    play_valid = (best_play_value >= play_threshold) & legal[play_action]

    # Discard candidate: pick the safest slot to discard (most likely trash,
    # least likely playable).
    safety_score = prob_trash_self - prob_playable_self
    discard_score = jnp.where(self_occupied, safety_score, -100.0)
    best_discard_slot = jnp.argmax(discard_score)
    discard_action = best_discard_slot
    discard_valid = legal[discard_action]

    # Tier 2: hint a partner card that is genuinely playable but not yet
    # publicly obvious to the partner (their own V0 does not already show it).
    actual_playable = jnp.sum(partner_hand * needed_onehot[None], axis=(1, 2))
    partner_known_playable = jnp.sum(partner_v0 * needed_onehot[None], axis=(1, 2))
    hint_candidate_score = actual_playable * (1.0 - partner_known_playable) * partner_occupied
    best_hint_slot = jnp.argmax(hint_candidate_score)
    hint_needed = hint_candidate_score[best_hint_slot] > HINT_THRESHOLD

    actual_color = jnp.argmax(jnp.sum(partner_hand[best_hint_slot], axis=1))
    actual_rank = jnp.argmax(jnp.sum(partner_hand[best_hint_slot], axis=0))

    v0_slot = partner_v0[best_hint_slot]
    color_mass = jnp.sum(v0_slot[actual_color, :])
    color_play_mass = v0_slot[actual_color, height[actual_color].astype(jnp.int32)]
    prob_after_color = jnp.where(color_mass > 1e-6, color_play_mass / color_mass, 0.0)

    rank_mass = jnp.sum(v0_slot[:, actual_rank])
    rank_needed_mask = (height.astype(jnp.int32) == actual_rank).astype(jnp.float32)
    rank_play_mass = jnp.sum(v0_slot[:, actual_rank] * rank_needed_mask)
    prob_after_rank = jnp.where(rank_mass > 1e-6, rank_play_mass / rank_mass, 0.0)

    use_color_hint = prob_after_color >= prob_after_rank
    hint_action_targeted = jnp.where(use_color_hint, 10 + actual_color, 15 + actual_rank)
    hint_targeted_valid = hint_needed & info_available & legal[hint_action_targeted]

    # Tier 4: no urgent play/hint/discard available (e.g. tokens capped at 8
    # so discard is illegal). Prefer revealing a partner card that is actual
    # trash but not yet publicly known as such (so the partner can safely
    # discard it later); otherwise fall back to the least-certain slot.
    actual_trash = jnp.sum(partner_hand * trash_indicator[None], axis=(1, 2))
    partner_known_trash = jnp.sum(partner_v0 * trash_indicator[None], axis=(1, 2))
    trash_hint_score = actual_trash * (1.0 - partner_known_trash) * partner_occupied
    slot_certainty = jnp.max(partner_v0.reshape(NUM_SLOTS, -1), axis=1)
    fallback_score = jnp.where(partner_occupied, -slot_certainty, -10.0)
    have_trash_target = trash_hint_score[jnp.argmax(trash_hint_score)] > 0.0
    fallback_slot = jnp.where(have_trash_target, jnp.argmax(trash_hint_score), jnp.argmax(fallback_score))
    fb_color = jnp.argmax(jnp.sum(partner_hand[fallback_slot], axis=1))
    fb_rank = jnp.argmax(jnp.sum(partner_hand[fallback_slot], axis=0))
    v0_fb = partner_v0[fallback_slot]
    fb_color_mass = jnp.sum(v0_fb[fb_color, :])
    fb_rank_mass = jnp.sum(v0_fb[:, fb_rank])
    fb_use_color = fb_color_mass <= fb_rank_mass
    fallback_hint_action = jnp.where(fb_use_color, 10 + fb_color, 15 + fb_rank)
    fallback_hint_valid = info_available & legal[fallback_hint_action]

    ultimate_fallback = jnp.argmax(legal)

    action = jnp.where(
        play_valid, play_action,
        jnp.where(
            hint_targeted_valid, hint_action_targeted,
            jnp.where(
                discard_valid, discard_action,
                jnp.where(
                    fallback_hint_valid, fallback_hint_action,
                    ultimate_fallback,
                ),
            ),
        ),
    ).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)

    tier = jnp.where(
        play_valid, 0,
        jnp.where(hint_targeted_valid, 1, jnp.where(discard_valid, 2, jnp.where(fallback_hint_valid, 3, 4))),
    ).astype(jnp.int32)
    trace = {"tier": tier}
    return action, next_memory, trace
