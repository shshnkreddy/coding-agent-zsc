"""Heuristic Hanabi policy driven only by public/observable information.

The strategy never relies on a private convention that would require the
partner to run the same code: every decision is derived from the truthful,
publicly-visible hint history encoded in the native V0 belief and hint-flag
blocks of the observation, plus the fully-visible partner hand. A card is
only played when the public belief already makes it certain to succeed, and
hints are only given when they convey genuine new information (measured the
same way any competent partner could measure it from their own side).
"""

import jax
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

NUM_COLORS = 5
NUM_RANKS = 5
HAND_SIZE = 5
MULTIPLICITY_LIST = (3.0, 2.0, 2.0, 2.0, 1.0)

PLAY_CERTAINTY = 0.999


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _next_rank_needed(fireworks):
    # fireworks: (colors, ranks) one-hot of highest played rank id per color.
    has_any = jnp.sum(fireworks, axis=1)
    idx = jnp.argmax(fireworks, axis=1)
    return jnp.where(has_any > 0.5, idx + 1, 0).astype(jnp.int32)


def _dead_mask(discard_counts, next_rank_needed, multiplicity):
    ranks = jnp.arange(NUM_RANKS, dtype=jnp.int32)[None, :]
    nrn = next_rank_needed[:, None]
    exhausted = discard_counts >= (multiplicity[None, :] - 0.5)
    relevant = exhausted & (ranks >= nrn)
    big = NUM_RANKS + 1
    blocked_rank = jnp.min(jnp.where(relevant, ranks, big), axis=1)
    return (ranks < nrn) | (ranks >= blocked_rank[:, None])


def agent_act(obs, reset, memory, rng):
    del rng
    feats = obs.features.astype(jnp.float32)
    legal = obs.legal_actions

    partner_hand = feats[0:125].reshape(HAND_SIZE, NUM_COLORS, NUM_RANKS)
    fireworks = feats[167:192].reshape(NUM_COLORS, NUM_RANKS)
    info_tokens = jnp.sum(feats[192:200])
    discard_block = feats[203:253].reshape(NUM_COLORS, 10)
    discard_counts = jnp.stack(
        [
            jnp.sum(discard_block[:, 0:3], axis=1),
            jnp.sum(discard_block[:, 3:5], axis=1),
            jnp.sum(discard_block[:, 5:7], axis=1),
            jnp.sum(discard_block[:, 7:9], axis=1),
            jnp.sum(discard_block[:, 9:10], axis=1),
        ],
        axis=1,
    )

    knowledge = feats[308:658].reshape(2, HAND_SIZE, 35)
    own_knowledge = knowledge[0]
    partner_knowledge = knowledge[1]

    multiplicity = jnp.asarray(MULTIPLICITY_LIST, dtype=jnp.float32)
    next_rank_needed = _next_rank_needed(fireworks)
    dead_mask = _dead_mask(discard_counts, next_rank_needed, multiplicity)

    ranks_row = jnp.arange(NUM_RANKS, dtype=jnp.int32)[None, :]
    playable_mask = ranks_row == next_rank_needed[:, None]
    played_mask = (ranks_row < next_rank_needed[:, None]).astype(jnp.float32)
    remaining_count = multiplicity[None, :] - discard_counts - played_mask
    critical_mask = (
        (remaining_count <= 1.0001)
        & (~dead_mask)
        & (ranks_row >= next_rank_needed[:, None])
    )

    own_v0_raw = own_knowledge[:, 0:25].reshape(HAND_SIZE, NUM_COLORS, NUM_RANKS)
    occupied_own = jnp.sum(own_knowledge[:, 0:25], axis=1) > 1e-6

    own_color_flag_block = own_knowledge[:, 25:30]
    own_rank_flag_block = own_knowledge[:, 30:35]
    own_color_known = jnp.sum(own_color_flag_block, axis=1) > 0.5
    own_rank_known = jnp.sum(own_rank_flag_block, axis=1) > 0.5
    own_known_color_idx = jnp.argmax(own_color_flag_block, axis=1)
    own_known_rank_idx = jnp.argmax(own_rank_flag_block, axis=1)
    full_known_own = own_color_known & own_rank_known & occupied_own

    color_onehot = jax.nn.one_hot(own_known_color_idx, NUM_COLORS)
    rank_onehot = jax.nn.one_hot(own_known_rank_idx, NUM_RANKS)
    known_identity_onehot = (
        full_known_own.astype(jnp.float32)[:, None, None]
        * color_onehot[:, :, None]
        * rank_onehot[:, None, :]
    )  # (slot, color, rank)
    total_known = jnp.sum(known_identity_onehot, axis=0)  # (colors, ranks)
    other_known = total_known[None, :, :] - known_identity_onehot

    # The native V0 belief does not subtract cards visible in the partner's
    # hand, nor cards already exactly identified in one's other own slots,
    # from the remaining-count term (see spec). Correct that ourselves:
    # recover the literal-possibility mask implied by V0 (1 where V0>0, since
    # the uncorrected remaining count and the mask are the only two factors),
    # then reweight by counts that exclude those known-elsewhere cards.
    partner_counts = jnp.sum(partner_hand, axis=0)  # (colors, ranks)
    mask_own = (own_v0_raw > 1e-9).astype(jnp.float32)
    adjusted_remaining = jnp.clip(
        multiplicity[None, :] - discard_counts - played_mask - partner_counts,
        a_min=0.0,
    )
    adjusted_remaining_per_slot = jnp.clip(
        adjusted_remaining[None, :, :] - other_known, a_min=0.0
    )
    own_v0_adj = mask_own * adjusted_remaining_per_slot
    own_v0_sum = jnp.sum(own_v0_adj, axis=(1, 2), keepdims=True)
    own_v0 = jnp.where(own_v0_sum > 1e-9, own_v0_adj / jnp.maximum(own_v0_sum, 1e-9), own_v0_raw)

    prob_playable_own = jnp.sum(own_v0 * playable_mask[None, :, :], axis=(1, 2))
    prob_dead_own = jnp.sum(own_v0 * dead_mask[None, :, :], axis=(1, 2))
    prob_critical_own = jnp.sum(own_v0 * critical_mask[None, :, :], axis=(1, 2))

    slot_idx = jnp.arange(HAND_SIZE, dtype=jnp.float32)

    play_conf = (prob_playable_own >= PLAY_CERTAINTY) & occupied_own
    play_score = jnp.where(
        play_conf,
        1.0e6 + prob_playable_own * 10.0 - slot_idx * 1.0e-3,
        -1.0e6,
    )

    discard_score = jnp.where(
        occupied_own,
        100.0 + prob_dead_own * 20.0 - prob_critical_own * 10.0 - slot_idx * 1.0e-4,
        -1.0e6,
    )

    # --- Hint value computation (partner side is fully visible to us). ---
    partner_present = jnp.sum(partner_hand, axis=(1, 2)) > 1e-6
    actual_color = jnp.argmax(jnp.sum(partner_hand, axis=2), axis=1)
    actual_rank = jnp.argmax(jnp.sum(partner_hand, axis=1), axis=1)

    partner_color_flag = jnp.sum(partner_knowledge[:, 25:30], axis=1) > 0.5
    partner_rank_flag = jnp.sum(partner_knowledge[:, 30:35], axis=1) > 0.5

    is_playable_partner = partner_present & (
        actual_rank == next_rank_needed[actual_color]
    )

    def color_hint_value(c):
        matches = partner_present & (actual_color == c)
        new_info = matches & (~partner_color_flag)
        full_playable = new_info & partner_rank_flag & is_playable_partner
        partial_playable = new_info & (~partner_rank_flag) & is_playable_partner
        full_not_playable = new_info & partner_rank_flag & (~is_playable_partner)
        generic = new_info & (~partner_rank_flag) & (~is_playable_partner)
        return (
            100.0 * jnp.sum(full_playable)
            + 10.0 * jnp.sum(partial_playable)
            + 2.0 * jnp.sum(full_not_playable)
            + 1.0 * jnp.sum(generic)
        )

    def rank_hint_value(r):
        matches = partner_present & (actual_rank == r)
        new_info = matches & (~partner_rank_flag)
        full_playable = new_info & partner_color_flag & is_playable_partner
        partial_playable = new_info & (~partner_color_flag) & is_playable_partner
        full_not_playable = new_info & partner_color_flag & (~is_playable_partner)
        generic = new_info & (~partner_color_flag) & (~is_playable_partner)
        return (
            100.0 * jnp.sum(full_playable)
            + 10.0 * jnp.sum(partial_playable)
            + 2.0 * jnp.sum(full_not_playable)
            + 1.0 * jnp.sum(generic)
        )

    color_values = jax.vmap(color_hint_value)(jnp.arange(NUM_COLORS, dtype=jnp.int32))
    rank_values = jax.vmap(rank_hint_value)(jnp.arange(NUM_RANKS, dtype=jnp.int32))
    hint_values = jnp.concatenate([color_values, rank_values])

    threshold = jnp.where(info_tokens > 7.5, 1.0, 10.0)
    hint_score = jnp.where(hint_values >= threshold, 1000.0 + hint_values, -1.0e6)

    scores = jnp.concatenate(
        [
            discard_score,
            play_score,
            hint_score,
            jnp.asarray([-1.0e7], dtype=jnp.float32),
        ]
    )

    final_scores = jnp.where(legal, scores, -jnp.inf)
    action = jnp.argmax(final_scores).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)
    trace = {
        "info_tokens": info_tokens.astype(jnp.float32),
        "prob_playable_own": prob_playable_own,
        "prob_dead_own": prob_dead_own,
        "prob_critical_own": prob_critical_own,
        "hint_values": hint_values,
        "firework_total": jnp.sum(next_rank_needed).astype(jnp.float32),
    }
    return action, next_memory, trace
