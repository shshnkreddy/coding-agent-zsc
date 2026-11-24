"""A conservative, convention-free Hanabi policy.

The environment exposes the native literal possibility distribution for each
card.  This policy plays a card only when that distribution makes the current
firework card certain.  Otherwise it gives a legal hint that makes a partner
card as close to certain-playable as possible, and discards or guesses only
when communication cannot help.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_guided_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions

    # Firework heights are one-hot rows, with an all-zero row meaning zero.
    fireworks = f[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks * jnp.arange(1, 6, dtype=f.dtype), axis=1)
    target = jnp.minimum(heights, 4).astype(jnp.int32)
    target_valid = heights < 5

    # Native knowledge is normalized over the literal possibilities for each
    # occupied slot.  The first player-relative block is this seat's hand.
    knowledge = f[308:658].reshape((2, 5, 35))
    own_native = knowledge[0, :, :25].reshape((5, 5, 5))
    partner = knowledge[1, :, :25].reshape((5, 5, 5))
    partner_hand = f[:125].reshape((5, 5, 5))
    partner_present = jnp.sum(partner_hand, axis=(1, 2)) > 0.0

    colors = jnp.arange(5, dtype=jnp.int32)
    ranks = jnp.arange(5, dtype=jnp.int32)
    rank_grid = ranks[None, :]
    target_grid = target[:, None]
    target_cells = (rank_grid == target_grid) & target_valid[:, None]

    # Improve the native single-slot posterior with card identities that are
    # public but intentionally omitted by V0: visible partner cards, played
    # ranks, and discarded copies.  Preserve V0's literal hint mask as the
    # support, then renormalize using the corrected public multiplicities.
    discard_blocks = f[203:253].reshape((5, 10))
    rank_starts = jnp.asarray([0, 3, 5, 7, 9], dtype=jnp.int32)
    rank_lengths = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)
    discard_members = (
        jnp.arange(10)[None, :] >= rank_starts[:, None]
    ) & (
        jnp.arange(10)[None, :] < (rank_starts + rank_lengths)[:, None]
    )
    discard_counts = jnp.sum(
        discard_blocks[:, None, :] * discard_members[None, :, :], axis=2
    )
    multiplicities = jnp.asarray([3, 2, 2, 2, 1], dtype=f.dtype)[None, :]
    partner_counts = jnp.sum(partner_hand, axis=0)
    used_firework = (heights[:, None] > rank_grid).astype(f.dtype)
    remaining = multiplicities - discard_counts - used_firework - partner_counts
    own_support = (own_native > 0.0).astype(f.dtype)
    own_weights = own_support * jnp.maximum(remaining, 0.0)
    own_total = jnp.sum(own_weights, axis=(1, 2))
    own_initial = jnp.where(
        own_total[:, None, None] > 0.0,
        own_weights / jnp.maximum(own_total[:, None, None], 1.0e-6),
        own_native,
    )
    # A singleton slot consumes one copy of its forced type.  Propagate this
    # constraint through the other slots with fixed-shape JAX control flow.
    def propagate_singletons(_, state):
        singleton = jnp.sum(state > 0.0, axis=(1, 2)) == 1
        singleton_cards = jnp.where(
            singleton[:, None, None], state, 0.0
        )
        known_counts = jnp.sum(singleton_cards, axis=0)
        available = jnp.maximum(
            remaining[None, :, :] - known_counts[None, :, :]
            + singleton_cards,
            0.0,
        )
        weights = own_support * available
        totals = jnp.sum(weights, axis=(1, 2))
        return jnp.where(
            totals[:, None, None] > 0.0,
            weights / jnp.maximum(totals[:, None, None], 1.0e-6),
            state,
        )

    own = jax.lax.fori_loop(0, 5, propagate_singletons, own_initial)
    own_present = jnp.sum(own, axis=(1, 2)) > 0.0

    # Probability that each own slot is the immediately playable card.
    own_play_probability = jnp.sum(
        own * target_cells[None, :, :], axis=(1, 2)
    )
    safe_play = own_present & (own_play_probability > 0.80) & legal[5:10]
    safe_priority = jnp.where(safe_play, 5 - jnp.arange(5), -jnp.ones(5))
    safe_slot = jnp.argmax(safe_priority).astype(jnp.int32)
    play_action = 5 + safe_slot
    have_safe_play = jnp.any(safe_play)

    # Simulate the literal mask effect of each possible color/rank hint on the
    # partner's current native distribution.  The visible partner hand tells
    # us which slots receive the positive part of a hint.
    p_color = jnp.sum(partner_hand, axis=2) > 0.0  # [slot, color]
    p_rank = jnp.sum(partner_hand, axis=1) > 0.0   # [slot, rank]
    hint_color_eq = colors[None, None, :, None] == colors[:, None, None, None]
    hint_rank_eq = ranks[None, None, None, :] == ranks[:, None, None, None]

    # For each hinted color, [hint, slot, card-color, rank].
    color_match = jnp.transpose(p_color, (1, 0))[:, :, None, None]
    color_hint_mask = jnp.where(
        color_match,
        hint_color_eq,
        ~hint_color_eq,
    )
    color_hint_mask = jnp.broadcast_to(color_hint_mask, (5, 5, 5, 5))
    # For each hinted rank, [hint, slot, color, rank].
    rank_match = jnp.transpose(p_rank, (1, 0))[:, :, None, None]
    rank_hint_mask = jnp.where(
        rank_match,
        hint_rank_eq,
        ~hint_rank_eq,
    )
    rank_hint_mask = jnp.broadcast_to(rank_hint_mask, (5, 5, 5, 5))

    # Reorder to [hint, slot, color, rank] and normalize after masking.
    color_masked = partner[None, :, :, :] * color_hint_mask
    rank_masked = partner[None, :, :, :] * rank_hint_mask
    color_total = jnp.sum(color_masked, axis=(2, 3))
    rank_total = jnp.sum(rank_masked, axis=(2, 3))
    color_safe = jnp.sum(
        color_masked * target_cells[None, None, :, :], axis=(2, 3)
    )
    rank_safe = jnp.sum(
        rank_masked * target_cells[None, None, :, :], axis=(2, 3)
    )
    color_certainty = jnp.where(color_total > 0, color_safe / color_total, 0.0)
    rank_certainty = jnp.where(rank_total > 0, rank_safe / rank_total, 0.0)
    color_safe = jnp.where(color_total > 0, color_safe / color_total, 0.0)
    rank_safe = jnp.where(rank_total > 0, rank_safe / rank_total, 0.0)

    # Strongly prefer a hint which creates a certain playable card.  A weak
    # hint is retained only as a fallback while all information tokens are
    # already full, when discarding is unavailable and a hint is safer than a
    # blind play.
    color_slot_value = jnp.where(
        color_certainty > 0.999, 1.0 + color_safe, color_safe
    )
    rank_slot_value = jnp.where(
        rank_certainty > 0.999, 1.0 + rank_safe, rank_safe
    )
    color_best = jnp.max(
        jnp.where(partner_present[None, :], color_slot_value, 0.0), axis=1
    )
    rank_best = jnp.max(
        jnp.where(partner_present[None, :], rank_slot_value, 0.0), axis=1
    )
    color_certain = jnp.max(
        jnp.where(partner_present[None, :], color_certainty, 0.0), axis=1
    )
    rank_certain = jnp.max(
        jnp.where(partner_present[None, :], rank_certainty, 0.0), axis=1
    )
    color_score = color_best + 0.01 * color_certain
    rank_score = rank_best + 0.01 * rank_certain
    color_score = jnp.where(legal[10:15], color_score, -jnp.ones(5))
    rank_score = jnp.where(legal[15:20], rank_score, -jnp.ones(5))
    best_color = jnp.argmax(color_score).astype(jnp.int32)
    best_rank = jnp.argmax(rank_score).astype(jnp.int32)
    best_color_score = jnp.max(color_score)
    best_rank_score = jnp.max(rank_score)
    use_color = best_color_score >= best_rank_score
    best_hint_score = jnp.maximum(best_color_score, best_rank_score)
    hint_action = jnp.where(use_color, 10 + best_color, 15 + best_rank)
    strong_hint = jnp.maximum(
        jnp.max(jnp.where(legal[10:15], color_certain, 0.0)),
        jnp.max(jnp.where(legal[15:20], rank_certain, 0.0)),
    ) > 0.999
    useful_hint = strong_hint | (
        (~jnp.any(legal[:5])) & (best_hint_score > 0.001)
    ) | (best_hint_score > 0.35)

    # A discard is selected by low expected future value.  Cards at or below
    # a firework are dead; immediate and high-rank cards are more valuable.
    value_by_type = jnp.where(
        target_valid[:, None] & (rank_grid >= target_grid),
        1.0 + 0.15 * (rank_grid - target_grid),
        0.0,
    )
    own_value = jnp.sum(own * value_by_type[None, :, :], axis=(1, 2))
    discard_priority = jnp.where(
        legal[:5] & own_present,
        -own_value + 0.0001 * (4 - jnp.arange(5)),
        -jnp.ones(5) * 1.0e6,
    )
    discard_slot = jnp.argmax(discard_priority).astype(jnp.int32)
    discard_action = discard_slot
    have_discard = jnp.any(legal[:5])

    # With no information token, a guess is preferable to an illegal discard.
    guess_priority = jnp.where(legal[5:10], own_play_probability, -jnp.ones(5))
    guess_slot = jnp.argmax(guess_priority).astype(jnp.int32)
    guess_action = 5 + guess_slot
    active_action = jnp.where(
        have_safe_play,
        play_action,
        jnp.where(
            useful_hint,
            hint_action,
            jnp.where(have_discard, discard_action, guess_action),
        ),
    )
    inactive = legal[20]
    action = jnp.where(inactive, jnp.asarray(20, dtype=jnp.int32), active_action)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}
