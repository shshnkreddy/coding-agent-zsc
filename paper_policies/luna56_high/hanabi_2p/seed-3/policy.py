"""A conservative, locally observable Hanabi policy.

The policy treats the environment's native V0 array as a literal possibility
distribution. It plays only when every remaining possibility is playable, and
uses clues to create the same kind of certainty for the partner.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'certainty_first',
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


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions

    # Relative player 0 is the observer and relative player 1 is the partner.
    knowledge = f[308:658].reshape((2, 5, 35))
    own_v = knowledge[0, :, :25]
    partner_v = knowledge[1, :, :25]
    partner_hand = f[:125].reshape((5, 5, 5))

    colors = jnp.arange(25, dtype=jnp.int32) // 5
    ranks = jnp.arange(25, dtype=jnp.int32) % 5
    rank_values = jnp.arange(5, dtype=jnp.float32)

    fireworks = f[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks * (rank_values[None, :] + 1.0), axis=1)
    deck = jnp.sum(f[127:167])
    lives = jnp.sum(f[200:203])
    tokens = jnp.sum(f[192:200])

    # Probability (under literal V0 possibilities) that each own slot is
    # immediately playable. Empty slots have zero V0 mass.
    playable_identity = (rank_values[ranks] == heights[colors]).astype(jnp.float32)
    own_mass = jnp.sum(own_v, axis=1)
    own_play_mass = jnp.sum(own_v * playable_identity[None, :], axis=1)
    own_play_prob = own_play_mass / jnp.maximum(own_mass, 1.0e-6)
    own_safe = (own_mass > 0.0) & (own_play_prob >= 0.999)
    # In the final round, a high-confidence play is preferable to spending a
    # turn on a clue that cannot be followed by another draw. Keep a life in
    # reserve and reject slots that are already likely dead.
    late_threshold = jnp.where(lives >= 3.0, 0.65, 0.80)
    late_play = (
        (deck <= 2.0) & (lives >= 2.0) & (own_mass > 0.0)
        & (own_play_prob >= late_threshold)
    )
    zero_token_risk = (
        (tokens <= 0.0) & (lives >= 2.0) & (own_play_prob >= 0.75)
    )
    risky_allowed = (late_play | zero_token_risk) & ~jnp.any(own_safe)

    # A rank clue restricts each matching partner slot to that rank. Compute
    # its resulting play certainty without assuming anything about the hidden
    # partner hand beyond the public V0 support.
    rank_match = jnp.sum(partner_hand, axis=1)  # [slot, rank]
    color_match = jnp.sum(partner_hand, axis=2)  # [slot, color]
    rank_masks = (ranks[None, :] == jnp.arange(5, dtype=jnp.int32)[:, None])
    rank_mass = jnp.einsum('sp, rp->sr', partner_v, rank_masks.astype(jnp.float32))
    rank_play_mass = jnp.einsum(
        'sp,rp->sr', partner_v * playable_identity[None, :],
        rank_masks.astype(jnp.float32),
    )
    rank_certainty = rank_play_mass / jnp.maximum(rank_mass, 1.0e-6)

    color_masks = (colors[None, :] == jnp.arange(5, dtype=jnp.int32)[:, None])
    color_mass = jnp.einsum('sp,cp->sc', partner_v, color_masks.astype(jnp.float32))
    color_play_mass = jnp.einsum(
        'sp,cp->sc', partner_v * playable_identity[None, :],
        color_masks.astype(jnp.float32),
    )
    color_certainty = color_play_mass / jnp.maximum(color_mass, 1.0e-6)

    rank_target = rank_match > 0.0
    color_target = color_match > 0.0
    rank_safe_target = rank_target & (rank_certainty >= 0.999)
    color_safe_target = color_target & (color_certainty >= 0.999)

    # A safe clue is preferred, then clues with the greatest aggregate chance
    # of making a matching card playable. The -1 baseline lets a useful
    # discard win over a weak clue when discarding is legal, but still forces
    # a clue when the hand is full and discarding is unavailable.
    rank_safe_count = jnp.sum(rank_safe_target.astype(jnp.float32), axis=0)
    color_safe_count = jnp.sum(color_safe_target.astype(jnp.float32), axis=0)
    rank_expected = jnp.sum(rank_target.astype(jnp.float32) * rank_certainty, axis=0)
    color_expected = jnp.sum(color_target.astype(jnp.float32) * color_certainty, axis=0)
    rank_quality = 10.0 * rank_safe_count + rank_expected - 1.0
    color_quality = 10.0 * color_safe_count + color_expected - 1.0

    can_hint = tokens > 0.0
    rank_actions = 15 + jnp.arange(5, dtype=jnp.int32)
    color_actions = 10 + jnp.arange(5, dtype=jnp.int32)

    scores = jnp.full((21,), -1.0e9, dtype=jnp.float32)
    play_actions = 5 + jnp.arange(5, dtype=jnp.int32)
    play_scores = jnp.where(
        own_safe,
        1000.0 - jnp.arange(5, dtype=jnp.float32),
        jnp.where(risky_allowed, 500.0 - jnp.arange(5, dtype=jnp.float32), -1.0e9),
    )
    scores = scores.at[play_actions].set(jnp.where(legal[play_actions], play_scores, -1.0e9))

    hint_scores_rank = jnp.where(
        legal[rank_actions] & can_hint, rank_quality, -1.0e9
    )
    hint_scores_color = jnp.where(
        legal[color_actions] & can_hint, color_quality, -1.0e9
    )
    scores = scores.at[rank_actions].set(hint_scores_rank)
    scores = scores.at[color_actions].set(hint_scores_color)

    # Prefer discarding a slot with low play probability, breaking ties toward
    # the oldest slot and toward cards already likely to be dead.
    dead_identity = (ranks[None, :] < heights[colors]).astype(jnp.float32)
    own_dead_prob = jnp.sum(own_v * dead_identity, axis=1) / jnp.maximum(own_mass, 1.0e-6)
    discard_quality = -own_play_prob + 0.25 * own_dead_prob - 0.001 * jnp.arange(5)
    discard_actions = jnp.arange(5, dtype=jnp.int32)
    scores = scores.at[discard_actions].set(
        jnp.where(legal[discard_actions], discard_quality, -1.0e9)
    )

    # The inactive seat must still return its sole legal action. This does not
    # affect active decisions because action 20 is then masked out.
    scores = scores.at[20].set(jnp.where(legal[20], 0.0, scores[20]))

    # Inactive seats expose only action 20. For active seats this argmax is
    # over legal actions, and the score hierarchy gives safe plays priority.
    scores = jnp.where(legal, scores, -1.0e9)
    action = jnp.argmax(scores).astype(jnp.int32)
    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory + 1)
    return action, next_memory, {}
