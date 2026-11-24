"""A conservative, literal-information Hanabi policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'literal_safe_clue',
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
    del rng
    x = obs.features
    legal = obs.legal_actions
    lives = jnp.sum(x[200:203])

    colors = jnp.repeat(jnp.arange(5, dtype=jnp.int32), 5)
    ranks = jnp.tile(jnp.arange(5, dtype=jnp.int32), 5)

    fireworks = x[167:192].reshape((5, 5))
    height = jnp.sum(fireworks * (jnp.arange(5, dtype=jnp.float32) + 1.0), axis=1)
    playable_flat = (ranks == height[colors]) & (height[colors] < 5.0)
    dead_flat = (ranks < height[colors]) | (height[colors] >= 5.0)
    dead_flat = (ranks < height[colors]) | (height[colors] >= 5.0)

    # Native knowledge is [relative player, slot, 35], with the first 25
    # entries being the literal possibility weights.
    knowledge = x[308:658].reshape((2, 5, 35))[:, :, :25]
    own_poss = knowledge[0]
    partner_poss = knowledge[1]

    # Decode the partner's visible hand.  This is public information and lets
    # the clue sender predict the clue's public matches.
    partner_hand = x[0:125].reshape((5, 5, 5))
    partner_occ = jnp.sum(partner_hand, axis=(1, 2)) > 0.0
    partner_color = jnp.sum(partner_hand, axis=2) > 0.0
    partner_rank = jnp.sum(partner_hand, axis=1) > 0.0

    own_occ = jnp.sum(own_poss, axis=1) > 0.0
    own_bad = jnp.sum(own_poss * (~playable_flat)[None, :], axis=1)
    own_good = jnp.sum(own_poss * playable_flat[None, :], axis=1)
    own_dead = jnp.sum(own_poss * dead_flat[None, :], axis=1)
    own_live = jnp.sum(own_poss * (~dead_flat)[None, :], axis=1)
    own_total = jnp.sum(own_poss, axis=1)
    safe_play = own_occ & (own_bad <= 1.0e-6)
    safe_dead = own_occ & (own_live <= 1.0e-6)

    # For each of the ten possible clues, intersect the recipient's current
    # possibilities with the clue's positive/negative literal information.
    clue_ids = jnp.arange(10, dtype=jnp.int32)
    clue_is_color = clue_ids < 5
    clue_value = jnp.where(clue_is_color, clue_ids, clue_ids - 5)
    candidate_color = colors[None, :] == clue_value[:, None]
    candidate_rank = ranks[None, :] == clue_value[:, None]
    candidate_property = jnp.where(
        clue_is_color[:, None], candidate_color, candidate_rank
    )
    actual_match = jnp.where(
        clue_is_color[:, None],
        partner_color[:, clue_value].T,
        partner_rank[:, clue_value].T,
    )
    actual_match = actual_match & partner_occ[None, :]
    filter_mask = jnp.where(
        actual_match[:, :, None], candidate_property[:, None, :],
        ~candidate_property[:, None, :]
    )
    post_clue = partner_poss[None, :, :] * filter_mask
    post_occ = jnp.sum(post_clue, axis=2) > 0.0
    post_bad = jnp.sum(post_clue * (~playable_flat)[None, None, :], axis=2)
    post_safe_play = post_occ & (post_bad <= 1.0e-6)
    safe_count = jnp.sum(post_safe_play, axis=1)
    partner_playable = jnp.sum(
        partner_hand * playable_flat.reshape((5, 5))[None, :, :], axis=(1, 2)
    ) > 0.0
    target_playable = jnp.sum(actual_match & partner_playable[None, :], axis=1)

    # Prefer a clue that creates a safe play; otherwise prefer the clue that
    # removes the most literal possibilities and teaches the partner.
    partner_count = jnp.sum(partner_poss, axis=1)
    post_count = jnp.sum(post_clue, axis=2)
    information_gain = jnp.sum(partner_count[None, :] - post_count, axis=1)
    hint_score = (
        safe_count.astype(jnp.float32) * 1000.0
        + target_playable.astype(jnp.float32) * 100.0
        + information_gain
    )
    hint_choice = jnp.argmax(
        jnp.where(legal[10:20], hint_score, -1.0)
    ).astype(jnp.int32)
    hint_action = hint_choice + 10
    has_useful_hint = jnp.any(legal[10:20] & (safe_count > 0))
    has_any_hint = jnp.any(legal[10:20])

    play_score = jnp.where(
        safe_play & legal[5:10],
        10.0 - jnp.arange(5),
        -1.0,
    )
    play_choice = jnp.argmax(play_score).astype(jnp.int32) + 5
    has_safe_play = jnp.any(safe_play & legal[5:10])

    discard_dead_score = jnp.where(
        safe_dead & legal[0:5], 10.0 - jnp.arange(5), -1.0
    )
    discard_dead_choice = jnp.argmax(discard_dead_score).astype(jnp.int32)
    has_safe_dead = jnp.any(safe_dead & legal[0:5])

    # If no safe move or useful clue remains, prefer a likely-dead discard;
    # only then risk the most likely currently playable card.
    discard_score = jnp.where(legal[0:5], own_bad + own_dead, -1.0)
    discard_choice = jnp.argmax(discard_score).astype(jnp.int32)
    has_discard = jnp.any(legal[0:5])
    risky_play_score = jnp.where(
        legal[5:10], own_good / jnp.maximum(own_total, 1.0e-6), -1.0
    )
    risky_play_choice = jnp.argmax(risky_play_score).astype(jnp.int32) + 5
    has_confident_risky_play = (
        (jnp.max(risky_play_score) >= 0.6) & (lives > 1.0)
    )

    action = jnp.where(
        has_safe_play,
        play_choice,
        jnp.where(
            has_safe_dead,
            discard_dead_choice,
            jnp.where(
                has_useful_hint | has_any_hint,
                hint_action,
                jnp.where(
                    has_discard & ~has_confident_risky_play,
                    discard_choice,
                    risky_play_choice,
                ),
            ),
        ),
    ).astype(jnp.int32)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), action)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}
