"""A small, local-observation Hanabi policy."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_hint_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    x = obs.features
    legal = obs.legal_actions
    inactive = legal[20]

    fw = x[167:192].reshape((5, 5))
    heights = jnp.sum(fw * jnp.arange(1, 6, dtype=jnp.float32), axis=1)
    ranks = jnp.arange(5, dtype=jnp.float32)
    slots = jnp.arange(5, dtype=jnp.float32)
    playable_type = ranks[None, :] == heights[:, None]
    dead_type = ranks[None, :] < heights[:, None]

    # Native knowledge for the observer's own hand.
    own = x[308:483].reshape((5, 35))
    own_v0 = own[:, :25].reshape((5, 5, 5))
    own_color_hint = own[:, 25:30] > 0.5
    own_rank_hint = own[:, 30:35] > 0.5
    occupied = jnp.sum(own_v0, axis=(1, 2)) > 0.5
    own_play_mass = jnp.sum(own_v0 * playable_type[None, :, :], axis=(1, 2))
    own_dead_mass = jnp.sum(own_v0 * dead_type[None, :, :], axis=(1, 2))
    safe_play = occupied & (own_play_mass > 0.999)
    hinted = jnp.any(own_color_hint, axis=1) | jnp.any(own_rank_hint, axis=1)
    last = x[253:308]
    recent_hint = jnp.sum(last[2:4]) > 0.5
    recent_matches = last[18:23] > 0.5
    unique_recent = recent_hint & (jnp.sum(recent_matches) == 1.0)
    unique_hint_play = occupied & recent_matches & unique_recent & (own_play_mass > 0.001)

    # The partner hand and its native positive-hint flags are public. Complete
    # a playable card's existing clue: rank after color, or color after rank.
    partner = x[:125].reshape((5, 5, 5))
    partner_knowledge = x[483:658].reshape((5, 35))
    partner_color_hint = partner_knowledge[:, 25:30] > 0.5
    partner_rank_hint = partner_knowledge[:, 30:35] > 0.5
    partner_has_color = jnp.any(partner_color_hint, axis=1)
    partner_has_rank = jnp.any(partner_rank_hint, axis=1)
    partner_play = partner * playable_type[None, :, :]
    partner_useful = partner * (ranks[None, :] >= heights[:, None])
    no_existing_hint = ~(partner_has_color | partner_has_rank)
    need_rank = partner_play * partner_color_hint[:, :, None] * (~partner_has_rank)[:, None, None]
    need_color = partner_play * partner_rank_hint[:, None, :] * (~partner_has_color)[:, None, None]
    first_clue = partner_play * no_existing_hint[:, None, None]
    color_play_attr = jnp.sum(need_color, axis=(0, 2))
    rank_play_attr = jnp.sum(need_rank, axis=(0, 1))
    color_first_attr = jnp.sum(first_clue, axis=(0, 2))
    rank_first_attr = jnp.sum(first_clue, axis=(0, 1))
    color_useful_attr = jnp.sum(partner_useful, axis=(0, 2))
    rank_useful_attr = jnp.sum(partner_useful, axis=(0, 1))
    clue_play = jnp.concatenate((color_play_attr, rank_play_attr))
    clue_first = jnp.concatenate((color_first_attr, rank_first_attr))
    clue_useful = jnp.concatenate((color_useful_attr, rank_useful_attr))
    clue_count = jnp.concatenate((jnp.sum(partner, axis=(0, 2)),
                                  jnp.sum(partner, axis=(0, 1))))
    single_play = (clue_play > 0.0) & (clue_count == 1.0)
    single_first = (clue_first > 0.0) & (clue_count == 1.0)
    # Do not issue a singleton clue for merely useful (non-playable) cards:
    # the recipient reserves the singleton convention for play orders.
    useful_for_score = jnp.where(clue_count > 1.0, clue_useful, 0.0)
    clue_score = (single_play * 100000.0 + clue_play * 10000.0
                  + single_first * 5000.0 + clue_first * 5000.0
                  + useful_for_score * 2.0 - clue_count * 0.01)
    clue_score = jnp.where((clue_play + clue_first + useful_for_score) > 0.0,
                           clue_score, -1000.0)

    # Fixed action priorities: safe play, partner clue, known-dead discard,
    # and oldest unhinted discard.
    scores = jnp.full((21,), -1.0e6, dtype=jnp.float32)
    play_scores = jnp.where(safe_play, 1000.0 - slots, -1.0e6)
    play_scores = jnp.maximum(play_scores,
                              jnp.where(unique_hint_play, 900.0 - slots, -1.0e6))
    scores = scores.at[5:10].set(play_scores)
    scores = scores.at[10:20].set(500.0 + clue_score)
    uncertain_value = (100.0 + 150.0 * own_dead_mass
                       - 100.0 * own_play_mass - 0.1 * slots)
    discard_scores = jnp.where((own_dead_mass > 0.999) & ~hinted, 300.0 - slots,
                               jnp.where(~hinted, uncertain_value, 10.0 - slots))
    scores = scores.at[:5].set(discard_scores)
    scores = jnp.where(legal, scores, -1.0e6)
    chosen = jnp.argmax(scores).astype(jnp.int32)
    action = jnp.where(inactive, jnp.asarray(20, dtype=jnp.int32), chosen)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}
