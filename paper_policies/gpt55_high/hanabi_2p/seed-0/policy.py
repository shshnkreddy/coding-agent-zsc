"""Literal-information Hanabi policy for the policy-search task."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NEG = -1.0e9


def init_agent():
    return jnp.zeros((6,), dtype=jnp.int32)


def _first_best(scores, mask):
    masked = jnp.where(mask, scores, NEG)
    return jnp.argmax(masked).astype(jnp.int32), jnp.max(masked)


def _decode(obs):
    f = obs.features
    partner = f[0:125].reshape((5, 5, 5))
    deck = jnp.sum(f[127:167])
    fireworks_oh = f[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks_oh * jnp.arange(1, 6, dtype=jnp.float32), axis=1)
    info = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    discards = f[203:253].reshape((5, 10))
    knowledge = f[308:658].reshape((2, 5, 35))
    return partner, deck, heights, info, lives, discards, knowledge


def _discard_counts_by_rank(discards):
    return jnp.stack(
        (
            jnp.sum(discards[:, 0:3], axis=1),
            jnp.sum(discards[:, 3:5], axis=1),
            jnp.sum(discards[:, 5:7], axis=1),
            jnp.sum(discards[:, 7:9], axis=1),
            discards[:, 9],
        ),
        axis=1,
    )


def agent_act(obs, reset, memory, rng):
    del rng
    legal = obs.legal_actions
    partner, deck, heights, info, lives, discards, knowledge = _decode(obs)
    f = obs.features
    last_actor = f[253:255]
    last_type = f[255:259]
    last_recipient = f[259:261]
    last_matches = f[271:276] > 0.5
    last_removed = f[276:281]

    reset_memory = jnp.zeros((6,), dtype=jnp.int32)
    base_memory = jnp.where(reset, reset_memory, memory)
    prompt_flags = base_memory[0:5]
    turn_count = base_memory[5]

    removed_idx = jnp.argmax(last_removed).astype(jnp.int32)
    idx = jnp.arange(5, dtype=jnp.int32)
    source_idx = jnp.where(idx < removed_idx, idx, idx + 1)
    shifted_prompts = jnp.where(source_idx < 5, prompt_flags[source_idx], 0)
    own_removed = (last_actor[0] > 0.5) & ((last_type[0] > 0.5) | (last_type[1] > 0.5))
    prompt_flags = jnp.where(own_removed, shifted_prompts, prompt_flags)

    clue_to_self = (
        (last_actor[1] > 0.5)
        & ((last_type[2] > 0.5) | (last_type[3] > 0.5))
        & (last_recipient[0] > 0.5)
    )
    clue_strength = jnp.where(last_type[3] > 0.5, 2, 1).astype(jnp.int32)
    clue_prompts = last_matches.astype(jnp.int32) * clue_strength
    prompt_flags = jnp.where(clue_to_self, jnp.maximum(prompt_flags, clue_prompts), prompt_flags)

    colors = jnp.arange(5, dtype=jnp.float32)[:, None]
    ranks = jnp.arange(5, dtype=jnp.float32)[None, :]
    del colors
    playable_type = ranks == heights[:, None]
    dead_type = ranks < heights[:, None]
    future_type = ranks == (heights[:, None] + 1.0)

    self_v0 = knowledge[0, :, 0:25].reshape((5, 5, 5))
    self_color_hints = knowledge[0, :, 25:30]
    self_rank_hints = knowledge[0, :, 30:35]
    self_occ = jnp.sum(self_v0, axis=(1, 2)) > 0.0
    own_play_prob = jnp.sum(self_v0 * playable_type[None, :, :], axis=(1, 2))
    own_dead_prob = jnp.sum(self_v0 * dead_type[None, :, :], axis=(1, 2))
    own_rank5_prob = jnp.sum(self_v0[:, :, 4], axis=1)
    touched = (jnp.sum(self_color_hints, axis=1) + jnp.sum(self_rank_hints, axis=1)) > 0.0

    multiplicity = jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0], dtype=jnp.float32)
    discarded_by_rank = _discard_counts_by_rank(discards)
    played = (ranks < heights[:, None]).astype(jnp.float32)
    remaining = multiplicity[None, :] - discarded_by_rank - played
    last_needed = (remaining <= 1.01) & (ranks >= heights[:, None])
    own_last_needed_prob = jnp.sum(self_v0 * last_needed[None, :, :], axis=(1, 2))

    play_slot, play_value = _first_best(
        own_play_prob + (4.0 - jnp.arange(5, dtype=jnp.float32)) * 0.001,
        legal[5:10] & self_occ & (own_play_prob > 0.999),
    )
    play_action = play_slot + 5

    rank_prompt = prompt_flags >= 2
    color_prompt = prompt_flags == 1
    prompt_threshold = jnp.where(rank_prompt, 0.12, 0.45)
    prompted_slot, prompted_value = _first_best(
        own_play_prob
        + prompt_flags.astype(jnp.float32) * 0.10
        + (4.0 - jnp.arange(5, dtype=jnp.float32)) * 0.001,
        legal[5:10]
        & self_occ
        & (prompt_flags > 0)
        & (own_play_prob >= prompt_threshold)
        & (lives > 1.0),
    )
    prompted_action = prompted_slot + 5

    partner_occ = jnp.sum(partner, axis=(1, 2)) > 0.0
    partner_playable = jnp.sum(partner * playable_type[None, :, :], axis=(1, 2)) > 0.0
    partner_future = jnp.sum(partner * future_type[None, :, :], axis=(1, 2)) > 0.0
    partner_v0 = knowledge[1, :, 0:25].reshape((5, 5, 5))
    partner_touched = jnp.sum(knowledge[1, :, 25:35], axis=1) > 0.0
    partner_known_play = jnp.sum(partner_v0 * playable_type[None, :, :], axis=(1, 2)) > 0.999
    partner_useful = partner_playable & (~partner_known_play)

    slot_color = jnp.sum(partner, axis=2).T > 0.0
    slot_rank = jnp.sum(partner, axis=1).T > 0.0
    known_color_attr = knowledge[1, :, 25:30].T > 0.0
    known_rank_attr = knowledge[1, :, 30:35].T > 0.0

    def _hint_scores(attr_slots, known_attr):
        matched = attr_slots
        repeated_literal = matched & known_attr
        total = jnp.sum(matched, axis=1)
        fresh_playable = jnp.sum(matched & partner_useful[None, :] & (~repeated_literal), axis=1)
        repeated_playable = jnp.sum(matched & partner_useful[None, :] & repeated_literal, axis=1)
        playable = fresh_playable + repeated_playable
        already_playable = jnp.sum(matched & partner_known_play[None, :], axis=1)
        future = jnp.sum(matched & partner_future[None, :], axis=1)
        repeated = jnp.sum(repeated_literal, axis=1)
        bad = total - playable - already_playable - future
        clean = (fresh_playable > 0.0) & (bad <= 0.0)
        useful = (fresh_playable > 0.0) | ((future > 0.0) & (info >= 7.0))
        scores = (
            clean.astype(jnp.float32) * 30.0
            + fresh_playable * 8.0
            + repeated_playable * 1.0
            + future * 2.0
            - bad * 7.0
            - repeated * 8.0
            - total * 0.05
        )
        return scores, clean, useful

    color_scores, color_clean, color_useful = _hint_scores(slot_color, known_color_attr)
    rank_scores, rank_clean, rank_useful = _hint_scores(slot_rank, known_rank_attr)
    hint_scores = jnp.concatenate([color_scores, rank_scores + 0.25])
    hint_clean = jnp.concatenate([color_clean, rank_clean])
    hint_useful = jnp.concatenate([color_useful, rank_useful])
    hint_legal = legal[10:20] & (info > 0.0)
    hint_idx, hint_value = _first_best(hint_scores, hint_legal & hint_useful)
    clean_hint_idx, clean_hint_value = _first_best(hint_scores, hint_legal & hint_clean)
    pressure_hint_idx, pressure_hint_value = _first_best(hint_scores, hint_legal)
    pressure_color_idx, pressure_color_value = _first_best(color_scores, hint_legal[0:5])
    hint_action = hint_idx + 10
    clean_hint_action = clean_hint_idx + 10
    pressure_hint_action = jnp.where(
        pressure_color_value > -1.0e8,
        pressure_color_idx + 10,
        pressure_hint_idx + 10,
    )
    pressure_hint_value = jnp.maximum(pressure_hint_value, pressure_color_value)

    discard_scores = (
        own_dead_prob * 80.0
        - own_play_prob * 25.0
        - own_rank5_prob * 12.0
        - own_last_needed_prob * 18.0
        - touched.astype(jnp.float32) * 4.0
        - prompt_flags.astype(jnp.float32) * 16.0
        + (4.0 - jnp.arange(5, dtype=jnp.float32)) * 0.01
    )
    safe_discard_slot, safe_discard_value = _first_best(
        discard_scores, legal[0:5] & self_occ & (own_dead_prob > 0.999)
    )
    discard_slot, discard_value = _first_best(discard_scores, legal[0:5] & self_occ)
    safe_discard_action = safe_discard_slot
    discard_action = discard_slot

    risky_slot, risky_value = _first_best(
        own_play_prob + lives * 0.01,
        legal[5:10] & self_occ & (own_play_prob > 0.88) & (lives > 1.0),
    )
    risky_action = risky_slot + 5

    fallback = jnp.argmax(legal).astype(jnp.int32)
    active = ~legal[20]
    has_play = play_value > -1.0e8
    has_prompted = prompted_value > -1.0e8
    has_clean_hint = clean_hint_value > -1.0e8
    has_safe_discard = safe_discard_value > -1.0e8
    has_hint = hint_value > -1.0e8
    has_pressure_hint = pressure_hint_value > -1.0e8
    has_risky = risky_value > -1.0e8
    has_discard = discard_value > -1.0e8

    active_action = fallback
    active_action = jnp.where(has_discard, discard_action, active_action)
    active_action = jnp.where(has_risky, risky_action, active_action)
    active_action = jnp.where(
        has_pressure_hint & ((info >= 7.0) | (~has_discard)),
        pressure_hint_action,
        active_action,
    )
    active_action = jnp.where(has_hint & ((info >= 7.0) | (deck <= 4.0)), hint_action, active_action)
    active_action = jnp.where(has_safe_discard, safe_discard_action, active_action)
    active_action = jnp.where(has_clean_hint, clean_hint_action, active_action)
    active_action = jnp.where(has_prompted, prompted_action, active_action)
    active_action = jnp.where(has_play, play_action, active_action)
    action = jnp.where(~active, jnp.asarray(20, dtype=jnp.int32), active_action).astype(jnp.int32)

    mode = jnp.where(
        ~active,
        0,
        jnp.where(
            has_play,
            1,
            jnp.where(
                has_prompted,
                5,
                jnp.where(
                    has_clean_hint,
                    2,
                    jnp.where(
                        has_safe_discard,
                        3,
                        jnp.where(has_pressure_hint & ((info >= 7.0) | (~has_discard)), 6, 4),
                    ),
                ),
            ),
        ),
    )
    next_prompt_flags = jnp.where(action == prompted_action, prompt_flags.at[prompted_slot].set(0), prompt_flags)
    next_memory = jnp.concatenate(
        [
            next_prompt_flags.astype(jnp.int32),
            jnp.asarray([turn_count + active.astype(jnp.int32)], dtype=jnp.int32),
        ]
    )
    next_memory = jnp.where(reset, reset_memory, next_memory)
    trace = {
        'mode': mode.astype(jnp.int32),
        'own_play_p_max': jnp.max(own_play_prob),
        'prompt_max': jnp.max(prompt_flags),
        'hint_score': jnp.maximum(jnp.maximum(hint_value, clean_hint_value), pressure_hint_value),
        'discard_score': discard_value,
        'info': info,
    }
    return action, next_memory, trace
