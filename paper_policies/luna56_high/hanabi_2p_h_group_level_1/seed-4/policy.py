"""A conservative, observation-only H-group Level 1 Hanabi policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'level1_literal_safe', 'adapter': 'hanabi', 'adapter_version': 1,
    'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5,
    'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local',
}

_MULTIPLICITY_VALUES = (3.0, 2.0, 2.0, 2.0, 1.0)


def init_agent():
    return jnp.zeros((5,), dtype=jnp.int32)


def _highest_true(mask):
    slots = jnp.arange(5, dtype=jnp.int32)
    return jnp.max(jnp.where(mask, slots, jnp.asarray(-1, dtype=jnp.int32)))


def _lowest_true(mask):
    slots = jnp.arange(5, dtype=jnp.int32)
    return jnp.min(jnp.where(mask, slots, jnp.asarray(5, dtype=jnp.int32)))


def agent_act(obs, reset, memory, rng):
    del rng
    ranks = jnp.arange(5, dtype=jnp.int32)
    colors = jnp.arange(5, dtype=jnp.int32)
    card_rank = jnp.broadcast_to(ranks[None, :], (5, 5))
    card_color = jnp.broadcast_to(colors[:, None], (5, 5))
    multiplicity = jnp.asarray(_MULTIPLICITY_VALUES, dtype=jnp.float32)
    f = obs.features
    legal = obs.legal_actions

    fireworks = f[167:192].reshape(5, 5)
    # The one-hot row stores the highest completed rank; the next required
    # rank is therefore row index + 1 (or zero for an empty row).
    stack = jnp.sum(fireworks * (ranks[None, :] + 1).astype(jnp.float32), axis=1)
    info = jnp.sum(f[192:200])
    discards = f[203:253].reshape(5, 10)

    # Native knowledge is relative-player, native-slot ordered.
    knowledge = f[308:658].reshape(2, 5, 35)
    own_k, partner_k = knowledge[0], knowledge[1]
    own_v = own_k[:, :25].reshape(5, 5, 5)
    partner_v = partner_k[:, :25].reshape(5, 5, 5)
    own_occupied = jnp.sum(own_v, axis=(1, 2)) > 0.0
    partner_hand = f[:125].reshape(5, 5, 5)
    partner_occupied = jnp.sum(partner_hand, axis=(1, 2)) > 0.0
    own_clued = jnp.sum(own_k[:, 25:35], axis=1) > 0.0
    partner_clued = jnp.sum(partner_k[:, 25:35], axis=1) > 0.0
    previous_clued = jnp.where(reset, jnp.zeros((5,), dtype=jnp.bool_), memory > 0)
    own_chop = _lowest_true(own_occupied & ~own_clued)
    partner_chop = _lowest_true(partner_occupied & ~partner_clued)

    playable_card = (card_rank[None, :, :] == stack[:, None, None])

    # Reconstruct the Level 1 focus of a clue given to this seat.  A new or
    # re-touched non-chop focus is a play promise; a chop focus remains a
    # possible save and is not blind-played.
    last = f[253:308]
    last_is_hint = (last[4] > 0.5) | (last[5] > 0.5)
    hint_to_self = last_is_hint & (last[6] > 0.5)
    touched = (last[18:23] > 0.5) & own_occupied
    newly_touched = touched & ~previous_clued
    has_new = jnp.any(newly_touched)
    pre_chop = _lowest_true(own_occupied & ~previous_clued)
    chop_touched = (pre_chop < 5) & touched[jnp.minimum(pre_chop, 4)]
    focus_new = _highest_true(newly_touched)
    focus_retouch = _highest_true(touched)
    focus = jnp.where(has_new,
                      jnp.where(chop_touched, pre_chop, focus_new),
                      focus_retouch)
    promised_base = hint_to_self & (focus >= 0) & (focus < 5) & ~chop_touched
    focus_slot = jnp.minimum(jnp.maximum(focus, 0), 4)
    hint_color = jnp.argmax(last[8:13])
    hint_rank = jnp.argmax(last[13:18])
    hint_attr = jnp.where(last[4] > 0.5,
                          card_color == hint_color, card_rank == hint_rank)
    focused_possibilities = own_v[focus_slot] * hint_attr
    focused_mass = jnp.sum(focused_possibilities)
    focused_play_mass = jnp.sum(focused_possibilities * playable_card[0])
    literal_safe_focus = (focus >= 0) & (focus < 5) & (focused_mass > 0.0) & (
        focused_play_mass > 0.999 * focused_mass)
    focus_rank_masses = jnp.sum(focused_possibilities, axis=0)
    focus_rank = jnp.argmax(focus_rank_masses)
    focus_color = jnp.argmax(jnp.sum(focused_possibilities, axis=1))
    rank_offsets_local = jnp.asarray([0, 3, 5, 7, 9], dtype=jnp.int32)
    focus_discarded = discards[focus_color, rank_offsets_local[focus_rank]]
    focus_critical = focus_discarded >= multiplicity[focus_rank] - 1.0
    focus_save_like = jnp.where(
        last[5] > 0.5,
        (focus_rank == 4) | (focus_rank == 1) | focus_critical,
        focus_critical & (focus_rank != 4))
    chop_playable_focus = chop_touched & literal_safe_focus & ~focus_save_like
    # The giver-side fallback is restricted to chop focus, so a non-chop
    # focus is safe to interpret as the shared Level 1 play promise, including
    # delayed play clues from independent convention-following partners.
    promised = hint_to_self & (focus >= 0) & (focus < 5) & (
        (~chop_touched) | chop_playable_focus)
    promised_card = promised & (jnp.arange(5, dtype=jnp.int32) == focus)

    own_play_mass = jnp.sum(own_v * playable_card, axis=(1, 2))
    own_safe = own_occupied & (own_play_mass > 0.999)
    own_dead = own_occupied & (
        jnp.sum(own_v * (card_rank[None, :, :] < stack[:, None, None]), axis=(1, 2)) > 0.999)
    play_priority = jnp.where(own_safe, 1000.0 + jnp.arange(5),
                              jnp.where(promised_card, 950.0 + jnp.arange(5), -1.0e6))
    discard_priority = jnp.where(own_dead & own_clued,
                                 850.0 + jnp.arange(5), -1.0e6)

    # The clue giver sees the partner's exact hand.
    p_identity = partner_hand
    p_playable = jnp.sum(p_identity * playable_card, axis=(1, 2)) > 0.5
    p_card_color = card_color[None, :, :]
    p_card_rank = card_rank[None, :, :]
    p_mass = jnp.sum(partner_v, axis=(1, 2))

    def clue_quality(kind, value):
        if kind == 0:
            attr = (p_card_color == value)
            matches = jnp.sum(p_identity * attr, axis=(1, 2)) > 0.5
            action = 10 + value
        else:
            attr = (p_card_rank == value)
            matches = jnp.sum(p_identity * attr, axis=(1, 2)) > 0.5
            action = 15 + value
        new = matches & ~partner_clued
        updated = partner_v * attr
        updated_mass = jnp.sum(updated, axis=(1, 2))
        updated_play = jnp.sum(updated * playable_card, axis=(1, 2))
        updated_safe = (updated_mass > 0.0) & (updated_play > 0.999 * updated_mass)
        # H-group focus is chop-first.  This must match the recipient's
        # reconstruction exactly or a save clue can be mistaken for a play
        # clue by one side of self-play.
        chop_touched = (partner_chop < 5) & matches[jnp.minimum(partner_chop, 4)]
        focus = jnp.where(chop_touched, partner_chop, _highest_true(new))
        focus_safe = jnp.where(focus >= 0, updated_safe[jnp.maximum(focus, 0)], False)
        focus_playable = jnp.where(focus >= 0, p_playable[jnp.maximum(focus, 0)], False)
        focus_identity = p_identity[jnp.maximum(focus, 0)]
        exact_focus_rank = jnp.argmax(jnp.sum(focus_identity, axis=0))
        exact_focus_color = jnp.argmax(jnp.sum(focus_identity, axis=1))
        exact_discarded = discards[exact_focus_color, rank_offsets_local[exact_focus_rank]]
        exact_critical = exact_discarded >= multiplicity[exact_focus_rank] - 1.0
        exact_save_like = jnp.where(
            kind == 1,
            (exact_focus_rank == 4) | (exact_focus_rank == 1) | exact_critical,
            exact_critical & (exact_focus_rank != 4))
        focus_is_valid_play = focus_playable & (~chop_touched | ~exact_save_like)
        # A non-chop focus on a currently playable card is a valid Play Clue
        # even when the named attribute leaves several possible identities.
        # A chop-focused low-priority clue is retained only as a mechanics
        # fallback; ordinary chop saves still receive the higher save score.
        score = jnp.where(
            (info > 0.0) & (focus >= 0) & focus_is_valid_play,
            820.0 + 10.0 * jnp.sum(new),
            jnp.where((info > 0.0) & (jnp.sum(matches) > 0) & chop_touched,
                      30.0 + jnp.max(updated_play), -1.0e6))
        return action, score

    color_actions, color_scores, rank_actions, rank_scores = [], [], [], []
    for value in range(5):
        action, score = clue_quality(0, value)
        color_actions.append(action)
        color_scores.append(score)
        action, score = clue_quality(1, value)
        rank_actions.append(action)
        rank_scores.append(score)
    clue_actions = jnp.asarray(color_actions + rank_actions, dtype=jnp.int32)
    clue_scores = jnp.asarray(color_scores + rank_scores)

    # Save only a convention-defined chop 5, chop 2 with the other copy
    # discarded, or any other critical chop card.
    partner_chop_valid = partner_chop < 5
    chop_identity = partner_hand[jnp.maximum(partner_chop, 0)]
    chop_color = jnp.argmax(jnp.sum(chop_identity, axis=1))
    chop_rank = jnp.argmax(jnp.sum(chop_identity, axis=0))
    rank_offsets = jnp.asarray([0, 3, 5, 7, 9], dtype=jnp.int32)
    chop_discarded = discards[chop_color, rank_offsets[chop_rank]]
    chop_mult = multiplicity[chop_rank]
    save5 = partner_chop_valid & (chop_rank == 4)
    save2 = partner_chop_valid & (chop_rank == 1) & (chop_discarded >= chop_mult - 1.0)
    critical = partner_chop_valid & (chop_discarded >= chop_mult - 1.0)
    save_ok = save5 | save2 | critical
    save_action = 15 + chop_rank
    save_score = jnp.where((info > 0.0) & save_ok, 780.0, -1.0e6)

    scores = jnp.full((21,), -1.0e6)
    scores = scores.at[0:5].set(discard_priority)
    scores = scores.at[5:10].set(play_priority)
    scores = scores.at[clue_actions].set(clue_scores)
    scores = scores.at[save_action].set(jnp.maximum(scores[save_action], save_score))

    # Full tokens can make discards illegal.  With any token available, the
    # Level 1 default is to discard chop and regain a token; reserve a blind
    # play for the forced full-token case.
    fallback_play = jnp.where(info >= 8.0,
                               10.0 + own_play_mass + jnp.arange(5) * 0.001,
                               -1.0e6)
    scores = scores.at[5:10].set(jnp.maximum(scores[5:10], fallback_play))
    fallback_discard = jnp.where(info < 8.0,
                                 5.0 - jnp.arange(5) * 0.001, -1.0e6)
    scores = scores.at[0:5].set(jnp.maximum(scores[0:5], fallback_discard))
    chop_fallback = jnp.where(own_chop < 5, own_chop, 0)
    scores = scores.at[chop_fallback].set(jnp.maximum(scores[chop_fallback], 5.0))
    masked_scores = jnp.where(legal, scores, -1.0e7)
    chosen = jnp.argmax(masked_scores).astype(jnp.int32)
    chosen = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), chosen)
    next_memory = own_clued.astype(jnp.int32)
    trace = {
        'chosen': chosen,
        'stack_total': jnp.sum(stack),
        'info_tokens': info,
        'lives': jnp.sum(f[200:203]),
        'own_play_mass_max': jnp.max(own_play_mass),
        'own_safe_count': jnp.sum(own_safe.astype(jnp.int32)),
        'previous_clued_count': jnp.sum(previous_clued.astype(jnp.int32)),
        'touched_count': jnp.sum(touched.astype(jnp.int32)),
        'pre_chop': pre_chop,
        'focus': focus,
        'chop_touched': chop_touched.astype(jnp.int32),
        'promised_count': jnp.sum(promised_card.astype(jnp.int32)),
        'hint_to_self': hint_to_self.astype(jnp.int32),
    }
    return chosen, next_memory, trace
