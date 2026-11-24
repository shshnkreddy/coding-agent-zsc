"""Convention-free literal-knowledge Hanabi policy.

Every inference is grounded in publicly visible evidence: the native V0
knowledge masks (literal hint consequences), public card counts, the visible
partner hand and the board.  No private signalling scheme is used, so the
behaviour stays interpretable to an unknown partner.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

_MULT_LIST = (3.0, 2.0, 2.0, 2.0, 1.0)


def _ranks():
    return jnp.arange(5, dtype=jnp.float32)


def _slots():
    return jnp.arange(5, dtype=jnp.float32)


def _mult():
    return jnp.asarray(_MULT_LIST, dtype=jnp.float32)

# Hint-value weights.
_W_KNOWN_PLAYABLE = 10.0     # hint turns a partner card into a literal "playable"
_W_TOUCH_PLAYABLE = 3.0      # hint marks an untouched, genuinely playable card
_W_RISK = 0.5                # hint marks a non-playable card that stays ambiguous
_W_CLEAN = 0.0               # every card the hint marks is genuinely playable
_W_SAVE = 5.0                # hint protects a critical card on the partner's chop
_W_KNOWN_DEAD = 0.6          # hint turns a partner card into a literal "dead"
_W_INFO = 0.3                # residual literal information

# Play thresholds on the probability that the chosen card is playable.
_THRESH_LIVES3 = 0.8
_THRESH_LIVES2 = 0.9
_THRESH_LIVES1 = 0.999
_THRESH_ENDGAME_RELIEF = 0.2

# Evidence weighting for a card the partner has just pointed at.  Partners of
# unknown design overwhelmingly hint cards they want played, so a recent hint
# is treated as soft evidence of playability on top of the literal belief.
_HINT_BOOST = 8.0
_THRESH_HINT3 = 0.7
_THRESH_HINT2 = 0.8
_HINT_BOOST_STALE = 1.0
_SAVE_GUARD = 1.0
_END_CONF3 = 0.25
_END_CONF2 = 0.45

# Discard preference weights and the token-dependent bar a hint must clear.
_D_DEAD = 4.0
_D_CRIT = 8.0
_D_PLAY = 3.0
_D_FLAG = 0.25
_D_AGE = 0.15
_BAR_HI = 1.2
_BAR_MID = 1.8
_BAR_LO = 2.5
_BAR_CRIT = 0.0


def _attribute_tables():
    eye = jnp.eye(5, dtype=jnp.float32)
    color_a = jnp.tile(eye[:, :, None], (1, 1, 5))   # [hint, color, rank]
    rank_a = jnp.tile(eye[:, None, :], (1, 5, 1))
    return jnp.concatenate([color_a, rank_a], axis=0)  # [10, 5, 5]


def _board(obs):
    """Decode the public board state from the flat feature vector."""
    f = obs.features
    partner_hand = f[0:125].reshape(5, 5, 5)          # [slot, color, rank]
    deck = jnp.sum(f[127:167])
    fireworks = f[167:192].reshape(5, 5)
    heights = jnp.sum(fireworks * (_ranks() + 1.0)[None, :], axis=1)   # [color]
    tokens = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    d = f[203:253].reshape(5, 10)
    discarded = jnp.stack(
        [jnp.sum(d[:, 0:3], axis=1), jnp.sum(d[:, 3:5], axis=1),
         jnp.sum(d[:, 5:7], axis=1), jnp.sum(d[:, 7:9], axis=1), d[:, 9]],
        axis=1)                                        # [color, rank]
    last = f[253:308]
    knowledge = f[308:658].reshape(2, 5, 35)
    return partner_hand, deck, heights, tokens, lives, discarded, last, knowledge


def _card_tables(heights, discarded):
    """Per card-type public facts: remaining copies, playable, dead, critical."""
    played = (heights[:, None] > _ranks()[None, :]).astype(jnp.float32)
    remaining = jnp.maximum(_mult()[None, :] - discarded - played, 0.0)
    playable = (heights[:, None] == _ranks()[None, :]).astype(jnp.float32)
    # A rank is unreachable when some lower, unplayed rank of that color is gone.
    ok = ((remaining > 0.0) | (played > 0.0)).astype(jnp.float32)
    chain = jnp.concatenate(
        [jnp.ones((5, 1), dtype=jnp.float32), jnp.cumprod(ok, axis=1)[:, :4]], axis=1)
    dead = jnp.clip(played + (1.0 - chain), 0.0, 1.0)
    critical = ((remaining == 1.0) & (dead == 0.0)).astype(jnp.float32)
    return remaining, playable, dead, critical


def _own_beliefs(knowledge, partner_hand, remaining):
    """Per-slot distribution over card types for the acting player's own hand."""
    mask = (knowledge[0, :, 0:25].reshape(5, 5, 5) > 0.0).astype(jnp.float32)
    seen = jnp.sum(partner_hand, axis=0)                      # [color, rank]
    public_w = mask * remaining[None, :, :]
    refined = mask * jnp.maximum(remaining - seen, 0.0)[None, :, :]
    refined_total = jnp.sum(refined, axis=(1, 2))
    weights = jnp.where((refined_total > 0.0)[:, None, None], refined, public_w)
    total = jnp.sum(weights, axis=(1, 2))
    occupied = (total > 0.0).astype(jnp.float32)
    return weights, jnp.maximum(total, 1e-6), occupied


def _recent_hint(last):
    """Slots of this hand touched by the partner's hint on the previous turn."""
    by_partner = last[1]
    is_hint = last[4] + last[5]
    at_me = last[6]
    return last[18:23] * by_partner * is_hint * at_me


def _carry_evidence(last, pointed):
    """Shift stored evidence with our own removals, then add the newest hint."""
    removed_here = last[0] * (last[2] + last[3])          # we played or discarded
    slot = jnp.argmax(last[23:28]).astype(jnp.float32)
    shifted = jnp.concatenate([pointed[1:], jnp.zeros((1,), dtype=jnp.float32)])
    moved = jnp.where(_slots() < slot, pointed, shifted)
    pointed = jnp.where(removed_here > 0.0, moved, pointed)
    return jnp.clip(pointed + _recent_hint(last), 0.0, 1.0)


def _discard_preference(mask, remaining, playable, dead, critical, flagged,
                        occupied):
    """The discard ranking this policy applies, evaluated for any hand view."""
    weights = mask * remaining
    total = jnp.maximum(jnp.sum(weights, axis=-1).sum(axis=-1), 1e-6)
    p_dead = jnp.sum(weights * dead, axis=(-2, -1)) / total
    p_crit = jnp.sum(weights * critical, axis=(-2, -1)) / total
    p_play = jnp.sum(weights * playable, axis=(-2, -1)) / total
    score = (_D_DEAD * p_dead - _D_CRIT * p_crit - _D_PLAY * p_play
             - _D_FLAG * flagged + _D_AGE * (4.0 - _slots()))
    return jnp.where(occupied > 0.0, score, -1e9)


def _hint_values(obs, partner_hand, knowledge, playable, dead, critical,
                 remaining, occupied_p, flagged_p):
    """Score all ten hints by their literal usefulness to the partner."""
    pm = (knowledge[1, :, 0:25].reshape(5, 5, 5) > 0.0).astype(jnp.float32)
    touch = (jnp.einsum('scr,hcr->hs', partner_hand, _attribute_tables()) > 0.0).astype(jnp.float32)
    sel = jnp.where(touch[:, :, None, None] > 0.0, _attribute_tables()[:, None, :, :],
                    1.0 - _attribute_tables()[:, None, :, :])
    post = pm[None, :, :, :] * sel                              # [10, 5, 5, 5]

    def _all_of(mask4, prop):
        good = jnp.sum(mask4 * prop, axis=(-2, -1))
        bad = jnp.sum(mask4 * (1.0 - prop), axis=(-2, -1))
        return ((good > 0.0) & (bad == 0.0)).astype(jnp.float32)

    kp_after = _all_of(post, playable[None, None, :, :])
    kp_before = _all_of(pm, playable[None, :, :])
    kd_after = _all_of(post, dead[None, None, :, :])
    kd_before = _all_of(pm, dead[None, :, :])
    new_kp = kp_after * (1.0 - kp_before[None, :])
    new_kd = kd_after * (1.0 - kd_before[None, :])

    truly_playable = (jnp.sum(partner_hand * playable[None, :, :], axis=(1, 2)) > 0.0)
    truly_playable = truly_playable.astype(jnp.float32)
    is_critical = (jnp.sum(partner_hand * critical[None, :, :], axis=(1, 2)) > 0.0)
    is_critical = is_critical.astype(jnp.float32)
    knows_unplayable = (jnp.sum(post * playable[None, None, :, :], axis=(-2, -1)) == 0.0)
    risk = touch * occupied_p[None, :] * (1.0 - truly_playable[None, :]) * (
        1.0 - knows_unplayable.astype(jnp.float32))
    fresh_playable = touch * (1.0 - flagged_p[None, :]) * truly_playable[None, :]

    # Would the partner's next discard throw away a last copy, and does this
    # hint change that?  Both branches use the same discard ranking as we do.
    before = _discard_preference(pm, remaining[None, :, :], playable[None, :, :],
                                 dead[None, :, :], critical[None, :, :],
                                 flagged_p, occupied_p)
    flagged_after = jnp.clip(flagged_p[None, :] + touch, 0.0, 1.0)
    after = _discard_preference(post, remaining[None, None, :, :],
                                playable[None, None, :, :], dead[None, None, :, :],
                                critical[None, None, :, :], flagged_after,
                                occupied_p[None, :])
    loss_before = (jnp.sum(jax.nn.one_hot(jnp.argmax(before), 5) * is_critical)
                   * (1.0 - jnp.clip(jnp.sum(kp_before), 0.0, 1.0)))
    loss_after = (jnp.sum(jax.nn.one_hot(jnp.argmax(after, axis=1), 5)
                          * is_critical[None, :], axis=1)
                  * (1.0 - jnp.clip(jnp.sum(kp_after, axis=1), 0.0, 1.0)))
    save = loss_before - loss_after
    info = jnp.sum(jnp.sum(pm, axis=(1, 2))[None, :] - jnp.sum(post, axis=(2, 3)),
                   axis=1) / 25.0

    clean = ((jnp.sum(touch * occupied_p[None, :] * (1.0 - truly_playable[None, :]),
                      axis=1) == 0.0)
             & (jnp.sum(touch * truly_playable[None, :], axis=1) > 0.0))
    score = (_W_CLEAN * clean.astype(jnp.float32)
             + _W_KNOWN_PLAYABLE * jnp.sum(new_kp, axis=1)
             + _W_TOUCH_PLAYABLE * jnp.sum(fresh_playable, axis=1)
             - _W_RISK * jnp.sum(risk, axis=1)
             + _W_SAVE * save
             + _W_KNOWN_DEAD * jnp.sum(new_kd * occupied_p[None, :], axis=1)
             + _W_INFO * info)
    legal = obs.legal_actions[10:20].astype(jnp.float32)
    return jnp.where(legal > 0.0, score, -1e9)


def _decide(obs, stale):
    partner_hand, deck, heights, tokens, lives, discarded, last, knowledge = _board(obs)
    remaining, playable, dead, critical = _card_tables(heights, discarded)
    weights, total, occupied = _own_beliefs(knowledge, partner_hand, remaining)

    p_play = jnp.sum(weights * playable[None, :, :], axis=(1, 2)) / total
    p_dead = jnp.sum(weights * dead[None, :, :], axis=(1, 2)) / total
    p_crit = jnp.sum(weights * critical[None, :, :], axis=(1, 2)) / total
    exp_rank = jnp.sum(weights * _ranks()[None, None, :], axis=(1, 2)) / total
    flagged_self = (jnp.sum(knowledge[0, :, 25:35], axis=1) > 0.0).astype(jnp.float32)

    occupied_p = (jnp.sum(partner_hand, axis=(1, 2)) > 0.0).astype(jnp.float32)
    flagged_p = (jnp.sum(knowledge[1, :, 25:35], axis=1) > 0.0).astype(jnp.float32)

    legal = obs.legal_actions
    legal_discard = legal[0:5].astype(jnp.float32)
    legal_play = legal[5:10].astype(jnp.float32)

    # --- play ---------------------------------------------------------------
    threshold = jnp.where(lives >= 3.0, _THRESH_LIVES3,
                          jnp.where(lives >= 2.0, _THRESH_LIVES2, _THRESH_LIVES1))
    threshold = jnp.where((deck <= 0.0) & (lives >= 2.0),
                          threshold - _THRESH_ENDGAME_RELIEF, threshold)
    # Soft evidence from the partner's hints at this hand.
    fresh = _recent_hint(last)
    strength = jnp.maximum(fresh * _HINT_BOOST, stale * _HINT_BOOST_STALE)
    pointed = jnp.maximum(fresh, stale)
    factor = 1.0 + jnp.maximum(strength - 1.0, 0.0)[:, None, None] * playable[None, :, :]
    boosted = weights * factor
    p_play_hint = (jnp.sum(boosted * playable[None, :, :], axis=(1, 2))
                   / jnp.maximum(jnp.sum(boosted, axis=(1, 2)), 1e-6))
    hint_threshold = jnp.where(lives >= 3.0, _THRESH_HINT3,
                               jnp.where(lives >= 2.0, _THRESH_HINT2, 1.1))
    hint_threshold = jnp.where((deck <= 0.0) & (lives >= 2.0),
                               hint_threshold - _THRESH_ENDGAME_RELIEF, hint_threshold)
    # A partner may equally be pointing at a last copy to stop us discarding it,
    # so only accept the play reading when playability is the better fit.
    save_reading = p_crit * _SAVE_GUARD > p_play
    playable_now = ((p_play >= threshold)
                    | ((pointed > 0.0) & (p_play_hint >= hint_threshold)
                       & ~save_reading))
    confidence = jnp.maximum(p_play, jnp.where(pointed > 0.0, p_play_hint, 0.0))
    # Once the deck is gone only two turns remain, so an unplayed card is worth
    # nothing; gambling then costs only a life we can still spare.
    endgame = jnp.where(lives >= 3.0, _END_CONF3,
                        jnp.where(lives >= 2.0, _END_CONF2, 1.1))
    playable_now = playable_now | ((deck <= 0.0) & (confidence >= endgame))

    play_score = jnp.where(legal_play > 0.0,
                           confidence * 10.0 + 5.0 * playable_now - 0.05 * exp_rank,
                           -1e9)
    play_slot = jnp.argmax(play_score)
    play_ok = jnp.any((legal_play > 0.0) & playable_now)

    # --- discard ------------------------------------------------------------
    # Chop discipline: oldest slot first, protecting critical/playable cards.
    discard_score = (_D_DEAD * p_dead - _D_CRIT * p_crit - _D_PLAY * p_play
                     - _D_FLAG * flagged_self + _D_AGE * (4.0 - _slots()))
    discard_score = jnp.where(legal_discard > 0.0, discard_score, -1e9)
    discard_slot = jnp.argmax(discard_score)
    discard_any = jnp.any(legal_discard > 0.0)

    # --- hint ---------------------------------------------------------------
    hint_score = _hint_values(obs, partner_hand, knowledge, playable, dead,
                              critical, remaining, occupied_p, flagged_p)
    hint_index = jnp.argmax(hint_score)
    hint_any = jnp.any(hint_score > -1e8)
    bar = jnp.where(tokens >= 6.0, _BAR_HI, jnp.where(tokens >= 3.0, _BAR_MID, _BAR_LO))
    # Prefer speaking over throwing away a card that may be the last copy.
    bar = bar - _BAR_CRIT * p_crit[discard_slot]
    want_hint = hint_any & (hint_score[hint_index] > bar)

    action = jnp.where(discard_any, discard_slot, 5 + play_slot).astype(jnp.int32)
    action = jnp.where(want_hint | (~discard_any & hint_any),
                       (10 + hint_index).astype(jnp.int32), action)
    action = jnp.where(play_ok, (5 + play_slot).astype(jnp.int32), action)

    # Legality guard: fall back to the best legal action under a unified value.
    fallback_value = jnp.concatenate(
        [jnp.where(legal_discard > 0.0, discard_score, -1e9),
         jnp.where(legal_play > 0.0, p_play - 1.0, -1e9),
         hint_score,
         jnp.asarray([-1e9], dtype=jnp.float32)])
    fallback = jnp.argmax(fallback_value).astype(jnp.int32)
    action = jnp.where(legal[action], action, fallback)

    trace = {
        'dscore': discard_score.astype(jnp.float32),
        'pplay_v': p_play.astype(jnp.float32),
        'pcrit_v': p_crit.astype(jnp.float32),
        'pdead_v': p_dead.astype(jnp.float32),
        'hscore': hint_score.astype(jnp.float32),
        'p_play': p_play[play_slot].astype(jnp.float32),
        'hint_value': hint_score[hint_index].astype(jnp.float32),
        'tokens': tokens.astype(jnp.float32),
        'lives': lives.astype(jnp.float32),
        'deck': deck.astype(jnp.float32),
    }
    return action, trace


def init_agent():
    return {'turn': jnp.asarray(0, dtype=jnp.int32),
            'pointed': jnp.zeros((5,), dtype=jnp.float32)}


def agent_act(obs, reset, memory, rng):
    last = obs.features[253:308]
    previous = jnp.where(reset, jnp.zeros((5,), dtype=jnp.float32), memory['pointed'])
    pointed = _carry_evidence(last, previous)
    active = jnp.logical_not(obs.legal_actions[20])
    action, trace = _decide(obs, pointed)
    action = jnp.where(active, action, jnp.asarray(20, dtype=jnp.int32))
    turn = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['turn'] + 1)
    trace = dict(trace)
    trace['active'] = active.astype(jnp.float32)
    trace['pointed'] = pointed.astype(jnp.float32)
    next_memory = {'turn': turn.astype(jnp.int32),
                   'pointed': pointed.astype(jnp.float32)}
    return action.astype(jnp.int32), next_memory, trace
