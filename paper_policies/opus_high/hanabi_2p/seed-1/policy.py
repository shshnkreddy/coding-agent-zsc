"""Two-player Hanabi policy built on grounded literal beliefs.

Every inference comes from public information: the native possibility mask in
the observation, public discard/firework counts, and the partner hand that is
directly visible.  Hints are chosen by simulating the literal posterior they
create in the receiver, so they are informative to any partner rather than to
one that shares a private code.  Received hints only lower a risk threshold;
a card is never played unless the public evidence itself makes it plausible.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_belief_hanabi', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

_NC = 5
_NR = 5
_HAND = 5
_EPS = 1e-6

# Copies of each rank id (human ranks 1..5) inside one color.
_RANK_MULT_LIST = [3.0, 2.0, 2.0, 2.0, 1.0]

# Maps the 10 per-color discard thermometer entries onto their rank id.
_DISCARD_W_LIST = [[1.0, 0.0, 0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0]]

# Playability probability a card must reach before it is played on speculation.
_T3 = 0.90          # untouched card, three lives
_T2 = 0.95          # untouched card, two lives
_TP3 = 0.18         # card the latest hint just touched, three lives
_TP2 = 0.3         # card a partner hint has pointed at, two lives
_TO3 = 0.88         # card an older hint touched, three lives
_TO2 = 0.95         # card an older hint touched, two lives
_ENDGAME_RELIEF = 0.12

# Action tier constants.
_PLAY_CERTAIN = 1000.0
_PLAY_FRESH = 860.0
_PLAY_POINTED = 800.0
_PLAY_SPEC = 600.0
_HINT_BASE = 560.0
_HINT_SCALE = 160.0

# Hint value weights, in units of "one extra card played".
_W_BOMB = 1.3       # penalty when the hint would trigger an unplayable play
_W_CLEAN = 0.30     # every touched card is playable: safe under any reading
_W_DIRTY = 0.16     # per newly touched card that is not playable
_W_CERT = 0.35      # per partner slot that becomes certainly playable
_W_DEAD = 0.22      # per partner slot that becomes certainly discardable
_W_SAVE = 0.55      # protects a last copy sitting on the partner's discard slot
_W_INFO = 0.006     # per eliminated card possibility
_CRIT_W = 1.5       # discard cost multiplier for a last surviving copy
_PAIR_W = 0.6       # discard cost when two copies remain
_DISC_W = 12.0      # how strongly risk ranks one slot against another
_ABS_W = 12.0       # how strongly the safest discard argues against discarding
_RISK_W = 12.0      # how hard the cost of burning a card pushes against playing
_RECV_T = 0.08      # assumed partner bar for a card a hint points at
_RECV_T_OTHER = 0.85  # bar for a card no hint has touched


def _constants():
    """Fixed lookup tables, rebuilt inside the traced function."""
    rank_mult = jnp.asarray(_RANK_MULT_LIST, dtype=jnp.float32)
    discard_w = jnp.asarray(_DISCARD_W_LIST, dtype=jnp.float32)
    colors = jnp.arange(_NC, dtype=jnp.float32)
    ranks = jnp.arange(_NR, dtype=jnp.float32)
    color_idx = colors[:, None] * jnp.ones((1, _NR), jnp.float32)
    rank_idx = jnp.ones((_NC, 1), jnp.float32) * ranks[None, :]
    attr = jnp.concatenate(
        [(color_idx[None, :, :] == colors[:, None, None]),
         (rank_idx[None, :, :] == ranks[:, None, None])], axis=0)
    return rank_mult, discard_w, rank_idx, attr


def init_agent():
    return {'pointed': jnp.zeros((_HAND,), dtype=jnp.float32)}


def _public_tables(features, rank_mult, discard_w, rank_idx):
    """Public per-card-type tables, each shaped [5 colors, 5 ranks]."""
    fireworks = features[167:192].reshape(_NC, _NR)
    height = jnp.sum(fireworks * (jnp.arange(_NR, dtype=jnp.float32)[None, :] + 1.0), axis=1)
    played = (rank_idx < height[:, None]).astype(jnp.float32)
    discarded = features[203:253].reshape(_NC, 10) @ discard_w
    remaining = jnp.maximum(rank_mult[None, :] - discarded - played, 0.0)

    playable = (rank_idx == height[:, None])
    # Useless: already on the firework, or cut off by a fully discarded lower rank.
    alive = jnp.where(rank_idx >= height[:, None], remaining > 0.0, True)
    cum = jnp.cumprod(alive.astype(jnp.float32), axis=1)
    reachable = jnp.concatenate([jnp.ones((_NC, 1), jnp.float32), cum[:, :-1]], axis=1) > 0.5
    useless = (rank_idx < height[:, None]) | (~reachable)
    critical = (remaining <= 1.0) & (remaining > 0.0) & (~useless)
    return height, remaining, playable, useless, critical


def _shift_out(vector, slot):
    """Remove entry ``slot`` and shift later entries left, appending a zero."""
    idx = jnp.arange(_HAND, dtype=jnp.int32)
    src = jnp.where(idx < slot, idx, idx + 1)
    padded = jnp.concatenate([vector, jnp.zeros((1,), vector.dtype)])
    return padded[src]


def _update_pointed(pointed, features, reset):
    """Track which own slots a partner hint has singled out.

    Each transition shows exactly one past action to this seat: the partner's
    action on our active turns, and our own action on our idle turns.  That is
    enough to keep the marker aligned with the shifting hand.
    """
    actor_me = features[253] > 0.5
    actor_partner = features[254] > 0.5
    is_play = features[255] > 0.5
    is_discard = features[256] > 0.5
    is_hint = (features[257] + features[258]) > 0.5
    to_me = features[259] > 0.5

    removed_slot = jnp.argmax(features[276:281]).astype(jnp.int32)
    shifted = _shift_out(pointed, removed_slot)
    pointed = jnp.where(actor_me & (is_play | is_discard), shifted, pointed)

    touched = (features[271:276] > 0.5).astype(jnp.float32)
    pointed = jnp.where(actor_partner & is_hint & to_me,
                        jnp.maximum(pointed, touched), pointed)
    fresh = (actor_partner & is_hint & to_me) * touched
    return jnp.where(reset, jnp.zeros((_HAND,), jnp.float32), pointed), fresh


def agent_act(obs, reset, memory, rng):
    features = obs.features
    legal = obs.legal_actions
    ones = jnp.ones((), jnp.float32)

    rank_mult, discard_w, rank_idx, attr = _constants()
    height, remaining, playable, useless, critical = _public_tables(
        features, rank_mult, discard_w, rank_idx)
    playable_f = playable.astype(jnp.float32)
    useless_f = useless.astype(jnp.float32)
    tokens = jnp.sum(features[192:200])
    lives = jnp.sum(features[200:203])
    deck = jnp.sum(features[127:167])
    endgame = deck < 0.5

    pointed, fresh = _update_pointed(memory['pointed'], features, reset)

    partner_hand = features[0:125].reshape(_HAND, _NC, _NR)
    partner_counts = jnp.sum(partner_hand, axis=0)
    remaining_self = jnp.maximum(remaining - partner_counts, 0.0)

    knowledge = features[308:658].reshape(2, _HAND, 35)
    own_v0 = knowledge[0, :, 0:25].reshape(_HAND, _NC, _NR)
    partner_v0 = knowledge[1, :, 0:25].reshape(_HAND, _NC, _NR)
    own_flag_count = jnp.sum(knowledge[0, :, 25:35], axis=1)
    partner_flags = knowledge[1, :, 25:35].T  # [10 hint actions, 5 slots]

    # ---- own hand ------------------------------------------------------------
    own_mask = own_v0 > 0.0
    own_occ = jnp.any(own_mask, axis=(1, 2))
    weight = own_mask.astype(jnp.float32) * remaining_self[None]
    total = jnp.sum(weight, axis=(1, 2), keepdims=True)
    belief = jnp.where(total > _EPS, weight / jnp.maximum(total, _EPS), own_v0)

    p_play = jnp.sum(belief * playable_f[None], axis=(1, 2))
    p_useless = jnp.sum(belief * useless_f[None], axis=(1, 2))
    certain_play = jnp.all(~(own_mask & (~playable[None])), axis=(1, 2)) & own_occ
    exp_rank = jnp.sum(belief * rank_idx[None], axis=(1, 2))

    # Losing the last copy of a card caps its color forever, so weigh that by
    # the number of points it would cost.
    discard_cost_tbl = jnp.where(
        useless, 0.0,
        jnp.where(remaining <= 1.0, _CRIT_W * (5.0 - rank_idx),
                  jnp.where(remaining <= 2.0, _PAIR_W, 0.2)))
    discard_cost = jnp.sum(belief * discard_cost_tbl[None], axis=(1, 2))
    # A failed speculative play also destroys the card it burns.
    bad_cost = jnp.sum(belief * (discard_cost_tbl * (1.0 - playable_f))[None], axis=(1, 2))

    # A positive hint flag is exactly the public record of "a hint touched this
    # slot", so it needs no private bookkeeping to stay aligned with the hand.
    is_pointed = own_flag_count > 0.0
    plain_t = jnp.where(lives >= 2.5, _T3, jnp.where(lives >= 1.5, _T2, 2.0))
    point_t = jnp.where(lives >= 2.5, _TP3, jnp.where(lives >= 1.5, _TP2, 2.0))
    stale_t = jnp.where(lives >= 2.5, _TO3, jnp.where(lives >= 1.5, _TO2, 2.0))
    relief = jnp.where(endgame & (lives >= 1.5), _ENDGAME_RELIEF, 0.0)
    is_fresh = fresh > 0.5
    threshold = jnp.where(is_fresh, point_t,
                          jnp.where(is_pointed, stale_t, plain_t)) - relief

    # A hint singles out a card most sharply when it has just arrived, so among
    # the slots it touched prefer the newest.  That tie-break costs nothing when
    # the partner meant nothing by it, because the literal threshold still binds.
    slot_order = jnp.arange(_HAND, dtype=jnp.float32)
    play_score = jnp.where(
        certain_play,
        _PLAY_CERTAIN + 0.5 * (4.0 - exp_rank),
        jnp.where(
            (p_play >= threshold) & is_fresh,
            _PLAY_FRESH + 4.0 * slot_order + 20.0 * p_play,
            jnp.where(
                p_play >= threshold,
                jnp.where(is_pointed, _PLAY_POINTED, _PLAY_SPEC) + 100.0 * p_play,
                50.0 + 50.0 * p_play)))
    play_score = play_score - jnp.where(certain_play, 0.0, _RISK_W * bad_cost)

    discard_base = jnp.where(tokens <= 1.5, 780.0,
                             jnp.where(tokens <= 2.5, 700.0,
                                       jnp.where(tokens <= 4.5, 620.0, 470.0)))
    discard_base = discard_base - jnp.where(endgame, 120.0, 0.0)
    # Choosing between slots should be dominated by risk; choosing whether to
    # discard at all should not be, or the hand simply clogs up.
    safest = jnp.min(jnp.where(own_occ, discard_cost, 1e3))
    discard_score = (discard_base
                     - _ABS_W * safest
                     - _DISC_W * (discard_cost - safest)
                     + jnp.where(p_useless > 0.999, 30.0, 0.0)
                     - 6.0 * jnp.minimum(own_flag_count, 2.0)
                     - 8.0 * pointed
                     - 0.3 * jnp.arange(_HAND, dtype=jnp.float32))

    # ---- partner hand --------------------------------------------------------
    partner_occ = jnp.sum(partner_hand, axis=(1, 2)) > 0.5
    partner_occ_f = partner_occ.astype(jnp.float32)
    partner_mask = partner_v0 > 0.0
    par_playable = jnp.sum(partner_hand * playable_f[None], axis=(1, 2))
    par_critical = jnp.sum(partner_hand * critical[None].astype(jnp.float32), axis=(1, 2))
    par_useless = jnp.sum(partner_hand * useless_f[None], axis=(1, 2))

    was_certain_play = jnp.all(~(partner_mask & (~playable[None])), axis=(1, 2)) & partner_occ
    was_certain_dead = jnp.all(~(partner_mask & (~useless[None])), axis=(1, 2)) & partner_occ

    before_w = partner_mask.astype(jnp.float32) * remaining[None]
    before_p = (jnp.sum(before_w * playable_f[None], axis=(1, 2))
                / jnp.maximum(jnp.sum(before_w, axis=(1, 2)), _EPS))
    before_p = jnp.where(partner_occ, before_p, 0.0)
    # If some partner card already reads as playable on public evidence alone,
    # they have something to do and a fresh hint buys nothing this turn.
    before_trigger = jnp.max(before_p) >= _RECV_T_OTHER

    # ---- hint evaluation -----------------------------------------------------
    touched = jnp.einsum('scr,hcr->hs', partner_hand, attr.astype(jnp.float32)) > 0.5
    touched_f = touched.astype(jnp.float32)
    new_mask = partner_mask[None] & jnp.where(
        touched[:, :, None, None], attr[:, None, :, :], ~attr[:, None, :, :])
    size_new = jnp.sum(new_mask, axis=(2, 3))
    new_w = new_mask.astype(jnp.float32) * remaining[None, None]
    after_p = (jnp.sum(new_w * playable_f[None, None], axis=(2, 3))
               / jnp.maximum(jnp.sum(new_w, axis=(2, 3)), _EPS))
    after_p = jnp.where(partner_occ[None], after_p, 0.0)

    # Model the receiver the same way this policy reads hints: among the slots
    # the hint touches, the newest one that is literally plausible enough.
    eligible = touched & (after_p >= _RECV_T) & partner_occ[None]
    elig_key = eligible.astype(jnp.float32) * (1.0 + jnp.arange(_HAND, dtype=jnp.float32))[None]
    pick = jnp.argmax(elig_key, axis=1)
    any_eligible = jnp.max(elig_key, axis=1) > 0.0
    picked_good = par_playable[pick]
    triggers = any_eligible & (~before_trigger)
    trigger_value = jnp.where(
        triggers, jnp.where(picked_good > 0.5, ones, -_W_BOMB * ones), 0.0)

    bad_play = jnp.sum(new_mask & (~playable[None, None]), axis=(2, 3))
    bad_dead = jnp.sum(new_mask & (~useless[None, None]), axis=(2, 3))
    now_certain_play = (size_new > 0) & (bad_play == 0) & partner_occ[None]
    now_certain_dead = (size_new > 0) & (bad_dead == 0) & partner_occ[None]
    gained_play = jnp.sum((now_certain_play & (~was_certain_play[None])).astype(jnp.float32), axis=1)
    gained_dead = jnp.sum((now_certain_dead & (~was_certain_dead[None])).astype(jnp.float32), axis=1)

    touched_play = jnp.sum(touched_f * par_playable[None], axis=1)
    fresh_touch = touched_f * (partner_flags <= 0.0).astype(jnp.float32)
    dirty = jnp.sum(fresh_touch * partner_occ_f[None] * (1.0 - par_playable[None]), axis=1)
    clean = (touched_play > 0.5) & (
        jnp.sum(touched_f * partner_occ_f[None] * (1.0 - par_playable[None]), axis=1) < 0.5)
    info_gain = jnp.sum(partner_mask) - jnp.sum(size_new.astype(jnp.float32), axis=1)

    # Protect a last copy sitting where the partner is most likely to discard.
    unhinted = partner_occ & (jnp.sum(knowledge[1, :, 25:35], axis=1) <= 0.0)
    order = jnp.arange(_HAND, dtype=jnp.float32)
    chop_key = jnp.where(unhinted, order, 100.0 + order)
    chop_key = jnp.where(partner_occ, chop_key, 1000.0 + order)
    chop = jnp.argmin(chop_key)
    chop_at_risk = (par_critical[chop] > 0.5) & (par_playable[chop] < 0.5) & (par_useless[chop] < 0.5)
    save_hint = (touched[:, chop] & chop_at_risk).astype(jnp.float32)

    hint_value = (trigger_value
                  + _W_CLEAN * clean.astype(jnp.float32)
                  - _W_DIRTY * dirty
                  + _W_CERT * gained_play
                  + _W_DEAD * gained_dead
                  + _W_SAVE * save_hint
                  + _W_INFO * info_gain)
    token_penalty = jnp.where(tokens <= 1.5, 110.0, jnp.where(tokens <= 2.5, 40.0, 0.0))
    hint_score = _HINT_BASE + _HINT_SCALE * hint_value - token_penalty
    hint_score = jnp.where(info_gain <= 0.0, -500.0, hint_score)

    scores = jnp.concatenate([discard_score, play_score, hint_score,
                              jnp.asarray([-1e6], dtype=jnp.float32)])
    masked = jnp.where(legal, scores, -jnp.inf)
    action = jnp.argmax(masked).astype(jnp.int32)

    trace = {
        'p_play_best': jnp.max(jnp.where(own_occ, p_play, 0.0)),
        'score': jnp.max(masked),
        'tokens': tokens,
        'lives': lives,
        'pointed': jnp.sum(pointed),
    }
    return action, {'pointed': pointed}, trace
