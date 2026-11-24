"""Grounded-belief rule-based two-player Hanabi policy.

Design goals (cross-play): decisions are driven by publicly grounded evidence
(card counts, discards, fireworks, native hint knowledge) rather than by a
private signalling code. Hints from a partner are only *weak* evidence: a
touched card is played solely when its grounded playability probability is
also high enough, so a partner with a different hinting convention degrades
performance gracefully instead of causing bombs.
"""

from __future__ import annotations

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'grounded_belief_hanabi',
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

EPS = 1e-8
NEG = -1e9

# Tunable behaviour parameters (kept in one literal table for easy sweeping).
P = {
    'play_l3': 0.80, 'play_l2': 0.88, 'play_l1': 0.999,
    'end_l3': 0.30, 'end_l2': 0.55,
    'trust_l3': 0.60, 'trust_l2': 0.80,
    'h_play': 3.0, 'h_clarity': 9.0, 'h_save': 4.0, 'h_mislead': 5.0,
    'h_redundant': 7.0, 'h_extra': 0.6, 'h_bad': 0.6, 'h_nonew': 2.0,
    'bar_hi': 0.0, 'bar_mid': 2.0, 'bar_lo': 4.0,
    'd_useless': 4.0, 'd_crit': 6.0, 'd_play': 3.0,
    'd_untouched': 1.0, 'd_order': 0.25,
    'urgent_save': 1.0, 'save_min_tokens': 1.0, 'strict_discard': 1.0,
    'd_crit_hard': 0.0,
    'h_pending': 3.0, 'h_pending_all': 2.0, 'pending_decay': 0.6,
    'h_min_after': 0.0, 'd_crit_cut': 1.01, 'd_crit_hard2': 20.0,
    'h_danger': 8.0, 'h_known': 12.0, 'danger_level': 0.8, 'h_bar_crit': 0.0,
    'h_next': 0.0, 'h_exact': 0.0, 'trust_all': 0.0, 'end_deck': 0.0, 'end_no_discard': 0.0, 'h_unlock': 0.0, 'bar_last': 4.0, 'focus_oldest': 0.0, 'save_by_value': 1.0, 'save_min_value': 2.0,
}

_MULT_LIST = (3.0, 2.0, 2.0, 2.0, 1.0)


def _slots():
    return jnp.arange(5)


def _color_of():
    return jnp.arange(25) // 5


def _rank_of():
    return jnp.arange(25) % 5


def _mult():
    return jnp.asarray(_MULT_LIST, dtype=jnp.float32)


def _rank_val():
    return jnp.arange(5, dtype=jnp.float32)


def _hint_match():
    """[10, 25]: rows 0..4 are colour hints, rows 5..9 are rank hints."""
    return jnp.concatenate(
        [
            (_color_of()[None, :] == jnp.arange(5)[:, None]).astype(jnp.float32),
            (_rank_of()[None, :] == jnp.arange(5)[:, None]).astype(jnp.float32),
        ],
        axis=0,
    )


def init_agent():
    return {
        'touched': jnp.zeros((5,), dtype=jnp.float32),
        'pend': jnp.zeros((5,), dtype=jnp.float32),
        'ppend': jnp.zeros((5,), dtype=jnp.float32),
    }


def _shift_left(arr, removed):
    """Slot arrays after removing `removed`: later slots shift left, slot 4 is new."""
    slots = _slots()
    src = jnp.where(slots >= removed, slots + 1, slots)
    gathered = jnp.take(arr, jnp.clip(src, 0, 4))
    return jnp.where(src > 4, 0.0, gathered)


def _norm_probs(weights, targets):
    """weights: [5, 25] unnormalised belief. targets: [25] indicator."""
    total = jnp.sum(weights, axis=1)
    return jnp.sum(weights * targets[None, :], axis=1) / jnp.maximum(total, EPS)


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions
    _SLOTS = _slots()
    _RANK_OF = _rank_of()
    _MULT = _mult()
    _RANK_VAL = _rank_val()
    _HINT_MATCH = _hint_match()

    # ---------------- observation decoding ----------------
    partner_hand = f[0:125].reshape(5, 25)                      # [slot, 5c+r]
    deck = jnp.sum(f[127:167])
    fireworks = f[167:192].reshape(5, 5)
    height = jnp.sum(fireworks * (_RANK_VAL + 1.0)[None, :], axis=1)      # [5]
    tokens = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    disc = f[203:253].reshape(5, 10)
    discarded = jnp.stack(
        [
            jnp.sum(disc[:, 0:3], axis=1),
            jnp.sum(disc[:, 3:5], axis=1),
            jnp.sum(disc[:, 5:7], axis=1),
            jnp.sum(disc[:, 7:9], axis=1),
            jnp.sum(disc[:, 9:10], axis=1),
        ],
        axis=1,
    )                                                           # [color, rank]
    la = f[253:308]
    know = f[308:658].reshape(2, 5, 35)

    rank_grid = jnp.broadcast_to(_RANK_VAL[None, :], (5, 5))
    height_grid = jnp.broadcast_to(height[:, None], (5, 5))
    mult_grid = jnp.broadcast_to(_MULT[None, :], (5, 5))

    left_total = mult_grid - discarded                     # copies not yet discarded
    played_grid = (rank_grid < height_grid).astype(jnp.float32)
    remaining_pub = jnp.maximum(left_total - played_grid, 0.0)

    playable_g = (rank_grid == height_grid).astype(jnp.float32)
    dead_g = (left_total <= 0.0) & (rank_grid >= height_grid)
    attainable = jnp.min(jnp.where(dead_g, rank_grid, 5.0), axis=1)        # [5]
    useless_g = (rank_grid < height_grid) | (rank_grid >= attainable[:, None])
    needed_g = jnp.logical_not(useless_g)
    crit_g = jnp.logical_and(left_total <= 1.0, needed_g)

    playable_next_g = ((rank_grid == height_grid + 1.0)
                       & (rank_grid < attainable[:, None])).astype(jnp.float32)
    playable_next = playable_next_g.reshape(-1)
    playable = playable_g.reshape(-1)
    useless = useless_g.astype(jnp.float32).reshape(-1)
    critical = crit_g.astype(jnp.float32).reshape(-1)
    rank_flat = _RANK_OF.astype(jnp.float32)
    w_public = remaining_pub.reshape(-1)                                   # [25]

    partner_counts = jnp.sum(partner_hand, axis=0)                         # [25]
    w_self = jnp.maximum(w_public - partner_counts, 0.0)

    self_mask = (know[0, :, 0:25] > 0.0).astype(jnp.float32)               # [5, 25]
    self_w = self_mask * w_self[None, :]
    occ_self = (jnp.sum(self_mask, axis=1) > 0.0).astype(jnp.float32)
    touched_self = (jnp.sum(know[0, :, 25:35], axis=1) > 0.0).astype(jnp.float32)

    p_play = _norm_probs(self_w, playable) * occ_self
    p_useless = _norm_probs(self_w, useless) * occ_self
    p_crit = _norm_probs(self_w, critical) * occ_self
    e_rank = _norm_probs(self_w, rank_flat)

    # ---------------- memory maintenance ----------------
    act_self = la[0] > 0.5
    act_partner = la[1] > 0.5
    is_play = la[2] > 0.5
    is_discard = la[3] > 0.5
    is_hint = (la[4] > 0.5) | (la[5] > 0.5)
    to_self = la[6] > 0.5
    matched = la[18:23]
    removed_slot = jnp.argmax(la[23:28]).astype(jnp.int32)

    zeros5 = jnp.zeros((5,), jnp.float32)
    mem_touched = jnp.where(reset, zeros5, memory['touched'])
    mem_pend = jnp.where(reset, zeros5, memory['pend'])
    mem_ppend = jnp.where(reset, zeros5, memory['ppend'])

    removed_mine = act_self & (is_play | is_discard)
    mem_touched = jnp.where(removed_mine, _shift_left(mem_touched, removed_slot), mem_touched)
    mem_pend = jnp.where(removed_mine, _shift_left(mem_pend, removed_slot), mem_pend)
    removed_theirs = act_partner & (is_play | is_discard)
    mem_ppend = jnp.where(removed_theirs, _shift_left(mem_ppend, removed_slot), mem_ppend)
    mem_ppend = jnp.where(act_partner, mem_ppend * P['pending_decay'], mem_ppend)

    # A fresh hint addressed to us: identify its focus slot the way we pick one
    # when hinting (chop first, otherwise the newest newly-touched slot).
    prev_untouched = (1.0 - mem_touched) * occ_self
    chop_rank = jnp.where(prev_untouched > 0.5, _SLOTS.astype(jnp.float32), 99.0)
    any_chop = jnp.min(chop_rank) < 50.0
    fallback_chop = jnp.min(jnp.where(occ_self > 0.5, _SLOTS.astype(jnp.float32), 99.0))
    my_chop = jnp.where(any_chop, jnp.min(chop_rank), fallback_chop)
    my_chop = jnp.clip(my_chop, 0.0, 4.0).astype(jnp.int32)

    newly = matched * (1.0 - mem_touched)
    chop_hit = matched[my_chop] > 0.5
    newest_new = jnp.max(jnp.where(newly > 0.5, _SLOTS.astype(jnp.float32), -1.0))
    newest_any = jnp.max(jnp.where(matched > 0.5, _SLOTS.astype(jnp.float32), -1.0))
    oldest_new = jnp.min(jnp.where(newly > 0.5, _SLOTS.astype(jnp.float32), 9.0))
    oldest_any = jnp.min(jnp.where(matched > 0.5, _SLOTS.astype(jnp.float32), 9.0))
    focus_in = jnp.where(
        P['focus_oldest'] > 0.5,
        jnp.where(oldest_new <= 4.0, oldest_new, oldest_any),
        jnp.where(chop_hit, my_chop.astype(jnp.float32),
                  jnp.where(newest_new >= 0.0, newest_new, newest_any)),
    )
    focus_in = jnp.clip(focus_in, 0.0, 4.0).astype(jnp.int32)
    got_hint = act_partner & is_hint & to_self
    fresh = got_hint & (jnp.max(matched) > 0.5)
    marked = jnp.where(P['trust_all'] > 0.5,
                       jnp.maximum(newly, (_SLOTS == focus_in).astype(jnp.float32)),
                       (_SLOTS == focus_in).astype(jnp.float32))
    new_pend = jnp.where(fresh, jnp.maximum(mem_pend, marked), mem_pend) * occ_self

    # ---------------- play decision ----------------
    endgame = deck <= P['end_deck']
    thr_normal = jnp.where(lives >= 3.0, P['play_l3'], jnp.where(lives >= 2.0, P['play_l2'], P['play_l1']))
    thr_end = jnp.where(lives >= 3.0, P['end_l3'], jnp.where(lives >= 2.0, P['end_l2'], P['play_l1']))
    thr = jnp.where(endgame, thr_end, thr_normal)
    thr_trust = jnp.where(lives >= 3.0, P['trust_l3'], jnp.where(lives >= 2.0, P['trust_l2'], 1.01))

    trusted = (new_pend > 0.5) & (p_crit < 0.5)
    playable_now = (p_play >= thr) | (trusted & (p_play >= thr_trust))
    playable_now = playable_now & (occ_self > 0.5) & legal[5:10]

    # Prefer plays that unlock a card the partner already holds.
    partner_has = jnp.sum(partner_hand, axis=0)                             # [25]
    next_idx = jnp.clip(jnp.arange(25) + 1, 0, 24)
    unlock = jnp.where(_RANK_OF < 4, jnp.take(partner_has, next_idx), 0.0)
    unlock_val = _norm_probs(self_w, unlock) * occ_self
    play_score = jnp.where(
        playable_now,
        p_play + P['h_unlock'] * unlock_val + 0.05 * (4.0 - e_rank) + 0.01 * (4.0 - _SLOTS),
        NEG,
    )
    play_slot = jnp.argmax(play_score)
    can_play = jnp.max(play_score) > NEG / 2.0

    # ---------------- hint evaluation ----------------
    occ_p = (jnp.sum(partner_hand, axis=1) > 0.0).astype(jnp.float32)
    playable_p = jnp.sum(partner_hand * playable[None, :], axis=1)
    crit_p = jnp.sum(partner_hand * critical[None, :], axis=1)
    # Points the team forfeits if this last copy is destroyed: the colour caps
    # at the rank below it.
    loss_flat = (jnp.maximum(attainable[:, None] - rank_grid, 0.0) * crit_g).reshape(-1)
    crit_loss_p = jnp.sum(partner_hand * loss_flat[None, :], axis=1)
    touched_p = (jnp.sum(know[1, :, 25:35], axis=1) > 0.0).astype(jnp.float32)

    p_untouched = (1.0 - touched_p) * occ_p
    p_chop_rank = jnp.where(p_untouched > 0.5, _SLOTS.astype(jnp.float32), 99.0)
    p_any_chop = jnp.min(p_chop_rank) < 50.0
    p_fallback = jnp.min(jnp.where(occ_p > 0.5, _SLOTS.astype(jnp.float32), 99.0))
    p_chop = jnp.clip(jnp.where(p_any_chop, jnp.min(p_chop_rank), p_fallback), 0.0, 4.0).astype(jnp.int32)

    pmask = (know[1, :, 0:25] > 0.0).astype(jnp.float32)                   # [5, 25]
    pw_before = pmask * w_public[None, :]
    pp_before = _norm_probs(pw_before, playable) * occ_p
    pu_before = _norm_probs(pw_before, useless) * occ_p
    p_trash = jnp.where((pu_before >= 0.99) & (occ_p > 0.5), _SLOTS.astype(jnp.float32), 99.0)
    p_has_trash = jnp.min(p_trash) < 50.0
    p_discard_slot = jnp.where(p_has_trash, jnp.min(p_trash), p_chop.astype(jnp.float32))
    p_discard_slot = jnp.clip(p_discard_slot, 0.0, 4.0).astype(jnp.int32)

    touch_hs = jnp.einsum('hk,sk->hs', _HINT_MATCH, partner_hand)          # [10, 5]
    sel = jnp.where(touch_hs[:, :, None] > 0.5,
                    _HINT_MATCH[:, None, :],
                    1.0 - _HINT_MATCH[:, None, :])                          # [10, 5, 25]
    pw_after = pmask[None, :, :] * w_public[None, None, :] * sel
    tot_after = jnp.sum(pw_after, axis=2)
    pp_after = jnp.sum(pw_after * playable[None, None, :], axis=2) / jnp.maximum(tot_after, EPS)
    pp_after = pp_after * occ_p[None, :]
    # Cards that this hint makes grounded for the *next* rank of a colour: they
    # become playable without spending another hint once the current card lands.
    pn_after = jnp.sum(pw_after * playable_next[None, None, :], axis=2) / jnp.maximum(tot_after, EPS)
    pn_after = pn_after * occ_p[None, :]
    pn_before = _norm_probs(pw_before, playable_next) * occ_p
    actual_next_p = jnp.sum(partner_hand * playable_next[None, :], axis=1)
    # Cards whose identity the hint pins down exactly and which are still needed.
    nz_after = jnp.sum((pw_after > 0.0).astype(jnp.float32), axis=2)
    nz_before = jnp.sum((pw_before > 0.0).astype(jnp.float32), axis=1)
    needed_p = 1.0 - jnp.sum(partner_hand * useless[None, :], axis=1)

    new_hs = touch_hs * (1.0 - touched_p)[None, :]
    chop_hit_h = touch_hs[:, p_chop] > 0.5
    newest_new_h = jnp.max(jnp.where(new_hs > 0.5, _SLOTS.astype(jnp.float32)[None, :], -1.0), axis=1)
    newest_any_h = jnp.max(jnp.where(touch_hs > 0.5, _SLOTS.astype(jnp.float32)[None, :], -1.0), axis=1)
    oldest_new_h = jnp.min(jnp.where(new_hs > 0.5, _SLOTS.astype(jnp.float32)[None, :], 9.0), axis=1)
    oldest_any_h = jnp.min(jnp.where(touch_hs > 0.5, _SLOTS.astype(jnp.float32)[None, :], 9.0), axis=1)
    focus_h = jnp.where(
        P['focus_oldest'] > 0.5,
        jnp.where(oldest_new_h <= 4.0, oldest_new_h, oldest_any_h),
        jnp.where(chop_hit_h, p_chop.astype(jnp.float32),
                  jnp.where(newest_new_h >= 0.0, newest_new_h, newest_any_h)),
    )
    focus_h = jnp.clip(focus_h, 0.0, 4.0).astype(jnp.int32)
    focus_oh = (jnp.arange(5)[None, :] == focus_h[:, None]).astype(jnp.float32)

    f_playable = jnp.sum(focus_oh * playable_p[None, :], axis=1)
    f_crit = jnp.sum(focus_oh * crit_p[None, :], axis=1)
    f_after = jnp.sum(focus_oh * pp_after, axis=1)
    f_before = jnp.sum(focus_oh * pp_before[None, :], axis=1)
    f_is_chop = jnp.sum(focus_oh * (jnp.arange(5) == p_discard_slot).astype(jnp.float32)[None, :], axis=1)

    n_new = jnp.sum(new_hs, axis=1)
    bad_new = jnp.sum(new_hs * (1.0 - playable_p)[None, :], axis=1) - (1.0 - f_playable) * jnp.sum(focus_oh * new_hs, axis=1)
    extra_playable = jnp.sum(touch_hs * playable_p[None, :], axis=1) - f_playable

    f_loss = jnp.sum(focus_oh * crit_loss_p[None, :], axis=1)
    save_value = (1.0 - f_playable) * f_is_chop * jnp.where(
        P['save_by_value'] > 0.5, f_loss, f_crit)
    hint_score = (
        f_playable * (f_after >= P['h_min_after']).astype(jnp.float32)
        * (P['h_play'] + P['h_clarity'] * f_after)
        + P['h_save'] * save_value
        - P['h_mislead'] * (1.0 - f_playable) * (1.0 - save_value)
        - P['h_redundant'] * f_playable * (f_before >= 0.85).astype(jnp.float32)
        + P['h_extra'] * extra_playable
        - P['h_bad'] * jnp.maximum(bad_new, 0.0)
        - P['h_nonew'] * (n_new <= 0.0).astype(jnp.float32)
        - P['h_danger'] * jnp.sum(
            (pp_after >= P['danger_level']).astype(jnp.float32)
            * (1.0 - playable_p)[None, :] * occ_p[None, :], axis=1)
        + P['h_known'] * jnp.sum(
            ((pp_after >= P['danger_level']) & (pp_before < P['danger_level'])[None, :]
             ).astype(jnp.float32) * playable_p[None, :], axis=1)
        + P['h_next'] * jnp.sum(
            ((pn_after >= P['danger_level']) & (pn_before < P['danger_level'])[None, :]
             ).astype(jnp.float32) * actual_next_p[None, :], axis=1)
        + P['h_exact'] * jnp.sum(
            ((nz_after <= 1.5) & (nz_before > 1.5)[None, :]).astype(jnp.float32)
            * (needed_p * occ_p)[None, :], axis=1)
        - P['h_pending'] * jnp.sum(focus_oh * mem_ppend[None, :], axis=1)
        - P['h_pending_all'] * jnp.max(mem_ppend)
    )
    hint_score = jnp.where(legal[10:20], hint_score, NEG)
    hint_idx = jnp.argmax(hint_score)
    best_hint = jnp.max(hint_score)
    hint_legal = best_hint > NEG / 2.0
    hint_bar = jnp.where(tokens >= 5.0, P['bar_hi'],
                         jnp.where(tokens >= 3.0, P['bar_mid'],
                                   jnp.where(tokens >= 2.0, P['bar_lo'], P['bar_last'])))

    gave_hint = act_self & is_hint & (la[7] > 0.5)
    new_ppend = jnp.where(gave_hint, jnp.maximum(mem_ppend, matched * playable_p), mem_ppend)
    new_ppend = jnp.maximum(new_ppend, (pp_before >= 0.85).astype(jnp.float32) * playable_p) * occ_p
    next_memory = {'touched': touched_self, 'pend': new_pend, 'ppend': new_ppend}

    # Urgent save: the partner's chop holds the last copy of a still-needed card,
    # so touching it (any hint) stops a routine discard from destroying the colour.
    chop_oh = (jnp.arange(5) == p_discard_slot).astype(jnp.float32)
    chop_crit = ((jnp.sum(chop_oh * crit_loss_p) >= P['save_min_value'])
                 & (jnp.sum(chop_oh * occ_p) > 0.5))
    save_scores = jnp.where(touch_hs[:, p_discard_slot] > 0.5, hint_score, NEG)
    save_idx = jnp.argmax(save_scores)
    save_possible = jnp.max(save_scores) > NEG / 2.0
    urgent_save = (
        chop_crit
        & save_possible
        & (tokens >= P['save_min_tokens'])
        & (P['urgent_save'] > 0.5)
    )

    # ---------------- discard decision ----------------
    soft_score = (
        P['d_useless'] * p_useless
        - P['d_crit'] * p_crit
        - P['d_play'] * p_play
        + P['d_untouched'] * (1.0 - touched_self)
        + P['d_order'] * (4.0 - _SLOTS)
    )
    # Strict rule: provably useless card first, otherwise the chop (oldest
    # untouched slot). This is what a partner can predict about us.
    my_chop_now_rank = jnp.where((touched_self < 0.5) & (occ_self > 0.5),
                                 _SLOTS.astype(jnp.float32), 99.0)
    has_chop_now = jnp.min(my_chop_now_rank) < 50.0
    my_chop_now = jnp.clip(jnp.where(has_chop_now, jnp.min(my_chop_now_rank), 0.0),
                           0.0, 4.0).astype(jnp.int32)
    strict_score = (
        100.0 * (p_useless >= 0.99).astype(jnp.float32)
        + 10.0 * (_SLOTS == my_chop_now).astype(jnp.float32)
        - P['d_crit_hard2'] * (p_crit >= P['d_crit_cut']).astype(jnp.float32)
        - P['d_crit'] * p_crit
        - P['d_play'] * p_play
        + P['d_order'] * (4.0 - _SLOTS)
    )
    disc_score = jnp.where(P['strict_discard'] > 0.5, strict_score, soft_score)
    disc_score = jnp.where(legal[0:5], disc_score, NEG)
    disc_slot = jnp.argmax(disc_score)
    my_disc_crit = jnp.sum((_SLOTS == disc_slot).astype(jnp.float32) * p_crit)
    want_hint = hint_legal & (best_hint > hint_bar - P['h_bar_crit'] * my_disc_crit)
    # With no draws left a discard permanently removes a playable candidate, so
    # spending a token on any legal hint is usually better.
    want_hint = want_hint | (hint_legal & (deck <= 0.0) & (P['end_no_discard'] > 0.5))
    can_discard = jnp.max(disc_score) > NEG / 2.0

    # ---------------- combine ----------------
    play_action = (5 + play_slot).astype(jnp.int32)
    hint_action = (10 + hint_idx).astype(jnp.int32)
    disc_action = disc_slot.astype(jnp.int32)

    fallback = jnp.where(hint_legal, hint_action,
                         jnp.where(can_discard, disc_action, jnp.argmax(legal[0:20]).astype(jnp.int32)))
    save_action = (10 + save_idx).astype(jnp.int32)
    action = jnp.where(
        urgent_save,
        save_action,
        jnp.where(
            can_play,
            play_action,
            jnp.where(want_hint, hint_action, jnp.where(can_discard, disc_action, fallback)),
        ),
    )
    action = jnp.where(legal[20], jnp.asarray(20, jnp.int32), action)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32)).astype(jnp.int32)

    dead_now = ((left_total <= 0.0) & (rank_grid >= height_grid)).astype(jnp.float32).reshape(-1)
    revealed = la[28:53]
    trace = {
        'p_play_best': jnp.max(p_play),
        'hint_score': best_hint,
        'tokens': tokens,
        'lives': lives,
        'deck': deck,
        'score': jnp.sum(height),
        'achievable': jnp.sum(jnp.maximum(attainable, height)),
        'last_bomb': (is_play & (la[53] < 0.5)).astype(jnp.float32),
        'last_waste': (is_discard.astype(jnp.float32)) * jnp.sum(revealed * dead_now),
        'pend': new_pend,
        'play_p': jnp.sum((_SLOTS == play_slot).astype(jnp.float32) * p_play),
        'play_trust': jnp.sum((_SLOTS == play_slot).astype(jnp.float32)
                              * (trusted & (p_play < thr)).astype(jnp.float32)),
    }
    return action, next_memory, trace
