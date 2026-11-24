"""Two-player Hanabi policy following H-group Level 1 conventions.

Coordination protocol (all public, documented in the pinned bundle):
  * Chop = lowest occupied native index among unclued cards.
  * Clue focus = chop if the chop was newly touched, else leftmost (= highest
    native index) newly touched card, else leftmost re-touched card.
  * A clue focusing a non-chop card is a play clue (possibly delayed).
    A chop-focused clue may be a play clue or one of the permitted saves
    (5 save with rank 5, 2 save with rank 2, critical save with any clue).
  * Good Touch Principle: only touch cards that will eventually be played, so
    touched cards are assumed non-trash.
  * Minimum Clue Value Principle: clues must get a new card played or saved
    (tempo clues only in stalling / high-value situations).
Cards are played only when every remaining candidate identity is currently
playable, because the scoring rule zeroes the episode on the third bomb.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_l1_v1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def _consts():
    return (jnp.array([3.0, 2.0, 2.0, 2.0, 1.0], dtype=jnp.float32),
            jnp.arange(5, dtype=jnp.float32),
            jnp.arange(5, dtype=jnp.int32),
            jnp.arange(10, dtype=jnp.int32),
            jnp.eye(5, dtype=jnp.bool_))


def init_agent():
    return {
        'cand': jnp.ones((5, 5, 5), dtype=jnp.float32),
        'play_note': jnp.zeros((5,), dtype=jnp.bool_),
        'save_note': jnp.zeros((5,), dtype=jnp.bool_),
        'p_play': jnp.zeros((5,), dtype=jnp.bool_),
        'p_save': jnp.zeros((5,), dtype=jnp.bool_),
        'prev_ts': jnp.zeros((5,), dtype=jnp.bool_),
        'prev_tp': jnp.zeros((5,), dtype=jnp.bool_),
        'cnt': jnp.zeros((4,), dtype=jnp.int32),
    }


def agent_act(obs, reset, memory, rng):
    TOTALS, R, I, I10, EYE = _consts()

    def hi(mask):
        return jnp.max(jnp.where(mask, I, -1))

    f = obs.features
    legal = obs.legal_actions
    reset = jnp.asarray(reset, dtype=jnp.bool_).reshape(())

    # ---------------- public board state ----------------
    ph = f[0:125].reshape(5, 5, 5)                     # partner hand [slot, color, rank]
    p_occ = ph.sum((1, 2)) > 0.5
    deck = f[127:167].sum()
    fw = f[167:192].reshape(5, 5)
    height = (fw * (R + 1.0)[None, :]).sum(1)         # per-color firework height
    tokens = f[192:200].sum()
    lives = f[200:203].sum()
    dsc = f[203:253].reshape(5, 10)
    dcount = jnp.stack([dsc[:, 0:3].sum(1), dsc[:, 3:5].sum(1), dsc[:, 5:7].sum(1),
                        dsc[:, 7:9].sum(1), dsc[:, 9:10].sum(1)], axis=1)

    rnk = R[None, :]
    below = rnk < height[:, None]
    remaining = TOTALS[None, :] - dcount - below.astype(jnp.float32)
    playable = rnk == height[:, None]
    gone = remaining <= 0.5
    blocked = gone & (~below)
    cb = jnp.cumsum(blocked.astype(jnp.float32), axis=1) - blocked.astype(jnp.float32)
    dead = cb > 0.5
    trash = below | dead | gone
    useful = ~trash
    critical = (remaining > 0.5) & (remaining < 1.5) & useful

    # ---------------- knowledge blocks ----------------
    kn = f[308:658].reshape(2, 5, 35)
    v0 = kn[0, :, 0:25].reshape(5, 5, 5)
    v0p = kn[1, :, 0:25].reshape(5, 5, 5)
    lit = v0 > 0.0
    s_occ = lit.any((1, 2))
    t_self = kn[0, :, 25:35].sum(1) > 0.5
    t_part = kn[1, :, 25:35].sum(1) > 0.5

    # ---------------- memory ----------------
    fresh = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), fresh, memory)
    cand = mem['cand']
    play_note = mem['play_note']
    save_note = mem['save_note']
    p_play = mem['p_play']
    p_save = mem['p_save']
    prev_ts = mem['prev_ts']
    prev_tp = mem['prev_tp']

    # ---------------- last public action ----------------
    la = f[253:308]
    act_me = la[0] > 0.5
    a_play = la[2] > 0.5
    a_disc = la[3] > 0.5
    a_ch = la[4] > 0.5
    a_rh = la[5] > 0.5
    tgt_me = la[6] > 0.5
    hint_rank_v = la[13:18]
    matched = la[18:23] > 0.5
    rem_slot = jnp.argmax(la[23:28]).astype(jnp.int32)
    moved = (a_play | a_disc) & (~reset)
    hinted = (a_ch | a_rh) & (~reset)

    # slot shift after a play/discard
    src = jnp.where(I < rem_slot, I, jnp.minimum(I + 1, 4))

    def shift1(arr, fresh_val):
        out = jnp.take(arr, src, axis=0)
        newmask = (I == 4).reshape((5,) + (1,) * (arr.ndim - 1))
        return jnp.where(newmask, fresh_val, out)

    mine_moved = moved & act_me
    part_moved = moved & (~act_me)
    cand = jnp.where(mine_moved, shift1(cand, jnp.float32(1.0)), cand)
    play_note = jnp.where(mine_moved, shift1(play_note, False), play_note)
    save_note = jnp.where(mine_moved, shift1(save_note, False), save_note)
    p_play = jnp.where(part_moved, shift1(p_play, False), p_play)
    p_save = jnp.where(part_moved, shift1(p_save, False), p_save)
    # A partner who had a play clue and still chose to clue or discard is not
    # holding a play any more; keeping a stale expectation would suppress the
    # save clues they depend on.
    partner_passed = (~reset) & (~act_me) & (a_disc | a_ch | a_rh)
    p_play = jnp.where(partner_passed, jnp.zeros((5,), dtype=jnp.bool_), p_play)

    # ---------------- queued plays / delayed playability ----------------
    # A card only counts as "accounted for" when its own holder can identify it
    # from public information, so both seats derive the same queue.
    cset0 = (cand > 0.0) & lit
    csum0 = cset0.sum((1, 2))
    lit_n = lit.sum((1, 2))
    p_play_slot0 = (ph * playable[None].astype(jnp.float32)).sum((1, 2)) > 0.5
    p_id_known = (v0p.sum((1, 2)) > 0.0) & ((v0p > 0.0).sum((1, 2)) == 1)
    p_ready = p_occ & t_part & (p_id_known | (p_play & p_play_slot0))
    p_known = ((ph > 0.5) & p_ready[:, None, None]).any(0) & useful
    mine_sure0 = ((cset0 & (~playable)[None]).sum((1, 2)) == 0) & (csum0 > 0)
    mine_ready = (s_occ & t_self & (csum0 == 1)
                  & ((lit_n == 1) | (play_note & mine_sure0)))
    my_known = (cset0 & mine_ready[:, None, None]).any(0) & useful
    queued = my_known | p_known
    eff = height
    for _ in range(4):
        oh = R[None, :] == eff[:, None]
        eff = jnp.minimum(eff + (queued & oh).any(1).astype(jnp.float32), 5.0)
    delayed = (R[None, :] == eff[:, None]) & useful

    # When reading someone else's clue, allow the looser reading in which any
    # clued card in their hand counts as an accounted-for connector.  Being
    # generous here only widens my candidate set, so an independent partner
    # whose delayed-clue threshold differs cannot push me into a false
    # certainty; the conservative set above still governs the clues I give.
    p_known_lax = ((ph > 0.5) & (t_part & p_occ)[:, None, None]).any(0) & useful
    queued_rx = queued | p_known_lax
    eff_rx = height
    for _ in range(4):
        oh = R[None, :] == eff_rx[:, None]
        eff_rx = jnp.minimum(eff_rx + (queued_rx & oh).any(1).astype(jnp.float32), 5.0)
    delayed_rx = (R[None, :] == eff_rx[:, None]) & useful

    # ---------------- interpret the last clue ----------------
    pre_t = jnp.where(tgt_me, prev_ts, prev_tp)
    r_occ = jnp.where(tgt_me, s_occ, p_occ)
    newc = matched & (~pre_t) & r_occ
    unclued = r_occ & (~pre_t)
    chop_ex = unclued.any()
    chop = jnp.argmax(unclued).astype(jnp.int32)
    any_new = newc.any()
    focus = jnp.where(any_new,
                      jnp.where(chop_ex & matched[chop], chop, hi(newc)),
                      hi(matched & pre_t & r_occ))
    focus = jnp.clip(focus, 0, 4)
    chop_focus = any_new & chop_ex & (focus == chop)
    fmask = I == focus

    rank5_clue = a_rh & (hint_rank_v[4] > 0.5)
    rank2_clue = a_rh & (hint_rank_v[1] > 0.5)
    is5 = rnk == 4.0
    is2 = rnk == 1.0
    save_set = (critical | (is5 & rank5_clue) | (is2 & rank2_clue)) & useful
    play_set = (playable | delayed_rx) & useful
    target = jnp.where(chop_focus, play_set | save_set, play_set)

    upd_me = hinted & tgt_me
    fc_norm = cand[focus] * target.astype(jnp.float32) * lit[focus].astype(jnp.float32)
    normal_ok = fc_norm.sum() > 0.0

    # The Prompt: if the focused card cannot be playable or delayed-playable,
    # the clue promises that one of my own clued cards is the connector.  The
    # leftmost (highest native index) compatible clued card is the prompted one.
    conn_now = ((rnk == height[:, None]) & useful).astype(jnp.float32)      # [c, r]
    conn_have = (cand * conn_now[None]).sum(2) > 0.0                       # [slot, color]
    conn_ok = (conn_have & (s_occ & t_self & (~fmask))[:, None]).any(0)    # [color]
    nxt = height + 1.0
    prompt_set = ((rnk == nxt[:, None]) & (nxt[:, None] <= 4.5)
                  & conn_ok[:, None] & useful)
    fc_pr = cand[focus] * prompt_set.astype(jnp.float32) * lit[focus].astype(jnp.float32)
    use_prompt = upd_me & (~normal_ok) & (fc_pr.sum() > 0.0) & (~chop_focus)

    fc = jnp.where(normal_ok, fc_norm, jnp.where(use_prompt, fc_pr, cand[focus]))
    cand = jnp.where(upd_me & fmask.reshape(5, 1, 1), fc[None], cand)
    could_save = (fc * save_set.astype(jnp.float32)).sum() > 0.0
    play_note = play_note | (upd_me & fmask & ((~chop_focus) | (~could_save)))
    save_note = save_note | (upd_me & fmask & chop_focus)

    poss_col = fc_pr.sum(1) > 0.0                                          # [color]
    conn_target = conn_now * poss_col[:, None].astype(jnp.float32)
    compat = ((cand * conn_target[None]).sum((1, 2)) > 0.0) & s_occ & t_self & (~fmask)
    pslot = hi(compat)
    do_prompt = use_prompt & (pslot >= 0)
    pmask = I == jnp.clip(pslot, 0, 4)
    cand = jnp.where(do_prompt & pmask.reshape(5, 1, 1),
                     (cand[jnp.clip(pslot, 0, 4)] * conn_target)[None], cand)
    play_note = play_note | (do_prompt & pmask)

    f_card = ph[focus]
    f_play_p = (f_card * playable.astype(jnp.float32)).sum() > 0.5
    upd_p = hinted & (~tgt_me)
    p_play = p_play | (upd_p & fmask & f_play_p)
    p_save = p_save | (upd_p & fmask & (~f_play_p))

    # ---------------- maintain candidates ----------------
    keep = jnp.where(t_self[:, None, None], useful[None], True)
    cand_n = cand * lit.astype(jnp.float32) * keep.astype(jnp.float32)
    empty = cand_n.sum((1, 2)) <= 0.0
    cand = jnp.where(empty[:, None, None], lit.astype(jnp.float32), cand_n)

    cset = cand > 0.0
    csum = cset.sum((1, 2))
    lit_sure = s_occ & (lit.sum((1, 2)) > 0) & ((lit & (~playable)[None]).sum((1, 2)) == 0)
    conv_sure = s_occ & (csum > 0) & ((cset & (~playable)[None]).sum((1, 2)) == 0)
    # Convention-derived certainty is only trusted for cards that were the focus
    # of a play clue; Good-Touch elimination alone can be wrong when a partner
    # unknowingly duplicates a card, and a third bomb zeroes the episode.
    v0sum0 = jnp.maximum(v0.sum((1, 2)), 1e-6)
    pplay0 = (v0 * playable[None].astype(jnp.float32)).sum((1, 2)) / v0sum0
    # Risk ladder: Good-Touch elimination can be wrong when a card was
    # unknowingly duplicated, so it is trusted in full only while two spare
    # lives remain and never on the last life.
    gtp_ok = conv_sure & ((lives > 2.5) | ((lives > 1.5) & (pplay0 > 0.3)))
    know_play = lit_sure | (conv_sure & play_note & (lives > 1.5)) | gtp_ok
    know_trash = s_occ & (csum > 0) & ((cset & useful[None]).sum((1, 2)) == 0)

    # ---------------- action scores ----------------
    v0sum = jnp.maximum(v0.sum((1, 2)), 1e-6)
    prob_play = (v0 * playable[None].astype(jnp.float32)).sum((1, 2)) / v0sum
    prob_trash = (v0 * trash[None].astype(jnp.float32)).sum((1, 2)) / v0sum
    minrank = jnp.min(jnp.where(cset, R[None, None, :], 9.0), axis=(1, 2))

    # Once the deck is empty a failed play only costs a card that would never
    # have been played anyway, so a likely play beats a pointless discard -
    # but never on the last life, where a bomb wipes out the whole score.
    endgame_gamble = (s_occ & (deck < 0.5) & (lives > 1.5) & (prob_play > 0.45)
                      & (~know_play))
    play_scores = (jnp.where(know_play, 1.0e5 + (5.0 - minrank) * 20.0, 0.0)
                   + jnp.where(know_play & play_note, 5.0, 0.0)
                   + jnp.where(endgame_gamble, 2.0e3 + 100.0 * prob_play, 0.0)
                   + 0.05 + 0.04 * prob_play)

    s_unclued = s_occ & (~t_self)
    my_chop_ex = s_unclued.any()
    my_chop = jnp.argmax(s_unclued).astype(jnp.int32)
    prob_crit = (v0 * critical[None].astype(jnp.float32)).sum((1, 2)) / v0sum
    disc_scores = (jnp.where(know_trash, 1.0e3 + (4 - I).astype(jnp.float32), 0.0)
                   + jnp.where(my_chop_ex & (I == my_chop), 5.0e2, 0.0)
                   + jnp.where(s_unclued & (I != my_chop), 4.0e2 - I.astype(jnp.float32), 0.0)
                   # last resort: an all-clued hand discards the clued card that
                   # is most likely useless and carries no play/save meaning
                   + jnp.where(s_occ & t_self & (~know_trash), 5.0e1, 0.0)
                   - jnp.where(s_occ & t_self & (play_note | save_note), 1.0e2, 0.0)
                   + jnp.where(s_occ & t_self, 150.0 * prob_trash - 400.0 * prob_crit, 0.0)
                   + jnp.where(s_occ, 1.0, 0.0) + 5.0 * prob_trash)
    # Any legal discard must outrank the blind-play fallback: giving up a card
    # is always better than gambling a life on an unknown card.
    disc_scores = jnp.maximum(disc_scores, 1.0)

    # ---- clue evaluation over the ten (color, rank) options ----
    m_color = (ph.sum(2) > 0.5).T
    m_rank = (ph.sum(1) > 0.5).T
    cmatch = jnp.concatenate([m_color, m_rank], axis=0)            # [10, 5]
    cnew = cmatch & (~t_part)[None, :] & p_occ[None, :]
    c_anynew = cnew.any(1)
    pc_unclued = p_occ & (~t_part)
    pchop_ex = pc_unclued.any()
    pchop = jnp.argmax(pc_unclued).astype(jnp.int32)
    touch_chop = cmatch[:, pchop] & pchop_ex
    hi_new_k = jnp.max(jnp.where(cnew, I[None, :], -1), axis=1)
    hi_ret_k = jnp.max(jnp.where(cmatch & (t_part & p_occ)[None, :], I[None, :], -1), axis=1)
    kfocus = jnp.where(c_anynew, jnp.where(touch_chop, pchop, hi_new_k), hi_ret_k)
    kfocus = jnp.clip(kfocus, 0, 4)
    fcards = ph[kfocus]                                            # [10, 5, 5]
    k_playable = (fcards * playable[None].astype(jnp.float32)).sum((1, 2)) > 0.5
    k_delayed = (fcards * delayed[None].astype(jnp.float32)).sum((1, 2)) > 0.5
    k_crit = (fcards * critical[None].astype(jnp.float32)).sum((1, 2)) > 0.5
    k_trash = (fcards * trash[None].astype(jnp.float32)).sum((1, 2)) > 0.5
    k_is5 = fcards[:, :, 4].sum(1) > 0.5
    k_is2 = fcards[:, :, 1].sum(1) > 0.5
    k_rank = (fcards * R[None, None, :]).sum((1, 2))
    k_chopfocus = c_anynew & pchop_ex & (kfocus == pchop)

    p_trash_slot = (ph * trash[None].astype(jnp.float32)).sum((1, 2)) > 0.5
    p_crit_slot = (ph * critical[None].astype(jnp.float32)).sum((1, 2)) > 0.5
    p_play_slot = (ph * playable[None].astype(jnp.float32)).sum((1, 2)) > 0.5
    p_is5_slot = ph[:, :, 4].sum(1) > 0.5
    p_is2_slot = ph[:, :, 1].sum(1) > 0.5

    pid = ph.reshape(5, 25)
    same = ((pid @ pid.T) > 0.5) & (~EYE) & p_occ[None, :] & p_occ[:, None]
    dup_prev = (same & (t_part & p_occ)[None, :]).any(1)
    dup_new = (same[None, :, :] & cnew[:, None, :]
               & (I[None, None, :] > I[None, :, None])).any(2)
    bad_slot = p_trash_slot[None, :] | dup_prev[None, :] | dup_new
    n_bad = (cnew & bad_slot).sum(1)
    n_good = (cnew & (~bad_slot)).sum(1)

    is_color_k = I10 < 5
    k_rank_val = jnp.clip(I10 - 5, 0, 4)
    rank5_k = (~is_color_k) & (k_rank_val == 4)
    rank2_k = (~is_color_k) & (k_rank_val == 1)
    chop_dup_visible = (same[pchop] & p_occ).any()

    k_play_ok = c_anynew & k_playable
    k_del_ok = c_anynew & k_delayed & (~k_playable) & (n_bad == 0)
    k_save_ok = (k_chopfocus & (~k_trash)
                 & (k_crit | (k_is5 & rank5_k) | (k_is2 & rank2_k & (~chop_dup_visible))))

    # partner knows a card of theirs is playable already?
    v0p_sum = jnp.maximum(v0p.sum((1, 2)), 1e-6)
    p_sure_play = ((v0p * (~playable)[None].astype(jnp.float32)).sum((1, 2)) < 1e-6) & p_occ
    partner_has_play = (p_occ & p_play_slot & (p_play | p_sure_play)).any()

    # Which clues "fill in" one of the partner's already clued cards so that
    # their own literal knowledge proves it playable?  Positive and negative
    # information are both applied, exactly as the native knowledge does.
    cmask = jnp.concatenate([
        (jnp.arange(5, dtype=jnp.int32)[:, None, None] == I[None, :, None]) & jnp.ones((1, 1, 5), dtype=jnp.bool_),
        (jnp.arange(5, dtype=jnp.int32)[:, None, None] == I[None, None, :]) & jnp.ones((1, 5, 1), dtype=jnp.bool_),
    ], axis=0).astype(jnp.float32)                                  # [10, 5, 5]
    keep_k = jnp.where(cmatch[:, :, None, None] > 0,
                       cmask[:, None, :, :], 1.0 - cmask[:, None, :, :])
    v0p_after = v0p[None, :, :, :] * keep_k                         # [10, 5, 5, 5]
    after_sum = v0p_after.sum((2, 3))
    after_bad = (v0p_after * (~playable)[None, None, :, :].astype(jnp.float32)).sum((2, 3))
    sure_after = (after_sum > 1e-6) & (after_bad < 1e-6) & p_occ[None, :]
    new_sure = sure_after & (~p_sure_play)[None, :] & (~p_play)[None, :] & t_part[None, :]
    n_fill = new_sure.sum(1).astype(jnp.float32)

    # tempo clue: no new card, but makes an already clued playable card known
    k_tempo = (~c_anynew) & (n_fill > 0.5) & (tokens > 1.5)

    chop_needs = pchop_ex & (~p_trash_slot[pchop]) & (
        p_crit_slot[pchop] | p_is5_slot[pchop]
        | (p_is2_slot[pchop] & (~chop_dup_visible)))
    urgent = chop_needs & (~partner_has_play)

    # Good Touch is a preference, not a hard veto: losing every copy of a card
    # costs more than the clutter of an unavoidable bad touch, so bad touches
    # are penalised rather than forbidden.
    badpen = 1200.0 * n_bad.astype(jnp.float32)
    save_tier = jnp.where(urgent, 1.0e6, jnp.where(tokens > 1.5, 5.0e3, 0.0))
    tier = (jnp.where(k_play_ok, jnp.maximum(1.0e4 - badpen, 6.0e2), 0.0)
            + jnp.where(k_del_ok, 9.0e3, 0.0)
            + jnp.where(k_save_ok, jnp.maximum(save_tier - badpen, 6.0e2), 0.0)
            + jnp.where(k_tempo, 3.0e3, 0.0))
    n_now = (cnew & (~bad_slot) & p_play_slot[None, :]).sum(1).astype(jnp.float32)
    bonus = (120.0 * n_good.astype(jnp.float32)
             + 300.0 * n_now
             + 800.0 * n_fill
             + 60.0 * is_color_k.astype(jnp.float32)
             + 40.0 * (5.0 - k_rank)
             - 150.0 * (k_chopfocus & k_play_ok & (k_is5 | k_is2 | k_crit)
                        & (~is_color_k)).astype(jnp.float32))
    clue_scores = (jnp.where(tier > 0.0, tier + bonus, 0.0)
                   + 0.5 + 0.2 * (~c_anynew).astype(jnp.float32)
                   + 0.1 * k_playable.astype(jnp.float32))

    scores = jnp.concatenate([disc_scores, play_scores, clue_scores,
                              jnp.zeros((1,), dtype=jnp.float32)])
    scores = jnp.where(legal, scores, -1.0e9)
    action = jnp.argmax(scores).astype(jnp.int32)

    mine_act = (~legal[20]) & (~reset)
    cnt = mem['cnt'] + jnp.stack([
        (mine_act & (action < 5)).astype(jnp.int32),
        (mine_act & (action >= 5) & (action < 10)).astype(jnp.int32),
        (mine_act & (action >= 10) & (action < 20)).astype(jnp.int32),
        jnp.asarray(1, dtype=jnp.int32)])
    next_mem = {
        'cnt': cnt,
        'cand': cand,
        'play_note': play_note & s_occ,
        'save_note': save_note & s_occ,
        'p_play': p_play & p_occ,
        'p_save': p_save & p_occ,
        'prev_ts': t_self,
        'prev_tp': t_part,
    }
    trace = {
        'cnt': cnt,
        'csum': csum.astype(jnp.int32),
        'kp': know_play.astype(jnp.int32),
        'pn': play_note.astype(jnp.int32),
        'ls': lit_sure.astype(jnp.int32),
        'cs': conv_sure.astype(jnp.int32),
        'cand_id': jnp.argmax(cand.reshape(5, 25), axis=1).astype(jnp.int32),
        'n_known_play': know_play.sum().astype(jnp.int32),
        'n_known_trash': know_trash.sum().astype(jnp.int32),
        'tokens': tokens.astype(jnp.float32),
        'lives': lives.astype(jnp.float32),
        'deck': deck.astype(jnp.float32),
        'score': height.sum().astype(jnp.float32),
        'action': action,
    }
    return action, next_mem, trace
