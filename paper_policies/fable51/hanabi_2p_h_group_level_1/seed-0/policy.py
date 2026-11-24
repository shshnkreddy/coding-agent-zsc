"""H-group Level 1 convention policy for two-player Hanabi (pure JAX).

Conventions implemented (see context/conventions/h_group_level_1):
  * chop = lowest occupied unclued native index; "leftmost" = highest index
  * clue focus: chop if newly touched, else leftmost new card, else leftmost
    re-touched card
  * play clues (direct through stacks and partner's visible clued cards),
    self-prompts only when no direct interpretation exists
  * save clues on chop: 5 Save (rank 5), 2 Save (rank 2, visible rule),
    Critical Save (colour or rank)
  * Good Touch, Save, Minimum Clue Value principles; early game = clue before
    discarding when a valid clue exists
Memory holds per-slot possibility masks (own hand and a model of the partner's
notes), promise flags and previous clued flags.  All control flow is fixed
shape.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_l1_v1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NC, NR, HS, NID = 5, 5, 5, 25
NEG = -1e9


def _consts():
    mult = jnp.array([3, 2, 2, 2, 1], dtype=jnp.int32)
    id_c = jnp.arange(NID) // NR
    id_r = jnp.arange(NID) % NR
    slot = jnp.arange(HS)
    ridx = jnp.arange(NR)[None, :]
    sub = jnp.zeros((10, NR), dtype=jnp.float32)
    sub = sub.at[0:3, 0].set(1.0).at[3:5, 1].set(1.0).at[5:7, 2].set(1.0).at[7:9, 3].set(1.0).at[9, 4].set(1.0)
    return mult, id_c, id_r, slot, ridx, sub


def init_agent():
    return {
        'own_poss': jnp.ones((HS, NID), dtype=jnp.bool_),
        'own_prom': jnp.zeros((HS,), dtype=jnp.bool_),
        'own_clued': jnp.zeros((HS,), dtype=jnp.bool_),
        'par_poss': jnp.ones((HS, NID), dtype=jnp.bool_),
        'par_prom': jnp.zeros((HS,), dtype=jnp.bool_),
        'par_clued': jnp.zeros((HS,), dtype=jnp.bool_),
        'own_flags': jnp.zeros((HS, 10), dtype=jnp.bool_),
        'par_flags': jnp.zeros((HS, 10), dtype=jnp.bool_),
        'midgame': jnp.zeros((), dtype=jnp.bool_),
    }


def _reach_set(acc, height):
    MULT, ID_C, ID_R, SLOT, RIDX, _SUB = _consts()
    """Identities reachable as (delayed) plays given accounted connectors."""
    acc2 = acc.reshape(NC, NR)
    ok = acc2 | (RIDX < height[:, None])
    notok = ~ok
    reach = jnp.where(notok.any(axis=1), jnp.argmax(notok, axis=1), NR)
    p = (RIDX >= height[:, None]) & (RIDX <= reach[:, None])
    return p.reshape(NID)


def _interpret(pre_clued, occ, touched, is_rank, val, lit_after, poss_prev, prom_prev,
               height, useful, critical, acc_a):
    """Level-1 reading of a clue from the receiver's perspective."""
    MULT, ID_C, ID_R, SLOT, RIDX, _SUB = _consts()
    new = touched & ~pre_clued & occ
    any_new = new.any()
    unclued_occ = occ & ~pre_clued
    has_chop = unclued_occ.any()
    chop = jnp.argmax(unclued_occ)
    max_touched = jnp.max(jnp.where(touched, SLOT, -1))
    max_new = jnp.max(jnp.where(new, SLOT, -1))
    chop_focus = any_new & has_chop & new[chop]
    focus = jnp.where(~any_new, max_touched, jnp.where(chop_focus, chop, max_new))
    focus = jnp.clip(focus, 0, HS - 1)
    fo = SLOT == focus
    clued_after = (pre_clued | touched) & occ
    others = clued_after & ~fo

    base = poss_prev & lit_after
    pp = base & useful[None, :]
    # Good Touch: newly touched cards are not duplicates of exactly-known
    # previously clued cards in the same hand (the giver sees both).
    pre_others = pre_clued & occ & ~fo
    exact = pre_others & (pp.sum(axis=1) == 1)
    known_other = (pp & exact[:, None]).any(axis=0)
    base = jnp.where(new[:, None], base & ~known_other[None, :], base)
    pp = base & useful[None, :]

    acc_b = (pp & others[:, None]).any(axis=0)
    p_a = _reach_set(acc_a, height)
    p_ab = _reach_set(acc_a | acc_b, height)
    s5 = (ID_R == 4) & useful
    s2 = ((ID_R == 1) & useful) | critical
    s_set = jnp.where(is_rank & (val == 4), s5, jnp.where(is_rank & (val == 1), s2, critical))
    s_set = s_set & chop_focus

    flit = base[focus]
    direct = flit & p_a & useful
    save = flit & s_set
    sp_full = flit & p_ab & useful
    has_direct = direct.any()
    play_set = jnp.where(has_direct, direct, sp_full)
    fposs = play_set | save
    empty = ~fposs.any()
    fposs = jnp.where(empty, flit & useful, fposs)
    prom_f = play_set.any() & ~empty

    # self-prompt: needed connector per colour = first rank not accounted
    sp = ~has_direct & ~save.any() & sp_full.any()
    miss = (~acc_a.reshape(NC, NR)) & (RIDX >= height[:, None])
    fm = jnp.where(miss.any(axis=1), jnp.argmax(miss, axis=1), NR)
    full2 = sp_full.reshape(NC, NR)
    maxr = jnp.max(jnp.where(full2, RIDX, -1), axis=1)
    need = ((RIDX == fm[:, None]) & (fm[:, None] < maxr[:, None])).reshape(NID)
    compat = others & (pp & need[None, :]).any(axis=1)
    prompted = jnp.max(jnp.where(compat, SLOT, -1))
    do_prompt = sp & compat.any()
    pm = (SLOT == prompted) & do_prompt

    # newly touched cards: Good Touch says they are useful now; previously
    # clued cards keep their notes (identities may have gone dead since).
    poss = jnp.where(new[:, None], pp, base)
    poss = jnp.where(fo[:, None], fposs[None, :], poss)
    poss = jnp.where(pm[:, None], poss & need[None, :], poss)
    prom = jnp.where(fo, prom_f, prom_prev) | pm
    info = {
        'focus': focus, 'chop_focus': chop_focus, 'fposs': fposs, 'new': new,
        'has_direct': has_direct, 'sp_full': sp_full, 'prompted': prompted,
        'do_prompt': do_prompt, 'fm': fm, 'save': save,
    }
    return poss, prom, info


def _maintain(poss, prom, clued, occ, lit, not_played):
    """Per-turn note upkeep.  Only identities already on the stacks are dropped
    (Good Touch); identities that became dead through discards stay possible."""
    MULT, ID_C, ID_R, SLOT, RIDX, _SUB = _consts()
    poss = jnp.where(clued[:, None], poss & lit & not_played[None, :], lit)
    poss = jnp.where(occ[:, None], poss, True)
    emp = clued & occ & ~poss.any(axis=1)
    # broken promise / possible duplicate: fall back to literal knowledge only
    poss = jnp.where(emp[:, None], lit, poss)
    prom = prom & clued & occ & ~emp
    return poss, prom


def _shift(poss, prom, k):
    MULT, ID_C, ID_R, SLOT, RIDX, _SUB = _consts()
    src = jnp.clip(jnp.where(SLOT >= k, SLOT + 1, SLOT), 0, HS - 1)
    poss = poss[src]
    prom = prom[src]
    last = SLOT == HS - 1
    poss = jnp.where(last[:, None], True, poss)
    prom = jnp.where(last, False, prom)
    return poss, prom


def agent_act(obs, reset, memory, rng):
    MULT, ID_C, ID_R, SLOT, RIDX, _SUB = _consts()
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    f = obs.features
    legal = obs.legal_actions

    # ---------------- decode observation ----------------
    p_hand = f[0:125].reshape(HS, NID) > 0.5
    p_occ = p_hand.any(axis=1)
    p_idx = jnp.argmax(p_hand, axis=1)
    fw = f[167:192].reshape(NC, NR)
    height = jnp.sum(fw * (jnp.arange(NR)[None, :] + 1), axis=1).astype(jnp.int32)
    tokens = jnp.sum(f[192:200]).astype(jnp.int32)
    disc = f[203:253].reshape(NC, 10)
    discarded = jnp.round(disc @ _SUB).astype(jnp.int32)
    la = f[253:308]
    know = f[308:658].reshape(2, HS, 35)
    lit_own_raw = know[0, :, :25] > 0
    own_clued_now = (know[0, :, 25:35] > 0.5).any(axis=1)
    own_occ = lit_own_raw.any(axis=1)
    lit_par = know[1, :, :25] > 0
    par_clued_now = (know[1, :, 25:35] > 0.5).any(axis=1)

    h = height[:, None]
    # literal refinement: identities whose every copy is visible (partner hand,
    # discards, stacks) cannot be in my hand
    on_stack = (RIDX < h).astype(jnp.int32).reshape(NID)
    vis = p_hand.astype(jnp.int32).sum(axis=0) + discarded.reshape(NID) + on_stack
    remaining = MULT[ID_R] - vis
    lit_own = lit_own_raw & (remaining > 0)[None, :]
    lit_own = jnp.where(own_occ[:, None], lit_own, False)
    dead = discarded >= MULT[None, :]
    bad = dead & (RIDX >= h)
    useful2 = (RIDX >= h) & ~(jnp.cumsum(bad, axis=1) > 0)
    playable2 = RIDX == h
    critical2 = useful2 & (discarded == MULT[None, :] - 1)
    useful = useful2.reshape(NID)
    playable = playable2.reshape(NID)
    critical = critical2.reshape(NID)

    # ---------------- last action ----------------
    act_me = la[0] > 0.5
    act_par = la[1] > 0.5
    is_play = la[2] > 0.5
    is_disc = la[3] > 0.5
    is_ch = la[4] > 0.5
    is_rh = la[5] > 0.5
    rec_me = la[6] > 0.5
    rec_par = la[7] > 0.5
    hcol = jnp.argmax(la[8:13])
    hrank = jnp.argmax(la[13:18])
    match = la[18:23] > 0.5
    removed = jnp.argmax(la[23:28])
    is_hint = is_ch | is_rh
    hval = jnp.where(is_rh, hrank, hcol)

    own_poss, own_prom = mem['own_poss'], mem['own_prom']
    par_poss, par_prom = mem['par_poss'], mem['par_prom']

    own_rem = (is_play | is_disc) & act_me
    par_rem = (is_play | is_disc) & act_par
    op_s, opr_s = _shift(own_poss, own_prom, removed)
    own_poss = jnp.where(own_rem, op_s, own_poss)
    own_prom = jnp.where(own_rem, opr_s, own_prom)
    pp_s, ppr_s = _shift(par_poss, par_prom, removed)
    par_poss = jnp.where(par_rem, pp_s, par_poss)
    par_prom = jnp.where(par_rem, ppr_s, par_prom)

    # a repeat clue that adds no literal information (every touched card
    # already carried this flag, nothing new) is treated as a stall: no inference
    own_flags_now = know[0, :, 25:35] > 0.5
    par_flags_now = know[1, :, 25:35] > 0.5
    flag_col = jnp.where(is_rh, 5 + hrank, hcol)
    zero_me = match.any() & ~(match & ~mem['own_flags'][:, flag_col]).any()
    zero_par = match.any() & ~(match & ~mem['par_flags'][:, flag_col]).any()

    # hint from partner to me
    hint_me = is_hint & act_par & rec_me & ~zero_me
    par_clued_ids = (p_hand & (par_clued_now & p_occ)[:, None]).any(axis=0)
    poss_h, prom_h, _ = _interpret(mem['own_clued'], own_occ, match, is_rh, hval, lit_own,
                                   own_poss, own_prom, height, useful, critical, par_clued_ids)
    own_poss = jnp.where(hint_me, poss_h, own_poss)
    own_prom = jnp.where(hint_me, prom_h, own_prom)

    # hint from me to partner: update partner-notes model
    hint_par = is_hint & act_me & rec_par & ~zero_par
    own_eff = own_poss & lit_own & useful[None, :]
    own_wide = (own_eff & (own_clued_now & own_occ)[:, None]).any(axis=0)
    poss_hp, prom_hp, _ = _interpret(mem['par_clued'], p_occ, match, is_rh, hval, lit_par,
                                     par_poss, par_prom, height, useful, critical, own_wide)
    par_poss = jnp.where(hint_par, poss_hp, par_poss)
    par_prom = jnp.where(hint_par, prom_hp, par_prom)

    not_played = (RIDX >= h).reshape(NID)
    own_poss, own_prom = _maintain(own_poss, own_prom, own_clued_now, own_occ, lit_own, not_played)
    par_poss, par_prom = _maintain(par_poss, par_prom, par_clued_now, p_occ, lit_par, not_played)
    midgame = mem['midgame'] | is_disc

    new_mem = {
        'own_poss': own_poss, 'own_prom': own_prom, 'own_clued': own_clued_now,
        'par_poss': par_poss, 'par_prom': par_prom, 'par_clued': par_clued_now,
        'own_flags': own_flags_now, 'par_flags': par_flags_now,
        'midgame': midgame,
    }

    # ---------------- decision ----------------
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    pe = own_poss & lit_own & useful[None, :]
    nonempty = pe.any(axis=1)
    # Play certainty: promised cards use convention notes; other clued cards use
    # notes plus the Good-Touch "useful" filter while lives remain, but literal
    # knowledge only at one life; unclued cards use literal knowledge only.
    pe_conv = own_poss & lit_own
    use_conv = own_clued_now & (own_prom | (lives > 2))
    pe_play = jnp.where(use_conv[:, None], pe_conv, lit_own)
    can_play = own_occ & pe_play.any(axis=1) & ~(pe_play & ~playable[None, :]).any(axis=1)
    min_rank = jnp.min(jnp.where(pe, ID_R[None, :], 9), axis=1)
    succ_ids = jnp.concatenate([par_clued_ids[1:], jnp.zeros((1,), dtype=jnp.bool_)]) & (ID_R < 4)
    succ_bonus = (pe & succ_ids[None, :]).any(axis=1)
    play_score = jnp.where(can_play, 100.0 - 10.0 * min_rank + 5.0 * succ_bonus + 3.0 * own_prom + 0.1 * SLOT, NEG)
    best_play = jnp.argmax(play_score)
    have_play = can_play.any() & legal[5 + best_play]
    trash = own_occ & ~nonempty

    own_exact = (pe & (own_clued_now & own_occ & (pe.sum(axis=1) == 1))[:, None]).any(axis=0)
    own_wide2 = (pe & (own_clued_now & own_occ)[:, None]).any(axis=0)
    p_pre_ids = par_clued_ids
    p_color = ID_C[p_idx]
    p_rank = ID_R[p_idx]

    unclued_p = p_occ & ~par_clued_now
    has_chop_p = unclued_p.any()
    chop_p = jnp.argmax(unclued_p)
    cid = p_idx[chop_p]
    other_copy = (p_hand[:, cid] & (SLOT != chop_p) & p_occ).any()
    needs_save_chop = has_chop_p & (
        critical[cid]
        | ((ID_R[cid] == 1) & useful[cid] & ~other_copy & ~own_exact[cid] & ~p_pre_ids[cid])
        | (playable[cid] & ~other_copy & ~p_pre_ids[cid]))

    def eval_clue(k):
        is_rank = k >= 5
        val = k % 5
        touched = p_occ & jnp.where(is_rank, p_rank == val, p_color == val)
        legal_k = legal[10 + k]
        consistent = jnp.where(is_rank, ID_R == val, ID_C == val)
        lit_after = jnp.where(touched[:, None], lit_par & consistent[None, :], lit_par & ~consistent[None, :])
        poss_m, prom_m, info = _interpret(par_clued_now, p_occ, touched, is_rank, val, lit_after,
                                          par_poss, par_prom, height, useful, critical, own_wide2)
        focus = info['focus']
        chop_focus = info['chop_focus']
        fposs_m = info['fposs']
        new = info['new']
        tid = p_idx[focus]
        tc = ID_C[tid]
        tr = ID_R[tid]
        # direct validity via my exactly-known cards only
        p_exact = _reach_set(own_exact, height)
        in_exact = p_exact[tid] & useful[tid]
        # self-prompt validity
        fm_c = info['fm'][tc]
        conn = tc * NR + jnp.clip(fm_c, 0, NR - 1)
        prompted = info['prompted']
        prompt_ok = (~info['has_direct']) & info['sp_full'][tid] & (tr == fm_c + 1) & info['do_prompt'] \
            & (p_idx[jnp.clip(prompted, 0, HS - 1)] == conn)
        valid_play = in_exact | prompt_ok
        # bad touches
        new_ids = p_hand & new[:, None]
        dup_within = new_ids.sum(axis=0) >= 2
        badc = new & (~useful[p_idx] | p_pre_ids[p_idx] | dup_within[p_idx] | own_exact[p_idx])
        n_bad = badc.sum()
        n_new = new.sum()
        n_new_useful = (new & ~badc).sum()
        other_new = new & (SLOT != focus)
        pm_eff = poss_m & lit_after & useful[None, :]
        m_known_play = pm_eff.any(axis=1) & ~(pm_eff & ~playable[None, :]).any(axis=1)
        n_extra_known = (other_new & m_known_play & playable[p_idx]).sum()
        n_extra_play = (other_new & playable[p_idx]).sum()
        immediate = fposs_m.any() & ~(fposs_m & ~playable).any()
        s5 = ID_R == 4
        s2 = (ID_R == 1) | critical
        s_k = jnp.where(is_rank & (val == 4), s5, jnp.where(is_rank & (val == 1), s2, critical))
        valid_save = chop_focus & s_k[tid] & useful[tid] & (
            critical[tid] | ((ID_R[tid] == 1) & ~other_copy & ~own_exact[tid] & ~p_pre_ids[tid]))
        save_override = chop_focus & needs_save_chop
        focus_bad = badc[focus]
        valid_play_f = valid_play & ((n_bad == 0) | save_override) & ~focus_bad
        valid = legal_k & touched.any() & (n_new > 0) & (valid_play_f | valid_save) & useful[tid]
        is_color = ~is_rank
        # soft penalty: touched identity might duplicate one of my own clued cards
        n_risky = (new & own_wide2[p_idx] & ~own_exact[p_idx]).sum()
        play_sc = (100.0 + 40.0 * immediate + 25.0 * n_extra_known + 10.0 * n_extra_play
                   + 6.0 * n_new_useful + 3.0 * is_color - 70.0 * n_bad + 20.0 * save_override - 12.0 * n_risky)
        save_sc = 60.0 + 3.0 * is_color + 6.0 * n_new_useful - 50.0 * n_bad + 10.0 * critical[tid] - 12.0 * n_risky
        # literal fill-in (tempo) clue: re-touch only, and the focus card becomes
        # literally known playable; safe under any convention reading.
        retouch = touched & par_clued_now & p_occ
        flit_after = lit_after[focus]
        tempo_known = flit_after.any() & ~(flit_after & ~playable).any()
        adds_info = (retouch & ~par_flags_now[:, k]).any()
        valid_tempo = legal_k & (n_new == 0) & retouch.any() & tempo_known & playable[tid] & (tokens >= 2) \
            & ~own_exact[tid] & adds_info
        other_known = retouch & (SLOT != focus) & ~(lit_after & ~playable[None, :]).any(axis=1) & playable[p_idx]
        tempo_sc = 55.0 + 3.0 * is_color + 10.0 * other_known.sum() - 15.0 * own_wide2[tid]
        score = jnp.where(valid_play_f, play_sc, jnp.where(valid_save, save_sc, jnp.where(valid_tempo, tempo_sc, NEG)))
        score = jnp.where(valid | valid_tempo, score, NEG)
        # forced-clue fallback (cannot discard, nothing valid).  Safe when the
        # focus truly is playable, or when no Level-1 play reading exists at all
        # (partner holds); anything else risks a misplay.
        tempo = ~new.any() & touched.any()
        no_play_reading = (~info['has_direct']) & (~info['sp_full'].any())
        safe = playable[tid] | no_play_reading
        fb = (40.0 * safe + 30.0 * (safe & playable[tid]) + 40.0 * tempo - 200.0 * (~safe)
              - 10.0 * n_new - 100.0 * n_bad + 3.0 * is_color + 15.0 * (is_rank & (val == 4)))
        fb = jnp.where(legal_k & touched.any(), fb, NEG)
        # token reserve: with a single token left, only saves and immediate
        # play clues are worth it (keeps a clue available for the next save)
        keep = (tokens >= 2) | valid_save | (valid_play_f & immediate)
        score = jnp.where(keep, score, NEG)
        return score, fb

    scores, fbs = jax.vmap(eval_clue)(jnp.arange(10))
    best_k = jnp.argmax(scores)
    have_clue = (scores[best_k] > 0.0) & (tokens > 0)
    fb_k = jnp.argmax(fbs)

    dlegal = legal[0:5]
    # value of clued cards: fraction of possible identities that are critical
    cnt = jnp.maximum(pe.sum(axis=1), 1)
    crit_frac = (pe & critical[None, :]).sum(axis=1) / cnt
    crit_frac = jnp.where(own_clued_now & ~trash, crit_frac, 0.0)
    disc_score = (50.0 * trash + 20.0 * (~own_clued_now) - 15.0 * own_prom - 1.0 * SLOT - 40.0 * crit_frac)
    disc_score = jnp.where(dlegal & own_occ, disc_score, NEG)
    disc_slot = jnp.argmax(disc_score)
    can_discard = (dlegal & own_occ).any()
    # all-clued hand with only valuable cards: prefer a safe stall clue
    bad_discard = can_discard & ~trash.any() & ~(own_occ & ~own_clued_now).any() \
        & ((crit_frac[disc_slot] > 0.34) | own_prom[disc_slot])

    danger_now = needs_save_chop & (tokens > 0) & have_clue
    clue_act = 10 + best_k
    fb_act = 10 + fb_k
    fb_safe = (fbs[fb_k] > 0.0) & (tokens > 0)
    use_discard = can_discard & ~(bad_discard & fb_safe)
    action = jnp.where(danger_now, clue_act,
             jnp.where(have_play, 5 + best_play,
             jnp.where(have_clue, clue_act,
             jnp.where(use_discard, disc_slot, fb_act))))
    decision = jnp.where(danger_now, 1, jnp.where(have_play, 2, jnp.where(have_clue, 3, jnp.where(use_discard, 4, 5))))
    # legality safety net: prefer any legal hint, then discard, then play
    order = jnp.concatenate([jnp.arange(10, 20), jnp.arange(0, 5), jnp.arange(5, 10)])
    safe_act = order[jnp.argmax(legal[order])]
    action = jnp.where(legal[action], action, safe_act)
    active = ~legal[20]
    action = jnp.where(active, action, 20).astype(jnp.int32)

    trace = {
        'decision': jnp.where(active, decision, 0).astype(jnp.int32),
        'best_clue_score': scores[best_k].astype(jnp.float32),
        'n_can_play': can_play.sum().astype(jnp.int32),
        'needs_save_chop': needs_save_chop,
    }
    return action, new_mem, trace
