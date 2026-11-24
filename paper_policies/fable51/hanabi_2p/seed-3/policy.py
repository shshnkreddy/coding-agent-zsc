"""Rule-based two-player Hanabi policy built on literal (public) card knowledge.

Design goals (cross-play objective):
  * Every play decision is grounded in the literal belief implied by public hints
    and visible card counts, with a modest prior that a freshly hinted card is
    meant to be playable.
  * Hints are chosen to make partner cards *literally* known-playable whenever
    possible, and hints that newly touch unplayable cards are penalised so that
    a partner using any reasonable "hinted card => playable" reading is safe.
  * Discards prefer literally useless cards, then the oldest untouched card.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_rule_v1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

F32 = jnp.float32
I32 = jnp.int32
MULT_LIST = (3.0, 2.0, 2.0, 2.0, 1.0)
EPS = 1e-6

# Tunables ---------------------------------------------------------------
HINT_PRIOR_LAMBDA = 0.25      # likelihood ratio: partner hints unplayable vs playable card
LAMBDA_AFTER_1 = 0.6          # trust after one failed play by the receiving seat
LAMBDA_AFTER_2 = 1.0          # no trust after two failed plays (literal probabilities only)
TRUST_SUCC_1 = 2              # successful uncertain hinted plays needed to raise trust
LAMBDA_TRUSTED_1 = 0.15
TRUST_SUCC_2 = 4
LAMBDA_TRUSTED_2 = 0.10
AMBIG_BASE = 1.0              # penalty for newly touching an unplayable card: base + slope*post
AMBIG_SLOPE = 5.0
PLAY_THR_3LIVES = 0.80
PLAY_THR_2LIVES = 0.90
PLAY_THR_ENDGAME = 0.50       # deck empty, lives >= 2
HINT_THR_HIGH_TOKENS = 1.0    # tokens >= 6: hint on almost any useful information
HINT_THR_MID_TOKENS = 2.5     # tokens 3..5
HINT_THR_LOW_TOKENS = 4.0     # tokens 1..2: keep tokens for strong/save hints
STEP_GUARD = 96               # after this many transitions, stop giving weak hints
ENDGAME_DECK = 5              # deck size at/below which weak hints are preferred to discards
HINT_THR_ENDGAME = 0.5
CHOP_BONUS = 2.0              # preference for discarding the oldest untouched card


def init_agent():
    # row 0: "freshly hinted" flags per own slot; row 1: previous positive-hint touch flags
    # row 2: [step counter, my failed plays, partner failed plays, my last play was trusted,
    #         my successful trusted plays]
    # row 3: 25-bit hint-time playable set per own slot
    # row 4: partner slots touched but not literally certain (as of previous call)
    # row 5: [0] partner successful plays from touched-uncertain slots
    return jnp.zeros((6, 5), dtype=I32)


def _belief_probs(mask, counts, target):
    """mask/counts/target: (..., 5, 5) -> probability that card is in target set."""
    w = mask.astype(F32) * jnp.maximum(counts, 0.0)
    wsum = jnp.sum(w, axis=(-2, -1))
    num = jnp.sum(w * target.astype(F32), axis=(-2, -1))
    return num / jnp.maximum(wsum, EPS), wsum


def _hand_beliefs(mask, base, playable, useless, critical):
    """mask: (..., 5 slots, 5, 5) bool; base: (5, 5) counts.

    Returns (p_play, p_useless, p_crit) of shape (..., 5). Fully determined cards in the
    same hand are removed from the counts available to the other slots.
    """
    m = mask.astype(F32)
    n_poss = jnp.sum(m, axis=(-2, -1))
    determined = n_poss == 1
    det = jnp.where(determined[..., None, None], m, 0.0)
    det_total = jnp.sum(det, axis=-3, keepdims=True)
    base = jnp.maximum(base, 0.0)
    cnt = base - (det_total - det)
    w = m * jnp.maximum(cnt, 0.0)
    wsum = jnp.sum(w, axis=(-2, -1))
    w = jnp.where((wsum > 0.5)[..., None, None], w, m * base)
    wsum = jnp.maximum(jnp.sum(w, axis=(-2, -1)), EPS)
    p_play = jnp.sum(w * playable.astype(F32), axis=(-2, -1)) / wsum
    p_useless = jnp.sum(w * useless.astype(F32), axis=(-2, -1)) / wsum
    p_crit = jnp.sum(w * critical.astype(F32), axis=(-2, -1)) / wsum
    return p_play, p_useless, p_crit, w / wsum[..., None, None]


def _trust_eff(wn, hset, playable, flag, lam):
    """Effective playability with a trust prior that the card lies in the hinted set.

    wn: normalised weights (5, 5, 5); hset: (5, 5, 5) bool hint-time playable set;
    flag: (5,) whether a trust hint applies.
    """
    pl = playable.astype(F32)[None]
    h = hset.astype(F32)
    p_h = jnp.sum(wn * h, axis=(-2, -1))
    p_hpl = jnp.sum(wn * h * pl, axis=(-2, -1))
    p_pl_noth = jnp.sum(wn * (1.0 - h) * pl, axis=(-2, -1))
    p_pl = p_hpl + p_pl_noth
    q = p_h / jnp.maximum(p_h + (1.0 - p_h) * lam, EPS)
    eff = q * p_hpl / jnp.maximum(p_h, EPS) + (1.0 - q) * p_pl_noth / jnp.maximum(1.0 - p_h, EPS)
    return jnp.where(flag > 0, eff, p_pl)


def agent_act(obs, reset, memory, rng):
    f = obs.features.astype(F32)
    legal = obs.legal_actions
    memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    flag = memory[0]
    prev_touched = memory[1]

    idx5 = jnp.arange(5)
    ranks = idx5[None, :]
    MULT = jnp.asarray(MULT_LIST, dtype=F32)

    # ---------------- decode observation ----------------
    partner = f[0:125].reshape(5, 5, 5)                      # slot, color, rank
    deck = jnp.sum(f[127:167])
    fw = f[167:192].reshape(5, 5)
    height = jnp.sum(fw * jnp.arange(1, 6, dtype=F32)[None, :], axis=1)   # (5,) in 0..5
    tokens = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    disc = f[203:253].reshape(5, 10)
    disc_cnt = jnp.stack([disc[:, 0:3].sum(1), disc[:, 3:5].sum(1), disc[:, 5:7].sum(1),
                          disc[:, 7:9].sum(1), disc[:, 9:10].sum(1)], axis=1)  # (color, rank)
    last = f[253:308]
    know = f[308:658].reshape(2, 5, 35)

    played = (ranks < height[:, None]).astype(F32)           # (c, r)
    remaining = MULT[None, :] - disc_cnt - played            # public remaining copies
    playable = (ranks == height[:, None])                    # (c, r)
    zero_rem = remaining <= 0.5
    z = (zero_rem & (ranks >= height[:, None])).astype(I32)
    dead_before = (jnp.cumsum(z, axis=1) - z) > 0
    useless = (ranks < height[:, None]) | dead_before        # (c, r)
    critical = (remaining <= 1.5) & (~useless)               # last copy still needed

    # ---------------- memory update from last public action ----------------
    actor_me = last[0] > 0.5
    is_play = last[2] > 0.5
    is_discard = last[3] > 0.5
    is_hint = (last[4] + last[5]) > 0.5
    recipient_me = last[6] > 0.5
    matches = last[18:23] > 0.5
    removed = jnp.argmax(last[23:28])
    hbits = memory[3]
    nxt = jnp.minimum(idx5 + 1, 4)
    i_removed = actor_me & (is_play | is_discard)
    # per-seat play-outcome counters (public events) drive adaptive trust in hint intent
    failed_play = is_play & (last[53] < 0.5)
    success_play = is_play & (last[53] > 0.5)
    actor_partner = last[1] > 0.5
    my_bombs = memory[2, 1] + jnp.where(failed_play & actor_me, 1, 0)
    partner_bombs = memory[2, 2] + jnp.where(failed_play & actor_partner, 1, 0)
    last_play_trusted = memory[2, 3] > 0
    my_trusted_succ = memory[2, 4] + jnp.where(success_play & actor_me & last_play_trusted, 1, 0)
    partner_unc_touched = memory[4] > 0          # partner slots touched but not literally certain (previous call)
    partner_trusted_succ = memory[5, 0] + jnp.where(success_play & actor_partner & partner_unc_touched[removed], 1, 0)
    partner_hinted_fail = memory[5, 1] + jnp.where(failed_play & actor_partner & partner_unc_touched[removed], 1, 0)
    shifted = jnp.where(idx5 >= removed, jnp.where(idx5 < 4, flag[nxt], 0), flag)
    flag = jnp.where(i_removed, shifted, flag)
    shifted_h = jnp.where(idx5 >= removed, jnp.where(idx5 < 4, hbits[nxt], 0), hbits)
    hbits = jnp.where(i_removed, shifted_h, hbits)
    hint_to_me = (~actor_me) & is_hint & recipient_me
    new_touch = matches & (prev_touched == 0)
    fresh = hint_to_me & new_touch
    cur_touched = (jnp.sum(know[0, :, 25:35], axis=1) > 0.5).astype(I32)
    step_count = memory[2, 0] + 1
    pow2 = (2 ** jnp.arange(25)).astype(I32)

    thr = jnp.where(lives <= 1.5, 0.999,
                    jnp.where(deck <= 0.5, PLAY_THR_ENDGAME,
                              jnp.where(lives <= 2.5, PLAY_THR_2LIVES, PLAY_THR_3LIVES)))

    def _lam_of(nb, ns):
        base = jnp.where(ns >= TRUST_SUCC_2, LAMBDA_TRUSTED_2,
                         jnp.where(ns >= TRUST_SUCC_1, LAMBDA_TRUSTED_1, HINT_PRIOR_LAMBDA))
        return jnp.where(nb >= 2, LAMBDA_AFTER_2, jnp.where(nb >= 1, LAMBDA_AFTER_1, base))

    lam = _lam_of(my_bombs, my_trusted_succ)                    # trust I place in hints I receive
    lam_partner = _lam_of(partner_bombs, partner_trusted_succ)  # trust the partner is assumed to place in mine

    # ---------------- my belief ----------------
    # public belief: literal mask x public counts (minus own determined cards); the partner
    # can compute exactly this, so hint intent can be evaluated consistently by both seats.
    # private belief: additionally removes the partner's visible cards from the counts.
    my_mask = know[0, :, :25].reshape(5, 5, 5) > 0.0         # s, c, r
    partner_cnt = jnp.sum(partner, axis=0)                   # (c, r)
    n_poss = jnp.sum(my_mask.reshape(5, 25), axis=1)
    occupied = n_poss > 0
    p_play_pub, p_useless_pub, p_crit_pub, wn_pub = _hand_beliefs(my_mask, remaining, playable, useless, critical)
    p_play_pr, p_useless_pr, p_crit_pr, wn_pr = _hand_beliefs(my_mask, remaining - partner_cnt, playable, useless, critical)
    p_useless = p_useless_pr
    p_crit = p_crit_pr

    # a freshly hinted card gets a trust flag (with the hint-time playable set) only if the
    # hint made it a trusted play under the public belief now (mirrors the hinter's evaluation).
    fresh_set = my_mask & playable[None]
    fresh_bits = jnp.sum(fresh_set.reshape(5, 25).astype(I32) * pow2[None, :], axis=1).astype(I32)
    cand_flag = jnp.where(fresh, 1, flag).astype(I32)
    cand_bits = jnp.where(fresh, fresh_bits, hbits).astype(I32)
    cand_set = ((cand_bits[:, None] // pow2[None, :]) % 2 > 0).reshape(5, 5, 5)
    eff_pub_cand = _trust_eff(wn_pub, cand_set, playable, cand_flag, lam)
    keep_fresh = eff_pub_cand >= thr
    flag = jnp.where(fresh, jnp.where(keep_fresh, 1, 0), flag).astype(I32)
    hbits = jnp.where(fresh, jnp.where(keep_fresh, fresh_bits, 0), hbits).astype(I32)
    hset = ((hbits[:, None] // pow2[None, :]) % 2 > 0).reshape(5, 5, 5)

    p_eff_pub = jnp.where(occupied, _trust_eff(wn_pub, hset, playable, flag, lam), 0.0)
    p_eff_pr = jnp.where(occupied, _trust_eff(wn_pr, hset, playable, flag, lam), 0.0)
    p_eff = jnp.minimum(p_eff_pub, p_eff_pr)
    p_play = p_play_pr

    # ---------------- play decision ----------------
    play_legal = legal[5:10]
    play_score = jnp.where(play_legal, p_eff - 0.001 * idx5, -1.0)
    best_play = jnp.argmax(play_score)
    do_play = (play_score[best_play] >= thr - 0.001 * best_play) & play_legal[best_play]

    # ---------------- hint evaluation ----------------
    pk_mask = know[1, :, :25].reshape(5, 5, 5) > 0.0         # partner literal mask
    p_touched = jnp.sum(know[1, :, 25:35], axis=1) > 0.5
    p_occ = jnp.sum(partner.reshape(5, 25), axis=1) > 0.5
    pre, pre_useless, pre_crit, _ = _hand_beliefs(pk_mask, remaining, playable, useless, critical)
    part_playable = jnp.sum(partner * playable[None], axis=(1, 2)) > 0.5
    part_critical = jnp.sum(partner * critical[None], axis=(1, 2)) > 0.5
    part_useless = jnp.sum(partner * useless[None], axis=(1, 2)) > 0.5
    part_rank = jnp.argmax(jnp.sum(partner, axis=1), axis=1)  # (5,)
    part_is5 = part_rank == 4
    untouched_occ = (~p_touched) & p_occ
    chop_idx = jnp.argmax(untouched_occ)                       # first untouched
    has_chop = jnp.any(untouched_occ)
    is_chop = (idx5 == chop_idx) & has_chop

    # attribute masks for the 10 hints: (10, c, r)
    color_attr = (jnp.arange(5)[:, None, None] == jnp.arange(5)[None, :, None]) & jnp.ones((1, 1, 5), bool)
    rank_attr = (jnp.arange(5)[:, None, None] == jnp.arange(5)[None, None, :]) & jnp.ones((1, 5, 1), bool)
    attr = jnp.concatenate([color_attr, rank_attr], axis=0)   # (10, 5, 5)
    touched = jnp.sum(partner[None] * attr[:, None], axis=(2, 3)) > 0.5   # (10, s)
    new_mask = pk_mask[None] & jnp.where(touched[:, :, None, None], attr[:, None], ~attr[:, None])
    post, post_useless, post_crit, _ = _hand_beliefs(new_mask, remaining, playable, useless, critical)  # (10, s)
    pre_b = pre[None]
    newly = touched & (~p_touched)[None]
    pl = part_playable[None]
    # what a partner using our own rule would do after this hint
    post_eff = jnp.where(newly, post / jnp.maximum(post + (1.0 - post) * lam_partner, EPS), post)
    would_play = post_eff >= thr
    already_known = pre_b >= 0.999
    known_gain = touched & pl & (post >= 0.999) & (~already_known)
    trust_play = touched & pl & would_play & (post < 0.999) & (~already_known)
    info_play = touched & pl & (~would_play) & (post > pre_b + 0.01)
    # Once the partner has misplayed a hinted-but-uncertain card, or when one wrong play
    # would end the game, treat any new touch of a possibly-playable unplayable card as a
    # wrong play (naive partners play freshly hinted cards).
    strict = (partner_hinted_fail >= 1) | (lives <= 1.5)
    bad_play = touched & (~pl) & (would_play | (strict & newly & (post > 0.001)))
    mis_new = newly & (~pl) & (~bad_play)
    save_hit = touched & is_chop[None] & part_critical[None] & (~pl)
    # literal information terms (help partner discard safely / avoid critical discards)
    avail = (remaining > 0.5)[None]
    n_pre = jnp.sum((pk_mask & avail).reshape(5, 25), axis=1)
    n_post = jnp.sum((new_mask & avail[None]).reshape(10, 5, 25), axis=2)
    info_bits = jnp.log2(jnp.maximum(n_pre[None], 1.0) / jnp.maximum(n_post, 1.0)) * p_occ[None]
    useless_reveal = (post_useless >= 0.999) & (pre_useless[None] < 0.999) & p_occ[None]
    crit_reveal = (post_crit >= 0.999) & (pre_crit[None] < 0.999) & part_critical[None] & (~pl)
    protect_hit = touched & (~is_chop[None]) & (~p_touched[None]) & part_critical[None] & (~pl) & (post <= 0.001)
    ambiguous = newly & (~pl) & (post > 0.15)
    score = (10.0 * known_gain + 8.0 * trust_play + 2.0 * info_play
             - 12.0 * bad_play
             - jnp.where(save_hit, 0.0, AMBIG_BASE + AMBIG_SLOPE * post) * mis_new
             + jnp.where(part_is5[None], 6.0, 5.0) * save_hit - 2.0 * post * save_hit
             + 2.5 * protect_hit
             + 2.0 * useless_reveal + 1.5 * crit_reveal + 0.4 * info_bits)
    hint_score = jnp.sum(score, axis=1) + jnp.concatenate([jnp.zeros(5), 0.2 * jnp.ones(5)])
    hint_legal = legal[10:20]
    hint_score = jnp.where(hint_legal, hint_score, -1e6)
    best_hint = jnp.argmax(hint_score)
    best_hint_score = hint_score[best_hint]
    comps = jnp.stack([jnp.sum(known_gain, 1), jnp.sum(trust_play, 1), jnp.sum(info_play, 1),
                       jnp.sum(ambiguous, 1), jnp.sum(mis_new, 1), jnp.sum(save_hit | protect_hit, 1)], axis=1)  # (10, 6)
    hcomp = comps[best_hint].astype(I32)

    # ---------------- discard decision ----------------
    disc_legal = legal[0:5]
    my_untouched = (cur_touched == 0) & occupied
    my_chop = jnp.argmax(my_untouched)
    chop_bonus = jnp.where((idx5 == my_chop) & jnp.any(my_untouched), CHOP_BONUS, 0.0)
    disc_score = (10.0 * p_useless - 8.0 * p_crit - 3.0 * p_eff
                  - 1.0 * (1.0 - p_useless - p_crit) + chop_bonus - 0.05 * idx5)
    disc_score = jnp.where(disc_legal, disc_score, -1e6)
    best_disc = jnp.argmax(disc_score)
    can_discard = disc_legal[best_disc]
    safe_discard = can_discard & (p_useless[best_disc] >= 0.999)

    # ---------------- hint vs discard ----------------
    # Hints cost only tokens (not deck); discards burn cards. Prefer strong hints, then
    # free (useless) discards, then informative hints while tokens are plentiful.
    hint_thr = jnp.where(tokens >= 5.5, HINT_THR_HIGH_TOKENS,
                         jnp.where(tokens >= 2.5, HINT_THR_MID_TOKENS, HINT_THR_LOW_TOKENS))
    step_guard = step_count >= STEP_GUARD    # keep well inside the 128-transition limit
    hint_thr = jnp.where(deck <= ENDGAME_DECK, jnp.minimum(hint_thr, HINT_THR_ENDGAME), hint_thr)
    hint_thr = jnp.where(step_guard, 7.5, hint_thr)
    hint_ok = hint_legal[best_hint]
    strong_hint = hint_ok & (best_hint_score >= 7.5)
    do_hint = strong_hint | (hint_ok & (~safe_discard) & (best_hint_score >= hint_thr)) | (hint_ok & (~can_discard))

    # ---------------- combine ----------------
    action = jnp.where(do_play, 5 + best_play,
                       jnp.where(do_hint, 10 + best_hint,
                                 jnp.where(can_discard, best_disc,
                                           jnp.where(hint_legal[best_hint], 10 + best_hint,
                                                     jnp.argmax(legal)))))
    action = jnp.where(legal[20], 20, action)
    action = jnp.where(legal[action], action, jnp.argmax(legal)).astype(I32)

    # ---------------- memory write ----------------
    acting = ~legal[20]
    playing_trusted = acting & do_play & (flag[best_play] > 0) & (p_play_pub[best_play] < 0.999)
    counter_row = jnp.where(idx5 == 0, step_count,
                            jnp.where(idx5 == 1, my_bombs,
                                      jnp.where(idx5 == 2, partner_bombs,
                                                jnp.where(idx5 == 3, jnp.where(playing_trusted, 1, 0),
                                                          my_trusted_succ)))).astype(I32)
    p_unc_row = (p_touched & (pre < 0.999) & p_occ).astype(I32)
    p_succ_row = jnp.where(idx5 == 0, partner_trusted_succ,
                           jnp.where(idx5 == 1, partner_hinted_fail, 0)).astype(I32)
    new_memory = jnp.stack([flag, cur_touched, counter_row, hbits, p_unc_row, p_succ_row], axis=0)
    mode = jnp.where(do_play, 1, jnp.where(do_hint, 2, jnp.where(can_discard, 3, 4))).astype(I32)
    revealed = last[28:53].reshape(5, 5)
    # critical/useless status of a discarded card, judged against pre-discard counts
    rem_before = remaining + revealed
    crit_before = (rem_before <= 1.5) & (~useless)
    disc_crit = is_discard & (jnp.sum(revealed * crit_before) > 0.5)
    disc_useless = is_discard & (jnp.sum(revealed * useless) > 0.5)
    bomb = is_play & (last[53] < 0.5)
    trace = {
        "p_eff": p_eff.astype(F32),
        "p_play": p_play.astype(F32),
        "hint_score": hint_score.astype(F32),
        "disc_score": disc_score.astype(F32),
        "mode": mode,
        "lives": lives.astype(F32),
        "tokens": tokens.astype(F32),
        "deck": deck.astype(F32),
        "score": jnp.sum(height).astype(F32),
        "bomb": bomb,
        "disc_crit": disc_crit,
        "disc_useless": disc_useless,
        "reset": reset,
        "actor_me": actor_me,
        "hcomp": hcomp,
        "p_crit": p_crit.astype(F32),
        "lam": lam.astype(F32),
        "lam_partner": lam_partner.astype(F32),
        "p_useless": p_useless.astype(F32),
    }
    return action, new_memory, trace
