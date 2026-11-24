"""Two-player Hanabi policy following H-group Level 1 conventions.

Fixed-shape JAX implementation. Private memory holds per-card convention
notes (25-entry possibility masks) for both hands plus the touched flags of
the previous observation, which are needed to determine clue focus with the
pre-clue hand (chop / newly touched / re-touched).

Convention summary implemented here:
  * chop = lowest occupied native index among unclued cards; finesse position
    and "leftmost" = highest occupied index.
  * clue focus: no new cards -> leftmost re-touched; chop touched -> chop;
    otherwise leftmost newly touched card.
  * non-chop focus = play clue (possibly delayed through cards accounted for
    in the giver's hand or the recipient's own known cards).
  * chop focus = play clue or save clue (5 save with rank 5, 2 save with rank
    2, critical save with colour or rank); the recipient keeps both readings.
  * Good Touch: notes exclude identities already played at clue time, the
    giver's known cards, and identities revealed by sibling clued cards.
  * Save Principle: 5s, unique 2s, critical cards and unique playable cards on
    the partner's chop are saved (play clue preferred when possible).
  * Minimum Clue Value: tempo clues only as a fallback with spare tokens.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_level1_rules', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

I32 = jnp.int32
MULT_LIST = [3, 2, 2, 2, 1]

# Tunable thresholds (open choices under the conventions).
TEMPO_MIN_TOKENS = 2      # allow a tempo clue (no new card) only with spare tokens
ENDGAME_RISK_P = 0.3      # final-round risky play probability threshold (lives >= 2)


class _K:
    """Constant arrays, built inside a function call (no module-scope calls)."""

    def __init__(self):
        card_c = [i // 5 for i in range(25)]
        card_r = [i % 5 for i in range(25)]
        mult25 = [MULT_LIST[i % 5] for i in range(25)]
        color_masks = [[(i // 5) == c for i in range(25)] for c in range(5)]
        rank_masks = [[(i % 5) == r for i in range(25)] for r in range(5)]
        self.CARD_C = jnp.asarray(card_c, dtype=I32)
        self.CARD_R = jnp.asarray(card_r, dtype=I32)
        self.MULT25 = jnp.asarray(mult25, dtype=I32)
        self.COLOR_MASKS = jnp.asarray(color_masks, dtype=jnp.bool_)
        self.RANK_MASKS = jnp.asarray(rank_masks, dtype=jnp.bool_)
        self.SLOTS = jnp.arange(5, dtype=I32)


def init_agent():
    return {
        "notes": jnp.ones((2, 5, 25), dtype=jnp.bool_),
        "touched": jnp.zeros((2, 5), dtype=jnp.bool_),
        # how each card was first touched: 0 none, 1 play clue (Good Touch
        # guaranteed), 2 save or forced clue (duplicates possible)
        "kind": jnp.zeros((2, 5), dtype=jnp.int32),
        "prev_deck": jnp.asarray(40, dtype=jnp.int32),
        "last_drawer": jnp.asarray(-1, dtype=jnp.int32),   # -1 none, 0 me, 1 partner
    }


def _lowest(mask):
    return jnp.where(mask.any(), jnp.argmax(mask), -1).astype(I32)


def _highest(mask):
    return jnp.where(mask.any(), 4 - jnp.argmax(mask[::-1]), -1).astype(I32)


def _chain(accounted, stack):
    """Consecutive accounted ranks above each stack (delayed-play depth)."""
    acc = accounted.reshape(5, 5)
    chain = jnp.zeros(5, dtype=I32)
    ok = jnp.ones(5, dtype=jnp.bool_)
    for j in range(5):
        r = stack + j
        a = acc[jnp.arange(5), jnp.minimum(r, 4)] & (r < 5)
        ok = ok & a
        chain = chain + ok.astype(I32)
    return chain


def _interpret(K, matched, touched_before, occ, is_color, value, stack, useful,
               critical, chain, exclude, old_notes):
    """Level 1 clue interpretation from the recipient's perspective.

    Returns updated notes for the recipient hand, focus slot, chop-focus flag,
    has_new flag and the newly touched mask.
    """
    attr = jnp.where(is_color, K.COLOR_MASKS[value], K.RANK_MASKS[value])
    matched = matched & occ
    new = matched & ~touched_before
    has_new = new.any()
    unclued = occ & ~touched_before
    chop = _lowest(unclued)
    chop_touched = unclued.any() & matched[jnp.maximum(chop, 0)] & (chop >= 0)
    leftmost_new = _highest(new)
    leftmost_re = _highest(matched)
    focus = jnp.where(~has_new, leftmost_re, jnp.where(chop_touched, chop, leftmost_new))
    focus = jnp.maximum(focus, 0)
    is_chop_focus = has_new & chop_touched
    rel = K.CARD_R - stack[K.CARD_C]
    playc = attr & useful & (rel >= 0) & (rel <= chain[K.CARD_C])
    save_c = critical & attr & (K.CARD_R != 4)
    save_5 = (~is_color) & (value == 4) & (K.CARD_R == 4) & useful
    save_2 = (~is_color) & (value == 1) & (K.CARD_R == 1) & useful
    savec = save_c | save_5 | save_2
    focus_note = jnp.where(is_chop_focus, playc | savec, playc)
    fn_ex = focus_note & ~exclude
    focus_note = jnp.where(fn_ex.any(), fn_ex, focus_note)
    fallback = attr & useful
    fb_ex = fallback & ~exclude
    fallback = jnp.where(fb_ex.any(), fb_ex, fallback)
    focus_note = jnp.where(focus_note.any(), focus_note, fallback)
    old_f = old_notes[focus]
    fn = old_f & focus_note
    fn = jnp.where(fn.any(), fn, old_f & attr)
    fn = jnp.where(fn.any(), fn, fallback)
    # Newly touched non-focus cards: useful at clue time (Good Touch), not the
    # giver's known cards. Re-touched non-focus cards: the clue could not
    # avoid them, so only the literal attribute narrows the old note.
    other_newc = old_notes & attr[None, :] & useful[None, :]
    other_ex = other_newc & ~exclude[None, :]
    other_newc = jnp.where(other_ex.any(1, keepdims=True), other_ex, other_newc)
    other_re = old_notes & attr[None, :]
    other = jnp.where(new[:, None], other_newc, other_re)
    other = jnp.where(other.any(1, keepdims=True), other, old_notes & attr[None, :])
    other = jnp.where(other.any(1, keepdims=True), other, fallback[None, :])
    per_slot = jnp.where((K.SLOTS == focus)[:, None], fn[None, :], other)
    new_notes = jnp.where(matched[:, None], per_slot, old_notes)
    return new_notes, focus, is_chop_focus, has_new, new


def _effective(notes, lit, touched):
    """Candidate identities: note ∩ literal for clued cards, literal otherwise.

    Falls back to the literal set if the note contradicts literal knowledge.
    """
    eff = jnp.where(touched[:, None], notes & lit, lit)
    return jnp.where(eff.any(1, keepdims=True), eff, lit)


def agent_act(obs, reset, memory, rng):
    K = _K()
    CARD_C, CARD_R, MULT25, SLOTS = K.CARD_C, K.CARD_R, K.MULT25, K.SLOTS
    f = obs.features
    legal = obs.legal_actions
    init = init_agent()
    notes = jnp.where(reset, init["notes"], memory["notes"])
    touched_prev = jnp.where(reset, init["touched"], memory["touched"])
    kind = jnp.where(reset, init["kind"], memory["kind"])
    prev_deck = jnp.where(reset, init["prev_deck"], memory["prev_deck"])
    last_drawer = jnp.where(reset, init["last_drawer"], memory["last_drawer"])

    # ---------------- board parsing ----------------
    ph = f[0:125].reshape(5, 25)
    p_occ = ph.sum(1) > 0
    p_id = jnp.argmax(ph, axis=1).astype(I32)
    p_onehot = ph > 0                                   # [5, 25]
    deck = jnp.sum(f[127:167]).astype(I32)
    fw = f[167:192].reshape(5, 5)
    stack = jnp.where(fw.sum(1) > 0, jnp.argmax(fw, 1) + 1, 0).astype(I32)
    tokens = jnp.sum(f[192:200]).astype(I32)
    lives = jnp.sum(f[200:203]).astype(I32)
    d = f[203:253].reshape(5, 10)
    disc = jnp.stack([d[:, 0:3].sum(1), d[:, 3:5].sum(1), d[:, 5:7].sum(1),
                      d[:, 7:9].sum(1), d[:, 9:10].sum(1)], axis=1).astype(I32).reshape(25)
    know = f[308:658].reshape(2, 5, 35)
    lit = know[:, :, 0:25] > 0                          # [2, 5, 25]
    occ = know[:, :, 0:25].sum(2) > 0                   # [2, 5]
    touched_now = know[:, :, 25:35].sum(2) > 0          # [2, 5]
    occ_me = occ[0]
    touched_me = touched_now[0]
    touched_p = touched_now[1]

    # last action
    la = f[253:308]
    la_valid = la[0:2].sum() > 0
    la_actor = jnp.argmax(la[0:2]).astype(I32)
    is_play = la[2] > 0
    is_disc = la[3] > 0
    is_hc = la[4] > 0
    is_hr = la[5] > 0
    la_recip = jnp.argmax(la[6:8]).astype(I32)
    la_val = jnp.where(is_hc, jnp.argmax(la[8:13]), jnp.argmax(la[13:18])).astype(I32)
    la_matched = la[18:23] > 0
    la_slot = jnp.argmax(la[23:28]).astype(I32)
    revealed = la[28:53] > 0
    removal = la_valid & (is_play | is_disc) & ~reset
    hint = la_valid & (is_hc | is_hr) & ~reset

    # ---------------- memory: sibling elimination + shift on removal ----------------
    # Good Touch: clued cards in one hand hold distinct identities, so a clued
    # card revealed as X removes X from the other clued cards of that hand.
    hand_notes = notes[la_actor]
    hand_kind = kind[la_actor]
    was_touched = touched_prev[la_actor, la_slot]
    elim_rows = touched_prev[la_actor] & (SLOTS != la_slot) & (hand_kind == 1)
    hn2 = jnp.where(elim_rows[:, None] & revealed[None, :], False, hand_notes)
    hn2 = jnp.where(hn2.any(1, keepdims=True), hn2, hand_notes)
    hand_notes = jnp.where(removal & was_touched & revealed.any(), hn2, hand_notes)
    src = jnp.where(SLOTS >= la_slot, SLOTS + 1, SLOTS)
    shifted = jnp.where((src >= 5)[:, None], True, hand_notes[jnp.minimum(src, 4)])
    notes = jnp.where(removal, notes.at[la_actor].set(shifted), notes)
    shifted_kind = jnp.where(src >= 5, 0, hand_kind[jnp.minimum(src, 4)])
    kind = jnp.where(removal, kind.at[la_actor].set(shifted_kind), kind)

    # ---------------- public card status ----------------
    played = CARD_R < stack[CARD_C]
    fully = disc >= MULT25
    dead = jnp.full(5, 5, dtype=I32)
    fully2 = fully.reshape(5, 5)
    for r in reversed(range(5)):
        cond = fully2[:, r] & (r >= stack)
        dead = jnp.where(cond, r, dead)
    useful = (CARD_R >= stack[CARD_C]) & (CARD_R < dead[CARD_C])
    critical = useful & (disc == MULT25 - 1)
    playable_now = CARD_R == stack[CARD_C]
    partner_count = p_onehot.sum(0).astype(I32)
    remaining_base = MULT25 - disc - played.astype(I32) - partner_count   # for my own cards
    lit_me = lit[0] & (remaining_base > 0)[None, :]
    lit_p = lit[1]

    def derive(notes_):
        eff0 = _effective(notes_[0], lit_me, touched_me)
        single = (eff0.sum(1) == 1) & occ_me
        single_ids = single[:, None] & eff0                    # [5, 25]
        cnt = single_ids.sum(0).astype(I32)
        others = cnt[None, :] - single_ids.astype(I32)
        lit_me2 = lit_me & ((remaining_base[None, :] - others) > 0)
        eff_me = _effective(notes_[0], lit_me2, touched_me)
        eff_p = _effective(notes_[1], lit_p, touched_p)
        single_me = (eff_me.sum(1) == 1) & occ_me
        my_single = (single_me[:, None] & eff_me).any(0)         # [25]
        my_single_touched = ((single_me & touched_me)[:, None] & eff_me).any(0)
        single_p = (eff_p.sum(1) == 1) & p_occ
        p_single_true = single_p & (eff_p[SLOTS, p_id])            # partner knows exactly & correct
        p_single_touched = ((p_single_true & touched_p)[:, None] & p_onehot).any(0)
        # Delayed-play connectors: only cards whose holder knows them exactly
        # AND whose knowledge the partner can verify from public information
        # (my notes ∩ my raw native knowledge, without private visible-card
        # subtraction). A giver cannot promise a connection through a card the
        # recipient cannot see is known.
        eff_pub = _effective(notes_[0], lit[0], touched_me)
        single_pub = (eff_pub.sum(1) == 1) & occ_me & touched_me
        my_single_pub = (single_pub[:, None] & eff_pub).any(0)
        acc_p = my_single_pub | p_single_touched
        chain_p = _chain(acc_p, stack)
        chain_me = chain_p
        exclude = my_single_pub | p_single_touched
        return eff_me, eff_p, my_single, chain_me, chain_p, lit_me2, exclude

    eff_me, eff_p, my_single, chain_me, chain_p, lit_me2, exclude = derive(notes)

    # ---------------- memory: process a hint ----------------
    n0, _, cf0, _, new0 = _interpret(K, la_matched, touched_prev[0], occ_me, is_hc, la_val,
                                     stack, useful, critical, chain_me, exclude, notes[0])
    n1, _, cf1, _, new1 = _interpret(K, la_matched, touched_prev[1], p_occ, is_hc, la_val,
                                     stack, useful, critical, chain_p, exclude, notes[1])
    notes_h = jnp.where(la_recip == 0, notes.at[0].set(n0), notes.at[1].set(n1))
    notes = jnp.where(hint, notes_h, notes)
    eff_me, eff_p, my_single, chain_me, chain_p, lit_me2, exclude = derive(notes)
    # clue kind for newly touched cards: chop-focused or forced (8 tokens) clues
    # may violate Good Touch, so mark them as kind 2
    forced_clue = (tokens + 1) >= 8
    k0 = jnp.where(new0, jnp.where(cf0 | forced_clue, 2, 1), kind[0])
    k1 = jnp.where(new1, jnp.where(cf1 | forced_clue, 2, 1), kind[1])
    kind_h = jnp.where(la_recip == 0, kind.at[0].set(k0), kind.at[1].set(k1))
    kind = jnp.where(hint, kind_h, kind)

    last_drawer = jnp.where(removal & (prev_deck == 1) & (deck == 0), la_actor, last_drawer)
    final_turn = (deck == 0) & (last_drawer == 0)     # partner has no turn after this one
    new_memory = {"notes": notes, "touched": touched_now, "kind": kind,
                  "prev_deck": deck, "last_drawer": last_drawer}

    # ---------------- own hand assessment ----------------
    eff_any = eff_me.any(1) & occ_me
    trash_me = occ_me & ~(eff_me & useful[None, :]).any(1)
    known_playable = eff_any & ~(eff_me & ~playable_now[None, :]).any(1)
    min_rank = jnp.where(eff_me, CARD_R[None, :], 9).min(1)
    play_pref = jnp.where(known_playable, 100 - 10 * min_rank + SLOTS, -1)
    best_play_slot = jnp.argmax(play_pref).astype(I32)
    have_play = known_playable.any()
    # risky play probability (final round only)
    w = jnp.maximum(remaining_base[None, :], 0) * eff_me
    p_play = (w * playable_now[None, :]).sum(1) / jnp.maximum(w.sum(1), 1)
    p_play = jnp.where(eff_any, p_play, 0.0)
    risky_slot = jnp.argmax(p_play).astype(I32)
    risky_ok = (deck == 0) & (lives >= 2) & (p_play[risky_slot] >= ENDGAME_RISK_P)
    # on my very last turn a clue is worthless, so any chance of a point is taken
    risky_final = final_turn & (lives >= 2) & (p_play[risky_slot] > 0.0)

    chop_me = _lowest(occ_me & ~touched_me)

    # ---------------- partner assessment ----------------
    p_eff_any = eff_p.any(1) & p_occ
    p_known_playable = p_eff_any & ~(eff_p & ~playable_now[None, :]).any(1) & playable_now[p_id]
    partner_will_play = p_known_playable.any()
    chop_p = _lowest(p_occ & ~touched_p)
    has_chop_p = chop_p >= 0
    cp = jnp.maximum(chop_p, 0)
    chop_id = p_id[cp]
    copies_elsewhere = (partner_count[chop_id] > 1) | my_single[chop_id]
    chop_useful = useful[chop_id]
    chop_needs = has_chop_p & chop_useful & (
        (CARD_R[chop_id] == 4) | critical[chop_id]
        | ((CARD_R[chop_id] == 1) & ~copies_elsewhere)
        | (playable_now[chop_id] & ~copies_elsewhere))

    # ---------------- clue evaluation ----------------
    neg = jnp.asarray(-1e9, dtype=jnp.float32)
    scores_play = jnp.full(10, -1e9, dtype=jnp.float32)
    scores_tempo = jnp.full(10, -1e9, dtype=jnp.float32)
    scores_forced = jnp.full(10, -1e9, dtype=jnp.float32)
    scores_chop = jnp.full(10, -1e9, dtype=jnp.float32)
    same_id = (p_id[:, None] == p_id[None, :]) & (SLOTS[:, None] != SLOTS[None, :]) & p_occ[None, :]
    dup_touched = (same_id & touched_p[None, :]).any(1)
    for k in range(10):
        is_color = k < 5
        value = k % 5
        attr = K.COLOR_MASKS[value] if is_color else K.RANK_MASKS[value]
        matched = p_occ & attr[p_id]
        lk = legal[10 + k] & (tokens > 0) & matched.any()
        nn, focus, is_chop_focus, has_new, new = _interpret(
            K, matched, touched_p, p_occ, jnp.asarray(is_color), jnp.asarray(value, dtype=I32),
            stack, useful, critical, chain_p, exclude, notes[1])
        fid = p_id[focus]
        fc = CARD_C[fid]
        fr = CARD_R[fid]
        f_useful = useful[fid]
        f_rel = fr - stack[fc]
        f_play_valid = f_useful & (f_rel >= 0) & (f_rel <= chain_p[fc])
        f_immediate = f_rel == 0
        sim_eff = nn[focus] & lit_p[focus]
        sim_all_playable = sim_eff.any() & ~(sim_eff & ~playable_now).any()
        sim_contains_true = sim_eff[fid]
        f_copies_elsewhere = (partner_count[fid] > 1) | my_single[fid]
        if is_color:
            save_valid = is_chop_focus & f_useful & critical[fid] & (fr != 4)
        else:
            save_valid = is_chop_focus & f_useful & (
                ((fr == 4) & (value == 4))
                | ((fr == 1) & (value == 1) & ~f_copies_elsewhere)
                | critical[fid])
        other_new = new & (SLOTS != focus)
        dup_in_clue = (same_id & matched[None, :]
                       & ((SLOTS[None, :] == focus) | (SLOTS[None, :] > SLOTS[:, None]))).any(1)
        dup = dup_touched | my_single[p_id] | dup_in_clue
        bad = other_new & (~useful[p_id] | dup)
        # re-touching a clued card that is now trash invites a wrong note
        bad_re = matched & touched_p & (SLOTS != focus) & ~useful[p_id]
        n_bad = bad.sum() + bad_re.sum()
        n_good_new = (other_new & ~bad).sum()
        focus_bad = dup_touched[focus] | my_single[fid]
        play_ok = lk & has_new & f_play_valid & sim_contains_true & (n_bad == 0) & ~focus_bad
        save_ok = lk & has_new & save_valid
        tempo_ok = (lk & ~has_new & f_play_valid & f_immediate & sim_all_playable
                    & sim_contains_true & (n_bad == 0) & ~p_known_playable[focus])
        sp = (100.0 + 20.0 * f_immediate + 15.0 * sim_all_playable + 8.0 * n_good_new
              + (3.0 if is_color else 0.0))
        ss = (60.0 + 8.0 * n_good_new - 25.0 * n_bad + (3.0 if is_color else 0.0)
              + 10.0 * (f_play_valid & sim_contains_true))
        st = 40.0 + (3.0 if is_color else 0.0) + 5.0 * (fr == 4)
        # forced clue: rank by simulated recipient outcome (never induce a misplay)
        misplay = sim_all_playable & ~(f_immediate & sim_contains_true)
        sf = (jnp.where(tempo_ok, 80.0, 0.0) + jnp.where(save_ok, 60.0, 0.0)
              + jnp.where(sim_contains_true & ~sim_all_playable, 40.0, 0.0)
              + jnp.where(sim_contains_true & sim_all_playable & f_immediate, 90.0, 0.0)
              - jnp.where(misplay, 500.0, 0.0)
              - jnp.where(~sim_contains_true, 100.0, 0.0)
              - 60.0 * n_bad - 8.0 * n_good_new)
        sf = jnp.where(lk, sf, neg)
        scores_play = scores_play.at[k].set(jnp.where(play_ok, sp, neg))
        scores_tempo = scores_tempo.at[k].set(jnp.where(tempo_ok, st, neg))
        scores_forced = scores_forced.at[k].set(sf)
        chop_score = jnp.maximum(jnp.where(play_ok, sp, neg), jnp.where(save_ok, ss, neg))
        scores_chop = scores_chop.at[k].set(jnp.where(is_chop_focus, chop_score, neg))

    best_play_k = jnp.argmax(scores_play).astype(I32)
    have_play_clue = scores_play[best_play_k] > -1e8
    best_chop_k = jnp.argmax(scores_chop).astype(I32)
    have_chop_clue = scores_chop[best_chop_k] > -1e8
    best_tempo_k = jnp.argmax(scores_tempo).astype(I32)
    have_tempo = (scores_tempo[best_tempo_k] > -1e8) & (tokens >= TEMPO_MIN_TOKENS)
    best_forced_k = jnp.argmax(scores_forced).astype(I32)
    have_forced = scores_forced[best_forced_k] > -1e8
    # a genuinely useful forced clue (tempo, save, or immediate play reading)
    harmless_forced = have_forced & (scores_forced[best_forced_k] >= 60.0)

    # ---------------- discard choice ----------------
    trash_any = trash_me.any()
    trash_slot = _lowest(trash_me)
    can_discard = tokens < 8
    # least valuable clued card (only used when there is no chop and no trash):
    # avoid possible criticals, possible useful 2s and playable cards; prefer
    # high ranks (needed latest) and cards with more candidates.
    crit_any = (eff_me & critical[None, :]).any(1)
    two_any = (eff_me & useful[None, :] & (CARD_R == 1)[None, :]).any(1)
    dis_val = jnp.where(occ_me, min_rank * 10 - crit_any.astype(I32) * 100
                        - two_any.astype(I32) * 40 - known_playable.astype(I32) * 1000
                        + eff_me.sum(1) - SLOTS, -10000)
    worst_slot = jnp.argmax(dis_val).astype(I32)
    have_chop = chop_me >= 0
    discard_slot = jnp.where(trash_any, trash_slot, jnp.where(have_chop, chop_me, worst_slot))
    no_chop_situation = ~trash_any & ~have_chop
    # with no chop: prefer a harmless clue over discarding a clued card
    discard_ok = can_discard & (~no_chop_situation | ~(harmless_forced & (tokens > 0)))

    # ---------------- decision (later picks override earlier ones) ----------------
    urgent = chop_needs & (tokens > 0) & ~partner_will_play & have_chop_clue
    urgent_action = jnp.where(have_play_clue & (tokens >= 2), 10 + best_play_k, 10 + best_chop_k)

    def pick(cond, act, current, cat, cur_cat):
        return jnp.where(cond, act, current), jnp.where(cond, cat, cur_cat)

    action = jnp.argmax(legal).astype(I32)
    cat = jnp.asarray(0, dtype=I32)
    action, cat = pick(have_forced & (tokens > 0), 10 + best_forced_k, action, 9, cat)
    action, cat = pick(discard_ok, discard_slot, action, 8, cat)
    action, cat = pick(risky_ok, 5 + risky_slot, action, 7, cat)
    action, cat = pick(have_tempo, 10 + best_tempo_k, action, 6, cat)
    action, cat = pick(chop_needs & (tokens > 0) & have_chop_clue, 10 + best_chop_k, action, 5, cat)
    action, cat = pick(have_play_clue & (tokens > 0), 10 + best_play_k, action, 3, cat)
    action, cat = pick(risky_final, 5 + risky_slot, action, 7, cat)
    action, cat = pick(have_play, 5 + best_play_slot, action, 2, cat)
    action, cat = pick(urgent & ~final_turn, urgent_action, action, 1, cat)

    action = action.astype(I32)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(I32))
    action = jnp.where(legal[20], jnp.asarray(20, dtype=I32), action)

    trace = {
        "cat": cat,
        "tokens": tokens,
        "have_play": have_play,
        "chop_needs": chop_needs,
        "p_play": p_play[risky_slot],
        "kp": known_playable,
        "effn": eff_me.sum(1),
        "sf": scores_forced,
        "sp": scores_play,
    }
    return action, new_memory, trace
