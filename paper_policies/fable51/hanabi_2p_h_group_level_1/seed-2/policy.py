"""H-group Level 1 rule-based two-player Hanabi policy.

Shared coordination protocol: H-group Level 1 (chop, single focus, play clues /
delayed play clues, 5-save / 2-save / critical save, Good Touch, Save and
Minimum Clue Value principles, self-prompts).  All inference is derived from
public information plus private fixed-shape card notes.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_l1_rules', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NC, NR, NID, HS = 5, 5, 25, 5
NEG = -1e9


class _K:
    """Constant arrays, built inside traced functions (no module-scope calls)."""

    def __init__(self):
        self.MULT = jnp.array([3, 2, 2, 2, 1], jnp.int32)
        self.ID_COLOR = jnp.arange(NID) // NR
        self.ID_RANK = jnp.arange(NID) % NR
        self.COLOR_IDS = self.ID_COLOR[None, :] == jnp.arange(NC)[:, None]
        self.RANK_IDS = self.ID_RANK[None, :] == jnp.arange(NR)[:, None]
        self.HINT_IDS = jnp.concatenate([self.COLOR_IDS, self.RANK_IDS], 0)  # [10, 25]
        self.SLOTS = jnp.arange(HS)


def init_agent():
    return {
        'notes': jnp.ones((2, HS, NID), dtype=bool),
        'promised': jnp.zeros((2, HS), dtype=bool),
        'clued': jnp.zeros((2, HS), dtype=bool),
        'early': jnp.asarray(True),
        'deck_by_me': jnp.asarray(False),
        'prev_deck': jnp.asarray(40, dtype=jnp.int32),
        'dup': jnp.zeros((2, HS, NID), dtype=bool),
    }


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def lowest_true(mask):
    return jnp.argmax(mask)


def highest_true(mask):
    return HS - 1 - jnp.argmax(mask[::-1])


def gettable(stack, cover, trash):
    """Identities playable now or via a chain of covered intermediates."""
    cov = cover.reshape(NC, NR)
    r_idx = jnp.arange(NR)[None, :]
    step = (r_idx < stack[:, None]) | cov
    chain = jnp.cumprod(step.astype(jnp.int32), axis=1)
    chain_excl = jnp.concatenate([jnp.ones((NC, 1), jnp.int32), chain[:, :NR - 1]], axis=1)
    g = (r_idx >= stack[:, None]) & (chain_excl > 0)
    return g.reshape(NID) & ~trash


def shift_hand(arr, slot, fresh):
    SLOTS = jnp.arange(HS)
    src = jnp.where(SLOTS >= slot, SLOTS + 1, SLOTS)
    out = arr[jnp.minimum(src, HS - 1)]
    keep = (src <= HS - 1).reshape((HS,) + (1,) * (arr.ndim - 1))
    return jnp.where(keep, out, fresh)


def interpret(notes, promised, clued_pre, occ, lit_post, match, clue_ids, is_rank,
              rank_val, stack, trash, crit, cover_other):
    """Level 1 clue interpretation from the receiver's public perspective."""
    K = _K()
    SLOTS, ID_COLOR, ID_RANK = K.SLOTS, K.ID_COLOR, K.ID_RANK
    notes = notes & lit_post
    match = match & occ
    new = match & ~clued_pre
    any_new = new.any()
    unclued_occ = occ & ~clued_pre
    has_chop = unclued_occ.any()
    chop = lowest_true(unclued_occ)
    chop_touched = has_chop & match[chop]
    lm_new = highest_true(new)
    lm_re = highest_true(match)
    focus = jnp.where(~any_new, lm_re, jnp.where(chop_touched, chop, lm_new))
    is_focus = SLOTS == focus

    self_src = clued_pre & occ & ~is_focus
    cover_self = (notes & self_src[:, None]).any(0)
    g_partner = gettable(stack, cover_other, trash)
    g_full = gettable(stack, cover_other | cover_self, trash)

    save5 = chop_touched & is_rank & (rank_val == 4)
    save2 = chop_touched & is_rank & (rank_val == 1)
    # A 5 can only be saved with a rank-5 clue, so a colour clue on chop never
    # saves a 5 (beginner guide, chop-focus example).
    crit_save = crit & (is_rank | (ID_RANK != 4))
    crit_possible = chop_touched & (crit_save & clue_ids & ~trash).any()
    focus_poss = jnp.where(chop_touched, g_full | crit_save, g_full)
    focus_poss = jnp.where(save5 | save2, ~trash, focus_poss)

    fnote_raw = notes[focus]
    fnote = fnote_raw & focus_poss
    fnote_ok = fnote.any()
    fnote = jnp.where(fnote_ok, fnote, fnote_raw & ~trash)
    fnote = jnp.where(fnote.any(), fnote, fnote_raw)

    nf = new & ~is_focus
    notes2 = jnp.where(nf[:, None], notes & ~trash[None, :], notes)
    notes2 = notes2.at[focus].set(fnote)
    notes2 = jnp.where(any_new, notes2, notes)

    pure_play = any_new & ~save5 & ~save2 & ~crit_possible
    promised2 = jnp.where(is_focus & any_new, pure_play & fnote_ok, promised)

    # Self-prompt: focus only gettable through own previously clued cards.
    need_prompt = pure_play & fnote_ok & ~(fnote & g_partner).any() & (fnote & g_full).any()
    fcol = fnote.reshape(NC, NR).any(1)
    needed = jnp.arange(NC) * NR + jnp.minimum(stack, NR - 1)
    valid_needed = stack < NR
    compat = self_src[:, None] & notes2[:, needed] & fcol[None, :] & valid_needed[None, :]
    has_c = compat.any(0)
    lm_k = HS - 1 - jnp.argmax(compat[::-1, :], axis=0)
    ks = jnp.where(fcol & has_c, lm_k, -1)
    kmax = ks.max()
    unique = ((ks == kmax) | (ks == -1)).all() & (kmax >= 0) & ~(fcol & ~has_c).any()
    narrow_ids = ((ID_COLOR[None, :] == jnp.arange(NC)[:, None]) &
                  (ID_RANK[None, :] == stack[:, None]) & fcol[:, None]).any(0)
    do_narrow = need_prompt & unique
    notes3 = jnp.where((do_narrow & (SLOTS == kmax))[:, None], notes2 & narrow_ids[None, :], notes2)
    # Non-focus cards touched incidentally by a save clue may be trash that the
    # giver had to touch (Save Principle overrides Good Touch): mark identities
    # already on the stacks as unsafe for those cards.
    is_save_clue = any_new & (save5 | save2 | crit_possible)
    save_marks = (nf & is_save_clue)[:, None] & (clue_ids & trash)[None, :]
    return notes3, promised2, focus, new, chop_touched, any_new, save_marks


def refresh(notes, promised, clued, occ, lit, cover_other, stack, trash, trash_played):
    # Only identities already on the stacks are pruned from convention notes;
    # identities that became dead later may still be the card (a stale save).
    notes = notes & lit & ~trash_played[None, :]
    src = clued & occ
    total = notes & src[:, None]
    cnt = total.sum(0)
    cover_self_k = (cnt[None, :] - total.astype(jnp.int32)) > 0
    g_k = jax.vmap(lambda cov: gettable(stack, cov, trash))(cover_other[None, :] | cover_self_k)
    pn = notes & g_k
    ok = pn.any(1)
    use = promised & ok
    notes2 = jnp.where(use[:, None], pn, notes)
    promised2 = promised & ok
    e = ~notes2.any(1)
    notes2 = jnp.where(e[:, None], lit & ~trash_played[None, :], notes2)
    e2 = ~notes2.any(1)
    notes2 = jnp.where(e2[:, None], lit, notes2)
    # Unclued cards carry no convention inference: literal knowledge only.
    notes2 = jnp.where(clued[:, None], notes2, lit)
    promised2 = promised2 & clued
    return notes2, promised2


# ----------------------------------------------------------------------------
# main act
# ----------------------------------------------------------------------------

def agent_act(obs, reset, memory, rng):
    K = _K()
    MULT, ID_COLOR, ID_RANK, SLOTS = K.MULT, K.ID_COLOR, K.ID_RANK, K.SLOTS
    COLOR_IDS, RANK_IDS, HINT_IDS = K.COLOR_IDS, K.RANK_IDS, K.HINT_IDS
    f = obs.features
    legal = obs.legal_actions
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    # ---- decode ----
    partner = f[0:125].reshape(HS, NID) > 0.5
    p_id_idx = jnp.argmax(partner, axis=1)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)
    fw = f[167:192].reshape(NC, NR)
    stack = jnp.sum(fw * (jnp.arange(NR) + 1)[None, :], axis=1).astype(jnp.int32)
    tokens = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    d = f[203:253].reshape(NC, 10)
    disc = jnp.stack([d[:, 0:3].sum(1), d[:, 3:5].sum(1), d[:, 5:7].sum(1),
                      d[:, 7:9].sum(1), d[:, 9:10].sum(1)], axis=1).astype(jnp.int32)
    la = f[253:308]
    la_valid = la[0:2].sum() > 0.5
    la_actor = jnp.argmax(la[0:2])
    la_is_play = la[2] > 0.5
    la_is_disc = la[3] > 0.5
    la_is_col = la[4] > 0.5
    la_is_rank = la[5] > 0.5
    la_target = jnp.argmax(la[6:8])
    la_color = jnp.argmax(la[8:13])
    la_rank = jnp.argmax(la[13:18])
    la_match = la[18:23] > 0.5
    la_slot = jnp.argmax(la[23:28])
    know = f[308:658].reshape(2, HS, 35)
    v0 = know[:, :, 0:25]
    lit = v0 > 0
    clued = know[:, :, 25:35].sum(-1) > 0.5
    occ = v0.sum(-1) > 0
    # Card counting: copies visible in the partner's hand cannot be in mine.
    p_id_count = (partner & occ[1][:, None]).sum(0)

    # ---- public identity masks ----
    trash_played = ID_RANK < stack[ID_COLOR]
    full = disc == MULT[None, :]
    dead_from = jnp.where(full.any(1), jnp.argmax(full, axis=1), NR)
    trash = trash_played | (ID_RANK >= dead_from[ID_COLOR])
    play_now = (ID_RANK == stack[ID_COLOR]) & ~trash
    crit = ~trash & (disc.reshape(NID) == MULT[ID_RANK] - 1)
    remaining = MULT[ID_RANK] - disc.reshape(NID) - trash_played.astype(jnp.int32) - p_id_count
    lit = jnp.stack([lit[0] & (remaining > 0), lit[1]])

    # ---- memory: shift on play/discard ----
    notes = mem['notes']
    promised = mem['promised']
    clued_pre = mem['clued']
    is_pd = la_valid & (la_is_play | la_is_disc)
    fresh_notes = jnp.ones((NID,), dtype=bool)
    fresh_flag = jnp.asarray(False)

    dup = mem['dup']
    fresh_dup = jnp.zeros((NID,), dtype=bool)

    def shifted(p):
        return (shift_hand(notes[p], la_slot, fresh_notes),
                shift_hand(promised[p], la_slot, fresh_flag),
                shift_hand(clued_pre[p], la_slot, fresh_flag),
                shift_hand(dup[p], la_slot, fresh_dup))

    s0 = shifted(0)
    s1 = shifted(1)
    do0 = is_pd & (la_actor == 0)
    do1 = is_pd & (la_actor == 1)
    notes = jnp.stack([jnp.where(do0, s0[0], notes[0]), jnp.where(do1, s1[0], notes[1])])
    promised = jnp.stack([jnp.where(do0, s0[1], promised[0]), jnp.where(do1, s1[1], promised[1])])
    clued_pre = jnp.stack([jnp.where(do0, s0[2], clued_pre[0]), jnp.where(do1, s1[2], clued_pre[1])])
    dup = jnp.stack([jnp.where(do0, s0[3], dup[0]), jnp.where(do1, s1[3], dup[1])])

    # ---- memory: interpret hint ----
    is_hint = la_valid & (la_is_col | la_is_rank)
    clue_ids = jnp.where(la_is_col, COLOR_IDS[la_color], RANK_IDS[la_rank])
    my_single = (notes[0].sum(1) == 1) & clued[0] & occ[0]
    cover_from_me = (notes[0] & my_single[:, None]).any(0)
    cover_from_partner = (partner & (clued[1] & occ[1])[:, None]).any(0)
    r0 = interpret(notes[0], promised[0], clued_pre[0], occ[0], lit[0], la_match, clue_ids,
                   la_is_rank, la_rank, stack, trash, crit, cover_from_partner)
    r1 = interpret(notes[1], promised[1], clued_pre[1], occ[1], lit[1], la_match, clue_ids,
                   la_is_rank, la_rank, stack, trash, crit, cover_from_me)
    h0 = is_hint & (la_target == 0)
    h1 = is_hint & (la_target == 1)
    notes = jnp.stack([jnp.where(h0, r0[0], notes[0]), jnp.where(h1, r1[0], notes[1])])
    promised = jnp.stack([jnp.where(h0, r0[1], promised[0]), jnp.where(h1, r1[1], promised[1])])

    # Possible-duplicate marks: a clued card of the giver that might be the
    # identity just clued in the receiver's hand is not safe to play once that
    # identity has been played, unless literal information excludes it.
    new_touch = la_match & ~clued_pre
    touched_ids_p = (partner & (new_touch[1] & occ[1])[:, None]).any(0)   # I clued partner
    touched_ids_m = (notes[0] & (new_touch[0] & occ[0])[:, None]).any(0)  # partner clued me
    mark0 = h1 & (la_actor == 0)
    mark1 = h0 & (la_actor == 1)
    # Giver-side marks on the giver's own clued cards.
    dup0 = dup[0] | (mark0 & (clued[0] & occ[0])[:, None] & notes[0] & touched_ids_p[None, :])
    dup1 = dup[1] | (mark1 & (clued[1] & occ[1])[:, None] & notes[1] & touched_ids_m[None, :])
    # Receiver-side marks on the newly touched cards: the giver's previously
    # clued cards that match the clue might be the same identity.
    # Only cards whose identity the giver does not know can be unintended
    # duplicates (a giver who knows the card would not touch its copy).
    giver_unsure_p = clued_pre[1] & occ[1] & (notes[1].sum(1) > 1)
    giver_unsure_m = clued_pre[0] & occ[0] & (notes[0].sum(1) > 1)
    giver_ids_for_me = (partner & giver_unsure_p[:, None] & clue_ids[None, :]).any(0)
    giver_ids_for_p = (notes[0] & giver_unsure_m[:, None] & clue_ids[None, :]).any(0)
    dup0 = dup0 | (mark1 & (new_touch[0] & occ[0])[:, None] & giver_ids_for_me[None, :])
    dup1 = dup1 | (mark0 & (new_touch[1] & occ[1])[:, None] & giver_ids_for_p[None, :])
    dup0 = dup0 | (h0 & r0[6])
    dup1 = dup1 | (h1 & r1[6])
    dup = jnp.stack([dup0, dup1])

    # ---- memory: refresh ----
    n0, p0 = refresh(notes[0], promised[0], clued[0], occ[0], lit[0], cover_from_partner, stack,
                     trash, trash_played)
    my_single = (n0.sum(1) == 1) & clued[0] & occ[0]
    cover_from_me = (n0 & my_single[:, None]).any(0)
    n1, p1 = refresh(notes[1], promised[1], clued[1], occ[1], lit[1], cover_from_me, stack,
                     trash, trash_played)
    notes = jnp.stack([n0, n1])
    promised = jnp.stack([p0, p1])

    early = mem['early'] & ~(la_valid & la_is_disc)
    deck_by_me = mem['deck_by_me'] | (is_pd & (la_actor == 0) & (deck == 0) & (mem['prev_deck'] > 0))

    new_mem = {
        'notes': notes,
        'promised': promised,
        'clued': clued,
        'early': early,
        'deck_by_me': deck_by_me,
        'prev_deck': deck,
        'dup': dup,
    }

    # ---- decision ----
    occ0 = occ[0]
    occ1 = occ[1]
    clued0 = clued[0]
    clued1 = clued[1]
    notes0 = notes[0]
    notes1 = notes[1]
    unsafe = (dup & trash_played[None, None, :] & lit).any(-1)
    known_play = occ0 & notes0.any(1) & ~(notes0 & ~play_now[None, :]).any(1) & ~unsafe[0]
    known_trash = occ0 & ~(lit[0] & ~trash[None, :]).any(1)
    p_known_play = occ1 & notes1.any(1) & ~(notes1 & ~play_now[None, :]).any(1) & ~unsafe[1]
    partner_has_play = p_known_play.any()

    w = v0[0] * notes0
    wsum = jnp.maximum(w.sum(1), 1e-6)
    p_play = (w * play_now[None, :]).sum(1) / wsum
    p_crit = (w * crit[None, :]).sum(1) / wsum
    p_useful = (w * (~trash)[None, :]).sum(1) / wsum

    my_unclued = occ0 & ~clued0
    my_has_chop = my_unclued.any()
    my_chop = lowest_true(my_unclued)

    p_unclued = occ1 & ~clued1
    p_has_chop = p_unclued.any()
    p_chop = lowest_true(p_unclued)
    p_chop_id = p_id_idx[p_chop]
    p_clued_ids = cover_from_partner
    mine_likely = cover_from_me | (notes0 & known_play[:, None]).any(0)
    mine_possible_count = (notes0 & (occ0 & clued0)[:, None]).sum(0)
    dist_id = ID_RANK - stack[ID_COLOR]
    useful_w = jnp.where(dist_id == 1, 15.0, jnp.where(dist_id == 2, 8.0,
                         jnp.where(dist_id == 3, 2.0, -8.0)))
    stall_w = jnp.where(dist_id <= 1, 12.0, jnp.where(dist_id == 2, 6.0, jnp.where(dist_id == 3, -3.0,
                        jnp.where(ID_RANK == 4, 0.0, -8.0))))
    visible_elsewhere = (p_id_count[p_chop_id] > 1) | mine_likely[p_chop_id]
    chop_crit = crit[p_chop_id]
    chop_two = (ID_RANK[p_chop_id] == 1) & ~trash[p_chop_id] & ~visible_elsewhere
    chop_play_unique = play_now[p_chop_id] & ~visible_elsewhere
    chop_needs = p_has_chop & (chop_crit | chop_two | chop_play_unique)
    urgent = chop_needs & ~partner_has_play

    my_last = deck == 0
    partner_after = ~deck_by_me

    # ---- own plays ----
    next_id = jnp.minimum(jnp.arange(NID) + 1, NID - 1)
    enables = (ID_RANK < NR - 1) & p_clued_ids[next_id]
    bonus_id = 30.0 * (ID_RANK == 4) + 20.0 * enables + (4.0 - ID_RANK)
    nsum = jnp.maximum(notes0.sum(1), 1)
    exp_bonus = (notes0 * bonus_id[None, :]).sum(1) / nsum
    guess_ok = my_last & (lives >= 2) & (p_play >= 0.15) & ~known_play
    play_val = jnp.where(known_play, 1000.0 + exp_bonus,
                         jnp.where(guess_ok, 250.0 + 100.0 * p_play, -1e6 + 100.0 * p_play))
    play_val = jnp.where(occ0, play_val, NEG)

    # ---- discards ----
    # Locked hand (no chop): discard the least valuable clued card rather than
    # burn clue tokens on re-touch stalls, unless the card may be critical.
    locked_val = 160.0 - 120.0 * p_crit - 20.0 * p_useful + 5.0 * (8 - tokens) - 100.0 * p_play
    disc_val = jnp.where(known_trash, 300.0,
                         jnp.where(my_has_chop & (SLOTS == my_chop), 200.0,
                                   jnp.where(my_has_chop, 50.0 - 30.0 * p_crit - 10.0 * p_useful,
                                             locked_val)))
    disc_val = jnp.where(occ0, disc_val, NEG)
    disc_val = jnp.where(my_last, jnp.minimum(disc_val, 20.0), disc_val)

    # ---- hints ----
    p_clued_count = (partner & (clued1 & occ1)[:, None]).sum(0)

    def eval_hint(h):
        cids = HINT_IDS[h]
        is_rank = h >= NC
        rank_val = h - NC
        match = occ1 & (partner & cids[None, :]).any(1)
        lit_post = lit[1] & jnp.where(match[:, None], cids[None, :], ~cids[None, :])
        n2, pr2, focus, new, chop_touched, any_new, _marks = interpret(
            notes1, promised[1], clued1, occ1, lit_post, match, cids, is_rank, rank_val,
            stack, trash, crit, cover_from_me)
        is_focus = SLOTS == focus
        f_id = p_id_idx[focus]
        actual_in_note = n2[SLOTS, p_id_idx]
        n_lit = notes1 & lit_post
        changed_conv = (n2 != n_lit).any(1) & ~new & ~is_focus
        consistent = actual_in_note[focus] & ~(changed_conv & occ1 & ~actual_in_note).any()
        all_consistent = ~(occ1 & clued1 & ~actual_in_note).any()
        fn = n2[focus]
        partner_plays_focus = fn.any() & ~(fn & ~play_now[None, :]).any()
        kp2 = occ1 & n2.any(1) & ~(n2 & ~play_now[None, :]).any(1) & ~unsafe[1]
        partner_plays_any = kp2.any()
        # A clue with no new cards is a tempo clue in Level 1 terms: its focus
        # (leftmost re-touched) must be playable or a partner may misplay it.
        unlock = (kp2 & ~p_known_play).any() & (any_new | play_now[p_id_idx[focus]])
        bomb_risk = (kp2 & ~play_now[p_id_idx]).any()
        new_count = (partner & new[:, None]).sum(0)
        # Two copies of one identity touched and both considered playable: the
        # second play would bomb.
        bomb_risk = bomb_risk | ((new & kp2 & (new_count[p_id_idx] > 1)).sum() > 1)

        cover_giver = cover_from_me | (partner & (clued1 & occ1 & ~is_focus)[:, None]).any(0)
        g_giver = gettable(stack, cover_giver, trash)
        dup_focus = (p_clued_count[f_id] > 0) | mine_likely[f_id]
        valid_play = (any_new & ~trash[f_id] & ~dup_focus & consistent & ~bomb_risk &
                      (play_now[f_id] | (g_giver[f_id] & all_consistent)))

        other_new = new & ~is_focus
        ids_bad = trash | (p_clued_count > 0) | mine_likely
        dup_within = other_new & (new_count[p_id_idx] > 1)
        bad = other_new & (ids_bad[p_id_idx] | dup_within)
        n_bad = bad.sum()
        n_dup = dup_within.sum()
        good_new = other_new & ~bad
        n_useful = good_new.sum()
        useful_score = (good_new * useful_w[p_id_idx]).sum()
        stall_score = (good_new * stall_w[p_id_idx]).sum()

        f_rank = ID_RANK[f_id]
        f_vis_else = (p_id_count[f_id] > 1) | mine_likely[f_id]
        save5 = is_rank & (rank_val == 4) & (f_rank == 4)
        save2 = is_rank & (rank_val == 1) & (f_rank == 1) & ~trash[f_id] & ~f_vis_else
        savec = crit[f_id] & (f_rank != 4)
        savep = play_now[f_id] & ~dup_focus
        valid_save = chop_touched & any_new & (save5 | save2 | savec | savep) & consistent & ~bomb_risk
        handles_chop = chop_touched & any_new & (valid_play | valid_save)

        color_pref = 5.0 * (~is_rank)
        dup_pen = 20.0 * mine_possible_count[f_id]
        pv = (500.0 + 100.0 * partner_plays_focus + 40.0 * play_now[f_id] + useful_score
              + color_pref + 2.0 * (fn.sum() == 1) - dup_pen)
        pv = jnp.where(valid_play & (n_bad == 0), pv, NEG)
        sv = (400.0 + color_pref + 100.0 * partner_plays_focus + useful_score
              - 100.0 * n_bad - 200.0 * n_dup - dup_pen)
        sv = jnp.where(valid_save, sv, NEG)
        retouch_safe = ~any_new & play_now[f_id]
        retouch_risky = ~any_new & ~play_now[f_id]
        stall = (100.0 + 60.0 * retouch_safe - 40.0 * retouch_risky + stall_score - 300.0 * n_bad
                 + jnp.where(any_new & ~trash[f_id], stall_w[f_id], 0.0)
                 + 300.0 * (unlock & (n_bad == 0)))
        stall = jnp.where(bomb_risk, NEG, stall - 400.0 * (~consistent))
        boost = jnp.maximum(
            jnp.where(urgent & handles_chop, 1100.0, 0.0),
            jnp.where(urgent & valid_play & partner_plays_any & (tokens >= 2), 1050.0, 0.0))
        val = jnp.maximum(jnp.maximum(pv, sv), stall) + boost
        # last-turn adjustments
        last_val = jnp.where(partner_after & valid_play & partner_plays_any, pv, 10.0)
        val = jnp.where(my_last, last_val, val)
        return jnp.where(legal[10 + h], val, NEG)

    hint_val = jax.vmap(eval_hint)(jnp.arange(10))

    vals = jnp.concatenate([disc_val, play_val, hint_val, jnp.array([NEG])])
    vals = jnp.where(legal, vals, NEG)
    vals = jnp.where(legal[20], jnp.where(jnp.arange(21) == 20, 1.0, NEG), vals)
    action = jnp.argmax(vals).astype(jnp.int32)

    trace = {
        'known_play': known_play,
        'p_known_play': p_known_play,
        'urgent': urgent,
        'chop_needs': chop_needs,
        'best_val': vals[action],
        'tokens': tokens,
        'lives': lives,
        'deck': deck,
        'score': stack.sum(),
        'nsz': notes0.sum(1),
        'pnsz': notes1.sum(1),
        'promised': promised[0],
        'phand': jnp.where(occ1, p_id_idx, -1),
        'stack': stack,
        'clued': clued0,
        'pclued': clued1,
    }
    return action, new_mem, trace
