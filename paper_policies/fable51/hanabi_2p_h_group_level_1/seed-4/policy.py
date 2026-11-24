"""H-group Level 1 rule-based Hanabi policy for the two-player game.

Conventions implemented (see context/conventions/h_group_level_1):
  * chop = lowest occupied native index among unclued cards
  * clue focus: leftmost re-touched (no new cards) > chop > leftmost new card
  * non-chop focus = play clue (possibly delayed through visibly clued cards)
  * chop focus = play clue OR 5 save (rank 5) / 2 save (rank 2) / critical save
  * good touch: non-focused touched cards are useful and not duplicates
  * save principle: partner's chop that is critical / unique 2 / playable is
    protected before the partner could discard it
  * minimum clue value: voluntary clues must get a new card (tempo fill-ins
    only when they make a card literally known playable and tokens allow)

Safety rules on top of the conventions:
  * unclued cards are only played when literal information proves them playable
  * candidate identities are pruned by trash status at clue time only; a clued
    card whose candidates later became trash is held (possible duplicate)
  * identities the holder played from their own hand are removed from the
    holder's other notes (the clue giver saw the whole hand)
  * clues never newly touch an identity that may sit in one of the giver's own
    narrowly-noted clued cards (partner would read it as a duplicate)
All logic is fixed-shape JAX; private memory holds per-slot card notes.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_l1_rules', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NC, NR, HS, NID = 5, 5, 5, 25
BLOCK_MAX = 3   # own clued cards with <= this many candidates block those identities


def _consts():
    rank_mult = jnp.array([3, 2, 2, 2, 1], dtype=jnp.int32)
    ids = jnp.arange(NID, dtype=jnp.int32)
    return rank_mult, ids // NR, ids % NR, jnp.arange(HS, dtype=jnp.int32), jnp.arange(NR, dtype=jnp.int32)


def init_agent():
    return {
        "notes": jnp.ones((2, HS, NID), dtype=jnp.bool_),
        "pfocus": jnp.zeros((2, HS), dtype=jnp.bool_),
        "touched": jnp.zeros((2, HS), dtype=jnp.bool_),
        "final_left": jnp.asarray(0, dtype=jnp.int32),
    }


# ----------------------------------------------------------------------------
# board helpers
# ----------------------------------------------------------------------------

def _board(f):
    RANK_MULT, ID_COLOR, ID_RANK, SLOT, RIDX = _consts()
    fw = f[167:192].reshape(NC, NR)
    stack = jnp.sum(fw * (RIDX[None, :] + 1), axis=1).astype(jnp.int32)
    disc = f[203:253].reshape(NC, 10)
    disc_cr = jnp.stack([disc[:, 0:3].sum(1), disc[:, 3:5].sum(1), disc[:, 5:7].sum(1),
                         disc[:, 7:9].sum(1), disc[:, 9:10].sum(1)], axis=1)
    remaining = RANK_MULT[None, :] - jnp.round(disc_cr).astype(jnp.int32)
    above = RIDX[None, :] >= stack[:, None]
    blocked = (remaining <= 0) & above
    dead = (jnp.cumsum(blocked, axis=1) - blocked) > 0
    useful = above & ~dead
    playable = RIDX[None, :] == stack[:, None]
    critical = useful & (remaining == 1)
    return (stack, remaining.reshape(NID), useful.reshape(NID),
            playable.reshape(NID), critical.reshape(NID))


def _chop(occ, touched):
    RANK_MULT, ID_COLOR, ID_RANK, SLOT, RIDX = _consts()
    cand = occ & ~touched
    idx = jnp.where(cand, SLOT, 99)
    return jnp.where(jnp.any(cand), jnp.min(idx), -1).astype(jnp.int32)


def _leftmost(mask):
    RANK_MULT, ID_COLOR, ID_RANK, SLOT, RIDX = _consts()
    return jnp.max(jnp.where(mask, SLOT, -1)).astype(jnp.int32)


def _play_cands(stack, useful, playable, acc_delay, acc_excl):
    """Identities playable now or delayed-playable via accounted cards."""
    RANK_MULT, ID_COLOR, ID_RANK, SLOT, RIDX = _consts()
    acc_cr = acc_delay.reshape(NC, NR)
    above = RIDX[None, :] >= stack[:, None]
    miss = above & ~acc_cr
    before = jnp.cumsum(miss, axis=1) - miss
    delayed = (RIDX[None, :] > stack[:, None]) & (before == 0)
    cands = (playable.reshape(NC, NR) | delayed) & useful.reshape(NC, NR)
    return cands.reshape(NID) & ~acc_excl


def _eff(notes_h, lit_h):
    e = notes_h & lit_h
    empty = ~jnp.any(e, axis=-1)
    return jnp.where(empty[:, None], lit_h, e), empty


def _all_playable(cands, occ, playable):
    return occ & jnp.any(cands, -1) & ~jnp.any(cands & ~playable[None, :], -1)


def _shift(arr, k, fill):
    RANK_MULT, ID_COLOR, ID_RANK, SLOT, RIDX = _consts()
    idx = jnp.minimum(jnp.where(SLOT >= k, SLOT + 1, SLOT), HS - 1)
    out = arr[idx]
    return out.at[HS - 1].set(fill)


def _singles(eff, occ):
    return jnp.where(((jnp.sum(eff, -1) == 1) & occ)[:, None], eff, False)


# ----------------------------------------------------------------------------
# clue interpretation (recipient's perspective, from public information)
# ----------------------------------------------------------------------------

def _interpret(notes_h, lit_h, occ_h, touched_prev, matched, is_color, value,
               stack, useful, playable, critical, acc_delay_base, acc_excl_base, single_h):
    RANK_MULT, ID_COLOR, ID_RANK, SLOT, RIDX = _consts()
    matched = matched & occ_h
    new = matched & ~touched_prev
    any_new = jnp.any(new)
    chop = _chop(occ_h, touched_prev)
    chop_s = jnp.maximum(chop, 0)
    chop_touched = (chop >= 0) & new[chop_s]
    focus = jnp.where(any_new, jnp.where(chop_touched, chop, _leftmost(new)), _leftmost(matched))
    focus_s = jnp.maximum(focus, 0)
    chop_focus = any_new & chop_touched
    clue_mask = jnp.where(is_color, ID_COLOR == value, ID_RANK == value)
    own_single = jnp.any(single_h & ~matched[:, None], axis=0)
    acc_delay = acc_delay_base | own_single
    acc_excl = acc_excl_base | own_single
    play_c = _play_cands(stack, useful, playable, acc_delay, acc_excl)
    is5 = (~is_color) & (value == 4)
    is2 = (~is_color) & (value == 1)
    # save candidates deliberately ignore suit death: a partner implementation
    # may save a 5 or last copy whose suit is already dead
    last_copy = critical  # caller passes the last-copy mask (remaining == 1)
    not_played = ID_RANK >= stack[ID_COLOR]
    save_c = (last_copy | (is5 & (ID_RANK == 4)) | (is2 & (ID_RANK == 1))) & not_played
    cands = clue_mask & (play_c | (chop_focus & save_c))
    focus_note = lit_h[focus_s] & cands
    focus_ok = jnp.any(focus_note)
    focus_single = jnp.where(jnp.sum(focus_note) == 1, focus_note, False)
    lit_useful = lit_h & useful[None, :]
    # a possible save clue may violate good touch on the other touched cards
    maybe_save = chop_focus & jnp.any(lit_h[focus_s] & clue_mask & save_c)
    side_base = jnp.where(maybe_save, lit_h, lit_useful)
    other_note = side_base & ~acc_excl[None, :] & ~focus_single[None, :]
    other_note = jnp.where(jnp.any(other_note, -1, keepdims=True), other_note, lit_useful)
    other_note = jnp.where(jnp.any(other_note, -1, keepdims=True), other_note, lit_h)
    new_focus = (SLOT == focus) & new
    fnote = jnp.where(focus_ok, focus_note,
                      jnp.where(jnp.any(lit_useful[focus_s]), lit_useful[focus_s], lit_h[focus_s]))
    notes_new = jnp.where(new_focus[:, None], fnote[None, :],
                          jnp.where(new[:, None], other_note, notes_h))
    pfocus_new = new_focus & focus_ok
    return notes_new, pfocus_new, focus, new, chop_focus, any_new, clue_mask


# ----------------------------------------------------------------------------
# clue evaluation from the giver's perspective
# ----------------------------------------------------------------------------

def _eval_clue(k, legal, notes1, lit1, occ1, touched1, pid, stack, useful, playable,
               critical, remaining, acc_delay1, acc_excl1, single1, acc_g, block_ids,
               litplay1_before, chop1):
    RANK_MULT, ID_COLOR, ID_RANK, SLOT, RIDX = _consts()
    is_color = k < 5
    value = k % 5
    matched = jnp.where(is_color, ID_COLOR[pid] == value, ID_RANK[pid] == value) & occ1
    notes_new, pfocus_new, focus, new, chop_focus, any_new, clue_mask = _interpret(
        notes1, lit1, occ1, touched1, matched, is_color, value,
        stack, useful, playable, remaining == 1, acc_delay1, acc_excl1, single1)
    lit_after = lit1 & jnp.where(matched[:, None], clue_mask[None, :], ~clue_mask[None, :])
    eff_after, _ = _eff(notes_new, lit_after)
    litplay_after = _all_playable(lit_after, occ1, playable)
    allplay_after = _all_playable(eff_after, occ1, playable) & (touched1 | matched | litplay_after)
    actual_playable = playable[pid] & occ1
    would_bomb = jnp.any(allplay_after & ~actual_playable)
    n_fill = jnp.sum(litplay_after & ~litplay1_before & actual_playable)
    focus_s = jnp.maximum(focus, 0)
    fid = pid[focus_s]
    giver_play_c = _play_cands(stack, useful, playable, acc_g, jnp.zeros((NID,), dtype=jnp.bool_))
    immediate = playable[fid] & allplay_after[focus_s]
    focus_ok = giver_play_c[fid] | playable[fid]
    same = (pid[:, None] == pid[None, :]) & new[:, None] & new[None, :] & (SLOT[:, None] > SLOT[None, :])
    dup_within = jnp.any(same, axis=1)
    good_new = new & useful[pid] & ~acc_g[pid] & ~block_ids[pid] & ~dup_within
    bad_touch = jnp.sum(new & ~good_new)
    n_new_useful = jnp.sum(good_new)
    n_new = jnp.sum(new)
    valid_play = legal & any_new & new[focus_s] & focus_ok & (bad_touch == 0) & ~would_bomb
    chop_s = jnp.maximum(chop1, 0)
    chop_id = pid[chop_s]
    is5 = (~is_color) & (value == 4)
    is2 = (~is_color) & (value == 1)
    dup_chop = jnp.any((pid == chop_id) & occ1 & (SLOT != chop_s))
    two_ok = (ID_RANK[chop_id] == 1) & (remaining[chop_id] == 2) & useful[chop_id] & ~dup_chop
    save_ok = ((is5 & (ID_RANK[chop_id] == 4)) | (is2 & two_ok)
               | (critical[chop_id] & (ID_RANK[chop_id] != 4))) & useful[chop_id]
    valid_save = legal & chop_focus & save_ok & ~would_bomb
    # a re-touch reads as a tempo play clue on the leftmost re-touched card, so
    # that card must really be playable
    tempo = legal & ~any_new & (n_fill > 0) & ~would_bomb & actual_playable[focus_s]
    n_new5 = jnp.sum(new & (ID_RANK[pid] == 4))
    # a clue whose convention note would exclude the true identity misleads the partner
    truth_in = notes_new[SLOT, pid]
    misleading = jnp.any(new & ~truth_in)
    # re-touching a card that is neither fully known nor playable invites a wrong play
    fully_known = jnp.sum(lit_after, -1) == 1
    risky_retouch = jnp.sum(matched & ~new & ~fully_known & ~actual_playable)
    stall = (10 - 30 * would_bomb.astype(jnp.int32) - 25 * misleading.astype(jnp.int32)
             - 15 * risky_retouch - 6 * bad_touch - 3 * n_new + 4 * n_new5
             - 2 * chop_focus.astype(jnp.int32))
    score = jnp.where(valid_play, 100 + 20 * immediate.astype(jnp.int32) + 4 * n_new_useful + is_color.astype(jnp.int32),
            jnp.where(valid_save, 60 - 8 * bad_touch + 2 * n_new_useful + is_color.astype(jnp.int32),
            jnp.where(tempo, 40 + n_fill, stall)))
    score = jnp.where(legal, score, -1000).astype(jnp.int32)
    return score, valid_play, immediate, valid_save, tempo


# ----------------------------------------------------------------------------
# main act function
# ----------------------------------------------------------------------------

def agent_act(obs, reset, memory, rng):
    RANK_MULT, ID_COLOR, ID_RANK, SLOT, RIDX = _consts()
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    f = obs.features
    legal = obs.legal_actions
    active = ~legal[20]

    stack, remaining, useful, playable, critical = _board(f)
    tokens = jnp.round(jnp.sum(f[192:200])).astype(jnp.int32)
    lives = jnp.round(jnp.sum(f[200:203])).astype(jnp.int32)
    deck = jnp.round(jnp.sum(f[127:167])).astype(jnp.int32)
    phand = f[0:125].reshape(HS, NID)
    occ1 = jnp.sum(phand, axis=1) > 0.5
    pid = jnp.argmax(phand, axis=1).astype(jnp.int32)
    pid_oh = (phand > 0.5)
    know = f[308:658].reshape(2, HS, 35)
    lit = know[:, :, 0:25] > 0
    touched_now = jnp.sum(know[:, :, 25:35], axis=-1) > 0.5
    occ0 = jnp.any(lit[0], axis=-1)
    occ = jnp.stack([occ0, occ1])

    # ---------------- memory update from the last public action ----------
    la = f[253:308]
    has_action = jnp.sum(la[0:2]) > 0.5
    actor = jnp.argmax(la[0:2]).astype(jnp.int32)
    atype = jnp.argmax(la[2:6]).astype(jnp.int32)
    is_removal = has_action & (atype <= 1)
    is_play = has_action & (atype == 0)
    is_hint = has_action & (atype >= 2)
    recip = jnp.argmax(la[6:8]).astype(jnp.int32)
    is_color = atype == 2
    value = jnp.where(is_color, jnp.argmax(la[8:13]), jnp.argmax(la[13:18])).astype(jnp.int32)
    matched = la[18:23] > 0.5
    rem_slot = jnp.argmax(la[23:28]).astype(jnp.int32)
    revealed = la[28:53] > 0.5

    notes = mem["notes"]
    pfocus = mem["pfocus"]
    touched_prev = mem["touched"]

    def shift_hand(h):
        cond = is_removal & (actor == h)
        n_sh = _shift(notes[h], rem_slot, jnp.ones((NID,), dtype=jnp.bool_))
        # a played identity cannot sit in another clued card of the same hand
        n_sh = jnp.where(is_play, n_sh & ~revealed[None, :], n_sh)
        p_sh = _shift(pfocus[h], rem_slot, jnp.asarray(False))
        return jnp.where(cond, n_sh, notes[h]), jnp.where(cond, p_sh, pfocus[h])

    n0, p0 = shift_hand(0)
    n1, p1 = shift_hand(1)
    notes = jnp.stack([n0, n1])
    pfocus = jnp.stack([p0, p1])

    # accounted identities: convention singletons may be wrong, so they only add
    # delayed candidates; exclusions rely on literal singletons and on partner
    # cards whose modeled note is a singleton
    eff_pre = [_eff(notes[h], lit[h])[0] for h in (0, 1)]
    single = [_singles(eff_pre[h], occ[h]) for h in (0, 1)]
    lit_single = [_singles(lit[h], occ[h]) for h in (0, 1)]
    partner_clued_ids = jnp.any(pid_oh & (touched_now[1] & occ1)[:, None], axis=0) & useful
    partner_known_ids = jnp.any(pid_oh & (touched_now[1] & occ1 & jnp.any(single[1], -1))[:, None], axis=0) & useful
    my_single_ids = jnp.any(single[0], axis=0)
    my_lit_ids = jnp.any(lit_single[0], axis=0)
    acc_delay = [partner_clued_ids | my_single_ids, my_single_ids]
    acc_excl = [partner_known_ids | my_lit_ids, my_lit_ids]

    def hint_hand(h):
        cond = is_hint & (recip == h)
        n_new, pf_new, _, new, _, _, _ = _interpret(
            notes[h], lit[h], occ[h], touched_prev[h], matched, is_color, value,
            stack, useful, playable, remaining == 1, acc_delay[h], acc_excl[h], lit_single[h])
        pf_out = jnp.where(new, pf_new, pfocus[h])
        return jnp.where(cond, n_new, notes[h]), jnp.where(cond, pf_out, pfocus[h])

    n0, p0 = hint_hand(0)
    n1, p1 = hint_hand(1)
    notes = jnp.stack([n0, n1])
    pfocus = jnp.stack([p0, p1])
    notes = notes & lit
    empty = ~jnp.any(notes, -1, keepdims=True)
    notes = jnp.where(empty, lit, notes)
    pfocus = pfocus & ~empty[:, :, 0]
    # final-round counter: turns remaining (including the upcoming one) after
    # the last card was drawn; both seats see every transition so it stays in sync
    fl_prev = mem["final_left"]
    fl = jnp.where(has_action & (fl_prev > 0), fl_prev - 1, fl_prev)
    fl = jnp.where((deck == 0) & is_removal & (fl_prev == 0), 2, fl).astype(jnp.int32)
    new_mem = {"notes": notes, "pfocus": pfocus, "touched": touched_now, "final_left": fl}

    # ---------------- decision --------------------------------------------
    eff0, _ = _eff(notes[0], lit[0])
    litplay0 = _all_playable(lit[0], occ0, playable)
    allplay0 = _all_playable(eff0, occ0, playable) & (touched_now[0] | litplay0)
    # on the last life only literal proof or a play-clue focus justifies a play
    allplay0 = allplay0 & (litplay0 | pfocus[0] | (lives >= 2))
    trash0 = occ0 & ~jnp.any(lit[0] & useful[None, :], -1)
    minrank0 = jnp.min(jnp.where(eff0, ID_RANK[None, :], 9), axis=-1)
    play_score = jnp.where(allplay0, 100 + 10 * litplay0.astype(jnp.int32) + 5 * pfocus[0].astype(jnp.int32) - minrank0, -1)
    my_play = jnp.argmax(play_score).astype(jnp.int32)
    has_play = jnp.max(play_score) >= 0

    eff1, _ = _eff(notes[1], lit[1])
    litplay1 = _all_playable(lit[1], occ1, playable)
    allplay1 = _all_playable(eff1, occ1, playable) & (touched_now[1] | litplay1)
    actual_playable1 = playable[pid] & occ1
    # only literal proof counts: an independent partner may hold a card my model
    # expects them to play, and would then discard their chop
    sure_play1 = jnp.any(litplay1 & actual_playable1)
    chop1 = _chop(occ1, touched_now[1])
    chop1_s = jnp.maximum(chop1, 0)
    chop_id = pid[chop1_s]
    dup_chop = jnp.any((pid == chop_id) & occ1 & (SLOT != chop1_s))
    two_ok = (ID_RANK[chop_id] == 1) & (remaining[chop_id] == 2) & useful[chop_id] & ~dup_chop
    needs_save = (chop1 >= 0) & (critical[chop_id] | two_ok | playable[chop_id])

    # clue evaluation
    single0 = _singles(eff0, occ0)
    single1 = _singles(eff1, occ1)
    my_single_ids2 = jnp.any(single0, axis=0)
    narrow0 = (jnp.sum(eff0, -1) <= BLOCK_MAX) & occ0 & touched_now[0]
    block_ids = jnp.any(eff0 & narrow0[:, None], axis=0)
    acc_g = my_single_ids2 | partner_clued_ids
    clue_legal = legal[10:20]
    eval_clues = jax.vmap(_eval_clue, in_axes=(0, 0) + (None,) * 17)
    score, valid_play, immediate, valid_save, tempo = eval_clues(
        jnp.arange(10, dtype=jnp.int32), clue_legal, notes[1], lit[1], occ1, touched_now[1], pid,
        stack, useful, playable, critical, remaining, my_single_ids2, my_lit_ids, lit_single[1],
        acc_g, block_ids, litplay1, chop1)
    best_k = jnp.argmax(score).astype(jnp.int32)
    best_score = score[best_k]
    imm_scores = jnp.where(valid_play & immediate, score, -1000)
    imm_k = jnp.argmax(imm_scores).astype(jnp.int32)
    has_imm = jnp.max(imm_scores) > -1000
    save_scores = jnp.where(valid_save, score, -1000)
    save_k = jnp.argmax(save_scores).astype(jnp.int32)
    has_save = jnp.max(save_scores) > -1000
    has_play_clue = best_score >= 100
    tempo_scores = jnp.where(tempo, score, -1000)
    tempo_k = jnp.argmax(tempo_scores).astype(jnp.int32)
    has_tempo = jnp.max(tempo_scores) > -1000

    chop0 = _chop(occ0, touched_now[0])
    has_chop = chop0 >= 0
    has_trash = jnp.any(trash0)
    trash_slot = jnp.argmax(trash0).astype(jnp.int32)
    # least valuable clued card (fallback discard): prefer note-trash, then few critical candidates
    note_trash0 = occ0 & ~jnp.any(eff0 & useful[None, :], -1)
    crit_frac = jnp.sum(eff0 & critical[None, :], -1) / jnp.maximum(jnp.sum(eff0, -1), 1)
    lv_key = jnp.where(occ0, crit_frac + 0.01 * SLOT - 5.0 * note_trash0, 9.0)
    lv_slot = jnp.argmin(lv_key).astype(jnp.int32)

    # final-round gamble: with two or more lives a failed play costs nothing once
    # no draws remain, so play the most likely playable card if nothing is sure
    v0 = know[0, :, 0:25]
    pmass = jnp.sum(v0 * (eff0 & playable[None, :]), -1)
    tmass = jnp.sum(v0 * eff0, -1)
    pprob = jnp.where(occ0 & (tmass > 0), pmass / jnp.maximum(tmass, 1e-6), 0.0)
    gamble_slot = jnp.argmax(pprob).astype(jnp.int32)
    gamble_p = jnp.max(pprob)
    last_turn = fl == 1
    first_final = fl == 2
    partner_can_play = jnp.any(allplay1 & actual_playable1)
    clue_better = (has_imm | has_tempo) & (tokens >= 1) & ~partner_can_play
    # with the partner still to act, keep a spare life for their final inference
    gamble_now = (gamble_p > 0.0) & ((last_turn & (lives >= 2))
                                     | (first_final & (lives >= 3) & ~clue_better))

    urgent = needs_save & ~sure_play1 & (tokens >= 1) & (fl == 0)
    can_discard = tokens < 8
    # with no chop, a harmless stall (re-touch or 5s) is preferred only when the
    # least valuable clued card is likely critical; otherwise discard it
    harmless_stall = (tokens >= 1) & (best_score >= 0)
    lv_crit = crit_frac[lv_slot] >= 0.5
    discard_clued = can_discard & (~lv_crit | ~harmless_stall)

    conds = jnp.stack([
        urgent & has_play & (tokens >= 8),                 # 0 play (partner cannot discard)
        urgent & has_imm,                                  # 1 immediate play clue over save
        urgent & has_save,                                 # 2 save clue
        has_play,                                          # 3 play
        gamble_now,                                        # 4 final-round gamble
        (tokens >= 1) & has_play_clue,                     # 5 play clue
        (tokens >= 1) & has_save,                          # 6 save clue
        ((tokens >= 2) | (deck <= 3)) & has_tempo,         # 7 tempo fill-in
        can_discard & has_trash,                           # 8 discard trash
        can_discard & has_chop,                            # 9 discard chop
        discard_clued,                                     # 10 discard least valuable clued card
        tokens >= 1,                                       # 11 forced clue
        jnp.asarray(True),                                 # 12 discard least valuable (no tokens)
    ])
    acts = jnp.stack([
        5 + my_play, 10 + imm_k, 10 + save_k, 5 + my_play, 5 + gamble_slot, 10 + best_k, 10 + save_k,
        10 + tempo_k, trash_slot, chop0, lv_slot, 10 + best_k, lv_slot,
    ]).astype(jnp.int32)
    branch = jnp.argmax(conds).astype(jnp.int32)
    action = acts[branch]
    action = jnp.where(legal[action], action, jnp.argmax(legal)).astype(jnp.int32)
    action = jnp.where(active, action, 20).astype(jnp.int32)

    trace = {
        "branch": jnp.where(active, branch, -1).astype(jnp.int32),
        "tokens": tokens,
        "lives": lives,
        "deck": deck,
        "best_score": best_score,
    }
    return action, new_mem, trace
