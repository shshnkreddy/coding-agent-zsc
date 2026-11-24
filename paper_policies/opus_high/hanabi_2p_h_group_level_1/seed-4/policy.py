"""Two-player Hanabi policy following H-group Level 1 conventions.

Coordination protocol (all from the pinned H-group Level 1 bundle):

* Chop  = lowest occupied *native* index among unclued cards (runtime mapping:
  H-group "rightmost" -> lowest native index, "leftmost" -> highest native index).
* Clue focus = chop if the chop was newly touched, else the leftmost (highest
  native index) newly touched card, else the leftmost re-touched card.
* A clue focusing a non-chop card is a Play Clue (possibly delayed).  A clue
  focusing the chop is a Save Clue when it is a rank-5 clue, a rank-2 clue, or
  when a critical non-playable identity is still possible; otherwise it is a
  Play Clue.
* Good Touch Principle, on both sides.  When giving a clue it is a hard
  constraint: we do not newly touch a trash card or a second copy of a card
  already touched, unless Save Principle demands it or the clue gets two new
  playable cards.  When receiving, a *newly* touched card is promised to be
  useful at the moment of that clue, so identities that were already trash then
  are struck off; identities that turn to trash later are kept, because such a
  card simply became trash.  A re-touch carries no promise, since a clue
  necessarily touches every matching card.
* All Play Clues may be Delayed Play Clues: a designated card is only played
  once no clued connector of that colour is still outstanding.
* Save Principle: 5s, unique 2s (visible rule) and critical cards are saved on
  the chop before the partner can discard them, ahead of our own tempo.
* Minimum Clue Value Principle: clues are Play, Delayed Play or Save Clues.
  Tempo (fill-in) clues are reserved for the special situations the guide
  allows - plentiful clues or the end of the deck - and a forced clue at eight
  tokens.  We never discard while such a clue is available and affordable,
  which also satisfies the Early Game requirement.

No private signalling is used: every inference here is derived from public
information and the documented Level 1 meanings only.  Because partners are
independently written, clues are also scored by how far the receiver can verify
the focus literally, and deductions that rest purely on Good Touch require the
surviving identities to keep most of the literal probability mass.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hgroup_l1',
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

def _idx():
    return jnp.arange(5)


def _totals():
    return jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0])


def _clue_types():
    """Card-type masks for the ten clues: 0..4 colour clues, 5..9 rank clues."""
    i = jnp.arange(5)
    zero = jnp.zeros((5, 5, 5), dtype=bool)
    colour = (i[:, None, None] == i[None, :, None]) | zero
    rank = (i[:, None, None] == i[None, None, :]) | zero
    return jnp.concatenate([colour, rank], axis=0)


def init_agent():
    z = jnp.zeros(5, dtype=bool)
    o = jnp.ones((5, 5, 5), dtype=bool)
    return {
        'sp': z,   # own slots designated as Play by clue focus
        'ss': z,   # own slots designated as Save by clue focus
        'sc': z,   # own clued mask as of the previous decision point
        'pp': z,   # partner slots we gave a Play Clue to
        'ps': z,   # partner slots we gave a Save Clue to
        'pc': z,   # partner clued mask as of the previous decision point
        'sg': o,   # Good-Touch candidate mask for own slots
        'pg': o,   # Good-Touch candidate mask for partner slots
    }


def _lowest(mask):
    return jnp.where(mask.any(), jnp.argmax(mask), -1)


def _highest(mask):
    return jnp.where(mask.any(), 4 - jnp.argmax(mask[::-1]), -1)


def _focus_of(touched, prev_clued, occupied):
    """H-group clue focus, computed on the pre-clue hand."""
    t = touched & occupied
    newly = t & (~prev_clued)
    chop = _lowest(occupied & (~prev_clued))
    chop_touched = (chop >= 0) & newly[jnp.maximum(chop, 0)]
    focus = jnp.where(newly.any(),
                      jnp.where(chop_touched, chop, _highest(newly)),
                      _highest(t))
    return focus, chop, newly


def _shift(arr, k):
    """Drop slot k, shift higher slots down one, clear the freshly drawn slot."""
    idx = jnp.arange(5)
    src = jnp.where(idx >= k, jnp.minimum(idx + 1, 4), idx)
    out = arr[src]
    out = out & (idx < 4)
    return jnp.where(k >= 0, out, arr)


def _shift_notes(arr, k):
    """Same slot shift for a [5, 5, 5] note mask; a fresh draw has no notes."""
    idx = jnp.arange(5)
    src = jnp.where(idx >= k, jnp.minimum(idx + 1, 4), idx)
    out = arr[src] | (idx == 4)[:, None, None]
    return jnp.where(k >= 0, out, arr)


def agent_act(obs, reset, memory, rng):
    idx = jnp.arange(5)
    totals = _totals()
    clue_types = _clue_types()

    feats = obs.features
    legal = obs.legal_actions

    # ------------------------------------------------------------------ parse
    partner = feats[0:125].reshape(5, 5, 5) > 0.5          # [slot, colour, rank]
    deck = feats[127:167].sum()
    fireworks = feats[167:192].reshape(5, 5)
    height = (fireworks * jnp.arange(1, 6, dtype=fireworks.dtype)).sum(1)   # [colour]
    tokens = feats[192:200].sum()
    lives = feats[200:203].sum()
    discards = feats[203:253].reshape(5, 10)
    last = feats[253:308]
    knowledge = feats[308:658].reshape(2, 5, 35)
    v0 = knowledge[:, :, 0:25].reshape(2, 5, 5, 5)          # [player, slot, colour, rank]
    clued = knowledge[:, :, 25:35].sum(2) > 0.5             # [player, slot]

    dcount = jnp.stack([discards[:, 0:3].sum(1), discards[:, 3:5].sum(1),
                        discards[:, 5:7].sum(1), discards[:, 7:9].sum(1),
                        discards[:, 9:10].sum(1)], axis=1)  # [colour, rank]

    hcol = height[:, None]
    rr = idx[None, :].astype(height.dtype)
    playable = jnp.abs(rr - hcol) < 0.5                     # rank == current height
    played = rr < hcol - 0.5
    remaining = totals[None, :] - dcount                   # copies not in the discard pile
    alive = remaining > 0.5
    chain = jnp.where(rr > hcol - 0.5, alive, True)
    reachable = jnp.cumprod(chain.astype(jnp.int32), axis=1) > 0
    trash = played | (~reachable)
    critical = (~trash) & (remaining < 1.5) & (remaining > 0.5)

    partner_f = partner.astype(height.dtype)
    pcount = partner_f.sum(0)                               # partner-visible copies
    unseen = totals[None, :] - dcount - played.astype(height.dtype) - pcount
    can_hold = unseen > 0.5

    occ_s = v0[0].sum((1, 2)) > 0.5
    occ_p = partner.any((1, 2))

    w = v0[0] * can_hold[None].astype(v0.dtype)
    w = jnp.where((w.sum((1, 2)) > 1e-6)[:, None, None], w, v0[0])
    mine = w > 1e-9                                         # own literal candidates
    theirs = v0[1] > 1e-9                                   # partner's literal candidates

    # ----------------------------------------------------------- memory update
    init = init_agent()
    mem = {k: jnp.where(reset, init[k], memory[k]) for k in init}

    acted = last[0:6].sum() > 0.5
    actor_self = last[0] > 0.5
    a_play = last[2] > 0.5
    a_disc = last[3] > 0.5
    a_hint = (last[4] > 0.5) | (last[5] > 0.5)
    a_rank_hint = last[5] > 0.5
    hint_rank = last[13:18]
    matched = last[18:23] > 0.5
    removed = jnp.argmax(last[23:28])

    # (a) partner clued us: read the focus and its Level 1 meaning.
    f_s, chop_s_pre, _ = _focus_of(matched, mem['sc'], occ_s)
    fs = jnp.maximum(f_s, 0)
    focus_cands = mine[fs]
    save_rank = a_rank_hint & ((hint_rank[1] > 0.5) | (hint_rank[4] > 0.5))
    crit_possible = (focus_cands & critical & (~playable)).any()
    focus_on_chop = (f_s >= 0) & (f_s == chop_s_pre)
    read_save = focus_on_chop & (save_rank | crit_possible)
    oh_s = (idx == f_s)
    to_me = acted & a_hint & (~actor_self)
    sp = jnp.where(to_me, mem['sp'] | (oh_s & (~read_save)), mem['sp'])
    ss = jnp.where(to_me, mem['ss'] | (oh_s & read_save), mem['ss'])

    # (b) we clued the partner: record what we asked them to do.
    f_p, chop_p_pre, _ = _focus_of(matched, mem['pc'], occ_p)
    fp = jnp.maximum(f_p, 0)
    target_playable = (partner[fp] & playable).any()
    oh_p = (idx == f_p)
    to_them = acted & a_hint & actor_self
    pp = jnp.where(to_them, mem['pp'] | (oh_p & target_playable), mem['pp'])
    ps = jnp.where(to_them, mem['ps'] | (oh_p & (~target_playable)), mem['ps'])

    # (c) Good Touch Principle: a touched card is promised to be useful *at the
    #     moment of the clue*, so identities that were already trash then are
    #     eliminated.  Identities that become trash later are NOT eliminated --
    #     the card simply turns into trash, which is why we keep a note mask
    #     instead of re-applying the current trash set every turn.
    #     Only *newly* touched cards carry the promise: a clue necessarily
    #     re-touches every other matching card, so a re-touch says nothing.
    good_s = (~trash)[None] | (~(matched & (~mem['sc'])))[:, None, None]
    good_p = (~trash)[None] | (~(matched & (~mem['pc'])))[:, None, None]
    sg = jnp.where(to_me, mem['sg'] & good_s, mem['sg'])
    pg = jnp.where(to_them, mem['pg'] & good_p, mem['pg'])

    # (d) a card left a hand: slide the notes with the cards.
    moved = acted & (a_play | a_disc)
    sp = jnp.where(moved & actor_self, _shift(sp, removed), sp)
    ss = jnp.where(moved & actor_self, _shift(ss, removed), ss)
    sg = jnp.where(moved & actor_self, _shift_notes(sg, removed), sg)
    pp = jnp.where(moved & (~actor_self), _shift(pp, removed), pp)
    ps = jnp.where(moved & (~actor_self), _shift(ps, removed), ps)
    pg = jnp.where(moved & (~actor_self), _shift_notes(pg, removed), pg)

    next_mem = {'sp': sp, 'ss': ss, 'sc': clued[0], 'pp': pp, 'ps': ps,
                'pc': clued[1], 'sg': sg, 'pg': pg}

    # ------------------------------------------------------------- own options
    cand = mine & sg
    cand = jnp.where(cand.any((1, 2))[:, None, None], cand, mine)
    n_cand = cand.sum((1, 2))
    n_not_now = (cand & (~playable)[None]).sum((1, 2))
    # Deductions that rest only on Good Touch (no clue focus on this slot) are
    # the ones an independently written partner can invalidate by bad-touching,
    # so they must also keep most of the literal probability mass.
    lit_share = (w * cand).sum((1, 2)) / jnp.maximum(w.sum((1, 2)), 1e-6)
    certain_play = (occ_s & (n_cand > 0) & (n_not_now == 0)
                    & (sp | ss | (lit_share > 0.5)))
    n_useful = (cand & (~trash)[None]).sum((1, 2))

    n_playable = (cand & playable[None]).sum((1, 2))
    wc = w * cand
    wtot = jnp.maximum(wc.sum((1, 2)), 1e-6)
    prob_playable = (wc * playable[None]).sum((1, 2)) / wtot
    # A card designated as playable can still have turned into trash since the
    # clue (for instance when the other copy was played meanwhile).
    prob_trash = (wc * trash[None]).sum((1, 2)) / wtot

    # Connecting cards that could make a Play Clue a *delayed* one.
    partner_conn = (partner & playable[None] & (clued[1] | pp)[:, None, None]).any(0)  # [c,r]
    pend_partner = partner_conn.any(1)                                                 # [colour]
    self_conn = (cand & playable[None]).any(2) & clued[0][:, None]                     # [slot, colour]
    other_conn = self_conn.sum(0)[None, :] - self_conn.astype(jnp.int32)
    pending = pend_partner[None, :] | (other_conn > 0)
    next_rank = jnp.abs(rr - (hcol + 1.0)) < 0.5
    delayed = (cand & next_rank[None] & pending[:, :, None]).any((1, 2))

    careful = (lives > 1.5)
    told_play = (occ_s & sp & (n_playable > 0) & (n_useful > 0) & (~delayed) & careful
                 & (prob_playable > 0.28) & (prob_trash < 0.2))

    # Final round: a failed play only costs a life, and lives are worth nothing
    # once the game ends, so a likely card is worth trying while lives remain.
    endgame = (deck < 0.5) & (lives > 1.5)
    gamble = (endgame & clued[0] & occ_s & (n_useful > 0)
              & ((prob_playable > 0.49) & (lives > 2.5)
                 | (prob_playable > 0.69)))
    play_score = jnp.where(certain_play, 3.0,
                           jnp.where(told_play, 2.0,
                                     jnp.where(gamble, 1.0, -50.0)))
    play_score = play_score + 0.3 * prob_playable - 0.01 * idx
    play_score = jnp.where(occ_s & legal[5:10], play_score, -100.0)
    best_play = jnp.argmax(play_score)
    has_play = play_score[best_play] > 0.0

    # --------------------------------------------------- partner model / chop
    chop_p = _lowest(occ_p & (~clued[1]))
    cslot = jnp.maximum(chop_p, 0)
    chop_card = partner[cslot]
    chop_trash = (chop_card & trash).any()
    chop_crit = (chop_card & critical).any()
    chop_play = (chop_card & playable).any()
    chop_is5 = (chop_card & (rr > 3.5)).any()
    chop_is2 = (chop_card & (jnp.abs(rr - 1.0) < 0.5)).any()
    dup_visible = (partner & chop_card[None] & (occ_p & (idx != chop_p))[:, None, None]).any()
    two_save = chop_is2 & (~dup_visible)
    # A playable chop is handled by a Play Clue instead (H-group: a clue to a
    # unique playable card is a Play Clue, not a Save Clue).
    chop_needs = ((chop_p >= 0) & (~chop_trash) & (~chop_play)
                  & (chop_crit | chop_is5 | two_save))

    p_cand = theirs & pg
    p_cand = jnp.where(p_cand.any((1, 2))[:, None, None], p_cand, theirs)
    p_not_now = (p_cand & (~playable)[None]).sum((1, 2))
    p_certain = occ_p & (p_cand.sum((1, 2)) > 0) & (p_not_now == 0)
    p_slot_playable = (partner & playable[None]).any((1, 2))
    # Only count the partner as "busy" when they can *prove* a card is playable:
    # a mere Play-Clue designation may still be held back by their own caution.
    partner_plays = (occ_p & p_slot_playable & p_certain).any()
    partner_trash = (occ_p & ((p_cand & (~trash)[None]).sum((1, 2)) == 0)).any()
    partner_may_discard = (~partner_plays) & (~partner_trash)

    # Colours whose currently playable card is already accounted for: it is
    # clued in the partner's hand, or we are about to play it ourselves.
    play_colour = (cand & playable[None]).any(2)                      # [slot, colour]
    single = play_colour.sum(1) == 1
    ours = ((certain_play | told_play) & single)[:, None] & play_colour
    conn_colour = pend_partner | ours.any(0)

    # ------------------------------------------------------------ clue scoring
    touch_all = jnp.concatenate([partner.any(2).T, partner.any(1).T], axis=0)  # [10, slot]

    def clue_value(touch, ctype, is_r2, is_r5):
        t = touch & occ_p
        newly = t & (~clued[1])
        chop_touched = (chop_p >= 0) & newly[cslot]
        focus = jnp.where(newly.any(),
                          jnp.where(chop_touched, chop_p, _highest(newly)),
                          _highest(t))
        fi = jnp.maximum(focus, 0)
        card = partner[fi]
        f_playable = (card & playable).any()
        f_trash = (card & trash).any()
        f_new = newly[fi] & (focus >= 0)
        f_chop = (focus == chop_p) & (chop_p >= 0)

        new_ident = (partner & newly[:, None, None]).sum(0).astype(height.dtype)
        bad = (new_ident * trash).sum()
        all_ident = (partner & (newly | clued[1])[:, None, None]).sum(0).astype(height.dtype)
        # Duplicates only matter when this clue is the one adding a copy.
        dup = jnp.where(new_ident > 0.5, jnp.maximum(all_ident - 1.0, 0.0), 0.0).sum()
        n_new = newly.sum().astype(height.dtype)
        n_new_play = (new_ident * playable).sum()

        # What the receiver can literally deduce about the focus afterwards.
        cons = jnp.where(t[:, None, None], ctype[None], ~ctype[None])
        after = p_cand & cons & jnp.where(newly[:, None, None], (~trash)[None], True)
        fu = after[fi]
        f_certain = fu.any() & (~(fu & (~playable)).any())
        wt = v0[1][fi] * fu
        f_clarity = (wt * playable).sum() / jnp.maximum(wt.sum(), 1e-6)

        # Delayed Play Clue: the focus is one rank above a connector that is
        # already clued in the partner's hand or known-playable in ours.
        f_next = (card & next_rank).any()
        f_colour = card.any(1)
        connected = (conn_colour & f_colour).any()

        misread = f_chop & (is_r2 | is_r5) & f_playable
        valid_play = f_new & f_playable & (~f_trash)
        valid_delay = f_new & f_next & connected & (~f_trash) & (~f_playable)
        valid_save = f_new & f_chop & chop_needs & (chop_crit | (chop_is5 & is_r5) | (two_save & is_r2))
        # Tempo clue: no brand new card, but it makes an already clued playable
        # card identifiable.  Only used when clues are plentiful.
        valid_tempo = ((focus >= 0) & (n_new < 0.5) & f_playable & (~f_trash)
                       & ((tokens > 2.5) | (deck < 12.5)) & f_certain)

        v_play = (12.0 + 3.0 * n_new_play + 0.5 * (n_new - n_new_play)
                  + 4.0 * f_certain + 6.0 * f_clarity - 6.0 * misread
                  + 5.0 * (f_chop & partner_may_discard))
        v_save = 10.0 + 8.0 * partner_may_discard + 0.5 * (n_new - 1.0)
        v_tempo = 6.0
        v_delay = (9.0 + 3.0 * n_new_play + 0.5 * (n_new - 1.0) + 2.0 * f_certain
                   - 6.0 * misread)
        value = jnp.maximum(jnp.maximum(jnp.where(valid_play, v_play, -100.0),
                                        jnp.where(valid_delay, v_delay, -100.0)),
                            jnp.maximum(jnp.where(valid_save, v_save, -100.0),
                                        jnp.where(valid_tempo, v_tempo, -100.0)))
        # Good Touch Principle is a hard constraint: never introduce a trash or
        # duplicated card.  Only Save Principle (5s and critical cards) may
        # override it, and then it is merely penalised.
        override = (valid_save & (chop_crit | chop_is5)) | (n_new_play > 1.5)
        value = value - 25.0 * bad - 20.0 * dup
        value = jnp.where(((bad > 0.5) | (dup > 0.5)) & (~override), -100.0, value)
        forced = (1.0 * n_new + 3.0 * f_playable + 2.0 * f_certain
                  - 25.0 * bad - 20.0 * dup)
        return value, forced, valid_save

    values = []
    forceds = []
    saves = []
    for k in range(10):
        v, fo, sv = clue_value(touch_all[k], clue_types[k], k == 6, k == 9)
        values.append(v)
        forceds.append(fo)
        saves.append(sv)
    values = jnp.stack(values)
    forceds = jnp.stack(forceds)
    saves = jnp.stack(saves)

    clue_legal = legal[10:20]
    values = jnp.where(clue_legal, values, -1000.0)
    forceds = jnp.where(clue_legal, forceds, -1000.0)
    best_clue = jnp.argmax(values)
    has_clue = values[best_clue] > 0.0
    best_forced = jnp.argmax(forceds)
    has_forced = forceds[best_forced] > -500.0

    save_values = jnp.where(saves, values, -1000.0)
    best_save = jnp.argmax(save_values)
    has_save = save_values[best_save] > 0.0
    # Save Principle: 5s and critical cards are protected as soon as we see
    # them on the chop; unique 2s only when the partner might actually discard.
    urgent_save = has_save & chop_needs & (chop_crit | chop_is5 | partner_may_discard)

    # ---------------------------------------------------------------- discard
    known_trash = occ_s & (n_useful == 0)
    chop_s = _lowest(occ_s & (~clued[0]))
    d_slot = jnp.where(known_trash.any(), _lowest(known_trash),
                       jnp.where(chop_s >= 0, chop_s, _lowest(occ_s)))
    d_slot = jnp.maximum(d_slot, 0)
    can_discard = legal[d_slot]

    # --------------------------------------------------------------- assemble
    action = jnp.where(
        urgent_save, 10 + best_save,
        jnp.where(has_play, 5 + best_play,
                  jnp.where(has_clue, 10 + best_clue,
                            jnp.where(can_discard, d_slot,
                                      jnp.where(has_forced, 10 + best_forced,
                                                jnp.argmax(legal))))))
    action = jnp.where(legal[20], 20, action)
    action = jnp.where(legal[jnp.clip(action, 0, 20)], action, jnp.argmax(legal))
    action = action.astype(jnp.int32)

    category = jnp.where(legal[20], 5,
                         jnp.where(urgent_save, 0,
                                   jnp.where(has_play, 1,
                                             jnp.where(has_clue, 2,
                                                       jnp.where(can_discard, 3, 4)))))
    trace = {
        'category': category.astype(jnp.int32),
        'tokens': tokens.astype(jnp.float32),
        'lives': lives.astype(jnp.float32),
        'deck': deck.astype(jnp.float32),
        'score': height.sum().astype(jnp.float32),
        'play_conf': play_score[best_play].astype(jnp.float32),
        'chop_needs': chop_needs.astype(jnp.int32),
        'play_kind': jnp.where(certain_play[best_play], 1,
                               jnp.where(told_play[best_play], 2,
                                         jnp.where(gamble[best_play], 3, 0))).astype(jnp.int32),
        'play_slot': best_play.astype(jnp.int32),
        'play_cands': n_cand[best_play].astype(jnp.int32),
        'play_prob': prob_playable[best_play].astype(jnp.float32),
    }
    return action, next_mem, trace
