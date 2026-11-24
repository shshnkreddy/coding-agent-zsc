"""Two-player Hanabi policy following H-group Level 1 conventions.

Coordination protocol (shared, documented in context/conventions/h_group_level_1):
  * Chop = lowest occupied native index among unclued cards (H-group "rightmost").
  * Clue focus = leftmost re-touched card if nothing new, else chop if the chop was
    touched, else leftmost newly touched card ("leftmost" = highest native index).
  * A clue focusing a non-chop card is a Play Clue (possibly Delayed).
  * A chop-focused clue is a Play Clue or one of the permitted Save Clues
    (5 Save with number 5, 2 Save with number 2 subject to the visible rule,
    Critical Save with either clue type).
  * Non-focused touched cards are useful by Good Touch Principle.
  * Minimum Clue Value Principle: every clue given here must touch a previously
    unclued card and must focus either a playable card or a legal save target.
  * Self-prompts: a play clue whose focus can only be delayed promises a
    connecting card among the recipient's already-clued cards; the leftmost
    compatible one is played first.

Action order on my turn: save the partner's chop when it holds a 5 or a
critical card and they have nothing certain to play; otherwise play a card I am
certain of; otherwise a play clue, then a permitted save clue; then, only to
avoid stalling, a tempo clue on a card that really is playable; then an endgame
gamble once the deck is nearly gone and a bomb cannot cost points; then discard
known trash or the chop; a forced clue when eight tokens make discarding
illegal.

All inference is from public information plus this seat's private card notes.
Notes are never narrowed by assuming a clued card cannot be a second copy of an
identity that has since been played, because that assumption bombs whenever the
team has touched a duplicate.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hgroup_level1',
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

HAND = 5
NCARD = 25
_COLOR_OF = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
             3, 3, 3, 3, 3, 4, 4, 4, 4, 4)
_RANK_OF = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4)
_COPIES = (3.0, 2.0, 2.0, 2.0, 1.0)
_SLOTS = (0, 1, 2, 3, 4)
_CLUE_IS_RANK = (False, False, False, False, False, True, True, True, True, True)
_CLUE_ATTR = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4)


def init_agent():
    return {
        'note': jnp.ones((HAND, NCARD), dtype=jnp.float32),
        'pending': jnp.zeros((HAND,), dtype=jnp.bool_),
        'promised': jnp.zeros((HAND,), dtype=jnp.bool_),
        'prev_clued': jnp.zeros((HAND,), dtype=jnp.bool_),
    }


def _first(mask):
    """Lowest index where mask is true (0 when empty)."""
    return jnp.argmax(mask).astype(jnp.int32)


def _last(mask):
    """Highest index where mask is true (0 when empty)."""
    return (4 - jnp.argmax(mask[::-1])).astype(jnp.int32)


def _last_rows(mask):
    """Highest true index per row of a [n, 5] mask."""
    return (4 - jnp.argmax(mask[:, ::-1], axis=1)).astype(jnp.int32)


def _shift(arr, removed, fill):
    """Remove `removed`, shift later slots down, append `fill` at slot 4."""
    slots = jnp.arange(5, dtype=jnp.int32)
    src = jnp.where(slots < removed, slots, slots + 1)
    padded = jnp.concatenate([arr, fill[None, ...]], axis=0)
    return padded[src]


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    COLOR_OF = jnp.asarray(_COLOR_OF, dtype=jnp.int32)
    RANK_OF = jnp.asarray(_RANK_OF, dtype=jnp.int32)
    COPIES = jnp.asarray(_COPIES, dtype=jnp.float32)
    SLOTS = jnp.asarray(_SLOTS, dtype=jnp.int32)
    CLUE_IS_RANK = jnp.asarray(_CLUE_IS_RANK, dtype=jnp.bool_)
    CLUE_ATTR = jnp.asarray(_CLUE_ATTR, dtype=jnp.int32)
    init = init_agent()
    note = jnp.where(reset, init['note'], memory['note'])
    pending = jnp.where(reset, init['pending'], memory['pending'])
    promised = jnp.where(reset, init['promised'], memory['promised'])
    prev_clued = jnp.where(reset, init['prev_clued'], memory['prev_clued'])

    # ---------------- decode the public board ----------------
    partner = f[0:125].reshape(HAND, NCARD)
    deck = jnp.sum(f[127:167])
    fireworks = f[167:192].reshape(5, 5)
    height = jnp.sum(fireworks * (jnp.arange(5, dtype=jnp.float32) + 1.0)[None, :], axis=1)
    info = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    dblocks = f[203:253].reshape(5, 10)
    disc = jnp.stack([
        jnp.sum(dblocks[:, 0:3], axis=1),
        jnp.sum(dblocks[:, 3:5], axis=1),
        jnp.sum(dblocks[:, 5:7], axis=1),
        jnp.sum(dblocks[:, 7:9], axis=1),
        dblocks[:, 9],
    ], axis=1).reshape(NCARD)

    h_at = height[COLOR_OF]
    rank_f = RANK_OF.astype(jnp.float32)
    playable = rank_f == h_at
    already = rank_f < h_at
    remaining = COPIES[RANK_OF] - disc - already.astype(jnp.float32)
    dead = ((remaining <= 0.0) & (rank_f >= h_at)).reshape(5, 5).astype(jnp.float32)
    blocked = ((jnp.cumsum(dead, axis=1) - dead) > 0.0).reshape(NCARD)
    trash = already | blocked
    useful = jnp.logical_not(trash)
    critical = (remaining <= 1.0) & useful
    usefulf = useful.astype(jnp.float32)
    playablef = playable.astype(jnp.float32)
    trashf = trash.astype(jnp.float32)

    # ---------------- hands ----------------
    p_occ = jnp.sum(partner, axis=1) > 0.0
    p_idx = jnp.argmax(partner, axis=1).astype(jnp.int32)
    p_color = p_idx // 5
    p_rank = p_idx - 5 * p_color
    know = f[308:658].reshape(2, HAND, 35)
    own_lit = know[0, :, 0:25] > 0.0
    own_occ = jnp.sum(know[0, :, 0:25], axis=1) > 0.0
    own_clued = jnp.sum(know[0, :, 25:35], axis=1) > 0.0
    p_clued = jnp.sum(know[1, :, 25:35], axis=1) > 0.0
    p_chop_mask = p_occ & jnp.logical_not(p_clued)
    p_has_chop = jnp.any(p_chop_mask)
    p_chop = _first(p_chop_mask)

    # ---------------- last public action ----------------
    la = f[253:308]
    la_live = (jnp.sum(jnp.abs(la)) > 0.0) & jnp.logical_not(reset)
    la_mine = la[0] > 0.5
    la_play = la[2] > 0.5
    la_disc = la[3] > 0.5
    la_chint = la[4] > 0.5
    la_rhint = la[5] > 0.5
    la_to_me = la[6] > 0.5
    la_rank_oh = la[13:18]
    la_match = la[18:23] > 0.5
    la_removed = jnp.argmax(la[23:28]).astype(jnp.int32)

    # --- my own card left the hand: shift the notes ---
    shifted = la_live & la_mine & (la_play | la_disc)
    note = jnp.where(shifted, _shift(note, la_removed, jnp.ones((NCARD,), jnp.float32)), note)
    pending = jnp.where(shifted, _shift(pending, la_removed, jnp.zeros((), jnp.bool_)), pending)
    promised = jnp.where(shifted, _shift(promised, la_removed, jnp.zeros((), jnp.bool_)), promised)
    prev_clued = jnp.where(shifted, _shift(prev_clued, la_removed, jnp.zeros((), jnp.bool_)), prev_clued)

    # ---------------- literal knowledge ----------------
    # Convention narrowing happens once, when the clue is given.  Nothing is
    # eliminated later just because an identity has since been played: if the
    # team ever touched a duplicate, my card may well be that second copy, and
    # eliminating it turns a harmless dead card into a confident bomb.
    note = note * own_lit.astype(jnp.float32)
    empty0 = (jnp.sum(note, axis=1) <= 0.0) & own_occ
    note = jnp.where(empty0[:, None], own_lit.astype(jnp.float32), note)
    pending = pending & jnp.logical_not(empty0)
    promised = promised & jnp.logical_not(empty0)

    # ---------------- delayed-play reach (connectors that are publicly gotten) ----------------
    partner_conn = jnp.sum(partner * (p_occ & p_clued).astype(jnp.float32)[:, None], axis=0) > 0.0
    note_sum0 = jnp.sum(note, axis=1)
    certain0 = (own_occ & prev_clued & (note_sum0 > 0.0)
                & (jnp.sum(note * (1.0 - playablef), axis=1) <= 0.0))
    own_conn_tight = jnp.sum(note * certain0.astype(jnp.float32)[:, None], axis=0) > 0.0
    conn_tight = partner_conn | (own_conn_tight & playable)
    narrow = (note_sum0 <= 5.0) & prev_clued & own_occ
    own_conn_wide = jnp.sum(note * narrow.astype(jnp.float32)[:, None], axis=0) > 0.0
    conn_wide = conn_tight | own_conn_wide

    def _reach(connflat):
        c2 = connflat.reshape(5, 5).astype(jnp.float32)
        rc = height
        for _ in range(4):
            r_int = jnp.clip(rc.astype(jnp.int32), 0, 4)
            has = jnp.take_along_axis(c2, r_int[:, None], axis=1)[:, 0] > 0.0
            rc = rc + jnp.where((rc < 5.0) & has, 1.0, 0.0)
        return rc

    reach_t = _reach(conn_tight)
    reach_w = _reach(conn_wide)
    play_tight = (rank_f >= h_at) & (rank_f <= reach_t[COLOR_OF]) & useful
    play_wide = (rank_f >= h_at) & (rank_f <= reach_w[COLOR_OF]) & useful

    # ---------------- interpret a clue that was given to me ----------------
    is_hint_to_me = la_live & jnp.logical_not(la_mine) & (la_chint | la_rhint) & la_to_me
    match = la_match & own_occ
    new_touch = match & jnp.logical_not(prev_clued)
    any_new = jnp.any(new_touch)
    chop_mask_pre = own_occ & jnp.logical_not(prev_clued)
    has_chop_pre = jnp.any(chop_mask_pre)
    chop_pre = _first(chop_mask_pre)
    chop_touched = has_chop_pre & match[chop_pre]
    focus = jnp.where(jnp.logical_not(any_new), _last(match),
                      jnp.where(chop_touched, chop_pre, _last(new_touch)))
    is_chop_focus = any_new & chop_touched
    rank2_clue = la_rhint & (la_rank_oh[1] > 0.5)
    save_set = critical | (rank2_clue & (RANK_OF == 1) & useful)
    extra = jnp.where(is_chop_focus, save_set.astype(jnp.float32), 0.0)
    focus_oh = SLOTS == focus
    apply_hint = is_hint_to_me & jnp.any(match)

    # Good Touch de-duplication: a clued identity is not held twice by the team.
    singles = (jnp.sum(note, axis=1) <= 1.0) & own_occ & prev_clued
    own_taken = jnp.sum(note * singles.astype(jnp.float32)[:, None], axis=0) > 0.0
    # Only de-duplicate against cards whose identity the partner also knows: a
    # card I can merely see in their hand is not something the clue giver knew.
    p_known = p_occ & (jnp.sum(know[1, :, 25:30], axis=1) > 0.0) & (
        jnp.sum(know[1, :, 30:35], axis=1) > 0.0)
    partner_known_id = jnp.sum(partner * p_known.astype(jnp.float32)[:, None], axis=0) > 0.0
    taken = (own_taken | partner_known_id) & jnp.logical_not(critical)
    lit_focus = jnp.sum(note * focus_oh.astype(jnp.float32)[:, None], axis=0)
    lit_focus = lit_focus * jnp.logical_not(taken).astype(jnp.float32)
    n_tight = lit_focus * jnp.maximum(play_tight.astype(jnp.float32), extra)
    n_wide = lit_focus * jnp.maximum(play_wide.astype(jnp.float32), extra)
    n_use = lit_focus * usefulf
    fnote = jnp.where(jnp.sum(n_tight) > 0.0, n_tight,
                      jnp.where(jnp.sum(n_wide) > 0.0, n_wide,
                                jnp.where(jnp.sum(n_use) > 0.0, n_use, lit_focus)))
    focus_factor = jnp.where(focus_oh[:, None], fnote[None, :], note)
    gtp_slots = new_touch & jnp.logical_not(focus_oh)
    gtp_factor = jnp.where(gtp_slots[:, None],
                           note * (usefulf * jnp.logical_not(taken).astype(jnp.float32))[None, :],
                           note)
    updated = jnp.where(focus_oh[:, None], focus_factor,
                        jnp.where(gtp_slots[:, None], gtp_factor, note))
    note = jnp.where(apply_hint, updated, note)
    pending = pending | (focus_oh & apply_hint)
    promised = promised | (focus_oh & apply_hint)

    # ---------------- literal knowledge and repair ----------------
    note = note * own_lit.astype(jnp.float32)
    empty_note = (jnp.sum(note, axis=1) <= 0.0) & own_occ
    note = jnp.where(empty_note[:, None], own_lit.astype(jnp.float32), note)
    pending = pending & jnp.logical_not(empty_note)
    promised = promised & jnp.logical_not(empty_note)

    # ---------------- self-prompt: the focus can only play later ----------------
    focus_delayed = apply_hint & (jnp.sum(n_tight) <= 0.0) & (jnp.sum(n_wide) > 0.0)
    fnote = jnp.sum(note * focus_oh.astype(jnp.float32)[:, None], axis=0)
    focus_colors = jnp.sum(fnote.reshape(5, 5), axis=1) > 0.0
    note_sum = jnp.sum(note, axis=1)
    certain_play = own_occ & (note_sum > 0.0) & (jnp.sum(note * (1.0 - playablef), axis=1) <= 0.0)
    head = (rank_f == h_at) & useful
    headf = head.astype(jnp.float32)
    own_coming = jnp.sum(note * headf[None, :] * certain_play.astype(jnp.float32)[:, None], axis=0) > 0.0
    covered = ((partner_conn | own_coming) & head).reshape(5, 5)
    color_needed = focus_colors & jnp.logical_not(jnp.any(covered, axis=1))
    needed = (head & color_needed[COLOR_OF]).astype(jnp.float32)
    prompt_cand = (own_occ & prev_clued & jnp.logical_not(focus_oh)
                   & (jnp.sum(note * needed[None, :], axis=1) > 0.0))
    prompt_slot = _last(prompt_cand)
    do_prompt = (apply_hint & any_new & focus_delayed & jnp.any(prompt_cand)
                 & (jnp.sum(needed) > 0.0))
    prompt_oh = (SLOTS == prompt_slot) & do_prompt
    note = jnp.where(prompt_oh[:, None], note * needed[None, :], note)
    pending = pending | prompt_oh

    # ---------------- what do I know about my own cards ----------------
    note_sum = jnp.sum(note, axis=1)
    known_playable = own_occ & (note_sum > 0.0) & (jnp.sum(note * (1.0 - playablef), axis=1) <= 0.0)
    known_trash = own_occ & (note_sum > 0.0) & (jnp.sum(note * (1.0 - trashf), axis=1) <= 0.0)
    # On the last life only act on a card that a clue explicitly focused, never
    # on one identified by elimination or by an inferred prompt.
    known_playable = known_playable & ((lives >= 2.0) | promised)
    have_play = jnp.any(known_playable)
    # Among certain plays, start the lowest rank: it unblocks the most cards.
    min_rank = jnp.min(jnp.where(note > 0.0, rank_f[None, :], 9.0), axis=1)
    play_pref = jnp.where(known_playable, 50.0 - 10.0 * min_rank - SLOTS.astype(jnp.float32),
                          -1e9)
    play_slot = jnp.argmax(play_pref).astype(jnp.int32)

    own_chop_mask = own_occ & jnp.logical_not(own_clued)
    own_chop = _first(own_chop_mask)
    discard_slot = jnp.where(jnp.any(known_trash), _first(known_trash),
                             jnp.where(jnp.any(own_chop_mask), own_chop, _first(own_occ)))

    # ---------------- evaluate every clue I could give ----------------
    touch_c = (p_color[None, :] == jnp.arange(5, dtype=jnp.int32)[:, None]) & p_occ[None, :]
    touch_r = (p_rank[None, :] == jnp.arange(5, dtype=jnp.int32)[:, None]) & p_occ[None, :]
    touched = jnp.concatenate([touch_c, touch_r], axis=0)             # [10, 5]
    c_legal = legal[10:20]
    c_new = touched & jnp.logical_not(p_clued)[None, :]
    c_any_new = jnp.any(c_new, axis=1)
    c_num_new = jnp.sum(c_new.astype(jnp.float32), axis=1)
    c_chop_touched = touched[:, p_chop] & p_has_chop
    c_focus = jnp.where(jnp.logical_not(c_any_new), _last_rows(touched),
                        jnp.where(c_chop_touched, p_chop, _last_rows(c_new)))
    c_focus_id = p_idx[c_focus]
    c_focus_occ = p_occ[c_focus]
    c_focus_playable = playable[c_focus_id] & c_focus_occ
    c_is_chop_focus = c_any_new & c_chop_touched

    same = (p_idx[:, None] == p_idx[None, :]) & p_occ[:, None] & p_occ[None, :]
    same = same & jnp.logical_not(jnp.eye(5, dtype=jnp.bool_))
    dup_clued = jnp.any(same & p_clued[None, :], axis=1)
    later = SLOTS[None, :] > SLOTS[:, None]
    dup_touch = jnp.any(same[None, :, :] & later[None, :, :] & touched[:, None, :], axis=2)
    own_single = (note_sum <= 1.0) & own_clued & own_occ
    own_single_id = jnp.sum(note * own_single.astype(jnp.float32)[:, None], axis=0) > 0.0
    held_by_me = own_single_id[p_idx]
    bad_trash = touched & trash[p_idx][None, :]
    bad_dup = touched & (dup_clued[None, :] | held_by_me[None, :] | dup_touch)
    c_bad_trash = jnp.sum(bad_trash.astype(jnp.float32), axis=1)
    c_bad = jnp.sum((bad_trash | bad_dup).astype(jnp.float32), axis=1)
    c_focus_dup = dup_clued[c_focus]

    chop_id = p_idx[p_chop]
    chop_useful = useful[chop_id] & p_has_chop
    chop_playable = playable[chop_id]
    chop_crit = critical[chop_id]
    chop_is5 = RANK_OF[chop_id] == 4
    chop_is2 = RANK_OF[chop_id] == 1
    other_copy = jnp.any((p_idx == chop_id) & p_occ & (SLOTS != p_chop))
    two_save = chop_is2 & jnp.logical_not(chop_crit) & jnp.logical_not(other_copy)
    save_needed = chop_useful & jnp.logical_not(chop_playable) & (chop_crit | chop_is5 | two_save)
    clue_is_rank5 = CLUE_IS_RANK & (CLUE_ATTR == 4)
    clue_is_rank2 = CLUE_IS_RANK & (CLUE_ATTR == 1)
    save_type_ok = jnp.where(chop_is5, clue_is_rank5,
                             jnp.where(chop_crit, jnp.ones((10,), jnp.bool_), clue_is_rank2))

    # A delayed play clue is only given when the receiver can see a clued card of
    # mine that is certainly the connecting card.
    colors = jnp.arange(5, dtype=jnp.int32)
    h_int = jnp.clip(height.astype(jnp.int32), 0, 4)
    conn_single = own_single_id[5 * colors + h_int] & (height <= 4.0)
    f_col = c_focus_id // 5
    f_rank = c_focus_id - 5 * f_col
    delayed_ok = ((f_rank == (height[f_col].astype(jnp.int32) + 1)) & conn_single[f_col]
                  & useful[c_focus_id])
    # How would the partner read the focus of this clue?  Their literal
    # possibilities after the clue are public, so a clue that would make them
    # read a playable identity onto an unplayable card can be avoided.
    p_v0 = know[1, :, 0:25] > 0.0
    attr_mask_c = (COLOR_OF[None, :] == jnp.arange(5, dtype=jnp.int32)[:, None])
    attr_mask_r = (RANK_OF[None, :] == jnp.arange(5, dtype=jnp.int32)[:, None])
    attr_mask = jnp.concatenate([attr_mask_c, attr_mask_r], axis=0)     # [10, 25]
    lit_post = p_v0[c_focus] & attr_mask                                # [10, 25]
    save_set_k = (critical[None, :]
                  | (clue_is_rank2[:, None] & ((RANK_OF == 1) & useful)[None, :]))
    chop_save_extra = c_is_chop_focus[:, None] & save_set_k
    read = lit_post & (playable[None, :] | chop_save_extra)
    read_any = jnp.any(read, axis=1)
    read_all_playable = jnp.logical_not(jnp.any(read & jnp.logical_not(playable)[None, :], axis=1))
    risky = (jnp.logical_not(playable[c_focus_id]) & read_any & read_all_playable
             & c_focus_occ)
    # Will the partner be able to act on it at once, or only write a note?
    read2 = lit_post & (play_tight[None, :] | chop_save_extra)
    resolves = (jnp.any(read2, axis=1)
                & jnp.logical_not(jnp.any(read2 & jnp.logical_not(playable)[None, :], axis=1)))

    # Good Touch Principle is enforced strictly for play clues: never touch a
    # card that is already trash, because the receiver eliminates trash by GTP.
    play_ok = (c_legal & c_any_new & (c_focus_playable | (delayed_ok & c_focus_occ))
               & jnp.logical_not(c_focus_dup) & jnp.logical_not(trash[c_focus_id])
               & (c_bad_trash <= 0.0))
    save_ok = (c_legal & c_any_new & c_is_chop_focus & save_needed & save_type_ok
               & jnp.logical_not(risky))
    colorpref = jnp.where(CLUE_IS_RANK, 0.0, 1.0)
    ambiguous = (c_is_chop_focus & (clue_is_rank5 | clue_is_rank2)).astype(jnp.float32)
    play_score = (100.0 + 2.0 * c_num_new - 12.0 * c_bad + colorpref
                  + 3.0 * c_is_chop_focus.astype(jnp.float32) - 4.0 * ambiguous
                  - 6.0 * (jnp.logical_not(c_focus_playable)).astype(jnp.float32)
                  + 18.0 * resolves.astype(jnp.float32))
    # A save clue that also touches the cards heading for the chop next is much
    # better than one that saves a single card: those cards are useful, so Good
    # Touch still holds and the partner's next discard is safe.
    protect = jnp.sum((c_new & (critical | (RANK_OF == 4))[p_idx][None, :]
                       ).astype(jnp.float32), axis=1)
    save_score = (50.0 - 5.0 * c_bad + colorpref + 2.0 * resolves.astype(jnp.float32)
                  + 4.0 * protect + 1.0 * c_num_new)
    score = jnp.where(play_ok, play_score, jnp.where(save_ok, save_score, -1000.0))
    has_play_clue = jnp.any(play_ok)
    has_save_clue = jnp.any(save_ok)
    best_play_clue = jnp.argmax(jnp.where(play_ok, play_score, -1e9)).astype(jnp.int32)
    best_save_clue = jnp.argmax(jnp.where(save_ok, save_score, -1e9)).astype(jnp.int32)
    # Tempo clue: touches no new card, so it breaks Minimum Clue Value Principle
    # and is only used when the team would otherwise stall.  Its focus is the
    # leftmost re-touched card, which the partner reads as a play clue, so it is
    # restricted to a card that really is playable now.
    tempo_ok = (c_legal & jnp.logical_not(c_any_new) & c_focus_playable & c_focus_occ
                & jnp.logical_not(risky))
    tempo_score = 10.0 - 3.0 * c_bad + colorpref
    has_tempo = jnp.any(tempo_ok)
    best_tempo_clue = jnp.argmax(jnp.where(tempo_ok, tempo_score, -1e9)).astype(jnp.int32)
    # Forced clue (8 tokens, nothing to play, no conventional clue available):
    # pick the least misleading legal clue.
    fb_score = jnp.where(c_legal,
                         20.0 * c_focus_playable.astype(jnp.float32)
                         - 60.0 * risky.astype(jnp.float32)
                         - 3.0 * c_bad - c_num_new
                         + 2.0 * jnp.logical_not(c_any_new).astype(jnp.float32),
                         -1e9)
    best_fb_clue = jnp.argmax(fb_score).astype(jnp.int32)
    del score

    # ---------------- choose an action ----------------
    # Check the partner's chop first (H-group "Check Team Chops"): with two
    # players nobody else can save it.  They will only skip their discard if
    # their own public information already makes a card certainly playable.
    p_certain = (p_occ & p_clued & jnp.any(p_v0, axis=1)
                 & jnp.logical_not(jnp.any(p_v0 & jnp.logical_not(playable)[None, :], axis=1)))
    partner_busy = jnp.any(p_certain)
    can_clue = info > 0.0
    can_discard = (info < 8.0) & jnp.any(own_occ)
    urgent = (can_clue & has_save_clue & save_needed & (chop_crit | chop_is5)
              & jnp.logical_not(partner_busy))

    # Final round: a failed play costs a life but no points, so a card that
    # might be playable is worth trying once the deck is gone.
    frac = jnp.sum(note * playablef, axis=1) / jnp.maximum(note_sum, 1.0)
    gamble_ok = own_occ & (lives >= 2.0) & (((frac > 0.0) & (deck <= 1.0))
                                           | ((frac >= 0.5) & (deck <= 4.0)))
    have_gamble = jnp.any(gamble_ok)
    gamble_slot = jnp.argmax(jnp.where(gamble_ok, frac, -1.0)).astype(jnp.int32)

    action = jnp.where(
        urgent, 10 + best_save_clue,
        jnp.where(
            have_play, 5 + play_slot,
            jnp.where(
                can_clue & has_play_clue, 10 + best_play_clue,
                jnp.where(
                    can_clue & has_save_clue, 10 + best_save_clue,
                    jnp.where(
                        (info >= 2.0) & has_tempo, 10 + best_tempo_clue,
                        jnp.where(
                            have_gamble, 5 + gamble_slot,
                            jnp.where(can_discard, discard_slot, 10 + best_fb_clue)))))))
    action = action.astype(jnp.int32)

    # Safety net: never emit an illegal action.
    order = jnp.concatenate([jnp.arange(10, 20, dtype=jnp.int32),
                             jnp.arange(0, 5, dtype=jnp.int32),
                             jnp.arange(5, 10, dtype=jnp.int32)])
    safe = order[jnp.argmax(legal[order])]
    action = jnp.where(legal[action], action, safe)
    action = jnp.where(legal[20], jnp.asarray(20, jnp.int32), action)

    next_memory = {
        'note': note,
        'pending': pending,
        'promised': promised,
        'prev_clued': own_clued,
    }
    trace = {
        'action': action,
        'gfocus': jnp.where((action >= 10) & (action < 20),
                            c_focus[jnp.clip(action - 10, 0, 9)], -1).astype(jnp.int32),
        # 0 urgent save, 1 play, 2 play clue, 3 save clue, 4 tempo clue,
        # 5 endgame gamble, 6 discard, 7 forced clue.
        'cat': jnp.where(
            urgent, 0,
            jnp.where(
                have_play, 1,
                jnp.where(
                    can_clue & has_play_clue, 2,
                    jnp.where(
                        can_clue & has_save_clue, 3,
                        jnp.where(
                            (info >= 2.0) & has_tempo, 4,
                            jnp.where(have_gamble, 5,
                                      jnp.where(can_discard, 6, 7))))))).astype(jnp.int32),
        'focus': jnp.where(apply_hint, focus, -1).astype(jnp.int32),
        'chop_focus': is_chop_focus & apply_hint,
        'prompt': do_prompt,
        'n_known_playable': jnp.sum(known_playable.astype(jnp.int32)),
        'n_known_trash': jnp.sum(known_trash.astype(jnp.int32)),
        'note_size': jnp.sum(note, axis=1),
        'info': info,
        'lives': lives,
        'deck': deck,
        'score': jnp.sum(height),
    }
    return action, next_memory, trace
