"""Two-player Hanabi policy following H-group Level 1 conventions.

Design notes (see ledger A01):

* Literal knowledge comes from the native V0 block (``V0 > 0`` is exactly the
  set of identities still literally possible for a slot).
* Private memory only stores *convention* inferences: per-slot candidate notes
  plus "focused"/"could be a save" flags, for both hands, kept in sync with
  slot shifts and clue focus.
* Clue focus uses the documented 3-step rule on the pre-clue hand, with the
  runtime mapping's native ordering (chop = lowest occupied unclued index,
  "leftmost" = highest occupied index).
* A clue focusing a non-chop card is a play clue.  A clue newly touching the
  chop may additionally be a 5 Save / 2 Save / Critical Save, so the focus note
  keeps both interpretations.
* Good Touch Principle: a touched card's note is intersected with the set of
  non-trash identities; a card is played only when every surviving candidate is
  playable right now, which makes every play bomb-free under correct notes.
* Clue giving respects Minimum Clue Value Principle (the focus must be newly
  touched), avoids bad touches, prefers colour clues and checks - by simulating
  the recipient's note - that the clue will actually be understood as a play.
"""

import jax
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

NC = 5
NR = 5
NS = 5
NID = 25

NEG = -1.0e9

# Flat identity index is 5 * colour + rank, matching the native V0 layout.
_IDC_LIST = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
             3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
_IDR_LIST = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
             0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
_COPIES_LIST = [3, 2, 2, 2, 1]
_SLOTS_LIST = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _shift1(arr, slot, fill):
    """Remove ``slot`` from a [5] array, shift left, append ``fill``."""
    slots = jnp.asarray(_SLOTS_LIST)
    src = jnp.minimum(jnp.where(slots < slot, slots, slots + 1), NS - 1)
    out = jnp.take(arr, src, axis=0)
    return jnp.where(slots == NS - 1, fill, out)


def _shift2(arr, slot, fill):
    """Remove ``slot`` from a [5, 25] array, shift left, append ``fill`` row."""
    slots = jnp.asarray(_SLOTS_LIST)
    src = jnp.minimum(jnp.where(slots < slot, slots, slots + 1), NS - 1)
    out = jnp.take(arr, src, axis=0)
    return jnp.where((slots == NS - 1)[:, None], fill, out)


def _lowest(mask):
    """Lowest index where ``mask`` is true (0 when empty)."""
    return jnp.argmax(mask).astype(jnp.int32)


def _highest(mask):
    """Highest index where ``mask`` is true (0 when empty)."""
    return (NS - 1 - jnp.argmax(mask[::-1])).astype(jnp.int32)


def _chop(touched, occupied):
    avail = occupied & jnp.logical_not(touched)
    return _lowest(avail), jnp.any(avail)


def _focus(matches, pre_touched, occupied):
    """Documented 3-step focus rule on the pre-clue hand."""
    touch = matches & occupied
    new = touch & jnp.logical_not(pre_touched)
    any_new = jnp.any(new)
    chop_i, has_chop = _chop(pre_touched, occupied)
    chop_new = has_chop & jnp.take(new, chop_i)
    focus = jnp.where(
        any_new,
        jnp.where(chop_new, chop_i, _highest(new)),
        _highest(touch),
    )
    return focus.astype(jnp.int32), any_new, chop_new


# --------------------------------------------------------------------------
# observation decoding
# --------------------------------------------------------------------------

def _decode(obs):
    f = obs.features
    d = {}
    d['partner_hand'] = f[0:125].reshape(NS, NID)
    d['deck'] = jnp.sum(f[127:167]).astype(jnp.int32)
    fw = f[167:192].reshape(NC, NR)
    d['height'] = jnp.sum(fw * jnp.arange(1, NR + 1)[None, :], axis=1).astype(jnp.int32)
    d['tokens'] = jnp.sum(f[192:200]).astype(jnp.int32)
    d['lives'] = jnp.sum(f[200:203]).astype(jnp.int32)
    dis = f[203:253].reshape(NC, 10)
    d['disc'] = jnp.stack(
        [jnp.sum(dis[:, 0:3], 1), jnp.sum(dis[:, 3:5], 1), jnp.sum(dis[:, 5:7], 1),
         jnp.sum(dis[:, 7:9], 1), dis[:, 9]], axis=1).astype(jnp.int32)
    d['know'] = f[308:658].reshape(2, NS, 35)
    d['la'] = f[253:308]
    return d


def _board_sets(height, disc):
    """Trash / playable / critical masks, shaped [5 colours, 5 ranks]."""
    ranks = jnp.arange(NR)[None, :]
    played = ranks < height[:, None]
    copies = jnp.asarray(_COPIES_LIST, dtype=jnp.int32)
    rem = copies[None, :] - disc - played.astype(jnp.int32)
    dead = (rem <= 0) & jnp.logical_not(played)
    unreachable = jnp.cumsum(dead.astype(jnp.int32), axis=1) > 0
    trash = played | unreachable
    playable = (ranks == height[:, None]) & jnp.logical_not(trash)
    critical = (rem == 1) & jnp.logical_not(played) & jnp.logical_not(unreachable)
    return trash, playable, critical


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

def init_agent():
    return {
        'note': jnp.ones((2, NS, NID), dtype=bool),
        'play': jnp.zeros((2, NS), dtype=bool),
        'save': jnp.zeros((2, NS), dtype=bool),
        'touched': jnp.zeros((2, NS), dtype=bool),
    }


def agent_act(obs, reset, memory, rng):
    IDC = jnp.asarray(_IDC_LIST, dtype=jnp.int32)
    IDR = jnp.asarray(_IDR_LIST, dtype=jnp.int32)
    SLOTS = jnp.asarray(_SLOTS_LIST, dtype=jnp.int32)
    init = init_agent()
    mem = {k: jnp.where(reset, init[k], memory[k]) for k in init}

    d = _decode(obs)
    legal = obs.legal_actions
    height = d['height']
    trash, playable, critical = _board_sets(height, d['disc'])
    trash_f = trash.reshape(NID)
    playable_f = playable.reshape(NID)
    crit_f = critical.reshape(NID)

    # literal possibility masks for both hands, plus occupancy
    lit = d['know'][:, :, 0:25] > 0.0                      # [2, 5, 25]
    own_occ = jnp.any(lit[0], axis=-1)
    p_hand = d['partner_hand']
    p_occ = jnp.any(p_hand > 0.0, axis=-1)

    # The native V0 mask deliberately ignores the partner's visible hand.  Every
    # copy we can see there is a copy we cannot be holding, which is an exact
    # deduction rather than a convention.
    copies_r = jnp.asarray(_COPIES_LIST, dtype=jnp.int32)
    ranks_m = jnp.arange(NR)[None, :]
    played_m = ranks_m < height[:, None]
    seen_partner = jnp.sum(jnp.where(p_occ[:, None], p_hand > 0.0, False),
                           axis=0).astype(jnp.int32)
    avail = (jnp.tile(copies_r, (NC,)) - d['disc'].reshape(NID)
             - played_m.reshape(NID).astype(jnp.int32) - seen_partner)
    avail_mask = avail > 0
    own_lit = lit[0] & avail_mask[None, :]
    own_lit = jnp.where(jnp.any(own_lit, axis=-1)[:, None], own_lit, lit[0])
    lit = jnp.stack([own_lit, lit[1]], axis=0)
    occ = jnp.stack([own_occ, p_occ], axis=0)
    p_id = jnp.argmax(p_hand, axis=-1).astype(jnp.int32)
    p_c = p_id // NR
    p_r = p_id % NR

    # ---------------- last public action ----------------
    la = d['la']
    la_type = la[2:6]
    is_move = (la_type[0] + la_type[1]) > 0.5
    is_hint = (la_type[2] + la_type[3]) > 0.5
    is_color_hint = la_type[2] > 0.5
    actor = jnp.argmax(la[0:2]).astype(jnp.int32)
    recip = jnp.argmax(la[6:8]).astype(jnp.int32)
    hc = jnp.argmax(la[8:13]).astype(jnp.int32)
    hr = jnp.argmax(la[13:18]).astype(jnp.int32)
    la_match = la[18:23] > 0.5
    removed = jnp.argmax(la[23:28]).astype(jnp.int32)

    # ---- (a) apply hand shift for a play/discard ----
    rows = jnp.arange(2)
    move_sel = (rows == actor) & is_move
    note_s = jnp.stack([_shift2(mem['note'][i], removed, jnp.ones((NID,), dtype=bool))
                        for i in range(2)], axis=0)
    mem = {
        'note': jnp.where(move_sel[:, None, None], note_s, mem['note']),
        'play': jnp.where(move_sel[:, None],
                          jnp.stack([_shift1(mem['play'][i], removed, False) for i in range(2)]),
                          mem['play']),
        'save': jnp.where(move_sel[:, None],
                          jnp.stack([_shift1(mem['save'][i], removed, False) for i in range(2)]),
                          mem['save']),
        'touched': jnp.where(move_sel[:, None],
                             jnp.stack([_shift1(mem['touched'][i], removed, False) for i in range(2)]),
                             mem['touched']),
    }

    # ---- (b) connected (delayed-play) heights from publicly gotten cards ----
    own_cand_pre = mem['note'][0] & lit[0] & jnp.logical_not(trash_f)[None, :]
    own_single = own_occ & (jnp.sum(own_cand_pre, axis=-1) == 1)
    own_sid = jnp.argmax(own_cand_pre, axis=-1)
    queued = jnp.zeros((NID,), dtype=bool)
    queued = queued | jnp.any(
        (jnp.arange(NID)[None, :] == (p_c * NR + p_r)[:, None])
        & (p_occ & mem['touched'][1])[:, None], axis=0)
    queued = queued | jnp.any(
        (jnp.arange(NID)[None, :] == own_sid[:, None]) & own_single[:, None], axis=0)
    qmat = queued.reshape(NC, NR)

    def _chain(start):
        cur = start
        for _ in range(NR - 1):
            idx = jnp.clip(cur, 0, NR - 1)
            nxt = jnp.take_along_axis(qmat, idx[:, None], axis=1)[:, 0] & (cur < NR)
            cur = cur + nxt.astype(jnp.int32)
        return cur

    # Conservative chain: cards whose identity is publicly settled for us.
    conn_give = _chain(height)

    # For clues we *give*, only route a delayed play clue through a partner card
    # whose owner can already identify it.  An independently written partner
    # will simply hold a card it cannot name, so such a clue would stall.
    p_cand_pre = mem['note'][1] & lit[1] & jnp.logical_not(trash_f)[None, :]
    p_settled = (p_occ & mem['touched'][1] & jnp.any(p_cand_pre, axis=-1)
                 & ((jnp.sum(p_cand_pre, axis=-1) == 1)
                    | jnp.all(jnp.logical_not(p_cand_pre) | playable_f[None, :],
                              axis=-1)))
    q_settled = jnp.any(
        (jnp.arange(NID)[None, :] == (p_c * NR + p_r)[:, None]) & p_settled[:, None],
        axis=0) | jnp.any(
        (jnp.arange(NID)[None, :] == own_sid[:, None]) & own_single[:, None], axis=0)
    qs = q_settled.reshape(NC, NR)
    conn_set = height
    for _ in range(NR - 1):
        nx = jnp.take_along_axis(qs, jnp.clip(conn_set, 0, NR - 1)[:, None],
                                 axis=1)[:, 0] & (conn_set < NR)
        conn_set = conn_set + nx.astype(jnp.int32)

    # Generous chain for *interpreting* a partner clue.  The clue giver can see
    # identities we cannot, so they may route a delayed play clue through one of
    # our own clued cards (a self-prompt).  Allow one extra step per colour
    # supplied by any of our touched cards that could hold the connector.
    own_touch_any = jnp.any(
        (mem['touched'][0] & own_occ)[:, None] & own_cand_pre, axis=0).reshape(NC, NR)
    conn_interp = conn_give
    for _ in range(2):
        idx_g = jnp.clip(conn_interp, 0, NR - 1)
        extend = (jnp.take_along_axis(own_touch_any, idx_g[:, None], axis=1)[:, 0]
                  & (conn_interp < NR))
        conn_interp = _chain(conn_interp + extend.astype(jnp.int32))

    ranks = jnp.arange(NR)[None, :]
    play_give = ((ranks >= height[:, None]) & (ranks <= conn_set[:, None])
                 & jnp.logical_not(trash)).reshape(NID)
    play_f = ((ranks >= height[:, None]) & (ranks <= conn_interp[:, None])
              & jnp.logical_not(trash)).reshape(NID)

    # ---- (c) interpret a hint that was just given ----
    save_c = crit_f & (IDC == hc)
    save_r = ((crit_f & (IDR == hr))
              | ((hr == 1) & (IDR == 1) & jnp.logical_not(trash_f))
              | ((hr == 4) & (IDR == 4) & jnp.logical_not(trash_f)))
    save_k = jnp.where(is_color_hint, save_c, save_r)

    pre_t = jnp.take(mem['touched'], recip, axis=0)
    occ_r = jnp.take(occ, recip, axis=0)
    lit_r = jnp.take(lit, recip, axis=0)
    f_idx, any_new, chop_new = _focus(la_match, pre_t, occ_r)
    lit_focus = jnp.take(lit_r, f_idx, axis=0)
    interp = play_f | jnp.where(chop_new, save_k, jnp.zeros((NID,), dtype=bool))
    cand_focus = lit_focus & interp & jnp.logical_not(trash_f)
    fallback = lit_focus & jnp.logical_not(trash_f)
    fallback = jnp.where(jnp.any(fallback), fallback, lit_focus)
    new_note = jnp.where(jnp.any(cand_focus), cand_focus, fallback)

    # A clue touching no new cards is a Tempo Clue: it still focuses the
    # leftmost re-touched card and asks for it to be played.
    onehot_f = (SLOTS == f_idx) & is_hint
    note_r = jnp.where(onehot_f[:, None], new_note[None, :],
                       jnp.take(mem['note'], recip, axis=0))
    play_r = jnp.take(mem['play'], recip, axis=0) | onehot_f
    save_r_flags = jnp.take(mem['save'], recip, axis=0) | (onehot_f & chop_new)


    hint_sel = (rows == recip) & is_hint
    mem = {
        'note': jnp.where(hint_sel[:, None, None], note_r[None, :, :], mem['note']),
        'play': jnp.where(hint_sel[:, None], play_r[None, :], mem['play']),
        'save': jnp.where(hint_sel[:, None], save_r_flags[None, :], mem['save']),
        'touched': mem['touched'],
    }

    # ---- (d) refresh literal touched flags from the observation ----
    touched_now = jnp.any(d['know'][:, :, 25:35] > 0.0, axis=-1) & occ
    mem = dict(mem, touched=touched_now)

    # ======================================================================
    # decision
    # ======================================================================
    own_cand0 = mem['note'][0] & lit[0] & jnp.logical_not(trash_f)[None, :]
    # Good Touch Principle: every clued card will eventually be played, so a
    # clued card of ours cannot duplicate an identity already clued elsewhere.
    ids_row = jnp.arange(NID)[None, :]
    p_committed = jnp.any((ids_row == (p_c * NR + p_r)[:, None])
                          & (p_occ & mem['touched'][1])[:, None], axis=0)
    o_single = own_occ & mem['touched'][0] & (jnp.sum(own_cand0, axis=-1) == 1)
    o_sid = jnp.argmax(own_cand0, axis=-1)
    o_onehot = (ids_row == o_sid[:, None]) & o_single[:, None]          # [5,25]
    o_excl = jnp.any(o_onehot, axis=0)[None, :] & jnp.logical_not(o_onehot)
    elim = ((p_committed[None, :] | o_excl) & mem['touched'][0][:, None]
            & (d['lives'] >= 2))
    cand_gtp = own_cand0 & jnp.logical_not(elim)
    own_cand = jnp.where(jnp.any(cand_gtp, axis=-1)[:, None], cand_gtp, own_cand0)
    own_any = jnp.any(own_cand, axis=-1)
    own_allplay = jnp.all(jnp.logical_not(own_cand) | playable_f[None, :], axis=-1)
    own_trashy = own_occ & jnp.logical_not(own_any)

    # Good Touch Principle lets us drop trash identities from a clued card's
    # candidate set, but if the team ever bad-touched a duplicate that inference
    # can be wrong.  Weigh that risk with the native V0 weights and tighten the
    # threshold as lives run out, because the last life ends the game with 0.
    strict = mem['note'][0] & lit[0]
    w_all = jnp.maximum(avail, 0).astype(jnp.float32)[None, :]
    w_den = jnp.maximum(jnp.sum(w_all * strict.astype(jnp.float32), axis=-1), 1e-6)
    p_trash = jnp.sum(w_all * (strict & trash_f[None, :]).astype(jnp.float32),
                      axis=-1) / w_den
    risk_cap = jnp.where(d['lives'] >= 3, 0.6,
                         jnp.where(d['lives'] >= 2, 0.2, 0.0))
    ok_play = own_any & own_allplay & (p_trash <= risk_cap)
    can_play = own_occ & ok_play & legal[5:10]

    min_rank = jnp.min(jnp.where(own_cand, IDR[None, :], NR + 1), axis=-1)
    play_score = jnp.where(can_play,
                           100.0 - 10.0 * min_rank.astype(jnp.float32)
                           + mem['play'][0].astype(jnp.float32)
                           - 0.1 * SLOTS.astype(jnp.float32),
                           NEG)
    play_slot = jnp.argmax(play_score)
    have_play = jnp.any(can_play)

    # ---- partner model ----
    p_cand = mem['note'][1] & lit[1] & jnp.logical_not(trash_f)[None, :]
    p_any = jnp.any(p_cand, axis=-1)
    p_allplay = jnp.all(jnp.logical_not(p_cand) | playable_f[None, :], axis=-1)
    p_will_play = p_occ & p_any & p_allplay
    p_has_trash = jnp.any(p_occ & mem['touched'][1] & jnp.logical_not(p_any))
    partner_busy = jnp.any(p_will_play) | p_has_trash

    # ---- partner chop and its save need ----
    chop_i, has_chop = _chop(mem['touched'][1], p_occ)
    cc = jnp.take(p_c, chop_i)
    cr = jnp.take(p_r, chop_i)
    cid = cc * NR + cr
    chop_trash = jnp.take(trash_f, cid)
    chop_crit = jnp.take(crit_f, cid)
    chop_playable = jnp.take(playable_f, cid)
    dup_in_hand = jnp.sum(jnp.where(p_occ, (p_c * NR + p_r) == cid, False)) > 1
    is_five = cr == 4
    is_two = cr == 1
    needs_save = (has_chop & jnp.logical_not(chop_trash)
                  & (is_five | chop_crit
                     | (is_two & jnp.logical_not(dup_in_hand))
                     | chop_playable))

    # ---- evaluate all ten clues ----
    ar = jnp.arange(NC)
    match_color = (p_c[None, :] == ar[:, None]) & p_occ[None, :]
    match_rank = (p_r[None, :] == ar[:, None]) & p_occ[None, :]
    matches_all = jnp.concatenate([match_color, match_rank], axis=0)         # [10,5]
    attr_all = jnp.concatenate([IDC[None, :] == ar[:, None],
                                IDR[None, :] == ar[:, None]], axis=0)        # [10,25]
    save_all = jnp.concatenate(
        [crit_f[None, :] & (IDC[None, :] == ar[:, None]),
         (crit_f[None, :] & (IDR[None, :] == ar[:, None]))
         | ((ar == 1)[:, None] & (IDR == 1)[None, :] & jnp.logical_not(trash_f)[None, :])
         | ((ar == 4)[:, None] & (IDR == 4)[None, :] & jnp.logical_not(trash_f)[None, :])],
        axis=0)                                                              # [10,25]
    is_color_clue = jnp.concatenate([jnp.ones(NC, bool), jnp.zeros(NR, bool)])
    clue_rank = jnp.concatenate([jnp.full((NC,), -1, jnp.int32), ar])
    legal_clue = legal[10:20]

    pre_touch1 = mem['touched'][1]
    note1 = mem['note'][1]
    lit1 = lit[1]
    p_ids = p_c * NR + p_r
    # identities already accounted for in our own hand (Good Touch: never
    # duplicate them in the partner's hand)
    own_now_single = own_occ & (jnp.sum(own_cand, axis=-1) == 1)
    own_committed = jnp.any(
        (jnp.arange(NID)[None, :] == jnp.argmax(own_cand, axis=-1)[:, None])
        & own_now_single[:, None], axis=0)

    def eval_clue(matches, attr, save_set):
        touch = matches & p_occ
        new = touch & jnp.logical_not(pre_touch1)
        n_new = jnp.sum(new)
        f_i, anyn, chopn = _focus(matches, pre_touch1, p_occ)

        # bad touch: newly touched trash, duplicate of an already touched card,
        # or duplicate of another newly touched card further left
        same_touched = jnp.any((p_ids[None, :] == p_ids[:, None])
                               & (touch & pre_touch1)[None, :]
                               & (SLOTS[None, :] != SLOTS[:, None]), axis=1)
        same_new_left = jnp.any((p_ids[None, :] == p_ids[:, None])
                                & new[None, :]
                                & (SLOTS[None, :] > SLOTS[:, None]), axis=1)
        bad = new & (jnp.take(trash_f, p_ids) | same_touched | same_new_left
                     | jnp.take(own_committed, p_ids))
        n_bad = jnp.sum(bad)

        lit_after = jnp.where(touch[:, None], lit1 & attr[None, :],
                              lit1 & jnp.logical_not(attr)[None, :])
        interp_k = play_f | jnp.where(chopn, save_set, jnp.zeros((NID,), dtype=bool))
        cf = (jnp.take(lit_after, f_i, axis=0) & jnp.take(note1, f_i, axis=0)
              & interp_k & jnp.logical_not(trash_f))
        clean = jnp.any(cf) & jnp.all(jnp.logical_not(cf) | play_f)
        now = jnp.any(cf) & jnp.all(jnp.logical_not(cf) | playable_f)

        fid = jnp.take(p_ids, f_i)
        focus_ok = jnp.take(play_give, fid)
        focus_now = jnp.take(playable_f, fid)
        focus_chop = anyn & chopn
        # A pure re-clue is a Tempo Clue; it is only safe when the re-touched
        # focus is genuinely playable.
        tempo_ok = jnp.logical_not(anyn) & jnp.any(touch) & focus_now
        n_new_play = jnp.sum(new & jnp.take(playable_f, p_ids))
        return n_new, n_bad, clean, now, focus_ok, focus_chop, tempo_ok, n_new_play

    (n_new, n_bad, clean, now, focus_ok, focus_chop, tempo_ok,
     n_new_play) = jax.vmap(eval_clue)(matches_all, attr_all, save_all)

    valid_play = (legal_clue & (n_new >= 1) & focus_ok & (n_bad == 0) & clean)
    base = (4.0 * n_new.astype(jnp.float32)
            + 0.0 * n_new_play.astype(jnp.float32)
            + 12.0 * now.astype(jnp.float32)
            + 2.0 * is_color_clue.astype(jnp.float32)
            + 3.0 * focus_chop.astype(jnp.float32)
            - 18.0 * n_bad.astype(jnp.float32))
    sc_play = jnp.where(valid_play, base, NEG)

    save_type_ok = jnp.where(
        is_five, clue_rank == 4,
        jnp.where(chop_crit, jnp.ones((10,), dtype=bool),
                  jnp.where(is_two, clue_rank == 1, jnp.zeros((10,), dtype=bool))))
    touches_chop = matches_all[:, :] & p_occ[None, :]
    touches_chop = jnp.take(touches_chop, chop_i, axis=1) & has_chop
    valid_save = legal_clue & needs_save & touches_chop & save_type_ok
    sc_save = jnp.where(valid_save,
                        3.0 * n_new.astype(jnp.float32)
                        + 2.0 * is_color_clue.astype(jnp.float32)
                        - 6.0 * n_bad.astype(jnp.float32), NEG)

    # Protection: a play clue focusing the chop also protects it.  Save
    # Principle is allowed to override Good Touch, so as a last resort any legal
    # clue that touches the chop is better than losing a critical card.
    sc_prot = jnp.maximum(jnp.where(valid_save, sc_save + 50.0, NEG),
                          jnp.where(valid_play & focus_chop, sc_play + 60.0, NEG))
    sc_prot = jnp.maximum(
        sc_prot,
        jnp.where(legal_clue & touches_chop & needs_save,
                  -10.0 * n_bad.astype(jnp.float32)
                  + 2.0 * is_color_clue.astype(jnp.float32) - 40.0, NEG))
    # Stall clues are only for forced situations.  A Tempo Clue on an already
    # clued, genuinely playable card is the safest of them.
    sc_stall = jnp.where(legal_clue,
                         40.0 * tempo_ok.astype(jnp.float32)
                         - 20.0 * n_bad.astype(jnp.float32)
                         - 4.0 * n_new.astype(jnp.float32), NEG)

    have_clue_play = jnp.any(valid_play)
    have_clue_save = jnp.any(valid_save)
    have_prot = jnp.any(sc_prot > NEG / 2.0)
    have_stall = jnp.any(legal_clue)
    have_tempo = jnp.any(legal_clue & tempo_ok)
    a_tempo = 10 + jnp.argmax(jnp.where(legal_clue & tempo_ok, sc_stall, NEG))

    # ---- final-round gamble ----
    # Once the deck is empty at most two turns remain, so an unsuccessful play
    # costs only the card itself as long as it is not our last life.  Use the
    # native V0 weights restricted to our convention note as the probability.
    weights = (jnp.maximum(avail, 0).astype(jnp.float32)[None, :]
               * own_cand.astype(jnp.float32))
    wsum = jnp.sum(weights, axis=-1)
    pplay = jnp.sum(weights * playable_f[None, :].astype(jnp.float32), axis=-1) / jnp.maximum(wsum, 1e-6)
    endgame = d['deck'] <= 0
    ochop_i, ohas_chop = _chop(mem['touched'][0], own_occ)
    safe_discard = jnp.any(legal[0:5]
                           & (own_trashy | ((SLOTS == ochop_i) & ohas_chop)))
    # Also gamble rather than throw away a clued card the team chose to keep.
    forced = jnp.logical_not(safe_discard) & (d['lives'] >= 3)
    thresh = jnp.where(endgame, 0.05, 0.7)
    gamble_ok = own_occ & legal[5:10] & (pplay > thresh) & (wsum > 0.0)
    gamble_slot = jnp.argmax(jnp.where(gamble_ok, pplay, NEG))
    have_gamble = (endgame | forced) & (d['lives'] >= 2) & jnp.any(gamble_ok)
    a_gamble = 5 + gamble_slot

    a_play = 5 + play_slot
    a_cplay = 10 + jnp.argmax(sc_play)
    a_csave = 10 + jnp.argmax(sc_save)
    a_cprot = 10 + jnp.argmax(sc_prot)
    a_stall = 10 + jnp.argmax(sc_stall)

    # ---- discard choice ----
    # Probability that a slot holds a critical card, from the native V0 weights
    # restricted to the convention note.
    wn = d['know'][0, :, 0:25] * own_cand.astype(jnp.float32)
    wden = jnp.maximum(jnp.sum(wn, axis=-1), 1e-6)
    p_crit = jnp.sum(wn * crit_f[None, :].astype(jnp.float32), axis=-1) / wden
    d_score = (1000.0 * own_trashy.astype(jnp.float32)
               + 500.0 * ((SLOTS == ochop_i) & ohas_chop).astype(jnp.float32)
               + 100.0
               - 40.0 * mem['play'][0].astype(jnp.float32)
               - 60.0 * mem['save'][0].astype(jnp.float32)
               - 200.0 * p_crit
               - 1.0 * SLOTS.astype(jnp.float32))
    d_score = jnp.where(legal[0:5], d_score, NEG)
    a_discard = jnp.argmax(d_score)
    # Discarding an already clued card that is not known trash throws away
    # exactly the cards the team decided to keep.  Prefer a stall clue then.
    can_discard = jnp.any(legal[0:5]) & (safe_discard
                                         | jnp.logical_not(have_tempo))

    # Urgent: the partner would otherwise discard the chop on their next turn.
    # Insurance: a 5 or a critical card is worth a spare token even if we think
    # the partner is busy, because that prediction can be wrong.
    # "Give Play Clues Over Save Clues": a clue the partner can act on at once
    # also stops them discarding their chop this turn.
    have_now_clue = jnp.any(valid_play & now)
    urgent = (needs_save & jnp.logical_not(partner_busy)
              & jnp.logical_not(have_now_clue))
    # If the partner has nothing to do, handing them a playable card first costs
    # nothing in tempo and saves them a blind discard.
    clue_first = (have_now_clue & jnp.logical_not(partner_busy)
                  & (d['tokens'] >= 2))
    insurance = needs_save & (d['tokens'] >= 9)
    protect = (urgent | insurance) & have_prot
    save_missed = urgent & jnp.logical_not(have_prot)
    ladder = jnp.asarray(20, dtype=jnp.int32)
    ladder = jnp.where(have_stall, a_stall, ladder)
    ladder = jnp.where(jnp.any(legal[0:5]), a_discard, ladder)
    ladder = jnp.where(have_tempo, a_tempo, ladder)
    ladder = jnp.where(can_discard, a_discard, ladder)
    ladder = jnp.where(have_clue_save, a_csave, ladder)
    ladder = jnp.where(have_gamble, a_gamble, ladder)
    ladder = jnp.where(have_clue_play, a_cplay, ladder)
    ladder = jnp.where(have_play, a_play, ladder)
    ladder = jnp.where(clue_first, a_cplay, ladder)
    action = jnp.where(protect, a_cprot, ladder)
    action = action.astype(jnp.int32)

    fallback_action = jnp.argmax(legal[0:20].astype(jnp.int32)).astype(jnp.int32)
    fallback_action = jnp.where(jnp.any(legal[0:20]), fallback_action,
                                jnp.asarray(20, dtype=jnp.int32))
    action = jnp.where(jnp.take(legal, action), action, fallback_action)

    inactive = legal[20]
    action = jnp.where(inactive, jnp.asarray(20, dtype=jnp.int32), action)

    ceiling = jnp.sum(jnp.max(jnp.where(jnp.logical_not(trash) | (jnp.arange(NR)[None, :]
                                                                  < height[:, None]),
                                        jnp.arange(1, NR + 1)[None, :], 0), axis=1))
    trace = {
        'have_play': have_play,
        'have_clue_play': have_clue_play,
        'protect': protect,
        'needs_save': needs_save,
        'tokens': d['tokens'],
        'lives': d['lives'],
        'deck': d['deck'],
        'score': jnp.sum(height),
        'ceiling': ceiling.astype(jnp.int32),
        'active': jnp.logical_not(inactive),
        'save_missed': save_missed & jnp.logical_not(inactive),
        'idle': jnp.logical_not(inactive) & jnp.logical_not(have_play)
        & jnp.logical_not(have_clue_play) & jnp.logical_not(protect),
        'blocked': jnp.logical_not(inactive) & jnp.logical_not(have_play)
        & jnp.logical_not(have_clue_play) & (d['tokens'] > 0)
        & jnp.any(p_occ & jnp.logical_not(mem['touched'][1])
                  & jnp.take(playable_f, p_c * NR + p_r)),
        'strict_block': jnp.logical_not(inactive) & jnp.logical_not(have_play)
        & jnp.any(own_occ & own_any & own_allplay & legal[5:10]),
        'b_bad': jnp.any(legal_clue & (n_new >= 1) & focus_ok & clean & (n_bad > 0)),
        'b_clean': jnp.any(legal_clue & (n_new >= 1) & focus_ok & (n_bad == 0)
                           & jnp.logical_not(clean)),
        'b_focus': jnp.logical_not(jnp.any(legal_clue & (n_new >= 1) & focus_ok)),
        'act_kind': jnp.where(
            inactive | (action > 4), 0,
            jnp.where(jnp.take(own_trashy, jnp.minimum(action, 4)), 2,
                      jnp.where((action == ochop_i) & ohas_chop, 1, 3))).astype(jnp.int32),
        'la_kind': jnp.where(
            la_type[1] > 0.5, 3,
            jnp.where(la_type[0] > 0.5, jnp.where(la[53] > 0.5, 1, 2),
                      jnp.where(is_hint, 4, 0))).astype(jnp.int32),
    }
    return action, mem, trace
