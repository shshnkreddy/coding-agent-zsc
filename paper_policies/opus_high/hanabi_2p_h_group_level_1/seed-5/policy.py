"""Two-player Hanabi policy following H-group Level 1 conventions.

Coordination protocol (public, shared with any Level-1 partner):

* Chop = lowest occupied native index with no positive clue flag.
* Clue focus = chop if the chop was newly touched, else the highest native
  index among newly touched cards, else the highest re-touched index.
* A clue focusing a non-chop card is a Play Clue (possibly delayed).
* A clue focusing the chop is a Play Clue or one of the three permitted Save
  Clues (5 Save with rank 5, 2 Save with rank 2, Critical Save with either).
* Good Touch Principle prunes candidates of clued cards (already-played ranks,
  exhausted identities, identities already clued in the partner's hand).
* Minimum Clue Value Principle: ordinary clues must touch at least one new card
  and must be a legal play or save clue; otherwise discard.
* Save Principle: never let the partner discard a 5, a 2, a critical card or a
  unique playable card; this may override Good Touch.
* Tempo Clue: when the only alternative is throwing away a card, a clue that
  re-touches an already clued card that is playable right now is given instead.
  Its focus is the leftmost re-touched card, exactly as Level 1 specifies for a
  clue that introduces no new card, so it reads as a play clue on that card.

No private signalling is used: every inference is derived from the public
clue/play/discard history and the documented Level-1 meanings.  Convention
inferences are held as strong-but-not-certain evidence on top of the literal
possibilities, so a partner whose tie-breaks, risk thresholds or save choices
differ cannot push this policy into a confident misplay.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hgroup_level1_2p',
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
NCOL = 5
NRANK = 5
NCARD = 25

_A5 = [0, 1, 2, 3, 4]
_A10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# card index k = 5 * color + rank
_K_COLOR = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
_K_RANK = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
_RANK_TOTAL = [3.0, 2.0, 2.0, 2.0, 1.0]
_T = True
_F = False
_EYE5 = [[_T, _F, _F, _F, _F],
         [_F, _T, _F, _F, _F],
         [_F, _F, _T, _F, _F],
         [_F, _F, _F, _T, _F],
         [_F, _F, _F, _F, _T]]
# [s, j] -> j > s
_JGT = [[_F, _T, _T, _T, _T],
        [_F, _F, _T, _T, _T],
        [_F, _F, _F, _T, _T],
        [_F, _F, _F, _F, _T],
        [_F, _F, _F, _F, _F]]

# Tunable weights for clue scoring.
W_PLAY_NOW = 12.0
W_PLAY_DELAYED = 6.0
W_SAVE = 6.0
W_SAVE_CRIT = 4.0
W_GOOD_TOUCH = 2.0
W_EXTRA_PLAYABLE = 7.0
W_BAD_TOUCH = 9.0
W_COLOR = 0.6
W_RANK_PENALTY = 0.25
W_URGENT = 60.0
W_KILL_CHOP = 4.0
W_FAR = 9.0
W_TEMPO = 10.0

# Interpretation of an incoming Play Clue: 'exact' pins the card to the next
# card of its suit, 'strict' keeps the whole playable-through-promised range,
# 'generous' widens the range with every pending clued card in my own hand.
RECV_MODE = 'generous'
# Whether to give Delayed Play Clues as well as immediate ones.
GIVE_DELAYED = True
# Which partner cards count as "accounted for" connectors: every still-useful
# clued card ('touched') or only the ones I gave an explicit play clue to.
ACCT_MODE = 'touched'
# Forbid an immediate play clue while a card of that suit is already promised.
GUARD_PLAY_NOW = False
# Good Touch pruning of duplicate identities inside my own hand.
OWN_PIN = False
# Skip save clues for cards already gotten elsewhere.
SAVE_SKIP_GOTTEN = True
# Soft-belief discounts and play thresholds.
CONV_DISCOUNT = 0.02
GT_DISCOUNT = 0.2
PLAY_T3 = 0.90
PLAY_T2 = 0.95
PLAY_T1 = 0.99
TRASH_T = 0.95
# Fallback discard when every card is clued: 'ev' minimises expected loss.
DISCARD_MODE = 'old'
# Allow a Tempo Clue when the alternative is discarding a useful clued card.
ALLOW_TEMPO = True
# Also prefer a tempo clue over a chop discard while clue tokens are plentiful.
TEMPO_TOKENS = 0.5
FAR_INCLUDES_FOCUS = True
# Endgame: once the deck is empty a failed play only costs the card, so the
# play threshold may be relaxed while spare lives remain.
PLAY_TEND = 0.55
END_DECK = 0.5
# Partner is "loaded" (will not discard next turn) only if a clued card of
# theirs is playable right now or is known trash they would discard instead.
LOADED_STRICT = False
NO_END_DISCARD = True
# Rather than discard an unknown chop with nothing else to do, gamble on a
# clued card that is probably playable (the partner is likely expecting it).
ALT_PLAY_T = 0.8
# Save Principle robustness: save a critical chop card even when the partner
# looks busy, because an independent partner may still choose to discard.
SAVE_CRIT_ALWAYS = True
CRIT_TOKENS = 3.5
CRIT_TRUST_PLAYABLE = True


def init_agent():
    return {
        'cand': jnp.ones((HAND, NCARD), dtype=jnp.bool_),
        'focused': jnp.zeros((HAND,), dtype=jnp.bool_),
        'saved': jnp.zeros((HAND,), dtype=jnp.bool_),
        'touched': jnp.zeros((HAND,), dtype=jnp.bool_),
        'p_play': jnp.zeros((HAND,), dtype=jnp.bool_),
        'p_save': jnp.zeros((HAND,), dtype=jnp.bool_),
    }


def _last_true(mask):
    """Highest index whose flag is set (0 when nothing is set)."""
    return 4 - jnp.argmax(mask[::-1])


def _shift(arr, slot, fresh):
    """Remove `slot`, shift later entries down, append `fresh` at index 4."""
    a5 = jnp.asarray(_A5)
    idx = jnp.minimum(a5 + (a5 >= slot).astype(jnp.int32), 4)
    out = arr[idx]
    return out.at[4].set(fresh)


def _extend_heights(heights, will_play):
    """Heights after every card already promised to play has played."""
    a5 = jnp.asarray(_A5)
    he = heights
    for _ in range(5):
        idx = 5 * a5 + jnp.clip(he, 0, 4)
        he = he + (will_play[idx] & (he < 5)).astype(jnp.int32)
    return he


def agent_act(obs, reset, memory, rng):
    A5 = jnp.asarray(_A5)
    A10 = jnp.asarray(_A10)
    K_COLOR = jnp.asarray(_K_COLOR)
    K_RANK = jnp.asarray(_K_RANK)
    RANK_TOTAL = jnp.asarray(_RANK_TOTAL)
    EYE5 = jnp.asarray(_EYE5)
    JGT = jnp.asarray(_JGT)

    feats = obs.features
    legal = obs.legal_actions
    fresh = init_agent()
    mem = {k: jnp.where(reset, fresh[k], memory[k]) for k in fresh}

    # ---------------- observation decoding ----------------
    partner_hand = feats[0:125].reshape(HAND, NCARD)
    p_occ = partner_hand.sum(-1) > 0.5
    p_k = jnp.argmax(partner_hand, axis=-1)
    p_c = p_k // 5
    p_r = p_k % 5

    deck = feats[127:167].sum()
    fw = feats[167:192].reshape(NCOL, NRANK)
    heights = jnp.where(fw.sum(-1) > 0.5, jnp.argmax(fw, axis=-1) + 1, 0).astype(jnp.int32)
    tokens = feats[192:200].sum()
    lives = feats[200:203].sum()

    disc = feats[203:253].reshape(NCOL, 10)
    dcount = jnp.stack([disc[:, 0:3].sum(-1), disc[:, 3:5].sum(-1), disc[:, 5:7].sum(-1),
                        disc[:, 7:9].sum(-1), disc[:, 9:10].sum(-1)], axis=-1)

    know = feats[308:658].reshape(2, HAND, 35)
    own_v0 = know[0, :, 0:25]
    own_touch = know[0, :, 25:35].sum(-1) > 0.5
    own_occ = own_v0.sum(-1) > 1e-9
    p_touch = (know[1, :, 25:35].sum(-1) > 0.5) & p_occ

    la = feats[253:308]
    la_actor_self = la[0] > 0.5
    la_actor_partner = la[1] > 0.5
    la_play = la[2] > 0.5
    la_discard = la[3] > 0.5
    la_color = la[4] > 0.5
    la_rank = la[5] > 0.5
    la_recip_self = la[6] > 0.5
    la_clue_rank = jnp.argmax(la[13:18])
    la_matches = la[18:23] > 0.5
    la_removed = jnp.argmax(la[23:28])

    # ---------------- slot bookkeeping ----------------
    removal = la_play | la_discard
    own_removal = removal & la_actor_self
    p_removal = removal & la_actor_partner

    cand = jnp.where(own_removal,
                     _shift(mem['cand'], la_removed, jnp.ones((NCARD,), dtype=jnp.bool_)),
                     mem['cand'])
    focused = jnp.where(own_removal, _shift(mem['focused'], la_removed, False), mem['focused'])
    saved = jnp.where(own_removal, _shift(mem['saved'], la_removed, False), mem['saved'])
    prev_touch = jnp.where(own_removal, _shift(mem['touched'], la_removed, False), mem['touched'])
    p_play = jnp.where(p_removal, _shift(mem['p_play'], la_removed, False), mem['p_play'])
    p_save = jnp.where(p_removal, _shift(mem['p_save'], la_removed, False), mem['p_save'])

    # ---------------- public card accounting ----------------
    played_copy = (heights[K_COLOR] > K_RANK).astype(jnp.float32)
    disc_flat = dcount.reshape(NCARD)
    remaining_pub = RANK_TOTAL[K_RANK] - disc_flat - played_copy
    pcount = partner_hand.sum(0)
    remaining_vis = remaining_pub - pcount

    playable_k = (K_RANK == heights[K_COLOR])
    useless_k = (K_RANK < heights[K_COLOR]) | (remaining_pub <= 0.5)
    critical_k = (remaining_pub <= 1.5) & (K_RANK >= heights[K_COLOR])

    # Identities already clued in the partner's hand: Good Touch says a clued
    # card of mine is not a duplicate of one of them.
    p_touch_ident = ((jax.nn.one_hot(p_k, NCARD) * (p_touch & p_occ)[:, None]).sum(0)) > 0.5

    base_possible = (own_v0 > 1e-9) & (remaining_vis[None, :] > 0.5)

    def _refine(candidate, extra_bad):
        gt = (~useless_k) & (~p_touch_ident) & (~extra_bad)
        a = base_possible & candidate & gt[None, :]
        b = base_possible & candidate
        sel = jnp.where(a.any(-1, keepdims=True), a,
                        jnp.where(b.any(-1, keepdims=True), b, base_possible))
        pos = jnp.where(own_touch[:, None], sel, base_possible)
        return jnp.where(own_occ[:, None], pos, False), b

    def beliefs(candidate):
        none = jnp.zeros((NCARD,), dtype=jnp.bool_)
        pos, _ = _refine(candidate, none)
        # Good Touch also forbids two clued copies inside my own hand: an
        # identity pinned down in one clued slot is excluded from the others.
        pinned = own_touch & (pos.sum(-1) == 1)
        own_ident = (pos & pinned[:, None]).any(0)
        if OWN_PIN:
            pos2, b = _refine(candidate, own_ident)
            pos = jnp.where(pinned[:, None], pos, pos2)
        else:
            _, b = _refine(candidate, jnp.zeros((NCARD,), dtype=jnp.bool_))
        sure_play = own_occ & own_touch & pos.any(-1) & (~(pos & (~playable_k)[None, :]).any(-1))
        sure_trash = own_occ & own_touch & b.any(-1) & (~(b & (~useless_k)[None, :]).any(-1))
        return pos, sure_play, sure_trash, b

    def weights(candidate):
        """Soft belief: literal counts, discounted by convention inferences.

        Convention and Good Touch inferences are treated as strong but not
        certain evidence, so a partner's Save Clue or Bad Touch cannot make me
        certain of a wrong identity.
        """
        counts = jnp.maximum(remaining_vis, 0.0)
        conv = jnp.where(own_touch[:, None] & (~candidate), CONV_DISCOUNT, 1.0)
        gt_ok = (~useless_k) & (~p_touch_ident)
        gtf = jnp.where(own_touch[:, None] & (~gt_ok)[None, :], GT_DISCOUNT, 1.0)
        w = base_possible * counts[None, :] * conv * gtf
        w = jnp.where(own_occ[:, None], w, 0.0)
        tot = jnp.maximum(w.sum(-1, keepdims=True), 1e-6)
        prob = w / tot
        return prob, w.sum(-1) > 1e-9

    pos0, sure_play0, _, _ = beliefs(cand)

    # Good Touch Principle: every clued card that is still useful is "accounted
    # for" and will eventually play, so it may serve as a delayed-play connector.
    p_useful0 = p_r >= heights[p_c]
    if ACCT_MODE == 'touched':
        _acct_src = p_occ & p_touch & p_useful0
    else:
        _acct_src = p_occ & p_play & p_useful0
    theirs_acct = ((jax.nn.one_hot(p_k, NCARD) * _acct_src[:, None]).sum(0)) > 0.5

    def mine_certain(pos):
        pinned = own_occ & own_touch & (pos.sum(-1) == 1)
        return (pos & pinned[:, None]).any(0)

    def mine_maybe(pos, skip):
        pending = own_occ & own_touch & (~skip)
        return (pos & pending[:, None]).any(0)

    no_slot = jnp.zeros((HAND,), dtype=jnp.bool_)

    # ---------------- interpret a clue the partner gave me ----------------
    got_clue = la_actor_partner & (la_color | la_rank) & la_recip_self
    own_chop_pre_mask = own_occ & (~prev_touch)
    has_chop_pre = own_chop_pre_mask.any()
    chop_pre = jnp.argmax(own_chop_pre_mask)

    new_mask = la_matches & (~prev_touch)
    any_new = new_mask.any()
    chop_hit = has_chop_pre & new_mask[chop_pre]
    focus = jnp.where(any_new,
                      jnp.where(chop_hit, chop_pre, _last_true(new_mask)),
                      _last_true(la_matches))

    # A Play Clue promises the card plays without further clues: it is playable
    # now, or after cards already promised in either hand.  The upper bound is
    # generous on purpose so that a partner's delayed clue is never misread as a
    # higher card than intended.
    skip_focus = no_slot.at[focus].set(True)
    if RECV_MODE == 'generous':
        h_top = _extend_heights(heights, theirs_acct | mine_maybe(pos0, skip_focus))
    else:
        h_top = _extend_heights(heights, theirs_acct | mine_certain(pos0))
    if RECV_MODE == 'exact':
        play_set = K_RANK == h_top[K_COLOR]
    else:
        play_set = (K_RANK >= heights[K_COLOR]) & (K_RANK <= h_top[K_COLOR])
    save_five = la_rank & (la_clue_rank == 4) & (K_RANK == 4)
    save_two = la_rank & (la_clue_rank == 1) & (K_RANK == 1)
    save_set = (critical_k | save_five | save_two) & (K_RANK >= heights[K_COLOR])
    allowed = play_set | (chop_hit & save_set)

    focus_row = cand[focus]
    narrowed = focus_row & allowed
    keep = got_clue & narrowed.any()
    cand = cand.at[focus].set(jnp.where(keep, narrowed, focus_row))
    focused = focused.at[focus].set(focused[focus] | got_clue)
    saved = saved.at[focus].set(saved[focus] | (got_clue & chop_hit & (narrowed & save_set).any()))

    pos, sure_play, sure_trash, cand_only = beliefs(cand)
    certain_k = mine_certain(pos)
    h_ext = _extend_heights(heights, theirs_acct | certain_k)

    # ---------------- partner analysis ----------------
    p_chop_mask = p_occ & (~p_touch)
    has_p_chop = p_chop_mask.any()
    p_chop = jnp.argmax(p_chop_mask)

    p_useful = p_r >= heights[p_c]
    p_playable = p_r == heights[p_c]
    p_rem = RANK_TOTAL[p_r] - dcount[p_c, p_r] - (heights[p_c] > p_r).astype(jnp.float32)
    p_crit = p_rem <= 1.5
    same_ident = (p_k[:, None] == p_k[None, :]) & p_occ[:, None] & p_occ[None, :] & (~EYE5)
    dup_prev = (same_ident & p_touch[None, :]).any(-1)                       # [5]
    promised_k = certain_k
    p_gotten = p_touch | dup_prev | promised_k[p_k]
    p_needs_save = p_occ & p_useful & (p_crit | (p_r == 4) | (p_r == 1))
    if SAVE_SKIP_GOTTEN:
        p_needs_save = p_needs_save & (~p_gotten)

    if LOADED_STRICT:
        partner_loaded = ((p_occ & p_touch & p_playable).any()
                          | (p_occ & p_touch & (~p_useful)).any())
    else:
        partner_loaded = ((p_occ & p_play).any()
                          | (p_occ & p_touch & p_playable).any()
                          | (p_occ & p_touch & (~p_useful)).any())
    chop_at_risk = has_p_chop & (p_needs_save[p_chop] | p_playable[p_chop])
    chop_critical = has_p_chop & p_occ[p_chop] & p_crit[p_chop] & p_useful[p_chop] \
        & (~p_gotten[p_chop])
    urgent = (chop_at_risk & (~partner_loaded)) & (tokens > 0.5)
    if SAVE_CRIT_ALWAYS:
        eager = chop_critical & (tokens > CRIT_TOKENS)
        if CRIT_TRUST_PLAYABLE:
            eager = eager & (~(p_occ & p_touch & p_playable).any())
        urgent = urgent | eager

    # ---------------- clue enumeration ----------------
    col_match = (p_c[None, :] == A5[:, None]) & p_occ[None, :]
    rnk_match = (p_r[None, :] == A5[:, None]) & p_occ[None, :]
    match = jnp.concatenate([col_match, rnk_match], axis=0)          # [10, 5]
    is_color = A10 < 5
    clue_rank_id = A10 - 5

    c_new = match & (~p_touch)[None, :]
    c_has_new = c_new.any(-1)
    c_chop_hit = has_p_chop & c_new[:, p_chop]
    c_focus = jnp.where(c_has_new,
                        jnp.where(c_chop_hit, p_chop, 4 - jnp.argmax(c_new[:, ::-1], axis=-1)),
                        4 - jnp.argmax(match[:, ::-1], axis=-1))
    fc = p_c[c_focus]
    fr = p_r[c_focus]

    # A play clue is only legal on the exact next card of its suit once every
    # already promised card is counted; cluing a copy of something already
    # promised would be a Bad Touch and would be read as the later card.
    f_play_now = fr == heights[fc]
    if GUARD_PLAY_NOW:
        f_play_now = f_play_now & (h_ext[fc] == heights[fc])
    f_delayed = (fr == h_ext[fc]) & (fr > heights[fc]) & GIVE_DELAYED
    f_crit = p_crit[c_focus]
    save_ok = (f_crit
               | ((~is_color) & (clue_rank_id == 4) & (fr == 4))
               | ((~is_color) & (clue_rank_id == 1) & (fr == 1)))
    f_save = c_chop_hit & (c_focus == p_chop) & has_p_chop & p_useful[c_focus] & save_ok

    dup_within = ((same_ident & JGT)[None, :, :] & c_new[:, None, :]).any(-1)  # [10, 5]
    slot_trash = p_occ & (~p_useful)
    dup_promised = promised_k[p_k] & p_occ
    bad = c_new & (slot_trash[None, :] | dup_prev[None, :] | dup_promised[None, :] | dup_within)
    bad_count = bad.sum(-1)
    good_new = c_new & (~bad)
    good_count = good_new.sum(-1)
    far = jnp.maximum(p_r - heights[p_c], 0).astype(jnp.float32)
    not_focus = jnp.arange(5)[None, :] != c_focus[:, None]
    if FAR_INCLUDES_FOCUS:
        far_cost = (good_new * far[None, :]).sum(-1)
    else:
        far_cost = (good_new & not_focus) * far[None, :]
        far_cost = far_cost.sum(-1)
    extra_playable = (good_new & p_playable[None, :]).sum(-1) - f_play_now.astype(jnp.float32)

    focus_bad = jnp.take_along_axis(bad, c_focus[:, None], axis=1)[:, 0]
    c_legal = legal[10:20]
    c_valid = (c_legal & c_has_new & (f_play_now | f_delayed | f_save)
               & ((~focus_bad) | f_save))

    score = (jnp.where(c_valid, 0.0, -1.0e6)
             + W_PLAY_NOW * f_play_now
             + W_PLAY_DELAYED * f_delayed
             + W_SAVE * f_save
             + W_SAVE_CRIT * (f_save & (f_crit | (fr == 4)))
             + W_GOOD_TOUCH * good_count
             + W_EXTRA_PLAYABLE * jnp.maximum(extra_playable, 0.0)
             - W_BAD_TOUCH * bad_count
             + W_COLOR * is_color
             - W_RANK_PENALTY * fr
             - W_FAR * far_cost)
    score = score + jnp.where(urgent, W_URGENT * (c_chop_hit & c_valid), 0.0)
    # Leaving the partner with no unclued card means they can no longer discard
    # safely and I can no longer save them; avoid it unless the clue is a save.
    kills_chop = c_chop_hit & (~(p_occ & (~p_touch) & (~c_new)).any(-1))
    score = score - W_KILL_CHOP * (kills_chop & (~f_save))

    best_clue = jnp.argmax(score)
    clue_ok = (score[best_clue] > -1.0e5) & (tokens > 0.5)

    # Tempo Clue: no new card, but it identifies an already clued playable card.
    # Minimum Clue Value Principle only permits this in forced situations, so it
    # is used solely when the alternative is discarding a useful clued card.
    tempo_ready = p_occ[c_focus] & p_playable[c_focus] & p_touch[c_focus]
    tempo_score = jnp.where(c_legal & (~c_has_new) & tempo_ready,
                            W_TEMPO + W_COLOR * is_color - W_RANK_PENALTY * fr, -1.0e6)
    best_tempo = jnp.argmax(tempo_score)
    tempo_ok = (tempo_score[best_tempo] > -1.0e5) & (tokens > 0.5) & ALLOW_TEMPO

    # Forced clue (8 tokens, nothing playable, no convention-valid clue).
    forced_score = jnp.where(c_legal, W_GOOD_TOUCH * good_count - W_BAD_TOUCH * bad_count
                             + W_PLAY_NOW * f_play_now, -1.0e6)
    forced_clue = jnp.argmax(forced_score)
    forced_ok = forced_score[forced_clue] > -1.0e5

    # ---------------- own play / discard choice ----------------
    prob, has_belief = weights(cand)
    prob_play = (prob * playable_k[None, :]).sum(-1)
    prob_trash = (prob * useless_k[None, :]).sum(-1)
    prob_crit = (prob * critical_k[None, :]).sum(-1)
    thresh = jnp.where(lives > 2.5, PLAY_T3, jnp.where(lives > 1.5, PLAY_T2, PLAY_T1))
    end_phase = deck <= END_DECK
    thresh = jnp.where(end_phase & (lives > 1.5), jnp.minimum(thresh, PLAY_TEND), thresh)
    playable_slot = own_occ & own_touch & has_belief & (prob_play >= thresh)

    alt_slot = own_occ & own_touch & has_belief & (prob_play >= ALT_PLAY_T)
    rank_est = jnp.min(jnp.where(pos, K_RANK[None, :], 99), axis=-1)
    alt_score = jnp.where(alt_slot, 100.0 * prob_play - A5, -1.0e6)
    alt_best = jnp.argmax(alt_score)
    play_score = jnp.where(playable_slot, 100.0 - 10.0 * rank_est - A5, -1.0e6)
    play_slot = jnp.argmax(play_score)
    have_play = playable_slot.any() & legal[5 + play_slot]

    own_chop_mask = own_occ & (~own_touch)
    has_own_chop = own_chop_mask.any()
    known_trash = own_occ & own_touch & (prob_trash >= TRASH_T)
    trash_slot = jnp.argmax(known_trash)
    # Expected cost of losing a card, used only when the hand is fully clued and
    # there is no ordinary chop to discard.
    card_value = jnp.where(useless_k, 0.0,
                           jnp.where(remaining_pub <= 1.5, 6.0,
                                     jnp.where(remaining_pub <= 2.5, 1.5, 1.0)))
    exp_value = (prob * card_value[None, :]).sum(-1)
    cheap = jnp.where(own_occ, exp_value, 1.0e6)
    if DISCARD_MODE == 'ev':
        fallback = jnp.argmin(cheap)
    else:
        safe_clued = own_occ & (prob_crit <= 0.05)
        fallback = jnp.where(safe_clued.any(), jnp.argmax(safe_clued), jnp.argmax(own_occ))
    discard_slot = jnp.where(known_trash.any(), trash_slot,
                             jnp.where(has_own_chop, jnp.argmax(own_chop_mask), fallback))
    can_discard = (tokens < 7.5) & legal[discard_slot]
    costly_discard = (((~has_own_chop) | (end_phase & NO_END_DISCARD)
                       | (tokens >= TEMPO_TOKENS)) & (~known_trash.any()))

    action = jnp.where(
        urgent & clue_ok, 10 + best_clue,
        jnp.where(have_play, 5 + play_slot,
                  jnp.where(clue_ok, 10 + best_clue,
                            jnp.where(costly_discard & tempo_ok, 10 + best_tempo,
                                      jnp.where(alt_slot.any() & (lives > 1.5) & legal[5 + alt_best],
                                                5 + alt_best,
                                      jnp.where(can_discard, discard_slot,
                                                jnp.where(forced_ok, 10 + forced_clue,
                                                          jnp.argmax(legal))))))))
    action = jnp.where(legal[action], action, jnp.argmax(legal))
    active = ~legal[20]
    action = jnp.where(active, action, 20).astype(jnp.int32)

    # ---------------- remember what my own clue meant ----------------
    gave = active & (action >= 10) & (action < 20)
    ci = jnp.clip(action - 10, 0, 9)
    gf = c_focus[ci]
    set_play = gave & (f_play_now[ci] | f_delayed[ci]) & (~f_save[ci])
    set_save = gave & f_save[ci]
    p_play = p_play.at[gf].set(p_play[gf] | set_play)
    p_save = p_save.at[gf].set(p_save[gf] | set_save)

    next_mem = {
        'cand': cand,
        'focused': focused,
        'saved': saved,
        'touched': own_touch,
        'p_play': p_play,
        'p_save': p_save,
    }
    la_fail = la_play & (la[53] < 0.5)
    _alive = (remaining_pub > 0.5) | (K_RANK < heights[K_COLOR])
    _reach = _extend_heights(heights * 0, _alive)
    trace = {
        'action': action,
        'have_play': have_play,
        'clue_ok': clue_ok,
        'urgent': urgent,
        'score': heights.sum().astype(jnp.int32),
        'tokens': tokens,
        'lives': lives,
        'deck': deck,
        'fail_k': jnp.where(la_fail, jnp.argmax(la[28:53]), -1).astype(jnp.int32),
        'fail_self': (la_fail & la_actor_self).astype(jnp.int32),
        'possize': jnp.where(active & have_play, pos[play_slot].sum(), -1).astype(jnp.int32),
        'pplay': jnp.where(active & have_play, prob_play[play_slot], -1.0),
        'was_focus': jnp.where(active & have_play, focused[play_slot], False),
        'bel_k': jnp.where(active & have_play, jnp.argmax(pos[play_slot]), -1).astype(jnp.int32),
        'n_sure': sure_play.sum().astype(jnp.int32),
        'n_touch': own_touch.sum().astype(jnp.int32),
        'maxscore': _reach.sum().astype(jnp.int32),
        'give_focus': jnp.where(gave, c_focus[ci], -1).astype(jnp.int32),
        'give_k': jnp.where(gave, p_k[c_focus[ci]], -1).astype(jnp.int32),
        'give_kind': jnp.where(gave, jnp.where(f_save[ci], 2,
                                               jnp.where(f_play_now[ci], 0, 1)), -1).astype(jnp.int32),
        'recv_focus': jnp.where(got_clue, focus, -1).astype(jnp.int32),
        'recv_size': jnp.where(got_clue, narrowed.sum(), -1).astype(jnp.int32),
        'recv_chop': jnp.where(got_clue, chop_hit, False),
    }
    return action, next_mem, trace
