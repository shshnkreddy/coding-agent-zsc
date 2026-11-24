"""Two-player Hanabi policy following H-group Level 1 conventions.

Coordination protocol (shared, documented in context/conventions/h_group_level_1):
  * Chop = lowest occupied native index among unclued cards.
  * Clue focus = leftmost re-touched card when nothing is new, else chop when the
    chop was touched, else the leftmost (highest native index) newly touched card.
  * A clue focusing a non-chop card is a Play Clue; a chop clue may be a Play Clue
    or one of the permitted Save Clues (5 Save, 2 Save, Critical Save).
  * Good Touch Principle narrows every newly touched card to cards that can still
    be played, excluding identities already touched in the partner's hand.
  * Minimum Clue Value Principle: only clues that newly touch a card are given.
Beliefs are literal (native V0) intersected with these convention masks; a card is
played only when the resulting distribution makes it playable with high confidence.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

_F32 = jnp.float32
_BASE_COUNTS = (3.0, 2.0, 2.0, 2.0, 1.0)
_COLOR_CLUE = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_TWO_CLUE_SEL = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
_UPPER_ROWS = (
    (0.0, 1.0, 1.0, 1.0, 1.0),
    (0.0, 0.0, 1.0, 1.0, 1.0),
    (0.0, 0.0, 0.0, 1.0, 1.0),
    (0.0, 0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
)
_EYE_ROWS = (
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0),
)


def _idx5():
    return jnp.arange(5, dtype=jnp.int32)


def _rankf():
    return jnp.arange(5, dtype=_F32)


def _ranks():
    return jnp.broadcast_to(jnp.arange(5, dtype=_F32)[None, :], (5, 5))


def init_agent():
    return {
        "conv": jnp.ones((5, 25), dtype=_F32),
        "play": jnp.zeros((5,), dtype=_F32),
        "save": jnp.zeros((5,), dtype=_F32),
        "touch": jnp.zeros((5,), dtype=_F32),
        "age": jnp.zeros((5,), dtype=_F32),
        "ptouch": jnp.zeros((5,), dtype=_F32),
        "pplay": jnp.zeros((5,), dtype=_F32),
        "pconv": jnp.ones((5, 25), dtype=_F32),
        "t": jnp.zeros((), dtype=_F32),
    }


def _pick(*options):
    """First option (in order) whose mask is non-empty; last is the fallback."""
    out = options[-1]
    for opt in reversed(options[:-1]):
        out = jnp.where(jnp.sum(opt) > 1e-6, opt, out)
    return out


def _pick_rows(*options):
    out = options[-1]
    for opt in reversed(options[:-1]):
        nonempty = jnp.sum(opt, axis=1, keepdims=True) > 1e-6
        out = jnp.where(nonempty, opt, out)
    return out


def _board(f):
    ranks = _ranks()
    base = jnp.asarray(_BASE_COUNTS, dtype=_F32)
    fireworks = f[167:192].reshape(5, 5)
    height = jnp.sum(fireworks * (_rankf() + 1.0)[None, :], axis=1)
    disc = f[203:253].reshape(5, 10)
    discarded = jnp.stack([
        jnp.sum(disc[:, 0:3], axis=1),
        jnp.sum(disc[:, 3:5], axis=1),
        jnp.sum(disc[:, 5:7], axis=1),
        jnp.sum(disc[:, 7:9], axis=1),
        disc[:, 9],
    ], axis=1)
    hcol = height[:, None]
    played = (ranks < hcol).astype(_F32)
    remaining = base[None, :] - discarded - played
    avail = (remaining > 0.5).astype(_F32)
    gate = jnp.where(ranks >= hcol, avail, 1.0)
    cprod = jnp.cumprod(gate, axis=1)
    unblocked = jnp.concatenate([jnp.ones((5, 1), _F32), cprod[:, :4]], axis=1)
    useful = ((ranks >= hcol) & (remaining > 0.5) & (unblocked > 0.5)).astype(_F32)
    playable = ((ranks == hcol) & (remaining > 0.5)).astype(_F32)
    critical = useful * (remaining < 1.5).astype(_F32)
    twos = useful * (ranks == 1.0).astype(_F32)
    return height, useful, playable, critical, twos


_SOFT_TRASH = 0.5


def _beliefs(v0, conv, touched, useful_f):
    """Literal V0 narrowed by convention notes and, for clued cards, Good Touch.

    A clued card was promised to be playable when it was clued, so identities that
    have since become unplayable are unlikely but not impossible: the card may be
    the copy that went dead in the meantime.  They are down-weighted rather than
    removed, which keeps "my clued card must be the useful one" reasoning without
    turning it into false certainty.
    """
    soft = useful_f + _SOFT_TRASH * (1.0 - useful_f)
    base = v0 * conv
    w = jnp.where((touched > 0.5)[:, None], base * soft[None, :], base)
    w = jnp.where(jnp.sum(w, axis=1, keepdims=True) > 1e-9, w, v0)
    return w / jnp.clip(jnp.sum(w, axis=1, keepdims=True), 1e-9, None)


def _lowest(mask):
    """Lowest index where mask is set, 99 when empty."""
    return jnp.min(jnp.where(mask > 0.5, _idx5(), 99))


def _highest(mask):
    return jnp.max(jnp.where(mask > 0.5, _idx5(), -1))


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions
    legalf = legal.astype(_F32)
    idx5 = _idx5()
    idx5f = idx5.astype(_F32)
    ranks = _ranks()
    is_color_clue = jnp.asarray(_COLOR_CLUE, dtype=_F32)
    two_clue_sel = jnp.asarray(_TWO_CLUE_SEL, dtype=_F32)
    upper = jnp.asarray(_UPPER_ROWS, dtype=_F32)
    eye5 = jnp.asarray(_EYE_ROWS, dtype=_F32)

    height, useful, playable, critical, twos = _board(f)
    useful_f = useful.reshape(25)
    playable_f = playable.reshape(25)
    crit_f = critical.reshape(25)
    twos_f = twos.reshape(25)
    trash_f = 1.0 - useful_f

    knowledge = f[308:658].reshape(2, 5, 35)
    v0 = knowledge[0, :, 0:25]
    occ = (jnp.sum(v0, axis=1) > 1e-6).astype(_F32)
    touch_now = (jnp.sum(knowledge[0, :, 25:35], axis=1) > 1e-6).astype(_F32) * occ
    pcards = f[0:125].reshape(5, 25)
    pocc = (jnp.sum(pcards, axis=1) > 0.5).astype(_F32)
    ptouch = (jnp.sum(knowledge[1, :, 25:35], axis=1) > 1e-6).astype(_F32) * pocc

    tokens = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    lit = (v0 > 1e-9).astype(_F32)

    fresh = init_agent()
    not_reset = jnp.logical_not(reset)
    conv = jnp.where(reset, fresh["conv"], memory["conv"])
    play = jnp.where(reset, fresh["play"], memory["play"])
    save = jnp.where(reset, fresh["save"], memory["save"])
    prev_touch = jnp.where(reset, fresh["touch"], memory["touch"])
    age = jnp.where(reset, fresh["age"], memory["age"])
    prev_ptouch = jnp.where(reset, fresh["ptouch"], memory["ptouch"])
    pplay = jnp.where(reset, fresh["pplay"], memory["pplay"])
    pconv = jnp.where(reset, fresh["pconv"], memory["pconv"])
    step = jnp.where(reset, 0.0, memory["t"]) + 1.0

    # ---- public last action -------------------------------------------------
    la = f[253:308]
    la_valid = (jnp.sum(la[0:2]) > 0.5) & not_reset
    actor_self = la[0] > 0.5
    was_play = la[2] > 0.5
    was_disc = la[3] > 0.5
    was_chint = la[4] > 0.5
    was_rhint = la[5] > 0.5
    hrank = la[13:18]
    hmatch = la[18:23]
    removed = jnp.argmax(la[23:28])

    # Own card left the hand: shift every private note toward index zero.
    did_remove = la_valid & actor_self & (was_play | was_disc)
    src = idx5 + (idx5 >= removed).astype(jnp.int32)
    keep = src <= 4
    gsrc = jnp.clip(src, 0, 4)
    shifted_conv = jnp.where(keep[:, None], jnp.take(conv, gsrc, axis=0), 1.0)
    shifted_play = jnp.where(keep, jnp.take(play, gsrc), 0.0)
    shifted_save = jnp.where(keep, jnp.take(save, gsrc), 0.0)
    shifted_age = jnp.where(keep, jnp.take(age, gsrc), 0.0)
    conv = jnp.where(did_remove, shifted_conv, conv)
    play = jnp.where(did_remove, shifted_play, play)
    save = jnp.where(did_remove, shifted_save, save)
    age = jnp.where(did_remove, shifted_age, age)

    # Good Touch Principle also forbids touching two copies of the same card, so
    # once a clued card is revealed by a play or discard, my other clued cards
    # cannot be that identity.
    removed_ident = la[28:53]
    removed_touched = jnp.where(actor_self, jnp.take(prev_touch, removed),
                                jnp.take(prev_ptouch, removed))
    do_exclude = la_valid & (was_play | was_disc) & (removed_touched > 0.5)
    excluded = conv * (1.0 - 0.8 * removed_ident)[None, :]
    excluded = jnp.where(jnp.sum(excluded * (v0 > 1e-9), axis=1, keepdims=True) > 1e-9,
                         excluded, conv)
    conv = jnp.where(do_exclude & (touch_now > 0.5)[:, None], excluded, conv)

    # ---- what the partner has been told to play -----------------------------
    card_playable = jnp.sum(pcards * playable_f[None, :], axis=1)
    partner_removed = la_valid & jnp.logical_not(actor_self) & (was_play | was_disc)
    psrc = idx5 + (idx5 >= removed).astype(jnp.int32)
    pkeep = psrc <= 4
    pplay = jnp.where(partner_removed,
                      jnp.where(pkeep, jnp.take(pplay, jnp.clip(psrc, 0, 4)), 0.0), pplay)
    pconv = jnp.where(partner_removed,
                      jnp.where(pkeep[:, None], jnp.take(pconv, jnp.clip(psrc, 0, 4), axis=0), 1.0),
                      pconv)
    i_hinted = la_valid & actor_self & (was_chint | was_rhint)
    pnew = hmatch * pocc * (1.0 - prev_ptouch)
    p_has_new = jnp.sum(pnew) > 0.5
    pchop_raw0 = _lowest(pocc * (1.0 - prev_ptouch))
    pchop0 = jnp.clip(pchop_raw0, 0, 4)
    pchop_hit0 = (hmatch[pchop0] > 0.5) & (pchop_raw0 < 5)
    pfocus = jnp.where(p_has_new,
                       jnp.where(pchop_hit0, pchop0, _highest(pnew)),
                       _highest(hmatch * pocc))
    pfocus = jnp.clip(pfocus, 0, 4)
    pfocus_mask = jax.nn.one_hot(pfocus, 5, dtype=_F32)
    # A discard proves the partner had nothing they knew how to play.
    pplay = jnp.where(la_valid & jnp.logical_not(actor_self) & was_disc,
                      jnp.zeros((5,), dtype=_F32), pplay)
    pplay = pplay * pocc

    # ---- promised cards and the delayed-play set ----------------------------
    pre_p = _beliefs(v0, conv, prev_touch, useful_f)
    own_known = ((jnp.max(pre_p, axis=1) > 0.99) & (play > 0.5) & (occ > 0.5)).astype(_F32)
    own_ident = jax.nn.one_hot(jnp.argmax(pre_p, axis=1), 25, dtype=_F32)
    # Good Touch tells me a clue giver avoids duplicating a card they can see is
    # already clued: my own other clued cards, and their own clued cards only when
    # their literal knowledge pins the identity down.  Their hand is visible to me
    # but not to them, so their unknown clued cards must not be eliminated.
    pv0 = knowledge[1, :, 0:25]
    p_known = ((jnp.sum((pv0 > 1e-9).astype(_F32), axis=1) < 1.5) & (ptouch > 0.5)).astype(_F32)
    gt_block = jnp.max(pcards * p_known[:, None], axis=0)
    own_certain = ((jnp.max(pre_p, axis=1) > 0.99) & (prev_touch > 0.5) & (occ > 0.5)).astype(_F32)
    gt_block = jnp.clip(gt_block + jnp.max(own_ident * own_certain[:, None], axis=0), 0.0, 1.0)
    promised = jnp.maximum(jnp.max(pcards * ptouch[:, None], axis=0),
                           jnp.max(own_ident * own_known[:, None], axis=0))

    prom = promised.reshape(5, 5)
    reach = height
    for _ in range(4):
        step_p = prom[idx5, jnp.clip(reach, 0.0, 4.0).astype(jnp.int32)]
        reach = reach + jnp.where(reach <= 4.5, step_p, 0.0)
    play_allowed = (((ranks >= height[:, None]) & (ranks <= reach[:, None])).astype(_F32)
                    * useful).reshape(25)

    # A 5 Save and a 2 Save both require a number clue: a color clue that touches
    # an unclued 5 (or 2) on chop is a Play Clue unless the card is critical for
    # another reason, so 5s never enter the save set of a color clue.
    rank5_f = (ranks == 4.0).astype(_F32).reshape(25)
    two_clue = was_rhint & (hrank[1] > 0.5)
    crit_save = jnp.where(was_chint, crit_f * (1.0 - rank5_f), crit_f)
    save_set = jnp.clip(crit_save + jnp.where(two_clue, twos_f, 0.0), 0.0, 1.0)

    # After my own clue, does the partner's literal knowledge let them conclude the
    # focus is playable?  Only then should I expect them to play instead of discard.
    p_allowed = jnp.clip(play_allowed + jnp.where(pchop_hit0, save_set, 0.0), 0.0, 1.0)
    pplay = jnp.where(i_hinted,
                      jnp.clip(pplay + pfocus_mask * jnp.take(card_playable, pfocus), 0.0, 1.0),
                      pplay)
    # Mirror of my own note keeping, for the notes my clue just gave the partner.
    plit = (pv0 > 1e-9).astype(_F32)
    pfocus_row = _pick(jnp.take(pconv, pfocus, axis=0) * p_allowed * jnp.take(plit, pfocus, axis=0),
                       p_allowed * jnp.take(plit, pfocus, axis=0),
                       useful_f * jnp.take(plit, pfocus, axis=0),
                       jnp.take(plit, pfocus, axis=0))
    pother = _pick_rows(pconv * useful_f[None, :] * plit, useful_f[None, :] * plit, plit)
    pconv_h = jnp.where(((pnew > 0.5) & (pfocus_mask < 0.5))[:, None], pother, pconv)
    pconv_h = jnp.where((pfocus_mask > 0.5)[:, None], pfocus_row[None, :], pconv_h)
    pconv = jnp.where(i_hinted, pconv_h, pconv)

    # ---- interpreting a clue the partner gave me ----------------------------
    is_hint_to_me = la_valid & jnp.logical_not(actor_self) & (was_chint | was_rhint)
    newly = hmatch * occ * (1.0 - prev_touch)
    has_new = jnp.sum(newly) > 0.5
    chop_raw = _lowest(occ * (1.0 - prev_touch))
    chop_ex = chop_raw < 5
    chop_i = jnp.clip(chop_raw, 0, 4)
    chop_hit = (hmatch[chop_i] > 0.5) & chop_ex
    focus = jnp.where(has_new,
                      jnp.where(chop_hit, chop_i, _highest(newly)),
                      _highest(hmatch * occ))
    focus = jnp.clip(focus, 0, 4)
    save_possible = has_new & chop_hit

    allowed = jnp.where(save_possible, jnp.clip(play_allowed + save_set, 0.0, 1.0), play_allowed)
    lit_focus = jnp.take(lit, focus, axis=0)
    conv_focus = jnp.take(conv, focus, axis=0)
    new_focus = _pick(
        conv_focus * allowed * (1.0 - gt_block) * lit_focus,
        allowed * (1.0 - gt_block) * lit_focus,
        allowed * lit_focus,
        useful_f * lit_focus,
        lit_focus,
    )
    other_rows = _pick_rows(
        conv * useful_f[None, :] * (1.0 - gt_block)[None, :] * lit,
        useful_f[None, :] * (1.0 - gt_block)[None, :] * lit,
        useful_f[None, :] * lit,
        lit,
    )
    fmask = jax.nn.one_hot(focus, 5, dtype=_F32)
    touched_other = (newly > 0.5) & (fmask < 0.5)
    conv_h = jnp.where(touched_other[:, None], other_rows, conv)
    conv_h = jnp.where((fmask > 0.5)[:, None], new_focus[None, :], conv_h)
    play_h = jnp.where(fmask > 0.5, jnp.where(save_possible, 0.0, 1.0), play)
    save_h = jnp.where(fmask > 0.5, save_possible.astype(_F32), save)
    age_h = jnp.where(fmask > 0.5, step, age)

    conv = jnp.where(is_hint_to_me, conv_h, conv)
    play = jnp.where(is_hint_to_me, play_h, play)
    save = jnp.where(is_hint_to_me, save_h, save)
    age = jnp.where(is_hint_to_me, age_h, age)

    # ---- own beliefs --------------------------------------------------------
    prob = _beliefs(v0, conv, touch_now, useful_f)
    p_play = jnp.sum(prob * playable_f[None, :], axis=1) * occ
    p_trash = jnp.sum(prob * trash_f[None, :], axis=1) * occ
    p_crit = jnp.sum(prob * crit_f[None, :], axis=1) * occ

    thresh = jnp.where(lives > 2.5, 0.85, jnp.where(lives > 1.5, 0.95, 0.999))
    # In the final round an unplayed card is worth nothing, so a spare life may be
    # spent on a gamble.
    deck = jnp.sum(f[127:167])
    thresh = jnp.where((deck < 0.5) & (lives > 1.5), jnp.minimum(thresh, 0.45), thresh)
    eligible = (occ > 0.5) & (((play > 0.5) & (p_play >= thresh)) | (p_play >= 0.999))
    exp_rank = jnp.sum(prob * jnp.tile(jnp.arange(5, dtype=_F32), 5)[None, :], axis=1)
    play_key = jnp.where(eligible, p_play * 1000.0 - exp_rank * 10.0 - age, -1e9)
    best_play = jnp.argmax(play_key)
    have_play = jnp.any(eligible)

    # ---- clue evaluation ----------------------------------------------------
    p3 = pcards.reshape(5, 5, 5)
    cmatch = jnp.concatenate([jnp.sum(p3, axis=2).T, jnp.sum(p3, axis=1).T], axis=0)
    cmatch = cmatch * pocc[None, :]
    cnew = cmatch * (1.0 - ptouch)[None, :]
    nnew = jnp.sum(cnew, axis=1)
    pchop_raw = _lowest(pocc * (1.0 - ptouch))
    pchop_ex = pchop_raw < 5
    pchop = jnp.clip(pchop_raw, 0, 4)
    chop_hit_c = (cmatch[:, pchop] > 0.5) & pchop_ex
    f_re = jnp.max(jnp.where(cmatch > 0.5, idx5[None, :], -1), axis=1)
    f_new = jnp.max(jnp.where(cnew > 0.5, idx5[None, :], -1), axis=1)
    focus_c = jnp.clip(jnp.where(nnew > 0.5, jnp.where(chop_hit_c, pchop, f_new), f_re), 0, 4)
    fcard = jnp.take(pcards, focus_c, axis=0)
    f_playable = jnp.sum(fcard * playable_f[None, :], axis=1)
    f_crit = jnp.sum(fcard * crit_f[None, :], axis=1)
    f_two = jnp.sum(fcard * twos_f[None, :], axis=1)

    same = (pcards @ pcards.T) * (1.0 - eye5)
    dup_touched = (jnp.sum(same * ptouch[None, :], axis=1) > 0.5).astype(_F32)
    dup_any = (jnp.sum(same, axis=1) > 0.5).astype(_F32)
    card_useful = jnp.sum(pcards * useful_f[None, :], axis=1)
    own_block = jnp.max(own_ident * own_known[:, None], axis=0)
    dup_mine = jnp.sum(pcards * own_block[None, :], axis=1)
    bad_base = jnp.clip((1.0 - card_useful) * pocc + dup_touched + dup_mine, 0.0, 1.0)
    within = ((cnew @ (same * upper).T) > 0.5).astype(_F32)
    bad_ind = jnp.clip(bad_base[None, :] + within, 0.0, 1.0)
    nbad = jnp.sum(cnew * bad_ind, axis=1)
    good_ind = cnew * (1.0 - bad_ind)
    ngood = jnp.sum(good_ind, axis=1)
    n_play_new = jnp.sum(good_ind * card_playable[None, :], axis=1)

    focus_bad = jnp.take(bad_base, focus_c)
    focus_dup = jnp.take(dup_any, focus_c)
    has_new_c = (nnew > 0.5).astype(_F32)
    clean = (nbad < 0.5).astype(_F32)
    f_is_five = jnp.sum(fcard * rank5_f[None, :], axis=1)
    # Mirror of the receiving rule: a colour clue never reads as a 5 Save.
    f_crit_save = f_crit * (1.0 - is_color_clue * f_is_five)
    play_ok = f_playable * has_new_c * (1.0 - focus_bad) * clean
    two_save = f_two * (1.0 - focus_dup) * two_clue_sel
    # Once the deck is empty nothing can be discarded away that still matters, so
    # the remaining turns belong to plays.
    endgame = (deck < 0.5).astype(_F32)
    save_ok = (chop_hit_c.astype(_F32) * has_new_c * (1.0 - focus_bad)
               * jnp.clip(f_crit_save + two_save, 0.0, 1.0)) * (1.0 - endgame)
    legal_clue = legalf[10:20]
    valid = legal_clue * jnp.clip(play_ok + save_ok, 0.0, 1.0)
    score = (10.0 * play_ok + 6.0 * save_ok + 2.0 * f_crit * save_ok
             + 1.5 * n_play_new + 1.0 * ngood - 7.0 * nbad + 0.3 * is_color_clue)

    # Model what the partner can conclude from the literal clue alone: a play clue
    # is only useful now if their own knowledge leaves no doubt that the focus is
    # playable, so prefer the clue that lets them act immediately.
    card_color = jnp.repeat(jnp.arange(5, dtype=_F32), 5)
    card_rank = jnp.tile(jnp.arange(5, dtype=_F32), 5)
    clue_lit = jnp.concatenate([
        (card_color[None, :] == jnp.arange(5, dtype=_F32)[:, None]).astype(_F32),
        (card_rank[None, :] == jnp.arange(5, dtype=_F32)[:, None]).astype(_F32)], axis=0)
    save_rows = jnp.concatenate([
        jnp.broadcast_to((crit_f * (1.0 - rank5_f))[None, :], (5, 25)),
        jnp.clip(crit_f[None, :] + two_clue_sel[5:, None] * twos_f[None, :], 0.0, 1.0)], axis=0)
    allowed_rows = jnp.clip(play_allowed[None, :]
                            + chop_hit_c.astype(_F32)[:, None] * save_rows, 0.0, 1.0)
    pconv_focus = jnp.take(pconv, focus_c, axis=0)
    recv_base = jnp.take(pv0, focus_c, axis=0) * clue_lit * allowed_rows
    recv_w = jnp.where(jnp.sum(recv_base * pconv_focus, axis=1, keepdims=True) > 1e-9,
                       recv_base * pconv_focus, recv_base)
    recv_mass = jnp.sum(recv_w, axis=1)
    recv_play = jnp.sum(recv_w * playable_f[None, :], axis=1)
    immediate = (recv_play >= 0.9 * jnp.maximum(recv_mass, 1e-9)).astype(_F32) * (recv_mass > 1e-9)
    # If they would confidently play a focus that is not playable, the clue costs a
    # life: never give such a clue, not even when forced to say something.
    danger = immediate * (1.0 - f_playable)
    score = score + 8.0 * immediate * play_ok - 30.0 * danger
    valid = valid * (1.0 - danger)
    # Filling in an already-clued playable card is worth something on top of the
    # new cards a clue gets.
    fill_in = jnp.sum(cmatch * (ptouch * card_playable)[None, :], axis=1)
    score = score + 2.0 * fill_in
    clue_key = jnp.where(valid > 0.5, score, -1e9)
    best_clue = jnp.argmax(clue_key)
    clue_ok = jnp.any(valid > 0.5)

    # Tempo clue: allowed only when there is nothing else worth doing, which is
    # the "forced" exception to Minimum Clue Value Principle.  The focus must be
    # playable right now and no card may be badly touched, so the partner reads a
    # plain Play Clue on the leftmost re-touched card.
    stall_valid = legal_clue * clean * f_playable * (1.0 - focus_bad)
    stall_score = (2.0 * fill_in + 1.0 * ngood - 0.5 * nnew + 0.3 * is_color_clue
                   - 30.0 * danger)
    stall_key = jnp.where(stall_valid > 0.5, stall_score, -1e9)
    best_stall = jnp.argmax(stall_key)
    stall_ok = jnp.any(stall_valid > 0.5) & (tokens > 0.5)

    # ---- urgent chop save ---------------------------------------------------
    chop_card = jnp.take(pcards, pchop, axis=0)
    chop_needs = pchop_ex & ((jnp.sum(chop_card * crit_f)
                              + jnp.sum(chop_card * twos_f) * (1.0 - dup_any[pchop])) > 0.5)
    # The partner only skips a discard when their own notes let them conclude that
    # a card is playable, so a save is urgent whenever that is not the case.
    p_belief = _beliefs(pv0, pconv, ptouch, useful_f)
    p_play_them = jnp.sum(p_belief * playable_f[None, :], axis=1) * pocc
    partner_has_play = jnp.any((p_play_them >= 0.85) & (pplay * ptouch * card_playable > 0.5))
    save_key = jnp.where((valid > 0.5) & (focus_c == pchop), score, -1e9)
    best_save = jnp.argmax(save_key)
    urgent = chop_needs & jnp.logical_not(partner_has_play) & (jnp.max(save_key) > -1e8)

    # ---- discard ------------------------------------------------------------
    own_chop_raw = _lowest(occ * (1.0 - touch_now))
    own_chop = jnp.clip(own_chop_raw, 0, 4)
    own_chop_ex = own_chop_raw < 5
    trash_elig = (occ > 0.5) & (p_trash >= 0.999)
    trash_key = jnp.where(trash_elig, 10.0 - idx5f * 0.01, -1e9)
    fallback_key = jnp.where(occ > 0.5,
                             p_trash * 10.0 - p_crit * 5.0 - p_play * 2.0 - age * 0.001,
                             -1e9)
    disc_slot = jnp.where(jnp.any(trash_elig), jnp.argmax(trash_key),
                          jnp.where(own_chop_ex, own_chop, jnp.argmax(fallback_key)))

    # Forced clue (8 tokens, nothing playable, no convention-valid clue). Best is
    # a tempo clue on a playable card; otherwise prefer a clue whose literal
    # information leaves no possible play interpretation, so the partner holds.
    color_playable = jnp.sum(playable, axis=1)
    rank_playable = jnp.sum(playable, axis=0)
    interp_possible = jnp.clip(jnp.concatenate([color_playable, rank_playable]), 0.0, 1.0)
    forced_score = (20.0 * f_playable * (1.0 - focus_bad) + 10.0 * (1.0 - interp_possible)
                    - 30.0 * danger - 2.0 * nbad - 0.5 * nnew + 0.2 * is_color_clue)
    forced_key = jnp.where(legal_clue > 0.5, forced_score, -1e9)
    has_legal_clue = jnp.any(legal_clue > 0.5)

    action = jnp.where(has_legal_clue, 10 + jnp.argmax(forced_key), jnp.argmax(legalf))
    mode = jnp.where(has_legal_clue, 4, 5)
    can_discard = legalf[disc_slot] > 0.5
    action = jnp.where(can_discard, disc_slot, action)
    mode = jnp.where(can_discard, 0, mode)
    action = jnp.where(stall_ok, 10 + best_stall, action)
    mode = jnp.where(stall_ok, 6, mode)
    action = jnp.where(clue_ok, 10 + best_clue, action)
    mode = jnp.where(clue_ok, 2, mode)
    action = jnp.where(have_play, 5 + best_play, action)
    mode = jnp.where(have_play, 1, mode)
    action = jnp.where(urgent, 10 + best_save, action)
    mode = jnp.where(urgent, 3, mode)
    action = jnp.where(legalf[action] > 0.5, action, jnp.argmax(legalf))
    action = jnp.where(legal[20], 20, action).astype(jnp.int32)

    next_memory = {
        "conv": conv,
        "play": play,
        "save": save,
        "touch": touch_now,
        "age": age,
        "ptouch": ptouch,
        "pplay": pplay,
        "pconv": pconv,
        "t": step,
    }
    trace = {
        "mode": mode.astype(jnp.int32),
        "p_play": jnp.max(p_play),
        "tokens": tokens,
        "lives": lives,
    }
    return action, next_memory, trace
