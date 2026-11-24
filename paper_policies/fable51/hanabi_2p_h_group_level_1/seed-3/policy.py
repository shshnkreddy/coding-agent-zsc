"""Two-player Hanabi policy following H-group Level 1 conventions.

Conventions implemented (see context/conventions/h_group_level_1):
- Chop = lowest occupied native index among unclued cards; finesse position unused
  (no two-player finesses at Level 1).
- Clue focus: chop if touched, else leftmost (highest index) new card, else leftmost
  re-touched card (tempo clue).
- Non-chop focus = play clue (direct, delayed through the giver's *known* clued
  connectors, or self-prompt through the recipient's own clued cards when no direct
  reading exists). Chop focus = play clue OR save clue (5 save with rank 5, 2 save
  with rank 2, critical save with colour/rank); recipient holds both readings.
- Non-focused touched cards are "useful later" (Good Touch Principle) and are played
  only once literal information plus trash elimination makes them certainly playable.
- Clue giving: Good Touch (no trash, no duplicates), Minimum Clue Value (at least one
  new card played or saved), Save Principle (5s, unique 2s, criticals, unique
  playables on chop), prefer play over save, prefer colour clues.
- Discard: known trash first, then chop.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_l1_rules', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NEG = -1e9
COLOR_LIST = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4)
RANK_LIST = (0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4)
MULT_LIST = (3, 2, 2, 2, 1, 3, 2, 2, 2, 1, 3, 2, 2, 2, 1, 3, 2, 2, 2, 1, 3, 2, 2, 2, 1)
DISC_LIST = ((1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0, 0.0),
             (0.0, 0.0, 0.0, 0.0, 1.0))


class K:
    """Constant arrays, constructed lazily inside traced functions."""

    def __init__(self):
        self.COLOR_OF = jnp.asarray(COLOR_LIST, jnp.int32)
        self.RANK_OF = jnp.asarray(RANK_LIST, jnp.int32)
        self.MULT = jnp.asarray(MULT_LIST, jnp.int32)
        self.SLOT = jnp.arange(5)
        self.DISC_M = jnp.asarray(DISC_LIST, jnp.float32)


def init_agent():
    return {
        'note': jnp.ones((2, 5, 25), jnp.bool_),     # convention play candidates
        'save': jnp.zeros((2, 5, 25), jnp.bool_),    # convention save candidates
        'status': jnp.zeros((2, 5), jnp.int32),      # 0 unclued, 1 clued, 2 play focus, 3 chop play/save
        'clued': jnp.zeros((2, 5), jnp.bool_),       # positive-hint flags at previous call
        'suspect': jnp.zeros((2, 5, 25), jnp.bool_),  # identities that became trash after the clue
        'forced': jnp.zeros((2, 5), jnp.bool_),      # slot was newly touched by a forced (8-token) clue
        'prior': jnp.zeros((2, 5), jnp.bool_),       # slot was already clued when a forced clue arrived
        'trash_prev': jnp.zeros((25,), jnp.bool_),
        'deck_prev': jnp.asarray(40, jnp.int32),
    }


def delayed_mask(accounted, height):
    """Identities playable now or after accounted lower cards of the same colour play."""
    k = K()
    acc = accounted.reshape(5, 5)
    r = jnp.arange(5)[None, :]
    ok = (r < height[:, None]) | acc
    prefix = jnp.cumprod(ok.astype(jnp.int32), axis=1) > 0
    prev_ok = jnp.concatenate([jnp.ones((5, 1), jnp.bool_), prefix[:, :4]], axis=1)
    out = (r >= height[:, None]) & prev_ok
    return out.reshape(25)


def shift_hand(note_h, save_h, status_h, suspect_h, rs):
    k = K()
    src = jnp.minimum(k.SLOT + (k.SLOT >= rs).astype(jnp.int32), 4)
    last = (k.SLOT == 4)
    note_n = jnp.where(last[:, None], True, note_h[src])
    save_n = jnp.where(last[:, None], False, save_h[src])
    status_n = jnp.where(last, 0, status_h[src])
    suspect_n = jnp.where(last[:, None], False, suspect_h[src])
    return note_n, save_n, status_n, suspect_n


def cand_now(note_h, save_h, status_h, suspect_h, lit_h, occ_h, rem_ok, useful, playable_now, height, other_acc):
    """Current candidate identities per slot for one hand.

    Unclued cards may be anything literal information allows (including trash).
    Clued cards are assumed useful (Good Touch) except for identities that became
    trash after the clue, which stay possible (they may be duplicates)."""
    k = K()
    clued_h = status_h >= 1
    gt = jnp.where(clued_h[:, None], useful[None, :] | suspect_h, True)
    base = lit_h & rem_ok[None, :] & occ_h[:, None] & gt
    conv = (status_h >= 2)[:, None]
    c0 = jnp.where(conv, (note_h | save_h) & base, base)
    c0 = jnp.where(c0.any(1, keepdims=True), c0, base)
    clued_h = status_h >= 1
    own_all = c0 & clued_h[:, None]
    excl = jnp.stack([(own_all & (k.SLOT != s)[:, None]).any(0) for s in range(5)])
    dyn = jax.vmap(delayed_mask, (0, None))(other_acc[None, :] | excl, height)
    play_part = note_h & base & dyn
    save_part = save_h & base
    susp_part = suspect_h & base
    cand = jnp.where(conv, play_part | save_part | susp_part, base)
    cand = jnp.where(cand.any(1, keepdims=True), cand, base)
    return cand


def interpret_clue(touched, clued_before, occ_h, is_color, value, lit_after,
                   giver_known, own_cands, note_h, save_h, status_h, ctx):
    k = K()
    height, playable_now, useful, critical = ctx
    new = touched & ~clued_before
    any_new = new.any()
    unclued_occ = occ_h & ~clued_before
    has_chop = unclued_occ.any()
    chop = jnp.clip(jnp.min(jnp.where(unclued_occ, k.SLOT, 99)), 0, 4)
    chop_touched = has_chop & touched[chop]
    leftmost_touched = jnp.clip(jnp.max(jnp.where(touched, k.SLOT, -1)), 0, 4)
    leftmost_new = jnp.clip(jnp.max(jnp.where(new, k.SLOT, -1)), 0, 4)
    focus = jnp.where(~any_new, leftmost_touched, jnp.where(chop_touched, chop, leftmost_new))
    is_chop_focus = any_new & chop_touched

    lit_f = lit_after[focus] & useful
    direct = lit_f & playable_now
    d_giver = lit_f & delayed_mask(giver_known, height)
    p1 = direct | d_giver
    own_other = (own_cands & clued_before[:, None] & (k.SLOT != focus)[:, None]).any(0)
    d_self = lit_f & delayed_mask(giver_known | own_other, height)
    play_f = jnp.where(p1.any(), p1, d_self)

    match = jnp.where(is_color, k.COLOR_OF == value, k.RANK_OF == value)
    is5 = (~is_color) & (value == 4)
    is2 = (~is_color) & (value == 1)
    save_set = jnp.where(is5, k.RANK_OF == 4,
                         jnp.where(is2, k.RANK_OF == 1, critical & match & ~(is_color & (k.RANK_OF == 4))))
    save_f = lit_after[focus] & useful & save_set & is_chop_focus
    has_play = play_f.any()
    has_save = save_f.any()
    status_f = jnp.where(has_play | has_save, jnp.where(has_save, 3, 2), 1)
    note_f = jnp.where(has_play, play_f, jnp.full((25,), True) & ~has_save)

    apply_focus = any_new | has_play
    fmask = (k.SLOT == focus)
    note_n = jnp.where((fmask & apply_focus)[:, None], note_f[None, :], note_h)
    save_n = jnp.where((fmask & apply_focus)[:, None], save_f[None, :], save_h)
    status_n = jnp.where(fmask & apply_focus, status_f, status_h)
    other_new = new & ~fmask
    note_n = jnp.where(other_new[:, None], True, note_n)
    save_n = jnp.where(other_new[:, None], False, save_n)
    status_n = jnp.where(other_new, 1, status_n)
    return note_n, save_n, status_n, focus, is_chop_focus, play_f, save_f, any_new


def agent_act(obs, reset, memory, rng):
    k = K()
    mem0 = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), mem0, memory)
    f = obs.features
    legal = obs.legal_actions

    # ---------------- decode ----------------
    ph = f[0:125].reshape(5, 25) > 0.5
    p_occ = ph.any(1)
    p_id = jnp.argmax(ph, 1)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)
    fw = f[167:192].reshape(5, 5)
    height = jnp.sum(fw * jnp.arange(1, 6, dtype=jnp.float32)[None, :], 1).astype(jnp.int32)
    tokens = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    disc = (f[203:253].reshape(5, 10) @ k.DISC_M).reshape(25).astype(jnp.int32)
    last = f[253:308]
    know = f[308:658].reshape(2, 5, 35)
    lit = know[:, :, :25] > 0
    my_occ = lit[0].any(1)
    occ = jnp.stack([my_occ, p_occ])
    hinted = know[:, :, 25:35].sum(2) > 0
    played = k.RANK_OF < height[k.COLOR_OF]
    playable_now = k.RANK_OF == height[k.COLOR_OF]
    remaining = k.MULT - disc - played.astype(jnp.int32)
    exhausted = (remaining <= 0) & ~played
    dead = (jnp.cumsum(exhausted.reshape(5, 5).astype(jnp.int32), axis=1) > 0).reshape(25)
    trash = played | dead
    useful = ~trash
    critical = useful & (remaining == 1)
    p_count = ph.sum(0).astype(jnp.int32)
    rem_me = remaining - p_count
    ctx = (height, playable_now, useful, critical)
    active = ~legal[20]

    # ---------------- last action ----------------
    has_last = (last[0:2].sum() > 0.5) & ~reset
    actor = jnp.argmax(last[0:2])
    atype = jnp.argmax(last[2:6])
    is_rem = has_last & (atype <= 1)
    is_hint = has_last & (atype >= 2)
    is_color_hint = atype == 2
    rec = jnp.argmax(last[6:8])
    value = jnp.where(is_color_hint, jnp.argmax(last[8:13]), jnp.argmax(last[13:18]))
    matched = last[18:23] > 0.5
    removed = jnp.argmax(last[23:28])

    note, save, status, suspect = mem['note'], mem['save'], mem['status'], mem['suspect']
    forced = mem['forced']
    prior = mem['prior']
    played_from_forced = jnp.stack([forced[h][removed] for h in range(2)])
    for h in range(2):
        sn, ss, st, su = shift_hand(note[h], save[h], status[h], suspect[h], removed)
        c = is_rem & (actor == h)
        note = note.at[h].set(jnp.where(c, sn, note[h]))
        save = save.at[h].set(jnp.where(c, ss, save[h]))
        status = status.at[h].set(jnp.where(c, st, status[h]))
        suspect = suspect.at[h].set(jnp.where(c, su, suspect[h]))
    # pre-action hint flags and forced flags, shifted along with a removal
    src = jnp.minimum(k.SLOT + (k.SLOT >= removed).astype(jnp.int32), 4)
    clued_before = mem['clued']
    for h in range(2):
        shifted = jnp.where(k.SLOT == 4, False, clued_before[h][src])
        clued_before = clued_before.at[h].set(jnp.where(is_rem & (actor == h), shifted, clued_before[h]))
        fshift = jnp.where(k.SLOT == 4, False, forced[h][src])
        forced = forced.at[h].set(jnp.where(is_rem & (actor == h), fshift, forced[h]))
        pshift = jnp.where(k.SLOT == 4, False, prior[h][src])
        prior = prior.at[h].set(jnp.where(is_rem & (actor == h), pshift, prior[h]))

    # identities that became trash since the previous call may be duplicates of
    # clued cards (the holder's own successful play is excluded; a delayed promise
    # whose connector just played is excluded too).
    newly_trash = trash & ~mem['trash_prev']
    is_play = has_last & (atype == 0)
    note_succ = jnp.concatenate([note[:, :, 1:], jnp.zeros((2, 5, 1), jnp.bool_)], axis=2)
    note_succ = note_succ & (k.RANK_OF < 4)[None, None, :]
    for h in range(2):
        # own play from a slot touched by a forced clue may duplicate another clued card
        own_forced_play = (actor == h) & is_play & played_from_forced[h]
        mark = newly_trash & ~((actor == h) & is_play)
        add = mark[None, :] & (status[h] >= 1)[:, None] & ~((status[h] >= 2)[:, None] & note_succ[h])
        # cards clued before a forced clue may be duplicated by that clue's cards
        add_forced = newly_trash[None, :] & own_forced_play & prior[h][:, None] & (status[h] >= 1)[:, None]
        suspect = suspect.at[h].set(suspect[h] | add | add_forced)
    # freshly clued cards start without suspects, except that a forced clue (giver
    # had 8 tokens) may touch trash on non-focus cards
    newly_clued = hinted & ~clued_before
    suspect = jnp.where(newly_clued[:, :, None], False, suspect)
    forced_clue = is_hint & (tokens == 7)

    # ---------------- candidate sets ----------------
    p_true_acc = (ph & hinted[1][:, None]).any(0)

    def all_cands(note, save, status, suspect):
        my_cand = cand_now(note[0], save[0], status[0], suspect[0], lit[0], occ[0], rem_me > 0,
                           useful, playable_now, height, p_true_acc)
        my_cand_pub = cand_now(note[0], save[0], status[0], suspect[0], lit[0], occ[0], remaining > 0,
                               useful, playable_now, height, p_true_acc)
        my_clued = status[0] >= 1
        my_clued_acc = (my_cand_pub & my_clued[:, None]).any(0)
        my_known_acc = (my_cand_pub & my_clued[:, None] & (my_cand_pub.sum(1) == 1)[:, None]).any(0)
        p_cand = cand_now(note[1], save[1], status[1], suspect[1], lit[1], occ[1], remaining > 0,
                          useful, playable_now, height, my_clued_acc)
        p_clued = status[1] >= 1
        p_known_acc = (p_cand & p_clued[:, None] & (p_cand.sum(1) == 1)[:, None]).any(0)
        return my_cand, my_clued, my_known_acc, p_cand, p_clued, p_known_acc

    my_cand, my_clued, my_known_acc, p_cand, p_clued, p_known_acc = all_cands(note, save, status, suspect)

    # ---------------- interpret a hint ----------------
    hint_to_me = is_hint & (rec == 0)
    hint_to_p = is_hint & (rec == 1)
    # connectors "present and accounted for" in the giver's hand: the partner's
    # clued cards that the partner either knows exactly or was play-clued on
    # (identities visible to me)
    # the giver must know a connector to promise a delayed play through it
    r0 = interpret_clue(matched, clued_before[0], occ[0], is_color_hint, value, lit[0],
                        p_known_acc, my_cand, note[0], save[0], status[0], ctx)
    r1 = interpret_clue(matched, clued_before[1], occ[1], is_color_hint, value, lit[1],
                        my_known_acc, p_cand, note[1], save[1], status[1], ctx)
    note = note.at[0].set(jnp.where(hint_to_me, r0[0], note[0]))
    save = save.at[0].set(jnp.where(hint_to_me, r0[1], save[0]))
    status = status.at[0].set(jnp.where(hint_to_me, r0[2], status[0]))
    note = note.at[1].set(jnp.where(hint_to_p, r1[0], note[1]))
    save = save.at[1].set(jnp.where(hint_to_p, r1[1], save[1]))
    status = status.at[1].set(jnp.where(hint_to_p, r1[2], status[1]))
    # keep status consistent with public hint flags
    status = jnp.where(hinted & (status == 0), 1, status)
    status = jnp.where(~occ, 0, status)
    # forced clue: newly touched non-focus cards may be trash
    focus_slot = jnp.stack([r0[3], r1[3]])
    forced_mask = newly_clued & forced_clue & (k.SLOT[None, :] != focus_slot[:, None])
    suspect = jnp.where(forced_mask[:, :, None], suspect | trash[None, None, :], suspect)
    # remember slots newly touched by a forced clue, and slots clued before it
    hint_rec = jnp.stack([hint_to_me, hint_to_p])
    fnew = forced_clue & hint_rec[:, None] & matched[None, :] & ~clued_before
    forced = jnp.where(fnew, True, forced)
    prior = jnp.where(forced_clue & hint_rec[:, None] & clued_before & fnew.any(1)[:, None], True, prior)
    forced = forced & occ
    prior = prior & occ

    # recompute candidates after interpretation
    my_cand, my_clued, my_known_acc, p_cand, p_clued, p_known_acc = all_cands(note, save, status, suspect)

    new_mem = {'note': note, 'save': save, 'status': status, 'clued': hinted,
               'suspect': suspect, 'forced': forced, 'prior': prior, 'trash_prev': trash, 'deck_prev': deck}

    # ---------------- my hand analysis ----------------
    my_size = my_cand.sum(1)
    my_nonempty = my_size > 0
    my_all_playable = my_nonempty & ~(my_cand & ~playable_now[None, :]).any(1)
    my_all_trash = my_occ & ~(my_cand & useful[None, :]).any(1)
    my_min_rank = jnp.min(jnp.where(my_cand, k.RANK_OF[None, :], 9), 1)
    my_any_crit = (my_cand & critical[None, :]).any(1)
    # probability of playable (weights = remaining copies not visible)
    w = jnp.maximum(rem_me, 0).astype(jnp.float32)[None, :] * my_cand
    p_play = (w * playable_now[None, :]).sum(1) / jnp.maximum(w.sum(1), 1e-6)
    my_unclued_occ = my_occ & ~my_clued
    my_chop = jnp.clip(jnp.min(jnp.where(my_unclued_occ, k.SLOT, 99)), 0, 4)
    my_has_chop = my_unclued_occ.any()

    # ---------------- partner analysis ----------------
    p_size = p_cand.sum(1)
    p_known_play = p_clued & (p_size > 0) & ~(p_cand & ~playable_now[None, :]).any(1) & playable_now[p_id]
    partner_busy = p_known_play.any()
    p_unclued_occ = p_occ & ~p_clued
    p_has_chop = p_unclued_occ.any()
    p_chop = jnp.clip(jnp.min(jnp.where(p_unclued_occ, k.SLOT, 99)), 0, 4)
    X = p_id[p_chop]
    danger = p_has_chop & useful[X] & (critical[X] | ((k.RANK_OF[X] == 1) & (p_count[X] == 1))
                                        | (playable_now[X] & (p_count[X] == 1)))
    p_clued_ids = (ph & p_clued[:, None]).any(0)
    # tempo clues are allowed when tokens are plentiful or my hand is locked
    tempo_ok = (tokens >= 2) | ~my_has_chop

    # ---------------- evaluate clues ----------------
    def eval_clue(ci):
        is_color = ci < 5
        val = ci % 5
        match = jnp.where(is_color, k.COLOR_OF == val, k.RANK_OF == val)
        touched = p_occ & match[p_id]
        lit_after = lit[1] & jnp.where(touched[:, None], match[None, :], ~match[None, :])
        # self-prompt connectors in the recipient's hand are checked against their
        # true identities (visible to the giver) restricted to the recipient's
        # post-clue candidates: the recipient will resolve the connector when it
        # plays and infer the focus card from it
        res = interpret_clue(touched, p_clued, p_occ, is_color, val, lit_after,
                             my_known_acc, p_cand & lit_after & ph, note[1], save[1], status[1], ctx)
        _, _, _, focus, is_chop_focus, play_f, save_f, any_new = res
        new = touched & ~p_clued
        ft = p_id[focus]
        cand_f = play_f | save_f
        plays_now = cand_f.any() & ~(cand_f & ~playable_now).any()
        true_play_now = playable_now[ft]
        unsafe = plays_now & ~true_play_now
        new_count = (ph & new[:, None]).sum(0)
        dup_id = p_clued_ids | my_known_acc | (new_count > 1)
        bad_new = new & (trash[p_id] | dup_id[p_id])
        n_bad = bad_new.sum()
        focus_bad = bad_new[focus]
        valid_play = any_new & play_f[ft] & useful[ft] & (n_bad == 0)
        imm = valid_play & plays_now
        is2 = k.RANK_OF[ft] == 1
        valid_save = is_chop_focus & save_f[ft] & useful[ft] & (~is2 | critical[ft] | (p_count[ft] == 1)) & ~focus_bad
        n_extra_good = (new & ~bad_new & (k.SLOT != focus)).sum()
        la = lit_after & useful[None, :] & (remaining > 0)[None, :]
        known_play_s = la.any(1) & ~(la & ~playable_now[None, :]).any(1) & playable_now[p_id]
        n_extra_play = (new & known_play_s & (k.SLOT != focus) & ~bad_new).sum()
        # tempo clue: no new card touched, focus = leftmost re-touched card, which is
        # truly playable now (documented focus rule; low value per MCVP)
        tempo = (~any_new) & touched.any() & playable_now[ft] & useful[ft] & tempo_ok
        play_val = jnp.where(imm, 12.0, jnp.where(valid_play, 7.0, jnp.where(tempo, 3.0, 0.0)))
        save_val = jnp.where(valid_save, jnp.where(partner_busy, 5.0, 9.0), 0.0)
        v = jnp.maximum(play_val, save_val) + 1.5 * n_extra_good + 2.5 * n_extra_play \
            - 4.0 * n_bad + 0.5 * is_color
        valid = (valid_play | valid_save | tempo) & ~unsafe & (any_new | tempo)
        # a forced clue must not create a false promise about the focus card
        base_f = lit_after[focus] & useful & (remaining > 0)
        cand_after = jnp.where(cand_f.any(), cand_f, base_f)
        lie = ~cand_after[ft]
        forced_val = v - 30.0 * unsafe - 3.0 * (~valid) - 2.0 * (~any_new) - 8.0 * lie
        # locked-hand stall: re-touch only, and the focus card has no play or save
        # reading under Level 1 (nothing new is promised)
        stall_c = (~any_new) & touched.any() & ~play_f.any() & ~save_f.any()
        stall_v = 0.5 * (p_size[focus] == 1) + 0.1 * touched.sum()
        return valid, v, forced_val, is_chop_focus, imm, stall_c, stall_v

    c_valid, c_val, c_forced, c_chop, c_imm, c_stall, c_stall_v = jax.vmap(eval_clue)(jnp.arange(10))
    clue_legal = legal[10:20]
    clue_score = jnp.where(c_valid, 40.0 + c_val, -50.0 + c_forced)
    save_first = danger & ~partner_busy & (tokens > 0)
    clue_score = clue_score + jnp.where(save_first & c_chop & c_valid, 100.0, 0.0)
    # stall-breaker: at 7 tokens with nothing useful to do, alternate (by deck
    # parity) between discarding and a harmless clue so that the partner is not
    # forced to clue every turn and both hands keep cycling.
    harmless = (~c_valid) & (c_forced >= -5.5) & clue_legal
    no_valid = ~(c_valid & clue_legal).any()
    stall = (tokens == 7) & ~partner_busy & no_valid & (deck % 2 == 0) & ~my_all_playable.any()
    clue_score = jnp.where(stall & harmless, 35.0 + c_forced, clue_score)
    clue_score = jnp.where(clue_legal, clue_score, NEG)
    p_crit = (w * critical[None, :]).sum(1) / jnp.maximum(w.sum(1), 1e-6)

    # ---------------- plays ----------------
    play_pref = (2.0 * (my_size == 1) + 0.3 * (4 - my_min_rank)
                 + jnp.where(tokens == 0, 1.0 * (my_min_rank == 4), 0.0)
                 + 3.0 * (status[0] >= 2) + 1.5 * (status[0] == 1))
    play_score = jnp.where(my_all_playable, 100.0 + play_pref, NEG)

    # endgame: last turn when deck is empty
    last_turn = active & (deck == 0)
    partner_after = (actor == 1) & is_rem & (mem['deck_prev'] == 1)
    gamble_ok = last_turn & (lives >= 2)
    gamble_score = jnp.where(gamble_ok & (p_play > 0.0) & ~my_all_playable, 80.0 + 10.0 * p_play, NEG)
    play_score = jnp.maximum(play_score, gamble_score)
    clue_score = clue_score + jnp.where(last_turn & partner_after & ~partner_busy & c_imm & c_valid & clue_legal, 50.0, 0.0)
    play_score = jnp.where(legal[5:10], play_score, NEG)

    # ---------------- discards ----------------
    # locked hand (no chop): discard the card least likely to be critical
    disc_score = (jnp.full((5,), 10.0) + 0.1 * my_size - 12.0 * p_crit - 3.0 * my_all_playable
                  - 1.0 * (status[0] == 3) - 1.0 * (my_min_rank == 4))
    disc_score = jnp.where(my_all_trash, 46.0, disc_score)
    disc_score = jnp.where((k.SLOT == my_chop) & my_has_chop & ~my_all_trash.any(), 30.0, disc_score)
    disc_score = jnp.where(legal[0:5], disc_score, NEG)

    scores = jnp.concatenate([disc_score, play_score, clue_score, jnp.full((1,), NEG)])
    scores = jnp.where(legal, scores, NEG)
    best = jnp.argmax(scores).astype(jnp.int32)
    fallback = jnp.argmax(legal).astype(jnp.int32)
    action = jnp.where(scores[best] > NEG / 2, best, fallback)
    action = jnp.where(active, action, jnp.asarray(20, jnp.int32))

    trace = {
        'action': action,
        'tokens': tokens,
        'lives': lives,
        'deck': deck,
        'score': jnp.sum(height),
        'last_lost': has_last & is_rem & exhausted[jnp.argmax(last[28:53])] & ~(last[53] > 0.5)
                     & ~jnp.concatenate([jnp.zeros((1,), jnp.bool_), dead[:-1]])[jnp.argmax(last[28:53])]
                     | (has_last & is_rem & exhausted[jnp.argmax(last[28:53])] & ~(last[53] > 0.5)
                        & (k.RANK_OF[jnp.argmax(last[28:53])] == 0)),
        'r0_focus': r0[3],
        'r0_play': r0[5].any(),
        'r0_save': r0[6].any(),
        'r0_anynew': r0[7],
        'hint_to_me': hint_to_me,
        'r0_status': r0[2],
        'last_actor': actor,
        'last_type': atype,
        'last_id': jnp.argmax(last[28:53]),
        'p_chop_danger': danger,
        'my_chop': my_chop,
        'danger': danger,
        'partner_busy': partner_busy,
        'my_status': status[0],
        'my_size': my_size,
        'my_all_playable': my_all_playable,
        'p_known_play': p_known_play,
        'best_clue_val': jnp.max(clue_score),
        'my_cand': my_cand,
        'clue_score': clue_score,
        'c_valid': c_valid,
        'c_forced': c_forced,
    }
    return action, new_mem, trace
