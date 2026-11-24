"""Belief-based rule policy for two-player Hanabi (convention-light, XP oriented).

Core ideas
----------
* Per-slot identity posterior for my own hand: public remaining counts
  (multiplicity - discards - fireworks) minus the partner's visible cards,
  masked by the native literal-possibility mask, and multiplied by a private
  prior kept in memory.  Partner hints that touch a card which could be
  playable multiply the prior of the currently-playable identities, so the
  posterior probability of playability rises without ever overriding literal
  public evidence.
* Play a card only when its posterior playability clears a lives-dependent
  threshold (certain-only when down to one life, because bomb-out zeroes the
  score).
* Give hints scored by how many newly touched partner cards are playable
  (and become literally certain), penalising newly touched non-playable cards
  a partner might misplay.  Safe save hints (rank hints on critical 5s etc.)
  score positively only when the partner can literally see they are unplayable.
* Discard known-useless cards first, otherwise the oldest untouched card with
  the lowest expected criticality.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_rule_v1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

NC, NR, HS = 5, 5, 5
MULT_T = (3.0, 2.0, 2.0, 2.0, 1.0)

# Tunables
TRUST_SINGLE = 0.8   # trust that the unique focus card of a hint is playable
TRUST_MULTI = 0.5    # trust when several touched cards could be playable (below play threshold alone)
TRUST_SAVE = 0.4     # rank-5 hint on my chop card: probably a save, not a play
TRUST_SAVE_MID = 0.55  # rank 2-4 hint on my chop card: may be a critical-card save
# Effective-probability thresholds by lives 0..3; at one life only literal certainty plays.
PLAY_THR_T = (2.0, 1.01, 0.8, 0.75)
PLAY_THR_END_T = (2.0, 1.01, 0.6, 0.5)    # deck empty
HINT_MIN = 1.0
STALL_TOKENS = 5.0   # at or above this many tokens prefer a harmless hint to a discard
NEG = -1e9


def init_agent():
    return {
        'imp': jnp.zeros((HS, NC * NR), dtype=jnp.float32),   # identity set implied by last hint
        'trust': jnp.zeros((HS,), dtype=jnp.float32),          # trust in that implication
        'touched': jnp.zeros((HS,), dtype=jnp.bool_),
        'turn': jnp.asarray(0, dtype=jnp.int32),
    }


def _shift_rows(arr, k):
    """Remove row k, shift later rows up, and append a zero row (works for 1-D and 2-D)."""
    s = jnp.arange(HS)
    idx = s + (s >= k).astype(jnp.int32)
    gathered = arr[jnp.minimum(idx, HS - 1)]
    keep = (idx <= HS - 1)
    keep = keep.reshape((HS,) + (1,) * (arr.ndim - 1))
    return jnp.where(keep, gathered, jnp.zeros_like(arr))


def agent_act(obs, reset, memory, rng):
    init = init_agent()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    imp = memory['imp']
    trust = memory['trust']
    prev_touched = memory['touched']

    f = obs.features.astype(jnp.float32)
    legal = obs.legal_actions.astype(jnp.bool_)
    MULT = jnp.asarray(MULT_T, dtype=jnp.float32)
    PLAY_THR = jnp.asarray(PLAY_THR_T, dtype=jnp.float32)
    PLAY_THR_END = jnp.asarray(PLAY_THR_END_T, dtype=jnp.float32)

    # ---------------- decode observation ----------------
    partner = f[0:125].reshape(HS, NC, NR)                 # [slot, color, rank]
    deck = jnp.sum(f[127:167])
    fw = f[167:192].reshape(NC, NR)
    height = jnp.sum(fw * jnp.arange(1, NR + 1, dtype=jnp.float32)[None, :], axis=1)  # (5,) float
    tokens = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    disc = f[203:253].reshape(NC, 10)
    disc_counts = jnp.stack([
        disc[:, 0:3].sum(1), disc[:, 3:5].sum(1), disc[:, 5:7].sum(1),
        disc[:, 7:9].sum(1), disc[:, 9:10].sum(1)], axis=1)  # (5,5)
    la = f[253:308]
    know = f[308:658].reshape(2, HS, 35)
    my_mask = know[0, :, :25] > 1e-6                        # (5,25)
    my_flags = know[0, :, 25:35] > 0.5
    my_touched_now = jnp.any(my_flags, axis=1)             # (5,)
    my_occupied = jnp.any(my_mask, axis=1)
    p_mask = know[1, :, :25] > 1e-6                         # partner literal mask (5,25)
    p_touched_before = jnp.any(know[1, :, 25:35] > 0.5, axis=1)
    p_occupied = jnp.sum(partner, axis=(1, 2)) > 0.5

    ranks = jnp.arange(NR, dtype=jnp.float32)[None, :]      # (1,5)
    playable = (ranks == height[:, None])                   # (5,5) bool
    remaining = MULT[None, :] - disc_counts - (ranks < height[:, None]).astype(jnp.float32)
    remaining = jnp.maximum(remaining, 0.0)
    blocked = (remaining <= 0.5) & (ranks >= height[:, None])
    cum_blocked = jnp.cumsum(blocked.astype(jnp.int32), axis=1) - blocked.astype(jnp.int32)
    dead = (ranks < height[:, None]) | (cum_blocked > 0)
    crit = (remaining <= 1.5) & (~dead)                     # last remaining copy

    playable_f = playable.reshape(-1)
    dead_f = dead.reshape(-1)
    crit_f = crit.reshape(-1)
    rank_f = jnp.tile(jnp.arange(NR, dtype=jnp.float32), NC)

    partner_counts = jnp.sum(partner, axis=0).reshape(-1)   # (25,)
    count_adj = jnp.maximum(remaining.reshape(-1) - partner_counts, 0.0)

    # ---------------- last action decoding & memory update ----------------
    actor_partner = la[1] > 0.5
    actor_me = la[0] > 0.5
    t_play, t_discard, t_chint, t_rhint = la[2] > 0.5, la[3] > 0.5, la[4] > 0.5, la[5] > 0.5
    target_me = la[6] > 0.5
    match = la[18:23] > 0.5
    removed_slot = jnp.argmax(la[23:28]).astype(jnp.int32)

    # my own play/discard: shift the implied-set rows
    my_removal = actor_me & (t_play | t_discard)
    imp = jnp.where(my_removal, _shift_rows(imp, removed_slot), imp)
    trust = jnp.where(my_removal, _shift_rows(trust, removed_slot), trust)

    # partner hinted me: the focus cards are implied to be one of the identities
    # that are playable right now (and literally possible)
    hint_to_me = actor_partner & (t_chint | t_rhint) & target_me
    poss_play = my_mask & playable_f[None, :]               # (5,25)
    has_s = jnp.any(poss_play, axis=1)                      # (5,)
    newly = match & (~prev_touched)
    focus_new = newly & has_s
    focus = jnp.where(jnp.any(focus_new), focus_new, match & has_s)
    n_focus = jnp.sum(focus.astype(jnp.int32))
    tr_new = jnp.where(n_focus == 1, TRUST_SINGLE, TRUST_MULTI)
    # a rank-5 hint that touches my oldest untouched card is a save in most conventions
    my_untouched_prev = my_occupied & (~prev_touched)
    my_chop = my_untouched_prev & (jnp.cumsum(my_untouched_prev.astype(jnp.int32)) == 1)
    rank5_hint = t_rhint & (la[17] > 0.5)
    rank234_hint = t_rhint & (jnp.sum(la[14:17]) > 0.5)
    on_chop = jnp.any(focus & my_chop)
    tr_new = jnp.where(rank5_hint & on_chop, TRUST_SAVE, tr_new)
    tr_new = jnp.where(rank234_hint & on_chop, jnp.minimum(tr_new, TRUST_SAVE_MID), tr_new)
    upd = hint_to_me & focus
    imp = jnp.where(upd[:, None], poss_play.astype(jnp.float32), imp)
    trust = jnp.where(upd, jnp.maximum(trust, tr_new), trust)
    # implied sets must stay inside the literal mask
    imp = imp * my_mask.astype(jnp.float32)

    # ---------------- my hand posterior ----------------
    w_lit = my_mask.astype(jnp.float32) * count_adj[None, :]
    w_lit = w_lit / jnp.maximum(jnp.sum(w_lit, axis=1, keepdims=True), 1e-9)
    p_lit = jnp.sum(w_lit * playable_f[None, :], axis=1)   # literal-evidence playability
    imp_mass = jnp.sum(w_lit * imp, axis=1)
    imp_play = jnp.sum(w_lit * imp * playable_f[None, :], axis=1)
    m_imp = jnp.where(imp_mass > 1e-9, imp_play / jnp.maximum(imp_mass, 1e-9), 0.0)
    p_play = p_lit + (1.0 - p_lit) * trust * m_imp
    lit_certain = my_occupied & (p_lit >= 0.999)
    p_dead = jnp.sum(w_lit * dead_f[None, :], axis=1)
    p_crit = jnp.sum(w_lit * crit_f[None, :], axis=1)
    e_rank = jnp.sum(w_lit * rank_f[None, :], axis=1)
    hinted_me = my_touched_now | ((trust > 0.0) & (m_imp > 0.0))

    # ---------------- play decision ----------------
    thr_tab = jnp.where(deck < 0.5, PLAY_THR_END, PLAY_THR)
    thr = thr_tab[jnp.clip(lives, 0, 3)]
    play_val = jnp.where(lit_certain, 2.0, p_play) - 0.01 * e_rank + 0.001 * jnp.arange(HS, dtype=jnp.float32)
    play_val = jnp.where(legal[5:10], play_val, NEG)
    best_play = jnp.argmax(play_val)
    best_play_p = p_play[best_play]
    can_play = (play_val[best_play] > NEG / 2) & (lit_certain[best_play] | (best_play_p >= thr))

    # ---------------- hint scoring ----------------
    p_id = partner.reshape(HS, NC * NR)                     # one-hot identity
    p_playable = jnp.sum(p_id * playable_f[None, :], axis=1) > 0.5
    p_crit_card = jnp.sum(p_id * crit_f[None, :], axis=1) > 0.5
    p_certain_play = p_occupied & (~jnp.any(p_mask & (~playable_f[None, :]), axis=1))
    colors_of = partner.sum(axis=2) > 0.5                   # (5 slots, 5 colors)
    ranks_of = partner.sum(axis=1) > 0.5                    # (5 slots, 5 ranks)
    # touched[h, s] for the 10 hints (5 color, 5 rank)
    touched = jnp.concatenate([colors_of.T, ranks_of.T], axis=0)  # (10,5)
    color_attr = (jnp.arange(25) // NR)[None, :] == jnp.arange(NC)[:, None]   # (5,25)
    rank_attr = (jnp.arange(25) % NR)[None, :] == jnp.arange(NR)[:, None]     # (5,25)
    hint_attr = jnp.concatenate([color_attr, rank_attr], axis=0)              # (10,25)
    post_mask = p_mask[None, :, :] & hint_attr[:, None, :]                    # (10,5,25)
    cert_play_after = jnp.any(post_mask, axis=2) & (~jnp.any(post_mask & (~playable_f)[None, None, :], axis=2))
    cert_noplay_after = ~jnp.any(post_mask & playable_f[None, None, :], axis=2)
    newly_t = touched & (~p_touched_before)[None, :]
    untouched_occ = p_occupied & (~p_touched_before)
    chop = untouched_occ & (jnp.cumsum(untouched_occ.astype(jnp.int32)) == 1)

    # duplicate identities among touched playable cards: only the first counts as good,
    # the later copies become unplayable once the first is played.
    same_id = (p_id @ p_id.T) > 0.5                                           # (5,5)
    lower = jnp.arange(HS)[None, :] < jnp.arange(HS)[:, None]                 # [s, s'] s' < s
    dup = jnp.any(touched[:, None, :] & (same_id & lower)[None, :, :] & p_playable[None, None, :], axis=2)  # (10,5)
    pl = p_playable[None, :] & (~dup)
    cp = p_certain_play[None, :]
    good = pl & (~cp)
    contrib = jnp.where(good & newly_t, 2.0, 0.0)
    contrib = contrib + jnp.where(good & touched & (~newly_t), 0.6, 0.0)
    contrib = contrib + jnp.where(good & touched & cert_play_after, 1.2, 0.0)
    contrib = contrib + jnp.where(dup & touched, -1.0, 0.0)
    bad = (~pl) & touched & (~dup)
    p_dead_card = jnp.sum(p_id * dead_f[None, :], axis=1) > 0.5
    # a touched unplayable card that is literally unplayable now but still live may look
    # playable to the partner later (e.g. a 3 touched together with the 1), so keep a penalty
    safe_pen = jnp.where(p_dead_card[None, :], -0.2, -0.6)
    contrib = contrib + jnp.where(bad & newly_t, jnp.where(cert_noplay_after, safe_pen, -2.5), 0.0)
    contrib = contrib + jnp.where(bad & (~newly_t), jnp.where(cert_noplay_after, 0.0, -0.8), 0.0)
    save = bad & newly_t & chop[None, :] & p_crit_card[None, :]
    # saves become urgent when tokens are nearly gone: after the next play hint there
    # would be no token left to protect the partner's chop before they discard it
    save_bonus = 2.6 + jnp.where(tokens <= 2.5, 1.5, 0.0)
    # only literally safe saves (the partner can see the card is unplayable) are used;
    # ambiguous rank saves were tested and caused misplays by the recipient
    contrib = contrib + jnp.where(save, jnp.where(cert_noplay_after, save_bonus, 0.8), 0.0)
    # information value: a touched card that becomes literally known-dead gives the partner
    # a safe discard instead of a blind one (robust to any partner's conventions)
    pre_dead_known = p_occupied & (~jnp.any(p_mask & (~dead_f)[None, :], axis=1))          # (5,)
    post_dead_known = jnp.any(post_mask, axis=2) & (~jnp.any(post_mask & (~dead_f)[None, None, :], axis=2))
    new_dead = touched & post_dead_known & (~pre_dead_known)[None, :]
    contrib = contrib + jnp.where(new_dead, 0.8, 0.0)
    # small bonus for safe new information (newly touched cards the partner can see are unplayable)
    contrib = contrib + jnp.where(newly_t & cert_noplay_after & (~new_dead), 0.15, 0.0)
    contrib = jnp.where(touched, contrib, 0.0)
    hint_score = jnp.sum(contrib, axis=1)                   # (10,)
    # clarity bonuses: a single newly touched playable card is the least ambiguous hint;
    # otherwise prefer hints whose newest newly touched card is playable.
    n_new = jnp.sum(newly_t.astype(jnp.int32), axis=1)
    single_good = (n_new == 1) & jnp.any(newly_t & good, axis=1)
    newest_idx = jnp.argmax(jnp.where(newly_t, jnp.arange(HS)[None, :], -1), axis=1)
    newest_good = (n_new > 1) & jnp.take_along_axis(good & newly_t, newest_idx[:, None], axis=1)[:, 0]
    hint_score = hint_score + jnp.where(single_good, 0.5, 0.0) + jnp.where(newest_good, 0.25, 0.0)
    hint_score = jnp.where(legal[10:20], hint_score, NEG)
    best_hint = jnp.argmax(hint_score)
    best_hint_score = hint_score[best_hint]
    hint_min = HINT_MIN + jnp.where(tokens <= 2.5, 1.0, 0.0)
    can_hint = (best_hint_score > NEG / 2) & (best_hint_score >= hint_min)
    any_hint = best_hint_score > NEG / 2
    # urgent save: the partner's chop is critical and they have nothing known to play,
    # so they will probably discard it on their turn; protect it before playing myself
    save_scores = jnp.where(jnp.any(save, axis=1) & legal[10:20], hint_score, NEG)
    best_save = jnp.argmax(save_scores)
    best_save_score = save_scores[best_save]
    partner_will_discard = (~jnp.any(p_certain_play)) & (~jnp.any(p_touched_before & p_playable))
    # (urgent-save priority over own plays was tested and lowered scores; disabled)
    urgent_save = (tokens >= 0.5) & partner_will_discard & (best_save_score >= HINT_MIN) & (deck >= 2.5) & False
    # with surplus tokens, a harmless information hint beats a blind discard: it gives the
    # partner a turn to save a critical chop card and costs only a token we cannot use yet
    # near the end of the deck, discarding only hastens the final round; a harmless hint
    # keeps the deck for later while hands fill with hinted playable cards
    stall_tokens = jnp.where(deck <= 3.5, 1.0, STALL_TOKENS)
    stall_ok = (tokens >= stall_tokens) & (best_hint_score >= -0.35) & (deck >= 0.5)

    # ---------------- discard scoring ----------------
    # Predictable chop: discard the oldest untouched card unless another card is known
    # useless or clearly more critical.  Predictability lets the partner's save hints
    # land on the card that will actually be discarded.
    disc_val = jnp.where(p_dead > 0.999, -10.0, 0.0)
    disc_val = disc_val + 5.0 * p_crit + 3.0 * hinted_me.astype(jnp.float32) + 1.0 * p_play
    disc_val = disc_val + 0.5 * jnp.arange(HS, dtype=jnp.float32)
    disc_val = jnp.where(legal[0:5], disc_val, 1e9)
    best_disc = jnp.argmin(disc_val)
    can_disc = disc_val[best_disc] < 1e8

    # ---------------- choose ----------------
    endgame = deck < 0.5
    prefer_hint_over_disc = endgame & any_hint
    action = jnp.where(
        urgent_save, 10 + best_save,
        jnp.where(can_play, 5 + best_play,
                  jnp.where(can_hint | prefer_hint_over_disc | stall_ok, 10 + best_hint,
                            jnp.where(can_disc, best_disc,
                                      jnp.where(any_hint, 10 + best_hint, jnp.argmax(legal))))))
    action = action.astype(jnp.int32)
    # final legality guard
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))

    new_memory = {
        'imp': imp,
        'trust': trust,
        'touched': my_touched_now,
        'turn': memory['turn'] + 1,
    }
    trace = {
        'p_play': p_play,
        'best_play_p': best_play_p,
        'best_hint_score': best_hint_score,
        'lives': lives,
        'tokens': tokens.astype(jnp.int32),
        'height': height.astype(jnp.int32),
        'revealed': jnp.where(jnp.sum(la[28:53]) > 0.5, jnp.argmax(la[28:53]), -1).astype(jnp.int32),
        'partner_ids': jnp.where(p_occupied, jnp.argmax(p_id, axis=1), -1).astype(jnp.int32),
        'my_mask_n': jnp.sum(my_mask.astype(jnp.int32), axis=1),
        'match': match,
        # my own discard/bomb just made a still-needed identity extinct
        'lost_crit': (actor_me & (t_play | t_discard) & (jnp.sum(la[28:53]) > 0.5)
                      & (jnp.sum(la[28:53] * (remaining.reshape(-1) <= 0.5) * (~dead_f) * (rank_f >= jnp.repeat(height, NR))) > 0.5)),
        'was_bomb': actor_me & t_play & (la[53] < 0.5),
        'deck': deck.astype(jnp.int32),
        'best_is_save': jnp.any(save[best_hint]) | urgent_save,
        'chose_hint': (action >= 10) & (action <= 19),
        'partner_chop_crit': jnp.any(chop & p_crit_card & (~p_playable)),
    }
    return action, new_memory, trace
