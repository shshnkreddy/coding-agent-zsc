"""Literal-belief rule-based Hanabi policy (2 players).

Design goals (XP objective): rely only on literal, publicly verifiable
information (hints x remaining card counts), give hints that make partner
cards *certainly* playable whenever possible, avoid bombs, and never depend
on a private convention that the partner must share.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_rule_v1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

F32 = jnp.float32
GOOD_HINT = 0.7
LOW_HINT_TOKENS = 1.0
LOW_HINT_VALUE = 0.3
END_DECK = 0            # when deck <= END_DECK: prefer any non-harmful hint over a discard
END_THR = 0.5           # play threshold at lives>=2 in that endgame window
THR_PLAY_T = (1.01, 1.0, 0.75, 0.6)          # index = lives (0 unused)
THR_PLAY_LAST_T = (1.01, 1.0, 0.5, 0.15)
Q_TRUST_T = (0.0, 0.45, 0.5)                 # hint-intent bonus by trust level
P_MIN_BONUS = 0.3       # hint-intent bonus only applies when literal P(playable) >= this
W_CERTAIN = 3.0
W_REACH = 0.8
W_TOUCH_PLAY = 0.5
W_TOUCH_BAD = 0.3
W_SAVE_CHOP = 0.9
W_SAVE_OTHER = 0.15
W_SAVE_NOPLAY = 0.8     # extra save value for a critical chop when partner has no certain play
JOINT = 1               # 1: remove fully-known own identities from other slots' counts
W_DISC_CRIT = 10.0      # discard penalty per unit of P(critical)
W_DISC_SLOT = 0.1       # discard preference for older (lower-index) slots
CHOP_MODE = 2           # 0: oldest unhinted; 1: predicted min-risk slot; 2: both
W_ELIM = 0.003
W_DELTA = 2.0
REACH_P = 0.75


def _consts():
    mult = jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0], dtype=F32)            # copies per rank
    color_of = jnp.arange(25) // 5
    rank_of = jnp.arange(25) % 5
    # Hint attribute masks: rows 0..4 colour hints, rows 5..9 rank hints. Shape (10, 25).
    hint_attr = jnp.concatenate([
        (color_of[None, :] == jnp.arange(5)[:, None]),
        (rank_of[None, :] == jnp.arange(5)[:, None]),
    ], axis=0)
    neg = jnp.asarray(-1e9, dtype=F32)
    thr_play = jnp.asarray(THR_PLAY_T, dtype=F32)
    thr_play_lastturn = jnp.asarray(THR_PLAY_LAST_T, dtype=F32)
    q_trust = jnp.asarray(Q_TRUST_T, dtype=F32)
    return mult, color_of, rank_of, hint_attr, neg, thr_play, thr_play_lastturn, q_trust


def init_agent():
    return {
        'had_hint': jnp.zeros((5,), dtype=jnp.int32),
        'deck_prev': jnp.asarray(40, dtype=jnp.int32),
        'last_drawer_me': jnp.asarray(0, dtype=jnp.int32),
        'trust': jnp.asarray(1, dtype=jnp.int32),
        'gamble_pending': jnp.asarray(0, dtype=jnp.int32),
    }


def _safe_div(num, den):
    return jnp.where(den > 0, num / jnp.maximum(den, 1e-6), 0.0)


def _belief_stats(mask, rem, playable_f, useless_f, critical_f):
    """mask: (n,25) bool, rem: (25,) or (n,25) counts -> per-slot probabilities."""
    rem = jnp.broadcast_to(rem, mask.shape)
    w = mask.astype(F32) * rem
    tot = w.sum(1)
    w = jnp.where((tot > 0)[:, None], w, mask.astype(F32))
    tot = w.sum(1)
    p_play = _safe_div((w * playable_f[None, :]).sum(1), tot)
    p_useless = _safe_div((w * useless_f[None, :]).sum(1), tot)
    p_crit = _safe_div((w * critical_f[None, :]).sum(1), tot)
    return p_play, p_useless, p_crit, tot


def agent_act(obs, reset, memory, rng):
    MULT, COLOR_OF, RANK_OF, HINT_ATTR, NEG, THR_PLAY, THR_PLAY_LASTTURN, Q_TRUST = _consts()
    init = init_agent()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    f = obs.features.astype(F32)
    legal = obs.legal_actions
    active = jnp.logical_not(legal[20])

    # ---------------- decode ----------------
    partner = f[0:125].reshape(5, 25)                     # slot -> onehot identity
    occ_p = partner.sum(1) > 0
    idx_p = jnp.argmax(partner, axis=1)
    deck = f[127:167].sum()
    fw = f[167:192].reshape(5, 5)
    height = (fw * (jnp.arange(5, dtype=F32) + 1.0)[None, :]).sum(1)   # (5,) float
    tokens = f[192:200].sum()
    lives = f[200:203].sum()
    lives_i = jnp.clip(lives.astype(jnp.int32), 0, 3)
    dis = f[203:253].reshape(5, 10)
    disc = jnp.stack([dis[:, 0:3].sum(1), dis[:, 3:5].sum(1), dis[:, 5:7].sum(1),
                      dis[:, 7:9].sum(1), dis[:, 9:10].sum(1)], axis=1)   # (5c,5r)
    last = f[253:308]
    know = f[308:658].reshape(2, 5, 35)

    rank_grid = jnp.arange(5, dtype=F32)[None, :]                 # (1,5)
    playable = (height[:, None] == rank_grid)                     # (5c,5r)
    played = (rank_grid < height[:, None])
    rem_pub = jnp.clip(MULT[None, :] - disc - played.astype(F32), 0.0, 3.0)
    dead = (rem_pub <= 0) & jnp.logical_not(played)
    blocked = (jnp.cumsum(dead.astype(F32), axis=1) - dead.astype(F32)) > 0
    useless = played | blocked
    critical = (rem_pub == 1) & jnp.logical_not(useless)
    playable_f = playable.reshape(25).astype(F32)
    useless_f = useless.reshape(25).astype(F32)
    critical_f = critical.reshape(25).astype(F32)
    rem_pub_f = rem_pub.reshape(25)
    partner_counts = partner.sum(0)
    rem_me = jnp.clip(rem_pub_f - partner_counts, 0.0, 3.0)

    # ---------------- own beliefs ----------------
    mask_me = know[0, :, 0:25] > 0
    occ_me = mask_me.any(1)
    had_hint_now = (know[0, :, 25:35].sum(1) > 0).astype(jnp.int32)
    # joint sharpening: identities of fully-known own cards are removed from other slots' counts
    w_me = mask_me.astype(F32) * rem_me[None, :]
    known_me = ((w_me > 0).sum(1) == 1) & occ_me
    known_id = jnp.where(known_me[:, None], (w_me > 0).astype(F32), 0.0)      # (5,25)
    rem_me_slot = jnp.clip(rem_me[None, :] - known_id.sum(0)[None, :] + known_id, 0.0, 3.0)
    rem_me_slot = jnp.where(JOINT > 0, rem_me_slot, rem_me[None, :])
    p_play, p_useless, p_crit, _ = _belief_stats(mask_me, rem_me_slot, playable_f, useless_f, critical_f)
    p_five = _safe_div((mask_me.astype(F32) * rem_me_slot * (RANK_OF == 4)[None, :]).sum(1),
                       (mask_me.astype(F32) * rem_me_slot).sum(1))

    # ---------------- last action / memory bookkeeping ----------------
    actor_me = last[0] > 0.5
    actor_partner = last[1] > 0.5
    is_play = last[2] > 0.5
    is_hint = (last[4] > 0.5) | (last[5] > 0.5)
    hint_to_me = is_hint & actor_partner & (last[6] > 0.5)
    touched_me = last[18:23] > 0.5
    fresh = hint_to_me & touched_me & (memory['had_hint'] == 0) & occ_me
    n_fresh = fresh.sum()
    q = Q_TRUST[jnp.clip(memory['trust'], 0, 2)]
    bonus_q = jnp.where(n_fresh == 1, q, jnp.where(n_fresh > 1, 0.5 * q, 0.0))
    p_eff = jnp.where(fresh & (p_play >= P_MIN_BONUS), p_play + bonus_q * (1.0 - p_play), p_play)

    # trust update from my own previous gamble (seen on the inactive call)
    my_play_result = actor_me & is_play
    gamble_done = (memory['gamble_pending'] == 1) & my_play_result
    success = last[53] > 0.5
    trust_new = jnp.where(gamble_done, jnp.where(success, 2, 0), memory['trust'])
    gamble_pending_new = jnp.where(my_play_result | actor_partner, 0, memory['gamble_pending'])

    deck_i = deck.astype(jnp.int32)
    just_emptied = (deck_i == 0) & (memory['deck_prev'] > 0)
    last_drawer_me = jnp.where(just_emptied, actor_me.astype(jnp.int32), memory['last_drawer_me'])

    # ---------------- partner beliefs and hint evaluation ----------------
    mask_p = know[1, :, 0:25] > 0                                  # (5,25)
    had_hint_p = know[1, :, 25:35].sum(1) > 0
    ident_p = partner                                              # (5,25) onehot
    playable_p = (ident_p * playable_f[None, :]).sum(1) > 0
    critical_p = (ident_p * critical_f[None, :]).sum(1) > 0
    useless_p = (ident_p * useless_f[None, :]).sum(1) > 0
    five_p = (ident_p * (RANK_OF == 4)[None, :]).sum(1) > 0
    pp_before, pu_before, pc_before, _ = _belief_stats(mask_p, rem_pub_f, playable_f, useless_f, critical_f)
    certain_before = (pp_before >= 0.999) & occ_p
    cuseless_before = (pu_before >= 0.999) & occ_p

    touched = HINT_ATTR[:, idx_p] & occ_p[None, :]                 # (10,5)
    new_mask = mask_p[None, :, :] & jnp.where(touched[:, :, None], HINT_ATTR[:, None, :],
                                              jnp.logical_not(HINT_ATTR)[:, None, :])   # (10,5,25)
    nm = new_mask.reshape(50, 25)
    pp_after, pu_after, _, _ = _belief_stats(nm, rem_pub_f, playable_f, useless_f, critical_f)
    pp_after = pp_after.reshape(10, 5)
    pu_after = pu_after.reshape(10, 5)
    certain_after = (pp_after >= 0.999) & occ_p[None, :]
    cuseless_after = (pu_after >= 0.999) & occ_p[None, :]
    gain_certain = certain_after & jnp.logical_not(certain_before)[None, :]
    gain_useless = cuseless_after & jnp.logical_not(cuseless_before)[None, :]
    touched_new = touched & jnp.logical_not(had_hint_p)[None, :]
    eliminated = (mask_p.astype(F32).sum(1)[None, :] - new_mask.astype(F32).sum(2)) * occ_p[None, :]
    playable_pf = playable_p.astype(F32)[None, :]
    # partner "chop": oldest (lowest index) occupied slot without any positive hint
    unhinted_p = occ_p & jnp.logical_not(had_hint_p)
    chop_idx = jnp.argmax(unhinted_p)
    chop_p = (jnp.arange(5) == chop_idx) & unhinted_p
    # predicted partner discard under a literal min-risk rule (same formula as our own discards)
    slot_f5 = jnp.arange(5, dtype=F32)
    risk_p = (-W_DISC_CRIT * pc_before - 10.0 * (pc_before >= 0.5).astype(F32) - 3.0 * pp_before
              - 1.5 * had_hint_p.astype(F32) + 2.0 * pu_before - W_DISC_SLOT * slot_f5)
    risk_p = jnp.where(occ_p, risk_p, -1e6)
    pred_idx = jnp.argmax(risk_p)
    pred_p = (jnp.arange(5) == pred_idx) & occ_p
    chop_p = jnp.where(CHOP_MODE == 1, pred_p, jnp.where(CHOP_MODE == 2, pred_p | chop_p, chop_p))
    partner_no_play = jnp.logical_not(certain_before.any())
    save_chop_w = W_SAVE_CHOP + W_SAVE_NOPLAY * partner_no_play.astype(F32)
    save_weight = jnp.where(chop_p, save_chop_w, W_SAVE_OTHER).astype(F32)[None, :]
    crit_pf = critical_p.astype(F32)[None, :]
    reach_likely = ((pp_after >= REACH_P) & (pp_before[None, :] < REACH_P)).astype(F32)
    hint_value = (
        W_CERTAIN * gain_certain.astype(F32)
        + W_DELTA * jnp.clip(pp_after - pp_before[None, :], 0.0, 1.0) * playable_pf
        + W_REACH * reach_likely * playable_pf
        + W_TOUCH_PLAY * touched_new.astype(F32) * playable_pf
        - W_TOUCH_BAD * touched_new.astype(F32) * (1.0 - playable_pf) * (1.0 - crit_pf)
        + save_weight * touched_new.astype(F32) * crit_pf * (1.0 - playable_pf)
        + 0.6 * gain_useless.astype(F32)
        - 0.3 * touched.astype(F32) * useless_p.astype(F32)[None, :]
        + W_ELIM * eliminated
    ).sum(1)                                                       # (10,)
    hint_legal = legal[10:20]
    hint_value = jnp.where(hint_legal, hint_value, NEG)
    exp_partner_plays = (gain_certain.astype(F32) + 0.4 * touched_new.astype(F32) * playable_pf).sum(1)

    # ---------------- action scoring ----------------
    slot_f = jnp.arange(5, dtype=F32)
    thr = THR_PLAY[lives_i]
    endgame = (deck_i <= END_DECK) & (deck_i > 0)
    thr = jnp.where(endgame & (lives_i >= 2), jnp.minimum(thr, END_THR), thr)
    can_play = occ_me & (p_eff >= thr)
    play_score = jnp.where(can_play, 1000.0 + 100.0 * p_eff + 3.0 * p_five - 0.1 * slot_f, NEG)

    good_hint = (tokens > 0) & hint_legal & (hint_value >= GOOD_HINT)
    hint_good_score = jnp.where(good_hint, 800.0 + 10.0 * hint_value, NEG)

    can_discard = legal[0:5]
    useless_disc = can_discard & (p_useless >= 0.999)
    disc_useless_score = jnp.where(useless_disc, 700.0 - 0.1 * slot_f, NEG)

    low_hint = (tokens > 0) & hint_legal & (hint_value >= LOW_HINT_VALUE) & (tokens >= LOW_HINT_TOKENS)
    hint_low_score = jnp.where(low_hint, 600.0 + 10.0 * hint_value, NEG)
    end_hint = endgame & (tokens > 0) & hint_legal & (hint_value > -0.5)
    hint_low_score = jnp.where(end_hint, 750.0 + 10.0 * hint_value, hint_low_score)

    likely_crit = (p_crit >= 0.5).astype(F32)
    disc_risk = (-W_DISC_CRIT * p_crit - 10.0 * likely_crit - 3.0 * p_play
                 - 1.5 * had_hint_now.astype(F32) + 2.0 * p_useless - W_DISC_SLOT * slot_f)
    disc_score = jnp.where(can_discard, 500.0 + disc_risk, NEG)

    hint_any_score = jnp.where(hint_legal, 400.0 + 10.0 * hint_value, NEG)
    play_any_score = jnp.where(occ_me, 300.0 + 100.0 * p_eff, NEG)

    play_scores = jnp.maximum(play_score, play_any_score)
    disc_scores = jnp.maximum(disc_useless_score, disc_score)
    hint_scores = jnp.maximum(jnp.maximum(hint_good_score, hint_low_score), hint_any_score)

    # ---------------- final-turn override (deck empty at my active turn) ----------------
    last_turn = deck_i == 0
    partner_has_turn_after = last_drawer_me == 0
    thr_lt = THR_PLAY_LASTTURN[lives_i]
    thr_lt = jnp.where(partner_has_turn_after, thr_lt, jnp.where(lives_i >= 2, 1e-6, 1.0))
    can_play_lt = occ_me & (p_eff >= thr_lt)
    best_p = jnp.max(jnp.where(can_play_lt, p_eff, -1.0))
    best_hint_gain = jnp.max(jnp.where(hint_legal & (tokens > 0), exp_partner_plays, -1.0))
    hint_wins = partner_has_turn_after & (best_hint_gain > best_p) & (best_hint_gain > 0)
    play_lt = jnp.where(can_play_lt, 1000.0 + 100.0 * p_eff - 0.1 * slot_f, NEG)
    hint_lt = jnp.where(hint_legal & (tokens > 0),
                        jnp.where(hint_wins, 1100.0, 400.0) + 10.0 * exp_partner_plays + hint_value * 0.1, NEG)
    play_scores = jnp.where(last_turn, jnp.maximum(play_lt, play_any_score), play_scores)
    hint_scores = jnp.where(last_turn, hint_lt, hint_scores)
    disc_scores = jnp.where(last_turn, jnp.where(can_discard, 500.0 + disc_risk, NEG), disc_scores)

    scores = jnp.concatenate([disc_scores, play_scores, hint_scores, jnp.asarray([NEG])])
    scores = jnp.where(legal, scores, NEG)
    action = jnp.argmax(scores).astype(jnp.int32)
    action = jnp.where(active, action, jnp.asarray(20, dtype=jnp.int32))

    # gamble bookkeeping: did the fresh-hint bonus decide this play?
    chosen_slot = jnp.clip(action - 5, 0, 4)
    chose_play = (action >= 5) & (action <= 9) & active
    was_bonus_gamble = chose_play & (p_play[chosen_slot] < 0.999) & fresh[chosen_slot] & (p_play[chosen_slot] < thr)
    gamble_pending_new = jnp.where(was_bonus_gamble, 1, gamble_pending_new)

    new_memory = {
        'had_hint': had_hint_now,
        'deck_prev': deck_i,
        'last_drawer_me': last_drawer_me.astype(jnp.int32),
        'trust': trust_new.astype(jnp.int32),
        'gamble_pending': gamble_pending_new.astype(jnp.int32),
    }
    trace = {
        'p_play': p_play,
        'p_eff': p_eff,
        'p_useless': p_useless,
        'p_crit': p_crit,
        'partner_playable': playable_p.astype(jnp.int32),
        'partner_certain': certain_before.astype(jnp.int32),
        'partner_critical': critical_p.astype(jnp.int32),
        'partner_chop': chop_idx.astype(jnp.int32),
        'my_chop': jnp.argmax(occ_me & (had_hint_now == 0)).astype(jnp.int32),
        'hint_value': hint_value,
        'tokens': tokens,
        'lives': lives,
        'deck': deck,
        'trust': memory['trust'],
        'max_score': (5.0 - useless.astype(F32).sum(1) + played.astype(F32).sum(1)).sum(),
        'score': height.sum(),
    }
    return action, new_memory, trace
