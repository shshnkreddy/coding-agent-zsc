"""Belief-based rule policy for two-player Hanabi.

Design goals (cross-play robustness):
  * Play decisions rely on a posterior over own cards built from literal hint
    knowledge, public card counts (discards, fireworks, visible partner hand,
    own fully-known cards) and a soft prior that a fresh partner hint marks a
    currently playable card.  The prior is soft, so no private convention is
    required: a partner that only hints literally still leads to sane play.
  * Hints are chosen so that every touched card is either playable or safely
    non-playable under literal reasoning, and they prefer making a playable
    card literally certain.
  * Discards prefer dead cards, then the oldest un-hinted, least critical card.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_rule_v1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

# ---------------------------------------------------------------------------
# constants (built inside a function: module-scope calls are not permitted)
def _constants():
    orig = jnp.asarray([3.0, 2.0, 2.0, 2.0, 1.0], dtype=jnp.float32)        # copies per rank
    type_color = jnp.repeat(jnp.arange(5), 5)                               # [25] colour of type 5c+r
    type_rank = jnp.tile(jnp.arange(5), 5)                                  # [25] rank of type 5c+r
    # discard thermometer sub-blocks [0:3],[3:5],[5:7],[7:9],[9:10] -> rank
    sub = jnp.asarray([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0], [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1]], dtype=jnp.float32)
    slot_idx = jnp.arange(5)
    # hint h in 0..9 : 0..4 colour hints, 5..9 rank hints. attr_match[h, t]
    attr_match = jnp.concatenate([
        type_color[None, :] == jnp.arange(5)[:, None],
        type_rank[None, :] == jnp.arange(5)[:, None],
    ], axis=0)                                                              # [10, 25] bool
    return orig, type_color, type_rank, sub, slot_idx, attr_match


# tunable parameters
PLAY_THR_L1 = 0.999
PLAY_THR_L2 = 0.72
PLAY_THR_L3 = 0.70
EPS_1 = 0.10      # prior weight for non-target types when a hint touches 1 card
EPS_2 = 0.35      # ... 2 cards
EPS_3 = 0.70      # ... 3+ cards
EPS_FORCED = 0.35 # hint given while the partner had 8 tokens (may be forced)
EPS_RANK_MULTI = 0.35  # multi-card rank hint with enough colours at that rank (0.35 = same as any 2-card hint, i.e. disabled)
GAMBLE_P_SAFE = 0.02   # final-turn gamble threshold when a failure cannot hurt the partner's last play
GAMBLE_P_RISKY = 0.50  # ... when it would drop the partner to certainty-only
BETA_SAVE = 2.0   # prior weight for critical non-playable types when the hint touches my chop
HINT_THR = 2.0
HINT_THR_RICH = 1.0   # threshold when tokens are plentiful
HINT_THR_LAST = 2.0   # last token (2.0 = no reserve)
RICH_TOKENS = 6
DISC_AGE_W = 0.6
DISC_CRIT_W = 6.0
SAVE_GAIN = 3.5
CRIT_INFO_GAIN = 2.5     # proactive, literally safe mark on a non-chop critical card
MISLEAD_W_NEW = 5.0      # penalty per touched, un-hinted, non-playable card a literal reader could see as playable (5.0 = strict)
MISLEAD_W_OLD = 3.0      # same for previously hinted cards (3.0 = strict)
SAVE_MISLEAD_W = 5.0     # penalty for a chop save a literal reader could see as playable (5.0 = such saves are never chosen)
SAVE_GAIN_URGENT = 3.5   # partner has no hinted playable card (3.5 = no urgency bonus)


def init_agent():
    return {
        "prior": jnp.ones((5, 25), dtype=jnp.float32),
        "prev_deck": jnp.asarray(40, dtype=jnp.int32),
        "partner_last": jnp.asarray(0, dtype=jnp.int32),
        "prev_flags": jnp.zeros((5,), dtype=jnp.bool_),
    }


def _shift_prior(prior, k, slot_idx):
    """Remove slot k, shift later slots left, new uniform prior at slot 4."""
    src = jnp.where(slot_idx >= k, slot_idx + 1, slot_idx)
    gathered = prior[jnp.clip(src, 0, 4)]
    return jnp.where((src > 4)[:, None], jnp.ones_like(prior), gathered)


def agent_act(obs, reset, memory, rng):
    ORIG, TYPE_COLOR, TYPE_RANK, _SUB, SLOT_IDX, _ATTR_MATCH = _constants()
    f = obs.features
    legal = obs.legal_actions
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    prior = mem["prior"]
    prev_deck = mem["prev_deck"]
    partner_last = mem["partner_last"]

    # ------------------------------------------------------------------ decode
    partner_hand = f[0:125].reshape(5, 25)                     # slot -> type one-hot
    p_occ = partner_hand.sum(1) > 0.5
    p_type = jnp.argmax(partner_hand, axis=1)
    p_color = p_type // 5
    p_rank = p_type % 5
    deck = jnp.sum(f[127:167]).astype(jnp.int32)
    fw = f[167:192].reshape(5, 5)
    height = jnp.sum(fw * jnp.arange(1, 6)[None, :], axis=1).astype(jnp.int32)   # [5] 0..5
    tokens = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    disc = f[203:253].reshape(5, 10) @ _SUB                     # [5,5] discarded copies
    last = f[253:308]
    actor_me = last[0] > 0.5
    actor_partner = last[1] > 0.5
    la_play = last[2] > 0.5
    la_disc = last[3] > 0.5
    la_chint = last[4] > 0.5
    la_rhint = last[5] > 0.5
    la_recip_me = last[6] > 0.5
    la_color = jnp.argmax(last[8:13])
    la_rank = jnp.argmax(last[13:18])
    la_matches = last[18:23] > 0.5
    la_removed = jnp.argmax(last[23:28])
    know = f[308:658].reshape(2, 5, 35)
    own_mask = know[0, :, 0:25] > 0.0                           # literal possibilities
    own_occ = own_mask.any(axis=1)
    own_flags = know[0, :, 25:35].any(axis=1)
    part_mask = know[1, :, 0:25] > 0.0
    part_flags = know[1, :, 25:35].any(axis=1)

    # ------------------------------------------------------------ card classes
    playable_t = TYPE_RANK == height[TYPE_COLOR]                 # [25]
    played = (jnp.arange(5)[None, :] < height[:, None]).astype(jnp.float32)   # [5,5]
    avail = jnp.clip(ORIG[None, :] - disc, 0.0)                               # not discarded
    public_cnt = jnp.clip(avail - played, 0.0).reshape(25)                    # unseen-by-public
    blocked = (jnp.arange(5)[None, :] >= height[:, None]) & (avail <= 0.0)
    cum = jnp.cumsum(blocked.astype(jnp.int32), axis=1) - blocked.astype(jnp.int32)
    reach = cum == 0
    useful = (jnp.arange(5)[None, :] >= height[:, None]) & reach              # [5,5]
    critical = useful & (avail <= 1.0)
    useful_t = useful.reshape(25)
    critical_t = critical.reshape(25)

    # ----------------------------------------------------------- memory update
    prev_flags = mem["prev_flags"]
    # (a) shift the prior if I removed a card
    shifted = _shift_prior(prior, la_removed, SLOT_IDX)
    prior = jnp.where(actor_me & (la_play | la_disc), shifted, prior)
    # (b) a partner hint to me: concentrate the prior of touched slots on the
    #     hint-consistent, currently playable types (soft, eps mixing).  If the
    #     hint touches my chop (oldest card without any previous hint) and a
    #     critical non-playable type is consistent, the hint may be a save:
    #     keep those types at weight BETA_SAVE so that I neither play nor
    #     discard until more evidence arrives.
    consistent = jnp.where(la_chint, TYPE_COLOR == la_color, TYPE_RANK == la_rank)
    target = consistent & playable_t                             # [25]
    save_t = consistent & critical_t & ~playable_t               # [25]
    n_touched = jnp.sum(la_matches.astype(jnp.int32))
    eps = jnp.where(n_touched <= 1, EPS_1, jnp.where(n_touched == 2, EPS_2, EPS_3))
    # a rank hint touching k cards while >= k colours sit at that rank can mark k playable cards
    n_play_colors = jnp.sum((height == la_rank).astype(jnp.int32))
    rank_multi = la_rhint & (n_touched >= 2) & (n_play_colors >= n_touched)
    eps = jnp.where(rank_multi, jnp.minimum(eps, EPS_RANK_MULTI), eps)
    # a hint given at 8 tokens may have been forced (discard illegal): weaker play intent
    forced_hint = tokens == 7
    eps = jnp.where(forced_hint, jnp.maximum(eps, EPS_FORCED), eps)
    chop_cands_me = own_occ & ~prev_flags
    chop_me = chop_cands_me.any() & (SLOT_IDX == jnp.argmax(chop_cands_me))       # [5]
    play_prior = jnp.where(target, 1.0, eps).astype(jnp.float32)
    ambig_prior = jnp.where(target, 1.0, jnp.where(save_t, BETA_SAVE, eps)).astype(jnp.float32)
    new_prior = jnp.where(chop_me[:, None], ambig_prior[None, :], play_prior[None, :])   # [5,25]
    hint_to_me = actor_partner & (la_chint | la_rhint) & la_recip_me
    apply = hint_to_me & la_matches & jnp.any(target)
    prior = jnp.where(apply[:, None], new_prior, prior)
    # (c) end-of-deck bookkeeping: did the partner draw the last card?
    drew_last = (deck == 0) & (prev_deck > 0)
    partner_last = jnp.where(drew_last, (actor_partner & (la_play | la_disc)).astype(jnp.int32), partner_last)
    partner_last = jnp.where(deck > 0, 0, partner_last)
    new_mem = {"prior": prior, "prev_deck": deck, "partner_last": partner_last, "prev_flags": own_flags}

    # -------------------------------------------------------------- own belief
    partner_cnt = partner_hand.sum(0)                                          # [25]
    single = own_occ & (own_mask.sum(1) == 1)
    own_known = jnp.where(single[:, None], own_mask.astype(jnp.float32), 0.0)  # [5,25]
    own_known_tot = own_known.sum(0)
    cnt_slot = public_cnt[None, :] - partner_cnt[None, :] - (own_known_tot[None, :] - own_known)
    cnt_slot = jnp.clip(cnt_slot, 0.0)
    w = own_mask.astype(jnp.float32) * cnt_slot * prior
    norm = w.sum(1, keepdims=True)
    w_fallback = own_mask.astype(jnp.float32) * prior
    norm_fb = jnp.maximum(w_fallback.sum(1, keepdims=True), 1e-9)
    belief = jnp.where(norm > 1e-9, w / jnp.maximum(norm, 1e-9), w_fallback / norm_fb)
    p_play = (belief * playable_t[None, :]).sum(1)
    p_useful = (belief * useful_t[None, :]).sum(1)
    p_crit = (belief * critical_t[None, :]).sum(1)

    # ----------------------------------------------------------- hint scoring
    p_playable_true = p_occ & (p_rank == height[p_color])
    p_crit_true = p_occ & critical_t[p_type]
    touched = jnp.concatenate([
        p_occ[None, :] & (p_color[None, :] == jnp.arange(5)[:, None]),
        p_occ[None, :] & (p_rank[None, :] == jnp.arange(5)[:, None]),
    ], axis=0)                                                                 # [10,5]
    am = _ATTR_MATCH[:, None, :]                                               # [10,1,25]
    post_mask = part_mask[None, :, :] & jnp.where(touched[:, :, None], am, ~am)  # [10,5,25]
    pub_poss = public_cnt > 0.0
    post_poss = post_mask & pub_poss[None, None, :]
    pre_poss = part_mask & pub_poss[None, :]
    possibly_play_post = (post_poss & playable_t[None, None, :]).any(2)         # [10,5]
    certain_post = post_poss.any(2) & ~(post_poss & ~playable_t[None, None, :]).any(2)
    pre_certain = pre_poss.any(1) & ~(pre_poss & ~playable_t[None, :]).any(1)   # [5]
    same_id = (p_type[:, None] == p_type[None, :]) & p_occ[:, None] & p_occ[None, :]
    lower = SLOT_IDX[None, :] < SLOT_IDX[:, None]                               # [s, s'] s' < s
    dup_touched = jnp.einsum("hj,ij->hi", touched.astype(jnp.float32),
                             (same_id & lower).astype(jnp.float32)) > 0.5       # [10,5]
    eff_playable = p_playable_true[None, :] & ~dup_touched
    chop_cands = p_occ & ~part_flags
    chop_idx = jnp.argmax(chop_cands)
    chop_mask = chop_cands.any() & (SLOT_IDX == chop_idx)

    t = touched
    n_t = t.sum(1)                                                             # [10]
    # would the receiver read this hint on its chop as a possible save?
    save_types = _ATTR_MATCH[:, None, :] & part_mask[None, :, :] & (critical_t & ~playable_t & pub_poss)[None, None, :]
    ambiguous = save_types.any(2) & chop_mask[None, :]                         # [10,5]
    gain_play = (t & eff_playable & ~pre_certain[None, :]).astype(jnp.float32) * (
        jnp.where(part_flags, 2.0, 3.0)[None, :] + 2.0 * certain_post.astype(jnp.float32))
    gain_play = jnp.where(ambiguous & ~certain_post, jnp.minimum(gain_play, 2.0), gain_play)
    gain_untouched = (~t & eff_playable & certain_post & ~pre_certain[None, :]).astype(jnp.float32) * 3.0
    # a single-card hint on a critical chop card is read as a save: mild penalty only
    save_ok = chop_mask[None, :] & p_crit_true[None, :] & (n_t == 1)[:, None]
    mislead_w = jnp.where(save_ok, SAVE_MISLEAD_W, jnp.where(part_flags, MISLEAD_W_OLD, MISLEAD_W_NEW)[None, :])
    pen_mislead = (t & ~eff_playable & possibly_play_post).astype(jnp.float32) * mislead_w
    p_useful_true = p_occ & useful_t[p_type]
    pen_noise = (t & ~eff_playable & p_useful_true[None, :] & ~possibly_play_post & ~part_flags[None, :]).astype(jnp.float32) * 1.0
    partner_busy = (p_playable_true & part_flags).any()
    save_w = jnp.where(partner_busy, SAVE_GAIN, SAVE_GAIN_URGENT)
    gain_save = (t & chop_mask[None, :] & p_crit_true[None, :] & ~eff_playable).astype(jnp.float32) * save_w
    # identifying dead cards gives the partner a safe discard
    pre_known_dead = p_occ & ~(pre_poss & useful_t[None, :]).any(1)
    post_known_dead = p_occ[None, :] & ~(post_poss & useful_t[None, None, :]).any(2)
    gain_dead = (p_occ & ~p_useful_true & ~pre_known_dead)[None, :] & post_known_dead
    gain_dead = gain_dead.astype(jnp.float32) * 1.0
    # proactively marking a critical card that a literal reader cannot see as playable (e.g. rank 5)
    gain_crit = (t & p_crit_true[None, :] & ~eff_playable & ~possibly_play_post & ~part_flags[None, :] & ~chop_mask[None, :]).astype(jnp.float32) * CRIT_INFO_GAIN
    hint_score = (gain_play + gain_untouched - pen_mislead - pen_noise + gain_save + gain_dead + gain_crit).sum(1)   # [10]
    legal_hint = legal[10:20]
    hint_score = jnp.where(legal_hint, hint_score, -1e9)
    best_h = jnp.argmax(hint_score)
    best_h_score = hint_score[best_h]
    can_hint = legal_hint.any()

    # --------------------------------------------------------- discard scoring
    # types the partner holds in hinted slots will almost surely be played by the partner:
    # my own copy of such a type is effectively dead for discard purposes
    covered_t = (partner_hand * part_flags[:, None].astype(jnp.float32)).sum(0) > 0.5     # [25]
    p_useful_d = (belief * (useful_t & ~covered_t)[None, :]).sum(1)
    # dead cards first; otherwise the oldest un-hinted card (predictable chop), unless clearly critical
    disc_score = p_crit * DISC_CRIT_W + p_useful_d * 1.0 + own_flags.astype(jnp.float32) * 2.0 + p_play * 3.0 + DISC_AGE_W * SLOT_IDX
    disc_score = jnp.where(own_occ, disc_score, 1e9)
    best_d = jnp.argmin(disc_score)
    can_discard = legal[0:5].any()

    # ------------------------------------------------------------ play choice
    exp_rank = (belief * TYPE_RANK[None, :].astype(jnp.float32)).sum(1)          # [5] 0..4
    play_val = jnp.where(own_occ, p_play + 0.01 * (4.0 - exp_rank) / 4.0, -1.0)
    best_p = jnp.argmax(play_val)
    best_p_val = play_val[best_p]
    thr = jnp.where(lives <= 1, PLAY_THR_L1, jnp.where(lives == 2, PLAY_THR_L2, PLAY_THR_L3))
    final_turn = deck == 0

    a_play = 5 + best_p
    a_hint = 10 + best_h
    a_disc = best_d

    sure_play = own_occ.any() & (best_p_val >= thr)
    hint_thr = jnp.where(tokens >= RICH_TOKENS, HINT_THR_RICH, jnp.where(tokens <= 1, HINT_THR_LAST, HINT_THR))
    good_hint = can_hint & (best_h_score >= hint_thr)
    # final turn: if partner still acts after me, a good hint beats a gamble
    partner_after = partner_last > 0
    # a failed gamble only hurts if the partner still acts and holds a known play that a
    # dropped life would make it hold back
    gamble_thr = jnp.where((lives >= 3) | ~partner_after | ~partner_busy, GAMBLE_P_SAFE, GAMBLE_P_RISKY)
    gamble = final_turn & (lives >= 2) & own_occ.any() & (best_p_val > gamble_thr)
    # on my final turn a discard is worthless: any non-harmful hint is better if the partner still acts
    final_hint = final_turn & partner_after & can_hint & (best_h_score >= 0.0)
    # with one card left and no sure play, delaying the last draw gives the partner time for two known plays
    n_partner_known = (p_playable_true & part_flags).sum()
    delay_draw = (deck == 1) & can_hint & (n_partner_known >= 2)

    # with one card left in the deck, a real play hint before my own play lets both cards be played
    endgame_hint_first = (deck == 1) & can_hint & (best_h_score >= 3.0) & ~partner_busy
    # priority chain (later entries are fallbacks)
    action = jnp.where(can_discard, a_disc, jnp.where(can_hint, a_hint, a_play))
    action = jnp.where(final_hint | delay_draw, a_hint, action)
    action = jnp.where(good_hint, a_hint, action)
    action = jnp.where(gamble & ~(partner_after & good_hint), a_play, action)
    action = jnp.where(sure_play & ~endgame_hint_first, a_play, action)
    # non-final turn with no good hint and no discard possible (tokens == 8): hint anyway
    action = action.astype(jnp.int32)
    # legality safety net
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    mode = jnp.where(sure_play, 1, jnp.where(gamble, 4, jnp.where(good_hint, 2, jnp.where(can_discard, 3, 5)))).astype(jnp.int32)

    # diagnostics about the last public action
    rev = last[28:53]
    rev_t = jnp.argmax(rev)
    rev_c = rev_t // 5
    rev_r = rev_t % 5
    removed_card = (la_play | la_disc) & (rev.sum() > 0.5)
    # a point is lost when the removed card was still needed and no copy remains
    lost_point = removed_card & (rev_r >= height[rev_c]) & (avail[rev_c, rev_r] <= 0.0) & reach[rev_c, rev_r]
    trace = {
        "p_play": p_play.astype(jnp.float32),
        "p_crit": p_crit.astype(jnp.float32),
        "hint_score": hint_score.astype(jnp.float32),
        "mode": mode,
        "lives": lives,
        "tokens": tokens,
        "deck": deck,
        "last_play_fail": (la_play & (last[53] < 0.5)).astype(jnp.int32),
        "last_play_ok": (la_play & (last[53] > 0.5)).astype(jnp.int32),
        "lost_point": lost_point.astype(jnp.int32),
        "last_actor_me": actor_me.astype(jnp.int32),
        "last_disc": la_disc.astype(jnp.int32),
        "chop_crit": (chop_mask & p_crit_true).any().astype(jnp.int32),
        "hint_touches_chop": (touched[best_h] & chop_mask).any().astype(jnp.int32),
        "disc_is_chop": (chop_cands_me.any() & (best_d == jnp.argmax(chop_cands_me))).astype(jnp.int32),
        "n_partner_playable": p_playable_true.sum().astype(jnp.int32),
        "n_partner_certain": (p_playable_true & pre_certain).sum().astype(jnp.int32),
        "n_partner_hinted_playable": (p_playable_true & part_flags).sum().astype(jnp.int32),
        # chosen hint touches a critical non-playable chop card that a literal reader could still see as playable
        # partner holds a playable, not-yet-known card but no hint reaches the threshold
        "stuck_playable": ((p_playable_true & ~pre_certain & ~part_flags).any() & (best_h_score < HINT_THR)).astype(jnp.int32),
        "n_mislead_best": (touched[best_h] & ~eff_playable[best_h] & possibly_play_post[best_h]).sum().astype(jnp.int32),
        "risky_save": (touched[best_h] & chop_mask & p_crit_true & ~p_playable_true & possibly_play_post[best_h]).any().astype(jnp.int32),
        "safe_save": (touched[best_h] & chop_mask & p_crit_true & ~p_playable_true & ~possibly_play_post[best_h]).any().astype(jnp.int32),
    }
    return action, new_mem, trace
