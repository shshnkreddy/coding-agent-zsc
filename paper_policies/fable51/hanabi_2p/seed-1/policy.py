"""Grounded two-player Hanabi policy.

Design (convention-light, robust to partner differences):
  * Own-card beliefs come from the native literal possibility mask (V0 > 0)
    weighted by remaining card counts (public counts minus the partner's
    visible hand).
  * Play a card when its playability probability clears a lives-dependent
    threshold.  A card touched by a partner hint while it had a non-trivial
    public playability gets a "likely-playable" marker (mild intent prior).
    The prior is scaled down when the hint touched more new cards than there
    are playable identities matching it (ambiguous focus) and shrinks after
    a marked card bombs.  A touched card that could not be playable gets a
    "protect" marker (the hint was informational / a save), lowering its
    discard priority.
  * Hints are scored by simulating the partner's public knowledge after the
    hint: reward touched cards that are truly playable and that the partner
    would then play (or that become certain), penalise touching non-playable
    cards the partner would play, and reward hints that move the partner's
    predicted next discard off a critical card (rank-weighted).
  * Discards prefer certainly-useless cards, then the oldest unhinted card
    with low (rank-weighted) critical probability, so the discard is
    predictable for the partner.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_belief_v4', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

F32 = jnp.float32
MARK_MIN_P = 0.15   # public playability needed for a hint to mark a card
BOOST = 0.8         # marker boost: p_eff = p + BOOST * scale * (1 - p)
BOOST_AFTER_BOMB = 0.5   # boost once a marked card has bombed
WEAK_HINT_TOKENS = 7.0   # weak info hints only when tokens >= this
MED_HINT_TOKENS = 1.0    # medium hints (0.5 <= score < 1) only when tokens >= this
ENDGAME_DECK = 4.0       # deck size at/below which we prefer hinting to discarding
THR_L3 = 0.7
THR_L2 = 0.85
PROTECT_PENALTY = 1.0
HINTED_PENALTY = 0.5
AGE_BONUS = 0.1
CRIT_WEIGHT = 3.0
SAVE_WEIGHT = 0.8
URGENT_SAVE = 1
MARK_ONESHOT = 1
TOUCH_NONPLAY_PENALTY = 0.25
RANK5_SCALE = 0.5        # intent prior multiplier for rank-5 hints (often saves)
TOUCH_PEN_CRIT = 0       # 1: apply touch penalty to critical cards too


def _consts():
    mult = jnp.array([3.0, 2.0, 2.0, 2.0, 1.0], dtype=F32)
    color_of = jnp.arange(25) // 5
    rank_of = jnp.arange(25) % 5
    # [10, 25] attribute masks: hints 0..4 = colors, 5..9 = ranks
    attr = jnp.concatenate([
        (color_of[None, :] == jnp.arange(5)[:, None]),
        (rank_of[None, :] == jnp.arange(5)[:, None]),
    ], axis=0).astype(F32)
    slot_idx = jnp.arange(5, dtype=F32)
    neg = jnp.asarray(-1e9, dtype=F32)
    return mult, attr, slot_idx, neg


def init_agent():
    # [0:5] own "likely playable" marker strength (0..100), [5:10] own protect markers,
    # [10] final drawer (0 unknown, 1 self, 2 partner), [11] marked-card bombs,
    # [12:17] partner marker strength estimate, [17:22] partner protect estimate
    return jnp.zeros(22, dtype=jnp.int32)


def _card_probs(mask, counts, playable, useless, critval):
    """mask [n,25] bool-ish, counts [25] -> per-card probabilities."""
    w = mask * jnp.maximum(counts, 0.0)[None, :]
    tot = w.sum(1)
    w = jnp.where(tot[:, None] > 0, w, mask)
    tot = jnp.maximum(w.sum(1), 1e-6)
    p_play = (w * playable[None, :]).sum(1) / tot
    p_useless = (w * useless[None, :]).sum(1) / tot
    p_crit = (w * critval[None, :]).sum(1) / tot
    return p_play, p_useless, p_crit


def _shift(v, removed):
    ext = jnp.concatenate([v, jnp.zeros(1, v.dtype)])
    return jnp.where(jnp.arange(5) < removed, v, ext[1:])


def _discard_pref(p_useless, p_crit, p_play, hinted, mark, prot, occ, slot_idx):
    d = (2.0 * p_useless - CRIT_WEIGHT * p_crit - 1.0 * p_play
         - HINTED_PENALTY * hinted - 0.5 * mark - PROTECT_PENALTY * prot
         + AGE_BONUS * (4.0 - slot_idx))
    d = jnp.where(p_useless >= 0.999, 100.0 + 0.01 * (4.0 - slot_idx), d)
    return jnp.where(occ > 0, d, -1e6)


def agent_act(obs, reset, memory, rng):
    MULT, ATTR, SLOT_IDX, NEG = _consts()
    f = obs.features.astype(F32)
    legal = obs.legal_actions
    memory = jnp.where(reset, jnp.zeros_like(memory), memory)
    markers = memory[0:5]
    protect = memory[5:10]
    final_drawer = memory[10]
    marked_bombs = memory[11]
    p_markers = memory[12:17]
    p_protect = memory[17:22]

    # ---------------- decode observation ----------------
    partner = f[0:125].reshape(5, 5, 5)                 # slot, color, rank
    partner_flat = partner.reshape(5, 25)
    deck = f[127:167].sum()
    fw = f[167:192].reshape(5, 5)
    height = (fw * (jnp.arange(5, dtype=F32) + 1.0)[None, :]).sum(1)   # [5] 0..5
    tokens = f[192:200].sum()
    lives = f[200:203].sum()
    disc = f[203:253].reshape(5, 10)
    disc_counts = jnp.stack([
        disc[:, 0:3].sum(1), disc[:, 3:5].sum(1), disc[:, 5:7].sum(1),
        disc[:, 7:9].sum(1), disc[:, 9:10].sum(1)], axis=1)          # [5 colors, 5 ranks]
    last = f[253:308]
    know = f[308:658].reshape(2, 5, 35)

    ranks = jnp.arange(5, dtype=F32)[None, :]
    placed = (ranks < height[:, None]).astype(F32)                      # [5,5]
    public_rem = MULT[None, :] - disc_counts - placed                   # [5,5]
    partner_counts = partner.sum(0)                                     # [5,5]
    private_rem = public_rem - partner_counts

    playable = (ranks == height[:, None]).astype(F32)                   # [5,5]
    blocked = ((public_rem <= 0) & (ranks >= height[:, None])).astype(F32)
    cum_blocked = jnp.cumsum(blocked, axis=1) - blocked                 # exclusive cumsum
    useful = ((ranks >= height[:, None]) & (cum_blocked == 0)).astype(F32)
    useless = 1.0 - useful
    critical = useful * (public_rem == 1).astype(F32)
    critval = critical * (1.0 + 0.25 * (4.0 - ranks))                  # 5s:1.0 ... 1s:2.0

    playable_f = playable.reshape(25)
    useless_f = useless.reshape(25)
    critval_f = critval.reshape(25)
    public_rem_f = public_rem.reshape(25)
    private_rem_f = private_rem.reshape(25)

    # ---------------- own beliefs ----------------
    own_mask = (know[0, :, :25] > 0).astype(F32)                        # [5,25]
    own_occ = (own_mask.sum(1) > 0).astype(F32)                         # V0 is all-zero in empty slots
    own_hinted = (know[0, :, 25:35].sum(1) > 0).astype(F32)
    p_play, p_useless, p_crit = _card_probs(own_mask, private_rem_f, playable_f, useless_f, critval_f)
    p_play_pub, _, _ = _card_probs(own_mask, public_rem_f, playable_f, useless_f, critval_f)
    p_play = p_play * own_occ
    p_play_pub = p_play_pub * own_occ

    # ---------------- partner public beliefs ----------------
    p_occ = (partner_flat.sum(1) > 0).astype(F32)                       # [5]
    p_hinted = (know[1, :, 25:35].sum(1) > 0).astype(F32)
    p_mask = (know[1, :, :25] > 0).astype(F32)                          # [5,25]
    pb_play, pb_useless, pb_crit = _card_probs(p_mask, public_rem_f, playable_f, useless_f, critval_f)
    pb_play = pb_play * p_occ

    # ---------------- memory update from last action ----------------
    actor_self = last[0] > 0.5
    actor_partner = last[1] > 0.5
    is_play = last[2] > 0.5
    is_discard = last[3] > 0.5
    is_hint = (last[4] > 0.5) | (last[5] > 0.5)
    recip_self = last[6] > 0.5
    recip_partner = last[7] > 0.5
    matched = last[18:23] > 0.5
    removed = jnp.argmax(last[23:28])
    success = last[53] > 0.5
    hint_id = jnp.argmax(last[8:18])                                     # 0..4 color, 5..9 rank
    n_play_attr = (playable_f * ATTR[hint_id]).sum()                     # playable identities matching hint

    self_removed = actor_self & (is_play | is_discard)
    partner_removed = actor_partner & (is_play | is_discard)
    bombed_marked = actor_self & is_play & (~success) & (markers[removed] > 0)
    marked_bombs = marked_bombs + bombed_marked.astype(jnp.int32)
    markers = jnp.where(self_removed, _shift(markers, removed), markers)
    protect = jnp.where(self_removed, _shift(protect, removed), protect)
    p_markers = jnp.where(partner_removed, _shift(p_markers, removed), p_markers)
    p_protect = jnp.where(partner_removed, _shift(p_protect, removed), p_protect)
    if MARK_ONESHOT:
        # intent markers survive only until the holder's next action
        markers = jnp.where(actor_self, jnp.zeros_like(markers), markers)
        p_markers = jnp.where(actor_partner, jnp.zeros_like(p_markers), p_markers)

    # hint to me
    hint_to_me = is_hint & recip_self
    new_mark_me = matched & (p_play_pub >= MARK_MIN_P) & (markers == 0)
    k_me = jnp.maximum(new_mark_me.sum().astype(F32), 1.0)
    rank5_factor = jnp.where(hint_id == 9, RANK5_SCALE, 1.0)   # rank-5 hints are often saves
    scale_me = jnp.minimum(1.0, n_play_attr / k_me) * rank5_factor
    strength_me = jnp.round(100.0 * scale_me).astype(jnp.int32)
    markers = jnp.where(hint_to_me, jnp.where(new_mark_me, jnp.maximum(markers, strength_me), markers), markers)
    new_protect_me = (matched & (p_play_pub < MARK_MIN_P)).astype(jnp.int32)
    protect = jnp.where(hint_to_me, jnp.maximum(protect, new_protect_me), protect)
    # hint to partner (from me)
    hint_to_p = is_hint & recip_partner
    new_mark_p = matched & (pb_play >= MARK_MIN_P) & (p_markers == 0)
    k_p = jnp.maximum(new_mark_p.sum().astype(F32), 1.0)
    scale_p = jnp.minimum(1.0, n_play_attr / k_p) * rank5_factor
    strength_p = jnp.round(100.0 * scale_p).astype(jnp.int32)
    p_markers = jnp.where(hint_to_p, jnp.where(new_mark_p, jnp.maximum(p_markers, strength_p), p_markers), p_markers)
    new_protect_p = (matched & (pb_play < MARK_MIN_P)).astype(jnp.int32)
    p_protect = jnp.where(hint_to_p, jnp.maximum(p_protect, new_protect_p), p_protect)

    occ_i = own_occ.astype(jnp.int32)
    markers = markers * occ_i
    protect = protect * occ_i
    p_occ_i = p_occ.astype(jnp.int32)
    p_markers = p_markers * p_occ_i
    p_protect = p_protect * p_occ_i
    final_drawer = jnp.where(
        (deck <= 0) & (final_drawer == 0) & (is_play | is_discard),
        jnp.where(actor_self, 1, 2), final_drawer)

    # ---------------- play evaluation ----------------
    mark_f = markers.astype(F32) / 100.0
    prot_f = protect.astype(F32)
    boost = jnp.where(marked_bombs >= 2, 0.0, jnp.where(marked_bombs >= 1, BOOST_AFTER_BOMB, BOOST))
    p_eff = p_play + boost * mark_f * (1.0 - p_play)
    p_eff = jnp.where(p_play > 0, p_eff, 0.0)
    thr = jnp.where(lives >= 3, THR_L3, jnp.where(lives >= 2, THR_L2, 0.999))
    deck_empty = deck <= 0
    last_turn = deck_empty & (final_drawer == 1)
    risky_ok = deck_empty & (lives >= 2)
    safe = p_eff >= thr
    play_scores = jnp.where(
        safe, 100.0 + p_eff + 0.01 * SLOT_IDX,
        jnp.where(risky_ok & (p_play > 0), 70.0 + p_play, 10.0 + p_play))

    # ---------------- hint evaluation ----------------
    truly_play = (partner_flat * playable_f[None, :]).sum(1)            # [5]
    truly_critval = (partner_flat * critval_f[None, :]).sum(1)
    pm_f = p_markers.astype(F32) / 100.0
    pp_f = p_protect.astype(F32)
    pb_eff = pb_play + BOOST * pm_f * (1.0 - pb_play)
    partner_will_play_before = (pb_eff >= thr) & (p_occ > 0)
    partner_has_play = partner_will_play_before.any()

    touched = ((partner_flat[None, :, :] * ATTR[:, None, :]).sum(2) > 0).astype(F32)   # [10,5]
    tch = touched > 0
    new_mask = jnp.where(touched[:, :, None] > 0,
                         p_mask[None] * ATTR[:, None, :],
                         p_mask[None] * (1.0 - ATTR[:, None, :]))                  # [10,5,25]
    pa_play, pa_useless, pa_crit = _card_probs(new_mask.reshape(50, 25), public_rem_f, playable_f, useless_f, critval_f)
    pa_play = pa_play.reshape(10, 5) * p_occ[None, :]
    pa_useless = pa_useless.reshape(10, 5)
    pa_crit = pa_crit.reshape(10, 5)

    certain_after = pa_play >= 0.999
    certain_before = (pb_play >= 0.999)[None, :]
    already_marked = (p_markers > 0)[None, :]
    newly_marked = tch & (pa_play >= MARK_MIN_P) & ~already_marked & (p_occ[None, :] > 0)   # [10,5]
    k_h = jnp.maximum(newly_marked.sum(1).astype(F32), 1.0)                                # [10]
    n_h = (playable_f[None, :] * ATTR).sum(1)                                              # [10]
    scale_h = jnp.minimum(1.0, n_h / k_h) * jnp.where(jnp.arange(10) == 9, RANK5_SCALE, 1.0)
    mark_strength_after = jnp.where(newly_marked, scale_h[:, None], pm_f[None, :])         # [10,5]
    pa_eff = pa_play + BOOST * mark_strength_after * (1.0 - pa_play)
    will_play = (pa_eff >= thr) & (p_occ[None, :] > 0)
    tp = truly_play[None, :] > 0
    wpb = partner_will_play_before[None, :]
    gain_play = (1.0 * (certain_after & ~certain_before)
                 + 0.6 * (will_play & ~certain_after & ~wpb)
                 + 0.15 * (tch & ~will_play & ~certain_after)
                 + 0.1 * (tch & will_play & ~certain_after & wpb))
    risk = (1.2 * (will_play & ~wpb)
            + 0.6 * (tch & will_play & wpb))
    useless_gain = 0.2 * ((pa_useless >= 0.999) & (pb_useless < 0.999)[None, :])
    # safety margin for partners who play any touched card: penalise touching a
    # non-playable card that could still look playable to the partner
    touch_pen = TOUCH_NONPLAY_PENALTY * (tch & (pa_play > 0.0) & ~will_play
                                         & ((truly_critval[None, :] <= 0.0) | (TOUCH_PEN_CRIT > 0)))
    per_card = jnp.where(tp, gain_play, -risk + useless_gain - touch_pen) + 0.05 * touched
    hint_scores = (per_card * p_occ[None, :]).sum(1)                    # [10]

    # predicted partner discard before / after each hint
    d_before = _discard_pref(pb_useless, pb_crit, pb_play, p_hinted, pm_f, pp_f, p_occ, SLOT_IDX)
    chop_b = jnp.argmax(d_before)
    hinted_after = jnp.maximum(p_hinted[None, :], touched)
    prot_after = jnp.maximum(pp_f[None, :], (tch & (pa_play < MARK_MIN_P)).astype(F32))
    d_after = _discard_pref(pa_useless, pa_crit, pa_play, hinted_after, mark_strength_after, prot_after,
                            p_occ[None, :], SLOT_IDX[None, :])
    chop_a = jnp.argmax(d_after, axis=1)                                 # [10]
    crit_b = truly_critval[chop_b] * (p_occ[chop_b] > 0)
    crit_a = truly_critval[chop_a] * (p_occ[chop_a] > 0)
    save_gain = SAVE_WEIGHT * (crit_b - crit_a)
    hint_scores = hint_scores + save_gain

    any_risk = ((risk > 0) & ~tp & (p_occ[None, :] > 0)).any(1)
    clean_save = (save_gain >= 0.6) & ~any_risk
    urgent = (URGENT_SAVE > 0) & clean_save & (~partner_has_play) & (deck > 3) & (tokens < 8)
    weak_ok = (tokens >= WEAK_HINT_TOKENS) | (deck <= ENDGAME_DECK)
    hint_tier = jnp.where(hint_scores >= 1.0, 80.0,
                          jnp.where((hint_scores >= 0.5) & (tokens >= MED_HINT_TOKENS), 75.0,
                                    jnp.where((hint_scores >= 0.1) & weak_ok, 60.0, 40.0)))
    hint_tier = jnp.where(urgent, 102.0, hint_tier)
    hint_tier = jnp.where(last_turn, jnp.minimum(hint_tier, 30.0), hint_tier)
    hint_action_scores = hint_tier + hint_scores

    # ---------------- discard evaluation ----------------
    d_own = _discard_pref(p_useless, p_crit, p_play, own_hinted, mark_f, prot_f, own_occ, SLOT_IDX)
    discard_scores = jnp.where(p_useless >= 0.999, 65.0 + 0.01 * (4.0 - SLOT_IDX), 50.0 + d_own)

    # ---------------- assemble ----------------
    scores = jnp.concatenate([discard_scores, play_scores, hint_action_scores, jnp.zeros(1, F32)])
    scores = jnp.where(legal, scores, NEG)
    action = jnp.argmax(scores).astype(jnp.int32)

    new_memory = jnp.concatenate([markers, protect, final_drawer[None], marked_bombs[None],
                                  p_markers, p_protect]).astype(jnp.int32)
    trace = {
        'p_play': p_play,
        'p_eff': p_eff,
        'hint_scores': hint_scores,
        'markers': markers,
        'protect': protect,
        'd_own': d_own,
    }
    return action, new_memory, trace
