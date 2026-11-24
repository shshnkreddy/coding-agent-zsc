"""Convention-light rule-based two-player Hanabi policy (JAX, fixed shape).

Core ideas (all derived from public information + own private memory):
  * Card counting: every own-slot possibility set = native literal mask
    restricted to identities that are still unaccounted for (public counts
    minus discards, fireworks and the partner's visible hand).
  * Play certainly-playable cards. Additionally trust the partner's most recent
    hint as a "play" signal (focus = newly touched cards, else all touched),
    with a probability floor that adapts within the episode to failed trust.
  * Hints: score every legal hint by how many partner cards it marks as
    playable, with a strong preference for hints that make a playable card
    provably playable from public information (works for partners that
    reason probabilistically and for partners that just play hinted cards).
    Hints whose focus would be an unplayable card that could be mistaken as
    playable are heavily penalised. Rank-5 saves on the partner's chop.
  * Discards: provably useless first, then oldest unhinted, never focus cards
    unless nothing else is available.
  * Final-turn logic: once the deck is empty each seat has exactly one turn.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'convention_light_v1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

MEM_SIZE = 44
# [41]    partner followed my play hints (count)
# [42]    partner ignored my play hints (count)
# [43]    bitmask of partner slots my last hint marked as play candidates

# Static behaviour switches (all False for the submitted policy). They exist so
# that exploratory partner variants with different conventions can be built by
# copying this file and flipping flags.
CFG_TRUST_ALL_TOUCHED = False   # receiver: every touched card is a play candidate
CFG_NEWEST_ONLY = False         # receiver: only the newest touched card is a play candidate
CFG_NO_SAVES = False            # neither give nor recognise save hints
CFG_PROB_ONLY = False           # receiver never trusts hints; plays only at p>=0.75
# memory layout
# [0:5]   focus flags per own slot
# [5:10]  previous positive-hint bitmask per own slot
# [10]    focus-play failures
# [11]    focus-play successes
# [12]    my last action was a focus (uncertain) play
# [13]    i drew the last card of the deck
# [14]    previous deck size
# [15]    turn counter

_MULT_LIST = [3.0, 2.0, 2.0, 2.0, 1.0]
_DISC_LIST = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]


def _constants():
    mult = jnp.asarray(_MULT_LIST, dtype=jnp.float32)
    disc_m = jnp.asarray(_DISC_LIST, dtype=jnp.float32)
    r = jnp.arange(5)
    return mult, disc_m, r


def init_agent():
    return jnp.zeros((MEM_SIZE,), dtype=jnp.int32)


def _safe_div(num, den):
    return jnp.where(den > 0, num / jnp.maximum(den, 1e-6), 0.0)


def agent_act(obs, reset, memory, rng):
    _MULT, _DISC_M, _R = _constants()
    _S = _R
    f = obs.features.astype(jnp.float32)
    legal = obs.legal_actions
    mem = jnp.where(reset, init_agent(), memory)

    # ---------------- parse observation ----------------
    ph = f[0:125].reshape(5, 5, 5)                     # partner hand [slot, color, rank]
    deck = jnp.sum(f[127:167])
    fw = f[167:192].reshape(5, 5)
    height = jnp.sum(fw * jnp.arange(1, 6, dtype=jnp.float32)[None, :], axis=1)  # [c]
    tokens = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    disc = f[203:253].reshape(5, 10) @ _DISC_M         # [c, r]
    la = f[253:308]
    kn = f[308:658].reshape(2, 5, 35)
    myV = kn[0, :, 0:25].reshape(5, 5, 5)
    my_flags = jnp.concatenate([kn[0, :, 25:30], kn[0, :, 30:35]], axis=1)   # [s, 10]
    pV = kn[1, :, 0:25].reshape(5, 5, 5)
    p_flags = jnp.concatenate([kn[1, :, 25:30], kn[1, :, 30:35]], axis=1)

    la_actor_self = la[0] > 0.5
    la_actor_partner = la[1] > 0.5
    la_play = la[2] > 0.5
    la_discard = la[3] > 0.5
    la_hint = (la[4] + la[5]) > 0.5
    la_to_me = la[6] > 0.5
    la_match = la[18:23] > 0.5
    la_removed = jnp.argmax(la[23:28])
    la_success = la[53] > 0.5

    # ---------------- card accounting ----------------
    played = (_R[None, :].astype(jnp.float32) < height[:, None]).astype(jnp.float32)
    total = jnp.broadcast_to(_MULT[None, :], (5, 5))
    rem_pub = total - disc - played                    # unseen by public (deck + both hands)
    partner_count = ph.sum(0)
    rem_me = rem_pub - partner_count                   # unseen by me (deck + my hand)
    playable = (_R[None, :].astype(jnp.float32) == height[:, None])   # [c, r] bool
    missing_mask = (_R[None, :].astype(jnp.float32) >= height[:, None]) & (rem_pub <= 0)
    first_missing = jnp.min(jnp.where(missing_mask, _R[None, :], 5), axis=1)   # [c]
    useless = (_R[None, :].astype(jnp.float32) < height[:, None]) | (_R[None, :] > first_missing[:, None])
    critical = (~useless) & (rem_pub <= 1.0)
    # diagnostics: did the last action lose a still-needed identity for good?
    la_rev = la[28:53].reshape(5, 5)
    la_rank = jnp.argmax(la_rev.sum(0))
    la_color = jnp.argmax(la_rev.sum(1))
    la_type = jnp.argmax(la[2:6])
    removed_any = (la_play | la_discard) & (la_rev.sum() > 0)
    lost_card = removed_any & (~la_success) & (rem_pub[la_color, la_rank] <= 0) & (first_missing[la_color] == la_rank)

    # ---------------- my hand beliefs ----------------
    occ = myV.sum((1, 2)) > 0
    poss = (myV > 0) & (rem_me[None] > 0)
    w = poss.astype(jnp.float32) * jnp.maximum(rem_me, 0.0)[None]
    wsum = w.sum((1, 2))
    p_play = _safe_div((w * playable[None]).sum((1, 2)), wsum)
    certain_play = occ & (wsum > 0) & ((w * (~playable)[None]).sum((1, 2)) == 0)
    p_useless = _safe_div((w * useless[None]).sum((1, 2)), wsum)
    certain_useless = occ & (wsum > 0) & ((w * (~useless)[None]).sum((1, 2)) == 0)
    p_crit = _safe_div((w * critical[None]).sum((1, 2)), wsum)
    e_rank = _safe_div((w * _R[None, None, :]).sum((1, 2)), wsum)
    hinted_any = my_flags.sum(1) > 0

    # ---------------- memory update (every call) ----------------
    focus = mem[0:5]
    prev_flags = mem[5:10]
    n_fail = mem[10]
    n_succ = mem[11]
    last_focus_play = mem[12]
    i_drew_last = mem[13]
    prev_deck = mem[14]
    turn = mem[15]
    fheight = mem[16:41].reshape(5, 5)      # fireworks heights when focus was set
    n_followed = mem[41]
    n_ignored = mem[42]
    my_hint_mask = mem[43]
    # partner's response to my last play hint
    bit_removed = jnp.left_shift(1, la_removed)
    hit = la_actor_partner & la_play & (jnp.bitwise_and(my_hint_mask, bit_removed) > 0)
    miss = la_actor_partner & (la_hint | la_discard) & (my_hint_mask > 0)
    n_followed = n_followed + jnp.where(hit, 1, 0)
    n_ignored = n_ignored + jnp.where(miss, 1, 0)
    my_hint_mask = jnp.where(la_actor_partner, 0, my_hint_mask)

    # my play/discard: shift memory left at removed slot
    removed_self = la_actor_self & (la_play | la_discard)
    shift_idx = jnp.where(_S >= la_removed, _S + 1, _S)
    shift_idx = jnp.minimum(shift_idx, 4)
    focus_shifted = jnp.where(_S == 4, 0, focus[shift_idx])
    prev_shifted = jnp.where(_S == 4, 0, prev_flags[shift_idx])
    fheight_shifted = jnp.where((_S == 4)[:, None], 0, fheight[shift_idx])
    focus = jnp.where(removed_self, focus_shifted, focus)
    prev_flags = jnp.where(removed_self, prev_shifted, prev_flags)
    fheight = jnp.where(removed_self, fheight_shifted, fheight)
    # track outcome of my trust plays
    fp = la_actor_self & la_play & (last_focus_play > 0)
    n_fail = n_fail + jnp.where(fp & (~la_success), 1, 0)
    n_succ = n_succ + jnp.where(fp & la_success, 1, 0)
    last_focus_play = jnp.where(la_actor_self, 0, last_focus_play)

    # partner hint to me: set focus
    hint_me = la_actor_partner & la_hint & la_to_me
    prev_hinted = prev_flags > 0
    touched_me = la_match & occ
    newly = touched_me & (~prev_hinted)
    any_new = jnp.any(newly)
    focus_new = jnp.where(any_new, newly, touched_me) & (p_play > 0)
    # a play hint should not touch new cards that are provably unplayable
    clean_hint = ~jnp.any(jnp.where(any_new, newly, touched_me) & (p_play <= 0))
    focus_new = focus_new & clean_hint
    if CFG_TRUST_ALL_TOUCHED:
        focus_new = touched_me & (p_play > 0)
    if CFG_NEWEST_ONLY:
        newest = jnp.max(jnp.where(touched_me, _S, -1))
        focus_new = touched_me & (_S == newest) & (p_play > 0)
    # save rule: a rank hint touching my chop (oldest unhinted card) that is
    # consistent with a critical, not-yet-playable identity is a save, not a play.
    my_chop = jnp.min(jnp.where(occ & (~prev_hinted), _S, 5))
    my_chop_c = jnp.minimum(my_chop, 4)
    save_worthy = jnp.any(poss & (critical & (~playable))[None], axis=(1, 2))
    la_rank_hint = la[5] > 0.5
    is_save_to_me = hint_me & la_rank_hint & (my_chop < 5) & la_match[my_chop_c] & save_worthy[my_chop_c]
    if CFG_NO_SAVES or CFG_PROB_ONLY:
        is_save_to_me = is_save_to_me & False
    focus_new = jnp.where(is_save_to_me, False, focus_new)
    # after a save, give the partner one turn to save my next chop as well
    just_saved = jnp.where(is_save_to_me, 1, jnp.where(la_actor_self, 0, turn))
    set_focus = hint_me & focus_new
    # newest card among the focus set is the primary candidate (3); others secondary (1)
    newest_focus = jnp.max(jnp.where(set_focus, _S, -1))
    focus = jnp.where(set_focus, jnp.where(_S == newest_focus, 3, 1), focus)
    fheight = jnp.where(set_focus[:, None], height.astype(jnp.int32)[None, :], fheight)
    # saved marker (2): the chop card of a save hint; never discard it voluntarily
    saved_slot = is_save_to_me & (_S == my_chop_c)
    focus = jnp.where(saved_slot, 2, focus)
    # play focus is void if card is provably unplayable now
    focus = jnp.where((p_play <= 0) & (focus != 2), 0, focus)
    focus = jnp.where(occ, focus, 0)
    # trust probability conditioned on the card having been playable when hinted
    playable_at_hint = (_R[None, None, :] == fheight[:, :, None])          # [s, c, r]
    w_focus = w * playable_at_hint
    p_focus = _safe_div((w_focus * playable[None]).sum((1, 2)), w_focus.sum((1, 2)))
    # refresh previous flag bitmask
    cur_flags = jnp.sum(my_flags.astype(jnp.int32) * (2 ** jnp.arange(10))[None, :], axis=1)
    prev_flags = cur_flags
    # deck tracking
    deck_i = deck.astype(jnp.int32)
    became_empty = (prev_deck > 0) & (deck_i == 0)
    i_drew_last = jnp.where(became_empty, jnp.where(la_actor_self, 1, 0), i_drew_last)
    prev_deck = jnp.where(reset, 40, deck_i)
    turn = just_saved

    # ---------------- partner hand & hint evaluation ----------------
    occ_p = ph.sum((1, 2)) > 0
    p_actual_playable = (ph * playable[None]).sum((1, 2)) > 0
    p_actual_useless = (ph * useless[None]).sum((1, 2)) > 0
    p_actual_critical = (ph * critical[None]).sum((1, 2)) > 0
    p_actual_rank = jnp.argmax(ph.sum(1), axis=1)
    pposs = (pV > 0)
    wp_pre = pposs.astype(jnp.float32) * jnp.maximum(rem_pub, 0.0)[None]
    wp_pre_sum = wp_pre.sum((1, 2))
    certain_pre = occ_p & (wp_pre_sum > 0) & ((wp_pre * (~playable)[None]).sum((1, 2)) == 0)
    p_prev_hinted = p_flags.sum(1) > 0
    # partner chop = lowest-index occupied unhinted slot
    chop_cand = occ_p & (~p_prev_hinted)
    chop_idx = jnp.min(jnp.where(chop_cand, _S, 5))
    is_chop = (_S == chop_idx)

    color_touch = ph.sum(2) > 0    # [t, c]
    rank_touch = ph.sum(1) > 0     # [t, r]
    touched = jnp.concatenate([color_touch.T, rank_touch.T], axis=0)  # [10, t]
    eq = (jnp.arange(5)[:, None] == jnp.arange(5)[None, :])          # [h, x]
    cons_color = (eq[:, None, :, None] == touched[0:5][:, :, None, None])   # [h,t,c,r]
    cons_color = jnp.broadcast_to(cons_color, (5, 5, 5, 5))
    cons_rank = (eq[:, None, None, :] == touched[5:10][:, :, None, None])
    cons_rank = jnp.broadcast_to(cons_rank, (5, 5, 5, 5))
    cons = jnp.concatenate([cons_color, cons_rank], axis=0)            # [10,t,c,r]
    post_poss = pposs[None] & cons
    wp = post_poss.astype(jnp.float32) * jnp.maximum(rem_pub, 0.0)[None, None]
    wp_sum = wp.sum((2, 3))
    p_post = _safe_div((wp * playable[None, None]).sum((2, 3)), wp_sum)   # [10, t]
    certain_post = occ_p[None] & (wp_sum > 0) & ((wp * (~playable)[None, None]).sum((2, 3)) == 0)
    n_pre = pposs.sum((1, 2)).astype(jnp.float32)
    n_post = post_poss.sum((2, 3)).astype(jnp.float32)

    touched = touched & occ_p[None]
    newly_h = touched & (~p_prev_hinted)[None]
    any_new_h = jnp.any(newly_h, axis=1)
    focus_h = jnp.where(any_new_h[:, None], newly_h, touched)
    if CFG_TRUST_ALL_TOUCHED:
        focus_h = touched
    if CFG_NEWEST_ONLY:
        newest_h = jnp.max(jnp.where(touched, _S[None, :], -1), axis=1)
        focus_h = touched & (_S[None, :] == newest_h[:, None])

    # save interpretation (mirror of receiver rule): rank hint touching chop with
    # a critical not-yet-playable identity still consistent for the chop card.
    chop_c = jnp.minimum(chop_idx, 4)
    save_worthy_post = jnp.any(post_poss & (critical & (~playable))[None, None], axis=(2, 3))  # [h, t]
    is_rank_h = jnp.arange(10) >= 5
    is_save_h = is_rank_h & (chop_idx < 5) & touched[:, chop_c] & save_worthy_post[:, chop_c]
    if CFG_NO_SAVES or CFG_PROB_ONLY:
        is_save_h = is_save_h & False
    focus_h = jnp.where(is_save_h[:, None], False, focus_h)
    chop_critical = (chop_idx < 5) & p_actual_critical[chop_c]

    # duplicate identities inside a focus set: only the first copy is playable
    p_id = jnp.argmax(ph.reshape(5, 25), axis=1)
    same_id = (p_id[:, None] == p_id[None, :]) & occ_p[:, None] & occ_p[None, :]
    earlier = (_S[None, :] < _S[:, None])                         # [t, t'] t' < t
    dup_focus = jnp.any(focus_h[:, None, :] & same_id[None] & earlier[None], axis=2)  # [h, t]
    dup_touch = jnp.any(touched[:, None, :] & same_id[None] & earlier[None], axis=2)
    eff_playable = p_actual_playable[None] & (~dup_focus)

    certain_good = touched & p_actual_playable[None] & (~dup_touch) & certain_post & (~certain_pre)[None]
    good = focus_h & eff_playable & (~certain_post) & (~certain_pre)[None]
    redundant = focus_h & eff_playable & certain_pre[None]
    bad = focus_h & (~eff_playable) & (p_post > 0)
    safe = focus_h & (~p_actual_playable)[None] & (p_post <= 0)
    info = touched & (~focus_h)
    # naive partners may play any touched card: penalise touching old unplayable cards
    old_bad = touched & (~focus_h) & (~p_actual_playable)[None] & (p_post > 0) & (~is_save_h)[:, None]
    useless_touch = newly_h & p_actual_useless[None]
    crit_touch = newly_h & p_actual_critical[None]
    reduce = jnp.where(touched, 1.0 - _safe_div(n_post, jnp.maximum(n_pre, 1.0)), 0.0)

    hint_score = (
        2.0 * certain_good.sum(1)
        + 1.0 * good.sum(1)
        + 0.05 * redundant.sum(1)
        - 3.0 * bad.sum(1)
        - 0.15 * safe.sum(1)
        + jnp.where(is_save_h, jnp.where(chop_critical, 0.8, -0.3), 0.0)
        + jnp.where(is_save_h, 0.5 * jnp.maximum(crit_touch.sum(1) - 1, 0), 0.0)
        + 0.05 * info.sum(1)
        + 0.1 * reduce.sum(1)
        - 0.3 * useless_touch.sum(1)
        - 0.4 * old_bad.sum(1)
    )
    if CFG_PROB_ONLY:
        # information-only hinting: value = certain plays created + possibility reduction
        hint_score = (1.5 * certain_good.sum(1) + 0.4 * reduce.sum(1)
                      + 0.3 * (touched & p_actual_playable[None]).sum(1)
                      - 0.3 * useless_touch.sum(1))
    hint_legal = legal[10:20]
    hint_score = jnp.where(hint_legal, hint_score, -1e6)
    best_hint = jnp.argmax(hint_score)
    hint_val = hint_score[best_hint]
    any_hint = jnp.any(hint_legal)
    partner_has_certain = jnp.any(certain_pre)

    # ---------------- play decision ----------------
    base_thr = jnp.where(lives >= 3, 0.25, jnp.where(lives >= 2, 0.35, 1.01))
    thr = jnp.maximum(base_thr + 0.4 * n_fail.astype(jnp.float32) - 0.08 * n_succ.astype(jnp.float32), 0.1)
    thr = jnp.where(lives <= 1, 1.01, thr)
    focus_b = (focus == 1) | (focus == 3)
    thr_slot = jnp.where(focus == 3, thr, thr + 0.3)
    p_trust = p_focus
    if CFG_PROB_ONLY:
        thr_slot = jnp.where(lives >= 2, 0.75, 1.01) + 0.0 * thr_slot
        focus_b = occ
        p_trust = p_play
    saved_b = focus == 2
    # defer a certain play when every possible identity is a copy the partner
    # holds as a hinted, plausibly playable card (partner will likely play it)
    partner_likely_play = occ_p & p_prev_hinted & p_actual_playable & (
        _safe_div((wp_pre * playable[None]).sum((1, 2)), wp_pre_sum) > 0)
    covered = jnp.any(ph * partner_likely_play[:, None, None].astype(jnp.float32) > 0, axis=0)   # [c, r]
    dup_defer = certain_play & (~jnp.any(poss & (~covered)[None], axis=(1, 2))) & (deck_i > 0)
    # partners that do not act on my play hints probably do not give play hints
    distrust = n_ignored > n_followed + 1
    # required unconditional playability for a trust play: rises with failed
    # trust plays, distrust evidence and low lives; falls with successes
    p_req = (0.45 * n_fail.astype(jnp.float32) - 0.1 * n_succ.astype(jnp.float32)
             + jnp.where(distrust, 0.6, 0.0) + jnp.where(lives <= 2, 0.25, 0.0))
    p_req = jnp.clip(p_req, 0.0, 1.0)
    p_req = jnp.where(lives <= 1, 1.01, p_req)
    p_req_slot = jnp.where(focus == 3, p_req, p_req + 0.3)
    if CFG_PROB_ONLY:
        p_req_slot = thr_slot
    play_score = jnp.where(
        certain_play & (~dup_defer), 2.0 + 0.01 * (5.0 - e_rank),
        jnp.where(focus_b & (p_trust >= 0.5) & (p_play > 0) & (p_play >= p_req_slot) & occ,
                  1.0 + p_trust + 0.05 * (focus == 3), -1.0))
    best_play = jnp.argmax(play_score)
    can_play = play_score[best_play] > 0
    play_is_focus = ~certain_play[best_play]

    # ---------------- discard decision ----------------
    # deterministic chop: oldest unhinted card; hinted cards ordered by value
    dscore = (
        jnp.where(certain_useless, 10.0, 0.0)
        + jnp.where(hinted_any, 2.0 * p_useless - 3.0 * p_crit - 0.1 * _S.astype(jnp.float32),
                    3.0 - 0.5 * _S.astype(jnp.float32))
        - 2.0 * focus_b
        - 5.0 * saved_b
    )
    dscore = jnp.where(legal[0:5], dscore, -1e6)
    best_discard = jnp.argmax(dscore)
    any_discard = jnp.any(legal[0:5])
    has_useless = jnp.any(certain_useless & legal[0:5])

    # ---------------- ordinary turn ----------------
    # measured: hinting any positively-scored (>=0.5) hint beats token hoarding
    hint_thr = 0.5 + 0.0 * tokens
    do_play_hint = any_hint & (hint_val >= hint_thr)
    do_save_hint = any_hint & (hint_val >= 0.5) & is_save_h[best_hint]
    # filler hints must not have a provably-unplayable focus (naive partners may play it)
    do_info_hint = any_hint & (((tokens >= 7) & (hint_val >= 0.0)) | ((tokens >= 5) & (hint_val >= 0.5)))
    # after being saved, avoid discarding for one turn (partner may need to save my next chop)
    do_info_hint = do_info_hint | (any_hint & (just_saved > 0) & (tokens >= 2) & (hint_val >= -0.5))

    # fallback when nothing else: risky best play
    risky_play = jnp.argmax(p_play)
    act_hint = 10 + best_hint
    act_play = 5 + best_play
    act_discard = best_discard
    act_risky = 5 + risky_play

    # urgent save: partner's chop is critical and partner has nothing certain to play
    save_act = 15 + p_actual_rank[chop_c]
    save_legal = legal[jnp.minimum(save_act, 19)]
    save_is_save = is_save_h[jnp.minimum(p_actual_rank[chop_c] + 5, 9)]
    urgent_save = chop_critical & (~partner_has_certain) & save_legal & save_is_save & (deck_i > 0)

    action = jnp.where(any_discard, act_discard, jnp.where(any_hint, act_hint, act_risky))
    action = jnp.where(do_info_hint & any_discard, act_hint, action)
    action = jnp.where(has_useless, act_discard, action)
    action = jnp.where(do_save_hint, act_hint, action)
    action = jnp.where(do_play_hint, act_hint, action)
    # near the end of the deck, a discard shortens the game; hint instead while tokens last
    near_end = (deck_i <= 3) & (deck_i > 0) & any_hint & (hint_val > -1.0)
    action = jnp.where(near_end, act_hint, action)
    action = jnp.where(can_play, act_play, action)
    action = jnp.where(urgent_save, save_act, action)

    # ---------------- last turn (deck empty) ----------------
    partner_has_turn = i_drew_last > 0
    best_p = p_play[risky_play]
    last_action = jnp.where(any_hint, act_hint, jnp.where(any_discard, act_discard, act_risky))
    gamble_min = jnp.where(partner_has_turn & (lives <= 2), 0.3, 0.0)
    last_action = jnp.where((lives >= 2) & (best_p > gamble_min), act_risky, last_action)
    last_action = jnp.where(partner_has_turn & any_hint & (hint_val >= 0.9) & (best_p < 0.5), act_hint, last_action)
    last_action = jnp.where(jnp.any(certain_play), act_play, last_action)
    is_last = deck_i == 0
    action = jnp.where(is_last, last_action, action)

    # illegal guard: fall back to first legal action
    action = action.astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    active = ~legal[20]
    action = jnp.where(active, action, jnp.asarray(20, dtype=jnp.int32))

    # record whether this play is a trust play
    did_play = active & (action >= 5) & (action <= 9)
    played_slot = jnp.clip(action - 5, 0, 4)
    last_focus_play = jnp.where(did_play & (~certain_play[played_slot]), 1, last_focus_play)

    # remember which partner slots my hint (if any) marked as play candidates
    did_hint = active & (action >= 10) & (action <= 19)
    h_chosen = jnp.clip(action - 10, 0, 9)
    play_hint_now = did_hint & (hint_score[h_chosen] >= 0.9) & (~is_save_h[h_chosen])
    focus_bits = jnp.sum(focus_h[h_chosen].astype(jnp.int32) * (2 ** _S))
    my_hint_mask = jnp.where(play_hint_now, focus_bits, my_hint_mask)

    new_mem = jnp.concatenate([
        focus.astype(jnp.int32),
        prev_flags.astype(jnp.int32),
        jnp.stack([n_fail, n_succ, last_focus_play, i_drew_last, prev_deck, turn]).astype(jnp.int32),
        fheight.reshape(25).astype(jnp.int32),
        jnp.stack([n_followed, n_ignored, my_hint_mask]).astype(jnp.int32),
    ])

    trace = {
        'p_play': p_play,
        'certain_play': certain_play,
        'focus': focus.astype(jnp.int32),
        'hint_val': hint_val,
        'best_hint': best_hint.astype(jnp.int32),
        'tokens': tokens,
        'lives': lives,
        'deck': deck,
        'n_fail': n_fail,
        'max_score': jnp.sum(first_missing).astype(jnp.int32),
        'score': jnp.sum(height).astype(jnp.int32),
        'crit_in_partner': jnp.sum(p_actual_critical).astype(jnp.int32),
        'lost': lost_card.astype(jnp.int32),
        'lost_rank': la_rank.astype(jnp.int32),
        'la_type': la_type.astype(jnp.int32),
        'la_actor_self': la_actor_self.astype(jnp.int32),
        'la_removed': la_removed.astype(jnp.int32),
        'chop_idx': chop_idx.astype(jnp.int32),
        'chop_critical': chop_critical.astype(jnp.int32),
        'partner_has_certain': partner_has_certain.astype(jnp.int32),
        'urgent_save': urgent_save.astype(jnp.int32),
        'save_is_save': save_is_save.astype(jnp.int32),
        'action': action.astype(jnp.int32),
        'n_followed': n_followed.astype(jnp.int32),
        'n_ignored': n_ignored.astype(jnp.int32),
        'my_hint_mask': my_hint_mask.astype(jnp.int32),
        'n_succ': n_succ.astype(jnp.int32),
    }
    return action, new_mem, trace
