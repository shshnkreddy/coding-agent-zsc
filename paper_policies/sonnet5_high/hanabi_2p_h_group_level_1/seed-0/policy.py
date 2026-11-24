"""H-group Level 1 style Hanabi policy.

Stateless per-turn heuristic: every call decodes the full public/native
observation into board state (fireworks, discards, hint history, partner
identities), scores all 21 actions according to an H-group Level 1 priority
ladder, and returns the legal action with the highest score. No private
memory is required because the native knowledge block (V0 + positive hint
flags) already persists and shifts hint history for us, and per-call scoring
avoids the risk of buggy incremental state tracking.

Own-hand elimination combines the native V0 feature with two extra,
purely-observational refinements available to any policy regardless of the
partner's implementation: (a) ruling out (color, rank) candidates that are
fully accounted for in partner's directly-observed hand, and (b) trusting a
just-received clue's focus (per the documented single-card-focus / chop-
focus rules) as an immediate play whenever it lands on a new, non-chop card,
since a non-chop focus can only ever be a Play Clue by the Save Clue rules.
Chop-focus clues remain ambiguous with a Save Clue and are only accepted via
plain V0 elimination. A retouch-only ("Tempo Clue") focus is never blindly
trusted on receipt (that would depend on an assumption about how an
independently-implemented partner chooses to give one), but verified Tempo
Clues are still offered when giving, since they only ever help a partner
whose own V0 elimination would otherwise take longer to reach the same
conclusion.

Priority ladder (highest first):
  1. Play a card we can prove is currently playable via V0 elimination.
  2. Give the (unique) Save Clue needed for partner's chop (5, unique 2,
     or critical card), per Save Principle / 2-Save / 5-Save.
  3. Give a Good-Touch, Minimum-Clue-Value Play Clue (including a verified
     Tempo Clue) whose focus lands on a currently playable partner card.
  4. Discard a known-useless (trash) own card.
  5. Discard our own chop.
  6. As a last resort (e.g. 8 info tokens, nothing better legal), give any
     legal clue (stall), strongly preferring one that only re-touches
     already-clued cards so it cannot masquerade as a new play promise.
Ties within a tier are broken by a small deterministic index bonus.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hgroup_level1_heuristic',
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

NUM_COLORS = 5
NUM_RANKS = 5
NUM_SLOTS = 5
EPS = 1e-6

T_PLAY_KNOWN = 100.0
T_SAVE = 90.0
T_PLAY_CLUE = 80.0
T_DISCARD_TRASH = 50.0
T_DISCARD_CHOP = 40.0
T_DISCARD_OTHER = 10.0
T_STALL_CLUE = 5.0
T_STALL_RISKY = -50.0


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _decode(features):
    f = features
    mult = jnp.array([3.0, 2.0, 2.0, 2.0, 1.0], dtype=jnp.float32)

    partner_hand = f[0:125].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    occupied_partner = jnp.sum(partner_hand.reshape(NUM_SLOTS, 25), axis=1) > 0.5
    partner_color = jnp.argmax(jnp.sum(partner_hand, axis=2), axis=1)
    partner_rank = jnp.argmax(jnp.sum(partner_hand, axis=1), axis=1)

    firework = f[167:192].reshape(NUM_COLORS, NUM_RANKS)
    rank_weight = (jnp.arange(NUM_RANKS, dtype=jnp.float32) + 1.0)
    height_f = jnp.sum(firework * rank_weight[None, :], axis=1)
    height_i = jnp.round(height_f).astype(jnp.int32)

    discards = f[203:253].reshape(NUM_COLORS, 10)
    d0 = jnp.sum(discards[:, 0:3], axis=1)
    d1 = jnp.sum(discards[:, 3:5], axis=1)
    d2 = jnp.sum(discards[:, 5:7], axis=1)
    d3 = jnp.sum(discards[:, 7:9], axis=1)
    d4 = discards[:, 9]
    discarded_count = jnp.stack([d0, d1, d2, d3, d4], axis=1)

    ranks_i = jnp.arange(NUM_RANKS, dtype=jnp.int32)
    already_played = ranks_i[None, :] < height_i[:, None]
    remaining_count = mult[None, :] - discarded_count - jnp.where(already_played, 1.0, 0.0)
    partner_visible_count = jnp.sum(partner_hand, axis=0)
    personal_remaining = remaining_count - partner_visible_count

    in_range = (ranks_i[None, None, :] >= height_i[:, None, None]) & \
        (ranks_i[None, None, :] <= ranks_i[None, :, None])
    zero_cm = remaining_count <= 0.5
    gap_exists = jnp.any(in_range & zero_cm[:, None, :], axis=-1)
    dead = (~already_played) & gap_exists
    trash_mask = already_played | dead
    playable_now_mat = ranks_i[None, :] == height_i[:, None]
    alive_future = ~trash_mask

    tokens = jnp.sum(f[192:200])
    tokens_available = tokens > 0.5
    deck_size = jnp.sum(f[127:167])
    lives = jnp.sum(f[200:203])

    nk = f[308:658].reshape(2, NUM_SLOTS, 35)
    self_block = nk[0]
    partner_block = nk[1]

    self_V0 = self_block[:, 0:25].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    self_color_hint = self_block[:, 25:30]
    self_rank_hint = self_block[:, 30:35]
    occupied_self = jnp.sum(self_V0.reshape(NUM_SLOTS, 25), axis=1) > 0.5
    clued_self = (jnp.sum(self_color_hint, axis=1) > 0.5) | (jnp.sum(self_rank_hint, axis=1) > 0.5)

    partner_color_hint = partner_block[:, 25:30]
    partner_rank_hint = partner_block[:, 30:35]
    clued_partner = (jnp.sum(partner_color_hint, axis=1) > 0.5) | (jnp.sum(partner_rank_hint, axis=1) > 0.5)

    la = f[253:308]
    la_type = la[2:6]
    la_recipient = la[6:8]
    la_named_rank = la[13:18]
    la_match = la[18:23]

    hint_to_me = (la_recipient[0] > 0.5) & ((la_type[2] > 0.5) | (la_type[3] > 0.5))
    is_color_hint = la_type[2] > 0.5
    is_rank_hint = la_type[3] > 0.5
    matched = la_match > 0.5

    any_prior_rank_hint = jnp.sum(self_rank_hint, axis=1) > 0.5
    any_prior_color_hint = jnp.sum(self_color_hint, axis=1) > 0.5
    new_approx = matched & jnp.where(
        is_color_hint, ~any_prior_rank_hint,
        jnp.where(is_rank_hint, ~any_prior_color_hint, False))
    pre_clued = clued_self & (~new_approx)

    idx5 = jnp.arange(NUM_SLOTS, dtype=jnp.int32)
    pre_chop_idx = jnp.min(jnp.where(occupied_self & (~pre_clued), idx5, NUM_SLOTS))
    pre_chop_valid = pre_chop_idx < NUM_SLOTS
    pre_chop_clip = jnp.clip(pre_chop_idx, 0, NUM_SLOTS - 1)

    n_new = jnp.sum(jnp.where(new_approx, 1, 0))
    chop_touched_pre = pre_chop_valid & matched[pre_chop_clip]
    focus_new_default = jnp.max(jnp.where(new_approx, idx5, -1))
    focus_retouch_default = jnp.max(jnp.where(matched, idx5, -1))
    focus_if_new = jnp.where(chop_touched_pre, pre_chop_idx, focus_new_default)
    focus = jnp.where(n_new > 0, focus_if_new, focus_retouch_default)
    focus_valid = hint_to_me & (focus >= 0)
    focus_clip = jnp.clip(focus, 0, NUM_SLOTS - 1)
    focus_is_chop = focus_valid & (n_new > 0) & chop_touched_pre

    # A clue focused on a non-chop card must be a Play Clue (Save Clues are
    # only ever legal on chop), so it is always safe to trust it as an
    # immediate play. A clue focused on chop is ambiguous with a Save Clue
    # only through the same elimination the docs use: a number-2/5 clue on
    # chop is always the dedicated 2-Save/5-Save move (never a coincidental
    # play), so never trust those; a Critical Save can only apply to a
    # human rank 1/3/4 card (2's and 5's have their own dedicated,
    # number-only save moves, so a candidate at rank id 1 or 4 within a
    # color clue's support can only mean "Play Clue"), so we only withhold
    # trust when this clue's own V0 support still contains a possible
    # critical (last-copy) candidate at rank id 0, 2, or 3.
    named_rank_25 = is_rank_hint & ((la_named_rank[1] > 0.5) | (la_named_rank[4] > 0.5))
    save_relevant_rank = jnp.array([True, False, True, True, False])
    support_focus = (self_V0[focus_clip] > EPS) & (personal_remaining > 0.5)
    relevant_critical = support_focus & save_relevant_rank[None, :] & (remaining_count <= 1.5)
    any_critical_focus = jnp.any(relevant_critical)
    chop_pure_save = focus_is_chop & (named_rank_25 | any_critical_focus)

    # Our own giving convention validates a retouch-only ("Tempo") clue the
    # same way as an ordinary play clue: it is only ever chosen when the
    # resolved focus is genuinely playable now. Chop can never be the
    # retouch-branch focus (chop is unclued by definition), so this cannot
    # be confused with a Save Clue. A meaningless forced stall is still
    # possible when no verified option exists at all, but giving always
    # prefers a verified Tempo Clue over an arbitrary one when available.
    trust_play_focus = focus_valid & (~chop_pure_save) & (n_new > 0)
    trusted_playable = jnp.where(idx5 == focus_clip, trust_play_focus, False)

    dbg_hint_to_me = hint_to_me
    dbg_focus = focus
    dbg_n_new = n_new
    dbg_trust = trust_play_focus

    return dict(
        occupied_partner=occupied_partner,
        partner_color=partner_color,
        partner_rank=partner_rank,
        height_i=height_i,
        remaining_count=remaining_count,
        personal_remaining=personal_remaining,
        trash_mask=trash_mask,
        alive_future=alive_future,
        playable_now_mat=playable_now_mat,
        tokens_available=tokens_available,
        self_V0=self_V0,
        occupied_self=occupied_self,
        clued_self=clued_self,
        clued_partner=clued_partner,
        trusted_playable=trusted_playable,
        tokens=tokens,
        deck_size=deck_size,
        lives=lives,
        dbg_hint_to_me=dbg_hint_to_me,
        dbg_focus=dbg_focus,
        dbg_n_new=dbg_n_new,
        dbg_trust=dbg_trust,
    )


def _score_actions(state):
    idx5 = jnp.arange(NUM_SLOTS, dtype=jnp.int32)

    occupied_self = state['occupied_self']
    clued_self = state['clued_self']
    self_V0 = state['self_V0']
    playable_now_mat = state['playable_now_mat']
    trash_mask = state['trash_mask']

    # The native V0 feature does not subtract copies visible in partner's
    # hand (per spec). Since partner's hand is directly observable to us,
    # rule out any (color, rank) candidate that is fully accounted for
    # there (personal_remaining<=0): we cannot be holding a card if every
    # remaining copy of it is visible in partner's hand.
    personal_remaining = state['personal_remaining']
    supp = (self_V0 > EPS) & (personal_remaining[None, :, :] > 0.5)
    supp_flat = supp.reshape(NUM_SLOTS, 25)
    any_supp = jnp.any(supp_flat, axis=1)

    playable_ok = jnp.where(supp, playable_now_mat[None, :, :], True)
    known_playable_self = occupied_self & any_supp & jnp.all(playable_ok.reshape(NUM_SLOTS, 25), axis=1)
    known_playable_self = known_playable_self | (occupied_self & state['trusted_playable'])

    trash_ok = jnp.where(supp, trash_mask[None, :, :], True)
    known_trash_self = occupied_self & any_supp & jnp.all(trash_ok.reshape(NUM_SLOTS, 25), axis=1)

    chop_self_idx = jnp.min(jnp.where(occupied_self & (~clued_self), idx5, NUM_SLOTS))

    play_score = jnp.where(
        known_playable_self, T_PLAY_KNOWN + (NUM_SLOTS - 1 - idx5) * 0.001, -jnp.inf)

    discard_base = jnp.where(
        known_trash_self, T_DISCARD_TRASH,
        jnp.where(idx5 == chop_self_idx, T_DISCARD_CHOP, T_DISCARD_OTHER))
    discard_score = jnp.where(occupied_self, discard_base, -jnp.inf)

    occupied_partner = state['occupied_partner']
    partner_color = state['partner_color']
    partner_rank = state['partner_rank']
    height_i = state['height_i']
    remaining_count = state['remaining_count']
    alive_future = state['alive_future']
    clued_partner = state['clued_partner']
    tokens_available = state['tokens_available']

    partner_playable_now = occupied_partner & (partner_rank == height_i[partner_color])
    alive_partner = alive_future[partner_color, partner_rank]

    chop_partner_idx = jnp.min(jnp.where(occupied_partner & (~clued_partner), idx5, NUM_SLOTS))
    valid_chop = chop_partner_idx < NUM_SLOTS
    chop_clip = jnp.clip(chop_partner_idx, 0, NUM_SLOTS - 1)

    c0 = partner_color[chop_clip]
    r0 = partner_rank[chop_clip]
    chop_playable_now = partner_playable_now[chop_clip]

    is5 = valid_chop & (r0 == 4)
    is2 = valid_chop & (r0 == 1)
    same_2 = occupied_partner & (partner_color == c0) & (partner_rank == 1)
    count_same_2 = jnp.sum(jnp.where(same_2, 1, 0)) - jnp.where(is2, 1, 0)
    unique2_ok = is2 & (count_same_2 <= 0)

    crit_val = remaining_count[c0, r0]
    critical = valid_chop & (crit_val <= 1.5) & alive_future[c0, r0] & (~chop_playable_now)

    needs_save = valid_chop & occupied_partner[chop_clip] & (~clued_partner[chop_clip]) & \
        (~chop_playable_now) & (is5 | unique2_ok | critical)
    save_uses_rank = is5 | unique2_ok

    save_color_scores = jnp.where(
        idx5 == c0, jnp.where(needs_save & (~save_uses_rank), T_SAVE, -jnp.inf), -jnp.inf)
    save_rank_scores = jnp.where(
        idx5 == r0, jnp.where(needs_save & save_uses_rank, T_SAVE, -jnp.inf), -jnp.inf)

    def eval_color(a):
        touched = occupied_partner & (partner_color == a)
        new = touched & (~clued_partner)
        n_new = jnp.sum(jnp.where(new, 1, 0))
        any_touched = jnp.sum(jnp.where(touched, 1, 0)) > 0
        chop_in_new = valid_chop & new[chop_clip]
        focus_new_default = jnp.max(jnp.where(new, idx5, -1))
        focus_retouch_default = jnp.max(jnp.where(touched, idx5, -1))
        focus_if_new = jnp.where(chop_in_new, chop_partner_idx, focus_new_default)
        focus = jnp.where(n_new > 0, focus_if_new, focus_retouch_default)
        focus_valid = focus >= 0
        focus_clip = jnp.clip(focus, 0, NUM_SLOTS - 1)
        focus_playable = partner_playable_now[focus_clip]
        good_touch = jnp.all(jnp.where(new & (idx5 != focus_clip), alive_partner, True))
        legal = tokens_available & any_touched & focus_valid
        valid = legal & focus_playable & good_touch
        is_tempo = n_new == 0
        bonus = jnp.where(chop_in_new, 1.0, 0.0) + jnp.where(is_tempo, -0.5, 0.0)
        return jnp.where(valid, T_PLAY_CLUE + bonus, -jnp.inf)

    def eval_rank(a):
        touched = occupied_partner & (partner_rank == a)
        new = touched & (~clued_partner)
        n_new = jnp.sum(jnp.where(new, 1, 0))
        any_touched = jnp.sum(jnp.where(touched, 1, 0)) > 0
        chop_in_new = valid_chop & new[chop_clip]
        focus_new_default = jnp.max(jnp.where(new, idx5, -1))
        focus_retouch_default = jnp.max(jnp.where(touched, idx5, -1))
        focus_if_new = jnp.where(chop_in_new, chop_partner_idx, focus_new_default)
        focus = jnp.where(n_new > 0, focus_if_new, focus_retouch_default)
        focus_valid = focus >= 0
        focus_clip = jnp.clip(focus, 0, NUM_SLOTS - 1)
        focus_playable = partner_playable_now[focus_clip]
        good_touch = jnp.all(jnp.where(new & (idx5 != focus_clip), alive_partner, True))
        legal = tokens_available & any_touched & focus_valid
        valid = legal & focus_playable & good_touch
        is_tempo = n_new == 0
        bonus = jnp.where(chop_in_new, 1.0, 0.0) + jnp.where(is_tempo, -0.5, 0.0)
        return jnp.where(valid, T_PLAY_CLUE + bonus, -jnp.inf)

    color_clue_scores = jax.vmap(eval_color)(jnp.arange(NUM_COLORS, dtype=jnp.int32))
    rank_clue_scores = jax.vmap(eval_rank)(jnp.arange(NUM_RANKS, dtype=jnp.int32))
    dbg_best_color = jnp.max(color_clue_scores)
    dbg_best_rank = jnp.max(rank_clue_scores)
    dbg_any_target = jnp.any(occupied_partner & (~clued_partner) & partner_playable_now)

    def touch_stats_color(a):
        touched = occupied_partner & (partner_color == a)
        new = touched & (~clued_partner)
        return jnp.sum(jnp.where(touched, 1, 0)) > 0, jnp.sum(jnp.where(new, 1, 0))

    def touch_stats_rank(a):
        touched = occupied_partner & (partner_rank == a)
        new = touched & (~clued_partner)
        return jnp.sum(jnp.where(touched, 1, 0)) > 0, jnp.sum(jnp.where(new, 1, 0))

    color_any_touched, color_n_new = jax.vmap(touch_stats_color)(jnp.arange(NUM_COLORS, dtype=jnp.int32))
    rank_any_touched, rank_n_new = jax.vmap(touch_stats_rank)(jnp.arange(NUM_RANKS, dtype=jnp.int32))
    # A forced stall clue (nothing else legal/valid) carries no real
    # playability promise. Strongly prefer a pure re-touch (no new card)
    # so the receiver's focus/new-card tracking is never misled into
    # trusting an arbitrary stall as an immediate play; only fall back to
    # touching a new card if literally no pure re-touch value is legal.
    color_stall = jnp.where(
        color_any_touched & (color_n_new == 0), T_STALL_CLUE,
        jnp.where(color_any_touched, T_STALL_RISKY - color_n_new.astype(jnp.float32), -jnp.inf))
    rank_stall = jnp.where(
        rank_any_touched & (rank_n_new == 0), T_STALL_CLUE,
        jnp.where(rank_any_touched, T_STALL_RISKY - rank_n_new.astype(jnp.float32), -jnp.inf))

    color_final = jnp.maximum(jnp.maximum(color_clue_scores, save_color_scores), color_stall)
    rank_final = jnp.maximum(jnp.maximum(rank_clue_scores, save_rank_scores), rank_stall)

    noop_score = jnp.zeros((1,), dtype=jnp.float32)

    full_scores = jnp.concatenate(
        [discard_score, play_score, color_final, rank_final, noop_score], axis=0)
    debug = dict(
        known_playable_count=jnp.sum(jnp.where(known_playable_self, 1, 0)).astype(jnp.int32),
        trusted_playable_any=jnp.any(state['trusted_playable']),
        needs_save=needs_save,
        chop_self_idx=chop_self_idx.astype(jnp.int32),
        chop_partner_idx=chop_partner_idx.astype(jnp.int32),
        dbg_best_color=dbg_best_color,
        dbg_best_rank=dbg_best_rank,
        dbg_any_target=dbg_any_target,
        tokens=state['tokens'].astype(jnp.int32),
        deck_size=state['deck_size'].astype(jnp.int32),
        lives=state['lives'].astype(jnp.int32),
        dbg_hint_to_me=state['dbg_hint_to_me'],
        dbg_focus=state['dbg_focus'],
        dbg_n_new=state['dbg_n_new'],
        dbg_trust=state['dbg_trust'],
    )
    return full_scores, debug


def agent_act(obs, reset, memory, rng):
    del rng
    state = _decode(obs.features)
    scores, debug = _score_actions(state)
    masked = jnp.where(obs.legal_actions, scores, -jnp.inf)
    action = jnp.argmax(masked).astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    trace = {
        'known_playable_count': debug['known_playable_count'],
        'trusted_playable_any': debug['trusted_playable_any'],
        'needs_save': debug['needs_save'],
        'chop_self_idx': debug['chop_self_idx'],
        'chop_partner_idx': debug['chop_partner_idx'],
        'tokens': debug['tokens'],
        'deck_size': debug['deck_size'],
        'lives': debug['lives'],
        'dbg_hint_to_me': debug['dbg_hint_to_me'],
        'dbg_focus': debug['dbg_focus'],
        'dbg_n_new': debug['dbg_n_new'],
        'dbg_trust': debug['dbg_trust'],
        'dbg_best_color': debug['dbg_best_color'],
        'dbg_best_rank': debug['dbg_best_rank'],
        'dbg_any_target': debug['dbg_any_target'],
    }
    return action, jnp.where(reset, zero, memory), trace
