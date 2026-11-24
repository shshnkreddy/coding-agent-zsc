"""H-group Level 1 rule-based Hanabi policy.

Single-turn, fully vectorized (fixed-shape) heuristic that scores all 21
actions using the documented H-group Level 1 conventions: chop
identification, the two clue types (Play / Save), single-card focus,
Good Touch / Save / Minimum-Clue-Value principles, and the 5-save / 2-save /
critical-save rules. Own-hand "certain play" decisions come from a
good-touch-trust-filtered elimination over the native V0 possibility
distribution plus current firework heights; no probabilistic hidden-state
simulation of the partner is used. A tiny amount of private memory tracks
the implied target rank of a color-clue Play Clue on a non-chop card (a
"Delayed Play Clue" promise) and a snapshot of our own previous clued-state
needed to compute clue focus without re-deriving turn history.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hgroup_level1_rule_agent',
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

NUM_SLOTS = 5
NUM_COLORS = 5
NUM_RANKS = 5
MULT_LIST = [3.0, 2.0, 2.0, 2.0, 1.0]
DISCARD_OFFSETS = [0, 3, 5, 7, 9, 10]

NEG = -1.0e9


def init_agent():
    return {
        'target_rank': jnp.zeros(NUM_SLOTS, dtype=jnp.int32),
        'target_valid': jnp.zeros(NUM_SLOTS, dtype=jnp.bool_),
        'prev_own_clued': jnp.zeros(NUM_SLOTS, dtype=jnp.bool_),
    }


def _chop_index(occupied, clued):
    """Lowest occupied native index among unclued cards; 5 if none."""
    unclued_occupied = occupied & (~clued)
    idx = jnp.arange(NUM_SLOTS)
    candidate = jnp.where(unclued_occupied, idx, NUM_SLOTS)
    return jnp.min(candidate)


def _highest_index(mask):
    idx = jnp.arange(NUM_SLOTS)
    candidate = jnp.where(mask, idx, -1)
    return jnp.max(candidate)


def _discard_counts(discards):
    # discards: [5 colors, 10] thermometer subblocks per rank.
    cols = []
    for r in range(NUM_RANKS):
        lo, hi = DISCARD_OFFSETS[r], DISCARD_OFFSETS[r + 1]
        cols.append(jnp.sum(discards[:, lo:hi], axis=1))
    return jnp.stack(cols, axis=1)  # [5 colors, 5 ranks]


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    mult = jnp.asarray(MULT_LIST, dtype=jnp.float32)

    # Sanitize input memory against episode boundaries up front: the input
    # `memory` may still hold the previous episode's carried-over values, and
    # the last-action block is all-zero at reset so none of the update
    # branches below would otherwise overwrite stale entries.
    memory = {
        'target_rank': jnp.where(reset, jnp.zeros(NUM_SLOTS, dtype=jnp.int32), memory['target_rank']),
        'target_valid': jnp.where(reset, jnp.zeros(NUM_SLOTS, dtype=jnp.bool_), memory['target_valid']),
        'prev_own_clued': jnp.where(reset, jnp.zeros(NUM_SLOTS, dtype=jnp.bool_), memory['prev_own_clued']),
    }

    partner_hand = f[0:125].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    fireworks_oh = f[167:192].reshape(NUM_COLORS, NUM_RANKS)
    info_tokens = jnp.sum(f[192:200])
    discards = f[203:253].reshape(NUM_COLORS, 10)
    own_know = f[308:483].reshape(NUM_SLOTS, 35)
    partner_know = f[483:658].reshape(NUM_SLOTS, 35)

    own_V0 = own_know[:, 0:25].reshape(NUM_SLOTS, NUM_COLORS, NUM_RANKS)
    own_color_flags = own_know[:, 25:30]
    own_rank_flags = own_know[:, 30:35]
    partner_color_flags = partner_know[:, 25:30]
    partner_rank_flags = partner_know[:, 30:35]

    own_occupied = jnp.sum(own_V0, axis=(1, 2)) > 0.0
    own_clued = (jnp.sum(own_color_flags, axis=1) + jnp.sum(own_rank_flags, axis=1)) > 0.0

    partner_occupied = jnp.sum(partner_hand, axis=(1, 2)) > 0.0
    partner_clued = (jnp.sum(partner_color_flags, axis=1) + jnp.sum(partner_rank_flags, axis=1)) > 0.0
    partner_true_color = jnp.argmax(jnp.sum(partner_hand, axis=2), axis=1)
    partner_true_rank = jnp.argmax(jnp.sum(partner_hand, axis=1), axis=1)

    heights = jnp.sum(fireworks_oh * (jnp.arange(NUM_RANKS) + 1)[None, :], axis=1)  # [5] float
    next_needed = heights  # next needed native rank id per color

    discard_count = _discard_counts(discards)  # [color, rank]

    own_chop = _chop_index(own_occupied, own_clued)
    partner_chop = _chop_index(partner_occupied, partner_clued)

    # ---- last-action decoding (absolute offsets, block starts at 253) ----
    acted_by_self = f[253] > 0.5
    acted_by_partner = f[254] > 0.5
    act_is_play = f[255] > 0.5
    act_is_discard = f[256] > 0.5
    act_is_color_hint = f[257] > 0.5
    act_is_rank_hint = f[258] > 0.5
    recipient_is_self = f[259] > 0.5
    hinted_color_idx = jnp.argmax(f[261:266])
    match_flags = f[271:276] > 0.5
    removed_slot = jnp.argmax(f[276:281])

    # ---- (1) interpret a clue given to us this transition ----
    is_clue_to_me = acted_by_partner & recipient_is_self & (act_is_color_hint | act_is_rank_hint)

    prev_clued = memory['prev_own_clued']
    chop_before = _chop_index(own_occupied, prev_clued)
    newly_touched_self = match_flags & (~prev_clued)
    any_new_self = jnp.any(newly_touched_self)
    chop_before_safe = jnp.clip(chop_before, 0, NUM_SLOTS - 1)
    chop_touched_self = (chop_before < NUM_SLOTS) & jnp.take(match_flags, chop_before_safe)

    focus_retouch = _highest_index(match_flags)
    focus_new_leftmost = _highest_index(newly_touched_self)
    focus_self = jnp.where(
        ~any_new_self, focus_retouch,
        jnp.where(chop_touched_self, chop_before, focus_new_leftmost))
    focus_self = jnp.clip(focus_self, 0, NUM_SLOTS - 1)

    is_chop_focus_self = (chop_before < NUM_SLOTS) & (focus_self == chop_before)
    non_chop_focus_self = is_clue_to_me & any_new_self & (~is_chop_focus_self)

    # A chop-focus color clue can only be an ambiguous Critical Save if a
    # critical rank (2/3/4, i.e. human 3/4 or an already-reduced other rank)
    # is actually possible for that color; 5-saves/2-saves always use rank
    # clues by convention, so a color clue never signals those. When no
    # critical interpretation is possible, a chop color clue is unambiguously
    # a Play Clue (see docs/beginner/single-card-focus.md ChopFocus example).
    def _color_critical_possible(c_idx):
        flag = jnp.asarray(False)
        for r in (0, 2, 3):
            played_cr = heights[c_idx] > r
            remaining_cr = mult[r] - discard_count[c_idx, r] - jnp.where(played_cr, 1.0, 0.0)
            valuable_cr = heights[c_idx] <= r
            flag = flag | (valuable_cr & (remaining_cr <= 1.0))
        return flag

    critical_possible_by_color = jnp.stack([_color_critical_possible(c) for c in range(NUM_COLORS)])
    chop_color_safe_play = (
        is_clue_to_me & is_chop_focus_self & act_is_color_hint
        & (~jnp.take(critical_possible_by_color, hinted_color_idx)))

    set_target = (non_chop_focus_self | chop_color_safe_play) & act_is_color_hint
    new_target_val = jnp.take(next_needed, hinted_color_idx).astype(jnp.int32)

    focus_onehot = jax.nn.one_hot(focus_self, NUM_SLOTS, dtype=jnp.bool_)
    update_mask = focus_onehot & set_target
    target_rank = jnp.where(update_mask, new_target_val, memory['target_rank'])
    target_valid = jnp.where(update_mask, True, memory['target_valid'])

    # ---- (2) shift our private per-slot notes if we played/discarded ----
    self_removed = acted_by_self & (act_is_play | act_is_discard)
    idxs = jnp.arange(NUM_SLOTS)
    shift_mask = idxs >= removed_slot
    rolled_rank = jnp.roll(target_rank, -1)
    rolled_valid = jnp.roll(target_valid, -1)
    shifted_rank = jnp.where(shift_mask, rolled_rank, target_rank)
    shifted_valid = jnp.where(shift_mask, rolled_valid, target_valid)
    shifted_rank = shifted_rank.at[NUM_SLOTS - 1].set(0)
    shifted_valid = shifted_valid.at[NUM_SLOTS - 1].set(False)
    target_rank = jnp.where(self_removed, shifted_rank, target_rank)
    target_valid = jnp.where(self_removed, shifted_valid, target_valid)

    new_prev_own_clued = own_clued

    # ================= decision logic =================
    rank_idx = jnp.arange(NUM_RANKS)
    dead_cell = rank_idx[None, :] < heights[:, None]      # [color, rank]
    current_cell = rank_idx[None, :] == heights[:, None]

    # Native V0 does not subtract cards visible in the partner's hand (see
    # environment README). We can see the partner's hand exactly, so we can
    # safely (with no "trust" assumption, just counting) rule out any
    # (color, rank) identity for our own hand that is already fully
    # accounted for elsewhere: by the partner's visible copies, discards, and
    # the one copy already placed in the firework.
    partner_count = jnp.zeros((NUM_COLORS, NUM_RANKS), dtype=jnp.float32)
    for c in range(NUM_COLORS):
        for r in range(NUM_RANKS):
            partner_count = partner_count.at[c, r].set(jnp.sum(jnp.where(
                partner_occupied & (partner_true_color == c) & (partner_true_rank == r), 1.0, 0.0)))
    played_once = jnp.where(dead_cell, 1.0, 0.0)  # rank < height[color] means that rank's one firework copy is used
    accounted_elsewhere = partner_count + discard_count + played_once
    exhausted_cell = accounted_elsewhere >= mult[None, :]

    def certain_play_slot(s):
        # No good-touch "trust" exclusion of dead candidates here: a card
        # that was a legitimate (non-dead) touch when clued can later become
        # a dead duplicate purely through unrelated same-color progress (the
        # other copy of that identity gets played by someone else), and that
        # residual probability mass is real, not an artifact of a GTP
        # violation. Excluding it would manufacture false certainty and risk
        # a bomb. A dead candidate is exactly as dangerous as a future one --
        # both would fail if played -- so certainty requires ALL remaining
        # support (dead or future) to be absent, not just future. The only
        # exclusion trusted here is pure counting (exhausted_cell), not GTP.
        v0 = own_V0[s]
        support = (v0 > 0.0) & (~exhausted_cell)
        any_support = jnp.any(support)
        off_target = jnp.any(support & (~current_cell))
        has_current = jnp.any(support & current_cell)
        return own_occupied[s] & any_support & has_current & (~off_target)

    def certainly_dead_slot(s):
        v0 = own_V0[s]
        support = v0 > 0.0
        all_dead = jnp.all(jnp.where(support, dead_cell, True))
        return own_occupied[s] & jnp.any(support) & all_dead

    def dead_mass_slot(s):
        v0 = own_V0[s]
        return jnp.sum(jnp.where(dead_cell, v0, 0.0))

    own_color_hint_present = jnp.sum(own_color_flags, axis=1) > 0.0
    assumed_color = jnp.argmax(own_color_flags, axis=1)

    certain_elim = jnp.stack([certain_play_slot(s) for s in range(NUM_SLOTS)])
    certainly_dead = jnp.stack([certainly_dead_slot(s) for s in range(NUM_SLOTS)])
    dead_mass = jnp.stack([dead_mass_slot(s) for s in range(NUM_SLOTS)])

    target_trigger = target_valid & own_color_hint_present & (
        jnp.take(next_needed, assumed_color).astype(jnp.int32) == target_rank)
    own_certain_play = certain_elim | (own_occupied & target_trigger)

    # ---- play scores ----
    play_scores = jnp.where(own_occupied, jnp.where(own_certain_play, 1000.0, -1000.0), NEG)

    # ---- discard scores ----
    is_chop_own = (idxs == own_chop) & (own_chop < NUM_SLOTS)
    base_discard = jnp.where(
        certainly_dead, 500.0,
        jnp.where(is_chop_own, 100.0, -800.0)) + dead_mass * 50.0
    discard_legal = own_occupied & (info_tokens < 8.0)
    discard_scores = jnp.where(discard_legal, base_discard, NEG)

    # ---- hint scores: 5 color hints + 5 rank hints, targeting partner ----
    def hint_score(is_color, value):
        if is_color:
            touched = partner_occupied & (partner_true_color == value)
        else:
            touched = partner_occupied & (partner_true_rank == value)
        newly_touched = touched & (~partner_clued)
        any_new = jnp.any(newly_touched)

        chop_idx_safe = jnp.clip(partner_chop, 0, NUM_SLOTS - 1)
        chop_exists = partner_chop < NUM_SLOTS
        chop_touched = chop_exists & jnp.take(touched, chop_idx_safe)

        focus_retouch_p = _highest_index(touched)
        focus_new_p = _highest_index(newly_touched)
        focus = jnp.where(~any_new, focus_retouch_p,
                           jnp.where(chop_touched, partner_chop, focus_new_p))
        focus = jnp.clip(focus, 0, NUM_SLOTS - 1)

        is_chop_focus = chop_exists & (focus == partner_chop)

        t_color = jnp.take(partner_true_color, focus)
        t_rank = jnp.take(partner_true_rank, focus)
        height_focus = jnp.take(next_needed, t_color)
        is_5 = t_rank == 4
        is_2 = t_rank == 1
        is_currently_playable = height_focus == t_rank
        is_valuable = height_focus <= t_rank
        played_flag = height_focus > t_rank
        dcount = discard_count[t_color, t_rank]
        remaining_copies = mult[t_rank] - dcount - jnp.where(played_flag, 1.0, 0.0)

        other_2_mask = (idxs != focus) & partner_occupied & (partner_true_color == t_color) & (partner_true_rank == 1)
        visible_other_copies = jnp.sum(jnp.where(other_2_mask, 1.0, 0.0)) + discard_count[t_color, 1]

        needs_save_5 = is_5 & is_valuable
        needs_save_2 = is_2 & is_valuable & (visible_other_copies == 0.0)
        needs_save_critical = is_valuable & (remaining_copies <= 1.0)

        # A retouch (no new card) is a Tempo Clue and normally discouraged by
        # MCVP, but a retouch that resolves an already-clued ambiguous card
        # into a certain current play is a valuable fill-in clue, not idle
        # repetition, so it is not penalized the same way.
        base = jnp.where(any_new, 10.0, jnp.where(is_currently_playable, 0.0, -500.0))

        is_this_rank4 = (not is_color) and (value == 4)
        is_this_rank1 = (not is_color) and (value == 1)

        # A color clue on a card that is NOT currently playable creates a
        # "delayed" Play Clue that our own receiver logic cannot safely track
        # (it cannot see its own hand to know the true rank, so it can only
        # assume "immediately playable now" -- wrong for a genuine delay,
        # which risks a bomb later). Only rank clues are safe to leave
        # deliberately delayed/ambiguous, since those never trigger that
        # assumption and instead wait for provable elimination.
        delayed_color_penalty = -100.0 if is_color else 0.0
        chop_bonus = jnp.where(
            needs_save_5, (300.0 if is_this_rank4 else 60.0),
            jnp.where(
                needs_save_2, (300.0 if is_this_rank1 else 60.0),
                jnp.where(
                    needs_save_critical, 250.0,
                    jnp.where(is_currently_playable, 280.0, -50.0 + delayed_color_penalty))))
        nonchop_bonus = jnp.where(is_currently_playable, 280.0, 40.0 + delayed_color_penalty)
        outcome_bonus = jnp.where(is_chop_focus, chop_bonus, nonchop_bonus)
        # Good Touch Principle: never intentionally focus a card that is
        # already dead (its rank has been surpassed for that color). Beyond
        # wasting the clue, this would violate the trust our own receiver
        # logic places in clues (that a clued card's rank is never a
        # since-surpassed duplicate), which is what an elimination-based
        # receiver relies on to safely resolve ambiguous clued identities.
        outcome_bonus = jnp.where(is_valuable, outcome_bonus, -1000.0)

        dead_partner = partner_true_rank < jnp.take(next_needed, partner_true_color).astype(jnp.int32)
        collateral = (idxs != focus) & newly_touched & dead_partner
        collateral_penalty = -40.0 * jnp.sum(jnp.where(collateral, 1.0, 0.0))

        color_pref = 5.0 if is_color else 0.0

        score = base + outcome_bonus + collateral_penalty + color_pref
        legal = (info_tokens > 0.0) & jnp.any(touched)
        return jnp.where(legal, score, NEG)

    color_hint_scores = jnp.stack([hint_score(True, c) for c in range(NUM_COLORS)])
    rank_hint_scores = jnp.stack([hint_score(False, r) for r in range(NUM_RANKS)])

    noop_score = jnp.asarray(0.0)

    raw_scores = jnp.concatenate([
        discard_scores, play_scores, color_hint_scores, rank_hint_scores,
        noop_score[None],
    ])
    final_scores = jnp.where(obs.legal_actions, raw_scores, NEG)
    action = jnp.argmax(final_scores).astype(jnp.int32)

    zeros5i = jnp.zeros(NUM_SLOTS, dtype=jnp.int32)
    zeros5b = jnp.zeros(NUM_SLOTS, dtype=jnp.bool_)
    next_memory = {
        'target_rank': jnp.where(reset, zeros5i, target_rank),
        'target_valid': jnp.where(reset, zeros5b, target_valid),
        'prev_own_clued': jnp.where(reset, zeros5b, new_prev_own_clued),
    }

    trace = {
        'action': action,
        'own_certain_play': own_certain_play,
        'own_target_trigger': target_trigger,
        'own_chop': own_chop.astype(jnp.int32),
        'partner_chop': partner_chop.astype(jnp.int32),
        'info_tokens': info_tokens,
        'firework_sum': jnp.sum(heights),
        'lives': jnp.sum(f[200:203]),
        'deck': jnp.sum(f[127:167]),
    }

    return action, next_memory, trace
