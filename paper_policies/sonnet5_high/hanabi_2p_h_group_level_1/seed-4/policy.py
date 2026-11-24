"""H-group Level 1 rule-based Hanabi policy.

Per-turn reactive policy that reconstructs the shared H-group Level 1
convention protocol from the observation, plus a small private memory that
tracks, for the agent's own hand, which slot is the believed *focus* of a
received (non-save-ambiguous) play clue. Slot indices are native indices
(runtime_mapping.md: chop = lowest occupied native index among unclued
cards; "leftmost" in H-group prose = highest occupied native index).

Conventions implemented:

  - Chop = lowest occupied native index among unclued cards.
  - Clue focus (leftmost-retouch / chop-priority / leftmost-new), computed
    exactly as documented. As the receiver, the focus of a hint that is
    given to us is computed from the pre-clue clued-status carried in
    memory plus this transition's public hint-match indicators; the focus
    slot of a clue that does *not* land on our chop is unambiguously a
    Play Clue and is trusted (played once it is our turn and nothing
    higher-priority is pending). A clue landing on chop is ambiguous
    (Play Clue vs. Save Clue) and is only played once literally resolved
    (see below), matching "hold it" guidance.
  - Good Touch Principle-filtered near-certain V0 test: a clued own card
    with all but a negligible amount of its (trash-excluded) V0 mass on a
    single (color, rank) whose rank equals that color's current firework
    height is played -- this both handles doubly-hinted (fully resolved)
    cards and lets chop-held ambiguous cards resolve once elimination
    narrows them down.
  - Save Principle: 5-save (rank clue on a 5), 2-save (rank clue on a
    unique-visible 2, respecting the visibility rule), critical save
    (color or rank clue, chosen for Good Touch cleanliness) on the
    partner's chop.
  - Minimum Clue Value Principle: no clue is given unless it saves or
    newly touches a target card that is currently playable and whose
    focus (computed the same way, from the giver's full-information view
    of the partner's hand) lands exactly on that target; otherwise
    discard chop (or, once fully clued, the most plausibly trash occupied
    card).
  - A legal-action safety net: every candidate action is only used if the
    native legal-action mask agrees; otherwise the policy falls back to
    the first legal action (so a logic bug can never produce an illegal
    move).

Own-hand identities are never observed directly; the private memory is
realigned every turn using the public "latest action" block (offsets
253:308 of the feature vector), which both seats see every transition.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
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

_RANKS5_LIST = (1.0, 2.0, 3.0, 4.0, 5.0)
_COPIES_LIST = (3.0, 2.0, 2.0, 2.0, 1.0)
_RANK_OFFSETS = (0, 3, 5, 7, 9, 10)


def init_agent():
    return {
        'own_clued_prev': jnp.zeros((5,), dtype=jnp.bool_),
        'believed_focus_play': jnp.zeros((5,), dtype=jnp.bool_),
    }


def _focus_of(matched, clued, chop_idx, chop_exists, idxs5):
    """H-group clue-focus rule: leftmost-retouch / chop / leftmost-new.

    "Leftmost" in H-group's left-to-right hand maps to the highest native
    index (see runtime_mapping.md); native slot 0 is oldest, 4 is newest.
    """
    new_mask = matched & (~clued)
    any_new = jnp.any(new_mask)
    highest_matched = jnp.max(jnp.where(matched, idxs5, -1))
    chop_touched = chop_exists & jnp.take(matched, chop_idx)
    highest_new = jnp.max(jnp.where(new_mask, idxs5, -1))
    focus = jnp.where(
        ~any_new,
        highest_matched,
        jnp.where(chop_touched, chop_idx, highest_new),
    )
    return focus


def _shift_left(arr, removed_idx, idxs5):
    """Compact a private-belief array the same way the environment compacts
    a hand: drop removed_idx, shift later entries left, clear the new slot
    at index 4 (fresh draw or empty append carries no prior belief)."""
    src = jnp.where(idxs5 < removed_idx, idxs5, idxs5 + 1)
    src = jnp.clip(src, 0, 4)
    shifted = jnp.take(arr, src)
    shifted = jnp.where(idxs5 == 4, jnp.zeros_like(shifted[:1])[0], shifted)
    return shifted


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    idxs5 = jnp.arange(5)
    ranks5 = jnp.asarray(_RANKS5_LIST, dtype=jnp.float32)
    copies5 = jnp.asarray(_COPIES_LIST, dtype=jnp.float32)

    partner_hand = f[0:125].reshape(5, 5, 5)
    fireworks = f[167:192].reshape(5, 5)
    info_thermo = f[192:200]
    lives_thermo = f[200:203]
    discards = f[203:253].reshape(5, 10)
    last = f[253:308]
    knowledge = f[308:658].reshape(2, 5, 35)

    own_know = knowledge[0]
    partner_know = knowledge[1]

    own_V0 = own_know[:, 0:25].reshape(5, 5, 5)
    own_pos_color = own_know[:, 25:30]
    own_pos_rank = own_know[:, 30:35]
    own_clued = (own_pos_color.sum(-1) > 0.5) | (own_pos_rank.sum(-1) > 0.5)
    own_occupied = own_V0.sum(axis=(1, 2)) > 1e-4

    partner_pos_color = partner_know[:, 25:30]
    partner_pos_rank = partner_know[:, 30:35]
    partner_clued = (partner_pos_color.sum(-1) > 0.5) | (partner_pos_rank.sum(-1) > 0.5)
    partner_occupied = partner_hand.sum(axis=(1, 2)) > 0.5

    height = jnp.sum(fireworks * ranks5[None, :], axis=1)  # [5] human height 0..5
    info_tokens = jnp.sum(info_thermo)

    trash_mask = idxs5[None, :].astype(height.dtype) < height[:, None]  # [color,rank]

    # Public remaining-copy count per (color, rank), needed both to decide
    # whether the partner's chop needs a Save Clue and, as receiver, whether
    # a chop-focus clue we received could possibly be a Save Clue at all.
    def _disc_count(c, r):
        lo = _RANK_OFFSETS[r]
        hi = _RANK_OFFSETS[r + 1]
        return jnp.sum(discards[c, lo:hi])

    disc_rows = []
    for c in range(5):
        disc_rows.append(jnp.stack([_disc_count(c, r) for r in range(5)]))
    disc_counts = jnp.stack(disc_rows)  # [5,5]
    played_counts = (idxs5[None, :].astype(height.dtype) < height[:, None]).astype(jnp.float32)
    remaining_counts = copies5[None, :] - disc_counts - played_counts

    # ---------------- realign private memory to current slot layout -------
    actor_onehot = last[0:2]
    action_type_onehot = last[2:6]
    recipient_onehot = last[6:8]
    rank_hint_onehot = last[13:18]
    hint_match = last[18:23]
    removed_slot_onehot = last[23:28]

    actor_is_self = actor_onehot[0] > 0.5
    action_is_play = action_type_onehot[0] > 0.5
    action_is_discard = action_type_onehot[1] > 0.5
    action_is_colorhint = action_type_onehot[2] > 0.5
    action_is_rankhint = action_type_onehot[3] > 0.5
    recipient_is_self = recipient_onehot[0] > 0.5
    rank_hint_value = jnp.argmax(rank_hint_onehot)

    own_removed_now = actor_is_self & (action_is_play | action_is_discard)
    removed_idx = jnp.argmax(removed_slot_onehot)

    m_clued_prev = jnp.where(
        own_removed_now,
        _shift_left(memory['own_clued_prev'], removed_idx, idxs5),
        memory['own_clued_prev'],
    )
    m_focus_prev = jnp.where(
        own_removed_now,
        _shift_left(memory['believed_focus_play'], removed_idx, idxs5),
        memory['believed_focus_play'],
    )

    hint_to_me = (~actor_is_self) & recipient_is_self & (action_is_colorhint | action_is_rankhint)
    matched_now = hint_match > 0.5
    chop_prev_unclued_occ = own_occupied & (~m_clued_prev)
    chop_prev_exists = jnp.any(chop_prev_unclued_occ)
    chop_prev_idx = jnp.argmax(chop_prev_unclued_occ)
    focus_now = _focus_of(matched_now, m_clued_prev, chop_prev_idx, chop_prev_exists, idxs5)
    is_chop_focus = hint_to_me & chop_prev_exists & (focus_now == chop_prev_idx)
    any_new_touch = jnp.any(matched_now & (~m_clued_prev))

    # A chop-focus clue is ambiguous (Play vs. Save) only while a Save
    # interpretation remains literally possible (beginner/single-card-focus.md
    # "Example 2: Chop-Focus"; docs/level-1.md "Save Clues"). A rank-5 or
    # rank-2 clue can always be a 5-/2-Save, so those stay ambiguous. Any
    # other clue can only be a Critical Save, which is ruled out once none
    # of the slot's current (post-clue) candidates are actually critical.
    could_be_5_or_2_save = action_is_rankhint & ((rank_hint_value == 4) | (rank_hint_value == 1))
    own_V0_now = own_know[:, 0:25].reshape(5, 5, 5)
    focus_candidates = own_V0_now[focus_now] > 1e-6  # [color, rank]
    focus_critical = remaining_counts <= 1.0
    any_critical_candidate = jnp.any(focus_candidates & focus_critical)
    save_impossible = hint_to_me & is_chop_focus & (~could_be_5_or_2_save) & (~any_critical_candidate)
    chop_focus_is_play = is_chop_focus & save_impossible

    # Only a focus that actually introduces a previously-unclued card is an
    # unambiguous (non-tempo) Play Clue; a pure retouch carries no reliable
    # "play this now" signal here (H-group treats Tempo Clues as a rare,
    # situational exception, not a default receiver inference) and must not
    # overwrite existing belief.
    new_focus_play = hint_to_me & (~is_chop_focus | chop_focus_is_play) & any_new_touch

    candidate_focus_arr = m_focus_prev.at[focus_now].set(True)
    believed_focus_play = jnp.where(new_focus_play, candidate_focus_arr, m_focus_prev)
    # A slot can stop being a believed-focus target once it is no longer
    # occupied by the same (unplayed) card, e.g. after we actually play it.
    believed_focus_play = believed_focus_play & own_occupied

    # ---------------- own-card play trust ----------------------------------
    # Good Touch only promises a clued card will *eventually* be played; it
    # does not retroactively prove a literal candidate whose rank the board
    # has since passed (a duplicate of an already-played rank -- genuinely
    # possible and, per docs/level-1.md "The Definition of Trash", real
    # trash) cannot be the true card. So certainty must be computed from the
    # literal (unfiltered) V0 distribution directly: a card is playable now
    # only if the literal probability mass sitting on immediately-playable
    # (color, rank) combinations is (near-)total, not merely dominant among
    # a GTP-trash-renormalized subset.
    playable_mask = idxs5[None, :].astype(height.dtype) == height[:, None]  # [color,rank]
    own_playable_mass = (own_V0 * playable_mask[None, :, :]).sum(axis=(1, 2))
    own_total_mass = own_V0.sum(axis=(1, 2))
    playable_frac = own_playable_mass / jnp.maximum(own_total_mass, 1e-6)
    # Still used (as a soft heuristic, not a certainty claim) to rank which
    # already-clued card is least likely to be useful when a discard must
    # be chosen from an all-clued hand.
    own_V0_filt = own_V0 * (~trash_mask)[None, :, :].astype(own_V0.dtype)
    own_filt_sum = own_V0_filt.sum(axis=(1, 2))

    play_ok_tight = own_occupied & own_clued & (own_total_mass > 1e-6) & (playable_frac > 0.999)
    # Focus establishes *that* a card is meant to be played, but a Play
    # Clue may still be a Delayed Play Clue (beginner/delayed-play-clues.md):
    # only actually play it once the board plausibly supports it (a clear
    # literal majority of the card's possible identities are immediately
    # playable), so we do not blind-play through an unresolved connector
    # dependency or a mostly-trash ambiguous hold.
    play_ok_focus = (
        own_occupied & believed_focus_play & (own_total_mass > 1e-6) & (playable_frac > 0.2)
    )
    play_ok = play_ok_tight | play_ok_focus
    play_score = (
        jnp.where(play_ok_focus, 2.0, 0.0) + jnp.where(play_ok_tight, 1.0, 0.0)
    )
    play_score = jnp.where(play_ok, play_score, -1.0)
    best_play_idx = jnp.argmax(play_score)
    best_play_valid = jnp.any(play_ok)
    best_play_action = (5 + best_play_idx).astype(jnp.int32)

    # ---------------- own chop / discard target -----------------------------
    own_unclued_occ = own_occupied & (~own_clued)
    own_chop_exists = jnp.any(own_unclued_occ)
    own_chop_idx = jnp.argmax(own_unclued_occ)
    fallback_chop_idx = jnp.argmin(
        jnp.where(own_occupied & own_clued, own_filt_sum, jnp.inf)
    )
    discard_slot = jnp.where(own_chop_exists, own_chop_idx, fallback_chop_idx).astype(jnp.int32)
    discard_action = discard_slot.astype(jnp.int32)
    discard_valid = info_tokens < 7.5

    # ---------------- partner chop / true identities -------------------------
    partner_unclued_occ = partner_occupied & (~partner_clued)
    partner_chop_exists = jnp.any(partner_unclued_occ)
    partner_chop_idx = jnp.argmax(partner_unclued_occ).astype(jnp.int32)

    partner_flat = partner_hand.reshape(5, 25)
    partner_ident_idx = jnp.argmax(partner_flat, axis=1)
    partner_color = partner_ident_idx // 5
    partner_rank = partner_ident_idx % 5

    bad_flag = (jnp.take(height, partner_color) > partner_rank.astype(height.dtype)) & partner_occupied

    chop_color = jnp.take(partner_color, partner_chop_idx)
    chop_rank = jnp.take(partner_rank, partner_chop_idx)

    chop_remaining = remaining_counts[chop_color, chop_rank]

    same_ident = (partner_color == chop_color) & (partner_rank == chop_rank) & partner_occupied
    other_visible = jnp.maximum(jnp.sum(same_ident.astype(jnp.int32)) - 1, 0)

    is_five = partner_chop_exists & (chop_rank == 4)
    is_two = partner_chop_exists & (chop_rank == 1)
    is_two_unique = is_two & (other_visible == 0)
    is_critical = partner_chop_exists & (chop_remaining <= 1.0)
    needs_save = is_five | is_two_unique | (is_critical & (~is_five))

    def _bad_count(matched, focus_idx):
        new_mask = matched & (~partner_clued)
        excl_focus = idxs5 != focus_idx
        return jnp.sum((new_mask & excl_focus & bad_flag).astype(jnp.int32))

    matched_color_save = (partner_color == chop_color) & partner_occupied
    matched_rank_save = (partner_rank == chop_rank) & partner_occupied
    bc_color = _bad_count(matched_color_save, partner_chop_idx)
    bc_rank = _bad_count(matched_rank_save, partner_chop_idx)
    critical_action = jnp.where(bc_color <= bc_rank, 10 + chop_color, 15 + chop_rank)

    save_action = jnp.where(
        is_five, 15 + 4,
        jnp.where(is_two_unique, 15 + 1, critical_action),
    ).astype(jnp.int32)
    save_valid = needs_save & (info_tokens >= 0.5)

    # ---------------- partner play clues --------------------------------------
    partner_playable = (
        partner_occupied
        & (~partner_clued)
        & (jnp.take(height, partner_color) == partner_rank.astype(height.dtype))
    )

    pc_actions = []
    pc_valid = []
    pc_priority = []
    for t in range(5):
        tgt_color = partner_color[t]
        tgt_rank = partner_rank[t]
        for htype in range(2):
            if htype == 0:
                matched = (partner_color == tgt_color) & partner_occupied
                action = (10 + tgt_color).astype(jnp.int32)
                type_bonus = 1.0
            else:
                matched = (partner_rank == tgt_rank) & partner_occupied
                action = (15 + tgt_rank).astype(jnp.int32)
                type_bonus = 0.0
            focus = _focus_of(matched, partner_clued, partner_chop_idx, partner_chop_exists, idxs5)
            bad_ct = _bad_count(matched, focus)
            valid = partner_playable[t] & (focus == t) & (info_tokens >= 0.5)
            priority = (
                800.0
                - bad_ct.astype(jnp.float32) * 10.0
                + type_bonus
                + jnp.where(t == partner_chop_idx, 5.0, 0.0)
            )
            pc_actions.append(action)
            pc_valid.append(valid)
            pc_priority.append(priority)
    pc_actions = jnp.stack(pc_actions)
    pc_valid = jnp.stack(pc_valid)
    pc_priority = jnp.stack(pc_priority)
    pc_scores = jnp.where(pc_valid, pc_priority, -1e9)
    best_pc_idx = jnp.argmax(pc_scores)
    best_pc_action = pc_actions[best_pc_idx]
    best_pc_valid = pc_valid[best_pc_idx]

    # ---------------- forced-stall fallback clue (8 tokens, nothing else) ------
    # This clue is not a genuine Play or Save Clue (Minimum Clue Value
    # Principle would otherwise require a discard); a convention-following
    # receiver must not read it as an immediate-play promise. Prefer a pure
    # retouch (zero previously-unclued cards newly touched) so the shared
    # any-new-touch focus rule cannot mistake it for one; only touch new
    # cards if no retouch-only legal hint exists, and then minimize Good
    # Touch violations.
    hint_legal = obs.legal_actions[10:20]
    fb_new_count = []
    fb_bad_count = []
    for v in range(5):
        matched = (partner_color == v) & partner_occupied
        new_ct = jnp.sum((matched & (~partner_clued)).astype(jnp.int32))
        bad_ct = jnp.sum((matched & (~partner_clued) & bad_flag).astype(jnp.int32))
        fb_new_count.append(new_ct)
        fb_bad_count.append(bad_ct)
    for v in range(5):
        matched = (partner_rank == v) & partner_occupied
        new_ct = jnp.sum((matched & (~partner_clued)).astype(jnp.int32))
        bad_ct = jnp.sum((matched & (~partner_clued) & bad_flag).astype(jnp.int32))
        fb_new_count.append(new_ct)
        fb_bad_count.append(bad_ct)
    fb_new_count = jnp.stack(fb_new_count).astype(jnp.float32)
    fb_bad_count = jnp.stack(fb_bad_count).astype(jnp.float32)
    fb_score = jnp.where(
        hint_legal,
        jnp.where(fb_new_count < 0.5, 200.0, 100.0 - fb_bad_count),
        -1e9,
    )
    fallback_clue_idx = jnp.argmax(fb_score)
    fallback_clue_action = (10 + fallback_clue_idx).astype(jnp.int32)
    fallback_clue_valid = jnp.any(hint_legal)

    # ---------------- combine by priority tier ---------------------------------
    all_actions = jnp.stack([
        best_play_action,
        save_action,
        best_pc_action,
        discard_action,
        fallback_clue_action,
    ])
    all_valid = jnp.stack([
        best_play_valid,
        save_valid,
        best_pc_valid,
        discard_valid,
        fallback_clue_valid,
    ])
    all_priority = jnp.array([1000.0, 900.0, 700.0, 100.0, 50.0], dtype=jnp.float32)
    scores = jnp.where(all_valid, all_priority, -1e9)
    chosen_idx = jnp.argmax(scores)
    chosen_action = all_actions[chosen_idx]

    legal = obs.legal_actions
    final_ok = legal[chosen_action]
    ultimate = jnp.argmax(legal.astype(jnp.int32)).astype(jnp.int32)
    action = jnp.where(final_ok, chosen_action, ultimate).astype(jnp.int32)

    stored_memory = {
        'own_clued_prev': own_clued,
        'believed_focus_play': believed_focus_play,
    }
    init_mem = init_agent()
    new_memory = {
        key: jnp.where(reset, init_mem[key], stored_memory[key])
        for key in stored_memory
    }
    trace = {
        'chosen_tier': chosen_idx.astype(jnp.int32),
        'lives': jnp.sum(lives_thermo),
        'height': height,
        'info_tokens': info_tokens,
        'believed_focus_play': believed_focus_play.astype(jnp.int32),
    }
    return action, new_memory, trace
