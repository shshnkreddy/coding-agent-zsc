"""A conservative two-player H-group Level 1 policy.

The only meanings assigned to clues here are the documented public meanings:
focus, play/delayed-play, 2/5/critical saves, and Good Touch. Memory contains
card-attached public inferences and is shifted exactly when a card is removed.
"""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_level1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


_MULT = (3, 2, 2, 2, 1)


def init_agent():
    return {
        "intent": jnp.zeros((2, 5, 25), dtype=jnp.bool_),
        "clued": jnp.zeros((2, 5), dtype=jnp.bool_),
        "trusted": jnp.zeros((2, 5), dtype=jnp.bool_),
    }


def _shift_slots(x, removed):
    """Apply native left shift and clear the newly drawn/final empty slot."""
    slots = jnp.arange(5, dtype=jnp.int32)
    src = jnp.minimum(slots + (slots >= removed).astype(jnp.int32), 4)
    moved = x[src]
    shape = (5,) + (1,) * (x.ndim - 1)
    keep = (slots < 4).reshape(shape)
    return jnp.where(keep, moved, jnp.zeros_like(moved))


def _lowest(mask):
    slots = jnp.arange(5, dtype=jnp.int32)
    return jnp.argmax(mask.astype(jnp.int32) * (5 - slots))


def _highest(mask):
    slots = jnp.arange(5, dtype=jnp.int32)
    return jnp.argmax(mask.astype(jnp.int32) * (slots + 1))


def _focus(matches, old_clued, occupied):
    """H-group focus in native order, using the pre-clue clued set."""
    eligible_chop = occupied & (~old_clued)
    has_chop = jnp.any(eligible_chop)
    chop = _lowest(eligible_chop)
    new = matches & (~old_clued)
    has_new = jnp.any(new)
    chop_touched = has_chop & matches[chop]
    focus_new = jnp.where(chop_touched, chop, _highest(new))
    focus = jnp.where(has_new, focus_new, _highest(matches))
    return focus, new, chop, has_chop


def _hall_propagate(candidates, occupied, capacity):
    """Enforce tight public copy-capacity sets across five hand slots."""
    narrowed = candidates
    for _ in range(3):
        for subset in range(1, 32):
            members = jnp.asarray(
                [bool(subset & (1 << s)) for s in range(5)], dtype=jnp.bool_
            ) & occupied
            union = jnp.any(narrowed & members[:, None], axis=0)
            demand = jnp.sum(members.astype(jnp.int32))
            available = jnp.sum(jnp.where(union, capacity, 0))
            tight = (demand > 0) & (available == demand)
            remove = (~members[:, None]) & union[None, :] & tight
            narrowed = narrowed & (~remove)
    return narrowed


def _board(features):
    fw_rows = features[167:192].reshape((5, 5))
    fw = jnp.sum(fw_rows * jnp.arange(1, 6, dtype=jnp.float32), axis=1).astype(jnp.int32)
    d = features[203:253].reshape((5, 10))
    discards = jnp.stack(
        [jnp.sum(d[:, 0:3], axis=1), jnp.sum(d[:, 3:5], axis=1),
         jnp.sum(d[:, 5:7], axis=1), jnp.sum(d[:, 7:9], axis=1),
         d[:, 9]], axis=1
    ).astype(jnp.int32)
    rank_ids = jnp.arange(5, dtype=jnp.int32)[None, :]
    multiplicities = jnp.asarray(_MULT, dtype=jnp.int32)
    remaining = multiplicities[None, :] - discards - (fw[:, None] > rank_ids).astype(jnp.int32)

    useful_parts = []
    for r in range(5):
        needed = (jnp.arange(5)[None, :] >= fw[:, None]) & (jnp.arange(5)[None, :] <= r)
        path_alive = jnp.all((~needed) | (remaining > 0), axis=1)
        useful_parts.append((r >= fw) & path_alive)
    useful = jnp.stack(useful_parts, axis=1)
    direct = rank_ids == fw[:, None]
    critical = (remaining == 1) & useful & (~direct) & (rank_ids < 4)
    two_save = (rank_ids == 1) & useful & (~direct)
    five_save = (rank_ids == 4) & useful
    return fw, remaining, useful.reshape(25), direct.reshape(25), critical.reshape(25), two_save.reshape(25), five_save.reshape(25)


def _process_public_action(features, reset, memory, literal, occupied,
                           useful, direct, critical, two_save, five_save):
    la = features[253:308]
    is_play = la[2] > 0.5
    is_discard = la[3] > 0.5
    is_color_hint = la[4] > 0.5
    is_rank_hint = la[5] > 0.5
    is_hint = (is_color_hint | is_rank_hint) & (~reset)
    is_remove = (is_play | is_discard) & (~reset)
    actor_self = la[0] > 0.5
    actor_partner = la[1] > 0.5
    removed = jnp.argmax(la[23:28]).astype(jnp.int32)

    base_intent = jnp.where(reset, jnp.zeros_like(memory["intent"]), memory["intent"])
    base_clued = jnp.where(reset, jnp.zeros_like(memory["clued"]), memory["clued"])
    base_trusted = jnp.where(reset, jnp.zeros_like(memory["trusted"]), memory["trusted"])
    shifted0_i = _shift_slots(base_intent[0], removed)
    shifted1_i = _shift_slots(base_intent[1], removed)
    shifted0_c = _shift_slots(base_clued[0], removed)
    shifted1_c = _shift_slots(base_clued[1], removed)
    shifted0_t = _shift_slots(base_trusted[0], removed)
    shifted1_t = _shift_slots(base_trusted[1], removed)
    intent = base_intent.at[0].set(jnp.where(is_remove & actor_self, shifted0_i, base_intent[0]))
    intent = intent.at[1].set(jnp.where(is_remove & actor_partner, shifted1_i, base_intent[1]))
    old_clued = base_clued.at[0].set(jnp.where(is_remove & actor_self, shifted0_c, base_clued[0]))
    old_clued = old_clued.at[1].set(jnp.where(is_remove & actor_partner, shifted1_c, base_clued[1]))
    trusted = base_trusted.at[0].set(jnp.where(is_remove & actor_self, shifted0_t, base_trusted[0]))
    trusted = trusted.at[1].set(jnp.where(is_remove & actor_partner, shifted1_t, base_trusted[1]))

    recipient = jnp.argmax(la[6:8]).astype(jnp.int32)
    matches = la[18:23] > 0.5
    focus, new, chop, has_chop = _focus(matches, old_clued[recipient], occupied[recipient])
    chop_focus = has_chop & (focus == chop)
    focus_literal = literal[recipient, focus]
    direct_candidates = focus_literal & direct

    rank_value = jnp.argmax(la[13:18]).astype(jnp.int32)
    rank_five = is_rank_hint & (rank_value == 4)
    rank_two = is_rank_hint & (rank_value == 1)
    save_types = critical | (rank_two & two_save) | (rank_five & five_save)
    save_candidates = focus_literal & save_types
    focused = jnp.where(chop_focus, direct_candidates | save_candidates, direct_candidates)
    fallback = focus_literal & useful
    focused = jnp.where(jnp.any(focused), focused, fallback)

    rec_intent = intent[recipient]
    slots = jnp.arange(5, dtype=jnp.int32)
    slot_is_focus = slots == focus
    fresh_nonfocus = matches & new & (~slot_is_focus)
    inferred_nonfocus = literal[recipient] & useful[None, :]
    focus_is_singleton = jnp.sum(focused.astype(jnp.int32)) == 1
    inferred_nonfocus = inferred_nonfocus & (~(focused[None, :] & focus_is_singleton))
    rec_intent = jnp.where(fresh_nonfocus[:, None], inferred_nonfocus, rec_intent)
    rec_intent = jnp.where(slot_is_focus[:, None], focused[None, :], rec_intent)
    intent = intent.at[recipient].set(jnp.where(is_hint, rec_intent, intent[recipient]))
    intent = intent & literal
    rec_trusted = trusted[recipient]
    clue_trusted = jnp.sum(features[192:200]) < 7.0
    focus_trusted = clue_trusted
    rec_trusted = jnp.where(fresh_nonfocus, False, rec_trusted)
    rec_trusted = jnp.where(slot_is_focus, focus_trusted, rec_trusted)
    trusted = trusted.at[recipient].set(jnp.where(is_hint, rec_trusted, trusted[recipient]))
    trusted = trusted & jnp.any(intent, axis=2)
    return intent, trusted


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    slots = jnp.arange(5, dtype=jnp.int32)
    partner_cards = f[0:125].reshape((5, 25))
    partner_occupied = jnp.any(partner_cards > 0.5, axis=1)
    partner_types = jnp.argmax(partner_cards, axis=1).astype(jnp.int32)
    knowledge = f[308:658].reshape((2, 5, 35))
    raw_literal = knowledge[:, :, 0:25] > 0.0
    own_occupied = jnp.any(raw_literal[0], axis=1)
    current_clued = jnp.any(knowledge[:, :, 25:35] > 0.5, axis=2)

    fw, remaining, useful, direct, critical, two_save, five_save = _board(f)
    visible_counts = jnp.sum(partner_cards > 0.5, axis=0).astype(jnp.int32)
    available_to_self = visible_counts < remaining.reshape(25)
    own_literal = raw_literal[0] & available_to_self[None, :]
    self_capacity = remaining.reshape(25) - visible_counts
    own_literal = _hall_propagate(own_literal, own_occupied, self_capacity)
    literal = raw_literal.at[0].set(own_literal)
    occupied = jnp.stack([own_occupied, partner_occupied], axis=0)
    intent, trusted = _process_public_action(
        f, reset, memory, literal, occupied, useful, direct, critical, two_save, five_save
    )
    next_memory = {"intent": intent, "clued": current_clued, "trusted": trusted}

    own_intent = intent[0]
    has_intent = jnp.any(own_intent, axis=1)
    intent_ready = has_intent & jnp.all((~own_intent) | direct[None, :], axis=1)
    has_literal = jnp.any(literal[0], axis=1)
    literal_ready = has_literal & jnp.all((~literal[0]) | direct[None, :], axis=1)
    more_than_one_life = jnp.sum(f[200:203]) > 1.0
    ready = (own_occupied & ((intent_ready & (more_than_one_life | trusted[0])) |
                             literal_ready) &
             legal[5:10])
    intent_authorizes = intent_ready & (more_than_one_life | trusted[0])
    play_belief = jnp.where(intent_authorizes[:, None], own_intent, literal[0])
    type_ranks = jnp.arange(25, dtype=jnp.int32) % 5
    lowest_rank = jnp.min(jnp.where(play_belief, type_ranks[None, :], 5), axis=1)
    play_score = (5 - lowest_rank) * 10.0 + slots
    play_slot = jnp.argmax(jnp.where(ready, play_score, -10000))
    has_play = jnp.any(ready)
    play_action = 5 + play_slot

    partner_chop_mask = partner_occupied & (~current_clued[1])
    partner_has_chop = jnp.any(partner_chop_mask)
    partner_chop = _lowest(partner_chop_mask)
    chop_type = partner_types[partner_chop]
    chop_color = chop_type // 5
    chop_rank = chop_type % 5
    chop_direct = direct[chop_type]
    chop_useful = useful[chop_type]
    same_type_elsewhere = jnp.any(
        partner_occupied & (slots != partner_chop) & (partner_types == chop_type)
    )
    chop_critical = (remaining[chop_color, chop_rank] == 1) & chop_useful & (~chop_direct)
    chop_two = (chop_rank == 1) & chop_useful & (~chop_direct) & (~same_type_elsewhere)
    chop_five = (chop_rank == 4) & chop_useful & (~chop_direct)
    chop_unique_playable = chop_direct & (~same_type_elsewhere)
    urgent = partner_has_chop & (chop_critical | chop_two | chop_five | chop_unique_playable)

    cards_3d = partner_cards.reshape((5, 5, 5))
    color_matches = cards_3d.sum(axis=2).T > 0.5
    rank_matches = cards_3d.sum(axis=1).T > 0.5
    clue_matches = jnp.concatenate([color_matches, rank_matches], axis=0)
    clue_legal = legal[10:20]
    focus_list = []
    new_list = []
    for q in range(10):
        q_focus, q_new, _, _ = _focus(clue_matches[q], current_clued[1], partner_occupied)
        focus_list.append(q_focus)
        new_list.append(q_new)
    clue_focus = jnp.stack(focus_list)
    clue_new = jnp.stack(new_list)
    focus_type = partner_types[clue_focus]
    focus_direct = direct[focus_type]
    any_new = jnp.any(clue_new, axis=1)

    exact_useful = useful[partner_types]
    all_new_useful = jnp.all((~clue_new) | exact_useful[None, :], axis=1)
    type_oh = jnp.eye(25, dtype=jnp.int32)[partner_types]
    existing_type_count = jnp.sum(type_oh * current_clued[1, :, None].astype(jnp.int32), axis=0)
    new_type_count = jnp.einsum("qs,st->qt", clue_new.astype(jnp.int32), type_oh)
    duplicate_new = jnp.any(
        clue_new & ((new_type_count[:, partner_types] > 1) |
                    (existing_type_count[partner_types][None, :] > 0)), axis=1
    )
    chop_play_override = ((clue_focus == partner_chop) & partner_has_chop &
                          chop_unique_playable)
    play_clue_valid = (clue_legal & any_new & focus_direct &
                       ((all_new_useful & (~duplicate_new)) | chop_play_override))
    relaxed_play_valid = clue_legal & any_new & focus_direct
    new_value = jnp.sum(clue_new & exact_useful[None, :], axis=1).astype(jnp.int32)
    color_bonus = (jnp.arange(10) < 5).astype(jnp.int32)
    q_ids = jnp.arange(10, dtype=jnp.int32)
    type_ids = jnp.arange(25, dtype=jnp.int32)
    save_alternatives = []
    for q in range(10):
        if q < 5:
            attribute = (type_ids // 5) == q
            save_kind = critical
        else:
            clue_rank = q - 5
            attribute = (type_ids % 5) == clue_rank
            if clue_rank == 4:
                save_kind = five_save
            elif clue_rank == 1:
                save_kind = critical | two_save
            else:
                save_kind = critical
        save_alternatives.append(jnp.any(literal[1, clue_focus[q]] & attribute & save_kind))
    save_ambiguous = (jnp.stack(save_alternatives) & partner_has_chop &
                      (clue_focus == partner_chop))
    focus_rank_bonus = (5 - (focus_type % 5)) * 4
    clue_score = (new_value * 20 + focus_rank_bonus + color_bonus * 3 - q_ids -
                  save_ambiguous.astype(jnp.int32) * 100)
    best_play_q = jnp.argmax(jnp.where(play_clue_valid, clue_score, -10000))
    has_play_clue = jnp.any(play_clue_valid)
    best_play_clue = 10 + best_play_q
    best_relaxed_q = jnp.argmax(jnp.where(relaxed_play_valid, clue_score, -10000))
    has_relaxed_play = jnp.any(relaxed_play_valid)
    best_relaxed_clue = 10 + best_relaxed_q

    save_semantic = jnp.where(
        chop_rank == 4,
        q_ids == 9,
        jnp.where(chop_rank == 1,
                  q_ids == 6,
                  (q_ids == chop_color) | (q_ids == 5 + chop_rank)),
    )
    save_focus = clue_focus == partner_chop
    save_valid = clue_legal & save_semantic & save_focus & any_new
    save_score = new_value * 5 + all_new_useful.astype(jnp.int32) * 20 + color_bonus * 2 - q_ids
    best_save_q = jnp.argmax(jnp.where(save_valid, save_score, -10000))
    has_save_clue = jnp.any(save_valid)
    best_save_clue = 10 + best_save_q

    tempo_play = clue_legal & (~any_new) & focus_direct
    trash = ~useful
    literal_trash = own_occupied & jnp.all((~literal[0]) | trash[None, :], axis=1)
    intent_trash = has_intent & jnp.all((~own_intent) | trash[None, :], axis=1)
    trash_discardable = (literal_trash | intent_trash) & legal[0:5]
    has_trash = jnp.any(trash_discardable)
    trash_slot = _highest(trash_discardable)
    own_chop_mask = own_occupied & (~current_clued[0]) & legal[0:5]
    has_own_chop = jnp.any(own_chop_mask)
    own_chop = _lowest(own_chop_mask)
    regular_discard = has_trash | has_own_chop

    # A fully-clued hand has no conventional chop. If no playable tempo clue
    # exists, prefer a transparent game-legal discard to inventing a new false
    # play promise. Rank it by public trash probability and critical risk.
    weights = knowledge[0, :, 0:25]
    danger = critical | two_save | five_save | direct
    trash_probability = jnp.sum(weights * trash[None, :], axis=1)
    danger_probability = jnp.sum(weights * danger[None, :], axis=1)
    emergency_score = trash_probability * 100.0 - danger_probability * 50.0 + slots.astype(jnp.float32) * 0.01
    emergency_legal = own_occupied & legal[0:5]
    special_relaxed_play = (~regular_discard) & has_relaxed_play
    has_emergency = (jnp.any(emergency_legal) & (~jnp.any(tempo_play)) &
                     (~special_relaxed_play))
    emergency_slot = jnp.argmax(jnp.where(emergency_legal, emergency_score, -10000.0))
    discard_action = jnp.where(has_trash, trash_slot,
                               jnp.where(has_own_chop, own_chop, emergency_slot))
    has_discard = regular_discard | has_emergency

    recipient_direct_possible = jnp.any(literal[1, clue_focus] & direct[None, :], axis=1)
    delayed_hold = ~recipient_direct_possible
    forced_score = (tempo_play.astype(jnp.int32) * 4000 +
                    focus_direct.astype(jnp.int32) * 3000 +
                    delayed_hold.astype(jnp.int32) * 2000 + new_value * 10 +
                    focus_rank_bonus +
                    all_new_useful.astype(jnp.int32) * 3 + color_bonus - q_ids -
                    save_ambiguous.astype(jnp.int32) * 100)
    forced_q = jnp.argmax(jnp.where(clue_legal, forced_score, -10000))
    has_forced_clue = jnp.any(clue_legal)

    fallback = jnp.argmax(legal).astype(jnp.int32)
    action = fallback
    action = jnp.where(has_forced_clue, 10 + forced_q, action)
    action = jnp.where(has_discard, discard_action, action)
    action = jnp.where(special_relaxed_play, best_relaxed_clue, action)
    action = jnp.where(has_play_clue, best_play_clue, action)
    information_capped = jnp.sum(f[192:200]) >= 8.0
    deck_positive = jnp.sum(f[127:167]) > 0.0
    prefer_capped_clue = information_capped & deck_positive & has_play_clue
    action = jnp.where(has_play & (~prefer_capped_clue), play_action, action)

    final_round_own_play = (~deck_positive) & has_play
    urgent_can_clue = urgent & (has_play_clue | has_save_clue) & (~final_round_own_play)
    urgent_action = jnp.where(has_play_clue, best_play_clue, best_save_clue)
    action = jnp.where(urgent_can_clue, urgent_action, action)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), action)
    action = action.astype(jnp.int32)

    trace = {
        "has_play": has_play,
        "has_play_clue": has_play_clue,
        "urgent_save": urgent,
        "intent_cards": jnp.sum(has_intent.astype(jnp.int32)),
    }
    return action, next_memory, trace
