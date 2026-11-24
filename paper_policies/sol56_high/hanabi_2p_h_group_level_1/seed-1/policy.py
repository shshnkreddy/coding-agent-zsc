"""H-group Level 1 policy for the standard two-player game."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_level1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "intent": jnp.zeros((5, 25), dtype=jnp.bool_),
        "partner_play": jnp.zeros((5,), dtype=jnp.bool_),
        "risky": jnp.zeros((5,), dtype=jnp.bool_),
        "self_clued": jnp.zeros((5,), dtype=jnp.bool_),
        "partner_clued": jnp.zeros((5,), dtype=jnp.bool_),
    }


def _shift_slots(x, slot):
    """Apply remove, left shift, and append-empty to card notes."""
    idx = jnp.arange(5, dtype=jnp.int32)
    src = jnp.minimum(idx + (idx >= slot).astype(jnp.int32), 4)
    moved = jnp.take(x, src, axis=0)
    keep = (idx < 4).reshape((5,) + (1,) * (x.ndim - 1))
    return jnp.where(keep, moved, jnp.zeros_like(moved))


def _first_true(mask):
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _last_true(mask):
    return (4 - jnp.argmax(mask[::-1].astype(jnp.int32))).astype(jnp.int32)


def _focus(match, prior_clued, occupied):
    """Native-index H-group single-card focus."""
    unclued = occupied & ~prior_clued
    has_chop = jnp.any(unclued)
    chop = _first_true(unclued)
    new = match & ~prior_clued
    has_new = jnp.any(new)
    return jnp.where(
        has_new,
        jnp.where(has_chop & match[chop], chop, _last_true(new)),
        _last_true(match),
    ).astype(jnp.int32)


def _decode(features):
    partner = features[0:125].reshape((5, 5, 5))
    deck = jnp.sum(features[127:167]).astype(jnp.int32)
    fireworks_oh = features[167:192].reshape((5, 5))
    fireworks = jnp.sum(
        fireworks_oh * jnp.arange(1, 6, dtype=fireworks_oh.dtype)[None, :],
        axis=1,
    ).astype(jnp.int32)
    info = jnp.sum(features[192:200]).astype(jnp.int32)
    lives = jnp.sum(features[200:203]).astype(jnp.int32)
    raw = features[203:253].reshape((5, 10))
    discards = jnp.stack([
        jnp.sum(raw[:, 0:3], axis=1),
        jnp.sum(raw[:, 3:5], axis=1),
        jnp.sum(raw[:, 5:7], axis=1),
        jnp.sum(raw[:, 7:9], axis=1),
        raw[:, 9],
    ], axis=1).astype(jnp.int32)
    last = features[253:308]
    knowledge = features[308:658].reshape((2, 5, 35))
    weights = knowledge[:, :, 0:25].reshape((2, 5, 5, 5))
    positive = jnp.any(knowledge[:, :, 25:35] > 0.5, axis=2)
    return partner, deck, fireworks, info, lives, discards, last, weights, positive


def _remaining_candidates(note, fireworks):
    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    return note.reshape((5, 5)) & (ranks >= fireworks[:, None])


def _note_is_safe(note, literal, fireworks):
    possible = _remaining_candidates(note, fireworks) & (literal > 0.0)
    playable = jnp.arange(5, dtype=jnp.int32)[None, :] == fireworks[:, None]
    return jnp.any(possible) & jnp.all(~possible | playable)


def _literal_safe(literal, fireworks):
    support = literal > 0.0
    playable = jnp.arange(5, dtype=jnp.int32)[None, :] == fireworks[:, None]
    return jnp.any(support) & jnp.all(~support | playable)


def _literal_trash(literal, fireworks):
    support = literal > 0.0
    obsolete = jnp.arange(5, dtype=jnp.int32)[None, :] < fireworks[:, None]
    return jnp.any(support) & jnp.all(~support | obsolete)


def _critical_mask(discards, fireworks):
    mult = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)[None, :]
    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    return (discards >= (mult - 1)) & (ranks >= fireworks[:, None])


def _guaranteed_types(own_literal, own_clued, old_intent):
    """Types represented by a literally or conventionally exact held card."""
    g = jnp.zeros((5, 5), dtype=jnp.bool_)
    literal_count = jnp.sum(own_literal > 0.0, axis=(1, 2))
    note_count = jnp.sum(old_intent, axis=1)
    for c in range(5):
        for r in range(5):
            t = 5 * c + r
            exact_lit = jnp.any(own_clued & (literal_count == 1) &
                                (own_literal[:, c, r] > 0.0))
            exact_note = jnp.any((note_count == 1) & old_intent[:, t])
            g = g.at[c, r].set(exact_lit | exact_note)
    return g


def _reachable_rank_suits(rank, fireworks, guaranteed):
    connected = jnp.ones((5,), dtype=jnp.bool_)
    for q in range(5):
        needed = (fireworks <= q) & (q < rank)
        connected = connected & (~needed | guaranteed[:, q])
    return (rank >= fireworks) & ((fireworks == rank) | connected)


def _incoming_note(last, is_chop, fireworks, discards,
                   own_literal, own_clued, old_intent,
                   partner, partner_play):
    color_hint = last[4] > 0.5
    color = jnp.argmax(last[8:13]).astype(jnp.int32)
    rank = jnp.argmax(last[13:18]).astype(jnp.int32)
    critical = _critical_mask(discards, fireworks)
    guaranteed = _guaranteed_types(own_literal, own_clued, old_intent)
    partner_connectors = jnp.any(
        partner_play[:, None, None] & (partner > 0.5), axis=0
    )
    guaranteed = guaranteed | partner_connectors

    # Direct color clue, extended only through guaranteed self-prompt cards.
    reachable = jnp.zeros((5,), dtype=jnp.bool_)
    chain = jnp.asarray(True)
    for q in range(5):
        after = q >= fireworks[color]
        reachable = reachable.at[q].set(after & chain)
        chain = chain & jnp.where(after, guaranteed[color, q], True)
    color_play = jnp.zeros((5, 5), dtype=jnp.bool_).at[color, :].set(reachable)

    suits = _reachable_rank_suits(rank, fireworks, guaranteed)
    rank_play = jnp.zeros((5, 5), dtype=jnp.bool_).at[:, rank].set(suits)

    # Chop clues retain every Level-1 play/save interpretation.
    cc = jnp.arange(5, dtype=jnp.int32)[:, None]
    rr = jnp.arange(5, dtype=jnp.int32)[None, :]
    color_save = critical & (rr < 4) & (cc == color)
    rank_save = critical & (rr == rank)
    two_save = (rr == 1) & (rr >= fireworks[:, None])
    five_save = (rr == 4) & (rr >= fireworks[:, None])
    rank_extra = jnp.where(rank == 1, two_save, rank_save)
    rank_extra = jnp.where(rank == 4, five_save, rank_extra)
    color_note = color_play | jnp.where(is_chop, color_save, False)
    rank_note = rank_play | jnp.where(is_chop, rank_extra, False)
    note = jnp.where(color_hint, color_note, rank_note)

    direct_color = jnp.zeros((5, 5), dtype=jnp.bool_).at[
        color, fireworks[color]
    ].set(True)
    fallback = jnp.where(color_hint, direct_color, rank_play)
    return jnp.where(jnp.any(note), note, fallback).reshape((25,))


def _clue_tables(partner, partner_literal, partner_clued,
                 partner_occupied, fireworks):
    card_c = jnp.argmax(jnp.sum(partner, axis=2), axis=1)
    card_r = jnp.argmax(jnp.sum(partner, axis=1), axis=1)
    card_direct = partner_occupied & (card_r == fireworks[card_c])
    card_useful = partner_occupied & (card_r >= fireworks[card_c])
    literal_count = jnp.sum(partner_literal > 0.0, axis=(1, 2))
    guaranteed = jnp.zeros((5, 5), dtype=jnp.bool_)
    for c in range(5):
        for r in range(5):
            guaranteed = guaranteed.at[c, r].set(jnp.any(
                partner_clued & (literal_count == 1)
                & (partner_literal[:, c, r] > 0.0)
            ))
    card_reachable = jnp.zeros((5,), dtype=jnp.bool_)
    connection_depth = jnp.zeros((5,), dtype=jnp.int32)
    for s in range(5):
        connected = jnp.asarray(True)
        for q in range(5):
            needed = (fireworks[card_c[s]] <= q) & (q < card_r[s])
            connected = connected & (~needed | guaranteed[card_c[s], q])
        reachable = card_useful[s] & connected
        card_reachable = card_reachable.at[s].set(reachable)
        connection_depth = connection_depth.at[s].set(
            jnp.where(reachable, card_r[s] - fireworks[card_c[s]], 0)
        )
    scores, focuses, focus_plays, new_counts = [], [], [], []
    chop_mask = partner_occupied & ~partner_clued
    chop = _first_true(chop_mask)
    has_chop = jnp.any(chop_mask)
    for k in range(10):
        match = partner_occupied & jnp.where(k < 5, card_c == k, card_r == k - 5)
        focus = _focus(match, partner_clued, partner_occupied)
        new = match & ~partner_clued
        nnew = jnp.sum(new.astype(jnp.int32))
        bad = jnp.sum((new & ~card_useful).astype(jnp.int32))
        same_identity = (
            partner.reshape((5, 25)) @ partner.reshape((5, 25)).T
        ) > 0.5
        distinct_slots = ~jnp.eye(5, dtype=jnp.bool_)
        duplicate_touch = jnp.any(
            same_identity & distinct_slots & match[:, None] & match[None, :]
        )
        is_play = card_reachable[focus]
        ambiguous = has_chop & (focus == chop) & ((k == 6) | (k == 9))
        clean_touch = (k != 5) | ~duplicate_touch
        valid = (jnp.any(match) & (nnew > 0) & is_play & (bad == 0)
                 & ~ambiguous & clean_touch)
        score = jnp.where(
            valid,
            100 + 7 * nnew + 12 * connection_depth[focus]
            + 15 * jnp.maximum(nnew - 1, 0) * (k == 5)
            + jnp.where(k < 5, 3, 0),
            -1000,
        )
        scores.append(score)
        focuses.append(focus)
        focus_plays.append(card_direct[focus])
        new_counts.append(nnew)
    return (jnp.stack(scores), jnp.stack(focuses),
            jnp.stack(focus_plays), jnp.stack(new_counts),
            card_c, card_r, chop, has_chop)


def agent_act(obs, reset, memory, rng):
    del rng
    legal = obs.legal_actions
    (partner, deck, fireworks, info, lives, discards, last,
     weights, positive) = _decode(obs.features)
    own_literal, partner_literal = weights[0], weights[1]
    own_occ = jnp.sum(own_literal, axis=(1, 2)) > 0.0
    partner_occ = jnp.sum(partner, axis=(1, 2)) > 0.0

    intent = jnp.where(reset, jnp.zeros_like(memory["intent"]), memory["intent"])
    partner_play = jnp.where(reset, jnp.zeros_like(memory["partner_play"]),
                             memory["partner_play"])
    risky = jnp.where(reset, jnp.zeros_like(memory["risky"]), memory["risky"])
    old_self_clued = jnp.where(reset, positive[0], memory["self_clued"])

    actor_self, actor_partner = last[0] > 0.5, last[1] > 0.5
    removed_action = (last[2] > 0.5) | (last[3] > 0.5)
    removed_slot = jnp.argmax(last[23:28]).astype(jnp.int32)
    own_removed = (~reset) & actor_self & removed_action
    partner_removed = (~reset) & actor_partner & removed_action
    intent = jnp.where(own_removed, _shift_slots(intent, removed_slot), intent)
    risky = jnp.where(own_removed, _shift_slots(risky, removed_slot), risky)
    partner_play = jnp.where(partner_removed,
                             _shift_slots(partner_play, removed_slot), partner_play)

    incoming = ((~reset) & actor_partner &
                ((last[4] > 0.5) | (last[5] > 0.5)) & (last[6] > 0.5))
    match = last[18:23] > 0.5
    focus = _focus(match, old_self_clued, own_occ)
    pre_unclued = own_occ & ~old_self_clued
    chop = _first_true(pre_unclued)
    is_chop = jnp.any(pre_unclued) & (focus == chop)
    incoming_note = _incoming_note(
        last, is_chop, fireworks, discards,
        own_literal, old_self_clued, intent, partner, partner_play
    )
    any_new_touch = jnp.any(match & ~old_self_clued)
    forced_no_new = incoming & (info == 7) & ~any_new_touch
    focus_literal_safe = _literal_safe(own_literal[focus], fireworks)
    # A clue from eight tokens may be forced.  An unfilled retouch which did
    # not make the focused card literally safe is not enough evidence to risk
    # a life; preserve the old note and wait for clarification.
    accept_incoming = incoming & ~(forced_no_new & ~focus_literal_safe)
    intent = intent.at[focus].set(
        jnp.where(accept_incoming, incoming_note, intent[focus])
    )
    risky = risky.at[focus].set(
        jnp.where(incoming, (info == 7) & ~focus_literal_safe, risky[focus])
    )

    # On a non-chop rank-1 play clue, every newly touched nonfocus card is a
    # future-useful 1 by Good Touch.  Distinct suits can therefore be played
    # in H-leftmost order as their literal/conventional candidates become safe.
    incoming_rank = jnp.argmax(last[13:18]).astype(jnp.int32)
    cascade = incoming & (last[5] > 0.5) & (incoming_rank == 0) & ~is_chop
    playable_ones = (
        (fireworks == 0)[:, None]
        & (jnp.arange(5, dtype=jnp.int32)[None, :] == 0)
    ).reshape((25,))
    for s in range(5):
        add = cascade & match[s] & ~old_self_clued[s] & (focus != s)
        intent = intent.at[s].set(jnp.where(add, playable_ones, intent[s]))
        risky = risky.at[s].set(jnp.where(add, info == 7, risky[s]))

    literal_safe = jnp.stack([
        _literal_safe(own_literal[s], fireworks) for s in range(5)
    ])
    noted_safe = jnp.stack([
        _note_is_safe(intent[s], own_literal[s], fireworks) for s in range(5)
    ])
    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    obsolete_support = jnp.stack([
        jnp.any(
            intent[s].reshape((5, 5))
            & (own_literal[s] > 0.0)
            & (ranks < fireworks[:, None])
        )
        for s in range(5)
    ])
    noted_safe = noted_safe & (
        (lives > 1) | (~obsolete_support & ~risky)
    )
    safe_play = own_occ & (literal_safe | noted_safe)
    has_noted_play = jnp.any(own_occ & noted_safe)
    has_literal_play = jnp.any(own_occ & literal_safe)
    prefer_noted = has_noted_play & ((deck > 0) | ~has_literal_play)
    play_slot = jnp.where(
        prefer_noted,
        _last_true(own_occ & noted_safe),
        _last_true(own_occ & literal_safe),
    ).astype(jnp.int32)

    literal_trash = jnp.stack([
        _literal_trash(own_literal[s], fireworks) for s in range(5)
    ])
    trash_slot = _first_true(own_occ & literal_trash)
    has_trash = jnp.any(own_occ & literal_trash)
    own_unclued = own_occ & ~positive[0]
    own_chop = _first_true(own_unclued)
    has_own_chop = jnp.any(own_unclued)

    (clue_scores, clue_focuses, clue_focus_playable, clue_new_counts,
     pc, pr, partner_chop, has_partner_chop) = _clue_tables(
        partner, partner_literal, positive[1], partner_occ, fireworks
    )
    clue_legal = legal[10:20]
    playable_scores = jnp.where(clue_legal, clue_scores, -1000)
    best_play_k = jnp.argmax(playable_scores).astype(jnp.int32)
    has_play_clue = playable_scores[best_play_k] > 0
    best_play_action = 10 + best_play_k

    chop_c, chop_r = pc[partner_chop], pr[partner_chop]
    same_visible = jnp.sum(partner[:, chop_c, chop_r] > 0.5).astype(jnp.int32)
    critical = _critical_mask(discards, fireworks)[chop_c, chop_r]
    chop_useful = has_partner_chop & (chop_r >= fireworks[chop_c])
    needs_five = chop_useful & (chop_r == 4)
    needs_two = chop_useful & (chop_r == 1) & (same_visible == 1)
    needs_critical = has_partner_chop & critical & (chop_r != 4)
    chop_playable = has_partner_chop & (chop_r == fireworks[chop_c])
    urgent = needs_five | needs_two | needs_critical | chop_playable

    save5_action = jnp.asarray(19, dtype=jnp.int32)
    save2_action = jnp.asarray(16, dtype=jnp.int32)
    color_action, rank_action = 10 + chop_c, 15 + chop_r
    color_match = partner_occ & (pc == chop_c)
    rank_match = partner_occ & (pr == chop_r)
    color_bad = jnp.sum((color_match & (pr < fireworks[pc])).astype(jnp.int32))
    rank_bad = jnp.sum((rank_match & (pr < fireworks[pc])).astype(jnp.int32))
    critical_action = jnp.where(color_bad <= rank_bad, color_action, rank_action)
    save_action = jnp.where(
        needs_five, save5_action,
        jnp.where(needs_two, save2_action, critical_action)
    ).astype(jnp.int32)
    save_legal = legal[save_action]

    best_play_focus_is_chop = (
        has_partner_chop & (clue_focuses[best_play_k] == partner_chop)
    )
    has_own_play = jnp.any(safe_play)
    final_round = deck == 0
    immediate_best_clue = clue_focus_playable[best_play_k]
    usable_play_clue = has_play_clue & (~final_round | immediate_best_clue)
    choose_play_clue_first = (
        urgent & usable_play_clue & (~final_round | ~has_own_play)
    )
    choose_save = (
        urgent & ~final_round & ~choose_play_clue_first & save_legal
    )

    discard_slot = jnp.where(has_trash, trash_slot, own_chop).astype(jnp.int32)
    can_discard = has_trash | has_own_chop
    discard_legal = can_discard & legal[discard_slot]
    optional_play_clue = usable_play_clue

    # At full information, a genuine tempo clue to an already-clued playable
    # card is the safest forced clue.  Otherwise minimize new touches and use a
    # rank clue: unlike a color clue, an unsupported future rank does not look
    # like an immediate specific card.
    tempo = (clue_new_counts == 0) & clue_focus_playable
    clue_ids = jnp.arange(10, dtype=jnp.int32)
    rank_clue = clue_ids >= 5
    rank_ids = jnp.maximum(clue_ids - 5, 0)
    no_direct_suit = ~jnp.any(
        fireworks[None, :] == rank_ids[:, None], axis=1
    )
    forced_focus_c = pc[clue_focuses]
    forced_focus_r = pr[clue_focuses]
    forced_focus_useful = (
        partner_occ[clue_focuses]
        & (forced_focus_r >= fireworks[forced_focus_c])
    )
    nonreactive = rank_clue & no_direct_suit
    force_scores = jnp.where(
        clue_legal,
        1000 * tempo.astype(jnp.int32)
        + 500 * (nonreactive & forced_focus_useful).astype(jnp.int32)
        + 200 * nonreactive.astype(jnp.int32)
        + 20 * forced_focus_useful.astype(jnp.int32)
        - 10 * clue_new_counts
        + 2 * rank_clue.astype(jnp.int32),
        -1000,
    )
    forced_k = jnp.argmax(force_scores).astype(jnp.int32)
    forced_action = 10 + forced_k
    any_clue = jnp.any(clue_legal)

    action = jnp.where(
        choose_play_clue_first, best_play_action,
        jnp.where(
            choose_save, save_action,
            jnp.where(
                has_own_play, 5 + play_slot,
                jnp.where(
                    optional_play_clue, best_play_action,
                    jnp.where(
                        discard_legal, discard_slot,
                        jnp.where(any_clue, forced_action, jnp.argmax(legal))
                    )
                )
            )
        )
    ).astype(jnp.int32)
    active = ~legal[20]
    action = jnp.where(active, action, jnp.asarray(20, dtype=jnp.int32))
    action = jnp.where(legal[action], action, jnp.argmax(legal)).astype(jnp.int32)

    gave_play = (active & (action == best_play_action) & has_play_clue)
    out_focus = clue_focuses[best_play_k]
    gave_unambiguous_play = gave_play & ~(
        has_partner_chop & (out_focus == partner_chop)
    )
    partner_play = partner_play.at[out_focus].set(
        jnp.where(gave_unambiguous_play, True, partner_play[out_focus])
    )
    tempo_focus = clue_focuses[forced_k]
    gave_unambiguous_tempo = (
        active & any_clue & (action == forced_action) & tempo[forced_k]
        & ~(has_partner_chop & (tempo_focus == partner_chop))
    )
    partner_play = partner_play.at[tempo_focus].set(
        jnp.where(gave_unambiguous_tempo, True, partner_play[tempo_focus])
    )
    next_memory = {
        "intent": intent,
        "partner_play": partner_play,
        "risky": risky,
        "self_clued": positive[0],
        "partner_clued": positive[1],
    }
    trace = {
        "safe_play": safe_play,
        "intent": intent,
        "urgent_chop": urgent,
        "chosen": action,
        "fireworks": fireworks,
        "lives": lives,
        "information": info,
        "last_type": last[2:6],
        "last_identity": last[28:53],
        "last_play_success": last[53],
        "incoming": incoming,
        "incoming_focus": focus,
        "incoming_is_chop": is_chop,
        "last_match": match,
        "old_self_clued": old_self_clued,
        "current_self_clued": positive[0],
        "risky_note": risky,
    }
    return action, next_memory, trace
