"""A conservative two-player H-group Level 1 Hanabi policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "hgroup_level1_conservative",
    "adapter": "hanabi",
    "adapter_version": 1,
    "num_agents": 2,
    "num_colors": 5,
    "num_ranks": 5,
    "hand_size": 5,
    "observation_size": 658,
    "num_actions": 21,
    "policy_structure": "shared_local",
}


def init_agent():
    # state: 0 = untouched, 1 = play promise, 2 = protected/ambiguous.
    return {
        "state": jnp.zeros((5,), dtype=jnp.int32),
        "intent": jnp.zeros((5, 25), dtype=jnp.bool_),
        "clued": jnp.zeros((5,), dtype=jnp.bool_),
    }


def _heights(features):
    rows = features[167:192].reshape(5, 5)
    return jnp.sum(rows * jnp.arange(1, 6, dtype=jnp.float32)[None, :], axis=1).astype(jnp.int32)


def _discard_counts(features):
    raw = features[203:253].reshape(5, 10)
    return jnp.stack(
        [
            jnp.sum(raw[:, 0:3], axis=1),
            jnp.sum(raw[:, 3:5], axis=1),
            jnp.sum(raw[:, 5:7], axis=1),
            jnp.sum(raw[:, 7:9], axis=1),
            raw[:, 9],
        ],
        axis=1,
    ).astype(jnp.int32)


def _highest(mask):
    return (4 - jnp.argmax(mask[::-1].astype(jnp.int32))).astype(jnp.int32)


def _lowest(mask):
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _shift_slots(values, removed):
    slots = jnp.arange(5, dtype=jnp.int32)
    source = jnp.minimum(slots + (slots >= removed).astype(jnp.int32), 4)
    moved = values[source]
    clear_shape = (5,) + (1,) * (values.ndim - 1)
    keep = (slots < 4).reshape(clear_shape)
    return jnp.where(keep, moved, jnp.zeros_like(moved))


def _decode_knowledge(features):
    knowledge = features[308:658].reshape(2, 5, 35)
    v0 = knowledge[:, :, :25]
    positive = jnp.any(knowledge[:, :, 25:35] > 0.5, axis=2)
    occupied = jnp.sum(v0, axis=2) > 0.0
    return v0, positive, occupied


def _refined_own(features, own_v0, own_occupied, heights, discards):
    """Public card-count refinement, including fixed-point singleton use."""
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)
    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    visible = features[0:125].reshape(5, 5, 5)
    visible_counts = jnp.sum(visible > 0.5, axis=0).astype(jnp.int32)
    played_types = ranks < heights[:, None]
    remaining = multiplicity[None, :] - discards - played_types.astype(jnp.int32)
    unseen = jnp.maximum(remaining - visible_counts, 0).reshape(25)
    possible = (own_v0 > 0.0) & (unseen[None, :] > 0) & own_occupied[:, None]
    pair_i = jnp.arange(25, dtype=jnp.int32)[:, None]
    pair_j = jnp.arange(25, dtype=jnp.int32)[None, :]
    for _ in range(5):
        singleton = own_occupied & (jnp.sum(possible, axis=1) == 1)
        reserved = jnp.sum(possible & singleton[:, None], axis=0).astype(jnp.int32)
        room_for_other = unseen > reserved
        possible = possible & (singleton[:, None] | room_for_other[None, :])

        # Hall elimination for all 300 unordered identity pairs. If every
        # unseen copy of a pair must occupy slots whose possibility sets are
        # contained in that pair, neither identity can occur in another slot.
        total = jnp.sum(possible, axis=1).astype(jnp.int32)
        inside = (
            possible[:, :, None].astype(jnp.int32)
            + possible[:, None, :].astype(jnp.int32)
        ).transpose(1, 2, 0)
        subset = (
            own_occupied[None, None, :]
            & (total[None, None, :] > 0)
            & (inside == total[None, None, :])
        )
        pair_capacity = unseen[:, None] + unseen[None, :]
        saturated = (
            (pair_i < pair_j)
            & (jnp.sum(subset, axis=2) > 0)
            & (jnp.sum(subset, axis=2) == pair_capacity)
        )
        eliminate_first = jnp.any(saturated[:, :, None] & (~subset), axis=1).T
        eliminate_second = jnp.any(saturated[:, :, None] & (~subset), axis=0).T
        possible = possible & (~(eliminate_first | eliminate_second))

    # Exact capacitated bipartite edge feasibility. For each proposed
    # (slot, identity), reserve one copy and apply Hall's condition to all 32
    # subsets of the other four slots. With only five cards this is small and
    # fully vectorized, while accounting for every joint count constraint.
    subset_bits = (
        (jnp.arange(32, dtype=jnp.int32)[:, None] >> jnp.arange(5, dtype=jnp.int32)[None, :])
        & 1
    ).astype(jnp.bool_)
    candidate_slot = jnp.arange(5, dtype=jnp.int32)[:, None, None]
    slot_axis = jnp.arange(5, dtype=jnp.int32)[None, None, :]
    selected = (
        subset_bits[None, :, :]
        & own_occupied[None, None, :]
        & (slot_axis != candidate_slot)
    )
    union_types = jnp.any(selected[:, :, :, None] & possible[None, None, :, :], axis=2)
    capacity_after_reserve = unseen[None, :] - jnp.eye(25, dtype=jnp.int32)
    available = jnp.einsum(
        "squ,tu->stq", union_types.astype(jnp.int32), capacity_after_reserve
    )
    required = jnp.sum(selected, axis=2).astype(jnp.int32)
    edge_feasible = jnp.all(available >= required[:, None, :], axis=2)
    possible = possible & edge_feasible
    return possible


def _update_memory(features, reset, memory):
    v0, positive, occupied = _decode_knowledge(features)
    last = features[253:308]
    actor_self = last[0] > 0.5
    actor_partner = last[1] > 0.5
    is_play = last[2] > 0.5
    is_discard = last[3] > 0.5
    is_color = last[4] > 0.5
    is_rank = last[5] > 0.5
    own_remove = actor_self & (is_play | is_discard)
    removed = jnp.argmax(last[23:28]).astype(jnp.int32)

    state0 = jnp.where(reset, jnp.zeros_like(memory["state"]), memory["state"])
    intent0 = jnp.where(reset, jnp.zeros_like(memory["intent"]), memory["intent"])
    clued0 = jnp.where(reset, jnp.zeros_like(memory["clued"]), memory["clued"])
    shifted_state = _shift_slots(state0, removed)
    shifted_intent = _shift_slots(intent0, removed)
    shifted_clued = _shift_slots(clued0, removed)
    state1 = jnp.where(own_remove, shifted_state, state0)
    intent1 = jnp.where(own_remove, shifted_intent, intent0)
    previous_clued = jnp.where(own_remove, shifted_clued, clued0)

    receives_hint = actor_partner & (is_color | is_rank)
    matches = last[18:23] > 0.5
    new = matches & (~previous_clued) & occupied[0]
    has_new = jnp.any(new)
    unclued_pre = occupied[0] & (~previous_clued)
    has_chop = jnp.any(unclued_pre)
    chop = _lowest(unclued_pre)
    focus = jnp.where(
        has_new,
        jnp.where(has_chop & matches[chop], chop, _highest(new)),
        _highest(matches),
    )

    heights = _heights(features)
    discards = _discard_counts(features)
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)
    colors = jnp.arange(5, dtype=jnp.int32)[:, None]
    ranks = jnp.arange(5, dtype=jnp.int32)[None, :]
    color_named = jnp.argmax(last[8:13]).astype(jnp.int32)
    rank_named = jnp.argmax(last[13:18]).astype(jnp.int32)
    own_literal = _refined_own(features, v0[0], occupied[0], heights, discards)
    literal_focus = own_literal[focus]

    play_by_color = (colors == color_named) & (ranks == heights[:, None])
    play_by_rank = (ranks == rank_named) & (heights[:, None] == rank_named)
    play_mask = jnp.where(is_color, play_by_color, play_by_rank).reshape(25)

    # Rank 5 has its own canonical rank clue; do not turn every color clue that
    # could include the single-copy 5 into an apparent generic critical save.
    critical = (ranks < 4) & (ranks >= heights[:, None]) & (
        discards >= (multiplicity[None, :] - 1)
    )
    save_color = (colors == color_named) & critical
    # The giver cannot see their own hand, so a matching copy visible there to
    # the receiver cannot rule out that the giver intended a canonical 2 Save.
    two_save = ranks == 1
    five_save = ranks == 4
    save_rank = (ranks == rank_named) & (
        critical | (((rank_named == 1) & two_save) | ((rank_named == 4) & five_save))
    )
    save_mask = jnp.where(is_color, save_color, save_rank).reshape(25)
    on_chop = has_chop & (focus == chop)
    conventional = play_mask | jnp.where(on_chop, save_mask, jnp.zeros_like(save_mask))
    inferred = conventional & literal_focus
    inferred = jnp.where(jnp.any(inferred), inferred, literal_focus)

    is_save_clue = on_chop & jnp.any(save_mask & literal_focus)
    protected_state = jnp.where(new, jnp.maximum(state1, 2), state1)
    protected_intent = jnp.where(new[:, None], own_literal, intent1)
    focus_state = jnp.where(is_save_clue, 2, 1).astype(jnp.int32)
    hinted_state = protected_state.at[focus].set(focus_state)
    hinted_intent = protected_intent.at[focus].set(inferred)

    # A Level-1 self-prompt can be established either by an unambiguous rank
    # clue, or by a color clue that retouches an older clued playable connector
    # while newly touching its same-color successor. Prompts take precedence
    # and use the leftmost compatible card.
    clear_prompt_rank = is_rank & (rank_named > 0) & (~jnp.any(heights == rank_named))
    rank_connector_types = (
        (ranks == (rank_named - 1)) & (heights[:, None] == (rank_named - 1))
    ).reshape(25)
    color_connector_types = (
        (colors == color_named) & (ranks == heights[:, None])
    ).reshape(25)
    connector_types = jnp.where(is_color, color_connector_types, rank_connector_types)
    connector_support = jnp.any(own_literal & connector_types[None, :], axis=1)
    connector_certain = connector_support & jnp.all(
        (~own_literal) | connector_types[None, :], axis=1
    )
    connector_slots = previous_clued & occupied[0] & jnp.where(
        is_color, connector_certain, connector_support
    )
    has_connector = jnp.any(connector_slots)
    prompt_slot = _highest(connector_slots)
    connector_intent = own_literal[prompt_slot] & connector_types
    connector_colors = jnp.any(connector_intent.reshape(5, 5), axis=1)
    rank_prompt_focus_types = ((ranks == rank_named) & connector_colors[:, None]).reshape(25)
    color_prompt_focus_types = (
        (colors == color_named) & (ranks == (heights[:, None] + 1))
    ).reshape(25)
    prompt_focus_types = jnp.where(is_color, color_prompt_focus_types, rank_prompt_focus_types)
    prompt_focus_intent = literal_focus & prompt_focus_types
    rank_self_prompt = (
        clear_prompt_rank
        & (jnp.sum(features[192:200]) != 7)
        & has_new
        & has_connector
        & (~is_save_clue)
        & jnp.any(prompt_focus_intent)
    )
    color_self_prompt = (
        is_color
        & (heights[color_named] < 4)
        & has_new
        & has_connector
        & (~is_save_clue)
        & jnp.any(prompt_focus_intent)
    )
    self_prompt = rank_self_prompt | color_self_prompt
    prompted_state = hinted_state.at[prompt_slot].set(jnp.asarray(1, dtype=jnp.int32))
    prompted_intent = hinted_intent.at[prompt_slot].set(connector_intent)
    prompted_intent = prompted_intent.at[focus].set(prompt_focus_intent)
    hinted_state = jnp.where(self_prompt, prompted_state, hinted_state)
    hinted_intent = jnp.where(self_prompt, prompted_intent, hinted_intent)
    state2 = jnp.where(receives_hint, hinted_state, state1)
    intent2 = jnp.where(receives_hint, hinted_intent, intent1)
    state2 = jnp.where(occupied[0], state2, jnp.zeros_like(state2))
    intent2 = jnp.where(occupied[0, :, None], intent2, jnp.zeros_like(intent2))
    return {
        "state": state2,
        "intent": intent2,
        "clued": positive[0] & occupied[0],
    }


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    mem = _update_memory(f, reset, memory)
    v0, positive, occupied = _decode_knowledge(f)
    heights = _heights(f)
    discards = _discard_counts(f)
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)
    info = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)

    type_colors = jnp.repeat(jnp.arange(5), 5)
    type_ranks = jnp.tile(jnp.arange(5), 5)
    type_playable = type_ranks == heights[type_colors]
    type_trash = type_ranks < heights[type_colors]

    partner_cards = f[0:125].reshape(5, 5, 5)
    own_possible = _refined_own(f, v0[0], occupied[0], heights, discards)
    literal_safe_play = jnp.any(own_possible, axis=1) & jnp.all((~own_possible) | type_playable[None, :], axis=1)
    literal_trash = jnp.any(own_possible, axis=1) & jnp.all((~own_possible) | type_trash[None, :], axis=1)
    # Re-intersect historical convention notes with the slot's current literal
    # constraints. Requiring a surviving positive clue prevents a stale note
    # from ever authorizing a freshly drawn card if bookkeeping is disrupted.
    intent_possible = mem["intent"] & own_possible
    promised_safe = (mem["state"] == 1) & mem["clued"] & jnp.any(intent_possible, axis=1) & jnp.all(
        (~intent_possible) | type_playable[None, :], axis=1
    )
    safe_play = occupied[0] & (literal_safe_play | promised_safe) & legal[5:10]
    play_priority = safe_play.astype(jnp.int32) * (
        10 + mem["state"] * 5 + jnp.arange(5)
    )
    play_slot = jnp.argmax(play_priority).astype(jnp.int32)
    has_play = jnp.any(safe_play)

    partner_occ = occupied[1]
    partner_clued = positive[1]
    p_unclued = partner_occ & (~partner_clued)
    has_pchop = jnp.any(p_unclued)
    pchop = _lowest(p_unclued)
    pcolor = jnp.argmax(jnp.sum(partner_cards, axis=2), axis=1).astype(jnp.int32)
    prank = jnp.argmax(jnp.sum(partner_cards, axis=1), axis=1).astype(jnp.int32)
    pplayable = partner_occ & (prank == heights[pcolor])
    ptrash = partner_occ & (prank < heights[pcolor])
    partner_possible = v0[1] > 0.0
    partner_literal_safe = partner_occ & jnp.any(partner_possible, axis=1) & jnp.all(
        (~partner_possible) | type_playable[None, :], axis=1
    )

    # Score all ten clues. A valid play clue must focus a currently playable
    # card and avoid newly touching already-useless cards.
    clue_score = jnp.full((10,), -10000, dtype=jnp.int32)
    chop_clue_score = jnp.full((10,), -10000, dtype=jnp.int32)
    for k in range(10):
        is_c = k < 5
        value = k if is_c else k - 5
        match = partner_occ & (pcolor == value) if is_c else partner_occ & (prank == value)
        new = match & (~partner_clued)
        has_new = jnp.any(new)
        partner_type = pcolor * 5 + prank
        clued_after = partner_clued | new
        same_type_count = jnp.sum(
            clued_after[None, :]
            & (partner_type[:, None] == partner_type[None, :]),
            axis=1,
        )
        creates_duplicate = jnp.any(new & (same_type_count > 1))
        focus = jnp.where(
            has_new,
            jnp.where(has_pchop & match[pchop], pchop, _highest(new)),
            _highest(match),
        )
        touches_trash = jnp.any(new & ptrash)
        focus_plays = pplayable[focus]
        chop_save_rank = (not is_c) and (value == 1 or value == 4)
        semantically_play = ~(jnp.asarray(chop_save_rank) & has_pchop & (focus == pchop))
        focus_color = pcolor[focus]
        attribute_types = (
            (type_colors == value) if is_c else (type_ranks == value)
        )
        after_hint = partner_possible & jnp.where(
            match[:, None], attribute_types[None, :], (~attribute_types)[None, :]
        )
        clear_prompt = jnp.asarray(not is_c) & (value > 0) & (~jnp.any(heights == value))
        rank_connector_types = (
            (type_ranks == (value - 1))
            & (heights[type_colors] == (value - 1))
        )
        rank_connector_candidates = partner_clued & jnp.any(
            partner_possible & rank_connector_types[None, :], axis=1
        )
        rank_connectors_aligned = jnp.any(rank_connector_candidates) & jnp.all(
            (~rank_connector_candidates)
            | (pplayable & (pcolor == focus_color) & (prank == (value - 1)))
        )
        has_exact_color_connector = jnp.any(
            partner_clued
            & jnp.any(
                after_hint
                & ((type_colors == value) & (type_ranks == heights[value]))[None, :],
                axis=1,
            )
            & jnp.all(
                (~after_hint)
                | ((type_colors == value) & (type_ranks == heights[value]))[None, :],
                axis=1,
            )
        )
        self_prompt_clue = (
            clear_prompt
            & has_new
            & (prank[focus] == value)
            & rank_connectors_aligned
            & semantically_play
        )
        color_focus_is_save = (
            jnp.asarray(is_c)
            & has_pchop
            & (focus == pchop)
            & (prank[focus] < 4)
            & (prank[focus] >= heights[pcolor[focus]])
            & (discards[pcolor[focus], prank[focus]] >= multiplicity[prank[focus]] - 1)
        )
        color_self_prompt_clue = (
            jnp.asarray(is_c)
            & has_new
            & (heights[value] < 4)
            & (pcolor[focus] == value)
            & (prank[focus] == heights[value] + 1)
            & has_exact_color_connector
            & (~color_focus_is_save)
        )
        self_prompt_clue = self_prompt_clue | color_self_prompt_clue
        safe_after_hint = partner_occ & jnp.any(after_hint, axis=1) & jnp.all(
            (~after_hint) | type_playable[None, :], axis=1
        )
        newly_filled = jnp.sum(
            partner_clued & (~partner_literal_safe) & safe_after_hint
        ).astype(jnp.int32)
        legal_hint = legal[10 + k]
        score = (
            100
            + jnp.sum(new & pplayable).astype(jnp.int32) * 8
            - jnp.sum(new & (~pplayable)).astype(jnp.int32) * 12
            + newly_filled * 12
            + jnp.asarray(5 if is_c else 0, dtype=jnp.int32)
            - jnp.sum(match & partner_clued).astype(jnp.int32)
        )
        valid_play = has_new & focus_plays & semantically_play
        valid = legal_hint & (~touches_trash) & (valid_play | self_prompt_clue)
        scored = jnp.where(self_prompt_clue & (~valid_play), 95, score)
        clue_score = clue_score.at[k].set(jnp.where(valid, scored, -10000))
        chop_clue_score = chop_clue_score.at[k].set(
            jnp.where(
                valid & (~creates_duplicate) & has_pchop & (focus == pchop),
                scored,
                -10000,
            )
        )

    # Urgent chop saves override ordinary action ordering. Use the canonical
    # rank clue for 5/2 saves; critical saves select the least disruptive clue.
    cc = pcolor[pchop]
    cr = prank[pchop]
    chop_playable = pplayable[pchop]
    copies_visible = jnp.sum(partner_cards[:, cc, cr] > 0.5)
    is_critical = (cr >= heights[cc]) & (discards[cc, cr] >= multiplicity[cr] - 1)
    is_two_save = (cr == 1) & (copies_visible == 1)
    is_five_save = cr == 4
    needs_save = has_pchop & (~chop_playable) & (is_critical | is_two_save | is_five_save)
    color_action = 10 + cc
    rank_action = 15 + cr
    color_matches = jnp.sum(partner_occ & (pcolor == cc)).astype(jnp.int32)
    rank_matches = jnp.sum(partner_occ & (prank == cr)).astype(jnp.int32)
    critical_action = jnp.where(color_matches <= rank_matches, color_action, rank_action)
    save_action = jnp.where(is_five_save | is_two_save, rank_action, critical_action).astype(jnp.int32)
    save_legal = needs_save & legal[save_action]
    best_chop_k = jnp.argmax(chop_clue_score).astype(jnp.int32)
    has_valid_chop_clue = chop_clue_score[best_chop_k] > 0
    best_chop_action = (10 + best_chop_k).astype(jnp.int32)
    remaining_chop_copies = multiplicity[cr] - discards[cc, cr]
    unique_chop_play = remaining_chop_copies <= 1
    urgent_chop_play = (
        has_pchop
        & chop_playable
        & (~jnp.any(partner_literal_safe))
        & (has_valid_chop_clue | (unique_chop_play & legal[color_action]))
    )
    urgent_chop_action = jnp.where(
        has_valid_chop_clue, best_chop_action, color_action
    ).astype(jnp.int32)

    best_clue_k = jnp.argmax(clue_score).astype(jnp.int32)
    best_clue_score = clue_score[best_clue_k]
    has_play_clue = best_clue_score > 0
    best_clue_action = (10 + best_clue_k).astype(jnp.int32)

    # Discard a known useless card first, otherwise the native chop (lowest
    # occupied unclued index). Protected convention notes are never chop.
    known_useless = occupied[0] & literal_trash & legal[0:5]
    useless_slot = _highest(known_useless)
    unclued_own = occupied[0] & (~mem["clued"]) & (mem["state"] == 0) & legal[0:5]
    has_unclued = jnp.any(unclued_own)
    chop_slot = _lowest(unclued_own)
    discard_slot = jnp.where(jnp.any(known_useless), useless_slot, chop_slot).astype(jnp.int32)
    can_discard = legal[discard_slot] & (jnp.any(known_useless) | has_unclued)

    # At eight clues a discard is illegal. Prefer a minimally disruptive legal
    # hint; if needed, repeat a clue on an actually playable partner card.
    fallback_hint_scores = jnp.full((10,), -10000, dtype=jnp.int32)
    for k in range(10):
        is_c = k < 5
        value = k if is_c else k - 5
        match = partner_occ & (pcolor == value) if is_c else partner_occ & (prank == value)
        new = match & (~partner_clued)
        has_new = jnp.any(new)
        focus = jnp.where(
            has_new,
            jnp.where(has_pchop & match[pchop], pchop, _highest(new)),
            _highest(match),
        )
        repeat_ok = legal[10 + k] & jnp.any(match)
        productive = (~has_new) & pplayable[focus]
        # If no color currently needs the named rank, a rank clue cannot be
        # mistaken for an immediate play clue. It is the safest unavoidable
        # clue when discarding is prohibited at eight tokens.
        safe_rank_clue = jnp.asarray(not is_c) & (~jnp.any(heights == value))
        repeat_score = (
            jnp.where(productive, 100, 0)
            + jnp.where((~productive) & safe_rank_clue, 60, 0)
            + jnp.where((~has_new) & (~productive), 20, 0)
            - jnp.where(has_new, 10, 0)
            - jnp.sum(match).astype(jnp.int32)
        )
        fallback_hint_scores = fallback_hint_scores.at[k].set(jnp.where(repeat_ok, repeat_score, -10000))
    fallback_k = jnp.argmax(fallback_hint_scores).astype(jnp.int32)
    fallback_hint = (10 + fallback_k).astype(jnp.int32)
    has_productive_repeat = fallback_hint_scores[fallback_k] > 80
    endgame_fill = (deck <= 20) & has_productive_repeat

    # A hand can exceptionally become fully protected with no playable card.
    # There is then no ordinary chop. If a productive fill-in is unavailable,
    # cycle the least risky card instead of stalling indefinitely. This branch
    # is outside normal play and prefers literal trash while strongly avoiding
    # 5s and public last copies.
    essential_type = (type_ranks >= heights[type_colors]) & (
        (type_ranks == 4)
        | (discards[type_colors, type_ranks] >= multiplicity[type_ranks] - 1)
    )
    refined_weights = jnp.where(own_possible, v0[0], 0.0)
    essential_mass = jnp.sum(refined_weights * essential_type[None, :], axis=1)
    trash_mass = jnp.sum(refined_weights * type_trash[None, :], axis=1)
    forced_discard_score = (
        trash_mass * 100.0 - essential_mass * 1000.0 - jnp.arange(5, dtype=jnp.float32)
    )
    forced_discard_score = jnp.where(occupied[0] & legal[0:5], forced_discard_score, -1.0e6)
    forced_discard = jnp.argmax(forced_discard_score).astype(jnp.int32)
    can_forced_discard = legal[forced_discard] & occupied[0, forced_discard]

    # Protect a threatened chop first. Otherwise discharge our own promises,
    # then create a new partner play, then discard predictably.
    action = jnp.where(
        urgent_chop_play,
        urgent_chop_action,
        jnp.where(
            save_legal,
            save_action,
            jnp.where(
                has_play,
                5 + play_slot,
                jnp.where(
                    has_play_clue,
                    best_clue_action,
                    jnp.where(
                        endgame_fill,
                        fallback_hint,
                        jnp.where(
                            can_discard,
                            discard_slot,
                            jnp.where(
                                has_productive_repeat,
                                fallback_hint,
                                jnp.where(can_forced_discard, forced_discard, fallback_hint),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), action)
    # Final legality guard is intentionally deterministic and only reachable if
    # an unexpected observation defeats the strategy's candidate construction.
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    trace = {
        "state": mem["state"],
        "safe_play": safe_play,
        "clue_score": clue_score,
        "save": save_legal,
        "chosen": action,
        "info": info,
        "lives": lives,
        "heights": heights,
        "intent_count": jnp.sum(intent_possible, axis=1),
        "intent_playable": jnp.sum(intent_possible & type_playable[None, :], axis=1),
        "last_reveal": jnp.argmax(f[281:306]).astype(jnp.int32),
        "last_success": f[306] > 0.5,
    }
    return action, mem, trace
