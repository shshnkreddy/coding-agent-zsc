"""A conservative H-group Level 1 policy for two-player Hanabi."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'hgroup_level1_notes_v1',
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


def init_agent():
    return {
        'notes': jnp.zeros((5, 25), dtype=jnp.bool_),
        'touched': jnp.zeros((5,), dtype=jnp.bool_),
        'age': jnp.zeros((5,), dtype=jnp.int32),
    }


def _heights(features):
    fw = features[167:192].reshape(5, 5)
    return jnp.sum(fw * jnp.arange(1, 6, dtype=jnp.float32)[None, :], axis=1).astype(jnp.int32)


def _discard_counts(features):
    d = features[203:253].reshape(5, 10)
    return jnp.stack((
        jnp.sum(d[:, 0:3], axis=1),
        jnp.sum(d[:, 3:5], axis=1),
        jnp.sum(d[:, 5:7], axis=1),
        jnp.sum(d[:, 7:9], axis=1),
        d[:, 9],
    ), axis=1).astype(jnp.int32)


def _identity_tables():
    colors = jnp.repeat(jnp.arange(5, dtype=jnp.int32), 5)
    ranks = jnp.tile(jnp.arange(5, dtype=jnp.int32), 5)
    return colors, ranks


def _useful_table(heights, discards):
    """Whether each identity can still contribute to its firework."""
    mult = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)
    dead = discards >= mult[None, :]
    rows = []
    for c in range(5):
        vals = []
        for r in range(5):
            ks = jnp.arange(5, dtype=jnp.int32)
            blocked = jnp.any(dead[c] & (ks >= heights[c]) & (ks <= r))
            vals.append((r >= heights[c]) & ~blocked)
        rows.append(jnp.stack(vals))
    return jnp.stack(rows).reshape(25)


def _focus(match, touched, occupied):
    """Native index of H-group focus, computed from the pre-clue hand."""
    chop_mask = occupied & ~touched
    has_chop = jnp.any(chop_mask)
    chop = jnp.argmax(chop_mask.astype(jnp.int32))
    new = match & ~touched
    has_new = jnp.any(new)
    weights = jnp.arange(1, 6, dtype=jnp.int32)
    left_new = jnp.argmax(new.astype(jnp.int32) * weights)
    left_retouch = jnp.argmax(match.astype(jnp.int32) * weights)
    focus = jnp.where(has_new, jnp.where(has_chop & match[chop], chop, left_new), left_retouch)
    return focus.astype(jnp.int32), chop.astype(jnp.int32), has_chop, new


def _shift_after_removal(array, removed, fill):
    idx = jnp.arange(5, dtype=jnp.int32)
    source = jnp.minimum(idx + (idx >= removed).astype(jnp.int32), 4)
    shifted = array[source]
    if array.ndim == 1:
        return shifted.at[4].set(fill)
    return shifted.at[4].set(jnp.full(array.shape[1:], fill, dtype=array.dtype))


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    heights = _heights(f)
    discards = _discard_counts(f)
    useful = _useful_table(heights, discards)
    colors, ranks = _identity_tables()
    playable_ids = ranks == heights[colors]
    info_count = jnp.sum(f[192:200]).astype(jnp.int32)
    lives_count = jnp.sum(f[200:203]).astype(jnp.int32)
    mult = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)

    knowledge = f[308:658].reshape(2, 5, 35)
    own_literal = knowledge[0, :, :25] > 0
    partner_literal = knowledge[1, :, :25] > 0
    own_occupied = jnp.any(own_literal, axis=1)
    partner_occupied = jnp.any(partner_literal, axis=1)
    own_touched_now = jnp.any(knowledge[0, :, 25:] > 0, axis=1) & own_occupied
    partner_touched = jnp.any(knowledge[1, :, 25:] > 0, axis=1) & partner_occupied
    partner_cards = f[0:125].reshape(5, 25) > 0
    visible_counts = jnp.sum(partner_cards.astype(jnp.int32), axis=0)
    already_played = ranks < heights[colors]
    public_remaining = jnp.tile(mult, 5) - discards.reshape(25) - already_played.astype(jnp.int32)
    own_remaining = public_remaining - visible_counts
    own_possible = own_literal & (own_remaining > 0)[None, :]
    # Propagate exact singleton occupancy through the five-card hand. This is
    # ordinary card counting, not a convention inference.
    for _ in range(5):
        singleton = jnp.sum(own_possible.astype(jnp.int32), axis=1) == 1
        singleton_masks = own_possible & singleton[:, None]
        refined = []
        for s in range(5):
            other = jnp.arange(5, dtype=jnp.int32) != s
            used_elsewhere = jnp.sum(singleton_masks & other[:, None], axis=0)
            refined.append(own_possible[s] & ((own_remaining - used_elsewhere) > 0))
        own_possible = jnp.stack(refined)

    notes = jnp.where(reset, jnp.zeros_like(memory['notes']), memory['notes'])
    old_touched = jnp.where(reset, jnp.zeros_like(memory['touched']), memory['touched'])
    ages = jnp.where(reset, jnp.zeros_like(memory['age']), memory['age'])

    la = f[253:308]
    actor_self = la[0] > 0
    self_removed = actor_self & ((la[2] > 0) | (la[3] > 0))
    removed = jnp.argmax(la[23:28]).astype(jnp.int32)
    shifted_notes = _shift_after_removal(notes, removed, False)
    shifted_touch = _shift_after_removal(old_touched, removed, False)
    shifted_ages = _shift_after_removal(ages, removed, 0)
    notes = jnp.where(self_removed, shifted_notes, notes)
    old_touched = jnp.where(self_removed, shifted_touch, old_touched)
    ages = jnp.where(self_removed, shifted_ages, ages)

    # Retain only identities still literally possible and conventionally useful.
    notes = notes & own_possible & useful[None, :]

    # Interpret a clue received from the partner using pre-clue focus and chop.
    got_hint = (la[1] > 0) & ((la[4] > 0) | (la[5] > 0))
    match = la[18:23] > 0
    recv_focus, recv_chop, recv_has_chop, recv_new = _focus(match, old_touched, own_occupied)
    clue_color = jnp.argmax(la[8:13]).astype(jnp.int32)
    clue_rank = jnp.argmax(la[13:18]).astype(jnp.int32)
    is_color = la[4] > 0
    attr_mask = jnp.where(is_color, colors == clue_color, ranks == clue_rank)

    # Direct and visibly delayed play possibilities. A delayed identity is
    # retained only when every intervening connector is visible in the giver's
    # hand; the card will not be played until every retained possibility is live.
    # A raw visible card is not an established connection. Only a card that
    # already has public clue information is "present and accounted for".
    visible_types = jnp.any(partner_cards & partner_touched[:, None], axis=0)
    reachable_rows = []
    for c in range(5):
        vals = []
        for r in range(5):
            connector_ok = jnp.asarray(True)
            for k in range(5):
                required = (k >= heights[c]) & (k < r)
                connector_ok = connector_ok & (~required | visible_types[5 * c + k])
            vals.append(useful[5 * c + r] & connector_ok)
        reachable_rows.append(jnp.stack(vals))
    reachable = jnp.stack(reachable_rows).reshape(25)
    play_note = attr_mask & reachable

    critical = (discards.reshape(25) == jnp.tile(mult - 1, 5)) & useful
    rank5_save = (ranks == 4) & useful
    rank2_save = (ranks == 1) & useful & ~playable_ids
    save_note = jnp.where(
        is_color,
        attr_mask & critical & (ranks != 4),
        attr_mask & jnp.where(clue_rank == 4, rank5_save,
                              jnp.where(clue_rank == 1, rank2_save, critical)),
    )
    focus_on_chop = recv_has_chop & (recv_focus == recv_chop)
    interpreted = play_note | jnp.where(focus_on_chop, save_note, jnp.zeros_like(save_note))
    interpreted = jnp.where(jnp.any(interpreted), interpreted, attr_mask & useful)
    # With no newly gotten card this is a Tempo Clue: its Level-1 purpose is
    # specifically to make the leftmost re-touched focus play now. It cannot
    # be a save, so delayed/save alternatives are excluded.
    tempo_received = ~jnp.any(recv_new)
    interpreted = jnp.where(tempo_received, attr_mask & playable_ids, interpreted)
    # A clue sent while the giver had eight tokens can be mechanically forced:
    # discarding was unavailable. If its literal information does not itself
    # establish a safe play (and it is not a recognizable save), preserve the
    # full useful interpretation instead of gambling on a play promise.
    focus_literal_now = own_possible[recv_focus]
    literal_now_safe = jnp.any(focus_literal_now) & ~jnp.any(focus_literal_now & ~playable_ids)
    recognizable_save = focus_on_chop & jnp.any(save_note)
    forced_ambiguous = got_hint & (info_count == 7) & ~literal_now_safe & ~recognizable_save
    interpreted = jnp.where(forced_ambiguous, attr_mask & useful, interpreted)

    # Every newly touched non-focus card is useful by Good Touch. The focused
    # card receives the stronger play/save interpretation above.
    new_note = attr_mask & useful
    for s in range(5):
        set_nonfocus = got_hint & ~forced_ambiguous & recv_new[s] & (recv_focus != s)
        notes = notes.at[s].set(jnp.where(set_nonfocus, new_note, notes[s]))
    notes = notes.at[recv_focus].set(jnp.where(got_hint & ~forced_ambiguous,
                                               interpreted, notes[recv_focus]))
    notes = notes & own_possible & useful[None, :]
    ages = jnp.where(own_occupied, ages + 1, jnp.zeros_like(ages))

    # Cards whose full retained note is currently playable are safe. Literal
    # singleton/all-playable knowledge is also authoritative.
    note_safe = jnp.any(notes, axis=1) & ~jnp.any(notes & ~playable_ids[None, :], axis=1)
    literal_safe = jnp.any(own_possible, axis=1) & ~jnp.any(own_possible & ~playable_ids[None, :], axis=1)
    # Convention notes are normally trusted, but bomb-zero scoring makes the
    # final life qualitatively different. On one life require literal safety.
    safe_play = own_occupied & (literal_safe | (note_safe & (lives_count > 1)))
    play_slot = jnp.argmax(safe_play.astype(jnp.int32)).astype(jnp.int32)
    has_safe_play = jnp.any(safe_play)

    # Exact visible partner cards and their H-group chop.
    partner_flat = f[0:125].reshape(5, 25)
    partner_type = jnp.argmax(partner_flat, axis=1).astype(jnp.int32)
    partner_color = partner_type // 5
    partner_rank = partner_type % 5
    _, partner_chop, partner_has_chop, _ = _focus(
        jnp.zeros((5,), dtype=jnp.bool_), partner_touched, partner_occupied)
    chop_type = partner_type[partner_chop]
    chop_c = partner_color[partner_chop]
    chop_r = partner_rank[partner_chop]
    chop_playable = chop_r == heights[chop_c]
    chop_still_useful = useful[chop_type]
    chop_critical = (discards[chop_c, chop_r] == (mult[chop_r] - 1)) & chop_still_useful
    same_type_elsewhere = jnp.sum((partner_type == chop_type) & partner_occupied) > 1
    chop_unique_playable = chop_playable & ~same_type_elsewhere
    chop_two_save = (chop_r == 1) & ~same_type_elsewhere & ~chop_playable & chop_still_useful
    chop_five_save = (chop_r == 4) & ~chop_playable & chop_still_useful
    chop_endangered = partner_has_chop & (chop_unique_playable | chop_critical |
                                          chop_two_save | chop_five_save)

    # Score all ordinary direct play clues. Only new, useful, nonduplicated
    # touches are accepted, and the H-group focus must be playable now.
    hint_scores = []
    save_scores = []
    tempo_scores = []
    for h in range(10):
        hc = h if h < 5 else 0
        hr = h - 5 if h >= 5 else 0
        hm = jnp.where(h < 5, partner_color == hc, partner_rank == hr) & partner_occupied
        hfocus, _, _, hnew = _focus(hm, partner_touched, partner_occupied)
        focus_play = partner_rank[hfocus] == heights[partner_color[hfocus]]
        new_useful = useful[partner_type] & hnew
        after_touch = partner_touched | hnew
        duplicate_bad = jnp.asarray(False)
        duplicate_count = jnp.asarray(0, dtype=jnp.int32)
        for s in range(5):
            duplicates = jnp.sum(after_touch & partner_occupied & (partner_type == partner_type[s]))
            duplicate_bad = duplicate_bad | (hnew[s] & (duplicates > 1))
            duplicate_count = duplicate_count + (hnew[s] & (duplicates > 1)).astype(jnp.int32)
        good = ~jnp.any(hnew & ~useful[partner_type]) & ~duplicate_bad
        new_count = jnp.sum(hnew.astype(jnp.int32))
        action_id = 10 + h
        ordinary_valid = legal[action_id] & (new_count > 0) & focus_play & good
        focus_after_hint = partner_literal[hfocus] & jnp.where(h < 5, colors == hc, ranks == hr)
        focus_literal_safe = jnp.any(focus_after_hint) & ~jnp.any(focus_after_hint & ~playable_ids)
        ordinary_score = (100 + 20 * new_count + jnp.sum(new_useful.astype(jnp.int32)) +
                          (h < 5) * 2 + focus_literal_safe * 60)
        hint_scores.append(jnp.where(ordinary_valid, ordinary_score, -10000))

        # A no-new-card play clue is a Level-1 tempo clue. It is reserved for
        # forced-clue positions, but when used its public literal information
        # must make the focused card unambiguously playable.
        hattr = jnp.where(h < 5, colors == hc, ranks == hr)
        focus_literal = partner_literal[hfocus] & hattr
        literal_focus_safe = jnp.any(focus_literal) & ~jnp.any(focus_literal & ~playable_ids)
        tempo_valid = legal[action_id] & (new_count == 0) & focus_play
        tempo_scores.append(jnp.where(tempo_valid, 50 + (h < 5) * 2, -10000))

        focuses_chop = partner_has_chop & hm[partner_chop] & (hfocus == partner_chop)
        right_form = jnp.where(
            chop_r == 4,
            h == 9,
            jnp.where(chop_two_save, h == 6,
                      jnp.where(chop_critical & (chop_r != 4),
                                ((h < 5) & (h == chop_c)) | ((h >= 5) & (h - 5 == chop_r)),
                                chop_unique_playable)),
        )
        bad_count = jnp.sum((hnew & ~useful[partner_type]).astype(jnp.int32))
        save_valid = legal[action_id] & chop_endangered & focuses_chop & right_form
        save_score = (200 - 80 * bad_count - 40 * duplicate_count +
                      20 * new_count + ((h < 5) * 2))
        save_scores.append(jnp.where(save_valid, save_score, -10000))

    hint_scores = jnp.stack(hint_scores)
    save_scores = jnp.stack(save_scores)
    tempo_scores = jnp.stack(tempo_scores)
    best_hint_ix = jnp.argmax(hint_scores).astype(jnp.int32)
    best_save_ix = jnp.argmax(save_scores).astype(jnp.int32)
    best_tempo_ix = jnp.argmax(tempo_scores).astype(jnp.int32)
    has_hint = jnp.max(hint_scores) >= 0
    has_save = jnp.max(save_scores) >= 0
    has_tempo = jnp.max(tempo_scores) >= 0
    partner_literal_safe = (partner_occupied & jnp.any(partner_literal, axis=1) &
                            ~jnp.any(partner_literal & ~playable_ids[None, :], axis=1))
    partner_has_public_play = jnp.any(partner_literal_safe)
    urgent_save = has_save & ~partner_has_public_play

    # Known trash first, otherwise ordinary chop (oldest unclued = low index).
    literal_useful = own_possible & useful[None, :]
    literal_trash = own_occupied & ~jnp.any(literal_useful, axis=1)
    known_trash = literal_trash
    trash_slot = jnp.argmax(known_trash.astype(jnp.int32)).astype(jnp.int32)
    own_chop_mask = own_occupied & ~own_touched_now
    own_chop = jnp.argmax(own_chop_mask.astype(jnp.int32)).astype(jnp.int32)
    conventional_discard = jnp.any(known_trash) | jnp.any(own_chop_mask)
    # Fully clued hands have no conventional chop. If there is no meaningful
    # clue or play, an arbitrary clue is actively dangerous because it looks
    # like a play promise. Prefer the least risky legal discard instead.
    important = useful & ((ranks == 4) | (ranks == 1) | critical)
    discard_risk = jnp.sum(knowledge[0, :, :25] *
                           (useful.astype(jnp.float32)[None, :] +
                            12.0 * important.astype(jnp.float32)[None, :]), axis=1)
    least_risk_slot = jnp.argmin(jnp.where(own_occupied, discard_risk, 1.0e6)).astype(jnp.int32)
    has_discard_choice = jnp.any(own_occupied)
    discard_slot = jnp.where(jnp.any(known_trash), trash_slot,
                             jnp.where(jnp.any(own_chop_mask), own_chop, least_risk_slot))

    legal_hints = legal[10:20]
    fallback_hint = 10 + jnp.argmax(legal_hints.astype(jnp.int32)).astype(jnp.int32)
    can_hint = jnp.any(legal_hints)

    action = jnp.where(
        urgent_save,
        10 + best_save_ix,
        jnp.where(
            has_safe_play,
            5 + play_slot,
            jnp.where(
                has_hint,
                10 + best_hint_ix,
                jnp.where(has_tempo & (~conventional_discard | ~legal[discard_slot]),
                          10 + best_tempo_ix,
                          jnp.where(has_discard_choice & legal[discard_slot], discard_slot,
                                    jnp.where(can_hint, fallback_hint, jnp.argmax(legal).astype(jnp.int32)))),
            ),
        ),
    ).astype(jnp.int32)
    action = jnp.where(legal[20], jnp.asarray(20, dtype=jnp.int32), action)

    next_memory = {'notes': notes, 'touched': own_touched_now, 'age': ages}
    trace = {
        'mode': jnp.where(urgent_save, 1, jnp.where(has_safe_play, 2,
                          jnp.where(has_hint, 3, jnp.where(has_tempo, 5, 4)))).astype(jnp.int32),
        'safe_count': jnp.sum(safe_play.astype(jnp.int32)),
        'note_count': jnp.sum(jnp.any(notes, axis=1).astype(jnp.int32)),
        'partner_chop': partner_chop,
        'chop_endangered': chop_endangered,
        'received_focus': recv_focus,
        'deck': jnp.sum(f[127:167]).astype(jnp.int32),
        'score': jnp.sum(heights).astype(jnp.int32),
        'info': jnp.sum(f[192:200]).astype(jnp.int32),
        'lives': jnp.sum(f[200:203]).astype(jnp.int32),
        'reset_seen': reset,
        'last_card': jnp.argmax(la[28:53]).astype(jnp.int32),
        'last_success': la[53] > 0,
        'heights': heights,
        'play_slot': play_slot,
    }
    return action, next_memory, trace
