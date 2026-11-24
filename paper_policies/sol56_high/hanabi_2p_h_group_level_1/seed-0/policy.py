"""Deterministic H-group Level 1 policy for two-player Hanabi.

The private state is only a compact copy of public convention notes for this
player's own cards. Clues use the documented native-slot mapping: low native
indices are chop, while high native indices are leftmost/finesse-side cards.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_l1_foundation', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

MULTIPLICITIES = (3, 2, 2, 2, 1)


def init_agent():
    return {
        'clued': jnp.zeros((5,), dtype=jnp.bool_),
        'note': jnp.zeros((5, 25), dtype=jnp.bool_),
        'status': jnp.zeros((5,), dtype=jnp.int32),
    }


def _decode(features):
    partner = features[0:125].reshape(5, 5, 5) > 0.5
    partner_occ = jnp.any(partner, axis=(1, 2))
    deck = jnp.sum(features[127:167]).astype(jnp.int32)
    fw_oh = features[167:192].reshape(5, 5)
    heights = jnp.sum(fw_oh * jnp.arange(1, 6, dtype=jnp.float32)[None, :], axis=1).astype(jnp.int32)
    info = jnp.sum(features[192:200]).astype(jnp.int32)
    lives = jnp.sum(features[200:203]).astype(jnp.int32)
    raw_disc = features[203:253].reshape(5, 10)
    discards = jnp.stack([
        jnp.sum(raw_disc[:, 0:3], axis=1),
        jnp.sum(raw_disc[:, 3:5], axis=1),
        jnp.sum(raw_disc[:, 5:7], axis=1),
        jnp.sum(raw_disc[:, 7:9], axis=1),
        raw_disc[:, 9],
    ], axis=1).astype(jnp.int32)
    last = features[253:308]
    know = features[308:658].reshape(2, 5, 35)
    weights = know[:, :, 0:25]
    literal = weights > 0.0
    touched = jnp.any(know[:, :, 25:35] > 0.5, axis=2)
    own_occ = jnp.any(literal[0], axis=1)
    return partner, partner_occ, own_occ, deck, heights, info, lives, discards, last, weights, literal, touched


def _shift_after_remove(x, removed, fill):
    slots = jnp.arange(5, dtype=jnp.int32)
    source = slots + (slots >= removed).astype(jnp.int32)
    valid = source < 5
    gathered = x[jnp.minimum(source, 4)]
    if x.ndim == 1:
        return jnp.where(valid, gathered, fill)
    return jnp.where(valid[:, None], gathered, fill)


def _lowest(mask):
    slots = jnp.arange(5, dtype=jnp.int32)
    return jnp.min(jnp.where(mask, slots, 5))


def _highest(mask):
    slots = jnp.arange(5, dtype=jnp.int32)
    return jnp.max(jnp.where(mask, slots, -1))


def _reachable(heights, discards):
    rows = []
    multiplicities = jnp.asarray(MULTIPLICITIES, dtype=jnp.int32)
    remaining = multiplicities[None, :] - discards
    for c in range(5):
        vals = []
        for r in range(5):
            ok = jnp.asarray(r >= heights[c])
            for k in range(r + 1):
                needed = (k >= heights[c]) & (k <= r)
                ok = ok & ((~needed) | (remaining[c, k] > 0))
            vals.append(ok)
        rows.append(jnp.stack(vals))
    return jnp.stack(rows)


def _connectable(heights, gotten, reachable):
    rows = []
    for c in range(5):
        vals = []
        for r in range(5):
            ok = reachable[c, r]
            for k in range(r):
                needed = (k >= heights[c]) & (k < r)
                ok = ok & ((~needed) | gotten[c, k])
            vals.append(ok)
        rows.append(jnp.stack(vals))
    return jnp.stack(rows)


def _card_arrays(partner):
    color = jnp.argmax(jnp.any(partner, axis=2), axis=1).astype(jnp.int32)
    rank = jnp.argmax(jnp.any(partner, axis=1), axis=1).astype(jnp.int32)
    return color, rank


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    partner, partner_occ, own_occ, deck, heights, info, lives, discards, last, weights, literal, touched_now = _decode(f)
    reachable = _reachable(heights, discards)
    ci = jnp.arange(5, dtype=jnp.int32)[:, None]
    ri = jnp.arange(5, dtype=jnp.int32)[None, :]
    multiplicities = jnp.asarray(MULTIPLICITIES, dtype=jnp.int32)
    current = ri == heights[:, None]

    clued = jnp.where(reset, jnp.zeros((5,), dtype=jnp.bool_), memory['clued'])
    note = jnp.where(reset, jnp.zeros((5, 25), dtype=jnp.bool_), memory['note'])
    status = jnp.where(reset, jnp.zeros((5,), dtype=jnp.int32), memory['status'])

    actor_self = last[0] > 0.5
    was_play_or_discard = jnp.any(last[2:4] > 0.5)
    own_removed = actor_self & was_play_or_discard & (~reset)
    removed = jnp.argmax(last[23:28]).astype(jnp.int32)
    shifted_clued = _shift_after_remove(clued, removed, jnp.asarray(False))
    shifted_note = _shift_after_remove(note, removed, jnp.asarray(False))
    shifted_status = _shift_after_remove(status, removed, jnp.asarray(0, jnp.int32))
    clued = jnp.where(own_removed, shifted_clued, clued)
    note = jnp.where(own_removed, shifted_note, note)
    status = jnp.where(own_removed, shifted_status, status)

    clue_type = jnp.any(last[4:6] > 0.5)
    clue_to_self = (~reset) & clue_type & (last[0] < 0.5) & (last[6] > 0.5)
    matches = last[18:23] > 0.5
    new = matches & (~clued) & own_occ
    any_new = jnp.any(new)
    chop = _lowest(own_occ & (~clued))
    chop_hit = (chop < 5) & matches[jnp.minimum(chop, 4)]
    focus_new = jnp.where(chop_hit, chop, _highest(new))
    focus_old = _highest(matches & own_occ)
    focus = jnp.where(any_new, focus_new, focus_old)
    focus_safe = jnp.maximum(focus, 0)

    pc, pr = _card_arrays(partner)
    partner_gotten = jnp.zeros((5, 5), dtype=jnp.bool_)
    for s in range(5):
        partner_gotten = partner_gotten.at[pc[s], pr[s]].set(partner_gotten[pc[s], pr[s]] | (partner_occ[s] & touched_now[1, s]))
    own_exact = literal[0] & (jnp.sum(literal[0], axis=1, keepdims=True) == 1)
    own_gotten = jnp.any(own_exact & clued[:, None], axis=0).reshape(5, 5)
    gotten = partner_gotten | own_gotten
    connectable = _connectable(heights, gotten, reachable)

    named_color = jnp.argmax(last[8:13]).astype(jnp.int32)
    named_rank = jnp.argmax(last[13:18]).astype(jnp.int32)
    is_color = last[4] > 0.5
    is_rank = last[5] > 0.5
    attr = jnp.where(is_color, ci == named_color, ri == named_rank)
    play_candidates = attr & connectable
    critical = (discards >= (multiplicities[None, :] - 1)) & reachable
    rank5_save = (ri == 4) & attr & reachable & is_rank
    rank2_save = (ri == 1) & attr & reachable & is_rank
    critical_save = critical & attr & (ri != 4)
    save_candidates = rank5_save | rank2_save | critical_save
    focus_is_chop = any_new & (focus == chop)
    interpreted = play_candidates | (save_candidates & focus_is_chop)
    focus_literal = literal[0, focus_safe].reshape(5, 5)
    proposed = interpreted & focus_literal
    fallback = attr & focus_literal
    proposed = jnp.where(jnp.any(proposed), proposed, fallback)
    old = note[focus_safe].reshape(5, 5) & focus_literal
    overlap = old & proposed
    refined = jnp.where((status[focus_safe] > 0) & jnp.any(overlap), overlap, proposed)
    note_update = refined.reshape(25)
    note = note.at[focus_safe].set(jnp.where(clue_to_self, note_update, note[focus_safe]))
    # A focused clue can still be a delayed play or a chop save.  Preserve a
    # separate origin tag so that only a focus whose entire note was directly
    # playable when received is trusted on the final life.
    direct_immediate = (jnp.sum(proposed.astype(jnp.int32)) == 1) & jnp.all((~proposed) | current)
    ordinary_focus_status = jnp.where(direct_immediate, 1, 5)
    focus_status = jnp.where(info == 7, 4, ordinary_focus_status).astype(jnp.int32)
    status = status.at[focus_safe].set(jnp.where(clue_to_self, focus_status, status[focus_safe]))
    for s in range(5):
        nonfocus_new = clue_to_self & new[s] & (s != focus)
        # At eight pre-clue tokens a clue may be mechanically forced and need
        # not satisfy Good Touch. Preserve raw literal alternatives then.
        trusted_gt = info < 7
        gt = jnp.where(
            trusted_gt,
            literal[0, s].reshape(5, 5) & reachable,
            literal[0, s].reshape(5, 5),
        ).reshape(25)
        note = note.at[s].set(jnp.where(nonfocus_new, gt, note[s]))
        # Side touches of a non-chop focus come from a definite play clue, so
        # Good Touch is reliable.  A chop-focused clue may instead be a save
        # whose urgency legitimately overrides Good Touch.
        trusted_play_gt = trusted_gt & (~focus_is_chop)
        nonfocus_status = jnp.where(trusted_play_gt, 6, jnp.where(trusted_gt, 2, 3)).astype(jnp.int32)
        status = status.at[s].set(jnp.where(nonfocus_new, nonfocus_status, status[s]))

    clued = jnp.where(clue_to_self, clued | matches, clued)
    clued = jnp.where(reset, touched_now[0], clued)
    clued = clued & own_occ

    own_playable = []
    own_useless = []
    for s in range(5):
        lit = literal[0, s].reshape(5, 5)
        noted = note[s].reshape(5, 5) & lit
        poss0 = jnp.where(jnp.any(noted), noted, lit)
        # Never erase an identity merely because another copy has since
        # played. A true focused promise remains safe only while every retained
        # alternative is currently playable; this prevents duplicate races.
        forced_origin = (status[s] == 3) | (status[s] == 4)
        poss = jnp.where(forced_origin, poss0, poss0 & reachable)
        promised = clued[s] & (status[s] > 0)
        safe = promised & jnp.any(poss) & jnp.all((~poss) | current)
        literal_live = lit
        safe_literal = jnp.any(literal_live) & jnp.all((~literal_live) | current)
        raw_focus_safe = ((status[s] == 1) | (status[s] == 5) | (status[s] == 6)) & promised & jnp.any(poss0) & jnp.all((~poss0) | current)
        own_playable.append(own_occ[s] & (safe_literal | (safe & (lives > 1)) | raw_focus_safe))
        own_useless.append(own_occ[s] & (~jnp.any(lit & reachable)))
    own_playable = jnp.stack(own_playable)
    own_useless = jnp.stack(own_useless)

    p_playable = partner_occ & (pr == heights[pc])
    p_useful = partner_occ & reachable[pc, pr]
    p_clued = touched_now[1] & partner_occ
    p_ready = p_clued & p_playable
    p_chop = _lowest(partner_occ & (~p_clued))
    p_chop_safe = jnp.minimum(p_chop, 4)
    chop_c = pc[p_chop_safe]
    chop_r = pr[p_chop_safe]
    same_type_count = jnp.sum(partner_occ & (pc == chop_c) & (pr == chop_r))
    chop_critical = critical[chop_c, chop_r]
    chop_endangered = (p_chop < 5) & p_useful[p_chop_safe] & ((chop_r == 4) | ((chop_r == 1) & (same_type_count == 1)) | chop_critical | p_playable[p_chop_safe])

    clue_scores = []
    save_scores = []
    for a in range(10):
        if a < 5:
            m = partner_occ & (pc == a)
            action_id = 10 + a
            color_bonus = 4
            candidate_attr = ci == a
            candidate_rank = jnp.asarray(False)
        else:
            rr = a - 5
            m = partner_occ & (pr == rr)
            action_id = 15 + rr
            color_bonus = 0
            candidate_attr = ri == rr
            candidate_rank = jnp.asarray(True)
        nn = m & (~p_clued)
        nnew = jnp.sum(nn.astype(jnp.int32))
        any_nn = nnew > 0
        c_hit = (p_chop < 5) & m[p_chop_safe]
        foc = jnp.where(c_hit, p_chop, _highest(nn))
        fs = jnp.maximum(foc, 0)
        no_dupes = jnp.asarray(True)
        no_cross_hand_dupe = jnp.asarray(True)
        for s in range(5):
            copies = jnp.sum(m & (pc == pc[s]) & (pr == pr[s]))
            no_dupes = no_dupes & ((~m[s]) | (copies <= 1))
            own_may_match = jnp.asarray(False)
            target_id = 5 * pc[s] + pr[s]
            for own_s in range(5):
                own_public = jnp.where(
                    jnp.any(note[own_s] & literal[0, own_s]),
                    note[own_s] & literal[0, own_s],
                    literal[0, own_s],
                )
                own_may_match = own_may_match | (clued[own_s] & own_public[target_id])
            no_cross_hand_dupe = no_cross_hand_dupe & ((~m[s]) | (~own_may_match))
        good_touch = jnp.all((~m) | p_useful) & no_dupes & no_cross_hand_dupe
        if a < 5:
            save_alt = critical[a] & (ri[0] != 4)
            save_alt_unready = jnp.any(save_alt & (~current[a]))
        else:
            rr = a - 5
            save_alt = critical[:, rr] & (rr != 4)
            save_alt = save_alt | ((rr == 4) & reachable[:, rr])
            save_alt = save_alt | ((rr == 1) & reachable[:, rr])
            save_alt_unready = jnp.any(save_alt & (~current[:, rr]))
        unambiguous = (~c_hit) | (~save_alt_unready)
        valid_play = legal[action_id] & any_nn & (foc >= 0) & p_playable[fs] & good_touch & unambiguous
        recipient_literal = literal[1, fs].reshape(5, 5)
        recipient_play = candidate_attr & connectable & recipient_literal
        recipient_save = critical & candidate_attr & (ri != 4)
        recipient_save = recipient_save | (candidate_rank & candidate_attr & reachable & ((ri == 1) | (ri == 4)))
        recipient_note = recipient_play | (recipient_save & c_hit)
        recipient_note = jnp.where(jnp.any(recipient_note), recipient_note, candidate_attr & recipient_literal)
        actionable = jnp.any(recipient_note) & jnp.all((~recipient_note) | current)
        score = jnp.where(valid_play, 1000 + 64 * actionable.astype(jnp.int32) + 24 * nnew + color_bonus + 2 * (foc == p_chop), -10000)
        clue_scores.append(score)

        is_prescribed_save = (p_chop < 5) & c_hit & chop_endangered & (((chop_r == 4) & (a == 9)) | ((chop_r == 1) & (~chop_critical) & (a == 6)) | (chop_critical & (chop_r != 4) & ((a == chop_c) | (a == 5 + chop_r))))
        save_score = jnp.where(legal[action_id] & is_prescribed_save, 800 - 8 * nnew + color_bonus, -10000)
        save_scores.append(save_score)

    clue_scores = jnp.stack(clue_scores)
    save_scores = jnp.stack(save_scores)
    best_play_clue_score = jnp.max(clue_scores)
    best_play_clue = (10 + jnp.argmax(clue_scores)).astype(jnp.int32)
    best_save_score = jnp.max(save_scores)
    best_save = (10 + jnp.argmax(save_scores)).astype(jnp.int32)
    have_play_clue = best_play_clue_score > 0
    have_save = best_save_score > 0

    have_own_play = jnp.any(own_playable)
    own_play_slot = _highest(own_playable)
    own_play_action = (5 + jnp.maximum(own_play_slot, 0)).astype(jnp.int32)
    partner_preoccupied = jnp.any(p_ready)
    urgent = chop_endangered & (~partner_preoccupied)
    choose_preoccupying_clue = urgent & have_play_clue
    final_round = deck == 0
    choose_save = urgent & (~have_play_clue) & have_save & (~final_round)

    action = jnp.asarray(-1, dtype=jnp.int32)
    action = jnp.where(final_round & have_own_play, own_play_action, action)
    action = jnp.where(final_round & (action < 0) & have_play_clue, best_play_clue, action)
    action = jnp.where(choose_preoccupying_clue, best_play_clue, action)
    action = jnp.where((action < 0) & choose_save, best_save, action)
    action = jnp.where((action < 0) & have_own_play, own_play_action, action)
    action = jnp.where((action < 0) & have_play_clue, best_play_clue, action)
    action = jnp.where((action < 0) & have_save & (~final_round), best_save, action)

    trash_slot = _lowest(own_useless)
    own_chop = _lowest(own_occ & (~clued))
    discard_slot = jnp.where(trash_slot < 5, trash_slot, own_chop)
    can_discard = (discard_slot < 5) & legal[jnp.minimum(discard_slot, 4)]
    action = jnp.where((action < 0) & can_discard, discard_slot.astype(jnp.int32), action)

    # A fully clued deadlock has no conventional chop. If discarding is
    # mechanically possible, release the least valuable card under public
    # literal/convention knowledge instead of sending a false play message.
    discard_risks = []
    rank_cost = jnp.asarray([2.0, 9.0, 5.0, 7.0, 40.0], dtype=jnp.float32)[None, :]
    value_cost = rank_cost + 55.0 * critical.astype(jnp.float32) + 18.0 * current.astype(jnp.float32)
    value_cost = jnp.where(reachable, value_cost, 0.0)
    for s in range(5):
        w = weights[0, s].reshape(5, 5)
        nm = note[s].reshape(5, 5)
        restricted = jnp.where(jnp.any(nm & (w > 0.0)), w * nm, w)
        norm = jnp.sum(restricted)
        risk = jnp.sum(restricted * value_cost) / jnp.maximum(norm, 1e-6)
        discard_risks.append(jnp.where(own_occ[s] & legal[s], risk, 1e6))
    discard_risks = jnp.stack(discard_risks)
    least_risk_slot = jnp.argmin(discard_risks).astype(jnp.int32)
    have_risk_discard = jnp.min(discard_risks) < 1e5
    action = jnp.where((action < 0) & have_risk_discard, least_risk_slot, action)

    tempo_scores = []
    final_tempo_scores = []
    for a in range(10):
        if a < 5:
            m = partner_occ & (pc == a)
            aid = 10 + a
            tempo_attr = ci == a
        else:
            m = partner_occ & (pr == a - 5)
            aid = 10 + a
            tempo_attr = ri == (a - 5)
        nn = m & (~p_clued)
        any_nn = jnp.any(nn)
        chop_match = (p_chop < 5) & m[p_chop_safe]
        focus_emergency = jnp.where(any_nn, jnp.where(chop_match, p_chop, _highest(nn)), _highest(m))
        fr = jnp.maximum(focus_emergency, 0)
        readable_play = (focus_emergency >= 0) & p_playable[fr]
        filled_literal = tempo_attr & literal[1, fr].reshape(5, 5)
        fills_to_current = jnp.any(filled_literal) & jnp.all((~filled_literal) | current)
        tempo_score = jnp.where(legal[aid], 1000 * readable_play.astype(jnp.int32) + 500 * fills_to_current.astype(jnp.int32) + 100 * (jnp.any(m & p_ready)).astype(jnp.int32) + 10 * any_nn.astype(jnp.int32) - jnp.sum(m.astype(jnp.int32)), -10000)
        tempo_scores.append(tempo_score)
        final_tempo_scores.append(jnp.where(legal[aid] & (~any_nn) & readable_play & fills_to_current, tempo_score, -10000))
    tempo_scores = jnp.stack(tempo_scores)
    final_tempo_scores = jnp.stack(final_tempo_scores)
    forced_clue = (10 + jnp.argmax(tempo_scores)).astype(jnp.int32)
    have_final_tempo = jnp.max(final_tempo_scores) > 0
    best_final_tempo = (10 + jnp.argmax(final_tempo_scores)).astype(jnp.int32)
    action = jnp.where(final_round & (~have_own_play) & (~have_play_clue) & have_final_tempo, best_final_tempo, action)
    action = jnp.where(action < 0, forced_clue, action)

    inactive = legal[20]
    guarded = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    action = jnp.where(inactive, jnp.asarray(20, jnp.int32), guarded)

    next_memory = {'clued': clued, 'note': note, 'status': status}
    trace = {
        'own_playable': own_playable,
        'partner_playable': p_playable,
        'partner_clued': p_clued,
        'own_chop': own_chop.astype(jnp.int32),
        'partner_chop': p_chop.astype(jnp.int32),
        'best_play_clue_score': best_play_clue_score.astype(jnp.int32),
        'best_save_score': best_save_score.astype(jnp.int32),
        'chosen': action,
        'deck': deck,
        'info': info,
        'lives': lives,
    }
    return action, next_memory, trace
