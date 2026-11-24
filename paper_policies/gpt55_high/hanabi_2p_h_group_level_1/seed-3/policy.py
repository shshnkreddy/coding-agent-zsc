"""Convention-readable Level 1 Hanabi policy."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'level1_rule_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


MULTS = (3, 2, 2, 2, 1)


def _slots():
    return jnp.arange(5, dtype=jnp.int32)


def _ranks():
    return jnp.arange(5, dtype=jnp.int32)


def _mults():
    return jnp.asarray(MULTS, dtype=jnp.int32)


def init_agent():
    z_b = jnp.zeros((5,), dtype=jnp.bool_)
    z_i = jnp.full((5,), -1, dtype=jnp.int32)
    return {
        'play': z_b,
        'hold': z_b,
        'target_color': z_i,
        'target_rank': z_i,
        'prev_self_clued': z_b,
        'prev_partner_clued': z_b,
    }


def _decode(obs):
    f = obs.features
    partner = f[0:125].reshape((5, 5, 5))
    partner_occ = jnp.sum(partner, axis=(1, 2)) > 0.5

    fireworks = f[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks * (_ranks() + 1), axis=1).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)

    disc_flat = f[203:253].reshape((5, 10))
    disc_counts = jnp.stack([
        jnp.sum(disc_flat[:, 0:3], axis=1),
        jnp.sum(disc_flat[:, 3:5], axis=1),
        jnp.sum(disc_flat[:, 5:7], axis=1),
        jnp.sum(disc_flat[:, 7:9], axis=1),
        jnp.sum(disc_flat[:, 9:10], axis=1),
    ], axis=1).astype(jnp.int32)

    last = f[253:308]
    knowledge = f[308:658].reshape((2, 5, 35))
    self_v0 = knowledge[0, :, 0:25].reshape((5, 5, 5))
    self_color_hints = knowledge[0, :, 25:30] > 0.5
    self_rank_hints = knowledge[0, :, 30:35] > 0.5
    partner_color_hints = knowledge[1, :, 25:30] > 0.5
    partner_rank_hints = knowledge[1, :, 30:35] > 0.5
    self_occ = jnp.sum(self_v0, axis=(1, 2)) > 0.0
    self_clued = (jnp.sum(self_color_hints, axis=1) + jnp.sum(self_rank_hints, axis=1)) > 0
    partner_clued = (jnp.sum(partner_color_hints, axis=1) + jnp.sum(partner_rank_hints, axis=1)) > 0
    return {
        'partner': partner,
        'partner_occ': partner_occ,
        'heights': heights,
        'lives': lives,
        'disc_counts': disc_counts,
        'last': last,
        'self_v0': self_v0,
        'self_occ': self_occ,
        'self_clued': self_clued,
        'partner_clued': partner_clued,
    }


def _onehot_index(vec):
    return jnp.argmax(vec).astype(jnp.int32)


def _lowest_slot(mask):
    return jnp.min(jnp.where(mask, _slots(), 5)).astype(jnp.int32)


def _highest_slot(mask):
    return jnp.max(jnp.where(mask, _slots(), -1)).astype(jnp.int32)


def _shift_bool(x, removed):
    slots = _slots()
    idx = jnp.where(slots < removed, slots, slots + 1)
    return jnp.take(jnp.concatenate([x, jnp.asarray([False])]), idx)


def _shift_int(x, removed):
    slots = _slots()
    idx = jnp.where(slots < removed, slots, slots + 1)
    return jnp.take(jnp.concatenate([x, jnp.asarray([-1], dtype=jnp.int32)]), idx)


def _clue_focus(match, old_clued, occ):
    newly = match & (~old_clued) & occ
    any_new = jnp.any(newly)
    chop = _lowest_slot(occ & (~old_clued))
    touched_chop = (chop < 5) & match[chop]
    retouched = match & old_clued & occ
    focus_new = _highest_slot(newly)
    focus_old = _highest_slot(retouched)
    return jnp.where(any_new, jnp.where(touched_chop, chop, focus_new), focus_old)


def _playable_matrix(heights):
    return (_ranks()[None, :] == heights[:, None]) & (heights[:, None] < 5)


def _trash_matrix(heights):
    return _ranks()[None, :] < heights[:, None]


def _critical_matrix(heights, disc_counts):
    needed = _ranks()[None, :] >= heights[:, None]
    return needed & (disc_counts >= (_mults()[None, :] - 1))


def _partner_identity(partner):
    by_color = jnp.sum(partner, axis=2)
    by_rank = jnp.sum(partner, axis=1)
    color = jnp.argmax(by_color, axis=1).astype(jnp.int32)
    rank = jnp.argmax(by_rank, axis=1).astype(jnp.int32)
    return color, rank


def _visible_same_count(partner_color, partner_rank, occ, color, rank):
    return jnp.sum((partner_color == color) & (partner_rank == rank) & occ)


def _legal_fallback(legal):
    return jnp.argmax(legal).astype(jnp.int32)


def _apply_score(scores, action, legal, score):
    return scores.at[action].set(jnp.where(legal[action], jnp.maximum(scores[action], score), scores[action]))


def _memory_after_public_action(memory, data, reset):
    init = init_agent()
    mem = {k: jnp.where(reset, init[k], memory[k]) for k in init}
    last = data['last']
    actor_self = last[0] > 0.5
    play_or_discard = (last[2] > 0.5) | (last[3] > 0.5)
    removed = _onehot_index(last[23:28])
    do_shift = actor_self & play_or_discard
    shifted = {
        'play': _shift_bool(mem['play'], removed),
        'hold': _shift_bool(mem['hold'], removed),
        'target_color': _shift_int(mem['target_color'], removed),
        'target_rank': _shift_int(mem['target_rank'], removed),
        'prev_self_clued': _shift_bool(mem['prev_self_clued'], removed),
        'prev_partner_clued': mem['prev_partner_clued'],
    }
    return {k: jnp.where(do_shift, shifted[k], mem[k]) for k in mem}


def _notice_received_clue(mem, data, reset):
    last = data['last']
    actor_partner = last[1] > 0.5
    is_color = last[4] > 0.5
    is_rank = last[5] > 0.5
    is_hint = is_color | is_rank
    to_self = last[6] > 0.5
    received = (~reset) & actor_partner & is_hint & to_self
    matches = last[18:23] > 0.5
    focus = _clue_focus(matches, mem['prev_self_clued'], data['self_occ'])
    valid_focus = (focus >= 0) & (focus < 5)

    color_value = _onehot_index(last[8:13])
    rank_value = _onehot_index(last[13:18])
    clue_rank = jnp.where(is_rank, rank_value, data['heights'][color_value])
    clue_color = jnp.where(is_color, color_value, -1)
    old_chop = _lowest_slot(data['self_occ'] & (~mem['prev_self_clued']))
    focus_is_chop = valid_focus & (focus == old_chop)

    rank_save = is_rank & focus_is_chop & ((rank_value == 4) | (rank_value == 1))
    immediate_one = clue_rank == 0
    mark_play = received & valid_focus & (~rank_save) & ((~focus_is_chop) | immediate_one)
    mark_hold = received & valid_focus & (rank_save | (focus_is_chop & (~immediate_one)))

    play = mem['play'].at[focus].set(jnp.where(mark_play, True, mem['play'][focus]))
    hold = mem['hold'].at[focus].set(jnp.where(mark_hold, True, jnp.where(mark_play, False, mem['hold'][focus])))
    target_color = mem['target_color'].at[focus].set(jnp.where(mark_play, clue_color, mem['target_color'][focus]))
    target_rank = mem['target_rank'].at[focus].set(jnp.where(mark_play, clue_rank, mem['target_rank'][focus]))
    return {
        'play': play,
        'hold': hold,
        'target_color': target_color,
        'target_rank': target_rank,
        'prev_self_clued': data['self_clued'],
        'prev_partner_clued': data['partner_clued'],
    }


def _own_play_scores(scores, legal, mem, data):
    playable = _playable_matrix(data['heights'])
    possible = data['self_v0'] > 0.0
    poss_count = jnp.sum(possible, axis=(1, 2))
    playable_count = jnp.sum(possible & playable[None, :, :], axis=(1, 2))
    literal_safe = data['self_occ'] & (poss_count > 0) & (poss_count == playable_count)

    target_rank_ok = mem['target_rank'] >= 0
    target_color_known = mem['target_color'] >= 0
    safe_colors = jnp.maximum(mem['target_color'], 0)
    safe_ranks = jnp.maximum(mem['target_rank'], 0)
    color_rank_now = data['heights'][safe_colors]
    exact_target_possible = data['self_v0'][_slots(), safe_colors, safe_ranks] > 0.0
    color_target_ready = target_color_known & exact_target_possible & (color_rank_now == mem['target_rank'])
    remembered_ready = mem['play'] & target_rank_ok & target_color_known & color_target_ready

    should_play = data['self_occ'] & (literal_safe | ((~mem['hold']) & remembered_ready))
    slot_bonus = (4 - _slots()).astype(jnp.float32) * 0.1
    for s in range(5):
        scores = _apply_score(scores, 5 + s, legal, jnp.where(should_play[s], 900.0 + slot_bonus[s], -1.0e9))
    return scores


def _clue_scores(scores, legal, data):
    partner = data['partner']
    occ = data['partner_occ']
    partner_color, partner_rank = _partner_identity(partner)
    heights_at_card = data['heights'][partner_color]
    p_playable = occ & (partner_rank == heights_at_card) & (heights_at_card < 5)
    p_trash = occ & (partner_rank < heights_at_card)
    critical = _critical_matrix(data['heights'], data['disc_counts'])

    old_clued = data['partner_clued']
    chop = _lowest_slot(occ & (~old_clued))
    chop_valid = chop < 5
    chop_color = partner_color[jnp.minimum(chop, 4)]
    chop_rank = partner_rank[jnp.minimum(chop, 4)]
    same_visible = _visible_same_count(partner_color, partner_rank, occ, chop_color, chop_rank)
    chop_critical = critical[chop_color, chop_rank]
    save_5 = chop_valid & (chop_rank == 4)
    save_2 = chop_valid & (chop_rank == 1) & (same_visible <= 1)
    save_critical = chop_valid & chop_critical & (chop_rank != 4)

    for c in range(5):
        match = occ & (partner_color == c)
        focus = _clue_focus(match, old_clued, occ)
        valid_focus = (focus >= 0) & (focus < 5)
        newly = match & (~old_clued)
        num_new = jnp.sum(newly).astype(jnp.float32)
        new_bad = jnp.any(newly & p_trash)
        focus_playable = valid_focus & p_playable[jnp.maximum(focus, 0)]
        focus_slot_bonus = focus.astype(jnp.float32) * 0.2
        play_ok = focus_playable & (num_new > 0) & (~new_bad)
        tempo_ok = focus_playable & (num_new == 0) & old_clued[jnp.maximum(focus, 0)]
        save_ok = save_critical & (focus == chop) & (num_new > 0)
        play_score = jnp.where(play_ok, 820.0 + 8.0 * num_new + focus_slot_bonus, -1.0e9)
        tempo_score = jnp.where(tempo_ok, 700.0 + focus_slot_bonus, -1.0e9)
        save_score = jnp.where(save_ok, 965.0, -1.0e9)
        scores = _apply_score(scores, 10 + c, legal, jnp.maximum(jnp.maximum(play_score, tempo_score), save_score))

    for r in range(5):
        match = occ & (partner_rank == r)
        focus = _clue_focus(match, old_clued, occ)
        valid_focus = (focus >= 0) & (focus < 5)
        newly = match & (~old_clued)
        num_new = jnp.sum(newly).astype(jnp.float32)
        new_bad = jnp.any(newly & p_trash)
        focus_playable = valid_focus & p_playable[jnp.maximum(focus, 0)]
        focus_slot_bonus = focus.astype(jnp.float32) * 0.2
        play_ok = focus_playable & (num_new > 0) & (~new_bad)
        tempo_ok = focus_playable & (num_new == 0) & old_clued[jnp.maximum(focus, 0)]
        save_ok = ((save_5 & (r == 4)) | (save_2 & (r == 1)) | (save_critical & (r == chop_rank))) & (focus == chop) & (num_new > 0)
        play_score = jnp.where(play_ok, 800.0 + 8.0 * num_new + focus_slot_bonus, -1.0e9)
        tempo_score = jnp.where(tempo_ok, 690.0 + focus_slot_bonus, -1.0e9)
        save_score = jnp.where(save_ok, jnp.where((r == 4) | (r == 1), 980.0, 960.0), -1.0e9)
        scores = _apply_score(scores, 15 + r, legal, jnp.maximum(jnp.maximum(play_score, tempo_score), save_score))
    return scores


def _discard_scores(scores, legal, mem, data):
    trash = _trash_matrix(data['heights'])
    possible = data['self_v0'] > 0.0
    poss_count = jnp.sum(possible, axis=(1, 2))
    trash_count = jnp.sum(possible & trash[None, :, :], axis=(1, 2))
    literal_trash = data['self_occ'] & (poss_count > 0) & (poss_count == trash_count)
    discardable = data['self_occ'] & (~mem['play']) & (~mem['hold'])
    chop = _lowest_slot(discardable & (~data['self_clued']))
    forced_chop = _lowest_slot(data['self_occ'] & (~mem['play']) & (~data['self_clued']))
    emergency = _lowest_slot(data['self_occ'] & (~mem['play']))
    for s in range(5):
        s_f = jnp.asarray(s, dtype=jnp.float32)
        trash_score = jnp.where(literal_trash[s] & data['self_occ'][s], 560.0 + s_f * 0.1, -1.0e9)
        chop_score = jnp.where((chop == s) & (chop < 5), 320.0, -1.0e9)
        forced_score = jnp.where((forced_chop == s) & (forced_chop < 5), 40.0, -1.0e9)
        emergency_score = jnp.where((emergency == s) & (emergency < 5), 20.0, -1.0e9)
        scores = _apply_score(scores, s, legal, jnp.maximum(jnp.maximum(jnp.maximum(trash_score, chop_score), forced_score), emergency_score))
    return scores


def agent_act(obs, reset, memory, rng):
    del rng
    data = _decode(obs)
    legal = obs.legal_actions
    mem0 = _memory_after_public_action(memory, data, reset)
    mem1 = _notice_received_clue(mem0, data, reset)

    base = jnp.full((21,), -1.0e9)
    base = base.at[0:5].set(-300.0)
    base = base.at[5:10].set(-900.0)
    base = base.at[10:20].set(-200.0)
    base = base.at[20].set(0.0)
    scores = jnp.where(legal, base, -1.0e9)
    scores = _clue_scores(scores, legal, data)
    scores = _own_play_scores(scores, legal, mem1, data)
    scores = _discard_scores(scores, legal, mem1, data)
    scores = scores.at[20].set(jnp.where(legal[20], 0.0, scores[20]))

    best = jnp.argmax(scores).astype(jnp.int32)
    action = jnp.where(legal[best], best, _legal_fallback(legal)).astype(jnp.int32)
    trace = {
        'chosen_score': scores[action],
        'play_notes': jnp.sum(mem1['play']).astype(jnp.float32),
        'hold_notes': jnp.sum(mem1['hold']).astype(jnp.float32),
    }
    return action, mem1, trace
