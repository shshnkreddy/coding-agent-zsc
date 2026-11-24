"""H-group Level 1 oriented two-player Hanabi policy."""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'h_group_l1_conservative', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}

N_SLOTS = 5
N_COLORS = 5
N_RANKS = 5
MULTIPLICITIES = (3.0, 2.0, 2.0, 2.0, 1.0)


def _slots():
    return jnp.arange(N_SLOTS, dtype=jnp.int32)


def _ranks():
    return jnp.arange(N_RANKS, dtype=jnp.int32)


def _mult():
    return jnp.asarray(MULTIPLICITIES, dtype=jnp.float32)


def init_agent():
    neg = -jnp.ones((N_SLOTS,), dtype=jnp.int32)
    return {
        'clued': jnp.zeros((N_SLOTS,), dtype=jnp.bool_),
        'play': jnp.zeros((N_SLOTS,), dtype=jnp.bool_),
        'save': jnp.zeros((N_SLOTS,), dtype=jnp.bool_),
        'inf_color': neg,
        'inf_rank': neg,
    }


def _decode(features):
    ranks = _ranks()
    partner = features[0:125].reshape((N_SLOTS, N_COLORS, N_RANKS))
    partner_present = jnp.sum(partner, axis=(1, 2)) > 0.5
    fw_rows = features[167:192].reshape((N_COLORS, N_RANKS))
    heights = jnp.sum(fw_rows * (ranks.astype(jnp.float32) + 1.0), axis=1).astype(jnp.int32)
    info = jnp.sum(features[192:200]).astype(jnp.int32)
    disc_raw = features[203:253].reshape((N_COLORS, 10))
    disc = jnp.stack([
        jnp.sum(disc_raw[:, 0:3], axis=1),
        jnp.sum(disc_raw[:, 3:5], axis=1),
        jnp.sum(disc_raw[:, 5:7], axis=1),
        jnp.sum(disc_raw[:, 7:9], axis=1),
        jnp.sum(disc_raw[:, 9:10], axis=1),
    ], axis=1)
    know = features[308:658].reshape((2, N_SLOTS, 35))
    own_probs = know[0, :, 0:25].reshape((N_SLOTS, N_COLORS, N_RANKS))
    own_pos = jnp.any(know[0, :, 25:35] > 0.5, axis=1)
    partner_pos = jnp.any(know[1, :, 25:35] > 0.5, axis=1)
    last = features[253:308]
    return partner, partner_present, heights, info, disc, own_probs, own_pos, partner_pos, last


def _shift_vec(v, slot, fill):
    idx = jnp.arange(v.shape[0])
    src = jnp.where(idx < slot, idx, idx + 1)
    shifted = jnp.take(v, jnp.minimum(src, v.shape[0] - 1), axis=0)
    return jnp.where(idx == v.shape[0] - 1, fill, shifted)


def _shift_memory(memory, slot):
    return {
        'clued': _shift_vec(memory['clued'], slot, jnp.asarray(False)),
        'play': _shift_vec(memory['play'], slot, jnp.asarray(False)),
        'save': _shift_vec(memory['save'], slot, jnp.asarray(False)),
        'inf_color': _shift_vec(memory['inf_color'], slot, jnp.asarray(-1, dtype=jnp.int32)),
        'inf_rank': _shift_vec(memory['inf_rank'], slot, jnp.asarray(-1, dtype=jnp.int32)),
    }


def _focus_from_match(matched, prev_clued):
    slots = _slots()
    new = matched & (~prev_clued)
    has_new = jnp.any(new)
    unclued = ~prev_clued
    has_unclued = jnp.any(unclued)
    chop = jnp.min(jnp.where(unclued, slots, N_SLOTS - 1))
    chop = jnp.where(has_unclued, chop, jnp.asarray(0, dtype=jnp.int32))
    hi_new = jnp.max(jnp.where(new, slots, jnp.asarray(0, dtype=jnp.int32)))
    hi_match = jnp.max(jnp.where(matched, slots, jnp.asarray(0, dtype=jnp.int32)))
    chop_touched = matched[chop] & has_unclued
    focus = jnp.where(has_new, jnp.where(chop_touched, chop, hi_new), hi_match)
    return focus, chop, has_new, chop_touched


def _playable_mask(heights):
    return _ranks()[None, :] == heights[:, None]


def _trash_mask(heights, disc):
    ranks = _ranks()
    mult = _mult()
    already_played = ranks[None, :] < heights[:, None]
    exhausted = disc >= mult[None, :]
    blocked = jnp.cumsum(exhausted, axis=1) > 0
    return already_played | exhausted | blocked


def _identity_counts(hand):
    return jnp.sum(hand, axis=0)


def _card_color(hand, slot):
    return jnp.argmax(jnp.sum(hand[slot], axis=1)).astype(jnp.int32)


def _card_rank(hand, slot):
    return jnp.argmax(jnp.sum(hand[slot], axis=0)).astype(jnp.int32)


def _legal_or(action, fallback, legal):
    return jnp.where(legal[action], action, fallback)


def _candidate_focus_for_clue(hand, present, clued, clue_idx):
    is_color = clue_idx < N_COLORS
    value = jnp.where(is_color, clue_idx, clue_idx - N_COLORS)
    colors = jnp.argmax(jnp.sum(hand, axis=2), axis=1)
    ranks = jnp.argmax(jnp.sum(hand, axis=1), axis=1)
    matched = present & jnp.where(is_color, colors == value, ranks == value)
    focus, chop, has_new, chop_touched = _focus_from_match(matched, clued)
    return matched, focus, chop, has_new, chop_touched


def _best_play_clue(hand, present, clued, heights, disc, legal):
    playable = _playable_mask(heights)
    trash = _trash_mask(heights, disc)
    colors = jnp.argmax(jnp.sum(hand, axis=2), axis=1)
    ranks = jnp.argmax(jnp.sum(hand, axis=1), axis=1)
    slot_playable = present & playable[colors, ranks]
    slot_trash = present & trash[colors, ranks]

    def one(clue_idx):
        matched, focus, chop, has_new, chop_touched = _candidate_focus_for_clue(hand, present, clued, clue_idx)
        c = _card_color(hand, focus)
        r = _card_rank(hand, focus)
        focus_playable = playable[c, r] & present[focus]
        touches_trash = jnp.any(matched & slot_trash)
        is_rank = clue_idx >= N_COLORS
        rank_val = clue_idx - N_COLORS
        rank_save_shape = (rank_val == 4) | (rank_val == 1)
        ambiguous_save = is_rank & rank_save_shape & chop_touched & (focus == chop)
        new_playables = jnp.sum(matched & slot_playable)
        single = jnp.sum(matched.astype(jnp.int32)) == 1
        rank_ready = (rank_val == 0) | jnp.all(heights == rank_val)
        rank_ok = (~is_rank) | rank_ready
        type_bonus = jnp.where(is_rank, jnp.where(rank_val == 0, 15, 0), 20)
        valid = legal[10 + clue_idx] & has_new & rank_ok & focus_playable & (~touches_trash) & (~ambiguous_save)
        score = jnp.where(valid, 1000 + 50 * new_playables + 8 * single + type_bonus + focus, -100000)
        return score

    scores = jax.vmap(one)(jnp.arange(10, dtype=jnp.int32))
    best = jnp.argmax(scores).astype(jnp.int32)
    return 10 + best, jnp.max(scores) > 0


def _best_save_clue(hand, present, clued, heights, disc, legal):
    trash = _trash_mask(heights, disc)
    counts = _identity_counts(hand)

    def one(clue_idx):
        matched, focus, chop, has_new, chop_touched = _candidate_focus_for_clue(hand, present, clued, clue_idx)
        c = _card_color(hand, focus)
        r = _card_rank(hand, focus)
        is_rank = clue_idx >= N_COLORS
        rank_val = clue_idx - N_COLORS
        other_visible = (counts[c, r] - 1.0) > 0.5
        rank5_save = is_rank & (rank_val == 4) & (r == 4)
        rank2_save = is_rank & (rank_val == 1) & (r == 1) & (~other_visible)
        mult = _mult()
        critical = (disc[c, r] >= mult[r] - 1.0) & (r != 4) & (r != heights[c]) & (~trash[c, r])
        critical_save = critical & ((~is_rank) | (rank_val == r))
        valid = legal[10 + clue_idx] & has_new & chop_touched & (focus == chop) & present[focus] & (rank5_save | rank2_save | critical_save)
        score = jnp.where(valid, 1000 + 10 * critical.astype(jnp.int32) + r, -100000)
        return score

    scores = jax.vmap(one)(jnp.arange(10, dtype=jnp.int32))
    best = jnp.argmax(scores).astype(jnp.int32)
    return 10 + best, jnp.max(scores) > 0


def _best_forced_clue(hand, present, clued, legal):
    def one(clue_idx):
        matched, _focus, _chop, has_new, _chop_touched = _candidate_focus_for_clue(hand, present, clued, clue_idx)
        count = jnp.sum(matched.astype(jnp.int32))
        valid = legal[10 + clue_idx] & (count > 0)
        no_new = ~has_new
        score = jnp.where(valid, 10 + 1000 * no_new.astype(jnp.int32) - count, -100000)
        return score

    scores = jax.vmap(one)(jnp.arange(10, dtype=jnp.int32))
    best = jnp.argmax(scores).astype(jnp.int32)
    return 10 + best, jnp.max(scores) > 0


def _update_memory(obs, memory):
    _partner, _partner_present, heights, _info, disc, own_probs, own_pos, _partner_pos, last = _decode(obs.features)
    actor_self = last[0] > 0.5
    is_play_or_discard = (last[2] > 0.5) | (last[3] > 0.5)
    removed_slot = jnp.argmax(last[23:28]).astype(jnp.int32)
    shifted = jax.lax.cond(actor_self & is_play_or_discard, lambda m: _shift_memory(m, removed_slot), lambda m: m, memory)

    is_color_hint = last[4] > 0.5
    is_rank_hint = last[5] > 0.5
    hint_to_self = (is_color_hint | is_rank_hint) & (last[6] > 0.5)
    matched = last[18:23] > 0.5
    focus, chop, has_new, chop_touched = _focus_from_match(matched, shifted['clued'])
    rank_val = jnp.argmax(last[13:18]).astype(jnp.int32)
    color_val = jnp.argmax(last[8:13]).astype(jnp.int32)
    rank_save = is_rank_hint & chop_touched & (focus == chop) & ((rank_val == 4) | (rank_val == 1))
    mult = _mult()
    rank_grid = _ranks()[None, :]
    critical_mask = (disc >= mult[None, :] - 1.0) & (rank_grid != 4) & (rank_grid != heights[:, None])
    possible_critical = jnp.any((own_probs[focus] > 0.0001) & critical_mask)
    critical_save = chop_touched & (focus == chop) & possible_critical
    is_save = hint_to_self & (rank_save | critical_save)
    focus_certain_playable = jnp.sum(own_probs[focus] * _playable_mask(heights)) > 0.999
    is_play = hint_to_self & (~is_save) & (has_new | focus_certain_playable)

    new_clued = jnp.where(hint_to_self, shifted['clued'] | matched, shifted['clued'])
    play = shifted['play'].at[focus].set(jnp.where(is_play, True, shifted['play'][focus]))
    save = shifted['save'].at[focus].set(jnp.where(is_save, True, shifted['save'][focus]))
    inf_color = shifted['inf_color'].at[focus].set(jnp.where(hint_to_self & is_color_hint, color_val, shifted['inf_color'][focus]))
    stored_rank = jnp.where(is_color_hint, heights[color_val], rank_val)
    inf_rank = shifted['inf_rank'].at[focus].set(jnp.where(hint_to_self, stored_rank, shifted['inf_rank'][focus]))

    present = jnp.sum(own_probs, axis=(1, 2)) > 0.5
    synced_clued = (new_clued | own_pos) & present
    play = play & present
    save = save & present
    inf_color = jnp.where(present, inf_color, -jnp.ones_like(inf_color))
    inf_rank = jnp.where(present, inf_rank, -jnp.ones_like(inf_rank))
    return {'clued': synced_clued, 'play': play, 'save': save, 'inf_color': inf_color, 'inf_rank': inf_rank}


def _choose_action(obs, memory):
    legal = obs.legal_actions
    partner, partner_present, heights, _info, disc, own_probs, _own_pos, partner_pos, _last = _decode(obs.features)
    playable = _playable_mask(heights)
    trash = _trash_mask(heights, disc)
    own_present = jnp.sum(own_probs, axis=(1, 2)) > 0.5
    prob_playable = jnp.sum(own_probs * playable[None, :, :], axis=(1, 2))
    prob_trash = jnp.sum(own_probs * trash[None, :, :], axis=(1, 2))
    certain_playable = own_present & (prob_playable > 0.999)
    certain_trash = own_present & (prob_trash > 0.999)

    known_color = memory['inf_color'] >= 0
    known_rank = memory['inf_rank'] >= 0
    exact_ready = known_color & known_rank & (heights[jnp.maximum(memory['inf_color'], 0)] == memory['inf_rank'])
    rank_ready = known_rank & (~known_color)

    def rank_slot_ready(slot):
        r = memory['inf_rank'][slot]
        compatible = own_probs[slot, :, jnp.maximum(r, 0)] > 0.0001
        ready = heights == r
        any_compat = jnp.any(compatible)
        return any_compat & jnp.all(jnp.where(compatible, ready, True))

    slots = _slots()
    rank_ready_each = jax.vmap(rank_slot_ready)(slots)
    convention_playable = memory['play'] & (exact_ready | (rank_ready & rank_ready_each) | certain_playable)
    play_slots = legal[5:10] & (certain_playable | convention_playable)
    play_scores = jnp.where(play_slots, 1000 + slots, -100000)
    play_action = 5 + jnp.argmax(play_scores).astype(jnp.int32)
    has_play = jnp.max(play_scores) > 0

    play_clue, has_play_clue = _best_play_clue(partner, partner_present, partner_pos, heights, disc, legal)
    save_clue, has_save_clue = _best_save_clue(partner, partner_present, partner_pos, heights, disc, legal)

    discard_slots = legal[0:5] & own_present
    trash_scores = jnp.where(discard_slots & certain_trash, 2000 + slots, -100000)
    trash_action = jnp.argmax(trash_scores).astype(jnp.int32)
    has_trash = jnp.max(trash_scores) > 0
    discardable_chop = discard_slots & (~memory['clued']) & (~memory['save']) & (~memory['play'])
    chop_scores = jnp.where(discardable_chop, 1000 - slots, -100000)
    chop_action = jnp.argmax(chop_scores).astype(jnp.int32)
    has_chop = jnp.max(chop_scores) > 0
    any_discard = jnp.any(discard_slots)
    fallback_discard = jnp.argmax(jnp.where(discard_slots, 1 + slots, -100000)).astype(jnp.int32)
    discard_action = jnp.where(has_trash, trash_action, jnp.where(has_chop, chop_action, fallback_discard))
    has_discard = has_trash | has_chop | any_discard

    first_legal = jnp.argmax(legal).astype(jnp.int32)
    forced_clue, has_forced_clue = _best_forced_clue(partner, partner_present, partner_pos, legal)
    fallback_action = jnp.where(has_forced_clue, forced_clue, first_legal)
    action = fallback_action
    action = jnp.where(has_discard, discard_action, action)
    action = jnp.where(has_save_clue, save_clue, action)
    action = jnp.where(has_play_clue, play_clue, action)
    action = jnp.where(has_play, play_action, action)
    action = _legal_or(action, first_legal, legal)
    inactive = legal[20]
    return jnp.where(inactive, jnp.asarray(20, dtype=jnp.int32), action)


def agent_act(obs, reset, memory, rng):
    del rng
    reset_memory = init_agent()
    base_memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), reset_memory, memory)
    next_memory = _update_memory(obs, base_memory)
    action = _choose_action(obs, next_memory)
    trace = {
        'action': action,
        'play_count': jnp.sum(next_memory['play'].astype(jnp.int32)),
        'save_count': jnp.sum(next_memory['save'].astype(jnp.int32)),
    }
    return action, next_memory, trace
