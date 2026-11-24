"""A small, observation-only H-group Level 1 Hanabi policy."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'h_group_literal_heuristic', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.zeros((12,), dtype=jnp.int32)


def _firework_heights(features):
    rows = features[167:192].reshape((5, 5))
    return jnp.sum(rows * (jnp.arange(5, dtype=features.dtype) + 1), axis=1)


def _discard_counts(features):
    rows = features[203:253].reshape((5, 10))
    return jnp.stack((
        jnp.sum(rows[:, 0:3], axis=1), jnp.sum(rows[:, 3:5], axis=1),
        jnp.sum(rows[:, 5:7], axis=1), jnp.sum(rows[:, 7:9], axis=1),
        rows[:, 9]), axis=1)


def _one_hot(index, size):
    return (jnp.arange(size, dtype=jnp.int32) == index).astype(jnp.bool_)


def _shift_flags(flags, removed):
    table = jnp.stack(tuple(jnp.concatenate((flags[s + 1:], jnp.zeros((s + 1,), dtype=flags.dtype))) for s in range(5)))
    return jnp.sum(table * removed[:, None].astype(flags.dtype), axis=0)


def _focus_for_touch(touch, old_clued, occupied):
    unclued = occupied & (~old_clued)
    chop_weight = jnp.arange(5, 0, -1, dtype=jnp.int32)
    chop = jnp.argmax(unclued.astype(jnp.int32) * chop_weight)
    chop_one = _one_hot(chop, 5) & (jnp.sum(unclued) > 0)
    new = touch & (~old_clued) & occupied
    any_new = jnp.sum(new) > 0
    touches_chop = jnp.sum(touch & chop_one) > 0
    left_weight = jnp.arange(1, 6, dtype=jnp.int32)
    left_new = jnp.argmax(new.astype(jnp.int32) * left_weight)
    retouched = touch & old_clued & occupied
    left_retouch = jnp.argmax(retouched.astype(jnp.int32) * left_weight)
    focus = jnp.where(any_new, jnp.where(touches_chop, chop_one, _one_hot(left_new, 5)), _one_hot(left_retouch, 5))
    return focus, chop_one


def agent_act(obs, reset, memory, rng):
    del rng
    f = obs.features
    legal = obs.legal_actions
    m = jnp.where(reset, jnp.zeros((12,), dtype=jnp.int32), memory)

    fw = _firework_heights(f)
    discarded = _discard_counts(f)
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=f.dtype)
    occupied_self = jnp.arange(5) < (5 - f[125]).astype(jnp.int32)
    occupied_partner = jnp.arange(5) < (5 - f[126]).astype(jnp.int32)

    knowledge = f[308:].reshape((2, 5, 35))
    own_poss = knowledge[0, :, 0:25] > 1.0e-6
    own_color_hints = knowledge[0, :, 25:30] > 0.5
    own_rank_hints = knowledge[0, :, 30:35] > 0.5
    current_clued = jnp.sum(jnp.concatenate((own_color_hints, own_rank_hints), axis=1), axis=1) > 0

    last = f[253:308]
    last_self = last[0] > 0.5
    last_play_discard = (last[2] + last[3]) > 0.5
    removed = last[23:28] > 0.5
    shifted = _shift_flags(m[:5], removed)
    old_clued = jnp.where(last_self & last_play_discard, shifted, m[:5].astype(jnp.bool_))
    hint_to_self = last[6] > 0.5
    touch_self = (last[18:23] > 0.5) & occupied_self
    own_focus, own_chop = _focus_for_touch(touch_self, old_clued, occupied_self)
    own_focus_idx = jnp.argmax(own_focus.astype(jnp.int32))
    own_focus_chop = jnp.sum(own_focus & own_chop) > 0

    type_colors = jnp.repeat(jnp.arange(5), 5)
    type_ranks = jnp.tile(jnp.arange(5), 5)
    type_is_playable = fw[type_colors] == type_ranks
    own_any = jnp.sum(own_poss, axis=1) > 0
    own_bad = jnp.sum(own_poss & (~type_is_playable[None, :]), axis=1) > 0
    own_safe_play = occupied_self & own_any & (~own_bad)
    focus_poss = own_poss[own_focus_idx]
    hinted_rank_idx = jnp.argmax(last[13:18])
    rank_focus_poss = focus_poss.reshape((5, 5))[:, hinted_rank_idx]
    rank_focus_has = jnp.sum(rank_focus_poss) > 0
    rank_focus_all_play = rank_focus_has & (jnp.sum(rank_focus_poss & (~(fw == hinted_rank_idx))) == 0)
    color_focus_any_play = jnp.sum(focus_poss & type_is_playable) > 0
    focus_has_play_poss = jnp.where(last[5] > 0.5, rank_focus_all_play, color_focus_any_play)
    focus_play = hint_to_self & own_focus & (~own_focus_chop) & focus_has_play_poss
    own_play = own_safe_play | focus_play
    play_weight = jnp.arange(1, 6, dtype=jnp.int32)
    play_score = own_play.astype(jnp.int32) * (play_weight + 1000 * focus_play.astype(jnp.int32))
    play_slot = jnp.argmax(play_score)
    play_action = 5 + play_slot

    partner_hand = f[0:125].reshape((5, 5, 5))
    partner_present = jnp.sum(partner_hand, axis=(1, 2)) > 0.5
    partner_color = jnp.argmax(jnp.sum(partner_hand, axis=2), axis=1)
    partner_rank = jnp.argmax(jnp.sum(partner_hand, axis=1), axis=1)
    partner_knowledge = knowledge[1]
    partner_clued = jnp.sum(partner_knowledge[:, 25:35], axis=1) > 0
    partner_playable = partner_present & (partner_rank == fw[partner_color])
    partner_identity_count = jnp.zeros((5, 5), dtype=f.dtype)
    partner_identity_count = partner_identity_count.at[partner_color, partner_rank].add(partner_present.astype(f.dtype))
    critical = discarded >= (multiplicity[None, :] - 1)
    unique_two = (partner_identity_count[:, 1] <= 1) & (discarded[:, 1] < 1)
    partner_important = (
        (partner_rank == 4) |
        ((partner_rank == 1) & (unique_two[partner_color] | critical[partner_color, 1])) |
        critical[partner_color, partner_rank]
    ) & partner_present

    p_unclued = partner_present & (~partner_clued)
    p_chop_weight = jnp.arange(5, 0, -1, dtype=jnp.int32)
    p_chop_idx = jnp.argmax(p_unclued.astype(jnp.int32) * p_chop_weight)
    p_chop_one = _one_hot(p_chop_idx, 5) & (jnp.sum(p_unclued) > 0)
    p_good = partner_present & (partner_rank >= fw[partner_color])
    p_playable = partner_playable

    candidate_scores = []
    for clue_kind in range(2):
        for value in range(5):
            touch = jnp.where(clue_kind == 0, partner_present & (partner_color == value), partner_present & (partner_rank == value))
            new = touch & (~partner_clued)
            any_touch = jnp.sum(touch) > 0
            valid_good_touch = jnp.sum(touch & (~p_good)) == 0
            has_new_good = jnp.sum(new & p_good) > 0
            focus, _ = _focus_for_touch(touch, partner_clued, partner_present)
            focus_idx = jnp.argmax(focus.astype(jnp.int32))
            focus_play_value = jnp.sum(focus & p_playable) > 0
            focus_save_value = jnp.sum(focus & partner_important & p_chop_one) > 0
            rank_save_form = (clue_kind == 1) & ((value == 1) | (value == 4))
            critical_non_five = critical[partner_color, partner_rank] & (partner_rank < 4)
            save_ok = focus_save_value & (rank_save_form | jnp.any((focus & partner_important & p_chop_one) & critical_non_five))
            play_ok = focus_play_value
            valid = any_touch & ((valid_good_touch & has_new_good) | save_ok)
            score = valid.astype(jnp.int32) * (
                play_ok.astype(jnp.int32) * (1200 + 20 * jnp.sum(touch & p_playable).astype(jnp.int32)) +
                save_ok.astype(jnp.int32) * (900 + 10 * jnp.sum(touch & p_good).astype(jnp.int32)) +
                jnp.sum(new & p_good).astype(jnp.int32) + int(clue_kind == 0))
            candidate_scores.append(score)
    candidate_scores = jnp.stack(candidate_scores)
    best_clue_idx = jnp.argmax(candidate_scores)
    best_clue_score = jnp.max(candidate_scores)
    best_clue_action = jnp.where(best_clue_idx < 5, 10 + best_clue_idx, 15 + (best_clue_idx - 5))

    partner_has_play = jnp.sum(partner_playable) > 0
    clue_tokens = jnp.sum(f[192:200]) > 0
    choose_clue = clue_tokens & (best_clue_score > 0) & ((~jnp.any(own_play)) | ((best_clue_score >= 900) & (~partner_has_play)))

    discard_score = (~current_clued & occupied_self).astype(jnp.int32) * jnp.arange(5, 0, -1, dtype=jnp.int32)
    discard_score = jnp.where(jnp.max(discard_score) > 0, discard_score, occupied_self.astype(jnp.int32) * jnp.arange(5, 0, -1, dtype=jnp.int32))
    discard_slot = jnp.argmax(discard_score)
    discard_action = discard_slot

    desired = jnp.where(jnp.any(own_play), play_action, discard_action)
    desired = jnp.where(choose_clue, best_clue_action, desired)
    action_scores = jnp.where(jnp.arange(21) == desired, 1000, 0).astype(jnp.int32)
    action = jnp.argmax(jnp.where(legal, action_scores + jnp.arange(21, dtype=jnp.int32), -1)).astype(jnp.int32)
    next_memory = m.at[:5].set(current_clued.astype(jnp.int32))
    return action, next_memory, {}
