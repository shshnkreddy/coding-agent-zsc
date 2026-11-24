"""A conservative, observation-only H-group Level 1 Hanabi policy."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "hgroup_literal_safe",
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
    # Arrays are indexed by native slot. A pending bit is an implicit play
    # promise inferred from the focus of the most recent clue to us.
    return (
        jnp.zeros((5,), dtype=jnp.int32),
        jnp.zeros((5,), dtype=jnp.int32),
        -jnp.ones((5,), dtype=jnp.int32),
        -jnp.ones((5,), dtype=jnp.int32),
    )


def _shift_after_removal(values, slot):
    positions = jnp.arange(5, dtype=jnp.int32)
    source = jnp.minimum(positions + (positions >= slot).astype(jnp.int32), 4)
    shifted = values[source]
    return jnp.where(positions < 4, shifted, jnp.asarray(0, dtype=values.dtype))


def _focus(matches, clued, occupied):
    """Return the H-group focus using native slot orientation."""
    positions = jnp.arange(5, dtype=jnp.int32)
    newly = matches * (1 - clued) * occupied
    old = matches * clued * occupied
    unclued = (1 - clued) * occupied
    chop = jnp.min(jnp.where(unclued > 0, positions, 5))
    newest_new = jnp.max(jnp.where(newly > 0, positions, -1))
    newest_old = jnp.max(jnp.where(old > 0, positions, 0))
    chop_touched = (chop < 5) * (jnp.take(matches, jnp.minimum(chop, 4)) > 0)
    has_new = jnp.any(newly > 0)
    return jnp.where(has_new, jnp.where(chop_touched, chop, newest_new), newest_old).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    features = obs.features
    legal = obs.legal_actions
    base_clued, base_pending, base_promise_color, base_promise_rank = init_agent()
    clued = jnp.where(reset, base_clued, memory[0])
    pending = jnp.where(reset, base_pending, memory[1])
    promise_color = jnp.where(reset, base_promise_color, memory[2])
    promise_rank = jnp.where(reset, base_promise_rank, memory[3])

    # Decode the previous public action. Fields are relative to this observer.
    last = features[253:308]
    actor_self = last[0] > 0.5
    action_type = last[2:6]
    was_remove = actor_self * (action_type[0] + action_type[1] > 0.5)
    removed_slot = jnp.argmax(last[23:28]).astype(jnp.int32)
    clued = jnp.where(was_remove, _shift_after_removal(clued, removed_slot), clued)
    pending = jnp.where(was_remove, _shift_after_removal(pending, removed_slot), pending)
    promise_color = jnp.where(was_remove, _shift_after_removal(promise_color, removed_slot), promise_color)
    promise_rank = jnp.where(was_remove, _shift_after_removal(promise_rank, removed_slot), promise_rank)

    self_knowledge = features[308:483].reshape((5, 35))
    occupied = (jnp.sum(self_knowledge[:, :25], axis=1) > 0).astype(jnp.int32)
    is_hint = (action_type[2] + action_type[3]) > 0.5
    recipient_self = last[6] > 0.5
    clue_event = is_hint * recipient_self * (1 - was_remove)
    matches = (last[18:23] > 0.5).astype(jnp.int32)
    focus = _focus(matches, clued, occupied)
    focus_hot = (jnp.arange(5, dtype=jnp.int32) == focus).astype(jnp.int32)
    positions = jnp.arange(5, dtype=jnp.int32)
    pre_clue_unclued = (1 - clued) * occupied
    pre_clue_chop = jnp.min(jnp.where(pre_clue_unclued > 0, positions, 5))
    hint_is_color = action_type[2] > 0.5
    hint_color = jnp.argmax(last[8:13]).astype(jnp.int32)
    hint_rank = jnp.argmax(last[13:18]).astype(jnp.int32)
    chop_save = clue_event * (focus == pre_clue_chop) * (pre_clue_chop < 5)
    chop_save = chop_save * (1 - hint_is_color) * ((hint_rank == 1) | (hint_rank == 4))
    play_promise_event = clue_event * (1 - chop_save)
    clued = jnp.where(clue_event, clued | matches, clued)
    pending = jnp.where(play_promise_event, pending | focus_hot, pending)
    color_update = promise_color.at[jnp.minimum(focus, 4)].set(hint_color)
    rank_update = promise_rank.at[jnp.minimum(focus, 4)].set(hint_rank)
    promise_color = jnp.where(play_promise_event * hint_is_color, color_update, promise_color)
    promise_rank = jnp.where(play_promise_event * hint_is_color, -jnp.ones((5,), dtype=jnp.int32), promise_rank)
    promise_rank = jnp.where(play_promise_event * (1 - hint_is_color), rank_update, promise_rank)
    promise_color = jnp.where(play_promise_event * (1 - hint_is_color), -jnp.ones((5,), dtype=jnp.int32), promise_color)

    fireworks = features[167:192].reshape((5, 5))
    heights = jnp.sum(fireworks * (jnp.arange(5, dtype=jnp.int32) + 1), axis=1)

    # Native literal possibilities; all possible identities must be playable.
    possibilities = self_knowledge[:, :25].reshape((5, 5, 5)) > 0
    ranks = jnp.arange(5, dtype=jnp.int32)[None, None, :]
    playable_identity = ranks == heights[None, :, None]
    possible_count = jnp.sum(possibilities, axis=(1, 2))
    playable_count = jnp.sum(possibilities * playable_identity, axis=(1, 2))
    all_playable = (possible_count > 0) * (possible_count == playable_count) * occupied
    singleton_playable = (possible_count == 1) * (playable_count == 1) * occupied
    color_target = (promise_color[:, None, None] == jnp.arange(5)[None, :, None])
    rank_target = (promise_rank[:, None, None] == jnp.arange(5)[None, None, :])
    named_target = (color_target | rank_target) * possibilities
    named_count = jnp.sum(named_target, axis=(1, 2))
    named_playable = jnp.sum(named_target * playable_identity, axis=(1, 2)) > 0
    named_playable_count = jnp.sum(named_target * playable_identity, axis=(1, 2))
    rank_promise = promise_rank >= 0
    color_promise = promise_color >= 0
    promise_now = pending * occupied * (
        (rank_promise * (named_playable_count > 0))
        | (color_promise * (named_count > 0) * (named_count == named_playable_count))
    )
    clued_safe = clued * all_playable
    play_priority = 4 * promise_now + 3 * clued_safe + 2 * singleton_playable + all_playable
    play_slot = jnp.argmax(
        jnp.where(play_priority > 0, play_priority * 10 + jnp.arange(5), 0)
    ).astype(jnp.int32)
    have_play = jnp.any(play_priority > 0)
    play_action = 5 + play_slot

    # Decode the visible partner hand and its public clue flags.
    partner_hand = features[:125].reshape((5, 5, 5))
    partner_occupied = (jnp.sum(partner_hand, axis=(1, 2)) > 0).astype(jnp.int32)
    partner_cards = jnp.argmax(partner_hand, axis=2).astype(jnp.int32)
    partner_color = (partner_cards // 5).astype(jnp.int32)
    partner_rank = (partner_cards % 5).astype(jnp.int32)
    partner_knowledge = features[483:658].reshape((5, 35))
    partner_clued = (jnp.sum(partner_knowledge[:, 25:35], axis=1) > 0).astype(jnp.int32)
    partner_chop = jnp.min(jnp.where((1 - partner_clued) * partner_occupied > 0, jnp.arange(5), 5))
    partner_chop_valid = partner_chop < 5
    chop_idx = jnp.minimum(partner_chop, 4)
    chop_color = jnp.take(partner_color, chop_idx)
    chop_rank = jnp.take(partner_rank, chop_idx)
    chop_playable = partner_chop_valid * (jnp.take(heights, chop_color) == chop_rank)

    # A delayed play clue is supportable when every lower card in the same
    # firework is either visible and clued or is currently playable. This is
    # based only on the partner hand visible to this seat and public stacks.
    exact_partner = partner_hand > 0
    accounted = jnp.zeros((5, 5), dtype=jnp.int32)
    for color_i in range(5):
        for rank_i in range(5):
            seen = jnp.any(exact_partner[:, color_i, rank_i] > 0)
            accounted = accounted.at[color_i, rank_i].set(
                (seen * ((jnp.any((partner_clued > 0) * exact_partner[:, color_i, rank_i])) | (heights[color_i] == rank_i))).astype(jnp.int32)
            )
    lower_rank = jnp.arange(5, dtype=jnp.int32)[None, :]
    target_accounted = accounted[partner_color]
    target_height = jnp.take(heights, partner_color)
    needed = (lower_rank >= target_height[:, None]) & (lower_rank < partner_rank[:, None])
    lower_ok = jnp.all(jnp.where(needed, target_accounted > 0, True), axis=1)
    partner_playable = partner_occupied * (target_height == partner_rank)
    discard_counts = features[203:253].reshape((5, 10))
    rank_offsets = jnp.asarray([0, 3, 5, 7, 9], dtype=jnp.int32)
    discard_count = discard_counts[jnp.arange(5), rank_offsets[partner_rank]]
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)[partner_rank]
    critical = discard_count >= (multiplicity - 1)
    partner_save = partner_occupied * ((partner_rank == 4) | (partner_rank == 1) | critical)
    partner_save = partner_save * (1 - partner_playable)

    # Score the ten clue actions by their H-group-determined focus.
    clue_scores = jnp.full((10,), -1000000, dtype=jnp.int32)
    clue_useful = jnp.zeros((10,), dtype=jnp.int32)
    for clue_index in range(10):
        if clue_index < 5:
            matches_i = partner_occupied * (partner_color == clue_index)
        else:
            matches_i = partner_occupied * (partner_rank == clue_index - 5)
        focus_i = _focus(matches_i, partner_clued, partner_occupied)
        focus_valid = focus_i < 5
        focus_play_i = jnp.take(partner_playable, jnp.minimum(focus_i, 4)) * focus_valid
        focus_save_i = jnp.take(partner_save, jnp.minimum(focus_i, 4)) * focus_valid
        value = 600 * focus_play_i + 400 * focus_save_i
        value = value + 2 * (clue_index < 5) + 20 * (clue_index >= 5) * focus_play_i
        value = value + jnp.sum(matches_i)
        clue_scores = clue_scores.at[clue_index].set(value)
        clue_useful = clue_useful.at[clue_index].set((focus_play_i | focus_save_i).astype(jnp.int32))
    clue_legal = legal[10:20]
    clue_available = features[192] > 0.5
    forced_clue = features[199] > 0.5
    useful_clue = clue_legal * clue_available * ((clue_useful > 0) | forced_clue)
    best_clue_offset = jnp.argmax(jnp.where(useful_clue, clue_scores, -1000000)).astype(jnp.int32)
    clue_action = 10 + best_clue_offset
    have_clue = jnp.any(useful_clue > 0)

    # A clued card whose every literal identity is below its firework height
    # is provably useless. H-group permits discarding it before chop.
    # If there is no useful play or clue, discard our native chop.
    own_chop = jnp.min(jnp.where((1 - clued) * occupied > 0, jnp.arange(5), 5))
    own_chop = jnp.minimum(own_chop, 4)
    discard_action = own_chop
    preferred = jnp.where(have_play, play_action, jnp.where(have_clue, clue_action, discard_action))
    preferred_legal = legal[preferred]
    fallback = jnp.argmax(legal).astype(jnp.int32)
    action = jnp.where(preferred_legal, preferred, fallback).astype(jnp.int32)
    return action, (clued, pending, promise_color, promise_rank), {}
