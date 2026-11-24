"""Convention-light Hanabi policy based on public literal knowledge."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    # Previous positive hint flags, used only to detect a newly touched focus.
    return {
        'positive': jnp.zeros((5, 10), dtype=jnp.bool_),
        'pending': jnp.zeros((5,), dtype=jnp.bool_),
        'deck': jnp.asarray(40, dtype=jnp.int32),
        'final_drawer': jnp.asarray(False, dtype=jnp.bool_),
    }


def _masked_argmax(values, mask):
    return jnp.argmax(jnp.where(mask, values, -1.0e9)).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions

    # Public state.
    partner = f[:125].reshape(5, 5, 5)
    fireworks = f[167:192].reshape(5, 5)
    heights = jnp.sum(fireworks * jnp.arange(1, 6)[None, :], axis=1).astype(jnp.int32)
    info = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)

    # Public remaining multiplicities.  Successful cards are no longer in the
    # unseen pool, just like discarded cards.
    d = f[203:253].reshape(5, 10)
    discards = jnp.stack(
        [jnp.sum(d[:, 0:3], axis=1), jnp.sum(d[:, 3:5], axis=1),
         jnp.sum(d[:, 5:7], axis=1), jnp.sum(d[:, 7:9], axis=1),
         jnp.sum(d[:, 9:10], axis=1)], axis=1)
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.float32)[None, :]
    played = jnp.arange(5)[None, :] < heights[:, None]
    remaining = multiplicity - discards - played.astype(jnp.float32)

    knowledge = f[308:658].reshape(2, 5, 35)
    native_own = knowledge[0, :, :25].reshape(5, 5, 5)
    partner_belief = knowledge[1, :, :25].reshape(5, 5, 5)
    own_positive = knowledge[0, :, 25:35] > 0.5
    partner_positive = knowledge[1, :, 25:35] > 0.5
    previous_positive = jnp.where(reset, jnp.zeros_like(memory['positive']), memory['positive'])
    previous_pending = jnp.where(reset, jnp.zeros_like(memory['pending']), memory['pending'])
    previous_deck = jnp.where(reset, jnp.asarray(40, dtype=jnp.int32), memory['deck'])
    occupied = jnp.sum(native_own, axis=(1, 2)) > 0.0

    # The native V0 deliberately does not subtract the visible hand.  Recover
    # its literal support and reweight it with the counts actually unavailable
    # to this observer.  A few fixed singleton-elimination passes provide safe
    # cross-slot deductions while retaining static JAX shapes.
    visible_counts = jnp.sum(partner, axis=0)
    available = jnp.maximum(remaining - visible_counts, 0.0)
    support = native_own > 0.0

    def refine(s):
        singleton = (jnp.sum(s, axis=(1, 2)) == 1) & occupied
        fixed = jnp.sum(s * singleton[:, None, None], axis=0)
        per_slot_available = available[None, :, :] - fixed[None, :, :] + s * singleton[:, None, None]
        w = s * jnp.maximum(per_slot_available, 0.0)
        return w > 0.0

    support = refine(support)
    support = refine(support)
    support = refine(support)

    singleton = (jnp.sum(support, axis=(1, 2)) == 1) & occupied
    fixed = jnp.sum(support * singleton[:, None, None], axis=0)
    weights = support * jnp.maximum(
        available[None, :, :] - fixed[None, :, :] + support * singleton[:, None, None], 0.0)
    own_belief = weights / jnp.maximum(jnp.sum(weights, axis=(1, 2), keepdims=True), 1.0e-8)

    playable_types = jnp.arange(5)[None, :] == heights[:, None]
    own_play_prob = jnp.sum(own_belief * playable_types[None, :, :], axis=(1, 2))
    own_five_prob = jnp.sum(own_belief[:, :, 4], axis=1)
    certain_play = occupied & (own_play_prob > 0.999)
    play_action = 5 + _masked_argmax(own_play_prob + 0.1 * own_five_prob, certain_play)

    # A newly touched newest card is a common, locally observable play focus.
    # Trust it only if the giver had room to discard (post-clue tokens <= 6),
    # and never stake the final life on this convention.
    last = f[253:308]
    just_emptied = (deck == 0) & (previous_deck > 0)
    final_drawer = jnp.where(
        reset, False,
        jnp.where(just_emptied, last[0] > 0.5, memory['final_drawer']))
    was_color_hint = last[1] > 0.5
    was_color_hint = was_color_hint & (last[4] > 0.5) & (last[6] > 0.5)
    was_rank_hint = (last[1] > 0.5) & (last[5] > 0.5) & (last[6] > 0.5)
    named_color = jnp.argmax(last[8:13])
    named_rank = jnp.argmax(last[13:18])
    named_rank_needed = jnp.any(heights == named_rank)
    match = last[18:23] > 0.5
    new_color = own_positive[:, named_color] & (~previous_positive[:, named_color]) & match
    new_rank = own_positive[:, 5 + named_rank] & (~previous_positive[:, 5 + named_rank]) & match
    newly_touched = jnp.where(was_color_hint, new_color, jnp.where(was_rank_hint, new_rank, False))
    fresh_focus_slot = _masked_argmax(jnp.arange(5, dtype=jnp.float32), newly_touched)
    focus_kind = was_color_hint | (was_rank_hint & named_rank_needed)
    focus_context = (info <= 7) & focus_kind
    literal_touch = jnp.any(newly_touched & certain_play)
    fresh_focus = jnp.any(newly_touched) & focus_context & (~literal_touch)
    new_pending = (jnp.arange(5) == fresh_focus_slot) & fresh_focus

    # Align a remembered focus after this player's previous play/discard shifted
    # its own hand.  The public removed slot is in pre-removal coordinates.
    self_removed = (last[0] > 0.5) & ((last[2] > 0.5) | (last[3] > 0.5))
    removed_slot = jnp.argmax(last[23:28])
    slots = jnp.arange(5)
    source = jnp.minimum(slots + (slots >= removed_slot), 4)
    shifted_pending = jnp.where(slots < 4, previous_pending[source], False)
    carried_pending = jnp.where(self_removed, shifted_pending, previous_pending)
    pending = carried_pending | new_pending
    focus_slot = _masked_argmax(jnp.arange(5, dtype=jnp.float32), pending)
    focus_play = jnp.any(pending) & (lives > 1)
    focus_action = 5 + focus_slot

    # Score literal hints by the public information they add and, especially,
    # by how many actually playable partner cards become certain afterwards.
    actual_playable = jnp.sum(partner * playable_types[None, :, :], axis=(1, 2)) > 0.5
    partner_occ = jnp.sum(partner, axis=(1, 2)) > 0.5
    partner_unhinted = partner_occ & (~jnp.any(partner_positive, axis=1))
    chop_slot = _masked_argmax(-jnp.arange(5, dtype=jnp.float32), partner_unhinted)
    has_chop = jnp.any(partner_unhinted)
    old_pp = jnp.sum(partner_belief * playable_types[None, :, :], axis=(1, 2))
    rank_ids = jnp.arange(5)[None, :]
    prerequisite_available = (rank_ids < heights[:, None]) | (remaining > 0.0)
    reachable_prefix = jnp.cumprod(prerequisite_available.astype(jnp.int32), axis=1) > 0
    reachable_live = (rank_ids >= heights[:, None]) & reachable_prefix
    critical_type = reachable_live & (remaining <= 1.0)

    def color_hint_score(c):
        attr = jnp.arange(5)[:, None] == c
        actual_match = jnp.sum(partner[:, c, :], axis=1) > 0.5
        q = jnp.sum(partner_belief[:, c, :], axis=1)
        keep = jnp.where(actual_match[:, None, None], attr[None, :, :], ~attr[None, :, :])
        post = partner_belief * keep
        z = jnp.sum(post, axis=(1, 2))
        post_pp = jnp.sum(post * playable_types[None, :, :], axis=(1, 2)) / jnp.maximum(z, 1.0e-8)
        new_safe = actual_playable & (old_pp < 0.999) & (post_pp > 0.999)
        gain = jnp.sum(jnp.where(partner_occ, jnp.where(actual_match, 1.0-q, q), 0.0))
        novel = actual_match & (~partner_positive[:, c])
        focus = _masked_argmax(jnp.arange(5, dtype=jnp.float32), novel)
        conventional = jnp.any(novel) & actual_playable[focus]
        focus_five = conventional & (partner[focus, c, 4] > 0.5)
        score = 60.0 * conventional + 6.0 * focus_five + 20.0 * jnp.sum(new_safe) + 2.0 * jnp.sum(actual_match & actual_playable) + gain
        return score, jnp.any(novel), conventional, gain

    def rank_hint_score(r):
        attr = jnp.arange(5)[None, :] == r
        actual_match = jnp.sum(partner[:, :, r], axis=1) > 0.5
        q = jnp.sum(partner_belief[:, :, r], axis=1)
        keep = jnp.where(actual_match[:, None, None], attr[None, :, :], ~attr[None, :, :])
        post = partner_belief * keep
        z = jnp.sum(post, axis=(1, 2))
        post_pp = jnp.sum(post * playable_types[None, :, :], axis=(1, 2)) / jnp.maximum(z, 1.0e-8)
        new_safe = actual_playable & (old_pp < 0.999) & (post_pp > 0.999)
        gain = jnp.sum(jnp.where(partner_occ, jnp.where(actual_match, 1.0-q, q), 0.0))
        novel = actual_match & (~partner_positive[:, 5+r])
        focus = _masked_argmax(jnp.arange(5, dtype=jnp.float32), novel)
        conventional = jnp.any(novel) & actual_playable[focus]
        chop_critical = jnp.sum(partner[chop_slot, :, r] * critical_type[:, r]) > 0.5
        high_value_save = (has_chop & novel[chop_slot] & chop_critical &
                           (r >= 1) & (r <= 3) &
                           (~jnp.any(heights == r)))
        focus_five = conventional & (r == 4)
        score = (60.0 * conventional + 6.0 * focus_five + 40.0 * high_value_save +
                 20.0 * jnp.sum(new_safe) +
                 2.0 * jnp.sum(actual_match & actual_playable) + gain)
        return score, jnp.any(novel), conventional, gain

    color_results = [color_hint_score(c) for c in range(5)]
    rank_results = [rank_hint_score(r) for r in range(5)]
    color_scores = jnp.stack([x[0] for x in color_results])
    rank_scores = jnp.stack([x[0] for x in rank_results])
    hint_scores = jnp.concatenate([color_scores, rank_scores])
    hint_novel = jnp.concatenate([
        jnp.stack([x[1] for x in color_results]),
        jnp.stack([x[1] for x in rank_results])])
    hint_gain = jnp.concatenate([
        jnp.stack([x[3] for x in color_results]),
        jnp.stack([x[3] for x in rank_results])])
    hint_legal = legal[10:20]
    hint_idx = _masked_argmax(hint_scores, hint_legal)
    hint_action = 10 + hint_idx
    best_hint_score = jnp.max(jnp.where(hint_legal, hint_scores, -1.0e9))
    useful_play_hint = best_hint_score >= 20.0

    # A discard is guaranteed non-critical when every still-useful identity in
    # its literal support has another public copy remaining.
    live = reachable_live
    critical_prob = jnp.sum(own_belief * critical_type[None, :, :], axis=(1, 2))
    stack_loss = (5 - jnp.arange(5, dtype=jnp.float32))[None, :]
    critical_loss = jnp.sum(
        own_belief * critical_type[None, :, :] * stack_loss[None, :, :], axis=(1, 2))
    dead_prob = jnp.sum(own_belief * (~live)[None, :, :], axis=(1, 2))
    gap = jnp.maximum(jnp.arange(5)[None, :] - heights[:, None], 0)
    gap_score = jnp.sum(own_belief * gap[None, :, :], axis=(1, 2))
    redundancy_score = jnp.sum(
        own_belief * jnp.maximum(remaining - 1.0, 0.0)[None, :, :], axis=(1, 2))
    safe_discard = occupied & (critical_prob < 1.0e-6) & legal[:5]
    discard_value = 10.0 * dead_prob + 0.7 * gap_score + 0.3 * redundancy_score - 2.0 * own_play_prob
    safe_discard_action = _masked_argmax(discard_value, safe_discard)
    any_safe_discard = jnp.any(safe_discard)

    # Least-dangerous fallback discard; prefer old/left cards on ties.
    fallback_discard = _masked_argmax(
        -40.0 * critical_loss + 10.0 * dead_prob + 0.7 * gap_score +
        0.3 * redundancy_score - 2.0 * own_play_prob - 0.01*jnp.arange(5), legal[:5])
    any_hint = jnp.any(hint_legal)
    quiet_hint_legal = hint_legal & (~hint_novel) & (hint_gain > 1.0)
    quiet_hint_action = 10 + _masked_argmax(hint_scores, quiet_hint_legal)
    any_quiet_hint = jnp.any(quiet_hint_legal)
    unneeded_ranks = ~jnp.any(heights[:, None] == jnp.arange(5)[None, :], axis=0)
    ceiling_safe_rank = legal[15:20] & unneeded_ranks
    ceiling_rank_action = 15 + jnp.where(
        jnp.any(ceiling_safe_rank),
        _masked_argmax(rank_scores, ceiling_safe_rank),
        _masked_argmax(rank_scores, legal[15:20]))
    generic_hint_action = jnp.where(info == 8, ceiling_rank_action, quiet_hint_action)
    informative_hint = jnp.where(info == 8, any_hint, any_quiet_hint)

    # Near the end, unused hints cannot create extra turns.  A probabilistic
    # play is preferable to throwing away a likely point, but only with spare
    # lives; otherwise retain the conservative ordering.
    uncertain_play_action = 5 + _masked_argmax(own_play_prob, occupied & legal[5:10])
    end_risk_play = (deck == 0) & (lives > 1) & (jnp.max(own_play_prob) > 1.0e-6)
    final_turn_play = final_drawer & (deck == 0) & (lives > 1) & jnp.any(occupied)

    active_action = jnp.where(
        jnp.any(certain_play), play_action,
        jnp.where(
            focus_play, focus_action,
            jnp.where(
                final_turn_play, uncertain_play_action,
                jnp.where(
                    useful_play_hint, hint_action,
                    jnp.where(
                        end_risk_play, uncertain_play_action,
                        jnp.where(
                            any_safe_discard, safe_discard_action,
                            jnp.where(informative_hint, generic_hint_action, fallback_discard)))))))
    action = jnp.where(legal[20], 20, active_action).astype(jnp.int32)
    trace = {
        'play_prob': jnp.max(own_play_prob),
        'hint_score': best_hint_score,
        'critical_prob': jnp.min(jnp.where(occupied, critical_prob, 2.0)),
        'deck': deck,
        'info': info,
        'lives': lives,
        'heights': heights,
        'focus_play': focus_play,
    }
    next_memory = {
        'positive': own_positive,
        'pending': pending,
        'deck': deck,
        'final_drawer': final_drawer,
    }
    return action, next_memory, trace
