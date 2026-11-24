"""Convention-legible H-group Level 1 policy for two-player Hanabi."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_l1_foundation', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        'intent': jnp.zeros((5, 25), dtype=jnp.bool_),
        'status': jnp.zeros((5,), dtype=jnp.int32),
        'clued': jnp.zeros((5,), dtype=jnp.bool_),
        'turn': jnp.asarray(0, dtype=jnp.int32),
    }


def _shift_after_remove(x, slot, fill):
    variants = []
    for k in range(5):
        variants.append(jnp.concatenate((x[:k], x[k + 1:], fill[None]), axis=0))
    return jnp.stack(variants, axis=0)[slot]


def _first_true(mask):
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _last_true(mask):
    return (4 - jnp.argmax(mask[::-1].astype(jnp.int32))).astype(jnp.int32)


def _decode(features):
    partner = features[0:125].reshape(5, 5, 5)
    fireworks_oh = features[167:192].reshape(5, 5)
    heights = jnp.sum(fireworks_oh * jnp.arange(1, 6)[None, :], axis=1).astype(jnp.int32)
    info = jnp.sum(features[192:200]).astype(jnp.int32)
    lives = jnp.sum(features[200:203]).astype(jnp.int32)
    raw_discards = features[203:253].reshape(5, 10)
    discards = jnp.stack((jnp.sum(raw_discards[:, 0:3], axis=1), jnp.sum(raw_discards[:, 3:5], axis=1), jnp.sum(raw_discards[:, 5:7], axis=1), jnp.sum(raw_discards[:, 7:9], axis=1), raw_discards[:, 9]), axis=1).astype(jnp.int32)
    knowledge = features[308:658].reshape(2, 5, 35)
    literal = knowledge[:, :, :25].reshape(2, 5, 5, 5)
    clued = jnp.any(knowledge[:, :, 25:] > 0.5, axis=2)
    occupied = jnp.sum(literal, axis=(2, 3)) > 0
    return partner, heights, info, lives, discards, literal, clued, occupied


def agent_act(obs, reset, memory, rng):
    del rng
    f, legal = obs.features, obs.legal_actions
    partner, heights, info, lives, discards, literal, clued_now, occupied = _decode(f)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)
    colors = jnp.repeat(jnp.arange(5), 5)
    ranks = jnp.tile(jnp.arange(5), 5)
    multiplicity = jnp.asarray([3, 2, 2, 2, 1], dtype=jnp.int32)

    intent = jnp.where(reset, jnp.zeros_like(memory['intent']), memory['intent'])
    status = jnp.where(reset, jnp.zeros_like(memory['status']), memory['status'])
    prior_clued = jnp.where(reset, jnp.zeros_like(memory['clued']), memory['clued'])
    turn = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['turn'] + 1)
    last_self = f[253] > 0.5
    last_remove = jnp.any(f[255:257] > 0.5)
    removed_slot = jnp.argmax(f[276:281]).astype(jnp.int32)
    shift_self = (~reset) & last_self & last_remove
    intent = jnp.where(shift_self, _shift_after_remove(intent, removed_slot, jnp.zeros((25,), dtype=jnp.bool_)), intent)
    status = jnp.where(shift_self, _shift_after_remove(status, removed_slot, jnp.asarray(0, dtype=jnp.int32)), status)
    prior_clued = jnp.where(shift_self, _shift_after_remove(prior_clued, removed_slot, jnp.asarray(False)), prior_clued)

    received = ((~reset) & (f[254] > 0.5) & jnp.any(f[257:259] > 0.5) & (f[259] > 0.5))
    matches = f[271:276] > 0.5
    pre_established = prior_clued | (status > 0)
    new_matches = matches & (~pre_established) & occupied[0]
    unclued_before = occupied[0] & (~pre_established)
    has_chop = jnp.any(unclued_before)
    chop_before = _first_true(unclued_before)
    any_new = jnp.any(new_matches)
    focus = jnp.where(any_new & has_chop & matches[chop_before], chop_before, jnp.where(any_new, _last_true(new_matches), _last_true(matches & occupied[0])))
    focus_is_chop = has_chop & (focus == chop_before) & matches[chop_before]
    focus_literal = literal[0, focus].reshape(25) > 0
    partner_clued = clued_now[1]

    accounted_rows = []
    for c in range(5):
        row = []
        for r in range(5):
            visible_promised = jnp.any((partner[:, c, r] > 0.5) & partner_clued)
            self_literal_exact = jnp.any((literal[0, :, c, r] > 0) & (jnp.max(literal[0], axis=(1, 2)) > 0.999) & (jnp.arange(5) != focus))
            self_inferred = jnp.any(intent[:, 5 * c + r] & (status > 0) & (jnp.arange(5) != focus) & (jnp.sum(intent, axis=1) == 1))
            row.append(visible_promised | self_literal_exact | self_inferred)
        accounted_rows.append(jnp.stack(row))
    accounted = jnp.stack(accounted_rows)
    reachable = []
    for c in range(5):
        for r in range(5):
            ok = jnp.asarray(True)
            for k in range(5):
                need = (k >= heights[c]) & (k < r)
                ok = ok & ((~need) | accounted[c, k])
            reachable.append((r >= heights[c]) & ok)
    reachable = jnp.stack(reachable)
    play_poss = focus_literal & reachable
    attainable_rows = []
    for c in range(5):
        path_alive = jnp.asarray(True)
        row = []
        for r in range(5):
            path_alive = path_alive & ((r < heights[c]) | (discards[c, r] < multiplicity[r]))
            row.append((r >= heights[c]) & path_alive)
        attainable_rows.append(jnp.stack(row))
    attainable_grid = jnp.stack(attainable_rows)
    attainable_types = attainable_grid.reshape(25)
    critical = ((discards >= (multiplicity[None, :] - 1))
                & attainable_grid)
    is_rank_hint = f[258] > 0.5
    named_rank = jnp.argmax(f[266:271]).astype(jnp.int32)
    save_allowed = ((is_rank_hint & (named_rank == 4) & (ranks == 4)) | (is_rank_hint & (named_rank == 1) & (ranks == 1)) | (critical.reshape(25) & (ranks < 4)))
    save_poss = focus_literal & save_allowed & focus_is_chop
    inferred = play_poss | save_poss
    inferred = jnp.where(jnp.any(inferred), inferred, focus_literal)
    focus_oh = jnp.arange(5) == focus
    intent = jnp.where((received & focus_oh)[:, None], inferred[None, :], intent)
    status = jnp.where(received & focus_oh, jnp.where(jnp.any(save_poss), 2, 1).astype(jnp.int32), status)
    side_new = received & new_matches & (~focus_oh)
    useful_types = attainable_types
    side_masks = (literal[0].reshape(5, 25) > 0) & useful_types[None, :]
    intent = jnp.where(side_new[:, None], side_masks, intent)
    status = jnp.where(side_new, jnp.asarray(3, dtype=jnp.int32), status)

    literal_masks = literal[0].reshape(5, 25) > 0
    intent = intent & literal_masks
    current_playable_types = ranks == jnp.repeat(heights, 5)
    still_useful_types = attainable_types
    had_intent = jnp.sum(intent, axis=1) > 0
    live_intent = intent & still_useful_types[None, :]
    inferred_trash = had_intent & (~jnp.any(live_intent, axis=1))
    # Good Touch promises that every newly touched card will eventually play.
    # As stacks advance, identities whose opportunity has passed can therefore
    # be removed from both focused and non-focused notes.
    intent = live_intent
    # A play focus, ambiguous play/save note, or Good-Touch side card is safe
    # once every still-live conventional identity is playable now.
    intent_safe = ((status > 0) & (jnp.sum(intent, axis=1) > 0) & jnp.all((~intent) | current_playable_types[None, :], axis=1))
    playable_grid = jnp.arange(5)[None, :] == heights[:, None]
    literal_mass = jnp.sum(literal[0] * playable_grid[None, :, :], axis=(1, 2))
    literal_safe = (literal_mass > 0.999) & occupied[0]
    # Convention inferences may become stale when another copy plays.  Spend
    # at most the two buffer lives on such risks; on the final life require
    # literal certainty so bomb-zero is never chosen deliberately.
    inference_risk_ok = (lives >= 3) | ((lives >= 2) & (status == 1))
    safe_play = ((intent_safe & inference_risk_ok) | literal_safe) & legal[5:10]

    p_occ = occupied[1]
    p_color = jnp.argmax(jnp.sum(partner, axis=2), axis=1)
    p_rank = jnp.argmax(jnp.sum(partner, axis=1), axis=1)
    p_unclued = p_occ & (~clued_now[1])
    p_has_chop = jnp.any(p_unclued)
    p_chop = _first_true(p_unclued)
    p_is_playable = p_rank == heights[p_color]
    p_is_useful = attainable_grid[p_color, p_rank]
    p_critical = critical[p_color, p_rank]
    same_identity_count = jnp.sum((p_color[:, None] == p_color[None, :]) & (p_rank[:, None] == p_rank[None, :]) & p_occ[:, None], axis=1)
    p_unique_visible = same_identity_count == 1
    preservation_needed = ((p_rank == 4) | ((p_rank == 1) & p_unique_visible)
                           | p_critical)
    playable_unique = p_is_playable & p_unique_visible
    p_save_needed = p_is_useful & (((deck > 0) & preservation_needed) | playable_unique)
    urgent_chop = p_has_chop & p_save_needed[p_chop]

    # Publicly exact, already-clued cards can support a Level-1 delayed clue or
    # self-prompt.  Requiring literal exactness keeps the connection legible to
    # an independently implemented partner.
    known_connectors = []
    for c in range(5):
        row = []
        for r in range(5):
            known_self = jnp.any(clued_now[0] & (literal[0, :, c, r] > 0.999))
            known_target = jnp.any(clued_now[1] & (literal[1, :, c, r] > 0.999))
            row.append(known_self | known_target)
        known_connectors.append(jnp.stack(row))
    known_connectors = jnp.stack(known_connectors)
    p_delayed_rows = []
    for s in range(5):
        c, r = p_color[s], p_rank[s]
        path_ok = jnp.asarray(True)
        for cc in range(5):
            for k in range(5):
                need = (cc == c) & (k >= heights[cc]) & (k < r)
                path_ok = path_ok & ((~need) | known_connectors[cc, k])
        p_delayed_rows.append(p_occ[s] & (r > heights[c]) & path_ok)
    p_delayed = jnp.stack(p_delayed_rows)

    clue_scores, clue_focuses, clue_is_play, clue_is_save, forced_scores = [], [], [], [], []
    for a in range(10):
        by_color = a < 5
        value = a if by_color else a - 5
        touch = p_occ & ((p_color == value) if by_color else (p_rank == value))
        new = touch & (~clued_now[1])
        any_new_c = jnp.any(new)
        foc = jnp.where(any_new_c & p_has_chop & touch[p_chop], p_chop, jnp.where(any_new_c, _last_true(new), _last_true(touch)))
        foc_chop = p_has_chop & (foc == p_chop) & touch[p_chop]
        direct_play = p_is_playable[foc]
        delayed_play = p_delayed[foc]
        connected_play = direct_play | delayed_play
        five_save = (~by_color) & (value == 4) & (p_rank[foc] == 4)
        two_save = (~by_color) & (value == 1) & (p_rank[foc] == 1) & p_unique_visible[foc]
        critical_save = p_critical[foc] & (p_rank[foc] < 4)
        save = (foc_chop & p_is_useful[foc] & (five_save | two_save | critical_save)
                & ((deck > 0) | direct_play))
        established = new | clued_now[1]
        same_established = jnp.sum(established[None, :] & (p_color[:, None] == p_color[None, :]) & (p_rank[:, None] == p_rank[None, :]), axis=1)
        dup_new = new & (same_established > 1)
        good_touch = jnp.all((~new) | (p_is_useful & (~dup_new)))
        play_valid = any_new_c & connected_play & good_touch
        save_valid = any_new_c & save
        # Save Principle permits an otherwise bad touch to protect the focused
        # unique playable chop.  A no-new-card play clue is a Tempo Clue and is
        # used only in the mechanically forced (eight-token) situation.
        urgent_direct = any_new_c & direct_play & foc_chop & urgent_chop
        tempo_valid = (~any_new_c) & direct_play
        new_useful = jnp.sum(new & p_is_useful).astype(jnp.float32)
        total_touch = jnp.sum(touch & p_is_useful).astype(jnp.float32)
        color_bonus = jnp.asarray(0.35 if by_color else 0.0)
        # Reward clues that also fill established cards into literal certainty.
        type_attr = ((jnp.arange(5)[:, None, None] == value) if by_color
                     else (jnp.arange(5)[None, None, :] == value))
        type_attr = jnp.broadcast_to(type_attr, (5, 5, 5))
        hypothetical = (literal[1] > 0) & jnp.where(touch[:, None, None], type_attr, ~type_attr)
        hypo_count = jnp.sum(hypothetical, axis=(1, 2))
        hypo_safe = (hypo_count > 0) & jnp.all((~hypothetical) | playable_grid[None, :, :], axis=(1, 2))
        pre_safe = jnp.sum(literal[1] * playable_grid[None, :, :], axis=(1, 2)) > 0.999
        fill_count = jnp.sum(clued_now[1] & hypo_safe & (~pre_safe)).astype(jnp.float32)
        # MCVP rejects a no-new-card retouch unless we are forced to clue or
        # the retouch actually resolves an established card into a safe play.
        useful_tempo = tempo_valid
        play_base = jnp.where(direct_play, 100.0, 84.0)
        score = jnp.where(play_valid, play_base + 8.0 * new_useful + total_touch + 5.0 * fill_count + color_bonus,
                          jnp.where(urgent_direct, 96.0 + new_useful + color_bonus,
                          jnp.where(useful_tempo, 52.0 + 5.0 * fill_count + color_bonus,
                          jnp.where(save_valid, 70.0 + 2.0 * new_useful + color_bonus, -1e6))))
        score = jnp.where(legal[10 + a], score, -1e6)
        # If no conventional clue exists at eight tokens, a high-rank clue for
        # which the recipient has no literally possible currently-playable
        # identity is the least misleading forced clue: it cannot trigger a
        # play under our conservative Level-1 interpretation.
        focus_knowledge = literal[1, foc]
        # Simulate the literal narrowing caused by this hypothetical clue.
        possible_play_mass = (jnp.sum(focus_knowledge[value, :] * playable_grid[value, :])
                              if by_color else
                              jnp.sum(focus_knowledge[:, value] * playable_grid[:, value]))
        forced_safe = ((~by_color) & any_new_c & (possible_play_mass < 0.001)
                       & legal[10 + a])
        forced_score = jnp.where(forced_safe, 10.0 * value + new_useful, -1e6)
        clue_scores.append(score); clue_focuses.append(foc); clue_is_play.append(play_valid | urgent_direct | useful_tempo); clue_is_save.append(save_valid); forced_scores.append(forced_score)
    clue_scores = jnp.stack(clue_scores)
    clue_focuses = jnp.stack(clue_focuses)
    clue_is_play = jnp.stack(clue_is_play)
    clue_is_save = jnp.stack(clue_is_save)
    forced_scores = jnp.stack(forced_scores)
    best_clue_i = jnp.argmax(clue_scores).astype(jnp.int32)
    best_clue = (10 + best_clue_i).astype(jnp.int32)
    has_clue = clue_scores[best_clue_i] > 0
    forced_i = jnp.argmax(forced_scores).astype(jnp.int32)
    forced_clue = (10 + forced_i).astype(jnp.int32)
    has_forced_clue = forced_scores[forced_i] > 0
    urgent_scores = jnp.where(clue_is_save | (clue_is_play & (clue_focuses == p_chop)), clue_scores, -1e6)
    urgent_i = jnp.argmax(urgent_scores).astype(jnp.int32)
    urgent_clue = (10 + urgent_i).astype(jnp.int32)
    has_urgent_clue = urgent_scores[urgent_i] > 0

    obsolete_types = ~attainable_types
    known_trash = ((jnp.sum(literal_masks, axis=1) > 0) & jnp.all((~literal_masks) | obsolete_types[None, :], axis=1)) | inferred_trash
    trash_discard = _first_true(known_trash & legal[0:5])
    has_trash = jnp.any(known_trash & legal[0:5])
    own_unclued = occupied[0] & (~clued_now[0])
    own_chop = _first_true(own_unclued & legal[0:5])
    has_own_chop = jnp.any(own_unclued & legal[0:5])
    any_discard = jnp.any(legal[0:5])
    fallback_discard = jnp.argmax(legal[0:5]).astype(jnp.int32)
    discard_action = jnp.where(has_trash, trash_discard, jnp.where(has_own_chop, own_chop, fallback_discard))
    play_action = (5 + _last_true(safe_play)).astype(jnp.int32)
    has_play = jnp.any(safe_play)
    active_action = jnp.argmax(legal).astype(jnp.int32)
    active_action = jnp.where(any_discard, discard_action, active_action)
    # At eight tokens discarding is mechanically unavailable.  Any legal clue
    # is safer than an unhinted play if the scored/forced sets are empty.
    fallback_clue_i = (9 - jnp.argmax(legal[10:20][::-1].astype(jnp.int32))).astype(jnp.int32)
    fallback_clue = (10 + fallback_clue_i).astype(jnp.int32)
    active_action = jnp.where((info >= 8) & jnp.any(legal[10:20]), fallback_clue, active_action)
    active_action = jnp.where((info >= 8) & has_forced_clue, forced_clue, active_action)
    active_action = jnp.where(has_clue, best_clue, active_action)
    active_action = jnp.where(has_play, play_action, active_action)
    active_action = jnp.where(urgent_chop & has_urgent_clue, urgent_clue, active_action)
    action = jnp.where(legal[20], 20, active_action).astype(jnp.int32)

    next_memory = {'intent': intent, 'status': status, 'clued': clued_now[0], 'turn': turn}
    last_failed = last_remove & (f[255] > 0.5) & (~(f[306] > 0.5))
    trace = {
        'chosen': action, 'safe_play_mask': safe_play,
        'intent_safe_mask': intent_safe, 'literal_safe_mask': literal_safe,
        'own_status': status, 'urgent_chop': urgent_chop,
        'best_clue_score': clue_scores[best_clue_i], 'info': info,
        'lives': lives, 'turn': turn, 'heights': heights,
        'last_failed_play': last_failed, 'last_actor_self': last_self,
        'last_revealed': jnp.argmax(f[281:306]).astype(jnp.int32),
    }
    return action, next_memory, trace
