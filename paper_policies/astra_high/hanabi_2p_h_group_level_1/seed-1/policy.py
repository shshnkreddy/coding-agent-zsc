"""Two-player H-group Level 1 policy, using local public card notes."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'h_group_level_1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        'notes': jnp.ones((2, 5, 25), dtype=jnp.bool_),
        'clued': jnp.zeros((2, 5), dtype=jnp.bool_),
        'dupes': jnp.zeros((2, 5, 25), dtype=jnp.bool_),
        'occupied': jnp.ones((2, 5), dtype=jnp.bool_),
        'early': jnp.asarray(True),
        'turn': jnp.asarray(0, jnp.int32),
    }


def decode(obs):
    x = obs.features
    hand = x[:125].reshape(5, 25)
    stacks = jnp.sum(x[167:192].reshape(5, 5) * jnp.arange(1, 6), axis=1).astype(jnp.int32)
    d = x[203:253].reshape(5, 10)
    disc = jnp.stack((d[:, :3].sum(1), d[:, 3:5].sum(1), d[:, 5:7].sum(1), d[:, 7:9].sum(1), d[:, 9]), axis=1)
    ranks = jnp.tile(jnp.arange(5), 5)
    colors = jnp.repeat(jnp.arange(5), 5)
    mult = jnp.tile(jnp.array([3, 2, 2, 2, 1]), 5)
    played = ranks < stacks[colors]
    remaining = mult - disc.reshape(25) - played
    blocked = jnp.cumsum((disc >= jnp.array([3, 2, 2, 2, 1])).astype(jnp.int32), axis=1) > 0
    useful = (~played) & (~blocked.reshape(25))
    playable = ranks == stacks[colors]
    k = x[308:].reshape(2, 5, 35)
    literal = k[:, :, :25] > 0
    clued = jnp.any(k[:, :, 25:] > 0, axis=-1)
    occupied = jnp.any(literal, axis=-1)
    return hand, stacks, disc.reshape(25), remaining, useful, playable, literal, clued, occupied


def constrain(notes, literal):
    narrow = notes & literal
    return jnp.where(jnp.any(narrow, axis=-1, keepdims=True), narrow, literal)


def chop_index(clued, occupied):
    return jnp.min(jnp.where(occupied & ~clued, jnp.arange(5), 5))


def focus_index(matches, clued, occupied):
    chop = chop_index(clued, occupied)
    new = matches & ~clued & occupied
    left_new = jnp.max(jnp.where(new, jnp.arange(5), -1))
    left_old = jnp.max(jnp.where(matches & occupied, jnp.arange(5), -1))
    focus = jnp.where(jnp.any(new), jnp.where(jnp.any(matches & (jnp.arange(5) == chop)), chop, left_new), left_old)
    return jnp.maximum(focus, 0), (focus == chop) & jnp.any(new), new


def clean_notes(notes, literal, clued, hand, useful, dupes):
    notes = constrain(notes, literal)
    # Good Touch eliminates useful duplicates already established in the other
    # hand; singleton notes stay attached even when the card becomes trash.
    own_other = jnp.any((hand > 0) & clued[1, :, None], axis=0)
    certain = (jnp.sum(notes[0], axis=-1) == 1) & clued[0]
    other_own = jnp.any(notes[0] & certain[:, None], axis=0)
    unavailable = jnp.stack((own_other, other_own))
    candidate = notes & (useful[None, None, :] | dupes) & (~unavailable[:, None, :] | dupes)
    count = jnp.sum(notes, axis=-1, keepdims=True)
    apply = clued[:, :, None] & (count > 1) & jnp.any(candidate, axis=-1, keepdims=True)
    notes = jnp.where(apply, candidate, notes)
    # Two distinct clued cards should not denote the same identity. Iterate
    # singleton elimination to make multi-card color clues useful over time.
    def eliminate(_, n):
        certain = (jnp.sum(n, axis=-1) == 1) & clued
        singles = n & certain[:, :, None]
        taken = jnp.any(singles[:, None, :, :] & (~jnp.eye(5, dtype=jnp.bool_))[None, :, :, None], axis=2)
        nxt = n & ~taken
        use = clued[:, :, None] & (jnp.sum(n, axis=-1, keepdims=True) > 1) & jnp.any(nxt, axis=-1, keepdims=True)
        return jnp.where(use, nxt, n)
    return jax.lax.fori_loop(0, 3, eliminate, notes)


def future_playable(stacks, accounted):
    # An identity is reachable iff all preceding unplayed ranks are accounted
    # for by previously clued cards. This retains delayed-play alternatives.
    counts = accounted.reshape(5, 5)
    ready = (jnp.arange(5)[None, :] < stacks[:, None]) | counts
    prefix = jnp.concatenate((jnp.ones((5, 1), dtype=jnp.bool_), jnp.cumprod(ready[:, :4], axis=1).astype(jnp.bool_)), axis=1)
    return prefix.reshape(25) & (jnp.tile(jnp.arange(5), 5) >= jnp.repeat(stacks, 5))


def interpret(notes, literal, preclued, occupied, target, matches, rank_hint, value, stacks, disc, useful, accounted):
    f, on_chop, new = focus_index(matches, preclued[target], occupied[target])
    n = constrain(notes[target], literal[target])
    touched_good = n & useful[None, :]
    n = jnp.where(new[:, None] & jnp.any(touched_good, axis=-1, keepdims=True), touched_good, n)
    ranks = jnp.tile(jnp.arange(5), 5)
    mult = jnp.tile(jnp.array([3, 2, 2, 2, 1]), 5)
    save = on_chop & (((disc == mult - 1) & (ranks < 4)) | (rank_hint & ((value == 1) | (value == 4))))
    possible = (future_playable(stacks, accounted) | save) & useful
    nf = n[f] & possible
    n = n.at[f].set(jnp.where(jnp.any(nf), nf, n[f]))
    return notes.at[target].set(n), f, on_chop


def agent_act(obs, reset, memory, rng):
    memory = jax.tree.map(lambda old, initial: jnp.where(reset, initial, old), memory, init_agent())
    hand, stacks, disc, remaining, useful, playable, literal, clued, occupied = decode(obs)
    la = obs.features[253:308]
    actor = jnp.argmax(la[:2])
    remove = jnp.any(la[2:4] > 0) & ~reset
    removed_slot = jnp.argmax(la[23:28])
    indices = jnp.arange(5)
    old_chop = chop_index(memory['clued'][actor], memory['occupied'][actor])
    early = memory['early'] & ~(remove & (la[3] > 0) & (removed_slot == old_chop))
    shifted = memory['notes'][actor][jnp.minimum(indices + (indices >= removed_slot), 4)]
    shifted = jnp.where((indices == 4)[:, None], jnp.ones((5, 25), dtype=jnp.bool_), shifted)
    notes = memory['notes'].at[actor].set(jnp.where(remove, shifted, memory['notes'][actor]))
    notes = constrain(notes, literal)
    shifted_dupes = memory['dupes'][actor][jnp.minimum(indices + (indices >= removed_slot), 4)]
    shifted_dupes = jnp.where((indices == 4)[:, None], False, shifted_dupes)
    dupes = memory['dupes'].at[actor].set(jnp.where(remove, shifted_dupes, memory['dupes'][actor]))
    hinted = jnp.any(la[4:6] > 0) & ~reset
    target = jnp.argmax(la[6:8])
    matches = la[18:23] > 0
    rank_hint = la[5] > 0
    value = jnp.argmax(jnp.where(rank_hint, la[13:18], la[8:13]))
    # When receiving, the giver's clued hand is visible. When giving, only
    # definitely identified cards in our own private notes can be connectors.
    actual_accounted = jnp.any((hand > 0) & clued[1, :, None], axis=0)
    known_accounted = jnp.any(notes[0] & clued[0, :, None] & (jnp.sum(notes[0], axis=-1) == 1)[:, None], axis=0)
    accounted = jnp.where(target == 0, actual_accounted, known_accounted)
    updated, focus, save_focus = interpret(notes, literal, memory['clued'], memory['occupied'], target, matches, rank_hint, value, stacks, disc, useful, accounted)
    notes = jnp.where(hinted, updated, notes)
    # Ambiguous saved cards in the giver's own hidden hand can cause an
    # unintentional duplicate touch. Retain that identity after its visible
    # counterpart leaves, rather than converting the old copy into a new play.
    other_actual = jnp.any((hand > 0) & clued[1, :, None], axis=0)
    own_possible = jnp.any(notes[0] & clued[0, :, None], axis=0)
    cross_overlap = jnp.stack((other_actual, own_possible))
    dupes = dupes | (notes & clued[:, :, None] & cross_overlap[:, None, :])
    notes = clean_notes(notes, literal, clued, hand, useful, dupes)

    # Local probabilities subtract visible cards, unlike native V0 features.
    hidden_counts = jnp.maximum(remaining - jnp.sum(hand, axis=0), 0)
    weights = notes[0] * hidden_counts[None, :]
    weights = jnp.where(jnp.sum(weights, axis=-1, keepdims=True) > 0, weights, literal[0] * hidden_counts[None, :])
    probs = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1)
    pplay = jnp.sum(probs * playable, axis=-1)
    ptrash = jnp.sum(probs * ~useful, axis=-1)
    own_canplay = occupied[0] & (pplay > .999)
    partner_probs = notes[1].astype(jnp.float32) / jnp.maximum(jnp.sum(notes[1], axis=-1, keepdims=True), 1)
    partner_ready = jnp.sum(partner_probs * playable, axis=-1) > .999
    partner_trash = jnp.sum(partner_probs * ~useful, axis=-1) > .999
    partner_has_move = jnp.any(partner_ready & occupied[1]) | jnp.any(partner_trash & occupied[1])
    own_chop = chop_index(clued[0], occupied[0])
    partner_chop = chop_index(clued[1], occupied[1])
    identities = jnp.argmax(hand, axis=-1)
    actual_useful = useful[identities] & occupied[1]
    actual_play = playable[identities] & occupied[1]
    multiplicities = jnp.tile(jnp.array([3, 2, 2, 2, 1]), 5)
    critical = disc == multiplicities - 1
    unique_in_hand = jnp.sum(hand, axis=0)[identities] == 1
    own_known = jnp.any(notes[0] & clued[0, :, None] & (jnp.sum(notes[0], axis=-1) == 1)[:, None], axis=0)
    own_possible = jnp.any(notes[0] & clued[0, :, None], axis=0)
    must_save = actual_useful & (critical[identities] | ((identities % 5 == 1) & unique_in_hand) | (actual_play & unique_in_hand)) & ~own_known[identities]
    danger = (partner_chop < 5) & must_save[jnp.minimum(partner_chop, 4)] & ~partner_has_move
    info = jnp.sum(obs.features[192:200])
    deck = jnp.sum(obs.features[127:167])

    def clue_value(i):
        rank = i >= 5
        v = i % 5
        match = occupied[1] & jnp.where(rank, identities % 5 == v, identities // 5 == v)
        f, on_chop, new = focus_index(match, clued[1], occupied[1])
        ident = identities[f]
        savable = on_chop & actual_useful[f] & (((ident % 5 < 4) & critical[ident]) | (rank & ((v == 4) | ((v == 1) & unique_in_hand[f]))))
        reach = future_playable(stacks, known_accounted)
        play_clue = reach[ident] & actual_useful[f]
        duplicate = (jnp.sum((hand > 0) & (clued[1] | match)[:, None], axis=0) > 1)[identities]
        bad = new & (~actual_useful | duplicate | own_possible[identities])
        bad_count = jnp.sum(bad)
        new_count = jnp.sum(new)
        saves_danger = on_chop & must_save[f]
        good = (savable | play_clue) & ((bad_count == 0) | saves_danger) & (new_count > 0)
        # At full tokens a useful retouch is a documented forced tempo clue.
        tempo = (info == 8) & (new_count == 0) & play_clue & actual_play[f]
        # Estimate whether the focus is immediately unambiguous in a receiver
        # who tracks all ordinary save and delayed-play alternatives.
        attr = jnp.where(rank, jnp.tile(jnp.arange(5), 5) == v, jnp.repeat(jnp.arange(5), 5) == v)
        save_mask = on_chop & (((disc == multiplicities - 1) & (jnp.tile(jnp.arange(5), 5) < 4)) | (rank & ((v == 1) | (v == 4))))
        poss = notes[1, f] & attr & useful & (reach | save_mask)
        immediate = jnp.any(poss) & jnp.all(~poss | playable)
        # Newly touched extra cards are valuable but do not inherit focus.
        score = 3.0 + 1.15 * new_count + 1.2 * immediate + .3 * (~rank) + .3 * play_clue
        score += jnp.where(danger & (saves_danger | immediate), 8.0, 0.0)
        score -= 4.0 * bad_count
        score += .25 * jnp.sum(new & critical[identities])
        score = jnp.where(tempo, 2.0 + 2.0 * immediate, score)
        valid = (good | tempo) & obs.legal_actions[10 + i]
        return jnp.where(valid, score, -1000.0)

    clue_scores = jax.vmap(clue_value)(jnp.arange(10))
    best_clue = jnp.argmax(clue_scores)
    best_clue_value = jnp.max(clue_scores)
    # Preserve conventional discard order. Known trash precedes the chop.
    discard_score = jnp.where(ptrash > .999, 3.0 - .02 * indices, jnp.where(indices == own_chop, 1.0, -10.0))
    # An all-clued hand has no ordinary chop: discard its least valuable card.
    loss = jnp.sum(probs * useful * (5 - jnp.tile(jnp.arange(5), 5)), axis=-1)
    discard_score = jnp.where((own_chop == 5) & (ptrash <= .999), -loss - .02 * indices, discard_score)
    discard_score = jnp.where(obs.legal_actions[:5], discard_score, -1000.0)
    discard_action = jnp.argmax(discard_score)
    play_score = jnp.where(own_canplay, 8.0 + .05 * indices + .3 * jnp.sum(probs * (jnp.tile(jnp.arange(5), 5) == 4), axis=-1), -1000.0)
    play_action = 5 + jnp.argmax(play_score)
    has_play = jnp.any(own_canplay)
    # Save/play before chop discards, and exhaust valid early-game clues.
    do_clue = (best_clue_value > -500) & (~has_play | (danger & (best_clue_value > 8)))
    action = jnp.where(has_play, play_action, discard_action)
    action = jnp.where(do_clue, 10 + best_clue, action)
    # Last-round shots spend spare lives only when their probability is useful.
    lives = jnp.sum(obs.features[200:203])
    gamble = (deck == 0) & (lives > 1) & (jnp.max(pplay) > .35)
    action = jnp.where(gamble & ~has_play, 5 + jnp.argmax(pplay), action)
    # Full-token forced clues: minimize literal damage if no convention clue
    # exists. No extra signal is attached to such a legal fallback.
    def forced_clue_score(i):
        rank = i >= 5
        v = i % 5
        match = occupied[1] & jnp.where(rank, identities % 5 == v, identities // 5 == v)
        f, on_chop, new = focus_index(match, clued[1], occupied[1])
        # Prefer retouches whose literal facts cannot create a false play.
        attr = jnp.where(rank, jnp.tile(jnp.arange(5), 5) == v, jnp.repeat(jnp.arange(5), 5) == v)
        possible_play = jnp.any(notes[1, f] & playable & attr)
        safe_focus = actual_play[f] | ~possible_play
        value = 5.0 * safe_focus + 2.0 * actual_play[f] - 3.0 * jnp.sum(new)
        value += .1 * jnp.sum(match & clued[1])
        return jnp.where(obs.legal_actions[10 + i], value, -1000.)
    forced_scores = jax.vmap(forced_clue_score)(jnp.arange(10))
    fallback = jnp.where(jnp.any(obs.legal_actions[10:20]), 10 + jnp.argmax(forced_scores), jnp.argmax(obs.legal_actions))
    action = jnp.where(obs.legal_actions[action], action, fallback)
    action = jnp.where(obs.legal_actions[20], 20, action).astype(jnp.int32)
    next_memory = {'notes': notes, 'clued': clued, 'occupied': occupied, 'early': early, 'dupes': dupes, 'turn': memory['turn'] + 1}
    trace = {'pplay': pplay, 'ptrash': ptrash, 'notes': notes, 'clue_scores': clue_scores, 'focus': jnp.where(hinted, focus, -1), 'save_focus': hinted & save_focus, 'danger': danger, 'public': obs.features, 'turn': memory['turn']}
    return action, next_memory, trace
