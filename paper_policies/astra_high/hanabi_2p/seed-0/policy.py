"""Convention-independent Hanabi policy using public literal information."""
import jax
import jax.numpy as jnp
from custom_primitives import exact_beliefs

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_cooperator', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {'previous_deck': jnp.int32(40), 'final_turns': jnp.int32(0), 'previous_own_positive': jnp.zeros(5, dtype=jnp.bool_), 'directed_clues': jnp.zeros(5, dtype=jnp.int32), 'cue_misread': jnp.bool_(False), 'last_hint_targets': jnp.zeros(5, dtype=jnp.bool_)}


def normalize(w):
    return w / jnp.maximum(w.sum(axis=-1, keepdims=True), 1e-12)


def beliefs(mask, counts):
    # Without-replacement correction for the other four uncertain own cards.
    p = normalize(mask * counts)
    for _ in range(3):
        available = jnp.maximum(counts - p.sum(axis=-2, keepdims=True) + p, 0.0)
        q = normalize(mask * available)
        p = 0.5 * p + 0.5 * q
    return normalize(p)


def agent_act(obs, reset, memory, rng):
    memory = jax.tree_util.tree_map(lambda old, initial: jnp.where(reset, initial, old), memory, init_agent())
    f = obs.features
    legal = obs.legal_actions
    removed = (f[253] > 0) & (f[255] + f[256] > 0)
    removed_slot = jnp.argmax(f[276:281])
    cue_misread = memory['cue_misread'] | ((f[254] > 0) & (f[255] > 0) & (f[306] == 0) & memory['last_hint_targets'][removed_slot])
    last_hint_targets = jnp.where((f[253] > 0) & (f[257] + f[258] > 0), f[271:276] > 0, jnp.zeros(5, dtype=jnp.bool_))
    shifted_indices = jnp.arange(5) + (jnp.arange(5) >= removed_slot)
    shifted_clues = jnp.where(shifted_indices < 5, memory['directed_clues'][jnp.minimum(shifted_indices, 4)], 0)
    directed_clues = jnp.where(removed, shifted_clues, memory['directed_clues'])
    hand = f[:125].reshape(5, 25)
    heights = (f[167:192].reshape(5, 5) * jnp.arange(1, 6)).sum(-1)
    info = f[192:200].sum()
    lives = f[200:203].sum()
    deck = f[127:167].sum()
    final_turns = jnp.where(deck > 0, 0, jnp.where(memory['previous_deck'] > 0, 2, jnp.maximum(memory['final_turns'] - 1, 0))).astype(jnp.int32)
    d = f[203:253].reshape(5, 10)
    disc = jnp.stack([d[:, :3].sum(-1), d[:, 3:5].sum(-1), d[:, 5:7].sum(-1), d[:, 7:9].sum(-1), d[:, 9]], -1)
    ranks = jnp.tile(jnp.arange(5), 5)
    board = jnp.repeat(heights, 5)
    remaining = (jnp.array([3, 2, 2, 2, 1])[None, :] - disc - (jnp.arange(5)[None, :] < heights[:, None])).reshape(25)
    blocked = jnp.cumsum(((remaining.reshape(5, 5) <= 0) & (jnp.arange(5)[None, :] >= heights[:, None])).astype(jnp.int32), axis=-1) > 0
    dead = (ranks < board) | blocked.reshape(25)
    playable = (ranks == board) & ~dead
    critical = (remaining <= 1) & ~dead
    ceiling = jnp.repeat(jnp.sum(~blocked, axis=-1), 5)
    future_value = jnp.where(dead, 0., (ceiling - ranks) / jnp.maximum(remaining, 1.))
    know = f[308:658].reshape(2, 5, 35)
    masks = know[:, :, :25] > 0
    counts = jnp.maximum(remaining - hand.sum(0), 0)
    own = exact_beliefs(masks[0], counts)
    # Approximate the counts the partner can see by marginalizing our hidden hand.
    partner_counts = jnp.maximum(remaining - own.sum(0), 0.05)
    partner = beliefs(masks[1], partner_counts)
    pplay = own @ playable
    pdead = own @ dead
    loss = own @ future_value
    own_critical = own @ critical
    failed_critical_loss = own @ (critical * (~playable) * (ceiling - ranks))
    threshold = jnp.where(lives >= 3, .80, jnp.where(lives >= 2, .97, .99999))
    threshold = jnp.where((lives >= 3) & (know[0, :, 25:].sum(-1) > 0) & (failed_critical_loss < .25), .65, threshold)
    hint_matches = f[271:276] > 0
    received_hint = (f[254] > 0) & (f[257] + f[258] > 0)
    directed_event = received_hint & (hint_matches.sum() == 1) & (info < 7)
    directed_clues = jnp.minimum(directed_clues + (directed_event & hint_matches & (pplay > .05)).astype(jnp.int32), 6)
    evidence_loss = failed_critical_loss * jnp.exp(-.9 * jnp.maximum(directed_clues - 1, 0))
    strong_repeat = received_hint & hint_matches & (hint_matches.sum() == 1) & (directed_clues >= 3) & (lives > 1) & (evidence_loss < .25)
    earlier_repeat = directed_event & hint_matches & (directed_clues >= 2) & (pplay >= .25) & (evidence_loss < .35) & (lives > 1)
    strong_repeat |= earlier_repeat
    fresh_probe = hint_matches & ~memory['previous_own_positive']
    fresh_probe &= received_hint & (hint_matches.sum() == 1) & (lives >= 3) & (f[270] == 0)
    fresh_probe &= failed_critical_loss < jnp.where(jnp.arange(5) >= 3, .4, .25)
    repeat_probe = received_hint & hint_matches & memory['previous_own_positive'] & (hint_matches.sum() == 1)
    repeat_probe &= (lives >= 3) & (f[270] == 0) & (failed_critical_loss < .25)
    threshold = jnp.where(fresh_probe | repeat_probe, .25, threshold)
    threshold = jnp.where(strong_repeat, jnp.minimum(threshold, .1), threshold)
    threshold = jnp.where((deck == 0) & (lives > 1), .35, threshold)
    play_value = 5. + pplay + .03 * (own @ (ranks == 4)) - .01 * jnp.arange(5)
    play_value = jnp.where(pplay >= threshold, play_value, -100.)

    # Evaluate each possible literal hint, including negative information.
    attributes = jnp.concatenate([jax.nn.one_hot(jnp.repeat(jnp.arange(5), 5), 5).T,
                                  jax.nn.one_hot(ranks, 5).T], axis=0) > 0
    matches = (hand @ attributes.T).T > 0
    after_mask = masks[1][None, :, :] & (attributes[:, None, :] == matches[:, :, None])
    after = beliefs(after_mask, partner_counts)
    before_play = partner @ playable
    after_play = after @ playable
    actual_play = hand @ playable
    actual_dead = hand @ dead
    actual_critical = hand @ critical
    safe_gain = ((after_play >= .9999).astype(jnp.float32) - (before_play >= .9999)[None, :]) * actual_play[None, :]
    soft_gain = (after_play ** 4 - before_play[None, :] ** 4) * actual_play[None, :]
    before_entropy = -jnp.sum(partner * jnp.log(jnp.maximum(partner, 1e-12)), -1)
    after_entropy = -jnp.sum(after * jnp.log(jnp.maximum(after, 1e-12)), -1)
    knowledge_gain = (before_entropy[None, :] - after_entropy) * (1 - actual_dead)[None, :]
    dead_gain = (after @ dead - (partner @ dead)[None, :]) * actual_dead[None, :]
    hint_value = 1.7 * safe_gain.sum(-1) + .9 * soft_gain.sum(-1) + (knowledge_gain * (.28 * actual_play + .025 * (1 - actual_play))).sum(-1) + .15 * dead_gain.sum(-1)
    hint_value += .08 * (knowledge_gain * actual_critical).sum(-1)
    hint_value -= .18 * jnp.maximum(2 - info, 0)
    # Prioritize a guaranteed play over information gathering.
    hint_value = jnp.minimum(hint_value, 4.8)
    # A mixture of common discard habits, not a required partner protocol.
    def discard_risk(p, positive):
        pd = p @ dead
        value = .25 + pd - .25 * (p @ future_value) - .35 * (p @ critical) - .012 * jnp.arange(5)
        value -= .8 * positive * (1 - pd)
        literal_risk = jax.nn.softmax(4 * value, axis=-1)
        oldest_value = 4 * (pd >= .99) + 1.5 * (~positive) - .5 * jnp.arange(5)
        oldest_risk = jax.nn.softmax(3 * oldest_value, axis=-1)
        return .5 * literal_risk + .5 * oldest_risk
    positive = know[1, :, 25:].sum(-1) > 0
    before_risk = discard_risk(partner, positive)
    after_risk = discard_risk(after, positive[None, :] | matches)
    at_stake = hand @ (critical * (ceiling - ranks) * jnp.exp(-.6 * jnp.maximum(ranks - board, 0)))
    save_gain = jnp.sum((before_risk[None, :] - after_risk) * at_stake[None, :], axis=-1)
    save_gain *= jnp.where(jnp.max(before_play) >= .9, .05, 1.)
    hint_value += 2.0 * save_gain
    fresh = matches & ~positive[None, :]
    misleading = fresh * (1 - actual_play[None, :]) * (after_play > .05)
    misleading = misleading * (jnp.arange(10) != 9)[:, None]
    hint_value -= .7 * misleading.sum(-1)
    coherent = (fresh.sum(-1) > 0) & (jnp.sum(fresh * (1 - actual_play[None, :]), axis=-1) < .1)
    newly_enabled = jnp.max(fresh * (1 - before_play[None, :]), axis=-1)
    clue_bonus = jnp.where(lives >= 3, 1.2, .6) * jnp.where(jnp.max(before_play) >= .8, .25, 1.)
    hint_value += clue_bonus * coherent * newly_enabled
    ambiguous_bad_match = jnp.any(matches & (actual_play[None, :] < .5) & (after_play > .05), axis=-1)
    protect_receiver = (lives <= 1) | cue_misread
    hint_value -= protect_receiver * (12. + heights.sum()) * ambiguous_bad_match
    discard_value = .25 + 1.0 * pdead - .25 * loss - .35 * own_critical - .012 * jnp.arange(5)
    discard_value -= .8 * (know[0, :, 25:].sum(-1) > 0) * (1 - pdead)
    retained_backup = jnp.minimum(1., (hand * jnp.where(positive, .95, .15)[:, None]).sum(0)) * (~dead)
    discard_value += 1.1 * (own @ retained_backup)
    other_own_known = jnp.any((own[None, :, :] >= .99999) & (~jnp.eye(5, dtype=jnp.bool_))[:, :, None], axis=1)
    discard_value += 1.1 * jnp.sum(own * other_own_known * (~dead)[None, :], axis=-1)
    discard_value += .35 * (jnp.max(actual_play) < .5)
    discard_value += .15 * (info <= 1)
    values = jnp.concatenate([discard_value, play_value, hint_value, jnp.array([0.])])
    # At most two actions remain after the last draw. A last-turn hint has
    # no future value, and a failed play with spare lives has no point cost.
    readiness = jnp.where(lives > 1, before_play ** 4, (before_play >= .99999).astype(jnp.float32))
    partner_now = jnp.max(actual_play * readiness)
    hint_readiness = jnp.where(lives > 1, after_play ** 4, (after_play >= .99999).astype(jnp.float32))
    final_hint = jnp.max(actual_play[None, :] * hint_readiness, axis=-1) - partner_now
    colors = jnp.repeat(jnp.arange(5), 5)
    next_board = board[None, :] + (colors[:, None] == colors[None, :]) * playable[:, None]
    next_playable = (ranks[None, :] == next_board) & ~dead[None, :]
    next_prob = partner @ next_playable.T
    next_actual = hand @ next_playable.T
    next_ready = jnp.where(lives > 1, next_prob ** 4, (next_prob >= .99999).astype(jnp.float32))
    next_gain = jnp.max(next_actual * next_ready, axis=0) - partner_now
    final_play = pplay + jnp.where(final_turns == 2, own @ next_gain, 0.)
    final_play -= jnp.where(lives <= 1, (1 - pplay) * heights.sum(), 0.)
    final_hint -= protect_receiver * 2. * ambiguous_bad_match
    final_hint = jnp.where(final_turns == 2, final_hint, 0.)
    final_values = jnp.concatenate([jnp.zeros(5), final_play, final_hint, jnp.zeros(1)])
    values = jnp.where(deck == 0, final_values, values)
    action = jnp.argmax(jnp.where(legal, values, -1e9)).astype(jnp.int32)
    trace = {'play_probability': pplay, 'dead_probability': pdead, 'discard_value': discard_value, 'hint_value': hint_value, 'info': info, 'lives': lives, 'deck': deck, 'board': heights, 'partner_ids': jnp.where(hand.sum(-1)>0, jnp.argmax(hand, -1), -1), 'last_actor': jnp.argmax(f[253:255]), 'last_kind': jnp.argmax(f[255:259]), 'last_card': jnp.argmax(f[281:306]), 'last_success': f[306], 'final_turns': final_turns, 'hint_probe': fresh_probe | repeat_probe | strong_repeat, 'directed_clues': directed_clues, 'strong_repeat': strong_repeat, 'cue_misread': cue_misread, 'failed_critical_loss': failed_critical_loss}
    return action, {'previous_deck': deck.astype(jnp.int32), 'final_turns': final_turns, 'previous_own_positive': know[0, :, 25:].sum(-1) > 0, 'directed_clues': directed_clues, 'cue_misread': cue_misread, 'last_hint_targets': last_hint_targets}, trace
