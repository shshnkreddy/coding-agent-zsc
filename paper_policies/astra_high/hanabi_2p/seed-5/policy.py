"""Hanabi policy using literal hints and publicly observable card counts."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_cooperator', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}



def init_agent():
    return {'turn': jnp.int32(0), 'empty': jnp.int32(0)}


def normalize(x):
    return x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), 1e-20)


def private_belief(mask, counts):
    # Deduct other cards whose identity is certain. This preserves literal support.
    def body(_, possibilities):
        known = (jnp.sum(possibilities, axis=-1) == 1)
        fixed = possibilities * known[:, None]
        available = counts[None, :] - jnp.sum(fixed, axis=0)[None, :] + fixed
        return possibilities & (available > 0)
    possible = jax.lax.fori_loop(0, 5, body, mask & (counts[None, :] > 0))
    known = jnp.sum(possible, axis=-1) == 1
    fixed = possible * known[:, None]
    available = jnp.maximum(counts[None, :] - jnp.sum(fixed, axis=0)[None, :] + fixed, 0)
    return normalize(possible * available)


def agent_act(obs, reset, memory, rng):
    COLORS = jnp.repeat(jnp.arange(5), 5)
    RANKS = jnp.tile(jnp.arange(5), 5)
    MULT = jnp.tile(jnp.array([3., 2., 2., 2., 1.]), 5)
    ATTR = jnp.concatenate((jax.nn.one_hot(COLORS, 5).T, jax.nn.one_hot(RANKS, 5).T), axis=0).astype(jnp.bool_)
    f = obs.features
    legal = obs.legal_actions
    hand = f[:125].reshape(5, 25)
    native = f[308:].reshape(2, 5, 35)
    masks = native[:, :, :25] > 0
    heights = jnp.sum(f[167:192].reshape(5, 5) * jnp.arange(1, 6)[None, :], axis=-1)
    d = f[203:253].reshape(5, 10)
    discards = jnp.stack((jnp.sum(d[:, :3], -1), jnp.sum(d[:, 3:5], -1), jnp.sum(d[:, 5:7], -1), jnp.sum(d[:, 7:9], -1), d[:, 9]), -1).reshape(25)
    completed = RANKS < heights[COLORS]
    counts = MULT - discards - completed
    playable = RANKS == heights[COLORS]
    unavailable = ((counts.reshape(5, 5) <= 0) & ~completed.reshape(5, 5))
    blocked = jnp.cumsum(unavailable, axis=1).reshape(25) > 0
    dead = completed | blocked
    useful = ~dead
    critical = useful & (counts == 1)
    info = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    deck = jnp.sum(f[127:167])
    turn = jnp.where(reset, 0, memory['turn'])
    empty = jnp.where(reset | (deck > 0), 0, memory['empty'] + 1)
    own = private_belief(masks[0], jnp.maximum(counts - jnp.sum(hand, axis=0), 0))
    pplay = own @ playable
    pdead = own @ dead
    pcritical = own @ critical
    occupied = jnp.sum(native[0, :, :25], -1) > 0
    rank = own @ (RANKS + 1.)
    # Sacrifice redundant cards before unique prerequisites; protect low ranks.
    loss = useful * (0.65 + 1.25 * critical + 0.5 * playable + 0.15 * (4 - RANKS))
    duplicate = own @ (jnp.sum(hand, axis=0) > 0)
    dscore = pdead * 2 - own @ loss + 0.20 * duplicate - 0.015 * jnp.arange(5)
    discard_slot = jnp.argmax(jnp.where(legal[:5], dscore, -1e9))

    # Literal consequences of each possible color or rank hint, including negatives.
    match = (hand @ ATTR.T).T > 0
    narrowed = masks[1][None, :, :] & jnp.where(match[:, :, None], ATTR[:, None, :], ~ATTR[:, None, :])
    partner_prior = normalize(masks[1] * counts)
    post = normalize(narrowed * counts[None, None, :])
    prior_play = partner_prior @ playable
    post_play = post @ playable
    actual_play = hand @ playable
    actual_dead = hand @ dead
    actual_critical = hand @ critical
    newly_safe = (post_play > 0.9999) & (prior_play[None, :] < 0.9999)
    # Count duplicate playable identities only once.
    duplicate_before = jnp.sum(jnp.tril(hand @ hand.T, -1), axis=-1)
    card_weight = 1. / (1. + duplicate_before)
    safe_gain = jnp.sum(newly_safe * actual_play[None, :] * card_weight, axis=-1)
    entropy_before = -jnp.sum(partner_prior * jnp.log(jnp.maximum(partner_prior, 1e-20)), -1)
    entropy_after = -jnp.sum(post * jnp.log(jnp.maximum(post, 1e-20)), -1)
    information = jnp.maximum(entropy_before[None, :] - entropy_after, 0)
    playable_progress = jnp.sum(jnp.maximum(post_play - prior_play[None, :], 0) * actual_play[None, :] * card_weight, -1)
    new_known = (jnp.sum(narrowed, -1) == 1) & (jnp.sum(masks[1], -1)[None, :] > 1)
    saved = jnp.sum(new_known * actual_critical[None, :], -1)
    # A partial hint to a currently playable card is a useful setup for a second hint.
    setup = jnp.max(information * actual_play[None, :] * (prior_play[None, :] < 0.9999), -1)
    hint_value = 3.0 * safe_gain + .9 * playable_progress + .35 * setup + .20 * saved + .025 * jnp.sum(information * (1 - actual_dead[None, :]), -1)
    hint_value -= .09 * (8 - info)
    hint_value -= .35 * (jnp.max(prior_play) > .9999)
    hint_action = 10 + jnp.argmax(jnp.where(legal[10:20], hint_value, -1e9))
    best_hint = jnp.max(jnp.where(legal[10:20], hint_value, -1e9))
    discard_value = .25 + .40 * jnp.max(jnp.where(legal[:5], pdead, 0)) + .12 * (info <= 2)
    action = jnp.where(best_hint > discard_value, hint_action, discard_slot)
    # Avoid gambling with the last life; extra lives can be used sparingly.
    threshold = jnp.where(lives >= 3, .95, jnp.where(lives >= 2, .98, .9999))
    threshold = jnp.where((deck <= 2) & (lives > 1), .65, threshold)
    play_score = pplay + .015 * (own @ critical) - .001 * rank - .0001 * jnp.arange(5)
    play_slot = jnp.argmax(jnp.where(occupied & (pplay >= threshold), play_score, -1e9))
    can_play = jnp.any(occupied & (pplay >= threshold))
    action = jnp.where(can_play, 5 + play_slot, action)
    action = jnp.where((info >= 8) & ~can_play, hint_action, action)
    # Last turn: take a beneficial chance if lives permit and nothing certain exists.
    last_play = jnp.argmax(jnp.where(occupied, pplay, -1))
    action = jnp.where((empty >= 2) & (lives > 1) & (pplay[last_play] > 0), 5 + last_play, action)
    action = jnp.where(legal[action], action, jnp.argmax(legal))
    action = jnp.where(legal[20], 20, action).astype(jnp.int32)
    return action, {'turn': (turn + 1).astype(jnp.int32), 'empty': empty.astype(jnp.int32)}, {'pplay': pplay, 'pdead': pdead, 'hint_value': hint_value, 'discard_score': dscore, 'info': info}
