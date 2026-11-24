"""Literal-information Hanabi policy for independently generated partners."""
import jax
import jax.numpy as jnp
from custom_primitives import exact_belief

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'exact_literal_cooperator', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def normalize(w):
    return w / jnp.maximum(w.sum(axis=-1, keepdims=True), 1e-12)


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions
    ranks = jnp.tile(jnp.arange(5), 5)
    colors = jnp.repeat(jnp.arange(5), 5)
    hand = f[:125].reshape(5, 25)
    heights = (f[167:192].reshape(5, 5) * jnp.arange(1, 6)).sum(-1)
    tokens = f[192:200].sum()
    lives = f[200:203].sum()
    deck = f[127:167].sum()
    d = f[203:253].reshape(5, 10)
    discards = jnp.stack([d[:, :3].sum(-1), d[:, 3:5].sum(-1), d[:, 5:7].sum(-1), d[:, 7:9].sum(-1), d[:, 9]], -1)
    counts = (jnp.array([3, 2, 2, 2, 1])[None, :] - discards - (jnp.arange(5)[None, :] < heights[:, None])).reshape(25)
    reachable = jnp.cumprod(((counts.reshape(5, 5) > 0) | (jnp.arange(5)[None, :] < heights[:, None])).astype(jnp.float32), axis=-1).reshape(25)
    playable = (ranks == heights[colors]).astype(jnp.float32)
    dead = ((ranks < heights[colors]) | (reachable == 0)).astype(jnp.float32)
    critical = (counts <= 1) * (1-dead)
    knowledge = f[308:].reshape(2, 5, 35)
    masks = knowledge[:, :, :25] > 0
    own_counts = jnp.maximum(counts - hand.sum(0), 0)
    own = exact_belief(masks[0], own_counts)
    occupied = own.sum(-1) > 0
    pplay = own @ playable
    pdead = own @ dead
    pcritical = own @ critical
    positive = knowledge[0, :, 25:].sum(-1) > 0
    distance = jnp.maximum(ranks - heights[colors], 0)
    value = (1-dead) * (1.0 + 1.6 * critical + 0.6 * playable) / (1 + 0.18*distance)
    discard_score = 2.8*pdead - own @ value - 0.22*positive - 0.015*jnp.arange(5)
    discard_score = jnp.where(legal[:5], discard_score, -1e6)
    discard_slot = jnp.argmax(discard_score)
    # Simulate the literal information conveyed by each possible hint.
    attributes = jnp.concatenate([colors[None, :] == jnp.arange(5)[:, None], ranks[None, :] == jnp.arange(5)[:, None]], axis=0)
    matched = jnp.einsum('st,ht->hs', hand, attributes.astype(jnp.float32)) > 0
    after_mask = masks[1][None, :, :] & (attributes[:, None, :] == matched[:, :, None])
    partner_counts = jnp.maximum(counts - own.sum(0), 0.05 * counts)
    before = normalize(masks[1] * partner_counts)
    after = normalize(after_mask * partner_counts)
    bp = before @ playable
    ap = after @ playable
    bd = before @ dead
    ad = after @ dead
    actual_play = hand @ playable
    actual_dead = hand @ dead
    actual_critical = hand @ critical
    # Higher powers reward knowledge that is close to a usable play.
    play_gain = ((ap**6 - bp[None, :]**6) * actual_play[None, :]).sum(-1)
    ready_gain = (((ap > 0.999) & (bp[None, :] < 0.999)) * actual_play[None, :]).sum(-1)
    discard_gain = ((ad**4-bd[None, :]**4) * actual_dead[None, :]).sum(-1)
    entropy_before = -(before*jnp.log(jnp.maximum(before, 1e-12))).sum(-1)
    entropy_after = -(after*jnp.log(jnp.maximum(after, 1e-12))).sum(-1)
    info_gain = jnp.maximum(entropy_before[None, :]-entropy_after, 0)
    gain = (info_gain * (0.13 + 0.24*actual_play + 0.12*actual_critical)[None, :]).sum(-1)
    unsafe = ((ap**6 - bp[None, :]**6) * (1-actual_play)[None, :]).sum(-1)
    hint_score = 1.7*play_gain + 0.8*ready_gain + 0.3*discard_gain + gain - 2.0*jnp.maximum(unsafe, 0)
    hint_score = jnp.where(legal[10:20], hint_score, -1e6)
    hint = 10 + jnp.argmax(hint_score)
    best_hint = hint_score.max()
    best_play = jnp.argmax(pplay + 0.0001*(own @ (ranks==4)))
    threshold = jnp.where(lives >= 3, 0.92, jnp.where(lives >= 2, 0.97, 0.99999))
    threshold = jnp.where(deck == 0, jnp.where(lives > 1, 0.0, jnp.maximum(0.9, heights.sum()/(heights.sum()+1))), threshold)
    hint_min = jnp.where(tokens >= 6, 0.15, jnp.where(tokens >= 3, 0.42, 0.75))
    action = jnp.where(best_hint > hint_min, hint, discard_slot)
    action = jnp.where(pplay[best_play] >= threshold, best_play+5, action)
    # At full tokens a hint is always available; use legal fallback if necessary.
    action = jnp.where(legal[action], action, hint)
    action = jnp.where(legal[20], 20, action).astype(jnp.int32)
    trace = {'pplay':pplay, 'pdead':pdead, 'discard_score':discard_score, 'hint_score':hint_score,
             'hand':jnp.argmax(hand, -1), 'heights':heights, 'tokens':tokens, 'lives':lives, 'deck':deck,
             'last':f[253:308], 'reset':reset}
    return action, jnp.where(reset, jnp.int32(0), memory), trace
