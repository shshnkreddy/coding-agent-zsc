"""Literal, convention-independent Hanabi cooperation."""
import jax
import jax.numpy as jnp
from math import factorial, prod

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_cooperation', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {'final_turn':jnp.int32(0), 'receiver_eager':jnp.bool_(False), 'sent_uncertain':jnp.zeros(5,dtype=jnp.bool_), 'partner_waits':jnp.zeros(5,dtype=jnp.int32), 'alpha':jnp.float32(2.), 'beta':jnp.float32(1.), 'seen':jnp.zeros(5,dtype=jnp.bool_), 'pending':jnp.zeros(5,dtype=jnp.bool_), 'at_hint':jnp.zeros((5,25),dtype=jnp.bool_), 'receiver_alarm':jnp.bool_(False), 'sent_matches':jnp.zeros(5,dtype=jnp.bool_)}


def normalize(x):
    return x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), 1e-8)


def entropy(p):
    return -jnp.sum(p * jnp.log(jnp.maximum(p, 1e-8)), axis=-1)


def exact_marginals(support, counts):
    # Add one distinct dummy card for each absent slot.
    empty = ~jnp.any(support, axis=-1)
    weights = jnp.concatenate([support.astype(jnp.float32), jnp.eye(5)*empty[:,None]], axis=-1)
    counts = jnp.concatenate([counts, empty.astype(jnp.float32)])
    partitions = [()]
    for slot in range(5):
        new = []
        for blocks in partitions:
            new.append(blocks+(1 << slot,))
            for idx in range(len(blocks)):
                new.append(blocks[:idx]+(blocks[idx] | (1 << slot),)+blocks[idx+1:])
        partitions = new
    indices = jnp.array([p+(0,)*(5-len(p)) for p in partitions])
    coefficients = jnp.array([(-1.)**(5-len(p)) * prod(factorial(b.bit_count()-1) for b in p) for p in partitions])
    bits = (jnp.arange(32)[:,None] & (1 << jnp.arange(5)[None,:])) != 0
    def partition(w):
        moments = jnp.sum(jnp.prod(jnp.where(bits[:,:,None],w[None,:,:],1.),axis=1)*counts[None,:],axis=-1)
        moments = moments.at[0].set(1.)
        return jnp.sum(coefficients*jnp.prod(moments[indices],axis=-1))
    deriv = jax.grad(partition)(weights)
    marginals = jnp.maximum(weights*deriv,0.)[:,:25]
    return marginals / jnp.maximum(marginals.sum(-1,keepdims=True),1e-8)


def update_hint_model(f, reset, memory, own, playable, partner_probability):
    mem = jax.tree_util.tree_map(lambda fresh, old: jnp.where(reset, fresh, old), init_agent(), memory)
    removed_slot = jnp.argmax(f[276:281])
    partner_failed = (f[254]>0) & (f[255]>0) & (f[306]==0)
    receiver_alarm = mem['receiver_alarm'] | (partner_failed & mem['sent_matches'][removed_slot])
    sent_matches = (f[253]>0) & ((f[257]+f[258])>0) & (f[271:276]>0)
    receiver_eager = mem['receiver_eager'] | ((f[254]>0)&(f[255]>0)&mem['sent_uncertain'][removed_slot])
    sent_uncertain = sent_matches & (partner_probability<.8)
    removed = (f[253] > 0) & ((f[255]+f[256]) > 0)
    scored = removed & mem['pending'][removed_slot]
    was_playable = jnp.sum(f[281:306]*mem['at_hint'][removed_slot])
    alpha = mem['alpha'] + scored*was_playable
    beta = mem['beta'] + scored*(1.-was_playable)
    indices = jnp.arange(5) + (jnp.arange(5) >= removed_slot)
    valid = indices < 5
    indices = jnp.minimum(indices,4)
    seen = jnp.where(removed, mem['seen'][indices] & valid, mem['seen'])
    pending = jnp.where(removed, mem['pending'][indices] & valid, mem['pending'])
    at_hint = jnp.where(removed, mem['at_hint'][indices] & valid[:,None], mem['at_hint'])
    incoming = f[254]*(f[257]+f[258])*f[271:276]
    event = (incoming>0) & (jnp.sum(incoming)==1) & (f[270]==0) & (~seen)
    seen = seen | event
    pending = pending | event
    at_hint = jnp.where(event[:,None],playable[None,:],at_hint)
    historical = jnp.sum(own*at_hint,axis=-1)
    positive = pending & (historical > .99999)
    negative = pending & (historical < .00001)
    alpha = alpha + jnp.sum(positive)
    beta = beta + jnp.sum(negative)
    pending = pending & (~positive) & (~negative)
    reliability = alpha / (alpha+beta)
    conditioned = jnp.sum(own*at_hint*playable[None,:],axis=-1)/jnp.maximum(historical,1e-8)
    alternative = jnp.sum(own*(~at_hint)*playable[None,:],axis=-1)/jnp.maximum(1-historical,1e-8)
    raw = own @ playable.astype(jnp.float32)
    inferred = reliability*conditioned + (1-reliability)*alternative
    inferred = jnp.where(pending & (historical>=.15),jnp.maximum(raw,inferred),raw)
    next_memory = {'receiver_eager':receiver_eager, 'sent_uncertain':sent_uncertain, 'partner_waits':mem['partner_waits'], 'final_turn':mem['final_turn'], 'alpha':alpha, 'beta':beta, 'seen':seen, 'pending':pending, 'at_hint':at_hint, 'receiver_alarm':receiver_alarm, 'sent_matches':sent_matches}
    return next_memory, inferred, reliability


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions
    hand = f[:125].reshape(5, 25)
    know = f[308:].reshape(2, 5, 35)
    support = know[:, :, :25] > 0
    height = jnp.sum(f[167:192].reshape(5, 5) * jnp.arange(1, 6), axis=-1)
    ranks = jnp.tile(jnp.arange(5), 5)
    colors = jnp.repeat(jnp.arange(5), 5)
    d = f[203:253].reshape(5, 10)
    discarded = jnp.stack([d[:, :3].sum(-1), d[:, 3:5].sum(-1), d[:, 5:7].sum(-1), d[:, 7:9].sum(-1), d[:, 9]], -1)
    public = jnp.maximum(jnp.array([3,2,2,2,1])[None,:] - discarded - (jnp.arange(5)[None,:] < height[:,None]), 0).reshape(25)
    available = jnp.maximum(public - hand.sum(0), 0)
    own = exact_marginals(support[0], available)
    other_counts = jnp.maximum(public - own.sum(0), 0.)
    other = normalize(support[1] * other_counts)
    playable = ranks == height[colors]
    waits = jnp.where(reset,jnp.zeros(5,dtype=jnp.int32),memory['partner_waits'])
    partner_removed = (f[254]>0)&((f[255]+f[256])>0)
    removed_slot = jnp.argmax(f[276:281])
    shifted = jnp.arange(5)+(jnp.arange(5)>=removed_slot)
    waits = jnp.where(partner_removed,jnp.where(shifted<5,waits[jnp.minimum(shifted,4)],0),waits)
    inferred_ready = (other @ playable.astype(jnp.float32))>.9999
    native_ready = (know[1,:,:25] @ playable.astype(jnp.float32))>.9999
    waits = jnp.where(inferred_ready&(~native_ready),jnp.minimum(waits+(f[254]>0).astype(jnp.int32),5),0)
    receiver_weights = jnp.where((waits>=2)[:,None],public[None,:],other_counts[None,:])
    other = normalize(support[1]*receiver_weights)
    exhausted = (public.reshape(5,5) == 0) & (jnp.arange(5)[None,:] >= height[:,None])
    ceiling = jnp.min(jnp.where(exhausted, jnp.arange(5)[None,:], 5), axis=-1)
    dead = (ranks < height[colors]) | (ranks >= ceiling[colors])
    critical = (~dead) & (public <= 1)
    partner_counts = hand.sum(0).reshape(5,5)
    own_counts = own.sum(0).reshape(5,5)
    next_partner = jnp.concatenate([partner_counts[:,1:],jnp.zeros((5,1))],axis=-1).reshape(25)
    next_own = jnp.concatenate([own_counts[:,1:],jnp.zeros((5,1))],axis=-1).reshape(25)
    progress_value = 1. + .12*jnp.maximum(ceiling[colors]-ranks-1,0) + .25*jnp.minimum(next_partner,1.) + .15*jnp.minimum(next_own,1.)
    progress_value += (ranks==4)*jnp.where(f[192:200].sum()<=2,.75,jnp.where(f[192:200].sum()<=5,.35,0.))
    ahead_heights = height[None,:]+jax.nn.one_hot(colors,5)*playable[:,None]
    ahead_playable = (ranks[None,:]==ahead_heights[:,colors]).astype(jnp.float32)
    ahead_probability = jnp.einsum('su,tu->ts',other,ahead_playable)
    ahead_truth = jnp.einsum('su,tu->ts',hand,ahead_playable)
    ready_now = (other @ playable.astype(jnp.float32))>=.999
    new_ready = (ahead_probability>=.999)*ahead_truth*(~ready_now)[None,:]
    distinct_ready = jnp.sum(jnp.minimum(jnp.einsum('ts,su->tu',new_ready,hand),1.),axis=-1)
    progress_value += .5*distinct_ready
    pplay = own @ playable.astype(jnp.float32)
    pdead = own @ dead.astype(jnp.float32)
    pcritical = own @ critical.astype(jnp.float32)
    tokens = f[192:200].sum()
    lives = f[200:203].sum()
    deck = f[127:167].sum()
    score = height.sum()
    next_memory, inferred_play, hint_reliability = update_hint_model(f, reset, memory, own, playable, other @ playable.astype(jnp.float32))
    next_memory = dict(next_memory,partner_waits=waits)
    use_hint_model = (lives>=3) & (hint_reliability>=.8) & (next_memory['alpha']>=4.)
    effective_play = jnp.where(use_hint_model,inferred_play,pplay)
    final_turn = jnp.where(deck == 0, next_memory['final_turn'] + jnp.int32(1), jnp.int32(0))
    next_memory = dict(next_memory, final_turn=final_turn)
    threshold = jnp.where(lives >= 3, 0.80, jnp.where(lives >= 2, 0.95, 0.9999))
    threshold = jnp.where(deck == 0, jnp.where(lives > 1, 0.35, jnp.maximum(0.8, score / (score + 1))), threshold)
    incoming = f[254] * (f[257] + f[258]) * f[271:276]
    single_prompt = (incoming > 0) & (jnp.sum(incoming) == 1) & (f[270] == 0)
    threshold = jnp.where(single_prompt & (lives >= 3) & (deck > 0), .55, threshold)
    only_play_or_junk = (own @ ((~playable)&(~dead)).astype(jnp.float32)) < 1e-6
    spare_threshold = jnp.where(deck>20,.7,jnp.where(deck>8,.6,.45))
    threshold = jnp.where((lives>=3)&only_play_or_junk&(deck>0),jnp.minimum(threshold,spare_threshold),threshold)
    play_score = jnp.where(effective_play >= threshold, 10 + pplay + .5*(own @ (playable*(progress_value-1.))) - 0.01 * jnp.arange(5), -100 + pplay)
    # Discard what is already obsolete, then cheap and distant duplicates.
    distance = jnp.maximum(ranks - height[colors], 0)
    card_cost = (~dead) * jnp.maximum(ceiling[colors]-ranks,0) * jnp.where(public <= 1, 1.0, jnp.where(public == 2, .16, .06))
    own_certain = own>.99999
    own_backup = (jnp.sum(own_certain,axis=0)[None,:]-own_certain)>0
    partner_backup = jnp.sum(hand*(know[1,:,:25]>.99999),axis=0)>0
    backup = jnp.where(own_backup,1.,.8*partner_backup[None,:])
    duplicate_safety = jnp.sum(own*backup*(~dead)[None,:],axis=-1)
    discard_safety = jnp.minimum(pdead+duplicate_safety,1.)
    discard_cost = jnp.sum(own*card_cost[None,:]*(1-backup),axis=-1)
    hinted = jnp.any(know[0,:,25:] > 0, axis=-1)
    discard_score = 0.9 + 1.1*discard_safety - 0.6*discard_cost - 0.8*pplay - 0.25*hinted*(1-discard_safety) - 0.015*jnp.arange(5)
    recent_hint = f[254] * (f[257] + f[258]) * f[271:276]
    discard_score -= 3.0 * recent_hint * (1-discard_safety)
    discard_score += jnp.maximum(0.6-jnp.max(discard_score), 0.)
    # Simulate the literal positive and negative information of every legal hint.
    attributes = jnp.concatenate([colors[None,:] == jnp.arange(5)[:,None], ranks[None,:] == jnp.arange(5)[:,None]], 0)
    matched = jnp.einsum('st,ht->hs', hand, attributes.astype(jnp.float32)) > 0
    after_support = support[1][None,:,:] & jnp.where(matched[:,:,None], attributes[:,None,:], ~attributes[:,None,:])
    after = normalize(after_support * receiver_weights[None,:,:])
    before_play = other @ playable.astype(jnp.float32)
    after_play = after @ playable.astype(jnp.float32)
    true_play = hand @ playable.astype(jnp.float32)
    true_dead = hand @ dead.astype(jnp.float32)
    true_progress = hand @ (playable*progress_value).astype(jnp.float32)
    new_safe = jnp.maximum((after_play >= .999).astype(jnp.float32) - (before_play >= .999).astype(jnp.float32), 0) * true_play
    gain = jnp.maximum(after_play - before_play, 0) * true_play
    info = jnp.maximum(entropy(other)[None,:] - entropy(after), 0)
    new_info = jnp.sum((support[1][None,:,:] & ~after_support), axis=(1,2)) > 0
    danger = jnp.sum(matched * (1-true_play) * (1-true_dead) * (after_play > .25), axis=-1)
    hint_score = 5*jnp.max(new_safe*true_progress,axis=-1) + 1.8*jnp.sum(new_safe*true_progress,axis=-1) + 2*jnp.sum(gain*true_progress,axis=-1) + .45*jnp.sum(info*true_play,axis=-1) + .06*jnp.sum(info*(1-true_dead),axis=-1) - .4*danger
    partner_hinted = jnp.any(know[1,:,25:] > 0,axis=-1)
    fresh_touch = matched & (~partner_hinted)[None,:] & (before_play[None,:] < .8)
    all_playable = jnp.sum(matched*(1-true_play)[None,:],axis=-1) < .01
    distinct_fresh = jnp.sum(jnp.minimum(jnp.einsum('hs,st->ht',fresh_touch.astype(jnp.float32),hand),1.),axis=-1)
    hint_score += 2.0*all_playable*distinct_fresh
    hint_score += .15 * (tokens >= 7) - .35 * (tokens <= 2)
    hint_score = jnp.where(new_info, hint_score, -2.)
    ambiguous_bad = jnp.sum(matched*(1-true_play)*(after_play>jnp.where(lives<=1,.01,.15)),axis=-1)
    caution = jnp.where(lives<=1,16.,jnp.where(next_memory['receiver_alarm'],5.,jnp.where(next_memory['receiver_eager'],1.,0.)))
    hint_score -= caution*ambiguous_bad
    hint_score = jnp.where(deck == 0, hint_score*.15 - .5, hint_score)
    advanced_heights = height[None,:] + jax.nn.one_hot(colors,5)*playable[:,None]
    future_playable = (ranks[None,:] == advanced_heights[:,colors]).astype(jnp.float32)
    future_partner_prob = jnp.einsum('su,tu->ts',other,future_playable)
    future_partner_true = jnp.einsum('su,tu->ts',hand,future_playable)
    future_partner_ready = jnp.max((future_partner_prob>=.999)*future_partner_true,axis=-1)
    partner_ready_now = jnp.max((before_play>=.999)*true_play)
    partner_after_play = own @ (future_partner_ready*jnp.where(lives<=1,playable,True))
    hinted_future_prob = jnp.einsum('hsu,tu->hts',after,future_playable)
    hinted_future_ready = jnp.max((hinted_future_prob>=.999)*future_partner_true[None,:,:],axis=-1)
    future_gain = jnp.maximum(hinted_future_ready-future_partner_ready[None,:],0.)
    our_next_slot = jnp.argmax(play_score)
    preparation_gain = future_gain @ (own[our_next_slot]*playable)
    harmless_setup = jnp.max(jnp.where(matched,after_play,0.),axis=-1)<.01
    hint_score += jnp.where((deck==1)&(pplay[our_next_slot]>.9999),12.*preparation_gain*harmless_setup,0.)
    scores = jnp.concatenate([discard_score, play_score, hint_score, jnp.array([-100.])])
    final_play = effective_play - jnp.where(lives <= 1, (1-pplay)*score, 0.)
    final_play += jnp.where(final_turn==1,partner_after_play-partner_ready_now,0.)
    partner_after_hint = jnp.max((after_play>=.999)*true_play[None,:],axis=-1)
    final_hint = jnp.where(final_turn==1,.99*jnp.maximum(partner_after_hint-partner_ready_now,0.),-1.)
    final_hint -= ((lives<=1) | next_memory['receiver_alarm'])*(ambiguous_bad>0)*jnp.where(lives<=1,score+1.,1.)
    final_scores = jnp.concatenate([jnp.zeros(5), final_play, final_hint, jnp.array([-100.])])
    scores = jnp.where(deck == 0, final_scores, scores)
    action = jnp.argmax(jnp.where(legal, scores, -1e6)).astype(jnp.int32)
    return action, next_memory, {'partner_waits':waits, 'receiver_eager':next_memory['receiver_eager'], 'pplay':pplay, 'pdead':pdead, 'hint_score':hint_score, 'discard_score':discard_score, 'score':score, 'features':f, 'hint_reliability':hint_reliability, 'effective_play':effective_play, 'receiver_alarm':next_memory['receiver_alarm']}
