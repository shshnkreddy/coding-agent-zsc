"""Literal-information Hanabi policy, with count-aware beliefs and helpful hints."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_cooperator', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {'end_age':jnp.asarray(0,dtype=jnp.int32),'hinted':jnp.zeros(5,dtype=jnp.bool_),'last_prompt':jnp.asarray(False),'failed_prompt':jnp.asarray(False),'discard_evidence':jnp.full(2,2.,dtype=jnp.float32),'discard_prediction':jnp.zeros(2,dtype=jnp.int32)}


def norm(x):
    return x / jnp.maximum(x.sum(axis=-1, keepdims=True), 1e-8)


def exact_belief(mask, counts):
    """Marginals over five slots sampled without replacement, given literal masks.

    Inclusion-exclusion over the 15 partitions of the other four slots sums
    all assignments of distinct physical copies without enumerating hands.
    Empty slots receive private dummy copies, which cancel in normalization.
    """
    mask = jnp.concatenate([mask, ~jnp.any(mask,axis=-1,keepdims=True)],axis=-1).astype(jnp.float32)
    counts = jnp.concatenate([counts,jnp.asarray([5.])])
    complement = jnp.asarray([[1,2,3,4],[0,2,3,4],[0,1,3,4],[0,1,2,4],[0,1,2,3]])
    subsets = (jnp.arange(16)[:,None] & (1 << jnp.arange(4)[None,:])) > 0
    products = jnp.prod(jnp.where(subsets[None,:,:,None],mask[complement][:,None,:,:],1.),axis=2)
    moments = (products*counts[None,None,:]).sum(-1)
    reduced = moments[:,:,None]-products
    reduced = reduced.at[:,0,:].set(1.)
    partitions = jnp.asarray([[1,2,4,8],[3,4,8,0],[5,2,8,0],[9,2,4,0],[6,1,8,0],[10,1,4,0],[12,1,2,0],[3,12,0,0],[5,10,0,0],[9,6,0,0],[7,8,0,0],[11,4,0,0],[13,2,0,0],[14,1,0,0],[15,0,0,0]])
    coefficients = jnp.asarray([1,-1,-1,-1,-1,-1,-1,1,1,1,2,2,2,2,-6])
    likelihood = (jnp.prod(reduced[:,partitions,:],axis=2)*coefficients[None,:,None]).sum(axis=1)
    weights = mask*counts[None,:]*jnp.maximum(likelihood,0.)
    return norm(weights)[:,:-1]


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions
    hand = f[:125].reshape(5,25)
    know = f[308:].reshape(2,5,35)
    ownmask = know[0,:,:25] > 0
    othermask = know[1,:,:25] > 0
    fireworks = (f[167:192].reshape(5,5) * jnp.arange(1,6)).sum(-1).astype(jnp.int32)
    ranks = jnp.tile(jnp.arange(5),5)
    colors = jnp.repeat(jnp.arange(5),5)
    dis = f[203:253].reshape(5,10)
    disc = jnp.stack([dis[:,:3].sum(-1), dis[:,3:5].sum(-1), dis[:,5:7].sum(-1), dis[:,7:9].sum(-1), dis[:,9]],-1)
    public = (jnp.asarray([3,2,2,2,1])[None,:] - disc - (jnp.arange(5)[None,:] < fireworks[:,None])).reshape(25)
    counts = jnp.maximum(public-hand.sum(0),0)
    own = exact_belief(ownmask, counts)
    other_counts = jnp.maximum(public-own.sum(0),0.)
    other = norm(othermask * other_counts)
    playable = ranks == fireworks[colors]
    exhausted = (public.reshape(5,5) <= 0) & (jnp.arange(5)[None,:] >= fireworks[:,None])
    blocked = jnp.cumsum(exhausted,axis=-1).reshape(25) > 0
    dead = (ranks < fireworks[colors]) | blocked
    useful = ~dead
    critical = (public <= 1) & useful
    ceiling = jnp.where(exhausted.any(-1),jnp.argmax(exhausted,axis=-1),5)
    protected_points = jnp.maximum(ceiling[colors]-ranks,0)
    pplay = own @ playable.astype(jnp.float32)
    pdead = own @ dead.astype(jnp.float32)
    pcritical = own @ (critical.astype(jnp.float32)*protected_points)
    discard_loss_type = useful*protected_points*(.20**jnp.maximum(public-1,0))
    discard_cost = own @ discard_loss_type
    tokens = f[192:200].sum()
    lives = f[200:203].sum()
    deck = f[127:167].sum()
    threshold = jnp.where(lives >= 3, .90, jnp.where(lives >= 2, .97, .9999))
    threshold = jnp.where((deck == 0) & (lives > 1), .15, threshold)
    prior_prompt = jnp.where(reset,False,memory['last_prompt'])
    failed_prompt = jnp.where(reset,False,memory['failed_prompt']) | (prior_prompt & (f[253]>0) & (f[255]>0) & (f[306]==0))
    prior_hinted = jnp.where(reset,jnp.zeros(5,dtype=jnp.bool_),memory['hinted'])
    incoming = f[271:276]>0
    new_incoming = incoming & ~prior_hinted
    old_dead = jnp.all(~(incoming & prior_hinted)|(pdead>.999))
    one_new = (new_incoming.sum()==1)&old_dead
    focus_slots = jnp.where(incoming.sum()==1,incoming,new_incoming&one_new)
    focused = ~failed_prompt & (f[259]>0)&((f[257]+f[258])>0)&((incoming.sum()==1)|one_new)&(tokens<7)
    implied = focused*focus_slots*(know[0,:,34]==0)
    prior_chop = jnp.argmax(jnp.where(~prior_hinted,5-jnp.arange(5),-1))
    possible_save = (jnp.arange(5)==prior_chop)&(pcritical>.25)
    prompt_threshold = jnp.where(lives>=3,jnp.where(possible_save,.65,.20),jnp.where(lives>=2,jnp.where(possible_save,.90,.75),.9999))
    threshold = jnp.where(implied,jnp.minimum(threshold,prompt_threshold),threshold)
    playval = 4 + pplay + .03 * (own @ (ranks == 4).astype(jnp.float32)) - .015*jnp.arange(5)
    # Tie-break certain plays by literal, visible follow-up opportunities.
    next_map = ranks[None,:] == (fireworks[colors][None,:]+(colors[:,None]==colors[None,:]))
    next_other_p = other @ next_map.astype(jnp.float32).T
    next_other_true = hand @ next_map.astype(jnp.float32).T
    current_other_true = hand @ playable.astype(jnp.float32)
    future_ready = jnp.any((next_other_p>.999)&(next_other_true>0)&(current_other_true[:,None]==0),axis=0)
    future_visible = jnp.any((next_other_true>0)&(current_other_true[:,None]==0),axis=0)
    own_follow = jnp.any(((own @ next_map.astype(jnp.float32).T)>.999)&(pplay[:,None]<.999),axis=0)
    sequence_value = own @ (.08*future_ready+.03*future_visible+.04*own_follow)
    playval += jnp.where(pplay>.999,sequence_value,0.)
    playval = jnp.where(pplay >= threshold,playval,-100.)
    # Preserve hinted and irreplaceable cards, and prefer known obsolete cards.
    hinted = know[0,:,25:].sum(-1) > 0
    fresh_hint = (f[259] > 0) & ((f[257]+f[258]) > 0)
    fresh = fresh_hint * f[271:276]
    discardval = 1.3*pdead - .8*discard_cost - 1.0*pplay - .65*hinted - .12*jnp.arange(5) - 1.5*fresh*(1-pdead)
    discardval = discardval - jnp.max(jnp.where(legal[:5],discardval,-100.)) + .25 + .15*(tokens<=2)
    # Literal effect of all ten possible hints, including negative information.
    attr = jnp.concatenate([colors[None,:] == jnp.arange(5)[:,None], ranks[None,:] == jnp.arange(5)[:,None]],0)
    matches = (hand[None,:,:] * attr[:,None,:]).sum(-1) > 0
    newmask = othermask[None,:,:] & (attr[:,None,:] == matches[:,:,None])
    after = norm(newmask * other_counts[None,None,:])
    before_play = other @ playable.astype(jnp.float32)
    after_play = after @ playable.astype(jnp.float32)
    trueplay = hand @ playable.astype(jnp.float32)
    truedead = hand @ dead.astype(jnp.float32)
    truecritical = hand @ critical.astype(jnp.float32)
    gain = (after_play**4 - before_play[None,:]**4) * trueplay[None,:]
    # Duplicate playable copies should not count as separate enabled plays.
    cardgain = jnp.max(gain[:,:,None] * hand[None,:,:],axis=1).sum(-1)
    certainty = ((after_play > .999) & (before_play[None,:] < .999)) * trueplay[None,:]
    certain_gain = jnp.max(certainty[:,:,None] * hand[None,:,:],axis=1).sum(-1)
    entropy_before = -(other*jnp.log(jnp.maximum(other,1e-8))).sum(-1)
    entropy_after = -(after*jnp.log(jnp.maximum(after,1e-8))).sum(-1)
    information = jnp.maximum(entropy_before[None,:]-entropy_after,0)
    info_value = (information*(.1+.6*trueplay[None,:]+.15*truecritical[None,:])*(1-truedead[None,:])).sum(-1)
    touched = know[1,:,25:].sum(-1)>0
    newly = matches & ~touched[None,:]
    dangerous = (newly * (1-trueplay)[None,:]).sum(-1)
    promising = (newly * trueplay[None,:]).sum(-1)
    ambiguity = jnp.where(promising>0,.75*dangerous,0.)
    prior_other_dead = (other @ dead.astype(jnp.float32))>.999
    other_one_new = (newly.sum(-1)==1)&jnp.all(~(matches&touched[None,:])|prior_other_dead[None,:],axis=-1)
    hint_focus = (matches.sum(-1)==1)|other_one_new
    hint_target = jnp.where((matches.sum(-1)==1)[:,None],matches,newly)
    false_prompt = hint_focus&((hint_target*trueplay[None,:]).sum(-1)==0)&(jnp.max(after_play*hint_target,axis=-1)>.15)&(jnp.arange(10)!=9)
    false_prompt &= ~jnp.any((after_play>.999)&(trueplay[None,:]>0),axis=-1)
    ambiguity += 1.2*false_prompt
    hintval = 2.0*cardgain + .5*certain_gain + .35*info_value - .18 - .18*(tokens<=2) - ambiguity
    pairmask = newmask[:,None,:,:] & (attr[None,:,None,:] == matches[None,:,:,None])
    pairbelief = norm(pairmask * other_counts[None,None,None,:])
    pairplay = pairbelief @ playable.astype(jnp.float32)
    pairgain = (pairplay**4 - before_play[None,None,:]**4) * trueplay[None,None,:]
    paircardgain = jnp.max(pairgain[:,:,:,None]*hand[None,None,:,:],axis=2).sum(-1)
    synergy = jnp.where(matches.any(-1)[None,:],jnp.maximum(paircardgain-cardgain[:,None]-cardgain[None,:],0),0).max(-1)
    hintval += .95*synergy
    # Once the partner has a certain move, allow them to take it before investing.
    hintval -= .5*jnp.any((before_play>.999)&(trueplay>0))
    # Rank five is an unambiguous literal statement of irreplaceability.
    other_dead = other @ dead.astype(jnp.float32)
    other_loss = other @ discard_loss_type
    other_discard = 1.3*other_dead-.8*other_loss-before_play-.65*touched-.12*jnp.arange(5)
    fresh_other_hint = (f[260]>0)&((f[257]+f[258])>0)
    other_discard -= 1.5*fresh_other_hint*f[271:276]*(1-other_dead)
    chop = jnp.argmax(jnp.where(~touched,5-jnp.arange(5),-1))
    chop = jnp.where(jnp.any(other_dead>.999),jnp.argmax(jnp.where(other_dead>.999,5-jnp.arange(5),-1)),chop)
    utility_chop = jnp.argmax(other_discard)
    evidence = jnp.where(reset,jnp.full(2,2.,dtype=jnp.float32),memory['discard_evidence'])
    prediction = memory['discard_prediction']
    observed_slot = jnp.argmax(f[276:281])
    observed_discard = ~reset & (f[254]>0)&(f[256]>0)
    hits = prediction==observed_slot
    informative_discard = observed_discard&(prediction[0]!=prediction[1])&jnp.any(hits)
    evidence += informative_discard*hits.astype(jnp.float32)
    model_weights = evidence/evidence.sum()
    threat = model_weights[0]*jax.nn.one_hot(chop,5)+model_weights[1]*jax.nn.one_hot(utility_chop,5)
    partner_ready = jnp.any((before_play>.85)&(trueplay>0))
    unrecognized_five = (hand @ ((ranks==4)&useful).astype(jnp.float32))*(know[1,:,34]==0)
    save_five = jnp.sum(threat*unrecognized_five)*~partner_ready
    hintval = hintval.at[9].add(1.6*save_five)
    urgent_five = (save_five>.80)&(tokens>=2)&(deck>6)
    hintval = hintval.at[9].set(jnp.where(urgent_five,jnp.maximum(hintval[9],5.3),hintval[9]))
    # Other irreplaceable prerequisites may be saved when the clue clearly
    # describes a future card, rather than suggesting an immediate play.
    after_dead = after @ dead.astype(jnp.float32)
    after_loss = after @ discard_loss_type
    after_discard = 1.3*after_dead-.8*after_loss-after_play-.65*(touched[None,:]|matches)-.12*jnp.arange(5)[None,:]-1.5*matches*(1-after_dead)
    after_choice = jnp.argmax(after_discard,axis=-1)
    after_chop = jnp.argmax(jnp.where(~(touched[None,:]|matches),5-jnp.arange(5)[None,:],-1),axis=-1)
    after_chop = jnp.where(jnp.any(after_dead>.999,axis=-1),jnp.argmax(jnp.where(after_dead>.999,5-jnp.arange(5)[None,:],-1),axis=-1),after_chop)
    after_threat = model_weights[1]*jax.nn.one_hot(after_choice,5)+model_weights[0]*jax.nn.one_hot(after_chop,5)
    actual_loss = hand @ (critical.astype(jnp.float32)*protected_points)
    saved = (threat[None,:]*actual_loss[None,:]*matches).sum(-1)-(after_threat*actual_loss[None,:]).sum(-1)
    clear_future = jnp.max(after_play*matches,axis=-1)<.15
    informative = jnp.any(newmask != othermask[None,:,:],axis=(1,2))
    saved = jnp.maximum(saved,0)*clear_future*informative*~partner_ready
    hintval += 1.2*saved
    urgent = (saved>=1.8)&(jnp.max(threat)>.80)&(tokens>=2)&(deck>6)
    hintval = jnp.where(urgent,jnp.maximum(hintval,5.4+.05*saved),hintval)
    # Let a partner with a known play take the final draw when our hand
    # has no move; their extra final turn can finish a visible sequence.
    ready = (before_play>.98)&(trueplay>0)
    stall = (deck==1)&(tokens>0)&(jnp.max(playval)<0)&jnp.any(ready)
    reinforce = (matches*ready[None,:]).sum(-1)
    stall_value = .8+.1*reinforce-.4*dangerous
    hintval = jnp.where(stall&(reinforce>0),jnp.maximum(hintval,stall_value),hintval)
    # The two turns after the final draw have a different value of information.
    end_age = jnp.where(reset,0,memory['end_age'])
    final_play = 5*(pplay-jnp.where(lives<=1,(1-pplay)*fireworks.sum(),0.))
    partner_already_ready = jnp.any(((other @ playable.astype(jnp.float32))>.999)&(current_other_true>0))
    final_play += jnp.where((end_age==0)&(pplay>.999)&~partner_already_ready,5*(own @ future_ready.astype(jnp.float32)),0.)
    final_play = jnp.where(pplay>1e-7,final_play,-1.)
    playval = jnp.where(deck==0,final_play,playval)
    next_gain = jnp.max(after_play*trueplay[None,:],axis=-1)-jnp.max(before_play*trueplay)
    final_hint = 5*next_gain-.05
    hintval = jnp.where(deck==0,jnp.where(end_age==0,final_hint,-.5),hintval)
    discardval = jnp.where(deck==0,discardval-jnp.max(discardval),discardval)
    scores = jnp.concatenate([discardval,playval,hintval,jnp.asarray([-100.])])
    action = jnp.argmax(jnp.where(legal,scores,-1e9)).astype(jnp.int32)
    prompt_used = (action>=5)&(action<10)&implied[jnp.clip(action-5,0,4)]&(pplay[jnp.clip(action-5,0,4)]<.9)
    return action,{'end_age':jnp.where(deck==0,end_age+1,0).astype(jnp.int32),'hinted':hinted,'last_prompt':prompt_used,'failed_prompt':failed_prompt,'discard_evidence':evidence,'discard_prediction':jnp.stack([chop,utility_chop]).astype(jnp.int32)},{'play_p':pplay,'implied':implied,'threshold':threshold,'failed_prompt':failed_prompt,'model_weights':model_weights,'dead_p':pdead,'hint_value':hintval,'hint_ambiguity':jnp.where(promising>0,dangerous,0.),'discard_value':discardval,'lives':lives,'deck':deck,'tokens':tokens,'board':fireworks,'hand':jnp.argmax(hand,axis=-1),'last':f[253:261],'revealed':jnp.argmax(f[281:306])}
