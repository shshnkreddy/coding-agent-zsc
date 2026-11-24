"""Local evidence-based cooperative Yōkai policy."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_clusters', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return {'known': jnp.zeros((16,4),jnp.float32), 'positions': jnp.zeros((16,2),jnp.int32), 'relations': jnp.zeros((16,16),jnp.float32)}


def closure(adj):
    def f(k, a):
        return a | (a[..., :, k, None] & a[..., None, k, :])
    return jax.lax.fori_loop(0,16,f,adj)


def beliefs(obs, known, relations):
    visible=obs.node_features[:16,:4].astype(jnp.float32)
    known=jnp.maximum(known,visible)
    checked=(known.sum(1)[:,None]>0) & (known.sum(1)[None,:]>0)
    checked=checked*relations
    total=checked.sum()/2.
    correct=(checked*(known@known.T)).sum()/2.
    trust=(8.+correct)/(10.+total)
    strength=jnp.clip(jnp.log(3.*trust/(1.-trust)),.7,3.)
    hints=obs.node_features[16:,4:8].astype(jnp.float32)
    linked=obs.adjacency_matrix[:16,16:].astype(jnp.float32)
    allowed=jnp.where(linked.sum(1,keepdims=True)>0, linked@hints, 1.)
    witnessed_error=jnp.any((known.sum(1)>0) & (linked.sum(1)>0) & ((known*allowed).sum(1)==0))
    allowed=jnp.where(witnessed_error,jnp.ones_like(allowed),allowed)
    base=jnp.where(known.sum(1,keepdims=True)>0,known,allowed)
    def deduce(_,a):
        definite=a.sum(1,keepdims=True)==1
        slots=4.-jnp.where(definite,a,0.).sum(0)
        a=jnp.where(definite,a,a*(slots[None,:]>0))
        possible=jnp.where(definite,0.,a)
        required=(possible.sum(0)==slots) & (slots>0)
        force=possible*required[None,:]
        return jnp.where(force.sum(1,keepdims=True)==1,force,a)
    base=jax.lax.fori_loop(0,8,deduce,base)
    # A color subset can accommodate at most four cards per included color.
    # Checking all 16 subsets detects collectively inconsistent hint constraints.
    subsets=((jnp.arange(16)[:,None] >> jnp.arange(4)[None,:]) & 1).astype(jnp.bool_)
    contained=jnp.all((base[None,:,:]==0) | subsets[:,None,:],axis=-1)
    contradiction=jnp.any(contained.sum(1)>4*subsets.sum(1))
    direct_only=jnp.where(known.sum(1,keepdims=True)>0,known,jnp.ones_like(known))
    fallback=jax.lax.fori_loop(0,8,deduce,direct_only)
    base=jnp.where(contradiction,fallback,base)
    fixed=base.sum(1,keepdims=True)==1
    anchors=jnp.where(fixed,base,0.)
    capacity=jnp.maximum(0.,4.-anchors.sum(0))
    def balance(_,p):
        p=p/jnp.maximum(p.sum(1,keepdims=True),1e-6)
        unknown=jnp.where(fixed,0.,p)
        unknown=unknown*capacity/jnp.maximum(unknown.sum(0),1e-6)
        return jnp.where(fixed,anchors,unknown)
    def infer(_,p):
        evidence=jnp.exp(jnp.clip(strength*(relations@p),0.,8.))
        q=jnp.where(fixed,anchors,base*evidence)
        return jax.lax.fori_loop(0,8,balance,q)
    p=jax.lax.fori_loop(0,5,infer,base/base.sum(1,keepdims=True))
    p=jax.lax.fori_loop(0,12,balance,p)
    p=p/jnp.maximum(p.sum(1,keepdims=True),1e-6)
    # Exact small residual posterior; initial uncertain boards retain the soft model.
    uncertain=~fixed[:,0]
    n=uncertain.sum()
    idx=jnp.argsort(~uncertain,stable=True)[:6]
    use=jnp.arange(6)<n
    assignments=(jnp.arange(4096)[:,None]//(4**jnp.arange(6)[None,:]))%4
    oh=jax.nn.one_hot(assignments,4)
    counts=jnp.sum(oh*use[None,:,None],axis=1)
    valid=jnp.all(counts==capacity[None,:],axis=1)
    valid &= jnp.all((base[idx][jnp.arange(6)[None,:],assignments]>0) | ~use[None,:],axis=1)
    valid &= jnp.all(use[None,:] | (assignments==0),axis=1)
    unary=relations[idx]@anchors
    energy=jnp.sum(unary[jnp.arange(6)[None,:],assignments]*use[None,:],axis=1)
    rr=relations[idx[:,None],idx[None,:]]*use[:,None]*use[None,:]
    def pair_energy(k,e):
        match=assignments[:,k,None]==assignments
        return e+jnp.sum(match*rr[k][None,:]*(jnp.arange(6)[None,:]<k),axis=1)
    energy=strength*jax.lax.fori_loop(0,6,pair_energy,energy)
    logw=jnp.where(valid,energy,-1e9)
    weights=jnp.where(valid,jnp.exp(logw-jnp.max(logw)),0.)
    z=jnp.maximum(weights.sum(),1e-20)
    marginal=jnp.einsum('b,bkc->kc',weights,oh,precision=jax.lax.Precision.HIGHEST)/z
    exact=p.at[idx].set(jnp.where(use[:,None],marginal,p[idx]))
    best=assignments[jnp.argmax(logw)]
    labels=p.argmax(-1).at[idx].set(jnp.where(use,best,p[idx].argmax(-1)))
    confidence=weights.max()/z
    usable=(n<=6) & jnp.any(valid)
    labels=jnp.where(usable,labels,p.argmax(-1))
    confidence=jnp.where(usable,confidence,jnp.prod(p.max(-1)))
    p=jnp.where(usable,exact,p)
    p=p/jnp.maximum(p.sum(1,keepdims=True),1e-20)
    return p,known,labels,confidence


def move_scores(pos,p,locked):
    ids=jnp.arange(16)
    grid=jnp.stack((jnp.arange(100)//10,jnp.arange(100)%10),1)
    dist=jnp.abs(pos[:,None]-pos[None,:]).sum(-1)
    adj=dist==1
    conf=p.max(-1)>.97
    colors=p.argmax(-1)
    same=(colors[:,None]==colors[None,:]) & conf[:,None] & conf[None,:]
    ca=adj & same
    # Connectivity after each possible source card is removed.
    keep=ids[:,None]!=ids[None,:]
    reach=closure(ca[None,:,:] & keep[:,:,None] & keep[:,None,:] | jnp.eye(16,dtype=bool)[None,:,:])
    roots=jnp.min(jnp.where(reach,ids[None,None,:],16),-1)
    reps=roots==ids[None,:]
    dst_dist=jnp.abs(grid[:,None,:]-pos[None,:,:]).sum(-1)
    near=(dst_dist==1)[None,:,:] & keep[:,None,:]
    old=adj & keep
    # Number of distinct same-color groups attached to this card.
    relevant=same & keep
    touched=jnp.einsum('sdi,sir->sdr',(near & relevant[:,None,:]).astype(jnp.float32),reach.astype(jnp.float32))>0
    newgroups=(touched & reps[:,None,:]).sum(-1)
    oldtouch=jnp.any(old[:,:,None] & relevant[:,:,None] & reach,axis=1)
    oldgroups=(oldtouch & reps).sum(-1)
    delta=newgroups-oldgroups[:,None]
    similarity=p@p.T
    similarity=similarity*keep
    edges=jnp.einsum('sdi,si->sd',near.astype(jnp.float32),similarity)-jnp.sum(adj*similarity,axis=1)[:,None]
    # Soft distance discourages scattering disconnected groups.
    newdistance=jnp.einsum('di,si->sd',dst_dist.astype(jnp.float32),similarity)
    olddistance=(dist*similarity).sum(-1)
    distance_delta=newdistance-olddistance[:,None]
    degree=near.sum(-1)
    old_degree=adj.sum(-1)
    score=3.*delta+.20*edges-.045*distance_delta+.025*(degree-old_degree[:,None])
    # Prefer acting on reliable information.
    score-=.08*(1.-p.max(-1))[:,None]
    fullreach=closure(adj[None,:,:] & keep[:,:,None] & keep[:,None,:] | jnp.eye(16,dtype=bool)[None,:,:])
    connected=jnp.einsum('sdi,sir->sdr',near.astype(jnp.float32),fullreach.astype(jnp.float32))>0
    empty=~jnp.any(dst_dist==0,axis=1)
    legal=jnp.all(connected | ~keep[:,None,:],axis=-1) & empty[None,:] & ~locked[:,None]
    return score, legal, ca, conf


def agent_act(obs, reset, memory, rng):
    known=jnp.where(reset,0.,memory['known'])
    relations=jnp.where(reset,0.,memory['relations'])
    moved=jnp.any(obs.positions!=memory['positions'],axis=1) & ~reset
    partner_move=(obs.node_features[0,12]==0) & (obs.node_features[0,13]==3)
    adj=jnp.abs(obs.positions[:,None]-obs.positions[None,:]).sum(-1)==1
    oldadj=jnp.abs(memory['positions'][:,None]-memory['positions'][None,:]).sum(-1)==1
    seen=obs.node_features[:16,11]>0
    additions=(moved[:,None] | moved[None,:]) & adj & ~oldadj & seen[:,None] & seen[None,:] & partner_move
    relations=jnp.maximum(relations,additions.astype(jnp.float32))
    p,known,assignment,confidence=beliefs(obs,known,relations)
    active=obs.node_features[0,12]>0
    sub=obs.node_features[0,13].astype(jnp.int32)
    legal=obs.legal_actions
    locked=obs.node_features[:16,10]>0
    scores,computed_legal,ca,conf=move_scores(obs.positions,p,locked)
    legalmove=legal[17:1617].reshape(16,100)
    masked=jnp.where(legalmove,scores,-1e9)
    # Include each movable source, ensuring geometric setup moves get considered.
    candidates=jnp.arange(16)*100+jnp.argmax(masked,axis=1)
    immediate=masked.reshape(-1)[candidates]
    def examine(k, values):
        a=candidates[k]
        cell=a%100
        pos=obs.positions.at[a//100].set(jnp.array([cell//10,cell%10]))
        future, allowed, _, _=move_scores(pos,p,locked)
        best_future=jnp.max(jnp.where(allowed,future,-1e6))
        return values.at[k].set(immediate[k]+.85*best_future)
    def planning_step(state):
        k,values=state
        return k+1,examine(k,values)
    _,values=jax.lax.while_loop(
        lambda state: (state[0]<16) & active & (sub==2),
        planning_step,(jnp.asarray(0,jnp.int32),jnp.zeros((16,),jnp.float32)))
    turns_used=jnp.sum(obs.node_features[16:,10])+jnp.sum(jnp.any(obs.node_features[16:,4:8]>0,axis=1))
    best_direct=jnp.argmax(masked)
    planned=candidates[jnp.argmax(values)]
    move=17+jnp.where(turns_used<13,planned,best_direct)
    dist=jnp.abs(obs.positions[:,None]-obs.positions[None,:]).sum(-1)
    deg=(dist==1).sum(-1)
    # New information is useful even if the partner has seen the card.
    entropy=1.-jnp.sum(p*p,axis=1)
    observed=known.sum(1)>0
    ids=jnp.arange(16)
    keep=ids[:,None]!=ids[None,:]
    allreach=closure((dist==1)[None,:,:] & keep[:,:,None] & keep[:,None,:] | jnp.eye(16,dtype=bool)[None,:,:])
    removable=jnp.all(allreach | ~keep[:,:,None] | ~keep[:,None,:],axis=(1,2))
    exposed=jnp.maximum(0,4-deg)
    target_known=known*(exposed[:,None]>0)
    match_opportunity=p@target_known.sum(0)
    observation_score=10.*entropy-2.*observed+.45*removable+.20*exposed+.15*match_opportunity-.6*obs.node_features[:16,11].astype(jnp.float32)*entropy
    observation_score+=.001*jax.random.uniform(rng,(16,))
    observe=1+jnp.argmax(jnp.where(legal[1:17],observation_score,-1e9))
    hints=obs.node_features[16:,4:8].astype(jnp.float32)
    hint_size=hints.sum(-1)
    # Reveal narrow hints when no safe, informative placement is available.
    reveal=jnp.argmax(legal[1617:1624])
    reveal_action=1617+reveal
    match=hints@p.T
    same_neighbors=(ca).sum(-1)
    reach=closure(ca | jnp.eye(16,dtype=bool))
    group_size=reach.sum(-1)
    lock_score=5.*match+.5*same_neighbors[None,:]+.2*group_size[None,:]+.05*deg[None,:]-.1*hint_size[:,None]
    cardcell=obs.positions[:,0]*10+obs.positions[:,1]
    place_ids=1624+jnp.arange(7)[:,None]*100+cardcell[None,:]
    locked=obs.node_features[:16,10]>0
    labels=p.argmax(-1)
    separate_anchor=jnp.any(locked[None,:] & (labels[:,None]==labels[None,:]) & ~reach,axis=1)
    safe=(group_size>=2) & (p.max(-1)>.97) & ~separate_anchor
    safe_place=legal[place_ids] & safe[None,:] & (match>.99)
    # Early single-color hints are particularly useful for a partner.
    information=(1.-obs.node_features[:16,11].astype(jnp.float32))[None,:]/jnp.maximum(hint_size[:,None],1.)
    lock_score=lock_score+2.*information
    early=jnp.any(safe_place)
    selected=jnp.argmax(jnp.where(jnp.where(early,safe_place,legal[place_ids]),lock_score,-1e9))
    place=place_ids.reshape(-1)[selected]
    hint=jnp.where(jnp.any(legal[1617:1624]) & ~early,reveal_action,place)
    tentative_reach=closure(((dist==1) & (assignment[:,None]==assignment[None,:])) | jnp.eye(16,dtype=bool))
    fully_connected=jnp.all((tentative_reach | (assignment[:,None]!=assignment[None,:]))) & (confidence>.65) & jnp.all(jax.nn.one_hot(assignment,4).sum(0)==4)
    action=jnp.where(sub==0,jnp.where(fully_connected,2324,observe),jnp.where(sub==1,observe,jnp.where(sub==2,move,hint)))
    action=jnp.where(active,action,0).astype(jnp.int32)
    action=jnp.where(legal[action],action,jnp.argmax(legal)).astype(jnp.int32)
    memory={'known':known,'positions':obs.positions,'relations':relations}
    return action,memory,{'belief':p,'move_value':jnp.max(jnp.where(legalmove,scores,-1e9)), 'solved':fully_connected,'joint_confidence':confidence,'legality_agreement':jnp.where(active & (sub==2),jnp.all(computed_legal==legalmove),True)}
