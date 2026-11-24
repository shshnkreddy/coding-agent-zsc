"""Local, evidence-based cooperative Yōkai policy."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_grouping', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return {'colors': jnp.zeros((16,4), jnp.float32),
            'positions': jnp.zeros((16,2), jnp.int32),
            'links': jnp.zeros((16,16), jnp.float32),
            'started': jnp.asarray(False)}


def adjacency(pos):
    return jnp.sum(jnp.abs(pos[:,None,:]-pos[None,:,:]),axis=-1)==1


def components(adj, labels):
    a=adj & (labels[:,None]==labels[None,:])
    ids=jnp.arange(16,dtype=jnp.int32)
    return jax.lax.fori_loop(0, 8, lambda _,v:jnp.minimum(v,jnp.min(jnp.where(a,v[None,:],99),axis=1)),ids)


def beliefs(obs, memory):
    cards=obs.node_features[:16]
    seen=cards[:,:4].astype(jnp.float32)
    known=jnp.where(jnp.sum(seen,axis=1,keepdims=True)>0,seen,memory['colors'])
    hints=obs.node_features[16:,4:8].astype(jnp.float32)
    attached=obs.adjacency_matrix[:16,16:].astype(jnp.float32)
    hintmask=attached@hints
    hintmask=jnp.where(jnp.sum(hintmask,axis=1,keepdims=True)>0,hintmask,1.)
    remaining=jnp.maximum(4-jnp.sum(known,axis=0),0.)
    prior=hintmask*jnp.maximum(remaining[None,:],1e-6)
    prior=prior/jnp.maximum(jnp.sum(prior,axis=1,keepdims=True),1e-8)
    exact=jnp.sum(known,axis=1)>0
    prior=jnp.where(exact[:,None],known,prior)
    moved=jnp.any(obs.positions!=memory['positions'],axis=1) & memory['started']
    partner_move=(cards[0,12]==0)&(cards[0,13]==3)
    adj=adjacency(obs.positions)
    # Movements towards a group are evidence of shared color, never hard facts.
    newlinks=moved[:,None]&adj&partner_move
    links=memory['links']
    links=jnp.where(moved[:,None]|moved[None,:],0.,links)
    links=jnp.maximum(links,jnp.maximum(newlinks,newlinks.T).astype(jnp.float32))
    def iteration(_,p):
        messages=links@p
        p=prior*jnp.exp(2.5*messages)
        p=jnp.where(exact[:,None],0.,p)
        def balance(_,q):
            q=q*remaining[None,:]/jnp.maximum(jnp.sum(q,axis=0,keepdims=True),1e-8)
            return q/jnp.maximum(jnp.sum(q,axis=1,keepdims=True),1e-8)
        p=jax.lax.fori_loop(0,12,balance,p)
        return jnp.where(exact[:,None],known,p)
    p=jax.lax.fori_loop(0,5,iteration,prior)
    return p,known,links


def quality(pos,labels,conf,locked):
    adj=adjacency(pos)
    comp=components(adj,labels)
    same=(labels[:,None]==labels[None,:]) & (conf[:,None]>.65) & (conf[None,:]>.65)
    connected=comp[:,None]==comp[None,:]
    distance=jnp.sum(jnp.abs(pos[:,None,:]-pos[None,:,:]),axis=-1)
    ncomp=jnp.sum((comp==jnp.arange(16))&(conf>.65))
    gap=jnp.sum(jnp.where(same & ~connected,distance,0))/2
    edges=jnp.sum(adj&same)/2
    size=jnp.sum(connected,axis=1)
    anchored=jnp.any(connected & locked[None,:],axis=1)
    color_members=(labels[:,None]==jnp.arange(4)[None,:]) & (conf[:,None]>.65)
    anchor_rank=jnp.where(color_members,size[:,None]+100*anchored[:,None],-1000)
    best_anchor=jnp.argmax(anchor_rank,axis=0)
    mass=jnp.sum(jnp.where(jnp.any(color_members,axis=0),size[best_anchor],0))
    return -10*ncomp-.13*gap+.12*edges+2.5*mass


def move_action(obs,p):
    conf=jnp.max(p,axis=1)
    labels=jnp.where(conf>.65,jnp.argmax(p,axis=1),jnp.arange(16)+4)
    ids=jnp.arange(1600)//100
    cells=jnp.stack(((jnp.arange(1600)%100)//10,jnp.arange(1600)%10),axis=1)
    positions=jnp.broadcast_to(obs.positions,(1600,16,2)).at[jnp.arange(1600),ids].set(cells)
    values=jax.vmap(quality,in_axes=(0,None,None,None))(positions,labels,conf,obs.node_features[:16,10]>0)
    # Mild preference for public, readable moves involving privately known cards.
    values+=.03*(conf[ids])
    values-=.0001*jnp.sum(jnp.abs(cells-obs.positions[ids]),axis=1)
    return 17+jnp.argmax(jnp.where(obs.legal_actions[17:1617],values,-1e9))


def observation_action(obs,p,known):
    adj=adjacency(obs.positions)
    exact=jnp.sum(known,axis=1)>0
    degree=jnp.sum(adj,axis=1)
    entropy=-jnp.sum(p*jnp.log(jnp.maximum(p,1e-8)),axis=1)
    distance=jnp.sum(jnp.abs(obs.positions[:,None,:]-obs.positions[None,:,:]),axis=-1)
    # Observations are more useful when a color match would join separate cards.
    opportunity=jnp.sum((p@known.T)*jnp.minimum(jnp.maximum(distance-1,0),3),axis=1)
    score=20*(~exact)+2*entropy+.6*(degree<4)+1.2*opportunity
    score-=.2*obs.node_features[:16,11]
    return 1+jnp.argmax(jnp.where(obs.legal_actions[1:17],score,-1e9))


def hint_action(obs,p):
    hints=obs.node_features[16:,4:8].astype(jnp.float32)
    conf=jnp.max(p,axis=1)
    labels=jnp.where(conf>.65,jnp.argmax(p,axis=1),jnp.arange(16)+4)
    comp=components(adjacency(obs.positions),labels)
    samecomp=comp[:,None]==comp[None,:]
    size=jnp.sum(samecomp,axis=1)
    locked=obs.node_features[:16,10]>0
    # Avoid freezing a second, separate anchor for the same color.
    otherlocked=jnp.sum((labels[:,None]==labels[None,:])&~samecomp&locked[None,:],axis=1)
    match=hints@p.T
    tilesize=jnp.sum(hints,axis=1)
    scores=100*match+3*size[None,:]-12*otherlocked[None,:]+.4*(4-tilesize[:,None])
    dest=obs.positions[:,0]*10+obs.positions[:,1]
    actions=1624+jnp.arange(7)[:,None]*100+dest[None,:]
    masked=jnp.where(obs.legal_actions[actions],scores,-1e9)
    place=actions.reshape(-1)[jnp.argmax(masked)]
    reveal=1617+jnp.argmax(obs.legal_actions[1617:1624])
    informative=(tilesize[:,None]==1)&(match>.999)&(size[None,:]>=2)&(otherlocked[None,:]==0)
    private=obs.node_features[:16,:4].astype(jnp.float32)
    private_pair=jnp.sum(adjacency(obs.positions)*(private@private.T),axis=1)>0
    informative &= private_pair[None,:] & obs.legal_actions[actions]
    info_scores=scores+2*(1-obs.node_features[:16,11])[None,:]
    info_place=actions.reshape(-1)[jnp.argmax(jnp.where(informative,info_scores,-1e9))]
    choice=jnp.where(jnp.any(obs.legal_actions[1617:1624]),reveal,place)
    return jnp.where(jnp.any(informative),info_place,choice)


def agent_act(obs,reset,memory,rng):
    memory=jax.tree.map(lambda a,b:jnp.where(reset,a,b),init_agent(),memory)
    p,known,links=beliefs(obs,memory)
    active=obs.node_features[0,12]>0
    phase=obs.node_features[0,13].astype(jnp.int32)
    conf=jnp.max(p,axis=1)
    labels=jnp.argmax(p,axis=1)
    comp=components(adjacency(obs.positions),labels)
    complete=jnp.all((labels[:,None]!=labels[None,:])|(comp[:,None]==comp[None,:])) & jnp.all(conf>.94) & jnp.all(jnp.sum(jax.nn.one_hot(labels,4),axis=0)==4)
    action=jax.lax.switch(phase,[lambda: jnp.where(complete,2324,observation_action(obs,p,known)),lambda:observation_action(obs,p,known),lambda:move_action(obs,p),lambda:hint_action(obs,p)])
    action=jnp.where(active,action,0).astype(jnp.int32)
    next_memory={'colors':known,'positions':obs.positions,'links':links,'started':jnp.asarray(True)}
    return action,next_memory,{'beliefs':p,'known':jnp.sum(known),'complete':complete}
