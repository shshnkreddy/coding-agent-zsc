"""Local evidence and color-component clustering for cooperative Yokai."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_components', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return {'colors': jnp.zeros((16,4), jnp.float32),
            'positions': jnp.zeros((16,2), jnp.int32),
            'moved': jnp.zeros(16, jnp.bool_)}


def components(adj):
    labels = jnp.arange(16, dtype=jnp.int32)
    labels = jnp.broadcast_to(labels, adj.shape[:-1])
    return jax.lax.fori_loop(0, 16, lambda _, v: jnp.min(jnp.where(adj, v[..., None, :], 16), axis=-1), labels)


def get_belief(obs, colors):
    hints = obs.node_features[16:,4:8].astype(jnp.float32)
    links = obs.adjacency_matrix[:16,16:].astype(jnp.bool_)
    allowed = jnp.prod(jnp.where(links[:,:,None], hints[None,:,:], 1.),axis=1)
    known = jnp.sum(colors,axis=1)>0
    allowed = jnp.where(known[:,None], colors, allowed)
    p = allowed / jnp.maximum(allowed.sum(-1,keepdims=True),1.)
    def balance(_, p):
        remaining = jnp.maximum(4.-colors.sum(0), 0.)
        free = jnp.where(known[:,None], 0., p)
        free = free*remaining[None,:]/jnp.maximum(free.sum(0,keepdims=True),1e-6)
        free = free/jnp.maximum(free.sum(-1,keepdims=True),1e-6)
        return jnp.where(known[:,None],colors,free)
    return jax.lax.fori_loop(0,12,balance,p)


def agent_act(obs, reset, memory, rng):
    mem = jax.tree.map(lambda a,b:jnp.where(reset,a,b),init_agent(),memory)
    nf=obs.node_features
    pos=obs.positions
    active=nf[0,12]>0
    phase=nf[0,13]
    visible=nf[:16,:4].astype(jnp.float32)
    colors=jnp.where(visible.sum(-1,keepdims=True)>0,visible,mem['colors'])
    moved=mem['moved'] | ((jnp.any(pos!=mem['positions'],axis=-1)) & ~reset)
    p=get_belief(obs,colors)
    certainty=p.max(-1)
    label=jnp.where(certainty>.97,p.argmax(-1),4+jnp.arange(16))
    same=label[:,None]==label[None,:]
    dist=jnp.abs(pos[:,None,:]-pos[None,:,:]).sum(-1)
    adj=(dist==1)
    ca=(adj & same)|jnp.eye(16,dtype=jnp.bool_)
    comp=components(ca)
    locked=nf[:16,10]>0
    known=colors.sum(-1)>0
    sizes=jnp.sum(comp[:,None]==comp[None,:],axis=-1)
    done=jnp.all(certainty>.97)&jnp.all(jnp.where(same,comp[:,None]==comp[None,:],True))
    ids=jnp.arange(16)
    entropy=-jnp.sum(p*jnp.log(jnp.maximum(p,1e-8)),axis=-1)
    degree=adj.sum(-1)
    # Learn moved pieces and border pieces, which can most readily be repaired.
    observe=entropy*10.+moved*1.5+(degree<=2)*.6+jnp.sum(adj*known[None,:],axis=-1)*.15
    oa=jnp.argmax(jnp.where(obs.legal_actions[1:17],observe,-1e9))+1
    oa=jnp.where((phase==0)&done,2324,oa)

    def move_action(_):
        # Connected components with each possible source removed. This counts
        # true component mergers and penalizes splitting an existing group.
        removal=(ids[:,None,None]!=ids[None,:,None])&(ids[:,None,None]!=ids[None,None,:])
        reduced=(ca[None,:,:]&removal)|jnp.eye(16,dtype=jnp.bool_)[None,:,:]
        rc=components(reduced)
        root=rc==ids[None,:]
        root=root & (ids[:,None]!=ids[None,:])
        old_count=jnp.sum(comp==ids)
        base_count=root.sum(-1)+1
        cells=jnp.stack((jnp.arange(100)//10,jnp.arange(100)%10),axis=-1)
        dd=jnp.abs(cells[:,None,:]-pos[None,:,:]).sum(-1)
        neighbors=(dd[None,:,:]==1)&same[:,None,:]&(ids[:,None,None]!=ids[None,None,:])
        reaches=jnp.any(neighbors[:,:,:,None]&(rc[:,None,:,None]==ids[None,None,None,:]),axis=2)
        merges=(reaches&root[:,None,:]).sum(-1)
        gain=old_count-(base_count[:,None]-merges)
        weights=(p@p.T)*(1-jnp.eye(16))
        # Only attraction supported by confident evidence; uncertain cards
        # otherwise bias every move toward the middle.
        weights=weights*(certainty[:,None]>.9)*(certainty[None,:]>.9)
        distance_gain=jnp.sum(weights[:,None,:]*(dist[:,None,:]-dd[None,:,:]),axis=-1)
        central_gain=dist.sum(-1)[:,None]-dd.sum(-1)[None,:]
        score=gain*8.+distance_gain*.16+central_gain*.006
        score+=jnp.minimum(jnp.sum(dd==1,axis=-1)[None,:]-(dd.T==1).astype(jnp.int32),3)*.015
        score+=jax.random.uniform(rng,(16,100))*.001
        return 17+jnp.argmax(jnp.where(obs.legal_actions[17:1617].reshape(16,100),score,-1e9))
    ma=jax.lax.cond(active&(phase==2),move_action,lambda _:jnp.asarray(17),operand=None)

    hints=nf[16:,4:8].astype(jnp.float32)
    correct=hints@p.T
    same_neighbors=jnp.sum(adj&same,axis=-1)
    placement=correct*20.+sizes[None,:]*1.+same_neighbors[None,:]*.4
    placement+=(hints.sum(-1)[:,None]==1)*(.6+~(nf[:16,11]>0)[None,:])
    cell_id=pos[:,0]*10+pos[:,1]
    place_scores=jnp.full((7,100),-1e9).at[:,cell_id].set(placement)
    # Every unplaced reveal preserves mobility for another turn.
    hint_scores=jnp.concatenate((50.-jnp.arange(7)*.1,place_scores.reshape(-1)))
    ha=1617+jnp.argmax(jnp.where(obs.legal_actions[1617:2324],hint_scores,-1e9))
    action=jnp.where(phase<2,oa,jnp.where(phase==2,ma,ha)).astype(jnp.int32)
    action=jnp.where(active,action,0)
    return action, {'colors':colors,'positions':pos,'moved':moved}, {'known':known.sum(),'inferred':(certainty>.97).sum(),'components':jnp.sum(comp==ids),'done':done,'belief':p}
