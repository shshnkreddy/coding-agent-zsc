"""Convention-free color grouping from local colors and public hint sets."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'color_groups', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return {'colors': jnp.zeros((16,4),jnp.float32), 'positions': jnp.zeros((16,2),jnp.int32), 'evidence':jnp.zeros((16,16),jnp.float32)}


def beliefs(obs, memory):
    f = obs.node_features
    seen = f[:16,:4].astype(jnp.float32)
    known = jnp.where(jnp.sum(seen,axis=1,keepdims=True)>0, seen, memory['colors'])
    hints = f[16:,4:8].astype(jnp.float32)
    links = obs.adjacency_matrix[:16,16:].astype(jnp.float32)
    allowed = jnp.where(jnp.sum(links,axis=1,keepdims=True)>0, links @ hints, 1.)
    allowed = jnp.where(jnp.sum(known,axis=1,keepdims=True)>0, known, allowed)
    def body(_, p):
        certain = (jnp.max(p,axis=1)>0.999).astype(jnp.float32)
        remaining = jnp.maximum(4.-jnp.sum(p*certain[:,None],axis=0),0.)
        q = allowed * remaining[None,:]
        q = q / jnp.maximum(q.sum(axis=1,keepdims=True),1e-8)
        return jnp.where(certain[:,None]>0,p,q)
    p = allowed/jnp.maximum(allowed.sum(axis=1,keepdims=True),1.)
    p = jax.lax.fori_loop(0,4,body,p)
    hard=p
    certain=(hard.max(-1)>0.999)
    remaining=jnp.maximum(4.-jnp.sum(hard*certain[:,None],axis=0),0.)
    def propagate(_,q):
        prob=hard*jnp.exp(memory['evidence']@q)
        prob=prob/jnp.maximum(prob.sum(-1,keepdims=True),1e-8)
        def balance(_,prob):
            totals=jnp.sum(prob*(~certain)[:,None],axis=0)
            scaled=prob*remaining[None,:]/jnp.maximum(totals[None,:],1e-6)
            scaled=scaled/jnp.maximum(scaled.sum(-1,keepdims=True),1e-8)
            return jnp.where(certain[:,None],hard,scaled)
        return jax.lax.fori_loop(0,4,balance,prob)
    p=jax.lax.fori_loop(0,4,propagate,p)
    return p, known, hard


def exact_tail(pos,p,hard,evidence):
    uncertain=hard.max(-1)<0.999
    ids=jnp.argsort((~uncertain).astype(jnp.int32),stable=True)[:4]
    outside=hard.argmax(-1)
    selected=jnp.zeros(16,jnp.bool_).at[ids].set(True)
    counts=4-jnp.sum(jax.nn.one_hot(outside,4)*(~selected)[:,None],axis=0).astype(jnp.int32)
    available=jnp.searchsorted(jnp.cumsum(counts),jnp.arange(4),side='right')
    perm=jnp.asarray(((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0), (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0)),jnp.int32)
    code=jnp.clip(available[perm],0,3)
    color=jnp.broadcast_to(outside,(24,16)).at[:,ids].set(code)
    onehot=jax.nn.one_hot(color,4)
    valid=jnp.all(onehot.sum(1)==4,axis=-1)
    allowed=jnp.all(hard[ids[None,:],code]>0,axis=-1)&valid
    evidence_value=0.5*jnp.sum((color[:,:,None]==color[:,None,:])*evidence[None,:,:],axis=(1,2))
    weights=jnp.exp(evidence_value-evidence_value.max())*allowed
    weight_sum=weights.sum()
    weights=weights/jnp.maximum(weight_sum,1e-12)
    marginals=jnp.einsum('s,sic->ic',weights,onehot)
    same=jnp.einsum('s,sic,sjc->ij',weights,onehot,onehot)
    adj=(jnp.abs(pos[:,None]-pos[None,:]).sum(-1)==1)
    edges=jnp.sum((color[:,:,None]==color[:,None,:])*adj[None,:,:],axis=(1,2))/2
    grid=jnp.zeros((24,10,10,4),jnp.float32).at[:,pos[:,0],pos[:,1]].set(onehot)
    square=(grid[:,:-1,:-1]*grid[:,1:,:-1]*grid[:,:-1,1:]*grid[:,1:,1:]).sum(axis=(1,2,3))
    solved=edges-square>11.99
    # Universal proof over all assignments allowed by hard information.
    feasible=jnp.prod(hard[ids[None,:],code],axis=-1)>0
    feasible=feasible&valid
    all_solved=jnp.all(solved|(~feasible))&jnp.any(feasible)
    enabled=(uncertain.sum()<=4)&(weight_sum>1e-12)
    return jnp.where(enabled,marginals,p),jnp.where(enabled,same,p@p.T),enabled&all_solved,color,weights,enabled,jnp.where(enabled,(weights*solved).sum(),0.)


def geometry(pos,p):
    dist = jnp.abs(pos[:,None,:]-pos[None,:,:]).sum(-1)
    adj = (dist==1).astype(jnp.float32)
    same = p@p.T
    grid = jnp.zeros((10,10,4),jnp.float32).at[pos[:,0],pos[:,1]].set(p)
    squares = (grid[:-1,:-1]*grid[1:,:-1]*grid[:-1,1:]*grid[1:,1:]).sum()
    value = (adj*same).sum()/2-squares
    return dist,adj,same,grid,value


def joint_squares(pos,color,weights):
    cells=jnp.stack((jnp.arange(100)//10,jnp.arange(100)%10),axis=-1)
    ids=jnp.full((10,10),-1,jnp.int32).at[pos[:,0],pos[:,1]].set(jnp.arange(16))
    result=jnp.zeros((16,100),jnp.float32)
    for dx,dy in ((1,1),(1,-1),(-1,1),(-1,-1)):
        c1=cells+jnp.array([dx,0]);c2=cells+jnp.array([0,dy]);c3=cells+jnp.array([dx,dy])
        i1=ids[jnp.clip(c1[:,0],0,9),jnp.clip(c1[:,1],0,9)]
        i2=ids[jnp.clip(c2[:,0],0,9),jnp.clip(c2[:,1],0,9)]
        i3=ids[jnp.clip(c3[:,0],0,9),jnp.clip(c3[:,1],0,9)]
        valid=jnp.all((c3>=0)&(c3<10),axis=-1)&(i1>=0)&(i2>=0)&(i3>=0)
        valid=valid[None,:]&(jnp.arange(16)[:,None]!=i1)&(jnp.arange(16)[:,None]!=i2)&(jnp.arange(16)[:,None]!=i3)
        match=(color[:,:,None]==color[:,None,i1])&(color[:,:,None]==color[:,None,i2])&(color[:,:,None]==color[:,None,i3])
        result+=jnp.einsum('s,sid->id',weights,match.astype(jnp.float32))*valid
    return result


def move_scores(pos,p,joint_same,tail_colors,tail_weights,tail_enabled):
    dist,adj,same,grid,value=geometry(pos,p)
    same=joint_same
    cells=jnp.stack((jnp.arange(100)//10,jnp.arange(100)%10),axis=-1)
    newdist=jnp.abs(cells[:,None,:]-pos[None,:,:]).sum(-1)
    neighbors=(newdist==1).astype(jnp.float32)
    eye=jnp.eye(16)
    weights=same*(1-eye)
    edge_gain=weights@neighbors.T-(weights*adj).sum(-1)[:,None]
    # A monochromatic 2x2 is the only possible cycle for a four-card color.
    def square_at(g, coords):
        accum=jnp.zeros((coords.shape[0],4),jnp.float32)
        for dx,dy in ((1,1),(1,-1),(-1,1),(-1,-1)):
            c1=coords+jnp.array([dx,0]); c2=coords+jnp.array([0,dy]); c3=coords+jnp.array([dx,dy])
            valid=jnp.all((c3>=0)&(c3<10),axis=-1)
            a=g[jnp.clip(c1[:,0],0,9),jnp.clip(c1[:,1],0,9)]
            b=g[jnp.clip(c2[:,0],0,9),jnp.clip(c2[:,1],0,9)]
            c=g[jnp.clip(c3[:,0],0,9),jnp.clip(c3[:,1],0,9)]
            accum=accum+a*b*c*valid[:,None]
        return accum
    oldsquare=(square_at(grid,pos)*p).sum(-1)
    def per_card(i):
        g=grid.at[pos[i,0],pos[i,1]].set(0.)
        return square_at(g,cells)@p[i]
    newsquare=jax.vmap(per_card)(jnp.arange(16))
    exact_squares=joint_squares(pos,tail_colors,tail_weights)
    newsquare=jnp.where(tail_enabled,exact_squares,newsquare)
    oldsquare=jnp.where(tail_enabled,exact_squares[jnp.arange(16),pos[:,0]*10+pos[:,1]],oldsquare)
    gain=edge_gain-newsquare+oldsquare[:,None]
    distance_gain=((weights*dist).sum(-1)[:,None]-weights@newdist.T)
    # Leave room around color groups, and prefer short, easy-to-read relocations.
    compact_gain=dist.sum(-1)[:,None]-((1-eye)@newdist.T)
    scores=gain+0.035*distance_gain+0.002*compact_gain
    return scores,gain,value


def physical_legal(pos,locked):
    ids=jnp.arange(16)
    alive=ids[:,None]!=ids[None,:]
    adj=jnp.abs(pos[:,None]-pos[None,:]).sum(-1)<=1
    reach=adj[None,:,:]&alive[:,:,None]&alive[:,None,:]
    reach=jax.lax.fori_loop(0,4,lambda _,r:r|(jnp.einsum('ikj,ijl->ikl',r.astype(jnp.float32),r.astype(jnp.float32))>0),reach)
    cells=jnp.stack((jnp.arange(100)//10,jnp.arange(100)%10),axis=-1)
    dist=jnp.abs(cells[:,None]-pos[None,:]).sum(-1)
    touched=jnp.einsum('dk,ikj->idj',(dist==1).astype(jnp.float32),reach.astype(jnp.float32))>0
    return jnp.all(touched|(~alive[:,None,:]),axis=-1)&jnp.all(dist>0,axis=-1)[None,:]&(~locked[:,None])


def planned_move(pos,p,joint_same,tail_colors,tail_weights,tail_enabled,locked,legal,ms,remaining):
    masked=jnp.where(legal,ms,-1e6)
    _,global_best=jax.lax.top_k(masked.reshape(-1),8)
    _,sources=jax.lax.top_k(masked.max(-1),8)
    diversified=sources*100+jnp.argmax(masked,axis=-1)[sources]
    candidates=jnp.concatenate((global_best,diversified))
    def follow(action):
        i=action//100;cell=action%100
        newpos=pos.at[i].set(jnp.array([cell//10,cell%10]))
        nextlegal=physical_legal(newpos,locked)
        future=move_scores(newpos,p,joint_same,tail_colors,tail_weights,tail_enabled)[0]
        return jnp.max(jnp.where(nextlegal,future,-1e6))
    next_values=jax.lax.map(follow,candidates)
    values=masked.reshape(-1)[candidates]+jnp.where(remaining>1,0.65*next_values,0.)
    return 17+candidates[jnp.argmax(values)]


def observation_value(pos,p,move_legal):
    eye=jnp.eye(16)
    dist=jnp.abs(pos[:,None]-pos[None,:]).sum(-1)
    cells=jnp.stack((jnp.arange(100)//10,jnp.arange(100)%10),axis=-1)
    neighbors=jnp.abs(cells[:,None]-pos[None,:]).sum(-1)==1
    cw=p.T[:,None,:]*(1-eye)[None,:,:]
    gains=jnp.einsum('cij,dj->cid',cw,neighbors.astype(jnp.float32))-jnp.sum(cw*(dist==1)[None,:,:],axis=-1)[:,:,None]
    best_by_color=jnp.max(jnp.where(move_legal[None,:,:],gains,-5.),axis=-1)
    revealed_best=jnp.sum(best_by_color*p.T,axis=0)
    expected_gain=jnp.sum(gains*p.T[:,:,None],axis=0)
    uncertain_best=jnp.max(jnp.where(move_legal,expected_gain,-5.),axis=-1)
    return jnp.maximum(revealed_best-uncertain_best,0.),revealed_best


def agent_act(obs, reset, memory, rng):
    memory=jax.tree.map(lambda x,z:jnp.where(reset,z,x),memory,init_agent())
    pos=obs.positions
    moved=jnp.any(pos!=memory['positions'],axis=-1)&(~reset)
    oldadj=jnp.abs(memory['positions'][:,None,:]-memory['positions'][None,:,:]).sum(-1)==1
    adj=jnp.abs(pos[:,None,:]-pos[None,:,:]).sum(-1)==1
    changed=moved[:,None]|moved[None,:]
    partnerseen=obs.node_features[:16,11]>0
    created=adj&(~oldadj)&changed&(obs.node_features[0,12]==0)&partnerseen[:,None]&partnerseen[None,:]
    evidence=jnp.where(changed,0.,memory['evidence'])+2.*created.astype(jnp.float32)
    memory={**memory,'evidence':evidence}
    p,known,hard=beliefs(obs,memory)
    p,joint_same,tail_solved,tail_colors,tail_weights,tail_enabled,finish_prob=exact_tail(obs.positions,p,hard,evidence)
    pos=obs.positions; f=obs.node_features; legal=obs.legal_actions
    dist,adj,same,grid,value=geometry(pos,p)
    certainty=p.max(-1)
    # Observe new cards, prioritizing accessible pieces near known information.
    entropy=1.-jnp.sum(p*p,axis=-1)
    reconstructed=physical_legal(pos,f[:16,10]>0)
    voi,potential=observation_value(pos,p,reconstructed)
    observe_score=entropy+1.5*voi+0.15*jnp.maximum(potential,0.)-0.025*adj.sum(-1)-0.15*f[:16,11]
    observe_score=jnp.where(entropy>0.001,observe_score,-1e3)
    observe_score+=0.01*jax.random.uniform(rng,(16,))
    observe_action=1+jnp.argmax(jnp.where(legal[1:17],observe_score,-1e9))
    ms,gain,value=move_scores(pos,p,joint_same,tail_colors,tail_weights,tail_enabled)
    remaining=14-jnp.sum(jnp.any(f[16:,4:8]>0,axis=-1))-jnp.sum(f[16:,10])
    move_action=planned_move(pos,p,joint_same,tail_colors,tail_weights,tail_enabled,f[:16,10]>0,legal[17:1617].reshape(16,100),ms,remaining)
    # Reveal all hints to preserve freedom, then lock the best established cards.
    hints=f[16:,4:8].astype(jnp.float32)
    hint_correct=hints@p.T
    degree=(adj*same).sum(-1)
    near=(same*jnp.maximum(dist-1,0)).sum(-1)
    confirmed=(same>0.999)&(adj>0)
    reach=confirmed|jnp.eye(16,dtype=jnp.bool_)
    reach=jax.lax.fori_loop(0,4,lambda _,r:r|((r.astype(jnp.float32)@r.astype(jnp.float32))>0),reach)
    split=(same*(~reach)*f[:16,10][None,:]).sum(-1)
    far=(same*(dist>3)*f[:16,10][None,:]).sum(-1)
    hint_score=10.*hint_correct+0.7*degree[None,:]-0.12*near[None,:]-3.*split[None,:]-6.*far[None,:]
    info = jnp.log2(4./jnp.maximum(hints.sum(-1),1.))[:,None]*(1-f[:16,11].astype(jnp.float32))[None,:]
    hint_score+=0.65*info+0.1*(1-certainty)[None,:]
    cellids=pos[:,0]*10+pos[:,1]
    place_scores=jnp.full((7,100),-1e8).at[:,cellids].set(hint_score)
    place_action=1624+jnp.argmax(jnp.where(legal[1624:2324].reshape(7,100),place_scores,-1e9))
    reveal_action=1617+jnp.argmax(legal[1617:1624])
    size=hints.sum(-1)[:,None]
    informative=(size==1)|((size==2)&(f[:16,11][None,:]==0))
    early_safe=informative&(hint_correct>0.999)&(confirmed.sum(-1)[None,:]>=1)&(split[None,:]<0.01)
    early_scores=jnp.full((7,100),-1e8).at[:,cellids].set(jnp.where(early_safe,hint_score,-1e8))
    early_scores=jnp.where(legal[1624:2324].reshape(7,100),early_scores,-1e9)
    early_action=1624+jnp.argmax(early_scores)
    hint_action=jnp.where(jnp.any(legal[1617:1624]),jnp.where(early_scores.max()>-1e7,early_action,reveal_action),place_action)
    revealed=jnp.any(f[16:,4:8]>0,axis=-1)
    unplaced=revealed&(f[16:,10]==0)
    current_reward=5*jnp.sum(~revealed)+2*jnp.sum(unplaced)+jnp.sum(f[16:,10])
    next_cost=jnp.where(jnp.any(unplaced),1.,3.)
    finish_threshold=jnp.maximum(0.85,1.-next_cost/jnp.maximum(current_reward,1)+0.015)
    solved=tail_solved|(finish_prob>finish_threshold)
    hint_action=jnp.where(solved&jnp.any(legal[1624:2324]),place_action,hint_action)
    action=jnp.where(f[0,13]==2,move_action,jnp.where(f[0,13]==3,hint_action,observe_action))
    action=jnp.where(legal[2324]&solved,2324,action)
    action=jnp.where(f[0,12]>0,action,0).astype(jnp.int32)
    return action,{'colors':known,'positions':pos,'evidence':evidence},{'beliefs':p,'known':known,'finish_prob':finish_prob,'finish_threshold':finish_threshold,'legal_error':jnp.where((f[0,12]>0)&(f[0,13]==2),jnp.sum(reconstructed!=legal[17:1617].reshape(16,100)),0),'connectivity':value,'positions':pos,'active':f[0,12],'step':f[0,13]}
