"""Color-group cooperation using private observations and public hints."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'evidence_clustering', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return {'colors': jnp.zeros((16, 4), jnp.float32),
            'positions': jnp.zeros((16, 2), jnp.int32),
            'moved': jnp.zeros((16,), jnp.float32),
            'relations': jnp.zeros((16,16), jnp.float32)}


def beliefs(obs, colors, relations):
    nf = obs.node_features
    visible = nf[:16, :4].astype(jnp.float32)
    colors = jnp.where(jnp.sum(visible, axis=1, keepdims=True)>0, visible, colors)
    known = jnp.sum(colors, axis=1)>0
    hint = nf[16:, 4:8] > 0
    attached = obs.adjacency_matrix[:16, 16:] > 0
    allowed = jnp.all(~attached[:, :, None] | hint[None, :, :], axis=1)
    allowed = jnp.where(known[:, None], colors > 0, allowed)
    # Exact four-of-each deductions, followed by soft count balancing.
    def narrow(_, a):
        certain = a & (a.sum(1, keepdims=True)==1)
        full = certain.sum(0) >= 4
        a = a & (~full[None, :] | certain)
        forced = a & (a.sum(0, keepdims=True)==4)
        forced_rows = forced.sum(1, keepdims=True)==1
        return jnp.where(forced_rows, forced, a)
    allowed = jax.lax.fori_loop(0, 6, narrow, allowed)
    allowed = jnp.where(allowed.any(1, keepdims=True), allowed, jnp.ones_like(allowed))
    p = allowed.astype(jnp.float32)
    def balance(_, p):
        p = p / jnp.maximum(p.sum(1, keepdims=True), 1e-6)
        return p * (4.0 / jnp.maximum(p.sum(0, keepdims=True), 1e-6))
    p = jax.lax.fori_loop(0, 12, balance, p)
    p = p / jnp.maximum(p.sum(1, keepdims=True), 1e-6)
    base = p
    def infer(_, q):
        q = base*jnp.exp(2.2*(relations @ q))
        q = jax.lax.fori_loop(0,8,balance,q)
        return q / jnp.maximum(q.sum(1,keepdims=True),1e-6)
    p = jax.lax.fori_loop(0,4,infer,p)
    return colors, p


def graph_data(pos, p):
    ids = jnp.arange(16)
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    adj = dist==1
    sure = p.max(1) > 0.999
    same = (p.argmax(1)[:, None] == p.argmax(1)[None, :]) & sure[:, None] & sure[None, :]
    color_adj = adj & same
    # Connected components after removal of each possible source card.
    retained = ids[:, None] != ids[None, :]
    graphs = color_adj[None, :, :] & retained[:, :, None] & retained[:, None, :]
    graphs = graphs | jnp.eye(16, dtype=jnp.bool_)[None, :, :]
    def propagate(_, labels):
        return jnp.min(jnp.where(graphs, labels[:, None, :], 16), axis=-1)
    labels = jax.lax.fori_loop(0, 15, propagate, jnp.broadcast_to(ids, (16,16)))
    members = jax.nn.one_hot(labels,16) * retained[:, :, None]
    # Full-board components, for stopping and placement choices.
    full = color_adj | jnp.eye(16, dtype=jnp.bool_)
    def prop(_, labels):
        return jnp.min(jnp.where(full, labels[None,:], 16),axis=-1)
    full_labels = jax.lax.fori_loop(0,15,prop,ids)
    size = (full_labels[:, None] == full_labels[None,:]).sum(1)
    return adj, dist, members, full_labels, size


def move_scores(pos, legal, p, members):
    cells = jnp.stack((jnp.arange(100)//10,jnp.arange(100)%10),axis=1)
    distances = jnp.abs(cells[:,None,:]-pos[None,:,:]).sum(-1)
    nearby = (distances==1).astype(jnp.float32)
    touched = jnp.einsum('dj,ijk->idk',nearby,members)>0
    similarity = p @ p.T
    links = jnp.einsum('idk,ik->id',touched.astype(jnp.float32),similarity)
    current_cell = pos[:,0]*10+pos[:,1]
    gain = links-links[jnp.arange(16),current_cell][:,None]
    # Smaller distance helps unite separate groups when a direct merge is unavailable.
    others = 1-jnp.eye(16)
    attraction = similarity * others
    distance_score = -jnp.einsum('dj,ij->id',distances.astype(jnp.float32),attraction)
    distance_gain = distance_score-distance_score[jnp.arange(16),current_cell][:,None]
    # Avoid needless disruption of fully assembled colors and permit future growth.
    edge_count = jnp.einsum('dj,ij->id',nearby,attraction)
    edge_gain = edge_count-edge_count[jnp.arange(16),current_cell][:,None]
    known = p.max(1)
    score = 10*gain + 0.12*distance_gain + 0.25*edge_gain
    score -= 0.1*(1-known[:,None])
    # Mild preference for compact whole-board configurations.
    board_dist = distances.sum(1)[None,:] - distances[current_cell,:].sum(1)[:,None]
    score -= 0.005*board_dist
    return jnp.where(legal,score,-1e9)


def future_legal(pos, locks):
    ids = jnp.arange(16)
    cells = jnp.stack((jnp.arange(100)//10,jnp.arange(100)%10),axis=1)
    adj = jnp.abs(pos[:,None,:]-pos[None,:,:]).sum(-1)==1
    retained = ids[:,None] != ids[None,:]
    graph = adj[None,:,:] & retained[:,:,None] & retained[:,None,:]
    graph |= jnp.eye(16,dtype=jnp.bool_)[None,:,:]
    def propagate(_, labels):
        return jnp.min(jnp.where(graph,labels[:,None,:],16),axis=-1)
    labels = jax.lax.fori_loop(0,15,propagate,jnp.broadcast_to(ids,(16,16)))
    members = jax.nn.one_hot(labels,16)*retained[:,:,None]
    num_components = ((labels==ids[None,:]) & retained).sum(1)
    near = jnp.abs(cells[:,None,:]-pos[None,:,:]).sum(-1)==1
    touched = jnp.einsum('dj,ijk->idk',near.astype(jnp.float32),members)>0
    occupied = jnp.any(jnp.all(cells[:,None,:]==pos[None,:,:],axis=-1),axis=-1)
    return (touched.sum(-1)==num_components[:,None]) & ~occupied[None,:] & ~locks[:,None]


def choose_move(obs, p, members, size):
    legal = obs.legal_actions[17:1617].reshape(16,100)
    scores = move_scores(obs.positions,legal,p,members)
    source_scores = scores.max(1)
    _, sources = jax.lax.top_k(source_scores,8)
    destinations = scores.argmax(1)[sources]
    def evaluate(src,dst):
        pos = obs.positions.at[src].set(jnp.array([dst//10,dst%10]))
        _,_,future_members,_,_ = graph_data(pos,p)
        future_mask = future_legal(pos,obs.node_features[:16,10]>0)
        next_scores = move_scores(pos,future_mask,p,future_members)
        next_score = jnp.maximum(next_scores.max(),-20.)
        return scores[src,dst] + 0.85*next_score
    values = jax.vmap(evaluate)(sources,destinations)
    chosen = jnp.argmax(values)
    planned = sources[chosen]*100+destinations[chosen]
    # Early moves have too little color information for useful lookahead.
    use_plan = (p.max(1)>0.999).sum()>=8
    index = jnp.where(use_plan,planned,jnp.argmax(scores))
    return (17+index).astype(jnp.int32), scores.reshape(-1)[index]


def choose_hint(obs, p, adj, size):
    nf = obs.node_features
    hint = nf[16:,4:8].astype(jnp.float32)
    hint_size = hint.sum(1)
    correct = hint @ p.T
    same = p @ p.T
    support = (adj*same).sum(1)
    lock = nf[:16,10]>0
    locked_support = (adj*same*lock[None,:]).sum(1)
    # Attach to well-established groups, especially those already carrying locks.
    anchor = 1.0*size + 0.8*support + 0.3*locked_support
    placement = 12*correct + anchor[None,:] - 0.3*hint_size[:,None]
    placement = jnp.where(correct>0.999, placement, placement-30)
    cell = obs.positions[:,0]*10+obs.positions[:,1]
    place_ids = 1624+jnp.arange(7)[:,None]*100+cell[None,:]
    legal_place = obs.legal_actions[place_ids]
    # Revealing preserves mobility. A correct singleton on a robust core is public evidence.
    reveal_score = jnp.array([24.,23.8,23.5,23.4,23.3,23.1,23.])
    free_neighbors = 4-adj.sum(1)
    exposed = free_neighbors>=2
    singleton_anchor = (size>=2) | exposed
    placement += jnp.where((hint_size==1)[:,None] & singleton_anchor[None,:],9.,0.)
    novel = 1-nf[:16,11].astype(jnp.float32)
    information = jnp.log2(4/jnp.maximum(hint_size,1))
    placement += 2.5*information[:,None]*novel[None,:]
    placement += jnp.where((hint_size==1)[:,None],0.8*jnp.minimum(free_neighbors,2)[None,:],0.)
    scores = jnp.concatenate((jnp.where(obs.legal_actions[1617:1624],reveal_score,-1e9), jnp.where(legal_place,placement,-1e9).reshape(-1)))
    actions = jnp.concatenate((jnp.arange(1617,1624),place_ids.reshape(-1)))
    return actions[jnp.argmax(scores)].astype(jnp.int32)


def solution_probability(obs, p, relations):
    certain = p.max(1)>0.999
    unknown = jnp.argsort(certain.astype(jnp.int32), stable=True)[:4]
    assignments = (jnp.arange(256)[:,None] // (4**jnp.arange(4))[None,:]) % 4
    candidate = jnp.broadcast_to(p.argmax(1)[None,:], (256,16))
    replacement = jnp.where(certain[unknown][None,:],candidate[:,unknown],assignments)
    candidate = candidate.at[:,unknown].set(replacement)
    color_counts = jax.nn.one_hot(candidate,4).sum(1)
    permitted = jnp.take_along_axis(jnp.broadcast_to(p[None,:,:],(256,16,4)),candidate[:,:,None],axis=2)[:,:,0]>0
    valid = jnp.all(color_counts==4,axis=1) & jnp.all(permitted,axis=1)
    same = candidate[:,:,None] == candidate[:,None,:]
    adjacency = obs.adjacency_matrix[:16,:16]>0
    graph = same & adjacency[None,:,:]
    def propagate(_, labels):
        return jnp.min(jnp.where(graph,labels[:,None,:],16),axis=-1)
    labels = jax.lax.fori_loop(0,3,propagate,jnp.broadcast_to(jnp.arange(16),(256,16)))
    solved = (labels==jnp.arange(16)[None,:]).sum(1)==4
    logweight = 1.1*jnp.sum(same*relations[None,:,:],axis=(1,2))
    weights = jnp.where(valid,jnp.exp(logweight-jnp.max(logweight)),0.)
    prob = jnp.sum(weights*solved)/jnp.maximum(jnp.sum(weights),1e-8)
    return jnp.where(jnp.sum(~certain)<=4,prob,0.)

def agent_act(obs, reset, memory, rng):
    memory = jax.tree.map(lambda z, init: jnp.where(reset,init,z),memory,init_agent())
    nf = obs.node_features
    active = nf[0,12]>0
    substep = nf[0,13].astype(jnp.int32)
    changed = jnp.any(obs.positions != memory['positions'],axis=1) & ~reset
    moved = jnp.where(changed,1.,memory['moved']*0.98)
    pos = obs.positions
    new_adj = jnp.abs(pos[:,None,:]-pos[None,:,:]).sum(-1)==1
    old_pos = memory['positions']
    old_adj = jnp.abs(old_pos[:,None,:]-old_pos[None,:,:]).sum(-1)==1
    partner_seen = nf[:16,11] > 0
    evidence = changed[:,None] & new_adj & ~old_adj & partner_seen[:,None] & partner_seen[None,:] & ~active
    weight = evidence.astype(jnp.float32)
    weight /= jnp.maximum(weight.sum(1,keepdims=True),1.)
    relations = jnp.maximum(memory['relations']*new_adj, weight+weight.T)
    colors,p = beliefs(obs,memory['colors'],relations)
    adj,dist,members,labels,size = graph_data(obs.positions,p)
    certain = p.max(1)>0.999
    solve_probability = solution_probability(obs,p,relations)
    solved = (jnp.all(certain) & jnp.all(size==4)) | (solve_probability>0.93)
    entropy = -jnp.sum(p*jnp.log(jnp.maximum(p,1e-8)),axis=1)
    degree = adj.sum(1)
    observation_score = 10*entropy + 0.6*moved + 0.15*(4-degree) + 0.1*nf[:16,11]
    observation_score += 0.01*jax.random.uniform(rng,(16,))
    observe = 1+jnp.argmax(jnp.where(obs.legal_actions[1:17],observation_score,-1e9))
    move,move_score = choose_move(obs,p,members,size)
    hint = choose_hint(obs,p,adj,size)
    action = jnp.where(substep<2,observe,jnp.where(substep==2,move,hint))
    action = jnp.where((substep==0)&solved,2324,action)
    action = jnp.where(active,action,0).astype(jnp.int32)
    next_memory = {'colors':colors,'positions':obs.positions,'moved':moved,'relations':relations}
    return action,next_memory,{'known':certain.sum(),'components':(labels==jnp.arange(16)).sum(),'solved':solved,'move_score':move_score,'beliefs':p,'solve_probability':solve_probability}
