"""Belief-aware local Yōkai policy with persistent private observations."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_cluster', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def assignments():
    rows = []
    triples = tuple((i,j,k) for i in range(9) for j in range(i+1,9) for k in range(j+1,9))
    for a in triples:
        for b in triples:
            if all(i not in a for i in b):
                rows.append(tuple(0 if i in a else 1 if i in b else 2 for i in range(9)))
    return jnp.asarray(rows, jnp.int32)


def init_agent():
    return {'colors': jnp.full((9,), -1, jnp.int32), 'steps': jnp.int32(0), 'partner_seen':jnp.zeros((9,),jnp.bool_), 'positions':jnp.zeros((9,2),jnp.int32), 'partner_moved':jnp.int32(-1)}


def beliefs(obs, colors):
    worlds = assignments()
    hard = jnp.all((colors[None, :] < 0) | (worlds == colors[None, :]), axis=1)
    hints = obs.node_features[9:, 3:6].astype(jnp.float32)
    attached = obs.adjacency_matrix[9:, :9].astype(jnp.float32)
    restrictions = attached.T @ (1-hints)
    mismatch = jnp.sum(jnp.take_along_axis(restrictions[None, :, :], worlds[:, :, None], axis=2)[:, :, 0], axis=1)
    weights = hard.astype(jnp.float32) * jnp.exp(-8.0*mismatch)
    weights /= jnp.maximum(weights.sum(), 1e-20)
    onehot = jax.nn.one_hot(worlds, 3)
    probs = jnp.einsum('w,wic->ic', weights, onehot)
    same = jnp.einsum('w,wij->ij', weights, (worlds[:,:,None] == worlds[:,None,:]).astype(jnp.float32))
    adj = obs.adjacency_matrix[:9,:9].astype(jnp.float32) - jnp.eye(9)
    edges = jnp.einsum('ij,wij->w', adj, (worlds[:,:,None] == worlds[:,None,:]).astype(jnp.float32))*0.5
    solved = jnp.sum(weights*(edges > 5.5))
    return probs, same, solved


def move_action(obs, same, probs, partner_moved):
    pos = obs.positions
    ids = jnp.arange(729)//81
    dest = jnp.stack(((jnp.arange(729)%81)//9, jnp.arange(729)%9), axis=1)
    distances = jnp.sum(jnp.abs(dest[:,None,:] - pos[None,:,:]), axis=2)
    old_dist = jnp.sum(jnp.abs(pos[:,None,:] - pos[None,:,:]), axis=2)
    keep = (ids[:,None] != jnp.arange(9)[None,:])
    match = same[ids] * keep
    edges = jnp.sum(match*(distances == 1),axis=1)
    old_edges = jnp.sum(match*(old_dist[ids] == 1),axis=1)
    dist_change = jnp.sum(match*(distances-old_dist[ids]),axis=1)
    neighbors = jnp.sum((distances==1)*keep,axis=1)
    old_neighbors = jnp.sum((old_dist[ids]==1)*keep,axis=1)
    score = 10*(edges-old_edges) - 0.22*dist_change + 0.06*(neighbors-old_neighbors)
    score -= 0.0001*jnp.sum((dest-4)**2,axis=1)
    score += 1.1*jnp.max(probs,axis=1)[ids] - 0.3*(ids==partner_moved)
    return jnp.argmax(jnp.where(obs.legal_actions[10:739], score, -1e9)).astype(jnp.int32)+10


def hint_action(obs, probs, same, partner_seen):
    features = obs.node_features
    hints = features[9:, 3:6].astype(jnp.float32)
    adj = obs.adjacency_matrix[:9,:9].astype(jnp.float32)-jnp.eye(9)
    stability = jnp.sum(same*adj,axis=1)
    match = hints @ probs.T
    score = 20*match + 2*stability[None,:]
    information = (1-partner_seen.astype(jnp.float32))[None,:] * (3-jnp.sum(hints,axis=1))[:,None]
    score += 0.5*information + 0.03*jnp.sum(adj,axis=1)[None,:]
    locked = features[:9,8].astype(jnp.float32)
    dist = jnp.sum(jnp.abs(obs.positions[:,None,:]-obs.positions[None,:,:]),axis=2)
    conflict = jnp.sum(same*locked[None,:]*(dist>2),axis=1)
    score -= 15*conflict[None,:]
    actions = 743 + jnp.arange(4)[:,None]*81 + (obs.positions[:,0]*9+obs.positions[:,1])[None,:]
    score = jnp.where(obs.legal_actions[actions], score, -1e9)
    placement = actions.reshape(-1)[jnp.argmax(score)]
    reveal = 739+jnp.argmax(obs.legal_actions[739:743]).astype(jnp.int32)
    safe_early = (match>0.999) & (stability[None,:]>0.95) & (conflict[None,:]<0.01) & obs.legal_actions[actions]
    early_score = jnp.where(safe_early,score,-1e9)
    early_placement = actions.reshape(-1)[jnp.argmax(early_score)]
    return jnp.where(jnp.any(safe_early),early_placement,jnp.where(jnp.any(obs.legal_actions[739:743]),reveal,placement))


def agent_act(obs, reset, memory, rng):
    memory = jax.tree.map(lambda x,y:jnp.where(reset,x,y),init_agent(),memory)
    visible = obs.node_features[:9,:3]
    colors = jnp.where(jnp.any(visible>0,axis=1),jnp.argmax(visible,axis=1),memory['colors']).astype(jnp.int32)
    partner_seen = memory['partner_seen'] | (obs.node_features[:9,9]>0)
    changed = jnp.any(obs.positions != memory['positions'],axis=1)
    partner_move_now = (obs.node_features[0,10]==0) & (obs.node_features[0,11]==3) & jnp.any(changed) & (~reset)
    partner_moved = jnp.where(partner_move_now,jnp.argmax(changed),memory['partner_moved']).astype(jnp.int32)
    probs,same,solved = beliefs(obs,colors)
    entropy = -jnp.sum(probs*jnp.log(jnp.maximum(probs,1e-8)),axis=1)
    adj = obs.adjacency_matrix[:9,:9].astype(jnp.float32)-jnp.eye(9)
    observation_score = entropy*(1-0.10*jnp.sum(adj,axis=1))
    observation_score += jax.random.uniform(rng,(9,))*0.001
    observe = 1+jnp.argmax(jnp.where(obs.legal_actions[1:10],observation_score,-1e9)).astype(jnp.int32)
    observe = jnp.where((solved>0.97)&obs.legal_actions[1067],1067,observe)
    step = obs.node_features[0,11].astype(jnp.int32)
    action = jax.lax.switch(step,(lambda:observe,lambda:observe,lambda:move_action(obs,same,probs,partner_moved),lambda:hint_action(obs,probs,same,partner_seen)))
    action = jnp.where(obs.node_features[0,10]>0,action,0).astype(jnp.int32)
    action = jnp.where(obs.legal_actions[action],action,jnp.argmax(obs.legal_actions)).astype(jnp.int32)
    return action, {'colors':colors,'steps':memory['steps']+1,'partner_seen':partner_seen,'positions':obs.positions,'partner_moved':partner_moved}, {'known':colors,'solved_probability':solved,'color_probabilities':probs}
