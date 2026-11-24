"""Local Bayesian color knowledge and geometry-based cooperative play."""
import jax
import jax.numpy as jnp
from custom_primitives import constants, partition_representatives

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'public_hint_knowledge', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return (jnp.zeros((9, 3), jnp.float32), jnp.zeros((9, 2), jnp.int32), jnp.zeros((1680,), jnp.float32))


def beliefs(obs, known, log_prior):
    COLORS, ONEHOT, PAIRS, TRIPLES, TRIPLE_SAME, CELLS, MOVERS, DESTS = constants()
    visible = jnp.sum(known, -1) > 0
    compatible = jnp.all((COLORS == jnp.argmax(known, -1)[None, :]) | ~visible[None, :], -1)
    hints = obs.node_features[9:, 3:6].astype(jnp.float32)
    attached = obs.adjacency_matrix[9:, :9].astype(jnp.float32)
    excluded = attached.T @ ((1-hints) * obs.node_features[9:, 8:9])
    violations = jnp.sum(ONEHOT * excluded[None, :, :], (1, 2))
    weights = compatible.astype(jnp.float32) * jnp.exp(log_prior-5.0 * violations)
    weights = weights / jnp.maximum(weights.sum(), 1e-20)
    probs = jnp.einsum('s,sic->ic', weights, ONEHOT)
    pair = jnp.einsum('s,sij->ij', weights, PAIRS)
    triple = weights @ TRIPLE_SAME
    return weights, probs, pair, triple


def geometry(positions, pair, triple):
    COLORS, ONEHOT, PAIRS, TRIPLES, TRIPLE_SAME, CELLS, MOVERS, DESTS = constants()
    dist = jnp.sum(jnp.abs(positions[..., :, None, :] - positions[..., None, :, :]), -1)
    edges = jnp.sum((dist == 1) * pair, (-2, -1)) * .5
    a = dist[..., TRIPLES[:, 0], TRIPLES[:, 1]]
    b = dist[..., TRIPLES[:, 0], TRIPLES[:, 2]]
    c = dist[..., TRIPLES[:, 1], TRIPLES[:, 2]]
    mst = jnp.sum((a+b+c-jnp.maximum(jnp.maximum(a,b),c)-2)*triple, -1)
    return 10*edges - mst, edges


def observation_value(obs, weights):
    COLORS, ONEHOT, PAIRS, TRIPLES, TRIPLE_SAME, CELLS, MOVERS, DESTS = constants()
    boards = jnp.broadcast_to(obs.positions, (729,9,2)).at[jnp.arange(729), MOVERS].set(DESTS)
    dist = jnp.sum(jnp.abs(boards[:,:,None,:]-boards[:,None,:,:]),-1)
    adjacent = dist==1
    reach = jnp.broadcast_to(jnp.arange(9)==0,(729,9))
    def connect(_, reach):
        return reach | jnp.any(adjacent & reach[:,None,:],-1)
    reach = jax.lax.fori_loop(0,8,connect,reach)
    no_collision = ~jnp.any((dist==0)&(~jnp.eye(9,dtype=jnp.bool_))[None,:,:],(-2,-1))
    moving = jnp.any(DESTS!=obs.positions[MOVERS],-1)
    legal = jnp.all(reach,-1)&no_collision&moving&(obs.node_features[MOVERS,8]==0)
    a=dist[:,TRIPLES[:,0],TRIPLES[:,1]]
    b=dist[:,TRIPLES[:,0],TRIPLES[:,2]]
    c=dist[:,TRIPLES[:,1],TRIPLES[:,2]]
    mst=a+b+c-jnp.maximum(jnp.maximum(a,b),c)-2
    joint = (weights[:,None,None]*ONEHOT).reshape((1680,27))
    joint_pair = joint.T @ PAIRS.reshape((1680,81))
    joint_triple = joint.T @ TRIPLE_SAME
    scores = 5*joint_pair @ adjacent.reshape((729,81)).T - joint_triple @ mst.T
    best = jnp.max(jnp.where(legal[None,:],scores,-1e6),-1).reshape((9,3))
    values = jnp.sum(best,-1)
    return values-jnp.min(values)


def partner_evidence(obs, previous_pos, seen):
    COLORS, ONEHOT, PAIRS, TRIPLES, TRIPLE_SAME, CELLS, MOVERS, DESTS = constants()
    reps, world_partition = partition_representatives()
    boards = jnp.broadcast_to(previous_pos,(729,9,2)).at[jnp.arange(729),MOVERS].set(DESTS)
    dist = jnp.sum(jnp.abs(boards[:,:,None,:]-boards[:,None,:,:]),-1)
    adj = dist==1
    reach = jnp.broadcast_to(jnp.arange(9)==0,(729,9))
    def connect(_, reach):
        return reach | jnp.any(adj & reach[:,None,:],-1)
    reach = jax.lax.fori_loop(0,8,connect,reach)
    legal = jnp.all(reach,-1)&(~jnp.any((dist==0)&(~jnp.eye(9,dtype=jnp.bool_))[None,:,:],(-2,-1)))
    legal &= jnp.any(DESTS!=previous_pos[MOVERS],-1)&(obs.node_features[MOVERS,8]==0)
    both = seen[:,None]*seen[None,:]
    same = PAIRS[reps]*both[None,:,:]
    attraction = 10*same-.4*(both[None,:,:]-same)+2*(1-both[None,:,:])
    all_scores = .5*(attraction.reshape((280,81)) @ adj.reshape((729,81)).T - .35*same.reshape((280,81)) @ dist.reshape((729,81)).T)
    best = jnp.max(jnp.where(legal[None,:],all_scores,-1e6),-1)
    now = jnp.sum(jnp.abs(obs.positions[:,None,:]-obs.positions[None,:,:]),-1)
    chosen = .5*jnp.sum(attraction*(now==1)[None,:,:]-.35*same*now[None,:,:],(-2,-1))
    regret = jnp.maximum(best-chosen,0)
    evidence = jnp.log(.1+.9*jnp.exp(-.5*regret))
    return evidence[world_partition]


def lookahead_move(obs, pair, triple, scores):
    COLORS, ONEHOT, PAIRS, TRIPLES, TRIPLE_SAME, CELLS, MOVERS, DESTS = constants()
    legal_scores = jnp.where(obs.legal_actions[10:739],scores,-1e6)
    _, top = jax.lax.top_k(legal_scores,6)
    by_card = jnp.argmax(legal_scores.reshape((9,81)),-1)+81*jnp.arange(9)
    candidates = jnp.concatenate([top,by_card])
    def consider(k, results):
        index = candidates[k]
        pos = obs.positions.at[MOVERS[index]].set(DESTS[index])
        boards = jnp.broadcast_to(pos,(729,9,2)).at[jnp.arange(729),MOVERS].set(DESTS)
        dist = jnp.sum(jnp.abs(boards[:,:,None,:]-boards[:,None,:,:]),-1)
        adj = dist==1
        reach = jnp.broadcast_to(jnp.arange(9)==0,(729,9))
        def connect(_, reach):
            return reach | jnp.any(adj & reach[:,None,:],-1)
        reach = jax.lax.fori_loop(0,8,connect,reach)
        legal = jnp.all(reach,-1)&(~jnp.any((dist==0)&(~jnp.eye(9,dtype=jnp.bool_))[None,:,:],(-2,-1)))
        legal &= jnp.any(DESTS!=pos[MOVERS],-1)&(obs.node_features[MOVERS,8]==0)
        future, _ = geometry(boards,pair,triple)
        best = jnp.max(jnp.where(legal,future,-1e6))
        value = .7*legal_scores[index] + best
        return results.at[k].set(value)
    values = jax.lax.fori_loop(0,15,consider,jnp.zeros((15,),jnp.float32))
    return 10+candidates[jnp.argmax(values)]


def agent_act(obs, reset, memory, rng):
    COLORS, ONEHOT, PAIRS, TRIPLES, TRIPLE_SAME, CELLS, MOVERS, DESTS = constants()
    memory = jax.tree.map(lambda x: jnp.where(reset, jnp.zeros_like(x), x), memory)
    previous_known, previous_pos, log_prior = memory
    colors = obs.node_features[:9, :3].astype(jnp.float32)
    known = jnp.where(jnp.sum(colors, -1, keepdims=True)>0, colors, previous_known)
    seen = obs.node_features[:9, 9].astype(jnp.float32)
    public_singletons = (jnp.sum(obs.node_features[9:,3:6],-1)==1).astype(jnp.float32)
    publicly_known = obs.adjacency_matrix[:9,9:].astype(jnp.float32) @ public_singletons
    model_seen = jnp.maximum(seen,publicly_known)
    old_dist = jnp.sum(jnp.abs(previous_pos[:, None, :] - previous_pos[None, :, :]), -1)
    new_dist = jnp.sum(jnp.abs(obs.positions[:, None, :] - obs.positions[None, :, :]), -1)
    edge_change = ((new_dist == 1).astype(jnp.float32) - (old_dist == 1).astype(jnp.float32)) * model_seen[:,None]*model_seen[None,:]
    evidence = jnp.sum(PAIRS * edge_change[None,:,:], (1,2)) * .375 + partner_evidence(obs, previous_pos, model_seen)
    update = (~reset) & (obs.node_features[0, 10] == 0) & (obs.node_features[0,11] == 3)
    log_prior = log_prior + jnp.where(update,evidence,0)
    log_prior = jnp.clip(log_prior-jnp.max(log_prior),-12,0)
    weights, probs, pair, triple = beliefs(obs, known, log_prior)
    pos = obs.positions
    dist = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), -1)
    edges_per_world = jnp.sum(PAIRS * (dist == 1)[None, :, :], (1,2)) * .5
    success = jnp.sum(weights * (edges_per_world == 6))
    entropy = -jnp.sum(probs*jnp.log(jnp.maximum(probs,1e-20)), -1)
    known_dist = jnp.min(jnp.where(jnp.sum(known,-1)[None,:]>0, dist, 20), -1)
    value = observation_value(obs,weights)
    observe_score = .25*value + entropy * (1 + .2*(1-seen)) + .01*jnp.minimum(known_dist, 6) - .001*jnp.arange(9)
    observe = 1+jnp.argmax(jnp.where(obs.legal_actions[1:10], observe_score, -1e9))
    moves = jnp.broadcast_to(pos, (729,9,2)).at[jnp.arange(729), MOVERS].set(DESTS)
    move_scores, _ = geometry(moves, pair, triple)
    move_scores -= .0001*jnp.sum(jnp.abs(DESTS-pos[MOVERS]), -1)
    move = 10+jnp.argmax(jnp.where(obs.legal_actions[10:739], move_scores, -1e9))
    planned_move = lookahead_move(obs,pair,triple,move_scores)
    use_plan = jnp.max(weights)>.15
    move = jnp.where(use_plan,planned_move,move)
    hint_sets = obs.node_features[9:, 3:6].astype(jnp.float32)
    truth = hint_sets @ probs.T
    same_neighbors = jnp.sum((dist == 1)*pair, -1)
    locks = obs.node_features[:9,8].astype(jnp.float32)
    far_lock = jnp.sum(pair*locks[None,:]*(dist>2),-1)
    info = (1-obs.node_features[:9,9].astype(jnp.float32))[None,:] / jnp.maximum(jnp.sum(hint_sets,-1)[:,None],1)
    hint_card_score = 20*truth + same_neighbors[None,:] + 2.0*info - 10*far_lock[None,:]
    safe_card = (same_neighbors>.9) & (far_lock<.1)
    no_color_anchor = jnp.sum(pair*locks[None,:],-1)<.1
    anchor = (jnp.sum(known,-1)>0) & (seen==0) & no_color_anchor & (jnp.sum(dist==1,-1)>=2)
    singleton = jnp.sum(hint_sets,-1)==1
    safe_hint = (truth>.99) & (safe_card[None,:] | (singleton[:,None]&anchor[None,:]))
    cell_cards = jnp.argmax(jnp.all(CELLS[:,None,:]==pos[None,:,:],-1),-1)
    placement_score = hint_card_score[:,cell_cards].reshape(-1)
    place = 743+jnp.argmax(jnp.where(obs.legal_actions[743:1067],placement_score,-1e9))
    reveal = 739+jnp.argmax(obs.legal_actions[739:743])
    safe_placements = obs.legal_actions[743:1067] & safe_hint[:,cell_cards].reshape(-1)
    safe_place = 743+jnp.argmax(jnp.where(safe_placements,placement_score,-1e9))
    hint = jnp.where(jnp.any(safe_placements),safe_place,jnp.where(jnp.any(obs.legal_actions[739:743]),reveal,place))
    step = obs.node_features[0,11]
    action = jnp.where(step<2,observe,jnp.where(step==2,move,hint))
    revealed = jnp.sum(obs.node_features[9:, 3:6],-1)>0
    placed = obs.node_features[9:,8]>0
    terminal_value = jnp.sum(jnp.where(placed,1,jnp.where(revealed,2,5)))
    next_cost = jnp.where(jnp.any(revealed & ~placed),1,3)
    hint_sizes = jnp.sum(hint_sets,-1)
    hinted_unknown = obs.adjacency_matrix[9:,:9].astype(jnp.float32) @ (1-seen)
    public_info = jnp.sum(hinted_unknown*jnp.where(hint_sizes==1,1.0,.37))
    information = jnp.sum(seen)+public_info
    extra_turn_probability = jnp.clip((6.5-information)/3.0,0,1)
    available_count = jnp.sum(revealed & ~placed)
    hidden_count = jnp.sum(~revealed)
    next_partner_cost = jnp.where((available_count==1)&(hidden_count>0),3,1)
    next_partner_cost = jnp.where((available_count==1)&(hidden_count==0),0,next_partner_cost)
    continuation = terminal_value-next_cost-extra_turn_probability*next_partner_cost
    threshold = .92*continuation/terminal_value
    action = jnp.where((step==0)&(success>threshold),1067,action)
    action = jnp.where(obs.node_features[0,10]>0,action,0).astype(jnp.int32)
    return action, (known, pos, log_prior), {'success':success, 'probs':probs, 'positions':pos, 'step':step, 'known':jnp.argmax(known,-1)*jnp.sum(known,-1) - (1-jnp.sum(known,-1))}
