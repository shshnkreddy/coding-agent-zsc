"""Recipe-grounded cooperation through public pot and inventory observations."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'persistent_grounded_recipe', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

_BOARD = ('WWWWWR2W0WW', '0      W  B', 'W     APA X', '1      W  B', 'WWWWWR2W1WW')
_CODES = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'L': 4, 'B': 5, '0': 6, '1': 7, '2': 8}


def localize(obs):
    _POSITIONS = tuple((r,c) for r in range(1,4) for c in range(1,10) if _BOARD[r][c] in ' A')
    _PATCHES = tuple(tuple(tuple(tuple(int(0 <= r+i-2 < 5 and 0 <= c+j-2 < 11 and _CODES.get(_BOARD[r+i-2][c+j-2], -1) == k) for k in range(9)) for j in range(5)) for i in range(5)) for r,c in _POSITIONS)

    costs = jnp.sum(jnp.abs(jnp.asarray(_PATCHES, jnp.int32) - obs[None,:,:,20:29]), axis=(1,2,3))
    return jnp.asarray(_POSITIONS, jnp.int32)[jnp.argmin(costs)]


def walk(pos, goal):
    r,c = pos
    tr,tc = goal
    return jnp.where(c<tc,0,jnp.where(c>tc,2,jnp.where(r<tr,1,jnp.where(r>tr,3,4)))).astype(jnp.int32)


def use(pos, facing, stand, direction):
    movement = walk(pos, stand)
    finish_horizontal = ((direction==2)&(pos[1]<stand[1])) | ((direction==3)&(pos[1]>stand[1]))
    vertical_first = jnp.where(pos[0]<stand[0],1,jnp.where(pos[0]>stand[0],3,movement))
    movement = jnp.where(finish_horizontal,vertical_first,movement)
    turn = jnp.asarray([3,1,0,2],jnp.int32)[direction]
    return jnp.where(jnp.all(pos==stand), jnp.where(facing==direction,5,turn), movement).astype(jnp.int32)


def counter_action(obs, pos, facing, ingredient, dropping, role):
    if role == 0:
        # Counter row, column, adjacent floor row, column, facing.
        sites = jnp.asarray([(0,c,1,c,0) for c in range(1,5)] + [(4,c,3,c,1) for c in range(1,5)] + [(2,0,2,1,3),(1,7,1,6,2),(3,7,3,6,2)],jnp.int32)
    else:
        sites = jnp.asarray([(0,9,1,9,0),(4,9,3,9,1),(1,7,1,8,3),(3,7,3,8,3)],jnp.int32)
    cells = sites[:,:2]-pos[None,:]+2
    visible = jnp.all((cells>=0)&(cells<5),axis=1)
    items = obs[jnp.clip(cells[:,0],0,4),jnp.clip(cells[:,1],0,4),29:34]
    empty = jnp.sum(items,axis=1)==0
    useful = (items[:,2+jnp.maximum(ingredient,0)]>0)&(items[:,0]==0)&(items[:,1]==0)
    eligible = visible & jnp.where(dropping,empty,useful)
    if role == 0:
        # Obsolete food is cached away from the partner's view, not on a handoff counter.
        eligible = eligible & ((~jnp.asarray(dropping)) | (sites[:,1]!=7))
    distance = jnp.sum(jnp.abs(sites[:,2:4]-pos[None,:]),axis=1)
    cost = distance + jnp.where(jnp.all(sites[:,2:4]==pos,axis=1)&(sites[:,4]==facing),0,1)
    index = jnp.argmin(jnp.where(eligible,cost,1000))
    site = sites[index]
    action = use(pos,facing,site[2:4],site[4])
    return action,jnp.any(eligible),jnp.where(jnp.any(eligible),cost[index]+1,1000)


def init_agent_0():
    # current recipe, previous visible pot count, clearance delay, time
    return jnp.asarray([-1,0,0,0],jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    memory = jnp.where(reset,init_agent_0(),memory)
    pos = localize(obs)
    facing = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:10]
    held = jnp.argmax(inv[2:])
    raw = (jnp.sum(inv[2:])>0)&(inv[1]==0)
    recipe_counts = jnp.max(obs[:,:,36:39],axis=(0,1))
    recipe = jnp.where(jnp.sum(recipe_counts)>0,jnp.argmax(recipe_counts),memory[0])
    pot_mask = obs[:,:,22]
    visible = jnp.sum(pot_mask)>0
    pot = jnp.sum(obs[:,:,29:34]*pot_mask[:,:,None],axis=(0,1))
    count = jnp.sum(pot[2:])
    prior = jnp.where(visible,count,memory[1])
    delay = jnp.maximum(memory[2]-1,0)
    partner = jnp.sum(obs[:,:,15:20],axis=(0,1))
    partner_visible = jnp.sum(obs[:,:,10])>0
    delay = jnp.where(visible&(count==0)&(memory[1]>0),5,delay)
    delay = jnp.where((count==0)&(partner[1]>0),5,delay)
    confirmed = ((recipe!=memory[0])&(recipe>=0)) | (partner_visible&(jnp.sum(partner)==0))
    delay = jnp.where(confirmed,0,delay)
    cooking = jnp.sum(obs[:,:,39])>0
    full = (count>=3)|cooking|(pot[1]>0)
    idle = walk(pos,jnp.asarray([2,6]))
    source_stand = jnp.asarray([jnp.where(recipe==0,1,3),1])
    fetch = use(pos,facing,source_stand,jnp.asarray(3))
    put = use(pos,facing,jnp.asarray([2,6]),jnp.asarray(2))
    discard, can_drop, _ = counter_action(obs,pos,facing,recipe,True,0)
    discard = jnp.where(can_drop,discard,walk(pos,jnp.asarray([2,3])))
    reuse, can_reuse, reuse_cost = counter_action(obs,pos,facing,recipe,False,0)
    source_cost = jnp.sum(jnp.abs(pos-source_stand))+2
    fetch = jnp.where(can_reuse&(reuse_cost<=source_cost),reuse,fetch)
    action = jnp.where((recipe>=0)&(delay==0)&(~visible|~full),fetch,idle)
    action = jnp.where(raw,jnp.where(held!=recipe,discard,jnp.where((delay>0)|full,idle,put)),action)
    memory = jnp.asarray([recipe,prior,delay,memory[3]+1],jnp.int32)
    return action,memory,{'pos':pos,'recipe':recipe,'pot_count':count,'delay':delay,'inventory':inv,'partner':partner,'pot':pot,'timer':jnp.sum(obs[:,:,39])}


def init_agent_1():
    # recipe, time, delivery cooldown, previous dish, partner side, stability,
    # empty-pot age, evidence strength, counter-query phase
    return jnp.asarray([-1,0,0,0,-1,0,0,0,0],jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    memory = jnp.where(reset,init_agent_1(),memory)
    pos = localize(obs)
    facing = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:10]
    raw = (jnp.sum(inv[2:])>0)&(inv[1]==0)
    held = jnp.argmax(inv[2:])
    mask = obs[:,:,22]
    pot = jnp.sum(obs[:,:,29:34]*mask[:,:,None],axis=(0,1))
    count = jnp.sum(pot[2:])
    partner = jnp.sum(obs[:,:,15:20],axis=(0,1))
    partner_raw = (jnp.sum(partner[2:4])==1)&(partner[1]==0)&(partner[4]==0)
    partner_mask = obs[:,:,10]
    partner_visible = jnp.sum(partner_mask)>0
    local_partner = jnp.asarray([jnp.argmax(jnp.sum(partner_mask,axis=1)),jnp.argmax(jnp.sum(partner_mask,axis=0))])
    partner_pos = pos+local_partner-2
    side = jnp.where(partner_visible&(partner_pos[1]==6),jnp.where(partner_pos[0]==1,0,jnp.where(partner_pos[0]==3,1,2)),-1)
    delivered = (memory[3]>0)&(inv[1]==0)
    cooldown = jnp.maximum(memory[2]-1,0)
    cooldown = jnp.where(delivered,6,cooldown)
    stable = jnp.where((side>=0)&(side==memory[4])&(~delivered),memory[5]+1,0)
    empty_age = jnp.where((count==0)&(inv[1]==0)&(~delivered),memory[6]+1,0)
    prior_recipe = jnp.where(delivered,-1,memory[0])
    prior_kind = jnp.where(delivered,0,memory[7])
    physical = partner_raw&(cooldown==0)
    settled = (stable>=6)&(empty_age>=6)&(cooldown==0)
    spatial = settled & (side<2)
    shared_cells = jnp.asarray([[1,7],[3,7]])-pos[None,:]+2
    shared_items = obs[shared_cells[:,0],shared_cells[:,1],29:34]
    shared_counts = jnp.sum(jnp.where((shared_items[:,0:1]==0)&(shared_items[:,1:2]==0),shared_items[:,2:5],0),axis=0)
    unambiguous = (jnp.sum(shared_counts[:2]>0)==1)&(shared_counts[2]==0)
    handoff = settled & unambiguous & (prior_kind<2)
    # Ingredient 2 is never a requested food here: a single stationary marker
    # can only support the weaker geographic interpretation, never pot filling.
    marker = (shared_items[:,4]>0)&(shared_items[:,0]==0)&(shared_items[:,1]==0)
    marker_signal = settled & (jnp.sum(marker)==1) & (jnp.sum(shared_counts[:2])==0)
    geographic = jnp.where(marker_signal,jnp.argmax(marker),side)
    spatial = spatial | marker_signal
    guess = jnp.where(spatial,geographic,prior_recipe)
    guess_kind = jnp.where(spatial,1,prior_kind)
    # Retained physical evidence is stronger than a position heuristic.
    guess = jnp.where(prior_kind>=2,prior_recipe,guess)
    guess_kind = jnp.where(prior_kind>=2,prior_kind,guess_kind)
    guess = jnp.where(handoff,jnp.argmax(shared_counts[:2]),guess)
    guess_kind = jnp.where(handoff,2,guess_kind)
    guess = jnp.where(physical,jnp.argmax(partner[2:4]),guess)
    guess_kind = jnp.where(physical,2,guess_kind)
    pot_recipe = jnp.where(jnp.sum(pot[2:4])>0,jnp.argmax(pot[2:4]),jnp.maximum(guess,0))
    recipe = jnp.where(count>0,pot_recipe,guess)
    evidence_kind = jnp.where(count>0,3,guess_kind)
    ambiguous = settled & (recipe<0)&(jnp.sum(shared_counts[:2]>0)==2)&(jnp.sum(inv)==0)
    query = jnp.where(ambiguous,1,memory[8])
    query = jnp.where((query==1)&raw,2,query)
    query = jnp.where((query==1)&(jnp.sum(shared_items[0,2:])==0),0,query)
    query = jnp.where((recipe>=0)|delivered,0,query)
    full = count>=3
    cooking = jnp.sum(obs[:,:,39])>0
    ready = pot[1]>0
    source = use(pos,facing,jnp.asarray([jnp.where(recipe==0,1,3),8]),jnp.where(recipe==0,0,1))
    atpot = use(pos,facing,jnp.asarray([2,8]),jnp.asarray(3))
    plates = use(pos,facing,jnp.asarray([jnp.where(pos[0]<=2,1,3),9]),jnp.asarray(2))
    deliver = use(pos,facing,jnp.asarray([2,9]),jnp.asarray(2))
    idle = walk(pos,jnp.asarray([2,8]))
    discard, can_drop, _ = counter_action(obs,pos,facing,recipe,True,1)
    reuse, can_reuse, reuse_cost = counter_action(obs,pos,facing,recipe,False,1)
    source_cost = jnp.abs(pos[0]-jnp.where(recipe==0,1,3))+jnp.abs(pos[1]-8)+2
    source = jnp.where(can_reuse,reuse,source)
    action = jnp.where(full|cooking|ready,plates,jnp.where(recipe>=0,source,idle))
    action = jnp.where(raw,jnp.where((recipe>=0)&(held!=recipe),discard,jnp.where(full|cooking|ready,discard,jnp.where(recipe>=0,atpot,idle))),action)
    plate_wait = jnp.where(jnp.all(pos==jnp.asarray([2,8])),jnp.where(facing==3,4,2),idle)
    action = jnp.where(inv[0]>0,jnp.where(inv[1]>0,deliver,jnp.where(ready,atpot,plate_wait)),action)
    query_action=use(pos,facing,jnp.asarray([1,8]),jnp.asarray(3))
    action=jnp.where(query==1,query_action,action)
    stable=jnp.where(query==1,0,stable)
    invalid_dish=(inv[1]>0)&((inv[4]>0)|((inv[2]>0)&(inv[3]>0)))
    action=jnp.where(invalid_dish&can_drop,discard,action)
    return action,jnp.asarray([recipe,memory[1]+1,cooldown,inv[1],side,stable,empty_age,evidence_kind,query],jnp.int32),{'pos':pos,'recipe':recipe,'pot_count':count,'inventory':inv,'partner':partner,'pot':pot,'timer':jnp.sum(obs[:,:,39]),'evidence':evidence_kind,'stable':stable,'query':query,'shared':shared_items}
