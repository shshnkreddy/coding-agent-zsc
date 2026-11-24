"""Coordinate through visible food and pot state across the separated rooms."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'observable_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

_LAYOUT = ('WWWWWR2W0WW', '0      W  B', 'W     APA X', '1      W  B', 'WWWWWR2W1WW')
_OBJECTS = 'WXPRLB012'
def _templates(positions):
    return tuple(tuple(tuple(tuple(int(0 <= r+i-2 < 5 and 0 <= c+j-2 < 11 and _LAYOUT[r+i-2][c+j-2] == obj) for obj in _OBJECTS) for j in range(5)) for i in range(5)) for r,c in positions)

def locate(obs, seat):
    coords = tuple((r,c) for r in range(1,4) for c in (range(1,7) if seat == 0 else range(8,10)))
    positions = jnp.asarray(coords, jnp.int32)
    templates = jnp.asarray(_templates(coords), jnp.int32)
    error = jnp.sum(jnp.abs(templates - obs[None,:,:,20:29]), axis=(1,2,3))
    return positions[jnp.argmin(error)]

def go(pos, direction, row, col, facing, interact=True):
    move = jnp.where(pos[1] < col, 0, jnp.where(pos[1] > col, 2, jnp.where(pos[0] < row, 1, 3)))
    at = (pos[0] == row) & (pos[1] == col)
    turn = jnp.asarray([3,1,0,2], jnp.int32)[facing]
    return jnp.where(at, jnp.where(direction == facing, 5 if interact else 4, turn), move).astype(jnp.int32)

def cell(obs, pos, row, col):
    r, c = row - pos[0] + 2, col - pos[1] + 2
    visible = (r >= 0) & (r < 5) & (c >= 0) & (c < 5)
    return jnp.where(visible, obs[jnp.clip(r,0,4),jnp.clip(c,0,4)], 0), visible

def counters_for(seat):
    if seat == 0:
        return jnp.asarray(((1,7,1,6,2),(3,7,3,6,2),
                            (0,4,1,4,0),(4,4,3,4,1),
                            (0,3,1,3,0),(4,3,3,3,1),
                            (0,2,1,2,0),(4,2,3,2,1),
                            (0,1,1,1,0),(4,1,3,1,1),(2,0,2,1,3)),jnp.int32)
    return jnp.asarray(((0,9,1,9,0),(4,9,3,9,1),(1,7,1,8,3),(3,7,3,8,3)),jnp.int32)

def observe_counters(obs,pos,coords,previous):
    rows = coords[:,0]-pos[0]+2
    cols = coords[:,1]-pos[1]+2
    visible = (rows>=0)&(rows<5)&(cols>=0)&(cols<5)
    return jnp.where(visible[:,None],obs[jnp.clip(rows,0,4),jnp.clip(cols,0,4),29:34],previous)

def counter_action(pos,direction,coords,index):
    return go(pos,direction,coords[index,2],coords[index,3],coords[index,4])

def init_agent_0():
    return {'state':jnp.asarray([-1,0,0,0], jnp.int32),
            'counters':jnp.zeros((11,5), jnp.int32)}

def agent_0_act(obs, reset, memory, rng):
    m = jnp.where(reset, init_agent_0()['state'], memory['state'])
    old_counters = jnp.where(reset, 0, memory['counters'])
    pos = locate(obs, 0)
    direction = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:10]
    recipe = jnp.max(obs[:,:,36:39], axis=(0,1))
    target = jnp.where(jnp.sum(recipe)>0, jnp.argmax(recipe), m[0])
    pot, visible = cell(obs,pos,2,7)
    total = jnp.sum(pot[31:34])
    count = jnp.where(visible,total,m[1])
    idle = jnp.where(visible & (count != m[1]),0,m[2]+1)
    scooped = visible & (m[1] == 3) & (total == 0)
    partner = jnp.sum(obs[:,:,15:20],axis=(0,1))
    partner_visible = jnp.any(obs[:,:,10]>0)
    delivery_evidence = ((target != m[0]) & (m[0]>=0)) | (partner_visible & (partner[1]==0))
    delay = jnp.where(scooped | (partner_visible & (partner[1]>0)), 8, jnp.maximum(m[3]-1,0))
    delay = jnp.where(delivery_evidence,0,delay)
    coords = counters_for(0)
    counters = observe_counters(obs,pos,coords,old_counters)
    distances = jnp.abs(coords[:,2]-pos[0])+jnp.abs(coords[:,3]-pos[1])
    carry = jnp.sum(inv[2:]) > 0
    kind = jnp.argmax(inv[2:])
    correct = kind == target
    empty_slots = jnp.sum(counters,axis=1)==0
    available = (counters[:,0]==0) & (counters[:,1]==0)
    stock = jnp.sum(counters[:,2:] * available[:,None],axis=0)
    matching = available & (counters[:,2+jnp.maximum(target,0)]>0)
    cache_index = jnp.argmin(jnp.where(matching,distances,100))
    wait = go(pos,direction,2,6,2,False)
    add = go(pos,direction,2,6,2)
    needed = (count == 0) | ((count < 3) & (idle >= 10))
    needed = needed & (delay==0) & (target>=0)
    missing = jnp.argmin(stock[:2])
    fetch_type = jnp.where(needed,target,missing)
    source = go(pos,direction,jnp.where(fetch_type == 0,1,3),1,3)
    pickup = counter_action(pos,direction,coords,cache_index)
    # Any matching raw food is a usable stock item, regardless of its shelf.
    fetch = jnp.where(needed & jnp.any(matching),pickup,source)
    restock = (target>=0) & jnp.any(empty_slots) & jnp.any(stock[:2]==0)
    action = jnp.where(needed | restock,fetch,wait)
    slot = jnp.argmin(jnp.where(empty_slots,distances+jnp.where(jnp.arange(11)<2,0,8),100))
    putaway = counter_action(pos,direction,coords,slot)
    action = jnp.where(carry, jnp.where(correct & (count<3) & (delay==0),add,
                        jnp.where(jnp.any(empty_slots),putaway,wait)),action)
    state = jnp.stack([target,count,idle,delay]).astype(jnp.int32)
    return action, {'state':state,'counters':counters}, {'pos':pos,'target':target,'pot_count':count,'delay':delay,'stock':stock}

def init_agent_1():
    return {'state':jnp.asarray([-1,0,0,0,0,-1,-1,0],jnp.int32),'counters':jnp.zeros((4,5),jnp.int32)}

def agent_1_act(obs, reset, memory, rng):
    m = jnp.where(reset,init_agent_1()['state'],memory['state'])
    old_counters = jnp.where(reset,0,memory['counters'])
    pos = locate(obs,1)
    direction = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:10]
    pot, visible = cell(obs,pos,2,7)
    total = jnp.sum(pot[31:34])
    partner, vp = cell(obs,pos,2,6)
    offering = vp & (partner[10]>0) & (partner[13]>0) & (partner[15]==0) & (jnp.sum(partner[17:20])==1)
    delivered = (m[1]>0) & (inv[1]==0)
    all_partner = jnp.sum(obs[:,:,15:20],axis=(0,1))
    raw_seen = (all_partner[0]==0)&(all_partner[1]==0)&(jnp.sum(all_partner[2:])==1)
    kind_seen = jnp.argmax(all_partner[2:])
    observing = (total==0)&(inv[1]==0)&raw_seen
    votes0 = jnp.where(delivered | (total>0),0,m[2])
    votes1 = jnp.where(delivered | (total>0),0,m[3])
    votes0 = jnp.minimum(20,votes0+jnp.asarray(observing & (kind_seen==0),jnp.int32))
    votes1 = jnp.minimum(20,votes1+jnp.asarray(observing & (kind_seen==1),jnp.int32))
    empty_age = jnp.where((total>0)|delivered| (inv[1]>0),0,m[4]+1)
    confident = (jnp.maximum(votes0,votes1)>=3)&(jnp.abs(votes0-votes1)>=2)&(empty_age>=8)
    belief = jnp.where(delivered,-1,m[0])
    inferred = jnp.where(votes0>=votes1,0,1)
    belief = jnp.where(confident,inferred,belief)
    target = jnp.where(total>0,jnp.argmax(pot[31:34]),jnp.where(offering,jnp.argmax(partner[17:20]),belief))
    raw = (inv[0]==0) & (jnp.sum(inv[2:])>0)
    plate = (inv[0]>0) & (inv[1]==0)
    dish = inv[1]>0
    empty = jnp.sum(inv)==0
    wait = go(pos,direction,2,8,3,False)
    fetch = go(pos,direction,jnp.where(target==0,1,3),8,jnp.where(target==0,0,1))
    plate_row = jnp.where(pos[0]<=2,1,3)
    getplate = go(pos,direction,plate_row,9,2)
    potaction = go(pos,direction,2,8,3)
    deliver = go(pos,direction,2,9,2)
    coords = counters_for(1)
    items = observe_counters(obs,pos,coords,old_counters)
    frees = jnp.sum(items,axis=1)==0
    distance = jnp.abs(coords[:,2]-pos[0])+jnp.abs(coords[:,3]-pos[1])
    slot = jnp.argmin(jnp.where(frees,distance+jnp.where(jnp.arange(4)<2,0,5),100))
    park = counter_action(pos,direction,coords,slot)
    matching = (items[:,0]==0) & (items[:,1]==0) & (items[:,2+jnp.maximum(target,0)]>0)
    take_slot = jnp.argmin(jnp.where(matching,distance,100))
    retrieve = counter_action(pos,direction,coords,take_slot)
    stalled = jnp.where(empty & (target>=0) & (total<3) & (pos[0]==m[5]) & (pos[1]==m[6]),m[7]+1,0)
    source_ready = (stalled>=2) & (pos[1]==8) & (pos[0]==jnp.where(target==0,1,3)) & (direction==jnp.where(target==0,0,1))
    fetch = jnp.where(jnp.any(matching) & ~source_ready,retrieve,fetch)
    action = wait
    action = jnp.where(empty & (target>=0) & (total<3),fetch,action)
    action = jnp.where(empty & (total==3),getplate,action)
    surplus = (total==3) | (target<0) | ((target>=0) & (jnp.argmax(inv[2:])!=target))
    action = jnp.where(raw,jnp.where(surplus,park,jnp.where(target>=0,potaction,wait)),action)
    action = jnp.where(plate,potaction,action)
    action = jnp.where(dish,deliver,action)
    return action, {'state':jnp.asarray([target,inv[1],votes0,votes1,empty_age,pos[0],pos[1],stalled],jnp.int32),'counters':items}, {'pos':pos,'target':target,'pot_count':total,'surplus':surplus,'inv':inv,'timer':pot[39],'votes':jnp.asarray([votes0,votes1],jnp.int32),'empty_age':empty_age}
