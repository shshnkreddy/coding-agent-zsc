"""Recipe-grounded cooperation through visible ingredients in the shared pot."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'adaptive_ingredient_cooperation', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

# Immutable geometry templates localize directly from the visible static objects.
_LAYOUT = ('WWWWBXBWWWW', 'WWW0 A 1WWW', 'WWWWWPWWWWW', 'W    A    W', '0  W3R3W  0', 'W1WWWWWWW1W')
_TYPES = 'WXPRLB0123'
def locate(obs):
    cells = tuple((r,c) for r in range(6) for c in range(11) if _LAYOUT[r][c] in ' A')
    templates = tuple(tuple(tuple(tuple(int(0 <= r+i-2 < 6 and 0 <= c+j-2 < 11 and _LAYOUT[r+i-2][c+j-2] == ch) for ch in _TYPES) for j in range(5)) for i in range(5)) for r,c in cells)
    errors = jnp.sum(jnp.abs(jnp.asarray(templates, jnp.int32) - obs[None,:,:,22:32]), axis=(1,2,3))
    return jnp.asarray(cells, jnp.int32)[jnp.argmin(errors)]

def init_agent_0():
    return {'state':jnp.zeros((8,), jnp.int32), 'items':jnp.zeros((6,11,6), jnp.int32)}

def init_agent_1():
    return {'state':jnp.zeros((8,), jnp.int32), 'items':jnp.zeros((6,11,6), jnp.int32)}

def face_action(direction):
    return jnp.asarray([3,1,0,2], jnp.int32)[direction]

def go(pos, direction, row, col, face, interact=True):
    # Lower room horizontal travel uses its uninterrupted row-three corridor.
    cross = pos[1] != col
    action = jnp.where(cross, jnp.where(pos[1] < col, 0, 2), jnp.where(pos[0] < row, 1, 3))
    action = jnp.where(cross & (pos[0] == 4), 3, action)
    at = (pos[0] == row) & (pos[1] == col)
    last = jnp.where(direction == face, 5 if interact else 4, face_action(face))
    return jnp.where(at, last, action).astype(jnp.int32)

def read(obs):
    pos = locate(obs)
    direction = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:11]
    pot = jnp.sum(obs[:,:,32:38] * obs[:,:,24,None], axis=(0,1))
    partner = jnp.sum(obs[:,:,16:22] * obs[:,:,11,None], axis=(0,1))
    recipe = jnp.sum(obs[:,:,40:44], axis=(0,1))
    return pos,direction,inv,pot,partner,recipe

def remember(obs, pos, memory):
    rows = jnp.arange(6)[:,None] - pos[0] + 2
    cols = jnp.arange(11)[None,:] - pos[1] + 2
    seen = (rows>=0) & (rows<5) & (cols>=0) & (cols<5)
    items = jnp.where(seen[:,:,None],obs[jnp.clip(rows,0,4),jnp.clip(cols,0,4),32:38],memory['items'])
    return items

def counter_options(pos, direction, items, seat):
    cells = tuple((r,c) for r in range(6) for c in range(11) if _LAYOUT[r][c] in ' A' and ((r==1) if seat==0 else (r>=3)))
    directions = ((-1,0),(1,0),(0,1),(0,-1))
    options = tuple((r,c,d,r+dr,c+dc) for r,c in cells for d,(dr,dc) in enumerate(directions) if 0<=r+dr<6 and 0<=c+dc<11 and _LAYOUT[r+dr][c+dc]=='W')
    targets = jnp.asarray(options,jnp.int32)
    distance = jnp.abs(pos[1]-targets[:,1]) + jnp.where(pos[1]==targets[:,1],jnp.abs(pos[0]-targets[:,0]),pos[0]+targets[:,0]-(2 if seat==0 else 6))
    at = (pos[0]==targets[:,0]) & (pos[1]==targets[:,1])
    costs = distance + 1 + jnp.where(at, direction!=targets[:,2],1)
    contents = items[targets[:,3],targets[:,4]]
    return targets,costs,contents

def choose_counter(pos,direction,targets,costs,mask):
    idx = jnp.argmin(jnp.where(mask,costs,10000))
    target=targets[idx]
    return go(pos,direction,target[0],target[1],target[2]),jnp.any(mask),jnp.min(jnp.where(mask,costs,10000))

def agent_0_act(obs, reset, memory, rng):
    state = jnp.where(reset,jnp.zeros_like(memory['state']),memory['state'])
    memory = {'state':state,'items':jnp.where(reset,jnp.zeros_like(memory['items']),memory['items'])}
    pos,direction,inv,pot,partner,recipe = read(obs)
    items = remember(obs,pos,memory)
    targets,costs,contents = counter_options(pos,direction,items,0)
    count = jnp.sum(pot[2:])
    delivered = (state[4]>0) & (inv[1]==0)
    age = jnp.where(delivered,0,state[3]+1)
    partner_index = jnp.argmax(obs[:,:,11])
    partner_row = partner_index//5 + pos[0]-2
    partner_col = partner_index%5 + pos[1]-2
    partner_dir = jnp.argmax(jnp.sum(obs[:,:,12:16],axis=(0,1)))
    toward_pot = (partner_row==3) & (((partner_col==5)&(partner_dir==0)) | ((partner_col==4)&(partner_dir==2)) | ((partner_col==6)&(partner_dir==3)))
    central_raw = (partner_row==3) & (partner_col==5) & (jnp.sum(partner[2:4])==1) & (partner[0]==0)
    raw_code = 1+jnp.argmax(partner[2:4])
    dwell = jnp.where(central_raw,jnp.where((state[6]==raw_code)&~delivered,state[5]+1,1),0)
    toward_pot = toward_pot | (dwell>=3)
    state = state.at[5].set(dwell).at[6].set(raw_code)
    fresh_raw = (jnp.sum(partner[2:4])==1) & (partner[0]==0) & toward_pot & (age>=3) & (inv[1]==0)
    known = ((state[1]>0)&~delivered) | (count>0) | fresh_raw
    kind = jnp.where(count>0,jnp.argmax(pot[2:]),jnp.where(fresh_raw,jnp.argmax(partner[2:4]),state[2]))
    state = state.at[1].set(known.astype(jnp.int32)).at[2].set(kind).at[3].set(age).at[4].set(inv[1])
    raw = jnp.sum(inv[2:]) > 0
    empty = jnp.sum(inv) == 0
    side = jnp.where(pos[1] <= 5, 4, 6)
    to_pot = go(pos,direction,1,5,1)
    to_plate = go(pos,direction,1,side,0)
    to_supply = go(pos,direction,1,jnp.where(kind==0,4,6),jnp.where(kind==0,3,2))
    cached,has_cached,cache_cost = choose_counter(pos,direction,targets,costs,(contents[:,2+kind]>0)&(contents[:,0]==0)&(contents[:,1]==0))
    supply_col = jnp.where(kind==0,4,6)
    supply_dir = jnp.where(kind==0,3,2)
    supply_cost = jnp.abs(pos[1]-supply_col) + 1 + jnp.where(pos[1]==supply_col,direction!=supply_dir,0)
    to_supply = jnp.where(has_cached & (cache_cost<=supply_cost),cached,to_supply)
    drop,can_drop,_ = choose_counter(pos,direction,targets,costs,jnp.sum(contents,axis=1)==0)
    wait = go(pos,direction,1,5,1,False)
    drop = jnp.where(can_drop,drop,wait)
    stock = jnp.any((contents[:,2:4]>0) & (contents[:,0:1]==0) & (contents[:,1:2]==0),axis=0)
    stock_kind = jnp.where(stock[0],1,0)
    timer = jnp.sum(obs[:,:,44])
    stage = can_drop & ~jnp.all(stock) & ((count==0) & ~known | ((count==3) & (timer>=12)))
    stock_action = go(pos,direction,1,jnp.where(stock_kind==0,4,6),jnp.where(stock_kind==0,3,2))
    action = jnp.where(empty & stage,stock_action,wait)
    action = jnp.where(empty & known & (count<3),to_supply,action)
    action = jnp.where(empty & (count>=3) & ~stage,to_plate,action)
    action = jnp.where(raw & (inv[0]==0),jnp.where(known & (count<3) & (inv[2+kind]>0),to_pot,drop),action)
    action = jnp.where((inv[0]>0) & (inv[1]==0),jnp.where(count<3,drop,to_pot),action)
    action = jnp.where(inv[1]>0,go(pos,direction,1,5,0),action)
    memory = {'state':state.at[0].add(1),'items':items}
    return action,memory,{'pos':pos,'pot_n':count,'kind':kind,'held':inv,'known':known,'fresh_raw':fresh_raw}

def agent_1_act(obs, reset, memory, rng):
    state = jnp.where(reset,jnp.zeros_like(memory['state']),memory['state'])
    memory = {'state':state,'items':jnp.where(reset,jnp.zeros_like(memory['items']),memory['items'])}
    pos,direction,inv,pot,partner,recipe = read(obs)
    items = remember(obs,pos,memory)
    targets,costs,contents = counter_options(pos,direction,items,1)
    visible = jnp.sum(recipe)>0
    known = visible | (state[1]>0)
    kind = jnp.where(visible,jnp.argmax(recipe),state[2])
    count = jnp.sum(pot[2:])
    pot_visible = jnp.sum(obs[:,:,24])>0
    state = state.at[1].set(known.astype(jnp.int32)).at[2].set(kind)
    side = jnp.where(pos[1] <= 5,1,9)
    supply_face = jnp.where(kind==0,jnp.where(side==1,3,2),1)
    to_supply = go(pos,direction,4,side,supply_face)
    cached,has_cached,cache_cost = choose_counter(pos,direction,targets,costs,(contents[:,2+kind]>0)&(contents[:,0]==0)&(contents[:,1]==0))
    supply_cost = jnp.abs(pos[1]-side) + (4-pos[0]) + 2
    to_supply = jnp.where(has_cached & (cache_cost<=supply_cost),cached,to_supply)
    to_pot = go(pos,direction,3,5,0)
    wait = go(pos,direction,3,5,0,False)
    drop,can_drop,_ = choose_counter(pos,direction,targets,costs,jnp.sum(contents,axis=1)==0)
    drop = jnp.where(can_drop,drop,wait)
    raw = jnp.sum(inv[2:])>0
    empty = jnp.sum(inv)==0
    busy = (count>=3) | (partner[1]>0)
    timer = jnp.sum(obs[:,:,44])
    # Attribute observed pot increases beyond our own preceding insertion to
    # the partner. Prefetch is especially valuable when we do all the cooking.
    assisted = jnp.where(pot_visible & (count==0),False,(state[5]>0) | (pot_visible & (count>state[4]+state[7])))
    prefetch = (count>=3) & (timer>=14) & ~assisted
    committed = empty & known & (prefetch | (state[3]>0))
    fetching = empty & known & (~pot_visible | ~busy | committed)
    state = state.at[3].set(committed.astype(jnp.int32)).at[4].set(jnp.where(pot_visible,count,state[4])).at[5].set(assisted.astype(jnp.int32))
    action = jnp.where(fetching,to_supply,wait)
    action = jnp.where(raw,jnp.where(inv[2+kind]>0,jnp.where(busy,wait,to_pot),drop),action)
    action = jnp.where(inv[0]>0,drop,action)
    put_attempt = (action==5) & (pos[0]==3) & (pos[1]==5) & (direction==0) & raw & (count<3)
    state = state.at[7].set(put_attempt.astype(jnp.int32))
    memory = {'state':state.at[0].add(1),'items':items}
    return action,memory,{'pos':pos,'pot_n':count,'kind':kind,'known':known,'held':inv,'assisted':assisted,'prefetch':committed}
