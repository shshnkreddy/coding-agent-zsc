"""Recipe-aware ingredient runner and observation-driven cook."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'observable_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}


def init_agent_0():
    return jnp.zeros((10,), jnp.int32)

def init_agent_1():
    return jnp.zeros((8,), jnp.int32)

def locate(obs):
    _LAYOUT = ('WWWWBXBWWWW', 'WWW0 A 1WWW', 'WWWWWPWWWWW', 'W    A    W', '0  W3R3W  0', 'W1WWWWWWW1W')
    _FLOOR = tuple((r,c) for r in range(6) for c in range(11) if _LAYOUT[r][c] in ' A')
    _CHANNEL = {'W':0, 'X':1, 'P':2, 'R':3, 'L':4, 'B':5, '0':6, '1':7, '2':8, '3':9}
    _PATCHES = jnp.asarray([[[[int(0 <= r+i-2 < 6 and 0 <= c+j-2 < 11 and _CHANNEL.get(_LAYOUT[r+i-2][c+j-2], -1) == k) for k in range(10)] for j in range(5)] for i in range(5)] for r,c in _FLOOR], jnp.int32)
    _POSITIONS = jnp.asarray(_FLOOR, jnp.int32)
    idx = jnp.argmin(jnp.sum(jnp.abs(_PATCHES - obs[None,:,:,22:32]), axis=(1,2,3)))
    return _POSITIONS[idx]

def cell(obs, pos, row, col):
    r, c = row-pos[0]+2, col-pos[1]+2
    valid = (r>=0)&(r<5)&(c>=0)&(c<5)
    return jnp.where(valid, obs[jnp.clip(r,0,4),jnp.clip(c,0,4)], jnp.zeros((45,),jnp.int32))

def turn_interact(facing, direction):
    moves = jnp.asarray([3,1,0,2],jnp.int32)
    return jnp.where(facing == direction, 5, moves[direction])

def go(pos, facing, row, col, direction):
    # The lower room has a straight central corridor; branches are one tile deep.
    horizontal = jnp.where(pos[1]<col,0,2)
    vertical = jnp.where(pos[0]<row,1,3)
    move = jnp.where(pos[1]!=col, jnp.where(pos[0]==4,3,horizontal),vertical)
    return jnp.where((pos[0]==row)&(pos[1]==col), turn_interact(facing,direction),move)

def drop(obs,pos,facing):
    # Find a locally visible empty counter reachable from the current room.
    rr, cc = jnp.indices((5,5))
    empty = (obs[:,:,22]>0)&(jnp.sum(obs[:,:,32:38],axis=-1)==0)
    gr, gc = rr+pos[0]-2, cc+pos[1]-2
    upper = pos[0]==1
    # Shared counters along row 2 are accessible in both rooms.
    shared = (gr==2)&((gc==4)|(gc==6))
    private = (gr==4)&((gc==3)|(gc==7))
    lower_reachable = ((gr==2)&(gc>=1)&(gc<=9)&(gc!=5))|private
    valid = empty & jnp.where(upper,shared,lower_reachable)
    shared_occupied = jnp.any(shared & (jnp.sum(obs[:,:,32:38],axis=-1)>0))
    penalty = jnp.where((~upper)&shared&shared_occupied,20,0)
    cost = jnp.abs(gc-pos[1]) + penalty + jnp.where(valid,0,100)
    index = jnp.argmin(cost)
    col = gc.reshape(-1)[index]
    row = gr.reshape(-1)[index]
    direction = jnp.where(upper|(row==4),1,0)
    action = go(pos,facing,jnp.where(upper,1,3),col,direction)
    # With no nearby free counter, search the corridor.
    return jnp.where(jnp.any(valid), action, jnp.where(pos[1]<5,0,2))

def recover(obs,pos,facing,target,fallback):
    rr, cc = jnp.indices((5,5))
    gr, gc = rr+pos[0]-2, cc+pos[1]-2
    item = (obs[:,:,22]>0)&(obs[:,:,32]==0)&(obs[:,:,33]==0)&(obs[:,:,34+target]>0)
    upper = pos[0]==1
    reachable = jnp.where(upper,(gr==2)&((gc==4)|(gc==6)),((gr==2)&(gc>=1)&(gc<=9)&(gc!=5))|((gr==4)&((gc==3)|(gc==7))))
    cost = jnp.abs(gc-pos[1]) + jnp.where(item&reachable,0,100)
    idx = jnp.argmin(cost)
    col = gc.reshape(-1)[idx]
    row = gr.reshape(-1)[idx]
    direction = jnp.where(upper|(row==4),1,0)
    return jnp.where(jnp.any(item&reachable),go(pos,facing,jnp.where(upper,1,3),col,direction),fallback)


def agent_0_act(obs, reset, memory, rng):
    m = jnp.where(reset,init_agent_0(),memory)
    pos = locate(obs)
    facing = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:11]
    pot = cell(obs,pos,2,5)
    counts = pot[34:38]
    total = jnp.sum(counts)
    delivered = (m[2]>0)&(inv[1]==0)
    age = jnp.where(reset,100,jnp.where(delivered,0,m[1]+1))
    remembered = jnp.where(delivered,0,m[3])
    partner_inv = jnp.sum(obs[:,:,16:22],axis=(0,1))
    raw_visible = (partner_inv[0]==0)&(jnp.sum(partner_inv[2:4])>0)
    _, local_cols = jnp.indices((5,5))
    partner_col = jnp.sum(obs[:,:,11]*(local_cols+pos[1]-2))
    partner_facing = jnp.argmax(jnp.sum(obs[:,:,12:16],axis=(0,1)))
    raw_type = jnp.argmax(partner_inv[2:])
    code = raw_type+1+4*partner_col+44*partner_facing
    stable = jnp.where(raw_visible,jnp.where(code==m[8],m[7]+1,1),0)
    approach = (jnp.abs(partner_col-5)<=1)&((partner_col==5)|(partner_facing!=0))
    guard = jnp.where(raw_type+1==m[4],8,3)
    offered = raw_visible&(age>=guard)&(approach|(stable>=6))
    target = jnp.where(total>0,jnp.argmax(counts[:2]),jnp.where(offered,jnp.argmax(partner_inv[2:]),jnp.maximum(remembered-1,0)))
    evidence = (total>0)|offered|(remembered>0)
    idle = jnp.where(evidence|(jnp.sum(inv)>0),0,m[5]+1)
    last_served = jnp.where(inv[1]>0,jnp.argmax(inv[2:])+1,m[4])
    probe = (~evidence)&(idle>=jnp.where(m[6]>0,12,24))
    target = jnp.where(probe,jnp.where(last_served==1,1,0),target)
    evidence = evidence|probe
    remembered = jnp.where(evidence,target+1,0)
    raw = (inv[0]==0)&(jnp.sum(inv[2:])>0)
    plate = (inv[0]>0)&(inv[1]==0)
    cooked = inv[1]>0
    busy = (pot[44]>0)|(pot[33]>0)
    pot_action = go(pos,facing,1,5,1)
    # Stock ingredients only after the recipe-aware partner establishes the pot.
    ingredient_action = go(pos,facing,1,jnp.where(target==0,4,6),jnp.where(target==0,3,2))
    ingredient_action = recover(obs,pos,facing,target,ingredient_action)
    plate_col = jnp.where(pos[1]<=5,4,6)
    plate_action = go(pos,facing,1,plate_col,0)
    action = jnp.where(evidence,ingredient_action,go(pos,facing,1,5,1))
    action = jnp.where((~evidence)&(pos[1]==5),4,action)
    action = jnp.where(busy,plate_action,action)
    compatible = jnp.argmax(inv[2:])==target
    action = jnp.where(raw,jnp.where((~busy)&(total<3)&(((total==0)&evidence&(jnp.argmax(inv[2:])==target))|((total>0)&compatible)),pot_action,drop(obs,pos,facing)),action)
    action = jnp.where(plate,jnp.where(pot[33]>0,pot_action,jnp.where((pos[1]!=5)|(facing!=1),go(pos,facing,1,5,1),4)),action)
    action = jnp.where(cooked,go(pos,facing,1,5,0),action)
    nxt = m.at[0].add(1).at[1].set(age).at[2].set(inv[1]).at[3].set(remembered).at[4].set(last_served).at[5].set(idle).at[6].set(jnp.maximum(m[6],probe.astype(jnp.int32))).at[7].set(stable).at[8].set(code)
    return jnp.asarray(action,jnp.int32),nxt,{'pos':pos,'pot':counts,'inventory':inv,'target':target,'probe':probe}

def agent_1_act(obs, reset, memory, rng):
    m = jnp.where(reset,init_agent_1(),memory)
    pos = locate(obs)
    facing = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:11]
    display = jnp.max(obs[:,:,40:44],axis=(0,1))
    seen = jnp.sum(display)>0
    recipe = jnp.where(seen,jnp.argmax(display),m[1])
    known = seen|(m[2]>0)
    pot = cell(obs,pos,2,5)
    visible = pot[24]>0
    total = jnp.sum(pot[34:38])
    busy = (pot[44]>0)|(pot[33]>0)
    removed = visible&(total==0)&(m[3]>0)
    cooldown = jnp.where(removed,5,jnp.maximum(m[4]-1,0))
    partner_cooked = jnp.max(obs[:,:,17])>0
    at_goal = cell(obs,pos,1,5)
    delivered = (m[5]>0)&(~partner_cooked)&(at_goal[11]>0)&(at_goal[12]>0)
    cooldown = jnp.where(delivered,0,cooldown)
    pending = (cooldown>0)|partner_cooked
    raw = (inv[0]==0)&(jnp.sum(inv[2:])>0)
    held_type = jnp.argmax(inv[2:])
    side = jnp.where(pos[1]<=5,1,9)
    supply_dir = jnp.where(recipe==1,1,jnp.where(side==1,3,2))
    fetch = recover(obs,pos,facing,recipe,go(pos,facing,4,side,supply_dir))
    center = go(pos,facing,3,5,0)
    wait_center = jnp.where((pos[0]==3)&(pos[1]==5),4,center)
    # Refresh the indicator before collecting or contributing to a fresh pot.
    action = jnp.where(known,fetch,wait_center)
    action = jnp.where(visible&(busy|pending),wait_center,action)
    can_add = visible&(~busy)&(~pending)&(total<3)
    raw_action = jnp.where((held_type==recipe)|(~known),jnp.where(can_add,center,wait_center),drop(obs,pos,facing))
    action = jnp.where(raw,raw_action,action)
    action = jnp.where(inv[0]>0,drop(obs,pos,facing),action)
    nxt = m.at[0].add(1).at[1].set(recipe).at[2].set(known.astype(jnp.int32)).at[3].set(jnp.where(visible,total,m[3])).at[4].set(cooldown).at[5].set(partner_cooked.astype(jnp.int32))
    return jnp.asarray(action,jnp.int32),nxt,{'pos':pos,'recipe':recipe,'pot':pot[34:38],'inventory':inv,'pending':pending}
