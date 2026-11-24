"""Local-evidence cooking policy for the two separated kitchen rooms."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'boundary_ingredients', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

_LAYOUT = ('WWWWWR2W0WW', '0      W  B', 'W     APA X', '1      W  B', 'WWWWWR2W1WW')
_OBJECTS = ('W', 'X', 'P', 'R', 'L', 'B', '0', '1', '2')
def _geometry():
    positions = tuple((r, c) for r in range(1, 4) for c in range(1, 10) if _LAYOUT[r][c] in ' A')
    templates = tuple(tuple(tuple(tuple(int(0 <= r+i-2 < 5 and 0 <= c+j-2 < 11 and _LAYOUT[r+i-2][c+j-2] == ch) for ch in _OBJECTS) for j in range(5)) for i in range(5)) for r,c in positions)
    return jnp.asarray(positions,jnp.int32),jnp.asarray(templates,jnp.int32)


def _init():
    return {'step': jnp.int32(0), 'recipe': jnp.zeros(3, jnp.int32),
            'pot': jnp.zeros(5, jnp.int32), 'timer': jnp.int32(0),
            'cooldown': jnp.int32(0), 'inferred': jnp.int32(-1), 'since_delivery': jnp.int32(100), 'yield_steps': jnp.int32(0), 'grid': jnp.zeros((5,11,5), jnp.int32)}


def init_agent_0():
    return _init()


def init_agent_1():
    return _init()


def _observe(obs, reset, memory):
    m = jax.tree.map(lambda fresh, old: jnp.where(reset, fresh, old), _init(), memory)
    positions,templates = _geometry()
    error = jnp.sum(jnp.abs(templates - obs[None,:,:,20:29]), axis=(1,2,3))
    pos = positions[jnp.argmin(error)]
    face = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:10]
    rec = jnp.max(obs[:,:,36:39], axis=(0,1))
    rec = jnp.where(jnp.sum(rec)>0, rec, m['recipe'])
    pot_visible = jnp.any(obs[:,:,22]>0)
    pot = jnp.sum(obs[:,:,29:34] * obs[:,:,22:23], axis=(0,1))
    pot = jnp.where(pot_visible, pot, m['pot'])
    timer = jnp.where(pot_visible, jnp.max(obs[:,:,39]), jnp.maximum(m['timer']-1,0))
    emptied = pot_visible & (jnp.sum(pot[2:])==0) & (jnp.sum(m['pot'][2:])>0)
    cooldown = jnp.where(emptied, 24, jnp.maximum(m['cooldown']-1,0))
    changed_recipe = jnp.any(rec != m['recipe'])
    partner_visible = jnp.any(obs[:,:,10]>0)
    partner_dish = jnp.any(obs[:,:,16]>0)
    cooldown = jnp.where(partner_dish & (jnp.sum(pot[2:])==0),24,cooldown)
    completion_evidence = changed_recipe | (partner_visible & ~partner_dish)
    cooldown = jnp.where(completion_evidence,0,cooldown)
    # A remembered public grid is useful for choosing an unoccupied counter.
    rows = pos[0] + jnp.arange(5)[:,None] - 2
    cols = pos[1] + jnp.arange(5)[None,:] - 2
    rows = jnp.broadcast_to(rows,(5,5)); cols = jnp.broadcast_to(cols,(5,5))
    valid = (rows>=0)&(rows<5)&(cols>=0)&(cols<11)
    grid = m['grid'].at[jnp.where(valid,rows,5),jnp.where(valid,cols,11)].set(obs[:,:,29:34],mode='drop')
    n = {'step':m['step']+1, 'recipe':rec, 'pot':pot, 'timer':timer, 'cooldown':cooldown, 'inferred':m['inferred'], 'since_delivery':m['since_delivery']+1, 'yield_steps':m['yield_steps'], 'grid':grid}
    return n, pos, face, inv, pot_visible


def _go(pos, face, r, c, direction=-1):
    move = jnp.where(pos[1]<c, 0, jnp.where(pos[1]>c, 2, jnp.where(pos[0]<r,1,jnp.where(pos[0]>r,3,4))))
    turn = jnp.asarray([3,1,0,2,4],jnp.int32)[jnp.where(direction<0,4,direction)]
    at = (pos[0]==r)&(pos[1]==c)
    return jnp.where(at, jnp.where(direction<0,4,jnp.where(face==direction,5,turn)),move).astype(jnp.int32)


def _dump(pos, face, grid, seat):
    # Each entry: counter row,column; approach row,column; facing.
    if seat == 0:
        places = jnp.asarray([(0,4,1,4,0),(4,4,3,4,1),(0,3,1,3,0),(4,3,3,3,1),(0,2,1,2,0),(4,2,3,2,1),(0,1,1,1,0),(4,1,3,1,1),(2,0,2,1,3)],jnp.int32)
    else:
        places = jnp.asarray([(1,7,1,8,3),(3,7,3,8,3),(0,9,1,9,0),(4,9,3,9,1)],jnp.int32)
    full = jnp.any(grid[places[:,0],places[:,1]]!=0,axis=-1)
    dist = jnp.abs(places[:,2]-pos[0])+jnp.abs(places[:,3]-pos[1])
    pick = jnp.argmin(dist + 1000*full)
    p = places[pick]
    return jnp.where(jnp.all(full),4,_go(pos,face,p[2],p[3],p[4]))



def _left_ingredient(pos, face, grid, kind):
    places = jnp.asarray([(1,7,1,6,2),(3,7,3,6,2),(0,4,1,4,0),(4,4,3,4,1),(0,3,1,3,0),(4,3,3,3,1),(0,2,1,2,0),(4,2,3,2,1),(0,1,1,1,0),(4,1,3,1,1),(2,0,2,1,3)],jnp.int32)
    items = grid[places[:,0],places[:,1]]
    matches = (items[:,2+kind]>0)&(items[:,1]==0)
    dist = jnp.abs(places[:,2]-pos[0])+jnp.abs(places[:,3]-pos[1])
    supply_row = jnp.where(kind==0,1,3)
    supply_dist = jnp.abs(supply_row-pos[0])+jnp.abs(1-pos[1])
    pick = jnp.argmin(jnp.where(matches,dist,1000))
    p = places[pick]
    return jnp.where(matches[pick]&(dist[pick]<=supply_dist),_go(pos,face,p[2],p[3],p[4]),_go(pos,face,supply_row,1,3))



def _right_pickup(pos, face, grid, kind, want_plate, fallback):
    places = jnp.asarray([(1,7,1,8,3),(3,7,3,8,3),(0,9,1,9,0),(4,9,3,9,1)],jnp.int32)
    items = grid[places[:,0],places[:,1]]
    matches = jnp.where(want_plate,(items[:,0]>0)&(jnp.sum(items[:,2:],axis=-1)==0),(items[:,2+kind]>0)&(items[:,0]==0)) & (items[:,1]==0)
    dist = jnp.abs(places[:,2]-pos[0])+jnp.abs(places[:,3]-pos[1])
    pick = jnp.argmin(jnp.where(matches,dist,1000)); p = places[pick]
    return jnp.where(matches[pick],_go(pos,face,p[2],p[3],p[4]),fallback)


def agent_0_act(obs, reset, memory, rng):
    n,pos,face,inv,visible = _observe(obs,reset,memory)
    recipe = n['recipe']; kind = jnp.argmax(recipe)
    raw = (jnp.sum(inv[2:])>0)&(inv[1]==0)
    empty = jnp.sum(inv)==0
    correct = jnp.all(inv[2:]<=recipe)
    count = jnp.sum(n['pot'][2:])
    pot_ok = jnp.all(n['pot'][2:]<=recipe)
    can_add = visible & (count<3)&(n['pot'][1]==0)&(n['timer']==0)&(n['cooldown']==0)
    pot_action = _go(pos,face,2,6,2)
    supply = _left_ingredient(pos,face,n['grid'],kind)
    action = jnp.where(empty,supply,jnp.where(raw & correct,jnp.where(can_add,pot_action,_go(pos,face,2,6)),_dump(pos,face,n['grid'],0)))
    action = jnp.where(jnp.sum(recipe)==0,_go(pos,face,2,6),action)
    # Refresh the target at the indicators before beginning another batch.
    action = jnp.where((n['cooldown']>0)&empty,_go(pos,face,2,6),action)
    return action,n,{'pos':pos,'recipe':recipe,'pot':n['pot'],'cooldown':n['cooldown'],'inv':inv,'partner':jnp.sum(obs[:,:,15:20],axis=(0,1))}


def agent_1_act(obs, reset, memory, rng):
    n,pos,face,inv,visible = _observe(obs,reset,memory)
    pot = n['pot']; count = jnp.sum(pot[2:])
    partner = jnp.sum(obs[:,:,15:20],axis=(0,1))
    rr,cc = jnp.unravel_index(jnp.argmax(obs[:,:,10]),(5,5))
    partner_at_pot = (pos[0]+rr-2==2)&(pos[1]+cc-2==6)&jnp.any(obs[:,:,10]>0)
    partner_at_boundary = (pos[1]+cc-2==6)&jnp.any(obs[:,:,10]>0)
    fresh_after = jnp.where(partner_at_pot,3,5)
    signal = partner_at_boundary & (partner[0]==0)&(partner[1]==0)&(jnp.sum(partner[2:4])==1)&(n['since_delivery']>=fresh_after)
    inferred = jnp.where((count==0)&signal,jnp.argmax(partner[2:]),n['inferred'])
    irreparable = (pot[4]>0)|(jnp.sum(pot[2:]>0)>1)
    kind = jnp.where(count>0,jnp.argmax(pot[2:4]),jnp.maximum(inferred,0))
    n['inferred'] = inferred
    raw = (jnp.sum(inv[2:])>0)&(inv[1]==0)
    empty = jnp.sum(inv)==0
    plate = (inv[0]>0)&(inv[1]==0)
    dish = inv[1]>0
    partial = ((count>0)|(inferred>=0))&(count<3)&(n['timer']==0)&(pot[1]==0)
    busy = (n['timer']>0)|(pot[1]>0)
    compatible = irreparable | jnp.where(count>0,jnp.all((inv[2:]==0)|(pot[2:]>0)),jnp.argmax(inv[2:])==kind)
    pot_action = _go(pos,face,2,8,3)
    supply = _go(pos,face,jnp.where(kind==0,1,3),8,jnp.where(kind==0,0,1))
    supply = _right_pickup(pos,face,n['grid'],kind,False,supply)
    plate_action = _go(pos,face,jnp.where(pos[0]<=2,1,3),9,2)
    plate_action = _right_pickup(pos,face,n['grid'],kind,True,plate_action)
    wait = _go(pos,face,2,8)
    action = jnp.where(empty,jnp.where(partial,supply,jnp.where(busy,plate_action,wait)),wait)
    partner_finishing = (count==2)&(jnp.sum(obs[:,:,13])>0)&partner_at_pot&(partner[0]==0)&(partner[1]==0)&(partner[2+kind]>0)
    n['yield_steps'] = jnp.where(partner_finishing,n['yield_steps']+1,0)
    action = jnp.where(empty & partner_finishing & (n['yield_steps']<=2),wait,action)
    action = jnp.where(raw,jnp.where(partial&compatible,pot_action,_dump(pos,face,n['grid'],1)),action)
    action = jnp.where(plate,jnp.where(partial,_dump(pos,face,n['grid'],1),pot_action),action)
    # Mixed dishes cannot match either legal recipe; clear them without penalty.
    uniform = jnp.max(inv[2:4])==3
    action = jnp.where(dish,jnp.where(uniform,_go(pos,face,2,9,2),_dump(pos,face,n['grid'],1)),action)
    delivered = dish & uniform & (pos[0]==2)&(pos[1]==9)&(face==2)&(action==5)
    n['since_delivery'] = jnp.where(delivered,0,n['since_delivery'])
    n['inferred'] = jnp.where(delivered,-1,n['inferred'])
    return action,n,{'pos':pos,'pot':pot,'kind':kind,'inferred':inferred,'since_delivery':n['since_delivery'],'inv':inv,'partner':jnp.sum(obs[:,:,15:20],axis=(0,1))}
