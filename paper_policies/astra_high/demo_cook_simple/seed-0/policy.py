"""Reactive cooks using public pot contents and the visible recipe."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'observable_cooks', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

_LAYOUT = ('WWWWWR2W0WW', '0      W  B', 'W      P  X', '1      W  B', 'WWWWWR2W1WW')
def _geometry():
    positions = jnp.asarray([(r,c) for r in range(1,4) for c in range(1,10) if _LAYOUT[r][c] == ' '], jnp.int32)
    crops = jnp.asarray([[[[int(0 <= r+i-2 < 5 and 0 <= c+j-2 < 11 and _LAYOUT[r+i-2][c+j-2] == obj) for obj in ('W','X','P','R','L','B','0','1','2')] for j in range(5)] for i in range(5)] for r in range(1,4) for c in range(1,10) if _LAYOUT[r][c] == ' '], jnp.int32)
    return positions,crops


def _counters(seat):
    if seat == 0:
        entries = [(0,c,1,c,0) for c in range(1,5)] + [(4,c,3,c,1) for c in range(1,5)] + [(2,0,2,1,3)]
    else:
        entries = [(0,9,1,9,0), (4,9,3,9,1), (1,7,1,8,3), (3,7,3,8,3)]
    return jnp.asarray(entries,jnp.int32)


def _init():
    return {'recipe': jnp.asarray(-1,jnp.int32), 'grid': jnp.zeros((5,11,5),jnp.int32), 'pot': jnp.zeros((5,),jnp.int32), 'cooldown': jnp.asarray(0,jnp.int32), 'time': jnp.asarray(0,jnp.int32), 'cue': jnp.asarray(-1,jnp.int32), 'post_delivery': jnp.asarray(0,jnp.int32), 'partner_raw_seen': jnp.asarray(False), 'partner_fills': jnp.asarray(False), 'own_added': jnp.asarray(0,jnp.int32), 'pot_idle': jnp.asarray(0,jnp.int32), 'service': jnp.asarray(False)}


def init_agent_0():
    return _init()


def init_agent_1():
    return _init()


def _decode(obs, reset, memory):
    m = jax.tree.map(lambda x,z:jnp.where(reset,z,x), memory, _init())
    positions,crops = _geometry()
    pos = positions[jnp.argmin(jnp.sum(jnp.abs(crops-obs[None,:,:,20:29]),axis=(1,2,3)))]
    face = jnp.argmax(obs[2,2,1:5]).astype(jnp.int32)
    inv = obs[2,2,5:10]
    rr,cc = jnp.meshgrid(jnp.arange(5),jnp.arange(11), indexing='ij')
    lr,lc = rr-pos[0]+2, cc-pos[1]+2
    visible = (lr>=0)&(lr<5)&(lc>=0)&(lc<5)
    grid = jnp.where(visible[:,:,None],obs[jnp.clip(lr,0,4),jnp.clip(lc,0,4),29:34],m['grid'])
    pot_visible = visible[2,7]
    pot = grid[2,7]
    scooped = pot_visible & (jnp.sum(m['pot'][2:])>0) & (jnp.sum(pot[2:])==0)
    recipe_counts = jnp.sum(obs[:,:,36:39],axis=(0,1))
    recipe = jnp.where(jnp.sum(recipe_counts)>0,jnp.argmax(recipe_counts).astype(jnp.int32),m['recipe'])
    partner_visible = jnp.sum(obs[:,:,10])>0
    partner_dish = jnp.sum(obs[:,:,16])>0
    counter_dish = jnp.any((grid[:,:,0]>0)&(grid[:,:,1]>0))
    carried_dish = inv[1]>0
    pending = scooped | (m['cooldown']>0) | (counter_dish | carried_dish | partner_dish)
    changed = (recipe>=0)&(m['recipe']>=0)&(recipe!=m['recipe'])
    returned = partner_visible & ~partner_dish & ~counter_dish & ~carried_dish & ~scooped
    cd = jnp.where(scooped,20,jnp.maximum(m['cooldown']-1,0))
    cd = jnp.where(pending & (partner_dish | counter_dish | carried_dish),20,cd)
    cd = jnp.where(pending & (changed | returned),0,cd)
    partner_raw = (jnp.sum(obs[:,:,17:20])>0)&(jnp.sum(obs[:,:,15])==0)
    increased = jnp.sum(pot[2:])-jnp.sum(m['pot'][2:])
    fills = pot_visible & (increased>m['own_added'])
    nm = {'service':m['service'], 'recipe':recipe, 'grid':grid, 'pot':pot, 'cooldown':cd, 'time':m['time']+1, 'cue':m['cue'], 'post_delivery':jnp.maximum(m['post_delivery']-1,0), 'partner_raw_seen':m['partner_raw_seen']|partner_raw, 'partner_fills':m['partner_fills']|fills, 'own_added':jnp.asarray(0,jnp.int32), 'pot_idle':jnp.where(pot_visible&jnp.all(pot==m['pot']),jnp.minimum(m['pot_idle']+1,1000),0)}
    return pos,face,inv,nm,pot_visible


def _go(pos,face,row,col,direction):
    horizontal=jnp.where(pos[1]<col,0,2)
    vertical=jnp.where(pos[0]<row,1,3)
    xfirst=jnp.where(pos[1]!=col,horizontal,vertical)
    yfirst=jnp.where(pos[0]!=row,vertical,horizontal)
    move=jnp.where(direction>=2,yfirst,xfirst)
    at = (pos[0]==row)&(pos[1]==col)
    return jnp.where(at,jnp.where(face==direction,5,jnp.asarray([3,1,0,2],jnp.int32)[direction]),move).astype(jnp.int32)


def _wait_pot(pos,face,seat):
    at = (pos[0]==2)&(pos[1]==(6 if seat==0 else 8))&(face==(2 if seat==0 else 3))
    return jnp.where(at,4,_go(pos,face,2,6 if seat==0 else 8,2 if seat==0 else 3))


def _cost(pos,face,entries):
    dist = jnp.abs(entries[:,2]-pos[0])+jnp.abs(entries[:,3]-pos[1])
    return dist*2+jnp.where((dist==0)&(entries[:,4]==face),0,1)


def _drop(pos,face,grid,seat):
    counters = _counters(seat)
    empty = jnp.sum(grid[counters[:,0],counters[:,1]],axis=1)==0
    cost = _cost(pos,face,counters)+jnp.where(empty,0,10000)
    ent = counters[jnp.argmin(cost)]
    action = _go(pos,face,ent[2],ent[3],ent[4])
    return jnp.where(jnp.any(empty),action,4)


def _fetch(pos,face,grid,kind,seat):
    counters = _counters(seat)
    if seat==0:
        source = jnp.stack([jnp.where(kind==0,1,3),jnp.asarray(0),jnp.where(kind==0,1,3),jnp.asarray(1),jnp.asarray(3)])
    else:
        source = jnp.stack([jnp.where(kind==0,0,4),jnp.asarray(8),jnp.where(kind==0,1,3),jnp.asarray(8),jnp.where(kind==0,0,1)])
    entries = jnp.concatenate([source[None,:],counters],axis=0)
    items = grid[counters[:,0],counters[:,1]]
    raw_match = (items[:,0]==0)&(items[:,1]==0)&(items[:,2+kind]>0)
    valid = jnp.concatenate([jnp.ones((1,),bool),raw_match])
    idx = jnp.argmin(_cost(pos,face,entries)+jnp.where(valid,0,10000))
    ent = entries[idx]
    return _go(pos,face,ent[2],ent[3],ent[4])


def agent_0_act(obs, reset, memory, rng):
    pos,face,inv,m,pv = _decode(obs,reset,memory)
    pot = m['pot']; n = jnp.sum(pot[2:]); recipe=m['recipe']
    raw = (inv[0]==0)&(jnp.sum(inv[2:])>0)
    empty = jnp.sum(inv)==0
    correct = inv[2+jnp.maximum(recipe,0)]>0
    # Fetch continually, including a spare while the current dish cooks.
    action = _fetch(pos,face,m['grid'],jnp.maximum(recipe,0),0)
    put = _go(pos,face,2,6,2)
    # Expose a carried ingredient at column six earlier when the partner helps fill.
    expose = jnp.where(pos[1]<6,0,jnp.where(pos[0]<2,1,3))
    put = jnp.where(m['partner_raw_seen']&((pos[0]!=2)|(pos[1]!=6)),expose,put)
    wait = _wait_pot(pos,face,0)
    action = jnp.where(raw,jnp.where(correct,jnp.where((n<3)&(pot[1]==0)&(m['cooldown']==0),put,wait),_drop(pos,face,m['grid'],0)),action)
    action = jnp.where(~empty & ~raw,_drop(pos,face,m['grid'],0),action)
    # A supplied shared plate enables an alternate, concrete division of labor.
    shared=jnp.asarray([[1,7,1,6,2],[3,7,3,6,2]],jnp.int32)
    si=m['grid'][shared[:,0],shared[:,1]]
    has_plate=(si[:,0]>0)&(si[:,1]==0)&(jnp.sum(si[:,2:],axis=1)==0)
    available=jnp.sum(si,axis=1)==0
    pe=shared[jnp.argmin(_cost(pos,face,shared)+jnp.where(has_plate,0,10000))]
    de=shared[jnp.argmin(_cost(pos,face,shared)+jnp.where(available,0,10000))]
    partner_plate=jnp.sum(obs[:,:,15])>0
    dish=inv[1]>0
    plate=(inv[0]>0)&~dish
    trigger=((pot[1]>0)&(m['pot_idle']>=6))|((n==3)&empty&~m['partner_raw_seen'])
    trigger=trigger&jnp.any(has_plate)&~partner_plate
    serve=(m['service']|trigger)&(jnp.any(has_plate)|plate|dish)&((n>0)|plate|dish)
    take_plate=_go(pos,face,pe[2],pe[3],pe[4])
    give_item=jnp.where(jnp.any(available),_go(pos,face,de[2],de[3],de[4]),wait)
    action=jnp.where(serve&raw,_drop(pos,face,m['grid'],0),action)
    action=jnp.where(serve&empty,take_plate,action)
    action=jnp.where(plate,jnp.where(n==3,jnp.where(pot[1]>0,_go(pos,face,2,6,2),wait),give_item),action)
    action=jnp.where(dish,give_item,action)
    handed=dish&(action==5)&(pos[0]==de[2])&(pos[1]==de[3])&(face==de[4])&jnp.any(available)
    m['service']=serve&~handed
    # Before seeing a recipe, walk to the shared pot where the indicator is visible.
    action = jnp.where(recipe<0,wait,action)
    m['own_added']=((action==5)&raw&(pos[0]==2)&(pos[1]==6)&(face==2)&(n<3)).astype(jnp.int32)
    return action.astype(jnp.int32),m,{'pos':pos,'recipe':recipe,'pot':pot,'cooldown':m['cooldown'],'service':m['service'],'inventory':inv}


def agent_1_act(obs, reset, memory, rng):
    pos,face,inv,m,pv = _decode(obs,reset,memory)
    pot=m['pot']; n=jnp.sum(pot[2:])
    raw = (inv[0]==0)&(jnp.sum(inv[2:])>0)
    dish = inv[1]>0
    plate = (inv[0]>0)&~dish
    empty=jnp.sum(inv)==0
    partner = jnp.sum(obs[:,:,15:20],axis=(0,1))
    partner_raw = (partner[0]==0)&(jnp.sum(partner[2:])>0)
    counters = _counters(1)
    items = m['grid'][counters[:,0],counters[:,1]]
    old_grid = jnp.where(reset,jnp.zeros_like(memory['grid']),memory['grid'])
    old_items=old_grid[counters[:,0],counters[:,1]]
    fresh_raw=(items[:,0]==0)&(items[:,1]==0)&(jnp.sum(items[:,2:],axis=1)>0)&(jnp.sum(old_items,axis=1)==0)
    counter_cue=jnp.argmax(jnp.sum(jnp.where(fresh_raw[:,None],items[:,2:],0),axis=0)).astype(jnp.int32)
    cue=jnp.where(jnp.any(fresh_raw),counter_cue,m['cue'])
    cue=jnp.where(partner_raw,jnp.argmax(partner[2:]).astype(jnp.int32),cue)
    cue=jnp.where(m['post_delivery']>0,-1,cue)
    cue=jnp.where(n>0,jnp.argmax(pot[2:]).astype(jnp.int32),cue)
    known=cue>=0
    kind=jnp.maximum(cue,0)
    wait=_wait_pot(pos,face,1)
    fill=_go(pos,face,2,8,3)
    plate_entries=jnp.concatenate([jnp.asarray([[1,10,1,9,2],[3,10,3,9,2]],jnp.int32),counters],axis=0)
    plate_valid=jnp.concatenate([jnp.ones((2,),bool),(items[:,0]>0)&(items[:,1]==0)&(jnp.sum(items[:,2:],axis=1)==0)])
    pe=plate_entries[jnp.argmin(_cost(pos,face,plate_entries)+jnp.where(plate_valid,0,10000))]
    get_plate=_go(pos,face,pe[2],pe[3],pe[4])
    action=jnp.where((n<3)&known,_fetch(pos,face,m['grid'],kind,1),jnp.where(n==3,get_plate,wait))
    # A partner who has actually added ingredients can finish the last slot.
    pcell=jnp.argmax(obs[:,:,10]).astype(jnp.int32)
    prow=pos[0]+pcell//5-2; pcol=pos[1]+pcell%5-2
    pface=jnp.argmax(jnp.sum(obs[:,:,11:15],axis=(0,1)))
    arriving=m['partner_fills']&partner_raw&(partner[2+kind]>0)&(prow==2)&(pcol==6)&(pface==2)
    action=jnp.where(empty&(n==2)&arriving&(m['pot_idle']<4),get_plate,action)
    action=jnp.where(raw,jnp.where((n<3)&known&(inv[2+kind]>0),fill,_drop(pos,face,m['grid'],1)),action)
    action=jnp.where(plate,jnp.where(pot[1]>0,fill,wait),action)
    stalled_plate=plate&(n<3)&(m['pot_idle']>=8)
    action=jnp.where(stalled_plate,_drop(pos,face,m['grid'],1),action)
    m['partner_fills']=jnp.where(stalled_plate,False,m['partner_fills'])
    # A cooked dish on a reachable counter is a usable physical handoff.
    cooked=(items[:,0]>0)&(items[:,1]>0)
    ent=counters[jnp.argmin(_cost(pos,face,counters)+jnp.where(cooked,0,10000))]
    handoff=_go(pos,face,ent[2],ent[3],ent[4])
    action=jnp.where(empty&jnp.any(cooked),handoff,action)
    action=jnp.where((plate|raw)&jnp.any(cooked),_drop(pos,face,m['grid'],1),action)
    action=jnp.where(dish,_go(pos,face,2,9,2),action)
    delivered=dish&(pos[0]==2)&(pos[1]==9)&(face==2)&(action==5)
    m['own_added']=((action==5)&raw&(pos[0]==2)&(pos[1]==8)&(face==3)&(n<3)).astype(jnp.int32)
    m['cue']=jnp.where(delivered,-1,cue)
    m['post_delivery']=jnp.where(delivered,4,m['post_delivery'])
    return action.astype(jnp.int32),m,{'pos':pos,'pot':pot,'kind':cue,'inventory':inv}
