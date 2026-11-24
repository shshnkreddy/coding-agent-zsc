"""Coordinate through visible ingredients, counters, and pot contents."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'observable_adaptation', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

_LAYOUT = ('WWXBWW', '0 A  0', '1    1', 'WPWPWW', '3 A  3', 'W    W', 'WWRWWW')
def _geometry():
    _STATIC = jnp.asarray([[[int(ch == obj) for obj in ('W','X','P','R','L','B','0','1','2','3')] for ch in row] for row in _LAYOUT], jnp.int32)
    _POSITIONS = jnp.asarray([(r,c) for r in (1,2,4,5) for c in (1,2,3,4)], jnp.int32)
    _OFFSETS = jnp.stack(jnp.meshgrid(jnp.arange(-2,3),jnp.arange(-2,3),indexing='ij'),-1)
    _PATCH_COORDS = _POSITIONS[:,None,None,:] + _OFFSETS
    _PATCH_IN = jnp.all((_PATCH_COORDS >= 0) & (_PATCH_COORDS < jnp.asarray([7,6])),axis=-1)
    _PATCHES = _STATIC[jnp.clip(_PATCH_COORDS[...,0],0,6),jnp.clip(_PATCH_COORDS[...,1],0,5)] * _PATCH_IN[...,None]
    return _POSITIONS, _PATCHES

def _objects():
    return jnp.asarray([[3,1],[3,3]],jnp.int32),jnp.asarray([[3,2],[3,4]],jnp.int32)

def _init():
    return {'grid':jnp.zeros((7,6,7),jnp.int32),'known':jnp.asarray(-1,jnp.int32),'step':jnp.asarray(0,jnp.int32),
        'signature':jnp.asarray(-1,jnp.int32),'stable':jnp.asarray(0,jnp.int32),'idle':jnp.asarray(0,jnp.int32),
        'response':jnp.full((4,),-1,jnp.int32),'wait_column':jnp.asarray(-1,jnp.int32),'wait_stable':jnp.asarray(0,jnp.int32),'item_partner':jnp.asarray(False),
        'physical_seen':jnp.asarray(False),'mode':jnp.asarray(False),'belief':jnp.zeros((64,),jnp.int32),'pending':jnp.asarray(-1,jnp.int32),
        'last_signature':jnp.asarray(0,jnp.int32),'last_guess':jnp.asarray(0,jnp.int32)}

def init_agent_0(): return _init()
def init_agent_1(): return _init()

def _observe(obs,reset,memory):
    m = jax.tree.map(lambda old,initial:jnp.where(reset,initial,old),memory,_init())
    _POSITIONS, _PATCHES = _geometry()
    errors = jnp.sum(jnp.abs(_PATCHES-obs[None,:,:,22:32]),axis=(1,2,3))
    pos = _POSITIONS[jnp.argmin(errors)]
    rr,cc = jnp.meshgrid(jnp.arange(7),jnp.arange(6),indexing='ij')
    lr,lc = rr-pos[0]+2, cc-pos[1]+2
    visible=(lr>=0)&(lr<5)&(lc>=0)&(lc<5)
    raw = obs[jnp.clip(lr,0,4),jnp.clip(lc,0,4)]
    dynamic=jnp.concatenate([raw[...,32:38],raw[...,44:45]],-1)
    old=m['grid']
    timer=jnp.maximum(old[...,6]-1,0)
    old=old.at[...,1].set(jnp.maximum(old[...,1],((old[...,6]>0)&(timer==0)).astype(jnp.int32)))
    old=old.at[...,6].set(timer)
    grid=jnp.where(visible[...,None],dynamic,old)
    inv=obs[2,2,5:11]
    facing=jnp.argmax(obs[2,2,1:5])
    return m,pos,grid,inv,facing

def _go(pos,facing,target):
    """Move inside a rectangular room to an interaction position."""
    top=pos[0]<3
    tr,tc=target[0],target[1]
    ar=jnp.where(tr==3,jnp.where(top,2,4),jnp.where(tr==0,1,jnp.where(tr==6,5,tr)))
    ac=jnp.clip(tc,1,4)
    dr,dc=tr-ar,tc-ac
    orient=jnp.where(dr<0,0,jnp.where(dr>0,1,jnp.where(dc>0,2,3)))
    turn=jnp.asarray([3,1,0,2],jnp.int32)[orient]
    move=jnp.where(pos[1]<ac,0,jnp.where(pos[1]>ac,2,jnp.where(pos[0]<ar,1,3)))
    return jnp.where((pos[0]==ar)&(pos[1]==ac),jnp.where(facing==orient,5,turn),move).astype(jnp.int32)

def _walk(pos,target):
    return jnp.where(pos[1]<target[1],0,jnp.where(pos[1]>target[1],2,jnp.where(pos[0]<target[0],1,jnp.where(pos[0]>target[0],3,4)))).astype(jnp.int32)

def _supply(pos,kind):
    # The left pot is faster for either ingredient, but use nearby supply.
    col=jnp.where(pos[1]<=2,0,5)
    return jnp.asarray([1,0],jnp.int32).at[0].set(kind+1).at[1].set(col)

def agent_0_act(obs, reset, memory, rng):
    m,pos,grid,inv,facing=_observe(obs,reset,memory)
    _POTS, _COUNTERS = _objects()
    pots=grid[_POTS[:,0],_POTS[:,1]]
    counters=grid[_COUNTERS[:,0],_COUNTERS[:,1]]
    counts=jnp.sum(pots[:,2:6],axis=-1)
    valid_pots=(counts>0)&(jnp.max(pots[:,2:4],axis=-1)==counts)
    blocked=(~jnp.any(valid_pots))&jnp.all(counts>0)
    active=jnp.argmax(jnp.where(valid_pots,10+counts,-100))
    active=jnp.where(blocked,jnp.argmax(counts),active)
    # Observe a partner's stable public behavior. These are learned labels,
    # not messages or shared memory.
    partner_visible=jnp.sum(obs[:,:,11])>0
    local_partner=jnp.asarray(jnp.unravel_index(jnp.argmax(obs[:,:,11]),(5,5)),jnp.int32)
    partner_pos=pos+local_partner-2
    partner_direction=jnp.argmax(jnp.sum(obs[:,:,12:16],axis=(0,1)))
    partner_inv=jnp.sum(obs[:,:,16:22],axis=(0,1))
    observed_signature=(((partner_pos[0]-4)*4+partner_pos[1]-1)*4+partner_direction+32*(partner_inv[5]>0)).clip(0,63)
    signature=jnp.where(partner_visible,observed_signature,m['signature'])
    stable=jnp.where(partner_visible,jnp.where(signature==m['signature'],m['stable']+1,1),m['stable'])
    pending=jnp.where(m['pending']>=0,m['pending']+1,-1)
    belief=m['belief']
    changed=(pending>=2)&(pending<=14)&partner_visible&(stable>=3)&(signature!=m['last_signature'])
    last_sign=2*m['last_guess']-1
    old_value=belief[m['last_signature']]
    no_change=(pending>=10)&((partner_visible&(stable>=3)&(signature==m['last_signature']))|(~partner_visible))
    revise=no_change&(jnp.abs(old_value)<50)
    belief=belief.at[m['last_signature']].set(jnp.where(changed,100*last_sign,jnp.where(revise,old_value-last_sign,old_value)))
    belief=belief.at[signature.clip(0,63)].set(jnp.where(changed,-100*last_sign,belief[signature.clip(0,63)]))
    # If a change used different positions, orientation is incidental to that
    # particular signal. Learn all orientations at those two positions.
    old_location=(m['last_signature']%32)//4
    location=(signature%32)//4
    indices=jnp.arange(64,dtype=jnp.int32)
    position_change=changed&(old_location!=location)
    belief=jnp.where(position_change&((indices%32)//4==old_location),100*last_sign,belief)
    belief=jnp.where(position_change&((indices%32)//4==location),-100*last_sign,belief)
    confirmed=(pending>=6)&partner_visible&(stable>=3)&(jnp.abs(belief[signature.clip(0,63)])>=50)
    pending=jnp.where(changed|no_change|confirmed|(pending>18),-1,pending)
    empty_hands=jnp.sum(inv)==0
    idle=jnp.where((~jnp.any(valid_pots))&empty_hands,m['idle']+1,0)
    physical_partner=jnp.sum(partner_inv[2:4])>0
    physical_seen=m['physical_seen']|physical_partner
    offers_ready=jnp.all(jnp.sum(counters[:,:6],axis=-1)>0)
    activate=((idle>jnp.where(physical_seen,70,12))&(stable>=6)&partner_visible&offers_ready)|(idle>jnp.where(physical_seen,90,50))
    mode=(m['mode']|activate)&~physical_partner
    value=belief[signature.clip(0,63)]
    prior=(partner_pos[1]>2).astype(jnp.int32)
    # Decode the remembered column even when the partner temporarily leaves view.
    prior=((signature%32)//4%4>=2).astype(jnp.int32)
    guess=jnp.where(value==0,prior,(value>0).astype(jnp.int32))
    active=jnp.where(jnp.any(valid_pots)|blocked,active,jnp.argmin(jnp.where(counts==0,jnp.abs(_POTS[:,1]-pos[1]),100)))
    p=pots[active]
    ptype=jnp.where(blocked,0,jnp.where(jnp.any(valid_pots),jnp.argmax(p[2:6]),guess))
    anypot=jnp.any(valid_pots)|blocked
    known=jnp.where(anypot,ptype,m['known'])
    holding=jnp.sum(inv)>0
    raw=(inv[0]==0)&(jnp.sum(inv[2:])>0)
    cooked=inv[1]>0
    plate=(inv[0]>0)&(~cooked)
    itype=jnp.argmax(inv[2:])
    ready=(p[1]>0)&anypot
    cooking=p[6]>0
    partial=(anypot&(jnp.sum(p[2:])<3)&(~ready)&(~cooking))|(~anypot&mode&(pending<0))
    missing=jnp.sum(counters[:,:6],axis=-1)==0
    present=jnp.any((counters[:,2:4]>0)&(counters[:,0:1]==0)&(counters[:,1:2]==0),axis=0)
    stock=jnp.argmin(present.astype(jnp.int32))
    slot=jnp.where(missing[itype.clip(0,1)],itype.clip(0,1),jnp.argmax(missing))
    storage=jnp.asarray([[0,1],[0,4],[3,2],[3,4]],jnp.int32)
    stored=grid[storage[:,0],storage[:,1]]
    free_store=jnp.sum(stored[:,:6],axis=-1)==0
    discard=storage[jnp.argmin(jnp.where(free_store,jnp.abs(storage[:,0]-pos[0])+jnp.abs(storage[:,1]-pos[1]),100))]
    need_type=jnp.where(partial,ptype,stock)
    reuse=(stored[:,2+need_type.clip(0,3)]>0)&(stored[:,0]==0)&(stored[:,1]==0)
    # Do not take back a useful public offer just to put it down again.
    reuse=reuse & ((jnp.arange(4)<2)|partial)
    reuse_target=storage[jnp.argmin(jnp.where(reuse,jnp.abs(storage[:,0]-pos[0])+jnp.abs(storage[:,1]-pos[1]),100))]
    ingredient_target=jnp.where(jnp.any(reuse),reuse_target,_supply(pos,need_type))
    private_free=free_store&(jnp.arange(4)<2)
    private_dump=storage[jnp.argmin(jnp.where(private_free,jnp.abs(storage[:,1]-pos[1]),100))]
    duplicate=present[itype.clip(0,1)]
    offer_target=jnp.where(jnp.any(missing)&~duplicate,_COUNTERS[slot],jnp.where(jnp.any(private_free),private_dump,discard))
    raw_target=jnp.where(partial&(itype==ptype),_POTS[active],offer_target)
    need_plate=anypot&(ready|cooking)&~(cooking&(p[6]>11)&jnp.any(missing))
    empty_action=jnp.where(partial,_go(pos,facing,ingredient_target),
        jnp.where(need_plate,_go(pos,facing,jnp.asarray([0,3])),
        jnp.where(jnp.any(missing),_go(pos,facing,ingredient_target),_walk(pos,jnp.asarray([2,2])))))
    duplicates=(~jnp.any(missing))&(~jnp.all(present))&jnp.all(counters[:,0:2]==0)
    empty_action=jnp.where(~anypot&duplicates&jnp.any(private_free),_go(pos,facing,_COUNTERS[jnp.argmin(jnp.abs(_COUNTERS[:,1]-pos[1]))]),empty_action)
    action=jnp.where(cooked,_go(pos,facing,jnp.asarray([0,2])),
        jnp.where(plate,jnp.where(anypot,_go(pos,facing,_POTS[active]),_go(pos,facing,discard)),
        jnp.where(raw,_go(pos,facing,raw_target),empty_action)))
    passed_dish=(counters[:,0]>0)&(counters[:,1]>0)&(jnp.max(counters[:,2:4],axis=-1)==3)
    dish_target=_COUNTERS[jnp.argmax(passed_dish)]
    partner_inv=jnp.sum(obs[:,:,16:22],axis=(0,1))
    partner_dish=partner_inv[1]>0
    clear_handoff=partner_dish&~jnp.any(missing)
    clear_target=_COUNTERS[jnp.argmin(jnp.abs(_COUNTERS[:,1]-pos[1]))]
    action=jnp.where(~holding&clear_handoff,_go(pos,facing,clear_target),action)
    action=jnp.where(~holding&jnp.any(passed_dish),_go(pos,facing,dish_target),action)
    # When a partner is carrying soup, keep the receiving counter empty.
    private_free=free_store&(jnp.arange(4)<2)
    private_dump=storage[jnp.argmin(jnp.where(private_free,jnp.abs(storage[:,1]-pos[1]),100))]
    action=jnp.where(raw&partner_dish&jnp.any(private_free),_go(pos,facing,private_dump),action)
    possible_dish=(jnp.sum(inv[2:])==3)&(jnp.max(inv[2:4])==3)
    invalid_dish=cooked&~possible_dish
    action=jnp.where(invalid_dish,jnp.where(jnp.any(free_store),_go(pos,facing,jnp.where(jnp.any(private_free),private_dump,discard)),_walk(pos,jnp.asarray([2,2]))),action)
    delivered=cooked&(pos[0]==1)&(pos[1]==2)&(facing==0)&(action==5)
    action=jnp.where(mode&~anypot&empty_hands&(pending>=0),_walk(pos,jnp.asarray([2,2])),action)
    known=jnp.where(delivered,-1,known)
    pending=jnp.where(delivered&mode,0,pending)
    nextm={**m,'grid':grid,'known':known.astype(jnp.int32),'step':m['step']+1,
        'signature':signature.astype(jnp.int32),'stable':stable.astype(jnp.int32),'idle':idle.astype(jnp.int32),
        'physical_seen':physical_seen,'mode':mode,'belief':belief,'pending':pending.astype(jnp.int32),
        'last_signature':jnp.where(delivered&mode,signature.clip(0,63),m['last_signature']).astype(jnp.int32),
        'last_guess':jnp.where(delivered&mode,itype,m['last_guess']).astype(jnp.int32)}
    return action.astype(jnp.int32),nextm,{'pos':pos,'known':known,'active':active,'pot_counts':counts,'inventory':inv,'mode':mode,'signature':signature,'pending':pending,'belief':belief}

def agent_1_act(obs, reset, memory, rng):
    m,pos,grid,inv,facing=_observe(obs,reset,memory)
    recipe=jnp.sum(obs[:,:,40:44],axis=(0,1))
    known=jnp.where(jnp.sum(recipe)>0,jnp.argmax(recipe),m['known'])
    _POTS, _COUNTERS = _objects()
    pots=grid[_POTS[:,0],_POTS[:,1]]
    counters=grid[_COUNTERS[:,0],_COUNTERS[:,1]]
    counts=jnp.sum(pots[:,2:6],axis=-1)
    compatible=counts==pots[:,2+known.clip(0,3)]
    matching=compatible&(counts>0)
    score=jnp.where(compatible,(counts>0)*30+counts*3-jnp.abs(_POTS[:,1]-pos[1]),-100)
    active=jnp.argmax(score)
    p=pots[active]
    available=compatible[active]&(counts[active]<3)&(p[1]==0)&(p[6]==0)
    complete=jnp.any(matching&(counts==3))
    full_index=jnp.argmax(matching&(counts==3))
    raw=(inv[0]==0)&(jnp.sum(inv[2:])>0)
    itype=jnp.argmax(inv[2:])
    plate=(inv[0]>0)&(inv[1]==0)
    cooked=inv[1]>0
    storage=jnp.asarray([[3,2],[3,4],[6,1],[6,3],[6,4]],jnp.int32)
    stored=grid[storage[:,0],storage[:,1]]
    free_store=jnp.sum(stored[:,:6],axis=-1)==0
    public_free=free_store&(jnp.arange(5)<2)
    private_free=free_store&(jnp.arange(5)>=2)
    distance=jnp.abs(storage[:,0]-pos[0])+jnp.abs(storage[:,1]-pos[1])
    dump=storage[jnp.argmin(jnp.where(free_store,distance,100))]
    private_dump=storage[jnp.argmin(jnp.where(private_free,distance,100))]
    public_dump=storage[jnp.argmin(jnp.where(public_free,distance,100))]
    reuse=(stored[:,2+known.clip(0,3)]>0)&(stored[:,0]==0)&(stored[:,1]==0)
    source_target=storage[jnp.argmin(jnp.where(reuse,distance,100))]
    plates=(stored[:,0]>0)&(stored[:,1]==0)
    plate_target=storage[jnp.argmin(jnp.where(plates,distance,100))]
    # A position is useful only if the other player responds to it. Learn
    # that response from ingredient additions, never from a presumed code.
    item_partner=m['item_partner']|jnp.any(jnp.sum(counters[:,:6],axis=-1)>0)
    wait_stable=jnp.where((pos[0]==4)&(pos[1]==m['wait_column']),m['wait_stable']+1,0)
    previous_counts=jnp.sum(m['grid'][_POTS[:,0],_POTS[:,1],2:6],axis=-1)
    fresh=(previous_counts==0)&(counts==1)&(jnp.max(pots[:,2:4],axis=-1)==1)
    response=m['response']
    teach=(~item_partner)&(wait_stable>=4)&jnp.any(fresh)&(jnp.sum(inv)==0)
    observed_type=jnp.argmax(pots[jnp.argmax(fresh),2:4])
    column=(pos[1]-1).clip(0,3)
    response=response.at[column].set(jnp.where(teach,observed_type,response[column]))
    preferred=2*known.clip(0,1)
    alternative=2*(1-known.clip(0,1))
    matches=response==known
    unknown=response<0
    chosen=jnp.where(response[preferred]<0,preferred,jnp.where(response[alternative]<0,alternative,jnp.argmax(unknown)))
    chosen=jnp.where(jnp.any(matches),jnp.argmin(jnp.where(matches,jnp.abs(jnp.arange(4)-preferred),100)),chosen)
    chosen=jnp.where(item_partner,1,chosen)
    wait=_walk(pos,jnp.asarray([4,chosen+1]))
    empty_action=jnp.where(available&jnp.any(reuse),_go(pos,facing,source_target),wait)
    empty_action=jnp.where(complete&jnp.any(plates),_go(pos,facing,plate_target),empty_action)
    raw_action=jnp.where((itype==known)&available,_go(pos,facing,_POTS[active]),jnp.where(itype!=known,_go(pos,facing,dump),wait))
    plate_action=jnp.where(complete,_go(pos,facing,_POTS[full_index]),jnp.where(jnp.any(public_free),_go(pos,facing,public_dump),wait))
    dish_matches=(inv[2+known.clip(0,3)]==3)&(jnp.sum(inv[2:])==3)
    dish_action=jnp.where(dish_matches,jnp.where(jnp.any(public_free),_go(pos,facing,public_dump),wait),jnp.where(jnp.any(private_free),_go(pos,facing,private_dump),wait))
    action=jnp.where(cooked,dish_action,jnp.where(plate,plate_action,jnp.where(raw,raw_action,empty_action)))
    action=jnp.where(known<0,wait,action)
    nextm={**m,'grid':grid,'known':known.astype(jnp.int32),'step':m['step']+1,'response':response,'wait_column':pos[1].astype(jnp.int32),'wait_stable':wait_stable.astype(jnp.int32),'item_partner':item_partner}
    return action.astype(jnp.int32),nextm,{'pos':pos,'known':known,'active':active,'pot_counts':counts,'inventory':inv,'response':response,'item_partner':item_partner}
