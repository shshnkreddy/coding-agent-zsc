"""Coordinate through visible food and pot state in the divided kitchen."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'observable_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 11, 'num_ingredients': 3}

_BOARD = ('WWWWWR2W0WW', '0      W  B', 'W     APA X', '1      W  B', 'WWWWWR2W1WW')
def _geometry():
    _CELLS = [(r,c) for r in range(5) for c in range(11) if _BOARD[r][c] in ' A']
    _CHANNEL = {'W':0, 'X':1, 'P':2, 'R':3, 'L':4, 'B':5, '0':6, '1':7, '2':8}
    _TEMPLATES = jnp.asarray([[[[int(0 <= r+i-2 < 5 and 0 <= c+j-2 < 11 and _CHANNEL.get(_BOARD[r+i-2][c+j-2], -1) == k) for k in range(9)] for j in range(5)] for i in range(5)] for r,c in _CELLS], jnp.int32)
    _POSITIONS = jnp.asarray(_CELLS, jnp.int32)
    _COUNTERS = []
    for _r,_c in _CELLS:
        for _dr,_dc,_direction in [(0,1,2),(1,0,1),(0,-1,3),(-1,0,0)]:
            if _BOARD[_r+_dr][_c+_dc] == 'W':
                _COUNTERS.append((_r,_c,_direction,_r+_dr,_c+_dc))
    _COUNTERS = jnp.asarray(_COUNTERS,jnp.int32)
    return _TEMPLATES, _POSITIONS, _COUNTERS


def _init():
    return {'tick':jnp.int32(0), 'recipe':jnp.int32(-1), 'pot':jnp.zeros(5,jnp.int32), 'age':jnp.int32(0), 'cooldown':jnp.int32(0), 'items':jnp.zeros((5,11,5),jnp.int32), 'partner':jnp.zeros(5,jnp.int32), 'delivery_wait':jnp.int32(0), 'cue':jnp.int32(-1), 'cue_age':jnp.int32(0), 'cue_trials':jnp.int32(0), 'cue_map':jnp.int32(-1), 'cue_batch':jnp.int32(0), 'batch_cue':jnp.int32(-1), 'batch_flip':jnp.int32(0), 'last_cue':jnp.int32(-1), 'last_flip':jnp.int32(0), 'cue_pending':jnp.int32(0), 'stock':jnp.int32(-1), 'refresh':jnp.int32(0), 'last_inv':jnp.zeros(5,jnp.int32), 'announced':jnp.int32(-2), 'relay_kind':jnp.int32(-1), 'relay_partner':jnp.int32(0)}

def init_agent_0():
    return _init()

def init_agent_1():
    return _init()

def _decode(obs, reset, memory):
    mem = {k:jnp.where(reset,_init()[k],v) for k,v in memory.items()}
    _TEMPLATES, _POSITIONS, _COUNTERS = _geometry()
    errors = jnp.sum(jnp.abs(_TEMPLATES - obs[None,:,:,20:29]),axis=(1,2,3))
    pos = _POSITIONS[jnp.argmin(errors)]
    r,c = pos[0],pos[1]
    facing = jnp.argmax(obs[2,2,1:5]).astype(jnp.int32)
    inv = obs[2,2,5:10]
    p_mask = obs[:,:,22]
    visible = jnp.any(p_mask > 0)
    pot = jnp.concatenate((jnp.sum(obs[:,:,31:34]*p_mask[:,:,None],axis=(0,1)), jnp.asarray([jnp.sum(obs[:,:,30]*p_mask),jnp.sum(obs[:,:,39]*p_mask)])))
    pot = jnp.where(visible,pot,mem['pot'])
    counts = jnp.sum(pot[:3])
    age = jnp.where(jnp.any(pot[:4] != mem['pot'][:4]),0,mem['age']+1)
    cooldown = jnp.where((mem['pot'][3]>0)&(counts==0),3,jnp.maximum(0,mem['cooldown']-1))
    recipe_counts = jnp.max(obs[:,:,36:39],axis=(0,1))
    recipe = jnp.where(jnp.sum(recipe_counts)>0,jnp.argmax(recipe_counts),mem['recipe']).astype(jnp.int32)
    partner_visible = jnp.any(obs[:,:,10]>0)
    partner = jnp.where(partner_visible,jnp.sum(obs[:,:,15:20],axis=(0,1)),mem['partner'])
    cleared = (mem['pot'][3]>0)&(counts==0)
    delivery_wait = jnp.where(cleared,1,mem['delivery_wait'])
    # A disappearing pot is plating, not yet a delivery. Observe the carrier.
    delivered = ((recipe!=mem['recipe'])&(recipe>=0)) | ((age>=3)&partner_visible&(partner[1]==0)) | ((age>=12)&(~partner_visible))
    delivery_wait = jnp.where(delivered,0,delivery_wait).astype(jnp.int32)
    partner_index = jnp.argmax(obs[:,:,10])
    partner_row = r + partner_index // 5 - 2
    partner_col = c + partner_index % 5 - 2
    partner_direction = jnp.argmax(jnp.sum(obs[:,:,11:15],axis=(0,1)))
    cue = jnp.where(partner_row==1,0,jnp.where(partner_row==3,1,jnp.where(partner_direction==0,0,jnp.where(partner_direction==1,1,-1))))
    cue = jnp.where(partner_visible&(partner_col==6)&(jnp.sum(partner)==0),cue,-1).astype(jnp.int32)
    cue_age = jnp.where((cue>=0)&(cue==mem['cue']),mem['cue_age']+1,0)
    confirmed = (mem['cue_pending']>0)&(cue_age>=3)&(cue!=mem['last_cue'])&(counts==0)
    cue_map = jnp.where(confirmed,mem['last_flip'],mem['cue_map'])
    cue_pending = jnp.where(confirmed,0,mem['cue_pending'])
    # Cache only observed public counter contents.
    rr = r+jnp.arange(5)[:,None]-2
    cc = c+jnp.arange(5)[None,:]-2
    valid = (rr>=0)&(rr<5)&(cc>=0)&(cc<11)
    items = mem['items'].at[jnp.where(valid,rr,5),jnp.where(valid,cc,11)].set(obs[:,:,29:34],mode='drop')
    nxt = {'tick':mem['tick']+1,'recipe':recipe,'pot':pot,'age':age,'cooldown':cooldown,'items':items,'partner':partner,'delivery_wait':delivery_wait,'cue':cue,'cue_age':cue_age, 'cue_map':cue_map, 'cue_pending':cue_pending, 'stock':mem['stock'], 'refresh':mem['refresh'], 'last_inv':inv, 'announced':jnp.where(cleared,-1,mem['announced']), 'relay_kind':jnp.where(cleared,-1,mem['relay_kind']), 'relay_partner':mem['relay_partner'], **{k:mem[k] for k in ('cue_trials','cue_batch','batch_cue','batch_flip','last_cue','last_flip')}}
    return r,c,facing,inv,pot,visible,nxt

def _go(r,c,facing,tr,tc,td,interact=True):
    direction_action = jnp.asarray([3,1,0,2],jnp.int32)[td]
    at = (r==tr)&(c==tc)
    horizontal = jnp.where(c<tc,0,2)
    vertical = jnp.where(r<tr,1,3)
    # Arrange the last translation to face the interaction when possible.
    horiz_last = ((td==2)&(c<tc)) | ((td==3)&(c>tc))
    move = jnp.where(horiz_last & (r!=tr),vertical,jnp.where(c!=tc,horizontal,vertical))
    return jnp.where(at,jnp.where(facing==td,jnp.where(interact,5,4),direction_action),move).astype(jnp.int32)

def _interaction_cost(r,c,facing,tr,tc,td):
    dx,dy = tc-c,tr-r
    distance = jnp.abs(dx)+jnp.abs(dy)
    arrives_facing = ((td==2)&(dx>0))|((td==3)&(dx<0))|((td==0)&(dy<0))|((td==1)&(dy>0))
    turn = jnp.where(distance==0,facing!=td,~arrives_facing)
    return distance+turn.astype(jnp.int32)+1

def _counter_source(r,c,d,items,seat,kind,want_plate=False):
    _,_,counters = _geometry()
    in_room = (counters[:,1]<7) if seat==0 else (counters[:,1]>7)
    if seat==0:
        in_room = in_room&((counters[:,1]<6)|(counters[:,0]==jnp.where(kind==0,1,3)))
    contents = items[counters[:,3],counters[:,4]]
    match_raw = (contents[:,0]==0)&(contents[:,1]==0)&(contents[:,2+jnp.clip(kind,0,2)]>0)
    match_plate = (contents[:,0]>0)&(contents[:,1]==0)
    valid = in_room & jnp.where(want_plate,match_plate,match_raw)
    cost = _interaction_cost(r,c,d,counters[:,0],counters[:,1],counters[:,2])
    cost = jnp.where(valid,cost,999)
    idx = jnp.argmin(cost)
    ctr = counters[idx]
    return _go(r,c,d,ctr[0],ctr[1],ctr[2]),cost[idx]


def _counter_space(items,seat,avoid_shared=False):
    _,_,counters = _geometry()
    in_room = ((counters[:,1]<7)&(counters[:,4]<=4)) if seat==0 else (counters[:,1]>7)
    in_room = in_room&((counters[:,4]!=7)|(~jnp.asarray(avoid_shared)))
    empty = jnp.sum(items[counters[:,3],counters[:,4]],axis=1)==0
    return jnp.any(in_room&empty)


def _counter_action(r,c,facing,items,seat,avoid_shared=False):
    _, _, _COUNTERS = _geometry()
    same_room = ((_COUNTERS[:,1]<7)&(_COUNTERS[:,4]<=4)) if seat==0 else (_COUNTERS[:,1]>7)
    same_room = same_room&((_COUNTERS[:,4]!=7)|(~jnp.asarray(avoid_shared)))
    empty = jnp.sum(items[_COUNTERS[:,3],_COUNTERS[:,4]],axis=1)==0
    cost = jnp.abs(_COUNTERS[:,0]-r)+jnp.abs(_COUNTERS[:,1]-c)+(facing!=_COUNTERS[:,2]).astype(jnp.int32)
    idx = jnp.argmin(jnp.where(same_room&empty,cost,999))
    ctr = _COUNTERS[idx]
    return jnp.where(jnp.any(same_room&empty),_go(r,c,facing,ctr[0],ctr[1],ctr[2]),4).astype(jnp.int32)

def agent_0_act(obs, reset, memory, rng):
    r,c,d,inv,pot,visible,nxt = _decode(obs,reset,memory)
    recipe = nxt['recipe']
    n = jnp.sum(pot[:3])
    raw = (jnp.sum(inv[2:])>0)&(inv[1]==0)
    held = jnp.argmax(inv[2:])
    empty = jnp.sum(inv)==0
    active = (pot[3]>0)|(pot[4]>0)|(n>=3)
    ready_for_seed = (nxt['cooldown']==0)&(~active)&(nxt['delivery_wait']==0)
    partner_plate = (nxt['partner'][0]>0)&(nxt['partner'][1]==0)
    should_fetch = ready_for_seed & ((n==0)|(nxt['age']>=10)|partner_plate)
    row = jnp.where(recipe==0,1,3)
    direction = jnp.where(recipe==0,0,1)
    announce_now = (n==0)&(r==row)&(c==6)&(recipe>=0)&(nxt['delivery_wait']==0)&(nxt['cooldown']==0)
    announced = jnp.where(announce_now,recipe,nxt['announced'])
    nxt['announced'] = announced
    signal_ready = (n>0)|(announced==recipe)|(announced==-2)
    should_fetch = should_fetch&signal_ready
    ready_for_seed = ready_for_seed&signal_ready
    home = _go(r,c,d,jnp.where(recipe<0,2,row),6,jnp.where(recipe<0,2,direction),False)
    home = jnp.where((c<6)&(r!=jnp.where(recipe<0,2,row)),jnp.where(r<jnp.where(recipe<0,2,row),1,3),home).astype(jnp.int32)
    supply = _go(r,c,d,row,1,3)
    deposit = _go(r,c,d,2,6,2)
    pantry = nxt['items'][jnp.asarray([0,4]),4]
    available = jnp.sum(pantry,axis=1)==0
    counter_fetch,counter_cost = _counter_source(r,c,d,nxt['items'],0,recipe)
    fetch = jnp.where(counter_cost<=_interaction_cost(r,c,d,row,1,3),counter_fetch,supply)

    stock_done = (nxt['stock']>=0)&empty&(jnp.sum(jnp.where(reset,jnp.zeros(5,jnp.int32),memory['last_inv'])[2:])>0)
    stock = jnp.where(stock_done,-1,nxt['stock'])
    refresh = jnp.where(stock_done,1,nxt['refresh'])
    refresh = jnp.where(visible,0,refresh)
    partner_filling = (nxt['partner'][0]==0)&(jnp.sum(nxt['partner'][2:])>0)
    spare_time = (pot[4]>=14) | ((n>0)&(n<3)&partner_filling&(nxt['age']<8))
    start_stock = empty&(stock<0)&(refresh==0)&visible&spare_time&jnp.any(available)
    preferred_stock = jnp.where((r==3)&available[1],1,jnp.where(available[0],0,1))
    stock = jnp.where(start_stock,preferred_stock,stock)
    stock = jnp.where(raw&active&(held<2)&available[jnp.minimum(held,1)],held,stock)
    sr = jnp.where(stock==0,1,3)
    sd = jnp.where(stock==0,0,1)
    stock_action = jnp.where(empty,_go(r,c,d,sr,1,3),_go(r,c,d,sr,4,sd))

    action = jnp.where(empty&should_fetch&(recipe>=0),fetch,home)
    action = jnp.where(raw&(held==recipe)&ready_for_seed,deposit,action)
    stash_held = _go(r,c,d,jnp.where(held==0,1,3),4,jnp.where(held==0,0,1))
    disposal = jnp.where((held<2)&available[jnp.minimum(held,1)]&raw,stash_held,_counter_action(r,c,d,nxt['items'],0))
    action = jnp.where((raw&(held!=recipe))|(inv[0]>0),disposal,action)
    action = jnp.where(stock>=0,stock_action,action)
    action = jnp.where((refresh>0)&empty,home,action)
    action = jnp.where(recipe<0,home,action)
    nxt['stock'] = stock.astype(jnp.int32)
    nxt['refresh'] = refresh.astype(jnp.int32)
    trace = {'pos':jnp.asarray([r,c]),'recipe':recipe,'pot':pot,'inv':inv,'age':nxt['age'],'cooldown':nxt['cooldown'],'stock':stock,'refresh':refresh,'pantry':pantry}
    return action,nxt,trace


def agent_1_act(obs, reset, memory, rng):
    r,c,d,inv,pot,visible,nxt = _decode(obs,reset,memory)
    n = jnp.sum(pot[:3])
    raw = (jnp.sum(inv[2:])>0)&(inv[1]==0)
    empty = jnp.sum(inv)==0
    confirmed_mapping = nxt['cue_map']>=0
    cue_stability = jnp.where(confirmed_mapping,1,6)
    cue_delay = jnp.where(confirmed_mapping,3,jnp.where(nxt['cue_trials']>0,8,12))
    cue_ready = (nxt['cue_age']>=cue_stability)&(nxt['age']>=cue_delay)&(nxt['cooldown']==0)
    flip = jnp.where(nxt['cue_map']>=0,nxt['cue_map'],nxt['cue_trials']%2)
    cue_kind = jnp.bitwise_xor(jnp.maximum(nxt['cue'],0),flip)
    partner_raw = jnp.sum(obs[:,:,17:19],axis=(0,1))
    partner_row = r+jnp.sum(obs[:,:,10]*jnp.arange(5)[:,None])-2
    incoming = (jnp.sum(partner_raw)>0)&(jnp.sum(obs[:,:,16])==0)&(partner_row==2)&jnp.any(obs[:,:,13]>0)&(nxt['cooldown']==0)
    shared = nxt['items'][jnp.asarray([1,3]),7]
    old_shared = jnp.where(reset,jnp.zeros((2,5),jnp.int32),memory['items'][jnp.asarray([1,3]),7])
    old_inv = jnp.where(reset,jnp.zeros(5,jnp.int32),memory['last_inv'])
    own_drop = (jnp.sum(old_inv)>0)&empty
    appeared = (jnp.sum(shared,axis=1)>0)&(jnp.sum(old_shared,axis=1)==0)&(~own_drop)
    nxt['relay_partner'] = jnp.maximum(nxt['relay_partner'],jnp.any(appeared).astype(jnp.int32))
    usable = appeared&(shared[:,0]==0)&(shared[:,1]==0)&(shared[:,4]==0)&(jnp.sum(shared[:,2:4],axis=1)==1)
    fresh_relay = jnp.any(usable)&(n==0)&(inv[1]==0)&(nxt['cooldown']==0)&(nxt['age']>=3)
    relay_kind = jnp.argmax(shared[jnp.argmax(usable),2:4]).astype(jnp.int32)
    nxt['relay_kind'] = jnp.where(fresh_relay,relay_kind,nxt['relay_kind'])
    relay_ready = (nxt['relay_kind']>=0)&(nxt['cooldown']==0)
    kind = jnp.where(n>0,jnp.argmax(pot[:3]),jnp.where(incoming,jnp.argmax(partner_raw),jnp.where(relay_ready,nxt['relay_kind'],cue_kind)))
    active = (pot[3]>0)|(pot[4]>0)|(n>=3)
    home = _go(r,c,d,2,8,3,False)
    supply = _go(r,c,d,jnp.where(kind==0,1,3),8,jnp.where(kind==0,0,1))
    plate = _go(r,c,d,jnp.where(r==3,3,1),9,2)
    deposit = _go(r,c,d,2,8,3)
    delivery = _go(r,c,d,2,9,2)
    counter_fetch,counter_cost = _counter_source(r,c,d,nxt['items'],1,kind)
    source_cost = _interaction_cost(r,c,d,jnp.where(kind==0,1,3),8,jnp.where(kind==0,0,1))
    supply = jnp.where((counter_cost<=source_cost)|((n==0)&relay_ready&(counter_cost<999)),counter_fetch,supply)
    counter_plate,counter_plate_cost = _counter_source(r,c,d,nxt['items'],1,0,True)
    plate_cost = _interaction_cost(r,c,d,jnp.where(r==3,3,1),9,2)
    plate = jnp.where(counter_plate_cost<=plate_cost,counter_plate,plate)
    stock_done = (nxt['stock']>=0)&empty&(jnp.sum(jnp.where(reset,jnp.zeros(5,jnp.int32),memory['last_inv'])[2:])>0)
    stock = jnp.where(stock_done,-1,nxt['stock'])
    shared = nxt['items'][jnp.asarray([1,3]),7]
    vacant = jnp.sum(shared,axis=1)==0
    start_stock = empty&(stock<0)&(pot[4]>=10)&jnp.any(vacant)&(nxt['relay_partner']==0)
    preferred_stock = jnp.where((r==3)&vacant[1],1,jnp.where(vacant[0],0,1))
    stock = jnp.where(start_stock,preferred_stock,stock)
    held = jnp.argmax(inv[2:])
    stock = jnp.where(raw&active&(held<2)&vacant[jnp.minimum(held,1)]&(nxt['relay_partner']==0),held,stock)
    stock = jnp.where((nxt['relay_partner']>0)&empty,-1,stock)
    stock = jnp.where((stock>=0)&raw&(~vacant[jnp.maximum(stock,0)]),-1,stock)
    stock_row = jnp.where(stock==0,1,3)
    stock_action = jnp.where(empty,_go(r,c,d,stock_row,8,jnp.where(stock==0,0,1)),_go(r,c,d,stock_row,8,3))
    action = home
    action = jnp.where(empty & ((n>0)|incoming|relay_ready|cue_ready)&(~active),supply,action)
    action = jnp.where(empty & active,plate,action)
    pot_possible = (pot[2]==0)&((pot[0]==0)|(pot[1]==0))
    conflict = (n>0)&pot_possible&(pot[jnp.argmax(inv[2:])]==0)
    free_counter = _counter_space(nxt['items'],1,nxt['relay_partner']>0)
    raw_action = jnp.where(conflict&free_counter,_counter_action(r,c,d,nxt['items'],1,nxt['relay_partner']>0),deposit)
    action = jnp.where(raw & (~active),raw_action,action)
    action = jnp.where(raw & active,_counter_action(r,c,d,nxt['items'],1,nxt['relay_partner']>0),action)
    action = jnp.where((inv[0]>0)&(inv[1]==0),jnp.where(pot[3]>0,deposit,home),action)
    malformed = (jnp.sum(inv[2:])!=3)|(inv[4]>0)|((inv[2]>0)&(inv[3]>0))
    discard = _counter_action(r,c,d,nxt['items'],1,nxt['relay_partner']>0)
    dish_action = jnp.where(malformed&free_counter,discard,delivery)
    action = jnp.where(inv[1]>0,dish_action,action)
    action = jnp.where(stock>=0,stock_action,action)
    nxt['stock'] = stock.astype(jnp.int32)
    cue_start = empty&(n==0)&cue_ready&(~incoming)&(~relay_ready)&(nxt['cue_batch']==0)
    nxt['cue_batch'] = jnp.where(cue_start,1,nxt['cue_batch'])
    nxt['batch_cue'] = jnp.where(cue_start,nxt['cue'],nxt['batch_cue'])
    nxt['batch_flip'] = jnp.where(cue_start,flip,nxt['batch_flip'])
    cue_delivery = (inv[1]>0)&(~malformed)&(action==5)&(r==2)&(c==9)&(nxt['cue_batch']>0)
    nxt['cue_trials'] = nxt['cue_trials']+cue_delivery.astype(jnp.int32)
    nxt['last_cue'] = jnp.where(cue_delivery,nxt['batch_cue'],nxt['last_cue'])
    nxt['last_flip'] = jnp.where(cue_delivery,nxt['batch_flip'],nxt['last_flip'])
    nxt['cue_pending'] = jnp.where(cue_delivery,1,nxt['cue_pending'])
    nxt['cue_batch'] = jnp.where(cue_delivery|((inv[1]>0)&malformed&(action==5)),0,nxt['cue_batch'])
    trace = {'relay_kind':nxt['relay_kind'],'relay_partner':nxt['relay_partner'],'stock':stock,'incoming':incoming,'shared':shared,'cue':nxt['cue'],'cue_age':nxt['cue_age'],'cue_map':nxt['cue_map'],'cue_trials':nxt['cue_trials'],'pos':jnp.asarray([r,c]),'recipe':kind,'pot':pot,'inv':inv,'age':nxt['age'],'cooldown':nxt['cooldown']}
    return action,nxt,trace
