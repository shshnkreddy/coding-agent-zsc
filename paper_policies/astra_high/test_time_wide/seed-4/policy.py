"""Observable ingredient exchange and cooperative kitchen control."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'observable_adaptive_cooperator', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}


def _static_map():
    board = ('WWXBWW','0 A  0','1    1','WPWPWW','3 A  3','W    W','WWRWWW')
    channels = {'W':0,'X':1,'P':2,'R':3,'L':4,'B':5,'0':6,'1':7,'2':8,'3':9}
    return jnp.array([[[int(channels.get(c,-1)==k) for k in range(10)] for c in row] for row in board],jnp.int32)


def _init():
    return {'grid':jnp.zeros((7,6,45),jnp.int32), 'recipe':jnp.int32(-1), 'pot':jnp.int32(0), 'last_delivery':jnp.int32(0), 'time':jnp.int32(0), 'cooldown':jnp.int32(0), 'was_full':jnp.int32(0), 'unknown':jnp.int32(0), 'signal_col':jnp.int32(-1), 'signal_age':jnp.int32(0), 'spatial':jnp.int32(0), 'source':jnp.int32(0), 'cue_seen':jnp.int32(-1), 'cue_cycle':jnp.int32(-1), 'cue_last':jnp.int32(-1), 'served_last':jnp.int32(-1), 'source_last':jnp.int32(0), 'cue_runs':jnp.int32(0), 'cue_table':jnp.full((256,),-1,jnp.int32), 'clearing':jnp.int32(0), 'delegated':jnp.int32(0), 'signal_flip':jnp.int32(0), 'signal_pending':jnp.int32(0), 'signal_calibrated':jnp.int32(0), 'token_seen':jnp.int32(0), 'token_mode':jnp.int32(0), 'token_recipe':jnp.int32(-1)}


def init_agent_0(): return _init()
def init_agent_1(): return _init()


def _observe(obs, reset, memory, seat):
    m=jax.tree.map(lambda a,b:jnp.where(reset,a,b),_init(),memory)
    positions=jnp.array([(r,c) for r in ((1,2) if seat==0 else (4,5)) for c in range(1,5)],jnp.int32)
    offsets=jnp.stack(jnp.meshgrid(jnp.arange(5)-2,jnp.arange(5)-2,indexing='ij'),-1)
    loc=positions[:,None,None,:]+offsets[None,:,:,:]
    valid=(loc[...,0]>=0)&(loc[...,0]<7)&(loc[...,1]>=0)&(loc[...,1]<6)
    expected=_static_map()[jnp.clip(loc[...,0],0,6),jnp.clip(loc[...,1],0,5)]*valid[...,None]
    match=jnp.sum(jnp.abs(expected-obs[None,:,:,22:32]),axis=(1,2,3))
    pos=positions[jnp.argmin(match)]
    rr=jnp.arange(7)[:,None]-pos[0]+2
    cc=jnp.arange(6)[None,:]-pos[1]+2
    visible=(rr>=0)&(rr<5)&(cc>=0)&(cc<5)
    patch=obs[jnp.clip(rr,0,4),jnp.clip(cc,0,4)]
    grid=jnp.where(visible[:,:,None],patch,m['grid'])
    m['grid']=grid
    m['time']+=1
    return m,pos,grid,obs[2,2,5:11],jnp.argmax(obs[2,2,1:5])


def _go(pos,facing,target,interact=True):
    dr=target[0]-pos[0]; dc=target[1]-pos[1]
    # All rooms are rectangles. Targets are floor cells or adjacent objects.
    adjacent=(jnp.abs(dr)+jnp.abs(dc))==1
    want_dir=jnp.where(dr<0,0,jnp.where(dr>0,1,jnp.where(dc>0,2,3)))
    direction_action=jnp.array([3,1,0,2],jnp.int32)[want_dir]
    # Move horizontally first unless target lies on a side wall.
    action=jnp.where(dc>0,0,jnp.where(dc<0,2,jnp.where(dr>0,1,jnp.where(dr<0,3,4))))
    side=(target[1]==0)|(target[1]==5)
    action=jnp.where(side&(dr!=0),jnp.where(dr>0,1,3),action)
    if interact:
        action=jnp.where(adjacent,jnp.where(facing==want_dir,5,direction_action),action)
    return jnp.int32(action)


def _counter(grid,pos,kind=-1,empty=False,seat=0):
    rr=jnp.arange(7)[:,None];cc=jnp.arange(6)[None,:]
    wall=_static_map()[:,:,0]>0
    allowed=wall&((rr==3)|((rr==0) if seat==0 else (rr==6)))&(cc>=1)&(cc<=4)
    contents=grid[:,:,32:38]
    if empty: eligible=jnp.sum(contents,-1)==0
    else: eligible=(contents[:,:,2+jnp.maximum(kind,0)]>0)&(contents[:,:,1]==0)&(contents[:,:,0]==0)
    cost=jnp.abs(rr-pos[0])+jnp.abs(cc-pos[1])
    if seat==1 and not empty:
        cost=cost-jnp.where(rr==6,3,0)
    index=jnp.argmin(jnp.where(allowed&eligible,cost,1000))
    return jnp.array([index//6,index%6]),jnp.any(allowed&eligible)


def _item_counter(grid,pos,mode,seat=0,band='all',recipe=-1):
    rr=jnp.arange(7)[:,None];cc=jnp.arange(6)[None,:]
    private_row=0 if seat==0 else 6
    rows=(rr==3) if band=='shared' else ((rr==private_row) if band=='private' else ((rr==3)|(rr==private_row)))
    allowed=(_static_map()[:,:,0]>0)&rows&(cc>=1)&(cc<=4)
    item=grid[:,:,32:38]
    if mode=='empty': eligible=jnp.sum(item,-1)==0
    elif mode=='plate': eligible=(item[:,:,0]>0)&(item[:,:,1]==0)
    elif mode=='raw': eligible=(jnp.sum(item[:,:,2:],-1)>0)&(item[:,:,0]==0)&(item[:,:,1]==0)
    else:
        eligible=(item[:,:,0]>0)&(item[:,:,1]>0)&((item[:,:,2]==3)|(item[:,:,3]==3))
        eligible=eligible&((recipe<0)|(item[:,:,2+jnp.maximum(recipe,0)]==3))
    cost=jnp.abs(rr-pos[0])+jnp.abs(cc-pos[1])
    index=jnp.argmin(jnp.where(allowed&eligible,cost,1000))
    return jnp.array([index//6,index%6]),jnp.any(allowed&eligible)


def agent_0_act(obs,reset,memory,rng):
    m,pos,g,inv,face=_observe(obs,reset,memory,0)
    pots=g[3,jnp.array([1,3])]
    counts=jnp.sum(pots[:,34:38],-1)
    full=(counts==3)
    pi=jnp.argmax(jnp.sum(obs[:,:,18:20],axis=(0,1)))
    partner_raw=(jnp.sum(obs[:,:,18:20])>0)&(jnp.sum(obs[:,:,16:18])==0)
    valid_pots=(counts>0)&(jnp.sum(pots[:,36:38],-1)==0)&((pots[:,34]==0)|(pots[:,35]==0))
    evidence=jnp.any(valid_pots)
    pot_idx=jnp.argmax(jnp.where(valid_pots,counts,-1))
    kind=jnp.argmax(pots[pot_idx,34:36])
    recipe=jnp.where(m['last_delivery']>0,-1,m['recipe'])
    partner_visible=jnp.sum(obs[:,:,11])>0
    partner_local=jnp.argmax(obs[:,:,11].reshape(-1))
    partner_col=pos[1]+partner_local%5-2
    partner_face=jnp.argmax(jnp.sum(obs[:,:,12:16],axis=(0,1)))
    partner_token=jnp.sum(obs[:,:,21])>0
    counter_token=(g[3,2,37]>0).astype(jnp.int32)+2*(g[3,4,37]>0).astype(jnp.int32)
    cue=jnp.where(partner_visible,partner_col*8+partner_face*2+partner_token.astype(jnp.int32),48)+64*counter_token
    scan_valid=(pos[0]==2)&(partner_visible|(pos[1]==2)|(pos[1]==3))
    signal_age=jnp.where(m['last_delivery']>0,0,m['signal_age'])
    signal_age=jnp.where(scan_valid,jnp.where(cue==m['cue_seen'],signal_age+1,1),0)
    unknown=jnp.where(m['last_delivery']>0,0,m['unknown'])
    unknown=jnp.where(recipe<0,unknown+1,0)
    token_seen=jnp.maximum(m['token_seen'],(partner_token|(counter_token>0)).astype(jnp.int32))
    m['token_seen']=token_seen
    spatial_ready=scan_valid&(signal_age>=3)&(unknown>=jnp.where(m['spatial']>0,3,20))&((token_seen==0)|partner_visible|(signal_age>=12))
    use_spatial=(recipe<0)&spatial_ready&(~partner_raw)&(~evidence)
    # A stable changed cue after delivery identifies both old and new recipes.
    changed=use_spatial&(m['source_last']==1)&(m['cue_last']>=0)&(cue!=m['cue_last'])
    table=m['cue_table']
    old=jnp.maximum(m['cue_last'],0)
    table=table.at[old].set(jnp.where(changed,m['served_last'],table[old]))
    table=table.at[cue].set(jnp.where(changed,1-m['served_last'],table[cue]))
    runs=jnp.where(cue==m['cue_last'],m['cue_runs']+1,0)
    intuitive=jnp.where(partner_visible,((partner_col>=3)|partner_token|(counter_token>0)).astype(jnp.int32),1)
    guess=jnp.where((runs>=2)&(m['served_last']>=0),1-m['served_last'],intuitive)
    decoded=jnp.where(table[cue]>=0,table[cue],guess)
    recipe=jnp.where(use_spatial,decoded,recipe)
    source=jnp.where(m['last_delivery']>0,0,m['source'])
    source=jnp.where(use_spatial,1,source)
    recipe=jnp.where(((recipe<0)|(source==1))&partner_raw,pi,recipe)
    source=jnp.where(partner_raw,2,source)
    m['cue_table']=table
    m['cue_runs']=jnp.where(use_spatial,runs,m['cue_runs']).astype(jnp.int32)
    steady_cycle_cue=(source==1)&(recipe>=0)&(signal_age>=3)&scan_valid
    m['cue_cycle']=jnp.where(use_spatial|steady_cycle_cue,cue,m['cue_cycle']).astype(jnp.int32)
    m['cue_seen']=jnp.where(scan_valid,cue,m['cue_seen']).astype(jnp.int32)
    m['source']=source.astype(jnp.int32)
    m['unknown']=unknown.astype(jnp.int32)
    m['signal_col']=jnp.where(partner_visible,partner_col,m['signal_col']).astype(jnp.int32)
    m['signal_age']=signal_age.astype(jnp.int32)
    m['spatial']=jnp.maximum(m['spatial'],use_spatial.astype(jnp.int32))
    recipe=jnp.where((recipe<0)&evidence,kind,recipe)
    compatible=(counts==0)|((pots[:,34+jnp.maximum(recipe,0)]==counts)&(recipe>=0))
    preferred=jnp.where(partner_raw,(partner_col>=3).astype(jnp.int32),m['pot'])
    chosen=jnp.argmin(jnp.where(compatible, -10*counts+(jnp.arange(2)!=preferred),1000))
    p=pots[chosen]; n=counts[chosen]
    pot_target=jnp.array([3,jnp.where(chosen==0,1,3)])
    raw=jnp.sum(inv[2:])>0
    empty=jnp.sum(inv)==0
    held_kind=jnp.argmax(inv[2:])
    plate=(inv[0]>0)&(inv[1]==0)
    dish=inv[1]>0
    # Default probe position keeps both pots and the central transfer surface in view.
    action=_go(pos,face,jnp.array([2,2]),False)
    shared_free,shared_space=_item_counter(g,pos,'empty',seat=0,band='shared')
    offers=jnp.sum(g[3,:,34:36]*_static_map()[3,:,0,None],axis=0)
    offer_kind=jnp.where(offers[0]==0,0,1)
    offer_target=jnp.array([3,jnp.where(held_kind==0,2,4)])
    preparing=(n==3)&(p[44]>12)&jnp.any(offers==0)&shared_space
    supply_kind=jnp.where((recipe>=0)&(~preparing),recipe,offer_kind)
    supply_target=jnp.array([1+supply_kind,jnp.where(pos[1]<=2,0,5)])
    need_offer=jnp.any(offers==0)&shared_space
    action=jnp.where(empty&(((recipe<0)&need_offer)|((recipe>=0)&(n<3)&((~partner_raw)|(n<2)))),_go(pos,face,supply_target),action)
    # A carried ingredient may become unwanted after the partner supplies evidence.
    target_free=jnp.sum(g[offer_target[0],offer_target[1],32:38])==0
    spare,_=_counter(g,pos,empty=True)
    drop=jnp.where(target_free,offer_target,jnp.where(shared_space,shared_free,spare))
    can_load=(recipe>=0)&(held_kind==recipe)&(n<3)&(p[44]==0)&(p[33]==0)&compatible[chosen]
    action=jnp.where(raw,_go(pos,face,jnp.where(can_load,pot_target,drop)),action)
    action=jnp.where(empty&(n==3),_go(pos,face,jnp.where(preparing,supply_target,jnp.array([0,3]))),action)
    action=jnp.where(plate,_go(pos,face,pot_target),action)
    dish_counter,dish_available=_item_counter(g,pos,'dish',seat=0)
    action=jnp.where(empty&dish_available,_go(pos,face,dish_counter),action)
    action=jnp.where(dish,_go(pos,face,jnp.array([0,2])),action)
    delivered=dish&(action==5)&(pos[0]==1)&(pos[1]==2)&(face==0)
    m['recipe']=recipe.astype(jnp.int32);m['pot']=chosen.astype(jnp.int32);m['last_delivery']=delivered.astype(jnp.int32)
    m['cue_last']=jnp.where(delivered,m['cue_cycle'],m['cue_last']).astype(jnp.int32)
    m['served_last']=jnp.where(delivered,held_kind,m['served_last']).astype(jnp.int32)
    m['source_last']=jnp.where(delivered,source,m['source_last']).astype(jnp.int32)
    return action,m,{'pos':pos,'recipe':recipe,'pot':chosen,'counts':counts,'source':source,'cue':cue,'cue_runs':m['cue_runs'],'intuitive':intuitive,'decoded':decoded,'table_cue':table[cue]}


def agent_1_act(obs,reset,memory,rng):
    m,pos,g,inv,face=_observe(obs,reset,memory,1)
    recipe=jnp.argmax(g[6,2,40:42])
    known=jnp.sum(g[6,2,40:44])>0
    pots=g[3,jnp.array([1,3])]
    counts=jnp.sum(pots[:,34:38],-1)
    correct=(pots[:,34+recipe]==counts)
    costs=jnp.abs(jnp.array([1,3])-pos[1])+jnp.where(counts>0,-5,0)
    eligible=(counts<3)&correct
    chosen=jnp.argmin(jnp.where(eligible,costs,1000))
    pot_target=jnp.array([3,jnp.where(chosen==0,1,3)])
    raw=(jnp.sum(inv[2:])>0)&(inv[1]==0)
    empty=jnp.sum(inv)==0
    held_kind=jnp.argmax(inv[2:])
    counter,available=_counter(g,pos,recipe,seat=1)
    full=jnp.any(counts==3)
    cooldown=jnp.maximum(0,m['cooldown']-1)
    cooldown=jnp.where((m['was_full']>0)&(~full),5,cooldown)
    delegated=jnp.maximum(m['delegated'],inv[1]).astype(jnp.int32)
    dish_in_transfer=(inv[1]>0)|jnp.any((g[3,:,32]>0)&(g[3,:,33]>0))|(jnp.sum(obs[:,:,17])>0)
    cooldown=jnp.where((delegated>0)&dish_in_transfer,5,cooldown)
    m['delegated']=delegated
    semantic_stock=jnp.sum(g[3,:,34:36]*_static_map()[3,:,0,None])>0
    wrong_full=jnp.any((counts==3)&(pots[:,34+recipe]==0)&(jnp.sum(pots[:,36:38],-1)==0))
    correct_started=jnp.any((counts>0)&correct)
    mismatch=wrong_full&(~correct_started)&(~semantic_stock)&(m['signal_calibrated']==0)
    try_token=wrong_full&(~correct_started)&(~semantic_stock)&(m['signal_flip']>0)&(m['token_mode']==0)
    token_mode=jnp.maximum(m['token_mode'],try_token.astype(jnp.int32))
    token_recipe=jnp.where(try_token,recipe,m['token_recipe'])
    m['token_mode']=token_mode.astype(jnp.int32)
    m['token_recipe']=token_recipe.astype(jnp.int32)
    pending=jnp.maximum(m['signal_pending'],mismatch.astype(jnp.int32))
    change_signal=(pending>0)&(m['was_full']>0)&(~full)
    flip=jnp.where(change_signal,1-m['signal_flip'],m['signal_flip'])
    m['signal_flip']=flip.astype(jnp.int32)
    m['signal_pending']=jnp.where(change_signal,0,pending).astype(jnp.int32)
    m['signal_calibrated']=jnp.maximum(m['signal_calibrated'],mismatch.astype(jnp.int32))
    signal_target=jnp.array([4,jnp.where((recipe ^ flip)==0,1,3)])
    action=_go(pos,face,signal_target,False)
    action=jnp.where(jnp.all(pos==signal_target)&(face!=0),3,action)
    transfer_evidence=jnp.any(counts>0)|(jnp.sum(g[3,2,32:38])>0)
    scan=(pos[1]==1)&((~available)|((delegated>0)&dish_in_transfer))&transfer_evidence&((m['time']%8)==0)
    action=jnp.where(scan,0,action)
    action=jnp.where(empty&known&available&jnp.any(eligible)&(~full)&(cooldown==0),_go(pos,face,counter),action)
    spare,_=_counter(g,pos,empty=True,seat=1)
    action=jnp.where(raw,_go(pos,face,jnp.where((held_kind==recipe)&jnp.any(eligible),pot_target,spare)),action)
    action=jnp.where(~known,_go(pos,face,jnp.array([4,2]),False),action)
    plate=(inv[0]>0)&(inv[1]==0)
    dish=inv[1]>0
    full_correct=(counts==3)&correct
    ready_choice=jnp.argmin(jnp.where(full_correct,jnp.abs(jnp.array([1,3])-pos[1]),1000))
    ready_target=jnp.array([3,jnp.where(ready_choice==0,1,3)])
    plate_counter,has_plate=_item_counter(g,pos,'plate',seat=1)
    private_free,private_space=_item_counter(g,pos,'empty',seat=1,band='private')
    shared_free,shared_space=_item_counter(g,pos,'empty',seat=1,band='shared')
    stored_dish,has_stored_dish=_item_counter(g,pos,'dish',seat=1,band='private',recipe=recipe)
    cleanup_raw,has_cleanup_raw=_item_counter(g,pos,'raw',seat=1,band='shared')
    needs_plate_space=jnp.any(full_correct&(pots[:,33]>0))&(~has_plate)&(jnp.sum(obs[:,:,16:18])==0)
    shared_dish=jnp.any((g[3,:,32]>0)&(g[3,:,33]>0))
    partner_raw_blocked=(jnp.sum(obs[:,:,18:22])>0)&(jnp.sum(obs[:,:,16:18])==0)
    handoff_blocked=shared_dish&partner_raw_blocked
    cleanup=empty&(has_stored_dish|needs_plate_space|handoff_blocked)&(~shared_space)&has_cleanup_raw&private_space
    clearing=jnp.where(empty,0,m['clearing'])
    action=jnp.where(empty&jnp.any(full_correct)&has_plate,_go(pos,face,plate_counter),action)
    action=jnp.where(plate,_go(pos,face,jnp.where(jnp.any(full_correct),ready_target,private_free)),action)
    action=jnp.where(empty&has_stored_dish&shared_space,_go(pos,face,stored_dish),action)
    action=jnp.where(cleanup,_go(pos,face,cleanup_raw),action)
    clearing=jnp.where(cleanup&(action==5),1,clearing)
    action=jnp.where(raw&(clearing>0),_go(pos,face,private_free),action)
    action=jnp.where(dish,_go(pos,face,jnp.where(shared_space,shared_free,private_free)),action)
    # Only try inventory signalling after an observed failed spatial repair.
    token_held=(inv[5]>0)&(inv[1]==0)
    token_wanted=(token_mode>0)&(recipe==token_recipe)&(~available)&(~has_plate)&(~has_stored_dish)
    stored_token,has_stored_token=_counter(g,pos,3,seat=1)
    token_source=jnp.array([4,jnp.where(pos[1]<=2,0,5)])
    token_signal=_go(pos,face,signal_target,False)
    token_signal=jnp.where(jnp.all(pos==signal_target)&(face!=0),3,token_signal)
    action=jnp.where(empty&token_wanted&(~cleanup),_go(pos,face,jnp.where(has_stored_token,stored_token,token_source)),action)
    token_drop=jnp.where(private_space,private_free,shared_free)
    action=jnp.where(token_held&(token_mode>0),jnp.where(token_wanted,token_signal,_go(pos,face,token_drop)),action)
    m['clearing']=clearing.astype(jnp.int32)
    m['recipe']=recipe.astype(jnp.int32);m['was_full']=full.astype(jnp.int32);m['cooldown']=cooldown.astype(jnp.int32)
    return action,m,{'pos':pos,'recipe':recipe,'pot':chosen,'counts':counts,'available':available,'cooldown':cooldown,'inventory':inv,'delegated':delegated,'signal_flip':flip,'pending':pending,'token_mode':token_mode}
