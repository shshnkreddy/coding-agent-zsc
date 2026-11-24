"""Local-observation cooperative cooking with a public pot handoff."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'informed_evidence_cooperative_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

# Fixed geometry, not hidden runtime state. Floor coordinates are disjoint by seat.
BOARD = ('WWWWBXBWWWW', 'WWW0   1WWW', 'WWWWWPWWWWW',
         'W         W', '0  W3R3W  0', 'W1WWWWWWW1W')
FLOORS = ((1,4),(1,5),(1,6),(3,1),(3,2),(3,3),(3,4),(3,5),
          (3,6),(3,7),(3,8),(3,9),(4,1),(4,2),(4,8),(4,9))


def static_map():
    channels = {'W':0, 'X':1, 'P':2, 'R':3, 'L':4, 'B':5,
                '0':6, '1':7, '2':8, '3':9}
    return jnp.array([[[int(channels.get(c,-1) == k) for k in range(10)]
                       for c in row] for row in BOARD], dtype=jnp.int32)


def locate(obs):
    floors = jnp.array(FLOORS, dtype=jnp.int32)
    rr = floors[:,0,None,None] + jnp.arange(5)[None,:,None] - 2
    cc = floors[:,1,None,None] + jnp.arange(5)[None,None,:] - 2
    valid = (rr >= 0) & (rr < 6) & (cc >= 0) & (cc < 11)
    templates = static_map()[jnp.clip(rr,0,5),jnp.clip(cc,0,10)] * valid[...,None]
    idx = jnp.argmin(jnp.sum(jnp.abs(templates-obs[None,:,:,22:32]),axis=(1,2,3)))
    return floors[idx]


def nav(pos, face, target, interact=True):
    # The lower room is a horizontal corridor with two short endpoint bays.
    # All crossings go through row 3; the upper room is a one-row corridor.
    floors = jnp.array(FLOORS, dtype=jnp.int32)
    adjacent = jnp.sum(jnp.abs(floors-target),axis=1) == 1
    same_room = (floors[:,0] < 2) == (pos[0] < 2)
    via3 = jnp.abs(pos[0]-3)+jnp.abs(floors[:,0]-3)+jnp.abs(pos[1]-floors[:,1])
    direct = jnp.sum(jnp.abs(floors-pos),axis=1)
    dist = jnp.where((pos[0]==1) | (pos[1]==floors[:,1]) | ((pos[0]==4)&(floors[:,0]==4)&(jnp.abs(pos[1]-floors[:,1])<=1)), direct, via3)
    goal = floors[jnp.argmin(jnp.where(adjacent & same_room,dist,1000))]
    # Move up from a lower bay before crossing obstacles.
    move = jnp.where(pos[1]<goal[1],0,2)
    move = jnp.where(pos[1]==goal[1],jnp.where(pos[0]<goal[0],1,3),move)
    move = jnp.where((pos[0]==4)&(pos[1]!=goal[1])&((goal[0]!=4)|(jnp.abs(pos[1]-goal[1])>1)),3,move)
    delta = target-pos
    desired = jnp.where(delta[0]<0,0,jnp.where(delta[0]>0,1,jnp.where(delta[1]>0,2,3)))
    turn = jnp.array([3,1,0,2],dtype=jnp.int32)[desired]
    at_goal = jnp.all(pos==goal)
    use = jnp.where(face==desired,5,turn) if interact else jnp.asarray(4,jnp.int32)
    return jnp.where(at_goal,use,move).astype(jnp.int32)


def init_agent_0():
    return {'items':jnp.zeros((6,11,6),jnp.int32), 'recipe':jnp.zeros(4,jnp.int32),
            'pot':jnp.zeros(6,jnp.int32), 'pending':jnp.asarray(False),
            'dish_seen':jnp.asarray(False),
            'goal_type':jnp.asarray(-1,jnp.int32), 'last_offer':jnp.asarray(-1,jnp.int32),
            'offer_streak':jnp.asarray(0,jnp.int32), 'empty_streak':jnp.asarray(0,jnp.int32),
            'own_counters':jnp.zeros(2,jnp.bool_), 'stale_counters':jnp.zeros(2,jnp.bool_),
            'tainted_hand':jnp.asarray(-1,jnp.int32), 'quarantine':jnp.asarray(-1,jnp.int32),
            'gesture_map':jnp.full(4,-1,jnp.int32), 'gesture_trial':jnp.array([0,1,0,1],jnp.int32),
            'gesture_runs':jnp.zeros(4,jnp.int32), 'gesture_code':jnp.asarray(-1,jnp.int32),
            'gesture_streak':jnp.asarray(0,jnp.int32), 'goal_origin':jnp.asarray(-1,jnp.int32),
            'previous_gesture':jnp.asarray(-1,jnp.int32), 'previous_guess':jnp.asarray(0,jnp.int32),
            'await_response':jnp.asarray(False), 'had_dish':jnp.asarray(False), 'fresh_partner':jnp.asarray(False), 'idle_steps':jnp.asarray(0,jnp.int32), 'last_blind':jnp.asarray(False),
            'partner_features':jnp.zeros(9,jnp.int32), 'partner_was_seen':jnp.asarray(False),
            'partner_change_step':jnp.asarray(0,jnp.int32), 'partner_change_ready':jnp.asarray(False), 'partner_cadence':jnp.asarray(7,jnp.int32),
            'timer':jnp.asarray(0,jnp.int32), 'fill_since':jnp.asarray(0,jnp.int32), 'pantry':jnp.asarray(False), 'step':jnp.asarray(0,jnp.int32)}


def init_agent_1():
    return init_agent_0()


def common(obs, reset, memory):
    initial = init_agent_0()
    m = {k:jnp.where(reset,initial[k],v) for k,v in memory.items()}
    pos = locate(obs)
    face = jnp.argmax(obs[2,2,1:5])
    inv = obs[2,2,5:11]
    rr = jnp.arange(6)[:,None]-pos[0]+2
    cc = jnp.arange(11)[None,:]-pos[1]+2
    visible = (rr>=0)&(rr<5)&(cc>=0)&(cc<5)
    grid = obs[jnp.clip(rr,0,4),jnp.clip(cc,0,4),:]
    items = jnp.where(visible[:,:,None],grid[:,:,32:38],m['items'])
    recipe_obs = jnp.max(obs[:,:,40:44],axis=(0,1))
    recipe = jnp.where(jnp.sum(recipe_obs)>0,recipe_obs,m['recipe'])
    partner = jnp.sum(obs[:,:,16:22],axis=(0,1))
    partner_seen = jnp.sum(obs[:,:,11])>0
    pot = items[2,5]
    pot_visible = visible[2,5]
    just_cleared = pot_visible & (jnp.sum(m['pot'][2:])==3) & (jnp.sum(pot[2:])==0)
    pending = m['pending'] | just_cleared | (partner[1]>0)
    pending = pending & ~(partner_seen & (partner[0]==0) & (jnp.sum(pot[2:])==0))
    timer=jnp.where(pot_visible,grid[2,5,44],jnp.maximum(m['timer']-1,0))
    prev_total=jnp.sum(m['pot'][2:]); total=jnp.sum(pot[2:])
    fill_since=jnp.where((total>0)&(prev_total==0),m['step'],m['fill_since'])
    pantry=jnp.where((total==3)&(prev_total<3),m['step']-fill_since>13,m['pantry'])
    m.update(items=items,recipe=recipe,pot=pot,pending=pending,timer=timer,fill_since=fill_since,pantry=pantry,
             dish_seen=partner[1]>0,step=m['step']+1)
    return pos,face,inv,partner,m


def counter_target(pos, items, ingredient, take=False, lower=False, private=False):
    rr,cc=jnp.meshgrid(jnp.arange(6),jnp.arange(11),indexing='ij')
    wall=static_map()[:,:,0]>0
    # Upper-accessible counters flank the pot; lower storage remains private.
    access = ((rr==2)&((cc==4)|(cc==6))) if not lower else (((rr==4)&((cc==3)|(cc==7)))|((rr==2)&(cc>=1)&(cc<=9)&(cc!=5)))
    if lower:
        access=jnp.where(private,((rr==4)|(rr==2))&((cc==3)|(cc==7)),access)
    good = (items[:,:,ingredient+2]>0)&(items[:,:,0]==0)&(items[:,:,1]==0) if take else jnp.sum(items,axis=2)==0
    cost = jnp.abs(rr-pos[0])+jnp.abs(cc-pos[1])
    idx=jnp.argmin(jnp.where(wall&access&good,cost,1000))
    return jnp.array([idx//11,idx%11],jnp.int32),jnp.any(wall&access&good)


def agent_0_act(obs, reset, memory, rng):
    pos,face,inv,partner,m=common(obs,reset,memory)
    pot=m['pot']; counts=pot[2:]; total=jnp.sum(counts)
    held=jnp.argmax(inv[2:]); raw=(jnp.sum(inv[2:])>0)&(inv[0]==0)
    partner_seen=jnp.sum(obs[:,:,11])>0
    partner_col=pos[1]+jnp.sum(obs[:,:,11]*(jnp.arange(5)[None,:]-2))
    partner_row=pos[0]+jnp.sum(obs[:,:,11]*(jnp.arange(5)[:,None]-2))
    centered=partner_seen&(partner_row==3)&(partner_col==5)
    partner_raw=(jnp.sum(partner[2:])==1)&(partner[0]==0)&(jnp.sum(partner[4:])==0)
    offer_type=jnp.argmax(partner[2:])
    eligible=centered&partner_raw&(total==0)&(inv[1]==0)
    streak=jnp.where(eligible,jnp.where(offer_type==m['last_offer'],m['offer_streak']+1,1),0)
    empty=centered&(jnp.sum(partner)==0)&(total==0)&(inv[1]==0)
    empty_streak=jnp.where(empty,m['empty_streak']+1,0)
    offered=m['items'][2,jnp.array([4,6])]
    owned=m['own_counters']&(jnp.sum(offered,axis=1)>0)
    occupied=jnp.sum(offered,axis=1)>0
    previous=jnp.where(reset,jnp.zeros_like(offered),memory['items'][2,jnp.array([4,6])])
    changed=jnp.any(offered!=previous,axis=1)
    offered_type=jnp.argmax(offered[:,2:],axis=1)
    stale=m['stale_counters']&occupied
    stale=jnp.where(changed,occupied&(offered_type==m['tainted_hand']),stale)
    stale=jnp.where(inv[1]>0,occupied,stale)
    tainted=jnp.where((inv[1]>0)&partner_raw,offer_type,m['tainted_hand'])
    tainted=jnp.where(partner_seen&((jnp.sum(partner)==0)|(partner_raw&(offer_type!=tainted))),-1,tainted)
    raw_counter=(jnp.sum(offered[:,2:4],axis=1)==1)&(jnp.sum(offered[:,0:2],axis=1)==0)
    valid=raw_counter&~owned&~stale
    counter_type=jnp.argmax(offered[jnp.argmax(valid.astype(jnp.int32)),2:])
    # A gesture is only a fallback after sustained absence of physical evidence.
    partner_face=jnp.argmax(jnp.sum(obs[:,:,12:16],axis=(0,1)))
    features=jnp.concatenate((jnp.array([partner_row,partner_col,partner_face],jnp.int32),partner))
    feature_change=partner_seen&(~m['partner_was_seen']|jnp.any(features!=m['partner_features']))
    interval=m['step']-m['partner_change_step']
    cadence_sample=feature_change&m['partner_was_seen']&m['partner_change_ready']&(interval>0)&(interval<=8)
    change_ready=partner_seen&jnp.where(feature_change,m['partner_was_seen'],m['partner_change_ready'])
    cadence=jnp.where(cadence_sample,jnp.minimum(m['partner_cadence'],interval),m['partner_cadence'])
    m.update(partner_features=jnp.where(partner_seen,features,m['partner_features']),
             partner_was_seen=partner_seen,partner_cadence=cadence,partner_change_ready=change_ready,
             partner_change_step=jnp.where(feature_change,m['step'],m['partner_change_step']))
    code=jnp.where(partner_col<5,0,jnp.where(partner_col>5,1,
         jnp.array([2,3,1,0],jnp.int32)[partner_face]))
    gesture_eligible=partner_seen&(partner_row==3)&(jnp.sum(partner)==0)&(total==0)&(jnp.sum(inv)==0)&~jnp.any(raw_counter&~owned)
    gesture_streak=jnp.where(gesture_eligible,jnp.where((code==m['gesture_code'])&~feature_change,m['gesture_streak']+1,1),0)
    new_dish=(inv[1]>0)&~m['had_dish']
    previous_gesture=jnp.where(new_dish,m['goal_origin'],m['previous_gesture'])
    previous_guess=jnp.where(new_dish,jnp.argmax(inv[2:]),m['previous_guess'])
    await_response=jnp.where(new_dish,(m['goal_origin']>=0)&(m['goal_origin']<4),m['await_response'])
    last_blind=jnp.where(new_dish,m['goal_origin']==4,m['last_blind'])
    response=await_response&gesture_eligible&(gesture_streak>=4)
    old_code=jnp.clip(previous_gesture,0,3)
    changed_code=code!=old_code
    mapping=m['gesture_map']; trials=m['gesture_trial']; runs=m['gesture_runs']
    # Recipe resampling is the only way a faithful recipe gesture can change
    # after delivery. Thus a change is positive evidence for the previous guess.
    learned=mapping.at[old_code].set(previous_guess).at[code].set(1-previous_guess)
    mapping=jnp.where(response&changed_code,learned,mapping)
    repeated=runs.at[old_code].add(1)
    runs=jnp.where(response&~changed_code,repeated,runs)
    switch_trial=response&~changed_code&(mapping[old_code]<0)&(runs[old_code]>=2)
    trials=jnp.where(switch_trial,trials.at[old_code].set(1-trials[old_code]),trials)
    runs=jnp.where(switch_trial,runs.at[old_code].set(1),runs)
    await_response=await_response&~response
    origin=jnp.where((inv[1]==0)&m['had_dish'],-1,m['goal_origin'])
    goal=m['goal_type']
    counter_signal=(empty_streak>=3)&jnp.any(valid)
    # Empty hands within indicator range establish an opportunity to read the current recipe.
    recipe_access=partner_seen&(jnp.abs(partner_row-4)<=2)&(jnp.abs(partner_col-5)<=2)
    fresh_partner=jnp.where(inv[1]>0,False,m['fresh_partner']|(recipe_access&(jnp.sum(partner)==0)&(total==0)))
    early_offer=fresh_partner&partner_raw&(total==0)&(inv[1]==0)
    held_signal=(streak>=jnp.maximum(4,cadence+1))|early_offer
    physical_signal=counter_signal|held_signal
    replace=(goal<0)|(origin>=0)
    physical_type=jnp.where(held_signal,offer_type,counter_type)
    goal=jnp.where(replace&physical_signal,physical_type,goal)
    origin=jnp.where(replace&physical_signal,-1,origin)
    await_response=await_response&~physical_signal
    gesture_type=jnp.where(mapping[code]>=0,mapping[code],trials[code])
    required_stability=jnp.where(mapping[code]>=0,4,10)
    use_gesture=(goal<0)&gesture_eligible&(gesture_streak>=required_stability)
    goal=jnp.where(use_gesture,gesture_type,goal)
    origin=jnp.where(use_gesture,code,origin)
    idle_steps=jnp.where((goal<0)&(total==0)&(jnp.sum(inv)==0),m['idle_steps']+1,0)
    timeout=jnp.where(last_blind,12,80)
    use_blind=(goal<0)&(idle_steps>=timeout)&~partner_raw&~jnp.any(raw_counter&~owned)
    blind_type=jnp.where(last_blind,1-previous_guess,0)
    goal=jnp.where(use_blind,blind_type,goal)
    origin=jnp.where(use_blind,4,origin)
    broken_pot=(total>0)&~((counts[0]==total)|(counts[1]==total))
    pot_type=jnp.where(jnp.argmax(counts)<2,jnp.argmax(counts),0)
    origin=jnp.where((total>0)&((goal!=pot_type)|broken_pot),-1,origin)
    goal=jnp.where(total>0,pot_type,goal)
    goal=jnp.where(inv[1]>0,-1,goal)
    m.update(gesture_map=mapping,gesture_trial=trials,gesture_runs=runs,
             gesture_code=code,gesture_streak=gesture_streak,goal_origin=origin,
             previous_gesture=previous_gesture,previous_guess=previous_guess,
             await_response=await_response,had_dish=inv[1]>0,fresh_partner=fresh_partner,idle_steps=idle_steps,last_blind=last_blind)
    typ=jnp.maximum(goal,0)
    source=jnp.array([1,jnp.where(typ==0,3,7)],jnp.int32)
    counter,has_counter=counter_target(pos,m['items'],typ,take=True)
    source=jnp.where(has_counter,counter,source)
    action=nav(pos,face,jnp.array([2,5]),interact=False)
    action=jnp.where((goal>=0)&(total<3),nav(pos,face,source),action)
    cleanup_possible=raw_counter&stale&~owned
    cleanup_idx=jnp.argmax(cleanup_possible.astype(jnp.int32))
    cleanup_target=jnp.array([2,jnp.where(cleanup_idx==0,4,6)],jnp.int32)
    cleanup=(goal<0)&(total==0)&(empty_streak>=3)&jnp.any(cleanup_possible)
    action=jnp.where(cleanup,nav(pos,face,cleanup_target),action)
    plate=jnp.array([0,jnp.where(pos[1]<=5,4,6)],jnp.int32)
    action=jnp.where(total==3,nav(pos,face,plate),action)
    drop,_=counter_target(pos,m['items'],held,take=False)
    quarantine=m['quarantine']
    alternate=jnp.array([2,jnp.where(quarantine==0,6,4)],jnp.int32)
    drop=jnp.where(quarantine>=0,alternate,drop)
    raw_ok=(total<3)&(goal>=0)&(((goal==held)&(counts[held]==total))|(broken_pot&(held<2)))
    raw_action=jnp.where(raw_ok,nav(pos,face,jnp.array([2,5])),nav(pos,face,drop))
    action=jnp.where(raw,raw_action,action)
    action=jnp.where(inv[0]>0,nav(pos,face,jnp.array([2,5])),action)
    action=jnp.where(inv[1]>0,nav(pos,face,jnp.array([0,5])),action)
    dropping=raw&~raw_ok&(action==5)&(inv[0]==0)&(jnp.sum(m['items'][drop[0],drop[1]])==0)
    owned=owned.at[jnp.where(drop[1]==4,0,1)].set(jnp.where(dropping,True,owned[jnp.where(drop[1]==4,0,1)]))
    quarantine=jnp.where(cleanup&(action==5)&(jnp.sum(inv)==0),cleanup_idx,quarantine)
    quarantine=jnp.where(dropping|(raw&raw_ok&(action==5)),-1,quarantine)
    m.update(goal_type=goal,last_offer=offer_type,offer_streak=streak,empty_streak=empty_streak,
             own_counters=owned,stale_counters=stale,tainted_hand=tainted,quarantine=quarantine)
    return action,m,{'pos':pos,'pot':pot,'inventory':inv,'goal':goal,'offer_streak':streak,'empty_streak':empty_streak,'stale':stale,'quarantine':quarantine,'gesture':code,'gesture_map':mapping,'origin':origin,'early_offer':early_offer,'idle_steps':idle_steps,'partner_cadence':cadence}


def agent_1_act(obs, reset, memory, rng):
    pos,face,inv,partner,m=common(obs,reset,memory)
    pot=m['pot']; total=jnp.sum(pot[2:]); typ=jnp.argmax(m['recipe'])
    raw=(jnp.sum(inv[2:])>0)&(inv[0]==0); held=jnp.argmax(inv[2:])
    side=jnp.where(pos[1]<=5,0,1)
    def supply(ingredient):
        return jnp.where(ingredient==0,jnp.array([4,jnp.where(side==0,0,10)]),jnp.array([5,jnp.where(side==0,1,9)]))
    counter,has_counter=counter_target(pos,m['items'],typ,take=True,lower=True)
    source=jnp.where(has_counter,counter,supply(typ))
    broken_pot=(total>0)&(total<3)&(pot[typ+2]!=total)
    recovery_supply=jnp.array([4,jnp.where(pos[1]<=5,4,6)],jnp.int32)
    source=jnp.where(broken_pot,recovery_supply,source)
    wait=nav(pos,face,jnp.array([2,5]),interact=False)
    need=(total<3)&~m['pending']&(jnp.sum(m['recipe'])>0)
    action=jnp.where(need,nav(pos,face,source),wait)
    drop,has_drop=counter_target(pos,m['items'],held,take=False,lower=True,private=m['pantry'])
    raw_action=jnp.where(need,nav(pos,face,jnp.array([2,5])),wait)
    # Refill private shelves during the otherwise unavoidable cooking delay.
    shelves=m['items'][jnp.array([2,4,2,4]),jnp.array([3,3,7,7]),2:4]
    stocks=jnp.sum(shelves,axis=0)
    missing=jnp.argmin(stocks)
    cooking=total==3
    fetch_stock=m['pantry']&cooking&(m['timer']>10)&jnp.any(stocks==0)
    action=jnp.where(fetch_stock,nav(pos,face,supply(missing)),action)
    arm=m['pantry']&cooking&(m['timer']<=8)&has_counter
    action=jnp.where(arm,nav(pos,face,counter),action)
    stash=m['pantry']&cooking&(m['timer']>7)&(stocks[jnp.minimum(held,1)]==0)&has_drop
    raw_action=jnp.where(stash,nav(pos,face,drop),raw_action)
    raw_action=jnp.where((held!=typ)&~cooking&~broken_pot,nav(pos,face,drop),raw_action)
    action=jnp.where(raw,raw_action,action)
    return action,m,{'pos':pos,'pot':pot,'inventory':inv,'recipe':m['recipe'],'pending':m['pending'],'timer':m['timer'],'stocks':stocks,'pantry':m['pantry']}
