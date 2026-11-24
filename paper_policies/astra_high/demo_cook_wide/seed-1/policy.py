"""Cooperate through visible food, inventory changes, and partner actions.

The lower room observes the recipe and physically seeds the shared pot. The
upper room completes observed recipes and serves. Both roles tolerate a
partner doing more or less collection. Counter contents have ordinary item
semantics; no private counter-position or timing convention is required.

Persistent nonverbal requests are a fallback only after physical progress
stalls. The upper role learns their mapping from post-delivery changes, while
raw ingredient IDs always keep the environment's fixed meanings.
"""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'rapid_partner_adaptation', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 6, 'max_width': 11, 'num_ingredients': 4}

_LAYOUT = ('WWWWBXBWWWW', 'WWW0 A 1WWW', 'WWWWWPWWWWW',
           'W    A    W', '0  W3R3W  0', 'W1WWWWWWW1W')
_FLOORS = [(1, 4), (1, 5), (1, 6), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 1), (4, 2), (4, 8), (4, 9)]
_STATIC_CODES = [
    [[0, 0, 0, 0, 0], [1, 1, 6, 2, 6], [1, 7, 0, 0, 0], [1, 1, 1, 3, 1], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [1, 6, 2, 6, 1], [7, 0, 0, 0, 8], [1, 1, 3, 1, 1], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [6, 2, 6, 1, 1], [0, 0, 0, 8, 1], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
    [[0, 1, 1, 1, 7], [0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 7, 0, 0, 1], [0, 1, 8, 1, 1]],
    [[1, 1, 1, 7, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [7, 0, 0, 1, 10], [1, 8, 1, 1, 1]],
    [[1, 1, 7, 0, 0], [1, 1, 1, 1, 3], [0, 0, 0, 0, 0], [0, 0, 1, 10, 4], [8, 1, 1, 1, 1]],
    [[1, 7, 0, 0, 0], [1, 1, 1, 3, 1], [0, 0, 0, 0, 0], [0, 1, 10, 4, 10], [1, 1, 1, 1, 1]],
    [[7, 0, 0, 0, 8], [1, 1, 3, 1, 1], [0, 0, 0, 0, 0], [1, 10, 4, 10, 1], [1, 1, 1, 1, 1]],
    [[0, 0, 0, 8, 1], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0], [10, 4, 10, 1, 0], [1, 1, 1, 1, 1]],
    [[0, 0, 8, 1, 1], [3, 1, 1, 1, 1], [0, 0, 0, 0, 0], [4, 10, 1, 0, 0], [1, 1, 1, 1, 8]],
    [[0, 8, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [10, 1, 0, 0, 7], [1, 1, 1, 8, 1]],
    [[8, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 0, 0, 7, 0], [1, 1, 8, 1, 0]],
    [[0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 7, 0, 0, 1], [0, 1, 8, 1, 1], [0, 0, 0, 0, 0]],
    [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [7, 0, 0, 1, 10], [1, 8, 1, 1, 1], [0, 0, 0, 0, 0]],
    [[1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [10, 1, 0, 0, 7], [1, 1, 1, 8, 1], [0, 0, 0, 0, 0]],
    [[1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 0, 0, 7, 0], [1, 1, 8, 1, 0], [0, 0, 0, 0, 0]],
]

def locate(obs):
    # Static object channels are mutually exclusive. Zero is floor/padding;
    # codes 1..10 follow channels 22..31. Only the local crop is matched.
    local_codes=jnp.sum(obs[:,:,22:32]*jnp.arange(1,11,dtype=jnp.int32),axis=2)
    errors=jnp.sum(jnp.asarray(_STATIC_CODES,jnp.int32)!=local_codes[None,:,:],axis=(1,2))
    return jnp.asarray(_FLOORS,jnp.int32)[jnp.argmin(errors)]


def cell(obs,pos,r,c):
    rr=r-pos[0]+2
    cc=c-pos[1]+2
    return jnp.where((rr>=0)&(rr<5)&(cc>=0)&(cc<5),obs[jnp.clip(rr,0,4),jnp.clip(cc,0,4)],jnp.zeros((45,),jnp.int32))


def face_interact(facing,direction):
    return jnp.where(facing==direction,5,jnp.asarray([3,1,0,2],jnp.int32)[direction])


def upper_use(pos,facing,r,c):
    # Accessible station cells are directly above/below our 3-cell corridor,
    # except ingredient supplies at the two ends.
    stand_c=jnp.clip(c,4,6)
    direction=jnp.where(r<1,0,jnp.where(r>1,1,jnp.where(c<4,3,2)))
    return jnp.where(pos[1]<stand_c,0,jnp.where(pos[1]>stand_c,2,face_interact(facing,direction)))


def lower_move(pos,r,c):
    # All horizontal routes use row 3; only outer columns extend to row 4.
    return jnp.where(pos[1]!=c,jnp.where(pos[0]>3,3,jnp.where(pos[1]<c,0,2)),
                     jnp.where(pos[0]<r,1,jnp.where(pos[0]>r,3,4)))


def lower_use(pos,facing,r,c):
    # Ingredient 0: stand at (4,1) or (4,9); ingredient 1: same stands.
    sr=jnp.where(r==2,3,jnp.where(r==5,4,jnp.where((c==0)|(c==10),4,3)))
    sc=jnp.clip(c,1,9)
    direction=jnp.where(r<sr,0,jnp.where(r>sr,1,jnp.where(c<sc,3,2)))
    return jnp.where((pos[0]==sr)&(pos[1]==sc),face_interact(facing,direction),lower_move(pos,sr,sc))


def init_agent_0():
    # 0 tick; 1 final-slot wait; 2 observed empty partner hands this cycle;
    # 3 recipe evidence + 1; 4 signal; 5 stable pose duration; 6 empty-pot age;
    # 7 using fallback this cycle; 8/9 last fallback signal/dish ingredient;
    # 10 awaiting post-delivery evidence; 11 unchanged trial count;
    # 12 unknown-code inversion; 14 cycle signal; 16 delivery tick;
    # 17 partner pose; 20:28 learned code -> ingredient+1 (zero = unknown).
    return jnp.zeros((32,),jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    m=jnp.where(reset,jnp.zeros_like(memory),memory)
    pos=locate(obs)
    facing=jnp.argmax(obs[2,2,1:5])
    inv=obs[2,2,5:11]
    raw=(inv[0]==0)&(jnp.sum(inv[2:])>0)
    empty=jnp.sum(inv)==0
    pot=cell(obs,pos,2,5)
    count=jnp.sum(pot[34:38])
    partner=jnp.sum(obs[:,:,16:22],axis=(0,1))
    partner_raw=(partner[0]==0)&(jnp.sum(partner[2:4])>0)
    partner_visible=jnp.sum(obs[:,:,11])>0
    seen_empty=jnp.where(reset|(partner_visible&(jnp.sum(partner)==0)),1,m[2])
    fresh_raw=partner_raw&(seen_empty>0)
    evidence=jnp.where(fresh_raw,jnp.argmax(partner[2:])+1,m[3])
    evidence=jnp.where(count>0,jnp.argmax(pot[34:38])+1,evidence)
    # Primary evidence is always actual food. A persistent empty-handed
    # partner may instead be signaling; try a mapping only after a stall.
    pi=jnp.argmax(obs[:,:,11])
    partner_col=pos[1]+(pi%5)-2
    partner_facing=jnp.argmax(jnp.sum(obs[:,:,12:16],axis=(0,1)))
    signal=jnp.where(partner_col<5,0,jnp.where(partner_col>5,1,partner_facing+2))
    partner_empty=partner_visible&(jnp.sum(partner)==0)
    signal=jnp.where(partner_raw,6+jnp.argmax(partner[2:4]),signal)
    signal_visible=partner_visible&(partner_empty|partner_raw)
    partner_row=pos[0]+(pi//5)-2
    pose=(partner_row*11+partner_col)*4+partner_facing
    stable=jnp.where(signal_visible,jnp.where((signal==m[4])&(pose==m[17]),m[5]+1,1),0)
    empty_wait=jnp.where((count==0)&(inv[1]==0),m[6]+1,0)
    mappings=m[20:28]
    last_signal=jnp.clip(m[8],0,7)
    same_signal_class=((signal<2)&(last_signal<2))|((signal>=2)&(signal<6)&(last_signal>=2)&(last_signal<6))
    confirmed=(m[10]>0)&signal_visible&(stable>=3)&(signal!=last_signal)&((m[0]-m[16])<=12)&same_signal_class
    mappings=mappings.at[last_signal].set(jnp.where(confirmed,m[9]+1,mappings[last_signal]))
    mappings=mappings.at[signal].set(jnp.where(confirmed,2-m[9],mappings[signal]))
    learned=mappings[signal]
    defaults=jnp.asarray([0,1,0,1,1,0,0,1],jnp.int32)
    flip=jnp.where(confirmed,defaults[last_signal]^m[9],m[12])
    unchanged=jnp.where(confirmed,0,m[11])
    threshold=jnp.where(signal>=6,8,jnp.where(learned>0,3,12))
    use_signal=(evidence==0)&(count==0)&empty&signal_visible&(stable>=threshold)&(empty_wait>=threshold)
    switch_trial=use_signal&(signal<6)&(learned==0)&(unchanged>=1)&(signal==last_signal)
    flip=jnp.where(switch_trial,1-flip,flip)
    unchanged=jnp.where(switch_trial,0,unchanged)
    signal_type=jnp.where(signal>=6,signal-6,jnp.where(learned>0,learned-1,defaults[signal]^flip))
    evidence=jnp.where(use_signal,signal_type+1,evidence)
    signal_active=jnp.where(use_signal,1,m[7])
    cycle_signal=jnp.where(use_signal,signal,m[14])
    known=evidence>0
    ingredient=jnp.maximum(evidence-1,0)
    pot_open=(count<3)&(pot[33]==0)&(pot[44]==0)
    # Counter pickups are grounded in their contents, not their positions.
    counter_cells=jnp.asarray([[2,4],[2,6]],jnp.int32)
    counter_obs=jnp.stack([cell(obs,pos,r,c) for r,c in [(2,4),(2,6)]])
    usable=(counter_obs[:,32]==0)&(counter_obs[:,33]==0)&(counter_obs[:,34+ingredient]>0)
    ci=jnp.argmin(jnp.where(usable,jnp.abs(counter_cells[:,1]-pos[1]),100))
    pickup=upper_use(pos,facing,counter_cells[ci,0],counter_cells[ci,1])
    gather=jnp.where(jnp.any(usable),pickup,upper_use(pos,facing,1,jnp.where(ingredient==0,3,7)))
    ready_or_cooking=(count>=3)|(pot[33]>0)|(pot[44]>0)
    plate_c=jnp.where(pos[1]<=5,4,6)
    idle=upper_use(pos,facing,2,5)
    # Empty hands: fill an observed recipe or fetch a plate for the current pot.
    action=jnp.where(ready_or_cooking,upper_use(pos,facing,0,plate_c),jnp.where(known,gather,idle))
    action=jnp.where((~known)&(~ready_or_cooking)&(pos[1]==5)&(facing==1),4,action)
    vacant=jnp.sum(counter_obs[:,32:38],axis=1)==0
    stocked=jnp.sum(counter_obs[:,34:36],axis=0)>0
    stock_type=jnp.where(stocked[0],1,jnp.where(stocked[1],0,jnp.where(pos[1]>5,1,0)))
    stock_possible=jnp.all(vacant)&(~jnp.all(stocked))
    stock_time=(((count==0)&(~known))|((pot[44]>10)&(pot[33]==0)))&stock_possible
    stock_action=upper_use(pos,facing,1,jnp.where(stock_type==0,3,7))
    action=jnp.where(stock_time,stock_action,action)
    last_incoming=(count==2)&partner_raw&(partner[2+ingredient]>0)
    yield_time=jnp.where(last_incoming,m[1]+1,0)
    action=jnp.where(last_incoming&(yield_time<=3),idle,action)
    # Raw ingredients that cannot be used are placed on an empty counter.
    raw_type=jnp.argmax(inv[2:])
    compatible=(count==0)|(pot[34+raw_type]==count)
    vacant=jnp.sum(counter_obs[:,32:38],axis=1)==0
    di=jnp.argmin(jnp.where(vacant,jnp.abs(counter_cells[:,1]-pos[1]),100))
    drop=upper_use(pos,facing,counter_cells[di,0],counter_cells[di,1])
    action=jnp.where(raw,jnp.where(pot_open&compatible&known&(raw_type==ingredient),upper_use(pos,facing,2,5),jnp.where(jnp.any(vacant),drop,4)),action)
    action=jnp.where(inv[0]>0,upper_use(pos,facing,2,5),action)
    action=jnp.where(inv[1]>0,upper_use(pos,facing,0,5),action)
    delivering=(inv[1]>0)&(action==5)
    signal_delivery=delivering&(signal_active>0)
    pending=jnp.where(confirmed|use_signal,0,m[10])
    nm=m.at[0].set(m[0]+1).at[1].set(yield_time).at[2].set(jnp.where(delivering,0,seen_empty)).at[3].set(jnp.where(delivering,0,evidence))
    nm=nm.at[4].set(signal).at[5].set(stable).at[6].set(empty_wait).at[7].set(jnp.where(delivering,0,signal_active))
    nm=nm.at[8].set(jnp.where(signal_delivery,cycle_signal,m[8])).at[9].set(jnp.where(signal_delivery,jnp.argmax(inv[2:]),m[9]))
    nm=nm.at[10].set(jnp.where(signal_delivery,1,pending)).at[11].set(unchanged+signal_delivery.astype(jnp.int32)).at[12].set(flip).at[14].set(cycle_signal).at[16].set(jnp.where(signal_delivery,m[0],m[16])).at[17].set(pose).at[20:28].set(mappings)
    return action.astype(jnp.int32),nm,{'pos':pos,'target':jnp.where(known,ingredient,-1),'pot_count':count,'inventory':inv,'signal':signal,'fallback':signal_active,'mappings':mappings}


def init_agent_1():
    # 0 tick; 1 remembered recipe ingredient; 2 recipe known; 3 last pot count;
    # 4 pot busy; 5 prefetch ingredient; 6 prefetch phase (idle, collect,
    # return to center, store); 7 incorrect pot; 8 bounded partial-pot wait.
    # No memory is shared between seats.
    return jnp.zeros((9,),jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    m=jnp.where(reset,jnp.zeros_like(memory),memory)
    pos=locate(obs)
    facing=jnp.argmax(obs[2,2,1:5])
    inv=obs[2,2,5:11]
    recipe=jnp.sum(obs[:,:,40:44],axis=(0,1))
    recipe_visible=jnp.sum(recipe)>0
    target=jnp.where(recipe_visible,jnp.argmax(recipe),m[1])
    known=recipe_visible|(m[2]>0)
    pot=cell(obs,pos,2,5)
    pot_visible=pot[24]>0
    count=jnp.sum(pot[34:38])
    count=jnp.where(pot_visible,count,m[3])
    busy=jnp.where(pot_visible,(pot[44]>0)|(pot[33]>0)|(count>=3),m[4]>0)
    partner=jnp.sum(obs[:,:,16:22],axis=(0,1))
    partner_dish=partner[1]>0
    # Wrong deliveries preserve the requested recipe. Preparing the correct
    # seed while a wrong dish cooks is therefore safe across that delivery.
    wrong_pot=jnp.where(pot_visible&(count>0),pot[34+target]<count,m[7]>0)
    wrong_wait=jnp.where(wrong_pot&(count>0)&(~busy),m[8]+1,0)
    avoid_mixing=wrong_pot&(count>0)&(wrong_wait<=12)
    empty=jnp.sum(inv)==0
    raw=(inv[0]==0)&(jnp.sum(inv[2:])>0)
    raw_type=jnp.argmax(inv[2:])
    # Keep checking recipe and pot at the center during cooking and delivery.
    wait=lower_move(pos,3,5)
    side=jnp.where(target==0,0,1)
    source_r=jnp.where(target==0,4,5)
    source_c=jnp.where(target==0,jnp.where(side==0,0,10),jnp.where(side==0,1,9))
    gather=lower_use(pos,facing,source_r,source_c)
    stash_cells=jnp.asarray([[2,4],[2,6],[4,3],[4,7],[2,3],[2,7],[2,2],[2,8]],jnp.int32)
    stash_obs=jnp.stack([cell(obs,pos,r,c) for r,c in [(2,4),(2,6),(4,3),(4,7),(2,3),(2,7),(2,2),(2,8)]])
    usable=(stash_obs[:,22]>0)&(stash_obs[:,32]==0)&(stash_obs[:,33]==0)&(stash_obs[:,34+target]>0)
    stash_side=(stash_cells[:,1]>5).astype(jnp.int32)
    si=jnp.argmin(jnp.where(usable,jnp.abs(stash_cells[:,1]-pos[1])+4*(stash_side!=target),100))
    gather=jnp.where(jnp.any(usable),lower_use(pos,facing,stash_cells[si,0],stash_cells[si,1]),gather)
    action=jnp.where(known&((~busy)|wrong_pot)&(~partner_dish),gather,wait)
    # Release a counter if the serving player is blocked holding raw food.
    partner_raw=(partner[0]==0)&(jnp.sum(partner[2:])>0)
    shared_raw=(stash_obs[:2,32]==0)&(stash_obs[:2,33]==0)&(jnp.sum(stash_obs[:2,34:38],axis=1)>0)
    rescue_i=jnp.argmin(jnp.where(shared_raw,jnp.abs(stash_cells[:2,1]-pos[1]),100))
    rescue=lower_use(pos,facing,stash_cells[rescue_i,0],stash_cells[rescue_i,1])
    action=jnp.where(busy&partner_raw&jnp.any(shared_raw),rescue,action)
    # One round trip fits early cooking, including returning and storing it.
    cached=jnp.sum(stash_obs[:,34:36],axis=0)>0
    pre_type=jnp.where(cached[0],1,jnp.where(cached[1],0,target))
    mode=jnp.where((m[6]==3)&empty,0,m[6])
    start_prefetch=(mode==0)&empty&busy&(~wrong_pot)&(pot[44]>=17)&(pos[0]==3)&(pos[1]==5)&(~jnp.all(cached))&(~partner_raw)
    pre_type=jnp.where(start_prefetch,pre_type,m[5])
    mode=jnp.where(start_prefetch,1,mode)
    mode=jnp.where((mode==1)&raw,2,mode)
    mode=jnp.where((mode==2)&(pos[0]==3)&(pos[1]==5),3,mode)
    pre_source=lower_use(pos,facing,jnp.where(pre_type==0,4,5),jnp.where(pre_type==0,0,9))
    action=jnp.where((mode==1)&empty,pre_source,action)
    # Always return to sight of the current recipe before depositing.
    insert=lower_use(pos,facing,2,5)
    discard_cells=jnp.asarray([[2,4],[2,6],[4,3],[4,7],[2,3],[2,7],[2,2],[2,8]],jnp.int32)
    discard_obs=jnp.stack([cell(obs,pos,r,c) for r,c in [(2,4),(2,6),(4,3),(4,7),(2,3),(2,7),(2,2),(2,8)]])
    vacant=(discard_obs[:,22]>0)&(jnp.sum(discard_obs[:,32:38],axis=1)==0)
    vacant=vacant&(~(busy&partner_raw&(jnp.arange(8)<2)))
    costs=jnp.abs(discard_cells[:,1]-pos[1])+6*((discard_cells[:,1]>5).astype(jnp.int32)!=raw_type)
    di=jnp.argmin(jnp.where(vacant,costs,100))
    drop=lower_use(pos,facing,discard_cells[di,0],discard_cells[di,1])
    bad=((raw_type!=target)&known)|(busy&(~wrong_pot))
    action=jnp.where(raw,jnp.where(bad,jnp.where(jnp.any(vacant),drop,lower_move(pos,3,jnp.where(pos[1]<=5,3,7))),jnp.where(busy|partner_dish|avoid_mixing,wait,insert)),action)
    action=jnp.where((mode==2)&raw,wait,action)
    nm=m.at[0].set(m[0]+1).at[1].set(target).at[2].set(known.astype(jnp.int32)).at[3].set(count).at[4].set(busy.astype(jnp.int32)).at[5].set(pre_type).at[6].set(mode).at[7].set(wrong_pot.astype(jnp.int32)).at[8].set(wrong_wait)
    return action.astype(jnp.int32),nm,{'pos':pos,'target':target,'pot_count':count,'busy':busy,'inventory':inv,'wrong_pot':wrong_pot}
