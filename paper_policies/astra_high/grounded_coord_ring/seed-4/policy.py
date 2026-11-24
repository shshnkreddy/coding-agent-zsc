"""Locally observed, independently competent kitchen roles."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}

from custom_primitives import FLOOR, TEMPLATES, DIST

def locate(obs):
    error = jnp.sum(jnp.abs(jnp.asarray(TEMPLATES, jnp.int32)-obs[None,:,:,20:29]), axis=(1,2,3))
    return jnp.asarray(FLOOR)[jnp.argmin(error)]


def navigate(pos, facing, targets, interact=True):
    """Choose the shortest accessible approach to a set of object cells."""
    cells = jnp.arange(81)
    rows, cols = cells//9, cells%9
    tr, tc = targets//9, targets%9
    adjacent = (jnp.abs(rows[:,None]-tr[None,:])+jnp.abs(cols[:,None]-tc[None,:]))==1
    desired = jnp.where(tr[None,:]<rows[:,None], 3, jnp.where(tr[None,:]>rows[:,None],1,jnp.where(tc[None,:]<cols[:,None],2,0)))
    distances = jnp.asarray(DIST)
    turn = jnp.asarray([3,1,0,2])[facing]
    cost = distances[pos,:,None]*2 + (desired != turn)
    cost = jnp.where(adjacent, cost, 9999)
    best = jnp.argmin(cost)
    approach = best//targets.size
    direction = desired.reshape(-1)[best]
    next_cells = jnp.clip(pos+jnp.asarray([1,9,-1,-9]),0,80)
    reachable = (jnp.abs(next_cells//9-pos//9)+jnp.abs(next_cells%9-pos%9))==1
    move = jnp.argmin(jnp.where(reachable,distances[next_cells,approach],9999))
    at = pos==approach
    return jnp.where(at,jnp.where(turn==direction,5 if interact else 4,direction),move).astype(jnp.int32)


def go_to(pos, target):
    nxt=jnp.clip(pos+jnp.asarray([1,9,-1,-9]),0,80)
    d=jnp.asarray(DIST)
    action=jnp.argmin(d[nxt,target])
    return jnp.where(pos==target,4,action).astype(jnp.int32)


def init_agent_0():
    return {'recipe':jnp.int32(-1), 'previous_cooked':jnp.int32(0), 'time':jnp.int32(0), 'stash':jnp.zeros((2,),jnp.int32), 'armed':jnp.int32(0), 'plate_age':jnp.int32(0), 'handoff':jnp.int32(-1), 'return_counter':jnp.int32(21)}


def init_agent_1():
    return init_agent_0()


def common(obs, reset, memory):
    pos=locate(obs)
    facing=jnp.argmax(obs[2,2,1:5])
    inv=obs[2,2,5:10]
    displayed=jnp.max(obs[:,:,36:39],axis=(0,1))
    previous=jnp.where(reset,-1,memory['recipe'])
    delivered=(memory['previous_cooked']==1)&(inv[1]==0)
    recipe=jnp.where(delivered,-1,previous)
    recipe=jnp.where(jnp.sum(displayed)>0,jnp.argmax(displayed),recipe).astype(jnp.int32)
    m={'recipe':recipe,'previous_cooked':inv[1], 'time':jnp.where(reset,0,memory['time'])+1, 'stash':jnp.where(reset,jnp.zeros((2,),jnp.int32),memory['stash']), 'armed':jnp.where(reset,0,memory['armed']), 'plate_age':jnp.where(reset,0,memory['plate_age']), 'handoff':jnp.where(reset,-1,memory['handoff']), 'return_counter':jnp.where(reset,21,memory['return_counter'])}
    return pos,facing,inv,recipe,m


def agent_0_act(obs, reset, memory, rng):
    pos,facing,inv,recipe,m=common(obs,reset,memory)
    # The outer observer shows a real ingredient beside its matching supply.
    # Unknown partners can observe its intent without learning an action code.
    source=jnp.where(recipe==1,47,29)
    idle_cell=jnp.where(recipe==1,46,28)
    occupied=jnp.sum(inv)>0
    held=jnp.argmax(inv[2:])
    wrong=occupied & (held!=recipe)
    # Two reachable storage counters let us swap inventory on every order change.
    storage=jnp.asarray([20,56])
    sr=storage//9-pos//9+2
    sc=storage%9-pos%9+2
    visible=(sr>=0)&(sr<5)&(sc>=0)&(sc<5)
    items=obs[jnp.clip(sr,0,4),jnp.clip(sc,0,4),31:34]
    stash=jnp.where(visible,jnp.sum(items,axis=-1),m['stash'])
    m['stash']=stash
    pickup=jnp.where(stash[jnp.clip(recipe,0,1)]>0,storage[jnp.clip(recipe,0,1)],source)
    discard=navigate(pos,facing,jnp.asarray([storage[jnp.clip(held,0,1)]]))
    action=jnp.where(occupied,go_to(pos,idle_cell),navigate(pos,facing,jnp.asarray([pickup])))
    action=jnp.where(wrong,discard,action)
    partner_plate=jnp.sum(obs[:,:,15])>0
    partner_cooked=jnp.sum(obs[:,:,16])>0
    m['plate_age']=jnp.where(partner_plate,m['plate_age']+1,0).astype(jnp.int32)
    partner_plating=partner_cooked|(partner_plate&(m['plate_age']<=30))
    clear_signal=jnp.where(occupied,discard,go_to(pos,37))
    action=jnp.where(partner_plating,clear_signal,action)
    action=jnp.where(recipe<0,go_to(pos,37),action)
    # Cooked dishes are unambiguous physical handoffs, not coded messages.
    rr=pos//9+jnp.arange(5)[:,None]-2
    cc=pos%9+jnp.arange(5)[None,:]-2
    grid=jnp.clip(rr*9+cc,0,80)
    desired_counts=jnp.eye(3,dtype=jnp.int32)[jnp.clip(recipe,0,2)]*3
    correct_dish=(obs[:,:,20]>0)&(obs[:,:,29]>0)&(obs[:,:,30]>0)&jnp.all(obs[:,:,31:34]==desired_counts,axis=-1)&(recipe>=0)
    target=m['handoff']
    hr=target//9-pos//9+2
    hc=target%9-pos%9+2
    target_visible=(target>=0)&(hr>=0)&(hr<5)&(hc>=0)&(hc<5)
    target=jnp.where(target_visible&(~correct_dish[jnp.clip(hr,0,4),jnp.clip(hc,0,4)]),-1,target)
    choose=jnp.argmin(jnp.where(correct_dish,jnp.abs(rr-pos//9)+jnp.abs(cc-pos%9),999))
    target=jnp.where(jnp.any(correct_dish),grid.reshape(-1)[choose],target).astype(jnp.int32)
    m['handoff']=target
    m['return_counter']=jnp.where(target>=0,target,m['return_counter'])
    handoff_action=jnp.where(occupied,discard,navigate(pos,facing,jnp.asarray([jnp.maximum(target,0)])))
    action=jnp.where(target>=0,handoff_action,action)
    matching=jnp.all(inv[2:]==desired_counts)&(recipe>=0)
    dish_action=jnp.where(matching,navigate(pos,facing,jnp.asarray([38,42])),navigate(pos,facing,jnp.asarray([m['return_counter']])))
    action=jnp.where(inv[1]>0,dish_action,action)
    return action,m,{'pos':pos,'recipe':recipe,'mode':jnp.where(partner_plating,3,jnp.where(wrong,2,jnp.where(occupied,1,0))), 'inventory':inv,'handoff':target,'plate_age':m['plate_age']}


def agent_1_act(obs, reset, memory, rng):
    pos,facing,inv,recipe,m=common(obs,reset,memory)
    potmask=obs[:,:,22]
    pot=jnp.sum(obs[:,:,29:34]*potmask[:,:,None],axis=(0,1))
    count=jnp.sum(pot[2:])
    source=jnp.where(recipe==1,47,29)
    timer=jnp.sum(obs[:,:,39]*potmask)
    # Empty hands: learn current order, then load three units and get a plate.
    action=jnp.where(count<3,navigate(pos,facing,jnp.asarray([source])),navigate(pos,facing,jnp.asarray([33,51])))
    # Use cooking time to activate a display that survives this delivery.
    query=navigate(pos,facing,jnp.asarray([22,58]))
    armed=jnp.where(count<3,0,m['armed'])
    prepare=(count==3)&(armed==0)&(jnp.sum(inv)==0)&(m['time']<=350)
    scheduled=jnp.where((query==5)&(timer>5),4,query)
    action=jnp.where(prepare,scheduled,action)
    m['armed']=jnp.where(prepare&(scheduled==5),1,armed).astype(jnp.int32)
    action=jnp.where(recipe<0,query,action)
    action=jnp.where((recipe<0)&(count==0)&(m['time']>358),4,action)
    approach=jnp.where(inv[3]>0,49,31)
    deposit_direction=jnp.where(approach==49,0,1)
    deposit_action=jnp.where(approach==49,3,1)
    deposit=jnp.where(pos==approach,jnp.where(facing==deposit_direction,5,deposit_action),go_to(pos,approach))
    action=jnp.where(jnp.sum(inv[2:])>0,deposit,action)
    scoop=jnp.where(pos==41,jnp.where(facing==3,5,2),go_to(pos,41))
    action=jnp.where(inv[0]>0,scoop,action)
    action=jnp.where(inv[1]>0,navigate(pos,facing,jnp.asarray([38,42])),action)
    return action,m,{'pos':pos,'recipe':recipe,'pot_count':count,'time':m['time'],'timer':timer,'armed':m['armed'],'inventory':inv}
