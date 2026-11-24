"""Local-observation cooking with map-grounded navigation."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'grounded_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 9, 'max_width': 9, 'num_ingredients': 3}

from custom_primitives import TEMPLATES, ACTIONS, COSTS, STATIC, FLOOR

def _decode(obs):
    features=obs[:,:,20:29].reshape(25,9)
    error=jnp.sum(jnp.abs(jnp.asarray(TEMPLATES)-features[None,:,:]),axis=(1,2))
    pos=jnp.argmin(error).astype(jnp.int32)
    direction=jnp.argmax(obs[2,2,1:5]).astype(jnp.int32)
    return pos,direction,obs[2,2,5:10]

def _nav(target,pos,direction):
    return jnp.asarray(ACTIONS,dtype=jnp.int32)[target,pos*4+direction]

def _costs(pos,direction):
    return jnp.asarray(COSTS,dtype=jnp.int32)[:,pos*4+direction]

def _nearest(targets,pos,direction):
    targets=jnp.asarray(targets,dtype=jnp.int32)
    return targets[jnp.argmin(_costs(pos,direction)[targets])]

def init_agent_0():
    return {'recipe':jnp.int32(-1),'items':jnp.zeros((81,5),jnp.int32),'phase':jnp.int32(0),'was_dish':jnp.int32(0),'step':jnp.int32(0),'ack_step':jnp.int32(0),'empty_steps':jnp.int32(0)}

def agent_0_act(obs,reset,memory,rng):
    memory={k:jnp.where(reset,v,memory[k]) for k,v in init_agent_0().items()}
    pos,direction,held=_decode(obs)
    displayed=jnp.max(obs[:,:,36:39],axis=(0,1))
    recipe=jnp.where(jnp.sum(displayed)>0,jnp.argmax(displayed),memory['recipe']).astype(jnp.int32)
    indices=jnp.arange(81)
    rr=indices//9-pos//9+2; cc=indices%9-pos%9+2
    visible=(rr>=0)&(rr<5)&(cc>=0)&(cc<5)
    items=jnp.where(visible[:,None],obs[jnp.clip(rr,0,4),jnp.clip(cc,0,4),29:34],memory['items'])
    partner=jnp.sum(obs[:,:,15:20]*obs[:,:,10,None],axis=(0,1))
    partner_visible=jnp.sum(obs[:,:,10])>0
    partner_empty=partner_visible&(jnp.sum(partner)==0)
    empty_steps=jnp.where(partner_empty,memory['empty_steps']+1,0)
    delivered=(memory['was_dish']>0)&partner_empty
    changed=(recipe>=0)&(recipe!=memory['recipe'])
    retry=(memory['phase']==2)&(memory['step']-memory['ack_step']>30)&(empty_steps>=6)
    phase=jnp.where(changed|delivered|retry,0,memory['phase'])
    acknowledged=partner_visible&((partner[2+jnp.maximum(recipe,0)]>0)|(partner[0]>0))
    acknowledged=acknowledged&(recipe>=0)&(phase==0)
    phase=jnp.where(acknowledged,1,phase)
    ack_step=jnp.where(acknowledged,memory['step'],memory['ack_step'])
    has_item=jnp.sum(held)>0
    phase=jnp.where((phase==1)&~has_item,2,phase)
    costs=_costs(pos,direction)
    walls=jnp.asarray(STATIC)[:,0]>0
    empty=jnp.sum(items,axis=1)==0
    preferred=jnp.where(held[3]>0,56,20)
    drop=jnp.argmin(jnp.where(walls&empty,costs,9999))
    drop=jnp.where(empty[preferred],preferred,drop)
    correct=(items[:,2+jnp.maximum(recipe,0)]==1)&(items[:,0]==0)&(items[:,1]==0)&walls
    source=jnp.where(recipe==0,29,47)
    pickup=jnp.where(jnp.any(correct),jnp.argmin(jnp.where(correct,costs,9999)),source)
    matches=(held[2+jnp.maximum(recipe,0)]==1)&(held[0]==0)&(held[1]==0)
    offer=jnp.where(has_item,jnp.where(matches,37,drop),pickup)
    target=jnp.where((phase==1)&has_item,drop,jnp.where(phase==2,37,offer))
    target=jnp.where(recipe<0,37,target)
    action=_nav(target,pos,direction)
    was_dish=jnp.where(partner_visible,partner[1],memory['was_dish'])
    return action,{'recipe':recipe,'items':items,'phase':phase,'was_dish':was_dish,'step':memory['step']+1,'ack_step':ack_step,'empty_steps':empty_steps},{'position':pos,'recipe':recipe,'target':target,'phase':phase,'held':held}

def init_agent_1():
    return {'recipe':jnp.int32(-1),'was_dish':jnp.int32(0),'step':jnp.int32(0),'helper_button':jnp.int32(-1),'calibration':jnp.int32(0),'verified':jnp.int32(0),'empty_seen':jnp.int32(0),'since_delivery':jnp.int32(400),'inferred':jnp.int32(0),'wait_failed':jnp.int32(0),'was_waiting':jnp.int32(0)}

def agent_1_act(obs,reset,memory,rng):
    memory={k:jnp.where(reset,v,memory[k]) for k,v in init_agent_1().items()}
    pos,direction,held=_decode(obs)
    displayed=jnp.max(obs[:,:,36:39],axis=(0,1))
    delivered=(memory['was_dish']>0)&(held[1]==0)
    recipe=jnp.where(delivered,-1,memory['recipe'])
    recipe=jnp.where(jnp.sum(displayed)>0,jnp.argmax(displayed),recipe).astype(jnp.int32)
    partner=jnp.sum(obs[:,:,15:20]*obs[:,:,10,None],axis=(0,1))
    partner_visible=jnp.sum(obs[:,:,10])>0
    partner_empty=partner_visible&(jnp.sum(partner)==0)
    raw=(jnp.sum(partner[2:])==1)&(partner[0]==0)&(partner[1]==0)&(partner[4]==0)
    signal=jnp.argmax(partner[2:]).astype(jnp.int32)
    shown=jnp.sum(displayed)>0
    verified=jnp.where(delivered,0,memory['verified'])
    verified=jnp.where(shown,1,verified)
    since_delivery=jnp.where(delivered,0,memory['since_delivery']+1)
    empty_seen=jnp.where(delivered,0,memory['empty_seen'])
    empty_seen=jnp.where(partner_empty,1,empty_seen)
    calibration=memory['calibration']
    matched=raw&(verified>0)&(recipe==signal)
    calibration=jnp.where(matched,jnp.bitwise_or(calibration,jnp.left_shift(jnp.int32(1),signal)),calibration)
    contradicted=raw&(verified>0)&(recipe!=signal)&(since_delivery>12)
    calibration=jnp.where(contradicted,0,calibration)
    inferred=jnp.where(shown,0,memory['inferred'])
    receptive=(calibration==3)&(empty_seen>0)&(inferred<2)
    accepted=(recipe<0)&receptive&raw
    recipe=jnp.where(accepted,signal,recipe)
    inferred=inferred+accepted.astype(jnp.int32)
    timeout=(recipe<0)&(memory['was_waiting']>0)&(since_delivery>=6)
    wait_failed=jnp.maximum(memory['wait_failed'],timeout.astype(jnp.int32))
    wait_signal=(recipe<0)&receptive&(since_delivery<6)&(wait_failed==0)
    pot_mask=obs[:,:,22]
    pot=jnp.sum(obs[:,:,29:34]*pot_mask[:,:,None],axis=(0,1))
    timer=jnp.sum(obs[:,:,39]*pot_mask)
    count=jnp.sum(pot[2:])
    source=jnp.where(recipe==0,29,47)
    partner_cell=jnp.argmax(obs[:,:,10]).astype(jnp.int32)
    partner_r=pos//9+partner_cell//5-2
    partner_c=pos%9+partner_cell%5-2
    helper_here=(jnp.sum(obs[:,:,10])>0)&(partner_c==4)&((partner_r==1)|(partner_r==7))
    helper_button=jnp.where(helper_here,jnp.where(partner_r==1,22,58),memory['helper_button'])
    button=_nearest((22,58),pos,direction)
    costs=_costs(pos,direction)
    button=jnp.where((helper_button>=0)&(costs[jnp.maximum(helper_button,0)]<=costs[button]),helper_button,button)
    plate=_nearest((33,51),pos,direction)
    goal=_nearest((38,42),pos,direction)
    target=jnp.where(recipe<0,button,jnp.where(count<3,source,plate))
    target=jnp.where(jnp.sum(held[2:])>0,40,target)
    target=jnp.where(held[0]>0,40,target)
    target=jnp.where(held[1]>0,goal,target)
    target=jnp.where(wait_signal,39,target)
    action=_nav(target,pos,direction)
    approach=jnp.where(held[0]>0,39,jnp.where(held[3]>0,49,31))
    pot_direction=jnp.where(approach==39,2,jnp.where(approach==49,0,1))
    turn=jnp.asarray((3,1,0,2),jnp.int32)[pot_direction]
    pot_action=jnp.where(pos==approach,jnp.where(direction==pot_direction,5,turn),_nav(approach,pos,direction))
    action=jnp.where((target==40)&(held[1]==0),pot_action,action)
    action=jnp.where((recipe<0)&(memory['step']>360),4,action)
    action=jnp.where((held[0]>0)&(held[1]==0)&(pot[1]==0)&(action==5),4,action)
    return action,{'recipe':recipe,'was_dish':held[1],'step':memory['step']+1,'helper_button':helper_button,'calibration':calibration,'verified':verified,'empty_seen':empty_seen,'since_delivery':since_delivery,'inferred':inferred,'wait_failed':wait_failed,'was_waiting':wait_signal.astype(jnp.int32)},{'position':pos,'recipe':recipe,'target':target,'pot_count':count,'timer':timer,'held':held,'calibration':calibration,'accepted':accepted,'wait_signal':wait_signal,'inferred':inferred,'wait_failed':wait_failed}
