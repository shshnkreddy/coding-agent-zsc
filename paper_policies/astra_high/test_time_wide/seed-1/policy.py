"""Adaptive local-observation cooking, partner inference, and physical handoffs."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'observable_cook', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

# Known board geometry; no runtime state is shared by the seats.
_BOARD = ('WWXBWW', '0 A  0', '1    1', 'WPWPWW', '3 A  3', 'W    W', 'WWRWWW')
_CODES = {'W': 0, 'X': 1, 'P': 2, 'R': 3, 'L': 4, 'B': 5, '0': 6, '1': 7, '2': 8, '3': 9}
_POTS = ((3,1),(3,3))
_COUNTERS = ((3,2),(3,4),(3,0),(3,5))

def _init():
    return {'t': jnp.int32(0), 'pots': jnp.zeros((2,7), jnp.int32),
            'stage': jnp.int32(0), 'recipe': jnp.int32(0), 'wait': jnp.int32(0),
            'deliveries': jnp.int32(0), 'prev_inv': jnp.zeros((6,),jnp.int32),
            'prev_action': jnp.int32(4), 'target': jnp.int32(0),
            'belief': jnp.asarray([0.72,0.08]+[0.005]*16+[0.12],jnp.float32),
            'blind_p': jnp.float32(0.5), 'signal':jnp.full((9,),-1,jnp.int32), 'last_dish':jnp.int32(0),
            'pending':jnp.bool_(False), 'b_phase':jnp.int32(0), 'b_invert':jnp.int32(0),
            'b_calibrated':jnp.bool_(False), 'b_stall':jnp.int32(0),
            'b_bad':jnp.int32(0), 'b_waited':jnp.bool_(False),
            'b_summary':jnp.zeros((10,),jnp.int32),
            'offer':jnp.int32(0), 'anchor':jnp.int32(-1), 'anchor_age':jnp.int32(100),
            'counter_items':jnp.zeros((4,6),jnp.int32),
            'watch_signature':jnp.full((9,),-1,jnp.int32), 'watch_stable':jnp.int32(0),
            'relocate':jnp.int32(0), 'relocate_clock':jnp.int32(0), 'gather':jnp.int32(-1),
            'gift_phase':jnp.int32(0), 'gift_job_clock':jnp.int32(0),
            'gift_pending':jnp.bool_(False), 'gift_clock':jnp.int32(0),
            'gift_attempted':jnp.bool_(False), 'helper_mode':jnp.bool_(False), 'helper_trial':jnp.bool_(False)}

def init_agent_0(): return _init()
def init_agent_1(): return _init()

def _localize(obs, bottom):
    static = jnp.asarray([[[int(_CODES.get(c, -1) == k) for k in range(10)] for c in row] for row in _BOARD], jnp.int32)
    padded = jnp.pad(static, ((2,2),(2,2),(0,0)))
    positions = jnp.asarray([[r,c] for r in ((4,5) if bottom else (1,2)) for c in (1,2,3,4)], jnp.int32)
    def error(pos):
        tile = jax.lax.dynamic_slice(padded, (pos[0],pos[1],0), (5,5,10))
        return jnp.sum(jnp.abs(tile - obs[:,:,22:32]))
    return positions[jnp.argmin(jax.vmap(error)(positions))]

def _at(obs, pos, target):
    local = target - pos + 2
    visible = jnp.all((local >= 0) & (local < 5))
    value = obs[jnp.clip(local[0],0,4), jnp.clip(local[1],0,4)]
    return value, visible

def _go(pos, direction, target, face):
    # Each room is an unobstructed 2x4 rectangle.
    dr, dc = target[0]-pos[0], target[1]-pos[1]
    move = jnp.where(dc>0, 0, jnp.where(dc<0, 2, jnp.where(dr>0,1,3)))
    turn = jnp.asarray([3,1,0,2],jnp.int32)[face]
    return jnp.where((dr==0)&(dc==0), jnp.where(direction==face,5,turn), move).astype(jnp.int32)

def _common(obs, reset, memory, bottom):
    m = jax.tree.map(lambda a,b: jnp.where(reset,a,b), _init(), memory)
    pos = _localize(obs,bottom)
    direction = jnp.argmax(obs[2,2,1:5]).astype(jnp.int32)
    inv = obs[2,2,5:11]
    cells, visible = jax.vmap(lambda p:_at(obs,pos,p))(jnp.asarray(_POTS))
    old = m['pots']
    timer = jnp.maximum(0,old[:,6]-1)
    old = old.at[:,1].set(jnp.where((old[:,6]>0)&(timer==0),1,old[:,1])).at[:,6].set(timer)
    current = jnp.concatenate([cells[:,32:38],cells[:,44:45]],axis=1)
    pots = jnp.where(visible[:,None],current,old)
    counter_positions = jnp.asarray([[3,2],[3,4],[0,1],[0,4]])
    counter_cells,counter_visible = jax.vmap(lambda p:_at(obs,pos,p))(counter_positions)
    counter_items = jnp.where(counter_visible[:,None],counter_cells[:,32:38],m['counter_items'])
    m = dict(m, t=m['t']+1,pots=pots,counter_items=counter_items)
    return m,pos,direction,inv

def agent_0_act(obs, reset, memory, rng):
    m,pos,direction,inv = _common(obs,reset,memory,False)
    empty = jnp.sum(inv)==0
    plated = inv[0]>0
    cooked = inv[1]>0
    raw = (jnp.sum(inv[2:])>0)&~cooked
    delivered = (m['prev_action']==5)&(m['prev_inv'][1]>0)&empty&jnp.all(pos==jnp.asarray([1,2]))&(direction==0)
    # Pot additions not explained by our own interaction are a partner's
    # concrete ingredient choice, expressed in the common ingredient IDs.
    old_counts = jnp.where(reset,0,memory['pots'][:,2:6])
    own_insert = (m['prev_action']==5)&(m['prev_inv'][0]==0)&(m['prev_inv'][1]==0)&(jnp.sum(m['prev_inv'][2:])>0)&(pos[0]==2)&(direction==1)
    own_pot = jnp.asarray([pos[1]==1,pos[1]==3]) & own_insert
    additions = jnp.maximum(m['pots'][:,2:6]-old_counts-own_pot[:,None]*m['prev_inv'][2:],0)
    old_items = jnp.where(reset,0,memory['counter_items'])
    items = m['counter_items']
    shelf_stands = jnp.asarray([[2,2],[2,4],[1,1],[1,4]])
    shelf_directions = jnp.asarray([1,1,0,0])
    own_dish_drop = (m['prev_action']==5)&(m['prev_inv'][1]>0)&jnp.all(shelf_stands==pos,axis=1)&(shelf_directions==direction)
    received = (items[:,0]>0)&(items[:,1]>0)&(jnp.sum(items[:,2:],axis=1)==3)&(jnp.max(items[:,2:4],axis=1)==3)&jnp.any(items!=old_items,axis=1)&~own_dish_drop
    received_food = jnp.argmax(items[jnp.argmax(received),2:4]).astype(jnp.int32)
    external = ((jnp.sum(additions[:,:2])>0)|jnp.any(received))&~delivered
    external_food = jnp.where(jnp.any(received),received_food,jnp.argmax(jnp.sum(additions[:,:2],axis=0))).astype(jnp.int32)
    anchor = jnp.where(external,external_food,jnp.where(delivered,-1,m['anchor']))
    anchor_age = jnp.where(external,0,m['anchor_age']+1)
    dropped_offer = (m['prev_action']==5)&empty&(m['prev_inv'][0]==0)&(jnp.sum(m['prev_inv'][2:])>0)&(pos[0]==2)&(direction==1)&((pos[1]==2)|(pos[1]==4))
    offer = jnp.where(dropped_offer & (m['offer']<2),m['offer']+1,m['offer'])
    offer = jnp.where((offer<2)&(m['t']>=25),3,offer)
    stage = jnp.where(delivered,0,m['stage'])
    deliveries = m['deliveries']+delivered.astype(jnp.int32)
    # Learn which interpretation of the partner predicts recipe changes.
    # A correct delivery resamples the recipe, whereas a wrong dish cannot
    # change it. Thus a subsequent change in a stable signal is evidence.
    partner_present = jnp.any(obs[:,:,11]>0)
    index = jnp.argmax(obs[:,:,11])
    partner = jnp.asarray([index//5,index%5])+pos-2
    partner_cell = obs[index//5,index%5]
    partner_dir = jnp.argmax(partner_cell[12:16])
    center,_ = _at(obs,pos,jnp.asarray([3,2]))
    right,_ = _at(obs,pos,jnp.asarray([3,4]))
    features = jnp.asarray([
        partner_present & (partner[1]>=3), partner_present,
        partner_present & (partner_dir==2), partner_present & (partner_dir==1),
        partner_present & (partner_cell[21]>0), center[37]>0, right[37]>0,
        partner_present & (partner[1]>=2), partner_present & (partner[1]>=4)
    ],jnp.int32)
    predictions = jnp.stack([features,1-features],axis=1).reshape((-1,))
    at_watch = jnp.all(pos==jnp.asarray([2,2]))
    wait = jnp.where(stage==0,jnp.where(at_watch,m['wait']+1,0),0)
    signature = jnp.concatenate([jnp.where(partner_present,partner,-1),jnp.asarray([partner_dir]),partner_cell[16:22]])
    stable = jnp.where(at_watch & (stage==0),jnp.where(jnp.all(signature==m['watch_signature'])&(m['wait']>0),m['watch_stable']+1,0),0)
    partner_busy = partner_present & (jnp.sum(partner_cell[18:20])>0)
    # The right counter is outside a far-left observer's view. Relocate that
    # offer once, using a private upper shelf for the unchosen central item.
    relocate = m['relocate']
    trigger = (relocate==0)&(offer==2)&at_watch&(wait>=8)&(anchor<0)&partner_present&(partner[1]==1)&(center[34]>0)&(right[35]>0)
    relocate = jnp.where(trigger,1,relocate)
    relocate = jnp.where((relocate==1)&raw,2,relocate)
    relocate = jnp.where((relocate==2)&empty&(m['prev_action']==5),3,relocate)
    relocate = jnp.where((relocate==3)&raw,4,relocate)
    relocate = jnp.where((relocate==4)&empty&(m['prev_action']==5),5,relocate)
    relocate_clock = jnp.where(relocate>0,m['relocate_clock']+1,0)
    relocate = jnp.where(relocate_clock>=25,5,relocate)
    settled = (stable>=1)&~partner_busy
    handoff_ready = (anchor>=0)&(anchor_age>=3)
    initial_ready = (offer==3)|handoff_ready|((wait>=10)&((relocate==0)|(relocate_clock>=25)))
    acquired = (stage==0)&at_watch&(m['t']>=5)&(wait>=3)&(offer>=2)&initial_ready&((relocate==0)|(relocate==5))&(settled|(wait>=12))
    last_dish = jnp.where(delivered,jnp.argmax(m['prev_inv'][2:]),m['last_dish']).astype(jnp.int32)
    blind_p = jnp.where(delivered,jnp.where(last_dish==0,0.5+0.5*m['blind_p'],0.5*m['blind_p']),m['blind_p'])
    pending = m['pending'] | delivered
    changed = jnp.repeat(features != m['signal'],2)
    any_change = jnp.any(changed)
    previous_predictions = jnp.stack([m['signal'],1-m['signal']],axis=1).reshape((-1,))
    expected_correct = previous_predictions == last_dish
    likelihood = jnp.where(expected_correct,0.5,jnp.where(changed,0.002,0.998))
    # Stable alternative channels should change together with the recipe.
    # Keep a small nuisance probability rather than ruling hypotheses out.
    likelihood = jnp.where(any_change & ~changed,0.015,likelihood)
    likelihood = jnp.concatenate([likelihood,jnp.asarray([jnp.where(any_change,0.02,0.98)])])
    update = acquired & pending & jnp.all(m['signal']>=0) & settled
    posterior = jnp.maximum(m['belief']*likelihood,1.e-8)
    posterior = posterior/jnp.sum(posterior)
    belief = jnp.where(update,posterior,m['belief'])
    ground_likelihood = jnp.concatenate([jnp.where(predictions==anchor,1.0,0.005),jnp.asarray([0.1])])
    grounded = belief*ground_likelihood
    grounded = grounded/jnp.sum(grounded)
    belief = jnp.where(acquired & (anchor>=0) & settled,grounded,belief)
    blind_p = jnp.where(acquired & (anchor>=0),anchor.astype(jnp.float32),blind_p)
    p_one = jnp.sum(belief[:-1]*predictions) + belief[-1]*blind_p
    p_one = jnp.where(partner_busy & (anchor<0),blind_p,p_one)
    recipe = jnp.where(acquired,(p_one>0.5).astype(jnp.int32),m['recipe'])
    recipe = jnp.where(anchor>=0,anchor,recipe)
    offer = jnp.where(acquired,3,offer)
    signal = jnp.where(acquired,jnp.where(settled,features,-1),jnp.where(cooked & at_watch,jnp.where(~partner_busy,features,-1),m['signal']))
    pending = pending & ~acquired
    stage = jnp.where(acquired,1,stage)
    # A plate offer is a bounded fallback for an otherwise uninformative
    # central observer. Actual plate acceptance and returned soup determine
    # whether cooperative plating is continued.
    gift_phase = m['gift_phase']
    gift_phase = jnp.where((gift_phase==1)&~empty,2,gift_phase)
    gift_phase = jnp.where((gift_phase==2)&empty&(m['prev_action']==5),3,gift_phase)
    gift_phase = jnp.where((gift_phase==3)&plated&~cooked,4,gift_phase)
    issued_plate = (gift_phase==4)&empty&(m['prev_action']==5)&(m['prev_inv'][0]>0)&(m['prev_inv'][1]==0)
    gift_phase = jnp.where(issued_plate,0,gift_phase)
    own_plate_pick = plated&~cooked&(jnp.sum(m['prev_inv'])==0)&(m['prev_action']==5)&jnp.all(pos==jnp.asarray([2,2]))&(direction==1)
    gift_pending = (m['gift_pending']|issued_plate)&~delivered&~own_plate_pick
    gift_clock = jnp.where(issued_plate,0,jnp.where(gift_pending,m['gift_clock']+1,0))
    accepted_plate = gift_pending&(jnp.sum(center[32:38])==0)&~plated
    helper_mode = m['helper_mode']|(jnp.any(received)&m['gift_pending'])
    helper_trial = (m['helper_trial']|accepted_plate)&~delivered
    expired = gift_pending&(gift_clock>=60)
    helper_mode = helper_mode&~expired
    helper_trial = helper_trial&~expired
    gift_pending = gift_pending&~expired
    private_free = jnp.sum(m['counter_items'][2:],axis=1)==0
    center_free = jnp.sum(center[32:38])==0
    center_plate = (center[32]>0)&(center[33]==0)&(jnp.sum(center[34:38])==0)
    offer_item = (jnp.sum(center[34:36])==1)&(jnp.sum(center[36:38])==0)&(center[32]==0)&(center[33]==0)
    central_waiter = (center_free|center_plate|offer_item)&acquired&(m['t']>=50)&(deliveries>0)&partner_present&(partner[1]==2)&(jnp.sum(partner_cell[16:22])==0)&(anchor<0)&(jnp.max(belief[:-1])<0.7)
    try_gift = (~m['gift_attempted']&central_waiter)|(helper_mode&~gift_pending)
    try_gift = try_gift&empty&(gift_phase==0)&(offer==3)&((relocate==0)|(relocate==5))&(center_free|center_plate|jnp.any(private_free))
    gift_attempted = m['gift_attempted']|try_gift
    gift_pending = gift_pending|(try_gift&center_plate)
    gift_phase = jnp.where(try_gift&~center_plate,jnp.where(center_free,3,1),gift_phase)
    gift_job_clock = jnp.where(gift_phase>0,m['gift_job_clock']+1,0)
    job_expired = gift_job_clock>=25
    gift_phase = jnp.where(job_expired,0,gift_phase)
    helper_mode = helper_mode&~job_expired
    helper_trial = helper_trial&~job_expired
    helper_running = helper_mode|helper_trial
    # Honor concrete partial pot contents before adding ingredients.
    counts = m['pots'][:,2:6]
    sizes = jnp.sum(counts,axis=1)
    homogeneous = jnp.max(counts,axis=1)==sizes
    compatible = homogeneous & (jnp.argmax(counts,axis=1)==recipe) & (sizes>0)
    empty_pots = sizes==0
    empty_target = jnp.where(empty_pots[recipe],recipe,jnp.argmax(empty_pots))
    pot_priority = 100*m['pots'][:,1]+jnp.where(m['pots'][:,6]>0,50-m['pots'][:,6],sizes)
    good_target = jnp.argmax(jnp.where(compatible,pot_priority,-1000))
    recovery = ~jnp.any(compatible|empty_pots)
    recovery_target = jnp.argmax(pot_priority)
    target = jnp.where(jnp.any(compatible),good_target,jnp.where(jnp.any(empty_pots),empty_target,recovery_target)).astype(jnp.int32)
    pot = m['pots'][target]
    n = sizes[target]
    potpos = jnp.asarray([2,1+2*target],jnp.int32)
    pot_action = _go(pos,direction,potpos,1)
    other = 1-target
    other_usable = (sizes[other]<3)&(m['pots'][other,1]==0)&(m['pots'][other,6]==0)&homogeneous[other]&(jnp.sum(counts[other,2:])==0)
    finished_transfer = empty & (m['prev_action']==5) & (m['prev_inv'][0]==0) & (jnp.sum(m['prev_inv'][2:])>0)
    gather = jnp.where(finished_transfer,-1,m['gather'])
    gather_index = jnp.maximum(gather,0)
    gather_valid = (gather>=0)&(sizes[gather_index]<3)&(m['pots'][gather_index,1]==0)&(m['pots'][gather_index,6]==0)&homogeneous[gather_index]&(jnp.sum(counts[gather_index,2:])==0)
    gather = jnp.where(gather_valid,gather,-1)
    background = (((pot[6]>1)|(helper_running&(n>=3)))&other_usable)|gather_valid
    work_target = jnp.where(gather_valid,gather_index,jnp.where(background,other,target))
    # An ingredient already in hand always goes to a compatible pot.
    heldtype = jnp.argmax(inv[2:])
    can_insert = (sizes<3)&(m['pots'][:,1]==0)&(m['pots'][:,6]==0)&((sizes==0)|(counts[:,heldtype]==sizes)|(recovery & (jnp.arange(2)==target)))
    expected_food = jnp.where(sizes>0,jnp.argmax(counts,axis=1),jnp.where(jnp.arange(2)==target,recipe,1-recipe))
    wanted_raw = can_insert & (expected_food==heldtype)
    preferred_raw = jnp.where(wanted_raw[work_target],work_target,jnp.where(jnp.any(wanted_raw),jnp.argmax(wanted_raw),jnp.argmax(can_insert)))
    work_target = jnp.where(raw,preferred_raw,work_target).astype(jnp.int32)
    work_food = jnp.where(sizes[work_target]>0,jnp.argmax(counts[work_target]),jnp.where(work_target==target,recipe,1-recipe)).astype(jnp.int32)
    work_food = jnp.where(work_food<2,work_food,recipe)
    workpos = jnp.asarray([2,1+2*work_target],jnp.int32)
    put_action = _go(pos,direction,workpos,1)
    source_side = work_target
    supplypos = jnp.asarray([1+jnp.minimum(work_food,1),jnp.where(source_side==0,1,4)],jnp.int32)
    supply_action = _go(pos,direction,supplypos,jnp.where(source_side==0,3,2))
    plate_action = _go(pos,direction,jnp.asarray([1,3]),0)
    deliver_action = _go(pos,direction,jnp.asarray([1,2]),0)
    # Counter items are genuine resources. Keep off-target dishes for later,
    # and free hands when the partner changes a pot while we carry inventory.
    shelves = jnp.asarray([[2,2],[2,4],[1,1],[1,4]])
    shelf_faces = jnp.asarray([1,1,0,0])
    shelf_items = m['counter_items']
    shelf_cost = jnp.sum(jnp.abs(shelves-pos),axis=1)
    free_shelves = jnp.sum(shelf_items,axis=1)==0
    free_shelf = jnp.argmin(jnp.where(free_shelves,shelf_cost,1000))
    discard_action = _go(pos,direction,shelves[free_shelf],shelf_faces[free_shelf])
    ready_shelves = (shelf_items[:,0]>0)&(shelf_items[:,1]>0)&(shelf_items[:,2+recipe]==3)&(jnp.sum(shelf_items[:,2:],axis=1)==3)&(~helper_running|(anchor>=0))
    ready_shelf = jnp.argmin(jnp.where(ready_shelves,shelf_cost,1000))
    take_dish = _go(pos,direction,shelves[ready_shelf],shelf_faces[ready_shelf])
    empty_plates = (shelf_items[:,0]>0)&(shelf_items[:,1]==0)&(jnp.sum(shelf_items[:,2:],axis=1)==0)
    empty_plates = empty_plates & ~((jnp.arange(4)==0)&gift_pending&(gift_clock<15))
    plate_shelf = jnp.argmin(jnp.where(empty_plates,shelf_cost,1000))
    use_stored_plate = jnp.any(empty_plates)&(shelf_cost[plate_shelf]<=jnp.sum(jnp.abs(pos-jnp.asarray([1,3]))))
    plate_action = jnp.where(use_stored_plate,_go(pos,direction,shelves[plate_shelf],shelf_faces[plate_shelf]),plate_action)
    action = jnp.where((n>=3)&~background,plate_action,supply_action)
    action = jnp.where(raw,jnp.where(jnp.any(can_insert),put_action,discard_action),action)
    action = jnp.where(plated,pot_action,action)
    action = jnp.where(plated & (pot[1]==0) & jnp.all(pos==potpos),4,action)
    action = jnp.where(plated & ~cooked & (n<3) & jnp.any(free_shelves),discard_action,action)
    held_matches = (inv[2+recipe]==3)&(jnp.sum(inv[2:])==3)
    cooked_action = jnp.where(held_matches|~jnp.any(free_shelves),deliver_action,discard_action)
    action = jnp.where(cooked,cooked_action,action)
    action = jnp.where(helper_running&empty&(n>=3)&~background,4,action)
    action = jnp.where(empty & jnp.any(ready_shelves),take_dish,action)
    action = jnp.where(stage==0,_go(pos,direction,jnp.asarray([2,2]),1),action)
    action = jnp.where((stage==0)&at_watch,4,action)
    offer_type = jnp.minimum(offer,1)
    offer_source = jnp.asarray([1+offer_type,1+3*offer_type],jnp.int32)
    offer_counter = jnp.asarray([2,2+2*offer_type],jnp.int32)
    offer_take = _go(pos,direction,offer_source,jnp.where(offer_type==0,3,2))
    offer_put = _go(pos,direction,offer_counter,1)
    offer_action = jnp.where(raw,offer_put,offer_take)
    action = jnp.where(offer<2,offer_action,action)
    relocate_actions = jnp.asarray([
        action,
        _go(pos,direction,jnp.asarray([2,2]),1),
        _go(pos,direction,jnp.asarray([1,4]),0),
        _go(pos,direction,jnp.asarray([2,4]),1),
        _go(pos,direction,jnp.asarray([2,2]),1),
        action
    ])
    action = jnp.where((relocate>0)&(relocate<5),relocate_actions[relocate],action)
    private_slot = 2+jnp.argmin(jnp.where(private_free,shelf_cost[2:],1000))
    usable_transfer = raw & (heldtype<2) & jnp.any(wanted_raw)
    gift_store = jnp.where(usable_transfer,put_action,_go(pos,direction,shelves[private_slot],shelf_faces[private_slot]))
    gift_actions = jnp.asarray([
        action,
        _go(pos,direction,jnp.asarray([2,2]),1),
        gift_store,
        _go(pos,direction,jnp.asarray([1,3]),0),
        _go(pos,direction,jnp.asarray([2,2]),1)
    ])
    action = jnp.where(gift_phase>0,gift_actions[gift_phase],action)
    start_gather = (gather<0)&empty&background&(stage==1)&(offer==3)&((relocate==0)|(relocate==5))&~jnp.any(ready_shelves)&(gift_phase==0)
    gather = jnp.where(start_gather,work_target,gather)
    m = dict(m,gift_phase=gift_phase,gift_job_clock=gift_job_clock,gift_pending=gift_pending,gift_clock=gift_clock,gift_attempted=gift_attempted,helper_mode=helper_mode,helper_trial=helper_trial,gather=gather,stage=stage,recipe=recipe,wait=wait,deliveries=deliveries,target=target,prev_inv=inv,prev_action=action,belief=belief,blind_p=blind_p,signal=signal,last_dish=last_dish,pending=pending,offer=offer,anchor=anchor,anchor_age=anchor_age,watch_signature=signature,watch_stable=stable,relocate=relocate,relocate_clock=relocate_clock)
    return action.astype(jnp.int32),m,{'position':pos,'recipe':recipe,'stage':stage,'target_pot':target,'work_pot':work_target,'deliveries':deliveries,'belief':belief,'blind_p':blind_p,'anchor':anchor,'offer':offer,'recovery':recovery,'relocate':relocate,'gather':gather,'gift_phase':gift_phase,'helper_mode':helper_mode,'helper_trial':helper_trial}

def agent_1_act(obs, reset, memory, rng):
    m,pos,direction,inv = _common(obs,reset,memory,True)
    display = jnp.sum(obs[:,:,40:44],axis=(0,1))
    recipe = jnp.where(jnp.sum(display)>0,jnp.argmax(display),m['recipe']).astype(jnp.int32)
    # Revisit request interpretation on concrete, non-background pot starts.
    sizes = jnp.sum(m['pots'][:,2:6],axis=1)
    partner_present = jnp.any(obs[:,:,11]>0)
    index = jnp.argmax(obs[:,:,11])
    partner_cell = obs[index//5,index%5]
    partner_pos = jnp.where(partner_present,jnp.asarray([index//5,index%5])+pos-2,-1)
    summary = jnp.concatenate([partner_pos,partner_cell[16:22],sizes])
    stationary = jnp.all(summary==m['b_summary']) & (jnp.sum(partner_cell[16:22])==0)
    stationary = stationary & ~jnp.any(m['pots'][:,6]>0) & (jnp.sum(inv)==0)
    stall = jnp.where(stationary,m['b_stall']+1,0)
    waited = m['b_waited'] | (stall>=6)
    switch = stall>=14
    phase = jnp.where(switch,(m['b_phase']+1)%6,m['b_phase'])
    calibrated = m['b_calibrated'] & ~switch
    fresh = (sizes > m['b_summary'][-2:]) & (m['b_summary'][-2:]==0)
    first = jnp.argmax(fresh)
    response_food = jnp.argmax(m['pots'][first,2:6])
    other_matches = (m['pots'][1-first,2+recipe]>0) | (memory['pots'][1-first,2+recipe]>0)
    own_put = (m['prev_action']==5)&(m['prev_inv'][0]==0)&(jnp.sum(m['prev_inv'][2:])>0)&(pos[0]==4)&(direction==0)&((pos[1]==1)|(pos[1]==3))
    response = jnp.any(fresh) & (response_food<2) & (m['t']>=5) & ~other_matches & ~own_put & (recipe==m['recipe'])
    invert = jnp.where(response,m['b_invert'] ^ (response_food!=recipe).astype(jnp.int32),m['b_invert'])
    calibrated = calibrated | response
    bad = jnp.where(response,jnp.where(response_food!=recipe,m['b_bad']+1,0),m['b_bad'])
    channel_failed = response & (bad>=jnp.where(waited,2,3))
    alternative = jnp.where(phase==4,5,jnp.where(phase==5,1,jnp.where(waited,5,4)))
    phase = jnp.where(channel_failed,alternative,phase)
    invert = jnp.where(channel_failed,0,invert)
    bad = jnp.where(channel_failed,0,bad)
    bit = recipe ^ invert
    # Try another simple visible request only when the partner remains idle.
    columns = jnp.asarray([1+2*bit,2+bit,1+3*bit,2+2*bit,2,2])
    goal = jnp.asarray([jnp.where(phase==4,4+bit,4),columns[phase]],jnp.int32)
    action = _go(pos,direction,goal,0)
    at_goal = jnp.all(pos==goal)
    action = jnp.where(at_goal & ((direction==0)|(phase==4)),4,action)
    # Facing at a floor cell is obtained by leaving and re-entering it.
    desired_face = jnp.where(bit==0,3,2)
    face_action = jnp.where(direction==desired_face,4,jnp.where(bit==0,0,2))
    action = jnp.where((phase==5)&at_goal,face_action,action)
    # If a partner offers real ingredients on a shared counter, use matching ones.
    helper_counters = jnp.asarray([[3,2],[3,4],[6,1],[6,3],[6,4],[5,0],[5,5]])
    helper_stands = jnp.asarray([[4,2],[4,4],[5,1],[5,3],[5,4],[5,1],[5,4]])
    helper_faces = jnp.asarray([0,0,1,1,1,3,2])
    cells,visible = jax.vmap(lambda p:_at(obs,pos,p))(helper_counters)
    good = visible & (cells[:,33]==0) & (cells[:,34+recipe]>0)
    chosen = jnp.argmax(good)
    counter = helper_counters[chosen]
    stand = helper_stands[chosen]
    face = helper_faces[chosen]
    take = _go(pos,direction,stand,face)
    pots = m['pots']
    sizes = jnp.sum(pots[:,2:6],axis=1)
    has_matching = jnp.any(pots[:,2+recipe]>0)
    valid = (sizes<3)&(pots[:,1]==0)&(pots[:,6]==0)&((sizes==0)|(pots[:,2+recipe]==sizes))&((sizes>0)|~has_matching)
    preferred = jnp.where(valid[recipe],recipe,jnp.argmax(valid))
    put = _go(pos,direction,jnp.asarray([4,1+2*preferred]),0)
    empty = jnp.sum(inv)==0
    matching = (inv[2+recipe]>0)&(inv[1]==0)&(inv[0]==0)
    action = jnp.where(empty&jnp.any(good)&jnp.any(valid),take,action)
    action = jnp.where(matching&jnp.any(valid),put,action)
    # Plate the visibly correct recipe when a partner supplies a plate.
    plate_pots = (pots[:,2+recipe]==3)&(sizes==3)&((pots[:,1]>0)|(pots[:,6]>0))
    plate_target = jnp.argmax(jnp.where(plate_pots,100*pots[:,1]+20-pots[:,6],-1000))
    plate_put = _go(pos,direction,jnp.asarray([4,1+2*plate_target]),0)
    counter_plates = visible & (cells[:,32]>0)&(cells[:,33]==0)&(jnp.sum(cells[:,34:38],axis=1)==0)
    plate_source = jnp.argmax(counter_plates)
    take_plate = _go(pos,direction,helper_stands[plate_source],helper_faces[plate_source])
    plate_in_hand = (inv[0]>0)&(inv[1]==0)
    action = jnp.where(empty & jnp.any(counter_plates)&jnp.any(plate_pots),take_plate,action)
    action = jnp.where(plate_in_hand&jnp.any(plate_pots),plate_put,action)
    good_dishes = visible&(cells[:,32]>0)&(cells[:,33]>0)&(cells[:,34+recipe]==3)&(jnp.sum(cells[:,34:38],axis=1)==3)
    private_dishes = good_dishes & (jnp.arange(7)>=2)
    cached = jnp.argmax(private_dishes)
    take_cached = _go(pos,direction,helper_stands[cached],helper_faces[cached])
    action = jnp.where(empty & jnp.any(private_dishes)&~jnp.any(good_dishes[:2]),take_cached,action)
    free = visible & (jnp.sum(cells[:,32:38],axis=1)==0)
    good_dish = (inv[1]>0)&(inv[2+recipe]==3)&(jnp.sum(inv[2:])==3)
    wrong_dish = (inv[1]>0)&~good_dish
    useful = ((jnp.sum(inv[2:4])>0)&(inv[1]==0))|plate_in_hand|good_dish
    shelf_cost = jnp.sum(jnp.abs(helper_stands-pos),axis=1)+jnp.where(useful & (jnp.arange(7)>=2),4,0)
    shelf_cost = shelf_cost+jnp.where(wrong_dish & (jnp.arange(7)<2),100,0)+jnp.where(good_dish & (jnp.arange(7)!=0),6,0)
    shelf = jnp.argmin(jnp.where(free,shelf_cost,1000))
    drop = _go(pos,direction,helper_stands[shelf],helper_faces[shelf])
    recover = ~empty & ~(matching&jnp.any(valid)) & ~(plate_in_hand&jnp.any(plate_pots))
    action = jnp.where(recover & jnp.any(free),drop,action)
    m = dict(m,recipe=recipe,prev_inv=inv,prev_action=action,b_phase=phase,b_invert=invert,b_calibrated=calibrated,b_stall=jnp.where(switch,0,stall),b_summary=summary,b_bad=bad,b_waited=waited)
    return action.astype(jnp.int32),m,{'position':pos,'recipe':recipe,'signal_phase':phase,'signal_invert':invert,'calibrated':calibrated}
