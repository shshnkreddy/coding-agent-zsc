"""Local-observation cooking and visible recipe indication."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'prospective_cue_learning', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 7, 'max_width': 6, 'num_ingredients': 4}

BOARD = ('WWXBWW', '0    0', '1    1', 'WPWPWW', '3    3', 'W    W', 'WWRWWW')
TYPES = 'WXPRLB0123'

def initial():
    return {'step': jnp.int32(0), 'recipe': jnp.int32(-1), 'last': jnp.int32(1),
            'wait': jnp.int32(0), 'grid': jnp.zeros((7, 6, 7), jnp.int32),
            'signal': jnp.int32(-1), 'stable': jnp.int32(0),
            'odds': jnp.ones((11,), jnp.int32), 'quality': jnp.zeros((11,), jnp.int32),
            'codes': jnp.zeros((11,), jnp.int32), 'cycle_codes': jnp.zeros((11,), jnp.int32),
            'active': jnp.int32(0), 'lower_locked': jnp.int32(0), 'lower_flip': jnp.int32(0),
            'handoff': jnp.int32(0), 'offer_done': jnp.int32(0), 'offer_age': jnp.int32(0),
            'labels': jnp.zeros((2,11),jnp.int32), 'label_seen': jnp.zeros((2,),jnp.bool_),
            'label_bad': jnp.zeros((11,),jnp.bool_), 'prospective': jnp.zeros((11,),jnp.bool_),
            'early_codes': jnp.zeros((11,),jnp.int32), 'early_valid': jnp.bool_(False), 'early_last': jnp.int32(1),
            'last_delivery_codes': jnp.full((11,),-1,jnp.int32), 'direct': jnp.int32(0),
            'offer_col': jnp.int32(2), 'partner_busy': jnp.int32(0), 'physical': jnp.int32(0), 'physical_age': jnp.int32(0),
            'pending': jnp.int32(0), 'cycle_signal': jnp.int32(-1)}

def init_agent_0():
    return initial()

def init_agent_1():
    return initial()

def decode(obs, mem, reset, seat):
    mem = jax.tree.map(lambda v, z: jnp.where(reset, z, v), mem, initial())
    rows = jnp.repeat(jnp.arange(1 if seat == 0 else 4, 3 if seat == 0 else 6), 4)
    cols = jnp.tile(jnp.arange(1, 5), 2)
    rr = rows[:, None, None] + jnp.arange(5)[None, :, None] - 2
    cc = cols[:, None, None] + jnp.arange(5)[None, None, :] - 2
    valid = (rr >= 0) & (rr < 7) & (cc >= 0) & (cc < 6)
    static = jnp.asarray([[[int(ch == t) for t in TYPES] for ch in row] for row in BOARD], jnp.int32)
    expected = static[jnp.clip(rr, 0, 6), jnp.clip(cc, 0, 5)] * valid[..., None]
    err = jnp.sum(jnp.abs(expected - obs[None, :, :, 22:32]), axis=(1, 2, 3))
    k = jnp.argmin(err)
    r, c = rows[k], cols[k]
    gr, gc = jnp.arange(7)[:, None], jnp.arange(6)[None, :]
    lr, lc = gr-r+2, gc-c+2
    vis = (lr >= 0) & (lr < 5) & (lc >= 0) & (lc < 5)
    local = obs[jnp.clip(lr, 0, 4), jnp.clip(lc, 0, 4)]
    dyn = jnp.concatenate([local[:, :, 32:38], local[:, :, 44:45]], axis=-1)
    mem['grid'] = mem['grid'].at[:, :, 6].set(jnp.maximum(0, mem['grid'][:, :, 6]-1))
    mem['grid'] = jnp.where(vis[..., None], dyn, mem['grid'])
    mem['step'] += 1
    return mem, r, c, jnp.argmax(obs[2, 2, 1:5]), obs[2, 2, 5:11]

def go(r, c, face, tr, tc, td, interact=True):
    move = jnp.where(c < tc, 0, jnp.where(c > tc, 2, jnp.where(r < tr, 1, 3)))
    vertical_first=jnp.where(r < tr,1,jnp.where(r > tr,3,jnp.where(c < tc,0,2)))
    move=jnp.where(td>=2,vertical_first,move)
    at = (r == tr) & (c == tc)
    final = jnp.where(face == td, 5 if interact else 4, jnp.asarray([3, 1, 0, 2], jnp.int32)[td])
    return jnp.where(at, final, move).astype(jnp.int32)

def choose_pot(grid, ingredient):
    pots = grid[3, jnp.asarray([1,3])]
    counts = jnp.sum(pots[:,2:6], axis=1)
    matching = (pots[:,ingredient+2] == counts)
    score = jnp.where(matching, counts*10 + pots[:,1]*20 + (jnp.arange(2)==ingredient), -100)
    index = jnp.argmax(score)
    return jnp.asarray([1,3])[index], pots[index], jnp.max(score) >= 0

def agent_1_act(obs, reset, memory, rng):
    m, r, c, d, inv = decode(obs, memory, reset, 1)
    rec = jnp.sum(obs[:, :, 40:44], axis=(0, 1))
    m['recipe'] = jnp.where(jnp.sum(rec) > 0, jnp.argmax(rec), m['recipe'])
    recipe = jnp.maximum(0, m['recipe'])
    potc, pot, usable = choose_pot(m['grid'], recipe)
    counters = m['grid'][3, jnp.asarray([2,4])]
    cn = jnp.sum(counters[:,:6],axis=1)
    rawok = (counters[:,recipe+2] == 1) & (cn == 1)
    plates = (counters[:,0] == 1) & (cn == 1)
    empty = cn == 0
    rawc = jnp.asarray([2,4])[jnp.argmax(rawok)]
    platec = jnp.asarray([2,4])[jnp.argmax(plates)]
    dropc = jnp.asarray([2,4])[jnp.argmax(empty)]
    # The first committed cooking choice reveals how a partner read our position.
    # Lock this once so later spare preparation or exploration cannot cause oscillation.
    pots=m['grid'][3,jnp.asarray([1,3])]
    amounts=jnp.sum(pots[:,2:4],axis=0)
    committed=jnp.max(amounts)>=2
    firsttype=jnp.argmax(amounts)
    learn=(m['lower_locked']==0)&committed
    m['lower_flip']=jnp.where(learn,(firsttype!=recipe).astype(jnp.int32),m['lower_flip'])
    m['lower_locked']=jnp.where(learn,1,m['lower_locked'])
    signalrecipe=jnp.bitwise_xor(recipe,m['lower_flip'])
    signalc = jnp.where(signalrecipe == 0, 1, 3)
    action = go(r,c,d,4,signalc,0,False)
    purpose = jnp.int32(0)
    takeplate = jnp.any(plates) & usable & (jnp.sum(pot[2:6]) == 3)
    action = jnp.where(takeplate, go(r,c,d,4,platec,0), action)
    purpose = jnp.where(takeplate, 1, purpose)
    takeraw = jnp.any(rawok) & usable & (jnp.sum(pot[2:6]) < 3)
    action = jnp.where(takeraw, go(r,c,d,4,rawc,0), action)
    purpose = jnp.where(takeraw, 2, purpose)
    raw = (inv[0] == 0) & (jnp.sum(inv[2:]) > 0)
    properraw = raw & (inv[recipe+2] > 0) & usable & (jnp.sum(pot[2:6]) < 3)
    action = jnp.where(properraw, go(r,c,d,4,potc,0), action)
    purpose = jnp.where(properraw, 3, purpose)
    unwanted = raw & ~properraw
    # Return overflow raw ingredients to shared counters, otherwise store below.
    storage=m['grid'][jnp.asarray([5,5,6,6,6]),jnp.asarray([0,5,1,3,4])]
    free=jnp.sum(storage[:,:6],axis=1)==0
    si=jnp.argmax(free)
    sr=jnp.asarray([5,5,5,5,5])[si]
    sc=jnp.asarray([1,4,1,3,4])[si]
    sd=jnp.asarray([3,2,1,1,1])[si]
    storeaction=go(r,c,d,sr,sc,sd)
    rawdrop=jnp.where(jnp.any(empty),go(r,c,d,4,dropc,0),storeaction)
    action=jnp.where(unwanted,rawdrop,action)
    plate = (inv[0] == 1) & (inv[1] == 0)
    action = jnp.where(plate & usable, go(r,c,d,4,potc,0), action)
    purpose = jnp.where(plate, 4, purpose)
    dish = inv[1] == 1
    correctdish=inv[recipe+2]==3
    action = jnp.where(dish & correctdish & jnp.any(empty), go(r,c,d,4,dropc,0), action)
    action = jnp.where(dish & ~correctdish & jnp.any(free),storeaction,action)
    purpose = jnp.where(dish, 5, purpose)
    return action, m, {'pos': jnp.array([r,c]), 'recipe': m['recipe'], 'purpose': purpose, 'flip': m['lower_flip']}

def agent_0_act(obs, reset, memory, rng):
    oldgrid=jnp.where(reset,jnp.zeros((7,6,7),jnp.int32),memory['grid'])
    m, r, c, d, inv = decode(obs, memory, reset, 0)
    partner = obs[:, :, 11]
    seen = jnp.sum(partner) > 0
    pc = c + jnp.argmax(jnp.sum(partner, axis=0)) - 2
    partnerinv = jnp.sum(obs[:,:,16:22],axis=(0,1))
    pd = jnp.argmax(jnp.sum(obs[:,:,12:16],axis=(0,1)))
    grid=m['grid']
    codes=jnp.asarray([pc>2, pc>1, pc>3, (pd==1)|(pd==2), pd>=2, seen,
                       grid[3,2,5]>0, grid[3,4,5]>0, partnerinv[5]>0,
                       grid[3,1,5]>0, grid[3,3,5]>0],jnp.int32)
    viewing=(r==2)&(c==2)
    unchanged=jnp.all(codes==m['codes'])
    m['stable']=jnp.where(viewing,jnp.where(unchanged,m['stable']+1,1),0)
    m['codes']=jnp.where(viewing,codes,m['codes'])
    grounded = (partnerinv[2]+partnerinv[3]) > 0
    unknown = m['recipe'] < 0
    m['wait'] = jnp.where(unknown, m['wait']+1, 0)
    minwait=jnp.where((m['active']>=6)|(~seen & (m['step']<15)),8,4)
    accept = unknown & viewing & (m['stable'] >= 2) & (m['wait'] >= minwait)
    normal_correct = m['last'] == m['cycle_codes']
    changes = codes != m['cycle_codes']
    unchanged_update = m['odds'] + jnp.where(normal_correct, -1, 1)
    changed_update = jnp.where(normal_correct, 8, -8)
    update = jnp.where(changes, changed_update, unchanged_update)
    evidence=accept & (m['pending']>0)
    m['odds']=jnp.where(evidence,jnp.clip(update,-12,12),m['odds'])
    m['quality']=jnp.where(evidence,jnp.minimum(12,m['quality']+changes*4),m['quality'])
    label_valid=jnp.all(m['label_seen'])&(m['labels'][0]!=m['labels'][1])&~m['label_bad']&m['prospective']
    labeled=jnp.any(label_valid)
    priorities=m['quality']+jnp.asarray([2,0,0,0,0,0,0,0,0,0,0],jnp.int32)
    chosen=jnp.where(labeled,jnp.argmax(jnp.where(label_valid,priorities,-100)),jnp.argmax(priorities))
    m['active']=jnp.where(accept,chosen,m['active'])
    sig=codes[m['active']]
    m['signal']=jnp.where(viewing,sig,m['signal'])
    inferred=jnp.where(m['odds'][m['active']]>=0,sig,1-sig)
    inferred=jnp.where(labeled,(sig==m['labels'][1,m['active']]).astype(jnp.int32),inferred)
    m['recipe']=jnp.where(accept,inferred,m['recipe'])
    m['recipe']=jnp.where(grounded,jnp.argmax(partnerinv[2:4]),m['recipe'])
    m['cycle_codes']=jnp.where(accept,codes,m['cycle_codes'])
    m['pending']=jnp.where(accept,0,m['pending'])
    m['recipe']=jnp.where(unknown & (m['wait']>18),1-m['last'],m['recipe'])
    rec = jnp.maximum(0, m['recipe'])
    potc, pot, usable = choose_pot(m['grid'], rec)
    otherc, other, otherusable = choose_pot(m['grid'], 1-rec)
    count = jnp.sum(pot[2:6])
    othercount = jnp.sum(other[2:6])
    hasraw = (inv[0] == 0) & (jnp.sum(inv[2:]) > 0)
    heldtype = jnp.argmax(inv[2:])
    plate = (inv[0] > 0) & (inv[1] == 0)
    dish = inv[1] > 0
    m['cycle_codes']=jnp.where(dish & viewing,codes,m['cycle_codes'])
    # Spend cooking time stocking the other recipe, then collect a plate.
    spare = (count == 3) & (pot[6] > 0) & (m['step']<365) & otherusable & (othercount < 3) & (otherc != potc)
    worktype = jnp.where(spare, 1-rec, rec)
    worktype = jnp.where(hasraw, heldtype, worktype)
    workc, workpot, workok = choose_pot(m['grid'], worktype)
    supplyc = jnp.where(workc <= 2, 1, 4)
    supplydir = jnp.where(supplyc == 1, 3, 2)
    fetch = go(r,c,d,worktype+1,supplyc,supplydir)
    fill = go(r,c,d,2,workc,1)
    getplate = go(r,c,d,1,3,0)
    deliver = go(r,c,d,1,2,0)
    observe = go(r,c,d,2,2,1,False)
    earlyoffer=(m['offer_done']==0)&(count>=3)
    action = jnp.where(((count >= 3) & ~spare)|earlyoffer, getplate, fetch)
    action = jnp.where(hasraw, fill, action)
    action = jnp.where(plate, go(r,c,d,2,potc,1), action)
    counters = m['grid'][3,jnp.asarray([2,4])]
    counterdish = counters[:,1] > 0
    takec = jnp.asarray([2,4])[jnp.argmax(counterdish)]
    returned=counters[jnp.argmax(counterdish)]
    returnedtype=jnp.argmax(returned[2:4])
    label_now=(jnp.sum(counterdish)==1)&viewing&(m['stable']>=2)&seen&(jnp.sum(partnerinv)==0)&(jnp.max(returned[2:4])==3)
    predicted_change=m['early_valid']&(returnedtype!=m['early_last'])
    informative=(m['early_codes']!=m['last_delivery_codes'])&(m['early_codes']==codes)
    m['prospective']=m['prospective']|(label_now&predicted_change&informative)
    conflicts=m['label_seen'][returnedtype]&(codes!=m['labels'][returnedtype])
    m['label_bad']=m['label_bad']|(label_now&conflicts)
    m['labels']=m['labels'].at[returnedtype].set(jnp.where(label_now,codes,m['labels'][returnedtype]))
    m['label_seen']=m['label_seen'].at[returnedtype].set(m['label_seen'][returnedtype]|label_now)
    # Mode changes happen only after the current handoff has fully completed.
    can_switch=labeled&(m['recipe']<0)&(jnp.sum(inv)==0)&(jnp.sum(counters[:,:2])==0)&(m['partner_busy']==0)
    m['direct']=jnp.where(can_switch,1,m['direct'])
    m['direct']=jnp.where(~labeled,0,m['direct'])
    action = jnp.where((jnp.sum(inv)==0) & jnp.any(counterdish), go(r,c,d,2,takec,1), action)
    action = jnp.where(dish, deliver, action)
    # Offer a single plate, retaining a bounded fallback for a partner who ignores it.
    csum=jnp.sum(counters[:,:6],axis=1)
    emptycounter=csum==0
    counterplates=(counters[:,0]>0)&(counters[:,1]==0)
    emptyhands=jnp.sum(inv)==0
    offerempty=m['grid'][3,m['offer_col'],0]==0
    taken=(m['handoff']>0)&emptyhands&(oldgrid[3,m['offer_col'],0]>0)&offerempty
    received=jnp.any(counterdish)
    m['partner_busy']=jnp.where((partnerinv[0]>0)|taken,1,m['partner_busy'])
    m['partner_busy']=jnp.where(received,0,m['partner_busy'])
    cooperate=(partnerinv[0]>0)|taken|received
    m['physical']=jnp.where((cooperate & (m['physical']!=2))|received,1,m['physical'])
    m['physical_age']=jnp.where((m['physical']==1)&~received,m['physical_age']+1,0)
    m['physical']=jnp.where(m['physical_age']>75,2,m['physical'])
    discover=plate&(m['offer_done']==0)&jnp.any(emptycounter)&(m['physical']!=1)
    m['offer_col']=jnp.where(discover,jnp.where(emptycounter[0],2,4),m['offer_col'])
    m['handoff']=jnp.where(discover,1,m['handoff'])
    m['offer_done']=jnp.where(plate,1,m['offer_done'])
    m['offer_age']=jnp.where((m['handoff']>0)&~plate,m['offer_age']+1,0)
    m['handoff']=jnp.where((m['physical']==1)|received,0,m['handoff'])
    offeraction=go(r,c,d,2,m['offer_col'],1)
    reclaim=emptyhands&(m['offer_age']>=22)&~offerempty
    waitoffer=go(r,c,d,2,2,1,False)
    sparetype=1-rec
    sparec,sparepot,spareok=choose_pot(m['grid'],sparetype)
    sparepossible=spareok&(jnp.sum(sparepot[2:6])<3)&(sparec!=potc)
    side=jnp.where(sparec==1,1,4)
    sideface=jnp.where(side==1,3,2)
    productive=go(r,c,d,sparetype+1,side,sideface)
    duringoffer=jnp.where(emptyhands&sparepossible,productive,waitoffer)
    duringoffer=jnp.where(hasraw,fill,duringoffer)
    duringoffer=jnp.where(plate|reclaim,offeraction,duringoffer)
    action=jnp.where(m['handoff']>0,duringoffer,action)
    reclaimed=reclaim&(r==2)&(c==m['offer_col'])&(d==1)&(action==5)
    m['handoff']=jnp.where(reclaimed|((m['offer_age']>16)&offerempty),0,m['handoff'])

    before_offer=plate&viewing&(m['physical']==1)&~received&~jnp.any(counterplates)&(m['last_delivery_codes'][0]>=0)
    m['early_codes']=jnp.where(before_offer,codes,m['early_codes'])
    m['early_last']=jnp.where(before_offer,m['last'],m['early_last'])
    m['early_valid']=m['early_valid']|before_offer

    # Once the partner actually uses plates, supply both recipes and let its
    # physically returned dish carry the recipe information.
    p0c,p0,u0=choose_pot(m['grid'],jnp.int32(0))
    p1c,p1,u1=choose_pot(m['grid'],jnp.int32(1))
    n0=jnp.sum(p0[2:6]);n1=jnp.sum(p1[2:6])
    can0=u0&(n0<3);can1=u1&(n1<3)
    make0=can0&(~can1|(n0>=n1))
    make=jnp.where(make0,0,1)
    make=jnp.where(hasraw,heldtype,make)
    makec,makepot,makeok=choose_pot(m['grid'],make)
    sc=jnp.where(makec==1,1,4)
    sd=jnp.where(sc==1,3,2)
    physical_action=go(r,c,d,make+1,sc,sd)
    physical_action=jnp.where(~(can0|can1),observe,physical_action)
    needplate=~jnp.any(counterplates)&(m['partner_busy']==0)&~jnp.any(counterdish)&((n0==3)|(n1==3))
    physical_action=jnp.where(needplate,getplate,physical_action)
    physical_action=jnp.where(hasraw,go(r,c,d,2,makec,1),physical_action)
    dropcol=jnp.where(emptycounter[0],2,4)
    physical_action=jnp.where(plate,go(r,c,d,2,dropcol,1),physical_action)
    physical_action=jnp.where(emptyhands&jnp.any(counterdish)&(~(can0|can1)|(m['step']>=365)),go(r,c,d,2,takec,1),physical_action)
    physical_action=jnp.where(dish,deliver,physical_action)
    servicing=(m['physical']==1)&(m['direct']==0)
    action=jnp.where(servicing,physical_action,action)
    action = jnp.where((m['recipe'] < 0)&~servicing&(m['handoff']==0), observe, action)
    delivered = dish & (r == 1) & (c == 2) & (d == 0) & (action == 5)
    m['last_delivery_codes']=jnp.where(delivered,m['cycle_codes'],m['last_delivery_codes'])
    m['early_valid']=m['early_valid']&~delivered
    m['last'] = jnp.where(delivered, jnp.argmax(inv[2:]), m['last'])
    m['recipe'] = jnp.where(delivered, -1, m['recipe'])
    m['stable'] = jnp.where(delivered, 0, m['stable'])
    m['pending'] = jnp.where(delivered, 1, m['pending'])
    return action.astype(jnp.int32), m, {'pos': jnp.array([r,c]), 'recipe': m['recipe'], 'signal': m['signal'], 'pot': pot, 'spare': spare, 'odds': m['odds'], 'active': m['active'], 'physical': m['physical'], 'direct': m['direct'], 'labels': m['label_seen'], 'prospective': m['prospective'], 'offer': m['handoff'], 'inventory': inv, 'delivered': delivered}
