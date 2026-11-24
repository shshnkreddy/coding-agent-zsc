"""Two-player H-group Level 1: persistent, observation-local card notes."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_level1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        'notes': jnp.ones((2, 5, 25), dtype=jnp.bool_),
        'touched': jnp.zeros((2, 5), dtype=jnp.bool_),
        'weak': jnp.zeros((2, 5), dtype=jnp.bool_),
        'links': jnp.full((2, 5, 25, 5), -1, dtype=jnp.int32),
        'chains': jnp.zeros((2,5,25,5),dtype=jnp.bool_),
        'pending': jnp.zeros((2,5,25),dtype=jnp.bool_),
        'early': jnp.asarray(True),
        'turn': jnp.asarray(0, jnp.int32),
        'empty_age': jnp.asarray(0, jnp.int32),
    }


def decode(f):
    hand = f[:125].reshape(5,25)
    heights = jnp.sum(f[167:192].reshape(5,5) * jnp.arange(1,6)[None,:],axis=1).astype(jnp.int32)
    d = f[203:253].reshape(5,10)
    disc = jnp.stack([jnp.sum(d[:,:3],-1),jnp.sum(d[:,3:5],-1),jnp.sum(d[:,5:7],-1),jnp.sum(d[:,7:9],-1),d[:,9]],-1)
    k = f[308:658].reshape(2,5,35)
    literal = k[:,:,:25] > 0
    touched = jnp.any(k[:,:,25:] > 0,axis=-1)
    return hand,heights,disc,literal,touched


def board_masks(heights, disc):
    rank = jnp.arange(5)[None,:]
    remain = jnp.array([3,2,2,2,1])[None,:]-disc
    blocked = jnp.cumsum((remain<=0).astype(jnp.int32),axis=1)>0
    useful = (rank>=heights[:,None]) & ~blocked
    playable = rank==heights[:,None]
    critical = (remain==1)&useful
    return useful.reshape(25),playable.reshape(25),critical.reshape(25)


def consistent(notes,literal):
    a = notes & literal
    return jnp.where(jnp.any(a,axis=-1,keepdims=True),a,literal)


def refine(notes,literal,touched,not_played):
    notes = consistent(notes,literal)
    # Good Touch narrows ambiguous touched cards; a wholly obsolete note remains trash.
    # Losing a prerequisite can turn a formerly good touch into trash; that
    # does not justify deleting its identity from an old ambiguous note.
    live = notes & not_played
    notes = jnp.where(touched[...,None]&jnp.any(live,axis=-1,keepdims=True),live,notes)
    def step(_,n):
        flat=n.reshape(10,25)
        tags=touched.reshape(10)
        single=(jnp.sum(flat,axis=-1)==1)&tags
        known=flat&single[:,None]
        others=jnp.sum(known.astype(jnp.int32),axis=0)[None,:]-known.astype(jnp.int32)
        a=flat&~(others>0)
        # Do not resolve two explicitly duplicated cards by an undisclosed order.
        change=tags&~single&jnp.any(a,axis=-1)
        return jnp.where(change[:,None],a,flat).reshape(2,5,25)
    return jax.lax.fori_loop(0,3,step,notes)


def chop(touched,occupied):
    return jnp.min(jnp.where(~touched&occupied,jnp.arange(5),5))


def clue_focus(touch,old_touched,occupied):
    cp=chop(old_touched,occupied)
    new=touch&~old_touched
    left=jnp.max(jnp.where(jnp.where(jnp.any(new),new,touch),jnp.arange(5),0))
    focus=jnp.where((cp<5)&touch[jnp.minimum(cp,4)]&jnp.any(new),cp,left)
    return focus,cp,new


def interpret(n,other,lit,old_touched,other_touched,touch,kind,value,heights,disc,forced=False):
    useful,playable,critical=board_masks(heights,disc)
    occupied=jnp.any(lit,-1)
    focus,cp,new=clue_focus(touch,old_touched,occupied)
    n=consistent(n,lit)
    original=n
    n=jnp.where(new[:,None],n&useful,n)
    n=consistent(n,lit)
    # Connections must be through previously clued cards. Potential self-prompts
    # are retained as alternatives, not permissions to play an unclued card.
    own_connect=n&old_touched[:,None]&(jnp.arange(5)!=focus)[:,None]
    other_connect=other&other_touched[:,None]
    connectors=jnp.any(own_connect,axis=0)|jnp.any(other_connect,axis=0)
    connectors=connectors.reshape(5,5)
    satisfied=connectors|(jnp.arange(5)[None,:]<heights[:,None])
    lower_ok=jnp.concatenate([jnp.ones((5,1),dtype=jnp.bool_),jnp.cumprod(satisfied[:,:4],axis=1).astype(jnp.bool_)],axis=1)
    play_options=lower_ok.reshape(25)&useful
    ranks=jnp.tile(jnp.arange(5),5)
    save_options=((critical&(ranks!=4))|((kind==1)&((value==1)|(value==4))))&useful
    save_options=save_options&(focus==cp)&jnp.any(new)
    options=(play_options|save_options)&n[focus]
    # Repeated clues at full tokens may carry no additional convention inference.
    options=jnp.where(jnp.any(options),options,n[focus])
    n=n.at[focus].set(options)
    save_possible=jnp.any(save_options&options)
    other_types=jnp.any(other&other_touched[:,None],axis=0)
    possible_other_dupe=jnp.any(original&other_types,axis=-1)
    overlap=jnp.any(original[:,None,:]&original[None,:,:],axis=-1)&~jnp.eye(5,dtype=jnp.bool_)
    possible_local_dupe=jnp.any(overlap&(old_touched|touch)[None,:],axis=-1)
    possible_trash=jnp.any(original&~useful,axis=-1)
    exception_risk=possible_other_dupe|(save_possible&possible_local_dupe)|((jnp.arange(5)!=focus)&possible_trash)
    weak_new=new&(focus==cp)&exception_risk
    old_new_overlap=overlap&((old_touched[:,None]&new[None,:])|(new[:,None]&old_touched[None,:]))
    weak_new=weak_new|(forced&((new&possible_other_dupe)|jnp.any(old_new_overlap,axis=-1)))
    n=jnp.where((weak_new&(jnp.arange(5)!=focus))[:,None],original,n)
    return n,focus,save_options,play_options,weak_new


def make_links(n, other, old_touched, other_touched, focus, saves, h):
    """Conditional links only to a unique already-clued connecting card.

    links[focus_identity, connector_slot] is its required identity, or -1.
    Save alternatives impose no play dependencies.
    """
    colors=jnp.arange(25)//5
    ranks=jnp.arange(25)%5
    req=colors[:,None]*5+jnp.arange(5)[None,:]
    needed=(jnp.arange(5)[None,:]>=h[colors,None])&(jnp.arange(5)[None,:]<ranks[:,None])&~saves[:,None]
    other_types=jnp.any(other&other_touched[:,None],axis=0)
    candidates=jnp.transpose(n[:,req],(1,2,0))&old_touched[None,None,:]&(jnp.arange(5)!=focus)[None,None,:]
    unique=jnp.sum(candidates,axis=-1)==1
    own_needed=needed&~other_types[req]
    ranksets=((jnp.arange(1,32)[:,None]>>jnp.arange(5)[None,:])&1)>0
    subsets=own_needed[:,None,:]&ranksets[None,:,:]
    available=jnp.any(candidates[:,None,:,:]&subsets[:,:,:,None],axis=2)
    complete=jnp.all(jnp.sum(available,axis=-1)>=jnp.sum(subsets,axis=-1),axis=-1)
    required=own_needed&unique&complete[:,None]
    assignments=candidates&required[:,:,None]
    links=jnp.max(jnp.where(assignments,req[:,:,None],-1),axis=1)
    # A single card cannot be two different links in the same chain.
    impossible=~complete|jnp.any(jnp.sum(assignments,axis=1)>1,axis=-1)
    return links,impossible


def apply_links(n, links, playable):
    def step(_, n):
        safe_dep=jnp.maximum(links,0)
        supported=jnp.all((links<0)|n[jnp.arange(5)[None,None,:],safe_dep],axis=-1)
        narrowed=n&supported
        n=jnp.where(jnp.any(narrowed,axis=-1,keepdims=True),narrowed,n)
        # A self-prompt is actionable when every retained focus interpretation
        # needs the same clued slot to supply a currently playable connector.
        ready=(links>=0)&playable[safe_dep]
        slot=jnp.max(jnp.where(ready,jnp.arange(5)[None,None,:],-1),axis=-1)
        first=jnp.min(jnp.where(n,slot,5),axis=-1)
        last=jnp.max(jnp.where(n,slot,-1),axis=-1)
        forced=(first==last)&(first>=0)&jnp.any(n,axis=-1)
        chosen=jnp.clip(first,0,4)
        expected=jnp.take_along_axis(links,chosen[:,None,None],axis=2)[:,:,0]
        allowed=jnp.any(jax.nn.one_hot(jnp.maximum(expected,0),25,dtype=jnp.bool_)&n[:,:,None]&(expected>=0)[:,:,None],axis=1)
        constrain=forced[:,None]&(jnp.arange(5)[None,:]==chosen[:,None])
        all_allowed=jnp.all(~constrain[:,:,None]|allowed[:,None,:],axis=0)
        narrowed=n&all_allowed
        return jnp.where(jnp.any(narrowed,axis=-1,keepdims=True),narrowed,n)
    return jax.lax.fori_loop(0,3,step,n)


def make_chains(n,other,old_touched,other_touched,focus,saves,h):
    ids=jnp.arange(25)
    other_types=jnp.any(other&other_touched[:,None],axis=0)
    needed=(ids[:,None]//5==ids[None,:]//5)&(ids[None,:]%5>=h[ids[:,None]//5])&(ids[None,:]%5<ids[:,None]%5)
    eligible=jnp.any(needed[:,None,:]&n[None,:,:],axis=-1)&old_touched[None,:]&(jnp.arange(5)!=focus)[None,:]
    # Hall's matching criterion: every subset of required ranks must have at
    # least that many distinct old cards available to supply it.
    ranksets=((jnp.arange(1,32)[:,None]>>jnp.arange(5)[None,:])&1)>0
    requirements=needed[:,None,:]&ranksets[None,:,jnp.arange(25)%5]
    candidates=jnp.any(requirements[:,:,None,:]&n[None,None,:,:],axis=-1)&old_touched[None,None,:]&(jnp.arange(5)!=focus)[None,None,:]
    complete=jnp.all(jnp.sum(candidates,axis=-1)>=jnp.sum(requirements,axis=-1),axis=-1)
    # Mixed-hand connections stay conservative; only an entirely own-hand
    # missing sequence imposes a prompt obligation on these original slots.
    external=jnp.any(needed&other_types[None,:],axis=-1)
    pending=(ids%5>h[ids//5])&~saves&~external&complete&jnp.any(eligible,axis=-1)
    return eligible&pending[:,None],pending


def apply_chains(n,chains,pending,playable):
    colors=jnp.arange(25)//5
    next_rank=jnp.argmax(playable.reshape(5,5),axis=-1)
    target=colors*5+next_rank[colors]
    def step(_,n):
        eligible=chains&n[:,target].T[None,:,:]
        supported=~pending|jnp.any(eligible,axis=-1)
        nn=n&supported
        n=jnp.where(jnp.any(nn,axis=-1,keepdims=True),nn,n)
        universal=jnp.all(~n|(pending&playable[target][None,:]&jnp.any(eligible,axis=-1)),axis=-1)&jnp.any(n,axis=-1)
        slots=jnp.any(eligible&n[:,:,None],axis=1)
        leftmost=jnp.max(jnp.where(slots,jnp.arange(5)[None,:],-1),axis=-1)
        force=jnp.any(universal[:,None]&(jnp.arange(5)[None,:]==leftmost[:,None]),axis=0)
        nn=n&jnp.where(force[:,None],playable[None,:],True)
        return jnp.where(jnp.any(nn,axis=-1,keepdims=True),nn,n)
    return jax.lax.fori_loop(0,2,step,n)


def count_elimination(n,counts):
    """Hall's capacity rule, using only the observer's unaccounted card pool."""
    subsets=((jnp.arange(1,32)[:,None]>>jnp.arange(5)[None,:])&1)>0
    occupied=jnp.any(n,axis=-1)
    valid=jnp.all(~subsets|occupied[None,:],axis=-1)
    sizes=jnp.sum(subsets,axis=-1)
    def step(_,n):
        union=jnp.any(n[None,:,:]&subsets[:,:,None],axis=1)
        capacity=jnp.sum(union*counts[None,:],axis=-1)
        saturated=valid&(capacity==sizes)
        remove=jnp.any(saturated[:,None,None]&~subsets[:,:,None]&union[:,None,:],axis=0)
        narrowed=n&~remove
        return jnp.where(jnp.any(narrowed,axis=-1,keepdims=True),narrowed,n)
    return jax.lax.fori_loop(0,3,step,n)


def own_belief(notes,literal,touched,hand,heights,disc):
    useful,playable,critical=board_masks(heights,disc)
    remaining=(jnp.array([3,2,2,2,1])[None,:]-disc-(jnp.arange(5)[None,:]<heights[:,None])).reshape(25)
    counts=jnp.maximum(remaining-jnp.sum(hand,axis=0),0)
    n=consistent(notes[0],literal[0]&(counts>0)[None,:])
    # Visible already-clued duplicates are excluded where Good Touch leaves a choice.
    known_partner=jnp.any((hand>0)&touched[1,:,None],axis=0)
    a=n&~known_partner
    n=jnp.where(touched[0,:,None]&jnp.any(a,axis=-1,keepdims=True),a,n)
    not_played=(jnp.arange(5)[None,:]>=heights[:,None]).reshape(25)
    live=n&not_played
    n=jnp.where(touched[0,:,None]&jnp.any(live,axis=-1,keepdims=True),live,n)
    n=count_elimination(n,counts)
    p=n*counts[None,:]
    p=p/jnp.maximum(jnp.sum(p,axis=-1,keepdims=True),1)
    pp=jnp.sum(p*playable,axis=-1)
    trash=jnp.sum(p*~useful,axis=-1)
    return n,p,pp,trash


def agent_act(obs,reset,memory,rng):
    memory=jax.tree.map(lambda x,z:jnp.where(reset,z,x),memory,init_agent())
    f=obs.features
    hand,h,d,lit,touched=decode(f)
    useful,playable,critical=board_masks(h,d)
    not_played=(jnp.arange(5)[None,:]>=h[:,None]).reshape(25)
    last=f[253:308]
    actor=jnp.argmax(last[:2])
    typ=jnp.argmax(last[2:6])
    has_last=jnp.any(last[:2]>0)&~reset
    removal=has_last&(typ<2)
    slot=jnp.argmax(last[23:28])
    idx=jnp.minimum(jnp.arange(5)+(jnp.arange(5)>=slot),4)
    links=memory['links']
    chains=memory['chains']
    pending=memory['pending']
    revealed=jnp.argmax(last[28:53])
    expected=links[actor,:,:,slot]
    contradicted=(expected>=0)&(expected!=revealed)&removal
    fulfilled=(revealed==(jnp.arange(25)-1))[None,:]&removal
    still_pending=pending[actor]&~fulfilled
    moved_chains=chains[actor][:,:,idx].at[:,:,4].set(False)
    chain_failed=still_pending&~jnp.any(moved_chains,axis=-1)&removal
    prior_notes=memory['notes'].at[actor].set(memory['notes'][actor]&~contradicted&~chain_failed)
    moved_chains=moved_chains[idx].at[4].set(False)
    chains=chains.at[actor].set(jnp.where(removal,moved_chains,chains[actor]))
    moved_pending=still_pending[idx].at[4].set(False)
    pending=pending.at[actor].set(jnp.where(removal,moved_pending,pending[actor]))
    pending=pending&((jnp.arange(25)%5-1)>=h[jnp.arange(25)//5])[None,None,:]
    moved_links=links[actor][idx][:,:,idx].at[4].set(-1).at[:,:,4].set(-1)
    links=links.at[actor].set(jnp.where(removal,moved_links,links[actor]))
    shifted=prior_notes[actor][idx].at[4].set(jnp.ones(25,dtype=jnp.bool_))
    notes=prior_notes.at[actor].set(jnp.where(removal,shifted,prior_notes[actor]))
    shifted_t=memory['touched'][actor][idx].at[4].set(False)
    old_t=memory['touched'].at[actor].set(jnp.where(removal,shifted_t,memory['touched'][actor]))
    shifted_w=memory['weak'][actor][idx].at[4].set(False)
    weak=memory['weak'].at[actor].set(jnp.where(removal,shifted_w,memory['weak'][actor]))
    old_cp=chop(memory['touched'][actor],jnp.ones(5,dtype=jnp.bool_))
    early=memory['early']&~(removal&(typ==1)&(slot==old_cp))
    notes=consistent(notes,lit)
    failed=has_last&(typ==0)&(last[53]==0)
    affected=failed&touched&lit[:,:,revealed]&~notes[:,:,revealed]
    notes=jnp.where(affected[:,:,None],lit,notes)
    weak=weak|affected
    links=jnp.where(failed,-1,links)
    pending=pending&~failed
    recipient=1-actor
    hint=has_last&(typ>=2)
    value=jnp.where(typ==2,jnp.argmax(last[8:13]),jnp.argmax(last[13:18]))
    # Use the visible other hand for interpreting a clue we receive. The
    # public approximation of our hand is used when tracking partner notes.
    other=jnp.where(recipient==0,hand>0,notes[0])
    giver_forced=(jnp.sum(f[192:200])>=7)|(chop(old_t[actor],jnp.any(lit[actor],axis=-1))==5)
    received,focus,received_saves,_,weak_new=interpret(notes[recipient],other,lit[recipient],old_t[recipient],old_t[actor],last[18:23]>0,typ-2,value,h,d,giver_forced)
    new_chains,new_pending=make_chains(received,other,old_t[recipient],old_t[actor],focus,received_saves,h)
    chains=chains.at[recipient,focus].set(jnp.where(hint,new_chains,chains[recipient,focus]))
    pending=pending.at[recipient,focus].set(jnp.where(hint,new_pending,pending[recipient,focus]))
    new_links,impossible=make_links(received,other,old_t[recipient],old_t[actor],focus,received_saves,h)
    narrowed=received[focus]&~impossible
    received=received.at[focus].set(jnp.where(jnp.any(narrowed),narrowed,received[focus]))
    links=links.at[recipient,focus].set(jnp.where(hint,new_links,links[recipient,focus]))
    notes=notes.at[recipient].set(jnp.where(hint,received,notes[recipient]))
    notes=jax.vmap(apply_links,in_axes=(0,0,None))(notes,links,playable)
    notes=jax.vmap(apply_chains,in_axes=(0,0,0,None))(notes,chains,pending,playable)
    new_weak=weak[recipient]|weak_new
    weak=weak.at[recipient].set(jnp.where(hint,new_weak,weak[recipient]))
    # A later required save can duplicate an older card: neither copy then
    # licenses Good Touch elimination of the older card's true identity.
    weak_partner_types=jnp.any((hand>0)&weak[1,:,None]&touched[1,:,None],axis=0)
    weak=weak.at[0].set(weak[0]|(touched[0]&jnp.any(notes[0]&weak_partner_types,axis=-1)))
    strong=touched&~weak
    notes=refine(notes,lit,strong,not_played)
    notes=jax.vmap(apply_links,in_axes=(0,0,None))(notes,links,playable)
    notes=jax.vmap(apply_chains,in_axes=(0,0,0,None))(notes,chains,pending,playable)
    n,p,pp,trash=own_belief(notes,lit,strong,hand,h,d)
    # Private visibility refinements are safe to retain for our own notes.
    notes=notes.at[0].set(n)
    occupied=jnp.any(lit,axis=-1)
    cp=chop(touched[0],occupied[0])
    partner_cp=chop(touched[1],occupied[1])
    info=jnp.sum(f[192:200])
    deck=jnp.sum(f[127:167])
    lives=jnp.sum(f[200:203])
    empty_age=jnp.where(deck>0,0,memory['empty_age']+1).astype(jnp.int32)
    ids=jnp.argmax(hand,axis=1)
    partner_play=playable[ids]&occupied[1]
    partner_useful=useful[ids]&occupied[1]
    partner_critical=critical[ids]&occupied[1]
    pc=jnp.minimum(partner_cp,4)
    partner_known_play=jnp.all(~notes[1]|playable[None,:],axis=1)&occupied[1]&partner_play
    partner_known_trash=jnp.all(~notes[1]|~useful[None,:],axis=1)&occupied[1]
    duplicate=(ids[:,None]==ids[None,:])&~jnp.eye(5,dtype=jnp.bool_)&occupied[1,:,None]&occupied[1,None,:]
    unique=~jnp.any(duplicate,axis=1)
    chop_danger=(partner_cp<5)&partner_useful[pc]&(partner_critical[pc]|((ids[pc]%5==1)&unique[pc])|(partner_play[pc]&unique[pc]))
    urgent=chop_danger&~jnp.any(partner_known_play|partner_known_trash)
    own_certain=(pp>0.999)&occupied[0]
    forced_clue=(info>=8)|((cp==5)&~jnp.any((trash>0.999)&occupied[0]))
    firm_prior=refine(notes,lit,touched,not_played)[1]
    firm_prior_correct=firm_prior[jnp.arange(5),ids]
    firm_prior_unsafe=jnp.all(~firm_prior|playable[None,:],axis=-1)&occupied[1]&~partner_play

    def clue_value(a):
        kind=(a>=15).astype(jnp.int32)
        value=jnp.where(kind==0,a-10,a-15)
        attr=jnp.where(kind==0,jnp.arange(25)//5==value,jnp.arange(25)%5==value)
        touch=jnp.sum(hand*attr[None,:],axis=-1)>0
        new=touch&~touched[1]
        clue_lit=lit[1]&jnp.where(touch[:,None],attr[None,:],~attr[None,:])
        nn,fc,saves,plays,new_weak=interpret(notes[1],notes[0],clue_lit,touched[1],touched[0],touch,kind,value,h,d,forced_clue)
        candidate_links,impossible=make_links(nn,notes[0],touched[1],touched[0],fc,saves,h)
        candidate_links=links[1].at[fc].set(candidate_links)
        narrowed=nn[fc]&~impossible
        nn=nn.at[fc].set(jnp.where(jnp.any(narrowed),narrowed,nn[fc]))
        nn=apply_links(nn,candidate_links,playable)
        candidate_chain,candidate_pending=make_chains(nn,notes[0],touched[1],touched[0],fc,saves,h)
        candidate_chain=chains[1].at[fc].set(candidate_chain)
        candidate_pending=pending[1].at[fc].set(candidate_pending)
        nn=apply_chains(nn,candidate_chain,candidate_pending,playable)
        future_weak=weak[1]|new_weak
        future_notes=refine(jnp.stack([notes[0],nn]),jnp.stack([lit[0],clue_lit]),jnp.stack([strong[0],(touched[1]|touch)&~future_weak]),not_played)
        nn=future_notes[1]
        certainly=jnp.all(~nn|playable[None,:],axis=1)&occupied[1]
        unsafe=jnp.any(certainly&~partner_play)
        gain=jnp.sum(jnp.any((hand>0)&(certainly&~partner_known_play)[:,None],axis=0).astype(jnp.float32))
        new_count=jnp.sum(new.astype(jnp.float32))
        touched_after=touched[1]|touch
        dup_touch=jnp.any(duplicate&touched_after[:,None]&touched_after[None,:]&(new[:,None]|new[None,:]))
        bad=jnp.sum((new&~partner_useful).astype(jnp.float32))
        # Avoid duplicating an identity we already know is held in a clued slot.
        own_known=jnp.any(notes[0]&touched[0,:,None]&(jnp.sum(notes[0],axis=-1)==1)[:,None],axis=0)
        own_possible=jnp.any(notes[0]&touched[0,:,None],axis=0)
        bad=bad+jnp.sum((new&own_possible[ids]).astype(jnp.float32))+dup_touch.astype(jnp.float32)
        focus_save=saves[ids[fc]]
        focus_play=plays[ids[fc]]
        actual_connections=jnp.any((hand>0)&touched[1,:,None]&(jnp.arange(5)!=fc)[:,None],axis=0)|own_known
        sat=actual_connections.reshape(5,5)|(jnp.arange(5)[None,:]<h[:,None])
        actual_reachable=jnp.concatenate([jnp.ones((5,1),dtype=jnp.bool_),jnp.cumprod(sat[:,:4],axis=1).astype(jnp.bool_)],axis=1).reshape(25)
        focus_play=focus_play&actual_reachable[ids[fc]]
        correct_focus=(focus_save|focus_play)&partner_useful[fc]&nn[fc,ids[fc]]
        # Prompt question 3 permits a tempo clue to put old clued cards into
        # playable order when their normal leftmost prompt would misplay.
        connector=(ids[fc]//5)*5+jnp.minimum(h[ids[fc]//5],4)
        possible_before=touched[1]&notes[1,:,connector]
        leftmost_before=jnp.max(jnp.where(possible_before,jnp.arange(5),0))
        out_of_order=(jnp.sum(possible_before)>=2)&~partner_play[leftmost_before]
        reorders=out_of_order&~partner_play[fc]&jnp.any(certainly&~partner_known_play&(jnp.arange(5)!=fc))
        gets=jnp.any(new)|forced_clue|reorders
        saves_chop=touch[pc]&(partner_cp<5)&chop_danger
        gt_ok=(bad==0)|saves_chop
        false_note=jnp.sum((~nn[jnp.arange(5),ids]&occupied[1]).astype(jnp.float32))
        valid=gets&correct_focus&gt_ok&~unsafe&(false_note==0)
        immediate=jnp.any(certainly&partner_play)
        avoids_loss=saves_chop|immediate|jnp.any(partner_known_play|partner_known_trash)
        score=10.5*gain+1.5*new_count+0.15*(kind==0)-8*bad
        score=score+30*(urgent&avoids_loss)+2*saves_chop
        next_cp=chop(touched_after,occupied[1])
        nc=jnp.minimum(next_cp,4)
        next_loss=jnp.where(partner_critical[nc],5-ids[nc]%5,jnp.where((ids[nc]%5==1)&unique[nc],1.5,jnp.where(partner_play[nc]&unique[nc],1.,0.)))
        exposed=(next_cp<5)&partner_useful[nc]&~immediate&~jnp.any(partner_known_trash)
        score=score-4*exposed*next_loss
        score=score+0.015*jnp.sum(notes[1].astype(jnp.float32)-nn.astype(jnp.float32))
        # A partner may apply Good Touch more strongly to save exceptions.
        # Prefer the less misleading necessary save under that receiver too.
        firm_notes=refine(jnp.stack([notes[0],nn]),jnp.stack([lit[0],clue_lit]),jnp.stack([touched[0],touched_after]),not_played)[1]
        firm_false=jnp.sum((~firm_notes[jnp.arange(5),ids]&firm_prior_correct&occupied[1]).astype(jnp.float32))
        firm_play=jnp.all(~firm_notes|playable[None,:],axis=-1)&occupied[1]
        firm_unsafe=jnp.any(firm_play&~partner_play&~firm_prior_unsafe)
        score=score-6*firm_false-20*firm_unsafe
        # Save Principle is mandatory when a valid clue prevents an otherwise
        # imminent loss; efficiency penalties may choose between such clues.
        score=jnp.where(urgent&avoids_loss,jnp.maximum(score,10.)+0.001*score,score)
        certain_own=notes[0]&(jnp.sum(notes[0],axis=-1)==1)[:,None]
        strict_nn,_,_,_,_=interpret(notes[1],certain_own,clue_lit,touched[1],touched[0],touch,kind,value,h,d,forced_clue)
        strict_false=jnp.sum((~strict_nn[jnp.arange(5),ids]&occupied[1]).astype(jnp.float32))
        fallback=-80.-100*unsafe-100*false_note-100*strict_false+gain-10*bad-0.1*new_count+0.05*kind*value
        score=jnp.where(valid,score,fallback)
        final_clue_value=10.*(immediate.astype(jnp.float32)-jnp.any(partner_known_play).astype(jnp.float32))
        score=jnp.where(deck==0,jnp.where((empty_age==1)&valid&(gain>0),final_clue_value,-90.),score)
        return score,fc,gain,new_count,valid

    clue_scores,clue_foci,clue_gains,clue_new,valid_clues=jax.vmap(clue_value)(jnp.arange(10,20))
    clue_scores=jnp.where(obs.legal_actions[10:20],clue_scores,-1000.)
    best_clue=jnp.argmax(clue_scores)
    best_clue_score=clue_scores[best_clue]
    play_scores=10.+0.2*jnp.sum(p*jnp.tile(jnp.arange(5),5),axis=-1)-0.02*jnp.arange(5)
    # Save Principle takes precedence when the partner may otherwise discard.
    play_scores=play_scores-10*urgent
    # Playing a connector can itself protect partner's chop, with no hint.
    colors=jnp.arange(25)//5
    ranks=jnp.arange(25)%5
    next_playable=ranks[None,:]==(h[colors][None,:]+(colors[:,None]==colors[None,:]))
    ready_after=jnp.all(~notes[1][None,:,:]|next_playable[:,None,:],axis=-1)&occupied[1][None,:]
    actually_ready=next_playable[:,ids]&occupied[1][None,:]
    protects=jnp.any(ready_after&actually_ready,axis=-1)
    protect_prob=jnp.sum(p*protects[None,:],axis=-1)
    play_scores=play_scores+35*urgent*(protect_prob>0.999)+1.0*protect_prob
    own_ready_after=jnp.all(~notes[0][None,:,:]|next_playable[:,None,:],axis=-1)&occupied[0][None,:]&~own_certain[None,:]
    own_unlock=jnp.sum(own_ready_after,axis=-1)
    play_scores=play_scores+1.5*jnp.sum(p*own_unlock[None,:],axis=-1)
    play_scores=jnp.where(own_certain,play_scores,-1e5)
    # A spare life can resolve an otherwise stalled, highly likely play.
    # This is a local risk choice and conveys no positional or hidden signal.
    mid_risk=(deck>0)&(lives==3)&(pp>0.85)&~own_certain
    play_scores=jnp.where(mid_risk,5.*pp,play_scores)
    # When the deck is gone, spending a life with a high chance of a point is rational.
    risk_ok=(deck==0)&(lives>1)&(pp>0)
    partner_ready=jnp.any(partner_known_play)
    ready_after_attempt=jnp.where(playable,protects,partner_ready)
    partner_delta=jnp.sum(p*ready_after_attempt[None,:],axis=-1)-partner_ready.astype(jnp.float32)
    final_play_value=10.*(pp+(empty_age==1)*partner_delta)
    play_scores=jnp.where((deck==0)&(own_certain|risk_ok),final_play_value+0.001*pp,play_scores)
    disc_scores=jnp.where(jnp.arange(5)==cp,0.,-2000.)
    disc_scores=jnp.where(trash>0.999,1.5-0.01*jnp.arange(5),disc_scores)
    # The early game must exhaust available play/save clues before chop discards.
    disc_scores=jnp.where(early&jnp.any(valid_clues&obs.legal_actions[10:20])&(jnp.arange(5)==cp),-50.,disc_scores)
    # Emergency fallback for a fully clued blocked hand at zero clues.
    loss=jnp.sum(p*useful*(1+3*critical),axis=-1)
    disc_scores=jnp.where((cp==5)&(info==0),-2-loss,disc_scores)
    disc_scores=jnp.where(deck==0,0.,disc_scores)
    scores=jnp.concatenate([disc_scores,play_scores,clue_scores,jnp.array([-1000.])])
    scores=jnp.where(obs.legal_actions,scores,-1e6)
    action=jnp.argmax(scores).astype(jnp.int32)
    memory={'notes':notes,'touched':touched,'weak':weak,'links':links,'chains':chains,'pending':pending,'early':early,'turn':memory['turn']+1,'empty_age':empty_age}
    trace={'pp':pp,'trash':trash,'notes':notes,'chop':cp,'partner_chop':partner_cp,'urgent':urgent,'clue_scores':clue_scores,'clue_foci':clue_foci,'clue_gains':clue_gains,'focus':jnp.where(hint,focus,-1),'info':info,'deck':deck,'scores':scores,'hand':hand,'heights':h,'last':last,'lives':lives,'touched':touched,'discard':d,'weak':weak}
    return action,memory,trace
