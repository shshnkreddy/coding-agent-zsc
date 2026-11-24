"""H-group Level 1 policy for the configured two-player Hanabi game.

Each seat tracks its own convention notes and a conservative model of its
partner's notes. All state is private, fixed-shape JAX memory. Decisions use
shared clue meanings; tactical tie-breaking carries no additional signal.
"""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_notes', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {'notes': jnp.ones((2,5,25), bool), 'touched': jnp.zeros((2,5), bool),
            'reliable': jnp.zeros((2,5),bool), 'trusted':jnp.zeros((2,5),bool), 'promised':jnp.zeros((2,5),bool), 'groups':jnp.full((2,5),-1,jnp.int32), 'links':jnp.zeros((2,5,5),bool), 'empty_seen':jnp.array(0,jnp.int32), 'early': jnp.array(True), 'turn': jnp.array(0, jnp.int32)}


def normalize(x):
    return x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), 1e-8)


def intersect(a, b):
    x = a & b
    return jnp.where(jnp.any(x, axis=-1, keepdims=True), x, b)


def restrict(a, b):
    x = a & b
    return jnp.where(jnp.any(x, axis=-1, keepdims=True), x, a)


def count_refine(notes, counts, occupied, reliable):
    # Hall-style capacity deductions: if a subset of slots exhausts every
    # physical copy of its possible identities, other slots cannot hold them.
    subsets = (jnp.arange(1,32)[:,None] & (1 << jnp.arange(5)[None,:])) > 0
    sizes = jnp.sum(subsets,axis=-1)
    valid = jnp.all(~subsets | occupied[None,:],axis=-1)
    for _ in range(2):
        union = jnp.any(subsets[:,:,None] & notes[None,:,:],axis=1)
        cap = jnp.sum(union*counts[None,:],axis=-1)
        # Reliable Good Touch cards are also distinct, even if multiple
        # physical copies remain outside our hand.
        distinct = jnp.all(~subsets | reliable[None,:],axis=-1)
        cap = jnp.where(distinct,jnp.minimum(cap,jnp.sum(union,axis=-1)),cap)
        tight = valid & (cap == sizes)
        # All-different deductions apply only to other reliable cards when
        # physical capacity itself was not exhausted.
        physical = jnp.sum(union*counts[None,:],axis=-1) == sizes
        forbidden = jnp.any(tight[:,None,None] & ~subsets[:,:,None] & union[:,None,:] & (physical[:,None,None] | reliable[None,:,None]),axis=0)
        notes = restrict(notes,~forbidden)
    return notes


def chop(touched, occupied):
    return jnp.min(jnp.where(~touched & occupied, jnp.arange(5), 5))


def focus_of(matches, touched, occupied):
    ch = chop(touched, occupied)
    new = matches & ~touched
    f = jnp.where(jnp.any(new), jnp.max(jnp.where(new, jnp.arange(5), -1)),
                  jnp.max(jnp.where(matches, jnp.arange(5), -1)))
    return jnp.where((ch < 5) & matches[jnp.minimum(ch,4)], ch, f)


def board_data(f):
    heights = jnp.sum(f[167:192].reshape(5,5) * jnp.arange(1,6), axis=-1).astype(jnp.int32)
    d = f[203:253].reshape(5,10)
    disc = jnp.stack([d[:,0:3].sum(-1),d[:,3:5].sum(-1),d[:,5:7].sum(-1),d[:,7:9].sum(-1),d[:,9]],-1)
    ranks = jnp.arange(5)[None,:]
    remaining = jnp.array([3,2,2,2,1])[None,:] - disc - (ranks < heights[:,None])
    blocked = jnp.cumsum((remaining <= 0) & (ranks >= heights[:,None]), axis=-1) > 0
    useful = (ranks >= heights[:,None]) & ~blocked
    playable = ranks == heights[:,None]
    critical = useful & (remaining == 1)
    return heights, remaining.reshape(25), useful.reshape(25), playable.reshape(25), critical.reshape(25)


def refine(notes, literal, touched, still_unplayed):
    notes = intersect(notes, literal)
    # A card that was useful when clued can become blocked after a lower
    # rank is lost. Keep that identity so it can be recognized as trash.
    good = still_unplayed
    notes = jnp.where(touched[:,None], restrict(notes, good), notes)
    # Exact notes establish mutual exclusion among clued cards.
    for _ in range(2):
        exact = (jnp.sum(notes,axis=-1) == 1) & touched
        count = jnp.sum(notes & exact[:,None],axis=0)
        other = count[None,:] - (notes & exact[:,None])
        clean = notes & (other == 0)
        notes = jnp.where(touched[:,None], restrict(notes,clean),notes)
    return notes


def clue_notes(notes, literal, touched, occupied, matches, is_rank, value,
               heights, _remaining, useful, critical, giver_cards, giver_touched, reliable, _forced):
    """Interpret a clue from the recipient's information, retaining save alternatives."""
    focus = focus_of(matches,touched,occupied)
    ch = chop(touched,occupied)
    is_chop = (focus == ch) & (ch < 5)
    base = intersect(notes,literal)
    giver_types = jnp.any(giver_cards & giver_touched[:,None],axis=0)
    # Existing clued connectors may give a delayed play or a self-prompt.
    eligible = touched & (jnp.arange(5) != focus)
    # Reachability over subsets of recipient cards. A single ambiguous card
    # cannot supply two different ranks in the same delayed-play chain.
    subset = jnp.arange(32)
    bits = 1 << jnp.arange(5)
    has_bit = (subset[:,None] & bits[None,:]) != 0
    previous = subset[:,None] ^ bits[None,:]
    own_options = base.reshape(5,5,5).transpose(1,2,0) & eligible[None,None,:]
    given = giver_types.reshape(5,5)
    initial_dp = jnp.zeros((5,32),bool).at[:,0].set(True)
    def connect(dp, rank):
        possible_focus = jnp.any(dp,axis=-1) & (rank>=heights)
        using_own = dp[:,previous] & has_bit[None,:,:] & own_options[:,rank,None,:]
        after = jnp.any(using_own,axis=-1)
        skip = (rank<heights) | given[:,rank]
        after = jnp.where(skip[:,None],dp,after)
        return after, possible_focus
    _, reachable = jax.lax.scan(connect,initial_dp,jnp.arange(5))
    reachable = reachable.T.reshape(25) & useful
    save = critical & (jnp.tile(jnp.arange(5),5) != 4)
    save = save | (is_rank & ((value == 1) | (value == 4)) & useful)
    save = save & is_chop
    # Giver cannot see its own identities: the rank-2 visible-copy rule must
    # not use the recipient's view of the giver's hidden hand.
    possible = (reachable | save) & ~giver_types
    foc = restrict(base[focus], possible)
    base = base.at[focus].set(foc)
    # Apply Good Touch at clue time, including side touches. Save Principle
    # can require a giver to violate it; misplay recovery retracts such notes.
    save_possible = jnp.any(save & literal[focus])
    new = matches & ~touched
    new_reliable = jnp.any(base[focus] & possible)
    reliability = jnp.where(new,new_reliable,reliable)
    focus_reliable = jnp.any(base[focus] & possible)
    reliability = reliability.at[focus].set(reliability[focus] | focus_reliable)
    # Existing side touches retain their earlier interpretation.
    base = jnp.where((new & new_reliable)[:,None],restrict(base,useful & ~giver_types),base)
    base = refine(base,literal,reliability,jnp.tile(jnp.arange(5),5)>=jnp.repeat(heights,5))
    is_play_promise = ~save_possible & jnp.any(base[focus] & reachable)
    base, reliability, linkrow = infer_prompt(base,reliability,touched,focus,heights,giver_types,is_play_promise)
    return base, focus, is_chop, reachable, save, reliability, linkrow, is_play_promise


def infer_prompt(base, reliability, touched, focus, heights, giver_types, promised):
    # A clued prefix can play successfully in another color before the needed
    # connector is found (Prompt chapter). Continue until the promise is met.
    colors = jnp.repeat(jnp.arange(5),5)
    ranks = jnp.tile(jnp.arange(5),5)
    needed = colors*5+jnp.minimum(heights[colors],4)
    eligible = touched & (jnp.arange(5)!=focus)
    compatible = base[:,needed] & eligible[:,None]
    playing = jnp.any(base,axis=-1) & jnp.all(~base | (ranks==heights[colors])[None,:],axis=-1)
    ready = compatible & playing[:,None]
    already_connecting = jnp.any(ready,axis=0)
    # A known play may support the promise without being newly prompted.
    candidates = jnp.where(already_connecting[None,:],ready,compatible)
    prompt_slots = jnp.max(jnp.where(candidates,jnp.arange(5)[:,None],-1),axis=0)
    needs_own = (ranks > heights[colors]) & ~giver_types[needed] & (prompt_slots >= 0)
    choices = base[focus]
    lo = jnp.min(jnp.where(choices,prompt_slots,5))
    hi = jnp.max(jnp.where(choices,prompt_slots,-1))
    do_link = jnp.any(choices) & jnp.all(~choices | needs_own) & (lo==hi) & (lo>=0) & promised
    do_prompt = do_link & ~jnp.any(choices & already_connecting)
    ps = jnp.clip(lo,0,4)
    # The first compatible clued card must play, but it need not be the
    # connecting color if further compatible clued cards remain.
    connector_types = ranks == heights[colors]
    base = base.at[ps].set(jnp.where(do_prompt,restrict(base[ps],connector_types),base[ps]))
    reliability = reliability.at[ps].set(reliability[ps] | do_prompt)
    linkrow = do_link & (jnp.arange(5)==ps)
    return base, reliability, linkrow


def safe_prompt_chain(notes, touched, focus, actual_types, heights, giver_types):
    """Check visible prefixes, without relying on a partner's tactical tie-break."""
    target = actual_types[focus]
    color, rank = target//5, target%5
    eligible = touched & (jnp.arange(5)!=focus)
    ranks = jnp.tile(jnp.arange(5),5)
    colors = jnp.repeat(jnp.arange(5),5)
    def advance(_,carry):
        h,available,n,ok = carry
        done = h[color]>=rank
        needed = color*5+jnp.minimum(h[color],4)
        giver = giver_types[needed]
        compatible = available & n[:,needed]
        playing = jnp.any(n,axis=-1) & jnp.all(~n | (ranks==h[colors])[None,:],axis=-1)
        ready = compatible & playing
        choices = jnp.where(jnp.any(ready),ready,compatible)
        slot = jnp.maximum(jnp.max(jnp.where(choices,jnp.arange(5),-1)),0)
        actual = actual_types[slot]
        safe = jnp.any(choices) & (actual%5==h[actual//5])
        take = ~done & ~giver & safe
        advance_color = jnp.where(giver,color,actual//5)
        move = ~done & (giver | safe)
        h = h + ((jnp.arange(5)==advance_color) & move).astype(jnp.int32)
        available = available & ~(take & (jnp.arange(5)==slot))
        n = restrict(n,ranks>=h[colors])
        return h,available,n,ok & (done | giver | safe)
    h,_,_,ok = jax.lax.fori_loop(0,10,advance,(heights,eligible,notes,jnp.array(True)))
    return ok & (h[color]>=rank)


def resolve_pending(notes,reliable,trusted,links,groups,promised,touched,heights,giver_types):
    def resolve(slot,carry):
        n,r,t,l,g = carry
        n,r,row = infer_prompt(n,r,touched,slot,heights,giver_types,promised[slot])
        l = l.at[slot].set(l[slot] | row)
        t = t | row
        g = jnp.where(row,g[slot],g)
        return n,r,t,l,g
    return jax.lax.fori_loop(0,5,resolve,(notes,reliable,trusted,links,groups))


def agent_act(obs, reset, memory, rng):
    m = jax.tree.map(lambda a,b:jnp.where(reset,a,b),init_agent(),memory)
    f = obs.features
    legal = obs.legal_actions
    cards = f[:125].reshape(5,25) > 0
    native = f[308:658].reshape(2,5,35)
    literal = native[:,:,:25] > 0
    touched = jnp.any(native[:,:,25:] > 0,axis=-1)
    occupied = jnp.any(literal,axis=-1)
    heights,remaining,useful,playable,critical = board_data(f)
    unseen = jnp.maximum(remaining-jnp.sum(cards,axis=0),0)
    literal = literal.at[0].set(literal[0] & (unseen > 0))
    last = f[253:308]
    actor = jnp.argmax(last[:2])
    kind = jnp.argmax(last[2:6])
    has_last = jnp.any(last[:2] > 0) & ~reset
    removed = jnp.argmax(last[23:28])
    removal = has_last & (kind < 2)
    slots = jnp.arange(5)
    shifted_indices = jnp.minimum(slots + (slots >= removed),4)
    oldnotes = m['notes']
    groups = m['groups']
    trusted = m['trusted']
    promised = m['promised']
    reliable = m['reliable']
    bad_play = removal & (kind==0) & (last[53]==0)
    bad_group = groups[actor,removed]
    withdrawn = (groups[actor]==bad_group) & (bad_group>=0) & bad_play
    oldnotes = oldnotes.at[actor].set(jnp.where(withdrawn[:,None],True,oldnotes[actor]))
    reliable = reliable.at[actor].set(reliable[actor] & ~withdrawn)
    trusted = trusted.at[actor].set(trusted[actor] & ~withdrawn)
    promised = promised.at[actor].set(promised[actor] & ~withdrawn)
    links = m['links']
    links = links.at[actor].set(links[actor] & ~withdrawn[:,None] & ~withdrawn[None,:])
    revealed = jnp.argmax(last[28:53])
    link_targets = links[actor,:,removed] & removal & (kind==0) & (last[53]>0)
    prior_heights = heights - ((jnp.arange(5)==revealed//5) & (last[53]>0)).astype(jnp.int32)
    req = jnp.arange(5)*5+jnp.minimum(prior_heights,4)
    eligible_other = m['touched'][actor][None,:] & (slots[None,:]!=removed) & (slots[:,None]!=slots[None,:])
    other_support = jnp.any(eligible_other[:,:,None] & oldnotes[actor][:,req][None,:,:],axis=1)
    keep_colors = other_support | (jnp.arange(5)==revealed//5)[None,:]
    linked_notes = restrict(oldnotes[actor],jnp.repeat(keep_colors,5,axis=-1))
    lost_connector = links[actor,:,removed] & removal & ((kind==1) | (last[53]==0))
    promised = promised.at[actor].set(promised[actor] & ~lost_connector)
    oldnotes = oldnotes.at[actor].set(jnp.where(link_targets[:,None],linked_notes,oldnotes[actor]))
    shifted_links = links[actor][shifted_indices[:,None],shifted_indices[None,:]] & (slots<4)[:,None] & (slots<4)[None,:]
    links = links.at[actor].set(jnp.where(removal,shifted_links,links[actor]))
    oldtouch = m['touched']
    shifted = jnp.where((slots < 4)[:,None],oldnotes[actor,shifted_indices],True)
    oldnotes = oldnotes.at[actor].set(jnp.where(removal,shifted,oldnotes[actor]))
    shifted_touch = jnp.where(slots < 4,oldtouch[actor,shifted_indices],False)
    old_chop = chop(oldtouch[actor],jnp.ones(5,bool))
    early = m['early'] & ~(removal & (kind == 1) & (removed == old_chop))
    oldtouch = oldtouch.at[actor].set(jnp.where(removal,shifted_touch,oldtouch[actor]))
    relshift = jnp.where(slots < 4,reliable[actor,shifted_indices],False)
    reliable = reliable.at[actor].set(jnp.where(removal,relshift,reliable[actor]))
    group_shift = jnp.where(slots<4,groups[actor,shifted_indices],-1)
    groups = groups.at[actor].set(jnp.where(removal,group_shift,groups[actor]))
    trust_shift = jnp.where(slots<4,trusted[actor,shifted_indices],False)
    trusted = trusted.at[actor].set(jnp.where(removal,trust_shift,trusted[actor]))
    promise_shift = jnp.where(slots<4,promised[actor,shifted_indices],False)
    promised = promised.at[actor].set(jnp.where(removal,promise_shift,promised[actor]))
    notes = intersect(oldnotes,literal)
    # A newly clued visible copy can conflict with an older ambiguous promise.
    new_partner_types = jnp.any(cards & (touched[1] & ~oldtouch[1])[:,None],axis=0)
    conflict = jnp.any(notes[0] & new_partner_types,axis=-1) & oldtouch[0]
    reliable = reliable.at[0].set(reliable[0] & ~conflict)
    # Own notes use visible partner cards. Partner notes only use our publicly
    # certain cards; no hidden card identities enter the policy.
    known_self = notes[0] & (jnp.sum(notes[0],axis=-1,keepdims=True) == 1)
    notes = notes.at[0].set(refine(notes[0],literal[0],reliable[0],jnp.tile(jnp.arange(5),5)>=jnp.repeat(heights,5)))
    notes = notes.at[1].set(refine(notes[1],literal[1],reliable[1],jnp.tile(jnp.arange(5),5)>=jnp.repeat(heights,5)))
    recipient = 1-actor
    giver_cards = jnp.where(recipient == 0,cards,known_self)
    newnotes,focus,_,_,_,newrel,linkrow,isplay = clue_notes(notes[recipient],literal[recipient],oldtouch[recipient],occupied[recipient],
        last[18:23] > 0,kind == 3,jnp.argmax(jnp.where(kind == 3,last[13:18],last[8:13])),
        heights,remaining,useful,critical,giver_cards,oldtouch[actor],reliable[recipient],jnp.sum(f[192:200])==7)
    is_hint = has_last & (kind >= 2)
    notes = notes.at[recipient].set(jnp.where(is_hint,newnotes,notes[recipient]))
    reliable = reliable.at[recipient].set(jnp.where(is_hint,newrel,reliable[recipient]))
    links = links.at[recipient,focus].set(links[recipient,focus] | (is_hint & linkrow))
    affected = (last[18:23]>0) | linkrow
    groups = groups.at[recipient].set(jnp.where(is_hint & affected,m['turn'],groups[recipient]))
    trust_new = (slots==focus) | linkrow
    trusted = trusted.at[recipient].set(trusted[recipient] | (is_hint & trust_new))
    promised = promised.at[recipient,focus].set(jnp.where(is_hint,isplay,promised[recipient,focus]))
    own_result = resolve_pending(notes[0],reliable[0],trusted[0],links[0],groups[0],promised[0],touched[0],heights,jnp.any(cards & touched[1,:,None],axis=0))
    notes = notes.at[0].set(own_result[0]); reliable = reliable.at[0].set(own_result[1])
    trusted = trusted.at[0].set(own_result[2]); links = links.at[0].set(own_result[3]); groups = groups.at[0].set(own_result[4])
    notes = notes.at[0].set(count_refine(notes[0],unseen,occupied[0],reliable[0]))
    # We can subtract the visible hand from public card counts for own probabilities.
    unseen = jnp.maximum(remaining-jnp.sum(cards,axis=0),0)
    ownprob = normalize(notes[0]*unseen[None,:])
    ownprob = jnp.where(jnp.sum(ownprob,axis=-1,keepdims=True)>0,ownprob,normalize(literal[0]*unseen[None,:]))
    pp = jnp.sum(ownprob*playable,axis=-1)
    trashp = jnp.sum(ownprob*~useful,axis=-1)
    risk = jnp.sum(ownprob*critical*(5-jnp.tile(jnp.arange(5),5)),axis=-1)
    info = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    deck = jnp.sum(f[127:167])
    end_phase = jnp.where(deck==0,m['empty_seen']+1,0)
    mychop = chop(touched[0],occupied[0])
    pchop = chop(touched[1],occupied[1])
    pprob = normalize(notes[1]*remaining[None,:])
    partner_pp = jnp.sum(pprob*playable,axis=-1)
    partner_safe = jnp.any((partner_pp > .999) & occupied[1])
    partner_trash = jnp.any((jnp.sum(pprob*~useful,axis=-1) > .999) & occupied[1])
    actual_types = jnp.argmax(cards,axis=-1)
    actual_useful = useful[actual_types] & occupied[1]
    actual_play = playable[actual_types] & occupied[1]
    actual_critical = critical[actual_types]
    pc = jnp.minimum(pchop,4)
    potential_own = jnp.any(notes[0] & touched[0,:,None],axis=0)
    unique = (jnp.sum(actual_types == actual_types[pc]) == 1) & ~potential_own[actual_types[pc]]
    # Level 1 defines playable to include delayed plays whose lower ranks
    # are already accounted for. Save Principle protects these on chop too.
    my_known = notes[0] & (jnp.sum(notes[0],axis=-1,keepdims=True) == 1)
    accounted_types = jnp.any(cards & touched[1,:,None],axis=0) | jnp.any(my_known & touched[0,:,None],axis=0)
    accounted = accounted_types.reshape(5,5) | (jnp.arange(5)[None,:] < heights[:,None])
    connected = jnp.concatenate([jnp.ones((5,1),bool),jnp.cumprod(accounted.astype(jnp.int32),axis=-1)[:,:-1]],axis=-1).reshape(25)
    chop_danger = unique & (pchop < 5) & actual_useful[pc] & (actual_critical[pc] | connected[actual_types[pc]] | (actual_types[pc]%5 == 1))
    urgent = chop_danger & ~partner_safe & ~partner_trash

    type_rank = jnp.tile(jnp.arange(5),5)
    next_type = jnp.minimum(jnp.arange(25)+1,24)
    partner_next = jnp.sum(cards*touched[1,:,None],axis=0)[next_type]*(type_rank<4)
    own_next = jnp.sum(ownprob*touched[0,:,None],axis=0)[next_type]*(type_rank<4)
    continuation = 2.2*partner_next + 1.5*own_next

    partner_literal_pp = jnp.sum(native[1,:,:25]*playable,axis=-1)
    partner_end_allowed = (lives>1) | trusted[1] | (partner_literal_pp>.999)
    partner_end_best = jnp.max(jnp.where(partner_end_allowed,partner_pp,0.))
    partner_end_best = jnp.where(lives>1,partner_end_best,jnp.where(partner_end_best>.999,1.,0.))
    partner_certain = jnp.any((partner_pp>.999) & actual_play & partner_end_allowed)

    def evaluate_hint(h):
        rank = h >= 5
        val = h%5
        attr = jnp.where(rank,actual_types%5,actual_types//5)
        matches = (attr == val) & occupied[1]
        mask = jnp.where(rank,jnp.tile(jnp.arange(5),5)==val,jnp.repeat(jnp.arange(5),5)==val)
        lit = literal[1] & jnp.where(matches[:,None],mask[None,:],~mask[None,:])
        pred,fc,_,reachable,save,_,linkrow,_ = clue_notes(notes[1],lit,touched[1],occupied[1],matches,rank,val,
                     heights,remaining,useful,critical,my_known,touched[0],reliable[1],info==8)
        actual = actual_types[fc]
        # Giving a delayed clue requires the actual visible connectors to exist;
        # mere compatibility in recipient notes is insufficient evidence.
        actual_accounted = jnp.any(cards & (touched[1] & (slots!=fc))[:,None],axis=0) | jnp.any(my_known & touched[0,:,None],axis=0)
        account = actual_accounted.reshape(5,5) | (jnp.arange(5)[None,:]<heights[:,None])
        actual_chain = jnp.concatenate([jnp.ones((5,1),bool),jnp.cumprod(account.astype(jnp.int32),axis=-1)[:,:-1]],axis=-1).reshape(25)
        chain_safe = safe_prompt_chain(pred,touched[1],fc,actual_types,heights,jnp.any(my_known & touched[0,:,None],axis=0))
        valid_meaning = (reachable[actual] & actual_chain[actual] & chain_safe) | save[actual]
        prediction = normalize(pred*remaining[None,:])
        afterpp = jnp.sum(prediction*playable,axis=-1)
        new = matches & ~touched[1]
        newcount = jnp.sum(new)
        # Bad touch includes duplicates already clued anywhere, and duplicate
        # copies introduced together. Save Principle permits the lesser evil.
        duplicate_own = jnp.any(notes[0] & touched[0,:,None],axis=0)[actual_types]
        eq = actual_types[:,None] == actual_types[None,:]
        duplicate_hand = jnp.any(eq & ~jnp.eye(5,dtype=bool) & (touched[1]|matches)[None,:],axis=-1)
        bad = jnp.sum(matches & (~actual_useful | duplicate_own | duplicate_hand))
        saves_chop = matches[pc] & (pchop < 5) & valid_meaning
        gets_play = jnp.any((afterpp > .999) & (partner_pp < .999) & actual_play)
        unsafe = jnp.any((afterpp > .999) & ~actual_play & occupied[1])
        same_color = actual_types[:,None]//5 == actual_types[None,:]//5
        out_of_order = jnp.any(touched[1,:,None] & touched[1,None,:] & same_color &
            (slots[:,None] < slots[None,:]) & (actual_types[:,None]%5 < actual_types[None,:]%5) &
            actual_play[:,None] & notes[1][:,actual_types].T)
        # Prompt question 3 explicitly permits tempo to repair clued cards
        # that would otherwise prompt in the wrong order.
        # Save Principle explicitly overrides MCVP when a play prevents an
        # otherwise imminent dangerous chop discard.
        can_tempo = (mychop == 5) | (info == 8) | (deck == 0) | ((out_of_order | urgent) & gets_play)
        convention_ok = pred[fc,actual] & valid_meaning & ((newcount > 0) | can_tempo) & ((bad == 0) | (saves_chop & urgent & ~jnp.any(matches & duplicate_own))) & ~unsafe
        resolves_urgent = saves_chop | gets_play
        new_certain = (afterpp>.999) & (partner_pp<.999) & actual_play
        value = 2.0 + .4*newcount + 6.0*jnp.sum(new_certain)
        value += .25*jnp.sum(jnp.maximum(afterpp-partner_pp,0)*actual_play)
        # Risk tolerance differs across partners. Prefer clues that do not
        # make an unplayable card appear likely to play.
        value -= 5.0*jnp.sum(jnp.maximum(afterpp-partner_pp,0)*(~actual_play) * occupied[1])
        new_play_slots = (afterpp>.999) & (partner_pp<.999) & actual_play
        value += jnp.sum(new_play_slots*(.7*continuation[actual_types]+.6*(actual_types%5==4)))
        value += .25*(~rank) + .3*jnp.sum(new*actual_critical) - 3.0*bad
        value += jnp.where(urgent & resolves_urgent,15.,0.)
        value -= jnp.where(urgent & ~resolves_urgent,15.,0.)
        value -= 1.2*jnp.sum((afterpp>.999)&(partner_pp>.999))
        value -= jnp.where((info<=1) & partner_safe & ~urgent,4.0,0.)
        value += jnp.where(early,.5,0.)
        value = jnp.where(early & (newcount>0),jnp.maximum(value,1.5),value)
        # When game legality forces a clue, minimize misleading implications.
        truth_lost = jnp.sum(matches & ~pred[jnp.arange(5),actual_types])
        fallback = -100. + 15*valid_meaning + 2*gets_play - 50*unsafe - 20*bad - 50*truth_lost
        fallback += .2*rank - .2*newcount + 3.0*(rank & (val==4))
        # Literal identity cannot be mistaken for an alternate delayed card
        # based on private knowledge that our partner has about our hand.
        fallback += 12.0*(jnp.sum(lit[fc])==1)
        post_allowed = (lives>1) | trusted[1] | (slots==fc) | linkrow | (partner_literal_pp>.999)
        post_best = jnp.max(jnp.where(post_allowed & actual_play,afterpp,0.))
        post_best = jnp.where(lives>1,post_best,jnp.where(post_best>.999,1.,0.))
        normal = jnp.where(convention_ok,value,fallback)
        # A certain partner play is robust to differences in willingness to
        # gamble. Future-only saves have no value after the deck is empty.
        certain_gain = (post_best>.999) & ~partner_certain
        end_gain = jnp.where(certain_gain,2.+1e-5*jnp.maximum(normal,0.),jnp.maximum(post_best-partner_end_best,0.))
        end_value = jnp.where((end_phase==1) & convention_ok,end_gain,-1.)
        return jnp.where(legal[10+h],jnp.where(end_phase>0,end_value,normal),-1000.)

    hint_values = jax.vmap(evaluate_hint)(jnp.arange(10))
    # Play only cards all remaining convention-consistent identities can play.
    play_values = 8.0 + .15*slots + .7*jnp.sum(ownprob*(jnp.tile(jnp.arange(5),5)==4),axis=-1)
    play_values += .3*jnp.sum(ownprob*critical,axis=-1)
    play_values += jnp.minimum(jnp.sum(ownprob*continuation[None,:],axis=-1),5.)
    literal_pp = jnp.sum(normalize(literal[0]*unseen[None,:])*playable,axis=-1)
    protected = (lives>1) | trusted[0] | (literal_pp>.999)
    play_values = jnp.where((pp >= .999)&occupied[0]&protected,play_values - 7*(1-pp),-1000.)
    partner_safe_end = jnp.where(partner_end_best>.999,1.,0.)
    risk_cost = jnp.where((end_phase==1) & (lives==2),(1-pp)*(partner_end_best-partner_safe_end),0.)
    end_play_value = pp - risk_cost + jnp.where(pp>.999,1e-5*play_values,0.)
    end_play_value += jnp.where((end_phase==1) & (pp>.999),2.,0.)
    end_play_allowed = ((lives>1) & (pp>0)) | ((pp>.999) & protected)
    play_values = jnp.where(end_phase>0,jnp.where(end_play_allowed & occupied[0],end_play_value,-1000.),play_values)
    # Publicly known trash precedes ordinary chop.
    discard_values = jnp.where((trashp>.999)&occupied[0],2.5-.01*slots,-1000.)
    discard_values = jnp.where((slots == mychop)&occupied[0],jnp.maximum(discard_values,1.0),discard_values)
    # If every card is clued, prefer a low-risk discard to a misleading clue.
    discard_values = jnp.where(mychop==5,trashp*3-risk*4-.01*slots,discard_values)
    discard_values = jnp.where((end_phase>0)&occupied[0],0.,discard_values)
    values = jnp.concatenate([discard_values,play_values,hint_values,jnp.array([-1000.])])
    values = jnp.where(legal,values,-1e6)
    # Legal fallback for forced-clue states; no extra signaling meaning.
    action = jnp.argmax(values).astype(jnp.int32)
    action = jnp.where(legal[20],20,action).astype(jnp.int32)
    nextmem = {'notes':notes,'touched':touched,'reliable':reliable,'trusted':trusted,'promised':promised,'groups':groups,'links':links,'empty_seen':end_phase,'early':early,'turn':m['turn']+1}
    trace = {'pp':pp,'trash':trashp,'hint_values':hint_values,'notes':notes[0].astype(jnp.int32),
             'chop':mychop,'urgent':urgent,'focus':jnp.where(is_hint,focus,-1),
             'reliable':reliable[0],'trusted':trusted[0],'promised':promised[0],'groups':groups[0],'links':links[0],'partner':jnp.where(occupied[1],actual_types,-1),'heights':heights,'info':info,'lives':lives,'deck':deck,'end_phase':end_phase,'last':last}
    return action,nextmem,trace
