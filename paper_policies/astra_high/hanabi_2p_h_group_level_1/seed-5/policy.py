"""Two-player H-group Level 1 policy with persistent, local card notes."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'level_one', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {'notes': jnp.ones((2, 5, 25), bool),
            'links': jnp.full((2, 5, 25, 5), -1, jnp.int32),
            'clued': jnp.zeros((2, 5), bool),
            'soft': jnp.zeros((2, 5), bool),
            'groups': jnp.full((2, 5),-1,jnp.int32),
            'early': jnp.asarray(True),
            'empty_steps': jnp.asarray(0, jnp.int32),
            'turn': jnp.asarray(0, jnp.int32)}


def nonempty_or(a, b):
    return jnp.where(jnp.any(a, axis=-1, keepdims=True), a, b)


def decode(obs):
    f = obs.features
    hand = f[:125].reshape(5,25) > 0
    k = f[308:].reshape(2,5,35)
    literal = k[:,:,:25] > 0
    clued = jnp.any(k[:,:,25:] > 0, axis=-1)
    occupied = jnp.any(literal, axis=-1)
    stacks = jnp.sum(f[167:192].reshape(5,5) * jnp.arange(1,6), axis=1).astype(jnp.int32)
    ds = f[203:253].reshape(5,10)
    disc = jnp.stack([ds[:,:3].sum(1),ds[:,3:5].sum(1),ds[:,5:7].sum(1),ds[:,7:9].sum(1),ds[:,9]],axis=-1)
    return hand,literal,clued,occupied,stacks,disc,jnp.sum(f[192:200]),jnp.sum(f[200:203]),jnp.sum(f[127:167])


def board_masks(stacks, disc):
    r = jnp.arange(5)[None,:]
    counts = jnp.array([3,2,2,2,1])[None,:] - disc
    played = r < stacks[:,None]
    blocked = jnp.cumsum((counts <= 0) & ~played, axis=1) > 0
    useful = ~played & ~blocked
    playable = r == stacks[:,None]
    remaining = jnp.maximum(counts - played, 0)
    critical = (counts == 1) & useful
    return useful.reshape(25),playable.reshape(25),remaining.reshape(25),critical.reshape(25)


def chop(tags, occupied):
    return jnp.min(jnp.where(~tags & occupied, jnp.arange(5), 5))


def focus_of(touch, tags, occupied):
    ch = chop(tags, occupied)
    new = touch & ~tags
    focus = jnp.where(jnp.any(new), jnp.where((ch < 5) & touch[jnp.minimum(ch,4)], ch,
                    jnp.max(jnp.where(new, jnp.arange(5), -1))),
                    jnp.max(jnp.where(touch, jnp.arange(5), -1)))
    return jnp.maximum(focus,0), ch, new


def clean_notes(notes, literal, tags, useful, other_known, soft=None):
    if soft is None:
        soft = jnp.zeros(5, bool)
    notes = nonempty_or(notes & literal, literal)
    # Good Touch removes identities already assigned to another clued card.
    known = (notes.sum(-1) == 1) & tags
    exact = notes & known[:,None]
    claimed = jnp.any(exact, axis=0)
    exclusions = claimed[None,:] & ~exact
    narrowed = notes & useful[None,:] & ~exclusions
    notes = jnp.where((tags & ~soft)[:,None], nonempty_or(narrowed, notes), notes)
    return notes


def resolve_prompt(notes, focus, links, stacks):
    """Only a connector shared by every remaining play interpretation is forced."""
    ids = jnp.arange(25)
    next_ranks = stacks[ids//5]
    slots = links[ids,jnp.minimum(next_ranks,4)]
    fn = notes[focus]
    needs = (next_ranks < ids%5) & (slots >= 0)
    ps = jnp.max(jnp.where(fn,slots,-1))
    forced = jnp.any(fn) & jnp.all(~fn | (needs & (slots == ps)))
    connector_ids = ids//5*5+jnp.minimum(next_ranks,4)
    # A leftmost compatible prompted card may be another playable color;
    # the retained link continues searching for the promised connector.
    options = ids%5 == stacks[ids//5]
    slot = jnp.maximum(ps,0)
    narrowed = nonempty_or(notes[slot] & options,notes[slot])
    return notes.at[slot].set(jnp.where(forced,narrowed,notes[slot]))


def interpret(notes, literal, tags, occupied, other_cards, other_tags,
              touch, is_rank, value, stacks, disc, soft=None):
    if soft is None:
        soft = jnp.zeros(5, bool)
    """Interpret one public clue, retaining all Level 1 play/save alternatives."""
    useful, playable, remaining, critical = board_masks(stacks, disc)
    focus, ch, new = focus_of(touch, tags, occupied)
    ids = jnp.arange(25)
    colors, ranks = ids // 5, ids % 5
    other_exact = other_cards & (other_cards.sum(-1) == 1)[:,None] & other_tags[:,None]
    other_known = jnp.any(other_exact, axis=0)
    n = nonempty_or(notes & literal, literal)
    raw = n
    # Apply Good Touch only to newly touched cards; the constraints persist.
    narrowed = n & useful[None,:]
    n = jnp.where(new[:,None], nonempty_or(narrowed, n), n)
    # All connecting cards must already be clued. External connections may be
    # visible identities (our perspective) or possible identities (our model of
    # what a receiver sees in our hidden hand). This latter use is conservative.
    available = jnp.any(other_cards & other_tags[:,None],axis=0)
    used = jnp.zeros((25,5),bool)
    chain_links = jnp.full((25,5),-1,jnp.int32)
    play_options = useful & ~other_known
    for rank in range(5):
        connector_ids = colors*5+rank
        needed = (rank >= stacks[colors]) & (rank < ranks)
        external = available[connector_ids]
        compatible = n[:,connector_ids].T & tags[None,:] & ~used & (jnp.arange(5)[None,:] != focus)
        selected = jnp.max(jnp.where(compatible,jnp.arange(5)[None,:],-1),axis=-1)
        internal = needed & ~external
        play_options &= ~needed | external | (selected >= 0)
        chain_links = chain_links.at[:,rank].set(jnp.where(internal,selected,-1))
        used |= internal[:,None] & (jnp.arange(5)[None,:] == selected[:,None])
    # A color clue never saves a five; rank 2/5 saves only focus on chop.
    save_options = useful & ((critical & (ranks != 4)) |
                            (is_rank & (value == 4) & (ranks == 4)) |
                            (is_rank & (value == 1) & (ranks == 1)))
    save_options &= focus == ch
    meaning = play_options | save_options
    fn = nonempty_or(n[focus] & meaning, n[focus])
    n = n.at[focus].set(fn)
    could_save = jnp.any(raw[focus] & save_options)
    soft = soft | (new & (jnp.arange(5) != focus) & could_save)
    soft = soft.at[focus].set(False)
    n = jnp.where((new & soft)[:,None], raw, n)
    links = jnp.where((fn & play_options & ~save_options)[:,None],chain_links,-1)
    n = resolve_prompt(n,focus,links,stacks)
    not_played = jnp.arange(25)%5 >= stacks[jnp.arange(25)//5]
    n = clean_notes(n, literal, tags | touch, not_played, other_known, soft)
    return n, links, focus, save_options, soft, play_options


def count_consistency(notes, counts):
    """Hall-set elimination for five hidden slots and known card multiplicities."""
    subsets = (jnp.arange(1,32)[:,None] & (1 << jnp.arange(5)[None,:])) > 0
    sizes = subsets.sum(-1)
    occupied = jnp.any(notes,axis=-1)
    def refine(_, masks):
        union = jnp.any(subsets[:,:,None] & masks[None,:,:],axis=1)
        capacity = (union * counts[None,:]).sum(-1)
        reserved = (capacity == sizes) & jnp.all(~subsets | occupied[None,:],axis=-1)
        excluded = jnp.any(reserved[:,None,None] & ~subsets[:,:,None] & union[:,None,:],axis=0)
        return nonempty_or(masks & ~excluded,masks)
    return jax.lax.fori_loop(0,3,refine,notes)


def probabilities(notes, counts):
    w = notes * counts[None,:]
    return w / jnp.maximum(w.sum(-1,keepdims=True), 1e-8)


def agent_act(obs, reset, memory, rng):
    mem = jax.tree.map(lambda a,b: jnp.where(reset,b,a), memory, init_agent())
    hand,literal,tags,occupied,stacks,disc,tokens,lives,deck = decode(obs)
    useful,playable,remaining,critical = board_masks(stacks,disc)
    last = obs.features[253:308]
    actor = jnp.argmax(last[:2])
    removed = jnp.argmax(last[23:28])
    removing = (last[2] + last[3] > 0) & ~reset
    hinting = (last[4] + last[5] > 0) & ~reset
    recipient = jnp.argmax(last[6:8])
    touch = last[18:23] > 0
    revealed = jnp.argmax(last[28:53])
    notes, links, prev_tags = mem['notes'], mem['links'], mem['clued']
    soft = mem['soft']
    groups = mem['groups']
    early = mem['early'] & ~(removing & (last[3] > 0) & (removed == chop(prev_tags[actor], jnp.any(notes[actor],axis=-1))))
    # A failed play whose identity contradicted our recorded note is public
    # evidence against that clue interpretation. Repair its surviving siblings,
    # keeping literal information and their clued status intact.
    unexpected_failure = removing & (last[2]>0) & (last[53]==0) & ~notes[actor,removed,revealed]
    group = groups[actor,removed]
    dependent = jnp.any(links[actor]==removed,axis=(1,2))
    broken_connection = removing & (last[2]>0) & (last[53]==0) & dependent
    repair = (unexpected_failure & (group>=0) & (groups[actor]==group)) | broken_connection
    notes = notes.at[actor].set(jnp.where(repair[:,None],True,notes[actor]))
    soft = soft.at[actor].set(soft[actor] | repair)
    links = links.at[actor].set(jnp.where(repair[:,None,None],-1,links[actor]))
    # A connecting play resolves the color alternatives retained by an earlier
    # prompt. If a compatible card plays another color, continue left-to-right
    # through the previously clued candidates, as the prompt chapter requires.
    ln = links[actor]
    types = jnp.arange(25)[:,None]//5*5+jnp.arange(5)[None,:]
    candidates = notes[actor][:,types].transpose(1,2,0)
    candidates = candidates[None,:,:,:] & prev_tags[actor,None,None,None,:]
    candidates &= (jnp.arange(5)[None,None,None,:] != removed)
    candidates &= (jnp.arange(5)[None,None,None,:] != jnp.arange(5)[:,None,None,None])
    replacements = jnp.max(jnp.where(candidates,jnp.arange(5)[None,None,None,:],-1),axis=-1)
    used_link = ln == removed
    correct = types == revealed
    valid = jnp.all(~used_link | correct[None,:,:] | (replacements>=0),axis=-1)
    revised = notes[actor] & valid
    notes = notes.at[actor].set(jnp.where(removing & (last[53]>0),nonempty_or(revised,notes[actor]),notes[actor]))
    new_links = jnp.where(used_link,jnp.where(correct[None,:,:],-1,replacements),ln)
    new_links = jnp.where(new_links>removed,new_links-1,new_links)
    links = links.at[actor].set(jnp.where(removing,new_links,ln))
    ix = jnp.arange(5) + (jnp.arange(5) >= removed)
    soft = soft.at[actor].set(jnp.where(removing, jnp.where(ix<5,soft[actor,jnp.minimum(ix,4)],False),soft[actor]))
    groups = groups.at[actor].set(jnp.where(removing,jnp.where(ix<5,groups[actor,jnp.minimum(ix,4)],-1),groups[actor]))
    shifted_notes = jnp.where((ix < 5)[:,None], notes[actor,jnp.minimum(ix,4)], True)
    shifted_links = jnp.where((ix < 5)[:,None,None], links[actor,jnp.minimum(ix,4)], -1)
    notes = notes.at[actor].set(jnp.where(removing,shifted_notes,notes[actor]))
    links = links.at[actor].set(jnp.where(removing,shifted_links,links[actor]))
    prev_tags = prev_tags.at[actor].set(jnp.where(removing,jnp.where(ix<5,prev_tags[actor,jnp.minimum(ix,4)],False),prev_tags[actor]))
    notes = nonempty_or(notes & literal,literal)
    last_life_repair = unexpected_failure & (lives<=1)
    notes = jnp.where(last_life_repair,literal,notes)
    soft = jnp.where(last_life_repair,tags,soft)
    links = jnp.where(last_life_repair,-1,links)
    # For our own inferences, the other hand is visible. For modelling partner
    # knowledge we retain possible publicly inferred identities in our hand.
    own_exact = notes[0] & (notes[0].sum(-1) == 1)[:,None]
    other_cards = jnp.where(recipient == 0,hand,notes[0])
    value = jnp.where(last[5]>0,jnp.argmax(last[13:18]),jnp.argmax(last[8:13]))
    ni,li,fi,sv,si,pi = interpret(notes[recipient],literal[recipient],prev_tags[recipient],occupied[recipient],
                           other_cards,tags[1-recipient],touch,last[5]>0,value,stacks,disc,soft[recipient])
    notes = notes.at[recipient].set(jnp.where(hinting,ni,notes[recipient]))
    soft = soft.at[recipient].set(jnp.where(hinting,si,soft[recipient]))
    regroup = (touch & ~prev_tags[recipient]) | (jnp.arange(5)==fi)
    groups = groups.at[recipient].set(jnp.where(hinting & regroup,mem['turn'],groups[recipient]))
    links = links.at[recipient,fi].set(jnp.where(hinting,li,links[recipient,fi]))
    # Giving a Save Principle exception may knowingly introduce a duplicate of
    # an unresolved own clued card. That weakens our own old Good Touch promise;
    # a later partner play must not silently relabel our old card as another rank.
    introduced = jnp.any(hand & (touch & ~prev_tags[1])[:,None],axis=0)
    duplicate_exception = jnp.any(notes[0] & introduced[None,:],axis=-1) & tags[0]
    soft = soft.at[0].set(soft[0] | (hinting & (recipient==1) & duplicate_exception))
    not_played = jnp.arange(25)%5 >= stacks[jnp.arange(25)//5]
    other_known = jnp.any(hand & tags[1,:,None],axis=0)
    notes = notes.at[0].set(clean_notes(notes[0],literal[0],tags[0],not_played,other_known,soft[0]))
    own_known = jnp.any(notes[0] & (notes[0].sum(-1)==1)[:,None] & tags[0,:,None],axis=0)
    notes = notes.at[1].set(clean_notes(notes[1],literal[1],tags[1],not_played,own_known,soft[1]))
    def resolve_hand(ns, ls):
        def one(f, current):
            return resolve_prompt(current,f,ls[f],stacks)
        return jax.lax.fori_loop(0,5,one,ns)
    notes = jax.vmap(resolve_hand)(notes,links)
    # Counts seen in the other hand cannot occur in our hidden hand.
    own_counts = jnp.maximum(remaining - hand.sum(0),0)
    own_possible = nonempty_or(notes[0] & (own_counts>0)[None,:],literal[0] & (own_counts>0)[None,:])
    notes = notes.at[0].set(count_consistency(own_possible,own_counts))
    own_prob = probabilities(notes[0],own_counts)
    pplay = (own_prob * playable).sum(-1)
    ptrash = (own_prob * ~useful).sum(-1)
    sureplay = jnp.all(~notes[0] | playable[None,:],axis=-1) & occupied[0]
    suretrash = jnp.all(~notes[0] | ~useful[None,:],axis=-1) & occupied[0]
    # Identify whether partner has a safe play or known trash, using conservative
    # public notes, so that saving their chop can be delayed only when justified.
    partner_safe = jnp.all(~notes[1] | playable[None,:],axis=-1) & occupied[1]
    partner_trash = jnp.all(~notes[1] | ~useful[None,:],axis=-1) & occupied[1]
    partner_ids = jnp.argmax(hand,axis=-1)
    pc = chop(tags[1],occupied[1])
    pcid = partner_ids[jnp.minimum(pc,4)]
    duplicate_pc = jnp.any((partner_ids == pcid) & (jnp.arange(5) != pc) & occupied[1])
    urgent = (pc<5) & useful[pcid] & (critical[pcid] | ((pcid%5 == 1) & ~duplicate_pc) | (playable[pcid] & ~duplicate_pc))
    urgent &= ~jnp.any((partner_safe & playable[partner_ids]) | (partner_trash & ~useful[partner_ids]))
    own_chop = chop(tags[0],occupied[0])
    forced = (tokens == 8) | (own_chop == 5)
    own_exact = notes[0] & (notes[0].sum(-1)==1)[:,None]

    def clue_score(a):
        is_rank = a >= 5
        value = a % 5
        attribute = jnp.where(is_rank,jnp.arange(25)%5,jnp.arange(25)//5) == value
        touches = jnp.any(hand & attribute[None,:],axis=-1)
        focus,ch,new = focus_of(touches,tags[1],occupied[1])
        lit = literal[1] & jnp.where(touches[:,None],attribute[None,:],~attribute[None,:])
        nn,ll,ff,save,ss,promise = interpret(notes[1],lit,tags[1],occupied[1],notes[0],tags[0],
                                 touches,is_rank,value,stacks,disc,soft[1])
        fid = partner_ids[focus]
        new_count = new.sum()
        # All new touches must be useful and nonduplicated, except an urgent save.
        existing = tags[1] | touches
        duplicates = jnp.sum(hand * existing[:,None],axis=0) > 1
        bad = new & (~useful[partner_ids] | duplicates[partner_ids] | own_known[partner_ids])
        bad_count = bad.sum()
        possible_own_duplicate = jnp.any(notes[0] & tags[0,:,None],axis=0)[partner_ids]
        duplicate_risk = jnp.sum(new & possible_own_duplicate)
        # A delayed clue is allowed when all lower cards are already accounted for.
        known_own = jnp.any(own_exact & tags[0,:,None],axis=0).reshape(5,5)
        known_target = jnp.any(hand & tags[1,:,None],axis=0).reshape(5,5)
        covered = known_own | known_target
        needed = (jnp.arange(5) >= stacks[fid//5]) & (jnp.arange(5) < fid%5)
        delayed_ok = jnp.all(~needed | covered[fid//5])
        focus_ok = useful[fid] & ((playable[fid] | delayed_ok) | save[fid])
        # If this relies on a self-prompt, the documented leftmost candidate must
        # be the actual connector. No assumption about a partner's tie-breaking.
        connector_ids = (fid//5)*5+jnp.arange(5)
        ps = ll[fid]
        prompt_ok = jnp.all((ps<0) | (partner_ids[jnp.maximum(ps,0)] == connector_ids))
        retains_true = nn[jnp.arange(5),partner_ids]
        valid_notes = jnp.all(~(tags[1] | touches) | retains_true)
        after_safe = jnp.all(~nn | playable[None,:],axis=-1) & occupied[1]
        actual_play = playable[partner_ids]
        gained_plays = jnp.sum(after_safe & ~partner_safe & actual_play)
        gets_focus = ~tags[1,focus] & focus_ok
        save_chop = (focus == pc) & focus_ok
        permitted_bad = urgent & save_chop
        allowed = focus_ok & prompt_ok & ((bad_count == 0) | permitted_bad)
        allowed &= retains_true[focus] | permitted_bad
        allowed &= (new_count > 0) | forced | (urgent & (gained_plays>0) & ~jnp.any(after_safe & ~actual_play))
        # Do not promise a new copy while an unresolved clued card in our hand
        # may already be that card. Play it first, or wait for public resolution.
        # A necessary chop save may override Good Touch only if no safe own play
        # is available to resolve the collision.
        avoidable_duplicate = (duplicate_risk > 0) & (jnp.any(sureplay) | ~urgent)
        allowed &= ~avoidable_duplicate
        # Prefer immediate productive clues, with broader useful touches adding value.
        false_play = jnp.sum(after_safe & ~actual_play)
        information = jnp.sum(jnp.where(touches,jnp.log(jnp.maximum(notes[1].sum(-1),1)) - jnp.log(jnp.maximum(nn.sum(-1),1)),0.))
        score = 3.0 + 2.2*gained_plays + 0.6*new_count + 0.5*(~is_rank)
        score += .50*information*valid_notes - 8.*false_play
        score += 5.0 * (urgent & (save_chop | (gained_plays>0)))
        score += 0.6 * (save_chop & critical[fid])
        score -= 3.0*bad_count + 1.0*(~valid_notes) + .4*duplicate_risk
        score -= 1.5*(new_count==0)
        score -= 1.0*(gained_plays==0)
        # A forced clue carries its literal information; it is never an invented
        # stall signal. Minimize false play interpretations when no normal clue exists.
        false_play = jnp.sum(after_safe & ~actual_play)
        fallback = -20.0 - 10.0*false_play - 2.0*jnp.sum(~retains_true & touches) - bad_count
        fallback += .1*jnp.sum(tags[1] & touches) + .2*is_rank - 3.*duplicate_risk
        # If the true focus has no proven Level 1 meaning, a receiver may narrow
        # to another play identity depending on what our hidden cards really are.
        unsupported_promise = ~focus_ok & jnp.any(promise & lit[focus] & notes[1,focus])
        fallback -= 15.*unsupported_promise
        rescue = allowed & urgent & (save_chop | ((gained_plays>0) & (false_play==0)))
        return jnp.where(allowed,score,jnp.where(forced,fallback,-100.0)),focus,gained_plays,bad_count,false_play,rescue

    cs,cf,cg,cb,cfp,cr = jax.vmap(clue_score)(jnp.arange(10))
    cs = jnp.where(obs.legal_actions[10:20],cs,-1000.)
    best_clue = jnp.argmax(cs)
    clue_value = cs[best_clue]
    # Playing known cards is normally best; urgent saves take precedence.
    play_value = jnp.where(sureplay,6.0 + .3*(own_prob*(jnp.arange(25)%5==4)).sum(-1) - .03*jnp.arange(5),-100.)
    # Endgame probabilistic plays are permitted tactical risks, not extra signals.
    risk_threshold = jnp.where(lives>=3,.90,jnp.where(lives>=2,.97,1.01))
    risky = (pplay >= risk_threshold) & (deck <= 2) & occupied[0]
    play_or_trash = jnp.all(~notes[0] | (playable | ~useful)[None,:],axis=-1)
    cheap_risk = (lives>=3) & (pplay>=.5) & play_or_trash & occupied[0]
    risky |= cheap_risk
    play_value = jnp.maximum(play_value,jnp.where(risky,4.+pplay,-100.))
    discard_value = jnp.where(suretrash,2.5 + .01*(4-jnp.arange(5)),-100.)
    discard_value = jnp.maximum(discard_value,jnp.where(jnp.arange(5)==own_chop,1.,-100.))
    # Fully clued hands may need a forced discard; minimize public-information loss.
    fallback_discard = ptrash - .3*(own_prob*critical).sum(-1) - .01*jnp.arange(5)
    discard_value = jnp.maximum(discard_value,jnp.where(occupied[0],-5.+fallback_discard,-100.))
    # The last draw leaves exactly two active turns. On the final action a
    # failure with spare lives has zero score cost; losing the final life costs
    # the whole existing stack score. No extra convention is attached to this.
    ids = jnp.arange(25)
    actual_partner_play = playable[partner_ids]
    partner_ready = jnp.any(partner_safe & actual_partner_play)
    next_stacks = stacks[None,:] + jax.nn.one_hot(ids//5,5,dtype=jnp.int32)*playable[:,None]
    next_playable = ids[None,:]%5 == next_stacks[:,ids//5]
    partner_after = jnp.any(jnp.all(~notes[1][None,:,:] | next_playable[:,None,:],axis=-1)
                            & next_playable[:,partner_ids] & occupied[1][None,:],axis=-1)
    partner_ev = (own_prob * jnp.where(playable,partner_after,partner_ready)).sum(-1)
    failure_cost = jnp.where(lives<=1,(1.-pplay)*stacks.sum(),0.)
    last_turn = mem['empty_steps'] >= 1
    end_play = pplay - failure_cost + jnp.where(last_turn,0.,partner_ev)
    end_play = jnp.where((lives<=1) & ~sureplay,-100.,end_play)
    end_discard = jnp.full(5,jnp.where(last_turn,0.,partner_ready.astype(jnp.float32)))
    end_clue = jnp.where((cs > -99.) & ~last_turn,((cg>0) | partner_ready) & (cfp==0),-.01).astype(jnp.float32)
    end_clue += jnp.where(~last_turn,-.002 + .00001*cs,0.)
    play_value = jnp.where(deck==0,end_play,play_value)
    discard_value = jnp.where(deck==0,end_discard,discard_value)
    cs = jnp.where(deck==0,end_clue,cs)
    # Save Principle outranks the other clue principles. An own play can also
    # protect chop when every inferred identity unlocks a known partner play.
    protection_play = sureplay & jnp.all(~notes[0] | partner_after[None,:],axis=-1)
    must_protect = urgent & jnp.any(cr & obs.legal_actions[10:20]) & (deck>0)
    cs = jnp.where(must_protect & ~cr,-1000.,cs)
    play_value = jnp.where(must_protect & ~protection_play,-1000.,play_value)
    discard_value = jnp.where(must_protect,-1000.,discard_value)
    values = jnp.concatenate([discard_value,play_value,cs,jnp.array([0.])])
    values = jnp.where(obs.legal_actions,values,-1e6)
    action = jnp.argmax(values).astype(jnp.int32)
    next_memory = {'notes': notes,'links':links,'clued':tags,'soft':soft,'groups':groups,'early':early,'turn':mem['turn']+1,'empty_steps':jnp.where(deck==0,mem['empty_steps']+1,0)}
    trace = {'pplay':pplay,'ptrash':ptrash,'notes':notes[0],'partner_notes':notes[1],
             'clue_values':cs,'focus':cf,'clue_plays':cg,'bad_touch':cb,'false_play':cfp,'rescue':cr,'urgent':urgent,'chop':own_chop,
             'tokens':tokens,'lives':lives,'deck':deck,'stacks':stacks,'partner_hand':hand,'last':last,'tags':tags,'soft':soft,'empty_steps':mem['empty_steps']}
    return action,next_memory,trace
