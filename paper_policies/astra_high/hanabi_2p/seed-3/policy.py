"""Literal-information Hanabi policy with public-count planning."""
import jax
import jax.numpy as jnp
from custom_primitives import exact_marginals, feasible_support

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'literal_counter', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.array([40, 3, 25, 25, 25, 25, 25, 0, 0, 0, 0, 0, 10000, 10000, 10000, 0, 3], jnp.int32)


def normalize(x):
    return x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), 1e-10)


def agent_act(obs, reset, memory, rng):
    f = obs.features
    legal = obs.legal_actions
    hand = f[:125].reshape(5, 25)
    heights = jnp.sum(f[167:192].reshape(5, 5) * jnp.arange(1, 6), axis=1)
    tokens = jnp.sum(f[192:200])
    lives = jnp.sum(f[200:203])
    deck = jnp.sum(f[127:167])
    memory = jnp.where(reset, init_agent(), memory)
    turns_left = jnp.where(deck>0,3,jnp.where(memory[0]>0,2,jnp.maximum(memory[1]-1,0)))
    next_memory = memory.at[:2].set(jnp.array([deck, turns_left],jnp.int32))
    ds = f[203:253].reshape(5,10)
    discards = jnp.stack([ds[:,:3].sum(-1),ds[:,3:5].sum(-1),ds[:,5:7].sum(-1),ds[:,7:9].sum(-1),ds[:,9]],-1)
    ranks = jnp.tile(jnp.arange(5),5)
    colors = jnp.repeat(jnp.arange(5),5)
    total = jnp.tile(jnp.array([3,2,2,2,1]),(5,1))
    unplayed = total - discards
    counts = (unplayed - (jnp.arange(5)[None,:] < heights[:,None])).reshape(25)
    possible = f[308:658].reshape(2,5,35)
    own_mask = possible[0,:,:25] > 0
    next_memory = next_memory.at[2:7].set(own_mask.sum(-1))
    partner_mask = possible[1,:,:25] > 0
    local_counts = jnp.maximum(counts-hand.sum(0),0)
    own_support = feasible_support(own_mask, local_counts)
    own = exact_marginals(own_mask, local_counts)
    occupied = jnp.sum(possible[0,:,:25],-1)>0
    playable = ranks == heights[colors]
    blocked = jnp.cumsum((unplayed == 0) & (jnp.arange(5)[None,:] >= heights[:,None]),axis=1)>0
    dead = (ranks < heights[colors]) | blocked.reshape(25)
    future = (~dead) & (~playable)
    cap = jnp.min(jnp.where((unplayed==0)&(jnp.arange(5)[None,:]>=heights[:,None]),jnp.arange(5)[None,:],5),axis=-1)
    reachable_points = jnp.maximum(cap[colors]-ranks,0)
    backup_discount = jnp.where(hand.sum(0)>0,.55,1.)
    scarce = jnp.where(dead,0.,reachable_points/jnp.maximum(unplayed.reshape(25),1)**1.2)*backup_discount
    critical = (~dead) & (unplayed.reshape(25)==1)
    pplay = own @ playable
    pdead = own @ dead
    loss = own @ scarce
    pc = own @ critical
    hint_flags = possible[0,:,25:].sum(-1)>0
    # Favor moves that immediately unlock cards already held by either seat.
    successor = (colors[:,None]==colors[None,:]) & (ranks[None,:]==ranks[:,None]+1)
    own_ready = jnp.sum(own*jnp.max(own,axis=-1,keepdims=True),axis=0)
    partner_ready = jnp.sum(hand*(possible[1,:,25:].sum(-1)>0)[:,None],axis=0)
    unlock = successor @ (.45*own_ready + .35*partner_ready + .10*hand.sum(0))
    play_utility = 1. + unlock + .15*(ranks==4)
    safe_slots = occupied & jnp.all(~own_support | playable[None,:],axis=-1)
    partner_literal_safe = jnp.all(~partner_mask | playable[None,:],axis=-1)&(hand.sum(-1)>0)
    shared_ready = (hand*partner_literal_safe[:,None]).sum(0)
    play_value = pplay + 2*safe_slots + .08*(own@play_utility) - .10*(own@shared_ready) - .001*jnp.arange(5)
    play_slot = jnp.argmax(jnp.where(occupied,play_value,-100.))
    play_slot = jnp.where(jnp.any(safe_slots),play_slot,jnp.argmax(jnp.where(occupied,pplay,-1.)))
    safe_play = safe_slots[play_slot]

    # The partner sees our cards. Approximate that subtraction with our posterior.
    other_counts = jnp.maximum(counts-own.sum(0),0.01)
    partner_before = normalize(partner_mask*other_counts)
    attributes = jnp.concatenate([jnp.arange(5)[:,None]==colors[None,:],jnp.arange(5)[:,None]==ranks[None,:]],0)
    matches = jnp.einsum('st,at->as',hand,attributes)>0
    hint_mask = jnp.where(matches[:,:,None],attributes[:,None,:],~attributes[:,None,:])
    partner_after = normalize(partner_mask[None,:,:]*hint_mask*other_counts)
    before_play = partner_before @ playable
    last_action = f[253:308]
    previous_confidence = jnp.dot(memory[7:12],last_action[23:28])/10000.
    removed_slot = jnp.argmax(last_action[23:28])
    followed_hint = (memory[15] & (1 << removed_slot))>0
    learn_response = (last_action[1]>0)&(last_action[2]>0)&followed_hint
    life_index = 11+jnp.clip(memory[16],1,3)
    observed_min = jnp.minimum(memory[life_index],(previous_confidence*10000).astype(jnp.int32))
    next_memory = next_memory.at[life_index].set(jnp.where(learn_response,observed_min,memory[life_index]))
    assertive = next_memory[11+lives.astype(jnp.int32)]
    hint_bits = jnp.dot(last_action[18:23],1<<jnp.arange(5)).astype(jnp.int32)
    self_hint = (last_action[0]>0)&((last_action[4]+last_action[5])>0)
    next_memory = next_memory.at[7:12].set((before_play*10000).astype(jnp.int32)).at[15].set(jnp.where(self_hint,hint_bits,0)).at[16].set(lives.astype(jnp.int32))
    after_play = partner_after @ playable
    before_dead = partner_before @ dead
    after_dead = partner_after @ dead
    actual_play = hand @ playable
    card_utility = hand @ play_utility
    actual_dead = hand @ dead
    actual_critical = hand @ critical
    actual_soon = hand @ ((ranks == heights[colors]+1) & (~dead))
    relevance = actual_play + .25*actual_soon + .15*actual_critical
    # Value literal new certainty most; partial clues can prepare a second clue.
    new_safe = (after_play>.9999) & (before_play[None,:]<.9999)
    progress = jnp.maximum(after_play-before_play,0)
    info = jnp.sum(partner_before[None,:,:]*jnp.log(jnp.maximum(partner_before[None,:,:],1e-8)),axis=-1)-jnp.sum(partner_after*jnp.log(jnp.maximum(partner_after,1e-8)),axis=-1)
    certainty = jnp.max(partner_after,-1)-jnp.max(partner_before,-1)
    hint_values = jnp.sum(2.5*new_safe*actual_play*card_utility + 1.0*progress*actual_play*card_utility + .15*(after_dead>.9999)*(before_dead[None,:]<.9999)*actual_dead + .22*certainty*relevance + -.22*info*relevance + .10*matches*actual_critical*(possible[1,:,25:].sum(-1)==0),axis=-1)
    new_touch = matches & (jnp.sum(partner_mask[None,:,:]*hint_mask,axis=-1)<jnp.sum(partner_mask,axis=-1))
    predicted_response = new_touch*(new_touch.sum(-1,keepdims=True)==1)*(after_play>assertive/10000.+.03)*(~new_safe)*actual_play
    predicted_response *= (assertive<9000)&(jnp.max(before_play)<.9999)
    hint_values += 1.8*jnp.sum(predicted_response*card_utility,axis=-1)
    # A save is only useful if the resulting alternative discard is cheaper.
    partner_hinted = possible[1,:,25:].sum(-1)>0
    other_scarce = jnp.where(dead,0.,reachable_points/jnp.maximum(unplayed.reshape(25),1)**1.2)*(1-.45*jnp.minimum(own.sum(0),1))
    discard_before = 2*before_dead-.55*(partner_before@other_scarce)-.8*(partner_before@critical)-.35*partner_hinted-before_play-.015*jnp.arange(5)
    discard_after = 2*after_dead-2*matches*(1-after_dead)-.55*(partner_after@other_scarce)-.8*(partner_after@critical)-.35*(partner_hinted[None,:]|matches)-after_play-.015*jnp.arange(5)
    likely_discard = jnp.argmax(discard_before)
    alternatives = jnp.argmax(discard_after,axis=-1)
    card_loss_value = critical*reachable_points + (~critical)*(.50*other_scarce+.50*playable)
    card_loss_value = jnp.where((ranks==4)&critical,card_loss_value*.85**jnp.maximum(4-heights[colors],0),card_loss_value)
    actual_loss = hand@card_loss_value
    after_loss = actual_loss[alternatives]*(jnp.max(after_play*actual_play,axis=-1)<.9999)
    endangered_five = (hand[likely_discard]@((ranks==4)&(~dead)))>0
    endangered_critical = (hand[likely_discard]@critical)>0
    safe_save_hint = jnp.sum(new_touch*(1-actual_play)*(after_play>.25),axis=-1)==0
    protect_info = new_touch[:,likely_discard] | jnp.any((after_dead>.9999)&(before_dead[None,:]<.9999),axis=-1)
    save_advantages = jnp.maximum(actual_loss[likely_discard]-after_loss,0.)*endangered_critical*safe_save_hint*(jnp.max(before_play)<.9999)*protect_info
    save_weight = jnp.where(endangered_five,2.0,.85)
    hint_values += save_weight*save_advantages
    save_advantage = jnp.max(save_advantages)
    ambiguous_bad = new_touch*(1-actual_play)*(after_play>jnp.maximum(.25,assertive/10000.-.05))*(new_touch.sum(-1,keepdims=True)==1)
    hint_values -= jnp.where(assertive<9000,1.3,0.)*jnp.sum(ambiguous_bad,axis=-1)
    hint_values -= .02*jnp.sum(matches*(1-actual_play)*(1-actual_dead),-1)
    last_life_danger = jnp.sum(new_touch*(1-actual_play)*(after_play>.25),axis=-1)
    clue_safety_penalty = jnp.where(lives==1,2.5+.25*heights.sum(),0.)*last_life_danger
    hint_values -= clue_safety_penalty
    hint_values = jnp.where(legal[10:20],hint_values,-100.)
    hint_slot = jnp.argmax(hint_values)
    best_hint = hint_values[hint_slot]
    # Prefer expendable cards, while preserving information and last copies.
    last = f[253:308]
    fresh = last[6] * (last[4]+last[5]) * last[18:23]
    discard_value = 2*pdead - 2.0*fresh*(1-pdead) - .55*loss - .8*pc - .35*hint_flags - 1.0*pplay - .015*jnp.arange(5)
    discard_slot = jnp.argmax(jnp.where(legal[:5],discard_value,-100.))
    known_junk = jnp.all(~own_support[discard_slot] | dead)
    choose_hint = (best_hint > .65) | (tokens>=8) | ((best_hint>.30)&(tokens>=3)&(~known_junk))
    action = jnp.where(choose_hint,10+hint_slot,discard_slot)
    action = jnp.where(safe_play,5+play_slot,action)
    after_support = partner_mask[None,:,:] & hint_mask
    clear_critical = jnp.all(~after_support[:,likely_discard,:] | critical[None,:],axis=-1)
    clear_play = jnp.any(jnp.all(~after_support | playable[None,None,:],axis=-1)&(actual_play[None,:]>0),axis=-1)
    urgent_options = legal[10:20] & (clear_critical | clear_play) & ((lives>1)|(last_life_danger==0))
    urgent_hint = jnp.argmax(jnp.where(urgent_options,save_advantages,-1.))
    urgent_save = safe_play & urgent_options[urgent_hint] & (tokens>=2) & (deck>2) & (save_advantages[urgent_hint]>1.0)
    action = jnp.where(urgent_save,10+urgent_hint,action)
    # Use spare lives for high-probability endgame opportunities.
    risk_threshold = jnp.where(lives>=3,.82,jnp.where(lives>=2,.93,1.01))
    risk = (deck<=5)&(pplay[play_slot]>risk_threshold)&(~safe_play)&(best_hint<.65)
    action = jnp.where(risk,5+play_slot,action)
    novel = (fresh>0)&(own_mask.sum(-1)<memory[2:7])
    focus = jnp.argmax(novel)
    future_loss = own[focus] @ (future*critical*reachable_points)
    cautious_threshold = jnp.where(future_loss<.10,.30,.45)
    cautious_threshold = jnp.where(jnp.all(~own_support[focus] | playable | dead),.25,cautious_threshold)
    focus_threshold = jnp.maximum(.25,jnp.minimum(cautious_threshold,assertive/10000.+.05))
    focused_play = (novel.sum()==1)&(pplay[focus]>focus_threshold)&(lives==3)&(~safe_play)
    action = jnp.where(focused_play,5+focus,action)
    # Before the last draw, reserve our certain point and teach a distinct
    # certain point to the partner rather than leaving its card unidentified.
    literal_after = jnp.all(~(partner_mask[None,:,:]&hint_mask) | playable[None,None,:],axis=-1)&(hand.sum(-1)>0)[None,:]
    partner_play_types = jnp.einsum('as,st->at',literal_after.astype(jnp.float32),hand)>0
    blocked_own = jnp.any(own_support[None,:,:]&partner_play_types[:,None,:],axis=-1)
    keep_own_point = jnp.any(safe_slots[None,:]&(~blocked_own),axis=-1)
    handoff_options = legal[10:20]&jnp.any(literal_after,axis=-1)&keep_own_point&((lives>1)|(last_life_danger==0))
    handoff_hint = jnp.argmax(jnp.where(handoff_options,hint_values,-1000.))
    own_followup = own[play_slot] @ (successor @ own_ready)
    handoff = (deck>0)&(deck<=2)&(safe_slots.sum()==1)&(own_followup<.10)&(jnp.max(before_play)<.95)&jnp.any(handoff_options)
    action = jnp.where(handoff,10+handoff_hint,action)
    # On our final turn, information and future preservation have no value.
    score = heights.sum()
    last_threshold = jnp.where(lives>1,0.,score/(score+1))
    last_play = (deck==0)&(turns_left==1)&(pplay[play_slot]>last_threshold)
    action = jnp.where(last_play,5+play_slot,action)
    # With one turn for each seat, compare our chance with the partner's
    # immediate opportunity; leave one life as insurance for its final action.
    other_ready = jnp.max(before_play*actual_play)
    after_ready = jnp.max(after_play*actual_play,axis=-1)
    clue_final = jnp.argmax(jnp.where(legal[10:20],after_ready-clue_safety_penalty,-100.))
    hint_gain = after_ready[clue_final]-other_ready
    end_threshold = jnp.where(lives>=3,.25,jnp.where(lives==2,.60,score/(score+1)))
    penult_play = (deck==0)&(turns_left==2)&(pplay[play_slot]>end_threshold)&(pplay[play_slot]>hint_gain)
    action = jnp.where(penult_play,5+play_slot,action)
    action = jnp.where((deck==0)&(turns_left==2)&(~penult_play)&(~safe_play)&legal[10+clue_final],10+clue_final,action)
    action = jnp.where(legal[20],20,action)
    action = jnp.where(legal[action],action,jnp.argmax(legal))
    partner_ids = jnp.where(hand.sum(-1)>0,jnp.argmax(hand,axis=-1),-1)
    discard_codes = (discards*(4**jnp.arange(5))[None,:]).sum(-1)
    state = jnp.concatenate([partner_ids,heights,jnp.array([deck,tokens,lives]),discard_codes,jnp.array([reset,assertive])]).astype(jnp.int32)
    mask_bits = jnp.sum(own_mask*(1<<jnp.arange(25))[None,:],axis=-1).astype(jnp.int32)
    positive = possible[0,:,25:30]@jnp.arange(1,6)*6 + possible[0,:,30:35]@jnp.arange(1,6)
    return action.astype(jnp.int32),next_memory,{'pplay':pplay,'pdead':pdead,'hint_values':hint_values,'discard_value':discard_value,'state':state,'mask_bits':mask_bits,'positive':positive.astype(jnp.int32),'safe':safe_slots,'save':save_advantage,'handoff':handoff}
