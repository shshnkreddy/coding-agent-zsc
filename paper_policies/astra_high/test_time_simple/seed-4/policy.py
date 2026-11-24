"""Spatial recipe communication with an observation-driven kitchen controller."""
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'geometry_anchored_adaptive_policy', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def decoder_models():
    # Each row predicts a decoder's recipe choice from the 3-bit message.
    # -1 means the decoder does not recognize that pattern.
    patterns = list(range(8))
    spatial = [0 if (p&1) and not (p&4) else 1 if (p&4) and not (p&1) else -1 for p in patterns]
    rows = [spatial, [1-x if x >= 0 else -1 for x in spatial]]
    weights = [8.,2.]
    for bit in (1,2,4):
        for invert in (0,1):
            rows.append([int(bool(p&bit)) ^ invert for p in patterns])
            weights.append(.5)
    preferred = {(1,4):2.,(4,1):.5,(2,0):.5,(0,2):.25,
                 (2,4):.4,(4,2):.2,(1,2):.4,(2,1):.2}
    codes = [1 if p&1 else 2 if p&2 else 4 if p&4 else 0 for p in patterns]
    for p0 in (0,1,2,4):
        for p1 in (0,1,2,4):
            if p0 != p1:
                rows.append([0 if c == p0 else 1 if c == p1 else -1 for c in codes])
                weights.append(preferred.get((p0,p1),.1))
    for p0 in patterns:
        for p1 in patterns:
            if p0 != p1:
                rows.append([0 if p == p0 else 1 if p == p1 else -1 for p in patterns])
                weights.append(.5*preferred.get((p0,p1),.01))
    weights = jnp.asarray(weights,jnp.float32)
    return jnp.asarray(rows,jnp.int32),weights/jnp.sum(weights)


def init_agent_0():
    return {
        'counts': jnp.zeros((8,2),jnp.int32),
        'decoder_belief': decoder_models()[1],
        'timeouts': jnp.zeros((8,),jnp.int32),
        'previous_count': jnp.asarray(0, jnp.int32),
        'previous_signal': jnp.asarray(-1, jnp.int32),
        'stable': jnp.asarray(0, jnp.int32),
        'idle': jnp.asarray(0, jnp.int32),
    }


def encoding_prior():
    # Encodings specify a three-bit counter occupancy pattern for each recipe.
    pair = jnp.full((8, 8), 0.0001, jnp.float32)
    for p0, p1, weight in ((1,4,.60),(4,1,.10),(3,4,.60),(4,3,.10),(2,0,.08),(0,2,.04),
                           (1,0,.04),(0,1,.02),(2,4,.04),(1,2,.04),
                           (4,2,.02),(2,1,.02),(0,4,.02),(4,0,.02),
                           (0,0,.10)):
        pair = pair.at[p0,p1].set(weight)
    belief = jnp.stack((pair, pair), axis=-1)
    return belief / jnp.sum(belief)


def init_agent_1():
    return {'clock': jnp.asarray(0,jnp.int32),
            'previous_dish': jnp.asarray(False),
            'was_delivering': jnp.asarray(False),
            'buffer_row': jnp.asarray(0,jnp.int32),
            'age': jnp.asarray(100,jnp.int32),
            'recipe': jnp.asarray(-1,jnp.int32),
            'last_type': jnp.asarray(1,jnp.int32),
            'pattern': jnp.asarray(-1,jnp.int32),
            'stable': jnp.asarray(0,jnp.int32),
            'accepted': jnp.asarray(-1,jnp.int32),
            'corrected': jnp.asarray(False),
            'previous_count': jnp.asarray(0,jnp.int32),
            'belief': encoding_prior()}


def position(obs, seat):
    channel = 23 if seat == 0 else 22
    idx = jnp.argmax(obs[:, :, channel])
    row = 4 - idx // 5
    col = (2 if seat == 0 else 6) - idx % 5
    return row, col


def cell(obs, row, col, rr, cc):
    i, j = rr - row + 2, cc - col + 2
    visible = (i >= 0) & (i < 5) & (j >= 0) & (j < 5)
    return jnp.where(visible, obs[jnp.clip(i, 0, 4), jnp.clip(j, 0, 4)], 0)


def interact_at(row, col, facing, tr, tc, direction):
    # All reachable floor in either room is a two-column rectangle.
    orient = jnp.asarray([3, 1, 0, 2], jnp.int32)[direction]
    return jnp.where(row < tr, 1, jnp.where(row > tr, 3,
           jnp.where(col < tc, 0, jnp.where(col > tc, 2,
           jnp.where(facing != direction, orient, 5)))))


def go_to(row, col, tr, tc):
    return jnp.where(col < tc, 0, jnp.where(col > tc, 2,
           jnp.where(row < tr, 1, jnp.where(row > tr, 3, 4))))


def agent_0_act(obs, reset, memory, rng):
    memory = {k: jnp.where(reset, init_agent_0()[k], v) for k, v in memory.items()}
    row, col = position(obs, 0)
    facing = jnp.argmax(obs[2, 2, 1:5])
    recipe = jnp.argmax(jnp.sum(obs[:, :, 36:39], axis=(0, 1)))
    markers = jnp.asarray([jnp.sum(cell(obs, row, col, rr, 3)[29:34]) > 0 for rr in (1, 2, 3)])
    signal = jnp.sum(markers.astype(jnp.int32) * jnp.asarray([1,2,4],jnp.int32))
    stable = jnp.where(signal == memory['previous_signal'], memory['stable'] + 1, 0)
    pot = cell(obs, row, col, 2, 4)
    count = jnp.sum(pot[31:34])
    visible = col == 2
    started = visible & (count > 0) & (memory['previous_count'] == 0)
    actual = jnp.argmax(pot[31:34])
    learned = started & (stable >= 6) & (actual < 2)
    counts = memory['counts'].at[signal,jnp.minimum(actual,1)].add(learned.astype(jnp.int32))
    idle = jnp.where(visible & (count == 0), memory['idle'] + 1, 0)
    stalled = (idle > 35) & (stable > 30)
    timeouts = memory['timeouts'].at[signal].set(jnp.where(stalled,1,memory['timeouts'][signal]))
    models,decoder_prior = decoder_models()
    response = models[:,signal]
    likelihood = jnp.where(response == actual,.98,jnp.where(response < 0,.35,.02))
    predict_decoder = jnp.where(learned,.85*memory['decoder_belief']+.15*decoder_prior,memory['decoder_belief'])
    posterior = predict_decoder * jnp.where(learned,likelihood,1.)
    timeout_event = stalled & (memory['timeouts'][signal] == 0)
    posterior *= jnp.where(timeout_event,jnp.where(response < 0,.98,.02),1.)
    posterior /= jnp.maximum(jnp.sum(posterior),1e-30)
    utility = jnp.where(models == recipe,1.,jnp.where(models < 0,-.25,-1.))
    score = jnp.sum(posterior[:,None]*utility,axis=0)
    patterns = jnp.arange(8)
    token_count = ((patterns&1)>0).astype(jnp.float32) + ((patterns&2)>0) + ((patterns&4)>0)
    score -= .012*token_count
    score += .012*(patterns == signal) + .008*(patterns == jnp.where(recipe == 0,1,4))
    target = jnp.argmax(score)
    # Finish a batch consistently before correcting the message; no additional
    # raw ingredients can enter after the pot reaches capacity.
    target = jnp.where((count > 0) & ((count < 3) | (actual == recipe)),signal,target)
    desired = (target & jnp.asarray([1,2,4],jnp.int32)) > 0
    missing = desired & ~markers
    unwanted = markers & ~desired
    holding = jnp.sum(obs[2,2,5:10]) > 0
    # Establish the spatial component before adding redundancy. Remove
    # obsolete spatial markers first to avoid briefly signaling both supplies.
    put_utility = -jnp.abs(jnp.arange(1,4)-row).astype(jnp.float32)
    take_utility = put_utility
    put_row = jnp.argmax(jnp.where(missing,put_utility,-10.)) + 1
    take_row = jnp.argmax(jnp.where(unwanted,take_utility,-10.)) + 1
    put = interact_at(row,col,facing,put_row,2,2)
    remove = interact_at(row,col,facing,take_row,2,2)
    supply_row = jnp.where(row > 2,3,1)
    get = interact_at(row,col,facing,supply_row,2,jnp.where(supply_row == 1,0,1))
    # Spare tokens stay on the sender's own counters and are never messages.
    storage_top = jnp.sum(cell(obs,row,col,1,0)[29:34]) > 0
    storage_bottom = jnp.sum(cell(obs,row,col,3,0)[29:34]) > 0
    store_row = jnp.where(~storage_top,1,3)
    stash = interact_at(row,col,facing,store_row,1,3)
    recover_row = jnp.where(storage_top,1,3)
    recover = interact_at(row,col,facing,recover_row,1,3)
    get = jnp.where(storage_top | storage_bottom,recover,get)
    action = jnp.where(holding,
             jnp.where(jnp.any(missing),put,jnp.where(jnp.any(unwanted),stash,4)),
             jnp.where(jnp.any(unwanted),remove,jnp.where(jnp.any(missing),get,4)))
    # Return to column 2 to retain the observable pot as feedback.
    action = jnp.where((action == 4) & (col != 2),go_to(row,col,row,2),action)
    next_memory = {'counts': counts, 'timeouts': timeouts, 'decoder_belief':posterior,
                   'previous_count': jnp.where(visible, count, memory['previous_count']),
                   'previous_signal': signal, 'stable': stable, 'idle': idle}
    return action.astype(jnp.int32), next_memory, {'recipe': recipe, 'row': row, 'col': col,
           'signal': signal, 'target': target, 'counts': counts, 'learned': learned,
           'stable':stable,'holding':holding,'pot_count':count,'actual':actual}


def condition_on_pattern(belief, pattern):
    p0 = jnp.arange(8)[:,None]
    p1 = jnp.arange(8)[None,:]
    likelihood = jnp.stack((jnp.broadcast_to(p0 == pattern,(8,8)),
                            jnp.broadcast_to(p1 == pattern,(8,8))),axis=-1)
    posterior = belief * likelihood
    fallback = encoding_prior() * likelihood
    posterior = jnp.where(jnp.sum(posterior) > 1e-12, posterior, fallback)
    return posterior / jnp.maximum(jnp.sum(posterior),1e-12)


def agent_1_act(obs, reset, memory, rng):
    initial = init_agent_1()
    memory = {k: jnp.where(reset, initial[k], v) for k,v in memory.items()}
    row, col = position(obs, 1)
    facing = jnp.argmax(obs[2, 2, 1:5])
    inv = obs[2, 2, 5:10]
    dish = inv[1] > 0
    plate = (inv[0] > 0) & ~dish
    raw = (jnp.sum(inv[2:]) > 0) & ~dish
    finished_carry = memory['previous_dish'] & ~dish
    delivered = finished_carry & memory['was_delivering']
    buffered = finished_carry & ~memory['was_delivering']
    age = jnp.where(delivered, 0, memory['age'] + 1)
    last_type = jnp.where(dish, jnp.argmax(inv[2:4]), memory['last_type'])
    pot = cell(obs, row, col, 2, 4)
    count = jnp.sum(pot[31:34])
    ready = pot[30] > 0
    cooking = pot[39] > 0
    flags = jnp.asarray([jnp.sum(cell(obs,row,col,rr,3)[29:34]) > 0 for rr in (1,2,3)])
    pattern = jnp.sum(flags.astype(jnp.int32) * jnp.asarray([1,2,4],jnp.int32))
    pattern = jnp.where(col == 5, pattern, memory['pattern'])
    stable = jnp.where(col == 5, jnp.where(pattern == memory['pattern'],memory['stable'] + 1,0),0)
    recipe = jnp.where(finished_carry,-1,memory['recipe'])
    belief = memory['belief']
    b0,b1 = belief[:,:,0],belief[:,:,1]
    predicted0 = jnp.where(last_type == 0, .5*b0, b0+.5*b1)
    predicted1 = jnp.where(last_type == 0, b1+.5*b0, .5*b1)
    belief = jnp.where(delivered, jnp.stack((predicted0,predicted1),axis=-1),belief)
    # Delay absence-as-message slightly longer than a positive marker to avoid
    # reading the empty counter during a move from one position to another.
    wait_steps = jnp.where(pattern == 0,12,8)
    can_read = (col == 5) & (stable >= 2) & (age >= wait_steps)
    can_read &= ((pattern != 0) & (pattern != 2)) | (memory['clock'] >= 12)
    decide = (recipe < 0) & can_read & ~delivered
    posterior = condition_on_pattern(belief, pattern)
    belief = jnp.where(decide,posterior,belief)
    chosen = jnp.argmax(jnp.sum(belief,axis=(0,1)))
    recipe = jnp.where(decide,chosen,recipe)
    accepted = jnp.where(decide,pattern,memory['accepted'])
    corrected = jnp.where(finished_carry,False,memory['corrected'])
    first_load = (count > 0) & (memory['previous_count'] == 0)
    stored_pickup = dish & ~memory['previous_dish'] & (memory['previous_count'] == 0)
    completion = (first_load | stored_pickup) & (pattern != accepted) & (pattern >= 0)
    # The recipe cannot change while the sender finishes constructing a message.
    # Preserve the inferred recipe probabilities while re-indexing the message.
    state_probability = jnp.sum(belief,axis=(0,1))
    completion_belief = condition_on_pattern(encoding_prior(),pattern)
    per_state = jnp.sum(completion_belief,axis=(0,1))
    completion_belief *= (state_probability/jnp.maximum(per_state,1e-12))[None,None,:]
    belief = jnp.where(completion,completion_belief,belief)
    accepted = jnp.where(completion,pattern,accepted)
    # A stable changed message before delivery cannot be recipe resampling.
    # It is evidence that the partner is correcting a misunderstanding.
    correction = (recipe >= 0) & (count > 0) & (age > 18) & (stable >= 6)
    correction &= (pattern > 0) & (pattern != accepted) & ~corrected & ~delivered
    wrong_type = jnp.argmax(pot[31:33])
    corrected_belief = condition_on_pattern(encoding_prior(),pattern)
    state_weight = jnp.where(jnp.arange(2) == wrong_type,.1,.9)
    corrected_belief = corrected_belief * state_weight[None,None,:]
    corrected_belief /= jnp.maximum(jnp.sum(corrected_belief),1e-12)
    belief = jnp.where(correction,corrected_belief,belief)
    accepted = jnp.where(correction,pattern,accepted)
    corrected |= correction
    ingredient = jnp.where(count > 0,jnp.argmax(pot[31:33]),recipe)
    target_row = jnp.where(ingredient == 0,1,3)
    get_raw = interact_at(row,col,facing,target_row,6,2)
    use_pot = interact_at(row,col,facing,2,5,3)
    get_plate = interact_at(row,col,facing,jnp.where(row <= 2,1,3),5,3)
    deliver = interact_at(row,col,facing,2,6,2)
    wait = go_to(row,col,2,5)
    top_food = cell(obs,row,col,0,5)
    bottom_food = cell(obs,row,col,4,5)
    top_empty = jnp.sum(top_food[29:34]) == 0
    bottom_empty = jnp.sum(bottom_food[29:34]) == 0
    probability_match = jnp.sum(belief[:,:,jnp.argmax(inv[2:4])])
    new_dish = dish & ~memory['previous_dish']
    from_pot = memory['previous_count'] > 0
    should_buffer = corrected & (probability_match < .35) & from_pot
    new_buffer_row = jnp.where(should_buffer,jnp.where(top_empty,1,jnp.where(bottom_empty,3,0)),0)
    buffer_row = jnp.where(new_dish,new_buffer_row,memory['buffer_row'])
    buffer_row = jnp.where(dish,buffer_row,0)
    put_food = interact_at(row,col,facing,buffer_row,5,jnp.where(buffer_row == 1,0,1))
    dish_action = jnp.where(buffer_row > 0,put_food,deliver)
    top_matches = (top_food[30] > 0) & (top_food[31+jnp.maximum(recipe,0)] == 3)
    bottom_matches = (bottom_food[30] > 0) & (bottom_food[31+jnp.maximum(recipe,0)] == 3)
    fetch_row = jnp.where(top_matches,1,3)
    fetch_food = interact_at(row,col,facing,fetch_row,5,jnp.where(fetch_row == 1,0,1))
    use_saved = (recipe >= 0) & (count == 0) & (top_matches | bottom_matches)
    empty_action = jnp.where(cooking | ready,get_plate,
                   jnp.where(use_saved,fetch_food,jnp.where(recipe >= 0,get_raw,wait)))
    action = jnp.where(dish,dish_action,jnp.where(raw,use_pot,jnp.where(plate,use_pot,empty_action)))
    was_delivering = dish & (buffer_row == 0) & (action == 5) & (row == 2) & (col == 6) & (facing == 2)
    next_memory = {'clock': memory['clock']+1,'previous_dish':dish,'age':age,
                   'was_delivering':was_delivering,'buffer_row':buffer_row,
                   'recipe':recipe,'last_type':last_type,'pattern':pattern,'stable':stable,
                   'accepted':accepted,'corrected':corrected,'belief':belief,'previous_count':count}
    return action.astype(jnp.int32),next_memory,{'recipe':recipe,'pattern':pattern,
           'row':row,'col':col,'pot_count':count,'age':age,'p0':jnp.sum(belief[:,:,0]),
           'decision':decide,'correction':correction,'buffer_row':buffer_row,
           'buffered':buffered,'delivered':delivered,'use_saved':use_saved}
