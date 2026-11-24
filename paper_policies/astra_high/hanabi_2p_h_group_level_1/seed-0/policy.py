"""Two-player H-group Level 1: public card notes and conservative tactics."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'hgroup_level1', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {'notes': jnp.ones((2, 5, 25), jnp.bool_),
            'exceptions': jnp.zeros((2, 5, 25), jnp.bool_),
            'required': jnp.zeros((2, 5, 25, 5), jnp.bool_),
            'support': jnp.zeros((2, 5, 25, 5, 2, 5), jnp.bool_),
            'touched': jnp.zeros((2, 5), jnp.bool_),
            'occupied': jnp.ones((2, 5), jnp.bool_),
            'early': jnp.asarray(True), 'turn': jnp.int32(0),
            'deck': jnp.int32(40)}


def nonempty_or(a, b):
    return jnp.where(jnp.any(a, axis=-1, keepdims=True), a, b)


def chop(touched, occupied):
    return jnp.min(jnp.where(occupied & ~touched, jnp.arange(5), 5))


def focus_of(matches, touched, occupied):
    new = matches & ~touched
    cp = chop(touched, occupied)
    focus = jnp.where(jnp.any(new),
                      jnp.where(jnp.any(matches & (jnp.arange(5) == cp)), cp,
                                jnp.max(jnp.where(new, jnp.arange(5), -1))),
                      jnp.max(jnp.where(matches, jnp.arange(5), -1)))
    return jnp.maximum(focus, 0), cp, new


def normalize(x):
    return x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), 1e-9)


def count_consistent_belief(notes, counts):
    """Exact count weighting of the bounded-support subhand.

    Positive hints leave at most five identities per card. Broader unhinted
    cards retain marginal count weights; their negative hints are not treated
    as a full joint posterior here. Every enumerated assignment respects the
    remaining physical copies after subtracting the visible partner hand.
    """
    size = jnp.sum(notes, axis=-1)
    bounded = (size > 0) & (size <= 5)
    options = jnp.sort(jnp.where(notes, jnp.arange(25)[None, :], 25), axis=-1)[:, :5]
    options = jnp.where(bounded[:, None], options, 25)
    option_valid = jnp.where(bounded[:, None], options < 25, jnp.arange(5)[None, :] == 0)
    choices = (jnp.arange(3125)[:, None] // (5 ** jnp.arange(5))[None, :]) % 5
    selected = options[jnp.arange(5)[None, :], choices]
    valid = jnp.all(option_valid[jnp.arange(5)[None, :], choices], axis=-1)
    repeated = jnp.sum((selected[:, :, None] == selected[:, None, :]) &
                       (jnp.arange(5)[None, :] < jnp.arange(5)[:, None])[None, :, :], axis=-1)
    padded_counts = jnp.concatenate([counts, jnp.ones(1, counts.dtype)])
    factors = jnp.where(bounded[None, :], jnp.maximum(padded_counts[selected] - repeated, 0.), 1.)
    weight = jnp.prod(factors, axis=-1) * valid
    total = jnp.sum(weight)
    weight = weight / jnp.maximum(total, 1e-9)
    marginals = jax.vmap(lambda x: jnp.bincount(x, weights=weight, length=26)[:25])(selected.T)
    fallback = normalize(notes * counts[None, :])
    return jnp.where(bounded[:, None] & (total > 0), marginals, fallback), total > 0


def prune(notes, literal, touched, useful, other_reserved, exceptions=None):
    protected = jnp.zeros_like(notes) if exceptions is None else (exceptions & literal)
    b = nonempty_or(notes & literal, literal) | protected
    good = b & useful[None, :] & ~other_reserved[None, :]
    b = jnp.where(touched[:, None], nonempty_or(good, b), b) | protected
    # Good Touch: a uniquely identified clued card excludes its duplicate
    # from another ambiguous clued card. Keep singleton trash recognisable.
    def one_pass(_, x):
        single = (jnp.sum(x, axis=-1) == 1) & touched
        reserved = x & single[:, None]
        others = (jnp.sum(reserved, axis=0)[None, :] - reserved) > 0
        reduced = x & ~others
        return jnp.where(touched[:, None], nonempty_or(reduced, x), x) | protected
    return jax.lax.fori_loop(0, 3, one_pass, b)


def clue_exceptions(old, literal, touched, occupied, matches, useful, reserved, forced):
    # Save Principle can override Good Touch. Collateral cards can therefore
    # be old trash; keep that alternative rather than converting it into a
    # future playable identity. The clue focus still has its Level 1 meaning.
    focus, cp, new = focus_of(matches, touched, occupied)
    collateral = new & (jnp.arange(5) != focus) & ((focus == cp) | forced)
    # Save Principle also permits collateral duplicates within this hand.
    prior_types = jnp.any(literal & touched[:, None], axis=0)
    other_collateral = jnp.any(literal[None, :, :] & collateral[None, :, None] &
                              ~jnp.eye(5, dtype=jnp.bool_)[:, :, None], axis=1)
    duplicate = prior_types[None, :] | other_collateral
    extra = literal & ((~useful | reserved)[None, :] | duplicate) & collateral[:, None]
    return (old | extra) & literal


def reachable(heights, available):
    # Every intervening rank must already have a clued connecting card.
    ranks = jnp.arange(5)
    a = available.reshape(5, 5) | (ranks[None, :] < heights[:, None])
    prefix = jnp.cumprod(a.astype(jnp.int32), axis=1).astype(jnp.bool_)
    prior = jnp.concatenate([jnp.ones((5, 1), jnp.bool_), prefix[:, :-1]], axis=1)
    return (prior & (ranks[None, :] >= heights[:, None])).reshape(25)


def save_candidates(clue, on_chop, useful, critical):
    ranks = jnp.tile(jnp.arange(5), 5)
    allowed = ((clue == 9) & (ranks == 4)) | ((clue == 6) & (ranks == 1))
    allowed |= critical & (ranks != 4)
    return allowed & useful & on_chop


def make_dependencies(cards, touched, other, other_touched, focus, recipient,
                      heights, save):
    colors = jnp.repeat(jnp.arange(5), 5)
    ranks = jnp.tile(jnp.arange(5), 5)
    need_ids = colors[:, None] * 5 + jnp.arange(5)[None, :]
    own = jnp.transpose(cards[:, need_ids], (1, 2, 0))
    own &= (touched & (jnp.arange(5) != focus))[None, None, :]
    external = jnp.transpose(other[:, need_ids], (1, 2, 0)) & other_touched[None, None, :]
    sources = jnp.stack([jnp.where(recipient == 0, own, external),
                         jnp.where(recipient == 0, external, own)], axis=2)
    required = ((jnp.arange(5)[None, :] >= heights[colors, None]) &
                (jnp.arange(5)[None, :] < ranks[:, None]) & ~save[:, None])
    required &= cards[focus, :, None]
    return required, sources & required[:, :, None, None]


def dependency_feasible(required, support):
    """Hall's condition: each missing rank needs a distinct clued card."""
    subset = jnp.arange(1, 32, dtype=jnp.uint32)
    rank_bits = jnp.sum(required.astype(jnp.uint32) * (jnp.uint32(1) << jnp.arange(5, dtype=jnp.uint32)), axis=-1)
    edges = jnp.sum(support.reshape(support.shape[:-2] + (10,)).astype(jnp.uint32) *
                    (jnp.uint32(1) << jnp.arange(10, dtype=jnp.uint32)), axis=-1)
    active_subsets = rank_bits[..., None] & subset
    neighbors = jnp.zeros_like(active_subsets)
    for r in range(5):
        neighbors |= jnp.where((active_subsets & jnp.uint32(1 << r)) != 0, edges[..., r, None], jnp.uint32(0))
    return jnp.all(jax.lax.population_count(neighbors) >= jax.lax.population_count(active_subsets), axis=-1)


def prompt_order_safe(focus_id, focus, heights, identities, touched, beliefs, giver_known):
    """The leftmost compatible prompt cards must all play successfully."""
    color = focus_id // 5
    target_rank = focus_id % 5
    def rank_step(rank, state):
        stacks, safe = state
        needed = (rank >= heights[color]) & (rank < target_rank)
        need_id = color * 5 + rank
        external = giver_known[need_id]
        eligible = touched & beliefs[:, need_id] & (jnp.arange(5) != focus)
        def slot_step(i, inner):
            hs, ok, found = inner
            slot = 4-i
            take = needed & ~external & ~found & eligible[slot]
            cid = identities[slot]
            plays = (cid % 5) == hs[cid // 5]
            ok &= ~take | plays
            hs = hs.at[cid // 5].add((take & plays).astype(jnp.int32))
            found |= take & (cid == need_id)
            return hs, ok, found
        stacks, safe, found = jax.lax.fori_loop(0, 5, slot_step, (stacks, safe, jnp.asarray(False)))
        safe &= ~needed | external | found
        stacks = stacks.at[color].set(jnp.where(needed & external, jnp.maximum(stacks[color], rank+1), stacks[color]))
        return stacks, safe
    _, safe = jax.lax.fori_loop(0, 5, rank_step, (heights, jnp.asarray(True)))
    return safe


def interpret(notes, literal, touched, occupied, matches, clue, heights, useful,
              critical, other_cards, other_touched, exceptions=None):
    """Interpret one actual or hypothetical clue from recipient's perspective.

    other_cards are visible identities when receiving, conservative possibility
    sets when modeling the partner. No inference uses the recipient's identity.
    """
    colors = jnp.repeat(jnp.arange(5), 5)
    ranks = jnp.tile(jnp.arange(5), 5)
    focus, cp, new = focus_of(matches, touched, occupied)
    now_touched = touched | matches
    certain_other = other_cards & (jnp.sum(other_cards, -1) == 1)[:, None]
    reserved = jnp.any(certain_other & other_touched[:, None], axis=0)
    not_played = jnp.tile(jnp.arange(5), 5) >= jnp.repeat(heights, 5)
    # A giver cannot see their own hidden cards. A required save can
    # therefore duplicate an uncertain clued card in the giver's hand.
    save_dup = literal & reserved[None, :] & (jnp.arange(5) == focus)[:, None]
    save_dup &= (focus == cp) & useful[None, :] & (((clue == 6) & (ranks == 1)) | (ranks == jnp.repeat(heights, 5)))[None, :]
    exceptions = save_dup if exceptions is None else (exceptions | save_dup)
    b = prune(notes, literal, now_touched, not_played, reserved, exceptions)
    b = jnp.where(new[:, None], nonempty_or(b & useful[None, :], b), b)
    own_connectors = b & touched[:, None] & (jnp.arange(5) != focus)[:, None]
    # Ambiguous clued connectors allow self-prompts, but never blind connectors.
    available = jnp.any(own_connectors, axis=0) | jnp.any(other_cards & other_touched[:, None], axis=0)
    play = reachable(heights, available) & useful
    save = save_candidates(clue, focus == cp, useful, critical)
    req, src = make_dependencies(b, touched, other_cards, other_touched, focus, 0, heights, save)
    play &= dependency_feasible(req, src)
    allowed = play | save
    focused = b[focus] & allowed
    b = b.at[focus].set(nonempty_or(focused, b[focus]))
    b = prune(b, literal, now_touched, not_played, reserved, exceptions)
    # A forced self-prompt is justified only if no direct or visible delayed
    # interpretation and no save remains. All compatible colors are preserved.
    direct_available = jnp.any(other_cards & other_touched[:, None], axis=0)
    direct = reachable(heights, direct_available) & b[focus]
    forced_prompt = ~jnp.any(direct | (save & b[focus])) & jnp.any(focused)
    predecessor = jnp.concatenate([b[focus].reshape(5, 5)[:, 1:], jnp.zeros((5, 1), jnp.bool_)], axis=1).reshape(25)
    immediate = ranks == jnp.repeat(heights, 5)
    compat = own_connectors & predecessor[None, :] & immediate[None, :]
    slot = jnp.max(jnp.where(jnp.any(compat, -1), jnp.arange(5), -1))
    prompt_note = b[jnp.maximum(slot, 0)] & immediate
    b = b.at[jnp.maximum(slot, 0)].set(jnp.where(forced_prompt & (slot >= 0),
                           nonempty_or(prompt_note, b[jnp.maximum(slot, 0)]), b[jnp.maximum(slot, 0)]))
    return b, focus, new


def continue_prompts(notes, required, support, heights, receiver):
    colors = jnp.repeat(jnp.arange(5), 5)
    ranks = jnp.tile(jnp.arange(5), 5)
    next_rank = jnp.minimum(heights[colors], 4)
    req = jnp.take_along_axis(required, next_rank[None, :, None], axis=2)[:, :, 0]
    own = jnp.take_along_axis(support[:, :, :, receiver, :], next_rank[None, :, None, None], axis=2)[:, :, 0, :]
    external = jnp.take_along_axis(support[:, :, :, 1-receiver, :], next_rank[None, :, None, None], axis=2)[:, :, 0, :]
    forced_type = req & (ranks[None, :] > heights[colors][None, :]) & jnp.any(own, -1) & ~jnp.any(external, -1)
    forced = jnp.any(notes, -1) & jnp.all(~notes | forced_type, -1)
    eligible = jnp.any(own & notes[:, :, None], axis=1)
    slot = jnp.max(jnp.where(eligible, jnp.arange(5)[None, :], -1), axis=-1)
    prompted = jnp.any(forced[:, None] & (slot[:, None] == jnp.arange(5)[None, :]), axis=0)
    immediate = ranks == heights[colors]
    narrow = notes & immediate[None, :]
    prompted &= jnp.any(narrow, -1)
    return jnp.where(prompted[:, None], narrow, notes), prompted


def agent_act(obs, reset, memory, rng):
    m = jax.tree.map(lambda x, z: jnp.where(reset, z, x), memory, init_agent())
    f = obs.features
    hand = f[:125].reshape(5, 25) > 0
    ids = jnp.argmax(hand, -1)
    pc, pr = ids // 5, ids % 5
    k = f[308:658].reshape(2, 5, 35)
    literal = k[:, :, :25] > 0
    occupied = jnp.any(literal, axis=-1)
    touched = jnp.any(k[:, :, 25:] > 0, axis=-1)
    heights = jnp.sum(f[167:192].reshape(5, 5) * jnp.arange(1, 6)[None, :], axis=-1).astype(jnp.int32)
    tokens = jnp.sum(f[192:200]).astype(jnp.int32)
    lives = jnp.sum(f[200:203]).astype(jnp.int32)
    deck = jnp.sum(f[127:167]).astype(jnp.int32)
    d = f[203:253].reshape(5, 10)
    disc = jnp.stack([jnp.sum(d[:, :3], -1), jnp.sum(d[:, 3:5], -1),
                      jnp.sum(d[:, 5:7], -1), jnp.sum(d[:, 7:9], -1), d[:, 9]], axis=-1)
    multiplicity = jnp.asarray([3, 2, 2, 2, 1])
    ranks = jnp.tile(jnp.arange(5), 5)
    played = jnp.arange(5)[None, :] < heights[:, None]
    counts = (multiplicity[None, :] - disc - played).reshape(25)
    dead_rank = jnp.min(jnp.where((multiplicity[None, :] - disc) <= 0,
                                 jnp.arange(5)[None, :], 5), axis=-1)
    not_played = ~played.reshape(25)
    useful = ((jnp.arange(5)[None, :] >= heights[:, None]) &
              (jnp.arange(5)[None, :] < dead_rank[:, None])).reshape(25)
    playable = ranks == jnp.repeat(heights, 5)
    critical = (counts == 1) & useful
    last = f[253:308]
    removed = (last[2] + last[3]) > 0
    actor = jnp.argmax(last[:2])
    removed_slot = jnp.argmax(last[23:28])
    shift = jnp.minimum(jnp.arange(5) + (jnp.arange(5) >= removed_slot), 4)
    shifted = m['notes'][actor, shift].at[4].set(jnp.ones(25, jnp.bool_))
    notes = m['notes'].at[actor].set(jnp.where(removed, shifted, m['notes'][actor]))
    shifted_ex = m['exceptions'][actor, shift].at[4].set(jnp.zeros(25, jnp.bool_))
    exceptions = m['exceptions'].at[actor].set(jnp.where(removed, shifted_ex, m['exceptions'][actor]))
    # Move both ends of every historical connection when cards shift.
    required = m['required']
    support = m['support']
    moved_req = required[actor, shift].at[4].set(False)
    moved_sup = support[actor, shift].at[4].set(False)
    required = required.at[actor].set(jnp.where(removed, moved_req, required[actor]))
    support = support.at[actor].set(jnp.where(removed, moved_sup, support[actor]))
    src = support[:, :, :, :, actor, :]
    src_shifted = jnp.take(src, shift, axis=-1).at[..., 4].set(False)
    support = support.at[:, :, :, :, actor, :].set(jnp.where(removed, src_shifted, src))
    early = m['early'] & ~(last[3].astype(jnp.bool_) &
                              (removed_slot == chop(m['touched'][actor], m['occupied'][actor])))
    own_counts = jnp.maximum(counts - jnp.sum(hand, axis=0), 0)
    own_literal = literal[0] & (own_counts > 0)[None, :]
    own_literal = nonempty_or(own_literal, literal[0])
    literal = literal.at[0].set(own_literal)
    other_reserved = jnp.any(hand & touched[1, :, None], axis=0)
    notes = notes.at[0].set(prune(notes[0], literal[0], touched[0], not_played, other_reserved, exceptions[0]))
    known_self = notes[0] & (jnp.sum(notes[0], axis=-1) == 1)[:, None]
    notes = notes.at[1].set(prune(notes[1], literal[1], touched[1], not_played,
                         jnp.any(known_self & touched[0, :, None], axis=0), exceptions[1]))
    # New draws and later clues do not retroactively support an old promise.
    need_ids = jnp.repeat(jnp.arange(5), 5)[:, None]*5 + jnp.arange(5)[None, :]
    src_self = jnp.stack([notes[0], hand])[:, :, need_ids].transpose(2, 3, 0, 1)
    src_partner = notes[:, :, need_ids].transpose(2, 3, 0, 1)
    support &= jnp.stack([src_self, src_partner])[:, None, :, :, :, :]
    satisfied = jnp.arange(5)[None, :] < heights[jnp.repeat(jnp.arange(5), 5), None]
    possible = dependency_feasible(required & ~satisfied[None, None, :, :], support)
    connected_notes = notes & possible
    contradicted = ~jnp.any(connected_notes, axis=-1) & occupied
    notes = jnp.where(contradicted[:, :, None], literal, connected_notes)
    exceptions |= contradicted[:, :, None] & literal
    required &= ~contradicted[:, :, None, None]
    support &= ~contradicted[:, :, None, None, None, None]
    is_hint = (last[4] + last[5]) > 0
    recipient = jnp.argmax(last[6:8])
    clue = jnp.where(last[4] > 0, jnp.argmax(last[8:13]), 5 + jnp.argmax(last[13:18]))
    other_cards = jnp.where(recipient == 0, hand, notes[0])
    actual_reserved = jnp.any(other_cards & (jnp.sum(other_cards, -1)==1)[:,None] & touched[1-recipient,:,None], axis=0)
    ex = clue_exceptions(exceptions[recipient], literal[recipient], m['touched'][recipient],
                        occupied[recipient], last[18:23]>0, useful, actual_reserved,
                        (tokens==7) | jnp.all(m['touched'][1-recipient] | ~m['occupied'][1-recipient]))
    received_focus, received_cp, _ = focus_of(last[18:23]>0, m['touched'][recipient], occupied[recipient])
    possibly_reserved = jnp.any(other_cards & touched[1-recipient, :, None], axis=0)
    save_dup = literal[recipient] & possibly_reserved[None, :] & (jnp.arange(5) == received_focus)[:, None]
    save_dup &= (received_focus == received_cp) & useful[None, :] & (((clue == 6) & (ranks == 1)) | playable)[None, :]
    ex |= save_dup
    exceptions = exceptions.at[recipient].set(jnp.where(is_hint, ex, exceptions[recipient]))
    # Protect the older counterpart too: saving an unknown duplicate must
    # not retroactively turn the giver's original card into a different rank.
    newly_touched = (last[18:23] > 0) & ~m['touched'][recipient]
    save_or_forced = (received_focus == received_cp) | (tokens == 7) | jnp.all(m['touched'][1-recipient] | ~m['occupied'][1-recipient])
    saved_cards = jnp.where(recipient == 1, hand, literal[0])
    saved_types = jnp.any(saved_cards & newly_touched[:, None], axis=0)
    saved_types &= is_hint & save_or_forced
    giver_dup = m['notes'][1-recipient] & saved_types[None, :] & m['touched'][1-recipient, :, None]
    exceptions = exceptions.at[1-recipient].set(exceptions[1-recipient] | giver_dup)
    notes = notes.at[1-recipient].set(notes[1-recipient] | (giver_dup & literal[1-recipient]))
    other_cards = jnp.where(recipient == 0, hand, notes[0])
    updated, last_focus, _ = interpret(notes[recipient], literal[recipient],
                m['touched'][recipient], occupied[recipient], last[18:23] > 0,
                clue, heights, useful, critical, other_cards, touched[1-recipient], exceptions[recipient])
    notes = notes.at[recipient].set(jnp.where(is_hint, updated, notes[recipient]))
    pre_chop = chop(m['touched'][recipient], occupied[recipient])
    saved = save_candidates(clue, last_focus == pre_chop, useful, critical)
    new_req, new_support = make_dependencies(updated, m['touched'][recipient], other_cards,
                      touched[1-recipient], last_focus, recipient, heights, saved)
    new_feasible = dependency_feasible(new_req, new_support)
    new_req &= new_feasible[:, None]
    new_support &= new_feasible[:, None, None, None]
    required = required.at[recipient, last_focus].set(jnp.where(is_hint, new_req, required[recipient, last_focus]))
    support = support.at[recipient, last_focus].set(jnp.where(is_hint, new_support, support[recipient, last_focus]))
    own_prompted_notes, own_prompted = continue_prompts(notes[0], required[0], support[0], heights, 0)
    partner_prompted_notes, partner_prompted = continue_prompts(notes[1], required[1], support[1], heights, 1)
    notes = jnp.stack([own_prompted_notes, partner_prompted_notes])
    prompted = jnp.stack([own_prompted, partner_prompted])
    exceptions &= ~prompted[:, :, None]
    # Re-run elimination after focus resolution.
    notes = notes.at[0].set(prune(notes[0], literal[0], touched[0], not_played, other_reserved, exceptions[0]))
    joint_belief, count_ok = count_consistent_belief(notes[0], own_counts)
    notes = notes.at[0].set(jnp.where(count_ok, notes[0] & (joint_belief > 0), literal[0]))
    exceptions = exceptions.at[0].set(jnp.where(count_ok, exceptions[0], literal[0]))
    required = required.at[0].set(jnp.where(count_ok, required[0], False))
    support = support.at[0].set(jnp.where(count_ok, support[0], False))
    own_belief = jnp.where(count_ok, joint_belief, normalize(literal[0] * own_counts[None, :]))
    known_self = notes[0] & (jnp.sum(notes[0], axis=-1) == 1)[:, None]
    self_reserved = jnp.any(known_self & touched[0, :, None], axis=0)
    notes = notes.at[1].set(prune(notes[1], literal[1], touched[1], not_played, self_reserved, exceptions[1]))
    pp = jnp.sum(own_belief * playable[None, :], -1)
    trash = ~useful | other_reserved
    pd = jnp.sum(own_belief * trash[None, :], -1)
    cp = chop(touched[0], occupied[0])
    pcp = chop(touched[1], occupied[1])
    final_turn = (deck == 0) & (m['deck'] == 0)
    partner_belief = normalize(notes[1] * counts[None, :])
    partner_safe = jnp.all(~notes[1] | playable[None, :], axis=-1) & occupied[1]
    partner_trash = jnp.all(~notes[1] | (~useful | self_reserved)[None, :], axis=-1) & occupied[1]
    partner_has_action = jnp.any(partner_safe | partner_trash)
    pch = jnp.minimum(pcp, 4)
    same = (ids[:, None] == ids[None, :]) & occupied[1, :, None] & occupied[1, None, :]
    unique_visible = jnp.sum(same[pch]) == 1
    own_possible_reserved = jnp.any(notes[0] & touched[0, :, None], axis=0)
    chop_important = (pcp < 5) & useful[ids[pch]] & ~self_reserved[ids[pch]] & (
                     critical[ids[pch]] | ((pr[pch] == 1) & unique_visible) |
                     (playable[ids[pch]] & unique_visible))
    urgent = chop_important & ~partner_has_action & (deck > 0)
    no_safe_discard = (cp == 5) & ~jnp.any((pd > .999) & occupied[0])

    def evaluate_clue(a):
        color_hint = a < 5
        value = a % 5
        attribute = jnp.where(color_hint, jnp.repeat(jnp.arange(5), 5), ranks)
        mask = attribute == value
        matches = jnp.any(hand & mask[None, :], axis=-1)
        post_literal = literal[1] & jnp.where(matches[:, None], mask[None, :], ~mask[None, :])
        candidate_ex = clue_exceptions(exceptions[1], post_literal, touched[1], occupied[1],
                                       matches, useful, self_reserved, (tokens==8) | no_safe_discard)
        preliminary_focus, _, _ = focus_of(matches, touched[1], occupied[1])
        save_dup = post_literal & own_possible_reserved[None, :] & (jnp.arange(5) == preliminary_focus)[:, None]
        save_dup &= (preliminary_focus == pcp) & useful[None, :] & (((a == 6) & (ranks == 1)) | playable)[None, :]
        candidate_ex |= save_dup
        after, focus, new = interpret(notes[1], post_literal, touched[1], occupied[1],
                             matches, a, heights, useful, critical, notes[0], touched[0], candidate_ex)
        after = prune(after, post_literal, touched[1] | matches, not_played, self_reserved, candidate_ex)
        # A literal refinement can activate an older delayed promise. Include
        # that public history when checking what this clue asks the partner to play.
        future_req, future_sup = make_dependencies(after, touched[1], notes[0], touched[0],
                    focus, jnp.int32(1), heights, save_candidates(a, focus == pcp, useful, critical))
        feasible = dependency_feasible(future_req, future_sup)
        projected_req = required[1].at[focus].set(future_req & feasible[:, None])
        projected_sup = support[1].at[focus].set(future_sup & feasible[:, None, None, None])
        projected_sources = jnp.stack([notes[0], after])[:, :, need_ids].transpose(2, 3, 0, 1)
        projected_sup &= projected_sources[None, :, :, :, :]
        projected_possible = dependency_feasible(projected_req & ~satisfied[None, :, :], projected_sup)
        after = nonempty_or(after & projected_possible, post_literal)
        after, projected_prompt = continue_prompts(after, projected_req, projected_sup, heights, 1)
        effective_ex = candidate_ex & ~projected_prompt[:, None]
        after = prune(after, post_literal, touched[1] | matches, not_played, self_reserved, effective_ex)
        after_safe = jnp.all(~after | playable[None, :], -1) & occupied[1]
        gained_safe = jnp.sum(after_safe & ~partner_safe)
        nnew = jnp.sum(new)
        # Only known existing connectors justify a delayed play when giving.
        actual_connectors = jnp.any(hand & touched[1, :, None] & (jnp.arange(5) != focus)[:, None], axis=0)
        actual_connectors |= self_reserved
        can_connect = reachable(heights, actual_connectors)
        focus_id = ids[focus]
        save_allowed = (focus == pcp) & (
               ((a == 9) & (pr[focus] == 4)) |
               ((a == 6) & (pr[focus] == 1) & (jnp.sum(same[focus]) == 1)) |
               (critical[focus_id] & (pr[focus] != 4)))
        before_prompt = prune(notes[1], post_literal, touched[1], not_played, self_reserved, candidate_ex)
        prompt_safe = prompt_order_safe(focus_id, focus, heights, ids, touched[1], before_prompt, self_reserved)
        valid_meaning = useful[focus_id] & ((can_connect[focus_id] & prompt_safe) | save_allowed)
        # New touches cannot duplicate public trash, each other, or a clued card.
        already_here = jnp.any(same & touched[1, None, :] & ~jnp.eye(5, dtype=jnp.bool_), axis=-1)
        new_duplicate = jnp.any(same & new[None, :] & (jnp.arange(5)[None, :] < jnp.arange(5)[:, None]), axis=-1)
        possible_own_dup = jnp.any(notes[0] & touched[0, :, None], axis=0)[ids]
        bad = new & (~useful[ids] | already_here | new_duplicate | possible_own_dup)
        bad_count = jnp.sum(bad)
        saves_chop = matches[pch] & (pcp < 5) & chop_important
        save_exception = saves_chop & (urgent | early)
        # Save Principle overrides Minimum Clue Value when a tempo play
        # actually occupies the partner instead of losing an important chop.
        tempo_ok = (tokens == 8) | no_safe_discard | (deck <= 5) | (urgent & (gained_safe > 0))
        valid = valid_meaning & ((nnew > 0) | tempo_ok) & ((bad_count == 0) | save_exception)
        # Avoid a prediction that explicitly asks the partner to misplay.
        unsafe_instruction = jnp.any(after_safe & ~playable[ids])
        valid &= ~unsafe_instruction
        valid &= ~(new[focus] & possible_own_dup[focus] & ~save_exception)
        # Prefer immediately actionable, efficient clues; color breaks close ties.
        early_tempo = early & (tokens >= 6) & ~jnp.any(pp > .999)
        touch_value = jnp.where(early_tempo, .25, 1.1)
        play_value = jnp.where(early_tempo, jnp.minimum(gained_safe, 1) + .15*jnp.maximum(gained_safe-1, 0), gained_safe)
        color_value = jnp.where(early_tempo, .85, .15)
        score = 2.0 + touch_value * nnew + 2.2 * play_value + color_value * color_hint
        score += jnp.where(urgent & (gained_safe > 0), 8., 0.)
        prepare_endgame = (deck > 0) & (deck <= 5) & (gained_safe > 0) & ~jnp.any(partner_safe)
        score += jnp.where(prepare_endgame, 8., 0.)
        score += .15 * jnp.sum(jnp.log1p(jnp.sum(notes[1], -1)) - jnp.log1p(jnp.sum(after, -1)))
        score += 1.2 * saves_chop + .4 * jnp.sum(new & critical[ids])
        score -= 12.0 * bad_count
        score += jnp.where(urgent & (saves_chop | jnp.any(after_safe)), 15., 0.)
        score -= jnp.where(urgent & ~(saves_chop | jnp.any(after_safe)), 20., 0.)
        score += jnp.where(early & (nnew > 0), 1., 0.)
        score -= jnp.where((tokens <= 2) & partner_has_action & ~urgent, 3., 0.)
        definite_after, _, _ = interpret(notes[1], post_literal, touched[1], occupied[1],
                           matches, a, heights, useful, critical, known_self, touched[0], candidate_ex)
        definite_safe = jnp.all(~definite_after | playable[None, :], -1) & occupied[1]
        wrong_notes = jnp.sum((touched[1] | matches) &
                      (~after[jnp.arange(5), ids] | ~definite_after[jnp.arange(5), ids]))
        unsafe_instruction |= jnp.any(definite_safe & ~playable[ids])
        forced_score = -40. + 10. * valid_meaning + 3. * playable[focus_id] - 15. * bad_count
        forced_score -= 40. * wrong_notes + 40. * unsafe_instruction
        forced_score += .1 * gained_safe + .05 * color_hint
        # Literal trash or a harmless retouch is preferable to an impossible
        # Level 1 promise when full tokens force a clue. No stall signal added.
        forced_score += 3. * jnp.sum(matches & touched[1]) - 2. * nnew
        protects = urgent & valid & (saves_chop | jnp.any(after_safe))
        return jnp.where(valid, score, -1e4), gained_safe, focus, bad_count, forced_score, protects

    clue_scores, clue_gain, clue_focus, clue_bad, forced_scores, protects = jax.vmap(evaluate_clue)(jnp.arange(10))
    protective_score = jnp.max(jnp.where(protects & obs.legal_actions[10:20], clue_scores, -1e4))
    clue_scores = jnp.where(protects, clue_scores - protective_score + 30., clue_scores)
    # Safe plays, older first; promote plays that unlock visible cards.
    unlock = jnp.sum(own_belief * jnp.roll(jnp.sum(hand, axis=0).reshape(5,5), -1, axis=1).reshape(25)[None,:], -1)
    play_scores = 7.5 + .3 * unlock - .03 * jnp.arange(5) + .3 * jnp.sum(own_belief * (ranks == 4), -1)
    play_scores = jnp.where(pp > .999, play_scores, -1e4)
    # Risk is a personal tactical choice, not a message to the partner.
    wrong_cost = (~playable) * (.25 + critical * (5-ranks))
    expected_wrong_cost = jnp.sum(own_belief * wrong_cost[None, :], axis=-1)
    calculated_play = (lives == 3) & (deck > 0) & touched[0] & (pp >= .7999) & (expected_wrong_cost < .45)
    calculated_play &= (tokens > 0) | partner_has_action
    play_scores = jnp.maximum(play_scores, jnp.where(calculated_play, 2.0 + pp, -1e4))
    # Literal/endgame probabilities are a fallback, never a private signal.
    end_threshold = jnp.where(lives >= 2, .001, .98)
    gamble = (deck == 0) & (pp >= end_threshold)
    play_scores = jnp.maximum(play_scores, jnp.where(gamble, 2.5 + pp, -1e4))
    discard_scores = jnp.where(jnp.arange(5) == cp, 0., -1e4)
    discard_scores = jnp.maximum(discard_scores, jnp.where(pd > .999, 1. + .2 * touched[0] - .02*jnp.arange(5), -1e4))
    forced_loss = jnp.sum(own_belief * (critical * (5-ranks))[None, :], axis=-1)
    discard_scores = jnp.where(no_safe_discard, -20. + 3. * pd - forced_loss, discard_scores)
    discard_scores = jnp.where(no_safe_discard & (tokens > 0), discard_scores - 1000., discard_scores)
    # At full clues a legal clue may be forced. Retouch known information with
    # minimal new bad meaning if no conventional play/save exists.
    fallback_clue = forced_scores
    clue_scores = jnp.maximum(clue_scores, jnp.where((tokens == 8) | no_safe_discard, fallback_clue, -1e4))
    clue_scores = jnp.where(final_turn, jnp.minimum(clue_scores, -50.), clue_scores)
    scores = jnp.concatenate([discard_scores, play_scores, clue_scores, jnp.asarray([0.])])
    scores = jnp.where(obs.legal_actions, scores, -1e9)
    action = jnp.argmax(scores).astype(jnp.int32)
    next_memory = {'notes': notes, 'exceptions': exceptions, 'required': required, 'support': support, 'touched': touched, 'occupied': occupied,
                   'early': early, 'turn': m['turn'] + 1, 'deck': deck}
    trace = {'pp': pp, 'pd': pd, 'notes': notes[0], 'partner_notes': notes[1],
             'clue_scores': clue_scores, 'chop': cp, 'partner_chop': pcp,
             'urgent': urgent, 'last_focus': jnp.where(is_hint, last_focus, -1),
             'scores': scores, 'early': early, 'visible_hand': ids,
             'heights': heights, 'tokens': tokens, 'lives': lives, 'deck': deck,
             'last': last, 'contradicted': contradicted, 'final_turn': final_turn, 'prompted': own_prompted, 'count_ok': count_ok, 'calculated_play': calculated_play, 'wrong_cost': expected_wrong_cost}
    return action, next_memory, trace
