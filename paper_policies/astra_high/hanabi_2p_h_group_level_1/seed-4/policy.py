"""Two-player H-group Level 1 policy; all inference is observation-local."""
import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'h1_local_notes', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return {'notes': jnp.ones((2, 5, 25), jnp.bool_),
            'previous': jnp.zeros(658, jnp.float32),
            'dependency': jnp.full((2, 5), -1, jnp.int32),
            'exempt': jnp.zeros((2, 5, 25), jnp.bool_),
            'wait': jnp.zeros((2, 5, 25, 5), jnp.bool_),
            'early': jnp.array(True)}


def decode(f):
    hand = f[:125].reshape(5, 25)
    k = f[308:].reshape(2, 5, 35)
    literal = k[:, :, :25] > 0
    touched = jnp.any(k[:, :, 25:] > 0, axis=-1)
    occupied = jnp.any(literal, axis=-1)
    heights = jnp.sum(f[167:192].reshape(5, 5) * jnp.arange(1, 6), axis=-1).astype(jnp.int32)
    d = f[203:253].reshape(5, 10)
    disc = jnp.stack([d[:, :3].sum(-1), d[:, 3:5].sum(-1), d[:, 5:7].sum(-1), d[:, 7:9].sum(-1), d[:, 9]], -1)
    counts = jnp.array([3, 2, 2, 2, 1])[None, :] - disc - (jnp.arange(5)[None, :] < heights[:, None])
    ranks = jnp.tile(jnp.arange(5), 5)
    colors = jnp.repeat(jnp.arange(5), 5)
    dead_suit = jnp.cumsum((counts <= 0) & (jnp.arange(5)[None, :] >= heights[:, None]), axis=1) > 0
    useful = ((jnp.arange(5)[None, :] >= heights[:, None]) & ~dead_suit).reshape(25)
    playable = ranks == heights[colors]
    return hand, literal, touched, occupied, heights, counts.reshape(25), useful, playable


def intersect_or(old, constraint, fallback):
    new = old & constraint
    return jnp.where(jnp.any(new, -1, keepdims=True), new, fallback)


def chop_of(touched, occupied):
    return jnp.min(jnp.where(~touched & occupied, jnp.arange(5), 5))


def focus_of(touch, touched, occupied):
    new = touch & ~touched
    chop = chop_of(touched, occupied)
    focus = jnp.where(jnp.any(new), jnp.max(jnp.where(new, jnp.arange(5), -1)), jnp.max(jnp.where(touch, jnp.arange(5), -1)))
    focus = jnp.where((chop < 5) & touch[jnp.minimum(chop, 4)], chop, focus)
    return jnp.maximum(focus, 0), chop, new


def tidy(notes, literal, touched, useful, visible, counts, heights, exempt):
    # Literal evidence always wins over a contradicted convention note.
    notes = intersect_or(notes, literal, literal)
    # Good Touch lets ambiguous clued cards shed played/unreachable identities.
    not_played = (jnp.tile(jnp.arange(5), 5) >= jnp.repeat(heights, 5))
    narrowed = notes & (not_played[None, None, :] | exempt)
    notes = jnp.where(touched[:, :, None] & jnp.any(narrowed, -1, keepdims=True), narrowed, notes)
    # Exact inferred cards cannot also be another useful clued card.
    def eliminate(_, n):
        exact = (n.sum(-1) == 1) & touched
        certain = n & exact[:, :, None]
        total = certain.sum(1)
        other = total[:, None, :] - certain > 0
        candidate = n & ~(other & touched[:, :, None] & ~exempt)
        n = jnp.where(jnp.any(candidate, -1, keepdims=True), candidate, n)
        return n
    notes = jax.lax.fori_loop(0, 2, eliminate, notes)
    # A visible, already-clued partner card is another Good Touch exclusion.
    partner_claim = jnp.any((visible > 0) & touched[1, :, None], axis=0)
    own = notes[0] & ~(partner_claim[None, :] & touched[0, :, None] & ~exempt[0])
    notes = notes.at[0].set(jnp.where(jnp.any(own, -1, keepdims=True), own, notes[0]))
    own_certain = jnp.any(notes[0] & touched[0, :, None] & (notes[0].sum(-1) == 1)[:, None], axis=0)
    partner = notes[1] & ~(own_certain[None, :] & touched[1, :, None] & ~exempt[1])
    notes = notes.at[1].set(jnp.where(jnp.any(partner, -1, keepdims=True), partner, notes[1]))
    return notes


def clue_notes(notes, literal_after, touched, occupied, receiver, touch, rank_clue, value, heights, counts, useful, other_cards):
    """Interpret one clue from its pre-clue hand, retaining play/save alternatives."""
    ranks = jnp.tile(jnp.arange(5), 5)
    colors = jnp.repeat(jnp.arange(5), 5)
    focus, chop, new = focus_of(touch, touched[receiver], occupied[receiver])
    recv = notes[receiver]
    other = 1 - receiver
    # Connectors must already be clued at the moment the clue is given.
    other_have = jnp.any(other_cards & touched[other, :, None], axis=0)
    exact_recv = recv & (recv.sum(-1) == 1)[:, None] & touched[receiver, :, None]
    current_playable = ranks == heights[colors]
    receiver_available = jnp.maximum(counts - other_cards.sum(0), 0)
    feasible_recv = recv & (receiver_available > 0)[None, :]
    ready_slots = touched[receiver] & jnp.any(feasible_recv, -1) & jnp.all(~feasible_recv | current_playable[None, :], -1)
    ready_cards = feasible_recv & ready_slots[:, None] & (jnp.arange(5) != focus)[:, None]
    own_have = jnp.any((exact_recv | ready_cards) & (jnp.arange(5) != focus)[:, None], axis=0)
    have = (other_have | own_have).reshape(5, 5)
    before = jnp.arange(5)[None, :] < heights[:, None]
    accounted = have | before
    connected = jnp.concatenate([jnp.ones((5, 1), jnp.bool_), jnp.cumprod(accounted[:, :4], axis=1).astype(jnp.bool_)], axis=1).reshape(25)
    play_set = connected & useful
    critical_save = (counts == 1) & (ranks != 4)
    special_save = rank_clue & ((value == 1) | (value == 4))
    save_set = useful & (critical_save | special_save) & (focus == chop)
    meaning = play_set | save_set
    new_recv = intersect_or(recv, literal_after[receiver], literal_after[receiver])
    good = new_recv & useful[None, :]
    new_recv = jnp.where(new[:, None] & jnp.any(good, -1, keepdims=True), good, new_recv)
    focused = new_recv[focus] & meaning
    # A Level 1 self-prompt needs an already-clued connector; never use an
    # unclued finesse position. Prefer ordinary interpretations when available.
    predecessor = jnp.maximum(jnp.arange(25) - 1, 0)
    compatible = (new_recv[:, predecessor] & touched[receiver, :, None]
                  & (jnp.arange(5) != focus)[:, None])
    connector = jnp.max(jnp.where(compatible, jnp.arange(5)[:, None], -1), axis=0)
    prompt_set = (new_recv[focus] & useful & (ranks == heights[colors] + 1)
                  & (connector >= 0) & ~other_have[predecessor])
    need_prompt = ~jnp.any(focused) & jnp.any(prompt_set)
    chosen = jnp.max(jnp.where(prompt_set, connector, -1))
    same_connector = jnp.all(~prompt_set | (connector == chosen))
    use_prompt = need_prompt & same_connector
    single_connector = jnp.sum(jnp.any(compatible & prompt_set[None, :], axis=-1)) == 1
    focused = jnp.where(need_prompt, prompt_set, focused)
    new_recv = new_recv.at[focus].set(jnp.where(jnp.any(focused), focused, new_recv[focus]))
    connector_meaning = jnp.concatenate([prompt_set.reshape(5, 5)[:, 1:], jnp.zeros((5, 1), jnp.bool_)], axis=1).reshape(25)
    connector_note = new_recv[jnp.maximum(chosen, 0)] & connector_meaning
    new_recv = new_recv.at[jnp.maximum(chosen, 0)].set(jnp.where(use_prompt, connector_note, new_recv[jnp.maximum(chosen, 0)]))
    # Only the cards present when the clue was given may support this
    # ambiguous ordinary delay. Later draws cannot retroactively justify it.
    ordinary_support = ready_cards[:, predecessor].T
    conditional = (play_set & ~save_set & new_recv[focus] & (ranks == heights[colors] + 1)
                   & ~other_have[predecessor])
    ordinary_support = ordinary_support & conditional[:, None]
    return notes.at[receiver].set(new_recv), focus, jnp.any(save_set & new_recv[focus]), jnp.where(use_prompt, chosen, -1).astype(jnp.int32), single_connector, ordinary_support


def beliefs(notes, counts, visible, touched):
    available = jnp.maximum(counts - visible.sum(0), 0)
    own = notes[0] * available[None, :]
    own = jnp.where(own.sum(-1, keepdims=True) > 0, own, notes[0].astype(jnp.float32))
    own /= jnp.maximum(own.sum(-1, keepdims=True), 1)
    partner = notes[1].astype(jnp.float32) * counts[None, :]
    partner /= jnp.maximum(partner.sum(-1, keepdims=True), 1)
    return jnp.stack([own, partner])



def exact_own_belief(mask, available, occupied, fallback):
    """Exact five-slot marginals, using subset DP and physical multiplicities.

    Each type can fill any compatible subset of the five labeled slots. The
    falling factorial counts the distinct physical copies assigned to it.
    Differentiating the partition sum gives all 125 marginal numerators.
    """
    subsets = jnp.arange(32, dtype=jnp.int32)
    bits = (subsets[:, None] & (1 << jnp.arange(5))[None, :]) != 0
    sizes = bits.sum(-1)
    allowed = (subsets[None, :] & subsets[:, None]) == subsets[None, :]
    previous = subsets[:, None] ^ subsets[None, :]
    copies = jnp.concatenate([available, jnp.array([5.0])])
    weights = jnp.concatenate([mask.astype(jnp.float32), (~occupied)[:, None].astype(jnp.float32)], axis=1)
    falling = jnp.prod(jnp.where(jnp.arange(5)[None, None, :] < sizes[None, :, None],
                                jnp.maximum(copies[:, None, None] - jnp.arange(5)[None, None, :], 0), 1), axis=-1)
    def partition(w):
        def one_type(dp, inputs):
            column, multiplicity = inputs
            term = jnp.prod(jnp.where(bits, column[None, :], 1.0), axis=-1) * multiplicity
            dp = jnp.sum(jnp.where(allowed, dp[previous] * term[None, :], 0.0), axis=-1)
            return dp, None
        initial = jnp.zeros(32).at[0].set(1.0)
        final, _ = jax.lax.scan(one_type, initial, (w.T, falling))
        return final[31]
    normalizer, gradient = jax.value_and_grad(partition)(weights)
    marginals = (gradient * weights / jnp.maximum(normalizer, 1e-20))[:, :25]
    marginals /= jnp.maximum(marginals.sum(-1, keepdims=True), 1e-20)
    return jnp.where(normalizer > 0, marginals, fallback), normalizer > 0

def agent_act(obs, reset, memory, rng):
    m = jax.tree.map(lambda a, b: jnp.where(reset, a, b), init_agent(), memory)
    f = obs.features
    hand, literal, touched, occupied, heights, counts, useful, playable = decode(f)
    _, prev_lit, prev_touched, prev_occ, _, _, _, _ = decode(m['previous'])
    last = f[253:308]
    actor = jnp.argmax(last[:2])
    is_remove = (last[2] + last[3]) > 0
    is_clue = (last[4] + last[5]) > 0
    removed = jnp.argmax(last[23:28])
    notes = m['notes']
    dependencies = m['dependency']
    exempt = m['exempt']
    wait = m['wait']
    was_waiting = jnp.any(wait, axis=-1)
    revealed = jnp.argmax(last[28:53])
    waiting = dependencies[actor] == removed
    linked_meaning = (jnp.repeat(jnp.arange(5), 5) == revealed // 5) & (jnp.tile(jnp.arange(5), 5) == revealed % 5 + 1)
    narrowed_link = notes[actor] & linked_meaning[None, :]
    resolve_link = waiting & is_remove & (last[2] > 0) & (last[53] > 0) & jnp.any(narrowed_link, -1)
    notes = notes.at[actor].set(jnp.where(resolve_link[:, None], narrowed_link, notes[actor]))
    shifted_dep = jnp.where(dependencies[actor] == removed, -1, dependencies[actor] - (dependencies[actor] > removed))
    shift_idx = jnp.minimum(jnp.arange(5) + (jnp.arange(5) >= removed), 4)
    shifted = notes[actor, shift_idx]
    shifted = shifted.at[4].set(jnp.ones(25, jnp.bool_))
    notes = notes.at[actor].set(jnp.where(is_remove & ~reset, shifted, notes[actor]))
    shifted_dep = shifted_dep[shift_idx].at[4].set(-1)
    dependencies = dependencies.at[actor].set(jnp.where(is_remove & ~reset, shifted_dep, dependencies[actor]))
    shifted_exempt = exempt[actor, shift_idx].at[4].set(False)
    exempt = exempt.at[actor].set(jnp.where(is_remove & ~reset, shifted_exempt, exempt[actor]))
    shifted_wait = wait[actor][:, :, shift_idx].at[:, :, 4].set(False)
    shifted_wait = shifted_wait[shift_idx].at[4].set(False)
    shifted_was_waiting = was_waiting[actor, shift_idx].at[4].set(False)
    wait = wait.at[actor].set(jnp.where(is_remove & ~reset, shifted_wait, wait[actor]))
    was_waiting = was_waiting.at[actor].set(jnp.where(is_remove & ~reset, shifted_was_waiting, was_waiting[actor]))
    early = m['early'] & ~(is_remove & (last[3] > 0) & (removed == chop_of(prev_touched[actor], prev_occ[actor])))
    receiver = jnp.argmax(last[6:8])
    other_cards = jnp.where(receiver == 0, hand > 0, notes[0] & (notes[0].sum(-1) == 1)[:, None])
    interpreted, last_focus, last_save, last_connector, last_linkable, new_wait = clue_notes(notes, literal, prev_touched, prev_occ, receiver, last[18:23] > 0, last[5] > 0, jnp.argmax(last[13:18]), heights, counts, useful, other_cards)
    notes = jnp.where(is_clue & ~reset, interpreted, notes)
    wait = wait.at[receiver, last_focus].set(jnp.where(is_clue & ~reset, new_wait, wait[receiver, last_focus]))
    was_waiting = was_waiting.at[receiver, last_focus].set(jnp.where(is_clue & ~reset, jnp.any(new_wait, -1), was_waiting[receiver, last_focus]))
    dependencies = dependencies.at[receiver, last_focus].set(jnp.where(is_clue & ~reset, jnp.where(last_linkable, last_connector, -1), dependencies[receiver, last_focus]))
    rank_two_save = is_clue & (last[5] > 0) & (last[14] > 0) & (last_focus == chop_of(prev_touched[receiver], prev_occ[receiver]))
    other_possible = jnp.where(receiver == 0, hand > 0, notes[0])
    clued_other_possible = jnp.any(other_possible & prev_touched[1-receiver, :, None], axis=0)
    chop_exception = is_clue & (last_focus == chop_of(prev_touched[receiver], prev_occ[receiver]))
    exceptional_types = playable | (rank_two_save & (jnp.tile(jnp.arange(5), 5) == 1))
    save_exempt = clued_other_possible & exceptional_types & literal[receiver, last_focus] & useful
    # A legitimate save can also duplicate a previously saved hidden card in
    # the giver's own hand; preserve that reciprocal alternative.
    target_types = jnp.where(receiver == 1, hand[last_focus] > 0, interpreted[receiver, last_focus]) & useful & exceptional_types
    reciprocal = prev_touched[1-receiver, :, None] & literal[1-receiver] & target_types[None, :]
    exempt = exempt.at[1-receiver].set(exempt[1-receiver] | (chop_exception & reciprocal))
    old_exempt = exempt[receiver, last_focus]
    exempt = exempt.at[receiver, last_focus].set(jnp.where(chop_exception, old_exempt | save_exempt, jnp.where(is_clue, jnp.zeros(25, jnp.bool_), old_exempt)))
    notes = tidy(notes, literal, touched, useful, hand, counts, heights, exempt)
    predecessor = jnp.maximum(jnp.arange(25) - 1, 0)
    still_possible = jnp.transpose(notes[:, :, predecessor], (0, 2, 1))
    wait = wait & still_possible[:, None, :, :]
    supported_by_board = jnp.tile(jnp.arange(5), 5) <= jnp.repeat(heights, 5)
    broken_delay = was_waiting & ~jnp.any(wait, -1) & ~supported_by_board[None, None, :]
    revised = notes & ~broken_delay
    notes = jnp.where(jnp.any(revised, -1, keepdims=True), revised, literal)
    wait = wait & ~supported_by_board[None, None, :, None]
    # A model of the partner's belief is uncertain; never treat a demonstrably
    # wrong predicted identity as public card evidence.
    partner_consistent = jnp.any(notes[1] & (hand > 0), axis=-1)
    notes = notes.at[1].set(jnp.where(partner_consistent[:, None], notes[1], literal[1]))
    wait = wait.at[1].set(wait[1] & partner_consistent[:, None, None])
    probs = beliefs(notes, counts, hand, touched)
    exact_probs, count_consistent = exact_own_belief(notes[0], jnp.maximum(counts - hand.sum(0), 0), occupied[0], probs[0])
    literal_fallback = beliefs(notes.at[0].set(literal[0]), counts, hand, touched)[0]
    exact_probs = jnp.where(count_consistent, exact_probs, literal_fallback)
    notes = notes.at[0].set(jnp.where(count_consistent, notes[0], literal[0]))
    dependencies = dependencies.at[0].set(jnp.where(count_consistent, dependencies[0], -1))
    wait = wait.at[0].set(wait[0] & count_consistent)
    probs = probs.at[0].set(exact_probs)
    pplay = (probs * playable).sum(-1)
    ptrash = (probs * ~useful).sum(-1)
    ranks = jnp.tile(jnp.arange(5), 5)
    colors = jnp.repeat(jnp.arange(5), 5)
    tokens = f[192:200].sum()
    lives = f[200:203].sum()
    deck = f[127:167].sum()
    partner_id = jnp.argmax(hand, -1)
    partner_live = occupied[1] & useful[partner_id]
    pchop = chop_of(touched[1], occupied[1])
    chop_id = partner_id[jnp.minimum(pchop, 4)]
    partner_ready = jnp.any((pplay[1] > .999) & occupied[1])
    partner_trash = jnp.any((ptrash[1] > .999) & occupied[1])
    unique_chop = jnp.sum((partner_id == chop_id) & occupied[1]) == 1
    chop_danger = unique_chop & (pchop < 5) & useful[chop_id] & ((counts[chop_id] == 1) | (ranks[chop_id] == 1) | playable[chop_id])
    urgent = chop_danger & ~partner_ready & ~partner_trash
    next_chop = chop_of(touched[1] | (jnp.arange(5) == pchop), occupied[1])
    next_id = partner_id[jnp.minimum(next_chop, 4)]
    next_danger = (next_chop < 5) & useful[next_id] & ((counts[next_id] == 1) | (ranks[next_id] == 1))
    proactive_save = chop_danger & next_danger & ~partner_trash & (jnp.sum((pplay[1] > .999) & occupied[1]) <= 1)
    own_certain = jnp.any((probs[0] > .99999) & touched[0, :, None], axis=0)
    own_claim = jnp.any((probs[0] > .00001) & touched[0, :, None], axis=0)
    known_duplicate = own_claim[partner_id]
    same_card = partner_id[:, None] == partner_id[None, :]
    forced = (tokens == 8) | jnp.all(touched[0] | ~occupied[0])

    def evaluate_hint(a):
        rank_hint = a >= 5
        val = a % 5
        touch = occupied[1] & jnp.where(rank_hint, ranks[partner_id] == val, colors[partner_id] == val)
        focus, chop, new = focus_of(touch, touched[1], occupied[1])
        fid = partner_id[focus]
        # Literal positive and negative information supplied by the clue.
        attribute = jnp.where(rank_hint, ranks == val, colors == val)
        after = literal.at[1].set(literal[1] & (touch[:, None] == attribute[None, :]))
        predicted, _, _, prompt_connector, prompt_single, _ = clue_notes(notes, after, touched, occupied, 1, touch, rank_hint, val, heights, counts, useful, notes[0] & (notes[0].sum(-1) == 1)[:, None])
        tagged = touched.at[1].set(touched[1] | touch)
        save_two = rank_hint & (val == 1) & (focus == chop)
        exceptional_types = playable | (save_two & (ranks == 1))
        pred_exempt = exempt.at[1, focus].set(jnp.where(focus == chop, exempt[1, focus] | (own_claim & after[1, focus] & useful & exceptional_types), jnp.zeros(25, jnp.bool_)))
        pred_exempt = pred_exempt.at[0].set(pred_exempt[0] | ((focus == chop) & touched[0, :, None] & literal[0] & (hand[focus] > 0)[None, :] & exceptional_types[None, :]))
        predicted = tidy(predicted, after, tagged, useful, hand, counts, heights, pred_exempt)
        pr = beliefs(predicted, counts, hand, tagged)[1]
        next_play = (pr * playable).sum(-1) > .999
        newly_ready = jnp.sum(next_play & ~(pplay[1] > .999) & occupied[1])
        next_trash = (pr * ~useful).sum(-1) > .999
        newly_trash = jnp.sum(next_trash & ~(ptrash[1] > .999) & occupied[1])
        literal_gain = jnp.sum(jnp.log1p(literal[1].sum(-1)) - jnp.log1p(after[1].sum(-1)))
        # Only established connectors authorize delayed clues given by this policy.
        have = own_certain | jnp.any((hand > 0) & touched[1, :, None], axis=0)
        accounted = have.reshape(5, 5) | (jnp.arange(5)[None, :] < heights[:, None])
        connected = jnp.concatenate([jnp.ones((5, 1), jnp.bool_), jnp.cumprod(accounted[:, :4], axis=1).astype(jnp.bool_)], axis=1).reshape(25)
        play_ok = (connected[fid] | (prompt_connector >= 0)) & useful[fid]
        save_ok = (focus == chop) & useful[fid] & ((rank_hint & ((val == 1) | (val == 4))) | ((counts[fid] == 1) & (ranks[fid] != 4)))
        duplicate = jnp.any(same_card & (touch | touched[1])[None, :] & ~jnp.eye(5, dtype=jnp.bool_), axis=1)
        bad = jnp.sum(new & (~partner_live | duplicate | known_duplicate))
        nnew = new.sum()
        truth_consistent = predicted[1, focus, fid]
        new_errors = jnp.sum((~predicted[1, jnp.arange(5), partner_id]) & notes[1, jnp.arange(5), partner_id] & (touch | touched[1]))
        false_plays = jnp.sum(next_play & ~playable[partner_id] & occupied[1])
        required_save = (focus == chop) & chop_danger
        prompt_correct = (prompt_connector < 0) | (prompt_single & (partner_id[jnp.maximum(prompt_connector, 0)] + 1 == fid) & (ranks[fid] > 0))
        valid = prompt_correct & truth_consistent & ((new_errors == 0) | required_save) & (play_ok | save_ok) & ((nnew > 0) | forced) & ((bad == 0) | required_save)
        protects = touch[jnp.minimum(pchop, 4)] & (pchop < 5)
        postpones = (jnp.any(next_play) | jnp.any(next_trash)) & ~partner_ready & ~partner_trash
        safety = urgent & (protects | postpones)
        value = 5.0 + 2.0 * newly_ready + .75 * nnew - 4.0 * bad + .2 * ~rank_hint
        after_chop = chop_of(touched[1] | touch, occupied[1])
        after_id = partner_id[jnp.minimum(after_chop, 4)]
        exposed = (after_chop < 5) & useful[after_id] & (counts[after_id] == 1) & ~jnp.any(next_play) & ~partner_trash
        value += 1000.0 * safety + 18.0 * (proactive_save & protects) + 1.0 * (save_ok & (counts[fid] == 1))
        value -= 6.0 * exposed + 4.0 * new_errors + 8.0 * false_plays
        value -= 2.5 * partner_ready + 1.2 * (nnew == 0)
        value -= .7 * (tokens <= 2) + .6 * (tokens <= 1)
        fallback = -100.0 - 30.0 * (~prompt_correct) - 30.0 * false_plays - 8.0 * new_errors - 2.0 * bad - .2 * nnew + .1 * rank_hint
        # Forced clues may use literal information without inventing a stall
        # signal. Prefer resolving a real play or safe discard to a blind guess.
        forced_information = forced & prompt_correct & truth_consistent & (new_errors == 0) & (false_plays == 0) & (literal_gain > .001)
        information_value = .2 * literal_gain + 4.0 * newly_ready + 3.0 * newly_trash + 1000.0 * safety
        value = jnp.where(valid, value, jnp.where(forced_information, information_value, fallback))
        valid = valid | forced_information
        # Forced legal clues carry no additional convention signal.
        return value, focus, newly_ready, bad, valid

    hint_values, hint_focus, hint_ready, hint_bad, hint_valid = jax.vmap(evaluate_hint)(jnp.arange(10))
    ochop = chop_of(touched[0], occupied[0])
    trash = (ptrash[0] > .999) & occupied[0]
    discard_value = jnp.where(trash, 1.0 + .02 * (4-jnp.arange(5)), -30.0)
    discard_value = jnp.where(jnp.arange(5) == ochop, .0, discard_value)
    all_clued = ochop == 5
    loss = (probs[0] * useful * (1 + (counts == 1) * (5-ranks))).sum(-1)
    discard_value = jnp.where(all_clued & ~trash, -2.0 - loss, discard_value)
    # A connector play can itself give the partner a safe action and protect chop.
    boards_after = heights[None, :] + jax.nn.one_hot(colors, 5, dtype=jnp.int32) * playable[:, None]
    playable_after = ranks[None, :] == boards_after[:, colors]
    other_after = jnp.einsum('st,it->is', probs[1], playable_after.astype(jnp.float32))
    protected_after = jnp.any((other_after > .999) & occupied[1][None, :], axis=-1)
    play_protects = (probs[0] * protected_after[None, :]).sum(-1) > .999
    newly_ready_after = jnp.sum((other_after > .999) & ~(pplay[1] > .999)[None, :] & occupied[1][None, :], axis=-1)
    unlock = (probs[0] * newly_ready_after[None, :]).sum(-1)
    play_value = 16.0 + .6 * (tokens < 8) * (probs[0] * (ranks == 4)).sum(-1) + .02 * (4-jnp.arange(5))
    play_value += 1000.0 * urgent * play_protects + 18.0 * proactive_save * (unlock > .999) + 1.5 * unlock
    safe = pplay[0] > .999
    threshold = jnp.where(lives >= 3, .985, jnp.where(lives >= 2, .995, .9999))
    risky = (pplay[0] >= threshold) & touched[0]
    play_value = jnp.where(safe, play_value, jnp.where(risky, 2.0 + pplay[0], -10000.0))
    # At the final two turns, unused information has no value beyond enabling a play.
    end_value = pplay[0] - (1-pplay[0]) * jnp.where(lives > 1, 0.0, heights.sum())
    play_value = jnp.where((deck == 0) & ~safe, jnp.where(end_value > 0, 2*end_value+2, -10000), play_value)
    first_final_turn = (deck == 0) & (m['previous'][127:167].sum() > 0)
    final_hint = (hint_ready > 0) & hint_valid & ~partner_ready & first_final_turn
    hint_values = jnp.where(deck == 0, jnp.where(final_hint, 4.0, -500.0), hint_values)
    values = jnp.concatenate([discard_value, play_value, hint_values, jnp.array([0.0])])
    action = jnp.argmax(jnp.where(obs.legal_actions, values, -1e6)).astype(jnp.int32)
    out = {'notes': notes, 'previous': f, 'early': early, 'dependency': dependencies, 'exempt': exempt, 'wait': wait}
    trace = {'wait_count': wait.sum((-1, -2)), 'broken_delay': broken_delay.sum(-1), 'count_consistent': count_consistent, 'pplay': pplay, 'ptrash': ptrash, 'notes': notes, 'hint_values': hint_values,
             'hint_focus': hint_focus, 'hint_valid': hint_valid, 'hint_bad': hint_bad,
             'dependency': dependencies, 'last_connector': last_connector, 'last_linkable': last_linkable, 'last_focus': last_focus, 'last_save': last_save, 'urgent': urgent, 'early': early}
    return action, out, trace
