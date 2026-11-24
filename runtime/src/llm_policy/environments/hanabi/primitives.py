"""Observation-only primitives for standard 2–5 player Hanabi.

Beliefs condition on literal evidence under uniform physical-card allocation,
not on a model of why a partner chose an action. No helper selects a convention.
"""

import itertools

import chex
import jax
import jax.numpy as jnp

from llm_policy.policies.types import PolicyTrace
from llm_policy.policies.api import HanabiObservation
from llm_policy.environments.hanabi.layout import hanabi_layout, observation_layout

PLAY, DISCARD, HINT_COLOR, HINT_RANK, NOOP = range(5)


@chex.dataclass
class HanabiFacts:
    partner_hand: jax.Array
    occupied: jax.Array
    fireworks: jax.Array
    discards: jax.Array
    deck_remaining: jax.Array
    info_tokens: jax.Array
    life_tokens: jax.Array
    is_my_turn: jax.Array
    legal_actions: jax.Array
    has_last_action: jax.Array
    last_actor: jax.Array
    last_kind: jax.Array
    last_target: jax.Array
    last_color: jax.Array
    last_rank: jax.Array
    last_matches: jax.Array
    last_slot: jax.Array
    last_card: jax.Array
    last_play_scored: jax.Array
    last_play_added_token: jax.Array
    upstream_v0: jax.Array
    positive_colors: jax.Array
    positive_ranks: jax.Array
    other_hands: jax.Array = None


@chex.dataclass
class HanabiKnowledge:
    possible: jax.Array
    previous_deck_remaining: jax.Array
    episode_step: jax.Array


@chex.dataclass
class HanabiBelief:
    probabilities: jax.Array
    feasible: jax.Array
    valid: jax.Array
    normalizer: jax.Array


@chex.dataclass
class HanabiHandSample:
    """One joint identity assignment; empty slots and invalid hands are zero."""
    hand: jax.Array
    valid: jax.Array
    log_probability: jax.Array


@chex.dataclass
class HanabiBeliefContext:
    """Optional literal reference calculations, not a learned partner model."""
    own_literal_belief: HanabiBelief
    own_hand_samples: HanabiHandSample
    partner_literal_beliefs: HanabiBelief
    valid: jax.Array


def decode_observation(obs):
    """Decode native features; all player axes are observer-relative (self=0)."""
    layout = observation_layout(obs.features.shape[-1])
    n, h = layout["num_agents"], layout["hand_size"]
    if obs.legal_actions.shape != (layout["num_actions"],):
        raise ValueError("Hanabi observation/action dimensions disagree")
    x = obs.features
    def block(name):
        start, end = layout["observation_blocks"][name]
        return x[start:end]
    discard_bits = block("discards").reshape(5, 10)
    discards = jnp.stack([discard_bits[:, :3].sum(-1),
                          discard_bits[:, 3:5].sum(-1),
                          discard_bits[:, 5:7].sum(-1),
                          discard_bits[:, 7:9].sum(-1), discard_bits[:, 9]], axis=-1)
    last = block("last_action")
    def event(name):
        start, end = layout["last_action_blocks"][name]
        return last[start:end]
    belief = block("native_knowledge").reshape(n, h, 35)
    others = block("other_hands").reshape(n-1, h, 5, 5).astype(jnp.int32)
    return HanabiFacts(
        partner_hand=others[0], other_hands=others,
        occupied=jnp.any(belief[..., :25] != 0, axis=-1),
        fireworks=(block("fireworks").reshape(5,5)*jnp.arange(1,6)).sum(-1).astype(jnp.int32),
        discards=discards.astype(jnp.int32),
        deck_remaining=block("deck").sum().astype(jnp.int32),
        info_tokens=block("information_tokens").sum().astype(jnp.int32),
        life_tokens=block("lives").sum().astype(jnp.int32),
        is_my_turn=~obs.legal_actions[-1].astype(jnp.bool_),
        legal_actions=obs.legal_actions.astype(jnp.bool_),
        has_last_action=jnp.any(event("actor") != 0),
        last_actor=jnp.argmax(event("actor")).astype(jnp.int32),
        last_kind=jnp.argmax(event("kind")).astype(jnp.int32),
        last_target=jnp.argmax(event("target")).astype(jnp.int32),
        last_color=jnp.argmax(event("color")).astype(jnp.int32),
        last_rank=jnp.argmax(event("rank")).astype(jnp.int32),
        last_matches=event("matches").astype(jnp.bool_),
        last_slot=jnp.argmax(event("slot")).astype(jnp.int32),
        last_card=event("card").reshape(5,5).astype(jnp.int32),
        last_play_scored=event("scored")[0].astype(jnp.bool_),
        last_play_added_token=event("added_token")[0].astype(jnp.bool_),
        upstream_v0=belief[..., :25].reshape(n,h,5,5),
        positive_colors=belief[...,25:30], positive_ranks=belief[...,30:35])


def play(slot, *, hand_size=5):
    return jnp.asarray(slot, jnp.int32) + hand_size


def discard(slot):
    return jnp.asarray(slot, jnp.int32)


def _hint(value, target, start, num_agents):
    valid = (target >= 1) & (target < num_agents) & (value >= 0) & (value < 5)
    return jnp.where(valid, start + (target-1)*5 + value, -1).astype(jnp.int32)


def hint_color(color, *, target=1, num_agents=2, hand_size=None):
    layout = hanabi_layout(num_agents, hand_size)
    return _hint(jnp.asarray(color), jnp.asarray(target), layout["actions"]["hint_color"][0], num_agents)


def hint_rank(rank, *, target=1, num_agents=2, hand_size=None):
    layout = hanabi_layout(num_agents, hand_size)
    return _hint(jnp.asarray(rank), jnp.asarray(target), layout["actions"]["hint_rank"][0], num_agents)


def noop(*, num_agents=2, hand_size=None):
    return jnp.asarray(hanabi_layout(num_agents, hand_size)["actions"]["inactive_noop"], jnp.int32)


def decode_action(action, *, num_agents=2, hand_size=None):
    """Return (kind, slot/color/rank); use action_target for a hint recipient."""
    layout = hanabi_layout(num_agents, hand_size)
    h, a = layout["hand_size"], jnp.asarray(action, jnp.int32)
    ranks, stop = layout["actions"]["hint_rank"][0], layout["actions"]["inactive_noop"]
    kind = jnp.where(a < h, DISCARD, jnp.where(a < 2*h, PLAY,
                     jnp.where(a < ranks, HINT_COLOR, jnp.where(a < stop, HINT_RANK, NOOP))))
    detail = jnp.where(a < h, a, jnp.where(a < 2*h, a-h,
                       jnp.where(a < ranks, (a-2*h)%5, jnp.where(a < stop, (a-ranks)%5, 0))))
    return kind.astype(jnp.int32), detail.astype(jnp.int32)


def action_target(action, *, num_agents=2, hand_size=None):
    """Observer-relative hint recipient, or -1 for non-hint actions."""
    layout = hanabi_layout(num_agents, hand_size)
    a = jnp.asarray(action, jnp.int32)
    colors, ranks = layout["actions"]["hint_color"][0], layout["actions"]["hint_rank"][0]
    stop = layout["actions"]["inactive_noop"]
    return jnp.where((a >= colors) & (a < stop),
                     (a-jnp.where(a < ranks, colors, ranks))//5+1, -1).astype(jnp.int32)


def legal_actions(facts):
    return facts.legal_actions


def is_legal(mask, action):
    return (action >= 0) & (action < mask.shape[0]) & mask[jnp.clip(action, 0, mask.shape[0]-1)]


def masked_argmax(values, mask):
    """Lowest-index tie break; returns -1 when there is no legal action."""
    return jnp.where(jnp.any(mask), jnp.argmax(jnp.where(mask, values, -jnp.inf)), -1).astype(jnp.int32)


def sample_legal(mask, rng):
    """Uniform legal random action, not a tactical fallback."""
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jnp.where(jnp.any(mask), jax.random.categorical(rng, logits), -1).astype(jnp.int32)


def init_knowledge(num_agents=2, hand_size=None):
    layout = hanabi_layout(num_agents, hand_size)
    return HanabiKnowledge(possible=jnp.ones((num_agents, layout["hand_size"], 5, 5), jnp.bool_),
                           previous_deck_remaining=jnp.asarray(layout["initial_deck_size"], jnp.int32),
                           episode_step=jnp.asarray(0, jnp.int32))


def apply_hint_constraints(possible, matches, kind, value):
    """Intersect [slot,color,rank] masks with positive AND negative evidence."""
    identity_matches = jnp.where(kind == HINT_COLOR,
                                 jnp.arange(5)[:, None] == value,
                                 jnp.arange(5)[None, :] == value)
    return possible & (identity_matches[None, :, :] == matches[:, None, None])


def hint_matches(visible_hand, kind, value):
    identities = jnp.where(kind == HINT_COLOR, jnp.arange(5)[:, None] == value,
                           jnp.arange(5)[None, :] == value)
    return jnp.any(visible_hand.astype(jnp.bool_) & identities, axis=(-2, -1))


def literal_hint_effect(possible, visible_hand, kind, value):
    matches = hint_matches(visible_hand, kind, value)
    return matches, apply_hint_constraints(possible, matches, kind, value)


def remap_after_card_removal(per_slot, slot, replacement):
    """Shift arbitrary [H,...] private card-associated data and append a value."""
    h = per_slot.shape[0]
    indices = jnp.minimum(jnp.arange(h) + (jnp.arange(h) >= slot), h-1)
    return per_slot[indices].at[h-1].set(replacement)


def update_knowledge(facts, reset, knowledge):
    """Call exactly once per global environment transition, including others' turns.

    Masks follow surviving cards; only a newly drawn card loses old hints.
    All inputs are public events or this observer's own history.
    """
    reset = jnp.asarray(reset, jnp.bool_)
    initial = init_knowledge(*knowledge.possible.shape[:2])
    k = select_pytree(reset, initial, knowledge)
    masks = k.possible
    hint = (facts.last_kind == HINT_COLOR) | (facts.last_kind == HINT_RANK)
    value = jnp.where(facts.last_kind == HINT_COLOR, facts.last_color, facts.last_rank)
    hinted = apply_hint_constraints(masks[facts.last_target], facts.last_matches,
                                    facts.last_kind, value)
    removed = remap_after_card_removal(masks[facts.last_actor], facts.last_slot,
                                       jnp.full((5, 5), k.previous_deck_remaining > 0))
    changed = jax.lax.cond(hint, lambda m: m.at[facts.last_target].set(hinted),
                           lambda m: m.at[facts.last_actor].set(removed), masks)
    masks = jnp.where(facts.has_last_action & ~reset, changed, masks)
    masks = masks & facts.occupied[..., None, None]
    return HanabiKnowledge(possible=masks, previous_deck_remaining=facts.deck_remaining,
                           episode_step=k.episode_step + jnp.asarray(1, jnp.int32))


def observe(obs, reset, knowledge):
    facts = decode_observation(obs)
    return facts, update_knowledge(facts, reset, knowledge)


def public_remaining_counts(facts):
    played = jnp.arange(5)[None, :] < facts.fireworks[:, None]
    return jnp.array([3, 2, 2, 2, 1], jnp.int32)[None, :] - facts.discards - played.astype(jnp.int32)


def unseen_counts(facts):
    """Counts in this observer's hidden hand plus undealt deck, not deck alone."""
    return public_remaining_counts(facts) - _other_hands(facts).sum(axis=(0, 1))


def _other_hands(facts):
    return facts.partner_hand[None] if facts.other_hands is None else facts.other_hands


def _partition_indices(hand_size):
    # Static host tables only: disjoint previous/current/later type subsets.
    partitions = tuple(itertools.product(range(3), repeat=hand_size))
    left = tuple(sum(1 << i for i, v in enumerate(p) if v == 0) for p in partitions)
    take = tuple(sum(1 << i for i, v in enumerate(p) if v == 1) for p in partitions)
    return jnp.asarray(left), jnp.asarray(take)


def _literal_hand_tables(possible, counts, occupied):
    """Shared subset partition function for marginals and joint sampling."""
    h = possible.shape[0]
    support = possible.reshape(h, 25).astype(jnp.bool_) & occupied[:, None]
    slots = (jnp.arange(1 << h)[:, None] & (1 << jnp.arange(h))[None, :]) != 0
    sizes = slots.sum(axis=-1)
    c = counts.reshape(25).astype(jnp.float32)
    falling = jnp.concatenate((jnp.ones((25, 1)),
                               jnp.cumprod(jnp.maximum(c[:, None] - jnp.arange(h), 0), axis=1)), axis=1)
    allowed = jnp.all(~slots[None, :, :] | support.T[:, None, :], axis=-1)
    weights = falling[:, sizes] * allowed
    left, take = _partition_indices(h)
    union = left | take
    initial = jnp.zeros(1 << h).at[0].set(1.0)

    def step(previous, weight):
        nxt = jnp.zeros(1 << h).at[union].add(previous[left] * weight[take])
        return nxt, nxt

    _, forward = jax.lax.scan(step, initial, weights)
    full = jnp.sum(occupied.astype(jnp.int32) * (1 << jnp.arange(h)))
    z = forward[-1, full]
    valid = (z > 0) & jnp.isfinite(z) & jnp.all(jnp.isfinite(c)) & jnp.all(c >= 0) & jnp.all(c == jnp.floor(c))
    return slots, weights, left, take, initial, forward, full, valid


def literal_hand_marginals(possible, counts, occupied):
    """Exact literal-evidence marginals [H,5,5] under sampling without replacement.

    possible: bool[H,5,5]; counts: nonnegative integer[5,5]; occupied: bool[H].
    No assumptions about hint intent or partner strategy. Invalid evidence gives
    valid=False and zero arrays, never a uniform or V0 fallback.
    """
    h = possible.shape[0]
    slots, weights, left, take, initial, forward, full, valid = _literal_hand_tables(
        possible, counts, occupied)
    union = left | take

    def step(previous, weight):
        nxt = jnp.zeros(1 << h).at[union].add(previous[left] * weight[take])
        return nxt, nxt

    _, backward_reversed = jax.lax.scan(step, initial, weights[::-1])
    prefix = jnp.concatenate((initial[None], forward[:-1]), axis=0)
    suffix = jnp.concatenate((backward_reversed[-2::-1], initial[None]), axis=0)
    rest = full & ~union
    partition_valid = (union & ~full) == 0
    mass = prefix[:, left] * weights[:, take] * suffix[:, rest] * partition_valid
    z = forward[-1, full]
    numerators = jnp.einsum("kp,pi->ik", mass, slots[take].astype(jnp.float32))
    probabilities = jnp.where(valid, numerators / jnp.where(z > 0, z, 1), 0)
    feasible = (numerators > 0) & valid
    return HanabiBelief(probabilities=probabilities.reshape(h, 5, 5),
                        feasible=feasible.reshape(h, 5, 5), valid=valid, normalizer=z)


def sample_literal_hand(possible, counts, occupied, rng):
    """Sample an exact joint hand, not independent draws from slot marginals.

    Same evidence contract as literal_hand_marginals. Returns HanabiHandSample:
    integer one-hot hand[H,5,5], scalar valid and log_probability of the identity
    assignment (not of distinct physical copies). Invalid evidence returns zeros
    and -inf. Sampling backtracks the subset DP, retaining all slot constraints.
    """
    h = possible.shape[0]
    slots, weights, _, _, initial, forward, full, valid = _literal_hand_tables(
        possible, counts, occupied)
    prefix = jnp.concatenate((initial[None], forward[:-1]), axis=0)
    subsets = jnp.arange(1 << h)
    keys = jax.random.split(rng, 25)

    def step(carry, card):
        remaining, hand, log_probability = carry
        allowed = (subsets & ~remaining) == 0
        masses = jnp.where(allowed, weights[card] * prefix[card, remaining & ~subsets], 0)
        total = masses.sum()
        usable = valid & (total > 0) & jnp.isfinite(total)
        # A finite dummy categorical keeps invalid inputs safe under jit/vmap.
        logits = jnp.where(usable, jnp.where(masses > 0, jnp.log(masses), -jnp.inf),
                           jnp.where(subsets == 0, 0., -jnp.inf))
        chosen = jax.random.categorical(keys[card], logits)
        hand = hand.at[:, card].set(slots[chosen].astype(jnp.int32))
        log_probability += jnp.where(usable, jnp.log(jnp.maximum(masses[chosen], 1e-30))
                                     - jnp.log(jnp.maximum(total, 1e-30)), 0.)
        return (remaining & ~chosen, hand, log_probability), None

    (remaining, hand, log_probability), _ = jax.lax.scan(
        step, (full, jnp.zeros((h, 25), jnp.int32), jnp.asarray(0., jnp.float32)),
        jnp.arange(24, -1, -1))
    valid = valid & (remaining == 0)
    return HanabiHandSample(hand=jnp.where(valid, hand.reshape(h, 5, 5), 0),
                            valid=valid, log_probability=jnp.where(valid, log_probability, -jnp.inf))


def own_hand_belief(facts, knowledge):
    """Own exact literal belief, including subtraction of all visible hands."""
    return literal_hand_marginals(knowledge.possible[0], unseen_counts(facts), facts.occupied[0])


def sample_own_hand(facts, knowledge, rng):
    """Joint draw from own_hand_belief's underlying literal hand distribution."""
    return sample_literal_hand(knowledge.possible[0], unseen_counts(facts), facts.occupied[0], rng)


def partner_hand_belief(facts, knowledge, hypothetical_own_hand, *, target=1):
    """Partner's literal own-hand belief CONDITIONAL on a hypothetical own hand.

    Supply a sampled/constructed one-hot integer[H,5,5] hand, never hidden runtime
    truth. The partner sees that hand: subtract it from PUBLIC remaining counts,
    not from unseen_counts(facts). Also subtract every other visible hand except
    the selected target's. Do not subtract their actual hand, which they
    cannot see. This is not their actual private/intent-conditioned belief.
    An own-hand hypothesis inconsistent with current literal evidence is invalid.
    """
    hand = jnp.asarray(hypothetical_own_hand)
    used = hand.sum(axis=0)
    compatible = (jnp.all((hand == 0) | (hand == 1))
                  & jnp.all(hand.sum(axis=(-2, -1)) == facts.occupied[0])
                  & jnp.all((hand == 0) | knowledge.possible[0])
                  & jnp.all(used <= unseen_counts(facts)))
    others = _other_hands(facts)
    target_valid = (target >= 1) & (target < knowledge.possible.shape[0])
    target_index = jnp.clip(jnp.asarray(target), 1, knowledge.possible.shape[0]-1)
    visible_elsewhere = others.sum(axis=(0, 1)) - others[target_index-1].sum(axis=0)
    belief = literal_hand_marginals(knowledge.possible[target_index],
                                    public_remaining_counts(facts) - used - visible_elsewhere,
                                    facts.occupied[target_index])
    valid = target_valid & compatible & belief.valid
    return HanabiBelief(probabilities=jnp.where(valid, belief.probabilities, 0),
                        feasible=belief.feasible & valid, valid=valid,
                        normalizer=jnp.where(valid, belief.normalizer, 0))


def build_belief_context(facts, knowledge, rng, *, num_samples, target=1):
    """Bundle optional own/partner literal calculations; sample count is static.

    partner_literal_beliefs has a leading sample axis, one belief conditional on
    each own_hand_samples.hand. valid[N] masks usable pairs. These are independent
    draws from the literal distribution, not persistent particles or intent models.
    No sampling is performed unless the policy explicitly calls this helper.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive static Python integer")
    own = own_hand_belief(facts, knowledge)
    samples = jax.vmap(sample_own_hand, in_axes=(None, None, 0))(
        facts, knowledge, jax.random.split(rng, num_samples))
    partner_literal_beliefs = jax.vmap(lambda hand: partner_hand_belief(facts, knowledge, hand, target=target))(
        samples.hand)
    return HanabiBeliefContext(own_literal_belief=own, own_hand_samples=samples,
                               partner_literal_beliefs=partner_literal_beliefs,
                               valid=own.valid & samples.valid & partner_literal_beliefs.valid)


def init_partner_belief_model_template():
    """Optional copy-and-edit scaffold for custom_primitives.py; no learned state.

    Replace the empty tuple with your own fixed-shape JAX PyTree if needed. Store
    it in the policy's private memory, separately from literal HanabiKnowledge.
    There are deliberately no predefined partner types, conventions or priors.
    """
    return ()


def update_partner_belief_model_template(facts, reset, belief_context, model_state):
    """Literal-only reference implementation, NOT strategic partner inference.

    Copy into custom_primitives.py to implement your own partner model. This
    default returns the conditional literal beliefs unchanged and learns nothing.
    A custom version may extend/reweight/replace them using permitted history;
    its model representation and output need not be HanabiBelief.

    When adding state: reset it on reset=True, retain fixed PyTree/shapes/dtypes,
    and process each observed transition once (including inactive turns). Preserve
    any pre-action context needed to interpret the next event; current beliefs
    are post-observation, not automatically the beliefs preceding that action.
    Samples are freshly drawn hypotheses: index i is not a persistent identity.
    Distinguish what the partner believes from how they choose actions. Do not
    treat an inferred convention as observed fact or assume access to their memory.
    """
    return belief_context.partner_literal_beliefs, model_state


def feasible_card_mask(belief):
    return belief.feasible


def card_properties(facts):
    """Identity masks, not action recommendations. 'Critical' means last useful copy."""
    ranks = jnp.arange(5)[None, :]
    played = ranks < facts.fireworks[:, None]
    available = public_remaining_counts(facts)
    lost_prerequisite = (available == 0) & ~played
    unreachable = jnp.cumsum(lost_prerequisite, axis=-1) > 0
    return {"playable": ranks == facts.fireworks[:, None], "already_played": played,
            "unreachable": unreachable, "critical": (available == 1) & ~played & ~unreachable}


def card_property_probabilities(belief, facts):
    """Returns probabilities plus valid flag; invalid beliefs must not guide actions."""
    return {**{name: (belief.probabilities * mask).sum(axis=(-2, -1))
               for name, mask in card_properties(facts).items()}, "valid": belief.valid}


def select_pytree(condition, yes, no):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(condition, a, b), yes, no)


def make_trace(action, phase_id=0, decision_id=0, *, num_agents=2, hand_size=None):
    kind, detail = decode_action(action, num_agents=num_agents, hand_size=hand_size)
    return PolicyTrace(phase_id=jnp.asarray(phase_id, jnp.int32),
                       decision_id=jnp.asarray(decision_id, jnp.int32), skill_id=kind,
                       target_type=kind, target_detail=detail, status=jnp.asarray(1, jnp.int32),
                       action=jnp.asarray(action, jnp.int32), movement_blocked=jnp.asarray(False))
