"""Pure observation-only Yokai helpers; no signaling or hint-trust convention."""

import chex
from itertools import combinations
import jax
import jax.numpy as jnp

from llm_policy.policies.types import PolicyTrace
from llm_policy.policies.api import YokaiObservation


@chex.dataclass
class YokaiFacts:
    visible_colors: jax.Array
    hint_colors: jax.Array
    positions: jax.Array
    hint_positions: jax.Array
    locked: jax.Array
    hints_placed: jax.Array
    seen_by_partner: jax.Array
    active: jax.Array
    substep: jax.Array
    adjacency_matrix: jax.Array
    legal_actions: jax.Array


@chex.dataclass
class YokaiMemory:
    known_colors: jax.Array


@chex.dataclass
class YokaiComponents:
    labels: jax.Array  # Minimum card ID in component; -1 outside mask.
    sizes: jax.Array  # Component size for each card; zero outside mask.
    count: jax.Array


@chex.dataclass
class YokaiBelief:
    assignments: jax.Array  # int32[1680,9], exactly three copies of each color.
    probabilities: jax.Array  # float32[1680]; zero on inconsistent evidence.
    valid: jax.Array


@chex.dataclass
class YokaiEvents:
    newly_revealed_hints: jax.Array
    newly_placed_hints: jax.Array
    newly_locked_cards: jax.Array
    moved_cards: jax.Array
    visible_color_changes: jax.Array
    newly_seen_by_partner: jax.Array


# Static combinatorics only: no hidden state, file access, or environment imports.
_COLOR_ASSIGNMENTS = tuple(
    tuple(0 if i in red else 1 if i in blue else 2 for i in range(9))
    for red in combinations(range(9), 3)
    for blue in combinations(tuple(i for i in range(9) if i not in red), 3)
)


def card_adjacency(positions):
    """Orthogonal card-only edges, without self-loops or hint nodes."""
    positions = jnp.asarray(positions, jnp.int32)
    return jnp.abs(positions[:, None, :] - positions[None, :, :]).sum(-1) == 1


def connected_components(adjacency, mask):
    """Undirected induced components; fixed-shape boolean transitive closure."""
    mask = jnp.asarray(mask, jnp.bool_)
    n = mask.shape[0]
    adjacency = jnp.asarray(adjacency, jnp.bool_)
    reach = (adjacency | adjacency.T | jnp.eye(n, dtype=jnp.bool_)) & mask[:, None] & mask[None, :]
    def expand(k, r):
        return r | (r[:, k, None] & r[None, k, :])
    reach = jax.lax.fori_loop(0, n, expand, reach)
    ids = jnp.arange(n, dtype=jnp.int32)
    labels = jnp.where(mask, jnp.min(jnp.where(reach, ids[None, :], n), axis=1), -1)
    return YokaiComponents(labels=labels, sizes=reach.sum(1).astype(jnp.int32),
                           count=jnp.sum(mask & (labels == ids)).astype(jnp.int32))


def is_connected(adjacency, mask):
    """False for an empty selection, true for a singleton."""
    return connected_components(adjacency, mask).count == 1


def color_components(positions, colors):
    """Batched components for colors 0..2; unknown -1 cards are excluded."""
    colors = jnp.asarray(colors)
    adjacency = card_adjacency(positions)
    return jax.vmap(lambda color: connected_components(adjacency, colors == color))(jnp.arange(3))


def is_solved(positions, colors):
    """Exact nine-card goal for a complete, count-consistent assignment only."""
    colors = jnp.asarray(colors, jnp.int32)
    complete = jnp.all(jnp.sum(colors[None, :] == jnp.arange(3)[:, None], axis=1) == 3)
    return complete & jnp.all(color_components(positions, colors).count == 1)


def literal_card_belief(known_colors):
    """Uniform shuffled-deck prior conditioned only on personally known colors."""
    assignments = jnp.asarray(_COLOR_ASSIGNMENTS, jnp.int32)
    prior = YokaiBelief(assignments=assignments,
        probabilities=jnp.full((1680,), 1. / 1680, jnp.float32), valid=jnp.asarray(True))
    return condition_on_colors(prior, known_colors)


def reweight_belief(belief, log_likelihood):
    """Apply caller-supplied assignment likelihoods; -inf excludes a hypothesis.

    NaN/+inf, negative/nonfinite input weights, or zero remaining mass produce
    valid=False and zero weights. No implicit fallback or hint-trust model.
    """
    likelihood = jnp.asarray(log_likelihood, jnp.float32)
    likelihood = jnp.broadcast_to(likelihood, belief.probabilities.shape)
    probs = belief.probabilities
    inputs_ok = (belief.valid & jnp.all(jnp.isfinite(probs) & (probs >= 0))
                 & jnp.all(jnp.isfinite(likelihood) | jnp.isneginf(likelihood)))
    log_weights = jnp.where(probs > 0, jnp.log(jnp.maximum(probs, jnp.finfo(jnp.float32).tiny)), -jnp.inf) + likelihood
    peak = jnp.max(log_weights)
    weights = jnp.exp(log_weights - jnp.where(jnp.isfinite(peak), peak, 0.))
    total = weights.sum()
    valid = inputs_ok & jnp.isfinite(total) & (total > 0)
    return YokaiBelief(assignments=belief.assignments,
        probabilities=jnp.where(valid, weights / jnp.where(valid, total, 1.), 0.), valid=valid)


def condition_on_colors(belief, observed_colors):
    """Hard observed identities; -1 means no new evidence. Does not use hints."""
    observed = jnp.asarray(observed_colors)
    admissible = jnp.all((observed == -1) | (observed == 0) | (observed == 1) | (observed == 2))
    matches = jnp.all((observed[None, :] == -1) | (belief.assignments == observed[None, :]), axis=1)
    return reweight_belief(belief, jnp.where(admissible & matches, 0., -jnp.inf))


def card_marginals(belief):
    """Derived [9,3] probabilities; all zeros when belief.valid is false."""
    indicators = jax.nn.one_hot(belief.assignments, 3, dtype=jnp.float32)
    return jnp.where(belief.valid, jnp.einsum('a,anc->nc', belief.probabilities, indicators), 0.)


def sample_assignment(belief, rng):
    """Optional joint draw; returns int32[9] of -1 if belief is invalid."""
    logits = jnp.where(belief.probabilities > 0, jnp.log(jnp.maximum(belief.probabilities, jnp.finfo(jnp.float32).tiny)), -jnp.inf)
    index = jax.random.categorical(rng, jnp.where(belief.valid, logits, jnp.zeros_like(logits)))
    return jnp.where(belief.valid, belief.assignments[index], -1)


def solved_probability(positions, belief):
    """Exact goal probability under the supplied joint belief; NaN if invalid."""
    solved = jax.vmap(lambda colors: is_solved(positions, colors))(belief.assignments)
    return jnp.where(belief.valid, jnp.sum(belief.probabilities * solved), jnp.nan)


def hint_card_links(facts):
    """bool[4,9] attachment matrix based on public placements and positions."""
    return facts.hints_placed[:, None] & jnp.all(facts.hint_positions[:, None, :] == facts.positions[None, :, :], axis=-1)


def hint_correct_probability(belief, hint_colors, card):
    """Probability listed colors contain this card; no trust update is applied.

    hint_colors is bool[3], not a hint ID. Unknown all-false colors, invalid card
    indices or an invalid belief return NaN rather than a misleading zero.
    """
    colors = jnp.asarray(hint_colors, jnp.bool_)
    valid = belief.valid & jnp.any(colors) & (card >= 0) & (card < 9)
    probability = jnp.sum(card_marginals(belief)[jnp.clip(card, 0, 8)] * colors)
    return jnp.where(valid, probability, jnp.nan)


def init_observation_snapshot():
    """Empty fixed-shape previous facts for optional event tracking."""
    return YokaiFacts(visible_colors=jnp.full((9,), -1, jnp.int32),
        hint_colors=jnp.zeros((4, 3), jnp.bool_), positions=jnp.full((9, 2), -1, jnp.int32),
        hint_positions=jnp.full((4, 2), -1, jnp.int32), locked=jnp.zeros((9,), jnp.bool_),
        hints_placed=jnp.zeros((4,), jnp.bool_), seen_by_partner=jnp.zeros((9,), jnp.bool_),
        active=jnp.asarray(False), substep=jnp.asarray(0, jnp.int32),
        adjacency_matrix=jnp.zeros((13, 13), jnp.uint8), legal_actions=jnp.zeros((1068,), jnp.bool_))


def observation_changes(previous, current, reset=False):
    """Deltas between consecutive local observations; no inferred intent.

    On reset only currently visible colors are reported. Save current as the
    next snapshot on EVERY global transition, including inactive turns. A color
    reappearing after masking can be reported again; this is not a new peek.
    """
    ongoing = ~jnp.asarray(reset, jnp.bool_)
    return YokaiEvents(
        newly_revealed_hints=ongoing & current.hint_colors.any(-1) & ~previous.hint_colors.any(-1),
        newly_placed_hints=ongoing & current.hints_placed & ~previous.hints_placed,
        newly_locked_cards=ongoing & current.locked & ~previous.locked,
        moved_cards=ongoing & jnp.any(current.positions != previous.positions, axis=-1),
        visible_color_changes=(current.visible_colors >= 0) & ((current.visible_colors != previous.visible_colors) | ~ongoing),
        newly_seen_by_partner=ongoing & current.seen_by_partner & ~previous.seen_by_partner)


def init_belief_model_template():
    """Optional custom model state. Copy/edit alongside the update template."""
    return ()


def update_belief_model(facts, events, literal_belief, model_state):
    """Neutral copy-and-edit template; adds no interpretation or likelihood."""
    return literal_belief, model_state


def legal_move_mask(facts):
    """Native current-substep legality, shaped [card,x,y]."""
    return facts.legal_actions[10:739].reshape((9, 9, 9))


def legal_hint_placement_mask(facts):
    """Native current-substep legality, shaped [hint,card]."""
    grid = facts.legal_actions[743:1067].reshape((4, 9, 9))
    positions = facts.positions
    in_bounds = jnp.all((positions >= 0) & (positions < 9), axis=-1)
    return grid[:, jnp.clip(positions[:, 0], 0, 8), jnp.clip(positions[:, 1], 0, 8)] & in_bounds[None, :]


def hypothetical_move(positions, card, x, y):
    """Geometry only: return (new_positions, adjacency); caller checks legality.

    Requires card 0..8 and grid coordinates 0..8. Does not simulate hidden colors,
    turns, hints, or observations, and cannot check locks from positions alone.
    """
    moved = jnp.asarray(positions, jnp.int32).at[card].set(jnp.asarray([x, y], jnp.int32))
    return moved, card_adjacency(moved)


def decode_observation(obs):
    """Color -1 means unknown; card rows are stable IDs 0..8, hints 0..3."""
    nodes = obs.node_features
    colors = nodes[:9, :3]
    return YokaiFacts(
        visible_colors=jnp.where(colors.any(axis=1), jnp.argmax(colors, axis=1), -1).astype(jnp.int32),
        hint_colors=nodes[9:, 3:6].astype(jnp.bool_),
        positions=obs.positions,
        hint_positions=nodes[9:, 6:8].astype(jnp.int32),
        locked=nodes[:9, 8].astype(jnp.bool_),
        hints_placed=nodes[9:, 8].astype(jnp.bool_),
        seen_by_partner=nodes[:9, 9].astype(jnp.bool_),
        active=nodes[0, 10].astype(jnp.bool_),
        substep=nodes[0, 11].astype(jnp.int32),
        adjacency_matrix=obs.adjacency_matrix,
        legal_actions=obs.legal_actions,
    )


def init_memory():
    return YokaiMemory(known_colors=jnp.full((9,), -1, jnp.int32))


def observe(obs, reset, memory):
    """Remember only colors this player observed; moving a card keeps its ID."""
    facts = decode_observation(obs)
    previous = jnp.where(reset, -1, memory.known_colors)
    known = jnp.where(facts.visible_colors >= 0, facts.visible_colors, previous)
    return facts, YokaiMemory(known_colors=known.astype(jnp.int32))


def observe_card(card):
    return jnp.asarray(1 + card, jnp.int32)


def move_card(card, x, y):
    return jnp.asarray(10 + card * 81 + x * 9 + y, jnp.int32)


def reveal_hint(hint):
    return jnp.asarray(739 + hint, jnp.int32)


def place_hint(hint, x, y):
    return jnp.asarray(743 + hint * 81 + x * 9 + y, jnp.int32)


def end_game():
    return jnp.asarray(1067, jnp.int32)


def noop():
    return jnp.asarray(0, jnp.int32)


def is_legal(legal_actions, action):
    return (action >= 0) & (action < 1068) & legal_actions[jnp.clip(action, 0, 1067)]


def sample_legal(legal_actions, rng):
    return jax.random.categorical(rng, jnp.where(legal_actions, 0., -jnp.inf)).astype(jnp.int32)


def make_trace(action, phase_id=0, decision_id=0):
    zero = jnp.asarray(0, jnp.int32)
    return PolicyTrace(phase_id=jnp.asarray(phase_id, jnp.int32),
        decision_id=jnp.asarray(decision_id, jnp.int32), skill_id=zero,
        target_type=zero, target_detail=zero, status=zero,
        action=jnp.asarray(action, jnp.int32), movement_blocked=jnp.asarray(False))
