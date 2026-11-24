"""Pure observation-only helpers for two-player, sixteen-card Yokai."""

import chex
import jax
import jax.numpy as jnp
from llm_policy.environments.yokai.primitives import (
    YokaiObservation, YokaiFacts, YokaiMemory, YokaiComponents, YokaiEvents,
    card_adjacency, connected_components, is_connected, hypothetical_move,
    hint_card_links, observation_changes, init_belief_model_template,
    update_belief_model, observe_card, noop, sample_legal, make_trace,
)


@chex.dataclass
class YokaiBelief:
    """Uniform count-constrained prior given direct identities; no hint inference."""
    known_colors: jax.Array  # int32[16], -1 unknown
    valid: jax.Array


def color_components(positions, colors):
    adjacency = card_adjacency(positions)
    return jax.vmap(lambda color: connected_components(adjacency, colors == color))(jnp.arange(4))


def is_solved(positions, colors):
    colors = jnp.asarray(colors, jnp.int32)
    complete = jnp.all(jnp.sum(colors[None, :] == jnp.arange(4)[:, None], axis=1) == 4)
    return complete & jnp.all(color_components(positions, colors).count == 1)


def literal_card_belief(known_colors):
    """Exact uniform shuffled-deck prior conditioned on personally observed cards."""
    known = jnp.asarray(known_colors)
    if known.shape != (16,):
        raise ValueError("known_colors must have shape (16,)")
    admissible = jnp.all((known == -1) | jnp.any(known[:, None] == jnp.arange(4), axis=-1))
    counts = jnp.sum(known[:, None] == jnp.arange(4), axis=0)
    return YokaiBelief(known_colors=known.astype(jnp.int32), valid=admissible & jnp.all(counts <= 4))


def condition_on_colors(belief, observed_colors):
    """Merge direct observations; contradictory identities make the belief invalid."""
    observed = jnp.asarray(observed_colors)
    evidence = literal_card_belief(observed)
    conflict = jnp.any((belief.known_colors >= 0) & (observed >= 0) & (belief.known_colors != observed))
    merged = literal_card_belief(jnp.where(observed >= 0, observed, belief.known_colors))
    return merged.replace(valid=belief.valid & evidence.valid & merged.valid & ~conflict)


def card_marginals(belief):
    """Exact float32[16,4] marginals; invalid beliefs yield all zeros."""
    known = belief.known_colors
    one_hot = jax.nn.one_hot(known, 4, dtype=jnp.float32)
    remaining = 4 - one_hot.sum(axis=0)
    unknown = (known < 0).sum()
    marginals = jnp.where((known >= 0)[:, None], one_hot, remaining[None, :] / jnp.maximum(unknown, 1))
    return jnp.where(belief.valid, marginals, 0.)


def sample_assignment(belief, rng):
    """Exact conditional joint draw without enumerating 63,063,000 boards.

    Every valid draw has four cards per color and preserves known identities.
    Invalid beliefs return int32[16] filled with -1.
    """
    known = belief.known_colors
    remaining = 4 - jax.nn.one_hot(known, 4, dtype=jnp.int32).sum(axis=0)
    keys = jax.random.split(rng, 16)
    def draw(i, carry):
        counts, assignment = carry
        logits = jnp.where(counts > 0, jnp.log(jnp.maximum(counts, 1)), -jnp.inf)
        sampled = jax.random.categorical(keys[i], jnp.where(jnp.any(counts > 0), logits, jnp.zeros(4)))
        unknown = known[i] < 0
        color = jnp.where(unknown, sampled, known[i]).astype(jnp.int32)
        counts = counts - unknown.astype(jnp.int32) * jax.nn.one_hot(sampled, 4, dtype=jnp.int32)
        return counts, assignment.at[i].set(color)
    _, assignment = jax.lax.fori_loop(0, 16, draw, (remaining, known))
    return jnp.where(belief.valid, assignment, -1)


def sampled_solved_probability(positions, belief, rng, num_samples=128):
    """Monte Carlo estimate under the literal joint prior; NaN if invalid.

    num_samples must be a positive static integer. This is not an exact query.
    """
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
        raise ValueError("num_samples must be a positive static integer")
    samples = jax.vmap(lambda key: sample_assignment(belief, key))(jax.random.split(rng, num_samples))
    solved = jax.vmap(lambda colors: is_solved(positions, colors))(samples)
    return jnp.where(belief.valid, solved.astype(jnp.float32).mean(), jnp.nan)


def hint_correct_probability(belief, hint_colors, card):
    colors = jnp.asarray(hint_colors, jnp.bool_)
    valid = belief.valid & jnp.any(colors) & (card >= 0) & (card < 16)
    probability = jnp.sum(card_marginals(belief)[jnp.clip(card, 0, 15)] * colors)
    return jnp.where(valid, probability, jnp.nan)


def decode_observation(obs):
    nodes = obs.node_features
    colors = nodes[:16, :4]
    return YokaiFacts(
        visible_colors=jnp.where(colors.any(axis=1), jnp.argmax(colors, axis=1), -1).astype(jnp.int32),
        hint_colors=nodes[16:, 4:8].astype(jnp.bool_), positions=obs.positions,
        hint_positions=nodes[16:, 8:10].astype(jnp.int32), locked=nodes[:16, 10].astype(jnp.bool_),
        hints_placed=nodes[16:, 10].astype(jnp.bool_), seen_by_partner=nodes[:16, 11].astype(jnp.bool_),
        active=nodes[0, 12].astype(jnp.bool_), substep=nodes[0, 13].astype(jnp.int32),
        adjacency_matrix=obs.adjacency_matrix, legal_actions=obs.legal_actions)


def init_memory():
    return YokaiMemory(known_colors=jnp.full((16,), -1, jnp.int32))


def observe(obs, reset, memory):
    facts = decode_observation(obs)
    previous = jnp.where(reset, -1, memory.known_colors)
    known = jnp.where(facts.visible_colors >= 0, facts.visible_colors, previous)
    return facts, YokaiMemory(known_colors=known.astype(jnp.int32))


def init_observation_snapshot():
    return YokaiFacts(visible_colors=jnp.full((16,), -1, jnp.int32),
        hint_colors=jnp.zeros((7, 4), jnp.bool_), positions=jnp.full((16, 2), -1, jnp.int32),
        hint_positions=jnp.full((7, 2), -1, jnp.int32), locked=jnp.zeros((16,), jnp.bool_),
        hints_placed=jnp.zeros((7,), jnp.bool_), seen_by_partner=jnp.zeros((16,), jnp.bool_),
        active=jnp.asarray(False), substep=jnp.asarray(0, jnp.int32),
        adjacency_matrix=jnp.zeros((23, 23), jnp.uint8), legal_actions=jnp.zeros((2325,), jnp.bool_))


def legal_move_mask(facts):
    return facts.legal_actions[17:1617].reshape((16, 10, 10))


def legal_hint_placement_mask(facts):
    grid = facts.legal_actions[1624:2324].reshape((7, 10, 10))
    positions = facts.positions
    in_bounds = jnp.all((positions >= 0) & (positions < 10), axis=-1)
    return grid[:, jnp.clip(positions[:, 0], 0, 9), jnp.clip(positions[:, 1], 0, 9)] & in_bounds[None, :]


def move_card(card, x, y):
    return jnp.asarray(17 + card * 100 + x * 10 + y, jnp.int32)


def reveal_hint(hint):
    return jnp.asarray(1617 + hint, jnp.int32)


def place_hint(hint, x, y):
    return jnp.asarray(1624 + hint * 100 + x * 10 + y, jnp.int32)


def end_game():
    return jnp.asarray(2324, jnp.int32)


def is_legal(legal_actions, action):
    return (action >= 0) & (action < 2325) & legal_actions[jnp.clip(action, 0, 2324)]
