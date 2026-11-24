"""Cooperative 16-card Yokai policy.

Terminal score is ``5*U + 2*R + C - I`` and is zero unless every colour forms one
orthogonally connected group, so the whole game is about clustering the colours
and then ending the episode as early as possible: ending after T completed full
turns can score at most ``35 - 3r - p`` with ``r + p = T`` and ``p <= r``, i.e.
about ``35 - 2T``.

The reward depends on exactly one board statistic: the number of monochrome
connected components (the reward is positive only when that number is four).
Every decision here is a greedy improvement of that statistic on the *current*
public board, evaluated under this seat's own colour evidence by sampling the
colours of the cards it has not observed:

* substep 0/1 - observe cards whose colour this seat does not know, preferring
  ones the partner has not observed either (public flag, no convention);
* substep 2   - relocate the single card that most reduces the expected number
  of monochrome components;
* substep 3   - reveal a large hint set, or place a hint when that is cheaper
  than revealing (placing costs 3 - 2*P(correct) instead of 3, but locks the
  card for the rest of the episode);
* substep 0   - end the game once the sampled expected terminal score beats an
  estimate of what continuing is worth.

Colour samples are reweighted by two pieces of public evidence that assume only
that the partner is *trying to score*, never that it shares any convention,
role split or signalling code with this policy:

* a card the partner relocated tends to have been put beside its own colour;
* a hint the partner placed tends to sit on a card whose colour is in that
  hint's (public) colour set - the only evidence available about locked cards,
  which can never be observed again by anybody.

If the partner is incompetent these likelihoods are wrong, but in that case the
colours will not get clustered and the episode scores about zero either way, so
the downside is bounded while the upside (knowing the colours of cards this seat
cannot observe) is what makes an early, correct end action possible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'belief_cluster_ev_stop',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 4,
    'deck_size': 16,
    'grid_size': 10,
    'num_hints': 7,
    'num_nodes': 23,
    'node_features': 15,
    'num_actions': 2325,
    'policy_structure': 'shared_local',
}

N_CARDS = 16
N_COLORS = 4
N_HINTS = 7
GRID = 10
N_CELLS = GRID * GRID
PER_COLOR = 4

MOVE_BASE = 17
REVEAL_BASE = 1617
PLACE_BASE = 1624
END_ACTION = 2324

N_PROPOSALS = 128     # uniform count-consistent colour assignments
N_SAMPLES = 32        # reweighted samples kept
N_MOVE_SAMPLES = 32   # samples used per candidate move (common random numbers)
N_CANDIDATES = 16     # moves short-listed for exact component evaluation
NEG = -1.0e9

# Weak global compactness pressure, so that moves which are neutral for the
# colour objective still keep the board tight instead of trailing cards away.
EPS_GLOBAL = 0.06
QUAD = 0.20           # superlinear distance: far-flung same-colour cards first

# Log-likelihood bonuses for the partner-competence evidence.
LAM_MOVED_SEEN = 1.1   # partner relocated this card and has observed it
LAM_MOVED_BLIND = 0.2  # partner relocated it without ever observing it
LAM_HINT = 1.5         # a hint the partner placed matches the card's colour
EVIDENCE_CAP = 2.0

# Continuation value per number of completed full turns (a public quantity:
# reveals + placements).  The game is ended as soon as the sampled expected
# terminal score exceeds this.  The schedule decays because every further turn
# burns hint value and the remaining chance to improve the board shrinks.
# Index 12 and 13 are the two seats' final turns (player 0 acts on odd turns),
# where there is no continuation value left for the acting seat, so both sit at
# the floor.
STOP_THRESHOLD = (9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.0, 4.0, 3.0, 2.2, 1.4,
                  0.05, 0.05, 0.05, 0.05)


def init_agent():
    return {'known': -jnp.ones((N_CARDS,), dtype=jnp.int32),
            'prev_pos': jnp.zeros((N_CARDS, 2), dtype=jnp.int32),
            'partner_last': jnp.zeros((N_CARDS,), dtype=jnp.float32)}


def _eye():
    return jnp.eye(N_CARDS, dtype=bool)


def _cell_coords():
    cells = jnp.arange(N_CELLS, dtype=jnp.int32)
    return cells // GRID, cells % GRID


def _pair_manhattan(pos):
    dx = jnp.abs(pos[:, None, 0] - pos[None, :, 0])
    dy = jnp.abs(pos[:, None, 1] - pos[None, :, 1])
    return (dx + dy).astype(jnp.float32)


def _cost(dist):
    return dist + QUAD * dist * dist


def _components(pos_cand, colors):
    """Monochrome component count of each (board, colour sample) pair.

    ``pos_cand`` is [K, 16, 2] candidate boards and ``colors`` is [n, 16] full
    colour assignments.  The result is [n, K] and equals four exactly when
    every colour forms a single orthogonally connected group.
    """
    dx = jnp.abs(pos_cand[:, :, None, 0] - pos_cand[:, None, :, 0])
    dy = jnp.abs(pos_cand[:, :, None, 1] - pos_cand[:, None, :, 1])
    adjacent = (dx + dy) == 1
    same = colors[:, None, :, None] == colors[:, None, None, :]
    mask = jnp.logical_or(jnp.logical_and(adjacent[None, :, :, :], same),
                          _eye()[None, None, :, :])
    # Four cards per colour, so paths never exceed three hops; two squarings of
    # the self-looped mask already give the transitive closure.  bfloat16 holds
    # these small integers exactly and keeps the candidate tensor small.
    reach = mask.astype(jnp.bfloat16)
    reach = jnp.minimum(reach @ reach, jnp.bfloat16(1.0))
    reach = jnp.minimum(reach @ reach, jnp.bfloat16(1.0))
    size = jnp.sum(reach.astype(jnp.float32), axis=-1)
    return jnp.sum(1.0 / size, axis=-1)


def _propose_colors(known, remaining, rng):
    """Uniform colour assignments for unobserved cards, consistent with counts."""
    unknown = known < 0
    slots = jnp.arange(N_CARDS, dtype=jnp.int32)
    cumulative = jnp.cumsum(remaining)
    labels = jnp.clip(
        (slots[:, None] >= cumulative[None, :]).sum(-1), 0, N_COLORS - 1
    ).astype(jnp.int32)

    noise = jax.random.uniform(rng, (N_PROPOSALS, N_CARDS))
    keyed = jnp.where(unknown[None, :], noise, 2.0 + noise)
    order = jnp.argsort(keyed, axis=-1)
    rank = jnp.argsort(order, axis=-1)
    return jnp.where(unknown[None, :], labels[rank],
                     jnp.maximum(known, 0)[None, :])


def agent_act(obs, reset, memory, rng):
    features = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    cards = features[:N_CARDS]
    hints = features[N_CARDS:]
    key_propose, key_resample, key_tie = jax.random.split(rng, 3)

    visible = cards[:, 0:N_COLORS]
    seen_now = jnp.sum(visible, axis=-1) > 0
    color_now = jnp.argmax(visible, axis=-1).astype(jnp.int32)
    partner_seen = cards[:, 11] > 0
    active = features[0, 12] > 0
    substep = features[0, 13]
    pos = obs.positions.astype(jnp.int32)
    cell_x, cell_y = _cell_coords()

    hint_sets = hints[:, N_COLORS:2 * N_COLORS].astype(jnp.float32)
    hint_size = jnp.sum(hint_sets, axis=-1)
    hint_revealed = hint_size > 0
    hint_placed = hints[:, 10] > 0
    hint_card = (obs.adjacency_matrix[N_CARDS:, 0:N_CARDS].astype(jnp.float32)
                 * hint_placed.astype(jnp.float32)[:, None])
    turns_done = jnp.sum(hint_revealed.astype(jnp.int32)) + jnp.sum(
        hint_placed.astype(jnp.int32))

    # ---- private colour knowledge ------------------------------------------
    prior = jnp.where(reset, -jnp.ones((N_CARDS,), jnp.int32), memory['known'])
    known = jnp.where(seen_now, color_now, prior)
    is_known = known >= 0
    known_onehot = jax.nn.one_hot(jnp.maximum(known, 0), N_COLORS,
                                  dtype=jnp.float32) * is_known[:, None]
    counts = jnp.sum(known_onehot, axis=0)
    remaining = jnp.maximum(PER_COLOR - counts, 0.0)

    # Attribute each relocation to whichever seat was active when the position
    # changed; a card last relocated by the partner is evidence about its
    # colour, because the partner was presumably joining it to its own group.
    prev_pos = jnp.where(reset, pos, memory['prev_pos'])
    just_moved = jnp.any(pos != prev_pos, axis=-1)
    partner_last = jnp.where(
        just_moved, jnp.where(active, 0.0, 1.0),
        jnp.where(reset, jnp.zeros((N_CARDS,), jnp.float32),
                  memory['partner_last']))

    dist = _pair_manhattan(pos)
    adjacent = dist == 1.0
    degree = jnp.sum(adjacent.astype(jnp.float32), axis=-1)

    # ---- colour samples reweighted by partner-competence evidence ----------
    proposals = _propose_colors(known, remaining, key_propose)
    onehot_prop = jax.nn.one_hot(proposals, N_COLORS, dtype=jnp.float32)

    neighbour_colors = jnp.minimum(
        adjacent.astype(jnp.float32) @ known_onehot, EVIDENCE_CAP)
    move_weight = partner_last * jnp.where(partner_seen, LAM_MOVED_SEEN,
                                           LAM_MOVED_BLIND)
    adjacency_evidence = move_weight[None, :] * jnp.sum(
        neighbour_colors[None, :, :] * onehot_prop, axis=-1)

    # A hint the partner placed is public: its colour set is very likely to
    # contain the hidden colour of the card it sits on.  For a locked card this
    # is the only information anybody will ever get.
    # (Hints this seat placed itself sit on cards whose colour it knows, so the
    # unknown-card mask below already restricts this to the partner's work.)
    card_hint_set = jnp.minimum(jnp.einsum('hi,hc->ic', hint_card, hint_sets),
                                1.0)
    # A placement only carries information if the placer knew what it was
    # covering, so scale it by the public flag that the partner observed the
    # card.  This keeps the inference honest against a partner that places
    # hints blindly.
    hint_trust = LAM_HINT * (0.25 + 0.75 * partner_seen.astype(jnp.float32))
    hint_evidence = hint_trust[None, :] * jnp.sum(
        card_hint_set[None, :, :] * onehot_prop, axis=-1)

    unknown_mask = (1.0 - is_known.astype(jnp.float32))[None, :]
    log_weight = jnp.sum(unknown_mask * (adjacency_evidence + hint_evidence),
                         axis=-1)
    picked = jax.random.categorical(key_resample, log_weight, shape=(N_SAMPLES,))
    samples = proposals[picked]

    # Everything downstream is an expectation over these samples.
    onehot = jax.nn.one_hot(samples, N_COLORS, dtype=jnp.float32)
    belief = jnp.mean(onehot, axis=0)
    offdiag = 1.0 - _eye().astype(jnp.float32)
    pair_w = jnp.mean((samples[:, :, None] == samples[:, None, :]
                       ).astype(jnp.float32), axis=0) * offdiag
    weight = pair_w + EPS_GLOBAL * offdiag

    # ---- move: short-list cheaply, then score components exactly -----------
    current = jnp.sum(weight * _cost(dist), axis=-1)
    cell_dist = (jnp.abs(cell_x[:, None] - pos[None, :, 0])
                 + jnp.abs(cell_y[:, None] - pos[None, :, 1])).astype(jnp.float32)
    moved_cost = (_cost(cell_dist) @ weight.T).T
    delta_dist = moved_cost - current[:, None]
    neighbours_now = jnp.sum(pair_w * adjacent, axis=-1)
    neighbours_new = (jnp.asarray(cell_dist == 1.0, jnp.float32) @ pair_w.T).T
    cheap = (neighbours_new - neighbours_now[:, None]) - 0.04 * delta_dist

    move_legal = legal[MOVE_BASE:REVEAL_BASE].reshape(N_CARDS, N_CELLS)
    cheap_flat = jnp.where(move_legal, cheap, NEG).reshape(-1)
    short_value, short_idx = jax.lax.top_k(cheap_flat, N_CANDIDATES)
    short_ok = short_value > (0.5 * NEG)
    short_card = (short_idx // N_CELLS).astype(jnp.int32)
    short_cell = (short_idx % N_CELLS).astype(jnp.int32)
    boards = jnp.tile(pos[None, :, :], (N_CANDIDATES, 1, 1))
    boards = boards.at[jnp.arange(N_CANDIDATES), short_card].set(
        jnp.stack([cell_x[short_cell], cell_y[short_cell]], axis=-1))
    comps = _components(boards, samples[:N_MOVE_SAMPLES])
    # Candidates very often tie on expected components; the distance potential
    # is what then decides, by pulling same-colour cards together so that a
    # later merge becomes available to whichever seat moves next.
    short_dist = jnp.clip(delta_dist.reshape(-1)[short_idx], -40.0, 40.0)
    move_value = (3.0 * jnp.mean(comps < 4.5, axis=0)
                  - jnp.mean(comps, axis=0)
                  - 0.06 * short_dist
                  + 0.02 * jnp.clip(short_value, -40.0, 40.0))
    move_pick = jnp.argmax(jnp.where(short_ok, move_value, NEG))
    move_action = MOVE_BASE + short_idx[move_pick].astype(jnp.int32)

    # ---- observe: unknown to me first, then unseen by the partner -----------
    observe_score = (8.0 * (1.0 - is_known.astype(jnp.float32))
                     + 2.0 * (1.0 - partner_seen.astype(jnp.float32))
                     + 0.3 * degree
                     + 0.05 * jax.random.uniform(key_tie, (N_CARDS,)))
    observe_legal = legal[1:1 + N_CARDS]
    observe_action = 1 + jnp.argmax(
        jnp.where(observe_legal, observe_score, NEG)).astype(jnp.int32)

    # ---- hint substep: a compulsory action, so a pure cost comparison ------
    # Revealing turns a 5 into a 2.  Placing turns a 2 into +1 or -1, i.e. it
    # costs 3 - 2*P(correct), but it also locks the card: it can never move or
    # be observed again, which is cheap for a card already sitting with its own
    # colour and expensive otherwise.
    same_known = jnp.logical_and(is_known[:, None] & is_known[None, :],
                                 known[:, None] == known[None, :])
    group_mask = jnp.logical_and(same_known, adjacent).astype(jnp.float32)
    group_mask = group_mask + _eye().astype(jnp.float32)
    reach = jnp.minimum(group_mask @ group_mask, 1.0)
    reach = jnp.minimum(reach @ reach, 1.0) > 0.0
    color_ok = jnp.all(jnp.logical_or(reach, ~same_known), axis=-1)
    has_same_neighbour = jnp.any(jnp.logical_and(same_known, adjacent), axis=-1)

    flexibility = (jnp.where(has_same_neighbour, 0.0, 3.5)
                   + jnp.where(color_ok, 0.0, 0.7)
                   + jnp.where(is_known, -0.3, 3.0))
    correct_prob = jnp.einsum('hc,ic->hi', hint_sets, belief)
    place_cost = (3.0 - 2.0 * correct_prob) + flexibility[None, :]
    card_cell = pos[:, 0] * GRID + pos[:, 1]
    place_legal = legal[PLACE_BASE:END_ACTION].reshape(N_HINTS, N_CELLS)[:, card_cell]
    place_score = jnp.where(place_legal, -place_cost, NEG)
    place_flat = jnp.argmax(place_score.reshape(-1))
    place_hint = (place_flat // N_CARDS).astype(jnp.int32)
    place_card = (place_flat % N_CARDS).astype(jnp.int32)
    place_action = PLACE_BASE + place_hint * N_CELLS + card_cell[place_card]
    place_best = place_score.reshape(-1)[place_flat]

    reveal_legal = legal[REVEAL_BASE:PLACE_BASE]
    reveal_score = jnp.where(reveal_legal, -3.0 + 0.1 * hint_size, NEG)
    reveal_idx = jnp.argmax(reveal_score)
    reveal_action = REVEAL_BASE + reveal_idx.astype(jnp.int32)
    hint_action = jnp.where(place_best >= reveal_score[reveal_idx], place_action,
                            reveal_action).astype(jnp.int32)

    # ---- stopping rule ------------------------------------------------------
    end_comps = _components(pos[None, :, :], samples)[:, 0]
    connected = end_comps < 4.5
    target_color = jnp.einsum('hi,nic->nhc', hint_card, onehot)
    in_set = jnp.sum(target_color * hint_sets[None, :, :], axis=-1)
    placed = hint_placed.astype(jnp.float32)[None, :]
    correct = jnp.sum(placed * in_set, axis=-1)
    wrong = jnp.sum(placed * (1.0 - in_set), axis=-1)
    unrevealed = jnp.sum(1.0 - hint_revealed.astype(jnp.float32))
    waiting = jnp.sum(hint_revealed.astype(jnp.float32)
                      * (1.0 - hint_placed.astype(jnp.float32)))
    end_score = 5.0 * unrevealed + 2.0 * waiting + correct - wrong
    end_value = jnp.mean(jnp.where(connected, end_score, 0.0))

    stop_threshold = jnp.asarray(STOP_THRESHOLD, dtype=jnp.float32)
    threshold = stop_threshold[jnp.clip(turns_done, 0, len(STOP_THRESHOLD) - 1)]
    want_end = jnp.logical_and(end_value >= threshold, legal[END_ACTION])
    first_action = jnp.where(want_end, END_ACTION, observe_action)

    action = jnp.where(substep == 0, first_action,
                       jnp.where(substep == 1, observe_action,
                                 jnp.where(substep == 2, move_action,
                                           hint_action)))
    action = jnp.where(active, action, 0).astype(jnp.int32)
    action = jnp.where(legal[action], action,
                       jnp.argmax(legal).astype(jnp.int32)).astype(jnp.int32)

    trace = {
        'end_value': end_value.astype(jnp.float32),
        'p_connected': jnp.mean(connected).astype(jnp.float32),
        'n_known': jnp.sum(is_known.astype(jnp.int32)),
        'turns_done': turns_done.astype(jnp.int32),
        'components': jnp.mean(end_comps).astype(jnp.float32),
        'action': action,
    }
    next_memory = {'known': known, 'prev_pos': pos,
                   'partner_last': partner_last}
    return action, next_memory, trace
