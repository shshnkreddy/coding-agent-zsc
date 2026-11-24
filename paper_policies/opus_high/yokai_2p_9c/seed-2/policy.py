"""Two-player 9-card Yokai: posterior-greedy clustering policy.

Design notes (no private convention is used anywhere):

* Colour connectivity of a colour only depends on the positions of that
  colour's three cards.  With exactly three cards per colour, a colour is
  connected iff at least two of its three internal pairs are orthogonally
  adjacent.  Every evaluation below is built from that fact.
* Colours the seat has observed itself are re-supplied in node_features on
  active turns (perfect memory), so the seat needs no colour memory.
* Unobserved cards are handled with the exact posterior: given the seat's
  own observations, every completion of the remaining colour multiset is
  equally likely.
* The terminal score is 5*unrevealed + 2*revealed + correct - incorrect, so
  revealing beats placing, placing correctly beats placing blindly, and
  ending early is worth roughly three points per turn saved.

All lookup tables are built at trace time inside ``_tables`` because module
scope may not contain calls.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'posterior_greedy_yokai', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

_NUM_CARDS = 9
_NUM_COLORS = 3
_GRID = 9
_NUM_HINTS = 4
_MOVE_BASE = 10
_NUM_MOVES = 729
_REVEAL_BASE = 739
_PLACE_BASE = 743
_NUM_PLACE = 324
_END_GAME = 1067
_NEG = -1.0e9

# Move-scoring weights.
_W_JOINT = 3.0
_W_MARGINAL = 1.0
_W_PAIRS = 1.0
_W_COMPACT = 0.01
_W_MOVE_UNKNOWN = 0.15
# End-game gamble: assumed own improvement per remaining own turn, and how
# much of the partner's measured effect on the posterior to project forward.
_OWN_GAIN_BASE = 0.25
_OWN_GAIN_KNOWN = 0.25
# Weight on the partner's measured effect on the colour deficit (one unit of
# deficit per partner move is treated as _PARTNER_TRUST of success probability).
_PARTNER_TRUST = 0.1
# Blanket pessimism about handing the board back to an unknown partner.
_RISK = 0.0
# Tie-break direction for which unknown card to inspect.  A positive value
# inspects high card ids first, which keeps this seat's colour knowledge
# disjoint from a partner that defaults to ascending order.
_OBS_ORDER = 0.01


def _tables():
    triples = []
    for a in range(_NUM_CARDS):
        for b in range(a + 1, _NUM_CARDS):
            for c in range(b + 1, _NUM_CARDS):
                triples.append((a, b, c))
    colorings = []
    for first in triples:
        rest = [i for i in range(_NUM_CARDS) if i not in first]
        for a in range(6):
            for b in range(a + 1, 6):
                for c in range(b + 1, 6):
                    row = [2] * _NUM_CARDS
                    for i in first:
                        row[i] = 0
                    for i in (rest[a], rest[b], rest[c]):
                        row[i] = 1
                    colorings.append(row)
    return {
        'tri_a': jnp.asarray([t[0] for t in triples], dtype=jnp.int32),
        'tri_b': jnp.asarray([t[1] for t in triples], dtype=jnp.int32),
        'tri_c': jnp.asarray([t[2] for t in triples], dtype=jnp.int32),
        'tri_member': jnp.asarray(
            [[1.0 if i in t else 0.0 for i in range(_NUM_CARDS)] for t in triples],
            dtype=jnp.float32),
        'colorings': jnp.asarray(colorings, dtype=jnp.int32),
        'coloring_onehot': jnp.asarray(
            [[[1.0 if row[i] == c else 0.0 for c in range(_NUM_COLORS)]
              for i in range(_NUM_CARDS)] for row in colorings], dtype=jnp.float32),
        'cards': jnp.arange(_NUM_CARDS, dtype=jnp.int32),
        'move_card': jnp.arange(_NUM_MOVES, dtype=jnp.int32) // 81,
        'move_x': (jnp.arange(_NUM_MOVES, dtype=jnp.int32) % 81) // _GRID,
        'move_y': jnp.arange(_NUM_MOVES, dtype=jnp.int32) % _GRID,
        'place_hint': jnp.arange(_NUM_PLACE, dtype=jnp.int32) // 81,
        'place_x': (jnp.arange(_NUM_PLACE, dtype=jnp.int32) % 81) // _GRID,
        'place_y': jnp.arange(_NUM_PLACE, dtype=jnp.int32) % _GRID,
        'hints': jnp.arange(_NUM_HINTS, dtype=jnp.float32),
    }


def init_agent():
    # [snapshot of P(success) when this seat handed over, sum of partner
    #  effects on P(success), number of partner effects seen, snapshot valid]
    return jnp.zeros((4,), dtype=jnp.float32)


def _adjacency(pos):
    """Orthogonal adjacency of card positions, shape (..., 9, 9) float32."""
    delta = jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]).sum(-1)
    return (delta == 1).astype(jnp.float32)


def _triple_pairs(adj, tab):
    """Number of adjacent pairs inside each three-card set, shape (..., 84)."""
    ab = adj[..., tab['tri_a'], tab['tri_b']]
    ac = adj[..., tab['tri_a'], tab['tri_c']]
    bc = adj[..., tab['tri_b'], tab['tri_c']]
    return ab + ac + bc


def _triple_validity(known_color, tab):
    """(84, 3) mask: triple t can be exactly the card set of colour c."""
    is_color = jnp.stack(
        [(known_color == c).astype(jnp.float32) for c in range(_NUM_COLORS)], axis=1)
    is_other = jnp.stack(
        [((known_color >= 0) & (known_color != c)).astype(jnp.float32)
         for c in range(_NUM_COLORS)], axis=1)
    contains_other = tab['tri_member'] @ is_other
    contains_color = tab['tri_member'] @ is_color
    needed = is_color.sum(0)
    return (contains_other == 0.0) & (contains_color == needed[None, :])


def _joint_success(pos, known_color, tab):
    """Exact posterior probability that every colour is currently connected."""
    adj = _adjacency(pos)
    onehot = tab['coloring_onehot']
    pair = jnp.einsum('mic,ij,mjc->mc', onehot, adj, onehot) * 0.5
    success = (pair >= 2.0).all(axis=1)
    consistent = ((known_color[None, :] < 0)
                  | (tab['colorings'] == known_color[None, :])).all(axis=1)
    total = consistent.sum()
    good = (consistent & success).sum()
    return good.astype(jnp.float32) / jnp.maximum(total, 1).astype(jnp.float32)


def _decode(obs):
    features = obs.node_features.astype(jnp.int32)
    cards = features[:_NUM_CARDS]
    hints = features[_NUM_CARDS:]
    color_onehot = cards[:, 0:3]
    known = color_onehot.sum(1) > 0
    known_color = jnp.where(known, jnp.argmax(color_onehot, axis=1), -1)
    partner_seen = cards[:, 9] > 0
    active = cards[0, 10] > 0
    substep = cards[0, 11]
    hint_sets = (hints[:, 3:6] > 0).astype(jnp.float32)
    hint_revealed = hint_sets.sum(1) > 0
    hint_placed = hints[:, 8] > 0
    hint_pos = hints[:, 6:8]
    pos = obs.positions.astype(jnp.int32)
    return (pos, known, known_color, partner_seen, active, substep,
            hint_sets, hint_revealed, hint_placed, hint_pos)


def _color_posterior(known, known_color):
    """(9, 3) posterior probability of each card's colour."""
    counts = jnp.stack(
        [(known_color == c).sum() for c in range(_NUM_COLORS)]).astype(jnp.float32)
    remaining = 3.0 - counts
    unknown = jnp.maximum((~known).sum().astype(jnp.float32), 1.0)
    prior = remaining / unknown
    onehot = jnp.stack(
        [(known_color == c).astype(jnp.float32) for c in range(_NUM_COLORS)], axis=1)
    return jnp.where(known[:, None], onehot, prior[None, :])


def _observe_action(obs, known, partner_seen, tab):
    legal = obs.legal_actions[1:1 + _NUM_CARDS]
    # Inspect what this seat does not know, preferring cards the partner has
    # not inspected either; _OBS_ORDER only breaks the remaining ties.
    score = (2.0 * (~known).astype(jnp.float32)
             + 0.5 * (~partner_seen).astype(jnp.float32)
             + _OBS_ORDER * tab['cards'].astype(jnp.float32))
    score = jnp.where(legal, score, _NEG)
    return (1 + jnp.argmax(score)).astype(jnp.int32)


def _move_action(obs, pos, known_color, valid, valid_count, tab):
    legal = obs.legal_actions[_MOVE_BASE:_MOVE_BASE + _NUM_MOVES]
    target = jnp.stack([tab['move_x'], tab['move_y']], axis=-1)
    replace = (tab['cards'][None, :] == tab['move_card'][:, None])
    cand = jnp.where(replace[:, :, None], target[:, None, :], pos[None, :, :])
    adj = _adjacency(cand)
    pairs = _triple_pairs(adj, tab)
    weight = valid.astype(jnp.float32) / jnp.maximum(valid_count, 1.0)
    # Posterior probability that each colour is already connected.
    q = ((pairs >= 2.0).astype(jnp.float32)) @ weight
    # Partial credit: a colour needs two internal adjacencies, so reward the
    # first one too.  Without this the search sees no progress on a colour
    # that still needs two moves.
    credit = jnp.minimum(pairs, 2.0) @ weight
    score = (_W_JOINT * q.prod(axis=1)
             + _W_MARGINAL * q.sum(axis=1)
             + _W_PAIRS * credit.sum(axis=1)
             + _W_COMPACT * adj.sum(axis=(1, 2))
             - _W_MOVE_UNKNOWN * (known_color[tab['move_card']] < 0).astype(jnp.float32))
    score = jnp.where(legal, score, _NEG)
    return (_MOVE_BASE + jnp.argmax(score)).astype(jnp.int32)


def _hint_action(obs, pos, known, posterior, hint_sets, q_now, tab):
    reveal_legal = obs.legal_actions[_REVEAL_BASE:_REVEAL_BASE + _NUM_HINTS]
    reveal_score = jnp.where(reveal_legal, -tab['hints'], _NEG)
    reveal_action = (_REVEAL_BASE + jnp.argmax(reveal_score)).astype(jnp.int32)

    place_legal = obs.legal_actions[_PLACE_BASE:_PLACE_BASE + _NUM_PLACE]
    on_cell = ((pos[None, :, 0] == tab['place_x'][:, None])
               & (pos[None, :, 1] == tab['place_y'][:, None])).astype(jnp.float32)
    match = on_cell @ (posterior @ hint_sets.T)
    correct = jnp.take_along_axis(match, tab['place_hint'][:, None], axis=1)[:, 0]
    card_known = on_cell @ known.astype(jnp.float32)
    settled = on_cell @ (posterior @ q_now)
    place_score = 10.0 * correct + 1.0 * card_known + 1.0 * settled
    place_score = jnp.where(place_legal, place_score, _NEG)
    place_action = (_PLACE_BASE + jnp.argmax(place_score)).astype(jnp.int32)
    return jnp.where(reveal_legal.any(), reveal_action, place_action)


def _end_value(posterior, hint_sets, hint_revealed, hint_placed, hint_pos, pos):
    """Terminal score obtainable by ending the game right now, and turns used."""
    unrevealed = (~hint_revealed).astype(jnp.float32).sum()
    waiting = (hint_revealed & (~hint_placed)).astype(jnp.float32).sum()
    placed = hint_placed.astype(jnp.float32)
    on_card = ((pos[None, :, 0] == hint_pos[:, 0:1])
               & (pos[None, :, 1] == hint_pos[:, 1:2])).astype(jnp.float32)
    correct = ((on_card @ posterior) * hint_sets).sum(1)
    value = 5.0 * unrevealed + 2.0 * waiting + (placed * (2.0 * correct - 1.0)).sum()
    turns = (4.0 - unrevealed) + placed.sum()
    return value, turns


def _substep_zero(obs, pos, known, known_color, posterior, hint_sets, hint_revealed,
                  hint_placed, hint_pos, partner_seen, success, deficit,
                  partner_effect, tab):
    value, turns = _end_value(posterior, hint_sets, hint_revealed, hint_placed,
                              hint_pos, pos)
    # Ending later costs hints; the schedule below is the value of stopping at
    # the seat's next turn assuming reveals are used before placements.
    future_turns = turns + 2.0
    future_value = jnp.where(future_turns <= 4.0, 20.0 - 3.0 * future_turns,
                             jnp.maximum(12.0 - future_turns, 4.0))
    # Expected improvement before this seat's next turn: this seat's own
    # informed move plus whatever the partner has been contributing.
    own_gain = (_OWN_GAIN_BASE
                + _OWN_GAIN_KNOWN * known.astype(jnp.float32).sum() / 9.0)
    quality = jnp.clip(partner_effect, -2.0, 2.0)
    future_success = jnp.clip(success + own_gain * (1.0 - success)
                              + _PARTNER_TRUST * quality - _RISK, 0.0, 1.0)
    stop = (success >= 0.9995) | ((success * value) >= (future_value * future_success))
    observe = _observe_action(obs, known, partner_seen, tab)
    end_legal = obs.legal_actions[_END_GAME]
    return jnp.where(stop & end_legal, jnp.asarray(_END_GAME, dtype=jnp.int32), observe)


def agent_act(obs, reset, memory, rng):
    tab = _tables()
    (pos, known, known_color, partner_seen, active, substep,
     hint_sets, hint_revealed, hint_placed, hint_pos) = _decode(obs)
    posterior = _color_posterior(known, known_color)
    valid = _triple_validity(known_color, tab)
    valid_count = valid.astype(jnp.float32).sum(0)
    weight_now = valid.astype(jnp.float32) / jnp.maximum(valid_count, 1.0)
    pairs_now = _triple_pairs(_adjacency(pos), tab)
    q_now = ((pairs_now >= 2.0).astype(jnp.float32)) @ weight_now
    credit_now = (jnp.minimum(pairs_now, 2.0) @ weight_now).sum()
    # Expected number of same-colour adjacencies still missing; roughly the
    # number of further moves the board needs.
    deficit = 6.0 - credit_now

    state = jnp.where(reset, jnp.zeros((4,), dtype=jnp.float32), memory)
    success = _joint_success(pos, known_color, tab)
    # The board only changes between this seat's own turns through the
    # partner's single move, so comparing the deficit now against the snapshot
    # taken when this seat handed over measures how useful the partner is.
    delta = jnp.where(state[3] > 0.0, state[0] - deficit, 0.0)
    fresh = active & (substep == 0) & (state[3] > 0.0)
    state = state.at[1].add(jnp.where(fresh, delta, 0.0))
    state = state.at[2].add(jnp.where(fresh, 1.0, 0.0))
    partner_effect = state[1] / jnp.maximum(state[2], 1.0)
    snapshot = active & (substep == 3)
    state = state.at[0].set(jnp.where(snapshot, deficit, state[0]))
    state = state.at[3].set(jnp.where(snapshot, 1.0, state[3]))

    def branch_zero(_):
        return _substep_zero(obs, pos, known, known_color, posterior, hint_sets,
                             hint_revealed, hint_placed, hint_pos, partner_seen,
                             success, deficit, partner_effect, tab)

    def branch_one(_):
        return _observe_action(obs, known, partner_seen, tab)

    def branch_two(_):
        return _move_action(obs, pos, known_color, valid, valid_count, tab)

    def branch_three(_):
        return _hint_action(obs, pos, known, posterior, hint_sets, q_now, tab)

    def act(_):
        return jax.lax.switch(jnp.clip(substep, 0, 3),
                              [branch_zero, branch_one, branch_two, branch_three], None)

    action = jax.lax.cond(active, act, lambda _: jnp.asarray(0, dtype=jnp.int32), None)
    action = jnp.where(obs.legal_actions[action], action,
                       jnp.argmax(obs.legal_actions)).astype(jnp.int32)

    trace = {
        'num_known': known.astype(jnp.int32).sum(),
        'q_sum': q_now.sum(),
        'deficit': deficit,
        'p_success': success,
        'partner_effect': partner_effect,
        'substep': substep.astype(jnp.int32),
        'active': active.astype(jnp.int32),
    }
    return action, state, trace
