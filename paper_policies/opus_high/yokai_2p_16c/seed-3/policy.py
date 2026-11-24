"""Two-player Yokai policy: locally-grounded colour-cluster merging.

Design notes (no shared convention is assumed with the partner):
  * All decisions are functions of the *public* board plus this seat's own
    privately observed colours.  Nothing is encoded into action choices for the
    partner to decode, and nothing is decoded from the partner's choices beyond
    the publicly documented meaning of the observation fields.
  * The move rule greedily minimises the number of same-colour connected
    components over the cards this seat has seen.  That potential is the actual
    task objective restricted to local knowledge, so an independently generated
    partner that also tries to cluster colours pushes the same quantity down
    instead of fighting a private target layout.
  * Substep 3 reveals the widest hint available until some card's colour is both
    seen and settled in a cluster, then places hints only where this seat's own
    observation proves the match.  The terminal score 5*U + 2*R + C - I punishes
    every reveal, but locking a card into a group that still has to move is
    worse, and that trade was measured rather than assumed.
  * The game is ended early only when the arrangement is provably solved under
    every colouring still consistent with this seat's information, or when a
    looser reading of the unseen cards is globally consistent and worth more
    than the 7 points a solved board scores at the 14-turn cap.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'yokai_cluster_greedy',
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

NCARD = 16
NCOL = 4
NHINT = 7
GRID = 10
NCELL = GRID * GRID
A_OBS = 1
A_MOVE = 17
A_REVEAL = 1617
A_PLACE = 1624
A_END = 2324

# Hint h's colour-set size is fixed by the reset rule (0,1 -> 1; 2,3,4 -> 2; 5,6 -> 3).
HINT_SIZE_TUPLE = (1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0)
NEG = -1.0e9


def _hint_size():
    return jnp.asarray(HINT_SIZE_TUPLE, dtype=jnp.float32)


def _cell_xy():
    index = jnp.arange(NCELL)
    return jnp.stack([index // GRID, index % GRID], axis=1)


def _eye():
    return jnp.eye(NCARD, dtype=bool)


def _lower():
    return jnp.tril(jnp.ones((NCARD, NCARD), dtype=bool), -1)


# Move-score weights: component count dominates, then colour bonds, then a mild
# preference for local, compact rearrangements.
W_PHI = 100.0
W_SIZE = 15.0
W_BOND = 8.0
W_SPREAD = 0.05
W_UNKNOWN = 8.0
W_SOFT = 20.0
OBS_UNKNOWN = 4.0
OBS_UNSEEN = 2.0

# Placement safety gate and the score floor below which a heuristic early end is
# not worth its risk (a solved board that runs to the 14-turn cap scores 7).
PLACE_MIN_CLUSTER = 3.0
MIN_END_POTENTIAL = 8.0


def _closure(edges):
    """Transitive closure of a (batched) 16x16 boolean adjacency with self loops."""
    r = edges.astype(jnp.float32)
    for _ in range(4):
        r = jnp.minimum(jnp.matmul(r, r), 1.0)
    return r > 0.5


def _all_colours_connected(adj, full_col):
    """True when every card reaches exactly the four cards of its own colour."""
    eye = _eye()
    same = full_col[:, None] == full_col[None, :]
    reach = _closure((adj & same) | eye)
    return jnp.all(jnp.sum(reach, axis=1) == NCARD // NCOL)


def _decode(obs):
    not_eye = ~_eye()
    nf = obs.node_features.astype(jnp.int32)
    onehot = nf[:NCARD, 0:NCOL]
    known = jnp.sum(onehot, axis=1) > 0
    col = jnp.where(known, jnp.argmax(onehot, axis=1), -1)
    pos = obs.positions.astype(jnp.int32)
    state = {
        'known': known,
        'col': col,
        'pos': pos,
        'pseen': nf[:NCARD, 11] > 0,
        'active': nf[0, 12] > 0,
        'substep': nf[0, 13],
        'hset': nf[NCARD:NCARD + NHINT, 4:8] > 0,
        'hplaced': nf[NCARD:NCARD + NHINT, 10] > 0,
        'legal': obs.legal_actions,
    }
    state['hrev'] = jnp.any(state['hset'], axis=1)
    delta = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
    state['adj'] = (delta == 1) & not_eye

    seen_onehot = (jnp.arange(NCOL)[None, :] == col[:, None]) & known[:, None]
    remaining = 4.0 - jnp.sum(seen_onehot.astype(jnp.float32), axis=0)

    # A placed hint publicly restricts its card's colour to the hint's revealed
    # set, provided the placement was correct.  Placing correctly is worth two
    # points per hint, so any partner playing for score satisfies that; this is
    # a competence assumption, not an agreed code.
    hint_pos = nf[NCARD:NCARD + NHINT, 8:10]
    on_card = (state['hplaced'][None, :]
               & jnp.all(hint_pos[None, :, :] == pos[:, None, :], axis=2))
    allowed = ~jnp.any(on_card[:, :, None] & ~state['hset'][None, :, :], axis=1)

    # Colours an unseen card could still have, given the 4-per-colour deck and
    # any hint sitting on it.  A single survivor is a deduction, not a guess.
    by_count = remaining[None, :] > 0.5
    constrained = by_count & allowed
    # A hint whose set excludes every colour still available was demonstrably
    # placed wrong; drop that card's constraint instead of believing it.
    constrained = jnp.where(jnp.any(constrained, axis=1, keepdims=True), constrained, by_count)
    possible = jnp.where(known[:, None], seen_onehot, constrained)
    deduced = (~known) & (jnp.sum(possible.astype(jnp.int32), axis=1) == 1)
    known_eff = known | deduced
    col_eff = jnp.where(known_eff, jnp.where(known, col, jnp.argmax(possible, axis=1)), -1)

    weight = possible.astype(jnp.float32) * remaining[None, :]
    weight = jnp.where(known_eff[:, None],
                       (jnp.arange(NCOL)[None, :] == col_eff[:, None]).astype(jnp.float32),
                       weight / jnp.maximum(jnp.sum(weight, axis=1, keepdims=True), 1e-6))

    state['known'] = known_eff
    state['col'] = col_eff
    state['possible'] = possible
    state['colour_posterior'] = weight
    state['same'] = (col_eff[:, None] == col_eff[None, :]) & known_eff[:, None] & known_eff[None, :]
    state['onehot_known'] = ((jnp.arange(NCOL)[None, :] == col_eff[:, None]) & known_eff[:, None])
    return state


def _observe_action(state):
    """Prefer cards nobody has seen (team coverage), then cards only I lack."""
    known = state['known']
    pseen = state['pseen']
    score = (OBS_UNKNOWN * (~known).astype(jnp.float32)
             + OBS_UNSEEN * (~pseen).astype(jnp.float32)
             - 0.001 * jnp.arange(NCARD, dtype=jnp.float32))
    score = jnp.where(state['legal'][A_OBS:A_OBS + NCARD], score, NEG)
    return A_OBS + jnp.argmax(score).astype(jnp.int32)


def _move_action(state):
    """Greedy: minimise same-colour components over locally known cards."""
    adj = state['adj']
    same = state['same']
    known = state['known']
    pos = state['pos']
    eye = _eye()
    not_eye = ~eye
    lower = _lower()
    cell_xy = _cell_xy()
    alive = not_eye  # alive[m, j] == (j != m)

    # Colour-restricted reachability of the board with each card m removed.
    edges = (adj & same)[None] & alive[:, :, None] & alive[:, None, :]
    reach = _closure(edges | eye[None])
    member = known[None, None, :] & alive[:, None, :]
    size = jnp.sum(reach & member, axis=2)
    comps = jnp.sum(
        jnp.where(known[None, :] & alive, 1.0 / jnp.maximum(size, 1).astype(jnp.float32), 0.0),
        axis=1,
    )

    # Lowest-index representative of each component, used to count how many
    # distinct components of m's colour a destination cell would join.
    has_lower = jnp.any(reach & lower[None] & member, axis=2)
    rep = known[None, :] & alive & (~has_lower)
    joinable = (same[:, :, None] & alive[:, :, None] & reach & rep[:, None, :]).astype(jnp.float32)

    nb = (jnp.sum(jnp.abs(cell_xy[:, None, :] - pos[None, :, :]), axis=-1) == 1).astype(jnp.float32)
    touched = jnp.einsum('dj,mjq->mdq', nb, joinable) > 0.5
    merged = jnp.sum(touched, axis=2).astype(jnp.float32)
    phi = comps[:, None] + jnp.where(known[:, None], 1.0 - merged, 0.0)

    # Sum of squared cluster sizes.  The component count alone is blind to the
    # commonest useful move -- feeding one card of a two-card group into another
    # two-card group of the same colour leaves the count unchanged -- whereas
    # this potential strictly rewards moving a card into a group at least as big
    # as the one it leaves.
    fsize = size.astype(jnp.float32)
    concentration = jnp.sum(jnp.where(known[None, :] & alive, fsize, 0.0), axis=1)
    joined = jnp.sum(jnp.where(touched, fsize[:, None, :], 0.0), axis=2)
    joined_sq = jnp.sum(jnp.where(touched, fsize[:, None, :] ** 2, 0.0), axis=2)
    concentration = concentration[:, None] + jnp.where(
        known[:, None], (1.0 + joined) ** 2 - joined_sq, 0.0)

    bond = (adj & same).astype(jnp.float32)
    bonds_wo = jnp.sum(bond) / 2.0 - jnp.sum(bond, axis=1)
    new_bonds = jnp.einsum('dj,mj->md', nb, (same & not_eye).astype(jnp.float32))

    p = pos.astype(jnp.float32)
    cell = cell_xy.astype(jnp.float32)
    s1 = jnp.sum(p, axis=0)[None, None, :] - p[:, None, :] + cell[None, :, :]
    s2 = jnp.sum(p * p, axis=0)[None, None, :] - (p * p)[:, None, :] + (cell * cell)[None, :, :]
    spread = jnp.sum(s2 - s1 * s1 / float(NCARD), axis=2)

    # Expected same-colour bonds involving cards this seat has never seen.  A
    # merge of two locally known cards is worthless if it tears the moved card
    # away from an unseen card of the same colour, which is common because the
    # partner clusters exactly the cards this seat is missing.  Counts of the
    # 4-per-colour deck give the posterior for an unseen card.
    q = state['colour_posterior']
    soft = jnp.matmul(q, q.T) * (~(known[:, None] & known[None, :])) * not_eye

    lost_soft = jnp.sum(adj.astype(jnp.float32) * soft, axis=1)
    gain_soft = jnp.einsum('dj,mj->md', nb, soft)

    score = (-W_PHI * phi + W_SIZE * concentration
             + W_BOND * (bonds_wo[:, None] + new_bonds)
             - W_SPREAD * spread - W_UNKNOWN * (~known[:, None]).astype(jnp.float32)
             + W_SOFT * (gain_soft - lost_soft[:, None]))
    legal_move = state['legal'][A_MOVE:A_MOVE + NCARD * NCELL].reshape(NCARD, NCELL)
    score = jnp.where(legal_move, score, NEG)
    return A_MOVE + jnp.argmax(score.reshape(-1)).astype(jnp.int32)


def _hint_action(state):
    """Reveal a wide hint, or lock in a hint that is certainly correct."""
    legal = state['legal']
    pos = state['pos']
    known = state['known']
    col = state['col']
    eye = _eye()
    hint_size = _hint_size()

    cell_index = pos[:, 0] * GRID + pos[:, 1]
    cmask = (jnp.arange(NCELL)[:, None] == cell_index[None, :]).astype(jnp.float32)
    cell_col = jnp.matmul(cmask, state['onehot_known'].astype(jnp.float32))  # [100, 4]
    cell_known = jnp.sum(cell_col, axis=1)

    hs = state['hset'].astype(jnp.float32)
    correct = jnp.einsum('ck,hk->hc', cell_col, hs)
    counts = jnp.sum(state['onehot_known'].astype(jnp.float32), axis=0)
    remaining = 4.0 - counts
    n_unknown = jnp.maximum(float(NCARD) - jnp.sum(known.astype(jnp.float32)), 1.0)
    guess = jnp.matmul(hs, remaining) / n_unknown
    prob = jnp.where(cell_known[None, :] > 0.5, correct, guess[:, None])

    reach = _closure((state['adj'] & state['same']) | eye)
    csize = jnp.sum(reach & known[None, :], axis=1).astype(jnp.float32)
    complete = ((counts[jnp.clip(col, 0, NCOL - 1)] == 4.0) & (csize == 4.0) & known)
    cluster = jnp.where(known, jnp.minimum(csize, 4.0), 0.0)
    cell_cluster = jnp.matmul(cmask, cluster)
    cell_complete = jnp.matmul(cmask, complete.astype(jnp.float32))
    cell_pseen = jnp.matmul(cmask, state['pseen'].astype(jnp.float32))

    place_legal = legal[A_PLACE:A_PLACE + NHINT * NCELL].reshape(NHINT, NCELL)
    place_score = (1000.0 * prob + 100.0 * cell_complete[None, :] + 25.0 * cell_cluster[None, :]
                   + 5.0 * cell_pseen[None, :] + 30.0 * (3.0 - hint_size)[:, None]
                   - 0.001 * jnp.arange(NHINT * NCELL, dtype=jnp.float32).reshape(NHINT, NCELL))
    place_score = jnp.where(place_legal, place_score, NEG)
    place_action = A_PLACE + jnp.argmax(place_score.reshape(-1)).astype(jnp.int32)

    reveal_legal = legal[A_REVEAL:A_REVEAL + NHINT]
    reveal_score = jnp.where(reveal_legal,
                             10.0 * hint_size - jnp.arange(NHINT, dtype=jnp.float32), NEG)
    reveal_action = A_REVEAL + jnp.argmax(reveal_score).astype(jnp.int32)

    # Lock a card only when this seat has seen its colour and it already sits in
    # a cluster of at least PLACE_MIN_CLUSTER same-colour cards, so the lock
    # never freezes a card that still has to move.  Otherwise pay for a reveal.
    safe_place = jnp.any(place_legal & (prob > 0.999)
                         & (cell_cluster[None, :] >= PLACE_MIN_CLUSTER))
    must_place = ~jnp.any(reveal_legal)
    return jnp.where(safe_place | must_place, place_action, reveal_action)


def _end_decision(state):
    """End only when the arrangement is provably solved, or solved under a
    tightly constrained reading of the cards this seat has never seen."""
    known = state['known']
    col = state['col']
    adj = state['adj']
    counts = jnp.sum(state['onehot_known'].astype(jnp.float32), axis=0)
    remaining = 4.0 - counts
    n_unknown = float(NCARD) - jnp.sum(known.astype(jnp.float32))

    # Proof that every colouring still consistent with this seat's information
    # leaves all four colours connected.  For colour c let S be the cards seen to
    # be c and U the unseen cards that could still be c.  Either every member of
    # U must be c (then S + U has to be connected), or c needs a strict subset of
    # U, in which case S connected plus "every candidate touches S" makes any
    # choice connected.  No guess about the partner is involved.
    eye = _eye()
    seen_c = state['onehot_known']  # [16, 4]
    candidate = (~known[:, None]) & state['possible']
    touches = jnp.matmul(adj.astype(jnp.float32), seen_c.astype(jnp.float32)) > 0.5

    def colour_ok(c):
        s = seen_c[:, c]
        u = candidate[:, c]
        s_reach = _closure((adj & (s[:, None] & s[None, :])) | eye)
        s_conn = jnp.all(~s | (jnp.sum(s_reach & s[None, :], axis=1) == jnp.sum(s)))
        both = s | u
        b_reach = _closure((adj & (both[:, None] & both[None, :])) | eye)
        b_conn = jnp.all(~both | (jnp.sum(b_reach & both[None, :], axis=1) == jnp.sum(both)))
        all_needed = jnp.sum(u.astype(jnp.float32)) <= remaining[c]
        attached = jnp.all(~u | touches[:, c])
        return jnp.where(
            remaining[c] < 0.5,
            s_conn & (jnp.sum(s.astype(jnp.float32)) == 4.0),
            (all_needed & b_conn) | (s_conn & jnp.any(s) & attached),
        )

    certain = jnp.all(jnp.stack([colour_ok(c) for c in range(NCOL)]))

    # Otherwise fall back on the board's own evidence: an unseen card counts as
    # settled when the partner has seen it and its neighbours point at a colour
    # that still has cards to place.  Reading the resulting colouring off the
    # board only pays off if it is globally consistent -- four cards of every
    # colour, all four connected -- which is what makes this test safe enough to
    # use.  It is a judgement about the partner, so it is gated on being worth
    # more than the 7 points a solved board scores by running to the cap.
    nbr = jnp.matmul(adj.astype(jnp.float32), state['onehot_known'].astype(jnp.float32))
    best = jnp.max(nbr, axis=1)
    putative = jnp.argmax(nbr, axis=1)
    settled = (best > 0.5) & (remaining[putative] > 0.5) & state['pseen']
    guess_col = jnp.where(known, col, putative)
    guess_counts = jnp.sum((guess_col[:, None] == jnp.arange(NCOL)[None, :]), axis=0)
    likely = (jnp.all(known | settled) & jnp.all(guess_counts == NCARD // NCOL)
              & _all_colours_connected(adj, guess_col))

    n_rev = jnp.sum(state['hrev'].astype(jnp.float32))
    n_place = jnp.sum(state['hplaced'].astype(jnp.float32))
    potential = 5.0 * (NHINT - n_rev) + 2.0 * (n_rev - n_place) + n_place
    return (certain | (likely & (potential >= MIN_END_POTENTIAL))), potential, n_unknown


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    state = _decode(obs)
    legal = state['legal']
    substep = state['substep']

    end_now, potential, n_unknown = _end_decision(state)
    observe = _observe_action(state)
    move = _move_action(state)
    hint = _hint_action(state)

    action = jnp.where(
        substep == 0,
        jnp.where(end_now, A_END, observe),
        jnp.where(substep == 1, observe, jnp.where(substep == 2, move, hint)),
    )
    action = jnp.where(state['active'], action, 0).astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))

    eye = _eye()
    reach = _closure((state['adj'] & state['same']) | eye)
    csize = jnp.sum(reach & state['known'][None, :], axis=1)
    phi = jnp.sum(jnp.where(state['known'], 1.0 / jnp.maximum(csize, 1).astype(jnp.float32), 0.0))
    trace = {
        'phi': (10.0 * phi).astype(jnp.int32),
        'known': jnp.sum(state['known'].astype(jnp.int32)),
        'unknown': n_unknown.astype(jnp.int32),
        'potential': potential.astype(jnp.int32),
        'end_now': end_now.astype(jnp.int32),
        'substep': substep.astype(jnp.int32),
    }
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action.astype(jnp.int32), jnp.where(reset, zero, memory), trace
