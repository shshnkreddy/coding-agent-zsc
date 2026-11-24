"""Belief-based cooperative planner for the two-player 9-card Yokai grid game.

Both seats run this code, and a seat only ever sees its own observation, so all
reasoning here is local.  Nothing in it is a private code: the policy never
tries to *send* a message, it only optimises the public objective (every color
orthogonally connected when the episode ends) from its own posterior, and it
reads the partner's public choices as weak evidence about what the partner saw.
That keeps it usable with a partner built to a completely different design.

What the planner knows
----------------------
* 9 cards, 3 colors, 3 cards each -> exactly 1680 color assignments.  The reset
  shuffle is uniform, so restricting a uniform prior to the assignments that
  agree with the colors I personally observed is an exact posterior.
* Three cards are connected exactly when two of their three pairs are
  orthogonally adjacent, so per-color connectivity is a three-term edge count
  and ``D = sum_c (2 - min(edges_c, 2))`` counts the relocations still owed.
  A move only changes the moving card's own color group, so D falls by at most
  one per move; with all colors known, descending D greedily is optimal.
* The initial 3x3 block is the 9-cell shape with the lowest expected D under a
  uniform coloring (E[D] = 3), so a move made before any color is known can
  only cost structure.  Information, not moves, is the binding constraint:
  two looks per turn, eight turns, and the score decays about two points per
  turn played.
* Hint scoring: 5 per unrevealed, 2 per revealed-unplaced, +-1 per placement.
  Revealing costs 3 points; placing costs 3 - 2 * P(correct).  Placing is
  therefore the cheaper substep-3 action whenever a covered card is both known
  and safe to freeze, since a placement also locks the card for good.

How the four substeps are decided
---------------------------------
0. End the episode when P(connected) * hint score beats a model of what
   playing on is worth; otherwise take a look.
1. Look at the color that adds most: unseen by anybody, breaking ties towards
   the card whose color would most move P(connected) (verification late on).
2. Move the card/destination pair maximising the posterior-expected discounted
   relocation cost, evaluated exactly over all 1680 assignments.
3. Place a hint whose color set covers a card I know and that is not the card
   still owing a relocation; otherwise reveal, keeping the one-color hint back.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'yokai_belief_planner', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
NUM_HINTS = 4
GRID = 9
NUM_CELLS = GRID * GRID
NUM_TRIPLES = 84
NUM_PARTS = 1680

MOVE_BASE = 10
REVEAL_BASE = 739
PLACE_BASE = 743
END_ACTION = 1067

# --- constants tuned against the official development scorer ---------------
LOG_BETA = -0.5978370008   # log(0.55): discount per relocation still owed
# how much of a same-color bond a pair at Manhattan distance d already supplies
CLOSENESS = [0.0,1.0,0.12,0.04,0.015,0.005,0.0,0.0,0.0]
COMPACT_W = 0.004    # tie-break: keep the blob compact so moves stay legal
RECENT_PEN = 0.02    # tie-break: avoid immediately undoing the last move
LOCK_PEN = 3.5       # points charged for freezing a card that still must move
COVER_W = 0.5        # prefer observing cards the partner has not seen either
VERIFY_W = 2.0       # weight on how much a look can swing P(board connected)
MOVE_LAM = 1.5       # soft likelihood ratio for "the partner joined two colors"
HINT_LAM = 2.0       # soft likelihood ratio for "the partner covered a match"
TURN_COST = 3.6      # hint points written off per further turn played
SIGMA = 0.72         # probability a still-needed relocation actually happens
TAU = 0.78           # probability an already-solved board survives to the end
MAX_TURNS = 8


def _build_tables():
    triples = []
    tindex = {}
    for i in range(NUM_CARDS):
        for j in range(i + 1, NUM_CARDS):
            for k in range(j + 1, NUM_CARDS):
                tindex[(i, j, k)] = len(triples)
                triples.append((i, j, k))
    parts_tri = []
    parts_col = []
    for a in range(len(triples)):
        t0 = triples[a]
        rest = [c for c in range(NUM_CARDS) if c not in t0]
        for x in range(6):
            for y in range(x + 1, 6):
                for z in range(y + 1, 6):
                    t1 = (rest[x], rest[y], rest[z])
                    t2 = tuple(c for c in rest if c not in t1)
                    parts_tri.append((a, tindex[t1], tindex[t2]))
                    col = [0] * NUM_CARDS
                    for c in t1:
                        col[c] = 1
                    for c in t2:
                        col[c] = 2
                    parts_col.append(col)
    card_tri = [[parts_tri[p][parts_col[p][i]] for i in range(NUM_CARDS)]
                for p in range(len(parts_tri))]
    pair_j = []
    pair_k = []
    pair_tri = []
    for i in range(NUM_CARDS):
        others = [c for c in range(NUM_CARDS) if c != i]
        rj, rk, rt = [], [], []
        for a in range(len(others)):
            for b in range(a + 1, len(others)):
                j, k = others[a], others[b]
                rj.append(j)
                rk.append(k)
                rt.append(tindex[tuple(sorted((i, j, k)))])
        pair_j.append(rj)
        pair_k.append(rk)
        pair_tri.append(rt)
    return triples, parts_tri, parts_col, card_tri, pair_j, pair_k, pair_tri


class _Tables(object):
    """Static combinatorial tables; instantiated at trace time, not at import.

    The evaluator forbids calls at module scope, so the constants live here and
    are rebuilt whenever ``agent_act`` is traced.  Nothing depends on traced
    values, so XLA folds all of it into compile-time constants.
    """

    def __init__(self):
        tri, part_tri, part_col, card_tri, pj, pk, pt = _build_tables()
        self.TRI = jnp.asarray(tri, dtype=jnp.int32)              # (84, 3)
        self.PART_TRI = jnp.asarray(part_tri, dtype=jnp.int32)    # (1680, 3)
        self.PART_COL = jnp.asarray(part_col, dtype=jnp.int32)    # (1680, 9)
        self.CARD_TRI = jnp.asarray(card_tri, dtype=jnp.int32)    # (1680, 9)
        self.PAIR_J = jnp.asarray(pj, dtype=jnp.int32)            # (9, 28)
        self.PAIR_K = jnp.asarray(pk, dtype=jnp.int32)            # (9, 28)
        self.PAIR_TRI = jnp.asarray(pt, dtype=jnp.int32)          # (9, 28)
        # one-hot of CARD_TRI: regroups the posterior by "triple holding card i"
        self.CARD_TRI_OH = (
            self.CARD_TRI.T[:, :, None] ==
            jnp.arange(NUM_TRIPLES, dtype=jnp.int32)[None, None, :]
        ).astype(jnp.float32)                                     # (9, 1680, 84)
        self.PART_COL_OH = (
            self.PART_COL[:, :, None] ==
            jnp.arange(NUM_COLORS, dtype=jnp.int32)[None, None, :]
        ).astype(jnp.float32)                                     # (1680, 9, 3)
        self.CELL_X = jnp.asarray([z // GRID for z in range(NUM_CELLS)], dtype=jnp.int32)
        self.CELL_Y = jnp.asarray([z % GRID for z in range(NUM_CELLS)], dtype=jnp.int32)
        self.CLOSE = jnp.asarray(CLOSENESS, dtype=jnp.float32)
        self.SIGMA_POW = jnp.asarray([SIGMA ** d for d in range(7)], dtype=jnp.float32)


def init_agent():
    return {
        'known': -jnp.ones((NUM_CARDS,), dtype=jnp.int32),
        'last_pos': jnp.zeros((NUM_CARDS, 2), dtype=jnp.int32),
        'ref_pos': jnp.zeros((NUM_CARDS, 2), dtype=jnp.int32),
        'ev': jnp.zeros((NUM_PARTS,), dtype=jnp.float32),
    }


def _grid_dist(pos):
    """Manhattan distance between cards (9, 9) from integer positions."""
    return jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)


def _triple_stats(TB, dij, dik, djk):
    """Return (integer moves still needed, graded potential cost) for a triple.

    A color triple is connected exactly when two of its three pairs are
    orthogonally adjacent, so ``2 - min(edges, 2)`` counts the relocations that
    are strictly still required.  For steering moves that is too coarse: it
    gives no credit for bringing same-color cards closer together.  The graded
    cost therefore scores each pair by proximity and keeps the two best bonds,
    which is exactly what a connected triple needs.
    """
    edges = ((dij == 1).astype(jnp.int32) + (dik == 1).astype(jnp.int32) +
             (djk == 1).astype(jnp.int32))
    int_cost = 2 - jnp.minimum(edges, 2)
    cij = TB.CLOSE[jnp.clip(dij, 0, 8)]
    cik = TB.CLOSE[jnp.clip(dik, 0, 8)]
    cjk = TB.CLOSE[jnp.clip(djk, 0, 8)]
    best_two = cij + cik + cjk - jnp.minimum(jnp.minimum(cij, cik), cjk)
    return int_cost, 2.0 - best_two


def _triple_costs(TB, dmat):
    i, j, k = TB.TRI[:, 0], TB.TRI[:, 1], TB.TRI[:, 2]
    return _triple_stats(TB, dmat[i, j], dmat[i, k], dmat[j, k])


def _consistent(TB, known):
    """Indicator of the assignments compatible with the colors I looked at."""
    ok = (known[None, :] < 0) | (TB.PART_COL == known[None, :])
    return jnp.all(ok, axis=1).astype(jnp.float32)


def _normalize(w):
    total = w.sum()
    w = jnp.where(total > 0.0, w, jnp.ones_like(w))
    return w / jnp.maximum(w.sum(), 1e-9)


def agent_act(obs, reset, memory, rng):
    TB = _Tables()
    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(jnp.bool_)
    substep = nf[0, 11]
    active = nf[0, 10] > 0

    # ---- private knowledge ------------------------------------------------
    onehot = nf[0:NUM_CARDS, 0:NUM_COLORS]
    seen_now = onehot.sum(-1) > 0
    color_now = jnp.argmax(onehot, axis=-1)
    prev_known = jnp.where(reset, -jnp.ones((NUM_CARDS,), jnp.int32), memory['known'])
    known = jnp.where(seen_now, color_now, prev_known)
    last_pos = jnp.where(reset, pos, memory['last_pos'])

    partner_seen = (nf[0:NUM_CARDS, 9] > 0).astype(jnp.float32)
    dmat = _grid_dist(pos)

    # ---- hint bookkeeping -------------------------------------------------
    hrows = nf[NUM_CARDS:NUM_CARDS + NUM_HINTS]
    hint_set = hrows[:, 3:6]                              # (4, 3) multi-hot
    revealed = (hint_set.sum(-1) > 0)
    placed = hrows[:, 8] > 0
    hint_pos = hrows[:, 6:8]
    num_revealed = jnp.sum(revealed.astype(jnp.int32))
    num_placed = jnp.sum(placed.astype(jnp.int32))
    turns_done = num_revealed + num_placed
    unrevealed = NUM_HINTS - num_revealed
    waiting = num_revealed - num_placed

    cell_of_card = pos[:, 0] * GRID + pos[:, 1]           # (9,)
    card_at = -jnp.ones((NUM_CELLS,), jnp.int32).at[cell_of_card].set(
        jnp.arange(NUM_CARDS, dtype=jnp.int32))
    hcell = jnp.clip(hint_pos[:, 0], 0, GRID - 1) * GRID + jnp.clip(hint_pos[:, 1], 0, GRID - 1)
    hcard = jnp.clip(card_at[hcell], 0, NUM_CARDS - 1)

    # ---- evidence from the partner's public choices -----------------------
    # Nothing here assumes a signalling code.  It only assumes the partner is
    # trying to score: a player that knows two card colors tends to park them
    # side by side, and tends to cover a card with a hint that matches it.
    # Both updates are deliberately soft, so a clueless partner cannot drag the
    # posterior away from what I saw with my own eyes.
    ref_pos = jnp.where(reset, pos, memory['ref_pos'])
    ev_prev = jnp.where(reset, jnp.zeros((NUM_PARTS,), jnp.float32), memory['ev'])
    moved_mask = jnp.any(pos != ref_pos, axis=-1).astype(jnp.float32)
    pair_ok = ((dmat == 1).astype(jnp.float32) * partner_seen[:, None] *
               partner_seen[None, :] * moved_mask[:, None])            # (9, 9)
    vc = jnp.transpose(TB.PART_COL_OH, (0, 2, 1))                      # (1680, 3, 9)
    reach = jnp.einsum('pci,ij->pcj', vc, pair_ok)
    joined = (jnp.einsum('pcj,pcj->pc', reach, vc).sum(-1) > 0).astype(jnp.float32)
    do_update = (active & (substep == 0)).astype(jnp.float32)
    ev = ev_prev + do_update * jnp.log1p(MOVE_LAM * joined)
    # a placed hint is evidence about the covered card's color
    hit = jnp.einsum('hj,hc,pjc->ph',
                     (hcard[:, None] == jnp.arange(NUM_CARDS)[None, :]).astype(jnp.float32),
                     hint_set.astype(jnp.float32), TB.PART_COL_OH)
    hint_ev = jnp.sum(jnp.where(placed[None, :], jnp.log1p(HINT_LAM * hit), 0.0), axis=-1)

    w = _normalize(_consistent(TB, known) * jnp.exp(ev + hint_ev))
    marg = jnp.einsum('p,pjc->jc', w, TB.PART_COL_OH)        # (9, 3) color marginals

    # ---- current board evaluation ----------------------------------------
    int_cost, soft_cost = _triple_costs(TB, dmat)             # (84,), (84,)
    conn = (int_cost == 0).astype(jnp.float32)
    g = jnp.exp(LOG_BETA * soft_cost)                         # (84,)
    prod = g[TB.PART_TRI[:, 0]] * g[TB.PART_TRI[:, 1]] * g[TB.PART_TRI[:, 2]]
    dist = (int_cost[TB.PART_TRI[:, 0]] + int_cost[TB.PART_TRI[:, 1]] +
            int_cost[TB.PART_TRI[:, 2]])
    p_solved = jnp.sum(w * (dist == 0).astype(jnp.float32))

    # probability that hint h is correct on card j
    p_match = jnp.einsum('hc,jc->hj', hint_set.astype(jnp.float32), marg)
    placed_match = p_match[jnp.arange(NUM_HINTS), hcard]
    placed_value = jnp.sum(jnp.where(placed, 2.0 * placed_match - 1.0, 0.0))
    hint_score = 5.0 * unrevealed + 2.0 * waiting + placed_value

    # ---- end-game value comparison ---------------------------------------
    turns_needed = jnp.maximum(dist, 1)
    feasible = (turns_done + turns_needed <= MAX_TURNS).astype(jnp.float32)
    future = jnp.maximum(hint_score - TURN_COST * turns_needed.astype(jnp.float32), 0.0)
    survive = jnp.where(dist == 0, TAU, TB.SIGMA_POW[jnp.clip(dist, 0, 6)])
    v_continue = jnp.sum(w * survive * future * feasible)
    v_end = p_solved * hint_score
    want_end = (v_end > v_continue) & legal[END_ACTION]

    # ---- observation choice ----------------------------------------------
    obs_legal = legal[1:1 + NUM_CARDS]
    unknown = (known < 0).astype(jnp.float32)
    # Spend my two looks on colors nobody has seen: my partner already acts on
    # what it saw, so duplicating its cards adds nothing to the team's picture.
    # Late in the episode the scarce resource is certainty, not coverage: a
    # look is worth most when learning that color can swing the probability
    # that every color is already connected.
    solved_col = jnp.einsum('p,pjc->jc', w * (dist == 0).astype(jnp.float32),
                            TB.PART_COL_OH)                           # (9, 3)
    swing = jnp.sum(jnp.abs(solved_col - marg * p_solved), axis=-1)    # (9,)
    obs_score = (2.0 * unknown + COVER_W * (1.0 - partner_seen)
                 + VERIFY_W * swing
                 - 0.01 * jnp.arange(NUM_CARDS, dtype=jnp.float32))
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    observe_action = 1 + jnp.argmax(obs_score).astype(jnp.int32)

    # ---- move search ------------------------------------------------------
    # regroup the posterior by the triple that contains each card, so a
    # candidate move only needs the 28 triples through the moved card.
    v_part = (w * prod)[:, None] / g[TB.CARD_TRI]            # (1680, 9)
    regroup = jnp.einsum('pi,ipt->it', v_part, TB.CARD_TRI_OH)   # (9, 84)

    cell_dist = (jnp.abs(TB.CELL_X[:, None] - pos[None, :, 0]) +
                 jnp.abs(TB.CELL_Y[:, None] - pos[None, :, 1]))        # (81, 9)
    pj, pk = TB.PAIR_J, TB.PAIR_K                                      # (9, 28)
    base_pair = dmat[pj, pk]                                           # (9, 28)
    _, soft_new = _triple_stats(TB, cell_dist[:, pj], cell_dist[:, pk],
                                base_pair[None, :, :])                 # (81, 9, 28)
    g_new = jnp.exp(LOG_BETA * soft_new)
    rw = regroup[jnp.arange(NUM_CARDS)[:, None], TB.PAIR_TRI]          # (9, 28)
    move_val = jnp.einsum('zik,ik->iz', g_new, rw)        # (9, 81)

    cell_adj = (cell_dist == 1).astype(jnp.int32)
    compact = cell_adj.sum(-1)[None, :] - cell_adj.T      # (9, 81)
    moved_recent = jnp.any(pos != last_pos, axis=-1).astype(jnp.float32)
    move_val = (move_val + COMPACT_W * compact.astype(jnp.float32)
                - RECENT_PEN * moved_recent[:, None])
    move_legal = legal[MOVE_BASE:MOVE_BASE + NUM_CARDS * NUM_CELLS].reshape(
        NUM_CARDS, NUM_CELLS)
    move_val = jnp.where(move_legal, move_val, -1e9)
    move_action = MOVE_BASE + jnp.argmax(move_val.reshape(-1)).astype(jnp.int32)

    # ---- hint action ------------------------------------------------------
    # Locking a card only hurts if that card is the one that still has to be
    # relocated, i.e. if it has no same-color neighbour under the posterior.
    reach2 = jnp.einsum('pck,kj->pcj', vc, (dmat == 1).astype(jnp.float32))
    deg_same = jnp.einsum('pcj,pcj->pj', reach2, vc)                      # (1680, 9)
    lock_risk = jnp.einsum('p,pj->j', w, (deg_same == 0).astype(jnp.float32))
    cell_card = jnp.clip(card_at, 0, NUM_CARDS - 1)
    # spend the narrow hint first while a matching card is available
    width = hint_set.sum(-1).astype(jnp.float32)[:, None]
    place_val = (2.0 * p_match[:, cell_card] - 1.0
                 - LOCK_PEN * lock_risk[cell_card][None, :]
                 - 0.05 * width) - 2.0                                # (4, 81)
    # placing the fourth hint ends the episode immediately
    place_val = place_val - jnp.where(num_placed == NUM_HINTS - 1,
                                      8.0 * (1.0 - p_solved), 0.0)
    place_legal = legal[PLACE_BASE:PLACE_BASE + NUM_HINTS * NUM_CELLS].reshape(
        NUM_HINTS, NUM_CELLS)
    place_val = jnp.where(place_legal, place_val, -1e9)
    best_place = jnp.argmax(place_val.reshape(-1))
    place_action = PLACE_BASE + best_place.astype(jnp.int32)
    reveal_legal = legal[REVEAL_BASE:REVEAL_BASE + NUM_HINTS]
    # hint 0 always carries a single color and hints 1..3 a pair, so keep the
    # narrow hint in reserve: a wide hint is easier to place correctly later.
    reveal_score = jnp.where(reveal_legal,
                             jnp.asarray([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
                             - 0.01 * jnp.arange(NUM_HINTS, dtype=jnp.float32), -1e9)
    reveal_action = REVEAL_BASE + jnp.argmax(reveal_score).astype(jnp.int32)
    take_place = (place_val.reshape(-1)[best_place] > -3.0) | (~jnp.any(reveal_legal))
    take_place = take_place & jnp.any(place_legal)
    hint_action = jnp.where(take_place, place_action, reveal_action)

    # ---- assemble ---------------------------------------------------------
    action = jnp.where(substep == 0, jnp.where(want_end, END_ACTION, observe_action),
                       jnp.where(substep == 1, observe_action,
                                 jnp.where(substep == 2, move_action, hint_action)))
    action = action.astype(jnp.int32)
    fallback = jnp.argmax(legal).astype(jnp.int32)
    action = jnp.where(legal[action], action, fallback)
    action = jnp.where(active, action, jnp.zeros((), jnp.int32))

    new_last = jnp.where(substep == 2, pos, last_pos)
    next_memory = {
        'known': known,
        'last_pos': new_last,
        'ref_pos': jnp.where(active & (substep >= 3), pos, ref_pos),
        'ev': ev,
    }
    trace = {
        'p_solved': p_solved.astype(jnp.float32),
        'exp_dist': jnp.sum(w * dist.astype(jnp.float32)),
        'hint_score': hint_score.astype(jnp.float32),
        'v_end': v_end.astype(jnp.float32),
        'v_continue': v_continue.astype(jnp.float32),
        'n_known': jnp.sum((known >= 0).astype(jnp.float32)),
        'turns_done': turns_done.astype(jnp.float32),
        'action': action.astype(jnp.float32),
    }
    return action, next_memory, trace
