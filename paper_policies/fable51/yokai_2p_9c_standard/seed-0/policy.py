"""Yokai 2p/9-card policy: exact posterior over colorings + greedy clustering.

Design (partner-agnostic):
  * Private memory keeps log-weights over all 1680 colorings of the 9 stable
    card IDs (3 cards per color).  Own observations are hard evidence; the
    3-per-color constraint yields deductions automatically.
  * Partner moves are mild soft evidence ("moved card is likely the same color
    as a card it was placed next to"), partner hint placements are mild soft
    evidence ("placed hint is probably correct").  Both are weak factors so a
    careless partner cannot make us over-confident.
  * Observe: card with highest marginal color entropy.
  * Move: maximize expected number of same-color adjacencies gained (for a
    3-card color, components = 3 - edges), with small shape tie-breaks.
  * Hint: place a revealed hint only where it is certainly correct and the
    card's color group is already complete; otherwise reveal (ID order);
    forced placements pick the most probably-correct, least disruptive card.
  * End: at substep 0 end when posterior P(all colors connected) exceeds a
    stage-dependent threshold derived from the current hint value.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'posterior_cluster_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NCARD = 9
NCOL = 3
GRID = 9
NCELL = GRID * GRID
NHINT = 4
NCOLORING = 1680
NEG = -1e9

# Soft-evidence strengths (log factors); deliberately mild.
MAX_EV = 4                 # at most 4 partner moves per episode
# join evidence: partner created an adjacency between moved card k and a card of a different colour
LOG_BETA_OBS = -1.897          # log 0.15: credible join (partner observed moved card and a new neighbour)
LOG_BETA_OBS_GOOD = -2.303     # log 0.10 after a verified-consistent credible join and no inconsistent one
LOG_BETA_OBS_BAD = -0.357      # log 0.70 after a verified-inconsistent credible join
LOG_BETA_UNOBS = -0.357        # log 0.70: partner observed only the moved card
LOG_BETA_UNOBS_GOOD = -0.511   # log 0.60
LOG_BETA_UNOBS_BAD = -0.105    # log 0.90
LOG_BETA_GUESS = -0.163        # log 0.85: partner never observed the moved card (filler / guess)
LOG_HINT_WRONG = -1.204        # log 0.30: partner placed an incorrect hint (forced late placements may be guesses)
LOG_HINT_WRONG_BAD = -0.357    # log 0.70 once the partner has placed a verifiably wrong hint
LOG_NOT_JOINED = -0.598        # log 0.55: partner observed two cards this turn and did not join them
LOG_NOT_JOINED_BAD = -0.163    # log 0.85 for a partner with verified-inconsistent joins
MI_WEIGHT = 1.0            # weight of connectivity-relevance when choosing which card to observe
LOOKAHEAD_K = 48           # number of first-move candidates evaluated with a second-move lookahead
LOOKAHEAD_GAMMA = 0.7      # discount on the best follow-up move's gain
CONT_P = 0.85              # assumed P(all connected) if we keep playing instead of ending now
CONT_P_DESTR = 0.6         # same, once the partner has been seen breaking a known same-colour adjacency
CONT_P_LAST = 0.75         # last turn: probability that one final fix move suffices


def _tables():
    """Constant tables built with jnp ops (validator forbids module-scope calls)."""
    codes = jnp.arange(3 ** NCARD)
    digits = jnp.stack([(codes // (3 ** i)) % 3 for i in range(NCARD)], axis=1)      # [19683, 9]
    counts = jnp.stack([(digits == c).sum(1) for c in range(NCOL)], axis=1)          # [19683, 3]
    valid = jnp.all(counts == 3, axis=1)
    idx = jnp.nonzero(valid, size=NCOLORING, fill_value=0)[0]
    col = digits[idx].astype(jnp.int32)                                              # [1680, 9]
    onehot = jax.nn.one_hot(col, NCOL, dtype=jnp.float32)                            # [1680, 9, 3]
    eye9 = jnp.eye(NCARD, dtype=bool)
    same = (col[:, :, None] == col[:, None, :]) & ~eye9[None]                        # [1680, 9, 9]
    cx = (jnp.arange(NCELL) // GRID).astype(jnp.float32)
    cy = (jnp.arange(NCELL) % GRID).astype(jnp.float32)
    cd = jnp.abs(cx[:, None] - cx[None, :]) + jnp.abs(cy[:, None] - cy[None, :])
    cell_adj = (cd == 1).astype(jnp.float32)                                         # [81, 81]
    return {
        'col': col, 'onehot': onehot, 'same': same, 'same_f': same.astype(jnp.float32),
        'cell_adj': cell_adj, 'cell_deg': cell_adj.sum(1), 'cell_x': cx, 'cell_y': cy,
    }


def _init_pos():
    i = jnp.arange(NCARD)
    return jnp.stack([i // 3 + 3, i % 3 + 3], axis=1).astype(jnp.int32)


def init_agent():
    return {
        'known': -jnp.ones((NCARD,), dtype=jnp.int32),
        'prev_pos': _init_pos(),
        'prev_placed': jnp.zeros((NHINT,), dtype=jnp.int32),
        'my_placed': jnp.zeros((NHINT,), dtype=jnp.int32),
        'partner_obs_ever': jnp.zeros((NCARD,), dtype=jnp.int32),
        'partner_moved': jnp.zeros((NCARD,), dtype=jnp.int32),
        # partner join events: [valid, moved card k, k-observed-by-partner flag, fresh-neighbour mask(9)]
        'join_ev': jnp.zeros((MAX_EV, 3 + NCARD), dtype=jnp.int32),
        # partner "observed two cards and did not join them" events: [valid, i1, i2]
        'nj_ev': jnp.zeros((MAX_EV, 3), dtype=jnp.int32),
        'n_join': jnp.zeros((), dtype=jnp.int32),
        'n_nj': jnp.zeros((), dtype=jnp.int32),
        'n_destr': jnp.zeros((), dtype=jnp.int32),
    }


def _card_adjacency(pos):
    d = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    return (d == 1).astype(jnp.float32)


def agent_act(obs, reset, memory, rng):
    T = _tables()
    COL, ONEHOT, SAME, SAME_F = T['col'], T['onehot'], T['same'], T['same_f']
    CELL_ADJ, CELL_DEG, CELL_X, CELL_Y = T['cell_adj'], T['cell_deg'], T['cell_x'], T['cell_y']
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    nf = obs.node_features.astype(jnp.int32)
    adj = obs.adjacency_matrix.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(bool)

    active = nf[0, 10] > 0
    substep = nf[0, 11]
    own = nf[:NCARD, 0:3]
    has_own = own.sum(1) > 0
    own_col = jnp.argmax(own, axis=1)
    partner_seen = nf[:NCARD, 9] > 0
    locked = nf[:NCARD, 8] > 0
    hset = nf[NCARD:NCARD + NHINT, 3:6] > 0          # [4,3]
    hset_f = hset.astype(jnp.float32)
    hplaced = nf[NCARD:NCARD + NHINT, 8] > 0         # [4]
    hrevealed = hset.any(1)                          # [4]
    # target card of each (placed) hint
    htarget = jnp.argmax(adj[NCARD:NCARD + NHINT, :NCARD], axis=1)   # [4]

    A = _card_adjacency(pos)                         # [9,9]
    cell = pos[:, 0] * GRID + pos[:, 1]              # [9]

    # ---------------- memory update: hard evidence ----------------
    known = jnp.where(has_own, own_col, mem['known'])
    consistent = jnp.all((known[None, :] < 0) | (COL == known[None, :]), axis=1)   # [1680]
    logw_hard = jnp.where(consistent, 0.0, NEG)
    # hard knowledge including deductions from the 3-per-colour constraint
    w_hard = jax.nn.softmax(logw_hard)
    marg_hard = jnp.einsum('x,xjc->jc', w_hard, ONEHOT)
    known_hard = jnp.where(marg_hard.max(1) > 0.99, jnp.argmax(marg_hard, axis=1), -1)   # [9]

    # ---------------- memory update: record partner events ----------------
    prev_pos = mem['prev_pos']
    moved = jnp.any(pos != prev_pos, axis=1)
    n_moved = moved.sum()
    k = jnp.argmax(moved)
    is_partner_move = (~active) & (n_moved == 1)
    A_old = _card_adjacency(prev_pos)
    newN = A[k] > 0
    oldN = A_old[k] > 0
    fresh = newN & ~oldN                                     # adjacencies created by the move
    partner_moved = jnp.where(is_partner_move, mem['partner_moved'].at[k].set(1), mem['partner_moved'])
    # destructive partner move: separated a card from a same-colour neighbour we know about
    left = oldN & ~newN
    kh = known_hard[k]
    destr = is_partner_move & (kh >= 0) & jnp.any(left & (known_hard == kh))
    n_destr = mem['n_destr'] + destr.astype(jnp.int32)
    # a move creating no new adjacency (a slide along existing neighbours) carries no join evidence
    rec_join = is_partner_move & jnp.any(fresh)
    slot_j = jnp.minimum(mem['n_join'], MAX_EV - 1)
    # credibility: 2 = partner has observed the moved card and a new neighbour (a deliberate join),
    # 1 = partner observed only the moved card, 0 = partner never observed the moved card (a guess)
    pobs_now = jnp.maximum(mem['partner_obs_ever'], partner_seen.astype(jnp.int32))
    k_obs = pobs_now[k] > 0
    nb_obs = jnp.any(fresh & (pobs_now > 0))
    cred = jnp.where(k_obs & nb_obs, 2, jnp.where(k_obs, 1, 0))
    join_row = jnp.concatenate([jnp.array([1, 0, 0], jnp.int32).at[1].set(k).at[2].set(cred),
                                fresh.astype(jnp.int32)])
    join_ev = jnp.where(rec_join, mem['join_ev'].at[slot_j].set(join_row), mem['join_ev'])
    n_join = mem['n_join'] + rec_join.astype(jnp.int32)
    # partner observed two cards this turn and did not join them
    seen_f = partner_seen.astype(jnp.float32)
    i1 = jnp.argmax(seen_f)
    i2 = jnp.argmax(seen_f.at[i1].set(0.0))
    two_seen = (jnp.sum(seen_f) >= 1.5) & (A[i1, i2] < 0.5) & (~locked[i1]) & (~locked[i2])
    rec_nj = is_partner_move & two_seen
    slot_n = jnp.minimum(mem['n_nj'], MAX_EV - 1)
    nj_row = jnp.array([1, 0, 0], jnp.int32).at[1].set(i1).at[2].set(i2)
    nj_ev = jnp.where(rec_nj, mem['nj_ev'].at[slot_n].set(nj_row), mem['nj_ev'])
    n_nj = mem['n_nj'] + rec_nj.astype(jnp.int32)
    partner_obs_ever = jnp.maximum(mem['partner_obs_ever'], partner_seen.astype(jnp.int32))

    # ---------------- soft evidence recomputed from events with adaptive trust ----------------
    # verify recorded joins against current hard knowledge
    good = jnp.zeros((), jnp.float32)
    bad = jnp.zeros((), jnp.float32)
    join_viol = []
    for e in range(MAX_EV):
        valid = join_ev[e, 0] > 0
        ke = join_ev[e, 1]
        credible = join_ev[e, 2] >= 2
        fr = join_ev[e, 3:] > 0                                          # [9]
        kc = known_hard[ke]
        nb_known = fr & (known_hard >= 0)
        cons = (kc >= 0) & jnp.any(nb_known & (known_hard == kc))
        incons = (kc >= 0) & jnp.any(fr) & jnp.all(~fr | ((known_hard >= 0) & (known_hard != kc)))
        good = good + jnp.where(valid & credible & cons, 1.0, 0.0)
        bad = bad + jnp.where(valid & credible & incons, 1.0, 0.0)
        viol = valid & ~jnp.any(SAME[:, ke, :] & fr[None, :], axis=1)   # [1680] colourings where the join is not same-colour
        join_viol.append((viol, join_ev[e, 2]))
    trusted = (good >= 1.0) & (bad < 0.5)
    untrusted = bad >= 0.5
    lb2 = jnp.where(untrusted, LOG_BETA_OBS_BAD, jnp.where(trusted, LOG_BETA_OBS_GOOD, LOG_BETA_OBS))
    lb1 = jnp.where(untrusted, LOG_BETA_UNOBS_BAD, jnp.where(trusted, LOG_BETA_UNOBS_GOOD, LOG_BETA_UNOBS))
    logw = logw_hard
    for viol, cred_e in join_viol:
        lb = jnp.where(cred_e >= 2, lb2, jnp.where(cred_e == 1, lb1, LOG_BETA_GUESS))
        logw = logw + jnp.where(viol, lb, 0.0)
    log_nj = jnp.where(untrusted, LOG_NOT_JOINED_BAD, LOG_NOT_JOINED)
    for e in range(MAX_EV):
        valid = nj_ev[e, 0] > 0
        logw = logw + jnp.where(valid & SAME[:, nj_ev[e, 1], nj_ev[e, 2]], log_nj, 0.0)
    # partner hint placements (read directly from the observation; own placements excluded)
    hbad = jnp.zeros((), jnp.float32)
    hint_viols = []
    for h in range(NHINT):
        j = htarget[h]
        partner_hint = hplaced[h] & (mem['my_placed'][h] == 0)
        in_s = (ONEHOT[:, j, :] * hset_f[h][None, :]).sum(1) > 0      # [1680]
        known_wrong = (known_hard[j] >= 0) & (hset_f[h][jnp.clip(known_hard[j], 0, 2)] < 0.5)
        hbad = hbad + jnp.where(partner_hint & known_wrong, 1.0, 0.0)
        hint_viols.append(partner_hint & ~in_s)
    log_hint = jnp.where(hbad >= 0.5, LOG_HINT_WRONG_BAD, LOG_HINT_WRONG)
    for hv in hint_viols:
        logw = logw + jnp.where(hv, log_hint, 0.0)
    logw = logw - jnp.max(logw)

    # ---------------- posterior quantities ----------------
    w = jax.nn.softmax(logw)                                            # [1680]
    marg = jnp.einsum('x,xjc->jc', w, ONEHOT)                           # [9,3]
    same_prob = jnp.einsum('x,xij->ij', w, SAME_F)                      # [9,9]
    edges_c = 0.5 * jnp.einsum('xjc,xkc,jk->xc', ONEHOT, ONEHOT, A)     # [1680,3]
    complete_c = edges_c >= 1.999                                       # [1680,3]
    allconn = jnp.all(complete_c, axis=1)                               # [1680]
    p_conn = jnp.sum(w * allconn)
    complete_card = jnp.einsum('xjc,xc->xj', ONEHOT, complete_c.astype(jnp.float32))  # [1680,9]
    e_safe = jnp.einsum('x,xj->j', w, complete_card)                    # [9]
    max_marg = marg.max(1)
    known_soft = (max_marg > 0.99).astype(jnp.float32)                  # [9]
    n_known = (max_marg > 0.99).sum()

    # ---------------- hint value / score if ended now ----------------
    correct_p = jnp.einsum('hc,jc->hj', hset_f, marg)                   # [4,9]
    n_unrev = jnp.sum(~hrevealed)
    n_rev_unplaced = jnp.sum(hrevealed & ~hplaced)
    placed_val = jnp.sum(jnp.where(hplaced, 2.0 * correct_p[jnp.arange(NHINT), htarget] - 1.0, 0.0))
    s_now = 5.0 * n_unrev + 2.0 * n_rev_unplaced + placed_val

    # ---------------- substep 0: end decision ----------------
    # Expected value of continuing: after this turn the hint value drops by 1 (a correct
    # placement is available) or 3 (must reveal); the board will then be connected with
    # probability ~CONT_P (we learn two more colours and fix one thing, partner may still err).
    n_done = (4 - n_unrev) + jnp.sum(hplaced)                           # hint actions taken so far
    can_place_next = n_rev_unplaced > 0
    s_next = s_now - jnp.where(can_place_next, 1.0, 3.0)
    last_turn = n_done >= 7
    # a partner that has broken a known same-colour adjacency makes continuing worth much less
    destructive = n_destr >= 1
    cont_p = jnp.where(destructive, CONT_P_DESTR, CONT_P)
    floor = jnp.where(destructive, 0.35, 0.45)
    # on the last turn nobody else acts: value of continuing is P(one final fix suffices) * (s_now - 1)
    v_cont = jnp.where(last_turn, jnp.minimum(cont_p, CONT_P_LAST) * s_next, cont_p * (0.96 * s_next - 0.36))
    tau = jnp.clip(v_cont / jnp.maximum(s_now, 1.0), floor, 0.9)
    do_end = (p_conn >= tau) & legal[1067]

    # ---------------- observe choice ----------------
    ent = -jnp.sum(marg * jnp.log(marg + 1e-9), axis=1)                # [9]
    # mutual information between card i's colour and "all colours connected" on the current board
    pc_ic = jnp.einsum('x,xic,x->ic', w, ONEHOT, allconn.astype(jnp.float32))   # P(chi_i=c, conn)

    def _h(p):
        p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
        return -(p * jnp.log(p) + (1.0 - p) * jnp.log(1.0 - p))

    cond = pc_ic / jnp.maximum(marg, 1e-6)                              # P(conn | chi_i=c)
    mi = _h(p_conn) - jnp.sum(marg * _h(cond), axis=1)                  # [9]
    rng, r1, r2, r3 = jax.random.split(rng, 4)
    obs_score = ent + MI_WEIGHT * jnp.maximum(mi, 0.0) \
        + 0.1 * (1.0 - partner_obs_ever.astype(jnp.float32)) \
        + 0.001 * jax.random.uniform(r1, (NCARD,))
    obs_legal = legal[1:1 + NCARD]
    obs_score = jnp.where(obs_legal, obs_score, NEG)
    obs_action = 1 + jnp.argmax(obs_score)

    # ---------------- move choice ----------------
    adjP = CELL_ADJ[:, cell]                                            # [81,9] cell p adjacent to card j
    gain = jnp.einsum('ij,pj->ip', same_prob, adjP) - jnp.sum(same_prob * A, axis=1)[:, None]   # [9,81]
    diff_prob = (1.0 - same_prob) * (1.0 - jnp.eye(NCARD))
    diff_adj = jnp.einsum('ij,pj->ip', diff_prob, adjP)                 # [9,81] adjacency to other colours
    occ_nb = jnp.einsum('pj->p', adjP)[None, :] - adjP.T                # [9,81] neighbours excluding card i
    cx = jnp.mean(pos[:, 0].astype(jnp.float32))
    cy = jnp.mean(pos[:, 1].astype(jnp.float32))
    dist = jnp.abs(CELL_X - cx) + jnp.abs(CELL_Y - cy)                  # [81]
    tiebreak = (- 0.10 * diff_adj
                + 0.08 * jnp.minimum(occ_nb, 2.0)                        # keep the board compact / 2-connected
                - 0.04 * dist[None, :]
                - 0.35 * (1.0 - known_soft)[:, None]                     # partner may know this card is well placed
                - 0.30 * partner_moved.astype(jnp.float32)[:, None]
                + 0.001 * jax.random.uniform(r2, (NCARD, NCELL)))
    move_legal = legal[10:10 + NCARD * NCELL].reshape(NCARD, NCELL)
    score1 = jnp.where(move_legal, gain + tiebreak, NEG).reshape(-1)     # [729]
    top_val, top_idx = jax.lax.top_k(score1, LOOKAHEAD_K)
    eye9f = jnp.eye(NCARD, dtype=jnp.float32)
    locked_f = locked.astype(jnp.float32)

    def follow_up_value(flat):
        """Best expected same-color-edge gain of one more move after first move `flat`."""
        i = flat // NCELL
        p = flat % NCELL
        newpos = pos.at[i].set(jnp.stack([p // GRID, p % GRID]))
        newcell = newpos[:, 0] * GRID + newpos[:, 1]
        A1 = _card_adjacency(newpos)                                    # [9,9]
        # board-minus-j adjacency for each j (row/col j zeroed): [9 j, 9, 9]
        keep = (1.0 - eye9f)                                             # keep[j, i] = i != j
        Aj = A1[None, :, :] * keep[:, :, None] * keep[:, None, :]
        R = jnp.minimum(Aj + eye9f[None], 1.0)
        for _ in range(3):
            R = jnp.minimum(jnp.einsum('jik,jkl->jil', R, R), 1.0)
        label = jnp.argmax(R > 0.5, axis=2)                              # [9 j, 9 i] min index reachable
        lab1h = jax.nn.one_hot(label, NCARD, dtype=jnp.float32) * keep[:, :, None]   # [9 j, 9 i, 9 c]
        comp_present = lab1h.sum(1) > 0.5                                # [9 j, 9 c]
        adjQ = CELL_ADJ[:, newcell]                                      # [81 q, 9 i]
        touch = jnp.einsum('qi,jic->jqc', adjQ, lab1h) > 0.5             # [9 j, 81 q, 9 c]
        connected2 = jnp.all(touch | ~comp_present[:, None, :], axis=2)  # [9 j, 81 q]
        occupied = jnp.zeros((NCELL,), jnp.float32).at[newcell].add(1.0) > 0.5
        legal2 = connected2 & (~occupied)[None, :] & (locked_f < 0.5)[:, None]
        gain2 = jnp.einsum('jk,qk->jq', same_prob, adjQ) - jnp.sum(same_prob * A1, axis=1)[:, None]
        best2 = jnp.max(jnp.where(legal2, gain2, NEG))
        return jnp.maximum(best2, 0.0)

    v2 = jax.lax.map(follow_up_value, top_idx)                           # [K]
    total = jnp.where(top_val > NEG / 2, top_val + LOOKAHEAD_GAMMA * v2, NEG)
    move_flat = top_idx[jnp.argmax(total)]
    move_action = 10 + move_flat

    # ---------------- hint choice ----------------
    place_legal = legal[743:743 + NHINT * NCELL].reshape(NHINT, NCELL)  # [4,81] by cell
    # per (hint, card): legal iff place at card's cell is legal
    place_legal_card = jnp.take_along_axis(place_legal, jnp.broadcast_to(cell[None, :], (NHINT, NCARD)), axis=1)  # [4,9]
    hsize = hset_f.sum(1)                                               # [4]
    # must_move[j]: probability that j's two colour-mates are adjacent to each other but not to j
    pair_adj = 0.5 * jnp.einsum('xjk,xjl,kl->xj', SAME_F, SAME_F, A)    # [1680,9]
    nbr_x = jnp.einsum('xjk,jk->xj', SAME_F, A)                         # [1680,9]
    must_move = jnp.einsum('x,xj->j', w, ((pair_adj >= 0.5) & (nbr_x < 0.5)).astype(jnp.float32))
    e_nbr = jnp.sum(same_prob * A, axis=1)                              # [9]
    partner_seen_f = partner_obs_ever.astype(jnp.float32)
    certain = place_legal_card & (correct_p >= 0.995)
    tier_a = certain & (e_safe[None, :] >= 0.95)                        # complete group: lock is free
    # lock only where the partner loses no information: single-colour hint (tells them the colour),
    # a card they already observed, or late in the game (turn >= 6) where the hint helps them deduce
    tier_b = certain & (must_move[None, :] < 0.4) & (
        (hsize[:, None] < 1.5) | (partner_seen_f[None, :] > 0.5) | (n_done >= 5))
    safe_ok = tier_a | tier_b
    safe_score = jnp.where(tier_a, 20.0 - hsize[:, None] + e_safe[None, :],
                 jnp.where(tier_b, 10.0 - hsize[:, None] + e_nbr[None, :] - must_move[None, :], NEG))
    any_safe = jnp.any(safe_ok)
    reveal_legal = legal[739:743]
    any_reveal = jnp.any(reveal_legal)
    reveal_h = jnp.argmax(jnp.where(reveal_legal, jnp.arange(NHINT, 0, -1).astype(jnp.float32), NEG))
    forced_score = (correct_p + 0.5 * e_safe[None, :] + 0.3 * e_nbr[None, :] - 0.5 * must_move[None, :]
                    - 0.1 * partner_moved.astype(jnp.float32)[None, :]
                    + 0.001 * jax.random.uniform(r3, (NHINT, NCARD)))
    forced_score = jnp.where(place_legal_card, forced_score, NEG)
    sel_flat = jnp.where(any_safe, jnp.argmax(safe_score.reshape(-1)), jnp.argmax(forced_score.reshape(-1)))
    sel_h = sel_flat // NCARD
    sel_j = sel_flat % NCARD
    place_action = 743 + sel_h * NCELL + cell[sel_j]
    hint_action = jnp.where(any_safe, place_action,
                            jnp.where(any_reveal, 739 + reveal_h, place_action))

    # ---------------- assemble ----------------
    action = jnp.where(substep == 0, jnp.where(do_end, 1067, obs_action),
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, 0)
    # safety fallback: never emit an illegal action
    action = jnp.where(legal[action], action, jnp.argmax(legal))
    action = action.astype(jnp.int32)

    placed_now = active & (action >= 743) & (action < 1067)
    placed_h = jnp.clip((action - 743) // NCELL, 0, NHINT - 1)
    my_placed = jnp.where(placed_now, mem['my_placed'].at[placed_h].set(1), mem['my_placed'])

    new_mem = {
        'known': known.astype(jnp.int32),
        'prev_pos': pos.astype(jnp.int32),
        'prev_placed': hplaced.astype(jnp.int32),
        'my_placed': my_placed.astype(jnp.int32),
        'partner_obs_ever': partner_obs_ever.astype(jnp.int32),
        'partner_moved': partner_moved.astype(jnp.int32),
        'join_ev': join_ev.astype(jnp.int32),
        'nj_ev': nj_ev.astype(jnp.int32),
        'n_join': n_join.astype(jnp.int32),
        'n_nj': n_nj.astype(jnp.int32),
        'n_destr': n_destr.astype(jnp.int32),
    }
    trace = {
        'p_conn': p_conn.astype(jnp.float32),
        's_now': s_now.astype(jnp.float32),
        'n_known': n_known.astype(jnp.int32),
        'substep': substep.astype(jnp.int32),
        'active': active,
        'known': jnp.where(max_marg > 0.99, jnp.argmax(marg, axis=1), -1).astype(jnp.int32),
        'trust_good': good.astype(jnp.float32),
        'trust_bad': (bad + hbad).astype(jnp.float32),
        'move_gain': jnp.max(gain.reshape(-1) * move_legal.reshape(-1).astype(jnp.float32)
                             + jnp.where(move_legal.reshape(-1), 0.0, NEG)).astype(jnp.float32),
    }
    return action, new_mem, trace
