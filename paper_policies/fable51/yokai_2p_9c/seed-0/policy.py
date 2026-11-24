"""Posterior-driven cooperative Yokai policy (convention-free).

Core idea: maintain an exact posterior over all 1680 colour arrangements using
this seat's own observed colours (perfect memory exposes them in the
observation) plus soft evidence from publicly placed hints.  Every decision is
derived from that posterior:

* observe: the unknown card whose colour carries the most information about
  whether all colours are connected (ties: cards the partner has not seen);
* move: 1-ply search over cells adjacent to the current cards, maximising the
  posterior probability that all colours are connected (plus expected number of
  connected colours / same-colour edges as tie-breakers);
* hint: place a revealed hint on a card whose colour we know and which matches
  the hint (never incorrect), preferring settled cards; otherwise reveal;
* end: end the game when P(all connected) * current score beats a rough
  continuation estimate.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'posterior_yokai', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
HINT_TRUST_EPS = 0.2      # likelihood of a placed hint being inconsistent
MOVE_TRUST_EPS = 0.3      # likelihood that a partner's move joined a different colour
W_UNDO = 0.15             # penalty for moving a card the partner moved / knows better
END_KAPPA = 0.9           # continuation optimism (lower => end more eagerly)
EARLY_TAU_CAP = 0.45      # end-threshold cap while at most 4 turns are done
W_NEED = 0.25             # weight of expected moves still needed in move score
W_EDGES = 0.20            # weight of expected same-colour edges in move score
W_COMPACT = 0.02          # weight of destination compactness
OBS_W_UNSEEN = 0.2        # preference for cards the partner has not seen
PLACE_W_SEEN = 0.4        # two-colour hints go on partner-seen cards (locking hides colour);
                          # the single-colour hint goes on partner-unseen cards (exact info)
OBS_W_MI = 1.0
OBS_W_VOI = 1.0           # weight of value-of-information (moves saved) in observation choice
NUM_ARR = 1680            # 9! / (3! 3! 3!)
NEG = -1e9


def _tables():
    """Constant tables built with fixed-shape jnp ops (constant under jit)."""
    codes = jnp.arange(3 ** 9, dtype=jnp.int32)
    powers = 3 ** jnp.arange(9, dtype=jnp.int32)
    digits = (codes[:, None] // powers[None, :]) % 3                 # [19683, 9]
    counts = (digits[:, :, None] == jnp.arange(3)[None, None, :]).sum(1)  # [19683, 3]
    valid = (counts == 3).all(-1)
    order = jnp.argsort(jnp.where(valid, 0, 1), stable=True)
    arr = digits[order[:NUM_ARR]]                                     # [1680, 9]
    arr1h = (arr[:, :, None] == jnp.arange(3)[None, None, :]).astype(jnp.float32)
    ii, jj = jnp.triu_indices(9, k=1)                                # 36 pairs
    pi = ii.astype(jnp.int32)
    pj = jj.astype(jnp.int32)
    same_pair = arr1h[:, pi, :] * arr1h[:, pj, :]                    # [1680, 36, 3]
    same_pair = jnp.transpose(same_pair, (0, 2, 1))                  # [1680, 3, 36]
    same = (arr[:, :, None] == arr[:, None, :]) & (~jnp.eye(9, dtype=bool))[None]  # [1680, 9, 9]
    pairq = (same[:, :, pi] & same[:, :, pj]).astype(jnp.float32)    # [1680, 9, 36]
    return arr, arr1h, same_pair, pairq, pi, pj


def _joinable(pa, pb, occ_flat):
    """True if a third card could be placed adjacent to both pa and pb in one move.

    pa, pb: [..., 2] int positions; occ_flat: [..., 81] bool occupancy grid
    (broadcastable to the leading dims).  Distance 1: already adjacent.
    Distance 2 straight: middle cell must be empty.  Distance 2 diagonal: one
    of the two corner cells must be empty.
    """
    dx = pa[..., 0] - pb[..., 0]
    dy = pa[..., 1] - pb[..., 1]
    D = jnp.abs(dx) + jnp.abs(dy)
    mid = jnp.clip((pa + pb) // 2, 0, 8)
    mid_i = mid[..., 0] * 9 + mid[..., 1]
    c1 = jnp.clip(pa[..., 0], 0, 8) * 9 + jnp.clip(pb[..., 1], 0, 8)
    c2 = jnp.clip(pb[..., 0], 0, 8) * 9 + jnp.clip(pa[..., 1], 0, 8)
    occ_mid = jnp.take_along_axis(occ_flat, mid_i[..., None], axis=-1)[..., 0]
    occ_c1 = jnp.take_along_axis(occ_flat, c1[..., None], axis=-1)[..., 0]
    occ_c2 = jnp.take_along_axis(occ_flat, c2[..., None], axis=-1)[..., 0]
    straight = (D == 2) & ((dx == 0) | (dy == 0))
    diag = (D == 2) & (dx != 0) & (dy != 0)
    return (D == 1) | (straight & ~occ_mid) | (diag & (~occ_c1 | ~occ_c2))


def _hb(x):
    x = jnp.clip(x, 1e-6, 1 - 1e-6)
    return -(x * jnp.log(x) + (1 - x) * jnp.log(1 - x))


def init_agent():
    return {
        'pos': jnp.full((9, 2), -1, dtype=jnp.int32),   # positions at previous call
        'ev': jnp.zeros((9, 9), dtype=jnp.int32),       # partner-move evidence rows
        'pmoved': jnp.zeros((9,), dtype=jnp.int32),     # cards last moved by partner
    }


def agent_act(obs, reset, memory, rng):
    ARR, ARR1H, SAME_PAIR, PAIRQ, PI, PJ = _tables()
    DIRS = jnp.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
    CARD_IDS = jnp.arange(9, dtype=jnp.int32)
    nf = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    cards = nf[:9]
    hints = nf[9:13]
    active = nf[0, 10] > 0
    substep = nf[0, 11]

    onehot = cards[:, 0:3]
    known = onehot.sum(-1) > 0                       # [9]
    col = jnp.argmax(onehot, -1)                     # [9]
    pos = obs.positions.astype(jnp.int32)            # [9, 2]
    locked = cards[:, 8] > 0
    pseen = cards[:, 9] > 0
    eye9 = jnp.eye(9, dtype=jnp.float32)
    A = obs.adjacency_matrix[:9, :9].astype(jnp.float32) * (1.0 - eye9)
    A_pairs = A[PI, PJ]                               # [36]

    hset = hints[:, 3:6] > 0                          # [4, 3]
    hrev = hset.any(-1)
    hplaced = hints[:, 8] > 0
    hpos = hints[:, 6:8]
    same_pos = (hpos[:, None, :] == pos[None, :, :]).all(-1)   # [4, 9]
    hcard = jnp.argmax(same_pos, -1)                  # [4]
    hvalid = hplaced & same_pos.any(-1)

    # ---------------- memory: detect partner moves ----------------
    mem_pos = jnp.where(reset, pos, memory['pos'])
    mem_ev = jnp.where(reset, 0, memory['ev'])
    mem_pmoved = jnp.where(reset, 0, memory['pmoved'])
    changed = (pos != mem_pos).any(-1)                # [9]
    partner_moved = changed & (~active)
    my_moved = changed & active
    nb_now = (A > 0) & pseen[None, :]                 # [9 X, 9 Y] seen neighbours
    new_ev = jnp.where(partner_moved[:, None] & pseen[:, None], nb_now, mem_ev > 0)
    mem_ev = new_ev.astype(jnp.int32)
    mem_pmoved = jnp.where(partner_moved, 1, jnp.where(my_moved, 0, mem_pmoved)).astype(jnp.int32)
    next_memory = {'pos': pos.astype(jnp.int32), 'ev': mem_ev, 'pmoved': mem_pmoved}

    # ---------------- posterior over arrangements ----------------
    match = (ARR == col[None, :]) | (~known)[None, :]
    w = match.all(-1).astype(jnp.float32)             # [1680]
    hc1h = ARR1H[:, hcard, :]                         # [1680, 4, 3]
    cons = (hc1h * hset[None].astype(jnp.float32)).sum(-1) > 0   # [1680, 4]
    lik = jnp.where(hvalid[None, :], jnp.where(cons, 1.0, HINT_TRUST_EPS), 1.0)
    w = w * lik.prod(-1)
    # partner-move evidence: moved card shares colour with a seen neighbour
    ev_f = mem_ev.astype(jnp.float32)                 # [9, 9]
    samecol = jnp.einsum('axc,ayc->axy', ARR1H, ARR1H)   # [1680, 9, 9]
    hit = jnp.einsum('axy,xy->ax', samecol, ev_f) > 0    # [1680, 9]
    row_has = ev_f.sum(-1) > 0                        # [9]
    mlik = jnp.where(row_has[None, :], jnp.where(hit, 1.0, MOVE_TRUST_EPS), 1.0)
    w = w * mlik.prod(-1)
    Z = jnp.maximum(w.sum(), 1e-9)

    edges_base = jnp.einsum('p,akp->ak', A_pairs, SAME_PAIR)   # [1680, 3]
    conn_base = (edges_base >= 2).astype(jnp.float32)
    nconn = conn_base.sum(-1)                         # [1680]
    allconn = (nconn >= 3).astype(jnp.float32)
    total_base = edges_base.sum(-1)
    p_all = (w * allconn).sum() / Z

    # per-colour connection probability
    p_conn_k = (w[:, None] * conn_base).sum(0) / Z    # [3]

    # ---------------- move choice (1-ply) ----------------
    cand = pos[:, None, :] + DIRS[None, :, :]         # [9, 4, 2]
    cand = cand.reshape(36, 2)
    in_grid = ((cand >= 0) & (cand <= 8)).all(-1)     # [36]
    cellnb = (jnp.abs(pos[:, None, :] - cand[None, :, :]).sum(-1) == 1).astype(jnp.float32)  # [9, 36]
    gp = cellnb[PI] + cellnb[PJ]                      # [36 pairs, 36 cells]
    gp1 = (gp >= 1).astype(jnp.float32)
    gp2 = (gp >= 2).astype(jnp.float32)

    base_c = jnp.einsum('aik,ak->ai', ARR1H, edges_base)     # [1680, 9]
    conn_c = jnp.einsum('aik,ak->ai', ARR1H, conn_base)      # [1680, 9]
    adj_pair_i = A[:, PI] + A[:, PJ]                          # [9, 36]
    lost = jnp.einsum('aiq,iq->ai', PAIRQ, adj_pair_i)        # [1680, 9]
    t = 2.0 - (base_c - lost)                                 # need gained >= t
    ind0 = (t <= 0).astype(jnp.float32)
    ind1 = (t == 1).astype(jnp.float32)
    ind2 = (t == 2).astype(jnp.float32)
    others_cnt = nconn[:, None] - conn_c                      # [1680, 9]
    others_ok = (others_cnt >= 2).astype(jnp.float32)

    def _agg(v):
        c0 = (v * ind0).sum(0)                                # [9]
        c1 = jnp.einsum('ai,aiq->iq', v * ind1, PAIRQ)        # [9, 36]
        c2 = jnp.einsum('ai,aiq->iq', v * ind2, PAIRQ)
        return c0[:, None] + c1 @ gp1 + c2 @ gp2              # [9 cards, 36 cells]

    v_all = w[:, None] * others_ok
    p_move = _agg(v_all) / Z                                  # P(all connected after move)
    w9 = jnp.broadcast_to(w[:, None], (w.shape[0], 9))
    wq = jnp.einsum('ai,aiq->iq', w9, PAIRQ)                  # [9, 36]
    e_edges = ((w9 * (total_base[:, None] - lost)).sum(0)[:, None] + wq @ gp) / Z
    ncards_adj = cellnb.sum(0)[None, :] - cellnb              # [9, 36]

    # expected number of moves still needed (per colour: 0 connected, 1 if a
    # single relocation connects, 2 otherwise, 4 if locks make it infeasible)
    D = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1).astype(jnp.float32)   # [9, 9]
    dpair = D[PI, PJ]                                         # [36]
    unl = (~locked).astype(jnp.float32)
    pos_flat = pos[:, 0] * 9 + pos[:, 1]                      # [9]
    occ = (jnp.arange(81)[None, :] == pos_flat[:, None])      # [9, 81] per-card occupancy
    occ_all = occ.any(0)                                      # [81]
    occ_wo = occ_all[None, :] & (~occ)                        # [9, 81] occupancy without card i
    join_pre = _joinable(pos[PI], pos[PJ], occ_all[None, :])  # [36]
    pj_others = jnp.einsum('ajq,q->aj', PAIRQ, join_pre.astype(jnp.float32))   # [1680, 9]
    flag = ((pj_others > 0) & (~locked)[None, :]).astype(jnp.float32)
    onemove_k = jnp.einsum('ajk,aj->ak', ARR1H, flag) > 0     # [1680, 3]
    nunl_k = jnp.einsum('ajk,j->ak', ARR1H, unl)              # [1680, 3]
    needed_k = jnp.where(conn_base > 0, 0.0,
                         jnp.where(onemove_k, 1.0, jnp.where(nunl_k >= 2, 2.0, 4.0)))
    total_needed = needed_k.sum(-1)                           # [1680]
    needed_ci = jnp.einsum('aik,ak->ai', ARR1H, needed_k)     # [1680, 9]
    others_needed = total_needed[:, None] - needed_ci         # [1680, 9]
    dc = jnp.abs(pos[:, None, :] - cand[None, :, :]).sum(-1).astype(jnp.float32)  # [9, 36]
    d1 = dc[PI][None]                                         # [1, 36 q, 36 cells]
    d2 = dc[PJ][None]
    d3 = dpair[None, :, None]
    lockI = locked[PI][None, :, None]
    lockJ = locked[PJ][None, :, None]
    conn_after = ((d1 == 1) & (d2 == 1)) | ((d1 == 1) & (d3 == 1)) | ((d2 == 1) & (d3 == 1))
    # joinability after mover i sits at `cell` (its old cell becomes empty)
    occ_i = occ_wo[:, None, None, :]                          # [9, 1, 1, 81]
    join_pair_i = _joinable(pos[PI][None], pos[PJ][None], occ_wo[:, None, :])[:, :, None]  # [9, 36, 1]
    join_b = _joinable(cand[None, None, :, :], pos[PI][None, :, None, :], occ_i)          # [9, 36, 36]
    join_c = _joinable(cand[None, None, :, :], pos[PJ][None, :, None, :], occ_i)
    one_after = join_pair_i | (join_b & (~lockJ)) | (join_c & (~lockI))
    two_after = ~(lockI & lockJ)
    needed_after = jnp.where(conn_after, 0.0, jnp.where(one_after, 1.0, jnp.where(two_after, 2.0, 4.0)))  # [9, 36, 36]
    e_needed = ((w9 * others_needed).sum(0)[:, None] + jnp.einsum('iq,iqc->ic', wq, needed_after)) / Z   # [9, 36]
    e_needed_now = (w * total_needed).sum() / Z

    mx = jnp.clip(cand[:, 0], 0, 8)
    my = jnp.clip(cand[:, 1], 0, 8)
    move_ids = 10 + CARD_IDS[:, None] * 81 + (mx * 9 + my)[None, :]   # [9, 36]
    move_legal = legal[move_ids] & in_grid[None, :]

    # ---------------- observation choice ----------------
    p_ic = jnp.einsum('a,aic->ic', w, ARR1H) / Z
    q_ic = jnp.einsum('a,aic->ic', w * allconn, ARR1H) / Z
    cond = q_ic / jnp.maximum(p_ic, 1e-9)
    mi = _hb(p_all) - (p_ic * _hb(cond)).sum(-1)      # [9]
    ent = -(p_ic * jnp.log(jnp.clip(p_ic, 1e-9, 1.0))).sum(-1)
    # value of information for the next move: expected best E[moves needed]
    # after learning card j's colour (posterior is linear in the weights)
    W_hyp = (w[:, None, None] * ARR1H).reshape(w.shape[0], 27)          # [1680, 27]
    Z_h = W_hyp.sum(0)                                                   # [27]
    others_term = W_hyp.T @ others_needed                                # [27, 9]
    wq_h = (W_hyp.T @ PAIRQ.reshape(w.shape[0], 9 * 36)).reshape(27, 9, 36)
    e_needed_h = (others_term[:, :, None] + jnp.einsum('hiq,iqc->hic', wq_h, needed_after)) \
        / jnp.maximum(Z_h, 1e-9)[:, None, None]                          # [27, 9, 36]
    big = 1e6
    best_h = jnp.min(jnp.where(move_legal[None], e_needed_h, big).reshape(27, -1), axis=-1)
    best_h = jnp.where(Z_h > 0, best_h, 0.0)
    p_h = Z_h / Z
    exp_after = (p_h * best_h).reshape(9, 3).sum(-1)                     # [9]
    best_now = jnp.min(jnp.where(move_legal, e_needed, big))
    voi = jnp.maximum(best_now - exp_after, 0.0)                         # [9]
    obs_legal = legal[1:10]
    obs_score = (jnp.where(obs_legal, 0.0, NEG) + 2.0 * (~known)
                 + OBS_W_MI * mi + OBS_W_VOI * voi + 0.1 * ent + OBS_W_UNSEEN * (~pseen)
                 - 0.001 * CARD_IDS)
    obs_action = 1 + jnp.argmax(obs_score).astype(jnp.int32)

    undo_pen =W_UNDO * ((mem_pmoved > 0) | (pseen & ~known)).astype(jnp.float32)
    move_score = (jnp.where(move_legal, 0.0, NEG) + p_move
                  - W_NEED * e_needed + W_EDGES * e_edges / 6.0
                  + W_COMPACT * ncards_adj / 4.0 - undo_pen[:, None])
    best_flat = jnp.argmax(move_score.reshape(-1))
    move_action = move_ids.reshape(-1)[best_flat].astype(jnp.int32)
    move_ok = move_score.reshape(-1)[best_flat] > NEG / 2
    fallback = jnp.argmax(legal).astype(jnp.int32)
    move_action = jnp.where(move_ok, move_action, fallback)

    # ---------------- hint choice ----------------
    # placement candidates [4 hints, 9 cards]
    hset_f = hset.astype(jnp.float32)
    card_match = hset_f[:, col] > 0                           # [4, 9] card colour in set
    conn_of_card = p_conn_k[col]                              # [9]
    settled = conn_of_card > 0.999
    # a locked card Y could share X's colour if Y's colour is known equal, or
    # Y's colour is unknown and the hint set placed on Y contains X's colour.
    hint_on_card = jnp.argmax((same_pos & hvalid[:, None]).astype(jnp.int32), axis=0)   # [9]
    set_on_card = hset_f[hint_on_card] * locked[:, None].astype(jnp.float32)          # [9, 3]
    could_match = jnp.where(known[None, :], col[:, None] == col[None, :],
                            set_on_card[:, col].T > 0)                                # [X, Y]
    risky_lock = (locked[None, :] & could_match & (~jnp.eye(9, dtype=bool))
                  & (A == 0)).any(-1)                                                  # [9]
    locked_nb = (A * locked[None, :].astype(jnp.float32)).sum(-1)
    safe = known & (~locked) & (settled | ((~risky_lock) & (locked_nb < 2)))
    place_ids = 743 + jnp.arange(4, dtype=jnp.int32)[:, None] * 81 + (pos[:, 0] * 9 + pos[:, 1])[None, :]
    place_legal = legal[place_ids] & hrev[:, None] & (~hplaced)[:, None]
    place_ok = place_legal & card_match & safe[None, :]
    single = hset_f.sum(-1) == 1
    place_score = (jnp.where(place_ok, 1.0, NEG) + 0.5 * settled[None, :]
                   + PLACE_W_SEEN * jnp.where(single[:, None], (~pseen)[None, :], pseen[None, :])
                   + 0.3 * single[:, None]
                   - 0.001 * CARD_IDS[None, :] - 0.01 * jnp.arange(4)[:, None])
    pflat = jnp.argmax(place_score.reshape(-1))
    can_place = place_score.reshape(-1)[pflat] > NEG / 2
    place_action = place_ids.reshape(-1)[pflat].astype(jnp.int32)
    reveal_pref = jnp.asarray([0.0, 3.0, 2.0, 1.0], jnp.float32)  # 2-colour hints first
    reveal_score = jnp.where(legal[739:743], reveal_pref, NEG)
    reveal_action = 739 + jnp.argmax(reveal_score).astype(jnp.int32)
    # last resort: any matching-colour known card even if not "safe"
    loose_ok = place_legal & card_match & known[None, :]
    lscore = jnp.where(loose_ok, 1.0, NEG) - 0.001 * CARD_IDS[None, :]
    lflat = jnp.argmax(lscore.reshape(-1))
    can_loose = lscore.reshape(-1)[lflat] > NEG / 2
    loose_action = place_ids.reshape(-1)[lflat].astype(jnp.int32)
    any_reveal = legal[739:743].any()
    hint_action = jnp.where(can_place, place_action,
                            jnp.where(any_reveal, reveal_action,
                                      jnp.where(can_loose, loose_action, fallback)))

    # ---------------- end decision ----------------
    U = (~hrev).sum()
    R = (hrev & ~hplaced).sum()
    e_cons = (w[:, None] * cons).sum(0) / Z                   # [4]
    placed_val = jnp.where(hplaced, 2.0 * e_cons - 1.0, 0.0).sum()
    s_now = 5.0 * U + 2.0 * R + placed_val
    turns_done = hrev.sum() + hplaced.sum()
    my_cost = jnp.where(loose_ok.any(), 1.0, jnp.where(U > 0, 3.0, 2.0))
    partner_cost = jnp.where(U > 0, 3.0, 1.5)
    delta = jnp.where(turns_done >= 7, my_cost, my_cost + partner_cost)
    tau = jnp.clip(END_KAPPA * (s_now - delta) / jnp.maximum(s_now, 1e-3), 0.0, 1.0)
    # early in the game the realistic continuation value (~7) is far below
    # s_now - delta, and the uniform posterior is pessimistic, so cap tau.
    tau = jnp.where(turns_done <= 4, jnp.minimum(tau, EARLY_TAU_CAP), tau)
    want_end = (p_all >= tau) & legal[1067]
    end_or_obs = jnp.where(want_end, jnp.int32(1067), obs_action)

    action = jnp.where(substep == 0, end_or_obs,
                       jnp.where(substep == 1, obs_action,
                                 jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, jnp.int32(0)).astype(jnp.int32)
    # final legality guard
    action = jnp.where(legal[action], action, fallback)

    trace = {
        'p_all': p_all.astype(jnp.float32),
        's_now': s_now.astype(jnp.float32),
        'tau': tau.astype(jnp.float32),
        'n_known': known.sum().astype(jnp.int32),
        'p_move': jnp.max(jnp.where(move_legal, p_move, 0.0)).astype(jnp.float32),
        'e_need': e_needed_now.astype(jnp.float32),
        'col': jnp.where(known, col, -1).astype(jnp.int32),
        'pos': pos.astype(jnp.int32),
        'ev': mem_ev.astype(jnp.int32),
    }
    return action, next_memory, trace
