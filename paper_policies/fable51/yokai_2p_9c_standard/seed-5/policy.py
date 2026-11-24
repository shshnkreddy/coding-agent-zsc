"""Belief-enumeration policy for two-player 9-card Yokai.

Each seat privately remembers the colors it has observed.  All 1680 colorings
consistent with that memory are enumerated (exact posterior under the uniform
shuffle), and every decision (observe / move / end / hint) is scored under that
posterior using only the local observation, private memory and PRNG.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_greedy', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NC = 9          # cards
NCOL = 3        # colors
NH = 4          # hints
G = 9           # grid side
NCELL = G * G
NCAND = 36      # max cells adjacent to any card

# action layout
A_OBS = 1
A_MOVE = 10
A_REVEAL = 739
A_PLACE = 743
A_END = 1067

# tunable constants
W_PCONN = 10.0        # weight of success probability in move score
W_PROG = 1.0          # weight of expected same-color adjacent pairs
W_TWO = 3.0           # weight of two-move completion under the MAP colouring
W_COMPACT = 0.05      # weight of board adjacency count (flexibility)
W_CENTER = 0.01       # penalty per Manhattan step from centroid
PARTNER_LOCK_ODDS = 0.25   # weight multiplier for colorings where a partner-placed hint is wrong
PARTNER_COST = 2.0    # assumed hint-value cost of the partner's turn
Q_PARTNER = 0.25      # chance the partner's turn completes an unfinished board
Q_FURTHER = 0.5       # discount for continuation beyond the next decision
SAFE_THRESH = 0.99
KNOWN_THRESH = 0.99
BETA_MOVE = 2.0       # log-likelihood weight per same-color adjacency gained by a partner move
W_PSEEN = 3.0         # observation preference for cards the partner has not observed


NK = 1680


def _tables():
    """Fixed-shape constant tables built with jnp only (no module-scope calls)."""
    idx = jnp.arange(NCOL ** NC, dtype=jnp.int32)
    digits = jnp.stack([(idx // (NCOL ** j)) % NCOL for j in range(NC)], axis=1)   # [19683, 9]
    counts = jnp.stack([(digits == c).sum(1) for c in range(NCOL)], axis=1)
    valid = (counts == 3).all(1)
    sel = jnp.nonzero(valid, size=NK, fill_value=0)[0]
    colorings = digits[sel]                                                  # [1680, 9]
    onehot = jax.nn.one_hot(colorings, NCOL, dtype=jnp.float32)              # [1680, 9, 3]
    cells = jnp.arange(NCELL, dtype=jnp.int32)
    cell_xy = jnp.stack([cells // G, cells % G], axis=1)
    eye9 = jnp.eye(NC, dtype=jnp.float32)
    return colorings, onehot, cell_xy, eye9


def init_agent():
    return {
        "known": -jnp.ones((NC,), dtype=jnp.int32),
        "pseen": jnp.zeros((NC,), dtype=jnp.int32),
        "prev_pos": -jnp.ones((NC, 2), dtype=jnp.int32),
        "logw": jnp.zeros((NK,), dtype=jnp.float32),
        "hcount": jnp.zeros((4,), dtype=jnp.int32),   # prev n_revealed, prev n_placed, partner reveals, partner places
    }


def _manhattan(a, b):
    return jnp.abs(a[:, None, 0] - b[None, :, 0]) + jnp.abs(a[:, None, 1] - b[None, :, 1])


def agent_act(obs, reset, memory, rng):
    f32 = jnp.float32
    COLORINGS, ONEHOT, CELL_XY, EYE9 = _tables()
    init = init_agent()
    known_prev = jnp.where(reset, init["known"], memory["known"])
    pseen_prev = jnp.where(reset, init["pseen"], memory["pseen"])
    logw_prev = jnp.where(reset, init["logw"], memory["logw"])
    hcount_prev = jnp.where(reset, init["hcount"], memory["hcount"])

    nf = obs.node_features.astype(jnp.int32)
    card_nf = nf[:NC]
    own_col = card_nf[:, 0:3]
    visible = own_col.sum(-1) > 0
    known = jnp.where(visible, jnp.argmax(own_col, axis=-1).astype(jnp.int32), known_prev)

    pos = obs.positions.astype(jnp.int32)                      # [9,2]
    prev_pos = jnp.where(reset, pos, memory["prev_pos"])
    prev_pos = jnp.where((prev_pos < 0).any(), pos, prev_pos)
    locked = card_nf[:, 8] > 0                                  # [9]
    active = nf[0, 10] > 0
    pseen = jnp.maximum(pseen_prev, (card_nf[:, 9] > 0).astype(jnp.int32))
    substep = nf[0, 11]
    hint_nf = nf[NC:NC + NH]
    hint_set = hint_nf[:, 3:6] > 0                              # [4,3]
    revealed = hint_set.any(-1)
    placed = hint_nf[:, 8] > 0
    adj = obs.adjacency_matrix.astype(jnp.int32)
    hint_card = adj[NC:NC + NH, :NC] > 0                        # [4,9]
    legal = obs.legal_actions

    # ---------------- partner-move evidence ----------------
    changed = (pos != prev_pos).any(-1)                          # [9]
    partner_moved = (~active) & (changed.sum() == 1)
    mv = jnp.argmax(changed)
    b_cell = pos[mv]
    a_cell = prev_pos[mv]
    near_b = (jnp.abs(pos - b_cell[None]).sum(-1) == 1).astype(f32)        # [9]
    near_a = (jnp.abs(prev_pos - a_cell[None]).sum(-1) == 1).astype(f32)
    seenw = 0.5 + 0.5 * pseen.astype(f32)                                  # partner knows neighbour?
    same_as_mv = (COLORINGS == COLORINGS[:, mv][:, None]).astype(f32)      # [K,9]
    same_as_mv = same_as_mv * (1.0 - jax.nn.one_hot(mv, NC, dtype=f32))[None]
    gained = jnp.minimum(1.0, same_as_mv @ (near_b * seenw))
    lost = jnp.minimum(1.0, same_as_mv @ (near_a * seenw))
    mover_known = 0.4 + 0.6 * pseen[mv].astype(f32)
    logw = logw_prev + jnp.where(partner_moved, BETA_MOVE * mover_known * (gained - 0.5 * lost), 0.0)
    logw = logw - logw.max()

    # ---------------- belief over colorings ----------------
    cons = jnp.all((known[None, :] < 0) | (COLORINGS == known[None, :]), axis=1)   # [K]
    card_idx = jnp.argmax(hint_card, axis=1)                    # [4]
    col_at_hint = COLORINGS[:, card_idx]                        # [K,4]
    correct = hint_set[jnp.arange(NH)[None, :], col_at_hint]    # [K,4]
    applies = placed & (known[card_idx] < 0)
    factor = jnp.where(applies[None, :] & ~correct, PARTNER_LOCK_ODDS, 1.0).prod(-1)
    w = cons.astype(f32) * factor * jnp.exp(logw)
    w = w / jnp.maximum(w.sum(), 1e-9)

    n_unrev = (~revealed).sum()
    n_rev = (revealed & ~placed).sum()
    n_revealed_total = revealed.sum().astype(jnp.int32)
    n_placed_total = placed.sum().astype(jnp.int32)
    p_rev_inc = (~active) & (n_revealed_total > hcount_prev[0])
    p_place_inc = (~active) & (n_placed_total > hcount_prev[1])
    hcount = jnp.stack([n_revealed_total, n_placed_total,
                        hcount_prev[2] + p_rev_inc.astype(jnp.int32),
                        hcount_prev[3] + p_place_inc.astype(jnp.int32)])
    partner_cost = (2.0 * PARTNER_COST + 3.0 * hcount[2] + 1.0 * hcount[3]) / (2.0 + hcount[2] + hcount[3])
    H_k = 5.0 * n_unrev + 2.0 * n_rev + (placed[None, :] * (2.0 * correct - 1.0)).sum(-1)   # [K]
    H_mean = jnp.dot(w, H_k)

    # ---------------- geometry ----------------
    D = _manhattan(pos, pos)                                    # [9,9]
    A = (D == 1).astype(f32)
    CN = (_manhattan(CELL_XY, pos) == 1).astype(f32)            # [81,9]
    occupied = (_manhattan(CELL_XY, pos) == 0).any(-1)          # [81]
    deg = A.sum(1)
    O = ONEHOT                                                  # [K,9,3]
    S = jnp.einsum('jl,klc->kjc', A, O)                         # same-color-c neighbours of card j
    pairs = 0.5 * jnp.einsum('kjc,kjc->kc', O, S)               # [K,3]
    own_nb = jnp.einsum('kic,kic->ki', O, S)                    # [K,9]
    total_pairs = pairs.sum(-1)                                 # [K]
    col_conn = pairs >= 1.5                                     # [K,3]
    conn_now = col_conn.all(-1)                                 # [K]
    P_conn_now = jnp.dot(w, conn_now.astype(f32))
    E_end = jnp.dot(w, conn_now.astype(f32) * H_k)

    # ---------------- candidate destination cells (adjacent to some card) ----------------
    cand = jnp.nonzero(CN.sum(1) > 0, size=NCAND, fill_value=0)[0]           # [36]
    cand_valid = jnp.arange(NCAND) < (CN.sum(1) > 0).sum()
    CNc = CN[cand]                                                            # [36,9]
    occ_c = occupied[cand] | ~cand_valid

    # ---------------- move scoring (single [K,9*36] tensor) ----------------
    NB = jnp.einsum('pj,kjc->kpc', CNc, O)                      # [K,36,3]
    added = jnp.einsum('kic,kpc->kip', O, NB) - CNc.T[None]     # [K,9,36] same-colour neighbours at target
    pairs_own = jnp.einsum('kic,kc->ki', O, pairs)              # [K,9]
    others_ok = ((O > 0) | col_conn[:, None, :]).all(-1)        # [K,9]
    need = (2.0 - (pairs_own - own_nb))[:, :, None]             # 1 or 2 adjacencies still needed
    conn_after = ((added >= need - 0.5) & others_ok[:, :, None]).astype(f32).reshape(NK, NC * NCAND)
    same_ij = jnp.einsum('kic,kjc->kij', O, O).reshape(NK, NC * NC)       # [K,81] P(same colour) support
    prog_base = total_pairs[:, None] - own_nb                             # [K,9]
    not_self = (1.0 - EYE9)

    def expected_terms(V):
        """V: [a,K] normalised beliefs -> P_after [a,9,36], E_prog [a,9,36]."""
        p_after = jnp.matmul(V, conn_after).reshape(-1, NC, NCAND)
        ms = jnp.matmul(V, same_ij).reshape(-1, NC, NC) * not_self[None]
        e_added = jnp.einsum('aij,pj->aip', ms, CNc) - CNc.T[None]
        e_prog = jnp.matmul(V, prog_base)[:, :, None] + e_added
        return p_after, e_prog

    P_after, E_prog = expected_terms(w[None])
    P_after = P_after[0]
    E_prog = E_prog[0]

    # own legality: unlocked card, empty cell, board stays connected
    cn_ip = CNc[None, :, :] * (1.0 - EYE9[:, None, :])          # [9,36,9]
    rowm = EYE9[:, None, :, None]
    colm = EYE9[:, None, None, :]
    Anew = A[None, None] * (1.0 - rowm) * (1.0 - colm) + rowm * cn_ip[:, :, None, :] + colm * cn_ip[:, :, :, None]
    R = Anew + EYE9[None, None]
    for _ in range(4):
        R = (jnp.matmul(R, R) > 0).astype(f32)
    board_conn = (R[:, :, 0, :] > 0).all(-1)                    # [9,36]
    move_ok = (~locked)[:, None] & (~occ_c)[None, :] & board_conn

    # ---------------- two-move completion under the most likely colouring ----------------
    k_map = jnp.argmax(w)
    p_map = w[k_map]
    c_map = COLORINGS[k_map]                                                   # [9]
    oh_map = jax.nn.one_hot(c_map, NCOL, dtype=f32)                            # [9,3]
    same_s = (c_map[:, None] == c_map[None, :]).astype(f32) * (1.0 - EYE9)     # [9,9]
    cand_xy = CELL_XY[cand]                                                    # [36,2]
    is_i = EYE9[:, None, :, None]                                              # [9,1,9,1]
    pos1 = jnp.where(is_i > 0, cand_xy[None, :, None, :], pos[None, None, :, :])   # [9,36,9,2]
    d1 = jnp.abs(pos1[:, :, :, None, :] - pos1[:, :, None, :, :]).sum(-1)      # [9,36,9,9]
    A1 = (d1 == 1).astype(f32)
    dq = jnp.abs(CELL_XY[None, None, :, None, :] - pos1[:, :, None, :, :]).sum(-1)   # [9,36,81,9]
    CN1 = (dq == 1).astype(f32)
    occ1 = (dq == 0).any(-1)                                                   # [9,36,81]
    S1 = jnp.einsum('ipjl,jl->ipj', A1, same_s)                                # [9,36,9]
    pairs1 = 0.5 * jnp.einsum('ipj,jc->ipc', S1, oh_map)                       # [9,36,3]
    pairs_own1 = jnp.einsum('ipc,jc->ipj', pairs1, oh_map)
    need1 = 2.0 - (pairs_own1 - S1)                                            # [9,36,9]
    others_ok1 = ((oh_map[None, None] > 0) | (pairs1[:, :, None, :] >= 1.5)).all(-1)   # [9,36,9]
    added2 = jnp.einsum('ipql,jl->ipjq', CN1, same_s)                          # [9,36,9,81]
    anchor = CN1.sum(-1)[:, :, None, :] - jnp.transpose(CN1, (0, 1, 3, 2))    # other-card adjacency at q
    conn2 = ((added2 >= need1[..., None] - 0.5) & others_ok1[..., None]
             & (~occ1)[:, :, None, :] & (~locked)[None, None, :, None] & (anchor >= 0.5))
    two_move = conn2.any(axis=(2, 3)).astype(f32)                              # [9,36]

    centroid = pos.astype(f32).mean(0)
    cell_dist = jnp.abs(CELL_XY[cand].astype(f32) - centroid[None]).sum(-1)   # [36]
    compact = (CNc.sum(1)[None, :] - CNc.T) - deg[:, None]                     # [9,36]
    move_score = (W_PCONN * P_after + W_PROG * E_prog + W_TWO * p_map * two_move
                  + W_COMPACT * compact - W_CENTER * cell_dist[None, :])
    move_score = jnp.where(move_ok, move_score, -1e9)
    move_legal_env = legal[A_MOVE:A_MOVE + NC * NCELL].reshape(NC, NCELL)[:, cand]
    move_score_env = jnp.where(move_legal_env, move_score, -1e9)
    best_move = jnp.argmax(move_score_env.reshape(-1))
    move_action = (A_MOVE + (best_move // NCAND) * NCELL + cand[best_move % NCAND]).astype(jnp.int32)

    # ---------------- hint placement analysis ----------------
    blocked = (_manhattan(CELL_XY, pos) == 0) & locked[None, :]          # [81,9]
    blocked_cell = blocked.any(-1)                                        # [81]
    cell_of = jnp.argmax(_manhattan(CELL_XY, pos) == 0, axis=0)          # [9] cell index of each card
    blocked_i = blocked_cell[None, :] | (jnp.arange(NCELL)[None, :] == cell_of[:, None])   # [9,81]
    free_i = (~blocked_i).astype(f32)
    hfc = jnp.einsum('pj,pl,ip->ijl', CN, CN, free_i) > 0                 # [9,9,9]
    pair_ok = (D == 1)[None] | ((D == 2)[None] & hfc)
    notok = (1.0 - pair_ok.astype(f32)) * (1.0 - EYE9)[None]              # [9,9,9]
    lockedf = locked.astype(f32)
    Lk = jnp.maximum(lockedf[None, :], EYE9)                              # [9 cand, 9]
    M = Lk[:, None, :, None] * O[None]                                    # [9,K,9,3]
    n_lock = M.sum(2)                                                     # [9,K,3]
    bad2 = jnp.einsum('ikjc,ijl,iklc->ikc', M, notok, M)
    lpairs = 0.5 * jnp.einsum('ikjc,jl,iklc->ikc', M, A, M)
    feas = (n_lock <= 1.5) | ((n_lock > 1.5) & (n_lock < 2.5) & (bad2 < 0.5)) | ((n_lock > 2.5) & (lpairs >= 1.5))
    feas_all = feas.all(-1).astype(f32)                                   # [9,K]
    safe = feas_all @ w                                                   # [9]
    E_same_nb = jnp.dot(w, own_nb)                                        # [9]
    P_col_conn = jnp.einsum('k,kic,kc->i', w, O, col_conn.astype(f32))    # [9]
    P_c = jnp.einsum('k,kic,hc->hi', w, O, hint_set.astype(f32))          # [4,9]
    place_idx = A_PLACE + jnp.arange(NH)[:, None] * NCELL + cell_of[None, :]
    place_legal = legal[place_idx] & (revealed & ~placed)[:, None] & (~locked)[None, :]
    settled = 1.0 + 0.5 * E_same_nb + 0.5 * P_col_conn
    place_pref = safe[None, :] * P_c * settled[None, :]
    place_pref = jnp.where(place_legal, place_pref, -1.0)
    best_place = jnp.argmax(place_pref.reshape(-1))
    bp_h = best_place // NC
    bp_i = best_place % NC
    bp_safe = (safe[bp_i] >= SAFE_THRESH) & (P_c[bp_h, bp_i] >= KNOWN_THRESH) & place_legal[bp_h, bp_i]
    place_action = place_idx[bp_h, bp_i].astype(jnp.int32)
    # forced placement when nothing safe and nothing to reveal: maximise safe * P_c
    forced_pref = jnp.where(place_legal, safe[None, :] * P_c + 1e-3 * P_c, -1.0)
    fp = jnp.argmax(forced_pref.reshape(-1))
    forced_action = place_idx[fp // NC, fp % NC].astype(jnp.int32)
    # reveal: prefer two-colour hints (1..3), then hint 0
    reveal_legal = legal[A_REVEAL:A_REVEAL + NH]
    reveal_pref = jnp.where(reveal_legal, jnp.asarray([1.0, 3.0, 2.9, 2.8]), -1.0)
    reveal_action = (A_REVEAL + jnp.argmax(reveal_pref)).astype(jnp.int32)
    any_reveal = reveal_legal.any()
    hint_action = jnp.where(bp_safe, place_action, jnp.where(any_reveal, reveal_action, forced_action))

    # ---------------- observation choice by value of information ----------------
    obs_legal = legal[A_OBS:A_OBS + NC]
    jc = jnp.arange(NC * NCOL)
    j_of = jc // NCOL
    c_of = jc % NCOL
    match_j = (COLORINGS[:, j_of].T == c_of[:, None])                        # [27,K]
    w_jc = w[None, :] * match_j.astype(f32)
    p_jc = w_jc.sum(-1)
    w_jc = w_jc / jnp.maximum(p_jc, 1e-9)[:, None]
    Pa_jc, Ep_jc = expected_terms(w_jc)
    U_jc = W_PCONN * Pa_jc + W_PROG * Ep_jc                                  # [27,9,81]
    U_best_jc = jnp.max(jnp.where(move_ok[None], U_jc, -1e9), axis=(1, 2))
    V_obs = (p_jc * U_best_jc).reshape(NC, NCOL).sum(-1)                     # [9]
    obs_score = V_obs + (known < 0).astype(f32) * 0.05 + W_PSEEN * 0.02 * (1.0 - pseen.astype(f32)) + 0.002 * deg - 0.0001 * jnp.arange(NC)
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    o1 = jnp.argmax(obs_score)
    obs_action = (A_OBS + o1).astype(jnp.int32)
    o2 = jnp.argmax(jnp.where(jnp.arange(NC) == o1, -1e9, obs_score))

    # ---------------- information lookahead: E[P_best after observing o1, o2] ----------------
    ab = jnp.arange(NCOL * NCOL)
    ca = ab // NCOL
    cb = ab % NCOL
    match = (COLORINGS[:, o1][None, :] == ca[:, None]) & (COLORINGS[:, o2][None, :] == cb[:, None])   # [9,K]
    w_ab = w[None, :] * match.astype(f32)
    p_ab = w_ab.sum(-1)                                                    # [9]
    w_ab = w_ab / jnp.maximum(p_ab, 1e-9)[:, None]
    P_after_ab, _ = expected_terms(w_ab)                                   # [9,9,81]
    P_best_ab = jnp.max(jnp.where(move_ok[None], P_after_ab, 0.0), axis=(1, 2))
    P1 = jnp.dot(p_ab, P_best_ab)

    # ---------------- end decision ----------------
    t_rem = 2 * n_unrev + n_rev
    c_turn = jnp.where(bp_safe, 1.0, jnp.where(any_reveal, 3.0, 1.0))
    n_unknown = (known < 0).sum()
    H_auto = jnp.dot(w, (placed[None, :] * (2.0 * correct - 1.0)).sum(-1)) + (NH - placed.sum()) * 1.0
    P2 = P1 + (1.0 - P1) * Q_PARTNER
    H1 = H_mean - c_turn - partner_cost
    E_cont_short = jnp.where(t_rem <= 1, P1, P2) * H_auto
    E_cont_long = P2 * jnp.maximum(H1, H_auto) + (1.0 - P2) * Q_FURTHER * jnp.maximum(H1 - 4.0, H_auto)
    E_cont = jnp.where(t_rem <= 2, E_cont_short, E_cont_long)
    do_end = legal[A_END] & (E_end > E_cont + 1e-4)
    P_best = P1

    # ---------------- select by substep ----------------
    fallback = jnp.argmax(legal).astype(jnp.int32)
    action = jnp.where(substep == 0, jnp.where(do_end, jnp.int32(A_END), obs_action),
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(legal[action], action, fallback)
    action = jnp.where(active, action, jnp.int32(0)).astype(jnp.int32)

    trace = {
        "p_conn": P_conn_now.astype(f32),
        "e_end": E_end.astype(f32),
        "e_cont": E_cont.astype(f32),
        "h_mean": H_mean.astype(f32),
        "p_best": P_best.astype(f32),
        "n_known": (NC - n_unknown).astype(jnp.int32),
        "safe_place": bp_safe,
        "partner_cost": partner_cost.astype(f32),
    }
    next_memory = {"known": known, "pseen": pseen, "prev_pos": pos, "logw": logw, "hcount": hcount}
    return action, next_memory, trace
