"""Bayesian yokai policy: exact posterior over all 1680 color assignments.

Knowledge sources (all local): own observed colors (hard constraints), the
fixed color multiset, and soft evidence from the partner's public actions
(moving a card it has observed next to other cards it has observed, leaving
two observed cards apart, placing a revealed hint on a card).  Every decision
is an expectation under this posterior, so no private convention is required
from the partner.  Progress is measured by a lock-aware "moves to connect"
potential per color.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bayes_cluster_v2', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NC = 9          # cards
NCOL = 3        # colors
NH = 4          # hints
G = 9           # grid
NQ = 81         # cells
NA = 1680       # color assignments (9!/(3!3!3!))

# Tunables
ALPHA = 12.0      # likelihood ratio: partner joined two cards it observed
GAMMA = 0.4       # likelihood ratio per turn: partner saw two cards but left them apart
BETA = 6.0        # likelihood ratio: partner placed hint on a card it observed
BETA_WEAK = 2.0   # ... on a card it had not observed
PHI = 0.75        # probability that a needed move gets made per available move
P_MIN_END = 0.45  # never end below this connectivity probability unless it is the last chance
BIG = 10.0        # value of full connectivity in move objective
PLACE_MIN = 0.85  # minimum match probability to place a hint voluntarily
RISK_MAX = 0.15   # maximum probability that locking makes a color impossible
INF = 6.0         # moves-to-connect value for impossible colors
PEN_PSEEN = 0.8   # penalty for moving a card the partner observed but I did not
PEN_PMOVED = 1.0  # extra penalty if the partner also moved that card
LEGIB = 0.5       # penalty per known-different card adjacent to a move destination
REQUIRE_PSEEN = True  # voluntary placement only on partner-seen cards while reveals remain


def _assignment_rows():
    rows = []
    for m0 in range(512):
        if bin(m0).count('1') != 3:
            continue
        for m1 in range(512):
            if bin(m1).count('1') != 3 or (m0 & m1) != 0:
                continue
            row = []
            for i in range(NC):
                if (m0 >> i) & 1:
                    row.append(0)
                elif (m1 >> i) & 1:
                    row.append(1)
                else:
                    row.append(2)
            rows.append(row)
    return rows


def _tables():
    rows = _assignment_rows()
    idx_rows = []    # [NA, 3, 3]: card ids of each color
    other_rows = []  # [NA, 9, 2]: the two other cards sharing card i's color
    for row in rows:
        per_color = [[i for i in range(NC) if row[i] == k] for k in range(NCOL)]
        idx_rows.append(per_color)
        other_rows.append([[j for j in per_color[row[i]] if j != i] for i in range(NC)])
    assign = jnp.asarray(rows, dtype=jnp.int32)                            # [NA, 9]
    onehot = jax.nn.one_hot(assign, NCOL, dtype=jnp.float32)               # [NA, 9, 3]
    same = (assign[:, :, None] == assign[:, None, :]).astype(jnp.float32)
    same = same * (1.0 - jnp.eye(NC, dtype=jnp.float32))[None]            # [NA, 9, 9]
    idx = jnp.asarray(idx_rows, dtype=jnp.int32)                           # [NA, 3, 3]
    other = jnp.asarray(other_rows, dtype=jnp.int32)                       # [NA, 9, 2]
    cell_xy = jnp.asarray([[q // G, q % G] for q in range(NQ)], dtype=jnp.int32)
    eye9 = jnp.eye(NC, dtype=jnp.float32)
    reveal_order = jnp.asarray([1, 2, 3, 0], dtype=jnp.int32)
    return assign, onehot, same, idx, other, cell_xy, eye9, reveal_order


def init_agent():
    init_pos = jnp.asarray([[i // 3 + 3, i % 3 + 3] for i in range(NC)], dtype=jnp.int32)
    return {
        'logw': jnp.zeros((NA,), dtype=jnp.float32),
        'known': -jnp.ones((NC,), dtype=jnp.int32),
        'prev_pos': init_pos,
        'prev_placed': jnp.zeros((NH,), dtype=jnp.int32),
        'applied': jnp.zeros((NC, NC), dtype=jnp.int32),
        'partner_moved': jnp.zeros((NC,), dtype=jnp.int32),
    }


def _f32(x):
    return x.astype(jnp.float32)


def _mtc(d_ab, d_ac, d_bc, la, lb, lc):
    """Moves needed to connect three same-colored cards a, b, c.

    d_* are Manhattan distances, l* are lock flags (0/1 floats).  Broadcasting
    over leading dimensions.  Returns float array; INF when impossible.
    """
    Aab = _f32(d_ab == 1)
    Aac = _f32(d_ac == 1)
    Abc = _f32(d_bc == 1)
    p = Aab + Aac + Abc
    L = la + lb + lc
    # p == 1: identify third card t and the adjacent pair (u, v)
    lt = jnp.where(Aab > 0, lc, jnp.where(Aac > 0, lb, la))
    lu = jnp.where(Aab > 0, la, jnp.where(Aac > 0, la, lb))
    lv = jnp.where(Aab > 0, lb, jnp.where(Aac > 0, lc, lc))
    d_ut = jnp.where(Aab > 0, d_ac, jnp.where(Aac > 0, d_ab, d_ab))
    d_vt = jnp.where(Aab > 0, d_bc, jnp.where(Aac > 0, d_bc, d_ac))
    d_lt = jnp.where(lu > 0, d_ut, d_vt)
    Lp = lu + lv
    val1 = jnp.where(lt == 0, 1.0,
                     jnp.where(Lp == 0, 2.0,
                               jnp.where((Lp == 1) & (d_lt <= 2), 1.0, INF)))
    # p == 0
    dmin = jnp.minimum(jnp.minimum(d_ab, d_ac), d_bc)
    d_ll = jnp.where(la == 0, d_bc, jnp.where(lb == 0, d_ac, d_ab))
    val0 = jnp.where(L <= 1, 2.0,
                     jnp.where((L == 2) & (d_ll <= 2), 1.0, INF))
    return jnp.where(p >= 2, 0.0, jnp.where(p == 1, val1, val0))


def agent_act(obs, reset, memory, rng):
    ASSIGN, ONEHOT, SAME, IDX, OTHER, CELL_XY, EYE9, REVEAL_ORDER = _tables()
    init = init_agent()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    # ------------------------------------------------------------------ parse
    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)                     # [9, 2]
    locked = nf[:NC, 8] > 0                                   # [9]
    lockedf = _f32(locked)
    pseen = nf[:NC, 9] > 0                                    # [9]
    active = nf[0, 10] > 0
    substep = nf[0, 11]
    own = nf[:NC, 0:3]
    has_col = own.sum(1) > 0
    col = jnp.argmax(own, axis=1)
    hint_sets = _f32(nf[NC:NC + NH, 3:6] > 0)                 # [4, 3]
    revealed = hint_sets.sum(1) > 0                           # [4]
    hplaced = nf[NC:NC + NH, 8] > 0                           # [4]
    hpos = nf[NC:NC + NH, 6:8]                                # [4, 2]
    A = _f32(obs.adjacency_matrix[:NC, :NC]) * (1.0 - EYE9)   # [9, 9]
    legal = obs.legal_actions
    target = _f32(jnp.all(pos[None, :, :] == hpos[:, None, :], axis=2)) * _f32(hplaced)[:, None]  # [4, 9]
    d_card = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(2)      # [9, 9]

    # ---------------------------------------------------------- memory update
    known = jnp.where(has_col, col, memory['known'])
    logw = memory['logw']
    applied = memory['applied']

    # partner move evidence (visible while we are inactive)
    moved = jnp.any(pos != memory['prev_pos'], axis=1)        # [9]
    partner_move = (~active) & jnp.any(moved)
    X = jnp.argmax(moved)
    nbX = A[X] > 0
    cand = nbX & pseen & (applied[X] == 0)
    apply_move = partner_move & pseen[X] & jnp.any(cand)
    sameX = jnp.take(SAME, X, axis=1)                         # [NA, 9]
    ev_move = jnp.max(sameX * _f32(cand)[None, :], axis=1)    # [NA]
    logw = logw + jnp.where(apply_move, jnp.log(ALPHA) * ev_move, 0.0)
    eX = _f32(jnp.arange(NC) == X)
    pair_new = (eX[:, None] * _f32(cand)[None, :] + _f32(cand)[:, None] * eX[None, :]) > 0
    applied = jnp.where(apply_move, jnp.maximum(applied, pair_new.astype(jnp.int32)), applied)
    # negative evidence: partner has seen both cards, at least one is movable,
    # yet they are not adjacent after the partner's move -> likely different
    movable_pair = (~locked)[:, None] | (~locked)[None, :]
    seen_pair = pseen[:, None] & pseen[None, :] & movable_pair & (A == 0) & (~EYE9.astype(bool))
    neg_count = jnp.einsum('nij,ij->n', SAME, _f32(seen_pair)) * 0.5
    logw = logw + jnp.where(partner_move, jnp.log(GAMMA) * neg_count, 0.0)
    partner_moved = jnp.maximum(memory['partner_moved'], jnp.where(partner_move, moved.astype(jnp.int32), 0))

    # partner placement evidence (visible at our substep 0)
    newly = hplaced & (memory['prev_placed'] == 0)
    partner_place = active & jnp.any(newly)
    h_new = jnp.argmax(newly)
    tcard = target[h_new]                                     # [9]
    S_new = hint_sets[h_new]                                  # [3]
    match_n = jnp.einsum('i,nik,k->n', tcard, ONEHOT, S_new)  # [NA] in {0,1}
    seen_t = jnp.sum(tcard * _f32(pseen)) > 0
    beta = jnp.where(seen_t, jnp.log(BETA), jnp.log(BETA_WEAK))
    logw = logw + jnp.where(partner_place, beta * match_n, 0.0)

    new_memory = {
        'logw': logw,
        'known': known,
        'prev_pos': pos,
        'prev_placed': hplaced.astype(jnp.int32),
        'applied': applied,
        'partner_moved': partner_moved,
    }

    # -------------------------------------------------------------- posterior
    consistent = jnp.all((known[None, :] < 0) | (ASSIGN == known[None, :]), axis=1)  # [NA]
    lw = jnp.where(consistent, logw, -1e9)
    w = jax.nn.softmax(lw)                                    # [NA]

    # moves-to-connect per assignment and color
    Pa = pos[IDX[:, :, 0]]                                    # [NA, 3, 2]
    Pb = pos[IDX[:, :, 1]]
    Pc = pos[IDX[:, :, 2]]
    la = lockedf[IDX[:, :, 0]]
    lb = lockedf[IDX[:, :, 1]]
    lc = lockedf[IDX[:, :, 2]]
    mtc = _mtc(jnp.abs(Pa - Pb).sum(2), jnp.abs(Pa - Pc).sum(2), jnp.abs(Pb - Pc).sum(2), la, lb, lc)  # [NA, 3]
    need = mtc.sum(1)                                         # [NA]
    conn = need == 0
    connf = _f32(conn)
    P_now = jnp.sum(w * connf)

    marg = jnp.einsum('n,nik->ik', w, ONEHOT)                 # [9, 3]
    ent = -jnp.sum(marg * jnp.log(marg + 1e-9), axis=1)       # [9]
    joint = jnp.einsum('n,nik->ik', w * connf, ONEHOT)        # [9, 3]
    conninfo = jnp.sum(jnp.abs(joint - P_now * marg), axis=1)  # [9]
    pair_prob = jnp.einsum('n,nij->ij', w, SAME)              # [9, 9]

    # ------------------------------------------------------- hint economy
    U = jnp.sum(_f32((~revealed) & (~hplaced)))
    R = jnp.sum(_f32(revealed & (~hplaced)))
    match_ph = jnp.einsum('hi,nik,hk->nh', target, ONEHOT, hint_sets)  # [NA, 4]
    D_n = jnp.sum((2.0 * match_ph - 1.0) * _f32(hplaced)[None, :], axis=1)  # [NA]
    S_n = 5.0 * U + 2.0 * R + D_n
    E_end = jnp.sum(w * connf * S_n)
    E_D = jnp.sum(w * D_n)

    # ------------------------------------------------------- move evaluation
    d_cell = jnp.abs(CELL_XY[:, None, :] - pos[None, :, :]).sum(2)  # [81, 9]
    Nq = _f32(d_cell == 1)                                          # [81, 9]
    crowd = Nq.sum(1)                                               # [81]
    centroid = _f32(pos).mean(0)
    dist_c = jnp.abs(_f32(CELL_XY) - centroid[None, :]).sum(1)      # [81]
    unknown_me = known < 0

    ev_rows = []
    pa_rows = []
    risk_rows = []
    cost_rows = []
    for i in range(NC):
        j1 = OTHER[:, i, 0]
        j2 = OTHER[:, i, 1]
        P1 = pos[j1]                                                # [NA, 2]
        P2 = pos[j2]
        l1 = lockedf[j1]
        l2 = lockedf[j2]
        d12 = jnp.abs(P1 - P2).sum(1)                               # [NA]
        oh_ci = ONEHOT[:, i, :]                                     # [NA, 3]
        mtc_ci = jnp.sum(mtc * oh_ci, axis=1)                       # [NA] current value of my color
        rest = need - mtc_ci                                        # [NA]
        # move lookahead: card i at every cell
        dq1 = jnp.abs(CELL_XY[None, :, :] - P1[:, None, :]).sum(2)  # [NA, 81]
        dq2 = jnp.abs(CELL_XY[None, :, :] - P2[:, None, :]).sum(2)
        mtc_new = _mtc(dq1, dq2, d12[:, None], 0.0, l1[:, None], l2[:, None])  # [NA, 81]
        need_new = rest[:, None] + mtc_new
        full = _f32(need_new == 0)
        V = BIG * full - need_new
        ev_rows.append(w @ V)
        pa_rows.append(w @ full)
        # lock lookahead: card i locked in place
        dX1 = jnp.abs(pos[i][None, :] - P1).sum(1)
        dX2 = jnp.abs(pos[i][None, :] - P2).sum(1)
        mtc_lock = _mtc(dX1, dX2, d12, 1.0, l1, l2)                 # [NA]
        risk_rows.append(jnp.sum(w * _f32(mtc_lock >= INF)))
        cost_rows.append(jnp.sum(w * (mtc_lock - mtc_ci)))
    EV = jnp.stack(ev_rows)                                         # [9, 81]
    PA = jnp.stack(pa_rows)                                         # [9, 81]
    risk = jnp.stack(risk_rows)                                     # [9]
    lock_cost = jnp.stack(cost_rows)                                # [9]
    move_legal = legal[10:739].reshape(NC, NQ)
    tie = 0.02 * (crowd[None, :] - Nq.T) - 0.005 * dist_c[None, :]
    # authority: the partner knows cards it observed; do not gamble with them
    tie = tie - PEN_PSEEN * _f32(pseen & unknown_me)[:, None]
    tie = tie - PEN_PMOVED * _f32((partner_moved > 0) & unknown_me)[:, None]
    tie = tie - 0.3 * _f32((partner_moved > 0) & (~unknown_me))[:, None]
    # legibility: avoid placing a known card next to cards I know are different,
    # because a partner may read adjacency between my observed cards as a join
    known_oh = jax.nn.one_hot(jnp.maximum(known, 0), NCOL, dtype=jnp.float32) * _f32(known >= 0)[:, None]  # [9,3]
    diff_known = (1.0 - known_oh @ known_oh.T) * (_f32(known >= 0)[:, None] * _f32(known >= 0)[None, :])  # [9,9]
    n_diff_adj = diff_known @ Nq.T                                   # [9, 81]
    tie = tie - LEGIB * n_diff_adj
    move_score = jnp.where(move_legal, EV + tie, -1e9)
    move_flat = jnp.argmax(move_score.reshape(-1))
    move_action = 10 + move_flat
    P_after = jnp.max(jnp.where(move_legal, PA, 0.0))

    # ------------------------------------------------------- observation choice
    obs_score = _f32(unknown_me) * (1.0 + ent + 2.0 * conninfo + 0.3 * _f32(~pseen))
    obs_legal = legal[1:10]
    obs_action = 1 + jnp.argmax(jnp.where(obs_legal, obs_score, -1e9))

    # ------------------------------------------------------- hint choice
    match = hint_sets @ marg.T                                      # [4, 9] P(c_X in S_h)
    settled = jnp.sum(A * pair_prob, axis=1)                        # [9]
    qcard = pos[:, 0] * G + pos[:, 1]                               # [9]
    place_idx = 743 + jnp.arange(NH)[:, None] * NQ + qcard[None, :]  # [4, 9]
    place_legal = legal[place_idx]
    single = _f32(jnp.arange(NH) == 0)
    place_score = (match - 5.0 * risk[None, :] - 1.0 * lock_cost[None, :] + 0.3 * settled[None, :]
                   + 0.5 * _f32(pseen)[None, :] + 0.1 * single[:, None])
    certain = place_legal & (match >= PLACE_MIN) & (risk[None, :] < RISK_MAX) & (lock_cost[None, :] <= 0.5)
    reveal_legal = legal[739:743]
    any_reveal = jnp.any(reveal_legal)
    # placement restraint: while a reveal is available, only lock cards the
    # partner has already observed so it never loses access to a color
    certain = certain & (pseen[None, :] | (~any_reveal)) if REQUIRE_PSEEN else certain
    any_certain = jnp.any(certain)
    best_certain = jnp.argmax(jnp.where(certain, place_score, -1e9).reshape(-1))
    best_any = jnp.argmax(jnp.where(place_legal, place_score, -1e9).reshape(-1))
    rl_order = reveal_legal[REVEAL_ORDER]
    reveal_h = REVEAL_ORDER[jnp.argmax(rl_order)]
    reveal_action = 739 + reveal_h
    place_flat = place_idx.reshape(-1)
    place_action_c = place_flat[best_certain]
    place_action_a = place_flat[best_any]
    hint_action = jnp.where(any_certain, place_action_c,
                            jnp.where(any_reveal, reveal_action, place_action_a))
    my_cost = jnp.where(any_certain, 1.0, jnp.where(any_reveal, 3.0, 1.7))

    # ------------------------------------------------------- end decision
    rem = 2.0 * U + R                       # turns (moves) left before forced end
    S_base = 5.0 * U + 2.0 * R + E_D - my_cost - 2.0
    V_cont = -1e9
    for h in range(1, 5):
        P_h = jnp.sum(w * (PHI ** need) * _f32(need <= 2.0 * h))
        S_h = S_base - 4.0 * (h - 1)
        valid = (2.0 * h) < rem
        V_cont = jnp.maximum(V_cont, jnp.where(valid, S_h * P_h, -1e9))
    P_term = jnp.sum(w * (PHI ** need) * _f32(need <= rem))
    V_term = (E_D + 0.65 * rem) * P_term
    V_cont = jnp.maximum(V_cont, V_term)
    end_ok = legal[1067] & (E_end >= V_cont) & ((P_now >= P_MIN_END) | (rem <= 2.0))
    step0_action = jnp.where(end_ok, 1067, obs_action)

    # ------------------------------------------------------- dispatch
    action = jnp.where(substep == 0, step0_action,
                       jnp.where(substep == 1, obs_action,
                                 jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, 0)
    action = jnp.where(legal[action], action, jnp.argmax(legal)).astype(jnp.int32)

    trace = {
        'p_conn': P_now,
        'e_end': E_end,
        'v_cont': V_cont,
        'p_after': P_after,
        'n_known': jnp.sum(_f32(known >= 0)),
        'need': jnp.sum(w * need),
        'known': known,
        'pos': pos.reshape(-1),
        'locked': locked.astype(jnp.int32),
        'pseen': pseen.astype(jnp.int32),
    }
    return action, new_memory, trace
