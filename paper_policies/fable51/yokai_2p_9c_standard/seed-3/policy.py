"""Belief-enumeration Yokai policy (convention-free, local-evidence only).

Each seat keeps a private belief over all 1680 colour assignments consistent
with the cards it has observed itself (plus soft evidence from the partner's
moves).  Moves greedily raise expected same-colour adjacency, hints are placed
only on cards known to match, and the game is ended when the probability that
every colour is connected makes ending worth more than continuing.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_cluster_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

# ----------------------------------------------------------------------------
# Static tables (built inside a function: module-scope calls are not allowed)
# ----------------------------------------------------------------------------
NC = 9          # cards
NH = 4          # hints
G = 9           # grid
NCELL = 81
NASSIGN = 1680
F32 = jnp.float32

# Tunables
EPS_EVIDENCE = 0.3      # weight of assignments contradicting a partner move
EPS_STRONG = 0.1        # ... once at least one partner move was verified consistent
EPS_WEAK = 1.0          # ... once a partner move was verified inconsistent
EPS_PLACE = 0.3         # weight of assignments contradicting a partner hint placement
RESTRICT_PSEEN = 1      # restrict move evidence to partner-seen neighbours when any
ADAPTIVE_TRUST = 1
MI_WEIGHT = 1.0         # observation bonus: mutual information with all-connected
LAM_ALLCONN = 1.0       # bonus for P(all connected) after move
PEN_UNKNOWN_MOVE = 0.15
PEN_PARTNER_MOVED = 0.5
PLACE_THRESHOLD = 0.95
P_NEXT_EST = 0.7
LAST_LOOKAHEAD = 1
SPEC_W = 1.0            # weight of uncertain (speculative) adjacency gains in move scoring
LOCK_PSEEN_BONUS = 1.0  # prefer locking cards the partner has already seen
OBS0_PSEEN_BONUS = 0.0  # at substep 0 prefer observing partner-seen (lockable) cards
OBS_UNSEEN_BONUS = 0.3  # prefer cards nobody has seen (coverage)
REVEAL_ZERO_FIRST = 1   # reveal the single-colour hint first
STRUCT_EVIDENCE = 1     # evidence strength depends on what the partner had seen
EPS_UNSEEN_MOVER = 0.7  # partner moved a card it never observed
EPS_STRONG_NB = 0.15    # partner moved a seen card next to partner-seen cards
EPS_VER_MULT = 0.5      # multiplier once a partner move was verified consistent
PROD_W = 0.5            # observation bonus for enabling a productive move when none is known


def _tables():
    idx = jnp.arange(3 ** NC, dtype=jnp.int32)
    powers = 3 ** jnp.arange(NC, dtype=jnp.int32)
    digits = (idx[:, None] // powers[None, :]) % 3                      # [19683, 9]
    counts = jnp.stack([jnp.sum(digits == c, axis=1) for c in range(3)], axis=1)
    valid = jnp.all(counts == 3, axis=1)
    sel = jnp.nonzero(valid, size=NASSIGN, fill_value=0)[0]
    assign = digits[sel]                                                 # [1680, 9]
    eye9 = jnp.eye(NC, dtype=F32)
    same_nd = (assign[:, :, None] == assign[:, None, :]).astype(F32) * (1.0 - eye9)[None]
    colmask = (assign[:, None, :] == jnp.arange(3, dtype=jnp.int32)[None, :, None]).astype(F32)   # [A,3,9]
    cidx = jnp.arange(NCELL, dtype=jnp.int32)
    cx = (cidx // G).astype(F32)
    cy = (cidx % G).astype(F32)
    cell_adj = ((jnp.abs(cx[:, None] - cx[None, :]) + jnp.abs(cy[:, None] - cy[None, :])) == 1).astype(F32)
    reveal_pref = jnp.asarray([0.0, 3.0, 2.0, 1.0], dtype=F32)
    return assign, same_nd, colmask, cell_adj, cx, cy, eye9, reveal_pref


def init_agent():
    return {
        'known': -jnp.ones((NC,), dtype=jnp.int32),
        'pseen': jnp.zeros((NC,), dtype=jnp.int32),
        'pmove': jnp.zeros((NC, NC), dtype=jnp.int32),
        'pmoved': jnp.zeros((NC,), dtype=jnp.int32),
        'pmove_cls': jnp.zeros((NC,), dtype=jnp.int32),
        'last_pos': jnp.zeros((NC, 2), dtype=jnp.int32),
        'my_turns': jnp.zeros((), dtype=jnp.int32),
        'seat': -jnp.ones((), dtype=jnp.int32),
        'pplaced': jnp.zeros((NH,), dtype=jnp.int32),
        'placed_prev': jnp.zeros((NH,), dtype=jnp.int32),
    }


def agent_act(obs, reset, memory, rng):
    ASSIGN, SAME_ND, COLMASK, CELL_ADJ, CELL_X, CELL_Y, EYE9, REVEAL_PREF = _tables()
    OFFDIAG = 1.0 - EYE9
    nf = obs.node_features.astype(jnp.int32)
    adj = obs.adjacency_matrix.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(jnp.bool_)

    substep = nf[0, 11]
    active = nf[0, 10] > 0

    # ------------------------------------------------------------------ memory
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    last_pos = jnp.where(reset, pos, mem['last_pos'])
    seat = jnp.where(reset, jnp.where(active, 0, 1), mem['seat'])

    col = nf[:NC, 0:3]
    has_col = col.sum(1) > 0
    known = jnp.where(has_col, jnp.argmax(col, axis=1).astype(jnp.int32), mem['known'])
    pseen = jnp.maximum(mem['pseen'], nf[:NC, 9])

    card_adj_i = adj[:NC, :NC] * (1 - jnp.eye(NC, dtype=jnp.int32))
    moved = jnp.any(pos != last_pos, axis=1)
    partner_moved = moved & (~active)
    # neighbours of the moved card; restricted to partner-seen cards when possible
    seen_nb = card_adj_i * (pseen > 0).astype(jnp.int32)[None, :]
    any_seen_nb = seen_nb.sum(1) > 0
    nb_evidence = jnp.where((any_seen_nb[:, None]) & (RESTRICT_PSEEN > 0), seen_nb, card_adj_i)
    pmove = jnp.where(partner_moved[:, None], nb_evidence, mem['pmove'])
    pmoved = jnp.maximum(mem['pmoved'], partner_moved.astype(jnp.int32))
    # evidence class of the partner move: 1 mover unseen by partner, 2 seen mover / unseen
    # neighbours, 3 seen mover with partner-seen neighbours
    mover_seen = pseen > 0
    cls_now = jnp.where(~mover_seen, 1, jnp.where(any_seen_nb, 3, 2)).astype(jnp.int32)
    pmove_cls = jnp.where(partner_moved, cls_now, mem['pmove_cls'])
    my_turns = mem['my_turns'] + (active & (substep == 0)).astype(jnp.int32)

    # partner hint placements (hint became placed while we were inactive)
    placed_now = (nf[NC:NC + NH, 8] > 0).astype(jnp.int32)
    # our own placement (our substep 3) is followed by an observation where we are inactive
    newly_placed = (placed_now > 0) & (mem['placed_prev'] == 0) & active & (~reset)
    pplaced = jnp.maximum(mem['pplaced'], newly_placed.astype(jnp.int32))

    # ------------------------------------------------------------------ belief
    consistent = jnp.all((known[None, :] < 0) | (ASSIGN == known[None, :]), axis=1)
    pm = pmove.astype(F32)
    has_m = pm.sum(1) > 0                                       # [9]
    ev = jnp.einsum('aij,ij->ai', SAME_ND, pm) > 0              # [A,9]
    # adaptive trust: verify partner moves against own knowledge
    kx = known >= 0
    same_known = (known[:, None] == known[None, :]) & kx[:, None] & kx[None, :]
    nb_same_known = jnp.sum(pmove * same_known.astype(jnp.int32), axis=1)
    nb_unknown = jnp.sum(pmove * (~kx).astype(jnp.int32)[None, :], axis=1)
    verified = has_m & kx & (nb_same_known > 0)
    inconsistent = has_m & kx & (nb_same_known == 0) & (nb_unknown == 0)
    n_ver = jnp.sum(verified)
    n_inc = jnp.sum(inconsistent)
    eps_eff = jnp.where(n_inc > 0, EPS_WEAK, jnp.where(n_ver >= 1, EPS_STRONG, EPS_EVIDENCE))
    eps_eff = jnp.where(ADAPTIVE_TRUST > 0, eps_eff, EPS_EVIDENCE)
    eps_cls_tab = jnp.asarray([1.0, EPS_UNSEEN_MOVER, EPS_EVIDENCE, EPS_STRONG_NB], dtype=F32)
    eps_card_s = eps_cls_tab[pmove_cls]
    eps_card_s = jnp.where(n_inc > 0, EPS_WEAK, jnp.where(n_ver >= 1, eps_card_s * EPS_VER_MULT, eps_card_s))
    eps_card = jnp.where(STRUCT_EVIDENCE > 0, eps_card_s, eps_eff)
    factor = jnp.prod(jnp.where(has_m[None, :] & (~ev), eps_card[None, :], 1.0), axis=1)
    # partner placements: colour of target card should be in the hint set
    hset_e = nf[NC:NC + NH, 3:6] > 0                            # [4,3]
    tgt_e = (adj[NC:NC + NH, :NC] > 0).astype(F32)              # [4,9]
    in_set = hset_e[:, ASSIGN].astype(F32)                      # [4,A,9]
    tgt_in_set = jnp.einsum('hai,hi->ha', in_set, tgt_e)        # [4,A] (0/1 when placed)
    place_ok = (tgt_in_set > 0.5) | (pplaced[:, None] == 0)
    # verify partner placements against own knowledge: a placement on a card we know does
    # not match means the partner's placements are unreliable
    known_onehot = (known[:, None] == jnp.arange(3)[None, :]) & kx[:, None]          # [9,3]
    tgt_known_col = jnp.einsum('hi,ic->hc', tgt_e, known_onehot.astype(F32)) > 0.5    # [4,3]
    tgt_known = jnp.any(tgt_known_col, axis=1)
    tgt_match_known = jnp.any(tgt_known_col & hset_e, axis=1)
    bad_place = (pplaced > 0) & tgt_known & (~tgt_match_known)
    eps_place_eff = jnp.where(jnp.any(bad_place), 1.0, EPS_PLACE)
    factor = factor * jnp.prod(jnp.where(place_ok, 1.0, eps_place_eff), axis=0)
    w = consistent.astype(F32) * factor
    wsum = jnp.maximum(w.sum(), 1e-9)
    wn = w / wsum                                               # [A]

    N = card_adj_i.astype(F32)                                  # [9,9]
    pairs = 0.5 * jnp.einsum('aci,ij,acj->ac', COLMASK, N, COLMASK)   # [A,3]
    conn = pairs >= 1.5                                         # [A,3]
    allconn = jnp.all(conn, axis=1)
    p_all = jnp.sum(wn * allconn)

    # per-card marginals
    pc = jnp.einsum('a,aci->ic', wn, COLMASK)                   # [9,3]
    ent = -jnp.sum(pc * jnp.log(pc + 1e-9), axis=1)             # [9]
    ps = jnp.einsum('a,aij->ij', wn, SAME_ND)                   # P(same colour) [9,9]

    # ------------------------------------------------------------------ moves
    cell = pos[:, 0] * G + pos[:, 1]                            # [9]
    ncell = CELL_ADJ[:, cell]                                   # [81,9]: cell p adjacent to card j
    mates_adj = jnp.einsum('ij,aij->ai', N, SAME_ND)            # [A,9] mates adjacent to i
    conn_ci = jnp.take_along_axis(conn, ASSIGN, axis=1)          # [A,9] colour of i connected
    # base: are the two mates of i adjacent to each other (independent of i)?
    pairs_ci = jnp.take_along_axis(pairs, ASSIGN, axis=1)        # [A,9]
    base = pairs_ci - mates_adj                                  # [A,9] in {0,1}
    b1 = base >= 0.5
    b0 = ~b1
    others_ok = jnp.all(conn[:, None, :] | (jnp.arange(3)[None, None, :] == ASSIGN[:, :, None]), axis=2)  # [A,9]

    def expect_new_conn(V):
        # V: [A,9] weights; E_V[[new pairs >= 2]] as a function of destination cell -> [9,81]
        z1 = (V * b1)[:, :, None] * SAME_ND                       # [A,9,9]
        w1 = jnp.sum(z1, axis=0)                                  # [9,9]
        q = jnp.einsum('aij,aik->ijk', (V * (b0.astype(F32) - b1.astype(F32)))[:, :, None] * SAME_ND, SAME_ND)
        q = q * OFFDIAG[None, :, :]
        t1 = jnp.einsum('pj,ij->ip', ncell, w1)
        t2 = 0.5 * jnp.einsum('pj,pk,ijk->ip', ncell, ncell, q)
        return t1 + t2

    ps_cert = (ps >= 0.999).astype(F32)
    ps_spec = ps - ps_cert
    ps_eff = ps_cert + SPEC_W * ps_spec
    e_added = jnp.einsum('pj,ij->ip', ncell, ps_eff)             # [9,81]
    e_removed = jnp.sum(N * ps_eff, axis=1)                      # [9]
    e_old_conn = jnp.einsum('a,ai->i', wn, conn_ci.astype(F32))  # [9]
    e_new_conn = expect_new_conn(jnp.broadcast_to(wn[:, None], (NASSIGN, NC)))   # [9,81]
    p_all_after = expect_new_conn(wn[:, None] * others_ok.astype(F32))          # [9,81]

    cx = jnp.mean(pos[:, 0].astype(F32))
    cy = jnp.mean(pos[:, 1].astype(F32))
    compact = -(jnp.abs(CELL_X - cx) + jnp.abs(CELL_Y - cy))     # [81]

    J = (e_added - e_removed[:, None]
         + 0.5 * (e_new_conn - e_old_conn[:, None])
         + LAM_ALLCONN * p_all_after
         + 0.01 * compact[None, :]
         - PEN_UNKNOWN_MOVE * (known < 0).astype(F32)[:, None]
         - PEN_PARTNER_MOVED * pmoved.astype(F32)[:, None])
    move_legal = legal[10:739].reshape(NC, NCELL)
    J = jnp.where(move_legal, J, -1e9)
    move_idx = jnp.argmax(J.reshape(-1))
    move_action = (10 + move_idx).astype(jnp.int32)

    # ------------------------------------------------------------------ observe
    deg = N.sum(1)
    joint = jnp.einsum('a,aci->ic', wn * allconn.astype(F32), COLMASK)   # [9,3] P(c_i=c, allconn)
    pcond = joint / jnp.maximum(pc, 1e-9)

    def hbin(p):
        p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
        return -(p * jnp.log(p) + (1.0 - p) * jnp.log(1.0 - p))

    mi = hbin(p_all) - jnp.sum(pc * hbin(pcond), axis=1)
    pseen_f = pseen.astype(F32)
    # productive-move enabling: is there a known card of colour c not adjacent to i?
    cert_gain = jnp.einsum('pj,ij->ip', ncell, ps_cert) - jnp.sum(N * ps_cert, axis=1)[:, None]
    have_prod = jnp.max(jnp.where(move_legal, cert_gain, -1.0)) >= 0.99
    known_col = (known[:, None] == jnp.arange(3)[None, :]) & kx[:, None]            # [9,3]
    nonadj_known = jnp.einsum('jc,ij->ic', known_col.astype(F32), (1.0 - N) * (1.0 - EYE9)) > 0   # [9,3]
    prod = jnp.sum(pc * nonadj_known.astype(F32), axis=1) * (ent > 0.05)
    obs_score = (ent + MI_WEIGHT * mi + OBS_UNSEEN_BONUS * (1.0 - pseen_f) + 0.05 * deg
                 + jnp.where(substep == 0, OBS0_PSEEN_BONUS, 0.0) * pseen_f * (ent > 0.05)
                 + PROD_W * (1.0 - have_prod.astype(F32)) * prod)
    obs_legal = legal[1:10]
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    obs_action = (1 + jnp.argmax(obs_score)).astype(jnp.int32)

    # ------------------------------------------------------------------ hints
    hset = nf[NC:NC + NH, 3:6] > 0                               # [4,3]
    revealed = jnp.any(hset, axis=1)
    placed = nf[NC:NC + NH, 8] > 0
    U = jnp.sum(~revealed).astype(F32)
    R = jnp.sum(revealed & (~placed)).astype(F32)
    hint_target = (adj[NC:NC + NH, :NC] > 0).astype(F32)         # [4,9]
    match = jnp.einsum('a,hai->hi', wn, hset[:, ASSIGN].astype(F32))   # [4,9] P(colour i in set h)
    eci = jnp.sum(placed.astype(F32)[:, None] * hint_target * (2.0 * match - 1.0))
    s_now = 5.0 * U + 2.0 * R + eci

    locked = nf[:NC, 8] > 0
    dist = jnp.abs(pos[:, None, 0] - pos[None, :, 0]) + jnp.abs(pos[:, None, 1] - pos[None, :, 1])
    risk_w = 10.0 * (dist >= 3).astype(F32) + 3.0 * (dist == 2).astype(F32)
    lock_risk = jnp.sum(locked.astype(F32)[None, :] * ps * risk_w, axis=1)     # [9]
    adj_same = jnp.einsum('a,ai->i', wn, (mates_adj > 0.5).astype(F32))
    L = adj_same + 0.5 * e_old_conn - lock_risk + LOCK_PSEEN_BONUS * pseen.astype(F32)   # [9]
    place_score = 6.0 * match + L[None, :]                       # [4,9]
    place_legal_cells = legal[743:1067].reshape(NH, NCELL)
    place_legal = place_legal_cells[:, cell]                     # [4,9]
    place_score = jnp.where(place_legal, place_score, -1e9)
    pidx = jnp.argmax(place_score.reshape(-1))
    ph = pidx // NC
    pi = pidx % NC
    best_match = match.reshape(-1)[pidx]
    best_L = L[pi]
    place_action = (743 + ph * NCELL + cell[pi]).astype(jnp.int32)

    reveal_legal = legal[739:743]
    any_reveal = jnp.any(reveal_legal)
    reveal_pref = jnp.where(REVEAL_ZERO_FIRST > 0, 4.0 - REVEAL_PREF, REVEAL_PREF)
    reveal_action = (739 + jnp.argmax(jnp.where(reveal_legal, reveal_pref, -1e9))).astype(jnp.int32)
    do_place = ((best_match >= PLACE_THRESHOLD) & (best_L > -2.0)) | (~any_reveal)
    hint_action = jnp.where(do_place, place_action, reveal_action)

    # ------------------------------------------------------------------ end
    last_turn_p1 = (seat == 1) & (my_turns >= 4)
    delta = jnp.where(last_turn_p1, 1.0, 4.0)
    end_value = p_all * s_now
    max_p_after = jnp.max(jnp.where(move_legal, p_all_after, 0.0))
    p_next = jnp.where(last_turn_p1 & (LAST_LOOKAHEAD > 0), jnp.maximum(P_NEXT_EST, max_p_after), P_NEXT_EST)
    cont_value = p_next * (s_now - delta)
    do_end = (end_value >= cont_value) & legal[1067]
    end_action = jnp.asarray(1067, dtype=jnp.int32)

    a0 = jnp.where(do_end, end_action, obs_action)
    chosen = jnp.select([substep == 0, substep == 1, substep == 2], [a0, obs_action, move_action], hint_action)
    chosen = jnp.where(active, chosen, jnp.asarray(0, dtype=jnp.int32))
    chosen = jnp.where(legal[chosen], chosen, jnp.argmax(legal).astype(jnp.int32))

    new_mem = {
        'known': known.astype(jnp.int32),
        'pseen': pseen.astype(jnp.int32),
        'pmove': pmove.astype(jnp.int32),
        'pmoved': pmoved.astype(jnp.int32),
        'pmove_cls': pmove_cls.astype(jnp.int32),
        'last_pos': pos.astype(jnp.int32),
        'my_turns': my_turns.astype(jnp.int32),
        'seat': seat.astype(jnp.int32),
        'pplaced': pplaced.astype(jnp.int32),
        'placed_prev': placed_now.astype(jnp.int32),
    }
    trace = {
        'p_all': p_all.astype(F32),
        's_now': s_now.astype(F32),
        'n_known': jnp.sum(known >= 0).astype(jnp.int32),
        'my_turns': my_turns.astype(jnp.int32),
        'do_end': do_end.astype(jnp.int32),
        'known': known.astype(jnp.int32),
        'map_assign': ASSIGN[jnp.argmax(wn)].astype(jnp.int32),
        'max_p_after': max_p_after.astype(F32),
        'eps_eff': eps_eff.astype(F32),
    }
    return chosen.astype(jnp.int32), new_mem, trace
