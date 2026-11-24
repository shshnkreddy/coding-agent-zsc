"""Belief-based Yokai policy.

Maintains an exact posterior over all 1680 colourings of the 9 cards.  Own
observations are hard constraints (read from the active observation under
perfect memory); partner moves are interpreted by inverse planning (softmax
over the partner's tier gain computed from the cards they have seen) and
partner hint placements are soft evidence.  Moves maximise expected
per-colour connectivity progress with a penalty for leaving misplaced cards
as articulation points, observations combine value of information for the
move with entropy, hints are placed only on cards with certainly matching
colours, and the game is ended when the expected value of ending beats a
turn-aware estimate of continuing.
"""

import jax
import jax.numpy as jnp
from jax import lax

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_sorter', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NCOL = 1680          # 9! / (3!)^3 colourings with three cards of each colour
NALL = 19683         # 3 ** 9
NEG = -1e9

# ---------------------------------------------------------------- tunables
BETA = 2.5             # softmax temperature of the partner move model (inverse planning)
LOG_HINT = 1.3862944   # log 4: partner placed a hint whose set contains the card colour
PHI = 0.5              # chance an unsorted board gets fixed before the next chance to end
BRIDGE_PEN = 0.3       # penalty per misplaced card left as an articulation point


def _tables():
    """Static colouring tables built with fixed-shape JAX ops."""
    idx = jnp.arange(NALL, dtype=jnp.int32)
    powers = 3 ** jnp.arange(9, dtype=jnp.int32)
    base3 = (idx[:, None] // powers[None, :]) % 3                     # [19683, 9]
    counts = (base3[:, :, None] == jnp.arange(3)[None, None, :]).sum(1)  # [19683, 3]
    valid = jnp.all(counts == 3, axis=1)
    order = jnp.argsort(jnp.where(valid, 0, 1), stable=True)
    cols = base3[order[:NCOL]]                                        # [1680, 9]
    col_oh = (cols[:, :, None] == jnp.arange(3)[None, None, :]).astype(jnp.float32)  # [1680, 9, 3]
    card = jnp.arange(9, dtype=jnp.int32)
    same = (cols[:, :, None] == cols[:, None, :]) & (card[None, :, None] != card[None, None, :])  # [1680, 9, 9]
    a = jnp.argmax(same, axis=2)                                      # first same-colour partner
    b = 8 - jnp.argmax(same[:, :, ::-1], axis=2)                       # last same-colour partner
    q = a * (17 - a) // 2 + (b - a - 1)                                # index into 36 unordered pairs
    pair_oh = (q[:, :, None] == jnp.arange(36)[None, None, :]).astype(jnp.float32)  # [1680, 9, 36]
    pa = jnp.repeat(card, 9)
    pb = jnp.tile(card, 9)
    keep = pa < pb
    porder = jnp.argsort(jnp.where(keep, 0, 1), stable=True)[:36]
    pair_a = pa[porder]
    pair_b = pb[porder]
    cell_xy = jnp.stack([jnp.arange(81, dtype=jnp.int32) // 9, jnp.arange(81, dtype=jnp.int32) % 9], axis=1)
    return col_oh, pair_oh, pair_a, pair_b, cell_xy


def _init_pos():
    i = jnp.arange(9, dtype=jnp.int32)
    return jnp.stack([i // 3 + 3, i % 3 + 3], axis=1)


def init_agent():
    return {
        'logw': jnp.zeros((NCOL,), dtype=jnp.float32),
        'prev_pos': _init_pos(),
        'prev_placed': jnp.zeros((4,), dtype=jnp.int32),
        'prev_active': jnp.asarray(0, dtype=jnp.int32),
    }


def _tier(E, D2):
    """Progress value of one colour: E adjacent same-colour pairs, D2 any pair at distance 2."""
    return jnp.where(E >= 2, 3.0, jnp.where(E >= 1, 2.0, jnp.where(D2, 1.0, 0.5)))


def _bridges_after_moves(Af, NB, EYE9):
    """bridge[i, p, j]: after moving card i to cell p, is card j an articulation point?"""
    f32 = jnp.float32
    eye = EYE9.astype(f32)
    ids = jnp.arange(9)
    notj = 1.0 - eye                                                   # notj[j, a] = (a != j)
    src = (ids[None, :] == (ids == 0).astype(jnp.int32)[:, None]).astype(f32)  # [9(j), 9]: source card 0, or 1 when j == 0

    def body(i):
        not_i = (ids != i).astype(f32)
        is_i = (ids == i).astype(f32)
        A_keep = Af * not_i[:, None] * not_i[None, :]
        row = NB * not_i[None, :]                                      # [81, 9] new edges of card i
        A_new = A_keep[None] + row[:, :, None] * is_i[None, None, :] + row[:, None, :] * is_i[None, :, None]
        M = A_new[:, None, :, :] * notj[None, :, :, None] * notj[None, :, None, :] + eye[None, None]   # [81, 9(j), 9, 9]
        reach = jnp.broadcast_to(src[None], (81, 9, 9))
        for _ in range(8):
            reach = jnp.minimum(1.0, jnp.einsum('pja,pjab->pjb', reach, M))
        connected = jnp.all((reach > 0.5) | EYE9[None], axis=-1)      # [81, 9(j)]
        return ~connected

    return lax.map(body, ids)                                          # [9, 81, 9]


def agent_act(obs, reset, memory, rng):
    init = init_agent()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    f32 = jnp.float32
    COL_OH, PAIR_OH, PAIR_A, PAIR_B, CELL_XY = _tables()
    EYE9 = jnp.eye(9, dtype=bool)

    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(bool)
    A = (obs.adjacency_matrix[:9, :9] > 0) & ~EYE9
    Af = A.astype(f32)
    active = nf[0, 10] == 1
    substep = nf[0, 11]
    locked = nf[:9, 8] == 1
    pseen = nf[:9, 9] == 1
    own_oh = (nf[:9, 0:3] > 0).astype(f32)
    known = own_oh.sum(1) > 0
    hset = (nf[9:13, 3:6] > 0).astype(f32)    # [4, 3]
    hrev = hset.sum(1) > 0
    hplaced = nf[9:13, 8] == 1
    hpos = nf[9:13, 6:8]
    htgt = jnp.argmax(jnp.all(pos[None, :, :] == hpos[:, None, :], axis=2), axis=1)  # [4]

    # ------------------------------------------------ soft evidence from partner
    partner_acted = memory['prev_active'] == 0
    ppos = memory['prev_pos']
    moved = jnp.any(pos != ppos, axis=1)
    x = jnp.argmax(moved)
    apply_move = partner_acted & moved.any()
    # Inverse planning: the partner is modelled as picking moves by softmax(beta * tier gain)
    # where gains are computed from the cards the partner has seen, on the board before the move.
    Kf = pseen.astype(f32)
    dist_p = jnp.abs(ppos[:, None, :] - ppos[None, :, :]).sum(-1)      # [9, 9]
    Ap = (dist_p == 1).astype(f32)
    D2pf = (dist_p == 2).astype(f32)
    COLK = COL_OH * Kf[None, :, None]
    E_K = 0.5 * jnp.einsum('nik,ij,njk->nk', COLK, Ap, COLK)
    D2_K = jnp.einsum('nik,ij,njk->nk', COLK, D2pf, COLK) > 0
    g_K = _tier(E_K, D2_K)                                             # [1680, 3]
    base_K = jnp.einsum('nik,nk->ni', COL_OH, g_K)                     # [1680, 9]
    cdp = jnp.abs(CELL_XY[:, None, :] - ppos[None, :, :]).sum(-1)      # [81, 9]
    NBp = (cdp == 1).astype(f32)
    C2p = cdp == 2
    ka = Kf[PAIR_A]
    kb = Kf[PAIR_B]
    Epk = ka[None, :] * NBp[:, PAIR_A] + kb[None, :] * NBp[:, PAIR_B] + (ka * kb * Ap[PAIR_A, PAIR_B])[None, :]
    D2pk = ((ka * kb * D2pf[PAIR_A, PAIR_B]) > 0)[None, :] | ((ka > 0)[None, :] & C2p[:, PAIR_A]) | ((kb > 0)[None, :] & C2p[:, PAIR_B])
    G_K = _tier(Epk, D2pk)                                             # [81, 36]
    occ_p = (cdp == 0).any(1)
    nb_other_p = NBp.sum(1)[:, None] - NBp
    geo_p = ((~occ_p)[None, :] & (nb_other_p.T >= 1) & (~locked)[:, None]).astype(f32)   # [9, 81]
    H = jnp.einsum('ip,pq->qi', geo_p, jnp.exp(BETA * G_K))            # [36, 9]
    Hn = jnp.einsum('niq,qi->ni', PAIR_OH, H)                          # [1680, 9]
    n_legal_i = geo_p.sum(1)                                           # [9]
    Z = jnp.sum(Kf[None, :] * jnp.exp(-BETA * base_K) * Hn + (1.0 - Kf)[None, :] * n_legal_i[None, :], axis=1)
    pstar = pos[x, 0] * 9 + pos[x, 1]
    Gx = jnp.einsum('nq,q->n', PAIR_OH[:, x, :], G_K[pstar])           # [1680]
    delta_x = Kf[x] * (Gx - base_K[:, x])
    lf_move = BETA * delta_x - jnp.log(jnp.maximum(Z, 1e-6))
    logw = memory['logw'] + jnp.where(apply_move, lf_move, 0.0)

    newly = hplaced & (memory['prev_placed'] == 0)
    h_new = jnp.argmax(newly)
    t_new = htgt[h_new]
    in_set = jnp.einsum('nk,k->n', COL_OH[:, t_new, :], hset[h_new]) > 0
    logw = logw + jnp.where(partner_acted & newly.any(), jnp.where(in_set, LOG_HINT, 0.0), 0.0)

    # ------------------------------------------------ posterior over colourings
    hard = jnp.einsum('nik,ik->ni', COL_OH, own_oh)          # [1680, 9]
    consistent = jnp.all((hard > 0) | ~known[None, :], axis=1)
    logit = jnp.where(consistent, logw, NEG)
    w = jax.nn.softmax(logit)
    P = jnp.einsum('n,nik->ik', w, COL_OH)                   # [9, 3]

    # ------------------------------------------------ board evaluation
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)  # [9, 9]
    D2f = (dist == 2).astype(f32)
    E = 0.5 * jnp.einsum('nik,ij,njk->nk', COL_OH, Af, COL_OH)     # [1680, 3]
    D2 = jnp.einsum('nik,ij,njk->nk', COL_OH, D2f, COL_OH) > 0
    g = _tier(E, D2)                                                # [1680, 3]
    sorted_n = jnp.all(E >= 2, axis=1)
    p_sorted = jnp.sum(w * sorted_n)

    # ------------------------------------------------ move evaluation
    cd = jnp.abs(CELL_XY[:, None, :] - pos[None, :, :]).sum(-1)     # [81, 9]
    NB = (cd == 1).astype(f32)
    C2 = cd == 2
    e_ab = Af[PAIR_A, PAIR_B]                                       # [36]
    d_ab = dist[PAIR_A, PAIR_B] == 2
    Ep = NB[:, PAIR_A] + NB[:, PAIR_B] + e_ab[None, :]              # [81, 36]
    D2p = d_ab[None, :] | C2[:, PAIR_A] | C2[:, PAIR_B]
    G = _tier(Ep, D2p)                                              # [81, 36]
    W = jnp.einsum('n,niq->iq', w, PAIR_OH)                         # [9, 36]
    gain = W @ G.T                                                  # [9, 81]
    base = jnp.einsum('n,nik,nk->i', w, COL_OH, g)                  # [9]
    delta = gain - base[:, None]
    rng_move, rng_obs = jax.random.split(rng)
    nbcount = NB.sum(1)
    move_legal = legal[10:739].reshape(9, 81)
    # legibility: a sure join of a card the partner has not seen onto a same-colour card the partner has seen
    same_known = (own_oh @ own_oh.T) > 0                                # [9, 9] both observed, same colour
    dec_nb = NB @ (same_known & pseen[None, :]).astype(f32).T           # [81, 9]: cell p adjacent to decodable same-colour card of i
    legible = (known & ~pseen)[:, None] & (dec_nb.T >= 1)               # [9, 81]
    # openness: empty neighbours around the destination keep the cluster joinable
    empty_nb = 4.0 - nbcount - ((CELL_XY[:, 0] == 0) | (CELL_XY[:, 0] == 8)).astype(f32) - ((CELL_XY[:, 1] == 0) | (CELL_XY[:, 1] == 8)).astype(f32)
    # stability (probability a card touches a same-colour card) now, and for the moved card after the move
    S_adj = jnp.einsum('xj,njk->nxk', Af, COL_OH)
    same_adj = jnp.einsum('nxk,nxk->nx', S_adj, COL_OH) >= 1          # [1680, 9]
    stable = w @ same_adj.astype(f32)                                 # [9]
    touch = jnp.minimum(1.0, NB[:, PAIR_A] + NB[:, PAIR_B])           # [81, 36]
    stable_new = W @ touch.T                                          # [9, 81]
    # articulation penalty: misplaced cards left as bridges cannot be moved next
    bridge = _bridges_after_moves(Af, NB, EYE9).astype(f32)           # [9, 81, 9]
    mis = (stable < 0.2).astype(f32)[None, None, :] * (1.0 - EYE9.astype(f32))[:, None, :]   # confidently misplaced, excluding the moved card
    pen = (bridge * mis).sum(-1) + bridge[jnp.arange(9), :, jnp.arange(9)] * (stable_new < 0.2).astype(f32)
    score_move = (delta - 0.02 * jnp.maximum(nbcount - 2.0, 0.0)[None, :]
                  + 0.15 * legible.astype(f32) + 0.02 * known[:, None].astype(f32) * empty_nb[None, :]
                  - BRIDGE_PEN * pen
                  + 1e-4 * jax.random.uniform(rng_move, (9, 81)))
    score_move = jnp.where(move_legal, score_move, NEG)
    move_action = 10 + jnp.argmax(score_move.ravel())
    # geometric approximation of move legality (mask has no moves at substeps 0/1)
    occupied = (cd == 0).any(1)                                        # [81]
    nb_other = NB.sum(1)[:, None] - NB                                 # [81, 9] neighbours excluding card i
    geo_legal = (~occupied)[None, :] & (nb_other.T >= 1) & (~locked)[:, None]   # [9, 81]
    best_now = jnp.max(jnp.where(geo_legal, delta, NEG))

    # ------------------------------------------------ observation choice: myopic VOI + entropy
    cond_w = w[None, None, :] * jnp.transpose(COL_OH, (1, 2, 0))        # [9, 3, 1680]
    cond_w = cond_w / jnp.maximum(cond_w.sum(-1, keepdims=True), 1e-12)
    W_c = jnp.einsum('jkn,niq->jkiq', cond_w, PAIR_OH)                  # [9, 3, 9, 36]
    gain_c = jnp.einsum('jkiq,pq->jkip', W_c, G)                        # [9, 3, 9, 81]
    base_c = jnp.einsum('jkn,nim,nm->jki', cond_w, COL_OH, g)           # [9, 3, 9]
    delta_c = gain_c - base_c[..., None]
    best_c = jnp.max(jnp.where(geo_legal[None, None], delta_c, NEG).reshape(9, 3, -1), axis=-1)  # [9, 3]
    voi = jnp.sum(P * best_c, axis=1) - best_now
    ent = -(P * jnp.log(P + 1e-9)).sum(1)
    obs_score = voi + 0.5 * ent + 0.05 * (~pseen).astype(f32) + 1e-4 * jax.random.uniform(rng_obs, (9,))
    obs_score = jnp.where(legal[1:10], obs_score, NEG)
    obs_action = 1 + jnp.argmax(obs_score)

    # ------------------------------------------------ end decision
    U = (~hrev).sum().astype(f32)
    R = (hrev & ~hplaced).sum().astype(f32)
    corr = jnp.einsum('nhk,hk->nh', COL_OH[:, htgt, :], hset) > 0     # [1680, 4]
    plc_val = ((2.0 * corr - 1.0) * hplaced[None, :].astype(f32)).sum(1)
    val_n = 5.0 * U + 2.0 * R + plc_val
    end_value = jnp.sum(w * sorted_n * val_n)
    s_base = end_value / jnp.maximum(p_sorted, 1e-6)
    remaining = 4.0 + U - hplaced.sum().astype(f32)   # hint actions left: 8 at turn 1 ... 1 at turn 8
    # Next chance to end: partner's turn (cost 1-3) or mine (cost 4); last turn ends by itself (cost ~1).
    cost = jnp.where(remaining >= 3, 3.0, jnp.where(remaining >= 2, 3.5, 1.0))
    p_next = p_sorted + (1.0 - p_sorted) * PHI
    cont_value = p_next * (s_base - cost)
    do_end = legal[1067] & (end_value >= cont_value) & (p_sorted > 0.3)

    # ------------------------------------------------ hint action
    match = jnp.einsum('ik,hk->hi', P, hset)                          # [4, 9]
    certain = match > 0.999
    same_prob = P @ P.T                                               # [9, 9]
    conf = ((locked.astype(f32))[:, None] * same_prob * (1.0 - Af)).sum(0)  # [9]
    cell_idx = pos[:, 0] * 9 + pos[:, 1]                              # [9]
    place_ids = 743 + jnp.arange(4)[:, None] * 81 + cell_idx[None, :]  # [4, 9]
    place_legal = legal[place_ids]
    good = certain & (stable > 0.9)[None, :] & (conf < 0.5)[None, :]
    place_score = (10.0 * good + 3.0 * match + 3.0 * (stable > 0.9)[None, :]
                   - 5.0 * (conf >= 0.5)[None, :] + 0.5 * pseen[None, :].astype(f32))
    place_score = jnp.where(place_legal, place_score, NEG)
    best_place = jnp.argmax(place_score.ravel())
    place_action = 743 + (best_place // 9) * 81 + cell_idx[best_place % 9]
    any_good = jnp.any(good & place_legal)
    forced = ~jnp.any(legal[739:743])
    reveal_score = jnp.where(legal[739:743], jnp.where(jnp.arange(4) == 0, 0.0, 1.0), NEG)
    reveal_action = 739 + jnp.argmax(reveal_score)
    hint_action = jnp.where(any_good | forced, place_action, reveal_action)

    # ------------------------------------------------ select
    act = jnp.where(substep == 0, jnp.where(do_end, 1067, obs_action),
                    jnp.where(substep == 1, obs_action,
                              jnp.where(substep == 2, move_action, hint_action)))
    act = jnp.where(active, act, 0)
    act = jnp.where(legal[act], act, jnp.argmax(legal)).astype(jnp.int32)

    new_memory = {
        'logw': logw,
        'prev_pos': pos,
        'prev_placed': hplaced.astype(jnp.int32),
        'prev_active': active.astype(jnp.int32),
    }
    trace = {
        'p_sorted': p_sorted.astype(f32),
        'n_known': known.sum().astype(jnp.int32),
        'end_value': end_value.astype(f32),
        'cont_value': cont_value.astype(f32),
        'kcol': jnp.where(known, jnp.argmax(own_oh, axis=1), -1).astype(jnp.int32),
        'pos': pos.astype(jnp.int32),
    }
    return act, new_memory, trace
