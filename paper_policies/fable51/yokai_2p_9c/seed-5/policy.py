"""Bayesian enumeration policy for two-player 9-card Yokai.

Enumerates all 1680 colour assignments, applies hard constraints from own
observations and soft public evidence (hint placements, partner clustering
moves), then chooses actions from the posterior:
  * observe: card with highest colour entropy / information about solvedness
  * move: legal move maximising posterior expected number of connected colours
  * hint: place a revealed hint on a card likely to match, else reveal
  * end: when posterior P(solved) * score beats expected value of continuing
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bayes_enum_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

# ----------------------------------------------------------------------------
# Static tables (built inside functions: module-scope calls are not allowed)
# ----------------------------------------------------------------------------
NC = 1680

# Tunables
LOG_ALPHA_SEEN = 0.0               # partner moved a seen card next to a same-colour partner-seen card
LOG_ALPHA_ANY = -0.5108256         # log(0.6): same-colour neighbour but not partner-seen
LOG_ALPHA_NONE = -1.6094379        # log(0.2): no same-colour neighbour at all
LOG_BETA_HINT = -2.3025851         # log(0.1): partner placed hint whose set excludes card colour
PLACE_THRESH = 0.7                 # min P(match) to place instead of reveal
END_GAIN = 0.25                    # anticipated increase of P(solved) after one more own turn
END_GAIN_MOVE = 0.15               # added on top of best one-move solved probability
END_MIN_P = 0.45
PSOLVED_MOVE_WEIGHT = 1.0
EDGE_WEIGHT = 1.0
CONN_WEIGHT = 0.3
OBS_UNSEEN_BONUS = 0.05
COMPACT_BONUS = 0.01
UNSEEN_MOVE_PENALTY = 0.15
FREE_MOVE_BONUS = 0.03
PLACE_SEEN_BONUS = 0.3             # 2-colour hint on a card the partner already saw (lock costs them nothing)
PLACE_UNSEEN_PENALTY = 0.5         # 2-colour hint on a card the partner never saw (hides it from them)
PLACE_INFO_BONUS = 0.15            # 1-colour hint on a partner-unseen card tells them its colour
PLACE_SCORE_THRESH = 0.9


def _tables():
    idx = jnp.arange(3 ** 9, dtype=jnp.int32)
    pows = 3 ** jnp.arange(9, dtype=jnp.int32)
    digits = (idx[:, None] // pows[None, :]) % 3                       # [19683, 9]
    counts = (digits[:, :, None] == jnp.arange(3)[None, None, :]).sum(1)
    valid = jnp.all(counts == 3, axis=1)
    order = jnp.argsort(jnp.logical_not(valid))
    col = digits[order[:NC]]                                            # [1680, 9]
    onehot = (col[:, :, None] == jnp.arange(3)[None, None, :]).astype(jnp.float32)
    cx = jnp.arange(81) // 9
    cy = jnp.arange(81) % 9
    cell_adj = ((jnp.abs(cx[:, None] - cx[None, :]) + jnp.abs(cy[:, None] - cy[None, :])) == 1).astype(jnp.float32)
    return col, onehot, cell_adj


def _init_pos():
    i = jnp.arange(9, dtype=jnp.int32)
    return jnp.stack([i // 3 + 3, i % 3 + 3], axis=1)


def init_agent():
    return {
        "logw": jnp.zeros((NC,), dtype=jnp.float32),
        "prev_pos": _init_pos(),
        "my_placed": jnp.zeros((4,), dtype=bool),
    }


def _hb(p):
    p = jnp.clip(p, 1e-9, 1.0 - 1e-9)
    return -(p * jnp.log(p) + (1.0 - p) * jnp.log(1.0 - p))


def agent_act(obs, reset, memory, rng):
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    COL, ONEHOT, CELL_ADJ = _tables()
    ARANGE_NC = jnp.arange(NC)
    ARANGE9 = jnp.arange(9)
    REVEAL_PREF = jnp.asarray([4.0, 3.0, 2.0, 1.0])

    nf = obs.node_features.astype(jnp.int32)
    adj = obs.adjacency_matrix.astype(jnp.float32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(bool)

    active = nf[0, 10] == 1
    substep = nf[0, 11]
    K = nf[:9, 0:3]
    known = K.sum(1) > 0
    kcol = jnp.argmax(K, axis=1)
    locked = nf[:9, 8] == 1
    pseen = nf[:9, 9] == 1
    HS = nf[9:13, 3:6] == 1                       # [4, 3]
    hplaced = nf[9:13, 8] == 1
    hrevealed = HS.any(1)
    htarget = jnp.argmax(adj[9:13, :9], axis=1)   # [4]
    A = adj[:9, :9] * (1.0 - jnp.eye(9, dtype=jnp.float32))
    cellidx = pos[:, 0] * 9 + pos[:, 1]           # [9]

    # ------------------------------------------------------------------
    # Soft evidence from partner move: partner moved a card it has seen
    # next to some card -> likely same colour.
    # ------------------------------------------------------------------
    moved = jnp.any(pos != mem["prev_pos"], axis=1)
    nmoved = moved.sum()
    m = jnp.argmax(moved)
    partner_move = (~active) & (nmoved == 1) & (~reset) & pseen[m]
    same_m = COL == COL[:, m][:, None]                       # [NC, 9]
    nb_m = (A[m] > 0)
    hit_any = jnp.any(same_m & nb_m[None, :], axis=1)        # [NC]
    hit_seen = jnp.any(same_m & (nb_m & pseen)[None, :], axis=1)
    ev = jnp.where(hit_seen, LOG_ALPHA_SEEN, jnp.where(hit_any, LOG_ALPHA_ANY, LOG_ALPHA_NONE))
    logw = mem["logw"] + jnp.where(partner_move, ev, 0.0)

    # ------------------------------------------------------------------
    # Posterior over colourings
    # ------------------------------------------------------------------
    cons = jnp.all((~known)[None, :] | (COL == kcol[None, :]), axis=1)   # [NC]
    tcol = COL[:, htarget]                                              # [NC, 4]
    in_set = HS[jnp.arange(4)[None, :], tcol]                           # [NC, 4]
    hint_lw = jnp.where(hplaced[None, :] & (~in_set) & (~mem["my_placed"][None, :]), LOG_BETA_HINT, 0.0).sum(1)
    lw = logw + hint_lw + jnp.where(cons, 0.0, -1e9)
    lw = lw - lw.max()
    w = jnp.exp(lw)
    w = w / w.sum()

    # base connectivity per colouring
    cnt_base = 0.5 * jnp.einsum('cik,ij,cjk->ck', ONEHOT, A, ONEHOT)    # [NC, 3]
    conn_base = cnt_base >= 2.0
    nconn_base = conn_base.sum(1).astype(jnp.float32)                   # [NC]
    nedge_base = jnp.minimum(cnt_base, 2.0).sum(1)                      # [NC] same-colour edges (capped 2/colour)
    solved = nconn_base >= 3.0
    p_solved = (w * solved).sum()

    # hint accounting
    p_h = (w[:, None] * in_set).sum(0)                                  # [4]
    U = (~hrevealed).sum().astype(jnp.float32)
    R = (hrevealed & ~hplaced).sum().astype(jnp.float32)
    s_now = 5.0 * U + 2.0 * R + jnp.where(hplaced, 2.0 * p_h - 1.0, 0.0).sum()

    # marginals
    pj = jnp.einsum('c,cjk->jk', w, ONEHOT)                             # [9, 3]

    # ------------------------------------------------------------------
    # Observe choice
    # ------------------------------------------------------------------
    Hc = -(pj * jnp.log(pj + 1e-12)).sum(1)
    ps_k = jnp.einsum('c,cjk->jk', w * solved, ONEHOT) / (pj + 1e-12)
    MI = _hb(p_solved) - (pj * _hb(ps_k)).sum(1)
    obs_score = Hc + 2.0 * MI + OBS_UNSEEN_BONUS * (~pseen) - 0.001 * ARANGE9
    obs_score = jnp.where(known | locked, obs_score - 1e6, obs_score)
    obs_score = jnp.where(legal[1:10], obs_score, -1e9)
    obs_action = 1 + jnp.argmax(obs_score)

    # ------------------------------------------------------------------
    # Move choice
    # ------------------------------------------------------------------
    same_deg = jnp.einsum('cik,ij,cjk->ci', ONEHOT, A, ONEHOT)          # [NC, 9]
    ADJcell = CELL_ADJ[cellidx]                                         # [9, 81]
    nb = jnp.einsum('cjk,jl->ckl', ONEHOT, ADJcell)                     # [NC, 3, 81]

    def per_card(i):
        ki = COL[:, i]
        cb = cnt_base[ARANGE_NC, ki]
        conb = conn_base[ARANGE_NC, ki].astype(jnp.float32)
        nbk = nb[ARANGE_NC, ki, :] - ADJcell[i][None, :]
        cnt_after = cb[:, None] - same_deg[:, i][:, None] + nbk
        conn_after = (cnt_after >= 2.0).astype(jnp.float32)
        nconn_after = nconn_base[:, None] - conb[:, None] + conn_after
        edges_after = nedge_base[:, None] - cb[:, None] + jnp.minimum(cnt_after, 2.0)
        E = (w[:, None] * (EDGE_WEIGHT * edges_after + CONN_WEIGHT * nconn_after)).sum(0)
        Ps = (w[:, None] * (nconn_after >= 3.0)).sum(0)
        return E, Ps

    E, Ps = jax.lax.map(per_card, ARANGE9)                              # [9, 81]
    nnb = ADJcell.sum(0)[None, :] - ADJcell                             # [9, 81]
    move_score = E + PSOLVED_MOVE_WEIGHT * Ps + COMPACT_BONUS * nnb
    move_score = move_score - UNSEEN_MOVE_PENALTY * (pseen & ~known)[:, None]
    move_score = move_score + FREE_MOVE_BONUS * (~pseen & ~known)[:, None]
    move_legal = legal[10:739].reshape(9, 81)
    move_score = jnp.where(move_legal, move_score, -1e9)
    move_flat = jnp.argmax(move_score.reshape(-1))
    move_action = 10 + move_flat

    # ------------------------------------------------------------------
    # Hint choice
    # ------------------------------------------------------------------
    p_hj = (HS.astype(jnp.float32) @ pj.T)                              # [4, 9]
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)           # [9, 9]
    far3 = (dist >= 3) & locked[None, :]
    far2 = (dist == 2) & locked[None, :]
    same_jl = COL[:, :, None] == COL[:, None, :]                        # [NC, 9, 9]
    bad3 = (w[:, None] * jnp.any(same_jl & far3[None], axis=2)).sum(0)  # [9]
    bad2 = (w[:, None] * jnp.any(same_jl & far2[None], axis=2)).sum(0)
    pconn_j = (w[:, None] * conn_base[ARANGE_NC[:, None], COL]).sum(0)  # [9]
    pnb_j = (w[:, None] * (same_deg >= 1.0)).sum(0)                      # [9] P(has same-colour neighbour)
    one_col = (HS.sum(1) == 1)[:, None]                                 # [4, 1]
    seen_term = jnp.where(one_col, PLACE_INFO_BONUS * (~pseen)[None, :],
                          PLACE_SEEN_BONUS * pseen[None, :] - PLACE_UNSEEN_PENALTY * (~pseen)[None, :])
    place_score = (p_hj - 1.0 * bad3[None, :] - 0.3 * bad2[None, :] + 0.3 * pconn_j[None, :]
                   + 0.2 * pnb_j[None, :] + seen_term + 0.05 * known[None, :])
    place_ids = 743 + jnp.arange(4)[:, None] * 81 + cellidx[None, :]    # [4, 9]
    place_mask_pub = (hrevealed & ~hplaced)[:, None] & (~locked)[None, :]
    place_mask = place_mask_pub & legal[place_ids]
    ps_pub = jnp.where(place_mask_pub, place_score, -1e9)
    best_pub = jnp.argmax(ps_pub.reshape(-1))
    best_p_pub = p_hj.reshape(-1)[best_pub]
    best_s_pub = ps_pub.reshape(-1)[best_pub]
    any_place_pub = place_mask_pub.any()
    ps_leg = jnp.where(place_mask, place_score, -1e9)
    best_leg = jnp.argmax(ps_leg.reshape(-1))
    best_p_leg = p_hj.reshape(-1)[best_leg]
    best_s_leg = ps_leg.reshape(-1)[best_leg]
    any_place = place_mask.any()
    place_action = place_ids.reshape(-1)[best_leg]
    reveal_score = jnp.where(legal[739:743], REVEAL_PREF, -1e9)
    any_reveal = legal[739:743].any()
    reveal_action = 739 + jnp.argmax(reveal_score)
    do_place = any_place & (((best_p_leg >= PLACE_THRESH) & (best_s_leg >= PLACE_SCORE_THRESH)) | (~any_reveal))
    hint_action = jnp.where(do_place, place_action, jnp.where(any_reveal, reveal_action, place_action))

    # ------------------------------------------------------------------
    # End choice (substep 0)
    # ------------------------------------------------------------------
    any_unrevealed = (~hrevealed).any()
    will_place = any_place_pub & (((best_p_pub >= PLACE_THRESH) & (best_s_pub >= PLACE_SCORE_THRESH)) | (~any_unrevealed))
    my_cost = jnp.where(will_place, 2.0 - (2.0 * best_p_pub - 1.0), 3.0)
    n_placed = hplaced.sum()
    ends_after_me = will_place & (n_placed == 3)
    partner_cost = jnp.where(ends_after_me, 0.0, 2.0)
    c_cont = my_cost + partner_cost
    occupied = jnp.zeros((81,), dtype=bool).at[cellidx].set(True)
    move_pub = (~locked)[:, None] & (~occupied)[None, :]
    max_ps = jnp.max(jnp.where(move_pub, Ps, 0.0))
    p_future = jnp.minimum(1.0, jnp.maximum(p_solved + END_GAIN, max_ps + END_GAIN_MOVE))
    end_ok = legal[1067] & (s_now > 0.0) & (
        (p_solved * s_now >= p_future * (s_now - c_cont)) & ((p_solved >= END_MIN_P) | (s_now - c_cont <= 0.0)))

    # ------------------------------------------------------------------
    # Select by substep
    # ------------------------------------------------------------------
    a0 = jnp.where(end_ok, 1067, obs_action)
    action = jnp.where(substep == 0, a0,
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(legal[action], action, jnp.argmax(legal))
    action = jnp.where(active, action, 0).astype(jnp.int32)

    is_place = active & (substep == 3) & (action >= 743) & (action < 1067)
    ph = jnp.clip((action - 743) // 81, 0, 3)
    my_placed = mem["my_placed"] | (is_place & (jnp.arange(4) == ph))

    new_mem = {"logw": logw.astype(jnp.float32), "prev_pos": pos, "my_placed": my_placed}
    trace = {
        "p_solved": p_solved.astype(jnp.float32),
        "s_now": s_now.astype(jnp.float32),
        "n_known": known.sum().astype(jnp.int32),
        "e_best": jnp.max(move_score).astype(jnp.float32),
        "pos": pos.astype(jnp.int32),
        "kcol": jnp.where(known, kcol, -1).astype(jnp.int32),
        "pseen": pseen,
        "locked": locked,
        "hs": HS,
        "hplaced": hplaced,
        "htarget": htarget.astype(jnp.int32),
    }
    return action, new_mem, trace
