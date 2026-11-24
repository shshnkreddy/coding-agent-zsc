"""Belief-based cooperative Yokai policy.

The hidden colour assignment is a uniform random arrangement of the multiset
{0,0,0,1,1,1,2,2,2} over the 9 stable card IDs, so a seat's own observations
induce an exact uniform posterior over the 1680 possible assignments.  Every
quantity this policy needs is an exact expectation under that posterior:

  * P(all three colours are connected) for the current board or a candidate move
  * E[number of connected colours] (used as the move potential)
  * P(card j has a colour inside hint set h)

Decisions:
  substep 0 : end the game when P(all connected) beats the ~4 points that two
              further turns of hint spending cost; otherwise observe.
  substep 1 : observe, preferring unknown cards, mildly preferring cards the
              partner already saw so the two seats converge on a shared view.
  substep 2 : the legal relocation maximising expected connectivity.
  substep 3 : place a revealed hint on a card that probably matches it (cheaper
              than revealing another hint), else reveal.

Nothing here assumes a private convention with the partner: all inference uses
public state plus this seat's own colour observations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'yokai_belief', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

# ---- tunables -------------------------------------------------------------
TOPK = 12            # move shortlist re-scored with exact P(all connected)
W_PALL = 8.0         # weight on P(all connected) for the chosen move
W_DIST = 0.04        # weight on expected intra-colour pair distance
PLACE_Q_MIN = 0.9    # minimum match probability before placing a hint
TRUST = 0.9          # 0.5 == ignore partner hint placements as colour evidence
BETA_MOVE = 2.0      # inverse-planning weight on the partner's relocations
BETA_REF = 3.0       # partner-exclusive card count at which beta saturates
BETA_UNSEEN = 1.0    # multiplier when the partner moved a card it has not seen
W_VOI = 2.0          # value-of-information term in the observation score
W_UNSETTLED = 3.0    # prefer observing cards with few expected same-colour neighbours
GAMMA_DIST = 0.0     # pair-distance part of the partner-move likelihood
OBS_PSEEN = -1.0      # >0 prefers observing cards the partner has already seen
# P(all connected) required to stop, indexed by hint actions already spent
END_TAU = (0.99, 0.80, 0.74, 0.68, 0.60, 0.50, 0.38, 0.22, 0.10)
FORCE_END_T = 99     # unconditionally end once this many hint actions are spent
NEG = -1.0e9


def _select(valid, k):
    """Indices of the first ``k`` true entries of ``valid`` (ascending)."""
    n = valid.shape[0]
    score = valid.astype(jnp.float32) * 1.0e6 - jnp.arange(n, dtype=jnp.float32)
    _, idx = jax.lax.top_k(score, k)
    return idx


def _tables():
    """Static combinatorial tables, built with JAX so they fold into the trace."""
    # all 1680 colourings
    digits = (jnp.arange(19683, dtype=jnp.int32)[:, None]
              // (3 ** jnp.arange(9, dtype=jnp.int32))[None, :]) % 3
    counts = jnp.sum(digits[:, :, None] == jnp.arange(3)[None, None, :], axis=1)
    col = digits[_select(jnp.all(counts == 3, axis=1), 1680)]            # [1680,9]

    # all 84 card triples and all 36 card pairs
    tcode = jnp.arange(729, dtype=jnp.int32)
    ti, tj, tk = tcode // 81, (tcode // 9) % 9, tcode % 9
    tri_sel = _select((ti < tj) & (tj < tk), 84)
    tri = jnp.stack([ti[tri_sel], tj[tri_sel], tk[tri_sel]], axis=1)     # [84,3]
    tri_code = tri[:, 0] * 81 + tri[:, 1] * 9 + tri[:, 2]

    pcode = jnp.arange(81, dtype=jnp.int32)
    pi, pj = pcode // 9, pcode % 9
    pair_sel = _select(pi < pj, 36)
    pair = jnp.stack([pi[pair_sel], pj[pair_sel]], axis=1)               # [36,2]

    # colour groups of every colouring, as triple indices
    mask = (col[:, None, :] == jnp.arange(3)[None, :, None])             # [1680,3,9]
    _, gidx = jax.lax.top_k(mask.astype(jnp.float32) * 100.0
                            - jnp.arange(9, dtype=jnp.float32)[None, None, :], 3)
    gcode = gidx[:, :, 0] * 81 + gidx[:, :, 1] * 9 + gidx[:, :, 2]       # [1680,3]
    grouptrip = jnp.argmax(gcode[:, :, None] == tri_code[None, None, :], axis=2)

    trimat = jnp.sum(grouptrip[:, :, None] == jnp.arange(84)[None, None, :],
                     axis=1).astype(jnp.float32)                          # [1680,84]
    same = (col[:, :, None] == col[:, None, :]).astype(jnp.float32).reshape(1680, 81)
    onehot = (col[:, :, None] == jnp.arange(3)[None, None, :]).astype(jnp.float32)

    # the 28 triples containing each card, given as the two other members
    notin = (pair[None, :, 0] != jnp.arange(9)[:, None]) & (
        pair[None, :, 1] != jnp.arange(9)[:, None])                       # [9,36]
    tw_sel = jax.vmap(lambda v: _select(v, 28))(notin)                    # [9,28]
    tw = pair[tw_sel]                                                     # [9,28,2]
    lo = jnp.minimum(tw[:, :, 0], jnp.arange(9)[:, None])
    hi = jnp.maximum(tw[:, :, 1], jnp.arange(9)[:, None])
    mid = (tw[:, :, 0] + tw[:, :, 1] + jnp.arange(9)[:, None]) - lo - hi
    twid = jnp.argmax((lo * 81 + mid * 9 + hi)[:, :, None] == tri_code[None, None, :],
                      axis=2)                                             # [9,28]

    cells = jnp.stack([jnp.arange(81) // 9, jnp.arange(81) % 9], axis=1).astype(jnp.int32)
    return col, tri, grouptrip, trimat, same, onehot, tw, twid, cells


def init_agent():
    return {
        'known': -jnp.ones((9,), dtype=jnp.int8),
        'logw': jnp.zeros((1680,), dtype=jnp.float32),
        'prev_pos': jnp.zeros((9, 2), dtype=jnp.int8),
        'prev_active': jnp.zeros((), dtype=jnp.int8),
    }


def _dist(pos_a, pos_b):
    return (jnp.abs(pos_a[..., :, None, 0] - pos_b[..., None, :, 0])
            + jnp.abs(pos_a[..., :, None, 1] - pos_b[..., None, :, 1]))


def agent_act(obs, reset, memory, rng):
    col, tri, grouptrip, trimat, same, onehot, tw, twid, cells = _tables()

    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    substep = nf[0, 11]
    active = nf[0, 10] > 0

    # ---- own colour knowledge -------------------------------------------
    prev = jnp.where(reset, -jnp.ones((9,), dtype=jnp.int8), memory['known'])
    seen = nf[0:9, 0:3]
    has = jnp.sum(seen, axis=1) > 0
    known = jnp.where(active, jnp.where(has, jnp.argmax(seen, axis=1).astype(jnp.int8), prev),
                      prev)
    prev_pos = jnp.where(reset, pos.astype(jnp.int8), memory['prev_pos']).astype(jnp.int32)
    prev_active = jnp.where(reset, jnp.ones((), jnp.int8), memory['prev_active'])
    logw = jnp.where(reset, jnp.zeros((1680,), jnp.float32), memory['logw'])

    # ---- public hint state ----------------------------------------------
    hset = nf[9:13, 3:6].astype(jnp.float32)                  # [4,3] multi-hot
    revealed = jnp.sum(hset, axis=1) > 0
    placed = nf[9:13, 8] > 0
    hpos = nf[9:13, 6:8]
    tgt = jnp.all(pos[None, :, :] == hpos[:, None, :], axis=2).astype(jnp.float32)  # [4,9]
    n_rev = jnp.sum(revealed.astype(jnp.int32))
    n_pla = jnp.sum(placed.astype(jnp.int32))
    turns_used = n_rev + n_pla

    def tri_conn(adj):
        e = (adj[..., tri[:, 0], tri[:, 1]] + adj[..., tri[:, 0], tri[:, 2]]
             + adj[..., tri[:, 1], tri[:, 2]])
        return (e >= 2).astype(jnp.float32)

    def nconn_per_colouring(conn):
        """Connected-colour count of every colouring, shape [...,1680]."""
        g = conn[..., grouptrip.reshape(-1)].reshape(conn.shape[:-1] + (1680, 3))
        return jnp.sum(g, axis=-1)

    d0 = _dist(pos, pos)
    a0 = (d0 == 1).astype(jnp.float32)
    conn0 = tri_conn(a0)

    # ---- inverse planning on the partner's relocation --------------------
    # The partner is assumed (only) to be competent: it relocates cards so that
    # cards it knows to share a colour end up connected, or at least closer.
    # Colourings under which its move did that gain posterior weight.
    # A partner move can only teach this seat about cards the partner has seen
    # and this seat has not, so the evidence is gated on that public count.
    exclusive = jnp.sum(((nf[0:9, 9] > 0) & (known < 0)).astype(jnp.float32))
    beta = BETA_MOVE * jnp.clip(exclusive / BETA_REF, 0.0, 1.0)
    card_moved = jnp.any(prev_pos != pos, axis=1).astype(jnp.float32)   # [9]
    seen_moved = jnp.sum(card_moved * (nf[0:9, 9] > 0).astype(jnp.float32))
    beta = beta * jnp.where(seen_moved > 0.5, 1.0, BETA_UNSEEN)
    moved = jnp.any(prev_pos != pos)
    partner_moved = moved & (prev_active == 0)
    d_prev = _dist(prev_pos, prev_pos)
    conn_prev = tri_conn((d_prev == 1).astype(jnp.float32))
    pd_now = same @ d0.reshape(81).astype(jnp.float32)        # [1680]
    pd_prev = same @ d_prev.reshape(81).astype(jnp.float32)
    gain = ((nconn_per_colouring(conn0) - nconn_per_colouring(conn_prev))
            - GAMMA_DIST * (pd_now - pd_prev))
    logw = logw + jnp.where(partner_moved, beta * gain, 0.0)

    # ---- posterior over colour assignments ------------------------------
    consistent = jnp.all((known[None, :] < 0) | (col == known[None, :]), axis=1)
    match = jnp.einsum('hj,njc,hc->hn', tgt, onehot, hset)     # placed hint consistency
    ev = jnp.where(placed[:, None], jnp.where(match > 0.5, TRUST, 1.0 - TRUST), 0.5)
    lw = jnp.where(consistent, logw + jnp.sum(jnp.log(ev / 0.5), axis=0), -jnp.inf)
    w = jnp.exp(lw - jnp.max(lw))
    wsum = jnp.maximum(jnp.sum(w), 1e-9)
    wn = w / wsum

    q_tri = wn @ trimat                                       # [84]
    qpm = (wn @ same).reshape(9, 9)
    qpm = qpm * (1.0 - jnp.eye(9, dtype=jnp.float32))
    p_col = jnp.einsum('n,njc->jc', wn, onehot)               # [9,3]

    def p_all(conn):
        return jnp.sum((nconn_per_colouring(conn) > 2.5) * wn, axis=-1)

    p_all0 = p_all(conn0)
    e_nconn0 = jnp.sum(q_tri * conn0)

    # ---- substep 0: stop or observe --------------------------------------
    q_corr = jnp.einsum('hj,jc,hc->h', tgt, p_col, hset)
    score_now = (5.0 * (4 - n_rev) + 2.0 * (n_rev - n_pla)
                 + jnp.sum(jnp.where(placed, 2.0 * q_corr - 1.0, 0.0)))
    tau = jnp.asarray(END_TAU, dtype=jnp.float32)[jnp.clip(turns_used, 0, 8)]
    want_end = ((p_all0 >= tau) | (turns_used >= FORCE_END_T)) & legal[1067]

    # value of information: how much would learning card j's colour move
    # P(all connected)?  Exact under the posterior, and cheap.
    allconn = (nconn_per_colouring(conn0) > 2.5).astype(jnp.float32)
    p_joint = jnp.einsum('n,njc->jc', wn * allconn, onehot)   # [9,3]
    p_given = p_joint / jnp.maximum(p_col, 1e-6)
    voi = jnp.sum(p_col * jnp.abs(p_given - p_all0), axis=1)  # [9]
    settled = jnp.sum(a0 * qpm, axis=1)

    unknown = (known < 0).astype(jnp.float32)
    pseen = (nf[0:9, 9] > 0).astype(jnp.float32)
    obs_score = (4.0 * unknown + OBS_PSEEN * unknown * pseen
                 + W_VOI * unknown * voi - W_UNSETTLED * unknown * settled
                 - 0.01 * jnp.arange(9, dtype=jnp.float32))
    obs_action = (1 + jnp.argmax(jnp.where(legal[1:10], obs_score, NEG))).astype(jnp.int32)

    # ---- substep 2: move --------------------------------------------------
    nd = jnp.broadcast_to(
        (jnp.abs(cells[None, :, 0, None] - pos[None, None, :, 0])
         + jnp.abs(cells[None, :, 1, None] - pos[None, None, :, 1])), (9, 81, 9))
    na = (nd == 1).astype(jnp.float32)
    na_b = jnp.take_along_axis(na, jnp.broadcast_to(tw[:, None, :, 0], (9, 81, 28)), axis=2)
    na_c = jnp.take_along_axis(na, jnp.broadcast_to(tw[:, None, :, 1], (9, 81, 28)), axis=2)
    a_bc = a0[tw[:, :, 0], tw[:, :, 1]][:, None, :]
    new_tw = ((na_b + na_c + a_bc) >= 2).astype(jnp.float32)
    q_tw = q_tri[twid]
    d_nconn = jnp.sum(q_tw[:, None, :] * (new_tw - conn0[twid][:, None, :]), axis=2)
    d_pd = jnp.sum(qpm[:, None, :] * nd, axis=2) - jnp.sum(qpm * d0, axis=1)[:, None]
    mv_score = jnp.where(legal[10:739].reshape(9, 81),
                         e_nconn0 + d_nconn - W_DIST * d_pd, NEG)

    top_val, top_idx = jax.lax.top_k(mv_score.reshape(-1), TOPK)
    sel = (jnp.arange(9)[None, :] == (top_idx // 81)[:, None])
    new_pos = jnp.where(sel[:, :, None], cells[top_idx % 81][:, None, :],
                        jnp.broadcast_to(pos[None], (TOPK, 9, 2)))
    connk = tri_conn((_dist(new_pos, new_pos) == 1).astype(jnp.float32))
    final = jnp.where(top_val > NEG / 2.0, top_val + W_PALL * p_all(connk), NEG)
    move_action = (10 + top_idx[jnp.argmax(final)]).astype(jnp.int32)

    # ---- substep 3: hints -------------------------------------------------
    unlocked = nf[0:9, 8] == 0
    avail = revealed & (~placed)
    q_match = jnp.einsum('jc,hc->hj', p_col, hset)             # [4,9]
    place_cell = pos[:, 0] * 9 + pos[:, 1]
    place_id = 743 + jnp.arange(4)[:, None] * 81 + place_cell[None, :]
    place_ok = avail[:, None] & unlocked[None, :] & legal[place_id]
    ph = jnp.where(place_ok, 4.0 * q_match + settled[None, :], NEG).reshape(-1)
    flat = jnp.argmax(ph)
    bh, bj = flat // 9, flat % 9
    best_place_ok = ph[flat] > NEG / 2.0
    place_action = (743 + bh * 81 + place_cell[bj]).astype(jnp.int32)

    rev_score = jnp.where(legal[739:743], jnp.arange(4, 0, -1).astype(jnp.float32), NEG)
    can_reveal = jnp.max(rev_score) > NEG / 2.0
    reveal_action = (739 + jnp.argmax(rev_score)).astype(jnp.int32)

    # the fourth placement terminates the episode: only take it when the board
    # is probably connected or when nothing is left to reveal.
    place_good = best_place_ok & (q_match[bh, bj] >= PLACE_Q_MIN)
    place_good = place_good & ((n_pla < 3) | (p_all0 >= 0.5) | (~can_reveal))
    hint_action = jnp.where(place_good | (~can_reveal),
                            jnp.where(best_place_ok, place_action, reveal_action),
                            reveal_action)

    act = jnp.where(substep == 0,
                    jnp.where(want_end, jnp.asarray(1067, jnp.int32), obs_action),
                    jnp.where(substep == 1, obs_action,
                              jnp.where(substep == 2, move_action, hint_action)))
    act = jnp.where(active, act, jnp.asarray(0, jnp.int32)).astype(jnp.int32)
    act = jnp.where(legal[act], act, jnp.argmax(legal).astype(jnp.int32))

    new_memory = {
        'known': known,
        'logw': logw,
        'prev_pos': pos.astype(jnp.int8),
        'prev_active': active.astype(jnp.int8),
    }
    trace = {
        'p_all': p_all0.astype(jnp.float32),
        'e_nconn': e_nconn0.astype(jnp.float32),
        'n_known': jnp.sum((known >= 0).astype(jnp.float32)),
        'n_eff': (jnp.sum(wn) ** 2) / jnp.maximum(jnp.sum(wn ** 2), 1e-9),
        'score_now': score_now.astype(jnp.float32),
        'turns_used': turns_used.astype(jnp.float32),
        'ended': (act == 1067).astype(jnp.float32),
    }
    return act.astype(jnp.int32), new_memory, trace
