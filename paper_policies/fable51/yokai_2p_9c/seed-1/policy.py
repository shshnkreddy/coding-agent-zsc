"""Belief-based cooperative Yokai policy.

Strategy (identical code on both seats, no private convention):
  * Track colours of cards this seat observed (perfect memory shows them while
    active; memory keeps them for inactive steps).
  * Maintain a belief over the 1680 colour assignments (3 cards per colour),
    weighted by hard own observations and by soft, rule-grounded partner
    evidence: (a) a hint the partner placed on a card is probably correct,
    (b) a card the partner had observed and moved was probably put next to a
    same-colour card the partner had also observed.
  * Substep 0: end the game when P(all colours connected) exceeds a
    turn-dependent threshold; otherwise observe the most uncertain card
    (preferring cards nobody has seen yet).
  * Substep 2: pick the legal move maximising expected number of connected
    colours (+ bonus for all connected) under the belief, preferring to move
    cards whose colour this seat knows.
  * Substep 3: place a revealed hint on a card whose colour is known to be in
    the hint set (preferring settled cards), otherwise reveal a hint.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_cluster_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NCARD = 9
GRID = 9
NCOL = 3
NHINT = 4
NMOVE = 729      # 9 cards * 81 cells
NPAIR = 36       # card pairs i<j
NASSIGN = 1680   # colour assignments with 3 cards per colour

# Soft-evidence parameters
EPS_HINT = 0.10     # partner placed a hint whose set excludes the card colour
EPS_MOVE = 0.40     # partner-moved card has no same-colour partner-seen neighbour
LAMBDA_ALL = 2.0    # bonus weight for all colours connected in move objective
KNOWN_MOVE_BONUS = 0.35
DIST_WEIGHT = 0.01   # penalty per unit expected same-colour pair distance (tie-break)
TWO_PLY_K = 12       # first-ply candidates kept for the two-ply lookahead
TWO_PLY_WEIGHT = 0.5
PROG_ADJ = 0.55
PROG_NEAR = 0.3


def _consts():
    """Build all constant tables with jnp ops (validator forbids module-scope calls)."""
    pi, pj = jnp.triu_indices(NCARD, k=1)
    pi = jnp.asarray(pi, dtype=jnp.int32)
    pj = jnp.asarray(pj, dtype=jnp.int32)
    a = jnp.arange(NCOL ** NCARD, dtype=jnp.int32)
    digits = (a[:, None] // (NCOL ** jnp.arange(NCARD, dtype=jnp.int32))[None, :]) % NCOL  # [19683, 9]
    counts = (digits[:, :, None] == jnp.arange(NCOL)[None, None, :]).sum(1)              # [19683, 3]
    valid = (counts == NCOL).all(1)
    sel = jnp.nonzero(valid, size=NASSIGN, fill_value=0)[0]
    assign = digits[sel]                                                                   # [1680, 9]
    ci = assign[:, pi]                                                                     # [1680, 36]
    cj = assign[:, pj]
    same = ((ci[:, :, None] == jnp.arange(NCOL)[None, None, :]) &
            (cj[:, :, None] == jnp.arange(NCOL)[None, None, :])).astype(jnp.float32)      # [1680, 36, 3]
    same_t = jnp.transpose(same, (1, 0, 2)).reshape(NPAIR, NASSIGN * NCOL)
    assign_onehot = jax.nn.one_hot(assign, NCOL, dtype=jnp.float32)                      # [1680, 9, 3]
    mv = jnp.arange(NMOVE, dtype=jnp.int32)
    mv_card = mv // (GRID * GRID)
    mv_x = (mv % (GRID * GRID)) // GRID
    mv_y = mv % GRID
    end_thresh = jnp.asarray([0.95, 0.9, 0.8, 0.75, 0.65, 0.6, 0.4, 0.5], dtype=jnp.float32)
    return pi, pj, assign, same_t, assign_onehot, mv_card, mv_x, mv_y, end_thresh


def init_agent():
    return {
        'known': -jnp.ones((NCARD,), dtype=jnp.int32),      # own observed colours
        'prev_pos': -jnp.ones((NCARD, 2), dtype=jnp.int32),  # positions at previous call
        'evid': jnp.zeros((NCARD,), dtype=jnp.int32),        # partner-move neighbour bitmask
        'mine': jnp.zeros((NHINT,), dtype=jnp.bool_),        # hints placed by this seat
        't': jnp.asarray(0, dtype=jnp.int32),
    }


def _pair_adj(pos, PI, PJ):
    """pos [..., 9, 2] -> [..., 36] float adjacency per card pair."""
    d = jnp.abs(pos[..., PI, :] - pos[..., PJ, :]).sum(-1)
    return (d == 1).astype(jnp.float32)


def _card_adj(pos):
    d = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    return d == 1  # [9, 9] bool


def _connectivity(adj_pairs, w, SAME_T, near_pairs=None):
    """adj_pairs [B, 36]; returns (exp_progress [B], p_all [B]) under weights w.

    Progress per colour: 1.0 connected, PROG_ADJ if a same-colour pair is adjacent,
    PROG_NEAR if a same-colour pair is within distance 2, else 0.
    """
    edges = adj_pairs @ SAME_T                       # [B, 1680*3]
    edges = edges.reshape(adj_pairs.shape[0], NASSIGN, NCOL)
    conn = edges >= 1.5                              # exactly 2 edges among 3 same-colour cards
    if near_pairs is None:
        prog = conn.astype(jnp.float32)
    else:
        near = (near_pairs @ SAME_T).reshape(adj_pairs.shape[0], NASSIGN, NCOL)
        prog = jnp.where(conn, 1.0, jnp.where(edges >= 0.5, PROG_ADJ, jnp.where(near >= 0.5, PROG_NEAR, 0.0)))
    nconn = conn.astype(jnp.float32).sum(-1)         # [B, 1680]
    allc = (nconn >= 2.5).astype(jnp.float32)
    wsum = w.sum() + 1e-9
    exp_p = (prog.sum(-1) * w[None, :]).sum(-1) / wsum
    p_all = (allc * w[None, :]).sum(-1) / wsum
    return exp_p, p_all


def agent_act(obs, reset, memory, rng):
    PI, PJ, ASSIGN, SAME_T, ASSIGN_ONEHOT, MV_CARD, MV_X, MV_Y, END_THRESH = _consts()
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(jnp.bool_)
    active = nf[0, 10] == 1
    substep = nf[0, 11]

    card_onehot = nf[:NCARD, 0:3]
    seen_now = card_onehot.sum(1) > 0
    obs_color = jnp.argmax(card_onehot, axis=1)
    known = jnp.where(seen_now, obs_color, mem['known'])       # [9] int, -1 unknown
    is_known = known >= 0
    locked = nf[:NCARD, 8] == 1
    pseen = nf[:NCARD, 9] == 1
    hint_set = nf[NCARD:, 3:6] > 0                              # [4, 3]
    hint_pos = nf[NCARD:, 6:8]
    hint_placed = nf[NCARD:, 8] == 1
    hint_revealed = hint_set.any(1)
    hint_size = hint_set.sum(1)
    n_actions_done = (hint_revealed.astype(jnp.int32).sum() + hint_placed.astype(jnp.int32).sum())
    k_idx = jnp.clip(n_actions_done, 0, NHINT * 2 - 1)

    cadj = _card_adj(pos)                                        # [9, 9]

    # ---- partner move evidence (detected on the inactive seat at substep 3)
    moved = (pos != mem['prev_pos']).any(1)
    partner_moved_now = (~active) & (substep == 3) & moved & pseen
    nb_mask_bits = (cadj & pseen[None, :]).astype(jnp.int32) * (2 ** jnp.arange(NCARD, dtype=jnp.int32))[None, :]
    nb_mask = nb_mask_bits.sum(1)                                # [9]
    new_evid = jnp.where(partner_moved_now & (nb_mask > 0), nb_mask, mem['evid'])

    # ---- belief over assignments
    w = jnp.ones((NASSIGN,), dtype=jnp.float32)
    # hard: own observations
    own_ok = jnp.where(is_known[None, :], ASSIGN == known[None, :], True).all(1)
    w = w * own_ok.astype(jnp.float32)
    # soft: partner-placed hints
    # trust a partner placement only on a card the partner had actually observed
    hmatch = ((pos[None, :, :] == hint_pos[:, None, :]).all(-1) & hint_placed[:, None]
              & (~mem['mine'])[:, None] & pseen[None, :])  # [4, 9]
    card_hset = (hmatch[:, :, None] & hint_set[:, None, :]).any(0)  # [9, 3]
    has_hset = card_hset.any(1)                                   # [9]
    # for each assignment, card colour in set?
    in_set = jnp.take_along_axis(jnp.broadcast_to(card_hset[None, :, :], (NASSIGN, NCARD, NCOL)),
                                 ASSIGN[:, :, None], axis=2)[:, :, 0]  # [1680, 9]
    hfac = jnp.where(has_hset[None, :] & (~in_set), EPS_HINT, 1.0).prod(1)
    w = w * hfac
    # soft: partner move evidence
    evbits = ((new_evid[:, None] // (2 ** jnp.arange(NCARD, dtype=jnp.int32))[None, :]) % 2) == 1  # [9 card, 9 nb]
    same_col = ASSIGN[:, :, None] == ASSIGN[:, None, :]           # [1680, 9, 9]
    ev_ok = (same_col & evbits[None, :, :]).any(-1)               # [1680, 9]
    has_ev = new_evid > 0
    mfac = jnp.where(has_ev[None, :] & (~ev_ok), EPS_MOVE, 1.0).prod(1)
    w = w * mfac
    # use only evidence on cards unknown to me (known ones are hard anyway)

    wsum = w.sum() + 1e-9
    marg = (ASSIGN_ONEHOT * w[:, None, None]).sum(0) / wsum       # [9, 3]
    certain = marg.max(1) >= 0.999                                # colour deducible with certainty
    cert_col = jnp.argmax(marg, axis=1)

    # ---- current board connectivity probability
    cur_adj = _pair_adj(pos, PI, PJ)[None, :]
    _, p_all_cur = _connectivity(cur_adj, w, SAME_T)
    p_all_cur = p_all_cur[0]

    # ---- substep 0/1: observation choice
    ent = -(marg * jnp.log(marg + 1e-9)).sum(1)                  # [9]
    # tie-break toward high card IDs: complements partners that scan cards from ID 0 upward
    obs_score = (ent + 0.3 * (~pseen).astype(jnp.float32) - 100.0 * certain.astype(jnp.float32)
                 + 0.001 * jnp.arange(NCARD, dtype=jnp.float32))
    obs_legal = legal[1:1 + NCARD]
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    obs_action = 1 + jnp.argmax(obs_score).astype(jnp.int32)
    end_ok = legal[1067] & (p_all_cur >= END_THRESH[k_idx])
    action0 = jnp.where(end_ok, jnp.asarray(1067, dtype=jnp.int32), obs_action)
    action1 = obs_action

    # ---- substep 2: move choice
    newpos = jnp.stack([MV_X, MV_Y], axis=-1)                     # [729, 2]
    cand = jnp.where((jnp.arange(NCARD)[None, :] == MV_CARD[:, None])[:, :, None],
                     newpos[:, None, :], pos[None, :, :])          # [729, 9, 2]
    cand_adj = _pair_adj(cand, PI, PJ)                            # [729, 36]
    # chunk over cards (81 cells each) to bound memory under episode batching
    cand_pd = jnp.abs(cand[:, PI, :] - cand[:, PJ, :]).sum(-1)                        # [729, 36] int
    # pair at distance 2 counts as 'near' only if a middle cell (common neighbour) is empty
    occ = jnp.zeros((NMOVE, GRID * GRID), dtype=jnp.bool_)
    cell_idx = cand[:, :, 0] * GRID + cand[:, :, 1]                                   # [729, 9]
    occ = occ.at[jnp.arange(NMOVE)[:, None], cell_idx].set(True)
    pa = cand[:, PI, :]
    pb = cand[:, PJ, :]
    mid1 = jnp.stack([pa[..., 0], pb[..., 1]], -1)                                    # (xa, yb)
    mid2 = jnp.stack([pb[..., 0], pa[..., 1]], -1)                                    # (xb, ya)
    mid3 = (pa + pb) // 2                                                              # straight-line middle
    def _free(m):
        ci = jnp.clip(m[..., 0], 0, GRID - 1) * GRID + jnp.clip(m[..., 1], 0, GRID - 1)
        return ~jnp.take_along_axis(occ, ci, axis=1)
    straight = (pa[..., 0] == pb[..., 0]) | (pa[..., 1] == pb[..., 1])
    mid_free = jnp.where(straight, _free(mid3), _free(mid1) | _free(mid2))
    cand_near = ((cand_pd == 2) & mid_free).astype(jnp.float32)
    def _chunk(args):
        chunk_adj, chunk_near = args
        return _connectivity(chunk_adj, w, SAME_T, chunk_near)
    exp_n, p_all = jax.lax.map(_chunk, (cand_adj.reshape(NCARD, GRID * GRID, NPAIR),
                                         cand_near.reshape(NCARD, GRID * GRID, NPAIR)))
    exp_n = exp_n.reshape(NMOVE)
    p_all = p_all.reshape(NMOVE)
    # expected total Manhattan distance between same-colour pairs (progress proxy)
    cand_dist = cand_pd.astype(jnp.float32)                                           # [729, 36]
    pair_same_prob = (SAME_T.reshape(NPAIR, NASSIGN, NCOL).sum(-1) * w[None, :]).sum(-1) / wsum  # [36]
    exp_dist = cand_dist @ pair_same_prob                                             # [729]
    mv_value = exp_n + LAMBDA_ALL * p_all - DIST_WEIGHT * exp_dist
    mv_value = mv_value + KNOWN_MOVE_BONUS * certain[MV_CARD].astype(jnp.float32)
    disp = jnp.abs(newpos - pos[MV_CARD]).sum(-1).astype(jnp.float32)
    centre = pos.astype(jnp.float32).mean(0)
    dcen = jnp.abs(newpos.astype(jnp.float32) - centre[None, :]).sum(-1)
    mv_value = mv_value - 0.002 * disp - 0.004 * dcen
    mv_legal = legal[10:10 + NMOVE]
    mv_value = jnp.where(mv_legal, mv_value, -1e9)
    # ---- restricted two-ply lookahead under the most probable assignment
    map_idx = jnp.argmax(w)
    p_map = w[map_idx] / wsum
    map_col = ASSIGN[map_idx]                                                  # [9]
    same_map_c = (jax.nn.one_hot(map_col[PI], NCOL, dtype=jnp.float32)
                  * (map_col[PI] == map_col[PJ])[:, None].astype(jnp.float32))  # [36, 3]
    K = TWO_PLY_K
    topv, topi = jax.lax.top_k(mv_value, K)
    b1 = cand[topi]                                                            # [K, 9, 2]
    sel = (jnp.arange(NCARD)[None, None, :] == MV_CARD[None, :, None])[..., None]  # [1, 729, 9, 1]
    b2 = jnp.where(sel, newpos[None, :, None, :], b1[:, None, :, :])           # [K, 729, 9, 2]
    occ1 = jnp.zeros((K, GRID * GRID), dtype=jnp.bool_)
    occ1 = occ1.at[jnp.arange(K)[:, None], b1[:, :, 0] * GRID + b1[:, :, 1]].set(True)
    dest_cell = MV_X * GRID + MV_Y
    ok2 = (~occ1[:, dest_cell]) & (~locked[MV_CARD])[None, :]                 # [K, 729]
    adj2 = _pair_adj(b2, PI, PJ)                                               # [K, 729, 36]
    conn2 = (adj2 @ same_map_c) >= 1.5                                         # [K, 729, 3]
    n2 = conn2.astype(jnp.float32).sum(-1) + LAMBDA_ALL * conn2.all(-1).astype(jnp.float32)
    n2 = jnp.where(ok2, n2, -1e9)
    conn1 = (_pair_adj(b1, PI, PJ) @ same_map_c) >= 1.5                        # [K, 3]
    n1 = conn1.astype(jnp.float32).sum(-1) + LAMBDA_ALL * conn1.all(-1).astype(jnp.float32)
    best2 = jnp.maximum(n2.max(-1), n1)                                        # [K]
    bonus = jnp.zeros((NMOVE,), dtype=jnp.float32).at[topi].add(TWO_PLY_WEIGHT * p_map * best2)
    mv_value = mv_value + bonus
    action2 = 10 + jnp.argmax(mv_value).astype(jnp.int32)

    # ---- substep 3: hint choice
    # placement candidates (h, card)
    p_in_set = (marg[None, :, :] * hint_set[:, None, :].astype(jnp.float32)).sum(-1)  # [4, 9]
    known_nb_same = ((cadj & is_known[None, :] & (known[:, None] == known[None, :])).any(1)) & is_known
    # colour group complete & connected among known cards
    same_known_pair = (known[PI] == known[PJ]) & is_known[PI]
    pair_adj_bool = _pair_adj(pos, PI, PJ) > 0.5
    col_edges = jnp.zeros((NCOL,), dtype=jnp.int32)
    col_known = jnp.zeros((NCOL,), dtype=jnp.int32)
    for c in range(NCOL):
        col_edges = col_edges.at[c].set(((known[PI] == c) & same_known_pair & pair_adj_bool).astype(jnp.int32).sum())
        col_known = col_known.at[c].set((known == c).astype(jnp.int32).sum())
    complete = (col_edges >= 2) & (col_known >= 3)               # [3]
    card_complete = jnp.where(is_known, complete[jnp.clip(known, 0, NCOL - 1)], False)
    dcard = jnp.abs(pos.astype(jnp.float32) - centre[None, :]).sum(-1)
    info = (~pseen).astype(jnp.float32)[None, :] * jnp.where(hint_size[:, None] == 1, 1.0, 0.3)
    info = info + 0.3 * pseen.astype(jnp.float32)[None, :] * (hint_size[:, None] >= 2).astype(jnp.float32)
    place_idx = 743 + jnp.arange(NHINT)[:, None] * 81 + pos[None, :, 0] * 9 + pos[None, :, 1]  # [4, 9]
    place_legal = legal[place_idx] & hint_revealed[:, None] & (~hint_placed)[:, None] & (~locked)[None, :]
    # bad lock: another locked card that may share X's colour and is more than 2 cells away
    cdist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)                       # [9, 9]
    may_same = (marg[:, None, :] * marg[None, :, :]).sum(-1) > 0.02                 # [9, 9]
    far_locked = (cdist > 1) & locked[None, :] & (jnp.arange(NCARD)[:, None] != jnp.arange(NCARD)[None, :])
    bad_lock = (far_locked & may_same).any(1)                                        # [9]
    diff_known = is_known[:, None] & is_known[None, :] & (known[:, None] != known[None, :])
    n_foreign_locked_nb = (cadj & locked[None, :] & (~may_same)).astype(jnp.float32).sum(1)
    n_foreign_nb = (cadj & diff_known).astype(jnp.float32).sum(1)
    pscore = (10.0 * p_in_set + 3.0 * card_complete[None, :].astype(jnp.float32)
              + 2.0 * known_nb_same[None, :].astype(jnp.float32) + 0.5 * info
              - 0.05 * dcard[None, :] - 4.0 * bad_lock[None, :].astype(jnp.float32)
              - 1.5 * n_foreign_locked_nb[None, :] - 0.5 * n_foreign_nb[None, :])
    pscore = jnp.where(place_legal, pscore, -1e9)
    best_place = jnp.argmax(pscore)
    best_place_score = pscore.reshape(-1)[best_place]
    best_h = best_place // NCARD
    best_c = best_place % NCARD
    place_action = place_idx.reshape(-1)[best_place].astype(jnp.int32)
    # reveal: prefer two-colour hints (1..3) before single-colour hint 0
    rev_legal = legal[739:743]
    rev_score = jnp.where(rev_legal, jnp.asarray([0.0, 3.0, 2.0, 1.0], dtype=jnp.float32), -1e9)
    reveal_action = 739 + jnp.argmax(rev_score).astype(jnp.int32)
    any_reveal = rev_legal.any()
    any_place = place_legal.any()
    best_p_in_set = p_in_set.reshape(-1)[best_place]
    best_bad_lock = bad_lock[best_c]
    place_confident = (best_p_in_set >= 0.999) & (~best_bad_lock)
    do_place = any_place & (place_confident | (~any_reveal))
    action3 = jnp.where(do_place, place_action, jnp.where(any_reveal, reveal_action, place_action))

    # ---- select action by substep; inactive seat -> no-op
    action = jnp.where(substep == 0, action0,
             jnp.where(substep == 1, action1,
             jnp.where(substep == 2, action2, action3)))
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32))
    # fallback: guarantee legality
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32)).astype(jnp.int32)

    placed_by_me = active & (substep == 3) & do_place & (action == place_action)
    new_mine = mem['mine'] | (placed_by_me & (jnp.arange(NHINT) == best_h))

    new_mem = {
        'known': known,
        'prev_pos': pos,
        'evid': new_evid,
        'mine': new_mine,
        't': mem['t'] + 1,
    }
    trace = {
        'p_all': p_all_cur.astype(jnp.float32),
        'n_known': is_known.astype(jnp.int32).sum(),
        'k': n_actions_done.astype(jnp.int32),
        'end': end_ok & active & (substep == 0),
        'pos': pos.astype(jnp.int32),
        'known': known.astype(jnp.int32),
        'locked': locked,
        'mv_val': jnp.max(mv_value).astype(jnp.float32),
    }
    return action, new_mem, trace
