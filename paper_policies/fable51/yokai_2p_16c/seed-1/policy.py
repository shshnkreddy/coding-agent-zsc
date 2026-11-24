"""Knowledge-driven cooperative Yokai policy (v2).

Local-evidence strategy (no partner convention assumed):
  * infer per-card colour possibilities from own observations, public hint
    placements (assumed correct) and the fixed colour multiset;
  * softly infer colours of partner-moved cards from the known colours of
    the cards the partner put them next to;
  * observe partner-moved cards whose colour cannot be inferred, then cards
    nobody has seen, then verify inferred cards;
  * move known cards to merge same-colour components (locked cards anchor);
  * reveal hints in a fixed order; place hints only on cards whose colour
    certainly matches, that are safe to lock, and (for ambiguous hints) that
    the partner has already seen;
  * end the game when a Monte-Carlo estimate of "all colours connected"
    beats the expected value of continuing.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'knowledge_cluster_v2', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NC = 16          # cards
NH = 7           # hints
G = 10           # grid
NCOL = 4
N_SAMPLES = 96

# --- tunables -------------------------------------------------------------
W_MERGE = 10.0        # weight of component-merge objective
W_ANCHOR = 0.5        # weight of largest/anchored component size
LOCK_WEIGHT = 3.0     # extra weight for locked cards inside a component
TRUST_BIAS = 5.0      # partner-structure bias in end-sampling
TRUST_MOVED = 1.0     # extra multiplier when the partner-seen card has been moved
INFER_BIAS = 20.0     # sampling bias for softly inferred cards
USE_INFER_MOVES = True
INFER_STRICT = False      # only infer from partner-moved known cards without a known same-colour neighbour
INFER_MAX_UNKNOWN_NB = 4  # max unknown neighbours of the partner-moved card for strict inference
W_OBS_PM = 4.0            # observation priority for partner-moved cards whose colour is not inferred
W_OBS_SENS = 4.0      # observation bonus for cards whose colour matters for connectivity
W_OBS_INFERRED = -2.0 # observation priority adjustment for softly inferred cards
W_OBS_KNOWN_NB = 0.0  # observation bonus per known neighbour (a match there is a free merge)
W_UU_SRC = 0.2        # penalty per unseen neighbour at the source of a move (may forfeit a free merge)
W_BREAK_PARTNER = 0.5 # penalty per partner-seen unknown neighbour when moving a partner-seen known card
W_COMPACT = 0.3       # bonus per extra neighbour at the destination (keeps the board blob-like)
W_LEAF = 0.5          # penalty for destinations with a single neighbour (creates cut vertices)
W_DIFF = 0.1          # penalty per known different-colour neighbour at destination
W_MISLEAD = 0.5       # extra penalty when that neighbour is unseen by the partner (they would mis-infer it)
W_PARTNER_MOVED = 1.5 # penalty for re-moving a card the partner placed (unless it merges)
W_MY_MOVED = 1.0      # penalty for re-moving a card I already placed (unless it merges)
W_PU = 0.15           # penalty per partner-seen unknown neighbour at destination
W_UU = 0.05           # penalty per unseen unknown neighbour at destination
END_GAIN = 0.2        # assumed probability gain if we wait one more round
END_COST = 4.0        # expected points lost until our next chance to end
END_GAIN_LAST = 0.1
MIN_LOCK_COMP = 2.0   # minimum same-colour component weight before locking with an ambiguous hint
AMBIG_UNSEEN_PEN = 6.0  # penalty for ambiguous hint on a card the partner has not seen
REVEAL_ORDER_LIST = (0, 1, 2, 3, 4, 5, 6)


def _consts():
    reveal_order = jnp.asarray(REVEAL_ORDER_LIST, dtype=jnp.int32)
    init_pos = jnp.stack([jnp.arange(NC) // 4 + 3, jnp.arange(NC) % 4 + 3], axis=1).astype(jnp.int32)
    nb_off = jnp.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
    return reveal_order, init_pos, nb_off


def init_agent():
    # bitmask of cards this seat has moved
    return jnp.asarray(0, dtype=jnp.int32)


# --------------------------------------------------------------------------
def _labels(A):
    """Min-label propagation on a [16,16] boolean adjacency (with self loops)."""
    lab = jnp.arange(NC, dtype=jnp.int32)

    def body(l, _):
        cand = jnp.where(A, l[None, :], 99)
        return jnp.minimum(l, jnp.min(cand, axis=1)), None

    lab, _ = jax.lax.scan(body, lab, None, length=NC)
    return lab


def _comp_stats(A, S, locked):
    """Components among subset S. Returns labels, n_comps, root weights, max weight."""
    AS = A & S[:, None] & S[None, :]
    lab = _labels(AS)
    idx = jnp.arange(NC)
    root = S & (lab == idx)
    node_w = 1.0 + LOCK_WEIGHT * locked
    root_w = jnp.sum(jnp.where(S[None, :] & (lab[None, :] == idx[:, None]), node_w[None, :], 0.0), axis=1)
    root_w = jnp.where(root, root_w, 0.0)
    n_comps = jnp.sum(root)
    max_w = jnp.max(root_w)
    return lab, n_comps, root_w, max_w


def _all_connected(A, colors):
    """True if every colour's cards form one connected component."""
    def per_color(c):
        S = colors == c
        _, n_comps, _, _ = _comp_stats(A, S, jnp.zeros(NC))
        return n_comps <= 1
    ok = jax.vmap(per_color)(jnp.arange(NCOL))
    return jnp.all(ok)


def _sample_colors(key, possible, known, color, bias):
    """Sample one full colour assignment consistent with possibilities & multiset."""
    remaining = 4.0 - jnp.sum(jax.nn.one_hot(color, NCOL) * known[:, None], axis=0)

    def body(carry, i):
        key, rem = carry
        key, sub = jax.random.split(key)
        w = possible[i] * jnp.maximum(rem, 0.0) * (1.0 + bias[i])
        w = jnp.where(jnp.sum(w) > 0, w, jnp.maximum(rem, 0.0) + 1e-3)
        s = jax.random.categorical(sub, jnp.log(w + 1e-9))
        s = jnp.where(known[i], color[i], s)
        rem = rem - jnp.where(known[i], 0.0, jax.nn.one_hot(s, NCOL))
        return (key, rem), s

    (_, _), cols = jax.lax.scan(body, (key, remaining), jnp.arange(NC))
    return cols


# --------------------------------------------------------------------------
def agent_act(obs, reset, memory, rng):
    nf = obs.node_features.astype(jnp.int32)
    adj = obs.adjacency_matrix.astype(bool)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(bool)

    REVEAL_ORDER, INIT_POS, NB_OFF = _consts()
    memory = jnp.where(reset, 0, memory).astype(jnp.int32)
    substep = nf[0, 13]
    active = nf[0, 12] > 0
    idx = jnp.arange(NC)

    own_col = nf[:NC, 0:4] > 0                     # [16,4]
    locked = nf[:NC, 10] > 0
    pseen = nf[:NC, 11] > 0
    A = adj[:NC, :NC]
    A_off = A & (idx[:, None] != idx[None, :])

    hint_set = nf[NC:, 4:8] > 0                    # [7,4]
    hint_rev = jnp.any(hint_set, axis=1)
    hint_placed = nf[NC:, 10] > 0
    hint_card_link = adj[NC:, :NC]                 # [7,16]

    # ---- colour possibilities (hard knowledge) ------------------------------
    possible = jnp.ones((NC, NCOL), dtype=bool)
    for h in range(NH):
        link = hint_card_link[h] & hint_placed[h]
        possible = jnp.where(link[:, None], possible & hint_set[h][None, :], possible)
    possible = jnp.where(jnp.any(possible, axis=1)[:, None], possible, True)
    # own observations are ground truth and override any (possibly wrong) partner hint
    possible = jnp.where(jnp.any(own_col, axis=1)[:, None], own_col, possible)

    def infer(p, _):
        single = jnp.sum(p, axis=1) == 1
        cnt_known = jnp.sum(p & single[:, None], axis=0)
        full = cnt_known >= 4
        p2 = jnp.where((~single)[:, None] & full[None, :], False, p)
        p2 = jnp.where(jnp.any(p2, axis=1)[:, None], p2, p)
        cand = jnp.sum(p2, axis=0)
        force = cand == 4
        forced_card = jnp.any(p2 & force[None, :], axis=1)
        p3 = jnp.where(forced_card[:, None], p2 & force[None, :], p2)
        p3 = jnp.where(jnp.any(p3, axis=1)[:, None], p3, p2)
        return p3, None

    possible, _ = jax.lax.scan(infer, possible, None, length=4)
    known = jnp.sum(possible, axis=1) == 1
    color = jnp.argmax(possible, axis=1).astype(jnp.int32)
    unknown = ~known
    n_known = jnp.sum(known)

    # ---- soft inference from partner structure ---------------------------------
    moved = jnp.any(pos != INIT_POS, axis=1)
    my_moved = ((memory >> idx) & 1) > 0
    partner_moved = moved & ~my_moved
    known_oh = jax.nn.one_hot(color, NCOL) * known[:, None]                      # [16,4]
    nb_known_col = jnp.sum(jnp.where(A_off[:, :, None], known_oh[None, :, :], 0.0), axis=1)  # [16,4]
    nb_known_col_pm = jnp.sum(jnp.where((A_off & partner_moved[None, :])[:, :, None], known_oh[None, :, :], 0.0), axis=1)
    n_adj_colors = jnp.sum(nb_known_col > 0, axis=1)
    adj_color = jnp.argmax(nb_known_col, axis=1).astype(jnp.int32)
    # a partner-moved known card K only supports inference about its unknown neighbours
    # when K has no known same-colour neighbour (otherwise the move merged with that one)
    same_known_off = A_off & (color[:, None] == color[None, :]) & known[:, None] & known[None, :]
    k_has_same = jnp.sum(same_known_off, axis=1) > 0
    k_n_unknown_nb = jnp.sum(A_off & unknown[None, :] & pseen[None, :], axis=1)
    k_supports = known & partner_moved & ~k_has_same & (k_n_unknown_nb <= INFER_MAX_UNKNOWN_NB)
    evidence_from_k = jnp.sum(A_off & k_supports[None, :], axis=1) > 0
    if INFER_STRICT:
        evidence = partner_moved | evidence_from_k
    else:
        evidence = partner_moved | (jnp.sum(nb_known_col_pm, axis=1) > 0)
    inferred = unknown & pseen & (n_adj_colors == 1) & evidence & possible[idx, adj_color]
    if USE_INFER_MOVES:
        s_known = known | inferred
        s_color = jnp.where(inferred, adj_color, color)
    else:
        s_known = known
        s_color = color
    s_unknown = ~s_known

    # ---- board occupancy ---------------------------------------------------
    occ = jnp.full((G, G), -1, dtype=jnp.int32).at[pos[:, 0], pos[:, 1]].set(idx)
    occ_pad = jnp.pad(occ, 1, constant_values=-1)

    def nb_of_cell(x, y):
        nx = x + 1 + NB_OFF[:, 0]
        ny = y + 1 + NB_OFF[:, 1]
        return occ_pad[nx, ny]

    same_col_mat = (s_color[:, None] == s_color[None, :]) & s_known[:, None] & s_known[None, :]
    src_same = jnp.sum(same_col_mat & A_off, axis=1)
    src_nb = jnp.sum(A_off, axis=1)
    src_pseen_nb = jnp.sum(A_off & pseen[None, :], axis=1)
    src_pu_nb = jnp.sum(A_off & pseen[None, :] & s_unknown[None, :], axis=1)
    src_uu_nb = jnp.sum(A_off & ~pseen[None, :] & unknown[None, :], axis=1)

    # ---- move scoring --------------------------------------------------------
    cells_x = jnp.repeat(jnp.arange(G), G)
    cells_y = jnp.tile(jnp.arange(G), G)
    nbs_all = jax.vmap(nb_of_cell)(cells_x, cells_y)    # [100,4]
    center_dist = jnp.abs(cells_x - 4.5) + jnp.abs(cells_y - 4.5)
    lockedf = locked.astype(jnp.float32)

    def score_card(i):
        c = s_color[i]
        S = s_known & (s_color == c)
        S_wo = S & (idx != i)
        n_c = jnp.sum(S)
        _, ncomp_b, _, maxw_b = _comp_stats(A, S, lockedf)
        F_before = (n_c - ncomp_b) + W_ANCHOR * maxw_b
        lab_w, ncomp_w, rw_w, maxw_w = _comp_stats(A, S_wo, lockedf)

        nbs = nbs_all
        valid = (nbs >= 0) & (nbs != i)
        nbs_c = jnp.where(valid, nbs, 0)
        nb_same = valid & S_wo[nbs_c]
        nb_lab = jnp.where(nb_same, lab_w[nbs_c], -1 - jnp.arange(4)[None, :])
        first = jnp.ones((G * G, 4), dtype=bool)
        for k in range(1, 4):
            for j in range(k):
                first = first.at[:, k].set(first[:, k] & (nb_lab[:, k] != nb_lab[:, j]))
        distinct_mask = nb_same & first
        merges = jnp.sum(distinct_mask, axis=1)
        merged_w = 1.0 + jnp.sum(jnp.where(distinct_mask, rw_w[jnp.where(nb_same, nb_lab, 0)], 0.0), axis=1)
        ncomp_after = ncomp_w + 1 - merges
        maxw_after = jnp.maximum(merged_w, maxw_w)
        F_after = (n_c - ncomp_after) + W_ANCHOR * maxw_after
        dF = F_after - F_before

        same_cnt = jnp.sum(nb_same, axis=1)
        nb_known = valid & s_known[nbs_c]
        diff_cnt = jnp.sum(nb_known & ~nb_same, axis=1)
        # neighbours the partner has not seen but I know differ in colour: partner would mis-infer them
        mislead_cnt = jnp.sum(nb_known & ~nb_same & ~pseen[nbs_c], axis=1)
        pu_cnt = jnp.sum(valid & s_unknown[nbs_c] & pseen[nbs_c], axis=1)
        uu_cnt = jnp.sum(valid & s_unknown[nbs_c] & ~pseen[nbs_c], axis=1)
        nb_cnt = jnp.sum(valid, axis=1)

        compact = W_COMPACT * jnp.minimum(nb_cnt - 1, 2) - W_LEAF * (nb_cnt <= 1)
        known_score = (W_MERGE * dF + (same_cnt - src_same[i]) - W_DIFF * diff_cnt
                       - W_MISLEAD * mislead_cnt
                       - W_PU * pu_cnt - W_UU * uu_cnt
                       - W_BREAK_PARTNER * pseen[i] * src_pu_nb[i]
                       - W_PARTNER_MOVED * partner_moved[i] - W_MY_MOVED * my_moved[i]
                       - W_UU_SRC * src_uu_nb[i])
        unk_score = (-1.5 - 1.5 * pseen[i] - 0.3 * src_pseen_nb[i]
                     - 0.3 * jnp.sum(nb_known, axis=1) - 0.3 * pu_cnt)
        sc = jnp.where(s_known[i], known_score, unk_score) + compact - 0.02 * center_dist
        return sc, jnp.where(s_known[i], dF, -1.0)

    move_scores, move_dF = jax.vmap(score_card)(idx)             # [16,100]
    rng, k_noise, k_samp, k_obs = jax.random.split(rng, 4)
    move_scores = move_scores + 0.001 * jax.random.uniform(k_noise, move_scores.shape)
    move_legal = legal[17:1617].reshape(NC, G * G)
    move_scores = jnp.where(move_legal, move_scores, -1e9)
    move_flat = jnp.argmax(move_scores.reshape(-1))
    move_action = 17 + move_flat
    chosen_dF = move_dF.reshape(-1)[move_flat]

    # ---- end-probability sampling (Monte-Carlo) ------------------------------------
    bias = TRUST_BIAS * nb_known_col * pseen[:, None] * (1.0 + TRUST_MOVED * moved[:, None])
    bias = bias + INFER_BIAS * jax.nn.one_hot(adj_color, NCOL) * inferred[:, None]
    keys = jax.random.split(k_samp, N_SAMPLES)
    samp = jax.vmap(lambda k: _sample_colors(k, possible.astype(jnp.float32), known, color, bias))(keys)
    conn = jax.vmap(lambda cols: _all_connected(A, cols))(samp)
    p_conn = jnp.mean(conn.astype(jnp.float32))
    onehot_s = jax.nn.one_hot(samp, NCOL)                                   # [S,16,4]
    cnt_c = jnp.sum(onehot_s, axis=0)
    cnt_conn_c = jnp.sum(onehot_s * conn[:, None, None], axis=0)
    p_cond = jnp.where(cnt_c > 0, cnt_conn_c / jnp.maximum(cnt_c, 1), p_conn)
    sens = jnp.max(p_cond, axis=1) - jnp.min(p_cond, axis=1)

    # ---- observation scoring --------------------------------------------------
    pu_nb_cnt = jnp.sum(A_off & pseen[None, :] & unknown[None, :] & (~inferred)[None, :], axis=1)
    obs_score = (-100.0 * known + W_OBS_INFERRED * inferred
                 + 3.0 * (~pseen)
                 + W_OBS_PM * (pseen & partner_moved & ~inferred)
                 + 0.3 * pu_nb_cnt
                 + W_OBS_KNOWN_NB * jnp.sum(A_off & known[None, :], axis=1)
                 + W_OBS_SENS * sens
                 + 0.001 * jax.random.uniform(k_obs, (NC,)))
    obs_legal = legal[1:17]
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    obs_action = 1 + jnp.argmax(obs_score)

    # ---- hint actions --------------------------------------------------------
    rev_legal = legal[1617:1624]
    rev_pri = jnp.where(rev_legal[REVEAL_ORDER], jnp.arange(NH), 99)
    rev_choice = REVEAL_ORDER[jnp.argmin(rev_pri)]
    reveal_action = 1617 + rev_choice
    any_reveal = jnp.any(rev_legal)

    def comp_w_of(i):
        c = color[i]
        S = known & (color == c)
        lab, _, rw, maxw = _comp_stats(A, S, lockedf)
        return rw[lab[i]], maxw
    comp_w, comp_maxw = jax.vmap(comp_w_of)(idx)
    manh = jnp.sum(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=2)
    could_same_c = possible[:, color].T                                      # [i,j] = possible[j, color_i]
    far_locked = locked[None, :] & (manh > 3) & (idx[:, None] != idx[None, :])
    conflict_known = jnp.sum(far_locked & known[None, :] & (color[None, :] == color[:, None]), axis=1)
    p_other = 1.0 / jnp.maximum(jnp.sum(possible, axis=1), 1)
    conflict_unknown = jnp.sum(jnp.where(far_locked & unknown[None, :] & could_same_c, p_other[None, :], 0.0), axis=1)

    remaining = 4.0 - jnp.sum(known_oh, axis=0)
    hint_size = jnp.sum(hint_set, axis=1)

    def place_score(h):
        hs = hint_set[h]
        match_known = hs[color]
        p_in = jnp.sum(jnp.where(hs[None, :] & possible, remaining[None, :], 0.0), axis=1) / jnp.maximum(
            jnp.sum(jnp.where(possible, remaining[None, :], 0.0), axis=1), 1e-6)
        sk = jnp.where(match_known, 2.0, -100.0)
        sk = sk - 12.0 * conflict_known - 12.0 * conflict_unknown
        sk = sk + jnp.where(comp_w < comp_maxw, -8.0, 0.5 * comp_w)
        min_comp = jnp.where(hint_size[h] == 1, 1.0, MIN_LOCK_COMP)
        sk = sk + jnp.where(comp_w < min_comp, -8.0, 0.0)
        sk = sk + jnp.where(hint_size[h] == 1, 2.0 * (~pseen), -AMBIG_UNSEEN_PEN * (~pseen))
        su = 2.0 * (2.0 * p_in - 1.0) - 4.0 - 12.0 * conflict_unknown - 3.0 * (~pseen)
        sc = jnp.where(known, sk, su)
        cell = pos[:, 0] * G + pos[:, 1]
        leg = legal[1624 + h * G * G + cell]
        return jnp.where(leg, sc, -1e9), 1624 + h * G * G + cell

    ps, pa = jax.vmap(place_score)(jnp.arange(NH))
    best_place_flat = jnp.argmax(ps.reshape(-1))
    best_place_score = ps.reshape(-1)[best_place_flat]
    best_place_action = pa.reshape(-1)[best_place_flat]
    do_place = (best_place_score > 0.0) | (~any_reveal)
    hint_action = jnp.where(do_place, best_place_action, reveal_action)

    # ---- end decision ------------------------------------------------------------
    U = jnp.sum(~hint_rev)
    R = jnp.sum(hint_rev & ~hint_placed)
    Pl = jnp.sum(hint_placed)
    S_now = 5.0 * U + 2.0 * R + 1.0 * Pl
    last_turn = (U == 0) & (R == 1)
    gain = jnp.where(last_turn, END_GAIN_LAST, END_GAIN)
    cost = jnp.where(last_turn, 1.0, END_COST)
    ev_end = p_conn * S_now
    ev_cont = jnp.minimum(1.0, p_conn + gain) * (S_now - cost)
    want_end = (ev_end >= ev_cont) & (p_conn > 0.0) & legal[2324]
    end_action = jnp.asarray(2324, dtype=jnp.int32)

    # ---- select by substep ------------------------------------------------------
    action = jnp.where(substep == 0, jnp.where(want_end, end_action, obs_action),
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = action.astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    action = jnp.where(active, action, 0).astype(jnp.int32)

    is_move = active & (action >= 17) & (action < 1617)
    moved_card = jnp.where(is_move, (action - 17) // (G * G), 0)
    new_mem = jnp.where(is_move, memory | (1 << moved_card), memory).astype(jnp.int32)
    trace = {
        "p_conn": p_conn,
        "n_known": n_known.astype(jnp.int32),
        "n_inferred": jnp.sum(inferred).astype(jnp.int32),
        "want_end": want_end,
        "best_place_score": best_place_score,
        "substep": substep,
        "move_dF": chosen_dF,
    }
    return action, new_mem, trace
