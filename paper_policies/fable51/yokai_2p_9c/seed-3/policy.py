"""Belief-based Yokai policy.

Each seat keeps a private record of the card colors it has observed itself,
enumerates every color assignment consistent with that record (uniform prior
over the 1680 permutations of the color multiset), and uses the posterior to
(a) decide when to end the game, (b) pick the move that maximises the
posterior probability that every color is connected, (c) choose which cards to
observe, and (d) reveal hints or place them on cards whose known color matches.
No shared convention is assumed. The partner's public actions are used only as
soft evidence: a card the partner has observed and then moved next to other
cards probably shares their color, and a hint the partner placed probably
matches its card. Both are generic "the partner is trying to cluster" priors,
not a protocol.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_yokai_v2', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NC = 9          # cards
NCOL = 3        # colors
NH = 4          # hints
G = 9           # grid size
NCELL = G * G
NA = 1680       # number of color assignments (9! / 3!^3)
NP = 36         # card pairs

# ----------------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------------
MOVE_EV_SEEN = 2.5          # log-odds per new adjacency when partner had observed the moved card
MOVE_EV_UNSEEN = 0.3        # ... when the partner had not observed the moved card
NEIGHBOR_UNSEEN_MULT = 0.5  # multiplier when the partner had not observed the neighbour
HINT_EV = 2.5               # log-odds that a partner-placed hint matches its card
END_THRESHOLD = 0.7         # end if P(all connected) >= this ...
END_MARGIN_COST = 4.0       # ... and p*S >= (S - cost) * min(1, p + END_DELTA)
END_DELTA = 0.25
OBS_PARTNER_SEEN_BONUS = -0.05
MOVE_UNKNOWN_PENALTY = 0.0  # subtract from move score when moving a card of unknown colour
MOVE_CNT_WEIGHT = 0.1       # weight of expected number of connected colours
MOVE_PAIR_WEIGHT = 0.0      # weight of expected number of same-colour adjacent pairs
PLACE_MIN_P = 0.99          # place a hint (instead of revealing) only if P(correct) >= this while reveals remain
PLACE_UNSEEN_BONUS = -0.05  # locked cards can never be observed: prefer placing hints on cards the partner has already seen
MOVE_PARTNER_KNOWN_PENALTY = 0.05  # avoid moving cards the partner knows but I do not
MOVE_PARTNER_MOVED_PENALTY = 0.03  # avoid undoing cards the partner has deliberately moved
NEG_EV = 1.0                # log-odds against same colour when partner leaves two freshly observed cards apart
VIOL_EV = 2.0              # log-odds against hypotheses where partner-seen same-colour cards stay disconnected after a partner move
OBS_MI_WEIGHT = 1.0         # weight of mutual information between a card's colour and current connectivity
FORCE_END_TURN = 0          # diagnostic only: end unconditionally at this global turn (0 = off)
OBS_ENT_WEIGHT = 1.0        # weight of marginal colour entropy


def _tables():
    """Static lookup tables built from constants with JAX ops (no host libs)."""
    idx = jnp.arange(3 ** NC)
    digits = (idx[:, None] // (3 ** jnp.arange(NC))[None, :]) % 3            # [19683, 9]
    counts = (digits[:, :, None] == jnp.arange(NCOL)[None, None, :]).sum(axis=1)
    valid = jnp.all(counts == 3, axis=1)
    order = jnp.argsort(jnp.where(valid, 0, 1), stable=True)
    A = digits[order[:NA]].astype(jnp.int32)                                   # [1680, 9]
    pj, pk = jnp.triu_indices(NC, k=1)                                         # [36]
    pj = pj.astype(jnp.int32)
    pk = pk.astype(jnp.int32)
    same36 = (A[:, pj] == A[:, pk]).astype(jnp.float32)                        # [1680, 36]
    colpair = ((A[:, None, pj] == jnp.arange(NCOL)[None, :, None]) &
               (A[:, None, pk] == jnp.arange(NCOL)[None, :, None])).astype(jnp.float32)  # [1680,3,36]
    same_mat = (A[:, :, None] == A[:, None, :]) & (~jnp.eye(NC, dtype=bool))[None]        # [1680,9,9]
    onehot_mate = (same_mat[:, :, pj] & same_mat[:, :, pk]).astype(jnp.float32)          # [1680,9,36]
    a_onehot = (A[:, :, None] == jnp.arange(NCOL)[None, None, :]).astype(jnp.float32)     # [1680,9,3]
    cells = jnp.arange(NCELL)
    cell_xy = jnp.stack([cells // G, cells % G], axis=1).astype(jnp.int32)              # [81,2]
    return A, colpair, onehot_mate, same36, pj, pk, a_onehot, cell_xy


def init_agent():
    return {
        'known': jnp.full((NC,), -1, dtype=jnp.int32),
        'prev_pos': jnp.full((NC, 2), -1, dtype=jnp.int32),
        'prev_placed': jnp.zeros((NH,), dtype=jnp.int32),
        'evidence': jnp.zeros((NP,), dtype=jnp.float32),
        'hint_ev': jnp.zeros((NC, NCOL), dtype=jnp.float32),
        'turn_start_ps': jnp.zeros((NC,), dtype=jnp.int32),
        'turn': jnp.zeros((), dtype=jnp.int32),
        'logw_extra': jnp.zeros((NA,), dtype=jnp.float32),
        'partner_moved_any': jnp.zeros((NC,), dtype=jnp.int32),
    }


def _pair_adjacency(pos, pj, pk):
    """[36] float: 1 if the two cards of the pair are orthogonally adjacent."""
    d = jnp.abs(pos[pj] - pos[pk]).sum(axis=1)
    return (d == 1).astype(jnp.float32)


def agent_act(obs, reset, memory, rng):
    (ASSIGN, COLPAIR, ONEHOT_MATE, SAME36, PAIR_J, PAIR_K, ASSIGN_ONEHOT, CELL_XY) = _tables()
    init = init_agent()
    known = jnp.where(reset, init['known'], memory['known'])
    prev_pos = jnp.where(reset, init['prev_pos'], memory['prev_pos'])
    prev_placed = jnp.where(reset, init['prev_placed'], memory['prev_placed'])
    evidence = jnp.where(reset, init['evidence'], memory['evidence'])
    hint_ev = jnp.where(reset, init['hint_ev'], memory['hint_ev'])
    turn_start_ps = jnp.where(reset, init['turn_start_ps'], memory['turn_start_ps'])
    turn = jnp.where(reset, init['turn'], memory['turn'])
    logw_extra = jnp.where(reset, init['logw_extra'], memory['logw_extra'])
    partner_moved_any = jnp.where(reset, init['partner_moved_any'], memory['partner_moved_any'])

    nf = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    pos = obs.positions.astype(jnp.int32)          # [9,2]
    active = nf[0, 10] == 1
    substep = nf[0, 11]
    turn = turn + (substep == 0).astype(jnp.int32)                 # global 1-based turn number

    # --- update known colors from own observation features ---------------
    col_feat = nf[:NC, 0:3]
    seen = col_feat.sum(axis=1) > 0
    seen_col = jnp.argmax(col_feat, axis=1).astype(jnp.int32)
    known = jnp.where(seen, seen_col, known)
    known_mask = known >= 0
    locked = nf[:NC, 8] == 1
    partner_seen = nf[:NC, 9] == 1
    ps_f = partner_seen.astype(jnp.float32)

    # --- hint state --------------------------------------------------------
    hint_sets = nf[NC:NC + NH, 3:6]                                  # [4,3]
    hint_f = hint_sets.astype(jnp.float32)
    hint_revealed = hint_sets.sum(axis=1) > 0
    hint_placed = nf[NC:NC + NH, 8] == 1
    hint_pos = nf[NC:NC + NH, 6:8]
    hint_on_card = (jnp.abs(pos[None, :, :] - hint_pos[:, None, :]).sum(axis=2) == 0) & hint_placed[:, None]  # [4,9]

    # --- detect partner moves and accumulate adjacency evidence ----------
    have_prev = prev_pos[0, 0] >= 0
    moved = (jnp.abs(pos - prev_pos).sum(axis=1) > 0) & have_prev
    partner_moved = moved & (~active)
    pair_adj = _pair_adjacency(pos, PAIR_J, PAIR_K)                  # [36]
    mover_w = jnp.where(partner_seen, MOVE_EV_SEEN, MOVE_EV_UNSEEN)  # [9]
    nb_w = jnp.where(partner_seen, 1.0, NEIGHBOR_UNSEEN_MULT)        # [9]
    pm_f = partner_moved.astype(jnp.float32)
    partner_moved_any = jnp.maximum(partner_moved_any, partner_moved.astype(jnp.int32))
    ev_pair = pair_adj * (pm_f[PAIR_J] * mover_w[PAIR_J] * nb_w[PAIR_K] +
                          pm_f[PAIR_K] * mover_w[PAIR_K] * nb_w[PAIR_J])
    evidence = evidence + ev_pair
    # negative evidence: partner observed both cards of a pair this turn, moved
    # a card, yet left the pair non-adjacent -> probably different colours
    turn_start_ps = jnp.where(substep == 0, partner_seen.astype(jnp.int32), turn_start_ps)
    fresh = partner_seen & (turn_start_ps == 0)
    fresh_f = fresh.astype(jnp.float32)
    any_partner_move = jnp.any(partner_moved) & (substep == 3)
    neg_pair = (1.0 - pair_adj) * fresh_f[PAIR_J] * fresh_f[PAIR_K] * any_partner_move.astype(jnp.float32)
    evidence = evidence - NEG_EV * neg_pair

    # hypothesis-level evidence: under assignment a, the partner knows the colours
    # of partner-seen cards; if two or more partner-seen cards of a colour are not
    # connected after the partner moved, that assignment is less likely.
    n_seen_c = jnp.einsum('aic,i->ac', ASSIGN_ONEHOT, ps_f)          # [NA,3]
    seen_pair_adj = pair_adj * ps_f[PAIR_J] * ps_f[PAIR_K]           # [36]
    cnt_seen_adj = COLPAIR @ seen_pair_adj                           # [NA,3]
    viol = ((n_seen_c >= 2) & (cnt_seen_adj < n_seen_c - 1)).astype(jnp.float32).sum(axis=1)
    logw_extra = logw_extra - VIOL_EV * any_partner_move.astype(jnp.float32) * viol

    # --- partner hint placements: card colour probably in hint set --------
    newly_placed = hint_placed & (prev_placed == 0) & (~active) & have_prev
    new_ev = jnp.einsum('h,hi,hc->ic', newly_placed.astype(jnp.float32), hint_on_card.astype(jnp.float32), hint_f)
    hint_ev = hint_ev + HINT_EV * new_ev

    # --- posterior over assignments ---------------------------------------
    consistent = jnp.all((ASSIGN == known[None, :]) | (~known_mask)[None, :], axis=1)
    logw = SAME36 @ evidence + jnp.einsum('aic,ic->a', ASSIGN_ONEHOT, hint_ev) + logw_extra
    logw = jnp.where(consistent, logw, -1e9)
    w = jnp.exp(logw - jnp.max(logw))
    w = w / jnp.sum(w)                                               # [NA]

    marg = jnp.einsum('a,aic->ic', w, ASSIGN_ONEHOT)                 # [9,3]

    # connectivity per assignment and color at current positions
    conn = (COLPAIR @ pair_adj) >= 2                                 # [NA,3] bool
    conn_f = conn.astype(jnp.float32)
    all_conn = jnp.all(conn, axis=1).astype(jnp.float32)
    p_conn_now = jnp.sum(w * all_conn)

    # --- move evaluation ---------------------------------------------------
    # For card i with color-mates pair p = {j,k}: after moving i to cell c,
    # color(i) connected iff adj(j,k) + adj(c,j) + adj(c,k) >= 2.
    n_conn = conn_f.sum(axis=1)                                      # [NA]
    own_conn = jnp.take_along_axis(conn_f, ASSIGN, axis=1)           # [NA,9]
    others_n = n_conn[:, None] - own_conn                            # [NA,9]
    others_all = (others_n >= 2).astype(jnp.float32)
    W_all = jnp.einsum('a,ai,aip->ip', w, others_all, ONEHOT_MATE)   # [9,36]
    W_cnt = jnp.einsum('a,ai,aip->ip', w, others_n, ONEHOT_MATE)     # [9,36]
    W_one = jnp.einsum('a,aip->ip', w, ONEHOT_MATE)                  # [9,36]
    cell_adj = (jnp.abs(CELL_XY[:, None, :] - pos[None, :, :]).sum(axis=2) == 1).astype(jnp.float32)  # [81,9]
    cnt = cell_adj[:, PAIR_J] + cell_adj[:, PAIR_K]                  # [81,36]
    cond = (pair_adj[None, :] + cnt >= 2).astype(jnp.float32)        # [81,36]
    p_all_after = W_all @ cond.T                                     # [9,81]
    e_cnt_after = W_cnt.sum(axis=1)[:, None] + W_one @ cond.T        # [9,81]
    p_same_pair = SAME36.T @ w                                       # [36] P(pair same colour)
    same_mat9 = jnp.zeros((NC, NC), jnp.float32).at[PAIR_J, PAIR_K].set(p_same_pair)
    same_mat9 = same_mat9 + same_mat9.T                              # [9,9]
    adj9 = jnp.zeros((NC, NC), jnp.float32).at[PAIR_J, PAIR_K].set(pair_adj)
    adj9 = adj9 + adj9.T
    tot_pairs = jnp.sum(adj9 * same_mat9) / 2.0
    pairs_lost = jnp.sum(adj9 * same_mat9, axis=1)                   # [9] pairs involving i now
    pairs_gain = cell_adj @ same_mat9.T                              # [81,9] -> [cell, i]
    e_pairs_after = tot_pairs - pairs_lost[:, None] + pairs_gain.T   # [9,81]
    move_score = p_all_after + MOVE_CNT_WEIGHT * e_cnt_after + MOVE_PAIR_WEIGHT * e_pairs_after
    move_score = move_score - MOVE_UNKNOWN_PENALTY * (~known_mask).astype(jnp.float32)[:, None]
    move_score = move_score - (MOVE_PARTNER_KNOWN_PENALTY * (partner_seen & ~known_mask).astype(jnp.float32)
                               + MOVE_PARTNER_MOVED_PENALTY * (partner_moved_any > 0).astype(jnp.float32))[:, None]
    move_legal = legal[10:739].reshape(NC, NCELL)
    move_score = jnp.where(move_legal, move_score, -1e9)
    move_flat = jnp.argmax(move_score.reshape(-1))
    move_action = 10 + move_flat

    # --- observation choice -----------------------------------------------
    ent = -jnp.sum(marg * jnp.log(marg + 1e-9), axis=1)              # [9]
    # mutual information between card colour and "all connected now"
    joint = jnp.einsum('a,aic->ic', w * all_conn, ASSIGN_ONEHOT)     # P(col_i=c, conn)
    p_c_given = joint / (marg + 1e-9)                                # P(conn | col_i=c)
    def _hb(p):
        p = jnp.clip(p, 1e-6, 1 - 1e-6)
        return -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))
    mi = _hb(p_conn_now) - jnp.sum(marg * _hb(p_c_given), axis=1)    # [9]
    obs_score = (OBS_ENT_WEIGHT * ent + OBS_MI_WEIGHT * mi
                 + OBS_PARTNER_SEEN_BONUS * ps_f + 1e-3 * jnp.arange(NC)[::-1])
    obs_score = jnp.where(known_mask, obs_score - 10.0, obs_score)
    obs_legal = legal[1:10]
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    obs_action = 1 + jnp.argmax(obs_score)

    # --- hint decision at substep 3 -----------------------------------------
    p_match = hint_f @ marg.T                                        # [4,9]
    placed_card = jnp.argmax(hint_on_card.astype(jnp.int32), axis=1)  # [4]
    placed_match = p_match[jnp.arange(NH), placed_card]
    placed_val = jnp.where(hint_placed, 2.0 * placed_match - 1.0, 0.0)
    u_cnt = jnp.sum(~hint_revealed)
    r_cnt = jnp.sum(hint_revealed & ~hint_placed)
    score_now = 5.0 * u_cnt + 2.0 * r_cnt + jnp.sum(placed_val)

    # placement candidates: revealed, unplaced hints on unlocked cards
    place_ok = hint_revealed & ~hint_placed
    # prefer cards that already sit next to a probable colour-mate (less likely to need moving)
    same_adj = pairs_lost
    place_score = (p_match + 0.02 * same_adj[None, :] + PLACE_UNSEEN_BONUS * (1.0 - ps_f)[None, :]
                   + 0.001 * jnp.arange(NC)[None, :])
    place_score = jnp.where(place_ok[:, None] & (~locked)[None, :], place_score, -1e9)
    best_place_flat = jnp.argmax(place_score.reshape(-1))
    best_place_h = best_place_flat // NC
    best_place_card = best_place_flat % NC
    best_place_p = p_match.reshape(-1)[best_place_flat]
    place_possible = place_score.reshape(-1)[best_place_flat] > -1.0
    place_cell = pos[best_place_card, 0] * G + pos[best_place_card, 1]
    place_action = 743 + best_place_h * NCELL + place_cell

    # reveal candidates: two-colour hints first (easier to place correctly)
    rev_rank = jnp.asarray([3.0, 0.0, 1.0, 2.0])
    rev_score = jnp.where(~hint_revealed, -rev_rank, -1e9)
    rev_h = jnp.argmax(rev_score)
    reveal_action = 739 + rev_h
    any_unrevealed = jnp.any(~hint_revealed)

    use_place = place_possible & ((best_place_p >= PLACE_MIN_P) | (~any_unrevealed))
    hint_action = jnp.where(use_place, place_action, reveal_action)
    hint_action = jnp.where(legal[hint_action], hint_action,
                            jnp.argmax(jnp.where(legal, 1, 0) * (jnp.arange(1068) >= 739)))

    # --- end decision at substep 0 ------------------------------------------
    end_legal = legal[1067]
    continue_value = (score_now - END_MARGIN_COST) * jnp.minimum(1.0, p_conn_now + END_DELTA)
    do_end = end_legal & (p_conn_now >= END_THRESHOLD) & (p_conn_now * score_now >= continue_value)
    do_end = do_end | (end_legal & (turn == FORCE_END_TURN))
    s0_action = jnp.where(do_end, 1067, obs_action)

    action = jnp.where(substep == 0, s0_action,
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, 0)
    action = jnp.where(legal[action], action, jnp.argmax(legal))
    action = action.astype(jnp.int32)

    new_memory = {
        'known': known.astype(jnp.int32),
        'prev_pos': pos.astype(jnp.int32),
        'prev_placed': hint_placed.astype(jnp.int32),
        'evidence': evidence.astype(jnp.float32),
        'hint_ev': hint_ev.astype(jnp.float32),
        'turn_start_ps': turn_start_ps.astype(jnp.int32),
        'turn': turn.astype(jnp.int32),
        'logw_extra': logw_extra.astype(jnp.float32),
        'partner_moved_any': partner_moved_any.astype(jnp.int32),
    }
    trace = {
        'p_conn': p_conn_now.astype(jnp.float32),
        'score_now': score_now.astype(jnp.float32),
        'n_known': known_mask.sum().astype(jnp.int32),
        'substep': substep.astype(jnp.int32),
        'turn': turn.astype(jnp.int32),
        'active': active,
    }
    return action, new_memory, trace
