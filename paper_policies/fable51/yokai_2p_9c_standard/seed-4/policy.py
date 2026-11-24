"""Belief-based cooperative Yokai policy.

Maintains an exact posterior over all 1680 colour assignments using private
observations, publicly placed hints (soft constraints) and the fixed colour
multiset.  Moves greedily maximise expected same-colour adjacency / solved
probability, hints are placed only on near-certain cards, and the game is
ended when the posterior probability that every colour is connected exceeds a
score-dependent threshold.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_greedy_v4', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
GRID = 9
NUM_HINTS = 4
NUM_ASSIGN = 1680
NUM_PAIRS = 36
NUM_CAND = 324
A_OBS = 1
A_MOVE = 10
A_REVEAL = 739
A_PLACE = 743
A_END = 1067

# ---- tunables -------------------------------------------------------------
HINT_EPS = 0.05        # likelihood of an assignment violating a placed hint
PARTNER_LAMBDA = 2.0   # likelihood ratio: partner moved a card it observed next to a same-colour card
PARTNER_LAMBDA_STRONG = 5.0   # ... and the partner has observed that neighbour too
COMM_BONUS = 0.1       # placement preference: single hint on partner-unseen card, pair hint on partner-seen card
W_ENT = 1.0            # entropy weight in observation choice (VOI is in potential units)
W_LOCK_DIST = 2.0      # extra distance weight for pairs involving a locked card
OBS_UNSEEN_BONUS = 0.3
PLACE_MIN_P = 0.9      # minimum P(correct) to voluntarily place a hint
END_TAU_LO = 0.5
END_TAU_HI = 0.9
END_Q = 0.97          # success discount applied to the value of continuing
K_CAL = 2.0           # calibration exponent: p_true ~ 1-(1-p)^K (partner rationality)
W_SOLVED = 100.0
W_EDGES = 10.0
W_DIST = 0.5
W_COMPACT = 0.05
W_MISLEAD = 5.0        # penalty for moves that falsely signal same-colour adjacency


def _tables():
    """Build all constant lookup tables with jnp only (constant-folded under jit)."""
    n = 3 ** NUM_CARDS
    codes = jnp.arange(n, dtype=jnp.int32)
    digits = (codes[:, None] // (3 ** jnp.arange(NUM_CARDS, dtype=jnp.int32))[None, :]) % 3   # [19683, 9]
    counts = (digits[:, :, None] == jnp.arange(NUM_COLORS, dtype=jnp.int32)[None, None, :]).sum(1)
    ok = (counts == 3).all(1)
    (sel,) = jnp.nonzero(ok, size=NUM_ASSIGN, fill_value=0)
    assign = digits[sel]                                                     # [1680, 9]
    pi, pj = jnp.triu_indices(NUM_CARDS, 1)
    pi = jnp.asarray(pi, dtype=jnp.int32)
    pj = jnp.asarray(pj, dtype=jnp.int32)
    same = (assign[:, pi] == assign[:, pj]).astype(jnp.float32)             # [1680, 36]
    onehot = jax.nn.one_hot(assign, NUM_COLORS, dtype=jnp.float32)          # [1680, 9, 3]
    pair_index = jnp.zeros((NUM_CARDS, NUM_CARDS), dtype=jnp.int32)
    pair_index = pair_index.at[pi, pj].set(jnp.arange(NUM_PAIRS, dtype=jnp.int32))
    pair_index = pair_index.at[pj, pi].set(jnp.arange(NUM_PAIRS, dtype=jnp.int32))
    cand = jnp.arange(NUM_CAND, dtype=jnp.int32)
    cand_i = cand // (NUM_CARDS * 4)
    cand_j = (cand // 4) % NUM_CARDS
    cand_d = cand % 4
    dirs = jnp.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
    card_idx = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    hint_idx = jnp.arange(NUM_HINTS, dtype=jnp.int32)
    init_pos = jnp.stack([card_idx // 3 + 3, card_idx % 3 + 3], axis=1).astype(jnp.int32)
    return dict(assign=assign, pi=pi, pj=pj, same=same, onehot=onehot, pair_index=pair_index,
                cand_i=cand_i, cand_j=cand_j, cand_d=cand_d, dirs=dirs, card_idx=card_idx,
                hint_idx=hint_idx, init_pos=init_pos)


def init_agent():
    card_idx = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    init_pos = jnp.stack([card_idx // 3 + 3, card_idx % 3 + 3], axis=1).astype(jnp.int32)
    return {
        'known': -jnp.ones((NUM_CARDS,), dtype=jnp.int32),
        'logw': jnp.zeros((NUM_ASSIGN,), dtype=jnp.float32),
        'prev_pos': init_pos,
        'pseen': jnp.zeros((NUM_CARDS,), dtype=jnp.int32),
    }


def _pair_adj(pos, pi, pj):
    d = jnp.abs(pos[..., pi, :] - pos[..., pj, :]).sum(-1)
    return (d == 1).astype(jnp.float32)


def _pair_dist(pos, pi, pj):
    return jnp.abs(pos[..., pi, :] - pos[..., pj, :]).sum(-1).astype(jnp.float32)


def _belief(known, hint_factor, logw, t):
    own = jnp.where(known[:, None] >= 0, jax.nn.one_hot(jnp.maximum(known, 0), NUM_COLORS), 1.0)
    allowed = own * hint_factor                                   # [9, 3]
    lik = jnp.prod(allowed[t['card_idx'][None, :], t['assign']], axis=1)    # [1680]
    w = lik * jnp.exp(logw - jnp.max(logw))
    total = w.sum()
    return jnp.where(total > 0, w / jnp.maximum(total, 1e-12), 1.0 / NUM_ASSIGN)


def agent_act(obs, reset, memory, rng):
    t = _tables()
    init = init_agent()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    f32 = jnp.float32
    SAME = t['same']
    ONEHOT = t['onehot']
    PI = t['pi']
    PJ = t['pj']
    CARD_IDX = t['card_idx']
    HINT_IDX = t['hint_idx']
    CAND_I = t['cand_i']
    CAND_J = t['cand_j']
    CAND_D = t['cand_d']
    DIRS = t['dirs']

    nf = obs.node_features.astype(jnp.int32)
    cards = nf[:NUM_CARDS]
    hints = nf[NUM_CARDS:NUM_CARDS + NUM_HINTS]
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(bool)
    active = cards[0, 10] == 1
    substep = cards[0, 11]

    col_oh = cards[:, 0:3]
    obs_now = col_oh.sum(1) > 0
    known = jnp.where(obs_now, jnp.argmax(col_oh, axis=1), memory['known'])
    locked = cards[:, 8] == 1
    pseen = jnp.maximum(memory['pseen'], cards[:, 9])

    hset = hints[:, 3:6].astype(f32)                              # [4, 3]
    revealed = hset.sum(1) > 0
    placed = hints[:, 8] == 1
    hpos = hints[:, 6:8]
    match = (hpos[:, None, :] == pos[None, :, :]).all(-1) & placed[:, None]   # [4, 9]
    viol = match[:, :, None] & (hset[:, None, :] < 0.5)                        # [4, 9, 3]
    hint_factor = jnp.prod(jnp.where(viol, HINT_EPS, 1.0), axis=0)             # [9, 3]

    # ---- partner-move evidence ---------------------------------------------
    adj_now = _pair_adj(pos, PI, PJ)                               # [36]
    moved = (pos != memory['prev_pos']).any(-1)
    partner_moved = (~active) & moved.any()
    mc = jnp.argmax(moved)
    inv_mc = ((PI == mc) | (PJ == mc)).astype(f32)
    nb_pairs = adj_now * inv_mc
    pseen_f = pseen.astype(f32)
    nb_seen_pairs = nb_pairs * pseen_f[PI] * pseen_f[PJ]           # partner has observed both endpoints
    same_strong = jnp.clip(SAME @ nb_seen_pairs, 0.0, 1.0)         # [1680]
    same_weak = jnp.clip(SAME @ nb_pairs, 0.0, 1.0) * pseen_f[mc]
    log_factor = jnp.where(same_strong > 0, jnp.log(PARTNER_LAMBDA_STRONG),
                           jnp.where(same_weak > 0, jnp.log(PARTNER_LAMBDA), 0.0))
    logw = memory['logw'] + jnp.where(partner_moved, log_factor, 0.0)

    # ---- posterior ----------------------------------------------------------
    w = _belief(known, hint_factor, logw, t)
    edges_now = SAME @ adj_now                                     # [1680]
    solved_now = (edges_now >= 6.0).astype(f32)
    p_now = w @ solved_now
    marg = jnp.einsum('a,ack->ck', w, ONEHOT)                      # [9, 3]
    p_same = w @ SAME                                              # [36]
    psame_mat = p_same[t['pair_index']]                            # [9, 9]

    n_unrev = (~revealed).sum()
    n_rev_unplaced = (revealed & ~placed).sum()
    p_corr = jnp.einsum('hk,ck->hc', hset, marg)                   # [4, 9]
    p_corr_placed = (match.astype(f32) * p_corr).sum(1)            # [4]
    score_now = 5.0 * n_unrev + 2.0 * n_rev_unplaced + (placed.astype(f32) * (2.0 * p_corr_placed - 1.0)).sum()

    k_obs, k_move, k_hint = jax.random.split(rng, 3)

    # ---- move choice --------------------------------------------------------
    dest = pos[CAND_J] + DIRS[CAND_D]                              # [324, 2]
    inb = (dest >= 0).all(1) & (dest < GRID).all(1)
    dclip = jnp.clip(dest, 0, GRID - 1)
    move_ids = A_MOVE + CAND_I * 81 + dclip[:, 0] * 9 + dclip[:, 1]
    valid = inb & (CAND_I != CAND_J) & legal[move_ids]
    occupied = (dclip[:, None, :] == pos[None, :, :]).all(-1).any(1)
    valid_struct = inb & (CAND_I != CAND_J) & ~occupied & ~locked[CAND_I]
    is_mover = (CARD_IDX[None, :] == CAND_I[:, None])[:, :, None]
    newpos = jnp.where(is_mover, dclip[:, None, :], pos[None, :, :])   # [324, 9, 2]
    adj_c = _pair_adj(newpos, PI, PJ)                              # [324, 36]
    dist_c = _pair_dist(newpos, PI, PJ)
    edges_c = adj_c @ SAME.T                                       # [324, 1680]
    p_solved_c = (edges_c >= 6.0).astype(f32) @ w
    e_edges_c = edges_c @ w
    pair_lock = (locked[PI] | locked[PJ]).astype(f32)
    e_dist_c = dist_c @ (p_same * (1.0 + W_LOCK_DIST * pair_lock))
    compact_c = adj_c.sum(1)
    # misleading-move penalty: moving a card I know next to cards unlikely to share its colour
    known_mover = (known[CAND_I] >= 0).astype(f32)                 # [324]
    mover_pair = ((PI[None, :] == CAND_I[:, None]) | (PJ[None, :] == CAND_I[:, None])).astype(f32)   # [324, 36]
    best_nb_same = jnp.max(adj_c * mover_pair * p_same[None, :], axis=1)   # [324]
    mislead_c = known_mover * (best_nb_same < 0.5).astype(f32)
    mscore = (W_SOLVED * p_solved_c + W_EDGES * e_edges_c - W_DIST * e_dist_c + W_COMPACT * compact_c
              - W_MISLEAD * mislead_c + 0.001 * jax.random.uniform(k_move, (NUM_CAND,)))
    mscore = jnp.where(valid, mscore, -1e9)
    best_c = jnp.argmax(mscore)
    move_fallback = A_MOVE + jnp.argmax(legal[A_MOVE:A_REVEAL])
    move_action = jnp.where(valid.any(), move_ids[best_c], move_fallback)
    p_best_move = jnp.max(jnp.where(valid, p_solved_c, 0.0))

    # ---- observation choice: 1-step value of information for the next move -----
    solved_c = (edges_c >= 6.0).astype(f32)                         # [324, 1680]
    wk = (w[:, None, None] * ONEHOT).reshape(NUM_ASSIGN, NUM_CARDS * NUM_COLORS)   # [1680, 27]
    marg_flat = marg.reshape(-1)                                    # [27]
    e_edges_ck = edges_c @ wk                                       # [324, 27]
    p_solved_ck = solved_c @ wk
    dist_w = dist_c * (1.0 + W_LOCK_DIST * pair_lock)[None, :]      # [324, 36]
    e_dist_ck = dist_w @ (SAME.T @ wk)                              # [324, 27]
    score_ck = (W_SOLVED * p_solved_ck + W_EDGES * e_edges_ck - W_DIST * e_dist_ck
                + W_COMPACT * compact_c[:, None] * marg_flat[None, :])
    score_ck = jnp.where(valid_struct[:, None], score_ck, -1e9)
    best_ck = jnp.max(score_ck, axis=0).reshape(NUM_CARDS, NUM_COLORS)   # already weighted by marg
    mscore_struct = (W_SOLVED * p_solved_c + W_EDGES * e_edges_c - W_DIST * e_dist_c + W_COMPACT * compact_c)
    best_now = jnp.max(jnp.where(valid_struct, mscore_struct, -1e9))
    voi = jnp.where(marg > 1e-6, best_ck, 0.0).sum(1) - best_now    # [9]
    ent = -(marg * jnp.log(marg + 1e-9)).sum(1)
    pjoint = jnp.einsum('a,ack->ck', w * solved_now, ONEHOT)
    p_cond = pjoint / jnp.maximum(marg, 1e-9)
    rel = (marg * jnp.abs(p_cond - p_now)).sum(1)
    obs_score = (voi + W_ENT * ent + 2.0 * rel + OBS_UNSEEN_BONUS * (1.0 - pseen.astype(f32))
                 + 0.01 * jax.random.uniform(k_obs, (NUM_CARDS,)))
    obs_legal = legal[A_OBS:A_OBS + NUM_CARDS]
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    obs_action = A_OBS + jnp.argmax(obs_score)

    # ---- hint choice --------------------------------------------------------
    dist_mat = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)  # [9, 9]
    unsafe_cd = locked[None, :] & (psame_mat > 0.1) & (dist_mat > 2) & (CARD_IDX[:, None] != CARD_IDX[None, :])
    unsafe = unsafe_cd.any(1)                                      # [9]
    same_nb = (psame_mat * (dist_mat == 1)).sum(1)                 # expected same-colour neighbours
    is_single = (hset.sum(1) == 1.0)
    comm_bonus = 1.0 - pseen[None, :].astype(f32)   # any hint on a partner-unseen card informs the partner
    place_ids = A_PLACE + HINT_IDX[:, None] * 81 + pos[None, :, 0] * 9 + pos[None, :, 1]   # [4, 9]
    avail = (revealed & ~placed)[:, None] & legal[place_ids]
    pscore = (p_corr + 0.05 * same_nb[None, :] + COMM_BONUS * comm_bonus + 0.02 * is_single[:, None].astype(f32)
              + 0.001 * jax.random.uniform(k_hint, (NUM_HINTS, NUM_CARDS)))
    safe_avail = avail & ~unsafe[None, :]
    pscore_safe = jnp.where(safe_avail, pscore, -1e9)
    pscore_any = jnp.where(avail, pscore, -1e9)
    use_safe = safe_avail.any()
    pscore_sel = jnp.where(use_safe, pscore_safe, pscore_any)
    flat = jnp.argmax(pscore_sel)
    bh = flat // NUM_CARDS
    bc = flat % NUM_CARDS
    best_p = p_corr[bh, bc]
    forced = (n_unrev == 0)
    place_ok = avail.any() & use_safe & (best_p >= PLACE_MIN_P)
    do_place = place_ok | (forced & avail.any())
    reveal_ok = (~revealed) & legal[A_REVEAL:A_REVEAL + NUM_HINTS]
    reveal_action = A_REVEAL + jnp.argmax(reveal_ok)
    hint_action = jnp.where(do_place, place_ids[bh, bc], reveal_action)

    # ---- end decision -------------------------------------------------------
    c_est = jnp.where(place_ok, 1.0, 3.0)
    forced_last = (n_unrev == 0) & (n_rev_unplaced == 1)
    v_cont = jnp.where(forced_last, score_now - 1.0, score_now - c_est / 2.0 - 2.0) * END_Q
    tau = jnp.clip(v_cont / jnp.maximum(score_now, 1.0), END_TAU_LO, END_TAU_HI)
    p_adj = 1.0 - (1.0 - p_now) ** K_CAL
    do_end = legal[A_END] & (p_adj >= tau)
    step0_action = jnp.where(do_end, A_END, obs_action)

    action = jnp.select(
        [substep == 0, substep == 1, substep == 2],
        [step0_action, obs_action, move_action],
        hint_action,
    )
    action = jnp.where(active, action, 0)
    action = jnp.where(legal[action], action, jnp.argmax(legal)).astype(jnp.int32)

    next_memory = {
        'known': known.astype(jnp.int32),
        'logw': logw.astype(jnp.float32),
        'prev_pos': pos,
        'pseen': pseen.astype(jnp.int32),
    }
    trace = {
        'p_solved': p_now.astype(f32),
        'score_now': score_now.astype(f32),
        'tau': tau.astype(f32),
        'p_best_move': p_best_move.astype(f32),
        'voi_max': jnp.max(jnp.where(obs_legal, voi, 0.0)).astype(f32),
        'voi': voi.astype(f32),
        'ent': ent.astype(f32),
        'obs_score': obs_score.astype(f32),
        'known': known.astype(jnp.int32),
        'best_now': best_now.astype(f32),
        'n_known': (known >= 0).sum().astype(jnp.int32),
        'substep': substep.astype(jnp.int32),
        'active': active.astype(jnp.int32),
    }
    return action, next_memory, trace
