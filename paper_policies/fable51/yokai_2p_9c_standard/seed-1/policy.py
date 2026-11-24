"""Yokai 2p policy: Bayesian enumeration over hidden colorings.

Private memory stores every color this seat has observed. All 1680 colorings
(280 unordered partitions x 6 labelings) are enumerated; the partitions that
are consistent with memory drive move selection (maximise the probability that
every color is orthogonally connected, then expected progress), the colorings
drive marginal color probabilities for hint placement and observation choice.
Hints are revealed rather than placed whenever possible. The game is ended at
substep 0 when the posterior probability of success exceeds a turn-dependent
threshold.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bayes_enum_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NC = 9          # cards
G = 9           # grid
NCELL = G * G   # 81
NH = 4          # hints
EVID_EPS = 0.25  # likelihood of a partner-move evidence pair being different colors
HARD_FLOOR = 0.0       # minimum evidence-free P(connected) required to end early (0 = off)
HARD_FLOOR_UNTIL = 6   # floor applies to turns before this one
LOOK_K = 6             # first moves expanded in the 2-ply lookahead
LOOK_W = 4.0           # weight of the best second-move P(connected)
SIG_W = 0.3            # bonus per own-known same-colored neighbour at the destination


def _tables():
    """Enumerate the 1680 colorings and 280 canonical partitions with jnp only."""
    n = 3 ** NC
    idx = jnp.arange(n, dtype=jnp.int32)
    powers = jnp.asarray([3 ** d for d in range(NC)], dtype=jnp.int32)
    digits = (idx[:, None] // powers[None, :]) % 3              # [19683, 9]
    counts = jax.nn.one_hot(digits, 3, dtype=jnp.int32).sum(1)  # [19683, 3]
    valid = (counts == 3).all(axis=1)
    order = jnp.argsort(jnp.where(valid, 0, 1), stable=True)
    cols = digits[order[:1680]]                                  # [1680, 9]
    # restricted growth strings: canonical labelling of partitions
    prefix_max = jax.lax.cummax(digits, axis=1)
    prev_max = jnp.concatenate([jnp.full((n, 1), -1, jnp.int32), prefix_max[:, :-1]], axis=1)
    rgs = (digits <= prev_max + 1).all(axis=1) & (digits[:, 0] == 0)
    order_p = jnp.argsort(jnp.where(valid & rgs, 0, 1), stable=True)
    pats = digits[order_p[:280]]                                 # [280, 9]
    cell_xy = jnp.stack([jnp.repeat(jnp.arange(G), G), jnp.tile(jnp.arange(G), G)], axis=1).astype(jnp.int32)
    # map every coloring to its canonical partition index
    first = jnp.argmax(cols[:, None, :] == jnp.arange(3)[None, :, None], axis=2)  # [1680, 3]
    rank = (first[:, None, :] < first[:, :, None]).sum(axis=2)                  # [1680, 3]
    canon = jnp.take_along_axis(rank, cols, axis=1)                             # [1680, 9]
    col2pat = (canon[:, None, :] == pats[None, :, :]).all(axis=2).astype(jnp.float32)  # [1680, 280]
    return cols, pats, cell_xy, col2pat


def init_agent():
    return {
        'known': jnp.full((NC,), -1, dtype=jnp.int32),
        'prev_pos': jnp.full((NC, 2), -1, dtype=jnp.int32),
        'pknown': jnp.zeros((NC,), dtype=jnp.int32),
        'evid': jnp.zeros((NC, NC), dtype=jnp.int32),
        'step': jnp.asarray(0, dtype=jnp.int32),
    }


def _card_adjacency(pos):
    d = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    return (d == 1).astype(jnp.float32)  # [9, 9]


def _cell_adjacency(pos, cell_xy):
    d = jnp.abs(cell_xy[:, None, :] - pos[None, :, :]).sum(-1)
    return (d == 1).astype(jnp.float32)  # [81, 9]


def _own_move_legal(pos, locked, cell_xy):
    """Legality of moving card j to cell x: empty target, unlocked card, board stays connected."""
    occ = jnp.zeros((NCELL,), dtype=jnp.bool_).at[pos[:, 0] * G + pos[:, 1]].set(True)
    empty = ~occ                                                   # [81]
    # positions after each hypothetical move: [9(j), 81(x), 9(card), 2]
    base = jnp.broadcast_to(pos[None, None], (NC, NCELL, NC, 2))
    sel = (jnp.arange(NC)[:, None, None, None] == jnp.arange(NC)[None, None, :, None])
    newp = jnp.where(sel, cell_xy[None, :, None, :], base)
    d = jnp.abs(newp[:, :, :, None, :] - newp[:, :, None, :, :]).sum(-1)  # [9, 81, 9, 9]
    reach = (d <= 1).astype(jnp.float32)
    for _ in range(4):
        reach = jnp.minimum(jnp.einsum('jxab,jxbc->jxac', reach, reach), 1.0)
    connected = (reach[:, :, 0, :] > 0).all(axis=-1)                # [9, 81]
    return connected & empty[None, :] & (~(locked > 0))[:, None]


def _eval_moves(pos, pats_oh, cell_xy):
    """Per-pattern connectivity and progress after moving card i to cell x."""
    A = _card_adjacency(pos)                                       # [9, 9]
    B = _cell_adjacency(pos, cell_xy)                              # [81, 9]
    M = jnp.einsum('kig,ij,kjg->kg', pats_oh, A, pats_oh) / 2.0    # [K, 3]
    nbr_same = jnp.einsum('kig,ij,kjg->ki', pats_oh, A, pats_oh)   # [K, 9]
    cellc = jnp.einsum('xj,kjg->kxg', B, pats_oh)                  # [K, 81, 3]
    new_same = jnp.einsum('kig,kxg->kix', pats_oh, cellc) - B.T[None]  # [K, 9, 81]
    Mi = jnp.einsum('kig,kg->ki', pats_oh, M)                      # [K, 9]
    n_ok = (M >= 2).sum(axis=1)                                    # [K]
    other_ok = (n_ok[:, None] - (Mi >= 2)) == 2                    # [K, 9]
    own_new = Mi[:, :, None] - nbr_same[:, :, None] + new_same     # [K, 9, 81]
    cond = (other_ok[:, :, None] & (own_new >= 2)).astype(jnp.float32)
    prog_other = jnp.minimum(M, 2).sum(axis=1)[:, None] - jnp.minimum(Mi, 2)  # [K, 9]
    prog_new = prog_other[:, :, None] + jnp.minimum(own_new, 2)
    return nbr_same, cond, prog_new, Mi


def _pattern_weights(known, pats, evid, eps):
    same_pat = pats[:, :, None] == pats[:, None, :]  # [K, 9, 9]
    kn = known >= 0
    both = kn[:, None] & kn[None, :]
    same_known = both & (known[:, None] == known[None, :])
    diff_known = both & (known[:, None] != known[None, :])
    bad = (same_known[None] & ~same_pat) | (diff_known[None] & same_pat)
    ok = ~bad.any(axis=(1, 2))
    # soft evidence from partner moves: pairs believed same-colored
    mism = (~same_pat).astype(jnp.float32) * evid.astype(jnp.float32)[None]
    logw = mism.sum(axis=(1, 2)) * jnp.log(eps)
    return ok.astype(jnp.float32) * jnp.exp(logw)  # [K]


def _coloring_weights(known, cols, evid, eps):
    ok = ((known[None, :] < 0) | (cols == known[None, :])).all(axis=1)
    same = cols[:, :, None] == cols[:, None, :]
    mism = (~same).astype(jnp.float32) * evid.astype(jnp.float32)[None]
    logw = mism.sum(axis=(1, 2)) * jnp.log(eps)
    return ok.astype(jnp.float32) * jnp.exp(logw)  # [1680]


def agent_act(obs, reset, memory, rng):
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    substep = nf[0, 11]
    active = nf[0, 10] == 1

    # update knowledge from currently visible colors
    vis_oh = nf[:NC, 0:3]
    vis = vis_oh.sum(1) > 0
    vis_col = jnp.argmax(vis_oh, axis=1)
    known = jnp.where(vis, vis_col, mem['known'])

    # partner knowledge and move evidence (processed on every call)
    pflag = nf[:NC, 9] > 0
    pknown = jnp.maximum(mem['pknown'], pflag.astype(jnp.int32))
    pk = pknown > 0
    prev_pos = mem['prev_pos']
    have_prev = (prev_pos[:, 0] >= 0).all()
    moved = (pos != prev_pos).any(axis=1) & have_prev
    partner_moved = (~active) & (substep == 3) & moved.any()
    z = jnp.argmax(moved)
    A0 = _card_adjacency(pos)
    nb_z = (A0[z] > 0) & pk & (jnp.arange(NC) != z)
    add = partner_moved & pk[z]
    ev_row = jnp.where(add & nb_z, 1, 0).astype(jnp.int32)
    evid = mem['evid'].at[z].add(ev_row).at[:, z].add(ev_row)

    hint_sets = (nf[NC:NC + NH, 3:6] > 0).astype(jnp.float32)  # [4, 3]
    hint_revealed = hint_sets.sum(1) > 0
    hint_placed = nf[NC:NC + NH, 8] == 1
    turn = jnp.clip(hint_revealed.sum() + hint_placed.sum(), 0, 7)

    # posterior over partitions / colorings
    COLS, PATS, CELL_XY, COL2PAT = _tables()
    COLS_OH = jax.nn.one_hot(COLS, 3, dtype=jnp.float32)  # [1680, 9, 3]
    PATS_OH = jax.nn.one_hot(PATS, 3, dtype=jnp.float32)  # [280, 9, 3]
    END_THRESH = jnp.asarray([0.95, 0.85, 0.6, 0.6, 0.6, 0.6, 0.55, 0.6], dtype=jnp.float32)
    # partner reliability: evidence pairs contradicted by my own observations
    both_known = (known[:, None] >= 0) & (known[None, :] >= 0)
    wrong = (evid > 0) & both_known & (known[:, None] != known[None, :])
    n_wrong = wrong.sum() // 2
    eps = jnp.where(n_wrong == 0, EVID_EPS, jnp.where(n_wrong == 1, 0.5, 0.8))
    wp = _pattern_weights(known, PATS, evid, eps)      # [280]
    wp_sum = jnp.maximum(wp.sum(), 1e-6)
    wc = _coloring_weights(known, COLS, evid, eps)     # [1680]
    wc_sum = jnp.maximum(wc.sum(), 1e-6)
    marg = jnp.einsum('k,kic->ic', wc, COLS_OH) / wc_sum  # [9, 3]

    A = _card_adjacency(pos)                      # [9, 9]
    B = _cell_adjacency(pos, CELL_XY)             # [81, 9]
    M = jnp.einsum('kig,ij,kjg->kg', PATS_OH, A, PATS_OH) / 2.0  # [280, 3]
    conn_now = (M >= 2).all(axis=1).astype(jnp.float32)
    p_now = (wp * conn_now).sum() / wp_sum
    # evidence-free posterior (own observations only) as a robustness floor
    wp_hard = _pattern_weights(known, PATS, jnp.zeros((NC, NC), dtype=jnp.int32), eps)
    p_hard = (wp_hard * conn_now).sum() / jnp.maximum(wp_hard.sum(), 1e-6)

    # ---- move evaluation (1-ply) ----
    nbr_same, cond, prog_new, Mi = _eval_moves(pos, PATS_OH, CELL_XY)
    p_move = jnp.einsum('k,kix->ix', wp, cond) / wp_sum            # [9, 81]
    e_prog = jnp.einsum('k,kix->ix', wp, prog_new) / wp_sum        # [9, 81]
    # compactness tie-break: distance of destination from centroid
    cen = pos.astype(jnp.float32).mean(0)
    dist = jnp.abs(CELL_XY.astype(jnp.float32) - cen[None]).sum(1)  # [81]
    # legible clustering: moving an own-observed card next to own-observed
    # same-colored cards is natural evidence for any partner
    kn_oh = jax.nn.one_hot(known, 3, dtype=jnp.float32) * (known >= 0)[:, None]
    same_kn = jnp.einsum('xj,jc,ic->ix', B, kn_oh, kn_oh) - B.T * (known >= 0)[:, None]
    move_score = 10.0 * p_move + e_prog + SIG_W * same_kn - 0.01 * dist[None, :]
    move_legal = legal[10:739].reshape(NC, NCELL)
    move_score = jnp.where(move_legal, move_score, -1e9)
    # ---- 2-ply lookahead over the top-K first moves ----
    locked = nf[:NC, 8]
    flat = move_score.reshape(-1)
    top_idx = jnp.argsort(-flat)[:LOOK_K]                          # [K]
    top_card = top_idx // NCELL
    top_cell = top_idx % NCELL

    def second_ply(card, cell):
        pos2 = pos.at[card].set(CELL_XY[cell])
        _, cond2, _, _ = _eval_moves(pos2, PATS_OH, CELL_XY)
        p2 = jnp.einsum('k,kix->ix', wp, cond2) / wp_sum
        legal2 = _own_move_legal(pos2, locked, CELL_XY)
        return jnp.max(jnp.where(legal2, p2, 0.0))

    best_p2 = jax.vmap(second_ply)(top_card, top_cell)              # [K]
    top_score = flat[top_idx] + LOOK_W * best_p2
    top_score = jnp.where(flat[top_idx] > -1e8, top_score, -1e9)
    move_action = 10 + top_idx[jnp.argmax(top_score)]

    # ---- observation choice: value of information for the coming move ----
    my_legal = _own_move_legal(pos, locked, CELL_XY)                # [9, 81]
    util = (10.0 * cond + prog_new).reshape(280, -1)                # [280, 729]
    hyp = wc[None, None, :] * (COLS.T[:, None, :] == jnp.arange(3)[None, :, None])  # [9, 3, 1680]
    wp_ic = jnp.einsum('icK,Kk->ick', hyp, COL2PAT)                 # [9, 3, 280]
    n_ic = wp_ic.sum(-1)                                            # [9, 3]
    val_ic = jnp.einsum('ick,km->icm', wp_ic, util) / jnp.maximum(n_ic, 1e-6)[:, :, None]
    val_ic = jnp.where(my_legal.reshape(1, 1, -1), val_ic, -1e9).max(-1)  # [9, 3]
    p_ic = n_ic / jnp.maximum(n_ic.sum(-1, keepdims=True), 1e-6)
    voi = (p_ic * val_ic).sum(-1)                                   # [9]
    ent = -(marg * jnp.log(marg + 1e-9)).sum(1)                     # [9]
    obs_score = voi + 0.05 * ent - 0.05 * pk.astype(jnp.float32) - 0.001 * jnp.arange(NC)
    obs_score = jnp.where(known >= 0, -5.0, obs_score)
    obs_legal = legal[1:10]
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    obs_action = 1 + jnp.argmax(obs_score)

    # ---- end decision ----
    p_best = jnp.max(jnp.where(move_legal, p_move, 0.0))
    late_ok = (turn >= 7) & (p_now >= 0.85 * p_best) & (p_now >= 0.45)
    hard_ok = (p_hard >= HARD_FLOOR) | (turn >= HARD_FLOOR_UNTIL)
    end_ok = legal[1067] & ((p_now >= END_THRESH[turn]) | late_ok) & hard_ok
    s0_action = jnp.where(end_ok, 1067, obs_action)

    # ---- hint handling ----
    reveal_legal = legal[739:743]
    reveal_pref = jnp.asarray([0.0, 3.0, 2.0, 1.0])
    reveal_score = jnp.where(reveal_legal, reveal_pref, -1e9)
    reveal_action = 739 + jnp.argmax(reveal_score)
    any_reveal = reveal_legal.any()
    match = jnp.einsum('hc,ic->hi', hint_sets, marg)               # [4, 9]
    # prefer locking cards whose own color group is already connected and
    # which are not needed as movers (many same-colored neighbours)
    conn_i = jnp.einsum('k,ki->i', wp, (Mi >= 2).astype(jnp.float32)) / wp_sum  # [9]
    nbr_i = jnp.einsum('k,ki->i', wp, nbr_same) / wp_sum                      # [9]
    settle = 0.6 * conn_i + 0.1 * jnp.minimum(nbr_i, 2.0) - 0.15 * (known < 0)
    cell_idx = pos[:, 0] * G + pos[:, 1]                           # [9]
    place_score = jnp.zeros((NH, NCELL), dtype=jnp.float32).at[:, cell_idx].set(match + settle[None, :])
    place_legal = legal[743:1067].reshape(NH, NCELL)
    place_score = jnp.where(place_legal, place_score, -1e9)
    place_action = 743 + jnp.argmax(place_score.reshape(-1))
    s3_action = jnp.where(any_reveal, reveal_action, place_action)

    action = jax.lax.switch(
        jnp.clip(substep, 0, 3),
        [lambda: s0_action, lambda: obs_action, lambda: move_action, lambda: s3_action],
    )
    action = jnp.where(active, action, 0)
    # safety: ensure legality
    action = jnp.where(legal[action], action, jnp.argmax(legal))
    action = action.astype(jnp.int32)

    next_mem = {
        'known': known,
        'prev_pos': pos,
        'pknown': pknown,
        'evid': evid,
        'step': mem['step'] + 1,
    }
    trace = {
        'p_now': p_now.astype(jnp.float32),
        'n_known': (known >= 0).sum().astype(jnp.int32),
        'turn': turn.astype(jnp.int32),
        'n_evid': (evid.sum() // 2).astype(jnp.int32),
        'p_best': p_best.astype(jnp.float32),
        'p_hard': p_hard.astype(jnp.float32),
        'n_wrong': n_wrong.astype(jnp.int32),
        'pos': pos.astype(jnp.int32),
        'known': known.astype(jnp.int32),
    }
    return action, next_mem, trace
