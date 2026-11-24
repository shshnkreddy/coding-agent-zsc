"""Belief-based Yokai policy (convention-free, local-observation only).

Core ideas
- Track privately observed colors in memory (standard memory mode forgets them).
- Maintain an exact posterior over all 1680 colorings consistent with own
  observations, softly weighted by public evidence: a placed hint is treated
  as strong-but-not-certain evidence that the card's color is in the hint's
  set (dropped if a placement contradicts an own observation), and a partner
  move of a card the partner has observed is treated as mild evidence that
  the card now touches a same-colored card the partner also observed
  (dropped after repeated partner moves that ignore mates both of us know).
- Move choice: maximize the expected number of "useful" same-color
  orthogonally adjacent pairs over the most likely colorings, where a pair
  only counts if it can still be part of the final group given locked cards,
  plus a bonus on the probability that all colors are connected after the
  move. All three colors are connected iff the pair count equals 6, so the
  expectation is a natural potential that also rewards hedged placements of
  unknown cards. With no locks yet, the plain expected-pairs objective over
  all consistent colorings is used.
- Hint choice: place a revealed hint on a card whose color is known to be in
  the hint set (cost 1 point) rather than revealing another hint (cost 3);
  otherwise reveal, single-color hint first. Never lock a card that would
  leave its color impossible to connect; prefer cards already joined to a
  mate and cards the partner has observed.
- End decision (substep 0): end when P(all connected) exceeds a per-turn
  threshold tuned against the value-if-ended-now ladder and calibration.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_useful_pairs','adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
GRID = 9
NUM_CELLS = 81
NUM_HINTS = 4
NUM_COLORINGS = 1680

# Tunables
HINT_TRUST = 0.1            # weight for colorings violating a placed hint
PARTNER_MOVE_FACTOR = 0.2   # weight for colorings where partner's informed move joined no known mate
PLACE_CERTAINTY = 0.999
LOOKAHEAD_GAMMA = 0.0
OBS_PSEEN_WEIGHT = -0.05
OBS_ACTIONABLE_WEIGHT = 0.0
REMOVE_PARTNER_MOVED_PENALTY = 0.15   # discourage re-moving an unknown card the partner just moved
REMOVE_PARTNER_SEEN_PENALTY = 0.05    # discourage moving unknown cards the partner has observed
TOPK_COLORINGS = 48
CONN_BONUS = 1.5                     # extra weight on P(all connected) after the move
PARTNER_DISTRUST_EVENTS = 2           # drop partner-move evidence after this many irrational partner moves
PLACE_MIN_VALUE = 0.95               # place only if P(correct) minus lock penalties is at least this
# End thresholds on P(all connected) at substep 0 of turn k = 0..7
END_THRESHOLD_LIST = [0.4, 0.4, 0.4, 0.4, 0.4, 0.25, 0.3, 0.3]
REVEAL_PREF_LIST = [4.0, 3.0, 2.0, 1.0]   # reveal single-color hint first
PLACE_HINT_PREF_LIST = [0.02, 0.0, 0.0, 0.0]
HINT_SEEN_PREF_LIST = [1.0, 1.0, 1.0, 1.0]   # +1: prefer locking cards the partner has observed


def _tables():
    codes = jnp.arange(3 ** NUM_CARDS, dtype=jnp.int32)
    digits = (codes[:, None] // (3 ** jnp.arange(NUM_CARDS, dtype=jnp.int32))[None, :]) % 3   # [19683, 9]
    counts = (digits[:, :, None] == jnp.arange(NUM_COLORS)[None, None, :]).sum(1)              # [19683, 3]
    valid = jnp.all(counts == 3, axis=-1)
    idx = jnp.nonzero(valid, size=NUM_COLORINGS, fill_value=0)[0]
    col = digits[idx]                                                                           # [1680, 9]
    ii, jj = jnp.triu_indices(NUM_CARDS, k=1)
    pair_i = ii.astype(jnp.int32)
    pair_j = jj.astype(jnp.int32)
    same = (col[:, pair_i] == col[:, pair_j]).astype(jnp.float32)                             # [1680, 36]
    onehot = (col[:, :, None] == jnp.arange(NUM_COLORS)[None, None, :]).astype(jnp.float32)   # [1680, 9, 3]
    cells = jnp.stack(jnp.meshgrid(jnp.arange(GRID), jnp.arange(GRID), indexing='ij'), -1).reshape(-1, 2).astype(jnp.int32)
    cell_cell = (jnp.abs(cells[:, None, :] - cells[None, :, :]).sum(-1) == 1).astype(jnp.float32)  # [81, 81]
    return col, same, onehot, pair_i, pair_j, cells, cell_cell


def init_agent():
    return {
        'known': jnp.full((NUM_CARDS,), -1, dtype=jnp.int32),
        'partner_seen': jnp.zeros((NUM_CARDS,), dtype=jnp.int32),
        'prev_pos': jnp.zeros((NUM_CARDS, 2), dtype=jnp.int32),
        'prev_active': jnp.asarray(0, dtype=jnp.int32),
        'prev_substep': jnp.asarray(-1, dtype=jnp.int32),
        'logw': jnp.zeros((NUM_COLORINGS,), dtype=jnp.float32),
        'partner_last_moved': jnp.asarray(-1, dtype=jnp.int32),
        'hints_unreliable': jnp.asarray(0, dtype=jnp.int32),
        'partner_irrational': jnp.asarray(0, dtype=jnp.int32),
    }


def _adjacency(pos):
    d = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    return (d == 1).astype(jnp.float32)


def _move_scores(pos, psame, locked, move_legal, cells, cell_cell):
    """Return [9, 81] move scores: 1-ply expected-pair gain + discounted best 2nd move."""
    adj = _adjacency(pos)                                                     # [9,9]
    cell_adj = (jnp.abs(cells[:, None, :] - pos[None, :, :]).sum(-1) == 1).astype(jnp.float32)  # [81,9]
    pos_flat = pos[:, 0] * GRID + pos[:, 1]                                   # [9]
    occupied = jnp.zeros((NUM_CELLS,), dtype=jnp.float32).at[pos_flat].set(1.0)
    new_gain = (cell_adj @ psame.T).T                                         # [9 card, 81 cell]
    old_gain = (adj * psame).sum(-1)                                          # [9]
    gain1 = new_gain - old_gain[:, None]                                      # [9,81]
    unlocked = (locked == 0).astype(jnp.float32)
    card_ids = jnp.arange(NUM_CARDS)

    def best_second(i):
        # first move: card i -> cell (81 options). Second move: card j -> cell2.
        ps_i = psame[:, i]                                                    # [9]
        # new_gain'[j, cell2 ; cell] = new_gain[j,cell2] - cell_adj[cell2,i]*ps_i[j] + cc[cell2,cell]*ps_i[j]
        t_new = (new_gain[:, :, None] - (cell_adj[:, i][None, :, None]) * ps_i[:, None, None]
                 + cell_cell[None, :, :] * ps_i[:, None, None])               # [9 j, 81 cell2, 81 cell]
        # old_gain'[j ; cell] = old_gain[j] - adj[j,i]*ps_i[j] + cell_adj[cell,j]*ps_i[j]
        t_old = (old_gain[:, None] - adj[:, i][:, None] * ps_i[:, None]
                 + cell_adj[:, :].T * ps_i[:, None])                          # [9 j, 81 cell]
        g2 = t_new - t_old[:, None, :]                                        # [9 j, 81 cell2, 81 cell]
        # legality (approximate): j != i, unlocked, cell2 empty after first move, cell2 != pos_j,
        # cell2 adjacent to some card other than j after first move.
        occ_after = occupied[:, None] * (1.0 - (jnp.arange(NUM_CELLS) == pos_flat[i]).astype(jnp.float32))[:, None] \
            + jnp.eye(NUM_CELLS)                                              # [81 cell2, 81 cell]
        occ_after = occ_after > 0
        nb_after = (cell_adj.sum(-1) - cell_adj[:, i])[:, None] + cell_cell   # [81 cell2, 81 cell] total nbrs after move
        nb_excl_j = nb_after[None, :, :] - jnp.where(card_ids[:, None, None] == i, cell_cell[None, :, :], cell_adj.T[:, :, None])  # [9,81,81]
        ok = (card_ids != i)[:, None, None] & (unlocked > 0)[:, None, None] & jnp.logical_not(occ_after)[None, :, :] \
            & (jnp.arange(NUM_CELLS)[None, :, None] != pos_flat[:, None, None]) & (nb_excl_j >= 1)
        g2 = jnp.where(ok, g2, -1e9)
        best = g2.max(axis=(0, 1))                                            # [81 cell]
        return jnp.maximum(best, 0.0)

    if LOOKAHEAD_GAMMA > 0.0:
        best2 = jnp.stack([best_second(i) for i in range(NUM_CARDS)], axis=0)     # [9, 81]
        score = gain1 + LOOKAHEAD_GAMMA * best2
    else:
        score = gain1
    return jnp.where(move_legal, score, -1e9), adj, cell_adj


def _useful_move_scores(pos, adj, cell_adj, locked, colK, wK):
    """[9,81] expected 'useful' same-color pairs after each move, over K weighted colorings.

    A same-color pair is useful if the color has no locked card, or the pair
    touches a locked card directly or through the third card of that color.
    """
    K = colK.shape[0]
    lock_f = (locked > 0).astype(jnp.float32)
    # triples of card indices per coloring and color: [K, 3, 3]
    order = jnp.argsort((colK[:, None, :] != jnp.arange(NUM_COLORS)[None, :, None]).astype(jnp.int32), axis=-1)
    trip = order[:, :, :3]
    # pair endpoints P0=(0,1), P1=(1,2), P2=(0,2)
    px = jnp.stack([trip[:, :, 0], trip[:, :, 1], trip[:, :, 0]], axis=-1)   # [K,3,3]
    py = jnp.stack([trip[:, :, 1], trip[:, :, 2], trip[:, :, 2]], axis=-1)
    l0 = lock_f[trip[:, :, 0]]
    l1 = lock_f[trip[:, :, 1]]
    l2 = lock_f[trip[:, :, 2]]
    nl = (l0 + l1 + l2) == 0                                                # [K,3]
    base_e = adj[px, py]                                                     # [K,3,3]

    def per_card(i):
        # e after moving card i to each cell: [81, K, 3, 3]
        cx = cell_adj[:, py]                                                 # [81,K,3,3] adj(cell, y)
        cy = cell_adj[:, px]                                                 # adj(cell, x)
        e = jnp.where((px == i)[None], cx, jnp.where((py == i)[None], cy, base_e[None]))
        e0, e1, e2 = e[..., 0], e[..., 1], e[..., 2]
        b0 = nl[None] | (l0[None] > 0) | (l1[None] > 0) | ((e2 > 0) & (l2[None] > 0)) | ((e1 > 0) & (l2[None] > 0))
        b1 = nl[None] | (l1[None] > 0) | (l2[None] > 0) | ((e0 > 0) & (l0[None] > 0)) | ((e2 > 0) & (l0[None] > 0))
        b2 = nl[None] | (l0[None] > 0) | (l2[None] > 0) | ((e0 > 0) & (l1[None] > 0)) | ((e1 > 0) & (l1[None] > 0))
        useful = e0 * b0 + e1 * b1 + e2 * b2                                 # [81,K,3]
        tot = useful.sum(-1)                                                 # [81,K]
        exp_useful = (tot * wK[None, :]).sum(-1)                             # [81]
        p_conn_after = ((tot >= 5.5).astype(jnp.float32) * wK[None, :]).sum(-1)
        return exp_useful + CONN_BONUS * p_conn_after

    scores = jax.lax.map(per_card, jnp.arange(NUM_CARDS))                     # [9,81]
    return scores


def agent_act(obs, reset, memory, rng):
    COL, SAME, ONEHOT, PAIR_I, PAIR_J, CELLS, CELL_CELL = _tables()
    END_THRESHOLD = jnp.asarray(END_THRESHOLD_LIST, dtype=jnp.float32)
    init = init_agent()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    nf = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    pos = obs.positions.astype(jnp.int32)
    active = nf[0, 10]
    substep = nf[0, 11]
    locked = nf[:NUM_CARDS, 8]

    # --- own observations -> known colors
    onehot = nf[:NUM_CARDS, 0:3]
    seen = onehot.sum(-1) > 0
    known = jnp.where(seen, jnp.argmax(onehot, axis=-1), memory['known'])

    # --- partner observation indicator (accumulated over the episode)
    partner_seen = jnp.maximum(memory['partner_seen'], nf[:NUM_CARDS, 9])
    pseen_f = (partner_seen > 0).astype(jnp.float32)

    # --- hint info
    hcol = nf[NUM_CARDS:, 3:6].astype(jnp.float32)      # [4,3]
    hplaced = nf[NUM_CARDS:, 8]                          # [4]
    hloc = nf[NUM_CARDS:, 6:8]                           # [4,2]
    hrevealed = (hcol.sum(-1) > 0).astype(jnp.int32)
    n_unrev = (1 - hrevealed).sum()
    n_placed = hplaced.sum()
    n_rev_unplaced = (hrevealed * (1 - hplaced)).sum()
    turn_idx = jnp.clip(n_rev_unplaced + 2 * n_placed, 0, 7)

    adj = _adjacency(pos)                                # [9,9]
    adj_pairs = adj[PAIR_I, PAIR_J]                      # [36]

    # --- partner move evidence (partner just moved: their substep 3, previous call was substep 2)
    moved = jnp.any(pos != memory['prev_pos'], axis=-1)
    partner_moved = (active == 0) & (substep == 3) & (memory['prev_substep'] == 2) & (memory['prev_active'] == 0) \
        & jnp.any(moved) & jnp.logical_not(reset)
    mv = jnp.argmax(moved)
    same_as_mv = (COL == COL[:, mv][:, None]).astype(jnp.float32)            # [1680, 9]
    mates_known = ((same_as_mv * pseen_f[None, :]).sum(-1) - pseen_f[mv]) > 0  # partner knows a mate of mv (under c)
    joined_known = ((same_as_mv * pseen_f[None, :] * adj[mv][None, :]).sum(-1)) > 0
    informative = partner_moved & (pseen_f[mv] > 0)
    penal = informative & mates_known & jnp.logical_not(joined_known)
    # partner rationality check: they moved a card whose color I know, which they observed, while a
    # mate they also observed exists, yet the card did not end next to any same-colored card I know.
    kn = memory['known']
    same_known = (kn[None, :] == kn[:, None]) & (kn[:, None] >= 0) & (jnp.arange(NUM_CARDS)[None, :] != jnp.arange(NUM_CARDS)[:, None])
    mate_known_to_both = jnp.any(same_known[mv] & (pseen_f > 0))
    joined_mine = jnp.any(same_known[mv] & (adj[mv] > 0))
    irrational_event = partner_moved & (kn[mv] >= 0) & (pseen_f[mv] > 0) & mate_known_to_both & jnp.logical_not(joined_mine)
    partner_irrational = memory['partner_irrational'] + irrational_event.astype(jnp.int32)
    distrust = partner_irrational >= PARTNER_DISTRUST_EVENTS
    logw = memory['logw'] + jnp.where(penal, jnp.log(PARTNER_MOVE_FACTOR), 0.0)
    logw = jnp.where(distrust, 0.0, logw)
    partner_last_moved = jnp.where(partner_moved, mv, memory['partner_last_moved'])

    # --- posterior over colorings
    known_ok = jnp.all((known[None, :] < 0) | (COL == known[None, :]), axis=-1)   # [1680]
    at_loc = jnp.all(pos[None, :, :] == hloc[:, None, :], axis=-1)              # [4,9]
    card_of_hint = jnp.argmax(at_loc, axis=-1)                                  # [4]
    hint_valid = (hplaced > 0) & jnp.any(at_loc, axis=-1)
    col_of_hint_card = COL[:, card_of_hint]                                     # [1680,4]
    in_set = hcol[jnp.arange(NUM_HINTS)[None, :], col_of_hint_card] > 0        # [1680,4]
    # if any placed hint contradicts a color I observed myself, stop trusting hint placements
    known_hint_card = known[card_of_hint]                                       # [4]
    contradiction = hint_valid & (known_hint_card >= 0) & (hcol[jnp.arange(NUM_HINTS), jnp.clip(known_hint_card, 0, 2)] == 0)
    hints_unreliable = jnp.maximum(memory['hints_unreliable'], jnp.any(contradiction).astype(jnp.int32))
    trust_log = jnp.where(hints_unreliable > 0, 0.0, jnp.log(HINT_TRUST))
    hint_logf = jnp.where(hint_valid[None, :] & jnp.logical_not(in_set), trust_log, 0.0).sum(-1)
    logw_total = jnp.where(known_ok, logw + hint_logf, -1e9)
    w = jnp.exp(logw_total - jnp.max(logw_total))
    w = w / w.sum()

    psame_pairs = w @ SAME                                # [36]
    psame = jnp.zeros((NUM_CARDS, NUM_CARDS), dtype=jnp.float32)
    psame = psame.at[PAIR_I, PAIR_J].set(psame_pairs)
    psame = psame + psame.T
    marg = jnp.einsum('c,cik->ik', w, ONEHOT)             # [9,3]

    pairs_per_col = SAME @ adj_pairs                      # [1680]
    p_conn = (w * (pairs_per_col >= 5.5)).sum()

    # --- value if ended now
    pcorrect_placed = jnp.einsum('hk,hk->h', hcol, marg[card_of_hint])          # [4]
    v_now = 5.0 * n_unrev + 2.0 * n_rev_unplaced + jnp.where(hint_valid, 2.0 * pcorrect_placed - 1.0, 0.0).sum()

    rng, k1, k2, k3 = jax.random.split(rng, 4)

    # --- observe choice: highest entropy, prefer cards the partner has seen (may get locked)
    ent = -(marg * jnp.log(marg + 1e-9)).sum(-1)          # [9]
    n_known_col = ((known[:, None] == jnp.arange(NUM_COLORS)[None, :]) & (known[:, None] >= 0)).sum(0).astype(jnp.float32)  # [3]
    actionable_col = ((n_known_col >= 1) & (n_known_col <= 2)).astype(jnp.float32)
    actionable = marg @ actionable_col                       # [9]
    obs_score = ent + OBS_PSEEN_WEIGHT * pseen_f + OBS_ACTIONABLE_WEIGHT * actionable \
        + 1e-3 * jax.random.uniform(k1, (NUM_CARDS,))
    obs_legal = legal[1:1 + NUM_CARDS]
    obs_score = jnp.where(obs_legal, obs_score, -1e9)
    observe_action = 1 + jnp.argmax(obs_score)

    # --- move choice
    move_legal = legal[10:10 + NUM_CARDS * NUM_CELLS].reshape(NUM_CARDS, NUM_CELLS)
    move_score_lin, _, cell_adj = _move_scores(pos, psame, locked, move_legal, CELLS, CELL_CELL)
    # lock-aware objective over the K most likely colorings (exact when few unknown cards)
    topk = jnp.argsort(-w)[:TOPK_COLORINGS]
    wK = w[topk]
    wK = wK / (wK.sum() + 1e-9)
    move_score_useful = _useful_move_scores(pos, adj, cell_adj, locked, COL[topk], wK)
    move_score_useful = jnp.where(move_legal, move_score_useful, -1e9)
    any_locked = jnp.any(locked > 0)
    move_score = jnp.where(any_locked, move_score_useful, move_score_lin)
    unknown_f = (known < 0).astype(jnp.float32)
    card_pen = REMOVE_PARTNER_MOVED_PENALTY * unknown_f * (jnp.arange(NUM_CARDS) == partner_last_moved).astype(jnp.float32) \
        + REMOVE_PARTNER_SEEN_PENALTY * unknown_f * pseen_f
    move_score = move_score - card_pen[:, None] + 1e-3 * jax.random.uniform(k2, (NUM_CARDS, NUM_CELLS))
    move_flat = jnp.argmax(move_score)
    move_action = 10 + move_flat

    # --- hint choice
    reveal_legal = legal[739:743]
    reveal_pref = jnp.asarray(REVEAL_PREF_LIST, dtype=jnp.float32)
    reveal_action = 739 + jnp.argmax(jnp.where(reveal_legal, reveal_pref, -1e9))
    any_reveal = jnp.any(reveal_legal)

    place_legal = legal[743:743 + NUM_HINTS * NUM_CELLS].reshape(NUM_HINTS, NUM_CELLS)
    cell_card = jnp.argmax((jnp.abs(CELLS[:, None, :] - pos[None, :, :]).sum(-1) == 0), axis=-1)  # [81]
    pcorrect_cell = (hcol @ marg.T)[:, cell_card]          # [4,81]
    same_adj_card = (adj * psame).sum(-1)                  # [9] expected same-color neighbours
    occupied = jnp.zeros((NUM_CELLS,), dtype=jnp.float32).at[pos[:, 0] * GRID + pos[:, 1]].set(1.0)
    free_nb = (CELL_CELL @ (1.0 - occupied))               # [81] free neighbouring cells per cell
    # lock-deadlock penalty: locking X near an already-locked probable mate at distance 2 is
    # risky, at distance >= 3 the color can never connect.
    dist_cards = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)          # [9,9]
    lock_f = (locked > 0).astype(jnp.float32)
    gdist = jnp.where(dist_cards >= 3, 2.0, jnp.where(dist_cards == 2, 0.4, 0.0))
    lock_pen_card = (psame * lock_f[None, :] * gdist).sum(-1)                # [9]
    cramped_pen_card = 0.25 * (free_nb[pos[:, 0] * GRID + pos[:, 1]] == 0).astype(jnp.float32)
    place_value = pcorrect_cell - (lock_pen_card + cramped_pen_card)[cell_card][None, :]   # [4,81]
    hint_seen_pref = jnp.asarray(HINT_SEEN_PREF_LIST, dtype=jnp.float32)
    place_pref = place_value \
        + 0.15 * same_adj_card[cell_card][None, :] \
        + 0.03 * jnp.minimum(free_nb, 2.0)[None, :] \
        + 0.05 * hint_seen_pref[:, None] * pseen_f[cell_card][None, :] \
        + jnp.asarray(PLACE_HINT_PREF_LIST, dtype=jnp.float32)[:, None] \
        + 1e-3 * jax.random.uniform(k3, (NUM_HINTS, NUM_CELLS))
    place_pref = jnp.where(place_legal, place_pref, -1e9)
    place_flat = jnp.argmax(place_pref)
    place_action = 743 + place_flat
    best_value = jnp.where(place_legal, place_value, -1e9).max()
    any_place = jnp.any(place_legal)
    do_place = any_place & ((best_value >= PLACE_MIN_VALUE) | jnp.logical_not(any_reveal))
    hint_action = jnp.where(do_place, place_action, reveal_action)

    # --- end decision
    end_legal = legal[1067]
    want_end = end_legal & (p_conn >= END_THRESHOLD[turn_idx])
    step0_action = jnp.where(want_end, 1067, observe_action)

    action = jnp.where(substep == 0, step0_action,
             jnp.where(substep == 1, observe_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active > 0, action, 0)
    action = action.astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))

    new_memory = {
        'known': known.astype(jnp.int32),
        'partner_seen': partner_seen.astype(jnp.int32),
        'prev_pos': pos,
        'prev_active': active.astype(jnp.int32),
        'prev_substep': substep.astype(jnp.int32),
        'logw': logw.astype(jnp.float32),
        'partner_last_moved': partner_last_moved.astype(jnp.int32),
        'hints_unreliable': hints_unreliable.astype(jnp.int32),
        'partner_irrational': partner_irrational.astype(jnp.int32),
    }
    trace = {
        'p_conn': p_conn.astype(jnp.float32),
        'v_now': v_now.astype(jnp.float32),
        'turn_idx': turn_idx.astype(jnp.int32),
        'n_known': (known >= 0).sum().astype(jnp.int32),
        'want_end': want_end.astype(jnp.int32),
        'known': known.astype(jnp.int32),
        'pos': pos.reshape(-1).astype(jnp.int32),
        'locked': locked.astype(jnp.int32),
        'hint_state': (hrevealed + hplaced).astype(jnp.int32),
    }
    return action, new_memory, trace
