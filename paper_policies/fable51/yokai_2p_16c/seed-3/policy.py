"""Two-player Yokai policy: belief-driven clustering with a Monte-Carlo end rule.

Every decision uses only the local observation (own observed colours, public
positions/locks/hints) and a small private memory that tracks which cards the
partner moved (and their neighbours at that time) to use as soft evidence.
No shared convention is assumed: the partner's moves are treated as weak
evidence about colours ("a moved, observed card probably touches a same
colour card"), never as a protocol.

Belief: unknown colours are sampled from the remaining colour multiset and
weighted by (a) partner-move evidence and (b) placed-hint evidence.
Moves: the top heuristic candidates are re-scored by the expected number of
colour components under the belief.  End: when the belief probability that all
four colours are connected exceeds a score-dependent threshold.
"""

import jax
import jax.numpy as jnp
from jax import lax

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'cluster_mc_v2', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NC = 16          # cards
NH = 7           # hints
G = 10           # grid
NCOL = 4
NCELL = G * G
NS = 96          # Monte-Carlo samples
K_MOVES = 32     # candidate moves re-scored by the belief
CLOSED_W = 0.8   # cost of an incomplete colour whose largest component cannot be joined
BLOCKED_W = 0.7  # cost of a card that must move but is an articulation point of the board
EPS_MOVE = 0.15  # likelihood of a partner move that does not touch a same-colour card
EPS_HINT = 0.2   # likelihood of a placed hint whose set misses the card colour
EPS_STAY = 0.5   # likelihood that a partner-observed, never-moved card touches no same-colour card

A_OBS = 1
A_MOVE = 17
A_REVEAL = 1617
A_PLACE = 1624
A_END = 2324

REVEAL_PREF_LIST = (6.0, 5.0, 1.0, 1.0, 1.0, 3.0, 3.0)  # hint order 0,1,5,6,2,3,4


def _consts():
    ids = jnp.arange(NC)
    cells = jnp.arange(NCELL)
    return ids, cells // G, cells % G


def init_agent():
    return {
        'have_prev': jnp.zeros((), jnp.bool_),
        'prev_pos': jnp.zeros((NC, 2), jnp.int32),
        'known': jnp.full((NC,), -1, jnp.int32),
        'ev_valid': jnp.zeros((NC,), jnp.bool_),
        'ev_neigh': jnp.zeros((NC, NC), jnp.bool_),
        'pmoved': jnp.zeros((NC,), jnp.int32),
        'imoved': jnp.zeros((NC,), jnp.bool_),
        'ok_cnt': jnp.zeros((), jnp.int32),
        'bad_cnt': jnp.zeros((), jnp.int32),
        'prev_hplaced': jnp.zeros((NH,), jnp.bool_),
    }


def _connected(adj, mask):
    """True iff the node set `mask` is connected under `adj` (empty set counts as connected)."""
    ids = jnp.arange(NC)
    start = jnp.argmax(mask)
    reach = (ids == start) & mask

    def body(_, r):
        return mask & (r | (adj & r[None, :]).any(-1))

    reach = lax.fori_loop(0, NC - 1, body, reach)
    return ~(mask & ~reach).any()


def _labels(adj, mask):
    """Component label (minimum member id) for nodes in mask, NC elsewhere."""
    ids = jnp.arange(NC)
    lab = jnp.where(mask, ids, NC)
    a = adj & mask[None, :] & mask[:, None]

    def body(_, lab):
        nb = jnp.where(a, lab[None, :], NC).min(-1)
        return jnp.where(mask, jnp.minimum(lab, nb), NC)

    return lax.fori_loop(0, NC - 1, body, lab)


def _adj_from_pos(pos):
    d = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    return d == 1


def _neighbor_cards(pos):
    """For each of the 100 cells: ids of orthogonally adjacent cards (-1 if none), shape [100,4]."""
    IDS, CELL_X, CELL_Y = _consts()
    occ = jnp.full((NCELL,), -1, jnp.int32).at[pos[:, 0] * G + pos[:, 1]].set(IDS)
    outs = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx = CELL_X + dx
        ny = CELL_Y + dy
        valid = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
        idx = jnp.clip(nx, 0, G - 1) * G + jnp.clip(ny, 0, G - 1)
        outs.append(jnp.where(valid, occ[idx], -1))
    return jnp.stack(outs, -1), occ


def _sample_belief(rng, known, ev_valid, ev_neigh, hint_card, hint_set, hint_active, stay_ev, stay_adj, eps_move, eps_stay, eps_hint):
    """Sample colour assignments for unknown cards and their evidence weights."""
    known_mask = known >= 0
    counts = jnp.stack([(known == c).sum() for c in range(NCOL)])
    rem = NCOL - counts
    ends = jnp.cumsum(rem)
    rem_list = (jnp.arange(NC)[:, None] >= ends[None, :]).sum(-1)  # colour for rank j
    rem_list = jnp.clip(rem_list, 0, NCOL - 1)
    u = jax.random.uniform(rng, (NS, NC))
    u = jnp.where(known_mask[None, :], 2.0, u)
    order = jnp.argsort(u, axis=-1)
    rank = jnp.argsort(order, axis=-1)
    sigma = jnp.where(known_mask[None, :], known[None, :], rem_list[rank])  # [NS, NC]

    def weight(s):
        same = s[:, None] == s[None, :]
        touch = (ev_neigh & same).any(-1)
        w_move = jnp.where(ev_valid, jnp.where(touch, 1.0, eps_move), 1.0).prod()
        hit = hint_set[jnp.arange(NH), jnp.clip(s[hint_card], 0, NCOL - 1)]
        w_hint = jnp.where(hint_active, jnp.where(hit, 1.0, eps_hint), 1.0).prod()
        touch_now = (stay_adj & same).any(-1)
        w_stay = jnp.where(stay_ev, jnp.where(touch_now, 1.0, eps_stay), 1.0).prod()
        return w_move * w_hint * w_stay, hit

    w, hits = jax.vmap(weight)(sigma)
    w = w / jnp.maximum(w.sum(), 1e-9)
    return sigma, w, hits


def _all_connected(adj, s):
    return jnp.stack([_connected(adj, s == c) for c in range(NCOL)]).all()


def _moves_to_go(adj, s, open_card, cut_card):
    """(cards outside their colour's largest component, number of colour components,
    number of incomplete colours whose largest component has no vacant neighbouring cell)."""
    a = adj & (s[:, None] == s[None, :])
    lab = _labels(a, jnp.ones((NC,), jnp.bool_))
    size = (lab[None, :] == lab[:, None]).sum(-1)
    maxsize = jnp.stack([jnp.where(s == c, size, 0).max() for c in range(NCOL)])
    ncomp = (lab == jnp.arange(NC)).sum()
    closed = jnp.stack([
        (maxsize[c] < NCOL) & ~(((s == c) & (size == maxsize[c])) & open_card).any()
        for c in range(NCOL)]).sum()
    in_largest = size == maxsize[s]
    blocked = (~in_largest & cut_card).sum()
    return NC - maxsize.sum(), ncomp, closed, blocked


def _open_cards(pos):
    """Per card: does it have at least one vacant in-bounds orthogonal neighbour cell."""
    occ = jnp.zeros((NCELL,), jnp.bool_).at[pos[:, 0] * G + pos[:, 1]].set(True)
    out = jnp.zeros((NC,), jnp.bool_)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx = pos[:, 0] + dx
        ny = pos[:, 1] + dy
        valid = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
        idx = jnp.clip(nx, 0, G - 1) * G + jnp.clip(ny, 0, G - 1)
        out = out | (valid & ~occ[idx])
    return out


def agent_act(obs, reset, memory, rng):
    IDS, CELL_X, CELL_Y = _consts()
    REVEAL_PREF = jnp.asarray(REVEAL_PREF_LIST)
    init = init_agent()
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    nf = obs.node_features.astype(jnp.int32)
    cards = nf[:NC]
    hints = nf[NC:]
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    col_oh = cards[:, :NCOL]
    known_now = jnp.where(col_oh.sum(-1) > 0, jnp.argmax(col_oh, -1), -1)
    locked = cards[:, 10] > 0
    pseen = cards[:, 11] > 0
    active = cards[0, 12] > 0
    substep = cards[0, 13]
    hset = hints[:, 4:8] > 0
    hplaced = hints[:, 10] > 0
    hpos = hints[:, 8:10]
    adj = (obs.adjacency_matrix[:NC, :NC] > 0) & ~jnp.eye(NC, dtype=bool)

    # ---- memory update (runs on every transition, active or not) ----
    known = jnp.where(known_now >= 0, known_now, mem['known'])
    known_mask = known >= 0
    moved = mem['have_prev'] & (pos != mem['prev_pos']).any(-1)
    pmove = moved & ~active
    imoved = mem['imoved'] | (moved & active)
    # The partner can only intend colour adjacency with cards whose colour it can know:
    # cards it observed, or cards I moved (my moves are evidence to it).
    partner_can_know = pseen | imoved
    claim_neigh = adj & partner_can_know[None, :]
    ev_new = pmove & pseen & claim_neigh.any(-1)
    ev_valid = jnp.where(moved, ev_new, mem['ev_valid'])
    ev_neigh = jnp.where(moved[:, None], claim_neigh, mem['ev_neigh'])
    pmoved = mem['pmoved'] + pmove.astype(jnp.int32)
    # Verify partner claims where I can: consistent (touches a same-colour card I know)
    # or clearly inconsistent (all claimed neighbours known to me, none matching).
    same_known = known_mask[:, None] & known_mask[None, :] & (known[:, None] == known[None, :])
    ver_ok = ev_new & known_mask & (claim_neigh & same_known).any(-1)
    ver_bad = ev_new & known_mask & ~(claim_neigh & ~known_mask[None, :]).any(-1) & ~(claim_neigh & same_known).any(-1)
    # Verify partner hint placements on cards whose colour I know.
    hint_card_now = jnp.argmax((pos[None, :, :] == hpos[:, None, :]).all(-1), axis=-1)  # [NH]
    new_partner_place = hplaced & ~mem['prev_hplaced'] & ~active & known_mask[hint_card_now]
    hint_hit_now = hset[jnp.arange(NH), jnp.clip(known[hint_card_now], 0, NCOL - 1)]
    ok_cnt = mem['ok_cnt'] + ver_ok.sum().astype(jnp.int32) + (new_partner_place & hint_hit_now).sum().astype(jnp.int32)
    bad_cnt = mem['bad_cnt'] + ver_bad.sum().astype(jnp.int32) + (new_partner_place & ~hint_hit_now).sum().astype(jnp.int32)
    new_mem = {
        'have_prev': jnp.ones((), jnp.bool_),
        'prev_pos': pos,
        'known': known,
        'ev_valid': ev_valid,
        'ev_neigh': ev_neigh,
        'pmoved': pmoved,
        'imoved': imoved,
        'ok_cnt': ok_cnt,
        'bad_cnt': bad_cnt,
        'prev_hplaced': hplaced,
    }
    unreliable = bad_cnt.astype(jnp.float32) / (ok_cnt + bad_cnt + 1).astype(jnp.float32)
    eps_move = EPS_MOVE + 0.6 * unreliable
    eps_stay = EPS_STAY + 0.4 * unreliable
    eps_hint = EPS_HINT + 0.6 * unreliable

    kcol = jnp.clip(known, 0, NCOL - 1)
    same_col = known_mask[:, None] & known_mask[None, :] & (known[:, None] == known[None, :])
    a_same = adj & same_col
    lab = _labels(a_same, known_mask)
    comp_size = jnp.where(known_mask, (lab[None, :] == lab[:, None]).sum(-1), 0)
    best_size = jnp.stack([jnp.where(known == c, comp_size, 0).max() for c in range(NCOL)])
    in_largest = known_mask & (comp_size == best_size[kcol])
    same_now = (adj & same_col).sum(-1)

    def leaf_of(x):
        rest = (lab == lab[x]) & (IDS != x)
        return _connected(a_same, rest)

    leaf = jax.vmap(leaf_of)(IDS)

    # ---- hint bookkeeping ----
    hint_card = jnp.argmax((pos[None, :, :] == hpos[:, None, :]).all(-1), axis=-1)  # [NH]
    hsize = hset.sum(-1)
    unrevealed = (hsize == 0) & ~hplaced
    revealed_unplaced = (hsize > 0) & ~hplaced

    # ---- belief ----
    rng_mc, rng_obs = jax.random.split(rng)
    stay_adj = adj & partner_can_know[None, :]
    stay_ev = pseen & (pmoved == 0) & ~known_mask & stay_adj.any(-1)
    sigma, w, hits = _sample_belief(rng_mc, known, ev_valid, ev_neigh, hint_card, hset, hplaced,
                                    stay_ev, stay_adj, eps_move, eps_stay, eps_hint)
    conn = jax.vmap(lambda s: _all_connected(adj, s))(sigma)
    p_conn = (w * conn).sum()
    p_correct = (w[:, None] * hits).sum(0)  # [NH] posterior correctness of placed hints
    s_end = 5.0 * unrevealed.sum() + 2.0 * revealed_unplaced.sum() + jnp.where(hplaced, 2.0 * p_correct - 1.0, 0.0).sum()

    # ---- end decision (substep 0) ----
    n_unknown_unlocked = (~known_mask & ~locked).sum()
    thr = 0.45 + 0.3 * jnp.clip(s_end, 0.0, 35.0) / 35.0
    thr = jnp.where(n_unknown_unlocked == 0, thr - 0.15, thr)
    do_end = legal[A_END] & (p_conn >= thr) & (s_end > 0)

    # ---- observation choice: value = P(card would reveal a needed merge) ----
    legal_obs = legal[A_OBS:A_OBS + NC]
    cand = legal_obs & ~known_mask & ~locked
    known_count = jnp.stack([(known == c).sum() for c in range(NCOL)])          # [4]
    adjk = jnp.stack([(adj & (known == c)[None, :]).any(-1) for c in range(NCOL)], -1)  # [16,4]
    s_cnt = known_count[sigma]                                                    # [NS,16]
    s_adj = adjk[IDS[None, :], sigma]                                             # [NS,16]
    reveal_need = (s_cnt >= 1) & ~s_adj
    v_obs = (w[:, None] * reveal_need).sum(0)                                    # [16]
    unk_deg = (adj & ~known_mask[None, :]).sum(-1).astype(jnp.float32)
    osc = jnp.where(cand, v_obs + 0.3 * (~pseen) + 0.05 * unk_deg - 0.001 * IDS, -1e9)
    osc = jnp.where(cand.any(), osc, jnp.where(legal_obs, 0.0, -1e9))
    obs_action = A_OBS + jnp.argmax(osc)

    # ---- move choice: heuristic prescore, then belief re-scoring of the top K ----
    nb_cards, occ = _neighbor_cards(pos)                      # [100,4]
    nb_set = (nb_cards[:, :, None] == IDS[None, None, :]).any(1)   # [100,16]
    nb_f = nb_set.astype(jnp.float32)
    same_mat = (same_col & ~jnp.eye(NC, dtype=bool)).astype(jnp.float32)    # [16,16]
    same_after = same_mat @ nb_f.T                             # [16,100]
    other_mat = (known_mask[:, None] & known_mask[None, :] & (known[:, None] != known[None, :])).astype(jnp.float32)
    other_after = other_mat @ nb_f.T
    unk_nb = (nb_f * (~known_mask).astype(jnp.float32)[None, :]).sum(-1)   # [100]
    diff_comp = same_col & (lab[:, None] != lab[None, :])     # [16,16] same colour, different comp
    other_max = jnp.where(diff_comp[:, None, :] & nb_set[None, :, :], comp_size[None, None, :], 0).max(-1)  # [16,100]
    own_adj = ((lab[:, None] == lab[None, :]) & known_mask[:, None] & ~jnp.eye(NC, dtype=bool))[:, None, :] & nb_set[None, :, :]
    own_adj = own_adj.any(-1)                                  # [16,100]
    lost_unknown = ((adj & ~known_mask[None, :])[:, None, :] & ~nb_set[None, :, :]).sum(-1).astype(jnp.float32)
    mean_pos = pos.astype(jnp.float32).mean(0)
    center_dist = jnp.abs(CELL_X - mean_pos[0]) + jnp.abs(CELL_Y - mean_pos[1])

    gain = jnp.where(other_max > 0, 8.0 + 2.0 * other_max - 2.0 * comp_size[:, None],
                     jnp.where(own_adj, 0.0, -6.0))
    sc = (gain - 8.0 * (~leaf)[:, None] + 1.0 * same_after - 1.0 * same_now[:, None]
          - 0.3 * other_after + 0.2 * unk_nb[None, :] - 0.7 * lost_unknown
          - 1.0 * jnp.minimum(pmoved, 2)[:, None] - 2.0 * ev_valid[:, None]
          - 0.05 * center_dist[None, :])
    sc = jnp.where(known_mask[:, None], sc, -8.0 - 0.7 * lost_unknown - 2.0 * ev_valid[:, None] - 0.05 * center_dist[None, :])
    legal_move = legal[A_MOVE:A_MOVE + NC * NCELL].reshape(NC, NCELL)
    sc = jnp.where(legal_move, sc, -1e9)
    flat_sc = sc.reshape(-1)
    top_vals, top_idx = lax.top_k(flat_sc, K_MOVES)
    cand_x = top_idx // NCELL
    cand_cell = top_idx % NCELL

    def eval_move(x, cell):
        newpos = pos.at[x].set(jnp.stack([cell // G, cell % G]))
        nadj = _adj_from_pos(newpos)
        open_card = _open_cards(newpos)
        cut_card = jax.vmap(lambda i: ~_connected(nadj, IDS != i))(IDS)
        mtg, ncomp, closed, blocked = jax.vmap(lambda s: _moves_to_go(nadj, s, open_card, cut_card))(sigma)
        pall = jax.vmap(lambda s: _all_connected(nadj, s))(sigma).astype(jnp.float32)
        cost = (mtg.astype(jnp.float32) + 0.3 * ncomp.astype(jnp.float32)
                + CLOSED_W * closed.astype(jnp.float32) + BLOCKED_W * blocked.astype(jnp.float32))
        return (w * cost).sum(), (w * pall).sum()

    e_comp, p_all = jax.vmap(eval_move)(cand_x, cand_cell)
    cost_now, _ = eval_move(jnp.int32(0), pos[0, 0] * G + pos[0, 1])  # no-op move: current board cost
    final = -e_comp + 2.0 * p_all + 0.02 * top_vals
    final = jnp.where(top_vals > -1e8, final, -1e9)
    bi = jnp.argmax(final)
    move_action = A_MOVE + top_idx[bi]
    move_score = final[bi]

    # ---- hint choice ----
    card_cell = pos[:, 0] * G + pos[:, 1]
    legal_place = legal[A_PLACE:A_PLACE + NH * NCELL].reshape(NH, NCELL)[:, card_cell]  # [NH,16]
    match = hset[:, kcol] & known_mask[None, :]                # [NH,16]
    settled = in_largest & (same_now >= 1)
    # Place only where nothing is hidden from the partner: the card is already
    # partner-seen, or the hint is one-colour (so the placement itself tells the colour).
    disclose = pseen[None, :] | (hsize == 1)[:, None]
    psc = jnp.where(match, 10.0 + 3.0 * settled[None, :] + 4.0 * disclose + 0.5 * same_now[None, :]
                    + 1.0 * pseen[None, :]
                    + jnp.where(hsize == 1, 2.0, jnp.where(hsize == 2, 1.0, 0.0))[:, None],
                    jnp.where(known_mask[None, :], -10.0, hsize.astype(jnp.float32)[:, None]))
    psc = jnp.where(legal_place & revealed_unplaced[:, None], psc, -1e9)
    best_p = jnp.argmax(psc.reshape(-1))
    best_p_score = psc.reshape(-1)[best_p]
    ph = best_p // NC
    py = best_p % NC
    place_action = A_PLACE + ph * NCELL + card_cell[py]
    legal_rev = legal[A_REVEAL:A_REVEAL + NH]
    rsc = jnp.where(legal_rev, REVEAL_PREF, -1e9)
    reveal_action = A_REVEAL + jnp.argmax(rsc)
    hint_action = jnp.where((best_p_score >= 17.0) | ~legal_rev.any(), place_action, reveal_action)

    # ---- assemble ----
    action = jnp.where(substep == 0, jnp.where(do_end, A_END, obs_action),
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, 0)
    action = jnp.where(legal[action], action, jnp.argmax(legal))
    action = action.astype(jnp.int32)

    trace = {
        'p_conn': p_conn.astype(jnp.float32),
        's_end': s_end.astype(jnp.float32),
        'n_known': known_mask.sum().astype(jnp.int32),
        'move_score': move_score.astype(jnp.float32),
        'pos': pos.astype(jnp.int32),
        'known': known.astype(jnp.int32),
        'pmoved': pmoved.astype(jnp.int32),
        'cost_now': cost_now.astype(jnp.float32),
        'cost_after': e_comp[bi].astype(jnp.float32),
        'unreliable': unreliable.astype(jnp.float32),
    }
    return action, new_mem, trace
