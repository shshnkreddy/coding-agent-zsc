"""Two-player Yokai policy: belief tracking + partner-move inference + safe ending.

Strategy summary (per active turn):
  substep 0: end the game if the enumerated probability that every color is
             connected (over all colorings consistent with hard evidence,
             weighted by soft partner evidence) clears a score-dependent
             threshold; otherwise observe the most informative unknown card.
  substep 1: observe the next most informative unknown card.
  substep 2: move the card/cell pair that most increases the number of cards
             attached to each colour's anchor component (a locked component
             first, else the largest), among cards whose colour we know/infer.
  substep 3: place a revealed hint on a card whose color is certain and sits
             in a safe cluster (cheaper than revealing); else reveal a hint.
Partner moves are read as "moved card has the same color as the partner-seen
card it was placed next to"; that trust is calibrated online by later
observations. No private conventions: anchors are public locks or largest
clusters, and single-colour hints are placed first to publish colours.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_cluster_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NC = 16          # cards
NH = 7           # hints
NCOL = 4         # colors
G = 10           # grid
K = 6            # max uncertain cards enumerated for the end decision
NP = 720         # 6! permutations

ACT_END = 2324
ACT_OBS0 = 1
ACT_MOVE0 = 17
ACT_REVEAL0 = 1617
ACT_PLACE0 = 1624


def _perms6():
    """All 720 permutations of range(6) via the factorial number system."""
    n = jnp.arange(NP, dtype=jnp.int32)
    facts = (120, 24, 6, 2, 1, 1)
    used = jnp.zeros((NP, K), dtype=bool)
    cols = []
    for i in range(K):
        di = (n // facts[i]) % (K - i)
        free_cum = jnp.cumsum((~used).astype(jnp.int32), axis=1)
        pick = jnp.argmax((free_cum == (di[:, None] + 1)) & ~used, axis=1)
        cols.append(pick)
        used = used | (jnp.arange(K)[None, :] == pick[:, None])
    return jnp.stack(cols, axis=1).astype(jnp.int32)


def _init_pos():
    i = jnp.arange(NC, dtype=jnp.int32)
    return jnp.stack([i // 4 + 3, i % 4 + 3], axis=1)


def _cells():
    c = jnp.arange(G * G, dtype=jnp.int32)
    return jnp.stack([c // G, c % G], axis=1)


def init_agent():
    return {
        'known': jnp.full((NC,), -1, jnp.int32),
        'prev_pos': _init_pos(),
        'link': jnp.full((NC,), -1, jnp.int32),
        'pmoved': jnp.zeros((NC,), jnp.int32),
        'judged': jnp.zeros((NC,), jnp.int32),
        'nver': jnp.zeros((), jnp.int32),
        'nviol': jnp.zeros((), jnp.int32),
        'step': jnp.zeros((), jnp.int32),
    }


def _label_prop(same):
    """Connected-component labels (min index) over boolean adjacency [..., n, n]."""
    n = same.shape[-1]
    lab = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), same.shape[:-1])

    def body(_, lab):
        cand = jnp.where(same, lab[..., None, :], n)
        return jnp.minimum(lab, cand.min(-1))

    return jax.lax.fori_loop(0, n, body, lab)


def _deduce(cand):
    """Propagate the 4-per-color multiset constraint over candidate sets."""
    for _ in range(3):
        certain = cand.sum(1) == 1
        col = jnp.argmax(cand, 1)
        cnt = jnp.zeros((NCOL,), jnp.int32).at[col].add(certain.astype(jnp.int32))
        full = cnt >= 4
        c2 = jnp.where(certain[:, None], cand, cand & ~full[None, :])
        c2 = jnp.where(c2.any(1)[:, None], c2, cand)
        poss = c2.sum(0)
        must = poss == 4
        mc = (c2 & must[None, :]).sum(1)
        c3 = jnp.where((mc == 1)[:, None], c2 & must[None, :], c2)
        cand = jnp.where(c3.any(1)[:, None], c3, c2)
    return cand


def agent_act(obs, reset, memory, rng):
    init = init_agent()
    PERMS = _perms6()
    CELLS = _cells()
    AR4 = jnp.arange(NCOL)
    ARC = jnp.arange(NC)
    EYE16 = jnp.eye(NC, dtype=bool)
    REVEAL_PRIO = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
    mem = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)

    nf = obs.node_features.astype(jnp.int32)
    cards = nf[:NC]
    hints = nf[NC:]
    legal = obs.legal_actions
    active = cards[0, 12] == 1
    substep = cards[0, 13]
    pos = obs.positions.astype(jnp.int32)
    locked = cards[:, 10] == 1
    pseen = cards[:, 11] == 1
    has_col = cards[:, :4].sum(1) > 0
    own_col = jnp.where(has_col, jnp.argmax(cards[:, :4], 1), -1)
    known = jnp.where(own_col >= 0, own_col, mem['known'])

    # ---------------- partner move detection -> soft links ----------------
    moved = jnp.any(pos != mem['prev_pos'], axis=1)
    pmove = (~active) & jnp.any(moved)
    d = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    adj = d == 1
    # link target: a partner-seen neighbour of the destination (the partner attaches to
    # cards it knows); among those prefer cards I know. No partner-seen neighbour -> no link.
    nb_score = adj.astype(jnp.int32) * (
        1 + 4 * pseen[None, :].astype(jnp.int32) + 2 * (known[None, :] >= 0).astype(jnp.int32)
        + mem['pmoved'][None, :])
    nb_best = nb_score.max(1)
    target = jnp.where(nb_best >= 5, jnp.argmax(nb_score, 1), -1)
    upd = pmove & moved
    link = jnp.where(upd, jnp.where(pseen, target, -1), mem['link'])
    pmoved = jnp.where(upd, 1, mem['pmoved'])

    # ---------------- candidate colors (hard evidence) ----------------
    cand = jnp.where(known[:, None] >= 0, AR4[None, :] == known[:, None], True)
    hset = hints[:, 4:8] > 0
    hrev = hset.any(1)
    hplaced = hints[:, 10] == 1
    hpos = hints[:, 8:10]
    match = jnp.all(pos[:, None, :] == hpos[None, :, :], -1) & hplaced[None, :]  # [16,7]
    restrict = jnp.all(jnp.where(match[:, :, None], hset[None, :, :], True), axis=1)
    cand2 = cand & restrict
    cand = jnp.where(cand2.any(1)[:, None], cand2, cand)
    cand = _deduce(cand)
    certain = cand.sum(1) == 1
    ccol = jnp.where(certain, jnp.argmax(cand, 1), -1)

    # ---------------- link propagation (soft colors) ----------------
    Lm = (link[:, None] == ARC[None, :]) & (link[:, None] >= 0)
    Lm = Lm | Lm.T
    eff = ccol
    for _ in range(4):
        support = (Lm[:, :, None] & (eff[None, :, None] == AR4[None, None, :])).any(1)
        allowed = support & cand
        new = jnp.where(allowed.any(1), jnp.argmax(allowed, 1), -1)
        eff = jnp.where(certain, ccol, new)
    linkcol = jnp.where(certain, -1, eff)

    # ---------------- link trust calibration ----------------
    tgt = jnp.where(link >= 0, link, 0)
    both = (link >= 0) & certain & certain[tgt]
    newj = both & (mem['judged'] == 0)
    agree = ccol == ccol[tgt]
    nver = mem['nver'] + (newj & agree).sum()
    nviol = mem['nviol'] + (newj & ~agree).sum()
    judged = jnp.where(newj, 1, mem['judged'])
    p_link = (0.9 * 2.0 + nver) / (2.0 + nver + nviol)

    # ---------------- components on current board (eff colors) ----------------
    same = adj & (eff[:, None] == eff[None, :]) & (eff[:, None] >= 0)
    lab = _label_prop(same | EYE16)
    samelab = lab[:, None] == lab[None, :]
    comp_size = samelab.sum(1)
    comp_all_certain = jnp.all(jnp.where(samelab, certain[None, :], True), 1)
    complete = (comp_size == 4) & comp_all_certain
    other_locked = (locked[None, :] & (eff[None, :] == eff[:, None]) & (eff[:, None] >= 0)
                    & ~samelab).any(1)

    # ---------------- success probability of ending now ----------------
    unc = ~certain
    k = unc.sum()
    order = jnp.argsort(jnp.where(unc, 0, 100) + ARC)
    idx = order[:K]
    cnt_c = jnp.zeros((NCOL,), jnp.int32).at[jnp.where(certain, ccol, 0)].add(certain.astype(jnp.int32))
    miss = 4 - cnt_c
    cum = jnp.cumsum(miss)
    jj = jnp.arange(K)
    items = (jj[:, None] >= cum[None, :]).sum(1)
    items = jnp.where(jj < k, jnp.clip(items, 0, 3), -1)
    slot_items = items[PERMS]                      # [720, K]
    jk = (jj < k)[None, :]
    valid = jnp.all((slot_items >= 0) == jk, 1)
    cc = cand[idx]                                 # [K, 4]
    ok = jnp.take_along_axis(jnp.broadcast_to(cc[None], (NP, K, NCOL)),
                             jnp.clip(slot_items, 0, 3)[:, :, None], axis=2)[..., 0]
    valid = valid & jnp.all(ok | ~jk, 1)
    slot_col = jnp.where(jk, slot_items, ccol[idx][None, :])
    full = jnp.broadcast_to(ccol, (NP, NC)).at[jnp.arange(NP)[:, None], idx[None, :]].set(slot_col)
    lc = linkcol[idx]
    w_slot = jnp.where((lc[None, :] >= 0) & jk,
                       jnp.where(slot_col == lc[None, :], p_link, (1.0 - p_link) / 3.0), 1.0)
    w = jnp.prod(w_slot, 1) * valid.astype(jnp.float32)
    srt = jnp.argsort(full, axis=1)
    gp = pos[srt].reshape(NP, NCOL, 4, 2)
    gd = jnp.abs(gp[:, :, :, None, :] - gp[:, :, None, :, :]).sum(-1)
    A = ((gd == 1) | jnp.eye(4, dtype=bool)).astype(jnp.int32)
    R = A @ A @ A
    conn = jnp.all(R > 0, axis=(1, 2, 3))
    num = (w * conn).sum()
    den = w.sum()
    p_succ = jnp.where((den > 0) & (k <= K), num / jnp.maximum(den, 1e-9), 0.0)

    # ---------------- current score if ended now ----------------
    U = (~hrev).sum()
    Rn = (hrev & ~hplaced).sum()
    hcard = jnp.argmax(match, 0)
    hcol = eff[hcard]
    hset_sz = hset.sum(1)
    correct = hplaced & ((hcol < 0) | hset[jnp.arange(NH), jnp.clip(hcol, 0, 3)])
    Cn = (hplaced & correct).sum()
    In = (hplaced & ~correct).sum()
    S = 5 * U + 2 * Rn + Cn - In
    thr = jnp.clip(1.0 - 4.5 / jnp.maximum(S, 1), 0.4, 0.85)
    do_end = (p_succ >= thr) & legal[ACT_END]

    # ---------------- observation choice ----------------
    r1, r2, r3 = jax.random.split(rng, 3)
    legal_obs = legal[ACT_OBS0:ACT_OBS0 + NC]
    nobody = (~pseen) & (known < 0)
    pm = pmoved == 1
    unresolved = ~certain & (linkcol < 0)
    obs_score = (jnp.where(certain, -5.0, 0.0)
                 + 3.5 * (pm & (link >= 0) & unresolved)      # resolves partner's cluster via link
                 + 3.0 * nobody
                 + 2.0 * (pseen & unresolved & ~pm)
                 + 2.0 * (pm & (link < 0) & unresolved)
                 + 1.0 * (~certain & (linkcol >= 0))
                 + 0.01 * jax.random.uniform(r1, (NC,)))
    obs_score = jnp.where(legal_obs, obs_score, -1e9)
    obs_action = ACT_OBS0 + jnp.argmax(obs_score)

    # ---------------- move choice (incremental component scoring) ----------------
    legal_move = legal[ACT_MOVE0:ACT_MOVE0 + NC * 100].reshape(NC, 100)
    keep = ~EYE16
    same_wo = same[None] & keep[:, :, None] & keep[:, None, :]           # [16 removed, 16, 16]
    lab_wo = _label_prop(same_wo | EYE16[None])                          # [16, 16]
    colored = eff >= 0
    pairs_same_total = same.sum() // 2
    pairs_same_wo = pairs_same_total - same.sum(1)                        # [16]
    dc = jnp.abs(CELLS[:, None, :] - pos[None, :, :]).sum(-1)             # [100, 16]
    nbc = dc == 1
    nb_same = (nbc[None] & (eff[None, None, :] == eff[:, None, None]) & colored[:, None, None]
               & keep[:, None, :])                                        # [16, 100, 16]
    nb_diff = (nbc[None] & (eff[None, None, :] != eff[:, None, None]) & colored[None, None, :]
               & colored[:, None, None] & keep[:, None, :])
    same_lab_wo = lab_wo[:, :, None] == lab_wo[:, None, :]                # [16, 16, 16]
    lower = ARC[:, None] > ARC[None, :]                                   # j > j'
    dup = (nb_same[:, :, :, None] & nb_same[:, :, None, :] & same_lab_wo[:, None, :, :]
           & lower[None, None]).any(-1)                                   # [16, 100, 16]
    nodup = nb_same & ~dup                                                # [16, 100, 16] one rep per nb comp
    pairs_after = pairs_same_wo[:, None] + nb_same.sum(-1)
    diff_after = nb_diff.sum(-1)
    centroid_wo = (pos.sum(0)[None, :] - pos) / (NC - 1.0)                # [16, 2]
    spread = jnp.abs(CELLS[None, :, :] - centroid_wo[:, None, :]).sum(-1)  # [16, 100]
    # --- anchor objective: cards attached to each colour's anchor component ---
    samecol = (eff[:, None] == eff[None, :]) & colored[:, None] & keep    # [i, j]: j same colour as i, j != i
    memb = same_lab_wo & samecol[:, None, :]                              # [i, j, j']: j' in comp(j) w/o i
    size_wo = memb.sum(-1)                                                # [i, j]
    lock_wo = (memb & locked[None, None, :]).sum(-1)                      # [i, j]
    key_wo = 100 * lock_wo + size_wo
    merged_j = (nb_same[:, :, None, :] & same_lab_wo[:, None, :, :]).any(-1)   # [i, x, j] comp(j) touches x
    merged_size = 1 + (nodup * size_wo[:, None, :]).sum(-1)               # [i, x]
    merged_lock = (nodup * lock_wo[:, None, :]).sum(-1)
    key_merged = 100 * merged_lock + merged_size
    key_non = jnp.where(samecol[:, None, :] & ~merged_j, key_wo[:, None, :], 0).max(-1)   # [i, x]
    anchor_after = jnp.maximum(key_merged, key_non) % 100
    lock_cur = (samelab & locked[None, :]).sum(1)                         # [j]
    key_cur = jnp.where(colored, 100 * lock_cur + comp_size, 0)
    anchor_c = jnp.where(eff[None, :] == AR4[:, None], key_cur[None, :], 0).max(1) % 100   # [4]
    total_cur = anchor_c.sum()
    anchor_obj = jnp.where(colored[:, None],
                           total_cur - anchor_c[jnp.clip(eff, 0, 3)][:, None] + anchor_after,
                           total_cur)
    # --- room: free cells around the moved card while its comp is incomplete ---
    occ_nb = (nbc[None] & keep[:, None, :]).sum(-1)                       # [i, x]
    offgrid = ((CELLS[:, 0] == 0) + (CELLS[:, 0] == G - 1) + (CELLS[:, 1] == 0) + (CELLS[:, 1] == G - 1))
    free_nb = 4 - occ_nb - offgrid[None, :]
    room = jnp.minimum(free_nb, jnp.maximum(4 - merged_size, 0)) * colored[:, None]
    # --- unstick: reward moves that give a stuck (cut-vertex) unattached card a 2nd path ---
    anchor_key_c = jnp.where(eff[None, :] == AR4[:, None], key_cur[None, :], 0).max(1)   # [4]
    in_anchor = colored & (key_cur == anchor_key_c[jnp.clip(eff, 0, 3)])
    anchor_adj = (nbc[None, :, :] & in_anchor[None, None, :]
                  & (eff[None, None, :] == AR4[:, None, None])).any(-1)   # [c, x] cell touches anchor of c
    attach_ok = (legal_move & anchor_adj[jnp.clip(eff, 0, 3)]).any(1)     # [s]
    stuck = colored & ~locked & ~in_anchor & ~attach_ok                   # [s]
    adj_wo = adj[None] & keep[:, :, None] & keep[:, None, :]
    lab_b_wo = _label_prop(adj_wo | EYE16[None])                          # [s, j] board comps without s
    nb_j = nbc[None, None] & keep[None, :, None, :] & keep[:, None, None, :]   # [s, i, x, j]
    labs = lab_b_wo[:, None, None, :]
    lmin = jnp.where(nb_j, labs, 99).min(-1)
    lmax = jnp.where(nb_j, labs, -1).max(-1)                              # [s, i, x]
    bridges = (lmin != lmax) & (lmax >= 0)                                # [s, i, x] i at x joins 2 comps of board\s
    # moving the stuck card itself between two components keeps it a cut vertex: exclude s == i
    unstick = (bridges & stuck[:, None, None] & keep[:, :, None]).sum(0)   # [i, x]
    # i at x would itself become a bridge (cut vertex) if x's neighbours lie in different comps of board\i
    self_bridge = jnp.diagonal(bridges, axis1=0, axis2=1).T               # [i, x]
    pen = (6.0 * (~colored) + 6.0 * (pm & ~certain) + 1.0 * (~certain & colored))
    mscore = (10.0 * anchor_obj + 3.0 * unstick + 0.6 * room + 0.5 * pairs_after - 0.7 * diff_after
              - 1.0 * self_bridge - 0.05 * spread - pen[:, None]
              + 0.01 * jax.random.uniform(r2, (NC, 100)))
    mscore = jnp.where(legal_move, mscore, -1e9)
    move_action = ACT_MOVE0 + jnp.argmax(mscore.reshape(-1))

    # ---------------- hint choice ----------------
    legal_place = legal[ACT_PLACE0:ACT_PLACE0 + NH * 100].reshape(NH, 100)
    legal_reveal = legal[ACT_REVEAL0:ACT_REVEAL0 + NH]
    cell_idx = pos[:, 0] * G + pos[:, 1]
    cell_card = jnp.full((100,), -1, jnp.int32).at[cell_idx].set(ARC)
    cc_safe = jnp.clip(cell_card, 0, NC - 1)
    has_card = cell_card >= 0
    col_at = jnp.where(has_card, ccol[cc_safe], -1)                       # [100]
    in_set = hset[:, jnp.clip(col_at, 0, 3)] & (col_at >= 0)[None, :]      # [7, 100]
    size_at = comp_size[cc_safe]
    complete_at = complete[cc_safe]
    olock_at = other_locked[cc_safe]
    pseen_at = pseen[cc_safe]
    single = (hset_sz == 1)[:, None]
    safe_multi = (size_at >= 2)[None, :] | complete_at[None, :]
    pscore = (10.0 + 6.0 * complete_at[None, :] + 2.0 * size_at[None, :]
              + 4.0 * single + 2.0 * (single & ~pseen_at[None, :])
              + 3.0 * (~single & pseen_at[None, :])
              - 20.0 * olock_at[None, :])
    # locking a component that hangs off the board through a single outside card can deadlock later
    outside_touch = ((samelab[:, :, None] & adj[None, :, :]) & ~samelab[:, None, :]).any(1).sum(1)   # [c]
    well_attached = (outside_touch >= 2)[cc_safe]
    pvalid = in_set & has_card[None, :] & (single | safe_multi) & well_attached[None, :] & legal_place
    pscore = jnp.where(pvalid, pscore, -1e9)
    best_place = jnp.argmax(pscore.reshape(-1))
    can_place = pscore.reshape(-1)[best_place] > 0
    rscore = jnp.where(legal_reveal, -REVEAL_PRIO, -1e9)
    reveal_action = ACT_REVEAL0 + jnp.argmax(rscore)
    place_action = ACT_PLACE0 + best_place
    # forced last placement (all revealed, 6 placed): choose best guess by eff colour
    any_reveal = legal_reveal.any()
    eff_at = jnp.where(has_card, eff[cc_safe], -1)
    guess_ok = hset[:, jnp.clip(eff_at, 0, 3)] & (eff_at >= 0)[None, :]
    gscore = jnp.where(legal_place, 1.0 * guess_ok + 0.5 * (hset_sz[:, None] == 3) + 0.1 * (hset_sz[:, None] == 2)
                       + 0.01 * jax.random.uniform(r3, (NH, 100)), -1e9)
    guess_action = ACT_PLACE0 + jnp.argmax(gscore.reshape(-1))
    hint_action = jnp.where(can_place, place_action, jnp.where(any_reveal, reveal_action, guess_action))

    # ---------------- dispatch by substep ----------------
    act0 = jnp.where(do_end, ACT_END, obs_action)
    action = jnp.where(substep == 0, act0,
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, 0)
    fallback = jnp.argmax(legal)
    action = jnp.where(legal[action], action, fallback).astype(jnp.int32)

    new_mem = {
        'known': known.astype(jnp.int32),
        'prev_pos': pos,
        'link': link.astype(jnp.int32),
        'pmoved': pmoved.astype(jnp.int32),
        'judged': judged.astype(jnp.int32),
        'nver': nver.astype(jnp.int32),
        'nviol': nviol.astype(jnp.int32),
        'step': mem['step'] + 1,
    }
    trace = {
        'p_succ': p_succ.astype(jnp.float32),
        'score_now': S.astype(jnp.int32),
        'n_uncertain': k.astype(jnp.int32),
        'p_link': p_link.astype(jnp.float32),
        'do_end': do_end,
        'can_place': can_place,
        'eff': eff.astype(jnp.int32),
        'certain': certain,
        'anchor_sum': total_cur.astype(jnp.int32),
        'n_attach_legal': (legal_move & (anchor_after > anchor_c[jnp.clip(eff, 0, 3)][:, None])
                           & colored[:, None]).sum().astype(jnp.int32),
        'n_unattached': (colored & ~in_anchor).sum().astype(jnp.int32),
        'stuck_n': stuck.sum().astype(jnp.int32),
        'unstick_legal': (unstick * legal_move).sum().astype(jnp.int32),
        'best_unstick': unstick.reshape(-1)[jnp.argmax(mscore.reshape(-1))].astype(jnp.int32),
    }
    return action, new_mem, trace
