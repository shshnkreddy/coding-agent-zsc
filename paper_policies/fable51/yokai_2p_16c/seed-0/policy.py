"""Two-player Yokai heuristic policy.

Belief tracking (own observations, public hint placements, count elimination,
soft inference from partner clustering moves), greedy clustering moves via a
local potential, place-if-known-else-reveal hint economy, and an end decision
that requires certainty about every card colour.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_cluster_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NCARD = 16
NHINT = 7
G = 10
NCOL = 4
A_OBS = 1
A_MOVE = 17
A_REVEAL = 1617
A_PLACE = 1624
A_END = 2324
BIG = 1e9

W_COMP = 15.0
W_DIST = 1.0
W_ROOM = 3.0
W_IMP = 0.5
W_CMP = 0.05
W_FIX = 6.0
W_TOWARD = 0.0
W_SIG = 1.0
INCONS_MAX = 2
ANCHOR_PM = 0
LOCK_D = 2
PEN_PMOVED = 0.0
REVEAL_PRIO = (9.0, 8.9, 5.0, 4.9, 4.8, 1.0, 0.9)  # higher first: singles, doubles, triples
OBS_REL = 1.0
INFO_W = 3.0
STRICT_END = 0
PLACE_SETTLED_ONLY = False
PLACE_SETTLED_BONUS = 50.0
PEN_UNK_SEEN = 15.0
PEN_UNK = 4.0
PEN_SOFT = 1.0
PEN_PADJ = 2.0


def init_agent():
    return {
        'own': jnp.full((NCARD,), -1, dtype=jnp.int32),
        'prev_pos': jnp.zeros((NCARD, 2), dtype=jnp.int32),
        'pm': jnp.zeros((NCARD, NCARD), dtype=jnp.int32),
        'my_hints': jnp.zeros((NHINT,), dtype=jnp.int32),
        'incons': jnp.zeros((), dtype=jnp.int32),
        'pmoved': jnp.zeros((NCARD,), dtype=jnp.int32),
    }


def _popcount4(m):
    return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1) + ((m >> 3) & 1)


def _log2_4(m):
    return jnp.where(m == 1, 0, jnp.where(m == 2, 1, jnp.where(m == 4, 2, jnp.where(m == 8, 3, -1))))


def _manh(a, b):
    return jnp.abs(a[..., 0] - b[..., 0]) + jnp.abs(a[..., 1] - b[..., 1])


def _components(same):
    """Min-label propagation over a [16,16] boolean relation with self loops."""
    lab = jnp.arange(NCARD, dtype=jnp.int32)
    for _ in range(8):
        lab = jnp.min(jnp.where(same, lab[None, :], NCARD), axis=1)
    return lab


def _eliminate(cand):
    for _ in range(4):
        pc = _popcount4(cand)
        single = pc == 1
        col = _log2_4(cand)
        for c in range(NCOL):
            bit = 1 << c
            n_c = jnp.sum(single & (col == c))
            cand = jnp.where((~single) & (n_c >= 4), cand & (15 - bit), cand)
        cand = jnp.where(cand == 0, 15, cand)
        for c in range(NCOL):
            bit = 1 << c
            has = (cand & bit) > 0
            m_c = jnp.sum(has)
            cand = jnp.where(has & (m_c == 4), bit, cand)
    return cand


def _shift_or(g):
    """Cells orthogonally adjacent to any set cell of boolean grid g [10,10]."""
    up = jnp.concatenate([g[1:, :], jnp.zeros((1, G), dtype=g.dtype)], axis=0)
    down = jnp.concatenate([jnp.zeros((1, G), dtype=g.dtype), g[:-1, :]], axis=0)
    left = jnp.concatenate([g[:, 1:], jnp.zeros((G, 1), dtype=g.dtype)], axis=1)
    right = jnp.concatenate([jnp.zeros((G, 1), dtype=g.dtype), g[:, :-1]], axis=1)
    return up | down | left | right


def agent_act(obs, reset, memory, rng):
    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    card = nf[:NCARD]
    hint = nf[NCARD:]
    init = init_agent()
    own_m = jnp.where(reset, init['own'], memory['own'])
    prev_pos = jnp.where(reset, pos, memory['prev_pos'])
    pm = jnp.where(reset, init['pm'], memory['pm'])
    my_hints = jnp.where(reset, init['my_hints'], memory['my_hints'])
    incons = jnp.where(reset, init['incons'], memory['incons'])
    pmoved = jnp.where(reset, init['pmoved'], memory['pmoved'])

    active = card[0, 12] == 1
    substep = card[0, 13]
    onehot = card[:, 0:4]
    seen_now = jnp.sum(onehot, axis=1) > 0
    own = jnp.where(active & seen_now, jnp.argmax(onehot, axis=1).astype(jnp.int32), own_m)
    locked = card[:, 10] == 1
    pseen = card[:, 11] == 1

    # ---- hints ----
    hset = hint[:, 4] | (hint[:, 5] << 1) | (hint[:, 6] << 2) | (hint[:, 7] << 3)
    hplaced = hint[:, 10] == 1
    hreveal = hset > 0
    hsize = _popcount4(hset)
    hpos = hint[:, 8:10]
    match = jnp.all(pos[None, :, :] == hpos[:, None, :], axis=2)  # [7,16]
    # partner hint reliability: a placed hint whose set excludes a colour I observed myself
    own_bit = jnp.left_shift(1, jnp.maximum(own, 0))
    contra = jnp.zeros((), dtype=bool)
    for h in range(NHINT):
        m = match[h] & hplaced[h] & (own >= 0) & ((hset[h] & own_bit) == 0)
        contra = contra | (jnp.any(m) & (my_hints[h] == 0))
    trust_h = (my_hints > 0) | (~contra)
    pub = jnp.full((NCARD,), 15, dtype=jnp.int32)
    for h in range(NHINT):
        m = match[h] & hplaced[h] & trust_h[h]
        pub = jnp.where(m, pub & hset[h], pub)
    pub = jnp.where(pub == 0, 15, pub)
    cand = jnp.where(own >= 0, jnp.left_shift(1, jnp.maximum(own, 0)), pub)
    cand = _eliminate(cand)
    pc = _popcount4(cand)
    hard = jnp.where(pc == 1, _log2_4(cand), -1)
    pubknown = _popcount4(pub) == 1
    pknows = pseen | pubknown

    # ---- partner move detection and memory ----
    moved = jnp.any(pos != prev_pos, axis=1)
    nmoved = jnp.sum(moved)
    is_pm = (~active) & (nmoved == 1) & (~reset)
    X = jnp.argmax(moved)
    dX = _manh(pos, pos[X][None, :])
    PN = (dX == 1) & pknows
    newrow = jnp.where(pknows[X], PN, False).astype(jnp.int32)
    pm = jnp.where(is_pm, pm.at[X].set(newrow), pm)
    pmoved = jnp.where(is_pm, pmoved.at[X].set(1), pmoved)
    pmoved_b = pmoved > 0
    # inconsistency: partner moved a card I know next to partner-known cards I know, none matching
    pn_known = PN & (hard >= 0)
    bad_move = (is_pm & pknows[X] & (hard[X] >= 0) & jnp.any(pn_known) & jnp.all((~PN) | (hard >= 0))
                & (~jnp.any(pn_known & (hard == hard[X]))))
    incons = incons + bad_move.astype(jnp.int32)
    infer_on = incons < INCONS_MAX

    # ---- soft beliefs from partner clustering moves ----
    pmb = (pm > 0) & infer_on
    row_has = jnp.stack([jnp.any(pmb & (hard == c)[None, :], axis=1) for c in range(NCOL)], axis=1)  # [16,4]
    nkc = jnp.sum(row_has, axis=1)
    row_color = jnp.argmax(row_has, axis=1)
    rowany = jnp.any(pmb, axis=1)
    row_cnt = jnp.stack([jnp.sum(pmb & (hard == c)[None, :], axis=1) for c in range(NCOL)], axis=1)  # [16,4]
    vX = rowany & (hard < 0) & (nkc == 1)
    votes = jnp.zeros((NCARD, NCOL), dtype=jnp.int32)
    votes = votes + (jax.nn.one_hot(row_color, NCOL, dtype=jnp.int32) * (vX[:, None] * row_cnt))
    okX = rowany & (hard >= 0) & ((nkc == 0) | ((nkc == 1) & (row_color == hard)))
    for c in range(NCOL):
        # weight: X itself plus other known cards of colour c in X's row
        src = (okX & (hard == c)).astype(jnp.int32) * (1 + row_cnt[:, c])
        v = jnp.sum(pmb.astype(jnp.int32) * src[:, None], axis=0) * (hard < 0)
        votes = votes.at[:, c].add(v)
    nv = jnp.sum(votes > 0, axis=1)
    soft = jnp.where(nv == 1, jnp.argmax(votes, axis=1), -1)
    soft = jnp.where(hard >= 0, -1, soft)
    strength = jnp.max(votes, axis=1)
    weak_soft = (soft >= 0) & (strength < 2)
    eff = jnp.where(hard >= 0, hard, soft)
    # relation gain for observation choice: unknown cards related to j via pm
    unk = hard < 0
    rel_gain = jnp.sum(pmb & unk[None, :], axis=1) + jnp.sum(pmb & unk[:, None], axis=0)
    # count-based completion: if only one colour is still missing cards, the
    # remaining unknown cards must be that colour
    cnt0 = jnp.stack([jnp.sum(eff == c) for c in range(NCOL)])
    missing = jnp.maximum(4 - cnt0, 0)
    n_missing_colors = jnp.sum(missing > 0)
    n_unknown0 = jnp.sum(eff < 0)
    single_missing = (n_missing_colors == 1) & (jnp.sum(missing) == n_unknown0)
    fill_color = jnp.argmax(missing)
    soft = jnp.where((eff < 0) & single_missing, fill_color, soft)
    eff = jnp.where(hard >= 0, hard, soft)

    # ---- geometry ----
    D = _manh(pos[:, None, :], pos[None, :, :])  # [16,16]
    adj = D == 1
    eye = jnp.eye(NCARD, dtype=bool)
    ar = jnp.arange(NCARD)
    effpos = eff >= 0
    samecol = (eff[:, None] == eff[None, :]) & effpos[:, None]
    same_full = (adj & samecol) | eye
    hardpos = hard >= 0
    samehard = (hard[:, None] == hard[None, :]) & hardpos[:, None]
    same_h = (adj & samehard) | eye
    lab_h = _components(same_h)
    comp_h = jnp.sum(hardpos & (lab_h == ar))
    ncolors_h = jnp.sum(jnp.stack([jnp.any(hard == c) for c in range(NCOL)]))
    all_known = jnp.all(hardpos)
    clustered = comp_h == ncolors_h
    lab_full = _components(same_full)
    comp_eff = jnp.sum(effpos & (lab_full == ar))
    cnt_eff = jnp.stack([jnp.sum(eff == c) for c in range(NCOL)])
    eff_ok = jnp.all(effpos) & jnp.all(cnt_eff == 4) & (comp_eff == 4)
    eff_ok_strict = eff_ok & (~jnp.any(weak_soft))
    weak_colors = jnp.sum(jnp.stack([jnp.any(weak_soft & (soft == c)) for c in range(NCOL)]))
    eff_ok_mid = eff_ok & (weak_colors <= 1)

    def _anchor(lab, m):
        samecomp = lab[:, None] == lab[None, :]
        size = jnp.sum(samecomp & m[None, :], axis=1)
        haslock = jnp.any(samecomp & m[None, :] & locked[None, :], axis=1)
        haspm = jnp.any(samecomp & m[None, :] & pmoved_b[None, :], axis=1)
        score = jnp.where(m, haslock.astype(jnp.int32) * 100 + haspm.astype(jnp.int32) * ANCHOR_PM + size * 10 - lab, -1)
        best = jnp.argmax(score)
        anchor = m & (lab == lab[best]) & (jnp.max(score) >= 0)
        dmin = jnp.min(jnp.where(anchor[None, :], D, 99), axis=1)
        cost = jnp.sum(jnp.where(m & ~anchor, dmin, 0))
        return cost, anchor

    cost_full = []
    anchor_full = []
    for c in range(NCOL):
        cc, aa = _anchor(lab_full, eff == c)
        cost_full.append(cc)
        anchor_full.append(aa)
    cost_full = jnp.stack(cost_full)  # [4]
    anchor_full = jnp.stack(anchor_full)  # [4,16]
    total_full = jnp.sum(cost_full)

    # ---- move evaluation ----
    grid_card = jnp.full((G, G), -1, dtype=jnp.int32).at[pos[:, 0], pos[:, 1]].set(ar.astype(jnp.int32))

    def per_i(i):
        keep = ar != i
        g_wo = grid_card.at[pos[i, 0], pos[i, 1]].set(-1)
        occ = g_wo >= 0
        cg = jnp.zeros((NCOL, G, G), dtype=jnp.int32).at[jnp.maximum(eff, 0), pos[:, 0], pos[:, 1]].add(
            (keep & effpos).astype(jnp.int32))
        cadj = jax.vmap(_shift_or)(cg > 0)  # [4,10,10]
        free = jnp.sum(cadj & (~occ)[None], axis=(1, 2))  # [4]
        same_wo = (same_full & keep[:, None] & keep[None, :]) | eye
        lab = _components(same_wo)
        root = keep & effpos & (lab == ar)
        comp_c = jnp.stack([jnp.sum(root & (eff == c)) for c in range(NCOL)])
        cnt_c = jnp.stack([jnp.sum(keep & (eff == c)) for c in range(NCOL)])
        cost_wo, anchor_wo = _anchor(lab, keep & (eff == eff[i]) & effpos)
        return g_wo, occ, cadj, free, lab, comp_c, cnt_c, cost_wo, anchor_wo

    g_wo, occ_wo, cadj_wo, free_wo, lab_wo, comp_c_wo, cnt_c_wo, cost_wo, anchor_wo = jax.vmap(per_i)(ar)
    comp_wo = jnp.sum(comp_c_wo, axis=1)  # [16]

    cx = jnp.repeat(jnp.arange(G), G)
    cy = jnp.tile(jnp.arange(G), G)
    cells = jnp.stack([cx, cy], axis=1)  # [100,2]
    cell_to_card = _manh(cells[:, None, :], pos[None, :, :])  # [100,16]
    d_anchor = jnp.min(jnp.where(anchor_wo[:, None, :], cell_to_card[None, :, :], 99), axis=2)  # [16,100]
    has_anchor = jnp.any(anchor_wo, axis=1)  # [16]
    cost_i = jnp.where(has_anchor[:, None], jnp.maximum(d_anchor - 1, 0), 0)
    dist_wo = total_full - jnp.where(effpos, cost_full[jnp.maximum(eff, 0)], 0) + cost_wo  # [16]
    # pull toward the other same-colour cards that are not in the anchor yet
    others = samecol & (~anchor_wo) & (~eye)  # [16i,16j]
    toward = jnp.sum(others[:, None, :] * cell_to_card[None, :, :], axis=2)  # [16,100]
    dist_add = cost_i.astype(jnp.float32) + W_TOWARD * toward.astype(jnp.float32)  # [16,100]

    offs = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
    nx = cells[:, None, 0] + offs[None, :, 0]  # [100,4]
    ny = cells[:, None, 1] + offs[None, :, 1]
    inb = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
    nxc = jnp.clip(nx, 0, G - 1)
    nyc = jnp.clip(ny, 0, G - 1)
    n_k = jnp.where(inb[None], g_wo[:, nxc, nyc], -1)  # [16,100,4]
    n_valid = n_k >= 0
    n_ks = jnp.maximum(n_k, 0)
    lab_k = jnp.take_along_axis(lab_wo[:, None, :].repeat(G * G, axis=1), n_ks, axis=2)  # [16,100,4]
    eff_k = eff[n_ks]
    valid_k = n_valid & (eff_k == eff[:, None, None]) & effpos[:, None, None]
    dcount = jnp.zeros((NCARD, G * G), dtype=jnp.int32)
    for k in range(4):
        dup = jnp.zeros((NCARD, G * G), dtype=bool)
        for m in range(k):
            dup = dup | (valid_k[:, :, m] & (lab_k[:, :, m] == lab_k[:, :, k]))
        dcount = dcount + (valid_k[:, :, k] & ~dup).astype(jnp.int32)
    comp_new = comp_wo[:, None] + effpos[:, None].astype(jnp.int32) - dcount  # [16,100]

    is_c = jnp.stack([eff == c for c in range(NCOL)], axis=1).astype(jnp.int32)  # [16,4]
    comp_c_new = comp_c_wo[:, None, :] + is_c[:, None, :] * (1 - dcount)[:, :, None]  # [16,100,4]
    cnt_c_new = cnt_c_wo[:, None, :] + is_c[:, None, :]
    cadj_cell = cadj_wo[:, :, cx, cy]  # [16,4,100]
    cadj_cell = jnp.transpose(cadj_cell, (0, 2, 1))  # [16,100,4]
    occ_nk = occ_wo[:, nxc, nyc] & inb[None]  # [16,100,4]
    cadj_nk = cadj_wo[:, :, nxc, nyc]  # [16,4,100,4]
    newfree = jnp.sum((inb[None, None] & ~occ_nk[:, None] & ~cadj_nk).astype(jnp.int32), axis=3)  # [16,4,100]
    newfree = jnp.transpose(newfree, (0, 2, 1))  # [16,100,4]
    free_new = free_wo[:, None, :] - cadj_cell.astype(jnp.int32) + is_c[:, None, :] * newfree
    need = jnp.clip(3 - cnt_c_new + comp_c_new, 0, 2)
    room = jnp.sum((cnt_c_new > 0) * jnp.maximum(0, need - free_new), axis=2)  # [16,100]

    imp_k = jnp.where(effpos[:, None, None],
                      jnp.where(eff_k >= 0, (eff_k != eff[:, None, None]).astype(jnp.float32), 0.25),
                      0.0)
    imp = jnp.sum(imp_k * n_valid, axis=2)  # [16,100]
    # signalling: same-colour neighbour at destination where the partner knows exactly one side
    pk_k = pknows[n_ks]
    sig = jnp.any(valid_k & (pk_k != pknows[:, None, None]), axis=2).astype(jnp.float32)  # [16,100]

    cen = jnp.mean(pos.astype(jnp.float32), axis=0)
    dcen = jnp.sum(jnp.abs(pos.astype(jnp.float32) - cen[None]), axis=1)  # [16]
    cmp_wo = jnp.sum(dcen) - dcen
    cmp_add = jnp.sum(jnp.abs(cells.astype(jnp.float32) - cen[None]), axis=1)  # [100]

    compat = (~effpos)[:, None] | (~effpos)[None, :] | (eff[:, None] == eff[None, :])
    padj = jnp.sum(adj & pknows[None, :] & compat, axis=1)
    pen_i = jnp.where(hardpos, 0.0, jnp.where(soft >= 0, PEN_SOFT, jnp.where(pseen, PEN_UNK_SEEN, PEN_UNK)))
    pen_i = pen_i + PEN_PADJ * pseen * padj + PEN_PMOVED * pmoved_b

    # ---- stranded cards: cards that need to move but are board cut vertices ----
    def _board_components(keep):
        rel = (adj & keep[:, None] & keep[None, :]) | eye
        lab = jnp.arange(NCARD, dtype=jnp.int32)
        for _ in range(16):
            lab = jnp.min(jnp.where(rel, lab[None, :], NCARD), axis=1)
        return lab

    in_anchor_eff = jnp.any(anchor_full, axis=0)  # [16]
    want_move = effpos & (~in_anchor_eff) & (~locked)
    lab_wo_j = jax.vmap(lambda j: _board_components(ar != j))(ar)  # [16j,16]
    ncomp_wo_j = jnp.sum(((ar[None, :] != ar[:, None]) & (lab_wo_j == ar[None, :])), axis=1)
    stranded = want_move & (ncomp_wo_j > 1)  # [16]
    jj, ii = jnp.meshgrid(ar, ar, indexing='ij')
    lab_ji = jax.vmap(jax.vmap(lambda j, i: _board_components((ar != j) & (ar != i))))(jj, ii)  # [16j,16i,16]
    keep_ji = (ar[None, None, :] != jj[:, :, None]) & (ar[None, None, :] != ii[:, :, None])
    ncomp_ji = jnp.sum(keep_ji & (lab_ji == ar[None, None, :]), axis=2)  # [16j,16i]
    n_full = jnp.where(inb, grid_card[nxc, nyc], -1)  # [100,4]
    nv_ji = (n_full[None, None] >= 0) & (n_full[None, None] != jj[:, :, None, None]) & (n_full[None, None] != ii[:, :, None, None])  # [16,16,100,4]
    nfs = jnp.maximum(n_full, 0)
    labn_ji = lab_ji[:, :, nfs]  # [16,16,100,4]
    dist_ji = jnp.zeros((NCARD, NCARD, G * G), dtype=jnp.int32)
    for k in range(4):
        dup = jnp.zeros((NCARD, NCARD, G * G), dtype=bool)
        for m in range(k):
            dup = dup | (nv_ji[..., m] & (labn_ji[..., m] == labn_ji[..., k]))
        dist_ji = dist_ji + (nv_ji[..., k] & ~dup).astype(jnp.int32)
    fixed_ji = (dist_ji == ncomp_ji[:, :, None]) & (jj != ii)[:, :, None]  # [16j,16i,100]
    fix_bonus = jnp.sum(stranded[:, None, None] * fixed_ji, axis=0).astype(jnp.float32)  # [16i,100]
    n_stranded = jnp.sum(stranded)

    phi = (W_COMP * comp_new + W_DIST * (dist_wo[:, None] + dist_add) + W_ROOM * room + W_IMP * imp
           + W_CMP * (cmp_wo[:, None] + cmp_add[None, :]) + pen_i[:, None] - W_FIX * fix_bonus - W_SIG * sig)
    legal_move = legal[A_MOVE:A_MOVE + NCARD * G * G].reshape(NCARD, G * G)
    phi = jnp.where(legal_move, phi, BIG)
    best_move = jnp.argmin(phi.reshape(-1))
    move_action = A_MOVE + best_move

    # ---- observation choice ----
    unknown = ~hardpos
    obs_score = jnp.where(unknown,
                          jnp.where(~pseen, jnp.where(soft < 0, 1.6, 1.4), jnp.where(soft < 0, 1.2, 1.0))
                          + OBS_REL * rel_gain,
                          0.0)
    legal_obs = legal[A_OBS:A_OBS + NCARD]
    obs_action = A_OBS + jnp.argmax(jnp.where(legal_obs, obs_score, -1.0))

    # ---- hint action ----
    rev_unpl = hreveal & ~hplaced
    hard_s = jnp.maximum(hard, 0)
    in_S = ((hset[:, None] >> hard_s[None, :]) & 1) > 0  # [7,16]
    in_main = anchor_full[hard_s, ar]
    comp_locked = jnp.any((lab_h[:, None] == lab_h[None, :]) & locked[None, :] & samehard, axis=1)
    adj_same = jnp.any(adj & samehard, axis=1)
    could_share = ((cand[None, :] >> hard_s[:, None]) & 1) > 0  # [X, L]: L could have X's colour
    lock_safe = jnp.all((~locked)[None, :] | (~could_share) | (D <= LOCK_D) | eye, axis=1)
    lock_adj = jnp.any(locked[None, :] & (hard[None, :] == hard[:, None]) & (D == 1), axis=1)
    place_idx = A_PLACE + jnp.arange(NHINT)[:, None] * 100 + pos[None, :, 0] * 10 + pos[None, :, 1]  # [7,16]
    legal_place = legal[place_idx]
    valid = rev_unpl[:, None] & hardpos[None, :] & in_S & (~locked)[None, :] & lock_safe[None, :] & legal_place
    info = INFO_W * jnp.where(hsize[:, None] == 1, (~pseen)[None, :], pseen[None, :])
    pscore = (4.0 * in_main + 3.0 * comp_locked + 2.0 * adj_same + 2.0 * lock_adj)[None, :] + info + 0.5 * (3 - hsize)[:, None]
    size_eff = jnp.sum((lab_full[:, None] == lab_full[None, :]) & effpos[None, :], axis=1)
    settled = in_main & ((size_eff >= 2) | comp_locked)
    pscore = jnp.where(valid, pscore + PLACE_SETTLED_BONUS * settled[None, :], -BIG)
    best_p = jnp.argmax(pscore.reshape(-1))
    can_place = jnp.max(pscore) > -BIG / 2
    can_place_settled = jnp.max(pscore) > PLACE_SETTLED_BONUS - 1.0
    place_action = place_idx.reshape(-1)[best_p]
    legal_rev = legal[A_REVEAL:A_REVEAL + NHINT]
    can_reveal = jnp.any(legal_rev)
    rev_action = A_REVEAL + jnp.argmax(jnp.where(legal_rev, jnp.asarray(REVEAL_PRIO), -1.0))
    if PLACE_SETTLED_ONLY:
        can_place = can_place_settled | (can_place & ~can_reveal)
    overlap = _popcount4(cand[None, :] & hset[:, None]).astype(jnp.float32) / pc[None, :].astype(jnp.float32)
    soft_s = jnp.maximum(soft, 0)
    soft_in = (((hset[:, None] >> soft_s[None, :]) & 1) > 0) & (soft >= 0)[None, :]
    fscore = jnp.where(legal_place & rev_unpl[:, None], overlap + 0.5 * soft_in + 0.1 * pseen[None, :], -BIG)
    forced_action = place_idx.reshape(-1)[jnp.argmax(fscore.reshape(-1))]
    hint_action = jnp.where(can_place, place_action, jnp.where(can_reveal, rev_action, forced_action))

    # ---- end decision ----
    end_belief = eff_ok_strict if STRICT_END == 2 else (eff_ok_mid if STRICT_END == 1 else eff_ok)
    do_end = ((all_known & clustered) | end_belief) & legal[A_END]
    a0 = jnp.where(do_end, A_END, obs_action)
    action = jnp.where(substep == 0, a0,
                       jnp.where(substep == 1, obs_action,
                                 jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, 0).astype(jnp.int32)
    placed_now = active & (substep == 3) & (action >= A_PLACE) & (action < A_END)
    h_now = jnp.clip((action - A_PLACE) // 100, 0, NHINT - 1)
    my_hints = jnp.where(placed_now, my_hints.at[h_now].set(1), my_hints)

    next_memory = {'own': own.astype(jnp.int32), 'prev_pos': pos.astype(jnp.int32), 'pm': pm.astype(jnp.int32),
                   'my_hints': my_hints.astype(jnp.int32), 'incons': incons.astype(jnp.int32),
                   'pmoved': pmoved.astype(jnp.int32)}
    trace = {
        'n_hard': jnp.sum(hardpos).astype(jnp.int32),
        'n_soft': jnp.sum(soft >= 0).astype(jnp.int32),
        'comp_h': comp_h.astype(jnp.int32),
        'do_end': do_end.astype(jnp.int32),
        'eff_ok': eff_ok.astype(jnp.int32),
        'n_stranded': n_stranded.astype(jnp.int32),
        'eff_ok_strict': eff_ok_strict.astype(jnp.int32),
        'eff_ok_mid': eff_ok_mid.astype(jnp.int32),
        'incons': incons.astype(jnp.int32),
        'contra': contra.astype(jnp.int32),
        'hard_vec': hard.astype(jnp.int32),
        'soft_vec': soft.astype(jnp.int32),
        'pos': pos.astype(jnp.int32),
        'locked': locked.astype(jnp.int32),
    }
    return action, next_memory, trace
