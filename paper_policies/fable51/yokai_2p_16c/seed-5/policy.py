"""Yokai 2p cluster-building policy (heuristic, JAX, fixed shapes).

Ideas:
  * Colors known to this seat come from own observations (perfect memory in obs),
    from trusted partner hint placements (hint set on a locked card the partner
    had observed), and from counting deductions.
  * Moves: pick (card, cell) maximising reduction of same-color component count,
    preferring cells adjacent to the color's anchor component (locked card or
    largest component). Unknown / partner-handled cards are left alone.
  * Hints: place a revealed hint on a known, unlocked card of matching color that
    sits in its color's anchor component (safe lock); otherwise reveal.
  * End: only when every color assignment of the (<=6) unknown cards that is
    consistent with the color multiset and candidate sets yields a connected
    board.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'yokai_cluster_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NC = 4
NCARD = 16
NH = 7
G = 10
K = 6
NA = 2325
NPERM = 720
W_COMPACT = 0.3
W_UNBLOCK = 1.0
W_ADJKNOWN = 0.0
REQUIRE_VERIFY = False
ANCHOR_SAFE = False
W_PUNSEEN = 1.5
W_OBSEND = 8.0
SINGLE_EARLY = True
W_PMOVED = 0.0
W_FIT = 0.0
FRAC_END = 0.7


def make_perms():
    """All permutations of range(6) via Lehmer-code decoding -> int32 [720,6]."""
    idx = jnp.arange(NPERM, dtype=jnp.int32)
    rem = idx
    digits = []
    for f in (120, 24, 6, 2, 1, 1):
        digits.append(rem // f)
        rem = rem % f
    avail = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32)[None, :], (NPERM, K))
    perm = []
    posk = jnp.arange(K, dtype=jnp.int32)[None, :]
    for k in range(K):
        d = digits[k]
        chosen = jnp.take_along_axis(avail, d[:, None], axis=1)[:, 0]
        perm.append(chosen)
        avail = jnp.where(posk >= d[:, None], jnp.roll(avail, -1, axis=1), avail)
    return jnp.stack(perm, 1)


def init_agent():
    return {
        'prev_pos': jnp.zeros((NCARD, 2), jnp.int32),
        'partner_moved': jnp.zeros((NCARD,), jnp.int32),
        'my_moved': jnp.zeros((NCARD,), jnp.int32),
        'assoc': jnp.zeros((NCARD, NCARD), jnp.int32),
        'my_hint': jnp.zeros((NH,), jnp.int32),
    }


def grid_adj(pos):
    """pos [...,16,2] -> bool [...,16,16] orthogonal adjacency, no self loops."""
    d = jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]).sum(-1)
    return d == 1


def components(ac):
    """ac [...,16,16] bool symmetric adjacency -> min-label components [...,16]."""
    lab = jnp.broadcast_to(jnp.arange(NCARD, dtype=jnp.int32), ac.shape[:-1])
    for _ in range(4):
        cand = jnp.where(ac, lab[..., None, :], NCARD)
        lab = jnp.minimum(lab, cand.min(-1))
    return lab


def deduce(cand):
    """cand [16,4] bool candidate colors -> tightened by counting (4 per color)."""
    for _ in range(3):
        exact = cand.sum(-1) == 1
        n_exact = (cand & exact[:, None]).sum(0)  # [4]
        full = n_exact >= 4
        cand = cand & ~(full[None, :] & ~exact[:, None])
        n_poss = cand.sum(0)
        forced = n_poss <= 4
        force_col = cand & forced[None, :]
        force_card = force_col.any(-1) & ~exact
        cand = jnp.where(force_card[:, None], force_col, cand)
        empty = cand.sum(-1) == 0
        cand = cand | empty[:, None]
    return cand


def agent_act(obs, reset, memory, rng):
    PERMS = make_perms()
    NB_OFF = jnp.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
    ARANGE16 = jnp.arange(NCARD, dtype=jnp.int32)
    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    active = nf[0, 12] > 0
    substep = nf[0, 13]

    # ---- memory: track partner moves -------------------------------------
    prev_pos = jnp.where(reset, pos, memory['prev_pos'])
    moved_now = (pos != prev_pos).any(-1)
    pm = jnp.where(reset, 0, memory['partner_moved'])
    mm = jnp.where(reset, 0, memory['my_moved'])
    partner_moved_now = moved_now & ~active
    pm = pm + jnp.where(partner_moved_now, 1, 0)
    mm = mm + jnp.where(moved_now & active, 1, 0)
    A = grid_adj(pos)  # [16,16]
    assoc = jnp.where(reset, 0, memory['assoc'])
    assoc = jnp.maximum(assoc, jnp.where(partner_moved_now[:, None] & A, 1, 0))
    new_memory = {'prev_pos': pos, 'partner_moved': pm, 'my_moved': mm, 'assoc': assoc}

    # ---- knowledge --------------------------------------------------------
    onehot = nf[:NCARD, 0:4] > 0
    observed = onehot.any(-1)
    locked = nf[:NCARD, 10] > 0
    pseen = nf[:NCARD, 11] > 0
    hint_sets = nf[NCARD:, 4:8] > 0  # [7,4]
    hint_placed = nf[NCARD:, 10] > 0
    hint_pos = nf[NCARD:, 8:10]
    revealed = hint_sets.any(-1)
    match = hint_placed[:, None] & (hint_pos[:, None, :] == pos[None, :, :]).all(-1)  # [7,16]
    card_hint_set = (match[:, :, None] & hint_sets[:, None, :]).any(0)  # [16,4]
    has_hint = match.any(0)
    my_hint = jnp.where(reset, 0, memory['my_hint'])  # [7] hints placed by this seat
    partner_hint_card = (match & (my_hint[:, None] == 0)).any(0)  # [16] card carries a partner-placed hint
    correct_here = (onehot & card_hint_set).any(-1)
    violation = (observed & has_hint & ~correct_here).any()
    n_verified = (observed & partner_hint_card & correct_here).sum()
    trusted = has_hint & pseen & ~observed & ~violation & ((n_verified >= 1) | (not REQUIRE_VERIFY))
    cand = jnp.where(observed[:, None], onehot, jnp.where(trusted[:, None], card_hint_set, True))
    cand = deduce(cand)
    known = cand.sum(-1) == 1
    col = jnp.argmax(cand, -1).astype(jnp.int32)  # valid where known
    unknown = ~known
    u = unknown.sum()
    # soft colors for unknown cards from partner-move associations
    assoc_sym = (assoc + assoc.T) > 0
    votes = (assoc_sym[:, :, None] & known[None, :, None] & (col[None, :, None] == jnp.arange(NC)[None, None, :])).sum(1)  # [16,4]
    soft_known = unknown & (votes.max(-1) > 0) & ((votes > 0).sum(-1) == 1)
    soft_col = jnp.argmax(votes, -1).astype(jnp.int32)
    USE_SOFT = False
    known_s = known | (soft_known & USE_SOFT)
    col_s = jnp.where(known | (not USE_SOFT), col, soft_col)
    unknown_s = ~known_s

    occ = jnp.full((G, G), -1, jnp.int32).at[pos[:, 0], pos[:, 1]].set(ARANGE16)

    # ---- end check: enumerate assignments of unknown cards -----------------
    slots = jnp.argsort(known.astype(jnp.int32))[:K]  # unknown first
    slot_valid = jnp.arange(K) < u
    n_known_c = (cand & known[:, None]).sum(0)
    miss_c = NC - n_known_c
    cum = jnp.cumsum(miss_c)
    missing = jnp.minimum((jnp.arange(K)[:, None] >= cum[None, :]).sum(-1), NC - 1)  # [6]
    assign = missing[PERMS]  # [720,6]
    col_full = jnp.broadcast_to(jnp.where(known, col, 0)[None, :], (PERMS.shape[0], NCARD))
    valid = jnp.ones((PERMS.shape[0],), bool)
    for k in range(K):
        sk = slots[k]
        col_full = col_full.at[:, sk].set(jnp.where(slot_valid[k], assign[:, k], col_full[:, sk]))
        valid = valid & (~slot_valid[k] | cand[sk, assign[:, k]])
    same = col_full[:, :, None] == col_full[:, None, :]
    lab = components(A[None] & same)
    roots = (lab == ARANGE16[None, :]).sum(-1)
    connected = roots == NC
    n_valid = valid.sum()
    n_conn = (valid & connected).sum()
    frac_conn = n_conn / jnp.maximum(n_valid, 1)
    end_strict = (u <= K) & (n_valid > 0) & (n_conn == n_valid)
    n_unrev = (~revealed).sum()
    n_rev_unpl = (revealed & ~hint_placed).sum()
    placed_ok = has_hint & (~observed | (onehot & card_hint_set).any(-1))
    n_c = placed_ok.sum()
    n_i = (has_hint & ~placed_ok).sum()
    s_now = (5 * n_unrev + 2 * n_rev_unpl + n_c - n_i).astype(jnp.float32)
    ev_end = frac_conn * s_now
    ev_cont = jnp.maximum(6.5, 0.9 * (s_now - 4.0))
    end_ok = end_strict | ((u <= K) & (n_valid > 0) & (frac_conn >= FRAC_END) & (ev_end >= ev_cont))

    # ---- known_s-card component structure ------------------------------------
    same_known_s = known_s[:, None] & known_s[None, :] & (col_s[:, None] == col_s[None, :])
    lab_all = components(A & same_known_s)
    comp_size = ((lab_all[:, None] == lab_all[None, :]) & known_s[None, :] & known_s[:, None]).sum(-1)
    comps_all = jnp.zeros((NC,), jnp.int32).at[col_s].add(jnp.where(known_s & (lab_all == ARANGE16), 1, 0))
    # anchor card per color: locked known_s card, else card in largest component
    prio = jnp.where(known_s, (locked & known_s) * 100 + comp_size, -1).astype(jnp.float32) - 0.01 * ARANGE16
    prio_c = jnp.where((col_s[None, :] == jnp.arange(NC)[:, None]) & known_s[None, :], prio[None, :], -1e9)  # [4,16]
    anchor_card = jnp.argmax(prio_c, -1)  # [4]
    anchor_exists = prio_c.max(-1) > -1e8
    anchor_lab = lab_all[anchor_card]  # [4]
    anchor_locked = (locked & known_s)[anchor_card]

    # components without each card i
    mask_wo = known_s[None, :] & (ARANGE16[:, None] != ARANGE16[None, :])  # [16 (i), 16 (j)]
    same_wo = mask_wo[:, :, None] & mask_wo[:, None, :] & (col_s[None, :, None] == col_s[None, None, :])
    lab_wo = components(A[None] & same_wo)  # [16,16]
    roots_wo = mask_wo & (lab_wo == ARANGE16[None, :])
    comps_wo_ci = jnp.where(roots_wo & (col_s[None, :] == col_s[:, None]), 1, 0).sum(-1)  # [16] comps of col(i) w/o i

    # neighbor cards of each cell
    xs = jnp.arange(G)[:, None, None]
    ys = jnp.arange(G)[None, :, None]
    nx = xs + NB_OFF[None, None, :, 0]
    ny = ys + NB_OFF[None, None, :, 1]
    inb = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
    nb_card = jnp.where(inb, occ[jnp.clip(nx, 0, G - 1), jnp.clip(ny, 0, G - 1)], -1)  # [10,10,4]
    nb_occ = nb_card >= 0
    nb_card_c = jnp.clip(nb_card, 0, NCARD - 1)
    n_occ_nb = nb_occ.sum(-1)  # [10,10]
    # per moving card i
    nb_same = nb_occ[None] & (nb_card[None] != ARANGE16[:, None, None, None]) & known_s[nb_card_c][None] \
        & (col_s[nb_card_c][None] == col_s[:, None, None, None]) & known_s[:, None, None, None]  # [16,10,10,4]
    nb_lab = jnp.where(nb_same, lab_wo[ARANGE16[:, None, None, None], nb_card_c[None]], -1)  # [16,10,10,4]
    merged = (nb_lab[..., None] == ARANGE16).any(-2).sum(-1)  # [16,10,10]
    new_comps = comps_wo_ci[:, None, None] + 1 - merged
    gain = jnp.where(known_s[:, None, None], comps_all[col_s][:, None, None] - new_comps, 0)
    in_anchor = known_s & (lab_all == anchor_lab[col_s])
    nb_anchor = (nb_same & (lab_all[nb_card_c][None] == anchor_lab[col_s][:, None, None, None])).any(-1)  # [16,10,10]
    anchor_bonus = jnp.where(nb_anchor & ~in_anchor[:, None, None],
                             jnp.where(anchor_locked[col_s], 4.0, 2.0)[:, None, None], 0.0)
    move_pen = jnp.where(unknown_s, -2.0 - 3.0 * pseen - 3.0 * (pm > 0), 0.0)[:, None, None]
    # blocking of incomplete anchor components (free cells adjacent to anchor comps are precious)
    n_known_s_c = (known_s[:, None] & (col_s[:, None] == jnp.arange(NC)[None, :])).sum(0)
    incomplete = anchor_exists & ((comps_all > 1) | (n_known_s_c < NC))  # [4]
    nb_in_anchor_c = nb_occ[None, None] & (nb_card[None, None] != ARANGE16[:, None, None, None, None]) \
        & known_s[nb_card_c][None, None] & (col_s[nb_card_c][None, None] == jnp.arange(NC)[None, :, None, None, None]) \
        & (lab_all[nb_card_c][None, None] == anchor_lab[None, :, None, None, None])  # [16(i),4(c),10,10,4]
    adj_c = nb_in_anchor_c.any(-1)  # [16,4,10,10]
    empty_cell = occ < 0
    nfree = jnp.where(anchor_exists, (empty_cell[None] & (nb_in_anchor_c.any(0)).any(-1)).sum((-1, -2)), 0)
    wblock = jnp.where(anchor_exists & (comps_all > 1),
                       jnp.where(nfree == 0, 3.0, jnp.where(nfree == 1, 1.0, 0.0)), 0.0)  # [4]
    other_c = unknown_s[:, None] | (col_s[:, None] != jnp.arange(NC)[None, :])  # [16,4]
    blocks_dest = (adj_c * (wblock[None, :, None, None] * other_c[:, :, None, None])).sum(1)  # [16,10,10]
    blocks_cur = blocks_dest[ARANGE16, pos[:, 0], pos[:, 1]]  # [16]
    noise = jax.random.uniform(rng, (NCARD, G, G)) * 0.01
    mscore = 10.0 * gain + anchor_bonus + move_pen + W_COMPACT * n_occ_nb[None] + noise \
        + W_PUNSEEN * ((gain >= 1) & (~pseen)[:, None, None]) \
        + W_UNBLOCK * (blocks_cur[:, None, None] - blocks_dest)
    move_legal = legal[17:1617].reshape(NCARD, G, G)
    mscore = jnp.where(move_legal, mscore, -1e9)
    move_action = 17 + jnp.argmax(mscore.reshape(-1)).astype(jnp.int32)

    # ---- observe choice ----------------------------------------------------
    incomplete_col = (n_known_c < NC)  # [4] colors with unknown cards remaining
    adj_known = (A[:, :, None] & known[None, :, None] & (col[None, :, None] == jnp.arange(NC)[None, None, :])
                 & incomplete_col[None, None, :]).any(1).sum(-1).astype(jnp.float32)  # distinct incomplete colors adjacent
    # end-aware targeting: probability that observing slot k makes the end check certain
    va = valid[:, None, None] & (assign[:, :, None] == jnp.arange(NC)[None, None, :])  # [720,6,4]
    n_va = va.sum(0)  # [6,4]
    n_ca = (va & connected[:, None, None]).sum(0)
    cert = (n_va > 0) & (n_ca == n_va)
    p_end_after = (cert * n_va).sum(-1) / jnp.maximum(n_valid, 1)  # [6]
    # fit: how often (over valid assignments) the slot's color matches a known neighbor's color
    nbcols = (A[:, :, None] & known[None, :, None] & (col[None, :, None] == jnp.arange(NC)[None, None, :])).any(1)  # [16,4]
    fit = (va & nbcols[slots][None]).sum((0, 2)) / jnp.maximum(n_valid, 1)  # [6]
    obs_end_bonus = jnp.zeros((NCARD,), jnp.float32).at[slots].set(
        jnp.where(slot_valid & (u <= K), p_end_after + W_FIT * (1.0 - fit), 0.0).astype(jnp.float32))
    have_gain = jnp.where(move_legal, gain, -99).max() >= 1
    oscore = 10.0 * unknown + 3.0 * (~pseen) + 1.0 * (pm == 0) + W_ADJKNOWN * adj_known * have_gain \
        + W_PMOVED * ((pm > 0) & pseen) + W_OBSEND * obs_end_bonus - 0.01 * ARANGE16
    obs_legal = legal[1:17]
    oscore = jnp.where(obs_legal, oscore, -1e9)
    obs_action = 1 + jnp.argmax(oscore).astype(jnp.int32)

    # ---- hint choice -------------------------------------------------------
    rev_unplaced = revealed & ~hint_placed
    set_size = hint_sets.sum(-1)  # [7]
    # placement candidates (h, card)
    c_match = hint_sets[:, col]  # [7,16] hint h contains col(card)
    p_match = (hint_sets[:, None, :] & cand[None, :, :]).sum(-1) / jnp.maximum(cand.sum(-1)[None, :], 1)  # prob
    anchor_has_lock = anchor_locked[col]  # [16]
    safe_card = known & ~locked & jnp.where(anchor_has_lock, in_anchor, in_anchor & (comp_size >= 2))
    # single-color hints may also go on a lone known card if that color has no lock yet (publishes exact color)
    single_ok = known & ~locked & ~anchor_has_lock & in_anchor & SINGLE_EARLY
    place_ok = rev_unplaced[:, None] & ~locked[None, :]
    pscore = jnp.where(known[None, :] & c_match, 20.0, 0.0) + 5.0 * p_match \
        + jnp.where((safe_card[None, :] & ((set_size == 1)[:, None] | anchor_has_lock[None, :] | (not ANCHOR_SAFE)))
                    | ((set_size == 1)[:, None] & single_ok[None, :]),
                    10.0 + comp_size[None, :] + 2.0 * anchor_has_lock[None, :], 0.0) \
        + (3.0 - set_size)[:, None] - 0.01 * ARANGE16[None, :]
    place_act = 1624 + jnp.arange(NH)[:, None] * 100 + pos[None, :, 0] * 10 + pos[None, :, 1]  # [7,16]
    place_legal = legal[jnp.clip(place_act, 0, NA - 1)] & place_ok
    pscore = jnp.where(place_legal, pscore, -1e9)
    best_p = jnp.argmax(pscore.reshape(-1))
    best_pscore = pscore.reshape(-1)[best_p]
    place_action = place_act.reshape(-1)[best_p].astype(jnp.int32)
    reveal_legal = legal[1617:1624]
    any_reveal = reveal_legal.any()
    reveal_action = 1617 + jnp.argmax(reveal_legal).astype(jnp.int32)
    do_place = (best_pscore >= 30.0) | (~any_reveal & (best_pscore > -1e8))
    hint_action = jnp.where(do_place, place_action, reveal_action)

    # ---- select by substep ---------------------------------------------------
    s0_action = jnp.where(end_ok, jnp.asarray(2324, jnp.int32), obs_action)
    action = jnp.where(substep == 0, s0_action,
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, 0).astype(jnp.int32)
    fallback = jnp.argmax(legal).astype(jnp.int32)
    action = jnp.where(legal[action], action, fallback)
    is_place = active & (substep == 3) & (action >= 1624) & (action < 2324)
    placed_h = jnp.clip((action - 1624) // 100, 0, NH - 1)
    my_hint = jnp.where(is_place & (jnp.arange(NH) == placed_h), 1, my_hint)
    new_memory['my_hint'] = my_hint

    trace = {
        'u': u.astype(jnp.int32),
        'end_ok': end_ok,
        'frac_conn': frac_conn.astype(jnp.float32),
        'n_known': known.sum().astype(jnp.int32),
        'comps_total': comps_all.sum().astype(jnp.int32),
        'best_gain': jnp.where(move_legal, gain, -99).max().astype(jnp.int32),
        'violation': violation,
        'end_strict': end_strict,
        's_now': s_now,
        'pos': pos.reshape(-1),
        'colk': jnp.where(known, col, -1).astype(jnp.int32),
        'cols': jnp.where(known_s, col_s, -1).astype(jnp.int32),
        'locked': locked,
    }
    return action, new_memory, trace
