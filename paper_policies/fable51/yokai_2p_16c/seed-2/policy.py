"""Yokai 2p policy: private knowledge tracking, greedy colour merging,
guarded hint placement, and enumeration-based end-of-game decision.

Convention-free design: every inference uses only public rules (hint sets,
locks, colour counts) plus this seat's own observations.  No private signalling
protocol is assumed from the partner.
"""

import jax
import jax.numpy as jnp
from jax import lax

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'knowledge_sorter', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NC = 16
NH = 7
G = 10
NA = 2325
OBS0 = 1
MOVE0 = 17
REVEAL0 = 1617
PLACE0 = 1624
END = 2324

NSAMP = 1024


def _idx():
    return jnp.arange(NC, dtype=jnp.int32)


def _init_positions():
    idx = _idx()
    return jnp.stack([idx // 4 + 3, idx % 4 + 3], axis=1).astype(jnp.int32)


def init_agent():
    return {
        'known': jnp.full((NC,), -1, dtype=jnp.int32),
        'step': jnp.zeros((), dtype=jnp.int32),
        'prev_pos': _init_positions(),
        'partner_moved': jnp.zeros((NC,), dtype=jnp.bool_),
        'my_moved': jnp.zeros((NC,), dtype=jnp.bool_),
        'link': jnp.full((NC,), -1, dtype=jnp.int32),
    }


def _manhattan(pos):
    return jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)


def _labels(dist, colors):
    """Connected-component labels among cards with known colour (same-colour edges)."""
    adj = dist == 1
    same = (colors[:, None] == colors[None, :]) & (colors[:, None] >= 0)
    A = adj & same
    idx = _idx()
    label = idx

    def body(_, lab):
        nb = jnp.where(A, lab[None, :], NC).min(1)
        return jnp.minimum(lab, nb)

    label = lax.fori_loop(0, 5, body, label)
    return label, same


def _comps(dist, colors):
    label, same = _labels(dist, colors)
    roots = (label == _idx()) & (colors >= 0)
    oh = jax.nn.one_hot(colors, 4, dtype=jnp.int32)
    ncomp = (oh * roots[:, None]).sum(0)
    present = (oh.sum(0) > 0).astype(jnp.int32)
    extra = (ncomp - present).sum()
    pd = (dist * same).sum() // 2
    return extra, pd, label


def agent_act(obs, reset, memory, rng):
    init = init_agent()
    memory = jax.tree_util.tree_map(lambda a, b: jnp.where(reset, a, b), init, memory)
    IDX = _idx()
    CELLS = jnp.arange(G * G, dtype=jnp.int32)
    CX = CELLS // G
    CY = CELLS % G

    nf = obs.node_features.astype(jnp.int32)
    card = nf[:NC]
    hint = nf[NC:NC + NH]
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions.astype(jnp.bool_)

    onehot = card[:, 0:4] > 0
    seen_color = jnp.where(onehot.any(1), jnp.argmax(onehot, 1), -1).astype(jnp.int32)
    known = jnp.where(seen_color >= 0, seen_color, memory['known'])
    locked = card[:, 10] > 0
    pseen = card[:, 11] > 0
    active = card[0, 12] > 0
    substep = card[0, 13]
    step = memory['step']
    turn = step // 4

    dist = _manhattan(pos)
    moved = (pos != memory['prev_pos']).any(1)
    partner_moved = memory['partner_moved'] | (moved & ~active & pseen)
    my_moved = memory['my_moved'] | (moved & active)
    adj_any = (dist == 1)
    n_nb = adj_any.sum(1)
    single_nb = jnp.argmax(adj_any, 1).astype(jnp.int32)
    link = jnp.where(moved, jnp.where((~active) & pseen & (n_nb == 1), single_nb, -1), memory['link'])
    link_alive = (link >= 0) & adj_any[IDX, jnp.maximum(link, 0)]
    link = jnp.where(link_alive, link, -1)

    hsets = hint[:, 4:8] > 0                      # [7,4]
    hrev = hsets.any(1)
    hplaced = hint[:, 10] > 0
    hcard = obs.adjacency_matrix[NC:NC + NH, :NC] > 0   # [7,16]
    hsize = hsets.sum(1)

    # ---------------- candidate colour sets ----------------
    restrict = hcard[:, :, None] & hplaced[:, None, None] & ~hsets[:, None, :]   # [7,16,4]
    # only trust a placed hint's colour constraint if the partner (or I) had observed the card
    trusted = pseen | (known >= 0)
    cand = ~(restrict.any(0) & trusted[:, None])
    # own observations take precedence over (possibly wrong) partner hint placements
    cand = jnp.where(known[:, None] >= 0, jax.nn.one_hot(known, 4) > 0, cand)

    def deduce(_, c):
        single = c.sum(1) == 1
        known_ct = (c & single[:, None]).sum(0)
        full = known_ct >= 4
        c = c & ~(full[None, :] & ~single[:, None])
        poss = c.sum(0)
        exact = poss == 4
        force = c & exact[None, :]
        c = jnp.where(force.any(1)[:, None], force, c)
        return c

    cand = lax.fori_loop(0, 4, deduce, cand)
    csize = cand.sum(1)
    kc = jnp.where(csize == 1, jnp.argmax(cand, 1), -1).astype(jnp.int32)
    unknown = kc < 0

    extra_now, pd_now, label_now = _comps(dist, kc)
    comp_size = (label_now[:, None] == label_now[None, :]).sum(1) * (kc >= 0)
    color_known_ct = jax.nn.one_hot(kc, 4, dtype=jnp.int32).sum(0)
    all_in_one = comp_size == jnp.where(kc >= 0, color_known_ct[jnp.maximum(kc, 0)], -1)
    adj_known = ((dist == 1)[:, :, None] & (jax.nn.one_hot(kc, 4) > 0)[None, :, :]).any(1)   # [16,4]

    # soft colours: partner-moved unknown cards adjacent to exactly one known colour
    adj_ct = adj_known.sum(1)
    soft_col = jnp.argmax(adj_known, 1).astype(jnp.int32)
    n_unknown_nb = (adj_any & unknown[None, :]).sum(1)
    strict = (adj_ct == 1) & (n_unknown_nb == 0) & (n_nb >= 1)
    soft_ok = unknown & partner_moved & strict & cand[IDX, soft_col]
    # single-neighbour placements by the partner: colour flows across the link both ways
    lk = jnp.maximum(link, 0)
    has_link = link >= 0
    fwd_col = jnp.where(has_link & (kc >= 0), kc, -1)                 # moved card known -> neighbour
    to_nb = jnp.full((NC,), -1, jnp.int32).at[lk].max(jnp.where(has_link, fwd_col, -1))
    from_nb = jnp.where(has_link, kc[lk], -1)                         # neighbour known -> moved card
    link_col = jnp.where(from_nb >= 0, from_nb, to_nb)
    link_ok = unknown & (link_col >= 0) & cand[IDX, jnp.maximum(link_col, 0)]
    soft_col = jnp.where(soft_ok, soft_col, jnp.where(link_ok, link_col, 0)).astype(jnp.int32)
    soft_ok = soft_ok | link_ok
    kc_soft = jnp.where(soft_ok, soft_col, kc)
    strict_adj = soft_ok[:, None] & (jax.nn.one_hot(soft_col, 4) > 0)

    # ---------------- observation choice ----------------
    oscore = (unknown * 100 + csize * 10 + (~pseen) * 6 + (~partner_moved) * 3).astype(jnp.float32)
    oscore = oscore + jax.random.uniform(jax.random.fold_in(rng, 1), (NC,)) * 0.5
    olegal = legal[OBS0:MOVE0]
    obs_action = OBS0 + jnp.argmax(jnp.where(olegal, oscore, -1e9))

    # ---------------- end decision ----------------
    u = unknown.sum()
    order = jnp.argsort(~unknown)                       # unknown card ids first
    known_ct_all = jax.nn.one_hot(kc, 4, dtype=jnp.int32).sum(0)
    missing = 4 - known_ct_all                          # multiset of colours still unassigned
    consistent_counts = (missing >= 0).all() & (missing.sum() == u)
    cum = jnp.cumsum(jnp.maximum(missing, 0))
    pool = (jnp.arange(NC)[:, None] >= cum[None, :]).sum(1).astype(jnp.int32)   # colour per slot, 4 = pad
    prior = 1.0 + 25.0 * strict_adj + 2.0 * (partner_moved[:, None] & adj_known & ~strict_adj) + 1.0 * (pseen[:, None] & adj_known)
    prior = jnp.concatenate([prior.astype(jnp.float32), jnp.zeros((NC, 1), jnp.float32)], axis=1)   # [16,5]
    cand5 = jnp.concatenate([cand, jnp.zeros((NC, 1), jnp.bool_)], axis=1)
    slot_valid = jnp.arange(NC) < u

    def sample_eval(key):
        shuffled = pool[jax.random.permutation(key, NC)]
        compact = shuffled[jnp.argsort(shuffled >= 4)]          # non-pad colours first, uniformly permuted
        colors = kc.at[order].set(jnp.where(slot_valid, compact, kc[order]))
        cons = jnp.where(slot_valid, cand5[order, compact], True).all()
        w = jnp.where(slot_valid, prior[order, compact], 1.0).prod()
        ex, _, _ = _comps(dist, colors)
        conn = (ex == 0) & (colors >= 0).all()
        wc = w * cons
        return wc, wc * conn, colors

    rng_obs, rng_samp = jax.random.split(rng)
    ws, wcs, scols = jax.vmap(sample_eval)(jax.random.split(rng_samp, NSAMP))
    den = ws.sum()
    p_conn = jnp.where(consistent_counts & (den > 0), wcs.sum() / jnp.maximum(den, 1e-9), 0.0)
    post = (jax.nn.one_hot(scols, 4, dtype=jnp.float32) * ws[:, None, None]).sum(0) / jnp.maximum(den, 1e-9)   # [16,4]
    unc = 1.0 - post.max(1)
    oscore2 = oscore + jnp.where(turn >= 7, 40.0 * unc * unknown, 0.0)
    obs_action = OBS0 + jnp.argmax(jnp.where(olegal, oscore2, -1e9))
    thr = jnp.where(turn >= 10, 0.45, jnp.where(turn >= 8, 0.5, jnp.where(turn >= 6, 0.65, 0.85)))
    do_end = legal[END] & (p_conn >= thr)
    a0 = jnp.where(do_end, END, obs_action)

    # ---------------- move choice ----------------
    nb_partner_unknown = (adj_any & (unknown & partner_moved)[None, :]).any(1)
    extra_soft_now, pd_soft_now, _ = _comps(dist, kc_soft)
    def move_score(i, j):
        newpos = pos.at[i].set(jnp.stack([CX[j], CY[j]]))
        d = _manhattan(newpos)
        ex, pd, _ = _comps(d, kc_soft)
        s = -10.0 * ex - 1.0 * pd - 0.05 * d.sum() / 2.0
        unk = kc[i] < 0
        fresh = unk & ~pseen[i] & ~partner_moved[i]
        s = s - jnp.where(fresh, 0.3, 0.0) - jnp.where(unk & ~fresh, 16.0, 0.0)
        s = s - 2.0 * nb_partner_unknown[i]
        no_gain = (ex >= extra_soft_now) & (pd >= pd_soft_now)
        s = s - jnp.where((~unk) & no_gain, 1.0, 0.0)
        s = s - jnp.where((~unk) & partner_moved[i], 12.0, 0.0)
        return s

    mscores = jax.vmap(jax.vmap(move_score, (None, 0)), (0, None))(IDX, CELLS)   # [16,100]
    mlegal = legal[MOVE0:REVEAL0].reshape(NC, G * G)
    mscores = jnp.where(mlegal, mscores, -1e9)
    move_action = MOVE0 + jnp.argmax(mscores)

    # ---------------- hint choice ----------------
    # a locked card outside my component that could still be my colour blocks locking
    same_comp = label_now[:, None] == label_now[None, :]
    could_be = cand & locked[:, None]                                   # [16,4]
    conflict = (could_be[None, :, :] & ~same_comp[:, :, None]).any(1)   # [16,4] for card c: any locked card outside comp that could be k
    my_conflict = conflict[IDX, jnp.maximum(kc, 0)] & (kc >= 0)
    safe = (kc >= 0) & all_in_one & (comp_size >= 2) & ~locked & ~my_conflict
    valid = hrev[:, None] & ~hplaced[:, None] & ~(cand[None, :, :] & ~hsets[:, None, :]).any(2)   # [7,16]
    info = jnp.where(hsize[:, None] == 1,
                     jnp.where(pseen[None, :], 5.0, 10.0),
                     jnp.where(pseen[None, :], 20.0, -100.0))
    pscore = valid * 50.0 + safe[None, :] * 100.0 + info + comp_size[None, :].astype(jnp.float32)
    frac = (cand & hsets[:, None, :]).sum(2) / jnp.maximum(csize, 1)[None, :]
    pscore = pscore + 5.0 * frac
    cell = pos[:, 0] * G + pos[:, 1]
    pidx = PLACE0 + jnp.arange(NH)[:, None] * 100 + cell[None, :]
    plegal = legal[pidx]
    pscore = jnp.where(plegal, pscore, -1e9)
    best_flat = jnp.argmax(pscore)
    best_val = pscore.reshape(-1)[best_flat]
    place_action = pidx.reshape(-1)[best_flat]
    rlegal = legal[REVEAL0:PLACE0]
    can_reveal = rlegal.any()
    reveal_action = REVEAL0 + jnp.argmax(rlegal)
    do_place = (best_val >= 150.0) | (~can_reveal)
    hint_action = jnp.where(do_place, place_action, reveal_action)

    # ---------------- assemble ----------------
    action = jnp.where(substep == 0, a0,
             jnp.where(substep == 1, obs_action,
             jnp.where(substep == 2, move_action, hint_action))).astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    action = jnp.where(active, action, 0).astype(jnp.int32)

    new_memory = {
        'known': known.astype(jnp.int32),
        'step': (step + 1).astype(jnp.int32),
        'prev_pos': pos,
        'partner_moved': partner_moved,
        'my_moved': my_moved,
        'link': link.astype(jnp.int32),
    }
    trace = {
        'p_conn': p_conn.astype(jnp.float32),
        'unknown': u.astype(jnp.int32),
        'extra_comps': extra_now.astype(jnp.int32),
        'do_end': do_end,
        'n_locked': locked.sum().astype(jnp.int32),
        'n_soft': soft_ok.sum().astype(jnp.int32),
    }
    return action, new_memory, trace
