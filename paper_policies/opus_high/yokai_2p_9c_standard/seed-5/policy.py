"""Belief-based cooperative policy for two-player 9-card Yokai.

Design summary
--------------
Each seat keeps an explicit posterior over the 1680 possible colourings of the
nine cards.  Hard evidence comes from the seat's own observations (retained in
private memory across turns).  Soft evidence comes from the partner's moves:
when the partner relocates a card it has publicly observed next to another card
it has publicly observed, it is likely joining a colour group, so colourings in
which those cards share a colour are up-weighted.  Both channels are public or
private-local; no code word, seat-dependent role, or action-index convention is
used, so the behaviour degrades gracefully with an arbitrary partner.

Per substep:
  0  end the game when P(all colours connected) clears a turn-dependent bar,
     otherwise observe the most useful unknown card;
  1  observe a second unknown card;
  2  move the card/destination pair maximising the expected number of connected
     colours (with a bonus for "all three connected");
  3  place a revealed hint on a card that is confidently correct and already
     settled (a correct placement costs one point, a reveal costs three),
     otherwise reveal a two-colour hint.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'belief_yokai', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

# ---------------------------------------------------------------- constants
NCARD = 9
NHINT = 4
GRID = 9
NCELL = 81
NCOL = 1680
NCAND = 24            # empty cells orthogonally adjacent to the board (<= 20)
LAM = 2.0             # bonus weight for "all three colours connected"
INFER = 0.9           # strength of the partner-move colour inference
INFER_PLACE = 0.9     # strength of the hint-placement colour inference
PLACE_THRESH = 0.7    # confidence needed before spending a turn placing a hint
SETTLE_THRESH = 1.0   # expected same-colour neighbours before locking a card
TAU_LIST = [0.95, 0.9, 0.85, 0.8, 0.6, 0.4, 0.2, 0.05]


def _colorings_list():
    """All 1680 assignments of three colours x three cards onto nine cards."""
    rows = [()]
    for _ in range(NCARD):
        nxt = []
        for row in rows:
            for c in range(3):
                if row.count(c) < 3:
                    nxt.append(row + (c,))
        rows = nxt
    return [list(r) for r in rows]


def _consts():
    """Trace-time constant tables (folded into the compiled graph)."""
    colorings = jnp.asarray(_colorings_list(), dtype=jnp.int32)          # [N,9]
    onehot = (colorings[:, :, None] == jnp.arange(3)[None, None, :]).astype(jnp.float32)
    same = (colorings[:, :, None] == colorings[:, None, :]).astype(jnp.float32)
    cells = jnp.stack([jnp.arange(NCELL) // GRID, jnp.arange(NCELL) % GRID], axis=-1)
    tau = jnp.asarray(TAU_LIST, dtype=jnp.float32)
    return colorings, onehot, same, cells, tau


def init_agent():
    return {
        'w': jnp.ones((NCOL,), dtype=jnp.float32),
        'known': -jnp.ones((NCARD,), dtype=jnp.int32),
        'pseen': jnp.zeros((NCARD,), dtype=jnp.int32),
        'prev_pos': -jnp.ones((NCARD, 2), dtype=jnp.int32),
        'prev_placed': jnp.zeros((NHINT,), dtype=jnp.int32),
    }


# ---------------------------------------------------------------- helpers
def _adjacency(pos):
    d = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    return (d == 1).astype(jnp.float32)


def _color_edges(pos, onehot):
    """[N,3] count of same-colour adjacent pairs, per colouring and colour."""
    a = _adjacency(pos)
    return 0.5 * jnp.einsum('nik,ij,njk->nk', onehot, a, onehot)


def _p_connected(pos, w, onehot):
    t = _color_edges(pos, onehot)
    conn = (t >= 2).all(-1).astype(jnp.float32)
    return (w * conn).sum() / jnp.maximum(w.sum(), 1e-9)


def _candidate_cells(pos, cells):
    """Indices of empty cells adjacent to the board (a superset of destinations)."""
    occ = jnp.zeros((NCELL,), dtype=jnp.float32).at[pos[:, 0] * GRID + pos[:, 1]].set(1.0)
    adjn = (jnp.abs(cells[:, None, :] - pos[None, :, :]).sum(-1) == 1).sum(-1)
    key = (adjn > 0).astype(jnp.float32) * (1.0 - occ)
    return jnp.argsort(-key)[:NCAND]


def _move_scores(pos, w, cand, colorings, onehot, same, cells):
    """[NCAND, 9] expected objective of moving card m onto each candidate cell."""
    a = _adjacency(pos)
    t = _color_edges(pos, onehot)
    tc = jnp.take_along_axis(t, colorings, axis=1)                    # [N,9] edges in m's colour
    im = jnp.einsum('nmj,mj->nm', same, a)                            # [N,9] edges at m
    need = 2.0 - (tc - im)
    other_ok = (t >= 2).sum(-1)[:, None].astype(jnp.float32) - (tc >= 2).astype(jnp.float32)
    base_pairs = t.sum(-1)[:, None] - im        # same-colour pairs not involving card m
    wn = w / jnp.maximum(w.sum(), 1e-9)

    def body(carry, cell):
        cxy = cells[cell]
        adjc = (jnp.abs(pos - cxy[None, :]).sum(-1) == 1).astype(jnp.float32)   # [9]
        s = jnp.einsum('j,njk->nk', adjc, onehot)                               # [N,3]
        sg = jnp.take_along_axis(s, colorings, axis=1)                          # [N,9]
        cnt = sg - adjc[None, :]
        ok = (cnt >= need).astype(jnp.float32)
        ncon = other_ok + ok
        # graded credit for every same-colour adjacency, plus a solved-board bonus
        val = (base_pairs + cnt) + LAM * (ncon >= 2.5).astype(jnp.float32)
        return carry, jnp.einsum('n,nm->m', wn, val)

    _, scores = jax.lax.scan(body, 0.0, cand)
    return scores


# ---------------------------------------------------------------- policy
def agent_act(obs, reset, memory, rng):
    colorings, onehot, same, cells, tau = _consts()
    feats = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    pos = obs.positions.astype(jnp.int32)
    card = feats[:NCARD]
    hint = feats[NCARD:]

    active = card[0, 10] > 0
    substep = card[0, 11]
    pseen_now = (card[:, 9] > 0).astype(jnp.int32)
    own_onehot = card[:, 0:3]
    own_known = own_onehot.sum(-1) > 0
    own_color = jnp.argmax(own_onehot, axis=-1)

    hint_sets = hint[:, 3:6] > 0
    hrevealed = hint_sets.any(-1)
    hplaced = hint[:, 8] > 0
    turn = jnp.clip(hrevealed.sum() + hplaced.sum(), 0, 7)

    init = init_agent()
    w = jnp.where(reset, init['w'], memory['w'])
    known = jnp.where(reset, init['known'], memory['known'])
    pseen = jnp.where(reset, init['pseen'], memory['pseen'])
    prev_pos = jnp.where(reset, init['prev_pos'], memory['prev_pos'])
    prev_placed = jnp.where(reset, init['prev_placed'], memory['prev_placed'])

    # -- hard evidence: colours this seat observes on its own turn
    known = jnp.where(own_known & (known < 0), own_color, known)
    pseen = jnp.maximum(pseen, pseen_now)

    # -- soft evidence: a partner move that joins two cards the partner has seen
    changed = (pos != prev_pos).any(-1)
    nmoved = changed.sum()
    valid_prev = (prev_pos >= 0).all()
    mvec = changed.astype(jnp.int32)
    mpos = (pos * mvec[:, None]).sum(0)
    mprev = (prev_pos * mvec[:, None]).sum(0)
    d_now = jnp.abs(pos - mpos[None, :]).sum(-1)
    d_prev = jnp.abs(prev_pos - mprev[None, :]).sum(-1)
    newnb = (d_now == 1) & (d_prev != 1) & (pseen > 0) & (~changed)
    cm = (colorings * mvec[None, :]).sum(-1)                             # [N] moved card's colour
    match = ((colorings == cm[:, None]) & newnb[None, :]).any(-1)
    moved_seen = (pseen * mvec).sum() > 0
    apply_inf = (nmoved == 1) & (~active) & valid_prev & moved_seen & newnb.any()
    like = jnp.where(match, 1.0, 1.0 - INFER)
    w = jnp.where(apply_inf, w * like, w)
    w = w / jnp.maximum(w.max(), 1e-12)

    # -- soft evidence: a hint tile asserts that the card under it has one of the
    #    hint's colours; trust it when the placer had publicly observed that card
    hint_pos = hint[:, 6:8]
    cardsel = (hint_pos[:, None, :] == pos[None, :, :]).all(-1).astype(jnp.float32)   # [4,9]
    ccoh = jnp.einsum('hi,nik->nhk', cardsel, onehot)                     # [N,4,3]
    hmatch = (ccoh * hint_sets.astype(jnp.float32)[None, :, :]).sum(-1)   # [N,4]
    newplace = (hplaced.astype(jnp.int32) - prev_placed) > 0
    hgate = newplace & ((cardsel * pseen[None, :].astype(jnp.float32)).sum(-1) > 0)
    hfactor = jnp.where(hgate[None, :], hmatch + (1.0 - hmatch) * (1.0 - INFER_PLACE), 1.0)
    w = w * hfactor.prod(-1)
    w = w / jnp.maximum(w.max(), 1e-12)

    consistent = ((colorings == known[None, :]) | (known[None, :] < 0)).all(-1).astype(jnp.float32)
    belief = w * consistent
    belief = belief / jnp.maximum(belief.sum(), 1e-12)

    # -- substeps 0/1: pick a card to observe (unknown first, partner-unseen next)
    rng_obs, _ = jax.random.split(rng)
    obs_legal = legal[1:1 + NCARD]
    pref = (4.0 * (known < 0) + 1.0 * (pseen == 0)
            + 0.5 * jax.random.uniform(rng_obs, (NCARD,)))
    obs_action = 1 + jnp.argmax(jnp.where(obs_legal, pref, -1e9))

    # -- substep 0: end the game when connectivity is likely enough
    p_conn = _p_connected(pos, belief, onehot)
    end_action = jnp.where(p_conn >= tau[turn], 1067, obs_action)

    # -- substep 2: best legal relocation
    cand = _candidate_cells(pos, cells)
    ms = _move_scores(pos, belief, cand, colorings, onehot, same, cells)
    move_ids = 10 + jnp.arange(NCARD)[None, :] * NCELL + cand[:, None]
    flat = jnp.full((1068,), -1e9, dtype=jnp.float32)
    flat = flat.at[move_ids.reshape(-1)].set(ms.reshape(-1))
    flat = jnp.where(legal, flat, -1e9)
    move_action = jnp.argmax(flat)

    # -- substep 3: spending a turn on a correct placement costs one point,
    #    revealing costs three, so place whenever a confident, settled card is
    #    available; otherwise reveal a two-colour hint (keeping hint 0 hidden).
    reveal_legal = legal[739:739 + NHINT]
    reveal_pref = jnp.asarray([1.0, 2.0, 2.0, 2.0], dtype=jnp.float32)
    reveal_action = 739 + jnp.argmax(jnp.where(reveal_legal, reveal_pref, -1e9))
    pcolor = jnp.einsum('n,nik->ik', belief, onehot)                     # [9,3]
    pcorrect = pcolor @ hint_sets.astype(jnp.float32).T                  # [9 cards, 4 hints]
    adj_now = _adjacency(pos)
    settled = jnp.einsum('n,nij,ij->i', belief, same, adj_now)           # E[same-colour neighbours]
    place_ids = 743 + jnp.arange(NHINT)[None, :] * NCELL + (pos[:, 0] * GRID + pos[:, 1])[:, None]
    place_legal = legal[place_ids]
    place_key = jnp.where(place_legal, pcorrect + 0.05 * settled[:, None], -1e9)
    flat_best = jnp.argmax(place_key.reshape(-1))
    place_action = place_ids.reshape(-1)[flat_best]
    best_pc = pcorrect.reshape(-1)[flat_best]
    best_settled = jnp.broadcast_to(settled[:, None], pcorrect.shape).reshape(-1)[flat_best]
    want_place = (best_pc >= PLACE_THRESH) & (best_settled >= SETTLE_THRESH) & place_legal.any()
    hint_action = jnp.where(reveal_legal.any() & (~want_place), reveal_action, place_action)

    action = jnp.where(substep == 0, end_action,
                       jnp.where(substep == 1, obs_action,
                                 jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(legal[action], action, jnp.argmax(legal))
    action = jnp.where(active, action, 0).astype(jnp.int32)

    new_memory = {
        'w': w,
        'known': known.astype(jnp.int32),
        'pseen': pseen.astype(jnp.int32),
        'prev_pos': pos,
        'prev_placed': hplaced.astype(jnp.int32),
    }
    trace = {
        'p_connected': p_conn.astype(jnp.float32),
        'turn': turn.astype(jnp.int32),
        'n_known': (known >= 0).sum().astype(jnp.int32),
        'inferred': apply_inf.astype(jnp.int32),
    }
    return action, new_memory, trace
