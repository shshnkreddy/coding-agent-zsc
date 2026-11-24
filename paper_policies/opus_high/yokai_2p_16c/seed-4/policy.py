"""Belief-greedy Yokai cooperator.

Every decision is a local repair of the *public* board, driven by this seat's
own colour memory plus three public signals that any partner necessarily
emits: card positions, the "partner has observed this card" column, and the
colour sets of the hints it places.

Beliefs.  A card's colour set starts as what we saw, is intersected with the
set of any hint placed on it (a partner is paid to place hints that match, and
the constraint is dropped for any card where it contradicts our own eyes, or
globally once two such contradictions appear), and is then narrowed by the
four-per-colour count.  Cards left over get a capacity-weighted marginal.

* substep 0/1: look at a card whose colour is still unknown, preferring one
  nobody has looked at, so team colour coverage grows twice as fast as it
  would if both seats looked at the same cards.
* substep 2: play the legal move minimising a potential made of
  (a) W_COMP * surplus connected components among cards of resolved colour,
  (b) -W_RISK * expected attachment mass of unresolved cards to groups they
      could still belong to, discounted for cards the partner has seen,
  plus a do-not-disturb term for the partner's cards and small compactness,
  displacement and tie-breaking terms.
* substep 3: place a revealed hint on an attached card whose colour is in the
  hint set.  That costs one point instead of three for a reveal and, when the
  hint is narrow and sits on a card the partner has not seen, it hands that
  seat a colour it would otherwise have to spend a scarce look on.  Otherwise
  reveal the narrowest hidden hint, for the same reason.
* substep 0: end the episode when p(every colour connected) * (score if
  connected) beats a realistic estimate of playing on to the horizon implied
  by the public hint counts.

Nothing here is a private convention: an independently written partner is
treated as a co-worker whose looks, moves and hint placements are public
evidence, and the one channel that assumes partner competence is checked
against our own observations before it is believed.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}

NCARD = 16
NCOLOR = 4
NHINT = 7
GRID = 10
NCELL = GRID * GRID
BIG = 1.0e6

# Public size class of each hint id (1,1,2,2,2,3,3).  Narrow hints say more
# about the card they end up on, so they are revealed and placed first.
_HINT_SIZE_LIST = (1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0)

W_COMP = 100.0      # surplus colour component: the dominant term
W_RISK = 12.0       # expected unattached mass of an unresolved card
W_KEEP = 20.0       # do not disturb a card only the partner has seen
W_PULL = 3.0        # soft attraction between probably-same-colour cards
W_DISP = 0.05       # prefer small displacements when nothing else differs
W_JITTER = 0.01     # symmetry breaking
GAMMA_PARTNER = 0.3  # weight of our own worry about cards the partner knows
TRUST = 0.6         # belief a partner-observed card sits next to its own group
TURN_VALUE = 4.0    # hint value given up per two further turns
REPAIR = 0.15       # realistic gain in P(connected) per two further turns
P_FLOOR = 0.25      # never gamble on a board we think is probably broken


def _cards():
    return jnp.arange(NCARD)


def _colors():
    return jnp.arange(NCOLOR)


def _slots():
    return jnp.arange(NCOLOR)


def _cellpos():
    cells = jnp.arange(NCELL)
    return jnp.stack([cells // GRID, cells % GRID], axis=-1)


def _hint_size():
    return jnp.asarray(_HINT_SIZE_LIST, dtype=jnp.float32)


def init_agent():
    return {'known': jnp.zeros((NCARD, NCOLOR), dtype=jnp.int8)}


def _beliefs(known, allowed):
    """Resolve colours from own observations, placed hints and card counting.

    ``allowed`` is the public constraint carried by placed hints: a hint that
    sits on a card publicly announces a colour set, and every partner is paid
    to place hints that really do contain the card's colour.  The constraint is
    dropped for any card where it contradicts what this seat has seen.
    """
    seen = known > 0
    observed = seen.any(axis=-1)
    base = jnp.where(observed[:, None], seen, jnp.ones_like(seen))
    # Sanity-check the channel before leaning on it: a hint that excludes a
    # colour we saw with our own eyes was placed by a partner who does not
    # place hints truthfully.  Our own placements only ever go on colours we
    # know, so a single clash is already the partner's, and the whole channel
    # stops being evidence from then on.
    clash = observed & ~(seen & allowed).any(axis=-1)
    reliable = ~clash.any()
    allowed = allowed | ~reliable
    poss = base & allowed
    poss = jnp.where(poss.any(axis=-1)[:, None], poss, base)
    for _ in range(3):
        single = poss.sum(axis=-1) == 1
        used = (poss & single[:, None]).sum(axis=0)
        free = (NCOLOR - used) > 0
        nxt = poss & (free[None, :] | single[:, None])
        poss = jnp.where(nxt.any(axis=-1)[:, None], nxt, poss)
    single = poss.sum(axis=-1) == 1
    colour = jnp.argmax(poss, axis=-1)
    used = (poss & single[:, None]).sum(axis=0)
    room = jnp.maximum(NCOLOR - used, 0).astype(jnp.float32)
    # Per-card colour marginal for cards we have not resolved.
    q = poss.astype(jnp.float32) * room[None, :]
    q = q / jnp.maximum(q.sum(axis=-1, keepdims=True), 1e-6)
    q = jnp.where(single[:, None], poss.astype(jnp.float32), q)
    return poss, single, colour, observed, q, reliable


def _hint_mask(nf, adjacency):
    """Colours a card may still have given the hints already placed on it."""
    rset = nf[NCARD:, 4:8] > 0
    placed = nf[NCARD:, 10] > 0
    link = (adjacency[NCARD:, :NCARD] > 0) & placed[:, None]         # [hint, card]
    banned = (link[:, :, None] & (~rset)[:, None, :]).any(axis=0)
    return ~banned


def _board(pos, poss, single, colour):
    """Board quantities that do not depend on a candidate move."""
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(axis=-1)
    adj = dist == 1
    same = single[:, None] & single[None, :] & (colour[:, None] == colour[None, :])
    adj_same = adj & same
    lab = _cards()
    for _ in range(5):
        lab = jnp.minimum(jnp.min(jnp.where(adj_same, lab[None, :], NCARD), axis=1), lab)
    root = (lab == _cards()) & single
    onehot = (colour[:, None] == _colors()[None, :]) & single[:, None]
    comps = (root[:, None] & onehot).sum(axis=0)
    frag = jnp.maximum(comps - 1, 0).sum()
    touch = (adj.astype(jnp.float32) @ onehot.astype(jnp.float32)) > 0
    return {'dist': dist, 'adj': adj, 'adj_same': adj_same, 'onehot': onehot,
            'comps': comps, 'frag': frag, 'touch': touch}


def _connect_prob(poss, single, q, info, partner_seen, reliable):
    """P(every colour connected) for the board as it stands."""
    touch_ok = info['touch'] & poss
    mass = (touch_ok.astype(jnp.float32) * q).sum(axis=-1)
    trust = TRUST * reliable.astype(jnp.float32)
    trusted = jnp.where(partner_seen & touch_ok.any(axis=-1),
                        mass + trust * (1.0 - mass), mass)
    p_card = jnp.where(single, 1.0, jnp.clip(trusted, 0.0, 1.0))
    return jnp.where(info['frag'] == 0, jnp.prod(p_card), 0.0)


def _move_scores(pos, poss, single, colour, observed, partner_seen, q, info, rng):
    """Potential of every (card, destination cell) pair, lower is better."""
    adj_f = info['adj'].astype(jnp.float32)
    onehot_f = info['onehot'].astype(jnp.float32)

    # --- components of the moved card's colour group after the move ---------
    key = jnp.where(info['onehot'].T, 0, 1)                        # [colour, card]
    order = jnp.argsort(key, axis=1)[:, :NCOLOR]
    ok = jnp.take_along_axis(key, order, axis=1) == 0
    idx = order[colour]                                            # [card, slot]
    valid = ok[colour]
    slot_pos = pos[idx]
    is_self = idx == _cards()[:, None]
    cand = jnp.where(is_self[:, None, :, None],
                     _cellpos()[None, :, None, :],
                     slot_pos[:, None, :, :])                      # [card, cell, slot, 2]
    d4 = jnp.abs(cand[:, :, :, None, :] - cand[:, :, None, :, :]).sum(axis=-1)
    adj4 = (d4 == 1) & valid[:, None, :, None] & valid[:, None, None, :]
    lab = jnp.broadcast_to(_slots(), adj4.shape[:3])
    for _ in range(3):
        lab = jnp.minimum(jnp.min(jnp.where(adj4, lab[:, :, None, :], NCOLOR), axis=-1), lab)
    comps_new = ((lab == _slots()[None, None, :]) & valid[:, None, :]).sum(axis=-1)
    d_comp = jnp.where(single[:, None], comps_new - info['comps'][colour][:, None], 0)
    d_comp = d_comp.astype(jnp.float32)

    # --- expected attachment mass of unresolved cards after the move --------
    gamma = jnp.where(partner_seen & (~observed), GAMMA_PARTNER, 1.0)
    weight = ((~single).astype(jnp.float32) * gamma)[:, None] * q           # [u, colour]
    weight = weight * poss.astype(jnp.float32)
    cnt = adj_f @ onehot_f                                         # [u, colour]
    excl = cnt[:, None, :] - adj_f[:, :, None] * onehot_f[None, :, :]
    has = (excl > 0).astype(jnp.float32)                           # [u, m, colour]
    off = (~jnp.eye(NCARD, dtype=bool)).astype(jnp.float32)[:, :, None]
    kept = (weight[:, None, :] * has * off).sum(axis=(0, 2))       # [m]
    src = jnp.take_along_axis(weight[:, None, :] * (1.0 - has) * off,
                              colour[None, :, None], axis=2)[:, :, 0]
    src = src * single.astype(jnp.float32)[None, :]                # [u, m]
    near = (jnp.abs(_cellpos()[:, None, :] - pos[None, :, :]).sum(axis=-1) == 1)
    near_f = near.astype(jnp.float32)
    gained = near_f @ src                                          # [cell, m]
    cell_touch = (near_f @ onehot_f) > 0                           # [cell, colour]
    self_gain = cell_touch.astype(jnp.float32) @ weight.T          # [cell, m]
    self_gain = self_gain * (~single).astype(jnp.float32)[None, :]
    attached = kept[:, None] + gained.T + self_gain.T              # [m, cell]

    # --- compactness, displacement, respect for the partner's cards ---------
    pn = poss.astype(jnp.float32)
    pn = pn / jnp.maximum(pn.sum(axis=-1, keepdims=True), 1.0)
    w = (pn @ pn.T) * (1.0 - jnp.eye(NCARD))
    cell_dist = jnp.abs(_cellpos()[:, None, :] - pos[None, :, :]).sum(axis=-1).astype(jnp.float32)
    pull = (cell_dist @ w.T).T - (info['dist'].astype(jnp.float32) * w).sum(axis=-1)[:, None]
    disp = cell_dist.T
    keep = jnp.where(partner_seen & (~observed), W_KEEP, 0.0)[:, None]
    jitter = jax.random.uniform(rng, (NCARD, NCELL))

    return (W_COMP * d_comp - W_RISK * attached + keep
            + W_PULL * pull + W_DISP * disp + W_JITTER * jitter)


def _hints(nf, adjacency, poss, single, colour, q):
    """Public hint state plus the value of the board if it is connected."""
    rset = nf[NCARD:, 4:8] > 0
    placed = nf[NCARD:, 10] > 0
    revealed = rset.any(axis=-1)
    link = adjacency[NCARD:, :NCARD] > 0                            # [hint, card]
    p_correct = rset.astype(jnp.float32) @ q.T                      # [hint, card]
    p_placed = (link.astype(jnp.float32) * p_correct).sum(axis=-1)
    unrevealed = (~revealed).sum().astype(jnp.float32)
    waiting = (revealed & (~placed)).sum().astype(jnp.float32)
    value = (5.0 * unrevealed + 2.0 * waiting
             + (placed.astype(jnp.float32) * (2.0 * p_placed - 1.0)).sum())
    # One hint is revealed or placed per completed turn, so the public hint
    # state says exactly how many turns are left before the forced finish.
    used_turns = revealed.sum() + placed.sum()
    left = jnp.maximum(2 * NHINT - used_turns, 0).astype(jnp.float32)
    return rset, placed, revealed, value, left


def _hint_action(nf, pos, single, colour, partner_seen, q,
                 info, legal, rset, placed, revealed):
    locked = nf[:NCARD, 10] > 0
    n_placed = placed.sum()
    legal_reveal = legal[1617:1624]
    legal_place = legal[1624:2324].reshape(NHINT, NCELL)

    hidden = (~revealed) & legal_reveal
    reveal_h = jnp.argmax(jnp.where(hidden, -_hint_size(), -100.0))
    can_reveal = hidden.any()

    cell_of = pos[:, 0] * GRID + pos[:, 1]
    depth = info['adj_same'].sum(axis=-1).astype(jnp.float32)
    settled = single & (~locked) & (depth > 0)
    hit = jnp.take_along_axis(rset, colour[None, :], axis=1)        # [hint, card]
    avail = revealed & (~placed)
    cell_legal = jnp.take_along_axis(legal_place, cell_of[None, :], axis=1)
    good = hit & avail[:, None] & settled[None, :] & cell_legal
    # Narrow hints first: they cost the same but tell the partner more.  A
    # narrow hint on a card the partner has not seen hands them a colour they
    # would otherwise have to spend an observation on, while a wide hint on
    # such a card only takes an observation away from them.
    width = _hint_size()[:, None]
    tell = jnp.where(partner_seen[None, :], 0.0,
                     (width - 1.0) * 3.0 - jnp.where(width <= 1.0, 5.0, 0.0))
    cost = jnp.where(good, width * 10.0 - depth[None, :] + tell, BIG)
    flat = jnp.argmin(cost.reshape(-1))
    place_action = 1624 + (flat // NCARD) * NCELL + cell_of[flat % NCARD]
    can_place = good.any() & (n_placed <= 5)

    prob = rset.astype(jnp.float32) @ q.T                           # [hint, card]
    ok_any = avail[:, None] & (~locked)[None, :] & cell_legal
    flat2 = jnp.argmax(jnp.where(ok_any, prob, -1.0).reshape(-1))
    forced_action = 1624 + (flat2 // NCARD) * NCELL + cell_of[flat2 % NCARD]

    action = jnp.where(can_place, place_action,
                       jnp.where(can_reveal, 1617 + reveal_h, forced_action))
    return action.astype(jnp.int32)


def _observe_action(single, observed, partner_seen, legal):
    """Look at a card whose colour we still do not know.

    A card already pinned down by a placed hint or by counting needs no look,
    and a card nobody has seen is worth more to the team than one the partner
    has already covered.
    """
    legal_obs = legal[1:17]
    rank = (jnp.where(single, 400, 0) + jnp.where(observed, 200, 0)
            + jnp.where(partner_seen, 20, 0) + _cards())
    rank = jnp.where(legal_obs, rank, 10000)
    return (1 + jnp.argmin(rank)).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    nf = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    pos = obs.positions.astype(jnp.int32)

    prev = jnp.where(reset, jnp.zeros_like(memory['known']), memory['known'])
    known = jnp.maximum(prev, nf[:NCARD, 0:4].astype(jnp.int8))

    active = nf[0, 12] > 0
    substep = nf[0, 13]
    partner_seen = nf[:NCARD, 11] > 0

    poss, single, colour, observed, q, reliable = _beliefs(
        known, _hint_mask(nf, obs.adjacency_matrix))
    info = _board(pos, poss, single, colour)
    rset, placed, revealed, value, turns_left = _hints(
        nf, obs.adjacency_matrix, poss, single, colour, q)
    p_conn = _connect_prob(poss, single, q, info, partner_seen, reliable)

    move_key, _ = jax.random.split(rng)
    scores = _move_scores(pos, poss, single, colour, observed, partner_seen, q, info, move_key)
    move_mask = legal[17:1617].reshape(NCARD, NCELL)
    move_action = (17 + jnp.argmin(jnp.where(move_mask, scores, BIG).reshape(-1))).astype(jnp.int32)

    # Stop when stopping now beats a realistic view of playing on: each pair of
    # further turns costs TURN_VALUE points of hint value and repairs the board
    # by only about REPAIR in probability, and the public hint counts say how
    # many turns are actually left.
    horizon = jnp.minimum(turns_left, 2.0)
    later = (jnp.minimum(p_conn + REPAIR * horizon * 0.5, 1.0)
             * jnp.maximum(value - TURN_VALUE * horizon * 0.5, 0.0))
    stop = (p_conn * value >= later) & (p_conn >= P_FLOOR) & legal[2324]
    observe_action = _observe_action(single, observed, partner_seen, legal)
    hint_action = _hint_action(nf, pos, single, colour, partner_seen,
                               q, info, legal, rset, placed, revealed)

    action = jnp.where(
        substep == 0, jnp.where(stop, 2324, observe_action),
        jnp.where(substep == 1, observe_action,
                  jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    action = jnp.where(active, action, 0).astype(jnp.int32)

    trace = {
        'known_cards': observed.sum().astype(jnp.int32),
        'resolved': single.sum().astype(jnp.int32),
        'covered': (observed | partner_seen).sum().astype(jnp.int32),
        'fragments': info['frag'].astype(jnp.int32),
        'p_conn': p_conn.astype(jnp.float32),
        'value': value.astype(jnp.float32),
        'ended': (action == 2324),
    }
    return action, {'known': known}, trace
