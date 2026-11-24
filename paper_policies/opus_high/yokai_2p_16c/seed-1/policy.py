"""Cooperative 16-card Yokai policy.

Every decision is grounded in what this seat can verify by itself: colours it
observed, public positions / locks / hint sets, the public "partner has looked
at this card" flags, the board change made by the partner since the previous
turn, and the legal-action mask.  No private signalling convention is assumed
of the partner.

Facts the design leans on (context/environment/README.md):
  * terminal score is 5*U + 2*R + C - I when every colour is connected and 0
    otherwise, so connectivity dominates every other consideration;
  * a turn always spends one hint action, and reveals plus placements cannot
    exceed 7 each, so the seventh placement can only fall on the final turn:
    placing a hint never terminates the episode early;
  * only the active seat at substep 0 can end the episode, which is worth
    roughly two points per turn saved but scores 0 if a colour is split.

Move selection minimises "stragglers": the number of cards of known colour not
yet in the largest group of that colour, i.e. a lower bound on the moves still
required.  Colours that are not observed are handled with an explicit belief
(four cards per colour, plus whatever a placed hint reveals), so ties break
towards same-colour proximity, a compact board and neighbours that are likely
to match - all colour-independent and free of any private convention.

The seat ends the episode only when it can prove from its own information that
every colour is connected, which is worth about two points per turn saved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'yokai_straggler_greedy',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 4,
    'deck_size': 16,
    'grid_size': 10,
    'num_hints': 7,
    'num_nodes': 23,
    'node_features': 15,
    'num_actions': 2325,
    'policy_structure': 'shared_local',
}

N_CARDS = 16
N_COLORS = 4
N_HINTS = 7
GRID = 10
N_CELLS = GRID * GRID
TOPK = 48

MOVE_BASE = 17
REVEAL_BASE = 1617
PLACE_BASE = 1624
END_ACTION = 2324

# Objective weights (lower cost is better).
W_STRAG = 1000.0     # observed card not yet in the largest group of its colour
W_DIST = 10.0        # pairwise distance inside an observed colour
W_TIGHT = 1.0        # overall board compactness
W_EADJ = 120.0       # expected same-colour adjacency under colour uncertainty
W_KEEP = 1000.0      # probability that a move detaches the card from its group
W_PARTNER = 700.0    # moving a card only the partner has looked at
W_RECENT = 500.0     # moving the card the partner just moved


def init_agent():
    return {
        "pos": jnp.zeros((N_CARDS, 2), dtype=jnp.int32),
        "have": jnp.zeros((), dtype=jnp.int32),
    }


def _cells():
    """Grid cells; row index equals the action offset x*GRID + y."""
    index = jnp.arange(N_CELLS)
    return jnp.stack([index // GRID, index % GRID], axis=-1)


def _eye():
    idx = jnp.arange(N_CARDS)
    return idx[:, None] == idx[None, :]


def _lower():
    idx = jnp.arange(N_CARDS)
    return idx[:, None] > idx[None, :]


def _beliefs(known, color, pos, hint_cols, hint_placed, hint_pos, seen):
    """Per-card colour belief from counting plus public hint placements.

    The deck holds four cards of each colour, so the still hidden cards carry
    m_c = 4 - n_c of colour c and unobserved cards are otherwise exchangeable.
    A hint that has been placed on a card publicly restricts that card to the
    hint's colour set for any partner that prefers a correct placement (it is
    worth two points to them), so it is treated as evidence.  The restriction
    is only used when somebody had actually observed that card: a hint dropped
    on a card neither seat ever looked at is a guess and carries no colour
    information.
    """
    onehot = ((jnp.arange(N_COLORS)[None, :] == color[:, None])
              & known[:, None]).astype(jnp.float32)
    miss = (N_COLORS - onehot.sum(0))                                    # [4]
    on_card = ((pos[None, :, 0] == hint_pos[:, None, 0])
               & (pos[None, :, 1] == hint_pos[:, None, 1])
               & hint_placed[:, None]
               & seen[None, :])                                          # [7, 16]
    inside = jnp.all(jnp.where(on_card[:, :, None], hint_cols[:, None, :], True),
                     axis=0)                                             # [16, 4]
    allowed = inside.astype(jnp.float32)
    weight = miss[None, :] * allowed
    weight = jnp.where(weight.sum(-1, keepdims=True) > 0.0, weight,
                       miss[None, :] + 1e-6)
    weight = weight / weight.sum(-1, keepdims=True)
    return jnp.where(known[:, None], onehot, weight)


def _pair_prob(belief):
    """P(cards i and j share a colour) under the per-card beliefs."""
    return (belief @ belief.T) * jnp.logical_not(_eye()).astype(jnp.float32)


def _metrics(pos, known, color, prob):
    """Statistics of the observed colour groups on a hypothetical board."""
    delta = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    adj = delta == 1
    samec = (color[:, None] == color[None, :]) & known[:, None] & known[None, :]
    reach = ((adj & samec) | _eye()).astype(jnp.float32)
    reach = ((reach @ reach) > 0).astype(jnp.float32)
    reach = (reach @ reach) > 0            # (I+A)^4 closes any 4-node group
    compsize = (reach & known[None, :]).sum(-1)
    onehot = (jnp.arange(N_COLORS)[None, :] == color[:, None]) & known[:, None]
    n_c = onehot.sum(0)
    biggest = jnp.max(jnp.where(onehot, compsize[:, None], 0), axis=0)
    stragglers = (n_c - biggest).sum()
    pairdist = jnp.where(samec & _lower(), delta, 0).sum()
    spread = jnp.where(_lower(), delta, 0).sum()
    # cnt[i, c]: observed colour-c cards orthogonally next to card i
    cnt = adj.astype(jnp.float32) @ onehot.astype(jnp.float32)
    label = jnp.min(jnp.where(reach, jnp.arange(N_CARDS)[None, :], N_CARDS), axis=-1)
    expected_adj = jnp.where(adj & _lower(), prob, 0.0).sum()
    return (stragglers, pairdist, spread, cnt, label, compsize, onehot,
            expected_adj)


def _choose_observe(legal, known, partner_seen):
    """Prefer cards nobody has looked at, so the pair covers the whole deck.

    ``known`` already includes cards whose colour is pinned by counting or by a
    placed hint, so no look is spent on a card whose colour is settled.
    """
    legal_obs = legal[1:1 + N_CARDS]
    unknown = jnp.logical_not(known)
    score = (2.0 * unknown.astype(jnp.float32)
             + 1.0 * (unknown & jnp.logical_not(partner_seen)).astype(jnp.float32))
    score = jnp.where(legal_obs, score, -1.0e9)
    return (1 + jnp.argmax(score)).astype(jnp.int32)


def _card_penalty(known, partner_seen, recent):
    """Cost of disturbing a card whose placement the partner is managing."""
    unknown = jnp.logical_not(known)
    return (W_PARTNER * (unknown & partner_seen).astype(jnp.float32)
            + W_RECENT * recent.astype(jnp.float32))


def _choose_move(legal, pos, known, color, partner_seen, recent, prob):
    """Cheap ranking of all 1600 moves, then the exact objective on the best."""
    legal_move = legal[MOVE_BASE:MOVE_BASE + N_CARDS * N_CELLS]
    cells = _cells()

    d_cell = jnp.abs(cells[:, None, :] - pos[None, :, :]).sum(-1)        # [100, 16]
    adj_cell = (d_cell == 1).astype(jnp.float32)
    d_card = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    adj_card = d_card == 1
    samec = ((color[:, None] == color[None, :])
             & known[:, None] & known[None, :] & jnp.logical_not(_eye()))
    samecf = samec.astype(jnp.float32)

    pd_new = samecf @ d_cell.astype(jnp.float32).T                      # [16, 100]
    pd_old = (samecf * d_card.astype(jnp.float32)).sum(-1)[:, None]
    e_new = samecf @ adj_cell.T
    e_old = (samec & adj_card).sum(-1)[:, None].astype(jnp.float32)
    all_new = (d_cell.astype(jnp.float32).sum(-1)[None, :]
               - d_cell.astype(jnp.float32).T)                          # exclude self
    all_old = d_card.astype(jnp.float32).sum(-1)[:, None]

    ea_new = prob @ adj_cell.T                                          # [16, 100]
    ea_old = (prob * adj_card.astype(jnp.float32)).sum(-1)[:, None]

    # Probability that the card is next to at least one card of its own colour,
    # before and after the move.  Unobserved neighbours count with the odds
    # implied by the colours still missing, so a card is not pulled away from
    # neighbours that may already form its group.
    keep_old = 1.0 - jnp.prod(1.0 - prob * adj_card.astype(jnp.float32), axis=-1)
    keep_new = 1.0 - jnp.prod(
        1.0 - prob[:, None, :] * adj_cell[None, :, :], axis=-1)          # [16, 100]
    keep_term = W_KEEP * (keep_old[:, None] - keep_new)

    penalty = _card_penalty(known, partner_seen, recent)[:, None]
    cheap = (-200.0 * (e_new - e_old)
             + keep_term
             - W_EADJ * (ea_new - ea_old)
             + W_DIST * (pd_new - pd_old)
             + W_TIGHT * (all_new - all_old)
             + penalty)
    cheap = jnp.where(legal_move, cheap.reshape(-1), 1.0e9)
    _, top = jax.lax.top_k(-cheap, TOPK)

    def evaluate(idx):
        card = idx // N_CELLS
        cell = idx % N_CELLS
        npos = pos.at[card].set(cells[cell])
        strag, pairdist, spread, _, _, _, _, eadj = _metrics(
            npos, known, color, prob)
        cost = (W_STRAG * strag.astype(jnp.float32)
                + W_DIST * pairdist.astype(jnp.float32)
                + W_TIGHT * spread.astype(jnp.float32)
                - W_EADJ * eadj
                + penalty[card, 0])
        return (cost + keep_term[card, cell]
                + jnp.where(legal_move[idx], 0.0, 1.0e9))

    costs = jax.vmap(evaluate)(top)
    best = top[jnp.argmin(costs)]
    return (MOVE_BASE + best).astype(jnp.int32)


def _choose_hint(legal, pos, known, color, eff_known, eff_color, locked,
                 partner_seen, hint_cols, label, compsize, onehot):
    """Publish anchors with single-colour hints, otherwise reveal, then place.

    Reveals and placements each cap at seven, so the seventh placement can only
    fall on the last turn: at full length the terminal score is C - I whatever
    order is used.  That makes an early placement of a single-colour hint free,
    and it is worth doing on a card this seat has already grouped, because the
    placement publicly pins that card's colour for the partner and anchors the
    whole colour group at one agreed spot.  Every other placement waits, so
    cards stay movable while the board is still being sorted.
    """
    cell_idx = pos[:, 0] * GRID + pos[:, 1]
    place_id = PLACE_BASE + jnp.arange(N_HINTS)[:, None] * N_CELLS + cell_idx[None, :]
    legal_place = legal[place_id]                                        # [7, 16]
    correct = hint_cols[:, color] & known[None, :]
    hint_size = hint_cols.sum(-1).astype(jnp.float32)

    # Locking card j is safe only when every already locked card of the same
    # colour lies in the same observed component, so the group stays joinable.
    # Locks made by either seat count here: a card locked by the partner whose
    # colour is pinned by its own hint is just as immovable as an observed one.
    conflict = ((locked & eff_known)[None, :]
                & (eff_color[None, :] == eff_color[:, None])
                & (label[None, :] != label[:, None]))
    lock_ok = jnp.logical_not(jnp.any(conflict, axis=1))

    legal_reveal = legal[REVEAL_BASE:REVEAL_BASE + N_HINTS]
    # Small hint sets first: they are the ones that pin a colour publicly.
    reveal_act = REVEAL_BASE + jnp.argmax(jnp.where(legal_reveal, -hint_size, -1.0e9))

    free = 4 - (jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1) == 1).sum(-1)
    grouped = jnp.where((hint_size == 1.0)[:, None], compsize[None, :] >= 2,
                        compsize[None, :] >= 3)
    anchor = (legal_place & correct & grouped
              & (lock_ok & (free >= 1))[None, :])
    anchor_score = jnp.where(
        anchor,
        10.0 * compsize[None, :].astype(jnp.float32)
        + 6.0 * jnp.logical_not(partner_seen)[None, :].astype(jnp.float32)
        + free[None, :].astype(jnp.float32)
        + 8.0 * (4.0 - hint_size)[:, None],
        -1.0e9,
    )
    best_anchor = jnp.argmax(anchor_score.reshape(-1))
    anchor_act = (PLACE_BASE + (best_anchor // N_CARDS) * N_CELLS
                  + cell_idx[best_anchor % N_CARDS])

    # A card whose whole colour group is already observed and connected will
    # never have to move again, so locking it costs nothing.
    n_c = onehot.sum(0)
    settled = (compsize.astype(jnp.float32)
               - 2.0 * (N_COLORS - n_c[color]).astype(jnp.float32))
    score = jnp.where(
        legal_place,
        200.0 * correct.astype(jnp.float32)
        + 60.0 * lock_ok[None, :].astype(jnp.float32)
        + 40.0 * settled[None, :]
        + 30.0 * partner_seen[None, :].astype(jnp.float32)
        + 3.0 * (4.0 - hint_size)[:, None]
        + 6.0 * hint_size[:, None] * jnp.logical_not(known)[None, :].astype(jnp.float32),
        -1.0e9,
    )
    best = jnp.argmax(score.reshape(-1))
    place_act = (PLACE_BASE + (best // N_CARDS) * N_CELLS
                 + cell_idx[best % N_CARDS])

    action = jnp.where(
        jnp.any(anchor), anchor_act,
        jnp.where(jnp.any(legal_reveal), reveal_act, place_act))
    return action.astype(jnp.int32)


def _perms4():
    """All 24 permutations of four slots."""
    return jnp.asarray([
        [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1],
        [0, 3, 1, 2], [0, 3, 2, 1], [1, 0, 2, 3], [1, 0, 3, 2],
        [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
        [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0],
        [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 0, 2, 1],
        [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0]],
        dtype=jnp.int32)


def _end_proof(legal, pos, known, color, cnt, stragglers, onehot):
    """True only when the seat can prove that every colour is connected.

    Two independent sufficient checks, both sound given what this seat holds:

    * every known colour group is one component and each card of unknown
      colour touches a known card of every colour it could still be; or
    * at most four cards are of unknown colour, so every assignment of the
      still missing colours to them can be enumerated, and every assignment
      consistent with the four-per-colour counts leaves all colours connected.
    """
    n_c = onehot.sum(0)
    miss = N_COLORS - n_c
    unknown = jnp.logical_not(known)
    cond_a = stragglers == 0
    need = (miss > 0)[None, :] & unknown[:, None]
    touching = cond_a & jnp.logical_not(jnp.any(need & (cnt <= 0)))

    # Exhaustive check over the remaining colour assignments.
    n_unknown = unknown.sum()
    order = jnp.argsort(known.astype(jnp.int32))          # unknown cards first
    bound = jnp.cumsum(miss)
    slot = jnp.searchsorted(bound, jnp.arange(N_COLORS), side="right")
    slot = jnp.clip(slot, 0, N_COLORS - 1)
    perms = _perms4()
    ones = jnp.ones((N_CARDS,), dtype=bool)

    def consistent_and_connected(perm):
        active = jnp.arange(N_COLORS) < n_unknown
        relevant = jnp.all(jnp.where(active, perm < n_unknown, True))
        full = color
        for k in range(N_COLORS):
            target = order[k]
            full = jnp.where(
                (jnp.arange(N_CARDS) == target) & active[k],
                slot[perm[k]], full)
        strag = _metrics(pos, ones, full, jnp.zeros((N_CARDS, N_CARDS)))[0]
        return jnp.logical_not(relevant) | (strag == 0)

    exhaustive = (n_unknown <= N_COLORS) & jnp.all(
        jax.vmap(consistent_and_connected)(perms))
    return (touching | exhaustive) & legal[END_ACTION]


def agent_act(obs, reset, memory, rng):
    nf = obs.node_features.astype(jnp.int32)
    card_nf = nf[:N_CARDS]
    color_onehot = card_nf[:, 0:N_COLORS]
    known = color_onehot.sum(-1) > 0
    color = jnp.argmax(color_onehot, axis=-1)
    pos = obs.positions.astype(jnp.int32)
    locked = card_nf[:, 10] > 0
    partner_seen = card_nf[:, 11] > 0
    active = nf[0, 12] > 0
    substep = nf[0, 13]

    hint_cols = nf[N_CARDS:N_CARDS + N_HINTS, N_COLORS:2 * N_COLORS] > 0

    prev_pos = jnp.where(reset, jnp.zeros_like(memory["pos"]), memory["pos"])
    have = jnp.where(reset, 0, memory["have"])
    recent = (have > 0) & jnp.any(pos != prev_pos, axis=-1)

    legal = obs.legal_actions
    hint_placed = nf[N_CARDS:N_CARDS + N_HINTS, 10] > 0
    hint_pos = nf[N_CARDS:N_CARDS + N_HINTS, 8:10]
    informed = partner_seen | known
    hard_belief = _beliefs(known, color, pos, hint_cols, hint_placed, hint_pos,
                           informed)
    hard_prob = _pair_prob(hard_belief)
    strag, pairdist, spread, cnt, label, compsize, onehot, _ = _metrics(
        pos, known, color, hard_prob)

    # A card whose colour is pinned by the counts still missing, or by a hint
    # already placed on it, can be treated as observed for move planning.
    certain = jnp.max(hard_belief, axis=-1) > 0.999
    eff_known = known | certain
    eff_color = jnp.where(known, color, jnp.argmax(hard_belief, axis=-1))
    eff_belief = _beliefs(eff_known, eff_color, pos, hint_cols, hint_placed,
                          hint_pos, informed)
    eff_prob = _pair_prob(eff_belief)

    (eff_strag, _, _, eff_cnt, eff_label, eff_compsize,
     eff_onehot, _) = _metrics(pos, eff_known, eff_color, eff_prob)
    end_ok = _end_proof(legal, pos, eff_known, eff_color, eff_cnt, eff_strag,
                        eff_onehot)
    observe_act = _choose_observe(legal, eff_known, partner_seen)

    def move_branch(_):
        return _choose_move(legal, pos, eff_known, eff_color, partner_seen,
                            recent, eff_prob)

    def hint_branch(_):
        return _choose_hint(legal, pos, known, color, eff_known, eff_color,
                            locked, partner_seen, hint_cols, eff_label,
                            eff_compsize, onehot)

    def obs_branch(_):
        return observe_act

    def first_branch(_):
        return jnp.where(end_ok, jnp.int32(END_ACTION), observe_act)

    action = jax.lax.switch(
        jnp.clip(substep, 0, 3),
        [first_branch, obs_branch, move_branch, hint_branch],
        0,
    )
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    action = jnp.where(active, action, jnp.int32(0)).astype(jnp.int32)

    # Remember the board as it stands after this seat's own move, so that next
    # turn the single card the partner relocated can be identified.
    store = active & (substep == 3)
    next_memory = {
        "pos": jnp.where(store, pos, prev_pos),
        "have": jnp.where(store, jnp.ones((), jnp.int32), have),
    }
    trace = {
        "stragglers": strag.astype(jnp.int32),
        "known": known.sum().astype(jnp.int32),
        "end_ok": end_ok.astype(jnp.int32),
        "locked": locked.sum().astype(jnp.int32),
        "pinned": certain.sum().astype(jnp.int32),
    }
    return action.astype(jnp.int32), next_memory, trace
