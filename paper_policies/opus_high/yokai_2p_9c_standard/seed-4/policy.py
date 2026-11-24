"""Two-player Yokai policy: exact colour posterior, greedy repair, cheap hints.

Every decision uses only public board information (positions, locks, revealed
hint sets, the public partner-observation flags) plus this seat's own private
colour observations accumulated in memory.  No private signalling convention is
used or assumed, so the behaviour does not depend on the partner running this
same code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'posterior_repair', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
NUM_HINTS = 4
GRID = 9
NUM_CELLS = 81

# Action layout (see environment README).
A_NOOP = 0
A_OBSERVE = 1
A_MOVE = 10
A_REVEAL = 739
A_PLACE = 743
A_END = 1067

# Continuation value of not ending, indexed by full-turn number 0..7.  Ending
# after k cheap hint actions is worth at most 20, 17, 16, 13, 12, 9, 8, 5 for
# k = 0..7 when every colour is clustered.
VCONT = [2.5] * 8

P_MIN = 0.4           # minimum belief that the board is clustered before ending
TRUST_BETA = 0.5      # how strongly to read the board as an informed arrangement
OBS_UNSEEN = 1.0      # prefer inspecting cards the partner has not inspected
OBS_PIVOT = 3.0       # prefer cards whose colour decides whether the board is done
MOVE_BONUS = 4.0      # extra weight on "certainly solved after this move"
SPREAD_WEIGHT = 0.15  # pull same-coloured cards together before they connect
CLOSE_CREDIT = 0.0    # partial credit for same-coloured cards two cells apart
LOCK_PENALTY = 6.0    # discourage locking a card whose colour group is unsolved
INFO_PENALTY = 2.5    # never lock away a card whose colour is still unknown here
PSEEN_BONUS = 0.5     # prefer locking cards the partner has already inspected
RISK_PENALTY = 4.0    # never strand two locked cards of one colour far apart
NEG = -1.0e9


def _colourings():
    """All 1680 assignments of three cards to each of the three colours."""
    rows = []
    for a0 in range(NUM_CARDS):
        for a1 in range(a0 + 1, NUM_CARDS):
            for a2 in range(a1 + 1, NUM_CARDS):
                rest = [i for i in range(NUM_CARDS) if i not in (a0, a1, a2)]
                for b0 in range(6):
                    for b1 in range(b0 + 1, 6):
                        for b2 in range(b1 + 1, 6):
                            row = [2] * NUM_CARDS
                            row[a0] = 0
                            row[a1] = 0
                            row[a2] = 0
                            row[rest[b0]] = 1
                            row[rest[b1]] = 1
                            row[rest[b2]] = 1
                            rows.append(row)
    return rows


def _constants():
    col = jnp.asarray(_colourings(), dtype=jnp.int8)
    onehot = (col[:, :, None] == jnp.arange(NUM_COLORS, dtype=jnp.int8)[None, None, :]).astype(jnp.float32)
    same_off = ((col[:, :, None] == col[:, None, :]).astype(jnp.float32)
                * (1.0 - jnp.eye(NUM_CARDS, dtype=jnp.float32)))
    grid = jnp.arange(GRID, dtype=jnp.int32)
    cells = jnp.stack([jnp.repeat(grid, GRID), jnp.tile(grid, GRID)], axis=-1)
    return col, onehot, same_off, cells, jnp.asarray(VCONT, dtype=jnp.float32)


def _need(pairs, close_pairs):
    """Progress of one colour towards a connected group: 2 when its three cards
    are already connected, 1 for one adjacent pair, and a partial credit when a
    single relocation could still bridge two cards two cells apart."""
    graded = jnp.minimum(pairs, 2.0) + CLOSE_CREDIT * jnp.minimum(close_pairs, 1.0)
    return jnp.minimum(graded, 2.0)


def init_agent():
    return {
        'known': jnp.full((NUM_CARDS,), -1, dtype=jnp.int8),
        'pseen': jnp.zeros((NUM_CARDS,), dtype=jnp.bool_),
        'count': jnp.asarray(0, dtype=jnp.int32),
        'seat': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_act(obs, reset, memory, rng):
    col_table, onehot, same_off, cells, vcont = _constants()

    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    substep = jnp.clip(nf[0, 11], 0, 3)
    active = nf[0, 10] > 0

    fresh = init_agent()
    known_prev = jnp.where(reset, fresh['known'], memory['known'])
    pseen_prev = jnp.where(reset, fresh['pseen'], memory['pseen'])
    count = jnp.where(reset, fresh['count'], memory['count'])
    seat = jnp.where(reset, jnp.where(active, 0, 1).astype(jnp.int32), memory['seat'])

    # --- private knowledge ------------------------------------------------
    colour_bits = nf[0:NUM_CARDS, 0:NUM_COLORS]
    visible = colour_bits.sum(axis=1) > 0
    colour_now = jnp.argmax(colour_bits, axis=1).astype(jnp.int8)
    known = jnp.where(visible, colour_now, known_prev)
    pseen = pseen_prev | (nf[0:NUM_CARDS, 9] > 0)
    locked = nf[0:NUM_CARDS, 8] > 0

    # --- posterior over colourings consistent with own observations -------
    consistent = jnp.all((known[None, :] < 0) | (col_table == known[None, :]), axis=1)

    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(axis=-1)
    adj = (dist == 1).astype(jnp.float32)
    # A card that some seat has inspected tends to have been moved next to its
    # own colour by whoever knew it, so the board itself is weak evidence about
    # the colours this seat has not seen.
    touch = jnp.minimum(jnp.einsum('jl,cjl->cj', adj, same_off), 1.0)
    trust = (pseen | (known >= 0)).astype(jnp.float32)
    logits = jnp.where(consistent, TRUST_BETA * (touch @ trust), -1.0e30)
    w = jnp.exp(logits - logits.max())
    w = w / jnp.maximum(w.sum(), 1e-9)
    close = (dist == 2).astype(jnp.float32)          # bridgeable by one relocation
    base = 0.5 * jnp.einsum('ij,cik,cjk->ck', adj, onehot, onehot)        # [C, 3]
    base_close = 0.5 * jnp.einsum('ij,cik,cjk->ck', close, onehot, onehot)
    need = _need(base, base_close)                                        # [C, 3]
    need_sum = need.sum(axis=-1)                                          # [C]
    solved = need_sum > 5.5                                               # all colours clustered
    p_solved = (w * solved).sum()

    # --- hint state -------------------------------------------------------
    hset = (nf[9:13, 3:6] > 0).astype(jnp.float32)                        # [4, 3]
    revealed = hset.sum(axis=1) > 0
    placed = nf[9:13, 8] > 0
    hint_card = obs.adjacency_matrix[9:13, 0:NUM_CARDS].astype(jnp.float32)
    match = jnp.einsum('cjk,hk->chj', onehot, hset)                       # [C, 4, 9]
    correct = jnp.einsum('chj,hj->ch', match, hint_card)                  # [C, 4]
    n_unrevealed = (~revealed).sum().astype(jnp.float32)
    n_held = (revealed & (~placed)).sum().astype(jnp.float32)
    placed_f = placed.astype(jnp.float32)
    score_now = (5.0 * n_unrevealed + 2.0 * n_held
                 + ((2.0 * correct - 1.0) * placed_f[None, :]).sum(axis=1))
    ev_end = (w * solved * score_now).sum()

    turn = jnp.clip(count // 4, 0, 7)

    def observe_action():
        # How much would learning this card's colour move the belief about the
        # board already being clustered?  Resolving that is what makes an early
        # ending affordable.
        num = jnp.einsum('c,cjk->jk', w * solved.astype(jnp.float32), onehot)
        den = jnp.einsum('c,cjk->jk', w, onehot)
        cond = num / jnp.maximum(den, 1e-6)
        pivot = (cond.max(axis=1) - cond.min(axis=1)) * (known < 0).astype(jnp.float32)
        value = (4.0 * (known < 0).astype(jnp.float32)
                 + OBS_UNSEEN * (~pseen).astype(jnp.float32)
                 + OBS_PIVOT * pivot
                 - 0.001 * jnp.arange(NUM_CARDS, dtype=jnp.float32))
        value = jnp.where(legal[A_OBSERVE:A_OBSERVE + NUM_CARDS], value, NEG)
        return (A_OBSERVE + jnp.argmax(value)).astype(jnp.int32)

    def branch_observe_or_end(_):
        act = observe_action()
        stop = legal[A_END] & (ev_end >= vcont[turn]) & (p_solved >= P_MIN)
        return jnp.where(stop, jnp.int32(A_END), act)

    def branch_observe(_):
        return observe_action()

    def branch_move(_):
        cell_dist = jnp.abs(cells[:, None, :] - pos[None, :, :]).sum(axis=-1).astype(jnp.float32)
        cell_adj = (cell_dist == 1.0).astype(jnp.float32)                 # [P, 9]
        cell_close = (cell_dist == 2.0).astype(jnp.float32)               # [P, 9]
        # Only the moved card's own colour changes its pair counts.
        new_deg = jnp.einsum('cij,pj->cip', same_off, cell_adj)           # [C, 9, P]
        new_close = jnp.einsum('cij,pj->cip', same_off, cell_close)
        base_col = jnp.einsum('cik,ck->ci', onehot, base)                 # [C, 9]
        close_col = jnp.einsum('cik,ck->ci', onehot, base_close)
        need_col = jnp.einsum('cik,ck->ci', onehot, need)                 # [C, 9]
        deg = jnp.einsum('ij,cij->ci', adj, same_off)                     # [C, 9]
        deg_close = jnp.einsum('ij,cij->ci', close, same_off)
        pairs_after = base_col[:, :, None] - deg[:, :, None] + new_deg
        close_after = close_col[:, :, None] - deg_close[:, :, None] + new_close
        need_after = (need_sum[:, None, None] - need_col[:, :, None]
                      + _need(pairs_after, close_after))                  # [C, 9, P]
        # Secondary potential: keep same-coloured cards close, so that the
        # later fully informed moves can finish the clustering.
        far = dist.astype(jnp.float32)
        spread = 0.5 * jnp.einsum('ij,cij->c', far, same_off)             # [C]
        spread_i = jnp.einsum('ij,cij->ci', far, same_off)                # [C, 9]
        spread_new = jnp.einsum('cij,pj->cip', same_off, cell_dist)       # [C, 9, P]
        after = spread[:, None, None] - spread_i[:, :, None] + spread_new
        gain = (need_after + MOVE_BONUS * (need_after > 5.5).astype(jnp.float32)
                - SPREAD_WEIGHT * after)
        value = jnp.einsum('c,cip->ip', w, gain)
        displacement = jnp.abs(cells[None, :, :] - pos[:, None, :]).sum(axis=-1).astype(jnp.float32)
        value = value - 0.01 * displacement - 0.05 * (known < 0).astype(jnp.float32)[:, None]
        mask = legal[A_MOVE:A_MOVE + NUM_CARDS * NUM_CELLS].reshape(NUM_CARDS, NUM_CELLS)
        value = jnp.where(mask, value, NEG)
        return (A_MOVE + jnp.argmax(value.reshape(-1))).astype(jnp.int32)

    def branch_hint(_):
        p_match = jnp.einsum('c,chj->hj', w, match)                       # [4, 9]
        group_ok = jnp.einsum('cjk,ck->cj', onehot, (need > 1.5).astype(jnp.float32))
        safe = jnp.einsum('c,cj->j', w, group_ok)                         # [9]
        # Locking a card also removes it from both seats' observable pool for
        # the rest of the episode, so prefer cards this seat already knows and
        # that the partner has already inspected.
        unknown_to_me = (known < 0).astype(jnp.float32)
        # Two locked cards of one colour that are more than two cells apart can
        # never be joined again, which forfeits the whole episode.
        stranded = jnp.einsum('cjl,jl->cj', same_off,
                              locked.astype(jnp.float32)[None, :] * (dist > 2).astype(jnp.float32))
        risk = jnp.einsum('c,cj->j', w, (stranded > 0.5).astype(jnp.float32))
        place_value = (2.0 * p_match - 3.0
                       - LOCK_PENALTY * (1.0 - safe)[None, :]
                       - INFO_PENALTY * unknown_to_me[None, :]
                       - RISK_PENALTY * risk[None, :]
                       + PSEEN_BONUS * pseen.astype(jnp.float32)[None, :])
        cell_id = pos[:, 0] * GRID + pos[:, 1]
        place_id = A_PLACE + jnp.arange(NUM_HINTS, dtype=jnp.int32)[:, None] * NUM_CELLS + cell_id[None, :]
        place_value = jnp.where(legal[place_id], place_value, NEG)
        reveal_value = -3.0 + 0.1 * (jnp.arange(NUM_HINTS) > 0).astype(jnp.float32)
        reveal_value = jnp.where(legal[A_REVEAL:A_REVEAL + NUM_HINTS], reveal_value, NEG)
        flat = jnp.argmax(place_value.reshape(-1))
        best_place = place_value.reshape(-1)[flat]
        best_reveal_h = jnp.argmax(reveal_value)
        best_reveal = reveal_value[best_reveal_h]
        act_place = (A_PLACE + (flat // NUM_CARDS) * NUM_CELLS
                     + cell_id[flat % NUM_CARDS]).astype(jnp.int32)
        act_reveal = (A_REVEAL + best_reveal_h).astype(jnp.int32)
        return jnp.where(best_place >= best_reveal, act_place, act_reveal)

    action = jax.lax.switch(
        substep, [branch_observe_or_end, branch_observe, branch_move, branch_hint], 0)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    action = jnp.where(active, action, jnp.int32(A_NOOP)).astype(jnp.int32)

    next_memory = {
        'known': known.astype(jnp.int8),
        'pseen': pseen,
        'count': (count + 1).astype(jnp.int32),
        'seat': seat.astype(jnp.int32),
    }
    trace = {
        'p_solved': p_solved.astype(jnp.float32),
        'ev_end': ev_end.astype(jnp.float32),
        'n_known': (known >= 0).sum().astype(jnp.float32),
        'n_beliefs': consistent.sum().astype(jnp.float32),
        'turn': turn.astype(jnp.float32),
        'seat': seat.astype(jnp.float32),
    }
    return action, next_memory, trace
