"""Cooperative 9-card Yokai policy.

Belief
------
Private memory keeps every colour this seat has observed during the episode (the
environment only exposes a colour while the observing turn lasts).  A posterior
over the 1680 balanced colour assignments drives every decision.  It has two
layers:

* the *verified* layer, a uniform distribution over the assignments consistent
  with this seat's own observations, and
* a *soft* layer that additionally reweights by public partner behaviour --
  a card the partner moved next to other cards probably shares a colour with one
  of them, and a card the partner covered with a revealed hint probably has a
  colour from that hint's set.

Behavioural evidence is only counted for cards the partner has publicly been
seen inspecting, since a partner cannot express colour intent about a card whose
colour it has never looked at.  This is inference from public facts, not a
signalling code: a partner that moves cards for other reasons merely makes the
soft layer less informative, and the seat still plays from the verified layer.

Decisions
---------
* substep 0 -- end the episode when P(all colours connected) * the score the
  arrangement is currently worth beats an estimate of the value of playing on,
  otherwise inspect the card that is most uncertain and most decisive for
  connectivity (measured on the verified layer, which also keeps the partner's
  implied colour claims falsifiable);
* substep 1 -- inspect a second such card;
* substep 2 -- move the (card, destination) pair maximising posterior
  P(all colours connected), then expected connected colours and expected
  same-colour adjacencies, with a penalty for undoing the partner's last move
  and for shuffling cards whose colour this seat does not know;
* substep 3 -- place a revealed hint on a card whose colour certainly belongs to
  the hint set, which costs one point instead of the three a fresh reveal costs
  and locks a card the partner can no longer disturb; never place it where it
  would strand a colour whose other card is already locked elsewhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'yokai_belief_grouper',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 3,
    'deck_size': 9,
    'grid_size': 9,
    'num_hints': 4,
    'num_nodes': 13,
    'node_features': 13,
    'num_actions': 1068,
    'policy_structure': 'shared_local',
}

N_CARDS = 9
N_COLORS = 3
GRID = 9
N_HINTS = 4

A_OBS = 1        # observe card i -> 1 + i
A_MOVE = 10      # move card i to (x, y) -> 10 + 81 i + 9 x + y
A_REVEAL = 739   # reveal hint h -> 739 + h
A_PLACE = 743    # place hint h at (x, y) -> 743 + 81 h + 9 x + y
A_END = 1067

# Move value weights.
W_PCONN = 1000.0     # posterior P(all three colours connected)
W_CERT = 0.0         # extra credit for same-colour adjacencies of known cards
W_ENUM = 4.0         # expected number of connected colours
W_EPAIR = 0.7        # expected same-colour adjacencies
W_DENSITY = 0.05     # keep the cards compact, which keeps options open
W_RECENT = 2.0       # do not undo the partner's latest move
W_UNKNOWN = 0.3      # mild preference for moving cards whose colour we know

# Observation weights.
W_COPY = 0.0         # bonus for inspecting a card the partner has inspected
W_DISC = 2.0         # bonus for the card that best resolves connectivity

# Strength of the public behavioural evidence, in log-likelihood units.
LAM_MOVE = 1.0       # a card the partner moved next to another
LAM_HINT = 1.0       # a card the partner covered with a revealed hint
PLACE_EARLY = True   # place matching hints before every hint has been revealed

# End-game value model.  Playing two more turns costs a few points of hint
# value while closing part of the remaining uncertainty; ending is worth the
# current arrangement's score times the probability that it really is connected.
TURN_COST = 4.0      # fallback cost of two further turns
LAST_TURN_COST = 1.5 # cost of the single hint action left on the final turn
LEARN_FRAC = 0.22    # share of the remaining doubt two more turns resolve
END_FLOOR = 2.6      # value of simply playing the episode out to its own end
P_END_SOFT = 1.0     # how much partner-behaviour evidence the end rule trusts

# Optional discount of behavioural evidence by how often the partner's implied
# colour claims agreed with colours this seat later observed.  Disabled
# (TRUST_MIX = 0) because self-play and cross-play probes both scored better
# without it, but retained since it is the natural guard against a partner whose
# movements carry no colour information at all.
TRUST_PRIOR_N = 2.0  # pseudo-observations backing the prior agreement rate
TRUST_PRIOR_R = 0.6  # prior probability that a partner join is really same colour
TRUST_CHANCE = 0.28  # agreement rate expected from a partner moving at random
TRUST_SPAN = 0.17    # rate above chance at which the partner is fully trusted
TRUST_MIX = 0.0      # how much the measured agreement discounts the evidence


def _coloring_rows():
    """All 1680 balanced assignments of 3 colours to 9 cards (trace time only)."""
    rows = []
    for a in range(N_CARDS):
        for b in range(a + 1, N_CARDS):
            for c in range(b + 1, N_CARDS):
                first = (a, b, c)
                rest = [i for i in range(N_CARDS) if i not in first]
                for x in range(6):
                    for y in range(x + 1, 6):
                        for z in range(y + 1, 6):
                            row = [2] * N_CARDS
                            row[a] = 0
                            row[b] = 0
                            row[c] = 0
                            row[rest[x]] = 1
                            row[rest[y]] = 1
                            row[rest[z]] = 1
                            rows.append(row)
    return rows


def _tables():
    """Static JAX tables; built eagerly while tracing, folded into constants."""
    col = jnp.asarray(_coloring_rows(), dtype=jnp.int32)          # [1680, 9]
    pi = []
    pj = []
    for i in range(N_CARDS):
        for j in range(i + 1, N_CARDS):
            pi.append(i)
            pj.append(j)
    pair_i = jnp.asarray(pi, dtype=jnp.int32)
    pair_j = jnp.asarray(pj, dtype=jnp.int32)
    onehot = (col[:, :, None] == jnp.arange(N_COLORS)[None, None, :]).astype(jnp.float32)
    same_f = []
    for f in range(N_COLORS):
        same_f.append(((col[:, pair_i] == f) & (col[:, pair_j] == f)).astype(jnp.float32))
    pairsame = jnp.stack(same_f, axis=1)                          # [1680, 3, 36]
    eye = jnp.eye(N_CARDS, dtype=bool)
    sameoff = ((col[:, :, None] == col[:, None, :]) & (~eye)[None, :, :]).astype(jnp.float32)
    dirs = jnp.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
    return col, pair_i, pair_j, onehot, pairsame, sameoff, dirs


def init_agent():
    return {
        'known': jnp.full((N_CARDS,), -1, dtype=jnp.int32),
        'seen_by_partner': jnp.zeros((N_CARDS,), dtype=jnp.int32),
        'recent': jnp.zeros((N_CARDS,), dtype=jnp.float32),
        'prev_pos': jnp.zeros((N_CARDS, 2), dtype=jnp.int32),
        'affinity': jnp.zeros((N_CARDS, N_CARDS), dtype=jnp.float32),
        'step': jnp.asarray(0, dtype=jnp.int32),
    }


def agent_act(obs, reset, memory, rng):
    colorings, pair_i, pair_j, onehot, pairsame, sameoff, dirs = _tables()

    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions
    reset_b = jnp.asarray(reset, dtype=bool)

    substep = nf[0, 11]
    active = nf[0, 10] > 0

    known_prev = jnp.where(reset_b, jnp.full((N_CARDS,), -1, dtype=jnp.int32), memory['known'])
    partner_prev = jnp.where(reset_b, jnp.zeros((N_CARDS,), dtype=jnp.int32),
                             memory['seen_by_partner'])
    recent_prev = jnp.where(reset_b, jnp.zeros((N_CARDS,), dtype=jnp.float32), memory['recent'])
    prev_pos = jnp.where(reset_b, pos, memory['prev_pos'])
    aff_prev = jnp.where(reset_b, jnp.zeros((N_CARDS, N_CARDS), dtype=jnp.float32),
                         memory['affinity'])
    step_prev = jnp.where(reset_b, jnp.asarray(0, dtype=jnp.int32), memory['step'])

    # --- remember privately revealed colours -------------------------------
    col_onehot = nf[:N_CARDS, 0:N_COLORS]
    revealed_now = jnp.sum(col_onehot, axis=1) > 0
    seen_color = jnp.argmax(col_onehot, axis=1)
    known = jnp.where(revealed_now, seen_color, known_prev)
    is_known = known >= 0

    # --- current board geometry -------------------------------------------
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    adj = (dist == 1).astype(jnp.float32)                    # [9, 9]
    adj_pair = adj[pair_i, pair_j]                           # [36]

    # --- track what the partner looked at and what it moved -----------------
    partner_seen = jnp.maximum(partner_prev, (nf[:N_CARDS, 9] > 0).astype(jnp.int32))
    moved_now = jnp.any(pos != prev_pos, axis=1)
    recent = jnp.where(substep == 0, recent_prev * 0.5, recent_prev)
    recent = jnp.where(moved_now & (~active), 1.0, recent)

    # A card the partner just placed beside others is weak public evidence that
    # it shares their colour; spread one unit of evidence over its new
    # neighbours.  This reads public behaviour only and never needs the partner
    # to follow any agreed code.
    mv = (moved_now & (~active)).astype(jnp.float32)
    share = mv / jnp.maximum(jnp.sum(adj, axis=1), 1.0)
    # Only cards the partner has publicly been seen inspecting can carry colour
    # intent: a partner that shuffles a card it never looked at says nothing.
    seen_f = (partner_seen > 0).astype(jnp.float32)
    delta = share[:, None] * adj * seen_f[:, None] * seen_f[None, :]
    affinity = jnp.clip(aff_prev + delta + delta.T, 0.0, 2.0)

    # --- posterior over consistent colourings ------------------------------
    hint_sets_raw = nf[N_CARDS:, 3:6].astype(jnp.float32)    # [4, 3]
    placed_raw = (nf[N_CARDS:, 8] > 0).astype(jnp.float32)   # [4]
    hpos = nf[N_CARDS:, 6:8]
    at_h = jnp.all(pos[None, :, :] == hpos[:, None, :], axis=2).astype(jnp.float32)
    ev_card = jnp.einsum('h,hi,hf->if', placed_raw, at_h, hint_sets_raw)   # [9, 3]

    # How reliable has this partner's behaviour been so far?  Every join it made
    # between two cards whose colours this seat has since observed is a free
    # validation sample; a partner whose joins are no better than chance gets
    # its evidence discounted to nothing.
    both_known = is_known[:, None] & is_known[None, :]
    same_known = both_known & (known[:, None] == known[None, :]) & (~jnp.eye(N_CARDS, dtype=bool))
    mass = jnp.sum(affinity * both_known.astype(jnp.float32))
    hits = jnp.sum(affinity * same_known.astype(jnp.float32))
    rate = ((hits + TRUST_PRIOR_R * TRUST_PRIOR_N)
            / (mass + TRUST_PRIOR_N))
    trust = jnp.clip((rate - TRUST_CHANCE) / TRUST_SPAN, 0.0, 1.0)
    trust = TRUST_MIX * trust + (1.0 - TRUST_MIX)

    consistent = jnp.all((known[None, :] < 0) | (colorings == known[None, :]), axis=1)
    logit = trust * (LAM_MOVE * jnp.einsum('ij,cij->c', affinity, sameoff)
                     + LAM_HINT * jnp.einsum('if,cif->c', ev_card, onehot))
    logit = logit - jnp.max(logit)
    w_hard = consistent.astype(jnp.float32)
    w_hard = w_hard / jnp.maximum(jnp.sum(w_hard), 1.0)      # verified belief only
    w = consistent.astype(jnp.float32) * jnp.exp(logit)
    w = w / jnp.maximum(jnp.sum(w), 1e-9)                    # [1680] soft belief
    pc = jnp.einsum('c,cif->if', w, onehot)                  # [9, 3] colour marginals
    pc_hard = jnp.einsum('c,cif->if', w_hard, onehot)        # verified marginals only

    e0 = jnp.einsum('cfp,p->cf', pairsame, adj_pair)         # [1680, 3]
    ok_f = (e0 >= 2.0).astype(jnp.float32)
    total_ok = jnp.sum(ok_f, axis=1)                         # [1680]
    conn_all = (total_ok >= 3.0).astype(jnp.float32)
    p_conn = jnp.sum(w * conn_all)
    # The end decision must stay calibrated, so it only trusts colours this seat
    # has actually seen, never the soft evidence read off partner behaviour.
    p_conn_hard = jnp.sum(w_hard * conn_all)
    total_pairs = jnp.sum(e0, axis=1)                        # [1680]

    deg_same = jnp.einsum('cmj,mj->cm', sameoff, adj)        # [1680, 9]
    own_e0 = jnp.take_along_axis(e0, colorings, axis=1)      # [1680, 9]
    own_ok = (own_e0 >= 2.0).astype(jnp.float32)
    others_cnt = total_ok[:, None] - own_ok                  # [1680, 9]
    others_ok = (others_cnt >= 2.0).astype(jnp.float32)
    rest_pairs = own_e0 - deg_same                           # 0 or 1

    # certain (fully known) same-colour structure
    cert_same = ((known[:, None] == known[None, :]) & is_known[:, None]
                 & (~jnp.eye(N_CARDS, dtype=bool))).astype(jnp.float32)
    cert_deg = jnp.sum(cert_same * adj, axis=1)              # [9]
    cert_base = 0.5 * jnp.sum(cert_deg)

    # --- hint bookkeeping ---------------------------------------------------
    hint_sets = hint_sets_raw                                # [4, 3]
    hint_revealed = jnp.sum(hint_sets, axis=1) > 0
    hint_placed = nf[N_CARDS:, 8] > 0
    hint_pos = nf[N_CARDS:, 6:8]
    n_placed = jnp.sum(hint_placed.astype(jnp.int32))
    n_unrevealed = jnp.sum((~hint_revealed).astype(jnp.int32))
    n_ready = jnp.sum((hint_revealed & (~hint_placed)).astype(jnp.int32))

    p_match = hint_sets @ pc.T                               # [4, 9]
    at_hint = at_h
    placed_quality = jnp.sum(at_hint * p_match, axis=1)      # [4]
    placed_value = jnp.sum(jnp.where(hint_placed, 2.0 * placed_quality - 1.0, 0.0))
    score_now = (5.0 * n_unrevealed.astype(jnp.float32)
                 + 2.0 * n_ready.astype(jnp.float32) + placed_value)
    turns_done = (N_HINTS - n_unrevealed) + n_placed

    k_obs, k_move, k_hint = jax.random.split(rng, 3)

    # --- substep 0/1: which card to observe --------------------------------
    # Uncertainty is measured on the verified belief: cards that only *look*
    # settled because of partner behaviour are still worth inspecting, both for
    # their own sake and because they validate that behaviour.
    ent = -jnp.sum(pc_hard * jnp.log(pc_hard + 1e-9), axis=1)  # [9]
    copy_bonus = W_COPY * ((partner_seen > 0) & (~is_known)).astype(jnp.float32)
    joint = jnp.einsum('c,cif->if', w * conn_all, onehot)
    disc = jnp.sum(joint * joint / jnp.maximum(pc, 1e-6), axis=1) - p_conn * p_conn
    obs_val = (copy_bonus + ent + W_DISC * disc
               + 1e-3 * jax.random.uniform(k_obs, (N_CARDS,)))
    obs_val = jnp.where(legal[A_OBS:A_OBS + N_CARDS], obs_val, -1e9)
    obs_action = A_OBS + jnp.argmax(obs_val).astype(jnp.int32)

    # --- substep 2: move search --------------------------------------------
    dest = (pos[:, None, :] + dirs[None, :, :]).reshape(-1, 2)    # [36, 2]
    dest_ok = jnp.all((dest >= 0) & (dest < GRID), axis=1)        # [36]
    dest_clip = jnp.clip(dest, 0, GRID - 1)
    cell_idx = dest_clip[:, 0] * GRID + dest_clip[:, 1]           # [36]
    nbr = (jnp.abs(pos[None, :, :] - dest[:, None, :]).sum(-1) == 1).astype(jnp.float32)

    def eval_dest(carry, nb):
        ncol = jnp.einsum('cjf,j->cf', onehot, nb)                # [1680, 3]
        cnt = jnp.take_along_axis(ncol, colorings, axis=1) - nb[None, :]
        b_ok = ((rest_pairs + cnt) >= 2.0).astype(jnp.float32)
        v_conn = w @ (others_ok * b_ok)                           # [9]
        v_num = w @ (others_cnt + b_ok)                           # [9]
        v_pair = w @ (total_pairs[:, None] - deg_same + cnt)      # [9]
        return carry, jnp.stack([v_conn, v_num, v_pair], axis=0)

    _, dest_vals = jax.lax.scan(eval_dest, jnp.asarray(0.0), nbr)  # [36, 3, 9]
    cert_cnt = nbr @ cert_same                                     # [36, 9]
    cert_after = cert_base - cert_deg[None, :] + cert_cnt          # [36, 9]
    density = jnp.sum(nbr, axis=1)[:, None] - nbr                  # [36, 9]
    penalty = (W_RECENT * recent + W_UNKNOWN * (~is_known).astype(jnp.float32))[None, :]
    move_val = (W_PCONN * dest_vals[:, 0, :]
                + W_CERT * cert_after
                + W_ENUM * dest_vals[:, 1, :]
                + W_EPAIR * dest_vals[:, 2, :]
                + W_DENSITY * density
                - penalty
                + 1e-3 * jax.random.uniform(k_move, (36, N_CARDS)))
    move_ids = A_MOVE + jnp.arange(N_CARDS, dtype=jnp.int32)[None, :] * 81 + cell_idx[:, None]
    move_val = jnp.where(legal[move_ids] & dest_ok[:, None], move_val, -1e9)
    move_action = move_ids.reshape(-1)[jnp.argmax(move_val.reshape(-1))].astype(jnp.int32)

    # --- substep 3: hint action --------------------------------------------
    place_ids = (A_PLACE + jnp.arange(N_HINTS, dtype=jnp.int32)[:, None] * 81
                 + pos[None, :, 0] * GRID + pos[None, :, 1])       # [4, 9]
    place_legal = legal[place_ids]
    degree = jnp.sum(adj, axis=1)
    lock_val = (cert_deg + 0.4 * degree + 0.3 * (w @ deg_same))[None, :] + 1e-3 * jax.random.uniform(
        k_hint, (N_HINTS, N_CARDS))
    # Locking a card is only safe if it cannot strand a colour: never lock a
    # second card of a colour that is already locked somewhere non-adjacent.
    locked_known = (nf[:N_CARDS, 8] > 0) & is_known
    clash = jnp.any(locked_known[None, :] & is_known[:, None]
                    & (known[:, None] == known[None, :]) & (adj < 0.5), axis=1)
    safe_lock = (~clash)[None, :]
    sure = place_legal & (p_match >= 0.999) & safe_lock & PLACE_EARLY
    sure_val = jnp.where(sure, lock_val, -1e9)
    s_flat = jnp.argmax(sure_val.reshape(-1))
    sure_action = place_ids.reshape(-1)[s_flat].astype(jnp.int32)
    any_sure = jnp.any(sure)

    fall_val = jnp.where(place_legal, p_match + 0.1 * lock_val, -1e9)
    fall_action = place_ids.reshape(-1)[jnp.argmax(fall_val.reshape(-1))].astype(jnp.int32)
    any_place = jnp.any(place_legal)

    reveal_legal = legal[A_REVEAL:A_REVEAL + N_HINTS]
    reveal_val = jnp.where(reveal_legal, jnp.asarray([0.1, 1.0, 0.9, 0.8]), -1e9)
    reveal_action = A_REVEAL + jnp.argmax(reveal_val).astype(jnp.int32)
    any_reveal = jnp.any(reveal_legal)

    hint_action = jnp.where(any_sure, sure_action,
                            jnp.where(any_reveal, reveal_action,
                                      jnp.where(any_place, fall_action, reveal_action)))

    # --- end-game decision --------------------------------------------------
    two_turn_cost = jnp.where(n_unrevealed == 0, 2.0, jnp.where(n_ready >= 1, 4.0, 6.0))
    cost = jnp.where(turns_done >= 7, LAST_TURN_COST, two_turn_cost)
    score_next = jnp.maximum(score_now - cost, 0.0)
    p_end = P_END_SOFT * p_conn + (1.0 - P_END_SOFT) * p_conn_hard
    p_next = p_end + LEARN_FRAC * (1.0 - p_end)
    floor = jnp.where(turns_done >= 7, 0.0, END_FLOOR)
    value_end = p_end * score_now
    end_now = (value_end >= p_next * score_next) & (value_end >= floor) & legal[A_END]

    substep0 = jnp.where(end_now, jnp.asarray(A_END, dtype=jnp.int32), obs_action)
    action = jnp.where(substep == 0, substep0,
                       jnp.where(substep == 1, obs_action,
                                 jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32))
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    action = action.astype(jnp.int32)

    next_memory = {
        'known': known,
        'seen_by_partner': partner_seen,
        'recent': recent,
        'prev_pos': pos,
        'affinity': affinity,
        'step': step_prev + 1,
    }
    trace = {
        'p_conn': p_conn.astype(jnp.float32),
        'p_hard': p_conn_hard.astype(jnp.float32),
        'trust': trust.astype(jnp.float32),
        'score_now': score_now.astype(jnp.float32),
        'n_known': jnp.sum(is_known.astype(jnp.int32)),
        'cert_pairs': cert_base.astype(jnp.float32),
        'turns_done': turns_done.astype(jnp.int32),
        'action': action,
    }
    return action, next_memory, trace
