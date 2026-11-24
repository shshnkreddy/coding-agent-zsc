"""Two-player 9-card Yokai policy.

Every decision is a function of the public board plus this seat's own private
colour observations.  No private signalling convention is used, so the same
code should behave sensibly next to an arbitrary competent partner.

Belief representation
---------------------
A full colouring of the 9 cards is equivalent to (partition of the cards into
three unordered triples, bijection triples -> colours).  There are 280
partitions and each contributes 1, 2 or 6 consistent colourings depending on
how many of its triples already contain an observed colour.  "All colours
connected" is a property of the partition alone: a triple is connected exactly
when at least two of its three card pairs are orthogonally adjacent.  So the
exact uniform posterior over colourings collapses to 280 weighted partitions,
which is cheap enough to evaluate for every candidate move of every card.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'yokai_belief_greedy',
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

NUM_CARDS = 9
GRID = 9
NUM_CELLS = 81
NUM_HINTS = 4
NUM_PARTITIONS = 280

# Partial credit for a triple that is one move away from being connected.
COMPACT = 0.001  # tie-break: keep the card blob tight

# A colour triple with two adjacent pairs is done, one adjacent pair is one
# move from done, and none is two moves from done.  Boards are therefore scored
# by DISCOUNT ** (moves still needed), where the discount depends on how many
# moves are left in the episode: with the very last move only a genuinely
# connected board counts, so the score collapses onto the posterior
# probability of success.  Indexed by completed turns.
DISCOUNT = [0.60, 0.60, 0.58, 0.56, 0.52, 0.46, 0.34, 0.0]

# How much to trust that a hint the partner placed really does match the card
# underneath it.  A correct placement scores +1 and a wrong one -1, so any
# score-seeking partner places hints it believes are right; this is a rational
# inference about a competent partner, not an agreed signalling code, and the
# evidence stays soft so a careless partner cannot force a false certainty.
HINT_TRUST = 0.93

# How much to trust that a card the partner just relocated was placed next to
# a card of its own colour.  Joining a card to its own colour group is the only
# way a move can help, so a competent partner's moves carry this information
# whatever strategy produced them; a partner that moves blindly only adds mild
# noise, which is why the evidence is soft and its total weight is capped.
# Relative likelihood of seeing the partner put the moved card next to 0, 1 or
# 2 cards of its own colour.  Early moves are largely blind, so the likelihood
# ratio is flattened for them and sharpened once both seats know enough to move
# deliberately; indexed by completed turns at the time of the move.
MOVE_LIKELIHOOD = [[0.30, 0.65, 0.85]] * 8
# A partner that relocates a card usually does so because that card's colour
# group was broken; relative likelihood of the move given the group was
# already connected / still broken beforehand.
PRE_MOVE_LIKELIHOOD = [0.35, 0.65]
LOGW_CAP = 3.0

# Reward obtained by ending the game after ``t`` completed turns, assuming the
# cheapest hint clock (reveal, place-correctly, reveal, place-correctly, ...):
# 5*unrevealed + 2*revealed + correct - incorrect.
END_SCORE = [20.0, 17.0, 16.0, 13.0, 12.0, 9.0, 8.0, 5.0, 4.0]
# Value of playing on instead of ending now: a floor, plus a multiplier applied
# to the best connectivity probability reachable with this turn's own move.
CONT_BASE = [4.0, 4.0, 3.8, 3.6, 2.8, 2.8, 2.9, 2.9]
CONT_MULT = [9.4, 8.8, 7.2, 6.6, 5.0, 4.4, 2.8, 4.0]


def _partition_tables():
    """Static tables describing the 280 partitions of 9 cards into triples."""
    groups = []
    for a in range(1, NUM_CARDS):
        for b in range(a + 1, NUM_CARDS):
            g0 = [0, a, b]
            rest0 = [c for c in range(NUM_CARDS) if c not in g0]
            for i in range(1, len(rest0)):
                for j in range(i + 1, len(rest0)):
                    g1 = [rest0[0], rest0[i], rest0[j]]
                    g2 = [c for c in rest0 if c not in g1]
                    groups.append([g0, g1, g2])
    gidx = []
    other = []
    ind = []
    for card in range(NUM_CARDS):
        gidx_c = []
        other_c = []
        ind_c = []
        for part in groups:
            for g in range(3):
                if card in part[g]:
                    rest = [m for m in part[g] if m != card]
                    gidx_c.append(g)
                    other_c.append(rest)
                    ind_c.append([1.0 if k in rest else 0.0 for k in range(NUM_CARDS)])
        gidx.append(gidx_c)
        other.append(other_c)
        ind.append(ind_c)
    # colour of each card under each of the 6 group -> colour labellings
    perms = [[a, b, c] for a in range(3) for b in range(3) for c in range(3)
             if a != b and a != c and b != c]
    colour = [[[perm[g] for g in gidx[card]] for card in range(NUM_CARDS)]
              for perm in perms]
    return (jnp.asarray(groups, dtype=jnp.int32),
            jnp.asarray(gidx, dtype=jnp.int32),
            jnp.asarray(other, dtype=jnp.int32),
            jnp.asarray(ind, dtype=jnp.float32),
            jnp.asarray(colour, dtype=jnp.int32))


def init_agent():
    return (jnp.zeros((NUM_CARDS, 3), dtype=jnp.int8),      # colours seen here
            jnp.zeros((NUM_PARTITIONS,), dtype=jnp.float32),  # log evidence
            jnp.zeros((NUM_CARDS, 2), dtype=jnp.int32),     # previous layout
            jnp.zeros((), dtype=jnp.int32))                 # was this seat active


def _triple_value(edges, discount):
    """Soft value of a colour triple given its number of adjacent card pairs."""
    return jnp.where(edges >= 2.0, 1.0,
                     jnp.where(edges >= 1.0, discount, discount * discount))


def _reachability(adj, eye):
    """Boolean transitive closure of a 9-node graph."""
    reach = adj + eye
    for _ in range(4):
        reach = jnp.minimum(reach @ reach, 1.0)
    return reach > 0.0


def agent_act(obs, reset, memory, rng):
    groups, gidx_all, other_all, ind_all, colour_all = _partition_tables()
    cells = jnp.arange(NUM_CELLS, dtype=jnp.int32)
    cell_x = cells // GRID
    cell_y = cells % GRID
    eye9 = jnp.eye(NUM_CARDS, dtype=jnp.float32)
    end_score = jnp.asarray(END_SCORE, dtype=jnp.float32)
    cont_base = jnp.asarray(CONT_BASE, dtype=jnp.float32)
    cont_mult = jnp.asarray(CONT_MULT, dtype=jnp.float32)
    discount_tab = jnp.asarray(DISCOUNT, dtype=jnp.float32)
    move_like_tab = jnp.asarray(MOVE_LIKELIHOOD, dtype=jnp.float32)
    pre_move_tab = jnp.asarray(PRE_MOVE_LIKELIHOOD, dtype=jnp.float32)
    del end_score  # documented schedule; the decision uses the live hint state

    feats = obs.node_features.astype(jnp.int32)
    legal = obs.legal_actions
    card_feats = feats[:NUM_CARDS]
    hint_feats = feats[NUM_CARDS:]

    # ---------------- own colour knowledge (observation + private memory) ---
    seen_mem, logw_mem, prev_pos_mem, prev_active_mem = memory
    prior = jnp.where(reset, jnp.zeros_like(seen_mem), seen_mem)
    seen = jnp.maximum(prior, jnp.clip(card_feats[:, 0:3], 0, 1).astype(jnp.int8))
    known_oh = seen.astype(jnp.float32)                       # [9, 3]
    known = jnp.clip(known_oh.sum(-1), 0.0, 1.0)              # [9]

    pos = obs.positions.astype(jnp.int32)                     # [9, 2]
    locked = (card_feats[:, 8] > 0).astype(jnp.float32)
    partner_seen = (card_feats[:, 9] > 0).astype(jnp.float32)
    substep = feats[0, 11]
    active = feats[0, 10]

    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    adj = (dist == 1).astype(jnp.float32)                     # [9, 9]

    # ---------------- hint state -------------------------------------------
    hint_set = jnp.clip(hint_feats[:, 3:6], 0, 1).astype(jnp.float32)  # [4, 3]
    set_size = hint_set.sum(-1)
    revealed = (set_size > 0.0).astype(jnp.float32)
    placed = (hint_feats[:, 8] > 0).astype(jnp.float32)
    n_revealed = revealed.sum()
    n_placed = placed.sum()
    turns_done = jnp.clip((n_revealed + n_placed).astype(jnp.int32), 0, 7)
    hint_card = obs.adjacency_matrix[NUM_CARDS:, :NUM_CARDS].astype(jnp.float32)

    # Soft evidence: a placed hint suggests its card's colour is in its set.
    # Only applied to cards whose colour this seat has not observed itself.
    match_factor = HINT_TRUST * hint_set + (1.0 - HINT_TRUST) * (1.0 - hint_set)
    on_card = placed[:, None] * hint_card                              # [4, 9]
    evidence = 1.0 + on_card.T @ (match_factor - 1.0)                  # [9, 3]
    evidence = jnp.where(known[:, None] > 0.0, 1.0, evidence)

    # ---------------- evidence from the partner's last move ----------------
    prev_pos = jnp.where(reset, pos, prev_pos_mem)
    prev_active = jnp.where(reset, 0, prev_active_mem)
    changed = jnp.any(pos != prev_pos, axis=-1)
    partner_moved = ((changed.sum() == 1) & (prev_active == 0)
                     & jnp.logical_not(reset))
    mover = jnp.argmax(changed)
    neighbours = (jnp.take(ind_all, mover, axis=0) @ adj[mover]).astype(jnp.int32)
    prev_dist = jnp.abs(prev_pos[:, None, :] - prev_pos[None, :, :]).sum(-1)
    prev_adj = (prev_dist == 1).astype(jnp.float32)
    mate = jnp.take(other_all, mover, axis=0)                          # [280, 2]
    was_joined = (prev_adj[mate[:, 0], mate[:, 1]]
                  + prev_adj[mover, mate[:, 0]]
                  + prev_adj[mover, mate[:, 1]]) >= 2.0
    step = (jnp.log(move_like_tab[turns_done][neighbours])
            + jnp.log(jnp.where(was_joined, pre_move_tab[0], pre_move_tab[1])))
    logw = jnp.where(reset, jnp.zeros_like(logw_mem), logw_mem)
    logw = logw + jnp.where(partner_moved, step, 0.0)
    logw = jnp.clip(logw - logw.max(), -LOGW_CAP, 0.0)

    # ---------------- partition posterior ----------------------------------
    g0, g1, g2 = groups[:, :, 0], groups[:, :, 1], groups[:, :, 2]
    edges = adj[g0, g1] + adj[g0, g2] + adj[g1, g2]           # [280, 3]

    # colour_all[s, j, p] is the colour of card j under labelling s of partition p
    card_ix = jnp.arange(NUM_CARDS)[None, :, None]
    agrees = known_oh[card_ix, colour_all]                    # [6, 9, 280]
    fits = 1.0 - known[None, :, None] + known[None, :, None] * agrees
    like = evidence[card_ix, colour_all]
    label_weight = jnp.prod(fits * like, axis=1) * jnp.exp(logw)[None, :]  # [6,280]
    weight = label_weight.sum(axis=0)                         # [280]
    wsum = jnp.maximum(weight.sum(), 1e-6)

    discount = discount_tab[turns_done]
    triple_val = _triple_value(edges, discount)
    val_now = jnp.prod(triple_val, axis=-1)                   # [280]
    connected_now = jnp.all(edges >= 2.0, axis=-1).astype(jnp.float32)
    p_now = (weight * connected_now).sum() / wsum
    obj_now = (weight * val_now).sum() / wsum

    # ---------------- move search ------------------------------------------
    occupied = jnp.zeros((NUM_CELLS,), dtype=jnp.float32).at[
        pos[:, 0] * GRID + pos[:, 1]].add(1.0)

    def per_card(carry, card):
        onehot = jnp.zeros((NUM_CARDS,), dtype=jnp.float32).at[card].set(1.0)
        keep = 1.0 - onehot
        adj_rest = adj * keep[:, None] * keep[None, :]
        reach = _reachability(adj_rest, eye9).astype(jnp.float32)      # [9, 9]

        cell_dist = (jnp.abs(cell_x[:, None] - pos[None, :, 0])
                     + jnp.abs(cell_y[:, None] - pos[None, :, 1]))
        touch = (cell_dist == 1).astype(jnp.float32) * keep[None, :]   # [81, 9]

        # Legality: card unlocked, target cell empty and different, and the
        # target must touch every component of the other eight cards.
        own_cell = pos[card, 0] * GRID + pos[card, 1]
        free = occupied - jnp.zeros((NUM_CELLS,), jnp.float32).at[own_cell].set(1.0)
        hit = touch @ reach.T                                          # [81, 9]
        joins = jnp.all((hit > 0.0) | (onehot[None, :] > 0.0), axis=-1)
        ok = (joins & (free <= 0.0) & (cells != own_cell)
              & (locked[card] <= 0.0))

        gid = gidx_all[card]
        mask_g = jnp.arange(3)[None, :] == gid[:, None]                # [280, 3]
        rest_ok = jnp.all(mask_g | (edges >= 2.0), axis=-1).astype(jnp.float32)
        rest_val = jnp.prod(jnp.where(mask_g, 1.0, triple_val), axis=-1)
        pair_uv = adj[other_all[card][:, 0], other_all[card][:, 1]]    # [280]

        count = pair_uv[None, :] + touch @ ind_all[card].T             # [81, 280]
        conn = rest_ok[None, :] * (count >= 2.0).astype(jnp.float32)
        soft = rest_val[None, :] * _triple_value(count, discount)
        p_cell = (conn * weight[None, :]).sum(-1) / wsum
        v_cell = (soft * weight[None, :]).sum(-1) / wsum
        score = (v_cell + COMPACT * touch.sum(-1) + jnp.where(ok, 0.0, -1e9))
        return carry, (score, jnp.where(ok, p_cell, -1.0))

    _, (move_score, move_p) = jax.lax.scan(per_card, None, jnp.arange(NUM_CARDS))
    move_score = move_score.reshape(-1)
    move_p = move_p.reshape(-1)
    best_move = jnp.argmax(move_score + jnp.where(legal[10:739], 0.0, -1e9))
    move_action = (10 + best_move).astype(jnp.int32)
    p_best = jnp.maximum(jnp.max(jnp.where(move_score > -1e8, move_p, -1.0)), p_now)

    # ---------------- hint bookkeeping -------------------------------------
    remaining = 3.0 - known_oh.sum(0)                                  # per colour
    n_unknown = jnp.maximum(NUM_CARDS - known.sum(), 1.0)
    marginal = jnp.clip(remaining / n_unknown, 0.0, 1.0)               # [3]
    match_known = known_oh @ hint_set.T                                # [9, 4]
    match_guess = (hint_set @ marginal)[None, :]                       # [1, 4]
    match_p = jnp.where(known[:, None] > 0.0, match_known, match_guess)

    placed_p = (hint_card * match_p.T).sum(-1)                         # [4]
    placed_value = (placed * (2.0 * placed_p - 1.0)).sum()
    score_now = (5.0 * (NUM_HINTS - n_revealed) + 2.0 * (n_revealed - n_placed)
                 + placed_value)

    # ---------------- end-game decision ------------------------------------
    value_end = p_now * score_now
    value_cont = jnp.maximum(cont_base[turns_done], cont_mult[turns_done] * p_best)
    want_end = (value_end > value_cont) & legal[1067]

    # ---------------- observation choice -----------------------------------
    # Prefer the unseen card whose colour would swing the board's value most:
    # a card that every consistent colouring treats the same way teaches
    # nothing useful, however uncertain its colour is.
    is_colour = (colour_all[..., None] == jnp.arange(3)).astype(jnp.float32)
    mass = jnp.einsum('sp,sjpc->jc', label_weight, is_colour)
    lam = turns_done.astype(jnp.float32) / 7.0
    target = (1.0 - lam) * val_now + lam * connected_now
    focus = (weight * target).sum() / wsum
    vmass = jnp.einsum('sp,sjpc->jc', label_weight * target[None, :], is_colour)
    swing = jnp.abs(vmass - mass * focus).sum(-1) / wsum        # [9]
    obs_pref = (4.0 * (1.0 - known) + 2.0 * swing
                + 0.3 * (1.0 - partner_seen))
    obs_action = (1 + jnp.argmax(jnp.where(legal[1:10], obs_pref, -1e9))).astype(jnp.int32)

    # ---------------- hint action ------------------------------------------
    # Placing a hint locks the card for good, so prefer cards that the belief
    # says are already sitting inside a finished colour group, and within those
    # the card with the most same-colour neighbours (the one a future move is
    # least likely to want to pick up).
    card_edges = edges[jnp.arange(NUM_PARTITIONS)[None, :], gidx_all]  # [9, 280]
    settled = (weight[None, :] * (card_edges >= 2.0)).sum(-1) / wsum
    mates = jnp.einsum('jpk,jk->jp', ind_all, adj)
    middling = (weight[None, :] * mates).sum(-1) / wsum
    cell_of_card = pos[:, 0] * GRID + pos[:, 1]
    place_legal = legal[743:1067].reshape(NUM_HINTS, NUM_CELLS)
    place_ok = place_legal[:, cell_of_card].astype(jnp.float32)        # [4, 9]
    place_value = (match_p.T
                   + 0.05 * (3.0 - set_size)[:, None]
                   + 0.20 * settled[None, :]
                   + 0.05 * middling[None, :])
    place_value = jnp.where(place_ok > 0.0, place_value, -1e9)
    best_place = jnp.argmax(place_value.reshape(-1))
    place_action = (743 + (best_place // NUM_CARDS) * NUM_CELLS
                    + cell_of_card[best_place % NUM_CARDS]).astype(jnp.int32)
    certain_place = jnp.max(jnp.where(place_ok > 0.0, match_p.T, -1.0)) >= 0.999
    any_place = jnp.any(place_ok > 0.0)

    reveal_pref = (jnp.arange(NUM_HINTS) != 0).astype(jnp.float32)
    reveal_action = (739 + jnp.argmax(
        jnp.where(legal[739:743], reveal_pref, -1e9))).astype(jnp.int32)
    any_reveal = jnp.any(legal[739:743])

    do_place = any_place & (certain_place | jnp.logical_not(any_reveal))
    hint_action = jnp.where(do_place, place_action, reveal_action)

    # ---------------- assemble ---------------------------------------------
    action = jnp.where(
        substep == 0,
        jnp.where(want_end, jnp.asarray(1067, jnp.int32), obs_action),
        jnp.where(substep == 1, obs_action,
                  jnp.where(substep == 2, move_action, hint_action)))
    action = jnp.where(active > 0, action, jnp.asarray(0, jnp.int32))
    action = jnp.where(legal[action], action,
                       jnp.argmax(legal).astype(jnp.int32)).astype(jnp.int32)

    trace = {
        'p_now': p_now.astype(jnp.float32),
        'p_best': p_best.astype(jnp.float32),
        'score_now': score_now.astype(jnp.float32),
        'value_end': value_end.astype(jnp.float32),
        'value_cont': value_cont.astype(jnp.float32),
        'turns_done': turns_done.astype(jnp.int32),
        'n_known': known.sum().astype(jnp.float32),
        'adj_total': (adj.sum() * 0.5).astype(jnp.float32),
        'exp_broken': ((weight * (edges < 2.0).sum(-1)).sum() / wsum).astype(jnp.float32),
        'move_known': known[best_move // NUM_CELLS].astype(jnp.float32),
        'n_locked': locked.sum().astype(jnp.float32),
        'ended': (want_end & (substep == 0) & (active > 0)).astype(jnp.int32),
    }
    return action, (seen, logw, pos, active.astype(jnp.int32)), trace
