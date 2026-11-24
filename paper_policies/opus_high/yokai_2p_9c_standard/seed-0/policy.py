"""Two-player 9-card Yokai policy.

Design notes
------------
Terminal reward is ``5*U + 2*R + C - I`` and is zero unless all three colours
are orthogonally connected, so two things matter: (1) actually clustering the
colours and (2) spending as few hint points as possible before the episode
ends.  Substep 3 forces a reveal or a place every turn.  Revealing costs 3
points (5 -> 2); placing an already revealed hint correctly costs only 1
(2 -> 1).  The cheapest schedule therefore alternates reveal / place, and any
seat can follow it from public state alone: place whenever a revealed unplaced
hint exists and some card is known to match it, otherwise reveal.  Placement
also locks the card, which protects a finished cluster from being disturbed.

Colour knowledge is private.  Each seat stores its own observations in memory
and maintains an exact posterior over the 1680 colour assignments (three cards
of each colour over nine stable ids).  Publicly placed hints are used only as
*soft* evidence: correct placement is exactly what the scoring rule rewards,
so a competent partner places truthfully, but a low trust weight stops a
careless partner from poisoning the belief.

Every decision is a function of the public board plus this seat's own
observations.  Nothing depends on the partner running this same code, and no
private signalling convention is used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'yokai_belief_cluster',
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
N_CELLS = 81
N_HINTS = 4
N_COL = 1680
N_SUBSET = 84
N_PART = 280
N_PAIR = 36

# Action layout.
A_OBSERVE = 1
A_MOVE = 10
A_REVEAL = 739
A_PLACE = 743
A_END = 1067

# Tunable behaviour constants.
TOP_K = 48              # legal moves given an exact posterior evaluation
HINT_TRUST = 0.20       # weight kept for colourings contradicting a placed hint
PLACE_THRESHOLD = 0.55  # minimum P(correct) before placing instead of revealing
# Placing locks a card for good, so prefer targets already surrounded by their
# own colour, which are the least likely to have to move again.
SETTLE = 0.50
DONE_THRESHOLD = 0.9    # only lock a card whose colour group is already joined
RHO = 0.60              # discount on the optimistic continuation value
# How hard an observably unhelpful partner discounts the value of playing on.
PARTNER_SENS = 0.40
Q_MIN = 0.12
# How strongly an unhelpful partner shifts the move objective from the long
# plan towards immediate posterior certainty.
P_SHIFT = 1.0
# Bias of the observation choice towards well-connected cards (0 keeps the
# plain low-id order).
OBS_DEG = -0.05
# Shaping weight on the expected number of monochrome adjacent pairs.  That
# count is 6 exactly when every colour is joined, and each missing pair costs
# roughly one further move (two points), so it is a direct "moves to go" signal.
BETA = 0.20
# Compactness weight: keeping the nine cards in a dense blob keeps cards out of
# cut-vertex positions, which is what makes the final joining move legal.
ALPHA = 0.15
# Split of that term between raw board edges and the exact count of cards that
# could still be legally relocated.
EDGE_W = 1.0
GAMMA = 0.0
# Weight on outright certainty.  The game can only be ended early when *every*
# still-possible colouring is connected, so an arrangement that is robust
# across the posterior support is worth more than the linear expectation.
KAPPA = 0.0
# Weight on letting the *partner* finish.  Which cards the partner has observed
# is public, and this seat usually knows those colours, so the partner's own
# posterior can be reconstructed.  An arrangement that is connected for every
# colouring still open to the partner lets them end the game on their next
# substep 0, which is worth a full extra turn of hint value.
MU = 1.0
# Only late, when this seat is ahead of the partner in observations and there
# are no further turns to spend, is it worth trading progress for the partner's
# certainty.
MU_FROM = 5

# Best terminal score reachable after ``t`` completed turns under the cheapest
# (alternating reveal / place) hint schedule, and a pessimistic floor on the
# value of playing on instead of ending now.
SCORE_AT_TURN = [20.0, 17.0, 16.0, 13.0, 12.0, 9.0, 8.0, 5.0, 4.0]
CONTINUE_FLOOR = [4.0, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.0, 0.0]


def _pair_id(i, j):
    """Index of the unordered pair (i, j), i != j, in row-major i < j order."""
    a = jnp.minimum(i, j)
    b = jnp.maximum(i, j)
    return a * N_CARDS - (a * (a + 1)) // 2 + (b - a - 1)


def _tables():
    """Static combinatorial tables, built from constants so XLA folds them."""
    pow3 = jnp.asarray([1, 3, 9, 27, 81, 243, 729, 2187, 6561], dtype=jnp.int32)
    pow2 = jnp.asarray([1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=jnp.int32)

    # All 1680 colour assignments with three cards of each colour.
    grid = jnp.arange(3 ** N_CARDS, dtype=jnp.int32)
    digits = (grid[:, None] // pow3[None, :]) % 3
    counts = jnp.stack([(digits == k).sum(-1) for k in range(N_COLORS)], axis=-1)
    valid = jnp.all(counts == 3, axis=-1)
    order = jnp.argsort(jnp.where(valid, 0, 1))
    col = digits[order[:N_COL]]                                   # [1680, 9]
    col_onehot = (
        col[:, :, None] == jnp.arange(N_COLORS, dtype=jnp.int32)[None, None, :]
    ).astype(jnp.float32).reshape(N_COL, N_CARDS * N_COLORS)

    # All 84 three-card subsets, as sorted member index triples.
    masks = jnp.arange(2 ** N_CARDS, dtype=jnp.int32)
    bits = (masks[:, None] // pow2[None, :]) % 2
    order2 = jnp.argsort(jnp.where(bits.sum(-1) == 3, 0, 1))
    sub_bits = bits[order2[:N_SUBSET]]                            # [84, 9]
    members = jnp.argsort(1 - sub_bits, axis=-1)[:, :3]           # [84, 3]
    subset_pairs = jnp.stack(
        [
            _pair_id(members[:, 0], members[:, 1]),
            _pair_id(members[:, 0], members[:, 2]),
            _pair_id(members[:, 1], members[:, 2]),
        ],
        axis=-1,
    )                                                             # [84, 3]

    # Endpoints of each of the 36 card pairs.
    idx_i = jnp.arange(N_CARDS, dtype=jnp.int32)[:, None] * jnp.ones(
        (1, N_CARDS), dtype=jnp.int32
    )
    idx_j = idx_i.T
    upper = idx_i < idx_j
    slot = jnp.where(upper, _pair_id(idx_i, idx_j), N_PAIR).reshape(-1)
    pair_a = jnp.zeros((N_PAIR + 1,), jnp.int32).at[slot].set(idx_i.reshape(-1))[:N_PAIR]
    pair_b = jnp.zeros((N_PAIR + 1,), jnp.int32).at[slot].set(idx_j.reshape(-1))[:N_PAIR]

    # Map a colouring to its unordered partition into three colour groups.
    subset_key = (sub_bits * pow2[None, :]).sum(-1)
    lut = jnp.zeros((2 ** N_CARDS,), jnp.int32).at[subset_key].set(
        jnp.arange(N_SUBSET, dtype=jnp.int32)
    )
    group_ids = jnp.stack(
        [lut[((col == k).astype(jnp.int32) * pow2[None, :]).sum(-1)] for k in range(N_COLORS)],
        axis=-1,
    )                                                             # [1680, 3]
    group_ids = jnp.sort(group_ids, axis=-1)
    keys = (
        group_ids[:, 0] * (N_SUBSET * N_SUBSET)
        + group_ids[:, 1] * N_SUBSET
        + group_ids[:, 2]
    )
    korder = jnp.argsort(keys)
    sorted_keys = keys[korder]
    fresh = jnp.concatenate(
        [
            jnp.ones((1,), jnp.int32),
            (sorted_keys[1:] != sorted_keys[:-1]).astype(jnp.int32),
        ]
    )
    rank = jnp.cumsum(fresh) - 1                                  # [1680] in 0..279
    col2part = jnp.zeros((N_COL,), jnp.int32).at[korder].set(rank)
    part_key = jnp.zeros((N_PART,), jnp.int32).at[rank].set(sorted_keys)
    part_s0 = part_key // (N_SUBSET * N_SUBSET)
    part_s1 = (part_key // N_SUBSET) % N_SUBSET
    part_s2 = part_key % N_SUBSET

    cells = jnp.arange(N_CELLS, dtype=jnp.int32)
    cellpos = jnp.stack([cells // GRID, cells % GRID], axis=-1)   # [81, 2]

    # Per colouring, whether the two cards of each pair share a colour.
    same_pair = (col[:, pair_a] == col[:, pair_b]).astype(jnp.float32)  # [1680, 36]

    # For every partition and card, the subset holding that card.
    in0 = sub_bits[part_s0] > 0                                   # [280, 9]
    in1 = sub_bits[part_s1] > 0
    part_card = jnp.where(
        in0, part_s0[:, None], jnp.where(in1, part_s1[:, None], part_s2[:, None])
    )                                                             # [280, 9]

    return {
        'part_card': part_card,
        'col': col,
        'col_onehot': col_onehot,
        'same_pair': same_pair,
        'subset_pairs': subset_pairs,
        'pair_a': pair_a,
        'pair_b': pair_b,
        'part_s0': part_s0,
        'part_s1': part_s1,
        'part_s2': part_s2,
        'col2part': col2part,
        'cellpos': cellpos,
    }


def init_agent():
    return {
        'known': -jnp.ones((N_CARDS,), dtype=jnp.int32),
        'seen': jnp.zeros((N_CARDS,), dtype=jnp.int32),
        # Partner-competence tracking: the best expected monochrome-pair count
        # reachable in one move when this seat handed the board over, and the
        # running shortfall the partner actually produced.
        'ref': jnp.asarray(-1.0, dtype=jnp.float32),
        'gap_sum': jnp.asarray(0.0, dtype=jnp.float32),
        'gap_n': jnp.asarray(0.0, dtype=jnp.float32),
    }


def _n_movable(pos, locked_f):
    """How many cards could still be legally relocated.

    A card may only move if the other eight stay orthogonally connected, so a
    cut vertex of the nine-card board is frozen in place even when relocating
    it is exactly what joining the last colour needs.
    """
    adj = (
        jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1) == 1
    ).astype(jnp.float32)
    off = 1.0 - jnp.eye(N_CARDS, dtype=jnp.float32)          # off[j, i] = i != j
    masked = adj[None, :, :] * off[:, :, None] * off[:, None, :]   # [9, 9, 9]
    # Flood fill from one surviving card per removed card j.
    seed = jnp.where(jnp.arange(N_CARDS) == 0, 1, 0)
    reach = (jnp.arange(N_CARDS)[None, :] == seed[:, None]).astype(jnp.float32)
    for _ in range(N_CARDS - 1):
        reach = jnp.clip(reach + jnp.einsum('jk,jkl->jl', reach, masked), 0.0, 1.0)
    intact = jnp.all(reach >= off, axis=-1).astype(jnp.float32)
    return (intact * (1.0 - locked_f)).sum()


def _board_stats(pos, tb):
    """Monochrome pair count per partition (0..6) and total board edges (8..12).

    Three cells of a grid span at most two edges (the grid has no triangles),
    so a colour group is joined exactly when it spans two, the per-partition
    total is at most six, and ``6 - total`` is the number of cards that still
    have to be relocated.  The board edge count measures compactness: a sparse,
    stringy board has many cut vertices, and relocating a cut vertex is illegal
    because the nine cards must stay orthogonally connected.
    """
    d = jnp.abs(pos[tb['pair_a']] - pos[tb['pair_b']]).sum(-1)
    edge = (d == 1).astype(jnp.float32)                 # [36]
    deg = edge[tb['subset_pairs']].sum(-1)              # [84]
    mono = deg[tb['part_s0']] + deg[tb['part_s1']] + deg[tb['part_s2']]
    return mono, edge.sum(), deg


def agent_act(obs, reset, memory, rng):
    tb = _tables()
    col = tb['col']
    nf = obs.node_features.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions

    # ---- decode observation --------------------------------------------
    card_color_onehot = nf[0:N_CARDS, 0:N_COLORS]
    visible_now = card_color_onehot.sum(-1) > 0
    visible_color = jnp.argmax(card_color_onehot, axis=-1)

    locked = nf[0:N_CARDS, 8] > 0
    locked_f = locked.astype(jnp.float32)
    card_cell = pos[:, 0] * GRID + pos[:, 1]                        # [9]

    hint_set = (nf[N_CARDS:, 3:6] > 0).astype(jnp.float32)          # [4, 3]
    revealed = hint_set.sum(-1) > 0
    placed = nf[N_CARDS:, 8] > 0
    hint_card = jnp.argmax(
        obs.adjacency_matrix[N_CARDS:, 0:N_CARDS].astype(jnp.int32), axis=-1
    )                                                               # [4]

    active = nf[0, 10] > 0
    substep = nf[0, 11]

    # ---- private memory of observed colours -----------------------------
    prev_known = jnp.where(reset, -jnp.ones((N_CARDS,), jnp.int32), memory['known'])
    known = jnp.where(visible_now, visible_color, prev_known)
    # Which cards the partner has looked at is public, but only flagged during
    # the turn it happened, so accumulate it privately.
    prev_seen = jnp.where(reset, jnp.zeros((N_CARDS,), jnp.int32), memory['seen'])
    seen = jnp.maximum(prev_seen, (nf[0:N_CARDS, 9] > 0).astype(jnp.int32))

    # ---- posterior over colour assignments ------------------------------
    consistent = jnp.all((col == known[None, :]) | (known[None, :] < 0), axis=-1)
    base_w = consistent.astype(jnp.float32)
    # The partner's reconstructed posterior: the colours of the cards it has
    # looked at, as far as this seat knows them.  Cards it saw but this seat has
    # not are treated as unknown to it, which only makes the estimate cautious.
    partner_known = jnp.where(seen > 0, known, -1)
    partner_consistent = jnp.all(
        (col == partner_known[None, :]) | (partner_known[None, :] < 0), axis=-1
    )
    hint_factor = jnp.ones((N_COL,), jnp.float32)
    for h in range(N_HINTS):
        col_at_target = jnp.take(col, hint_card[h], axis=1)          # [1680]
        in_set = jnp.take(hint_set[h], col_at_target)
        factor = jnp.where(in_set > 0.5, 1.0, HINT_TRUST)
        hint_factor = hint_factor * jnp.where(placed[h], factor, 1.0)
    w = base_w * hint_factor
    w = jnp.where(w.sum() > 1e-6, w, base_w)
    wtot = jnp.maximum(w.sum(), 1e-9)
    pw = partner_consistent.astype(jnp.float32) * hint_factor
    pw = jnp.where(pw.sum() > 1e-6, pw, partner_consistent.astype(jnp.float32))
    pwtot = jnp.maximum(pw.sum(), 1e-9)

    wpart = jnp.zeros((N_PART,), jnp.float32).at[tb['col2part']].add(w)
    pwpart = jnp.zeros((N_PART,), jnp.float32).at[tb['col2part']].add(pw)
    q = (w[None, :] @ tb['col_onehot']).reshape(N_CARDS, N_COLORS) / wtot

    n_revealed = revealed.sum()
    n_placed = placed.sum()
    turns = jnp.clip(n_revealed + n_placed, 0, 8)
    # Terminal score for a board that becomes connected at the end of turn k,
    # under the cheapest alternating reveal / place schedule; play cannot run
    # past turn 8, so later completions are worthless.
    value_tab = jnp.concatenate(
        [
            jnp.asarray(SCORE_AT_TURN, dtype=jnp.float32),
            jnp.zeros((8,), jnp.float32),
        ]
    )

    cur_mono, _, cur_deg = _board_stats(pos, tb)
    p_now = (wpart * (cur_mono >= 5.5)).sum() / wtot
    # Probability that each card's own colour group is already joined; such a
    # card is finished and can be locked by a hint without losing flexibility.
    card_done = (
        wpart[None, :] @ (cur_deg[tb['part_card']] >= 1.5).astype(jnp.float32)
    ).reshape(N_CARDS) / wtot

    # ---- move search -----------------------------------------------------
    # Exact posterior probability that each card pair shares a colour.
    same = (w[None, :] @ tb['same_pair']).reshape(N_PAIR) / wtot      # [36]
    sp = jnp.zeros((N_CARDS, N_CARDS), jnp.float32)
    sp = sp.at[tb['pair_a'], tb['pair_b']].set(same)
    sp = sp.at[tb['pair_b'], tb['pair_a']].set(same)                 # [9, 9], zero diagonal

    cur_adj = (
        jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1) == 1
    ).astype(jnp.float32)
    weighted = cur_adj * sp
    base_excl = 0.5 * weighted.sum() - weighted.sum(-1)              # [9]
    cell_adj = (
        jnp.abs(tb['cellpos'][:, None, :] - pos[None, :, :]).sum(-1) == 1
    ).astype(jnp.float32)                                            # [81, 9]
    # Expected monochrome adjacent pairs after moving card i to each cell.
    mono = base_excl[:, None] + sp @ cell_adj.T                      # [9, 81]

    # ---- partner competence ----------------------------------------------
    # Between this seat's substep 3 and its next substep 0 nothing changes its
    # own knowledge, so the two boards are directly comparable.  The shortfall
    # between the best single move that had been available to the partner and
    # what the board is actually worth now measures how much the partner helps.
    # A partner that keeps undoing progress makes playing on worth far less
    # than the self-play schedule assumes, which argues for finishing sooner.
    mono_now = 0.5 * weighted.sum()
    prev_ref = jnp.where(reset, -1.0, memory['ref'])
    prev_gap_sum = jnp.where(reset, 0.0, memory['gap_sum'])
    prev_gap_n = jnp.where(reset, 0.0, memory['gap_n'])
    scoring = active & (substep == 0) & (prev_ref >= 0.0)
    gap = jnp.clip(prev_ref - mono_now, 0.0, 6.0)
    gap_sum = jnp.where(scoring, prev_gap_sum + gap, prev_gap_sum)
    gap_n = jnp.where(scoring, prev_gap_n + 1.0, prev_gap_n)
    quality = jnp.where(
        gap_n >= 1.0,
        jnp.clip(
            1.0 - PARTNER_SENS * gap_sum / jnp.maximum(gap_n, 1.0), Q_MIN, 1.0
        ),
        1.0,
    )

    legal_move = legal[A_MOVE:A_REVEAL].reshape(N_CARDS, N_CELLS)
    # Substeps other than 2 expose no legal moves, so fall back on a public
    # proxy (empty cell touching another card) to keep the lookahead useful.
    occupied = jnp.zeros((N_CELLS,), jnp.float32).at[card_cell].add(1.0)
    free = (occupied[None, :] == 0.0)                                # [1, 81]
    touching = (cell_adj.sum(-1) > 0.0)[None, :]                     # [1, 81]
    proxy_move = free & touching & (~locked)[:, None]
    move_mask = jnp.where(substep == 2, legal_move, proxy_move)
    mono = jnp.where(move_mask, mono, -1e9).reshape(-1)

    top_val, top_idx = jax.lax.top_k(mono, TOP_K)

    def _eval_move(flat):
        new_pos = pos.at[flat // N_CELLS].set(tb['cellpos'][flat % N_CELLS])
        part_mono, edges, _ = _board_stats(new_pos, tb)
        movable = _n_movable(new_pos, locked_f)
        # A colouring needing d = 6 - mono further relocations can be finished
        # at the end of turn (turns + 1 + d); score it accordingly.
        finish = jnp.clip(
            (turns + 1).astype(jnp.float32) + 6.0 - part_mono, 0.0, 15.0
        ).astype(jnp.int32)
        joined = (part_mono >= 5.5).astype(jnp.float32)
        value = (wpart * value_tab[finish]).sum() / wtot
        p_all = (wpart * joined).sum() / wtot
        p_partner = (pwpart * joined).sum() / pwtot
        return value, p_all, EDGE_W * edges + GAMMA * movable, p_partner

    cand_value, cand_p, cand_edges, cand_pp = jax.vmap(_eval_move)(top_idx)
    # With an unhelpful partner the long plan behind ``cand_value`` will not
    # survive, so weight immediate certainty instead: it is what allows this
    # seat to end the episode on its own next substep 0.
    cand_score = jnp.where(
        top_val > -1e8,
        cand_value
        + P_SHIFT
        * (1.0 - quality)
        * cand_p
        * value_tab[jnp.minimum(turns + 2, 8)]
        + BETA * top_val
        + ALPHA * cand_edges
        + KAPPA * (cand_p >= 0.999).astype(jnp.float32)
        + MU
        * jnp.where(turns >= MU_FROM, 1.0, 0.0)
        * (cand_pp >= 0.999).astype(jnp.float32)
        * value_tab[jnp.minimum(turns + 1, 8)],
        -1e9,
    )
    best_i = jnp.argmax(cand_score)
    move_action = A_MOVE + top_idx[best_i]
    p_best = cand_p[best_i]
    value_best = cand_value[best_i]
    pp_best = cand_pp[best_i]

    # Hand the partner a reference point: the best expected monochrome-pair
    # count reachable in one move from the board this seat is leaving behind.
    ref = jnp.where(active & (substep == 3), top_val[0], prev_ref)

    # ---- end-game value comparison ---------------------------------------
    unrevealed = (N_HINTS - n_revealed).astype(jnp.float32)
    revealed_unplaced = (n_revealed - n_placed).astype(jnp.float32)
    p_correct_hint = (hint_set * q[hint_card]).sum(-1)               # [4]
    placed_value = jnp.where(placed, 2.0 * p_correct_hint - 1.0, 0.0).sum()
    end_value = p_now * (5.0 * unrevealed + 2.0 * revealed_unplaced + placed_value)
    floor_tab = jnp.asarray(CONTINUE_FLOOR, dtype=jnp.float32)
    continue_value = jnp.maximum(floor_tab[turns], RHO * value_best) * quality
    want_end = (end_value >= continue_value) & legal[A_END]

    # ---- observation choice -----------------------------------------------
    # Unobserved cards first; among them a fixed low-id order keeps the two
    # seats' beliefs aligned without requiring the partner to agree, optionally
    # biased towards cards with many neighbours, whose colour constrains
    # connectivity the most.
    obs_score = (
        jnp.where(known < 0, 1.0, 0.0)
        + OBS_DEG * cur_adj.sum(-1)
        + 0.01 * (N_CARDS - jnp.arange(N_CARDS, dtype=jnp.float32))
    )
    legal_obs = legal[A_OBSERVE:A_OBSERVE + N_CARDS]
    observe_action = A_OBSERVE + jnp.argmax(jnp.where(legal_obs, obs_score, -1e9))

    # ---- substep 3: place a revealed hint, else reveal a new one ----------
    place_legal = legal[A_PLACE:A_END].reshape(N_HINTS, N_CELLS)[:, card_cell]
    p_correct = hint_set @ q.T                                        # [4, 9]
    # A hint placement locks its card for the rest of the episode, so target a
    # card whose own colour group is already joined: it should not need to move
    # again, and locking it protects the finished group from later disturbance.
    settle = SETTLE * card_done + 0.25 * jnp.where(known >= 0, 1.0, 0.0)
    flat_place = jnp.where(place_legal, p_correct + settle[None, :], -1e9).reshape(-1)
    best_place = jnp.argmax(flat_place)
    place_h = best_place // N_CARDS
    place_j = best_place % N_CARDS
    place_action = A_PLACE + place_h * N_CELLS + card_cell[place_j]
    place_ok = flat_place[best_place] > -1e8
    best_p_correct = p_correct[place_h, place_j]
    best_done = card_done[place_j]

    legal_reveal = legal[A_REVEAL:A_PLACE]
    # Two-colour hints (ids 1..3) are easier to place correctly than hint 0.
    reveal_pref = jnp.asarray([0.0, 1.0, 0.9, 0.8], dtype=jnp.float32)
    reveal_action = A_REVEAL + jnp.argmax(jnp.where(legal_reveal, reveal_pref, -1e9))
    reveal_ok = legal_reveal.any()

    prefer_place = place_ok & (
        ((best_p_correct >= PLACE_THRESHOLD) & (best_done >= DONE_THRESHOLD))
        | (~reveal_ok)
    )
    hint_action = jnp.where(
        prefer_place, place_action, jnp.where(reveal_ok, reveal_action, place_action)
    )

    # ---- assemble ----------------------------------------------------------
    action = jnp.where(
        substep == 0,
        jnp.where(want_end, A_END, observe_action),
        jnp.where(
            substep == 1,
            observe_action,
            jnp.where(substep == 2, move_action, hint_action),
        ),
    )
    action = jnp.where(active, action, 0).astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal)).astype(jnp.int32)

    trace = {
        'p_now': p_now,
        'p_best': p_best,
        'value_best': value_best,
        'mono_best': top_val[best_i],
        'pp_best': pp_best,
        'quality': quality,
        'turns': turns.astype(jnp.int32),
        'n_known': (known >= 0).sum().astype(jnp.int32),
        'end_value': end_value,
        'continue_value': continue_value,
        'action': action,
    }
    return (
        action,
        {
            'known': known,
            'seen': seen,
            'ref': ref,
            'gap_sum': gap_sum,
            'gap_n': gap_n,
        },
        trace,
    )
