"""Organic color-cluster growth policy with evidence-based hint placement.

Strategy (identical code for both seats; only local, own observations, public
state and this seat's own PRNG/memory are used):

- Observe: prefer unlocked cards this seat has not personally seen yet.
  Among those, primarily avoid cards the partner has already been publicly
  seen to observe (a live, adaptive signal), with a deterministic per-seat
  forward/backward index preference (derived from the public fact of which
  seat is active at episode reset) used only as a small tie-break. Together
  these spread combined team coverage across more of the 16 cards than a
  fixed shared order, while still degrading gracefully if an independently
  generated partner explores differently or gives no useful signal.
- Move: for a card whose color this seat personally knows, steer it toward
  whichever position minimizes distance to (a) the nearest OTHER card of the
  same color this seat has already personally observed, if any, so the known
  same-color cards attach into one growing connected cluster by induction; or
  (b) a fixed per-color escape anchor just outside a distinct corner of the
  initially fully packed 4x4 starting block, if this is the first known card
  of its color (a deterministic function of color id alone, not a private
  convention). The starting block has no spare cell, so aiming at a fixed
  final slot inside it causes gridlock; growing outward into empty territory
  means every beneficial destination is immediately free. Among several
  candidate known cards, the single move slot available each turn is
  prioritized by (i) how far a card still is from its target and (ii) an
  extra bonus for whichever card is the LAST remaining piece needed to fully
  connect its color, since connectivity is all-or-nothing across all 4
  colors and a finished color secures value permanently. A card with no
  improving destination is left alone; if the mandatory move must disturb
  something, moving an unknown-color card that is still at its pristine
  start position is preferred, then a known card that stays adjacent to its
  cluster without orphaning a same-color neighbor whose only connection was
  to it ("sideways-safe"), then a "leaf" over a "bridge" card, in that
  order -- since disturbing a card the partner may have deliberately placed
  (signaled simply by it having moved from its start cell) risks silently
  undoing the partner's own progress.
- Hints: always prefer revealing an unrevealed hint over placing one (never
  lowers, and typically raises, the eventual correctness of forced placements
  by allowing more observation turns first). When forced to place a hint,
  never lock a card that still needs to move to reach its color's cluster
  (known or not, judged by whether it has already moved from its start cell
  when unknown), since a stranded piece can zero out the whole episode;
  among the remaining safe cards, prefer one whose known color is confirmed
  to be in the hint's color set, avoid a card known NOT to match, and
  otherwise fall back to the action with best expected value given the
  hint's color-set size.
- End the game only when full connectivity is logically certain, never as a
  gamble: a single seat directly observing all 16 cards is nearly impossible
  inside the 14-turn cap. But the deck composition (exactly 4 colors of 4
  cards each) is fixed public knowledge, so once this seat has directly,
  personally observed all 4 members of 3 of the 4 colors, the remaining 4
  cards are provably the 4th color by elimination. Combined with the always
  public card positions, this gives a logically certain full-board
  connectivity verdict. Ending then is strictly best: every turn's move is
  mandatory with no "pass" option, so continuing after connectivity is
  already achieved can only risk breaking it for no benefit besides losing
  hint value. Otherwise the episode runs its natural course until all 7
  hints are placed (turn 14).

All logic is a single vectorized score-and-mask-and-argmax computation, with
no data-dependent Python control flow.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'organic_cluster_growth',
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

NUM_CARDS = 16
NUM_COLORS = 4
GRID = 10
NUM_HINTS = 7
NUM_ACTIONS = 2325
NEG = -1e9


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng

    node_features = obs.node_features
    positions = obs.positions
    legal = obs.legal_actions

    card_feats = node_features[:NUM_CARDS]
    hint_feats = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS]

    # Exploration role, fixed for the whole episode: whichever seat is active
    # at the very first observation is (by the public rules) player 0, who
    # always moves first; the other is player 1. This is an objective, public
    # fact derivable identically by any policy from its own observation alone
    # -- not a private convention -- so using it to assign complementary
    # observe orders (player 0 explores low-index-first, player 1
    # high-index-first) reliably roughly doubles combined team card coverage
    # over a fixed shared order, and degrades gracefully (to no worse than a
    # fixed shared order) if an independently generated partner does not use
    # the same idea.
    is_player0_now = card_feats[0, 12] > 0
    reverse_explore = jnp.where(reset, jnp.where(is_player0_now, 0, 1), memory).astype(jnp.int32)

    color_onehot = card_feats[:, 0:4].astype(jnp.int32)
    known = jnp.sum(color_onehot, axis=-1) > 0
    color_id = jnp.argmax(color_onehot, axis=-1)

    # Deck-composition deduction: the deck is fixed public knowledge (exactly
    # 4 colors of 4 cards each), so once this seat has directly, personally
    # observed all 4 members of 3 of the 4 colors, the remaining 4 cards
    # (whichever card IDs those are) must be the 4th color by elimination,
    # with full certainty, even though their color was never personally
    # observed. Folding this deduced knowledge into `known`/`color_id`
    # immediately (rather than only using it for the end-game certainty
    # check below) lets the observe, move and hint-placement logic all
    # benefit from it too: e.g. no need to waste an observation on an
    # already-deducible card, and a deduced card can be clustered, protected
    # from unsafe locking, or hint-matched just like a directly seen one.
    counts_per_color = jnp.sum(
        known[:, None] & (color_id[:, None] == jnp.arange(NUM_COLORS)[None, :]), axis=0
    )  # [4]
    fully_known_color = counts_per_color == NUM_CARDS // NUM_COLORS  # [4]
    num_fully_known = jnp.sum(fully_known_color)
    belongs_to_fully_known = known & fully_known_color[color_id]  # [16]
    leftover_mask = ~belongs_to_fully_known  # [16]
    can_deduce = num_fully_known >= 3
    missing_color_id = jnp.argmin(fully_known_color.astype(jnp.int32))
    deduced_mask = leftover_mask & can_deduce  # [16]
    known = known | deduced_mask
    color_id = jnp.where(deduced_mask, missing_color_id, color_id)

    pos_x = positions[:, 0]
    pos_y = positions[:, 1]

    # Whether each card still sits at its pristine reset position. A card
    # elsewhere is very likely a deliberate placement by whichever seat put
    # it there (this seat, or the other seat using knowledge this seat may
    # not have), used below to protect such cards from careless disturbance
    # by a fallback move or hint lock.
    original_x = (jnp.arange(NUM_CARDS) // 4 + 3).astype(jnp.int32)
    original_y = (jnp.arange(NUM_CARDS) % 4 + 3).astype(jnp.int32)
    still_original = (pos_x == original_x) & (pos_y == original_y)  # [16]

    hint_color_set = hint_feats[:, 4:8].astype(jnp.int32)
    hint_revealed = jnp.sum(hint_color_set, axis=-1) > 0
    hint_size = jnp.sum(hint_color_set, axis=-1)

    # Per-color escape anchors: one step outside a distinct corner of the
    # initial 4x4 block (x=3..6, y=3..6), so the very first known card of a
    # color has an immediately legal, uncontested place to start growing its
    # cluster. Anchoring at the four corners in a pinwheel layout (rather
    # than the four edge midpoints) maximizes the minimum pairwise distance
    # between different colors' anchors, reducing the chance that one
    # color's cluster, as it grows, spills into and blocks another color's
    # escape route or growth path.
    anchor_x = jnp.asarray([2, 7, 3, 6], dtype=jnp.int32)
    anchor_y = jnp.asarray([3, 6, 7, 2], dtype=jnp.int32)

    scores = jnp.full((NUM_ACTIONS,), NEG, dtype=jnp.float32)

    # no-op (only ever the sole legal action for the inactive seat)
    scores = scores.at[0].set(0.0)

    # observe actions [1, 17): prefer cards not yet personally observed.
    # Among unknown cards, primarily avoid ones the PARTNER has already been
    # publicly seen to observe (node_features column 11, valid under perfect
    # memory) -- this adapts to whatever the other seat actually does,
    # rather than assuming it follows the same direction convention, so it
    # degrades gracefully against an independently generated partner that
    # explores differently. The deterministic forward/backward rank (still
    # derived from the public fact of who is active at reset, not a private
    # convention) is kept only as a much smaller-magnitude tie-break for
    # when partner-seen status does not yet distinguish candidates (e.g.
    # early in the episode, or against a partner that gives no such signal).
    partner_seen = card_feats[:, 11] > 0  # [16]
    card_index = jnp.arange(NUM_CARDS, dtype=jnp.float32)
    rank = jnp.where(reverse_explore > 0, NUM_CARDS - 1 - card_index, card_index)
    observe_score = jnp.where(
        known, -5.0, jnp.where(partner_seen, 5.0, 10.0) - 0.01 * rank
    )
    scores = scores.at[1:17].set(observe_score)

    # end-game action: normally never chosen voluntarily (see module
    # docstring), UNLESS full board connectivity can be logically proven from
    # this seat's own observations plus the fixed 4-colors-of-4-cards deck
    # composition (see certain_win below), in which case ending immediately
    # is strictly best: every further forced move risks breaking an already
    # perfect, fully verified board (the mandatory move each turn has no
    # "pass" option, so once nothing needs to move, only harm can follow).
    # `fully_known_color`, `num_fully_known` and `leftover_mask` were already
    # computed above (before `known`/`color_id` were extended with deduced
    # cards) and reused here unchanged.
    group_mask = jnp.where(
        fully_known_color[:, None],
        known[None, :] & (color_id[None, :] == jnp.arange(NUM_COLORS)[:, None]),
        leftover_mask[None, :],
    )  # [4, 16]: row c is the card-id membership for color c (direct if
    #    fully known, else the deduced leftover group when applicable)

    adjacency16 = (jnp.abs(pos_x[:, None] - pos_x[None, :])
                   + jnp.abs(pos_y[:, None] - pos_y[None, :])) == 1  # [16, 16]
    base_reach = (adjacency16[None, :, :] & group_mask[:, :, None] & group_mask[:, None, :]) \
        | jnp.eye(NUM_CARDS, dtype=bool)[None, :, :]  # [4, 16, 16]
    reach = base_reach.astype(jnp.float32)
    for _ in range(4):
        step = jnp.einsum('cij,cjk->cik', reach, reach) > 0
        reach = jnp.maximum(reach, step.astype(jnp.float32))
    reach = reach > 0  # [4, 16, 16]

    anchor_idx = jnp.argmax(group_mask.astype(jnp.int32), axis=-1)  # [4]
    anchor_reach = reach[jnp.arange(NUM_COLORS), anchor_idx, :]  # [4, 16]
    group_connected = jnp.all(jnp.where(group_mask, anchor_reach, True), axis=-1)  # [4]

    certain_win = (num_fully_known >= 3) & jnp.all(group_connected)
    end_score = jnp.where(certain_win, jnp.float32(1e7), jnp.float32(-1e6))
    scores = scores.at[2324].set(end_score)

    # move actions [17, 1617): 17 + i*100 + x*10 + y
    xs = jnp.arange(GRID)
    ys = jnp.arange(GRID)
    xg, yg = jnp.meshgrid(xs, ys, indexing='ij')
    xg = xg.reshape(-1)  # [100]
    yg = yg.reshape(-1)  # [100]

    # distance from every destination cell to every card's current position
    dist_cell_to_card = (jnp.abs(xg[:, None] - pos_x[None, :])
                          + jnp.abs(yg[:, None] - pos_y[None, :])).astype(jnp.float32)  # [100, 16]

    same_color = color_id[:, None] == color_id[None, :]  # [16, 16]
    not_self = ~jnp.eye(NUM_CARDS, dtype=bool)
    cluster_mask = same_color & known[None, :] & not_self  # [16, 16], row i = usable partners for card i
    has_partner = jnp.any(cluster_mask, axis=-1)  # [16]

    big = jnp.float32(1e6)
    masked_dist = jnp.where(cluster_mask[:, None, :], dist_cell_to_card[None, :, :], big)  # [16, 100, 16]
    cluster_dist = jnp.min(masked_dist, axis=-1)  # [16, 100]

    anchor_x_i = anchor_x[color_id]  # [16]
    anchor_y_i = anchor_y[color_id]  # [16]
    anchor_dist = (jnp.abs(xg[None, :] - anchor_x_i[:, None])
                   + jnp.abs(yg[None, :] - anchor_y_i[:, None])).astype(jnp.float32)  # [16, 100]

    reference_dist = jnp.where(has_partner[:, None], cluster_dist, anchor_dist)  # [16, 100]

    # Distance from each card's CURRENT position to its own reference target,
    # so a card that is already optimally placed never gets rescored highly
    # just because a same-scoring alternative destination exists (which would
    # otherwise waste a scarce move on a purely sideways relocation).
    pd_card = (jnp.abs(pos_x[:, None] - pos_x[None, :])
               + jnp.abs(pos_y[:, None] - pos_y[None, :])).astype(jnp.float32)  # [16, 16]
    masked_pd = jnp.where(cluster_mask, pd_card, big)  # [16, 16]
    current_cluster_dist = jnp.min(masked_pd, axis=-1)  # [16]
    current_anchor_dist = (jnp.abs(pos_x - anchor_x_i) + jnp.abs(pos_y - anchor_y_i)).astype(jnp.float32)
    current_ref_dist = jnp.where(has_partner, current_cluster_dist, current_anchor_dist)  # [16]

    is_improvement = reference_dist < current_ref_dist[:, None]  # [16, 100]
    # A known card with no improving destination is left alone: fallback
    # moves must never prefer disturbing an already-settled known card (which
    # may be a load-bearing bridge holding its color's cluster together) over
    # moving a card of unknown color, so every forced-fallback tier below is
    # scored well below every unknown-card destination. Among forced
    # disturbances, prefer (a) a destination that stays adjacent to some
    # known same-color card even though it is not a strict improvement
    # ("sideways-safe") over one that abandons the cluster entirely, and (b)
    # prefer disturbing a "leaf" (few current same-color adjacencies) over a
    # "bridge" (many), since moving a bridge is more likely to disconnect it.
    adj_same_color = cluster_mask & (pd_card <= 1.0)  # [16, 16]
    current_degree = jnp.sum(adj_same_color, axis=-1)  # [16]
    # A "leaf" (degree 1) is not always safe to move: if its one same-color
    # neighbor has no OTHER same-color connection either (degree 1 in a
    # 2-node pair, or more generally the sole link keeping that neighbor
    # attached), moving the leaf away would orphan that neighbor even though
    # the leaf itself lands somewhere "sideways-safe". Detect this and treat
    # such a card as a critical bridge too, not a safe leaf.
    orphans_a_neighbor = jnp.any(adj_same_color & (current_degree == 1)[None, :], axis=-1)  # [16]
    sideways_safe = (reference_dist <= 1.0) & ~orphans_a_neighbor[:, None]  # [16, 100]
    forced_sideways_penalty = -50.0 - 10.0 * current_degree.astype(jnp.float32)[:, None]
    forced_abandon_penalty = -1500.0 - 500.0 * current_degree.astype(jnp.float32)[:, None]
    forced_penalty = jnp.where(sideways_safe, forced_sideways_penalty, forced_abandon_penalty)
    # A single move typically closes any starting gap in one hop (movement
    # is a relocation, not a path), so the destination term alone would let
    # cards that are already nearly done keep winning the one move slot each
    # turn purely because their absolute destination score is higher, while
    # far-behind cards are perpetually skipped -- yet skipping a straggler
    # is exactly as fatal to the all-or-nothing connectivity requirement as
    # skipping a nearly-finished card. Adding a bonus proportional to how far
    # a card currently is reprioritizes the scarce move slot toward the most
    # urgently neglected card when several known cards could improve this
    # turn, while the destination term still picks the best cell for
    # whichever card wins.
    urgency_bonus = 5.0 * current_ref_dist[:, None]

    # Connectivity is all-or-nothing across all 4 colors, so completing a
    # color for good is worth far more than partial progress on one that
    # will still need further, possibly unavailable, future turns. Give an
    # extra bonus to whichever misplaced known card is the LAST remaining
    # piece needed to fully connect its color (per this seat's own
    # knowledge), so the scarce move slot secures a finished color instead
    # of nudging a color that still has other pieces left to fix anyway.
    is_misplaced = known & has_partner & (current_ref_dist > 1.0)  # [16]
    color_remaining = jnp.sum(
        is_misplaced[:, None] & (color_id[:, None] == jnp.arange(NUM_COLORS)[None, :]), axis=0
    )  # [4]
    my_color_remaining = color_remaining[color_id]  # [16]
    completion_bonus = jnp.where(
        is_misplaced,
        jnp.where(
            my_color_remaining <= 1, 200.0,
            jnp.where(my_color_remaining == 2, 100.0, jnp.where(my_color_remaining == 3, 40.0, 0.0)),
        ),
        0.0,
    )  # [16]

    known_dest_score = jnp.where(
        is_improvement, 100.0 + urgency_bonus + completion_bonus[:, None] - 10.0 * reference_dist, forced_penalty
    )

    # A card (of unknown color to this seat) that has already moved away from
    # its pristine reset position is very likely a deliberate placement by
    # the OTHER seat (e.g. an anchor or cluster member it personally knows
    # about but this seat does not) -- moving it as a "harmless" unknown-card
    # fallback would silently undo that partner progress. Only cards still
    # sitting exactly where the episode started are treated as safe,
    # never-yet-claimed fallback material.
    unknown_dest_score = jnp.where(
        still_original[:, None],
        -10.0 - 0.1 * dist_cell_to_card.T,
        -200.0 - 0.1 * dist_cell_to_card.T,
    )  # [16, 100]

    move_score_card = jnp.where(known[:, None], known_dest_score, unknown_dest_score)
    scores = scores.at[17:1617].set(move_score_card.reshape(-1))

    # reveal actions [1617, 1624): 1617 + h
    any_unrevealed = jnp.any(~hint_revealed)
    reveal_score = jnp.where(any_unrevealed, jnp.float32(1000.0), NEG)
    scores = scores.at[1617:1624].set(jnp.full((NUM_HINTS,), reveal_score, dtype=jnp.float32))

    # place actions [1624, 2324): 1624 + h*100 + x*10 + y
    cell_idx_per_card = pos_x * 10 + pos_y
    card_at_cell = jnp.full((GRID * GRID,), -1, dtype=jnp.int32)
    card_at_cell = card_at_cell.at[cell_idx_per_card].set(jnp.arange(NUM_CARDS, dtype=jnp.int32))

    safe_idx = jnp.clip(card_at_cell, 0, NUM_CARDS - 1)
    known_at_cell = jnp.where(card_at_cell >= 0, known[safe_idx], False)
    color_at_cell = jnp.where(card_at_cell >= 0, color_id[safe_idx], 0)

    # A known card that has not yet reached its reference target (a same-
    # color partner, or its escape anchor while still the lone known member
    # of its color) still needs to move; if a hint locks it now it can never
    # get there, and since ALL colors must connect for any nonzero reward,
    # that single lock can zero out the whole episode. Strongly avoid
    # placing any hint on such a card, whether or not it has a partner yet.
    unsafe_to_lock = known & (current_ref_dist > 1.0)  # [16]
    # An unknown-color card that has already moved away from its pristine
    # start position is, by the same reasoning as the move fallback above,
    # very likely a deliberate placement by the other seat; locking it could
    # strand whatever cluster it belongs to just as surely as locking a
    # known-but-unsettled card would.
    unsafe_to_lock = unsafe_to_lock | (~known & ~still_original)  # [16]
    unsafe_at_cell = jnp.where(card_at_cell >= 0, unsafe_to_lock[safe_idx], False)  # [100]

    in_set = hint_color_set[:, color_at_cell] > 0  # [7, 100]
    ev_unknown = 2.0 * hint_size.astype(jnp.float32) / NUM_COLORS - 1.0  # [7]

    # A card that is safe to lock (known and either already connected or
    # still a lone anchor other cards can move to) costs nothing to place a
    # hint on regardless of match, since it no longer needs to move itself;
    # locking it instead of an unknown-color card preserves the unknown
    # card's future mobility, which matters far more than a single hint's
    # +/-2 correctness swing once weighed against the all-or-nothing
    # connectivity requirement. So safe-known cards are preferred for
    # placement over unknown-color cards even when the match is uncertain.
    place_known_score = jnp.where(in_set, 2.0, 0.1)
    place_score = jnp.where(known_at_cell[None, :], place_known_score, ev_unknown[:, None])
    place_score = jnp.where(unsafe_at_cell[None, :], -1000.0, place_score)
    scores = scores.at[1624:2324].set(place_score.reshape(-1))

    scores = jnp.where(legal, scores, NEG)
    action = jnp.argmax(scores).astype(jnp.int32)

    next_memory = reverse_explore
    trace = {
        'dbg_num_known': jnp.sum(known).astype(jnp.int32),
        'dbg_num_legal': jnp.sum(legal).astype(jnp.int32),
        'dbg_best_score': jnp.max(scores).astype(jnp.float32),
        'dbg_chosen_score': scores[action],
        'dbg_substep': card_feats[0, 13].astype(jnp.int32),
        'dbg_color_id': color_id.astype(jnp.int32),
        'dbg_known': known.astype(jnp.int32),
        'dbg_pos_x': pos_x.astype(jnp.int32),
        'dbg_pos_y': pos_y.astype(jnp.int32),
    }
    return action, next_memory, trace
