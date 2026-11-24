"""Reactive Yokai coordination policy.

Design notes (see experiment_ledger.md attempts A01-A02 for rationale):
  * The environment's "perfect memory" mode replays a seat's full private
    observation history into node_features whenever that seat is active, so
    no private memory carry is needed -- the policy is purely reactive.
  * Hint economy: an unrevealed hint is worth 5, a revealed-but-unplaced
    hint is worth a flat 2 regardless of correctness, and a placed hint is
    worth at most +1 (or -1). Placing is therefore only ever forced (no
    unrevealed hint remains); revealing is always weakly preferred while
    an unrevealed hint is available.
  * Ending the game (action 1067) is only attempted once the full 9-card
    color layout is either directly observed or deducible by elimination
    (at most one color still unidentified), and the deduced layout already
    satisfies per-color connectivity.
  * Forced card moves greedily maximize same-known-color adjacency using a
    local delta (new neighbor gain minus old neighbor loss), which naturally
    avoids breaking up already-good same-color clusters; ties favor keeping
    the board compact (more total neighbors at the target cell).
  * Placing a hint permanently locks its target card, so forced placements
    prefer cards that are already settled (known color, existing same-color
    neighbor) over cards that may still need to move, and use possible-color
    exclusion (not just exact knowledge) to detect guaranteed-correct or
    guaranteed-wrong placements.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'bare_candidate',
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

NEG_INF = -1e9
NUM_ACTIONS = 1068
DECK = 9
GRID = 9
NUM_CELLS = GRID * GRID
NUM_COLORS = 3
NUM_HINTS = 4


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _deduce_colors(node_features):
    own = node_features[:DECK, 0:3].astype(jnp.int32)
    known_mask = jnp.sum(own, axis=-1) > 0
    color_known = jnp.argmax(own, axis=-1)
    counts = jnp.sum(own, axis=0)
    remaining_need = 3 - counts
    num_needing = jnp.sum(remaining_need > 0)
    deducible = num_needing <= 1
    deduced_color = jnp.argmax(remaining_need)
    eff_known_mask = known_mask | deducible
    eff_color = jnp.where(known_mask, color_known, deduced_color)

    # Even without full deduction, an unobserved card's color is constrained
    # to whichever colors are not yet fully accounted for by known cards
    # (a color with remaining_need == 0 is impossible for any unknown card).
    onehot_known = jnp.arange(NUM_COLORS)[None, :] == color_known[:, None]
    still_possible = remaining_need > 0
    possible_colors = jnp.where(known_mask[:, None], onehot_known, still_possible[None, :])
    return eff_known_mask, eff_color, deducible, possible_colors


def _connectivity_ok(eff_known_mask, eff_color, positions):
    pos = positions.astype(jnp.int32)
    diff = pos[:, None, :] - pos[None, :, :]
    manh = jnp.sum(jnp.abs(diff), axis=-1)
    adj = manh == 1

    same_color = (eff_color[:, None] == eff_color[None, :])
    known_pair = eff_known_mask[:, None] & eff_known_mask[None, :]
    same_color_adj = adj & same_color & known_pair

    reach = same_color_adj | jnp.eye(DECK, dtype=bool)
    for _ in range(4):
        reach_i = reach.astype(jnp.int32)
        sq = (reach_i @ reach_i) > 0
        reach = reach | sq

    not_diag = ~jnp.eye(DECK, dtype=bool)
    pair_same_color = same_color & known_pair & not_diag
    violation = pair_same_color & (~reach)
    return ~jnp.any(violation)


def _observe_end_score(node_features, eff_known_mask, end_condition):
    other_observed = node_features[:DECK, 9] > 0
    pref = jnp.where(eff_known_mask, jnp.float32(-10.0), jnp.float32(0.0))
    pref = pref + jnp.where(other_observed, jnp.float32(-1.0), jnp.float32(0.0))

    score = jnp.full((NUM_ACTIONS,), NEG_INF, dtype=jnp.float32)
    score = score.at[1:10].set(pref)
    score = score.at[1067].set(jnp.where(end_condition, jnp.float32(1e6), NEG_INF))
    score = score.at[0].set(jnp.float32(0.0))
    return score


def _move_score(eff_known_mask, eff_color, positions):
    pos = positions.astype(jnp.int32)
    cell_idx = jnp.arange(NUM_CELLS, dtype=jnp.int32)
    cx = cell_idx // GRID
    cy = cell_idx % GRID

    dist_to_cell = jnp.abs(pos[:, 0:1] - cx[None, :]) + jnp.abs(pos[:, 1:2] - cy[None, :])
    adjacent_mask = (dist_to_cell == 1).astype(jnp.float32)  # (9, 81)

    diff = pos[:, None, :] - pos[None, :, :]
    manh = jnp.sum(jnp.abs(diff), axis=-1)
    curr_adjacent = manh == 1  # (9, 9)

    same_color_pair = (
        (eff_color[:, None] == eff_color[None, :])
        & eff_known_mask[:, None]
        & eff_known_mask[None, :]
        & (~jnp.eye(DECK, dtype=bool))
    )
    same_color_pair_f = same_color_pair.astype(jnp.float32)

    new_contrib = same_color_pair_f @ adjacent_mask  # (9, 81)
    old_contrib = jnp.sum(same_color_pair_f * curr_adjacent.astype(jnp.float32), axis=1)  # (9,)

    # Small tie-break: among otherwise-equal candidates, prefer keeping the
    # board compact (more total neighbors at the target cell), which leaves
    # more legal repositioning options available for later, better-informed
    # turns without ever overriding a genuine known-color adjacency gain.
    total_neighbor_count = jnp.sum(adjacent_mask, axis=0)  # (81,)
    move_score = new_contrib - old_contrib[:, None] + jnp.float32(0.1) * total_neighbor_count[None, :]
    flat = move_score.reshape(-1)

    score = jnp.full((NUM_ACTIONS,), NEG_INF, dtype=jnp.float32)
    score = score.at[10:10 + DECK * NUM_CELLS].set(flat)
    score = score.at[0].set(jnp.float32(0.0))
    return score


def _hint_score(node_features, eff_known_mask, eff_color, possible_colors, positions):
    hint_rows = node_features[DECK:DECK + NUM_HINTS, :]
    revealed_mask = jnp.sum(hint_rows[:, 3:6].astype(jnp.int32), axis=-1) > 0
    unrevealed = ~revealed_mask

    hint_colorset = hint_rows[:, 3:6] > 0  # (4, 3)

    pos = positions.astype(jnp.int32)
    cell_idx = jnp.arange(NUM_CELLS, dtype=jnp.int32)
    cx = cell_idx // GRID
    cy = cell_idx % GRID
    match_pos = (pos[:, 0:1] == cx[None, :]) & (pos[:, 1:2] == cy[None, :])  # (9, 81)
    card_at_cell = jnp.argmax(match_pos.astype(jnp.int32), axis=0)  # (81,)

    # A card's color need not be fully known to be provably safe or unsafe
    # to hint: if every color still possible for it (given colors already
    # fully accounted for elsewhere) lies inside the hint's set, placement
    # is guaranteed correct; if none of its possible colors overlap the
    # hint's set, placement is guaranteed wrong.
    possible_at_cell = possible_colors[card_at_cell]  # (81, 3)
    implication = (~possible_at_cell[None, :, :]) | hint_colorset[:, None, :]  # (4, 81, 3)
    guaranteed_match = jnp.all(implication, axis=-1)  # (4, 81)
    intersection = possible_at_cell[None, :, :] & hint_colorset[:, None, :]  # (4, 81, 3)
    guaranteed_mismatch = ~jnp.any(intersection, axis=-1)  # (4, 81)

    place_score = jnp.where(
        guaranteed_match, jnp.float32(2.0),
        jnp.where(guaranteed_mismatch, jnp.float32(-2.0), jnp.float32(0.0)),
    )

    # Placing permanently locks the target card (no further moves or
    # observations for it), so prefer locking cards that are already
    # well-settled (known color with an existing same-color neighbor) over
    # cards that may still need to be repositioned to reach connectivity.
    diff = pos[:, None, :] - pos[None, :, :]
    manh = jnp.sum(jnp.abs(diff), axis=-1)
    curr_adjacent = manh == 1
    same_color_pair = (
        (eff_color[:, None] == eff_color[None, :])
        & eff_known_mask[:, None]
        & eff_known_mask[None, :]
        & (~jnp.eye(DECK, dtype=bool))
    )
    settled_count = jnp.sum((same_color_pair & curr_adjacent).astype(jnp.float32), axis=1)  # (9,)
    lock_safety = jnp.where(
        eff_known_mask, settled_count, jnp.float32(-0.5)
    )  # unknown cards are risky to lock
    lock_safety_at_cell = lock_safety[card_at_cell]  # (81,)

    place_score = place_score + jnp.float32(0.25) * lock_safety_at_cell[None, :]
    flat_place = place_score.reshape(-1)

    score = jnp.full((NUM_ACTIONS,), NEG_INF, dtype=jnp.float32)
    reveal_scores = jnp.where(unrevealed, jnp.float32(1000.0), NEG_INF)
    score = score.at[739:739 + NUM_HINTS].set(reveal_scores)
    score = score.at[743:743 + NUM_HINTS * NUM_CELLS].set(flat_place)
    score = score.at[0].set(jnp.float32(0.0))
    return score


def agent_act(obs, reset, memory, rng):
    del rng
    node_features = obs.node_features
    substep = node_features[0, 11].astype(jnp.int32)

    eff_known_mask, eff_color, deducible, possible_colors = _deduce_colors(node_features)
    connectivity_ok = _connectivity_ok(eff_known_mask, eff_color, obs.positions)
    end_condition = deducible & connectivity_ok

    score_observe = _observe_end_score(node_features, eff_known_mask, end_condition)
    score_move = _move_score(eff_known_mask, eff_color, obs.positions)
    score_hint = _hint_score(node_features, eff_known_mask, eff_color, possible_colors, obs.positions)

    score = jnp.where(substep == 2, score_move, jnp.where(substep == 3, score_hint, score_observe))
    score = jnp.where(obs.legal_actions, score, NEG_INF)
    score = score.at[0].set(jnp.where(obs.legal_actions[0], jnp.float32(0.0), NEG_INF))

    action = jnp.argmax(score).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)

    trace = {
        'deducible': deducible.astype(jnp.int32),
        'end_condition': end_condition.astype(jnp.int32),
        'connectivity_ok': connectivity_ok.astype(jnp.int32),
        'substep': substep,
    }
    return action, next_memory, trace
