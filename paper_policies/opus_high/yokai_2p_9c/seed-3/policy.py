"""Two-player Yokai policy.

Each seat keeps an exact uniform posterior over the colour assignments that are
consistent with the colours it has personally observed, greedily moves cards so
that the posterior probability of full colour connectivity rises, spends hint
actions by explicit point cost, and ends the episode when its own posterior says
the board is almost surely clustered.  No agreement with the partner about
roles, target layouts or signalling is assumed anywhere.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
GRID = 9
NUM_SAMPLES = 64
NUM_ASSIGN = 1680

# Value of playing on instead of ending, indexed by completed turns
# (= revealed hints + placed hints).  Every episode that runs the full eight
# turns scores C-I <= 4, so the continuation value sits just above 4.
CONTINUE_VALUE = [2.6, 2.6, 2.6, 2.6, 2.5, 2.5, 2.4, 2.3]
# Likelihood factor for a world that contradicts a hint already placed on a card.
# Soft on purpose: a partner that places hints badly must not be able to drive
# this seat's belief to a confident wrong answer.
# Never end on a board the belief cannot vouch for, however many hints are left.
MIN_END_PROBABILITY = 0.2
HINT_TRUST = 0.15
# Strength of the evidence that the partner relocated a card whose colour they
# have seen: worlds where the card landed beside same-coloured cards gain weight
# in proportion to the number of such neighbours.
MOVE_EVIDENCE = 1.0
# Extra cost charged for locking a card that is not yet settled in its colour
# group, in the reveal-versus-place comparison at substep 3.
LOCK_RISK = 1.5
# Locking a card the partner has never observed both costs them an observation
# target and tells them the hint's colour set; the sign of the net effect is
# tuned rather than assumed.
INFO_PENALTY = 0.5
# Preference for spending hints that few cards can satisfy while options remain.
SCARCITY_WEIGHT = 0.2
# Weight of the "expected same-colour adjacent pairs" shaping term.
POTENTIAL_WEIGHT = 0.30
# Weight on shrinking the remaining distance between cards of the same colour.
SPREAD_WEIGHT = 0.003
# Reluctance to relocate a card whose colour this seat has never seen, and extra
# reluctance when the partner has seen it (they may have placed it on purpose).
UNKNOWN_MOVE_PENALTY = 0.08
PARTNER_CARD_PENALTY = 0.08
# Among cards this seat has not seen, prefer ones the partner has NOT seen: the
# team then holds more distinct colours and both seats can act on real knowledge.
OVERLAP_BONUS = -1.0
# Preference for looking at the card whose colour the belief is least certain of.
UNCERTAINTY_BONUS = 2.0
# Tiny preference for short relocations, used only to break ties.
DISTANCE_WEIGHT = 0.001


def _assignment_table():
    """All 1680 labellings of 9 cards with three cards of each of three colours."""
    grid = jnp.arange(729)
    i = grid // 81
    j = (grid // 9) % 9
    k = grid % 9
    picks = jnp.nonzero((i < j) & (j < k), size=84)[0]
    first = jnp.stack([picks // 81, (picks // 9) % 9, picks % 9], axis=1)      # [84, 3]

    grid6 = jnp.arange(216)
    i6 = grid6 // 36
    j6 = (grid6 // 6) % 6
    k6 = grid6 % 6
    picks6 = jnp.nonzero((i6 < j6) & (j6 < k6), size=20)[0]
    second = jnp.stack([picks6 // 36, (picks6 // 6) % 6, picks6 % 6], axis=1)  # [20, 3]

    rows = jnp.arange(84)
    member = jnp.zeros((84, NUM_CARDS), dtype=jnp.int32).at[rows[:, None], first].set(1)
    rest = jnp.argsort(member, axis=1)[:, :6]                                  # [84, 6]

    first_b = jnp.broadcast_to(first[:, None, :], (84, 20, 3))
    rest_b = jnp.broadcast_to(rest[:, None, :], (84, 20, 6))
    second_b = jnp.broadcast_to(second[None, :, :], (84, 20, 3))
    second_cards = jnp.take_along_axis(rest_b, second_b, axis=2)               # [84, 20, 3]

    labels = jnp.full((84, 20, NUM_CARDS), 2, dtype=jnp.int32)
    ra = jnp.arange(84)[:, None, None]
    rb = jnp.arange(20)[None, :, None]
    labels = labels.at[ra, rb, first_b].set(0)
    labels = labels.at[ra, rb, second_cards].set(1)
    return labels.reshape(-1, NUM_CARDS)


def init_agent():
    return {
        'known': -jnp.ones(NUM_CARDS, dtype=jnp.int8),
        'log_weight': jnp.zeros(NUM_ASSIGN, dtype=jnp.float32),
        'previous_cells': jnp.zeros(NUM_CARDS, dtype=jnp.int16),
        'was_active': jnp.zeros((), dtype=jnp.bool_),
    }


def agent_act(obs, reset, memory, rng):
    assign = _assignment_table()                                    # [1680, 9]
    same_static = (assign[:, :, None] == assign[:, None, :]).astype(jnp.float32)
    onehot_static = (
        assign[:, :, None] == jnp.arange(NUM_COLORS)[None, None, :]
    ).astype(jnp.float32)
    deltas = jnp.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
    continue_value = jnp.asarray(CONTINUE_VALUE, dtype=jnp.float32)

    features = obs.node_features.astype(jnp.int32)
    positions = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions

    # ------------------------------------------------------------- decoding --
    cards = features[:NUM_CARDS]
    hints = features[NUM_CARDS:]
    own_onehot = cards[:, 0:NUM_COLORS]
    seen_now = own_onehot.sum(axis=1) > 0
    colour_now = jnp.argmax(own_onehot, axis=1)
    partner_seen = cards[:, 9] > 0
    active = cards[0, 10] > 0
    substep = cards[0, 11]

    fresh = -jnp.ones(NUM_CARDS, dtype=jnp.int8)
    remembered = jnp.where(reset, fresh, memory['known']).astype(jnp.int32)
    known = jnp.where(seen_now, colour_now, remembered)              # [9], -1 unknown

    cells_now = (positions[:, 0] * GRID + positions[:, 1]).astype(jnp.int32)
    previous_cells = jnp.where(
        reset, cells_now, memory['previous_cells'].astype(jnp.int32)
    )
    was_active = jnp.logical_and(memory['was_active'], jnp.logical_not(reset))

    hint_sets = (hints[:, 3:6] > 0).astype(jnp.float32)              # [4, 3]
    hint_revealed = hint_sets.sum(axis=1) > 0
    hint_placed = hints[:, 8] > 0
    hint_pos = hints[:, 6:8]
    turns_done = hint_revealed.sum() + hint_placed.sum()

    adjacency = (
        jnp.abs(positions[:, None, :] - positions[None, :, :]).sum(axis=-1) == 1
    ).astype(jnp.float32)                                            # [9, 9]

    # A hint that sits on a card is public evidence: whoever placed it normally
    # knew the colour, so worlds contradicting the placement are down-weighted.
    # The factor is soft, so an unreliable partner cannot force a wrong belief.
    placed_here = (
        jnp.all(positions[None, :, :] == hint_pos[:, None, :], axis=-1)
        & hint_placed[:, None]
    ).astype(jnp.float32)                                            # [4, 9]
    card_hint = jnp.einsum('hj,hc->jc', placed_here, hint_sets)      # [9, 3]
    card_locked = placed_here.sum(axis=0)                            # [9]

    # ------------------------------------------------ exact belief over cards -
    hint_ok = jnp.einsum('kjc,jc->kj', onehot_static, card_hint)
    violations = jnp.einsum('kj,j->k', 1.0 - hint_ok, card_locked)

    # dsum[k, i]: same-colour neighbours of card i; group[k, i]: twice the number
    # of same-colour adjacent pairs inside the colour class of card i.
    dsum = jnp.einsum('kij,ij->ki', same_static, adjacency)

    # A card that the partner relocated, and whose colour the partner has seen,
    # was almost certainly put next to its own colour.  Worlds that agree with
    # that reading gain weight; the factor is bounded so a careless partner only
    # blurs the belief instead of dictating it.
    partner_moved = (
        (cells_now != previous_cells)
        & jnp.logical_not(was_active)
        & partner_seen
        & (known < 0)
    ).astype(jnp.float32)                                            # [9]
    evidence = jnp.einsum(
        'ki,i->k', jnp.log1p(MOVE_EVIDENCE * dsum), partner_moved
    )
    log_weight = jnp.where(reset, 0.0, memory['log_weight']) + evidence
    log_weight = log_weight - jnp.max(log_weight)

    weights = jnp.all(
        (known[None, :] < 0) | (assign == known[None, :]), axis=1
    ).astype(jnp.float32) * jnp.power(HINT_TRUST, violations) * jnp.exp(log_weight)
    total_weight = jnp.maximum(weights.sum(), 1e-6)
    group = jnp.einsum('kij,kj->ki', same_static, dsum)
    connected = (jnp.min(group, axis=1) >= 3.5).astype(jnp.float32)
    p_connected = (weights * connected).sum() / total_weight
    marginal = jnp.einsum('k,kjc->jc', weights, onehot_static) / total_weight
    group_safe = jnp.einsum(
        'k,kj->j', weights, (group >= 3.5).astype(jnp.float32)
    ) / total_weight
    attached = jnp.einsum(
        'k,kj->j', weights, (dsum >= 0.5).astype(jnp.float32)
    ) / total_weight
    settled = 0.5 * group_safe + 0.5 * attached

    # --------------------------------------------------- expected end score ---
    placement_score = jnp.einsum('kj,j->k', 2.0 * hint_ok - 1.0, card_locked)
    unrevealed = jnp.logical_not(hint_revealed).sum().astype(jnp.float32)
    revealed_free = jnp.logical_and(
        hint_revealed, jnp.logical_not(hint_placed)
    ).sum().astype(jnp.float32)
    end_score = 5.0 * unrevealed + 2.0 * revealed_free + placement_score
    expected_end = (weights * connected * end_score).sum() / total_weight

    # ---------------------------------------------------- sampled posterior ---
    # Draw worlds from the same weighted posterior used for the end decision, so
    # that move search and ending never disagree about what is believed.
    logits = jnp.log(jnp.maximum(weights, 1e-30))
    sampled = assign[jax.random.categorical(rng, logits, shape=(NUM_SAMPLES,))]
    sample_weight = jnp.full((NUM_SAMPLES,), 1.0 / NUM_SAMPLES, dtype=jnp.float32)
    same_s = (sampled[:, :, None] == sampled[:, None, :]).astype(jnp.float32)
    dsum_s = jnp.einsum('nij,ij->ni', same_s, adjacency)
    pairs_i = jnp.einsum('nij,nj->ni', same_s, dsum_s) / 2.0
    total_pairs = dsum_s.sum(axis=1) / 2.0                           # [N]
    other_pairs = total_pairs[:, None] - pairs_i                     # [N, 9]
    other_min = jnp.min(
        jnp.where(same_s > 0, 9.0, jnp.broadcast_to(pairs_i[:, None, :], same_s.shape)),
        axis=2,
    )
    other_ok = other_min >= 1.9                                      # [N, 9]

    # -------------------------------------------------------- move selection --
    candidates = (positions[:, None, :] + deltas[None, :, :]).reshape(-1, 2)   # [36, 2]
    in_grid = jnp.all((candidates >= 0) & (candidates < GRID), axis=-1)
    clipped = jnp.clip(candidates, 0, GRID - 1)
    cell_id = clipped[:, 0] * GRID + clipped[:, 1]
    neighbours = (
        jnp.abs(clipped[:, None, :] - positions[None, :, :]).sum(axis=-1) == 1
    ).astype(jnp.float32)                                            # [36, 9]

    spacing = jnp.abs(positions[:, None, :] - positions[None, :, :]).sum(axis=-1)
    spacing = spacing.astype(jnp.float32)
    to_cell = jnp.abs(clipped[:, None, :] - positions[None, :, :]).sum(axis=-1)
    to_cell = to_cell.astype(jnp.float32)                            # [36, 9]

    gained = jnp.einsum('pj,nij->nip', neighbours, same_s) - neighbours.T[None, :, :]
    new_pairs = pairs_i[:, :, None] - dsum_s[:, :, None] + gained
    new_ok = new_pairs >= 1.9
    connected_after = jnp.logical_and(other_ok[:, :, None], new_ok).astype(jnp.float32)
    potential_after = other_pairs[:, :, None] + jnp.minimum(new_pairs, 2.0)
    # Smooth guidance for boards that cannot be finished by this single move:
    # how far apart the cards of one colour still are once the card has moved.
    spread_before = jnp.einsum('nij,ij->ni', same_s, spacing)
    spread_after = jnp.einsum('pj,nij->nip', to_cell, same_s)
    spread = jnp.einsum('n,nip->ip', sample_weight, spread_after - spread_before[:, :, None])

    move_score = jnp.einsum('n,nip->ip', sample_weight, connected_after) + (
        POTENTIAL_WEIGHT * jnp.einsum('n,nip->ip', sample_weight, potential_after) / 6.0
    ) - SPREAD_WEIGHT * spread
    distance = jnp.abs(clipped[None, :, :] - positions[:, None, :]).sum(axis=-1)
    move_score = move_score - DISTANCE_WEIGHT * distance.astype(jnp.float32)
    unknown_card = (known < 0).astype(jnp.float32)
    disturb = UNKNOWN_MOVE_PENALTY * unknown_card + PARTNER_CARD_PENALTY * (
        unknown_card * partner_seen.astype(jnp.float32)
    )
    move_score = move_score - disturb[:, None]

    legal_moves = legal[10:739].reshape(NUM_CARDS, GRID * GRID)
    move_ok = legal_moves[:, cell_id] & in_grid[None, :]
    move_score = jnp.where(move_ok, move_score, -1e9)
    best_move = jnp.argmax(move_score.reshape(-1))
    move_card = best_move // 36
    move_cell = best_move % 36
    move_action = 10 + move_card * (GRID * GRID) + cell_id[move_cell]

    # ---------------------------------------------------- observation choice --
    observe_legal = legal[1:10]
    # Spend a look on the card whose colour the belief is least sure about; a
    # card the evidence has already pinned down teaches almost nothing.
    uncertainty = 1.0 - jnp.max(marginal, axis=1)
    observe_score = (
        4.0 * (known < 0).astype(jnp.float32)
        + UNCERTAINTY_BONUS * uncertainty
        + OVERLAP_BONUS * partner_seen.astype(jnp.float32)
    )
    observe_score = jnp.where(observe_legal, observe_score, -1e9)
    observe_action = 1 + jnp.argmax(observe_score)

    # ---------------------------------------------------------- hint choice ---
    reveal_legal = legal[739:743]
    # Hint 0 shows a single colour: revealing it first measured better, since a
    # single-colour hint on a card is the sharpest public colour evidence there is.
    reveal_action = 739 + jnp.argmax(
        jnp.where(reveal_legal, -jnp.arange(4, dtype=jnp.float32), -1e9)
    )
    reveal_cost = jnp.where(reveal_legal.any(), 3.0, 1e9)

    card_cell = positions[:, 0] * GRID + positions[:, 1]
    place_legal = legal[743:1067].reshape(4, GRID * GRID)[:, card_cell]         # [4, 9]
    correct_p = jnp.einsum('hc,jc->hj', hint_sets, marginal)
    scarcity = (correct_p * place_legal.astype(jnp.float32)).sum(axis=1)        # [4]
    place_cost = (
        (3.0 - 2.0 * correct_p)
        + LOCK_RISK * (1.0 - settled[None, :])
        + INFO_PENALTY * jnp.logical_not(partner_seen)[None, :].astype(jnp.float32)
        + SCARCITY_WEIGHT * scarcity[:, None]
    )
    place_cost = jnp.where(place_legal, place_cost, 1e9)
    best_place = jnp.argmin(place_cost.reshape(-1))
    place_hint = best_place // NUM_CARDS
    place_card = best_place % NUM_CARDS
    place_action = 743 + place_hint * (GRID * GRID) + card_cell[place_card]

    hint_action = jnp.where(jnp.min(place_cost) < reveal_cost, place_action, reveal_action)

    # ------------------------------------------------------------- end choice -
    keep_playing = continue_value[jnp.clip(turns_done, 0, 7)]
    do_end = (
        legal[1067]
        & (expected_end >= keep_playing)
        & (p_connected >= MIN_END_PROBABILITY)
    )
    substep0_action = jnp.where(do_end, 1067, observe_action)

    action = jnp.where(
        substep == 0,
        substep0_action,
        jnp.where(substep == 1, observe_action, jnp.where(substep == 2, move_action, hint_action)),
    )
    action = jnp.where(active, action, 0).astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal)).astype(jnp.int32)

    next_memory = {
        'known': known.astype(jnp.int8),
        'log_weight': log_weight,
        'previous_cells': cells_now.astype(jnp.int16),
        'was_active': active,
    }

    trace = {
        'p_connected': p_connected.astype(jnp.float32),
        'expected_end': expected_end.astype(jnp.float32),
        'turns_done': turns_done.astype(jnp.int32),
        'num_known': (known >= 0).sum().astype(jnp.int32),
        'action': action,
        'known': known.astype(jnp.int32),
        'cells': (positions[:, 0] * GRID + positions[:, 1]).astype(jnp.int32),
    }
    return action, next_memory, trace
