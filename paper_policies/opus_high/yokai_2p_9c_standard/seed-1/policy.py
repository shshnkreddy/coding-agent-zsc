"""Yokai 9-card cooperative policy.

Design summary
--------------
Each seat privately remembers every card colour it has observed.  The colour
multiset is public (three cards of each of three colours), so a seat can reason
with the exact posterior over the balanced colourings consistent with what it
has seen.  Every decision uses only this seat's own observations plus fully
public facts (positions, locks, hint sets, substep); no convention with the
partner is assumed.

* substep 0 -- end the episode when (posterior probability that all three
  colours are connected) * (score that ending now locks in) beats the value of
  playing on; otherwise inspect the lowest-id card of unknown colour.
* substep 1 -- inspect another unknown card.
* substep 2 -- two-ply search over relocations: rank legal moves with a
  same-colour attraction heuristic, then score the best ones by posterior
  connectivity now and after the best possible follow-up relocation.
* substep 3 -- prefer a certainly-correct hint placement (costs one point) over
  revealing a fresh hint (costs three), penalising locks on cards of unknown
  colour or in a colour group that is not yet whole.

The value of playing on is discounted by a running estimate of how much the
partner's relocations help or hurt: our own knowledge is unchanged between the
end of our turn and our next substep 0, so the change in board value across
that gap is attributable to the partner alone.  This is an inference from
public evidence, not an assumed convention, and it makes the seat stop earlier
when it is paired with someone who keeps breaking the board.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'yokai_posterior_connect', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}

NUM_CARDS = 9
NUM_COLORS = 3
GRID = 9
NUM_CELLS = 81
NUM_HINTS = 4
NUM_COLOURINGS = 1680
NUM_RAW = 19683
MOVE_BASE = 10
REVEAL_BASE = 739
PLACE_BASE = 743
END_ACTION = 1067

NUM_SAMPLES = 64          # posterior samples used inside the move search
TOP_K = 12                # first-ply relocation candidates
TOP_K2 = 8                # follow-up relocation candidates per first-ply move
LOOKAHEAD = 0.7           # weight of the best follow-up connectivity
W_PARTIAL = 0.25          # multiplicative partial-connectivity shaping
CONTINUE_VALUE = 3.9      # points a good partner is worth if we play on
QUALITY_DECAY = 0.5       # update rate of the running partner-quality estimate
PARTNER_SENSITIVITY = 3.0 # how strongly a damaging partner discounts playing on


def _static_tables():
    """Exhaustive balanced colourings and a cell grid (folded at trace time)."""
    raw = jnp.arange(NUM_RAW, dtype=jnp.int32)
    powers = jnp.power(jnp.asarray(3, dtype=jnp.int32), jnp.arange(NUM_CARDS, dtype=jnp.int32))
    digits = (raw[:, None] // powers[None, :]) % 3
    counts = (digits[:, :, None] == jnp.arange(NUM_COLORS)[None, None, :]).astype(jnp.int32).sum(1)
    balanced = jnp.all(counts == 3, axis=1)
    order = jnp.argsort(jnp.where(balanced, 0, 1), stable=True)[:NUM_COLOURINGS]
    colourings = digits[order]
    cells = jnp.arange(NUM_CELLS, dtype=jnp.int32)
    cell_xy = jnp.stack([cells // GRID, cells % GRID], axis=1).astype(jnp.float32)
    return colourings, cell_xy


def _pair_index(colourings):
    """Flat 9x9 adjacency indices of the three pairs inside each colour group."""
    num = colourings.shape[0]
    card_ids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    key = jnp.where(
        colourings[:, None, :] == jnp.arange(NUM_COLORS)[None, :, None],
        card_ids[None, None, :],
        card_ids[None, None, :] + 100,
    )
    members = jnp.sort(key, axis=2)[:, :, :3]
    return jnp.stack(
        [
            members[:, :, 0] * NUM_CARDS + members[:, :, 1],
            members[:, :, 0] * NUM_CARDS + members[:, :, 2],
            members[:, :, 1] * NUM_CARDS + members[:, :, 2],
        ],
        axis=2,
    ).reshape(num, NUM_COLORS * 3)


def _adjacency(positions):
    delta = jnp.abs(positions[:, None, :] - positions[None, :, :]).sum(-1)
    return (delta == 1).astype(jnp.float32)


def _group_edges(positions, pair_index):
    flat = _adjacency(positions).reshape(-1)
    num = pair_index.shape[0]
    return flat[pair_index.reshape(-1)].reshape(num, NUM_COLORS, 3).sum(-1)


def _connectivity(positions, pair_index):
    """(fraction of sampled colourings fully connected, partial credit)."""
    edges = _group_edges(positions, pair_index)
    whole = jnp.prod((edges >= 2.0).astype(jnp.float32), axis=1)
    partial = jnp.prod(jnp.minimum(edges, 2.0) * 0.5, axis=1)
    return whole.mean(), partial.mean()


def _board_connected(positions):
    """Is the whole card set orthogonally connected, ignoring colours?"""
    reach = _adjacency(positions) + jnp.eye(NUM_CARDS, dtype=jnp.float32)
    reach = jnp.minimum(reach @ reach, 1.0)
    reach = jnp.minimum(reach @ reach, 1.0)
    reach = jnp.minimum(reach @ reach, 1.0)
    return jnp.all(reach > 0.5)


def init_agent():
    return {
        'known': -jnp.ones(NUM_CARDS, dtype=jnp.int8),
        'calls': jnp.asarray(0, dtype=jnp.int32),
        'prev_value': jnp.asarray(0.0, dtype=jnp.float32),
        'have_prev': jnp.asarray(0.0, dtype=jnp.float32),
        'partner_quality': jnp.asarray(0.0, dtype=jnp.float32),
    }


def _sample_colourings(known, rng):
    """Draw balanced colourings uniformly from the seat's exact posterior."""
    known32 = known.astype(jnp.int32)
    unknown = known32 < 0
    num_unknown = unknown.sum()
    seen = (~unknown)[:, None] & (known32[:, None] == jnp.arange(NUM_COLORS)[None, :])
    residual = 3 - seen.sum(axis=0)
    bounds = jnp.cumsum(residual)
    slots = jnp.arange(NUM_CARDS)
    # Colour of the j-th unassigned slot, before shuffling.
    bag = jnp.minimum(jnp.sum(slots[:, None] >= bounds[None, :], axis=1), NUM_COLORS - 1)
    targets = jnp.argsort(jnp.where(unknown, 0, 1), stable=True)
    current = known32[targets]

    def one(key):
        noise = jax.random.uniform(key, (NUM_CARDS,)) + jnp.where(slots < num_unknown, 0.0, 10.0)
        shuffled = bag[jnp.argsort(noise)]
        return known32.at[targets].set(jnp.where(slots < num_unknown, shuffled, current))

    return jax.vmap(one)(jax.random.split(rng, NUM_SAMPLES))


def _safe(action, legal):
    action = action.astype(jnp.int32)
    return jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))


def _observe_action(known, locked, legal):
    unknown = (known < 0) & (locked < 0.5)
    score = (
        legal[1:1 + NUM_CARDS].astype(jnp.float32) * 1000.0
        + unknown.astype(jnp.float32) * 100.0
        - jnp.arange(NUM_CARDS, dtype=jnp.float32)
    )
    return 1 + jnp.argmax(score).astype(jnp.int32)


def _heuristic(positions, p_same, cell_xy):
    """Cheap ranking of every (card, destination) pair."""
    fpos = positions.astype(jnp.float32)

    def pull(dest):
        dist = jnp.abs(dest[:, None, :] - fpos[None, :, :]).sum(-1)
        return 2.0 * (dist == 1.0) - 0.15 * dist

    gain = (pull(cell_xy)[None, :, :] * p_same[:, None, :]).sum(-1)
    loss = (pull(fpos) * p_same).sum(-1)[:, None]
    centre = fpos.mean(axis=0)
    compact = -0.05 * jnp.abs(cell_xy - centre[None, :]).sum(-1)[None, :]
    return (gain - loss + compact).reshape(-1)


def _apply(positions, slot):
    cell = slot % NUM_CELLS
    return positions.at[slot // NUM_CELLS].set(
        jnp.stack([cell // GRID, cell % GRID]).astype(positions.dtype)
    )


def _move_action(positions, p_same, cell_xy, pair_index, locked, legal):
    heuristic = _heuristic(positions, p_same, cell_xy)
    legal_moves = legal[MOVE_BASE:MOVE_BASE + NUM_CARDS * NUM_CELLS]
    top_values, top_slots = jax.lax.top_k(jnp.where(legal_moves, heuristic, -1e9), TOP_K)

    def first_ply(carry, item):
        slot, hval = item
        moved = _apply(positions, slot)
        p_now, partial = _connectivity(moved, pair_index)

        # Value of the best single follow-up relocation from the new board.
        occupied = jnp.zeros(NUM_CELLS, dtype=jnp.float32).at[
            moved[:, 0] * GRID + moved[:, 1]
        ].set(1.0)
        blocked = jnp.maximum(
            jnp.tile(occupied[None, :], (NUM_CARDS, 1)), locked[:, None]
        ).reshape(-1)
        follow_heuristic = jnp.where(blocked > 0.5, -1e9, _heuristic(moved, p_same, cell_xy))
        _, follow_slots = jax.lax.top_k(follow_heuristic, TOP_K2)

        def follow(slot2):
            board = _apply(moved, slot2)
            p2, _ = _connectivity(board, pair_index)
            return jnp.where(_board_connected(board), p2, -1.0)

        p_next = jnp.max(jax.vmap(follow)(follow_slots))
        objective = (
            p_now
            + LOOKAHEAD * jnp.maximum(p_next, p_now)
            + W_PARTIAL * partial
            + 1e-3 * hval
        )
        objective = jnp.where(hval < -1e8, -1e9, objective)
        best_obj, best_slot = carry
        take = objective > best_obj
        return (jnp.where(take, objective, best_obj), jnp.where(take, slot, best_slot)), None

    init = (jnp.asarray(-1e30, dtype=jnp.float32), top_slots[0])
    (_, chosen), _ = jax.lax.scan(first_ply, init, (top_slots, top_values))
    return MOVE_BASE + chosen


def _hint_action(positions, known, q_card_hint, revealed, placed, locked, card_connected, legal):
    unknown_penalty = 1.5 * (known < 0).astype(jnp.float32)[:, None]
    unsettled_penalty = 1.0 * (1.0 - card_connected)[:, None]
    place_value = (2.0 * q_card_hint - 1.0) - 2.0 - unknown_penalty - unsettled_penalty
    placeable = (revealed * (1.0 - placed))[None, :] * (1.0 - locked)[:, None]
    place_value = jnp.where(placeable > 0.5, place_value, -1e9)

    cells = positions[:, 0] * GRID + positions[:, 1]
    place_actions = PLACE_BASE + jnp.arange(NUM_HINTS)[None, :] * NUM_CELLS + cells[:, None]
    place_value = jnp.where(legal[place_actions], place_value, -1e9)
    flat = place_value.reshape(-1)
    best_place = jnp.argmax(flat)

    reveal_value = jnp.where(legal[REVEAL_BASE:REVEAL_BASE + NUM_HINTS], -3.0, -1e9)
    best_reveal = jnp.argmax(reveal_value)
    take_place = flat[best_place] >= reveal_value[best_reveal]
    return jnp.where(
        take_place, place_actions.reshape(-1)[best_place], REVEAL_BASE + best_reveal
    ).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    colourings, cell_xy = _static_tables()

    legal = obs.legal_actions
    features = obs.node_features.astype(jnp.int32)
    positions = obs.positions.astype(jnp.int32)
    adjacency = obs.adjacency_matrix.astype(jnp.float32)

    visible = features[0:NUM_CARDS, 0:NUM_COLORS]
    seen_now = visible.sum(axis=1) > 0
    colour_now = jnp.argmax(visible, axis=1).astype(jnp.int8)
    previous = jnp.where(reset, -jnp.ones(NUM_CARDS, dtype=jnp.int8), memory['known'])
    known = jnp.where(seen_now, colour_now, previous)
    known32 = known.astype(jnp.int32)

    locked = features[0:NUM_CARDS, 8].astype(jnp.float32)
    hint_colours = features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 3:3 + NUM_COLORS].astype(jnp.float32)
    revealed = (hint_colours.sum(axis=1) > 0).astype(jnp.float32)
    placed = features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 8].astype(jnp.float32)
    active = features[0, 10] > 0
    substep = features[0, 11]

    # Exact posterior marginals: known cards are certain, unknown cards share
    # the residual colour multiset uniformly.
    onehot = (known32[:, None] == jnp.arange(NUM_COLORS)[None, :]).astype(jnp.float32)
    is_known = (known32 >= 0).astype(jnp.float32)
    residual = 3.0 - (onehot * is_known[:, None]).sum(axis=0)
    num_unknown = jnp.maximum(NUM_CARDS - is_known.sum(), 1.0)
    marginals = (
        is_known[:, None] * onehot + (1.0 - is_known)[:, None] * (residual / num_unknown)[None, :]
    )

    # Exact posterior over full colourings, used for the end decision.
    exact_pairs = _pair_index(colourings)
    consistent = jnp.all(
        (colourings == known32[None, :]) | (known32[None, :] < 0), axis=1
    ).astype(jnp.float32)
    consistent_sum = jnp.maximum(consistent.sum(), 1e-6)
    colour_whole = (_group_edges(positions, exact_pairs) >= 2.0).astype(jnp.float32)
    p_all = jnp.dot(consistent, jnp.prod(colour_whole, axis=1)) / consistent_sum
    card_connected = (
        consistent @ jnp.take_along_axis(colour_whole, colourings, axis=1)
    ) / consistent_sum

    # Value of ending now, assuming the board is connected.
    q_card_hint = marginals @ hint_colours.T
    hint_target = adjacency[NUM_CARDS:NUM_CARDS + NUM_HINTS, 0:NUM_CARDS]
    q_placed = (hint_target * q_card_hint.T).sum(axis=1)
    end_score = (
        5.0 * (NUM_HINTS - revealed.sum())
        + 2.0 * (revealed * (1.0 - placed)).sum()
        + (placed * (2.0 * q_placed - 1.0)).sum()
    )
    sample_pairs = _pair_index(_sample_colourings(known, rng))
    p_same = marginals @ marginals.T
    p_same = p_same * (1.0 - jnp.eye(NUM_CARDS, dtype=jnp.float32))

    # How much is the partner worth?  Our own knowledge does not change between
    # the end of our turn (substep 3) and our next substep 0, so the change in
    # board value across that gap measures the partner's relocation alone.
    board_p, board_partial = _connectivity(positions, sample_pairs)
    board_value = board_p + 0.5 * board_partial
    previous_value = jnp.where(reset, 0.0, memory['prev_value'])
    have_previous = jnp.where(reset, 0.0, memory['have_prev'])
    quality_old = jnp.where(reset, 0.0, memory['partner_quality'])
    turn_start = active & (substep == 0)
    quality = jnp.where(
        turn_start & (have_previous > 0.5),
        (1.0 - QUALITY_DECAY) * quality_old + QUALITY_DECAY * (board_value - previous_value),
        quality_old,
    )
    # A partner whose moves keep damaging the board makes playing on worth less.
    continue_value = CONTINUE_VALUE * jnp.clip(1.0 + PARTNER_SENSITIVITY * quality, 0.3, 1.0)
    should_end = (p_all * end_score > continue_value) & legal[END_ACTION]

    turn_end = active & (substep == 3)
    prev_value = jnp.where(turn_end, board_value, previous_value)
    have_prev = jnp.where(turn_end, 1.0, have_previous)

    observe = _observe_action(known, locked, legal)
    move = _move_action(positions, p_same, cell_xy, sample_pairs, locked, legal)
    hint = _hint_action(positions, known, q_card_hint, revealed, placed, locked,
                        card_connected, legal)

    action = jnp.where(
        substep == 0,
        jnp.where(should_end, jnp.asarray(END_ACTION, dtype=jnp.int32), observe),
        jnp.where(substep == 1, observe, jnp.where(substep == 2, move, hint)),
    )
    action = jnp.where(active, _safe(action, legal), jnp.asarray(0, dtype=jnp.int32))

    next_memory = {
        'known': known,
        'calls': jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory['calls']) + 1,
        'prev_value': prev_value.astype(jnp.float32),
        'have_prev': have_prev.astype(jnp.float32),
        'partner_quality': quality.astype(jnp.float32),
    }
    trace = {
        'p_connected': p_all.astype(jnp.float32),
        'num_known': jnp.sum(known >= 0).astype(jnp.int32),
        'end_score': end_score.astype(jnp.float32),
        'ended': should_end.astype(jnp.int32),
        'partner_quality': quality.astype(jnp.float32),
    }
    return action.astype(jnp.int32), next_memory, trace
