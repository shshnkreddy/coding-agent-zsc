"""Heuristic cooperative policy for the two-player 16-card Yokai grid game.

Design summary (see policy_ledger A01):
  - Own-card colors we have ever personally observed are exposed directly by
    the environment via node_features[:16, 0:4] whenever this seat is active,
    so no private memory is needed to remember them.
  - Once 3 of the 4 colors are fully identified (4 cards each), the last
    color's cards are deduced by elimination without observing them.
  - substep 0/1 (observe): prefer the lowest-index unknown, unlocked card.
  - substep 2 (move): greedily move a known-color card to the legal empty
    cell maximizing new same-color orthogonal neighbors minus the same-color
    adjacency lost at its old cell; unknown-color cards score 0 (neutral),
    so the greedy pick never actively harms a cluster we cannot evaluate.
  - substep 0 (end game): only ever end once every card's color is known
    (observed or deduced) AND every color's 4 cards are verified orthogonally
    connected using the public adjacency matrix restricted to same-color
    pairs. This can never trigger a false positive.
  - substep 3 (hints): a forced action every turn. Prefer placing a hint we
    can *guarantee* correct (revealed hint whose color set contains a known
    card's true color) since that only costs 1 point relative to leaving it
    revealed-unplaced. Otherwise reveal a fresh unrevealed hint (costs 3
    points but preserves optionality). Only as a last resort (all 7 hints
    already revealed, no guaranteed match) place blindly, preferring an
    unlocked card whose color we do NOT know (avoids a guaranteed-wrong
    placement).
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'informed_single_mover',
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
NUM_HINTS = 7
GRID = 10
NUM_ACTIONS = 2325
NO_OP_VALUE = 0
END_GAME_VALUE = 2324
NEIGHBOR_OFFSET_LIST = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def _own_color_ids(node_features):
    """[16] int32 in {-1,0,1,2,3}; fills in the 4th color by elimination."""
    card_rows = node_features[:NUM_CARDS]
    onehot = card_rows[:, 0:NUM_COLORS]
    has_color = jnp.any(onehot > 0, axis=1)
    observed_id = jnp.argmax(onehot, axis=1).astype(jnp.int32)
    color_id = jnp.where(has_color, observed_id, -1)

    counts = jnp.sum(jnp.where(has_color[:, None], onehot > 0, False).astype(jnp.int32), axis=0)
    full_colors = counts >= 4
    num_full = jnp.sum(full_colors.astype(jnp.int32))
    deduced_color = jnp.argmin(jnp.where(full_colors, jnp.int32(1000), counts)).astype(jnp.int32)
    can_deduce = num_full == (NUM_COLORS - 1)
    color_id = jnp.where((color_id == -1) & can_deduce, deduced_color, color_id)
    return color_id


def _connectivity_ok(color_id, adjacency_matrix):
    """True iff every card's color is known and each color's 4 cards form
    one orthogonally connected group under the public card adjacency."""
    all_known = jnp.all(color_id >= 0)
    card_adj = adjacency_matrix[:NUM_CARDS, :NUM_CARDS] > 0

    def color_connected(c):
        nodes = color_id == c
        same = card_adj & nodes[:, None] & nodes[None, :]
        start = jnp.argmax(nodes)
        reach = jax.nn.one_hot(start, NUM_CARDS, dtype=jnp.bool_)

        def body(_, reach):
            spread = jnp.any(same & reach[None, :], axis=1)
            return reach | spread

        reach = jax.lax.fori_loop(0, NUM_CARDS, body, reach)
        return jnp.all(jnp.where(nodes, reach, True))

    all_colors_connected = jnp.all(jax.vmap(color_connected)(jnp.arange(NUM_COLORS)))
    return all_known & all_colors_connected


def _pick_observe(node_features, legal_actions, color_id, adjacency_matrix, rng):
    """Pick a random legal unknown-color card (random, not lowest-index, so
    that the two independently-seeded seats tend to explore complementary
    cards instead of both deterministically re-observing 0,1,2,... first).

    A deterministic seat-dependent split (e.g. seat 0 explores card IDs
    0-7 first, seat 1 explores 8-15 first) was tried and scored worse: since
    a color's 4 cards are randomly scattered across all 16 stable IDs, such
    a split systematically causes each seat to learn only *some* members of
    a given color, so the two seats independently start separate same-color
    clusters that never merge. Unbiased randomness avoids this."""
    del adjacency_matrix
    legal_observe = legal_actions[1:1 + NUM_CARDS]
    unknown = color_id == -1
    noise = jax.random.uniform(rng, shape=(NUM_CARDS,), maxval=0.5)
    priority = jnp.where(legal_observe & unknown, 1.0, jnp.where(legal_observe, 0.0, -jnp.inf)) + noise
    best = jnp.argmax(priority)
    action = jnp.asarray(1, dtype=jnp.int32) + best.astype(jnp.int32)
    any_legal = jnp.any(legal_observe)
    return jnp.where(any_legal, action, jnp.asarray(NO_OP_VALUE, dtype=jnp.int32))


def _pick_move(node_features, positions, legal_actions, color_id, rng):
    grid_owner = -jnp.ones((GRID, GRID), dtype=jnp.int32)
    grid_owner = grid_owner.at[positions[:, 0], positions[:, 1]].set(jnp.arange(NUM_CARDS, dtype=jnp.int32))
    grid_color = -jnp.ones((GRID, GRID), dtype=jnp.int32)
    grid_color = grid_color.at[positions[:, 0], positions[:, 1]].set(color_id)

    xs = jnp.arange(GRID)
    ys = jnp.arange(GRID)

    def neighbor_same_color_count(x, y, card):
        my_color = color_id[card]

        def one_offset(offset):
            nx = x + offset[0]
            ny = y + offset[1]
            in_bounds = (nx >= 0) & (nx < GRID) & (ny >= 0) & (ny < GRID)
            nx_c = jnp.clip(nx, 0, GRID - 1)
            ny_c = jnp.clip(ny, 0, GRID - 1)
            owner = grid_owner[nx_c, ny_c]
            occupied = in_bounds & (owner != -1) & (owner != card)
            same = occupied & (grid_color[nx_c, ny_c] == my_color) & (my_color != -1)
            return same.astype(jnp.int32)

        return jnp.sum(jax.vmap(one_offset)(jnp.asarray(NEIGHBOR_OFFSET_LIST, dtype=jnp.int32)))

    loss = jax.vmap(lambda c: neighbor_same_color_count(positions[c, 0], positions[c, 1], c))(jnp.arange(NUM_CARDS))

    def gain_grid(card):
        def cell(x, y):
            return neighbor_same_color_count(x, y, card)

        return jax.vmap(lambda x: jax.vmap(lambda y: cell(x, y))(ys))(xs)

    gain = jax.vmap(gain_grid)(jnp.arange(NUM_CARDS))  # [16,10,10]
    score = (gain - loss[:, None, None]).astype(jnp.float32)

    noise = jax.random.uniform(rng, shape=(NUM_CARDS, GRID, GRID), maxval=0.2)
    score = score + noise

    card_idx = jnp.arange(NUM_CARDS)[:, None, None]
    x_idx = jnp.arange(GRID)[None, :, None]
    y_idx = jnp.arange(GRID)[None, None, :]
    action_ids = 17 + card_idx * 100 + x_idx * 10 + y_idx
    action_ids = jnp.broadcast_to(action_ids, (NUM_CARDS, GRID, GRID))

    legal_move = legal_actions[17:1617].reshape(NUM_CARDS, GRID, GRID)
    score = jnp.where(legal_move, score, -jnp.inf)

    flat_best = jnp.argmax(score.reshape(-1))
    chosen_action = action_ids.reshape(-1)[flat_best]
    any_legal = jnp.any(legal_move)
    return jnp.where(any_legal, chosen_action.astype(jnp.int32), jnp.asarray(NO_OP_VALUE, dtype=jnp.int32))


def _pick_hint(node_features, legal_actions, color_id, connected):
    hint_rows = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS]
    hint_colorset = hint_rows[:, 4:8] > 0
    hint_placed = hint_rows[:, 10] > 0
    hint_revealed = jnp.any(hint_colorset, axis=1)

    unlocked = node_features[:NUM_CARDS, 10] <= 0
    known = color_id >= 0

    color_onehot_by_card = jax.nn.one_hot(jnp.clip(color_id, 0, NUM_COLORS - 1), NUM_COLORS, dtype=jnp.bool_)
    match = jnp.einsum('hc,kc->hk', hint_colorset.astype(jnp.int32), color_onehot_by_card.astype(jnp.int32)) > 0
    match = match & known[None, :] & unlocked[None, :] & hint_revealed[:, None] & (~hint_placed)[:, None]

    has_confident = jnp.any(match)
    flat_confident = jnp.argmax(match.reshape(-1))
    h_confident = flat_confident // NUM_CARDS
    c_confident = flat_confident % NUM_CARDS

    hint_unrevealed = (~hint_revealed) & (~hint_placed)
    has_unrevealed = jnp.any(hint_unrevealed)
    h_reveal = jnp.argmax(hint_unrevealed)

    placeable_hint = hint_revealed & (~hint_placed)
    unknown_unlocked = unlocked & (~known)
    any_unknown_unlocked_card = jnp.any(unknown_unlocked)
    c_fallback_unknown = jnp.argmax(unknown_unlocked)
    c_fallback_any = jnp.argmax(unlocked)
    c_fallback = jnp.where(any_unknown_unlocked_card, c_fallback_unknown, c_fallback_any)
    h_fallback = jnp.argmax(placeable_hint)

    reveal_action = jnp.asarray(1617, dtype=jnp.int32) + h_reveal.astype(jnp.int32)

    positions = node_features[:NUM_CARDS, 8:10]

    def place_action(h, c):
        px = positions[c, 0].astype(jnp.int32)
        py = positions[c, 1].astype(jnp.int32)
        return jnp.asarray(1624, dtype=jnp.int32) + h.astype(jnp.int32) * 100 + px * 10 + py

    confident_action = place_action(h_confident, c_confident)
    fallback_place_action = place_action(h_fallback, c_fallback)

    # Placing a hint permanently LOCKS that card, so it can never move again.
    # A lock on a card whose neighbourhood is not yet finished could make
    # full color-connectivity permanently unreachable, which is far more
    # costly (zeroes the whole terminal reward) than the few points a hint
    # placement saves versus a plain reveal. So while the board is not yet
    # verified connected, always prefer revealing a fresh hint (deferring
    # any lock-in) over placing, even a confidently-correct one; only place
    # once genuinely forced (no unrevealed hints remain). Once the board is
    # already verified connected we are about to end the episode anyway, so
    # a confident placement now is safe and strictly better than a reveal.
    action_while_unsettled = jnp.where(has_unrevealed, reveal_action, jnp.where(has_confident, confident_action, fallback_place_action))
    action_while_connected = jnp.where(has_confident, confident_action, jnp.where(has_unrevealed, reveal_action, fallback_place_action))
    action = jnp.where(connected, action_while_connected, action_while_unsettled)

    legal = legal_actions[action]
    fallback_legal_reveal = legal_actions[1617:1624]
    any_reveal_legal = jnp.any(fallback_legal_reveal)
    safe_reveal = jnp.asarray(1617, dtype=jnp.int32) + jnp.argmax(fallback_legal_reveal).astype(jnp.int32)
    action = jnp.where(legal, action, jnp.where(any_reveal_legal, safe_reveal, action))
    return action


def agent_act(obs, reset, memory, rng):
    node_features = obs.node_features
    positions = obs.positions
    legal_actions = obs.legal_actions
    adjacency_matrix = obs.adjacency_matrix

    is_active = node_features[0, 12] > 0
    substep = node_features[0, 13].astype(jnp.int32)

    color_id = _own_color_ids(node_features)
    connected = _connectivity_ok(color_id, adjacency_matrix)

    observe_action = _pick_observe(node_features, legal_actions, color_id, adjacency_matrix, rng)
    move_action = _pick_move(node_features, positions, legal_actions, color_id, rng)
    hint_action = _pick_hint(node_features, legal_actions, color_id, connected)

    end_game = jnp.asarray(END_GAME_VALUE, dtype=jnp.int32)
    substep0_action = jnp.where(connected & legal_actions[end_game], end_game, observe_action)

    active_action = jax.lax.switch(
        jnp.clip(substep, 0, 3),
        [
            lambda: substep0_action,
            lambda: observe_action,
            lambda: move_action,
            lambda: hint_action,
        ],
    )

    fallback_action = jnp.argmax(legal_actions).astype(jnp.int32)
    active_action = jnp.where(legal_actions[active_action], active_action, fallback_action)

    action = jnp.where(is_active, active_action, jnp.asarray(NO_OP_VALUE, dtype=jnp.int32)).astype(jnp.int32)

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)

    return action, next_memory, {}
