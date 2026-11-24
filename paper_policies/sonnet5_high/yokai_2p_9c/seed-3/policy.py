"""Cooperative Yokai policy: local, convention-free greedy coordination.

Design goals (for robustness to independently generated partners):
- Never rely on a private convention (fixed target zones, hint codes, etc.).
- Use only publicly observable board state (positions, locks, revealed hint
  sets, the partner-observed flag) plus this seat's own accumulated private
  color knowledge (persisted across turns via ``memory``).
- Greedily improve same-color adjacency for whichever card is moved, using
  the *current* true board layout, so it self-corrects regardless of what a
  partner (running arbitrary code) has done.
- Only end the game when this seat's own knowledge (direct observation plus
  provably-correct elimination) can certify that every color is fully
  connected, which guarantees the ending is never based on a false belief.
- Prefer a confirmed-correct hint placement over revealing a new hint
  (placing correctly costs 1 point off the terminal score, revealing always
  costs 3), and only reveal when no confirmed-correct placement is available.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'greedy_local_cluster_v1',
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
NUM_COLORS = 3
NUM_HINTS = 4
GRID = 9

END_ACTION = 1067
OBSERVE_BASE = 1
MOVE_BASE = 10
REVEAL_BASE = 739
PLACE_BASE = 743

NEG_INF = -1e9


def init_agent():
    return {"known_color": jnp.full((NUM_CARDS,), -1, dtype=jnp.int8)}


def agent_act(obs, reset, memory, rng):
    del rng

    known_color = jnp.where(reset, jnp.full_like(memory["known_color"], -1), memory["known_color"])

    node_features = obs.node_features
    card_features = node_features[:NUM_CARDS]  # (9, 13)

    # --- merge this call's directly-observed colors into persistent memory ---
    color_onehot = card_features[:, 0:3]
    has_color_info = jnp.any(color_onehot > 0, axis=-1)
    observed_color = jnp.argmax(color_onehot, axis=-1).astype(jnp.int8)
    known_color = jnp.where(has_color_info, observed_color, known_color)
    new_memory = {"known_color": known_color}

    # --- elimination inference: if only one color still has a deficit, any
    # remaining unknown card(s) must be that color. Provably correct given
    # the fixed 3-per-color multiset, never a guess. ---
    color_idx = jnp.clip(known_color, 0, NUM_COLORS - 1)
    known_mask_f = (known_color >= 0).astype(jnp.float32)
    onehot_known = jax.nn.one_hot(color_idx, NUM_COLORS) * known_mask_f[:, None]
    counts = jnp.sum(onehot_known, axis=0)  # (3,)
    deficit = 3.0 - counts
    nonzero_deficit = deficit > 0.5
    num_deficit_colors = jnp.sum(nonzero_deficit.astype(jnp.int32))
    single_deficit_color = jnp.argmax(nonzero_deficit).astype(jnp.int8)
    unknown_mask = known_color < 0
    inferable = (jnp.sum(unknown_mask.astype(jnp.int32)) > 0) & (num_deficit_colors == 1)
    inferred_color = jnp.where(unknown_mask & inferable, single_deficit_color, known_color)

    fully_known = jnp.all(inferred_color >= 0)

    # --- geometry ---
    pos = obs.positions.astype(jnp.int32)  # (9, 2)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = jnp.sum(jnp.abs(diff), axis=-1)  # (9, 9)
    adjacent = dist == 1

    # --- connectivity check per color (safe: undercounts, never overcounts) ---
    color_mask = (inferred_color[:, None] == jnp.arange(NUM_COLORS)[None, :]) & (inferred_color[:, None] >= 0)
    color_mask_f = color_mask.astype(jnp.float32)
    adjacent_f = adjacent.astype(jnp.float32)
    edge_count_per_color = jnp.einsum('ic,jc,ij->c', color_mask_f, color_mask_f, adjacent_f)
    all_connected = jnp.all(edge_count_per_color >= 4.0)
    should_end = fully_known & all_connected & obs.legal_actions[END_ACTION]

    # --- per-card same-known-color adjacency at the *current* position.
    # A card with a positive count already sits next to a same-color card
    # it is not going anywhere useful by moving away from, and locking it
    # with a hint placement costs nothing in future flexibility. ---
    same_color_ij = (inferred_color[:, None] == inferred_color[None, :]) & (inferred_color[:, None] >= 0)
    eye = jnp.eye(NUM_CARDS, dtype=bool)
    combined = (same_color_ij & (~eye)).astype(jnp.float32)  # (i, j)
    current_same_color_neighbors = jnp.sum(combined * adjacent_f, axis=1)  # (i,)
    settled = current_same_color_neighbors > 0.0  # (9,)

    # --- observe action: prefer new info for the team, then new info for
    # me, then any legal observation. Uses only public + own-private info.
    # Within the top tier, prefer cards adjacent to an already-known card:
    # if it turns out to match, that is a same-color edge gained for free,
    # without spending any of the scarce move actions. ---
    legal_observe = obs.legal_actions[OBSERVE_BASE:OBSERVE_BASE + NUM_CARDS]
    partner_seen = card_features[:, 9] > 0
    unknown_to_me = inferred_color < 0
    known_mask_for_adjacency = inferred_color >= 0
    adjacent_known_count = jnp.sum(adjacent.astype(jnp.int32) * known_mask_for_adjacency[None, :], axis=1)
    observe_tier = jnp.where(unknown_to_me, jnp.where(partner_seen, 2, 3), 1)
    observe_priority = jnp.where(legal_observe, observe_tier * 10 + adjacent_known_count, -1)
    observe_action = OBSERVE_BASE + jnp.argmax(observe_priority).astype(jnp.int32)

    # --- move action: greedily maximize the moved card's same-known-color
    # adjacency at the destination versus its current position. Purely
    # reactive to the current true board, so it self-corrects regardless of
    # partner behavior. ---
    grid_x = jnp.arange(GRID)
    grid_y = jnp.arange(GRID)
    X, Y = jnp.meshgrid(grid_x, grid_y, indexing='ij')
    target = jnp.stack([X, Y], axis=-1).astype(jnp.int32)  # (9, 9, 2)

    new_dist = jnp.sum(jnp.abs(target[:, :, None, :] - pos[None, None, :, :]), axis=-1)  # (x, y, j)
    new_adjacent_j = (new_dist == 1).astype(jnp.float32)  # (x, y, j)

    new_adjacent_count = jnp.einsum('ij,xyj->ixy', combined, new_adjacent_j)  # (i, x, y)
    improvement = new_adjacent_count - current_same_color_neighbors[:, None, None]

    legal_moves = obs.legal_actions[MOVE_BASE:MOVE_BASE + NUM_CARDS * GRID * GRID].reshape(NUM_CARDS, GRID, GRID)
    move_score = jnp.where(legal_moves, improvement, NEG_INF)
    move_flat_idx = jnp.argmax(move_score.reshape(-1)).astype(jnp.int32)
    move_action = MOVE_BASE + move_flat_idx

    # --- hint action. Scoring (5*unrevealed + 2*revealed-unplaced +
    # correct - incorrect) means revealing an unrevealed hint always costs
    # 3 points (5 -> 2), while placing an already-revealed hint on a card
    # we know matches its color set costs only 1 point (2 -> 1). So a
    # confirmed-correct placement is strictly cheaper than revealing, and
    # should be preferred; revealing is only the better of the two mandatory
    # options when no confirmed-correct placement is available (keeping the
    # option open to place correctly later, once more colors are known,
    # rather than gambling on an uncertain placement now). ---
    legal_reveal = obs.legal_actions[REVEAL_BASE:REVEAL_BASE + NUM_HINTS]
    any_reveal_legal = jnp.any(legal_reveal)
    reveal_action = REVEAL_BASE + jnp.argmax(legal_reveal).astype(jnp.int32)

    hint_colorset = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 3:6]  # (4, 3)
    cell_match = (pos[:, 0:1] == X.reshape(1, -1)) & (pos[:, 1:2] == Y.reshape(1, -1))  # (9, 81)
    occupant_color = jnp.sum(jnp.where(cell_match, inferred_color[:, None].astype(jnp.int32), 0), axis=0).reshape(GRID, GRID)
    occupant_known = (jnp.sum(jnp.where(cell_match, (inferred_color[:, None] >= 0).astype(jnp.int32), 0), axis=0)
                       .reshape(GRID, GRID) > 0)
    occupant_settled = (jnp.sum(jnp.where(cell_match, settled[:, None].astype(jnp.int32), 0), axis=0)
                         .reshape(GRID, GRID) > 0)
    occupant_color_clip = jnp.clip(occupant_color, 0, NUM_COLORS - 1)
    occupant_onehot = jax.nn.one_hot(occupant_color_clip, NUM_COLORS)  # (x, y, 3)
    member_grid = jnp.einsum('hc,xyc->hxy', hint_colorset.astype(jnp.float32), occupant_onehot)
    place_score = jnp.where(occupant_known[None, :, :], jnp.where(member_grid > 0, 1.0, -1.0), 0.0)

    legal_place = obs.legal_actions[PLACE_BASE:PLACE_BASE + NUM_HINTS * GRID * GRID].reshape(NUM_HINTS, GRID, GRID)
    place_score = jnp.where(legal_place, place_score, NEG_INF)
    place_flat_idx = jnp.argmax(place_score.reshape(-1)).astype(jnp.int32)
    place_action = PLACE_BASE + place_flat_idx

    # Only treat a placement as "confirmed" (worth doing ahead of a reveal)
    # when it is both correct AND targets an already-settled card, so we
    # never sacrifice future move flexibility on a card that still needs to
    # relocate. Recompute the argmax restricted to settled+correct cells.
    settled_correct_score = jnp.where(
        legal_place & occupant_known[None, :, :] & occupant_settled[None, :, :],
        jnp.where(member_grid > 0, 1.0, -1.0),
        NEG_INF,
    )
    best_settled_correct = jnp.max(settled_correct_score)
    have_confirmed_place = best_settled_correct >= 1.0
    settled_place_flat_idx = jnp.argmax(settled_correct_score.reshape(-1)).astype(jnp.int32)
    settled_place_action = PLACE_BASE + settled_place_flat_idx

    hint_action = jnp.where(
        have_confirmed_place, settled_place_action,
        jnp.where(any_reveal_legal, reveal_action, place_action),
    )

    # --- substep dispatch ---
    substep = card_features[0, 11].astype(jnp.int32)
    active = card_features[0, 10] > 0

    substep0_action = jnp.where(should_end, END_ACTION, observe_action)
    action = jnp.where(
        substep == 0, substep0_action,
        jnp.where(substep == 1, observe_action,
                  jnp.where(substep == 2, move_action, hint_action)),
    )
    action = jnp.where(active, action, jnp.int32(0)).astype(jnp.int32)

    trace = {
        "should_end": should_end,
        "fully_known": fully_known,
        "num_known": jnp.sum((inferred_color >= 0).astype(jnp.int32)),
    }

    return action, new_memory, trace
