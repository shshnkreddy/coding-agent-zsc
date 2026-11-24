"""Cooperative Yokai policy: private color memory + heuristic sort-and-stop play.

Strategy rationale: terminal score is 5*U + 2*R + C - I (if all colors are
connected, else 0). Leaving hints unrevealed (U) always dominates revealing or
placing them, so the priority is to privately track known card colors across
a seat's own turns, use that knowledge to observe informatively and move
same-color cards together, and end the game voluntarily the moment full
connectivity can be verified -- before any hint has to be touched. Every
completed turn forces one hint action (reveal preferred over place, since
revealing is worth strictly more than placing), so minimizing turns matters.
This relies only on locally observable public/private evidence, not on a
private signaling convention with the partner.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'color_memory_heuristic',
    'adapter': 'yokai',
    'adapter_version': 1,
    'num_agents': 2,
    'num_colors': 3,
    'deck_size': 9,
    'grid_size': 9,
    'num_hints': 4,
    'num_nodes': 13,
    'node_features': 13,
    'policy_structure': 'shared_local',
    'num_actions': 1068,
}

NUM_CARDS = 9
NUM_HINTS = 4
NUM_COLORS = 3
GRID = 9

_DIRS_LIST = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def _dirs():
    return jnp.array(_DIRS_LIST, dtype=jnp.int32)


def _init_known():
    return jnp.full((NUM_CARDS,), -1, dtype=jnp.int8)


def init_agent():
    return {"known": _init_known(), "is_p0": jnp.asarray(-1, dtype=jnp.int8)}


def _update_known(known, node_features):
    onehot = node_features[:NUM_CARDS, 0:3].astype(jnp.int32)
    has_color = jnp.sum(onehot, axis=-1) > 0
    new_color = jnp.argmax(onehot, axis=-1).astype(jnp.int8)
    return jnp.where(has_color, new_color, known)


def _transitive_closure(reach):
    def step(r, _):
        ri = r.astype(jnp.int32)
        r2 = (ri @ ri) > 0
        return r | r2, None

    reach, _ = jax.lax.scan(step, reach, xs=None, length=4)
    return reach


def _connected_all_colors(known, adj_no_self):
    same_color_adj = adj_no_self & (known[None, :] == known[:, None]) & (known[None, :] >= 0)
    reach = same_color_adj | jnp.eye(NUM_CARDS, dtype=jnp.bool_)
    reach = _transitive_closure(reach)
    connected = jnp.asarray(True)
    for c in range(NUM_COLORS):
        mask_c = known == c
        ref = jnp.argmax(mask_c)
        ok = jnp.all(jnp.where(mask_c, reach[ref], True))
        connected = connected & ok
    return connected


def _move_intended(known, positions, legal_actions, adj_no_self, is_p0):
    i_idx = jnp.arange(NUM_CARDS)
    pos = positions.astype(jnp.int32)

    # A known card already touching a same-known-color neighbor is "settled":
    # moving it risks breaking a good cluster, so it is deprioritized as mover.
    has_same_neighbor = jnp.any(
        adj_no_self & (known[None, :] >= 0) & (known[None, :] == known[:, None]), axis=1
    )
    unknown_i = known == -1

    # Pair-join candidates: move card i next to a same-known-color card j.
    target = pos[:, None, :] + _dirs()[None, :, :]  # (j, d, 2)
    tx, ty = target[..., 0], target[..., 1]
    valid = (tx >= 0) & (tx < GRID) & (ty >= 0) & (ty < GRID)
    txc, tyc = jnp.clip(tx, 0, GRID - 1), jnp.clip(ty, 0, GRID - 1)

    i_b = i_idx[:, None, None]
    tx_b = txc[None, :, :]
    ty_b = tyc[None, :, :]
    valid_b = valid[None, :, :]
    action_id = 10 + i_b * 81 + tx_b * 9 + ty_b  # (i, j, d)
    legal = legal_actions[action_id]

    color_i = known[i_idx][:, None, None]
    color_j = known[i_idx][None, :, None]
    same_color = (color_i >= 0) & (color_j >= 0) & (color_i == color_j)
    i_ne_j = i_idx[:, None, None] != i_idx[None, :, None]
    adj_b = adj_no_self[:, :, None]
    not_adj_yet = ~adj_b
    isolated_i = (~has_same_neighbor)[:, None, None]
    # Anchor j already part of a pair -> this move would complete a full trio.
    anchor_forms_trio = has_same_neighbor[i_idx][None, :, None]

    join_ok = valid_b & legal & same_color & i_ne_j & not_adj_yet
    tier = jnp.where(
        isolated_i & anchor_forms_trio, 5.0,
        jnp.where(isolated_i, 4.0, jnp.where(anchor_forms_trio, 3.0, 2.0)),
    )
    pair_tie_break = jnp.where(is_p0 > 0, 0.0, jnp.arange(NUM_CARDS, dtype=jnp.float32)) * 1e-3
    score_pair = jnp.where(join_ok, tier + pair_tie_break[:, None, None], -jnp.inf)
    flat = jnp.argmax(score_pair.reshape(-1))
    best_pair_score = score_pair.reshape(-1)[flat]
    best_pair_action = action_id.reshape(-1)[flat]

    # Nudge fallback: move card i to an adjacent free cell of its own position.
    # Prefer moving cards with unknown color (protects already-verified structure).
    target2 = pos[:, None, :] + _dirs()[None, :, :]  # (i, d, 2)
    tx2, ty2 = target2[..., 0], target2[..., 1]
    valid2 = (tx2 >= 0) & (tx2 < GRID) & (ty2 >= 0) & (ty2 < GRID)
    tx2c, ty2c = jnp.clip(tx2, 0, GRID - 1), jnp.clip(ty2, 0, GRID - 1)
    action_id2 = 10 + i_idx[:, None] * 81 + tx2c * 9 + ty2c
    legal2 = legal_actions[action_id2]
    nudge_pref = jnp.where(unknown_i, 2.0, jnp.where(has_same_neighbor, 0.5, 1.0))[:, None]
    tie_break = jnp.where(is_p0 > 0, 0.0, jnp.arange(NUM_CARDS, dtype=jnp.float32)) * 1e-3
    score_nudge = jnp.where(valid2 & legal2, nudge_pref + tie_break[:, None], -jnp.inf)
    flat2 = jnp.argmax(score_nudge.reshape(-1))
    best_nudge_score = score_nudge.reshape(-1)[flat2]
    best_nudge_action = action_id2.reshape(-1)[flat2]

    move_slice = legal_actions[10:739]
    abs_fallback = 10 + jnp.argmax(move_slice.astype(jnp.int32))

    intended = jnp.where(
        best_pair_score > -jnp.inf, best_pair_action,
        jnp.where(best_nudge_score > -jnp.inf, best_nudge_action, abs_fallback),
    )
    return intended.astype(jnp.int32)


def _deduce(known):
    """Elimination: colors are always exactly 3-per-color (public rule), so if
    only one color still has open slots, every remaining unknown card must be
    that color -- certain knowledge, not an observation or a convention."""
    counts = jnp.stack([jnp.sum(known == c) for c in range(NUM_COLORS)])
    deficient = counts < 3
    num_deficient = jnp.sum(deficient.astype(jnp.int32))
    target_color = jnp.argmax(deficient.astype(jnp.int32)).astype(jnp.int8)
    unknown_mask = known == -1
    return jnp.where(unknown_mask & (num_deficient == 1), target_color, known)


def _observe_intended(known, legal_actions, adj_no_self, is_p0):
    unknown = known == -1
    has_same_neighbor = jnp.any(
        adj_no_self & (known[None, :] >= 0) & (known[None, :] == known[:, None]), axis=1
    )
    # Neighbor already part of a forming pair -> observing this card could
    # confirm/complete a full trio for free.
    pair_neighbor = jnp.any(adj_no_self & (known[None, :] >= 0) & has_same_neighbor[None, :], axis=1)
    known_neighbor = jnp.any(adj_no_self & (known[None, :] >= 0), axis=1)
    score = jnp.where(
        unknown & pair_neighbor, 4.0,
        jnp.where(unknown & known_neighbor, 3.0, jnp.where(unknown, 2.0, 1.0)),
    )
    # Both seats start from identical blank memory and would otherwise break
    # ties the same (low-index-first) way, causing redundant overlap in what
    # each seat personally learns. Player 1 unilaterally reverses its own tie
    # preference to high-index-first, diversifying team-wide coverage -- this
    # requires nothing from the partner and is harmless if it does not help.
    tie_break = jnp.where(is_p0 > 0, 0.0, jnp.arange(NUM_CARDS, dtype=jnp.float32)) * 1e-3
    score = score + tie_break
    legal_obs = legal_actions[1 + jnp.arange(NUM_CARDS)]
    score = jnp.where(legal_obs, score, -jnp.inf)
    return (1 + jnp.argmax(score)).astype(jnp.int32)


def _per_color_connected(known, adj_no_self):
    """For each color: True iff all 3 members are known and mutually
    connected via same-color adjacency (i.e. that color's group is already
    finished and none of its cards should ever need to move again)."""
    same_color_adj = adj_no_self & (known[None, :] == known[:, None]) & (known[None, :] >= 0)
    reach = same_color_adj | jnp.eye(NUM_CARDS, dtype=jnp.bool_)
    reach = _transitive_closure(reach)
    results = []
    for c in range(NUM_COLORS):
        mask_c = known == c
        ref = jnp.argmax(mask_c)
        ok = jnp.all(jnp.where(mask_c, reach[ref], True)) & (jnp.sum(mask_c) == 3)
        results.append(ok)
    return jnp.stack(results)


def _hint_intended(known, node_features, positions, legal_actions, adj_no_self):
    revealed = jnp.any(node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 3:6] != 0, axis=-1)
    placed = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 8] == 1
    colorsets = node_features[NUM_CARDS:NUM_CARDS + NUM_HINTS, 3:6] > 0

    unrevealed = ~revealed
    reveal_legal = legal_actions[739 + jnp.arange(NUM_HINTS)]
    reveal_score = jnp.where(unrevealed & reveal_legal, 1.0, -jnp.inf)
    best_reveal_idx = jnp.argmax(reveal_score)
    best_reveal_score = reveal_score[best_reveal_idx]
    reveal_action = 739 + best_reveal_idx

    pos = positions.astype(jnp.int32)
    x_i, y_i = pos[:, 0], pos[:, 1]
    h_idx = jnp.arange(NUM_HINTS)
    action_place = 743 + h_idx[:, None] * 81 + x_i[None, :] * 9 + y_i[None, :]
    legal_place = legal_actions[action_place]
    revealed_unplaced_h = (revealed & ~placed)[:, None]

    known_i = (known >= 0)[None, :]
    onehot_known = jax.nn.one_hot(jnp.clip(known, 0, NUM_COLORS - 1), NUM_COLORS)
    match_matrix = (colorsets.astype(jnp.float32) @ onehot_known.T) > 0

    # A card is "safe to lock" only if its whole color group is already
    # verified connected -- placing on it can never later block a needed
    # move. Placing on an unsettled (still-movable) card risks locking it
    # out of position and forcing the whole board's terminal reward to 0,
    # which dwarfs the hint-score arithmetic.
    per_color_connected = _per_color_connected(known, adj_no_self)
    safe_color = per_color_connected[jnp.clip(known, 0, NUM_COLORS - 1)][None, :]

    place_score = jnp.where(
        revealed_unplaced_h & legal_place,
        jnp.where(
            known_i,
            jnp.where(match_matrix, jnp.where(safe_color, 3.0, 1.0), -1.0),
            0.0,
        ),
        -jnp.inf,
    )
    flat = jnp.argmax(place_score.reshape(-1))
    best_place_score = place_score.reshape(-1)[flat]
    best_place_action = action_place.reshape(-1)[flat]

    # Score math: U=5,R=2,C=+1,I=-1. Revealing a new hint always costs -3
    # (U->R). Placing a hint on a card whose color group is already fully
    # settled and matches costs only -1 (R->C) with zero risk, strictly
    # better than revealing. Any other placement is at best as costly as
    # revealing (and risks locking an unsettled card), so it is only taken
    # when forced (no reveal available).
    confident_safe_place = best_place_score >= 2.5
    intended = jnp.where(
        confident_safe_place, best_place_action,
        jnp.where(best_reveal_score > -jnp.inf, reveal_action, best_place_action),
    )
    return intended.astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    known_in = jnp.where(reset, _init_known(), memory["known"])
    known = _update_known(known_in, obs.node_features)

    active_now = (obs.node_features[0, 10] > 0).astype(jnp.int8)
    is_p0_in = jnp.where(reset, jnp.asarray(-1, dtype=jnp.int8), memory["is_p0"])
    is_p0 = jnp.where(is_p0_in == -1, active_now, is_p0_in)

    adj = obs.adjacency_matrix[:NUM_CARDS, :NUM_CARDS] > 0
    adj_no_self = adj & ~jnp.eye(NUM_CARDS, dtype=jnp.bool_)

    substep = obs.node_features[0, 11].astype(jnp.int32)
    believed = _deduce(known)

    observe_action = _observe_intended(believed, obs.legal_actions, adj_no_self, is_p0)
    known_all = jnp.all(believed != -1)
    connected_all = _connected_all_colors(believed, adj_no_self)
    end_ready = known_all & connected_all & obs.legal_actions[1067]
    substep0_action = jnp.where(end_ready, jnp.int32(1067), observe_action)

    move_action = _move_intended(believed, obs.positions, obs.legal_actions, adj_no_self, is_p0)
    hint_action = _hint_intended(believed, obs.node_features, obs.positions, obs.legal_actions, adj_no_self)

    intended = jnp.where(
        substep == 0, substep0_action,
        jnp.where(substep == 1, observe_action,
                  jnp.where(substep == 2, move_action, hint_action)),
    ).astype(jnp.int32)

    legal_mask = obs.legal_actions
    fallback = jnp.argmax(legal_mask.astype(jnp.int32)).astype(jnp.int32)
    final_action = jnp.where(legal_mask[intended], intended, fallback).astype(jnp.int32)

    trace = {
        "substep": substep,
        "known_count": jnp.sum(believed != -1).astype(jnp.int32),
        "intended_action": intended,
        "final_action": final_action,
    }

    next_memory = {"known": known, "is_p0": is_p0}
    return final_action, next_memory, trace
