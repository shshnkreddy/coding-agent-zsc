"""Observation-driven, coordination-free heuristic for two-player 9-card Yokai,
with protective hint-locking for already-connected known cards.

Strategy (uses only public game rules, no private convention with a partner):
- Track own knowledge of card colors from the perfect-memory feature block
  that the environment re-exposes on every active turn, plus a logically
  certain elimination inference (two colors fully observed implies the
  third for all remaining cards).
- End the game only when this seat can prove, from public adjacency plus its
  own known colors, that all three colors are already fully connected.
- Otherwise, prefer observing not-yet-self-known cards (randomized among
  ties, to spread coverage across turns/episodes without relying on a
  partner sharing the same tie-break rule).
- On the mandatory move substep, greedily choose the legal move that most
  reduces Manhattan distance (and rewards direct adjacency) between pairs of
  cards whose color is known to this seat and equal, leaving unrelated
  pairs untouched; ties are broken with small random noise.
- On the mandatory hint substep: if a known card is already adjacent to a
  same-known-color neighbor and some revealed hint matches its color, place
  (lock) it immediately to protect that structure from being disturbed by
  future moves (by either seat). Otherwise prefer revealing an unrevealed
  hint over placing one. When forced to place with no protective option and
  no reveal available, place on a card whose known color is inside the
  hint's set, preferring one already known over one still unknown.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'connectivity_heuristic_protect',
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


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    nf = obs.node_features.astype(jnp.int32)
    adj = obs.adjacency_matrix.astype(jnp.int32)
    pos = obs.positions.astype(jnp.int32)
    legal = obs.legal_actions

    card_nf = nf[0:9]
    hint_nf = nf[9:13]

    color_onehot = card_nf[:, 0:3]
    known_mask = jnp.sum(color_onehot, axis=1) > 0
    color_id = jnp.argmax(color_onehot, axis=1)

    counts_known = jnp.sum(jnp.where(known_mask[:, None], color_onehot, 0), axis=0)
    remaining_needed = 3 - counts_known
    num_unknown = 9 - jnp.sum(known_mask.astype(jnp.int32))
    is_sole_remaining = (remaining_needed == num_unknown) & (num_unknown > 0)
    inferred_color_exists = jnp.any(is_sole_remaining)
    inferred_color = jnp.argmax(is_sole_remaining)

    directly_known = jnp.sum(color_onehot, axis=1) > 0
    color_id = jnp.where(directly_known, color_id, jnp.where(inferred_color_exists, inferred_color, color_id))
    known_mask = directly_known | inferred_color_exists

    substep = card_nf[0, 11]
    active = card_nf[0, 10]

    off_diag = adj[0:9, 0:9] * (1 - jnp.eye(9, dtype=jnp.int32))

    def color_connected(c):
        mask_c = known_mask & (color_id == c)
        mask_c_i = mask_c.astype(jnp.int32)
        sub_sum = jnp.sum(off_diag * mask_c_i[:, None] * mask_c_i[None, :])
        full = jnp.sum(mask_c_i) == 3
        return full & (sub_sum >= 4)

    connected_all = jax.vmap(color_connected)(jnp.arange(3))
    full_verified_connected = jnp.all(connected_all)

    observe_legal = legal[1:10]
    rng, observe_rng = jax.random.split(rng)
    unknown_logits = jnp.where(observe_legal & (~known_mask), 0.0, -1e9)
    have_unknown = jnp.any(observe_legal & (~known_mask))
    fallback_logits = jnp.where(observe_legal, 0.0, -1e9)
    observe_logits = jnp.where(have_unknown, unknown_logits, fallback_logits)
    observe_idx = jax.random.categorical(observe_rng, observe_logits)
    observe_action = (1 + observe_idx).astype(jnp.int32)

    end_ok = full_verified_connected & legal[1067]
    action_s0 = jnp.where(end_ok, jnp.asarray(1067, jnp.int32), observe_action)
    action_s1 = observe_action

    manhattan_old = (jnp.abs(pos[:, None, 0] - pos[None, :, 0])
                      + jnp.abs(pos[:, None, 1] - pos[None, :, 1]))
    adjacent_old = (manhattan_old == 1).astype(jnp.float32)
    metric_old = -manhattan_old.astype(jnp.float32) + 5.0 * adjacent_old
    same_color_mask = (known_mask[:, None] & known_mask[None, :]
                        & (color_id[:, None] == color_id[None, :])
                        & (~jnp.eye(9, dtype=bool)))
    old_contrib = jnp.sum(jnp.where(same_color_mask, metric_old, 0.0), axis=1)

    xs = jnp.arange(9)
    X, Y = jnp.meshgrid(xs, xs, indexing='ij')
    dist_bxy = (jnp.abs(X[None, :, :] - pos[:, 0][:, None, None])
                + jnp.abs(Y[None, :, :] - pos[:, 1][:, None, None]))
    adj_bxy = (dist_bxy == 1).astype(jnp.float32)
    metric_bxy = -dist_bxy.astype(jnp.float32) + 5.0 * adj_bxy

    new_contrib = jnp.einsum('ib,bxy->ixy', same_color_mask.astype(jnp.float32), metric_bxy)
    move_score = new_contrib - old_contrib[:, None, None]

    disruption = -0.001 * (jnp.abs(X[None, :, :] - pos[:, 0][:, None, None])
                            + jnp.abs(Y[None, :, :] - pos[:, 1][:, None, None]))
    rng, move_rng = jax.random.split(rng)
    tie_noise = jax.random.uniform(move_rng, (9, 9, 9), minval=0.0, maxval=1e-3)

    move_score_total = move_score + disruption.astype(jnp.float32) + tie_noise

    legal_move_grid = legal[10:739].reshape(9, 9, 9)
    move_final = jnp.where(legal_move_grid, move_score_total, -1e9)
    move_flat_idx = jnp.argmax(move_final.reshape(-1))
    action_s2 = (10 + move_flat_idx).astype(jnp.int32)

    reveal_legal = legal[739:743]
    reveal_any = jnp.any(reveal_legal)
    reveal_h = jnp.argmax(reveal_legal)
    reveal_action = (739 + reveal_h).astype(jnp.int32)

    place_legal = legal[743:1067].reshape(4, 9, 9)
    hint_sets = hint_nf[:, 3:6] > 0

    card_index_grid = jnp.full((9, 9), 0, dtype=jnp.int32)
    card_index_grid = card_index_grid.at[pos[:, 0], pos[:, 1]].set(jnp.arange(9, dtype=jnp.int32))
    known_at = known_mask[card_index_grid]
    color_at = color_id[card_index_grid]
    matched = hint_sets[:, color_at]

    already_connected_degree = jnp.sum(off_diag * same_color_mask.astype(jnp.int32), axis=1)
    degree_at = already_connected_degree[card_index_grid]
    safety_bonus = 0.1 * degree_at.astype(jnp.float32)
    base_score = jnp.where(known_at[None, :, :], jnp.where(matched, 2.0, 0.5), -1.0)
    score_grid = base_score + safety_bonus[None, :, :]
    place_final = jnp.where(place_legal, score_grid, -1e9)
    place_flat_idx = jnp.argmax(place_final.reshape(-1))
    place_action = (743 + place_flat_idx).astype(jnp.int32)

    protect_worthy = known_at[None, :, :] & matched & (degree_at[None, :, :] >= 1)
    protect_legal = place_legal & protect_worthy
    protect_any = jnp.any(protect_legal)
    protect_final = jnp.where(protect_legal, score_grid, -1e9)
    protect_flat_idx = jnp.argmax(protect_final.reshape(-1))
    protect_action = (743 + protect_flat_idx).astype(jnp.int32)

    action_s3 = jnp.where(protect_any, protect_action,
                           jnp.where(reveal_any, reveal_action, place_action))

    actions_stack = jnp.stack([action_s0, action_s1, action_s2, action_s3])
    chosen = actions_stack[substep]
    chosen = jnp.where(active > 0, chosen, jnp.asarray(0, jnp.int32))
    chosen = jnp.where(legal[chosen], chosen, jnp.argmax(legal).astype(jnp.int32))

    zero = jnp.asarray(0, dtype=jnp.int32)
    next_memory = jnp.where(reset, zero, memory)

    return chosen.astype(jnp.int32), next_memory, {}
