"""A local, inventory-aware policy for the two-player Yokai board."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'inventory_public_cluster', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    # known stores persistent observations.  The other fields are fixed-shape
    # private bookkeeping values.
    return (
        -jnp.ones((9,), dtype=jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
    )


def _public_colors(obs):
    """Decode only the three unambiguous outer encoding strips."""
    p = obs.positions
    x = p[:, 0]
    y = p[:, 1]
    top = (y == 2) & (x >= 3) & (x <= 5)
    bottom = (y == 6) & (x >= 3) & (x <= 5)
    left = (x == 2) & (y >= 3) & (y <= 5)
    return jnp.where(top, 0, jnp.where(bottom, 1, jnp.where(left, 2, -1)))


def _board_is_clustered(obs, known):
    """For three cards per color, two same-color edges are sufficient."""
    full = jnp.all(known >= 0)
    same = known[:, None] == known[None, :]
    edges = (obs.adjacency_matrix[:9, :9] > 0) & same
    edges = jnp.triu(edges, k=1)
    counts = jnp.asarray([
        jnp.sum(edges & ((known[:, None] == c) & (known[None, :] == c)))
        for c in range(3)
    ])
    return full & jnp.all(counts >= 2)


def _first_legal(mask):
    return jnp.argmax(mask).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng
    old_known, old_role, old_turn, old_first = memory
    reset = jnp.asarray(reset, dtype=jnp.bool_)
    nf = obs.node_features
    active = nf[0, 10] == 1
    substep = nf[0, 11].astype(jnp.int32)
    role = jnp.where(reset, jnp.where(active, 0, 1), old_role)
    known = jnp.where(reset, -jnp.ones((9,), dtype=jnp.int32), old_known)
    visible = jnp.sum(nf[:9, :3], axis=1) > 0
    colors = jnp.argmax(nf[:9, :3], axis=1).astype(jnp.int32)
    known = jnp.where(visible, colors, known)

    legal = obs.legal_actions
    card_ids = jnp.arange(9, dtype=jnp.int32)
    legal_observe = legal[1:10]
    unknown = known < 0
    first_mask = legal_observe & unknown
    first_id = jnp.where(jnp.any(first_mask), _first_legal(first_mask), _first_legal(legal_observe))
    second_mask = legal_observe & (card_ids != old_first)
    second_unknown = second_mask & unknown
    second_id = jnp.where(
        jnp.any(second_unknown), _first_legal(second_unknown), _first_legal(second_mask))

    # The deck contains exactly three of each color.  Once eight card colors
    # are known, fill in the last one from the inventory.
    counts = jnp.asarray([jnp.sum(known == c) for c in range(3)])
    missing_color = jnp.argmin(counts).astype(jnp.int32)
    known = jnp.where((jnp.sum(known >= 0) == 8) & (known < 0), missing_color, known)

    clustered = _board_is_clustered(obs, known)
    can_end = active & (substep == 0) & clustered & legal[1067]
    observe_action = 1 + jnp.where(substep == 0, first_id, second_id)
    move_offsets = jnp.arange(729, dtype=jnp.int32)
    move_card = move_offsets // 81
    move_rem = move_offsets % 81
    move_x = move_rem // 9
    move_y = move_rem % 9
    target_x = 3 + known[move_card]
    target_candidates = (
        legal[10:739] & (known[move_card] >= 0) &
        (move_x == target_x) & (obs.positions[move_card, 0] != target_x)
    )
    move_action = 10 + jnp.where(
        jnp.any(target_candidates), _first_legal(target_candidates), _first_legal(legal[10:739]))
    reveal_legal = legal[739:743]
    place_legal = legal[743:1067]
    place_offsets = jnp.arange(324, dtype=jnp.int32)
    place_h = place_offsets // 81
    place_rem = place_offsets % 81
    place_x = place_rem // 9
    place_y = place_rem % 9
    at_cell = (obs.positions[:, 0, None] == place_x[None, :]) & (obs.positions[:, 1, None] == place_y[None, :])
    place_card = jnp.argmax(at_cell, axis=0)
    place_color = known[place_card]
    place_sets = nf[9:13, 3:6][place_h]
    good_places = place_legal & (place_color >= 0) & (
        jnp.sum(place_sets * (jnp.arange(3)[None, :] == place_color[:, None]), axis=1) > 0)
    place_action = 743 + jnp.where(
        jnp.any(good_places), _first_legal(good_places), _first_legal(place_legal))
    hint_action = jnp.where(
        jnp.any(reveal_legal), 739 + _first_legal(reveal_legal), place_action)
    action = jnp.where(
        ~active,
        0,
        jnp.where(can_end, 1067, jnp.where(
            substep < 2, observe_action,
            jnp.where(substep == 2, move_action, hint_action))),
    ).astype(jnp.int32)
    action = jnp.where(legal[action], action, jnp.argmax(legal).astype(jnp.int32))
    next_first = jnp.where(active & (substep == 0), first_id, old_first)
    next_turn = old_turn + jnp.where(active & (substep == 3), 1, 0)
    return action, (known, role, next_turn, next_first), {}
