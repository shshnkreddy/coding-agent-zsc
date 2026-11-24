"""A small, local-memory heuristic for the two-player Yokai board."""

import jax.numpy as jnp


POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'four_turn_color_memory',
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
    # known is -1 for a card whose private color has not been observed.
    return (
        jnp.full((9,), -1, dtype=jnp.int8),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((9, 2), dtype=jnp.int32),
    )


def _first_legal(legal, preferred):
    fallback = jnp.argmax(legal).astype(jnp.int32)
    return jnp.where(legal[preferred], preferred.astype(jnp.int32), fallback)


def _move_action(obs, known):
    actions = jnp.arange(1068, dtype=jnp.int32)
    move = (actions >= 10) & (actions < 739) & obs.legal_actions
    raw = jnp.maximum(actions - 10, 0)
    card = raw // 81
    rem = raw % 81
    x = rem // 9
    y = rem % 9
    moved_color = known[card]
    moved_known = moved_color >= 0
    code_match = (y % 3) == moved_color

    dx = jnp.abs(obs.positions[:, 0][None, :] - x[:, None])
    dy = jnp.abs(obs.positions[:, 1][None, :] - y[:, None])
    adjacent = (dx + dy) == 1
    same = adjacent & (known[None, :] == moved_color[:, None])
    same = same & (known[None, :] >= 0)
    same = same & (jnp.arange(9, dtype=jnp.int32)[None, :] != card[:, None])
    destination_count = same.astype(jnp.int32).sum(axis=1)

    source_dx = jnp.abs(obs.positions[:, 0][card, None] - obs.positions[:, 0][None, :])
    source_dy = jnp.abs(obs.positions[:, 1][card, None] - obs.positions[:, 1][None, :])
    source_same = (source_dx + source_dy) == 1
    source_same = source_same & (known[card, None] == known[None, :])
    source_same = source_same & (known[None, :] >= 0)
    source_same = source_same & (jnp.arange(9, dtype=jnp.int32)[None, :] != card[:, None])
    source_count = source_same.astype(jnp.int32).sum(axis=1)
    delta = destination_count - source_count

    score = (
        moved_known.astype(jnp.int32) * 1000
        + code_match.astype(jnp.int32) * moved_known.astype(jnp.int32) * 500
        + delta * 30
        + destination_count * 5
        - card
        - x
        - y
    )
    score = jnp.where(move, score, jnp.asarray(-1000000, dtype=jnp.int32))
    return jnp.argmax(score).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    old_known, old_role, old_positions = memory
    active = obs.node_features[0, 10].astype(jnp.int32)
    visible = obs.node_features[:9, :3].sum(axis=1) > 0
    visible_color = jnp.argmax(obs.node_features[:9, :3], axis=1).astype(jnp.int8)
    base_known = jnp.where(reset, jnp.full((9,), -1, dtype=jnp.int8), old_known)
    known = jnp.where(visible, visible_color, base_known)
    role = jnp.where(reset, active, old_role)
    previous_positions = jnp.where(reset, obs.positions, old_positions)
    substep = obs.node_features[0, 11].astype(jnp.int32)
    changed = (obs.positions != previous_positions).any(axis=1)
    moved_id = jnp.argmax(changed).astype(jnp.int32)
    decoded_color = (obs.positions[moved_id, 1] % 3).astype(jnp.int8)
    moved_card = jnp.arange(9, dtype=jnp.int32) == moved_id
    decode_mask = changed[moved_id] & (substep == 3) & moved_card
    known = jnp.where(decode_mask, decoded_color, known)

    revealed = (obs.node_features[9:13, 3:6].sum(axis=1) > 0).astype(jnp.int32)
    num_revealed = revealed.sum()
    turn = num_revealed
    pair_start = jnp.minimum(turn * 2, 6).astype(jnp.int32)
    first_card = jnp.where(turn >= 2, 4, pair_start)
    second_card = jnp.where(turn >= 2, 8, pair_start + 1)
    preferred_first = (first_card + 1).astype(jnp.int32)
    preferred_second = (second_card + 1).astype(jnp.int32)

    observe_first = _first_legal(obs.legal_actions, preferred_first)
    observe_second = _first_legal(obs.legal_actions, preferred_second)
    move_action = _move_action(obs, known)
    hint_ids = jnp.arange(4, dtype=jnp.int32) + 739
    hint_legal = obs.legal_actions[739:743]
    hint_fallback = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    reveal_action = jnp.where(
        hint_legal.any(),
        hint_ids[jnp.argmax(hint_legal)],
        hint_fallback,
    )

    end_action = jnp.asarray(1067, dtype=jnp.int32)
    action = jnp.where(substep == 0, observe_first, move_action)
    action = jnp.where(substep == 1, observe_second, action)
    action = jnp.where(substep == 3, reveal_action, action)
    action = jnp.where((substep == 0) & (num_revealed >= 3), end_action, action)
    action = jnp.where(active == 1, action, jnp.asarray(0, dtype=jnp.int32))
    return action, (known, role, obs.positions), {}
