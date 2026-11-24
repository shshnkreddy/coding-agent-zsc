"""Convention-light Yokai policy.

The policy uses color-id rows as the public convention: cards believed to be
color c are moved toward row y=3+c.  A card newly moved into such a row is
treated as a public color signal, which lets independently written partners
using the same natural convention coordinate without private state sharing.
"""

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'row_sorter_v1', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    return {
        "known": jnp.full((9,), -1, dtype=jnp.int8),
        "prev_positions": jnp.zeros((9, 2), dtype=jnp.int32),
        "role": jnp.asarray(-1, dtype=jnp.int8),
        "own_turn": jnp.asarray(0, dtype=jnp.int8),
    }


def _first_true(mask):
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def _pair_upper():
    return jnp.triu(jnp.ones((9, 9), dtype=bool), k=1)


def _effective_colors(known):
    counts = jnp.stack([jnp.sum(known == 0), jnp.sum(known == 1), jnp.sum(known == 2)])
    missing_color = jnp.argmax((3 - counts).astype(jnp.int32)).astype(jnp.int8)
    one_unknown = jnp.sum(known < 0) == 1
    return jnp.where((known < 0) & one_unknown, missing_color, known)


def _solved(positions, known):
    colors = _effective_colors(known)
    all_known = jnp.all(colors >= 0)
    dx = jnp.abs(positions[:, None, 0] - positions[None, :, 0])
    dy = jnp.abs(positions[:, None, 1] - positions[None, :, 1])
    adjacent = (dx + dy) == 1
    same = colors[:, None] == colors[None, :]
    valid = (colors[:, None] >= 0) & (colors[None, :] >= 0)
    pair_edges = adjacent & same & valid & _pair_upper()
    edges = jnp.stack([
        jnp.sum(pair_edges & (colors[:, None] == 0)),
        jnp.sum(pair_edges & (colors[:, None] == 1)),
        jnp.sum(pair_edges & (colors[:, None] == 2)),
    ])
    return all_known & jnp.all(edges >= 2)


def _public_row_solved(positions, known):
    x = positions[:, 0]
    y = positions[:, 1]
    known_mask = known >= 0
    enough_known = jnp.sum(known_mask.astype(jnp.int32)) >= 7
    in_target_square = (x >= 3) & (x <= 5) & (y >= 3) & (y <= 5)
    row_counts = jnp.stack([
        jnp.sum((y == 3).astype(jnp.int32)),
        jnp.sum((y == 4).astype(jnp.int32)),
        jnp.sum((y == 5).astype(jnp.int32)),
    ])
    known_rows_match = jnp.all(jnp.where(known_mask, y == (3 + known.astype(jnp.int32)), True))
    return enough_known & jnp.all(in_target_square) & jnp.all(row_counts == 3) & known_rows_match


def _choose_observe(obs, known, role, own_turn, substep):
    obs_schedule = jnp.asarray(
        [[0, 1, 4, 5, 8, 2, 3, 6],
         [2, 3, 6, 7, 8, 0, 1, 4]],
        dtype=jnp.int32,
    )
    legal = obs.legal_actions[1:10]
    obs_index = jnp.clip(own_turn.astype(jnp.int32) * 2 + (substep == 1).astype(jnp.int32), 0, 7)
    safe_role = jnp.clip(role.astype(jnp.int32), 0, 1)
    scheduled = obs_schedule[safe_role, obs_index]
    scheduled_ok = legal[scheduled] & (known[scheduled] < 0)

    unknown_legal = legal & (known < 0)
    any_unknown = jnp.any(unknown_legal)
    fallback_unknown = _first_true(unknown_legal)
    fallback_any = _first_true(legal)
    chosen = jnp.where(
        scheduled_ok,
        scheduled,
        jnp.where(any_unknown, fallback_unknown, fallback_any),
    )
    any_obs = jnp.any(legal)
    return jnp.where(any_obs, 1 + chosen, _first_true(obs.legal_actions)).astype(jnp.int32)


def _move_scores(positions, colors):
    action_ix = jnp.arange(729, dtype=jnp.int32)
    card = action_ix // 81
    rem = action_ix - card * 81
    dest_x = rem // 9
    dest_y = rem - dest_x * 9

    current = jnp.broadcast_to(positions, (729, 9, 2))
    dest = jnp.stack([dest_x, dest_y], axis=1)[:, None, :]
    moved_card = jnp.arange(9, dtype=jnp.int32)[None, :, None] == card[:, None, None]
    new_positions = jnp.where(moved_card, dest, current)

    valid_color = colors >= 0
    home_row = 3 + colors.astype(jnp.int32)
    new_x = new_positions[:, :, 0]
    new_y = new_positions[:, :, 1]
    in_home_band = (new_y == home_row[None, :]) & (new_x >= 3) & (new_x <= 5) & valid_color[None, :]
    home_count = jnp.sum(in_home_band.astype(jnp.int32), axis=1)

    dx = jnp.abs(new_positions[:, :, None, 0] - new_positions[:, None, :, 0])
    dy = jnp.abs(new_positions[:, :, None, 1] - new_positions[:, None, :, 1])
    adjacent = (dx + dy) == 1
    same_color = colors[None, :, None] == colors[None, None, :]
    valid_pair = valid_color[None, :, None] & valid_color[None, None, :]
    pair_edges = adjacent & same_color & valid_pair & _pair_upper()[None, :, :]
    same_edges = jnp.sum(pair_edges.astype(jnp.int32), axis=(1, 2))
    color_edges = jnp.stack([
        jnp.sum((pair_edges & (colors[None, :, None] == 0)).astype(jnp.int32), axis=(1, 2)),
        jnp.sum((pair_edges & (colors[None, :, None] == 1)).astype(jnp.int32), axis=(1, 2)),
        jnp.sum((pair_edges & (colors[None, :, None] == 2)).astype(jnp.int32), axis=(1, 2)),
    ], axis=1)
    complete_clusters = jnp.all(colors >= 0) & jnp.all(color_edges >= 2, axis=1)
    capped_color_edges = jnp.sum(jnp.minimum(color_edges, 2), axis=1)

    target_band = valid_color[None, :] & (new_x >= 3) & (new_x <= 5) & (new_y >= 3) & (new_y <= 5)
    wrong_band_count = jnp.sum((target_band & (new_y != home_row[None, :])).astype(jnp.int32), axis=1)

    distance = jnp.where(
        valid_color[None, :],
        jnp.abs(new_y - home_row[None, :]) + jnp.abs(new_x - 4),
        0,
    )
    dist_sum = jnp.sum(distance.astype(jnp.int32), axis=1)

    moved_color = colors[card]
    moved_valid = moved_color >= 0
    moved_home = moved_valid & (dest_y == (3 + moved_color.astype(jnp.int32))) & (dest_x >= 3) & (dest_x <= 5)
    cur_x = positions[card, 0]
    cur_y = positions[card, 1]
    current_home = moved_valid & (cur_y == (3 + moved_color.astype(jnp.int32))) & (cur_x >= 3) & (cur_x <= 5)
    moved_wrong = moved_valid & (~current_home)

    return (
        25 * moved_valid.astype(jnp.int32)
        + 45 * moved_wrong.astype(jnp.int32)
        + 90 * moved_home.astype(jnp.int32)
        + 24 * home_count
        + 15 * same_edges
        + 45 * capped_color_edges
        + 4000 * complete_clusters.astype(jnp.int32)
        - dist_sum
        - 35 * wrong_band_count
        - 90 * (moved_valid & current_home & (~moved_home)).astype(jnp.int32)
        - 35 * (~moved_valid).astype(jnp.int32)
    )


def _choose_move(obs, known):
    colors = _effective_colors(known)
    scores = _move_scores(obs.positions, colors)
    legal = obs.legal_actions[10:739]
    masked = jnp.where(legal, scores, jnp.asarray(-1_000_000, dtype=jnp.int32))
    return (10 + jnp.argmax(masked).astype(jnp.int32)).astype(jnp.int32)


def _best_place(obs, known):
    colors = _effective_colors(known)
    action_ix = jnp.arange(324, dtype=jnp.int32)
    hint = action_ix // 81
    rem = action_ix - hint * 81
    x = rem // 9
    y = rem - x * 9
    legal = obs.legal_actions[743:1067]

    at_target = (obs.positions[None, :, 0] == x[:, None]) & (obs.positions[None, :, 1] == y[:, None])
    card = jnp.argmax(at_target.astype(jnp.int32), axis=1)
    card_color = colors[card]
    known_card = card_color >= 0
    clipped_color = jnp.clip(card_color.astype(jnp.int32), 0, 2)
    hint_sets = obs.node_features[9:13, 3:6] > 0
    hint_size = jnp.sum(hint_sets.astype(jnp.int32), axis=1)
    match = known_card & hint_sets[hint, clipped_color]
    settled = known_card & (obs.positions[card, 1] == (3 + card_color.astype(jnp.int32)))
    single_hint = hint_size[hint] == 1
    score = (
        140 * (match & settled).astype(jnp.int32)
        + 80 * match.astype(jnp.int32)
        + 20 * settled.astype(jnp.int32)
        + 8 * single_hint.astype(jnp.int32)
        - 60 * (~known_card).astype(jnp.int32)
    )
    masked = jnp.where(legal, score, jnp.asarray(-1_000_000, dtype=jnp.int32))
    best_ix = jnp.argmax(masked).astype(jnp.int32)
    good = jnp.any(legal & match & settled)
    any_place = jnp.any(legal)
    return (743 + best_ix).astype(jnp.int32), good, any_place


def _choose_hint(obs, known):
    place_action, good_place, any_place = _best_place(obs, known)
    reveal_legal = obs.legal_actions[739:743]
    any_reveal = jnp.any(reveal_legal)
    reveal_action = (739 + _first_true(reveal_legal)).astype(jnp.int32)
    fallback = _first_true(obs.legal_actions)
    return jnp.where(
        good_place & any_place,
        place_action,
        jnp.where(any_reveal, reveal_action, jnp.where(any_place, place_action, fallback)),
    ).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng

    active = obs.node_features[0, 10] > 0
    substep = obs.node_features[0, 11].astype(jnp.int32)

    init = init_agent()
    known0 = jnp.where(reset, init["known"], memory["known"])
    prev_positions = jnp.where(reset, obs.positions, memory["prev_positions"])
    role0 = jnp.where(reset, jnp.where(active, jnp.asarray(0, dtype=jnp.int8), jnp.asarray(1, dtype=jnp.int8)), memory["role"])
    own_turn0 = jnp.where(reset, init["own_turn"], memory["own_turn"])

    color_bits = obs.node_features[:9, 0:3] > 0
    visible = jnp.any(color_bits, axis=1)
    visible_color = jnp.argmax(color_bits.astype(jnp.int32), axis=1).astype(jnp.int8)

    moved = jnp.any(obs.positions != prev_positions, axis=1)
    moved_row_color = (obs.positions[:, 1] - 3).astype(jnp.int8)
    row_signal = moved & (moved_row_color >= 0) & (moved_row_color <= 2)

    known1 = jnp.where(row_signal & (known0 < 0), moved_row_color, known0)
    known2 = jnp.where(visible, visible_color, known1)

    solved_belief = _solved(obs.positions, known2) | _public_row_solved(obs.positions, known2)
    end_now = obs.legal_actions[1067] & solved_belief
    observe0 = jnp.where(end_now, jnp.asarray(1067, dtype=jnp.int32), _choose_observe(obs, known2, role0, own_turn0, substep))
    observe1 = _choose_observe(obs, known2, role0, own_turn0, substep)
    move = _choose_move(obs, known2)
    hint = _choose_hint(obs, known2)
    active_action = jax.lax.switch(
        substep,
        (
            lambda _: observe0,
            lambda _: observe1,
            lambda _: move,
            lambda _: hint,
        ),
        operand=None,
    )
    action = jnp.where(active, active_action, jnp.asarray(0, dtype=jnp.int32)).astype(jnp.int32)

    own_turn1 = own_turn0 + (active & (substep == 3)).astype(jnp.int8)
    next_memory = {
        "known": known2.astype(jnp.int8),
        "prev_positions": obs.positions.astype(jnp.int32),
        "role": role0.astype(jnp.int8),
        "own_turn": own_turn1.astype(jnp.int8),
    }
    trace = {
        "known_count": jnp.sum((known2 >= 0).astype(jnp.int32)),
        "solved_belief": solved_belief,
    }
    return action, next_memory, trace
