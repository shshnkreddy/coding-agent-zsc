"""Public-geometry clustering policy for the two-player Yokai board."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {
    'name': 'public_geometry_cluster', 'adapter': 'yokai', 'adapter_version': 1,
    'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10,
    'num_hints': 7, 'num_nodes': 23, 'node_features': 15,
    'num_actions': 2325, 'policy_structure': 'shared_local',
}

# Four connected color regions: a central 4x4 polyomino with its four
# corners replaced by outside cells.  Each color occupies one connected arm.
TARGET_CELLS = ((4, 3, 0), (3, 4, 0), (4, 4, 0), (2, 4, 0),
                (5, 3, 1), (5, 4, 1), (6, 4, 1), (7, 4, 1),
                (3, 5, 2), (4, 5, 2), (4, 6, 2), (2, 5, 2),
                (5, 5, 3), (6, 5, 3), (5, 6, 3), (7, 5, 3))
PARK_CELLS = ((2, 3), (7, 3), (2, 6), (7, 6))


def init_agent():
    # Colors and own observations are private; partner-seen is not used as
    # substitute knowledge.
    return (jnp.full((16,), -1, dtype=jnp.int32),
            jnp.zeros((16,), dtype=jnp.bool_),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False, dtype=jnp.bool_))


def _first_legal(mask):
    return jnp.argmax(mask).astype(jnp.int32)


def agent_act(obs, reset, memory, rng):
    old_colors, old_seen, old_last, old_role, old_mode = memory
    zero_colors, zero_seen, zero_last, zero_role, zero_mode = init_agent()
    colors0 = jnp.where(reset, zero_colors, old_colors)
    seen0 = jnp.where(reset, zero_seen, old_seen)
    last0 = jnp.where(reset, zero_last, old_last)
    mode0 = jnp.where(reset, zero_mode, old_mode)

    target_color = jnp.full((10, 10), -1, dtype=jnp.int32)
    for x, y, color in TARGET_CELLS:
        target_color = target_color.at[x, y].set(color)
    park_x = jnp.asarray([p[0] for p in PARK_CELLS], dtype=jnp.int32)
    park_y = jnp.asarray([p[1] for p in PARK_CELLS], dtype=jnp.int32)
    grid_x = jnp.arange(10, dtype=jnp.int32)[:, None]
    grid_y = jnp.arange(10, dtype=jnp.int32)[None, :]
    parking = jnp.any(
        (grid_x == park_x[:, None, None]) & (grid_y == park_y[:, None, None]), axis=0)

    features = obs.node_features[:16, :]
    visible = jnp.sum(features[:, :4], axis=1) > 0
    visible_colors = jnp.argmax(features[:, :4], axis=1).astype(jnp.int32)
    colors = jnp.where(visible, visible_colors, colors0)
    seen = seen0 | visible

    substep = obs.node_features[0, 13].astype(jnp.int32)
    active = obs.node_features[0, 12] > 0
    role0 = jnp.where(reset, active.astype(jnp.int32), old_role)
    role = jnp.where(reset, role0, old_role)
    legal = obs.legal_actions
    positions = obs.positions
    locked = features[:, 10] > 0
    position_colors = target_color[positions[:, 0], positions[:, 1]]
    at_parking = parking[positions[:, 0], positions[:, 1]]
    correct = seen & (position_colors == colors) & (~locked | (position_colors >= 0))
    # Occupancy is public.  A private seat may not have seen all cards, but a
    # full target polyomino is nevertheless a safe completion signal when
    # coupled to this policy's public color geometry.
    all_target = jnp.all(position_colors >= 0)

    ids = jnp.arange(16, dtype=jnp.int32)
    # The initially active seat (role>0) is the coverage seat; the other
    # seat handles the upper half unless public evidence requires expansion.
    own_ids = jnp.where(role > 0, jnp.ones((16,), dtype=jnp.bool_), ids >= 8)
    partner_seen = features[:, 11] > 0
    unexpected_partner = jnp.any(partner_seen & own_ids)
    mode = mode0 | unexpected_partner
    scope = jnp.where(mode, jnp.ones((16,), dtype=jnp.bool_), own_ids)
    unknown = (~seen) & scope
    first_mask = legal[1:17] & unknown
    first_id = jnp.argmax(jnp.where(
        first_mask, 16 - ids, -jnp.ones((16,), dtype=jnp.int32)))
    first_action = (first_id + 1).astype(jnp.int32)
    fallback_observe = _first_legal(legal)
    observe_action = jnp.where(jnp.any(first_mask), first_action, fallback_observe)
    end_action = jnp.asarray(2324, dtype=jnp.int32)
    step0_action = jnp.where(all_target & legal[2324], end_action, observe_action)

    second_mask = legal[1:17] & unknown & (ids != last0)
    second_id = jnp.argmax(jnp.where(
        second_mask, 16 - ids, -jnp.ones((16,), dtype=jnp.int32)))
    second_action = (second_id + 1).astype(jnp.int32)
    second_fallback_mask = legal[1:17] & (ids != last0)
    second_fallback = _first_legal(second_fallback_mask) + 1
    step1_action = jnp.where(jnp.any(second_mask), second_action, second_fallback)

    # Move a known card to a vacant cell of its assigned region.  The legal
    # mask handles both vacancies and whole-board connectivity.
    move_legal = legal[17:1617].reshape((16, 10, 10))
    eligible = seen & (position_colors != colors) & (~locked) & scope
    target_for_card = target_color[None, :, :] == colors[:, None, None]
    target_candidates = move_legal & eligible[:, None, None] & target_for_card
    target_flat = target_candidates.reshape((1600,))
    target_choice = jnp.argmax(target_flat).astype(jnp.int32)
    have_target_move = jnp.any(target_flat)

    # A vacant initial corner can break a blocked permutation while retaining
    # a legal connected board.
    park_candidates = move_legal & (eligible & (~at_parking))[:, None, None] & parking[None, :, :]
    park_flat = park_candidates.reshape((1600,))
    park_choice = jnp.argmax(park_flat).astype(jnp.int32)
    move_fallback = _first_legal(legal)
    chosen_move = jnp.where(
        have_target_move, target_choice + 17,
        jnp.where(jnp.any(park_flat), park_choice + 17, move_fallback))

    # Resolve hints only as required by substep 3.  Reveal first (unrevealed
    # hints retain more terminal value), then prefer a compatible placement.
    reveal_legal = legal[1617:1624]
    reveal_choice = _first_legal(reveal_legal) + 1617
    hint_sets = obs.node_features[16:23, 4:8] > 0
    card_valid = seen & (~locked)
    cell_color = jnp.full((10, 10), -1, dtype=jnp.int32)
    cell_color = cell_color.at[positions[:, 0], positions[:, 1]].set(
        jnp.where(seen, colors, -1))
    cell_valid = jnp.zeros((10, 10), dtype=jnp.bool_)
    cell_valid = cell_valid.at[positions[:, 0], positions[:, 1]].set(card_valid)
    safe_cell_color = jnp.maximum(cell_color, 0)
    compatible = hint_sets[:, safe_cell_color]
    place_legal = legal[1624:2324].reshape((7, 10, 10))
    place_candidates = place_legal & cell_valid[None, :, :] & compatible
    place_flat = place_candidates.reshape((700,))
    place_choice = _first_legal(place_flat) + 1624
    place_fallback = _first_legal(legal)
    step3_action = jnp.where(
        jnp.any(reveal_legal), reveal_choice,
        jnp.where(jnp.any(place_flat), place_choice, place_fallback))

    action = jnp.where(substep == 0, step0_action,
                       jnp.where(substep == 1, step1_action,
                                 jnp.where(substep == 2, chosen_move,
                                           step3_action)))
    action = jnp.where(active, action, jnp.asarray(0, dtype=jnp.int32))
    next_last = jnp.where(substep == 0, first_id, last0)
    return action.astype(jnp.int32), (colors, seen, next_last, role, mode), {}
