"""A local, convention-light solver for the nine-card Yōkai board.

The policy uses the three central columns as color homes.  A card whose color
we have personally seen is moved into its color's column; keeping all moves
inside the initial 3x3 footprint preserves board connectivity.  Colors are
remembered privately because they disappear from observations while the seat
is inactive.
"""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 3, 'deck_size': 9, 'grid_size': 9, 'num_hints': 4, 'num_nodes': 13, 'node_features': 13, 'num_actions': 1068, 'policy_structure': 'shared_local'}


def init_agent():
    # 0 means unknown; colors are stored as 1, 2, or 3.
    return jnp.zeros((9,), dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    del rng

    features = obs.node_features
    card_features = features[:9]
    positions = obs.positions
    legal = obs.legal_actions

    # The active seat sees all colors it has learned so far.  During inactive
    # calls those columns are masked, so the private memory carries them.
    visible = jnp.sum(card_features[:, :3], axis=1) > 0
    visible_color = jnp.argmax(card_features[:, :3], axis=1).astype(jnp.int32) + 1
    remembered = jnp.where(visible, visible_color, memory)
    next_memory = jnp.where(reset, jnp.zeros_like(memory), remembered)

    substep = features[0, 11].astype(jnp.int32)
    unlocked = card_features[:, 8] == 0
    known = next_memory > 0

    # A solved board is certified when every known card is in its color home,
    # at most one card remains unknown, and the three home columns are full.
    home_x = next_memory + 2
    known_in_home = (~known) | (positions[:, 0] == home_x)
    footprint = (positions[:, 0] >= 3) & (positions[:, 0] <= 5) & (
        positions[:, 1] >= 3) & (positions[:, 1] <= 5)
    column_counts = jnp.stack([
        jnp.sum(positions[:, 0] == 3),
        jnp.sum(positions[:, 0] == 4),
        jnp.sum(positions[:, 0] == 5),
    ])
    solved = (
        jnp.all(footprint)
        & jnp.all(known_in_home)
        & jnp.all(column_counts == 3)
    )

    # Substep 0/1: inspect the first two cards not yet known to this seat.
    observe_legal = legal[1:10]
    new_card = observe_legal & (~known)
    partner_seen = card_features[:, 9] > 0
    complementary = new_card & (~partner_seen)
    observe_pool = jnp.where(jnp.any(complementary), complementary,
                              jnp.where(jnp.any(new_card), new_card, observe_legal))
    observe_index = jnp.argmax(observe_pool)
    observe_action = (observe_index + 1).astype(jnp.int32)

    # Substep 2: select a legal move of a known, unlocked card into an empty
    # cell of that card's home column.  The mask is fixed-size and the native
    # legal-action mask handles occupancy and connectivity.
    move_offsets = jnp.arange(729, dtype=jnp.int32)
    move_card = move_offsets // 81
    move_cell = move_offsets % 81
    move_x = move_cell // 9
    move_y = move_cell % 9
    needs_move = known & unlocked & (positions[:, 0] != home_x)
    move_candidates = (
        legal[10:739]
        & needs_move[move_card]
        & (move_x == home_x[move_card])
    )
    any_move = jnp.any(move_candidates)
    move_offset = jnp.argmax(jnp.where(any_move, move_candidates, legal[10:739]))
    move_action = (move_offset + 10).astype(jnp.int32)

    # Substep 3 must reveal or place a hint.  Reveal all available hints first
    # (unrevealed hints are worth more than revealed ones).  Once revelation is
    # exhausted, place only a known-correct hint on a card already in its home
    # column, avoiding needless damage to the board arrangement.
    reveal_legal = legal[739:743]
    reveal_action = (jnp.argmax(reveal_legal) + 739).astype(jnp.int32)

    place_offsets = jnp.arange(324, dtype=jnp.int32)
    place_hint = place_offsets // 81
    place_cell = place_offsets % 81
    place_x = place_cell // 9
    place_y = place_cell % 9
    at_cell = (positions[:, 0][None, :] == place_x[:, None]) & (
        positions[:, 1][None, :] == place_y[:, None]
    )
    place_card = jnp.argmax(at_cell, axis=1)
    hint_colors = features[9:13, 3:6]
    place_known_correct = (
        known[place_card]
        & unlocked[place_card]
        & (positions[place_card, 0] == home_x[place_card])
        & (hint_colors[place_hint, next_memory[place_card] - 1] > 0)
    )
    place_candidates = legal[743:1067] & place_known_correct
    any_place = jnp.any(place_candidates)
    place_offset = jnp.argmax(jnp.where(any_place, place_candidates, legal[743:1067]))
    place_action = (place_offset + 743).astype(jnp.int32)

    # Inactive seats have only action 0 legal, so the final legality guard also
    # keeps all branch computations harmless when this policy is called there.
    desired = jnp.where(
        substep == 0,
        jnp.where(solved, jnp.asarray(1067, dtype=jnp.int32), observe_action),
        jnp.where(substep == 1, observe_action,
                  jnp.where(substep == 2, move_action,
                            jnp.where(jnp.any(reveal_legal), reveal_action,
                                      place_action))),
    )
    safe_action = jnp.argmax(jnp.where(legal[desired], legal, False)).astype(jnp.int32)
    # If desired is legal, retain it; otherwise choose the first legal action.
    action = jnp.where(legal[desired], desired, safe_action).astype(jnp.int32)
    return action, next_memory, {}
