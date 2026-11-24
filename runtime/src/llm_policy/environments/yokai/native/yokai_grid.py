"""Yokai Environment where actions happen in a predefined grid.
"""
from typing import Tuple, Dict
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct

from llm_policy.environments.yokai.native.constants import (
    YOKAI_COLORS, INITIAL_POSITIONS_3x3, INITIAL_POSITIONS_4x4, INITIAL_neighbors_3x3,
    INITIAL_neighbors_4x4, REL_MOVE_INDEX_TO_OFFSET, ALL_ONE_COLORS_3x3, ALL_ONE_COLORS_4x4,
    ALL_TWO_COLORS_3x3, ALL_TWO_COLORS_4x4, ALL_THREE_COLORS_4x4
)
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
import jaxmarl.environments.spaces as spaces


@struct.dataclass
class State:
    positions: chex.Array  # 16 x 2
    neighbors: chex.Array  # 16 x 16

    # Array of boolean values if an agent has seen the card before
    card_knowledge: chex.Array  # 2 x 16 Who has seen which card
    card_knowledge_last_added: chex.Array
    current_cards_revealed: chex.Array  # 16 x 1
    card_previously_seen: chex.Array  # 16 x 1

    colors: chex.Array  # 16 x 1
    card_locked: chex.Array  # 16 x 1

    hints_color_on_card: chex.Array  # 16 x 1
    hints_on_top_of_card: chex.Array
    hints_on_top_of_card_by_player: chex.Array
    hints: chex.Array  # save all the hint cards color # 7 x 4
    hints_pos: chex.Array  # 7 x 2
    hints_revealed: chex.Array  # 7 x 1
    hints_placed: chex.Array  # 7 x 1
    hints_placed_by_player: chex.Array  # 7 x 2 x 1
    hints_id_on_top_of_card: chex.Array  # Y x H

    hints_terminal: bool  # indicate if one round of the game ends or not
    player_terminal: bool  # indicate if one of the player ends the game or not

    # Belief Labels; Used by some methods to do representation learning
    # Defaults to state.colors
    belief_labels: chex.Array

    # Used for being added to observation
    current_active_action: int
    current_active_greedy_action: int

    # For the intermediate reward calculation
    num_clustered: int
    corr_hints: int
    incorr_hints: int
    num_opened_per_player: chex.Array

    t: int
    substep: int
    cur_player_idx: chex.Array


class YokaiGrid(MultiAgentEnv):

    def __init__(
        self,
        num_agents,
        num_colors,
        max_hint_of_color_num,
        deck_size,
        num_hint_of_color_num,
        obs_kind,
        grid_size: int,
        all_colors_revealed: bool = False,
        past_colors_revealed: bool = False,
        color_probs_calculated: bool = False,
        encode_actions: bool = False,
        sad: bool = False,
    ):
        """Returns an instance of YokaiGrid.

        Args:
            num_agents (int): Number of agents.
            num_colors (int): Number of colours.
            max_hint_of_color_num (list): Per color combinations how many hints.
            deck_size (int): Number of Cards.
            num_hint_of_color_num (list): Number of hints of each type.
            obs_kind (enum): graph, image, etc.
            grid_size (int]): Makes a grid_sizexgrid_size grid.
            all_colors_revealed (bool, optional): Whether to reveal all colours. Defaults to False.
            past_colors_revealed (bool, optional): Whether to reveal previously observed colours. Defaults to False.
            color_probs_calculated (bool, optional): Whether to calculate color probs in observation. Defaults to False.
            encode_actions (bool, optional): Whether to encode actions. Defaults to False.
            sad (bool, optional): Whether to encode greedy actions. Defaults to False.
        """
        super().__init__(num_agents)

        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)
        self.num_colors = num_colors
        self.max_hint_of_color_num = max_hint_of_color_num
        self.num_hint_of_color_num = num_hint_of_color_num
        self.grid_size = grid_size

        # total num of the chosen hint cards
        self.hint_size = np.sum(num_hint_of_color_num)
        self.deck_size = deck_size  # numbers of total yokai cards

        self.all_colors_revealed = all_colors_revealed
        self.past_colors_revealed = past_colors_revealed
        self.color_probs_calculated = color_probs_calculated
        self.encode_actions = encode_actions
        self.sad = sad
        self.sad_obs_size = 4 + self.deck_size + 2 + self.hint_size + 1

        if obs_kind == "image":
            self.get_obs = self.get_image_obs
        elif obs_kind == "graph":
            self.get_obs = self.get_graph_obs
        elif obs_kind == "graph_global":
            self.get_obs = self.get_graph_obs_global
        elif obs_kind == "linear":
            self.get_obs = self.get_linear_obs
        elif obs_kind == "graph_featurised":
            self.get_obs = self.get_graph_obs_featurised
        elif obs_kind == "graph_featurised_global":
            self.get_obs = self.get_graph_obs_featurised_global
        else:
            raise ValueError(f"{obs_kind}: Is not supported!")

        self.agents = [f"agent_{i}" for i in range(num_agents)]

        self.hint_card_total1 = ALL_ONE_COLORS_4x4 if self.num_colors == 4 else ALL_ONE_COLORS_3x3
        self.hint_card_total2 = ALL_TWO_COLORS_4x4 if self.num_colors == 4 else ALL_TWO_COLORS_3x3
        self.hint_card_total3 = ALL_THREE_COLORS_4x4 if self.num_colors == 4 else None

        self.num_skip = 1
        self.num_moves_ob = deck_size
        self.num_moves_move = deck_size * grid_size**2
        self.num_moves_hint = self.hint_size + self.hint_size * grid_size**2
        self.num_terminate = 1
        self.num_features = self.num_colors*2+6+1

        if self.get_obs == self.get_linear_obs:
            self.obs_size = self.num_features * \
                (self.deck_size + self.hint_size) + \
                (self.deck_size + self.hint_size)**2
            self.obs_space = spaces.Dict({
                "linear": spaces.Discrete(self.obs_size)
            })
        elif self.get_obs == self.get_image_obs:
            self.obs_size = (self.grid_size, self.grid_size +
                             1, self.num_features)

            if self.encode_actions and self.sad:
                self.obs_space = spaces.Dict({
                    "image": spaces.Box(low=0, high=1, shape=self.obs_size, dtype=jnp.int32),
                    "action_obs": spaces.Box(low=0, high=1, shape=(self.sad_obs_size*2,), dtype=jnp.int32)
                })
            elif self.encode_actions:
                self.obs_space = spaces.Dict({
                    "image": spaces.Box(low=0, high=1, shape=self.obs_size, dtype=jnp.int32),
                    "action_obs": spaces.Box(low=0, high=1, shape=(self.sad_obs_size,), dtype=jnp.int32)
                })
            elif self.sad:
                self.obs_space = spaces.Dict({
                    "image": spaces.Box(low=0, high=1, shape=self.obs_size, dtype=jnp.int32),
                    "action_obs": spaces.Box(low=0, high=1, shape=(self.sad_obs_size,), dtype=jnp.int32)
                })
            else:
                self.obs_space = spaces.Dict({
                    "image": spaces.Box(low=0, high=1, shape=self.obs_size, dtype=jnp.int32)
                })
        elif self.get_obs == self.get_graph_obs:
            self.obs_size = self.deck_size+self.hint_size
            self.node_embedding_size = self.num_features
            self.obs_space = spaces.Dict({
                "node_features": spaces.Box(low=0, high=1, shape=(self.obs_size, self.node_embedding_size), dtype=jnp.int32),
                "adjacency_matrix": spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size), dtype=jnp.int32),
                "positions": spaces.Box(low=0, high=1, shape=(self.deck_size, 2), dtype=jnp.int32),
            })
        elif self.get_obs == self.get_graph_obs_global:
            self.num_nodes = self.grid_size * (self.grid_size + 1)
            self.node_embedding_size = self.num_features
            self.obs_space = spaces.Dict({
                "node_features": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.node_embedding_size), dtype=jnp.int32),
                "adjacency_matrix": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=jnp.int32),
            })
        elif self.get_obs == self.get_graph_obs_featurised:
            # Returns the graph observation with the features already featurised
            self.obs_size = self.deck_size+self.hint_size
            self.obs_space = spaces.Dict({
                "adjacency_matrix": spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size), dtype=jnp.int32),
                "color": spaces.Box(low=0, high=1, shape=(self.obs_size, self.num_colors), dtype=jnp.int32),
                "positions": spaces.Box(low=0, high=1, shape=(self.obs_size, 2), dtype=jnp.int32),
                "card_locked": spaces.Box(low=0, high=1, shape=(self.obs_size, 1), dtype=jnp.int32),
                "previously_seen_by_others": spaces.Box(low=0, high=1, shape=(self.obs_size, 1), dtype=jnp.int32),
                "is_current_player": spaces.Box(low=0, high=1, shape=(self.obs_size,), dtype=jnp.int32),
                "substep": spaces.Box(low=0, high=1, shape=(self.obs_size,), dtype=jnp.int32),
                "card_id": spaces.Box(low=-7, high=16, shape=(self.obs_size, 1), dtype=jnp.int32),
            })
        else:
            raise ValueError("Is not supported!")

        # NOTE: These are deprecated and not used in our work.
        # if self.all_colors_revealed:
        #     self.feature_fn = self._built_observation_features_all_colors_revealed
        # if self.color_probs_calculated:
        #     self.feature_fn = self._built_observation_features_color_probs
        # else:
        self.feature_fn = self._built_observation_features

        self.action_dim = self.num_skip + self.num_moves_ob + \
            self.num_moves_move + self.num_moves_hint + self.num_terminate
        self.action_spaces = {i: spaces.Discrete(self.action_dim)
                              for i in self.agents}
        self.observation_spaces = {i: self.obs_space for i in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def move_card_to(
        self,
        positions: chex.Array,
        neighbors: chex.Array,
        card_idx: int,
        x: int,
        y: int
    ) -> State:
        """
        NOTE: This function assumes that the move is legal. It does do no sanity checks!
        There are other functions that check for legality.

        :param positions: The current positions of the cards in the environment
        :param neighbors: Who is neighbour to whom currently
        :param card_from_index: The index of the card to move
        :param card_to_index: The index of the card to move to
        :param rel_pos: The relative position to move the card to
        :return: (new_positions, new_neighbors, illegal_pos_flag)
        """
        card_from_pos = positions[card_idx]  # 0,0
        new_pos = jnp.array([x, y], dtype=jnp.uint8)

        # First set all old neighbors from the adjecency matrix to zero
        neighbors = neighbors.at[card_idx, :].set(0)
        neighbors = neighbors.at[:, card_idx].set(0)

        # Then set the new neighbors based on the new positions
        new_positions = positions.at[card_idx].set(new_pos)

        query_pos_l = (
            new_pos+REL_MOVE_INDEX_TO_OFFSET[0] == new_positions).all(axis=-1)
        query_pos_r = (
            new_pos+REL_MOVE_INDEX_TO_OFFSET[1] == new_positions).all(axis=-1)
        query_pos_u = (
            new_pos+REL_MOVE_INDEX_TO_OFFSET[2] == new_positions).all(axis=-1)
        query_pos_d = (
            new_pos+REL_MOVE_INDEX_TO_OFFSET[3] == new_positions).all(axis=-1)

        new_neighbors = jnp.logical_or(query_pos_l, query_pos_r)
        new_neighbors = jnp.logical_or(new_neighbors, query_pos_u)
        new_neighbors = jnp.logical_or(new_neighbors, query_pos_d)

        neighbors = neighbors.at[card_idx, :].set(new_neighbors)
        neighbors = neighbors.at[:, card_idx].set(new_neighbors)

        # Cant be in old positions
        illegal_pos_flag = jnp.any(jnp.all(new_pos == positions, axis=1))
        illegal_pos_flag = jnp.logical_or(
            illegal_pos_flag, (new_pos == card_from_pos).all())
        return new_positions, neighbors, illegal_pos_flag

    def _is_connected(self, new_neighbors):
        new_neighbors = jnp.fill_diagonal(
            new_neighbors, 1, inplace=False)
        reachability_matrix = jnp.linalg.matrix_power(
            new_neighbors, self.deck_size)
        reachable = jnp.all(reachability_matrix > 0).astype(jnp.uint8)
        return reachable

    @partial(jax.jit, static_argnums=[0])
    def get_legal_moves(self, state: State) -> chex.Array:

        def _legal_substep_0_and_1(state):
            def _get_legal_obs(state):
                not_locked = jnp.logical_not(state.card_locked)
                not_revealed = jnp.logical_not(state.current_cards_revealed)
                legal_observe = jnp.logical_and(
                    not_locked, not_revealed).squeeze()
                return legal_observe.astype(jnp.uint8)

            # Only in move 0 and 1 the player can observe
            legal_obs = _get_legal_obs(state)
            substep_0 = state.substep == 0
            return jnp.concatenate([
                jnp.zeros((self.num_skip,), dtype=jnp.uint8),  # This is skip
                legal_obs,
                jnp.zeros((self.num_moves_move + self.num_moves_hint,),
                          dtype=jnp.uint8),
                jnp.array([substep_0], dtype=jnp.uint8),
            ])

        def _legal_substep_2(state: State) -> chex.Array:

            @partial(jax.vmap, in_axes=(0, None, None))
            def _check_per_card(card, positions, neighbors):

                @partial(jax.vmap, in_axes=(0, None, None, None))
                def _check_each_position(target_pos, card, positions, neighbors):
                    x = target_pos[0]
                    y = target_pos[1]
                    _, new_neighbors, illegal_pos_flag = self.move_card_to(
                        positions, neighbors, card, x, y)
                    reachable = self._is_connected(new_neighbors)
                    not_illegal = jnp.logical_not(illegal_pos_flag)
                    card_not_locked = jnp.logical_not(state.card_locked[card])
                    card_x_in_grid = jnp.logical_and(
                        x >= 0, x < self.grid_size)
                    card_y_in_grid = jnp.logical_and(
                        y >= 0, y < self.grid_size)
                    card_in_grid = jnp.logical_and(
                        card_x_in_grid, card_y_in_grid)
                    return jnp.logical_and(
                        jnp.logical_and(reachable, not_illegal),
                        jnp.logical_and(card_not_locked, card_in_grid))

                # all_possible_positions = jnp.arange(self.grid_size**2)
                all_possible_targets = jnp.vstack([
                    positions + jnp.array([0, 1]),
                    positions + jnp.array([0, -1]),
                    positions + jnp.array([-1, 0]),
                    positions + jnp.array([1, 0])
                ])
                all_possible_moves = jnp.zeros(
                    (self.grid_size**2, ), dtype=jnp.uint8)
                legal_moves = _check_each_position(
                    all_possible_targets, card, positions, neighbors)
                # We need to insert the legal moves into the correct position in all_possible_moves
                # all_pssoible_moves is a 1D array with the possible moves whereas legal_moves is a 1D array for only the positions stored in all_possible_targets
                # We need to find the index of the positions in all_possible_targets in all_possible_moves
                # Neighbor targets can lie outside the grid. Flattening them
                # directly aliases valid cells (e.g. (7, -1) and (6, 9)), and
                # duplicate scatter-set updates can then disagree across JIT
                # contexts. Drop out-of-grid targets before flattening, and
                # combine repeated valid targets with an order-independent OR.
                in_grid = jnp.all(
                    (all_possible_targets >= 0)
                    & (all_possible_targets < self.grid_size), axis=1)
                target_indices = jnp.where(
                    in_grid,
                    all_possible_targets[:, 0] * self.grid_size
                    + all_possible_targets[:, 1],
                    self.grid_size**2,
                )
                all_possible_moves = all_possible_moves.at[target_indices].max(
                    legal_moves, mode="drop")
                return all_possible_moves

            card_ids = jnp.arange(self.deck_size)
            legal_moves_card = _check_per_card(
                card_ids, state.positions, state.neighbors)
            legal_moves_card = legal_moves_card.reshape(
                (self.num_moves_move,)).astype(jnp.uint8)

            return jnp.concatenate([
                jnp.zeros((self.num_skip + self.num_moves_ob,),
                          dtype=jnp.uint8),
                legal_moves_card,
                jnp.zeros((self.num_moves_hint + self.num_terminate,),
                          dtype=jnp.uint8)
            ])

        def _legal_substep_3(state):

            @partial(jax.vmap, in_axes=(0, None, None, None))
            def _check_per_hint_card(hint_card, positions, hints_placed, hints_revealed):

                @partial(jax.vmap, in_axes=(0, None, None, None, None))
                def _check_each_position(target_pos, hint_card, positions, hints_placed, hints_revealed):
                    x = target_pos[0]
                    y = target_pos[1]
                    pos = jnp.array([x, y], dtype=jnp.uint8)
                    pos_is_card = jnp.any(jnp.all(pos == positions, axis=1))
                    card_id = jnp.where(
                        jnp.all(pos == positions, axis=1), size=1)[0][0]
                    card_not_locked = jnp.logical_not(
                        state.card_locked[card_id])

                    can_place_hint = jnp.logical_and(
                        jnp.logical_not(hints_placed[hint_card]),
                        hints_revealed[hint_card])
                    can_place_hint_onto_card = jnp.logical_and(
                        can_place_hint, card_not_locked)
                    is_legal = jnp.logical_and(
                        pos_is_card, can_place_hint_onto_card)
                    # jax.debug.print("hint_card {h} Position {pos} is_legal {l} can_place_hint {l1} can_place_hint_onto_card {l2} pos_is_card {l3}", h=hint_card, pos=pos, l=is_legal, l1=can_place_hint, l2=can_place_hint_onto_card, l3=pos_is_card)
                    return is_legal

                all_possible_moves = jnp.zeros(
                    (self.grid_size**2, ), dtype=jnp.uint8)
                all_possible_targets = state.positions
                legal_moves = _check_each_position(
                    all_possible_targets, hint_card, positions, hints_placed, hints_revealed)
                # We need to insert the legal moves into the correct position in all_possible_moves
                # all_pssoible_moves is a 1D array with the possible moves whereas legal_moves is a 1D array for only the positions stored in all_possible_targets
                # We need to find the index of the positions in all_possible_targets in all_possible_moves
                all_possible_moves = all_possible_moves.at[all_possible_targets[:, 0]
                                                           * self.grid_size + all_possible_targets[:, 1]].set(legal_moves)
                return all_possible_moves

            can_reveal = jnp.logical_not(state.hints_revealed)
            # Needs to be revealed and not placed

            hint_ids = jnp.arange(self.hint_size)
            can_be_placed_onto = _check_per_hint_card(
                hint_ids, state.positions, state.hints_placed, state.hints_revealed)
            legal_hint = jnp.concatenate(
                [can_reveal.squeeze(), can_be_placed_onto.reshape(-1)], axis=-1).astype(jnp.uint8)

            return jnp.concatenate([
                jnp.zeros((self.num_skip + self.num_moves_ob +
                          self.num_moves_move,), dtype=jnp.uint8),
                legal_hint,
                jnp.zeros((self.num_terminate,), dtype=jnp.uint8)
            ])

        @partial(jax.vmap, in_axes=[0, None])
        def _get_legal_moves_for_all(aidx, legal_moves_active_player):
            """Sets the legal moves for all players by giving the current player
            all legal moves and setting the rest to skip.
            """
            cur_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            return jax.lax.cond(
                aidx == cur_player,
                lambda _: legal_moves_active_player,
                lambda _: jnp.zeros((self.action_dim,),
                                    dtype=jnp.uint8).at[0].set(1),
                legal_moves_active_player
            )

        legal_moves_active_player = jax.lax.switch(
            state.substep,
            [
                _legal_substep_0_and_1,
                _legal_substep_0_and_1,
                _legal_substep_2,
                _legal_substep_3,
            ],
            state
        )

        # legal_moves_active_player = legal_moves_fn(state)
        legal_moves_both = _get_legal_moves_for_all(
            self.agent_range, legal_moves_active_player)
        dict_result = {}
        for i, a in enumerate(self.agents):
            dict_result[a] = legal_moves_both[i]
        return dict_result

    def get_initial_hints(self, key: chex.PRNGKey):
        """Returns the initial hints for the game since this 
        depends on the number of hints and their rank this function
        needs to be defined by any subclass.

        Args:
            key (chex.PRNGKey): Random key

        Raises:
            ValueError: Not Implemented.
        """
        raise ValueError("Not implemented")

    @partial(jax.jit, static_argnums=(0,))
    def _is_color_connected(self, colors, adj, color):
        color_ids = jnp.where(colors == color, size=self.num_colors)[
            0]  # .all(axis=-1)
        local_color_adjacency = adj.at[color_ids,
                                       :].get().at[:, color_ids].get()
        local_color_adjacency = jnp.fill_diagonal(
            local_color_adjacency, 1, inplace=False)
        reachability_matrix = jnp.linalg.matrix_power(
            local_color_adjacency, self.deck_size)
        return jnp.all(reachability_matrix > 0)

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        """Reset the environment"""

        key, _key = jax.random.split(key)
        colors = np.array([
            YOKAI_COLORS[i] for _ in range(self.deck_size//self.num_colors)
            for i in range(self.num_colors)
        ])
        shuffled_colors = jax.random.permutation(_key, colors, axis=0)

        hints_color_on_card = jnp.zeros(
            (colors.shape[0], self.num_colors), dtype=jnp.uint8)

        initial_pos = INITIAL_POSITIONS_4x4 if self.deck_size == 16 else INITIAL_POSITIONS_3x3
        initial_pos = initial_pos + \
            int(self.grid_size // 2 - np.sqrt(self.deck_size) // 2)  # Center the grid
        initial_pos = initial_pos.astype(jnp.int32)
        initial_neighbors = INITIAL_neighbors_4x4 if self.deck_size == 16 else INITIAL_neighbors_3x3

        hints = self.get_initial_hints(key)
        hints_revealed = jnp.zeros((self.hint_size), dtype=jnp.uint8)
        hints_placed = jnp.zeros((self.hint_size), dtype=jnp.uint8)
        hints_placed_by_player = jnp.zeros(
            (self.hint_size, self.num_agents), dtype=jnp.uint8)

        card_locked = jnp.zeros((self.deck_size), dtype=jnp.uint8)
        cards_revealed = jnp.zeros((self.deck_size), dtype=jnp.uint8)

        card_knowledge = jnp.zeros(
            (self.num_agents, self.deck_size), dtype=jnp.uint8)
        card_knowledge_last_added = jnp.zeros(
            (self.num_agents, self.deck_size), dtype=jnp.uint8)
        hints_on_top_of_card = jnp.zeros((self.deck_size), dtype=jnp.uint8)
        hints_on_top_of_card_by_player = jnp.zeros(
            (self.deck_size, self.num_agents), dtype=jnp.uint8)

        card_previously_seen = jnp.zeros(
            (self.num_agents, self.deck_size), dtype=jnp.uint8)

        # reset current_player_index
        cur_player_idx = jnp.zeros(
            self.num_agents, dtype=jnp.int32).at[0].set(1)

        # Initial position is next to each other at the side of the board
        hints_pos = jnp.concatenate(
            [jnp.arange(self.hint_size)[:, None], jnp.full(self.hint_size, fill_value=-1)[:, None]], axis=-1)

        num_color_clustered = jnp.array([
            self._is_color_connected(shuffled_colors, initial_neighbors, YOKAI_COLORS[i]) for i in range(self.num_colors)
        ]).sum()

        hints_id_on_top_of_card = jnp.zeros(
            (self.deck_size, self.hint_size), dtype=jnp.uint8)
        belief_labels = jnp.stack([shuffled_colors]*self.num_agents)

        state = State(
            positions=initial_pos,
            neighbors=initial_neighbors,
            card_knowledge=card_knowledge,
            card_knowledge_last_added=card_knowledge_last_added,
            card_previously_seen=card_previously_seen,
            colors=shuffled_colors,
            hints_color_on_card=hints_color_on_card,
            current_cards_revealed=cards_revealed,
            card_locked=card_locked,
            hints_revealed=hints_revealed,
            hints=hints,
            hints_pos=hints_pos,
            hints_on_top_of_card=hints_on_top_of_card,
            hints_on_top_of_card_by_player=hints_on_top_of_card_by_player,
            hints_placed=hints_placed,
            hints_id_on_top_of_card=hints_id_on_top_of_card,
            hints_placed_by_player=hints_placed_by_player,
            hints_terminal=False,
            player_terminal=False,
            cur_player_idx=cur_player_idx,
            num_clustered=num_color_clustered,
            current_active_action=-1,
            current_active_greedy_action=-1,
            corr_hints=0,
            incorr_hints=0,
            belief_labels=belief_labels,
            num_opened_per_player=jnp.array([0, 0], dtype=jnp.int32),
            t=0,
            substep=0
        )
        obs = self.get_obs(state)
        return obs, state

    def _built_observation_features_all_colors_revealed(self, aidx: int, state: State) -> chex.Array:
        mask = jnp.zeros((self.num_agents,), dtype=jnp.uint8).at[aidx].set(1)
        mask = jnp.logical_not(mask)
        previously_seen_by_others = jax.lax.cond(
            self.past_colors_revealed,
            lambda: (state.card_knowledge * mask[:, None]).sum(axis=0),
            lambda: (state.card_knowledge_last_added *
                     mask[:, None]).sum(axis=0),
        )

        cards = jnp.concatenate([
            jax.nn.one_hot(state.colors, num_classes=self.num_colors),
            jnp.zeros_like(jax.nn.one_hot(
                state.colors, num_classes=self.num_colors)),
            state.positions,
            state.card_locked[:, None],
            previously_seen_by_others[:, None],
            jnp.tile(state.cur_player_idx[aidx], (self.deck_size, 1)),
            jnp.tile(state.substep, (self.deck_size, 1)),
            jnp.arange(self.deck_size)[:, None],
        ], axis=1)

        hints = jnp.concatenate([
            jnp.zeros_like(state.hints),
            state.hints,
            state.hints_pos,
            state.hints_placed[:, None],
            jnp.zeros((self.hint_size, 1)),
            jnp.tile(state.cur_player_idx[aidx], (self.hint_size, 1)),
            jnp.tile(state.substep, (self.hint_size, 1)),
            (jnp.arange(self.hint_size)[:, None]+1)*-1,
        ], axis=1)
        return cards, hints

    def _built_observation_features_color_probs(self, aidx, state):
        is_current_player = state.cur_player_idx[aidx]
        is_current_player = state.cur_player_idx[aidx]

        what_to_show = (jnp.logical_or(
            state.card_knowledge.at[aidx].get(),
            state.current_cards_revealed).astype(jnp.uint8) * is_current_player
        ).astype(jnp.uint8)
        colors_revealed = jax.nn.one_hot(
            state.colors, num_classes=self.num_colors)
        colors_revealed = (colors_revealed *
                           what_to_show[:, None]).astype(jnp.uint8)

        # This is where we need to insert colors
        open_spots = jnp.logical_not(what_to_show)
        place_holder = jnp.ones((self.deck_size, self.num_colors))
        num_revlead_per_color = colors_revealed.sum(
            axis=0)  # Gives count per colors
        all_revealed_per_color = num_revlead_per_color == self.deck_size // self.num_colors
        should_be_masked = jnp.logical_not(
            all_revealed_per_color).astype(jnp.uint8)
        card_probs = place_holder * should_be_masked

        colors_revealed = jnp.where(
            jnp.tile(open_spots[..., None], self.deck_size // self.num_colors),
            card_probs,
            colors_revealed)

        # Need previous move summary!
        mask = jnp.zeros((self.num_agents,), dtype=jnp.uint8).at[aidx].set(1)
        mask = jnp.logical_not(mask)
        previously_seen_by_others = jax.lax.cond(
            self.past_colors_revealed,
            lambda: (state.card_knowledge * mask[:, None]).sum(axis=0),
            lambda: (state.card_knowledge_last_added *
                     mask[:, None]).sum(axis=0),
        )

        # Previous moves include knowing that the other person has viewed cards before.
        cards = jnp.concatenate([
            colors_revealed,
            jnp.zeros_like(colors_revealed),
            state.positions,
            state.card_locked[:, None],
            previously_seen_by_others[:, None],
            jnp.tile(is_current_player, (self.deck_size, 1)),
            jnp.tile(state.substep, (self.deck_size, 1)),
            jnp.arange(self.deck_size)[:, None],
        ], axis=1)

        hints = jnp.concatenate([
            jnp.zeros_like(state.hints),
            state.hints * state.hints_revealed[:, None],
            state.hints_pos,
            state.hints_placed[:, None],
            jnp.zeros((self.hint_size, 1)),
            jnp.tile(is_current_player, (self.hint_size, 1)),
            jnp.tile(state.substep, (self.hint_size, 1)),
            (jnp.arange(self.hint_size)[:, None]+1)*-1,
        ], axis=1)

        return cards, hints

    def _add_past_info_to_features(self, aidx, state, colors_revealed):
        player_colors = state.card_knowledge.at[aidx].get()
        colors_revealed = jnp.logical_or(colors_revealed, player_colors)
        return colors_revealed.astype(jnp.uint8)

    def _built_oberservation_colors_revealed(self, aidx: int, state: State) -> chex.Array:
        is_current_player = state.cur_player_idx[aidx]

        what_to_show = jax.lax.cond(
            self.past_colors_revealed,
            lambda aidx, state: (jnp.logical_or(state.card_knowledge.at[aidx].get(
            ), state.current_cards_revealed).astype(jnp.uint8) * is_current_player).astype(jnp.uint8),
            lambda aidx, state: (
                state.current_cards_revealed * is_current_player).astype(jnp.uint8),
            aidx, state
        )
        colors_revealed = jax.nn.one_hot(
            state.colors, num_classes=self.num_colors)
        colors_revealed = (colors_revealed *
                           what_to_show[:, None]).astype(jnp.uint8)
        return colors_revealed

    def _built_observation_features(self, aidx: int, state: State) -> chex.Array:
        is_current_player = state.cur_player_idx[aidx]
        colors_revealed = self._built_oberservation_colors_revealed(
            aidx, state)

        # Need previous move summary!
        mask = jnp.zeros((self.num_agents,), dtype=jnp.uint8).at[aidx].set(1)
        mask = jnp.logical_not(mask)
        previously_seen_by_others = jax.lax.cond(
            self.past_colors_revealed,
            lambda: (state.card_knowledge * mask[:, None]).sum(axis=0),
            lambda: (state.card_knowledge_last_added *
                     mask[:, None]).sum(axis=0),
        )

        # Previous moves include knowing that the other person has viewed cards before.
        cards = jnp.concatenate([
            colors_revealed,
            jnp.zeros_like(colors_revealed),
            state.positions,
            state.card_locked[:, None],
            previously_seen_by_others[:, None],
            jnp.tile(is_current_player, (self.deck_size, 1)),
            jnp.tile(state.substep, (self.deck_size, 1)),
            jnp.arange(self.deck_size)[:, None],
        ], axis=1)

        hints = jnp.concatenate([
            jnp.zeros_like(state.hints),
            state.hints * state.hints_revealed[:, None],
            state.hints_pos,
            state.hints_placed[:, None],
            jnp.zeros((self.hint_size, 1)),
            jnp.tile(is_current_player, (self.hint_size, 1)),
            jnp.tile(state.substep, (self.hint_size, 1)),
            (jnp.arange(self.hint_size)[:, None]+1)*-1,
        ], axis=1)

        return cards, hints

    @partial(jax.jit, static_argnums=(0,))
    def get_linear_obs(self, state: State) -> dict[str, chex.Array]:
        """Linear obs are a flattened vector of node features and adjacency matrix.

        Args:
            state (State): State of the game

        Returns:
            dict[str, chex.Array]: dict containing the linear observation per agent.
        """

        @partial(jax.vmap, in_axes=[0, None])
        def _observe(aidx: int, state: State) -> chex.Array:
            cards, hints = self.feature_fn(aidx, state)
            features = jnp.concatenate([cards, hints]).astype(jnp.uint8)
            return {"linear": jnp.concatenate([state.neighbors.reshape(-1), features.reshape(-1)])}

        obs = _observe(self.agent_range, state)
        obs = {a: {key: obs[key][i] for key in self.obs_space.spaces.keys()}
               for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_object_obs(self, state: State) -> dict[str, chex.Array]:
        """DEPRECATED: Same is attainable via the graph observation.
        The idea is simple: Give back a matrix n x m where n is the number of cards 
        and m is their dimension plus and ajecency matrix.

        Args:
            state (State): State of the game

        Returns:
            dict[str, chex.Array]: dict containing the linear observation per agent.
        """
        @partial(jax.vmap, in_axes=[0, None])
        def _observe(aidx: int, state: State) -> chex.Array:
            cards, hints = self.feature_fn(aidx, state)
            features = jnp.concatenate([cards, hints]).astype(jnp.int8)
            return {"node_features": features, "adjacency_matrix": state.neighbors.reshape(-1)}

        obs = _observe(self.agent_range, state)
        obs = {a: {key: obs[key][i] for key in self.obs_space.spaces.keys()}
               for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_image_obs(self, state: State) -> dict[str, chex.Array]:
        """Generates a gxg+1 image where g is the grid size and +1 is the hint cards.
        Represents a CNN passable image. Channel depth is determined by feature size.

        Args:
            state (State): State of the game

        Returns:
            dict[str, chex.Array]: dict containing the linear observation per agent.
        """
        def _get_image_from_features(state: State, cards: chex.Array, hints: chex.Array) -> chex.Array:
            # Grid for yokai cards
            grid_yokai = jnp.zeros(
                (self.grid_size, self.grid_size+1, self.num_features))
            grid_hints = jnp.zeros_like(grid_yokai)  # Grid for hint cards
            # pos = jnp.concatenate([state.positions, hint_pos], axis=0)
            grid_yokai = grid_yokai.at[
                state.positions[:, 0], state.positions[:, 1], :].set(cards)
            grid_hints = grid_hints.at[
                state.hints_pos[:, 0], state.hints_pos[:, 1], :].set(hints)
            grid = jnp.where(grid_yokai != 0, grid_yokai,
                             grid_hints)  # .transpose(2,1,0)
            return grid

        @partial(jax.vmap, in_axes=[0, None])
        def _observe(aidx: int, state: State) -> chex.Array:
            cards, hints = self.feature_fn(aidx, state)
            grid = _get_image_from_features(state, cards, hints)
            obs = {"image": grid}

            if self.encode_actions and not self.sad:
                action_obs = self.get_action_obs(
                    state, state.current_active_action)
                obs = {**obs, "action_obs": action_obs}
            elif not self.encode_actions and self.sad:
                action_obs = self.get_action_obs(
                    state, state.current_active_greedy_action)
                obs = {**obs, "action_obs": action_obs}
            elif self.encode_actions and self.sad:
                action_obs = jnp.concatenate([
                    self.get_action_obs(state, state.current_active_action),
                    self.get_action_obs(
                        state, state.current_active_greedy_action),
                ], axis=-1)
                obs = {**obs, "action_obs": action_obs}
            return obs

        # ACTION
        obs = _observe(self.agent_range, state)
        obs = {a: {key: obs[key][i] for key in self.obs_space.spaces.keys()}
               for i, a in enumerate(self.agents)}

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_graph_obs_global(self, state: State) -> dict[str, chex.Array]:
        """Generates a graph of n nodes and a nxn adjacency matrix 
        where n is the number of total grid positions (g x (g+1)).
        Returns:
            {
                "node_features": [num_players, n_nodes, num_features],
                "adj":           [n_nodes, n_nodes],
            }
        """

        num_nodes = self.grid_size * (self.grid_size + 1)

        # Precompute adjacency for the full g x (g+1) grid (4-connectivity, no self loops).
        idx = jnp.arange(num_nodes, dtype=jnp.int32)               # [n]
        rows = idx // (self.grid_size + 1)                          # [n]
        cols = idx %  (self.grid_size + 1)                          # [n]

        drow = jnp.abs(rows[:, None] - rows[None, :])               # [n, n]
        dcol = jnp.abs(cols[:, None] - cols[None, :])               # [n, n]

        # Manhattan distance 1 = 4-neighbour connectivity
        adj = (drow + dcol == 1).astype(jnp.uint8)                  # [n, n]

        @partial(jax.vmap, in_axes=[0, None])
        def _observe(aidx: int, state: State) -> chex.Array:
            cards, hints = self.feature_fn(aidx, state)
            # node_features: [n_nodes, num_features]
            features = jnp.zeros((num_nodes, self.num_features), dtype=jnp.int8)

            # Flatten (r, c) -> node index k = r * (g+1) + c
            card_positions_flat = (
                state.positions[:, 0] * (self.grid_size + 1)
                + state.positions[:, 1]
            )
            features = features.at[card_positions_flat, :].set(cards)

            hint_positions_flat = (
                state.hints_pos[:, 0] * (self.grid_size + 1)
                + state.hints_pos[:, 1]
            )
            features = features.at[hint_positions_flat, :].set(hints)

            return {"node_features": features, "adjacency_matrix": adj}

        # One graph per agent: [num_players, n_nodes, num_features]
        obs = _observe(self.agent_range, state)
        obs = {a: {key: obs[key][i] for key in self.obs_space.spaces.keys() if key in obs}
               for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_graph_obs(self, state: State) -> dict[str, chex.Array]:
        """Generates a graph of n nodes and a nxn adjacency matrix where n is the number of cards + hints.

        Args:
            state (State): State of the game

        Returns:
            dict[str, chex.Array]: dict containing the linear observation per agent.
        """
        @partial(jax.vmap, in_axes=[0, None])
        def _observe(aidx: int, state: State) -> chex.Array:
            cards, hints = self.feature_fn(aidx, state)
            features = jnp.concatenate([cards, hints]).astype(jnp.int8)
            adj = state.neighbors
            adj = jnp.pad(adj, (
                (0, self.hint_size), (0, self.hint_size)), mode='constant', constant_values=0)
            # hints that are on top of the cards need to be connected to the cards
            # for this we use hints_id_on_top_of_card_after
            adj = adj.at[:self.deck_size, self.deck_size:].set(
                state.hints_id_on_top_of_card)
            adj = adj.at[self.deck_size:, :self.deck_size].set(
                state.hints_id_on_top_of_card.T)
            diag_elements = jnp.diag_indices_from(adj)
            adjacency_matrix = adj.at[diag_elements].set(1)
            return {
                "node_features": features,
                "adjacency_matrix": adjacency_matrix,
                "positions": state.positions
            }

        obs = _observe(self.agent_range, state)
        obs = {a: {key: obs[key][i] for key in self.obs_space.spaces.keys() if key in obs}
               for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_graph_obs_featurised(self, state: State) -> dict[str, chex.Array]:
        """Generates a gxg+1 image where g is the grid size and +1 is the hint cards.
        Represents a CNN passable image. Channel depth is determined by feature size.

        Args:
            state (State): State of the game

        Returns:
            dict[str, chex.Array]: dict containing the linear observation per agent.
        """
        @partial(jax.vmap, in_axes=[0, None])
        def _observe(aidx: int, state: State) -> chex.Array:
            cards, hints = self.feature_fn(aidx, state)
            features = jnp.concatenate([cards, hints]).astype(jnp.int8)
            adj = state.neighbors
            adj = jnp.pad(adj, (
                (0, self.hint_size), (0, self.hint_size)), mode='constant', constant_values=0)
            adj = adj.at[:self.deck_size, self.deck_size:].set(
                state.hints_id_on_top_of_card)
            adj = adj.at[self.deck_size:, :self.deck_size].set(
                state.hints_id_on_top_of_card.T)
            diag_elements = jnp.diag_indices_from(adj)
            adjacency_matrix = adj.at[diag_elements].set(1)
            return {
                "adjacency_matrix": adjacency_matrix,
                "color": jnp.concatenate([cards[:, 0:self.num_colors], hints[:, self.num_colors:2*self.num_colors]]),
                "positions": features[:, 2*self.num_colors:2*self.num_colors+2],
                "card_locked": features[:, 2*self.num_colors+2][:, None],
                "previously_seen_by_others": features[:, 2*self.num_colors+3][:, None],
                "card_id": features[:, -1][:, None].astype(jnp.int8),
                "is_current_player": features[:, -3].astype(jnp.int8),
                "substep": features[:, -2].astype(jnp.int8),
            }

        obs = _observe(self.agent_range, state)
        obs = {a: {key: obs[key][i] for key in self.obs_space.spaces.keys() if key in obs}
               for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_action_obs(self, state: State, action: int):
        # Takes in the past active player action and hands it out to be observed
        def _encode_obs_action(a):
            card = a - self.num_skip
            return jnp.concatenate([
                jax.nn.one_hot(card, num_classes=self.deck_size),
                jnp.array([-1, -1]),
                jnp.zeros((self.hint_size,))
            ])

        def _encode_move_action(a):
            # One continous action space, needs to account for the offset
            action = a - self.num_moves_ob - self.num_skip
            card, x, y = self._convert_move_action(action)
            return jnp.concatenate([
                jax.nn.one_hot(card, num_classes=self.deck_size),
                jnp.array([x, y]),
                jnp.zeros((self.hint_size,))
            ])

        def _encode_reveal_hint(a):
            return jnp.concatenate([
                jnp.zeros((self.deck_size,)),
                jnp.array([-1, -1]),
                jax.nn.one_hot(a, num_classes=self.hint_size),
            ])

        def _encode_place_hint(a):
            action = a - self.hint_size
            hint_id_to_use, x, y = self._convert_hint_action(action)
            card_id_to_hint = jnp.where(
                jnp.all(state.positions == jnp.array(
                    [x, y], dtype=jnp.uint8), axis=1),
                size=1
            )[0][0]
            return jnp.concatenate([
                jax.nn.one_hot(card_id_to_hint, num_classes=self.deck_size),
                jnp.array([x, y]),
                jax.nn.one_hot(hint_id_to_use, num_classes=self.hint_size),
            ])

        def _encode_hint_action(a):
            action = a - self.num_moves_ob - self.num_moves_move - self.num_skip
            is_reveal = action < self.hint_size
            return jax.lax.cond(is_reveal, _encode_reveal_hint, _encode_place_hint, action)

        move_type = jax.nn.one_hot((state.substep - 1) % 4, num_classes=4)

        # Previous substep for determining move
        # new_substep = (state.substep + 1) % 4
        encoding = jax.lax.switch(
            (state.substep - 1) % 4,
            [
                _encode_obs_action,
                _encode_obs_action,
                _encode_move_action,
                _encode_hint_action,
            ],
            action
        )

        encoding = jax.lax.cond(
            (action == self.action_dim - 1),
            lambda: jnp.zeros_like(encoding),
            lambda: encoding
        )

        terminal_encoding = jnp.array([(action == self.action_dim - 1)])
        action_obs = jnp.concatenate(
            [
                move_type,
                encoding,
                terminal_encoding
            ]
        )

        action_obs = jax.lax.cond(
            action == -1,  # Place holder on reset for no action taken yet.
            lambda: jnp.zeros_like(action_obs),
            lambda: action_obs
        )

        return action_obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]

        if self.sad:
            greedy_actions = jnp.array(
                [actions["sad_action"][i] for i in self.agents])
            greedy_action = greedy_actions.at[aidx].get()
        else:
            greedy_action = -1

        actions = jnp.array([actions[i] for i in self.agents])
        # get the current agent index

        action = actions.at[aidx].get()
        state = self.step_agent(key, state, aidx, action)
        done = self.terminal(state)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        state = state.replace(
            current_active_action=action,
            current_active_greedy_action=greedy_action,
        )

        ###########################################
        # Calculate auxiliary statistics for reward
        num_clustered = jnp.array([
            self._is_color_connected(
                state.colors, state.neighbors, YOKAI_COLORS[i])
            for i in range(self.num_colors)
        ]).sum()
        num_clustered_prev = state.num_clustered
        state = state.replace(num_clustered=num_clustered)

        num_hints_correct_prev = state.corr_hints
        num_hints_incorrect_prev = state.incorr_hints

        card_ids = jnp.arange(self.deck_size)
        num_hints_placed = state.hints_placed.sum().astype(jnp.int32)
        corr_hints = state.hints_color_on_card.at[
            card_ids, state.colors].get().sum().astype(jnp.int32)
        incorr_hints = num_hints_placed - corr_hints

        state = state.replace(corr_hints=corr_hints, incorr_hints=incorr_hints)
        ###########################################

        reward, reward_dense = self.calculate_reward(
            state, num_clustered_prev, num_hints_correct_prev, num_hints_incorrect_prev)

        rewards = {agent: reward for agent in self.agents}
        rewards["__all__"] = reward
        # reward_dense = reward_dense + reward_dense_end

        state = state.replace(
            card_previously_seen=state.card_knowledge
        )

        info = {
            "reward_dense": {
                a: reward_dense * state.cur_player_idx[i] for i, a in enumerate(self.agents)},
            # Only return the number of clustered yokai cards if the game is done, i.e. done == 1
            "num_clustered_at_end": num_clustered * done,
            "success": (state.num_clustered == self.num_colors).astype(jnp.uint8),
            "done": done,
            "reward": reward,
        }

        def _update_substep_and_player(state):
            new_substep = (state.substep + 1) % 4
            # if old substep = 4 then new substep = 0 and change the player
            new_aidx = jax.lax.cond(
                new_substep == 0,
                lambda aidx: (aidx + 1) % self.num_agents,
                lambda aidx: aidx,
                aidx
            )
            cur_player_idx = jnp.zeros(self.num_agents).astype(jnp.int32)
            cur_player_idx = cur_player_idx.at[new_aidx].set(1)

            # Clean up the open cards
            state = jax.lax.cond(
                new_substep == 0,
                lambda state: state.replace(
                    current_cards_revealed=jnp.zeros_like(
                        state.current_cards_revealed),
                    card_knowledge_last_added=jnp.zeros_like(
                        state.card_knowledge_last_added),
                ),
                lambda state: state,
                state
            )
            return state.replace(substep=new_substep, cur_player_idx=cur_player_idx)

        state = _update_substep_and_player(state)

        obs = lax.stop_gradient(self.get_obs(state))
        return (
            obs,
            lax.stop_gradient(state),
            rewards,
            dones,
            info
        )

    def _convert_move_action(self, action: int) -> tuple[int, int, int]:
        move_card_index = (action) // (self.grid_size**2)
        x_and_y = (action) % (self.grid_size**2)
        x = x_and_y // self.grid_size
        y = x_and_y % self.grid_size
        return move_card_index, x, y

    def _convert_hint_action(self, action: int) -> tuple[int, int, int]:
        hint_id_to_use = action // (self.grid_size**2)
        x_and_y = action % (self.grid_size**2)
        x = x_and_y // self.grid_size
        y = x_and_y % self.grid_size
        return hint_id_to_use, x, y

    def step_agent(
        self, key: chex.PRNGKey, state: State, aidx: int, action: chex.Array,
    ) -> tuple[State, int]:

        def _step_agent_obs(state: State, aidx: int, action: int) -> State:
            action = action - self.num_skip
            current_cards_revealed = state.current_cards_revealed.at[action].set(
                True)
            card_knowledge = state.card_knowledge.at[aidx, action].set(True)
            new_card_knowledge_last_added = state.card_knowledge_last_added.at[aidx].set(
                jax.lax.cond(
                    state.card_knowledge_last_added[aidx].sum() == 2,
                    lambda x: jnp.zeros_like(x).at[action].set(True),
                    lambda x: x.at[action].set(True),
                    state.card_knowledge_last_added[aidx]
                )
            )
            return state.replace(
                current_cards_revealed=current_cards_revealed,
                card_knowledge=card_knowledge,
                card_knowledge_last_added=new_card_knowledge_last_added)

        def _step_agent_move(
            state: State, aidx: int, action: int,
        ) -> Tuple[State, int]:
            """
            Execute the  action (move one yokai card to another position)
            """

            def _no_op(state, action, aidx):
                return state

            def _move(state, action, aidx):
                # One continous action space, needs to account for the offset
                action = action - self.num_moves_ob - self.num_skip
                card, x, y = self._convert_move_action(action)
                # jax.debug.print("Move card {card} to {x}, {y}", card=card, x=x, y=y)
                new_positions, new_neighbors, _ = self.move_card_to(
                    state.positions, state.neighbors, card, x, y)
                # CHECK state new_positions for the new one
                state_new = state.replace(
                    positions=new_positions,
                    neighbors=new_neighbors
                )

                return state_new

            is_no_op = (action == 0)
            state = lax.cond(
                is_no_op, _no_op, _move, state, action, aidx)
            return state

        def _step_agent_hint(state: State, aidx: int, action: int) -> State:
            """
            Execute the  action (reveal one hint card or place one hint card)
            """
            # reveal one hint card
            def _reveal_hint(state: State, action: int):
                return state.replace(
                    hints_revealed=state.hints_revealed.at[action].set(True))

            # place one hint card
            def _place_hint(state: State, action: int):
                action = action - self.hint_size

                hint_id_to_use, x, y = self._convert_hint_action(action)
                card_id_to_hint = jnp.where(
                    jnp.all(state.positions == jnp.array(
                        [x, y], dtype=jnp.uint8), axis=1),
                    size=1
                )[0][0]

                hint_place_after = state.hints_placed.at[hint_id_to_use].set(
                    True)
                hints_placed_by_player_after = state.hints_placed_by_player.at[hint_id_to_use, aidx].set(
                    True)
                hint_color = state.hints.at[hint_id_to_use].get()
                card_locked_after = state.card_locked.at[card_id_to_hint].set(
                    True)
                hints_pos_after = state.hints_pos.at[hint_id_to_use].set(
                    state.positions.at[card_id_to_hint].get())

                hints_on_top_of_card_after = state.hints_on_top_of_card.at[card_id_to_hint].set(
                    True)

                hints_color_on_card_after = state.hints_color_on_card.at[card_id_to_hint].set(
                    hint_color)

                hints_id_on_top_of_card_after = state.hints_id_on_top_of_card.at[
                    card_id_to_hint, hint_id_to_use].set(1)

                hints_on_top_of_card_by_player_after = state.hints_on_top_of_card_by_player.at[
                    card_id_to_hint, aidx].set(True)

                return state.replace(
                    hints_placed=hint_place_after,
                    hints_placed_by_player=hints_placed_by_player_after,
                    card_locked=card_locked_after,
                    hints_pos=hints_pos_after,
                    hints_on_top_of_card=hints_on_top_of_card_after,
                    hints_on_top_of_card_by_player=hints_on_top_of_card_by_player_after,
                    hints_color_on_card=hints_color_on_card_after,
                    hints_id_on_top_of_card=hints_id_on_top_of_card_after
                )

            action = action - self.num_moves_ob - self.num_moves_move - self.num_skip

            is_reveal = action < self.hint_size
            state = jax.lax.cond(is_reveal, _reveal_hint,
                                 _place_hint, state, action)
            hint_terminal = state.hints_placed.all()
            return state.replace(hints_terminal=hint_terminal)

        def _execute_steps(state, aidx, action):
            return jax.lax.cond(
                action == 0,  # Is no action
                lambda s, aidx, action: state,
                lambda s, aidx, action: jax.lax.switch(
                    state.substep,
                    [
                        _step_agent_obs,
                        _step_agent_obs,
                        _step_agent_move,
                        _step_agent_hint,
                    ],
                    s, aidx, action
                ),
                state, aidx, action
            )

        state = jax.lax.cond(
            (action == self.action_dim - 1),
            lambda state, aidx, move:
                state.replace(player_terminal=True),
            _execute_steps,
            state, aidx, action
        )
        return state

    @partial(jax.jit, static_argnums=[0])
    def calculate_reward(
        self, state: State, num_clustered_prev: int, num_hints_correct_prev: int, num_hints_incorrect_prev: int
    ) -> tuple[int, int]:
        """Calculate the reward for the current state.

        Args:
            state (State): The current state of the game
            num_clustered_prev (int): The number of clustered yokai cards in the previous state
            num_hints_correct_prev (int): The number of correctly placed hints in the previous state

        Returns:
            Tuple[int, int]: The reward and the intermediate reward
        """

        def _get_intermediate_obs_reward(state):
            # current_cards_revealed
            aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            card_previously_seen = state.card_previously_seen.at[aidx].get()
            current_cards_revealed = state.current_cards_revealed
            new_card_previously_seen = jnp.logical_or(
                card_previously_seen, current_cards_revealed).astype(jnp.uint8)
            # Get a reward for opening up a new card
            r = (new_card_previously_seen.sum() -
                 card_previously_seen.sum()).astype(jnp.int32)
            return r / self.deck_size

        def _get_intermediate_move_reward(state):
            return (state.num_clustered-num_clustered_prev).astype(jnp.float32)

        def _get_intermediate_hint_reward(state):
            # 1 point if hint was placed correctly
            correct_new_hint = state.corr_hints - num_hints_correct_prev
            # -1 point if hint was placed incorrectly
            incorrect_new_hint = state.incorr_hints - num_hints_incorrect_prev
            # Between -1, 0, 1
            return (correct_new_hint - incorrect_new_hint).astype(jnp.float32)

        def _get_intermediate_reward(state):
            return jax.lax.switch(state.substep, [
                lambda state: _get_intermediate_obs_reward(state),
                lambda state: _get_intermediate_obs_reward(state),
                lambda state: _get_intermediate_move_reward(state),
                lambda state: _get_intermediate_hint_reward(state),
            ], state)

        def _not_end(state):
            # if the game does not ends, only get the intermediate rewards
            return 0, _get_intermediate_reward(state)

        # if the game ends, get the final rewards
        def _end(state):
            def _not_cluster(state):
                # Since the game ends, the player should be punished for not
                # clustering the colors. It is important that ending the game
                # with all colors clustered is the best outcome always
                # since there might be negative rewards for ending the game
                # correctly (i.e. many wrong hints) we need to make sure that
                # players always revieve less reward if they fail the task altogether
                # 1) Wrong terminal should be punished
                wrong_terminal = jax.lax.cond(
                    state.player_terminal, lambda: -1, lambda: 0)
                # 2) Additionally, if the game ends, the player should be punished for not clustering the colors
                missing_cluster = self.num_colors - state.num_clustered
                # 3) Finally all misplaced hints should also be punished
                return 0, wrong_terminal - missing_cluster - state.incorr_hints

            def _calculate_reward(state):
                # Idea is:
                # 1 point per correctly placed hint card
                # -1 for every wrong one
                # 2 points for every card that is opened but not palced
                # 5 points for every card that has not been opened
                card_ids = jnp.arange(self.deck_size)
                num_non_revelead = self.hint_size - state.hints_revealed.sum()
                num_placed = state.hints_placed.sum()
                num_not_placed = self.hint_size - num_placed
                num_not_placed_not_revelaed = num_not_placed - num_non_revelead
                num_correctly_placed = state.hints_color_on_card.at[
                    card_ids, state.colors].get().sum()
                num_incorrectly_placed = num_placed - num_correctly_placed
                final_r = (5 * num_non_revelead
                           + 2 * num_not_placed_not_revelaed
                           + num_correctly_placed
                           - num_incorrectly_placed)
                return final_r, 0

            ir = _get_intermediate_reward(state)
            are_all_clustered = state.num_clustered == self.num_colors

            reward, ir_end = lax.cond(
                are_all_clustered,
                _calculate_reward,  # All clustered, get real rewards
                _not_cluster,       # Not clustered, possibly get intermediate reward
                state
            )

            return reward, ir + ir_end

        is_end = self.terminal(state)
        reward, ir = lax.cond(is_end, _end, _not_end, state)
        return reward, ir

    def terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        return jnp.logical_or(state.hints_terminal, state.player_terminal)

    def observation_space(self, agent: str):
        """ Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """ Action space for a given agent."""
        return self.action_spaces[agent]


######################################################################
########### PREDEFINED ENVIRONMENTS with 3x3 and 4x4 colors ##########
######################################################################


class YokaiGrid9(YokaiGrid):
    """3x3 with 3 colors in a 9x9 grid."""

    def __init__(
        self,
        num_agents=2,
        obs_kind="graph",
        all_colors_revealed: bool = False,
        past_colors_revealed: bool = False,
        color_probs_calculated: bool = False,
        sad: bool = False,
        encode_actions: bool = False,
    ):
        if num_agents == 2:
            num_hint_of_color_num = [1, 3]
        elif num_agents == 3:
            num_hint_of_color_num = [2, 3]
        elif num_agents == 4:
            num_hint_of_color_num = [3, 3]
        else:
            raise ValueError("Only 2-4 players are supported.")
        self.num_hint_of_color_1_num = num_hint_of_color_num[0]
        self.num_hint_of_color_2_num = num_hint_of_color_num[1]
        self.max_possible_steps = sum(num_hint_of_color_num)*2*4
        super().__init__(
            num_agents=num_agents,
            num_colors=3,
            max_hint_of_color_num=np.array([3, 3]),
            deck_size=3*3,
            num_hint_of_color_num=np.array(num_hint_of_color_num),
            obs_kind=obs_kind,
            grid_size=9,
            all_colors_revealed=all_colors_revealed,
            past_colors_revealed=past_colors_revealed,
            color_probs_calculated=color_probs_calculated,
            sad=sad,
            encode_actions=encode_actions,
        )

    def get_initial_hints(self, key):
        key, key_hint_1, key_hint_2 = jax.random.split(key, 3)
        hint1 = jax.random.choice(key_hint_1, self.hint_card_total1, shape=(
            self.num_hint_of_color_1_num,), replace=False)
        hint2 = jax.random.choice(key_hint_2, self.hint_card_total2, shape=(
            self.num_hint_of_color_2_num,), replace=False)
        hints = jnp.vstack((hint1, hint2))
        return hints


class YokaiGrid16(YokaiGrid):
    """4x4 with 4 colors in a 10x10 grid."""

    def __init__(
        self,
        num_agents=2,
        obs_kind="graph",
        all_colors_revealed: bool = False,
        past_colors_revealed: bool = False,
        color_probs_calculated: bool = False,
        sad: bool = False,
        encode_actions: bool = False,
    ):
        num_hint_of_color_num = [2, 3, 2]
        self.num_hint_of_color_1_num = num_hint_of_color_num[0]
        self.num_hint_of_color_2_num = num_hint_of_color_num[1]
        self.num_hint_of_color_3_num = num_hint_of_color_num[2]
        self.max_possible_steps = sum(num_hint_of_color_num)*2*4
        super().__init__(
            num_agents=num_agents,
            num_colors=4,
            max_hint_of_color_num=np.array([4, 6, 4]),
            deck_size=4*4,
            num_hint_of_color_num=np.array(num_hint_of_color_num),
            obs_kind=obs_kind,
            grid_size=10,
            all_colors_revealed=all_colors_revealed,
            past_colors_revealed=past_colors_revealed,
            color_probs_calculated=color_probs_calculated,
            sad=sad,
            encode_actions=encode_actions,
        )

    def get_initial_hints(self, key):
        key, key_hint_1, key_hint_2, key_hint_3 = jax.random.split(key, 4)
        hint1 = jax.random.choice(key_hint_1, self.hint_card_total1, shape=(
            self.num_hint_of_color_1_num,), replace=False)
        hint2 = jax.random.choice(key_hint_2, self.hint_card_total2, shape=(
            self.num_hint_of_color_2_num,), replace=False)
        hint3 = jax.random.choice(key_hint_3, self.hint_card_total3, shape=(
            self.num_hint_of_color_3_num,), replace=False)
        hints = jnp.vstack((hint1, hint2, hint3))
        return hints
