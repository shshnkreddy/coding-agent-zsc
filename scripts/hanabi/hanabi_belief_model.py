import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import functools

# DIMENSION KEYS:
# ABC_ means either pytree with all leaves having shape ABC, or a jnp.ndarray whose shape begins with ABC
# T: NUM_STEPS
# K: k
# E: NUM_ENVS
# P: NUM_PLAYERS
# B: BATCH_SIZE = NUM_ACTORS = E * P
# O: OBSERVATION_SPACE_SIZE
# A: ACTION_SPACE_SIZE
# H: HAND_SIZE
# C: NUM_COLORS
# R: NUM_RANKS
# X: C * R
# Y: H * C * R
# Z: (P - 1) * H * C * R
# D: DECK_SIZE
# M: MINIBATCH_SIZE
# W: LAYER_WIDTH
# L: NUM_RNN_LAYERS

class LSTMEncoder(nn.Module):
    layer_width: int = 512
    num_ff_layers: int = 2
    num_rnn_layers: int = 2

    @nn.compact
    def __call__(self, hidden_LBW_, x):
        obs_BO, done_B = x
        y_BW = nn.relu(nn.Dense(self.layer_width, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs_BO))
        for layer in range(1, self.num_ff_layers):
            y_BW = nn.relu(nn.Dense(self.layer_width, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y_BW))

        hidden_LBW_ = tuple(
            jnp.where(done_B[:, None], self.initialize_hidden_state(self.num_rnn_layers, y_BW.shape[0], self.layer_width)[i], hidden_LBW_[i])
            for i in range(len(hidden_LBW_))
        )
        cs_LBW, hs_LBW = hidden_LBW_
        new_cs_LBW, new_hs_LBW = jnp.zeros_like(cs_LBW), jnp.zeros_like(hs_LBW)

        for layer in range(self.num_rnn_layers):
            (new_c_BW,  new_h_BW), y_BW = nn.LSTMCell(features=self.layer_width)(
                (cs_LBW[layer], hs_LBW[layer]), y_BW
            )
            new_cs_LBW = new_cs_LBW.at[layer].set(new_c_BW)
            new_hs_LBW = new_hs_LBW.at[layer].set(new_h_BW)
            
        return (new_cs_LBW, new_hs_LBW), y_BW

    @staticmethod
    def initialize_hidden_state(num_rnn_layers, batch_size, layer_width):
        single_LBW = jnp.zeros((num_rnn_layers, batch_size, layer_width))
        return (single_LBW, single_LBW)

class LSTMDecoder(nn.Module):
    card_embed_size: int = 64
    layer_width: int = 512
    num_colors: int = 5
    num_ranks: int = 5
    hand_size: int = 5
    num_ff_layers: int = 0
    count_based: bool = False

    @nn.compact
    def sample_card(self, rng, hidden_BW_, previous_card_BX, encoded_state_BW, legal_cards_BX):
        probabilities_new_card_count_BX = legal_cards_BX / (legal_cards_BX.sum(axis=-1)[..., None] + 1e-10)
        logits_new_card_count_BX = jnp.log(probabilities_new_card_count_BX + 1e-10)
        if self.count_based:
            logits_new_card_BX = logits_new_card_count_BX
            new_hidden_BW_ = hidden_BW_
        else:
            encoded_previous_card_B_ = nn.Dense(self.card_embed_size, use_bias=False)(previous_card_BX)
            input_to_net_B_ = jnp.concatenate((encoded_previous_card_B_, encoded_state_BW), axis=-1)
            new_hidden_BW_, y_BW = nn.LSTMCell(self.layer_width)(hidden_BW_, input_to_net_B_)
            for layer in range(self.num_ff_layers):
                y_BW = nn.relu(nn.Dense(self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0))(y_BW))
            logits_new_card_BX = nn.Dense(self.num_colors * self.num_ranks)(y_BW)
        
        logits_new_card_legal_BX = logits_new_card_BX - 1e10 * (legal_cards_BX == 0)

        rng, _rng = jax.random.split(rng)
        sampled_card_B = jax.random.categorical(_rng, logits_new_card_legal_BX, axis=-1)
        sampled_card_BX = jax.nn.one_hot(sampled_card_B, self.num_colors * self.num_ranks)
        sampled_card_BCR = sampled_card_BX.reshape((sampled_card_BX.shape[0], self.num_colors, self.num_ranks))
        return new_hidden_BW_, sampled_card_BCR, logits_new_card_count_BX, logits_new_card_legal_BX

    # THIS SCAN IS OVER THE CARD DIMENSION IN EACH HAND
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    def __call__(self, carry, step):
        rng, hidden_BW_, remaining_cards_BX = carry
        previous_card_BX, encoded_state_BW, card_knowledge_BX = step

        new_remaining_cards_BX = remaining_cards_BX - previous_card_BX
        legal_cards_BX = new_remaining_cards_BX * card_knowledge_BX

        rng, _rng = jax.random.split(rng)
        new_hidden_BW_, sampled_card_BCR, logits_new_card_count_BX, logits_new_card_legal_BX = self.sample_card(
            _rng, hidden_BW_, previous_card_BX, encoded_state_BW, legal_cards_BX,
        )

        return (
            rng, 
            new_hidden_BW_,
            new_remaining_cards_BX,
        ), (
            logits_new_card_count_BX,
            logits_new_card_legal_BX,
        )

    # THIS SCAN IS OVER THE CARD DIMENSION IN EACH HAND
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    # HERE WE SAMPLE AUTOREGRESSIVELY, WITHOUT USING TEACHER FORCING
    def ar_sample(self, carry, step):
        rng, hidden_EW_, previous_card_EX, remaining_cards_EX, valid_E = carry
        encoded_state_EW, card_knowledge_EX = step

        legal_cards_EX = remaining_cards_EX * card_knowledge_EX

        new_valid_E = valid_E * (legal_cards_EX.sum(axis=-1) > 0)

        rng, _rng = jax.random.split(rng)
        new_hidden_EW_, sampled_card_ECR, logits_new_card_count_EX, logits_new_card_legal_EX = self.sample_card(
            _rng, hidden_EW_, previous_card_EX, encoded_state_EW, legal_cards_EX,
        )

        sampled_card_EX = sampled_card_ECR.reshape((sampled_card_ECR.shape[0], -1,))
        new_remaining_cards_EX = remaining_cards_EX - sampled_card_EX

        return (
            rng,
            new_hidden_EW_,
            sampled_card_EX,
            new_remaining_cards_EX,
            new_valid_E,
        ), (
            sampled_card_ECR, 
            logits_new_card_count_EX,
            logits_new_card_legal_EX,
        )

    @staticmethod
    def initialize_hidden_state(batch_size, layer_width):
        single = jnp.zeros((batch_size, layer_width))
        return (single, single)

class BeliefModel(nn.Module):
    layer_width: int = 512
    num_colors: int = 5
    num_ranks: int = 5
    hand_size: int = 5
    deck_size: int = 50
    num_rnn_layers: int = 2
    num_ff_layers_encoder: int = 2
    num_ff_layers_decoder: int = 0
    count_based: bool = False

    def setup(self):
        self.encoder = LSTMEncoder(
            self.layer_width,
            self.num_ff_layers_encoder,
            self.num_rnn_layers,
        )
        self.decoder = LSTMDecoder(
            self.layer_width // 8, 
            self.layer_width, 
            self.num_colors, 
            self.num_ranks,
            self.hand_size,
            self.num_ff_layers_decoder,
            self.count_based,
        )

    def encode(self, hidden_LBW_, obs_BO, done_B):
        if self.count_based:
            encoded_state_BW = jnp.zeros((obs_BO.shape[0], self.layer_width))
        else:
            hidden_LBW_, encoded_state_BW = self.encoder(hidden_LBW_, (obs_BO, done_B))
        return hidden_LBW_, encoded_state_BW
    
    def decode(self, encoded_state_EW, obs_EO, rng, remaining_cards_EX, card_knowledge_EHX):
        initial_decoder_hidden_EW_ = self.decoder.initialize_hidden_state(obs_EO.shape[0], self.layer_width)
        initial_card_EX = jnp.zeros((obs_EO.shape[0], self.num_colors * self.num_ranks))
        initial_valid_E = jnp.ones(obs_EO.shape[0])
        encoded_state_EHW = jnp.repeat(encoded_state_EW[:, None], self.hand_size, axis=1)
        
        (
            rng,
            _,
            _,
            cards_in_sampled_deck_EX,
            valid_E,
        ), (
            sampled_hand_EHCR,
            logits_new_card_count_EHX,
            logits_new_card_legal_EHX,
        ) = self.decoder.ar_sample(
            (
                rng, 
                initial_decoder_hidden_EW_, 
                initial_card_EX,
                remaining_cards_EX,
                initial_valid_E,
            ),(
                encoded_state_EHW,
                card_knowledge_EHX,
            ),
        )
        
        def _sample_deck(rng, cards_in_sampled_deck_X):
            cards_in_new_deck_indices_D = jnp.repeat(
                jnp.arange(self.num_colors * self.num_ranks + 1),
                jnp.concatenate([cards_in_sampled_deck_X.astype(jnp.int32), jnp.array([0])]),
                total_repeat_length=self.deck_size
            )

            def _index_to_card(idx):
                card_CR = jnp.zeros((self.num_colors, self.num_ranks))
                return jax.lax.cond(
                    idx < self.num_colors * self.num_ranks,
                    lambda : card_CR.at[jnp.unravel_index(idx, (self.num_colors, self.num_ranks))].set(1),
                    lambda : card_CR
                )

            cards_in_new_deck_DCR = jax.vmap(_index_to_card)(cards_in_new_deck_indices_D)
            cards_in_new_deck_DCR = jax.random.permutation(rng, cards_in_new_deck_DCR)

            nonzero_cards_indices_D = 1 * (0 != cards_in_new_deck_DCR.sum(axis=[1, 2]))
            sort_indices_D = jnp.argsort(nonzero_cards_indices_D, stable=True)
            new_deck_DCR = cards_in_new_deck_DCR[sort_indices_D]

            return new_deck_DCR

        rng_E_ = jax.random.split(rng, obs_EO.shape[0])
        sampled_deck_EDCR = jax.vmap(_sample_deck)(rng_E_, cards_in_sampled_deck_EX)
            
        return valid_E, sampled_hand_EHCR, sampled_deck_EDCR, logits_new_card_count_EHX, logits_new_card_legal_EHX

    # THIS SCAN IS OVER THE TIME DIMENSION
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, step): # Technically all the B shapes have to be M shapes, as this function is only called on minibatches
        rng, hidden_LBW_ = carry
        obs_BO, done_B, hand_BY, remaining_cards_BX, card_knowledge_BHX = step

        new_hidden_LBW_, encoded_state_BW = self.encode(hidden_LBW_, obs_BO, done_B)
        encoded_state_BHW = jnp.repeat(encoded_state_BW[:, None], self.hand_size, axis=1)

        initial_decoder_hidden_BW_ = self.decoder.initialize_hidden_state(obs_BO.shape[0], self.layer_width)

        initial_card_BX = jnp.zeros((obs_BO.shape[0], self.num_colors * self.num_ranks))
        hand_BHX = hand_BY.reshape((hand_BY.shape[0], self.hand_size, -1))
        previous_cards_BHX = jnp.concatenate([initial_card_BX[:, None], hand_BHX[:, :-1]], axis=1)

        # AUTOREGRESSIVELY DECODE THE HAND, USING TEACHER FORCING
        (rng, _, _), (logits_new_card_count_BHX, logits_new_card_legal_BHX) = self.decoder(
            (
                rng, 
                initial_decoder_hidden_BW_, 
                remaining_cards_BX
            ), (
                previous_cards_BHX,
                encoded_state_BHW,
                card_knowledge_BHX,
            ),
        )
        return (rng, new_hidden_LBW_), (logits_new_card_count_BHX, logits_new_card_legal_BHX)
    
    @staticmethod
    def initialize_hidden_state(num_rnn_layers, batch_size, layer_width):
        single_LBW = jnp.zeros((num_rnn_layers, batch_size, layer_width))
        return (single_LBW, single_LBW)
    
def sample_hand_from_belief(bm_train_state, rng, env_state_E_, encoded_state_BW, obs_BO, belief_player_idx): # TODO could put all this into BeliefModel.decode??? Or fit decode in here?
    hands_EPHCR = env_state_E_.player_hands
    E, P = hands_EPHCR.shape[:2]
    encoded_state_EW = encoded_state_BW.reshape((P, E, -1)).swapaxes(0, 1)[jnp.arange(E), belief_player_idx]
    obs_EO = obs_BO.reshape((P, E, -1)).swapaxes(0, 1)[jnp.arange(E), belief_player_idx]
    hand_EHCR = hands_EPHCR[jnp.arange(E), belief_player_idx]
    deck_EDCR = env_state_E_.deck
    remaining_cards_EX = deck_EDCR.sum(axis=1).reshape((E, -1)) + hand_EHCR.sum(axis=1).reshape((E, -1))
    card_knowledge_EPHX = env_state_E_.card_knowledge
    card_knowledge_EHX = card_knowledge_EPHX[jnp.arange(E), belief_player_idx]

    valid_E, sampled_hand_EHCR, sampled_deck_EDCR, logits_new_card_count_EHX, logits_new_card_legal_EHX = bm_train_state.apply_fn(
        bm_train_state.params,
        encoded_state_EW, obs_EO, rng, remaining_cards_EX, card_knowledge_EHX,
        method=BeliefModel.decode,
    )
    entropy_count = distrax.Categorical(logits_new_card_count_EHX).entropy().mean()
    entropy_legal = distrax.Categorical(logits_new_card_legal_EHX).entropy().mean()

    valid_E111 = valid_E.reshape(-1, 1, 1, 1)
    hand_to_use_EHCR = hand_EHCR * (1 - valid_E111) + sampled_hand_EHCR * valid_E111
    hands_to_use_EPHCR = hands_EPHCR.at[jnp.arange(E), belief_player_idx].set(hand_to_use_EHCR)
    deck_to_use_EDCR = deck_EDCR * (1 - valid_E111) + sampled_deck_EDCR * valid_E111

    return (
        env_state_E_.replace(player_hands=hands_to_use_EPHCR, deck=deck_to_use_EDCR),
        {
            "valid_percentage_hands": valid_E.mean(),
            "entropy_legal": entropy_legal,
            "entropy_count": entropy_count,
        },
    )