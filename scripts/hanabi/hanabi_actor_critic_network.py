import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Dict
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

class ScannedMultiLayerGRU(nn.Module):
    num_layers: int
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry_LBW, x_B_):
        ins_BW, resets_B = x_B_
        carry_LBW = jnp.where(resets_B[None, :, None], self.initialize_hidden_state(self.num_layers, ins_BW.shape[0], ins_BW.shape[1]), carry_LBW)
        new_carry_LBW = jnp.zeros_like(carry_LBW)
        y_BW = ins_BW

        for layer in range(self.num_layers):
            new_carry_at_layer_BW, y_BW = nn.GRUCell(features=ins_BW.shape[1])(
                carry_LBW[layer], y_BW
            )
            new_carry_LBW = new_carry_LBW.at[layer].set(new_carry_at_layer_BW)

        return new_carry_LBW, y_BW

    @staticmethod
    def initialize_hidden_state(num_rnn_layers, batch_size, layer_width):
        return jnp.zeros((num_rnn_layers, batch_size, layer_width))
    
class ScannedMultiLayerLSTM(nn.Module):
    num_layers: int
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry_LBW_, x_B_):
        ins_BW, resets_B = x_B_
        carry_LBW_ = tuple(
            jnp.where(resets_B[None, :, None], self.initialize_hidden_state(self.num_layers, ins_BW.shape[0], ins_BW.shape[1])[i], carry_LBW_[i])
            for i in range(len(carry_LBW_))
        )
        cs_LBW, hs_LBW = carry_LBW_
        new_cs_LBW, new_hs_LBW = jnp.zeros_like(cs_LBW), jnp.zeros_like(hs_LBW)
        y_BW = ins_BW
        for layer in range(self.num_layers):
            (new_c_BW,  new_h_BW), y_BW = nn.LSTMCell(features=ins_BW.shape[1])(
                (cs_LBW[layer], hs_LBW[layer]), y_BW
            )
            new_cs_LBW = new_cs_LBW.at[layer].set(new_c_BW)
            new_hs_LBW = new_hs_LBW.at[layer].set(new_h_BW)
            
        return (new_cs_LBW, new_hs_LBW), y_BW

    @staticmethod
    def initialize_hidden_state(num_rnn_layers, batch_size, layer_width):
        single = jnp.zeros((num_rnn_layers, batch_size, layer_width))
        return (single, single)

class ActorCritic(nn.Module):
    action_dim: int
    config: Dict

    def setup(self):
        assert self.config["RNN"] in [False, "LSTM", "GRU"]
        NUM_PLAYERS = self.config["ENV_KWARGS"]["num_agents"] if "num_agents" in self.config["ENV_KWARGS"] else 2
        NUM_COLORS = self.config["ENV_KWARGS"]["num_colors"] if "num_colors" in self.config["ENV_KWARGS"] else 5
        NUM_RANKS = self.config["ENV_KWARGS"]["num_ranks"] if "num_ranks" in self.config["ENV_KWARGS"] else 5
        HAND_SIZE = 5 if NUM_PLAYERS < 4 else 4
        self.OBS_PRIVATE_SIZE = (NUM_PLAYERS - 1) * HAND_SIZE * NUM_COLORS * NUM_RANKS

    @nn.compact
    def __call__(self, hidden_state_LBW_, x_TB_):
        obs_TBO, hand_TBY, dones_TB, avail_actions_TBA = x_TB_

        if self.config["TRAIN_BELIEF_MODEL"] and not self.config["ACTOR_CRITIC_FOLDER"]: # Uniformly random policy
            action_logits_TBA = jnp.zeros_like(avail_actions_TBA) - 1e10 * (1 - avail_actions_TBA)
            critic_TB = 1.0 * jnp.zeros_like(dones_TB)
            return hidden_state_LBW_, action_logits_TBA, critic_TB

        if self.config["PUBLIC_RNN"]:
            obs_rnn_TB_ = obs_TBO[..., self.OBS_PRIVATE_SIZE:]
        else:
            obs_rnn_TB_ = obs_TBO

        obs_actor_TBO = obs_TBO

        if self.config["CRITIC_CENTRALIZED"]:
            obs_critic_TB_ = jnp.concatenate([hand_TBY, obs_TBO], axis=-1) # Strictly speaking the last dimension is Y + O
        else:
            obs_critic_TB_ = obs_TBO

        embedding_obs_actor_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs_actor_TBO))
        embedding_obs_critic_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs_critic_TB_))
        for layer in range(self.config["NUM_RNN_LAYERS"]):
            embedding_obs_actor_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding_obs_actor_TBW))
            embedding_obs_critic_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(embedding_obs_critic_TBW))

        if self.config["RNN"] not in ["LSTM", "GRU"]:
            actor_mean_TBW = embedding_obs_actor_TBW
            critic_mean_TBW = embedding_obs_critic_TBW
        else:
            embedding_obs_rnn_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs_rnn_TB_))

            rnn_in_TB_ = (embedding_obs_rnn_TBW, dones_TB)
            if self.config["RNN"] == "LSTM":
                hidden_state_LBW_, embedding_obs_rnn_TBW = ScannedMultiLayerLSTM(self.config["NUM_RNN_LAYERS"])(hidden_state_LBW_, rnn_in_TB_)
            elif self.config["RNN"] == "GRU":
                hidden_state_LBW_, embedding_obs_rnn_TBW = ScannedMultiLayerGRU(self.config["NUM_RNN_LAYERS"])(hidden_state_LBW_, rnn_in_TB_)

            actor_mean_TBW = embedding_obs_rnn_TBW
            critic_mean_TBW = embedding_obs_rnn_TBW
            if self.config["ACTOR_FF_STREAM"]:
                if self.config["MULTIPLY_PUBLIC_PRIVATE"]:
                    actor_mean_TBW = embedding_obs_actor_TBW * actor_mean_TBW
                else:
                    actor_mean_TBW = jnp.concatenate([embedding_obs_actor_TBW, actor_mean_TBW], axis=-1) # Strictly speaking the last dimension is 2 * B
            if self.config["CRITIC_FF_STREAM"]:
                if self.config["MULTIPLY_PUBLIC_PRIVATE"]:
                    critic_mean_TBW = embedding_obs_critic_TBW * critic_mean_TBW
                else:
                    critic_mean_TBW = jnp.concatenate([embedding_obs_critic_TBW, critic_mean_TBW], axis=-1) # Strictly speaking the last dimension is 2 * B

        for layer in range(self.config["NUM_FF_LAYERS"]):
            actor_mean_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(2), bias_init=constant(0.0))(actor_mean_TBW))
            critic_mean_TBW = nn.relu(nn.Dense(self.config["LAYER_WIDTH"], kernel_init=orthogonal(2), bias_init=constant(0.0))(critic_mean_TBW))

        action_logits_TBA = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean_TBW)
        action_logits_TBA = action_logits_TBA - 1e10 * (1 - avail_actions_TBA)

        critic_TB1 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic_mean_TBW)
        critic_TB = critic_TB1.squeeze(-1)

        return hidden_state_LBW_, action_logits_TBA, critic_TB
        
    @staticmethod
    def initialize_hidden_state(rnn, num_rnn_layers, batch_size, layer_width):
        if rnn == "LSTM":
            initial_hidden_state_LBW_ = ScannedMultiLayerLSTM.initialize_hidden_state(num_rnn_layers, batch_size, layer_width)
        elif rnn == "GRU":
            initial_hidden_state_LBW_ = ScannedMultiLayerGRU.initialize_hidden_state(num_rnn_layers, batch_size, layer_width)
        else:
            initial_hidden_state_LBW_ = ScannedMultiLayerGRU.initialize_hidden_state(1, batch_size, 1)
        return initial_hidden_state_LBW_
