"""Minimal local-observation-compatible OvercookedV2 logging wrapper."""

from functools import partial
from typing import Dict

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import State
from jaxmarl.environments.overcooked_v2.common import DynamicObject


class JaxMARLWrapper:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _batchify_floats(self, values):
        return jnp.stack([values[agent] for agent in self._env.agents])


@struct.dataclass
class OvercookedV2LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_episode_recipe_returns: Dict[str, float]


class OvercookedV2LogWrapper(JaxMARLWrapper):
    def __init__(self, env, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info
        self.recipe_dict = {
            f"{recipe[0]}_{recipe[1]}_{recipe[2]}": DynamicObject.get_recipe_encoding(
                recipe
            )
            for recipe in env.possible_recipes
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, env_state = self._env.reset(key)
        recipe_returns = {
            recipe: jnp.zeros((self._env.num_agents,))
            for recipe in self.recipe_dict
        }
        state = OvercookedV2LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            recipe_returns,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: OvercookedV2LogEnvState, action):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        episode_done = done["__all__"]
        new_return = state.episode_returns + self._batchify_floats(reward)
        new_length = state.episode_lengths + 1
        recipe_returns = {
            recipe_id: jax.lax.select(
                (state.env_state.recipe == encoding) & episode_done,
                new_return,
                state.returned_episode_recipe_returns[recipe_id],
            )
            for recipe_id, encoding in self.recipe_dict.items()
        }
        state = OvercookedV2LogEnvState(
            env_state=env_state,
            episode_returns=new_return * (1 - episode_done),
            episode_lengths=new_length * (1 - episode_done),
            returned_episode_returns=jax.lax.select(
                episode_done, new_return, state.returned_episode_returns
            ),
            returned_episode_lengths=jax.lax.select(
                episode_done, new_length, state.returned_episode_lengths
            ),
            returned_episode_recipe_returns=recipe_returns,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full(
            (self._env.num_agents,), episode_done
        )
        info["returned_episode_recipe_returns"] = recipe_returns
        return obs, state, reward, done, info


__all__ = ["OvercookedV2LogEnvState", "OvercookedV2LogWrapper"]
