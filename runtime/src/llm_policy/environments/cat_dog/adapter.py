"""Trusted Cat & Dog observations, action validation and episode accounting."""

import hashlib
import inspect
from functools import partial

import chex
import jax
import jax.numpy as jnp

from llm_policy.environments.cat_dog.native import CatDog
from llm_policy.policies.api import CatDogObservation


DIMENSIONS = {"num_agents": 2, "observation_size": 10, "num_actions": 4}


@chex.dataclass
class CatDogAdapterState:
    env_state: object
    episode_return: jax.Array
    episode_length: jax.Array
    invalid_action: jax.Array


class CatDogAdapter:
    policy_adapter = "cat_dog"

    def __init__(self, environment):
        options = environment.config.get("llm_policy", {})
        if set(options) - {"policy_structure"}:
            raise ValueError("unknown Cat & Dog policy options")
        if options.get("policy_structure", "separate_roles") != "separate_roles":
            raise ValueError("Cat & Dog v1 requires policy_structure=separate_roles")
        if environment.config.get("evaluation", {}).get("max_steps", 2) != 2:
            raise ValueError("Cat & Dog v1 requires evaluation.max_steps=2")
        self.native_env = CatDog(**environment.env_kwargs)
        self.agents = ["agent_0", "agent_1"]
        self.num_agents = 2
        self.max_steps = 2

    def runtime_contract(self):
        return {"adapter": self.policy_adapter, "adapter_version": 1, **DIMENSIONS,
                "policy_structure": "separate_roles", "max_steps": self.max_steps,
                "source_sha256": source_hashes()}

    def _observations(self, native_obs, state):
        masks = self.native_env.get_avail_actions(state)
        return {role: CatDogObservation(features=native_obs[native_agent],
                                       legal_actions=masks[native_agent].astype(jnp.bool_))
                for role, native_agent in zip(self.agents, self.native_env.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, state = self.native_env.reset(key)
        return self._observations(obs, state), CatDogAdapterState(
            env_state=state, episode_return=jnp.asarray(0., jnp.float32),
            episode_length=jnp.asarray(0, jnp.int32), invalid_action=jnp.asarray(False))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions):
        masks = self.native_env.get_avail_actions(state.env_state)
        invalid = state.invalid_action
        safe = {}
        for index, agent in enumerate(self.agents):
            native_agent = self.native_env.agents[index]
            active = (state.env_state.step_count == index) & ~state.env_state.done
            mask = masks[native_agent].astype(jnp.bool_)
            requested = actions[agent]
            legal = (requested >= 0) & (requested < 4) & mask[jnp.clip(requested, 0, 3)]
            invalid = invalid | (active & ~legal)
            safe[native_agent] = jnp.where(active & legal, requested, jnp.argmax(mask)).astype(jnp.int32)
        obs, raw, rewards, dones, info = self.native_env.step(key, state.env_state, safe)
        done = dones["__all__"]
        reward = rewards[self.native_env.agents[0]]
        total = state.episode_return + reward
        length = state.episode_length + 1
        # Hidden native diagnostics (including the pet) never enter public info.
        info = {"returned_episode": jnp.full((2,), done),
                "returned_episode_returns": jnp.full((2,), jnp.where(invalid, jnp.nan, total)),
                "returned_episode_lengths": jnp.full((2,), length)}
        next_state = CatDogAdapterState(
            env_state=raw, episode_return=jnp.where(done, 0., total),
            episode_length=jnp.where(done, 0, length), invalid_action=invalid & ~done)
        return (self._observations(obs, raw), next_state,
                {agent: reward for agent in self.agents},
                {**{agent: done for agent in self.agents}, "__all__": done}, info)


def source_hashes():
    from llm_policy.environments.cat_dog import native

    modules = (native, inspect.getmodule(CatDogAdapter))
    return {module.__name__: hashlib.sha256(inspect.getsource(module).encode()).hexdigest()
            for module in modules}
