"""Trusted Yokai adapter: native observations, terminal scoring and accounting."""

import hashlib
import importlib
import inspect
import json
from functools import partial
from pathlib import Path

import chex
import jax
import jax.numpy as jnp

from llm_policy.environments.yokai.native import YokaiGrid9, YokaiGrid16
from llm_policy.policies.api import YokaiObservation


DIMENSIONS = dict(num_agents=2, num_colors=3, deck_size=9, grid_size=9,
                  num_hints=4, num_nodes=13, node_features=13, num_actions=1068)
DIMENSIONS_BY_DECK = {9: DIMENSIONS,
    16: dict(num_agents=2, num_colors=4, deck_size=16, grid_size=10,
             num_hints=7, num_nodes=23, node_features=15, num_actions=2325)}
SOURCE_MODULES = ("llm_policy.environments.yokai.native.constants", "llm_policy.environments.yokai.native.yokai_grid",
                  "jaxmarl.environments.multi_agent_env", "jaxmarl.environments.spaces",
                  "llm_policy.environments.yokai.primitives", "llm_policy.environments.yokai.primitives16",
                  "llm_policy.environments.yokai.adapter")


def source_hashes():
    result = {name: hashlib.sha256(inspect.getsource(importlib.import_module(name)).encode()).hexdigest()
              for name in SOURCE_MODULES}
    directory = Path(inspect.getfile(YokaiGrid9)).parent
    result["UPSTREAM.json"] = hashlib.sha256((directory / "UPSTREAM.json").read_bytes()).hexdigest()
    return result


@chex.dataclass
class YokaiAdapterState:
    env_state: object
    episode_length: jax.Array
    invalid_action: jax.Array


class YokaiAdapter:
    policy_adapter = "yokai"
    episode_metric_names = ("upstream_sparse_return", "terminal_penalty_return",
                            "success", "ended_early", "episode_length")

    def __init__(self, environment):
        variants = {"yokai_grid_9_game": (YokaiGrid9, 9),
                    "yokai_grid_16_game": (YokaiGrid16, 16)}
        if environment.env_name.lower() not in variants:
            raise ValueError("unsupported Yokai environment")
        native_class, deck_size = variants[environment.env_name.lower()]
        self.dimensions = DIMENSIONS_BY_DECK[deck_size]
        native_steps = self.dimensions["num_hints"] * 8
        kwargs = dict(environment.env_kwargs)
        defaults = dict(num_agents=2, obs_kind="graph", past_colors_revealed=True,
                        all_colors_revealed=False, color_probs_calculated=False,
                        sad=False, encode_actions=False)
        if set(kwargs) - set(defaults):
            raise ValueError("unsupported Yokai constructor overrides")
        for name, default in defaults.items():
            value = kwargs.get(name, default)
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"Yokai {name} must be boolean")
            if name != "past_colors_revealed" and (value != default or type(value) is not type(default)):
                raise ValueError(f"Yokai v1 requires {name}={default!r}")
            kwargs[name] = value
        options = environment.config.get("llm_policy", {})
        if set(options) - {"policy_structure", "reward_mode"}:
            raise ValueError("unknown Yokai policy options")
        self.policy_structure = options.get("policy_structure", "shared_local")
        if self.policy_structure != "shared_local":
            raise ValueError("Yokai v1 requires policy_structure=shared_local")
        self.reward_mode = options.get("reward_mode", "upstream_sparse")
        if self.reward_mode not in ("upstream_sparse", "terminal_penalty"):
            raise ValueError("Yokai reward_mode must be upstream_sparse or terminal_penalty")
        cap = environment.config.get("evaluation", {}).get("max_steps", native_steps)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < native_steps:
            raise ValueError(f"Yokai evaluation.max_steps must be an integer >= {native_steps}")
        self.max_steps = cap
        self.past_colors_revealed = kwargs["past_colors_revealed"]
        self.native_env = native_class(**kwargs)
        self.agents = self.native_env.agents
        self.num_agents = 2
        if (self.native_env.action_dim != self.dimensions["num_actions"]
                or self.native_env.max_possible_steps != native_steps):
            raise ValueError("Yokai native dimensions changed")

    def runtime_contract(self):
        hints = self.dimensions["num_hints"]
        return dict(adapter="yokai", adapter_version=1, **self.dimensions,
            policy_structure=self.policy_structure, obs_kind="graph",
            past_colors_revealed=self.past_colors_revealed,
            memory_mode="perfect" if self.past_colors_revealed else "standard",
            reward_mode=self.reward_mode, max_steps=self.max_steps,
            score_min=-hints if self.reward_mode == "upstream_sparse" else -(self.dimensions["num_colors"] + hints),
            score_max=5 * hints,
            upstream_commit=json.loads((Path(inspect.getfile(YokaiGrid9)).parent / "UPSTREAM.json").read_text())["commit"],
            source_sha256=source_hashes())

    def _observations(self, observations, state):
        masks = self.native_env.get_legal_moves(state)
        return {a: YokaiObservation(**observations[a], legal_actions=masks[a].astype(jnp.bool_))
                for a in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        observations, state = self.native_env.reset(key)
        return self._observations(observations, state), YokaiAdapterState(
            env_state=state, episode_length=jnp.asarray(0, jnp.int32), invalid_action=jnp.asarray(False))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions):
        masks = self.native_env.get_legal_moves(state.env_state)
        invalid = state.invalid_action
        safe = {}
        for index, agent in enumerate(self.agents):
            mask = masks[agent].astype(jnp.bool_)
            active = state.env_state.cur_player_idx[index].astype(jnp.bool_)
            action = actions[agent]
            num_actions = self.dimensions["num_actions"]
            legal = (action >= 0) & (action < num_actions) & mask[jnp.clip(action, 0, num_actions - 1)]
            invalid = invalid | (active & ~legal)
            safe[agent] = jnp.where(active, jnp.where(legal, action, jnp.argmax(mask)), 0).astype(jnp.int32)

        # Match MultiAgentEnv.step's key split; capture terminal data BEFORE reset.
        step_key, reset_key = jax.random.split(key)
        obs, terminal, native_rewards, dones, _ = self.native_env.step_env(step_key, state.env_state, safe)
        done = dones["__all__"]
        length = state.episode_length + 1
        success = terminal.num_clustered == self.dimensions["num_colors"]
        sparse = native_rewards["__all__"].astype(jnp.float32)
        penalty = (-terminal.player_terminal.astype(jnp.int32)
                   - (self.dimensions["num_colors"] - terminal.num_clustered) - terminal.incorr_hints).astype(jnp.float32)
        penalized = jnp.where(done & ~success, penalty, sparse)
        selected = sparse if self.reward_mode == "upstream_sparse" else penalized
        selected = jnp.where(invalid, jnp.nan, selected)
        metrics = dict(upstream_sparse_return=jnp.where(invalid, jnp.nan, sparse),
            terminal_penalty_return=jnp.where(invalid, jnp.nan, penalized),
            success=success.astype(jnp.float32), ended_early=(length < self.native_env.max_possible_steps).astype(jnp.float32),
            episode_length=length.astype(jnp.float32))
        info = dict(returned_episode=jnp.full((2,), done),
            returned_episode_returns=jnp.full((2,), selected),
            returned_episode_lengths=jnp.full((2,), length), episode_metrics=metrics)
        reset_obs, reset_state = self.native_env.reset(reset_key)
        raw = jax.tree_util.tree_map(lambda new, old: jax.lax.select(done, new, old), reset_state, terminal)
        obs = jax.tree_util.tree_map(lambda new, old: jax.lax.select(done, new, old), reset_obs, obs)
        next_state = YokaiAdapterState(env_state=raw,
            episode_length=jnp.where(done, 0, length), invalid_action=invalid & ~done)
        rewards = {a: selected for a in (*self.agents, "__all__")}
        return self._observations(obs, raw), next_state, rewards, dones, info
