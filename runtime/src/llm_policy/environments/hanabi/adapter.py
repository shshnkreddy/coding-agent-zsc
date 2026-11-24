"""Trusted installed-JaxMARL Hanabi adapter; never imported by policy code."""

import hashlib
import inspect
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.hanabi.hanabi import HanabiEnv

from llm_policy.policies.api import HanabiObservation


from llm_policy.environments.hanabi.layout import hanabi_layout


STANDARD = {"num_agents": 2, "hand_size": 5, "num_colors": 5, "num_ranks": 5,
            "max_info_tokens": 8, "max_life_tokens": 3}


class CorrectedHanabi(HanabiEnv):
    """Minimal correction of 0.1.0's stale final-life terminal flag."""

    @partial(jax.jit, static_argnums=(0,))
    def step_game(self, state, aidx, action):
        next_state, reward = super().step_game(state, aidx, action)
        return next_state.replace(terminal=next_state.terminal | next_state.out_of_lives), reward


@chex.dataclass
class HanabiAdapterState:
    env_state: object
    episode_return: jax.Array
    episode_length: jax.Array
    invalid_action: jax.Array


class HanabiAdapter:
    """Native dynamics + private observation PyTrees + generic episode accounting.

    An illegal active action marks the whole episode invalid. A legal placeholder
    advances the batch only; its return is NaN and is rejected by the public API.
    """

    policy_adapter = "hanabi"

    def __init__(self, environment):
        kwargs = dict(environment.env_kwargs)
        if set(kwargs) - (set(STANDARD) | {"num_cards_of_rank", "num_agents", "hand_size"}):
            raise ValueError("unsupported Hanabi constructor overrides")
        self.layout = hanabi_layout(kwargs.get("num_agents", 2), kwargs.get("hand_size"))
        kwargs.update({k: self.layout[k] for k in ("num_agents", "hand_size")})
        for name, value in STANDARD.items():
            if name in ("num_agents", "hand_size"):
                continue
            if kwargs.get(name, value) != value:
                raise ValueError(f"Hanabi v1 requires {name}={value}")
            kwargs[name] = value
        if list(kwargs.get("num_cards_of_rank", [3, 2, 2, 2, 1])) != [3, 2, 2, 2, 1]:
            raise ValueError("Hanabi v1 requires standard card multiplicities")
        kwargs["num_cards_of_rank"] = np.array([3, 2, 2, 2, 1])
        options = environment.config.get("llm_policy", {})
        if set(options) - {"belief_mode", "fix_final_life_termination", "policy_structure", "convention_profile"}:
            raise ValueError("unknown Hanabi policy options")
        from llm_policy.environments.hanabi.context import resolve_convention_profile
        resolve_convention_profile(environment.config)
        self.policy_structure = options.get("policy_structure", "shared_local")
        if self.policy_structure not in ("shared_local", "separate_roles"):
            raise ValueError("policy_structure must be shared_local or separate_roles")
        if options.get("belief_mode", "exact_literal") != "exact_literal":
            raise ValueError("Hanabi v1 supports belief_mode=exact_literal only")
        self.fix_final_life_termination = options.get("fix_final_life_termination", True)
        if not isinstance(self.fix_final_life_termination, bool):
            raise ValueError("fix_final_life_termination must be a boolean")
        cls = CorrectedHanabi if self.fix_final_life_termination else HanabiEnv
        self.native_env = cls(**kwargs)
        self.episode_metric_names = ()
        self.finesse_enabled = environment.config.get("evaluation", {}).get("finesse", False)
        if not isinstance(self.finesse_enabled, bool):
            raise ValueError("evaluation.finesse must be boolean")
        if self.finesse_enabled:
            from envs.wrappers.hanabi import HanabiFinesseWrapper
            from envs.wrappers.hanabi.finesse import COUNTERS, EXCLUSIONS
            self.native_env = HanabiFinesseWrapper(self.native_env)
            self.finesse_contract = self.native_env.detector_contract()
            self.episode_metric_names = tuple("finesse/" + k for k in COUNTERS) + (
                "finesse/structural_connections",) + tuple("finesse/excluded/" + k for k in EXCLUSIONS)
        self.play_card_knowledge_enabled = environment.config.get("evaluation", {}).get("play_card_knowledge", False)
        if not isinstance(self.play_card_knowledge_enabled, bool):
            raise ValueError("evaluation.play_card_knowledge must be boolean")
        if self.play_card_knowledge_enabled:
            from envs.wrappers.hanabi.play_card_knowledge import HanabiPlayCardKnowledgeWrapper, CATEGORIES
            self.native_env = HanabiPlayCardKnowledgeWrapper(self.native_env)
            self.episode_metric_names += tuple("play_card_knowledge/" + k for k in CATEGORIES) + (
                "play_card_knowledge/plays", "play_card_knowledge/successes", "play_card_knowledge/failures")
        self.agents = self.native_env.agents
        self.num_agents = self.layout["num_agents"]
        cap = environment.config.get("evaluation", {}).get("max_steps", 128)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError("evaluation.max_steps must be a positive integer")
        self.max_steps = cap
        if (int(self.native_env.obs_size) != self.layout["observation_size"]
                or int(self.native_env.num_moves) != self.layout["num_actions"]):
            raise ValueError("installed Hanabi observation/action contract changed")

    def runtime_contract(self):
        contract = {"adapter": "hanabi", "adapter_version": 1, **STANDARD,
                "policy_structure": self.policy_structure,
                "score_min": 0,
                "score_max": int(self.native_env.num_colors * self.native_env.num_ranks),
                **{k: self.layout[k] for k in ("num_agents", "hand_size", "observation_size", "num_actions")},
                "max_steps": self.max_steps,
                "belief_mode": "exact_literal",
                "fix_final_life_termination": self.fix_final_life_termination,
                "source_sha256": source_hashes()}
        if self.finesse_enabled:
            contract["finesse"] = self.finesse_contract
        if self.play_card_knowledge_enabled:
            contract["play_card_knowledge"] = self.native_env.detector_contract()
        return contract

    def unwrap_native_state(self, state):
        """For trusted drivers only; wrapped diagnostics never enter observations."""
        if self.play_card_knowledge_enabled:
            state = state.env_state
        return state.env_state if self.finesse_enabled else state

    def _observations(self, native_obs, state):
        masks = self.native_env.get_legal_moves(state)
        return {a: HanabiObservation(features=native_obs[a],
                                     legal_actions=masks[a].astype(jnp.bool_)) for a in self.agents}

    def _initial(self, obs, state):
        return self._observations(obs, state), HanabiAdapterState(
            env_state=state, episode_return=jnp.asarray(0., jnp.float32),
            episode_length=jnp.asarray(0, jnp.int32), invalid_action=jnp.asarray(False))

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        return self._initial(*self.native_env.reset(key))

    @partial(jax.jit, static_argnums=(0,))
    def reset_from_deck(self, deck):
        """Trusted diagnostic API; never an input to generated policy code."""
        return self._initial(*self.native_env.reset_from_deck(deck))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions):
        masks = self.native_env.get_legal_moves(state.env_state)
        invalid = state.invalid_action
        safe = {}
        noop = self.layout["actions"]["inactive_noop"]
        for a in self.agents:
            mask = masks[a].astype(jnp.bool_)
            active = ~mask[noop]
            requested = actions[a]
            legal = (requested >= 0) & (requested <= noop) & mask[jnp.clip(requested, 0, noop)]
            invalid = invalid | (active & ~legal)
            safe[a] = jnp.where(active, jnp.where(legal, requested, jnp.argmax(mask)), noop).astype(jnp.int32)
        obs, raw, rewards, dones, info = self.native_env.step(key, state.env_state, safe)
        if self.play_card_knowledge_enabled:
            invalid = invalid | info["play_card_knowledge"]["diagnostic_failure"]
        done = dones["__all__"]
        total = state.episode_return + rewards[self.agents[0]]
        length = state.episode_length + 1
        info = dict(info)
        info["returned_episode"] = jnp.full((self.num_agents,), done)
        info["returned_episode_returns"] = jnp.full((self.num_agents,), jnp.where(invalid, jnp.nan, total))
        info["returned_episode_lengths"] = jnp.full((self.num_agents,), length)
        if self.finesse_enabled:
            diagnostic = info["finesse"]
            info["episode_metrics"] = {
                **{"finesse/" + k: v for k, v in diagnostic["counts"].items()},
                "finesse/structural_connections": diagnostic["structural_connections"],
                **{"finesse/excluded/" + k: v for k, v in diagnostic["exclusion_counts"].items()},
            }
        if self.play_card_knowledge_enabled:
            from envs.wrappers.hanabi.play_card_knowledge import CATEGORIES
            diagnostic = info["play_card_knowledge"]
            info.setdefault("episode_metrics", {}).update({
                **{"play_card_knowledge/" + k: diagnostic["counts"][:, i].sum()
                   for i, k in enumerate(CATEGORIES)},
                **{"play_card_knowledge/" + k: diagnostic[k].sum()
                   for k in ("plays", "successes", "failures")},
            })
        next_state = HanabiAdapterState(env_state=raw,
            episode_return=jnp.where(done, 0., total),
            episode_length=jnp.where(done, 0, length), invalid_action=invalid & ~done)
        return self._observations(obs, raw), next_state, rewards, dones, info


def source_hashes():
    from jaxmarl.environments.hanabi import hanabi, hanabi_game
    from llm_policy import hanabi_primitives
    from llm_policy.environments.hanabi import layout
    modules = (hanabi, hanabi_game, hanabi_primitives, layout)
    result = {m.__name__: hashlib.sha256(inspect.getsource(m).encode()).hexdigest() for m in modules}
    result["llm_policy.environments.hanabi.adapter"] = hashlib.sha256(
        inspect.getsource(inspect.getmodule(HanabiAdapter)).encode()).hexdigest()
    return result
