"""Stable public API for compiling and evaluating generated joint policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from llm_policy.context.environment import (
    ResolvedEnvironment,
    load_environment_config,
    make_resolved_env,
    resolved_runtime_contract,
)
from llm_policy.evaluation.rollout import (
    rollout_roles,
    cross_play_matrix_rollout,
    cross_play_rollout,
    self_play_rollout,
)
from llm_policy.policies.runtime import CompiledJointPolicy, CompiledRolePolicy, compile_policy_module
from llm_policy.policies.types import ProgramRolloutResult


@dataclass(frozen=True)
class EvaluationTask:
    """Resolved environment configuration and the exact instantiated task."""

    environment: ResolvedEnvironment
    raw_env: Any
    sample_observation: Any
    primitive_mode: str = "provided"
    restrict_introspection: bool = False
    generation_mode: str = "iterative"

    @property
    def max_steps(self):
        return int(self.raw_env.max_steps)


def evaluation_env(task):
    if getattr(task.raw_env, "policy_adapter", None) in ("hanabi", "yokai", "cat_dog"):
        return task.raw_env  # These adapters provide generic episode logging.
    return OvercookedV2LogWrapper(task.raw_env, replace_info=False)


def _validate_completed_result(task, result):
    if getattr(task.raw_env, "policy_adapter", None) in ("hanabi", "yokai", "cat_dog"):
        if not np.isfinite(np.asarray(result.first_returns)).all():
            raise ValueError(f"{task.raw_env.policy_adapter} evaluation invalid: illegal active action or non-finite episode return")
        if not np.asarray(result.captured).all():
            raise ValueError(f"{task.raw_env.policy_adapter} evaluation incomplete: episode did not terminate within evaluation.max_steps")
    return result


def load_task(config_path: str | Path) -> EvaluationTask:
    """Load the exact task described by one repository environment config."""
    environment = load_environment_config(config_path)
    raw_env = make_resolved_env(environment)
    observations, _ = raw_env.reset(jax.random.PRNGKey(0))
    from llm_policy.context.modes import task_profile
    profile = task_profile(Path(config_path).resolve().parent.parent.parent)
    if environment.env_name.lower() == "cat_dog":
        from llm_policy.context.modes import ContextProfile

        profile_path = Path(config_path).resolve().parent.parent / "context_profile.json"
        if not profile_path.is_file():
            profile = ContextProfile()
        profile.validate_environment(environment)
    import json
    generation_path = Path(config_path).resolve().parent.parent / "generation.json"
    generation_mode = json.loads(generation_path.read_text())["mode"] if generation_path.is_file() else "iterative"
    return EvaluationTask(
        environment=environment,
        raw_env=raw_env,
        sample_observation=observations[raw_env.agents[0]],
        primitive_mode=profile.primitive_mode,
        restrict_introspection=profile.restricted,
        generation_mode=generation_mode,
    )


def compile_candidate(policy_path, task):
    from llm_policy.search.provenance import register_candidate, evaluation
    ref = register_candidate(policy_path, explicit=False, origin="compile")
    frozen = ref["source_path"] if ref else policy_path
    with evaluation([ref], "compile", task=task):
        return _compile_candidate(frozen, task)


def _compile_candidate(
    policy_path: str | Path,
    task: EvaluationTask,
) -> CompiledJointPolicy:
    """Validate and JIT-compile a policy against the task's real observation."""
    from llm_policy.policies.bundle import SOURCE_MODE, source_bundle
    if source_bundle(policy_path) and task.generation_mode != SOURCE_MODE:
        raise ValueError("source-aware bundle requires generation_mode=open_source_zsc")
    if task.generation_mode == SOURCE_MODE:
        from llm_policy.policies.source_aware import compile_source_bundle
        from llm_policy.evaluation.commands import _evaluator_hash
        import hashlib
        identity = hashlib.sha256((_evaluator_hash(task) +
            hashlib.sha256(task.environment.config_path.read_bytes()).hexdigest()).encode()).hexdigest()
        return compile_source_bundle(policy_path, task, identity)
    return compile_policy_module(
        policy_path,
        task.sample_observation,
        expected_metadata=resolved_runtime_contract(task.raw_env),
        primitive_mode=task.primitive_mode,
        restrict_introspection=task.restrict_introspection,
    )


def run_role_pair(
    *, task: EvaluationTask, role_agent_0: CompiledRolePolicy,
    role_agent_1: CompiledRolePolicy, root_seed: int, num_episodes: int,
    num_steps: int | None = None,
) -> ProgramRolloutResult:
    """Evaluate a pair of compiled policy seats."""
    if len(task.raw_env.agents) != 2:
        raise ValueError("run_role_pair is available only for two-player tasks")
    result = rollout_roles(
        jax.random.PRNGKey(int(root_seed)), role_agent_0, role_agent_1,
        evaluation_env(task), num_envs=int(num_episodes),
        num_steps=task.max_steps if num_steps is None else int(num_steps),
    )
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
    return _validate_completed_result(task, result)


from llm_policy.search.provenance import tracked_rollout


@tracked_rollout
def run_self_play(
    *,
    task: EvaluationTask,
    policy: CompiledJointPolicy,
    root_seed: int,
    num_episodes: int,
    num_steps: int | None = None,
    capture_traces: bool = False,
    partner_source: str = "exchange",
) -> ProgramRolloutResult:
    """Run one joint policy over independently reset, vmapped episodes.

    Traces are always produced by the compiled rollout contract.  The
    ``capture_traces`` argument is retained in the public API so a future
    memory-optimized rollout can disable them without changing callers.
    """
    del capture_traces
    from llm_policy.policies.source_aware import SourceAwarePolicy, run_source_pair
    if isinstance(policy, SourceAwarePolicy):
        return run_source_pair(task, policy, policy, root_seed=root_seed, num_episodes=num_episodes,
                               num_steps=num_steps, partner_source=partner_source)[0]
    if partner_source != "exchange":
        raise ValueError("null partner source is only available in open_source_zsc")
    env = evaluation_env(task)
    result = self_play_rollout(
        jax.random.PRNGKey(int(root_seed)),
        policy,
        env,
        num_envs=int(num_episodes),
        num_steps=task.max_steps if num_steps is None else int(num_steps),
    )
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
    return _validate_completed_result(task, result)


@tracked_rollout
def run_cross_play(
    *,
    task: EvaluationTask,
    policy_agent_0: CompiledJointPolicy,
    policy_agent_1: CompiledJointPolicy,
    root_seed: int,
    num_episodes: int,
    num_steps: int | None = None,
    capture_traces: bool = False,
    partner_source: str = "exchange",
) -> ProgramRolloutResult:
    """Pair role 0 from one policy with all remaining roles from another.

    For a three-player task this is the A/BB assignment: the first policy
    supplies agent 0, while the second supplies agents 1 and 2.

    Passing the same ``root_seed`` to every ordered population pairing gives
    each cell the same environment-reset batch and role-specific random keys.
    This common-random-numbers convention makes matrix comparisons less noisy.
    """
    del capture_traces
    from llm_policy.policies.source_aware import SourceAwarePolicy, run_source_pair
    if isinstance(policy_agent_0, SourceAwarePolicy) or isinstance(policy_agent_1, SourceAwarePolicy):
        return run_source_pair(task, policy_agent_0, policy_agent_1, root_seed=root_seed,
                               num_episodes=num_episodes, num_steps=num_steps,
                               partner_source=partner_source)[0]
    if partner_source != "exchange":
        raise ValueError("null partner source is only available in open_source_zsc")
    env = evaluation_env(task)
    result = cross_play_rollout(
        jax.random.PRNGKey(int(root_seed)),
        policy_agent_0,
        policy_agent_1,
        env,
        num_envs=int(num_episodes),
        num_steps=task.max_steps if num_steps is None else int(num_steps),
    )
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
    return _validate_completed_result(task, result)


def run_cross_play_matrix(
    *,
    task: EvaluationTask,
    policies: tuple[CompiledJointPolicy, ...],
    root_seeds: np.ndarray,
    num_episodes: int,
    num_steps: int | None = None,
):
    """Run every directed population pairing in one matrix-wide rollout."""
    env = evaluation_env(task)
    result = cross_play_matrix_rollout(
        root_seeds,
        policies,
        env,
        num_envs=int(num_episodes),
        num_steps=task.max_steps if num_steps is None else int(num_steps),
    )
    return _validate_completed_result(task, result)


from llm_policy.execution.public_analysis import (
    select_episode, summarize_episode_trace, summarize_rollout,
    summarize_episode_metrics, summarize_return_batch,
)


__all__ = [
    "EvaluationTask",
    "compile_candidate",
    "load_task",
    "run_cross_play",
    "run_cross_play_matrix",
    "run_self_play",
    "run_role_pair",
    "select_episode",
    "summarize_episode_trace",
    "summarize_rollout",
    "summarize_episode_metrics",
    "summarize_return_batch",
]
