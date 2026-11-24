"""JIT-compiled self- and cross-play rollouts for generated programs."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from llm_policy.policies.runtime import CompiledJointPolicy, CompiledRolePolicy
from llm_policy.policies.types import ProgramRolloutResult, XPMatrixRolloutResult


def _reshape_leading(tree, consumed_axes: int, new_prefix: tuple[int, ...]):
    return jax.tree_util.tree_map(
        lambda value: jnp.reshape(
            value, new_prefix + tuple(value.shape[int(consumed_axes) :])
        ),
        tree,
    )


def _flatten_leading(tree, num_axes: int):
    return _reshape_leading(tree, num_axes, (-1,))


def _reset_env(env, key, task):
    if task is None:
        return env.reset(key)
    if not hasattr(env, "reset_to_level"):
        raise ValueError("fixed tasks require an environment with reset_to_level()")
    task_obs, task_state = env.reset_to_level(key, task)
    _, logged_state = env.reset(key)
    if not hasattr(logged_state, "env_state") or not hasattr(logged_state, "replace"):
        raise ValueError("fixed tasks require a compatible JaxMARL logging wrapper")
    return task_obs, logged_state.replace(env_state=task_state)


def rollout_roles(
    rng: jax.Array,
    role_agent_0: CompiledRolePolicy,
    role_agent_1: CompiledRolePolicy,
    env: Any,
    *,
    role_agent_2: CompiledRolePolicy | None = None,
    role_agent_3: CompiledRolePolicy | None = None,
    role_agent_4: CompiledRolePolicy | None = None,
    num_envs: int,
    num_steps: int | None = None,
    task: Any = None,
) -> ProgramRolloutResult:
    """Evaluate two through five compiled roles from their local observations.

    Each parallel environment contributes its first episode-end return.
    The traces remain on device with shape ``(num_steps, num_envs, ...)`` so a
    caller can selectively copy failed or interesting trajectories to the host.
    """
    num_envs = int(num_envs)
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if num_steps is None:
        num_steps = int(getattr(env, "max_steps"))
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    agents = tuple(env.agents)
    if len(agents) not in (2, 3, 4, 5):
        raise ValueError("generated rollouts support two through five agents")
    supplied = (role_agent_0, role_agent_1, role_agent_2, role_agent_3, role_agent_4)
    if any(role is None for role in supplied[:len(agents)]) or any(role is not None for role in supplied[len(agents):]):
        raise ValueError("the compiled role count must match the environment")
    roles = supplied[:len(agents)]
    reset_rng, rollout_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    if task is None:
        observations, env_state = jax.vmap(env.reset)(reset_rngs)
    else:
        observations, env_state = jax.vmap(lambda key: _reset_env(env, key, task))(
            reset_rngs
        )

    memories = tuple(role.init(num_envs) for role in roles)
    resets = tuple(jnp.ones((num_envs,), dtype=jnp.bool_) for _ in roles)
    first_returns = jnp.zeros((num_envs,), dtype=jnp.float32)
    captured = jnp.zeros((num_envs,), dtype=jnp.bool_)
    episode_metrics = {name: jnp.zeros((num_envs,), jnp.float32)
                       for name in getattr(env, "episode_metric_names", ())}

    def step(carry, _):
        (
            observations,
            env_state,
            memories,
            resets,
            first_returns,
            captured,
            step_rng,
            episode_metrics,
        ) = carry
        key_parts = jax.random.split(step_rng, len(roles) + 2)
        step_rng = key_parts[0]
        env_rng = key_parts[-1]
        env_rngs = jax.random.split(env_rng, num_envs)
        role_outputs = tuple(
            role.batched_act(
                observations[agent], resets[index], memories[index],
                jax.random.split(key_parts[index + 1], num_envs),
            )
            for index, (role, agent) in enumerate(zip(roles, agents))
        )
        # Only errors in the first captured episode can invalidate its return.
        role_outputs = tuple(
            (action, memory, {**trace, "__source_exchange_invalid_action":
                trace["__source_exchange_invalid_action"] & ~captured})
            if isinstance(trace, dict) and "__source_exchange_invalid_action" in trace
            else (action, memory, trace)
            for action, memory, trace in role_outputs
        )
        actions = {agent: role_outputs[index][0] for index, agent in enumerate(agents)}
        next_obs, next_env_state, _rewards, dones, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(env_rngs, env_state, actions)

        completed = info["returned_episode"][..., 0]
        episode_returns = info["returned_episode_returns"][..., 0]
        capture_now = completed & ~captured
        next_first_returns = jnp.where(capture_now, episode_returns, first_returns)
        next_metrics = {name: jnp.where(capture_now, info["episode_metrics"][name], previous)
                        for name, previous in episode_metrics.items()}
        next_captured = captured | capture_now
        next_carry = (
            next_obs,
            next_env_state,
            tuple(output[1] for output in role_outputs),
            tuple(dones[agent] for agent in agents),
            next_first_returns,
            next_captured,
            step_rng,
            next_metrics,
        )
        return next_carry, (
            tuple(output[2] for output in role_outputs),
            tuple(output[0] for output in role_outputs),
        )

    final_carry, traces = jax.lax.scan(
        step,
        (
            observations,
            env_state,
            memories,
            resets,
            first_returns,
            captured,
            rollout_rng,
            episode_metrics,
        ),
        None,
        length=num_steps,
    )
    final_returns = final_carry[4]
    final_captured = final_carry[5]
    # Every reset contributes to the SP objective. A policy that fails to
    # complete an episode therefore receives that reset's accumulated return
    # (zero when no rewards or costs occurred), rather than being averaged only
    # over its successful subset.
    mean_return = jnp.mean(final_returns)
    return ProgramRolloutResult(
        mean_return=mean_return,
        first_returns=final_returns,
        captured=final_captured,
        trace_agent_0=traces[0][0],
        trace_agent_1=traces[0][1],
        trace_agent_2=traces[0][2] if len(agents) >= 3 else None,
        trace_agent_3=traces[0][3] if len(agents) >= 4 else None,
        trace_agent_4=traces[0][4] if len(agents) >= 5 else None,
        episode_metrics=final_carry[7],
        actions_agent_0=traces[1][0],
        actions_agent_1=traces[1][1],
        actions_agent_2=traces[1][2] if len(agents) >= 3 else None,
        actions_agent_3=traces[1][3] if len(agents) >= 4 else None,
        actions_agent_4=traces[1][4] if len(agents) >= 5 else None,
    )


def self_play_rollout(
    rng: jax.Array,
    policy: CompiledJointPolicy,
    env: Any,
    **kwargs,
) -> ProgramRolloutResult:
    """Evaluate every seat from one generated joint-policy module."""
    if policy.num_agents != len(tuple(env.agents)):
        raise ValueError("the compiled policy must match the environment agent count")
    return rollout_roles(
        rng, policy.agent_0, policy.agent_1, env,
        role_agent_2=policy.agent_2, role_agent_3=policy.agent_3, role_agent_4=policy.agent_4, **kwargs,
    )


def cross_play_rollout(
    rng: jax.Array,
    policy_agent_0: CompiledJointPolicy,
    policy_agent_1: CompiledJointPolicy,
    env: Any,
    **kwargs,
) -> ProgramRolloutResult:
    """Evaluate row role 0 against all remaining roles from another module."""
    expected = len(tuple(env.agents))
    if policy_agent_0.num_agents != expected or policy_agent_1.num_agents != expected:
        raise ValueError("both compiled policies must match the environment agent count")
    return rollout_roles(
        rng,
        policy_agent_0.agent_0,
        policy_agent_1.agent_1,
        env,
        role_agent_2=policy_agent_1.agent_2,
        role_agent_3=policy_agent_1.agent_3, role_agent_4=policy_agent_1.agent_4,
        **kwargs,
    )


def cross_play_matrix_rollout(
    root_seeds: jax.Array,
    policies: tuple[CompiledJointPolicy, ...],
    env: Any,
    *,
    num_envs: int,
    num_steps: int | None = None,
) -> XPMatrixRolloutResult:
    """Evaluate a complete directed XP matrix without compiling each cell.

    The policy population can contain arbitrary memory PyTrees, so it cannot be
    stacked into one ordinary ``vmap``. Instead, each role policy processes all
    of its partners in one batch while a single batched environment step
    advances every ``(singleton policy, remaining-seat policy, episode)`` combination.
    A short host loop over environment steps deliberately avoids constructing a
    distinct fused ``lax.scan`` executable for every ordered policy pairing.

    ``root_seeds`` has shape ``(population, population)``. This preserves both
    common-random-number evaluation (all entries equal) and independent,
    deterministic per-cell seeds.
    """
    size = len(policies)
    num_envs = int(num_envs)
    if size < 1:
        raise ValueError("policies must contain at least one policy")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if num_steps is None:
        num_steps = int(getattr(env, "max_steps"))
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    root_seeds = jnp.asarray(root_seeds)
    if root_seeds.shape != (size, size):
        raise ValueError(
            f"root_seeds must have shape {(size, size)}, got {root_seeds.shape}"
        )

    agents = tuple(env.agents)
    if len(agents) not in (2, 3, 4, 5):
        raise ValueError("generated XP matrices support two through five agents")
    if any(policy.num_agents != len(agents) for policy in policies):
        raise ValueError("every compiled policy must match the environment agent count")
    num_agents = len(agents)
    num_cells = size * size
    num_rollouts = num_cells * num_envs

    cell_keys = jax.vmap(jax.random.PRNGKey)(root_seeds.reshape(-1))
    split_roots = jax.jit(
        jax.vmap(lambda key: jax.random.split(key, 2))
    )(cell_keys)
    reset_cell_keys = split_roots[:, 0]
    step_keys = split_roots[:, 1].reshape(size, size, 2)
    split_reset_keys = jax.jit(
        jax.vmap(lambda key: jax.random.split(key, num_envs))
    )
    reset_keys = split_reset_keys(reset_cell_keys).reshape(num_rollouts, 2)

    batched_reset = jax.jit(jax.vmap(env.reset))
    batched_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))
    observations, env_state = batched_reset(reset_keys)
    observations = _reshape_leading(observations, 1, (size, size, num_envs))
    env_state = _reshape_leading(env_state, 1, (size, size, num_envs))

    # A row policy owns role 0 for every column; a column policy owns every
    # remaining role for every row. Thus 3P cell [A,B] is A/BB. Keeping each
    # policy and seat's memory separate supports unrelated memory PyTrees.
    policy_batch = size * num_envs
    memories = tuple(
        tuple(policy.role(agent).init(policy_batch) for policy in policies)
        for agent in agents
    )
    resets = tuple(
        jnp.ones((size, size, num_envs), dtype=jnp.bool_) for _ in agents
    )
    first_returns = jnp.zeros((size, size, num_envs), dtype=jnp.float32)
    captured = jnp.zeros((size, size, num_envs), dtype=jnp.bool_)
    episode_metrics = {name: jnp.zeros((size, size, num_envs), jnp.float32)
                       for name in getattr(env, "episode_metric_names", ())}

    split_step_keys = jax.jit(
        jax.vmap(lambda key: jax.random.split(key, num_agents + 2))
    )
    split_episode_keys = jax.jit(
        jax.vmap(lambda key: jax.random.split(key, num_envs))
    )

    for _ in range(num_steps):
        key_parts = split_step_keys(step_keys.reshape(num_cells, 2)).reshape(
            size, size, num_agents + 2, 2
        )
        step_keys = key_parts[:, :, 0]

        def episode_keys(part_index: int):
            return split_episode_keys(
                key_parts[:, :, part_index].reshape(num_cells, 2)
            ).reshape(size, size, num_envs, 2)

        policy_keys = tuple(episode_keys(index + 1) for index in range(num_agents))
        env_keys = episode_keys(num_agents + 1)

        action_arrays = []
        next_memories = []
        for seat_index, agent in enumerate(agents):
            seat_actions = []
            seat_memories = []
            if seat_index == 0:
                for row_index, policy in enumerate(policies):
                    action, memory, _trace = policy.role(agent).batched_act(
                        _flatten_leading(jax.tree_util.tree_map(
                            lambda x: x[row_index], observations[agent]), 2),
                        resets[seat_index][row_index].reshape(-1),
                        memories[seat_index][row_index],
                        _flatten_leading(policy_keys[seat_index][row_index], 2),
                    )
                    seat_actions.append(action.reshape(size, num_envs))
                    seat_memories.append(memory)
                action_arrays.append(jnp.stack(seat_actions, axis=0).reshape(-1))
            else:
                for column_index, policy in enumerate(policies):
                    action, memory, _trace = policy.role(agent).batched_act(
                        _flatten_leading(jax.tree_util.tree_map(
                            lambda x: x[:, column_index], observations[agent]), 2),
                        resets[seat_index][:, column_index].reshape(-1),
                        memories[seat_index][column_index],
                        _flatten_leading(policy_keys[seat_index][:, column_index], 2),
                    )
                    seat_actions.append(action.reshape(size, num_envs))
                    seat_memories.append(memory)
                action_arrays.append(jnp.stack(seat_actions, axis=1).reshape(-1))
            next_memories.append(tuple(seat_memories))

        actions = {agent: action_arrays[index] for index, agent in enumerate(agents)}
        next_obs, next_env_state, _rewards, dones, info = batched_step(
            env_keys.reshape(num_rollouts, 2),
            _flatten_leading(env_state, 3),
            actions,
        )
        next_obs = _reshape_leading(next_obs, 1, (size, size, num_envs))
        next_env_state = _reshape_leading(
            next_env_state, 1, (size, size, num_envs)
        )
        dones = _reshape_leading(dones, 1, (size, size, num_envs))
        info = _reshape_leading(info, 1, (size, size, num_envs))

        completed = info["returned_episode"][..., 0]
        episode_returns = info["returned_episode_returns"][..., 0]
        capture_now = completed & ~captured
        first_returns = jnp.where(capture_now, episode_returns, first_returns)
        episode_metrics = {name: jnp.where(capture_now, info["episode_metrics"][name], previous)
                           for name, previous in episode_metrics.items()}
        captured = captured | capture_now
        observations = next_obs
        env_state = next_env_state
        memories = tuple(next_memories)
        resets = tuple(dones[agent] for agent in agents)

    result = XPMatrixRolloutResult(
        first_returns=first_returns,
        captured=captured,
        episode_metrics=episode_metrics,
    )
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
    return result


__all__ = [
    "cross_play_matrix_rollout",
    "cross_play_rollout",
    "rollout_roles",
    "self_play_rollout",
]
