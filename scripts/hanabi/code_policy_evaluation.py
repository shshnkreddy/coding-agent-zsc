"""Student/code-policy games with fixed evaluation seeds and raw episode evidence."""

import json
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from hanabi.hanabi_actor_critic_network import ActorCritic
from llm_policy.environments.hanabi.adapter import CorrectedHanabi
from llm_policy.policies.api import HanabiObservation


def make_code_policy_evaluator(config, teacher):
    """Compile once; subsequent evaluations pass current student weights as data.

    Each reset contributes one score. Illegal actions and unfinished games score
    zero and are reported separately, so tuning cannot omit failed episodes.
    Teacher actions are executed, with private history in the occupied seat.
    """
    env = CorrectedHanabi(**config["ENV_KWARGS"])
    model = ActorCritic(env.action_space(env.agents[0]).n, config=config)
    roles = (teacher.agent_0, teacher.agent_1)
    num_episodes = config["TEACHER_EVAL_EPISODES"]
    batch_size = min(config["TEACHER_EVAL_BATCH_SIZE"], num_episodes)
    max_steps = config["TEACHER_EVAL_MAX_STEPS"]
    root_key = jax.random.PRNGKey(config["TEACHER_EVAL_SEED"])

    def make_rollout(student_seat, greedy):
        def episode(params, key):
            reset_key, rng = jax.random.split(key)
            obs, state = env.reset(reset_key)
            hidden = ActorCritic.initialize_hidden_state(
                config["RNN"], config["NUM_RNN_LAYERS"], 1, config["LAYER_WIDTH"]
            )
            memories = tuple(role.initial_memory for role in roles)

            def condition(carry):
                return ~carry[6] & ~carry[7] & (carry[8] < max_steps)

            def step(carry):
                obs, state, hidden, memories, rng, total, done, invalid, length = carry
                keys = jax.random.split(rng, 4)
                masks = env.get_legal_moves(state)
                next_memories = list(memories)
                actions = {}
                for seat, agent in enumerate(env.agents):
                    mask = masks[agent].astype(jnp.bool_)
                    if seat == student_seat:
                        hidden, logits, _ = model.apply(
                            params, hidden,
                            (obs[agent][None, None],
                             jnp.zeros((1, 1, env.hand_size * env.num_colors * env.num_ranks)),
                             (length == 0).reshape((1, 1)), mask[None, None]),
                        )
                        logits = logits[0, 0]
                        action = (jnp.argmax(logits) if greedy else
                                  distrax.Categorical(logits=logits).sample(seed=keys[seat + 1]))
                    else:
                        action, next_memories[seat], _ = roles[seat].scalar_act(
                            HanabiObservation(features=obs[agent], legal_actions=mask),
                            length == 0, memories[seat], keys[seat + 1],
                        )
                    active = ~mask[20]
                    legal = (action >= 0) & (action < 21) & mask[jnp.clip(action, 0, 20)]
                    invalid = invalid | (active & ~legal)
                    # A placeholder keeps env.step well-defined on the failure
                    # transition; the entire invalid episode receives zero.
                    actions[agent] = jnp.where(active, jnp.where(legal, action, jnp.argmax(mask)), 20).astype(jnp.int32)
                obs, state, rewards, dones, _ = env.step(keys[3], state, actions)
                return (obs, state, hidden, tuple(next_memories), keys[0],
                        total + rewards[env.agents[0]], dones["__all__"], invalid, length + 1)

            result = jax.lax.while_loop(
                condition, step,
                (obs, state, hidden, memories, rng, jnp.asarray(0., jnp.float32),
                 jnp.asarray(False), jnp.asarray(False), jnp.asarray(0, jnp.int32)),
            )
            total, completed, invalid, length = result[5:]
            return {"scores": jnp.where(completed & ~invalid, total, 0.),
                    "raw_returns": total, "completed": completed,
                    "invalid": invalid, "truncated": ~completed & ~invalid,
                    "lengths": length}

        return jax.jit(jax.vmap(episode, in_axes=(None, 0)))

    kernels = {(seat, greedy): make_rollout(seat, greedy)
               for seat in (0, 1) for greedy in (True, False)}
    teacher_kernel = make_rollout(-1, False)
    teacher_reference = None

    def run_batches(kernel, params):
        parts = []
        for start in range(0, num_episodes, batch_size):
            # Keep compiled batch shape fixed; omit padded episodes in outputs.
            ids = jnp.arange(start, start + batch_size, dtype=jnp.uint32)
            keys = jax.vmap(lambda index: jax.random.fold_in(root_key, index))(ids)
            result = jax.device_get(kernel(params, keys))
            count = min(batch_size, num_episodes - start)
            parts.append({name: np.asarray(value)[:count] for name, value in result.items()})
        return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}

    def statistics(result):
        scores = result["scores"]
        return {"mean_return": float(scores.mean()),
                "episode_sem": float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.,
                "completion_rate": float(result["completed"].mean()),
                "invalid_rate": float(result["invalid"].mean()),
                "truncation_rate": float(result["truncated"].mean())}

    def evaluate(params_list, output_dir, outer_update, update_step):
        nonlocal teacher_reference
        directory = Path(output_dir) / "teacher_eval"
        directory.mkdir(parents=True, exist_ok=True)
        identity = {
            "teacher_bundle_sha256": teacher.source_hash,
            "evaluation_seed": config["TEACHER_EVAL_SEED"],
            "episodes_per_pairing": num_episodes, "max_steps": max_steps,
            "common_random_numbers": True, "fix_final_life_termination": True,
            "failure_scoring": "Illegal active actions or unfinished episodes score zero; no episodes omitted.",
        }
        if teacher_reference is None:
            teacher_reference = run_batches(teacher_kernel, params_list[0])
            np.savez(directory / "teacher_self_play.npz", **teacher_reference)
            (directory / "teacher_self_play.json").write_text(json.dumps(
                {**identity, **statistics(teacher_reference)}, indent=2) + "\n")
        metrics = {f"teacher_eval/teacher_self_play/{key}": value
                   for key, value in statistics(teacher_reference).items()}
        destination = directory / f"update_{outer_update:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        for greedy in (True, False):
            mode = "greedy" if greedy else "sampled"
            results = [[run_batches(kernels[seat, greedy], params) for seat in (0, 1)]
                       for params in params_list]
            raw = {name: np.stack([[cell[name] for cell in row] for row in results])
                   for name in results[0][0]}
            if not np.isfinite(raw["scores"]).all():
                raise ValueError("Non-finite student/code-policy evaluation scores")
            np.savez(destination / f"{mode}.npz", **raw)
            per_replica = []
            for replica, row in enumerate(results):
                seat_stats = [statistics(cell) for cell in row]
                paired_scores = raw["scores"][replica].mean(axis=0)
                summary = {"replica": replica, "student_seat_0": seat_stats[0],
                           "student_seat_1": seat_stats[1],
                           "mean_return": float(paired_scores.mean()),
                           "paired_episode_sem": float(paired_scores.std(ddof=1) / np.sqrt(num_episodes)) if num_episodes > 1 else 0.}
                per_replica.append(summary)
                prefix = f"teacher_eval/{mode}/replica_{replica}"
                metrics[f"{prefix}/mean_return"] = summary["mean_return"]
                for seat, stats in enumerate(seat_stats):
                    metrics.update({f"{prefix}/student_seat_{seat}/{key}": value for key, value in stats.items()})
            replica_means = np.asarray([item["mean_return"] for item in per_replica])
            aggregate = {"mean_return": float(replica_means.mean()),
                         "replica_sem": float(replica_means.std(ddof=1) / np.sqrt(len(replica_means))) if len(replica_means) > 1 else 0.,
                         "invalid_rate": float(raw["invalid"].mean()),
                         "truncation_rate": float(raw["truncated"].mean()),
                         "completion_rate": float(raw["completed"].mean())}
            metrics.update({f"teacher_eval/{mode}/{key}": value for key, value in aggregate.items()})
            (destination / f"{mode}.json").write_text(json.dumps({
                **identity, "student_action_selection": mode, "teacher_action_selection": "policy_native",
                "outer_update": outer_update, "ppo_update": int(update_step),
                "environment_steps_per_replica": int(update_step) * config["NUM_ENVS"] * config["NUM_STEPS"],
                "array_axes": ["replica", "student_seat", "episode"],
                "aggregate": aggregate, "replicas": per_replica,
            }, indent=2) + "\n")
        return metrics

    return evaluate
