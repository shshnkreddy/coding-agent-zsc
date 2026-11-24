"""Train a recurrent Hanabi policy in self-play with optional code-policy supervision."""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil

# Bound XLA's host thread pools on large shared machines before importing JAX.
if hasattr(os, 'sched_getaffinity'):
    cores = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cores[:max(1, int(os.environ.get('OMP_NUM_THREADS', '2')))])

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import optax

from llm_policy.evaluation.api import compile_candidate, load_task
from llm_policy.policies.bundle import bundle_hash


HANABI_TASKS = {'hanabi_2p': 'two_player', 'hanabi_2p_h_group_level_1': 'two_player_h_group_level_1'}


class ActorCritic(nn.Module):
    hidden_size: int
    num_actions: int

    @nn.compact
    def __call__(self, memory, features, reset, legal):
        memory = jnp.where(reset[..., None], 0., memory)
        encoded = nn.relu(nn.Dense(self.hidden_size)(features))
        memory, features = nn.GRUCell(self.hidden_size)(memory, encoded)
        logits = nn.Dense(self.num_actions)(features)
        value = nn.Dense(1)(features)[..., 0]
        return memory, jnp.where(legal, logits, -1e9), value


def selected(log_probs, actions):
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]


def mean_where(value, mask):
    return jnp.sum(jnp.where(mask, value, 0.)) / jnp.maximum(1, mask.sum())


def train(args):
    config = Path(__file__).resolve().parents[1] / 'coding_agent_zsc/hanabi' / HANABI_TASKS[args.env] / 'config.yaml'
    task = replace(load_task(config), primitive_mode='none', restrict_introspection=True)
    env = task.raw_env
    seats = tuple(env.agents)
    args.output.mkdir(parents=True, exist_ok=False)
    teacher = None
    teacher_hash = None
    if args.teacher:
        source = args.teacher / 'policy.py' if args.teacher.is_dir() else args.teacher
        snapshot = args.output / 'teacher'
        snapshot.mkdir()
        shutil.copyfile(source, snapshot / 'policy.py')
        helper = source.with_name('custom_primitives.py')
        if helper.is_file():
            shutil.copyfile(helper, snapshot / helper.name)
        teacher_hash = bundle_hash(snapshot)
        if args.ce_weight:
            teacher = compile_candidate(snapshot / 'policy.py', task)

    rng, init_key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    observations, state = jax.vmap(env.reset)(jax.random.split(reset_key, args.num_envs))

    def inputs(obs):
        return tuple(jnp.stack([getattr(obs[s], field) for s in seats])
                     for field in ('features', 'legal_actions'))

    features, legal = inputs(observations)
    model = ActorCritic(args.hidden_size, legal.shape[-1])
    reset = jnp.ones(args.num_envs, bool)
    memory = jnp.zeros((*features.shape[:2], args.hidden_size))
    params = model.init(init_key, memory, features, reset, legal)
    optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(args.learning_rate))
    opt_state = optimizer.init(params)
    roles = (teacher.agent_0, teacher.agent_1) if teacher else ()
    teacher_mem = tuple(role.init(args.num_envs) for role in roles)

    def collect(params, carry):
        def step(carry, _):
            obs, state, reset, memory, teacher_mem, key = carry
            key, action_key, env_key, *teacher_keys = jax.random.split(key, 5)
            features, legal = inputs(obs)
            memory, logits, value = model.apply(params, memory, features, reset, legal)
            actions = jax.random.categorical(action_key, logits).astype(jnp.int32)
            logp = selected(jax.nn.log_softmax(logits), actions)
            targets = []
            next_teacher_mem = []
            for i, role in enumerate(roles):
                target, mem, _ = role.batched_act(obs[seats[i]], reset, teacher_mem[i],
                                                  jax.random.split(teacher_keys[i], args.num_envs))
                targets.append(target)
                next_teacher_mem.append(mem)
            target = jnp.stack(targets) if targets else jnp.full_like(actions, -1)
            seat_actions = dict(zip(seats, actions))
            obs, state, reward, dones, _ = jax.vmap(env.step)(
                jax.random.split(env_key, args.num_envs), state, seat_actions)
            done = dones['__all__']
            data = (features, legal, reset, actions, logp, value, reward[seats[0]], done, target)
            return (obs, state, done, memory, tuple(next_teacher_mem), key), data

        return jax.lax.scan(step, carry, None, length=args.rollout_steps)

    @jax.jit
    def update(params, opt_state, carry):
        initial_memory = carry[3]
        carry, batch = collect(params, carry)
        features, legal, resets, actions, old_logp, old_value, rewards, dones, targets = batch
        next_features, next_legal = inputs(carry[0])
        _, _, bootstrap = model.apply(params, carry[3], next_features, carry[2], next_legal)

        def gae(carry, step):
            following_value, following_advantage = carry
            reward, done, value = step
            discount = .999 * (1. - done)
            delta = reward + discount * following_value - value
            advantage = delta + discount * .9 * following_advantage
            return (value, advantage), advantage

        _, advantages = jax.lax.scan(gae, (bootstrap, jnp.zeros_like(bootstrap)),
                                     (rewards, dones, old_value), reverse=True)
        returns = advantages + old_value
        active = ~legal[..., env.layout['actions']['inactive_noop']]
        centered = advantages - mean_where(advantages, active)
        advantages = centered / jnp.sqrt(mean_where(centered**2, active) + 1e-8)
        safe_target = jnp.clip(targets, 0, model.num_actions - 1)
        valid = active & (safe_target == targets) & selected(legal, safe_target)

        def loss(weights):
            def replay(memory, step):
                x, reset, mask = step
                memory, logits, value = model.apply(weights, memory, x, reset, mask)
                return memory, (logits, value)

            _, (logits, value) = jax.lax.scan(replay, initial_memory, (features, resets, legal))
            log_probs = jax.nn.log_softmax(logits)
            ratio = jnp.exp(selected(log_probs, actions) - old_logp)
            actor = -mean_where(jnp.minimum(ratio * advantages,
                                           jnp.clip(ratio, .8, 1.2) * advantages), active)
            critic = jnp.mean((value - returns)**2)
            entropy = -mean_where((jnp.exp(log_probs) * log_probs).sum(-1), active)
            ce = -mean_where(selected(log_probs, safe_target), valid)
            return actor + .5 * critic - .01 * entropy + args.ce_weight * ce, ce

        def epoch(carry, _):
            weights, opt = carry
            (total, ce), grad = jax.value_and_grad(loss, has_aux=True)(weights)
            changes, opt = optimizer.update(grad, opt, weights)
            return (optax.apply_updates(weights, changes), opt), (total, ce)

        (params, opt_state), (losses, ce) = jax.lax.scan(epoch, (params, opt_state), None, length=args.epochs)
        metrics = dict(loss=losses.mean(), teacher_ce=ce.mean(), labels=valid.sum(),
                       mean_step_reward=rewards.mean(), episodes=dones.sum())
        return params, opt_state, carry, metrics

    metadata = dict(vars(args), teacher_sha256=teacher_hash, num_actions=model.num_actions,
                    input_size=features.shape[-1], network='shared-gru-actor-critic')
    (args.output / 'config.json').write_text(json.dumps(metadata, default=str, indent=2) + '\n')
    carry = (observations, state, reset, memory, teacher_mem, rng)
    with (args.output / 'metrics.jsonl').open('x') as log:
        for n in range(1, args.updates + 1):
            params, opt_state, carry, metrics = update(params, opt_state, carry)
            row = dict(update=n, **{k: v.item() for k, v in metrics.items()})
            line = json.dumps(row, allow_nan=False)
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()
            checkpoint = args.output / 'student.tmp'
            checkpoint.write_bytes(serialization.to_bytes(jax.device_get(params)))
            checkpoint.replace(args.output / 'student.msgpack')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=HANABI_TASKS, default='hanabi_2p')
    parser.add_argument('--teacher', type=Path, help='Seed directory or saved Python policy')
    parser.add_argument('--ce-weight', type=float, default=.1)
    parser.add_argument('--output', type=Path, required=True, help='New training output directory')
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--rollout-steps', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.0005)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if min(args.updates, args.num_envs, args.rollout_steps, args.epochs, args.hidden_size) < 1:
        parser.error('Training dimensions must be positive.')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.seed < 2**32:
        parser.error('Supply a positive finite learning rate and a uint32 seed.')
    if not math.isfinite(args.ce_weight) or args.ce_weight < 0 or (args.ce_weight and not args.teacher):
        parser.error('Supply a teacher for a positive CE weight, or use --ce-weight 0 for ordinary self-play.')
    train(args)


if __name__ == '__main__':
    main()
