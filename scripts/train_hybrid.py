"""Recurrent Hanabi IPPO with a frozen code teacher on student trajectories.

Learning and evaluation are adapted from the frozen October 2026 Astra/Fable
experiment. See hanabi/README.md for provenance and the paper configuration.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, NamedTuple

if hasattr(os, "sched_getaffinity"):
    cores = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cores[:max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))])

import distrax
from flax import serialization
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jaxmarl.wrappers.baselines import LogWrapper
from llm_policy.environments.hanabi.adapter import CorrectedHanabi, source_hashes
from llm_policy.policies.api import HanabiObservation
from llm_policy.policies.runtime import compile_policy_module
import numpy as np
import optax
from safetensors.flax import save_file

from hanabi.code_policy_evaluation import make_code_policy_evaluator
from hanabi.hanabi_actor_critic_network import ActorCritic
from hanabi.hanabi_belief_model import BeliefModel

def prepare_teacher(config, env, output_dir):
    """Snapshot and compile the exact source used by this run, before any updates."""
    output_dir = Path(output_dir)
    teacher = None
    if config["CE_COEF"] > 0 or config.get("TEACHER_EVAL", False):
        source = Path(config["TEACHER_POLICY_PATH"]).expanduser().resolve(strict=True)
        bundle_dir = output_dir / "teacher"
        # Do not overwrite a previous run's teacher or pick up a stale helper.
        bundle_dir.mkdir(exist_ok=False)
        for path in (source, source.with_name("custom_primitives.py")):
            if path.is_file():
                shutil.copyfile(path, bundle_dir / ("policy.py" if path == source else path.name))
        sample_obs, sample_state = env.reset(jax.random.PRNGKey(0))
        masks = env.get_legal_moves(sample_state)
        teacher = compile_policy_module(
            bundle_dir / "policy.py",
            HanabiObservation(features=sample_obs[env.agents[0]], legal_actions=masks[env.agents[0]].astype(jnp.bool_)),
        )
        if teacher.metadata["adapter"] != "hanabi":
            raise ValueError("Teacher must implement the Hanabi policy interface")
    identity = {
        "environment": "hanabi", "fix_final_life_termination": True,
        "observation_size": int(env.obs_size), "num_actions": int(env.num_moves),
        "env_kwargs": config["ENV_KWARGS"], "source_sha256": source_hashes(),
        "teacher_source_path": str(source) if teacher is not None else None,
        "teacher_bundle_sha256": teacher.source_hash if teacher is not None else None,
        "teacher_metadata": teacher.metadata if teacher is not None else None,
    }
    (output_dir / "hybrid_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return teacher


def teacher_step(teacher, agents, obs, avail_actions, resets, memories, rng):
    """Update both private histories; label only legal active-player decisions."""
    keys = jax.random.split(rng, len(agents) + 1)
    actions, next_memories, valid, active = [], [], [], []
    for index, agent in enumerate(agents):
        role = teacher.agent_0 if index == 0 else teacher.agent_1
        mask = avail_actions[agent].astype(jnp.bool_)
        action, memory, _ = role.batched_act(
            HanabiObservation(features=obs[agent], legal_actions=mask), resets[index], memories[index],
            jax.random.split(keys[index + 1], mask.shape[0]),
        )
        is_active = ~mask[:, 20]
        in_range = (action >= 0) & (action < mask.shape[-1])
        safe_action = jnp.clip(action, 0, mask.shape[-1] - 1)
        is_valid = is_active & in_range & jnp.take_along_axis(mask, safe_action[:, None], axis=-1)[:, 0]
        # Use a legal placeholder before log_prob; invalid/inactive labels have
        # zero weight, including when a future network masks with -inf.
        actions.append(jnp.where(is_valid, safe_action, jnp.argmax(mask, axis=-1)).astype(jnp.int32))
        next_memories.append(memory)
        valid.append(is_valid)
        active.append(is_active)
    return (jnp.concatenate(actions), jnp.concatenate(valid), jnp.concatenate(active),
            tuple(next_memories), keys[0])


def imitation_loss(logits, teacher_actions, valid):
    """Masked hard-label CE, finite even when there are no valid labels."""
    # Sanitize masked entries before indexing even if called independently of
    # teacher_step (e.g. on a completely empty supervision minibatch).
    labels = jnp.where(valid, teacher_actions, jnp.argmax(logits, axis=-1))
    log_prob = distrax.Categorical(logits=logits).log_prob(labels)
    return -jnp.where(valid, log_prob, 0.0).sum() / jnp.maximum(valid.sum(), 1)

def write_metrics(directory, values):
    row = {key: np.asarray(value).item() for key, value in values.items()}
    with (directory / "metrics.jsonl").open("a") as log:
        log.write(json.dumps(row, allow_nan=False) + "\n")


def save_student(params, directory):
    directory.mkdir(exist_ok=True)
    weights = directory / "student.tmp"
    save_file(flatten_dict(params, sep=","), str(weights))
    weights.replace(directory / "student.safetensors")
    weights.write_bytes(serialization.to_bytes(params))
    weights.replace(directory / "student.msgpack")


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    hand: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    env_state: Any
    hidden_state: Any
    encoded_state: jnp.ndarray
    hand_plus_deck: jnp.ndarray
    card_knowledge: jnp.ndarray
    teacher_action: jnp.ndarray
    teacher_valid: jnp.ndarray
    teacher_active: jnp.ndarray


def batchify(x_E_: dict, agent_list, num_actors):
    x_PE_ = jnp.stack([x_E_[a] for a in agent_list])
    x_B_ = x_PE_.reshape((num_actors, -1))
    return x_B_


def unbatchify(x_B_: jnp.ndarray, agent_list, num_envs, num_agents):
    x_PE_ = x_B_.reshape((num_agents, num_envs, -1))
    x_E_ = {a: x_PE_[i] for i, a in enumerate(agent_list)}
    return x_E_


def make_train(config, run_output_dir):
    env = CorrectedHanabi(**config["ENV_KWARGS"])
    code_policy = prepare_teacher(config, env, run_output_dir)
    teacher = code_policy if config["CE_COEF"] > 0 else None
    evaluate_code_policy = (make_code_policy_evaluator(config, code_policy)
                            if config.get("TEACHER_EVAL", False) else None)

    # The paper's IPPO branch uses values from both seats to compute GAE.
    config["ACTING_PLAYER_ONLY_CRITIC"] = False

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["NUM_UPDATES_PER_INNER_LOOP"] = config["NUM_UPDATES"] // config["NUM_OUTER_UPDATES"]
    config["NUM_GPUS"] = jax.device_count()
    (run_output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    initial_params = jnp.zeros(config["NUM_GPUS"])

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def _get_init_update_state(rng, actor_critic_params, belief_model_params):

        # INIT ACTOR-CRITIC NETWORK
        ac_network = ActorCritic(env.action_space(env.agents[0]).n, config=config)

        if config["ACTOR_CRITIC_FOLDER"] == False:
            rng, _rng = jax.random.split(rng)
            init_x = (
                jnp.zeros((1, config["NUM_ACTORS"], env.observation_space(env.agents[0]).shape)), # obs
                jnp.zeros((1, config["NUM_ACTORS"], env.hand_size * env.num_colors * env.num_ranks)), # hand
                jnp.zeros((1, config["NUM_ACTORS"])), # done
                jnp.zeros((1, config["NUM_ACTORS"], env.action_space(env.agents[0]).n)) # avail_actions
            )
            init_hidden_state_LBW_ = ActorCritic.initialize_hidden_state(config["RNN"], config["NUM_RNN_LAYERS"], config["NUM_ACTORS"], config["LAYER_WIDTH"])
            ac_network_params = ac_network.init(_rng, init_hidden_state_LBW_, init_x)
        else:
            ac_network_params = actor_critic_params

        if config["ANNEAL_LR"]:
            ac_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            ac_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        train_state = TrainState.create(
            apply_fn=ac_network.apply,
            params=ac_network_params,
            tx=ac_tx,
        )

        # INIT BELIEF MODEL NETWORK
        count_based = config["TRAIN_POLICY"] and not config["BELIEF_MODEL_FOLDER"]
        belief_model = BeliefModel(
            layer_width=config["LAYER_WIDTH"],
            num_colors=env.num_colors,
            num_ranks=env.num_ranks,
            hand_size=env.hand_size,
            num_rnn_layers=config["NUM_RNN_LAYERS"],
            num_ff_layers_encoder=config["NUM_FF_LAYERS_ENCODER"],
            num_ff_layers_decoder=config["NUM_FF_LAYERS_DECODER"],
            count_based=count_based,
        )

        if config["BELIEF_MODEL_FOLDER"] == False:
            init_x = (
                jnp.zeros((1, config["NUM_ACTORS"], env.observation_space(env.agents[0]).shape)), # obs
                jnp.zeros((1, config["NUM_ACTORS"])), # done
                jnp.zeros((1, config["NUM_ACTORS"], env.hand_size * env.num_colors * env.num_ranks)), # hand
                jnp.zeros((1, config["NUM_ACTORS"], env.num_colors * env.num_ranks)), # hand_plus_deck
                jnp.zeros((1, config["NUM_ACTORS"], env.hand_size, env.num_colors * env.num_ranks)), # card_knowledge
            )
            bm_hidden_state_LBW_ = belief_model.initialize_hidden_state(config["NUM_RNN_LAYERS"], config["NUM_ACTORS"], config["LAYER_WIDTH"])
            rng, _rng = jax.random.split(rng)
            bm_params = belief_model.init(_rng, (_rng, bm_hidden_state_LBW_), init_x)
        else:
            bm_params = belief_model_params

        if config["ANNEAL_LR"]:
            bm_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            bm_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=config["BM_LR"], eps=1e-5))

        bm_train_state = TrainState.create(
            apply_fn=belief_model.apply,
            params=bm_params,
            tx=bm_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        rng_E_ = jax.random.split(_rng, config["NUM_ENVS"])
        obs_EO_, env_state_E_ = jax.vmap(env.reset, in_axes=(0,))(rng_E_)
        done_B = jnp.ones((config["NUM_ACTORS"]), dtype=bool)
        init_hidden_state_LBW_ = ActorCritic.initialize_hidden_state(config["RNN"], config["NUM_RNN_LAYERS"], config["NUM_ACTORS"], config["LAYER_WIDTH"])
        bm_init_hidden_state_LBW_ = belief_model.initialize_hidden_state(config["NUM_RNN_LAYERS"], config["NUM_ACTORS"], config["LAYER_WIDTH"])

        teacher_memories = (() if teacher is None else
                            (teacher.agent_0.init(config["NUM_ENVS"]), teacher.agent_1.init(config["NUM_ENVS"])))
        # Derive a separate stream without advancing the baseline RNG.
        teacher_rng = jax.random.fold_in(rng, 0xCE)
        runner_state = (train_state, bm_train_state, env_state_E_, obs_EO_, done_B, teacher_memories, teacher_rng, init_hidden_state_LBW_, bm_init_hidden_state_LBW_, rng)

        return (runner_state, 0)

    @jax.jit
    def update_agents(update_state, process_id):

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            init_hidden_state_LBW_, bm_init_hidden_state_LBW_ = runner_state[-3], runner_state[-2]
            def _env_step(runner_state, predetermined_action_B):
                train_state, bm_train_state, env_state_E_, obs_EO_, done_B, teacher_memories, teacher_rng, hidden_state_LBW_, bm_hidden_state_LBW_, rng = runner_state

                # SELECT ACTION
                avail_actions_EA_ = jax.vmap(env.get_legal_moves)(env_state_E_.env_state)
                avail_actions_BA = jax.lax.stop_gradient(batchify(avail_actions_EA_, env.agents, config["NUM_ACTORS"]))
                obs_BO = batchify(obs_EO_, env.agents, config["NUM_ACTORS"])
                hand_EPHCR = env_state_E_.env_state.player_hands
                hand_BHX = hand_EPHCR.swapaxes(0, 1).reshape((config["NUM_ACTORS"], hand_EPHCR.shape[2], -1))
                deck_EDCR = env_state_E_.env_state.deck
                deck_BDX = jnp.stack([deck_EDCR] * env.num_agents).reshape((config["NUM_ACTORS"], deck_EDCR.shape[1], -1))
                hand_plus_deck_BX = hand_BHX.sum(axis=1) + deck_BDX.sum(axis=1)
                hand_BY = hand_BHX.reshape((config["NUM_ACTORS"], -1))
                card_knowledge_EPHX = env_state_E_.env_state.card_knowledge
                card_knowledge_BHX = card_knowledge_EPHX.swapaxes(0, 1).reshape((config["NUM_ACTORS"], ) + card_knowledge_EPHX.shape[2:])

                ac_in_1B_ = (obs_BO[None], hand_BY[None], done_B[None], avail_actions_BA[None])
                next_hidden_state_LBW_, action_logits_1BA, value_1B = train_state.apply_fn(train_state.params, hidden_state_LBW_, ac_in_1B_)
                action_logits_BA = action_logits_1BA.squeeze()
                value_B = value_1B.squeeze()
                pi_BA_ = distrax.Categorical(action_logits_BA)
                rng, _rng = jax.random.split(rng)
                sampled_action_B = pi_BA_.sample(seed=_rng)
                action_B = sampled_action_B * (predetermined_action_B == -1) + predetermined_action_B * (predetermined_action_B != -1)
                log_prob_B = pi_BA_.log_prob(action_B)
                env_act_E1_ = unbatchify(action_B, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act_E_ = jax.tree.map(lambda x: x.squeeze(), env_act_E1_)

                if teacher is None:
                    teacher_action_B = jnp.zeros_like(action_B)
                    teacher_valid_B = jnp.zeros_like(action_B, dtype=jnp.bool_)
                    teacher_active_B = jnp.zeros_like(action_B, dtype=jnp.bool_)
                else:
                    teacher_action_B, teacher_valid_B, teacher_active_B, teacher_memories, teacher_rng = teacher_step(
                        teacher, env.agents, obs_EO_, avail_actions_EA_,
                        done_B.reshape((env.num_agents, config["NUM_ENVS"])),
                        teacher_memories, teacher_rng,
                    )

                # ENCODE HISTORY
                next_bm_hidden_state_LBW_, encoded_state_BW = bm_train_state.apply_fn(
                    bm_train_state.params,
                    bm_hidden_state_LBW_, obs_BO, done_B,
                    method=BeliefModel.encode
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_E_ = jax.random.split(_rng, config["NUM_ENVS"])
                next_obs_EO_, next_env_state_E_, reward_E_, next_done_E_, info_EP_ = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_E_, env_state_E_, env_act_E_
                )
                reward_B = batchify(reward_E_, env.agents, config["NUM_ACTORS"]).squeeze()
                next_done_B = batchify(next_done_E_, env.agents, config["NUM_ACTORS"]).squeeze()
                info_B_ = jax.tree.map(lambda x: x.swapaxes(0, 1).reshape((config["NUM_ACTORS"])), info_EP_)
                transition_B_ = Transition(
                    done_B,
                    action_B,
                    value_B,
                    reward_B,
                    log_prob_B,
                    obs_BO,
                    hand_BY,
                    info_B_,
                    avail_actions_BA,
                    env_state_E_, # Technically doesn't have leading dimension B, but E. Doesn't matter though, as only used to create targets
                    hidden_state_LBW_,
                    encoded_state_BW,
                    hand_plus_deck_BX,
                    card_knowledge_BHX,
                    teacher_action_B,
                    teacher_valid_B,
                    teacher_active_B,
                )
                runner_state = (train_state, bm_train_state, next_env_state_E_, next_obs_EO_, next_done_B, teacher_memories, teacher_rng, next_hidden_state_LBW_, next_bm_hidden_state_LBW_, rng)
                return runner_state, transition_B_

            predetermined_actions_TB = - jnp.ones((config["NUM_STEPS"], config["NUM_ACTORS"]), dtype=int)
            runner_state, traj_batch_TB_ = jax.lax.scan(
                _env_step, runner_state, predetermined_actions_TB, config["NUM_STEPS"]
            )

            train_state, bm_train_state, last_env_state_E_, last_obs_EO_, last_done_B, teacher_memories, teacher_rng, last_hidden_state_LBW_, last_bm_hidden_state_LBW_, rng = runner_state

            # COMPUTE TD(LAMBDA) TARGETS
            def _compute_targets(last_env_state_E_, last_obs_EO_, last_done_B, last_hidden_state_LBW_, traj_batch_TB_):
                last_obs_BO = batchify(last_obs_EO_, env.agents, config["NUM_ACTORS"])
                last_hand_EPHCR = last_env_state_E_.env_state.player_hands
                last_hand_BY = last_hand_EPHCR.swapaxes(0, 1).reshape((config["NUM_ACTORS"],-1))
                avail_actions = jax.vmap(env.get_legal_moves)(last_env_state_E_.env_state)
                avail_actions_BA = jax.lax.stop_gradient(batchify(avail_actions, env.agents, config["NUM_ACTORS"]))
                ac_in_1B_ = (last_obs_BO[None], last_hand_BY[None], last_done_B[None], avail_actions_BA[None])
                _, _, last_value_1B = train_state.apply_fn(train_state.params, last_hidden_state_LBW_, ac_in_1B_)
                last_value_B = last_value_1B.squeeze()

                def _compute_targets_from_gae(traj_batch_TB_, last_value_B, last_done_B):
                    def _get_gae(gae_and_next_value_B_, transition_B_):
                        gae_B, next_value_B, next_done_B = gae_and_next_value_B_
                        done_B, value_B, reward_B = (
                            transition_B_.done,
                            transition_B_.value,
                            transition_B_.reward,
                        )
                        delta_B  = reward_B + config["GAMMA"] * next_value_B * (1 - next_done_B) - value_B
                        gae_B  = delta_B  + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done_B) * gae_B
                        return (gae_B, value_B, done_B), gae_B

                    init_B_ = (jnp.zeros_like(last_value_B), last_value_B, last_done_B)
                    _, gae_TB = jax.lax.scan(
                        _get_gae,
                        init_B_,
                        traj_batch_TB_,
                        reverse=True,
                        unroll=16,
                    )
                    return gae_TB + traj_batch_TB_.value

                targets_TB = _compute_targets_from_gae(traj_batch_TB_, last_value_B, last_done_B)
                return targets_TB

            targets_TB = _compute_targets(
                last_env_state_E_, last_obs_EO_, last_done_B, last_hidden_state_LBW_, traj_batch_TB_
            )
            gae_TB = targets_TB - traj_batch_TB_.value
            bm_metrics_TE_ = {
                "valid_percentage_hands": jnp.ones(1),
                "entropy_legal": jnp.zeros(1),
                "entropy_count": jnp.zeros(1),
            }

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    train_state, bm_train_state = train_states
                    init_hidden_state_LMW_, bm_init_hidden_state_LMW_, traj_batch_TM_, gae_TM, targets_TM = batch_info

                    def _loss_fn(params):
                        mask_TM = (1 < traj_batch_TM_.avail_actions.sum(axis=-1))
                        if config["ACTING_PLAYER_ONLY_ACTOR"]:
                            actor_mask_TM = mask_TM
                        else:
                            actor_mask_TM = 1 + jnp.zeros_like(mask_TM)
                        if config["ACTING_PLAYER_ONLY_CRITIC"]:
                            critic_mask_TM = mask_TM
                        else:
                            critic_mask_TM = 1 + jnp.zeros_like(mask_TM)

                        # RERUN NETWORK
                        _, action_logits_TMA, value_TM = train_state.apply_fn(
                            params,
                            init_hidden_state_LMW_,
                            (traj_batch_TM_.obs, traj_batch_TM_.hand, traj_batch_TM_.done, traj_batch_TM_.avail_actions),
                        )
                        pi_TMA_ = distrax.Categorical(action_logits_TMA)
                        log_prob_TM = pi_TMA_.log_prob(traj_batch_TM_.action)

                        # CALCULATE CRITIC LOSS
                        value_pred_clipped_TM = traj_batch_TM_.value + (
                                value_TM - traj_batch_TM_.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_TM = jnp.square(value_TM - targets_TM)
                        value_losses_clipped_TM = jnp.square(value_pred_clipped_TM - targets_TM)
                        value_losses_TM = jnp.maximum(value_losses_TM, value_losses_clipped_TM)
                        value_loss = 0.5 * (critic_mask_TM * value_losses_TM).sum()/critic_mask_TM.sum()

                        # CALCULATE ACTOR LOSS
                        logratio_TM = log_prob_TM - traj_batch_TM_.log_prob
                        ratio_TM = jnp.exp(logratio_TM)
                        gae_mean = jnp.sum(actor_mask_TM * gae_TM) / actor_mask_TM.sum()
                        gae_std = jnp.sqrt(
                            jnp.sum(actor_mask_TM * (gae_TM - gae_mean)**2) / actor_mask_TM.sum()
                        )
                        gae_normalized_TM = (gae_TM - gae_mean) / (gae_std + 1e-10)
                        actor_losses_TM = ratio_TM * gae_normalized_TM
                        actor_losses_clipped_TM = (
                                jnp.clip(
                                    ratio_TM,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae_normalized_TM
                        )
                        actor_losses_TM = -jnp.minimum(actor_losses_TM, actor_losses_clipped_TM)
                        actor_loss = (actor_mask_TM * actor_losses_TM).sum() / actor_mask_TM.sum()

                        # CALCULATE TOTAL LOSS
                        entropy = (actor_mask_TM * pi_TMA_.entropy()).sum() / actor_mask_TM.sum()

                        total_loss = (
                                actor_loss
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )

                        ce_loss = jnp.asarray(0.0)
                        if teacher is not None:
                            ce_loss = imitation_loss(
                                action_logits_TMA, traj_batch_TM_.teacher_action, traj_batch_TM_.teacher_valid
                            )
                            total_loss = total_loss + config["CE_COEF"] * ce_loss

                        # EXTRA DIAGNOSTICS
                        clip_frac = (actor_mask_TM * (jnp.abs(ratio_TM - 1) > config["CLIP_EPS"])).sum() / actor_mask_TM.sum()
                        ratio = ratio_TM.mean()

                        return total_loss, (value_loss, actor_loss, entropy, clip_frac, ratio, ce_loss)

                    if config["TRAIN_POLICY"]:
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                    else:
                        total_loss = (0, (0, 0, 0, 0, 0, 0))

                    bm_total_loss = (0, (0, 0))

                    return (train_state, bm_train_state), (total_loss, bm_total_loss)

                train_state, bm_train_state, init_hidden_state_LBW_, bm_init_hidden_state_LBW_, traj_batch_TB_, gae_TB, targets_TB, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (init_hidden_state_LBW_, bm_init_hidden_state_LBW_, traj_batch_TB_, gae_TB, targets_TB)

                # SHUFFLE ALONG THE BATCH AXIS
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1, 0,
                    ),
                    shuffled_batch,
                )

                (train_state, bm_train_state), (total_loss, bm_total_loss) = jax.lax.scan(
                    _update_minbatch, (train_state, bm_train_state), minibatches
                )
                update_state = (train_state, bm_train_state, init_hidden_state_LBW_, bm_init_hidden_state_LBW_, traj_batch_TB_, gae_TB, targets_TB, rng)
                return update_state, (total_loss, bm_total_loss)

            update_state = (train_state, bm_train_state, init_hidden_state_LBW_, bm_init_hidden_state_LBW_, traj_batch_TB_, gae_TB, targets_TB, rng)
            update_state, (loss_info, bm_loss_info) = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state, bm_train_state = update_state[0], update_state[1]
            metric_TB_ = traj_batch_TB_.info
            metric = jax.tree.map(lambda x: x[-1].mean(), metric_TB_)

            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            bm_loss_info = jax.tree.map(lambda x: x.mean(), bm_loss_info)
            bm_metrics = jax.tree.map(lambda x: x.mean(), bm_metrics_TE_)

            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "clip_frac": loss_info[1][3],
                "ratio": loss_info[1][4],
                "ce_loss": loss_info[1][5],
                "weighted_ce_loss": config["CE_COEF"] * loss_info[1][5],
                "ce_coefficient": jnp.asarray(config["CE_COEF"]),
                "teacher_valid_label_count": traj_batch_TB_.teacher_valid.sum(),
                "teacher_active_count": traj_batch_TB_.teacher_active.sum(),
                "teacher_invalid_label_rate": (
                    (traj_batch_TB_.teacher_active & ~traj_batch_TB_.teacher_valid).sum()
                    / jnp.maximum(traj_batch_TB_.teacher_active.sum(), 1)
                ),
                "student_teacher_action_agreement": (
                    ((traj_batch_TB_.action == traj_batch_TB_.teacher_action) & traj_batch_TB_.teacher_valid).sum()
                    / jnp.maximum(traj_batch_TB_.teacher_valid.sum(), 1)
                ),
                "valid_percentage_hands": bm_metrics["valid_percentage_hands"],
                "entropy_learned_BM": bm_metrics["entropy_legal"],
                "entropy_count_BM": bm_metrics["entropy_count"],
                "loss_belief_model": bm_loss_info[0],
                "entropy_belief_model": bm_loss_info[1][0],
                "loss_count": bm_loss_info[1][1],
            }

            def callback(metric, process_id, update_steps):
                metric["loss"] = {key + f"_{process_id}": metric["loss"][key] for key in metric["loss"]}
                dict_to_log = {
                    f"SP_{process_id}": metric["returned_episode_returns"],
                    f"UpdateStep": update_steps,
                    **metric["loss"],
                }
                write_metrics(run_output_dir, dict_to_log)

            jax.lax.cond(
                update_steps % 10 == 0,
                lambda : jax.experimental.io_callback(callback, None, metric, process_id, update_steps),
                lambda : None,
            )

            rng = update_state[-1]
            update_steps = update_steps + 1

            runner_state = (train_state, bm_train_state, last_env_state_E_, last_obs_EO_, last_done_B, teacher_memories, teacher_rng, last_hidden_state_LBW_, last_bm_hidden_state_LBW_, rng)

            update_runner_state = (runner_state, update_steps)
            return update_runner_state, None

        update_state, metric = jax.lax.scan(_update_step, update_state, None, config["NUM_UPDATES_PER_INNER_LOOP"])
        return update_state

    def train(rng):
        _, init_key = jax.random.split(rng)
        device_keys = jax.random.split(init_key, config["NUM_GPUS"])
        states = jax.pmap(_get_init_update_state, axis_name="devices")(
            device_keys, initial_params, initial_params)
        update = jax.pmap(update_agents, axis_name="devices")
        best_score = -np.inf
        for outer in range(1, config["NUM_OUTER_UPDATES"] + 1):
            states = update(states, jnp.arange(config["NUM_GPUS"]))
            jax.block_until_ready(states)
            ppo_update = int(states[1][0])
            params = jax.tree.map(lambda x: x[0], states[0][0].params)
            if evaluate_code_policy is not None:
                metrics = evaluate_code_policy([params], run_output_dir, outer, ppo_update)
                score = metrics["teacher_eval/greedy/mean_return"]
                if score > best_score:
                    best_score = score
                    best_dir = run_output_dir / "best_teacher_eval"
                    save_student(params, best_dir)
                    selection = dict(metric="teacher_eval/greedy/mean_return", value=score,
                                     outer_update=outer, ppo_update=ppo_update,
                                     teacher_bundle_sha256=code_policy.source_hash,
                                     evaluation_seed=config["TEACHER_EVAL_SEED"],
                                     evaluation_results=f"teacher_eval/update_{outer:02d}/greedy.json")
                    (best_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
                write_metrics(run_output_dir, {"UpdateStep": ppo_update, **metrics})
            save_student(params, run_output_dir)
            print(f"Finished evaluation interval {outer}, PPO update {ppo_update}", flush=True)

    return train


def main():
    config = json.loads((Path(__file__).parent / 'hanabi/paper_config.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=['hanabi_2p', 'hanabi_2p_h_group_level_1'], default='hanabi_2p',
                        help='Both tasks use standard two-player dynamics. H-Group supplies teacher conventions.')
    parser.add_argument('--teacher', type=Path, help='Seed directory or saved Python policy')
    parser.add_argument('--output', type=Path, required=True, help='New training output directory')
    parser.add_argument('--ce-weight', type=float, default=config['CE_COEF'])
    parser.add_argument('--seed', type=int, default=config['SEED'])
    parser.add_argument('--updates', type=int, help='Override the paper budget with this many PPO updates')
    parser.add_argument('--num-envs', type=int, default=config['NUM_ENVS'])
    parser.add_argument('--rollout-steps', type=int, default=config['NUM_STEPS'])
    parser.add_argument('--epochs', type=int, default=config['UPDATE_EPOCHS'])
    parser.add_argument('--hidden-size', type=int, default=config['LAYER_WIDTH'])
    parser.add_argument('--learning-rate', type=float, default=config['LR'])
    parser.add_argument('--eval-episodes', type=int, default=config['TEACHER_EVAL_EPISODES'])
    args = parser.parse_args()
    if min(args.num_envs, args.rollout_steps, args.epochs, args.hidden_size, args.eval_episodes) < 1:
        parser.error('Training and evaluation dimensions must be positive.')
    if args.num_envs < 2 or args.num_envs % 2:
        parser.error('Use an even environment count so the four recurrent minibatches are equally sized.')
    if args.updates is not None and args.updates < 1:
        parser.error('Supply a positive number of PPO updates.')
    if not 0 <= args.seed < 2**32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Supply a uint32 seed and a positive finite learning rate.')
    if not math.isfinite(args.ce_weight) or args.ce_weight < 0 or (args.ce_weight and not args.teacher):
        parser.error('Supply a saved teacher for positive CE or use --ce-weight 0 for ordinary self-play.')
    teacher = args.teacher
    if teacher is not None:
        teacher = teacher / 'policy.py' if teacher.is_dir() else teacher
        if not teacher.is_file():
            parser.error(f'Teacher policy does not exist: {teacher}')
    if jax.device_count() != 1:
        parser.error('Select one device for each student replica, as in the paper run.')
    config.update(SEED=args.seed, CE_COEF=args.ce_weight, NUM_ENVS=args.num_envs, NUM_STEPS=args.rollout_steps,
                  LAYER_WIDTH=args.hidden_size, UPDATE_EPOCHS=args.epochs, LR=args.learning_rate,
                  TEACHER_EVAL=teacher is not None, TEACHER_EVAL_EPISODES=args.eval_episodes,
                  TEACHER_POLICY_PATH=str(teacher.resolve()) if teacher is not None else None)
    if args.updates is not None:
        config['TOTAL_TIMESTEPS'] = args.updates * args.num_envs * args.rollout_steps
        config['NUM_OUTER_UPDATES'] = math.gcd(args.updates, config['NUM_OUTER_UPDATES'])
    if config['TOTAL_TIMESTEPS'] // args.num_envs // args.rollout_steps < config['NUM_OUTER_UPDATES']:
        parser.error('The timestep budget must allow at least one update per evaluation interval.')
    args.output.mkdir(parents=True, exist_ok=False)
    train = make_train(config, args.output)
    train(jax.random.PRNGKey(args.seed))


if __name__ == '__main__':
    main()
