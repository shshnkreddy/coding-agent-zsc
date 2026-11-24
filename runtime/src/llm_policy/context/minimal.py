"""Render a data-only contract and a valid, algorithm-free policy starter."""

import hashlib
import json


def starter(contract):
    adapter = contract.get("adapter", "overcooked_v2")
    names = {
        "cat_dog": ("num_agents", "observation_size", "num_actions"),
        "overcooked_v2": ("max_height", "max_width", "num_ingredients"),
        "hanabi": ("num_agents", "num_colors", "num_ranks", "hand_size", "observation_size", "num_actions"),
        "yokai": ("num_agents", "num_colors", "deck_size", "grid_size", "num_hints", "num_nodes", "node_features", "num_actions"),
    }[adapter]
    metadata = {"name": "bare_candidate", "adapter": adapter, "adapter_version": 1,
                **{name: contract[name] for name in names}}
    if adapter == "overcooked_v2" and int(contract.get("num_agents", 2)) == 3:
        metadata["num_agents"] = 3
    shared = contract.get("policy_structure") == "shared_local"
    if adapter != "overcooked_v2":
        metadata["policy_structure"] = contract["policy_structure"]
    if adapter == "overcooked_v2":
        from jaxmarl.environments.overcooked_v2.common import Actions
        action = f"jnp.asarray({int(Actions.stay)}, dtype=jnp.int32)"
    else:
        action = "jnp.argmax(obs.legal_actions).astype(jnp.int32)"
    source = ('"""Valid interface-only starter; implement all policy logic here."""\n\n'
              'import jax.numpy as jnp\n\n'
              'POLICY_API_VERSION = 1\nPOLICY_METADATA = ' + repr(metadata) + '\n')
    num_agents = int(contract.get("num_agents", 2))
    for suffix in (("",) if shared else tuple(f"_{index}" for index in range(num_agents))):
        source += f'''

def init_agent{suffix}():
    return jnp.asarray(0, dtype=jnp.int32)


def agent{suffix}_act(obs, reset, memory, rng):
    action = {action}
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {{}}
'''
    return source


def policy_api(contract):
    adapter = contract.get("adapter", "overcooked_v2")
    observation = {
        "cat_dog": "Observations use CatDogObservation, imported from llm_policy.policy_api.",
        "overcooked_v2": "Observations are JAX arrays.",
        "hanabi": "Observations use HanabiObservation, imported from llm_policy.policy_api.",
        "yokai": "Observations use YokaiObservation, imported from llm_policy.policy_api.",
    }[adapter]
    shared = contract.get("policy_structure") == "shared_local"
    num_agents = int(contract.get("num_agents", 2))
    functions = ("init_agent() and agent_act(obs, reset, memory, rng)" if shared else
                 ", ".join(
                     [*(f"init_agent_{index}()" for index in range(num_agents)),
                      *(f"agent_{index}_act(obs, reset, memory, rng)" for index in range(num_agents))]
                 ))
    xp_note = ("For three-player OCV2, each module contains three seat-specific roles. Population\n"
               "XP cell [A,B] runs A's agent_0 role with B's agent_1 and agent_2 roles (A/BB);\n"
               "the diagonal is AAA self-play. The transpose is B/AA, not AA/B.\n\n"
               if adapter == "overcooked_v2" and num_agents == 3 else "")
    if adapter == "hanabi" and num_agents > 2:
        xp_note = (f"Directed population XP cell [A,B] uses A in seat 0 and private instances "
                   f"of B in seats 1..{num_agents-1}. The diagonal is self-play. There is no seat rotation.\n\n")
    return f"""# Policy interface

Keep POLICY_API_VERSION=1 and the starter's metadata. Implement {functions}.
Each initializer returns a private fixed-shape JAX PyTree. Each act call returns
(scalar integer action, next_memory, trace). Memory structure, shapes and
dtypes must match the value returned by the initializer. Honor reset; seats never share private memory.
Use only local observations and the supplied PRNG, never hidden state or reward.
Initial memory must contain at least one array leaf: an empty dict, empty tuple
or None is not sufficient. A stateless policy can initialize memory with
`jnp.asarray(0, dtype=jnp.int32)` and return it unchanged. An empty trace dict
is still allowed.

{observation}
Trace is a dictionary whose string keys and contents you choose, mapping to
fixed-shape numeric or boolean JAX arrays. Return {{}} when no diagnostics are
needed. Keys, shapes and dtypes must remain fixed across calls for each act
function; different policies and seats may use different trace schemas.
There are no required diagnostic fields or predefined meanings. The runner
records actual actions separately. Trace is recorded only for inspection: it
is not passed into the next act call, shared with another seat, or used to
compute reward. Only next_memory is carried into the next decision, and that
memory is reset to the value returned by the initializer at episode boundaries.

{xp_note}See environment/README.md and environment/specification.json for observation
semantics, shapes, action encodings and rules. All implementations beyond the
bare interface are yours, including observation decoding and memory updates.
No supplied primitive libraries may be imported. You may write your own
custom_primitives.py.

## Execution requirements

The runner applies `jax.jit` and `jax.vmap` to each act function to compile it
and run many episodes in parallel. Write the function for one observation and
one scalar action; the runner supplies the batch dimension. Use `jnp.where`,
`jax.lax.cond` or `jax.lax.switch` for choices depending on array values, not
Python `if` or conversions such as `bool(array)` or `int(array)`. Keep array
shapes fixed. Carry changing state only through next_memory, never mutable
globals; use the supplied PRNG for randomness. Runtime I/O, subprocesses,
network access and host callbacks are not allowed.

The policy validator allows these external import modules: `__future__`,
`chex`, `dataclasses`, `functools`, `jax`, `jax.lax`, `jax.numpy`, `math`, and
`typing`. The environment-specific data types documented above may also be
imported from `llm_policy.policy_api`. Ordinary `numpy` is not allowed.
For randomness, use `import jax` and `jax.random`; a direct `import jax.random`
is not on the allowed import list.

Function calls at module or class scope are rejected, including
`TABLE = jnp.array([0, 1])`. Define literal constants there and construct arrays
inside an initializer, helper or act function. Python `while` loops, global
or nonlocal mutation declarations, context managers, async functions,
generators, wildcard imports, private/interpreter attributes, and introspection
calls such as `getattr`, `setattr`, `globals` or `vars` are also rejected.
These are evaluator restrictions in addition to JAX compatibility.

policy-eval validate/score/trace and context.rollout_api support development.
Rollout exploration exposes observations and returns, not simulator source.
Only policy-eval score records an official development measurement.
"""


def dossier(environment, objective, contract, profile, guidance, guidance_mode, meta_update_id,
            *, runtime_env=None):
    from llm_policy.context.guidance import validate_guidance_mode

    guided = validate_guidance_mode(guidance_mode, guidance, meta_update_id)
    record = dict(schema_version=1, objective=objective.to_record(),
                  runtime_contract={key: value for key, value in contract.items()
                                    if key not in ("source_sha256", "belief_mode")},
                  environment=dict(env_name=environment.env_name, env_kwargs=environment.env_kwargs),
                  context_profile=profile.record(),
                  meta_guidance=dict(mode=guidance_mode, meta_update_id=meta_update_id, present=guided,
                                     sha256=hashlib.sha256(guidance.encode()).hexdigest()))
    text = f"""# Policy-search task

## Objective: {objective.name}

{objective.task_instruction}

{objective.final_selection_instruction}

Read context/environment/README.md and context/environment/specification.json for the exact game,
and context/policy_api.md for the required interface. These describe mechanics,
not a strategy. Runtime seats receive only their own observation, reset flag,
private memory and PRNG. Development rollouts may inspect public observations,
rewards and traces; hidden environment state and implementation are not exposed.

{"Read context/policy_guidance.md subject to the task contract." if guided else "No policy guidance is supplied."}

Disclosure profile: {json.dumps(profile.record(), sort_keys=True)}.
"""
    if contract.get("adapter") == "hanabi":
        from llm_policy.environments.hanabi.context import score_completion_nudge

        text += "\n" + score_completion_nudge(contract)
    elif environment.env_name.lower() in ("overcooked_v2", "pcg_overcooked_v2"):
        from llm_policy.context.task_dossier import _delivery_throughput_heuristic

        throughput_record, throughput_markdown = _delivery_throughput_heuristic(environment, runtime_env)
        record["search_heuristics"] = {"delivery_throughput": throughput_record}
        text += "\n## Delivery-throughput sanity check\n\n" + throughput_markdown + "\n"
    return record, text
