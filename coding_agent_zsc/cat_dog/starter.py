"""Valid interface-only starter; implement all policy logic here."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'cat_dog', 'adapter_version': 1, 'num_agents': 2, 'observation_size': 10, 'num_actions': 4, 'policy_structure': 'separate_roles'}


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    action = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    action = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}
