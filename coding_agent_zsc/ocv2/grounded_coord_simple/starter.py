"""Valid interface-only starter; implement all policy logic here."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'overcooked_v2', 'adapter_version': 1, 'max_height': 5, 'max_width': 8, 'num_ingredients': 3}


def init_agent_0():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_0_act(obs, reset, memory, rng):
    action = jnp.asarray(4, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}


def init_agent_1():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_1_act(obs, reset, memory, rng):
    action = jnp.asarray(4, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}
