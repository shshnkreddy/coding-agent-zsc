"""Valid interface-only starter; implement all policy logic here."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'hanabi', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 5, 'num_ranks': 5, 'hand_size': 5, 'observation_size': 658, 'num_actions': 21, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    action = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}
