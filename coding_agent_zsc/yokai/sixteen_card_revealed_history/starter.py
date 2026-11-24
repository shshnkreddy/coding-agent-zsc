"""Valid interface-only starter; implement all policy logic here."""

import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {'name': 'bare_candidate', 'adapter': 'yokai', 'adapter_version': 1, 'num_agents': 2, 'num_colors': 4, 'deck_size': 16, 'grid_size': 10, 'num_hints': 7, 'num_nodes': 23, 'node_features': 15, 'num_actions': 2325, 'policy_structure': 'shared_local'}


def init_agent():
    return jnp.asarray(0, dtype=jnp.int32)


def agent_act(obs, reset, memory, rng):
    action = jnp.argmax(obs.legal_actions).astype(jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return action, jnp.where(reset, zero, memory), {}
