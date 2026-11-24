"""Editable starting point for one decentralized OvercookedV2 joint policy."""

import jax.numpy as jnp

from llm_policy import overcooked_primitives as p


POLICY_API_VERSION = 1
TASK_MAX_HEIGHT = 3
TASK_MAX_WIDTH = 5
TASK_NUM_INGREDIENTS = 2
POLICY_METADATA = {
    "name": "overcooked_candidate",
    "adapter": "overcooked_v2",
    "adapter_version": 1,
    "max_height": TASK_MAX_HEIGHT,
    "max_width": TASK_MAX_WIDTH,
    "num_ingredients": TASK_NUM_INGREDIENTS,
}


def init_agent_0():
    """Initialize agent 0's private fixed-shape memory."""
    return p.init_controller(
        max_height=TASK_MAX_HEIGHT,
        max_width=TASK_MAX_WIDTH,
        num_ingredients=TASK_NUM_INGREDIENTS,
    )


def init_agent_1():
    """Initialize agent 1's private fixed-shape memory."""
    return p.init_controller(
        max_height=TASK_MAX_HEIGHT,
        max_width=TASK_MAX_WIDTH,
        num_ingredients=TASK_NUM_INGREDIENTS,
    )


def _starting_action(obs, reset, controller, rng, *, role_id):
    """Valid stay-only skeleton; replace this with policy logic."""
    del rng
    facts, controller = p.observe(obs, reset, controller)

    # Implement policy logic here using the supported API.
    result = p.stay(facts, controller)
    phase_id = jnp.asarray(0, dtype=jnp.int32)
    decision_id = jnp.asarray(role_id, dtype=jnp.int32)
    return result.action, result.controller, p.make_trace(
        phase_id, decision_id, result
    )


def agent_0_act(obs, reset, memory, rng):
    """Choose agent 0's action from only its observation and private memory."""
    return _starting_action(obs, reset, memory, rng, role_id=0)


def agent_1_act(obs, reset, memory, rng):
    """Choose agent 1's action from only its observation and private memory."""
    return _starting_action(obs, reset, memory, rng, role_id=1)


TRACE_LABELS = {
    "agent_0": {0: "starting_policy"},
    "agent_1": {0: "starting_policy"},
}
