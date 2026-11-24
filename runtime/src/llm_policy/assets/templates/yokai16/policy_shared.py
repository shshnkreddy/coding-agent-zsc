"""Neutral legal-random Yokai starter, instantiated privately in each seat."""

from llm_policy import yokai16_primitives as p

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "yokai_candidate", "adapter": "yokai", "adapter_version": 1,
    "policy_structure": "shared_local",
    "num_agents": 2, "num_colors": 4, "deck_size": 16, "grid_size": 10,
    "num_hints": 7, "num_nodes": 23, "node_features": 15, "num_actions": 2325,
}


def init_agent():
    return p.init_memory()


def agent_act(obs, reset, memory, rng):
    facts, memory = p.observe(obs, reset, memory)
    action = p.sample_legal(facts.legal_actions, rng)
    return action, memory, p.make_trace(action, phase_id=facts.substep)
