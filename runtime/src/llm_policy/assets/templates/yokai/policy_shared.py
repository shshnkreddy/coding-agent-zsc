"""Neutral legal-random Yokai starter, instantiated privately in each seat."""

from llm_policy import yokai_primitives as p

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "yokai_candidate", "adapter": "yokai", "adapter_version": 1,
    "policy_structure": "shared_local",
    "num_agents": 2, "num_colors": 3, "deck_size": 9, "grid_size": 9,
    "num_hints": 4, "num_nodes": 13, "node_features": 13, "num_actions": 1068,
}


def init_agent():
    return p.init_memory()


def agent_act(obs, reset, memory, rng):
    facts, memory = p.observe(obs, reset, memory)
    action = p.sample_legal(facts.legal_actions, rng)
    return action, memory, p.make_trace(action, phase_id=facts.substep)
