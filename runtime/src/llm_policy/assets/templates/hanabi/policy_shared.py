"""Neutral legal-random local policy, instantiated privately for each player."""

from llm_policy import hanabi_primitives as p

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "hanabi_candidate", "adapter": "hanabi", "adapter_version": 1,
    "policy_structure": "shared_local",
    "num_agents": 2, "num_colors": 5, "num_ranks": 5, "hand_size": 5,
    "observation_size": 658, "num_actions": 21,
}


def init_agent():
    return p.init_knowledge()


def agent_act(obs, reset, memory, rng):
    facts, memory = p.observe(obs, reset, memory)
    action = p.sample_legal(facts.legal_actions, rng)
    return action, memory, p.make_trace(action)
