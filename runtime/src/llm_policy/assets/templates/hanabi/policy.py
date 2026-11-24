"""Neutral legal-random Hanabi starter; both roles have independent memory."""

from llm_policy import hanabi_primitives as p

POLICY_API_VERSION = 1
POLICY_METADATA = {
    "name": "hanabi_candidate", "adapter": "hanabi", "adapter_version": 1,
    "policy_structure": "separate_roles",
    "num_agents": 2, "num_colors": 5, "num_ranks": 5, "hand_size": 5,
    "observation_size": 658, "num_actions": 21,
}


def init_agent_0():
    return p.init_knowledge()


def init_agent_1():
    return p.init_knowledge()


def agent_0_act(obs, reset, memory, rng):
    facts, memory = p.observe(obs, reset, memory)
    action = p.sample_legal(facts.legal_actions, rng)
    return action, memory, p.make_trace(action)


def agent_1_act(obs, reset, memory, rng):
    facts, memory = p.observe(obs, reset, memory)
    action = p.sample_legal(facts.legal_actions, rng)
    return action, memory, p.make_trace(action)
