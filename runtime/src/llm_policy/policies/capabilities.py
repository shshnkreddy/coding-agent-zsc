"""Machine-readable capability manifest for policy-generation prompts."""

from __future__ import annotations


POLICY_CAPABILITIES = {
    "api_version": 1,
    "adapter": "overcooked_v2",
    "information_boundary": {
        "available": [
            "the acting role's local observation",
            "that role's private fixed-shape memory",
            "a JAX PRNG key",
            "an episode-reset flag",
        ],
        "unavailable": [
            "hidden environment state",
            "the other role's observation or memory",
            "reward and return",
            "absolute layout coordinates",
            "host I/O and network access",
        ],
    },
    "module_contract": {
        "constants": ["POLICY_API_VERSION", "POLICY_METADATA"],
        "functions": [
            "init_agent_0()",
            "init_agent_1()",
            "agent_0_act(obs, reset, memory, rng)",
            "agent_1_act(obs, reset, memory, rng)",
        ],
        "act_return": "(scalar integer action, same-structure memory, PolicyTrace)",
    },
    "facts": [
        "self facing and inventory",
        "visible partner pose, facing, and inventory",
        "visible recipe counts and source",
        "visible pot contents and cooking state",
        "visible buttons, indicators, piles, goals, walls, and floor",
        "last movement blocked and observation/inventory changes",
    ],
    "memory": [
        "agent-relative belief map",
        "inferred relative pose",
        "remembered object and pot locations",
        "active primitive progress",
        "policy-defined JAX arrays and dataclasses",
    ],
    "queries": [
        "target(object_type, ingredient_id, pot_state, source, selection)",
        "ingredient_pile(ingredient_id, source, selection)",
        "pot(state, source, selection)",
        "target_known / target_visible / target_adjacent",
        "target_relative_direction / target_known_distance / path_known",
    ],
    "skills": [
        "stay",
        "move",
        "interact",
        "face_direction",
        "face_target",
        "navigate_to",
        "interact_with",
        "explore_frontier",
        "wait",
    ],
    "composition": [
        "select_result",
        "select_trace",
        "select_pytree",
        "make_trace",
        "cancel_active_skill",
        "status_is",
    ],
    "ingredient_arithmetic": [
        "missing_ingredient_counts",
        "first_nonzero_ingredient",
        "sample_nonzero_ingredient",
    ],
}


def capability_manifest(adapter="overcooked_v2"):
    """Return the stable JSON-serializable prompt manifest."""
    if adapter == "yokai":
        from llm_policy.environments.yokai.context import capabilities
        return capabilities()
    if adapter == "hanabi":
        from llm_policy.environments.hanabi.context import capabilities
        return capabilities()
    if adapter != "overcooked_v2":
        raise ValueError(f"unknown policy adapter: {adapter}")
    return POLICY_CAPABILITIES


__all__ = ["POLICY_CAPABILITIES", "capability_manifest"]
