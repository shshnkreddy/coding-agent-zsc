"""Build compact, deterministic task specifications for policy-search agents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_policy.policies.capabilities import capability_manifest
from llm_policy.context.environment import (
    ResolvedEnvironment,
    build_environment_briefing,
    resolved_runtime_contract,
)
from llm_policy.context.objectives import PolicyObjective


def _compact_capabilities():
    capabilities = capability_manifest()
    return {
        name: capabilities[name]
        for name in (
            "api_version",
            "adapter",
            "information_boundary",
            "module_contract",
            "facts",
            "queries",
            "skills",
            "composition",
            "ingredient_arithmetic",
        )
    }


def _delivery_throughput_heuristic(
    environment: ResolvedEnvironment,
    runtime_env=None,
):
    """Describe repeat-delivery pressure without asserting a known optimum."""
    kwargs = environment.env_kwargs

    def effective_value(name, default=None):
        if name in kwargs:
            return kwargs[name]
        return getattr(runtime_env, name, default)

    max_steps = effective_value("max_steps")
    sample_recipe_on_delivery = bool(
        effective_value("sample_recipe_on_delivery", False)
    )
    terminate_on_delivery = bool(effective_value("terminate_on_delivery", False))
    repeats = not terminate_on_delivery
    record = {
        "applicable": repeats,
        "formula": (
            "max_steps / estimated_full_cook_and_delivery_cycle_steps"
            if repeats
            else None
        ),
        "max_steps": int(max_steps) if max_steps is not None else None,
        "reference_cycle_steps": None,
        "reference_expected_deliveries": None,
        "is_known_optimum": False,
    }
    if not repeats:
        return record, (
            "A successful delivery ends the episode, so no repeated-delivery "
            "target is suggested."
        )

    reference = ""
    if (
        environment.env_name.lower()
        in {"overcooked_v2", "pcg_overcooked_v2"}
        and int(max_steps or 0) == 400
    ):
        # This is deliberately a suite-specific reference, not an environment
        # invariant.  Other horizons and layouts must estimate their own cycle.
        reference_cycle_steps = 40
        reference_expected = int(max_steps) / reference_cycle_steps
        record["reference_cycle_steps"] = reference_cycle_steps
        record["reference_expected_deliveries"] = reference_expected
        reference = (
            " For the standard 400-step OCV2 tasks, a useful nominal full-cycle "
            "estimate is about 40 steps, so a good throughput-oriented policy "
            f"should aim for roughly 400 / 40 = {reference_expected:g} deliveries."
        )

    recipe_after_delivery = (
        "the recipe is resampled after delivery"
        if sample_recipe_on_delivery
        else "the same fixed recipe remains active after delivery"
    )
    return record, (
        "This is a continuing-delivery task: a successful delivery does not end "
        f"the episode, {recipe_after_delivery}, and episode return accumulates. "
        "Use expected deliveries ≈ max_steps / estimated full cook-and-delivery "
        "cycle steps as a throughput sanity check."
        f"{reference} The cycle estimate includes communication, ingredient "
        "collection, cooking, plating, delivery, and reset/re-arming; estimate it "
        "for other layouts or horizons from geometry and exploratory rollouts, "
        "then replace it with observed inter-delivery time after two deliveries. "
        "This is only a heuristic, not a known maximum or a final stopping target. "
        "In particular, one delivery is not evidence of saturation when the "
        "remaining horizon can plausibly fit another cycle."
    )


def build_task_dossier(
    environment: ResolvedEnvironment,
    objective: PolicyObjective,
    *,
    runtime_env=None,
    policy_guidance: str = "",
    guidance_mode: str = "standalone_no_guidance",
    meta_update_id: str | None = None,
):
    """Return JSON-safe provenance and a concise Markdown task contract."""
    if environment.env_name.lower() in ("yokai_grid_9_game", "yokai_grid_16_game"):
        from llm_policy.environments.yokai.context import task_dossier
        if runtime_env is None:
            from llm_policy.context.environment import make_resolved_env
            runtime_env = make_resolved_env(environment)
        return task_dossier(environment, objective, runtime_env, policy_guidance, guidance_mode, meta_update_id)
    if environment.env_name.lower() == "hanabi":
        from llm_policy.environments.hanabi.context import task_dossier
        if runtime_env is None:
            from llm_policy.context.environment import make_resolved_env
            runtime_env = make_resolved_env(environment)
        return task_dossier(environment, objective, runtime_env, policy_guidance, guidance_mode, meta_update_id)
    config_text = environment.config_path.read_text(encoding="utf-8")
    capabilities = _compact_capabilities()
    throughput_record, throughput_markdown = _delivery_throughput_heuristic(
        environment,
        runtime_env,
    )
    runtime_contract = (
        resolved_runtime_contract(runtime_env) if runtime_env is not None else None
    )
    record = {
        "schema_version": 1,
        "objective": objective.to_record(),
        "environment": {
            "config_name": environment.config_path.name,
            "config_sha256": hashlib.sha256(
                config_text.encode("utf-8")
            ).hexdigest(),
            "name": environment.config.get("name", environment.config_path.stem),
            "env_name": environment.env_name,
            "env_kwargs": environment.env_kwargs,
            "score_min": environment.config.get("score_min"),
            "score_max": environment.config.get("score_max"),
        },
        "capabilities": capabilities,
        "runtime_contract": runtime_contract,
        "search_heuristics": {"delivery_throughput": throughput_record},
        "meta_guidance": {
            "mode": guidance_mode,
            "meta_update_id": meta_update_id,
            "sha256": hashlib.sha256(policy_guidance.encode("utf-8")).hexdigest(),
            "present": bool(policy_guidance.strip()),
        },
    }
    boundary = capabilities["information_boundary"]
    available = "\n".join(f"- {value}" for value in boundary["available"])
    unavailable = "\n".join(f"- {value}" for value in boundary["unavailable"])
    module_functions = "\n".join(
        f"- `{value}`" for value in capabilities["module_contract"]["functions"]
    )
    facts = ", ".join(capabilities["facts"])
    queries = "\n".join(f"- `{value}`" for value in capabilities["queries"])
    skills = ", ".join(f"`{value}`" for value in capabilities["skills"])
    briefing = build_environment_briefing(
        environment,
        include_source_instructions=False,
    ).strip()
    runtime_contract_markdown = (
        json.dumps(runtime_contract, indent=2, sort_keys=True)
        if runtime_contract is not None
        else "Unavailable: the environment was not instantiated."
    )
    markdown = f"""# Compact policy-search task dossier

This file is generated deterministically from the selected objective profile,
resolved environment configuration, layout registry, and stable policy
capability manifest. Read it before inspecting implementation source.

You are jointly designing both decentralized policies. During design, you may
use the complete environment specification—including the layout, dynamics,
reward function, observation semantics, action semantics, and known task
distribution—to construct complementary roles, conventions, and a joint plan.

At execution time, each policy remains strictly decentralized: it may use only
its own local observation, reset flag, private recurrent state, and PRNG key. It
cannot access hidden environment state, the other agent's observation or
memory, or episode-specific information that was not locally observed.
Therefore, every joint strategy must be implementable through predetermined roles, local observations,
and observable actions or events.

## Objective: {objective.name}

Metric: `{objective.metric}`  
Evaluator: `{objective.evaluator}`

{objective.instruction}

Selection rule: {objective.selection_rule}

## Meta-policy guidance

The immutable file `context/policy_guidance.md` contains additional
environment-agnostic guidance derived from analysis of XP performance across
independently generated policies. Use this guidance throughout policy design,
testing, and candidate selection, and adhere to it unless it conflicts with the
objective, decentralization boundary, or policy API.

Meta update: `{meta_update_id or "unspecified"}`
Guidance SHA-256: `{record["meta_guidance"]["sha256"]}`

## Exact policy dimensions

These resolved values define the current environment and the minimum controller
map size. `num_ingredients` must match exactly. `max_height` and `max_width`
must be at least the resolved values. The generated starter policy already uses
the exact values; do not retain dimensions from another environment.

```json
{runtime_contract_markdown}
```

## Delivery-throughput sanity check

{throughput_markdown}

## Resolved environment

{briefing}

## Runtime information boundary

Available independently to the acting role:

{available}

Unavailable to runtime policy code:

{unavailable}

The two roles never share private memory. Any coordination must pass through
environment actions and locally observed environment history.

## Required policy interface

Constants: `POLICY_API_VERSION`, `POLICY_METADATA`

{module_functions}

Each act function returns
`(scalar integer action, next_memory, PolicyTrace)` and must use fixed-shape
JAX-compatible control flow. The next memory must have the same structure,
shapes, and dtypes as the input memory.

## Compact primitive index

Observable facts include: {facts}.

Target queries:

{queries}

Available skills: {skills}.

## Source escalation rule

Treat this dossier and `context/environment/config.yaml` as the primary task
specification. `context/environment/layout.md` contains the selected layout,
and `context/environment/source/` contains the complete policy-relevant
environment implementation. The other exposed
implementation references are `context/policy_api.md`,
`context/overcooked_primitives.py`, and `context/rollout_api.py`. Inspect only
the particular source needed to resolve a concrete ambiguity. If this dossier
conflicts with the documented API or environment source, record the discrepancy
instead of probing files outside the task bundle.
"""
    from llm_policy.context.guidance import validate_guidance_mode

    guided = validate_guidance_mode(guidance_mode, policy_guidance, meta_update_id)
    start = markdown.index("## Meta-policy guidance")
    end = markdown.index("## Exact policy dimensions", start)
    if not guided:
        section = (
            "## Meta-learning baseline\n\n"
            "This is the generation-0 baseline. No policy guidance is supplied.\n\n"
            if guidance_mode == "meta_baseline" else ""
        )
        markdown = markdown[:start] + section + markdown[end:]
    elif guidance_mode == "standalone_guided":
        section = (
            "## Policy guidance\n\n"
            "Use the supplied immutable `context/policy_guidance.md` during design, "
            "testing, and selection, subordinate to the objective, decentralization "
            "boundary, and policy API. This is a standalone guided run.\n\n"
        )
        markdown = markdown[:start] + section + markdown[end:]
    return record, markdown


def write_task_dossier(
    environment: ResolvedEnvironment,
    objective: PolicyObjective,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
    runtime_env=None,
    policy_guidance: str = "",
    guidance_mode: str = "standalone_no_guidance",
    meta_update_id: str | None = None,
):
    """Write paired human- and machine-readable dossier artifacts."""
    record, markdown = build_task_dossier(
        environment,
        objective,
        runtime_env=runtime_env,
        policy_guidance=policy_guidance,
        guidance_mode=guidance_mode,
        meta_update_id=meta_update_id,
    )
    markdown_path = Path(markdown_path)
    markdown_path.write_text(markdown, encoding="utf-8")
    if json_path is not None:
        Path(json_path).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return record


__all__ = ["build_task_dossier", "write_task_dossier"]
