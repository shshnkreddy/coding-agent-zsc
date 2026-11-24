"""Load explicit objective configurations for agentic policy search."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from llm_policy.resources import asset_path


@dataclass(frozen=True)
class PolicyObjective:
    """Validated, immutable instructions controlling candidate selection."""

    name: str
    metric: str
    instruction: str
    selection_rule: str
    evaluator: str
    config_path: Path
    config_sha256: str

    @property
    def task_instruction(self):
        return f"{self.instruction}\n\nSelection rule: {self.selection_rule}"

    def for_policy_structure(self, policy_structure: str):
        """Select configured wording without changing scoring or selection rules."""
        if policy_structure not in ("shared_local", "separate_roles"):
            raise ValueError(f"unsupported policy structure: {policy_structure!r}")
        if policy_structure == "shared_local":
            # Rephrase only the standard pairing sentence. Preserve any custom
            # objective text and all remaining search/selection instructions.
            return replace(self, instruction=self.instruction.replace(
                "Maximize expected cooperative episode return when either role of this joint "
                "policy is paired with the complementary role from an independently generated policy.",
                "Maximize expected cooperative episode return when your local policy is paired "
                "with an independently generated partner policy, evaluated in both player positions.",
            ))
        return self

    def for_runtime_contract(self, contract):
        """Render task-specific pairing semantics without changing scoring."""
        n = int(contract.get("num_agents", 2))
        if contract.get("adapter") == "hanabi" and n > 2:
            instruction = self.instruction.replace("Both players", f"All {n} players").replace("both players", f"all {n} players")
            if self.evaluator == "deferred_population_cross_play":
                instruction = instruction.replace(
                    ", evaluated in both player positions", "")
                instruction = instruction.replace(
                    "either role of this joint policy is paired with the complementary role from an independently generated policy",
                    "your policy is paired with independently generated partner policies")
            return replace(self, instruction=instruction,
                           selection_rule=self.selection_rule.replace("both roles use this same policy", f"all {n} seats use this same policy"))
        if (self.evaluator != "deferred_population_cross_play"
                or contract.get("adapter", "overcooked_v2") != "overcooked_v2"
                or int(contract.get("num_agents", 2)) != 3):
            return self
        pairing = (
            "Population XP uses one directed A/BB matrix. In cell [A,B], "
            "policy A supplies agent_0 while independently generated policy B "
            "supplies agent_1 and agent_2. The diagonal is AAA self-play, and "
            "the transpose is B/AA; AA/B is not evaluated."
        )
        selection = (
            "Judge compatibility under that exact A/BB assignment. Development "
            "self-play runs AAA and is diagnostic only."
        )
        instruction = self.instruction.replace(
            "Maximize expected cooperative episode return when either role of this joint "
            "policy is paired with the complementary role from an independently generated "
            "policy.",
            "Maximize expected cooperative episode return in A/BB cross-play: this joint "
            "policy supplies agent_0 while one independently generated joint policy supplies "
            "agent_1 and agent_2.",
        ).replace(
            "both roles use this same policy",
            "all three roles use this same policy",
        )
        if pairing not in instruction:
            instruction = instruction.rstrip() + "\n\n" + pairing
        rule = self.selection_rule
        if selection not in rule:
            rule = rule.rstrip() + "\n\n" + selection
        return replace(self, instruction=instruction, selection_rule=rule)

    @property
    def coordination_review_instruction(self) -> str:
        instruction = (
            "Describe how each role coordinates with its partner, including any "
            "communication or conventions it uses."
        )
        if self.evaluator == "deferred_population_cross_play":
            instruction += (
                " For XP, explain why an independently generated partner could "
                "interpret or accommodate those behaviors."
            )
        return instruction

    @property
    def ledger_selection_note(self) -> str:
        if self.evaluator == "deferred_population_cross_play":
            return "; for XP the highest\ndevelopment self-play score need not be your best choice."
        return "."

    @property
    def ledger_finalization_instruction(self) -> str:
        if self.evaluator == "deferred_population_cross_play":
            return (
                "For XP, after your final declaration run `policy-ledger restore-best` and make\n"
                "no further policy edits. It restores both files exactly (including absence of\n"
                "an optional custom-primitives file), preserves displaced working files in a\n"
                "backup directory reported by the command, and verifies the restored hash.\n"
                "It does not change your declaration. New XP runs fail finalization if no\n"
                "preference is declared or the submitted bundle differs from that declaration.\n"
            )
        return ""

    @property
    def final_selection_instruction(self) -> str:
        """Describe selection ownership for the resolved evaluator."""
        if self.evaluator == "self_play":
            return (
                "For SP, the runner independently re-evaluates candidate policies "
                "after the session and selects the one with the highest development "
                "mean self-play return. Your logged preference does not override "
                "this score-based selection."
            )
        if self.evaluator == "deferred_population_cross_play":
            return (
                "For XP, you select the final policy using the objective-specific "
                "selection rule in `context/task_spec.md`. The runner independently "
                "validates and diagnostically scores your submitted policy bundle, "
                "then finalizes it; it does not replace your choice with the "
                "highest-scoring self-play candidate."
            )
        raise ValueError(f"unsupported objective evaluator: {self.evaluator!r}")

    def to_record(self):
        return {
            "name": self.name,
            "metric": self.metric,
            "instruction": self.instruction,
            "selection_rule": self.selection_rule,
            "evaluator": self.evaluator,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
        }


def load_policy_objective(
    name: str,
    *,
    repo_root: str | Path,
    config_path: str | Path | None = None,
) -> PolicyObjective:
    """Resolve a named ``config/llm_policy/objective/<name>.yaml`` profile."""
    from omegaconf import OmegaConf

    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in name
        )
    ):
        raise ValueError(f"invalid policy objective name: {name!r}")
    path = (
        Path(config_path).resolve()
        if config_path is not None
        else asset_path(f"config/llm_policy/objective/{name}.yaml")
    )
    if not path.is_file():
        available = sorted(
            candidate.stem for candidate in path.parent.glob("*.yaml")
        )
        raise FileNotFoundError(
            f"policy objective {name!r} not found at {path}; available: {available}"
        )
    raw = path.read_text(encoding="utf-8")
    data = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: objective config must contain a mapping")
    expected = {
        "name",
        "metric",
        "instruction",
        "selection_rule",
        "evaluator",
    }
    missing = sorted(expected - set(data))
    extra = sorted(set(data) - expected)
    if missing or extra:
        raise ValueError(
            f"{path}: objective fields mismatch; missing={missing}, extra={extra}"
        )
    values = {field: data[field] for field in expected}
    for field, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: {field} must be a non-empty string")
        values[field] = value.strip()
    if values["name"] != name:
        raise ValueError(
            f"{path}: objective name {values['name']!r} does not match {name!r}"
        )
    supported_evaluators = {"self_play", "deferred_population_cross_play"}
    if values["evaluator"] not in supported_evaluators:
        raise ValueError(
            f"{path}: evaluator must be one of {sorted(supported_evaluators)}"
        )
    return PolicyObjective(
        **values,
        config_path=path,
        config_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


__all__ = ["PolicyObjective", "load_policy_objective"]
