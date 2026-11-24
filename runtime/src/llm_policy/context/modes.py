"""Validated disclosure profiles, independent of environment dynamics."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContextProfile:
    environment_context: str = "specification"
    primitive_mode: str = "none"

    def __post_init__(self):
        if self.environment_context not in ("source", "specification"):
            raise ValueError("environment_context must be source or specification")
        if self.primitive_mode not in ("provided", "none"):
            raise ValueError("primitive_mode must be provided or none")

    @property
    def restricted(self):
        return self.environment_context != "source" or self.primitive_mode != "provided"

    def record(self):
        return asdict(self)

    def validate_environment(self, environment):
        if environment.env_name.lower() == "cat_dog":
            if (self.environment_context, self.primitive_mode) != ("specification", "none"):
                raise ValueError("Cat & Dog v1 requires environment_context=specification and primitive_mode=none")
            return
        supported = {"overcooked_v2", "hanabi", "yokai_grid_9_game", "yokai_grid_16_game"}
        if self.restricted and environment.env_name.lower() not in supported:
            raise ValueError("restricted context modes support fixed OCV2, Hanabi and Yokai only")


def resolve_context_profile(config=None, *, environment_context=None, primitive_mode=None):
    if config is not None:
        if environment_context is not None or primitive_mode is not None:
            raise ValueError("choose --context-config or individual context mode flags, not both")
        value = json.loads(Path(config).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"environment_context", "primitive_mode"}:
            raise ValueError("context preset must contain environment_context and primitive_mode")
        return ContextProfile(**value)
    return ContextProfile(
        "specification" if environment_context is None else environment_context,
        "none" if primitive_mode is None else primitive_mode,
    )


def task_profile(task_root):
    """Read recorded exposure; unprofiled historical tasks used source/provided."""
    path = Path(task_root) / "context/context_profile.json"
    return ContextProfile(**json.loads(path.read_text())) if path.is_file() else ContextProfile("source", "provided")


def require_same_profile(source_task, destination_task):
    if task_profile(source_task) != task_profile(destination_task):
        raise ValueError("cannot resume across disclosure profiles; start a fresh run")
    def generation_mode(task):
        path = Path(task) / "context/generation.json"
        return json.loads(path.read_text())["mode"] if path.is_file() else "iterative"
    if generation_mode(source_task) != generation_mode(destination_task):
        raise ValueError("cannot resume across policy source-exchange contracts")
    from llm_policy.context.paraphrasing import require_same_paraphrase

    require_same_paraphrase(source_task, destination_task)
