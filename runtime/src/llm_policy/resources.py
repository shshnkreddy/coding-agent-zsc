"""Installed resources shared by previews, workers, and source disclosure."""

from pathlib import Path

try:
    from importlib.resources import files as _resource_files
except ImportError:  # Python 3.8 execution hosts; wheel contents are unpacked.
    _resource_files = None

_MODULES = {
    "context_modes": "context/modes",
    "env_context": "context/environment",
    "environment_specification": "context/specification",
    "card_specification_markdown": "context/card_specification",
    "ocv2_specification_markdown": "context/ocv2_specification",
    "guidance_modes": "context/guidance",
    "minimal_context": "context/minimal",
    "objectives": "context/objectives",
    "task_dossier": "context/task_dossier",
    "workspace": "context/workspace",
    "meta_context": "context/meta",
    "codex_provider": "agents/codex",
    "claude_code": "agents/claude",
    "source_isolation": "execution/isolation",
    "environment_service": "execution/environment_service",
    "public_client": "execution/public_client",
    "public_analysis": "execution/public_analysis",
    "filesystem_transport": "execution/filesystem_transport",
    "ledger": "search/ledger",
    "lifetime": "search/lifetime",
    "meta_ledger": "search/meta_ledger",
    "meta_guidance": "search/meta_guidance",
    "hanabi_adapter": "environments/hanabi/adapter",
    "hanabi_context": "environments/hanabi/context",
    "hanabi_primitives": "environments/hanabi/primitives",
    "yokai_adapter": "environments/yokai/adapter",
    "yokai_context": "environments/yokai/context",
    "yokai_primitives": "environments/yokai/primitives",
    "yokai16_primitives": "environments/yokai/primitives16",
    "overcooked_primitives": "environments/ocv2/primitives",
    "primitives": "policies/primitives",
    "runtime": "policies/runtime",
    "validation": "policies/validation",
    "capabilities": "policies/capabilities",
    "policy_api": "policies/api",
    "types": "policies/types",
    "eval_api": "evaluation/api",
    "rollout": "evaluation/rollout",
    "trajectory": "evaluation/trajectory",
    "observability": "evaluation/observability",
    "codex_search": "search/population",
    "meta_search": "search/meta"
}
_ENVIRONMENTS = {
    "configurable_overcooked_v2": "environment",
    "overcooked_v2_layouts": "layouts",
    "overcooked_observation": "observation"
}


def package_root() -> Path:
    """Return the installed package directory (wheel installations are unpacked)."""
    if _resource_files is None:
        return Path(__file__).resolve().parent
    return Path(str(_resource_files("llm_policy")))


def _remove_prefix(value: str, prefix: str) -> str:
    return value[len(prefix):] if value.startswith(prefix) else value


def _remove_suffix(value: str, suffix: str) -> str:
    return value[:-len(suffix)] if suffix and value.endswith(suffix) else value


def source_path(module: str) -> Path:
    """Locate package source without importing its runtime dependencies."""
    if not module.startswith("llm_policy.") or not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("source module must belong to llm_policy")
    return package_root().joinpath(*module.split(".")[1:]).with_suffix(".py")


def asset_path(relative: str) -> Path:
    """Resolve bundled configuration, templates, or historical source references."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("resource paths must be package-relative")
    value = path.as_posix()
    template_prefix = "src/llm_policy/sandbox_templates"
    if value == template_prefix or value.startswith(template_prefix + "/"):
        result = package_root() / "assets/templates" / value[len(template_prefix):].lstrip("/")
    elif value.startswith("src/llm_policy/"):
        name = _remove_suffix(_remove_prefix(value, "src/llm_policy/"), ".py")
        result = package_root() / (_MODULES.get(name, name) + ".py")
    elif value.startswith("src/envs/"):
        name = _remove_suffix(_remove_prefix(value, "src/envs/"), ".py")
        if name.startswith("yokai/"):
            result = package_root() / "environments/yokai/native" / (name[6:] + ".py")
        else:
            result = package_root() / "environments/ocv2" / (_ENVIRONMENTS[name] + ".py")
    else:
        result = package_root() / "assets" / path
    if not result.exists():
        raise FileNotFoundError(f"missing packaged resource: {relative}")
    return result
