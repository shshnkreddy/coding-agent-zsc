"""Evaluate saved policy populations or plot saved results from YAML configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from llm_policy.evaluation.reports import summarize_groups, validate_groups


def _load_config(path):
    import yaml

    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("invalid configuration YAML/JSON") from error
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def _path(value, base_dir):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def evaluate(config: Mapping, *, base_dir: Path | None = None):
    """Evaluate saved policies; relative paths are resolved beside the config."""
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    allowed = {
        "env_config", "policies", "root_seed", "num_episodes", "num_steps",
        "common_random_numbers", "output_dir", "groups",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown evaluation settings: {sorted(unknown)}")
    if "env_config" not in config:
        raise ValueError("env_config is required")
    policies = config.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("policies must be a nonempty list of saved policy paths")
    groups = validate_groups(config["groups"], len(policies)) if "groups" in config else None
    common_random_numbers = config.get("common_random_numbers", True)
    if not isinstance(common_random_numbers, bool):
        raise ValueError("common_random_numbers must be a boolean")
    options = {
        name: config[name]
        for name in ("root_seed", "num_episodes", "num_steps")
        if name in config
    }
    for name in ("num_episodes", "num_steps"):
        value = options.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    environment = config["env_config"]
    if not isinstance(environment, Mapping):
        environment = _path(environment, base_dir)
    policy_paths = [_path(value, base_dir) for value in policies]
    output_dir = _path(config.get("output_dir", "evaluation"), base_dir)

    from llm_policy.search.population import evaluate_codex_xp_matrix

    result = evaluate_codex_xp_matrix(
        environment, policy_paths, common_random_numbers=common_random_numbers,
        output_dir=output_dir, **options,
    )
    summary = result.to_summary()
    if groups is not None:
        scores = summarize_groups(summary["returns"], groups)
        summary["group_scores"] = scores
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "group_summary.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        summary.setdefault("artifacts", {})["group_summary"] = path.name
    return summary


def plot(config: Mapping, *, base_dir: Path | None = None):
    """Plot an existing XP summary without running policy evaluation."""
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    unknown = set(config) - {"summary_path", "output_path"}
    if unknown:
        raise ValueError(f"unknown plot settings: {sorted(unknown)}")
    if "summary_path" not in config:
        raise ValueError("summary_path is required")
    summary_path = _path(config["summary_path"], base_dir)
    output_path = (
        _path(config["output_path"], base_dir)
        if config.get("output_path") is not None else None
    )

    from llm_policy.search.population import plot_saved_codex_xp_matrix

    return {"output_path": str(plot_saved_codex_xp_matrix(summary_path, output_path))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "plot"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        path = args.config.resolve()
        config = _load_config(path)
        operation = evaluate if args.command == "evaluate" else plot
        result = operation(config, base_dir=path.parent)
    except (OSError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "command": args.command,
                          "error": f"{type(error).__name__}: {error}"}))
        return 2
    print(json.dumps({"ok": True, "command": args.command, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
