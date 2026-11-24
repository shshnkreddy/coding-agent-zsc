"""Build a minimal immutable context and one editable agent workspace."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
from functools import partial
from pathlib import Path
from llm_policy.resources import asset_path, source_path

from llm_policy.context.environment import (
    load_environment_config,
    make_resolved_env,
    resolved_runtime_contract,
    write_environment_source_bundle,
)
from llm_policy.search.ledger import initialize_ledger
from llm_policy.context.objectives import load_policy_objective
from llm_policy.context.modes import ContextProfile
from llm_policy.context.task_dossier import write_task_dossier


TEMPLATE_DIR = asset_path("templates")
BUNDLED_ENV_CONFIG_RELATIVE = Path("context/environment/config.yaml")
TASK_DOSSIER_RELATIVE = Path("context/task_spec.md")
POLICY_GUIDANCE_RELATIVE = Path("context/policy_guidance.md")
CONTEXT_MANIFEST_VERSION = 1


def _render_task_dimensions(source: str, runtime_contract: dict):
    if runtime_contract.get("adapter") == "hanabi":
        if runtime_contract["num_agents"] == 2:
            return source
        from llm_policy.environments.hanabi.context import provided_starter, provided_api
        if "POLICY_API_VERSION = 1" in source:
            return provided_starter(runtime_contract)
        if source.startswith("# Hanabi policy API"):
            return provided_api(runtime_contract)
        return source
    if runtime_contract.get("adapter") == "yokai":
        return source  # Yokai dimensions are validated by the adapter.
    replacements = {
        "TASK_MAX_HEIGHT = 3": (
            f"TASK_MAX_HEIGHT = {runtime_contract['max_height']}"
        ),
        "TASK_MAX_WIDTH = 5": (
            f"TASK_MAX_WIDTH = {runtime_contract['max_width']}"
        ),
        "TASK_NUM_INGREDIENTS = 2": (
            f"TASK_NUM_INGREDIENTS = {runtime_contract['num_ingredients']}"
        ),
    }
    for original, rendered in replacements.items():
        if original not in source:
            raise RuntimeError(f"task template is missing marker: {original}")
        source = source.replace(original, rendered)
    return source


def bundled_env_config_path(task_root: str | Path):
    """Return the canonical environment config path in a task bundle."""
    return Path(task_root) / BUNDLED_ENV_CONFIG_RELATIVE


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_read_only(path: Path):
    for directory, directories, files in os.walk(path):
        directory_path = Path(directory)
        for name in files:
            (directory_path / name).chmod(0o444)
        for name in directories:
            (directory_path / name).chmod(0o555)
    path.chmod(0o555)


def _integrity_files(task_root: Path):
    files = [task_root / "AGENTS.md"]
    files.extend(
        path for path in (task_root / "context").rglob("*") if path.is_file()
    )
    return sorted(files)


def _context_manifest(task_root: Path):
    files = {
        str(path.relative_to(task_root)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in _integrity_files(task_root)
    }
    try:
        jaxmarl_version = importlib.metadata.version("jaxmarl")
    except importlib.metadata.PackageNotFoundError:
        jaxmarl_version = "unknown"
    return {
        "schema_version": CONTEXT_MANIFEST_VERSION,
        "jaxmarl_version": jaxmarl_version,
        "files": files,
    }


def verify_policy_workspace(task_root: str | Path, manifest: dict):
    """Fail closed if any model-immutable task input changed or appeared."""
    task_root = Path(task_root).resolve()
    expected = manifest.get("files", {})
    actual_paths = {
        str(path.relative_to(task_root)): path
        for path in _integrity_files(task_root)
    }
    if set(actual_paths) != set(expected):
        missing = sorted(set(expected) - set(actual_paths))
        unexpected = sorted(set(actual_paths) - set(expected))
        raise RuntimeError(
            "immutable task file set changed; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, path in actual_paths.items():
        record = expected[relative]
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record[
            "sha256"
        ]:
            raise RuntimeError(f"immutable task file changed: {relative}")
    return True


def build_policy_workspace(
    task_root: str | Path,
    *,
    env_config: str | Path,
    repo_root: str | Path,
    objective: str = "XP",
    objective_config: str | Path | None = None,
    policy_guidance_path: str | Path | None = None,
    guidance_mode: str = "standalone_no_guidance",
    meta_update_id: str | None = None,
    lifetime_awareness: bool = False,
    timeout_seconds: int | None = None,
    inherited_baseline: bool = False,
    environment_context: str = "specification",
    primitive_mode: str = "none",
    paraphrase_config: str | Path | None = None,
    generation_mode: str = "iterative",
):
    """Create a task containing only documented model-relevant inputs."""
    if not isinstance(lifetime_awareness, bool):
        raise TypeError("lifetime_awareness must be a boolean")
    if not isinstance(inherited_baseline, bool):
        raise TypeError("inherited_baseline must be a boolean")
    if lifetime_awareness and (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise ValueError(
            "enabled lifetime awareness requires a positive integer timeout_seconds"
        )
    task_root = Path(task_root).resolve()
    env_config = Path(env_config).resolve()
    repo_root = Path(repo_root).resolve()
    if task_root.exists() and any(task_root.iterdir()):
        raise FileExistsError(f"task root is not empty: {task_root}")
    task_root.mkdir(parents=True, exist_ok=True)
    context_dir = task_root / "context"
    environment_dir = context_dir / "environment"
    workspace_dir = task_root / "workspace"
    bin_dir = task_root / "bin"
    environment_dir.mkdir(parents=True)
    workspace_dir.mkdir()
    bin_dir.mkdir()

    shutil.copy2(env_config, bundled_env_config_path(task_root))
    resolved_environment = load_environment_config(
        bundled_env_config_path(task_root)
    )
    profile = ContextProfile(environment_context, primitive_mode)
    profile.validate_environment(resolved_environment)
    environment_sources = {}
    if environment_context == "source":
        environment_sources = write_environment_source_bundle(
            resolved_environment, repo_root=repo_root, environment_dir=environment_dir,
        )
        if primitive_mode == "none":
            for path in (environment_dir / "source").glob("*.py"):
                if "primitives" in path.name or path.name == "repository_observation.py":
                    path.unlink()
    runtime_env = make_resolved_env(resolved_environment)
    runtime_contract = resolved_runtime_contract(runtime_env)
    if (runtime_contract.get("adapter", "overcooked_v2") == "overcooked_v2"
            and int(runtime_contract.get("num_agents", 2)) == 3):
        if (environment_context, primitive_mode) != ("specification", "none"):
            raise ValueError(
                "three-player OCV2 v1 requires environment_context=specification "
                "and primitive_mode=none"
            )
        if guidance_mode == "meta_baseline":
            raise ValueError("three-player OCV2 v1 does not support meta search")
    if profile.restricted:
        from llm_policy.context.specification import write_specification
        specification = write_specification(resolved_environment, runtime_env, environment_dir, runtime_contract)
        environment_sources = {**environment_sources, **specification}
        if environment_context == "source":
            environment_sources["source_files"] = sorted(path.name for path in (environment_dir / "source").glob("*"))
            source_manifest = environment_dir / "source_manifest.json"
            if source_manifest.exists():
                hashes = json.loads(source_manifest.read_text())
                source_manifest.write_text(json.dumps({name: digest for name, digest in hashes.items()
                    if (environment_dir / "source" / name).is_file()}, indent=2) + "\n")
        (context_dir / "context_profile.json").write_text(json.dumps(profile.record(), sort_keys=True) + "\n")
    adapter = runtime_contract.get("adapter", "overcooked_v2")
    is_card_game = adapter in ("hanabi", "yokai")
    task_templates = TEMPLATE_DIR / adapter if is_card_game else TEMPLATE_DIR
    primitive_name = f"{adapter}_primitives.py" if is_card_game else "overcooked_primitives.py"
    if adapter == "yokai" and runtime_contract["deck_size"] == 16:
        task_templates = TEMPLATE_DIR / "yokai16"
        primitive_name = "yokai16_primitives.py"
    resolved_objective = load_policy_objective(
        objective,
        repo_root=repo_root,
        config_path=objective_config,
    ).for_policy_structure(runtime_contract.get("policy_structure", "separate_roles"))
    convention_profile = "none"
    if adapter == "hanabi":
        from llm_policy.environments.hanabi.context import CONVENTION_ROOT, convention_objective, resolve_convention_profile
        convention_profile = resolve_convention_profile(resolved_environment.config)
        resolved_objective = convention_objective(resolved_objective, convention_profile, runtime_contract["num_agents"])
        if convention_profile != "none":
            shutil.copytree(CONVENTION_ROOT / convention_profile,
                            context_dir / "conventions" / convention_profile)
            if runtime_contract["num_agents"] > 2:
                from llm_policy.environments.hanabi.context import adapt_convention_bundle
                adapt_convention_bundle(context_dir / "conventions" / convention_profile, runtime_contract)
    if paraphrase_config is not None:
        from llm_policy.context.paraphrasing import apply

        if objective_config is not None or guidance_mode != "standalone_no_guidance":
            raise ValueError("paraphrasing requires canonical objectives and no guidance")
        resolved_objective = apply(task_root, resolved_objective, runtime_contract,
                                   convention_profile, profile, paraphrase_config)
    resolved_objective = resolved_objective.for_runtime_contract(runtime_contract)
    policy_guidance = ""
    if policy_guidance_path is not None:
        policy_guidance_path = Path(policy_guidance_path).resolve()
        if not policy_guidance_path.is_file() or policy_guidance_path.is_symlink():
            raise FileNotFoundError(
                f"policy guidance must be a regular file: {policy_guidance_path}"
            )
        policy_guidance = policy_guidance_path.read_text(encoding="utf-8")
    from llm_policy.context.guidance import validate_guidance_mode

    guided = validate_guidance_mode(guidance_mode, policy_guidance, meta_update_id)
    guidance_destination = task_root / POLICY_GUIDANCE_RELATIVE
    if guided:
        guidance_destination.write_text(policy_guidance, encoding="utf-8")
    if not profile.restricted:
        task_record = write_task_dossier(
            resolved_environment,
            resolved_objective,
            markdown_path=task_root / TASK_DOSSIER_RELATIVE,
            runtime_env=runtime_env,
            policy_guidance=policy_guidance,
            guidance_mode=guidance_mode,
            meta_update_id=meta_update_id,
        )
    if profile.restricted:
        from llm_policy.context.minimal import dossier
        task_record, task_text = dossier(resolved_environment, resolved_objective, runtime_contract,
                                        profile, policy_guidance, guidance_mode, meta_update_id,
                                        runtime_env=runtime_env)
        (task_root / TASK_DOSSIER_RELATIVE).write_text(task_text, encoding="utf-8")
    if is_card_game:
        if adapter == "yokai":
            from llm_policy.environments.yokai.context import policy_interface
        else:
            from llm_policy.environments.hanabi.context import policy_interface
            policy_interface = partial(policy_interface, convention_profile=convention_profile)
    starting_policy_path = context_dir / "starting_policy.py"
    shared_local = runtime_contract.get("policy_structure") == "shared_local"
    starter_name = "policy_shared.py" if shared_local else "policy.py"
    if primitive_mode == "none":
        from llm_policy.context.minimal import starter, policy_api
        starting_policy_path.write_text(starter(runtime_contract), encoding="utf-8")
        (context_dir / "policy_api.md").write_text(policy_api(runtime_contract), encoding="utf-8")
    else:
        starting_policy_path.write_text(
            _render_task_dimensions(
                (task_templates / starter_name).read_text(encoding="utf-8"), runtime_contract,
            ), encoding="utf-8",
        )
        api_text = (task_templates / "policy_api.md").read_text(encoding="utf-8")
        if is_card_game:
            api_text = api_text.replace("{{POLICY_STRUCTURE_INTERFACE}}", policy_interface(runtime_contract))
        if adapter == "hanabi" and runtime_contract["num_agents"] > 2:
            from llm_policy.environments.hanabi.context import provided_api
            api_text = provided_api(runtime_contract, convention_profile)
        else:
            api_text = _render_task_dimensions(api_text, runtime_contract)
        (context_dir / "policy_api.md").write_text(api_text, encoding="utf-8")
    for source, destination in (
        (
            asset_path(f"src/llm_policy/{primitive_name}"),
            context_dir / primitive_name,
        ),
        (source_path("llm_policy.evaluation.api"), context_dir / "rollout_api.py"),
    ):
        if primitive_mode == "none" and destination.name == primitive_name:
            continue
        if profile.restricted and destination.name == "rollout_api.py":
            source = source_path("llm_policy.execution.public_client")
        if not source.is_file():
            raise FileNotFoundError(f"required task input missing: {source}")
        if destination.name == "rollout_api.py":
            from llm_policy.execution.isolation import public_source, rollout_api_source

            text = source.read_text(encoding="utf-8")
            if profile.restricted:
                text = public_source(text)
            destination.write_text(
                rollout_api_source(text, evaluator=resolved_objective.evaluator), encoding="utf-8"
            )
        elif destination.name == primitive_name or profile.restricted:
            from llm_policy.execution.isolation import public_source

            destination.write_text(public_source(source.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(source, destination)

    agents_text = (TEMPLATE_DIR / "AGENTS.md").read_text(encoding="utf-8")
    if profile.restricted:
        agents_text = agents_text.replace("exact configuration, selected layout, and complete\n  policy-relevant environment source",
                                          "exact configuration, factual rules, observation and action specification")
        if environment_context == "source":
            agents_text += "\ncontext/environment/source contains the permitted environment source reference.\n"
        if primitive_mode == "none":
            agents_text = agents_text.replace("- `context/overcooked_primitives.py`: public primitive implementation\n", "")
            agents_text = agents_text.replace("`PolicyTrace`", "a policy-defined trace dictionary (or `{}`)")
            agents_text += "\nNo pre-existing primitive libraries are available. Implement your own helpers if needed.\n"
    if is_card_game:
        agents_text = agents_text.replace(
            "Keep `POLICY_API_VERSION`, all required `POLICY_METADATA` keys, both\n"
            "initializers, and both act functions in `policy.py`.",
            "Keep `POLICY_API_VERSION` and all required `POLICY_METADATA` keys in `policy.py`.\n"
            + (("Implement init_agent() and agent_act(obs, reset, memory, rng). "
                f"All {runtime_contract['num_agents']} seats run this code with separate local observations, memory and PRNG keys; "
                "all process every transition and only the active seat acts.")
               if primitive_mode == "none" and runtime_contract["policy_structure"] == "shared_local"
               else policy_interface(runtime_contract)))
        agents_text = agents_text.replace("overcooked_primitives.py", primitive_name)
        agents_text = agents_text.replace("exact configuration, selected layout, and complete", "exact configuration, rules, and complete")
        agents_text = agents_text.replace(
            "dimensions are rendered from the instantiated environment: ingredient count\n"
            "must remain exact, while map dimensions must not be reduced below those values.",
            f"dimensions describe the configured {runtime_contract['num_agents']}-player {adapter} game and must remain exact.\n"
            "The observation also includes this player's native legal-action mask.")
    if adapter == "cat_dog":
        agents_text = agents_text.replace(
            "dimensions are rendered from the instantiated environment: ingredient count\n"
            "must remain exact, while map dimensions must not be reduced below those values.",
            "dimensions describe the two-player Cat & Dog game and must remain exact.\n"
            "Agent 0 is Alice and agent 1 is Bob. Both process each transition with private\n"
            "observations and memory; only the active role's action affects the game.")
    if primitive_mode == "none":
        agents_text = agents_text.replace("PolicyTrace", "a policy-defined trace dictionary (or `{}`)")
    if adapter == "overcooked_v2" and int(runtime_contract.get("num_agents", 2)) == 3:
        agents_text = agents_text.replace(
            "both\ninitializers, and both act functions in `policy.py`.",
            "all three\ninitializers, and all three act functions in `policy.py`.",
        )
        agents_text = agents_text.replace(
            'policy-ledger conclude --agent-0-role "..." --agent-1-role "..." \\\n',
            'policy-ledger conclude --agent-0-role "..." --agent-1-role "..." \\\n'
            '  --agent-2-role "..." \\\n',
        )
    agents_text = agents_text.replace(
        "{{FINAL_POLICY_SELECTION}}", resolved_objective.final_selection_instruction
    )
    agents_text = agents_text.replace("{{LEDGER_SELECTION_NOTE}}", resolved_objective.ledger_selection_note)
    agents_text = agents_text.replace(
        "{{LEDGER_FINALIZATION_INSTRUCTION}}", resolved_objective.ledger_finalization_instruction
    )
    if resolved_objective.evaluator == "self_play":
        agents_text = agents_text.replace("\npolicy-ledger restore-best\n", "\n")
    if guidance_mode == "standalone_guided":
        agents_text = agents_text.replace(
            "immutable environment-agnostic meta guidance",
            "immutable environment-agnostic policy guidance",
        )
    if not guided:
        agents_text = agents_text.replace(
            "- `context/policy_guidance.md`: immutable environment-agnostic meta guidance\n", ""
        )
        start = agents_text.index("Read `context/policy_guidance.md`")
        end = agents_text.index("Keep `POLICY_API_VERSION`", start)
        agents_text = agents_text[:start] + agents_text[end:]
        start = agents_text.index(' \\\n  --guidance-alignment')
        end = agents_text.index('\n', agents_text.index('--guidance-rationale', start))
        agents_text = agents_text[:start] + agents_text[end:]
    if convention_profile != "none":
        from llm_policy.environments.hanabi.context import CONVENTION_READING
        agents_text += "\n\n## Shared Hanabi conventions\n\n" + CONVENTION_READING + "\n"
    if inherited_baseline:
        agents_text = agents_text.replace(
            "A00 is open for the starting template.",
            (
                "A00 contains the runner-scored inherited selected-policy "
                "baseline and is already concluded. Start at A01."
            ),
        )
    if lifetime_awareness:
        agents_text += f"""

## Agent-search lifetime

Your runner-enforced policy-search lifetime is `{timeout_seconds}` seconds.
Run `policy-lifetime` to obtain the live elapsed and remaining time. Every
`policy-eval` result also includes an `agent_lifetime` snapshot. Check the
lifetime before opening another material experiment and before choosing the
final policy. This clock covers your agent-search process only; runner-owned
final candidate evaluation happens afterward.
"""
    (task_root / "AGENTS.md").write_text(
        # Instructions use workspace-relative paths. Replacing '/task' here
        # also corrupts the filename in 'context/task_spec.md'.
        agents_text, encoding="utf-8"
    )
    for name in (
        "custom_primitives.py",
        "rollouts.py",
        "rollout_utils.py",
        "strategy.md",
    ):
        if name in ("custom_primitives.py", "strategy.md") and primitive_mode == "none":
            continue
        source = task_templates / name if (task_templates / name).is_file() else TEMPLATE_DIR / name
        shutil.copy2(source, workspace_dir / name)
    strategy_path = workspace_dir / "strategy.md"
    if strategy_path.is_file():
        coordination_review = resolved_objective.coordination_review_instruction
        if convention_profile != "none":
            from llm_policy.environments.hanabi.context import convention_partner_wording
            coordination_review = convention_partner_wording(coordination_review)
        coordination_review = coordination_review[:1].lower() + coordination_review[1:]
        strategy_path.write_text(
            strategy_path.read_text(encoding="utf-8").replace(
                "{{COORDINATION_REVIEW}}", coordination_review
            ),
            encoding="utf-8",
        )
    shutil.copy2(starting_policy_path, workspace_dir / "policy.py")
    if generation_mode == "open_source_zsc":
        if adapter != "cat_dog" or objective != "XP" or environment_context != "specification" or primitive_mode != "none":
            raise ValueError("open_source_zsc requires Cat & Dog XP, specification context and no primitives")
        from llm_policy.context.source_aware import install
        starting_policy_path = install(task_root, runtime_contract)
    ledger_path = workspace_dir / "policy_ledger.jsonl"
    ledger_markdown_path = workspace_dir / "experiment_ledger.md"
    if not inherited_baseline:
        initialize_ledger(
            ledger_path,
            ledger_markdown_path,
            run_id=task_root.parent.name,
            inherited_baseline=inherited_baseline,
        )

    (workspace_dir / "AGENTS.md").symlink_to(task_root / "AGENTS.md")
    (workspace_dir / "context").symlink_to(context_dir, target_is_directory=True)
    command_names = ["policy_eval.py", "policy_ledger.py"]
    if lifetime_awareness:
        command_names.append("policy_lifetime.py")
    for name in command_names:
        destination = bin_dir / name.removesuffix(".py").replace("_", "-")
        if profile.restricted:
            destination.write_text("#!/usr/bin/env python3\nfrom llm_policy.public_client import command_main\ncommand_main()\n")
        else:
            modules = {"policy_eval.py": "llm_policy.evaluation.commands",
                       "policy_ledger.py": "llm_policy.search.ledger_command",
                       "policy_lifetime.py": "llm_policy.search.lifetime_command"}
            destination.write_text(
                f"#!/usr/bin/env python3\nfrom {modules[name]} import main\n"
                "raise SystemExit(main())\n"
            )
        destination.chmod(0o555)

    if profile.restricted:
        from llm_policy.execution.isolation import build_public_sdk
        build_public_sdk(context_dir / "public_sdk", repo_root, profile, adapter, runtime_contract,
                         evaluator=resolved_objective.evaluator)

    from llm_policy.context.provenance import install as install_provenance
    install_provenance(task_root)
    if adapter == "hanabi" and runtime_contract["num_agents"] > 2:
        contract_record = {key: value for key, value in runtime_contract.items()
                           if key not in ("source_sha256", "belief_mode")}
        (context_dir / "runtime_contract.json").write_text(json.dumps(contract_record, sort_keys=True) + "\n")
        agents_path = task_root / "AGENTS.md"
        text = agents_path.read_text()
        flags = " ".join(f'--agent-{i}-role "..."' for i in range(runtime_contract["num_agents"]))
        text += "\n\n## Multiplayer conclusions\n\n"
        if runtime_contract.get("policy_structure") == "shared_local":
            text += (
                "The --agent-i-role fields summarize behavior at each seat; they do not "
                "request separate policies or distinct assigned roles. All seats execute "
                "the same shared policy with private observations, memory, and PRNG keys. "
                "Summarize the actual shared decision rule in each field; repeating the "
                "same summary for every seat is appropriate. Describe seat-specific "
                "differences only if your policy actually implements them.\n\n")
        else:
            text += "Explain the behavior implemented by each seat's policy entry point.\n\n"
        text += f'```bash\npolicy-ledger conclude --attempt-id A01 {flags} --coordination-protocol "..." --decision follow_up\n```\n'
        if convention_profile != "none" and CONVENTION_READING not in text:
            text += "\n## Shared Hanabi conventions\n\n" + CONVENTION_READING + "\n"
        agents_path.write_text(text)
    manifest = _context_manifest(task_root)
    _make_read_only(context_dir)
    (task_root / "AGENTS.md").chmod(0o444)
    bin_dir.chmod(0o555)
    return {
        "task_root": task_root,
        "workspace_dir": workspace_dir,
        "context_dir": context_dir,
        "bin_dir": bin_dir,
        "ledger_path": ledger_path,
        "ledger_markdown_path": ledger_markdown_path,
        "context_manifest": manifest,
        "task_record": task_record,
        "task_dossier_path": task_root / TASK_DOSSIER_RELATIVE,
        "policy_guidance_path": guidance_destination,
        "starting_policy_path": starting_policy_path,
        "objective": resolved_objective,
        "environment_sources": environment_sources,
        "runtime_contract": runtime_contract,
        "context_profile": profile.record(),
    }


__all__ = [
    "BUNDLED_ENV_CONFIG_RELATIVE",
    "POLICY_GUIDANCE_RELATIVE",
    "TASK_DOSSIER_RELATIVE",
    "build_policy_workspace",
    "bundled_env_config_path",
    "verify_policy_workspace",
]
