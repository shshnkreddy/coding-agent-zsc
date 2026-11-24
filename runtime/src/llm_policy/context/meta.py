"""Build immutable, non-heuristic evidence bundles for outer meta agents."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

CONTEXT_SCHEMA_VERSION = 3
MODES = frozenset({"propose", "critic"})


def _outer_update_dir(meta_root, update_id):
    generation = int(str(update_id)[1:]) + 1
    return Path(meta_root) / f"generation-{generation:03d}" / "outer" / update_id


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _within(path, root, label):
    path = Path(path).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the meta-run directory: {path}") from error
    return path


def _copy_file(source, destination, records, meta_root):
    source = _within(source, meta_root, "context source")
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"context source must be a regular file: {source}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    records.append(
        {
            "path": destination.as_posix(),
            "source": str(source),
            "sha256": _sha256(destination),
            "size_bytes": destination.stat().st_size,
        }
    )


def _copy_optional(source, destination, records, meta_root):
    source = Path(source)
    if source.is_file() and not source.is_symlink():
        _copy_file(source, destination, records, meta_root)


def _copy_tree(source, destination, records, meta_root, *, exclude_parts=()):
    """Copy every regular file, skipping symlinks and explicitly excluded trees."""
    source = _within(source, meta_root, "context tree")
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"context source must be a directory: {source}")
    excluded = set(exclude_parts)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in excluded for part in relative.parts):
            continue
        if path.is_symlink():
            continue
        if path.is_file():
            _copy_file(path, Path(destination) / relative, records, meta_root)


def _write_text(path, value, records, source):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    records.append(
        {
            "path": path.as_posix(),
            "source": source,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    )


def _critic_domain_summary(env_name="overcooked_v2"):
    if env_name in ("yokai_grid_9_game", "yokai_grid_16_game", "hanabi"):
        concepts = ("face-down cards, colors, hints, revealing, moving, locking, clusters, "
                    "early ending, positions, directions, private knowledge and memory"
                    if env_name in ("yokai_grid_9_game", "yokai_grid_16_game") else
                    "cards, hands, colors, ranks, hints, discarding, playing, tokens, lives and fireworks")
        return f"""# Compact task-domain summary

The evaluated domain is a two-player decentralized cooperative card game.
Generic domain concepts include {concepts}.

Public guidance must not select a concrete meaning, mapping, ordering, role
convention, spatial convention, task constant, or observation-to-action rule
involving these concepts. This is leakage vocabulary, not a task solution.
"""
    return """# Compact task-domain summary

The evaluated domain is a two-player decentralized cooperative cooking task.
Generic domain concepts include movement, interaction, ingredients, recipes,
plates, pots, cooking, delivery locations, buttons or indicators, agent roles,
inventories, layouts, positions, and directions.

Public guidance must not select a concrete meaning, mapping, ordering, role
convention, spatial convention, task constant, or observation-to-action rule
involving these concepts. This is leakage vocabulary, not a task solution.
"""


def _overall_experiment_objective():
    return """## Overall experiment objective

The overall task is to generate robust decentralized policies that maximize
cross-play (XP) return with independently generated policies. Each population
is created by separate policy-generation runs that receive the same public
guidance but otherwise search independently. For a directed cross-play matrix,
cell `(i, j)` pairs policy `i`'s first player with policy `j`'s second player.
The primary outer score is the arithmetic mean of every directed off-diagonal
return. Diagonal self-play measures basic task competence but is not the outer
optimization objective.

The outer optimizer changes one shared artifact across generations:
`policy_guidance.md`. It should improve how fresh policy generators reason,
design, test, and select broadly compatible policies while remaining reusable
across environments. It must not prescribe a concrete task-specific convention.

After a proposed guidance version passes its safety gate, one fresh independent
population is generated with it and the complete directed cross-play matrix is
evaluated. The evaluated guidance becomes the latest iterate regardless of
whether its score rises or falls. The best observed guidance is tracked
separately for reporting and comparison; strict score improvement updates it,
and score ties retain the earliest best generation.
"""


def _task_spec(run_id, update_id, revision, mode, state):
    if mode == "critic":
        return f"""# Meta-policy guidance audit

Run: `{run_id}`
Iteration: `{update_id}`
Revision: `{revision}`
Mode: `critic`

{_overall_experiment_objective()}

## Your specific role: convention-neutrality critic

Act as an independent safety gate for the public guidance, not as a second
proposer or an XP judge. Audit only `proposal/policy_guidance.md`, using
`proposal/linter_result.json` as trusted deterministic evidence and
`domain_summary.md` as leakage vocabulary. Decide whether the public text
remains environment-agnostic or directly or indirectly installs an arbitrary
convention.

Reject any environment-specific instruction, concrete action or object,
direction, role assignment, identity, ordering, task constant,
observation-to-action mapping, canonicalization rule, or private codebook.
Generic guidance about competence, robustness, evidence, uncertainty,
adaptation, testing, and fallback behavior is allowed.

Do not assess the proposer's private diagnosis, predict whether XP will
improve, choose an optimization strategy, or redesign otherwise neutral
guidance. Your deliberately narrow evidence bundle exists so the verdict
depends only on the candidate public guidance and the neutrality contract.
Fail closed when a concrete convention is present or encoded indirectly.

Write only the exact verdict schema requested in `AGENTS.md`.
"""

    revision_rule = (
        "Create a structured diagnosis, a compact proposal, and public guidance."
        if int(revision) == 0
        else "Revise only `policy_guidance.md` in response to earlier gate feedback. "
        "The copied `meta_analysis.json` and `proposal.json` are immutable in "
        "meaning and the runner will reject changes to them."
    )
    return f"""# Continuous meta-policy optimization task

Run: `{run_id}`
Iteration: `{update_id}`
Revision: `{revision}`
Mode: `propose`

{_overall_experiment_objective()}

## Your specific role: meta-guidance proposer

Act as the outer optimizer of policy-search guidance, not as the author of a
task policy. Use the complete chronological record to determine which past
search strategies produced broadly compatible policies, exactly where and how
other strategies failed, and which reusable change to the generators' design,
testing, or selection process is most likely to improve the next population.

Form a structured diagnosis before proposing the intervention. Connect claims
to raw XP cells, policy implementations, candidate histories, evaluations, and
prior outer artifacts. State what should be preserved, distinguish competing
failure explanations, and give a falsifier for the proposed mechanism.

Record environment-specific evidence and reasoning only in
`meta_analysis.json`. Distill the resulting intervention into a concise public
`policy_guidance.md` that helps independent generators discover compatible
policies without embedding the private task evidence or a concrete convention.

## Current optimization state

- Latest generation: `{int(state['latest_generation'])}`
- Latest mean XP: `{float(state['latest_mean_xp']):.10f}`
- Best generation: `{int(state['best_generation'])}`
- Best observed mean XP: `{float(state['best_mean_xp']):.10f}`

Use the full chronological record. XP matrices show measured compatibility;
policy source, search history, prior analyses, and recorded evaluations supply
possible explanations. Because each generation uses fresh inner seeds and
there is no simultaneous control population, describe cross-generation effects
as associations and do not infer a causal mechanism from score deltas alone.

## Evidence boundary

- `meta/`: manifest, latest/best state, complete ledger, and artifact layout.
- `environment/`: the exact environment configuration, layout, and source.
- `inner_contract/`: the policy contract and documented mechanics from an inner
  run.
- `history/generation-NNN/xp/`: the raw saved XP artifacts for every completed
  population. The runner does not rank cells or precompute failure diagnoses.
- `history/generation-NNN/population/seed-X/`: every regular artifact from each
  completed inner policy run, including final and archived candidate policies,
  policy-search ledgers, strategies, evaluations, and transcripts when present.
- `history/generation-NNN/outer/`: prior proposal, analysis, guidance, gate,
  transcript, authored files, and accounting artifacts.
- `current_update/`: earlier failed safety revisions for this iteration, when
  present, including exact linter and critic feedback.

## Revision responsibility

{revision_rule}
"""


def _copy_outer_artifacts(source, destination, records, meta_root):
    if not Path(source).is_dir():
        return
    # Context directories recursively contain earlier evidence bundles and are
    # excluded only to prevent unbounded duplication. All authored and gate
    # artifacts, including other agent-authored outputs, are retained.
    _copy_tree(
        source,
        destination,
        records,
        meta_root,
        exclude_parts={"context"},
    )


def _copy_completed_history(meta_root, state, destination, records):
    history = state.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("proposer context requires non-empty schema-v2 history")
    seen_generations = set()
    for entry in sorted(history, key=lambda item: int(item["generation"])):
        generation = int(entry["generation"])
        if generation in seen_generations:
            raise ValueError(f"duplicate generation in meta history: {generation}")
        seen_generations.add(generation)
        generation_dir = destination / f"generation-{generation:03d}"
        _copy_file(
            entry["guidance_path"],
            generation_dir / "guidance/policy_guidance.md",
            records,
            meta_root,
        )
        xp_summary = _within(entry["xp_summary_path"], meta_root, "XP summary")
        _copy_tree(
            xp_summary.parent,
            generation_dir / "xp",
            records,
            meta_root,
        )
        inner_runs = entry.get("inner_runs")
        if not isinstance(inner_runs, list) or not inner_runs:
            raise ValueError(f"generation {generation} has no inner runs")
        for index, run in enumerate(inner_runs):
            run_dir = _within(run["run_dir"], meta_root, "inner run")
            from llm_policy.context.modes import task_profile
            excluded = ("environment_service.log", "environment_service_private") if task_profile(run_dir / "task").restricted else ()
            _copy_tree(
                run_dir,
                generation_dir / "population" / f"seed-{index}",
                records,
                meta_root,
                exclude_parts=excluded,
            )
        update_id = entry.get("update_id")
        if update_id:
            _copy_outer_artifacts(
                _outer_update_dir(meta_root, update_id),
                generation_dir / "outer" / update_id,
                records,
                meta_root,
            )
        _copy_optional(
            Path(entry["guidance_path"]).parent / "cumulative_cost.json",
            generation_dir / "outer/cumulative_cost.json",
            records,
            meta_root,
        )


def _copy_current_revision_history(
    meta_root, update_id, revision, destination, records
):
    update_dir = _outer_update_dir(meta_root, update_id)
    for prior in range(int(revision)):
        source = update_dir / f"revision-{prior:02d}"
        if source.is_dir():
            _copy_outer_artifacts(
                source,
                destination / f"revision-{prior:02d}",
                records,
                meta_root,
            )


def verify_meta_agent_context(context_dir):
    context_dir = Path(context_dir)
    manifest_path = context_dir / "context_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"outer context manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported outer context schema")
    expected = {item["path"] for item in manifest.get("files", [])}
    actual = {
        path.relative_to(context_dir).as_posix()
        for path in context_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise ValueError("outer context file set does not match its manifest")
    for item in manifest["files"]:
        path = context_dir / item["path"]
        if path.stat().st_size != item["size_bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"outer context integrity check failed: {path}")
    return manifest


def build_meta_agent_context(meta_run_dir, update_id, revision, mode):
    """Create or verify one immutable proposer/critic context directory."""
    if mode not in MODES:
        raise ValueError(f"unsupported outer-agent mode: {mode}")
    if not update_id.startswith("M") or not update_id[1:].isdigit():
        raise ValueError("update_id must be M00, M01, ...")
    if int(revision) < 0:
        raise ValueError("revision must be non-negative")
    meta_root = Path(meta_run_dir).resolve(strict=True)
    revision_dir = (
        _outer_update_dir(meta_root, update_id) / f"revision-{int(revision):02d}"
    )
    revision_dir.mkdir(parents=True, exist_ok=True)
    revision_dir = _within(revision_dir, meta_root, "revision directory")
    workspace = revision_dir if mode == "propose" else revision_dir / "critic"
    workspace.mkdir(parents=True, exist_ok=True)
    context_dir = workspace / "context"
    if context_dir.exists():
        verify_meta_agent_context(context_dir)
        return context_dir

    state = json.loads((meta_root / "state.json").read_text(encoding="utf-8"))
    from llm_policy.context.modes import task_profile
    latest = state.get("latest_inner_runs", [])
    disclosure = task_profile(Path(latest[0]["run_dir"]) / "task") if latest else None
    if disclosure is not None:
        for entry in state.get("history", []):
            for run in entry.get("inner_runs", []):
                if task_profile(_within(run["run_dir"], meta_root, "inner run") / "task") != disclosure:
                    raise ValueError("meta history cannot mix disclosure profiles")
    if state.get("schema_version") != 2:
        raise ValueError("outer context requires schema-v2 continuous state")
    temporary = workspace / f".context-{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    records = []
    try:
        _write_text(
            temporary / "task_spec.md",
            _task_spec(meta_root.name, update_id, int(revision), mode, state).replace(
                "the exact environment configuration, layout, and source.",
                "the exact environment configuration and permitted factual reference."),
            records,
            "generated:outer-task-spec",
        )
        if mode == "critic":
            from llm_policy.context.environment import load_environment_config
            latest_runs = state.get("latest_inner_runs")
            if not isinstance(latest_runs, list) or not latest_runs:
                raise ValueError("latest inner runs are missing")
            first_run = _within(latest_runs[0]["run_dir"], meta_root, "inner run")
            domain_config = load_environment_config(first_run / "task/context/environment/config.yaml")
            _write_text(
                temporary / "domain_summary.md",
                _critic_domain_summary(domain_config.env_name.lower()),
                records,
                "generated:critic-domain-summary",
            )
            _copy_file(
                revision_dir / "policy_guidance.md",
                temporary / "proposal/policy_guidance.md",
                records,
                meta_root,
            )
            _copy_file(
                workspace / "linter_result.json",
                temporary / "proposal/linter_result.json",
                records,
                meta_root,
            )
        else:
            for name in (
                "manifest.json",
                "state.json",
                "meta_ledger.jsonl",
                "meta_ledger.md",
                "artifact_layout.json",
            ):
                _copy_optional(
                    meta_root / name,
                    temporary / "meta" / name,
                    records,
                    meta_root,
                )
            latest_runs = state.get("latest_inner_runs")
            if not isinstance(latest_runs, list) or not latest_runs:
                raise ValueError("latest inner runs are missing")
            first_run = _within(latest_runs[0]["run_dir"], meta_root, "inner run")
            _copy_tree(
                first_run / "task/context/environment",
                temporary / "environment",
                records,
                meta_root,
            )
            for name in (
                "task_spec.md",
                "policy_api.md",
                "overcooked_primitives.py",
                "hanabi_primitives.py",
                "yokai_primitives.py",
                "yokai16_primitives.py",
                "context_profile.json",
                "rollout_api.py",
                "starting_policy.py",
            ):
                _copy_optional(
                    first_run / "task/context" / name,
                    temporary / "inner_contract" / name,
                    records,
                    meta_root,
                )
            _copy_completed_history(
                meta_root,
                state,
                temporary / "history",
                records,
            )
            _copy_current_revision_history(
                meta_root,
                update_id,
                int(revision),
                temporary / "current_update",
                records,
            )

        normalized = [
            {
                **item,
                "path": Path(item["path"]).relative_to(temporary).as_posix(),
            }
            for item in records
        ]
        manifest = {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "run_id": meta_root.name,
            "update_id": update_id,
            "revision": int(revision),
            "mode": mode,
            "files": sorted(normalized, key=lambda item: item["path"]),
        }
        (temporary / "context_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in temporary.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o444)
        temporary.chmod(0o755)
        os.replace(temporary, context_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    verify_meta_agent_context(context_dir)
    return context_dir


__all__ = [
    "CONTEXT_SCHEMA_VERSION",
    "build_meta_agent_context",
    "verify_meta_agent_context",
]
