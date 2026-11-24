"""Post-run learning curves from explicit model preferences, with resumable scoring.

This module is harness-neutral. It never asks the model to choose retrospectively.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_policy.search.ledger import load_events, verified_candidate_bundle


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("trajectory timestamps must have a timezone")
    return parsed


def _seed_history(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = load_events(run_dir / "task/workspace/policy_ledger.jsonl")
    declarations = [event for event in events if event.get("model_preference")]
    record = {"run_dir": str(run_dir), "run_id": manifest["run_id"],
              "budget_seconds": float(manifest["timeout_seconds"]),
              "started_at": manifest.get("agent_search_started_at"),
              "finished_at": manifest.get("agent_search_finished_at"),
              "declarations": []}
    if not record["started_at"]:
        if declarations:
            raise ValueError(f"{run_dir}: declarations have no authoritative agent start time")
        record["status"] = "unavailable_legacy"
        return record
    started = _timestamp(record["started_at"])
    finished = _timestamp(record["finished_at"]) if record["finished_at"] else None
    if finished is None or finished < started:
        raise ValueError(f"{run_dir}: no valid completed agent-search interval")
    evaluations = {event["event_id"]: event for event in events if event["event"] == "evaluation_recorded"}
    previous_elapsed = -1.0
    for event in declarations:
        declared = _timestamp(event["created_at"])
        elapsed = (declared - started).total_seconds()
        if elapsed < previous_elapsed or elapsed < 0 or declared > finished:
            raise ValueError(f"{run_dir}: model declaration outside chronological search interval")
        previous_elapsed = elapsed
        preference = event["model_preference"]
        evaluation = evaluations[preference["evaluation_event_id"]]
        if _timestamp(evaluation["created_at"]) > declared:
            raise ValueError(f"{run_dir}: preference predates its official evaluation")
        record["declarations"].append({
            **preference, "event_id": event["event_id"], "sequence": event["sequence"],
            "declared_at": event["created_at"], "elapsed_seconds": elapsed,
            "evaluation_recorded_at": evaluation["created_at"],
            "evaluation_attempt_id": evaluation["attempt_id"],
        })
    record["status"] = "available" if declarations else "missing_declarations"
    return record


def selection_at(history, elapsed_seconds):
    """Return only a preference already declared at this cutoff; never a fallback."""
    return next((dict(declaration) for declaration in reversed(history["declarations"])
                 if declaration["elapsed_seconds"] <= elapsed_seconds), None)


def _bundle(run_dir, digest):
    run_dir = Path(run_dir)
    for root in (run_dir / "finalization/candidates", run_dir / "task/workspace/.policy_eval/candidates"):
        candidate = root / digest
        if candidate.exists():
            try:
                return verified_candidate_bundle(candidate, digest)
            except ValueError:
                continue
    return verified_candidate_bundle(run_dir / "final_policy", digest)


def _metrics(matrix):
    n = len(matrix)
    sp = sum(matrix[i][i] for i in range(n)) / n
    xp = sum(matrix[i][j] for i in range(n) for j in range(n) if i != j) / (n * (n - 1)) if n > 1 else None
    return sp, xp


def matrix_statistics(matrix):
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("checkpoint matrix must be square and nonempty")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("checkpoint matrix has nonfinite returns")
    sp, xp = _metrics(matrix)
    sp_se = xp_se = None
    if n > 1:
        sp_se = math.sqrt(sum((matrix[i][i] - sp) ** 2 for i in range(n)) / (n * (n - 1)))
    if n > 2:
        leave_one = [_metrics([[matrix[i][j] for j in range(n) if j != omitted]
                              for i in range(n) if i != omitted])[1] for omitted in range(n)]
        average = sum(leave_one) / n
        xp_se = math.sqrt((n - 1) / n * sum((value - average) ** 2 for value in leave_one))
    return {"sp": sp, "xp": xp, "sp_se": sp_se, "xp_se": xp_se}


def _render(payload, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = payload["checkpoints"]
    fields = ("elapsed_minutes", "status", "sp", "sp_se", "xp", "xp_se")
    with (output_dir / "learning_trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: point.get(key) for key in fields} for point in points)
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        for metric, color in (("sp", "#2563EB"), ("xp", "#D97706")):
            x = [point["elapsed_minutes"] for point in points]
            y = [point.get(metric) if point.get(metric) is not None else float("nan") for point in points]
            lower = [point[metric] - point[metric + "_se"]
                     if point.get(metric + "_se") is not None else float("nan") for point in points]
            upper = [point[metric] + point[metric + "_se"]
                     if point.get(metric + "_se") is not None else float("nan") for point in points]
            ax.plot(x, y, marker="o", color=color, label=metric.upper())
            ax.fill_between(x, lower, upper, color=color, alpha=.18)
            if (not payload["runner_final"]["matches_endpoint_declarations"]
                    or not payload["runner_final"]["same_evaluator"]):
                final_value = payload["runner_final"][metric]
                if final_value is not None:
                    ax.scatter([payload["budget_seconds"] / 60], [final_value],
                               marker="D", facecolors="none", edgecolors=color, s=70,
                               label=f"Runner final {metric.upper()} (post-run)")
        title = "Last model-declared best policy"
        contract = payload.get("runtime_contract") or {}
        if contract.get("adapter") == "yokai":
            title += f"\nYokai | {contract['memory_mode']} memory | {contract['reward_mode']}"
        ax.set(title=title, xlabel="Agent-search wall-clock time (minutes)",
               ylabel="Episode return", xlim=(0, payload["budget_seconds"] / 60))
        ax.grid(alpha=.2)
        ax.legend()
        fig.text(.5, .01,
                 f"{payload['interval_minutes']:g}-minute checkpoints; missing declarations remain gaps. Shading: ±1 policy-seed SE.\n"
                 "Runner final markers, when different, are post-run results shown at the budget boundary.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .07, 1, 1))
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"model-choice-sp-xp.{extension}", dpi=180)
    finally:
        plt.close(fig)


def evaluate_model_choice_trajectory(env, run_dirs, final_summary, output_dir, *, interval_minutes=15):
    """Evaluate and persist complete populations selected from declarations at cutoffs."""
    from llm_policy.search.population import (
        _plain_environment_config, evaluate_codex_xp_matrix, evaluation_fingerprint,
    )

    if not math.isfinite(interval_minutes) or interval_minutes <= 0:
        raise ValueError("interval_minutes must be finite and positive")
    if not run_dirs:
        raise ValueError("run_dirs must be nonempty and ordered like the final XP matrix")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = [_seed_history(path) for path in run_dirs]
    if all(history["status"] == "unavailable_legacy" for history in histories):
        return {"status": "unavailable_legacy", "reason": "Historical runs have no model-choice declarations or agent start timestamps."}
    budgets = {history["budget_seconds"] for history in histories}
    if len(budgets) != 1 or not all(math.isfinite(value) and value > 0 for value in budgets):
        raise ValueError("trajectory population must have one common positive search budget")
    budget = budgets.pop()
    settings = {key: final_summary[key] for key in (
        "root_seed", "num_episodes_per_cell", "num_steps", "common_random_numbers")}
    fingerprint = evaluation_fingerprint(_plain_environment_config(env))
    for path in run_dirs:
        candidate_config = Path(path) / "task/context/environment/config.yaml"
        if _plain_environment_config(candidate_config) != _plain_environment_config(env):
            raise ValueError(f"{path}: search environment differs from XP environment")
    final_hashes = final_summary["candidate_sha256s"]
    if len(final_hashes) != len(run_dirs) or len(final_summary["returns"]) != len(run_dirs):
        raise ValueError("final matrix population size differs from run_dirs")
    for path, digest in zip(run_dirs, final_hashes):
        verified_candidate_bundle(Path(path) / "final_policy", digest)
    checkpoints = [index * interval_minutes * 60 for index in range(int(budget // (interval_minutes * 60)) + 1)]
    if checkpoints[-1] != budget:
        checkpoints.append(budget)
    payload = {"schema_version": 1, "selection_mode": "last_model_declaration_at_or_before_cutoff",
               "clock": "UTC elapsed since runner-recorded agent_search_started_at",
               "budget_seconds": budget, "interval_minutes": interval_minutes,
               "evaluation_settings": settings, "evaluation_fingerprint": fingerprint,
               "runtime_contract": final_summary.get("runtime_contract"),
               "uncertainty": "Delete-one-policy-seed jackknife standard error; SP uses the equivalent seed SEM. XP SE requires at least three policies.",
               "seed_trajectories": histories, "checkpoints": [],
               "runner_final": {**matrix_statistics(final_summary["returns"]),
                                "candidate_sha256s": final_hashes,
                                "evaluation_fingerprint": final_summary.get("evaluation_fingerprint"),
                                "same_evaluator": final_summary.get("evaluation_fingerprint") == fingerprint,
                                "matches_endpoint_declarations": False}}
    failures = []
    for cutoff in checkpoints:
        selections = [{"seed_index": index, "run_id": history["run_id"],
                       "selection": selection_at(history, cutoff)} for index, history in enumerate(histories)]
        point = {"elapsed_minutes": cutoff / 60, "selections": selections}
        missing = [item["seed_index"] for item in selections if item["selection"] is None]
        if missing:
            point.update(status="missing_declarations", missing_seed_indices=missing)
        else:
            hashes = [item["selection"]["candidate_sha256"] for item in selections]
            if cutoff == budget:
                payload["runner_final"]["matches_endpoint_declarations"] = hashes == final_hashes
            try:
                paths = [_bundle(path, digest) for path, digest in zip(run_dirs, hashes)]
                identity = {"candidate_sha256s": hashes, "settings": settings, "evaluation_fingerprint": fingerprint}
                cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                cache_dir = output_dir / "matrices" / cache_key
                cache_path = cache_dir / "checkpoint_cache.json"
                cached = None
                if cache_path.exists():
                    try:
                        candidate_cache = json.loads(cache_path.read_text())
                        if candidate_cache["identity"] == identity:
                            matrix_statistics(candidate_cache["returns"])
                            if len(candidate_cache["returns"]) == len(hashes):
                                cached = candidate_cache
                    except (ValueError, KeyError, TypeError):
                        pass
                if cached is not None:
                    matrix = cached["returns"]
                    source = "checkpoint_cache"
                elif hashes == final_hashes and final_summary.get("evaluation_fingerprint") == fingerprint:
                    matrix = final_summary["returns"]
                    source = "final_xp_matrix"
                else:
                    result = evaluate_codex_xp_matrix(
                        env, paths, root_seed=settings["root_seed"],
                        num_episodes=settings["num_episodes_per_cell"], num_steps=settings["num_steps"],
                        common_random_numbers=settings["common_random_numbers"], output_dir=cache_dir,
                    )
                    if list(result.candidate_sha256s) != hashes:
                        raise ValueError("evaluated candidate hashes differ from the declared selection")
                    matrix = result.returns.tolist()
                    source = "evaluated"
                statistics = matrix_statistics(matrix)
                _write_json(cache_path, {"identity": identity, "returns": matrix})
                point.update(status="completed", directed_xp_matrix=matrix, **statistics,
                             matrix_source=source, cache_key=cache_key)
            except Exception as error:
                point.update(status="failed", error=f"{type(error).__name__}: {error}")
                failures.append(cutoff / 60)
        payload["checkpoints"].append(point)
        _write_json(output_dir / "learning_trajectory.json", payload)
        print(f"Model-choice trajectory: {cutoff / 60:g} min: {point['status']}", flush=True)
    measured = sum(point["status"] == "completed" for point in payload["checkpoints"])
    # The zero checkpoint normally predates the first declaration and is intentionally missing.
    missing_after_zero = any(point["elapsed_minutes"] > 0 and point["status"] == "missing_declarations"
                             for point in payload["checkpoints"])
    status = "completed" if measured and not failures and not missing_after_zero else "partial" if measured else "failed"
    payload["status"] = status
    _write_json(output_dir / "learning_trajectory.json", payload)
    _render(payload, output_dir)
    return {"status": status, "measured_checkpoints": measured, "failed_checkpoints_minutes": failures,
            "artifacts": {"data": "learning_trajectory.json", "csv": "learning_trajectory.csv",
                          "png": "model-choice-sp-xp.png", "pdf": "model-choice-sp-xp.pdf"}}


def generate_model_choice_artifacts(env, run_dirs, xp_output_dir, *, interval_minutes=15):
    """Best-effort population hook: preserve XP success and publish separate status."""
    xp_output_dir = Path(xp_output_dir)
    output_dir = xp_output_dir / "model_choice_trajectory"
    status_path = output_dir / "status.json"
    summary_path = xp_output_dir / "xp_summary.json"
    try:
        _write_json(status_path, {"status": "running"})
        summary = json.loads(summary_path.read_text())
        status = evaluate_model_choice_trajectory(env, run_dirs, summary, output_dir, interval_minutes=interval_minutes)
    except Exception as error:
        status = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    status["directory"] = "model_choice_trajectory"
    if status["status"] in {"failed", "partial"}:
        print(f"Model-choice trajectory needs attention: {json.dumps(status)}", file=sys.stderr, flush=True)
    try:
        _write_json(status_path, status)
        summary = json.loads(summary_path.read_text())
        summary["model_choice_trajectory"] = status
        _write_json(summary_path, summary)
    except (OSError, ValueError) as error:
        print(f"Unable to publish trajectory status: {error}", file=sys.stderr, flush=True)
    return status
