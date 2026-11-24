#!/usr/bin/env python3
"""Immutable scorer exposed to an agentic joint-policy workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import traceback
import uuid
from llm_policy.search.provenance import tracked_command, ACTIVE_EVALUATION
from datetime import datetime, timezone
from pathlib import Path


import jax
import numpy as np

import llm_policy.evaluation.api as eval_api_module
import llm_policy.search.lifetime as lifetime_module
import llm_policy.evaluation.rollout as rollout_module
import llm_policy.policies.runtime as runtime_module
from llm_policy.evaluation.api import (
    compile_candidate,
    load_task,
    run_self_play,
    summarize_episode_trace,
    summarize_rollout,
)
from llm_policy.search.ledger import (
    load_events,
    record_official_evaluation,
    require_current_attempt,
)
from llm_policy.policies.runtime import PolicyValidationError
from llm_policy.resources import asset_path


DEFAULT_ENV_CONFIG = asset_path("config/env/static_cat_dog.yaml")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluator_hash(task=None):
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(eval_api_module.__file__).resolve(),
        Path(lifetime_module.__file__).resolve(),
        Path(rollout_module.__file__).resolve(),
        Path(runtime_module.__file__).resolve(),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    if task is not None and task.environment.env_name.lower() in ("yokai_grid_9_game", "yokai_grid_16_game"):
        from llm_policy.environments.yokai.adapter import source_hashes
        digest.update(json.dumps(source_hashes(), sort_keys=True).encode())
    if task is not None and task.environment.env_name.lower() == "hanabi":
        from llm_policy.environments.hanabi.adapter import source_hashes
        digest.update(json.dumps(source_hashes(), sort_keys=True).encode())
    from llm_policy.policies import source_aware, bundle, validation
    from llm_policy.search import provenance, ledger
    for module in (source_aware, bundle, validation, provenance, ledger):
        digest.update(Path(module.__file__).read_bytes())
    if task is not None and task.environment.env_name.lower() == "cat_dog":
        from llm_policy.environments.cat_dog.adapter import source_hashes
        digest.update(json.dumps(source_hashes(), sort_keys=True).encode())
    return digest.hexdigest()


def _archive_candidate(policy_path: Path, compiled, artifact_dir: Path):
    candidate_dir = artifact_dir / "candidates" / compiled.source_hash
    if candidate_dir.exists():
        from llm_policy.search.ledger import verified_candidate_bundle
        return verified_candidate_bundle(candidate_dir, compiled.source_hash)
    candidate_dir.mkdir(parents=True)
    from llm_policy.policies.bundle import copy_bundle
    copy_bundle(policy_path, candidate_dir)
    return candidate_dir


def _update_best(artifact_dir: Path, report: dict, candidate_dir: Path):
    best_path = artifact_dir / "best.json"
    previous = None
    if best_path.exists():
        try:
            previous = json.loads(best_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
    improved = previous is None or report["mean_return"] > previous["mean_return"]
    if not improved:
        return False
    best_dir = artifact_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    from llm_policy.policies.bundle import copy_bundle
    copy_bundle(candidate_dir, best_dir)
    _write_json(best_path, report)
    return True


def _common_report(args, task, compiled):
    report = {
        "policy": compiled.metadata["name"],
        **({"attempt": args.attempt_ref} if getattr(args,"attempt_ref",None) else {}),
        **({"evaluation_id": ACTIVE_EVALUATION.get()["evaluation_id"]} if ACTIVE_EVALUATION.get() else {}),
        **({"source_exchange": compiled.last_evaluation} if hasattr(compiled, "last_evaluation") else {}),
        "candidate_sha256": compiled.source_hash,
        "policy_path": str(Path(args.policy).resolve()),
        "environment_config": str(task.environment.config_path),
        "environment_config_sha256": _file_sha256(task.environment.config_path),
        "layout": task.environment.env_kwargs.get("layout"),
        "max_steps": task.max_steps,
        "evaluator_sha256": _evaluator_hash(task),
        "devices": [str(device) for device in jax.devices()],
        "created_at": _utc_now(),
    }
    if ACTIVE_EVALUATION.get() is not None:
        ACTIVE_EVALUATION.get().update(evaluator_sha256=report["evaluator_sha256"],
            environment_config_sha256=report["environment_config_sha256"])
    if task.environment.env_name.lower() in ("hanabi", "yokai_grid_9_game", "yokai_grid_16_game", "cat_dog"):
        from llm_policy.context.environment import resolved_runtime_contract
        report["runtime_contract"] = resolved_runtime_contract(task.raw_env)
    lifetime = lifetime_module.agent_lifetime_snapshot()
    if lifetime is not None:
        report["agent_lifetime"] = lifetime
    return report


@tracked_command("validation")
def _validate(args):
    task = load_task(args.env_config)
    compiled = compile_candidate(args.policy, task)
    report = {
        "ok": True,
        "command": "validate",
        **_common_report(args, task, compiled),
        "metadata": compiled.metadata,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


@tracked_command("official_development")
def _score(args):
    policy_path = Path(args.policy).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    ledger_path = Path(
        os.environ.get(
            "LLM_POLICY_LEDGER_PATH",
            artifact_dir.parent / "policy_ledger.jsonl",
        )
    )
    if getattr(args, "attempt_ref", None):
        from llm_policy.search.provenance import locate
        ledger_path = locate(policy_path)
    ledger_context = None
    if ledger_path.exists() and os.environ.get("LLM_POLICY_EVALUATION_KIND") != "final":
        ledger_events = load_events(ledger_path)
        ledger_context = {
            "attempt_id": args.attempt_ref["attempt_id"] if getattr(args,"attempt_ref",None) else require_current_attempt(ledger_events),
            "run_id": ledger_events[0]["run_id"],
            "markdown_path": Path(
                os.environ.get(
                    "LLM_POLICY_LEDGER_MARKDOWN_PATH",
                    ledger_path.with_name("experiment_ledger.md"),
                )
            ),
        }
    if args.error_report_path is not None:
        args.policy_error_phase = "syntax"
        compile(policy_path.read_text(encoding="utf-8"), str(policy_path), "exec")
    args.policy_error_phase = "setup"
    task = load_task(args.env_config)
    args.policy_error_phase = "interface"
    compiled = compile_candidate(policy_path, task)
    args.policy_error_phase = "runtime"
    result = run_self_play(
        task=task,
        policy=compiled,
        root_seed=args.seed,
        num_episodes=args.num_episodes,
        partner_source=args.partner_source,
    )
    if args.error_report_path is not None:
        if not np.isfinite(np.asarray(jax.device_get(result.first_returns))).all():
            raise ValueError("Policy execution produced non-finite output.")
        if not np.asarray(jax.device_get(result.captured), dtype=bool).all():
            raise ValueError("Policy evaluation did not capture every completed episode.")
    # Anything after execution is reporting/archive infrastructure, rather
    # than a policy defect that should trigger another model response.
    args.policy_error_phase = "reporting"
    summary = summarize_rollout(result)
    returns = np.asarray(jax.device_get(result.first_returns), dtype=np.float64)
    captured = np.asarray(jax.device_get(result.captured), dtype=bool)
    worst = np.argsort(returns, kind="stable")[: min(8, returns.size)].tolist()
    report = {
        "ok": True,
        "command": "score",
        "root_seed": int(args.seed),
        **_common_report(args, task, compiled),
        **summary,
        "worst_episode_ids": worst,
    }
    candidate_dir = _archive_candidate(policy_path, compiled, artifact_dir)
    detailed = {
        **report,
        "returns": returns.tolist(),
        "captured": captured.tolist(),
        **({"episode_metric_batches": {name: np.asarray(jax.device_get(value)).tolist()
                                      for name, value in result.episode_metrics.items()}}
           if result.episode_metrics else {}),
    }
    source_suffix = f"_source_{args.partner_source}" if hasattr(compiled, "last_evaluation") else ""
    score_path = candidate_dir / f"score_seed{args.seed}{source_suffix}_{uuid.uuid4().hex}.json"
    _write_json(score_path, detailed)
    report["new_best"] = _update_best(artifact_dir, report, candidate_dir)
    history_path = artifact_dir / "evaluation_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True) + "\n")
    if ledger_context is not None:
        try:
            artifact = str(score_path.relative_to(ledger_path.parent))
        except ValueError:
            artifact = str(score_path)
        record_official_evaluation(
            ledger_path,
            ledger_context["markdown_path"],
            run_id=ledger_context["run_id"],
            attempt_id=ledger_context["attempt_id"],
            report=report,
            artifact=artifact,
        )
    if args.report_path is not None:
        _write_json(Path(args.report_path).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


@tracked_command("trace")
def _trace(args):
    task = load_task(args.env_config)
    compiled = compile_candidate(args.policy, task)
    episode_ids = [int(value) for value in args.episodes.split(",") if value.strip()]
    if not episode_ids:
        raise ValueError("--episodes must contain at least one integer")
    if min(episode_ids) < 0 or max(episode_ids) >= args.num_episodes:
        raise ValueError(
            f"episode IDs must be within [0, {args.num_episodes})"
        )
    result = run_self_play(
        task=task,
        policy=compiled,
        root_seed=args.seed,
        num_episodes=args.num_episodes,
        partner_source=args.partner_source,
    )
    report = {
        "ok": True,
        "command": "trace",
        "root_seed": int(args.seed),
        **_common_report(args, task, compiled),
        "episodes": [
            summarize_episode_trace(result, index) for index in episode_ids
        ],
    }
    if ACTIVE_EVALUATION.get() is not None:
        ACTIVE_EVALUATION.get()["requested_traces"] = report["episodes"]
    print(json.dumps(report, indent=2, sort_keys=True))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    defaults = {
        "env_config": Path(
            os.environ.get("LLM_POLICY_ENV_CONFIG", DEFAULT_ENV_CONFIG)
        ),
        "artifact_dir": Path(
            os.environ.get("LLM_POLICY_ARTIFACT_DIR", ".policy_eval")
        ),
        "num_episodes": int(os.environ.get("LLM_POLICY_NUM_EPISODES", "1024")),
        "seed": int(os.environ.get("LLM_POLICY_DEV_SEED", "0")),
    }

    validate = subparsers.add_parser("validate")
    validate.add_argument("policy", type=Path)
    validate.add_argument("--env-config", type=Path, default=defaults["env_config"])
    validate.set_defaults(function=_validate)

    score = subparsers.add_parser("score")
    score.add_argument("policy", type=Path)
    score.add_argument("--env-config", type=Path, default=defaults["env_config"])
    score.add_argument("--artifact-dir", type=Path, default=defaults["artifact_dir"])
    score.add_argument(
        "--report-path",
        type=Path,
        help="optional compact machine-readable score destination",
    )
    score.add_argument("--error-report-path", type=Path,
                       help="optional structured failure diagnostic for repairs without score feedback")
    score.add_argument("--num-episodes", type=int, default=defaults["num_episodes"])
    score.add_argument("--seed", type=int, default=defaults["seed"])
    score.add_argument("--partner-source", choices=("exchange", "null"), default="exchange")
    score.set_defaults(function=_score)

    trace = subparsers.add_parser("trace")
    trace.add_argument("policy", type=Path)
    trace.add_argument("--episodes", required=True)
    trace.add_argument("--env-config", type=Path, default=defaults["env_config"])
    trace.add_argument("--num-episodes", type=int, default=defaults["num_episodes"])
    trace.add_argument("--seed", type=int, default=defaults["seed"])
    trace.add_argument("--partner-source", choices=("exchange", "null"), default="exchange")
    trace.set_defaults(function=_trace)
    return parser


def _policy_error_report(args, error):
    """Classify policy failures separately from setup/OOM/archive failures."""
    phase = getattr(args, "policy_error_phase", "setup")
    chain = []
    current = error
    while current is not None and id(current) not in {id(item) for item in chain}:
        chain.append(current)
        current = current.__cause__ or current.__context__
    infrastructure = any(isinstance(item, (OSError, MemoryError, TimeoutError, ModuleNotFoundError))
                         for item in chain)
    details = "\n".join(str(item) for item in chain)
    infrastructure |= bool(re.search(
        r"out of memory|resource_exhausted|cuda_error|cudnn_status|"
        r"failed to allocate|failed to initialize.*(?:backend|cuda)|"
        r"unable to initialize.*(?:backend|cuda)|std::bad_alloc|"
        r"failed to create.*thread|no visible gpu|no gpu devices|"
        r"(?:cuda|gpu|dnn|backend).*initializ.*(?:failed|error)|"
        r"memory allocation|allocation failed|cannot allocate memory|could not allocate",
        details, re.IGNORECASE,
    ))
    policy_path = str(Path(args.policy).resolve())
    frames = [frame for item in chain for frame in traceback.extract_tb(item.__traceback__)
              if frame.filename == policy_path]
    ordinary_policy_error = any(isinstance(item, (
        PolicyValidationError, SyntaxError, TypeError, ValueError, NameError,
        AttributeError, KeyError, IndexError, ArithmeticError, AssertionError, ImportError,
    )) for item in chain)
    repairable = (phase in {"syntax", "interface", "runtime"}
                  and not infrastructure and (ordinary_policy_error or bool(frames)))
    message = str(error).replace(policy_path, "policy.py")
    # Only an exception diagnostic is exposed. Never include rollout reports,
    # traces, observations, metrics or numeric performance in repair feedback.
    if "illegal active action or non-finite episode return" in message:
        message = "Policy produced an illegal active action or non-finite output."
    else:
        lines = message.splitlines()
        message = "\n".join(line for line in lines if not re.search(
            r"\b(?:(?:mean|average|avg|median|total|episode)[_ -]*(?:returns?|scores?|rewards?)|"
            r"(?:returns?|scores?|rewards?)['\"]?\s*[:=]|episode_metrics?|"
            r"success[_ -]rate|delivery[_ -]count)", line, re.IGNORECASE,
        ))
    diagnostic = {
        "phase": phase, "type": type(error).__name__,
        "message": message[:2000] or "Policy failed validation or execution.",
        "line": frames[-1].lineno if frames else None,
    }
    if isinstance(error, SyntaxError):
        diagnostic.update(message=error.msg, line=error.lineno, column=error.offset,
                          source_line=error.text)
    return {"ok": False, "command": "score", "repairable": repairable,
            "error": diagnostic}


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        args.function(args)
        return 0
    except Exception as error:
        if args.command == "score" and args.error_report_path is not None:
            report = _policy_error_report(args, error)
            _write_json(args.error_report_path.resolve(), report)
            traceback.print_exc()  # Runner-only log; never included in a repair prompt.
            print(json.dumps(report, indent=2, sort_keys=True))
            return 2
        if not isinstance(error, (PolicyValidationError, ValueError, OSError)):
            raise
        print(
            json.dumps(
                {
                    "ok": False,
                    "command": args.command,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
