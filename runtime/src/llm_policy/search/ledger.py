"""Versioned append-only records for agentic policy search experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


LEDGER_SCHEMA_VERSION = 4
SUPPORTED_LEDGER_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
ATTEMPT_ID_PATTERN = re.compile(r"^A\d{2,}$")
DECISIONS = frozenset({"keep", "reject", "follow_up"})
OBSERVATION_KINDS = frozenset({"exploratory", "validation", "trace", "note"})
COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "event",
        "event_id",
        "sequence",
        "run_id",
        "created_at",
        "author",
    }
)
EVENT_FIELDS = {
    "working_copy_restored": frozenset({"attempt_id", "working_path"}),
    "attempt_annotated": frozenset({"attempt_id", "proposal"}),
    "evaluation_activity": frozenset({"evaluation_id", "attempt_ids", "kind", "status", "artifact"}),
    "attempt_started": frozenset(
        {
            "attempt_id",
            "title",
            "parent",
            "hypothesis",
            "planned_change",
            "success_criterion",
        }
    ),
    "observation_recorded": frozenset({"attempt_id", "observation"}),
    "evaluation_recorded": frozenset(
        {
            "attempt_id",
            "candidate_sha256",
            "evaluation",
            "provenance",
            "artifact",
        }
    ),
    "attempt_concluded": frozenset(
        {
            "attempt_id",
            "candidate_sha256",
            "policy_summary",
            "diagnosis",
            "decision",
            "next_step",
        }
    ),
    "final_selection": frozenset({"selection"}),
    "model_preference_recorded": frozenset({"model_preference"}),
    "run_accounting": frozenset({"accounting"}),
}
GUIDANCE_ALIGNMENT_STATUSES = frozenset(
    {"aligned", "partially_aligned", "not_aligned", "no_guidance"}
)


class LedgerValidationError(ValueError):
    """Raised when an event or attempt lifecycle violates the ledger schema."""


@contextmanager
def _exclusive_ledger_lock(ledger_path: str | Path):
    """Serialize ledger read-modify-append transactions across processes."""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _agent_author():
    """Name the active harness without changing the ledger schema."""
    value = os.environ.get("LLM_POLICY_HARNESS", "codex").strip().lower()
    return value or "agent"


def _require_string(value, field, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise LedgerValidationError(f"{field} must be a non-empty string")


def _require_optional_string(value, field):
    if value is not None:
        _require_string(value, field)


def _require_string_list(value, field):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise LedgerValidationError(f"{field} must be a list of non-empty strings")


def _require_attempt_id(value, field="attempt_id"):
    if not isinstance(value, str) or ATTEMPT_ID_PATTERN.fullmatch(value) is None:
        raise LedgerValidationError(f"{field} must match A00, A01, ...")


def _require_sha256(value, field, *, optional=False):
    if optional and value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise LedgerValidationError(f"{field} must be a 64-character SHA-256")


def _require_mapping(value, field, required_fields):
    if not isinstance(value, dict):
        raise LedgerValidationError(f"{field} must be an object")
    missing = set(required_fields) - set(value)
    extra = set(value) - set(required_fields)
    if missing or extra:
        raise LedgerValidationError(
            f"{field} fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def validate_event(event):
    """Validate one fully normalized ledger event."""
    if not isinstance(event, dict):
        raise LedgerValidationError("ledger event must be an object")
    event_type = event.get("event")
    if event_type not in EVENT_FIELDS:
        raise LedgerValidationError(f"unsupported ledger event {event_type!r}")
    schema_version = event.get("schema_version")
    event_fields = set(EVENT_FIELDS[event_type])
    if event_type == "attempt_started" and schema_version in {2, 3, 4}:
        event_fields.add("policy_guidance_alignment")
    if event_type == "attempt_concluded" and schema_version in {3, 4}:
        event_fields.add("model_preference")
    if event_type == "model_preference_recorded" and schema_version not in {3, 4}:
        raise LedgerValidationError("model preferences require schema version 3")
    if event_type == "attempt_started" and schema_version == 4:
        event_fields.update({"snapshot", "proposal_complete"})
    expected = COMMON_FIELDS | event_fields
    missing = expected - set(event)
    extra = set(event) - expected
    if missing or extra:
        raise LedgerValidationError(
            f"{event_type} fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if event["schema_version"] not in SUPPORTED_LEDGER_SCHEMA_VERSIONS:
        raise LedgerValidationError(
            f"unsupported schema_version {event['schema_version']!r}"
        )
    _require_string(event["event_id"], "event_id")
    if not isinstance(event["sequence"], int) or event["sequence"] < 0:
        raise LedgerValidationError("sequence must be a non-negative integer")
    for field in ("run_id", "created_at", "author"):
        _require_string(event[field], field)

    if "attempt_id" in event:
        _require_attempt_id(event["attempt_id"])

    if event_type == "attempt_started" and schema_version == 4:
        snapshot = event["snapshot"]
        _require_mapping(snapshot, "snapshot", {"candidate_sha256", "archive", "working_path", "origin", "files"})
        _require_sha256(snapshot["candidate_sha256"], "snapshot.candidate_sha256")
        if snapshot["archive"] != f".policy_eval/attempts/{event['attempt_id']}":
            raise LedgerValidationError("snapshot archive does not match attempt")
        if type(event["proposal_complete"]) is not bool or not isinstance(snapshot["files"],dict):
            raise LedgerValidationError("invalid snapshot metadata")
    if event_type == "attempt_annotated":
        from llm_policy.search.provenance import _proposal
        _proposal(event["proposal"])
    if event_type == "evaluation_activity":
        if event["kind"] not in {"validation","compile","trace","exploratory","official_development","final","population_xp"} or event["status"] not in {"running","completed","error"}:
            raise LedgerValidationError("invalid evaluation activity")
        for attempt in event["attempt_ids"]: _require_attempt_id(attempt)
    if event_type == "attempt_started":
        for field in ("title", "hypothesis", "success_criterion"):
            _require_string(event[field], field)
        _require_optional_string(event["parent"], "parent")
        _require_string_list(event["planned_change"], "planned_change")
        if event["schema_version"] in {2, 3, 4}:
            alignment_fields = {"status", "principles", "rationale"}
            alignment = event["policy_guidance_alignment"]
            _require_mapping(
                alignment,
                "policy_guidance_alignment",
                alignment_fields,
            )
            if alignment["status"] not in GUIDANCE_ALIGNMENT_STATUSES:
                raise LedgerValidationError(
                    "policy_guidance_alignment.status must be one of "
                    f"{sorted(GUIDANCE_ALIGNMENT_STATUSES)}"
                )
            _require_string_list(
                alignment["principles"],
                "policy_guidance_alignment.principles",
            )
            _require_string(
                alignment["rationale"],
                "policy_guidance_alignment.rationale",
            )
            if (
                alignment["status"] in {"aligned", "partially_aligned"}
                and not alignment["principles"]
            ):
                raise LedgerValidationError(
                    "aligned proposals must identify at least one guidance principle"
                )
    elif event_type == "observation_recorded":
        observation_fields = {
            "kind",
            "summary",
            "candidate_sha256",
            "root_seed",
            "num_episodes",
            "artifact",
        }
        _require_mapping(event["observation"], "observation", observation_fields)
        observation = event["observation"]
        if observation["kind"] not in OBSERVATION_KINDS:
            raise LedgerValidationError(
                f"observation.kind must be one of {sorted(OBSERVATION_KINDS)}"
            )
        _require_string(observation["summary"], "observation.summary")
        _require_sha256(
            observation["candidate_sha256"],
            "observation.candidate_sha256",
            optional=True,
        )
        for field in ("root_seed", "num_episodes"):
            value = observation[field]
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (1 if field == "num_episodes" else 0)
            ):
                raise LedgerValidationError(
                    f"observation.{field} must be null or a valid non-negative integer"
                )
        _require_optional_string(observation["artifact"], "observation.artifact")
    elif event_type == "evaluation_recorded":
        _require_sha256(event["candidate_sha256"], "candidate_sha256")
        common_evaluation_fields = {
            "kind",
            "root_seed",
            "num_episodes",
            "mean_return",
            "std_return",
            "minimum_return",
            "maximum_return",
            "quantiles",
            "worst_episode_ids",
            "new_best",
            "delta_from_parent",
            "delta_from_best_before",
        }
        provenance_fields = {
            "policy",
            "environment_config_sha256",
            "evaluator_sha256",
            "layout",
            "max_steps",
        }
        evaluation = event["evaluation"]
        if not isinstance(evaluation, dict):
            raise LedgerValidationError("evaluation must be an object")
        # Accept the old field when reading existing append-only ledgers, but
        # all newly recorded events use the accurate episode-capture name.
        capture_fields = {
            field
            for field in ("episode_capture_rate", "completion_rate")
            if field in evaluation
        }
        if len(capture_fields) != 1:
            raise LedgerValidationError(
                "evaluation must contain exactly one of episode_capture_rate "
                "or legacy completion_rate"
            )
        _require_mapping(
            evaluation,
            "evaluation",
            common_evaluation_fields | capture_fields,
        )
        _require_mapping(event["provenance"], "provenance", provenance_fields)
        if evaluation["kind"] != "official_development":
            raise LedgerValidationError(
                "evaluation.kind must be official_development"
            )
        if not isinstance(evaluation["root_seed"], int):
            raise LedgerValidationError("evaluation.root_seed must be an integer")
        if not isinstance(evaluation["num_episodes"], int) or evaluation[
            "num_episodes"
        ] <= 0:
            raise LedgerValidationError("evaluation.num_episodes must be positive")
        for field in (
            "mean_return",
            "std_return",
            "minimum_return",
            "maximum_return",
            next(iter(capture_fields)),
        ):
            if not isinstance(evaluation[field], (int, float)) or isinstance(
                evaluation[field], bool
            ):
                raise LedgerValidationError(f"evaluation.{field} must be numeric")
        for field in ("delta_from_parent", "delta_from_best_before"):
            value = evaluation[field]
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise LedgerValidationError(f"evaluation.{field} must be numeric or null")
        if not isinstance(evaluation["quantiles"], dict):
            raise LedgerValidationError("evaluation.quantiles must be an object")
        if not isinstance(evaluation["worst_episode_ids"], list) or any(
            not isinstance(value, int) for value in evaluation["worst_episode_ids"]
        ):
            raise LedgerValidationError(
                "evaluation.worst_episode_ids must be a list of integers"
            )
        if not isinstance(evaluation["new_best"], bool):
            raise LedgerValidationError("evaluation.new_best must be boolean")
        _require_string(event["provenance"]["policy"], "provenance.policy")
        for field in ("environment_config_sha256", "evaluator_sha256"):
            _require_sha256(event["provenance"][field], f"provenance.{field}")
        _require_optional_string(event["provenance"]["layout"], "provenance.layout")
        if not isinstance(event["provenance"]["max_steps"], int):
            raise LedgerValidationError("provenance.max_steps must be an integer")
        _require_string(event["artifact"], "artifact")
    elif event_type == "attempt_concluded":
        _require_sha256(event["candidate_sha256"], "candidate_sha256", optional=True)
        policy_fields = {"agent_0_role", "agent_1_role", "coordination_protocol"}
        for i in range(2, 5):
            if f"agent_{i}_role" in event["policy_summary"]:
                policy_fields.add(f"agent_{i}_role")
        diagnosis_fields = {
            "failure_modes",
            "supporting_episode_ids",
            "limitations",
        }
        next_fields = {
            "potential_improvement",
            "expected_mechanism",
            "regression_risk",
        }
        _require_mapping(event["policy_summary"], "policy_summary", policy_fields)
        _require_mapping(event["diagnosis"], "diagnosis", diagnosis_fields)
        _require_mapping(event["next_step"], "next_step", next_fields)
        for field in policy_fields:
            _require_string(event["policy_summary"][field], f"policy_summary.{field}")
        for field in ("failure_modes", "limitations"):
            _require_string_list(event["diagnosis"][field], f"diagnosis.{field}")
        episode_ids = event["diagnosis"]["supporting_episode_ids"]
        if not isinstance(episode_ids, list) or any(
            not isinstance(value, int) or value < 0 for value in episode_ids
        ):
            raise LedgerValidationError(
                "diagnosis.supporting_episode_ids must be non-negative integers"
            )
        if event["decision"] not in DECISIONS:
            raise LedgerValidationError(f"decision must be one of {sorted(DECISIONS)}")
        for field in next_fields:
            _require_optional_string(event["next_step"][field], f"next_step.{field}")
    elif event_type == "final_selection":
        if not isinstance(event["selection"], dict):
            raise LedgerValidationError("selection must be an object")
        _require_sha256(
            event["selection"].get("candidate_sha256"),
            "selection.candidate_sha256",
        )
    elif event_type == "run_accounting":
        if not isinstance(event["accounting"], dict):
            raise LedgerValidationError("accounting must be an object")
    if "model_preference" in event:
        preference = event["model_preference"]
        if preference is None:
            if event_type != "attempt_concluded" or (event["author"] != "runner" and event["schema_version"] < 4):
                raise LedgerValidationError("agent conclusions require a model preference")
        else:
            _require_mapping(preference, "model_preference", {
                "candidate_sha256", "evaluation_event_id", "reason",
            })
            _require_sha256(preference["candidate_sha256"], "model_preference.candidate_sha256")
            for field in ("evaluation_event_id", "reason"):
                _require_string(preference[field], f"model_preference.{field}")
    return event


def load_events(ledger_path: str | Path):
    """Load and validate all complete events from a canonical JSONL ledger."""
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    events = []
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise LedgerValidationError(
                f"{ledger_path}:{line_number}: invalid JSON: {error}"
            ) from error
        validate_event(event)
        if event["sequence"] != len(events):
            raise LedgerValidationError(
                f"{ledger_path}:{line_number}: expected sequence {len(events)}, "
                f"got {event['sequence']}"
            )
        _validate_preference_reference(events, event)
        events.append(event)
    return events


def _run_id(events, requested):
    if events:
        existing = events[0]["run_id"]
        if requested is not None and requested != existing:
            raise LedgerValidationError(
                f"run_id mismatch: ledger has {existing!r}, requested {requested!r}"
            )
        return existing
    _require_string(requested, "run_id")
    return requested


def _attempt_state(events):
    starts = {}
    concluded = set()
    for event in events:
        if event["event"] == "attempt_started":
            attempt_id = event["attempt_id"]
            if attempt_id in starts:
                raise LedgerValidationError(f"duplicate attempt {attempt_id}")
            starts[attempt_id] = event
        elif event["event"] == "attempt_concluded":
            concluded.add(event["attempt_id"])
    open_attempts = [attempt_id for attempt_id in starts if attempt_id not in concluded]
    if events and events[0]["schema_version"] >= 4:
        return starts, concluded, next(reversed(starts), None)
    if len(open_attempts) > 1:
        raise LedgerValidationError(f"multiple open attempts: {open_attempts}")
    return starts, concluded, open_attempts[0] if open_attempts else None


def current_attempt(events):
    """Return the single open attempt, or ``None`` when all are concluded."""
    return _attempt_state(events)[2]


def require_current_attempt(events):
    attempt_id = current_attempt(events)
    if attempt_id is None:
        raise LedgerValidationError(
            "no open attempt; run `policy-ledger start` before evaluating"
        )
    return attempt_id


def next_attempt_id(events):
    starts, _, _ = _attempt_state(events)
    highest = max((int(value[1:]) for value in starts), default=-1)
    return f"A{highest + 1:02d}"


def ledger_progress(events):
    starts, concluded, open_attempt = _attempt_state(events)
    return {
        "attempt_count": len(starts),
        "concluded_attempt_count": len(concluded),
        "current_attempt": open_attempt,
        "model_preference": next(
            ({**event["model_preference"], "declared_at": event["created_at"],
              "event_id": event["event_id"], "sequence": event["sequence"]}
             for event in reversed(events) if event.get("model_preference")),
            None,
        ),
    }


def _validate_preference_reference(events, event):
    preference = event.get("model_preference")
    if preference is None:
        return
    if not any(
        prior["event"] == "evaluation_recorded"
        and prior["event_id"] == preference["evaluation_event_id"]
        and prior["candidate_sha256"] == preference["candidate_sha256"]
        and prior["run_id"] == event["run_id"]
        and datetime.fromisoformat(prior["created_at"].replace("Z", "+00:00"))
        <= datetime.fromisoformat(event["created_at"].replace("Z", "+00:00"))
        for prior in events
    ):
        raise LedgerValidationError("model preference must reference an earlier official evaluation in this run")


def verified_candidate_bundle(candidate_dir, candidate_sha256):
    """Check the exact archived source without importing agent-authored code."""
    candidate_dir = Path(candidate_dir)
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise LedgerValidationError(f"missing regular candidate directory: {candidate_dir}")
    from llm_policy.policies.bundle import bundle_hash
    try:
        actual = bundle_hash(candidate_dir)
    except (ValueError, OSError) as error:
        raise LedgerValidationError(str(error)) from error
    if actual != candidate_sha256:
        raise LedgerValidationError(f"archived source hash mismatch: {candidate_dir}")
    return candidate_dir


def _model_preference(events, ledger_path, candidate_sha256, reason, artifact_dir=None):
    _require_sha256(candidate_sha256, "best_candidate_sha256")
    _require_string(reason, "best_reason")
    evaluation = next((event for event in reversed(events)
                       if event["event"] == "evaluation_recorded"
                       and event["candidate_sha256"] == candidate_sha256), None)
    if evaluation is None:
        raise LedgerValidationError("preferred candidate must have an official evaluation in this run")
    if events[0]["schema_version"] >= 4:
        from llm_policy.search.provenance import reference
        reference(ledger_path, evaluation["attempt_id"])
    root = Path(artifact_dir) if artifact_dir is not None else Path(ledger_path).parent / ".policy_eval"
    verified_candidate_bundle(root / "candidates" / candidate_sha256, candidate_sha256)
    return {"candidate_sha256": candidate_sha256,
            "evaluation_event_id": evaluation["event_id"], "reason": reason}


def record_model_preference(ledger_path, markdown_path, *, run_id,
                            candidate_sha256, reason, artifact_dir=None):
    """Record the model's explicit submission preference, independent of score rank."""
    with _exclusive_ledger_lock(ledger_path):
        events = load_events(ledger_path)
        preference = _model_preference(events, ledger_path, candidate_sha256, reason, artifact_dir)
        return _append_event_unlocked(ledger_path, markdown_path, {
            "event": "model_preference_recorded", "author": _agent_author(),
            "model_preference": preference,
        }, run_id=run_id)


def _latest_verified_preference(events, ledger_path, artifact_dir=None):
    event = next((event for event in reversed(events) if event.get("model_preference")), None)
    if event is None:
        raise LedgerValidationError(
            "no declared preference; score a candidate and run policy-ledger select-best first"
        )
    preference = event["model_preference"]
    _model_preference(events, ledger_path, preference["candidate_sha256"],
                      preference["reason"], artifact_dir)
    return event


def require_declared_submission(ledger_path, *, artifact_dir=None):
    """Require the working bundle to match the latest verified model preference."""
    ledger_path = Path(ledger_path)
    from llm_policy.search.provenance import register_candidate
    register_candidate(ledger_path.parent / "policy.py", explicit=False, origin="final_submission")
    with _exclusive_ledger_lock(ledger_path):
        event = _latest_verified_preference(load_events(ledger_path), ledger_path, artifact_dir)
        digest = event["model_preference"]["candidate_sha256"]
        try:
            verified_candidate_bundle(ledger_path.parent, digest)
        except LedgerValidationError as error:
            raise LedgerValidationError(
                "XP submission does not match the last declared candidate; run "
                "policy-ledger restore-best after your final select-best declaration"
            ) from error
        return digest


def restore_model_preference(ledger_path, *, run_id, artifact_dir=None):
    """Restore an exact declared bundle, retaining recoverable working-file backups.

    No preference or evaluation events are added. The archive remains unchanged.
    Both files are staged before replacement; a failed replacement rolls back.
    """
    ledger_path = Path(ledger_path)
    workspace = ledger_path.parent
    from llm_policy.policies.bundle import ALL_FILES
    names = ALL_FILES
    from llm_policy.search.provenance import register_candidate
    register_candidate(workspace / "policy.py", explicit=False, origin="before_restore")
    with _exclusive_ledger_lock(ledger_path):
        events = load_events(ledger_path)
        if not events or events[0]["run_id"] != run_id:
            raise LedgerValidationError("restore run ID does not match the ledger")
        event = _latest_verified_preference(events, ledger_path, artifact_dir)
        digest = event["model_preference"]["candidate_sha256"]
        root = Path(artifact_dir) if artifact_dir is not None else workspace / ".policy_eval"
        archive = verified_candidate_bundle(root / "candidates" / digest, digest)
        if archive.resolve() == workspace.resolve():
            raise LedgerValidationError("archive and working directory must be distinct")
        for name in names:
            target = workspace / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise LedgerValidationError(f"working source must be a regular file: {target}")
        recovery = Path(tempfile.mkdtemp(prefix=".policy-restore-", dir=workspace))
        backup = recovery / "previous"
        staged = recovery / "selected"
        backup.mkdir()
        staged.mkdir()
        original_names = []
        for name in names:
            if (workspace / name).exists():
                shutil.copy2(workspace / name, backup / name)
                original_names.append(name)
            if (archive / name).exists():
                shutil.copyfile(archive / name, staged / name)
        verified_candidate_bundle(staged, digest)
        (recovery / "restore.json").write_text(json.dumps({
            "candidate_sha256": digest, "preference_event_id": event["event_id"],
            "previous_files": original_names,
        }, indent=2)+"\n", encoding="utf-8")
        try:
            for name in names:
                target = workspace / name
                if (staged / name).exists():
                    os.replace(staged / name, target)
                elif target.exists():
                    # Selected bundle has no custom file: move the stale working
                    # file out of the workspace root; never leave a mixed bundle.
                    os.replace(target, recovery / ("displaced-"+name))
            verified_candidate_bundle(workspace, digest)
        except Exception as error:
            try:
                for name in names:
                    target = workspace / name
                    if name in original_names:
                        rollback = recovery / ("rollback-"+name)
                        shutil.copy2(backup / name, rollback)
                        os.replace(rollback, target)
                    elif target.exists():
                        os.replace(target, recovery / ("failed-"+name))
            except Exception as rollback_error:
                raise LedgerValidationError(
                    f"restore and rollback failed; recover original files from {backup}: "
                    f"{rollback_error}"
                ) from error
            raise LedgerValidationError(
                f"restore failed; original working files restored, backups at {backup}: {error}"
            ) from error
        if events[0]["schema_version"] >= 4:
            selected_evaluation = next(e for e in events if e["event_id"] == event["model_preference"]["evaluation_event_id"])
            _append_event_unlocked(ledger_path, ledger_path.with_name("experiment_ledger.md"),
                {"event":"working_copy_restored","author":"runner", "attempt_id":selected_evaluation["attempt_id"],
                 "working_path":str(workspace / "policy.py")},run_id=run_id)
        return {"candidate_sha256": digest, "preference_event_id": event["event_id"],
                "restored_files": [name for name in names if (archive / name).exists()],
                "backup_dir": str(backup), "preference_unchanged": True}


def _normalize_event(events, event, *, run_id=None):
    value = dict(event)
    value.update(
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "sequence": len(events),
            "run_id": _run_id(events, run_id),
            "created_at": _utc_now(),
        }
    )
    return validate_event(value)


def _append_event_unlocked(
    ledger_path: str | Path,
    markdown_path: str | Path,
    event,
    *,
    run_id=None,
):
    """Append one event while the caller holds the ledger lock."""
    ledger_path = Path(ledger_path)
    markdown_path = Path(markdown_path)
    events = load_events(ledger_path)
    normalized = _normalize_event(events, event, run_id=run_id)
    _validate_preference_reference(events, normalized)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    events.append(normalized)
    render_markdown(events, markdown_path)
    return normalized


def append_event(
    ledger_path: str | Path,
    markdown_path: str | Path,
    event,
    *,
    run_id=None,
):
    """Atomically validate, append, and render one canonical event."""
    with _exclusive_ledger_lock(ledger_path):
        return _append_event_unlocked(
            ledger_path,
            markdown_path,
            event,
            run_id=run_id,
        )


def initialize_ledger(
    ledger_path: str | Path,
    markdown_path: str | Path,
    *,
    run_id,
    inherited_baseline=False,
):
    """Freeze the canonical A00 baseline before generation."""
    if not Path(ledger_path).exists():
        from llm_policy.search.provenance import register_candidate
        return register_candidate(Path(ledger_path).parent / "policy.py", ledger_path=ledger_path,
            run_id=run_id, origin="inherited_baseline" if inherited_baseline else "starter",
            runner=True, metadata={"title":"Inherited baseline" if inherited_baseline else "Starting template",
            "hypothesis":"Runner baseline; no agent-authored hypothesis.",
            "planned_change":["Preserve the exact baseline source before generation."],
            "success_criterion":"Establish the configured baseline."})
    raise FileExistsError(f"ledger already exists: {ledger_path}")


def start_attempt(
    ledger_path,
    markdown_path,
    *,
    run_id,
    attempt_id,
    title,
    parent,
    hypothesis,
    planned_change,
    success_criterion,
    guidance_alignment,
    guidance_principles,
    guidance_rationale,
):
    with _exclusive_ledger_lock(ledger_path):
        events = load_events(ledger_path)
        if events and events[0]["schema_version"] >= 4:
            raise LedgerValidationError("Edit first, then use policy-ledger register policy.py; start no longer opens mutable attempts")
        if current_attempt(events) is not None:
            raise LedgerValidationError(
                f"attempt {current_attempt(events)} is still open; conclude it first"
            )
        attempt_id = attempt_id or next_attempt_id(events)
        _require_attempt_id(attempt_id)
        if any(
            event.get("attempt_id") == attempt_id
            and event["event"] == "attempt_started"
            for event in events
        ):
            raise LedgerValidationError(f"attempt {attempt_id} already exists")
        return _append_event_unlocked(
            ledger_path,
            markdown_path,
            {
                "event": "attempt_started",
                "author": _agent_author(),
                "attempt_id": attempt_id,
                "title": title,
                "parent": parent,
                "hypothesis": hypothesis,
                "planned_change": list(planned_change),
                "success_criterion": success_criterion,
                "policy_guidance_alignment": {
                    "status": guidance_alignment,
                    "principles": list(guidance_principles),
                    "rationale": guidance_rationale,
                },
            },
            run_id=run_id,
        )


def record_observation(
    ledger_path,
    markdown_path,
    *,
    run_id,
    attempt_id=None,
    kind,
    summary,
    candidate_sha256=None,
    root_seed=None,
    num_episodes=None,
    artifact=None,
):
    with _exclusive_ledger_lock(ledger_path):
        events = load_events(ledger_path)
        if attempt_id is None:
            attempt_id = require_current_attempt(events)
        else:
            _require_attempt_id(attempt_id)
            starts, _, _ = _attempt_state(events)
            if attempt_id not in starts:
                raise LedgerValidationError(f"unknown attempt {attempt_id}")
        return _append_event_unlocked(
            ledger_path,
            markdown_path,
            {
                "event": "observation_recorded",
                "author": _agent_author(),
                "attempt_id": attempt_id,
                "observation": {
                    "kind": kind,
                    "summary": summary,
                    "candidate_sha256": candidate_sha256,
                    "root_seed": root_seed,
                    "num_episodes": num_episodes,
                    "artifact": artifact,
                },
            },
            run_id=run_id,
        )


def _latest_candidate(events, attempt_id):
    candidates = [
        event["candidate_sha256"]
        for event in events
        if event["event"] == "evaluation_recorded"
        and event["attempt_id"] == attempt_id
    ]
    return candidates[-1] if candidates else None


def conclude_attempt(
    ledger_path,
    markdown_path,
    *,
    run_id,
    candidate_sha256,
    agent_0_role,
    agent_1_role,
    agent_2_role=None,
    agent_3_role=None,
    agent_4_role=None,
    coordination_protocol,
    failure_modes,
    supporting_episode_ids,
    limitations,
    decision,
    potential_improvement,
    expected_mechanism,
    regression_risk,
    author=None,
    best_candidate_sha256=None,
    best_reason=None,
    artifact_dir=None,
    attempt_id=None,
):
    with _exclusive_ledger_lock(ledger_path):
        events = load_events(ledger_path)
        attempt_id = attempt_id or require_current_attempt(events)
        if events[0]["schema_version"] >= 4:
            from llm_policy.search.provenance import reference
            frozen = reference(ledger_path, attempt_id)["candidate_sha256"]
            if candidate_sha256 not in (None, frozen):
                raise LedgerValidationError("conclusion hash differs from frozen attempt")
            candidate_sha256 = frozen
        candidate_sha256 = candidate_sha256 or _latest_candidate(events, attempt_id)
        preference = None
        if (events[0]["schema_version"] < 4 and author != "runner") or best_candidate_sha256 is not None:
            preference = _model_preference(
                events, ledger_path, best_candidate_sha256, best_reason, artifact_dir
            )
        return _append_event_unlocked(
            ledger_path,
            markdown_path,
            {
                "event": "attempt_concluded",
                "author": author or _agent_author(),
                "attempt_id": attempt_id,
                "candidate_sha256": candidate_sha256,
                "model_preference": preference,
                "policy_summary": {
                    "agent_0_role": agent_0_role,
                    "agent_1_role": agent_1_role,
                    "coordination_protocol": coordination_protocol,
                    **{f"agent_{i}_role": role for i, role in enumerate((agent_2_role, agent_3_role, agent_4_role), 2) if role is not None},
                },
                "diagnosis": {
                    "failure_modes": list(failure_modes),
                    "supporting_episode_ids": list(supporting_episode_ids),
                    "limitations": list(limitations),
                },
                "decision": decision,
                "next_step": {
                    "potential_improvement": potential_improvement,
                    "expected_mechanism": expected_mechanism,
                    "regression_risk": regression_risk,
                },
            },
            run_id=run_id,
        )


def _evaluation_score(event):
    return event["evaluation"]["mean_return"]


def _parent_score(events, attempt_id):
    starts, _, _ = _attempt_state(events)
    parent = starts[attempt_id]["parent"]
    if parent is None:
        return None
    matches = []
    for event in events:
        if event["event"] != "evaluation_recorded":
            continue
        if event["attempt_id"] == parent or event["candidate_sha256"] == parent:
            matches.append(event)
    return _evaluation_score(matches[-1]) if matches else None


def record_official_evaluation(
    ledger_path,
    markdown_path,
    *,
    run_id,
    attempt_id,
    report,
    artifact,
):
    """Append one immutable-scorer result to the current attempt."""
    with _exclusive_ledger_lock(ledger_path):
        events = load_events(ledger_path)
        starts, _, _ = _attempt_state(events)
        if attempt_id not in starts:
            raise LedgerValidationError(
                f"unknown scoring attempt: expected existing {attempt_id}"
            )
        for event in events:
            if (
                events[0]["schema_version"] < 4
                and event["event"] == "evaluation_recorded"
                and event["attempt_id"] == attempt_id
                and event["candidate_sha256"] == report["candidate_sha256"]
                and event["evaluation"]["root_seed"] == int(report["root_seed"])
                and event["evaluation"]["num_episodes"]
                == int(report["num_episodes"])
            ):
                return event
        if events[0]["schema_version"] >= 4:
            from llm_policy.search.provenance import reference
            if reference(ledger_path, attempt_id)["candidate_sha256"] != report["candidate_sha256"]:
                raise LedgerValidationError("evaluation differs from frozen attempt")
        prior_scores = [
            _evaluation_score(event)
            for event in events
            if event["event"] == "evaluation_recorded"
        ]
        mean_return = float(report["mean_return"])
        parent_score = _parent_score(events, attempt_id)
        best_before = max(prior_scores) if prior_scores else None
        return _append_event_unlocked(
            ledger_path,
            markdown_path,
            {
                "event": "evaluation_recorded",
                "author": "policy_eval",
                "attempt_id": attempt_id,
                "candidate_sha256": report["candidate_sha256"],
                "evaluation": {
                    "kind": "official_development",
                    "root_seed": int(report["root_seed"]),
                    "num_episodes": int(report["num_episodes"]),
                    "mean_return": mean_return,
                    "std_return": float(report["std_return"]),
                    "minimum_return": float(report["minimum_return"]),
                    "maximum_return": float(report["maximum_return"]),
                    "episode_capture_rate": float(report["episode_capture_rate"]),
                    "quantiles": dict(report["quantiles"]),
                    "worst_episode_ids": list(report["worst_episode_ids"]),
                    "new_best": bool(report["new_best"]),
                    "delta_from_parent": (
                        mean_return - parent_score if parent_score is not None else None
                    ),
                    "delta_from_best_before": (
                        mean_return - best_before if best_before is not None else None
                    ),
                },
                "provenance": {
                    "policy": report["policy"],
                    "environment_config_sha256": report[
                        "environment_config_sha256"
                    ],
                    "evaluator_sha256": report["evaluator_sha256"],
                    "layout": report.get("layout"),
                    "max_steps": int(report["max_steps"]),
                },
                "artifact": str(artifact),
            },
            run_id=run_id,
        )


def append_final_selection(ledger_path, markdown_path, *, run_id, final):
    return append_event(
        ledger_path,
        markdown_path,
        {
            "event": "final_selection",
            "author": "runner",
            "selection": dict(final),
        },
        run_id=run_id,
    )


def append_run_accounting(ledger_path, markdown_path, *, run_id, usage):
    return append_event(
        ledger_path,
        markdown_path,
        {
            "event": "run_accounting",
            "author": "runner",
            "accounting": dict(usage),
        },
        run_id=run_id,
    )


def _md(value):
    if value is None:
        return "none"
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_list(lines, label, values):
    lines.append(f"- {label}:")
    if values:
        lines.extend(f"  - {_md(value)}" for value in values)
    else:
        lines.append("  - none")


def render_markdown(events, markdown_path: str | Path):
    """Render a deterministic human-readable view of canonical events."""
    markdown_path = Path(markdown_path)
    run_id = events[0]["run_id"] if events else "unknown"
    lines = [
        "# Policy experiment ledger",
        "",
        "> Generated from `policy_ledger.jsonl`; do not edit this file directly.",
        "",
        f"- Schema version: {LEDGER_SCHEMA_VERSION}",
        f"- Run: `{_md(run_id)}`",
    ]
    starts = [event for event in events if event["event"] == "attempt_started"]
    if events and events[0]["schema_version"] >= 4:
        from llm_policy.search.provenance import effective_start
        starts = [effective_start(events, e["attempt_id"]) for e in starts]
        lines += ["", "## Attempt index", "", "| Attempt | Source | Proposal notes | Conclusion |", "|---|---|---|---|"]
        for start in starts:
            concluded = any(e["event"] == "attempt_concluded" and e["attempt_id"] == start["attempt_id"] for e in events)
            lines.append(f"| {start['attempt_id']} | [{start['snapshot']['candidate_sha256'][:12]}]({start['snapshot']['archive']}/policy.py) | {'complete' if start['proposal_complete'] else 'MISSING'} | {'recorded' if concluded else 'MISSING'} |")
    else:
        lines += ["", "> Legacy ledger: unregistered or overwritten revisions cannot be reconstructed."]
    for start in starts:
        attempt_id = start["attempt_id"]
        related = [event for event in events if event.get("attempt_id") == attempt_id]
        conclusion = next(
            (event for event in related if event["event"] == "attempt_concluded"),
            None,
        )
        lines.extend(
            [
                "",
                f"## {attempt_id} — {_md(start['title'])}",
                "",
                f"- Status: {'concluded' if conclusion else 'open'}",
                f"- Parent: `{_md(start['parent'])}`",
                f"- Hypothesis: {_md(start['hypothesis'])}",
                f"- Success criterion: {_md(start['success_criterion'])}",
            ]
        )
        alignment = start.get("policy_guidance_alignment")
        if alignment is None:
            lines.append("- Policy-guidance alignment: not recorded (legacy event)")
        elif alignment["status"] != "no_guidance":
            lines.extend(
                [
                    "- Policy-guidance alignment: "
                    f"`{_md(alignment['status'])}`",
                    "- Policy-guidance rationale: "
                    f"{_md(alignment['rationale'])}",
                ]
            )
            _render_list(
                lines,
                "Policy-guidance principles",
                alignment["principles"],
            )
        _render_list(lines, "Planned change", start["planned_change"])

        observations = [
            event for event in related if event["event"] == "observation_recorded"
        ]
        evaluations = [
            event for event in related if event["event"] == "evaluation_recorded"
        ]
        if "snapshot" in start:
            lines += ["", f"- Frozen bundle: `{start['snapshot']['candidate_sha256']}`", f"- Capture origin: {start['snapshot']['origin']}", "", "### Evaluation history", ""]
            for event in events:
                if event["event"] == "evaluation_activity" and attempt_id in event["attempt_ids"]:
                    lines.append(f"- {event['kind']}: **{event['status']}** — [{event['evaluation_id']}]({event['artifact']})")
        if observations:
            lines.extend(["", "### Exploratory evidence", ""])
            for event in observations:
                observation = event["observation"]
                details = [observation["kind"]]
                if observation["root_seed"] is not None:
                    details.append(f"seed={observation['root_seed']}")
                if observation["num_episodes"] is not None:
                    details.append(f"episodes={observation['num_episodes']}")
                lines.append(
                    f"- **{', '.join(details)}:** {_md(observation['summary'])}"
                )
        if evaluations:
            lines.extend(["", "### Official evaluations", ""])
            lines.append(
                "| Candidate | Episodes | Mean | Std | Δ parent | Δ prior best | New best |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|:---:|")
            for event in evaluations:
                evaluation = event["evaluation"]
                lines.append(
                    "| `{candidate}` | {episodes} | {mean:.6g} | {std:.6g} | "
                    "{parent} | {best} | {new_best} |".format(
                        candidate=event["candidate_sha256"][:12],
                        episodes=evaluation["num_episodes"],
                        mean=evaluation["mean_return"],
                        std=evaluation["std_return"],
                        parent=_md(evaluation["delta_from_parent"]),
                        best=_md(evaluation["delta_from_best_before"]),
                        new_best="yes" if evaluation["new_best"] else "no",
                    )
                )
        if conclusion:
            policy = conclusion["policy_summary"]
            diagnosis = conclusion["diagnosis"]
            next_step = conclusion["next_step"]
            lines.extend(
                [
                    "",
                    "### Policy and conclusion",
                    "",
                    f"- Candidate: `{_md(conclusion['candidate_sha256'])}`",
                    f"- Agent 0 role: {_md(policy['agent_0_role'])}",
                    f"- Agent 1 role: {_md(policy['agent_1_role'])}",
                    *[f"- Agent {i} role: {_md(policy[f'agent_{i}_role'])}"
                      for i in range(2, 5) if f"agent_{i}_role" in policy],
                    f"- Coordination protocol: {_md(policy['coordination_protocol'])}",
                    f"- Decision: **{_md(conclusion['decision'])}**",
                ]
            )
            _render_list(lines, "Failure modes", diagnosis["failure_modes"])
            _render_list(lines, "Supporting episode IDs", diagnosis["supporting_episode_ids"])
            _render_list(lines, "Limitations", diagnosis["limitations"])
            lines.extend(
                [
                    f"- Potential improvement: {_md(next_step['potential_improvement'])}",
                    f"- Expected mechanism: {_md(next_step['expected_mechanism'])}",
                    f"- Regression risk: {_md(next_step['regression_risk'])}",
                ]
            )

    preferences = [event for event in events if event.get("model_preference")]
    if preferences:
        lines.extend(["", "## Model-declared best policy", "",
                      "The last declaration remains the model's recorded preference until replaced.", "",
                      "| Declared at (UTC) | Candidate | Evaluation event | Reason |",
                      "|---|---|---|---|"])
        for event in preferences:
            preference = event["model_preference"]
            lines.append(f"| {_md(event['created_at'])} | `{preference['candidate_sha256']}` | "
                         f"`{preference['evaluation_event_id']}` | {_md(preference['reason'])} |")
    selections = [event for event in events if event["event"] == "final_selection"]
    if selections:
        selection = selections[-1]["selection"]
        episode_capture_rate = selection.get(
            "episode_capture_rate",
            selection.get("completion_rate"),
        )
        lines.extend(
            [
                "",
                "## Runner-authenticated final selection",
                "",
                f"- Candidate: `{_md(selection.get('candidate_sha256'))}`",
                f"- Evaluation type: `{_md(selection.get('split'))}`",
                f"- Seed and episodes: {selection.get('root_seed')}, "
                f"{selection.get('num_episodes')}",
                f"- Mean return: {selection.get('mean_return')}",
                f"- Episode capture rate: {episode_capture_rate}",
            ]
        )
    accounting_events = [
        event for event in events if event["event"] == "run_accounting"
    ]
    if accounting_events:
        accounting = accounting_events[-1]["accounting"]
        harness = accounting.get("harness", "codex")
        agent = accounting.get("agent", accounting.get(harness, {}))
        cost = accounting.get("openrouter_cost", {})
        provider = accounting.get("provider", {})
        lines.extend(
            [
                "",
                "## Runner-authenticated final accounting",
                "",
                f"- Captured: `{_md(accounting.get('captured_at'))}`",
                f"- Model: `{_md(accounting.get('model'))}`",
                f"- Harness: `{_md(harness)}`",
                f"- Input tokens: {agent.get('input_tokens', 0):,}",
                f"- Output tokens: {agent.get('output_tokens', 0):,}",
                "- Reasoning output tokens: "
                f"{agent.get('reasoning_output_tokens', 0):,}",
                f"- Billing route: `{_md(provider.get('billing_route'))}`",
                f"- Measured provider charge: {_md(cost.get('charged_credits'))}",
                f"- Cost attribution: `{_md(cost.get('attribution'))}`",
            ]
        )
    text = "\n".join(lines) + "\n"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown_path.with_name(
        f".{markdown_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(markdown_path)
    finally:
        temporary.unlink(missing_ok=True)


def render_ledger(ledger_path: str | Path, markdown_path: str | Path):
    """Render a consistent ledger snapshot while excluding concurrent writers."""
    with _exclusive_ledger_lock(ledger_path):
        events = load_events(ledger_path)
        render_markdown(events, markdown_path)
        return len(events)


__all__ = [
    "DECISIONS",
    "GUIDANCE_ALIGNMENT_STATUSES",
    "LEDGER_SCHEMA_VERSION",
    "LedgerValidationError",
    "OBSERVATION_KINDS",
    "append_event",
    "append_final_selection",
    "append_run_accounting",
    "conclude_attempt",
    "current_attempt",
    "initialize_ledger",
    "ledger_progress",
    "load_events",
    "next_attempt_id",
    "record_observation",
    "record_model_preference",
    "restore_model_preference",
    "require_declared_submission",
    "record_official_evaluation",
    "render_ledger",
    "render_markdown",
    "require_current_attempt",
    "start_attempt",
    "validate_event",
    "verified_candidate_bundle",
]
