"""Agent-facing wall-clock lifetime snapshots for policy search."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping

LIFETIME_AWARENESS_ENV = "LLM_POLICY_LIFETIME_AWARENESS"
LIFETIME_STARTED_MONOTONIC_ENV = "LLM_POLICY_AGENT_STARTED_MONOTONIC"
LIFETIME_TIMEOUT_SECONDS_ENV = "LLM_POLICY_AGENT_TIMEOUT_SECONDS"


class LifetimeConfigurationError(ValueError):
    """Raised when enabled lifetime metadata is missing or malformed."""


def _enabled(environ: Mapping[str, str]) -> bool:
    value = environ.get(LIFETIME_AWARENESS_ENV, "false")
    if value not in {"true", "false"}:
        raise LifetimeConfigurationError(
            f"{LIFETIME_AWARENESS_ENV} must be 'true' or 'false'"
        )
    return value == "true"


def agent_lifetime_snapshot(
    *,
    environ: Mapping[str, str] | None = None,
    now_monotonic: float | None = None,
    required: bool = False,
):
    """Return the current agent-search lifetime, or ``None`` when disabled."""
    environ = os.environ if environ is None else environ
    if not _enabled(environ):
        if required:
            raise LifetimeConfigurationError("agent lifetime awareness is disabled")
        return None

    try:
        started = float(environ[LIFETIME_STARTED_MONOTONIC_ENV])
        configured = float(environ[LIFETIME_TIMEOUT_SECONDS_ENV])
    except KeyError as error:
        raise LifetimeConfigurationError(
            f"enabled lifetime awareness is missing {error.args[0]}"
        ) from error
    except (TypeError, ValueError) as error:
        raise LifetimeConfigurationError(
            "enabled lifetime awareness has a non-numeric clock value"
        ) from error
    try:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
    except (TypeError, ValueError) as error:
        raise LifetimeConfigurationError(
            "current monotonic clock value must be numeric"
        ) from error
    if not all(math.isfinite(value) for value in (started, configured, now)):
        raise LifetimeConfigurationError("agent lifetime clock values must be finite")
    if configured <= 0:
        raise LifetimeConfigurationError("agent lifetime must be positive")

    elapsed = max(now - started, 0.0)
    remaining = max(configured - elapsed, 0.0)
    return {
        "configured_seconds": configured,
        "elapsed_seconds": elapsed,
        "remaining_seconds": remaining,
        "fraction_used": min(elapsed / configured, 1.0),
        "deadline_reached": elapsed >= configured,
    }


__all__ = [
    "LIFETIME_AWARENESS_ENV",
    "LIFETIME_STARTED_MONOTONIC_ENV",
    "LIFETIME_TIMEOUT_SECONDS_ENV",
    "LifetimeConfigurationError",
    "agent_lifetime_snapshot",
]
