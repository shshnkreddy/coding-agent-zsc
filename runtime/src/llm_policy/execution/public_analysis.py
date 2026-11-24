"""Source-independent summaries of observable rollout results."""

from __future__ import annotations

from collections import Counter
import dataclasses
from typing import Any
import jax
import numpy as np

def summarize_return_batch(returns, captured, episode_metrics=None) -> dict[str, Any]:
    """Return compact host-side statistics for one episode-return batch."""
    returns = np.asarray(jax.device_get(returns), dtype=np.float64)
    captured = np.asarray(jax.device_get(captured), dtype=bool)
    if returns.ndim != 1 or captured.shape != returns.shape:
        raise ValueError("rollout returns and capture mask must be 1-D peers")
    histogram = Counter(float(value) for value in returns.tolist())
    return {
        **summarize_episode_metrics(episode_metrics, captured),
        "num_episodes": int(returns.size),
        "mean_return": float(returns.mean()) if returns.size else 0.0,
        # ``captured`` only says that the logging wrapper emitted an episode-end
        # return within the requested rollout.  It does not imply that the task
        # was solved or that even one delivery occurred.
        "episode_capture_rate": float(captured.mean()) if captured.size else 0.0,
        "episodes_captured": int(captured.sum()),
        "minimum_return": float(returns.min()) if returns.size else 0.0,
        "maximum_return": float(returns.max()) if returns.size else 0.0,
        "std_return": float(returns.std()) if returns.size else 0.0,
        "quantiles": {
            "q05": float(np.quantile(returns, 0.05)) if returns.size else 0.0,
            "q25": float(np.quantile(returns, 0.25)) if returns.size else 0.0,
            "q50": float(np.quantile(returns, 0.50)) if returns.size else 0.0,
            "q75": float(np.quantile(returns, 0.75)) if returns.size else 0.0,
            "q95": float(np.quantile(returns, 0.95)) if returns.size else 0.0,
        },
        "return_histogram": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
    }


def summarize_episode_metrics(metrics, captured):
    """Aggregate terminal-only Yokai diagnostics; empty for existing adapters."""
    if not metrics:
        return {}
    mask = np.asarray(jax.device_get(captured), dtype=bool)
    arrays = {name: np.asarray(jax.device_get(value)) for name, value in metrics.items()}
    if any(value.shape != mask.shape for value in arrays.values()):
        raise ValueError("episode metrics must match the return batch shape")
    arrays = {name: value[mask] for name, value in arrays.items()}
    if not mask.any():
        return {"episode_metrics": {"num_episodes": 0}}
    wins = arrays["success"].astype(bool)
    early = arrays["ended_early"].astype(bool)
    return {"episode_metrics": {
        "num_episodes": int(mask.sum()),
        "mean_upstream_sparse_return": float(arrays["upstream_sparse_return"].mean()),
        "mean_terminal_penalty_return": float(arrays["terminal_penalty_return"].mean()),
        "win_rate": float(wins.mean()),
        "successful_early_ending_rate": float((wins & early).mean()),
        "early_ending_rate": float(early.mean()),
        "win_rate_given_early_ending": float(wins[early].mean()) if early.any() else None,
        "early_endings": int(early.sum()),
        "successful_early_endings": int((wins & early).sum()),
        "mean_episode_length": float(arrays["episode_length"].mean()),
    }}


def summarize_rollout(result: ProgramRolloutResult) -> dict[str, Any]:
    """Return compact host-side statistics for an SP rollout batch."""
    return summarize_return_batch(result.first_returns, result.captured, result.episode_metrics)


def trace_arrays(trace, *, episode_index=None):
    """Copy policy-chosen fields (or a legacy record) to host arrays."""
    if dataclasses.is_dataclass(trace):
        fields = {field.name: getattr(trace, field.name) for field in dataclasses.fields(trace)}
    elif isinstance(trace, dict):
        fields = trace
    else:
        # Unknown dataclass records decoded by the source-free client.
        fields = vars(trace)
    return {name: np.asarray(jax.device_get(
        value if episode_index is None else value[:, episode_index]
    )) for name, value in fields.items()}


def _episode_arrays(result, seat, index):
    fields = trace_arrays(getattr(result, f"trace_agent_{seat}"), episode_index=index)
    actions = getattr(result, f"actions_agent_{seat}", None)
    if actions is not None:
        actions = np.asarray(jax.device_get(actions[:, index]))
    return fields, actions


def _seat_indices(result):
    seats = [0, 1]
    for seat in range(2, 5):
        if getattr(result, f"trace_agent_{seat}", None) is not None:
            seats.append(seat)
    return tuple(seats)


def select_episode(result: ProgramRolloutResult, episode_index: int) -> dict[str, Any]:
    """Copy one episode's policy diagnostics and actual actions to the host."""
    index = int(episode_index)
    returns = np.asarray(jax.device_get(result.first_returns))
    if index < 0 or index >= int(returns.shape[0]):
        raise IndexError(f"episode index {index} is outside [0, {returns.shape[0]})")
    output = {
        "episode": index,
        "return": float(returns[index]),
        **({"episode_metrics": {name: float(np.asarray(jax.device_get(value))[index])
                               for name, value in result.episode_metrics.items()}} if result.episode_metrics else {}),
        "episode_captured": bool(np.asarray(jax.device_get(result.captured))[index]),
    }
    for seat in _seat_indices(result):
        fields, actions = _episode_arrays(result, seat, index)
        output[f"agent_{seat}"] = {name: value.tolist() for name, value in fields.items()}
        if actions is not None:
            output[f"actions_agent_{seat}"] = actions.tolist()
    return output


def summarize_episode_trace(
    result: ProgramRolloutResult,
    episode_index: int,
    *,
    max_segments_per_agent: int = 32,
) -> dict[str, Any]:
    """Bounded segments of identical diagnostics and actual actions.

    Field names and values are opaque: no strategy or status meanings are
    inferred. Preserve both ends when the number of segments exceeds the cap.
    """
    index = int(episode_index)
    max_segments = int(max_segments_per_agent)
    if max_segments < 2:
        raise ValueError("max_segments_per_agent must be at least 2")
    returns = np.asarray(jax.device_get(result.first_returns))
    if index < 0 or index >= int(returns.shape[0]):
        raise IndexError(f"episode index {index} is outside [0, {returns.shape[0]})")
    output = {
        "episode": index,
        "return": float(returns[index]),
        **({"episode_metrics": {name: float(np.asarray(jax.device_get(value))[index])
                               for name, value in result.episode_metrics.items()}} if result.episode_metrics else {}),
        "episode_captured": bool(np.asarray(jax.device_get(result.captured))[index]),
    }

    for seat in _seat_indices(result):
        fields, actions = _episode_arrays(result, seat, index)
        length = (len(actions) if actions is not None else
                  len(next(iter(fields.values()))) if fields else 0)
        comparison = list(fields.values()) + ([] if actions is None else [actions])
        segments = []
        if length:
            start = 0
            for step in range(1, length + 1):
                if step < length and all(
                    np.array_equal(value[step], value[start], equal_nan=True)
                    for value in comparison
                ):
                    continue
                segment = {"start_step": start, "end_step": step - 1,
                           "length": step - start}
                values = {name: value[start].tolist() for name, value in fields.items()}
                if isinstance(getattr(result, f"trace_agent_{seat}"), dict):
                    # User keys cannot overwrite segment metadata or actual action.
                    segment["trace"] = values
                else:
                    segment.update(values)
                if actions is not None:
                    segment["action"] = int(actions[start])
                segments.append(segment)
                start = step
        total_segments = len(segments)
        if total_segments > max_segments:
            tail_count = max(1, max_segments // 4)
            segments = segments[:max_segments - tail_count] + segments[-tail_count:]
        output[f"agent_{seat}"] = {
            "num_steps": length,
            "total_segments": total_segments,
            "omitted_middle_segments": max(0, total_segments - max_segments),
            "segments": segments,
        }
        output.pop(f"actions_agent_{seat}", None)
    return output
