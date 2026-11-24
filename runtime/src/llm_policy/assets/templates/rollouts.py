"""Editable exploratory rollout program for the current candidate policy."""

from __future__ import annotations

import os
from pathlib import Path

# This read-only context API is the required basis for all SP measurements.
# This editable script may add diagnostics, alternate seeds, and trace analysis,
# but only ``policy-eval score`` produces an authoritative performance number.
from context.rollout_api import (
    compile_candidate,
    load_task,
    run_self_play,
    summarize_episode_trace,
    summarize_rollout,
)


TASK_ROOT = Path(os.environ.get("LLM_POLICY_TASK_ROOT", Path(__file__).parents[1]))
TASK_CONFIG = Path(
    os.environ.get(
        "LLM_POLICY_ENV_CONFIG",
        TASK_ROOT / "context/environment/config.yaml",
    )
)
POLICY = TASK_ROOT / "workspace/policy.py"


def main():
    task = load_task(TASK_CONFIG)
    policy = compile_candidate(POLICY, task)
    result = run_self_play(
        task=task,
        policy=policy,
        root_seed=7,
        num_episodes=32,
        capture_traces=True,
    )

    print(summarize_rollout(result))
    # Customize this list or add analyses in rollout_utils.py. For matching an
    # official failure ID, run policy-eval trace on the full 1,024-reset batch.
    for episode_id in (0,):
        print(summarize_episode_trace(result, episode_id))


if __name__ == "__main__":
    main()
