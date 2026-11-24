"""Container entry point. Evaluate one ordered pair on a common batch of games."""
import json
from pathlib import Path
import sys

sys.path.insert(0, '/runtime')
from coding_agent_zsc.common import load_game, score


if __name__ == '__main__':
    preset, seed, episodes = sys.argv[1:]
    task = load_game(preset)
    from llm_policy.evaluation.api import compile_candidate, run_cross_play
    result = run_cross_play(task=task, policy_agent_0=compile_candidate(Path('/input/a/policy.py'), task),
                            policy_agent_1=compile_candidate(Path('/input/b/policy.py'), task),
                            root_seed=int(seed), num_episodes=int(episodes))
    print(json.dumps(score(result), allow_nan=False))
