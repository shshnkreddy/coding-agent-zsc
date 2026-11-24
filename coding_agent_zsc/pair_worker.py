"""Container entry point. Evaluate one ordered pair on a common batch of games."""
import json
from pathlib import Path
import sys

sys.path.insert(0, '/runtime')
from coding_agent_zsc.common import compile_solution, load_game, score


if __name__ == '__main__':
    preset, seed, episodes = sys.argv[1:]
    task = load_game(preset)
    from llm_policy.evaluation.api import run_cross_play
    result = run_cross_play(task=task, policy_agent_0=compile_solution(Path('/input/a.py'), task),
                            policy_agent_1=compile_solution(Path('/input/b.py'), task),
                            root_seed=int(seed), num_episodes=int(episodes))
    print(json.dumps(score(result), allow_nan=False))
