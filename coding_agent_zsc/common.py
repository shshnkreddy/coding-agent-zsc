"""Build task objects without importing the JAX runtime on the controller."""
from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from autoastra.task import Task

ZSC_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Preset:
    path: str
    title: str

    @property
    def directory(self):
        return ZSC_ROOT / self.path


# Keys are the environment IDs stored in result.json across repository layouts.
PRESETS = {
    'block_grid_2p': Preset('ocv2/block_grid_2p', 'Overcooked V2 — Two-Player Block Grid'),
    'cat_dog': Preset('cat_dog', 'Cat and Dog'),
    'demo_cook_simple': Preset('ocv2/demo_cook_simple', 'Overcooked V2 — Demo Cook Simple'),
    'demo_cook_wide': Preset('ocv2/demo_cook_wide', 'Overcooked V2 — Demo Cook Wide'),
    'grounded_coord_ring': Preset('ocv2/grounded_coord_ring', 'Overcooked V2 — Grounded Coordination Ring'),
    'grounded_coord_simple': Preset('ocv2/grounded_coord_simple', 'Overcooked V2 — Grounded Coordination Simple'),
    'hanabi_2p': Preset('hanabi/two_player', 'Two-Player Hanabi'),
    'hanabi_2p_h_group_level_1': Preset('hanabi/two_player_h_group_level_1', 'Two-Player Hanabi (H-Group Level 1)'),
    'pipeline_cook_2p': Preset('ocv2/pipeline_cook_2p', 'Overcooked V2 — Two-Player Pipeline Cook'),
    'test_time_simple': Preset('ocv2/test_time_simple', 'Overcooked V2 — Test Time Simple'),
    'test_time_wide': Preset('ocv2/test_time_wide', 'Overcooked V2 — Test Time Wide'),
    'yokai_2p_16c': Preset('yokai/sixteen_card_revealed_history', 'Two-Player Yōkai (16 Cards, Revealed Color History)'),
    'yokai_2p_9c': Preset('yokai/nine_card_revealed_history', 'Two-Player Yōkai (9 Cards, Revealed Color History)'),
    'yokai_2p_9c_standard': Preset('yokai/nine_card_standard', 'Two-Player Yōkai (9 Cards, Standard)'),
}


def make_variants(preset, evaluate):
    root = PRESETS[preset].directory
    run_config = json.loads((ZSC_ROOT / 'run_config.json').read_text())
    task_config = root / 'run_config.json'
    if task_config.is_file():
        run_config.update(json.loads(task_config.read_text()))
    image = os.environ.get('AUTOASTRA_ZSC_IMAGE', 'autoastra-zsc:cpu')
    common = {'policy_api.md': (root / 'policy_api.md').read_text()}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if path.name in ('config.yaml', 'layout.md', 'specification.json'):
            common['environment/' + path.name] = path.read_text()
        elif 'conventions' in path.relative_to(root).parts:
            common[str(path.relative_to(root))] = path.read_text()
    for index in range(6):
        variant = f'p{index}'
        wording = root / 'paraphrases' / variant
        spec = dict(name=f'{preset}_{variant}', title=PRESETS[preset].title,
                    description=(wording / 'objective.md').read_text().strip(),
                    solution_interface='Implement the functions in final_solution.py using context/policy_api.md. Read context/environment/README.md for the rules.',
                    metric='development_self_play_return', direction='maximize',
                    metric_label='Development Self-Play Return',
                    development_proxy=(
                        'Aim to maximise cross-play return with independently generated partners. '
                        'During development, `fitness-eval` measures only self-play return, '
                        'the mean team return over 1,024 episodes with your policy in both seats. '
                        'A higher self-play score alone cannot tell you whether a change improves cross-play. '
                        'Cross-play scores become available after the independent runs finish, '
                        'when the XP wrapper evaluates their frozen policies together on held-out episodes.'
                    ),
                    zsc={'environment': preset, 'paraphrase': variant, 'interface': 'joint-policy-v1',
                         'config_sha256': hashlib.sha256((root / 'config.yaml').read_bytes()).hexdigest()})
        yield Task(spec, evaluate, image, context_files={**common, 'environment/README.md':
                   (wording / 'rules.md').read_text()}, starter=(root / 'starter.py').read_text(),
                   run_config=dict(run_config))


def load_game(preset):
    # XLA sizes native thread pools from CPU affinity. Docker CPU quotas alone
    # expose every host core and can exhaust the evaluation container PID limit.
    cores = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cores[:max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))])
    from dataclasses import replace
    from llm_policy.evaluation.api import load_task
    return replace(load_task(PRESETS[preset].directory / 'config.yaml'), primitive_mode='none',
                   restrict_introspection=True)


def compile_solution(path, game):
    """Present a frozen AutoAstra source file in the upstream bundle format."""
    from llm_policy.evaluation.api import compile_candidate
    with tempfile.TemporaryDirectory() as tmp:
        policy = Path(tmp) / 'policy.py'
        shutil.copyfile(path, policy)
        return compile_candidate(policy, game)


def score(result):
    import numpy as np
    returns = np.asarray(result.first_returns)
    if not np.asarray(result.captured).all() or not np.isfinite(returns).all():
        raise ValueError('Incomplete episode or invalid return')
    return float(returns.mean())


def self_play(preset, path, *, split, seed):
    if split != 'dev':
        raise ValueError('Use the population wrapper for held-out evaluation')
    game = load_game(preset)
    from llm_policy.evaluation.api import run_self_play
    policy = compile_solution(path, game)
    return score(run_self_play(task=game, policy=policy, root_seed=seed, num_episodes=1024))
