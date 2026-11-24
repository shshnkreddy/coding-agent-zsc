"""Generate and evaluate six independent policies per environment."""
import argparse
import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .common import PRESETS
from .xp import report, write_json

MAIN = (
    'demo_cook_simple', 'demo_cook_wide', 'grounded_coord_simple', 'grounded_coord_ring',
    'test_time_simple', 'test_time_wide', 'hanabi_2p', 'hanabi_2p_h_group_level_1',
    'yokai_2p_9c_standard', 'yokai_2p_9c', 'yokai_2p_16c',
)
SUITES = {'main': MAIN, 'novel': ('block_grid_2p', 'pipeline_cook_2p'),
          'cat-dog': ('cat_dog',), 'all': tuple(PRESETS)}
SEEDS = tuple(range(6))
# Registry names in the submission's 462 main XP policy records.
PAPER_MODELS = {
    'astra_high': ('codex', 'gpt-6-astra'),
    'gpt55_high': ('codex', 'gpt-5.5'),
    'luna56_high': ('codex', 'gpt-5.6-luna'),
    'sol56_high': ('codex', 'gpt-5.6-sol'),
    'fable51': ('anthropic', 'claude-fable-5-1'),
    'opus_high': ('anthropic', 'claude-opus-5'),
    'sonnet5_high': ('anthropic', 'claude-sonnet-5'),
}


def population(root):
    return [root / f'seed-{seed}' for seed in SEEDS]


def run_environment(preset, root, provider, model, args, selection):
    module = 'coding_agent_zsc.' + PRESETS[preset].path.replace('/', '.')
    task = importlib.import_module(module).p0
    dest = root / preset
    command = [sys.executable, '-m', 'autoastra.autoresearch', module + ':p0',
               '--seeds', *map(str, SEEDS), '--paired-paraphrases', '--jobs', str(args.jobs),
               '--memory-mb', '4096',
               '--selection', selection, '--provider', provider, '--output', str(dest)]
    if model:
        command.extend(['--model', model])
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    config = dict(provider=provider, model=model, selection=selection, image=task.image,
                  budgets=task.run_config, seeds=list(SEEDS), config_sha256=task.task_spec['zsc']['config_sha256'])
    record = root / (preset + '.json')
    if record.exists():
        if json.loads(record.read_text()) != config:
            raise ValueError(f'{record}: configuration changed. Choose a fresh output directory.')
    elif dest.exists():
        raise ValueError(f'{dest}: existing run has no matching experiment configuration.')
    else:
        write_json(record, config)
    if not dest.exists():
        subprocess.run(command, check=True)
    for seed, directory in enumerate(population(dest)):
        result_path = directory / 'result.json'
        if not result_path.is_file():
            raise ValueError(f'{result_path} is missing. Preserve the partial run and retry in a fresh output directory.')
        result = json.loads(result_path.read_text())
        meta = result.get('zsc') or {}
        if (result['seed'] != seed or meta.get('paraphrase') != f'p{seed}' or meta.get('environment') != preset
                or result['status'] == 'failed' or not result.get('selected')):
            raise ValueError(f'{directory}: expected a selected policy for {preset}, seed {seed}, p{seed}.')
    scores = report(population(dest), root / 'reports')
    print(preset, json.dumps(scores[preset]), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--suite', choices=SUITES, default='main')
    group.add_argument('--env', choices=PRESETS, action='append', help='Run one named environment (repeatable)')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', help='Agent model; otherwise use the provider default')
    parser.add_argument('--provider', choices=('codex', 'anthropic', 'opencode', 'openrouter'))
    parser.add_argument('--paper-models', action='store_true', help='Run the seven-model main XP grid')
    parser.add_argument('--jobs', type=int, choices=range(1, 7), default=1, help='Concurrent seeds per environment')
    parser.add_argument('--selection', choices=('best-fitness', 'model'))
    parser.add_argument('--dry-run', action='store_true', help='Print generation commands without starting sessions')
    args = parser.parse_args(argv)
    if args.paper_models and (args.model or args.provider or args.suite != 'main' or args.env):
        parser.error('--paper-models selects the entire main grid and its own providers')
    if args.provider in ('openrouter', 'opencode') and not args.model:
        parser.error('this provider requires --model')
    if args.env and len(set(args.env)) != len(args.env):
        parser.error('each environment may be selected once')
    try:
        if args.paper_models:
            for label, (provider, model) in PAPER_MODELS.items():
                for preset in MAIN:
                    run_environment(preset, args.output / label, provider, model, args, args.selection or 'model')
            if not args.dry_run:
                from .between import compare
                for preset in MAIN:
                    compare({label: args.output / label / preset for label in PAPER_MODELS},
                            preset, args.output / 'between' / preset)
        else:
            for preset in args.env or SUITES[args.suite]:
                run_environment(preset, args.output, args.provider or 'codex', args.model, args,
                                args.selection or 'best-fitness')
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Experiment failed: {exc}\n')
