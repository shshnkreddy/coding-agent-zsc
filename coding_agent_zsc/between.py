"""Evaluate ordered model pairs on the same held-out batch, excluding equal seeds."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path

from autoastra.sandbox import SandboxConfig
from .common import PRESETS, ZSC_ROOT
from .experiment import SEEDS, population
from .xp import read_population, run_cell, write_json


def compare(populations, preset, output, *, seed=2000003, episodes=1024, timeout=600):
    if not populations or not 0 <= seed < 2**32 or seed in SEEDS or min(episodes, timeout) < 1:
        raise ValueError('Supply populations, a held-out seed, and positive episodes and timeout.')
    rows = {}
    for label, directory in populations.items():
        groups = read_population(population(directory))
        if set(groups) != {preset}:
            raise ValueError(f'{directory}: expected only {preset}')
        rows[label] = groups[preset]
        labels = [(p['seed'], p['paraphrase']) for p, _ in rows[label]]
        if labels != [(s, f'p{s}') for s in SEEDS]:
            raise ValueError(f'{directory}: expected one policy per paired seed and paraphrase')
    images = {p['image'] for lane in rows.values() for p, _ in lane}
    if len(images) != 1:
        raise ValueError('All populations must share the same runtime image')
    image = images.pop()
    resources = SandboxConfig(cpus=2, memory_mb=4096, pids_limit=256)
    output.mkdir(exist_ok=True, parents=True)
    labels = list(populations)
    fingerprint = hashlib.sha256()
    for path in (Path(__file__), ZSC_ROOT / 'xp.py', ZSC_ROOT / 'common.py', ZSC_ROOT / 'pair_worker.py'):
        fingerprint.update(path.read_bytes())
    identity = dict(environment=preset, labels=labels, episodes=episodes, seed=seed,
                    evaluator_sha256=fingerprint.hexdigest(),
                    config_sha256=hashlib.sha256((PRESETS[preset].directory / 'config.yaml').read_bytes()).hexdigest(),
                    populations={k: [p for p, _ in lane] for k, lane in rows.items()})
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / 'population.json'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError(f'{output}: inputs changed. Use a new report directory.')
        write_json(manifest, identity)
        matrix = []
        for i, first in enumerate(labels):
            values = []
            for j, second in enumerate(labels):
                scores = []
                for a, (_, policy_a) in enumerate(rows[first]):
                    for b, (_, policy_b) in enumerate(rows[second]):
                        if a == b:
                            continue
                        cell = output / f'pair-{i}-{j}-{a}-{b}.json'
                        if not cell.exists():
                            score = run_cell(preset, image, policy_a, policy_b, seed, episodes, resources, timeout)
                            write_json(cell, score)
                        scores.append(json.loads(cell.read_text()))
                values.append(sum(scores) / len(scores))
            matrix.append(values)
        with (output / 'between.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['seat 0 / seat 1', *labels])
            writer.writerows([label, *row] for label, row in zip(labels, matrix))
        write_json(output / 'summary.json', {**identity, 'matrix': matrix, 'pairs_per_cell': 30})
    return matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', required=True, choices=PRESETS)
    parser.add_argument('--population', action='append', required=True, metavar='LABEL=DIRECTORY')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    populations = {}
    for item in args.population:
        label, sep, directory = item.partition('=')
        if not label or not sep or not directory or label in populations:
            parser.error('Supply each population as a unique LABEL=DIRECTORY')
        populations[label] = Path(directory)
    print(json.dumps(compare(populations, args.env, args.output)))
