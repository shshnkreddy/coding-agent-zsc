"""Score archived paper policies or frozen AutoAstra runs, grouped by environment."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

import autoastra
from autoastra.sandbox import SandboxConfig
from autoastra.sandbox.sandbox import _container_options, _docker
from coding_agent_zsc.common import PRESETS, ZSC_ROOT


def _read_run(directory):
    result = json.loads((directory / 'result.json').read_text())
    meta = result.get('zsc') or {}
    preset = meta.get('environment', '')
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError(f'{directory}: unknown ZSC environment')
    if not result.get('solution') or not result.get('selected') or result['status'] == 'failed':
        raise ValueError(f'{directory}: missing valid final selection')
    if (meta['interface'] != 'joint-policy-v1' or meta['config_sha256'] !=
            hashlib.sha256((PRESETS[preset].directory / 'config.yaml').read_bytes()).hexdigest()):
        raise ValueError(f'{directory}: incompatible environment configuration or policy interface')
    digest = result['selected']['source_sha256']
    source = (directory / 'task/workspace/final_solution.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError(f'{directory}: selected policy hash mismatch')
    image = result['image_id']
    row = dict(seed=result['seed'], paraphrase=meta['paraphrase'], sha256=digest,
               image=image, selection=result['selection'])
    return preset, row, source


def _read_paper_policy(directory, manifest, image):
    preset = manifest['environment']
    if manifest['format'] != 'paper-xp-v1' or preset not in PRESETS:
        raise ValueError(f'{directory}: unknown paper population format or environment')
    config = PRESETS[preset].directory / 'config.yaml'
    if hashlib.sha256(config.read_bytes()).hexdigest() != manifest['config_sha256']:
        raise ValueError(f'{directory}: environment configuration differs from paper archive')
    entry = manifest['policies'][directory.name]
    if directory.name != f"seed-{entry['seed']}":
        raise ValueError(f'{directory}: seed differs from paper manifest')
    files = {'policy.py': (directory / 'policy.py').read_bytes()}
    helper = directory / 'custom_primitives.py'
    if helper.exists():
        files[helper.name] = helper.read_bytes()
    digest = hashlib.sha256()
    for name, content in files.items():
        digest.update(name.encode() + b'\0' + content + b'\0')
    if digest.hexdigest() != entry['candidate_sha256']:
        raise ValueError(f'{directory}: paper policy bundle hash mismatch')
    row = dict(seed=entry['seed'], paraphrase=entry['paraphrase'], model=manifest['model'],
               sha256=hashlib.sha256(files['policy.py']).hexdigest(), image=image,
               selection='archived paper selection', archive_id=entry['archive_id'],
               candidate_sha256=entry['candidate_sha256'], context=entry['context'])
    return preset, row, files


def read_population(directories):
    groups = {}
    paper_image = None
    for directory in directories:
        directory = Path(directory).resolve()
        manifest = directory.parent / 'paper_population.json'
        if (directory / 'result.json').is_file():
            preset, row, source = _read_run(directory)
        elif manifest.is_file():
            if paper_image is None:
                tag = os.environ.get('AUTOASTRA_ZSC_IMAGE', 'autoastra-zsc:cpu')
                paper_image = _docker('image', 'inspect', '--format', '{{.Id}}', tag).stdout.strip()
            preset, row, source = _read_paper_policy(directory, json.loads(manifest.read_bytes()), paper_image)
        else:
            raise ValueError(f'{directory}: expected an AutoAstra run or an archived paper seed directory')
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', row['image']):
            raise ValueError(f'{directory}: expected immutable Docker image ID')
        groups.setdefault(preset, []).append((row, source))
    for preset, rows in groups.items():
        rows.sort(key=lambda pair: (pair[0]['seed'], pair[0]['paraphrase']))
        labels = [(row['paraphrase'], row['seed']) for row, _ in rows]
        if len(set(labels)) != len(rows) or len({row['image'] for row, _ in rows}) != 1:
            raise ValueError(f'{preset}: duplicate run labels or incompatible runtime images')
    return groups


def run_cell(preset, image, first, second, seed, episodes, resources, timeout):
    framework = Path(autoastra.__file__).resolve().parent
    name = 'autoastra-xp-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp).resolve()
        for seat, source in [('a', first), ('b', second)]:
            folder = directory / seat
            folder.mkdir()
            files = {'policy.py': source} if isinstance(source, bytes) else source
            (folder / 'policy.py').write_bytes(files['policy.py'])
            if 'custom_primitives.py' in files:
                (folder / 'custom_primitives.py').write_bytes(files['custom_primitives.py'])
        try:
            run = _docker('run', '--rm', '--name', name, '--label', 'autoastra.managed=true',
                          *_container_options(resources, '/usr/local/bin:/usr/bin:/bin'),
                          '--mount', f'type=bind,src={directory},dst=/input,readonly',
                          '--mount', f'type=bind,src={ZSC_ROOT},dst=/runtime/coding_agent_zsc,readonly',
                          '--mount', f'type=bind,src={framework},dst=/runtime/autoastra,readonly',
                          '--entrypoint', resources.python_executable, image, '-I',
                          '/runtime/coding_agent_zsc/pair_worker.py', preset, str(seed), str(episodes), timeout=timeout)
            value = float(json.loads(run.stdout.splitlines()[-1]))
            if not math.isfinite(value):
                raise ValueError('Nonfinite cross-play return')
            return value
        finally:
            _docker('rm', '-f', name, check=False)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def report(directories, output, *, seed=2000003, episodes=1024, timeout=600, resources=None):
    from matplotlib.figure import Figure
    if not directories or min(episodes, timeout) <= 0 or not 0 <= seed < 2**32:
        raise ValueError('Supply run directories, positive episodes and timeout, and a uint32 held-out seed')
    resources = resources or SandboxConfig(cpus=2, memory_mb=4096, pids_limit=256)
    summaries = {}
    for preset, rows in read_population(directories).items():
        dest = Path(output) / preset
        dest.mkdir(exist_ok=True, parents=True)
        with (dest / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            policies = [row for row, _ in rows]
            if seed in {row['seed'] for row in policies}:
                raise ValueError('Use a held-out seed different from the development seeds')
            evaluator = hashlib.sha256()
            for source in (ZSC_ROOT / 'common.py', ZSC_ROOT / 'pair_worker.py', PRESETS[preset].directory / 'config.yaml'):
                evaluator.update(source.read_bytes())
            identity = dict(environment=preset, policies=policies, episodes=episodes, root_seed=seed,
                            evaluator_sha256=evaluator.hexdigest(), gpu=resources.gpu_arguments())
            manifest = dest / 'population.json'
            if manifest.exists() and json.loads(manifest.read_text()) != identity:
                raise ValueError(f'{dest}: inputs differ from the existing evaluation')
            write_json(manifest, identity)
            labels = [f"{p['paraphrase']}/s{p['seed']}" for p in policies]
            matrix = []
            for i, (row, first) in enumerate(rows):
                values = []
                for j, (_, second) in enumerate(rows):
                    cell = dest / f'cell-{i}-{j}.json'
                    if not cell.exists():
                        value = run_cell(preset, row['image'], first, second, seed, episodes, resources, timeout)
                        write_json(cell, value)
                    values.append(json.loads(cell.read_text()))
                matrix.append(values)
            n = len(rows)
            sp = sum(matrix[i][i] for i in range(n)) / n
            ixp = (sum(matrix[i][j] for i in range(n) for j in range(n) if i != j)
                   / (n * (n - 1))) if n > 1 else None
            with (dest / 'xp.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['seat 0 / seat 1', *labels])
                writer.writerows([label, *row] for label, row in zip(labels, matrix))
            fig = Figure(figsize=(max(5, n * .6), max(4, n * .6)), layout='constrained')
            ax = fig.subplots()
            fig.colorbar(ax.imshow(matrix), ax=ax, label='Mean team return')
            ax.set(xticks=range(n), yticks=range(n), xticklabels=labels, yticklabels=labels,
                   xlabel='Seat 1', ylabel='Seat 0', title=PRESETS[preset].title)
            ax.tick_params(axis='x', labelrotation=60)
            fig.savefig(dest / 'xp.png')
            summaries[preset] = dict(SP=sp, IXP=ixp)
            write_json(dest / 'summary.json', {**identity, **summaries[preset], 'labels': labels, 'matrix': matrix})
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, default=1024)
    parser.add_argument('--eval-seed', type=int, default=2000003)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--memory-mb', type=int, default=4096)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.directories, args.output, seed=args.eval_seed, episodes=args.episodes,
                            timeout=args.timeout, resources=SandboxConfig(cpus=2, memory_mb=args.memory_mb,
                            pids_limit=256, gpus=args.gpus)), indent=2))


if __name__ == '__main__':
    main()
