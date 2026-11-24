"""Regenerate the Markdown task previews for the paper's main experiment grid."""
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

from autoastra.preview import create_preview, print_preview
from coding_agent_zsc.common import PRESETS
from coding_agent_zsc.experiment import MAIN, SEEDS


def main():
    output = Path(__file__).resolve().parents[1] / 'docs' / 'previews'
    for env in MAIN:
        module = import_module('coding_agent_zsc.' + PRESETS[env].path.replace('/', '.'))
        for seed in SEEDS:
            variant = f'p{seed}'
            task = getattr(module, variant)
            config = dict(selection='model', seed=seed, provider='codex')
            dest = output / env / 'codex' / f'{variant}.md'
            dest.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=dest.parent) as tmp:
                layout = create_preview(task, Path(tmp) / f'{env}-{variant}', run_config=config)
                print_preview(layout, Path(tmp) / 'preview.md').replace(dest)


if __name__ == '__main__':
    main()
