import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from autoastra.autoresearch import run
from autoastra.sandbox import SandboxConfig
from coding_agent_zsc.common import PRESETS


class ZscTests(unittest.TestCase):
    def test_presets_have_six_distinct_wordings(self):
        self.assertEqual(len(PRESETS), 14)
        for preset, info in PRESETS.items():
            module = importlib.import_module('coding_agent_zsc.' + info.path.replace('/', '.'))
            self.assertEqual(module.task.task_spec['zsc']['environment'], preset)
            variants = [getattr(module, f'p{i}') for i in range(6)]
            self.assertEqual(len({v.task_spec['description'] for v in variants}), 6, preset)
            self.assertEqual(len({v.task_spec['zsc']['config_sha256'] for v in variants}), 1)

    def test_agent_sees_cross_play_objective_and_development_proxy(self):
        from autoastra.task.layout import _agents_markdown, _task_markdown
        from coding_agent_zsc import cat_dog

        for task in (cat_dog.p0, cat_dog.p5):
            spec = task.task_spec
            document = _task_markdown(spec)
            self.assertTrue(document.startswith('# Cat and Dog\n'))
            self.assertNotIn('## ZSC', document)
            self.assertNotIn(spec['zsc']['config_sha256'], document)
            self.assertIn('## Development Self-Play Return', document)
            self.assertNotIn('## Fitness metric', document)
            self.assertNotIn('Maximise the fitness value', document)
            for selection in ('best-fitness', 'model'):
                instructions = _agents_markdown(spec, selection)
                self.assertIn('maximise cross-play return', instructions)
                self.assertIn('measures only self-play return', instructions)
                self.assertNotIn('goal is to maximise the value returned', instructions)

    def test_seed_batch_and_paired_tasks(self):
        from autoastra.autoresearch.__main__ import main
        for paired in (False, True):
            with tempfile.TemporaryDirectory() as tmp, patch('autoastra.autoresearch.batch.run') as run_one:
                run_one.return_value = dict(sandbox='test', solution='test', status='completed',
                                            stop_reason='completed', error=None)
                args = ['coding_agent_zsc.cat_dog:p2', '--output', str(Path(tmp)/'batch'),
                        '--seeds', '9', '8', '7', '6', '5', '4']
                self.assertEqual(main(args + (['--paired-paraphrases'] if paired else [])), 0)
                for i, call in enumerate(run_one.call_args_list):
                    self.assertEqual(call.args[0].task_spec['zsc']['paraphrase'], f'p{i}' if paired else 'p2')
                    self.assertEqual(call.kwargs['run_config']['seed'], 9-i)
                    self.assertEqual(call.args[1].name, f'seed-{9-i}')

    @unittest.skipUnless(os.environ.get('AUTOASTRA_ZSC_IMAGE'), 'requires the ZSC image')
    def test_opposite_conventions(self):
        from coding_agent_zsc import cat_dog
        from coding_agent_zsc.xp import report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'codex'
            shutil.copyfile(Path(__file__).parent / 'fixtures/codex_stub.py', binary)
            binary.chmod(0o755)
            try:
                for i, task in enumerate((cat_dog.p0, cat_dog.p1)):
                    home = root / f'home-{i}'
                    home.mkdir()
                    (home / 'mode').write_text(('cat_dog', 'cat_dog_reverse')[i])
                    with patch('autoastra.autoresearch.runner._agent_paths', return_value=(binary, home)):
                        result = run(task, root / str(i), run_config={'selection': 'model', 'seed': i,
                                     'evaluation_timeout_seconds': 120})
                    self.assertEqual(result['status'], 'completed', (result, (root/str(i)/'codex.stderr.log').read_text()))
                    self.assertEqual(result['selected']['fitness'], 10)
                report([root/'1', root/'0'], root/'report', episodes=16)
                data = json.loads((root/'report/cat_dog/summary.json').read_text())
                self.assertEqual(data['matrix'], [[10,-10],[-10,10]])
                self.assertEqual((data['SP'],data['IXP']), (10,-10))
            finally:
                for directory, _, _ in os.walk(root):
                    Path(directory).chmod(0o700)
