import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coding_agent_zsc.between import compare
from coding_agent_zsc.experiment import main, MAIN, PAPER_MODELS, SEEDS


class ExperimentTests(unittest.TestCase):
    def test_paper_grid_is_paired_and_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as log:
            target = Path(tmp) / 'runs'
            main(['--output', str(target), '--paper-models', '--dry-run'])
            lines = log.getvalue().splitlines()
            self.assertEqual(len(lines), len(MAIN) * len(PAPER_MODELS))
            self.assertEqual(len(lines), 77)
            for line in lines:
                self.assertIn('--seeds 0 1 2 3 4 5 --paired-paraphrases', line)
                self.assertIn('--selection model', line)
            self.assertFalse(target.exists())

    def test_directed_between_model_means_and_resume(self):
        def load(directories):
            role = int(directories[0].parent.name)
            return {'cat_dog': [(dict(image='same', seed=s, paraphrase=f'p{s}'), bytes([role, s]))
                                for s in SEEDS]}

        def score(preset, image, policy_a, policy_b, *rest):
            self.assertNotEqual(policy_a[1], policy_b[1])
            return 10.0 * policy_a[0] + policy_b[0]

        with tempfile.TemporaryDirectory() as tmp, patch('coding_agent_zsc.between.read_population', side_effect=load), \
                patch('coding_agent_zsc.between.run_cell', side_effect=score) as evaluate:
            root = Path(tmp)
            models = {'one': root/'1', 'two': root/'2'}
            expected = [[11, 12], [21, 22]]
            self.assertEqual(compare(models, 'cat_dog', root/'reports'), expected)
            self.assertEqual(evaluate.call_count, 120)
            self.assertEqual(compare(models, 'cat_dog', root/'reports'), expected)
            self.assertEqual(evaluate.call_count, 120)
            self.assertEqual(json.loads((root/'reports/summary.json').read_text())['pairs_per_cell'], 30)
