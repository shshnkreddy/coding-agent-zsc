"""The ZSC launcher must explicitly request provider-only networking."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from coding_agent_zsc.experiment import required_model_hosts, run_environment


class NetworkDefaultsTests(unittest.TestCase):
    def test_dry_run_explicitly_allows_only_selected_provider(self):
        for provider, model, expected in (
            ('codex', 'gpt-6.1-sol', ['api.openai.com', 'chatgpt.com', 'auth.openai.com']),
            ('openrouter', 'test/model', ['openrouter.ai']),
            ('opencode', 'openai/test', ['api.openai.com']),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'not-created'
                output = io.StringIO()
                with contextlib.redirect_stdout(output), patch('subprocess.run') as launch:
                    run_environment('cat_dog', root, provider, model,
                                    argparse.Namespace(jobs=1, dry_run=True), 'best-fitness')
                argv = shlex.split(output.getvalue())
                self.assertEqual([argv[i+1] for i, x in enumerate(argv) if x == '--model-host'], expected)
                self.assertNotIn('--allow-internet', argv)
                self.assertFalse(root.exists())
                launch.assert_not_called()

    def test_legacy_autoastra_fails_before_launch_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'not-created'
            with patch.dict(sys.modules, {'autoastra.autoresearch.network': None}), \
                 patch('subprocess.run') as launch:
                with self.assertRaisesRegex(RuntimeError, 'provider-only networking'):
                    run_environment('cat_dog', root, 'codex', 'gpt-6.1-sol',
                                    argparse.Namespace(jobs=1, dry_run=False), 'best-fitness')
            launch.assert_not_called()
            self.assertFalse(root.exists())

    def test_policy_recorded_before_launch(self):
        class LaunchObserved(Exception):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('subprocess.run', side_effect=LaunchObserved), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(LaunchObserved):
                    run_environment('cat_dog', root, 'codex', 'gpt-6.1-sol',
                                    argparse.Namespace(jobs=1, dry_run=False), 'best-fitness')
            record = json.loads((root/'cat_dog.json').read_text())
            self.assertEqual(record['network'], {
                'mode': 'model-provider-only',
                'allowed_hosts': list(required_model_hosts('codex', 'gpt-6.1-sol'))})


if __name__ == '__main__':
    unittest.main()
