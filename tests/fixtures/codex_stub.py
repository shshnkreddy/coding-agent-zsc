#!/usr/local/bin/python
"""Deterministic app-server peer for Docker integration tests; no model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

home = Path(os.environ['CODEX_HOME'])
mode = (home / 'mode').read_text()


def emit(message):
    print(json.dumps(message), flush=True)


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return json.loads(result.stdout)


for line in sys.stdin:
    message = json.loads(line)
    with (home / 'requests.jsonl').open('a') as stream:
        stream.write(line)
    method = message['method']
    if method == 'initialize':
        emit({'id': message['id'], 'result': {'userAgent': 'test'}})
    elif method == 'thread/start':
        emit({'id': message['id'], 'result': {'thread': {'id': 'thread-test'}, 'model': 'test-model'}})
    elif method == 'turn/start':
        emit({'id': message['id'], 'result': {'turn': {'id': 'turn-test'}}})
        Path('scratch.py').write_text('# Informal work from this session\n')
        Path('notes').mkdir()
        Path('notes/findings.md').write_text('Research notes retained at shutdown\n')
        if mode == 'startup_error':
            emit({'method': 'turn/completed', 'params': {'threadId': 'thread-test',
                 'turn': {'status': 'failed', 'error': {'message': 'test failure'}}}})
            continue
        if mode == 'idle':
            time.sleep(120)
        if mode == 'crash':
            sys.exit(2)
        if mode == 'interactive':
            emit({'id': 99, 'method': 'item/tool/requestUserInput', 'params': {}})
            time.sleep(120)
        for tokens in (20, 40, 40, 35):
            emit({'method': 'thread/tokenUsage/updated', 'params': {
                'threadId': 'thread-test', 'tokenUsage': {'total': {'totalTokens': tokens}}}})
        if mode == 'tokens':
            time.sleep(120)
        if mode.startswith('cat_dog'):
            source = Path('final_solution.py').read_text()
            flip = int(mode == 'cat_dog_reverse')
            source = source.replace('jnp.argmax(obs.legal_actions).astype(jnp.int32)',
                f'(obs.features[5].astype(jnp.int32) ^ {flip})', 1)
            source = source.replace('jnp.argmax(obs.legal_actions).astype(jnp.int32)',
                f'1 + (obs.features[7].astype(jnp.int32) ^ {flip})', 1)
            Path('final_solution.py').write_text(source)
            command('experiment-ledger','register','--title','test','--hypothesis','opposite conventions',
                    '--planned-change','encode the private pet','--success-criterion','perfect SP')
            assert command('fitness-eval')['fitness'] == 10
            command('experiment-ledger','select','A00','--reason','Validated convention')
        elif mode != 'no_score':
            for index, multiplier in enumerate((2, 3)):
                Path('final_solution.py').write_text(f'def predict(xs): return [{multiplier}*x+1 for x in xs]\n')
                attempt = command('experiment-ledger', 'register', 'final_solution.py',
                    '--title', f'Experiment {index}', '--hypothesis', 'A linear rule works',
                    '--planned-change', 'Change slope', '--success-criterion', 'Zero error')
                result = command('fitness-eval', 'final_solution.py')
                assert result['status'] == 'ok', result
                command('experiment-ledger', 'conclude', '--attempt-id', result['attempt_id'],
                        '--decision', 'keep' if index == 0 else 'reject', '--summary', 'Measured fitness')
                if mode == 'quota':
                    time.sleep(120)
            if mode == 'choose_second':
                command('experiment-ledger', 'select', 'A01', '--reason', 'Prefer this for held-out partners')
            Path('final_solution.py').unlink()
            # The runner must replace a symlink rather than overwrite its target.
            Path('final_solution.py').symlink_to('/tmp/unscored.py')
        emit({'method': 'turn/completed', 'params': {'threadId': 'thread-test',
             'turn': {'status': 'completed', 'error': None}}})
