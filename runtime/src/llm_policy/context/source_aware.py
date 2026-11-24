"""Factual interface and two-stage protocol for source-aware Cat & Dog."""
import ast
import json

from llm_policy.context.minimal import starter, policy_api
from llm_policy.policies.bundle import SOURCE_LOCATOR, copy_bundle


SETTING = '''# Open-source zero-shot coordination

You will be evaluated in an open-source zero-shot coordination (ZSC) setting.

The aim is better coordination: give each player an exact model of its partner's
decision rules by sharing the partner's policy code. A player can then adapt to
that code. Private observations, memory and random choices remain hidden, so
knowing the code does not guarantee knowing the next action.

There are two stages:

1. **Develop policies independently.** Each model writes policies for Alice and
   Bob without seeing other submissions. It can test and improve its own pair.
2. **Exchange code and play.** Once all submissions are final, the runner pairs
   Alice from one submission with Bob from another and gives each player the
   other's code. The policies then play without further model calls or edits.
   Every pairing is evaluated, including each submission paired with itself.
'''

DETAILS = '''## What to submit

Write Alice in `agent_0.py` and Bob in `agent_1.py`, each with its own metadata,
initializer, action function and helpers. Do not import the other role or helper
files. Leave `policy.py` unchanged; tools use it to identify your submission.

Each action function receives `(obs, reset, memory, rng, partner_policy_source)`.
The last argument is the partner's complete role file as a string, or `None`.
Alice receives Bob's file; Bob receives Alice's. It stays fixed for the matchup.

You may read the string and parse it with `ast`, but may not execute or simulate
it, call models, access files or inspect the simulator. Source handling happens
during JAX compilation. See `policy_api.md` for the JAX and interface rules.

## Development and fallback

Use the normal registration, validation, scoring and trace tools. Development
self-play shares your own opposite-role file. Test without source using
`policy-eval score policy.py --partner-source null` (also available for `trace`
and as `partner_source="null"` in the rollout client).

Both roles must work with `None` and choose legal actions. The runner checks
this before accepting a submission. If a role fails to compile or run with
partner source, or chooses an illegal active action, the runner retries the
whole matchup with that role receiving `None`. It resets both memories and
reuses the same episode seeds. Each role gets one fallback; failure with `None`
or an infrastructure failure remains an evaluation error. Inactive-role actions
are ignored.

Only the completed run counts toward the score. Reports save the source hashes
and any fallback reasons. The game's rules, rewards and XP scoring are unchanged.
'''

PROTOCOL = SETTING + '\n' + DETAILS


def install(task_root, contract):
    context = task_root / 'context'
    workspace = task_root / 'workspace'
    baseline = context / 'starting_policy'
    baseline.mkdir()
    (baseline / 'policy.py').write_text(SOURCE_LOCATOR)
    tree = ast.parse(starter(contract))
    for role in range(2):
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name not in (f'init_agent_{role}', f'agent_{role}_act'):
                    continue
                if node.name.endswith('_act'):
                    node.args.args.append(ast.arg(arg='partner_policy_source'))
            nodes.append(node)
        (baseline / f'agent_{role}.py').write_text(ast.unparse(ast.Module(body=nodes, type_ignores=[])) + '\n')
    (context / 'starting_policy.py').unlink()
    api = policy_api(contract).replace('You may write your own\ncustom_primitives.py.',
                                      'Keep helpers inline in each self-contained role file.')
    api = api.replace('obs, reset, memory, rng)', 'obs, reset, memory, rng, partner_policy_source)')
    api = api.replace('Use only local observations and the supplied PRNG, never hidden state or reward.',
                      'Use local observations, the supplied PRNG and partner source; never hidden state or reward.')
    api += '\nThe final argument is a static `str | None`; it is not a JAX array.\n'
    api += 'Each role file contains only its own initializer and action function.\n'
    api += 'The `ast` module is additionally allowed for parsing, never executing, source.\n'
    (context / 'policy_api.md').write_text(api)
    (context / 'source_exchange.md').write_text(PROTOCOL)
    environment = context / 'environment/README.md'
    environment.write_text(environment.read_text().replace(
        'Policies receive only their own observation, reset,\nprivate memory and PRNG;',
        'Policies receive their own observation, reset,\nprivate memory, PRNG and frozen opposite-role source;'))
    (context / 'generation.json').write_text(json.dumps({'mode': 'open_source_zsc', 'version': 1}) + '\n')
    agents = task_root / 'AGENTS.md'
    # Keep the existing ledger/tool instructions, replacing single-file authoring language.
    text = agents.read_text().replace(
        'Keep `POLICY_API_VERSION`, all required `POLICY_METADATA` keys, both\n'
        'initializers, and both act functions in `policy.py`.',
        'Keep `POLICY_API_VERSION`, all required `POLICY_METADATA` keys, the role\n'
        'initializer and action function in each of `agent_0.py` and `agent_1.py`.')
    text = text.replace(
        'Each act function sees only its local observation, reset flag, private\n'
        'fixed-shape memory, and PRNG key,',
        'Each act function receives its local observation, reset flag, private\n'
        'fixed-shape memory, PRNG key and static opposite-role source (or None),')
    text = text.replace(
        'identifies the exact `policy.py` and optional `custom_primitives.py` bundle,\n'
        'not `policy.py` alone.',
        'identifies the exact `agent_0.py` and `agent_1.py` files together with\n'
        'the fixed runner-owned `policy.py` locator.')
    text = text.replace(
        'It restores both files exactly (including absence of\n'
        'an optional custom-primitives file),',
        'It restores both role files and the fixed locator exactly,')
    text = text.replace(
        'For SP, the runner still selects by independently recomputed development score;\n'
        'your logged preference does not override its selection.\n', '')
    text = text.replace('Leave in `policy.py` the candidate',
                        'Leave in `agent_0.py` and `agent_1.py` the candidate')
    text += '\n\nRead context/source_exchange.md for the two-stage protocol before editing.\n'
    agents.write_text(text)
    task_spec = context / 'task_spec.md'
    text = task_spec.read_text().replace('Leave in policy.py the policy',
                                         'Leave in agent_0.py and agent_1.py the policy')
    text = text.replace(
        'Runtime seats receive only their own observation, reset flag,\n'
        'private memory and PRNG.',
        'Runtime seats receive their own observation, reset flag,\n'
        'private memory, PRNG and static opposite-role source (or None).')
    heading, body = text.split('\n\n', 1)
    setting = SETTING.replace('# Open-source', '## Open-source', 1)
    task_spec.write_text(heading + '\n\n' + setting + '\n' + body.rstrip()
                         + '\n\n' + DETAILS)
    copy_bundle(baseline, workspace)
    return baseline / 'policy.py'
