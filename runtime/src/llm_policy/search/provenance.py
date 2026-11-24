"""Immutable attempt snapshots and append-only evaluation provenance.

This module captures bytes before importing policy code, including invalid code.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import uuid

from llm_policy.search import ledger as L
from llm_policy.policies.bundle import ALL_FILES, SOURCE_LOCATOR

ACTIVE_EVALUATION = ContextVar('policy_evaluation', default=None)
PROPOSAL_FIELDS = ('title', 'hypothesis', 'planned_change', 'success_criterion', 'policy_guidance_alignment')


def locate(path):
    """Find a run ledger without requiring a policy to compile."""
    path = Path(path).absolute()
    for parent in (path.parent, *path.parents):
        for candidate in (parent / 'policy_ledger.jsonl', parent / 'task/workspace/policy_ledger.jsonl'):
            if candidate.is_file():
                events = L.load_events(candidate)
                if events and events[0]['schema_version'] >= 4:
                    return candidate
    return None


def _json(path, data):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _hgroup_bundle(ledger_path):
    generation = ledger_path.parent / 'context/generation.json'
    return generation.is_file() and json.loads(generation.read_bytes()).get('mode') == 'hgroup_lab'


def _hgroup_sources(root):
    package = root / 'hgroup_jax'
    if package.is_symlink() or not package.is_dir():
        raise ValueError('hgroup_jax must be a regular package directory')
    sources = []
    for entry in package.rglob('*'):
        if entry.is_symlink():
            raise ValueError('symlinks are not allowed in H-group policy bundles')
        if entry.is_dir() or '__pycache__' in entry.parts or entry.suffix == '.pyc':
            continue
        if not entry.is_file():
            raise ValueError('unsupported H-group package file: ' + str(entry))
        sources.append(entry.relative_to(root).as_posix())
    if not {'hgroup_jax/__init__.py', 'hgroup_jax/policy.py', 'hgroup_jax/LICENSE'} <= set(sources):
        raise ValueError('H-group bundle is missing its package marker, policy or license')
    return sorted(sources)


def _read(path, source_mode=False, *, hgroup=False):
    path = Path(path)
    if path.name != 'policy.py':
        raise ValueError('candidate filename must be policy.py')
    source_mode = source_mode or (not hgroup and path.is_file() and path.read_bytes() == SOURCE_LOCATOR.encode())
    names = ('policy.py', 'agent_0.py', 'agent_1.py', 'custom_primitives.py') if source_mode else ('policy.py', 'custom_primitives.py')
    if hgroup:
        names = ('policy.py', *_hgroup_sources(path.parent))
    payload = {}
    for name in names:
        if hgroup:
            payload[name] = _read_hgroup_file(path.parent, name)
            continue
        p = path.with_name(name)
        if p.is_symlink() or (p.exists() and not p.is_file()):
            raise ValueError('source must be a regular file: ' + str(p))
        if p.exists():
            if p.stat().st_size > 2 * 1024 * 1024:
                raise ValueError('source exceeds 2 MiB: ' + str(p))
            payload[name] = p.read_bytes()
    if 'policy.py' not in payload:
        raise ValueError('policy.py is missing')
    # Match the established valid-bundle hash order.
    order = names
    digest = hashlib.sha256()
    for name in order:
        if name in payload:
            digest.update(name.encode() + b'\0' + payload[name] + b'\0')
    return payload, digest.hexdigest()


def _read_hgroup_file(root, name):
    parts = Path(name).parts
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    limit = 2 * 1024**2
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
                raise ValueError('Invalid H-group source file: ' + name)
            data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError('H-group source exceeds 2 MiB: ' + name)
            return data
    finally:
        os.close(directory)


def effective_start(events, attempt_id):
    start = next((dict(e) for e in events if e['event'] == 'attempt_started' and e['attempt_id'] == attempt_id), None)
    if start is None:
        raise L.LedgerValidationError('unknown attempt ' + attempt_id)
    for event in events:
        if event['event'] == 'attempt_annotated' and event['attempt_id'] == attempt_id:
            start.update(event['proposal'])
            start['proposal_complete'] = True
    return start


def reference(ledger_path, attempt_id):
    ledger_path = Path(ledger_path)
    start = effective_start(L.load_events(ledger_path), attempt_id)
    ref = dict(start['snapshot'], run_id=start['run_id'], attempt_id=attempt_id)
    directory = ledger_path.parent / ref['archive']
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('missing regular attempt snapshot')
    hgroup = _hgroup_bundle(ledger_path)
    if hgroup:
        payload, digest = _read(directory / 'policy.py', hgroup=True)
        if digest != ref['candidate_sha256'] or payload.keys() != ref['files'].keys():
            raise ValueError('H-group attempt snapshot changed')
    for name, item in ref['files'].items():
        if hgroup and name in payload:
            if hashlib.sha256(payload[name]).hexdigest() != item['sha256'] or len(payload[name]) != item['bytes']:
                raise ValueError('H-group attempt file changed: ' + name)
            continue
        if name not in ALL_FILES:
            raise ValueError('invalid snapshot filename')
        p = directory / name
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('attempt snapshot changed: ' + str(p))
    actual = set(payload) if hgroup else {p.name for p in directory.iterdir() if p.name in ALL_FILES}
    if actual != set(ref['files']):
        raise ValueError('attempt snapshot file set changed')
    ref['source_path'] = str(directory / 'policy.py')
    return ref


def reference_for_path(path):
    ledger = locate(path)
    if ledger is None:
        return None
    path = Path(path).absolute()
    for e in L.load_events(ledger):
        if e['event'] == 'attempt_started' and 'snapshot' in e:
            if path == ledger.parent / e['snapshot']['archive'] / 'policy.py':
                return reference(ledger, e['attempt_id'])
    return None


def _proposal(metadata, *, runner=False):
    metadata = dict(metadata or {})
    unknown = set(metadata) - set(PROPOSAL_FIELDS)
    if unknown:
        raise ValueError('unknown proposal fields: ' + ', '.join(sorted(unknown)))
    if not metadata:
        return dict(title='Unannotated revision', hypothesis='Not supplied (automatic capture).',
                    planned_change=[], success_criterion='Not supplied (automatic capture).',
                    policy_guidance_alignment={'status':'no_guidance','principles':[],
                                               'rationale':'No proposal annotation has been supplied.'}), False
    for name in ('title', 'hypothesis', 'success_criterion'):
        L._require_string(metadata.get(name), name)
    L._require_string_list(metadata.get('planned_change'), 'planned_change')
    if not metadata['planned_change']:
        raise ValueError('planned_change must contain at least one description')
    metadata.setdefault('policy_guidance_alignment', {'status':'no_guidance','principles':[],
                           'rationale':'Runner metadata.' if runner else 'No policy guidance supplied.'})
    alignment = metadata['policy_guidance_alignment']
    L._require_mapping(alignment,'policy_guidance_alignment',{'status','principles','rationale'})
    if alignment['status'] not in L.GUIDANCE_ALIGNMENT_STATUSES:
        raise ValueError('invalid guidance alignment status')
    L._require_string(alignment['rationale'],'guidance rationale')
    L._require_string_list(alignment['principles'],'guidance principles')
    if alignment['status'] in ('aligned','partially_aligned') and not alignment['principles']:
        raise ValueError('aligned proposals must identify guidance principles')
    return metadata, True


def register_candidate(path, *, ledger_path=None, run_id=None, explicit=True, parent=None,
                       metadata=None, origin='registration', runner=False):
    path = Path(path).absolute()
    ledger_path = Path(ledger_path) if ledger_path else locate(path)
    if ledger_path is None:
        return None
    if not explicit:
        frozen = reference_for_path(path)
        if frozen is not None:
            return frozen
    markdown = ledger_path.with_name('experiment_ledger.md')
    generation = ledger_path.parent / 'context/generation.json'
    source_mode = generation.is_file() and json.loads(generation.read_text()).get('mode') == 'open_source_zsc'
    hgroup = _hgroup_bundle(ledger_path)
    proposal, complete = _proposal(metadata, runner=runner)
    with L._exclusive_ledger_lock(ledger_path):
        events = L.load_events(ledger_path)
        if events and events[0]['schema_version'] < 4:
            raise ValueError('legacy ledger is read-only for registration; continue in a new run')
        payload, digest = _read(path, source_mode, hgroup=hgroup)
        starts = [e for e in events if e['event'] == 'attempt_started']
        previous = next((e for e in reversed(starts) if e['snapshot']['working_path'] == str(path)), None)
        restored = next((e for e in reversed(events) if e['event'] == 'working_copy_restored' and e['working_path'] == str(path)),None)
        if restored and (previous is None or restored['sequence'] > previous['sequence']):
            previous = next(e for e in starts if e['attempt_id'] == restored['attempt_id'])

        if not explicit and path.parent != ledger_path.parent and previous is None:
            scored = next((e for e in reversed(events) if e['event'] == 'evaluation_recorded' and e['candidate_sha256'] == digest),None)
            same = next((e for e in starts if scored and e['attempt_id'] == scored['attempt_id']),None)
            same = same or next((e for e in reversed(starts) if e['snapshot']['candidate_sha256'] == digest), None)
            if same: return reference(ledger_path, same['attempt_id'])
        if not explicit and previous and previous['snapshot']['candidate_sha256'] == digest:
            return reference(ledger_path, previous['attempt_id'])
        attempts = ledger_path.parent / '.policy_eval/attempts'
        attempts.mkdir(parents=True, exist_ok=True)
        # Orphaned published directories are retained after an interrupted ledger append.
        used = [int(e['attempt_id'][1:]) for e in starts]
        used += [int(p.name[1:]) for p in attempts.iterdir() if L.ATTEMPT_ID_PATTERN.fullmatch(p.name)]
        attempt_id = f'A{max(used, default=-1)+1:02d}'
        if parent is None and previous:
            parent = previous['attempt_id']
        if parent is not None and parent not in {e['attempt_id'] for e in starts}:
            raise ValueError('parent must identify an existing attempt in this run')
        temporary = Path(tempfile.mkdtemp(prefix='.capture-', dir=attempts))
        try:
            for name, raw in payload.items():
                (temporary / name).parent.mkdir(parents=True, exist_ok=True)
                with (temporary / name).open('xb') as handle:
                    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
            if _read(path, source_mode, hgroup=hgroup) != (payload, digest):
                raise ValueError('policy changed during capture; retry registration')
            snapshot = {'candidate_sha256':digest, 'archive':f'.policy_eval/attempts/{attempt_id}',
                        'working_path':str(path), 'origin':origin,
                        'files':{name:{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)} for name,raw in payload.items()}}
            _json(temporary / 'manifest.json', snapshot)
            os.rename(temporary, attempts / attempt_id)
            L._append_event_unlocked(ledger_path, markdown, {
                'event':'attempt_started','author':'runner' if runner else L._agent_author(),
                'attempt_id':attempt_id,'parent':parent, **proposal,
                'snapshot':snapshot,'proposal_complete':complete,
            },run_id=run_id)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
    return reference(ledger_path, attempt_id)


def annotate(ledger_path, attempt_id, metadata):
    proposal, complete = _proposal(metadata)
    if not complete:
        raise ValueError("annotation requires complete proposal fields")
    ledger_path = Path(ledger_path)
    with L._exclusive_ledger_lock(ledger_path):
        reference(ledger_path, attempt_id)
        guidance = ledger_path.parent / 'context/policy_guidance.md'
        if guidance.is_file() and guidance.read_text().strip() and proposal['policy_guidance_alignment']['status'] == 'no_guidance':
            raise ValueError('guided runs require policy-guidance alignment')
        return L._append_event_unlocked(ledger_path, ledger_path.with_name('experiment_ledger.md'),
            {'event':'attempt_annotated','author':L._agent_author(),'attempt_id':attempt_id,'proposal':proposal})


def require_proposal(ledger_path, attempt_id):
    start = effective_start(L.load_events(ledger_path), attempt_id)
    if not start.get('proposal_complete'):
        raise ValueError(f'{attempt_id} needs proposal notes before official scoring; use policy-ledger annotate')
    guidance = Path(ledger_path).parent / 'context/policy_guidance.md'
    if start['author'] != 'runner' and guidance.is_file() and guidance.read_text().strip() and start['policy_guidance_alignment']['status'] == 'no_guidance':
        raise ValueError('guided attempts require policy-guidance alignment before official scoring')


def restore(ledger_path, attempt_id, *, staging_root=None):
    ledger_path = Path(ledger_path)
    ref = reference(ledger_path, attempt_id)
    # Preserve displaced complete/invalid source before replacing it.
    register_candidate(ledger_path.parent/'policy.py',ledger_path=ledger_path,explicit=False,origin='before_restore')
    source = Path(ref['source_path']).parent
    workspace = ledger_path.parent
    # Restore the manifest exactly, even when the source is syntactically invalid.
    staged = Path(tempfile.mkdtemp(prefix='.policy-restore-', dir=staging_root or workspace))
    if _hgroup_bundle(ledger_path):
        temporary_package = staged / '.displaced'
        try:
            for name in ref['files']:
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            if (staged / 'policy.py').read_bytes() != _read_hgroup_file(workspace, 'policy.py'):
                raise ValueError('Immutable H-group locator changed')
            os.replace(workspace / 'hgroup_jax', temporary_package)
            try:
                os.replace(staged / 'hgroup_jax', workspace / 'hgroup_jax')
            except OSError:
                os.replace(temporary_package, workspace / 'hgroup_jax')
                raise
        finally:
            shutil.rmtree(staged)
        L.append_event(ledger_path, ledger_path.with_name('experiment_ledger.md'),
            {'event':'working_copy_restored','author':'runner','attempt_id':attempt_id,'working_path':str(workspace/'policy.py')})
        return ref
    for name in ALL_FILES:
        target = workspace / name
        if target.is_symlink(): raise ValueError('working source cannot be a symlink')
        if name in ref['files']:
            shutil.copyfile(source/name,staged/name)
    for name in ALL_FILES:
        target = workspace / name
        if name in ref['files']: os.replace(staged/name,target)
        elif target.exists(): target.unlink()
    staged.rmdir()
    L.append_event(ledger_path,ledger_path.with_name('experiment_ledger.md'),
        {'event':'working_copy_restored','author':'runner','attempt_id':attempt_id,'working_path':str(workspace/'policy.py')})
    return ref


def _plain(value):
    if hasattr(value, 'tolist'): return _plain(value.tolist())
    if isinstance(value, dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path): return str(value)
    return value


@contextmanager
def evaluation(refs, kind, parameters=None, task=None):
    refs = [r for r in refs if r is not None]
    if ACTIVE_EVALUATION.get() is not None and task is not None:
        from llm_policy.evaluation.commands import _evaluator_hash
        ACTIVE_EVALUATION.get().update(evaluator_sha256=_evaluator_hash(task),
            environment_config_sha256=hashlib.sha256(task.environment.config_path.read_bytes()).hexdigest())
    if not refs or ACTIVE_EVALUATION.get() is not None:
        yield None
        return
    ledgers = {str(locate(r['source_path'])) for r in refs}
    evaluation_id = uuid.uuid4().hex
    details = {'evaluation_id':evaluation_id,'kind':kind,'attempts':refs,
               'parameters':_plain(parameters or {}),'status':'running'}
    if task is not None:
        from llm_policy.evaluation.commands import _evaluator_hash
        details.update(evaluator_sha256=_evaluator_hash(task),
                       environment_config_sha256=hashlib.sha256(task.environment.config_path.read_bytes()).hexdigest())
    def save(status):
        details['status'] = status
        for name in ledgers:
            ledger = Path(name)
            directory = ledger.parent / '.policy_eval/evaluations' / evaluation_id
            directory.mkdir(parents=True,exist_ok=True)
            _json(directory / f'{status}.json', _plain(details))
            L.append_event(ledger,ledger.with_name('experiment_ledger.md'),{
                'event':'evaluation_activity','author':'runner','evaluation_id':evaluation_id,
                'attempt_ids':[r['attempt_id'] for r in refs if str(locate(r['source_path'])) == name],
                'kind':kind,'status':status,'artifact':str((directory/f'{status}.json').relative_to(ledger.parent)),
            })
    save('running')
    token = ACTIVE_EVALUATION.set(details)
    try:
        yield details
    except BaseException as error:
        details['error'] = {'type':type(error).__name__,'message':str(error)}
        save('error')
        raise
    else:
        save('completed')
    finally:
        ACTIVE_EVALUATION.reset(token)


def record_result(result, *, traces=False):
    record = ACTIVE_EVALUATION.get()
    if record is None: return
    from llm_policy.execution.public_analysis import summarize_rollout
    record['result'] = _plain(summarize_rollout(result))
    record['returns'] = _plain(result.first_returns)
    record['captured'] = _plain(result.captured)
    if getattr(result,'source_exchange',None): record['source_exchange'] = result.source_exchange
    if traces:
        record['traces'] = {name:_plain(getattr(result,name)) for name in tuple(f'{prefix}_agent_{i}' for prefix in ('trace', 'actions') for i in range(5)) if getattr(result,name,None) is not None}


def tracked_rollout(function):
    @wraps(function)
    def call(**kwargs):
        policies = [kwargs[k] for k in ('policy','policy_agent_0','policy_agent_1') if k in kwargs]
        refs = [reference_for_path(p.source_path) for p in policies]
        params = {k:v for k,v in kwargs.items() if k not in ('task','policy','policy_agent_0','policy_agent_1')}
        with evaluation(refs,'exploratory',params,kwargs['task']):
            result = function(**kwargs)
            record_result(result,traces=kwargs.get('capture_traces',False))
            return result
    return call


def tracked_command(kind):
    def decorate(function):
        @wraps(function)
        def call(args):
            ref = register_candidate(args.policy,explicit=False,origin=args.command if hasattr(args,'command') else kind)
            if ref is None: return function(args)
            args.policy = Path(ref['source_path'])
            args.attempt_ref = ref
            parameters = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('function','attempt_ref')}
            actual_kind = os.environ.get('LLM_POLICY_EVALUATION_KIND', kind) if kind == 'official_development' else kind
            with evaluation([ref],actual_kind,parameters) as record:
                if actual_kind == 'official_development': require_proposal(locate(args.policy),ref['attempt_id'])
                result = function(args)
                return result
        return call
    return decorate
