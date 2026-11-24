# Policy-search workspace

Your goal is to generate a joint policy as Python code that maximizes the
objective in `context/task_spec.md`.

Work alone and edit only files in this `workspace/`. Do not inspect paths
outside the task bundle. The immutable inputs are:

- `context/task_spec.md`: objective, task facts, and policy contract
- `context/policy_guidance.md`: immutable environment-agnostic meta guidance
- `context/environment/`: exact configuration, selected layout, and complete
  policy-relevant environment source
- `context/policy_api.md`: supported programming interface
- `context/overcooked_primitives.py`: public primitive implementation
- `context/rollout_api.py`: exploratory rollout facade

During development, use the provided tools and local commands to inspect the
task bundle, edit workspace files, and run permitted validation and evaluation.
Do not access the network, spawn subagents, attempt to discover credentials,
processes, mounts, other runs, hidden seeds, or evaluator internals, or bypass
the evaluator. Treat `policy-eval` as an opaque authority.

Runtime policy code must remain JAX-compatible and must not perform host I/O,
launch subprocesses, access the network, or use host callbacks.

{{FINAL_POLICY_SELECTION}}

Read `context/policy_guidance.md` before opening a new attempt. Apply it only
as general policy-search discipline, subordinate to the objective and runtime
information boundary. Every attempt proposal must explicitly record whether it
aligns with that guidance, identify the relevant general principle(s), and
explain the relationship. This is an auditable claim, not a presumption that
all proposed changes are aligned.

Keep `POLICY_API_VERSION`, all required `POLICY_METADATA` keys, both
initializers, and both act functions in `policy.py`. The starter's task
dimensions are rendered from the instantiated environment: ingredient count
must remain exact, while map dimensions must not be reduced below those values.
Each act function sees only its local observation, reset flag, private
fixed-shape memory, and PRNG key, and returns a scalar integer action,
next memory with the same structure, shapes, and dtypes, and `PolicyTrace`. Use JAX arrays and
fixed-shape control flow (`jnp.where`, `jax.lax.cond` or `jax.lax.switch`),
never array-dependent Python branching or mutable globals.

Use `policy-ledger` for the append-only ledger; never hand-edit its JSONL or
Markdown view. A00 is open for the starting template. For every material
revision: score and conclude the current attempt, open the next attempt with a
falsifiable hypothesis, edit, validate, evaluate, diagnose, then conclude.

After scoring the baseline, explicitly declare the scored policy you would
submit now using `policy-ledger select-best`. Every attempt conclusion must
reaffirm or change that preference with `--best-candidate-sha256` and
`--best-reason`, including rejected attempts and unchanged preferences. Choose
according to the objective in `context/task_spec.md`{{LEDGER_SELECTION_NOTE}} An older candidate
is allowed. Only candidates officially scored in this run can be selected.
Use `select-best` whenever your judgment changes between attempts and once
more immediately before finishing. This records a preference without changing
the working policy files.

Use `candidate_sha256` from an official `policy-eval score` report. This hash
identifies the exact `policy.py` and optional `custom_primitives.py` bundle,
not `policy.py` alone. Scored bundles are archived under
`.policy_eval/candidates/HASH/` inside the workspace. Keep these archives intact.
{{LEDGER_FINALIZATION_INSTRUCTION}}For SP, the runner still selects by independently recomputed development score;
your logged preference does not override its selection.

Core commands:

```bash
policy-eval validate policy.py
policy-eval score policy.py
policy-eval trace policy.py --episodes 3,17
python rollouts.py
policy-ledger start --title "..." --parent A00 \
  --hypothesis "..." --planned-change "..." --success-criterion "..." \
  --guidance-alignment aligned --guidance-principle "..." \
  --guidance-rationale "..."
policy-ledger observe --kind exploratory --summary "..."
policy-ledger select-best --candidate-sha256 HASH --reason "..."
policy-ledger restore-best
policy-ledger conclude --agent-0-role "..." --agent-1-role "..." \
  --coordination-protocol "..." --failure-mode "..." --decision follow_up \
  --limitation "..." --best-candidate-sha256 HASH --best-reason "..."
```

`policy-eval score` is the only authoritative development measurement.
Editable rollouts are exploratory and must be labelled as such. Official
development scores are provisional: the runner independently revalidates and
evaluates candidate source after the session. Leave in `policy.py` the candidate
selected according to the objective-specific selection rule in
`context/task_spec.md`. Record evidence, failure modes, and limitations in the ledger.
