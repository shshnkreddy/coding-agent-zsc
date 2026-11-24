"""Agent-facing immutable revision workflow."""
INSTRUCTIONS = """# Policy revision provenance

Make your policy refinement process traceable: someone reviewing the run should
be able to see what you tried, why you tried it, and which results informed your
final choice. Use `policy-ledger` to keep that history, including unsuccessful
and unofficial attempts.

1. **Save each revision.** After editing, run `policy-ledger register policy.py`
   before testing or replacing it. This saves the policy files under a new ID
   such as A01. A00 is the starting policy; saved revisions never change.
2. **Explain the change.** Record its parent, hypothesis, changes and success
   criterion when registering. You can add these later with
   `policy-ledger annotate A01`, but they are required before official scoring.
   If guidance was supplied, explain how the revision follows or departs from it.
3. **Record what you learned.** `policy-eval` saves results and errors
   automatically. Use `policy-ledger conclude --attempt-id A01` to explain
   the outcome and your next step. Distinguish your original expectations
   from what you learned afterward.
4. **Explain your final choice.** Use `policy-ledger select-best` with an
   officially scored candidate's hash and your reason, following the task's
   objective. For XP, run `policy-ledger restore-best` before submitting.

For example, after editing the starting policy:

```bash
policy-ledger register policy.py --parent A00 --title "..." \\
  --hypothesis "..." --planned-change "..." --success-criterion "..."
policy-eval validate policy.py
policy-eval score policy.py
```

Read `experiment_ledger.md` to review the history. Use `inspect` and `restore`
to retrieve saved attempts, and each command's `--help` for its arguments.
Never edit the ledger or `.policy_eval` archive directly.

Runner checks also capture changed source automatically, keeping the same ID
while the source is unchanged. Explicit registration always creates a new ID.
Drafts overwritten before either step are lost. Exploratory results are saved;
only `policy-eval score` counts as official development evidence.
"""


def install(task_root):
    path = task_root / 'AGENTS.md'
    text = path.read_text()
    start = text.index('Use `policy-ledger` for the append-only ledger')
    end = text.index('No pre-existing primitive libraries', start) if 'No pre-existing primitive libraries' in text[start:] else text.find('\n\n## Agent-search lifetime', start)
    if end < 0:
        end = len(text)
    text = text[:start] + INSTRUCTIONS + '\n\n' + text[end:]
    # The inherited baseline is immutable even before it has an evaluation.
    text = text.replace('A00 contains the runner-scored inherited selected-policy baseline and is already concluded. Start at A01.',
                        'A00 contains the frozen inherited baseline. Register revisions starting at A01.')
    path.write_text(text)
    (task_root / 'context/provenance.md').write_text(INSTRUCTIONS)
