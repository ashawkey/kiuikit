---
name: subagent
description: Delegate a self-contained task to a fresh, independent kia agent and wait for its structured result. Use when the user asks for a subagent, when an investigation or implementation can be delegated, or when several independent agent tasks should run in parallel.
---

# Subagent Delegation

Run a fresh kia agent through `scripts/run_subagent.py`. The child has its own
conversation and token usage, but shares the selected working directory and
writes directly to it. It does not inherit this conversation, loaded skills, or
session/rewind history.

Use a subagent when a task is substantial and can be specified independently.
Handle small steps directly, and use the `batch` skill instead when one identical
operation repeats over many items.

## Prepare the task

Make the delegated task self-contained. Include:

- the exact goal and expected deliverable;
- relevant paths, constraints, and known evidence from this conversation;
- whether it may edit files or must only investigate;
- the checks it should run and the desired response shape.

Do not assume the child can see prior messages. It discovers project instructions
and available skills from its working directory, but if it needs a particular
skill, explicitly tell it to load that skill. The child runs in autonomous
execution mode, whose system prompt forbids invoking `subagent` or launching
another agent through `run_agent`; delegation depth is always one.

Subagents share the live filesystem. Their edits are already present when they
return; do not ask them for patches to reapply. Do not run parallel writers
against the same files or otherwise coupled state.

## Run one subagent

Resolve `scripts/run_subagent.py` against the absolute skill directory shown when
this skill was loaded. Run it as a foreground command with `exec_command`:

```bash
python <skill-dir>/scripts/run_subagent.py --task 'Inspect the parser and report the root cause. Do not edit files.'
```

Set `cwd` on `exec_command` to the intended project directory and choose an
explicit timeout appropriate to the task. Let `exec_command` own waiting,
timeout, and interruption; do not add an internal timeout or launch the command
in the shell background. Its timeout or user interruption terminates the child
process tree.

Useful runner options:

```text
--task-file PATH
--model-alias ALIAS
--persona NAME
--work-dir PATH
--reasoning-effort {none,minimal,low,medium,high,xhigh}
```

Omitting `--model-alias` selects the first configured model, as `run_agent` does.
Omitting `--persona` uses the default `coder` persona. Normally set the command's
`cwd` and omit `--work-dir`; use `--work-dir` only when it should differ.

Shell-quote every argument. For a multiline or quote-heavy prompt, write a unique
temporary task file under `.kia/scratch/`, pass `--task-file`, and remove it after
the run. Never place credentials in a delegated prompt.

The runner prints exactly one JSON object with:

- `success`, `outcome`, `response`, and `error`;
- `token_usage` for the independent run.

Exit code `0` means completed, `1` means failed or could not start, and `130`
means interrupted. Read the JSON even after a nonzero exit because it contains
the child error when one was available. A forced `exec_command` timeout or
interruption may terminate the runner before it emits JSON; in that case use the
tool's `timed_out` or `interrupted` status. If command output is compacted,
recover the full capture using the artifact guidance returned by `exec_command`.

## Run independent subagents in parallel

Only parallelize tasks that can safely operate on shared state concurrently.
For read-only investigations, disjoint files, or separate worktrees:

1. Load the `monitor` skill.
2. Call `start_process` once per runner command, retaining every process ID and
   log path. Give each child a self-contained task and the correct `cwd`.
3. Inspect statuses immediately. While any remain active, call the core `wait` tool first and then `inspect_processes` in the same sequential tool-call batch.
4. Read each process log's JSON result. Use a bounded log tail first; read the
   log file directly if needed.
5. If abandoning the group because of a deadline, failure, or user request,
   call `stop_process` for every child still running.

Managed background processes use the monitor skill's lifecycle rather than an
`exec_command` timeout. Prefer one foreground child unless parallel execution is
materially useful.

## Validate and integrate

Treat a successful child response as evidence, not automatic proof. Inspect the
claimed files and run the smallest relevant parent-side check, especially when
multiple children wrote files. Resolve conflicting recommendations yourself.

Report the delegated outcome, material findings or edits, verification actually
performed, and any failed/interrupted child. Include token usage only when it is
useful or requested.
