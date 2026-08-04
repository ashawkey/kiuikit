---
name: monitor
description: Run long-lived background commands and actively monitor or relaunch servers, jobs, services, logs, GPU usage, and periodic health checks.
---

# Background Processes and Monitoring

These process tools are built in and available when the active persona permits them:

- `start_process`: launch a managed background process with file-backed combined output;
- `wait_processes`: block locally until a selected process exits, optionally writes output, or an optional timeout expires;
- `inspect_processes`: return one immediate status snapshot and an optional bounded log tail;
- `stop_process`: terminate one managed process and its process tree.

Loading this skill adds the active-monitoring workflow below.

Use `exec_command` for non-interactive foreground commands expected to finish reliably. Its default timeout is 300 seconds; set `timeout=null` only when no timeout is intentionally required. Use managed tools for servers, long-running commands, and observers.

## Start and manage a process

1. Call `start_process(command, cwd)` and retain its `process_id` and `log_path`.
2. For a finite job, use `wait_processes(process_ids=[...])` with no timeout. It consumes no additional model rounds while blocked and lazily returns when a selected process exits.
3. Use a timeout only for a real deadline or scheduled check; never for progress polling.
4. Set `wake_on_output=true` only when each new log write requires diagnosis. Otherwise leave it false to avoid waking on routine output.
5. After an exit or meaningful output event, use `inspect_processes(process_id, log_tail_chars=M)` only when log content is needed. For exact incremental output, track a line offset and read `log_path` with `read_file`.
6. Call `stop_process` only when the user wants the managed process terminated. Interrupting `wait_processes` stops monitoring but leaves all managed processes running.

## Active monitoring loop

1. Establish the target, authorized corrective action, and stopping condition. Ask only when a missing detail blocks safe action.
2. Check the authoritative current state immediately. A quiet log or running observer does not prove that the target is healthy.
3. If the target or an external-system observer must remain alive, launch it with `start_process`.
4. Wait with one `wait_processes` call over all relevant managed process IDs. Omit the timeout for ordinary finite jobs so process exit drives the next model round. If an independent check is genuinely due first, set one long timeout to that deadline. Never emulate polling with repeated timed waits and progress inspections.
5. On each returned event:
   - if a process exited, inspect its relevant final log and authoritative state;
   - if output woke the wait, inspect only the new relevant output;
   - if an intentionally scheduled timeout expired, run the authoritative health/status check that made the timeout necessary;
   - diagnose meaningful changes or failures;
   - recheck current state immediately before any relaunch or mutation;
   - apply only the requested corrective action and verify its result.
6. If the stopping condition is not met, call `wait_processes` again. Do not use the general `wait` followed by `inspect_processes` for managed-process monitoring.

For schedulers or services not represented by managed processes, start a lightweight observer command that polls the external system locally and exits on a meaningful state change. Monitor that observer with `wait_processes`; if it reports changes without exiting, set `wake_on_output=true`. This avoids spending one model round per scheduler poll.

**Every non-terminal checkpoint must continue with `wait_processes`.** Do not end with a text-only progress update, ask the user to request another check, or confuse a running process with active agent monitoring.

## Stopping and failures

- Treat “continuously,” “keep watching,” “every N minutes,” and “until I stop it” as an open-ended tool loop.
- Continue until the explicit completion condition is met or the user interrupts.
- Stop normally when the requested terminal state is reached. If a monitored process exits and no restart was requested, report the result.
- If an observer or managed job fails unexpectedly, inspect its complete relevant error, diagnose it, and restart only when consistent with the user's request.
- Preserve exact job names, IDs, namespaces, users, clusters, and launch arguments. Recheck before remediation to avoid duplicate actions.

## Completion output

Report the terminal condition, corrective actions taken, final verified state, and whether any managed process remains running. If blocked, report the concrete failure and relevant log path.
