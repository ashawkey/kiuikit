---
name: monitor
description: Run and manage background processes or actively watch servers, jobs, queues, services, logs, GPU usage, and periodic health checks. Use when the user asks to start a long-running or potentially stuck command, monitor repeatedly or on an interval, keep work healthy, relaunch failures, or continue until stopped.
---

# Background Processes and Monitoring

Loading this skill enables:

- `start_process`: launch a managed background process with file-backed combined output;
- `inspect_processes`: return one status snapshot and an optional bounded log tail;
- `stop_process`: terminate one managed process and its process tree.

Use `exec_command` for non-interactive foreground commands expected to finish reliably. It times out after 300 seconds by default; set `timeout=null` only when no timeout is intentionally required. Use the managed tools for servers and long-running commands, and the core `wait` tool for monitoring intervals.

## Start and manage a process

1. Call `start_process(command, cwd)` and retain its `process_id` and `log_path`.
2. Inspect it with `inspect_processes(process_id, log_tail_chars=M)`. Each call returns one immediate snapshot.
3. Use the returned tail for occasional output. For frequent checks or exact incremental output, track a line offset and read `log_path` with `read_file`.
4. Call `stop_process` only when the user wants the managed process terminated. Interrupting a `wait` stops the agent's monitoring loop but leaves the process running.

## Active monitoring loop

1. Establish the target, interval, authorized corrective action, and stopping condition. Ask only when a missing detail blocks safe action.
2. Check the authoritative current state immediately. A quiet log or running observer does not prove that the target is healthy.
3. If the command itself must remain alive, launch it with `start_process`. Use the core `wait(seconds=N)` tool between monitoring checkpoints.
4. At each checkpoint:
   - inspect relevant process status and new logs;
   - run authoritative status commands;
   - diagnose meaningful changes or failures;
   - recheck current state immediately before any relaunch or mutation;
   - apply only the requested corrective action and verify its result.
5. If the stopping condition is not met, issue `wait(seconds=N)` first, followed by the next inspection or status calls in the same sequential tool-call batch, then repeat.

**Every non-terminal checkpoint must end with the next `wait` and check batch.** Do not end with a text-only progress update, ask the user to request another check, or confuse a running background process with active agent monitoring. Do not put `wait` and its subsequent checks in a parallel tool-call group.

## Stopping and failures

- Treat “continuously,” “keep watching,” “every N minutes,” and “until I stop it” as an open-ended tool loop.
- Continue until the explicit completion condition is met or the user interrupts.
- Stop normally when the requested terminal state is reached. If a monitored process exits and no restart was requested, report the result.
- If an observer or managed job fails unexpectedly, inspect its complete relevant error, diagnose it, and restart only when consistent with the user's request.
- Preserve exact job names, IDs, namespaces, users, clusters, and launch arguments. Recheck before remediation to avoid duplicate actions.
- Keep checkpoint updates brief, and follow any non-terminal update with the next `wait` and check batch.

## Completion output

Report the terminal condition, corrective actions taken, final verified state, and whether any managed process remains running. If blocked, report the concrete failure and relevant log path.
