---
name: monitor
description: Continuously monitor long-running processes, GPU usage, jobs, queues, services, logs, or periodic health checks, including detecting failures and relaunching work. Use when the user asks to monitor continuously, watch, check repeatedly or periodically, run checks on an interval, keep something healthy, or continue until stopped.
---

# Continuous Monitoring

Keep the agentic tool loop alive across checkpoints so you can inspect state, reason about it, take corrective action, and wait for the next checkpoint.

## Core protocol

1. Establish the target, check interval, corrective actions, and stopping condition from the request. Ask only when a missing detail blocks safe action.
2. Use `start_process` for a long-running observer, server, job, or monitoring command. Record its `process_id` and `log_path`.
3. Call `inspect_processes(process_id, wait=<interval seconds>)` to wait for one checkpoint. The call returns one snapshot; it does not schedule another check.
4. At the checkpoint, inspect new log output and run any other commands needed to determine current state.
5. Diagnose and act. Examples include reporting GPU processes, checking all job IDs, collecting failure details, or relaunching failed jobs with the exact requested configuration.
6. If the stopping condition is not met, call `inspect_processes` again with the same interval. Repeat from step 4.

**The next waiting `inspect_processes` call is part of every non-terminal checkpoint.** Do not end with a text-only response, ask the user to request another check, or merely state that the background command is still running. Any progress update must be followed in the same turn by the next waiting tool call.

## Control flow

- Treat phrases such as “monitor continuously,” “keep watching,” “every N minutes,” and “until I stop it” as an open-ended tool loop.
- A managed background process running by itself is not active agent monitoring. Active monitoring means the agent repeatedly regains control after each wait, evaluates the checkpoint, acts, and starts the next wait.
- Use `wait=0` only for an immediate snapshot. Use the requested polling interval for the continuing wait.
- Do not replace `inspect_processes` with shell `sleep` or a foreground command; the managed-process tool keeps the wait visible and interruptible.
- Continue indefinitely when the user says “until stopped.” The user can interrupt the waiting tool call with Escape or Ctrl+C. An interrupted inspection ends the agentic loop but does not stop the managed background process; call `stop_process` only when the user also wants that process terminated.
- Stop normally only when an explicit terminal condition is reached, such as all jobs completing, or when the monitored process exits and the request does not require restarting it.
- If the observer exits unexpectedly, inspect its log, diagnose it, and restart it when that is consistent with the request; otherwise report the blocker.

## Logs and checkpoint state

- Read only output added since the previous checkpoint when practical. Track the next line offset after each `read_file` call instead of rereading the whole log.
- A quiet log does not prove health. Run the authoritative status command when the task requires current state.
- Preserve stable identifiers such as job names, namespaces, users, clusters, and launch arguments when taking corrective action.
- Avoid duplicate remediation. Recheck current state immediately before relaunching or mutating external work.
- Keep checkpoint updates concise so they do not obscure the continuing tool call.

## Examples

### GPU usage every 10 seconds until interrupted

1. Start an unbuffered loop that writes timestamped `nvidia-smi` utilization and compute-process data.
2. Call `inspect_processes(process_id, wait=10)`.
3. Read only the new log lines and summarize meaningful changes.
4. If not interrupted and the observer still runs, call `inspect_processes(process_id, wait=10)` again—never finish with “ask me to check again.”

### Relaunch randomly failing jobs every 30 minutes

1. Start or identify a managed observer process and retain its process ID. The observer may collect periodic job state, or it may simply remain alive while authoritative job-status commands run at each checkpoint.
2. Call `inspect_processes(process_id, wait=1800)`.
3. Query all expected job IDs with the authoritative scheduler command.
4. For each failed or missing job, verify its latest state and relaunch it using the required exact name and configuration. Confirm submission results.
5. Check the complete job set again.
6. Unless all jobs satisfy the user’s completion condition, immediately call `inspect_processes(process_id, wait=1800)` and continue.

Correct checkpoint shape:

```text
inspect_processes(wait=1800)
→ inspect logs and scheduler state
→ relaunch failures
→ verify job set
→ optional brief progress text plus inspect_processes(wait=1800)
```

Incorrect checkpoint shape:

```text
inspect_processes(wait=1800)
→ inspect and relaunch
→ “Monitoring is running; ask me to check again.”
→ turn ends
```
