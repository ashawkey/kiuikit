---
name: code-review
description: Review source code, patches, pull requests, diffs, commits, and implementation designs for bugs, regressions, security risks, inefficiency, maintainability problems, incorrect comments, and code quality. Use whenever the user asks to review, audit, inspect, critique, or assess code or a code change.
---
# Code Review

Perform a correctness-first review and report only the information most useful to the user.

## Review process

1. Establish the scope from the request and repository state. Inspect the relevant diff, surrounding code, callers, tests, configuration, and contracts before drawing conclusions. Do not guess from an isolated snippet when context is available.
2. Look for concrete, user-impacting issues:
   - functional bugs, regressions, edge cases, invalid assumptions, and error-handling failures;
   - security, privacy, concurrency, resource-lifetime, and data-integrity risks;
   - unnecessary work, poor algorithms, repeated I/O, excessive allocation, or other meaningful inefficiency;
   - brittle APIs, excessive coupling, misplaced responsibilities, duplication, and designs that make correct maintenance difficult;
   - missing or inadequate tests for important changed behavior.
3. Check clarity and accuracy:
   - ensure names, types, control flow, and abstractions communicate the actual behavior;
   - verify comments and documentation against the code; flag stale, misleading, redundant, or unclear comments;
   - identify confusing or needlessly complex code, but avoid subjective style complaints already handled by formatters or linters.
4. Validate each finding. Trace the execution path and state the conditions that trigger it. Use focused tests or static checks when practical. Do not present speculation as a confirmed bug.
5. Prioritize ruthlessly. Prefer a few high-confidence, actionable findings over an exhaustive list. Do not bury important defects under minor cleanup suggestions.

## Finding criteria

Report a finding only when it is specific and actionable. For each finding, include:

- severity: `Critical`, `High`, `Medium`, or `Low`;
- a concise title;
- the exact file and line or smallest relevant range;
- why it matters and the concrete failure or maintenance cost;
- a brief fix direction when it is not obvious.

Use severity according to impact, not ease of repair:

- `Critical`: likely catastrophic security, data-loss, or broad outage risk.
- `High`: serious correctness/security issue affecting common or important paths.
- `Medium`: real defect or significant design/performance issue under plausible conditions.
- `Low`: limited-impact defect or worthwhile maintainability problem. Omit cosmetic nits by default.

If evidence is incomplete, label the concern as needing verification and say what evidence is missing. Do not invent file paths, line numbers, behavior, or test results.

## Output format

Keep the final review concise and accurate:

1. Start with **Findings**, ordered by severity and then impact. Usually report no more than five; include more only when independently important.
2. Write each finding as a compact paragraph or bullet with its severity and location. Include enough evidence to understand the issue, not a full investigation log.
3. End with a one- or two-sentence **Summary** describing overall risk and the most important next step.
4. If no actionable findings exist, say so explicitly and briefly mention any material review gaps or untested risk.

Do not output every detail inspected, long walkthroughs, generic praise, or a large list of minor suggestions. Offer to expand on findings if the user wants details. Unless explicitly asked, review the code without modifying it.
