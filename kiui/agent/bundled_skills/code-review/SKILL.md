---
name: code-review
description: Find actionable bugs, regressions, security risks, performance problems, and maintainability issues in source code, diffs, commits, pull requests, or implementation designs. Use when the user asks for a code review, audit, critique, inspection, or risk assessment.
---

# Code Review

Review for concrete risk, not stylistic preference. Unless the user asks for changes, inspect and report without modifying code.

## Workflow

1. Establish the review target and baseline from the request and repository state. If scope is ambiguous, identify the smallest likely diff and state the assumption.
2. Inspect the complete relevant diff plus enough surrounding context to understand contracts and behavior: callers, data flow, tests, configuration, and platform constraints. Do not infer behavior from an isolated hunk when the repository can answer it.
3. Look for user-impacting problems:
   - incorrect behavior, regressions, edge cases, and broken error handling;
   - security, privacy, concurrency, resource-lifetime, and data-integrity risks;
   - meaningful algorithmic, allocation, or I/O inefficiency;
   - brittle interfaces, misplaced responsibility, duplication, or complexity that makes defects likely;
   - missing tests for important changed behavior;
   - comments or documentation that no longer match the code.
4. Validate every candidate finding. Trace a concrete failure path and run the smallest useful test or static check when practical. Separate confirmed defects from concerns that need evidence.
5. Keep only findings that are specific, actionable, and worth the user's attention. Omit formatter issues, cosmetic preferences, generic advice, and speculative future concerns.

## Findings

For each finding, provide:

- severity: `Critical`, `High`, `Medium`, or `Low`;
- a concise title and exact `file:line` location;
- the triggering conditions and resulting impact;
- a brief fix direction when it is not obvious.

Assign severity by impact:

- `Critical`: likely catastrophic security, data-loss, or broad outage risk.
- `High`: serious correctness or security failure on an important path.
- `Medium`: real defect or significant design/performance problem under plausible conditions.
- `Low`: limited-impact defect or substantial maintainability issue.

Do not invent paths, line numbers, behavior, or test results. Label incomplete evidence explicitly and say what would verify it.

## Output

Start with **Findings**, ordered by severity and impact; usually include no more than five. End with a brief **Summary** of overall risk and the most important next step.

If there are no actionable findings, say so and mention any material scope or verification gap. Report only checks actually run; omit investigation logs, generic praise, and minor suggestion lists.
