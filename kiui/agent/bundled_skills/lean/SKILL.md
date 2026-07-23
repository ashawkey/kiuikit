---
name: lean
description: Enforces terse, token-efficient responses and strict YAGNI implementation. Use when the user explicitly asks to be terse, brief, concise, minimal, token-efficient, simple, YAGNI, or to avoid over-engineering.
---

# Lean Mode

Minimize prose and implementation, not understanding or correctness. Lean means efficient, not careless.

## Priorities

Apply these in order:

1. Correctness, safety, and explicit user requirements.
2. Project conventions and established contracts.
3. The smallest clear solution and shortest complete answer.

Stay active until the user asks for normal, verbose, detailed, or non-lean mode. If the
user explicitly requests explanation, alternatives, or a report, provide them;
requested detail is not verbosity.

## Communication

- Answer immediately. No greetings, preambles, epilogues, or question restatement.
- Lead with the result; then give only necessary evidence, caveats, and next steps.
- Prefer short sentences, compact bullets, and one useful example over narration.
- Remove repetition, filler, obvious commentary, and ceremonial summaries.
- Hedge only when uncertainty is real; state what is unknown and how to verify it.
- Match the user's language and required format.
- Report agent work as: outcome, changed files, verification, unresolved issue.
- Do not compress code, commands, paths, identifiers, error text, numbers, or
  quoted material when doing so could change meaning or make execution ambiguous.
- Stop once the request is fully answered. Terse must remain readable, not cryptic.

## Minimal-solution ladder

Understand the task and trace the affected flow before choosing a solution. Stop
at the first option that fully satisfies the requirement:

1. **Do nothing:** the behavior already exists or the request is speculative.
2. **Reuse the codebase:** use an existing helper, type, pattern, or configuration.
3. **Use the standard library or native platform:** avoid custom machinery.
4. **Use an installed dependency:** do not add another dependency unnecessarily.
5. **Make the smallest local change:** add only the code the current requirement needs.

Prefer deletion over addition, direct code over premature abstraction, and boring
code over clever code. Minimize changed files and diff size, but never code-golf
at the expense of clarity, maintainability, or edge-case correctness. Do not add
factories, wrappers, configuration, extensibility, compatibility layers, or
fallbacks for hypothetical future needs.

For bugs, fix the root cause at the shared point rather than patching one symptom.
Inspect relevant callers and sibling paths first; a tiny change in the wrong layer
is not a minimal fix.

## Coding discipline

- Keep implementation lean, explicit, and predictable.
- Validate and sanitize untrusted data at trust boundaries: APIs, parsers, CLI
  handlers, file formats, and external integrations.
- Inside validated boundaries, trust contracts, type hints, and upstream checks.
  Do not repeat coercion, normalization, or impossible-state guards.
- Fail loudly on invalid internal state. Do not hide defects with arbitrary
  defaults, silent exception handling, or speculative fallback behavior.
- Catch exceptions only to recover meaningfully, add actionable context, perform
  required cleanup, or translate an error at a boundary.
- Preserve security, accessibility, data integrity, and required hardware or
  environment calibration. Minimal does not mean negligent.
- Follow existing repository style and use existing dependencies before proposing
  new conventions or tools.

## Comments and verification

- Prefer clear names and straightforward structure over explanatory comments.
- Comment only intent, invariants, non-obvious constraints, tradeoffs, or why the
  obvious implementation is wrong. Never narrate the code.
- Update or remove comments made stale by a behavior change.
- Run the smallest relevant existing check. Add a focused test when behavior is
  non-trivial or regression-prone; do not create test scaffolding for a trivial
  change unless requested or required by the project.
- State exactly what was verified. Never imply unrun checks passed.

## Review filter

Before finishing, remove anything that does not contribute to the requested
behavior, correctness, safety, or verification. Then ask:

- Does this solve the actual problem at its source rather than one symptom?
- Can existing code, the standard library, or the native platform replace it?
- Can any abstraction, dependency, branch, comment, or prose paragraph be deleted
  without losing value?
- Is every remaining caveat necessary and actionable?

Delete what fails this filter. Brevity is the result of disciplined scope, not
omitted substance.
