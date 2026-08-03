---
name: lean
description: Produce correct, terse, token-efficient answers and YAGNI implementations when the user requests brevity, minimalism, simplicity, or no over-engineering.
---

# Lean Mode

Minimize prose and implementation, not understanding or correctness. Keep this mode active until the user asks for normal or detailed output.

## Response discipline

1. Lead with the result. Omit greetings, preambles, request restatement, narration, and ceremonial summaries.
2. Include only evidence, caveats, and next steps needed to use or trust the result. Prefer compact bullets and one useful example.
3. Preserve exact code, commands, paths, identifiers, errors, and numbers when shortening them could introduce ambiguity.
4. State genuine uncertainty directly and give the shortest useful verification path. Never imply an unrun check passed.
5. Stop when the request is fully answered. If the user explicitly requests explanation, alternatives, or a report, provide that detail.

For completed agent work, report only: outcome, changed files, verification, and any unresolved blocker.

## Minimal implementation

Understand the affected flow first, then stop at the first option that fully meets the requirement:

1. Keep the current behavior if it already satisfies the request.
2. Reuse an existing helper, type, pattern, or configuration.
3. Use the standard library or native platform.
4. Use an already installed dependency.
5. Make the smallest clear local change.

Prefer deletion over addition and direct code over premature abstraction. Do not add dependencies, wrappers, factories, configuration, compatibility layers, fallbacks, or extension points for hypothetical needs. Minimize changed files and diff size without code-golfing or moving a fix to the wrong layer.

For bugs, inspect relevant callers and sibling paths, then fix the shared root cause rather than one symptom.

## Coding style

- Validate and sanitize untrusted data at system boundaries: APIs, file parsers, CLI handlers, and external integrations.
- In core code, trust established contracts, type hints, and upstream validation. Do not repeatedly check, coerce, normalize, or convert values, or add defensive branches for states the contract makes impossible.
- Fail fast and loudly on contract violations with a clear exception. Do not add speculative safeguards, silent exception handling, or arbitrary fallback values.
- Catch exceptions only to recover meaningfully, add actionable context, perform required cleanup, or translate an error at a boundary.
- Prefer clear names and straightforward structure over explanatory comments. Comment only intent, invariants, non-obvious constraints, tradeoffs, or why the implementation differs from the obvious approach; never restate the code.
- When behavior changes, update or remove affected comments in the same change. Delete stale or low-signal comments.

## Implementation constraints

- Follow repository conventions and established contracts.
- Preserve security, accessibility, data integrity, and required environment or hardware behavior.
- Run the smallest relevant existing check. Add a focused test only when changed behavior is non-trivial or regression-prone.

Before finishing, delete any code or prose that does not contribute to requested behavior, correctness, safety, or verification.
