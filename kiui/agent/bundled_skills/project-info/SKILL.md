---
name: project-info
description: Generate, audit, or refine a repository's AGENTS.md from verified manifests, source structure, tests, CI, and documentation. Use when users want project instructions, onboarding context, or a more accurate and token-efficient AGENTS.md.
---

# Project Info

Create or improve the working-directory root `AGENTS.md` as durable, always-loaded context for future agents. Optimize for correct decisions per token, not minimum length alone.

This is repository-context synthesis, not a general code review or a global brevity mode. Inspect source only to verify project instructions; do not audit unrelated defects, modify implementation files, or impose terse behavior on other tasks.

## Workflow

1. Establish whether the user wants creation, refinement, validation, or recommendations only. Inspect Git status and the existing `AGENTS.md` diff before editing so user changes are preserved. Kia currently loads only `<work_dir>/AGENTS.md`; do not create nested instruction files unless the user explicitly targets a harness that supports them.
2. Gather evidence with focused reads and searches. Start with package/build manifests, lockfiles, CI workflows, test configuration, top-level source/test layout, README or contributor docs, ignore rules, and the implementation of any claimed non-obvious contract. Prefer manifests and code over prose when they disagree.
3. Extract only information that is repository-specific, verified, durable, and likely to change an agent's implementation or verification choices:
   - supported runtimes, package managers, and installation modes;
   - authoritative manifests and source-of-truth modules;
   - subsystem-level repository map;
   - architectural ownership, public API, and lifecycle invariants;
   - optional-dependency, generated-file, security, and compatibility constraints;
   - focused build, test, lint, typecheck, and documentation commands.
4. Write or refine the file. A useful default structure is `Project`, `Repository map`, `Important contracts`, `Verification`, and `Change discipline`; omit empty sections. Use exact paths and runnable commands. Point to an authoritative manifest instead of duplicating volatile lists.
5. Prune content whose expected prompt value is low: generic coding-agent rules, marketing, exhaustive file inventories, implementation trivia, history, current branch/status, duplicated README text, speculative guidance, and unverified commands. Keep an unusual constraint when violating it would be costly even if it applies infrequently.
6. Validate the result:
   - confirm referenced paths and factual claims against repository evidence;
   - run `git diff --check`;
   - exercise the smallest safe documented command when practical, without installing large dependency sets or running expensive full suites solely to validate the document;
   - for refinements, compare before/after size and inspect the complete diff, but never trade away necessary constraints just to reduce a count.
7. Report the changed file, material corrections, checks actually run, and unresolved assumptions. If review-only was requested, provide prioritized recommendations without editing.

## Writing rules

- Write directives only where a future agent must act differently; state ordinary project facts directly.
- Prefer short bullets and compact path-to-responsibility mappings over prose catalogs.
- Distinguish required checks from optional or area-specific checks.
- Do not include secrets, local machine details, personal paths, or ignored configuration values.
- Preserve explicit user policies unless they are factually stale; explain any policy removal or semantic change.
- Keep commands platform-neutral when the repository is cross-platform, or label platform-specific commands.

## Completion criteria

The final `AGENTS.md` should let an unfamiliar agent quickly answer: what is this project, where should a change go, which contracts must remain true, and what is the smallest relevant verification? Every non-obvious claim must be traceable to repository evidence.
