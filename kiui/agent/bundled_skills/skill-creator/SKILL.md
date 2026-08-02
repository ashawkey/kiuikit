---
name: skill-creator
description: Use this skill when the user wants to create, revise, or validate an Agent Skill; package a repeatable workflow or domain procedure; add scripts, references, or assets.
---

# Skill Creator

Create skills that are clear, accurate, focused, and useful.

## Workflow

### 1. Understand the task

Establish:

- the recurring outcome the skill should produce;
- when it should activate;
- the inputs, outputs, and important constraints;
- whether invoking `/<skill-name>` without task context should run a specific
  default workflow or simply load the skill and ask what to do;
- one or two representative examples.

Use existing conversation context when sufficient. Ask only for important
missing facts. Do not create a skill that merely restates generic advice the
agent already knows.

### 2. Verify the facts

Ground instructions in the best available evidence: user-provided procedures,
project code and documentation, successful task traces, or authoritative
external documentation. Do not invent commands, APIs, paths, or conventions.
Call out any material assumption that remains unverified.

### 3. Write a focused skill

Create the skill as project-local state under `.kia/skills/<skill-name>/`. 
Use the `kib` library workflow to share or synchronize skills: `kib upload <skill-name>` publishes a local skill, `kib install <skill-name>` installs it in another project, and `kib update <skill-name>` synchronizes an installed copy. 
Do not try to track the skill in git.

Choose one kebab-case name and one coherent responsibility. Read
`references/skill-format.md` for the required format. Start from
`assets/SKILL.template.md` when useful, then delete unused sections.

Write the `description` around user intent: say what outcome the skill enables
and when to use it. Include likely user terminology, but avoid keyword stuffing.

Write the body as operational instructions:

- use imperative steps;
- include concrete constraints and non-obvious gotchas;
- define the expected output when consistency matters;
- include a verification step;
- omit generic statements such as “follow best practices.”

Add a `## Default invocation` section only when `/<skill-name>` without
additional context can safely perform a clearly defined workflow. Specify its
outcome, input discovery, permitted actions, verification, and stopping
condition. General capability skills should omit this section; direct invocation
will then load the skill and ask the user for a task instead of inventing one.

### 4. Add only useful resources

- Keep frequently needed instructions in `SKILL.md`.
- Put lengthy or conditional detail in `references/` and say when to read it.
- Use `scripts/` for repeated deterministic operations.
- Use `assets/` for templates and static files.
- Prefer scripts over `tools.py` unless a structured callable tool or
  session-scoped in-process state provides a real benefit.

For native tools, read `references/native-tools.md` first.

### 5. Validate

Run:

```bash
python scripts/validate_skill.py <path-to-skill>
```

Resolve the script against the absolute skill-creator directory shown when this
skill loaded. Fix reported errors. Run focused checks for any bundled scripts or
native tools.

The validator checks structure and loadability. It does not prove that the
instructions are factually correct or useful.

### 6. Review and report

Before finishing, check:

- **Clear:** procedures have explicit inputs, outputs, and completion conditions.
- **Invocation:** any declared default is self-contained and safe; otherwise the
  skill correctly omits `## Default invocation`.
- **Accurate:** commands, paths, APIs, and examples are verified.
- **Focused:** every instruction changes likely agent behavior.
- **Consistent:** `SKILL.md`, references, scripts, and examples agree.

Report the files created, checks actually run, and unresolved assumptions. Tell
the user to run `/skills reload` after adding or changing a project or personal
skill.
