---
name: persona-creator
description: Create or revise and validate declarative kia personas under .kia/personas. Use when the user wants a custom persona, system prompt, behavioral profile, or restricted tool surface.
---

# Persona Creator

Create one focused persona at `.kia/personas/<name>/PERSONA.md`.

## Workflow

1. Establish the persona's recurring role, activation intent, tone, constraints, and required tools. Ask only for missing choices that materially affect behavior or permissions.
2. Choose a name of 1-64 lowercase alphanumeric segments separated by single hyphens. `reload` and bundled names such as `coder`, `chatter`, and `reviewer` cannot be overridden; use a new name when customizing a bundled persona.
3. Write `PERSONA.md` with valid frontmatter and a complete static system prompt. Prefer concise operational instructions over generic traits or duplicated agent defaults.
4. Select the narrowest tool set that supports the role. Use an explicit YAML list by default; use `all` only when the persona should automatically receive future built-in tools; use `[]` for no tools.
5. Add only the runtime markers the prompt needs, each on its own line.
6. Validate the directory with `read_persona`, fix every error, and inspect the final file for consistency between the description, prompt, and tool surface.

## Format

```markdown
---
name: my-persona
description: Review Python changes for correctness and report findings without editing files.
tools:
  - read_file
  - grep_files
  - exec_command
---
You are a correctness-first Python reviewer.

Review requested changes without modifying files. Report actionable findings first.

{{kia:project-instructions}}
{{kia:current-context}}
```

`name`, `description`, and `tools` are required. The description should say what the persona does and when a user would select it.

`tools` accepts `all` or a YAML list of built-in tool names. Skill-provided native tools cannot be named directly; include `load_skill` when the persona should discover and load skills, whose tools are then advertised while loaded.

## Runtime markers

Available whole-line markers are:

- `{{kia:autonomous-mode}}`: autonomous execution rules; empty in interactive mode.
- `{{kia:sub-agents}}`: sub-agent guidance when `spawn_subagent` is allowed and the agent is not itself a sub-agent.
- `{{kia:skills}}`: discovered skills when `load_skill` is allowed.
- `{{kia:project-instructions}}`: the working directory's `AGENTS.md`, when present.
- `{{kia:current-context}}`: working directory, operating system, and Git branch/status.

Each marker may appear at most once and must occupy its own line. There is no persona inheritance or arbitrary templating. Write all static safety, workflow, output, and style requirements explicitly so the file remains understandable before rendering.

## Validate

Run from the project root:

```bash
python -c "from kiui.agent.personas import read_persona; print(read_persona(r'.kia/personas/<name>'))"
```

If runtime marker behavior matters, also render it with a representative context:

```bash
python -c "from kiui.agent.personas import PersonaContext, read_persona; p=read_persona(r'.kia/personas/<name>'); print(p.build(PersonaContext(work_dir='.')))"
```

Report the created path and validation run. Tell the user to run `/persona reload` in an active kia session; changing the active persona restarts the conversation.
