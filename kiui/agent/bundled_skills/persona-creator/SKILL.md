---
name: persona-creator
description: Create and validate declarative kia personas. Use when the user wants a custom persona, system prompt, or tool surface under .kia/personas.
---

# Persona Creator

Create each persona as `.kia/personas/<name>/PERSONA.md`. Names are 1-64 lowercase alphanumeric characters separated by single hyphens; `reload` is reserved. Do not reuse bundled names (`coder`, `chatter`, or `reviewer`); copy one under a new name when customizing it.

Use this format:

```markdown
---
name: my-persona
description: Short description shown by /persona.
tools:
  - read_file
  - web_search
---
The complete system prompt goes here.

{{kia:current-context}}
```

`tools` is required. It is either `all`, `[]`, or a YAML list of built-in tool names. Prefer an explicit list unless the persona intentionally follows future additions to the built-in tool set.

Available whole-line markers:

- `{{kia:autonomous-mode}}` — autonomous execution rules, omitted in interactive mode.
- `{{kia:sub-agents}}` — sub-agent guidance when `spawn_subagent` is available and the agent is not already a sub-agent.
- `{{kia:skills}}` — discovered skill names and descriptions when `load_skill` is available.
- `{{kia:project-instructions}}` — the working directory's `AGENTS.md`, if present.
- `{{kia:current-context}}` — working directory, operating system, and Git branch/status.

Markers must occupy their own lines, may appear at most once, and are expanded exactly once. There is no inheritance or arbitrary templating. Write all static safety, workflow, and style instructions explicitly so the resulting persona remains understandable by reading one file.

After writing the file, validate it with:

```bash
python -c "from kiui.agent.personas import read_persona; print(read_persona(r'.kia/personas/<name>'))"
```

Then tell the user to run `/persona reload` in an active kia session. A changed active persona restarts the conversation.
