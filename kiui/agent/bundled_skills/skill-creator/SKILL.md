---
name: skill-creator
description: Create, structure, and validate new Agent Skills (SKILL.md packs), including kia-specific skills that ship native tools via tools.py. Use this when the user wants to author a new skill, package reusable domain knowledge or a repeatable workflow, add bundled scripts/references/assets, give a skill its own callable tools, or fix an existing skill so it follows the Agent Skills format.
---

# Skill Creator

Guide for authoring effective Agent Skills that follow the open Agent Skills
format (https://agentskills.io). A skill is a directory containing a `SKILL.md`
file (YAML frontmatter + markdown instructions) plus optional bundled resources.

## When to use this skill

Use it whenever the user wants to:
- create a new skill from scratch,
- turn a repeatable workflow or piece of domain knowledge into a reusable pack,
- add or restructure bundled `scripts/`, `references/`, or `assets/`,
- fix a skill that fails validation or is not being triggered reliably.

## Directory layout

```
<skill-name>/
├── SKILL.md          # required: frontmatter + instructions
├── tools.py          # optional (kia extension): native tools injected when the skill loads
├── scripts/          # optional: executable code the agent runs (Bash/exec)
├── references/       # optional: docs the agent reads on demand
└── assets/           # optional: templates / data files referenced by path
```

`tools.py` is a **kia-specific extension** (not part of the open Agent Skills
format). See "Native tools (kia extension)" below. Everything else in this
guide is standard and portable across Agent Skills hosts.

Create the skill under `.kia/skills/<skill-name>/` (kia's own dir) unless the
user asks for a different location.

## SKILL.md format

`SKILL.md` MUST start with a YAML frontmatter block delimited by `---`, followed
by markdown instructions.

```markdown
---
name: pdf-processing
description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
---
Step-by-step instructions go here.
```

### Frontmatter fields

| Field | Required | Rules |
|-------|----------|-------|
| `name` | yes | 1–64 chars, lowercase alphanumeric + single hyphens, no leading/trailing/consecutive hyphens. **Must equal the directory name.** |
| `description` | yes | 1–1024 chars. State **what** the skill does **and when** to use it, with concrete trigger keywords. |
| `license` | no | License name or bundled license file reference. |
| `compatibility` | no | ≤500 chars. Only if the skill needs specific tools/packages/network/product. |
| `metadata` | no | Arbitrary string→string map (e.g. `author`, `version`). |
| `allowed-tools` | no | Space-separated pre-approved tools. **Parsed for compatibility but NOT enforced by kia** (kia uses its own permission model). |

## The description is the most important field

`description` is the ONLY thing the agent sees at startup — it decides when to
load the skill purely from this text. Make it action-oriented and specific.

- Good: `Extracts text and tables from PDFs, fills forms, and merges files. Use when working with PDFs, forms, or document extraction.`
- Poor: `Helps with PDFs.`

Include the concrete nouns/verbs a user would mention so the match is reliable.

## Progressive disclosure — keep SKILL.md small

Skills load in three stages; structure content to exploit this:

1. **Metadata (~100 tokens)** — `name` + `description`, always loaded.
2. **Instructions (< 5000 tokens, ≤ ~500 lines)** — the `SKILL.md` body, loaded on activation.
3. **Resources (as needed)** — files under `scripts/`/`references/`/`assets/`, loaded only when the instructions call for them.

Keep the body focused. Move long reference material, schemas, and checklists
into `references/` and point to them by relative path.

## Referencing bundled files

Reference bundled files by **relative path from the skill root** and keep
references one level deep:

```markdown
See references/REFERENCE.md for the full field list.
Run scripts/extract.py to pull the tables.
```

When kia activates a skill it prepends the skill's absolute directory to the
loaded body, so resolve relative paths against that directory using `read_file`
(for references/assets) or `exec_command` (for scripts).

- `references/` — text loaded **into context** via `read_file`. Keep files small and focused.
- `scripts/` — executable code run via `exec_command`. Self-contained, clear errors, handles edge cases.
- `assets/` — templates/binaries referenced **by path**, not read into context (copy, fill, or emit them).

## Native tools (kia extension)

Standard skills add capability through `scripts/` run via `exec_command`. That is
portable and should be the **default choice**. kia adds one non-standard option:
a skill may ship a `tools.py` that injects real, model-callable tools into the
agent **only while the skill is loaded**. Other Agent Skills hosts ignore
`tools.py`, so the `SKILL.md` still works everywhere; the extra tools simply do
not appear there.

### When to use tools.py vs scripts/

Prefer `scripts/` + `exec_command` unless the capability genuinely needs to live
inside the agent process. Reach for `tools.py` only when the tool must:

- share **session-scoped state** with the executor (e.g. a managed-process
  registry, long-lived handles) that a one-shot script cannot hold, or
- present a **structured, schema-validated call** the model invokes directly and
  repeatedly, rather than a shell command.

The bundled `monitor` skill is the reference example: its
`start_process` / `inspect_processes` / `stop_process` tools drive a process
registry the core executor owns, so they must run in-process. Read
`bundled_skills/monitor/tools.py` before writing your own.

> **Trust:** loading a skill imports and executes its `tools.py` in the agent
> process. Only add native tools to skills you author and trust. This is
> appropriate for kia (a personal harness where all skills are user-authored),
> not for skills fetched from untrusted third parties.

### tools.py contract

`tools.py` exposes a module-level `TOOLS` list. Each entry is a dict:

| Key | Required | Meaning |
|-----|----------|---------|
| `schema` | yes | OpenAI function schema (`{"type": "function", "function": {"name", "description", "parameters"}}`). |
| `run` | yes | Callable invoked as `run(executor, **arguments)`; returns a result dict. |
| `permission` | no | `"safe"` (never prompts) or `"risky"` (prompts in default mode). Defaults to `"risky"`. |

`tools.py` is loaded as a standalone module (not part of a package), so it MUST
be **self-contained**: import installed packages absolutely — `kiui.agent.*` is
available (e.g. `from kiui.agent.tools.constants import ...`), as are numpy,
torch, etc. — but do NOT import sibling files in the skill folder (`import
helper` or `from . import helper` will fail). Put all tool code in `tools.py`.
If a tool needs a bundled file, reference it by **path** and read/run it (as
`monitor/tools.py` does with `process_supervisor.py`), rather than importing it.

Rules the executor enforces at load time:

- A tool `name` MUST NOT collide with a built-in tool (`read_file`,
  `exec_command`, …) or with a tool another loaded skill already provides. A
  collision (or any import error in `tools.py`) fails the whole `load_skill`
  atomically: the skill is not marked loaded and no tools are registered, so a
  broken tools.py never leaves the skill half-loaded.
- Tools are advertised and callable **only while the skill is loaded**. They are
  injected in the same turn `load_skill` runs, removed on unload / `/clear` /
  session restart, and re-registered automatically on `--resume`.
- The active **persona** must be able to load skills. Any persona that can call
  `load_skill` (the default `coder`, or any persona whose `TOOLS` whitelist
  includes `load_skill`) advertises a loaded skill's tools; a persona that
  cannot load skills never sees them.
- Choose `permission` honestly: read-only inspection is `"safe"`; anything that
  launches, mutates, or terminates work is `"risky"`. Shell-command safety guards
  still apply regardless of this flag.

### run(executor, ...) contract

- **First argument is the `ToolExecutor`.** Use it for shared services:
  `executor.console` (status output via `.tool(...)`), `executor._resolve_path(p)`
  (resolve a path against the working dir), `executor.cancellation` (honor user
  interrupts during waits), and any session-scoped registry the tool manages.
- **Remaining kwargs are exactly the model-supplied arguments**, already parsed
  from the schema — trust them; do not re-validate types the schema guarantees.
- **Return a dict and always include a `success` boolean.** On failure return
  an `error` message with `success` false. Keep payloads small and bounded; large
  text should be truncated with a clear note, matching built-in tools.

### Minimal template

```python
# <skill-name>/tools.py

def echo(executor, text=""):
    executor.console.tool(f"echo: {text[:60]}")
    if not text:
        return {"error": "text is required.", "success": False}
    return {"echoed": text, "success": True}

TOOLS = [
    {
        "permission": "safe",
        "run": echo,
        "schema": {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text back. Use to demonstrate a skill tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Text to echo"}},
                    "required": ["text"],
                },
            },
        },
    },
]
```

Reference the tools from the `SKILL.md` body so the model knows they exist once
loaded (e.g. "Loading this skill enables `echo` …"), and put the triggers that
should pull the skill in into the `description`.

## Recommended body structure

```markdown
---
# frontmatter
---
# <Skill Name>

One or two sentences on purpose.

## When to use this skill
[explicit triggers]

## Instructions
### Step 1: ...
### Step 2: ...

## Examples
[concrete input → output]

## Edge cases / error handling
[what to do when things fail]

## Resources
[reference scripts/, references/, assets/ if bundled]
```

Write instructions in imperative voice ("Extract the tables…", not "You should…").

## Creation workflow

1. **Clarify** what task the skill automates and gather 1–2 concrete examples.
2. **Choose a name** (kebab-case) and create `.kia/skills/<name>/`.
3. **Write SKILL.md** with valid frontmatter and a focused body.
4. **Add resources** only if needed; keep each reference file small. If the skill
   needs its own capability, prefer `scripts/` + `exec_command`; add a `tools.py`
   (see "Native tools (kia extension)") only when the tool must run in-process.
5. **Validate** the name/description rules below, then confirm with the user.

## Validation checklist

- [ ] `SKILL.md` begins with a `---` frontmatter block.
- [ ] `name` present, kebab-case, ≤64 chars, and equals the directory name.
- [ ] `description` present, non-empty, ≤1024 chars, says what + when.
- [ ] Body under ~500 lines; long material moved to `references/`.
- [ ] File references are relative and one level deep.
- [ ] If `tools.py` is present: each `TOOLS` entry has `schema` + `run`; tool names do not shadow built-ins; `permission` is set correctly (`safe` for read-only, `risky` otherwise); the `SKILL.md` body mentions the tools it enables.
- [ ] If `tools.py` is present: it is self-contained — only absolute imports of installed packages (`kiui.agent.*`, numpy, …), no imports of sibling files in the skill folder.
- [ ] Run `/skills reload` in kia to pick up the new skill, then `/skills` to confirm it is discovered without warnings (or `/skills <name>` to load it manually). After loading a skill with `tools.py`, confirm its tools appear and are callable.
