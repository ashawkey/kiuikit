---
name: skill-creator
description: Create, structure, and validate new Agent Skills (SKILL.md packs). Use this when the user wants to author a new skill, package reusable domain knowledge or a repeatable workflow, add bundled scripts/references/assets, or fix an existing skill so it follows the Agent Skills format.
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
├── scripts/          # optional: executable code the agent runs (Bash/exec)
├── references/       # optional: docs the agent reads on demand
└── assets/           # optional: templates / data files referenced by path
```

Create the skill under `.kia/skills/<skill-name>/` (kia's own dir) unless the
user asks for a different location. `.codex/skills/`, `.claude/skills/`, and
`.agents/skills/` are also discovered for cross-agent compatibility.

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
4. **Add resources** only if needed; keep each reference file small.
5. **Validate** the name/description rules below, then confirm with the user.

## Validation checklist

- [ ] `SKILL.md` begins with a `---` frontmatter block.
- [ ] `name` present, kebab-case, ≤64 chars, and equals the directory name.
- [ ] `description` present, non-empty, ≤1024 chars, says what + when.
- [ ] Body under ~500 lines; long material moved to `references/`.
- [ ] File references are relative and one level deep.
- [ ] Run `/skills reload` in kia to pick up the new skill, then `/skills` to confirm it is discovered without warnings (or `/skills <name>` to load it manually).
