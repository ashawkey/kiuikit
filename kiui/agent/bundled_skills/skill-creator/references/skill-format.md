# Agent Skill Format

Use this reference when creating or restructuring a skill. The standard format
is documented at https://agentskills.io/specification.

## Layout

```text
<skill-name>/
├── SKILL.md          # required
├── tools.py          # optional kia extension
├── scripts/          # optional executable helpers
├── references/       # optional documentation
└── assets/           # optional templates and static files
```

Create project skills under `.kia/skills/<skill-name>/` unless the user requests
a different location.

## SKILL.md

```markdown
---
name: pdf-processing
description: Use this skill when the user needs to extract PDF text or tables, fill PDF forms, or merge PDF documents.
---

# PDF Processing

Operational instructions go here.
```

| Field | Required | Rules |
|---|---:|---|
| `name` | yes | 1–64 lowercase letters, digits, and single hyphens; no leading, trailing, or consecutive hyphens; must equal the directory name. |
| `description` | yes | 1–1024 characters; state what the skill enables and when to use it. |
| `license` | no | License name or bundled license-file reference. |
| `compatibility` | no | Up to 500 characters for environment or dependency requirements. |
| `metadata` | no | String-to-string mapping. |
| `allowed-tools` | no | Space-separated pre-approved tools; parsed but not enforced by kia. |

Only names and descriptions are advertised before skills load, so the
description is the primary activation signal.

## Direct invocation in kia

A discovered skill can be invoked explicitly with `/<skill-name>` or
`/<skill-name> <task context>`. This is a kia host convention, not an Agent
Skills frontmatter extension.

When a no-argument invocation should perform a specific workflow, document it in
an optional `## Default invocation` section. Define the outcome, how inputs are
discovered, permitted actions, verification, stopping conditions, and behavior
when required information is missing. If the skill is a general capability and
has no safe, unambiguous default task, omit that section: kia will load the
skill and ask the user what to do. An ordinary `## Workflow` section does not by
itself declare a default invocation.

With trailing task context, kia loads the skill and applies its instructions to
that request. Built-in slash commands take precedence over skills with the same
name. Skill names therefore use kebab-case, for example `/monitor-jobs`.

## Progressive disclosure

Keep `SKILL.md` focused and generally below 500 lines. Put conditional detail in
small reference files and state when to read each one:

```markdown
Read references/api-errors.md if the API returns a non-success response.
Run scripts/validate.py before producing the final output.
```

Resolve resource paths against the absolute skill directory shown when kia loads
the skill.

Scripts should be non-interactive, have clear inputs and errors, and document
non-standard dependencies. Use assets for content copied or consumed by path.

Mechanical validation only checks packaging and loadability. Verify factual
accuracy through source inspection and a representative run.
