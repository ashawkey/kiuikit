---
name: skill-library
description: Manage reusable skills with the kib Git-backed personal skill library. Use when the user asks to list available or installed library skills, install a remote skill, upload or update a project skill, or remove a remote skill.
---

# Skill Library

Use `kib` through `exec_command` from the project working directory.

## Commands

- `kib list [pattern]` — list remote skills and descriptions, optionally filtering names by substring. Installed skills are marked.
- `kib list [pattern] --local` — list project skills, optionally filtering names by substring. Uploaded status is checked when a library is configured.
- `kib install <name> [<name> ...]` — install one or more remote skills into `./.kia/skills/`.
- `kib upload <name> [<name> ...]` — upload one or more project skills from `./.kia/skills/`.
- `kib upload <name> [<name> ...] --force` — update existing remote skills.
- `kib remove <name> [<name> ...]` — remove one or more skills from the remote library.

The library repository comes from `kia_lib` in `.kiui.yaml` and uses the current Git/SSH authentication environment. Do not overwrite an existing local skill; `kib install` rejects that case.

After installing a skill during an active kia session, tell the user to run `/skills reload` so the agent discovers it. Then use `load_skill` when the task matches the newly installed skill.
