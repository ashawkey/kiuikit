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
- `kib update [<name> ...]` — synchronize all installed skills or only the named skills, uploading local-only changes and downloading remote-only changes.
- `kib update <name> --prefer local|remote` — resolve conflicting changes by keeping one side.
- `kib upload <name> [<name> ...]` — upload one or more project skills from `./.kia/skills/`.
- `kib upload <name> [<name> ...] --force` — update existing remote skills.
- `kib remove <name> [<name> ...]` — remove one or more skills from the remote library.
- `kib remove <name> [<name> ...] --local` — remove one or more project copies without accessing the library.
- `kib --verbose <command> ...` — show operation progress, Git commands, output, exit status, and timing details for troubleshooting.

The library repository comes from `kia_lib` in `.kiui.yaml` and uses the current Git/SSH authentication environment. Do not overwrite an existing local skill with `kib install`; use `kib update` to synchronize it. Update tracks the last common copy in each skill's committed `.kib.json`, works across machines, and refuses to guess when both local and remote changed.

After installing or updating a skill during an active kia session, tell the user to run `/skills reload` so the agent discovers the change. Then use `load_skill` when the task matches the skill.
