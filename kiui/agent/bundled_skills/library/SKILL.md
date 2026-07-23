---
name: library
description: Manage reusable skills and personas with the kib Git-backed personal library. Use when the user asks to list, install, upload, update, or remove a library skill or persona.
---

# Resource Library

Use `kib` through `exec_command` from the project working directory. Commands operate on skills by default; add `--kind persona` to operate on personas.

## Commands

- `kib list [pattern] [--kind skill|persona]` — list remote resources and descriptions. Installed resources are marked.
- `kib list [pattern] --local [--kind skill|persona]` — list project resources.
- `kib install <name> [<name> ...] [--kind skill|persona]` — install remote resources into `./.kia/skills/` or `./.kia/personas/`.
- `kib update [<name> ...] [--kind skill|persona]` — synchronize all installed resources of one kind or only the named resources.
- `kib update <name> --prefer local|remote [--kind skill|persona]` — resolve conflicting changes by keeping one side.
- `kib upload <name> [<name> ...] [--kind skill|persona]` — upload project resources.
- `kib upload <name> --force [--kind skill|persona]` — update an existing remote resource.
- `kib remove <name> [<name> ...] [--kind skill|persona]` — remove remote resources.
- `kib remove <name> --local [--kind skill|persona]` — remove project copies without accessing the library.
- `kib --verbose <command> ...` — show operation progress and Git details.

The library repository comes from `kia_lib` in `.kiui.yaml` and uses the current Git/SSH authentication environment. Install never overwrites an existing local resource; use update to synchronize it. Update records the last common tree in each resource's committed `.kib.json` and refuses to guess when both copies changed.

After installing or updating a skill during an active session, tell the user to run `/skills reload`. After changing a persona, tell them to run `/persona reload`. Then use `load_skill` when the task matches an installed skill.
