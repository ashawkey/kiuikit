---
name: library
description: Manage reusable kia skills and personas in the kib Git-backed library. Use when the user wants to list, find, install, synchronize, upload, publish, update, or remove a library skill or persona.
---

# Resource Library

Run `kib` with `exec_command` from the project working directory. Skills are the default resource kind; add `--kind persona` for personas.

## Workflow

1. Determine the resource kind, names, and requested operation. Use `kib list [pattern]` when the user is searching or the exact remote name is unknown.
2. Run the narrowest matching command below. Do not manually edit library Git state or committed `.kib.json` metadata.
3. Check the command result before continuing. If `kib update` reports changes on both sides, use `--prefer local` or `--prefer remote` only when the user's intended winner is clear; otherwise ask.
4. Report the affected resources and outcome. After an install or update, reload the changed resource in an active kia session.

## Commands

```text
kib list [pattern] [--kind skill|persona]
kib list [pattern] --local [--kind skill|persona]
kib install <name> [<name> ...] [--kind skill|persona]
kib update [<name> ...] [--kind skill|persona]
kib update <name> --prefer local|remote [--kind skill|persona]
kib upload <name> [<name> ...] [--kind skill|persona]
kib upload <name> --force [--kind skill|persona]
kib remove <name> [<name> ...] [--kind skill|persona]
kib remove <name> --local [--kind skill|persona]
```

Use `kib --verbose <command> ...` only when normal output is insufficient to diagnose Git, authentication, or synchronization failures.

## Operation rules

- `list` shows remote resources and descriptions; `--local` lists project resources. Remote entries indicate installed status.
- `install` writes into `./.kia/skills/` or `./.kia/personas/` and never overwrites an existing local resource. Use `update` for an installed copy.
- `update` synchronizes all installed resources of the selected kind when no names are given. Its committed `.kib.json` base lets it detect independent local and remote changes rather than guessing.
- `upload` publishes project resources. Use `--force` only when replacing an existing remote resource is intended.
- `remove` deletes remote resources; `--local` deletes only project copies.
- The library repository is configured by `kia_lib` in `.kiui.yaml` and uses the current Git/SSH authentication environment.

After installing or updating a skill, tell the user to run `/skills reload`; after changing a persona, tell them to run `/persona reload`. Then load an installed skill when it matches the current task.
