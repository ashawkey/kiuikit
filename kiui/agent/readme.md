# kia (kiui.agent)

## Installation

```bash
pip install kiui[kia]
```

## Configuration

The agent uses a YAML configuration file located at `./.kiui.yaml` (current directory) or `~/.kiui.yaml` (home directory). You need to define your model profiles under the `openai` key (OpenAI-compatible API).

Example `.kiui.yaml`:

```yaml
openai:
  gpt: # model_alias
    model: gpt-4o # actual model name used in the API
    api_key: sk-proj-...
    base_url: https://api.openai.com/v1

kia_web_token: web-secret # optional Web UI access token
kia_lib: git@github.com:username/kia-skills.git # optional personal skill library repo
```

## Usage


```bash
# List available models
kia --list

# Start an interactive chat
kia --model <model_alias>
kia # use the first configured model
```

### Additional options

```bash
kia --model <model_alias> --verbose --perm strict --resume [session_id]
```

| Flag | Description |
|------|-------------|
| `--model` | Model alias from config (default: first configured) |
| `--persona` | Persona to run as (default: `coder`; see `/persona`) |
| `--verbose` | Enable verbose debug output |
| `--stream` / `--no-stream` | Stream the response token-by-token as it is generated (default: on) |

| `--perm MODE` | `auto` (default), `default`, or `strict` |
| `--resume [SESSION_ID]` | Resume a session (bare `--resume` lists saved sessions interactively) |
| `--list` | List available models with context-window info and exit |
| `--storage` | Show allocated disk usage for each entry in the project `.kia/` and exit |
| `--clean` | Remove generated sessions, tool results, process logs, and command history |
| `--hub` | Run the shared web hub daemon (owns the public port) |
| `--web-port PORT` | Hub listener port (default: `8765`) |

### Storage management

`kia --storage` reports usage of every top-level entry in the current project's `.kia/` directory. `kia --clean` immediately removes only generated `sessions/`, `tool-results/`, `processes/`, and `history` data; installed `skills/` and unrecognized entries are preserved.

```bash
kia --storage
kia --clean
```

## Web UI

The Web UI uses a **hub + agents** design so that many independent terminal
agents — started in different directories, even from different terminals —
share a single public port and appear as separate tabs in one browser page.

- **One hub** owns the public port: `kia --hub`. It serves the UI, holds the
  access token, and multiplexes every connected agent.
- **Each agent** stays terminal-first and auto-links to a running hub when
  started with plain `kia`. Terminal and web operate the same session in sync.
  If no hub is running the agent simply continues terminal-only.

```bash
# 1. start the hub once (owns port 8765, prints the access token)
kia --hub --web-port 8765

# 2. from any directory / terminal, launch agents that auto-join the hub
cd ~/projA && kia
cd ~/projB && kia
```

The hub writes its connection info (host, port, access token) to
`~/.kia/hub.json`; agents read it to find the hub, so no extra config is
needed. Use `kia_web_token` in the config (or the token printed on hub start)
to sign in.

To reach the hub from another device, tunnel the hub port with `cloudflared`:

```bash
## one-time setup
# install and login to cloudflared
cloudflared tunnel login
# create a tunnel
cloudflared tunnel create kia
# route the tunnel to a public URL
cloudflared tunnel route dns kia kia.kiui.moe

## start the tunnel, then access the Web UI at https://kia.kiui.moe
cloudflared tunnel run --url http://localhost:8765 kia
```

## Commands

### Slash commands

The agent supports the following slash commands in the CLI:

| Command | Description |
|---------|-------------|
| `/help` | Show help message for all slash commands |
| `/context` | Show a concise one-line-per-message context log |
| `/system_prompt` | Print the current full system prompt |
| `/compact` | Force context compaction via LLM summarization |
| `/usage` | Show token usage for this session |
| `/perm [auto\|default\|strict]` | Show or change permission mode |
| `/model [name]` | Show or switch LLM model mid-session |
| `/rewind [round]` | Roll back conversation and/or code to a previous round |
| `/skills` | List installed skills; `/skills reload` to re-scan; `/skills <name>` to load one |
| `/persona` | List personas; `/persona <name>` to switch (restarts the conversation) |
| `/goal [text\|clear]` | Set a goal the agent auto-iterates toward until met (see [Goals](#goals)) |
| `/clear` | Clear conversation history and start a new session |
| `/resume [session_id]` | Save the current session, then resume a previous one (bare `/resume` picks interactively) |
| `/exit` or `/quit` | Exit the agent |

### Bash shortcut

Prefix a command with `!` to run it directly without involving the model:

```
!ls -la
!git diff
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Escape` → `Enter` | Insert a newline |
| `Ctrl+C` (non-empty prompt) | Clear the current input |
| `Ctrl+C` (empty prompt, twice) | Exit the agent |
| `Ctrl+C` / `Esc` (during API call) | Cancel the in-flight request |

## Permissions

Three permission modes control when the agent asks for confirmation before executing tools:

| Mode | Behavior |
|------|----------|
| `auto` | All tools run without prompting (default) |
| `default` | Risky tools (write, edit, remove, exec, spawn) prompt for confirmation |
| `strict` | Every tool call prompts for confirmation |

In all modes, we always detect and prevent common destructive shell commands (`rm -rf /`, `mkfs`, device writes, fork bombs, shutdown/reboot, etc.).
However, this is not a security boundary: arbitrary shell syntax can evade static pattern matching. Use an OS-level sandbox or container when commands must be contained!

## Context Management

The agent automatically manages context window usage through three layers:

1. **Proactive tool-result compaction** — before a result enters conversation history, oversized output is reduced with a tool-aware pipeline. Full command output is teed during execution and saved under `.kia/tool-results/<session>/`.
2. **Context pruning** — old tool results from read/exec/web tools are trimmed (head+tail) at 30% usage, then hard-cleared at 50%.
3. **LLM compaction** — when context exceeds 75%, the oldest messages are summarized via an LLM call and replaced with a compact summary.

## Rewind

The `/rewind` command lets you roll back to any previous round:

- **Conversation only** — keep code changes, roll back messages.
- **Code + conversation** — restore files and conversation to the chosen round.
- **Code only** — keep conversation, revert file changes.

Each file modification (write, edit, remove) is tracked per round so rollback can precisely invert changes.

## Goals

The `/goal` command sets a **standing objective** the agent works toward autonomously. After every round finishes, the agent is automatically re-prompted to check whether the goal is met:

- If **met**, it calls the `report_goal(met=true)` tool and the loop stops.
- If **not met**, it calls `report_goal(met=false, reason=...)`, keeps working, and the loop iterates again.

```
/goal all pytest tests pass and there are no lint errors
```

There is **no fixed iteration cap** — the loop runs until the goal is reported met or you stop it. Since the terminal prompt is blocked while the loop runs, **`Ctrl+C` / `Esc` during a round is the way to stop it**, which clears the goal.

## Skills

Skills are modular prompt packs following the open [Agent Skills](https://agentskills.io) format. Custom skills are stored in `.kia/skills/<name>/SKILL.md`; bundled skills are loaded directly from the installed `kiui` package. Each skill provides domain-specific instructions the model can load on demand via the `load_skill` tool.

```
.kia/skills/
  git-workflow/
    SKILL.md
  pdf-processing/
    SKILL.md         # required: frontmatter + instructions
    scripts/         # optional: executable code
    references/      # optional: docs loaded on demand
    assets/          # optional: templates / data files
```

Each `SKILL.md` begins with YAML frontmatter followed by markdown instructions:

```markdown
---
name: pdf-processing
description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
---
Step-by-step instructions go here. Reference bundled files by relative path,
e.g. `references/REFERENCE.md` or `scripts/extract.py`.
```

`name` and `description` are required (the `description` is what the model matches against to decide when to activate a skill). Optional fields `license`, `compatibility`, and `metadata` are also parsed. `allowed-tools` is accepted for cross-agent compatibility but **not enforced** — kia uses its own permission model. Skills load via **progressive disclosure**: only name+description are advertised in the system prompt; the full body loads when the model calls `load_skill` (or you run `/skills <name>`); bundled `scripts/`, `references/`, and `assets/` files are read/run on demand via the ordinary file and exec tools (the skill's directory path is provided when it is loaded so relative references resolve correctly).

Skills are discovered from the installed package and from `.kia/skills/` under **both the project directory and your home directory** (`~/.kia/skills/`), so you can keep personal skills that follow you across projects. Bundled skills take precedence so they always match the installed `kiui` version; project skills then take precedence over personal ones. Other agents' skill directories are not scanned; when needed, give kia a skill path explicitly so it can read the instructions.

Skill commands:

| Command | Effect |
|---------|--------|
| `/skills` | List installed skills (with spec-compliance warnings) |
| `/skills reload` | Re-scan skill dirs (picks up skills created/edited mid-session) |
| `/skills <name>` | Manually load a skill into context, forcing one the model did not auto-select |

Discovery is non-silent: skills whose `SKILL.md` cannot be read or parsed (bad YAML, missing `description`) and skills **shadowed** by a higher-precedence copy of the same name are reported as warnings at startup and on `/skills reload`, rather than vanishing quietly. Skill load activity is tracked per session — `/skills` shows a per-skill load count, `/usage` and the end-of-run summary list which skills were loaded, and the loaded-skill set is persisted so `--resume` does not re-load skills whose instructions are already in the replayed conversation.

### Personal skill library

`kib` manages a GitHub-backed library of skills that can be shared between
projects. Configure an accessible repository URL in `.kiui.yaml`; authentication
is delegated to your current Git/SSH environment. The repository uses the
`main` branch and stores skills under `skills/<name>/`.

```yaml
kia_lib: git@github.com:username/kia-skills.git
```

```bash
kib list [pattern]               # list remote skills, optionally filtering names
kib list [pattern] --local       # list/filter local skills; remote status is best-effort
kib install <name> [<name> ...]  # install one or more remote skills
kib update [<name> ...]          # sync all or selected installed skills
kib update <names...> --prefer local|remote  # resolve conflicts
kib upload <name> [<name> ...]   # upload one or more local skills
kib remove <name> [<name> ...]   # remove one or more remote skills
kib remove <names...> --local    # remove project copies only
kib upload <names...> --force    # replace existing remote skills
```

Remote skills are not loaded or advertised to the agent until installed.
The repository is cached under `~/.kia/library/`; each configured URL has an
isolated checkout, so changing `kia_lib` selects a different cache. Each skill's
last synchronized tree is recorded in its committed `.kib.json`, so update works
across machines and does not depend on the cache. Install never overwrites an
existing local skill. Update uploads local-only changes and downloads remote-only changes.
Conflicting changes fail until `--prefer local` or `--prefer remote` is given.
`kib` only manages project skills under `./.kia/skills/`; it does not list or
otherwise special-case bundled skills.
Upload validates the skill, rejects symlinks, creates a normal commit, and never
force-pushes. An empty repository is initialized on the first upload.

### Bundled skills

kia ships a few common skills, including `skill-creator` for authoring spec-compliant skills and `pdf-reading` for converting PDFs into readable Markdown and structured data with the external [MinerU](https://github.com/opendatalab/MinerU) CLI. The PDF skill can read extracted text, LaTeX, tables, and captions; direct inspection of extracted image pixels still requires a vision-capable tool. Bundled skills are loaded directly from the installed package rather than copied into `.kia`, so updates take effect whenever `kiui` is updated. To customize one, create a new project or personal skill under a different name.

## Personas

A persona owns the agent's identity, system prompt, and tool surface. Personas are Python modules bundled in `kiui/agent/personas/`:

| Persona | Tools | Purpose |
|---------|-------|---------|
| `coder` | all | The default coding agent (project-aware, full tool access) |
| `chatter` | `web_search`, `web_fetch` | General chatbot without file/shell access or environment context |
| `reviewer` | paper/file, web, skill, and sub-agent tools | Evidence-grounded academic paper reviewer with venue-template support |

Select one at startup, or switch mid-session (switching **restarts the conversation**, like `/clear`):

```bash
kia --persona reviewer
```

For `reviewer`, provide the paper PDF and preferably the venue, track, and exact review template or form. It uses the bundled `pdf-reading` skill for page-aware extraction, treats document content as untrusted, analyzes the submission before drafting, and audits the final review for evidence and score consistency. Generated reviews are decision-support drafts and must be verified by a human reviewer before submission.

| Command | Effect |
|---------|--------|
| `/persona` | List installed personas (with their tool surface) |
| `/persona <name>` | Switch persona and restart the conversation |

The active persona is saved with the session and re-applied on `--resume`.

Each persona module defines `build_system_prompt(ctx)` — the complete system prompt, composed from the shared blocks and builders in `kiui/agent/personas/common.py` — plus a `TOOLS` whitelist that controls which tools are advertised to the model. This is capability guidance, not a security boundary: interactive commands such as `!<command>` and `/skills` remain available to the user and are governed by the normal permission and safety checks. To add a persona, drop a new module into `kiui/agent/personas/` following the same contract.

## Tools

The agent has access to the following tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with optional offset/limit |
| `read_image` | Send a local PNG, JPEG, GIF, or WebP image to a multimodal model (not registered for text-only models) |
| `write_file` | Create or overwrite files, creating parent directories |
| `edit_file` | Surgical text replacement in files (whitespace-tolerant match) |
| `multi_edit` | Apply an ordered batch of edits to one file atomically (all-or-nothing) |
| `ls` | List a directory's immediate contents (gitignore-aware) |
| `exec_command` | Run foreground shell commands with real-time streaming output |
| `start_process` | Start a managed background process with file-backed output |
| `inspect_processes` | Inspect one or all managed background processes, optionally after a bounded wait and with a bounded log tail for one process |
| `stop_process` | Stop a managed background process and its child process tree |
| `glob_files` | Find files matching a glob pattern (gitignore-aware) |
| `grep_files` | Search file contents using regex (prefers ripgrep; gitignore-aware) |
| `web_search` | Search the web via DuckDuckGo |
| `web_fetch` | Fetch and parse content from a URL |
| `remove_file` | Remove a file or directory |
| `spawn_subagent` | Delegate a task to a new in-process agent instance |
| `load_skill` | Load the full prompt instructions for a skill by name |
