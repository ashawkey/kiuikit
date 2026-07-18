# kiui.agent

A lightweight, terminal-based AI agent that can browse the web, read/write files, and execute shell commands.

## Features

- **File Operations**: Read, write, and surgically edit files with syntax-highlighted diffs.
- **Shell Execution**: Run arbitrary shell commands with real-time streaming output.
- **Web Capabilities**: Search the web and fetch/parse webpage content.
- **Tool-use**: Automatically chooses the right tool for the task.
- **Streaming**: Responses render token-by-token in both the terminal and Web UI, with reasoning/thinking stream shown automatically for compatible models.
- **Sub-agents**: Can spawn sub-agents to handle complex sub-tasks in-process.
- **Skills**: Load domain-specific instructions via customizable skill packs (`.kia/skills/`).
- **Context Management**: Proactive tool-output filtering, automatic pruning, and LLM-based compaction keep context focused.
- **Rewind**: Roll back conversation and/or code changes to any previous round.
- **Permissions**: Three confirmation modes (auto / default / strict) and defense-in-depth detection of common destructive shell commands.
- **Model Switching**: Hot-swap between configured models mid-session.
- **Interactive**: Rich terminal interface with syntax highlighting, file-path autocomplete, and progress indicators.
- **Remote Web UI**: Optional authenticated, mobile-friendly companion synchronized with the terminal. A shared hub multiplexes many terminal agents into one page (a tab per agent) behind a single port.

## Installation

```bash
pip install kiui[kia]
```

## Configuration

The agent uses a YAML configuration file located at `./.kiui.yaml` (current directory) or `~/.kiui.yaml` (home directory). You need to define your model profiles under the `openai` key (even for non-OpenAI providers, as long as they provide an OpenAI-compatible API).

Example `.kiui.yaml`:

```yaml
openai:
  gpt: # model_alias, convenient name to use in the CLI
    model: gpt-4o # actual model name used in the API
    api_key: sk-proj-...
    base_url: https://api.openai.com/v1

kia_web_token: web-secret # optional Web UI access token
kia_lib: git@github.com:username/kia-skills.git # optional personal skill library
```

## Usage

### List available models

```bash
kia --list
```

### Start an interactive chat

```bash
kia --model <model_alias>
```

Simply running `kia` (with no arguments) starts a chat with the first configured model.

### Additional options

```bash
kia --model <model_alias> --verbose --perm strict --resume [session_id]
```

| Flag | Description |
|------|-------------|
| `--model` | Model alias from config (default: first configured) |
| `--verbose` | Enable verbose debug output |
| `--stream` / `--no-stream` | Stream the response token-by-token as it is generated (default: on) |

| `--perm MODE` | `auto` (default), `default`, or `strict` |
| `--resume [SESSION_ID]` | Resume a session (bare `--resume` lists saved sessions interactively) |
| `--list` | List available models with context-window info and exit |
| `--hub` | Run the shared web hub daemon (owns the public port) |
| `--web-port PORT` | Hub listener port (default: `8765`) |

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
| `/help` | Show help message |
| `/context` | Show a concise one-line-per-message context log |
| `/system_prompt` | Print the current full system prompt |
| `/compact` | Force context compaction via LLM summarization |
| `/usage` | Show token usage for this session |
| `/perm [auto\|default\|strict]` | Show or change permission mode |
| `/model [name]` | Show or switch LLM model mid-session |
| `/rewind [round]` | Roll back conversation and/or code to a previous round |
| `/skills` | List installed skills; `/skills reload` to re-scan; `/skills <name>` to load one |
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
| `auto` | All tools run without prompting (useful for sub-agents / pipe mode) |
| `default` | Risky tools (write, edit, remove, exec, spawn) prompt for confirmation |
| `strict` | Every tool call prompts for confirmation |

The permission layer also provides:
- File tools are not workspace-confined. `auto` runs safety-approved calls without prompting; `default` and `strict` apply the tool-level confirmation rules above.
- Defense-in-depth detection of common destructive shell commands (`rm -rf /`, `mkfs`, device writes, fork bombs, shutdown/reboot, etc.), including direct `!` commands.

Shell detection is heuristic, not a security boundary: arbitrary shell syntax can evade static pattern matching. Use an OS-level sandbox or container when commands must be contained.

## Context Management

The agent automatically manages context window usage through three layers:

1. **Proactive tool-result compaction** — before a result enters conversation history, oversized output is reduced with a tool-aware pipeline. Shell results use semantic reducers for pytest, git, Python package tools, and compiler diagnostics before falling back to ANSI/progress cleanup, repeated-line collapse, and diagnostic-preserving edge samples. File reads retain contiguous prefixes, while search/list results use weighted edge samples. Full command output is teed during execution and saved privately under `.kia/tool-results/<session>/`, retained for up to seven days within a 100 MB project cap. Compact results include estimated token savings, reducer provenance, confidence tier, and a focused recovery path.
2. **Context pruning** — old tool results from read/exec/web tools are trimmed (head+tail) at 30% usage, then hard-cleared at 50%.
3. **LLM compaction** — when context exceeds 75%, the oldest messages are summarized via an LLM call and replaced with a compact summary.

Deterministic ingress compaction is preferred over automatically spawning a summarization sub-agent: it has no extra latency or model cost, cannot hallucinate away exact errors, and still preserves searchable access to the captured output.

Live context/token usage is shown inline in the `Working... (Xs)` status bar (terminal and web UI), in addition to the full `/usage` breakdown.

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

Usage:

| Form | Effect |
|------|--------|
| `/goal <text>` | Set a new goal and start auto-iterating |
| `/goal` | Show the current goal and status |
| `/goal clear` | Clear the goal and stop iterating |

There is **no fixed iteration cap** — the loop runs until the goal is reported met or you stop it. Since the terminal prompt is blocked while the loop runs, **`Ctrl+C` / `Esc` during a round is the way to stop it**, which clears the goal. Goals are saved with the session and **auto-resume** on `--resume`.

## Skills

Skills are modular prompt packs following the open [Agent Skills](https://agentskills.io) format, stored in `.kia/skills/<name>/SKILL.md`. Each skill provides domain-specific instructions the model can load on demand via the `load_skill` tool.

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

Skills are discovered from `.kia/skills/` under **both the project directory and your home directory** (`~/.kia/skills/`), so you can keep personal skills that follow you across projects. Project skills take precedence over personal ones. Other agents' skill directories are not scanned; when needed, give kia a skill path explicitly so it can read the instructions.

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
kib list                         # list remote names and descriptions
kib list --local                 # list skills in ./.kia/skills
kib install <name>               # install into ./.kia/skills/<name>
kib upload <name>                # upload from ./.kia/skills/<name>
kib remove <name>                # remove a remote skill
kib upload <name> --force        # update an existing remote skill
```

Remote skills are not loaded or advertised to the agent until installed.
The repository is cached under `~/.kia/library/`; each configured URL has an
isolated checkout, so changing `kia_lib` selects a different cache. Existing
local skills are never overwritten. Bundled skills shipped with kia cannot be
uploaded. Upload validates the skill, rejects symlinks, creates a normal commit,
and never force-pushes. An empty repository
is initialized on the first upload.

### Bundled skills

kia ships a few common skills (currently `skill-creator`, which teaches the agent how to author new spec-compliant skills). On first run in a project, these are copied into `.kia/skills/` so they are available out of the box and remain fully editable. An existing skill of the same name is never overwritten, so your edits are preserved; delete a copy to have it reinstalled on the next run.

## Sessions

Sessions are automatically saved after each round (throttled to ≤ once per 30 seconds) to `.kia/sessions/<session_id>.json`. Use `--resume` to pick up where you left off, or `/clear` to start a fresh session.

## Tools

The agent has access to the following tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with optional offset/limit |
| `write_file` | Create or overwrite files, creating parent directories |
| `edit_file` | Surgical text replacement in files (whitespace-tolerant match) |
| `multi_edit` | Apply an ordered batch of edits to one file atomically (all-or-nothing) |
| `ls` | List a directory's immediate contents (gitignore-aware) |
| `exec_command` | Run shell commands with real-time streaming output |
| `glob_files` | Find files matching a glob pattern (gitignore-aware) |
| `grep_files` | Search file contents using regex (prefers ripgrep; gitignore-aware) |
| `web_search` | Search the web via DuckDuckGo |
| `web_fetch` | Fetch and parse content from a URL |
| `remove_file` | Remove a file or directory |
| `spawn_subagent` | Delegate a task to a new in-process agent instance |
| `load_skill` | Load the full prompt instructions for a skill by name |
