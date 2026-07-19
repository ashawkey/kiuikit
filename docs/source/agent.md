# `kia` — an AI agent for your terminal

[[source]](https://github.com/ashawkey/kiuikit/tree/main/kiui/agent) · `pip install "kiui[kia]"`

`kia` is a lightweight coding agent that works where you do: in the terminal and inside your project. It can inspect a repository, edit files, run commands, search the web, delegate work, and keep long sessions focused. Bring any model served through an OpenAI-compatible API.

```text
$ kia --model gpt
> find the cause of this failing test, fix it, and run the smallest relevant check

● reads the repository and test output
● edits only the affected files
● runs the focused test
✓ explains what changed and what was verified
```

## Features

| | |
|---|---|
| **Agentic coding**<br>Reads, searches, edits, and runs commands with live output. | **Model agnostic**<br>Works with OpenAI-compatible APIs and switches models without leaving the session. |
| **Terminal + Web UI**<br>A polished Rich terminal experience with an optional synchronized, mobile-friendly browser UI. | **Sub-agents and goals**<br>Delegates independent research or iterates autonomously toward a standing objective. |
| **Skills**<br>Loads reusable [Agent Skills](https://agentskills.io) only when relevant, keeping the base prompt lean. | **Long-context management**<br>Compacts large tool results, prunes old output, and summarizes history as context fills. |
| **Rewind and sessions**<br>Resumes previous work or rolls conversation and file changes back to an earlier round. | **Human control**<br>Offers three confirmation modes plus a hard guard against recognized destructive operations. |
| **Web access**<br>Searches current information and turns web pages into readable text. | **Developer-friendly UI**<br>Streams responses, shows diffs and token usage, and autocompletes file references with `@`. |

> **One agent, two interfaces.** Start work in the terminal, open the optional Web UI from another device, and interact with the same live session from either place.

## Install

Install the agent and its optional dependencies from PyPI:

```bash
pip install "kiui[kia]"
```

For the latest development version:

```bash
pip install -U "kiui[kia] @ git+https://github.com/ashawkey/kiuikit.git"
```

This installs two commands:

- `kia` — the coding agent and shared Web UI hub.
- `kib` — the optional Git-backed personal skill library.

## Configure a model

Create `.kiui.yaml` in the current project or `~/.kiui.yaml` for a global configuration. A project configuration takes precedence.

```yaml
openai:
  gpt:                         # short alias used by kia
    model: gpt-5               # model ID sent to the provider
    api_key: sk-...
    base_url: https://api.openai.com/v1
    reasoning_effort: high     # optional

  local:
    model: my-model
    api_key: not-needed
    base_url: http://localhost:8000/v1

kia_web_token: change-me       # optional fixed Web UI token
```

The provider only needs to expose an OpenAI-compatible chat-completions API. Keep API keys out of files committed to version control.

List the configured profiles and their detected capabilities:

```bash
kia --list
```

## Quick start

Run `kia` from the root of the project you want it to work on:

```bash
cd my-project
kia --model gpt
```

If `--model` is omitted, the first profile in `openai` is used.

Talk to it naturally:

```text
> explain how authentication flows through this repository
> fix the parser bug reported in issue #42 and run its tests
> compare these two approaches using current upstream documentation
> @src/server.py simplify this function without changing its API
```

Prefix a command with `!` to bypass the model and run it directly:

```text
> !git diff --stat
> !pytest tests/test_parser.py -q
```

Type `@` to autocomplete a project path. Responses, reasoning for compatible models, tool activity, command output, and diffs stream in the terminal.

## A workflow built for real repositories

### Inspect and change code

The agent has focused tools for reading files, searching by glob or regular expression, listing directories, applying surgical edits, and batching atomic edits. It prefers narrow repository-aware operations over dumping an entire project into context.

### Run and verify

Shell commands stream in real time. Large output is compacted before entering model context, while the complete captured output remains available under `.kia/tool-results/` for focused follow-up. Common pytest, Git, package-manager, and compiler output receives tool-aware reduction so useful diagnostics survive.

### Delegate independent work

The agent can spawn an in-process sub-agent for a self-contained task. The child works autonomously with the same model profile and returns a concise result to the parent session.

```text
> spawn a sub-agent to review the database migration for data-loss risks
```

### Set a standing goal

Use `/goal` when the agent should keep checking and working until an objective is met:

```text
> /goal all focused tests pass and the changed files have no lint errors
```

There is no fixed iteration cap. Press `Ctrl+C` or `Esc` during a round to stop the loop. Use `/goal clear` to remove the objective.

### Rewind safely

`/rewind` opens an interactive history picker. Choose a round, then restore:

1. conversation only,
2. conversation and code, or
3. code only.

File writes, edits, and removals are tracked per round, so an experiment does not have to become a permanent change.

## Web UI

`kia` remains terminal-first, but a shared hub can mirror many agents into one authenticated browser UI. Each running agent appears as a tab, even when agents were started in different projects or terminals.

```text
┌──────────────────────────────┐       WebSocket       ┌─────────────────────┐
│ kia --hub                    │◀──────────────────────▶│ browser             │
│  ├─ project-a · gpt          │                       │  [project-a] [docs] │
│  └─ docs · local             │                       └─────────────────────┘
└──────────────▲───────────────┘
               │ local agent links
        ┌──────┴──────┐
        │ kia sessions │
        └──────────────┘
```

Start the hub once, then launch agents normally:

```bash
# terminal 1: owns the browser UI port
kia --hub --web-port 8765

# other terminals: automatically discover and join the hub
cd ~/project-a && kia --model gpt
cd ~/project-b && kia --model local
```

Open the URL printed by the hub and sign in with its generated token, or set `kia_web_token` in `.kiui.yaml`. Connection information is stored in `~/.kia/hub.json`; agents continue terminal-only if no hub is available.

The hub listens on loopback. To use it from another machine, forward the port with SSH or place it behind an authenticated tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 user@workstation
```

Then open `http://127.0.0.1:8765` locally.

## Skills

Skills are reusable instruction packs in the open [Agent Skills](https://agentskills.io) format. `kia` discovers them in both:

- `./.kia/skills/` — project skills; these take precedence.
- `~/.kia/skills/` — personal skills shared across projects.

Only each skill's name and description enter the base prompt. Full instructions and bundled resources are loaded on demand, preserving context for the actual task.

```text
.kia/skills/pdf-processing/
├── SKILL.md
├── scripts/       # optional
├── references/    # optional
└── assets/        # optional
```

A minimal `SKILL.md` looks like this:

```markdown
---
name: pdf-processing
description: Extract, inspect, merge, and fill PDF documents. Use for PDF tasks.
---

Follow the workflow in `references/workflow.md`.
Use `scripts/extract.py` when text extraction is required.
```

Useful commands:

```text
/skills                 list discovered skills
/skills reload          discover new or edited skills
/skills pdf-processing  load one manually
```

`kia` ships with a `skill-creator` skill that can draft and validate new skill packs. Bundled skills are copied into the project on first use and never overwrite your edits.

## Personas

A persona owns the agent's identity, system prompt, and tool surface — unlike skills, which add instructions, a persona replaces them. Personas are Python modules bundled in `kiui/agent/personas/`; `coder` (the default) is the full coding agent, while `chatter` is a general chatbot limited to `web_search` and `web_fetch`, with no file/shell access and no environment context in its prompt.

```bash
kia --persona chatter   # start as another persona
```

```text
/persona                list installed personas and their tool surface
/persona chatter        switch persona (restarts the conversation, like /clear)
```

Each persona module defines `build_system_prompt(ctx)` — composed from the shared blocks in `kiui/agent/prompts.py` — and an optional `TOOLS` whitelist enforced in both the advertised tool definitions and the tool executor.

### Personal skill library

Configure a Git repository to share skills between projects:

```yaml
kia_lib: git@github.com:username/kia-skills.git
```

Then manage it with `kib`:

```bash
kib list
kib list --local
kib install pdf-processing
kib upload pdf-processing
kib upload pdf-processing --force
kib remove pdf-processing
```

Remote skills are not exposed to the agent until installed. Upload validates the pack and rejects symlinks; existing local skills are never overwritten.

## Context that stays useful

Long sessions are managed in layers:

1. **Tool-result compaction** reduces oversized output before it reaches conversation history and retains a private full capture for recovery.
2. **Context pruning** trims, then clears, old large tool results as the context window fills.
3. **LLM compaction** summarizes the oldest conversation when context pressure becomes high.

The status display reports live context usage. `/usage` shows token totals and compaction statistics, `/context` provides a concise message log, and `/compact` forces summarization immediately.

## Permissions and safety

Choose how often tools require confirmation with `--perm` or `/perm`:

| Mode | Confirmation behavior |
|------|------------------------|
| `auto` | Runs safety-approved tool calls without prompting. This is the CLI default. |
| `default` | Automatically allows read/search/web/skill tools; prompts for writes, edits, shell commands, removals, and sub-agents. |
| `strict` | Prompts before every tool call. |

```bash
kia --perm default
kia --perm strict
```

A hard safety guard runs before the selected mode and blocks recognized destructive operations such as recursive deletion of critical paths, filesystem formatting, writes to block devices, fork bombs, and shutdown commands.

> The shell detector is defense in depth, **not a sandbox**. Static checks cannot understand every equivalent shell expression. Use a container, VM, restricted account, or OS-level sandbox when commands must be contained.

File tools are not restricted to the current workspace. Review prompts carefully in `default` and `strict` modes.

## Sessions and project data

Sessions are saved automatically during work (throttled to at most once every 30 seconds) and can be resumed later:

```bash
kia --resume                 # choose interactively
kia --resume <session_id>
```

You can also run `/resume` without leaving the current agent. Project-local state lives under `.kia/`:

```text
.kia/
├── sessions/       saved conversations
├── tool-results/   retained full command output
├── skills/         project skill packs
└── history         terminal input history
```

Inspect or clean generated data:

```bash
kia --storage
kia --clean
```

`--clean` removes generated sessions, tool results, and command history. It preserves installed skills and unrecognized entries.

## Command reference

### CLI options

| Option | Description |
|--------|-------------|
| `--model NAME` | Use a configured model alias. |
| `--persona NAME` | Run as a persona (default: `agent`). |
| `--reasoning-effort LEVEL` | Set `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`. |
| `--perm MODE` | Set `auto`, `default`, or `strict`. |
| `--stream` / `--no-stream` | Enable or disable token-by-token output. |
| `--resume [SESSION_ID]` | Resume by ID or choose a session interactively. |
| `--verbose` | Show API timing and additional diagnostics. |
| `--list` | List configured models and exit. |
| `--storage` | Report project `.kia/` disk usage and exit. |
| `--clean` | Remove generated project state and exit. |
| `--hub` | Run the shared Web UI hub. |
| `--web-port PORT` | Select the hub port; default is `8765`. |

Run `kia --help` for the options supported by your installed version.

### In-session commands

| Command | Action |
|---------|--------|
| `/help` | Show built-in help. |
| `/context` | Show a one-line-per-message context log. |
| `/system_prompt` | Print the complete current system prompt. |
| `/compact` | Force LLM context compaction. |
| `/usage` | Show token and tool-compaction usage. |
| `/model [name]` | Show or switch the active model. |
| `/reasoning [level]` | Show or change reasoning effort. |
| `/skills [name\|reload]` | List, load, or rediscover skills. |
| `/persona [name]` | List personas, or switch (restarts the conversation). |
| `/goal [text\|clear]` | Show, set, or clear a standing goal. |
| `/perm [mode]` | Show or change confirmation mode. |
| `/rewind [round]` | Roll conversation and/or files back. |
| `/clear` | Save the current session and start a new one. |
| `/resume [session_id]` | Save the current session, then resume another. |
| `/exit`, `/quit` | Exit the agent. |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send the prompt. |
| `Escape`, then `Enter` | Insert a newline. |
| `Ctrl+C` on non-empty input | Clear the input. |
| `Ctrl+C` twice on empty input | Exit. |
| `Ctrl+C` or `Esc` during a request | Cancel the in-flight operation. |

## Built-in tools

| Tool | Purpose |
|------|---------|
| `read_file` | Read a focused range of a file. |
| `write_file` | Create or replace a file. |
| `edit_file` | Replace an exact text region surgically. |
| `multi_edit` | Apply ordered edits to one file atomically. |
| `ls` | List one directory level. |
| `glob_files` | Find files with gitignore-aware globbing. |
| `grep_files` | Search file contents with a regular expression. |
| `exec_command` | Run a shell command with streaming output. |
| `web_search` | Search the live web. |
| `web_fetch` | Fetch a URL as readable text. |
| `remove_file` | Remove a file or directory. |
| `spawn_subagent` | Delegate an independent task. |
| `load_skill` | Load a skill's full instructions. |
| `report_goal` | Report whether a standing goal has been met. |

No separate tool server is required; these tools ship with `kia`.
