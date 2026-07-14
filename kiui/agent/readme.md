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
- **Context Management**: Automatic pruning and LLM-based compaction to stay within context windows.
- **Rewind**: Roll back conversation and/or code changes to any previous round.
- **Permissions**: Three-tier safety system (auto / default / strict) with hard safety guard for dangerous operations.
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

| `--perm MODE` | `auto`, `default`, or `strict` |
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

The hub writes its connection info (host, port, internal secret) to
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
| `/compact` | Force context compaction via LLM summarization |
| `/usage` | Show token usage for this session |
| `/perm [auto\|default\|strict]` | Show or change permission mode |
| `/model [name]` | Show or switch LLM model mid-session |
| `/rewind [round]` | Roll back conversation and/or code to a previous round |
| `/skills` | List installed skills from `.kia/skills/` |
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

All modes are protected by a **hard safety guard** that blocks:
- File writes/edits/removals outside the allowed working directory
- Destructive shell commands (`rm -rf /`, `mkfs`, `dd` to devices, fork bombs, shutdown/reboot, etc.)

## Context Management

The agent automatically manages context window usage through three layers:

1. **Tool result truncation** — individual large results are capped relative to context window size.
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

Usage:

| Form | Effect |
|------|--------|
| `/goal <text>` | Set a new goal and start auto-iterating |
| `/goal` | Show the current goal and status |
| `/goal clear` | Clear the goal and stop iterating |

There is **no fixed iteration cap** — the loop runs until the goal is reported met or you stop it. Since the terminal prompt is blocked while the loop runs, **`Ctrl+C` / `Esc` during a round is the way to stop it**, which clears the goal. Goals are saved with the session and **auto-resume** on `--resume`.

## Skills

Skills are modular prompt packs stored in `.kia/skills/<name>/SKILL.md`. Each skill provides domain-specific instructions the model can load on demand via the `load_skill` tool.

```
.kia/skills/
  git-workflow/
    SKILL.md
  python-testing/
    SKILL.md
```

The `SKILL.md` should describe what the skill does and provide specialized instructions. When relevant, the model invokes `load_skill` to load the full prompt into context.

For compatibility with other agent tools that share this convention, skills are also discovered from `.codex/skills/`, `.claude/skills/`, and `.agents/skills/`. When the same skill name appears in more than one directory, `.kia` wins, then `.codex`, `.claude`, `.agents`.

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
