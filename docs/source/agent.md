# Agent

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/agent/)

A terminal-based AI agent (`kia`) with built-in tool-use, web access, shell execution, sub-agents, skills, context compaction, and rewind/rollback. Supports any OpenAI-compatible API.

## API keys

The API keys are stored in the `~/.kiui.yaml` file (or `./.kiui.yaml`), which is automatically loaded as `kiui.conf`.

```yaml
openai:
  my-model:              # alias name for this model
    model: gpt-4o        # the model name used in the API call
    api_key: sk-...      # API key
    base_url: https://api.openai.com/v1  # optional, defaults to OpenAI
```

## CLI

```bash
python -m kiui.agent.cli --help
# short cut:
kia --help

# list all available models (from ~/.kiui.yaml)
kia --list

# start interactive chat (model defaults to the first in config)
kia

# use a specific model
kia --model my-model

# set permission mode: auto (confirm destructive), default (ask), strict (read-only)
kia --perm auto
kia --perm strict

# enable verbose debugging (shows API timing, token counts, context stats)
kia --model my-model --verbose

# resume a previous session
kia --resume <session_id>

# pick a session interactively from saved sessions
kia --resume
```

## Built-in tools

The agent includes built-in tools — no external tool files needed:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with offset/limit |
| `write_file` | Create or overwrite a file |
| `edit_file` | Surgical string replacement in files |
| `exec_command` | Execute shell commands (real-time streaming) |
| `glob_files` | Find files by glob pattern |
| `grep_files` | Search file contents by regex |
| `web_search` | Search the web for real-time information |
| `web_fetch` | Fetch and convert URL content to text |
| `remove_file` | Remove a file or directory |
| `spawn_subagent` | Spawn a sub-agent for delegated tasks |
| `load_skill` | Load specialized skill instructions |

## Permissions

Three-tier permission system, configurable via `--perm`. A **hard safety guard** runs before mode-based checks and always blocks recognized dangerous operations (e.g. `rm -rf /`, `mkfs`, or writing directly to block devices). File tools may access paths outside the working directory.

| Mode | Behavior |
|------|----------|
| `auto` | All tools run without prompting (used for sub-agents / pipe mode) |
| `default` | Safe tools (read, glob, grep, web search/fetch, load skill) auto-approve; risky tools (write, edit, exec, remove, spawn) prompt for confirmation |
| `strict` | Every tool prompts for confirmation |

## Skills

The agent auto-discovers **skill packs** from `.kia/skills/` in the working directory. A skill pack is a directory containing a prompt file that provides specialized guidance. The agent can load skills on demand via the `load_skill` tool.

Place skill packs in `.kia/skills/<skill_name>/` — the agent's system prompt will list them as available.

## Sub-agents

The `spawn_subagent` tool spawns a **child agent** that runs autonomously (no user interaction) and returns a result. Useful for decomposing large tasks into independent sub-tasks.

```bash
# Example sub-agent usage (from within a kia session):
# "spawn a sub-agent to review the codebase and write a summary"
```

## Context management

The agent automatically manages the LLM context window:

- **Pruning**: Old, large tool results are truncated to stay within limits.
- **Compaction**: When the context window is under pressure, the agent uses the LLM itself to summarize older messages into a compact form, preserving essential information.

## Rewind / rollback

The agent tracks file changes during a session. You can ask it to **rewind** (undo) changes made during the conversation:

```
> rewind to before I asked you to refactor utils.py
```

This reverts files to their pre-change state and truncates the conversation history.

## Sessions

Sessions are automatically saved to `~/.kia/sessions/` on each turn. Resume with:

```bash
kia --resume           # pick interactively
kia --resume <session_id>
```
