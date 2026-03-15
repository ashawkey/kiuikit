# kiui.agent

A lightweight, terminal-based AI agent that can browse the web, read/write files, and execute shell commands.

## Features

- **File Operations**: Read, write, and surgically edit files.
- **Shell Execution**: Run arbitrary shell commands with real-time streaming output.
- **Web Capabilities**: Search the web and fetch/parse webpage content.
- **Tool-use**: Automatically chooses the right tool for the task.
- **Sub-agents**: Can spawn sub-agents to handle complex sub-tasks.
- **Interactive**: Rich terminal interface with syntax highlighting and progress indicators.

## Installation

```bash
pip install kiui[kia]
```

## Configuration

The agent uses a YAML configuration file located at `./.kiui.yaml` (current directory) or `~/.kiui.yaml` (home directory). You need to define your model profiles under the `openai` key (even for non-OpenAI providers, as long as they provide an OpenAI-compatible API).

Example `.kiui.yaml`:

```yaml
openai:
  gpt: # name for convenience
    model: gpt-4o # actual model name used in the API call
    api_key: sk-proj-...
    base_url: https://api.openai.com/v1
```

## Usage

### List available models

```bash
kia list
```

### Start an interactive chat

```bash
kia chat --model <model_name>
```

### Execute a single command

```bash
kia exec --model <model_name> "your prompt here"
```

## Slash commands

The agent supports the following slash commands in the CLI:
- `/help`: Show this help message.
- `/context`: Show a concise one-line-per-message context log.
- `/compact`: Force context compaction via LLM summarization.
- `/usage`: Show token usage for this session.
- `/perm`: Show or change permission mode (/perm auto|default|strict).
- `/save`: Save session to .kia/sessions/ (default: timestamp).
- `/load`: Load a saved session (no name: list available).
- `/clear`: Clear conversation history (keep system prompt).

## Tools

The agent has access to the following tools:
- `read_file`: Read file contents.
- `write_file`: Create or overwrite files.
- `edit_file`: Surgical text replacement in files.
- `exec_command`: Run shell commands.
- `glob_files`: List files matching a pattern.
- `grep_files`: Search for text in files.
- `web_search`: Search the web.
- `web_fetch`: Fetch and parse content from a URL.
- `spawn_subagent`: Delegate a task to a new agent instance.
