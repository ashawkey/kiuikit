# Agent (kia)

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
- `kib` — the optional Git-backed skill/persona library.

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

Skills are reusable instruction packs in the open [Agent Skills](https://agentskills.io) format. `kia` discovers them from:

- The installed `kiui` package — bundled skills; these take precedence and stay in sync with the installed version.
- `./.kia/skills/` — project skills; these take precedence over personal skills.
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

`kia` ships with `skill-creator` for drafting and validating skill packs and `pdf-reading` for converting PDFs to Markdown and structured data with the external [MinerU](https://github.com/opendatalab/MinerU) CLI. The PDF skill reads extracted text, LaTeX, tables, and captions; direct image-pixel inspection requires a vision-capable tool. Bundled skills are loaded directly from the installed package rather than copied into `.kia`, so upgrading `kiui` also updates them. Create custom skills under a different name.


### Personal resource library

Configure a Git repository to share skills and personas between projects:

```yaml
kia_lib: git@github.com:username/kia-skills.git
```

Then manage it with `kib`:

```bash
kib list
kib list --local
kib install pdf-processing
kib update [pdf-processing ...]
kib update pdf-processing --prefer local
kib upload pdf-processing
kib upload pdf-processing --force
kib remove pdf-processing
kib remove pdf-processing --local
kib list --kind persona
kib install my-coder --kind persona
kib upload my-coder --kind persona
kib --verbose list  # show operation, Git command, and timing details
```

Remote resources are unavailable until installed. `kib update` safely synchronizes all installed resources of the selected kind; conflicts require `--prefer local` or `--prefer remote`. The committed `.kib.json` records the last synchronized tree. `--kind persona` applies any command to `personas/<name>/` and `./.kia/personas/<name>/`; skill remains the default kind. Upload validates packs and rejects symlinks, and install never overwrites an existing local resource.

## Personas

A persona replaces the agent's identity, complete system prompt, and tool surface. Bundled personas live under `kiui/agent/bundled_personas/`; custom personas are discovered from `./.kia/personas/` and `~/.kia/personas/`. Bundled names are reserved, while project personas take precedence over personal personas.

A persona pack contains `PERSONA.md` with YAML frontmatter and a Markdown prompt:

```markdown
---
name: my-coder
description: A concise project coding assistant.
tools: all
---
You are a terminal coding assistant.

{{kia:skills}}
{{kia:project-instructions}}
{{kia:current-context}}
```

`tools` accepts `all`, `[]`, or a list of built-in tool names. Supported whole-line markers are `autonomous-mode`, `sub-agents`, `skills`, `project-instructions`, and `current-context`, prefixed with `kia:`. They expand exactly once.

```text
kia --persona chatter   start as another persona
/persona                list personas, sources, and tool surfaces
/persona chatter        switch persona and restart the conversation
/persona reload         re-scan custom personas
```

Persona identity and a content digest are stored with sessions, so resume warns when a custom persona changed. Tool selection controls what is advertised to the model; normal permission checks still apply.
