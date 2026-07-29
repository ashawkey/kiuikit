# kia (kiui.agent)

## Installation

```bash
pip install kiui[kia]
```

## Configuration

The agent uses a YAML configuration file located at `./.kiui.yaml` (current directory) or `~/.kiui.yaml` (home directory). Model aliases remain under the `openai` key. Each entry may select a provider; omitted `provider` defaults to the currently bundled `openai` provider, which uses the OpenAI-compatible Chat Completions transport.

Example `.kiui.yaml`:

```yaml
openai:
  gpt: # model_alias
    provider: openai # optional; this is the default
    model: gpt-4o # actual model name used in the API
    api_key: sk-proj-...
    base_url: https://api.openai.com/v1

  gpt-5.6-sol:
    provider: openai-codex
    model: gpt-5.6-sol
    # No api_key/base_url: authenticate with /login openai-codex.

kia_web_token: web-secret # optional Web UI access token
kia_lib: git@github.com:username/kia-skills.git # optional personal skill/persona library repo
```

The `openai-codex` provider uses a ChatGPT Plus/Pro subscription through OpenAI OAuth and the Codex Responses endpoint. OAuth credentials are stored globally in `~/.kia/auth.json`, not in project configuration or session history. The file is plaintext and restricted to mode `0600` on Unix. Start kia with the Codex alias and run `/login` (or `/login openai-codex`); browser, pasted-redirect, and device-code flows are available.

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
| `--clean [ENTRY ...]` | Remove selected project `.kia/` entries, or all non-preserved entries by default |
| `--hub` | Run the shared web hub daemon (owns the public port) |
| `--web-port PORT` | Hub listener port (default: `8765`) |

### Storage management

`kia --storage` reports every user-facing top-level entry in the current project's `.kia/` directory and whether a default clean removes it. Kia maintains `.kia/.gitignore` with a `*` rule so the entire directory stays ignored; this internal file is hidden from storage listings and is never cleaned. Bare `kia --clean` removes every entry except persistent `skills/`. Pass one or more entry names to remove only those entries; explicitly named preserved entries can also be removed.

```bash
kia --storage
kia --clean                    # everything except skills/
kia --clean pdf-cache          # one entry
kia --clean sessions pdf-cache # selected entries
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
to sign in. The browser monitors its hub connection, shows a reconnecting
banner when it becomes unhealthy, and preserves drafts while actions are
disabled until the connection recovers.

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
| `/continue` | Resume an unfinished round without adding a user message; warns if the last round is complete (output-limit and missing-terminal truncations continue automatically) |
| `/usage` | Show token usage for this session |
| `/perm [auto\|default\|strict]` | Show or change permission mode |
| `/model [name]` | Show or switch LLM model mid-session |
| `/login [provider\|model-alias]` | Authenticate an OAuth provider; defaults to the current provider |
| `/logout [provider\|model-alias]` | Remove stored OAuth credentials |
| `/auth [provider\|model-alias]` | Show authentication status |
| `/rewind` | Return to before a user prompt, restore it to the chatbox, then branch |
| `/skills` | List installed skills; `/skills reload` to re-scan; `/skills <name>` to load one |
| `/persona` | List personas; `/persona <name>` to switch (restarts the conversation) |
| `/goal [text\|clear]` | Set a goal the agent auto-iterates toward until met (see [Goals](#goals)) |
| `/wait <duration> <prompt>` | Send a prompt after a delay, e.g. `/wait 1h check whether the other agent finished` |
| `/clear` | Clear conversation history and start a new session |
| `/resume [session_id]` | Save the current session, then resume a previous one (bare `/resume` picks interactively) |
| `/exit` or `/quit` | Exit the agent |

A message sent while the agent is working normally steers the next tool-call iteration. `/wait` is deliberately different: its prompt becomes ready only after the requested seconds (`s`), minutes (`m`), or hours (`h`) and always starts a fresh round after the current round finishes. While the agent is idle, the same activity indicator used for `Working...` and `Executing...` shows `Waiting...` with a live countdown in the terminal and Web UI. Only one prompt, immediate or delayed, can be pending; use the existing pending-message edit/withdraw action to cancel it.

A command sent while the agent is working does not have to wait for the round: a round owns the conversation, the provider, and the terminal prompt, so any command that merely reads session state (`/help`, `/usage`, `/context`, `/system_prompt`, `/auth`, and the bare listing form of `/model`, `/persona`, `/skills`) or takes effect on the next API call (`/perm`, `/reasoning`, `/goal`) is answered immediately — from the terminal and the Web UI alike. Commands that rewrite the conversation or swap what runs it (`/clear`, `/compact`, `/rewind`, `/resume`, `/login`, a `/model` or `/persona` switch, `/skills <name>`) stay queued until the round ends.

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

1. **Proactive tool-result compaction** — before a result enters conversation history, oversized output is cut to an excerpt taken from whichever end carries the signal: a file or listing from the top, a log from the bottom, a process snapshot keeps its status and latest tail, anything else both ends. Full command output is teed during execution and saved under `.kia/tool-results/<session>/`, and the excerpt carries a pointer to it — so the excerpt only has to be enough to decide whether to go and read the rest, which is why no heuristic tries to guess the "interesting" lines. A capture is also kept for command output that is small enough to enter history whole but large enough for eviction to trim later: a file read can simply be repeated, a command cannot. The pointer to a capture lives in the message text, so it survives eviction and session save/resume alike. Captures for all but the 20 most recent sessions are dropped at startup.
2. **Context eviction** — past 55% usage, old tool results from read/exec/web tools are trimmed to their per-tool retention policy, and fully cleared when the complete output is recoverable from disk. Results a later call superseded (a re-read, or a write to a file that was read earlier) go first. The newest 25% of the window — capped at 40k tokens, and never fewer than three turns — is never touched, and a pass that cannot free 8% of the window is skipped entirely, so the provider's cached prefix is invalidated rarely and in bulk. Eviction always stays below the compaction trigger, so a model whose output reserve pulls that trigger down (gpt-5 reserves 128k of a 258k window) gets a cheap deterministic pass before an LLM round-trip is spent.
3. **LLM compaction** — near the end of the window (85%, or sooner if the model's output limit needs the room), the oldest messages are summarized and replaced with a structured handoff. A pass always aims to land at least 15% of the window *below* the trigger that woke it: on a model whose output reserve drags the trigger down to its floor (gpt-5 puts both at 129k of a 258k window) a pass that merely hit a flat 50% target would land back on the trigger and be fired again by the very next tool result. If a request overflows anyway, the agent compacts and retries once instead of failing the turn.

Usage is measured against the token count the API last reported, plus an estimate of what has been added since — so the system prompt, tool schemas, and provider framing that character counting cannot see are all accounted for.

### What survives a compaction

The summary follows a fixed section structure (goal, constraints, progress, key decisions, next steps, critical context). Around the model's summary, several things are carried **deterministically** rather than trusted to survive re-summarization:

- **The original request**, verbatim, through every subsequent compaction.
- **Files read and modified**, as a list of paths — never their contents, since re-reading them is what makes compaction refill the window and cascade into compacting again.
- **Loaded skills**, so the agent knows to call `load_skill` again for the full instructions.

The newest 15% of the window (capped at 20k tokens) is never summarized away, and when the summarization input itself has to be cut it drops the *middle* — the oldest messages anchor the task, the newest carry the current state. Compacting an already-compacted session updates the previous summary instead of re-compressing it, so history does not degrade with each pass.

A `pre-compaction` session revision is saved before the history is replaced, so rewinding to the prompt boundary for that round can restore the pre-round conversation and code state.

Two guards keep an unproductive pass from repeating every round. Before the round-trip, a split that would free less than it writes back — the summary is the one part of a compaction that *adds* context — is abandoned without calling the model at all. After it, every pass sets a floor that suppresses the next one until the context has actually grown by 5% of the window, whether or not the pass went well: a pass that clears the yield bar by a hair used to reset that floor, leaving the marginal pass right behind it unguarded. Summarization runs at low reasoning effort under a fixed output cap, since rewriting a conversation into a fixed section structure is transcription rather than reasoning.

## Rewind

The `/rewind` command returns to the checkpoint immediately before a user prompt was sent. It takes no arguments: the picker lists those prompt boundaries in conversation order, followed by the current checkpoint. Each prompt row shows the checkpoint round and how many files were changed while answering that prompt. After a conversation rewind, the removed prompt is restored to the terminal and web chatbox for editing and resubmission.

Picking a prompt previews the checkout before anything is applied — the selected prompt and later rounds that would be dropped, and every file the move would create, modify, or delete with its line counts. A file edited outside the agent since it was recorded is called out, because a rewind would overwrite it:

```
Revision     a80161e46f  ·  round 1  ·  9m ago  ·  saved as round
Prompt       set up the parser module
Conversation 5 → 1 messages, round 3 → 1
             2 round(s) will be dropped:
               round 3  drop util.py, it is unused
               round 2  make the parser handle floats and add tests
Files        3 will change  (+3 / -8)
             modify  parser.py   +2 -6
             delete  test_parser.py   -2
             create  util.py   +1
             ! 1 file(s) changed on disk since they were recorded and would be overwritten: parser.py
```

The mode prompt then restates that effect on each option, and can show the full diffs first:

- **Conversation + code** — restore both histories to the selected revision.
- **Conversation only** — restore messages while keeping current files.
- **Code only** — restore files while keeping the current conversation.
- **Show the file diffs** — render every hunk the checkout would apply, then ask again.

The preview and the checkout consume the same walk through the code DAG, so what is shown is exactly what is applied, and paths a walk touches but leaves byte-identical are dropped from both — `no files will change` means the rewind is conversation-only in practice.

Session messages, revisions, head movements, and code revisions are stored in an append-only JSONL DAG. Rewinding never deletes descendants: the next save creates a branch, so an accidental rewind can be reversed by checking out the former revision. Removed files and directories are stored as immutable, deduplicated content-addressed objects under the session's `objects/` directory.

### Replay

Restoring a conversation — after a rewind, `/resume`, or `--resume` — reprints it. Tool calls are described by the same function that labels them live (`read_file a.txt:1-1000`, not `read_file({"file": "a.txt"})`), so a replayed transcript reads like the session did.

Results are the part that cannot be fully reproduced: the live view renders them from the result object (a coloured diff for an edit, an exit code for a command, a line count for a read), and only the formatted text is persisted. A replayed result is therefore its text summary, marked as a failure only when the text is formatted as one.

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

### Personal resource library

`kib` manages a Git-backed library of skills and personas shared between projects. Configure an accessible repository URL in `.kiui.yaml`; authentication is delegated to your current Git/SSH environment. The repository uses `main` and stores resources under `skills/<name>/` and `personas/<name>/`.

```yaml
kia_lib: git@github.com:username/kia-skills.git
```

```bash
kib list [pattern]               # list remote skills, optionally filtering names
kib list [pattern] --local       # list/filter local skills; remote status is best-effort
kib install <name> [<name> ...]  # install one or more remote skills
kib update [<name> ...]          # sync all or selected installed skills
kib update <names...> --force  # replace conflicting library copies with local trees
kib upload <name> [<name> ...]   # upload one or more local skills
kib remove <name> [<name> ...]   # remove one or more remote skills
kib remove <names...> --local    # remove project copies only
kib upload <names...> --force    # replace existing remote skills

# Add --kind persona to any command to operate on personas:
kib list --kind persona
kib install my-coder --kind persona
kib upload my-coder --kind persona
```

Remote resources are not available to the agent until installed.
The repository is cached under `~/.kia/library/`; each configured URL has an
isolated checkout, so changing `kia_lib` selects a different cache. Each resource's last synchronized tree is recorded in its committed `.kib.json`, so update works across machines and does not depend on the cache. Install never overwrites an existing local resource. Update uploads local-only changes and downloads remote-only changes.
Conflicting changes fail with instructions to merge both copies into the project-local resource. After reviewing and validating that merge, `kib update <name> --force` replaces the complete library resource with the local tree; it does not merge and will discard any remote changes absent from the local copy.
`kib` only manages project resources under `./.kia/skills/` and `./.kia/personas/`; it does not special-case bundled resources. Upload validates the resource, rejects symlinks, creates a normal commit, and never
force-pushes. An empty repository is initialized on the first upload.

### Bundled skills

kia ships a few common skills, including `skill-creator` for authoring spec-compliant skills and `pdf-reading` for converting PDFs into readable Markdown and structured data with the external [MinerU](https://github.com/opendatalab/MinerU) CLI. The PDF skill can read extracted text, LaTeX, tables, and captions; direct inspection of extracted image pixels still requires a vision-capable tool. Bundled skills are loaded directly from the installed package rather than copied into `.kia`, so updates take effect whenever `kiui` is updated. To customize one, create a new project or personal skill under a different name.

## Personas

A persona owns the agent's identity, complete system prompt, and tool surface. Bundled personas live under `kiui/agent/bundled_personas/`; custom personas are discovered from `./.kia/personas/` and `~/.kia/personas/`. Bundled names are reserved, and project personas take precedence over personal personas.

| Persona | Tools | Purpose |
|---------|-------|---------|
| `coder` | all | The default coding agent (project-aware, full tool access) |
| `chatter` | `web_search`, `web_fetch` | General chatbot without file/shell access |
| `reviewer` | paper/file, web, skill, and sub-agent tools | Evidence-grounded academic paper reviewer |

Each persona is a directory containing `PERSONA.md`:

```markdown
---
name: my-coder
description: A concise project coding assistant.
tools: all
---
You are a terminal-based coding assistant.

{{kia:skills}}
{{kia:project-instructions}}
{{kia:current-context}}
```

`tools` is required and is either `all` or a YAML list of built-in tool names; use `[]` for no tools. Supported whole-line markers are `autonomous-mode`, `sub-agents`, `skills`, `project-instructions`, and `current-context`, each prefixed with `kia:` as above. Markers are expanded once, so marker-like text inside project instructions is not interpreted.

Project instructions normally come from `./AGENTS.md`. If `./.kia/AGENTS.md` exists, it replaces that file; an exact `@AGENTS.md` line imports the root file at that position, allowing local instructions to extend it. No other import paths are supported.

```bash
kia --persona reviewer
```

| Command | Effect |
|---------|--------|
| `/persona` | List discovered personas, sources, and tool surfaces |
| `/persona <name>` | Switch persona and restart the conversation |
| `/persona reload` | Re-scan persona directories; restart if the active persona changed |

The active persona name and content digest are saved with the session. Resume warns if its content changed and fails clearly if it is no longer installed. Tool whitelists guide the advertised model capabilities; interactive user commands remain governed by normal permission and safety checks.

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
| `glob_files` | Find files matching a glob pattern (gitignore-aware) |
| `grep_files` | Search file contents using ripgrep regex (gitignore-aware) |
| `web_search` | Search the web via DuckDuckGo |
| `web_fetch` | Fetch and parse content from a URL |
| `remove_file` | Remove a file or directory |
| `spawn_subagent` | Delegate a task to a new in-process agent instance |
| `load_skill` | Load the full prompt instructions for a skill by name |

### Skill-provided tools

A skill may ship a `tools.py` at its root (a module-level `TOOLS` list of
`{schema, run, permission}` entries). Those tools are registered and advertised
to the model only while the skill is loaded, and removed when it is unloaded.
The bundled **`monitor`** skill uses this to provide the managed background
process tools, so they appear only after `load_skill("monitor")`:

| Tool | Description |
|------|-------------|
| `start_process` | Start a managed background process with file-backed output |
| `inspect_processes` | Inspect one or all managed background processes, optionally after a bounded wait and with a bounded log tail for one process |
| `stop_process` | Stop a managed background process and its child process tree |

The executor always owns the process *registry* and cleanup, so any background
processes are terminated on `/clear`, session switch, and exit even when the
`monitor` skill is not loaded.
