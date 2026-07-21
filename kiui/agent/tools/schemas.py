"""OpenAI-format definitions for built-in tools."""

from typing import Any

def get_tool_definitions(
    include_subagent: bool = True,
    allowed: set[str] | frozenset | None = None,
    supports_image_input: bool = False,
) -> list[dict[str, Any]]:
    """Return OpenAI-format tool definitions for the built-in tools.

    Set *include_subagent* to False for sub-agents, which must not be able to
    spawn further sub-agents (keeps spawning a single, sequential level deep).
    Set *allowed* to a persona's tool whitelist to advertise only those tools;
    None advertises all tools. ``read_image`` is advertised only when
    *supports_image_input* is true.
    """
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. Defaults to the first 1000 lines. Large results are compacted; use grep_files first and offset/limit for focused reads.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to read"},
                        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                        "limit": {"type": "integer", "description": "Maximum number of lines to read"},
                    },
                    "required": ["file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_image",
                "description": "Read a local PNG, JPEG, GIF, or WebP image and send it to the model for visual inspection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the image file"},
                    },
                    "required": ["file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file with content. Creates parent directories automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"},
                    },
                    "required": ["file", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Make a surgical edit to a file by replacing exact text. old_text should match the file content; minor differences in trailing whitespace / line endings are tolerated. By default old_text must resolve to exactly one location; set replace_all=true to replace every occurrence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to edit"},
                        "old_text": {"type": "string", "description": "Exact text to find and replace"},
                        "new_text": {"type": "string", "description": "New text to replace the old text with"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences instead of requiring exactly one (default: false)"},
                    },
                    "required": ["file", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multi_edit",
                "description": (
                    "Apply a sequence of edits to a SINGLE file in one atomic operation. "
                    "Edits are applied in order, each to the result of the previous one. "
                    "If any edit fails to match, NO changes are written (all-or-nothing). "
                    "Prefer this over multiple edit_file calls when changing several places in one file. "
                    "Same tolerant matching as edit_file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to edit"},
                        "edits": {
                            "type": "array",
                            "description": "Ordered list of edits to apply",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_text": {"type": "string", "description": "Exact text to find and replace"},
                                    "new_text": {"type": "string", "description": "Replacement text"},
                                    "replace_all": {"type": "boolean", "description": "Replace every occurrence (default: false)"},
                                },
                                "required": ["old_text", "new_text"],
                            },
                        },
                    },
                    "required": ["file", "edits"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": (
                    "List the contents of a directory (non-recursive). Shows entry names, "
                    "type (file/dir) and size. Respects .gitignore and skips noise dirs by default. "
                    "Large listings are compacted; use glob_files with a narrow pattern when possible. "
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory to list (default: working directory)"},
                        "all": {"type": "boolean", "description": "Include hidden and gitignored entries (default: false)"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": (
                    "Run a foreground shell command and stream its output. Returns stdout, stderr, and exit code; "
                    "large output is compacted with full capture available in an artifact."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_process",
                "description": (
                    "Start a managed background process and return immediately. "
                    "Its combined output is written to a readable log file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run in the background"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_processes",
                "description": (
                    "Inspect managed background process status after an optional bounded wait. "
                    "Omit process_id to list all processes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "Process ID to inspect (optional)"},
                        "wait": {
                            "type": "number",
                            "minimum": 0,
                            "default": 0,
                            "description": "Seconds to wait before returning a status snapshot (default: 0)",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_process",
                "description": "Stop a managed background process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "Managed process ID returned by start_process"},
                    },
                    "required": ["process_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob_files",
                "description": (
                    "Find files matching a glob pattern. Preferred over exec_command with find. "
                    "Searches recursively by default. Respects .gitignore and skips noise dirs. "
                    "Set recursive=false to match only in the immediate directory. Max 500 results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py', 'src/**/*.ts', or '*.py'"},
                        "base_dir": {"type": "string", "description": "Directory to search in (default: current directory)"},
                        "recursive": {"type": "boolean", "description": "Search subdirectories recursively (default: true)"},
                        "include_ignored": {"type": "boolean", "description": "Include .gitignored files (default: false)"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_files",
                "description": "Search file contents using a regex pattern. Returns up to 200 matching lines with file path and line number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "File or directory to search (default: current directory)"},
                        "file_glob": {"type": "string", "description": "Filename filter, e.g. '*.py' or '*.ts'"},
                        "case_insensitive": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for real-time information. Returns summarized results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch URL content and convert to readable text. Content capped at 20K chars.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remove_file",
                "description": "Remove a file or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file or directory to remove"},
                    },
                    "required": ["file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_subagent",
                "description": (
                    "Spawn a sub-agent to run a task. Blocks until the sub-agent completes and returns the result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language task for the sub-agent",
                        },
                    },
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": "Load the full prompt instructions for a skill by name. Use this when a task matches a skill's domain so you can follow its specialized guidance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the skill to load"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "report_goal",
                "description": (
                    "Report whether the current standing goal (set by the user via /goal) is met. "
                    "Call this exactly once at the end of a goal-check turn. "
                    "When met=true the automatic goal iteration stops; when met=false the agent is "
                    "prompted again to keep working toward the goal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "met": {"type": "boolean", "description": "True if the goal is now fully satisfied."},
                        "reason": {"type": "string", "description": "Brief explanation of the current status or what still remains."},
                    },
                    "required": ["met"],
                },
            },
        },
    ]

    if not supports_image_input:
        definitions = [
            d for d in definitions
            if d.get("function", {}).get("name") != "read_image"
        ]
    if not include_subagent:
        definitions = [
            d for d in definitions
            if d.get("function", {}).get("name") != "spawn_subagent"
        ]
    if allowed is not None:
        definitions = [
            d for d in definitions
            if d.get("function", {}).get("name") in allowed
        ]
    return definitions
