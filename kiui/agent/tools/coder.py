import os
import fnmatch
import subprocess
import json
from rich.console import Console
from rich.theme import Theme

from kiui.agent.utils import get_image_content_dict
from kiui.agent.tools.apply_patch import apply_patch as _apply_patch

# list all functions to be loaded
__all__ = ["read", "list_dir", "shell", "update_plan", "apply_patch"]

console = Console(theme=Theme({"tool": "bold cyan"}))
_TODO_LIST: list[dict] = []

# helpers to log todos and patch
def _print_todos(todos: list[dict]):

    # Graceful handling of empty or invalid input
    if not isinstance(todos, list) or len(todos) == 0:
        console.print("[dim]No todos.[/dim]")
        return

    # Styling helpers based on coder_cc.py constraints
    status_symbols = {
        "completed": "[green]✔[/green]",
        "in_progress": "[yellow]•[/yellow]",
        "pending": "[white]•[/white]",
    }
    priority_colors = {
        "high": "red",
        "medium": "yellow",
        "low": "blue",
    }

    # Compute simple summary
    completed = sum(1 for t in todos if t.get("status") == "completed")
    in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
    pending = sum(1 for t in todos if t.get("status") == "pending")

    console.print(
        f"[dim]{len(todos)} items — {completed} completed, {in_progress} in progress, {pending} pending.[/dim]"
    )

    # Print each todo as a list item
    for item in todos:
        todo_id = item.get("id", "-")
        content = (item.get("content") or "").strip()
        status = item.get("status", "pending")
        priority = item.get("priority", "low")

        symbol = status_symbols.get(status, "[white]•[/white]")
        pr_color = priority_colors.get(priority, "white")
        # status label for readability
        status_label = (status or "").replace("_", " ")

        # Render line with a checkmark for completed, colored priority, and dim metadata
        console.print(
            f"{symbol} {content} "
            f"[dim](id {todo_id}, {status_label}, priority: [/dim][{pr_color}]{priority}[/][dim])[/dim]"
        )

def _print_patch(patch: str):

    if not isinstance(patch, str) or len(patch.strip()) == 0:
        console.print("[bold]Patch[/bold]\n[dim]No changes.[/dim]")
        return

    # Render line-by-line with styles based on leading tokens
    for raw_line in patch.splitlines():
        line = raw_line.rstrip("\n\r")

        # Envelope markers
        if line.startswith("*** Begin Patch") or line.startswith("*** End Patch"):
            console.print(f"[bold cyan]{line}[/]")
            continue

        # File operations
        if line.startswith("*** Add File: "):
            try:
                _, path_part = line.split(": ", 1)
            except ValueError:
                path_part = line
            console.print(f"[bold green]*** Add File:[/] [green]{path_part}[/]")
            continue

        if line.startswith("*** Delete File: "):
            try:
                _, path_part = line.split(": ", 1)
            except ValueError:
                path_part = line
            console.print(f"[bold red]*** Delete File:[/] [red]{path_part}[/]")
            continue

        if line.startswith("*** Update File: "):
            try:
                _, path_part = line.split(": ", 1)
            except ValueError:
                path_part = line
            console.print(f"[bold yellow]*** Update File:[/] [yellow]{path_part}[/]")
            continue

        if line.startswith("*** Move to: "):
            try:
                _, path_part = line.split(": ", 1)
            except ValueError:
                path_part = line
            console.print(f"[bold blue]*** Move to:[/] [blue]{path_part}[/]")
            continue

        # Hunk and footer markers
        if line.startswith("@@"):
            console.print(f"[bold magenta]{line}[/]")
            continue

        if line.startswith("*** End of File"):
            console.print(f"[dim]{line}[/]")
            continue

        # Diff lines
        if line.startswith("+"):
            console.print(f"[green]{line}[/]")
            continue
        if line.startswith("-"):
            console.print(f"[red]{line}[/]")
            continue
        if line.startswith(" "):
            console.print(f"[dim]{line}[/]")
            continue

        # Fallback for any other content
        if len(line.strip()) == 0:
            console.print("")
        else:
            console.print(line)
    
def read(
    file_path: str,
    offset: int = 1,
    limit: int = 2000,
) -> str:
    """Reads a local file with 1-indexed line numbers.
    
    - By default, it reads up to 2000 lines starting from the beginning of the file. Any lines longer than 2000 characters will be truncated.
    - Results are returned using `cat -n` format, with line numbers starting at 1.
    - This tool can also read images (eg PNG, JPG, etc). The image is returned as a base64-encoded JPEG image.
    
    Args:
        file_path: The absolute path to the file to read. (must be an absolute path, not relative)
        offset: he line number to start reading from. Must be 1 or greater. (default: 1)
        limit: The maximum number of lines to return. (default: 2000)
    """

    console.print(f"[TOOL] Read {file_path}", style="tool")

    # Validate absolute path
    if not os.path.isabs(file_path):
        raise ValueError("file_path must be an absolute path")
    # Fast existence and size checks
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")
    if os.path.isdir(file_path):
        raise IsADirectoryError(f"Is a directory: {file_path}")
    
    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff")
    if file_path.lower().endswith(IMAGE_EXTS):
        # image file
        content = [
            # get_text_content_dict("This is the image returned by the Read tool:"),
            get_image_content_dict(file_path),
        ]
        return f"Success: read image from {file_path}.", content
    else:
        # text file
        with open(file_path, "r", encoding="utf-8") as f:
            # Stream lines and apply offset/limit
            selected: list[str] = []
            for idx, raw in enumerate(f, start=1):
                if idx < offset:
                    continue
                if len(selected) >= limit:
                    break
                line = raw.rstrip("\n\r")
                if len(line) > 2000:
                    line = line[:2000]
                selected.append(line)
            # Format as cat -n and return the content
            return "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(selected, start=1))

def list_dir(path: str, ignore: list[str] | None = None) -> str:
    """Lists files and directories in a given path. 

    Args:
        path: The absolute path to the directory to list
        ignore: List of glob patterns to ignore (default: None)
    """

    console.print(f"[TOOL] List dir {path}", style="tool")

    if not os.path.isabs(path):
        raise ValueError("path must be an absolute path")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such path: {path}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Not a directory: {path}")

    ignore = ignore or []

    try:
        entries: list[str] = []
        for name in os.listdir(path):
            # Skip dot-files/directories for cleaner output
            if name.startswith("."):
                continue
            abs_path = os.path.abspath(os.path.join(path, name))
            # Apply ignore globs against full absolute path
            if any(fnmatch.fnmatch(abs_path, pattern) for pattern in ignore):
                continue
            entries.append(abs_path)
        return "\n".join(sorted(entries))
    except PermissionError as e:
        raise PermissionError(f"Permission denied listing {path}: {e}")

def shell(
    command: str,
    timeout: int = 120000,
    description: str = "",
) -> str:
    """Executes a given bash command with optional timeout and capture output.

    Args:
        command: The command to execute
        timeout: Optional timeout in milliseconds (default: 120000, max: 600000)
        description: Clear, concise description of what this command does in 5-10 words (default: "")
    """

    console.print(f"[TOOL] Shell {command}", style="tool")

    if not command or not isinstance(command, str):
        raise ValueError("command must be a non-empty string")

    # Clamp timeout
    if timeout <= 0:
        timeout = 120000
    if timeout > 600000:
        timeout = 600000

    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout / 1000.0,
        )
        output = result.stdout or ""
        if len(output) > 30000:
            output = output[:30000]
        # Treat non-zero codes as errors except 1 (common no-match)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"Command failed with exit code {result.returncode}. Output: {output}")
        return output
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or ""
        if len(partial) > 30000:
            console.print(f"[TOOL] Output is too long: {len(partial)} characters. Truncating to 30000 characters.", style="tool")
            partial = partial[:30000]
        raise RuntimeError(f"Command timed out after {timeout}ms. Partial output: \n{partial}")

def update_plan(todos: str) -> str:
    """Use this tool to create and manage a structured task list for your current coding session. 
    This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
    It also helps the user understand the progress of the task and overall progress of their requests.

    Args:
        todos: A JSON string representing the todo list (array of objects). Each item must include: (1) id (integer), (2) content (string), (3) status ("pending" | "in_progress" | "completed"), (4) priority ("high" | "medium" | "low").
    """

    console.print("[TOOL] Update plan:", style="tool")

    allowed_status = {"pending", "in_progress", "completed"}
    allowed_priority = {"high", "medium", "low"}

    seen_ids: set[str] = set()
    in_progress_count = 0

    validated: list[dict] = []
    # parse JSON input
    try:
        todos = json.loads(todos)
    except Exception:
        raise ValueError("todos must be a valid JSON string representing a list of objects")

    if not isinstance(todos, list):
        raise ValueError("todos must decode to a list of objects")

    for idx, item in enumerate(todos):
        if not isinstance(item, dict):
            raise ValueError(f"todos[{idx}] must be a dict")
        
        # check all required fields are present
        required_fields = ["content", "status", "priority", "id"]
        for field in required_fields:
            if field not in item:
                raise ValueError(f"todos[{idx}] must contain the field {field}")

        content = item["content"]
        status = item["status"]
        priority = item["priority"]
        todo_id = int(item["id"])

        if not isinstance(content, str) or len(content.strip()) == 0:
            raise ValueError(f"todos[{idx}].content must be a non-empty string")
        if status not in allowed_status:
            raise ValueError(
                f"todos[{idx}].status must be one of {sorted(allowed_status)}"
            )
        if priority not in allowed_priority:
            raise ValueError(
                f"todos[{idx}].priority must be one of {sorted(allowed_priority)}"
            )
        if not isinstance(todo_id, int):
            raise ValueError(f"todos[{idx}].id must be an integer")
        if todo_id in seen_ids:
            raise ValueError(f"Duplicate todo id detected: {todo_id}")
        seen_ids.add(todo_id)

        if status == "in_progress":
            in_progress_count += 1

        # Only keep the required fields; ignore extras
        validated.append({
            "id": todo_id,
            "content": content.strip(),
            "status": status,
            "priority": priority,
        })

    _print_todos(validated)

    if in_progress_count > 1:
        raise ValueError("Only one todo can be in_progress at a time")
    
    # update the todo list
    global _TODO_LIST
    _TODO_LIST = validated

    # Return a summary (human-readable) that embeds the JSON representation for compatibility
    summary_lines = [
        "Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable:",
        json.dumps({"todos": _TODO_LIST}),
    ]
    return "\n".join(summary_lines)


def apply_patch(
    patch_text: str,
) -> str:
    """Use the `apply_patch` tool to edit files.
    Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

    *** Begin Patch
    [ one or more file sections ]
    *** End Patch

    Within that envelope, you get a sequence of file operations.
    You MUST include a header to specify the action you are taking.
    Each operation starts with one of three headers:

    *** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
    *** Delete File: <path> - remove an existing file. Nothing follows.
    *** Update File: <path> - patch an existing file in place (optionally with a rename).

    May be immediately followed by *** Move to: <new path> if you want to rename the file.
    Then one or more “hunks”, each introduced by @@ (optionally followed by a hunk header).
    Within a hunk each line starts with:

    For instructions on [context_before] and [context_after]:
    - By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
    - If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
    @@ class BaseClass
    [3 lines of pre-context]
    - [old_code]
    + [new_code]
    [3 lines of post-context]

    - If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

    @@ class BaseClass
    @@ 	 def method():
    [3 lines of pre-context]
    - [old_code]
    + [new_code]
    [3 lines of post-context]

    The full grammar definition is below:
    Patch := Begin { FileOp } End
    Begin := "*** Begin Patch" NEWLINE
    End := "*** End Patch" NEWLINE
    FileOp := AddFile | DeleteFile | UpdateFile
    AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
    DeleteFile := "*** Delete File: " path NEWLINE
    UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
    MoveTo := "*** Move to: " newPath NEWLINE
    Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
    HunkLine := (" " | "-" | "+") text NEWLINE

    A full patch can combine several operations:

    *** Begin Patch
    *** Add File: hello.txt
    +Hello world
    *** Update File: src/app.py
    *** Move to: src/main.py
    @@ def greet():
    -print("Hi")
    +print("Hello, world!")
    *** Delete File: obsolete.txt
    *** End Patch

    It is important to remember:

    - You must include a header with your intended action (Add/Delete/Update)
    - You must prefix new lines with `+` even when creating a new file
    - File references can only be relative, NEVER ABSOLUTE.

    Args:
        patch_text: The patch language string.
    """

    console.print(f"[TOOL] Apply patch:", style="tool")
    _print_patch(patch_text)

    # internally call openai's official apply_patch script
    return _apply_patch(patch_text)
