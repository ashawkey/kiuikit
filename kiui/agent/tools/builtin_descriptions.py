"""Call descriptions owned by kia's built-in tools."""

from typing import Any, Callable

from .constants import MAX_READ_LINES
from .formatting import ToolCallDescription, quote_tool_call_value


def _description(name: str, primary: str = "", *qualifiers: str) -> ToolCallDescription:
    return ToolCallDescription(name, primary, tuple(item for item in qualifiers if item))


def _describe_exec(args: dict[str, Any]) -> ToolCallDescription:
    """exec_command is special: the user must see the full command they are
    about to run, so the command is never truncated (unlike other tools' compact
    values), and realtime output is streamed separately while it runs.
    """
    qualifiers = []
    if args.get("cwd"):
        qualifiers.append(f"cwd {args['cwd']}")
    if "timeout" in args and args["timeout"] != 300:
        qualifiers.append("no timeout" if args["timeout"] is None else f"timeout {args['timeout']}s")
    return _description(
        "exec_command", quote_tool_call_value(args["command"], compact=False), *qualifiers
    )


def _describe_read_file(args: dict[str, Any]) -> ToolCallDescription:
    start = max(1, args.get("offset") or 1)
    limit = args.get("limit")
    effective_limit = limit if limit is not None else MAX_READ_LINES
    return _description("read_file", str(args["file"]), f"lines {start}–{start + effective_limit - 1}")


def _describe_edit(args: dict[str, Any]) -> ToolCallDescription:
    return _description("edit_file", str(args["file"]), "replace all" if args.get("replace_all") else "")


def _describe_ls(args: dict[str, Any]) -> ToolCallDescription:
    return _description("ls", str(args.get("path") or "."), "include ignored" if args.get("all") else "")


def _describe_glob(args: dict[str, Any]) -> ToolCallDescription:
    return _description(
        "glob_files",
        quote_tool_call_value(args["pattern"]),
        f"in {args['base_dir']}" if args.get("base_dir") else "",
        "non-recursive" if args.get("recursive") is False else "",
        "include ignored" if args.get("include_ignored") else "",
    )


def _describe_grep(args: dict[str, Any]) -> ToolCallDescription:
    return _description(
        "grep_files",
        quote_tool_call_value(args["pattern"]),
        f"in {args['path']}" if args.get("path") else "",
        f"glob {args['file_glob']}" if args.get("file_glob") else "",
        "ignore case" if args.get("case_insensitive") else "",
    )


BUILTIN_CALL_DESCRIBERS: dict[str, Callable[[dict[str, Any]], ToolCallDescription]] = {
    "exec_command": _describe_exec,
    "wait": lambda a: _description("wait", f"{a['seconds']:g}s"),
    "read_file": _describe_read_file,
    "read_image": lambda a: _description("read_image", str(a["file"])),
    "write_file": lambda a: _description("write_file", str(a["file"])),
    "edit_file": _describe_edit,
    "multi_edit": lambda a: _description("multi_edit", str(a["file"]), f"{len(a.get('edits') or [])} edits"),
    "ls": _describe_ls,
    "remove_file": lambda a: _description("remove_file", str(a["file"])),
    "glob_files": _describe_glob,
    "grep_files": _describe_grep,
    "load_skill": lambda a: _description("load_skill", str(a["name"])),
    "web_search": lambda a: _description("web_search", quote_tool_call_value(a["query"])),
    "web_fetch": lambda a: _description("web_fetch", str(a["url"])),
}
