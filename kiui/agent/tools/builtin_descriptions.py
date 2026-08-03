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


def _describe_start_process(args: dict[str, Any]) -> ToolCallDescription:
    return _description(
        "start_process",
        quote_tool_call_value(args["command"]),
        f"cwd {args['cwd']}" if args.get("cwd") else "",
    )


def _describe_inspect_processes(args: dict[str, Any]) -> ToolCallDescription:
    return _description(
        "inspect_processes",
        str(args.get("process_id") or "all"),
        f"tail {args['log_tail_chars']:,} chars" if args.get("log_tail_chars") else "",
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


def _describe_process_output(result: dict[str, Any]) -> str:
    processes = result.get("processes") or [result]
    lines = []
    for process in processes:
        status = str(process.get("status", "unknown"))
        if status == "exited":
            status += f" ({process.get('exit_code', '?')})"
        command = " ".join(str(process.get("command", "")).split())
        if len(command) > 80:
            command = command[:79] + "…"
        line = f"{process.get('process_id', '?')} · {status} · pid {process.get('pid', '?')}"
        if command:
            line += f" · {command}"
        lines.append(line)
    count = result.get("count", len(processes))
    if not lines:
        return "No managed processes"
    if count > len(lines):
        lines.append(f"… {count - len(lines)} more processes")
    process = processes[0] if len(processes) == 1 else None
    if process is not None and "log_tail" in process:
        tail = str(process.get("log_tail", ""))
        state = "truncated tail" if process.get("log_tail_truncated") else "log tail"
        lines.append(f"{state}: {len(tail):,} chars · {process.get('log_path', '?')}")
        tail_lines = tail.rstrip().splitlines()
        for line in tail_lines[-2:]:
            lines.append(line if len(line) <= 120 else "…" + line[-119:])
    return "\n".join(lines)


def _describe_read_output(result: dict[str, Any]) -> str:
    message = f"{result.get('lines_read', 0)} lines read"
    reason = result.get("truncation_reason")
    if result.get("truncated") and reason != "line cap":
        message += f" · truncated{f' ({reason})' if reason else ''}"
    return message


def _describe_search_output(result: dict[str, Any], noun: str) -> str:
    message = f"{result.get('count', 0)} {noun}"
    if result.get("truncated"):
        reason = result.get("truncation_reason")
        message += f" · truncated{f' ({reason})' if reason else ''}"
    return message


BUILTIN_CALL_DESCRIBERS: dict[str, Callable[[dict[str, Any]], ToolCallDescription]] = {
    "start_process": _describe_start_process,
    "inspect_processes": _describe_inspect_processes,
    "stop_process": lambda a: _description("stop_process", str(a["process_id"])),
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


BUILTIN_OUTPUT_DESCRIBERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "exec_command": lambda r: (
        f"exit code {r.get('exit_code', '?')}"
        + (" · interrupted" if r.get("interrupted") else "")
        + (" · timed out" if r.get("timed_out") else "")
    ),
    "start_process": _describe_process_output,
    "inspect_processes": _describe_process_output,
    "stop_process": _describe_process_output,
    "read_file": _describe_read_output,
    "read_image": lambda r: str(r["message"]),
    "write_file": lambda r: str(r["message"]),
    "edit_file": lambda r: str(r["message"]).splitlines()[0],
    "multi_edit": lambda r: str(r["message"]),
    "ls": lambda r: _describe_search_output(r, "entries"),
    "remove_file": lambda r: str(r["message"]),
    "glob_files": lambda r: _describe_search_output(r, "files matched"),
    "grep_files": lambda r: _describe_search_output(r, "matches"),
    "wait": lambda r: f"Waited {r['waited_seconds']:g}s",
}
