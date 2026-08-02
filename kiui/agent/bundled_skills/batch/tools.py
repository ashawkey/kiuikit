"""Sequential batch processing over independent items, owned by the ``batch`` skill.

``run_batch`` drives ``LLMAgent.run_isolated_turn`` (injected as
``executor.isolated_turn``) once per item. Each item therefore sees the same
clean context, and per-item results go to a JSONL file rather than into the
conversation: the tool returns only counts, a path, and a few sample failures.

Everything policy-shaped lives here — how items are read, how results are
recorded, when a hopeless run gives up — so it can be adjusted without touching
the agent core.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from kiui.agent.tools import ToolCallDescription, quote_tool_call_value
from kiui.agent.utils import get_kia_dir

# Results live in their own directory under .kia so a run is resumable across
# sessions and safe from `kia --clean`, which preserves this entry. A run is
# named rather than given a path: the name is the resume key, and a free path
# would let a typo silently start a second run instead of continuing the first.
BATCH_DIR_NAME = "batch"
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# Where a rerun parks the previous results. Overwriting them in place would be
# the one way a run can still lose a deliverable, and the model reaches for
# resume=False exactly when it is unsure about what is already recorded.
BACKUP_SUFFIX = ".bak"

# Upper bound on one run. Well past any interactive workload; a larger job wants
# a standalone script, not an agent loop.
MAX_BATCH_ITEMS = 10_000
# Inline `items` cap. Tool-call arguments cannot be evicted from history, so an
# inline list is a permanent context cost — long lists belong in a file built by
# a command. Also keeps the list well inside the model's output-token limit.
MAX_INLINE_ITEMS = 100
# A template that is broken (or a tool that is unavailable) fails identically on
# every item. Give up once this many have failed with nothing succeeding rather
# than burning the whole list on it.
EARLY_ABORT_FAILURES = 3
# Sample of failures returned to the model. The JSONL holds them all.
MAX_REPORTED_FAILURES = 5

ITEM_PLACEHOLDER = "{item}"


def _output_path(executor, name: str) -> Path:
    """Resolve a run name to its results file inside the managed batch directory."""
    directory = get_kia_dir(executor._work_dir) / BATCH_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.jsonl"


def _read_items_file(path: Path) -> list[str]:
    """Read one item per line, skipping blanks and ``#`` comments."""
    items: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            item = line.strip()
            if item and not item.startswith("#"):
                items.append(item)
    return items


def _completed_items(path: Path) -> set[str]:
    """Items already recorded as successful in an existing result file.

    Keyed by item text, so a list containing the same item twice resumes as one
    item. Deduplicating up front would renumber the remaining indices; this way
    an index keeps naming a position in the list the caller passed.

    Unparsable lines are ignored: a run killed mid-write leaves a partial final
    line, and re-processing that one item is the safe reading.
    """
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("ok") and "item" in record:
                done.add(str(record["item"]))
    return done


def _append_record(handle, record: dict[str, Any]) -> None:
    """Append one durable result line.

    Flushed and fsynced per item: the file is the only place results exist, and
    a resumed run trusts it after a crash or a hard interrupt.
    """
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _display_path(executor, path: Path) -> str:
    """Path as the model should refer to it: relative to the working directory."""
    try:
        return str(path.relative_to(Path(executor._work_dir)))
    except ValueError:
        return str(path)


def _resolve_items(
    executor, items: list[str] | None, items_file: str
) -> tuple[list[str], str | None]:
    """Return ``(items, error)`` for the two mutually exclusive input forms."""
    if bool(items) == bool(items_file):
        return [], "Provide exactly one of items or items_file."

    if items_file:
        path = executor._resolve_path(items_file)
        if not path.is_file():
            return [], f"items_file not found: {items_file}"
        try:
            resolved = _read_items_file(path)
        except OSError as e:
            return [], f"Cannot read items_file {items_file}: {e}"
    else:
        if not isinstance(items, list) or any(not isinstance(i, str) for i in items):
            return [], "items must be a list of strings."
        if len(items) > MAX_INLINE_ITEMS:
            return [], (
                f"items is limited to {MAX_INLINE_ITEMS} entries "
                f"({len(items)} given). Write the list to a file with a command "
                "(for example `ls images/*.png > items.txt`) and pass items_file: "
                "inline arguments stay in the conversation permanently."
            )
        resolved = [item.strip() for item in items if item.strip()]

    if not resolved:
        return [], "No items to process."
    if len(resolved) > MAX_BATCH_ITEMS:
        return [], f"Too many items: {len(resolved)} (limit {MAX_BATCH_ITEMS})."
    return resolved, None


def _build_prompt(task: str, item: str) -> str:
    """Substitute *item* into *task*.

    Plain replacement, not ``str.format``: a task routinely contains braces of
    its own (JSON shapes, code) that formatting would choke on. A task without
    the placeholder gets the item appended, which is the obvious reading of
    "do this, to this".
    """
    if ITEM_PLACEHOLDER in task:
        return task.replace(ITEM_PLACEHOLDER, item)
    return f"{task}\n\nItem: {item}"


def run_batch(
    executor,
    task: str = "",
    items: list[str] | None = None,
    items_file: str = "",
    name: str = "",
    resume: bool = True,
    label: str = "",
) -> dict[str, Any]:
    """Run *task* once per item in an isolated context, recording results to disk."""
    if executor.isolated_turn is None:
        return {"error": "Batch processing is not available.", "success": False}
    if not task.strip():
        return {"error": "task is required.", "success": False}
    if not _RUN_NAME_RE.fullmatch(name):
        return {
            "error": (
                "name is required: 1-64 characters, starting with a letter or digit "
                "and containing only letters, digits, '.', '_', or '-'. Reusing a "
                "name resumes that run."
            ),
            "success": False,
        }

    resolved, error = _resolve_items(executor, items, items_file)
    if error is not None:
        return {"error": error, "success": False}

    try:
        output_path = _output_path(executor, name)
    except OSError as e:
        return {"error": f"Cannot create the batch results directory: {e}", "success": False}
    output = _display_path(executor, output_path)
    done = _completed_items(output_path) if resume else set()
    # Positions come from the full list, so an item's index means the same
    # thing in a resumed run as in the original one.
    pending = [
        (position, item)
        for position, item in enumerate(resolved, start=1)
        if item not in done
    ]
    skipped = len(resolved) - len(pending)

    if not pending:
        return {
            "message": (
                f"Nothing to do: all {len(resolved)} item(s) are already recorded "
                f"in {output}."
            ),
            "output": output,
            "total": len(resolved),
            "succeeded": 0,
            "failed": 0,
            "skipped": skipped,
            "failures": [],
            "interrupted": False,
            "success": True,
        }

    # resume=False means a fresh run, not another generation appended to stale
    # records: an older success would otherwise make a later resume skip an item
    # whose fresh attempt failed. The old file is moved aside rather than
    # truncated, so a mistaken rerun cannot destroy a completed run's results.
    backup = None
    if not resume and output_path.is_file():
        backup_path = output_path.with_name(output_path.name + BACKUP_SUFFIX)
        try:
            output_path.replace(backup_path)
        except OSError as e:
            return {
                "error": f"Cannot move the previous results aside: {e}",
                "success": False,
            }
        backup = _display_path(executor, backup_path)

    try:
        # Append: a resumed run continues its file, and a fresh one has just
        # moved any previous results away, so nothing stale can remain.
        handle = output_path.open("a", encoding="utf-8")
    except OSError as e:
        return {"error": f"Cannot write results to {output}: {e}", "success": False}

    succeeded = 0
    failed = 0
    failures: list[dict[str, str]] = []
    interrupted = False
    aborted = False

    try:
        # One indicator for the whole run, updated in place. An indicator per
        # item would publish a start/stop event pair each time, and a long run
        # would evict the real conversation from the bounded event history that
        # reconnecting web clients replay; each restart also costs a tick-thread
        # join, which is pure latency multiplied by the item count.
        with executor.console.thinking(
            label=label or "Batch", status_suffix=f"0/{len(pending)}"
        ) as indicator:
            for attempt, (index, item) in enumerate(pending, start=1):
                indicator.set_status_suffix(f"{attempt}/{len(pending)}")
                try:
                    response, interrupted = executor.isolated_turn(
                        _build_prompt(task, item)
                    )
                    failure = None if response else "The turn produced no response."
                except Exception as e:
                    response, failure = None, f"{type(e).__name__}: {e}"

                if interrupted:
                    # The item never really ran; leave it unrecorded so a
                    # resumed run picks it up again.
                    break

                ok = failure is None
                _append_record(handle, {
                    "item": item,
                    "index": index,
                    "ok": ok,
                    "result": response,
                    "error": failure,
                })
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                    if len(failures) < MAX_REPORTED_FAILURES:
                        failures.append({"item": item, "error": failure})
                    if succeeded == 0 and failed >= EARLY_ABORT_FAILURES:
                        aborted = True
                        break
    finally:
        handle.close()

    processed = succeeded + failed
    if interrupted:
        headline = f"Batch interrupted after {processed}/{len(pending)} item(s)"
    elif aborted:
        headline = (
            f"Batch aborted after {failed} consecutive failure(s) with no success — "
            "check the task instructions before retrying"
        )
    else:
        headline = f"Batch complete: {processed} item(s) processed"

    remaining = len(pending) - processed
    parts = [f"{succeeded} succeeded", f"{failed} failed"]
    if skipped:
        parts.append(f"{skipped} already done")
    if remaining:
        parts.append(f"{remaining} not attempted")

    return {
        "message": (
            f"{headline} ({', '.join(parts)}). Results: {output}. "
            "Read that file for per-item results; they are not in this reply."
            + (f" Previous results were kept at {backup}." if backup else "")
        ),
        "output": output,
        "total": len(resolved),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "failures": failures,
        "interrupted": interrupted,
        "success": True,
    }


def _describe_run_batch(args: dict[str, Any]) -> ToolCallDescription:
    source = args.get("items_file") or f"{len(args.get('items') or [])} items"
    primary = quote_tool_call_value(args.get("label") or args.get("task", ""))
    return ToolCallDescription("run_batch", primary, (str(source), f"→ {args['name']}"))


TOOLS = [
    {
        "permission": "risky",
        "run": run_batch,
        "describe": _describe_run_batch,
        "schema": {
            "type": "function",
            "function": {
                "name": "run_batch",
                "description": (
                    "Run one identical task over many independent items, each in a "
                    "fresh context, appending per-item results to a JSONL file. "
                    "Returns only counts and the output path, never the results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": (
                                "Self-contained instruction for one item. '{item}' is "
                                "replaced with the item; without it the item is appended. "
                                "The item turn sees no prior conversation."
                            ),
                        },
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                f"Inline item list (max {MAX_INLINE_ITEMS}). Use items_file "
                                "for anything longer or produced by a command."
                            ),
                        },
                        "items_file": {
                            "type": "string",
                            "description": (
                                "File with one item per line ('#' comments and blank "
                                "lines skipped). Build it with a command rather than "
                                "by writing out a long literal list. Duplicate items "
                                "resume as one, so de-duplicate the file if each "
                                "occurrence must be processed."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Short identifier for this run (letters, digits, '.', "
                                "'_', '-'). Results are written to "
                                f".kia/{BATCH_DIR_NAME}/<name>.jsonl, one record per item: "
                                "item, index (its position in the item list), ok, "
                                "result, error. Reuse the same name to resume a run."
                            ),
                        },
                        "resume": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Skip items already recorded successfully under this name "
                                "(default: true). Failed items are retried. False restarts "
                                f"the run, keeping the old file as <name>.jsonl{BACKUP_SUFFIX}."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "Short label shown in the progress indicator.",
                        },
                    },
                    "required": ["task", "name"],
                },
            },
        },
    },
]
