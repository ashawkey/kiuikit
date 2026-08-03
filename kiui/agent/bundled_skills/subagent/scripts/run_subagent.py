#!/usr/bin/env python3
"""Run one independent kia agent and emit its result as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from kiui.agent import run_agent
from kiui.agent.models import REASONING_EFFORTS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task", help="self-contained task for the subagent")
    task.add_argument("--task-file", type=Path, help="UTF-8 file containing the task")
    parser.add_argument("--model-alias", help="configured model alias")
    parser.add_argument("--persona", help="persona name (default: coder)")
    parser.add_argument("--work-dir", type=Path, help="subagent working directory")
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORTS)
    return parser.parse_args(argv)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        task = args.task
        if args.task_file is not None:
            task = args.task_file.read_text(encoding="utf-8")

        # Keep stdout machine-readable even if construction reports discovery
        # warnings before run_agent's normal quiet-output context begins.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = run_agent(
                task,
                model_alias=args.model_alias,
                persona=args.persona,
                work_dir=args.work_dir,
                reasoning_effort=args.reasoning_effort,
            )
        _emit({
            "success": result.success,
            "outcome": result.outcome.value,
            "response": result.response,
            "error": result.error,
            "token_usage": result.token_usage,
        })
        if result.success:
            return 0
        if result.outcome.value == "user_interrupted":
            return 130
        return 1
    except KeyboardInterrupt:
        _emit({
            "success": False,
            "outcome": "user_interrupted",
            "response": None,
            "error": "Subagent interrupted.",
            "token_usage": {},
        })
        return 130
    except Exception as exc:
        _emit({
            "success": False,
            "outcome": "failed",
            "response": None,
            "error": f"{type(exc).__name__}: {exc}",
            "token_usage": {},
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
