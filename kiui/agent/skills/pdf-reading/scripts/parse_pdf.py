#!/usr/bin/env python3
"""Run MinerU on a PDF and report validated, readable output artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

BACKENDS = ("pipeline", "vlm-engine", "hybrid-engine")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="input PDF")
    parser.add_argument("--output", type=Path, default=Path(".kia/pdf-cache"))
    parser.add_argument("--backend", choices=BACKENDS, default="pipeline")
    parser.add_argument("--method", choices=("auto", "txt", "ocr"), default="auto")
    parser.add_argument("--effort", choices=("medium", "high"), default="medium")
    parser.add_argument("--image-analysis", choices=("true", "false"), default="true")
    parser.add_argument("--start", type=int, help="first page, zero-based")
    parser.add_argument("--end", type=int, help="last page, zero-based and inclusive")
    parser.add_argument("--api-url", help="existing local MinerU FastAPI base URL")
    parser.add_argument(
        "--mineru",
        default=os.environ.get("MINERU_BIN", "mineru"),
        help="MinerU executable (default: MINERU_BIN or mineru)",
    )
    parser.add_argument("--force", action="store_true", help="parse even if valid output exists")
    return parser.parse_args(argv)


def expected_parse_dir(output: Path, stem: str, backend: str, method: str) -> Path:
    if backend.startswith("pipeline"):
        leaf = method
    elif backend.startswith("vlm"):
        leaf = "vlm"
    else:
        leaf = f"hybrid_{method}"
    return output / stem / leaf


def locate_markdown(output: Path, stem: str, backend: str, method: str) -> Path:
    expected = expected_parse_dir(output, stem, backend, method) / f"{stem}.md"
    if expected.is_file():
        return expected

    matches = sorted((output / stem).glob(f"**/{stem}.md"))
    if not matches:
        raise FileNotFoundError(f"MinerU did not produce {stem}.md under {output / stem}")
    if len(matches) > 1:
        paths = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(f"Ambiguous MinerU outputs; expected {expected}:\n  {paths}")
    return matches[0]


def validate_output(markdown_path: Path) -> dict:
    stem = markdown_path.stem
    parse_dir = markdown_path.parent
    content_list_path = parse_dir / f"{stem}_content_list.json"
    if markdown_path.stat().st_size == 0:
        raise ValueError(f"Empty Markdown output: {markdown_path}")
    if not content_list_path.is_file():
        raise FileNotFoundError(f"Missing content list: {content_list_path}")

    content = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError(f"Content list is not a list: {content_list_path}")

    blocks = [block for block in content if isinstance(block, dict)]
    pages = [block["page_idx"] for block in blocks if isinstance(block.get("page_idx"), int)]
    images_dir = parse_dir / "images"
    image_count = sum(path.is_file() for path in images_dir.iterdir()) if images_dir.is_dir() else 0

    return {
        "markdown_path": str(markdown_path.resolve()),
        "content_list_path": str(content_list_path.resolve()),
        "content_list_v2_path": str((parse_dir / f"{stem}_content_list_v2.json").resolve()),
        "images_dir": str(images_dir.resolve()),
        "markdown_bytes": markdown_path.stat().st_size,
        "content_blocks": len(content),
        "page_count": max(pages) + 1 if pages else 0,
        "block_types": dict(sorted(Counter(block.get("type", "unknown") for block in blocks).items())),
        "extracted_images": image_count,
    }


def find_executable(value: str) -> str | None:
    executable = shutil.which(value)
    if executable:
        return executable

    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    if value == "mineru":
        relative_paths = (
            ".kia/mineru-venv/bin/mineru",
            ".venv/bin/mineru",
            "venv/bin/mineru",
            ".kia/mineru-venv/Scripts/mineru.exe",
            ".venv/Scripts/mineru.exe",
            "venv/Scripts/mineru.exe",
        )
        for path in relative_paths:
            candidate = Path.cwd() / path
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def build_command(args: argparse.Namespace, pdf: Path, output: Path, executable: str) -> list[str]:
    command = [
        executable,
        "-p", str(pdf),
        "-o", str(output),
        "-b", args.backend,
        "-m", args.method,
        "--effort", args.effort,
        "--image-analysis", args.image_analysis,
        "-f", "true",
        "-t", "true",
    ]
    if args.start is not None:
        command.extend(("-s", str(args.start)))
    if args.end is not None:
        command.extend(("-e", str(args.end)))
    if args.api_url:
        command.extend(("--api-url", args.api_url))
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pdf = args.pdf.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Input is not a PDF: {pdf}")
    if args.start is not None and args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.end is not None and args.end < 0:
        raise ValueError("--end must be non-negative")
    if args.start is not None and args.end is not None and args.end < args.start:
        raise ValueError("--end must not be less than --start")

    if not args.force:
        try:
            manifest = validate_output(
                locate_markdown(output, pdf.stem, args.backend, args.method)
            )
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
        else:
            manifest["reused"] = True
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0

    executable = find_executable(args.mineru)
    if executable is None:
        raise FileNotFoundError(
            f"MinerU executable not found: {args.mineru}. MinerU is required; "
            "do not substitute another PDF parser. Check `uv tool list` and "
            "project environments, then retry with `--mineru <path>` or ask "
            "before installing it."
        )

    output.mkdir(parents=True, exist_ok=True)
    command = build_command(args, pdf, output, executable)
    print(f"Running MinerU: {subprocess.list2cmdline(command)}", file=sys.stderr)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"MinerU exited with status {completed.returncode}")

    manifest = validate_output(locate_markdown(output, pdf.stem, args.backend, args.method))
    manifest["reused"] = False
    manifest["mineru_executable"] = executable
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
