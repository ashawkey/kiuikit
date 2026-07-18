"""Command-line interface for the Git-backed kia skill library."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from kiui.config import conf
from kiui.agent.library import LibraryError, install_skill, list_skills, upload_skill


def _repo() -> str:
    config = conf if isinstance(conf, dict) else {}
    repo = config.get("kia_lib")
    if not isinstance(repo, str) or not repo.strip():
        raise LibraryError("missing 'kia_lib' GitHub repository in .kiui.yaml")
    return repo.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kib",
        description="Manage skills in a personal GitHub-backed library.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list skills available in the library")

    install = commands.add_parser("install", help="install a library skill into this project")
    install.add_argument("name")

    upload = commands.add_parser("upload", help="upload a project skill to the library")
    upload.add_argument("name")
    upload.add_argument("--force", action="store_true", help="replace an existing library skill")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    try:
        repo = _repo()
        if args.command == "list":
            skills, errors = list_skills(repo)
            table = Table(title="Kia Skill Library")
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Description")
            for name, info in skills.items():
                table.add_row(name, info["description"])
            console.print(table)
            for issue in errors:
                console.print(
                    f"[yellow]Warning:[/yellow] {issue['name']}: {issue['reason']}",
                    stderr=True,
                )
            return 0

        if args.command == "install":
            dest = install_skill(repo, args.name)
            console.print(f"Installed [cyan]{args.name}[/cyan] to {dest}")
            return 0

        commit = upload_skill(repo, args.name, force=args.force)
        if commit is None:
            console.print(f"[cyan]{args.name}[/cyan] is already up to date.")
        else:
            console.print(f"Uploaded [cyan]{args.name}[/cyan] ({commit[:12]})")
        return 0
    except LibraryError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}", stderr=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
