"""Command-line interface for the Git-backed kia skill library."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from kiui.config import conf
from kiui.agent.library import (
    LibraryError,
    install_skill,
    list_local_skills,
    list_skills,
    remove_skill,
    upload_skill,
)
from kiui.agent.skills import BUNDLED_SKILLS_DIR


def _repo() -> str:
    config = conf if isinstance(conf, dict) else {}
    repo = config.get("kia_lib")
    if not isinstance(repo, str) or not repo.strip():
        raise LibraryError(
            "kia_lib is not configured; add a GitHub repository URL to .kiui.yaml, "
            "for example: kia_lib: git@github.com:user/kia-skills.git"
        )
    return repo.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kib",
        description="Manage skills in a personal GitHub-backed library.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    list_command = commands.add_parser("list", help="list available skills")
    list_command.add_argument(
        "--local", action="store_true", help="list skills installed in this project"
    )

    install = commands.add_parser("install", help="install a library skill into this project")
    install.add_argument("name")

    remove = commands.add_parser("remove", help="remove a skill from the library")
    remove.add_argument("name")

    upload = commands.add_parser("upload", help="upload a project skill to the library")
    upload.add_argument("name")
    upload.add_argument("--force", action="store_true", help="replace an existing library skill")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    try:
        local_names: set[str] = set()

        if args.command == "list":
            if not args.local:
                repo = _repo()
                skills, errors = list_skills(repo)
                local_names = set(list_local_skills()[0])
                title = "Skill Library"
                console.print(f"[bold]{title}[/bold] ({repo})")
            else:
                skills, errors = list_local_skills()
                title = "Local Skills"
                console.print(f"[bold]{title}[/bold]")
            for name in sorted(skills):
                info = skills[name]
                label = Text(f"• {name}", style="magenta")
                if name in local_names:
                    label.append(" (installed)", style="green")
                if args.local and (BUNDLED_SKILLS_DIR / name / "SKILL.md").is_file():
                    label.append(" (built-in)", style="green")
                console.print(label)
                console.print(Padding(Text(info["description"], style="grey50"), (0, 0, 0, 2)))
            for issue in errors:
                Console(stderr=True).print(
                    f"[yellow]Warning:[/yellow] {issue['name']}: {issue['reason']}"
                )
            return 0

        repo = _repo()
        if args.command == "install":
            dest = install_skill(repo, args.name)
            console.print(f"Installed [cyan]{args.name}[/cyan] to {dest}")
            return 0

        if args.command == "remove":
            commit = remove_skill(repo, args.name)
            console.print(f"Removed [cyan]{args.name}[/cyan] ({commit[:12]})")
            return 0

        commit = upload_skill(repo, args.name, force=args.force)
        if commit is None:
            console.print(f"[cyan]{args.name}[/cyan] is already up to date.")
        else:
            console.print(f"Uploaded [cyan]{args.name}[/cyan] ({commit[:12]})")
        return 0
    except LibraryError as exc:
        Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
