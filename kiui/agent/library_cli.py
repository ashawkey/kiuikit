"""Command-line interface for the Git-backed kia skill library."""

from __future__ import annotations

import argparse
import logging
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
    remove_local_skill,
    remove_skill,
    update_skill,
    upload_skill,
)


def _configured_repo() -> str | None:
    config = conf if isinstance(conf, dict) else {}
    repo = config.get("kia_lib")
    return repo.strip() if isinstance(repo, str) and repo.strip() else None


def _repo() -> str:
    repo = _configured_repo()
    if repo is None:
        raise LibraryError(
            "kia_lib is not configured; add a GitHub repository URL to .kiui.yaml, "
            "for example: kia_lib: git@github.com:user/kia-skills.git"
        )
    return repo


def _add_verbose(parser: argparse.ArgumentParser, *, global_option: bool = False) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False if global_option else argparse.SUPPRESS,
        help="show detailed operation and Git logs",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kib",
        description="Manage skills in a personal GitHub-backed library.",
    )
    _add_verbose(parser, global_option=True)
    commands = parser.add_subparsers(dest="command", required=True)
    list_command = commands.add_parser("list", help="list available skills")
    _add_verbose(list_command)
    list_command.add_argument("pattern", nargs="?", help="filter skill names by substring")
    list_command.add_argument(
        "--local", action="store_true", help="list skills installed in this project"
    )

    install = commands.add_parser("install", help="install library skills into this project")
    _add_verbose(install)
    install.add_argument("names", nargs="+", metavar="name")

    update = commands.add_parser("update", help="sync installed skills with the library")
    _add_verbose(update)
    update.add_argument("names", nargs="*", metavar="name")
    update.add_argument(
        "--prefer",
        choices=("local", "remote"),
        help="resolve conflicts by keeping the local or remote copy",
    )

    remove = commands.add_parser("remove", help="remove skills from the library or project")
    _add_verbose(remove)
    remove.add_argument("names", nargs="+", metavar="name")
    remove.add_argument("--local", action="store_true", help="remove from this project")

    upload = commands.add_parser("upload", help="upload project skills to the library")
    _add_verbose(upload)
    upload.add_argument("names", nargs="+", metavar="name")
    upload.add_argument("--force", action="store_true", help="replace an existing library skill")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    library_logger = logging.getLogger("kiui.agent.library")
    verbose_handler: logging.Handler | None = None
    previous_level = library_logger.level
    previous_propagate = library_logger.propagate
    if args.verbose:
        verbose_handler = logging.StreamHandler()
        verbose_handler.setFormatter(logging.Formatter("[kib] %(message)s"))
        library_logger.addHandler(verbose_handler)
        library_logger.setLevel(logging.INFO)
        library_logger.propagate = False

    try:
        local_names: set[str] = set()
        remote_names: set[str] = set()

        if args.command == "list":
            if not args.local:
                repo = _repo()
                skills, errors = list_skills(repo)
                local_names = set(list_local_skills()[0])
                title = "Skill Library"
                console.print(f"[bold]{title}[/bold] ({repo})")
            else:
                skills, errors = list_local_skills()
                repo = _configured_repo()
                if repo is not None:
                    try:
                        remote_names = set(list_skills(repo)[0])
                    except LibraryError as exc:
                        Console(stderr=True).print(
                            f"[yellow]Warning:[/yellow] could not check uploaded status: {exc}"
                        )
                title = "Local Skills"
                console.print(f"[bold]{title}[/bold]")
            if args.pattern is not None:
                skills = {name: info for name, info in skills.items() if args.pattern in name}
                errors = [issue for issue in errors if args.pattern in issue["name"]]
            for name in sorted(skills):
                info = skills[name]
                label = Text(f"• {name}", style="magenta")
                if name in local_names:
                    label.append(" (installed)", style="green")
                if name in remote_names:
                    label.append(" (uploaded)", style="green")
                console.print(label)
                console.print(Padding(Text(info["description"], style="grey50"), (0, 0, 0, 2)))
            for issue in errors:
                Console(stderr=True).print(
                    f"[yellow]Warning:[/yellow] {issue['name']}: {issue['reason']}"
                )
            return 0

        if args.command == "remove" and args.local:
            for name in args.names:
                path = remove_local_skill(name)
                console.print(f"Removed local [cyan]{name}[/cyan] from {path}")
            return 0

        repo = _repo()
        if args.command == "install":
            for name in args.names:
                dest = install_skill(repo, name)
                console.print(f"Installed [cyan]{name}[/cyan] to {dest}")
            return 0

        if args.command == "update":
            if args.names:
                names = args.names
            else:
                skills, errors = list_local_skills()
                if errors:
                    detail = ", ".join(issue["name"] for issue in errors)
                    raise LibraryError(f"cannot update invalid local skills: {detail}")
                names = sorted(skills)
            for name in names:
                action = update_skill(repo, name, prefer=args.prefer)
                if action == "current":
                    console.print(f"[cyan]{name}[/cyan] is already up to date.")
                elif action == "pulled":
                    console.print(f"Updated local [cyan]{name}[/cyan] from the library.")
                else:
                    console.print(f"Updated library [cyan]{name}[/cyan] from local.")
            return 0

        if args.command == "remove":
            for name in args.names:
                commit = remove_skill(repo, name)
                console.print(f"Removed [cyan]{name}[/cyan] ({commit[:12]})")
            return 0

        for name in args.names:
            commit = upload_skill(repo, name, force=args.force)
            if commit is None:
                console.print(f"[cyan]{name}[/cyan] is already up to date.")
            else:
                console.print(f"Uploaded [cyan]{name}[/cyan] ({commit[:12]})")
        return 0
    except LibraryError as exc:
        Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    finally:
        if verbose_handler is not None:
            library_logger.removeHandler(verbose_handler)
            library_logger.setLevel(previous_level)
            library_logger.propagate = previous_propagate


if __name__ == "__main__":
    sys.exit(main())
