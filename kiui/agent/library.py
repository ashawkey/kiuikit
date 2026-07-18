"""Git-backed personal skill library used by the ``kib`` command."""

from __future__ import annotations

import filecmp
import hashlib
import os
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock

from kiui.agent.skills import BUNDLED_SKILLS_DIR, read_skill, valid_skill_name


class LibraryError(RuntimeError):
    """A user-facing skill library error."""


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LibraryError(f"git {' '.join(args)} failed: {detail}")
    return result


def _validate_repo(repo: str) -> str:
    if not isinstance(repo, str) or not repo.strip():
        raise LibraryError("missing 'kia_lib' GitHub repository in .kiui.yaml")
    return repo.strip()


def _sync_checkout(root: Path) -> None:
    """Reset the managed checkout to the remote main branch."""
    _git("reset", "--hard", cwd=root, check=False)
    _git("clean", "-fdx", cwd=root)
    _git("fetch", "--quiet", "origin", cwd=root)

    main = _git(
        "ls-remote", "--exit-code", "--heads", "origin", "refs/heads/main",
        cwd=root,
        check=False,
    )
    if main.returncode == 0:
        _git("checkout", "--quiet", "-B", "main", "origin/main", cwd=root)
    elif not _git("ls-remote", "origin", cwd=root).stdout.strip():
        # An empty remote has no branch yet; the first upload creates main.
        _git("update-ref", "-d", "refs/heads/main", cwd=root)
        _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=root)
        _git("reset", cwd=root, check=False)
        _git("clean", "-fdx", cwd=root)
    else:
        raise LibraryError("repository has no main branch")


@contextmanager
def _checkout(repo: str) -> Iterator[Path]:
    """Lock and refresh a persistent checkout fixed to the main branch."""
    repo = _validate_repo(repo)
    library_root = Path.home() / ".kia" / "library"
    key = hashlib.sha256(repo.encode()).hexdigest()[:16]
    root = library_root / key
    library_root.mkdir(parents=True, exist_ok=True)

    with FileLock(str(library_root / f"{key}.lock")):
        if not root.exists():
            _git("clone", "--quiet", repo, str(root))
        elif not (root / ".git").is_dir():
            raise LibraryError(f"invalid library cache: {root}")
        _sync_checkout(root)
        yield root


def _reject_symlinks(root: Path) -> None:
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in dirs + files:
            path = base / name
            if path.is_symlink():
                raise LibraryError(f"skill contains a symlink: {path.relative_to(root)}")


def _load_strict(skill_dir: Path) -> dict:
    if skill_dir.is_symlink():
        raise LibraryError(f"skill directory is a symlink: {skill_dir}")
    try:
        info = read_skill(skill_dir, strict=True)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LibraryError(f"invalid skill '{skill_dir.name}': {exc}") from exc
    _reject_symlinks(skill_dir)
    return info


def list_local_skills(
    work_dir: str | Path | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    """Return skills installed in the current project's ``.kia`` directory."""
    base = Path(work_dir) if work_dir is not None else Path.cwd()
    skills_dir = base / ".kia" / "skills"
    skills: dict[str, dict] = {}
    errors: list[dict] = []
    if not skills_dir.is_dir():
        return skills, errors

    for item in sorted(skills_dir.iterdir()):
        if not item.is_dir():
            continue
        try:
            skills[item.name] = read_skill(item)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            errors.append({"name": item.name, "reason": str(exc)})
    return skills, errors


def list_skills(repo: str) -> tuple[dict[str, dict], list[dict]]:
    """Return valid marketplace skills and malformed-entry diagnostics."""
    with _checkout(repo) as checkout:
        skills_dir = checkout / "skills"
        skills: dict[str, dict] = {}
        errors: list[dict] = []
        if not skills_dir.is_dir():
            return skills, errors

        for item in sorted(skills_dir.iterdir()):
            if not item.is_dir() or item.is_symlink():
                continue
            try:
                info = _load_strict(item)
                skills[item.name] = {
                    "description": info["description"],
                    "frontmatter": info["frontmatter"],
                }
            except LibraryError as exc:
                errors.append({"name": item.name, "reason": str(exc)})
        return skills, errors


def _remote_skill(checkout: Path, name: str) -> Path:
    if not valid_skill_name(name):
        raise LibraryError(f"invalid skill name: {name!r}")
    skill_dir = checkout / "skills" / name
    if not skill_dir.is_dir():
        raise LibraryError(f"skill not found in library: {name}")
    _load_strict(skill_dir)
    return skill_dir


def install_skill(repo: str, name: str, work_dir: str | Path | None = None) -> Path:
    """Install one remote skill into ``<work_dir>/.kia/skills`` atomically."""
    base = Path(work_dir) if work_dir is not None else Path.cwd()
    dest_root = base / ".kia" / "skills"
    dest = dest_root / name
    if dest.exists() or dest.is_symlink():
        raise LibraryError(f"local skill already exists: {dest}")

    with _checkout(repo) as checkout:
        source = _remote_skill(checkout, name)
        dest_root.mkdir(parents=True, exist_ok=True)
        staging = dest_root / f".{name}.tmp-{uuid.uuid4().hex}"
        try:
            shutil.copytree(source, staging)
            os.replace(staging, dest)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return dest


def _same_tree(left: Path, right: Path) -> bool:
    comparison = filecmp.dircmp(left, right)
    if comparison.left_only or comparison.right_only or comparison.funny_files:
        return False
    if any(not filecmp.cmp(left / name, right / name, shallow=False) for name in comparison.common_files):
        return False
    return all(_same_tree(left / name, right / name) for name in comparison.common_dirs)


def upload_skill(
    repo: str,
    name: str,
    work_dir: str | Path | None = None,
    force: bool = False,
) -> str | None:
    """Commit and push a project skill to the library's main branch.

    Returns the new commit hash, or ``None`` when the remote copy is identical.
    """
    if not valid_skill_name(name):
        raise LibraryError(f"invalid skill name: {name!r}")
    if (BUNDLED_SKILLS_DIR / name / "SKILL.md").is_file():
        raise LibraryError(f"bundled skill cannot be uploaded: {name}")
    base = Path(work_dir) if work_dir is not None else Path.cwd()
    source = base / ".kia" / "skills" / name
    if not source.is_dir():
        raise LibraryError(f"local project skill not found: {source}")
    _load_strict(source)

    with _checkout(repo) as checkout:
        dest = checkout / "skills" / name
        updating = dest.exists()
        if updating:
            _load_strict(dest)
            if _same_tree(source, dest):
                return None
            if not force:
                raise LibraryError(
                    f"skill already exists in library: {name} (use --force to update it)"
                )
            shutil.rmtree(dest)

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest)
        _git("add", "--force", "--", f"skills/{name}", cwd=checkout)
        action = "update" if updating else "add"
        _git("commit", "--quiet", "-m", f"skill: {action} {name}", cwd=checkout)
        commit = _git("rev-parse", "HEAD", cwd=checkout).stdout.strip()
        _git("push", "--quiet", "origin", "main", cwd=checkout)
        return commit
