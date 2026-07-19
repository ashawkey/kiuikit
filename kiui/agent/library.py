"""Git-backed personal skill library used by the ``kib`` command."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock

from kiui.agent.skills import BUNDLED_SKILLS_DIR, read_skill, valid_skill_name


class LibraryError(RuntimeError):
    """A user-facing skill library error."""


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise LibraryError("Git is not installed or is not available on PATH") from exc
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LibraryError(f"git {' '.join(args)} failed: {detail}")
    return result


def _validate_repo(repo: str) -> str:
    if not isinstance(repo, str) or not repo.strip():
        raise LibraryError(
            "kia_lib is not configured; add a GitHub repository URL to .kiui.yaml, "
            "for example: kia_lib: git@github.com:user/kia-skills.git"
        )
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
        creating = not root.exists()
        try:
            if creating:
                _git("clone", "--quiet", repo, str(root))
            elif not (root / ".git").is_dir():
                raise LibraryError(f"invalid library cache: {root}")
            _sync_checkout(root)
        except LibraryError as exc:
            if creating and root.exists():
                shutil.rmtree(root)
            if str(exc).startswith((
                "Git is not installed",
                "invalid library cache",
                "repository has no main branch",
            )):
                raise
            raise LibraryError(
                "cannot access the kia_lib repository using the current Git "
                f"credentials: {exc}"
            ) from exc
        yield root


def _validate_tree(root: Path) -> None:
    if root.is_symlink():
        raise LibraryError(f"skill directory is a symlink: {root}")
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in dirs:
            path = base / name
            relative = path.relative_to(root)
            if name == ".git":
                raise LibraryError(f"skill contains Git metadata: {relative}")
            mode = path.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise LibraryError(f"skill contains a symlink: {relative}")
            if not stat.S_ISDIR(mode):
                raise LibraryError(f"skill contains an invalid directory: {relative}")
        for name in files:
            path = base / name
            relative = path.relative_to(root)
            if name == ".git":
                raise LibraryError(f"skill contains Git metadata: {relative}")
            mode = path.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise LibraryError(f"skill contains a symlink: {relative}")
            if not stat.S_ISREG(mode):
                raise LibraryError(f"skill contains a non-regular file: {relative}")


def _load_strict(skill_dir: Path) -> dict:
    try:
        _validate_tree(skill_dir)
        return read_skill(skill_dir, strict=True)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LibraryError(f"invalid skill '{skill_dir.name}': {exc}") from exc


def _skills_root(checkout: Path) -> Path:
    skills_dir = checkout / "skills"
    if skills_dir.is_symlink():
        raise LibraryError("library skills directory is a symlink")
    if skills_dir.exists() and not skills_dir.is_dir():
        raise LibraryError("library skills path is not a directory")
    return skills_dir


def _copy_tree(source: Path, dest: Path) -> None:
    """Copy regular files without following source symlinks."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)

    def copy_dir(source_fd: int, target: Path) -> None:
        for entry in os.scandir(source_fd):
            relative = target / entry.name
            if entry.name == ".git":
                raise LibraryError(f"skill contains Git metadata: {entry.name}")
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise LibraryError(f"skill contains a symlink: {entry.name}")
            if stat.S_ISDIR(mode):
                relative.mkdir()
                child_fd = os.open(
                    entry.name, os.O_RDONLY | directory | nofollow, dir_fd=source_fd
                )
                try:
                    copy_dir(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(mode):
                source_file = os.open(entry.name, os.O_RDONLY | nofollow, dir_fd=source_fd)
                try:
                    opened_mode = os.fstat(source_file).st_mode
                    if not stat.S_ISREG(opened_mode):
                        raise LibraryError(f"skill contains a non-regular file: {entry.name}")
                    with os.fdopen(source_file, "rb", closefd=False) as src, relative.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    relative.chmod(stat.S_IMODE(opened_mode))
                finally:
                    os.close(source_file)
            else:
                raise LibraryError(f"skill contains a non-regular file: {entry.name}")

    dest.mkdir()
    source_fd = os.open(source, os.O_RDONLY | directory | nofollow)
    try:
        copy_dir(source_fd, dest)
    finally:
        os.close(source_fd)


@contextmanager
def _snapshot_skill(source: Path) -> Iterator[Path]:
    """Copy and validate an immutable snapshot of a local skill."""
    with tempfile.TemporaryDirectory(prefix="kia-skill-") as temp_dir:
        snapshot = Path(temp_dir) / source.name
        try:
            _copy_tree(source, snapshot)
        except OSError as exc:
            raise LibraryError(f"cannot snapshot local skill '{source.name}': {exc}") from exc
        _load_strict(snapshot)
        yield snapshot


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

    for item in sorted(skills_dir.iterdir(), key=lambda path: path.name):
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
        skills_dir = _skills_root(checkout)
        skills: dict[str, dict] = {}
        errors: list[dict] = []
        if not skills_dir.is_dir():
            return skills, errors

        for item in sorted(skills_dir.iterdir(), key=lambda path: path.name):
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
    skill_dir = _skills_root(checkout) / name
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


def _tree_manifest(root: Path) -> dict[str, tuple[str, bool]]:
    manifest: dict[str, tuple[str, bool]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        executable = bool(path.stat().st_mode & 0o111)
        manifest[path.relative_to(root).as_posix()] = (digest, executable)
    return manifest


def _same_tree(left: Path, right: Path) -> bool:
    return _tree_manifest(left) == _tree_manifest(right)


def remove_skill(repo: str, name: str) -> str:
    """Remove a skill from the library and return the new commit hash."""
    if not valid_skill_name(name):
        raise LibraryError(f"invalid skill name: {name!r}")

    with _checkout(repo) as checkout:
        dest = _skills_root(checkout) / name
        if not dest.exists() and not dest.is_symlink():
            raise LibraryError(f"skill not found in library: {name}")

        _git("rm", "--quiet", "-r", "--", f"skills/{name}", cwd=checkout)
        _git("commit", "--quiet", "-m", f"skill: remove {name}", cwd=checkout)
        commit = _git("rev-parse", "HEAD", cwd=checkout).stdout.strip()
        _git("push", "--quiet", "origin", "main", cwd=checkout)
        return commit


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
    with _snapshot_skill(source) as snapshot:
        with _checkout(repo) as checkout:
            dest = _skills_root(checkout) / name
            updating = dest.exists() or dest.is_symlink()
            if updating:
                _load_strict(dest)
                if _same_tree(snapshot, dest):
                    return None
                if not force:
                    raise LibraryError(
                        f"skill already exists in library: {name} (use --force to update it)"
                    )
                shutil.rmtree(dest)

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(snapshot, dest)
            _git("add", "--force", "--", f"skills/{name}", cwd=checkout)
            action = "update" if updating else "add"
            _git("commit", "--quiet", "-m", f"skill: {action} {name}", cwd=checkout)
            commit = _git("rev-parse", "HEAD", cwd=checkout).stdout.strip()
            _git("push", "--quiet", "origin", "main", cwd=checkout)
            return commit
