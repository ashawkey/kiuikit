"""Tests for the Git-backed personal skill library."""

import os
import tempfile
from pathlib import Path
import subprocess

import pytest

from kiui.agent.skills import read_skill
from kiui.agent.library import (
    LibraryError,
    _validate_repo,
    install_skill,
    list_local_skills,
    list_skills,
    remove_local_skill,
    remove_skill,
    update_skill,
    upload_skill,
)


def _symlinks_supported() -> bool:
    """Windows requires elevated privileges (or Developer Mode) for symlinks."""
    with tempfile.TemporaryDirectory() as d:
        try:
            Path(d, "link").symlink_to(Path(d, "target"))
            return True
        except OSError:
            return False


symlink_required = pytest.mark.skipif(
    not _symlinks_supported(), reason="symlinks not permitted on this system"
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = str(tmp_path / "home")
    # Path.home() follows HOME on POSIX, USERPROFILE on Windows.
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    # Env-based git identity: works regardless of the (now hidden) gitconfig.
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "kia-tests")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "kia-tests@example.com")


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def _skill(root: Path, name: str, description: str = "Useful skill") -> Path:
    path = root / ".kia" / "skills" / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nDo it.\n",
        encoding="utf-8",
    )
    return path


def _remote(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    remote = tmp_path / "library.git"
    _git("init", "--bare", "--initial-branch=main", str(remote))
    return remote


def _seed_remote(remote: Path, tmp_path: Path, name: str = "alpha") -> None:
    project = tmp_path / f"seed-{name}"
    _skill(project, name)
    upload_skill(str(remote), name, project)


def test_upload_list_and_install_skill(tmp_path):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    skill = _skill(source, "alpha", "Alpha description")
    (skill / "references").mkdir()
    (skill / "references" / "notes.md").write_text("notes", encoding="utf-8")

    commit = upload_skill(str(remote), "alpha", source)
    skills, errors = list_skills(str(remote))
    dest = install_skill(str(remote), "alpha", tmp_path / "target")

    assert len(commit) == 40
    assert skills["alpha"]["description"] == "Alpha description"
    assert errors == []
    assert (dest / "references" / "notes.md").read_text() == "notes"


def test_remove_skill(tmp_path):
    remote = _remote(tmp_path)
    _seed_remote(remote, tmp_path)

    commit = remove_skill(str(remote), "alpha")
    skills, errors = list_skills(str(remote))

    assert len(commit) == 40
    assert skills == {}
    assert errors == []


def test_update_skill_pulls_changed_library_copy(tmp_path):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    skill = _skill(source, "alpha", "First")
    upload_skill(str(remote), "alpha", source)

    target = tmp_path / "target"
    dest = install_skill(str(remote), "alpha", target)
    assert update_skill(str(remote), "alpha", target) == "current"

    (skill / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Second\n---\nUpdated.\n",
        encoding="utf-8",
    )
    upload_skill(str(remote), "alpha", source, force=True)

    assert update_skill(str(remote), "alpha", target) == "pulled"
    assert read_skill(dest)["description"] == "Second"


def test_update_skill_rejects_conflicting_changes(tmp_path):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    skill = _skill(source, "alpha", "First")
    upload_skill(str(remote), "alpha", source)

    target = tmp_path / "target"
    dest = install_skill(str(remote), "alpha", target)
    (dest / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Local\n---\nLocal.\n",
        encoding="utf-8",
    )
    (skill / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Remote\n---\nRemote.\n",
        encoding="utf-8",
    )
    upload_skill(str(remote), "alpha", source, force=True)

    with pytest.raises(LibraryError, match="both copies changed.*--prefer"):
        update_skill(str(remote), "alpha", target)
    assert update_skill(str(remote), "alpha", target, prefer="local") == "pushed"
    assert list_skills(str(remote))[0]["alpha"]["description"] == "Local"


@symlink_required
def test_library_rejects_symlinked_skills_root(tmp_path):
    remote = _remote(tmp_path)
    seed = tmp_path / "seed"
    seed.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (seed / "skills").symlink_to(victim, target_is_directory=True)
    _git("init", "--initial-branch=main", cwd=seed)
    _git("add", "skills", cwd=seed)
    _git("commit", "-m", "symlink skills", cwd=seed)
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "origin", "main", cwd=seed)

    with pytest.raises(LibraryError, match="skills directory is a symlink"):
        list_skills(str(remote))
    assert victim.is_dir()
