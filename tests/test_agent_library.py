"""Tests for the Git-backed personal skill library."""

import os
import tempfile
from pathlib import Path
import subprocess

import pytest

from kiui.agent.library import (
    LibraryError,
    _validate_repo,
    install_skill,
    list_local_skills,
    list_skills,
    remove_skill,
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


def test_missing_repo_error_explains_configuration():
    with pytest.raises(LibraryError, match="kia_lib is not configured.*kia_lib:"):
        _validate_repo("")


def test_inaccessible_repo_error_is_clear(tmp_path):
    missing = tmp_path / "does-not-exist.git"
    with pytest.raises(LibraryError, match="cannot access the kia_lib repository"):
        list_skills(str(missing))


def test_cache_is_persistent_and_separated_by_repo(tmp_path):
    first = _remote(tmp_path / "first")
    second = _remote(tmp_path / "second")

    list_skills(str(first))
    cache_root = Path.home() / ".kia" / "library"
    first_caches = {path.name for path in cache_root.iterdir() if path.is_dir()}
    list_skills(str(first))
    assert {path.name for path in cache_root.iterdir() if path.is_dir()} == first_caches

    list_skills(str(second))
    assert len([path for path in cache_root.iterdir() if path.is_dir()]) == 2


def test_remote_list_is_sorted_and_marks_local_skills(monkeypatch, capsys):
    from kiui.agent import library_cli

    monkeypatch.setattr(library_cli, "_repo", lambda: "git@example.com:skills.git")
    monkeypatch.setattr(
        library_cli,
        "list_skills",
        lambda repo: (
            {
                "zeta": {"description": "Last"},
                "alpha": {"description": "First"},
            },
            [],
        ),
    )
    monkeypatch.setattr(
        library_cli,
        "list_local_skills",
        lambda: ({"zeta": {"description": "Local"}}, []),
    )

    assert library_cli.main(["list"]) == 0
    output = capsys.readouterr().out
    assert output.index("• alpha") < output.index("• zeta (installed)")


def test_remote_list_filters_skill_names_by_pattern(monkeypatch, capsys):
    from kiui.agent import library_cli

    monkeypatch.setattr(library_cli, "_repo", lambda: "repo")
    monkeypatch.setattr(
        library_cli,
        "list_skills",
        lambda repo: (
            {
                "image-tools": {"description": "Images"},
                "pdf-reading": {"description": "PDFs"},
            },
            [],
        ),
    )
    monkeypatch.setattr(library_cli, "list_local_skills", lambda: ({}, []))

    assert library_cli.main(["list", "image"]) == 0
    output = capsys.readouterr().out
    assert "• image-tools" in output
    assert "pdf-reading" not in output


def test_local_list_filters_skill_names_by_pattern(monkeypatch, capsys):
    from kiui.agent import library_cli

    monkeypatch.setattr(library_cli, "_configured_repo", lambda: None)
    monkeypatch.setattr(
        library_cli,
        "list_local_skills",
        lambda: (
            {
                "image-tools": {"description": "Images"},
                "pdf-reading": {"description": "PDFs"},
            },
            [],
        ),
    )

    assert library_cli.main(["list", "pdf", "--local"]) == 0
    output = capsys.readouterr().out
    assert "• pdf-reading" in output
    assert "image-tools" not in output


def test_local_list_marks_uploaded_skills(monkeypatch, capsys):
    from kiui.agent import library_cli

    monkeypatch.setattr(
        library_cli, "_configured_repo", lambda: "git@example.com:skills.git"
    )
    monkeypatch.setattr(
        library_cli,
        "list_local_skills",
        lambda: ({"alpha": {"description": "Local"}}, []),
    )
    monkeypatch.setattr(
        library_cli,
        "list_skills",
        lambda repo: ({"alpha": {"description": "Remote"}}, []),
    )

    assert library_cli.main(["list", "--local"]) == 0
    assert "• alpha (uploaded)" in capsys.readouterr().out


def test_local_list_survives_remote_failure(monkeypatch, capsys):
    from kiui.agent import library_cli

    monkeypatch.setattr(
        library_cli, "_configured_repo", lambda: "git@example.com:skills.git"
    )
    monkeypatch.setattr(
        library_cli,
        "list_local_skills",
        lambda: ({"alpha": {"description": "Local"}}, []),
    )
    monkeypatch.setattr(
        library_cli,
        "list_skills",
        lambda repo: (_ for _ in ()).throw(LibraryError("offline")),
    )

    assert library_cli.main(["list", "--local"]) == 0
    output = capsys.readouterr()
    assert "• alpha" in output.out
    assert "could not check uploaded status" in output.err


def test_cli_installs_multiple_skills(monkeypatch, capsys, tmp_path):
    from kiui.agent import library_cli

    calls = []
    monkeypatch.setattr(library_cli, "_repo", lambda: "repo")
    monkeypatch.setattr(
        library_cli,
        "install_skill",
        lambda repo, name: calls.append((repo, name)) or tmp_path / name,
    )

    assert library_cli.main(["install", "alpha", "beta"]) == 0
    assert calls == [("repo", "alpha"), ("repo", "beta")]
    output = capsys.readouterr().out
    assert "Installed alpha" in output
    assert "Installed beta" in output


def test_cli_removes_multiple_skills(monkeypatch, capsys):
    from kiui.agent import library_cli

    calls = []
    monkeypatch.setattr(library_cli, "_repo", lambda: "repo")
    monkeypatch.setattr(
        library_cli,
        "remove_skill",
        lambda repo, name: calls.append((repo, name)) or "a" * 40,
    )

    assert library_cli.main(["remove", "alpha", "beta"]) == 0
    assert calls == [("repo", "alpha"), ("repo", "beta")]
    output = capsys.readouterr().out
    assert "Removed alpha" in output
    assert "Removed beta" in output


def test_cli_uploads_multiple_skills_with_force(monkeypatch, capsys):
    from kiui.agent import library_cli

    calls = []
    monkeypatch.setattr(library_cli, "_repo", lambda: "repo")

    def upload(repo, name, force=False):
        calls.append((repo, name, force))
        return None if name == "beta" else "b" * 40

    monkeypatch.setattr(library_cli, "upload_skill", upload)

    assert library_cli.main(["upload", "alpha", "beta", "--force"]) == 0
    assert calls == [("repo", "alpha", True), ("repo", "beta", True)]
    output = capsys.readouterr().out
    assert "Uploaded alpha" in output
    assert "beta is already up to date" in output


def test_list_local_skills_only_scans_current_project(tmp_path):
    project = tmp_path / "project"
    _skill(project, "alpha", "Local alpha")
    malformed = project / ".kia" / "skills" / "broken"
    malformed.mkdir()
    (malformed / "SKILL.md").write_text("invalid", encoding="utf-8")

    skills, errors = list_local_skills(project)

    assert skills["alpha"]["description"] == "Local alpha"
    assert [error["name"] for error in errors] == ["broken"]


def test_empty_library_lists_no_skills(tmp_path):
    skills, errors = list_skills(str(_remote(tmp_path)))
    assert skills == {}
    assert errors == []


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


def test_remove_missing_skill_fails(tmp_path):
    remote = _remote(tmp_path)
    with pytest.raises(LibraryError, match="not found"):
        remove_skill(str(remote), "alpha")


def test_install_refuses_existing_skill(tmp_path):
    remote = _remote(tmp_path)
    _seed_remote(remote, tmp_path)
    target = tmp_path / "target"
    _skill(target, "alpha")

    with pytest.raises(LibraryError, match="already exists"):
        install_skill(str(remote), "alpha", target)


def test_upload_requires_force_to_update(tmp_path):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    skill = _skill(source, "alpha")
    upload_skill(str(remote), "alpha", source)
    (skill / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Changed\n---\nNew body.\n",
        encoding="utf-8",
    )

    with pytest.raises(LibraryError, match="--force"):
        upload_skill(str(remote), "alpha", source)

    upload_skill(str(remote), "alpha", source, force=True)
    skills, _ = list_skills(str(remote))
    assert skills["alpha"]["description"] == "Changed"


def test_upload_identical_skill_is_noop(tmp_path):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    _skill(source, "alpha")
    upload_skill(str(remote), "alpha", source)
    assert upload_skill(str(remote), "alpha", source) is None


@pytest.mark.skipif(os.name != "posix", reason="executable bit is POSIX-only")
def test_upload_preserves_executable_mode_and_ignores_empty_directories(tmp_path):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    skill = _skill(source, "alpha")
    script = skill / "run.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o644)
    upload_skill(str(remote), "alpha", source)

    script.chmod(0o755)
    (skill / "empty").mkdir()
    assert upload_skill(str(remote), "alpha", source, force=True) is not None
    assert upload_skill(str(remote), "alpha", source) is None

    mode = _git("--git-dir", str(remote), "ls-tree", "main", "skills/alpha/run.sh")
    assert mode.startswith("100755 ")


@symlink_required
def test_upload_uses_validated_snapshot(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from kiui.agent import library

    remote = _remote(tmp_path)
    source = tmp_path / "source"
    skill = _skill(source, "alpha")
    note = skill / "note.txt"
    note.write_text("safe", encoding="utf-8")
    secret = tmp_path / "secret"
    secret.write_text("secret", encoding="utf-8")
    original_checkout = library._checkout

    @contextmanager
    def mutate_after_snapshot(repo):
        note.unlink()
        note.symlink_to(secret)
        with original_checkout(repo) as checkout:
            yield checkout

    monkeypatch.setattr(library, "_checkout", mutate_after_snapshot)
    upload_skill(str(remote), "alpha", source)

    content = _git("--git-dir", str(remote), "show", "main:skills/alpha/note.txt")
    assert content == "safe"


def test_upload_rejects_git_metadata(tmp_path):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    skill = _skill(source, "alpha")
    (skill / ".git").mkdir()

    with pytest.raises(LibraryError, match="Git metadata"):
        upload_skill(str(remote), "alpha", source)


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


@symlink_required
def test_upload_rejects_invalid_skill_and_symlink(tmp_path):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    skill = _skill(source, "alpha")
    (skill / "SKILL.md").write_text(
        "---\nname: wrong\ndescription: mismatch\n---\nBody\n", encoding="utf-8"
    )
    with pytest.raises(LibraryError, match="does not match"):
        upload_skill(str(remote), "alpha", source)

    (skill / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: valid\n---\nBody\n", encoding="utf-8"
    )
    (skill / "leak").symlink_to(tmp_path / "secret")
    with pytest.raises(LibraryError, match="symlink"):
        upload_skill(str(remote), "alpha", source)
