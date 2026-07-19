"""Tests for the Git-backed personal skill library."""

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


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


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


def test_local_list_marks_uploaded_skills(monkeypatch, capsys):
    from kiui.agent import library_cli

    monkeypatch.setattr(library_cli, "_repo", lambda: "git@example.com:skills.git")
    monkeypatch.setattr(
        library_cli,
        "list_local_skills",
        lambda: (
            {
                "zeta": {"description": "Last"},
                "alpha": {"description": "First"},
            },
            [],
        ),
    )
    monkeypatch.setattr(
        library_cli,
        "list_skills",
        lambda repo: ({"zeta": {"description": "Uploaded"}}, []),
    )

    assert library_cli.main(["list", "--local"]) == 0
    output = capsys.readouterr().out
    assert "• alpha\n" in output
    assert "• zeta (uploaded)" in output


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


@pytest.mark.parametrize("name", ["lean", "skill-creator"])
def test_upload_rejects_bundled_skill(tmp_path, name):
    remote = _remote(tmp_path)
    source = tmp_path / "source"
    _skill(source, name)

    with pytest.raises(LibraryError, match="bundled skill cannot be uploaded"):
        upload_skill(str(remote), name, source)


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
