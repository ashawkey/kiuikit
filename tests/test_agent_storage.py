"""Tests for project-local agent storage cleanup."""

import subprocess

from kiui.agent.utils import get_kia_dir
from kiui.agent.utils.storage import clean_storage, cleanable_entries, storage_entries


def _write_entry(root, name: str, content: str = "data"):
    path = root / ".kia" / name
    path.mkdir(parents=True)
    (path / "data").write_text(content)
    return path


def test_kia_dir_ignores_itself_and_all_contents(tmp_path):
    subprocess.run(
        ["git", "init", "--quiet"], cwd=tmp_path, check=True, capture_output=True
    )

    root = get_kia_dir(tmp_path)
    (root / "cache").mkdir()
    (root / "cache" / "data").write_text("data")

    assert (root / ".gitignore").read_text(encoding="utf-8") == "*\n"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""


def test_default_clean_removes_all_entries_except_skills(tmp_path):
    skills = _write_entry(tmp_path, "skills")
    pdf_cache = _write_entry(tmp_path, "pdf-cache")
    custom_cache = _write_entry(tmp_path, "custom-cache")

    assert {entry.name for entry in cleanable_entries(tmp_path)} == {
        "pdf-cache",
        "custom-cache",
    }

    removed = clean_storage(tmp_path)

    assert removed > 0
    assert skills.exists()
    assert (tmp_path / ".kia" / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert not pdf_cache.exists()
    assert not custom_cache.exists()


def test_selected_clean_only_removes_selected_entries(tmp_path):
    skills = _write_entry(tmp_path, "skills")
    pdf_cache = _write_entry(tmp_path, "pdf-cache")
    entries = {entry.name: entry for entry in storage_entries(tmp_path)}

    clean_storage(entries=[entries["skills"]])

    assert not skills.exists()
    assert pdf_cache.exists()
