"""Tests for project-local agent storage cleanup."""

from kiui.agent.utils.storage import clean_storage, cleanable_entries, storage_entries


def _write_entry(root, name: str, content: str = "data"):
    path = root / ".kia" / name
    path.mkdir(parents=True)
    (path / "data").write_text(content)
    return path


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
    assert not pdf_cache.exists()
    assert not custom_cache.exists()


def test_selected_clean_only_removes_selected_entries(tmp_path):
    skills = _write_entry(tmp_path, "skills")
    pdf_cache = _write_entry(tmp_path, "pdf-cache")
    entries = {entry.name: entry for entry in storage_entries(tmp_path)}

    clean_storage(entries=[entries["skills"]])

    assert not skills.exists()
    assert pdf_cache.exists()
